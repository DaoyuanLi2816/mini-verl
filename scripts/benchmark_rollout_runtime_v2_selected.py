#!/usr/bin/env python3
"""Measure the selected Rollout Runtime v2 backends on the frozen token workload.

This is a rollout-only qualification. It runs one backend per process so CUDA
ownership and teardown remain observable. The output is raw machine evidence;
``publish_rollout_runtime_v2_evidence.py`` combines completed runs without
changing their measured fields.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import socket
import statistics
import subprocess
import threading
import time
from collections.abc import Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import torch
import yaml

from miniverl import __version__
from miniverl.config.models import RolloutEngineConfig, StudentModelConfig
from miniverl.models.hf import HFBackend
from miniverl.models.tokenizers import HFTokenizerAdapter
from miniverl.runtime.backends.hf_cached import HFCachedGenerationBackend
from miniverl.runtime.backends.vllm import VLLMGenerationBackend, parse_vllm_completion
from miniverl.runtime.generation import (
    GenerationRequest,
    PolicySnapshot,
    RolloutBackendKind,
    RolloutGroupIdentity,
    SamplingParameters,
    derive_sample_seed,
)
from miniverl.runtime.policy_sync import build_rollout_policy_identity
from miniverl.utils.runs import canonical_json, write_json

ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "benchmarks/preregistration/rollout-runtime-v2.yaml"
FROZEN_CALCULATOR = ROOT / "benchmarks/results/gpu-calc-hard-equal-update-v2.json"
FROZEN_CALCULATOR_SHA256 = "53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc"
PROFILE_IDENTITY = hashlib.sha256(b"rollout-runtime-v2-selected-backends-v1").hexdigest()
EXECUTION_PLAN_DIGEST = hashlib.sha256(b"rollout-runtime-v2-measurement-plan-v1").hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _gpu_used_mib() -> int:
    raw = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    return int(raw.splitlines()[0].strip())


def _process_max_rss_bytes() -> int | None:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF)  # type: ignore[attr-defined]
        return int(usage.ru_maxrss * 1024)
    except (ImportError, OSError):
        return None


class _GpuSampler:
    def __init__(self) -> None:
        self.peak_mib = _gpu_used_mib()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.05):
            try:
                self.peak_mib = max(self.peak_mib, _gpu_used_mib())
            except (OSError, subprocess.SubprocessError, ValueError):
                pass

    def __enter__(self) -> _GpuSampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=2)
        self.peak_mib = max(self.peak_mib, _gpu_used_mib())


def _environment() -> dict[str, Any]:
    driver = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).splitlines()[0]
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "driver_version": driver.strip(),
        "cuda_runtime": torch.version.cuda,
        "packages": {
            name: _package_version(name)
            for name in (
                "torch",
                "transformers",
                "peft",
                "bitsandbytes",
                "accelerate",
                "vllm",
            )
        },
    }


def _fixed_prompt(tokenizer: Any, *, length: int, index: int) -> tuple[int, ...]:
    prefix = list(tokenizer.encode(f"System benchmark prompt {index}. Continue with text."))
    filler = list(tokenizer.encode(" benchmark"))
    if not filler:
        raise RuntimeError("tokenizer produced no filler tokens")
    while len(prefix) < length:
        prefix.extend(filler)
    prompt = prefix[:length]
    if tokenizer.eos_token_id is not None and prompt[-1] == tokenizer.eos_token_id:
        prompt[-1] = next(token for token in filler if token != tokenizer.eos_token_id)
    return tuple(prompt)


def _requests(
    tokenizer: Any,
    *,
    identity: Any,
    prompt_length: int,
    response_bound: int,
    samples_per_prompt: int,
    logical_prompts: int,
    run_seed: int,
) -> list[GenerationRequest]:
    requests: list[GenerationRequest] = []
    for prompt_index in range(logical_prompts):
        prompt = _fixed_prompt(tokenizer, length=prompt_length, index=prompt_index)
        prompt_digest = hashlib.sha256(canonical_json(prompt).encode("utf-8")).hexdigest()
        group_id = f"benchmark-p{prompt_index:02d}-{prompt_digest[:12]}"
        for sample_index in range(samples_per_prompt):
            group = RolloutGroupIdentity(
                prompt_group_id=group_id,
                prompt_digest=prompt_digest,
                sample_index=sample_index,
                samples_per_prompt=samples_per_prompt,
            )
            requests.append(
                GenerationRequest(
                    request_id=f"{group_id}-s{sample_index}",
                    group=group,
                    deterministic_sample_seed=derive_sample_seed(
                        run_seed=run_seed,
                        prompt_digest=prompt_digest,
                        policy_version=identity.parameter_version,
                        sample_index=sample_index,
                    ),
                    prompt_token_ids=prompt,
                    max_new_tokens=response_bound,
                    sampling=SamplingParameters(temperature=0.0, top_p=1.0, top_k=0),
                    need_sampled_token_logprobs=False,
                    expected_policy_identity=identity,
                )
            )
    return requests


def _generate_partitioned(backend: Any, requests: Sequence[GenerationRequest]) -> list[Any]:
    results: list[Any] = []
    for start in range(0, len(requests), 4):
        results.extend(backend.generate(requests[start : start + 4]).results)
    return results


def _digest_results(results: Sequence[Any]) -> str:
    payload = [list(result.output_token_ids) for result in results]
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _cell(
    backend: Any,
    tokenizer: Any,
    *,
    identity: Any,
    prompt_length: int,
    response_bound: int,
    samples_per_prompt: int,
    logical_prompts: int,
    run_seed: int,
    warmups: int,
    repetitions: int,
) -> dict[str, Any]:
    requests = _requests(
        tokenizer,
        identity=identity,
        prompt_length=prompt_length,
        response_bound=response_bound,
        samples_per_prompt=samples_per_prompt,
        logical_prompts=logical_prompts,
        run_seed=run_seed,
    )
    for _ in range(warmups):
        _generate_partitioned(backend, requests)
    torch.cuda.synchronize()
    durations: list[float] = []
    digests: list[str] = []
    final: list[Any] = []
    with _GpuSampler() as memory:
        for _ in range(repetitions):
            started = time.perf_counter()
            final = _generate_partitioned(backend, requests)
            torch.cuda.synchronize()
            durations.append(time.perf_counter() - started)
            digests.append(_digest_results(final))
    if len(set(digests)) != 1:
        raise RuntimeError("backend output changed across greedy repetitions")
    generated_tokens = sum(len(result.output_token_ids) for result in final)
    prompt_tokens = sum(len(request.prompt_token_ids) for request in requests)
    median = float(statistics.median(durations))
    return {
        "cell_id": f"p{prompt_length}-r{response_bound}-n{samples_per_prompt}-greedy",
        "prompt_length": prompt_length,
        "response_bound": response_bound,
        "samples_per_prompt": samples_per_prompt,
        "sampling": "greedy",
        "logical_prompts": logical_prompts,
        "generated_trajectories": len(final),
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "physical_concurrency": 4,
        "warmup_repetitions": warmups,
        "measured_repetitions": repetitions,
        "rollout_seconds": durations,
        "median_rollout_seconds": median,
        "output_tokens_per_second": generated_tokens / median,
        "prompt_tokens_per_second": prompt_tokens / median,
        "output_token_ids_sha256": digests[0],
        "peak_total_gpu_memory_mib": memory.peak_mib,
    }


def _conformance_probe(
    actor: HFBackend, external: VLLMGenerationBackend, tokenizer: Any
) -> dict[str, Any]:
    identity = external._active_identity
    adapter_name = external._active_adapter_name
    if identity is None or adapter_name is None:
        raise RuntimeError("external backend is not synchronized for conformance")
    prompt = _fixed_prompt(tokenizer, length=128, index=0)
    local = actor.generate(
        list(prompt),
        max_new_tokens=32,
        temperature=0.0,
        top_p=1.0,
        top_k=0,
        seed=20260829,
        record_logprobs=True,
    )
    raw = external.manager.complete(
        {
            "model": adapter_name,
            "prompt": list(prompt),
            "max_tokens": 32,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": 20260829,
            "logprobs": 1,
        }
    )
    remote = parse_vllm_completion(raw, need_logprobs=True)
    compared = min(len(local.token_ids), len(remote.token_ids))
    agreement = sum(
        left == right
        for left, right in zip(local.token_ids[:compared], remote.token_ids[:compared], strict=True)
    )
    differences = [
        abs(left - right)
        for left, right in zip(local.logprobs[:compared], remote.logprobs[:compared], strict=True)
    ]
    ordered = sorted(differences)
    p99 = ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))] if ordered else None
    return {
        "prompt_token_ids_sha256": hashlib.sha256(
            canonical_json(prompt).encode("utf-8")
        ).hexdigest(),
        "compared_output_tokens": compared,
        "token_agreement_count": agreement,
        "token_agreement_fraction": agreement / compared if compared else None,
        "local_output_token_ids_sha256": hashlib.sha256(
            canonical_json(local.token_ids).encode("utf-8")
        ).hexdigest(),
        "engine_output_token_ids_sha256": hashlib.sha256(
            canonical_json(remote.token_ids).encode("utf-8")
        ).hexdigest(),
        "sampled_logprob_max_abs_difference": max(differences) if differences else None,
        "sampled_logprob_mean_abs_difference": (
            statistics.fmean(differences) if differences else None
        ),
        "sampled_logprob_p99_abs_difference": p99,
        "pg_nf4_threshold": 0.01,
        "pg_threshold_passed": bool(differences) and max(differences) <= 0.01,
    }


def _refresh_probe(
    actor: HFBackend,
    backend: VLLMGenerationBackend,
    tokenizer: Any,
    *,
    run_seed: int,
) -> dict[str, Any]:
    trainable = [parameter for parameter in actor.model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("refresh probe requires a trainable adapter")
    rows: list[dict[str, Any]] = []
    for version in range(1, 9):
        with torch.no_grad():
            trainable[0].view(-1)[0].add_(1e-4)
        identity = build_rollout_policy_identity(
            backend=actor,
            parameter_version=version,
            generation_backend=RolloutBackendKind.VLLM,
            backend_version=backend.backend_version,
            profile_identity=PROFILE_IDENTITY,
            execution_plan_digest=EXECUTION_PLAN_DIGEST,
        )
        started = time.perf_counter()
        synced = backend.synchronize(PolicySnapshot(identity))
        sync_seconds = time.perf_counter() - started
        request = _requests(
            tokenizer,
            identity=identity,
            prompt_length=128,
            response_bound=8,
            samples_per_prompt=1,
            logical_prompts=1,
            run_seed=run_seed,
        )
        result = backend.generate(request).results[0]
        rows.append(
            {
                "policy_version": version,
                "policy_identity_digest": identity.digest,
                "adapter_tensor_digest": identity.adapter_tensor_digest,
                "sync_confirmed": synced.active_policy_digest == identity.digest,
                "sync_seconds": sync_seconds,
                "output_token_ids_sha256": hashlib.sha256(
                    canonical_json(result.output_token_ids).encode("utf-8")
                ).hexdigest(),
                "total_gpu_memory_mib": _gpu_used_mib(),
            }
        )
    memory = [row["total_gpu_memory_mib"] for row in rows]
    return {
        "cycles": rows,
        "all_policy_identities_unique": len({row["policy_identity_digest"] for row in rows})
        == len(rows),
        "all_syncs_confirmed": all(row["sync_confirmed"] for row in rows),
        "strictly_monotonic_memory_growth": all(right > left for left, right in pairwise(memory)),
    }


def run(
    *,
    backend_name: str,
    wheel: Path,
    expected_commit: str,
    output: Path,
    repetitions: int,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("selected-backend benchmark requires CUDA")
    if platform.system() != "Linux" or "microsoft" not in platform.release().lower():
        raise RuntimeError("formal selected-backend benchmark requires WSL2/Linux")
    if _sha256(FROZEN_CALCULATOR) != FROZEN_CALCULATOR_SHA256:
        raise RuntimeError("frozen calculator benchmark hash changed")
    if _git("rev-parse", "HEAD") != expected_commit:
        raise RuntimeError("source commit does not match --expected-commit")
    if _git("status", "--porcelain"):
        raise RuntimeError("formal selected-backend benchmark requires a clean worktree")
    if not wheel.is_file():
        raise FileNotFoundError(wheel)

    prereg = yaml.safe_load(PREREGISTRATION.read_text(encoding="utf-8"))
    actor_spec = prereg["models"]["actor"]
    spec = StudentModelConfig(
        model_id=actor_spec["id"],
        revision=actor_spec["revision"],
        tokenizer_revision=actor_spec["tokenizer_revision"],
        dtype=prereg["models"]["dtype"],
        quantization=prereg["models"]["quantization"],
        attn_implementation=prereg["models"]["attention_implementation"],
        gradient_checkpointing=False,
    )
    baseline_gpu_mib = _gpu_used_mib()
    load_started = time.perf_counter()
    tokenizer = HFTokenizerAdapter.load(
        spec.tokenizer_id or spec.model_id,
        revision=spec.tokenizer_revision or spec.revision,
        trust_remote_code=spec.trust_remote_code,
        local_files_only=True,
    )
    actor = HFBackend.load(
        spec,
        device="cuda",
        tokenizer=tokenizer,
        trainable=True,
        local_files_only=True,
    )
    torch.cuda.synchronize()
    actor_load_seconds = time.perf_counter() - load_started
    if backend_name == "hf_cached":
        backend: Any = HFCachedGenerationBackend(actor)
        kind = RolloutBackendKind.HF_CACHED
    elif backend_name == "vllm":
        backend = VLLMGenerationBackend(
            actor,
            engine_config=RolloutEngineConfig(
                managed=True,
                host="127.0.0.1",
                startup_timeout_seconds=120,
                request_timeout_seconds=120,
                memory_fraction=0.5,
            ),
            max_model_len=1024,
        )
        kind = RolloutBackendKind.VLLM
    else:
        raise ValueError(f"unsupported backend {backend_name!r}")

    identity = build_rollout_policy_identity(
        backend=actor,
        parameter_version=0,
        generation_backend=kind,
        backend_version=backend.backend_version,
        profile_identity=PROFILE_IDENTITY,
        execution_plan_digest=EXECUTION_PLAN_DIGEST,
    )
    sync_started = time.perf_counter()
    sync = backend.synchronize(PolicySnapshot(identity))
    initial_sync_seconds = time.perf_counter() - sync_started
    cells = [
        _cell(
            backend,
            tokenizer,
            identity=identity,
            prompt_length=128,
            response_bound=response_bound,
            samples_per_prompt=samples_per_prompt,
            logical_prompts=4,
            run_seed=int(prereg["workload"]["run_seed"]),
            warmups=1,
            repetitions=repetitions,
        )
        for response_bound in (64, 256, 512)
        for samples_per_prompt in (1, 4)
    ]
    conformance = _conformance_probe(actor, backend, tokenizer) if backend_name == "vllm" else None
    refresh = (
        _refresh_probe(
            actor,
            backend,
            tokenizer,
            run_seed=int(prereg["workload"]["run_seed"]),
        )
        if backend_name == "vllm"
        else None
    )
    lifecycle = backend.lifecycle_metrics() if backend_name == "vllm" else {}
    port = getattr(getattr(backend, "manager", None), "port", None)
    teardown_started = time.perf_counter()
    backend.close()
    actor.release()
    del actor
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    teardown_seconds = time.perf_counter() - teardown_started
    time.sleep(1)
    residual_gpu_mib = _gpu_used_mib()
    port_closed = None
    if isinstance(port, int):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            port_closed = probe.connect_ex(("127.0.0.1", port)) != 0

    payload = {
        "schema_version": 1,
        "name": f"rollout-runtime-v2-{backend_name}-rtx4080-raw",
        "measurement_status": "completed",
        "source": {
            "commit": expected_commit,
            "dirty": False,
            "miniverl_version": __version__,
            "wheel_sha256": _sha256(wheel),
        },
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "frozen_calculator_sha256": FROZEN_CALCULATOR_SHA256,
        "environment": _environment(),
        "models": prereg["models"],
        "backend": {
            "name": backend_name,
            "version": backend.backend_version,
            "capabilities": backend.inspect().to_dict(),
        },
        "policy_identity": {
            "parameter_version": identity.parameter_version,
            "adapter_tensor_digest": identity.adapter_tensor_digest,
            "identity_digest": identity.digest,
        },
        "workload": {
            "manifest_version": prereg["workload"]["manifest_version"],
            "run_seed": prereg["workload"]["run_seed"],
            "logical_prompts": 4,
            "prompt_length": 128,
            "response_bounds": [64, 256, 512],
            "samples_per_prompt": [1, 4],
            "sampling": "greedy",
            "physical_concurrency": 4,
        },
        "timing": {
            "actor_load_seconds": actor_load_seconds,
            "initial_sync_seconds": initial_sync_seconds,
            "engine_lifecycle": lifecycle,
            "teardown_seconds": teardown_seconds,
        },
        "memory": {
            "baseline_total_gpu_memory_mib": baseline_gpu_mib,
            "peak_total_gpu_memory_mib": max(cell["peak_total_gpu_memory_mib"] for cell in cells),
            "residual_total_gpu_memory_mib": residual_gpu_mib,
            "process_max_rss_bytes": _process_max_rss_bytes(),
        },
        "cells": cells,
        "conformance": conformance,
        "refresh_probe": refresh,
        "teardown": {
            "port": port,
            "port_closed": port_closed,
            "backend_state": backend.state.value,
        },
        "sync_confirmed": sync.active_policy_digest == identity.digest,
    }
    write_json(output, payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("hf_cached", "vllm"), required=True)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    payload = run(
        backend_name=args.backend,
        wheel=args.wheel.resolve(),
        expected_commit=args.expected_commit,
        output=args.output.resolve(),
        repetitions=args.repetitions,
    )
    print(
        json.dumps(
            {
                "status": payload["measurement_status"],
                "backend": payload["backend"]["name"],
                "cells": len(payload["cells"]),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
