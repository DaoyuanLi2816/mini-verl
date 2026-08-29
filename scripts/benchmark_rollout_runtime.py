#!/usr/bin/env python3
"""Measure the preregistered pre-v0.11 Hugging Face rollout baseline.

The baseline intentionally exercises the public ``generate_batch`` contract
that existed before Rollout Runtime v2. It does not import or anticipate the
new backend implementation.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any

import jsonschema
import torch
import yaml

from miniverl import __version__
from miniverl.config.models import StudentModelConfig
from miniverl.models.hf import HFBackend
from miniverl.models.tokenizers import HFTokenizerAdapter
from miniverl.utils.runs import canonical_json, write_json

ROOT = Path(__file__).resolve().parents[1]
PREREGISTRATION = ROOT / "benchmarks/preregistration/rollout-runtime-v2.yaml"
SCHEMA = ROOT / "benchmarks/schema/rollout-runtime-v2.schema.json"
DEFAULT_OUTPUT = ROOT / "benchmarks/results/rollout-runtime-v2-hf-reference.json"
FROZEN_CALCULATOR = ROOT / "benchmarks/results/gpu-calc-hard-equal-update-v2.json"
FROZEN_CALCULATOR_SHA256 = "53fc1d4d5b7adee09618d77ad62d4086ba56b78569832d6fc7c3bcd5c2695bbc"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def _phase(
    status: str,
    *,
    values: list[float] | None = None,
    note: str = "",
) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": status}
    if values:
        payload["median_seconds"] = float(statistics.median(values))
        payload["minimum_seconds"] = float(min(values))
    if note:
        payload["note"] = note
    return payload


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _environment() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "cuda_runtime": torch.version.cuda,
        "packages": {
            name: _package_version(name)
            for name in ("torch", "transformers", "peft", "bitsandbytes", "accelerate")
        },
    }


def _rss_bytes() -> int | None:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    except (ImportError, OSError):
        return None


def _fixed_prompt(tokenizer: Any, *, length: int, index: int) -> list[int]:
    prefix = list(tokenizer.encode(f"System benchmark prompt {index}. Continue with text."))
    filler = list(tokenizer.encode(" benchmark"))
    if not filler:
        raise RuntimeError("tokenizer produced no filler tokens")
    while len(prefix) < length:
        prefix.extend(filler)
    prompt = prefix[:length]
    if tokenizer.eos_token_id is not None and prompt[-1] == tokenizer.eos_token_id:
        replacement = next(token for token in filler if token != tokenizer.eos_token_id)
        prompt[-1] = replacement
    if len(prompt) != length:
        raise AssertionError("fixed prompt construction changed the requested token length")
    return prompt


def _digest_outputs(outputs: list[Any]) -> tuple[str, str]:
    ids = [output.token_ids for output in outputs]
    probabilities = [output.logprobs for output in outputs]
    return (
        hashlib.sha256(canonical_json(ids).encode("utf-8")).hexdigest(),
        hashlib.sha256(canonical_json(probabilities).encode("utf-8")).hexdigest(),
    )


def _generate_once(
    backend: HFBackend,
    *,
    prompts: list[list[int]],
    seeds: list[int],
    response_bound: int,
    sampling: dict[str, Any],
    physical_batch_size: int,
) -> tuple[list[Any], int]:
    outputs: list[Any] = []
    physical_batches = 0
    for start in range(0, len(prompts), physical_batch_size):
        end = start + physical_batch_size
        outputs.extend(
            backend.generate_batch(
                prompts[start:end],
                max_new_tokens=response_bound,
                temperature=float(sampling["temperature"]),
                top_p=float(sampling["top_p"]),
                top_k=int(sampling["top_k"]),
                seeds=seeds[start:end],
                record_logprobs=True,
            )
        )
        physical_batches += 1
    return outputs, physical_batches


def _failed_cell(
    *,
    cell_id: str,
    prompt_length: int,
    response_bound: int,
    samples_per_prompt: int,
    sampling: str,
    status: str,
    error: str,
) -> dict[str, Any]:
    unavailable = _phase("failed", note=error)
    return {
        "cell_id": cell_id,
        "prompt_length": prompt_length,
        "response_bound": response_bound,
        "samples_per_prompt": samples_per_prompt,
        "sampling": sampling,
        "status": status,
        "counts": {
            "logical_prompts": 0,
            "generated_trajectories": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "physical_batches": 0,
        },
        "phases": {
            name: dict(unavailable)
            for name in (
                "cold_start",
                "prefill",
                "decode",
                "rollout_total",
                "policy_sync",
                "teacher_scoring",
                "actor_update",
                "full_cycle",
                "teardown",
            )
        },
        "rates": {
            "time_to_first_token_seconds": None,
            "prompt_tokens_per_second": None,
            "output_tokens_per_second": None,
        },
        "memory": {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "process_rss_bytes": _rss_bytes(),
            "teardown_residual_allocated_bytes": None,
        },
        "output_token_ids_sha256": None,
        "sampled_logprobs_sha256": None,
        "oom_downshifts": 0,
        "error": error,
    }


def _cell(
    backend: HFBackend,
    *,
    tokenizer: Any,
    prompt_length: int,
    response_bound: int,
    samples_per_prompt: int,
    sampling: dict[str, Any],
    workload: dict[str, Any],
    cold_start_seconds: float,
) -> dict[str, Any]:
    sampling_name = str(sampling["name"])
    cell_id = f"p{prompt_length}-r{response_bound}-n{samples_per_prompt}-{sampling_name}"
    logical_prompts = int(workload["logical_prompts"])
    run_seed = int(workload["run_seed"])
    base_prompts = [
        _fixed_prompt(tokenizer, length=prompt_length, index=index)
        for index in range(logical_prompts)
    ]
    prompts: list[list[int]] = []
    seeds: list[int] = []
    for prompt in base_prompts:
        prompt_digest = hashlib.sha256(canonical_json(prompt).encode("utf-8")).hexdigest()
        for sample_index in range(samples_per_prompt):
            prompts.append(prompt)
            seed_payload = f"rollout-runtime-v2-seed-v1:{run_seed}:{prompt_digest}:0:{sample_index}"
            seeds.append(int(hashlib.sha256(seed_payload.encode("ascii")).hexdigest()[:16], 16))

    physical_batch_size = min(4, len(prompts))
    try:
        for _ in range(int(workload["warmup_repetitions"])):
            _generate_once(
                backend,
                prompts=prompts,
                seeds=seeds,
                response_bound=response_bound,
                sampling=sampling,
                physical_batch_size=physical_batch_size,
            )
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        measured_seconds: list[float] = []
        output_digests: list[tuple[str, str]] = []
        final_outputs: list[Any] = []
        physical_batches = 0
        for _ in range(int(workload["measured_repetitions"])):
            started = time.perf_counter()
            final_outputs, physical_batches = _generate_once(
                backend,
                prompts=prompts,
                seeds=seeds,
                response_bound=response_bound,
                sampling=sampling,
                physical_batch_size=physical_batch_size,
            )
            torch.cuda.synchronize()
            measured_seconds.append(time.perf_counter() - started)
            output_digests.append(_digest_outputs(final_outputs))
        if len(set(output_digests)) != 1:
            raise RuntimeError("seeded reference output changed across measured repetitions")
        generated_tokens = sum(len(output.token_ids) for output in final_outputs)
        prompt_tokens = sum(len(prompt) for prompt in prompts)
        median_seconds = float(statistics.median(measured_seconds))
        output_ids_digest, logprobs_digest = output_digests[0]
        return {
            "cell_id": cell_id,
            "prompt_length": prompt_length,
            "response_bound": response_bound,
            "samples_per_prompt": samples_per_prompt,
            "sampling": sampling_name,
            "status": "completed",
            "counts": {
                "logical_prompts": logical_prompts,
                "generated_trajectories": len(final_outputs),
                "prompt_tokens": prompt_tokens,
                "generated_tokens": generated_tokens,
                "physical_batches": physical_batches,
            },
            "phases": {
                "cold_start": _phase(
                    "measured",
                    values=[cold_start_seconds],
                    note="shared actor/tokenizer load measurement repeated on each cell",
                ),
                "prefill": _phase(
                    "not_measured",
                    note="the pre-v0.11 public backend does not expose a prefill boundary",
                ),
                "decode": _phase(
                    "not_measured",
                    note="the pre-v0.11 public backend does not expose a decode boundary",
                ),
                "rollout_total": _phase("measured", values=measured_seconds),
                "policy_sync": _phase(
                    "not_applicable",
                    note="hf_reference uses the actor already loaded in the trainer process",
                ),
                "teacher_scoring": _phase(
                    "not_measured",
                    note="rollout-only baseline cell; full-cycle measurement is a separate gate",
                ),
                "actor_update": _phase(
                    "not_measured",
                    note="rollout-only baseline cell; full-cycle measurement is a separate gate",
                ),
                "full_cycle": _phase(
                    "not_measured",
                    note="rollout-only baseline cell; no phase is represented as zero",
                ),
                "teardown": _phase(
                    "not_measured",
                    note="filled after the shared backend is closed",
                ),
            },
            "rates": {
                "time_to_first_token_seconds": None,
                "prompt_tokens_per_second": prompt_tokens / median_seconds,
                "output_tokens_per_second": generated_tokens / median_seconds,
            },
            "memory": {
                "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                "process_rss_bytes": _rss_bytes(),
                "teardown_residual_allocated_bytes": None,
            },
            "output_token_ids_sha256": output_ids_digest,
            "sampled_logprobs_sha256": logprobs_digest,
            "oom_downshifts": 0,
            "error": None,
        }
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return _failed_cell(
            cell_id=cell_id,
            prompt_length=prompt_length,
            response_bound=response_bound,
            samples_per_prompt=samples_per_prompt,
            sampling=sampling_name,
            status="oom",
            error=f"{type(exc).__name__}: {exc}",
        )
    except Exception as exc:
        return _failed_cell(
            cell_id=cell_id,
            prompt_length=prompt_length,
            response_bound=response_bound,
            samples_per_prompt=samples_per_prompt,
            sampling=sampling_name,
            status="failed",
            error=f"{type(exc).__name__}: {exc}",
        )


def run(*, wheel: Path, output: Path, expected_commit: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("rollout-runtime-v2 baseline requires one CUDA GPU")
    if platform.system() != "Linux" or "microsoft" not in platform.release().lower():
        raise RuntimeError("formal rollout-runtime-v2 baseline requires WSL2/Linux")
    if _sha256(FROZEN_CALCULATOR) != FROZEN_CALCULATOR_SHA256:
        raise RuntimeError("frozen calculator benchmark hash changed")
    commit = _git("rev-parse", "HEAD")
    if commit != expected_commit:
        raise RuntimeError(f"expected source commit {expected_commit}, found {commit}")
    if _git("status", "--porcelain"):
        raise RuntimeError("formal baseline requires a clean worktree")
    if not wheel.is_file():
        raise FileNotFoundError(wheel)

    prereg = yaml.safe_load(PREREGISTRATION.read_text(encoding="utf-8"))
    if prereg["status"] != "preregistered_before_baseline_measurement":
        raise RuntimeError("rollout runtime preregistration is not in its frozen initial state")
    actor = prereg["models"]["actor"]
    spec = StudentModelConfig(
        model_id=actor["id"],
        revision=actor["revision"],
        tokenizer_revision=actor["tokenizer_revision"],
        dtype=prereg["models"]["dtype"],
        quantization=prereg["models"]["quantization"],
        attn_implementation=prereg["models"]["attention_implementation"],
        gradient_checkpointing=False,
    )

    started = time.perf_counter()
    tokenizer = HFTokenizerAdapter.load(
        spec.tokenizer_id or spec.model_id,
        revision=spec.tokenizer_revision or spec.revision,
        trust_remote_code=spec.trust_remote_code,
        local_files_only=False,
    )
    backend = HFBackend.load(
        spec,
        device="cuda",
        tokenizer=tokenizer,
        trainable=False,
        local_files_only=False,
    )
    torch.cuda.synchronize()
    cold_start_seconds = time.perf_counter() - started
    workload = prereg["workload"]
    cells: list[dict[str, Any]] = []
    for prompt_length in workload["prompt_lengths"]:
        for response_bound in workload["response_bounds"]:
            for samples_per_prompt in workload["samples_per_prompt"]:
                for sampling in workload["sampling"]:
                    cells.append(
                        _cell(
                            backend,
                            tokenizer=tokenizer,
                            prompt_length=int(prompt_length),
                            response_bound=int(response_bound),
                            samples_per_prompt=int(samples_per_prompt),
                            sampling=sampling,
                            workload=workload,
                            cold_start_seconds=cold_start_seconds,
                        )
                    )

    teardown_started = time.perf_counter()
    backend.release()
    del backend
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    teardown_seconds = time.perf_counter() - teardown_started
    residual = int(torch.cuda.memory_allocated())
    for cell in cells:
        cell["phases"]["teardown"] = _phase(
            "measured",
            values=[teardown_seconds],
            note="shared backend teardown measurement repeated on each cell",
        )
        cell["memory"]["teardown_residual_allocated_bytes"] = residual

    failures = [f"{cell['cell_id']}: {cell['error']}" for cell in cells if cell["error"]]
    result = {
        "schema_version": 1,
        "name": "rollout-runtime-v2-hf-reference",
        "measurement_status": "completed_with_failures" if failures else "measured_baseline",
        "source": {
            "commit": commit,
            "dirty": False,
            "miniverl_version": __version__,
            "wheel_sha256": _sha256(wheel),
        },
        "preregistration_sha256": _sha256(PREREGISTRATION),
        "workload_manifest_sha256": hashlib.sha256(
            canonical_json(workload).encode("utf-8")
        ).hexdigest(),
        "frozen_calculator_sha256": FROZEN_CALCULATOR_SHA256,
        "environment": _environment(),
        "models": {
            "actor": actor,
            "teacher": prereg["models"]["teacher"],
            "dtype": prereg["models"]["dtype"],
            "quantization": prereg["models"]["quantization"],
        },
        "backend": {
            "name": "hf_reference",
            "version": str(_package_version("transformers")),
            "reproducibility_class": "same_process_seeded_reference",
        },
        "cells": cells,
        "failures": failures,
    }
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    jsonschema.validate(result, schema)
    write_json(output, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(
        wheel=args.wheel.resolve(),
        output=args.output.resolve(),
        expected_commit=args.expected_commit,
    )
    print(json.dumps({"status": result["measurement_status"], "cells": len(result["cells"])}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
