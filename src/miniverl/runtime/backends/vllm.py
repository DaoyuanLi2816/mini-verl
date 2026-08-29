"""Managed localhost vLLM rollout backend for the measured direct-GKD path."""

from __future__ import annotations

import json
import math
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Protocol

from miniverl.config.models import RolloutEngineConfig
from miniverl.models.base import GenerationOutput
from miniverl.runtime.generation import (
    BackendCapabilities,
    BackendLifecycleState,
    BackendMetrics,
    GenerationBatch,
    GenerationRequest,
    GenerationResult,
    PolicySnapshot,
    PolicySyncResult,
    ReproducibilityClass,
    RolloutBackendKind,
    RolloutPolicyIdentity,
)

__all__ = [
    "VLLMGenerationBackend",
    "build_vllm_server_command",
    "parse_vllm_completion",
]

_QUALIFIED_VLLM_VERSION = "0.28.0"
_BACKEND_VERSION = f"vllm-{_QUALIFIED_VLLM_VERSION}-direct-gkd-v1"


def build_vllm_server_command(
    *,
    python_executable: str,
    model_path: str,
    host: str,
    port: int,
    memory_fraction: float,
    max_model_len: int,
    wsl_compatibility: bool,
) -> tuple[list[str], dict[str, str]]:
    """Build the pinned, localhost-only command without importing vLLM."""

    if host != "127.0.0.1":
        raise ValueError("the managed vLLM server may bind only to localhost")
    command = [
        python_executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        str(memory_fraction),
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--return-tokens-as-token-ids",
        "--generation-config",
        "vllm",
        "--logprobs-mode",
        "raw_logprobs",
        "--enable-lora",
        "--max-lora-rank",
        "64",
        "--max-loras",
        "1",
        "--max-cpu-loras",
        "2",
    ]
    environment = {
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
    }
    if wsl_compatibility:
        # vLLM 0.28's V2 runner requires UVA, which WSL2 does not expose, and
        # the FlashInfer sampler JIT requires a system nvcc installation.
        environment["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        environment["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    return command, environment


def _only_choice(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise RuntimeError("vLLM returned an invalid completion choice")
    return choices[0]


def parse_vllm_completion(payload: Mapping[str, Any], *, need_logprobs: bool) -> GenerationOutput:
    """Parse the raw-ID encoding used by vLLM's OpenAI completion endpoint."""

    choice = _only_choice(payload)
    logprob_payload = choice.get("logprobs")
    if not isinstance(logprob_payload, dict):
        raise RuntimeError("vLLM did not return the logprobs envelope needed for raw token ids")
    raw_tokens = logprob_payload.get("tokens")
    raw_logprobs = logprob_payload.get("token_logprobs")
    if not isinstance(raw_tokens, list) or not raw_tokens:
        raise RuntimeError("vLLM returned no raw token ids")
    token_ids: list[int] = []
    for token in raw_tokens:
        if not isinstance(token, str) or not token.startswith("token_id:"):
            raise RuntimeError("vLLM response contained a token without a raw token id")
        try:
            token_id = int(token.removeprefix("token_id:"))
        except ValueError as exc:
            raise RuntimeError("vLLM response contained an invalid raw token id") from exc
        if token_id < 0:
            raise RuntimeError("vLLM response contained a negative raw token id")
        token_ids.append(token_id)
    if not isinstance(raw_logprobs, list) or len(raw_logprobs) != len(token_ids):
        raise RuntimeError("vLLM sampled-token logprobs do not align with raw token ids")
    try:
        parsed_logprobs = [float(value) for value in raw_logprobs]
    except (TypeError, ValueError) as exc:
        raise RuntimeError("vLLM returned a non-numeric sampled-token logprob") from exc
    if not all(math.isfinite(value) for value in parsed_logprobs):
        raise RuntimeError("vLLM returned a non-finite sampled-token logprob")

    finish_reason = choice.get("finish_reason")
    raw_stop = choice.get("stop_reason")
    matched_stop: str | None = None
    if finish_reason == "length":
        stop_reason = "max_new_tokens"
    elif isinstance(raw_stop, str):
        stop_reason = "stop_sequence"
        matched_stop = raw_stop
    else:
        stop_reason = "eos"
    return GenerationOutput(
        token_ids=token_ids,
        text=str(choice.get("text", "")),
        stop_reason=stop_reason,
        matched_stop=matched_stop,
        logprobs=parsed_logprobs if need_logprobs else [],
    )


def _json_request(
    base_url: str,
    path: str,
    *,
    payload: Mapping[str, Any] | None,
    timeout: float,
) -> Any:
    data = None if payload is None else json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError(f"managed vLLM request {path} failed: {exc}") from exc
    content_type = response.headers.get("Content-Type", "")
    if "json" not in content_type:
        return body
    try:
        return json.loads(body)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"managed vLLM request {path} returned invalid JSON") from exc


def _free_local_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind((host, 0))
        return int(listener.getsockname()[1])


def _is_wsl() -> bool:
    return platform.system() == "Linux" and "microsoft" in platform.release().lower()


def _resolve_model_snapshot(model_id: str, revision: str | None) -> Path:
    direct = Path(model_id)
    if direct.is_dir():
        return direct.resolve()
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = try_to_load_from_cache(model_id, "config.json", revision=revision)
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"could not resolve cached base snapshot {model_id!r}: {exc}") from exc
    if not isinstance(cached, str):
        revision_hint = f" --revision {revision}" if revision else ""
        raise RuntimeError(
            f"base snapshot {model_id!r} is not cached; run `hf download {model_id}"
            f"{revision_hint}` before using rollout.backend=vllm"
        )
    # Hub snapshot entries are commonly symlinks into ``blobs/``. Resolve the
    # containing snapshot directory, not the config symlink target.
    snapshot = Path(cached).parent.resolve()
    if not (snapshot / "config.json").is_file():
        raise RuntimeError("resolved vLLM base snapshot has no config.json")
    return snapshot


class _Manager(Protocol):
    def start(self) -> None: ...
    def load_adapter(self, name: str, path: Path) -> None: ...
    def unload_adapter(self, name: str) -> None: ...
    def complete(self, payload: dict[str, object]) -> dict[str, object]: ...
    def close(self) -> None: ...


class _VLLMServerManager:
    def __init__(
        self,
        *,
        model_path: Path,
        config: RolloutEngineConfig,
        max_model_len: int,
        workspace: Path,
    ) -> None:
        self.model_path = model_path
        self.config = config
        self.max_model_len = max_model_len
        self.workspace = workspace
        self.port = _free_local_port(config.host)
        self.base_url = f"http://{config.host}:{self.port}"
        self.process: subprocess.Popen[bytes] | None = None
        self._log_handle: Any | None = None

    def start(self) -> None:
        if self.process is not None and self.process.poll() is None:
            return
        if platform.system() != "Linux":
            raise RuntimeError("the qualified vLLM rollout backend requires Linux or WSL2")
        try:
            installed = metadata.version("vllm")
        except metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                "rollout.backend=vllm requires `pip install miniverl[rollout-vllm]`"
            ) from exc
        if installed != _QUALIFIED_VLLM_VERSION:
            raise RuntimeError(
                f"qualified vLLM version is {_QUALIFIED_VLLM_VERSION}, found {installed}"
            )
        self.port = _free_local_port(self.config.host)
        self.base_url = f"http://{self.config.host}:{self.port}"
        command, additions = build_vllm_server_command(
            python_executable=sys.executable,
            model_path=str(self.model_path),
            host=self.config.host,
            port=self.port,
            memory_fraction=self.config.memory_fraction,
            max_model_len=self.max_model_len,
            wsl_compatibility=_is_wsl(),
        )
        environment = os.environ.copy()
        environment.update(additions)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._log_handle = (self.workspace / "server.log").open("ab")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            env=environment,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.config.startup_timeout_seconds
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                return_code = self.process.returncode
                self.close()
                raise RuntimeError(f"managed vLLM exited during startup with code {return_code}")
            try:
                _json_request(
                    self.base_url,
                    "/health",
                    payload=None,
                    timeout=min(1.0, self.config.request_timeout_seconds),
                )
                models = _json_request(
                    self.base_url,
                    "/v1/models",
                    payload=None,
                    timeout=self.config.request_timeout_seconds,
                )
                if not isinstance(models, dict) or not isinstance(models.get("data"), list):
                    raise RuntimeError("managed vLLM model inventory is invalid")
                return
            except RuntimeError:
                time.sleep(0.1)
        self.close()
        raise RuntimeError(
            f"managed vLLM did not become healthy within "
            f"{self.config.startup_timeout_seconds:g} seconds"
        )

    def load_adapter(self, name: str, path: Path) -> None:
        _json_request(
            self.base_url,
            "/v1/load_lora_adapter",
            payload={"lora_name": name, "lora_path": str(path)},
            timeout=self.config.request_timeout_seconds,
        )

    def unload_adapter(self, name: str) -> None:
        _json_request(
            self.base_url,
            "/v1/unload_lora_adapter",
            payload={"lora_name": name},
            timeout=self.config.request_timeout_seconds,
        )

    def complete(self, payload: dict[str, object]) -> dict[str, object]:
        result = _json_request(
            self.base_url,
            "/v1/completions",
            payload=payload,
            timeout=self.config.request_timeout_seconds,
        )
        if not isinstance(result, dict):
            raise RuntimeError("managed vLLM returned a non-object completion")
        return result

    def close(self) -> None:
        process = self.process
        self.process = None
        try:
            if process is not None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)  # type: ignore[attr-defined]
                if process.poll() is None:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        with suppress(ProcessLookupError):
                            os.killpg(  # type: ignore[attr-defined]
                                process.pid,
                                signal.SIGKILL,  # type: ignore[attr-defined]
                            )
                        process.wait(timeout=10)
        finally:
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None


@dataclass(frozen=True)
class _CompletedRequest:
    request: GenerationRequest
    output: GenerationOutput
    raw_request_id: str | None


class VLLMGenerationBackend:
    """Time-multiplexed vLLM 0.28 backend, qualified for direct GKD only."""

    kind = RolloutBackendKind.VLLM
    backend_version = _BACKEND_VERSION

    def __init__(
        self,
        model_backend: Any,
        *,
        engine_config: RolloutEngineConfig,
        max_model_len: int,
        workspace: Path | None = None,
        manager: _Manager | None = None,
    ) -> None:
        self.model_backend = model_backend
        self.engine_config = engine_config
        parent = workspace or Path(tempfile.gettempdir())
        parent.mkdir(parents=True, exist_ok=True)
        self.workspace = Path(tempfile.mkdtemp(prefix="miniverl-vllm-", dir=parent))
        try:
            model_path = (
                _resolve_model_snapshot(
                    str(getattr(model_backend, "model_id", "")),
                    getattr(model_backend, "model_revision", None),
                )
                if manager is None
                else Path(str(getattr(model_backend, "model_id", "model")))
            )
            self.manager: _Manager = manager or _VLLMServerManager(
                model_path=model_path,
                config=engine_config,
                max_model_len=max_model_len,
                workspace=self.workspace,
            )
        except BaseException:
            shutil.rmtree(self.workspace, ignore_errors=True)
            raise
        self._state = BackendLifecycleState.NEW
        self._active_identity: RolloutPolicyIdentity | None = None
        self._active_adapter_name: str | None = None
        self._server_adapter_name: str | None = None
        self._lifecycle: dict[str, object] = {
            "prefix_cache_enabled": False,
            "numerical_equivalence_class": "unmeasured",
        }

    @property
    def state(self) -> BackendLifecycleState:
        return self._state

    def inspect(self) -> BackendCapabilities:
        return BackendCapabilities(
            kind=self.kind,
            backend_version=self.backend_version,
            supports_greedy=True,
            supports_seeded_sampling=True,
            supports_sampled_token_logprobs=False,
            supports_text_stops=True,
            reproducibility=ReproducibilityClass.DETERMINISTIC_GREEDY,
        )

    def lifecycle_metrics(self) -> dict[str, object]:
        """Return the latest bounded sync/teardown measurements for event logs."""

        return dict(self._lifecycle)

    @staticmethod
    def _adapter_name(identity: RolloutPolicyIdentity) -> str:
        return f"miniverl-p{identity.parameter_version:08d}-{identity.adapter_tensor_digest[:12]}"

    def _materialize_adapter(self, identity: RolloutPolicyIdentity) -> Path:
        target = self.workspace / self._adapter_name(identity)
        if (
            target.is_dir()
            and (target / "adapter_config.json").is_file()
            and any(target.glob("*.safetensors"))
        ):
            return target
        if target.exists():
            shutil.rmtree(target)
        exporter = getattr(self.model_backend, "export_rollout_adapter", None)
        try:
            if callable(exporter):
                exporter(target)
            else:
                model = getattr(self.model_backend, "model", None)
                save = getattr(model, "save_pretrained", None)
                if not callable(save):
                    raise RuntimeError(
                        "vLLM policy synchronization needs a PEFT actor with save_pretrained()"
                    )
                target.mkdir(parents=True)
                save(target, safe_serialization=True)
        except BaseException:
            shutil.rmtree(target, ignore_errors=True)
            raise
        if not (target / "adapter_config.json").is_file() or not any(target.glob("*.safetensors")):
            shutil.rmtree(target, ignore_errors=True)
            raise RuntimeError("materialized vLLM policy adapter is incomplete")
        return target

    def synchronize(self, snapshot: PolicySnapshot) -> PolicySyncResult:
        if self._state is BackendLifecycleState.CLOSED:
            raise RuntimeError("generation backend is closed")
        identity = snapshot.identity
        if identity.generation_backend is not self.kind:
            raise RuntimeError("policy identity does not select the vLLM backend")
        if identity.backend_version != self.backend_version:
            raise RuntimeError("policy identity does not match the qualified vLLM backend version")
        previous = self._active_identity.digest if self._active_identity is not None else None
        name = self._adapter_name(identity)
        sync_started = time.perf_counter()
        try:
            adapter = self._materialize_adapter(identity)
            startup_started = time.perf_counter()
            self.manager.start()
            startup_seconds = time.perf_counter() - startup_started
            adapter_started = time.perf_counter()
            if self._server_adapter_name != name:
                old_name = self._server_adapter_name
                if old_name is not None:
                    self.manager.unload_adapter(old_name)
                    self._server_adapter_name = None
                self.manager.load_adapter(name, adapter)
                self._server_adapter_name = name
            adapter_sync_seconds = time.perf_counter() - adapter_started
        except BaseException:
            self._server_adapter_name = None
            self._state = BackendLifecycleState.QUIESCED
            self.manager.close()
            raise
        old_active_name = self._active_adapter_name
        self._active_identity = identity
        self._active_adapter_name = name
        self._state = BackendLifecycleState.SYNCHRONIZED
        if old_active_name is not None and old_active_name != name:
            shutil.rmtree(self.workspace / old_active_name, ignore_errors=True)
        self._lifecycle = {
            "policy_identity_digest": identity.digest,
            "policy_version": identity.parameter_version,
            "adapter_name": name,
            "startup_seconds": startup_seconds,
            "adapter_sync_seconds": adapter_sync_seconds,
            "sync_total_seconds": time.perf_counter() - sync_started,
            "prefix_cache_enabled": False,
            "numerical_equivalence_class": (
                f"bf16-external-vs-{identity.quantization}-actor-direct-gkd"
            ),
        }
        return PolicySyncResult(
            previous_policy_digest=previous,
            active_policy_digest=identity.digest,
            changed=previous != identity.digest,
            state=self._state,
        )

    def _validate_requests(self, requests: Sequence[GenerationRequest]) -> RolloutPolicyIdentity:
        if self._state is not BackendLifecycleState.SYNCHRONIZED:
            raise RuntimeError("generation backend must be synchronized before generate()")
        if not requests:
            raise ValueError("generation request batch cannot be empty")
        assert self._active_identity is not None
        seen: set[str] = set()
        for request in requests:
            if request.request_id in seen:
                raise ValueError(f"duplicate generation request id {request.request_id!r}")
            seen.add(request.request_id)
            if request.expected_policy_identity != self._active_identity:
                raise RuntimeError("generation request policy identity is stale")
            if request.need_sampled_token_logprobs:
                raise RuntimeError(
                    "vLLM sampled-token logprobs are not qualified for PG; use direct GKD"
                )
        return self._active_identity

    def _complete(self, request: GenerationRequest) -> _CompletedRequest:
        assert self._active_adapter_name is not None
        payload: dict[str, object] = {
            "model": self._active_adapter_name,
            "prompt": list(request.prompt_token_ids),
            "max_tokens": request.max_new_tokens,
            "temperature": request.sampling.temperature,
            "top_p": request.sampling.top_p,
            # miniVERL uses zero while vLLM uses -1 for disabled top-k sampling.
            "top_k": request.sampling.top_k or -1,
            "seed": request.deterministic_sample_seed,
            # vLLM encodes exact token ids in this envelope even when the
            # caller does not consume sampled-token logprobs.
            "logprobs": 1,
        }
        if request.stop_sequences:
            payload["stop"] = list(request.stop_sequences)
        raw = self.manager.complete(payload)
        output = parse_vllm_completion(raw, need_logprobs=False)
        request_id = raw.get("id")
        return _CompletedRequest(
            request=request,
            output=output,
            raw_request_id=request_id if isinstance(request_id, str) else None,
        )

    def generate(self, requests: Sequence[GenerationRequest]) -> GenerationBatch:
        identity = self._validate_requests(requests)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=min(len(requests), 8)) as executor:
            completed = list(executor.map(self._complete, requests))
        elapsed = time.perf_counter() - started
        divisor = len(completed)
        results = tuple(
            GenerationResult(
                request_id=item.request.request_id,
                group=item.request.group,
                output_token_ids=tuple(item.output.token_ids),
                decoded_text=item.output.text,
                sampled_token_logprobs=tuple(item.output.logprobs),
                stop_reason=item.output.stop_reason,
                matched_stop=item.output.matched_stop,
                policy_identity=identity,
                backend_metrics=BackendMetrics(
                    total_seconds=elapsed / divisor,
                    prompt_tokens=len(item.request.prompt_token_ids),
                    generated_tokens=len(item.output.token_ids),
                ),
                raw_backend_request_id=item.raw_request_id,
            )
            for item in completed
        )
        return GenerationBatch(
            results=results,
            policy_identity=identity,
            physical_batch_sizes=(len(requests),),
        )

    def quiesce(self) -> None:
        if self._state is not BackendLifecycleState.SYNCHRONIZED:
            raise RuntimeError("only a synchronized generation backend can quiesce")
        self._state = BackendLifecycleState.QUIESCED

    def release_generation_memory(self) -> None:
        if self._state is BackendLifecycleState.CLOSED:
            raise RuntimeError("generation backend is closed")
        started = time.perf_counter()
        self.manager.close()
        self._server_adapter_name = None
        self._state = BackendLifecycleState.QUIESCED
        self._lifecycle["teardown_seconds"] = time.perf_counter() - started

    def close(self) -> None:
        if self._state is BackendLifecycleState.CLOSED:
            return
        self.manager.close()
        self._active_identity = None
        self._active_adapter_name = None
        self._server_adapter_name = None
        shutil.rmtree(self.workspace, ignore_errors=True)
        self._state = BackendLifecycleState.CLOSED
