from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import pytest

from miniverl.config.models import RolloutEngineConfig
from miniverl.runtime.generation import (
    BackendLifecycleState,
    GenerationRequest,
    PolicySnapshot,
    RolloutBackendKind,
    RolloutGroupIdentity,
    RolloutPolicyIdentity,
    SamplingParameters,
)


def _identity(version: int) -> RolloutPolicyIdentity:
    return RolloutPolicyIdentity(
        parameter_version=version,
        base_model_id="org/model",
        base_model_revision="revision",
        tokenizer_structural_identity="1" * 64,
        student_adapter_manifest_digest="2" * 64,
        adapter_tensor_digest=f"{version:x}" * 64,
        quantization="nf4",
        dtype="bfloat16",
        generation_backend=RolloutBackendKind.VLLM,
        backend_version="vllm-0.28.0-direct-gkd-v1",
        profile_identity="4" * 64,
        execution_plan_digest="5" * 64,
    )


def _request(identity: RolloutPolicyIdentity, index: int = 0) -> GenerationRequest:
    return GenerationRequest(
        request_id=f"request-{index}",
        group=RolloutGroupIdentity(
            prompt_group_id="group",
            prompt_digest="a" * 64,
            sample_index=index,
            samples_per_prompt=2,
        ),
        deterministic_sample_seed=101 + index,
        prompt_token_ids=(7, 8, 9),
        max_new_tokens=2,
        sampling=SamplingParameters(temperature=0.8, top_p=0.9, top_k=8),
        need_sampled_token_logprobs=False,
        expected_policy_identity=identity,
    )


def test_vllm_launch_is_local_managed_and_disables_persistent_prefix_cache() -> None:
    from miniverl.runtime.backends.vllm import build_vllm_server_command

    command, environment = build_vllm_server_command(
        python_executable="/venv/bin/python",
        model_path="/models/snapshot",
        host="127.0.0.1",
        port=30123,
        memory_fraction=0.7,
        max_model_len=2048,
        wsl_compatibility=True,
    )

    assert command[:3] == ["/venv/bin/python", "-m", "vllm.entrypoints.openai.api_server"]
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert "--no-enable-prefix-caching" in command
    assert "--enable-lora" in command
    assert environment["VLLM_ALLOW_RUNTIME_LORA_UPDATING"] == "True"
    assert environment["VLLM_USE_V2_MODEL_RUNNER"] == "0"
    assert environment["VLLM_USE_FLASHINFER_SAMPLER"] == "0"


def test_vllm_snapshot_resolution_preserves_hub_snapshot_symlink_parent(
    tmp_path: Path,
) -> None:
    from miniverl.runtime.backends.vllm import _resolve_model_snapshot

    snapshot = tmp_path / "models--org--model" / "snapshots" / ("a" * 40)
    blobs = tmp_path / "models--org--model" / "blobs"
    snapshot.mkdir(parents=True)
    blobs.mkdir(parents=True)
    config_blob = blobs / "config-digest"
    config_blob.write_text("{}", encoding="utf-8")
    config = snapshot / "config.json"
    try:
        config.symlink_to(config_blob)
    except OSError:
        pytest.skip("symlinks are unavailable")

    hub = ModuleType("huggingface_hub")
    hub.try_to_load_from_cache = lambda *_args, **_kwargs: str(config)  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"huggingface_hub": hub}):
        assert _resolve_model_snapshot("org/model", "a" * 40) == snapshot.resolve()


def test_vllm_completion_parser_requires_raw_ids_and_aligned_logprobs() -> None:
    from miniverl.runtime.backends.vllm import parse_vllm_completion

    payload = {
        "id": "cmpl-1",
        "choices": [
            {
                "finish_reason": "length",
                "stop_reason": None,
                "text": "ignored",
                "logprobs": {
                    "tokens": ["token_id:17", "token_id:19"],
                    "token_logprobs": [-0.25, -0.5],
                },
            }
        ],
    }
    parsed = parse_vllm_completion(payload, need_logprobs=True)
    assert parsed.token_ids == [17, 19]
    assert parsed.logprobs == [-0.25, -0.5]
    assert parsed.stop_reason == "max_new_tokens"

    malformed = json.loads(json.dumps(payload))
    malformed["choices"][0]["logprobs"]["tokens"][0] = "plain text"
    with pytest.raises(RuntimeError, match="raw token id"):
        parse_vllm_completion(malformed, need_logprobs=True)

    unaligned = json.loads(json.dumps(payload))
    unaligned["choices"][0]["logprobs"]["token_logprobs"] = [-0.25]
    with pytest.raises(RuntimeError, match="align"):
        parse_vllm_completion(unaligned, need_logprobs=True)

    nonfinite = json.loads(json.dumps(payload))
    nonfinite["choices"][0]["logprobs"]["token_logprobs"][0] = math.inf
    with pytest.raises(RuntimeError, match="non-finite"):
        parse_vllm_completion(nonfinite, need_logprobs=True)


class _FakeManager:
    def __init__(self) -> None:
        self.started = 0
        self.closed = 0
        self.loaded: list[tuple[str, Path]] = []
        self.unloaded: list[str] = []
        self.requests: list[dict[str, object]] = []

    def start(self) -> None:
        self.started += 1

    def load_adapter(self, name: str, path: Path) -> None:
        self.loaded.append((name, path))

    def unload_adapter(self, name: str) -> None:
        self.unloaded.append(name)

    def complete(self, payload: dict[str, object]) -> dict[str, object]:
        self.requests.append(payload)
        return {
            "id": f"cmpl-{len(self.requests)}",
            "choices": [
                {
                    "finish_reason": "length",
                    "stop_reason": None,
                    "text": "decoded",
                    "logprobs": {
                        "tokens": ["token_id:17", "token_id:19"],
                        "token_logprobs": [-0.25, -0.5],
                    },
                }
            ],
        }

    def close(self) -> None:
        self.closed += 1


class _FailingLoadManager(_FakeManager):
    def load_adapter(self, name: str, path: Path) -> None:
        if self.loaded:
            raise RuntimeError("injected adapter load failure")
        super().load_adapter(name, path)


class _FakeModelBackend:
    model_id = "org/model"
    model_revision = "revision"
    capabilities = SimpleNamespace(device="cuda", name="org/model")

    def export_rollout_adapter(self, target: Path) -> None:
        target.mkdir(parents=True)
        (target / "adapter_config.json").write_text("{}", encoding="utf-8")
        (target / "adapter_model.safetensors").write_bytes(b"weights")


def test_vllm_backend_uses_unique_content_bound_adapter_names_and_tears_down(
    tmp_path: Path,
) -> None:
    from miniverl.runtime.backends.vllm import VLLMGenerationBackend

    manager = _FakeManager()
    backend = VLLMGenerationBackend(
        _FakeModelBackend(),
        engine_config=RolloutEngineConfig(),
        max_model_len=2048,
        workspace=tmp_path,
        manager=manager,
    )
    assert backend.inspect().supports_sampled_token_logprobs is False
    first = _identity(1)
    second = _identity(2)

    backend.synchronize(PolicySnapshot(first))
    first_lifecycle = backend.lifecycle_metrics()
    first_name = manager.loaded[-1][0]
    assert "p00000001" in first_name
    assert first.adapter_tensor_digest[:12] in first_name
    generated = backend.generate([_request(first, 0), _request(first, 1)])
    assert [row.output_token_ids for row in generated.results] == [(17, 19), (17, 19)]
    assert all(not row.sampled_token_logprobs for row in generated.results)

    backend.synchronize(PolicySnapshot(second))
    second_name = manager.loaded[-1][0]
    assert second_name != first_name
    assert manager.unloaded == [first_name]
    lifecycle = backend.lifecycle_metrics()
    assert lifecycle["policy_identity_digest"] == second.digest
    assert lifecycle["adapter_name"] == second_name
    assert lifecycle["prefix_cache_enabled"] is False
    assert lifecycle["numerical_equivalence_class"] == "bf16-external-vs-nf4-actor-direct-gkd"
    assert lifecycle["startup_seconds"] == first_lifecycle["startup_seconds"]
    assert lifecycle["initial_startup_seconds"] == first_lifecycle["startup_seconds"]
    assert lifecycle["latest_start_check_seconds"] >= 0.0

    backend.quiesce()
    backend.release_generation_memory()
    assert manager.closed == 1
    assert backend.state is BackendLifecycleState.QUIESCED
    assert backend.lifecycle_metrics()["teardown_seconds"] >= 0.0
    backend.close()
    assert backend.state is BackendLifecycleState.CLOSED


def test_vllm_backend_translates_disabled_top_k_to_engine_sentinel(tmp_path: Path) -> None:
    from miniverl.runtime.backends.vllm import VLLMGenerationBackend

    manager = _FakeManager()
    backend = VLLMGenerationBackend(
        _FakeModelBackend(),
        engine_config=RolloutEngineConfig(),
        max_model_len=2048,
        workspace=tmp_path,
        manager=manager,
    )
    identity = _identity(1)
    request = _request(identity)
    request = GenerationRequest(
        **{
            **request.__dict__,
            "sampling": SamplingParameters(temperature=0.8, top_p=0.9, top_k=0),
        }
    )

    backend.synchronize(PolicySnapshot(identity))
    backend.generate([request])

    assert manager.requests[-1]["top_k"] == -1
    backend.close()


def test_vllm_constructor_removes_workspace_when_snapshot_resolution_fails(
    tmp_path: Path,
) -> None:
    from miniverl.runtime.backends.vllm import VLLMGenerationBackend

    before = set(tmp_path.iterdir())
    with (
        patch(
            "miniverl.runtime.backends.vllm._resolve_model_snapshot",
            side_effect=RuntimeError("injected snapshot failure"),
        ),
        pytest.raises(RuntimeError, match="injected snapshot failure"),
    ):
        VLLMGenerationBackend(
            _FakeModelBackend(),
            engine_config=RolloutEngineConfig(),
            max_model_len=2048,
            workspace=tmp_path,
        )

    assert set(tmp_path.iterdir()) == before


def test_vllm_adapter_export_failure_removes_partial_materialization(tmp_path: Path) -> None:
    from miniverl.runtime.backends.vllm import VLLMGenerationBackend

    class BrokenBackend(_FakeModelBackend):
        def export_rollout_adapter(self, target: Path) -> None:
            target.mkdir(parents=True)
            (target / "adapter_config.json").write_text("{}", encoding="utf-8")
            raise RuntimeError("injected export failure")

    manager = _FakeManager()
    backend = VLLMGenerationBackend(
        BrokenBackend(),
        engine_config=RolloutEngineConfig(),
        max_model_len=2048,
        workspace=tmp_path,
        manager=manager,
    )
    with pytest.raises(RuntimeError, match="injected export failure"):
        backend.synchronize(PolicySnapshot(_identity(1)))

    assert manager.started == 0
    assert not list(backend.workspace.glob("miniverl-p*"))
    backend.close()


def test_vllm_failed_policy_refresh_cannot_generate_with_previous_identity(
    tmp_path: Path,
) -> None:
    from miniverl.runtime.backends.vllm import VLLMGenerationBackend

    manager = _FailingLoadManager()
    backend = VLLMGenerationBackend(
        _FakeModelBackend(),
        engine_config=RolloutEngineConfig(),
        max_model_len=2048,
        workspace=tmp_path,
        manager=manager,
    )
    first = _identity(1)
    backend.synchronize(PolicySnapshot(first))

    with pytest.raises(RuntimeError, match="injected adapter load failure"):
        backend.synchronize(PolicySnapshot(_identity(2)))
    assert backend.state is BackendLifecycleState.QUIESCED
    with pytest.raises(RuntimeError, match="synchronized"):
        backend.generate([_request(first)])
    backend.close()
