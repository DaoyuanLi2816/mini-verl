"""Bounded launchability smoke under the exact pinned verl source."""

from __future__ import annotations

import importlib.metadata
import json
from pathlib import Path

import pytest

pytestmark = [pytest.mark.torch, pytest.mark.network, pytest.mark.verl_conformance]

_MODEL_ID = "trl-internal-testing/tiny-Qwen3ForCausalLM"
_REVISION = "52b2e48b0004586eff92c403efa5ce5547c43a45"


def test_materialized_bundle_reaches_bounded_launchable_state(tmp_path: Path) -> None:
    try:
        importlib.metadata.distribution("verl")
    except importlib.metadata.PackageNotFoundError:
        pytest.skip("official verl v0.8.0 is not installed")
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from miniverl.bridge.contract import VERL_TAG
    from miniverl.bridge.doctor import inspect_bridge_bundle
    from miniverl.bridge.export import export_verl_bundle
    from miniverl.bridge.materialize import materialize_verl_bundle
    from tests.unit.test_verl_opd_export import _opd_run

    run, _, _ = _opd_run(tmp_path)
    source_path = run / "verl-source-config.json"
    source = json.loads(source_path.read_text(encoding="utf-8"))
    source["actor_rollout_ref"]["model"]["path"] = _MODEL_ID
    source["distillation"]["teacher_models"]["teacher_model"]["model_path"] = _MODEL_ID
    source["miniverl"]["student_revision"] = _REVISION
    source["miniverl"]["teacher_revision"] = _REVISION
    source_path.write_text(json.dumps(source), encoding="utf-8")
    compatibility_path = run / "verl-compatibility-report.json"
    compatibility = json.loads(compatibility_path.read_text(encoding="utf-8"))
    compatibility["source"] = source
    compatibility_path.write_text(json.dumps(compatibility), encoding="utf-8")

    adapter = run / "final-peft-adapter"
    model = AutoModelForCausalLM.from_pretrained(
        _MODEL_ID, revision=_REVISION, trust_remote_code=False
    )
    peft_model = get_peft_model(
        model,
        LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"]),
    )
    peft_model.save_pretrained(adapter, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(
        _MODEL_ID, revision=_REVISION, trust_remote_code=False
    )
    tokenizer.save_pretrained(adapter)
    adapter_config = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    adapter_config["base_model_name_or_path"] = _MODEL_ID
    adapter_config["revision"] = _REVISION
    (adapter / "adapter_config.json").write_text(json.dumps(adapter_config), encoding="utf-8")
    del peft_model, model, tokenizer

    bundle = tmp_path / "scaleout"
    exported = export_verl_bundle(run, target_verl=VERL_TAG, out=bundle)
    assert exported["launchable"] is False
    assert not (bundle / "recipe/launch.sh").exists()

    materialized = materialize_verl_bundle(bundle, download=True)
    assert materialized["launchable"] is True
    assert materialized["distributed_execution_tested"] is False
    assert (bundle / "recipe/launch.sh").is_file()
    assert not (bundle / "recipe/launch.template.sh").exists()

    diagnosis = inspect_bridge_bundle(
        bundle,
        require_verl=True,
        require_tokenizer_load=True,
        require_adapter_payload=True,
    )
    assert diagnosis["verdict"] == "ok", diagnosis
    assert diagnosis["student_artifact_loadable"] is True
    assert diagnosis["teacher_artifact_loadable"] is True
    assert diagnosis["upstream_parse_passed"] is True
    assert diagnosis["launchable"] is True
    assert diagnosis["distributed_execution_tested"] is False
