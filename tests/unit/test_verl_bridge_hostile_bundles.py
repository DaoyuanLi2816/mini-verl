"""`bridge doctor` treats an exported bundle as hostile input.

A bundle arrives from someone else. These tests tamper with a real exported
bundle and assert that the doctor reports the damage, never executes anything,
and never lets one passing check launder another.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

MARKER_NAME = "PWNED.txt"


# ------------------------------------------------------------------ fixtures


def _safetensors_bytes() -> bytes:
    header = json.dumps(
        {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    return struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.0)


def _run(tmp_path: Path) -> Path:
    run = tmp_path / "run"
    model = run / "model"
    data = run / "data"
    model.mkdir(parents=True)
    data.mkdir()
    (run / "manifest.json").write_text(
        json.dumps({"run_id": "hostile-source", "status": "complete"}), encoding="utf-8"
    )
    (run / "result.json").write_text(json.dumps({"strict_success": 1.0}), encoding="utf-8")
    (model / "adapter_config.json").write_text(
        json.dumps(
            {
                "peft_type": "LORA",
                "base_model_name_or_path": "Qwen/Qwen3-0.6B",
                "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
                "target_modules": ["q_proj"],
                "r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.0,
                "bias": "none",
                "task_type": "CAUSAL_LM",
            }
        ),
        encoding="utf-8",
    )
    (model / "adapter_model.safetensors").write_bytes(_safetensors_bytes())
    (model / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer"}), encoding="utf-8"
    )
    rows = [
        {
            "data_source": "calculator",
            "prompt": [{"role": "user", "content": "2+2"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": "4"},
            "extra_info": {"split": "train"},
        }
    ]
    pq.write_table(pa.Table.from_pylist(rows), data / "train.parquet")
    pq.write_table(pa.Table.from_pylist(rows), data / "val.parquet")
    return run


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """A genuine exported bundle that the doctor accepts before tampering."""
    from miniverl.bridge.contract import VERL_TAG
    from miniverl.bridge.export import export_verl_bundle

    out = tmp_path / "export"
    export_verl_bundle(_run(tmp_path), target_verl=VERL_TAG, out=out)
    return out


def _reseal(bundle: Path) -> None:
    """Recompute SHA256SUMS, exactly as a competent attacker would."""
    checksum = bundle / "provenance" / "SHA256SUMS"
    lines = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file() or path == checksum:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(bundle).as_posix()}")
    checksum.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


def _inspect(bundle: Path, **kwargs: Any) -> dict[str, Any]:
    from miniverl.bridge.doctor import inspect_bridge_bundle

    return inspect_bridge_bundle(bundle, **kwargs)


# ------------------------------------------------------------- baseline


def test_untampered_bundle_passes(bundle: Path) -> None:
    report = _inspect(bundle)
    assert report["verdict"] == "ok"
    assert report["artifact_hashes"]["status"] == "ok"
    assert report["reward_verification_level"] == "interface_statically_verified"
    assert report["reward_code_executed"] is False


# ------------------------------------------ checksums do not confer trust


def test_self_consistent_checksums_cannot_launder_hostile_reward_code(
    bundle: Path, tmp_path: Path
) -> None:
    """An attacker who reseals the bundle still cannot get code executed."""
    marker = tmp_path / MARKER_NAME
    (bundle / "reward" / "reward_or_verifier_scaffold.py").write_text(
        "from pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('owned', encoding='utf-8')\n"
        "\n"
        "def compute_score(data_source, solution_str, ground_truth, extra_info=None):\n"
        "    return 1.0\n",
        encoding="utf-8",
    )
    _reseal(bundle)

    report = _inspect(bundle)

    assert not marker.exists(), "diagnosing a bundle executed its reward code"
    # The attacker's checksums are internally consistent...
    assert report["artifact_hashes"]["status"] == "ok"
    # ...and that buys them nothing.
    assert report["verdict"] == "fail"
    assert report["reward_verification_level"] == "syntax_valid"
    assert report["reward_code_executed"] is False


def test_resealed_truncated_adapter_is_still_rejected(bundle: Path) -> None:
    """A matching digest says the bytes arrived intact, not that they are valid."""
    header = json.dumps({"weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]}})
    raw = header.encode("utf-8")
    (bundle / "model" / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    _reseal(bundle)

    report = _inspect(bundle)

    assert report["artifact_hashes"]["status"] == "ok"
    assert report["verdict"] == "fail"
    assert report["safetensors_verification_level"] == "header_only"


def test_tampered_file_without_resealing_fails_the_hash_check(bundle: Path) -> None:
    (bundle / "recipe" / "verl-overrides.yaml").write_text("tampered: true\n", encoding="utf-8")
    report = _inspect(bundle)
    assert report["artifact_hashes"]["status"] == "fail"
    assert "recipe/verl-overrides.yaml" in report["artifact_hashes"]["problems"]
    assert report["verdict"] == "fail"


# ------------------------------------------------- conflicting provenance


def test_conflicting_pinned_verl_fields_fail_closed(bundle: Path) -> None:
    path = bundle / "recipe" / "REQUIRED_VERL.txt"
    text = path.read_text(encoding="utf-8").replace("VERL_TAG=v0.8.0", "VERL_TAG=v9.9.9")
    path.write_text(text, encoding="utf-8")
    _reseal(bundle)

    report = _inspect(bundle)

    assert report["target_verl"]["status"] == "fail"
    assert report["verdict"] == "fail"


def test_config_contradicting_the_adapter_fails_closed(bundle: Path) -> None:
    """The overrides claim a LoRA rank the adapter does not have."""
    path = bundle / "recipe" / "verl-overrides.yaml"
    text = path.read_text(encoding="utf-8").replace("lora_rank: 4", "lora_rank: 64")
    path.write_text(text, encoding="utf-8")
    _reseal(bundle)

    report = _inspect(bundle)

    assert report["config_profile"]["status"] == "fail"
    assert "actor_rollout_ref.model.lora_rank" in report["config_profile"]["model_handoff_problems"]
    assert report["verdict"] == "fail"


def test_malformed_reward_python_is_reported_not_raised(bundle: Path) -> None:
    (bundle / "reward" / "reward_or_verifier_scaffold.py").write_text(
        "def compute_score(:\n", encoding="utf-8"
    )
    _reseal(bundle)

    report = _inspect(bundle)

    assert report["reward_verification_level"] == "not_present"
    assert report["reward_scaffold_interface"]["findings"][0]["category"] == "syntax_error"
    assert report["verdict"] == "fail"


# ------------------------------------------------------------ hostile shapes


def test_implausible_safetensors_header_is_not_decoded(bundle: Path) -> None:
    """A hostile length must not become a multi-gigabyte read."""
    (bundle / "model" / "adapter_model.safetensors").write_bytes(struct.pack("<Q", 2**60) + b"{}")
    _reseal(bundle)

    report = _inspect(bundle)

    assert report["safetensors_verification_level"] == "header_only"
    assert "implausible" in report["model_adapter_loadability"]["safetensors"]["problems"][0]
    assert report["verdict"] == "fail"


def test_many_row_group_parquet_is_not_materialized_by_the_schema_check(
    bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A large dataset must not be read just to answer a schema question."""
    rows = [
        {
            "data_source": "calculator",
            "prompt": [{"role": "user", "content": f"q{index}"}],
            "ability": "math",
            "reward_model": {"style": "rule", "ground_truth": str(index)},
            "extra_info": {"split": "train"},
        }
        for index in range(4)
    ]
    for split in ("train", "val"):
        path = bundle / "data" / f"{split}.parquet"
        writer = pq.ParquetWriter(path, pa.Table.from_pylist(rows).schema)
        try:
            for _ in range(16):
                writer.write_table(pa.Table.from_pylist(rows))
        finally:
            writer.close()
    _reseal(bundle)

    def _forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the schema check decoded row data")

    monkeypatch.setattr(pq, "read_table", _forbidden)
    monkeypatch.setattr(pq.ParquetFile, "read_row_group", _forbidden)

    report = _inspect(bundle)

    assert report["parquet_schema"]["status"] == "ok"
    assert report["parquet_schema"]["rows"]["train"] == 64
    assert report["verdict"] == "ok"


@pytest.mark.skipif(sys.platform == "win32", reason="symlink creation needs privileges")
def test_symlinked_bundle_file_does_not_escape_the_hash_check(bundle: Path, tmp_path: Path) -> None:
    """A symlink pointing outside the bundle must not silently validate."""
    outside = tmp_path / "outside.txt"
    outside.write_text("not part of this bundle\n", encoding="utf-8")
    link = bundle / "recipe" / "linked.txt"
    link.symlink_to(outside)

    report = _inspect(bundle)

    # The link is a file inside the bundle that SHA256SUMS does not declare.
    assert report["artifact_hashes"]["status"] == "fail"
    assert "recipe/linked.txt" in report["artifact_hashes"]["problems"]


# ---------------------------------------------------- strict option wiring


def test_strict_adapter_payload_option_reaches_the_check(bundle: Path) -> None:
    header = json.dumps({"weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]}})
    raw = header.encode("utf-8")
    (bundle / "model" / "adapter_model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    _reseal(bundle)

    report = _inspect(bundle, require_adapter_payload=True)

    safetensors = report["model_adapter_loadability"]["safetensors"]
    assert safetensors["strict_payload_required"] is True
    assert safetensors["strict_payload_satisfied"] is False


def test_trusted_import_is_never_reached_through_require_verl(bundle: Path, tmp_path: Path) -> None:
    """--require-verl must not be a back door into executing bundle code."""
    marker = tmp_path / MARKER_NAME
    (bundle / "reward" / "reward_or_verifier_scaffold.py").write_text(
        '"""Clean."""\n'
        "from pathlib import Path\n"
        "\n"
        "def compute_score(data_source, solution_str, ground_truth, extra_info=None):\n"
        f"    Path({str(marker)!r}).write_text('called', encoding='utf-8')\n"
        "    return 1.0\n",
        encoding="utf-8",
    )
    _reseal(bundle)

    report = _inspect(bundle, require_verl=True)

    assert not marker.exists()
    assert report["reward_code_executed"] is False
    # A clean interface still verifies statically; the body is never run.
    assert report["reward_verification_level"] == "interface_statically_verified"
