"""A safetensors header is not a payload.

Up to v0.6.2 the bridge check parsed the header, found one tensor key and
reported the file valid. These tests pin the structural contract the official
reader enforces: offsets must tile the data segment exactly.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path
from typing import Any

import pytest

from miniverl.bridge.safetensors_check import inspect_safetensors

# ------------------------------------------------------------------ fixtures


def _write(path: Path, header: dict[str, Any], payload: bytes) -> Path:
    raw = json.dumps(header).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
    return path


def _valid(path: Path) -> Path:
    """Two contiguous tensors that exactly cover the payload."""
    header = {
        "__metadata__": {"format": "pt"},
        "a": {"dtype": "F32", "shape": [2, 2], "data_offsets": [0, 16]},
        "b": {"dtype": "I64", "shape": [2], "data_offsets": [16, 32]},
    }
    return _write(path, header, b"\x00" * 32)


# ---------------------------------------------------------------- valid file


def _official_reader_available() -> bool:
    """The numpy framework needs numpy; the torch-free environment has neither."""
    try:
        import numpy  # noqa: F401
        from safetensors import safe_open  # noqa: F401
    except ImportError:
        return False
    return True


def test_valid_file_passes_structural_validation(tmp_path: Path) -> None:
    check = inspect_safetensors(_valid(tmp_path / "adapter_model.safetensors"))
    assert check["status"] == "ok"
    assert check["problems"] == []
    assert check["tensors"] == 2
    assert check["actual_payload_bytes"] == 32
    # The stronger level is only claimed where the official reader can run.
    if _official_reader_available():
        assert check["verification_level"] == "tensor_materialization_validated"
    else:
        assert check["verification_level"] == "payload_structure_validated"
        assert check["official_reader_status"] == "dependency_missing"


def test_a_dependency_gap_is_not_evidence_the_file_is_invalid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The torch-free environment has safetensors but no numpy.

    Treating that ImportError as a rejection failed every structurally valid
    adapter in CI while the file was fine.
    """
    monkeypatch.setitem(sys.modules, "numpy", None)
    check = inspect_safetensors(_valid(tmp_path / "m.safetensors"))
    assert check["status"] == "ok"
    assert check["verification_level"] == "payload_structure_validated"
    assert check["official_reader_status"] == "dependency_missing"
    assert check["problems"] == []


def test_absent_file_reports_not_present(tmp_path: Path) -> None:
    check = inspect_safetensors(tmp_path / "absent.safetensors")
    assert check["status"] == "fail"
    assert check["verification_level"] == "not_present"


# ------------------------------------------------------- structural defects


def test_truncated_payload_is_rejected(tmp_path: Path) -> None:
    """The v0.6.2 reproducer: declared 64 bytes, zero bytes present."""
    path = _write(
        tmp_path / "adapter_model.safetensors",
        {"weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]}},
        b"",
    )
    check = inspect_safetensors(path)
    assert check["status"] == "fail"
    assert check["verification_level"] == "header_only"
    assert "only 0 payload bytes exist" in check["problems"][0]


def test_partially_truncated_payload_is_rejected(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "adapter_model.safetensors",
        {"weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]}},
        b"\x00" * 32,
    )
    check = inspect_safetensors(path)
    assert check["status"] == "fail"
    assert check["verification_level"] == "header_only"


def test_overlapping_offsets_are_rejected(tmp_path: Path) -> None:
    header = {
        "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
        "b": {"dtype": "F32", "shape": [4], "data_offsets": [8, 24]},
    }
    check = inspect_safetensors(_write(tmp_path / "m.safetensors", header, b"\x00" * 24))
    assert check["status"] == "fail"
    assert any("overlap" in problem for problem in check["problems"])


def test_uncovered_gap_is_rejected(tmp_path: Path) -> None:
    header = {
        "a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]},
        "b": {"dtype": "F32", "shape": [4], "data_offsets": [24, 40]},
    }
    check = inspect_safetensors(_write(tmp_path / "m.safetensors", header, b"\x00" * 40))
    assert check["status"] == "fail"
    assert any("uncovered" in problem for problem in check["problems"])


def test_trailing_bytes_are_rejected(tmp_path: Path) -> None:
    header = {"a": {"dtype": "F32", "shape": [4], "data_offsets": [0, 16]}}
    check = inspect_safetensors(_write(tmp_path / "m.safetensors", header, b"\x00" * 24))
    assert check["status"] == "fail"
    assert any("trailing" in problem for problem in check["problems"])


def test_impossible_shape_size_is_rejected(tmp_path: Path) -> None:
    """A 4x4 F32 tensor needs 64 bytes; the header claims 16."""
    header = {"a": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 16]}}
    check = inspect_safetensors(_write(tmp_path / "m.safetensors", header, b"\x00" * 16))
    assert check["status"] == "fail"
    assert "needs 64 bytes" in check["problems"][0]


def test_unknown_dtype_is_rejected(tmp_path: Path) -> None:
    header = {"a": {"dtype": "COMPLEX128", "shape": [1], "data_offsets": [0, 16]}}
    check = inspect_safetensors(_write(tmp_path / "m.safetensors", header, b"\x00" * 16))
    assert check["status"] == "fail"
    assert "unknown dtype" in check["problems"][0]


def test_reversed_offsets_are_rejected(tmp_path: Path) -> None:
    header = {"a": {"dtype": "F32", "shape": [4], "data_offsets": [16, 0]}}
    check = inspect_safetensors(_write(tmp_path / "m.safetensors", header, b"\x00" * 16))
    assert check["status"] == "fail"
    assert "not ordered" in check["problems"][0]


def test_malformed_header_json_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "m.safetensors"
    raw = b"{not json"
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    check = inspect_safetensors(path)
    assert check["status"] == "fail"
    assert "malformed safetensors header JSON" in check["problems"][0]


def test_zero_tensors_is_rejected(tmp_path: Path) -> None:
    check = inspect_safetensors(_write(tmp_path / "m.safetensors", {"__metadata__": {}}, b""))
    assert check["status"] == "fail"
    assert check["problems"] == ["safetensors contains no tensors"]


def test_header_longer_than_file_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "m.safetensors"
    path.write_bytes(struct.pack("<Q", 4096) + b"{}")
    check = inspect_safetensors(path)
    assert check["status"] == "fail"
    assert "past the end of the file" in check["problems"][0]


def test_missing_header_length_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "m.safetensors"
    path.write_bytes(b"\x00\x01")
    check = inspect_safetensors(path)
    assert check["status"] == "fail"
    assert check["problems"] == ["missing safetensors header length"]


def test_implausible_header_length_is_rejected_without_decoding(tmp_path: Path) -> None:
    """A hostile length must not become a multi-hundred-megabyte read."""
    path = tmp_path / "m.safetensors"
    path.write_bytes(struct.pack("<Q", 2**40) + b"{}")
    check = inspect_safetensors(path)
    assert check["status"] == "fail"
    assert "implausible" in check["problems"][0]


def test_checksum_correct_file_can_still_be_structurally_invalid(tmp_path: Path) -> None:
    """A matching SHA-256 says the bytes arrived intact, not that they are valid."""
    import hashlib

    path = _write(
        tmp_path / "m.safetensors",
        {"weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]}},
        b"",
    )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    check = inspect_safetensors(path)
    assert check["status"] == "fail"


# ------------------------------------------------------------- strict option


def test_strict_payload_requirement_fails_on_header_only(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "m.safetensors",
        {"weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]}},
        b"",
    )
    check = inspect_safetensors(path, require_payload=True)
    assert check["status"] == "fail"
    assert check["strict_payload_required"] is True
    assert check["strict_payload_satisfied"] is False


@pytest.mark.skipif(
    not _official_reader_available(), reason="the official safetensors reader is unavailable"
)
def test_strict_payload_requirement_passes_on_valid_file(tmp_path: Path) -> None:
    check = inspect_safetensors(_valid(tmp_path / "m.safetensors"), require_payload=True)
    assert check["status"] == "ok"
    assert check["strict_payload_satisfied"] is True
    assert check["verification_level"] == "tensor_materialization_validated"


def test_strict_payload_requirement_is_not_satisfied_without_the_official_reader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strict mode demands the strongest evidence, not the best available."""
    monkeypatch.setitem(sys.modules, "numpy", None)
    check = inspect_safetensors(_valid(tmp_path / "m.safetensors"), require_payload=True)
    assert check["status"] == "fail"
    assert check["strict_payload_satisfied"] is False
    assert check["verification_level"] == "payload_structure_validated"
    assert "--require-adapter-payload needs the official safetensors reader" in check["detail"]
    # Without the strict flag the same file still passes.
    relaxed = inspect_safetensors(_valid(tmp_path / "m.safetensors"))
    assert relaxed["status"] == "ok"


# ----------------------------------------------------- dependency behaviour


def test_missing_official_reader_stops_at_structural_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without the dependency the level must not claim materialization."""
    monkeypatch.setitem(sys.modules, "safetensors", None)
    check = inspect_safetensors(_valid(tmp_path / "m.safetensors"))
    assert check["status"] == "ok"
    assert check["verification_level"] == "payload_structure_validated"
    assert check["official_reader_status"] == "dependency_missing"


def test_official_reader_rejection_overrides_structural_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The official reader is the authority on the format."""
    import miniverl.bridge.safetensors_check as module

    monkeypatch.setattr(
        module, "_materialize", lambda path: ("rejected", "SafetensorError: synthetic", 0)
    )
    check = inspect_safetensors(_valid(tmp_path / "m.safetensors"))
    assert check["status"] == "fail"
    assert check["verification_level"] == "header_only"


# --------------------------------------------------------- doctor integration


def test_doctor_model_check_surfaces_the_level(tmp_path: Path) -> None:
    from miniverl.bridge.doctor import _check_model

    model = tmp_path / "model"
    model.mkdir(parents=True)
    (model / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": 8, "lora_alpha": 16, "target_modules": ["q_proj"]}),
        encoding="utf-8",
    )
    _write(
        model / "adapter_model.safetensors",
        {"weight": {"dtype": "F32", "shape": [4, 4], "data_offsets": [0, 64]}},
        b"",
    )

    check = _check_model(tmp_path)

    assert check["status"] == "fail"
    assert check["safetensors_verification_level"] == "header_only"
    assert "loadable" not in check["detail"].lower()
