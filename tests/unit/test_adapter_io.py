from __future__ import annotations

import json
from pathlib import Path


def test_exported_adapter_config_uses_portable_base_identity(tmp_path: Path) -> None:
    from miniverl.models.adapter_io import _normalize_exported_adapter_config

    config_path = tmp_path / "adapter_config.json"
    config_path.write_text(
        json.dumps(
            {
                "base_model_name_or_path": str(tmp_path / "machine-cache" / "snapshot"),
                "revision": None,
                "peft_type": "LORA",
            }
        ),
        encoding="utf-8",
    )

    _normalize_exported_adapter_config(
        config_path,
        model_id="Qwen/Qwen3-1.7B",
        revision="70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    )

    normalized = json.loads(config_path.read_text(encoding="utf-8"))
    assert normalized["base_model_name_or_path"] == "Qwen/Qwen3-1.7B"
    assert normalized["revision"] == "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
    assert normalized["peft_type"] == "LORA"
