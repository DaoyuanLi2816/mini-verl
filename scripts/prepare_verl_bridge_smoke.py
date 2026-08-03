"""Create a tiny standards-only source run for the pinned bridge smoke."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from miniverl.bridge.contract import BRIDGE_PROFILE, VERL_COMMIT, VERL_TAG
from miniverl.utils.runs import write_json


def _safetensors_bytes() -> bytes:
    header = json.dumps(
        {"bridge_probe": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * ((8 - len(header) % 8) % 8)
    return struct.pack("<Q", len(header)) + header + struct.pack("<f", 0.0)


def prepare_smoke_run(out: str | Path) -> Path:
    """Write one tiny PEFT/Parquet run directory without model execution."""
    root = Path(out)
    if root.exists():
        raise FileExistsError(f"smoke source already exists: {root}")
    model = root / "model"
    data = root / "data"
    model.mkdir(parents=True)
    data.mkdir()
    write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "run_id": "verl-bridge-standards-smoke",
            "status": "complete",
            "measurement_status": "artifact_smoke_not_model_training",
            "profile": BRIDGE_PROFILE,
            "target_verl": {"tag": VERL_TAG, "commit": VERL_COMMIT},
        },
    )
    write_json(
        root / "result.json",
        {
            "status": "artifact_smoke",
            "distributed_execution_status": "not tested",
            "model_training_status": "not run",
        },
    )
    write_json(
        model / "adapter_config.json",
        {
            "peft_type": "LORA",
            "base_model_name_or_path": "hf-internal-testing/tiny-random-gpt2",
            "revision": "71034c5d8bde858ff824298bdedc65515b97d2b9",
            "target_modules": ["c_attn"],
            "r": 4,
            "lora_alpha": 8,
            "lora_dropout": 0.0,
            "bias": "none",
            "task_type": "CAUSAL_LM",
        },
    )
    (model / "adapter_model.safetensors").write_bytes(_safetensors_bytes())
    write_json(model / "tokenizer_config.json", {"tokenizer_class": "GPT2Tokenizer"})
    rows = [
        {
            "data_source": "miniverl-bridge-smoke",
            "prompt": [{"role": "user", "content": "Return the number four."}],
            "ability": "formatting",
            "reward_model": {"style": "rule", "ground_truth": "4"},
            "extra_info": {"split": "smoke", "synthetic": True},
        }
    ]
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, data / "train.parquet")
    pq.write_table(table, data / "val.parquet")
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(prepare_smoke_run(args.out))


if __name__ == "__main__":
    main()
