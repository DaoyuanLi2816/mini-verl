"""Generate the two installed, bounded RL workflows and their upstream adaptation ledger."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from miniverl.bridge.rl_v09 import _flatten

ROOT = Path(__file__).resolve().parents[1]


def artifacts() -> dict[str, bytes]:
    corpus = ROOT / "benchmarks/compatibility/verl-v0.9.0"
    manifest = json.loads((corpus / "manifest.json").read_text())
    sources = {row["name"]: row for row in manifest["cases"]}
    outputs = {}
    for algorithm in ("ppo", "grpo"):
        source = yaml.safe_load((ROOT / "examples/verl-rl-v0.9-single-gpu-ppo.yaml").read_text())
        source["data"]["train_files"] = ["data/rl-prompts.parquet"]
        source["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"] = 2
        source["critic"]["ppo_mini_batch_size"] = 2
        source["miniverl"]["actor"]["quantization"] = "none"
        source["miniverl"]["reward"]["provider"] = "target_length"
        source["trainer"]["experiment_name"] = f"local-{algorithm}"
        source["trainer"].pop("total_epochs")
        if algorithm == "grpo":
            source.pop("critic")
            source["miniverl"].pop("critic")
            source["actor_rollout_ref"]["actor"]["ppo_epochs"] = 1
            source["algorithm"] = {
                "adv_estimator": "grpo",
                "norm_adv_by_std_in_grpo": True,
                "use_kl_in_reward": False,
            }
        original = _flatten(yaml.safe_load((corpus / f"{algorithm}-launch.yaml").read_text()))
        adapted = _flatten(source)
        changes = [
            {
                "field": key,
                "upstream_present": key in original,
                "upstream": original.get(key),
                "local_present": key in adapted,
                "local": adapted.get(key),
            }
            for key in sorted(original.keys() | adapted.keys())
            if original.get(key) != adapted.get(key)
        ]
        provenance = {
            "schema_version": 1,
            "kind": "educational_local_adaptation",
            "upstream": {
                key: sources[algorithm][key]
                for key in ("repository", "tag", "commit", "path", "source_sha256")
            },
            "scope": "A new bounded length-reward exercise, not an execution of the original GSM8K/MATH experiment.",
            "changes": changes,
        }
        outputs[f"verl_{algorithm}.yaml"] = (
            "# Packaged single-GPU educational workflow; profile verl-rl-v0.9-single-gpu-v3.\n"
            "# Provenance and every upstream adaptation: corresponding .provenance.json.\n"
            + yaml.safe_dump(source, sort_keys=False)
        ).encode()
        outputs[f"verl_{algorithm}.provenance.json"] = (
            json.dumps(provenance, indent=2, sort_keys=True) + "\n"
        ).encode()
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for name, content in artifacts().items():
        path = ROOT / "src/miniverl/resources" / name
        if args.check:
            if not path.exists() or path.read_bytes() != content:
                raise SystemExit(f"stale RL example: {name}")
        else:
            path.write_bytes(content)
    print("PPO and GRPO installed examples verified")


if __name__ == "__main__":
    main()
