"""Generate small upstream-shaped direct-run workflows, not native recipes."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def generate() -> dict[Path, bytes]:
    outputs = {}
    for algorithm in ("ppo", "grpo"):
        original = ROOT / f"src/miniverl/resources/verl_{algorithm}.yaml"
        source = yaml.safe_load(original.read_bytes())
        source.pop("miniverl")
        source["data"]["filter_overlong_prompts"] = True
        source["data"]["val_files"] = ["data/rl-prompts.parquet"]
        source["data"]["validation_shuffle"] = False
        source["trainer"].update(total_epochs=1, val_before_train=True, test_freq=1)
        source["actor_rollout_ref"]["actor"]["ppo_epochs"] = 2
        source["actor_rollout_ref"]["actor"]["optim"]["betas"] = [0.9, 0.999]
        source["actor_rollout_ref"]["rollout"]["val_kwargs"] = {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "n": 1,
        }
        source["reward"] = {
            "reward_manager": {"name": "naive"},
            "custom_reward_function": {"path": "local-reward.py", "name": "compute_score"},
        }
        if algorithm == "ppo":
            source["critic"]["model"] = copy.deepcopy(source["actor_rollout_ref"]["model"])
        else:
            source["actor_rollout_ref"]["actor"]["loss_agg_mode"] = "seq-mean-token-mean"
        target = ROOT / f"src/miniverl/resources/verl_direct_{algorithm}.yaml"
        outputs[target] = (
            "# Bounded systems workflow in upstream verl schema; runtime bindings supply local inputs.\n"
            + yaml.safe_dump(source, sort_keys=False)
        ).encode()
        outputs[target.with_suffix(".provenance.json")] = (
            json.dumps(
                {
                    "source": original.relative_to(ROOT).as_posix(),
                    "source_sha256": hashlib.sha256(original.read_bytes()).hexdigest(),
                    "purpose": "bounded direct-run systems qualification; no quality benchmark",
                    "adaptations": [
                        "remove miniVERL-specific source keys",
                        "two actor epochs",
                        "validation each cycle",
                        "prompt prefilter",
                        "explicit target-length reward implementation binding",
                        "GRPO sequence-mean reduction",
                    ],
                    "sha256": hashlib.sha256(outputs[target]).hexdigest(),
                },
                indent=2,
            )
            + "\n"
        ).encode()
    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    for path, content in generate().items():
        if args.check:
            if not path.is_file() or path.read_bytes() != content:
                raise SystemExit(f"stale direct example: {path.name}")
        else:
            path.write_bytes(content)
