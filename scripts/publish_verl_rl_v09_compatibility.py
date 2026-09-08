"""Publish the deterministic compatibility report for the verl v0.9 RL example."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from miniverl.bridge.rl_v09 import VERL_RL_V09_PPO_PROFILE, publish_imported_verl_rl_v09

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "examples/verl-rl-v0.9-single-gpu.yaml"
TARGET = ROOT / "docs/generated/verl-rl-v0.9-compatibility.json"
PPO_SOURCE = ROOT / "examples/verl-rl-v0.9-single-gpu-ppo.yaml"
PPO_TARGET = ROOT / "docs/generated/verl-rl-v0.9-ppo-compatibility.json"


def render(source: Path = SOURCE, *, profile: str | None = None) -> str:
    with tempfile.TemporaryDirectory(prefix="miniverl-rl-v09-") as directory:
        output = Path(directory) / "verl-rl-v0.9-single-gpu.yaml"
        report = publish_imported_verl_rl_v09(
            source,
            out=output,
            target_verl="v0.9.0",
            **({"profile": profile} if profile is not None else {}),
        )
    if report["status"] != "accepted" or report["executable"] is not True:
        raise RuntimeError("the committed verl v0.9 RL example no longer compiles")
    return json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = render()
    expected_ppo = render(PPO_SOURCE, profile=VERL_RL_V09_PPO_PROFILE)
    if args.check:
        stale = [
            path.relative_to(ROOT)
            for path, content in ((TARGET, expected), (PPO_TARGET, expected_ppo))
            if not path.is_file() or path.read_text(encoding="utf-8") != content
        ]
        if stale:
            parser.error(f"generated compatibility report is stale: {stale}")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(expected, encoding="utf-8", newline="\n")
    PPO_TARGET.write_text(expected_ppo, encoding="utf-8", newline="\n")
    print(TARGET.relative_to(ROOT).as_posix())
    print(PPO_TARGET.relative_to(ROOT).as_posix())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
