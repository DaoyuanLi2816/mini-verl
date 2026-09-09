"""Resolve pinned upstream launch examples without importing or launching verl.

The small shell reader handles only literal variable assignments and argument
arrays. It never evaluates shell commands. Hydra composes the upstream config
tree, including defaults and OmegaConf resolvers; both the complete config and
the explicit launch surface receive independent compiler reports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from miniverl.algorithms.contract import UPSTREAM_VERL_COMMIT, UPSTREAM_VERL_TAG
from miniverl.bridge.rl_v09 import (
    VERL_REPOSITORY,
    VERL_RL_V09_PRODUCT_PROFILE,
    publish_imported_verl_rl_v09,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    "ppo": "examples/ppo_trainer/run_qwen3_8b_fsdp.sh",
    "grpo": "examples/grpo_trainer/run_qwen3_4b_fsdp.sh",
    "dr-grpo": "examples/grpo_trainer/run_qwen3_4b_fsdp.sh",
    "rloo": "examples/rloo_trainer/run_qwen3_8b_fsdp.sh",
    "reinforce-plus-plus": "examples/reinforce_plus_plus_trainer/run_qwen3_8b_fsdp.sh",
    "lora-grpo": "examples/tuning/lora/run_qwen3_8b_fsdp.sh",
    "trained-rm-grpo": "examples/grpo_trainer/run_mistral_nemo_12b_skyworkrm_fsdp.sh",
}


def launch_arguments(source: str) -> list[str]:
    """Read this pinned corpus's literal shell subset; refuse other syntax."""
    variables = {"HOME": "/upstream-home", "DEVICE": "gpu"}
    arrays: dict[str, list[str]] = {}
    current = None

    def expand(text: str) -> str:
        def replace(match: re.Match[str]) -> str:
            name = match.group(1) or match.group(2)
            if name not in variables:
                raise ValueError(f"unbound shell variable {name}")
            return variables[name]

        expanded = re.sub(r"\$\{(\w+)\}|\$(\w+)", replace, text)
        if "$" in expanded or "`" in expanded:
            raise ValueError(f"non-literal shell syntax: {text}")
        return expanded

    for line in source.splitlines():
        if re.match(r"^\w+=\($", line):
            current = line[:-2]
            arrays[current] = []
        elif current is not None:
            if line.strip() == ")":
                current = None
            else:
                for token in shlex.split(line, comments=True):
                    arrays[current].append(expand(token))
        elif match := re.match(r"^(\w+)=(.*)$", line):
            name, raw = match.groups()
            if name == "DEVICE":
                continue  # Explicit CUDA branch binding; never execute its probe.
            raw = raw.replace("$(date +%Y%m%d_%H%M)", "19700101_0000")
            tokens = shlex.split(raw, comments=True)
            if len(tokens) != 1:
                raise ValueError(f"non-literal assignment: {line}")
            value = tokens[0]
            default = re.fullmatch(r"\$\{\w+:-(.*)\}", value)
            variables[name] = expand(default.group(1) if default else value)
    launch = source.split("-m verl.trainer.main_ppo", 1)
    if len(launch) != 2:
        raise ValueError("expected the pinned main_ppo launch")
    order = re.findall(r"\$\{(\w+)\[@\]\}", launch[1])
    if not order or set(order) != set(arrays):
        raise ValueError("unaccounted launch argument array")
    return [argument for name in order for argument in arrays[name]]


def resolve(upstream: Path, arguments: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    with initialize_config_dir(
        version_base="1.3", config_dir=str(upstream / "verl/trainer/config")
    ):
        config = compose(config_name="ppo_trainer", overrides=arguments)
        full = OmegaConf.to_container(config, resolve=True, throw_on_missing=False)
    # Preserve every explicit launch leaf; derive its value from the composed
    # upstream configuration, not from a second YAML parser's number semantics.
    explicit = OmegaConf.create({})
    for argument in arguments:
        key = argument.split("=", 1)[0].lstrip("+")
        OmegaConf.update(explicit, key, OmegaConf.select(config, key), merge=False)
    return dict(full), dict(OmegaConf.to_container(explicit, resolve=True))


def generate(upstream: Path) -> dict[str, str]:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=upstream, text=True).strip()
    if commit != UPSTREAM_VERL_COMMIT:
        raise ValueError(f"expected pinned upstream {UPSTREAM_VERL_COMMIT}, got {commit}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=upstream, text=True).strip():
        raise ValueError("upstream corpus source must be clean")
    outputs = {}
    results = []
    with tempfile.TemporaryDirectory(prefix="miniverl-corpus-") as temporary:
        for name, path in CASES.items():
            content = (upstream / path).read_bytes()
            arguments = launch_arguments(content.decode("utf-8"))
            documentation = None
            if name == "dr-grpo":
                doc_path = "examples/grpo_trainer/README.md"
                doc_bytes = (upstream / doc_path).read_bytes()
                additions = [
                    "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm",
                    "actor_rollout_ref.actor.use_kl_loss=False",
                    "algorithm.norm_adv_by_std_in_grpo=False",
                ]
                if not all(value in doc_bytes.decode() for value in additions):
                    raise ValueError("the upstream Dr.GRPO instruction changed")
                arguments.extend(additions)
                documentation = {
                    "path": doc_path,
                    "sha256": hashlib.sha256(doc_bytes).hexdigest(),
                    "overrides": additions,
                }
            full, explicit = resolve(upstream, arguments)
            case: dict[str, Any] = {
                "name": name,
                "repository": VERL_REPOSITORY,
                "tag": UPSTREAM_VERL_TAG,
                "commit": commit,
                "path": path,
                "source_sha256": hashlib.sha256(content).hexdigest(),
                "environment_bindings": {
                    "HOME": "/upstream-home",
                    "DEVICE": "gpu",
                    "date_label": "19700101_0000",
                },
                "resolver": "hydra-core==1.3.2; upstream ppo_trainer defaults; no model imports",
                "launch_arguments": arguments,
                "documented_variant": documentation,
            }
            for scope, payload in (("complete", full), ("launch", explicit)):
                rendered = yaml.safe_dump(payload, sort_keys=True, allow_unicode=True)
                filename = f"{name}-{scope}.yaml"
                outputs[filename] = rendered
                source = Path(temporary) / filename
                source.write_text(rendered, encoding="utf-8", newline="\n")
                report = publish_imported_verl_rl_v09(
                    source,
                    out=Path(temporary) / f"{name}-{scope}-local.yaml",
                    target_verl=UPSTREAM_VERL_TAG,
                    profile=VERL_RL_V09_PRODUCT_PROFILE,
                )
                outputs[f"{name}-{scope}-report.json"] = (
                    json.dumps(report, indent=2, sort_keys=True) + "\n"
                )
                counts = Counter(row["classification"] for row in report["field_classification"])
                case[scope] = {
                    "status": report["status"],
                    "executable": report["executable"],
                    "fields": len(report["field_classification"]),
                    "classification_counts": dict(counts),
                    "required_user_input": report["required_user_input"],
                    "unsupported_fields": report["unsupported_fields"],
                    "resolved_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
                }
            results.append(case)
    totals = {
        "upstream_examples_examined": len(set(CASES.values())),
        "configuration_cases": len(results),
        "locally_executable": sum(row["complete"]["executable"] for row in results),
        "requires_user_local_input": sum(
            row["complete"]["status"] == "needs_user_input" for row in results
        ),
        "unsupported_or_ambiguous": sum(row["complete"]["status"] == "rejected" for row in results),
        "distributed_fields_safely_lowered": sum(
            row["complete"]["classification_counts"].get(kind, 0)
            for row in results
            for kind in ("distributed_only", "locally_lowered")
        ),
        "not_implemented_fields": sum(
            row["complete"]["classification_counts"].get("not_implemented", 0) for row in results
        ),
    }
    outputs["manifest.json"] = (
        json.dumps(
            {"schema_version": 1, "totals": totals, "cases": results}, indent=2, sort_keys=True
        )
        + "\n"
    )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "benchmarks/compatibility/verl-v0.9.0")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    outputs = generate(args.upstream.resolve())
    for name, text in outputs.items():
        path = args.out / name
        if args.check:
            if not path.exists() or path.read_bytes() != text.encode():
                raise SystemExit(f"stale compatibility corpus: {name}")
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode())
    print(f"verified {len(CASES)} upstream configuration cases; {len(outputs)} artifacts")


if __name__ == "__main__":
    main()
