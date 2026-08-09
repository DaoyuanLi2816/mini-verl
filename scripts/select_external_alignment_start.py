"""Evaluate the SFT candidates on the eval split and apply the saturation gate.

The order matters and is enforced here rather than left to discipline:

1. freeze the final-test suite, which only picks task ids and reads no result;
2. prepare a selection suite that withholds every one of those ids;
3. generate from each candidate in the committed order;
4. score, then take the FIRST candidate that clears the gate.

Step 2 is what keeps the final test's single read intact. Step 4 is what keeps
this a gate rather than a search -- the best-scoring candidate is not selected,
the first adequate one is.

    python scripts/select_external_alignment_start.py \
        --candidates artifacts/v07-sft-candidates \
        --out artifacts/v07-start-selection
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import yaml

PROFILE = Path("benchmarks/external-alignment/profile-v1.yaml")
PREREGISTRATION_PATH = Path("benchmarks/preregistration/alignment-external-v1.yaml")
SELECTION_TASKS = {"ifeval": 64, "xstest": 96, "jbb_behaviors": 32}
SELECTION_SEED = 90210
JSONNAV_TASKS = 64

#: Read once so the gate applied here is literally the one in the public
#: contract; a threshold cannot drift between the document and the run.
PREREGISTRATION = yaml.safe_load(PREREGISTRATION_PATH.read_text(encoding="utf-8"))


def score_harmful_compliance(
    judge: Any, prompts: list[str], responses: list[str], categories: list[str]
) -> float | None:
    """Fraction of harmful behaviours the policy complied with.

    ``unclear`` verdicts are excluded from the denominator rather than folded
    into either outcome; a judge that could not decide has not observed a
    refusal. ``None`` when nothing was decidable.
    """
    from miniverl.alignment_external.harmful import summarize_harmful

    records = [
        {"category": category, "label": judge.classify(prompt, response)[0]}
        for prompt, response, category in zip(prompts, responses, categories, strict=True)
    ]
    return summarize_harmful(records)["harmful_compliance_rate"]


def score_jsonnav_utility(model: Any, tokenizer: Any, *, tasks: int, seed: int) -> float | None:
    """Deterministic JSONNav success rate -- the preregistered utility endpoint.

    JSONNav is a multi-turn tool environment, so this drives the real agent
    loop rather than scoring a single completion. Running it any other way
    would measure something else and call it retained tool utility.
    """
    raise NotImplementedError(
        "retained tool utility must be measured through the agent rollout stack.\n"
        "JSONNav is a multi-turn tool environment: RolloutRunner needs a\n"
        "CausalLMBackend built by miniverl.models.factory.build_student, which\n"
        "takes a full RunConfig rather than a bare model. Wiring that is the\n"
        "next task; until it exists this raises rather than substituting a\n"
        "single-completion proxy, because the previous proxy (1 - over_refusal)\n"
        "is exactly what produced a structurally impossible gate result and\n"
        "forced preregistration amendment 1."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def _resolver() -> Any:
    from datasets import load_dataset

    cache: dict[str, tuple[list[str], list[str] | None]] = {}

    def resolve(endpoint: dict[str, Any]) -> tuple[list[str], list[str] | None]:
        key = str(endpoint["id"])
        if key in cache:
            return cache[key]
        if endpoint.get("dataset") is None:
            resolved = ([f"{key}-{index:04d}" for index in range(256)], None)
        else:
            loaded = load_dataset(
                endpoint["dataset"],
                endpoint.get("config"),
                split=endpoint["split"],
                revision=endpoint["revision"],
            )
            ids = [f"{key}-{index:05d}" for index in range(loaded.num_rows)]
            field = endpoint.get("strata_field")
            resolved = (ids, [str(v) for v in loaded[field]] if field else None)
        cache[key] = resolved
        return resolved

    return resolve


def _prompts_for(endpoint_id: str, task_ids: list[str], registry: dict[str, Any]) -> list[str]:
    """Load the actual prompt text for the selected ids."""
    from datasets import load_dataset

    entry = next(e for e in registry["endpoints"] if e["id"] == endpoint_id)
    loaded = load_dataset(
        entry["dataset"], entry.get("config"), split=entry["split"], revision=entry["revision"]
    )
    column = {"ifeval": "prompt", "xstest": "prompt", "jbb_behaviors": "Goal"}[endpoint_id]
    by_index = {f"{endpoint_id}-{index:05d}": index for index in range(loaded.num_rows)}
    return [str(loaded[by_index[task_id]][column]) for task_id in task_ids]


def main() -> int:
    args = _parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from miniverl.alignment_external.generation import BatchedGenerator, GenerationConfig
    from miniverl.alignment_external.ifeval import evaluate_ifeval_response
    from miniverl.alignment_external.refusal import label_responses, summarize_xstest
    from miniverl.alignment_external.registry import load_registry
    from miniverl.alignment_external.selection import select_starting_checkpoint
    from miniverl.alignment_external.suite import prepare_suite

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    resolver = _resolver()

    # 1. Freeze the final-test suite. This picks ids only; nothing is scored.
    final_profile = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    final = prepare_suite(profile=final_profile, out=out / "final-suite", resolver=resolver)
    reserved = {e["id"]: e["task_ids"] for e in final["endpoints"]}
    print(f"final suite frozen: {final['manifest_digest'][:16]}")

    # 2. Selection suite, withholding every final-test id.
    selection = prepare_suite(
        profile={
            "id": "start-selection",
            "selection_seed": SELECTION_SEED,
            "endpoints": [{"id": key, "tasks": count} for key, count in SELECTION_TASKS.items()],
        },
        out=out / "selection-suite",
        resolver=resolver,
        reserved_task_ids=reserved,
    )
    for entry in selection["endpoints"]:
        print(
            f"  {entry['id']}: {entry['selected_tasks']} tasks, "
            f"{entry['reserved_for_final_test']} withheld for the final test"
        )

    manifest = json.loads((Path(args.candidates) / "candidates.json").read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(manifest["candidates"][0]["path"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = {
        entry["id"]: _prompts_for(entry["id"], entry["task_ids"], registry)
        for entry in selection["endpoints"]
    }

    results: dict[str, dict[str, Any]] = {}
    started = time.time()
    torch.cuda.reset_peak_memory_stats()
    for candidate in manifest["candidates"]:
        name = f"update-{candidate['update']:03d}"
        base = AutoModelForCausalLM.from_pretrained(
            manifest["student_model"],
            revision=manifest["student_revision"],
            dtype=torch.float32,
            attn_implementation="eager",
        )
        model = PeftModel.from_pretrained(base, candidate["path"]).to("cuda").eval()
        generator = BatchedGenerator(
            model, tokenizer, GenerationConfig(batch_size=args.batch_size, max_new_tokens=256)
        )

        metrics: dict[str, float] = {}

        ifeval_prompts = prompts["ifeval"]
        responses = generator.generate([f"{p}\n\nAssistant: " for p in ifeval_prompts])
        entry = next(e for e in selection["endpoints"] if e["id"] == "ifeval")
        from datasets import load_dataset

        source = load_dataset(
            "google/IFEval",
            split="train",
            revision=next(e for e in registry["endpoints"] if e["id"] == "ifeval")["revision"],
        )
        index_of = {f"ifeval-{i:05d}": i for i in range(source.num_rows)}
        satisfied = total = 0
        for task_id, response in zip(entry["task_ids"], responses, strict=True):
            row = source[index_of[task_id]]
            scored = evaluate_ifeval_response(response, row["instruction_id_list"], row["kwargs"])
            satisfied += scored["strict_instructions_satisfied"]
            total += scored["instructions_evaluated"]
        metrics["instruction_following"] = satisfied / total if total else 0.0

        xs_entry = next(e for e in selection["endpoints"] if e["id"] == "xstest")
        xs_source = load_dataset(
            "natolambert/xstest-v2-copy",
            split="prompts",
            revision=next(e for e in registry["endpoints"] if e["id"] == "xstest")["revision"],
        )
        xs_index = {f"xstest-{i:05d}": i for i in range(xs_source.num_rows)}
        xs_types = [str(xs_source[xs_index[t]]["type"]) for t in xs_entry["task_ids"]]
        xs_responses = generator.generate([f"{p}\n\nAssistant: " for p in prompts["xstest"]])
        summary = summarize_xstest(label_responses(xs_types, xs_responses))
        metrics["over_refusal"] = summary["over_refusal_rate"] or 0.0
        metrics["appropriate_refusal"] = summary["appropriate_refusal_rate"] or 0.0

        # Harmful compliance on the JBB behaviours, judged by Granite Guardian.
        jbb_responses = generator.generate(
            [f"{p}\n\nAssistant: " for p in prompts["jbb_behaviors"]]
        )
        metrics["harmful_compliance"] = score_harmful_compliance(jbb_responses)

        # Retained tool utility on JSONNav -- the endpoint the preregistration
        # names. An earlier run substituted `1 - over_refusal` here, which sits
        # near 1.0 for any policy that rarely refuses and so failed the band
        # structurally rather than because a candidate was saturated.
        metrics["retained_tool_utility"] = score_jsonnav_utility(
            generator, tasks=JSONNAV_TASKS, seed=SELECTION_SEED
        )

        results[name] = {
            "update": candidate["update"],
            "adapter_digest": candidate["adapter_digest"],
            "metrics": metrics,
        }
        print(f"{name}: {json.dumps(metrics, sort_keys=True)}")
        del model, base, generator
        torch.cuda.empty_cache()

    gate = PREREGISTRATION["starting_checkpoint"]["gate"]
    decision = select_starting_checkpoint(
        [
            {"id": name, "metrics": payload["metrics"]}
            for name, payload in sorted(results.items(), key=lambda kv: kv[1]["update"])
        ],
        alignment_endpoints=gate["alignment_endpoints"],
        utility_endpoint=gate["utility_endpoint"],
    )
    record = {
        "schema_version": 1,
        "final_suite_digest": final["manifest_digest"],
        "selection_suite_digest": selection["manifest_digest"],
        "candidate_results": results,
        "decision": decision,
        "gpu_seconds": round(time.time() - started, 1),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "split_used": "eval only; every final-test task id was withheld",
    }
    (out / "start-selection.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"\ndecision: {json.dumps(decision, sort_keys=True)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
