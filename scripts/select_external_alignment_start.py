"""Evaluate the SFT candidates on the eval split and apply the saturation gate.

The order matters and is enforced here rather than left to discipline:

1. freeze the final-test suite, which only picks task ids and reads no result;
2. prepare a selection suite that withholds every one of those ids, JSONNav
   included;
3. score every candidate in the committed order;
4. take the FIRST candidate that clears the gate.

Step 2 keeps the final test's single read intact. Step 4 keeps this a gate
rather than a search: the best-scoring candidate is not selected, the first
adequate one is.

Retained tool utility is the real JSONNav verifier rate through the agent
runtime. The superseded `1 - over_refusal` proxy is what forced preregistration
amendment 1, so there is no proxy path here at all.

    python scripts/select_external_alignment_start.py \
        --candidates artifacts/v07-sft-candidates \
        --out artifacts/v07-start-selection
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path
from typing import Any

import yaml

PROFILE = Path("benchmarks/external-alignment/profile-v1.yaml")
PREREGISTRATION_PATH = Path("benchmarks/preregistration/alignment-external-v1.yaml")
SELECTION_TASKS = {"ifeval": 64, "xstest": 96, "jbb_behaviors": 32, "jsonnav_utility": 64}
SELECTION_SEED = 90210

#: Read once, so the gate applied here is literally the one in the public
#: contract and a threshold cannot drift between document and run.
PREREGISTRATION = yaml.safe_load(PREREGISTRATION_PATH.read_text(encoding="utf-8"))
GATE = PREREGISTRATION["starting_checkpoint"]["gate"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=8)
    return parser.parse_args()


def _resolver() -> Any:
    """Upstream task ids per endpoint, cached so each dataset loads once."""
    from datasets import load_dataset

    from miniverl.alignment_external.jsonnav_utility import JSONNAV_TASK_PREFIX

    cache: dict[str, tuple[list[str], list[str] | None]] = {}

    def resolve(endpoint: dict[str, Any]) -> tuple[list[str], list[str] | None]:
        key = str(endpoint["id"])
        if key in cache:
            return cache[key]
        if key == JSONNAV_TASK_PREFIX:
            # A fixed generated pool, so selection and final draw from the same
            # id space and disjointness is checkable.
            resolved = ([f"{key}-{index:05d}" for index in range(256)], None)
        elif endpoint.get("dataset") is None:
            resolved = ([f"{key}-{index:05d}" for index in range(256)], None)
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


def _rows_for(endpoint_id: str, task_ids: list[str], registry: dict[str, Any]) -> list[Any]:
    """Upstream rows behind the selected ids, in manifest order."""
    from datasets import load_dataset

    entry = next(e for e in registry["endpoints"] if e["id"] == endpoint_id)
    loaded = load_dataset(
        entry["dataset"], entry.get("config"), split=entry["split"], revision=entry["revision"]
    )
    index_of = {f"{endpoint_id}-{i:05d}": i for i in range(loaded.num_rows)}
    return [loaded[index_of[task_id]] for task_id in task_ids]


def main() -> int:
    args = _parse_args()
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from miniverl.alignment_external.generation import BatchedGenerator, GenerationConfig
    from miniverl.alignment_external.harmful import HarmfulJudge, summarize_harmful
    from miniverl.alignment_external.ifeval import evaluate_ifeval_response
    from miniverl.alignment_external.jsonnav_utility import JsonNavSettings, score_jsonnav_tasks
    from miniverl.alignment_external.refusal import label_responses, summarize_xstest
    from miniverl.alignment_external.registry import load_registry
    from miniverl.alignment_external.selection import select_starting_checkpoint
    from miniverl.alignment_external.suite import prepare_suite
    from miniverl.config.models import Quantization
    from miniverl.models.hf import HFBackend
    from miniverl.models.tokenizers import HFTokenizerAdapter

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    registry = load_registry()
    resolver = _resolver()

    final = prepare_suite(
        profile=yaml.safe_load(PROFILE.read_text(encoding="utf-8")),
        out=out / "final-suite",
        resolver=resolver,
    )
    reserved = {e["id"]: e["task_ids"] for e in final["endpoints"]}
    print(f"final suite frozen: {final['manifest_digest'][:16]}")

    selection = prepare_suite(
        profile={
            "id": "start-selection",
            "selection_seed": SELECTION_SEED,
            "endpoints": [{"id": key, "tasks": n} for key, n in SELECTION_TASKS.items()],
        },
        out=out / "selection-suite",
        resolver=resolver,
        reserved_task_ids=reserved,
    )
    by_id = {e["id"]: e for e in selection["endpoints"]}
    for entry in selection["endpoints"]:
        print(
            f"  {entry['id']}: {entry['selected_tasks']} tasks, "
            f"{entry['reserved_for_final_test']} withheld for the final test"
        )
    overlap = {
        key: sorted(set(by_id[key]["task_ids"]) & set(reserved.get(key, []))) for key in by_id
    }
    if any(overlap.values()):
        raise SystemExit(f"selection overlaps the final test: {overlap}")

    manifest = json.loads((Path(args.candidates) / "candidates.json").read_text(encoding="utf-8"))
    raw_tokenizer = AutoTokenizer.from_pretrained(manifest["candidates"][0]["path"])
    if raw_tokenizer.pad_token_id is None:
        raw_tokenizer.pad_token = raw_tokenizer.eos_token

    ifeval_rows = _rows_for("ifeval", by_id["ifeval"]["task_ids"], registry)
    xstest_rows = _rows_for("xstest", by_id["xstest"]["task_ids"], registry)
    jbb_rows = _rows_for("jbb_behaviors", by_id["jbb_behaviors"]["task_ids"], registry)
    jbb_prompts = [str(row["Goal"]) for row in jbb_rows]
    jbb_categories = [str(row["Category"]) for row in jbb_rows]

    generations: dict[str, dict[str, Any]] = {}
    started = time.time()
    torch.cuda.reset_peak_memory_stats()

    # Pass one: every candidate generates. The judge is not resident yet.
    for candidate in manifest["candidates"]:
        name = f"update-{candidate['update']:03d}"
        base = AutoModelForCausalLM.from_pretrained(
            manifest["student_model"],
            revision=manifest["student_revision"],
            dtype=torch.float32,
            attn_implementation="eager",
        )
        model = PeftModel.from_pretrained(base, candidate["path"]).to("cuda").eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)

        generator = BatchedGenerator(
            model,
            raw_tokenizer,
            GenerationConfig(batch_size=args.batch_size, max_new_tokens=256),
        )
        ifeval_responses = generator.generate(
            [f"{row['prompt']}\n\nAssistant: " for row in ifeval_rows]
        )
        xstest_responses = generator.generate(
            [f"{row['prompt']}\n\nAssistant: " for row in xstest_rows]
        )
        jbb_responses = generator.generate([f"{p}\n\nAssistant: " for p in jbb_prompts])

        backend = HFBackend(
            model=model,
            tokenizer=HFTokenizerAdapter(
                raw_tokenizer, manifest["student_model"], manifest["student_revision"]
            ),
            model_id=manifest["student_model"],
            model_revision=manifest["student_revision"],
            device="cuda",
            dtype=torch.float32,
            quantization=Quantization.NONE,
            gradient_checkpointing=False,
            attn_implementation="eager",
            lora=True,
            adapter_provenance={"adapter_digest": candidate["adapter_digest"]},
        )
        jsonnav = score_jsonnav_tasks(
            backend=backend,
            task_ids=by_id["jsonnav_utility"]["task_ids"],
            settings=JsonNavSettings(),
        )

        generations[name] = {
            "update": candidate["update"],
            "adapter_digest": candidate["adapter_digest"],
            "ifeval": ifeval_responses,
            "xstest": xstest_responses,
            "jbb": jbb_responses,
            "jsonnav": jsonnav,
        }
        print(
            f"{name}: generated; jsonnav {jsonnav['solved']}/{jsonnav['tasks']} "
            f"= {jsonnav['success_rate']:.3f}"
        )

        del backend, generator, model, base
        gc.collect()
        torch.cuda.empty_cache()

    # Pass two: the judge, now that no candidate is resident.
    judge_entry = next(e for e in registry["endpoints"] if e["id"] == "jbb_behaviors")["evaluator"]
    judge = HarmfulJudge(judge_entry["model"], judge_entry["model_revision"]).load()

    results: dict[str, dict[str, Any]] = {}
    for name, payload in generations.items():
        satisfied = total = 0
        for row, response in zip(ifeval_rows, payload["ifeval"], strict=True):
            scored = evaluate_ifeval_response(response, row["instruction_id_list"], row["kwargs"])
            satisfied += scored["strict_instructions_satisfied"]
            total += scored["instructions_evaluated"]

        xs = summarize_xstest(
            label_responses([str(row["type"]) for row in xstest_rows], payload["xstest"])
        )
        harmful = summarize_harmful(
            [
                {"category": category, "label": judge.classify(prompt, response)[0]}
                for prompt, response, category in zip(
                    jbb_prompts, payload["jbb"], jbb_categories, strict=True
                )
            ]
        )

        results[name] = {
            "update": payload["update"],
            "adapter_digest": payload["adapter_digest"],
            "metrics": {
                "instruction_following": (satisfied / total) if total else None,
                "over_refusal": xs["over_refusal_rate"],
                "appropriate_refusal": xs["appropriate_refusal_rate"],
                "harmful_compliance": harmful["harmful_compliance_rate"],
                "retained_tool_utility": payload["jsonnav"]["success_rate"],
            },
            "jsonnav": {
                key: value for key, value in payload["jsonnav"].items() if key != "records"
            },
        }
        print(f"{name}: {json.dumps(results[name]['metrics'], sort_keys=True)}")

    decision = select_starting_checkpoint(
        [
            {"id": name, "metrics": payload["metrics"]}
            for name, payload in sorted(results.items(), key=lambda kv: kv[1]["update"])
        ],
        alignment_endpoints=GATE["alignment_endpoints"],
        utility_endpoint=GATE["utility_endpoint"],
    )

    record = {
        "schema_version": 1,
        "lineage": "primary: Qwen3-0.6B continued on HH-RLHF",
        "final_suite_digest": final["manifest_digest"],
        "selection_suite_digest": selection["manifest_digest"],
        "candidate_results": results,
        "decision": decision,
        "judge": judge.identity(),
        "gpu_seconds": round(time.time() - started, 1),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
        "split_used": "eval only; every final-test task id was withheld",
        "final_test_scored": False,
    }
    (out / "start-selection.json").write_text(
        json.dumps(record, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out / "jsonnav-records.json").write_text(
        json.dumps(
            {name: payload["jsonnav"]["records"] for name, payload in generations.items()},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(f"\ndecision: {json.dumps(decision['selected'])} ({decision['status']})")
    for entry in decision["candidates"]:
        print(f"  {entry['id']}: {entry['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
