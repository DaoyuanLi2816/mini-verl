"""Measure rollout decode throughput under different student configurations.

The 16 GB recipe should be sized from measurements, not from folklore about what
QLoRA costs.  This script times single-sequence decoding for the pinned Qwen3
student under bf16-LoRA and NF4-QLoRA, with and without torch's deterministic
algorithm mode, and prints a JSON table.

    python scripts/gpu_probe_throughput.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

STUDENT = "Qwen/Qwen3-0.6B"
STUDENT_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"


def main() -> int:
    """Time decode throughput across configurations."""
    import torch

    from miniverl.config.models import LoRAConfig, Precision, Quantization, StudentModelConfig
    from miniverl.models.hf import HFBackend
    from miniverl.models.tokenizers import HFTokenizerAdapter
    from miniverl.utils import gpu

    if not gpu.cuda_available():
        print(json.dumps({"status": "not_run", "reason": "no CUDA device"}, indent=2))
        return 2

    tokenizer = HFTokenizerAdapter.load(STUDENT, revision=STUDENT_REVISION)
    prompt = (
        "<|im_start|>system\nYou are a tool-using assistant.<|im_end|>\n"
        "<|im_start|>user\nCompute 12 * (3 + 4) and report the value.<|im_end|>\n"
        "<|im_start|>assistant\n"
    )
    prefix = tokenizer.encode(prompt)
    rows = []

    for quantization, dtype, deterministic in (
        (Quantization.NF4, Precision.BFLOAT16, True),
        (Quantization.NF4, Precision.BFLOAT16, False),
        (Quantization.NONE, Precision.BFLOAT16, True),
        (Quantization.NONE, Precision.BFLOAT16, False),
    ):
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic
        try:
            torch.use_deterministic_algorithms(deterministic, warn_only=True)
        except RuntimeError:  # pragma: no cover
            pass

        spec = StudentModelConfig(
            model_id=STUDENT,
            revision=STUDENT_REVISION,
            dtype=dtype,
            quantization=quantization,
            gradient_checkpointing=False,
            attn_implementation="sdpa",
            lora=LoRAConfig(enabled=True, r=16, alpha=32),
        )
        gpu.empty_cache()
        gpu.reset_peak_stats()
        load_start = time.perf_counter()
        backend = HFBackend.load(spec, device="cuda", tokenizer=tokenizer, trainable=True)
        load_seconds = time.perf_counter() - load_start

        backend.generate(prefix, max_new_tokens=4, temperature=0.0)  # warm up
        gpu.reset_peak_stats()
        started = time.perf_counter()
        output = backend.generate(prefix, max_new_tokens=64, temperature=1.0, seed=1)
        elapsed = time.perf_counter() - started
        snapshot = gpu.snapshot()
        rows.append(
            {
                "quantization": quantization.value,
                "dtype": dtype.value,
                "deterministic_algorithms": deterministic,
                "load_seconds": round(load_seconds, 2),
                "tokens": len(output.token_ids),
                "seconds": round(elapsed, 3),
                "tokens_per_second": round(len(output.token_ids) / elapsed, 2),
                "peak_allocated_gib": round(snapshot.peak_allocated_gib, 3),
                "peak_reserved_gib": round(snapshot.peak_reserved_gib, 3),
                "trainable_params": backend.capabilities.num_trainable_parameters,
            }
        )
        del backend
        gpu.empty_cache()

    print(json.dumps({"status": "measured", "prefix_tokens": len(prefix), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
