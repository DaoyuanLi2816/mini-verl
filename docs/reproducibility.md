# Reproducibility

This document describes exactly what miniVERL controls, what it records, what
it does not control, and how to reproduce a published run.

Source files behind this document:

- `src/miniverl/utils/seeding.py` - seeding, RNG capture and restore
- `src/miniverl/utils/env.py` - environment capture for manifests
- `src/miniverl/utils/runs.py` - run directory layout
- `src/miniverl/training/checkpoint.py` - pickle-free checkpoints and resume
- `src/miniverl/selection/selectors.py` - deterministic position sub-sampling
- `src/miniverl/models/sampling.py` - the shared generation loop

## Every source of randomness, and how it is controlled

| source | where | how it is controlled |
| --- | --- | --- |
| Task generation | `CalculatorEnvironment.generate_task` and the other environments | `random.Random(f"calculator:{seed}:{difficulty}:{index}")`, a string-seeded generator per task; the seed is `environment.split_seed` |
| Split membership | `make_splits` in `src/miniverl/environments/base.py` | splits are built in the fixed order train, eval, test from index offsets 0, 1000000, 2000000, and any prompt already produced by an earlier split is skipped, so a task cannot leak from train into eval |
| Training task order | `OPDTrainer._build_task_order` | `random.Random(run.seed ^ 0x5EED).shuffle(order)`; the position in that order is `task_cursor`, which is checkpointed |
| Training rollout sampling | `OPDTrainer._collect` | per-trajectory seed `run.seed + global_step * 1013 + offset` |
| Evaluation rollout sampling | `OPDTrainer.evaluate` | per-task seed `eval.seed + offset` |
| Per-turn generation | `RolloutRunner.rollout` | each turn calls the backend with `seed * 1_000_003 + turn_id`, so turns inside one trajectory do not share a stream |
| Token sampling | `sample_from_logits` in `src/miniverl/models/sampling.py` | `temperature <= 0.0` is exact argmax; otherwise a CPU `torch.Generator` seeded per `generate()` call |
| Teacher-position sub-sampling | `select_positions` | `random.Random(derive_seed(run_seed, trajectory_id))` |
| Toy model initialization | `ToyBackend.__init__` | a CPU generator seeded from `run.seed` (student) or `models.teacher.toy_teacher_seed` (teacher) draws one integer, which seeds a forked global torch RNG so initialization cannot disturb the ambient stream |
| Toy teacher fitting | `OPDTrainer._prepare_toy_teacher` | wrapped in `torch.random.fork_rng(devices=[])`, because `fit_toy_model` calls `torch.manual_seed` and would otherwise clobber the stream the rollouts and a restored checkpoint depend on |
| LoRA dropout | `models.student.lora.dropout` | defaults to `0.0`, so the default configuration has no dropout randomness |
| Global RNG state | `seed_everything(config.run.seed, deterministic=config.run.deterministic)`, called once in `OPDTrainer.from_config` | seeds `random`, NumPy (`seed % 2**32`), torch CPU and all CUDA devices |

`seed_everything` with `deterministic=True` (the default, from
`run.deterministic`) additionally:

- sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` if it is not already set
- sets `torch.backends.cudnn.deterministic = True`
- sets `torch.backends.cudnn.benchmark = False`
- calls `torch.use_deterministic_algorithms(True, warn_only=True)`

`warn_only=True` is a deliberate choice, stated in the docstring: a kernel
without a deterministic implementation degrades to a warning rather than
crashing a long run. The `use_deterministic_algorithms` call is additionally
wrapped in `contextlib.suppress(RuntimeError, AttributeError)`, so on a torch
build that rejects it the request is dropped rather than failing the run. Both
consequences are in
[What is not deterministic](#what-is-not-deterministic-and-why).

## The `derive_seed` scheme

```python
def derive_seed(*parts: object) -> int:
    import hashlib

    blob = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:8], "big") >> 1
```

Parts are stringified, joined with `|`, hashed with SHA-256, and the first
eight bytes are read as a big-endian integer shifted right by one bit. The
shift yields a non-negative 63-bit value, which is inside the range every
`random.Random` and `torch.Generator` accepts on every platform.

The call site that matters is position sub-sampling:

```python
rng = random.Random(derive_seed(run_seed, trajectory.trajectory_id))
```

### Why a salted hash would break it

Python's built-in `hash()` for `str` and `bytes` is randomized per process by
`PYTHONHASHSEED` unless that variable is pinned. Seeding a sampler with
`hash(trajectory_id)` would mean the same trajectory selects different teacher
positions in a different process, on a different OS, or on a rerun of the same
command. The selection would be unreproducible, the teacher-target cache would
be keyed to positions that cannot be recomputed, and two runs with the same
`run.seed` would not be comparable.

SHA-256 has no such salt. Measured on this build:

```console
$ PYTHONHASHSEED=1 python -c "from miniverl.utils.seeding import derive_seed; \
print(derive_seed(1234, 'calc-train-7:v0:s1234'), hash('calc-train-7'))"
7740022099863669268 4211654694876181374

$ PYTHONHASHSEED=2 python -c "from miniverl.utils.seeding import derive_seed; \
print(derive_seed(1234, 'calc-train-7:v0:s1234'), hash('calc-train-7'))"
7740022099863669268 -7731668170901394408
```

`derive_seed` is identical across both; `hash()` is not.

## Why sampling always happens on the CPU

`sample_from_logits` filters on whatever device the logits live on, then moves
the probability vector to the host and draws there:

```python
probs = torch.softmax(scaled, dim=-1).to("cpu")
return int(torch.multinomial(probs, num_samples=1, generator=generator).item())
```

Two reasons, both in the docstring:

1. A CUDA tensor cannot be drawn with a CPU generator at all. Keeping the
   generator on the CPU and the draw on the CPU makes the seeding contract
   uniform across backends.
2. A given seed produces the *same* token sequence on CPU and on GPU. Without
   this, a toy CPU run and a GPU run of the same recipe would diverge at the
   first sampled token even with identical seeds.

The cost is one device-to-host copy per generated token. Measured on this
machine (RTX 4080, torch 2.13.0+cu130), a 151936-float device-to-host copy
takes 0.11 ms, while a cached single-token decode step costs 30.9 ms. The copy
is therefore a small fraction of a decode step, and decoding is kernel-launch
bound rather than compute bound: a 14-token prefill costs 37.0 ms against
30.9 ms for a 1-token cached step.

Greedy evaluation (`eval.temperature: 0.0`) takes the `torch.argmax` branch and
never reaches the sampler at all.

## What the manifest records

`OPDTrainer.build_manifest()` writes `manifest.json` in the run directory.
Measured on a real toy run, its top-level keys are:

```
created_at, deterministic, environment, git_commit, gpu, measurement_status,
memory, miniverl_version, mode, models, objective, os, os_release, packages,
platform, policy_version, python_version, run_id, run_name, seed
```

The load-bearing sub-objects:

- `environment` - environment `name` and construction `params`, plus
  `difficulty`, `split_seed`, and the realized `split_sizes` per split
- `models` - `backend`, `device`, and for student and teacher the `model_id`,
  `revision`, `tokenizer_revision`, `quantization`, `precision` and full
  backend capabilities; plus `tokenizer_fingerprint` and
  `tokenizer_vocab_size`
- `objective` - `loss_mode`, `divergence`, `temperature`,
  `scale_by_temperature_squared`, the effective `top_k`, `jsd_beta`,
  `ce_weight`, `selector`, `selection_ratio`
- `memory` - the resolved `strategy`, the `reason` string explaining why,
  `projection_chunk_size`, `oom_retries_used` and `chunk_size_history`
- `packages` and `gpu` - see below
- `measurement_status` - `cpu_metrics`, `cuda_metrics`, and
  `simulated_results: "none"`

`environment.json` is written alongside it by `collect_environment()`:

```
cpu_count, git_commit, gpu, machine, os, os_release, packages, platform,
processor_family, python_implementation, python_version, tracked_env_vars
```

`packages` covers `miniverl`, `torch`, `transformers`, `peft`, `accelerate`,
`bitsandbytes`, `safetensors`, `pydantic`, `typer`, `rich`, `numpy`, with
`null` for anything not installed. `tracked_env_vars` records only the
allowlist in `TRACKED_ENV_VARS`, and only the entries that are actually set:
`CUBLAS_WORKSPACE_CONFIG`, `PYTORCH_CUDA_ALLOC_CONF`, `CUDA_VISIBLE_DEVICES`,
`OMP_NUM_THREADS`, `TOKENIZERS_PARALLELISM`.

`git_commit` is resolved by reading `.git` directly rather than by running a
subprocess, and is `null` outside a checkout (for example when miniVERL is
installed from a wheel). That is recorded honestly rather than faked.

Deliberately excluded, because a run directory is meant to be shareable:
hostname, username, home directory, absolute paths outside the run, and every
environment variable outside the allowlist.

Two caveats worth knowing before you quote the manifest:

- `manifest["measurement_status"]["cuda_metrics"]` is `"measured"` whenever the
  *machine* has a visible CUDA device, not whenever the *run* used one. A CPU
  run on a GPU box reports `"measured"` here. The authoritative per-run signals
  are `manifest["models"]["device"]` and, in an exported result, the
  `cuda_available` flag computed by `ReportData.throughput()`, which requires
  both a `cuda` device string and an available GPU.
- `gpu.driver_version` is read through `pynvml`, which is optional. It is
  `null` when `pynvml` is not installed.

Also written, alongside the manifest: `config.original.yaml` and
`config.resolved.yaml`. Neither is a byte copy of your recipe file. Both are
`RunConfig.to_yaml()` output, which writes every field including defaults, so
a recipe that omitted a key will show that key's default value here. The
resolved file additionally has `run.run_id`, `memory.strategy`,
`loss.chunk_size` and `models.device` replaced with the values actually used,
and for the Hugging Face backend `loss.top_k` clamped to the student vocabulary
size.

## Re-verifying the pinned model revisions

`recipes/qwen_consumer_gpu_calc.yaml` pins both sides of the pair by commit
SHA. Re-check the pins against the Hugging Face API before trusting them.
Every command below was executed on 2026-07-27 and the output shown is what it
returned.

Revision exists (200 means the SHA resolves in that repository):

```console
$ curl -s -o /dev/null -w "%{http_code}\n" \
  https://huggingface.co/api/models/Qwen/Qwen3-0.6B/revision/c1899de289a04d12100db370d81485cdf75e47ca
200
$ curl -s -o /dev/null -w "%{http_code}\n" \
  https://huggingface.co/api/models/Qwen/Qwen3-1.7B/revision/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
200
```

Negative control, so you know the endpoint really validates the revision rather
than ignoring it:

```console
$ curl -s -o /dev/null -w "%{http_code}\n" \
  https://huggingface.co/api/models/Qwen/Qwen3-0.6B/revision/0000000000000000000000000000000000000000
404
```

`tokenizer.json` is byte-identical across the pair. The Hub exposes the file's
SHA-256 as `X-Linked-ETag` on the resolve endpoint:

```console
$ curl -sI https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/tokenizer.json \
  | grep -i x-linked-etag
X-Linked-ETag: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
$ curl -sI https://huggingface.co/Qwen/Qwen3-1.7B/resolve/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e/tokenizer.json \
  | grep -i x-linked-etag
X-Linked-ETag: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4"
```

Model configuration at those revisions (`-L` is required; the resolve endpoint
redirects to a CDN):

```console
$ curl -sL https://huggingface.co/Qwen/Qwen3-0.6B/resolve/c1899de289a04d12100db370d81485cdf75e47ca/config.json \
  | python -c "import sys,json;d=json.load(sys.stdin);print({k:d.get(k) for k in ('model_type','vocab_size','num_hidden_layers','hidden_size','tie_word_embeddings')})"
{'model_type': 'qwen3', 'vocab_size': 151936, 'num_hidden_layers': 28, 'hidden_size': 1024, 'tie_word_embeddings': True}
```

The 1.7B teacher at its pinned revision returns the same values except
`hidden_size: 2048`. Both repositories report `apache-2.0`:

```console
$ curl -s https://huggingface.co/api/models/Qwen/Qwen3-0.6B \
  | python -c "import sys,json;print(json.load(sys.stdin)['cardData']['license'])"
apache-2.0
```

Finally, verify the tokenizers behaviourally, which is what miniVERL actually
checks. `HFTokenizerAdapter` computes a fingerprint from the tokenizer class,
`len(tokenizer)`, the EOS and PAD ids, the added special tokens, and the token
ids produced for a fixed probe string:

```console
$ python -c "
from miniverl.models.tokenizers import HFTokenizerAdapter
a = HFTokenizerAdapter.load('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca')
b = HFTokenizerAdapter.load('Qwen/Qwen3-1.7B', revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e')
print('vocab_size', a.vocab_size, b.vocab_size)
print('identical', a.fingerprint == b.fingerprint)
print(a.fingerprint)
"
vocab_size 151669 151669
identical True
f2f5e826dddc3ff1e2481111075f2ed6eced4e553168222d67650931d25be035
```

Note the two vocabulary numbers. `config.json` declares `vocab_size: 151936`
(the padded embedding matrix), while `len(tokenizer)` is 151669. miniVERL
records the tokenizer number as `models.tokenizer_vocab_size` in the manifest
and uses the model's own `vocab_size` for the LM-head projection.

The fingerprint value above was measured with transformers 5.14.1. It depends
on the tokenizer class name and the added-token list, so a transformers upgrade
can change it even when `tokenizer.json` does not. Compare fingerprints between
student and teacher within one environment; do not compare a fingerprint across
library versions.

## What is not deterministic, and why

- **Non-deterministic CUDA kernels.**
  `torch.use_deterministic_algorithms(True, warn_only=True)` warns instead of
  raising when an operation has no deterministic implementation, and the call
  itself is suppressed on builds that reject it. A GPU run can therefore
  contain non-deterministic reductions. Keep `run.deterministic: true` (the
  default) and read the warnings; miniVERL does not suppress those.
- **Cross-hardware and cross-version floating point.** Different GPU
  architectures, driver versions, CUDA versions, cuBLAS versions and
  transformers versions produce different last-bit results. The manifest
  records all of these so a mismatch is visible; it cannot make them agree.
  `CUBLAS_WORKSPACE_CONFIG` is recorded in `tracked_env_vars` for the same
  reason.
- **Reduced precision.** The 16 GB recipe uses bf16 activations and NF4
  weights. Both are lossy relative to fp32, and NF4 dequantization ordering is
  a bitsandbytes implementation detail.
- **`cache.dtype: float16`.** The shipped GPU recipe stores teacher
  log-probabilities in fp16, which the config documents as costing roughly
  1e-3 relative precision. Set `cache.dtype: float32` to round-trip the
  targets exactly.
- **Wall clock and throughput.** `seconds`,
  `train_selected_tokens_per_second`, `rollout_tokens_per_second` and every
  memory counter are measurements of one machine at one moment. They are
  reported, never asserted.
- **`git_commit`** is `null` outside a git checkout, and `gpu.driver_version`
  is `null` without `pynvml`.
- **Resume equality is numerical, not bitwise.** See the next section.

## Exact-resume guarantees

A checkpoint directory holds exactly three files, and
`tests/integration/test_resume_and_swap.py::test_checkpoint_files_are_pickle_free`
asserts that list is exact:

```
checkpoints/step-000012/
  adapter.safetensors      # trainable weights only
  optimizer.safetensors    # optimizer moment tensors
  state.json               # everything that is not a tensor
```

Tensors go through safetensors and structure goes through JSON. `torch.save`
is never used, so loading a checkpoint cannot execute code. The same test
asserts neither safetensors file begins with a pickle protocol opcode.

`state.json` carries, measured from a real run:

```
config_digest, cycle, global_step, metrics, miniverl_version,
optimizer_param_groups, optimizer_scalars, optimizer_state_keys,
policy_version, rng, scaler, scheduler, schema_version, task_cursor
```

`rng` contains `python_state`, `torch_state`, `cuda_states` and `numpy_state`.
`capture_rng` serializes Python's `random.getstate()` as JSON (keeping the
snapshot pickle-free), base64-encodes the torch CPU generator state and every
CUDA generator state, and captures NumPy's legacy 5-tuple state explicitly so
the layout cannot silently change to the dict form. `restore_rng` restores the
CUDA states only when the device count matches.

### The test that proves it

`tests/integration/test_resume_and_swap.py::test_uninterrupted_and_resumed_training_agree_exactly`
runs four cycles in one process, then runs the same schedule with an
interruption after two cycles, a checkpoint, and a fresh trainer that resumes.
It asserts the final `global_step`, `policy_version` and `task_cursor` match
exactly, and that the largest absolute difference over all trainable
parameters is at most `1e-6`.

That is a numerical tolerance, not bitwise equality. The module docstring of
`src/miniverl/training/checkpoint.py` says "match bit-for-bit" and names
`tests/integration/test_resume.py`; the test that exists is
`test_resume_and_swap.py` and its tolerance is `atol=1e-6`. Quote the
tolerance, not the docstring.

Three further tests in the same file:

- `test_checkpoint_round_trip_restores_optimizer_and_schedule` asserts the
  optimizer moments are non-empty before saving and non-empty after loading,
  and that the restored LR schedule returns the same learning rate at the
  restored step
- `test_resuming_a_checkpoint_from_a_different_config_is_refused` asserts a
  `ConfigError` when the config digest differs
- `test_swap_and_resident_produce_the_same_update` asserts the two memory
  strategies reach the same parameters within `1e-6`

### The config digest, and the one trap in it

`OPDTrainer._config_digest()` is `sha256(self.config.to_yaml())`, and
`load_from_checkpoint` refuses to resume when the stored digest differs:

```
error the checkpoint was written by a different configuration
hint  resume with the run's config.resolved.yaml, not a modified recipe
```

The digest is computed over the config object the trainer was constructed with,
which is `config.original.yaml`, not `config.resolved.yaml`. Because
`config.resolved.yaml` has `run.run_id` filled in (and `memory.strategy`,
`loss.chunk_size` and `models.device` resolved), it usually hashes differently.
Measured on a real run whose only original-versus-resolved difference was
`run_id: null` becoming `run_id: docs-check`, resuming from
`config.resolved.yaml` was refused and resuming from `config.original.yaml`
succeeded. **Resume from `config.original.yaml`.** The hint in the error
message is misleading whenever the recipe left `run.run_id` unset, which is the
normal case.

### Resuming

There is no `miniverl resume` subcommand. Resume is a library API:

```python
from pathlib import Path

from miniverl.config import RunConfig
from miniverl.trainer import OPDTrainer
from miniverl.training.checkpoint import latest_checkpoint

run = Path("runs/<run-id>")
config = RunConfig.from_yaml(run / "config.original.yaml")

trainer = OPDTrainer.from_config(config, output_dir=run.parent, run_id="<run-id>-resumed")
state = trainer.load_from_checkpoint(latest_checkpoint(run / "checkpoints"))
print("resuming from step", state.global_step, "cycle", state.cycle)
trainer.train()
trainer.close()
```

Pass a *new* `run_id`, as above, unless you intend to write into the original
run directory: `RunPaths.create` uses `exist_ok=True` and will append to the
existing JSONL logs.

`train()` on a resumed trainer starts at `state.cycle + 1`, and emits a
`resumed` event noting that it is skipping the baseline evaluation and the SFT
cold start. Both are correct to skip: a "baseline" measured after training is
not a baseline, and repeating the cold start would double it.

Checkpoints are written by `train.save_every_cycles` (`0` disables periodic
checkpoints) and always at the end of a run as `checkpoints/final`.

## Reproducing a published run from its run directory

A run directory is self-describing. Given `runs/<run-id>`:

**1. Read what was run.**

```bash
python -c "import json;m=json.load(open('runs/<run-id>/manifest.json'));\
print(m['miniverl_version'], m['git_commit'], m['mode'], m['seed']);\
print(m['models']['student']['model_id'], m['models']['student']['revision']);\
print(m['models']['teacher']['model_id'], m['models']['teacher']['revision'])"
```

**2. Match the software.** `environment.json` lists the exact versions of
`torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes`, `numpy` and
`safetensors`, plus the Python version and the tracked environment variables.
Install those versions, and check out `git_commit` of this repository.

**3. Re-run the training.** Use the resolved config, which has every `auto`
already frozen, and give it a new run id so it does not write into the original
directory:

```bash
miniverl train runs/<run-id>/config.resolved.yaml \
  --output runs --run-id <run-id>-repro
```

`config.resolved.yaml` sets `run.run_id`, and `make_run_id` prefers an explicit
`--run-id` over it; without `--run-id` the reproduction would reuse the
original run id and write into the original directory.

**4. Or re-evaluate only.** To reproduce the reported numbers from the
weights that were published, without retraining:

```bash
miniverl eval --run runs/<run-id>
```

`evaluate_run` rebuilds the trainer from `config.resolved.yaml` (not the
original recipe, so any `auto` decision stays frozen), restores the latest
checkpoint with `include_rng=False`, evaluates deterministically, and writes
`eval.<tag>.json` next to the run. `--split`, `--tasks`, `--checkpoint`,
`--out` and `--tag` are available. It refuses to run against a directory with
no `config.resolved.yaml`, on the grounds that only runs created by
`miniverl train` can be re-evaluated.

**5. Compare.** The comparison that matters is the evaluation payload:
`tasks`, `success_rate`, `avg_turns`, `avg_tool_calls`,
`invalid_tool_call_rate`, `generated_tokens_per_task` and
`success_by_difficulty`. Wall clock and memory will differ on different
hardware; see [What is not deterministic](#what-is-not-deterministic-and-why)
for which differences are expected.

## See also

- `docs/benchmarking.md` - matched-budget comparisons and metric definitions
- `docs/troubleshooting.md` - resume refusals, cache errors, tokenizer
  mismatches
- `docs/limitations.md` - what miniVERL does not support
