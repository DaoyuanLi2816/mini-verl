# Troubleshooting

Every miniVERL error carries a `message` and, where an action exists, a `hint`.
The CLI prints them on two lines:

```console
$ miniverl report recipes
error recipes does not look like a miniVERL run (no manifest.json)
hint  point at the run directory itself, not its parent
```

Start with the diagnostic command. It imports no heavy dependency and works
from a bare `pip install miniverl`:

```bash
miniverl doctor
miniverl doctor --json
```

`run_doctor` checks the required dependencies (`typer`, `rich`, `pydantic`,
`yaml`, `jinja2`, `platformdirs`, `safetensors`), the optional ones (`torch`,
`transformers`, `peft`, `accelerate`, `numpy`, `bitsandbytes`), the Python
version, the registered environments, the CUDA device, bf16 support, and
whether the output directory is writable. It ends with four verdicts:
`core_commands`, `cpu_training`, `gpu_training`, `qlora_4bit`.

Contents:

- [Missing extras](#missing-extras)
- [No CUDA](#no-cuda)
- [bitsandbytes on Windows and WSL](#bitsandbytes-on-windows-and-wsl)
- [CUDA out of memory](#cuda-out-of-memory)
- [Tokenizer mismatch](#tokenizer-mismatch)
- [Stale or corrupt teacher cache](#stale-or-corrupt-teacher-cache)
- [Gated or moved model revisions](#gated-or-moved-model-revisions)
- [Slow decoding](#slow-decoding)
- [Config validation errors](#config-validation-errors)
- [Report or eval on a non-run directory](#report-or-eval-on-a-non-run-directory)
- [Resume refused: config digest mismatch](#resume-refused-config-digest-mismatch)

## Missing extras

**Symptom**

```
error miniverl train requires the optional dependency 'torch', which is not installed.
hint  pip install "miniverl[train]"
```

**Cause.** The core install has seven light dependencies and no torch.
`miniverl --help`, `doctor`, `validate`, `inspect`, `report`, `cache` and
`schema` are designed to work without it. `train`, `demo`, `eval` and
`benchmark` call `_require_training_stack()` before any heavy import, which
raises `MissingDependencyError` for the first of `torch`, `transformers`,
`peft` that is missing. Any stray `ModuleNotFoundError` that escapes is also
converted into the same message by `_fail`.

**Fix.** Install the extra the message names:

```bash
pip install "miniverl[train]"          # torch, transformers, peft, accelerate, numpy
pip install "miniverl[train,cuda]"     # the above plus bitsandbytes for 4-bit QLoRA
```

Note that the hint contains square brackets, which Rich would otherwise strip
as markup. The CLI escapes every dynamic string, so what you see printed is
what you should paste.

## No CUDA

**Symptom, from `miniverl doctor`**

```
cuda  missing  torch.cuda.is_available() is False; CPU-only paths still work
```

with the suggestion
`install a CUDA build of torch, e.g. pip install torch --index-url https://download.pytorch.org/whl/cu130`.

**Symptom, from a recipe that demands a GPU**

```
error models.device is 'cuda' but no CUDA device is visible to torch
hint  run `miniverl doctor` to see what torch reports, or set models.device: auto
```

**Cause.** `resolve_device` in `src/miniverl/models/factory.py` refuses an
explicit `models.device: cuda` when `torch.cuda.is_available()` returns false.
The usual reason is a CPU-only torch wheel; `miniverl doctor --json` reports
`torch_cuda_version` so you can tell a CPU build (`null`) from a driver problem.

**Fix.** Either install a CUDA build of torch, or run the CPU paths:

- set `models.device: auto` in the recipe, which resolves to `cpu` when no
  device is visible
- run the toy recipe, which needs no GPU and no network:
  `miniverl train recipes/toy_cpu.yaml`
- run the embedded pipeline: `miniverl demo --fast`

Everything except the Hugging Face backend works on CPU. `memory.strategy:
auto` resolves to `resident` on CPU with the recorded reason
`auto -> resident: no CUDA device, host memory is not partitioned`.

## bitsandbytes on Windows and WSL

**Symptom**

```
error nf4 weight quantization requires the optional dependency 'bitsandbytes', which is not installed.
hint  pip install "miniverl[cuda]"
```

**Cause.** `_quantization_config` in `src/miniverl/models/hf.py` raises before
building a `BitsAndBytesConfig` whenever `models.student.quantization` or
`models.teacher.quantization` is `nf4` or `int8` and bitsandbytes is absent.
The `cuda` extra declares `bitsandbytes>=0.43; platform_system != 'Darwin'`,
so it is not installed on macOS at all.

**Fix.**

```bash
pip install "miniverl[cuda]"
miniverl doctor --json    # check "qlora_4bit": true in the verdict block
```

Windows works. Measured on this machine (Windows 11 Pro 10.0.22631, RTX 4080,
driver 596.49, CPython 3.12.13), bitsandbytes 0.50.0 imports cleanly against
torch 2.13.0+cu130 and `miniverl doctor` reports `"qlora_4bit": true`. WSL2 is
not required for the QLoRA path on this configuration.

If bitsandbytes cannot be installed or cannot import on your platform, run the
same recipe unquantized:

```yaml
models:
  student:
    quantization: none
```

That raises student VRAM. A measured comparison on this machine: NF4 plus
bf16 decodes at 11.19 tok/s with 0.862 GiB peak allocated, while bf16 LoRA
decodes at 12.84 tok/s with 1.170 GiB peak allocated, both single-sequence with
deterministic algorithms enabled. On a 0.6B student the NF4 saving is small;
it is the path that scales to a larger teacher.

One further constraint, enforced in `OPDTrainer.from_config`:

```
error memory.strategy=swap cannot be used with a quantized model: bitsandbytes
      4-bit/8-bit parameters are pinned to the device they were quantized on and
      cannot be moved to host memory and back.
hint  use memory.strategy: resident (a 0.6B QLoRA student plus a bf16 1.7B
      teacher fits in 16 GB), or set both quantization fields to 'none' if you
      really need swap
```

With `memory.strategy: auto` and any quantization, miniVERL resolves to
`resident` automatically and records the reason
`auto -> resident: a quantized model cannot be moved off the accelerator, so
swap is unavailable`.

## CUDA out of memory

**Symptom**

```
error CUDA ran out of memory and the 3 equivalence-preserving retries were
      exhausted (projection chunk size reached 16).
hint  reduce rollout.max_total_tokens, reduce train.gradient_accumulation_steps,
      lower loss.top_k, switch models.student.quantization to nf4, enable
      models.student.gradient_checkpointing, or set memory.strategy: swap.
```

**Cause.** `run_with_oom_retry` in `src/miniverl/training/memory.py` catches
CUDA OOM during an update, clears gradients, empties the allocator cache and
halves `loss.chunk_size`, up to `memory.oom_retries` times and never below
`memory.min_chunk_size`. Halving the projection chunk is mathematically
neutral: the loss and gradient are identical for any chunk size, which
`tests/unit/test_chunked_equivalence.py` asserts for chunk sizes 1, 5 and 37
across all three divergences. Nothing else is changed behind your back, so
when the retries run out the run fails instead of quietly shrinking your
sequence length or your batch.

Each retry emits an `oom_chunk_retry` event with the note
`objective unchanged; only the projection chunk size shrank`, and the surviving
chunk size plus `oom_retries_used` and `chunk_size_history` are recorded in
`manifest["memory"]`.

**Fix.** Apply the config keys the hint names, in roughly this order of
cost-effectiveness:

```yaml
loss:
  chunk_size: 128              # fewer positions projected through the LM head at once
  top_k: 32                    # smaller teacher target payload
rollout:
  max_total_tokens: 512        # shorter trajectories
train:
  gradient_accumulation_steps: 3   # fewer trajectories held per optimizer step
models:
  student:
    quantization: nf4          # requires the cuda extra and lora.enabled: true
    gradient_checkpointing: true
memory:
  strategy: swap               # unquantized pairs only
  min_chunk_size: 8
  oom_retries: 3
```

Reducing `train.gradient_accumulation_steps` below
`train.rollouts_per_cycle` produces more than one optimizer step per rollout
batch, which makes steps after the first only approximately on-policy.
`miniverl validate` warns about exactly that, and the trainer emits an
`opd_multi_update_warning` event.

For strategy details and the measured peak-VRAM numbers, see `docs/memory.md`.
A one-cycle GPU smoke test on this machine reached 4.251 GiB peak allocated and
4.762 GiB peak reserved with the resident strategy, projection chunk 256, and
zero OOM retries.

Related error, raised before any allocation:

```
error loss.mode=exact_full_vocab with memory.strategy=swap must persist a
      [positions, <vocab_size>] teacher tensor, which exceeds the
      loss.exact_max_vocab=8192 guard rail.
hint  use loss.mode=bucketed_topk_tail for large vocabularies, or set
      memory.strategy=resident so the exact teacher distribution can be rebuilt
      one chunk at a time, or set loss.allow_large_exact=true if you really mean it
```

## Tokenizer mismatch

**Symptom, at model load**

```
error the student tokenizer (Qwen/Qwen3-0.6B) and the teacher tokenizer
      (meta-llama/Llama-3.2-1B) tokenize differently
hint  miniVERL v0.1 only supports same-tokenizer distillation. Pick a teacher
      from the same model family, e.g. Qwen/Qwen3-0.6B with Qwen/Qwen3-1.7B.
      Cross-tokenizer distillation is a roadmap item (docs/limitations.md).
```

**Symptom, during alignment**

```
error student and teacher tokenizers differ (<12 hex chars>... vs <12 hex chars>...);
      miniVERL v0.1 only supports same-tokenizer distillation
```

**Cause.** `build_tokenizer` loads one tokenizer for both sides. When the
teacher declares its own `tokenizer_id`, that tokenizer is loaded and
fingerprint-compared rather than trusted. The fingerprint is behavioural: it
hashes the tokenizer class name, `len(tokenizer)`, the EOS and PAD ids, the
added special tokens, and the token ids produced for a fixed probe string that
includes the chat template markers and the tool-call tags. Two tokenizers with
the same declared vocabulary size but different merge behaviour therefore do
not match.

**Fix.** Use a teacher from the same family as the student. The pinned pair in
`recipes/qwen_consumer_gpu_calc.yaml` is verified byte-identical on
`tokenizer.json`:

```yaml
models:
  student:
    model_id: Qwen/Qwen3-0.6B
    revision: c1899de289a04d12100db370d81485cdf75e47ca
    tokenizer_revision: c1899de289a04d12100db370d81485cdf75e47ca
  teacher:
    model_id: Qwen/Qwen3-1.7B
    revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
    tokenizer_revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
```

Check a candidate pair before starting a run:

```bash
python -c "
from miniverl.models.tokenizers import HFTokenizerAdapter
a = HFTokenizerAdapter.load('Qwen/Qwen3-0.6B', revision='c1899de289a04d12100db370d81485cdf75e47ca')
b = HFTokenizerAdapter.load('Qwen/Qwen3-1.7B', revision='70d244cc86ccca08cf5af4e1e306ecf908b1ad5e')
print(a.fingerprint == b.fingerprint, a.vocab_size, b.vocab_size)
"
```

`miniverl inspect <trajectories.jsonl>` prints the tokenizer fingerprints
present in a trajectory file, so a mixed-fingerprint file is visible after the
fact.

A different error from the same area, specific to the toy backend:

```
error character 'é' (U+00E9) is outside the toy tokenizer's ASCII vocabulary
hint  the toy backend only handles printable ASCII; use the 'hf' backend for
      arbitrary text
```

The toy tokenizer raises rather than substituting an unknown token, because a
lossy tokenizer would break the token-provenance guarantees the project rests
on.

## Stale or corrupt teacher cache

**Symptom, stale**

```
error teacher targets for 'calc-train-7:v3:s1234' were produced by policy version
      3 but the update is running policy version 4
hint  that would make the update off-policy. Re-score the trajectory, or switch
      to run.mode=offline_kd if fixed targets are intended.
```

**Symptom, schema drift**

```
error cache schema_version 0 is not readable by this miniVERL build (expected 1)
hint  delete the cache directory and re-score; teacher targets are cheap to
      regenerate and must never be silently reinterpreted
```

**Symptom, corruption**

```
error checksum mismatch for 'calc-train-7:v0:s1234' in shard shard-00001.safetensors:
      expected <first 16 hex chars>..., got <first 16 hex chars>...
```

or `shard <name> referenced by the index is missing`, or
`<file> header is not valid JSON`.

**Cause.** The cache index records the teacher model id and revision, the
tokenizer fingerprint, the vocabulary size, `top_k`, the temperature and the
loss mode; every entry additionally records its `policy_version`. Reading with
`expect_policy_version` set, which the OPD trainer always does, raises
`StaleCacheError` on a mismatch. Every shard entry is checksummed with SHA-256
over its tensor bytes and verified on read.

**Fix.**

Inspect first, both commands work without torch:

```bash
miniverl cache stats runs/<run-id>/teacher-cache
miniverl cache validate runs/<run-id>/teacher-cache
```

`cache validate` exits non-zero and lists every structural problem it found.
`cache stats --no-verify` skips checksum recomputation if you only want the
provenance header.

For a stale-policy-version error, decide which semantics you want:

```yaml
# genuine on-policy distillation: targets are per-cycle and never reused
run:   {mode: opd}
cache: {strict_policy_version: true, reuse_across_policy_versions: false}

# offline KD: one fixed teacher cache reused for every update, explicitly
run:   {mode: offline_kd}
cache: {strict_policy_version: false, reuse_across_policy_versions: true}
```

`RunConfig` rejects any other combination at parse time.

For a corrupt or schema-drifted cache, delete the directory and re-score. The
cache is derived data:

```bash
rm -rf runs/<run-id>/teacher-cache
```

Related knobs: `cache.dir` relocates the cache outside the run directory,
`cache.keep_cycles` controls how many policy versions are retained before
`prune_before` deletes older entries, `cache.entries_per_shard` sets the shard
size, and `cache.verify_checksums_on_load` can be set to `false` if you have
measured that verification is your bottleneck.

## Gated or moved model revisions

**Symptom**

```
error could not load model 'Qwen/Qwen3-0.6B' at revision 'c1899de2...'
hint  check the model id and revision, that you are online for the first
      download, and that the license has been accepted on the Hub. Original
      error: <the underlying OSError>
```

**Cause.** `HFBackend.load` wraps the `OSError` that
`AutoModelForCausalLM.from_pretrained` raises. The common causes are a repo
that now requires accepting a license, a revision that no longer exists in the
repository, a private repository without a token, and no network on a first
download.

**Fix.**

Confirm the revision still resolves, and prove the check is real with a
negative control:

```bash
curl -s -o /dev/null -w "%{http_code}\n" \
  https://huggingface.co/api/models/Qwen/Qwen3-0.6B/revision/c1899de289a04d12100db370d81485cdf75e47ca
# 200
curl -s -o /dev/null -w "%{http_code}\n" \
  https://huggingface.co/api/models/Qwen/Qwen3-0.6B/revision/0000000000000000000000000000000000000000
# 404
```

Then:

- if the model is gated, accept the license on the Hub with the account whose
  token is configured, then retry
- if the revision has moved, update `models.student.revision`,
  `models.student.tokenizer_revision`, `models.teacher.revision` and
  `models.teacher.tokenizer_revision` in the recipe, and re-verify the
  tokenizer fingerprints afterwards
- if you have the weights locally already, force the offline path:
  `miniverl train <recipe> --offline`, which sets `local_files_only=True` on
  every `from_pretrained` call and refuses network access

Leaving a revision unset is legal but not recommended. `miniverl validate`
warns:

```
warning models.student.revision is unpinned; the manifest will record 'unpinned'
warning models.teacher.revision is unpinned
```

Plan the download before you commit to it:

```bash
miniverl train recipes/qwen_consumer_gpu_calc.yaml --dry-run
```

`--dry-run` resolves and prints the plan, including `downloads_required`, and
loads no models.

## Slow decoding

**Symptom.** Rollouts dominate wall clock; `rollout_tokens_per_second` in
`metrics.jsonl` and in the `rollouts_collected` events is around ten tokens per
second on a modern GPU.

**Cause, measured.** Single-sequence decoding in miniVERL is kernel-launch
bound, not compute bound. On this machine a 14-token prefill costs 37.0 ms
while a cached 1-token step costs 30.9 ms: almost all of the per-step cost is
fixed overhead rather than work proportional to the token count. Supporting
measurements from the same probe, at 64 new tokens with a 36-token prefix:

| configuration | tok/s | peak allocated |
| --- | --- | --- |
| NF4 + bf16, deterministic | 11.19 | 0.862 GiB |
| NF4 + bf16, non-deterministic | 11.29 | not recorded |
| bf16 LoRA, deterministic | 12.84 | 1.170 GiB |
| bf16 LoRA, non-deterministic | 14.12 | not recorded |

For reference, applying the LM head at one position costs 0.48 ms, decoding 64
token ids costs 0.02 ms, and copying a 151936-float vector to the host costs
0.11 ms. None of those is the bottleneck.

Reproduce the probe on your own machine:

```bash
python scripts/gpu_probe_throughput.py
```

**Fix.** There is no batched rollout in this version, so the levers are budget
levers rather than throughput levers:

```yaml
rollout:
  max_new_tokens_per_turn: 48    # cap tokens generated per turn
  max_turns: 3                   # cap turns per episode
  max_total_tokens: 704          # hard cap on the whole trajectory
train:
  rollouts_per_cycle: 6          # fewer episodes per cycle
  cycles: 8
eval:
  tasks: 12                      # evaluate on fewer held-out tasks
  temperature: 0.0               # greedy skips the sampler entirely
models:
  student:
    attn_implementation: sdpa    # sdpa or eager
```

Turning off deterministic mode with `run: {deterministic: false}` recovered
about 10 percent on bf16 LoRA in the table above and almost nothing on NF4. It
also gives up the guarantees in `docs/reproducibility.md`; the default is
`true` for that reason.

Note that `models.student.gradient_checkpointing: true` sets
`model.config.use_cache = False` for training, but `HFBackend.generate`
temporarily re-enables `use_cache` for the duration of a generation call and
restores the previous value afterwards, so gradient checkpointing does not
disable the KV cache during rollouts.

## Config validation errors

**Symptom**

```console
$ miniverl validate bad.yaml
invalid recipe bad.yaml
  <root>: Value error, cache.reuse_across_policy_versions=true contradicts
  run.mode=opd: reusing one teacher cache across policy versions is offline KD.
  Set run.mode: offline_kd, or keep the cache strictly per-cycle.

hint  compare against recipes/toy_cpu.yaml, which is validated in CI
```

**Cause.** `RunConfig` uses `extra="forbid"`, so a misspelled key is an error
rather than a silently ignored one, and a `model_validator` rejects
contradictory combinations rather than letting you discover them three minutes
into a GPU run. The checks currently enforced:

| rejected combination | reason given |
| --- | --- |
| `run.mode: opd` with `cache.reuse_across_policy_versions: true` | reusing one cache across policy versions is offline KD |
| `run.mode: opd` with `cache.strict_policy_version: false` | OPD requires that targets can never be consumed by a different policy version |
| `run.mode: offline_kd` without `cache.reuse_across_policy_versions: true` | that flag is what makes the fixed-target semantics explicit |
| `divergence: jsd` with `jsd_beta` at 0.0 or 1.0 | at either endpoint the mixture collapses and the divergence is identically zero |
| `loss.mode: exact_full_vocab` with an explicit `top_k` other than 1 | `top_k` is meaningless in exact mode; remove the key |
| `run.mode: sft` with `loss.ce_weight` outside {0.0, 1.0} | SFT trains with cross-entropy only |
| `run.mode: sft` with `train.sft_warmup_cycles > 0` | redundant; the warmup applies to offline_kd and opd |
| toy backend with any quantization | the toy backend has no quantized path |
| `rollout.max_total_tokens <= rollout.max_new_tokens_per_turn` | the total budget must exceed one turn |
| `eval.max_turns > 4 * rollout.max_turns` | implausible |
| a quantized model with `memory.strategy: swap` | bitsandbytes parameters are device-pinned |
| `models.student.quantization` set with `lora.enabled: false` | a quantized student must be trained with LoRA adapters |
| `models.teacher.mode: privileged_context` on an environment without one | the environment provides no privileged context |
| `schema_version` other than 1 | not supported by this build |

Most of these are parse-time checks in `src/miniverl/config/models.py` and so
fire during `miniverl validate`. Two are enforced later, in
`OPDTrainer.from_config`, because they need the environment or the resolved
device: the quantized-plus-`swap` combination, and `privileged_context` on an
environment that provides none. `miniverl validate` downgrades the second to a
warning (`environment provides no privileged context`). The
`exact_full_vocab` guard rail lives in `src/miniverl/teachers/local.py` and
fires at scoring time.

**Fix.** Validate before training. `miniverl validate` downloads nothing and
allocates nothing:

```bash
miniverl validate recipes/qwen_consumer_gpu_calc.yaml
miniverl validate recipes/qwen_consumer_gpu_calc.yaml --json
```

On success it prints the resolved plan, including `planned_optimizer_steps`,
`optimizer_steps_per_cycle`, `eval_tasks` and `is_on_policy`, plus any
warnings. Warnings do not fail the command; the current ones cover unpinned
revisions and more than one optimizer step per rollout batch under `opd`.

For the loading path as well as the config, add:

```bash
miniverl train <recipe> --dry-run
```

Pydantic reports the failing field path as the location. `<root>` means a
cross-field validator rejected the combination, so the fix is in the
relationship between keys, not in one key's value.

## Report or eval on a non-run directory

**Symptom**

```console
$ miniverl eval --run recipes
error recipes does not look like a miniVERL run (no manifest.json)
hint  point at the run directory itself, not its parent
```

or, for a directory that does not exist:

```
error run directory not found: runs/typo
hint  pass the path printed by `miniverl train`, e.g. runs/<run-id>
```

or, for a directory with a manifest but no resolved config:

```
error runs/<id> has no config.resolved.yaml
hint  only runs created by `miniverl train` can be re-evaluated
```

**Cause.** `RunPaths.open` requires the directory to exist and to contain
`manifest.json`. The most common mistake is passing the parent directory
(`runs`) instead of one run inside it (`runs/<run-id>`). `evaluate_run`
additionally requires `config.resolved.yaml`, because it rebuilds the trainer
from the resolved config so any `auto` decision stays frozen at the value the
original run used.

**Fix.** Point at the run directory itself. It is the path `miniverl train`
prints on its last lines, in the shape
`run complete runs/<run-id>` followed by ready-to-paste `miniverl eval --run`
and report paths. Unless you pass `--run-id`, the run id is
`<UTC %Y%m%d-%H%M%S>-<slugified run.name>`, built by `make_run_id`.

List the candidates:

```bash
ls runs
python -c "import pathlib;print([p.parent.name for p in pathlib.Path('runs').glob('*/manifest.json')])"
```

A valid run directory contains `config.original.yaml`,
`config.resolved.yaml`, `manifest.json`, `environment.json`, `metrics.jsonl`,
`events.jsonl`, `trajectories.jsonl`, `eval_trajectories.jsonl`, `eval.json`,
`teacher-cache/` and `checkpoints/`.

A related error from `miniverl export-benchmark`:

```
error runs/<id> has no evaluation results to export
hint  run `miniverl eval --run <run-dir>` first, or train with eval.enabled: true
```

## Resume refused: config digest mismatch

**Symptom**

```
error the checkpoint was written by a different configuration
hint  resume with the run's config.resolved.yaml, not a modified recipe
```

**Cause.** `save_checkpoint` stores `config_digest = sha256(config.to_yaml())`
in `state.json`, and `load_from_checkpoint` refuses when the digest of the
trainer's current config differs. The check exists because resuming under a
changed learning rate, schedule, loss mode or model would silently produce a
run that is neither the original nor a clean new one.

**The trap.** The digest is computed over the config the trainer was
constructed with, which is what `config.original.yaml` holds.
`config.resolved.yaml` additionally has `run.run_id` filled in, and
`memory.strategy`, `loss.chunk_size` and `models.device` replaced with their
resolved values, so it hashes differently. Measured on a run whose only
difference was `run_id: null` becoming `run_id: docs-check`, resuming from
`config.resolved.yaml` was refused and resuming from `config.original.yaml`
succeeded. The hint quoted above is misleading whenever the recipe left
`run.run_id` unset, which is the normal case.

**Fix.** Resume from `config.original.yaml`, and give the resumed trainer a new
run id so it does not append to the original run's logs:

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

There is no `miniverl resume` subcommand; resume is a library API.

If you genuinely intend to continue under a changed configuration, that is a
new experiment. Load the weights only, which is what the benchmark harness
does for its shared cold start:

```python
from miniverl.training.checkpoint import load_checkpoint

load_checkpoint(
    checkpoint_dir,
    backend=trainer.student,
    optimizer=trainer.optimizer,
    device=trainer.student.device,
    include_optimizer=False,
    include_rng=False,
)
```

`load_checkpoint` performs no digest check; only `OPDTrainer.load_from_checkpoint`
does. Loading weights only also avoids inheriting optimizer momentum from a
schedule that no longer applies.

Two neighbouring checkpoint errors:

```
error <path> is not a miniVERL checkpoint (missing state.json)
hint  pass a directory such as runs/<run-id>/checkpoints/step-000010
```

```
error checkpoint contains 8 unknown parameter names, e.g. base_model.model...
hint  the checkpoint was written by a different model or LoRA config
```

The second means the adapter shapes or names do not match the student that is
loaded, usually because `models.student.lora.r` or `target_modules` changed.

## See also

- `docs/memory.md` - memory strategies, swap, and measured peak VRAM
- `docs/reproducibility.md` - seeding, determinism and exact resume
- `docs/benchmarking.md` - metric definitions and matched comparisons
- `docs/limitations.md` - what is out of scope in this version
