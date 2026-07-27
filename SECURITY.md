# Security policy

## Supported versions

miniVERL is at `0.1.0`. Security fixes land on `main` and in the next release.
There are no maintained older branches.

## Reporting a vulnerability

Please use GitHub's private vulnerability reporting on this repository
("Security" → "Report a vulnerability"), or email `lidaoyuan2816@gmail.com` with
`miniverl security` in the subject line. Please do not open a public issue for
anything that lets an attacker execute code or read files.

Expect an acknowledgement within a week. This is a personal open-source project,
not a funded product, so please calibrate expectations accordingly — but a real
code-execution report will be treated as urgent.

## Threat model

miniVERL is a local training tool. It assumes you trust the recipes you run and
the model weights you load. It does **not** assume you trust:

* **A trajectory file.** `trajectories.jsonl` is JSON, validated against a strict
  schema on every read. Nothing in the load path executes code. A file whose
  masks disagree with its span partition is rejected rather than trained on.
* **A teacher-target cache.** `teacher-cache/` is JSON metadata plus safetensors
  shards, each checksummed with SHA-256. `torch.save` and `pickle` are not used
  anywhere in the project. A corrupted or truncated shard raises
  `CacheCorruptionError`.
* **A checkpoint.** Checkpoints are safetensors plus JSON, for the same reason.
* **Text produced by the model.** The tool-call parser is strict, bounded
  (`MAX_TOOL_CALL_JSON_CHARS`), and never falls back to a permissive
  interpretation. Rollouts are bounded in turns, tokens, parse errors and
  repeated identical calls.

## Environment sandboxing

The three shipped environments are the parts most exposed to model output, and
each is locked down explicitly:

* **Calculator** parses expressions into an `ast` and walks it. There is no
  `eval` and no `exec`. Only numeric literals and `+ - * / // % **` with unary
  sign are accepted; names, calls, attribute access, subscripts, comprehensions,
  f-strings and boolean literals are all rejected. Recursion depth, operand
  magnitude and exponent size are bounded.
* **SQLite** runs against an in-memory database built from the task seed. A
  `sqlite3` **authorizer** denies every action by default and allows only
  `SQLITE_SELECT`, `SQLITE_READ` on the two known tables, and `SQLITE_FUNCTION`
  for a whitelist of aggregates. `ATTACH`, `PRAGMA`, `INSERT`, `UPDATE`,
  `DELETE`, `CREATE`, `DROP` and reads of `sqlite_master` are refused by the
  engine itself, not by string matching. A progress handler aborts a query after
  a bounded number of VM instructions. One statement per call. No file on disk is
  ever opened.
* **JSON navigation** is read-only over an in-memory document, with a bounded
  path length and a bounded result count.

If you add an environment, the same standard applies; see
[CONTRIBUTING.md](CONTRIBUTING.md).

## What miniVERL records

Run manifests deliberately exclude usernames, hostnames, home directories, API
keys and environment variables — apart from a short allowlist of variables that
change numerical results (`CUBLAS_WORKSPACE_CONFIG`, `PYTORCH_CUDA_ALLOC_CONF`,
`CUDA_VISIBLE_DEVICES`, `OMP_NUM_THREADS`, `TOKENIZERS_PARALLELISM`). A test
asserts that the hostname, the current username and the home path do not appear
anywhere in a manifest.

There is no telemetry. miniVERL makes no network request except the Hugging Face
downloads you ask for by naming a model, and `--offline` refuses even those.

## Known limitations

* Model weights you download are executed as tensors, not as code, but
  `trust_remote_code` is exposed as a config option and defaults to `false`.
  Turning it on runs code from the model repository; that is the Hugging Face
  threat model, not one miniVERL can improve.
* Reports are static HTML with no scripts and no external requests, but they
  embed model-generated text. Open them locally; do not serve them on a domain
  that shares cookies with something you care about.
