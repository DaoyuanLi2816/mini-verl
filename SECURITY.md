# Security policy

## Supported versions

The current supported stable line is `0.10.1`; security fixes land on `main` and
in the next patch release. There are no separately maintained older branches.

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
* **A qualification workflow artifact.** Release validation downloads it only
  from a successful manual workflow run bound to the exact source SHA. Archive
  members are count/size bounded; absolute paths, traversal and symlinks are
  rejected before extraction, then every declared file hash and portable JSON
  field is checked without importing Torch.
* **Text produced by the model.** The tool-call parser uses bounded strict JSON:
  duplicate keys, non-finite/pathological numbers, excessive depth or members,
  invalid Unicode and trailing action blocks are rejected. Rollouts are bounded
  in turns, tokens, parse errors and repeated identical calls.

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

Raw run configuration may contain a local adapter path needed for resume.
Shareable HTML/Markdown reports and benchmark exports use one canonical
portable view that removes home paths, host/user identity, environment
references and credential-like values. Review any artifact before publishing
it; never paste raw config or logs into an issue without checking them.

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
