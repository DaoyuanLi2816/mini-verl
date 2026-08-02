# RecoveryBench v1 technical report

`build_report.py` is the report source. It loads and schema-validates the three
committed RecoveryBench result JSON files plus the frozen paired analysis, then
builds `recoverybench-v1.pdf` with ReportLab. Tables and plots are data-bound;
the script refuses unexpected result hashes.

Read the [generated six-page report](recoverybench-v1.pdf). Its release-source
SHA-256 is
`b6000a9e0d1c665382cebd39ad99dd3de2176ca89a3845db7bb2516a68adaefb`.

From the repository root, using the Codex PDF runtime shown in the validation
record:

```powershell
& "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe" `
  paper\recoverybench-v1\build_report.py
```

The public report is scoped to one Qwen3 student/teacher pair, one SQLite
recovery environment, three seeds and one RTX 4080. It preserves the negative
fresh-state result and the cycle-cap limitation in the wall-time diagnostic.
