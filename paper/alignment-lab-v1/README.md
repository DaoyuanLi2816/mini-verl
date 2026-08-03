# Alignment Lab v1 technical report

`alignment-lab-v1.pdf` is generated deterministically from the frozen final
result, the task-level record, the State × Supervision diagnostic, the public
preregistration and the immutable calculator result.

Build with a Python environment containing ReportLab:

```bash
python paper/alignment-lab-v1/build_report.py
```

The builder refuses any source whose SHA-256 differs from the expected frozen
value. The committed PDF is rendered to images and inspected page by page
before release; its digest is pinned by the publication test.
