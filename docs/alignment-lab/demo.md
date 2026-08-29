# Alignment Lab demo recording script

This reproducible recording plan uses the CPU toy backend to show the workflow
and artifact surfaces. Its 0% task score belongs to the demo fixture; the
measured v0.5 result is the separate three-seed Qwen3 artifact.

## Install

From a clone, install a PyTorch build appropriate for the machine, then the
training extra:

```powershell
python -m pip install torch
python -m pip install -e ".[train]"
```

For CUDA, install the matching PyTorch CUDA wheel first; the miniVERL extra does
not choose a CUDA build of PyTorch.

## Record

```powershell
powershell -ExecutionPolicy Bypass -File scripts/demo_alignment_lab.ps1
```

Suggested pacing:

| time | screen | truthful narration |
| ---: | --- | --- |
| 0–8 s | install and `miniverl --version` | one process; no Ray or cluster |
| 8–18 s | `miniverl pilot` | the empty toy pilot refuses to overclaim from zero evidence |
| 18–28 s | `align --dry-run` | base → SFT checkpoint → teacher → alignment → eval → card |
| 28–50 s | short CPU alignment | real SFT warmup, strict fresh rollouts, teacher scoring and updates |
| 50–62 s | `miniverl inspect` | only typed assistant spans enter the loss; tool output stays context |
| 62–75 s | Alignment Card and checkpoint | method, policy, cost, limitations and hashes are exportable artifacts |

The reviewed local run completed in 17.6 seconds on the development machine;
recording duration includes narration and terminal pauses. Runtime varies by
CPU. Do not edit out the card's toy-backend limitation or 0% result.

## Sanitized examples

- [Formal benchmark pilot decision](https://github.com/DaoyuanLi2816/mini-verl/blob/main/examples/alignment-lab/pilot.json)
- [Reviewed toy Alignment Card](https://github.com/DaoyuanLi2816/mini-verl/blob/main/examples/alignment-lab/alignment-card.json)
- [Outcome and cost matrix](outcome-cost-matrix.svg)

No user quote, adoption number or unmeasured GPU claim is part of this demo.
