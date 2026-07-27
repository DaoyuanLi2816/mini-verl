# Third-party notices

miniVERL is licensed under Apache-2.0 (see [LICENSE](LICENSE)).

## Vendored or adapted source code

**None.** Every source file under `src/miniverl/` was written for this project.
No source code from another repository has been copied, vendored or adapted.

This matters for two projects in particular, because miniVERL is positioned next
to them:

* **verl** (`https://github.com/verl-project/verl`, Apache-2.0). No verl code,
  configuration schema, documentation wording, logo or visual identity is used
  here. miniVERL is an independent implementation of a much narrower scope and
  is not compatible with, derived from, or endorsed by verl. The name is a nod to
  the problem space only.
* **OPSD** (`https://github.com/HJSang/OPSD_OnPolicyDistillation`). This
  repository has no LICENSE file and is therefore all-rights-reserved; nothing
  from it has been copied. It is cited in `docs/comparisons.md` and
  `docs/references.md` as prior art only.

## Algorithms implemented from published descriptions

Implementing a published formula from its mathematical description is not
copying code, but the sources deserve credit. Full citations are in
[`docs/references.md`](docs/references.md).

| What | Source |
| --- | --- |
| The temperature-squared gradient correction on the distillation term | Hinton, Vinyals and Dean, *Distilling the Knowledge in a Neural Network*, arXiv:1503.02531 |
| Generalized Jensen-Shannon divergence and training on student-sampled sequences | Agarwal et al., *GKD*, arXiv:2306.13649 |
| Teaching a student on its own trajectories against a context-conditioned teacher (the `privileged_context` mode) | Ye, Dong, Wu, Huang and Wei, *On-Policy Context Distillation for Language Models*, arXiv:2602.12275 |
| Recording per-token teacher entropy as a first-class metric | Jin et al., *Entropy-Aware On-Policy Distillation of Language Models*, arXiv:2603.07079 |
| Numerically stable `log(1 - exp(x))` in two regimes | Machler, *Accurately Computing log(1 - exp(-\|a\|))*, R `Rmpfr` vignette, 2012 |
| NF4 quantization with double quantization, and LoRA adapters on a quantized base | Dettmers et al., *QLoRA*, arXiv:2305.14314; Hu et al., *LoRA*, arXiv:2106.09685 |

The top-k plus tail coarse-graining of a teacher distribution is not novel
either; TRL's `ServerDistillationTrainer` exposes `loss_top_k` with an optional
tail bucket. miniVERL's contribution there is naming it accurately and proving
the lower-bound relationship in tests, not inventing it.

## Runtime dependencies

Installed from PyPI, not redistributed in this repository. Each remains under
its own license; consult the installed distribution for the authoritative text.

| Package | License | Used for |
| --- | --- | --- |
| typer | MIT | CLI |
| rich | MIT | terminal rendering |
| pydantic | MIT | configuration and schema validation |
| PyYAML | MIT | recipe parsing |
| Jinja2 | BSD-3-Clause | HTML report template |
| platformdirs | MIT | platform-correct paths |
| safetensors | Apache-2.0 | the pickle-free tensor format used by the cache and checkpoints |
| torch (extra `train`) | BSD-3-Clause | tensors, autograd |
| transformers (extra `train`) | Apache-2.0 | causal-LM backend |
| peft (extra `train`) | Apache-2.0 | LoRA / QLoRA adapters |
| accelerate (extra `train`) | Apache-2.0 | device placement helpers |
| numpy (extra `train`) | BSD-3-Clause | array interop |
| bitsandbytes (extra `cuda`) | MIT | NF4 quantization, 8-bit optimizer |
| pytest, hypothesis, jsonschema, ruff, mypy, build, twine (extra `dev`) | MIT / Apache-2.0 / BSD | development only, not redistributed |

## Models

miniVERL downloads no weights by default. The published recipes name two models,
both **Apache-2.0**, pinned by revision:

* `Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`
* `Qwen/Qwen3-1.7B` at `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`

No model weights are committed to this repository, and `.gitignore` refuses the
common weight extensions so they cannot be added by accident.

## Other assets

`docs/banner.svg` was drawn for this project. It uses no third-party artwork,
font file, icon set or trademark. The `Contributor Covenant` text in
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) is version 2.1, distributed under
CC BY 4.0 by the Contributor Covenant project.
