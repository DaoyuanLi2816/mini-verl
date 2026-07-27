# Examples

Two runnable examples, both CPU-only and network-free. Each one is executed as
part of the release check, so if it is here it works.

```bash
pip install "miniverl[train]"      # CPU torch is enough

python examples/custom_environment/reverse_environment.py runs/examples
python examples/custom_teacher/sharpened_teacher.py       runs/examples
```

## `custom_environment/reverse_environment.py`

A complete new `ToolEnvironment`: two tools, three difficulties, a deterministic
oracle, an exact verifier and a privileged context string for the teacher. It
registers itself with `@register`, verifies that its own oracle solves 8/8 hard
tasks, and then trains a toy student on it with genuine on-policy distillation.

The important line in its output is the oracle check:

```text
registered environments: ['calculator', 'jsonnav', 'reverse', 'sqlite']
oracle solves 8/8 hard tasks
```

The final task-success number will usually be low or zero. That is expected: the
toy models are around 200k parameters and the budget is a smoke test. The example
demonstrates the extension point, not a capability. See
[`docs/limitations.md`](../docs/limitations.md).

Copy this file as the starting point for your own environment. The checklist for
a correct one is in the "New environment" issue template.

## `custom_teacher/sharpened_teacher.py`

A custom `TeacherScorer` that wraps the built-in local scorer and sharpens its
top-k distribution before handing it to the loss. It shows the whole contract:
score the student's own states, return a provider the chunked loss can slice, and
keep the returned targets a normalized distribution.

It asserts what it claims:

```text
teacher: sharpened sharpness 2.0
mean teacher entropy: 1.1773 -> 0.5559 (nats)
sharpened targets stayed normalized and trained without error
```

Sharpening measurably reduced the teacher's entropy, the `K+1` bucket
distribution still sums to one to within `1e-5`, and a real training step ran
against the modified targets.

## Recipes rather than code

If you only need to change hyper-parameters, an environment or an objective, you
do not need Python at all — edit a recipe. Start from
[`recipes/toy_cpu.yaml`](../recipes/toy_cpu.yaml), which is commented
line by line, and validate it before running:

```bash
miniverl validate recipes/toy_cpu.yaml
miniverl train    recipes/toy_cpu.yaml --dry-run
```
