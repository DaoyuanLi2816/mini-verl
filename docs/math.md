# The miniVERL objective

This document defines exactly what miniVERL optimizes, and states which
property is asserted by which test. It covers the five modules under
`src/miniverl/losses/`:

| module | responsibility |
| --- | --- |
| `numerics.py` | float32 reductions, `log1mexp`, KL and entropy from log-probabilities |
| `exact.py` | full-vocabulary forward KL, reverse KL, beta-JSD, temperature scaling |
| `bucketed.py` | the `top-k + tail` coarse-graining and its divergences |
| `reduction.py` | masked, weight-normalized reduction |
| `chunked.py` | the chunked evaluation and the two-stage backward |

Nothing here depends on a model, a config object or a device. That is what makes
the brute-force reference tests possible.

---

## 1. Notation

Fix one **selected prediction position**. Let $V$ be the vocabulary size,
$z^{T} \in \mathbb{R}^{V}$ the teacher logits at that position and
$z^{S} \in \mathbb{R}^{V}$ the student logits. Let $\tau > 0$ be
`loss.temperature`. Define

$$
p_v \;=\; \frac{\exp(z^{T}_v/\tau)}{\sum_{u}\exp(z^{T}_u/\tau)},
\qquad
q_v \;=\; \frac{\exp(z^{S}_v/\tau)}{\sum_{u}\exp(z^{S}_u/\tau)} .
$$

Throughout, $P$ is the teacher and $Q$ is the student. Orientation is part of
every function name in `exact.py` and `bucketed.py`, because a swapped-argument
bug in a KL is invisible in the loss curve.

All quantities are in **nats** (natural log). All reductions are accumulated in
float32: `to_float32` upcasts bf16 and fp16 inputs before any `log_softmax` or
`logsumexp`, because a 151936-entry softmax reduction in half precision loses
several digits of the tail mass this project depends on. float64 inputs are left
alone, which is what lets the property tests check the *mathematics* rather than
float32 accumulation.

---

## 2. The three divergences

`src/miniverl/losses/exact.py` materializes the full $[N, V]$ distributions for
$N$ selected positions and returns a per-position vector of shape $[N]$.

### 2.1 Forward KL — mass-covering

$$
D_{\mathrm{KL}}(P \,\|\, Q) \;=\; \sum_{v=1}^{V} p_v \log \frac{p_v}{q_v}
$$

`exact_forward_kl(teacher_logits, student_logits, ...)`. The expectation is
taken under the teacher, so every token the teacher gives mass to costs the
student something if it does not cover it. The student is pushed to spread mass
over the whole teacher support.

### 2.2 Reverse KL — mode-seeking

$$
D_{\mathrm{KL}}(Q \,\|\, P) \;=\; \sum_{v=1}^{V} q_v \log \frac{q_v}{p_v}
$$

`exact_reverse_kl(teacher_logits, student_logits, ...)`. The expectation is
taken under the student, so the student is penalized only where it *puts* mass.
It is free to ignore teacher modes it cannot represent, and it is the default
`loss.divergence` because it is the usual on-policy distillation objective.

### 2.3 Beta-weighted Jensen-Shannon

With $M = \beta P + (1-\beta) Q$,

$$
\mathrm{JS}_{\beta}(P, Q) \;=\; \beta\, D_{\mathrm{KL}}(P \,\|\, M)
\;+\; (1-\beta)\, D_{\mathrm{KL}}(Q \,\|\, M)
$$

`exact_jsd(teacher_logits, student_logits, beta=..., ...)`. The mixture is
formed in log space with `torch.logaddexp(log_p + log(beta), log_q + log(1-beta))`,
so no probability is ever exponentiated and re-logged.

$\beta = 0.5$ is the symmetric Jensen-Shannon divergence, which in nats is
bounded above by $\log 2 \approx 0.6931$.

### 2.4 Why $\beta$ must be strictly inside $(0, 1)$

At $\beta = 1$ the mixture collapses to $M = P$, so

$$
\mathrm{JS}_{1} = 1 \cdot D_{\mathrm{KL}}(P\|P) + 0 \cdot D_{\mathrm{KL}}(Q\|P) = 0 ,
$$

and symmetrically $\mathrm{JS}_{0} = 0$. Both endpoints give a divergence that
is **identically zero for every input pair** — a silently non-training
objective, not a KL limit.

The KL limits are recovered from the *ratio*, not the value. Expanding around
$\beta = 0$, the second term is $O(\beta^{2})$ and

$$
\frac{\mathrm{JS}_{\beta}(P,Q)}{\beta} \;\xrightarrow[\beta \to 0]{}\; D_{\mathrm{KL}}(P \,\|\, Q) .
$$

This is checkable:

```python
import torch

from miniverl.losses.exact import exact_forward_kl, exact_jsd

g = torch.Generator().manual_seed(13)
teacher = torch.randn(4, 11, generator=g)
student = torch.randn(4, 11, generator=g)

print("KL(P||Q) =", float(exact_forward_kl(teacher, student).mean()))
for beta in (0.5, 0.1, 0.01, 0.001):
    js = float(exact_jsd(teacher, student, beta=beta).mean())
    print(f"beta={beta:<6} JS={js:.6e}  JS/beta={js / beta:.6f}")
```

Output on the development machine:

```
KL(P||Q) = 0.8532112240791321
beta=0.5    JS=1.839693e-01  JS/beta=0.367939
beta=0.1    JS=7.205847e-02  JS/beta=0.720585
beta=0.01   JS=8.379206e-03  JS/beta=0.837921
beta=0.001  JS=8.516113e-04  JS/beta=0.851611
```

The endpoints are rejected in two places rather than one: `exact_jsd` and
`bucketed_jsd` both raise `ValueError` at $\beta \notin (0,1)$, and
`RunConfig._validate_combination` rejects the recipe at parse time when
`loss.divergence: jsd`. The config field itself allows `ge=0.0, le=1.0` so that
the value is still storable when the divergence is not JSD.

---

## 3. Temperature and the $\tau^{2}$ factor

Both distributions are softmaxed at the same $\tau$. When
`loss.scale_by_temperature_squared` is true and $\tau \neq 1$, the per-position
divergence is multiplied by $\tau^{2}$ (`exact.temperature_scale`). At
$\tau = 1$ the factor is exactly 1 either way, so the flag is a no-op there.

### 3.1 Why the factor exists

Softening by $\tau$ shrinks the gradient the objective delivers to the student
logits. For forward KL, with $P$ held fixed,

$$
\frac{\partial}{\partial z^{S}_{j}} D_{\mathrm{KL}}(P\|Q)
\;=\; -\sum_{v} p_v \frac{\partial \log q_v}{\partial z^{S}_{j}}
\;=\; -\sum_{v} p_v \cdot \frac{1}{\tau}\left(\delta_{vj} - q_j\right)
\;=\; \frac{1}{\tau}\left(q_j - p_j\right) .
$$

That is one explicit factor of $1/\tau$ from the chain rule through $z/\tau$. A
second appears in the softened regime: for zero-meaned logits and $\tau$ large
enough that $\exp(z/\tau) \approx 1 + z/\tau$,

$$
q_j - p_j \;\approx\; \frac{z^{S}_{j} - z^{T}_{j}}{V\tau},
\qquad\text{so}\qquad
\frac{\partial L}{\partial z^{S}_{j}} \;\approx\; \frac{z^{S}_{j} - z^{T}_{j}}{V\tau^{2}} .
$$

This is the classic argument from Hinton, Vinyals and Dean (2015). Without the
correction the effective learning rate of a distillation term depends on $\tau$,
so raising the temperature to soften the targets silently also weakens the
update. Multiplying by $\tau^{2}$ separates the two: $\tau$ then controls only
*what* is matched.

miniVERL applies the factor to all three divergences, not only to forward KL, so
that changing $\tau$ never changes the update scale by itself.

### 3.2 Measured

```python
import torch

from miniverl.losses.exact import exact_forward_kl

g = torch.Generator().manual_seed(3)
teacher = torch.randn(1, 2000, generator=g) * 0.05  # near-uniform: the high-T regime

for temperature in (1.0, 2.0, 4.0, 8.0):
    row = []
    for scale in (False, True):
        student = (
            torch.randn(1, 2000, generator=torch.Generator().manual_seed(4)) * 0.05
        ).requires_grad_(True)
        exact_forward_kl(
            teacher,
            student,
            temperature=temperature,
            scale_by_temperature_squared=scale,
        ).sum().backward()
        row.append(float(student.grad.abs().mean()))
    print(f"T={temperature:<4}  without T^2: {row[0]:.3e}   with T^2: {row[1]:.3e}")
```

Output on the development machine:

```
T=1.0   without T^2: 2.857e-05   with T^2: 2.857e-05
T=2.0   without T^2: 7.144e-06   with T^2: 2.858e-05
T=4.0   without T^2: 1.786e-06   with T^2: 2.858e-05
T=8.0   without T^2: 4.465e-07   with T^2: 2.858e-05
```

The uncorrected gradient falls by exactly a factor of four per doubling of
$\tau$, which is the $1/\tau^{2}$ prediction. The corrected gradient is
$\tau$-independent to three digits.

Note that the cross-entropy term described in [section 7.3](#73-the-cross-entropy-term)
is **not** temperature-scaled: `_cross_entropy` in `chunked.py` takes a plain
`log_softmax` of the raw student logits. Only the divergence uses $\tau$.

---

## 4. The `top-k + tail` coarse-graining

`loss.mode: bucketed_topk_tail` (the default) does not compute a
full-vocabulary KL. This section says precisely what it computes instead.

### 4.1 Definition

Let $S \subseteq \{1,\dots,V\}$ be the indices of the $k$ largest **teacher**
probabilities, $|S| = k$ (`torch.topk`, with $k$ clipped to $V$). Define the
map

$$
T(v) \;=\;
\begin{cases}
v & v \in S \\
\perp & v \notin S
\end{cases}
$$

which partitions the vocabulary into $k$ singleton cells plus one aggregate
tail cell $S^{c}$. The **same** map is applied to both distributions, giving
the $k+1$-category pushforwards

$$
\tilde p_j = p_j \;(j \in S), \qquad \tilde p_{\perp} = \sum_{v \notin S} p_v ,
$$
$$
\tilde q_j = q_j \;(j \in S), \qquad \tilde q_{\perp} = \sum_{v \notin S} q_v .
$$

The bucketed divergence is the corresponding divergence between $\tilde P$ and
$\tilde Q$.

In code:

- `teacher_topk_targets(teacher_logits, top_k, temperature)` returns
  `(topk_indices [N,k], topk_log_probs [N,k], tail_log_prob [N])`. The
  `topk_log_probs` are log-probabilities over the **full** vocabulary restricted
  to $S$, so they do not sum to one; `tail_log_prob` is
  $\log(1 - \sum_{j \in S} p_j)$, computed by `log1mexp`.
- `student_bucket_log_probs(student_logits, topk_indices, temperature)`
  normalizes the student over the **full** vocabulary first, then gathers at the
  teacher's indices. Normalizing first is what makes $\tilde q_{\perp}$
  meaningful.
- `build_bucket_distributions(...)` floors both tails, concatenates and
  renormalizes to two exact $[N, k+1]$ log-probability vectors.

### 4.2 This is not full-vocabulary KL

The functions are named `bucketed_*` so no call site can pretend otherwise, and
the manifest records `loss_mode` alongside `top_k`. The number reported by a
`bucketed_topk_tail` run is a divergence between two $(k+1)$-category
distributions. It is a **lower bound** on the full-vocabulary divergence.

```python
import torch

from miniverl.losses.bucketed import bucketed_divergence, teacher_topk_targets
from miniverl.losses.exact import exact_divergence

g = torch.Generator().manual_seed(0)
teacher = torch.randn(4, 512, generator=g) * 3.0
student = torch.randn(4, 512, generator=g) * 3.0
exact = float(exact_divergence(teacher, student, divergence="reverse_kl").mean())

for k in (1, 4, 16, 64, 256, 512):
    idx, lp, tail = teacher_topk_targets(teacher, top_k=k)
    bucketed = bucketed_divergence(
        teacher_topk_log_probs=lp,
        teacher_tail_log_prob=tail,
        topk_indices=idx,
        student_logits=student,
        divergence="reverse_kl",
    )
    print(f"k={k:4d}  bucketed={float(bucketed.mean()):.4f}  exact={exact:.4f}")
```

Output on the development machine:

```
k=   1  bucketed=0.6266  exact=8.0730
k=   4  bucketed=1.7172  exact=8.0730
k=  16  bucketed=2.5005  exact=8.0730
k=  64  bucketed=3.8186  exact=8.0730
k= 256  bucketed=6.1714  exact=8.0730
k= 512  bucketed=8.0730  exact=8.0730
```

On this deliberately adversarial pair (independent random logits at scale 3,
$V = 512$) the default $k = 64$ recovers under half the exact reverse KL. On a
real teacher and a partly-trained student the top-64 mass is far higher and the
gap is far smaller, but the *direction* of the bias is fixed: bucketing can only
under-report.

### 4.3 Proof sketch: why it is a lower bound

The general statement is the **data-processing inequality** for $f$-divergences:
for any Markov kernel $T$, $D_f(P \circ T^{-1} \| Q \circ T^{-1}) \le D_f(P\|Q)$.
A deterministic partition map is a Markov kernel, so coarse-graining can only
destroy information. For KL specifically the argument is one line of the
**log-sum inequality**.

*Log-sum inequality.* For non-negative $a_i, b_i$,

$$
\sum_i a_i \log \frac{a_i}{b_i} \;\ge\; \Big(\sum_i a_i\Big) \log \frac{\sum_i a_i}{\sum_i b_i} ,
$$

with equality if and only if $a_i / b_i$ is constant over $i$.

*Forward KL.* Apply it with $a_v = p_v$, $b_v = q_v$ on each cell $A$ of the
partition $\mathcal{A} = \{\{j\} : j \in S\} \cup \{S^{c}\}$:

$$
\sum_{v \in A} p_v \log \frac{p_v}{q_v} \;\ge\; \tilde p_A \log \frac{\tilde p_A}{\tilde q_A} .
$$

Summing over the $k+1$ cells, the left-hand side is exactly
$D_{\mathrm{KL}}(P\|Q)$ and the right-hand side is exactly
$D_{\mathrm{KL}}(\tilde P \| \tilde Q)$. Hence

$$
D_{\mathrm{KL}}(\tilde P \,\|\, \tilde Q) \;\le\; D_{\mathrm{KL}}(P \,\|\, Q) .
$$

*Reverse KL.* Identical, with $a_v = q_v$ and $b_v = p_v$. The partition is
unchanged — it is still defined by the *teacher's* top-k — so the same argument
gives $D_{\mathrm{KL}}(\tilde Q \| \tilde P) \le D_{\mathrm{KL}}(Q\|P)$.

*Beta-JSD.* Mixing commutes with the pushforward:
$\tilde M = \beta \tilde P + (1-\beta)\tilde Q$ is the coarse-graining of
$M = \beta P + (1-\beta) Q$, because summing over a cell is linear. Applying the
KL bound to each of the two terms gives
$\mathrm{JS}_{\beta}(\tilde P, \tilde Q) \le \mathrm{JS}_{\beta}(P, Q)$.

### 4.4 Equality condition

The $k$ singleton cells contribute equality trivially. The bound is therefore
tight **if and only if the log-sum inequality is tight on the tail cell**, that
is, if and only if

$$
\frac{p_v}{q_v} \;\text{is constant for all } v \notin S .
$$

Three cases where that holds:

1. $k = V$. The tail cell is empty. `teacher_topk_targets` detects this and sets
   `tail_log_prob` to exactly $-\infty$ rather than relying on floating-point
   cancellation; `student_bucket_log_probs` does the same. Both are then clamped
   up to $\log \varepsilon$ by `build_bucket_distributions` and renormalized, so
   both distributions carry an identical $\varepsilon/(1+\varepsilon)$ bucket
   that contributes $\varepsilon \log 1 = 0$ to any of the three divergences.
   Every remaining bucket is scaled by the same $1/(1+\varepsilon)$, so the
   result equals the exact divergence times $1/(1+\varepsilon)$ — a relative
   perturbation of about $10^{-9}$ at the default $\varepsilon$, inside the
   tolerance the float64 property test uses.
2. $k = V - 1$. The tail cell is a singleton, so the ratio is trivially constant.
3. Teacher and student are proportional on the tail — for instance both uniform
   there.

For beta-JSD, tightness needs $p_v/m_v$ and $q_v/m_v$ both constant on the tail,
which since $m = \beta p + (1-\beta) q$ is equivalent to the same condition.

### 4.5 Monotone in $k$

Going from $k$ to $k+1$ splits the tail cell into a new singleton and a smaller
tail — a strict refinement of the partition. Applying the log-sum inequality to
the split cell alone shows the coarse-grained divergence cannot decrease.
Empirically this is visible in the table in section 4.2 and asserted in
`tests/unit/test_losses_bucketed.py::test_bucketed_is_monotone_non_decreasing_in_k`.

### 4.6 Entropy

`bucketed_teacher_entropy` returns $H(\tilde P)$, which by the grouping property

$$
H(P) \;=\; H(\tilde P) \;+\; \sum_{A \in \mathcal{A}} \tilde p_A \, H(P \mid A) \;\ge\; H(\tilde P)
$$

lower-bounds the true full-vocabulary entropy: merging the tail into one bucket
discards its internal spread. Reports label it as the coarse-grained entropy.

### 4.7 What the compression actually saves

Teacher-side storage and transfer: $k$ indices, $k$ log-probabilities and one
tail scalar per position, instead of $V$ logits. It is also what makes the
`swap` memory strategy possible at all, because the targets survive the teacher
being evicted from VRAM.

It does **not** reduce the student forward/backward cost. The student still
needs a full-vocabulary `log_softmax` over each selected position to normalize
correctly — that is precisely what makes $\tilde q_{\perp}$ meaningful — so the
$[chunk, V]$ student tensor is built either way.

---

## 5. Floors

Two independent floors keep the tail arithmetic finite. They interact, and the
interaction is easy to get wrong, so both are spelled out.

### 5.1 `log1mexp` and its two regimes

`log1mexp(x)` computes $\log(1 - e^{x})$ for $x \le 0$, which is needed to turn
the covered mass $\log \sum_{j \in S} p_j$ into the tail
$\log\left(1 - \sum_{j \in S} p_j\right)$.

A single formula loses precision at one end or the other, so two are used, split
at $x^{*} = -\log 2$ (Mächler, 2012):

| regime | condition | formula | why |
| --- | --- | --- | --- |
| near zero | $x > -\log 2$, i.e. $e^{x} > \tfrac12$ | $\log(-\mathrm{expm1}(x))$ | `expm1` avoids the catastrophic cancellation in $e^{x} - 1$ for small $\lvert x\rvert$ |
| far from zero | $x \le -\log 2$ | $\mathrm{log1p}(-e^{x})$ | `log1p` avoids losing the small $e^{x}$ against 1 |

Both branches are evaluated on **sanitized** inputs. `torch.where` substitutes
the safe value $-\log 2 - 1$ into whichever branch is inactive, then selects.
The naive version — evaluating both branches on the raw tensor and selecting
with `torch.where` — produces a correct forward value and a NaN gradient,
because the discarded branch still contributes to the backward pass.

### 5.2 `NEG_CLAMP`

`log(1 - exp(x))` diverges to $-\infty$ as $x \to 0^{-}$. Before anything else,
`log1mexp` clamps its input:

```python
x = x.clamp(max=NEG_CLAMP)  # NEG_CLAMP = -1.0e-7
```

so the output is bounded below by

$$
\log\!\left(1 - e^{-10^{-7}}\right) \;\approx\; \log(10^{-7}) \;\approx\; -16.118 .
$$

The consequence, which matters for reading a reported tail mass: because the
clamp acts on the **input**, any true tail mass below $10^{-7}$ is reported as
exactly $10^{-7}$.

```python
import math

import torch

from miniverl.losses.numerics import log1mexp

for tail in (1e-3, 1e-6, 1e-7, 1e-8, 1e-30):
    covered = torch.tensor([math.log1p(-tail)], dtype=torch.float64)  # log(1 - tail)
    recovered = math.exp(float(log1mexp(covered)))
    print(f"true tail {tail:.0e} -> log1mexp recovers {recovered:.3e}")
```

Output on the development machine:

```
true tail 1e-03 -> log1mexp recovers 1.000e-03
true tail 1e-06 -> log1mexp recovers 1.000e-06
true tail 1e-07 -> log1mexp recovers 1.000e-07
true tail 1e-08 -> log1mexp recovers 1.000e-07
true tail 1e-30 -> log1mexp recovers 1.000e-07
```

This is a deliberate trade. The alternative — an exact $-\infty$ for a
near-deterministic teacher — makes the reverse-KL tail term $+\infty$ whenever
the student leaks any probability outside the top-k, which is always true early
in training.

### 5.3 `tail_epsilon`

`build_bucket_distributions` applies a second, configurable floor before
concatenating. From the source:

```python
log_eps = math.log(tail_epsilon)
teacher = torch.cat(
    [
        to_float32(teacher_topk_log_probs),
        to_float32(teacher_tail_log_prob).clamp_min(log_eps).unsqueeze(-1),
    ],
    dim=-1,
)
# ... the same two lines for the student ...
teacher = teacher - torch.logsumexp(teacher, dim=-1, keepdim=True)
student = student - torch.logsumexp(student, dim=-1, keepdim=True)
```

Only the tails are floored; the top-k log-probabilities are untouched. The
final subtraction renormalizes both $[N, k+1]$ vectors,
so they are exact probability distributions and the divergence is guaranteed
non-negative. The normalizer is $Z \in [1, 1+\varepsilon]$.

### 5.4 The bound the floor buys

Write $\varepsilon$ for `tail_epsilon`. After flooring and renormalizing,
$\tilde p_{\perp} \ge \varepsilon / (1+\varepsilon)$, so the reverse-KL tail
term is bounded:

$$
\tilde q_{\perp} \log \frac{\tilde q_{\perp}}{\tilde p_{\perp}}
\;\le\; 1 \cdot \log \frac{1+\varepsilon}{\varepsilon}
\;\le\; \log\frac{1}{\varepsilon} + \varepsilon .
$$

At the default $\varepsilon = 10^{-9}$ that is $20.72$ nats. Without the floor
the same term is $+\infty$.

**The two floors compose, and the tighter one wins.** For $k < V$ the teacher
tail has already passed through `log1mexp`, so it is at least $10^{-7}$, and the
`tail_epsilon` clamp does nothing unless $\varepsilon > 10^{-7}$. The effective
bound in the default configuration is therefore the tighter
$\log(10^{7}) \approx 16.12$ nats, not $20.72$.

```python
import math

import torch

from miniverl.losses.bucketed import bucketed_divergence, teacher_topk_targets

teacher = torch.full((1, 256), -80.0)
teacher[0, :2] = 40.0  # top-2 hold essentially all the teacher mass
student = torch.zeros(1, 256)  # uniform: 254/256 of the student mass is tail

idx, lp, tail = teacher_topk_targets(teacher, top_k=2)
for eps in (1e-3, 1e-6, 1e-9):
    value = bucketed_divergence(
        teacher_topk_log_probs=lp,
        teacher_tail_log_prob=tail,
        topk_indices=idx,
        student_logits=student,
        divergence="reverse_kl",
        tail_epsilon=eps,
    )
    print(f"eps={eps:.0e}  reverse_kl={float(value):.4f}  log(1/eps)={math.log(1 / eps):.4f}")
```

Output on the development machine:

```
eps=1e-03  reverse_kl=6.8091  log(1/eps)=6.9078
eps=1e-06  reverse_kl=13.6619  log(1/eps)=13.8155
eps=1e-09  reverse_kl=15.9465  log(1/eps)=20.7233
```

At $\varepsilon = 10^{-3}$ and $10^{-6}$ the `tail_epsilon` clamp binds and the
value tracks $\log(1/\varepsilon)$. At $\varepsilon = 10^{-9}$ it does not: the
value saturates at 15.95, just under the $16.12$ that `NEG_CLAMP` allows. The
run stays finite either way, which is the property being bought.

### 5.5 `LOG_PROB_FLOOR`

`safe_log_prob` clamps log-probabilities at `LOG_PROB_FLOOR = -1e30` before any
subtraction in `kl_from_log_probs`. In float32 $\exp(-10^{30})$ is exactly
`0.0`, so clamping never changes a probability, but it keeps every difference
$\log p - \log q$ finite and therefore keeps the gradient free of the
$\infty - \infty = \mathrm{NaN}$ that an unclamped $-\infty$ would produce.

---

## 6. Weight normalization

Every miniVERL objective is normalized by the **sum of effective token
weights** — never by the padded sequence length and never by the raw selected
position count.

$$
\mathcal{L} \;=\;
\frac{\sum_{i=1}^{N} w_i \, \ell_i}{\max\!\left(\sum_{i=1}^{N} w_i,\; \varepsilon_W\right)},
\qquad \varepsilon_W = \texttt{MIN\_TOTAL\_WEIGHT} = 10^{-12}
$$

where $\ell_i$ is the per-position value at selected position $i$ and $w_i$ is
`alignment.token_weights[i]`, which the selector sets to
`selection.critical_weight` for tool-call and final-answer tokens and
`selection.other_weight` otherwise.

Three consequences:

- **A zero weight masks a position exactly.** It contributes nothing to the
  numerator *and* nothing to the denominator, so the loss and every gradient are
  identical to a run where the position was never selected. This is not
  "approximately zero because it was averaged over a larger denominator", and it
  is asserted for both the loss and the gradient.
- **Loss magnitudes are comparable across selection budgets.** An
  `all_model_tokens` run and a `uniform_ratio: 0.35` run supervise very
  different numbers of positions. With a sequence-length denominator the second
  run's loss would read roughly a third of the first's for reasons having
  nothing to do with the policy.
- **Re-weighting does not rescale.** Doubling `critical_weight` changes which
  tokens dominate the average without changing its scale.

`reduction.weighted_mean` is the reference implementation and takes an optional
externally supplied `denominator`. The training path does not call it: for the
reason given in [section 7](#7-chunking), `chunked_selected_position_loss`
computes `denom = torch.clamp(w.sum(), min=MIN_TOTAL_WEIGHT)` once, from the
global weight vector, and divides every chunk by it.

### 6.1 An exact statement about the floor

When every weight is exactly zero the numerator is exactly zero too, so the loss
is exactly `0.0` and every gradient is zero.

The clamp is not a no-op in general, though. When the weight sum is strictly
between $0$ and $10^{-12}$ the denominator is raised to $10^{-12}$ and the
result exceeds the true weighted mean. No miniVERL configuration can reach that
state — `selection.critical_weight` and `selection.other_weight` are both
constrained to $(0, 100]$ and nothing scales them down — but the Hypothesis
search in `tests/property/test_property_losses.py::test_weighted_mean_is_a_weighted_mean`
does reach it directly, with `values=[0,0,0,0,1.0]` and
`weights=[0,0,0,0,2.22e-16]`, where the function returns `2.22e-4` rather than
`0`. That branch of the test asserts a stronger property than the
implementation provides.

---

## 7. Chunking

### 7.1 Why

A naive distillation step computes `[batch, seq_len, vocab]` logits. For a
151936-entry vocabulary at sequence length 768 that is 116 M floats per sequence
*before* the backward pass. miniVERL never builds that tensor:

1. the backbone runs once and produces hidden states;
2. only the selected prediction positions are gathered, giving `[N, H]`;
3. those are projected through the LM head in slices of `loss.chunk_size`, so
   the largest vocabulary-sized tensor alive at any moment is `[chunk_size, V]`.

`chunk_size` is a memory and throughput knob. It does not change the objective,
and that is the property an OOM retry relies on when it halves the value.

### 7.2 The two-stage backward, and why the gradient is identical

Backpropagating each chunk straight through the backbone would re-run the
backbone once per chunk. Instead, excerpted from
`chunked_selected_position_loss` (`...` marks elided lines):

```python
denom = torch.clamp(w.sum(), min=MIN_TOTAL_WEIGHT)

use_two_stage = backward and hidden_states.requires_grad
work = hidden_states.detach().requires_grad_(True) if use_two_stage else hidden_states

for start in range(0, n, chunk_size):
    end = min(start + chunk_size, n)
    chunk_hidden = work[start:end]
    student_logits = lm_head(chunk_hidden)
    ...
    chunk_loss = (combined * w[start:end]).sum() / denom
    if backward:
        (chunk_loss * loss_scale).backward()  # stage one
    ...
    del student_logits, divergence, ce, combined, chunk_loss  # [chunk, V] freed

if use_two_stage:
    grad_hidden = work.grad
    ...
    hidden_states.backward(gradient=grad_hidden)  # stage two
```

*Claim.* The parameter gradients are identical to those of the unchunked
computation, in exact arithmetic, for any `chunk_size`.

*Proof.* Let $h \in \mathbb{R}^{N \times H}$ be the selected hidden states, a
differentiable function of the backbone parameters $\theta$, and let the LM head
have parameters $\phi$. The loss is

$$
\mathcal{L} \;=\; \frac{1}{W}\sum_{i=1}^{N} w_i \, \ell_i(h_i; \phi),
\qquad W = \sum_{i=1}^{N} w_i ,
$$

where $\ell_i$ depends on row $i$ of $h$ and on nothing else. Partition
$\{1..N\}$ into contiguous chunks $C_1,\dots,C_m$ and set

$$
\mathcal{L}_c \;=\; \frac{1}{W}\sum_{i \in C_c} w_i \, \ell_i(h_i; \phi) .
$$

Because **every chunk divides by the same global $W$**, $\sum_c \mathcal{L}_c = \mathcal{L}$
identically — this is the single detail that makes the whole scheme correct, and
it is why `denom` is computed once from the full weight vector before the loop.

Gradients are linear in that decomposition:

$$
\frac{\partial \mathcal{L}}{\partial \phi} = \sum_{c} \frac{\partial \mathcal{L}_c}{\partial \phi},
\qquad
\frac{\partial \mathcal{L}}{\partial h} = \sum_{c} \frac{\partial \mathcal{L}_c}{\partial h},
$$

and the rows of $\partial \mathcal{L}_c/\partial h$ are zero outside $C_c$.
Stage one runs `backward()` on each $\mathcal{L}_c$ with `work` as a leaf, which
accumulates exactly $\sum_c \partial \mathcal{L}_c / \partial \phi$ into the LM
head parameters and exactly $\sum_c \partial \mathcal{L}_c / \partial h$ into
`work.grad`. Stage two calls `hidden_states.backward(gradient=work.grad)`, which
by the chain rule contributes

$$
\frac{\partial \mathcal{L}}{\partial \theta}
= \left(\frac{\partial h}{\partial \theta}\right)^{\!\top} \frac{\partial \mathcal{L}}{\partial h}
$$

and runs the backbone backward exactly once. $\square$

In floating point the only difference is summation order. Measured:

```python
import torch

from miniverl.losses.chunked import ExactTargetProvider, chunked_selected_position_loss

g = torch.Generator().manual_seed(2026)
hidden0 = torch.randn(37, 16, generator=g)
lm_head = torch.nn.Linear(16, 48, bias=False)
teacher_logits = torch.randn(37, 48, generator=g) * 2.0
weights = torch.rand(37, generator=g) + 0.1
provider = ExactTargetProvider(teacher_logits_fn=lambda a, b: teacher_logits[a:b])

grads = {}
for chunk in (1, 5, 37, 1000):
    hidden = (hidden0 * 1.0).requires_grad_(True)
    out = chunked_selected_position_loss(
        hidden_states=hidden,
        lm_head=lm_head,
        weights=weights,
        provider=provider,
        chunk_size=chunk,
        backward=True,
    )
    grads[chunk] = out.grad_hidden.clone()
    print(f"chunk={chunk:5d}  chunks={out.num_chunks:3d}  loss={out.loss:.10f}")

reference = grads[1000]
for chunk, grad in grads.items():
    print(chunk, "max grad diff vs unchunked:", float((grad - reference).abs().max()))
```

Output on the development machine:

```
chunk=    1  chunks= 37  loss=1.9166372251
chunk=    5  chunks=  8  loss=1.9166371822
chunk=   37  chunks=  1  loss=1.9166370630
chunk= 1000  chunks=  1  loss=1.9166370630
1 max grad diff vs unchunked: 2.35741026699543e-09
5 max grad diff vs unchunked: 3.055902197957039e-10
37 max grad diff vs unchunked: 0.0
1000 max grad diff vs unchunked: 0.0
```

Two details worth noting:

- The two-stage path is taken only when `backward=True` **and**
  `hidden_states.requires_grad`. An evaluation-mode call falls through to the
  single-stage path.
- `loss_scale` multiplies the value passed to `backward()` and nothing else. The
  returned `LossOutput.loss` is the unscaled value, which is what makes
  gradient accumulation over `train.gradient_accumulation_steps` trajectories
  (`loss_scale = 1 / len(group)`) report a comparable loss.

### 7.3 The cross-entropy term

Per selected position, with target token $y_i$,

$$
\ell^{\mathrm{CE}}_{i} \;=\; -\log \mathrm{softmax}(z^{S})_{y_i}
$$

at temperature 1, and the combined per-position value is a convex combination

$$
\ell_i \;=\; (1 - c)\,\ell^{\mathrm{div}}_{i} \;+\; c\,\ell^{\mathrm{CE}}_{i},
\qquad c = \texttt{loss.ce\_weight} \in [0, 1] .
$$

When no teacher provider is supplied — the SFT path — the loss is pure
cross-entropy and the trainer forces $c = 1$. When `ce_weight` is 0, the CE
branch is skipped entirely rather than multiplied by zero.

---

## 8. Test index

Every claim above is backed by at least one executed test. The mapping:

### `tests/unit/test_losses_exact.py`

| Claim | Test |
| --- | --- |
| Forward KL matches the textbook definition | `test_forward_kl_matches_brute_force` |
| Reverse KL matches the textbook definition | `test_reverse_kl_matches_brute_force` |
| Beta-JSD matches the textbook definition at $\beta \in \{0.1, 0.5, 0.9\}$ | `test_jsd_matches_brute_force` |
| Forward and reverse are distinguishable, and swapping the arguments swaps them | `test_orientation_forward_and_reverse_differ` |
| $D(P\|P) = 0$ for all three | `test_identical_distributions_are_zero` |
| All three are non-negative | `test_divergences_are_non_negative` |
| Symmetric JSD is bounded by $\log 2$ | `test_jsd_is_bounded_by_log_two` |
| $\beta \in \{0, 1\}$ is rejected | `test_jsd_rejects_degenerate_beta` |
| Finite at logit scale $10^3$ and $10^4$ | `test_extreme_logits_stay_finite` |
| Gradients are finite, non-zero and reach the student | `test_gradients_are_finite_and_flow_to_student` |
| fp16 and bf16 inputs reduce in float32 | `test_half_precision_inputs_reduce_in_float32` |
| The $\tau^{2}$ factor is exactly $\tau^{2}$ | `test_temperature_squared_scaling_is_applied` |
| The flag is a no-op at $\tau = 1$ | `test_temperature_one_is_unaffected_by_the_scaling_flag` |
| An unknown divergence name raises | `test_unknown_divergence_name_is_rejected` |
| Teacher entropy matches the definition | `test_teacher_entropy_matches_brute_force` |

The brute-force references (`_softmax_rows`, `_brute_kl`, `_brute_jsd`) are
plain Python loops written from the definitions, so a bug in the vectorized
implementation cannot hide behind the same expression on both sides.

### `tests/unit/test_losses_bucketed.py`

| Claim | Test |
| --- | --- |
| `log1mexp` matches `log1p(-exp(x))` across both regimes | `test_log1mexp_matches_reference` |
| `log1mexp` is finite and differentiable near zero | `test_log1mexp_is_finite_and_differentiable_near_zero` |
| Top-k mass plus tail mass equals one | `test_topk_targets_have_valid_tail_mass` |
| $k = V$ gives an exactly empty tail | `test_topk_equal_to_vocab_gives_exactly_empty_tail` |
| $k = V$ reproduces the exact loss (all three divergences) | `test_full_k_converges_to_the_exact_loss` |
| **Bucketed $\le$ exact** for $k \in \{1,2,8,32\}$, all three divergences | `test_bucketed_lower_bounds_exact` |
| Bucketed is non-decreasing in $k$ | `test_bucketed_is_monotone_non_decreasing_in_k` |
| Identical distributions give zero | `test_identical_distributions_give_zero` |
| A near-deterministic teacher produces no inf or NaN | `test_tail_edge_case_when_topk_mass_is_almost_one` |
| The reverse-KL tail penalty is bounded | `test_reverse_kl_tail_penalty_is_bounded_by_log_one_over_epsilon` |
| Gradients flow to the student only | `test_gradients_flow_to_the_student_only` |
| Student buckets sum to one | `test_student_bucket_log_probs_sum_to_one` |
| Coarse-grained entropy lower-bounds exact entropy | `test_bucketed_entropy_lower_bounds_the_exact_entropy` |
| bf16 student logits are upcast | `test_half_precision_student_logits_are_upcast` |
| `top_k < 1`, unknown divergence and `tail_epsilon = 0` are rejected | `test_invalid_arguments_are_rejected` |

### `tests/unit/test_chunked_equivalence.py`

| Claim | Test |
| --- | --- |
| The loss value is chunk-invariant for `chunk_size` $\in \{1,3,8,37,1000\}$ | `test_chunked_value_matches_unchunked` |
| The **gradient** matches an unchunked reference for `chunk_size` $\in \{1,5,37\}$ across all three divergences (`atol=1e-5`) | `test_chunked_gradients_match_unchunked` |
| A zero weight changes neither the loss nor any gradient | `test_zero_weight_positions_contribute_nothing` |
| All-zero weights give loss exactly 0 and zero gradients | `test_all_zero_weights_give_a_safe_zero_loss` |
| An empty selection is a documented no-op | `test_empty_selection_is_a_documented_no_op` |
| The bucketed provider matches a direct `bucketed_divergence` call | `test_bucketed_provider_matches_direct_call` |
| CE-only mode matches `torch.nn.functional.cross_entropy` | `test_cross_entropy_only_mode_matches_torch_reference` |
| CE mixing is a convex combination | `test_ce_mixing_is_a_convex_combination` |
| `loss_scale` affects gradients only | `test_loss_scale_only_affects_gradients` |
| Missing provider, missing targets and `chunk_size < 1` are rejected | `test_invalid_arguments_are_rejected` |

`test_chunked_gradients_match_unchunked` is the load-bearing one. It builds a
second backbone and LM head with identical initialization, runs one unchunked
backward through them, and compares the backbone weight gradient, the backbone
bias gradient and the LM head weight gradient against the chunked path.

### `tests/property/test_property_losses.py`

Hypothesis searches over vocabulary sizes 2–24, 1–5 rows and logits in
$[-30, 30]$, 60 examples per property.

| Claim | Test |
| --- | --- |
| All three divergences are finite and non-negative | `test_exact_divergences_are_finite_and_non_negative` |
| Symmetric JSD is symmetric and bounded by $\log 2$ | `test_symmetric_jsd_is_symmetric_and_bounded` |
| Self-divergence is zero | `test_self_divergence_is_zero` |
| Invariant to a constant logit shift (softmax shift-invariance) | `test_divergence_is_invariant_to_a_constant_logit_shift` |
| Entropy lies in $[0, \log V]$ | `test_entropy_is_between_zero_and_log_vocab` |
| **Bucketed $\le$ exact in float64**, all $k$, all three divergences | `test_bucketed_never_exceeds_exact` |
| $k = V$ reproduces exact to `atol=1e-9` in float64 | `test_full_k_reproduces_exact` |
| Top-k mass plus tail equals one | `test_teacher_topk_mass_and_tail_sum_to_one` |
| `log1mexp` matches the reference outside the clamp | `test_log1mexp_matches_the_reference_outside_the_clamp` |
| `log1mexp` clamps rather than diverging, at exactly $\log(10^{-7})$ | `test_log1mexp_clamps_instead_of_diverging_near_zero` |
| `weighted_mean` is a weighted mean and lies within the value range | `test_weighted_mean_is_a_weighted_mean` (see [6.1](#61-an-exact-statement-about-the-floor)) |
| Zero weights mask positions exactly | `test_zero_weights_mask_positions_exactly` |
| A shape mismatch raises | `test_weighted_mean_rejects_a_shape_mismatch` |
| Selection is reproducible for any ratio, seed and selector | `test_selection_is_reproducible_for_any_ratio_and_seed` |

`test_bucketed_never_exceeds_exact` runs in float64 on purpose. The bound is
tight (an equality) whenever the coarse-graining is the identity — for example
$V = k + 1$ — so in float32 the two sides can differ by a few units in the last
place in either direction. The float32 behaviour is covered separately by
`test_bucketed_lower_bounds_exact`.

Note that the data-processing inequality proved in section 4.3 applies to the
*unfloored* coarse-graining. The implementation additionally floors both tails
(section 5) and renormalizes, which perturbs both bucketed distributions. Both
tests above are run on the floored implementation, so they check the bound as
shipped rather than the theorem in isolation.

---

## 9. References

- Hinton, Vinyals, Dean. *Distilling the Knowledge in a Neural Network.* 2015.
  The source of the $\tau^{2}$ argument in section 3.
- Mächler. *Accurately Computing $\log(1-\exp(-\lvert a\rvert))$.* 2012. The
  two-regime `log1mexp` split at $-\log 2$ in section 5.1.
- Cover and Thomas, *Elements of Information Theory*, for the log-sum inequality
  and the grouping property of entropy used in sections 4.3 and 4.6.
- arXiv:2602.12275, *On-Policy Context Distillation for Language Models* (Ye,
  Dong, Wu, Huang, Wei). The reverse-KL-against-a-context-conditioned-teacher
  objective that `models.teacher.mode: privileged_context` implements.
- arXiv:2603.07079, *Entropy-Aware On-Policy Distillation of Language Models*
  (Jin, Min, Yang, Wei, Zhou, Kadhe, Baracaldo, Lee). Motivates recording
  teacher entropy per selected token. The entropy-aware mixing itself is
  **not implemented** in miniVERL — see the Roadmap in
  [design.md](design.md#8-roadmap-not-implemented).
