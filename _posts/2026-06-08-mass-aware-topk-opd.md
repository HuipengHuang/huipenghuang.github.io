---
title: "Top-k Reverse KL for Distillation: What Normalizing Throws Away"
date: 2026-06-08
categories:
  - blog
tags:
  - distillation
  - reverse-kl
  - top-k
  - llm
permalink: /posts/mass-aware-topk-opd/
---

# Top-$k$ Reverse KL for Distillation: What Normalizing Throws Away

If you've implemented on-policy distillation for an LLM, you've run into the same wall everyone does. The natural training signal is the reverse KL between the student $p$ and the teacher $q$ at each step, but computing it over a full $100$k–$200$k-token vocabulary, for every position, is expensive. So people reach for a top-$k$ approximation: keep the few tokens that matter, ignore the rest.

The catch is that there's more than one way to "keep the top $k$," and the most common one — renormalizing on the top-$k$ set — quietly discards information you almost certainly care about. This post walks through four objectives through a single lens, their **gradient with respect to the student logits**, and shows why a small change to the top-$k$ recipe (keeping the leftover mass as one aggregated tail category) recovers a *provable lower bound* of the full reverse KL.

The four objectives:

1. **Full-vocabulary** reverse KL — the thing we actually want.
2. **Sampled-token** — a one-sample policy-gradient estimator (with a trap worth knowing about).
3. **Normalized top-$k$** — the common approximation, and its blind spot.
4. **Mass-aware top-$k$** — the fix.

## Setup

Fix one prompt-prefix state. Let the student logits be $z = (z_1, \dots, z_M) \in \mathbb{R}^M$ over a vocabulary $\mathcal V$ of size $M$. The student distribution is the softmax

$$
p_i = \operatorname{softmax}(z)_i = \frac{e^{z_i}}{\sum_m e^{z_m}},
$$

and the teacher distribution $q_i = \pi_T(i \mid \text{prefix})$ is treated as fixed (we differentiate only w.r.t. $z$). Assume $q_i > 0$ throughout.

Two facts do all the work. The softmax Jacobian,

$$
\frac{\partial p_i}{\partial z_j} = p_i\big(\mathbf 1[i=j] - p_j\big),
$$

and the per-token log-ratio,

$$
r_i = \log \frac{p_i}{q_i}.
$$

## The full-vocabulary gradient

The objective is

$$
L_{\text{full}} = D_{\text{KL}}(p \Vert q) = \sum_{i \in \mathcal V} p_i \log \frac{p_i}{q_i} = \sum_i p_i r_i.
$$

Differentiating $p_i(\log p_i - \log q_i)$ w.r.t. $p_i$ gives $r_i + 1$. Chaining through the softmax Jacobian and using $\sum_i p_i(r_i + 1) = D_{\text{KL}}(p \Vert q) + 1$, the $+1$ cancels and we get a clean result:

$$
\boxed{\;\frac{\partial L_{\text{full}}}{\partial z_j} = p_j\left(\log\frac{p_j}{q_j} - D_{\text{KL}}(p \Vert q)\right)\;}
$$

The interpretation is a **relative reweighting**: each logit is compared against the *average* log-ratio (which is just $D_{\text{KL}}$ itself). Tokens where the student is too confident relative to the teacher ($\log(p_j/q_j) > D_{\text{KL}}$) get pushed down; the rest get pushed up. This baseline-subtraction structure shows up in every variant below.

## Sampling one token: the policy-gradient view

Computing the full sum is exactly what we're trying to avoid. So sample a single token $a \sim p$ and use its log-ratio $r_a = \log(p_a/q_a)$, which is an unbiased estimator of the KL *value*: $\mathbb E_{a\sim p}[r_a] = D_{\text{KL}}(p \Vert q)$.

For the *gradient*, the right estimator is the policy-gradient (REINFORCE) form:

$$
\boxed{\;\widehat g^{\text{sample}}_j = \left(\log\frac{p_a}{q_a}\right)\big(\mathbf 1[j=a] - p_j\big)\;}
$$

Taking the expectation over $a \sim p$ recovers the full-vocabulary gradient exactly — so this is an honest Monte Carlo estimator of $\partial L_{\text{full}}/\partial z_j$.

**The trap.** It's tempting to just sample $a$, treat it as a fixed label, and backprop through the scalar loss $\log p_a - \log q_a$. That gives $\partial/\partial z_j = \mathbf 1[j=a] - p_j$, whose expectation is **zero**. Naive backprop through the sampled loss value is *not* the policy-gradient estimator. If you want the policy-gradient behavior, detach the scalar weight:

```python
loss = detach(logp_a - logq_a) * logp_a
```

(As an aside: the "missing" $+1$ from the full-vocab derivation is just a constant baseline — $\mathbb E_{a\sim p}[\nabla_z \log p_a] = \nabla_z 1 = 0$ — so any constant baseline $b$ in $(r_a - b)$ stays unbiased.)

## Normalized top-$k$ and its blind spot

Now the common approximation. Take $S = \operatorname{TopK}(p, k)$, renormalize both sides onto $S$,

$$
\bar p_i = \frac{p_i}{P_S}, \quad \bar q_i = \frac{q_i}{Q_S}, \qquad P_S = \sum_{u\in S} p_u, \quad Q_S = \sum_{u\in S} q_u,
$$

and minimize $L_{\text{top}k} = D_{\text{KL}}(\bar p \Vert \bar q)$. Because $\bar p$ is just a softmax over the selected logits, the derivation mirrors the full-vocab case on the subset $S$:

$$
\boxed{\;\frac{\partial L_{\text{top}k}}{\partial z_j} =
\begin{cases}
\bar p_j\left(\log\dfrac{\bar p_j}{\bar q_j} - D_{\text{KL}}(\bar p \Vert \bar q)\right), & j \in S,\\[1em]
0, & j \notin S.
\end{cases}\;}
$$

Look at what renormalization did. The objective depends only on the *conditional shape* $\bar p$ inside $S$ — it is **invariant to the total head mass $P_S$**. The gradient is identically zero outside $S$, and a mismatch between $P_S$ and $Q_S$ is never penalized.

That's the blind spot: **if $\bar p$ matches $\bar q$ inside $S$, the loss is zero even if $P_S$ is wildly different from $Q_S$** — i.e. even if the student puts a totally wrong fraction of its mass on the head region in the first place. The shape can be perfect while the budget is wrong, and normalized top-$k$ can't see it.

## Mass-aware top-$k$: keep the mass

The fix is small. Take the teacher top-$k$ set $S = \operatorname{TopK}(q, k)$, but **don't renormalize**. Keep the original probabilities on $S$, and collapse everything else into a single tail category:

$$
p_\tau = 1 - P_S = \sum_{i \notin S} p_i, \qquad q_\tau = 1 - Q_S = \sum_{i \notin S} q_i.
$$

The loss is the KL over the resulting $(k{+}1)$-category distribution:

$$
L_{\text{MA}} = \sum_{i \in S} p_i \log\frac{p_i}{q_i} + p_\tau \log\frac{p_\tau}{q_\tau}
= D_{\text{KL}}\!\Big( [\{p_i\}_{i\in S}, p_\tau] \,\big\Vert\, [\{q_i\}_{i\in S}, q_\tau] \Big).
$$

The name says what it does: it's a top-$k$ objective that stays **aware of the total mass** on the head region, because the tail term compares $p_\tau$ against $q_\tau$ directly — which is exactly the comparison normalized top-$k$ throws away.

### Why it's a principled lower bound

Here's the part that makes this more than a heuristic. Decompose the full reverse KL into head and tail, and rewrite the tail using the within-tail conditionals $\tilde p_i = p_i/p_\tau$, $\tilde q_i = q_i/q_\tau$:

$$
\sum_{i \notin S} p_i \log\frac{p_i}{q_i}
= \underbrace{p_\tau \log\frac{p_\tau}{q_\tau}}_{\text{kept by } L_{\text{MA}}}
+ p_\tau\, D_{\text{KL}}(\tilde p_{\text{tail}} \Vert \tilde q_{\text{tail}}).
$$

So the difference between the two objectives is exactly the discarded within-tail divergence:

$$
\boxed{\; L_{\text{full}} - L_{\text{MA}} = p_\tau\, D_{\text{KL}}(\tilde p_{\text{tail}} \Vert \tilde q_{\text{tail}}) \ge 0 \quad\Longrightarrow\quad L_{\text{MA}} \le L_{\text{full}} \;}
$$

This isn't an approximation with an unknown sign. $L_{\text{MA}}$ is *literally* the full reverse KL after a deterministic **coarse-graining** of the vocabulary — merge all non-top-$k$ tokens into one bucket. By the log-sum inequality (equivalently the data processing inequality for KL), coarse-graining can never *increase* KL, which gives the bound for free. The slack is explicit and non-negative, and the bound is tight exactly when the student and teacher agree *conditionally* inside the tail, $p_i/p_\tau = q_i/q_\tau$ for all $i \notin S$.

Minimizing $L_{\text{MA}}$ therefore drives down a guaranteed lower bound of the thing we actually want. Normalized top-$k$, by contrast, isn't a bound on $L_{\text{full}}$ at all — it can be zero while $L_{\text{full}}$ is arbitrarily large.

### The gradient

A short derivation (differentiate w.r.t. the selected $p_i$, note $\partial p_\tau/\partial p_i = -1$, then chain through the full softmax Jacobian) gives:

$$
\boxed{\;\frac{\partial L_{\text{MA}}}{\partial z_j} =
\begin{cases}
p_j\left(\log\dfrac{p_j}{q_j} - L_{\text{MA}}\right), & j \in S,\\[1em]
p_j\left(\log\dfrac{p_\tau}{q_\tau} - L_{\text{MA}}\right), & j \notin S.
\end{cases}\;}
$$

Two things to notice. On the head, this is **exactly the full-vocab gradient form**, with the baseline $D_{\text{KL}}(p \Vert q)$ replaced by the coarse-grained loss $L_{\text{MA}}$. On the tail, every outside token shares one log-ratio $\log(p_\tau/q_\tau)$ — the objective can't tell individual tail tokens apart, but it *does* give them a nonzero, common-signed gradient. When the student over-allocates to the tail ($\log(p_\tau/q_\tau) > L_{\text{MA}}$), all outside logits are pushed down together; when it under-allocates, they're pushed up together. The total outside mass is steered toward $q_\tau$.

### Mass-aware vs normalized, precisely

The two objectives differ by one structural choice — renormalize, or add a tail term — and everything follows from it:

- **Normalized top-$k$** compares only the conditional shape on $S$. It is invariant to head mass $P_S$, gives zero gradient outside $S$, and is *not* a bound on $L_{\text{full}}$ (zero loss is compatible with arbitrarily large $L_{\text{full}}$).
- **Mass-aware top-$k$** keeps the unnormalized head probabilities plus the tail term $p_\tau\log(p_\tau/q_\tau)$. That term is minimized at $p_\tau = q_\tau$, so head-mass mismatch is penalized (loss side) and outside logits get a collective mass-correcting gradient (gradient side) — two views of the *same* term, not two separate tricks.

The honest tradeoff: mass-aware top-$k$ still collapses the tail to one bucket, so it carries no information about *which* tail token the teacher prefers. All outside logits get the same per-unit signal, scaled only by $p_j$. It sits strictly between normalized top-$k$ and full reverse KL — recovering the head/tail mass that normalization discards, while still dropping the intra-tail structure that the full objective keeps. (Normalization's mass-invariance is occasionally what you want — e.g. if you only trust the conditional shape and consider absolute head mass unreliable — but if the goal is to approximate full reverse KL, mass-aware is the better-justified surrogate.)


### Experiment

We evaluate mass-aware top-k OPD against normalized top-k OPD in a math distillation setting. The teacher model is **Qwen3-8B**, and the student model is **Qwen3-1.7B-Base**. We use **top-k = 8** teacher probabilities for distillation. The training set is **DAPO-MATH-17K**, and the validation set during training is **MATH-500**. Unless otherwise specified, the maximum generation length is **3072** tokens.

#### Entropy Behavior

A key empirical difference between mass-aware top-k OPD and normalized top-k OPD is their entropy behavior during training.

![Entropy of mass-aware top-k OPD](/images/posts/mass-aware-topk-opd/mass-aware-entropy.png)


Entropy of mass-aware top-k OPD


![Entropy of normalized top-k OPD](/images/posts/mass-aware-topk-opd/normalized-entropy.png)
Entropy of normalized top-k OPD

Mass-aware top-k OPD maintains a healthier entropy trajectory, while normalized top-k OPD leads to extremely large entropy. This difference is important because generation at higher temperature is much more sensitive to the shape of the learned distribution. When the model distribution becomes overly flat, high-temperature sampling further amplifies this uncertainty, making the model more likely to sample low-quality tokens.

This explains why the performance gap becomes especially large at **temperature = 1.0**. Normalized top-k OPD only matches the relative probabilities inside the teacher top-k set after renormalization. It does not constrain how much total probability mass the student assigns to the teacher top-k tokens. Therefore, the student can reduce its probability mass on the teacher top-k set while still matching the normalized top-k proportions. This creates a failure mode where the loss appears small, but the student distribution becomes too diffuse over the full vocabulary.

In contrast, mass-aware top-k OPD preserves the teacher's total top-k mass. It encourages the student to match both:

1. the relative distribution among teacher top-k tokens;
2. the total probability mass assigned to the teacher top-k set.

As a result, mass-aware top-k OPD avoids the entropy explosion observed in normalized top-k OPD. This leads to more stable sampling behavior and significantly better accuracy, especially under **temperature = 1.0**.

#### Results at Temperature 1.0

At generation temperature **1.0**, mass-aware top-k OPD consistently outperforms normalized top-k OPD on AIME24, AIME25, and MATH-500.

| Dataset | Method | Pass@1 | Pass@16 | N | Avg Gen |
|---|---:|---:|---:|---:|---:|
| AIME24 | mass-aware top-k OPD | **7.92%** | **23.33%** | 30 | 16.00 |
| AIME24 | normalized top-k OPD | 5.00% | 16.67% | 30 | 16.00 |
| AIME25 | mass-aware top-k OPD | **3.96%** | **16.67%** | 30 | 16.00 |
| AIME25 | normalized top-k OPD | 2.50% | 13.33% | 30 | 16.00 |
| MATH-500 | mass-aware top-k OPD | **67.10%** | **89.60%** | 500 | 16.00 |
| MATH-500 | normalized top-k OPD | 56.67% | 82.60% | 500 | 16.00 |

The improvement is especially clear on MATH-500, where mass-aware top-k OPD improves Pass@1 from **56.67%** to **67.10%**, and Pass@16 from **82.60%** to **89.60%**. This supports the entropy-based explanation above: when normalized top-k OPD produces an overly high-entropy student distribution, sampling at temperature **1.0** becomes unstable. Mass-aware top-k OPD avoids this issue by explicitly controlling the student mass on the teacher top-k set.

#### Results at Temperature 0.7

We also evaluate the two methods at generation temperature **0.7** on a broader set of math benchmarks. Mass-aware top-k OPD still achieves better overall performance.

| Dataset | Mass-aware Avg@16 | Mass-aware Pass@16 | Normalized Avg@16 | Normalized Pass@16 |
|---|---:|---:|---:|---:|
| AIME24 | 7.92% | 23.33% | **8.33%** | 23.33% |
| AIME25 | **6.46%** | **30.00%** | 4.79% | 20.00% |
| AMC | **33.96%** | 66.27% | 31.48% | **67.47%** |
| MATH-500 | **69.01%** | **90.80%** | 68.65% | 90.60% |
| Minerva | **25.80%** | **52.57%** | 24.68% | 51.84% |
| OlympiadBench | **31.56%** | **59.56%** | 30.16% | 58.22% |
| Dataset-level average | **29.12%** | **53.75%** | 28.01% | 51.91% |

At temperature **0.7**, the gap is smaller than at temperature **1.0**, but mass-aware top-k OPD still improves the dataset-level average from **28.01%** to **29.12%** on Avg@16, and from **51.91%** to **53.75%** on Pass@16.

This result is also consistent with the entropy analysis. Lower-temperature sampling partially suppresses the harmful effect of an overly high-entropy distribution, so normalized top-k OPD becomes less unstable than at temperature **1.0**. However, mass-aware top-k OPD still gives a more reliable student distribution because it directly constrains the total top-k probability mass instead of only matching normalized top-k proportions.



## Summary

All four objectives share the same baseline-subtracted gradient structure; they differ only in *what* they sum over and *what* baseline they subtract.

| Objective | Gradient $\partial L / \partial z_j$ | Lower bound on $L_{\text{full}}$? | Mass-aware? |
|---|---|---|---|
| **Full-vocabulary** | $p_j\big(\log\frac{p_j}{q_j} - D_{\text{KL}}(p \Vert q)\big)$ | — (it *is* the target) | yes |
| **Sampled-token** | $\big(\log\frac{p_a}{q_a}\big)(\mathbf 1[j{=}a] - p_j)$, unbiased for the above | unbiased estimator | yes (in expectation) |
| **Normalized top-$k$** | $\bar p_j\big(\log\frac{\bar p_j}{\bar q_j} - D_{\text{KL}}(\bar p \Vert \bar q)\big)$ on $S$, else $0$ | **no** | **no** |
| **Mass-aware top-$k$** | $p_j\big(\log\frac{p_j}{q_j} - L_{\text{MA}}\big)$ on $S$;  $p_j\big(\log\frac{p_\tau}{q_\tau} - L_{\text{MA}}\big)$ else | **yes** (gap $= p_\tau D_{\text{KL}}(\tilde p_{\text{tail}} \Vert \tilde q_{\text{tail}})$) | **yes** |

The takeaway: if you're going to truncate to top-$k$ for cost reasons, renormalizing is the lossy choice — it silently discards your control over how much mass lives in the head. Keeping the leftover as a single tail category costs you almost nothing (one extra `logsumexp` and a `log1mexp`), turns the objective into a provable lower bound of the full reverse KL, and gives every out-of-top-$k$ logit a gradient that actually corrects the mass budget.
