---
title: "Statistics tutorial — comparing two models on a behavior metric"
topic: behavior_measures
status: active
created: 2026-06-30
last_updated: 2026-06-30
aliases: [stats-tutorial, model-comparison-statistics]
---

# Statistics tutorial — comparing two models on a behavior metric

## Purpose (read this first)

When we measure a behavior metric (say **spatial spread** / radius of gyration) on two
agents — a mature model and an early one — and ask *"are they different?"*, a single
p-value is not enough. This doc explains, in plain language **and** with the math, every
statistical tool used in the behavior-measure study's model comparisons:

- **Two-sample tests** (Welch's t, Mann–Whitney U): is group A different from group B?
- **Paired tests** (paired t, Wilcoxon signed-rank): is the A→B shift *consistent* across conditions?
- **Effect size** (Cohen's d, pooled SD): *how big* is the difference, independent of sample size?
- **Multiple comparisons** (Bonferroni): how to not fool yourself when running many tests.
- **The big lesson**: why "statistically significant" can be *meaningless* when the probe is
  near-deterministic, and what to look at instead.

It assumes you know the basics (null hypothesis, p-value) and fills in the "advanced" pieces.
Math renders in VSCode/Obsidian (KaTeX) — not in a raw terminal.

> **The one-sentence takeaway:** a p-value answers *"am I sure there's a difference?"*; an
> **effect size** answers *"is the difference big enough to care?"* — and when your measurement
> has tiny noise, p-values become tiny for trivial differences, so you must judge by effect size
> and absolute magnitude, not p.

---

## 0. The shared skeleton of every test

Every significance test computes a **test statistic** of the form

$$\text{statistic} = \frac{\text{signal (how far apart the groups look)}}{\text{noise (how much they'd vary by chance)}}$$

then asks: *if the null hypothesis $H_0$ (no real difference) were true, how often would I see a
statistic at least this extreme?* That tail probability is the **p-value**. Small p ⇒ the data are
surprising under $H_0$ ⇒ we reject $H_0$. The tests below differ only in how they define "signal"
and "noise," and in what they assume about the data.

Notation: $\bar x$ = sample mean, $s^2$ = sample variance, $s$ = standard deviation, $n$ = sample
size, $N = n_1 + n_2$.

---

## 1. Effect size vs. significance (the most important idea)

Two distinct questions:

| Question | Answered by | Depends on sample size $n$? |
|---|---|---|
| *Am I sure there is a difference?* | p-value | **Yes** — more data ⇒ smaller p |
| *How big is the difference?* | effect size (Cohen's d) | **No** |

A difference can be **statistically significant but trivially small** (huge $n$ or tiny noise), or
**large but not significant** (tiny $n$). Always report both. Section 6 shows how this exact trap
appeared in our data.

---

## 2. Two-sample tests — "is group A different from group B?"

Used per condition, with $n=30$ episodes per model. We run **both** a parametric and a
non-parametric test and check they agree.

### 2.1 Welch's t-test (parametric)

**Idea.** Compare the two group **means**, *without* assuming the two groups have equal variance.
(The classic Student t-test assumes equal variance; Welch relaxes that and is the safer default.)

**Math.**

$$t = \frac{\bar x_1 - \bar x_2}{\sqrt{\dfrac{s_1^2}{n_1} + \dfrac{s_2^2}{n_2}}}$$

The numerator is the **signal** (gap between means); the denominator is the **standard error of
that gap** (the noise). Degrees of freedom are approximated (Welch–Satterthwaite):

$$\nu \approx \frac{\left(\dfrac{s_1^2}{n_1}+\dfrac{s_2^2}{n_2}\right)^{2}}
{\dfrac{(s_1^2/n_1)^2}{n_1-1}+\dfrac{(s_2^2/n_2)^2}{n_2-1}}$$

Compare $t$ to a Student-$t$ distribution with $\nu$ degrees of freedom to get the p-value.

**Assumptions / reading.** Assumes the *sample means* are roughly normal — true for $n=30$ by the
Central Limit Theorem even if the raw data are not. Large $|t|$ ⇒ small p.

### 2.2 Mann–Whitney U test (non-parametric, a.k.a. Wilcoxon rank-sum)

**Idea.** Don't compare means at all — ask *"does one group tend to produce larger values?"* using
only the **ranks** of the data. No normality assumption; robust to outliers and odd shapes. We
include it because behavior metrics are often skewed or nearly constant, where the t-test's
normality assumption is shaky.

**Math.** Pool all $N$ values, rank them $1,\dots,N$. Let $R_1$ = sum of the ranks that belong to
group 1. Then

$$U_1 = R_1 - \frac{n_1(n_1+1)}{2}, \qquad U_2 = n_1 n_2 - U_1, \qquad U = \min(U_1, U_2).$$

Interpretation: $U_1$ counts, over all $n_1 \times n_2$ cross-pairs, how many times a group-1 value
beats a group-2 value. Under $H_0$ the two groups are interchangeable, so that count should be about
half of all pairs:

$$\mathbb{E}[U] = \frac{n_1 n_2}{2}, \qquad \operatorname{Var}(U) = \frac{n_1 n_2 (N+1)}{12}.$$

For large samples, $z = \dfrac{U - \mathbb{E}[U]}{\sqrt{\operatorname{Var}(U)}}$ is approximately
standard normal and gives the p-value.

**Reading.** Tests **stochastic dominance** (roughly, a median shift), not the mean. If Welch and
Mann–Whitney agree, the result is not an artifact of the normality assumption.

---

## 3. Paired tests — "is the shift consistent across conditions?"

The per-condition tests treat the two models as independent samples *within one condition*. But we
also ask: **across all 12 conditions, does model B sit systematically higher than model A?**
Conditions differ a lot in baseline value (e.g. no-animal vs predator), so we **pair** them — each
condition contributes one difference, cancelling the between-condition variation.

### 3.1 Paired t-test

**Idea.** For $m$ paired conditions, compute the within-pair difference and test whether its mean is
zero.

**Math.** With $d_i = x_i^{B} - x_i^{A}$ for condition $i$, run a one-sample t-test on the $d_i$:

$$t = \frac{\bar d}{s_d / \sqrt{m}}, \qquad \text{df} = m - 1,$$

where $\bar d$ and $s_d$ are the mean and SD of the differences and $m$ is the number of pairs.

**Why pair?** Pairing removes the huge condition-to-condition variability, so you see only the
consistent model shift — far more powerful than treating $2m$ numbers as independent.

### 3.2 Wilcoxon signed-rank test (non-parametric paired)

**Idea.** The rank-based counterpart of the paired t-test — robust if the differences aren't normal.

**Math.** Take the differences $d_i$, drop any zeros, and rank their **absolute values**
$|d_i|$ as $1,\dots,m$. Let

$$W^{+} = \!\!\sum_{d_i > 0}\!\operatorname{rank}(|d_i|), \qquad
W^{-} = \!\!\sum_{d_i < 0}\!\operatorname{rank}(|d_i|), \qquad W = \min(W^{+}, W^{-}).$$

Under $H_0$ (differences symmetric about 0):

$$\mathbb{E}[W^{+}] = \frac{m(m+1)}{4}, \qquad
\operatorname{Var}(W^{+}) = \frac{m(m+1)(2m+1)}{24},$$

and a normal approximation gives the p-value.

**Reading.** Like a sign test, but it weights bigger differences more (via their ranks). Answers
"is B reliably higher than A across conditions?"

---

## 4. Effect size — "how big, regardless of n?"

### 4.1 Pooled standard deviation

The common "ruler," combining both groups weighted by their degrees of freedom:

$$s_{\text{pooled}} = \sqrt{\frac{(n_1 - 1)\,s_1^2 + (n_2 - 1)\,s_2^2}{n_1 + n_2 - 2}}.$$

### 4.2 Cohen's d

$$d = \frac{\bar x_2 - \bar x_1}{s_{\text{pooled}}}.$$

**Idea.** The mean gap measured **in units of standard deviation**. $d = 1$ means the means are one
full SD apart; $d = 2$ means two SDs apart (the two bell curves barely overlap). Rough conventions:

| $|d|$ | ~0.2 | ~0.5 | ~0.8 | $\geq 1.5$ |
|---|---|---|---|---|
| meaning | small | medium | large | very large (little overlap) |

**Key property.** $d$ does **not** grow with sample size, whereas $t \propto \sqrt{n}$. So $d$ tells
you *how separated* the distributions are; p tells you *how sure* you are they're separated at all.
These are different questions — report both.

---

## 5. Multiple comparisons — Bonferroni

**Idea.** Run $k$ independent tests each at $\alpha = 0.05$ and the chance of **at least one** false
positive is $1 - (1-0.05)^k$ — for $k=12$ that is ~46%, not 5%. Bonferroni controls this
**family-wise error rate**.

**Math.** Use the stricter per-test threshold

$$\alpha_{\text{per-test}} = \frac{\alpha}{k}.$$

For 12 tests at family-wise $\alpha = 0.05$: $0.05/12 \approx 0.0042$. A test "survives Bonferroni"
if $p < 0.0042$. It is deliberately **conservative** — it trades statistical power (more false
negatives) for protection against fluke positives. (Less conservative alternatives exist, e.g.
Holm–Bonferroni or Benjamini–Hochberg FDR, but Bonferroni is the simplest to state.)

---

## 6. The punchline: why significance is "cheap" with a near-deterministic probe

Take Welch's $t$ when both groups share roughly variance $s^2$:

$$t \approx \frac{\bar x_1 - \bar x_2}{s\,\sqrt{\dfrac{1}{n_1}+\dfrac{1}{n_2}}}.$$

As the within-model variance $s \to 0$ (a deterministic-ish probe with only tiny initial-state
jitter), the denominator $\to 0$, so $t \to \infty$ and $p \to 0$ for **any** non-zero mean gap —
even a behaviorally meaningless one. Cohen's $d = \text{gap}/s_{\text{pooled}}$ inflates the same way
($d \to \infty$ as $s_{\text{pooled}} \to 0$).

**Worked example (our spatial-spread test).** For the no-animal · inj0 condition the radius of
gyration was $3.19 \pm 0.03$ (mature) vs $3.67 \pm 0.11$ (early). A gap of $0.47$ cells with a std
of $\pm 0.03$ gives $d \approx 5.7$ and $p \sim 10^{-21}$ — an *absurd* "effect" that is really just
a measurement so precise that any wobble looks gigantic.

So **p-values and even Cohen's d are inflated** when the probe is near-deterministic. The honest
substitutes are:

1. **Absolute magnitude in real units** — e.g. R_g differs by ~0.6 cells.
2. **Overlap relative to a reference scale** — both models roam ~4–5 cells (R_g), so a 0.6-cell gap
   is a *faint* separation; whereas **bush-use** differs *categorically* (100% vs 13% entered-bush,
   no overlap). On that scale, spatial spread is a **weak** discriminator and bush-use is a **strong**
   one — even though both are "significant" at $p \ll 0.0042$.

This is the classic **statistical vs. practical significance** distinction, and a near-deterministic
RL probe is a textbook case where they diverge. **Decide with effect size + magnitude, then use the
p-value only as a sanity gate.**

---

## 7. Quick-reference cheat sheet

| Tool | Question | Parametric? | Statistic (signal / noise) |
|---|---|---|---|
| Welch's t | Do two independent groups have different means? | yes (means ~normal) | mean gap / SE of gap |
| Mann–Whitney U | Does one group tend to exceed the other? | no (ranks) | rank-sum vs chance |
| Paired t | Is the within-pair mean difference ≠ 0? | yes | mean diff / SE of diff |
| Wilcoxon signed-rank | Are paired differences centred above/below 0? | no (ranks) | signed rank-sum vs chance |
| Cohen's d | How big is the gap, in SD units? | — (effect size) | mean gap / pooled SD |
| Bonferroni | Avoid false positives over many tests | — (correction) | threshold $\alpha/k$ |

**Rule of thumb for this project:** report Welch p **and** Mann–Whitney p (agreement = robust),
**always** report Cohen's d and the absolute gap, use the paired test for "systematic across
conditions," apply Bonferroni when scanning many conditions — and **never** read a p-value from a
near-deterministic probe without the effect size beside it.

---

## References

- Used in the spatial-spread model comparison for the behavior-measure study; see
  [[interoceptive_behavior_measure_study]] (anchor doc) and the avoidance heatmaps under
  `results/eval/avoidance_stat{,_randpred}/STATS/`.
- Tooling: `scripts/behavior_measures/` (a significance helper that reports all of the above
  side-by-side may be added).
- Standard sources: Welch (1947) on unequal-variance t; Mann & Whitney (1947); Wilcoxon (1945);
  Cohen, *Statistical Power Analysis for the Behavioral Sciences* (1988) for d conventions.
