> **Per-paper review — continual-learning corpus, paper 20 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§20); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 20. Cui et al. 2025 — The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models

**PDF:** `docs/project/references/continual_learning/sources/Cui et al. 2025 - The Entropy Mechanism of RL for Reasoning LLMs (preprint).pdf`
**Venue:** arXiv preprint 2505.22617v1 (28 May 2025). **Lead authors:** Ganqu Cui, Yuchen Zhang, Jiacheng Chen, et al. (Shanghai AI Lab / Tsinghua / UIUC).

**Primer connection.** Primer §5 flags this as the *policy-entropy-collapse* thread: PPO-style training can drive a policy onto a near-deterministic point it cannot recover from, a *behavioural* near-absorbing failure that compounds with plasticity loss. The primer notes the project's "eat-once-then-starve" degeneration and the modulator-temperature-head as a candidate adaptive entropy floor. This review supplies the full mechanism: the entropy-performance exchange law $R\approx-a\,e^{H}+b$ and the covariance identity $-\mathrm{d}H\propto\operatorname{Cov}(\log\pi, \pi\cdot A)$ that explains monotone collapse and motivates the two fixes (Clip-Cov, KL-Cov). §6.5(3) of the primer's verification notes: treat as a **2025 preprint** (NeurIPS 2025 acceptance not independently confirmed).

<a id="p1-cui"></a>
## Phase 1 — Foundational Overview

**The problem.** When you fine-tune a large language model with reinforcement learning to make it reason better (reward it for correct math/code answers), a reliable and damaging pattern appears: the policy's **entropy** — a measure of how much randomness/uncertainty is left in its choice of next token — **collapses toward zero within the first few hundred training steps.** The model becomes overconfident and stops exploring alternative reasoning paths. At the same moment, validation accuracy stops improving. Over 95% of both the entropy drop and the performance gain happen in the first ~1/3 of training; the remaining 2/3 of compute yields almost nothing.

**The empirical law.** Cui et al. show that *without any entropy intervention*, downstream performance $R$ and policy entropy $H$ are tied by a simple, tight, two-parameter curve:

$$R = -a\,e^{H} + b.$$

This holds across 11 base models (0.5B–32B params), 4 model families, math and code tasks, and 4 RL algorithms. Consequences: (1) you can **predict** the final performance from the first ~15% of training; (2) the **ceiling is fixed** — when entropy is exhausted ($H=0$), $R = -a + b$, and no amount of extra RL compute gets past it. So scaling RL naively hits a wall set by the entropy mechanism.

**Why entropy falls (the mechanism).** They prove the change in entropy from one step to the next equals (the negative of) the **covariance** between how likely an action already is and how much its logit is being pushed up. Under policy-gradient updates that logit push is proportional to the action's **advantage** (how much better than average it is). So: a token that is *already high-probability AND high-advantage* gets reinforced, which lowers entropy. Early in training the model is well-calibrated (confident tokens really are good), so this covariance is large and positive — entropy plummets. It stays positive throughout, so entropy keeps falling.

**The fix.** Since a *tiny fraction* of "pivotal" high-covariance tokens drive the collapse, restrain *just those*. Two simple surgical methods:
- **Clip-Cov** — randomly detach (stop-gradient) a small fraction of high-covariance tokens so they don't contribute to the update.
- **KL-Cov** — apply a KL penalty to the top-covariance tokens, softly holding them near the old policy.

Both keep entropy an order of magnitude higher, sustain exploration, lengthen responses, and beat vanilla GRPO by +2.0% (7B) and +6.4% (32B) on math benchmarks. Naive entropy-bonus or reference-KL regularization *fails* (hyper-parameter-sensitive or performance-degrading).

**Initial takeaway.** RL for reasoning LLMs "trades entropy for reward" on a predictable exponential curve; the trade is driven by a small set of high-covariance tokens; controlling those tokens breaks the ceiling. For our project the transferable idea is that **entropy collapse is a covariance-driven near-absorbing state**, and the right lever is *targeted* control of the actions that dominate the covariance — not a blunt global entropy bonus.

<a id="p2-cui"></a>
## Phase 2 — Graduate-Level Deep Dive

### 2.1 Setup and definitions

An LLM policy $\pi_\theta$ autoregressively generates $y=\{y_1,\dots,y_T\}$ from prompt $x$; RL maximizes verifier reward
$$\max_\theta\ J(\theta) := \mathbb{E}_{x\sim D,\,y\sim\pi_\theta(x)}\,[\,r(y)\,].$$
The policy-gradient estimator (Williams 1992):
$$\nabla_\theta J(\theta) = \mathbb{E}_{x\sim D,\,y\sim\pi_\theta(x)}\!\left[\sum_{t=0}^{T}\nabla_\theta \log\pi_\theta(y_t\mid y_{<t})\,A_t\right].$$
Advantage $A_t$ varies by algorithm: REINFORCE uses $A_t=r(y)$; **GRPO** normalizes within a group of $K$ samples,
$$A_t = \frac{r(y) - \operatorname{mean}\big(r(y^{1:K})\big)}{\operatorname{std}\big(r(y^{1:K})\big)};$$
**PPO** optimizes the clipped surrogate
$$L(\theta)=\mathbb{E}_t\!\left[\min\!\Big(\tfrac{\pi_\theta(y_t\mid y_{<t})}{\pi_{\theta_{old}}(y_t\mid y_{<t})}A_t,\ \operatorname{clip}\big(\tfrac{\pi_\theta}{\pi_{\theta_{old}}},1-\epsilon,1+\epsilon\big)A_t\Big)\right].$$

**Policy entropy** (token-level, averaged over data):
$$H(\pi_\theta, D) = -\mathbb{E}_{D,\pi_\theta}\big[\log\pi_\theta(y_t\mid y_{<t})\big] = -\frac{1}{|D|}\sum_{x\in D}\frac{1}{|y|}\sum_{t=1}^{|y|}\mathbb{E}_{y_t\sim\pi_\theta}\big[\log\pi_\theta(y_t\mid y_{<t},x)\big].$$

### 2.2 The entropy–performance law and its corollaries

Empirically (over 200+ data points per curve, fit with just 2 coefficients):
$$\boxed{\,R = -a\,e^{H} + b\,}$$
Differentiating,
$$\frac{\mathrm{d}R}{\mathrm{d}H} = -a\,e^{H},$$
so $a$ is the **conversion rate** of entropy into performance (larger $a$ = trades more efficiently). At full exhaustion $H=0$:
$$R_{\max} = -a + b,$$
the deterministic ceiling. Both $a$ and $b$ are found to be **algorithm-irrelevant** (GRPO, RLOO, PRIME, REINFORCE++ collapse onto one curve → $a,b$ are intrinsic to *model + data*), and both vary **log-linearly with model size**, enabling extrapolation of a large model's ceiling from small-model runs.

### 2.3 Entropy dynamics — the covariance identity (Lemma 1) with full derivation

Consider a **tabular softmax** policy, each state-action pair with its own logit $z_{s,a}=\theta_{s,a}$:
$$\pi_\theta(a\mid s)=\frac{\exp(z_{s,a})}{\sum_{a'\in A}\exp(z_{s,a'})}.$$

**Lemma 1 (entropy difference of softmax policy).** Under a first-order (small-$\eta$) update,
$$H(\pi_\theta^{k+1}\mid s) - H(\pi_\theta^{k}\mid s) \approx -\operatorname{Cov}_{a\sim\pi_\theta^k(\cdot\mid s)}\!\Big(\log\pi_\theta^k(a\mid s),\ z^{k+1}_{s,a}-z^{k}_{s,a}\Big).$$

*Derivation (step by step).* First-order Taylor expansion of entropy along the logit update $z^{k+1}=z^k+\eta\nabla J$:
$$H(\pi^{k+1}_\theta\mid s)\approx H(\pi^k_\theta\mid s) + \big\langle \nabla H(\pi^k_\theta\mid s),\ z^{k+1}-z^k\big\rangle.$$
Compute the entropy gradient. With $H=-\mathbb{E}_{a\sim\pi}[\log\pi]$,
$$\nabla_\theta H(\pi_\theta\mid s) = -\mathbb{E}_{a\sim\pi}\big[\nabla_\theta\log\pi_\theta(a\mid s) + \log\pi_\theta(a\mid s)\,\nabla_\theta\log\pi_\theta(a\mid s)\big].$$
The first term vanishes because $\mathbb{E}_{a\sim\pi}[\nabla_\theta\log\pi_\theta(a\mid s)] = \nabla_\theta\!\sum_a\pi = \nabla_\theta 1 = 0$. Hence
$$\nabla_\theta H(\pi_\theta\mid s) = -\mathbb{E}_{a\sim\pi}\big[\log\pi_\theta(a\mid s)\,\nabla_\theta\log\pi_\theta(a\mid s)\big].$$
Insert the **softmax log-derivative** (Lemma 2): $\dfrac{\partial\log\pi_\theta(a\mid s)}{\partial\theta_{s,a'}} = \mathbf{1}\{a=a'\}-\pi_\theta(a'\mid s)$. Then
$$\big\langle\nabla_\theta H, z^{k+1}-z^k\big\rangle = -\mathbb{E}_{a\sim\pi}\!\left[\log\pi(a\mid s)\sum_{a'}\big(\mathbf{1}\{a=a'\}-\pi(a'\mid s)\big)\big(\theta^{k+1}_{s,a'}-\theta^{k}_{s,a'}\big)\right].$$
The inner sum equals $(\theta^{k+1}_{s,a}-\theta^k_{s,a}) - \sum_{a'}\pi(a'\mid s)(\theta^{k+1}_{s,a'}-\theta^k_{s,a'}) = (z^{k+1}_{s,a}-z^k_{s,a}) - \mathbb{E}_{a'\sim\pi}[z^{k+1}_{s,a'}-z^k_{s,a'}]$. Substituting and recognizing the structure $\mathbb{E}[\,X\cdot(Y-\mathbb{E}Y)\,]=\operatorname{Cov}(X,Y)$ with $X=\log\pi(a\mid s)$ (and using $\mathbb{E}[\log\pi\cdot\mathbb{E}(\Delta z)]=\mathbb{E}[\log\pi]\cdot\mathbb{E}[\Delta z]$):
$$\big\langle\nabla_\theta H, z^{k+1}-z^k\big\rangle = -\operatorname{Cov}_{a\sim\pi}\big(\log\pi(a\mid s),\ z^{k+1}_{s,a}-z^k_{s,a}\big).\qquad\blacksquare$$

Interpretation: entropy falls when actions that already have **high log-probability** get their **logits increased** — the covariance is positive.

### 2.4 Coupling to advantage (Proposition 1, Theorem 1) with derivation

**Proposition 1 (logit change under vanilla PG).** With tabular softmax updated by $z^{k+1}_{s,a}=z^k_{s,a}+\eta\,\nabla_{\theta_{s,a}}J(\theta)$,
$$z^{k+1}_{s,a}-z^{k}_{s,a} = \eta\,\pi_\theta(a\mid s)\,A(s,a).$$

*Derivation.* $\nabla_{\theta_{s,a}}J = \mathbb{E}_{a'\sim\pi}\big[\nabla_{\theta_{s,a}}\log\pi(a'\mid s)\,A(s,a')\big] = \sum_{a'}\pi(a'\mid s)\big(\mathbf{1}\{a=a'\}-\pi(a\mid s)\big)A(s,a')$ (Lemma 2). Expand:
$$= \pi(a\mid s)\Big[(1-\pi(a\mid s))A(s,a) - \!\!\sum_{a'\neq a}\!\pi(a'\mid s)A(s,a')\Big] = \pi(a\mid s)\Big[A(s,a) - \sum_{a'}\pi(a'\mid s)A(s,a')\Big].$$
The bracketed baseline $\sum_{a'}\pi(a'\mid s)A(s,a') = \mathbb{E}_{a'\sim\pi}[A(s,a')] = 0$ (Lemma 3: advantage has zero mean under $\pi$). Thus $\nabla_{\theta_{s,a}}J=\pi(a\mid s)A(s,a)$ and the result follows. $\blacksquare$

**Theorem 1 (entropy change under PG).** Substituting Proposition 1 into Lemma 1:
$$H(\pi^{k+1}_\theta\mid s) - H(\pi^{k}_\theta\mid s) \approx -\eta\,\operatorname{Cov}_{a\sim\pi_\theta^k(\cdot\mid s)}\!\Big(\log\pi^k_\theta(a\mid s),\ \pi^k_\theta(a\mid s)\,A(s,a)\Big).$$

**Theorem 2 (entropy change under natural PG).** For NPG the logit change is simply $z^{k+1}_{s,a}-z^k_{s,a}=\eta\,A(s,a)$ (from Agarwal et al. 2021), giving the cleaner
$$H(\pi^{k+1}_\theta\mid s) - H(\pi^{k}_\theta\mid s) \approx -\eta\,\operatorname{Cov}_{a\sim\pi_\theta^k(\cdot\mid s)}\!\Big(\log\pi^k_\theta(a\mid s),\ A(s,a)\Big).$$

**Conclusion of the analysis:** a strong positive correlation between an action's probability $\pi(a)$ and its advantage $A(a)$ drives entropy *down*; a negative correlation drives it *up*. A rare (low-prob) high-advantage action would *raise* entropy — which is exactly what exploration needs.

### 2.5 Empirical verification

On on-policy GRPO (Qwen2.5-7B, bandit view: prompt = state, whole response = action), the measured covariance $\operatorname{Cov}(\cdot)$ and the negative entropy difference $-\mathrm{d}H$ track each other almost exactly over 2000 steps (Fig. 8 left) — direct confirmation of Theorem 1. $\operatorname{Cov}(\cdot)$ stays positive throughout (→ monotone entropy decrease) and is *larger for easy/high-accuracy prompts* (well-calibrated → strong prob-advantage alignment) and *smaller for hard prompts* (Fig. 8 right). This difficulty-dependence of the covariance is the sharpest empirical handle in the paper.

### 2.6 The two interventions (Clip-Cov, KL-Cov)

A small fraction of tokens carry outsized covariance (Table 1: top 0.02% of tokens have mean covariance $5.65$ vs. overall $0.003$). Define the token-wise centered cross-product estimator over a batch of $N$ rollout tokens:
$$\operatorname{Cov}(y_i) = \Big(\log\pi_\theta(y_i) - \tfrac{1}{N}\textstyle\sum_j \log\pi_\theta(y_j)\Big)\cdot\Big(A(y_i) - \tfrac{1}{N}\textstyle\sum_j A(y_j)\Big).$$

**Clip-Cov.** Uniformly sample a small fraction $r$ of tokens whose covariance lies in a high band $[\omega_{low},\omega_{high}]$ (both $\gg$ average, $>500\times$):
$$I_{clip} = I\sim\operatorname{Uniform}\big(\{i\mid \operatorname{Cov}(y_i)\in[\omega_{low},\omega_{high}]\},\ \lfloor r\cdot N\rfloor\big),$$
and **detach** those tokens from the gradient:
$$L_{\text{Clip-Cov}}(\theta) = \begin{cases}\mathbb{E}_t\big[\tfrac{\pi_\theta(y_t\mid y_{<t})}{\pi_{\theta_{old}}(y_t\mid y_{<t})}A_t\big], & t\notin I_{clip}\\[4pt] 0, & t\in I_{clip}\end{cases}$$

**KL-Cov.** Select the top-$k$ proportion by covariance, $I_{KL}=\{i\mid \operatorname{Rank}(\operatorname{Cov}(y_i))\le k\cdot N\}$, $k\ll 1$, and apply a KL penalty on those tokens:
$$L_{\text{KL-Cov}}(\theta) = \begin{cases}\mathbb{E}_t\big[\tfrac{\pi_\theta}{\pi_{\theta_{old}}}A_t\big], & t\notin I_{KL}\\[4pt] \mathbb{E}_t\big[\tfrac{\pi_\theta}{\pi_{\theta_{old}}}A_t - \beta\,D_{KL}(\pi_{\theta_{old}}\,\|\,\pi_\theta)\big], & t\in I_{KL}\end{cases}$$

**Results.** With $r=2\times10^{-4}$ (Clip-Cov) or $k=2\times10^{-3}$/$2\times10^{-4}$ and $\beta=1$ (KL-Cov), both beat GRPO and the clip-higher baseline: +2.0% avg (7B), +6.4% avg (32B), with +15.0%/+14.6% on the hardest AIME24/AIME25 for 32B. Entropy stays $>10\times$ higher than the collapsing baseline; response length grows; no plateau. Entropy is *tunable* by $r$ or $\beta$ (Fig. 12); KL-Cov gives stabler entropy curves than Clip-Cov. Naive entropy-loss and reference-KL both fail (Figs. 9–10). Connection to clip-higher: raising PPO's upper clip $\epsilon$ implicitly admits more *low-covariance* (low-prob, high-advantage) tokens — Cui et al. make the covariance the *explicit* control variable instead.

### 2.7 Relevance to the project

The mechanism is architecture-agnostic (it holds for any softmax policy, which includes the project's categorical action heads). Two transferable points: (1) entropy collapse is not random decay but a *covariance-driven* near-absorbing dynamic, so an "eat-once-then-starve" degeneracy can be read as high-covariance reinforcement of an early-dominant action; (2) the effective remedy is *targeted* suppression of the pivotal high-covariance actions, which is a sharper instrument than a global entropy bonus (shown here to fail). The primer's proposed modulator-temperature-head as an adaptive entropy floor is consistent with this — but Cui et al.'s finding is that *where* you spend the entropy budget (which tokens/actions) matters more than the global level.

<a id="bb-cui"></a>
## Appendix: Section-by-Section Backbone

**§1 Introduction.** RL for reasoning LLMs faces policy-entropy collapse: entropy drops to ~0 in a few steps, performance saturates. Establishes empirical law $R=-a\,e^{H}+b$ (fully predictable, fixed ceiling at $H=0$). Two corollaries: (1) exploitation-exploration curve is predetermined (predictable like scaling laws); (2) upper bound deterministic → naive RL-compute scaling has marginal return. Naive entropy regularization fails. Motivates mechanistic analysis + covariance-based control.

**§2 The Predictable "Collapse" of Policy Entropy.**
- **§2.1 Preliminaries.** Objective $J(\theta)=\mathbb{E}[r(y)]$; PG estimator; GRPO advantage (group normalization); PPO clipped surrogate; token-level entropy definition.
- **§2.2 Settings.** 4 model families, 11 base models (0.5–32B), math+code (8 benchmarks), 4 RL algorithms (GRPO, REINFORCE++, PRIME, RLOO), veRL framework, "Zero" setting, KL coef 0 by default.
- **§2.3 First glance.** Entropy sharp-drops and monotonically →0; performance rises then saturates. 73% entropy consumption + 76% performance gain in first 200/2400 steps; 93%/94% by step 800.
- **§2.4 Fitting curves.** $R=-a\,e^{H}+b$ fits all runs with 2 coefficients. Can predict final performance from first 15% of steps (RMSE ~0.5–1.9%). At $H=0$, $R=-a+b$.
- **§2.5 Understanding coefficients.** $a,b$ are algorithm-irrelevant (GRPO/RLOO/PRIME/REINFORCE++ share a curve) → intrinsic to model+data. $\mathrm{d}R/\mathrm{d}H=-a\,e^H$ ($a$=conversion rate; $-a+b$=max). $a,b$ vary log-linearly with model size → extrapolate large from small.
- **§2.6 Discussion.** Predictability echoes scaling laws but is not universal (off-policy / different policies differ). Conditionally supports the "RL only elicits pretrained behaviors / ceiling exists" claim, but attributes the ceiling to the entropy mechanism, not an intrinsic RL limit.

**§3 Dynamics Analysis of Policy Entropy.**
- **§3.1 Softmax policy.** **Lemma 1**: $H^{k+1}-H^k\approx-\operatorname{Cov}(\log\pi, \Delta z)$.
- **§3.2 Under PG / NPG.** **Proposition 1**: $\Delta z_{s,a}=\eta\,\pi(a\mid s)A(s,a)$. **Theorem 1**: $\Delta H\approx-\eta\operatorname{Cov}(\log\pi,\pi A)$. **Theorem 2** (NPG): $\Delta H\approx-\eta\operatorname{Cov}(\log\pi,A)$. Positive prob-advantage correlation → entropy down.
- **§3.3 Empirical verification.** On-policy GRPO on Qwen2.5-7B; $\operatorname{Cov}(\cdot)$ and $-\mathrm{d}H$ track exactly (Fig. 8L); Cov stays positive; higher for easy prompts, lower for hard (Fig. 8R).

**§4 Entropy Control by Covariance Regularization.**
- **§4.1 Effect of entropy regularization.** Entropy loss hyper-parameter-sensitive (too small→no effect, too large→explosion); reference-KL stabilizes entropy but degrades performance. Naive methods insufficient.
- **§4.2 Suppressing high-covariance tokens.** Table 1: top 0.02% tokens dominate covariance. Token-wise covariance estimator (Eq. 10). **Clip-Cov** (detach random high-cov tokens, Eqs. 11–12) and **KL-Cov** (KL-penalize top-k cov tokens, Eqs. 13–14). Pseudocode Listing 1 (a few lines of change).
- **§4.3 Experiments.** Qwen2.5-7B/32B on DAPO-MATH; beat GRPO (+2.0%/+6.4% avg; +15%/+14.6% on AIME for 32B); entropy $>10\times$ higher; longer responses; no plateau; more stable than clip-higher (Table 2, Fig. 11).
- **§4.4 Controlled entropy.** Entropy tunable via clip ratio $r$ or KL coef $\beta$ (Fig. 12); KL-Cov stabler.
- **§4.5 Discussion.** Clip-higher = implicitly adding low-covariance tokens; Cov is the direct control. A few pivotal tokens ($10^{-4}$–$10^{-3}$) control entropy; optimal entropy value still open.

**§5 Related Work.** Maximum-entropy RL lineage; predictability / scaling-laws / reward-model overoptimization; RL for LLM post-training.

**§6 Conclusion.** Performance gains bought with exploratory capacity → foreseeable ceiling; covariance-based Clip-Cov / KL-Cov counteract collapse; scaling RL needs more than entropy minimization.

**Appendix E (proofs).** Lemma 2 (softmax log-derivative), Lemma 3 ($\mathbb{E}_\pi[A]=0$), full proofs of Lemma 1, Proposition 1, Theorem 2 (reproduced in Phase 2 above).

---
