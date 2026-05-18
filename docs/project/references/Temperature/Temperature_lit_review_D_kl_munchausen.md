# Temperature corpus — KL-regularized RL & Munchausen family

**Sub-topic of:** [Temperature corpus](./) — the project's reference library on entropy/KL/softmax temperature in RL.
**Scope of this file:** the three papers that form the *theoretical* axis of the corpus — KL-regularized Bellman analysis, its practical Munchausen distillation, and the Tsallis-KL generalization. The corpus's other axis (SAC foundations, SAC variants, softmax operators) lives in adjacent reviews in this folder.

---

## What this collection is about (plain-language entry point)

**KL-regularized reinforcement learning** means: instead of letting the policy jump freely toward whichever action looks best under the current Q-estimate, you add a soft penalty for *moving too far from a reference policy* (usually the previous iteration's policy). The penalty is the Kullback–Leibler divergence — a standard measure of how different two probability distributions are. In plain English: "small steps, not big jumps". That's the practical surface.

The deeper claim this collection makes is that KL regularization is **not just a stability hack** — it is a **robustness mechanism with a precise mathematical interpretation**. The first paper (Vieillard 2020a, *Leverage the Average*) proves a theorem that makes this concrete: when you KL-regularize toward the previous policy at every iteration, the resulting policy ends up depending on a (geometric, exponentially-weighted) **average of all past Q-value estimates** rather than only the latest. If your Q-estimates are noisy but unbiased, that average cancels noise — by the law of large numbers, the algorithm converges to the optimum even when an unregularized algorithm would not. KL regularization is therefore a kind of **free implicit ensembling over training iterations**, with rigorous error-propagation bounds: linear (not quadratic) in the horizon, and averaging (not summing) the per-iteration errors.

The second paper (Vieillard 2020b, *Munchausen RL*) turns this theory into a **one-line code change**: add a bonus $\alpha\tau\log\pi(a|s)$ — the log-probability of the chosen action, scaled — to the reward. No new network, no new loss term, no auxiliary policy head. They prove this implicit-bootstrap is exactly equivalent to running KL+entropy regularized value iteration, and the resulting Munchausen-DQN beats Rainbow on Atari.

The third paper (Zhu 2023, *Tsallis-KL Munchausen*) generalizes the divergence itself: replace Shannon KL with **Tsallis q-divergence**. At $q=1$ you recover the standard Munchausen. At $q>1$ you get *sparse* policies — the optimal policy puts exact zero probability on bad actions — which the paper shows helps in environments with many low-value or distractor actions.

**Why this collection matters for the project.** The project's grid-world-pain agents train with a learned modulator that adjusts policy "temperature" online based on interoceptive signals. Two things from this collection are directly load-bearing: (1) The Vieillard theorem shows that any KL term *to the previous policy* is doing implicit Q-averaging — so a modulator that scales the KL coefficient is effectively scaling how aggressively the agent ensembles its own past iterations. That is a robustness lever, not just a stability lever. (2) The Munchausen trick is the cheapest possible implementation path — it requires only a reward-bonus computed from the current policy, with no architectural change. If the modulator's output were to scale $\alpha$ or $\tau$ in the Munchausen reward bonus, the modulator becomes a direct controller of the implicit averaging strength. Zhu's Tsallis extension matters for the sparse-policy regime, e.g. environments with many irrelevant action choices.

The rest of this entry-point note is structured paper-by-paper. Each paper has a section-ordered backbone appendix at the end and a Phase 1 / Phase 2 synthesis up front.

---

## Table of Contents

1. [Paper 1 — Vieillard et al. 2020 (NeurIPS) — *Leverage the Average: An Analysis of KL Regularization in RL*](#paper-1--vieillard-et-al-2020--leverage-the-average)
   - 1.1 [Phase 1 — Foundational Overview](#11-phase-1--foundational-overview-vieillard-kl-reg)
   - 1.2 [Phase 2 — Graduate-Level Deep Dive](#12-phase-2--graduate-level-deep-dive-vieillard-kl-reg)
   - 1.3 [Appendix — Section-by-Section Backbone](#13-appendix--section-by-section-backbone-vieillard-kl-reg)
2. [Paper 2 — Vieillard et al. 2020 (NeurIPS) — *Munchausen Reinforcement Learning*](#paper-2--vieillard-et-al-2020--munchausen-rl)
   - 2.1 [Phase 1 — Foundational Overview](#21-phase-1--foundational-overview-munchausen)
   - 2.2 [Phase 2 — Graduate-Level Deep Dive](#22-phase-2--graduate-level-deep-dive-munchausen)
   - 2.3 [Appendix — Section-by-Section Backbone](#23-appendix--section-by-section-backbone-munchausen)
3. [Paper 3 — Zhu et al. 2023 — *General Munchausen RL with Tsallis-KL Divergence*](#paper-3--zhu-et-al-2023--tsallis-kl-munchausen)
   - 3.1 [Phase 1 — Foundational Overview](#31-phase-1--foundational-overview-tsallis-kl)
   - 3.2 [Phase 2 — Graduate-Level Deep Dive](#32-phase-2--graduate-level-deep-dive-tsallis-kl)
   - 3.3 [Appendix — Section-by-Section Backbone](#33-appendix--section-by-section-backbone-tsallis-kl)
4. [Cross-paper structural connection](#cross-paper-structural-connection)
5. [Where this connects to the project's modulator-conditioned temperature](#where-this-connects-to-the-projects-modulator-conditioned-temperature)

---

## Paper 1 — Vieillard et al. 2020 — *Leverage the Average*

**Full title:** Leverage the Average: an Analysis of KL Regularization in Reinforcement Learning
**Authors:** Nino Vieillard, Tadashi Kozuno, Bruno Scherrer, Olivier Pietquin, Rémi Munos, Matthieu Geist
**Venue:** NeurIPS 2020
**PDF:** `docs/project/references/Temperature/sources/Vieillard et al. 2020 - Leverage the average - An analysis of KL regularization in reinforcement learning.pdf`

### 1.1 Phase 1 — Foundational Overview (Vieillard KL-reg)

**Introduction — what problem this paper solves.**
A lot of the best modern deep-RL algorithms — TRPO, MPO, SAC — all share one ingredient: at each update, they penalize the new policy for *moving too far* from the previous one, using the Kullback–Leibler divergence as the "distance". Everyone in the field knew this trick worked empirically. Almost no one had a clean theoretical explanation of *why* it worked. Prior theory (Geist et al. 2019) had shown that adding such regularization "doesn't hurt", but stopped short of saying it *helps* — the bounds were no better than for unregularized value iteration.

This paper closes that gap. It picks a single concrete question — *what does KL regularization do to the propagation of errors in approximate value iteration?* — and gives a sharp answer.

**Key findings — what they show.**

1. **Implicit Q-value averaging.** When you regularize the greedy step at iteration $k$ by penalizing distance to the previous policy $\pi_{k-1}$, the closed-form solution for the new policy turns out to be a softmax over the *sum* of all past Q-estimates — not just the latest one. So KL regularization doesn't compute on $Q_k$ alone; it implicitly takes a (running) average across $Q_0, Q_1, \ldots, Q_k$. Past iterations get a vote in the current decision.

2. **A much tighter performance bound.** Once you see the averaging, you can prove error-propagation bounds that have:
   - **linear** dependency on the time horizon $1/(1-\gamma)$ (standard value iteration is *quadratic* — $1/(1-\gamma)^2$ — and that is known to be tight),
   - the bound involves the *norm of the average of the per-iteration errors* instead of the *sum of the norms*.

3. **Compensation of errors in practice.** Suppose your per-iteration estimation error is centered, zero-mean noise (a martingale difference — think: an unbiased estimator with finite variance). Then ordinary approximate value iteration *cannot guarantee convergence* to the optimal policy; the noise can pile up. But KL-regularized value iteration *does* converge, because the average of i.i.d. centered noise tends to zero. This is the law of large numbers buying you free robustness.

4. **A unified abstract scheme.** They define a generic "regularized modified policy iteration" (MD-MPI / DA-MPI) that has two knobs: $\lambda$ for KL strength, $\tau$ for entropy strength. Setting these gives back essentially every major algorithm in the family — SAC, TRPO, MPO, soft Q-learning, DPP, Mellowmax, AL/CVI, MoDQN, Politex, softened LSPI. A two-knob taxonomy.

5. **Empirical confirmation in deep RL.** Their theory assumes the greedy step is solved exactly, which is not true with neural networks. They run a clean Cartpole + Atari (Asterix) sweep over $(\lambda, \tau)$ and confirm: regularization helps DQN, regularizing the *evaluation* step (not just the greedy step) is never harmful, and KL alone (no entropy) is sometimes enough.

**Initial takeaway — high-level significance in simple terms.**
Before this paper, KL regularization in RL had the status of "magic stabilizer" — it just made gradients tame and let big-policy networks train. After this paper, KL regularization has a *precise mechanistic explanation*: it implicitly ensembles your own past Q-estimates across iterations. That ensembling is what gives you robustness to noisy/biased Q estimates, and it is provable. For the project's purpose, this means **any architectural mechanism that scales the KL coefficient up or down — a modulator, a temperature head, a learned $\lambda$ — is in effect controlling *how strongly the agent trusts its history vs. the current estimate***. That's a meaningful interoceptive lever, not a hyperparameter tweak.

### 1.2 Phase 2 — Graduate-Level Deep Dive (Vieillard KL-reg)

#### 1.2.1 Setup: regularized greediness and the regularized Bellman operator

Let $\mathcal{S}$, $\mathcal{A}$ be finite. The standard Bellman evaluation operator is

$$T_\pi q = r + \gamma P_\pi q, \qquad (P_\pi q)(s,a) = \sum_{s'} P(s'\mid s,a) \sum_{a'} \pi(a'\mid s') q(s',a').$$

The unregularized greedy operator is $\mathcal{G}(q) = \arg\max_\pi \langle \pi, q\rangle$. The paper *augments* greediness by **two simultaneous regularizers**: an entropy bonus and a KL penalty toward an anchor policy $\mu$:

$$
\mathcal{G}^{\lambda,\tau}_\mu(q) \;=\; \arg\max_{\pi \in \Delta_\mathcal{A}^\mathcal{S}} \left( \langle \pi, q\rangle \;-\; \lambda\, \mathrm{KL}(\pi\,\|\,\mu) \;+\; \tau\, \mathcal{H}(\pi) \right),
$$

where $\mathcal{H}(\pi) = -\langle \pi, \ln \pi\rangle$ and $\mathrm{KL}(\pi\|\mu) = \langle \pi, \ln \pi - \ln \mu\rangle$. The regularized evaluation operator must also be modified to remain consistent with the regularized greediness:

$$
T_{\pi \mid \mu}^{\lambda,\tau} q \;=\; r + \gamma P_\pi\!\left( \langle \pi, q\rangle - \lambda\, \mathrm{KL}(\pi\,\|\,\mu) + \tau\, \mathcal{H}(\pi) \right).
$$

Setting $\lambda = \tau = 0$ recovers vanilla MPI. The abstract scheme **MD-MPI($\lambda,\tau$)** is then:

$$
\boxed{
\begin{aligned}
\pi_{k+1} &= \mathcal{G}^{\lambda,\tau}_{\pi_k}(q_k) \\
q_{k+1} &= \bigl( T_{\pi_{k+1}\mid \pi_k}^{\lambda,\tau} \bigr)^m q_k + \varepsilon_{k+1}
\end{aligned}
} \tag{1}
$$

with $m=1$ giving regularized value iteration and $m=\infty$ giving regularized policy iteration. $\varepsilon_{k+1}$ is whatever error is incurred (e.g. function-approximation regression error).

#### 1.2.2 The implicit-averaging identity (heart of the paper)

Set $\tau = 0$ for clarity, so the regularizer is purely the KL term. The greedy step at iteration $k$ asks: maximize over $\pi$,

$$
\pi_{k+1} \;=\; \arg\max_\pi \left( \langle \pi, q_k\rangle \;-\; \lambda\, \mathrm{KL}(\pi\,\|\,\pi_k) \right).
$$

This is a classic Lagrange/KKT problem. The Lagrangian for a fixed state $s$ is $L(\pi) = \sum_a \pi(a)q_k(a) - \lambda\sum_a \pi(a)(\ln\pi(a) - \ln\pi_k(a)) - \nu(\sum_a\pi(a)-1)$. Setting $\partial L/\partial \pi(a) = 0$:

$$
q_k(a) - \lambda\bigl(\ln \pi(a) - \ln\pi_k(a)\bigr) - \lambda - \nu = 0
\;\Longrightarrow\;
\pi(a) \;\propto\; \pi_k(a) \exp\!\left(\frac{q_k(a)}{\lambda}\right).
$$

So **the closed-form regularized greedy policy** is

$$
\pi_{k+1}(a\mid s) \;\propto\; \pi_k(a\mid s)\, \exp\!\left(\frac{q_k(s,a)}{\lambda}\right). \tag{†}
$$

Now iterate $(\dagger)$ backwards. With $\pi_0$ taken uniform,

$$
\pi_1 \propto \exp\!\left(\frac{q_0}{\lambda}\right), \quad
\pi_2 \propto \pi_1 \exp\!\left(\frac{q_1}{\lambda}\right) \propto \exp\!\left(\frac{q_0 + q_1}{\lambda}\right), \quad \ldots
$$

$$
\boxed{\;\pi_{k+1}(a\mid s) \;\propto\; \exp\!\left(\frac{1}{\lambda} \sum_{j=0}^k q_j(s,a)\right)\;} \tag{‡}
$$

This is the **implicit Q-averaging identity**: KL-regularizing toward the previous policy turns the policy into a softmax over the **sum** (equivalently the average, up to the normalization absorbed into $\lambda$) of *all* past Q-estimates. The identity $(\ddagger)$ is the entire reason every theorem in the paper goes through — averaging $q_0,\ldots,q_k$ cancels zero-mean noise, while taking only $q_k$ does not.

This motivates the equivalent **DA-MPI (Dual Averaging MPI)** form, which makes the running sum explicit rather than rolling it through the previous policy:

$$
\text{DA-MPI}(\lambda, 0):\quad
\begin{cases}
\pi_{k+1} = \mathcal{G}^{0, (k+1)\lambda}(h_k) \\
h_{k+1} = \bigl(T^{\lambda,0}_{\pi_{k+1}\mid \pi_k}\bigr)^m q_k + h_k + \varepsilon_{k+1} \\
q_{k+1} = h_{k+1}/(k+1)
\end{cases}
$$

So $h_k = \sum_{j=0}^k q_j$ is the running sum of Q-estimates, $q_k$ is the running average, and the greedy policy is a softmax over $h_k$ with temperature $(k+1)\lambda$. For $\tau > 0$ the analog uses an *exponential* moving average instead of a cumulative sum:

$$
\text{DA-MPI}(\lambda, \tau > 0):\quad
\begin{cases}
\pi_{k+1} = \mathcal{G}^{0,\tau}(h_k) \\
h_{k+1} = \beta h_k + (1-\beta) q_{k+1}, \qquad \beta := \dfrac{\lambda}{\lambda + \tau} \\
q_{k+1} = \bigl(T^{\lambda,\tau}_{\pi_{k+1}\mid \pi_k}\bigr)^m q_k + \varepsilon_{k+1}
\end{cases}
$$

**Proposition 1.** For any $\lambda > 0$, MD-MPI($\lambda, 0$) and DA-MPI($\lambda, 0$) are equivalent. For any $\tau > 0$, MD-MPI($\lambda, \tau$) and DA-MPI($\lambda, \tau$) are equivalent.

The DA form is the analyst's tool — it lays the averaging bare so error-propagation arguments work term-by-term.

#### 1.2.3 Theorem 1 — Linear horizon + error averaging (KL-only)

**Setting:** DA-VI($\lambda, 0$) — i.e. $m = 1$, pure KL regularization, no entropy.

Define:
- $E_k := \dfrac{1}{k}\sum_{j=1}^k \varepsilon_j$ — the running average of per-iteration errors.
- $A_1 := (I - \gamma P_{\pi^\star})^{-1} - (I - \gamma P_{\pi_k})^{-1}$ — a horizon-correction operator (sign-controlled but bounded in $1/(1-\gamma)$).
- $g_1(k) := (4/k)\, v_{\max}^\tau$ — a vanishing initialization-dependent residual.

Assume $\|q_k\|_\infty \le v_{\max}$ (enforced by clipping; non-restrictive).

**Theorem 1.** For all $k \ge 1$, component-wise:
$$
0 \;\le\; q^\star - q^{\pi_k} \;\le\; A_1 \, E_k + g_1(k) \cdot \mathbf{1}.
$$

The $\ell_\infty$ corollary is:

$$
\|q^\star - q^{\pi_k}\|_\infty \;\le\; \frac{2}{1-\gamma}\left\|\frac{1}{k}\sum_{j=1}^k \varepsilon_j\right\|_\infty + \frac{4\, v_{\max}}{\lambda\, k}.
$$

**Compare to the classical AVI bound** (Munos / Scherrer, scaled by $1-\gamma$ for normalization):
$$
\|q^\star - q^{\pi_k}\|_\infty \;\le\; \frac{2\gamma}{(1-\gamma)^2}(1-\gamma)\sum_{j=1}^k \gamma^{k-j} \|\varepsilon_j\|_\infty \;+\; \frac{2}{1-\gamma}\gamma^k v_{\max}.
$$

Two qualitative differences:
1. **Horizon factor:** $1/(1-\gamma)$ (linear) vs. $1/(1-\gamma)^2$ (quadratic). Quadratic dependency is known to be *tight* for unregularized AVI/API (Scherrer et al.). KL regularization breaks the lower bound.
2. **Error term:** $\|(1/k)\sum_j \varepsilon_j\|_\infty$ (norm of the average) vs. $\sum_j \gamma^{k-j}\|\varepsilon_j\|_\infty$ (sum of norms). If the $\varepsilon_j$ form a martingale difference sequence with $\mathbb{E}[\varepsilon_j]=0$ and bounded variance, then by the law of large numbers $(1/k)\sum_j \varepsilon_j \to 0$ almost surely — *DA-VI converges to $q^\star$*. Meanwhile $\sum_j \gamma^{k-j}\|\varepsilon_j\|_\infty$ does not vanish for AVI in the same regime.

The cost is paid in the *initialization* term: $g_1(k) = 4 v_{\max}/(\lambda k)$ decays as $1/k$, slower than the $\gamma^k$ exponential decay of AVI. Larger $\lambda$ makes this initialization term worse but the error-averaging better. There is a clean trade-off.

#### 1.2.4 Theorem 2 — Quadratic horizon + EMA error averaging (KL + entropy)

**Setting:** DA-VI($\lambda, \tau$) with $\tau > 0$ — so the entropy term biases the fixed point. The bound is necessarily against the *regularized* optimum $q^\star_\tau$, defined as the fixed point of $T^{0,\tau}_\pi$ at the optimal policy $\pi^\star_\tau = \mathcal{G}^{0,\tau}(q^\star_\tau)$.

Define:
- $\beta := \lambda/(\lambda+\tau) \in [0,1)$ — the EMA decay rate.
- $E_k^\beta := (1-\beta)\sum_{j=1}^k \beta^{k-j}\varepsilon_j$ — exponentially weighted moving average of errors.
- $A_2^{k:j} := P_{\pi^\star_\tau}^{k-j} + (I-\gamma P_{\pi_{k+1}})^{-1} P^{k:j+1}(I-\gamma P_{\pi_j})^{-1}$, with $P^{k:j} := P_{\pi_k}P_{\pi_{k-1}}\cdots P_{\pi_j}$ (and $P^{k:j}=I$ if $j > k$).
- $g_2(k) := \gamma^k\!\left(1 + \tfrac{1}{1-\gamma}\right)\sum_{j=0}^k (\beta/\gamma)^j\, v_{\max}^\tau$ — vanishing init residual.

**Theorem 2.** Component-wise,
$$
0 \;\le\; q^\star_\tau - q^{\pi_{k+1}}_\tau \;\le\; \sum_{j=1}^k \gamma^{k-j} A_2^{k:j}\, E_j^\beta + g_2(k)\cdot\mathbf{1}.
$$

The $\ell_\infty$ corollary:
$$
\|q^\star_\tau - q^{\pi_{k+1}}_\tau\|_\infty \;\le\; \frac{2}{(1-\gamma)^2}\,(1-\gamma)\sum_{j=1}^k \gamma^{k-j}\|E_j^\beta\|_\infty \;+\; \gamma^k\!\left(1 + \tfrac{1}{1-\gamma}\right)\sum_{j=0}^k \left(\tfrac{\beta}{\gamma}\right)^j v_{\max}^\tau.
$$

**Reading the bound.** The horizon factor is back to *quadratic* (the price of using entropy regularization is that it does the "vanilla" Bellman-style propagation). But each error term in the discounted sum is no longer a raw $\varepsilon_j$ — it is an *exponentially-weighted moving average* of past errors, $E_j^\beta$. The EMA has the recursion
$$E_k^\beta = \beta\, E_{k-1}^\beta + (1-\beta)\varepsilon_k.$$
For i.i.d. centered $\varepsilon_j$ with variance $\sigma^2 = 1$, the steady-state variance of $E_k^\beta$ is
$$\mathrm{Var}(E_k^\beta) \;=\; (1-\beta)^2 \sum_{m=0}^\infty \beta^{2m}\sigma^2 \;=\; (1-\beta)^2\cdot \frac{1}{1-\beta^2} \;=\; \frac{1-\beta}{1+\beta} \;<\; 1-\beta,$$
so the per-step error norm is reduced by a factor of $\sqrt{(1-\beta)/(1+\beta)}$ compared to AVI ($\beta=0$ gives variance 1; $\beta\to 1$ gives variance $\to 0$). This is variance reduction, not error cancellation.

The bound rewards **larger $\beta$** (smaller per-step variance) but penalizes it through the $g_2(k)$ initialization term, which decays more slowly as $\beta\to 1$. The function $g_2$ is given in two regimes — for $\beta=\gamma$, $g_2(k) = 2(k+1)\gamma^k v_{\max}^\tau$, asymptotically $o(1)$ but slower than $\gamma^k$.

**Interplay $\lambda$ vs. $\tau$.** Critically, the l.h.s. of the bound (the suboptimality being measured) depends only on $\tau$, since it's against $q^\star_\tau$. The r.h.s. depends only on $\beta = \lambda/(\lambda+\tau)$. So one can shrink $\tau$ to make the regularization bias arbitrarily small (Geist et al. 2019 gives $\|q^\star - q^\star_\tau\|_\infty \le \tau\ln|\mathcal{A}|/(1-\gamma)$) while *independently* choosing $\lambda$ to fix $\beta$. The KL term doesn't add bias; only entropy does. KL purely shapes error propagation.

#### 1.2.5 Practical algorithms encompassed

Table 1 of the paper organizes the family by (regularization type) × (whether regularization is in the evaluation step or only the greedy step):

| | only entropy | only KL | both |
|---|---|---|---|
| reg. eval | Soft Q-learning, SAC, Mellowmax | DPP, SQL | CVI, AL |
| unreg. eval | softmax DQN | TRPO, MPO, Politex, MoVI | softened LSPI, MoDQN |

The paper notes that "$w/o$" reg-eval is *outside* their analysis — when $\tau > 0, \lambda = 0$ without regularized evaluation, the Bellman operator can have multiple fixed points (Asadi & Littman 2017). Including the regularizer in evaluation is the safer recipe.

#### 1.2.6 Empirical study — when does the theory translate to deep RL?

The theoretical analysis assumes the regularized greedy step is solved exactly. With function approximation by neural networks this is not true — the policy is parameterized and trained, introducing its own error. The paper builds six deep-RL variants from a Dopamine DQN baseline:

- **MD direct** (TRPO-like): minimize the regularized objective directly via policy-gradient with a KL-to-target-network penalty.
- **MD indirect** (MPO-like): compute the closed-form $\pi_{k+1} \propto \pi_k^\beta \exp((1-\beta) q_k/\lambda)$ analytically, then distill it into a policy network.
- **DA** (MoDQN-like): maintain an $h$-network that is a moving average of the $q$-network; the analytical regularized greedy reads off $h$.

Each crossed with (regularized vs. unregularized evaluation), giving six variants (five for $\tau=0$).

Sweeps over $(\beta, \tau)$ for Cartpole (10 seeds) and Asterix Atari (3 seeds, sticky actions). Headline findings:

- Regularization in any form beats DQN baseline over a *wide* parameter range.
- Adding regularization to the *evaluation step* (the "w/" row) is never harmful and is mildly helpful, especially with large entropy.
- Best greediness type (MD-direct vs. MD-indirect vs. DA) is problem-dependent.
- Strikingly: with $\tau = 0$ (no entropy at all), well-tuned KL alone matches or beats KL+entropy combinations. Entropy is *not* required for the gains.

This is the empirical hook that justifies the companion Munchausen paper: if KL alone suffices, then a one-line reward-bonus trick that injects implicit KL (no policy network needed) ought to deliver the same gains. That's exactly the next paper.

### 1.3 Appendix — Section-by-Section Backbone (Vieillard KL-reg)

The synthesis above regroups material by *concept* (averaging identity → Thm 1 → Thm 2 → empirics). For traceability, the paper's own section order is reproduced here.

**§1 Introduction.** Successful deep RL (TRPO, MPO, SAC) is built on KL regularization. Prior theory (Geist et al. 2019 on Bregman-regularized MDPs) showed regularization "doesn't harm" but not that it helps. This paper: KL regularization implicitly averages successive q-estimates; the resulting bound is the first with both linear horizon dependency and error averaging. Only DPP and SQL had error-averaging before, but both have quadratic horizon. Analysis covers entropy + KL combined.

**§2 Background and Notations.** Finite MDP $\langle\mathcal{S},\mathcal{A},P,r,\gamma\rangle$. $v_{\max}^\tau = r_{\max}/(1-\gamma) + \tau\ln|\mathcal{A}|/(1-\gamma)$. Standard Bellman $T_\pi q = r + \gamma P_\pi q$. AMPI scheme $\pi_{k+1}\in\mathcal{G}(q_k)$, $q_{k+1} = (T_{\pi_{k+1}})^m q_k + \varepsilon_{k+1}$, with $m=1$ being AVI and $m=\infty$ being API. DQN fits in this scheme with $m=1$.

**§3 Regularized MPI.** Introduces regularized greedy $\mathcal{G}^{\lambda,\tau}_\mu$ and regularized evaluation $T^{\lambda,\tau}_{\pi\mid \mu}$. Defines MD-MPI($\lambda,\tau$) (Eq. 1) and DA-MPI($\lambda,\tau$) (Eq. 2). Proposition 1: MD-MPI and DA-MPI are equivalent (with the limit case $\lambda\to 0$ at $\tau=0$ as the exception). Table 1: links to TRPO, MPO, SAC, DPP, SQL, CVI, AL, Politex, MoVI, MoDQN, softened LSPI, Mellowmax, soft Q-learning, softmax DQN.

**§4 Theoretical Analysis.** Restricted to $m=1$ (extension to $m>1$ open). Theorem 1: component-wise bound for DA-VI($\lambda,0$); $\ell_\infty$ corollary has linear horizon and error averaging. Discussion: rate of convergence $1/k$ (slower than $\gamma^k$ but converges in the noisy case where AVI doesn't); larger $\lambda$ → slower convergence but more averaging. Theorem 2: bound for DA-VI($\lambda,\tau>0$) against $q^\star_\tau$; quadratic horizon, EMA error variance reduction with $\beta=\lambda/(\lambda+\tau)$. Interplay: l.h.s. controlled by $\tau$ (bias), r.h.s. controlled by $\beta$ (variance). Limitations: greedy step assumed exact; only $m=1$ analyzed; says nothing about how to *bound* individual errors (only about how they *propagate*).

**§5 Empirical Study.** Builds 6 deep-RL variants (MD direct / MD indirect / DA × w/ reg eval / w/o). Two environments: Cartpole and Asterix. Sweeps over $(\beta,\tau)$. Findings: regularization helps; regularized evaluation helps especially with entropy; greedy choice is problem-dependent; KL alone (no entropy) can match KL+entropy. Hypothesis: KL provides *adaptive* exploration (decays with training) where entropy provides static exploration.

**§6 Conclusion.** KL regularization → implicit Q-averaging. First combination of linear horizon + error averaging. Companion paper (Munchausen RL) is a reparameterization that eliminates the greedy-step error, making the bounds applicable verbatim.

---

## Paper 2 — Vieillard et al. 2020 — *Munchausen Reinforcement Learning*

**Full title:** Munchausen Reinforcement Learning
**Authors:** Nino Vieillard, Olivier Pietquin, Matthieu Geist
**Venue:** NeurIPS 2020
**PDF:** `docs/project/references/Temperature/sources/Vieillard et al. 2020 - Munchausen Reinforcement Learning.pdf`

### 2.1 Phase 1 — Foundational Overview (Munchausen)

**Introduction — what problem this paper solves.**
The companion paper (§1) proves that KL-regularized value iteration has dramatically better error-propagation properties than vanilla AVI — but only *if you solve the regularized greedy step exactly*. With a neural-network policy, you cannot, because the closed-form solution $\pi_{k+1}(a|s) \propto \pi_k(a|s)\exp(q_k(s,a)/\lambda)$ requires you to *remember every past policy* to recurse, and policy networks don't store that explicitly. So the elegant theory of Vieillard 2020a refuses to apply where it would most help — to deep Q-learning agents like DQN or its descendants.

This paper finds a workaround so clean it borders on trickery. They observe: **you can get the same KL regularization without ever computing it, just by adding a scaled log-policy term to the reward.** No regularized greedy step, no policy network, no constraint optimization. Just modify the regression target of DQN by adding $\alpha\tau\log\pi_{\bar\theta}(a_t|s_t)$ — the log-probability of the action you actually took, scaled by hyperparameters $\alpha$ and $\tau$. They name the trick "Munchausen RL" after the Baron who, in Raspe's tales, pulls himself out of a swamp by yanking on his own hair — the agent bootstraps its learning signal from its *own current policy*.

**Key findings — what they show.**

1. **The trick.** Take any TD-style algorithm. Replace the reward $r_t$ in the target by $r_t + \alpha\tau\log\pi(a_t|s_t)$. Done. That's the entire method.

2. **Munchausen-DQN beats C51.** Built as a minimal modification of DQN (add an entropy term to define a softmax policy, then add the log-policy reward bonus), M-DQN is the *first* non-distributional-RL agent to outperform C51 on the 60-game Atari benchmark — without n-step returns, without prioritized replay, without distributional value functions.

3. **Munchausen-IQN beats Rainbow.** The same trick applied to Implicit Quantile Networks (IQN) produces M-IQN, which surpasses Rainbow — the previous state-of-the-art single-agent Atari benchmark — on both mean and median human-normalized scores.

4. **Theoretical equivalence: Munchausen ≡ KL+entropy regularized VI.** They prove (Theorem 1) that M-VI($\alpha, \tau$) — the abstract scheme behind M-DQN — produces *exactly the same sequence of policies* as Mirror Descent VI with KL coefficient $\alpha\tau$ and entropy coefficient $(1-\alpha)\tau$. So Munchausen is a reparameterization of Vieillard 2020a's MD-VI scheme. Critically, the Munchausen reparameterization has **no greedy-step error** — the greedy step is now just a softmax over the Q-network, which can be computed exactly — so the companion paper's strong bounds (linear horizon, error averaging) apply to M-DQN verbatim.

5. **Quantifiable action-gap increase.** They prove (Theorem 2) that M-VI multiplies the action-gap of the regularized MDP by $(1+\alpha)/(1-\alpha)$. In the limit $\alpha \to 1$ the action-gap is *infinite* for suboptimal actions. A large action-gap is known to mitigate the impact of Q-estimation errors on the induced greedy policy. M-RL is the first scheme to *quantify* the gap-increase, not just induce it (cf. Advantage Learning).

6. **Hyperparameters that just work.** $\alpha = 0.9, \tau = 0.03, l_0 = -1$ (a log-policy clipping threshold for numerical stability), tuned on a handful of games, are used unchanged for all 60 Atari games.

**Initial takeaway — high-level significance in simple terms.**
Munchausen RL is the rare deep-RL contribution where the implementation is *easier* than the existing baseline. You write four extra lines: compute the softmax policy from your Q-network, take its log at the action that was played, scale it by $\alpha\tau$, add to the target. That's the algorithm. The theoretical equivalence to MD-VI means you're not stumbling onto an empirical accident — you are *exactly* running KL-regularized value iteration, and you inherit a rigorous performance bound. For the project's modulator-conditioned temperature, this is a critical observation: **a modulator that scales $\alpha$ or $\tau$ on the Munchausen bonus is a modulator that scales the implicit KL coefficient — directly controlling how strongly the agent averages its own past Q-estimates.** No architectural surgery required.

### 2.2 Phase 2 — Graduate-Level Deep Dive (Munchausen)

#### 2.2.1 From DQN to Soft-DQN to M-DQN

The classical DQN regression target, for transition $(s_t, a_t, r_t, s_{t+1})$, is
$$\hat{q}_{\text{dqn}}(r_t, s_{t+1}) \;=\; r_t \;+\; \gamma \sum_{a'} \pi_{\bar\theta}(a'|s_{t+1}) q_{\bar\theta}(s_{t+1}, a'), \qquad \pi_{\bar\theta}\in\mathcal{G}(q_{\bar\theta}),$$
where $\bar\theta$ denotes target-network weights and $\pi_{\bar\theta}$ is the deterministic greedy policy. Define the **Soft-DQN** target by introducing entropy regularization at temperature $\tau$, so the policy is a softmax $\pi_{\bar\theta} = \mathrm{sm}(q_{\bar\theta}/\tau)$:
$$\hat{q}_{\text{s-dqn}}(r_t, s_{t+1}) \;=\; r_t \;+\; \gamma \sum_{a'} \pi_{\bar\theta}(a'|s_{t+1})\bigl( q_{\bar\theta}(s_{t+1}, a') - \tau \log\pi_{\bar\theta}(a'|s_{t+1}) \bigr). \tag{1}$$
This is the discrete-action analog of SAC. The Munchausen modification then adds the log-policy of the action *actually taken at time $t$* to the immediate reward, scaled by $\alpha \in [0,1]$:
$$\boxed{\;\hat{q}_{\text{m-dqn}}(r_t, s_{t+1}) \;=\; r_t \;+\; \alpha\tau\log\pi_{\bar\theta}(a_t|s_t) \;+\; \gamma \sum_{a'} \pi_{\bar\theta}(a'|s_{t+1})\bigl( q_{\bar\theta}(s_{t+1}, a') - \tau \log\pi_{\bar\theta}(a'|s_{t+1}) \bigr).\;} \tag{2}$$
Setting $\alpha = 0$ recovers Soft-DQN. Setting $\tau \to 0$ recovers DQN (with the convention that $\tau\log\pi \to 0$). **The full Munchausen algorithm is exactly Eq. (2)**: replace the DQN regression target by Eq. (2). Nothing else changes — same network, same replay buffer, same optimizer (in their experiments, Adam instead of RMSProp).

#### 2.2.2 The abstract M-VI scheme

Stripping away function approximation and sampling, M-DQN is an instance of:
$$
\text{M-VI}(\alpha, \tau):\quad
\begin{cases}
\pi_{k+1} = \arg\max_{\pi\in\Delta_\mathcal{A}^\mathcal{S}}\bigl( \langle \pi, q_k\rangle + \tau\mathcal{H}(\pi) \bigr) = \mathrm{sm}(q_k/\tau) \\
q_{k+1} = r + \alpha\tau\log\pi_{k+1} + \gamma P\bigl\langle \pi_{k+1}, q_k - \tau\log\pi_{k+1}\bigr\rangle + \varepsilon_{k+1}
\end{cases}
\tag{3}
$$
**The greedy step is just an entropy-regularized softmax over $q_k$** — no $\mu$ anchor, no KL term, no policy network. This is the key engineering insight: the entire KL effect will be made to live inside the *evaluation* update via the $\alpha\tau\log\pi$ term in the reward.

#### 2.2.3 Theorem 1 — Munchausen is implicit MD-VI

Rewrite the evaluation step of M-VI($1, \tau$) (set $\alpha = 1$ for clarity) using the substitution $q'_k := q_k - \tau\log\pi_k$ (an "advantage-like" reparameterization):

Starting from $q_{k+1} = r + \tau\log\pi_{k+1} + \gamma P\langle \pi_{k+1}, q_k - \tau\log\pi_{k+1}\rangle + \varepsilon_{k+1}$,

subtract $\tau\log\pi_{k+1}$ from both sides:
$$q_{k+1} - \tau\log\pi_{k+1} \;=\; r + \gamma P\left\langle \pi_{k+1}, q_k - \tau\log\pi_{k+1} - \tau\log\frac{\pi_{k+1}}{\pi_k}\right\rangle + \varepsilon_{k+1}.$$
Inside the inner product, $\langle \pi_{k+1}, q_k - \tau\log\pi_k\rangle = \langle \pi_{k+1}, q'_k\rangle$, and $\langle\pi_{k+1}, \tau\log(\pi_{k+1}/\pi_k)\rangle = \tau\mathrm{KL}(\pi_{k+1}\|\pi_k)$. So:
$$\boxed{\; q'_{k+1} \;=\; r \;+\; \gamma P\bigl( \langle \pi_{k+1}, q'_k\rangle - \tau\, \mathrm{KL}(\pi_{k+1}\|\pi_k) \bigr) \;+\; \varepsilon_{k+1}.\;}$$
This is the regularized Bellman evaluation operator $T^{\tau,0}_{\pi_{k+1}\mid\pi_k} q'_k$ from Vieillard 2020a — **the KL regularization has appeared spontaneously**, sourced entirely from the $\alpha\tau\log\pi_{k+1}$ bonus.

Now rewrite the greedy step using the same substitution. The Munchausen greedy step asks $\pi_{k+1} = \arg\max_\pi \langle\pi, q_k\rangle + \tau\mathcal{H}(\pi)$. But $q_k = q'_k + \tau\log\pi_k$, so
$$\langle \pi, q_k\rangle + \tau\mathcal{H}(\pi) \;=\; \langle\pi, q'_k\rangle + \tau\langle\pi,\log\pi_k\rangle - \tau\langle\pi,\log\pi\rangle \;=\; \langle\pi, q'_k\rangle - \tau\mathrm{KL}(\pi\|\pi_k).$$
This is the regularized greedy of MD-VI($\tau, 0$). So **the entire M-VI($1,\tau$) scheme, expressed in the $q'$ variables, is exactly MD-VI($\tau, 0$)** — pure KL regularization at strength $\tau$, no entropy. The general $\alpha\in[0,1]$ case interpolates:

**Theorem 1.** For any $k \ge 0$, define $q'_k := q_k - \alpha\tau\log\pi_k$. Then M-VI($\alpha,\tau$) is equivalent to
$$
\begin{cases}
\pi_{k+1} = \arg\max_\pi \langle\pi, q'_k\rangle - \alpha\tau\,\mathrm{KL}(\pi\|\pi_k) + (1-\alpha)\tau\,\mathcal{H}(\pi) \\
q'_{k+1} = r + \gamma P\bigl( \langle\pi_{k+1}, q'_k\rangle - \alpha\tau\,\mathrm{KL}(\pi_{k+1}\|\pi_k) + (1-\alpha)\tau\,\mathcal{H}(\pi_{k+1}) \bigr) + \varepsilon_{k+1}.
\end{cases}
$$
**This is MD-VI($\alpha\tau, (1-\alpha)\tau$).** Therefore Vieillard 2020a's Theorem 1 applies to M-VI($1, \tau$), and Theorem 2 applies to M-VI($\alpha < 1, \tau$).

**Crucially, the greedy step of M-VI is just a softmax over $q_k$**, which can be computed exactly even with a Q-network. The KL term lives inside the reward bonus, not inside an optimization problem. So the "greedy step assumed exact" caveat of Vieillard 2020a is *not violated* by M-VI. The companion bound applies directly.

#### 2.2.4 Inheriting the linear-horizon error-averaging bound

Let $q_{\bar\theta_k}$ be the target-network at iteration $k$, $\pi_{k+1} = \mathrm{sm}(q_k/\tau)$, and define
$$\varepsilon_{k+1} := q_{k+1} - \bigl( r_{k+1} + \alpha\log\pi_{k+1} - \gamma P\langle\pi_{k+1}, q_k - \tau\log\pi_{k+1}\rangle \bigr)$$
the per-iteration discrepancy between actual and ideal updates. Then for $\alpha = 1$, as a direct corollary of Vieillard 2020a's Theorem 1:
$$\boxed{\; \|q^\star - q^{\pi_k}\|_\infty \;\le\; \frac{2}{1-\gamma}\left\| \frac{1}{k}\sum_{j=1}^k \varepsilon_j \right\|_\infty \;+\; \frac{4\, r_{\max}}{(1-\gamma)^2\, k} \;+\; \frac{\tau\log|\mathcal{A}|}{k}. \;}$$
- **Linear horizon factor** $1/(1-\gamma)$ scales the error term (AVI would have $1/(1-\gamma)^2$).
- **Norm of the average** of errors (AVI would have a discounted *sum of norms*). With centered noise, $\|(1/k)\sum\varepsilon_j\|_\infty \to 0$ by LLN, while $\sum \gamma^{k-j}\|\varepsilon_j\|_\infty$ does not.
- **Initialization term** vanishes as $1/k$.

So M-DQN inherits the strongest known error-propagation bound for an ADP scheme — *and* it is implementable as a one-line modification to DQN's loss. This combination — empirical accessibility + tight theoretical guarantee — is what makes Munchausen RL load-bearing for the field.

#### 2.2.5 Theorem 2 — Quantified action-gap increase

The **action-gap** at state $s$ for a Q-function $q$ is $\delta(s) := \max_a q(s,a) - q(s,\cdot) \in \mathbb{R}_+^\mathcal{A}$ (the vector of differences between the best action's value and every action's value). A wider action gap means small Q-estimation errors are less likely to flip the argmax. Advantage Learning (Bellemare et al. 2016) was designed to inflate the gap; Bellemare et al. introduced gap-increasing operators as a family.

For M-VI, let $\delta^\tau_\star(s) := \max_a q^\tau_\star(s,a) - q^\tau_\star(s,\cdot)$ be the gap of the entropy-regularized MDP, and $\delta^{\alpha,\tau}_k(s)$ the gap at iteration $k$ of M-VI (no error). Then:

**Theorem 2.** For any state $s\in\mathcal{S}$, any $0 \le \alpha \le 1$, any $\tau > 0$:
$$\lim_{k\to\infty} \delta^{\alpha,\tau}_k(s) \;=\; \frac{1+\alpha}{1-\alpha}\, \delta^{(1-\alpha)\tau}_\star(s)$$
(with the convention $\infty \cdot 0 = 0$ for $\alpha = 1$).

**Reading the theorem.**
- The action-gap of the underlying entropy-regularized MDP is scaled by **$(1+\alpha)/(1-\alpha)$**.
- At $\alpha = 0$: scale $= 1$. No gap increase — this is just Soft-DQN.
- At $\alpha = 0.5$: scale $= 3$.
- At $\alpha = 0.9$ (the paper's chosen hyperparameter): scale $= 19$.
- At $\alpha \to 1$: scale $\to \infty$ — the gap diverges for suboptimal actions, suggesting why $\alpha$ slightly below 1 is preferred (numerical stability) but should be large.

This is the **first quantified gap-increase result** in the gap-increasing operator literature. Advantage Learning was known to be gap-increasing but the increase wasn't computed in closed form.

#### 2.2.6 Connections via the log-sum-exp identity

A useful detail: by the Legendre–Fenchel duality of entropy and softmax,
$$\max_\pi \langle q, \pi\rangle + \tau\mathcal{H}(\pi) \;=\; \tau\log\sum_a \exp(q(\cdot,a)/\tau) \;=:\; \tau\log\langle\mathbf{1}, \exp(q/\tau)\rangle.$$
Substituting this into the M-VI evaluation update yields the equivalent form (Eq. 5 in the paper):
$$q_{k+1} \;=\; r + \gamma P\bigl( \tau\log\langle \mathbf{1}, \exp(q_k/\tau)\rangle \bigr) \;+\; \alpha\bigl( q_k - \tau\log\langle\mathbf{1}, \exp(q_k/\tau)\rangle \bigr) + \varepsilon_{k+1}.$$

This is **almost identical to Conservative Value Iteration (CVI)** of Kozuno et al. 2019 — differing only by a constant factor inside the log-sum-exp ($1/|\mathcal{A}|$ in CVI; $1$ here). And:
- $\alpha = 0$: **Soft Q-learning** (Fox et al. 2016; Haarnoja et al. 2017).
- $\alpha = 1$: **Dynamic Policy Programming** (Azar et al. 2012).
- $\tau \to 0$, $\alpha > 0$: $q_{k+1} = r + \gamma P\langle \pi_{k+1}, q_k\rangle + \alpha(q_k - \langle\pi_{k+1}, q_k\rangle) + \varepsilon_{k+1}$ where $\pi_{k+1}\in\mathcal{G}(q_k)$ — this is **Advantage Learning** (Baird 1999; Bellemare et al. 2016).

So Munchausen RL unifies four previously distinct algorithm families under a single one-line reward augmentation.

#### 2.2.7 Practical details: log-policy clipping, Adam, and ε-greedy exploration

The log-policy term $\tau\log\pi(a|s)$ is unbounded below as $\pi(a|s) \to 0$, which causes numerical issues when the policy becomes near-deterministic. The fix is a **log-policy clip** $[\cdot]^0_{l_0}$ with $l_0 < 0$: replace $\tau\log\pi(a|s)$ by $\max(\tau\log\pi(a|s), l_0)$. The paper uses $l_0 = -1$.

A log-sum-exp numerical trick is used to compute $\tau\log\pi(a|s)$ stably (Appx. B.1): compute $\tau\log\pi(a|s) = q(s,a) - \tau\log\sum_{a'}\exp(q(s,a')/\tau)$ in a log-sum-exp-stable way rather than computing $\pi(a|s)$ and then taking log.

The agent uses ε-greedy exploration on top of its stochastic policy. The paper notes (Appx. B.2) that using the stochastic Boltzmann policy for exploration is also possible and gives similar results.

Final hyperparameters (used unchanged across all 60 Atari games): $\alpha = 0.9$, $\tau = 0.03$, $l_0 = -1$.

#### 2.2.8 Empirical results

- **M-DQN beats C51.** First non-distributional-RL agent to outperform C51 on the 60-game ALE benchmark (sticky actions, Machado et al. methodology).
- **M-IQN beats Rainbow.** New state-of-the-art for single-agent ALE (Table 1 of the paper):

| Agent | Human-normalized mean | Human-normalized median | Rainbow-normalized mean | Rainbow-normalized median |
|---|---|---|---|---|
| M-IQN | 563% | 165% | 130% | 109% |
| Rainbow | 414% | 150% | 100% | 100% |
| IQN | 441% | 139% | 105% | 99% |
| M-DQN | 340% | 124% | 89% | 92% |
| C51 | 339% | 111% | 84% | 70% |
| DQN | 228% | 71% | 51% | 51% |

- **Per-game improvement:** M-DQN improves on DQN on 53 of 60 games; M-IQN improves on IQN on 40 of 60 games.
- **Ablation:** Adam-DQN > DQN (so the optimizer change alone is meaningful, but small compared to the Munchausen effect). Soft-DQN($\tau$) > DQN but ≈ Adam-DQN (max entropy alone is not enough). AL > Adam-DQN. M-DQN > everything. This is the cleanest demonstration that the Munchausen term (the $\alpha\tau\log\pi$ reward bonus) is doing the work, not the maximum-entropy formulation alone.
- **Action-gap illustration (Fig. 2, Asterix):** both M-DQN and AL grow the action gap over training. AL grows it more but less stably; M-DQN grows it less but more uniformly. Final returns: M-DQN 20k, Adam-DQN 15k, AL 13k. Van Seijen et al. 2019's hypothesis applies: uniform gap, not maximum gap, drives performance.

### 2.3 Appendix — Section-by-Section Backbone (Munchausen)

**§1 Introduction.** TD learning bootstraps Q from its own current estimate. The paper proposes bootstrapping from the agent's *current policy* by adding the scaled log-policy to the reward. Named after the Baron Munchausen who pulls himself out of a swamp by his hair. Applied to DQN → M-DQN, which surpasses DQN and C51 on Atari without distributional RL, n-step returns, or PER. Applied to IQN → M-IQN, which surpasses Rainbow. Theoretical hook: M-RL implicitly performs KL regularization between successive policies, inheriting the strong bounds of Vieillard 2020a, and demonstrably increases the action-gap.

**§2 Munchausen RL.** MDP setup. Q-learning recap. The core idea: if the optimal policy $\pi^\star$ were known, $\log\pi^\star(a|s)$ would be $0$ for optimal actions and $-\infty$ for suboptimal — a strong learning signal that could augment the reward without changing the optimum. We don't know $\pi^\star$, so use the current $\pi$. Demands stochastic policies for numerical stability. DQN doesn't have stochastic policies → introduce **Soft-DQN** (Eq. 1) by adding entropy regularization at temperature $\tau$, recovering DQN as $\tau\to 0$ and being the discrete-action analog of SAC. **M-DQN** is Soft-DQN with the Munchausen reward bonus $\alpha\tau\log\pi_{\bar\theta}(a_t|s_t)$ added (Eq. 2). Same recipe applied to IQN gives M-IQN.

**§3 What happens under the hood?** Abstract M-VI scheme (Eq. 3). Manipulations: defining $q' = q - \tau\log\pi$, M-VI($1,\tau$) becomes pure KL-regularized VI (MD-VI($\tau, 0$)). General Theorem 1: M-VI($\alpha, \tau$) ≡ MD-VI($\alpha\tau, (1-\alpha)\tau$). Therefore Vieillard 2020a's bounds apply directly — and the greedy-step-is-exact caveat is satisfied, because M-VI's greedy step is just a softmax over a Q-network, not an optimization problem. Concrete bound for $\alpha=1$: linear horizon, norm of average of errors, $1/k$ initialization residual. Log-sum-exp identity gives M-VI's relation to CVI (almost identical), Soft Q-learning ($\alpha=0$), DPP ($\alpha=1$), Advantage Learning ($\tau\to 0$). **Theorem 2** quantifies the action-gap increase: $(1+\alpha)/(1-\alpha)$ multiplier on the entropy-regularized MDP's action-gap, infinite in the $\alpha\to 1$ limit.

**§4 Experiments.** ALE 60-game benchmark (sticky actions, Machado methodology). M-DQN and M-IQN built from Dopamine. Hyperparameters $\alpha=0.9, \tau=0.03, l_0=-1$ used across all games. Baselines: DQN, C51, IQN, Rainbow (all from Dopamine). Action-gap illustration on Asterix. Ablation: Adam-DQN, Soft-DQN($\tau$), Soft-DQN($(1-\alpha)\tau$), AL, all using Adam, all underperform M-DQN. Comparison: M-DQN > C51, M-IQN > Rainbow. Per-game improvements: 53/60 (M-DQN vs DQN), 40/60 (M-IQN vs IQN).

**§5 Conclusion.** Munchausen RL is a one-line reward augmentation. M-DQN outperforms DQN on 53/60 Atari games and C51 overall. M-IQN sets new state-of-the-art. Theoretical equivalence to MD-VI means strong bounds apply. Connections to CVI, DPP, AL formalized. Action-gap increase quantified. Suggests revisiting "obvious" components of standard RL can yield large gains.

---

## Paper 3 — Zhu et al. 2023 — *Tsallis-KL Munchausen*

**Full title:** General Munchausen Reinforcement Learning with Tsallis Kullback-Leibler Divergence
**Authors:** Lingwei Zhu, Zheng Chen, Matthew Schlegel, Martha White
**Venue:** NeurIPS 2023
**PDF:** `docs/project/references/Temperature/sources/Zhu et al. 2023 - General Munchausen Reinforcement Learning with Tsallis Kullback-Leibler Divergence.pdf`

### 3.1 Phase 1 — Foundational Overview (Tsallis-KL)

**Introduction — what problem this paper solves.**
The previous two papers establish: KL-regularized RL is theoretically powerful (implicit Q-averaging, error robustness), and Munchausen RL turns it into a one-line practical trick. But there's a known weakness in the *underlying divergence*: standard KL — a Shannon-entropy-based divergence — induces a policy that is a **uniform** softmax-average over all past Q-estimates. Uniform averages are susceptible to outliers. If one past Q-estimate is wildly biased, it gets the same weight as every other estimate. This is a known limitation of KL in statistics and generative modeling.

The paper asks: *what if we replace Shannon KL with a generalized KL — specifically the Tsallis $q$-divergence, parameterized by a real number $q \ge 1$?* When $q = 1$, Tsallis-KL is exactly Shannon KL, recovering Munchausen. When $q > 1$, the divergence has two new properties simultaneously:
1. The induced policy is **sparse** — it concentrates probability on the highest-valued actions and assigns exactly zero probability to the worst ones (a "sparsemax" policy).
2. The policy averages past Q-estimates *non-uniformly* — it computes a weighted average with extra cross-product terms, so consistently-high-value actions get reinforced more than transient spikes.

**Key findings — what they show.**

1. **Tsallis entropy & Tsallis-KL definitions.** The $q$-logarithm $\ln_q x = (x^{1-q}-1)/(1-q)$ and its inverse the $q$-exponential $\exp_q x = [1+(1-q)x]_+^{1/(1-q)}$ generalize $\ln$ and $\exp$. Tsallis entropy is $S_q(\pi) = \langle -\pi^q, \ln_q \pi\rangle$. Tsallis-KL is $D^q_{\mathrm{KL}}(\pi\|\mu) = \langle\pi, -\ln_q(\mu/\pi)\rangle$.

2. **Sparsemax policies (Theorem 1).** Optimizing $\langle\pi, Q\rangle + \tau S_q(\pi)/(1-q)$ yields a policy of the form $\pi(a|s) = \exp_q(Q(s,a)/\tau - \psi_q(Q/\tau))$, which is *truncated* at zero for low-valued actions. Larger $q$ → more truncation. Importantly, the truncation effect of $(q, \tau)$ pair can be reproduced with $(q=2, \tau/(q-1))$ — so $q=2$ is sufficient for any sparsity level by scaling $\tau$.

3. **Convergence (Theorem 3).** For $q = 2$, Tsallis-KL is strongly convex, so the regularized value-iteration recursion converges to a unique fixed point. Pseudo-additivity of $\ln_q$ ($q \ne 1$) breaks the standard convergence proofs but $q=2$ is special-case tractable.

4. **More than uniform averaging (Theorem 4).** Whereas standard Munchausen's optimal policy is $\pi_{k+1}\propto\exp(\sum Q_j / \tau)$ — uniform average — Tsallis-KL's optimal policy at $q=2$ is
$$\pi_{k+1}\propto \exp_2\!\left(\sum_j Q_j + \sum_{j\ge 2}\sum_{i_1<\cdots<i_j} Q_{i_1}\cdots Q_{i_j}\right).$$
The cross-product terms $Q_{i_1}\cdots Q_{i_j}$ amplify actions that have *consistently* had high values across iterations. This is more robust than averaging.

5. **MVI($q$) algorithm.** Extension of Munchausen-VI: replace the standard log-policy term $\tau\log\pi_{k+1}$ in the reward bonus with the $q$-log $\tau\ln_q\pi_{k+1}$, equivalently the action-gap $Q_k - M_{q,\tau} Q_k$ where $M_{q,\tau}$ is the Boltzmann-softmax-style operator using $\exp_q$. At $q=1$ this *is* Munchausen. At $q=2$ this is a strict generalization with sparse policies. There's a small residual term that the paper proves stays negligible for $q=2$ on the Cartpole testbed.

6. **Empirical wins on 35 Atari games.** MVI(2) beats MVI(1) on roughly half the games by significant margins, doubles performance on another fifth, ties on a quarter, and loses on 3 hard-exploration games (PrivateEye, Chopper, Seaquest). Compared to Tsallis-VI (no Munchausen, just sparsemax), MVI(2) wins by >100% improvement on more than half the games. This shows the KL term (Munchausen) is *essential* — Tsallis entropy alone is not enough.

**Initial takeaway — high-level significance in simple terms.**
The Tsallis-KL paper makes precise a long-suspected fact: *the Shannon KL is not the only sensible divergence to regularize toward*. Tsallis-KL at $q=2$ adds two properties on top of the standard Munchausen trick — sparse policies (exact zeros for bad actions) and weighted-not-uniform averaging of past Q-estimates (more robust to outliers). For the project's modulator-conditioned temperature, this introduces a second knob beyond the $\tau$ that controls "how stochastic to be": $q$ controls *how sparse the support is*. A modulator that learns to push $q$ up in confident regimes (commit to a small action set) and back down in exploratory regimes (keep full support) would have a clean information-theoretic interpretation. In tasks with many irrelevant actions (a noisy action space), $q > 1$ is a concrete win; in hard-exploration regimes, $q = 1$ is safer.

### 3.2 Phase 2 — Graduate-Level Deep Dive (Tsallis-KL)

#### 3.2.1 The $q$-logarithm, $q$-exponential, and Tsallis entropy

The Tsallis (1988) generalization defines, for $q \in \mathbb{R} \setminus \{1\}$:
$$
\boxed{\;\ln_q x := \frac{x^{1-q} - 1}{1 - q},\qquad \exp_q x := [1 + (1-q)x]_+^{\frac{1}{1-q}},\;}
$$
where $[\cdot]_+ := \max(\cdot, 0)$. By L'Hôpital, $\lim_{q\to 1}\ln_q x = \ln x$ and $\lim_{q\to 1}\exp_q x = \exp x$. So $q=1$ is the Shannon special case. **Key non-Shannon property:** $\exp_q$ is invertible only for $x > -1/(1-q)$ — beyond this, $1 + (1-q)x$ goes negative and the bracket clips to zero. This **clipping** is precisely what produces sparse policies.

Tsallis entropy is
$$S_q(\pi) := \langle -\pi^q, \ln_q\pi\rangle = \frac{1}{q-1}\left( 1 - \sum_a \pi(a)^q \right).$$
For $q\to 1$, $S_q(\pi)\to \mathcal{H}(\pi) = -\langle\pi,\ln\pi\rangle$. Larger $q$ flattens $\ln_q$ and steepens $\exp_q$ (Figure 1 of the paper).

#### 3.2.2 Theorem 1 — Tsallis-regularized policies are sparsemax

Consider the entropy-regularized value-iteration recursion (Eq. 1 of the paper) with $\Omega(\pi) = -S_q(\pi)$:
$$\pi_{k+1} = \arg\max_\pi\bigl(\langle\pi, Q_k\rangle + \tau S_q(\pi)\bigr).$$

**Theorem 1.** The optimal policy under Tsallis-entropy regularization is
$$
\pi(a|s) \;=\; \left[ \frac{1-q}{\tau}\Bigl( Q(s,a) - \tilde\psi_q(Q(s,\cdot)/\tau)\Bigr) \right]_+^{\frac{1}{1-q}} \;=\; \exp_q\!\left( \frac{Q(s,a)}{\tau} - \psi_q\!\left(\frac{Q(s,\cdot)}{\tau}\right)\right),
$$
where $\psi_q(\cdot)$ is a state-dependent normalizing threshold satisfying $\langle\mathbf{1},\exp_q(Q/\tau - \psi_q)\rangle = 1$. Moreover, for any $(q,\tau)$ with $q>1$, the *same* truncation pattern can be achieved using $(q=2, \tau/(q-1))$.

**Consequence:** the policy has **sparse support** — actions with $Q(s,a)/\tau - \psi_q < -1/(1-q)$ are clipped to zero. Larger $q$ → stronger truncation.

**Theorem 2 (Taylor-approximate threshold).** For $q \ne 1, \infty$, the threshold $\psi_q$ has no closed form except at $q=2$. A Taylor-expansion approximation gives
$$\hat\psi_q\!\left(\frac{Q(s,\cdot)}{\tau}\right) \;=\; \frac{\sum_{a\in K(s)} Q(s,a)/\tau \;-\; 1}{|K(s)|} + 1,$$
where $K(s)$ is the set of "highest-valued actions" (the support of the sparsemax). $K(s)$ is determined by a self-referential ordering condition $1 + i\, Q(s,a_{(i)})/\tau > \sum_{j=1}^i Q(s,a_{(j)})/\tau$, ranking actions by Q-value. At $q=2$, the Taylor expansion is exact: $\hat\psi_2 = \psi_2$. This is the sparsemax operator of Martins & Astudillo 2016.

#### 3.2.3 Tsallis-KL divergence

The Tsallis-KL divergence between two policies (Furuichi et al. 2004) is
$$
\boxed{\;D^q_{\mathrm{KL}}(\pi\,\|\,\mu) \;:=\; \left\langle \pi, -\ln_q\frac{\mu}{\pi}\right\rangle \;=\; \frac{1}{q-1}\left( 1 - \sum_a \pi(a)^q \mu(a)^{1-q}\right).\;}
$$
At $q=1$ this is Shannon KL. The $f$-divergence representation: $D^q_{\mathrm{KL}}(\pi\|\mu) = \langle\mu, f(\pi/\mu)\rangle$ with $f(t) = -\ln_q t$ (which is convex for $q > 0$).

**Mass-covering vs. mode-seeking.** Standard KL is mode-seeking: it penalizes $\pi$ putting mass where $\mu$ has little ($\pi/\mu$ large drives $\ln(\pi/\mu)$ large). Tsallis-KL with $q > 1$ raises $\pi/\mu$ to the $q$-th power *inside the divergence* (via the $\pi^q$ term), so it **penalizes large $\pi/\mu$ ratios more strongly**, making it mass-covering / mode-covering. This matches Wang et al. 2018's tail-adaptive $f$-divergence behavior. (At $q = 2$, Tsallis-KL coincides with $\alpha$-divergence at $\alpha = 2$.)

**Pseudo-additivity.** The $q$-logarithm satisfies
$$\ln_q(xy) \;=\; \ln_q x + \ln_q y + (1-q)\ln_q x \cdot \ln_q y.$$
This **non-extensivity** is what breaks the standard convergence proofs and what produces cross-product terms in the optimal policy (Theorem 4 below).

**Theorem 3.** The recursion (Eq. 1) with $\Omega(\pi) = D^q_{\mathrm{KL}}(\pi\|\mu)$ for $q=2$ converges to a unique fixed point. *Proof sketch: $D^{q=2}_{\mathrm{KL}}$ is strongly convex; Geist et al. 2019's general convergence theorem applies.* Convergence for arbitrary $q > 1$ is open.

#### 3.2.4 Theorem 4 — Beyond uniform averaging

For the recursion
$$
\begin{cases}
\pi_{k+1} = \arg\max_\pi \langle \pi, Q_k\rangle - D^q_{\mathrm{KL}}(\pi\|\pi_k) \\
Q_{k+1} = r + \gamma P\langle \pi_{k+1}, Q_k - D^q_{\mathrm{KL}}(\pi_{k+1}\|\pi_k)\rangle
\end{cases}
$$
the optimal greedy policy is

$$
\boxed{\;
\pi_{k+1} \;\propto\; \bigl( \exp_q Q_1 \cdots \exp_q Q_k \bigr)^{\!\!\frac{1}{q-1}} \;=\; \left[ \exp_q\!\left( \sum_{j=1}^k Q_j \;+\; \sum_{j=2}^k (q-1)^j \!\!\sum_{i_1 < \cdots < i_j}\!\! Q_{i_1}\cdots Q_{i_j} \right) \right]^{\!\!\frac{1}{q-1}}.
\;}
$$

**At $q=1$:** $(q-1)^j = 0$ for $j\ge 2$, the cross-product terms vanish, and we recover the standard Munchausen identity $\pi_{k+1}\propto \exp(\sum_j Q_j/\tau)$ — uniform averaging.

**At $q=2$:** $(q-1)^j = 1$ for all $j$, so $\pi_{k+1}\propto\exp_2\!\bigl(\sum_j Q_j + \sum_{j\ge 2}\sum_{i_1<\cdots<i_j} Q_{i_1}\cdots Q_{i_j}\bigr)$. Two effects in one formula:

1. **Weighted average inside $\exp_q$.** The expansion identity (Eq. 8 of the paper)
$$\exp_q\!\left(\sum_{i=1}^k Q_i\right) \;=\; \exp_q(Q_1)\cdot \prod_{i=2}^k \exp_q\!\left( \frac{Q_i}{1 + (1-q)\sum_{j<i} Q_j}\right)$$
shows that each $Q_i$ is *divided* by a normalizer that includes all *prior* Q-estimates. This is a robust-divergence-style weighted average: large prior $Q$-history shrinks the contribution of the current $Q_i$. This is the same scheme as the $\gamma$-divergence robust weighting (Futami et al. 2018, Table 1).

2. **Cross-product reinforcement.** Actions $a$ that have consistently high values across many iterations get $Q_{i_1}(a)\cdots Q_{i_j}(a)$ products that compound. Actions whose high values are *transient spikes* get smaller products. This is the precise mathematical realization of "mode-covering with consistency weighting" — robust to outliers, strict about consistency.

**Proof sketch (paper's Appendix D):** part 1 establishes $\pi_{k+1}\propto\exp_q Q_1 \cdots\exp_q Q_k$ by induction on the regularized greedy step. Part 2 uses the *two-point equation* of Yamano 2002 and the $(2-q)$ duality of Naudts 2002 / Suyari–Tsukada 2005 to derive the identity $(\exp_q x \cdot \exp_q y)^{q-1} = \exp_q((x+y)^{q-1} + (q-1)^2 xy)$, which expands recursively into the cross-product sum.

#### 3.2.5 MVI($q$) — the practical algorithm

The pure Tsallis-KL value iteration faces the same engineering problem as standard KL-regularized VI: the closed-form regularized greedy step depends on $\pi_k$, which would require remembering every past policy. Munchausen (Vieillard 2020b) solved this for $q=1$ by adding $\alpha\tau\log\pi$ to the reward. Zhu et al. extend the trick:

**Standard MVI ($q=1$, for comparison).** Eq. (9) of the paper:
$$\begin{cases}
\pi_{k+1} = \arg\max_\pi \langle\pi, Q_k - \tau\log\pi\rangle = \mathrm{sm}(Q_k/\tau) \\
Q_{k+1} = r + \alpha\tau\log\pi_{k+1} + \gamma P\langle\pi_{k+1}, Q_k - \tau\log\pi_{k+1}\rangle.
\end{cases}$$
With $Q'_{k+1} := Q_{k+1} - \alpha\tau\log\pi_{k+1}$, this is equivalent to
$$Q'_{k+1} = r + \gamma P\bigl( \langle\pi_{k+1}, Q'_k\rangle - \alpha\tau D_{\mathrm{KL}}(\pi_{k+1}\|\pi_k) + (1-\alpha)\tau\mathcal{H}(\pi_{k+1})\bigr).$$
**Implementation trick.** The original Munchausen paper notes that $\tau\log\pi_{k+1} = \alpha(Q_k - M_\tau Q_k)$ where $M_\tau Q_k := (1/Z_k)\langle \exp(Q_k/\tau), Q_k\rangle$ is the Boltzmann softmax operator. Using the **action-gap form** $Q_k - M_\tau Q_k$ is more numerically stable than computing $\log\pi$ directly.

**MVI($q$).** Replace the action-gap operator with the Tsallis version:
$$M_{q,\tau} Q_k \;:=\; \Bigl\langle \exp_q\!\bigl(Q_k/\tau - \psi_q(Q_k/\tau)\bigr),\, Q_k\Bigr\rangle.$$
At $q=1$, $M_{q,\tau} Q_k = M_\tau Q_k$. At $q=\infty$, $M_{q,\tau} Q_k = \max_a Q_k(s,a)$ — recovering Advantage Learning (no regularization).

The MVI($q$) algorithm:
$$\boxed{\;\begin{cases}
\pi_{k+1} = \exp_q(Q_k/\tau - \psi_q(Q_k/\tau)) \\
\hat r_t = r_t + \alpha\bigl( Q_k(s_t, a_t) - M_{q,\tau} Q_k(s_t) \bigr) \\
Q_{k+1} = \hat r + \gamma P\langle\pi_{k+1}, Q_k - \tau\ln_q\pi_{k+1}\rangle + \varepsilon_{k+1}
\end{cases}\;}$$

**The approximation.** Unlike standard MVI, MVI($q$) does *not exactly* implement Tsallis-KL regularized VI — there is a residual term from $\ln_q$'s pseudo-additivity. From the paper's derivation (Eq. 11):
$$Q'_{k+1} = r + \gamma P\langle\pi_{k+1}, Q'_k + (1-\alpha)\tau S_q(\pi_{k+1})\rangle - \gamma P\bigl\langle \pi_{k+1}, \alpha\tau D^q_{\mathrm{KL}}(\pi_{k+1}\|\pi_k) - \alpha\tau R_q(\pi_{k+1},\pi_k)\bigr\rangle$$
with residual
$$R_q(\pi_{k+1},\pi_k) := (1-q)\ln_q\!\left(\frac{1}{\pi_{k+1}}\right)\ln_q\pi_k.$$
For $q=2$, the empirical magnitude of $R_q$ stays small throughout Cartpole training. For $q\ge 4$, it grows non-negligibly. **So $q=2$ is the sweet spot** where MVI($q$) approximates the Tsallis-KL-regularized scheme well.

**Why use action-gap instead of $\tau\ln_q\pi$ directly?** The paper finds the action-gap form $Q_k - M_{q,\tau} Q_k$ more numerically stable than computing $\tau\ln_q\pi_{k+1}$ directly. Two reasons: (1) for actions outside the sparsemax support, $\pi_{k+1}(a) = 0$ and $\ln_q 0 = -\infty$ — the action-gap stays finite; (2) empirically the action-gap and $\tau\ln_q\pi$ differ by a near-constant (≈ -0.5) plus a transient deviation early in training — the action-gap is more uniform across timesteps.

#### 3.2.6 Empirical study

**Setup.** Quantile Regression DQN backbone (Dabney et al. 2018) implemented in Stable-Baselines3. 35 Atari games, 50M frames each, averaged over 3 seeds. Hyperparameter search on Asterix + Seaquest: $\alpha \in \{0.01, 0.1, 0.5, 0.9, 0.99\}$, $\tau \in \{0.01, 0.1, 1, 10, 100\}$.

**MVI(q=2) vs. MVI(q=1).** On 35 games:
- ~5 games: large improvement (>100% gain).
- ~5 games: ~2× performance.
- ~7 games: comparable.
- 3 games: MVI(q=2) worse — PrivateEye, Chopper, Seaquest (all hard-exploration games).

**Interpretation of the failures.** Sparsemax policies have *less* exploration than Boltzmann — they assign exact zero probability to low-value actions, so they can't "discover" that a low-Q action is actually good after some random exploration. On hard-exploration games this is a liability. On games with clear good/bad action structure, the sparsity is a feature.

**MVI(q=2) vs. Tsallis-VI (sparsemax without Munchausen, $\alpha=0$).** MVI(q=2) wins by >100% on more than half the games; >400% on 10 games. **The Munchausen term (the KL regularization) is essential** — Tsallis-entropy regularization alone is not enough. Previously, prior work using Tsallis entropy without Munchausen (Lee et al. 2018, 2020) had found *no benefit* over Shannon entropy. The combination of Tsallis sparsemax + KL regularization is what unlocks the gains.

#### 3.2.7 What this paper adds to the corpus

Three substantive generalizations over Vieillard 2020a/b:

1. **A new policy class.** Sparsemax policies (exact zero on low-value actions) instead of Boltzmann softmax.
2. **Weighted, not uniform, averaging.** The optimal policy under Tsallis-KL averages past Q-estimates non-uniformly, downweighting outliers via the cross-product reinforcement of consistently high actions.
3. **Empirical confirmation that the trick still works.** A one-symbol change in the Munchausen reward bonus (replace $\log$ with $\ln_q$ at $q=2$) inherits all of Munchausen's empirical advantages and adds a modest but real ~50% mean improvement on Atari.

The cost: convergence proofs only work cleanly at $q=2$ (open for general $q$); the algorithm is no longer an *exact* reparameterization of Tsallis-KL VI (small residual). Both are acceptable trade-offs for the empirical wins.

### 3.3 Appendix — Section-by-Section Backbone (Tsallis-KL)

**§1 Introduction.** KL regularization to the previous policy gives implicit Q-averaging and tighter bounds (recap of Vieillard 2020a). But uniform average is fragile to outliers (Futami et al. 2018). Heuristics exist (vanishing regularization coefficients on some estimates, Grau-Moya et al. 2019; Haarnoja et al. 2018) but break the theoretical guarantees. Goal: find an alternative divergence that preserves the benefits while improving outlier robustness. Tsallis-KL (Tsallis 1988): replace $\ln$ with $\ln_q$. Tsallis entropy at $q=2$ has been used to induce sparsemax policies (Chow et al. 2018; Lee et al. 2018), but those papers found it didn't help. *Combining* Tsallis-entropy with Tsallis-KL — and using the Munchausen-style implicit-regularization trick — does help. The paper extends MVI to MVI($q$).

**§2 Problem Setting.** Discrete-time MDP. Entropy-regularized recursion (Eq. 1). $f$-divergence framework. Standard KL recovered by $f(t) = -\ln t$. Optimal policy under KL: $\pi_{k+1}\propto\exp(\tau^{-1}\sum_{j=1}^k Q_j)$ — uniform softmax average over history.

**§3 Generalizing to Tsallis Regularization.**

- §3.1 *Tsallis entropy regularization.* Definitions of $\ln_q, \exp_q, S_q$. **Theorem 1**: optimal policies under Tsallis-entropy regularization are sparsemax in the form $\exp_q(Q/\tau - \psi_q)$; $q$ and $\tau$ are interchangeable for controlling truncation (set $q=2$ and scale $\tau$). **Theorem 2**: Taylor-approximate threshold $\hat\psi_q$ for general $q$; exact at $q=2$.
- §3.2 *Tsallis-KL regularization and convergence.* Definition $D^q_{\mathrm{KL}}(\pi\|\mu)$. More mass-covering than KL for $q>1$. Pseudo-additivity (Eq. 5) breaks standard proofs. **Theorem 3**: convergence at $q=2$ via strong convexity (Geist et al. 2019).
- §3.3 *Beyond uniform averaging.* **Theorem 4**: optimal policy under Tsallis-KL involves *weighted* average + cross-product terms $\sum_j Q_j + \sum_j \sum_{i_1<\cdots<i_j} Q_{i_1}\cdots Q_{i_j}$. Cross-products amplify consistency.

**§4 A practical algorithm for Tsallis-KL regularization.**
- §4.1 *Implicit regularization with MVI.* Recap of Vieillard 2020b's trick: $\tau\log\pi_{k+1} = \alpha(Q_k - M_\tau Q_k)$, using action-gap instead of log-policy.
- §4.2 *MVI($q$) for general $q$.* Replace $M_\tau$ with $M_{q,\tau}$ (Tsallis sparsemax operator). At $q=1$ recovers MVI. At $q=\infty$ recovers Advantage Learning. Derivation (Eq. 11) shows there is a residual term $R_q(\pi_{k+1},\pi_k) = (1-q)\ln_q(1/\pi_{k+1})\ln_q\pi_k$, which stays small for $q=2$ but grows for $q \ge 4$ (Fig. 3, Cartpole). Why action-gap over $\tau\ln_q\pi$: bounded for zero-probability actions; empirically more stable.

**§5 Experiments.** Quantile Regression DQN backbone, Stable-Baselines3, 35 Atari games, 50M frames, 3 seeds. Hyperparameter sweep on Asterix + Seaquest.
- §5.1 *MVI($q=2$) vs. MVI($q=1$).* Mixed wins. ~5 games >100% gain, ~5 games ~2×, ~7 comparable, 3 worse (hard-exploration: PrivateEye, Chopper, Seaquest). Sparsemax = less exploration; bad on hard-exploration games but good elsewhere.
- §5.2 *Importance of the KL term.* MVI(q=2) vs. Tsallis-VI ($\alpha = 0$, sparsemax without Munchausen): MVI(q=2) wins on more than half by >100%, on 10 games by >400%. Confirms that the Munchausen / KL component is doing the work, not just the sparsemax. Previous Tsallis-entropy-only work (Lee et al. 2018, 2020) found no benefit; this work shows that *combining* Tsallis-entropy with KL is what helps.

**§6 Conclusion and Discussion.** Five contributions: (a) Tsallis policies are $\exp_q$ functions; (b) Tsallis-KL-regularized policies do weighted-average + cross-products of past Q-estimates; (c) convergence at $q=2$; (d) MVI($q$) algorithm; (e) Atari improvements from $q=2$.

---

## Cross-paper structural connection

The three papers share **the same KL-regularized Bellman operator** as their analytical kernel, then attack it from three complementary angles. The diagram:

```
                  Vieillard 2020a (theory)
            ─ "what does KL regularization DO?" ─
                          │
                          ▼
            Implicit Q-averaging identity (‡):
            π_{k+1} ∝ exp( (1/λ) Σ_{j=0}^k q_j )
                          │
                          ▼
    Linear horizon × error averaging performance bound
                          │
            (assumes exact regularized greedy step — broken with NNs)
                          │
                          ▼
                  Vieillard 2020b (practice)
        ─ "how do we IMPLEMENT it with deep nets?" ─
                          │
                          ▼
         Munchausen trick: add α·τ·log π(a|s) to reward
                          │
                          ▼
    Equivalent to MD-VI(ατ, (1-α)τ); greedy step is exact softmax
    →  Vieillard 2020a's bounds apply VERBATIM to M-DQN
                          │
                          ▼
                  Zhu 2023 (generalization)
       ─ "what if the DIVERGENCE were generalized?" ─
                          │
                          ▼
        Replace Shannon KL with Tsallis-KL (q-divergence)
                          │
                          ▼
        Sparse policies (exact zeros) + cross-product
        averaging (consistency reinforcement)
                          │
                          ▼
        MVI(q): replace log with ln_q in Munchausen reward bonus
        At q=1: recover MVI exactly.  At q=2: sparsemax + Tsallis-KL.
```

**The same operator, three lenses.**

1. **Theory lens (Vieillard 2020a).** The MD-VI / DA-VI operator with KL coefficient $\lambda$ and entropy coefficient $\tau$. Studied via the dual-averaging reformulation, yielding the implicit-Q-averaging identity and the linear-horizon-error-averaging bound. Limitation: assumes the regularized greedy step is solved exactly (impossible with neural policies).

2. **Practice lens (Vieillard 2020b).** *Same operator*, but reparameterized via the substitution $q' := q - \alpha\tau\log\pi$. The reparameterization turns the regularized greedy step into a *plain entropy-regularized softmax* (exact under neural Q-networks), while the KL regularization migrates into a one-term reward bonus $\alpha\tau\log\pi(a_t|s_t)$. The greedy-step-error caveat disappears; the bounds become operational.

3. **Generalization lens (Zhu 2023).** *Same operator structure*, but with the divergence replaced by Tsallis $D^q_{\mathrm{KL}}$. The implicit-Q-averaging identity becomes a *weighted, non-uniform* average with cross-product terms. The Munchausen reparameterization extends: replace $\log\pi$ with $\ln_q\pi$ (or, more stably, the Tsallis action-gap $Q - M_{q,\tau} Q$). Convergence proof for $q=2$ inherits via strong convexity.

**A single line of code captures the family.** The Munchausen reward bonus
$$
\hat r_t \;=\; r_t \;+\; \alpha\, \tau\, \ln_q\pi(a_t|s_t)
$$
parameterized by $(\alpha, \tau, q)$:
- $\alpha = 0$: maximum-entropy RL (Soft-DQN / SAC).
- $\alpha > 0$, $q = 1$: Munchausen RL (Vieillard 2020b).
- $\alpha > 0$, $q = 2$: Tsallis-KL Munchausen (Zhu 2023).
- $\alpha \to 1$, $q \to \infty$: Advantage Learning (Bellemare et al. 2016).
- $\tau \to 0$, $\alpha > 0$, $q = 1$: still Advantage Learning.

So the entire family — SAC, Munchausen, Tsallis-Munchausen, Advantage Learning, Dynamic Policy Programming, Conservative VI — is a *three-knob* parameterization of one reward augmentation. The three papers above give the theory ($\lambda, \tau$), the implementation trick ($\alpha$), and the divergence-family generalization ($q$).

## Where this connects to the project's modulator-conditioned temperature

The grid-world-pain project uses a learned modulator that, conditioned on interoceptive signals, adjusts policy "temperature" online. The three papers reframe what such a modulator could *mean*:

**1. Modulator as KL-coefficient controller (most direct hook).** If the modulator's output scales $\alpha$ or $\tau$ in the Munchausen reward bonus $\alpha\tau\log\pi(a_t|s_t)$, then by Vieillard 2020b's Theorem 1 it is *exactly* scaling the implicit KL coefficient $\alpha\tau$ in the equivalent MD-VI scheme. By Vieillard 2020a's implicit-Q-averaging identity, scaling the KL coefficient up means *more aggressive averaging across past Q-estimates*; scaling it down means *trust the current estimate more*. This gives the modulator a precise interoceptive interpretation: "how much should I trust my history vs. my recent observation?" — exactly the right question for an interoceptive agent confronting changing internal states (e.g., a transition into a pain context where past low-cost behavior is no longer relevant).

**2. Modulator as action-gap controller.** Vieillard 2020b's Theorem 2 says M-VI multiplies the entropy-regularized action-gap by $(1+\alpha)/(1-\alpha)$. A modulator that pushes $\alpha$ toward 1 sharpens the agent's preferences (large action-gap → robust to Q-estimation noise); a modulator that pushes $\alpha$ toward 0 softens them. This is a "commit vs. explore" axis that a pain/threat signal could meaningfully drive — high threat → commit (large $\alpha$) → don't second-guess; low threat → soften.

**3. Modulator as divergence-shape controller (Zhu 2023).** With Tsallis-KL, the modulator could additionally adjust $q$ — the divergence shape parameter — to switch between Boltzmann (full-support) and sparsemax (zero-support-on-bad-actions) policy classes. In a noisy action space (many irrelevant or distractor actions), $q > 1$ truncates them away; in a hard-exploration regime, $q = 1$ keeps every action live. This is an explicit "do I commit to a small action set" lever that is genuinely different from temperature.

**4. Cheapest implementation path.** Munchausen RL is a *reward augmentation*, not an architectural addition. The project does not need a new policy head, a new loss term, or a constraint-optimization step. A modulator output $m_t$ feeding into the Munchausen reward as $\alpha(m_t) \tau(m_t) \ln_{q(m_t)}\pi(a_t|s_t)$ requires only modifying the target computation in the existing training loop. The theoretical guarantees (Vieillard 2020a's linear-horizon-error-averaging bound) apply for fixed $\alpha, \tau$ — they do not immediately extend to a learned $\alpha(m_t)$, which would be a worthy follow-up theoretical question for the project, but the empirical robustness on Atari shows the family is forgiving.

**5. Empirical caveat from Zhu 2023.** Sparsemax ($q=2$) hurts on hard-exploration games. If the project's tasks include phases that require broad exploration (e.g., novel-environment introduction, sparse-reward foraging), a modulator that defaults to $q = 1$ in those phases and ramps to $q = 2$ only in commit-phase regimes (familiar environment, well-learned Q) would be the safer pattern.

**Recommended path for the project (if the team chooses to engage this corpus):**
- Start at $(\alpha, \tau, q) = (0.9, 0.03, 1)$ — exactly Vieillard 2020b's tuning. Add the $\alpha\tau\log\pi$ bonus to the existing reward. This is the smallest possible code change and inherits all of Vieillard's theory + empirical wins.
- Once the baseline Munchausen agent is solid, replace the constant $\alpha$ with $\alpha(m_t)$ from the modulator. The theoretical bound no longer applies pointwise — but the operator-level interpretation (modulator controls implicit Q-averaging strength) is preserved.
- Only after both above are working, consider $q > 1$ — and only on tasks where action-space is genuinely large and contains distractors.

The rest of the Temperature corpus — Haarnoja's SAC foundations, Lin's CAT-SAC, Wang's Meta-SAC, Asadi & Littman's softmax operators, Ahmed's entropy impact study — covers the empirical / SAC-variant axis of this same question, in case the project's eventual architectural choice is closer to actor-critic than to value-based.

---


