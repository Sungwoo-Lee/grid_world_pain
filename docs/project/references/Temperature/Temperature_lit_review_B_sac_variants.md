# Temperature Corpus — Shard B: SAC Variants with Adaptive Entropy Temperature

**Reviewer:** literature-reviewer
**Date:** 2026-05-19
**Shard scope:** SAC variants that go beyond the vanilla auto-tuned temperature of Haarnoja et al. (SAC-v2). Two papers, processed sequentially.

---

## Purpose (plain-language entry point)

Soft Actor-Critic (SAC) is an off-policy reinforcement-learning algorithm that learns a stochastic policy by maximizing a weighted sum of two things: (a) the usual environment reward and (b) the *entropy* of the policy — a measure of how spread-out / non-greedy the policy's action distribution is. The weight on the entropy term is called the **entropy temperature**, written $\alpha$. A large $\alpha$ tells the agent to keep exploring (act randomly); a small $\alpha$ tells it to commit to its best-known action.

The original SAC (Haarnoja et al. 2018a, "SAC-v1") fixed $\alpha$ to a single hand-tuned number per task — costly to grid-search and often suboptimal at different stages of training. The follow-up "SAC-v2" (Haarnoja et al. 2018b) made $\alpha$ adapt automatically by adding a constraint "policy entropy must stay above some target $\bar{H}$" and solving for $\alpha$ as the Lagrange multiplier (dual variable) on that constraint. But SAC-v2 has two limits **the papers in this shard set out to fix**:

1. **The target entropy $\bar{H}$ is itself a hyperparameter** — Haarnoja's "use $-\dim(\mathcal{A})$" rule of thumb works on Mujoco but is unprincipled. So you've just traded tuning $\alpha$ for tuning $\bar{H}$.
2. **$\alpha$ is one scalar, applied identically to every state.** SAC-v2's $\alpha$ is *adaptive over training* but *uniform over the state space* — the same exploration pressure is applied in a state the agent has visited a million times as in a state it has never seen.

The two papers in this shard attack these two limits with different tools:

- **Meta-SAC (Wang & Ni 2020)** — drops SAC-v2's entropy constraint entirely and instead treats $\alpha$ as a *meta-parameter* optimized by **metagradient** on a downstream objective that mirrors how the agent will actually be evaluated (greedy return, no entropy bonus). No more $\bar{H}$ to tune. $\alpha$ is still a single scalar but its trajectory through training is shaped by what actually helps task return, not by an arbitrary entropy floor.
- **Cat-SAC (Lin et al. 2020)** — makes $\alpha$ **state-dependent**. It defines a curiosity signal $c(s)$ from a random-network-distillation–style predictor, normalizes it to zero mean, *adds it to the target entropy* so unfamiliar states get a higher entropy target and familiar states get a lower one, then learns an *instance-level* temperature $\alpha_\delta(s) = g_\delta(c(s))$ that maps the curiosity bucket of $s$ to a temperature value. This is the closest published precedent for "$\alpha$ conditioned on a modulator signal" — exactly the design question this project is exploring with a neuromodulator-conditioned head.

The body of this document gives a per-paper section-ordered backbone, an undergraduate-level Phase 1 synthesis, and a graduate-level Phase 2 deep-dive with full LaTeX (meta-objective, metagradient chain rule for Meta-SAC; curiosity signal, instance-level $\alpha_\delta(s)$, integration with the SAC actor/critic losses for Cat-SAC). A final cross-paper section flags concrete architecture and training-loop details that carry over to the project's "modulator-conditioned temperature head" design.

---

## Table of Contents

- [Purpose (plain-language entry point)](#purpose-plain-language-entry-point)
- [Paper 1 — Wang & Ni (2020), Meta-SAC: Auto-tune the Entropy Temperature of SAC via Metagradient](#paper-1--wang--ni-2020-meta-sac-auto-tune-the-entropy-temperature-of-sac-via-metagradient)
  - [Backbone (section-ordered)](#meta-sac-backbone-section-ordered)
  - [Phase 1 — Undergraduate synthesis](#meta-sac-phase-1--undergraduate-synthesis)
  - [Phase 2 — Graduate deep-dive](#meta-sac-phase-2--graduate-deep-dive)
- [Paper 2 — Lin et al. (2020), Cat-SAC: SAC with Curiosity-Aware Entropy Temperature](#paper-2--lin-et-al-2020-cat-sac-sac-with-curiosity-aware-entropy-temperature)
  - [Backbone (section-ordered)](#cat-sac-backbone-section-ordered)
  - [Phase 1 — Undergraduate synthesis](#cat-sac-phase-1--undergraduate-synthesis)
  - [Phase 2 — Graduate deep-dive](#cat-sac-phase-2--graduate-deep-dive)
- [Cross-paper synthesis — implications for the project's modulator-conditioned temperature head](#cross-paper-synthesis--implications-for-the-projects-modulator-conditioned-temperature-head)

---

# Paper 1 — Wang & Ni (2020), Meta-SAC: Auto-tune the Entropy Temperature of SAC via Metagradient

**PDF:** `docs/project/references/Temperature/sources/Wang and Ni 2020 - Meta-SAC - Auto-tune the entropy temperature of Soft Actor-Critic via metagradient.pdf`
**Venue:** 7th ICML Workshop on Automated Machine Learning, 2020 (arXiv:2007.01932v2)
**Authors:** Yufei Wang, Tianwei Ni (CMU, equal contribution)
**Code:** https://github.com/twni2016/Meta-SAC

## Meta-SAC: Backbone (section-ordered)

### Abstract
- Problem: balancing exploration/exploitation in RL via SAC's entropy temperature $\alpha$ is sensitive and requires per-task tuning; the SAC-v2 fix (Haarnoja 2018b, constrained optimization with a target-entropy hyperparameter $\bar{H}$) trades one hyperparameter for another.
- Proposal (Meta-SAC): use **metagradient** to auto-tune $\alpha$ in SAC, with a **novel meta-objective** distinct from prior metagradient work (which typically uses the policy-gradient loss as the meta-loss).
- Result: state-of-the-art on Mujoco, +10% over SAC-v2 on Humanoid-v2 (one of the harder continuous-control tasks).

### Section 1 — Introduction
- Exploration/exploitation framing; SAC augments the RL objective with a policy-entropy term, $J(\pi) := \sum_t \gamma^t \mathbb{E}_{s_t,a_t \sim \rho_\pi}[r(s_t,a_t) + \alpha \mathcal{H}(\pi(\cdot|s_t))]$ (Eq. 1).
- $\alpha$ governs the exploration/exploitation balance. Large $\alpha \Rightarrow$ near-uniform policy, fails to exploit. Small $\alpha \Rightarrow$ near-deterministic policy, fails to explore. Optimal $\alpha$ varies *across tasks* and *across training stages*.
- SAC-v1: grid-search $\alpha$ per task (expensive).
- SAC-v2: cast the entropy term as a constraint $\mathbb{E}_{s_t,a_t\sim\rho_\pi}[-\log\pi(a_t|s_t)] \geq \bar{H}$ and apply dual gradient descent; $\alpha$ becomes the dual variable. Works empirically but (i) requires convexity assumptions that don't hold for NNs, (ii) drops time dependence as approximation, (iii) **introduces $\bar{H}$ as a new task-specific hyperparameter** with only a heuristic formula $\bar{H} = -\dim(\mathcal{A})$.
- Contribution: metagradient on $\alpha$ with a meta-loss aligned to the evaluation metric (greedy task return); no adaptive hyperparameters beyond $\alpha$ itself.

### Section 2 — Preliminaries

**2.1 Metagradient.** Following Zahavy et al. (2020). Let $\theta$ be the learnable parameters (policy + Q-network weights), $\zeta$ the hyperparameters, $\eta \subseteq \zeta$ the subset to adapt (the "metaparameters"). At step $t$:

$$\theta_{t+1}(\eta_t) \leftarrow \theta_t - \lambda_\theta \nabla_\theta L_{\text{learn}}(\theta_t, \eta_t) \quad (\text{Eq. 2})$$

$$\eta_{t+1} \leftarrow \eta_t - \lambda_\eta \nabla_\eta L_{\text{meta}}(\theta_{t+1}(\eta_t)) \quad (\text{Eq. 3})$$

The dependence of $\theta_{t+1}$ on $\eta_t$ — explicit in $\theta_{t+1}(\eta_t)$ — is what makes this "meta": $\nabla_\eta L_{\text{meta}}$ is computed by chain rule *through* the inner SGD step on $\theta$.

**2.2 SAC.** Replay buffer $\mathcal{D}$; policy $\pi_\phi$, Q-network $Q_\omega$. Losses:

$$L_Q(\omega) := \mathbb{E}_{s_t,a_t\sim\mathcal{D}}\left[\frac{1}{2}\left(Q_\omega(s_t,a_t) - Q^{\text{tar}}(s_t,a_t)\right)^2\right] \quad (\text{Eq. 4})$$

$$Q^{\text{tar}}(s_t,a_t) := r(s_t,a_t) + \gamma \mathbb{E}_{s_{t+1}\sim p_e, a_{t+1}\sim\pi_\phi}\left[\hat{Q}(s_{t+1},a_{t+1}) - \alpha\log\pi_\phi(a_{t+1}|s_{t+1})\right]$$

$$L_\pi(\phi) := \mathbb{E}_{s_t\sim\mathcal{D}, a_t\sim\pi_\phi}[\alpha\log\pi_\phi(a_t|s_t) - Q_\omega(s_t,a_t)] \quad (\text{Eq. 5})$$

SAC-v2 entropy constraint:

$$\max_{\pi_{0:T}} \mathbb{E}\left[\sum_t r(s_t,a_t)\right] \text{ s.t. } \mathbb{E}[-\log\pi(a_t|s_t)] \geq \bar{H} \quad \forall t \quad (\text{Eq. 6})$$

with dual update $L(\alpha) := \mathbb{E}[-\alpha \log\pi(a_t|s_t) - \alpha\bar{H}]$ (dropping time dependence).

Two critiques of SAC-v2 the paper foregrounds: (1) convexity assumption fails for NNs and time-dependence dropped; (2) $\bar{H}$ is a new per-task hyperparameter, contradicting the goal of eliminating tuning.

### Section 3 — Meta-SAC method

Learnable parameters $\theta = \{\phi, \omega\}$, metaparameter $\eta = \{\alpha\}$. Inner update:

$$\phi_{t+1}(\alpha_t) \leftarrow \phi_t - \lambda_\phi \nabla_\phi L_\pi(\phi_t, \alpha_t)$$
$$\omega_{t+1}(\alpha_t) \leftarrow \omega_t - \lambda_\omega \nabla_\omega L_Q(\omega_t, \alpha_t) \quad (\text{Eq. 7})$$

**Meta-loss (the novel piece):** policy-gradient loss as meta-loss "performs poorly in our initial experiments." Instead:

$$L_{\text{meta}}(\alpha_t) := \mathbb{E}_{s_0\sim\mathcal{D}_0}\left[-Q_{\omega_t}\left(s_0, \pi^{\text{det}}_{\phi_{t+1}(\alpha_t)}(s_0)\right)\right] \quad (\text{Eq. 8})$$

where $\pi^{\text{det}}$ is the deterministic version of the updated policy (mean of the Gaussian for squashed-Gaussian policies), and $\mathcal{D}_0$ is a special replay buffer of **initial states** sampled from environment resets.

Four design choices justified:

1. **Drop entropy from meta-loss** (use $Q$ not soft target with $-\alpha\log\pi$): the *evaluation metric* doesn't reward entropy, so neither should the meta-loss.
2. **Deterministic evaluation policy** ($\pi^{\text{det}}$): mirrors how policies are evaluated.
3. **Initial-state buffer $\mathcal{D}_0$**: the evaluation metric $M(\pi) = \mathbb{E}_{s_0,a_0}[Q^\pi(s_0,a_0)]$ is an expectation over the initial state distribution, so sampling $s_0$ from $\mathcal{D}_0$ rather than arbitrary states from $\mathcal{D}$ aligns the meta-loss with the metric. Ablation D.1: this matters a lot on Ant and Humanoid.
4. **Old $Q_{\omega_t}$ not updated $Q_{\omega_{t+1}(\alpha_t)}$**: backpropagating through both the policy *and* Q-network updates w.r.t. $\alpha$ is numerically unstable; using only the updated $\phi_{t+1}(\alpha_t)$ is cheaper and more stable.

The meta-loss "lies between the DDPG loss and the SAC loss": DDPG-like in that it drops the entropy term, SAC-like in that it still uses the soft-$Q$ value (computed with $\alpha$-aware targets) rather than a classic Q.

### Section 4 — Experiments

**Setup.** Mujoco: Ant, Hopper, Humanoid, Walker2d. Five seeds; eval every 10K env steps with 10 rollouts; baselines SAC-v1 (grid-search), SAC-v2, TD3.

**4.1 Mujoco learning curves.** Meta-SAC comparable to SAC-v2 on Ant/Hopper/Walker; slightly worse than SAC-v1 (which has per-task grid-searched $\alpha$); **significantly better on Humanoid** — +10% final return, faster convergence.

**4.2 How $\alpha$ evolves.** In SAC-v2, $\log\alpha$ plateaus after a few learning steps. In Meta-SAC, $\log\alpha$ changes dramatically and over a much larger scale; "generally in later learning stages, $\alpha$ almost converges to zero" — Meta-SAC starts SAC-like (large $\alpha$ for exploration) and ends DDPG-like (greedy exploitation). The authors verify that *naively* decaying $\alpha$ in SAC-v1 doesn't reproduce this benefit (Appendix E) — the *shape* of the schedule, not just the decay, matters.

### Section 5 — Conclusion
- Metagradient with an evaluation-aligned meta-loss replaces SAC-v2's constrained-optimization formulation. No new task-specific hyperparameters. SOTA on Humanoid-v2.

### Appendices (key details for implementation)

- **Appendix A — Algorithm.** Inside the per-step loop: SAC-style sample-and-update Q and $\pi$, then a separate update on $\alpha$ using $L_{\text{meta}}$ on initial-state minibatch $\mathcal{D}_0$. Uses resampling: a fresh minibatch $\mathcal{B}'$ from the replay buffer for the Q/$\pi$ updates that are differentiated through (avoids overfitting on the same minibatch used to update $\alpha$) — same online cross-validation trick from MAML and Zheng et al. 2018.
- **Appendix C — Hyperparameters.** Standard SAC hyperparameters preserved. Three changes specific to Meta-SAC:
  - Policy optimizer changed to **RMSProp** with $\epsilon=10^{-12}$ (Adam's update rule is harder to backprop through stably).
  - $\alpha$ parameterized as $\log\alpha$ with clipping at $\log\alpha \leq 0$, i.e. $0 < \alpha \leq 1$.
  - Learning rate for $\log\alpha$ is $3\times 10^{-4}$ (matches policy/Q learning rates); $\nabla_{\log\alpha} L_{\text{meta}}$ is gradient-norm-clipped at 0.05.
  - Initial-state buffer $\mathcal{D}_0$ has size 256 (= batch size).
  - **All Meta-SAC hyperparameters are constant across tasks**, in contrast to SAC-v1's per-task $\alpha$ and SAC-v2's per-task $\bar{H}$.
- **Appendix D — Ablations.**
  - D.1: Using initial states $\mathcal{D}_0$ vs arbitrary states $\mathcal{D}$ — significant degradation without $\mathcal{D}_0$ on Ant, Humanoid.
  - D.2: Resampling — without resampling, performance degrades (overfitting to the meta-update minibatch).
  - D.3: Using soft-$Q$ (with $\alpha$-aware target) vs classic-$Q$ in meta-loss — classic-$Q$ underperforms; "discourages exploration."
- **Appendix E.** Naive $\alpha$-decay in SAC-v1 with various fixed-decay schedules on Humanoid-v2 — none match Meta-SAC. The early-phase high-$\alpha$ behavior in Meta-SAC is what enables late-phase greedy exploitation.

---

## Meta-SAC: Phase 1 — Undergraduate synthesis

**The limit of vanilla auto-$\alpha$ this paper fixes.** SAC-v2's auto-tuning of $\alpha$ works by saying "force the policy's entropy to stay above some floor $\bar{H}$." But $\bar{H}$ has to be hand-set per task (heuristic: "negative of the action dimension"), and that floor is *not the thing we actually care about* — we care about *task return when the policy acts greedily at evaluation time*. SAC-v2 optimizes $\alpha$ to satisfy an entropy budget; Meta-SAC optimizes $\alpha$ to **make the policy better at the thing it'll be tested on**.

**The proposal in one sentence.** Treat $\alpha$ as a meta-hyperparameter, take one SGD step on the policy with the current $\alpha$, evaluate the updated greedy policy on initial-state $Q$-values, and backpropagate that meta-loss back to $\alpha$ through the inner SGD step.

**Key empirical result.** Comparable to SAC-v2 on three Mujoco tasks (Ant, Hopper, Walker2d); +10% better on Humanoid-v2 (the hard one). $\log\alpha$ no longer plateaus — it has a meaningful "high early for exploration, low late for exploitation" schedule that the data discovers automatically.

---

## Meta-SAC: Phase 2 — Graduate deep-dive

### The meta-objective in full

The meta-loss is

$$\boxed{L_{\text{meta}}(\alpha_t) = \mathbb{E}_{s_0 \sim \mathcal{D}_0}\left[-Q_{\omega_t}\left(s_0,\; \pi^{\text{det}}_{\phi_{t+1}(\alpha_t)}(s_0)\right)\right]}$$

Three substantive deviations from earlier metagradient work (Xu et al. 2018; Zheng et al. 2018; Zahavy et al. 2020), all motivated by alignment with the evaluation metric

$$M(\pi) := \mathbb{E}_{s_0\sim p_e, a_0\sim\pi}[Q^\pi(s_0, a_0)] = \mathbb{E}\left[\sum_t \gamma^t r(s_t, a_t)\right]$$

| Earlier work | Meta-SAC | Reason |
|---|---|---|
| Meta-loss is policy-gradient loss | Negative $Q$ value (DDPG-style critic loss without entropy bonus) | $M(\pi)$ doesn't include entropy |
| State sampled from $\mathcal{D}$ (replay buffer) | State sampled from $\mathcal{D}_0$ (initial states only) | $M(\pi)$ is expectation over $s_0$ |
| Stochastic policy | Deterministic policy $\pi^{\text{det}}$ | Standard practice for SAC evaluation: act greedy |
| $Q_{\omega_{t+1}}$ (updated) | $Q_{\omega_t}$ (pre-update) | Numerical stability when differentiating through both $\phi$ and $\omega$ updates |

### How the metagradient flows through the inner SAC update

This is the load-bearing chain rule. The inner update with $\alpha = \alpha_t$ produces

$$\phi_{t+1}(\alpha_t) = \phi_t - \lambda_\phi \nabla_\phi L_\pi(\phi_t, \alpha_t)$$

The actor loss is, recall,

$$L_\pi(\phi, \alpha) = \mathbb{E}_{s_t\sim\mathcal{D}, a_t\sim\pi_\phi}\left[\alpha \log\pi_\phi(a_t|s_t) - Q_\omega(s_t, a_t)\right]$$

so $\nabla_\phi L_\pi$ has an $\alpha$-dependent term ($\alpha \nabla_\phi \mathbb{E}[\log\pi]$) and an $\alpha$-independent term ($-\nabla_\phi \mathbb{E}[Q_\omega]$). Therefore

$$\frac{\partial \phi_{t+1}}{\partial \alpha_t} = -\lambda_\phi \nabla_\phi \mathbb{E}_{s_t,a_t\sim\pi_{\phi_t}}\left[\log\pi_{\phi_t}(a_t|s_t)\right]$$

(plus a re-parameterized-gradient correction from $a_t \sim \pi_\phi$, which the paper handles via the squashed-Gaussian reparameterization standard in SAC). The full meta-gradient is then

$$\frac{d L_{\text{meta}}}{d \alpha_t} = \frac{\partial L_{\text{meta}}}{\partial \phi_{t+1}}\,\frac{\partial \phi_{t+1}}{\partial \alpha_t}$$

with $\partial L_{\text{meta}}/\partial \phi_{t+1} = -\nabla_{\phi_{t+1}} \mathbb{E}_{s_0\sim\mathcal{D}_0}[Q_{\omega_t}(s_0, \pi^{\text{det}}_{\phi_{t+1}}(s_0))]$.

**Why RMSProp not Adam.** The chain rule above requires differentiating *through the optimizer's update step*. Adam's update rule includes square-root and bias-correction terms whose gradients are numerically fragile; RMSProp is simpler. This is a known issue in differentiable-optimization / meta-learning work.

**Why old $Q_{\omega_t}$ not $Q_{\omega_{t+1}(\alpha_t)}$.** If you also differentiate through the critic update, you pick up $\partial \omega_{t+1}/\partial \alpha_t$ from $L_Q$'s dependence on $\alpha$ (the soft-Bellman target $\hat Q - \alpha \log \pi$ depends on $\alpha$). This adds a second chain-rule path with its own numerical hazards. Using $Q_{\omega_t}$ in the meta-loss cuts this path and was found to be much more stable empirically.

### Contrast with the SAC-v2 Lagrangian update

SAC-v2 derives $\alpha$'s update by treating SAC's entropy term as a constraint and using **dual gradient descent**:

$$L_{\text{SAC-v2}}(\alpha) = \mathbb{E}_{s_t\sim\mathcal{D},a_t\sim\pi_\phi}\left[-\alpha \log\pi(a_t|s_t) - \alpha\bar{H}\right]$$

$$\nabla_\alpha L_{\text{SAC-v2}} = \mathbb{E}\left[-\log\pi(a_t|s_t) - \bar{H}\right] = -\left(\mathcal{H}(\pi(\cdot|s_t)) - \bar{H}\right)$$

so $\alpha$ moves to push the policy's entropy *toward* $\bar{H}$. Two key differences from Meta-SAC:

1. **Signal driving $\alpha$.** SAC-v2: "is policy entropy above/below $\bar{H}$?" Meta-SAC: "would a smaller/larger $\alpha$ produce a greedy policy with higher $Q$-value at initial states?"
2. **Hyperparameter footprint.** SAC-v2 introduces $\bar{H}$ per task. Meta-SAC uses one constant hyperparameter set across all four Mujoco tasks.

Two prices Meta-SAC pays:

- **Compute.** Each $\alpha$ update requires differentiating through an inner SGD step on $\phi$ — roughly one extra forward+backward pass on the policy and the meta-Q evaluation.
- **Theoretical guarantee.** SAC-v2 has a (partial) convergence story from dual ascent. Meta-SAC has none — but the paper notes SAC-v2's guarantee already relies on convexity, which fails for NNs anyway.

### Where Meta-SAC's $\alpha$-schedule ends up

Late in training, $\alpha \to 0$ — the soft-Bellman target collapses to the classic Bellman target and SAC becomes DDPG-like. This is *not* what SAC-v2 produces (its $\alpha$ plateaus), and is *not* what naive $\alpha$-decay produces (Appendix E shows fixed-decay SAC-v1 underperforms). The conclusion is: **the trajectory of $\alpha$ — high early, low late — matters more than any fixed value, and that trajectory is task-dependent in ways the metagradient discovers automatically.**

---

# Paper 2 — Lin et al. (2020), Cat-SAC: SAC with Curiosity-Aware Entropy Temperature

**PDF:** `docs/project/references/Temperature/sources/Lin et al. 2020 - Cat-sac - Soft actor-critic with curiosity-aware entropy temperature.pdf`
**Venue:** Submitted to ICLR 2021 (under double-blind review; arXiv 2020)
**Authors:** Anonymous at submission (Lin et al. is the inferred / archived attribution)

## Cat-SAC: Backbone (section-ordered)

### Abstract
- Critique of SAC-v2: a single $\alpha$ applied indiscriminately to every state ignores that some states *deserve* more exploration than others.
- Proposal: Curiosity-Aware entropy Temperature (CAT-SAC) — state prediction error (curiosity) drives a *per-state* temperature.
- Curiosity is **added to the target entropy** (so unfamiliar states get a higher entropy target) and an **instance-level $\alpha_\delta(s)$** is then learned to satisfy that augmented per-state target.
- Also proposes **X-RND**, a contrastive-learning variant of Random Network Distillation that works better for *feature inputs* than vanilla RND.
- Significant sample-efficiency gains on Mujoco vs SAC, TD3, SUNRISE, PETS, METRPO, POPLIN.

### Section 1 — Introduction
- Sample inefficiency of deep RL; exploration/exploitation balance the key bottleneck. SAC objective recapped:
  $$\pi^* = \arg\max_\pi \mathbb{E}_{s_t,a_t\sim\rho_\pi}\left[\sum_t r_t + \alpha \mathcal{H}(\pi(\cdot|s_t))\right] \quad (\text{Eq. 1})$$
- SAC-v2's auto-$\alpha$ uses a single target entropy $\tilde{H}$ across all transitions — neglects state particularity. Intuition: "when playing a new game, humans explore a lot at the beginning, then exploit once they understand the basic logic." Different states need different exploration pressure.
- Two contributions:
  1. CAT-SAC: per-state curiosity injected into the target entropy plus a per-state temperature head.
  2. X-RND: a contrastive-loss variant of RND that prevents the curiosity for "unvisited" feature states from being pulled down by interpolation with visited states.

### Section 2 — Related work
- Maximum-entropy RL lineage (Ziebart 2008; Haarnoja 2017–2019).
- Intrinsic-bonus exploration: count-based (Bellemare 2016; Tang 2017), prediction-error (Pathak 2017; Burda 2018a). RND (Burda 2018b) addresses the "noisy TV" problem by predicting the output of a fixed random network rather than dynamics.
- Undirected exploration (Tokic 2010 and follow-ups): increase action variance at unfamiliar states. CAT-SAC's contribution is to inject this into SAC's automatic temperature tuning.

### Section 3 — Preliminaries

**3.1 RL framing.** Standard MDP. $s_t$, $a_t$, $r_t$, $s_{t+1} \sim p_e(s_t, a_t)$, discounted return $\eta_t = \sum_{k=0}^\infty \gamma^k r_{t+k}$.

**3.2 SAC** — same as before. Critic loss:

$$L_{\text{critic}}(\theta) = \mathbb{E}_{s_t,a_t\sim\mathcal{D}}\left[\left\|Q_\theta(s_t,a_t) - Q_{\text{target}}(s_t,a_t)\right\|_2^2\right] \quad (\text{Eq. 2})$$

$$Q_{\text{target}}(s_t,a_t) = r_t + \gamma \mathbb{E}_{s_{t+1}, a_{t+1}\sim\pi_\phi}\left[\hat{Q}(s_{t+1},a_{t+1}) - \alpha\log\pi_\phi(a_{t+1}|s_{t+1})\right] \quad (\text{Eq. 3})$$

$$L_{\text{actor}}(\phi) = \mathbb{E}_{s_t\sim\mathcal{D}, a_t\sim\pi_\phi}[\alpha\log\pi_\phi(a_t|s_t) - Q_\theta(s_t,a_t)] \quad (\text{Eq. 4})$$

SAC-v2 alpha update with target entropy $\tilde{H}$:

$$L(\alpha) = \mathbb{E}_{s_t\sim\mathcal{D}, a_t\sim\pi_\phi}\left[-\alpha\left(\log\pi(a_t|s_t) + \tilde{H}\right)\right] \quad (\text{Eq. 5})$$

**3.3 RND.** Curiosity from prediction error between a trainable network $f_\omega(s_t)$ and a fixed random network $\hat{f}(s_t)$:

$$c_\omega(s_t) = \|f_\omega(s_t) - \hat{f}(s_t)\|_2^2 \quad \text{and} \quad L(\omega) = \mathbb{E}_{s_t\sim\mathcal{D}}[c_\omega(s_t)] \quad (\text{Eq. 6})$$

Training $f_\omega$ minimizes $c$ on visited states; for unvisited states, $f_\omega$ has no signal so $c$ stays high $\Rightarrow$ "curiosity" of a state = how unfamiliar.

### Section 4 — Cat-SAC method

Three ingredients:

**4.1 Curiosity-augmented target entropy.** Replace the single $\tilde{H}$ with per-state $h(s)$ by adding zero-mean normalized curiosity:

$$h(s) = \tilde{H} + \frac{c(s) - \mu}{\sigma} \quad (\text{Eq. 7})$$

where $\mu, \sigma$ are running mean/std of $c(s)$. By construction $\mathbb{E}_s[h(s)] = \tilde{H}$ — the new target entropy is consistent with the old one *in expectation*. Unfamiliar state ($c$ large) $\Rightarrow$ target entropy pushed up $\Rightarrow$ more exploration at that state.

**4.2 Instance-level entropy temperature.** Make $\alpha$ state-dependent:

$$\delta^* = \arg\min_\delta \mathbb{E}_{s_t\sim\mathcal{D}, a_t\sim\pi_\phi}\left[-\alpha_\delta(s_t)\left(\log\pi_\phi(a_t|s_t) + h(s_t)\right)\right] \quad (\text{Eq. 8})$$

Two implementation issues:

1. If $\alpha$ were state-independent, Eq. 8 would reduce to Eq. 5 — no effect (because the zero-mean curiosity cancels in expectation). So state-dependence is essential.
2. If $\alpha_\delta(s_t)$ were *arbitrarily* state-dependent (e.g., a neural net of $s_t$), the optimization is trivial: pick $\alpha$ extremely high or extremely low to make the loss as negative as possible depending on the sign of $\log\pi(a_t|s_t) + h(s_t)$. This forces $-\log\pi(a_t|s_t) \to h(s_t)$ rapidly, which (as Haarnoja warns) hampers policy flexibility.

The fix: don't condition $\alpha$ on $s$ directly. Condition on the **discretized curiosity** $c(s)$ rounded to the nearest integer:

$$\alpha_\delta(s_t) = g_\delta(c(s_t)) \quad (\text{Eq. 9})$$

where $g_\delta$ is a simple **linear layer** (Appendix details: it's effectively a lookup table — a list of scalar variables indexed by integer curiosity value from $-|\tilde{H}|$ to $+|\tilde{H}|$).

This is a deliberate **information bottleneck**: $\alpha$ depends on $s$ only through $c(s)$, and $c(s)$ is discretized. The paper frames this as "stratified SAC" — states are grouped into curiosity strata, each stratum shares a temperature parameter, each stratum gets the same SAC-v2-style automatic update.

**4.3 X-RND: contrastive RND for feature inputs.** The authors find vanilla RND degenerates on feature inputs (low-dimensional state vectors, not images): unvisited feature states close to visited ones get low prediction errors too — RND can't tell them apart.

X-RND adds a contrastive loss:

$$L(\omega) = \mathbb{E}_{s_t\sim\mathcal{D}}\left[c_\omega(s_t) + \beta \max(m - c_\omega(s'), 0)\right] \quad (\text{Eq. 10})$$

where $s'$ is a synthesized "unvisited" state constructed by entry-wise blending of two visited states: $s' = s_1 + \epsilon \odot (s_2 - s_1)$ with $s_1, s_2 \sim \mathcal{D}$, $s_1 \neq s_2$, and $\epsilon \sim U[0, 1.5]^{|s|}$. Note the upper bound 1.5 (not 1.0) — extrapolation, not interpolation. $m$ is the curiosity margin; $\beta$ weights the contrastive term.

### Section 5 — Experiments

**5.1 Mujoco results.** Cat-SAC vs METRPO, PETS, POPLIN-A, POPLIN-P, TD3, SAC, SUNRISE at 200K timesteps. Cat-SAC wins on all four tasks (Cheetah, Walker, Hopper, Ant). Improvement is most pronounced on Cheetah and Walker (~6160 and ~2666 vs SAC's 5471 and 1419).

**5.2 X-RND vs RND on Swiss Roll maze.** Toy environment with position-only feature input. After visiting 10 unique blocks, RND wrongly assigns low curiosity (= "familiar") to distant unvisited blocks; X-RND maintains high curiosity for genuinely unvisited blocks with a sharp diagonal "explored/unexplored" boundary.

**5.3 Ablation.**
- Vary $\beta$ (contrastive weight): positive $\beta$ helps; too large hurts (the visited-state RND loss has trouble decreasing).
- Vary curiosity margin $m$: Ant is sensitive, Hopper less so.
- Component-wise ablation: real-valued curiosity instead of discretized — degrades. RND/ICM instead of X-RND — degrades. **`rev`** (subtract curiosity from target entropy instead of add — reverse the design intent) — degrades, sometimes below baseline. This is the strongest piece of evidence for the *direction* of the curiosity injection.

**5.4 Sparse-reward BipedalWalker.** Two interval settings (5-step easy, 10-step hard). On the easy task both SAC and Cat-SAC solve it; Cat-SAC faster. On the hard task neither pure-entropy method handles the sparsity well; intrinsic-bonus methods (RND, ICM augmenting the reward) outperform. Honest discussion: Cat-SAC is for *exploration via variance adjustment*, not for *intrinsic-bonus reward shaping*; dense-reward tasks are where it shines.

### Section 6 — Conclusion
Frames Cat-SAC as a unified framework for curiosity + maximum-entropy RL: large entropy in unfamiliar states, small entropy in familiar ones, achieved via the instance-level temperature head.

### Appendix
- **Algorithm 1.** Per-step loop: collect transition; on episode end, sample real states $\{s\}$ and synthesized $\{s'\}$, update $f_\omega$ via Eq. 10, update running $\mu, \sigma$ of $c$. On update step: sample batch $\mathcal{B}$, compute $c(s_t)$ and $\alpha_\delta(s_t)$ for the batch, do standard SAC critic + actor updates (using $\alpha_\delta(s_t)$ wherever vanilla SAC uses $\alpha$), update target networks, build $h(s_t)$ via Eq. 7, update $\alpha_\delta$ via Eq. 8.
- **Instance-level $\alpha_\delta$ implementation.** "We do not adopt deep neural networks for the instance-level entropy temperature. Instead, we initialize a list of scalar variables. Each variable is assigned to an integer index, ranging from $-|\tilde{H}|$ to $|\tilde{H}|$. Given the discrete curiosity input, we match it with the variable with the closest index..." — i.e., $g_\delta$ is a *bin-indexed scalar table*, not a neural network. For real-valued curiosity ablation, two adjacent bins are interpolated by weighted-sum so the sum of weighted indices equals the curiosity value.
- **Hyperparameters.** Standard SAC (256 hidden units, 2 layers, ReLU, $\gamma=0.99$, Adam, lr $3\times10^{-4}$). X-RND feature size 128, 4 hidden layers in $f_\omega$. Curiosity margin $m=10$ for Mujoco (much smaller for the maze toy, $m=0.2$). Importance weight $\beta=0.1$. Initial temperature 1. Target entropy $\tilde{H} = -|\mathcal{A}|$ (Haarnoja's heuristic, kept).

---

## Cat-SAC: Phase 1 — Undergraduate synthesis

**The limit of vanilla auto-$\alpha$ this paper fixes.** SAC-v2's automatic temperature is a single number applied to every state in the environment. But intuitively, some states are familiar — the agent has been there many times — and others are brand-new. A familiar state warrants exploitation (low $\alpha$, near-deterministic action); an unfamiliar state warrants exploration (high $\alpha$, more random action). Applying the same $\alpha$ everywhere wastes exploration budget in familiar regions and underexplores novel ones.

**The proposal in one sentence.** Build a *curiosity signal* $c(s)$ from a Random Network Distillation predictor (high curiosity = state the agent doesn't predict well = unfamiliar), bin it into integer levels, and have a lookup-table-style head $g_\delta$ produce a per-state temperature $\alpha_\delta(s) = g_\delta(c(s))$ — trained by the SAC-v2-style entropy-target rule but with the target itself shifted up at high-curiosity states and down at low-curiosity states.

**Key empirical result.** Beats SAC, TD3, SUNRISE, and several model-based baselines on all four Mujoco tasks at 200K timesteps. Most striking on Walker (1926 → 2666) and Hopper (2602 → 3011). Ablation that *reverses* the direction of curiosity injection (low entropy at unfamiliar states) drops performance below the baseline — a strong sign the mechanism is real, not just regularization.

---

## Cat-SAC: Phase 2 — Graduate deep-dive

### The curiosity signal

The curiosity is a state-only function, no action involved:

$$c(s) = \|f_\omega(s) - \hat{f}(s)\|_2^2$$

with $\hat{f}$ frozen random and $f_\omega$ trained to minimize $c$ on the replay buffer's *visited* states. For feature inputs, vanilla RND degenerates because the contrast between visited and nearby-unvisited is weak; X-RND adds a hinge contrastive loss:

$$L_{\text{X-RND}}(\omega) = \mathbb{E}_{s\sim\mathcal{D}}\left[c_\omega(s) + \beta \max\big(m - c_\omega(s'),\, 0\big)\right]$$

where $s' = s_1 + \epsilon \odot (s_2 - s_1)$, $\epsilon \sim U[0,1.5]^{|s|}$, $s_1, s_2 \sim \mathcal{D}, s_1 \neq s_2$. The 1.5 upper bound on $\epsilon$ means $s'$ can extrapolate slightly *beyond* the convex hull of visited states, not just interpolate — important for sustaining curiosity on the boundary of the explored region.

The curiosity is normalized:

$$\tilde{c}(s) = \frac{c(s) - \mu}{\sigma}$$

with running $\mu, \sigma$, then *discretized to the nearest integer* (call this $\lfloor\tilde{c}(s)\rceil$). All downstream temperature lookups use this discrete level.

### The state-dependent temperature function

The temperature is a lookup table indexed by integer curiosity level:

$$\alpha_\delta(s) = g_\delta\big(\lfloor\tilde{c}(s)\rceil\big)$$

where $g_\delta$ is a "linear layer" — in the implementation, a vector of scalars $\delta = (\delta_{-K}, \delta_{-K+1}, \dots, \delta_K)$ with $K = |\tilde{H}|$ (target entropy magnitude), and $\alpha_\delta(s) = \delta_{\lfloor\tilde{c}(s)\rceil}$ (or, in the real-valued ablation, a linear interpolation between adjacent bins).

Crucially:

- **The $\alpha$ head sees only curiosity, not the raw state.** This is the information bottleneck preventing the trivial-optimum failure mode described in Section 4.2.
- **The head is non-parametric in $s$**; the only parameters are the per-bin scalars $\delta_k$.
- Effective architecture: SAC-v1 had one $\alpha$; SAC-v2 has one $\alpha$ updated by dual gradient descent; Cat-SAC has $|2\tilde{H}|+1$ values of $\alpha$ each updated by dual gradient descent, one per curiosity stratum.

### How it integrates with the SAC actor / critic losses

The trick: *wherever vanilla SAC uses $\alpha$, Cat-SAC substitutes $\alpha_\delta(s_t)$* — i.e., the temperature is **state-dependent at evaluation time** in both the critic target and the actor loss.

**Critic target (modified Eq. 3):**

$$Q_{\text{target}}(s_t, a_t) = r_t + \gamma \mathbb{E}_{s_{t+1}, a_{t+1}\sim\pi_\phi}\left[\hat{Q}(s_{t+1}, a_{t+1}) - \alpha_\delta(s_{t+1})\log\pi_\phi(a_{t+1}|s_{t+1})\right]$$

Note: $\alpha_\delta$ is evaluated at $s_{t+1}$, the *next* state — this is what the entropy-bonus regularizes.

**Actor loss (modified Eq. 4):**

$$L_{\text{actor}}(\phi) = \mathbb{E}_{s_t\sim\mathcal{D}, a_t\sim\pi_\phi}\left[\alpha_\delta(s_t)\log\pi_\phi(a_t|s_t) - Q_\theta(s_t, a_t)\right]$$

The actor's exploration pressure $\alpha_\delta(s_t)\log\pi(a_t|s_t)$ now varies state-by-state.

**Temperature update (Eq. 8):**

$$L(\delta) = \mathbb{E}_{s_t\sim\mathcal{D}, a_t\sim\pi_\phi}\left[-\alpha_\delta(s_t)\big(\log\pi_\phi(a_t|s_t) + h(s_t)\big)\right]$$

with $h(s) = \tilde{H} + (c(s) - \mu)/\sigma$. Gradient w.r.t. a single bin $\delta_k$:

$$\nabla_{\delta_k} L(\delta) = \mathbb{E}_{s_t\sim\mathcal{D}_k, a_t\sim\pi_\phi}\left[-\big(\log\pi_\phi(a_t|s_t) + h(s_t)\big)\right]$$

where $\mathcal{D}_k = \{s_t \in \mathcal{D} : \lfloor\tilde{c}(s_t)\rceil = k\}$ — i.e., each bin is updated by the SAC-v2 Lagrangian rule, but **restricted to the states in its curiosity stratum** and with the **augmented per-state target** $h(s_t)$. This is the "stratified SAC" framing: one SAC-v2 update per stratum.

### Is this still consistent with the soft-Bellman contraction?

A subtle and important question. The original SAC convergence proof (Haarnoja 2018a) shows that the soft-Bellman operator

$$(\mathcal{T}^\pi Q)(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim p}\left[V(s')\right], \quad V(s) = \mathbb{E}_{a\sim\pi}[Q(s, a) - \alpha\log\pi(a|s)]$$

is a $\gamma$-contraction *in $Q$* for any fixed $\pi$ and any fixed $\alpha > 0$. The contraction proof uses $\alpha$ as a scalar — but careful inspection shows that the only properties of $\alpha$ used are (i) positivity and (ii) **independence from $Q$**. The proof does *not* rely on $\alpha$ being constant across states.

So a **state-dependent but $Q$-independent $\alpha(s)$** preserves the contraction:

$$(\mathcal{T}^\pi Q)(s, a) = r(s, a) + \gamma \mathbb{E}_{s'\sim p}\left[\mathbb{E}_{a'\sim\pi}\left[Q(s', a') - \alpha(s')\log\pi(a'|s')\right]\right]$$

is still a $\gamma$-contraction in $Q$ in the sup-norm, because the per-state $\alpha(s')$ only changes the entropy bonus at $s'$ — it does not couple $Q$-values across states or break the convex-combination structure of the Bellman operator. The corresponding "soft" value function is

$$V(s) = \mathbb{E}_{a\sim\pi}[Q(s, a) - \alpha(s)\log\pi(a|s)] = \mathbb{E}_{a\sim\pi}[Q(s, a)] + \alpha(s)\mathcal{H}(\pi(\cdot|s))$$

and the corresponding soft policy improvement step is still the closed-form

$$\pi_{\text{new}}(a|s) \propto \exp\!\left(\frac{Q(s, a)}{\alpha(s)}\right)$$

i.e., a state-dependent **softmax temperature** — exactly what one would expect.

**Caveats / what *is* sacrificed:**

1. **Cat-SAC does not prove this contraction.** The paper does not include a contraction theorem; it relies on the empirical track record. The above derivation is the natural extension and appears correct, but a careful proof would require checking that $\alpha(s)$ is bounded away from zero (which the bin-indexed scalar table guarantees if initialized positively) and that the entropy bonus remains bounded (which requires the policy entropy to stay bounded — standard for Gaussian / squashed-Gaussian actors).
2. **The dependence of $\alpha$ on a *learned* curiosity signal $c(s)$ couples the value-iteration loop to the curiosity-network's learning dynamics.** During training, $c(s)$ shifts as $f_\omega$ learns to match $\hat{f}$, so $\alpha_\delta(s)$ shifts even with $\delta$ held fixed. This is *not* in any classical convergence-proof setting; the contraction property holds for any single snapshot of $c$, but the snapshot is drifting. In practice this is no different from the standard problem of $\alpha$ drifting in SAC-v2 (which also moves under dual ascent during training).
3. **No formal claim that the meta-learning of $\delta$ converges to a useful place.** Same as SAC-v2 — the Lagrangian update of $\delta$ is heuristic in the deep-NN setting.

### Comparison to Meta-SAC

| | Meta-SAC | Cat-SAC |
|---|---|---|
| What is adaptive | $\alpha$ scalar | $\alpha$ as state-conditional function |
| Adaptation signal | Greedy-evaluation $Q$ at initial states | Curiosity $c(s)$ from RND |
| Adaptation mechanism | Metagradient through inner SGD step | SAC-v2 Lagrangian update on per-bin scalar table |
| Architecture | Single $\log\alpha$ scalar | Lookup table $\delta_k$ over curiosity bins |
| Hyperparameters introduced | None (vs SAC-v2's $\bar H$) | $\beta$ (contrastive weight), $m$ (curiosity margin), $|\tilde H|$ (range of bins), RND architecture |
| Soft-Bellman contraction | Preserved (single $\alpha$) | Preserved if $\alpha(s) > 0$ and $Q$-independent |
| Empirical winner | Humanoid-v2 (+10%) | All four Mujoco tasks at 200K timesteps |
| State-dependence | No | **Yes — this is the closest published precedent for modulator-conditioned $\alpha$** |

---

# Cross-paper synthesis — implications for the project's modulator-conditioned temperature head

The project's design question: replace SAC's single $\alpha$ with $\alpha(s, m)$ where $m$ is a modulator signal (neuromodulator-style). Cat-SAC is the most directly relevant precedent in the SAC literature; Meta-SAC is relevant for the *adaptation mechanism* if the modulator itself needs to be tuned. Carry-over details flagged below.

### Architecture carry-overs

1. **Use a bin-indexed scalar table, not a neural net, for the temperature head — at least for the first pass.** Cat-SAC explicitly considered a neural-network $\alpha(s)$ and rejected it because the Lagrangian update has a trivial-optimum failure mode: a sufficiently expressive $\alpha(s)$ can be pushed to $\pm\infty$ to make the loss arbitrarily negative, which then forces $-\log\pi(a|s) \to h(s)$ rapidly and destroys policy flexibility. The fix is an information bottleneck: condition only on a low-dimensional, discretized "context" (curiosity bin in Cat-SAC's case). For the project, the analogue is: condition $\alpha$ only on the modulator value $m$ (low-dimensional by design), and consider discretizing $m$ into bins. If $m$ is already discrete (e.g., a pain level), the design maps directly to Cat-SAC's table.
2. **Keep $\alpha$ positive.** Cat-SAC parameterizes via $\log\alpha$ and clips. Meta-SAC parameterizes via $\log\alpha$ and clips at $\log\alpha \leq 0$ (so $\alpha \in (0, 1]$). The contraction proof needs $\alpha > 0$; the practical training needs $\alpha$ bounded above to avoid the actor going fully random.
3. **The temperature appears in both the critic target and the actor loss — and must be evaluated at the right state.** In Cat-SAC, the critic target uses $\alpha_\delta(s_{t+1})$ (next state, because that's where the next-step entropy bonus lives); the actor loss uses $\alpha_\delta(s_t)$. The project's implementation must replicate this — *don't* pass the same temperature instance to both losses if the modulator changes per timestep.
4. **The soft-policy-improvement closed form $\pi(a|s) \propto \exp(Q(s,a)/\alpha(s))$ is still well-defined with $\alpha(s, m)$.** The contraction analysis carried out for Cat-SAC above transfers if the modulator signal $m$ is (a) a function of $s$ alone (so $\alpha(s, m(s)) = \alpha'(s)$) or (b) part of the augmented state (so the MDP is over $(s, m)$ jointly). Either way the formal property is preserved.

### Training-loop carry-overs

5. **Update the temperature parameters via the SAC-v2 Lagrangian rule restricted to the relevant stratum.** Don't try to backprop the temperature through a meta-objective unless the project's specific scientific question requires it (Meta-SAC's metagradient is more sophisticated but also more fragile — requires RMSProp, gradient clipping, careful old-vs-new $Q$ choice). Start with the simple per-stratum dual-ascent rule.
6. **If the modulator is fixed/exogenous (e.g., set by the environment, not learned), then no metagradient is needed — Cat-SAC's mechanism applies essentially unchanged.** If the modulator is *learned* by the agent (e.g., a neuromodulator head that the agent itself updates), Meta-SAC's metagradient becomes relevant for the modulator's parameters — and the four design tricks in Meta-SAC (deterministic eval policy, initial-state buffer $\mathcal{D}_0$, soft-$Q$ not classic-$Q$, old-$Q$ not new-$Q$) all carry over verbatim.
7. **Resample fresh minibatches for the differentiated-through update.** A small but consistent finding across both Meta-SAC and the broader meta-learning literature (MAML; Zheng et al. 2018): if you train $\alpha$ on the same minibatch you used to update $\phi$ and $\omega$, you overfit to that minibatch. Resample $\mathcal{B}'$ from the replay buffer for the temperature update. Cheap and improves performance.
8. **The target entropy can stay at Haarnoja's $-|\mathcal{A}|$ heuristic.** Both papers keep this; only Cat-SAC modifies it per-state with zero-mean curiosity, preserving the *expectation* across states. For the project, the analogue is to keep some baseline $\tilde{H}$ and add a zero-mean modulator-derived shift — preserving SAC-v2-compatibility in expectation while introducing state-dependent variation around it.

### Specific risks the project should pre-empt

9. **Trivial-optimum risk for the temperature head.** If the modulator-conditioned head is over-expressive (e.g., a deep NN over $(s, m)$ jointly), the Lagrangian update will push it to extremes. Cat-SAC's mitigation — discretize the conditioning signal — should be the default.
10. **Drift in the conditioning signal.** Cat-SAC's curiosity $c(s)$ drifts as $f_\omega$ learns; this is benign in practice but breaks classical convergence settings. If the project's modulator signal is *learned* and drifts, expect similar empirical robustness but no formal guarantee.
11. **State-dependence in the critic target induces a state-dependent soft Bellman operator** — well-defined and contraction-preserving as shown above, but means the target $Q$ now implicitly encodes the modulator's value function at *every future state*. If the modulator changes meaningfully over an episode, the agent's effective horizon is the *joint* horizon of reward and modulator dynamics. Worth pre-computing whether the modulator's "natural" timescale is comparable to or shorter than $\gamma$'s effective horizon.
12. **Cat-SAC's empirical wins are on dense-reward Mujoco; on sparse-reward BipedalWalker, the method's advantage shrinks.** Honest acknowledgment from the authors. If the project's environment has sparse rewards, the modulator-conditioned $\alpha$ should be paired with a separate intrinsic-bonus mechanism — *don't* expect the variance-modulation mechanism to substitute for reward shaping.

### What does *not* carry over directly

- Cat-SAC's X-RND curiosity model is a curiosity-*signal* engineering contribution. If the project's modulator is not curiosity-derived (e.g., a pain signal, an interoceptive signal, or another neuromodulator-inspired signal), X-RND has no direct relevance. The architecture-level lessons (table-based head, information bottleneck on conditioning) are what generalize.
- Meta-SAC's RMSProp choice is specific to the metagradient pipeline. If the project does not metagradient, Adam is fine.

---

## Handoff notes

- This shard covers two of the Temperature corpus's ten PDFs. Sibling shards (foundations: Haarnoja x2; softmax operators: Asadi & Littman; Song et al.; KL-regularized / Munchausen: Vieillard x2; Zhu et al.; entropy impact: Ahmed et al.) are not addressed here.
- If a deeper graduate-level treatment of either paper is needed (e.g., a formal contraction proof for Cat-SAC's state-dependent $\alpha$, or a full derivation of Meta-SAC's metagradient through both $\phi$ and $\omega$ updates), recommend the `literature-deepdive` agent.
- The architecture-carry-over list above is design-relevant but is *not* an implementation plan. If the user wants to translate (1)–(12) into a concrete `senior-developer` issue plan, that handoff should be made explicitly.

