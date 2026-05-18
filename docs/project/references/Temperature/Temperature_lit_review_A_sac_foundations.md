# Temperature literature review — Part A: SAC foundations

**Reviewer:** literature-reviewer
**Date:** 2026-05-19
**Corpus shard:** A (SAC foundations — 2 papers)
**Project context:** the broader Temperature corpus collects work on the entropy coefficient $\alpha$ in maximum-entropy RL, with an eye toward making $\alpha$ a **modulated, per-context variable** (e.g. driven by an interoceptive "precision" signal) rather than a fixed hyperparameter. This shard reviews the two Haarnoja Soft Actor-Critic (SAC) papers that establish (a) the canonical max-entropy RL objective and (b) the Lagrangian dual that lets $\alpha$ be learned online — both of which are the direct mathematical precedents for any modulator-conditioned temperature head.

## Purpose of this document

Maximum-entropy reinforcement learning is a small but consequential change to the standard RL objective. Standard RL trains a policy to maximize expected cumulative reward, $\sum_t \mathbb{E}[r(s_t, a_t)]$. Maximum-entropy RL adds a bonus for *randomness in the policy itself*, becoming $\sum_t \mathbb{E}[r(s_t, a_t) + \alpha \mathcal{H}(\pi(\cdot \mid s_t))]$, where $\mathcal{H}$ is Shannon entropy of the action distribution and $\alpha$ is the **temperature**: a positive scalar that says "how much do I value being random vs. how much do I value reward?".

Why bother? Three reasons that the Haarnoja papers spell out and that matter for our project:

1. **Exploration.** A policy that maximizes only reward will collapse to a sharp distribution as soon as one action looks slightly better; an entropy bonus keeps a long tail of "what if I try this other thing?" alive, which discovers reward faster on hard tasks.
2. **Robustness / multi-modal solutions.** If two action sequences are equally good, the standard RL agent picks one arbitrarily; the max-entropy agent keeps both with equal probability, which makes it less brittle when the environment shifts.
3. **A handle for modulation.** The temperature $\alpha$ is the single knob that interpolates between "act randomly" ($\alpha \to \infty$) and "act greedily" ($\alpha \to 0$). If $\alpha$ can be made **state-dependent** or **modulator-dependent** rather than a fixed number, you get a policy whose exploration-vs-exploitation balance is controlled by something outside the policy itself — which is exactly the move our project wants to make for an interoceptive signal.

The two papers reviewed here pin down the math behind those three statements. **Paper 1 (Haarnoja et al. 2018, ICML)** treats $\alpha$ as a fixed hyperparameter and proves that the resulting "soft policy iteration" converges to a unique optimum. **Paper 2 (Haarnoja et al. 2018, arXiv follow-up)** turns $\alpha$ itself into a learned variable by re-deriving max-entropy RL as a *constrained* optimization with a minimum-entropy constraint — and the temperature falls out as the Lagrange multiplier on that constraint. That second step is the mathematical scaffolding our project will lean on most.

---

## Table of contents

- [Paper 1 — Haarnoja et al. 2018a (ICML): Soft Actor-Critic — Off-Policy Maximum Entropy Deep RL with a Stochastic Actor](#paper-1--haarnoja-et-al-2018a-icml-soft-actor-critic--off-policy-maximum-entropy-deep-rl-with-a-stochastic-actor)
  - [Phase 1: Foundational overview (undergraduate-level)](#phase-1-foundational-overview-undergraduate-level-paper-1)
  - [Phase 2: Graduate-level deep dive](#phase-2-graduate-level-deep-dive-paper-1)
  - [Appendix: Section-by-section backbone (Paper 1)](#appendix-section-by-section-backbone-paper-1)
- [Paper 2 — Haarnoja et al. 2018b (arXiv): Soft Actor-Critic Algorithms and Applications](#paper-2--haarnoja-et-al-2018b-arxiv-soft-actor-critic-algorithms-and-applications)
  - [Version notes / delta against Paper 1](#version-notes--delta-against-paper-1)
  - [Phase 1: Foundational overview (undergraduate-level)](#phase-1-foundational-overview-undergraduate-level-paper-2)
  - [Phase 2: Graduate-level deep dive](#phase-2-graduate-level-deep-dive-paper-2)
  - [Appendix: Section-by-section backbone (Paper 2)](#appendix-section-by-section-backbone-paper-2)
- [Cross-paper synthesis and project implications](#cross-paper-synthesis-and-project-implications)

---

## Paper 1 — Haarnoja et al. 2018a (ICML): Soft Actor-Critic — Off-Policy Maximum Entropy Deep RL with a Stochastic Actor

**PDF:** `docs/project/references/Temperature/sources/Haarnoja et al. 2018 - Soft Actor-Critic - Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor.pdf`
**Venue:** ICML 2018 (PMLR 80)
**Authors:** Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine (Berkeley AI Research)

### Phase 1: Foundational overview (undergraduate-level) — Paper 1

**The problem the paper attacks.** Deep RL on continuous-action tasks (robot locomotion, manipulation) suffers from two cooperating failure modes: (a) **sample inefficiency** — on-policy algorithms like TRPO/PPO/A3C must throw away data after each gradient step, requiring millions of samples for moderately complex tasks; (b) **brittleness** — the leading off-policy continuous-control algorithm at the time, DDPG, is notoriously sensitive to hyperparameters and random seeds, often failing outright on high-dimensional benchmarks like the 21-dimensional Humanoid. The paper claims the right fix combines three ingredients: an actor-critic architecture (separate policy and value networks), off-policy training (reuse a replay buffer), and *entropy maximization* (make the policy itself stochastic by giving it a bonus for being unpredictable).

**The core idea — soft policy iteration.** Standard policy iteration alternates two steps:
1. **Policy evaluation** — given a fixed policy, compute its Q-values via the Bellman backup $\mathcal{T}^\pi Q(s,a) = r + \gamma \mathbb{E}[Q(s', a')]$.
2. **Policy improvement** — make a new policy that's greedy with respect to those Q-values.

SAC replaces both pieces with "soft" versions:
1. **Soft Bellman backup** — same form, but the bootstrapped value $V(s')$ now subtracts a $\log \pi(a' \mid s')$ entropy term, so transitions to high-entropy states are valued more.
2. **Soft policy improvement** — instead of picking the greedy action, the new policy is set to the *softmax of the Q-function*, $\pi_{\text{new}}(a \mid s) \propto \exp(Q(s,a))$, which is the maximum-entropy distribution at a given expected Q.

The paper proves that iterating these two soft steps in the tabular case converges to a unique fixed point — the *optimal max-entropy policy*. That convergence guarantee is the theoretical anchor.

**The practical algorithm.** In a real continuous-action problem you can't do exact tabular updates, so SAC parameterizes three networks — a Q-network, a value network, and a Gaussian policy — and trains all three with stochastic gradients on a replay buffer:
- Q-network is trained on the soft Bellman residual (Eq. 7).
- Value network is trained to match $\mathbb{E}_{a \sim \pi}[Q(s,a) - \log \pi(a \mid s)]$ (Eq. 5).
- Policy network is trained by minimizing the KL divergence between the current policy and the softmax-of-Q, implemented via the **reparameterization trick** (sample $\epsilon \sim \mathcal{N}(0, I)$, then $a = f_\phi(\epsilon; s)$) so the gradient flows through the sample.

A second Q-network is added (the "double Q" trick from TD3) to mitigate positive bias in Q-value estimates.

**Key empirical findings.** On six MuJoCo continuous-control benchmarks (Hopper, Walker2d, HalfCheetah, Ant, Humanoid, Humanoid-rllab):
- SAC matches DDPG/PPO/SQL/TD3 on the easy tasks.
- SAC **substantially outperforms** all baselines on the hard high-dimensional tasks (Ant, both Humanoids), where DDPG often fails to make any progress at all.
- SAC has dramatically lower variance across random seeds than DDPG — the stochastic actor *stabilizes* training, not just exploration.

**The "reward scale" finding (critical for our project).** SAC is highly sensitive to reward scaling, because in this paper $\alpha$ is fixed at $1$ and the reward magnitude effectively acts as the inverse temperature. Too-small rewards give a near-uniform policy that can't exploit; too-large rewards collapse to near-determinism and miss exploration. The authors recommend tuning reward scale per task. *This is precisely the brittleness that Paper 2 fixes by learning $\alpha$ online.*

**Initial takeaway.** Paper 1 establishes that adding an entropy bonus to the RL objective is not just an exploration trick — it's a principled reformulation with its own convergence theory, and it produces a deep-RL algorithm that is simultaneously more sample-efficient than on-policy methods and dramatically more stable than off-policy deterministic methods. The catch: the entropy weight $\alpha$ is a fixed hyperparameter that effectively needs per-task tuning via reward scaling. That catch is the seed of Paper 2.

### Phase 2: Graduate-level deep dive — Paper 1

#### 2.1 The maximum-entropy RL objective

Standard RL maximizes
$$
J_{\text{std}}(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\!\left[ r(s_t, a_t) \right].
$$
The maximum-entropy generalization (Eq. 1 of the paper) augments this with the expected entropy of the policy at every visited state:
$$
J(\pi) = \sum_{t=0}^{T} \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\!\left[ r(s_t, a_t) + \alpha\, \mathcal{H}\!\big(\pi(\cdot \mid s_t)\big) \right],
\tag{1}
$$
where $\mathcal{H}(\pi(\cdot \mid s)) = -\mathbb{E}_{a \sim \pi}[\log \pi(a \mid s)]$ is the Shannon entropy of the policy at $s$, and $\alpha \geq 0$ is the **temperature**. The paper notes two equivalent views of $\alpha$:

- As a relative weight on the entropy term vs. reward.
- As a *scaling of the reward*: dividing rewards by $\alpha$ recovers the same fixed-point structure. Hence the paper sets $\alpha = 1$ in the algorithm and treats reward scale as the de facto temperature hyperparameter.

In the limit $\alpha \to 0$ the entropy term vanishes and standard RL is recovered. The infinite-horizon discounted formulation (Appendix A in the original) introduces $\gamma$ and is non-trivial because state-visitation marginals must also be discounted; the paper defers that derivation but uses the discounted objective in practice.

#### 2.2 The soft Bellman backup and soft policy iteration

**Soft state-value function** (Eq. 3):
$$
V^\pi(s_t) = \mathbb{E}_{a_t \sim \pi}\!\left[ Q^\pi(s_t, a_t) - \log \pi(a_t \mid s_t) \right],
$$
i.e. the standard expected-Q value minus the negative-log-policy term — equivalently, $V^\pi(s_t) = \mathbb{E}_a[Q(s,a)] + \mathcal{H}(\pi(\cdot \mid s_t))$ (with $\alpha = 1$ absorbed).

**Soft Bellman backup operator** (Eq. 2):
$$
\mathcal{T}^\pi Q(s_t, a_t) \;\triangleq\; r(s_t, a_t) + \gamma\, \mathbb{E}_{s_{t+1} \sim p}\!\left[ V^\pi(s_{t+1}) \right].
\tag{2}
$$

**Lemma 1 (Soft Policy Evaluation).** For any initial $Q_0: \mathcal{S} \times \mathcal{A} \to \mathbb{R}$ with $|\mathcal{A}| < \infty$, define $Q_{k+1} = \mathcal{T}^\pi Q_k$. Then $Q_k \to Q^\pi$ as $k \to \infty$.

**Derivation sketch.** Define an entropy-augmented reward $r_\pi(s_t, a_t) \triangleq r(s_t, a_t) + \gamma \,\mathbb{E}_{s_{t+1}}\!\big[\mathcal{H}(\pi(\cdot \mid s_{t+1}))\big]$. Substituting this into the soft Bellman backup gives
$$
\mathcal{T}^\pi Q(s_t, a_t) = r_\pi(s_t, a_t) + \gamma\, \mathbb{E}_{s_{t+1}, a_{t+1}}\!\left[ Q(s_{t+1}, a_{t+1}) \right],
$$
which is the *standard* Bellman backup with a modified reward. Hence $\mathcal{T}^\pi$ is a $\gamma$-contraction in the sup-norm on the (finite, bounded) space of Q-functions, and Banach fixed-point applies. The entropy boundedness condition $|\mathcal{A}| < \infty$ ensures $\mathcal{H}$ is bounded; in continuous-action settings, the differential entropy must be bounded by other means.

**Soft policy improvement.** Restrict the policy class to some $\Pi$ (e.g., parameterized Gaussians). For each state $s_t$, define the new policy as the $\text{KL}$-projection of the softmax-of-$Q$ onto $\Pi$:
$$
\pi_{\text{new}} = \arg\min_{\pi' \in \Pi} D_{\text{KL}}\!\left( \pi'(\cdot \mid s_t) \,\big\|\, \frac{\exp(Q^{\pi_{\text{old}}}(s_t, \cdot))}{Z^{\pi_{\text{old}}}(s_t)} \right).
\tag{4}
$$
The partition function $Z^{\pi_{\text{old}}}(s_t) = \int \exp(Q^{\pi_{\text{old}}}(s_t, a)) \, da$ is intractable but, crucially, is independent of $\pi'$ — so its gradient w.r.t. the policy parameters is zero and it can be dropped during optimization.

**Lemma 2 (Soft Policy Improvement).** With $\pi_{\text{new}}$ defined by Eq. 4, $Q^{\pi_{\text{new}}}(s,a) \geq Q^{\pi_{\text{old}}}(s,a)$ for all $(s,a)$.

**Derivation sketch.** The KL minimizer satisfies $\mathbb{E}_{a \sim \pi_{\text{new}}}[\log \pi_{\text{new}}(a \mid s_t) - Q^{\pi_{\text{old}}}(s_t, a)] \leq \mathbb{E}_{a \sim \pi_{\text{old}}}[\log \pi_{\text{old}}(a \mid s_t) - Q^{\pi_{\text{old}}}(s_t, a)]$ (the old policy is a valid candidate, so the optimum is at least as good). Rearranging gives $\mathbb{E}_{\pi_{\text{new}}}[Q^{\pi_{\text{old}}}(s_t, a) - \log \pi_{\text{new}}(a \mid s_t)] \geq V^{\pi_{\text{old}}}(s_t)$. Then by induction on the soft Bellman equation, $Q^{\pi_{\text{new}}} \geq Q^{\pi_{\text{old}}}$ pointwise.

**Theorem 1 (Soft Policy Iteration).** Alternating soft policy evaluation (Lemma 1) and soft policy improvement (Lemma 2) from any $\pi \in \Pi$ converges to $\pi^* \in \Pi$ with $Q^{\pi^*}(s,a) \geq Q^\pi(s,a)$ for all $\pi \in \Pi$ and all $(s,a)$.

The proof (Appendix B.3 in the original) follows from the monotone improvement of Lemma 2 plus the boundedness of the Q-function on the finite MDP, so the sequence $\{Q^{\pi_k}\}$ converges to its supremum.

#### 2.3 The deep SAC algorithm — three parameterized networks

In the deep / continuous-action regime, exact tabular iteration is replaced by stochastic gradient updates on three function approximators with parameters $\psi$ (value), $\theta$ (Q-function), $\phi$ (policy):

**(i) Value function loss** (Eq. 5):
$$
J_V(\psi) = \mathbb{E}_{s_t \sim \mathcal{D}}\!\left[ \tfrac{1}{2} \big( V_\psi(s_t) - \mathbb{E}_{a_t \sim \pi_\phi}[ Q_\theta(s_t, a_t) - \log \pi_\phi(a_t \mid s_t) ] \big)^2 \right].
$$
Its stochastic gradient (Eq. 6) is
$$
\hat\nabla_\psi J_V(\psi) = \nabla_\psi V_\psi(s_t) \big( V_\psi(s_t) - Q_\theta(s_t, a_t) + \log \pi_\phi(a_t \mid s_t) \big),
$$
where $a_t$ is sampled from the *current policy* (not the replay buffer) so the regression target tracks the current $\pi$. The paper notes that a separate $V$-network is not strictly required — one could define $V$ directly as $\mathbb{E}_a[Q - \log \pi]$ — but a separate network with an EMA target stabilizes training.

**(ii) Q-function loss** (Eq. 7):
$$
J_Q(\theta) = \mathbb{E}_{(s_t, a_t) \sim \mathcal{D}}\!\left[ \tfrac{1}{2} \big( Q_\theta(s_t, a_t) - \hat Q(s_t, a_t) \big)^2 \right],
$$
with target (Eq. 8):
$$
\hat Q(s_t, a_t) = r(s_t, a_t) + \gamma\, \mathbb{E}_{s_{t+1} \sim p}\!\left[ V_{\bar\psi}(s_{t+1}) \right],
$$
where $V_{\bar\psi}$ is the **target value network**, an exponentially-moving-average of the online $V_\psi$ with smoothing coefficient $\tau$: $\bar\psi \leftarrow \tau \psi + (1-\tau) \bar\psi$. The gradient (Eq. 9) is the standard TD-error form.

**(iii) Policy loss** (Eq. 10):
$$
J_\pi(\phi) = \mathbb{E}_{s_t \sim \mathcal{D}}\!\left[ D_{\text{KL}}\!\left( \pi_\phi(\cdot \mid s_t) \,\big\|\, \frac{\exp(Q_\theta(s_t, \cdot))}{Z_\theta(s_t)} \right) \right].
$$
Expanding the KL and dropping the state-only $\log Z_\theta(s_t)$ term:
$$
J_\pi(\phi) = \mathbb{E}_{s_t \sim \mathcal{D}, a_t \sim \pi_\phi}\!\left[ \log \pi_\phi(a_t \mid s_t) - Q_\theta(s_t, a_t) \right].
$$

**Reparameterization trick.** Rather than use the likelihood-ratio (REINFORCE) estimator, the policy is reparameterized as $a_t = f_\phi(\epsilon_t; s_t)$ where $\epsilon_t \sim \mathcal{N}(0, I)$ is sampled from a fixed noise distribution. The objective (Eq. 12) becomes
$$
J_\pi(\phi) = \mathbb{E}_{s_t \sim \mathcal{D}, \epsilon_t \sim \mathcal{N}}\!\left[ \log \pi_\phi(f_\phi(\epsilon_t; s_t) \mid s_t) - Q_\theta(s_t, f_\phi(\epsilon_t; s_t)) \right],
$$
with gradient (Eq. 13):
$$
\hat\nabla_\phi J_\pi(\phi) = \nabla_\phi \log \pi_\phi(a_t \mid s_t) + \big( \nabla_{a_t} \log \pi_\phi(a_t \mid s_t) - \nabla_{a_t} Q_\theta(s_t, a_t) \big) \nabla_\phi f_\phi(\epsilon_t; s_t),
$$
evaluated at $a_t = f_\phi(\epsilon_t; s_t)$. This is a lower-variance estimator than REINFORCE because the gradient of $Q$ w.r.t. $a$ is exploited directly.

**Double Q-learning** (added to mitigate Q-value overestimation, after Fujimoto et al. 2018): two Q-networks $Q_{\theta_1}, Q_{\theta_2}$ are trained on the same target, and the minimum is used wherever a Q-value enters the value-network or policy gradient:
$$
\min_{i \in \{1, 2\}} Q_{\theta_i}(s_t, a_t).
$$

**Algorithm 1 — one outer iteration of SAC:**
1. Take one environment step: $a_t \sim \pi_\phi(\cdot \mid s_t)$, $s_{t+1} \sim p(\cdot \mid s_t, a_t)$. Push to replay $\mathcal{D}$.
2. For each gradient step: sample a minibatch from $\mathcal{D}$; do gradient steps on $\psi$, $\theta_1$, $\theta_2$, $\phi$; do EMA update $\bar\psi \leftarrow \tau \psi + (1-\tau) \bar\psi$.

#### 2.4 Empirical evaluation and ablations

**Benchmarks** (Fig. 1 in the paper): Hopper-v1, Walker2d-v1, HalfCheetah-v1, Ant-v1, Humanoid-v1 (17-dim, OpenAI Gym), Humanoid-rllab (21-dim).

**Baselines**: DDPG, PPO, SQL (soft Q-learning, Haarnoja 2017), TD3 (concurrent work), Trust-PCL (in appendix).

**Main result**: SAC matches baselines on easy tasks (Hopper, HalfCheetah) and substantially outperforms them on hard tasks. DDPG fails entirely on Ant-v1 and both Humanoids; SAC succeeds on all six.

**Ablations:**
- **Stochastic vs. deterministic policy (Fig. 2)**: a deterministic SAC variant (DDPG-like but with double Q + hard target updates + fixed exploration noise) shows much higher seed-to-seed variance than stochastic SAC on Humanoid-rllab. The stochastic policy stabilizes *training*, not just exploration.
- **Deterministic evaluation (Fig. 3a)**: at evaluation time, replacing sampling with the policy mean yields higher returns — the training entropy bonus encourages exploration that costs reward, so the policy mean is the better deployment choice.
- **Reward scale (Fig. 3b)**: SAC is highly sensitive to the reward multiplier on Ant-v1. Scales of $\{1, 3, 10, 30, 100\}$ are tested; $10$ is approximately optimal. Too-small rewards: near-uniform policy, no exploitation. Too-large: near-deterministic, no exploration. Direct interpretation: the reward scale plays the role of inverse temperature.
- **Target smoothing $\tau$ (Fig. 3c)**: large $\tau$ (fast-moving target) causes instabilities; small $\tau$ slows learning. The authors use $\tau = 0.005$ throughout.

#### 2.5 Connections to prior work

- **Soft Q-learning (Haarnoja et al. 2017)**: precursor to SAC, also max-entropy, but solves for the optimal Q-function and treats the actor as an approximate sampler — *not* a true actor-critic. The actor's convergence depends on sampler quality. SAC fixes this by giving the actor a direct policy-improvement guarantee.
- **DDPG (Lillicrap et al. 2015)**: deterministic actor-critic, off-policy, sample-efficient but brittle. SAC is the stochastic-actor analog with entropy regularization.
- **SVG(0) (Heess et al. 2015)**: closest cousin to SAC — reparameterized stochastic actor with off-policy critic — but optimizes the standard expected-return objective (no entropy term) and lacks a separate value network. SAC adds both.
- **PGQ / Trust-PCL / Bridging-value-and-policy (Nachum et al., O'Donoghue et al., Schulman et al. 2017a)**: noted the connection between Q-learning and policy gradient under entropy regularization in *on-policy* settings. SAC extends this to fully off-policy continuous control.

### Appendix: Section-by-section backbone (Paper 1)

**Abstract.** Proposes SAC, an off-policy actor-critic deep RL algorithm in the max-entropy framework, where the actor maximizes both expected reward and policy entropy. State-of-the-art performance on continuous-control benchmarks; very stable across random seeds.

**§1. Introduction.** Two motivating challenges of model-free deep RL: sample inefficiency (on-policy methods) and brittleness (off-policy DDPG). Max-entropy RL is presented as the unifying fix: improved exploration, robust to model and estimation errors, and recovers standard RL in the $\alpha \to 0$ limit. Contribution: a convergence proof for soft policy iteration plus a practical deep algorithm.

**§2. Related work.** Three threads: (i) actor-critic architectures with separate policy/value networks; (ii) off-policy training via replay; (iii) entropy maximization (vs. entropy regularization). Prior max-entropy methods are either on-policy (poor sample complexity, e.g. O'Donoghue 2016) or off-policy soft Q-learning variants that struggle with continuous actions due to approximate inference (Haarnoja 2017, Nachum 2017a, Schulman 2017a). DDPG and SVG(0) are the closest off-policy actor-critic cousins.

**§3. Preliminaries.**
- §3.1 Notation: MDP $(\mathcal{S}, \mathcal{A}, p, r)$ with continuous state and action spaces, bounded reward, infinite horizon.
- §3.2 Max-entropy RL objective: Eq. 1 above. Temperature $\alpha$ controls the entropy-vs-reward trade-off; sub-summed into the reward by scaling. Three benefits stated: wider exploration, multi-modal capture, improved learning speed.

**§4. From soft policy iteration to soft actor-critic.**
- §4.1 Derivation of soft policy iteration. Lemma 1 (Soft Policy Evaluation, contraction via entropy-augmented reward), KL-projection-based soft policy improvement (Eq. 4), Lemma 2 (Soft Policy Improvement, monotone), Theorem 1 (Soft Policy Iteration converges to optimum).
- §4.2 Soft actor-critic. Three networks ($V_\psi$, $Q_\theta$, $\pi_\phi$). Value loss (Eq. 5), Q loss (Eq. 7) with target net (Eq. 8), policy loss as KL (Eq. 10). Reparameterization trick (Eqs. 11–13). Double-Q correction. Algorithm 1 listing.

**§5. Experiments.**
- §5.1 Comparative evaluation. Six MuJoCo benchmarks; SAC matches on easy tasks, dominates on hard tasks; DDPG fails on Ant + both Humanoids.
- §5.2 Ablation study. (a) Stochastic policy stabilizes training; (b) deterministic-mean evaluation yields higher return; (c) reward scale sensitivity = inverse temperature; (d) target smoothing $\tau$ trade-off.

**§6. Conclusion.** SAC combines off-policy efficiency, stochastic-actor stability, and entropy maximization. Future directions: trust regions, more expressive policies, second-order information.

**References.** ~30 entries; key precedents are Haarnoja 2017 (soft Q-learning), Lillicrap 2015 (DDPG), Schulman 2015/2017 (TRPO/PPO), Fujimoto 2018 (TD3), Ziebart 2010 (max-entropy IRL).

---

## Paper 2 — Haarnoja et al. 2018b (arXiv): Soft Actor-Critic Algorithms and Applications

**PDF:** `docs/project/references/Temperature/sources/Haarnoja et al. 2018 - Soft Actor-Critic Algorithms and Applications.pdf`
**Venue:** arXiv:1812.05905 (v2 dated Jan 2019)
**Authors:** Tuomas Haarnoja, Aurick Zhou, Kristian Hartikainen, George Tucker, Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter Abbeel, Sergey Levine (UC Berkeley + Google Brain)

### Version notes / delta against Paper 1

Paper 2 is **not a duplicate**: it is an explicit follow-up that (i) restates the Paper 1 theory in condensed form, (ii) adds the **automatic temperature adjustment** as a new Section 5 (the single largest methodological contribution), and (iii) adds two **real-robot applications** (quadruped locomotion, dexterous valve manipulation) as a new Section 7.2–7.3. Specifically:

| Aspect | Paper 1 (ICML) | Paper 2 (arXiv) |
|---|---|---|
| Temperature $\alpha$ | Fixed hyperparameter; effectively tuned via reward scaling | **Learned** by gradient descent on a Lagrangian dual; reward-scale brittleness eliminated |
| Networks | $V_\psi$, $Q_\theta$ (×2), $\pi_\phi$ — three families | $Q_\theta$ (×2), $\pi_\phi$ — value net dropped (see footnote 1 of Paper 2: "we found it to be unnecessary") |
| Soft Bellman target | Uses target value net $V_{\bar\psi}$ (Eq. 8 of Paper 1) | Uses target Q-net directly: $\hat Q = r + \gamma(Q_{\bar\theta}(s', a') - \alpha \log \pi_\phi(a' \mid s'))$, $a' \sim \pi_\phi$ (Eq. 6 of Paper 2) |
| Policy loss | KL form (Eq. 10) | Same KL form with $\alpha$ multiplier explicit: Eq. 7 of Paper 2 |
| Theory of soft policy iteration | §4 + Appendices B.1–B.3 (proofs not in main text) | §4 carries over **with proofs included as Appendix B.1–B.3** of Paper 2 — same Lemmas 1, 2 and Theorem 1, same proof structure |
| Empirical scope | MuJoCo benchmarks only | MuJoCo benchmarks + Minitaur quadruped (real hardware, 2h training) + D'Claw dexterous valve rotation from raw 32×32 RGB (real hardware, 20h training) |
| Action squashing detail | Mentioned, derivation in supplementary | **Explicit derivation in Appendix C**: tanh-squashed Gaussian, with change-of-variables Eq. 25–26 |
| Hyperparameter defaults | Reward scale = 10 typical; $\tau = 0.005$ | Same $\tau = 0.005$, $\gamma = 0.99$, Adam lr $3 \times 10^{-4}$; entropy target $\bar{\mathcal{H}} = -\dim(\mathcal{A})$ replaces reward scaling as the user-facing knob |

**What carries over unchanged from Paper 1:**
- The maximum-entropy RL objective (Eq. 1 of both papers — identical).
- The soft Bellman operator definition (Eq. 2, identical, with the soft value $V(s) = \mathbb{E}_a[Q(s,a) - \alpha \log \pi(a \mid s)]$ in Eq. 3 now showing $\alpha$ explicitly rather than absorbed).
- Soft policy iteration theory (Lemmas 1, 2, Theorem 1 — same statements; Paper 2 just makes the proofs visible in Appendix B).
- The reparameterization-trick policy update (Eqs. 8–10 of Paper 2 ≡ Eqs. 11–13 of Paper 1).
- Double Q-learning correction.
- MuJoCo benchmark setup; Figure 1 compares the **same six tasks** but now with two SAC variants plotted (fixed-$\alpha$, learned-$\alpha$) plus DDPG/PPO/TD3.

**The delta — Section 5 (Automating Entropy Adjustment) — is the focus of this review.**

### Phase 1: Foundational overview (undergraduate-level) — Paper 2

**Why this paper exists.** Paper 1's main empirical caveat was that SAC is highly sensitive to the reward scale — and reward scale was the de facto temperature knob, with $\alpha$ fixed at $1$. In practice that meant tuning the reward multiplier per task, which is fragile and *also moves during training* (a better policy collects bigger rewards, so the effective temperature drifts). Paper 2 attacks this directly: instead of fixing $\alpha$ and tuning reward scale, **fix the reward and learn $\alpha$**.

**The reframing.** Maximum-entropy RL is rewritten as a **constrained optimization** — maximize expected return *subject to* the policy's expected entropy being at least some target $\bar{\mathcal{H}}$:
$$
\max_\pi \mathbb{E}\!\sum_t r(s_t, a_t) \quad \text{s.t.} \quad \mathbb{E}[-\log \pi(a_t \mid s_t)] \geq \bar{\mathcal{H}} \text{ for all } t.
$$
By Lagrangian duality, this becomes the same max-entropy RL problem but with $\alpha$ now a **Lagrange multiplier** that is itself optimized. The dual objective for $\alpha$ turns out to be very simple: minimize $\mathbb{E}[-\alpha \log \pi(a \mid s) - \alpha \bar{\mathcal{H}}]$. If the current policy's entropy is *above* target, the gradient pushes $\alpha$ down (relaxing the entropy bonus, allowing the policy to sharpen); if entropy is *below* target, the gradient pushes $\alpha$ up (forcing more exploration). This is exactly the negative-feedback control loop you would design by hand — but here it falls out of the math.

**The target entropy.** The paper sets the entropy target $\bar{\mathcal{H}} = -\dim(\mathcal{A})$ (e.g. $-6$ for HalfCheetah's 6-dim action space, $-17$ for the 17-dim Humanoid). This is a heuristic: for a $D$-dimensional Gaussian with unit per-axis variance, the differential entropy is roughly $D/2 \log(2\pi e)$, so $-\dim(\mathcal{A})$ is "moderately stochastic in every action dimension." The point is that this heuristic **does not need per-task tuning** — the same rule works across all six MuJoCo tasks.

**Why this matters for our project.** The Lagrangian view is the key formal move that lets $\alpha$ become a *trainable variable* rather than a hyperparameter. Once $\alpha$ is a trainable scalar, it is one short step further to making $\alpha$ a *function* — of the state, of a context, of a modulator. The dual gradient $\nabla_\alpha J(\alpha) = -\mathbb{E}[\log \pi(a \mid s) + \bar{\mathcal{H}}]$ is the analytical handle on which a modulator-conditioned temperature would be trained.

**Other changes worth noting:**
1. **No separate value network.** Paper 2 drops the value network entirely (footnote 1) and uses a target Q-network in the Bellman target. This is a small simplification that matches what the broader RL community converged to.
2. **Same six MuJoCo benchmarks.** Paper 2 confirms SAC dominates DDPG/TD3/PPO/SQL and shows the learned-$\alpha$ variant matches or exceeds the per-task-tuned fixed-$\alpha$ variant — *without per-task tuning*.
3. **Real-robot deployment.** A Minitaur quadruped learns underactuated walking from scratch in ~2 hours of real-world training (160k env steps), and generalizes zero-shot to slopes, obstacles, and stairs because the entropy-maximized policy is intrinsically robust. A D'Claw hand learns valve rotation from raw 32×32 RGB images in ~20 hours. These are presented as evidence that the entropy bonus produces policies that transfer.

**Initial takeaway.** Paper 2 makes SAC self-tuning. The mathematical content is small — a Lagrangian, a dual gradient — but the practical content is large: SAC moves from "great algorithm with a fragile hyperparameter" to "algorithm with no tunable hyperparameters that works across benchmarks and real robots." The Lagrangian re-derivation is precisely the scaffold our project needs to make $\alpha$ depend on an external modulator.

### Phase 2: Graduate-level deep dive — Paper 2

#### 2.1 Constrained-entropy reformulation

The starting point is the same Eq. 1 max-entropy objective as Paper 1, but now $\alpha$ is treated as a Lagrange multiplier rather than a hyperparameter. The constrained primal problem (Eq. 11 of Paper 2) is:
$$
\max_{\pi_{0:T}} \mathbb{E}_{\rho_\pi}\!\left[ \sum_{t=0}^T r(s_t, a_t) \right]
\quad \text{s.t.} \quad \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}[-\log \pi_t(a_t \mid s_t)] \geq \bar{\mathcal{H}} \;\; \forall t.
\tag{11}
$$

Two structural notes:
- **Per-timestep constraint.** The constraint is imposed at every $t$ rather than globally. In a fully-observed MDP the policy that maximizes expected return is deterministic (zero entropy), so the constraint is generically tight and there is no need for an upper bound on entropy.
- **Time-varying $\alpha_t$.** During the backward-DP derivation, each timestep gets its own dual variable $\alpha_t$. In the practical algorithm these are tied to a single stationary $\alpha$ (the policy is also taken to be stationary), but the per-timestep derivation is what makes the duality clean.

#### 2.2 Backward-DP derivation of the dual

Since the policy at time $t$ affects only future returns, the objective can be rewritten as an iterated maximization (Eq. 12):
$$
\max_{\pi_0}\!\left( \mathbb{E}[r(s_0, a_0)] + \max_{\pi_1}\!\left( \mathbb{E}[\cdots] + \max_{\pi_T} \mathbb{E}[r(s_T, a_T)] \right) \right),
$$
subject to the per-$t$ entropy constraint. Starting from the last timestep $T$, with constraint $\mathbb{E}_{(s_T, a_T)}[-\log \pi_T(a_T \mid s_T)] \geq \bar{\mathcal{H}}$, **strong duality** applies (the objective is linear in $\pi_T$ and the constraint $-\log \pi$ is convex in $\pi$), so (Eq. 13):
$$
\max_{\pi_T} \mathbb{E}[r(s_T, a_T)] = \min_{\alpha_T \geq 0}\, \max_{\pi_T}\, \mathbb{E}\!\left[r(s_T, a_T) - \alpha_T \log \pi_T(a_T \mid s_T)\right] - \alpha_T \bar{\mathcal{H}}.
\tag{13}
$$

For any fixed $\alpha_T$, the inner $\max_{\pi_T}$ is exactly the soft Bellman policy improvement at temperature $\alpha_T$: the optimum is the softmax policy $\pi_T^*(a \mid s; \alpha_T) \propto \exp(Q_T(s, a) / \alpha_T)$ (taking $Q_T(s_T, a_T) = \mathbb{E}[r(s_T, a_T)]$). Substituting this optimal inner policy gives the dual objective in $\alpha_T$ alone:
$$
\alpha_T^* = \arg\min_{\alpha_T \geq 0}\, \mathbb{E}_{s_T, a_T \sim \pi_T^*}\!\left[ -\alpha_T \log \pi_T^*(a_T \mid s_T; \alpha_T) - \alpha_T \bar{\mathcal{H}} \right].
\tag{14}
$$

**Inductive step.** Define the soft Q-function recursively (Eq. 15):
$$
Q_t^*(s_t, a_t; \pi_{t+1:T}^*, \alpha_{t+1:T}^*) = \mathbb{E}[r(s_t, a_t)] + \mathbb{E}_{\rho_\pi}\!\left[ Q_{t+1}^*(s_{t+1}, a_{t+1}) - \alpha_{t+1}^* \log \pi_{t+1}^*(a_{t+1} \mid s_{t+1}) \right],
$$
with terminal $Q_T^*(s_T, a_T) = \mathbb{E}[r(s_T, a_T)]$. Applying the dual at step $T-1$ (Eq. 16):
$$
\max_{\pi_{T-1}}\!\left( \mathbb{E}[r(s_{T-1}, a_{T-1})] + \max_{\pi_T} \mathbb{E}[r(s_T, a_T)] \right)
= \min_{\alpha_{T-1} \geq 0} \max_{\pi_{T-1}}\!\Big( \mathbb{E}[Q_{T-1}^*(s_{T-1}, a_{T-1})] - \mathbb{E}[\alpha_{T-1} \log \pi(a_{T-1} \mid s_{T-1})] - \alpha_{T-1} \bar{\mathcal{H}} + \alpha_T^* \bar{\mathcal{H}} \Big).
$$
Iterating backward yields, for each $t$, the optimal dual update (Eq. 17):
$$
\alpha_t^* = \arg\min_{\alpha_t}\, \mathbb{E}_{a_t \sim \pi_t^*}\!\left[ -\alpha_t \log \pi_t^*(a_t \mid s_t; \alpha_t) - \alpha_t \bar{\mathcal{H}} \right].
\tag{17}
$$

In words: at each timestep, the optimal $\alpha_t$ is the one whose gradient w.r.t. the dual is zero, i.e. $\mathbb{E}[-\log \pi_t^*(a_t \mid s_t) - \bar{\mathcal{H}}] = 0$, or **policy entropy exactly equals the target $\bar{\mathcal{H}}$**.

#### 2.3 Practical implementation — approximate dual gradient descent

The exact backward-DP recursion is intractable in the function-approximation regime. The paper invokes **approximate dual gradient descent** (Boyd & Vandenberghe 2004): alternate one gradient step on the primal (policy + Q-function) and one gradient step on the dual ($\alpha$). The dual objective is dropped to a stationary scalar (Eq. 18):
$$
J(\alpha) = \mathbb{E}_{a_t \sim \pi_t}\!\left[ -\alpha \log \pi_t(a_t \mid s_t) - \alpha \bar{\mathcal{H}} \right].
\tag{18}
$$

**Gradient.** Differentiating:
$$
\nabla_\alpha J(\alpha) = \mathbb{E}_{a_t \sim \pi_t}\!\left[ -\log \pi_t(a_t \mid s_t) - \bar{\mathcal{H}} \right] = \mathbb{E}[\mathcal{H}(\pi_t(\cdot \mid s_t))] - \bar{\mathcal{H}}.
$$
Hence:
- If current entropy $> \bar{\mathcal{H}}$: $\nabla_\alpha J > 0$, so $\alpha \leftarrow \alpha - \lambda \nabla_\alpha J$ *decreases*. Less entropy bonus → policy can become sharper.
- If current entropy $< \bar{\mathcal{H}}$: $\nabla_\alpha J < 0$, so $\alpha$ *increases*. More entropy bonus → policy is forced to be more random.

This is a stable negative-feedback loop on policy entropy.

**Numerical stability.** In implementations, $\alpha$ is parameterized as $\alpha = \exp(\log\alpha)$ with the gradient taken w.r.t. $\log\alpha$ to keep $\alpha > 0$ without an explicit projection — this is mentioned in the implementation footnote but not in the math.

**Convergence caveat.** Boyd & Vandenberghe's convergence result for truncated dual gradient descent requires convexity of the primal in the policy parameters; this is *not* satisfied for neural-network policies. The paper states this honestly and reports that the procedure works in practice nonetheless.

#### 2.4 Practical algorithm changes vs. Paper 1

Three concrete changes beyond auto-$\alpha$:

**(i) No separate value network.** The soft Q-function target (Eq. 5 of Paper 2) is
$$
J_Q(\theta) = \mathbb{E}_{(s_t, a_t) \sim \mathcal{D}}\!\left[ \tfrac{1}{2}\big( Q_\theta(s_t, a_t) - (r(s_t, a_t) + \gamma\, \mathbb{E}_{s_{t+1} \sim p}[V_{\bar\theta}(s_{t+1})]) \big)^2 \right],
$$
with the value function $V(s_{t+1}) = \mathbb{E}_{a_{t+1} \sim \pi_\phi}[Q_{\bar\theta}(s_{t+1}, a_{t+1}) - \alpha \log \pi_\phi(a_{t+1} \mid s_{t+1})]$ implicitly computed from a single sample of $a_{t+1} \sim \pi_\phi(\cdot \mid s_{t+1})$. The stochastic gradient (Eq. 6) is:
$$
\hat\nabla_\theta J_Q(\theta) = \nabla_\theta Q_\theta(s_t, a_t)\big( Q_\theta(s_t, a_t) - (r(s_t, a_t) + \gamma( Q_{\bar\theta}(s_{t+1}, a_{t+1}) - \alpha \log \pi_\phi(a_{t+1} \mid s_{t+1}) )) \big),
$$
where $\bar\theta$ is the EMA-target Q-net.

**(ii) Policy loss with explicit $\alpha$** (Eq. 7):
$$
J_\pi(\phi) = \mathbb{E}_{s_t \sim \mathcal{D}}\!\left[ \mathbb{E}_{a_t \sim \pi_\phi}\!\left[ \alpha \log \pi_\phi(a_t \mid s_t) - Q_\theta(s_t, a_t) \right] \right].
$$
With the reparameterization $a_t = f_\phi(\epsilon_t; s_t)$ (Eq. 8), the objective becomes Eq. 9, and its gradient (Eq. 10):
$$
\hat\nabla_\phi J_\pi(\phi) = \nabla_\phi \alpha \log \pi_\phi(a_t \mid s_t) + \big( \nabla_{a_t} \alpha \log \pi_\phi(a_t \mid s_t) - \nabla_{a_t} Q(s_t, a_t) \big) \nabla_\phi f_\phi(\epsilon_t; s_t).
$$

**(iii) tanh-squashed Gaussian policy** (Appendix C). The policy is a Gaussian $\mu(u \mid s)$ on $u \in \mathbb{R}^D$, with the action computed as $a = \tanh(u)$ componentwise so that $a \in (-1, 1)^D$. The change-of-variables formula gives
$$
\pi(a \mid s) = \mu(u \mid s) \left| \det\!\left( \tfrac{da}{du} \right) \right|^{-1},
$$
and since the Jacobian is diagonal with $\partial a_i / \partial u_i = 1 - \tanh^2(u_i)$:
$$
\log \pi(a \mid s) = \log \mu(u \mid s) - \sum_{i=1}^D \log\!\big(1 - \tanh^2(u_i)\big).
\tag{26}
$$
This $\log\pi$ is the one that enters all SAC losses and the $\alpha$-gradient. Numerically, $\log(1 - \tanh^2(u)) = 2(\log 2 - u - \mathrm{softplus}(-2u))$ is the standard stable form (not stated in the paper but used in reference implementations).

**Final algorithm** (Algorithm 1 of Paper 2): per outer iteration —
1. Environment step: $a_t \sim \pi_\phi$, $s_{t+1} \sim p$; push to replay.
2. Per gradient step:
   - Update $\theta_i \leftarrow \theta_i - \lambda_Q \hat\nabla_{\theta_i} J_Q(\theta_i)$, $i \in \{1, 2\}$ (double-Q).
   - Update $\phi \leftarrow \phi - \lambda_\pi \hat\nabla_\phi J_\pi(\phi)$.
   - **Update $\alpha \leftarrow \alpha - \lambda \hat\nabla_\alpha J(\alpha)$.**
   - EMA target: $\bar\theta_i \leftarrow \tau \theta_i + (1-\tau) \bar\theta_i$, $i \in \{1, 2\}$.

#### 2.5 Empirical findings

**Six MuJoCo tasks** (Hopper-v2, Walker2d-v2, HalfCheetah-v2, Ant-v2, Humanoid-v2, Humanoid-rllab). Figure 1 plots **two SAC variants** (learned-$\alpha$ in blue, fixed-$\alpha$ in orange — fixed per-task tuned) vs. DDPG/PPO/TD3. The learned-$\alpha$ variant matches or exceeds the hand-tuned fixed-$\alpha$ variant on every task, with no per-task hyperparameter adjustment. The entropy target is fixed at $\bar{\mathcal{H}} = -\dim(\mathcal{A})$ throughout.

**Minitaur quadruped** (§7.2). Real robot, 8 direct-drive actuators, learned end-to-end with no simulation/pretraining. Training time ~2 h (160k env steps, ~400 episodes of $\leq 500$ steps). Zero-shot generalization to slopes, wooden-block obstacles, and stairs — the paper attributes this generalization to the entropy maximization producing policies with margin. State is constructed from the last 6 observations + 5 actions to handle non-Markovian latency/contact dynamics. First demonstration of underactuated real-world quadruped RL with no sim/pretrain.

**D'Claw valve rotation** (§7.3). 9-DoF dexterous hand, 3 fingers, rotates a faucet-like valve from any initial orientation to a target. Learned from raw 32×32 RGB images (two 3×3 conv + maxpool + two 256-unit FC) in ~20 h (300k env steps). Without images (state from valve angle directly): ~3 h, 2× faster than the prior PPO result on the same task (Zhu et al. 2018).

**Hyperparameter table** (Appendix D, Table 1):
- Optimizer: Adam, lr $3 \times 10^{-4}$.
- Discount $\gamma = 0.99$.
- Replay buffer $10^6$ transitions; batch size 256.
- 2 hidden layers, 256 units each, ReLU.
- Entropy target $\bar{\mathcal{H}} = -\dim(\mathcal{A})$ (e.g. $-6$ for HalfCheetah-v1).
- Target smoothing $\tau = 0.005$; target update interval 1; 1 gradient step per env step.

#### 2.6 Connections to related work

- **MPO (Abdolmaleki et al. 2018)** is now cited as the closest cousin: MPO also uses a probabilistic view and imposes a KL-to-previous-policy constraint via a Lagrangian. The mathematical scaffolding is parallel; MPO constrains *policy change*, SAC constrains *policy entropy*.
- **TD3 (Fujimoto et al. 2018)** is now a co-leading baseline; SAC matches or exceeds it.
- The constrained-entropy formulation is presented as enabling **expressive multi-modal policies** (e.g. normalizing flows, Haarnoja et al. 2018a) where no closed-form entropy exists — because the dual update needs only $\log \pi(a \mid s)$ at sampled actions, not the full entropy integral. This is a forward-looking point.

### Appendix: Section-by-section backbone (Paper 2)

**Abstract.** Summarizes SAC from the ICML paper, then announces the two new contributions: a *constrained formulation that automatically tunes the temperature hyperparameter*, and *evaluation on real-world tasks* (quadrupedal locomotion, dexterous manipulation). State-of-the-art on benchmarks and on real robots.

**§1. Introduction.** Same motivation as Paper 1. Names the brittleness-to-$\alpha$ problem explicitly: "in maximum entropy RL the scaling factor has to be compensated by the choice of a suitable temperature, and a sub-optimal temperature can drastically degrade performance." The paper's fix: "an automatic gradient-based temperature tuning method that adjusts the expected entropy over the visited states to match a target value."

**§2. Related Work.** Adds MPO as a parallel Lagrangian-style method. Otherwise close to Paper 1.

**§3. Preliminaries.** Identical structure to Paper 1; Eq. 1 unchanged (max-entropy objective, with $\alpha$ now shown explicitly rather than absorbed).

**§4. From Soft Policy Iteration to Soft Actor-Critic.** Condensed restatement of Paper 1's §4. Soft Bellman backup (Eq. 2), soft state-value (Eq. 3 now with $\alpha$), Lemma 1 (Soft Policy Evaluation), KL-projection improvement (Eq. 4), Lemma 2 (Soft Policy Improvement), Theorem 1 (convergence). **Proofs appear in Appendix B** (whereas Paper 1 only references them). Practical update equations (Eqs. 5–10) without the separate value network.

**§5. Automating Entropy Adjustment for Maximum Entropy RL.** *This is the heart of the paper.* Constrained primal (Eq. 11), backward-DP rewrite (Eq. 12), strong-duality move (Eq. 13), terminal-step dual minimizer (Eq. 14), recursive soft-Q definition (Eq. 15), inductive step (Eq. 16), per-$t$ optimal dual (Eq. 17). Justifies stationary approximation.

**§6. Practical Algorithm.** Double-Q correction (carried over). Approximate dual gradient descent for $\alpha$ with objective Eq. 18: $J(\alpha) = \mathbb{E}[-\alpha \log \pi(a \mid s) - \alpha \bar{\mathcal{H}}]$. Algorithm 1 listing — adds the $\alpha \leftarrow \alpha - \lambda \nabla_\alpha J(\alpha)$ update to the per-gradient-step block.

**§7. Experiments.**
- §7.1 Simulated benchmarks. Same 6 MuJoCo tasks (now -v2). Two SAC variants plotted: fixed-$\alpha$ (per-task tuned) and learned-$\alpha$. Result: learned-$\alpha$ matches or exceeds fixed-$\alpha$ across all tasks, with no per-task tuning needed.
- §7.2 Quadrupedal locomotion in the real world. Minitaur, 8 actuators, 160k env steps ≈ 2 h real-world training. Zero-shot generalization to slopes / obstacles / stairs.
- §7.3 Dexterous hand manipulation. D'Claw valve rotation. Raw RGB observations: 20 h. State observations: 3 h (2× faster than PPO on the same task).

**§8. Conclusion.** SAC + auto-$\alpha$ = sample-efficient, stable, no per-task hyperparameter tuning, deployable on real robots.

**Appendix A. Infinite-horizon discounted objective.** Eq. 19. Explicit form of the discounted max-entropy objective: $J(\pi) = \sum_t \mathbb{E}_{(s_t, a_t) \sim \rho_\pi}\!\left[ \sum_{l \geq t} \gamma^{l-t} \mathbb{E}_{s_l \sim p, a_l \sim \pi}[r(s_l, a_l) + \alpha \mathcal{H}(\pi(\cdot \mid s_l)) \mid s_t, a_t] \right]$. Notes the well-known subtlety that policy-gradient methods do not actually optimize the discounted objective exactly (Thomas 2014).

**Appendix B. Proofs.** B.1 Lemma 1 (Soft Policy Evaluation) — define $r_\pi = r + \gamma \mathbb{E}_{s'}[\mathcal{H}(\pi(\cdot \mid s'))]$, get standard $\gamma$-contraction. B.2 Lemma 2 (Soft Policy Improvement) — the new policy minimizes the KL, so it is at least as good as the old policy under that KL, leading to $\mathbb{E}_{\pi_{\text{new}}}[Q^{\pi_{\text{old}}}(s,a) - \log\pi_{\text{new}}(a \mid s)] \geq V^{\pi_{\text{old}}}(s)$, then iterate the soft Bellman equation to get $Q^{\pi_{\text{new}}} \geq Q^{\pi_{\text{old}}}$. B.3 Theorem 1 — by Lemma 2, $\{Q^{\pi_i}\}$ is monotone increasing and bounded, so it converges; by construction any other policy in $\Pi$ has lower soft value at the limit, so the limit is optimal in $\Pi$.

**Appendix C. Enforcing action bounds.** tanh squashing of a Gaussian, change-of-variables to compute $\log\pi(a \mid s)$ (Eqs. 25–26). $\log\pi(a \mid s) = \log\mu(u \mid s) - \sum_i \log(1 - \tanh^2(u_i))$.

**Appendix D. Hyperparameters.** Table 1: Adam, lr $3\times10^{-4}$, $\gamma = 0.99$, replay $10^6$, batch 256, 2×256 ReLU, target $\bar{\mathcal{H}} = -\dim(\mathcal{A})$, $\tau = 0.005$, 1 gradient step per env step.

---

## Cross-paper synthesis and project implications

### What the two papers together establish

1. **Maximum-entropy RL is well-defined and convergent.** Soft policy iteration (Lemmas 1, 2, Theorem 1, identical in both papers) gives the formal guarantee that the algorithm targets a unique fixed-point policy that maximizes expected return plus $\alpha \mathcal{H}(\pi)$.

2. **The deep / continuous-action version (SAC) is a practical actor-critic with three ingredients:** off-policy replay buffer, reparameterized stochastic Gaussian-tanh actor, double-Q critic. The version of SAC most projects implement is Paper 2's (no separate value network, learned $\alpha$, tanh-squashed Gaussian).

3. **Temperature $\alpha$ is a Lagrange multiplier, not a free hyperparameter.** Paper 2 reframes max-entropy RL as a *constrained* problem (entropy ≥ $\bar{\mathcal{H}}$) and derives $\alpha$ as the dual variable. The gradient $\nabla_\alpha J(\alpha) = \mathbb{E}[\mathcal{H}(\pi)] - \bar{\mathcal{H}}$ is a self-correcting negative-feedback signal on policy entropy.

4. **The target entropy heuristic $\bar{\mathcal{H}} = -\dim(\mathcal{A})$ works robustly** across all benchmarks and even on real robots, eliminating the per-task reward-scale tuning that plagued Paper 1.

### Implications for the project's modulator-conditioned temperature

The project's working hypothesis is that an interoceptive "precision" signal modulates the temperature of a stochastic policy. The SAC line gives three levers:

- **Lever A: $\alpha$ as a learned scalar (Paper 2 baseline).** The cleanest precedent. Replace the user-set hyperparameter with a learned scalar tied to entropy target. This is the "do nothing extra" baseline that any modulator-conditioned variant must beat.

- **Lever B: $\alpha$ as a state-dependent function $\alpha(s)$.** A natural generalization — the dual update $\nabla_\alpha J(\alpha) = \mathbb{E}[\mathcal{H}(\pi(\cdot \mid s)) - \bar{\mathcal{H}}]$ remains a valid gradient if $\alpha \to \alpha_\phi(s)$ is a small network with parameters $\phi_\alpha$ trained on the per-state dual loss $\mathbb{E}_s[-\alpha_\phi(s) \log \pi(a \mid s) - \alpha_\phi(s) \bar{\mathcal{H}}(s)]$. The entropy target $\bar{\mathcal{H}}$ could also become state-dependent — this is the analytical handle the project needs.

- **Lever C: $\alpha$ as a function of an external modulator $m_t$.** Substituting $m_t$ for $s_t$ in Lever B gives the project's target architecture: $\alpha = \alpha_\phi(m_t)$. The dual loss and its gradient transfer directly; what changes is what entropy target is appropriate and how $m_t$ is generated. The Lagrangian interpretation says: the modulator is *setting an entropy budget* for the policy, and the temperature self-tunes to respect that budget.

Two cautions worth flagging:
- The Boyd-Vandenberghe convergence guarantee for dual gradient descent does not survive nonlinear $\alpha_\phi$. Empirical fall-back is the only validation.
- The "reward scale ≡ inverse temperature" identity from Paper 1 implies that if reward magnitude varies systematically with the modulator (e.g. nociceptive context), the modulator-conditioned $\alpha$ must compensate or training will be confounded.

### Connections to the rest of the Temperature corpus

The papers shard A reviews here are the *direct precedents*. The remaining Temperature corpus extends or critiques this scaffold:

- **Ahmed et al. 2018 (Understanding entropy in policy optimization)** — empirical study of why entropy helps; complements SAC's stability claims.
- **Asadi & Littman 2017 (Alternative softmax operator)** — the softmax-of-Q in soft policy improvement is one of several softmax-like operators; relevant if the project considers alternatives (e.g. mellowmax) for the policy improvement step.
- **Song et al. 2018 (Softmax Bellman operator)** — analyzes the soft Bellman operator's contraction properties; relevant to convergence-theory questions about state-dependent $\alpha(s)$.
- **Vieillard et al. 2020 (Leverage the average / KL regularization in RL)** — generalizes max-entropy RL to general KL-to-prior regularization. Provides a strict generalization of the Paper-2 Lagrangian when the constraint is on KL rather than entropy.
- **Vieillard et al. 2020 (Munchausen RL)** — adds a log-policy term to the reward, which is structurally close to the modulator term in the project's hypothesis. Worth diff-reviewing against Paper 2.
- **Zhu et al. 2023 (General Munchausen / Tsallis KL)** — generalizes the KL divergence to Tsallis divergences, which is the natural mathematical home of "entropy-with-a-temperature-and-a-shape-parameter."
- **Lin et al. 2020 (Cat-SAC)** — curiosity-aware entropy temperature: a direct precedent for *signal-driven* $\alpha$.
- **Wang & Ni 2020 (Meta-SAC)** — meta-gradient $\alpha$ adjustment: an alternative to Paper 2's dual-gradient mechanism.

The project's modulator-conditioned temperature is most cleanly understood as **(Paper 2's Lagrangian)** ∘ **(state-dependent generalization à la Vieillard 2020 / Cat-SAC)** ∘ **(an external modulator signal carrying the per-state entropy target)**. The SAC papers reviewed here pin down the first link of that chain.

---
