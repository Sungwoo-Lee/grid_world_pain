# Gamma Lit Review — Batch A: Universal Value Function Family

## What this batch is about (plain English, read this first)

A **value function** in reinforcement learning is a learned scalar that tells you "from this state, how good is the future?" In standard RL, the agent learns one such function tied to one fixed reward signal — one "goal" baked into the world. The two papers reviewed here ask a more ambitious question: **can a single neural network learn an entire family of value functions, parameterized by something the agent can read off at run time — a goal, or a task descriptor — so that the agent generalizes to goals it has never been trained on?**

The progression is:

- **Schaul et al. 2015 — Universal Value Function Approximators (UVFA).** A single function $V(s, g; \theta)$ that takes both a **state** $s$ and a **goal** $g$ and returns the value of being in $s$ when you are trying to reach $g$. The key trick is a **two-stream architecture** ($s \to \phi(s)$, $g \to \psi(g)$, combined by a simple function $h$) trained either end-to-end or in two stages via matrix factorization of a partially observed value table. The paper shows the network generalizes to unseen goals on grid-worlds and Ms. Pacman.
- **Borsa et al. 2018 — Universal Successor Features Approximators (USFA).** A successor of UVFA that combines two ideas — **universal value functions** (generalize over a parameter $z$ describing the *policy* or *task*) and **successor features** (decompose $Q^\pi(s,a) = \boldsymbol{\psi}^\pi(s,a)^\top \mathbf{w}$ so that the dynamics-dependent piece $\boldsymbol{\psi}^\pi$ and the reward-dependent piece $\mathbf{w}$ live in separate factors). The result is a single network $\boldsymbol{\psi}(s,a,z;\theta)$ that, combined with **generalized policy improvement (GPI)** over a set of sampled task descriptors, gives strong zero-shot transfer to unseen tasks that share dynamics.

**Why this matters for the project.** The user is building intuition for **generalized value-function estimation** — value functions parameterized by something other than a fixed reward, so a single network can carry a *whole family* of predictions. This underlies a multi-horizon / multi-$\gamma$ / multi-modulator design thread (gamma-nets, hyperbolic discounting, neuromodulator-conditioned value heads): each of those is structurally a UVFA whose "goal" axis is a discount or a modulator setting rather than a literal goal state, and the conditioning architectures (two-stream, FiLM, hypernet) cited elsewhere in the project's references map directly onto the UVFA/USFA two-factor decomposition. Reading these two papers together gives the canonical "value function as a function of a parameter" template that the project will adapt to "value function as a function of a discount $\gamma$ or modulator level $m$."

The Phase 1 sections below are an undergraduate-level walkthrough; Phase 2 carries the full math (matrix factorization, successor-feature decomposition, GPI bound). The section-by-section backbone for each paper is preserved as an appendix.

---

## Table of Contents

- [1. Schaul et al. 2015 — Universal Value Function Approximators (UVFA)](#1-schaul-et-al-2015--universal-value-function-approximators-uvfa)
  - [1.1 Phase 1 — Foundational Overview](#11-phase-1--foundational-overview)
  - [1.2 Phase 2 — Graduate-Level Deep Dive](#12-phase-2--graduate-level-deep-dive)
  - [1.3 Appendix — Section-by-Section Backbone](#13-appendix--section-by-section-backbone)
- [2. Borsa et al. 2018 — Universal Successor Features Approximators (USFA)](#2-borsa-et-al-2018--universal-successor-features-approximators-usfa)
  - [2.1 Phase 1 — Foundational Overview](#21-phase-1--foundational-overview)
  - [2.2 Phase 2 — Graduate-Level Deep Dive](#22-phase-2--graduate-level-deep-dive)
  - [2.3 Appendix — Section-by-Section Backbone](#23-appendix--section-by-section-backbone)
- [3. Cross-Paper Synthesis](#3-cross-paper-synthesis)
  - [3.1 The shared architectural pattern](#31-the-shared-architectural-pattern)
  - [3.2 The progression UVFA → USFA in three steps](#32-the-progression-uvfa--usfa-in-three-steps)
  - [3.3 Connection to the project's discount / modulator / FiLM threads](#33-connection-to-the-projects-discount--modulator--film-threads)
  - [3.4 Take-home messages for the project](#34-take-home-messages-for-the-project)

---

## 1. Schaul et al. 2015 — Universal Value Function Approximators (UVFA)

**PDF:** `docs/project/references/Gamma/sources/Schaul et al. 2015 - Universal Value Function Approximators.pdf`
**Venue:** ICML 2015
**Authors:** Tom Schaul, Dan Horgan, Karol Gregor, David Silver (Google DeepMind)

### 1.1 Phase 1 — Foundational Overview

**The problem.** A standard value function $V(s; \theta)$ caches "how good is state $s$?" — under one fixed reward signal. If the agent's *goal* changes (a new waypoint, a new pellet to chase, a new task), the value function must be retrained. The Horde architecture (Sutton et al. 2011) addressed this by training many independent **demons**, one per goal — but Horde does not share structure across demons, so the cost grows linearly with the number of goals and the agent cannot generalize to a goal it has never been trained on.

**The proposal.** Replace many separate value functions $V_g(s)$ with a single **universal value function** $V(s, g; \theta)$ that takes the goal as input. If goals live in a structured space (e.g. goals *are* states, $\mathcal{G} \subseteq \mathcal{S}$, with the goal "go to grid cell $g$"), then nearby goals should induce similar value functions — and a flexible neural network should be able to exploit that.

**The core architectural idea.** Instead of concatenating $s$ and $g$ and feeding them through one MLP (the "concatenated" baseline), use a **two-stream architecture**:

$$V(s, g; \theta) \;=\; h\bigl(\phi(s),\; \psi(g)\bigr)$$

where $\phi: \mathcal{S} \to \mathbb{R}^n$ embeds states, $\psi: \mathcal{G} \to \mathbb{R}^n$ embeds goals, and $h$ is a simple combiner (dot product or a distance-based function). This factorization mirrors the structure of the problem — values depend on the *interaction* of state and goal, not on either alone.

**The training trick.** Direct end-to-end regression of $V(s,g;\theta)$ on observed values $V^*_g(s)$ is slow. Schaul et al. introduce a **two-stage training procedure**:

1. **Stage 1** — Lay out the observed values in a matrix $M$ (rows = states, columns = goals), and run **low-rank matrix factorization** to get target embeddings $\hat{\boldsymbol{\phi}}_s$ and $\hat{\boldsymbol{\psi}}_g$ such that $M \approx \hat{\boldsymbol{\Phi}}^\top \hat{\boldsymbol{\Psi}}$.
2. **Stage 2** — Train the networks $\phi$ and $\psi$ by ordinary supervised regression to hit those target embeddings.

This is an order-of-magnitude faster than end-to-end training on the LavaWorld benchmark.

**Key results.**

- **Tabular completion.** Even with 50–80% of the $(s,g)$ entries missing, matrix completion recovers near-optimal policies; policy quality saturates at low rank well before MSE does, suggesting low-rank structure is the right inductive bias.
- **Interpolation.** Trained on half the goals, the UVFA gives near-optimal values on the held-out half (LavaWorld, pixel observations).
- **Extrapolation.** Trained only on goals in rooms 1–3 of a 4-room maze, the UVFA still induces a useful policy for goals in room 4 — 24% ± 8% policy quality (vs. 0% for random) — by exploiting the **partially symmetric** architecture (sharing the first layers of $\phi$ and $\psi$, since states and goals share representation).
- **Ms. Pacman.** A UVFA trained on 29 pellet-eating "training goals" approximately recovers the value functions for 120 unseen pellet goals — a demonstration that the approach scales beyond toy grids to deep-RL-scale visual input.
- **Reinforcement learning (no ground-truth values).** Two algorithms shown to work: (a) Horde-seeded UVFA — train a small Horde of demons, build the value matrix from their estimates, factorize, and learn $\phi, \psi$; (b) Direct bootstrapping — Q-learning style updates directly on the UVFA, with a distance-based $h$ to keep things stable.

**Initial takeaway.** UVFA is the first clean demonstration that a *single neural network* can represent an entire family of value functions indexed by goals, and generalize to unseen goals. This works because (i) goals have structure that the network can learn, (ii) the two-stream factorization aligns architecture with that structure, and (iii) matrix factorization is a remarkably cheap way to bootstrap good embeddings before regression takes over.

**Connection to UVFA → USFA progression.** UVFA factors *the value function itself* as $V(s,g) = h(\phi(s), \psi(g))$. USFA goes further: it factors $Q^\pi(s,a)$ as $\boldsymbol{\psi}^\pi(s,a)^\top \mathbf{w}$ where $\boldsymbol{\psi}^\pi$ depends on dynamics and policy and $\mathbf{w}$ depends on reward. UVFA's generalization is over **goals** $g$; USFA's generalization is over **task / policy descriptors** $z$ that parameterize $\mathbf{w}$. Both rely on the same insight — exploit shared structure across a parametric family rather than retraining one network per task.

**Connection to the project's discount / modulator threads.** Replace "goal $g$" with "discount $\gamma$" or "modulator level $m$" and the architecture stays exactly the same: $V(s, \gamma; \theta) = h(\phi(s), \psi(\gamma))$ is the canonical gamma-net / hyperbolic-discounting / modulator-conditioned setup. The two-stream architecture is structurally identical to FiLM and hypernet conditioning (the project's other reference threads), with $\psi$ playing the role of the conditioning vector that modulates $\phi$.

### 1.2 Phase 2 — Graduate-Level Deep Dive

#### 1.2.1 Setup and notation

Consider a Markov Decision Process $(\mathcal{S}, \mathcal{A}, T, \gamma)$ with transition kernel $T(s,a,s') = \mathbb{P}(s_{t+1} = s' \mid s_t = s, a_t = a)$. For each goal $g \in \mathcal{G}$, define a **pseudo-reward** $R_g(s,a,s')$ and a **pseudo-discount** $\gamma_g(s) \in [0,1]$. The pseudo-discount plays a double role: state-dependent discounting *and* soft termination — $\gamma_g(s) = 0$ iff $s$ is a terminal state for goal $g$ (e.g. the waypoint has been reached).

The general value function under policy $\pi$ is

$$V_{g,\pi}(s) \;:=\; \mathbb{E}\!\left[ \sum_{t=0}^{\infty} R_g(s_{t+1}, a_t, s_t) \prod_{k=0}^{t} \gamma_g(s_k) \;\middle|\; s_0 = s \right]$$

with the action-value

$$Q_{g,\pi}(s,a) \;:=\; \mathbb{E}_{s'}\!\left[ R_g(s,a,s') + \gamma_g(s') \cdot V_{g,\pi}(s') \right]$$

and optimal versions $V^*_g$, $Q^*_g$ achieved by $\pi^*_g(s) := \arg\max_a Q^*_g(s,a)$.

A **UVFA** is a parametric approximator

$$V(s, g; \theta) \;\approx\; V^*_g(s), \qquad Q(s, a, g; \theta) \;\approx\; Q^*_g(s, a)$$

with $\theta \in \mathbb{R}^d$.

#### 1.2.2 Three architectures

Schaul et al. compare three architectures (Figure 1 of the paper):

1. **Concatenated**: $V(s,g;\theta) = f([s; g]; \theta)$ where $[s;g]$ is the concatenation and $f$ is an MLP. No factorization is exploited.
2. **Two-stream**: $V(s,g;\theta) = h(\phi(s; \theta_\phi),\; \psi(g; \theta_\psi))$ where $\phi, \psi: \to \mathbb{R}^n$ are MLPs and $h: \mathbb{R}^n \times \mathbb{R}^n \to \mathbb{R}$ is simple (dot product, weighted distance).
3. **Symmetric / partially symmetric two-stream**: when $\mathcal{G} \subseteq \mathcal{S}$, the parameters of $\phi$ and $\psi$ may be tied (partially: share first layers; fully: $\phi = \psi$). The fully symmetric case requires $h$ symmetric (e.g. dot product) and assumes $V^*_g(s) = V^*_s(g)$.

The **dot-product two-stream** is the workhorse in the paper: $V(s,g;\theta) = \phi(s)^\top \psi(g)$. Note this is exactly a rank-$n$ matrix factorization of $V^*$ when $\phi, \psi$ are linear, and a *non-linear* low-rank model otherwise.

#### 1.2.3 Two-stage training via matrix factorization

The two-stage procedure exploits the fact that, given the dot-product structure, the regression problem decomposes.

**Stage 1 — Matrix factorization.** Lay out observed values in a sparse data matrix $M \in \mathbb{R}^{|\mathcal{S}_{\text{obs}}| \times |\mathcal{G}_{\text{obs}}|}$ with $M_{s,g} \approx V^*_g(s)$. Find a rank-$n$ factorization

$$M \;\approx\; \hat{\boldsymbol{\Phi}}^\top \hat{\boldsymbol{\Psi}}, \qquad \hat{\boldsymbol{\Phi}} \in \mathbb{R}^{n \times |\mathcal{S}|},\; \hat{\boldsymbol{\Psi}} \in \mathbb{R}^{n \times |\mathcal{G}|}$$

so that for each observed state $s$ and goal $g$ we have target embeddings $\hat{\boldsymbol{\phi}}_s, \hat{\boldsymbol{\psi}}_g \in \mathbb{R}^n$ satisfying $M_{s,g} \approx \hat{\boldsymbol{\phi}}_s^\top \hat{\boldsymbol{\psi}}_g$.

When $M$ is dense and $h$ is the dot product, this is solved by **singular value decomposition**: let $M = U\Sigma V^\top$; then $\hat{\boldsymbol{\Phi}}^\top = U_n \Sigma_n^{1/2}$ and $\hat{\boldsymbol{\Psi}} = \Sigma_n^{1/2} V_n^\top$ using the top $n$ singular values/vectors.

When $M$ is **sparse** (as in tabular completion experiments), the paper uses **OptSpace** (Keshavan et al. 2009), which solves matrix completion by gradient descent on the manifold of rank-$n$ matrices. The OptSpace objective is

$$\min_{\hat{\boldsymbol{\Phi}}, \hat{\boldsymbol{\Psi}}} \;\sum_{(s,g) \in \Omega} \bigl(M_{s,g} - \hat{\boldsymbol{\phi}}_s^\top \hat{\boldsymbol{\psi}}_g\bigr)^2$$

over the observed index set $\Omega \subset \mathcal{S}_{\text{obs}} \times \mathcal{G}_{\text{obs}}$.

**Stage 2 — Multivariate regression of the embedding networks.** Train $\phi$ and $\psi$ as two independent supervised-regression problems against the targets from Stage 1:

$$\mathcal{L}_\phi(\theta_\phi) \;=\; \sum_{s \in \mathcal{S}_{\text{obs}}} \bigl\| \phi(s; \theta_\phi) - \hat{\boldsymbol{\phi}}_s \bigr\|_2^2, \qquad \mathcal{L}_\psi(\theta_\psi) \;=\; \sum_{g \in \mathcal{G}_{\text{obs}}} \bigl\| \psi(g; \theta_\psi) - \hat{\boldsymbol{\psi}}_g \bigr\|_2^2$$

solved by SGD. After Stage 2, the UVFA is $V(s,g;\theta) = \phi(s;\theta_\phi)^\top \psi(g;\theta_\psi)$. An optional Stage 3 fine-tunes both networks jointly end-to-end.

**Why two-stage beats end-to-end.** End-to-end backprop must (a) push gradients through both branches, (b) handle the non-convex coupled regression $\min_\theta \sum (V^*_g(s) - h(\phi(s), \psi(g)))^2$, which has multiple symmetric optima (the $n$-dim embedding can be rotated freely if $h$ is rotation-invariant). Stage 1 isolates the "what should the embeddings be?" sub-problem and solves it with a well-conditioned matrix-completion algorithm; Stage 2 reduces to standard supervised regression of a network onto fixed targets — much better-conditioned than joint training.

The paper reports an **order-of-magnitude speedup** on LavaWorld 7×7 (Figure 3 of the paper). For low ranks $n$, policy quality saturates well before MSE does, indicating low-rank structure suffices for *near-optimal control* even when the values themselves are not perfectly recovered (Figure 5).

#### 1.2.4 Output combiner $h$

The paper considers two main choices:

- **Dot product**: $h(\mathbf{a}, \mathbf{b}) = \mathbf{a}^\top \mathbf{b}$. Compatible with SVD, matrix factorization, and partially symmetric architectures. Naturally captures bilinear value structure.
- **Distance-based**: $h(\mathbf{a}, \mathbf{b}) = \gamma^{\|\mathbf{a} - \mathbf{b}\|_2}$. Used in the **direct-bootstrapping RL experiment** (Section 5.3) because it bounds outputs to $[0, 1]$ when pseudo-rewards are bounded the same way. The exponentiated negative distance ensures small distances in embedding space correspond to high values — a metric-learning flavor.

The distance-based choice trades off generality (it bakes in the assumption that values are bounded probabilities of reaching $g$ within a discount horizon) for **stability** of TD-learning with function approximation over goals. Stability of bootstrapping over a multi-task family is a recurring issue, and this paper picks a closed-form fix rather than a learning-rate hack.

#### 1.2.5 RL Algorithm 1 — UVFA from Horde targets

The pseudocode (Algorithm 1 of the paper) makes the workflow explicit:

```
1. Collect transition history H by interacting with environment (budget b_1).
2. For each transition in H and each training goal g in G_T, update demon Q_g
   off-policy via Q-learning (budget b_2). This produces a Horde.
3. Build matrix M_{t,g} = Q_g(s_t, a_t) for each (transition t, goal g) cell.
4. Factorize M ≈ Φ̂^T Ψ̂ at rank n (Stage 1).
5. Train φ, ψ by regression toward Φ̂, Ψ̂ (Stage 2, budget b_3).
6. Return Q(s, a, g) = h(φ(s, a), ψ(g)).
```

The factorization here is over transition-indices $\times$ goals rather than states $\times$ goals — this lets the algorithm respect off-policy distributions and lets $\phi(s,a)$ be an action-value embedding.

**Why this works.** The Horde demons each solve a tractable single-goal RL problem. The matrix $M$ then encodes the consistent solutions; factorization extracts the shared low-rank structure; regression compresses many demons into one universal network. Generalization to new goals comes for free as long as $\psi$ can produce a reasonable embedding for unseen $g$ — which it does because $\psi$ is trained on the *features* of $g$ (e.g. pixels of a goal-state image), not on $g$'s identity.

#### 1.2.6 RL Algorithm 2 — Direct bootstrapping

Equation (1) of the paper is a goal-conditioned Q-learning update applied at a randomly sampled transition $(s_t, a_t, s_{t+1})$ and a randomly sampled goal $g$:

$$Q(s_t, a_t, g) \;\leftarrow\; \alpha \!\left( r_g + \gamma_g \max_{a'} Q(s_{t+1}, a', g) \right) + (1 - \alpha) Q(s_t, a_t, g)$$

with $r_g = R_g(s_t, a_t, s_{t+1})$ and the bootstrapped target evaluated through the UVFA itself. This is the multi-goal off-policy Q-learning generalization. The challenge is that **bootstrapping + function approximation + multi-task** is exactly the configuration where TD divergence is most likely. Schaul et al. report that smaller learning rates help; the distance-based $h(\mathbf{a},\mathbf{b}) = \gamma^{\|\mathbf{a}-\mathbf{b}\|_2}$ helps more.

This direct-bootstrapping algorithm is the conceptual ancestor of later goal-conditioned RL methods (e.g. HER, Andrychowicz et al. 2017): the UVFA $Q(s,a,g)$ trained on a stream of $(s,a,r,s')$ transitions, where $g$ is sampled (possibly off-policy from any achieved state).

#### 1.2.7 Generalization analysis

The empirical analysis in Sections 4.1–4.3 produces three findings worth recording:

1. **Low rank is sufficient for policies, even if MSE is not yet low.** Figure 5 shows policy quality saturates at rank $\sim 5$ on 4-rooms, while MSE keeps improving with rank. The policy is robust to value perturbations as long as the *argmax* over actions is preserved — which low-rank structure does.
2. **Sparsity is tolerable.** Figure 6 shows that with 50% of the $(s,g)$ entries missing, matrix completion still recovers ~80% policy quality at rank $n = 7$; degradation is graceful up to 80% sparsity.
3. **Extrapolation is possible via partial symmetry.** Figure 8 shows that with $\phi$ and $\psi$ sharing first layers, training on goals in rooms 1–3 transfers to room 4 because the *state* representation of cells in room 4 has been seen during training and is shared with the *goal* representation.

The third finding is the most consequential: it shows that the bottleneck for generalization is not the goal-side network's parameters per se, but whether the goal representation $\psi(g)$ for an unseen $g$ falls in a region of embedding space where $\phi(s)$ has already been observed.

#### 1.2.8 Practical use-cases (Section 7)

The discussion enumerates four downstream uses of a trained UVFA:

1. **Transfer learning**: initialize $V_g(s)$ for a new task $g$ from $V(s, g; \theta)$. Figure 12 shows ~100× sample-efficiency improvement on LavaWorld over from-scratch training.
2. **Predictive state representation**: use $\phi(s)$ as a feature vector representing state, where the features are predictions of value across many goals (Sutton & Tanner 2005; Schaul & Ring 2013).
3. **Universal option / temporal abstraction**: define an option for each goal $g$ that acts greedily w.r.t. $V(s,g;\theta)$ and terminates when $g$ is reached. The UVFA gives a continuous family of options indexed by $g$.
4. **Universal option model**: when pseudo-rewards encode goal-achievement, $V(s,g;\theta)$ approximates the discounted probability of reaching $g$ from $s$ — i.e. a learned reachability map.

#### 1.2.9 Strengths and limitations

**Strengths.**
- Clean conceptual unification of Horde, multi-task value learning, and goal-conditioned policies.
- The two-stage matrix-factorization trick is a genuinely cheap algorithmic innovation that decouples representation learning from goal generalization.
- Generalization works in *both* supervised and RL settings, including with deep visual inputs (Ms. Pacman).

**Limitations / open questions.**
- The paper assumes $\mathcal{G} \subseteq \mathcal{S}$ for the headline experiments — goals are states. More abstract goal spaces (vectors of pseudo-reward parameters, language descriptions) are only mentioned in passing.
- The direct-bootstrapping algorithm is fragile and required a domain-specific $h$ to stabilize.
- No formal bound on how generalization quality depends on the geometry of the training goal set — the paper is empirical throughout.
- The matrix-factorization stage is only well-defined when $h$ is bilinear; for arbitrary $h$, only end-to-end training applies.

These limitations are exactly what motivates the move from UVFA to **successor features** (the basis of USFA): rather than factorizing the value table $M$ post-hoc, successor features factor the *Bellman equation itself* into a dynamics-features piece and a reward-weights piece, giving a structured (rather than learned) basis for transfer.

### 1.3 Appendix — Section-by-Section Backbone

This appendix preserves the paper's argument flow in original section order.

#### Abstract
Introduces UVFA $V(s, g; \theta)$ that generalizes over both states *and* goals. Proposes a two-stage supervised training via value-matrix factorization into state embeddings $\phi(s)$ and goal embeddings $\psi(g)$. Extends to RL via off-policy bootstrapping. Demonstrates generalization to unseen goals.

#### 1. Introduction
Value functions are central to RL but typically tied to one fixed reward. **General value functions** $V_g(s)$ (Sutton et al. 2011) extend this to pseudo-rewards and pseudo-discounts; the **Horde architecture** trains many such demons in parallel off-policy. Limitation: no parameter sharing across demons. Proposal: a single $V(s, g; \theta)$ exploiting the structure in goal space exactly as standard $V(s; \theta)$ exploits state structure. UVFA is "an infinite Horde of demons" in one network. Learning challenge: agent sees only a small subset of $(s, g)$ pairs; introduce a matrix-factorization-based two-stage training procedure.

#### 2. Background
MDP $(\mathcal{S}, \mathcal{A}, T)$. For each $g \in \mathcal{G}$, pseudo-reward $R_g(s,a,s')$ and pseudo-discount $\gamma_g(s)$ (which doubles as soft termination: $\gamma_g(s) = 0 \iff s$ terminal for $g$). General value $V_{g,\pi}(s) = \mathbb{E}[\sum_t R_g(s_{t+1},a_t,s_t) \prod_k \gamma_g(s_k) \mid s_0=s]$ and action-value $Q_{g,\pi}(s,a) = \mathbb{E}_{s'}[R_g + \gamma_g(s') V_{g,\pi}(s')]$. Optimal $V^*_g, Q^*_g$ via $\pi^*_g = \arg\max_a Q^*_g$.

#### 3. Universal Value Function Approximators
Three architectures (Figure 1): (i) **concatenated** $F: \mathcal{S} \times \mathcal{G} \to \mathbb{R}$; (ii) **two-stream** with $\phi: \mathcal{S} \to \mathbb{R}^n$, $\psi: \mathcal{G} \to \mathbb{R}^n$, combiner $h$; (iii) **symmetric / partially symmetric** when $\mathcal{G} \subseteq \mathcal{S}$ and/or $V^*_g(s) = V^*_s(g)$. Two-stream exploits common state/goal structure (share first layers between $\phi$ and $\psi$).

##### 3.1 Supervised Learning of UVFAs
**End-to-end**: MSE loss $\mathbb{E}[(V^*_g(s) - V(s,g;\theta))^2]$, SGD. **Two-stage**: (Stage 1) lay out values in matrix $M_{s,g} = V^*_g(s)$, factor $M \approx \hat{\boldsymbol{\Phi}}^\top \hat{\boldsymbol{\Psi}}$ via OptSpace (sparse) or SVD (dense bilinear $h$); (Stage 2) regress $\phi(s) \to \hat{\boldsymbol{\phi}}_s$ and $\psi(g) \to \hat{\boldsymbol{\psi}}_g$ independently. Optional Stage 3 = end-to-end fine-tuning. Reports order-of-magnitude speedup over end-to-end on 2-room 7×7 LavaWorld (Figure 3).

#### 4. Supervised Learning Experiments
Goals = states ($\mathcal{G} \subseteq \mathcal{S}$); pseudo-rewards $R_g(s,a,s') = 1$ iff $s' = g$ and not yet terminal, $0$ otherwise; pseudo-discount $\gamma_g(s) = 0$ if $s = g$ else $\gamma_{\text{ext}}$. Two domains: 4-rooms grid-world and LavaWorld (lava blocks, multi-room with teleporting doors, room-local pixel observation).

##### 4.1 Tabular Completion
1-hot states and goals; $\phi, \psi$ identity. OptSpace recovers value matrix from sparse observations. **Figure 5**: policy quality saturates at low rank ($n \approx 5$) while MSE keeps improving with rank. **Figure 6**: graceful degradation up to 80% sparsity at rank $n = 7$.

##### 4.2 Interpolation
Pixel observations on 7×7 LavaWorld (1 room). Training set = random half of goals; test set = held-out half. Three-layer MLP for $\phi$ and $\psi$; rank $n = 7$; two-stage training. **Figure 7**: policy quality and MSE recover on training set, generalize to test set.

##### 4.3 Extrapolation
4-room maze, train on goals in rooms 1–3 only, test on goals in room 4. Partially symmetric architecture (share first layers). **Figure 8**: 24% ± 8% policy quality on test goals.

#### 5. Reinforcement Learning Experiments
Two algorithms: Horde-seeded UVFA, and direct bootstrapping.

##### 5.1 Generalizing from Horde
**Algorithm 1**: collect history (budget $b_1$); train Horde demons off-policy via Q-learning (budget $b_2$); build matrix $M_{t,g} = Q_g(s_t, a_t)$ over transitions $\times$ goals; factor at rank $n$; train embedding networks by regression (budget $b_3$); return $Q(s,a,g) = h(\phi(s,a), \psi(g))$. **Figure 9**: heatmaps of training-vs-test prediction error as functions of (data samples, learning updates).

##### 5.2 Ms Pacman
150-demon Horde, each demon = goal of eating one pellet (goal = $(x,y)$ coordinate). 29 demons used as training set, 121 held out. **Figure 10**: UVFA recovers test-pellet value functions from training-only demons, visually matching Horde's directly trained predictions.

##### 5.3 Direct Bootstrapping
Equation (1): Q-learning update $Q(s,a,g) \leftarrow \alpha(r_g + \gamma_g \max_{a'} Q(s',a',g)) + (1-\alpha)Q(s,a,g)$ on random transitions $\times$ random goals. Stabilized by distance-based $h(\mathbf{a},\mathbf{b}) = \gamma^{\|\mathbf{a}-\mathbf{b}\|_2}$. **Figure 11**: ~80% policy quality with ~25% of $(s,g)$ pairs observed.

#### 6. Related Work
Distinction: **tasks** can have different MDP dynamics; **goals** only change rewards. Most prior multi-task work is policy-search (parameterized skills, motor primitives, model-based RL). UVFA is model-free, value-based, off-policy. Closest prior: Foster & Dayan (2002) — mixture-of-Gaussians value functions; limited by mixture-of-constants representation. Also van Otterlo (2009) on relational generalization.

#### 7. Discussion
Four downstream uses: (i) **transfer learning** — initialize $V_g$ for new task from UVFA (Figure 12: ~100× sample efficiency); (ii) **predictive state representation** — $\phi(s)$ as features; (iii) **universal options** — one option per goal, all from one UVFA; (iv) **universal option model** — $V(s,g;\theta)$ as discounted reach probability.

---

## 2. Borsa et al. 2018 — Universal Successor Features Approximators (USFA)

**PDF:** `docs/project/references/Gamma/sources/Borsa et al. 2018 - Universal Successor Features Approximators.pdf`
**Venue:** ICLR 2019 (submitted), arXiv:1812.07626
**Authors:** Diana Borsa, André Barreto, John Quan, Daniel Mankowitz, Rémi Munos, Hado van Hasselt, David Silver, Tom Schaul (DeepMind)

### 2.1 Phase 1 — Foundational Overview

**The problem.** A reinforcement-learning agent that has learned to solve some tasks should be able to transfer that knowledge to *new* tasks defined by new reward functions — ideally zero-shot. Two prior approaches each capture one piece of the puzzle:

- **UVFAs** (Schaul et al. 2015, Paper 1 above) generalize over **tasks / goals** by making the value function take the task as an input: $\tilde{Q}(s, a, w)$. This works when the optimal-value function is smooth in task-space — small change in $w$ implies small change in $Q^*$. UVFAs exploit structure in *the value function*.
- **Successor Features + Generalized Policy Improvement (SF&GPI)** (Barreto et al. 2017) generalize over tasks by **factorizing the value function** itself: if the reward is linear in features $\phi$, i.e. $r_w(s,a,s') = \boldsymbol{\phi}(s,a,s')^\top \mathbf{w}$, then the action-value of any policy $\pi$ on any task $\mathbf{w}$ decomposes as $Q^\pi_\mathbf{w}(s,a) = \boldsymbol{\psi}^\pi(s,a)^\top \mathbf{w}$ where $\boldsymbol{\psi}^\pi$ is the *successor feature* (the discounted expected sum of feature outcomes under $\pi$). Given a set of pre-learned policies $\pi_1, \dots, \pi_n$, **GPI** says: act greedily w.r.t. $\max_i Q^{\pi_i}_\mathbf{w}$. This is guaranteed (Barreto's GPI theorem) to be at least as good as any single $\pi_i$ on the new task. SF&GPI exploits structure in *the RL problem itself*.

These two mechanisms are **complementary**. UVFAs interpolate smoothly across task-space but offer no guarantees and can degrade arbitrarily if the smoothness assumption fails. SF&GPI gives a hard lower-bound guarantee but is limited to the set of policies it has — it can't squeeze any more performance out of a small training set.

**The proposal.** USFA = "a UVFA, but for successor features." Instead of one successor feature per policy, train a single network $\tilde{\boldsymbol{\psi}}(s, a, \mathbf{z}; \theta)$ that takes a **policy descriptor** $\mathbf{z} \in \mathbb{R}^d$ as input and returns the successor features of the policy associated with $\mathbf{z}$. By the linear-reward assumption, this immediately gives

$$\tilde{Q}(s, a, \mathbf{w}, \mathbf{z}) \;=\; \tilde{\boldsymbol{\psi}}(s, a, \mathbf{z})^\top \mathbf{w}$$

i.e. a value function that is **decoupled** in two arguments: $\mathbf{z}$ (which policy to evaluate) and $\mathbf{w}$ (on which task). At act time, choose a set $\mathcal{C} \subset \mathbb{R}^d$ of policy descriptors and apply GPI:

$$\pi(s) \;\in\; \arg\max_a \max_{\mathbf{z} \in \mathcal{C}} \tilde{\boldsymbol{\psi}}(s, a, \mathbf{z})^\top \mathbf{w}'$$

The choice of $\mathcal{C}$ is the key knob:

- $\mathcal{C} = \{\mathbf{w}'\}$ — recovers UVFA-style generalization (evaluate "the policy for the test task" via parametric inference).
- $\mathcal{C} = \mathcal{M}$ (training tasks) — recovers SF&GPI.
- $\mathcal{C} = \mathcal{M} \cup \{\mathbf{w}'\}$ — combines both.
- $\mathcal{C} = $ samples from a distribution — explores a tunable spectrum between the two extremes.

**Key results.**

- **Trip MDP (toy).** A two-state navigation task with a coffee-vs-food trade-off. Training set $\mathcal{M} = \{[1,0], [0,1]\}$ (pure coffee, pure food); test set: 50 directions on the unit circle. UVFA and vanilla SF&GPI each fail in characteristic ways. USFA with $\mathcal{C} = \{\mathbf{w}'\}$ and $\mathcal{C} = $ 5 random samples each beats both precursors, approaching near-optimal performance.
- **DeepMind Lab 3D navigation (scale test).** 3D environment with 4 object types (TVs, balls, hats, balloons); reward task $\mathbf{w} \in \mathbb{R}^4$ specifies the weight of each object type. Training set = 4 canonical-basis tasks ($\mathbf{w}$ one-hot). Test set: arbitrary $\mathbf{w} \in [-1, 1]^4$ including tasks with negative rewards never seen during training. All USFA variants generalize zero-shot, beating UVFA baselines (both on-policy and off-policy). The relative ranking of $\mathcal{C}$ choices depends on the policy-sampling distribution: with tight $\sigma = 0.1$, $\mathcal{C} = \mathcal{M}$ wins (GPI flavor); with wide $\sigma = 0.5$, $\mathcal{C} = \{\mathbf{w}'\}$ also performs well (UVFA flavor wakes up because the network has seen a broad slice of policy-space during training).
- **Theoretical bound (Proposition 1).** Sub-optimality of the GPI policy is bounded by the sum of two error terms, $\delta_d(\mathbf{z}) = \|\boldsymbol{\phi}\|_\infty \|\mathbf{w}' - \mathbf{z}\|$ (task distance) and $\delta_\psi(\mathbf{z}) = \|\mathbf{w}'\| \cdot \|\boldsymbol{\psi}^{\pi_\mathbf{z}} - \tilde{\boldsymbol{\psi}}(s,a,\mathbf{z})\|_\infty$ (SF approximation error), minimized over $\mathcal{C}$ — making explicit the trade-off that USFA navigates.

**Initial takeaway.** USFA is the synthesis: UVFA's smooth-interpolation generalization layered on top of SF&GPI's structurally guaranteed combination of policies. The key conceptual move is **disentangling task and policy** — using a *policy descriptor* $\mathbf{z}$ (which may differ from the *reward weight* $\mathbf{w}$) so that the agent can evaluate any policy on any task, and choose the GPI candidate set at act time. This lets a single network span the spectrum from "pure UVFA" to "pure SF&GPI" by tuning $\mathcal{C}$.

**Connection to UVFA → USFA progression.** UVFA factors $V(s, g) = h(\phi(s), \psi(g))$ — the *value* is bilinear in two learned embeddings. USFA factors $Q^\pi(s, a, \mathbf{w}, \mathbf{z}) = \boldsymbol{\psi}^\pi(s, a, \mathbf{z})^\top \mathbf{w}$ — the *value* is bilinear in a *learned vector* $\boldsymbol{\psi}$ and a *given task weight* $\mathbf{w}$. UVFA's factorization is opportunistic (learned by matrix-completion); USFA's factorization is *structural* (follows from the linear-reward assumption). Both share the same architectural skeleton: a two-stream network that takes $(s, a)$ on one branch and a parameter ($g$ or $\mathbf{z}$) on the other, combined to produce a scalar via a simple operation (dot product). USFA adds two crucial ingredients on top: (i) the SF intermediate representation $\boldsymbol{\psi}$ is multi-dimensional (it carries the discounted-feature-expectation, which is decision-relevant beyond just the scalar value), and (ii) GPI gives a *guarantee* — UVFA had none.

**Connection to the project's discount / modulator threads.** USFA's machinery — a learned policy/task descriptor $\mathbf{z}$ that conditions a multi-dimensional intermediate representation, combined linearly with a task-specific weight $\mathbf{w}$ — is structurally identical to what a multi-$\gamma$ or multi-modulator network would need. If $\mathbf{z}$ indexes a discount factor $\gamma$ (gamma-net flavor) or a modulator level $m$ (5-HT / DA / NE flavor), the same network can produce a *family of value functions* (one per $\mathbf{z}$), and GPI gives a way to combine them at decision time — answering "which $\gamma$ / which modulator setting should the agent commit to *for this state*?" by argmax-ing over the candidate set. The conditioning architecture is, again, structurally identical to FiLM / hypernet conditioning that the project's other reference threads cover.

### 2.2 Phase 2 — Graduate-Level Deep Dive

#### 2.2.1 The multi-task setup and the linear-reward assumption

A multi-task RL setup is an environment $(\mathcal{S}, \mathcal{A}, p, \gamma)$ shared across tasks, with task-specific reward $R_\mathbf{w}(s, a, s')$. The crucial structural assumption is **linear reward in feature vector**:

$$\mathbb{E}[R_\mathbf{w}(s, a, s')] \;=\; r_\mathbf{w}(s, a, s') \;=\; \boldsymbol{\phi}(s, a, s')^\top \mathbf{w} \tag{1}$$

where $\boldsymbol{\phi}(s, a, s') \in \mathbb{R}^d$ are features (treated as observable in this paper; can be learned) and $\mathbf{w} \in \mathbb{R}^d$ are task-specific reward weights. In the DM-Lab experiments, $\boldsymbol{\phi}$ are indicator functions for "agent collects object of type $i$"; $\mathbf{w}_i$ is the per-object-type reward.

This assumption is the linchpin of the whole framework — when it holds, the value function decomposes into a part depending on $\pi$ (and the environment) and a part depending only on $\mathbf{w}$:

$$Q^\pi_\mathbf{w}(s, a) \;=\; \mathbb{E}^\pi\!\left[\sum_{t=0}^\infty \gamma^t r_\mathbf{w}(s_t, a_t, s_{t+1}) \,\Big|\, s_0=s, a_0=a\right] \;=\; \underbrace{\mathbb{E}^\pi\!\left[\sum_{t=0}^\infty \gamma^t \boldsymbol{\phi}(s_t, a_t, s_{t+1})\right]}_{\boldsymbol{\psi}^\pi(s, a)}^{\!\!\!\top} \;\mathbf{w}$$

i.e.

$$\boxed{\;Q^\pi_\mathbf{w}(s, a) \;=\; \boldsymbol{\psi}^\pi(s, a)^\top \mathbf{w}\;} \tag{*}$$

This is the **successor-feature decomposition** at the heart of Barreto et al. 2017 and inherited by USFA.

**Derivation of (*) in one step.** Linearity of expectation, combined with $r_\mathbf{w} = \boldsymbol{\phi}^\top \mathbf{w}$, lets us pull $\mathbf{w}$ outside the discounted sum and the expectation:

$$Q^\pi_\mathbf{w}(s,a) = \mathbb{E}^\pi\!\left[\sum_t \gamma^t \boldsymbol{\phi}_t^\top \mathbf{w}\right] = \left(\mathbb{E}^\pi\!\left[\sum_t \gamma^t \boldsymbol{\phi}_t\right]\right)^{\!\top} \mathbf{w} = \boldsymbol{\psi}^\pi(s,a)^\top \mathbf{w}$$

where $\boldsymbol{\phi}_t := \boldsymbol{\phi}(s_t, a_t, s_{t+1})$. The two-line proof matters because it shows that the dynamics-dependent term $\boldsymbol{\psi}^\pi$ does *not* depend on $\mathbf{w}$ at all — once learned for some training reward, it can be **dot-producted with any new $\mathbf{w}$ at zero further cost** to evaluate $\pi$ on the new task.

#### 2.2.2 Successor features satisfy their own Bellman equation

Successor features obey a Bellman recursion in which $\boldsymbol{\phi}$ plays the role of reward:

$$\boldsymbol{\psi}^\pi(s, a) \;=\; \mathbb{E}\!\left[\boldsymbol{\phi}(s, a, s') + \gamma\, \boldsymbol{\psi}^\pi(s', \pi(s'))\right] \tag{2}$$

so any standard RL algorithm (TD, Q-learning, SARSA) can learn $\boldsymbol{\psi}^\pi$ by treating each of its $d$ components as a separate scalar value function with reward $\phi_i$. This is what makes SFs practically learnable.

#### 2.2.3 Generalized policy improvement (Barreto 2017)

**GPI theorem.** Suppose we have policies $\pi_1, \dots, \pi_n$ and corresponding action-value functions $Q^{\pi_1}, \dots, Q^{\pi_n}$ on some task. Define

$$Q^{\max}(s, a) \;:=\; \max_{i \in \{1, \dots, n\}} Q^{\pi_i}(s, a), \qquad \pi(s) \;\in\; \arg\max_a Q^{\max}(s, a)$$

Then $Q^\pi(s,a) \geq Q^{\max}(s,a)$ for all $(s,a)$. That is, acting greedily w.r.t. the pointwise max of a set of value functions yields a policy at least as good as the best of the input policies — at every state and action.

Combined with (*), GPI becomes computationally cheap: given SFs $\{\boldsymbol{\psi}^{\pi_1}, \dots, \boldsymbol{\psi}^{\pi_n}\}$ of policies trained on past tasks, the SFs can be evaluated on a new task $\mathbf{w}'$ via $Q^{\pi_i}_{\mathbf{w}'}(s,a) = \boldsymbol{\psi}^{\pi_i}(s,a)^\top \mathbf{w}'$, and GPI selects $\pi(s) \in \arg\max_a \max_i \boldsymbol{\psi}^{\pi_i}(s,a)^\top \mathbf{w}'$ — **zero-shot**, no further RL training required.

#### 2.2.4 Universal successor features

The conceptual move of USFA is to **disentangle task from policy** — generalize SFs over a parametric family of policies indexed by some descriptor $\mathbf{z}$. Define a **policy-encoding mapping** $e: (\mathcal{S} \to \mathcal{A}) \to \mathbb{R}^k$; then

$$\boldsymbol{\psi}(s, a, e(\pi)) \;\equiv\; \boldsymbol{\psi}^\pi(s, a)$$

is the **universal successor feature**. In practice the paper uses an elegant encoding: since any reward function induces a set of optimal policies *and* (by construction) any deterministic policy can be made optimal by some reward function, we can use reward-weight vectors $\mathbf{z} \in \mathbb{R}^d$ as policy descriptors — i.e. $e(\pi_\mathbf{z}) = \mathbf{z}$, where $\pi_\mathbf{z}$ is the optimal policy for the task with reward weights $\mathbf{z}$. So $\boldsymbol{\psi}(s, a, \mathbf{z})$ is the successor features of "the policy optimal for hypothetical task $\mathbf{z}$".

With this encoding, the universal Q-function becomes

$$Q(s, a, \mathbf{w}, \mathbf{z}) \;=\; \boldsymbol{\psi}(s, a, \mathbf{z})^\top \mathbf{w}$$

with **two semantically distinct inputs**: $\mathbf{z}$ tells you *which policy* you are evaluating, $\mathbf{w}$ tells you *on which task*. UVFAs are the special case $\mathbf{z} = \mathbf{w}$ (the policy you evaluate is the one you would want for the task you are facing).

#### 2.2.5 The GPI policy under USFA

Equation (3) of the paper defines the act-time policy of a trained USFA:

$$\pi(s) \;\in\; \arg\max_a \max_{\mathbf{z} \in \mathcal{C}} \tilde{Q}(s, a, \mathbf{w}', \mathbf{z}) \;=\; \arg\max_a \max_{\mathbf{z} \in \mathcal{C}} \tilde{\boldsymbol{\psi}}(s, a, \mathbf{z})^\top \mathbf{w}' \tag{3}$$

where $\mathcal{C} \subset \mathbb{R}^d$ is a **candidate set of policy descriptors** chosen at act time. As noted in Phase 1, three canonical choices recover known methods:

- $\mathcal{C} = \{\mathbf{w}'\}$ → query the network at the "right" policy descriptor for the test task → pure UVFA-style generalization through the smoothness of $\tilde{\boldsymbol{\psi}}$.
- $\mathcal{C} = \mathcal{M}$ → use only training-task policies → pure SF&GPI.
- $\mathcal{C} = \mathcal{M} \cup \{\mathbf{w}'\}$ or $\mathcal{C} = $ samples from a distribution → a continuous spectrum.

This $\mathcal{C}$ is a *test-time* hyperparameter, freely chosen — the trained network supports any choice.

#### 2.2.6 Generalization bound (Proposition 1)

The paper states a generalization of Barreto's Theorem 2 (2017) tailored to USFAs:

$$\|Q^*_{\mathbf{w}'} - Q^\pi_{\mathbf{w}'}\|_\infty \;\leq\; \frac{2}{1 - \gamma} \min_{\mathbf{z} \in \mathcal{C}} \Bigl( \underbrace{\|\boldsymbol{\phi}\|_\infty \,\|\mathbf{w}' - \mathbf{z}\|}_{\delta_d(\mathbf{z})} \,+\, \underbrace{\|\mathbf{w}'\| \cdot \|\boldsymbol{\psi}^{\pi_\mathbf{z}} - \tilde{\boldsymbol{\psi}}(s, a, \mathbf{z})\|_\infty}_{\delta_\psi(\mathbf{z})} \Bigr) \tag{4}$$

**Interpretation.** For each candidate $\mathbf{z} \in \mathcal{C}$ the bound has two terms:

- $\delta_d(\mathbf{z})$ — **task-distance error**: how far is the policy descriptor $\mathbf{z}$ from the actual test task $\mathbf{w}'$? Even if SFs are perfect, choosing $\pi_\mathbf{z}$ with $\mathbf{z}$ far from $\mathbf{w}'$ gives a policy potentially suboptimal for $\mathbf{w}'$.
- $\delta_\psi(\mathbf{z})$ — **SF approximation error**: how well do we actually approximate $\boldsymbol{\psi}^{\pi_\mathbf{z}}$ for this descriptor? Approximation error matters more for descriptors that were rarely seen during training.

The bound is **tightest at the $\mathbf{z}$ that minimizes the sum** — exactly the design freedom USFA gives the agent. Note: if $\mathcal{C} = \mathcal{M}$, there is an irreducible $\min_{\mathbf{z} \in \mathcal{M}} \delta_d(\mathbf{z})$ even with perfect SFs. If $\mathcal{C} = \{\mathbf{w}'\}$, $\delta_d(\mathbf{w}') = 0$ but $\delta_\psi(\mathbf{w}')$ may be large.

The bound makes formal the qualitative observation that UVFA-style and SF&GPI-style generalization are complementary, and that USFA can spend "design budget" on whichever side is cheaper.

#### 2.2.7 Training a USFA — the TD error decomposes

A transition is $(s_t, a_t, \boldsymbol{\phi}_{t+1}, s_{t+1})$ — the feature vector $\boldsymbol{\phi}$ replaces the scalar reward $r$ in the experience tuple. From this transition we can compute the $n$-step TD error for *any* policy $\pi_\mathbf{z}$ on *any* task $\mathbf{w}$:

$$\delta^{t,n}_{\mathbf{w},\mathbf{z}} = \sum_{i=t}^{t+n-1} \gamma^{i-t} r_\mathbf{w}(s_i, a_i, s_{i+1}) + \gamma^n \tilde{Q}(s_{t+n}, \pi_\mathbf{z}(s_{t+n}), \mathbf{w}, \mathbf{z}) - \tilde{Q}(s_t, a_t, \mathbf{w}, \mathbf{z})$$

Plugging in (*) and pulling $\mathbf{w}$ outside:

$$\delta^{t,n}_{\mathbf{w},\mathbf{z}} = \underbrace{\!\!\left[\sum_{i=t}^{t+n-1} \gamma^{i-t} \boldsymbol{\phi}(s_i, a_i, s_{i+1}) + \gamma^n \tilde{\boldsymbol{\psi}}(s_{t+n}, a_{t+n}, \mathbf{z}) - \tilde{\boldsymbol{\psi}}(s_t, a_t, \mathbf{z})\right]\!\!}_{=: \boldsymbol{\delta}^{t,n}_\mathbf{z}}^{\,\top} \mathbf{w} \;=\; (\boldsymbol{\delta}^{t,n}_\mathbf{z})^\top \mathbf{w} \tag{5}$$

with $a_{t+n} = \arg\max_b \tilde{\boldsymbol{\psi}}(s_{t+n}, b, \mathbf{z})^\top \mathbf{z}$ (the action greedy w.r.t. $\pi_\mathbf{z}$'s own task, since $\pi_\mathbf{z}$ is the optimal policy of task $\mathbf{z}$).

**Consequences.**

- **The TD error decomposes**: $\delta^{t,n}_{\mathbf{w}, \mathbf{z}} = (\boldsymbol{\delta}^{t,n}_\mathbf{z})^\top \mathbf{w}$ where the vector-valued $\boldsymbol{\delta}^{t,n}_\mathbf{z}$ depends only on $\mathbf{z}$ (and the data). The task vector $\mathbf{w}$ only appears in the final dot product.
- **Training updates do not depend on $\mathbf{w}$**: the gradient update to $\theta$ is driven by $\boldsymbol{\delta}^{t,n}_\mathbf{z}$ (the SF-side TD error), not by the scalar $\delta^{t,n}_{\mathbf{w},\mathbf{z}}$. The training task influences *which transitions are sampled* (via the behavior policy), not the update rule directly.
- **Training is naturally off-policy**: you can update $\tilde{\boldsymbol{\psi}}(\cdot, \mathbf{z})$ using data collected by any behavior policy — even one induced by a different task — as long as you can correct for the policy mismatch (e.g. via Watkins's Q$(\lambda)$ trace-cutting, which the paper uses).

**Practical algorithm.** Algorithm 1 of the paper sketches the training loop with $\epsilon$-greedy Q-learning:

```
At each step:
  - Sample a training task w ~ Uniform(M)
  - Sample n_z policy descriptors {z_1, ..., z_{n_z}} ~ D_z(·|w)
    (D_z = N(w, σI) in the experiments; σ controls coverage)
  - Behavior: ε-greedy on argmax_a max_i ψ̃(s,a,z_i)^T w  (GPI over the samples)
  - Observe (φ, s')
  - For each z_i: TD-update ψ̃(s,a,z_i) using ψ̃(s', π_i(s'), z_i), where π_i(s') is greedy w.r.t. z_i
```

The choice of $\mathcal{D}_\mathbf{z}$ — concentrated near $\mathbf{w}$ (tight $\sigma$) vs. broadly covering policy-space (wide $\sigma$) — is the single most consequential decision. Tight $\sigma$ → most policies seen are close to training-task policies, off-policy-ness is mild, and $\mathcal{C} = \mathcal{M}$ at act time is the natural choice. Wide $\sigma$ → broad policy coverage, severe off-policy-ness during training, but $\mathcal{C} = \{\mathbf{w}'\}$ at act time becomes viable because the network has actually seen what to predict at $\mathbf{w}'$-like descriptors.

#### 2.2.8 Architecture

The USFA agent (Figure 1 and Figure 12 of the paper) has three modules:

1. **Input processing**: convolutional encoder → LSTM(256) → ReLU → state embedding $f(\mathbf{h}_t) \in \mathbb{R}^{128}$. Depends only on the observation history.
2. **Policy conditioning**: takes $\mathbf{z}$ through a 2-layer MLP(32, 32) → policy embedding $g(\mathbf{z}) \in \mathbb{R}^{32}$. Concatenates with $f(\mathbf{h}_t)$. Passes through a 2-layer MLP to output a tensor of shape $d \times |\mathcal{A}|$ — these are $\tilde{\boldsymbol{\psi}}(s, a, \mathbf{z})$ for every action.
3. **Task evaluation**: parameter-free dot product $\tilde{Q}(s, a, \mathbf{w}, \mathbf{z}) = \tilde{\boldsymbol{\psi}}(s, a, \mathbf{z})^\top \mathbf{w}$.

**Architectural design choice — when to inject the conditioning.** The paper injects $\mathbf{z}$ **late** (after the LSTM). This is important: late conditioning means the LSTM unroll is *policy-independent*, so the same hidden-state trajectory can serve many sampled $\mathbf{z}$'s in parallel — both training and acting become $O(1)$ in $n_z$ for the heavy part (input processing) and linear in $n_z$ only for the cheap part (the small MLP and the dot product). If $\mathbf{z}$ were injected early (before the LSTM), every sampled policy would need its own LSTM unroll — prohibitive at $n_z = 30$ (the experimental value). This is the same architectural lesson the project sees in FiLM / hypernet literature: **conditioning is cheapest when injected late, at the smallest representational scale**.

#### 2.2.9 Empirical results

**Toy experiment (Trip MDP).** Two-state MDP, $\phi(s_1, E) = (-\epsilon, -\epsilon)$, $\phi(s_2, P_i)$ on the unit circle. Training set $\mathcal{M} = \{(1,0), (0,1)\}$; test set: 50 directions. Both UVFA (slow smoothness, only 2 training points) and vanilla SF&GPI (only 2 policies = "all-C" and "all-F", which are suboptimal for mixed-preference tasks) fail. USFA with $\mathcal{C} = \{\mathbf{w}'\}$ and $\mathcal{C} = $ 5 random samples both perform near-optimally (Figure 2 of paper).

**Large-scale experiment (DM Lab).** 3D first-person navigation, 4 object types. Reward task $\mathbf{w} \in \mathbb{R}^4$ specifies per-object weight; training set = 4 canonical-basis vectors $\{e_1, e_2, e_3, e_4\}$. Test set: arbitrary $\mathbf{w}$ including **negative** weights (objects to avoid — never experienced during training). Agents trained with IMPALA + Q($\lambda = 0.9$) and asynchronous data collection.

Results (Figure 4):
- All architectures generalize *somewhat* to test tasks — even UVFA baselines.
- USFAs outperform unstructured UVFAs across the board, including on negative-reward tasks the agent never trained on.
- Among USFA variants, with $\sigma = 0.1$ (tight): $\mathcal{C} = \mathcal{M}$ and $\mathcal{C} = \mathcal{M} \cup \{\mathbf{w}'\}$ tend to be best.
- With $\sigma = 0.5$ (broad): $\mathcal{C} = \{\mathbf{w}'\}$ approaches the performance of $\mathcal{C} = \mathcal{M}$ (Figure 5).

The negative-reward generalization is striking: the agent learned only positive-reward policies but, via SF&GPI, can correctly *avoid* an object class at test time. This is because the SFs $\boldsymbol{\psi}^{\pi_\mathbf{z}}$ encode "the discounted count of objects of each type collected under $\pi_\mathbf{z}$"; if $\mathbf{w}'$ has $w'_i < 0$, the GPI argmax simply *deprefers* policies that score high on $\psi_i$. The behavior of avoiding object $i$ is **compositional** — never seen as a policy during training, assembled at act time by the GPI argmax.

#### 2.2.10 Counter-intuitive findings and the limits of GPI

The paper highlights one initially-counter-intuitive observation: **$\mathcal{C} = \mathcal{M} \cup \{\mathbf{w}'\}$ sometimes performs *worse* than $\mathcal{C} = \mathcal{M}$**, especially on tasks with negative rewards. This seems to contradict the intuition "more candidates = more guarantees" — but the bound (4) is a guarantee on the *worst-case sub-optimality*, not the *expected return*. The decomposition tells the full story:

- Adding $\mathbf{w}'$ to $\mathcal{C}$ shrinks $\delta_d$ (now zero).
- But adding $\mathbf{w}'$ exposes the agent to the SF approximation error $\delta_\psi(\mathbf{w}')$ at a point that may not have been well-trained.
- If $\delta_\psi(\mathbf{w}')$ is large and a poor candidate's $\boldsymbol{\psi}^\top \mathbf{w}'$ happens to exceed the better candidates' values *by error*, GPI's argmax will pick the wrong policy.

This is a real failure mode: **GPI is robust against unknown-task errors but vulnerable to large SF-approximation errors at the candidate set**. Practical implication for the project: when adapting USFA-like machinery to new conditioning axes ($\gamma$, modulator level), be very careful about *which* candidate set you use at test time — there's an implicit assumption that approximation is reliable across the full $\mathcal{C}$.

#### 2.2.11 Strengths and limitations

**Strengths.**
- Cleanly subsumes UVFA, SF&GPI, and a continuum of hybrids in one architecture.
- Decouples task and policy in a way that exposes a meaningful test-time hyperparameter ($\mathcal{C}$).
- Provides an honest generalization bound (Proposition 1) that names the relevant error terms.
- Demonstrates zero-shot generalization to *negative* rewards never seen during training — non-trivial compositional behavior.
- Architectural design choice (late conditioning, $O(1)$ unroll cost) makes the approach practical at deep-RL scale.

**Limitations / open questions.**
- The linear-reward assumption $r = \boldsymbol{\phi}^\top \mathbf{w}$ is restrictive. Most real reward functions are not linear in a small fixed feature vector.
- $\boldsymbol{\phi}$ is assumed observable in the paper (the agent gets to see one-hot indicators of which object was collected). Learning $\boldsymbol{\phi}$ end-to-end is mentioned as an extension but not demonstrated.
- The policy-encoding $e(\pi_\mathbf{z}) = \mathbf{z}$ only covers policies that are optimal for some linear-reward task. Stochastic policies, sub-optimal policies, or policies derived from other principles fall outside the framework as stated.
- The choice of $\mathcal{D}_\mathbf{z}$ at training time and $\mathcal{C}$ at act time are both crucial and unsupervised — no automatic mechanism for finding good ones.
- The GPI guarantee is in $L_\infty$ norm and scales as $1/(1-\gamma)$, which is loose at long horizons.

### 2.3 Appendix — Section-by-Section Backbone

#### Abstract
USFAs combine the scalability of UVFAs (parametric generalization across tasks), the instant inference of SFs (a Bellman-equation-respecting decomposition of $Q$), and the strong generalization of GPI (a worst-case-better policy from any set of pre-trained policies). Practical benefit demonstrated on a 3D first-person DeepMind Lab navigation task.

#### 1. Introduction
Multitask RL motivation; transfer to unseen tasks is the focus. Two complementary sources of structure: (i) similarity between optimal policies / value functions across tasks (exploited by UVFAs); (ii) shared dynamics (exploited by SF&GPI). UVFA and SF&GPI generalize in different ways and are complementary. **Main idea**: extend SFs to a universal form, $\tilde{\boldsymbol{\psi}}(s, a, \mathbf{z})$, exactly as UVFAs extend $V^*_g$ to $\tilde{V}(s, g)$. The result is USFA, which strictly generalizes both precursors (recovers UVFA at $\mathcal{C} = \{\mathbf{w}'\}$, recovers SF&GPI at $\mathcal{C} = \mathcal{M}$) and opens up a spectrum of hybrids.

#### 2. Background

##### 2.1 Multitask RL
MDP $(\mathcal{S}, \mathcal{A}, p, R, \gamma)$, multitask = a set of MDPs sharing $(\mathcal{S}, \mathcal{A}, p, \gamma)$ but differing in $R$. Linear-reward assumption (Eq. 1): $r_\mathbf{w}(s, a, s') = \boldsymbol{\phi}(s,a,s')^\top \mathbf{w}$, $\boldsymbol{\phi}$ observable. Notation: tilde indicates approximation, $\theta$ for tunable params.

##### 2.2 Transfer Learning
Goal: leverage knowledge from training tasks $\mathcal{M} \sim \mathcal{D}_\mathbf{w}$ to perform on test tasks $\mathcal{M}' \sim \mathcal{D}_\mathbf{w}$. UVFAs (Schaul 2015): $\tilde{Q}(s, a, \mathbf{w})$ extends value function to take task as input; relies on smoothness in $\mathbf{w}$-space. SFs (Eq. 2): $\boldsymbol{\psi}^\pi(s,a) = \mathbb{E}^\pi[\sum_{i \geq t} \gamma^{i-t} \boldsymbol{\phi}_{i+1} \mid s_t=s, a_t=a]$. Under (1), $Q^\pi_\mathbf{w} = (\boldsymbol{\psi}^\pi)^\top \mathbf{w}$. SFs satisfy a Bellman equation with $\boldsymbol{\phi}$ as reward → learnable by standard RL. GPI: given $\{\boldsymbol{\psi}^{\pi_i}\}$, GPI policy is $\pi(s) \in \arg\max_a \max_i Q^{\pi_i}(s,a)$; $Q^\pi \geq Q^{\max}$ everywhere.

#### 3. Universal Successor Features Approximators
Compares UVFA and SF&GPI: UVFA exploits structure in $Q^*(s,a,\mathbf{w})$ via function approximator; SF&GPI exploits structure in the RL problem itself, agnostic to representation. Trade-off: UVFA may fail if smoothness breaks; SF&GPI cannot exploit functional regularities even when present.

##### 3.1 Universal Successor Features
**Key disentanglement**: rewrite $Q^*(s, a, \mathbf{w})$ as $Q^{\pi_\mathbf{w}}(s, a, \mathbf{w})$ — separate task and policy. Define $Q(s, a, \mathbf{w}, \pi) = (\boldsymbol{\psi}^\pi)^\top \mathbf{w}$ generalizing over both. Policy-encoding mapping $e: (\mathcal{S} \to \mathcal{A}) \to \mathbb{R}^k$; using reward-weight vectors as encodings: $e(\pi_\mathbf{z}) = \mathbf{z}$. Then $Q(s, a, \mathbf{w}, \mathbf{z}) = \boldsymbol{\psi}(s, a, \mathbf{z})^\top \mathbf{w}$. UVFs are the special case $\mathbf{w} = \mathbf{z}$.

##### 3.2 USFA Generalization
GPI policy (Eq. 3): $\pi(s) \in \arg\max_a \max_{\mathbf{z} \in \mathcal{C}} \tilde{\boldsymbol{\psi}}(s, a, \mathbf{z})^\top \mathbf{w}'$. Choice of $\mathcal{C}$ recovers UVFA ($\{\mathbf{w}'\}$), SF&GPI ($\mathcal{M}$), or any hybrid. **Proposition 1** (Eq. 4): $\|Q^*_{\mathbf{w}'} - Q^\pi_{\mathbf{w}'}\|_\infty \leq \frac{2}{1-\gamma} \min_{\mathbf{z} \in \mathcal{C}} (\delta_d(\mathbf{z}) + \delta_\psi(\mathbf{z}))$ with $\delta_d(\mathbf{z}) = \|\boldsymbol{\phi}\|_\infty \|\mathbf{w}' - \mathbf{z}\|$, $\delta_\psi(\mathbf{z}) = \|\mathbf{w}'\| \cdot \|\boldsymbol{\psi}^{\pi_\mathbf{z}} - \tilde{\boldsymbol{\psi}}\|_\infty$.

##### 3.3 How to Train a USFA
Transition $(s_t, a_t, \boldsymbol{\phi}_{t+1}, s_{t+1})$ enables learning $\tilde{\boldsymbol{\psi}}(s, a, \mathbf{z})$ for any $\mathbf{z}$. TD error decomposition (Eq. 5): $\delta^{t,n}_{\mathbf{w}, \mathbf{z}} = (\boldsymbol{\delta}^{t,n}_\mathbf{z})^\top \mathbf{w}$ — task only enters via the dot product, updates to $\theta$ are $\mathbf{w}$-independent. **Algorithm 1**: $\epsilon$-greedy Q-learning loop sampling tasks from $\mathcal{M}$ and policy descriptors from $\mathcal{D}_\mathbf{z}(\cdot \mid \mathbf{w})$; behavior is GPI over the $n_z$ samples; update each $\tilde{\boldsymbol{\psi}}(\cdot, \mathbf{z}_i)$ off-policy.

#### 4. Experiments

##### 4.1 Trip MDP
Two-state navigation, coffee-vs-food trade-off. $\boldsymbol{\phi}(\cdot, C) = (1,0)$, $\boldsymbol{\phi}(\cdot, F) = (0,1)$, exploratory action $E$ with $\boldsymbol{\phi}(s_1, E) = (-\epsilon, -\epsilon)$; from $s_2$, agent can reach $N = 5$ intermediate places. Training $\mathcal{M} = \{(1,0), (0,1)\}$; test = 50 directions on unit circle. SF&GPI on training policies misses intermediate optima; UVFA struggles with only 2 training points. **USFA with both $\mathcal{C} = \{\mathbf{w}'\}$ and $\mathcal{C} = $ 5 random samples approaches optimal** (Figure 2).

##### 4.2 Large Scale (DeepMind Lab)
3D first-person environment with 4 object types. Reward task $\mathbf{w} \in \mathbb{R}^4$; features $\phi_i$ = indicator of collecting object $i$. Training set $\mathcal{M} = \{e_1, e_2, e_3, e_4\}$ canonical basis. Architecture (Figure 1, detailed in Appendix C): conv + LSTM input module → policy-conditioning MLP (late injection of $\mathbf{z}$) → SF tensor $d \times |\mathcal{A}|$ → dot-product with $\mathbf{w}$. Training: IMPALA-style asynchronous Q($\lambda = 0.9$); $\mathcal{D}_\mathbf{z} = \mathcal{N}(\mathbf{w}, \sigma^2 I)$ with $\sigma \in \{0.1, 0.5\}$; $n_z = 30$. Baselines: UVFA on-policy, UVFA off-policy.

##### 4.3 Results and Discussion
All architectures generalize to test tasks zero-shot. USFAs beat UVFAs across the board. **Negative-reward tasks** (never experienced during training) are solved zero-shot via GPI. With tight $\sigma = 0.1$: $\mathcal{C} = \mathcal{M}$ tends to win. With wide $\sigma = 0.5$: $\mathcal{C} = \{\mathbf{w}'\}$ catches up (Figure 5). **Counter-intuitive**: $\mathcal{C} = \mathcal{M} \cup \{\mathbf{w}'\}$ sometimes worse than $\mathcal{C} = \mathcal{M}$ — adding a test point exposes the policy to large SF approximation errors at that point.

#### 5. Related Work
UVFAs (Schaul 2015), hindsight experience replay (Andrychowicz 2017), goal-conditioned hierarchical RL (Vezhnevets 2017, Andreas 2016, Oh 2017), modular composition (Devin 2017, Heess 2016), meta-learning (Finn 2017), continual learning (Kirkpatrick 2016, Rusu 2016). Prior SF+NN work (Kulkarni 2016, Zhang 2016) lacked GPI. Concurrent Ma et al. 2018 also combines SFs with UVFAs but in a critic-only role and on-policy.

#### 6. Conclusion
USFA = UVFA on top of SFs + GPI. Two empirical regimes: (i) UVFA-favored scenarios where policy space has many optima but the value function is smooth in $\mathbf{w}$ (Trip MDP) — USFA wins through decoupled training; (ii) SF&GPI-favored scenarios where structure of the RL problem dominates (DM Lab) — USFA recovers GPI's strengths. The model exposes three structural sources of regularity: shared dynamics (via SFs), policy-space structure (UVFA-style), and RL-problem structure (via GPI).

#### Appendix A — Two Types of Generalisation: Intuition
Simple 1-state MDP with $\mathbf{w} \in \mathbb{R}$; optimal value space = piecewise linear. UVFA fits a parametric function in $\mathbf{w}$-space → extrapolation degrades with limited training points (Figure 7). SF&GPI evaluates each pre-trained policy on $\mathbf{w}'$ and trusts the argmax → can be perfect if any training policy is optimal at $\mathbf{w}'$ (Figure 8). USFA recovers both behaviors via choice of $\mathcal{C}$.

#### Appendix B — Trip MDP Details
$N = 6$ intermediate places, $\theta \in \{k\pi/2N\}$ for $k = 0, \dots, N$; $\epsilon = 0.05$. Worst-case test set = diagonal $\{(w'_1, w'_2) : w'_1 = w'_2\}$ (maximally far from training $\mathcal{M} = \{(1,0), (0,1)\}$); both UVFA and SF&GPI struggle; USFA still approaches optimal (Figure 11).

#### Appendix C — Large-Scale Architecture and Training Details
USFA agent has three modules: (i) Input processing — 3 conv layers + LSTM(256) + ReLU producing $f(\mathbf{h}_t) \in \mathbb{R}^{128}$. (ii) Policy conditioning — 2-layer MLP(32, 32) on $\mathbf{z}$ produces $g(\mathbf{z}) \in \mathbb{R}^{32}$; concatenated with $f(\mathbf{h}_t)$; 2-layer MLP produces tensor of shape $d \times |\mathcal{A}|$; computation parallelizes over $n_z$ samples because input processing is shared. (iii) Task evaluation — parameter-free $\tilde{Q} = \tilde{\boldsymbol{\psi}}^\top \mathbf{w}$. Training uses IMPALA + Q($\lambda = 0.9$); 50 actors per task; trajectory length 32; $\epsilon = 0.1$ for behavior, $\epsilon = 0.001$ for evaluation. UVFA baseline architecture is structurally similar but conditions on $\mathbf{w}$ instead of $\mathbf{z}$.

#### Appendix D — Additional Results
Full breakdown of test tasks by difficulty (close to / far from training $\mathcal{M}$), positive / negative / mixed rewards. USFA's robustness to negative rewards (object-avoidance tasks) is striking: the agent never saw a policy that avoids any object during training, but GPI reconstructs avoidance behavior compositionally via $\max$-over-policies on $\mathbf{w}'$ with negative weights.

---

## 3. Cross-Paper Synthesis

This section pulls out the through-lines connecting Schaul 2015 (UVFA) and Borsa 2018 (USFA), with explicit pointers to the project's broader threads.

### 3.1 The shared architectural pattern

Both papers instantiate the same skeleton:

$$\text{value} \;=\; \text{simple-combiner}\bigl(\text{stream}_A(\text{state-side input}),\; \text{stream}_B(\text{parameter-side input})\bigr)$$

Concretely:

| Paper | Stream A (state) | Stream B (parameter) | Combiner |
|---|---|---|---|
| UVFA | $\phi(s)$ | $\psi(g)$ — goal embedding | dot product (or distance-based $\gamma^{\|\cdot\|}$) |
| USFA | $\tilde{\boldsymbol{\psi}}(s, a, \mathbf{z})$ — $d$-dim per action | $\mathbf{w}$ — task weight | dot product (parameter-free) |

**Differences worth marking.**

1. **Where the parameter sits.** In UVFA, the parameter $g$ goes through a *learned* embedding $\psi(g)$ — the network learns how to encode goals. In USFA, the parameter $\mathbf{w}$ is *not* learned-embedded at all — it enters as a raw weight in the final dot product. The "learned conditioning" happens on the *policy descriptor* $\mathbf{z}$ side, which goes through its own MLP. USFA splits the "policy / task" role that UVFA had collapsed.

2. **What the intermediate stream represents.** UVFA's $\phi(s)$ has no semantic content — it is just whatever the network needs to make the dot product equal the value. USFA's $\boldsymbol{\psi}^\pi(s,a)$ has a precise semantic meaning — it is the discounted expected feature outcome under $\pi$, learnable by its own Bellman recursion. This is what gives USFA the **structural** decomposition (and the GPI guarantee) that UVFA lacks.

3. **Generalization mechanism.** UVFA generalizes by hoping that $\psi(g_{\text{new}})$ lands in a region of embedding space the network has learned about — soft, opportunistic, no guarantee. USFA generalizes by *constructing* the new task's Q-function from learned successor features and the new $\mathbf{w}'$ — and by argmax-ing over a candidate set with a worst-case guarantee. Soft + hard hybrid.

### 3.2 The progression UVFA → USFA in three steps

| Step | Add | What it buys |
|---|---|---|
| Start: standard $V(s)$ | — | One task only. |
| UVFA (2015) | Add a parameter $g$ as input; two-stream architecture | Generalization across goals; smooth interpolation; matrix-factorization trick. |
| Successor features (Barreto 2017) | Replace $V$ with $\boldsymbol{\psi}^\pi$; assume $r = \boldsymbol{\phi}^\top \mathbf{w}$; add GPI | Zero-shot transfer with worst-case guarantee — but only over a *fixed* set of pre-trained policies. |
| USFA (2018) | Make SFs universal: $\boldsymbol{\psi}(s, a, \mathbf{z})$ | Bring back UVFA's parametric generalization on top of the SF&GPI machinery; expose $\mathcal{C}$ as a test-time hyperparameter spanning the spectrum. |

The single unifying insight: **factorization of $Q$ along axes that the agent can control at test time** — first along (state, goal), then along (state, policy, task). Each axis gives a new lever for generalization without retraining.

### 3.3 Connection to the project's discount / modulator / FiLM threads

The project context names two design threads that map cleanly onto UVFA/USFA:

**Discount-conditioned value functions ($\gamma$-nets, hyperbolic discounting).** Replace UVFA's "goal $g$" with a discount factor $\gamma$. The architecture becomes $V(s, \gamma; \theta) = h(\phi(s), \psi(\gamma))$ — exactly UVFA's two-stream skeleton with $\gamma$ playing the goal role. The learned $\psi(\gamma)$ embedding lets the network share computation across discount horizons; the matrix-factorization trick would still apply if you stack value functions across many $\gamma$'s. Hyperbolic discounting can be implemented as an integral over $\gamma$'s with a hyperbolic prior, which is naturally supported by a single $\gamma$-conditioned network. (Sherstan et al. 2020 — "$\gamma$-nets" — and Fedus et al. 2019 — "Hyperbolic discounting" — both live in the same Gamma folder; their core machinery is UVFA-with-$\gamma$-as-goal.)

**Modulator-conditioned value heads (5-HT / DA / NE / ACh).** Replace UVFA's "goal $g$" with a modulator level $m \in \mathbb{R}^k$ (per-neurotransmitter). The architecture is identical: $V(s, m; \theta) = h(\phi(s), \psi(m))$. USFA's machinery applies if the agent has multiple modulator settings to choose from at run time: train $\tilde{\boldsymbol{\psi}}(s, a, m)$, sample candidate modulator settings $\mathcal{C}$ at test time, and let GPI argmax over them — i.e. the agent picks "which modulator state to commit to" pointwise per state. The linear-reward assumption in USFA maps to a *per-modulator-axis* additive reward decomposition, which is precisely how neuromodulator literature typically frames "this modulator scales this reward channel".

**FiLM and hypernet conditioning.** UVFA's two-stream concatenation is the weakest form of conditioning. FiLM (feature-wise affine modulation: scale and shift each channel of $\phi(s)$ as a function of $g$) and hypernets (generate the weights of $\phi$ from $g$) are strictly stronger; they fit naturally onto the same skeleton by replacing the simple combiner with a more expressive one. The USFA paper's architectural choice — *late, concatenation-based* conditioning — is a deliberate compromise: weaker per-token expressivity, but $O(1)$ in number of policy samples because the LSTM unroll is shared. This is the same trade-off that surfaces in the project's FiLM / hypernet design discussions.

### 3.4 Take-home messages for the project

1. **A single network can carry a whole family of value functions.** This is the fundamental enabler for any multi-$\gamma$ / multi-modulator / multi-goal architecture. UVFA proves the network capacity exists; USFA proves the structural decomposition can be made principled.

2. **Generalization mechanism = structural assumption + architectural alignment.** UVFA needs smoothness in goal-space + two-stream architecture. USFA needs linear-reward + SF Bellman + GPI argmax. The lesson for the project's $\gamma$ / modulator threads: identify *the* structural assumption (e.g. "value is approximately a polynomial in $\gamma$", or "modulators affect reward additively"), and let it dictate the architecture.

3. **Conditioning location matters more than conditioning strength.** USFA's late-injection design — share the heavy backbone, condition cheaply on the parameter — is the right pattern for any architecture that wants to evaluate many candidate values of the conditioning parameter in parallel at act time. For a multi-$\gamma$ network that needs to sample many $\gamma$'s at decision time, this design choice is *more* consequential than the choice of FiLM vs. dot-product vs. hypernet.

4. **Generalization guarantees come from $\arg\max$-of-$\max$, not from interpolation.** UVFA-style interpolation is fragile (Proposition 1's $\delta_\psi$ term can blow up); SF&GPI's argmax-over-policies is robust. If the project ever wants worst-case guarantees on multi-$\gamma$ or multi-modulator generalization, the GPI pattern (a candidate set + argmax) is the most likely route — at the cost of needing a successor-feature-like decomposition.

5. **Beware compositional negative settings.** USFA solves "avoid object type $i$" zero-shot from positive-reward training data alone, via GPI on a sign-flipped $\mathbf{w}'$. For a modulator network, the analog is "behave under modulator setting $m^{\text{novel}}$ never seen during training" — and the same compositional mechanism may or may not work, depending on how the SF-side approximation generalizes. The DM-Lab result is the existence-proof; the failure-mode discussion in §2.2.10 is the warning label.

---

## Notes on processing

- Processed sequentially, paper-by-paper, per literature-reviewer protocol.
- Source PDFs at `docs/project/references/Gamma/sources/` (under the project's standard `<topic>/sources/` convention).
- Extracted text snapshots at `tmp/schaul2015_uvfa.txt` and `tmp/borsa2018_usfa.txt`.
- Each paper has: (i) Phase 1 plain-English synthesis, (ii) Phase 2 graduate-level deep dive with full LaTeX equations and derivations, (iii) section-by-section backbone appendix preserving the paper's original argument flow.
- Cross-paper synthesis section explicitly links to the project's $\gamma$ / modulator / FiLM threads as required by the framing brief.

