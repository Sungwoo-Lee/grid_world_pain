# TD Lit Review — Batch B: Successor Features and Transfer

## What this batch is about (plain English, read this first)

A standard value function $Q^\pi(s, a)$ in reinforcement learning answers one question — *"under policy $\pi$, starting from state $s$ and action $a$, how much reward do I expect to collect (discounted) in the future?"* — and it bakes two things into a single scalar: **(i) what will happen** (the dynamics of the world under $\pi$) and **(ii) how much I care about each thing that happens** (the reward function). Change the reward signal — say, the agent stops liking coffee and starts liking food — and the value function has to be relearned from scratch, even though *the world itself has not changed*.

**Successor features (SFs)** are a clean way to break that scalar apart. Instead of caching one number per state-action, you cache a *vector* $\boldsymbol{\psi}^\pi(s, a)$ whose $i$-th entry is "the discounted future amount of feature $i$ that I expect to encounter under $\pi$." Intuitively: $\boldsymbol{\psi}^\pi(s, a)$ is a **future-visitation profile** — a $\gamma$-weighted forecast of *what kinds of things will happen* if the agent commits to $\pi$ from $(s, a)$. Once you have that vector, getting a Q-value for *any* reward function that is linear in those features — $r(s, a, s') = \boldsymbol{\phi}(s, a, s')^\top \mathbf{w}$ — is just a dot product:

$$Q^\pi(s, a) \;=\; \boldsymbol{\psi}^\pi(s, a)^\top \mathbf{w}$$

This is the **dynamics-and-reward decoupling**. The vector $\boldsymbol{\psi}^\pi$ depends on $\pi$ and the world dynamics but *not on the reward*; the vector $\mathbf{w}$ depends on the reward but *not on the dynamics*. So when the agent's preferences shift — coffee yesterday, food today — only $\mathbf{w}$ changes, and *all the work the agent has done learning $\boldsymbol{\psi}^\pi$ is preserved*.

**Generalised policy improvement (GPI)** is the second piece. The classic policy-improvement theorem says: given one value function, acting greedily gives you a better policy. GPI generalises this to *many* value functions at once — given $n$ policies $\pi_1, \dots, \pi_n$ and their value functions on the *new* task, the policy that, at each state, picks the action that maximises the pointwise max of those value functions is provably **at least as good as any individual $\pi_i$**. Combined with SFs, this is a zero-shot transfer mechanism: pretrain a library of $\boldsymbol{\psi}^{\pi_i}$, dot-product each one with the new task's $\mathbf{w}$, and GPI assembles a good policy at the new task without any further training.

**Why the project cares.** The project is building "modulator-conditioned" value architectures where a one-dimensional or few-dimensional context vector (a serotonin level, a pain weight, a discount factor) reshapes the agent's value function. SFs are *exactly* this kind of decoupling, formalised — the modulator-like quantity is the weight vector $\mathbf{w}$, the dynamics-only piece is $\boldsymbol{\psi}^\pi$, and GPI is a principled way to *blend* policies trained at different modulator settings when the agent enters a new context. The three papers in this batch trace the conceptual arc:

- **Barreto et al. 2017 (NeurIPS)** — establishes the linear-reward / SF / GPI framework with theoretical guarantees (Theorems 1 and 2) and toy + MuJoCo-reacher experiments.
- **Barreto et al. 2019 (ICML, also known as Barreto et al. 2018)** — relaxes the linear-reward assumption (Proposition 1 covers *any* reward), shows that **the base tasks' own rewards can be used as the feature vector $\boldsymbol{\phi}$**, and scales the framework to a 3-D first-person DeepMind Lab environment with images-as-observations.
- **Borsa et al. 2018 — Universal Successor Features Approximators (USFA)** — completes the arc by combining SFs with UVFA-style policy conditioning, training one network $\boldsymbol{\psi}(s, a, \mathbf{z})$ that generalises across an entire family of policy descriptors, and using GPI over a *sampled* candidate set at act time. (This paper was already fully reviewed in `Gamma_lit_review_A_universal_vf.md` from a UVFA-context-framing perspective — see brief recap and cross-reference in Section 3 below.)

Phase 1 of each paper is plain-language; Phase 2 carries the math, derivations, and theorems. The per-paper section-by-section backbones are kept as appendices so the synthesis remains traceable.

---

## Table of Contents

- [1. Barreto et al. 2017 — Successor Features for Transfer in Reinforcement Learning](#1-barreto-et-al-2017--successor-features-for-transfer-in-reinforcement-learning)
  - [1.1 Phase 1 — Foundational Overview](#11-phase-1--foundational-overview)
  - [1.2 Phase 2 — Graduate-Level Deep Dive](#12-phase-2--graduate-level-deep-dive)
  - [1.3 Appendix — Section-by-Section Backbone](#13-appendix--section-by-section-backbone)
- [2. Barreto et al. 2019 — Transfer in Deep RL Using SF and GPI](#2-barreto-et-al-2019--transfer-in-deep-rl-using-sf-and-gpi)
  - [2.1 Phase 1 — Foundational Overview](#21-phase-1--foundational-overview)
  - [2.2 Phase 2 — Graduate-Level Deep Dive](#22-phase-2--graduate-level-deep-dive)
  - [2.3 Appendix — Section-by-Section Backbone](#23-appendix--section-by-section-backbone)
- [3. Borsa et al. 2018 — Universal Successor Features Approximators (brief TD-centric recap)](#3-borsa-et-al-2018--universal-successor-features-approximators-brief-td-centric-recap)
  - [3.1 The SF arc in three steps — SF → deep SF + GPI → universal SF](#31-the-sf-arc-in-three-steps--sf--deep-sf--gpi--universal-sf)
  - [3.2 The SF Bellman equation and the universal-SF generalisation](#32-the-sf-bellman-equation-and-the-universal-sf-generalisation)
  - [3.3 The GPI policy under USFA](#33-the-gpi-policy-under-usfa)
  - [3.4 Late conditioning — the architectural lesson](#34-late-conditioning--the-architectural-lesson)
  - [3.5 Cross-reference to the full review](#35-cross-reference-to-the-full-review)
- [4. Cross-paper synthesis and project connections](#4-cross-paper-synthesis-and-project-connections)

---

## 1. Barreto et al. 2017 — Successor Features for Transfer in Reinforcement Learning

**PDF:** `docs/project/references/TD/sources/Barreto et al. 2017 - Successor Features for Transfer in Reinforcement Learning.pdf`
**Venue:** NeurIPS 2017
**Authors:** André Barreto, Will Dabney, Rémi Munos, Jonathan J. Hunt, Tom Schaul, Hado van Hasselt, David Silver (DeepMind)

### 1.1 Phase 1 — Foundational Overview

**The problem.** A reinforcement-learning agent that has mastered one task is usually helpless when the reward changes. If we want agents that *transfer* between tasks — that exploit experience on past tasks to learn faster on new ones — we need a representation that separates "what is the agent doing in the world" from "what the agent currently wants." Standard $Q^\pi(s, a)$ entangles them.

**The two key ideas.**

1. **Successor features (SF).** Assume the reward function is **linear in some features** of the transition:
   $$r(s, a, s') \;=\; \boldsymbol{\phi}(s, a, s')^\top \mathbf{w}$$
   where $\boldsymbol{\phi}(s, a, s') \in \mathbb{R}^d$ is a feature vector ("what salient events happened in this transition?") and $\mathbf{w} \in \mathbb{R}^d$ is a task-specific weight vector ("how much do I care about each event?"). Then the value function $Q^\pi$ factorises cleanly:
   $$Q^\pi(s, a) \;=\; \boldsymbol{\psi}^\pi(s, a)^\top \mathbf{w}, \qquad \boldsymbol{\psi}^\pi(s, a) \;=\; \mathbb{E}^\pi\!\left[\sum_{i = t}^{\infty} \gamma^{i - t}\, \boldsymbol{\phi}(s_i, a_i, s_{i+1}) \,\Big|\, s_t = s,\, a_t = a\right]$$
   The vector $\boldsymbol{\psi}^\pi(s, a)$ — the **successor features of $(s, a)$ under $\pi$** — is the discounted expected feature occurrence; it depends on the environment's dynamics and on $\pi$, but **not** on $\mathbf{w}$. So when the agent moves to a new task with weights $\mathbf{w}'$, evaluating its old policy on the new task is just one matrix-vector product: $Q^\pi_{\mathbf{w}'}(s, a) = \boldsymbol{\psi}^\pi(s, a)^\top \mathbf{w}'$. **No retraining of the dynamics piece is needed.**

2. **Generalised policy improvement (GPI).** Bellman's classical policy-improvement theorem says: from one value function $Q^\pi$, the greedy policy is no worse than $\pi$. **GPI extends this to many policies at once** — given $n$ value functions $Q^{\pi_1}, \dots, Q^{\pi_n}$, define
   $$\pi(s) \;\in\; \arg\max_a \max_i Q^{\pi_i}(s, a)$$
   Then $\pi$ is **at least as good as every $\pi_i$**, at every state and action. The proof handles the realistic case where the $Q^{\pi_i}$ are only approximated up to error $\epsilon$ — the bound just gets a $\tfrac{2}{1 - \gamma}\epsilon$ slack term.

**Why the combination works.** SFs make it cheap to evaluate every $\pi_i$ on a brand-new task — just dot-product with $\mathbf{w}'$. GPI then assembles a guaranteed-good policy from those evaluations *without any further RL training*. The agent gets a useful initial policy on the new task **before any learning has taken place** — a remarkable property that opens the door to building libraries of reusable skills.

**Key results.**

- **Theorem 1 (GPI).** Acting greedy w.r.t. the pointwise max of $n$ approximate Q-functions yields a policy whose return is at least $\max_i Q^{\pi_i}(s, a) - \tfrac{2}{1 - \gamma}\epsilon$.
- **Theorem 2 (SF transfer bound).** If the agent has SFs for $n$ training tasks with weights $\mathbf{w}_1, \dots, \mathbf{w}_n$, and is now facing task $\mathbf{w}_i$, the gap between the optimal $Q^{\pi^*_i}_i$ and the GPI policy's $Q^\pi$ is bounded by $\tfrac{2}{1-\gamma}\bigl(\phi_{\max}\, \min_j \|\mathbf{w}_i - \mathbf{w}_j\| + \epsilon\bigr)$ — i.e., performance degrades smoothly with the distance from the closest training task in **task-weight space**.
- **Four-room navigation experiment.** A 2-D four-room grid with three classes of pickup objects whose rewards re-sample uniformly from $[-1, 1]^3$ every 20 000 transitions. The agent must survive a sequence of 250 tasks. The proposed SF agent (SFQL) — with either ground-truth $\boldsymbol{\phi}$ or features learned by multi-task regression on the first 20 tasks — outperforms Q-learning by ~3× and probabilistic policy reuse (PRQL) by ~2× on average return.
- **MuJoCo reacher experiment (SFDQN).** A two-joint robotic arm with 12 candidate targets; the agent trains on only 4 of them. SFDQN — SFs + DQN + GPI — generalises to the 8 unseen targets, achieving near-DQN-on-target performance even on tasks for which it has *no specialised policy*. Crucially, training on *any* one task improves performance on *all* tasks via the shared $\boldsymbol{\phi}$ representation.

**Initial takeaway.** The 2017 paper does two structural things. It establishes that under the linear-reward assumption, a single learned vector $\boldsymbol{\psi}^\pi$ replaces an entire family of Q-functions across tasks — and that GPI lets you combine multiple such SFs without retraining. The theoretical guarantees (Theorems 1 and 2) are unusually clean for an RL transfer paper. The main limitations are (a) the linear-reward assumption is restrictive, (b) features $\boldsymbol{\phi}$ must be specified or learned offline, and (c) experiments use 2-D state spaces. The 2019 paper addresses (a) and (b); USFA (Borsa 2018) addresses scaling generalisation over policies as well as tasks.

### 1.2 Phase 2 — Graduate-Level Deep Dive

#### 1.2.1 Setup and the linear-reward assumption

Consider a Markov decision process $M \equiv (\mathcal{S}, \mathcal{A}, p, R, \gamma)$ with state space $\mathcal{S}$, action space $\mathcal{A}$, transition kernel $p(\cdot \mid s, a)$, reward variable $R(s, a, s')$ with expectation $r(s, a, s') = \mathbb{E}_{S' \sim p(\cdot \mid s, a)}[R(s, a, S')]$, and discount factor $\gamma \in [0, 1)$. The action-value function of a policy $\pi: \mathcal{S} \to \mathcal{A}$ is

$$Q^\pi(s, a) \;\equiv\; \mathbb{E}^\pi\!\left[\sum_{i = 0}^{\infty} \gamma^i R_{t + i + 1} \,\Big|\, S_t = s,\, A_t = a\right] \tag{1}$$

The **structural assumption** that powers the rest of the paper is that the expected reward decomposes as a linear function in features:

$$r(s, a, s') \;=\; \boldsymbol{\phi}(s, a, s')^\top \mathbf{w} \tag{2}$$

with $\boldsymbol{\phi} \in \mathbb{R}^d$ and $\mathbf{w} \in \mathbb{R}^d$. Barreto et al. note that **(2) is not restrictive in principle** — if one component $\phi_i(s, a, s') = r(s, a, s')$ then $\mathbf{w} = \mathbf{e}_i$ recovers the reward exactly. The substance of the assumption is that **a *small* $d$ suffices** — i.e., that the reward across the family of tasks of interest lives in a low-dimensional space.

#### 1.2.2 Derivation of the successor-feature decomposition

Write $\boldsymbol{\phi}_t \equiv \boldsymbol{\phi}(s_t, a_t, s_{t+1})$. Substituting (2) into the definition of $Q^\pi$:

$$\begin{aligned}
Q^\pi(s, a) &\;=\; \mathbb{E}^\pi[r_{t+1} + \gamma r_{t+2} + \cdots \mid S_t = s, A_t = a] \\
&\;=\; \mathbb{E}^\pi[\boldsymbol{\phi}_{t}^\top \mathbf{w} + \gamma \boldsymbol{\phi}_{t+1}^\top \mathbf{w} + \cdots \mid S_t = s, A_t = a] \\
&\;=\; \mathbb{E}^\pi\!\left[\sum_{i = t}^{\infty} \gamma^{i - t}\, \boldsymbol{\phi}_i \,\Big|\, S_t = s, A_t = a\right]^{\!\top} \mathbf{w} \\
&\;=\; \boldsymbol{\psi}^\pi(s, a)^\top \mathbf{w} \tag{3}
\end{aligned}$$

The two-line derivation hinges on linearity: $\mathbf{w}$ commutes with the discounted sum and the expectation. The vector

$$\boldsymbol{\psi}^\pi(s, a) \;\equiv\; \mathbb{E}^\pi\!\left[\sum_{i = t}^{\infty} \gamma^{i - t}\, \boldsymbol{\phi}(s_i, a_i, s_{i+1}) \,\Big|\, S_t = s, A_t = a\right] \in \mathbb{R}^d$$

is the **successor features of $(s, a)$ under $\pi$**. Each component is itself a value function — but with $\boldsymbol{\phi}_i$ (the $i$-th feature) playing the role of reward.

In the tabular case with $\boldsymbol{\phi}$ a one-hot indicator of the transition triple $(s, a, s') \in \mathcal{S} \times \mathcal{A} \times \mathcal{S}$, $\boldsymbol{\psi}^\pi(s, a)$ recovers Dayan's (1993) **successor representation** (SR) — the discounted count of transitions under $\pi$. The SF formulation is strictly more general:
- It handles continuous state and action spaces without modification.
- It explicitly invites function approximation by parameterising $\boldsymbol{\psi}^\pi(s, a; \theta)$ as a neural network.
- It collapses to SR when $\boldsymbol{\phi}$ is the indicator basis.

#### 1.2.3 SFs satisfy their own Bellman equation

The decomposition (3) implies that $\boldsymbol{\psi}^\pi$ obeys a vector-valued Bellman recursion:

$$\boldsymbol{\psi}^\pi(s, a) \;=\; \mathbb{E}\!\left[\boldsymbol{\phi}(s, a, S') + \gamma\, \boldsymbol{\psi}^\pi(S', \pi(S')) \,\big|\, s, a\right] \tag{4}$$

This is the key practical implication: **SFs can be learned by any standard RL algorithm** (TD, Q-learning, SARSA, deep variants) by treating each of the $d$ components of $\boldsymbol{\psi}^\pi$ as a scalar value function with reward $\phi_i$. The Bellman operator $\mathcal{T}^\pi$ acts component-wise, so contraction properties carry over.

#### 1.2.4 Theorem 1 — Generalised Policy Improvement (GPI)

**Statement (Theorem 1).** Let $\pi_1, \dots, \pi_n$ be $n$ decision policies and $\tilde{Q}^{\pi_1}, \dots, \tilde{Q}^{\pi_n}$ be approximations of their action-value functions such that
$$\bigl|Q^{\pi_i}(s, a) - \tilde{Q}^{\pi_i}(s, a)\bigr| \;\leq\; \epsilon \qquad \forall\, s \in \mathcal{S},\, a \in \mathcal{A},\, i \in \{1, \dots, n\} \tag{6}$$
Define
$$\pi(s) \;\in\; \arg\max_a \max_i \tilde{Q}^{\pi_i}(s, a) \tag{7}$$
Then
$$Q^\pi(s, a) \;\geq\; \max_i Q^{\pi_i}(s, a) \;-\; \frac{2}{1 - \gamma}\epsilon \qquad \forall\, s, a \tag{8}$$

**Proof sketch (full proof in the paper's supplement).** Let $Q_{\max}(s, a) \equiv \max_i Q^{\pi_i}(s, a)$ and $\tilde{Q}_{\max}(s, a) \equiv \max_i \tilde{Q}^{\pi_i}(s, a)$. From (6),
$$\bigl|Q_{\max}(s, a) - \tilde{Q}_{\max}(s, a)\bigr| \;=\; \bigl|\max_i Q^{\pi_i} - \max_i \tilde{Q}^{\pi_i}\bigr| \;\leq\; \max_i |Q^{\pi_i} - \tilde{Q}^{\pi_i}| \;\leq\; \epsilon$$
For any $i$, applying the Bellman operator $\mathcal{T}^\pi$ (where $\pi$ is defined by (7)) to $\tilde{Q}_{\max}$:
$$\begin{aligned}
\mathcal{T}^\pi \tilde{Q}_{\max}(s, a) &\;=\; r(s, a) + \gamma \sum_{s'} p(s' \mid s, a) \tilde{Q}_{\max}(s', \pi(s')) \\
&\;=\; r(s, a) + \gamma \sum_{s'} p(s' \mid s, a) \max_b \tilde{Q}_{\max}(s', b) \\
&\;\geq\; r(s, a) + \gamma \sum_{s'} p(s' \mid s, a) \max_b Q_{\max}(s', b) - \gamma \epsilon \\
&\;\geq\; r(s, a) + \gamma \sum_{s'} p(s' \mid s, a) Q_{\max}(s', \pi_i(s')) - \gamma \epsilon \\
&\;\geq\; r(s, a) + \gamma \sum_{s'} p(s' \mid s, a) Q^{\pi_i}(s', \pi_i(s')) - \gamma \epsilon \\
&\;=\; \mathcal{T}^{\pi_i} Q^{\pi_i}(s, a) - \gamma \epsilon \;=\; Q^{\pi_i}(s, a) - \gamma \epsilon
\end{aligned}$$
Since this holds for any $i$, $\mathcal{T}^\pi \tilde{Q}_{\max}(s, a) \geq Q_{\max}(s, a) - \gamma \epsilon \geq \tilde{Q}_{\max}(s, a) - \epsilon - \gamma \epsilon$. The Bellman operator $\mathcal{T}^\pi$ is a $\gamma$-contraction and $Q^\pi$ is its fixed point, so iterating yields
$$Q^\pi(s, a) \;=\; \lim_{k \to \infty} (\mathcal{T}^\pi)^k \tilde{Q}_{\max}(s, a) \;\geq\; \tilde{Q}_{\max}(s, a) - \frac{1 + \gamma}{1 - \gamma}\epsilon \;\geq\; Q_{\max}(s, a) - \epsilon - \frac{1 + \gamma}{1 - \gamma}\epsilon$$
Simplifying $\epsilon + \tfrac{1+\gamma}{1-\gamma}\epsilon = \tfrac{2}{1-\gamma}\epsilon$ gives (8). $\square$

**Interpretation.** In the exact case ($\epsilon = 0$), $\pi$ is at least as good as **every** $\pi_i$, simultaneously, at every state-action — a non-trivial extension of the single-policy improvement theorem. Note that $\max_i Q^{\pi_i}$ is **not** in general the value function of any single policy, so the result has structure beyond ordinary policy improvement.

When applied within a standard DP loop where each successive policy dominates its predecessors, GPI collapses to standard improvement (the most recent policy dominates the max). GPI is genuinely useful in two scenarios: (i) when many policies are evaluated in parallel, and (ii) when the underlying MDP changes — i.e., transfer.

#### 1.2.5 Theorem 2 — SF-specific transfer bound

For transfer, Barreto et al. prove a stronger bound that exploits the structure of the linear-reward family $\mathcal{M}^{\boldsymbol{\phi}}$.

**Statement (Theorem 2).** Let $M_i \in \mathcal{M}^{\boldsymbol{\phi}}$ and let $Q^{\pi^*_j}_i$ be the action-value function of an optimal policy of $M_j \in \mathcal{M}^{\boldsymbol{\phi}}$ when executed in $M_i$. Given approximations $\{\tilde{Q}^{\pi^*_j}_i\}_{j=1}^n$ such that
$$\bigl|Q^{\pi^*_j}_i(s, a) - \tilde{Q}^{\pi^*_j}_i(s, a)\bigr| \;\leq\; \epsilon \qquad \forall\, s, a, j$$
let $\pi(s) \in \arg\max_a \max_j \tilde{Q}^{\pi^*_j}_i(s, a)$ and $\phi_{\max} \equiv \max_{s, a} \|\boldsymbol{\phi}(s, a)\|$. Then
$$Q^{\pi^*_i}_i(s, a) - Q^\pi_i(s, a) \;\leq\; \frac{2}{1 - \gamma}\!\left(\phi_{\max}\, \min_j \|\mathbf{w}_i - \mathbf{w}_j\| \;+\; \epsilon\right) \tag{10}$$

**Proof structure.** The proof combines Theorem 1 with a **reward-distance lemma** (Lemma 1 in the supplement): for any task pair $(i, j)$ with reward difference $\delta_{ij} = \max_{s, a}|r_i(s, a) - r_j(s, a)|$,
$$Q^{\pi^*_i}_i(s, a) - Q^{\pi^*_j}_i(s, a) \;\leq\; \frac{2 \delta_{ij}}{1 - \gamma}$$
This lemma in turn factors as $|Q^{\pi^*_i}_i - Q^{\pi^*_i}_j| + |Q^{\pi^*_i}_j - Q^{\pi^*_j}_j|$, each bounded by $\delta_{ij}/(1-\gamma)$ via a standard Bellman-contraction argument. Substituting $\delta_{ij} = \max_{s,a}|\boldsymbol{\phi}(s,a)^\top(\mathbf{w}_i - \mathbf{w}_j)| \leq \phi_{\max}\|\mathbf{w}_i - \mathbf{w}_j\|$ (Cauchy-Schwarz) and chaining with Theorem 1 yields (10). $\square$

**Interpretation.** The bound relates the **distance in task-weight space** $\|\mathbf{w}_i - \mathbf{w}_j\|$ to the **performance gap** on the new task. This formalises the intuition that an agent should perform well on a task if it has solved a similar one before. Operationally, the bound gives an algorithm for **memory management**: when storing SFs $\tilde{\boldsymbol{\psi}}^{\pi^*_j}$ has cost, the agent should keep SFs whose $\mathbf{w}_j$'s cover the task space densely — and can decide to add a new SF $\tilde{\boldsymbol{\psi}}^{\pi^*_i}$ when $\min_j \|\mathbf{w}_i - \mathbf{w}_j\|$ exceeds a threshold.

#### 1.2.6 The SF-based transfer protocol

The 2017 framework operationalises transfer as follows:

1. **Training tasks.** Agent solves $n$ tasks $M_1, \dots, M_n \in \mathcal{M}^{\boldsymbol{\phi}}$ with weights $\mathbf{w}_1, \dots, \mathbf{w}_n$, learning SFs $\tilde{\boldsymbol{\psi}}^{\pi^*_j}(s, a)$ for each.
2. **New task.** Agent is exposed to $M_{n+1}$ with weights $\mathbf{w}_{n+1}$. It first solves a **supervised regression** to estimate $\mathbf{w}_{n+1}$ from observed transitions $(s, a, s', r)$:
   $$\tilde{\mathbf{w}}_{n+1} \;\in\; \arg\min_{\mathbf{w}} \sum_{(s, a, s', r) \in \mathcal{D}} \bigl(r - \boldsymbol{\phi}(s, a, s')^\top \mathbf{w}\bigr)^2$$
3. **Instantaneous evaluation.** For each $j$,
   $$\tilde{Q}^{\pi^*_j}_{n+1}(s, a) \;=\; \tilde{\boldsymbol{\psi}}^{\pi^*_j}(s, a)^\top \tilde{\mathbf{w}}_{n+1}$$
   No RL training is needed.
4. **GPI policy on new task.** The agent's behaviour policy is
   $$\pi_{n+1}(s) \;\in\; \arg\max_a \max_j \tilde{\boldsymbol{\psi}}^{\pi^*_j}(s, a)^\top \tilde{\mathbf{w}}_{n+1}$$
5. **Optional specialisation.** Concurrently, the agent can learn its own SFs $\tilde{\boldsymbol{\psi}}^{\pi^*_{n+1}}$ for $M_{n+1}$ — using the GPI policy for exploration and (4) as the TD target — and add them to the library for future tasks.

#### 1.2.7 SFQL — the practical algorithm

The four-room experiments instantiate this protocol with $\boldsymbol{\phi}(s, a, s')$ a $(n_c + 1)$-dimensional indicator of "agent picked up an object of class $c$" plus "agent reached the goal." Two variants:

- **SFQL-$\boldsymbol{\phi}$.** Ground-truth $\boldsymbol{\phi}$ given to the agent.
- **SFQL-$h$.** Features $\tilde{\boldsymbol{\phi}} \in \mathbb{R}^h$ learned by multi-task regression on the first 20 tasks, following Caruana (1997) and Baxter (2000). Notably, **SFQL-$h$ outperforms SFQL-$\boldsymbol{\phi}$**, hypothesised to be because the learned features are active across more of $\mathcal{S} \times \mathcal{A} \times \mathcal{S}$, providing a denser pseudo-reward signal for TD learning of $\boldsymbol{\psi}$.

The SF representation in SFQL is $\tilde{\boldsymbol{\psi}}^\pi_a(s) = \boldsymbol{\varphi}(s)^\top \mathbf{Z}^\pi_a$ where $\boldsymbol{\varphi}(s) \in \mathbb{R}^D$ is a hand-designed feature representation (radial-basis functions + object inventory) and $\mathbf{Z}^\pi_a \in \mathbb{R}^{D \times d}$ are learned weights. The Q-value collapses to $\tilde{Q}^\pi(s, a) = \boldsymbol{\varphi}(s)^\top \mathbf{Z}^\pi_a \tilde{\mathbf{w}}$, exposing the bilinear structure.

#### 1.2.8 SFDQN — the deep variant

For the MuJoCo reacher domain (continuous state, sparse reward), SFDQN replaces Q-learning with DQN as the backbone. Architecturally:
- Input: state $s$ (joint angles + velocities).
- The network outputs $\tilde{\boldsymbol{\psi}}^{\pi^*_i}(s, a) \in \mathbb{R}^d$ for **each of the $n = 4$ training-target policies** $\pi^*_i$ (one head per task).
- $\boldsymbol{\phi}(s, a, s') \in \mathbb{R}^{12}$ is the negation of distances to each of 12 target regions (4 training + 8 test); $\mathbf{w}$ is a one-hot indicator of the current target.
- Behaviour policy on training task $i$: GPI over $\tilde{\boldsymbol{\psi}}^{\pi^*_j}(s, b)^\top \mathbf{w}_i$ for $j = 1, \dots, n$ — *the agent uses GPI even during training*.

Crucially, **every transition is used to update all four** $\tilde{\boldsymbol{\psi}}^{\pi^*_j}$ heads (each with its own TD target derived from policy $\pi^*_j$'s greedy action). Result: training on any one of the 4 training tasks improves performance on all 4, plus generalises to the 8 test tasks via Theorem 2.

#### 1.2.9 Connections to related work

- **Successor representation (Dayan 1993).** SF is the function-approximator generalisation; SR is the tabular special case with one-hot $\boldsymbol{\phi}$.
- **Predictive state representations (Littman et al. 2001).** PSR summarises the dynamics of the *environment*; SF summarises the dynamics of a *policy* $\pi$ in the environment.
- **General value functions / Horde (Sutton et al. 2011).** GVFs predict pseudo-rewards under fixed policies; $\psi^\pi_i$ is a particular GVF with $\phi_i$ as pseudo-reward. The contribution of SF&GPI over Horde is the **structural assumption (2) plus the GPI combination rule**, which together turn a collection of independent predictions into a transfer mechanism with formal guarantees.
- **UVFA (Schaul et al. 2015).** $\max_j \tilde{\boldsymbol{\psi}}^{\pi^*_j}(s, a)^\top \tilde{\mathbf{w}}$ can be viewed as a UVFA whose goal axis is the task descriptor $\tilde{\mathbf{w}}$. The 2017 paper notes this connection but does not pursue it; USFA (Borsa 2018) makes it explicit.
- **Hierarchical RL / Options (Sutton, Precup, Singh 1999).** Each $\boldsymbol{\psi}^{\pi^*_j}$ corresponds to a one-step option for $\pi^*_j$; GPI is planning at a higher level over this set.

#### 1.2.10 Strengths and limitations

**Strengths.**
- Two-line derivation of the SF decomposition (3) — conceptually clean.
- Theorem 1 (GPI) extends a classical theorem in a structurally meaningful way; the $\tfrac{2}{1-\gamma}\epsilon$ slack matches the standard single-policy approximation penalty.
- Theorem 2 (transfer bound) explicitly links **distance in task-weight space** to **performance gap** — operationally useful for memory management.
- The framework is RL-paradigm-agnostic: SFs satisfy a Bellman equation and can be learned by any RL algorithm.
- The DQN integration (SFDQN) shows the framework works with deep nonlinear approximators.

**Limitations.**
- **Linear-reward assumption (2).** Real reward functions are rarely linear in a small fixed feature vector. The 2019 paper addresses this.
- **Features $\boldsymbol{\phi}$ must be given or learned offline** by multi-task regression. The 2019 paper proposes using base-task *rewards themselves* as features, eliminating this step.
- **The framework as stated requires storing one $\boldsymbol{\psi}^\pi$ per past task.** USFA (Borsa 2018) collapses this into a single network conditioned on a policy descriptor $\mathbf{z}$.
- **GPI's guarantee scales as $1/(1-\gamma)$**, loose at long horizons.
- **GPI is greedy over the candidate set.** It does not learn how to combine policies smoothly; the max operation is brittle to large approximation errors.

### 1.3 Appendix — Section-by-Section Backbone

#### Abstract
Transfer in RL: scenario where reward changes between tasks but dynamics remain the same. Two key ideas — **successor features** (value-function representation decoupling dynamics from rewards) and **generalised policy improvement** (extension of DP's policy improvement to multiple policies). Together: an approach integrated with RL, allowing free flow of information across tasks, with **performance guarantees before any learning** has taken place. Two theorems support the framework; experiments in four-room navigation and a two-joint robotic arm.

#### 1. Introduction
RL framework, examples of task decomposition. Transfer: leveraging knowledge on past tasks to improve performance on new ones. Two desiderata: (i) information flow not dictated by hierarchical/temporal rigidity; (ii) transfer integrated into RL. The paper builds on two pillars: **SF** (generalisation of Dayan's SR — extends to continuous spaces, facilitates function approximation, decouples dynamics from rewards) and **GPI** (generalisation of Bellman's policy improvement to multiple policies — provides performance guarantees on new tasks before learning).

#### 2. Background and problem formulation
MDP $(\mathcal{S}, \mathcal{A}, p, R, \gamma)$, Q-function (1), policy improvement, DP. Transfer defined: training on $\mathcal{T}$ always at least as good as training on $\mathcal{T}' \subset \mathcal{T}$.

#### 3. Successor features
Reward model (2): $r = \boldsymbol{\phi}^\top \mathbf{w}$. Derivation (3): $Q^\pi(s, a) = \boldsymbol{\psi}^\pi(s, a)^\top \mathbf{w}$. SR as special case (tabular, one-hot $\boldsymbol{\phi}$). SF extends SR to continuous spaces and function approximation. Bellman equation (4): $\boldsymbol{\psi}^\pi(s, a) = \boldsymbol{\phi}_{t+1} + \gamma \mathbb{E}^\pi[\boldsymbol{\psi}^\pi(S_{t+1}, \pi(S_{t+1}))]$.

#### 4. Transfer via successor features
Setup: $\mathcal{M}^{\boldsymbol{\phi}}(\mathcal{S}, \mathcal{A}, p, \gamma) \equiv \{M : r = \boldsymbol{\phi}^\top \mathbf{w}\}$ (5). Each instantiation of $\mathbf{w}$ is a task. Examples: hunger/thirst preference shifts, varying product prices, changing feature availability.

##### 4.1 Generalised policy improvement
Theorem 1 statement and proof structure. Discussion: $\pi$ strictly better than all $\pi_i$ if no single policy dominates; GPI useful for transfer (MDP changes) or parallel evaluation.

##### 4.2 Generalised policy improvement with successor features
SF-based transfer protocol: store $\tilde{\boldsymbol{\psi}}^{\pi^*_j}$, estimate $\tilde{\mathbf{w}}_{n+1}$ on new task, compute $\tilde{Q}^{\pi^*_j}_{n+1} = \tilde{\boldsymbol{\psi}}^{\pi^*_j \top}\tilde{\mathbf{w}}_{n+1}$, apply GPI. Theorem 2 statement — performance gap bounded by task-weight distance. Memory-management heuristic: add new $\tilde{\boldsymbol{\psi}}^{\pi^*_i}$ when $\min_j \|\tilde{\mathbf{w}}_i - \tilde{\mathbf{w}}_j\|$ exceeds threshold.

#### 5. Experiments
Four-room domain: 250 tasks, rewards $\sim \mathrm{Unif}[-1, 1]^3$, every 20 000 steps. SFQL-$\boldsymbol{\phi}$ (given features), SFQL-$h$ (learned $\tilde{\boldsymbol{\phi}}$ via multi-task regression). Baselines: Q-learning (QL), probabilistic policy reuse (PRQL). Result: SFQL ≫ PRQL ≫ QL (Figure 2). Reacher domain (MuJoCo): 12 targets, 4 training. SFDQN with one $\tilde{\boldsymbol{\psi}}^{\pi^*_i}$ head per training task. Result: training on any task improves all tasks, including 8 test targets (Figure 3).

#### 6. Related work
Mehta et al. (2008): variable-reward HRL, average-reward setting, single-policy selection (no GPI). Fernández et al. (2010): probabilistic policy reuse (baseline). Bernstein (1999): relearns all $\tilde{\boldsymbol{\psi}}^{\pi^*_i}$ from scratch per task. PSR (Littman et al. 2001): summarises environment, not policy. GVF (Sutton et al. 2011) / Horde: SF as a special GVF. UVFA (Schaul et al. 2015): $\max_j \tilde{\boldsymbol{\psi}}^{\pi^*_j} {}^\top \tilde{\mathbf{w}}$ is a UVFA in disguise. Options (Sutton et al. 1999): SF as one-step options. Deep SR (Kulkarni 2016; Zhang 2016): similar architectures but no GPI.

#### 7. Conclusion
SF and GPI as complementary generalisations of classical ideas. SF&GPI is an elegant extension of DP for transfer; Theorem 2 formalises similar-task transfer; experiments demonstrate the principle. Future work: specialising components for diverse task families.

#### Supplementary material
Theorem 1 and 2 proofs (full); Lemma 1 (reward-distance bound); experiment details (four-room MDP definition, RBF feature design, task sequence, multi-task feature-learning protocol; reacher domain, SFDQN architecture, training schedule); additional empirical analysis.

---

## 2. Barreto et al. 2019 — Transfer in Deep RL Using SF and GPI

**PDF:** `docs/project/references/TD/sources/Barreto et al. 2019 - Transfer in deep reinforcement learning using successor features and generalised policy improvement.pdf`
**Venue:** ICML 2018 (arXiv 2019; the bibliographic entry in the project uses 2019)
**Authors:** André Barreto, Diana Borsa, John Quan, Tom Schaul, David Silver, Matteo Hessel, Daniel Mankowitz, Augustin Žídek, Rémi Munos (DeepMind)

### 2.1 Phase 1 — Foundational Overview

**The problem.** The 2017 framework requires (a) the reward function to be linear in a feature vector $\boldsymbol{\phi}$ that is *given or pre-learned*, and (b) experiments to live in low-dimensional state spaces. The 2018/2019 paper attacks both restrictions: it removes the strict linear-reward requirement *theoretically*, gives a *practical* recipe for learning $\boldsymbol{\phi}$ online while doing RL, and demonstrates SF&GPI on a 3-D first-person navigation task with image observations (DeepMind Lab).

**The two extensions.**

1. **Proposition 1 — guarantees beyond $\mathcal{M}^{\boldsymbol{\phi}}$.** The original Theorem 2 only bounds performance on tasks whose reward is *exactly* in $\mathrm{span}(\boldsymbol{\phi})$. Proposition 1 generalises to **any** task $M \in \mathcal{M}$ — i.e., any task that shares dynamics, regardless of whether its reward is linearly expressible in $\boldsymbol{\phi}$. The bound has an extra term $\|r - r_i\|_\infty$ that captures the "distance from $M$ to $\mathcal{M}^{\boldsymbol{\phi}}$"; the framework **degrades gracefully** when (2) is violated.

2. **Rewards as features.** Instead of solving an offline multi-task regression to learn $\boldsymbol{\phi}$, the authors propose using the **base-task rewards themselves** as features. Concretely, if the agent solves $D$ base tasks with reward functions $r_1, \dots, r_D$, take
   $$\tilde{\boldsymbol{\phi}}(s, a, s') \;\equiv\; \tilde{\mathbf{r}}(s, a, s') \;\equiv\; \bigl(\tilde{r}_1(s, a, s'), \dots, \tilde{r}_D(s, a, s')\bigr)^\top$$
   This is justified because *any* set of $D$ tasks can be expressed in this basis exactly (if they are linearly independent; otherwise their span). The trick has two payoffs:
   - **The SF components are now ordinary value functions.** $\psi^\pi_j(s, a) = Q^{\pi}_j(s, a)$ where $Q^\pi_j$ is the value of $\pi$ on base task $j$. So the agent can learn $\boldsymbol{\psi}^{\pi_i}$ by **standard Q-learning on the actual reward signals** rather than on noisy approximations of $\boldsymbol{\phi}$.
   - **The dependency chain collapses.** In the original recipe, learning $\boldsymbol{\psi}^{\pi_i}$ depends on $\tilde{\boldsymbol{\phi}}$, and $\tilde{\boldsymbol{\phi}}$'s data distribution depends on $\boldsymbol{\psi}^{\pi_i}$ — a circular instability. With rewards-as-features, $\tilde{\boldsymbol{\phi}} = \tilde{\mathbf{r}}$ is learned **independently** by a supervised regression, while $\tilde{\boldsymbol{\psi}}^{\pi_i} = \tilde{\mathbf{Q}}^{\pi_i}$ is learned by Q-learning on the actual rewards. **The two estimators do not share approximation error.**

**Key results.**

- **Proposition 1 (extended SF&GPI bound).** For any task $M \in \mathcal{M}$ (not just $\mathcal{M}^{\boldsymbol{\phi}}$), the GPI policy from approximate Q-functions over base tasks $M_j$ satisfies
  $$\|Q^* - Q^\pi\|_\infty \;\leq\; \frac{2}{1 - \gamma}\!\left(\|r - r_i\|_\infty + \min_j \|r_i - r_j\|_\infty + \epsilon\right)$$
  for any reference task $M_i$ (typically the closest to $M$ in $\mathcal{M}^{\boldsymbol{\phi}}$). The first term is the "distance from $M$ to $\mathcal{M}^{\boldsymbol{\phi}}$"; the second is the closest-training-task term inherited from Theorem 2.
- **Algorithm 1 — SF&GPI Q-learning.** Single deep network outputs $\{\tilde{\boldsymbol{\psi}}^{\pi_1}, \dots, \tilde{\boldsymbol{\psi}}^{\pi_n}\}$ and $\tilde{\mathbf{w}}$; behaviour policy is GPI over the SF library; optional "extend basis" mode learns $\tilde{\boldsymbol{\psi}}^{\pi_{n+1}}$ for the current task to be added to the library.
- **Algorithm 2 — build $\boldsymbol{\phi}$ and $\boldsymbol{\Psi}$ concurrently.** While solving $D$ base tasks, learn $\tilde{r}_i$ by supervised regression and $\tilde{Q}^{\pi_i}_j$ by Q-learning on actual rewards. GPI is used as the behaviour policy from the start, so base-task policies can already "cooperate" during the initial phase.
- **DeepMind Lab 3D navigation.** First-person image observations ($84 \times 84$); 4 object types (TV, ball, hat, balloon); base tasks $\hat{\mathcal{M}} = \{e_1, e_2, e_3, e_4\}$ (one-hot rewards). Architecture: CNN + LSTM for state $\tilde{s}_t = f(h_t)$, then $D + 1$ MLP blocks producing $\tilde{\boldsymbol{\phi}}(\tilde{s}_t, a)$ and $\tilde{\boldsymbol{\psi}}^{\pi_i}(\tilde{s}_t, a)$, then a final $\tilde{\mathbf{w}}$. Two variants:
  - **SF&GPI-transfer.** Pretrain on base tasks; at test time use only the supervised step to estimate $\tilde{\mathbf{w}}$ and behave under GPI. **Learns good policies almost instantaneously** on unseen test tasks.
  - **SF&GPI-continual.** Additionally learn a new specialised $\tilde{\boldsymbol{\psi}}^{\pi_{n+1}}$ on the test task. **Outperforms all baselines on almost every task.**
- **Negative-reward generalisation.** Base tasks have only positive rewards, but the agent generalises correctly to test tasks with negative rewards (e.g., "1100" means $+1$ for TVs and balls, $0$ for hats and balloons; "-1100" inverts the sign of the TV reward). This is the SF&GPI signature: the framework *combines* base-task policies in non-trivial ways via $\boldsymbol{\psi}^{\pi_i \top}\mathbf{w}'$ with possibly-negative $\mathbf{w}'$.
- **Linearly-dependent base tasks.** Even when base tasks $\hat{\mathcal{M}}' = \{1000, 0100, 0011, 1100\}$ span only a 3-D subspace of the 4-D task space, transfer **degrades gracefully** to tasks outside the span — consistent with Proposition 1.

**Initial takeaway.** Two structural moves: (a) the linear-reward assumption is theoretically *relaxed* (Proposition 1) so the framework has guarantees even when the assumption fails, and (b) the rewards-as-features trick **eliminates the bootstrap-instability** that plagued the original recipe (learning $\boldsymbol{\phi}$ and $\boldsymbol{\psi}$ in mutual dependence). Together they make SF&GPI a practical drop-in for deep RL at scale. The DeepMind Lab experiments confirm the framework works on image-based 3-D environments and exhibits non-trivial compositional behaviour (negative rewards never seen at training). The remaining limitations: the policy library is still discrete ($n$ stored SFs), and adding a new SF per task scales linearly with task count — USFA (Borsa 2018) addresses this by making policies *continuous-valued* via a learnable descriptor $\mathbf{z}$.

### 2.2 Phase 2 — Graduate-Level Deep Dive

#### 2.2.1 Extending the notion of environment

The 2017 paper restricts attention to $\mathcal{M}^{\boldsymbol{\phi}}(\mathcal{S}, \mathcal{A}, p, \gamma) \equiv \{M : r = \boldsymbol{\phi}^\top \mathbf{w}\}$ — MDPs whose reward lives in $\mathrm{span}(\boldsymbol{\phi})$. The 2019 paper enlarges this to

$$\mathcal{M}(\mathcal{S}, \mathcal{A}, p, \gamma) \;\equiv\; \{M(\mathcal{S}, \mathcal{A}, p, \cdot, \gamma)\} \tag{6}$$

i.e., **all** MDPs sharing $(\mathcal{S}, \mathcal{A}, p, \gamma)$, regardless of whether their reward is in $\mathrm{span}(\boldsymbol{\phi})$. Clearly $\mathcal{M} \supset \mathcal{M}^{\boldsymbol{\phi}}$.

#### 2.2.2 Proposition 1 — guarantees on the extended environment

**Statement (Proposition 1).** Let $M \in \mathcal{M}$ and $M_i \in \mathcal{M}^{\boldsymbol{\phi}}$ be a reference task. Let $Q^{\pi^*_j}_i$ be the action-value function of an optimal policy of $M_j \in \mathcal{M}^{\boldsymbol{\phi}}$ executed in $M_i$. Given approximations $\{\tilde{Q}^{\pi^*_j}_i\}$ such that $|Q^{\pi^*_j}_i - \tilde{Q}^{\pi^*_j}_i|_\infty \leq \epsilon$ for $j = 1, \dots, n$, let $\pi(s) \in \arg\max_a \max_j \tilde{Q}^{\pi^*_j}_i(s, a)$. Then

$$\|Q^* - Q^\pi\|_\infty \;\leq\; \frac{2}{1 - \gamma}\!\left(\|r - r_i\|_\infty + \min_j \|r_i - r_j\|_\infty + \epsilon\right) \tag{8}$$

where $Q^*$ is the optimal value function of $M$ and $Q^\pi$ is the value function of $\pi$ in $M$.

**Interpretation.** The bound decomposes into three pieces:
- $\|r - r_i\|_\infty$ — the **misspecification gap**: how far the target task $M$ is from the nearest task $M_i$ in $\mathcal{M}^{\boldsymbol{\phi}}$ that the framework can represent exactly. If $M \in \mathcal{M}^{\boldsymbol{\phi}}$, this term vanishes.
- $\min_j \|r_i - r_j\|_\infty$ — the **closest-training-task gap**, inherited from Theorem 2 (2017): how close the chosen reference $M_i$ is to one of the base tasks $M_j$ for which the agent has stored SFs.
- $\epsilon$ — the **approximation error** of the SFs themselves.

The bound says: SF&GPI guarantees performance for **any** task $M \in \mathcal{M}$, with a quality that **degrades gracefully** with the misspecification gap. This is a genuinely useful relaxation — in deep RL, the linear-reward assumption rarely holds exactly, so the original Theorem 2 has limited bite, while Proposition 1 still applies.

**Proof (sketch).** The proof generalises Theorem 2 via the same chain (Theorem 1 + reward-distance lemma) but introduces an additional triangle-inequality step linking $M$ to its nearest neighbour $M_i$ in $\mathcal{M}^{\boldsymbol{\phi}}$:
$$\|Q^*_M - Q^\pi_M\|_\infty \;\leq\; \underbrace{\|Q^*_M - Q^{\pi^*_M}_{M_i}\|_\infty}_{\text{from } \|r - r_i\|_\infty} + \underbrace{\|Q^{\pi^*_M}_{M_i} - Q^\pi_{M_i}\|_\infty}_{\text{Theorem 2 on } M_i} + \underbrace{\|Q^\pi_{M_i} - Q^\pi_M\|_\infty}_{\text{from } \|r - r_i\|_\infty}$$
The first and third terms each contribute $\frac{1}{1-\gamma}\|r - r_i\|_\infty$ via the standard reward-difference Bellman-contraction bound; the middle term is bounded by $\frac{2}{1-\gamma}(\min_j \|r_i - r_j\|_\infty + \epsilon)$ from Theorem 2. Summing and absorbing constants gives (8). $\square$ (Full proof in the paper's supplement.)

#### 2.2.3 Rewards as features — the structural move

The 2017 framework asks the agent to find features $\tilde{\boldsymbol{\phi}}$ such that $\tilde{\boldsymbol{\phi}}(s, a, s')^\top \tilde{\mathbf{w}}_i \approx r_i(s, a, s')$ for $i = 1, \dots, D$. This is a multi-task regression with $D$ targets, and learning $\tilde{\boldsymbol{\phi}}$ and the $\tilde{\mathbf{w}}_i$ jointly is delicate.

**The new observation.** Suppose we have a feature function $\boldsymbol{\phi}$ and weight vectors $\mathbf{w}_i$ that satisfy (2) exactly. Stack the weights into $\mathbf{W} \in \mathbb{R}^{D \times d}$, so $\mathbf{r}(s, a, s') = \mathbf{W}\boldsymbol{\phi}(s, a, s')$ where $\mathbf{r} \in \mathbb{R}^D$ stacks the $D$ reward signals. If $d$ of the $\mathbf{w}_i$'s are linearly independent, then $\boldsymbol{\phi}(s, a, s') = (\mathbf{W}^\top \mathbf{W})^{-1}\mathbf{W}^\top \mathbf{r}(s, a, s') = \mathbf{W}^\dagger \mathbf{r}(s, a, s')$. So $\boldsymbol{\phi}$ is a **linear transformation of $\mathbf{r}$** — meaning $\boldsymbol{\phi}$ and $\mathbf{r}$ have the **same span** as features, and the choice between them is just a change of basis.

This motivates **using the rewards themselves as features**:

$$\tilde{\boldsymbol{\phi}}(s, a, s') \;\equiv\; \tilde{\mathbf{r}}(s, a, s') \;\approx\; \mathbf{r}(s, a, s') \tag{10}$$

with $\tilde{r}_i$ a supervised approximation of $r_i$.

**Trade-off.** Setting $\tilde{\boldsymbol{\phi}} = \tilde{\mathbf{r}}$ fixes the dimensionality of $\boldsymbol{\phi}$ at $D$, removing the flexibility of choosing $d < D$ for a more compact representation. However, the simplifications make this a worthwhile trade in practice.

#### 2.2.4 Why rewards-as-features simplifies learning

With $\tilde{\boldsymbol{\phi}} = \tilde{\mathbf{r}}$, the SFs collapse to:

$$\tilde{\psi}^{\pi_i}_j(s, a) \;=\; \mathbb{E}^{\pi_i}\!\left[\sum_{k=0}^\infty \gamma^k \tilde{r}_j(s_k, a_k, s_{k+1})\right] \;\approx\; \mathbb{E}^{\pi_i}\!\left[\sum_{k=0}^\infty \gamma^k r_j(s_k, a_k, s_{k+1})\right] \;=\; \tilde{Q}^{\pi_i}_j(s, a)$$

That is, the $j$-th component of $\tilde{\boldsymbol{\psi}}^{\pi_i}$ is **literally the Q-value of $\pi_i$ on base task $j$**:

$$\tilde{\boldsymbol{\psi}}^{\pi_i} \;\equiv\; \tilde{\mathbf{Q}}^{\pi_i} \;=\; [\tilde{Q}^{\pi_i}_1, \tilde{Q}^{\pi_i}_2, \dots, \tilde{Q}^{\pi_i}_D]$$

**Key practical consequence.** When learning $\tilde{Q}^{\pi_i}_j$ by TD, the agent can use the **actual reward signal $r_j$**, not the approximation $\tilde{r}_j$. Compare:
- **Original SF Bellman update.** $\tilde{\boldsymbol{\psi}}^{\pi_i}(s, a) \leftarrow \tilde{\boldsymbol{\phi}}(s, a, s') + \gamma \tilde{\boldsymbol{\psi}}^{\pi_i}(s', \pi_i(s'))$ — approximation errors in $\tilde{\boldsymbol{\phi}}$ propagate into $\tilde{\boldsymbol{\psi}}^{\pi_i}$.
- **Rewards-as-features update.** $\tilde{Q}^{\pi_i}_j(s, a) \leftarrow r_j(s, a, s') + \gamma \tilde{Q}^{\pi_i}_j(s', \pi_i(s'))$ — uses the **true reward**, no $\tilde{r}_j$ dependency.

**This breaks the bootstrap instability.** In the original recipe, $\tilde{\boldsymbol{\phi}}$ depended on $\tilde{\boldsymbol{\psi}}^{\pi_i}$ (via the data distribution) and vice versa (via the TD target). With rewards-as-features, $\tilde{\boldsymbol{\psi}}^{\pi_i} = \tilde{\mathbf{Q}}^{\pi_i}$ is learned on actual rewards independently of $\tilde{\mathbf{r}}$ (which is learned by supervised regression). They share no approximation error.

#### 2.2.5 Algorithm 1 — SF&GPI Q-learning with optional basis extension

```
INPUT: φ̃, Ψ̃ ≡ {ψ̃^π_1, …, ψ̃^π_n}            // features and SFs from previous tasks
       extend_basis ∈ {true, false}            // learn a new SF on this task?
       α_ψ, α_w, ε, n_s                        // hyper-parameters

IF extend_basis:
  create ψ̃^π_{n+1} parameterised by θ_ψ; Ψ̃ ← Ψ̃ ∪ {ψ̃^π_{n+1}}

select initial state s ∈ S

FOR k = 1 … n_s:
  IF Bernoulli(ε) = 1: a ← Uniform(A)             // exploration
  ELSE: a ← argmax_b max_i ψ̃^π_i(s, b)^⊤ w̃          // GPI behaviour policy

  execute a, observe r, s'
  // supervised update of task-weight estimate
  w̃ ← w̃ − α_w · (r − φ̃(s, a, s')^⊤ w̃) · φ̃(s, a, s')

  IF extend_basis:
    a' ← argmax_b ψ̃^π_{n+1}(s, b)^⊤ w̃
    FOR j = 1, …, d:
      δ_j ← φ̃_j(s, a, s') + γ · ψ̃^π_{n+1}_j(s', a') − ψ̃^π_{n+1}_j(s, a)
      θ_ψ ← θ_ψ − α_ψ · δ_j · ∇_{θ_ψ} ψ̃^π_{n+1}_j(s, a)   // TD learn new SF

  s ← s' (or reinit)
```

**Two modes.**
- `extend_basis = false`: the agent relies entirely on GPI over the pre-trained SF library; the only learning is supervised regression of $\tilde{\mathbf{w}}$ on observed rewards.
- `extend_basis = true`: in addition, the agent learns a new $\tilde{\boldsymbol{\psi}}^{\pi_{n+1}}$ specialised to the current task, which is added to the library for future use.

#### 2.2.6 Algorithm 2 — building $\boldsymbol{\phi}$ and $\boldsymbol{\Psi}$ concurrently

```
INPUT: M_1, …, M_D base tasks
       α_Q, α_r, ε, n_s

FOR k = 1 … n_s:
  select task t ∈ {1, …, D} and state s ∈ S
  IF Bernoulli(ε) = 1: a ← Uniform(A)
  ELSE: a ← argmax_b max_i Q̃^π_i_t(s, b)              // GPI on Q̃^π_i for task t

  execute a in M_t, observe r and s'
  // supervised regression of r̃_t
  θ_r ← θ_r − α_r · (r − r̃_t(s, a, s')) · ∇_{θ_r} r̃_t(s, a, s')

  FOR i = 1, …, D:
    a' ← argmax_b Q̃^π_i_i(s', b)                    // greedy in i-th task
    θ_Q ← θ_Q − α_Q · (r_t + γ · Q̃^π_i_t(s', a') − Q̃^π_i_t(s, a)) · ∇_{θ_Q} Q̃^π_i_t(s, a)

RETURN φ̃ ≡ [r̃_1, …, r̃_D] and Ψ̃ ≡ {Q̃^π_1, …, Q̃^π_D}
```

**Key design choice.** Q-learning uses the actual reward $r_t$ from the observed transition, not the approximation $\tilde{r}_t$. The two estimators are loosely coupled — $\tilde{\mathbf{r}}$ is used only at act time (to compute the GPI argmax via $\tilde{\boldsymbol{\phi}} = \tilde{\mathbf{r}}$ and $\tilde{\mathbf{w}}$), not in the TD target.

GPI is used as the **behaviour policy from the very start** (line 4) — the agent does not wait for $\tilde{\mathbf{r}}$ to converge. This means the base-task policies can already "cooperate" during the initial phase: each $\pi_i$ benefits from $\pi_j$'s exploration via the GPI argmax.

#### 2.2.7 Architecture for DeepMind Lab

The deep architecture has three modules:

1. **Encoder.** Pixel observation $o_t \in \mathbb{R}^{84 \times 84 \times 3}$ → CNN → LSTM with state $h_t$ → state embedding $\tilde{s}_t = f(h_t)$. The LSTM is necessary because the first-person view is non-Markovian.
2. **Specialised heads.** $D + 1$ MLP blocks, each a single-hidden-layer MLP. The first $D$ produce $\tilde{\boldsymbol{\psi}}^{\pi_i}(\tilde{s}_t, a) \in \mathbb{R}^D \times |\mathcal{A}|$ for $i = 1, \dots, D$; the $(D+1)$-th produces $\tilde{\boldsymbol{\phi}}(\tilde{s}_t, a) \in \mathbb{R}^D$. (Note: the paper uses $\tilde{\boldsymbol{\phi}}(s, a)$ rather than $\tilde{\boldsymbol{\phi}}(s, a, s')$ because the network produces a per-state-action prediction of the *expected* next-step reward.)
3. **Task-weight head.** A learnable vector $\tilde{\mathbf{w}} \in \mathbb{R}^D$. Combined with $\tilde{\boldsymbol{\phi}}$ and $\tilde{\boldsymbol{\psi}}^{\pi_i}$ to produce the final Q-estimates.

**Trained end-to-end** through Algorithm 2 (build basis) on the base tasks $\hat{\mathcal{M}} = \{e_1, e_2, e_3, e_4\}$. Watkins's Q($\lambda$) with eligibility traces; distributed data collection via IMPALA (Espeholt et al. 2018).

#### 2.2.8 Experimental results — DeepMind Lab 3-D navigation

**Environment.** 3-D room with 4 object types (TV, ball, hat, balloon), 5 instances per type. Picking up an object respawns it elsewhere. Episodes last one minute. A task is a 4-vector specifying the reward per object type (e.g., "1-100" means $+1$ for TVs, $-1$ for balls, $0$ for hats and balloons).

**Base tasks.** $\hat{\mathcal{M}} = \{1000, 0100, 0010, 0001\}$ — only positive rewards on single object types.

**Test tasks.** Arbitrary $\mathbf{w} \in \mathbb{R}^4$, **including tasks with negative rewards** (e.g., "-1100", "1-101") — never seen at training.

**Methods compared.**
- **Q($\lambda$):** baseline with frozen $f(h)$ from base-task training, learn a fresh MLP on $\tilde{s}_t$.
- **DQ($\lambda$) fine-tuned:** baseline with end-to-end fine-tuning of $f(h)$ on the test task.
- **DQ($\lambda$) from scratch:** baseline with $f(h)$ randomly re-initialised on the test task.
- **SF&GPI-transfer:** Algorithm 1 with `extend_basis = false`. Only $\tilde{\mathbf{w}}$ is learned (supervised).
- **SF&GPI-continual:** Algorithm 1 with `extend_basis = true`. Adds $\tilde{\boldsymbol{\psi}}^{\pi_{n+1}}$.

**Findings.**
- **SF&GPI-transfer is near-instantaneous** on test tasks: the supervised regression converges so fast that the learning curve looks flat compared to the RL baselines.
- **SF&GPI-continual outperforms all baselines** on nearly every test task; it inherits SF&GPI-transfer's fast start and continues improving.
- **Negative-reward generalisation.** Base tasks had only positive rewards. On test tasks like "-1100" (avoid TVs, collect balls), SF&GPI is *considerably* better than the RL-from-scratch baselines, because the GPI argmax over $\tilde{\boldsymbol{\psi}}^{\pi_i \top}\mathbf{w}'$ with negative $w'_i$ correctly **disprefers** policies that score high on $\psi^{\pi_i}_i$. The avoid-TV behaviour is *compositional* — assembled at act time, never seen as a training-time policy.
- **Linearly-dependent base tasks.** Replacing $\hat{\mathcal{M}}$ with $\hat{\mathcal{M}}' = \{1000, 0100, 0011, 1100\}$ (3-D span in 4-D task space) **degrades transfer gracefully** — consistent with Proposition 1's $\|r - r_i\|_\infty$ misspecification term.

#### 2.2.9 The behavioural-basis caveat

The paper raises an interesting observation: spanning the *reward* space is **not enough** to span the *policy* space. Suppose the base-task rewards are replaced by their negations $\hat{\mathcal{M}}_{\mathrm{neg}} = \{-e_1, -e_2, -e_3, -e_4\}$. The base tasks still span the same reward subspace; but **the optimal policy for each negative base task is "stand still"** (avoid all objects). So the GPI library contains only standstill policies, and the agent cannot generalise to *positive-reward* test tasks via GPI.

This formalises that **GPI generalises behaviours, not rewards**. Designing a base-task set is implicitly designing a **behavioural basis** — a set of policies whose combination, via GPI, spans the policies needed for the test distribution. How to do this principled is left open.

#### 2.2.10 Strengths and limitations

**Strengths.**
- **Proposition 1** removes the linear-reward straitjacket: SF&GPI now has guarantees on *any* task in $\mathcal{M}$, with graceful misspecification degradation.
- **Rewards-as-features** eliminates the dual-bootstrap instability that made the original recipe hard to scale, and reduces the learning problem to standard Q-learning on actual rewards.
- **First scale demonstration**: end-to-end deep RL on 3-D first-person image input (DeepMind Lab) — SF&GPI is not just a tabular trick.
- **Compositional generalisation to negative rewards** despite only positive-reward training — strong evidence that GPI does non-trivial policy synthesis.
- **Algorithm 2** runs from scratch and uses GPI as the behaviour policy from step 1 — base policies "cooperate" during their own training, accelerating multi-task learning.

**Limitations.**
- **Discrete policy library.** Each new task either uses the existing library or extends it by 1 — no continuous parameterisation. USFA (Borsa 2018) addresses this with a learnable policy descriptor $\mathbf{z}$.
- **Dimensionality cost.** With rewards-as-features, $d = D$, losing the original flexibility of choosing $d < D$ for a more compact basis.
- **Behavioural-basis problem.** Spanning the reward space ≠ spanning the policy space. No principled mechanism for choosing base tasks to maximise behavioural coverage.
- **GPI argmax brittleness.** Same as 2017 — large approximation errors at any candidate's $\tilde{\boldsymbol{\psi}}^{\pi_i \top}\tilde{\mathbf{w}}'$ can mislead the argmax. Especially relevant in the continual setting where new SFs may be under-trained.

### 2.3 Appendix — Section-by-Section Backbone

#### Abstract
Two extensions to Barreto et al. 2017's SF&GPI: (i) the theoretical guarantees can be extended to **any** set of tasks differing only in reward (not just the linear-reward family), and (ii) one can use the reward functions themselves as features, eliminating the need to specify a feature set in advance — enabling stable combination with deep learning. Verified on a complex 3-D first-person environment.

#### 1. Introduction
Deep RL successes; ambition to compose skills. Recap of SF&GPI from Barreto et al. 2017. Two main contributions of this paper: relax the linear-reward assumption (Proposition 1) and use rewards-as-features for scalable online learning.

#### 2. Background
##### 2.1 Reinforcement learning
MDP notation, $Q^\pi$ definition, policy improvement, deep RL approximation conventions ($\tilde{\cdot}$, $\theta$).

##### 2.2 SF&GPI
Recap of Barreto et al. 2017: linear-reward assumption (2), $\mathcal{M}^{\boldsymbol{\phi}}$ family (3), SF decomposition (4), SF Bellman equation (5), GPI in Mφ family.

#### 3. Extending the notion of environment
$\mathcal{M}$ defined as all MDPs sharing dynamics (6); $\mathcal{M} \supset \mathcal{M}^{\boldsymbol{\phi}}$.

##### 3.1 Guarantees on the extended environment
Proposition 1 statement (8): for any $M \in \mathcal{M}$, the SF&GPI bound becomes $\tfrac{2}{1-\gamma}(\|r - r_i\|_\infty + \min_j \|r_i - r_j\|_\infty + \epsilon)$. Discussion: $\|r - r_i\|_\infty$ is the distance from $M$ to $\mathcal{M}^{\boldsymbol{\phi}}$; degrades gracefully.

##### 3.2 Uncovering the structure of the environment
Challenge: learning $\boldsymbol{\phi}$ online. 2017 recipe: multi-task regression $\tilde{\boldsymbol{\phi}}^\top \tilde{\mathbf{w}}_i \approx r_i$ (9). New observation: since $\mathbf{r} = \mathbf{W}\boldsymbol{\phi}$ and $\boldsymbol{\phi} = \mathbf{W}^\dagger \mathbf{r}$, **rewards and features have the same span**. Replace (9) with $\tilde{\boldsymbol{\phi}} \equiv \tilde{\mathbf{r}}$ (10). Consequence: SFs become collections of ordinary Q-values, $\tilde{\boldsymbol{\psi}}^{\pi_i} = \tilde{\mathbf{Q}}^{\pi_i}$.

#### 4. Transfer in deep RL
Online use of SF&GPI with deep nets.

##### Algorithm 1 (SF&GPI Q-learning, given $\tilde{\boldsymbol{\phi}}$, $\tilde{\boldsymbol{\Psi}}$).
GPI behaviour policy; supervised update of $\tilde{\mathbf{w}}$; optional `extend_basis` mode learns new SF for current task.

##### 4.1 Challenges of building features
Original recipe: $\tilde{\boldsymbol{\phi}}$ and $\tilde{\boldsymbol{\psi}}^{\pi_i}$ mutually depend on each other through data distribution and TD targets — circular instability.

##### 4.2 Learning features online while retaining transferable knowledge
With rewards-as-features, $\tilde{\boldsymbol{\psi}}^{\pi_i} = \tilde{\mathbf{Q}}^{\pi_i}$ learned by Q-learning on actual rewards $r_j$ (not $\tilde{r}_j$). Loose coupling: $\tilde{\mathbf{r}}$ learned by supervised regression in parallel.

##### Algorithm 2 (Build $\tilde{\boldsymbol{\phi}}$ and $\tilde{\boldsymbol{\Psi}}$ concurrently).
GPI used as behaviour policy from step 1; concurrent supervised regression of $\tilde{r}_t$ and Q-learning of $\tilde{Q}^{\pi_i}_t$ for all $i$ from a single transition.

#### 5. Experiments
##### 5.1 Environment
DeepMind Lab: 84×84 first-person images, 4 object types, 5 instances each. Reward = 4-vector specifying per-object reward. Base tasks $\hat{\mathcal{M}} = \{1000, 0100, 0010, 0001\}$.

##### 5.2 Agents
Architecture: CNN+LSTM encoder, $D+1$ MLP heads for $\tilde{\boldsymbol{\phi}}$ and $\tilde{\boldsymbol{\psi}}^{\pi_i}$, $\tilde{\mathbf{w}}$ head. Q($\lambda$) + IMPALA. SF&GPI-transfer (no basis extension), SF&GPI-continual (extends basis on test task). Baselines: Q($\lambda$) with frozen encoder, DQ($\lambda$) fine-tuned, DQ($\lambda$) from scratch.

##### 5.3 Results and discussion
SF&GPI-transfer is near-instant on test tasks; SF&GPI-continual best overall (Figure 3). Negative-reward generalisation. Linearly-dependent base tasks degrade gracefully (Figure 4). Behavioural-basis caveat: spanning reward space ≠ spanning policy space.

#### 6. Related work
Other deep-RL transfer approaches (Teh et al. 2017 "Distral", Heess et al., Frans et al., Oh et al.); hierarchical RL (Vezhnevets et al., Bacon et al.); UVFA (Schaul 2015); options.

#### 7. Conclusion
Two extensions: theoretical (any task in $\mathcal{M}$) and practical (rewards-as-features). DeepMind Lab demonstration. Open question: behavioural-basis design.

---

## 3. Borsa et al. 2018 — Universal Successor Features Approximators (brief TD-centric recap)

**PDF:** `docs/project/references/TD/sources/Borsa et al. 2018 - Universal Successor Features Approximators.pdf`
**Venue:** ICLR 2019 (arXiv 2018)
**Authors:** Diana Borsa, André Barreto, John Quan, Daniel Mankowitz, Rémi Munos, Hado van Hasselt, David Silver, Tom Schaul (DeepMind)

**Reading note.** Borsa et al. 2018 was already fully reviewed under the UVFA-context framing in `Gamma_lit_review_A_universal_vf.md`. This section gives a **brief TD-centric recap** that places USFA in the SF arc; for the full Phase 1 + Phase 2 deep dive (including the generalisation bound, training algorithm, and counter-intuitive findings on $\mathcal{C} = \mathcal{M} \cup \{\mathbf{w}'\}$), see Section 3.5 below for the explicit cross-reference.

### 3.1 The SF arc in three steps — SF → deep SF + GPI → universal SF

Reading the three papers in order, the conceptual arc is:

1. **SF (2017).** Linear-reward decomposition $Q^\pi = \boldsymbol{\psi}^{\pi \top}\mathbf{w}$. The agent stores **one $\boldsymbol{\psi}^{\pi^*_j}$ per training task**. GPI assembles the test-task policy from those.
2. **Deep SF + GPI (2019).** Same factorisation, but $\boldsymbol{\psi}^{\pi^*_j}$ now produced by a deep network; theoretical guarantees extended to any task (Proposition 1); rewards-as-features for stable online learning. The discrete library of SFs is still there — one $\boldsymbol{\psi}^{\pi^*_j}$ per base task, indexed by $j \in \{1, \dots, D\}$.
3. **Universal SF (Borsa 2018).** Replace the discrete library with a **single network conditioned on a continuous policy descriptor $\mathbf{z}$**:
   $$\boldsymbol{\psi}(s, a, \mathbf{z}) \;\approx\; \boldsymbol{\psi}^{\pi_{\mathbf{z}}}(s, a)$$
   where $\pi_{\mathbf{z}}$ is the optimal policy for the hypothetical task with reward weights $\mathbf{z}$. Now an entire *family* of policies is parameterised continuously, and GPI is over a **sampled** candidate set $\mathcal{C} \subset \mathbb{R}^d$ rather than a fixed library.

The key conceptual move from Barreto 2019 to Borsa 2018 is **separating the policy descriptor $\mathbf{z}$ from the task weights $\mathbf{w}$**:
- $\mathbf{w}$ tells you *which task* you are facing (the reward weights).
- $\mathbf{z}$ tells you *which policy* you are evaluating (the descriptor of an optimal policy for some hypothetical task).
- The agent can evaluate *any* policy on *any* task: $\tilde{Q}(s, a, \mathbf{w}, \mathbf{z}) = \tilde{\boldsymbol{\psi}}(s, a, \mathbf{z})^\top \mathbf{w}$.

### 3.2 The SF Bellman equation and the universal-SF generalisation

The 2017 SF Bellman equation is

$$\boldsymbol{\psi}^\pi(s, a) \;=\; \mathbb{E}\!\left[\boldsymbol{\phi}(s, a, s') + \gamma\, \boldsymbol{\psi}^\pi(s', \pi(s'))\right] \tag{4}$$

USFA inherits this and adds a $\mathbf{z}$ argument:

$$\boldsymbol{\psi}(s, a, \mathbf{z}) \;=\; \mathbb{E}\!\left[\boldsymbol{\phi}(s, a, s') + \gamma\, \boldsymbol{\psi}(s', \pi_{\mathbf{z}}(s'), \mathbf{z})\right]$$

where $\pi_{\mathbf{z}}(s') = \arg\max_b \boldsymbol{\psi}(s', b, \mathbf{z})^\top \mathbf{z}$ — i.e., $\pi_{\mathbf{z}}$ is the greedy policy of the network *evaluated at its own task* $\mathbf{z}$. The Bellman recursion is the same algorithmic backbone; the policy descriptor $\mathbf{z}$ travels with the recursion.

The crucial decomposition of the TD error from `Gamma_lit_review_A_universal_vf.md` carries over:

$$\delta^{t, n}_{\mathbf{w}, \mathbf{z}} \;=\; (\boldsymbol{\delta}^{t, n}_{\mathbf{z}})^\top \mathbf{w}$$

where the vector-valued $\boldsymbol{\delta}^{t, n}_{\mathbf{z}}$ depends only on $\mathbf{z}$ (and the data), not on $\mathbf{w}$. **The gradient update to the network is driven by $\boldsymbol{\delta}^{t, n}_{\mathbf{z}}$, not by the scalar $\delta^{t, n}_{\mathbf{w}, \mathbf{z}}$**. So training updates are *task-agnostic* (in the sense of $\mathbf{w}$) — they only depend on the policy descriptor $\mathbf{z}$. This is what lets USFA learn one network that serves *any* task on first-shot evaluation.

### 3.3 The GPI policy under USFA

At act time, given a test task $\mathbf{w}'$ and a chosen candidate set $\mathcal{C} \subset \mathbb{R}^d$:

$$\pi(s) \;\in\; \arg\max_a \max_{\mathbf{z} \in \mathcal{C}} \tilde{\boldsymbol{\psi}}(s, a, \mathbf{z})^\top \mathbf{w}' \tag{3}$$

The candidate set $\mathcal{C}$ is a test-time hyperparameter. Three canonical choices recover and span known methods:

- $\mathcal{C} = \{\mathbf{w}'\}$ — UVFA-style: ask the network at the policy descriptor *equal to* the test task.
- $\mathcal{C} = \mathcal{M}$ — pure SF&GPI: use only the training-task policies.
- $\mathcal{C} = \mathcal{M} \cup \{\mathbf{w}'\}$ or $\mathcal{C} \sim \mathcal{D}_{\mathbf{z}}(\cdot \mid \mathbf{w}')$ — a continuum of hybrids.

This is the direct generalisation of GPI from Theorem 1: max over *any* candidate set, not just the training-task set. The 2017 bound (Theorem 2) and the 2019 bound (Proposition 1) re-emerge as the special cases $\mathcal{C} = \mathcal{M}$.

### 3.4 Late conditioning — the architectural lesson

USFA's deep architecture **injects $\mathbf{z}$ late** — after the LSTM encoder produces $f(h_t) \in \mathbb{R}^{128}$, $\mathbf{z}$ is processed by a small MLP to $g(\mathbf{z}) \in \mathbb{R}^{32}$ and concatenated. The combined vector passes through one MLP to produce $\tilde{\boldsymbol{\psi}}(s, a, \mathbf{z})$ for every action.

**Why this matters.** If $\mathbf{z}$ were injected early (before the LSTM), every sampled $\mathbf{z}_i \in \mathcal{C}$ at training (typically $n_{\mathbf{z}} = 30$) or act time would require its own LSTM unroll — prohibitive. With late conditioning, the heavy convolutional + LSTM forward pass is **shared across all $\mathbf{z}_i$**, and only the small MLP and dot product run per $\mathbf{z}_i$.

This is the same architectural lesson that recurs in the project's FiLM and hypernet references: **conditioning is cheapest when injected late, at the smallest representational scale**. The project's modulator-conditioned architectures should follow this pattern — process the observation history once into a context-free embedding, then apply the modulator-dependent transformation in a small final head.

### 3.5 Cross-reference to the full review

For the full Phase 1 + Phase 2 review of Borsa 2018 with UVFA-context framing — including:
- The multi-task setup and linear-reward assumption (full derivation).
- Generalised policy improvement (Barreto 2017) restatement.
- Universal successor features definition and the policy-encoding $e(\pi_{\mathbf{z}}) = \mathbf{z}$.
- The GPI policy under USFA (eq. 3).
- Generalisation bound (Proposition 1 of Borsa 2018), with the $\delta_d(\mathbf{z})$ and $\delta_\psi(\mathbf{z})$ decomposition.
- Training a USFA — TD error decomposition (eq. 5), Algorithm 1 with sampling distribution $\mathcal{D}_{\mathbf{z}}$.
- Architecture details (late conditioning rationale).
- Empirical results on the Trip MDP and DM Lab.
- Counter-intuitive findings — $\mathcal{C} = \mathcal{M} \cup \{\mathbf{w}'\}$ sometimes worse than $\mathcal{C} = \mathcal{M}$.

see [Gamma_lit_review_A_universal_vf.md, Section 2](../Gamma/Gamma_lit_review_A_universal_vf.md#2-borsa-et-al-2018--universal-successor-features-approximators-usfa).

---

## 4. Cross-paper synthesis and project connections

### 4.1 The three structural moves

| Move | Paper | What it buys |
|---|---|---|
| **Linear-reward decomposition** $Q^\pi = \boldsymbol{\psi}^{\pi \top}\mathbf{w}$ | Barreto 2017 | Decouples dynamics-and-policy from reward-weights; enables one-shot evaluation on a new task |
| **GPI over a set of value functions** | Barreto 2017 (Theorem 1) | Provably-at-least-as-good policy combination; pointwise max across policies |
| **SF transfer bound via task-weight distance** | Barreto 2017 (Theorem 2) | Formalises "similar tasks have similar solutions" — performance gap bounded by $\min_j \|\mathbf{w}_i - \mathbf{w}_j\|$ |
| **Extended guarantees beyond $\mathcal{M}^{\boldsymbol{\phi}}$** | Barreto 2019 (Proposition 1) | Linear-reward assumption is no longer load-bearing — graceful degradation outside the span |
| **Rewards as features** $\boldsymbol{\phi} \equiv \mathbf{r}$ | Barreto 2019 | Eliminates $\boldsymbol{\phi}$-$\boldsymbol{\psi}$ bootstrap instability; SFs become collections of ordinary Q-values |
| **Policy-descriptor conditioning** $\boldsymbol{\psi}(s, a, \mathbf{z})$ | Borsa 2018 | Replaces the discrete SF library with a single network over a continuous policy descriptor; GPI over sampled $\mathcal{C}$ |

### 4.2 Connection to the project's modulator thread

The project is building neural networks whose value heads are reshaped by a **modulator vector $\mathbf{m}$** (a context vector representing serotonin level, pain weight, exploration drive). The structural mapping to SF is direct:

| SF&GPI concept | Project equivalent |
|---|---|
| $\boldsymbol{\phi}(s, a, s')$ | Per-step feature vector — possibly the actual rewards, possibly a learned feature |
| $\boldsymbol{\psi}^\pi(s, a)$ | Dynamics-and-policy summary — what kinds of outcomes does $\pi$ produce? |
| $\mathbf{w}$ (task weights) | $\mathbf{m}$ (modulator level) — *the agent's current preferences* |
| GPI over $\{\boldsymbol{\psi}^{\pi^*_j}\}$ | GPI over a library of modulator-specific policies |
| Universal SF $\boldsymbol{\psi}(s, a, \mathbf{z})$ | A FiLM/hypernet-conditioned value head — one network spanning the modulator space |
| Linear-reward assumption | Reward is linear in $\mathbf{m}$ — e.g., $r(s, a, s') = \boldsymbol{\phi}(s, a, s')^\top \mathbf{m}$ with $\mathbf{m}$ supplying the trade-off weights |
| Proposition 1's $\|r - r_i\|_\infty$ misspecification term | Graceful degradation when the modulator space does not exactly span the reward space |
| Late conditioning in USFA architecture | The same architectural lesson that motivates project's FiLM/hypernet choice — modulator goes in at the smallest layer |

**Two project-relevant implications.**

- **A modulator-conditioned agent that learns SFs is implicitly learning a USFA.** If the modulator vector $\mathbf{m}$ enters the value head via FiLM or hypernet, and the value head outputs a vector of per-feature returns rather than a scalar, then "evaluating policy $\pi_{\mathbf{m}}$ on a different modulator $\mathbf{m}'$" is just a dot product. The reward decomposition $r = \boldsymbol{\phi}^\top \mathbf{m}$ — which is exactly the kind of reward shaping the project uses for pain-vs-resource trade-offs — is the same algebraic structure Barreto uses.
- **GPI gives a principled way to blend modulator-specific policies when context shifts.** If the agent has trained policies for modulator settings $\mathbf{m}_1, \dots, \mathbf{m}_n$, and at test time finds itself in modulator setting $\mathbf{m}'$, it can compute $\boldsymbol{\psi}^{\pi^*_j}(s, a)^\top \mathbf{m}'$ for each $j$ and act greedy w.r.t. the max — *without* retraining. This is the structural answer to "which modulator-setting policy should the agent commit to in this state?" in a multi-modulator regime.

### 4.3 What's NOT solved by SF&GPI (relevant caveats for the project)

- **GPI is greedy over the candidate set.** It does not learn a smooth blend; it picks the argmax. For modulator-conditioned settings where the agent should *interpolate* between two known policies (e.g., 50% pain-avoidant, 50% resource-seeking), GPI's argmax may be too brittle — Borsa's Proposition 1 explicitly shows the failure mode when SF approximation error is large at one candidate.
- **The behavioural-basis problem (Barreto 2019, §5.3).** Spanning the reward space does *not* guarantee spanning the policy space. If the project trains modulator-specific policies at $\mathbf{m}_1, \dots, \mathbf{m}_n$ that all happen to be passive ("avoid everything"), GPI cannot synthesise active behaviour. Designing base modulators is implicitly designing a *behavioural basis* — this is non-trivial and open in the SF&GPI literature.
- **Discrete library scaling.** Barreto 2017/2019 store one $\boldsymbol{\psi}^{\pi^*_j}$ per training task — linear in $n$. USFA (Borsa 2018) collapses this to one network conditioned on $\mathbf{z}$, but introduces the policy-sampling distribution $\mathcal{D}_{\mathbf{z}}$ as a new hyperparameter. The project should default to the USFA route — single network conditioned on $\mathbf{m}$ — rather than building a discrete library.

### 4.4 Suggested follow-on reading

- **For the universal-SF + UVFA-context perspective** — Borsa 2018 full review at [`Gamma_lit_review_A_universal_vf.md`](../Gamma/Gamma_lit_review_A_universal_vf.md), Section 2.
- **For the UVFA precursor** — Schaul et al. 2015, also reviewed in `Gamma_lit_review_A_universal_vf.md`, Section 1.
- **For successor-representation foundations** — Dayan 1993 "Improving generalization for temporal difference learning: the successor representation" (Neural Computation 5(4):613–624); not in the current corpus but cited as the conceptual ancestor.
- **For deep SR architectures pre-GPI** — Kulkarni et al. 2016 "Deep Successor Reinforcement Learning" (arXiv:1606.02396); Zhang et al. 2016 "Deep RL with Successor Features for Navigation Across Similar Environments" (arXiv:1612.05533). Both predate the SF&GPI combination but introduce the deep-SR architecture.

---

## Notes on processing

- Backbone extracted from the two PDFs via `pdfplumber` (Barreto 2017: 24 pages, 73 K chars; Barreto 2019: 19 pages, 59 K chars), saved to `tmp/Barreto et al. 2017 ... .txt` and `tmp/Barreto et al. 2019 ... .txt` for traceability.
- Borsa 2018 was reviewed in `Gamma_lit_review_A_universal_vf.md` previously; the brief recap here covers the SF-arc-relevant material (SF Bellman, universal-SF generalisation, GPI policy, late-conditioning architecture). Cross-reference to the full review is in §3.5.
- Per-paper synthesis (Phase 1 / Phase 2) regroups material thematically rather than following the paper's section order; the section-by-section backbone is retained as an appendix for traceability.
