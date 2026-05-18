# Distributional Reinforcement Learning — Literature Review

**Corpus:** four papers tracing the development of distributional RL, from the original
categorical formulation through quantile-regression and implicit-quantile parameterizations
to risk-sensitive policies built on top of the learned return distribution.

**Reviewer:** `literature-reviewer` agent.
**Source PDFs:** `docs/project/references/TD/sources/`
**This file:** master review at the topic root.

---

## Top-of-Doc — What Is "Distributional RL" and Why Care?

In ordinary reinforcement learning, a state-action pair $(s,a)$ is associated with a
single scalar value $Q(s,a)$ — the **expected** discounted future reward the agent will
collect if it starts from $(s,a)$ and follows policy $\pi$ thereafter. But the actual
future reward an agent receives, the *return*

$$Z = \sum_{t=0}^{\infty} \gamma^t R_t,$$

is a **random variable**: it depends on the random reward at each step, the random
next-state transition, and the random next action. Two runs starting from the same
$(s,a)$ can give wildly different returns. The mean of those returns is $Q(s,a)$,
but the mean is only one number summarizing a whole distribution that may be
bimodal, heavy-tailed, or skewed.

**Distributional RL** says: don't collapse to the mean. Learn the entire distribution
of $Z$ as a function of $(s,a)$. Even if you ultimately act greedily on the mean of
that distribution (i.e., behave just like a standard RL agent at decision time),
**learning the full distribution as the prediction target makes the learning signal
richer, more stable, and empirically much more sample-efficient**. The C51 paper
(Bellemare, Dabney, Munos 2017) was the first to demonstrate this at scale on Atari,
producing what was then state-of-the-art results essentially "for free" by swapping
DQN's scalar prediction for a 51-bin distribution.

Once you have the distribution, a second door opens: **risk-sensitive behavior**. An
agent that knows the whole distribution can ask "what's my 5th-percentile outcome?"
and act so as to maximize *that* rather than the mean — i.e., it can be cautious in
the face of catastrophic downside. This is the bridge from distributional RL to the
project's modulator thread: a neuromodulator signaling "threat / high uncertainty"
could rationally re-weight the agent's value calculation away from the mean and
toward a lower quantile of the return distribution. The four papers reviewed below
build this machinery in two pieces — first the mathematical objects (Bellemare 2017
→ Dabney QR-DQN 2018 → Dabney IQN 2018), then the policy-side translation that
turns "I know the distribution" into "I act risk-aversely" (Lim & Malik 2022).

**Why this corpus exists for the project:** the modulator-conditioned agent already
uses FiLM (feature-wise linear modulation) to inject a neuromodulator signal into
the policy network. IQN's $\tau$-conditioning is, structurally, the very same FiLM
mechanism applied to a *risk-level scalar* $\tau \in [0,1]$. So this corpus is not
just adjacent reading — IQN is a literal architectural precedent for "FiLM-condition
the value head on a scalar that the agent interprets as risk tolerance," and Lim &
Malik supply the policy-side argument that connects such a scalar to behaviorally
meaningful risk-sensitivity. See the IQN section's Phase 2 for the explicit
cross-connection.

---

## Table of Contents

1. [Bellemare, Dabney, Munos (2017) — A Distributional Perspective on Reinforcement Learning (C51)](#1-bellemare-dabney-munos-2017--a-distributional-perspective-on-reinforcement-learning-c51)
2. [Dabney, Rowland, Bellemare, Munos (2018) — Distributional Reinforcement Learning with Quantile Regression (QR-DQN)](#2-dabney-rowland-bellemare-munos-2018--distributional-reinforcement-learning-with-quantile-regression-qr-dqn)
3. [Dabney, Ostrovski, Silver, Munos (2018) — Implicit Quantile Networks for Distributional Reinforcement Learning (IQN)](#3-dabney-ostrovski-silver-munos-2018--implicit-quantile-networks-for-distributional-reinforcement-learning-iqn)
4. [Lim and Malik (2022) — Distributional RL for Risk-Sensitive Policies](#4-lim-and-malik-2022--distributional-rl-for-risk-sensitive-policies)
    - [Phase 1 — Foundational Overview](#phase-1--foundational-overview-undergraduate-level-4)
    - [Phase 2 — Graduate-Level Deep Dive](#phase-2--graduate-level-deep-dive-4)
    - [Appendix: Section-by-Section Backbone (Lim & Malik 2022)](#appendix-section-by-section-backbone-lim--malik-2022)
5. [Cross-Paper Synthesis: C51 → QR-DQN → IQN → Lim & Malik](#5-cross-paper-synthesis-c51--qr-dqn--iqn--lim--malik)
    - [Representation evolution table](#representation-evolution-table)
    - [Contraction story across the four papers](#contraction-story-across-the-four-papers)
    - [The IQN–FiLM connection (load-bearing for the project)](#the-iqnfilm-connection-load-bearing-for-the-project)
    - [Project mapping: modulator as distortion / risk-level signal](#project-mapping-modulator-as-distortion--risk-level-signal)

---

## 1. Bellemare, Dabney, Munos (2017) — A Distributional Perspective on Reinforcement Learning (C51)

**PDF:** `docs/project/references/TD/sources/Bellemare et al. 2017 - A Distributional Perspective on Reinforcement Learning.pdf`
**Venue:** ICML 2017 (PMLR vol. 70).
**Key contribution:** founds modern distributional RL — characterizes the
*distributional Bellman operator* as a $\gamma$-contraction in a maximal Wasserstein
metric over value distributions (policy evaluation), exhibits non-contraction of the
distributional Bellman *optimality* operator (control), and proposes **C51**, a
DQN-style agent that represents the return as a categorical distribution over 51
fixed atoms and trains via a projected KL loss.

### Phase 1 — Foundational Overview (Undergraduate-Level)

**The core problem.** Standard Q-learning learns a single number $Q(s,a)$ — the
expected discounted return. But the return is genuinely random: sometimes the agent
gets lucky, sometimes it dies early, sometimes there are bimodal "you either escape
or die" outcomes that no scalar can faithfully summarize. The paper asks: what if we
learn the whole *distribution* of the return instead of just its mean?

**The core idea.** Define the value distribution $Z^\pi(s,a)$ to be the random
variable equal to the discounted sum of rewards collected by starting at $(s,a)$
and following $\pi$. The classical Bellman equation describes how $\mathbb{E}[Z]$
propagates between states; analogously, there is a *distributional* Bellman equation

$$Z^\pi(s,a) \stackrel{D}{=} R(s,a) + \gamma\, Z^\pi(S', A'),$$

where $\stackrel{D}{=}$ means "equal in distribution" (the random variables on the
two sides have the same probability law). This is the same recursion as before,
but operating on whole distributions, not on expectations.

**Why it works.** The paper proves two complementary results. (i) In *policy
evaluation* — fixed policy $\pi$ — the distributional Bellman operator $\mathcal{T}^\pi$
is a $\gamma$-contraction in the maximal Wasserstein metric, so iterating it
converges exponentially to the true value distribution $Z^\pi$ (just like the scalar
case). (ii) In *control* — the distributional optimality operator with greedy action
selection — the operator is **not** a contraction in any metric over distributions,
even though it is still a contraction in expected value. So distributional RL is
theoretically cleaner in evaluation than in control.

**The algorithm: C51.** To make this practical with a neural net, the authors
discretize: pick $N=51$ evenly spaced "atoms" $\{z_i\}_{i=0}^{50}$ between $V_{\min}=-10$
and $V_{\max}=10$, and predict a probability $p_i(s,a)$ for each. The Bellman backup
$\hat{\mathcal{T}}Z = r + \gamma Z(s', a^*)$ produces a distribution whose support is
shifted off the grid; **project** it back onto the 51 atoms by linear interpolation
and minimize the **KL divergence** between projected target and current prediction
(rather than the Wasserstein distance the theory motivates — Wasserstein cannot be
sample-minimized in general). This swap from "right loss" to "tractable loss" is a
known wart of C51 and the central motivation for QR-DQN (paper 2).

**Results.** On the 57-game Atari Learning Environment with the DQN architecture
swapped only at the output head (51 categorical outputs instead of scalar Q-values),
C51 achieves mean 1010% / median 178% of human baseline — a dramatic jump over DQN
(228% / 79%), Dueling DQN (373% / 151%), and Prioritized Replay (434% / 124%). C51
sets state-of-the-art on a number of games (notably SEAQUEST and the sparse-reward
VENTURE and PRIVATE EYE) with no other algorithmic changes.

**Initial takeaway.** Learning the value *distribution* — not just its mean — yields
a substantially better learning signal even when the policy acts on $\mathbb{E}[Z]$.
The paper interprets the gain as (a) reduced "chattering" from greedy non-contraction,
(b) automatic handling of state-aliased stochasticity, (c) a richer auxiliary-prediction
signal that couples accuracy with control, and (d) better-conditioned categorical
KL loss landscape. This single paper opened the field.

### Phase 2 — Graduate-Level Deep Dive

#### 2.1 The distributional Bellman operator

Let $(\Omega, \mathcal{F}, \mathrm{Pr})$ be a probability space and let $\mathcal{Z}$
be the space of value distributions with bounded moments. For a fixed policy $\pi$,
define the **transition operator** $P^\pi : \mathcal{Z} \to \mathcal{Z}$ by

$$P^\pi Z(s,a) \stackrel{D}{=} Z(S', A'), \quad S' \sim P(\cdot \mid s,a),\; A' \sim \pi(\cdot \mid S'),$$

and the **distributional Bellman operator** $\mathcal{T}^\pi : \mathcal{Z} \to \mathcal{Z}$
by

$$\mathcal{T}^\pi Z(s,a) \stackrel{D}{=} R(s,a) + \gamma\, P^\pi Z(s,a). \tag{C51.1}$$

Compare to the scalar Bellman operator $T^\pi Q(s,a) = \mathbb{E}[R] + \gamma \mathbb{E}_{P,\pi}[Q(s',a')]$:
the structure is identical, but expectations are removed and equalities are in
distribution. The three sources of randomness in $\mathcal{T}^\pi Z(s,a)$ are
(a) the reward $R$, (b) the transition $S' \sim P^\pi$, and (c) the next-state value
distribution $Z(S', A')$, all standardly assumed independent.

#### 2.2 The Wasserstein metric and the contraction proof

The **$p$-Wasserstein distance** between two real-valued distributions with c.d.f.s
$F$, $G$ is

$$d_p(F, G) = \inf_{U, V} \|U - V\|_p = \left( \int_0^1 |F^{-1}(u) - G^{-1}(u)|^p \, du \right)^{1/p}, \tag{C51.2}$$

where the infimum is over all couplings $(U,V)$ with marginals $F$, $G$, and the
second form follows from the inverse-c.d.f. (quantile-coupling) optimal transport
on $\mathbb{R}$. The key properties used in the contraction proof are

- (P1) **Scaling:** $d_p(aU, aV) \le |a|\, d_p(U, V)$.
- (P2) **Translation invariance:** $d_p(A + U,\, A + V) \le d_p(U, V)$ (no
  independence of $A$ from $U, V$ assumed).
- (P3) **Multiplicative coupling:** $d_p(AU, AV) \le \|A\|_p\, d_p(U, V)$.

Extend to vectors of distributions via the **supremal Wasserstein**

$$\bar{d}_p(Z_1, Z_2) := \sup_{s,a}\, d_p\big(Z_1(s,a),\, Z_2(s,a)\big). \tag{C51.3}$$

**Lemma 3 (Bellemare et al.):** $\mathcal{T}^\pi$ is a $\gamma$-contraction in
$\bar{d}_p$ — i.e., for all $Z_1, Z_2 \in \mathcal{Z}$,

$$\bar{d}_p(\mathcal{T}^\pi Z_1,\, \mathcal{T}^\pi Z_2) \le \gamma\, \bar{d}_p(Z_1, Z_2). \tag{C51.4}$$

**Proof sketch.** Pick any $(s,a)$. By (P2), the reward $R(s,a)$ on both sides
cancels in the Wasserstein bound:

$$d_p(\mathcal{T}^\pi Z_1(s,a),\, \mathcal{T}^\pi Z_2(s,a)) = d_p(R + \gamma P^\pi Z_1,\, R + \gamma P^\pi Z_2) \le d_p(\gamma P^\pi Z_1,\, \gamma P^\pi Z_2).$$

By (P1) (scaling by $\gamma$),

$$\le \gamma\, d_p(P^\pi Z_1(s,a),\, P^\pi Z_2(s,a)).$$

Then $P^\pi Z(s,a) \stackrel{D}{=} Z(S', A')$ where $(S', A')$ is sampled according
to the transition-and-policy kernel; using the partition lemma (Lemma 1, summing
over the partition $\{S' = s', A' = a'\}$ of $\Omega$),

$$d_p(P^\pi Z_1(s,a),\, P^\pi Z_2(s,a)) \le \sum_{s', a'} \pi(a' \mid s') P(s' \mid s,a) \, d_p(Z_1(s', a'),\, Z_2(s', a')) \le \sup_{s', a'} d_p(Z_1(s', a'), Z_2(s', a')) = \bar{d}_p(Z_1, Z_2).$$

Combining, $\bar{d}_p(\mathcal{T}^\pi Z_1, \mathcal{T}^\pi Z_2) \le \gamma\, \bar{d}_p(Z_1, Z_2)$.
$\square$

By Banach's fixed-point theorem, $\mathcal{T}^\pi$ has a unique fixed point — by
inspection, $Z^\pi$ itself — and iterating $Z_{k+1} = \mathcal{T}^\pi Z_k$ converges
to $Z^\pi$ at rate $\gamma$ in $\bar{d}_p$.

#### 2.3 Why not Wasserstein in practice? Cramér, KL, total variation

The paper explicitly notes (Chung & Sobel 1987, and later in this same paper) that
$\mathcal{T}^\pi$ is **not** a contraction in:

- **Total variation** $d_{\text{TV}}(P, Q) = \tfrac{1}{2}\|P - Q\|_1$
- **KL divergence** $D_{\text{KL}}(P \| Q) = \int P \log(P/Q)$
- **Kolmogorov distance** $\sup_x |F_P(x) - F_Q(x)|$
- **Cramér / $L^2$-c.d.f.** $\int (F_P - F_Q)^2$

So Wasserstein is *special* — it's the right metric for the theory. But the
algorithm uses KL, because KL admits unbiased sample estimates and stable
neural-net training while sample-Wasserstein cannot be minimized via SGD with
unbiased gradients (Bellemare et al. 2017, "Cramér distance" follow-up paper). This
sample-Wasserstein-minimization gap is exactly what QR-DQN (paper 2) fixes — using
quantile regression to minimize Wasserstein-1 *without* projecting.

#### 2.4 Non-contraction of the distributional optimality operator

For control, define the **distributional Bellman optimality operator** $\mathcal{T}$
as any operator $\mathcal{T}Z = \mathcal{T}^\pi Z$ for some greedy $\pi \in \mathcal{G}_Z$,
where $\mathcal{G}_Z := \{\pi : \sum_a \pi(a \mid s)\, \mathbb{E} Z(s,a) = \max_{a'} \mathbb{E} Z(s,a')\}$.

**Proposition 1.** $\mathcal{T}$ is not a contraction.

**Counterexample (Figure 2 of the paper).** Two-state MDP with deterministic
transition $x_1 \to x_2$; at $x_2$ action $a_1$ yields no reward, $a_2$ yields
$\pm 1 + \epsilon$ with equal probability; both terminal. Unique optimal policy
selects $a_2$ at $x_2$. Construct an off-optimal $Z$ with $Z(x_2, a_1) = 0$,
$Z(x_2, a_2) = \pm 1$ (no $\epsilon$); then $\bar{d}_1(Z, Z^*) = 2\epsilon$ but
$\mathcal{T}Z$ greedily switches to $a_1$ at $x_1$ (since both expectations equal $0$
in the tie-break sense the example sets up), producing $\bar{d}_1(\mathcal{T}Z, Z^*) > 2\epsilon$
for small $\epsilon$. So $\bar{d}_1(\mathcal{T}Z, \mathcal{T}Z^*) > \bar{d}_1(Z, Z^*)$
— not even a non-expansion, let alone a contraction.

**Lemma 4** still saves us in expectation: $\|\mathbb{E}[\mathcal{T} Z_1] - \mathbb{E}[\mathcal{T} Z_2]\|_\infty \le \gamma\, \|\mathbb{E} Z_1 - \mathbb{E} Z_2\|_\infty$,
so the *mean* still converges to $Q^*$ exponentially. The *distribution* converges
only pointwise to the larger set of *nonstationary* optimal value distributions
(Definition 3, Theorem 1).

**Operationally:** in control, distributional RL retains scalar convergence but loses
distributional contraction. In practice this manifests as "chattering" of the
distribution shape near the optimal policy. The paper conjectures this is the
mechanism that's making approximate RL unstable, and that explicitly modelling the
distribution dampens the instability.

#### 2.5 The C51 algorithm — categorical parameterization and projected KL

**Parameterization.** Fix $V_{\min}, V_{\max}$ and $N$ atoms

$$z_i = V_{\min} + i\, \Delta z, \quad \Delta z = \frac{V_{\max} - V_{\min}}{N - 1}, \quad i = 0, \ldots, N-1.$$

The categorical value distribution is

$$Z_\theta(s,a) = z_i \text{ w.p. } p_i(s,a) := \frac{e^{\theta_i(s,a)}}{\sum_j e^{\theta_j(s,a)}}, \tag{C51.5}$$

i.e., the network outputs logits $\theta(s,a) \in \mathbb{R}^N$ that softmax into
atom probabilities.

**The support-misalignment problem.** A Bellman backup $\hat{\mathcal{T}}Z(s,a) = r + \gamma z_j$
generically produces atoms at locations $\hat{\mathcal{T}} z_j = r + \gamma z_j$ that
are **not** on the grid $\{z_i\}$. So the target distribution and the parameterized
distribution have **disjoint supports** in general, and KL divergence between them
is infinite.

**The Cramér / categorical projection $\Phi$.** Project the off-grid target onto the
on-grid support by linear interpolation: for each $j$, the mass $p_j(s', \pi(s'))$
is split between the two grid atoms $z_{l_j}$ and $z_{u_j}$ that bracket
$\hat{\mathcal{T}} z_j$, weighted by distance. The $i$-th component of the projected
update is

$$(\Phi \hat{\mathcal{T}} Z_\theta(s,a))_i = \sum_{j=0}^{N-1} \left[ 1 - \frac{|[\hat{\mathcal{T}} z_j]_{V_{\min}}^{V_{\max}} - z_i|}{\Delta z} \right]_0^1 p_j(s', \pi(s')), \tag{C51.6}$$

where $[\cdot]_a^b$ clips to $[a,b]$ and $[\hat{\mathcal{T}}z_j]_{V_{\min}}^{V_{\max}}$
clips the backup target to the supported range. This is the C51 algorithm's
**Algorithm 1**, computable in $O(N)$.

**Loss.** Cross-entropy between projected target (treated as a constant target,
parameterized by the frozen target network $\tilde\theta$) and current logits:

$$\mathcal{L}_{s,a}(\theta) = D_{\text{KL}}\!\left( \Phi \hat{\mathcal{T}} Z_{\tilde\theta}(s,a) \,\Big\|\, Z_\theta(s,a) \right) = -\sum_i m_i \log p_i(s,a), \tag{C51.7}$$

where $m_i = (\Phi \hat{\mathcal{T}} Z_{\tilde\theta}(s,a))_i$ is the projected target
probability for atom $i$. Greedy action selection uses the predicted expectation
$Q(s,a) = \sum_i z_i\, p_i(s,a)$.

**Projection bias.** The projection $\Phi$ is a **biased** approximation in the
Wasserstein sense — projecting a delta at $z_j + \epsilon$ onto the grid
$\{z_j, z_{j+1}\}$ does not preserve Wasserstein distance, and the projection
operator $\Phi \mathcal{T}^\pi$ is no longer Wasserstein-contracting in general
(it is, however, contracting in the Cramér distance under appropriate support
choice; see the "Cramér distance" follow-up). This bias is the second
known wart of C51, addressed cleanly by QR-DQN's quantile representation.

#### 2.6 Empirical results — Atari 57

- Per-game ablation over $N \in \{5, 11, 21, 51\}$ shows monotone improvement;
  $N = 51$ is the sweet spot.
- C51 (5-game training set, $\epsilon = 0.05$ behavioral) substantially exceeds DQN
  on all five training games and sets SOTA on SEAQUEST.
- Full 57-game eval ($\epsilon = 0.01$): mean 1010% / median 178% of human baseline
  vs. DQN 228% / 79%, Dueling 373% / 151%, Prioritized 434% / 124%.
- C51 outperforms a fully-trained DQN on **45 of 57** games within 50M frames.
- Sparse-reward games (VENTURE, PRIVATE EYE) show particularly large improvements,
  suggesting distributional learning propagates rarely-occurring events better than
  scalar regression.

#### 2.7 Why does learning a distribution help even with mean-greedy policies?

Four mechanisms proposed in §6.1:

1. **Reduced chattering.** Greedy non-contraction (Prop. 1) destabilizes the policy
   under function approximation. The gradient-based categorical algorithm
   "averages" the chattering distributions, mimicking conservative policy iteration.
2. **State aliasing → effective stochasticity.** Even in deterministic environments,
   partial observability induces stochastic returns; explicitly modelling that
   distribution gives a stable target. Cf. the PONG reward-timing example.
3. **Auxiliary prediction richness.** $p_i(s,a)$ is a per-bin prediction whose
   accuracy is tightly coupled to control — a Caruana-style auxiliary task that
   actually helps the main task.
4. **Inductive-bias framework.** Hyperparameters $V_{\min}, V_{\max}$ are an
   inductive bias on return scale. Treating returns outside $[V_{\min}, V_{\max}]$
   as equivalent (clipping) is a natural reformulation that simple value clipping
   in DQN gets wrong (DQN value-clipping degrades performance; C51 support-clipping
   improves it).

### Appendix: Section-by-Section Backbone (C51)

**Abstract.** Argues the central role of the value distribution; proves contraction
of the policy-evaluation distributional Bellman operator and instability of the
optimality operator; introduces categorical-distribution DQN (C51); reports SOTA
on Atari 57.

**§1 Introduction.** Standard RL targets $Q(s,a) = \mathbb{E} Z(s,a)$; the paper
argues for modelling $Z$ itself via the distributional Bellman equation
$Z(s,a) \stackrel{D}{=} R(s,a) + \gamma Z(S', A')$. Lists four contributions:
contraction in Wasserstein for $\mathcal{T}^\pi$, instability for $\mathcal{T}$,
the categorical (C51) algorithm, and empirical state-of-the-art.

**§2 Setting.** Standard MDP $(\mathcal{X}, \mathcal{A}, R, P, \gamma)$ with random
reward. Scalar Bellman operators (eqs. 2-3) reviewed for contrast.

**§3 Distributional Bellman operators.** §3.1 introduces distributional equations
and the partition lemma. §3.2 defines the Wasserstein metric and its supremal
extension $\bar{d}_p$. §3.3 proves Lemma 3 (contraction in $\bar{d}_p$ for $\mathcal{T}^\pi$)
and notes non-contraction in TV, KL, Kolmogorov, plus contraction in centered moments.
§3.4 covers control: defines optimal value distributions $\mathcal{Z}^*$, proves
Prop. 1 (non-contraction), Prop. 2 (no fixed point in general), Prop. 3 (fixed-point
existence insufficient for convergence), and Thm. 1 (pointwise convergence to the
set of nonstationary optimal value distributions $\mathcal{Z}^{**}$).

**§4 Approximate distributional learning.** §4.1 chooses the parametric family —
discrete with $N$ atoms on $[V_{\min}, V_{\max}]$ — for expressiveness and
computational friendliness. §4.2 introduces the categorical projection $\Phi$
(eq. 7, Algorithm 1) and the KL-cross-entropy loss; notes that Wasserstein cannot
be sample-minimized.

**§5 Evaluation on Atari 2600 games.** §5.1 ablates $N$ on five training games
(BREAKOUT, PONG, Q*BERT, SEAQUEST, SPACE INVADERS); $N = 51$ wins. §5.2 full-suite
results: mean 1010% / median 178% of human baseline; C51 beats DQN on 45/57 games.

**§6 Discussion.** §6.1 lists four mechanisms by which distributional learning
helps mean-greedy policies: reduced chattering, state-aliasing absorption,
auxiliary-prediction richness, inductive-bias framework, well-behaved KL
optimization. Closes by noting that a closer Wasserstein-minimizing approximation
"should yield even better results" — explicit foreshadowing of QR-DQN.

**References.** Includes the parallel-submitted "Cramér distance as a solution to
biased Wasserstein gradients" (Bellemare et al. 2017 arXiv).

---

## 2. Dabney, Rowland, Bellemare, Munos (2018) — Distributional Reinforcement Learning with Quantile Regression (QR-DQN)

**PDF:** `docs/project/references/TD/sources/Dabney et al. 2018 - Distributional reinforcement learning with quantile regression.pdf`
**Venue:** AAAI 2018.
**Key contribution:** closes the theory-practice gap left open by C51. By **transposing**
the parameterization — fixed *probabilities* $1/N$ on *learned* atom locations
$\{\theta_i(s,a)\}$ instead of learned probabilities on fixed locations — and using
**quantile regression** to fit those locations, the algorithm minimizes the
Wasserstein-1 distance to the Bellman target *end-to-end with unbiased stochastic
gradients*. The combined projection-plus-Bellman operator is proved to be a
$\gamma$-contraction in the $\infty$-Wasserstein metric. Empirically QR-DQN beats
C51 on Atari-57 (mean 915%, median 211% with Huber-$\kappa=1$, vs. C51's 701% / 178%).

### Phase 1 — Foundational Overview (Undergraduate-Level)

**The gap C51 left.** C51 *motivated* its design via the Wasserstein contraction
theorem (paper 1, Lemma 3), but the actual loss it minimizes is the KL divergence
between a categorically *projected* target and the current categorical prediction.
Two consequences: (i) KL is not the metric the theory says is contracting — KL is
not even a true probability metric, and the projection $\Phi$ destroys the
Wasserstein-contraction property; (ii) C51 needs the user to pre-specify
$[V_{\min}, V_{\max}]$ as a domain hyperparameter. Distributional RL was working,
but it was working "for reasons orthogonal to the theory."

**The transpose.** QR-DQN flips the C51 parameterization on its head. Instead of
$N$ **fixed atom locations** $\{z_i\}$ with $N$ **learned probabilities**
$\{p_i(s,a)\}$, use $N$ **fixed cumulative probabilities** $\{\tau_i = i/N\}_{i=1}^N$
with $N$ **learned atom locations** $\{\theta_i(s,a)\}$. Each $\theta_i(s,a)$ is now
an estimate of the $\hat\tau_i := (\tau_{i-1} + \tau_i)/2 = (2i-1)/(2N)$ quantile of
the return distribution. The parameterized distribution is the uniform mixture of
$N$ Dirac deltas at those learned locations:

$$Z_\theta(s,a) = \frac{1}{N} \sum_{i=1}^{N} \delta_{\theta_i(s,a)}.$$

**Why this fixes everything.** Because the locations are free, support never has
to be projected onto a grid — there are no disjoint-support issues, $[V_{\min},
V_{\max}]$ disappears, and the family is automatically adapted to whatever range
of returns the environment produces. And because quantiles can be learned from
single samples via the **quantile regression loss**

$$\rho_\tau(u) = u(\tau - \mathbb{1}\{u < 0\}),$$

we get an *unbiased* stochastic gradient for the Wasserstein-1 distance — the very
thing C51 could not have. The proof: the unique minimizer of
$\mathbb{E}_{\hat Z \sim Z}[\rho_\tau(\hat Z - \theta)]$ in $\theta$ is the $\tau$-quantile
of $Z$, $F_Z^{-1}(\tau)$ (a classical result going back to Koenker & Hallock 2001).

**The algorithm: QR-DQN.** Three minimal changes to DQN. (i) Output head has size
$|\mathcal{A}| \times N$ instead of $|\mathcal{A}|$ — each action gets $N$ quantile
estimates. (ii) Replace the DQN Huber loss on TD error with a **quantile Huber loss**
$\rho_{\hat\tau_i}^\kappa$ summed over all $(i,j)$ pairs of quantile indices. (iii)
Adam instead of RMSProp. Action selection: $a^* = \arg\max_a \frac{1}{N} \sum_i \theta_i(s,a)$,
i.e., greedy on the empirical mean of the predicted quantile distribution.

**Results.** Atari-57 best-agent: QR-DQN-1 (Huber $\kappa=1$) reaches 915% mean /
211% median human-normalized score, vs. C51 701%/178%, Prioritized Dueling 592%/172%,
Dueling 373%/151%, DQN 228%/79%. QR-DQN-1 surpasses human on 41 games and beats DQN
on 54 of 57. On a 2-room windy gridworld, QRTD (the policy-evaluation variant) is
shown to converge to the true value distribution under the Wasserstein-1 metric,
confirming the theory.

**Initial takeaway.** QR-DQN is the "clean" distributional RL algorithm — minimizes
the right metric, has no support hyperparameter, beats C51. Its quantile
representation is also the right substrate for what comes next (IQN): once the
function $i \mapsto \theta_i(s,a)$ is a discrete table, the natural generalization
is to make it a continuous function of $\tau$.

### Phase 2 — Graduate-Level Deep Dive

#### 2.1 The quantile parameterization

Let $\mathcal{Z}_Q$ be the space of **quantile distributions** for fixed $N$ — value
distributions of the form

$$Z_\theta(s,a) = \frac{1}{N} \sum_{i=1}^{N} \delta_{\theta_i(s,a)}, \quad \theta : \mathcal{X} \times \mathcal{A} \to \mathbb{R}^N. \tag{QR.1}$$

The cumulative probabilities are $\tau_i = i/N$ for $i = 0, \ldots, N$ (with $\tau_0 = 0$),
and the **quantile midpoints** are $\hat\tau_i := (\tau_{i-1} + \tau_i)/2 = (2i-1)/(2N)$.
By Lemma 2 (Dabney et al. 2018) — proved by setting the subgradient of
$\int_\tau^{\tau'} |F^{-1}(\omega) - \theta|\, d\omega$ to zero — the value
$\theta_i^* = F_Z^{-1}(\hat\tau_i)$ minimizes the 1-Wasserstein distance from
$Z$ to a single Dirac on the interval $[\tau_{i-1}, \tau_i]$ of the c.d.f. So the
optimal $\theta^*$ for QR-DQN places each atom at the $\hat\tau_i$-quantile of the
true return distribution. This gives the Wasserstein-1-minimizing **quantile
projection**

$$\Pi_{W_1} Z := \arg\min_{Z_\theta \in \mathcal{Z}_Q} W_1(Z, Z_\theta), \quad (\Pi_{W_1} Z)_i = F_Z^{-1}(\hat\tau_i). \tag{QR.2}$$

#### 2.2 The quantile regression loss and its unbiased sample gradient

The **quantile regression loss** for quantile level $\tau \in [0,1]$ is the
asymmetric "pinball" loss

$$\rho_\tau(u) = u\,(\tau - \mathbb{1}\{u < 0\}) = \begin{cases} \tau\, u & u \ge 0 \\ -(1 - \tau)\, u & u < 0. \end{cases} \tag{QR.3}$$

For a random variable $\hat Z \sim Z$ and a candidate quantile estimate $\theta$,
the expected loss is

$$\mathcal{L}_{\text{QR}}^\tau(\theta) = \mathbb{E}_{\hat Z \sim Z}\big[\rho_\tau(\hat Z - \theta)\big].$$

**Theorem (Koenker 2005).** $\theta^* = \arg\min_\theta \mathcal{L}_{\text{QR}}^\tau(\theta) = F_Z^{-1}(\tau)$
whenever $F_Z^{-1}$ is continuous at $\tau$.

**Proof sketch.** $\rho_\tau(u)$ is convex in $u$ with subgradient $\tau$ for $u > 0$,
$-(1-\tau) = \tau - 1$ for $u < 0$, and $[\tau - 1, \tau]$ at $u = 0$. So

$$\frac{\partial}{\partial \theta} \mathbb{E}[\rho_\tau(\hat Z - \theta)] = \mathbb{E}\big[\tau \cdot \mathbb{1}\{\hat Z > \theta\} - (1-\tau)\cdot \mathbb{1}\{\hat Z < \theta\}\big] = \tau - F_Z(\theta) - \tau\cdot \mathbb{1}\{\hat Z = \theta\}_{\text{neglig.}}.$$

Setting equal to zero: $F_Z(\theta^*) = \tau$, i.e., $\theta^* = F_Z^{-1}(\tau)$. $\square$

**Crucial property — unbiased sample gradient.** A single sample $\hat z \sim Z$
gives

$$\nabla_\theta \rho_\tau(\hat z - \theta) = -(\tau - \mathbb{1}\{\hat z < \theta\}),$$

with expectation $-(\tau - F_Z(\theta))$, exactly matching the population gradient.
This is the property that Wasserstein-minimization-via-EMD-loss does **not** have
(Proposition 1 in Dabney et al. 2018; Theorem 1 in Bellemare et al. 2017's "Cramér"
paper). The QR loss converts a Wasserstein-flavored objective into one with
sample-unbiased gradients — that's the whole trick.

#### 2.3 The quantile Huber loss

For nonlinear function approximation, the kink of $\rho_\tau$ at $u = 0$ can
destabilize training. The **Huber loss** (Huber 1964) is the standard smoothed L1:

$$\mathcal{L}_\kappa(u) = \begin{cases} \tfrac{1}{2} u^2 & |u| \le \kappa, \\ \kappa\, (|u| - \tfrac{1}{2}\kappa) & \text{otherwise.} \end{cases} \tag{QR.4}$$

The **quantile Huber loss** is the asymmetric version,

$$\rho_\tau^\kappa(u) = |\tau - \mathbb{1}\{u < 0\}| \cdot \frac{\mathcal{L}_\kappa(u)}{\kappa}, \tag{QR.5}$$

which reduces to $\rho_\tau$ as $\kappa \to 0^+$. QR-DQN-$\kappa$ uses
$\kappa \in \{0, 1\}$; the paper reports best performance at $\kappa = 1$ (915%/211%)
vs. $\kappa = 0$ (881%/199%).

#### 2.4 Contraction of the combined projection-Bellman operator

The non-trivial theoretical result of the paper:

**Proposition 2.** Let $\Pi_{W_1}$ be the quantile projection onto $\mathcal{Z}_Q$
(applied state-wise). For any $Z_1, Z_2 \in \mathcal{Z}$ over an MDP with countable
state-action spaces,

$$\bar d_\infty(\Pi_{W_1} \mathcal{T}^\pi Z_1,\, \Pi_{W_1} \mathcal{T}^\pi Z_2) \le \gamma\, \bar d_\infty(Z_1, Z_2). \tag{QR.6}$$

So the combined operator $\Pi_{W_1} \mathcal{T}^\pi$ is a $\gamma$-contraction in
$\infty$-Wasserstein, has a unique fixed point $\hat Z^\pi$, and the stochastic
approximation (QRTD) converges to that fixed point under standard step-size
conditions.

**Note on the $\infty$ in the metric.** Interestingly, the contraction holds in
$d_\infty$ (largest gap between c.d.f.s) but **not** in $d_p$ for finite $p$. The
intuition: $d_\infty$ acts like a worst-case quantile gap, which is exactly what
the projection-then-backup geometry preserves. Convergence in $d_p$ for $p < \infty$
follows from $d_p \le d_\infty$ but is not a separate contraction guarantee.

**Proof sketch (Lemma 3 + 4 in the appendix).** The proof first reduces to the
case $\gamma = 1$, $r \equiv 0$ (since $\mathcal{T}^\pi$ is already $\gamma$-contracting
in $d_\infty$ by C51 Lemma 3, and Wasserstein is translation-invariant in support).
Then it reduces value distributions to Diracs by "unrolling" the partition of next
states into individual Dirac sub-distributions (Figure 5 in the paper). The key
combinatorial argument: for each quantile level $\tau$, the $\tau$-quantile of
$\Pi_\tau \mathcal{T}^\pi Z$ at $(s',a')$ equals some specific $\theta_u$, and the
$\tau$-quantile of $\Pi_\tau \mathcal{T}^\pi Y$ equals some $\psi_v$, both drawn
from the partition over accessible $(x_i, a_i)$. An index-set partition argument
(I_{\le \theta_u}, I_{> \theta_u}, I_{< \psi_v}, I_{\ge \psi_v}) shows
$|\theta_u - \psi_v| \le \max_{i \in I} |\theta_i - \psi_i| = \sup_{(s', a')} d_\infty(Z(s', a'), Y(s', a'))$.

#### 2.5 The QRTD / QR-DQN algorithms

**Quantile regression TD (QRTD), policy evaluation.** For a sampled transition
$(s, a, r, s')$ with $a \sim \pi(\cdot | s)$, $z' \sim Z_\theta(s')$:

$$\theta_i(s) \leftarrow \theta_i(s) + \alpha\big(\hat\tau_i - \mathbb{1}\{r + \gamma z' < \theta_i(s)\}\big). \tag{QR.7}$$

This is precisely the quantile-regression step (eq. QR.3 subgradient) on the TD
target $r + \gamma z'$. Multiple samples $z'$ can be averaged (the paper recommends
summing over all $j = 1, \ldots, N$ next-state quantiles for the update).

**QR-DQN, control.** Algorithm 1 of the paper:

1. Compute next-state expected values $Q(s', a') = \frac{1}{N} \sum_j \theta_j(s', a')$.
2. Greedy next action: $a^* = \arg\max_{a'} Q(s', a')$ (note: greedy on the **mean**
   of the quantile distribution, exactly like DQN; the distributional structure is
   only in the *prediction target*, not in the action rule).
3. Distributional Bellman target: $\mathcal{T}\theta_j = r + \gamma\, \theta_j(s', a^*)$
   for each $j$.
4. Quantile-Huber loss:

$$\mathcal{L}(s, a, r, s'; \theta) = \sum_{i=1}^{N} \mathbb{E}_j\!\left[ \rho_{\hat\tau_i}^\kappa\!\big(\mathcal{T}\theta_j - \theta_i(s, a)\big) \right] = \frac{1}{N} \sum_{i=1}^{N} \sum_{j=1}^{N} \rho_{\hat\tau_i}^\kappa\!\big(\mathcal{T}\theta_j - \theta_i(s, a)\big). \tag{QR.8}$$

Each prediction $\theta_i(s,a)$ is fit to be the $\hat\tau_i$-quantile of the
distribution $\{ \mathcal{T}\theta_j \}_{j=1}^{N}$, weighted by the $\hat\tau_i$
pinball weights. Backpropagate through $\theta$, use a target network for the
$\mathcal{T}\theta_j$ side, and that is the whole algorithm.

#### 2.6 Empirical results — Atari 57

- **Best agent ($\epsilon = 0.001$ at test time):**
  | Algorithm | Mean | Median | # Games > human | # Games > DQN |
  |---|---:|---:|---:|---:|
  | DQN | 228% | 79% | 24 | 0 |
  | DDQN | 307% | 118% | 33 | 43 |
  | Dueling | 373% | 151% | 37 | 50 |
  | Prioritized | 434% | 124% | 39 | 48 |
  | Prior. Dueling | 592% | 172% | 39 | 44 |
  | C51 | 701% | 178% | 40 | 50 |
  | QR-DQN-0 | 881% | 199% | 38 | 52 |
  | **QR-DQN-1** | **915%** | **211%** | **41** | **54** |

- **Hyperparameters:** $\alpha = 5\times 10^{-5}$, Adam $\epsilon = 0.01/32$,
  $N = 200$ quantiles, $\kappa = 1$ Huber.
- **Online performance** (training without early-stopping): QRTD provides
  prioritization-like sample-complexity gains *and* better final performance.
- Note: the Atari Learning Environment's stochastic execution mode is used to test
  robustness — QR-DQN remains state-of-the-art.

#### 2.7 The Cramér-Wold theorem connection

A quick comment relevant to why "fit quantiles, get distribution" works in any
dimension. The **Cramér-Wold theorem** says that the distribution of a random
vector $\mathbf{Z}$ in $\mathbb{R}^d$ is uniquely determined by the distributions
of all its 1-D projections $\langle \mathbf{u}, \mathbf{Z}\rangle$ for $\mathbf{u}$
on the unit sphere. In QR-DQN we are in 1-D from the start (the return is scalar),
so the relevant statement is the simpler "$F_Z^{-1}$ determines $Z$": specifying
the full quantile function at all $\tau \in [0,1]$ is equivalent to specifying the
distribution. QR-DQN truncates this to $N$ quantile levels, IQN (paper 3) restores
the full continuum.

### Appendix: Section-by-Section Backbone (QR-DQN)

**Abstract.** Closes the theory-practice gap in distributional RL by combining the
Wasserstein-contraction theory of C51 with quantile regression for unbiased
stochastic-gradient training. New algorithm QR-DQN sets SOTA on Atari-57.

**§1 Introduction.** Reviews C51 and the open question of a sample-Wasserstein-minimizing
distributional RL algorithm. Lists three contributions: transpose the C51
parameterization (fixed probabilities, variable locations), use quantile regression
for unbiased gradients, prove contraction.

**§2 Distributional RL.** Recaps MDP setting, value distribution, the distributional
Bellman operator, the Wasserstein metric, the max-Wasserstein $\bar d_p$, Lemma 1
(C51's $\gamma$-contraction in $\bar d_p$), Theorem 1 (Bellemare et al.'s
"biased-Wasserstein-gradients" negative result), motivating the quantile approach.

**§3 Approximately minimizing Wasserstein.** The transpose: quantile distribution
$\mathcal{Z}_Q$ with $N$ Diracs at uniform probabilities, learned locations
$\theta_i$. Quantile projection $\Pi_{W_1}$. Lemma 2 (the midpoint $F^{-1}((\tau + \tau')/2)$
is the W1-minimizer). Quantile regression loss $\rho_\tau$ and the unbiased-gradient
property. Quantile Huber loss $\rho_\tau^\kappa$. Proposition 2 (combined-operator
$\gamma$-contraction in $d_\infty$).

**§4 Distributional RL using quantile regression.** Algorithm 1 — QRTD update for
policy evaluation; QR-DQN for control with $|\mathcal{A}| \times N$ output head,
quantile-Huber loss, Adam.

**§5 Experimental results.** §5.1 windy-gridworld policy-evaluation experiment
showing QRTD converges in $W_1$ to the true distribution while TD(0) converges
only in mean. §5.2 Atari-57 best-agent and online evaluations; QR-DQN-1 wins.

**§6 Conclusions.** Bridges the Wasserstein theory-practice gap. Highlights the
"richer policy class" enabled by action-value distributions, including
risk-sensitive policies — explicit hand-off to IQN and Lim & Malik.

**Appendix.** Proofs of Lemma 2, Proposition 1, Proposition 2 (with the
Dirac-reduction Lemma 3 and the quantile-projection Lemma 4).

---

## 3. Dabney, Ostrovski, Silver, Munos (2018) — Implicit Quantile Networks for Distributional Reinforcement Learning (IQN)

**PDF:** `docs/project/references/TD/sources/Dabney et al. 2018 - Implicit quantile networks for distributional reinforcement learning.pdf`
**Venue:** ICML 2018.
**Key contribution:** generalizes QR-DQN from a fixed grid of $N$ quantiles to a
**continuous implicit quantile function** $Z_\tau(s,a) = F_{Z(s,a)}^{-1}(\tau)$
parameterized by $\tau \in [0,1]$. The agent samples $\tau \sim U([0,1])$,
embeds it via a cosine basis, fuses with the convolutional state representation
via a **Hadamard product** (a FiLM-style multiplicative modulation), and outputs
the corresponding quantile of the return. This (i) removes $N$ as a representational
hyperparameter (network capacity controls fidelity, not output dimensionality),
(ii) supports any number of samples per update for data-efficiency tuning, and
(iii) immediately gives access to **distortion risk measures** $\beta(\tau)$ — any
risk-sensitive policy can be implemented by changing the sampling distribution of
$\tau$ from uniform to $\beta$. Atari-57: IQN 1019% mean / 218% median, halving the
gap from QR-DQN to Rainbow.

### Phase 1 — Foundational Overview (Undergraduate-Level)

**The leap from QR-DQN.** QR-DQN's $\theta_i(s, a)$ is a discrete lookup table:
"give me the return value at quantile level $\hat\tau_i = (2i-1)/(2N)$." IQN
replaces that table with a **continuous function** $Z(s, a; \tau)$ of an explicit
quantile-level input $\tau \in [0,1]$. At training time, IQN samples $\tau \sim U([0,1])$
and trains the network — using the same quantile regression / Huber loss as
QR-DQN — to output the $\tau$-quantile of the return distribution. At test time,
the agent can query any $\tau$ it wants, including ranges or distortions.

**The network architecture.** Take a DQN-style ConvNet that produces a state
embedding $\psi(s) \in \mathbb{R}^d$. Take a cosine-basis embedding of $\tau$,
$\phi(\tau) \in \mathbb{R}^d$. **Multiply elementwise:** $\psi(s) \odot \phi(\tau)$,
then feed into a small MLP to produce action-conditioned quantile values
$Z(s, a; \tau) \approx f(\psi(s) \odot \phi(\tau))_a$. The Hadamard product
$\psi(s) \odot \phi(\tau)$ is structurally a **FiLM** (feature-wise linear
modulation, Perez et al. 2018) operation with no shift term — $\tau$ rescales each
feature of $\psi(s)$. **This is the project's load-bearing connection: IQN's
$\tau$-conditioning IS FiLM.** See Phase 2.

**Risk-sensitive policies for free.** Yaari (1987) and Wang (1996) showed that any
**distortion risk measure** — a monotonic reweighting $\beta: [0,1] \to [0,1]$ of
cumulative probabilities — defines a coherent risk-sensitive utility, and that the
distorted expectation $\mathbb{E}_\beta[Z] = \int_0^1 F_Z^{-1}(\tau)\, d\beta(\tau)$
is equivalently $\mathbb{E}_{\tau \sim \beta}[Z_\tau]$. So once IQN can produce
$Z_\tau$ for any $\tau$, the agent can implement *any* distortion-risk policy by
sampling $\tau \sim \beta$ instead of $\tau \sim U([0,1])$ when computing the
policy's value estimate. Examples explored in the paper:

- **Risk-neutral:** $\beta(\tau) = \tau$ (identity) → standard mean expectation.
- **CVaR$_\eta$:** $\tau \sim U([0, \eta])$ → average only the bottom-$\eta$ fraction
  of returns. $\eta < 1$ is risk-averse.
- **Wang$_\eta$:** $\beta(\tau) = \Phi(\Phi^{-1}(\tau) + \eta)$ with $\Phi$ the
  standard-normal CDF. $\eta < 0$ risk-averse, $\eta > 0$ risk-seeking.
- **Pow$_\eta$:** simple power-law distortion — risk-averse for $\eta < 0$,
  risk-seeking for $\eta > 0$.
- **CPW (cumulative-prospect-theory):** $\beta(\tau) = \tau^\eta / (\tau^\eta + (1-\tau)^\eta)^{1/\eta}$
  with $\eta = 0.71$ matching human prospect-theory behavior.

**Results.** Atari-57: IQN 1019% mean / 218% median / human-gap 0.141, vs. QR-DQN
864%/193%/0.165 and Rainbow 1189%/230%/0.144 (Rainbow combines C51 with 5 other
DQN improvements; IQN matches it as a single-shot algorithm). On human-starts, IQN
**beats Rainbow** at 162% vs 153% median. Risk-sensitive runs (CVaR(0.1), Wang(-0.75))
show that risk-averse policies actually *improve* training performance on several
games, e.g. ASTERIX, ASSAULT — interpreted as risk-aversion implicitly encoding
"stay alive longer."

**Initial takeaway.** IQN unifies three things that were previously separate.
(1) A function-of-$\tau$ replaces a vector-of-quantiles, removing the discretization
hyperparameter. (2) The same network gives the agent simultaneous access to all
quantiles, so distortion-risk measures plug in by changing the sampling distribution
— no retraining. (3) The conditioning mechanism is FiLM, which is exactly the
mechanism the project already uses for modulator-conditioning. The project's
modulator-as-risk-signal hypothesis maps onto this architecture directly.

### Phase 2 — Graduate-Level Deep Dive

#### 3.1 The implicit quantile function

Define the quantile function

$$Z_\tau(s, a) := F_{Z(s, a)}^{-1}(\tau), \quad \tau \in [0, 1]. \tag{IQN.1}$$

For $\tau \sim U([0, 1])$, $Z_\tau(s, a)$ is a sample from the return distribution
$Z(s, a)$ — this is the **probability-integral transform**. IQN models $Z_\tau(s, a)$
as a parameterized, differentiable function

$$Z_\tau(s, a) \approx f_\theta\big(\psi(s) \odot \phi(\tau)\big)_a, \tag{IQN.2}$$

where $\psi : \mathcal{S} \to \mathbb{R}^d$ is the DQN convolutional trunk,
$\phi : [0, 1] \to \mathbb{R}^d$ is the $\tau$-embedding, $f : \mathbb{R}^d \to \mathbb{R}^{|\mathcal{A}|}$
is a small MLP, and $\odot$ is the Hadamard (element-wise) product.

The Hadamard combination is the architectural choice the paper validates against
two alternatives — concatenation $[\psi(s); \phi(\tau)]$ and "residual" $\psi(s) \odot (1 + \phi(\tau))$
— in an ablation across six Atari games. The Hadamard form is robust and slightly
preferred; the paper's chosen form.

#### 3.2 The cosine basis $\tau$-embedding

The $\tau$-embedding is

$$\phi_j(\tau) = \mathrm{ReLU}\!\left( \sum_{i=0}^{n-1} \cos(\pi i \tau)\, w_{ij} + b_j \right), \quad n = 64, \tag{IQN.3}$$

i.e., a $n$-term cosine Fourier-feature basis $\{\cos(\pi i \tau)\}_{i=0}^{n-1}$
followed by a learned linear-then-ReLU layer. The Fourier basis is crucial: it
makes $\phi$ smooth and periodic-friendly while letting the network learn arbitrary
non-monotone dependencies on $\tau$ at the resolution permitted by the basis.

#### 3.3 IQN as a FiLM mechanism — the load-bearing project connection

**Feature-wise Linear Modulation (FiLM; Perez et al. 2018).** Given a feature
vector $h \in \mathbb{R}^d$ and a conditioning input $c$, FiLM produces

$$\mathrm{FiLM}(h \mid c) = \gamma(c) \odot h + \beta(c),$$

where $\gamma(c), \beta(c) \in \mathbb{R}^d$ are computed from $c$ by a "FiLM
generator" subnetwork. The geometric content is: $c$ rescales (gain $\gamma$) and
shifts (bias $\beta$) every feature of $h$ independently.

**IQN's $\tau$-conditioning matches FiLM with $\beta = 0$.** Set $h = \psi(s)$
(the state embedding) and $c = \tau$ (the conditioning scalar). Set
$\gamma(\tau) = \phi(\tau)$ (the cosine embedding), $\beta(\tau) = 0$. Then

$$\mathrm{FiLM}\big(\psi(s) \mid \tau\big) = \phi(\tau) \odot \psi(s) = \psi(s) \odot \phi(\tau),$$

which is exactly IQN's eq. (IQN.2) up to the trivial commutativity of $\odot$.
IQN is FiLM with no shift term and a cosine-basis FiLM-generator. The paper
predates the term "FiLM-conditioning" in the RL literature but the architecture is
structurally identical.

**Why this matters for the project.** The modulator-conditioned agent already uses
FiLM heads to inject a scalar (or low-dimensional) neuromodulator signal into the
policy network. Schematically:

$$\mathrm{Policy}_{\text{project}}(h \mid m) = \gamma(m) \odot h + \beta(m), \quad m = \text{modulator signal.}$$

If we map $m \mapsto \tau$ — interpret the modulator as a **risk-level scalar**
in $[0,1]$ — then IQN's eq. (IQN.2) **is** the project's modulator-conditioned
value head, with a specific cosine-Fourier-basis generator and a specific training
signal (quantile regression on the distributional Bellman target). The modulator's
behavioral content under this mapping is: "$m$ high → sample low-$\tau$ quantile
during action selection → behave risk-aversely (avoid downside)." That is a literal,
non-metaphorical translation of "high serotonin → risk-aversion" into a trained
neural architecture. Lim & Malik (paper 4) is the policy-side companion that
formalizes the training-time aspect.

A subtlety worth flagging: IQN samples $\tau$ randomly at *training* time and uses
the distortion $\beta$ only at *acting* time, so the value head is trained to be
*all* quantiles simultaneously. The project's modulator agent in its current form
trains $m$-conditioned policies (the modulator changes the policy at training and
acting time). The IQN trick — train on the full distribution, modulate only at
acting time — is a viable alternative training scheme worth highlighting.

#### 3.4 The IQN loss

The pairwise quantile-Huber loss is the QR-DQN loss with two independent quantile
samples replacing the fixed $\hat\tau_i$ grid:

$$\mathcal{L}_t(\theta) = \frac{1}{N'} \sum_{i=1}^{N} \sum_{j=1}^{N'} \rho_{\tau_i}^\kappa\!\left( \delta_t^{\tau_i, \tau_j'} \right), \tag{IQN.4}$$

where $\tau_i, \tau_j' \sim U([0, 1])$ are i.i.d. across $i, j$ and across
gradient steps, and the sample TD error is

$$\delta_t^{\tau, \tau'} = r_t + \gamma\, Z_{\tau'}\!\big(x_{t+1}, \pi_\beta(x_{t+1})\big) - Z_\tau(x_t, a_t). \tag{IQN.5}$$

$N$ controls the number of quantile **predictions** per state and $N'$ the number
of next-state **target quantiles**. Ablation finds $N = N' = 8$ near-saturates
long-term performance; $N$ has stronger early-learning effect, $N'$ stronger
gradient-variance effect.

#### 3.5 Distortion risk measures and risk-sensitive policies

For a distortion risk measure $\beta : [0, 1] \to [0, 1]$ (continuous,
non-decreasing, $\beta(0) = 0$, $\beta(1) = 1$), the **distorted expectation** of
the return is

$$Q_\beta(s, a) := \mathbb{E}_{\tau \sim U([0, 1])}\big[ Z_{\beta(\tau)}(s, a) \big] = \int_0^1 F_{Z(s, a)}^{-1}(\tau)\, d\beta(\tau). \tag{IQN.6}$$

The two forms are equivalent by change-of-variable. The risk-sensitive greedy
policy is

$$\pi_\beta(s) = \arg\max_{a \in \mathcal{A}} Q_\beta(s, a), \tag{IQN.7}$$

approximated at acting time by

$$\tilde\pi_\beta(s) = \arg\max_{a \in \mathcal{A}} \frac{1}{K} \sum_{k=1}^{K} Z_{\beta(\tilde\tau_k)}(s, a), \quad \tilde\tau_k \sim U([0, 1]),\; K = 32. \tag{IQN.8}$$

**Distortion-measure examples in the paper.** With $\eta$ a risk parameter:

- $\mathrm{CVaR}(\eta, \tau) = \eta \tau$ (sample $\tau \sim U([0, \eta])$): take the
  $\eta$-conditional-value-at-risk — average over the bottom-$\eta$ fraction. Risk-averse.
- $\mathrm{Wang}(\eta, \tau) = \Phi(\Phi^{-1}(\tau) + \eta)$ with $\Phi$ standard
  normal: a smooth shift in standard-normal coordinates. $\eta < 0$ risk-averse,
  $\eta > 0$ risk-seeking, allows symmetric trade-off.
- $\mathrm{Pow}(\eta, \tau) = \tau^{1/(1+|\eta|)}$ for $\eta \ge 0$,
  $1 - (1-\tau)^{1/(1+|\eta|)}$ otherwise — a power-law distortion.
- $\mathrm{CPW}(\eta, \tau) = \tau^\eta / (\tau^\eta + (1-\tau)^\eta)^{1/\eta}$,
  $\eta = 0.71$ → matches human prospect-theory weighting (locally concave for
  small $\tau$, convex for large; the only non-monotone-curvature example).

**Connection to expected utility.** Yaari (1987) proved that distortion-risk-measure
maximization is the *dual* of utility-function maximization: rather than weighting
outcomes by a utility $U(z)$ and integrating against the distribution, you reweight
the distribution itself and integrate against the identity. The two are inverses
under appropriate regularity (Yaari 1987 §3). This is what distinguishes Yaari's
"dual theory of choice" from von Neumann–Morgenstern expected utility — and it's
what makes IQN's risk-sensitive policy class as expressive as the full distortion
family.

#### 3.6 IQN as a universal value function approximator (UVFA)

Viewing $Z(s, a; \tau)$ as parameterized by an auxiliary "goal" $\tau$, IQN
generalizes Schaul et al.'s (2015) **universal value function approximator** —
$V(s; g)$ for goal $g$ — to the distributional case with $\tau$ as the goal.
This perspective predicts (and the paper confirms) generalization across $\tau$:
training on many $\tau$ samples yields better between-quantile representations
than training each quantile head independently. The connection foreshadows the
project's broader interest in goal/condition-vectored value functions (see also
Borsa et al. 2018 USF, in the same TD corpus).

#### 3.7 Empirical results — Atari 57

**Best-agent, no-op-start, 30-game-averaged:**

| Algorithm | Mean | Median | Human-Gap | Seeds |
|---|---:|---:|---:|---:|
| DQN | 228% | 79% | 0.334 | 1 |
| Prioritized | 434% | 124% | 0.178 | 1 |
| C51 | 701% | 178% | 0.152 | 1 |
| Rainbow | 1189% | 230% | 0.144 | 2 |
| QR-DQN | 864% | 193% | 0.165 | 3 |
| **IQN** | **1019%** | **218%** | **0.141** | **5** |

- **Human-starts (median):** IQN 162% > Rainbow 153% > C51 125% > Prioritized 128% > DQN 68%. IQN is the *highest* on this metric.
- **Risk-sensitive runs (six-game subset):** CVaR(0.1) and Wang(-0.75) improve over
  risk-neutral on ASTERIX, ASSAULT; Wang(1.5) (risk-seeking) hurts on three games.
  Risk-aversion interpreted as an implicit "stay alive longer" inductive bias.
- **Sample-budget ablation:** $N = 1$ (one quantile per gradient step) gives
  ~3× DQN long-term — the distributional update helps even with a single sample;
  $N \ge 8$ near-saturates.

### Appendix: Section-by-Section Backbone (IQN)

**Abstract.** IQN extends QR-DQN from discrete quantiles to a continuous quantile
function via reparameterization. The implicitly-defined return distribution
supports a large class of risk-sensitive policies (distortion risk measures).
Significant Atari-57 gains.

**§1 Introduction.** Distributional RL background. C51 fixed-grid → QR-DQN
adjustable-location → IQN continuous function. Lists three benefits of IQN over
QR-DQN: (i) representation capacity tied to network not output size; (ii)
adjustable samples-per-update for data-efficiency tuning; (iii) implicit
representation enables risk-sensitive policies via $\beta$-distorted sampling.

**§2 Background / related work.** §2.1 distributional RL recap (Bellman-equality-in-distribution,
contraction in Wasserstein for evaluation, not contraction for control). §2.2 the
$p$-Wasserstein metric. §2.3 QR-DQN (uniform mixture of Diracs, quantile-Huber loss,
pairwise TD errors $\delta_{ij}$). §2.4 risk in RL — utility (von Neumann–Morgenstern)
vs. distortion (Yaari 1987, Wang 1996), the Allais paradox, distortion risk
measures, CVaR, CPW, prospect theory.

**§3 Implicit Quantile Networks.** The implicit quantile $Z_\tau(s,a)$. Distortion
$Q_\beta = \mathbb{E}_{\tau \sim U([0,1])}[Z_{\beta(\tau)}(s,a)]$ and risk-sensitive
policy $\pi_\beta$. Sample TD error $\delta_t^{\tau, \tau'}$ (eq. 2). IQN loss
(eq. 3) with $N$ and $N'$ controlling prediction and target samples. Sample-based
policy $\tilde\pi_\beta$ with $K$ samples. §3.1 implementation: architecture
$Z_\tau(s,a) \approx f(\psi(s) \odot \phi(\tau))_a$, cosine embedding eq. 4,
$n = 64$, architectural ablation (Hadamard wins), sample-count ablation
($N = N' = 8$ near-saturates).

**§4 Risk-sensitive RL.** Five distortion measures studied: CPW (prospect theory),
Wang (Gaussian shift, smooth risk-averse↔seeking), Pow, CVaR (bottom-$\eta$). On
six Atari games: risk-averse beats risk-neutral on some, risk-seeking hurts. Open
question why; possibly risk-aversion ≈ longevity bias.

**§5 Full Atari-57 results.** IQN 1019%/218%/0.141 vs Rainbow 1189%/230%/0.144,
QR-DQN 864%/193%/0.165 (no-op start, 5 vs 2 vs 3 seeds). Human-starts: IQN 162%
beats Rainbow 153%. Human-gap (avg distance below human on sub-human games): IQN
best of all.

**§6 Discussion / Conclusions.** IQN is a fully integrated distributional RL agent
without prior support assumptions; trainable with any samples-per-update; enables
distortion-risk policies; significant gains on Atari-57 (halves the QR-DQN→Rainbow
gap). Open theoretical questions: sample-based convergence for QR-class algorithms;
contraction for approximate-quantile (vs. fixed-grid) parameterizations; convergence
of distorted expectations under the distributional Bellman operator (this last is
precisely what Lim & Malik 2022 takes up).

---

## 4. Lim and Malik (2022) — Distributional RL for Risk-Sensitive Policies

**PDF:** `docs/project/references/TD/sources/Lim and Malik 2022 - Distributional reinforcement learning for risk-sensitive policies.pdf`
**Venue:** NeurIPS 2022.
**Key contribution:** identifies that the standard "swap the expectation for a
distortion / CVaR in the action-selection step" recipe used by Dabney et al.
(IQN, 2018a) and Keramati et al. (2020) **fails to converge to either of the two
canonical risk-sensitive objectives** — neither the dynamic (time-consistent,
Markovian) CVaR nor the static (time-inconsistent, history-dependent) CVaR —
even when the optimal static-CVaR policy is itself stationary and Markov.
Proposes a new distributional Bellman operator $\tilde{\mathcal{T}}_\psi$ whose
action selection uses a state-indexed *threshold* $\psi(x)$ — effectively pulling
the static-CVaR augmented-MDP threshold $s$ inside the operator — and proves
that for the class of MDPs with a stationary Markov optimal static-CVaR policy,
the optimal value distribution $Z^{\pi^*}$ is a fixed point of this operator.
Builds a drop-in modification to QR-DQN that runs Algorithm 1 (a threshold-tracked
policy-execution loop) and trains via Algorithm 2 (quantile-regression loss with
the threshold-aware action selection). Empirically outperforms the Dabney-style
Markov action-selection strategy on a 4-state synthetic MDP with known optimal
CVaR policy, on an American-option-exercise task with real Dow-component prices,
and on the Atari game Asterix at CVaR-level $\alpha = 0.25$.

### Phase 1 — Foundational Overview (Undergraduate-Level) <a name="phase-1--foundational-overview-undergraduate-level-4"></a>

**The problem in one sentence.** If you already have a distributional RL agent
that has learned the full return distribution $Z(s,a)$, and you want it to be
risk-averse rather than risk-neutral, what is the *right* way to plug a risk
measure (specifically CVaR) into the Bellman update so that the fixed point of
your learning rule is actually the risk-sensitive policy you wanted?

**What is CVaR?** Conditional Value-at-Risk at level $\alpha \in (0,1]$, written
$C_\alpha(Z)$, is the mean of the *worst $\alpha$-fraction* of outcomes of a
return random variable $Z$. Concretely, if $Z$ is continuous, $C_\alpha(Z) =
\mathbb{E}[Z \mid Z < q_\alpha(Z)]$ where $q_\alpha(Z)$ is the $\alpha$-quantile.
At $\alpha = 1$, CVaR reduces to the ordinary expectation; at $\alpha = 0.1$, it
is the average over the bottom 10% of outcomes — so an agent that maximizes
CVaR-at-$0.1$ behaves as if it only cared about how bad the bottom decile is.
This is the classical "tail-aware" risk measure used throughout finance and
control, and it satisfies the four coherence axioms of Artzner et al. (1999).

**Two flavors of CVaR over a trajectory.** In a multi-step MDP, "CVaR of the
total return" can mean two different things, and these two definitions give
different optimal policies:

- **Static CVaR:** $\max_\pi C_\alpha[Z^\pi]$ where $Z^\pi = \sum_t \gamma^t r_t$
  is the *total* discounted return. Because this is a single CVaR taken over the
  whole trajectory's distribution, the optimal policy is in general
  *time-inconsistent* and *history-dependent* — once you have already collected
  large reward, you can afford to take a riskier action late in the episode; if
  you are running below expectations, you should become more conservative.
- **Dynamic CVaR:** the recursively defined Markovian variant of Ruszczyński
  (2010), with optimality operator $\mathcal{T}_\alpha^D D(x,a) := C_\alpha[r_t +
  \gamma \max_{a'} D(x_{t+1}, a') \mid x_t = x, a_t = a]$. By construction it
  admits a stationary deterministic optimal policy and is a $\gamma$-contraction
  in sup-norm, but it can be wildly over-conservative or even over-optimistic
  relative to what an outside observer would call "risk-averse behavior."

**The paper's core negative result.** The Dabney et al. (2018a) recipe — keep
the distributional Bellman update, but replace $\arg\max_{a'} \mathbb{E}[U(x',a')]$
with $\arg\max_{a'} C_\alpha[U(x',a')]$ in the action-selection step — looks
intuitively reasonable. Lim & Malik show by **counterexample** (two tiny
3-state, 2-action MDPs, Fig. 1 a/c) that this fixed-point recursion **converges
to neither static-CVaR-optimal nor dynamic-CVaR-optimal policies**, even when
the static-CVaR optimum is stationary and Markov. The intuitive reason: the
inner $\arg\max_{a'} C_\alpha[U(x',a')]$ asks "which action looks best in the
$\alpha$-tail *starting from $x'$, ignoring the history that got me here*",
whereas the correct static-CVaR comparison is "which action looks best in the
$\alpha$-tail *given the threshold I would have inherited from the trajectory
to $x'$*."

**The paper's positive result.** They propose an alternative operator
$\tilde{\mathcal{T}}_\psi$ parameterized by a state-indexed scalar $\psi(x)$
that *plays the role of the inherited threshold*. The action selection in
$\tilde{\mathcal{T}}_\psi$ is "pick the $a'$ that maximizes $\mathbb{E}[-(s -
U(x',a'))_+]$ at threshold $s = (\psi(x) - r)/\gamma$" — exactly the
augmented-MDP value $W^{\tilde\pi}(x', s, a')$ of Bäuerle & Ott (2011). They
prove that if the original MDP has a stationary Markov optimal static-CVaR
policy $\pi^*$, then with $\psi(x)$ chosen from the active set $S^*(x)$ of
thresholds reachable under $\pi^*$, the optimal return distribution
$Z^{\pi^*}(x, \pi^*(x))$ is a **fixed point** of $\tilde{\mathcal{T}}_\psi$
(Proposition 3 for the on-policy state-actions, Proposition 4 for all
state-actions under a stronger condition on $\psi$).

**The headline takeaway.** Distributional RL + "swap expectation for CVaR at
action selection" is *not* a valid recipe — but distributional RL + a properly
threshold-tracked Bellman operator IS, at least at the level of fixed-point
optimality. The price is that the policy-execution loop now has to carry a
running threshold $s$ updated as $s \leftarrow (s - r)/\gamma$ along the
trajectory — i.e., the agent at runtime becomes non-Markov in the original MDP
even when the optimal policy *as a function of $(x, s)$* is Markov in the
augmented MDP. The training algorithm (Algorithm 2) is a small modification of
QR-DQN: replay-buffer entries store $(x_k, s_k, a_k, r_k, x'_k)$ rather than
$(x_k, a_k, r_k, x'_k)$, and the target action $a'_k = \arg\max_{a'} W^{\theta'}
(x'_k, (s_k - r_k)/\gamma, a')$ uses the threshold-aware value.

### Phase 2 — Graduate-Level Deep Dive <a name="phase-2--graduate-level-deep-dive-4"></a>

#### 2.1 — The distortion-risk view of CVaR

To place Lim & Malik in the lineage of the previous three papers, it helps to
first restate CVaR as a *distortion-risk measure* in the sense used by IQN
(Dabney et al. 2018a, Section 2.3 of the IQN paper). A **distortion measure** is
a non-decreasing function $g: [0,1] \to [0,1]$ with $g(0) = 0$ and $g(1) = 1$.
Given a return random variable $Z$ with CDF $F_Z$, the **distorted expectation**
is

$$
\rho_g[Z] \;:=\; \int_0^1 F_Z^{-1}(\tau)\,dg(\tau)
\;=\; \int_0^1 F_Z^{-1}(\tau)\, g'(\tau)\,d\tau
\quad\text{(when $g$ is differentiable).}
$$

Different $g$ give different risk attitudes:

- $g(\tau) = \tau$ (identity) recovers the plain expectation $\mathbb{E}[Z] = \int_0^1 F_Z^{-1}(\tau)\,d\tau$.
- $g(\tau) = \min(\tau/\alpha,\, 1)$ (clip-and-rescale at level $\alpha$) gives
  $\rho_g[Z] = \frac{1}{\alpha}\int_0^\alpha F_Z^{-1}(\tau)\,d\tau = C_\alpha(Z)$
  — i.e., **CVaR is the distortion with derivative $g'(\tau) = \mathbf{1}[\tau \le \alpha]/\alpha$**,
  which concentrates all the probability mass uniformly on the bottom-$\alpha$ quantiles.
- Wang's distortion $g_\eta(\tau) = \Phi(\Phi^{-1}(\tau) + \eta)$ gives a smooth
  Gaussian shift; CPW is the prospect-theory distortion of Tversky & Kahneman; etc.

This frames the question Lim & Malik tackle as: *given a distorted-expectation
objective $\rho_g$ (specifically, CVaR), does the distributional Bellman operator
constructed by replacing $\mathbb{E}$ with $\rho_g$ at action selection
converge to a $\rho_g$-optimal policy?* Their answer is **no for the
naïve recipe; yes for a redesigned operator that tracks the implicit threshold
that CVaR-as-distortion hides.**

#### 2.2 — Standard distributional RL recap (notation alignment)

We follow the paper's notation. The value distribution $U \in \mathcal{Z}$
sends $(x, a)$ to a probability distribution over returns. The distributional
Bellman operator $\tilde{\mathcal{T}}^\pi$ acts by

$$
\tilde{\mathcal{T}}^\pi U(x,a) \stackrel{D}{=} R + \gamma\, U(X', \pi(X')),
$$

where $\stackrel{D}{=}$ is equality in distribution and $R, X'$ are induced by
$p(r,x'\mid x,a)$. Bellemare et al. (2017) prove this is a $\gamma$-contraction
in the maximal-Wasserstein metric $d(U,V) := \sup_{x,a} W_1(U(x,a), V(x,a))$.
The distributional Bellman *optimality* operator,

$$
\tilde{\mathcal{T}} U(x,a) \stackrel{D}{=} R + \gamma\, U(X', A'),\qquad
A' = \arg\max_{a'} \mathbb{E}[U(X', a')], \tag{6}
$$

is a $\gamma$-contraction in $\mathcal{Q}$ (the space of $Q$-functions) under
sup-norm of element-wise expectation — i.e., $\|\mathbb{E}\tilde{\mathcal{T}} U
- \mathbb{E}\tilde{\mathcal{T}} V\|_\infty \le \gamma \|\mathbb{E} U -
\mathbb{E} V\|_\infty$ — but is **not** in general a contraction in $\mathcal{Z}$,
because two distinct optimal policies can have identical expected return but
very different return distributions. This is the standard caveat already
flagged in the C51 review above.

#### 2.3 — The Dabney–Keramati Markov action-selection operator and why it fails

The Markov risk-sensitive operator proposed in IQN (Dabney et al. 2018a) and
used in Keramati et al. (2020) replaces $\mathbb{E}$ with $C_\alpha$ in the
action-selection step of (6):

$$
\tilde{\mathcal{T}}_\alpha^D U(x,a) \stackrel{D}{=} R + \gamma\, U(X', A'),\qquad
A' = \arg\max_{a'} C_\alpha[U(X', a')]. \tag{7}
$$

The natural conjecture is that the fixed point of (7) is the dynamic-CVaR-optimal
distribution $D_\alpha^*$ of (5). **Proposition 1** disproves this by
counterexample. Consider the MDP in Fig. 1(c) of the paper, with initial state
$X_1$, intermediate state $X_2$, terminal state $X_3$, two actions $A_1, A_2$
available only at $X_2$, and stochastic rewards parameterized by small $\epsilon$
and $p$ with $0 < \epsilon \ll p < 1$ and $p^2 + \epsilon < \alpha < p$.

Iterating (7) drives $U(X_2, A_1) \to \{(p; 0),\,(1-p; 1)\}$ and $U(X_2, A_2) \to
\{(1; \epsilon)\}$. At CVaR level $\alpha < p$, the $\alpha$-tail of $U(X_2, A_1)$
sees only the "$0$" atom, so $C_\alpha[U(X_2, A_1)] = 0 < \epsilon =
C_\alpha[U(X_2, A_2)]$, and (7) selects $A_2$ at $X_2$. Propagating back through
$X_1$ gives $U^*(X_1) = \{(p; \epsilon),\,(1-p; 1)\}$, so the *trajectory-level*
$C_\alpha[U^*(X_1)] = \epsilon$. But the static-CVaR-optimal policy is to
choose $A_1$ at $X_2$ (so the static-CVaR of the trajectory starting at $X_1$
is $> \epsilon$ as long as $\alpha > p^2 + \epsilon$), and this policy is itself
stationary and Markov. Hence (7) fails to find a stationary-Markov static-CVaR
optimum that *exists*. A symmetric example (Fig. 1(a)) shows it also fails to
recover dynamic-CVaR optimality.

**Why this happens, in one line:** the inner $\arg\max C_\alpha[U(X', a')]$
re-evaluates the tail "from scratch" at every next state, with no memory of
what reward already occurred on the way to $X'$. Static CVaR, by contrast, is a
property of the *whole trajectory's* return, so the correct local comparison at
$X'$ must condition on the threshold $s$ that the trajectory has accumulated.

#### 2.4 — The augmented-MDP detour (Bäuerle & Ott 2011)

The classical fix for static CVaR (Bäuerle & Ott 2011) is to lift the original
MDP $\mathcal{M}$ to an augmented MDP $\tilde{\mathcal{M}}$ whose state is the
pair $\tilde x = (x, s)$, with $s$ a real-valued *threshold* tracking cumulative
shifted return. The augmented transition is

$$
\tilde p\!\left(0,\, (x',\, (s - r)/\gamma) \,\Big|\, (x, s),\, a\right) \;:=\; p(r,\, x' \mid x, a),
$$

i.e., rewards in $\tilde{\mathcal{M}}$ are zero except at terminals, and the
threshold updates deterministically as $s \leftarrow (s - r)/\gamma$. The
optimal static-CVaR policy in $\mathcal{M}$ at initial state $x_0$ and CVaR
level $\alpha$ corresponds to the optimal policy in $\tilde{\mathcal{M}}$
starting from $(\tilde x_0) = (x_0, s_0^*)$ with $s_0^* = q_\alpha(Z^{\tilde\pi^*}(x_0,
s_0^*))$. The augmented Q-value is

$$
W^{\tilde\pi}(x, s, a) \;:=\; \mathbb{E}_{z \sim Z^{\tilde\pi}(x, s, a)}\!\big[-(s - z)_+\big]. \tag{8}
$$

This is exactly the inner objective of the Rockafellar–Uryasev representation
$C_\alpha(Z) = \max_s\, s - \frac{1}{\alpha}\mathbb{E}[(s - Z)_+]$ rearranged so
that $W$ is the contribution to $C_\alpha$ from $Z$ for *fixed* threshold $s$.
Solving $\tilde{\mathcal{M}}$ directly with model-free RL is expensive — every
$(x, a, r, x')$ would have to be re-experienced under many different $s$ — so
Lim & Malik's contribution is to **bring the threshold into the distributional
Bellman operator on the unaugmented state space** rather than literally
augmenting the MDP.

#### 2.5 — The proposed operator $\tilde{\mathcal{T}}_\psi$ and its fixed-point theorem

Given a function $\psi: \mathcal{X} \to \mathbb{R}$ assigning a single threshold
to each state, define

$$
\tilde{\mathcal{T}}_\psi U(x,a) \stackrel{D}{=} R + \gamma\, U(X', A'),\qquad
A' = \arg\max_{a'} W^U\!\left(X',\, \frac{\psi(x) - R}{\gamma},\, a'\right), \tag{10}
$$

where, in line with (8),

$$
W^U(x, s, a) \;:=\; \mathbb{E}_{z \sim U(x, a)}\!\big[-(s - z)_+\big]. \tag{9}
$$

Compared with (7), the key change is that the action selection at the *next*
state $X'$ is performed at the threshold $(\psi(x) - R)/\gamma$ that the
augmented-MDP recursion would have generated, rather than at $C_\alpha$ of the
next-state distribution evaluated in isolation. The mapping $\psi$ is the
mechanism through which the threshold pops out from a hidden internal variable
into a *function of the current state alone* — and the central technical
question is whether $\psi$ can be chosen so that the fixed-point structure
matches static-CVaR optimality.

**Proposition 3 (fixed point on the optimal policy's support).** Suppose
$\tilde\pi^*$ is the unique stationary-Markov optimal $\alpha$-CVaR policy in
$\mathcal{M}$, with corresponding $\pi^*(x) := \tilde\pi^*(x, s)$ valid for all
$s \in S^*(x)$ where $S^*(x) := \{s : \exists t \ge 0,\,\Pr^{\tilde\pi^*}[\tilde
x_t = (x, s) \mid \tilde x_0 = (x_0, s_0^*)] > 0\}$ is the *active set* of $x$
under $\tilde\pi^*$ — i.e., the set of threshold values one can actually
encounter at state $x$ while rolling $\tilde\pi^*$ from $(x_0, s_0^*)$. Choose
any $\psi: \mathcal{X} \to \mathbb{R}$ with $\psi(x) \in S^*(x)$ for every $x$.
Then

$$
\tilde{\mathcal{T}}_\psi Z^{\pi^*}(x,\, \pi^*(x)) \stackrel{D}{=} Z^{\pi^*}(x,\, \pi^*(x))
\quad\text{for every } x \in \mathcal{X}.
$$

**Proof sketch (Algorithm 1 as a witness).** The paper's Algorithm 1 is a
policy-execution loop that initializes $s \leftarrow q_\alpha(U(x_0,\, a_0))$
with $a_0 = \arg\max_{a'} C_\alpha[U(x_0, a')]$, then in the loop performs
$s \leftarrow (s - r)/\gamma$ after each transition and selects the next action
by $a \leftarrow \arg\max_{a'} W^U(x, s, a')$. This is exactly the
augmented-MDP optimal policy in $\tilde{\mathcal{M}}$ at the current
$(x, s)$. **Proposition 2** establishes by induction that if $U(x,a) \stackrel{D}{=}
Z^{\pi^*}(x,a)$ for all $(x,a)$, then Algorithm 1 in fact executes $\pi^*$ in
$\mathcal{M}$ — because at every encountered $(x_t, s_t)$ we have $s_t \in
S^*(x_t)$, so $\tilde\pi^*(x_t, s_t) = \pi^*(x_t)$. The action-selection rule
of $\tilde{\mathcal{T}}_\psi$ matches Algorithm 1 exactly when $\psi(x) \in
S^*(x)$ for every reachable $x$, because then $(\psi(x) - r)/\gamma \in S^*(x')$
for every $r, x'$ in the support of $p(\cdot, \cdot \mid x, \pi^*(x))$.
Therefore the action $A'$ selected at $X'$ inside $\tilde{\mathcal{T}}_\psi$ is
$\pi^*(X')$, which makes $\tilde{\mathcal{T}}_\psi Z^{\pi^*}(x, \pi^*(x))
\stackrel{D}{=} R + \gamma\, Z^{\pi^*}(X', \pi^*(X')) \stackrel{D}{=}
Z^{\pi^*}(x, \pi^*(x))$ by the standard distributional Bellman recursion. $\square$

**Proposition 4 (fixed point over all $(x,a)$, stronger condition).** With the
same setup and the *larger* active-set generator $S^{**}(x) := \{(\psi(x') - r)/
\gamma : \exists x', a,\, p(r, x \mid x', a) > 0\}$, if there exists $\pi^*$ in
$\mathcal{M}$ with $\tilde\pi^*(x, s) = \pi^*(x)$ for all $s \in S^{**}(x)$, then
$Z^{\pi^*}$ — viewed as a function of all $(x, a)$, not only of the on-policy
$(x, \pi^*(x))$ pairs — is a fixed point of $\tilde{\mathcal{T}}_\psi$. The
proof reduces to showing $\tilde{\mathcal{T}}_\psi = \tilde{\mathcal{T}}^{\pi^*}$
on $Z^{\pi^*}$ under the stronger active-set hypothesis, so the original
distributional Bellman policy-evaluation fixed point of (Bellemare et al. 2017)
applies. $\square$

**What is *not* proved.** Lim & Malik are explicit (end of Section 2.3) that
**they do not prove $\tilde{\mathcal{T}}_\psi U \to Z^{\pi^*}$ from arbitrary
initial $U$**. They establish only that $Z^{\pi^*}$ is a fixed point under the
stated active-set conditions. Whether the operator is a contraction in any
useful metric — and hence whether iteration from arbitrary initialization
converges — is left as an open question, although they report empirically on
simple MDPs that it does converge to $Z^{\pi^*}$ and even to non-stationary
optima in some cases.

#### 2.6 — Algorithm: quantile-regression distributional Q-learning for static CVaR

The proposed algorithm specializes $\tilde{\mathcal{T}}_\psi$ to a QR-DQN-style
parameterization, with $N$ quantiles $\theta_i(x,a)$ at levels $\hat\tau_i =
(i - 0.5)/N$, and replaces the standard Markov action-selection step of QR-DQN
with the threshold-tracked one. Two algorithms are stated:

**Algorithm 1 — Policy execution for static CVaR (one episode).** Given
$\gamma$, $\alpha$, and a learned distribution $U \in \mathcal{Z}$:
1. $x \leftarrow x_0$.
2. $a \leftarrow \arg\max_{a'} C_\alpha[U(x, a')]$.
3. $s \leftarrow q_\alpha(U(x, a))$.
4. While $x$ is not terminal:
   - (a) execute $a$, observe $r, x'$;
   - (b) $x \leftarrow x'$;
   - (c) $s \leftarrow (s - r)/\gamma$;
   - (d) $a \leftarrow \arg\max_{a'} W^U(x, s, a')$.

Note the *asymmetry* between step 2 (initial action uses $C_\alpha$) and step
4(d) (all subsequent actions use $W^U$ at the running threshold). The initial
threshold $s_0 = q_\alpha(U(x_0, a_0))$ is exactly the $s^* = q_\alpha
(Z^{\tilde\pi^*}(x_0, s^*))$ of Bäuerle & Ott, computed self-consistently from
the initial CVaR-greedy action.

**Algorithm 2 — Quantile-regression distributional Q-learning for static CVaR.**
Given $\gamma$, online quantiles $\theta$, target quantiles $\theta'$, and a
mini-batch of transitions $(x_k, s_k, a_k, r_k, x'_k)$ for $k = 1, \dots, m$:
1. For each $k$:
   - (a) $a'_k \leftarrow \arg\max_{a'} W^{\theta'}\!\left(x'_k,\, (s_k - r_k)/\gamma,\, a'\right)$.
   - (b) $\tilde{\mathcal{T}}_\psi \theta_j(x_k, a_k) \leftarrow r_k + \gamma\, \theta'_j(x'_k, a'_k)$ for $j = 1, \dots, N$.
2. $L \leftarrow \frac{1}{m} \sum_{k=1}^m \frac{1}{N^2} \sum_{i,j} \rho_{\hat\tau_i}\!\left(\tilde{\mathcal{T}}_\psi \theta_j(x_k, a_k) - \theta_i(x_k, a_k)\right)$, where $\rho_\tau(u) = u(\tau - \delta_{u<0})$ is the quantile-regression loss.
3. Output $\nabla L$.

Compared with plain QR-DQN, the only line that changes is **1(a)** — the target
action is selected by the threshold-aware augmented-MDP value $W^{\theta'}$ at
$(s_k - r_k)/\gamma$, rather than by the next-state Markov $\mathbb{E}$ or
$C_\alpha$. The replay buffer must therefore additionally store $s_k$ — i.e., a
sampled trajectory carries its accumulated threshold along, so the offline
training loop is consistent with the augmented-MDP recursion.

**Quantile-based computation of $W$ and $C_\alpha$.** Both $C_\alpha$ and $W^U$
are easy to compute from quantile-parameterized distributions: given
$\theta(x,a) = (\theta_1, \dots, \theta_N)$,

$$
C_\alpha[U(x,a)] \approx \frac{1}{\lfloor \alpha N \rfloor} \sum_{i=1}^{\lfloor \alpha N \rfloor} \theta_{(i)}(x,a),
\qquad
W^U(x, s, a) \approx \frac{1}{N} \sum_{i=1}^{N} -\bigl(s - \theta_i(x,a)\bigr)_+,
$$

where $\theta_{(i)}$ denotes the $i$-th order statistic. The CVaR formula
follows from $C_\alpha(Z) = \frac{1}{\alpha} \int_0^\alpha F_Z^{-1}(\tau)\,d\tau$
(the distortion derivative $g'(\tau) = \mathbf{1}[\tau \le \alpha]/\alpha$
applied to $F_Z^{-1}$); the $W^U$ formula is a direct quantile-average of
$-(s - z)_+$.

#### 2.7 — Mean-variance and other distortion instantiations

Although the paper focuses exclusively on CVaR, the operator construction is
not specific to CVaR — the action-selection rule "use $W^U$ at threshold
$(\psi(x) - r)/\gamma$" generalizes to any distortion $\rho_g$ whose
Rockafellar–Uryasev-style representation admits a one-dimensional auxiliary
parameter that plays the role of $s$. **CVaR's representation**
$C_\alpha(Z) = \max_s \,\bigl(s - \frac{1}{\alpha}\mathbb{E}[(s-Z)_+]\bigr)$
gives the cleanest case, with the optimal $s$ equal to the $\alpha$-quantile.
**Mean-variance** $\mathbb{E}[Z] - \lambda\,\mathrm{Var}(Z)$ is a different
distortion (non-coherent in general; not addressed by the paper's
fixed-point theorem) and would require its own auxiliary-parameter analysis.
The paper does not extend to mean-variance, but the same architectural skeleton
— quantile-parameterized $U$, threshold-tracked action selection — is a natural
template for any one-parameter risk-shift family.

#### 2.8 — Empirical setting

Three experiments:

- **Synthetic 4-state MDP** (§4.1) with two actions per state, where $a_0$ is
  reward $\mathcal{N}(1, 1)$ and $a_1$ is reward $\mathcal{N}(0.8, 0.4^2)$. The
  ground-truth optimal stationary policy switches around $\alpha \approx 0.625$:
  for $\alpha > 0.625$ always choose $a_0$ (higher mean), for $\alpha < 0.625$
  always choose $a_1$ (lower variance). The proposed strategy tracks the
  ground-truth optimum across CVaR levels; the Dabney Markov strategy diverges
  noticeably below $\alpha = 0.83$ because at $X_2$ it switches to $a_1$ too
  early, ignoring that the trajectory has already accumulated reward.
- **American option exercise** (§4.2) on top-10 Dow components 2005–2019, with
  GBM-simulated prices for training and real prices for testing. $\gamma =
  0.999$, $T = 100$. The proposed approach beats the Markov action-selection
  strategy at every tested $\alpha$, with the largest gap at low $\alpha$
  (deep-tail) levels. A risk-neutral baseline ($\alpha = 1$) is far worse at
  low $\alpha$ on the test distribution.
- **Atari Asterix** (§4.3) at $\alpha = 0.25$, $\gamma = 0.99$, IQN
  architecture, 30M training steps, 3 random seeds. The proposed approach
  yields a *concentrated* histogram of episode-level discounted returns over
  100 evaluation runs, with higher CVaR-at-$0.25$ than both the Markov and
  risk-neutral baselines. Interestingly, on this particular game the Markov
  strategy underperforms even the risk-neutral baseline in mean return — the
  authors flag this as worth more extensive investigation rather than a
  general claim about Markov-CVaR.

### Appendix: Section-by-Section Backbone (Lim & Malik 2022)

This appendix preserves the paper's original argument flow. Phase 1/2 above
already regroup the content thematically; this is the trace back to the source.

**Abstract.** Distributional RL for CVaR-risk-sensitive policies. Shows the
standard Markov action-selection strategy (Dabney 2018a, Keramati 2020) does not
converge to either static or dynamic CVaR-optimal policies. Proposes a new
distributional Bellman operator and shows it greatly expands the utility of
distributional RL for CVaR. Synthetic + real-data evaluation.

**§1 Introduction.** Standard RL maximizes expected discounted return; in
high-uncertainty domains, a risk-averse policy is preferable. Focuses on CVaR
(Rockafellar & Uryasev 2000) as the tail-aware risk measure. Distributional RL
(Bellemare 2017, Morimura 2010) is attractive because it learns the full return
distribution as a side-effect of normal training, so risk-sensitive policies
could be extracted post hoc. Dabney 2018a proposed a simple recipe; this paper
shows it has subtle theoretical issues.

**§1.1 Related work.** Distributional RL: C51, QR-DQN, IQN, D4PG, FQF.
Risk-sensitive MDPs: Howard & Matheson 1972, Ruszczyński 2010, Bäuerle & Ott
2011, Borkar 2001, Tamar 2012/2015/2017, Chow & Ghavamzadeh 2014, Chow & Pavone
2014, Petrik & Subramanian 2012, Huang & Haskell 2020. Closest prior work:
Stanko & Macek 2019, who also use distributional RL but with a Markov
action-selection step that suffers from the same problem.

**§2 Problem setup.** Finite MDP $(\mathcal{X}, \mathcal{A}, p, \gamma)$,
bounded countable reward set $\mathcal{R}$, fixed initial state $x_0$, history
$h_t = (x_0, a_0, r_0, \dots, x_t)$. Policy $\pi: \mathcal{H} \to P(\mathcal{A})$.
Standard expected-return value $V^\pi$ and $Q^\pi$, Bellman equations (1) and
(2), operator $\mathcal{T}^\pi$ is a $\gamma$-contraction in sup-norm.

**§2.1 Static and dynamic CVaR.** Definition of $C_\alpha(Z) = \max_s\, s -
(1/\alpha)\mathbb{E}[(s-Z)_+]$ with optimum at $s = q_\alpha(Z)$. Static-CVaR
objective $\max_\pi C_\alpha(Z^\pi)$ is time-inconsistent and history-dependent.
Dynamic CVaR is the recursive Ruszczyński variant with Bellman operator
$\mathcal{T}_\alpha^D D(x,a) = C_\alpha[r_t + \gamma \max_{a'} D(x_{t+1}, a')
\mid x_t = x, a_t = a]$ which IS a $\gamma$-contraction in sup-norm, with a
stationary Markov optimum $D_\alpha^*$. Bäuerle & Ott 2011 augmented-MDP fix
for static CVaR: states $\tilde x = (x, s)$, threshold updates $s \leftarrow
(s - r)/\gamma$.

**§2.2 Distributional RL.** Standard distributional Bellman operator
$\tilde{\mathcal{T}}^\pi$ on $U \in \mathcal{Z}$: $\tilde{\mathcal{T}}^\pi U(x,a)
\stackrel{D}{=} R + \gamma U(X', \pi(X'))$. $\gamma$-contraction in maximal
1-Wasserstein. Optimality operator (6) is contraction in $\mathcal{Q}$ under
$\mathbb{E}$-sup-norm but not in $\mathcal{Z}$. Equation (7) defines the Markov
risk-sensitive operator $\tilde{\mathcal{T}}_\alpha^D$. **Proposition 1**:
$\tilde{\mathcal{T}}_\alpha^D$ converges to neither static- nor dynamic-CVaR
optimum even when the optimal CVaR policy is stationary Markov; proved by two
small-MDP counterexamples (Fig. 1 (a) and (c)).

**§2.3 Distributional RL for CVaR.** Re-derives Bäuerle & Ott's augmented-MDP
representation $W^{\tilde\pi}(x, s, a) = \mathbb{E}_{z \sim Z^{\tilde\pi}(x, s,
a)}[-(s-z)_+]$ — eq. (8). Defines Algorithm 1 (threshold-tracking policy
execution). **Proposition 2**: if $U(x,a) \stackrel{D}{=} Z^{\pi^*}(x,a)$ for
all $(x,a)$ and $\pi^*$ is the unique stationary-Markov static-CVaR optimum,
then Algorithm 1 executes $\pi^*$. Defines the new operator $\tilde{\mathcal{T}}_\psi$
in eq. (10) with action selection $A' = \arg\max_{a'} W^U(X', (\psi(x) - R)/\gamma, a')$.
**Proposition 3**: $Z^{\pi^*}(x, \pi^*(x))$ is a fixed point of
$\tilde{\mathcal{T}}_\psi$ when $\psi(x) \in S^*(x)$. **Proposition 4**:
$Z^{\pi^*}$ is a fixed point at *all* $(x,a)$ under the stronger active-set
condition $\psi(x') \in S^{**}(x)$. Convergence from arbitrary initialization
is left open.

**§3 Algorithm.** QR-DQN-style parameterization with $N$ quantiles at levels
$\hat\tau_i = (i-0.5)/N$. Algorithm 2: target-network QR-DQN training loop
with action selection $a'_k = \arg\max_{a'} W^{\theta'}(x'_k, (s_k - r_k)/
\gamma, a')$ and quantile-regression loss $\rho_{\hat\tau_i}(u) = u(\hat\tau_i
- \delta_{u<0})$. Replay buffer stores $(x_k, s_k, a_k, r_k, x'_k)$.

**§4 Empirical results.** Two hidden layers, ReLU, Adam, $N = 100$ quantiles.

- **§4.1 Synthetic data.** 4-state MDP, two-action choice between $\mathcal{N}(1,1)$
  and $\mathcal{N}(0.8, 0.16)$ rewards. Proposed beats Markov action selection
  at all tested CVaR levels (Fig. 2 left). Middle/right of Fig. 2 show the
  ground-truth CVaR for each stationary policy with the switching points the
  proposed strategy tracks.
- **§4.2 Option trading.** American option exercise, Dow top-10, 2005–2019,
  GBM training data, real test data. Proposed beats Markov at all $\alpha$;
  biggest gap at low $\alpha$ (deep tail).
- **§4.3 Atari Games.** Asterix, $\alpha = 0.25$, $\gamma = 0.99$, IQN
  architecture, 30M steps, 3 seeds. Proposed: more concentrated return
  histogram, higher 0.25-CVaR than both Markov and risk-neutral.

**§5 Conclusion and future work.** Convergence of $\tilde{\mathcal{T}}_\psi$
from arbitrary $U$ is open. Practical question of how to manage stored $s$ in
replay buffer after distribution shifts. Optimistic-exploration bias of
Keramati 2020 could be combined with this approach.

---

## 5. Cross-Paper Synthesis: C51 → QR-DQN → IQN → Lim & Malik

The four papers reviewed above trace a single 5-year arc — *learn the return
distribution; then exploit it.* The first three (Bellemare et al. 2017,
Dabney et al. 2018a/b) build the **representation machinery** progressively
more expressive; the fourth (Lim & Malik 2022) finally addresses what happens
when we try to **act risk-sensitively** on the learned distribution and shows
that the obvious policy-side recipe is broken in a way that needs an operator
redesign, not just a parameterization upgrade.

### Representation evolution table

| Paper | Year | Representation of $Z(s,a)$ | Loss / projection | Quantile-level parameterization | Risk-sensitivity story |
|---|---|---|---|---|---|
| Bellemare et al. (C51) | 2017 | Categorical: 51 fixed atoms $\{z_i\}_{i=1}^{51}$ on a pre-specified support $[V_{\min}, V_{\max}]$, with learned probabilities $p_i(s,a)$ | Cross-entropy after Cramér / categorical projection $\Pi_{\mathcal{C}}$ | None — atoms are reward-axis-fixed, not quantile-indexed | Not addressed; agent acts greedily on the mean |
| Dabney et al. (QR-DQN) | 2018b | Quantile: $N = 200$ fixed quantile *levels* $\hat\tau_i = (i - 0.5)/N$, with learned quantile *values* $\theta_i(s,a)$ | Quantile-regression (Huber) loss $\rho_{\hat\tau_i}^\kappa$ | Levels fixed at $\hat\tau_i$; only values learned | Not addressed by the original paper; mean-greedy at action time |
| Dabney et al. (IQN) | 2018a | Implicit: $\theta_\tau(s,a) = f_\psi(\psi(s) \odot \phi(\tau))$ — a single neural network outputs the quantile *value* for *any* $\tau \in [0,1]$, conditioned by a cosine embedding of $\tau$ | Quantile-regression Huber loss summed over $N$ sampled $\tau \sim U(0,1)$ and $N'$ sampled $\tau' \sim U(0,1)$ | Levels are *sampled at training and test time* — continuous quantile function | First-class: distortion measures $g$ replace $\mathbb{E}_\tau$ to give $Q_g(s,a) = \mathbb{E}_{\tau \sim g}[\theta_\tau(s,a)]$; CPW, Wang, Pow, CVaR studied |
| Lim & Malik | 2022 | Same as QR-DQN ($N$ fixed quantiles, learned values) | Quantile-regression loss, same as QR-DQN | Levels fixed at $\hat\tau_i$ | First-class and policy-side rigorous: replaces the action-selection step with the augmented-MDP $W^U(x, s, a)$ at running threshold $s$, so the operator's fixed point is the static-CVaR-optimal distribution |

The *representation* arc moves from fixed atoms with fixed reward-axis support
(C51) → fixed quantile *levels* with learned values (QR-DQN) → continuous
quantile function with learned *levels-as-inputs* (IQN); the final paper picks
the QR-DQN parameterization for clean theory and grafts a redesigned operator
on top.

### Contraction story across the four papers

Each paper has a corresponding contraction or fixed-point result, but they live
in different metrics and assert different things:

- **Bellemare 2017 (C51).** The distributional Bellman *policy-evaluation*
  operator $\tilde{\mathcal{T}}^\pi$ is a $\gamma$-contraction in the maximal
  $p$-Wasserstein metric $\bar d_p(U, V) = \sup_{s,a} W_p(U(s,a), V(s,a))$ for
  any $p \in [1, \infty]$. The *optimality* operator $\tilde{\mathcal{T}}$ in
  general is **not** a contraction in any common distribution metric (multiple
  optima with different distributions but identical means). What C51 actually
  trains, however, is the *categorical projection* $\Pi_{\mathcal{C}}
  \tilde{\mathcal{T}}^\pi$ — and the projected operator's contraction status
  in Wasserstein had to be re-examined by Rowland et al. (2018) and
  later papers; C51 itself trains on KL after categorical projection, which is
  a heuristic not directly tied to a Wasserstein-contraction argument.
- **QR-DQN (Dabney 2018b).** The headline theoretical result of QR-DQN is that
  the *quantile-projection* operator $\Pi_{W_1} \tilde{\mathcal{T}}^\pi$ —
  Bellman update followed by projection onto the $N$-quantile representation
  with optimal $1$-Wasserstein error — *is* a $\gamma$-contraction in the
  maximal $1$-Wasserstein metric. This is *cleaner than C51* in the sense that
  the projection-and-update pair has a contraction guarantee that C51 lacks.
  The quantile-regression loss is then justified as a *sample-based stochastic
  approximation* of this projection, because the asymmetric pinball loss
  $\rho_\tau(u)$ has unique minimizer at the $\tau$-quantile of $u$'s
  distribution.
- **IQN (Dabney 2018a).** IQN inherits the QR-DQN $1$-Wasserstein contraction
  guarantee in spirit (because IQN is the implicit-parameterization continuous
  limit of QR-DQN — at any finite sampled set of $\tau$, IQN with the same
  pinball loss is doing QR-DQN-like quantile regression at those $\tau$).
  Whether the contraction holds *exactly* for the IQN parameterization, or
  whether function-approximation introduces caveats, is the open question
  flagged in IQN's Section 6. For the distortion-greedy action selection
  $Q_g(s,a) = \mathbb{E}_{\tau \sim g}[\theta_\tau(s,a)]$, IQN gives **no
  contraction guarantee at all** — it inherits the same "optimality operator
  not a contraction in $\mathcal{Z}$" caveat as C51.
- **Lim & Malik (2022).** This is precisely the gap the paper fills, for the
  CVaR special case. They prove that the redesigned operator
  $\tilde{\mathcal{T}}_\psi$ admits $Z^{\pi^*}$ as a **fixed point** when
  $\psi$ is chosen from the active set of the static-CVaR-optimal policy
  (Propositions 3 and 4). They do *not* prove contraction from arbitrary
  initialization — empirically it converges on small MDPs, but the contraction
  status in any explicit metric is open.

Read end-to-end, the chain is: **C51 establishes that distributional Bellman
operators contract in Wasserstein for policy evaluation** → **QR-DQN gives a
*projected* operator that exactly contracts in $W_1$ under quantile
projection** → **IQN inherits this for an implicit, continuous quantile
function, but only for the risk-neutral mean-greedy version; the
distortion-greedy version has no contraction theorem** → **Lim & Malik
identifies that the distortion-greedy operator can fail to find risk-sensitive
optima, redesigns the operator, and proves $Z^{\pi^*}$ is a fixed point for
the redesigned operator, leaving contraction itself open.**

### The IQN–FiLM connection (load-bearing for the project)

IQN's quantile architecture uses a *Hadamard-product fusion* of the state/CNN
embedding $\psi(s)$ with a *cosine embedding* $\phi(\tau)$ of the quantile
level $\tau$:

$$
Z(s, a;\,\tau) \;=\; f_\psi\!\bigl(\psi(s) \odot \phi(\tau)\bigr)_a,
\qquad
\phi_j(\tau) \;=\; \text{ReLU}\!\Bigl(\sum_{i=1}^{n} \cos(\pi\, i\, \tau)\, w_{ij} + b_j\Bigr),
$$

with $n = 64$ Fourier basis elements and the Hadamard fusion the empirical
winner over concatenation or residual connection in the IQN ablation
(Dabney 2018a §3.1). **This is structurally the same operation as FiLM
(feature-wise linear modulation, Perez et al. 2018)**: a scalar
conditioning input $\tau \in [0,1]$ is expanded by an MLP-on-cosine-features
into a per-feature gain vector $\phi(\tau) \in \mathbb{R}^d$, which then
multiplicatively modulates the per-feature representation $\psi(s) \in
\mathbb{R}^d$. The only differences from textbook FiLM are (i) the gain is
parameterized through a fixed cosine basis rather than a plain MLP, which is
in effect a strong prior favoring smooth $\tau$-dependence — important for
generalizing to unseen $\tau$ — and (ii) IQN uses only a multiplicative gain
($\gamma_{\text{FiLM}} \odot x$), not the FiLM additive bias term ($+\beta_{
\text{FiLM}}$); the additive term is omitted in IQN because the cosine basis
already includes the constant $\cos(0) = 1$.

This matters for the project because **the modulator-conditioned agent in this
codebase already uses FiLM to inject a neuromodulator scalar into the policy
network.** IQN is then a *literal architectural precedent for using FiLM to
condition the value head on a one-dimensional risk-tolerance scalar* — exactly
the same mechanism, just with the conditioning variable interpreted as a risk
level rather than a neuromodulator level. Lim & Malik then completes the chain
by giving the *policy-side argument* that turns "the agent knows
$Z(s,a;\tau)$ for any $\tau$" into a behaviorally meaningful risk-sensitive
policy (with the caveat that the proper recipe is the threshold-tracking
$\tilde{\mathcal{T}}_\psi$, not the naïve $\arg\max C_\alpha$).

### Project mapping: modulator as distortion / risk-level signal

If the project wants to translate this lineage into a concrete prototype,
the mapping is:

- **Modulator scalar** (currently FiLM-conditioning the policy / value head)
  $\longleftrightarrow$ **risk-level $\alpha$** or, more generally, the
  parameter of a one-parameter distortion family $g_\alpha$ (CVaR-$\alpha$,
  Wang-$\eta$, Pow-$\eta$, etc.). The agent's network is already wired to
  consume a scalar via FiLM; promoting that scalar from "neuromodulator
  level" to "risk-level" changes the *interpretation* of the input without
  changing the architecture.
- **Value head** $\longleftrightarrow$ **quantile head $Z(s, a;\,\tau)$**.
  This is a non-trivial architectural change — the head must output a
  quantile *value* indexed by $\tau$, not a scalar $Q$. The cheapest version
  is QR-DQN-style fixed quantile levels (Lim & Malik's setup); the more
  expressive version is IQN-style sampled-$\tau$ quantile networks where the
  FiLM input is *literally* $\tau$.
- **Loss** $\longleftrightarrow$ **quantile-regression Huber loss**
  $\rho_{\hat\tau_i}^\kappa(u)$, summed across quantile levels (or across
  sampled $\tau, \tau'$ pairs for the IQN variant).
- **Action selection at runtime** $\longleftrightarrow$ here is where Lim &
  Malik's negative result bites: if the project wants the agent's behavior
  to track the *static* CVaR objective at the modulator-set risk level, the
  Bellman-update action selection has to use the threshold-tracking
  $W^U(x, s, a)$ at running $s \leftarrow (s - r)/\gamma$, not just
  $\arg\max C_\alpha[U(x', a')]$. The cleanest interpretation is that the
  modulator sets both (i) the FiLM-input $\tau$ controlling which part of
  the return distribution the agent reads, and (ii) the *initial* threshold
  $s_0 = q_\alpha(U(x_0, a_0))$ in Algorithm 1; thereafter $s$ runs along
  with the trajectory.

**Handoff to `senior-developer` (if the project wants to prototype).** A
minimal proof-of-concept would: (a) keep the existing FiLM-conditioned policy
architecture, (b) add a quantile head outputting $N$ quantile values per
action (QR-DQN-style — simpler than full IQN sampling), (c) feed the
existing modulator scalar into the FiLM block (no new input wiring needed),
(d) replace the value-target step in the current actor-critic loss with the
quantile-regression Huber loss against either the mean-greedy QR-DQN target
or the Lim-&-Malik threshold-tracked target depending on the experiment
goal, and (e) add the running threshold $s$ to the replay-buffer rollout
storage if the threshold-tracked variant is used. The threshold-tracked
variant is the *correct* choice if the experimental claim is about
risk-sensitive behavior; the plain QR-DQN/IQN variant is the cheaper choice
if the claim is only about distributional-RL-style improved sample efficiency
with FiLM-conditioning on a risk-level scalar. **Recommend delegating to
`senior-developer` for an `issue_plan` that fixes the scope to one of these
two variants before implementation begins.**

---
