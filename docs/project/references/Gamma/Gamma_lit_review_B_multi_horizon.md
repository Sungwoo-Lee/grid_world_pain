# Gamma Literature Review — Batch B: Multi-Horizon / Multi-Timescale Value Estimation

**Curator:** literature-reviewer
**Created:** 2026-05-19
**Scope:** Three papers on simultaneously estimating value at many discount factors, treated as a candidate component for the precision-modulated agent.

---

## Question this batch asks

When a reinforcement-learning agent estimates "value" — the expected discounted future return — it normally commits to **one** discount factor $\gamma$, a number between 0 and 1 that controls how myopic vs. far-sighted the value estimate is. A $\gamma$ near 0 means "only the next step matters"; a $\gamma$ near 1 means "many future steps matter almost equally". Picking a single $\gamma$ is awkward because different sub-tasks of the same problem live on different horizons: a driving agent wants short-horizon predictions about collisions and long-horizon predictions about fuel; a foraging agent wants short-horizon predictions about the next bite and long-horizon predictions about the next refill.

These three papers propose **three different ways to estimate value at many horizons simultaneously**, rather than picking a single "right" $\gamma$:

1. **Sherstan et al. 2020 — γ-nets** treats the discount factor as **an extra input** to a single value network: the network learns one function $V(s, \gamma)$, queryable at any $\gamma$ at run time. This is the function-approximation route — one shared head, generalising over $\gamma$.

2. **Romoff et al. 2019 — TD($\Delta$)** trains **separate value heads for an ordered ladder of discount factors**, but parameterises each as the *difference* between its $\gamma$ and the next-shorter $\gamma$, $V_{\gamma_k} = V_{\gamma_{k-1}} + W_{\gamma_{k-1}, \gamma_k}$. Long-horizon estimation, normally high-variance, gets to lean on the easier short-horizon estimates as a base.

3. **Fedus et al. 2019 — Hyperbolic Q-learning** also trains many parallel value heads at different $\gamma$, but then **integrates over them** via the identity $V_{\text{hyp}}(s) = \int_0^1 V_\gamma(s)\, d\gamma$ to recover the hyperbolic discounting curve seen in animals. As a side effect, multi-$\gamma$ prediction acts as an **auxiliary task** that improves representation learning even when the policy itself uses only one $\gamma$.

**Why this matters for the project.** The precision-modulated agent under construction (FiLM / hypernet conditioning on an interoceptive modulator) currently consumes a single $\gamma$. If the modulator is functionally analogous to serotonin's role in temporal-horizon control (Doya), then "value at multiple horizons, conditioned on modulator level" becomes a natural design space. The three routes above are the three most-cited concrete instantiations of that design space — γ-as-input, γ-as-ladder, γ-as-auxiliary-bag. This review's job is to surface the equations and the engineering trade-offs cleanly enough that the project's planners can pick the right route for the next experiment.

---

## Table of contents

1. [Sherstan et al. 2020 — Γ-nets: Generalizing Value Estimation over Timescale](#1-sherstan-et-al-2020--%CE%93-nets-generalizing-value-estimation-over-timescale)
2. [Romoff et al. 2019 — Separating Value Functions across Time-Scales (TD(Δ))](#2-romoff-et-al-2019--separating-value-functions-across-time-scales-td%CE%94)
3. [Fedus et al. 2019 — Hyperbolic Discounting and Learning over Multiple Horizons](#3-fedus-et-al-2019--hyperbolic-discounting-and-learning-over-multiple-horizons)
4. [Cross-paper synthesis](#4-cross-paper-synthesis)

---

## 1. Sherstan et al. 2020 — Γ-nets: Generalizing Value Estimation over Timescale

**Venue:** AAAI-20 (Thirty-Fourth AAAI Conference on Artificial Intelligence).
**Authors:** Craig Sherstan, Shibhansh Dohare, James MacGlashan, Johannes Günther, Patrick M. Pilarski (Univ. Alberta + Cogitai).
**PDF:** `docs/project/references/Gamma/sources/Sherstan et al. 2020 - Gamma-nets - Generalizing value estimation over timescale.pdf`
**arXiv long version:** Sherstan et al. 2019, `arXiv:1911.07794`.

### Phase 1 — Foundational overview (undergrad-level)

#### Introduction

A standard value network in reinforcement learning answers a single question: "what is the discounted future return from state $s$, under one fixed discount $\gamma$?" That's awkward when an agent wants predictions at several horizons — e.g. "will my motors overheat in the next 30 seconds?" *and* "will my battery last through the whole task?" The traditional fix is to train a separate predictor per horizon, which means one network per question and no sharing between them.

The Sherstan paper proposes a different fix: train **one** network that takes the timescale $\gamma$ as an extra input alongside the state $s$, so the same network can be queried at any $\gamma$. They call this a **Γ-net**. The trick is built on top of *General Value Functions* (GVFs) from Sutton et al. 2011 — a GVF is a value-like prediction about *any* cumulant signal (motor current, bumper bit, joint speed, reward), not just reward. Γ-nets generalise GVFs over the third parameter of the GVF spec (the discount), in the same way that Universal Value Function Approximators (UVFAs, Schaul et al. 2015) earlier generalised them over the first parameter (the goal).

#### Key findings

- **Architecture.** Feed the state $s$ *and* the discount $\gamma$ into one network; train the same weights $w$ to predict $V(s, \gamma; w)$ correctly at many $\gamma$ values drawn each step. The TD target for one drawn $\gamma_k$ is the standard target with $\gamma_k$ plugged in: $\delta_{t;\gamma_k} = C_{t+1} + \gamma_k V(S_{t+1}, \gamma_k; w) - V(S_t, \gamma_k; w)$. Sum the TD losses across the drawn $\gamma_k$.
- **Two practical knobs.** (i) Provide both $\gamma$ **and** the expected-lifetime $\tau = 1/(1-\gamma)$ as inputs — they are non-linear transforms of each other, and each is well-resolved on a different part of the timescale range. (ii) Scale the prediction by $(1-\gamma)$ so the magnitudes at long horizons don't dominate the loss.
- **Sampling matters.** Drawing the training $\gamma$ values uniformly from $\gamma$-scale over-weights short horizons; uniformly from $\tau$-scale over-weights long horizons; drawing *half from each* is the recommended default.
- **Empirical scope.** Three demos: a synthetic square-wave signal (linear function approximation), a real robot-arm task predicting joint speed under human teleoperation, and policy evaluation in Atari (deep nets, Rainbow agent). In every case Γ-nets get accuracy competitive with — but slightly worse than — networks trained for one fixed $\gamma$, in exchange for being queryable at any $\gamma$.

#### Initial takeaway

Γ-nets are the simplest answer to "how do I get value at many horizons from one network?" — make $\gamma$ an input. The cost is a small accuracy hit vs. a horizon-specialised predictor; the benefit is unlimited query resolution from a single set of weights. The paper's most useful artefacts for a downstream user are the **engineering recipe** (use both $\gamma$ and $\tau$ as input, scale the loss by $(1-\gamma)$, sample training $\gamma$ from both scales) and the **negative finding** that for fixed and known horizons of interest, a small set of horizon-specialised heads plus linear interpolation can match or beat a Γ-net.

### Phase 2 — Graduate-level deep dive

#### Background: the GVF triple and where γ-nets live

A General Value Function is a function-approximated prediction defined by three pieces: a policy $\pi$, a cumulant signal $C$ (any scalar sensor, not necessarily reward), and a discount $\gamma$. The GVF predicts

$$
V^\pi_C(s; \gamma) \;=\; \mathbb{E}_\pi\!\left[\sum_{k=0}^{\infty} \gamma^k\, C_{t+k+1} \,\middle|\, S_t = s\right].
$$

UVFA (Schaul et al. 2015) generalises across the goal embedding; Xu, van Hasselt, Silver 2018 already provided $\gamma$ as input in a meta-learning context. The Γ-nets contribution is to make $\gamma$ a *generic* input axis for any GVF, with the practical claim that one network can cover many $\gamma$ at small accuracy cost.

#### The Γ-net update rule

Let the network parameters be $w$. For a transition $(S_t, A_t, S_{t+1}, C_{t+1})$ the agent picks a **set of timescales** $\Gamma_t = \{\gamma_1, \dots, \gamma_K\}$ on which to train this step. For each $\gamma_k \in \Gamma_t$ the standard one-step TD error is

$$
\boxed{\;\delta_{t;\gamma_k} \;=\; C_{t+1} \;+\; \gamma_k\, V(S_{t+1}, \gamma_k; w) \;-\; V(S_t, \gamma_k; w).\;}
$$

(eq. 1 in the paper). The total semi-gradient update is the sum over $\gamma_k \in \Gamma_t$:

$$
w \;\leftarrow\; w \;+\; \alpha \sum_{\gamma_k \in \Gamma_t} \delta_{t;\gamma_k}\, \nabla_w V(S_t, \gamma_k; w).
$$

Two design choices about $\Gamma_t$ now follow.

#### Why naïve uniform sampling of γ fails — and the $\gamma$ vs. $\tau$ duality

The expected number of timesteps before "termination" under discount $\gamma$ — equivalently, the expected lifetime / time-to-horizon — is

$$
\tau \;=\; \frac{1}{1 - \gamma}.
$$

This relationship is **strongly non-linear** for $\gamma$ near 1: small changes in $\gamma$ at the high end produce huge changes in $\tau$. Practical consequences:

| Sampling scheme | Effect |
|---|---|
| Uniform $\gamma \in [0, 1)$ | Almost all mass at short horizons ($\tau$ small). |
| Uniform $\tau \in [\tau_{\min}, \tau_{\max}]$ | Very little mass at short horizons. |
| Half from each | Reasonable coverage across both ends. |

The paper recommends drawing $|\Gamma_t|$ samples per step from a mixture: a fixed pair $\{\tau_{\min}, \tau_{\max}\}$ at the endpoints plus equal-count draws from $\gamma$-uniform and $\tau$-uniform. For the Atari setup they use $|\Gamma_t| = 8$ with $\tau \in \{1, 100\}$ pinned and six additional samples (three from $\gamma$, three from $\tau$).

Similarly, providing **both** $\gamma$ and $\tau$ as **network inputs** matters: each scale linearises a different end of the horizon space, so giving the net both lets it learn its representation in whichever space discriminates best for the local query. Empirically (Fig. 7 in the paper) "input = $\gamma$ only" is universally worse than "input = $\tau$ only" or "input = both", and "input = both" is best at the shortest horizons.

#### Loss scaling: keeping return magnitudes comparable across γ

Because returns at large $\gamma$ accumulate over many more timesteps, the magnitude of the target $G_t = \sum_k \gamma^k C_{t+k+1}$ can vary by orders of magnitude across the $\gamma$ values in $\Gamma_t$. Without scaling, the large-$\gamma$ targets dominate the gradient.

The paper's fix is to learn the **scaled** value $f(s, \gamma; w) \approx (1-\gamma) V(s, \gamma)$:

$$
f(s, \gamma_k; w) \;=\; \mathbb{E}_\pi\!\left[\sum_{t=0}^{\infty} \gamma_k^t (1 - \gamma_k)\, C_{t+1} \,\Big|\, S_0 = s\right],
$$

then recover the unscaled value as $V(s, \gamma_k; w) = f(s, \gamma_k; w) / (1 - \gamma_k)$. The TD error correspondingly becomes

$$
\boxed{\;\delta_{t;\gamma_k}^{\text{scaled}} \;=\; (1 - \gamma_k)\Big(\,C_{t+1} + \gamma_k\, V(S_{t+1}, \gamma_k) - V(S_t, \gamma_k)\,\Big).\;}
$$

The factor $(1-\gamma)$ pulls long-horizon errors down toward the same order of magnitude as short-horizon errors. The empirical result is mixed: scaling helps on Centipede@25M cleanly (lower MSE, lower variance across all probe horizons; Fig. 9), but on other Atari games it improves short horizons at the *cost* of mid/long horizons. The recommendation is "try with and without".

#### How γ enters the network — embedding comparison

The paper compares four ways to fuse $\gamma$ (and $\tau$) with the agent's feature vector $\phi$ at the value-head input:

| Method | Definition | Result |
|---|---|---|
| **direct** | $\nu = [\phi, \gamma]$ (concatenation). | Simplest, lowest compute, performance ≈ best. |
| **`l_embed`** | $\gamma$ passes through a small linear FC ($\to$ length 16), then concatenated: $\nu = [\phi, \xi(\gamma)]$. | Slightly lower variance, like Xu, van Hasselt, Silver 2018. |
| **Hadamard `hl_embed`** | $\xi(\gamma)$ matches $|\phi|$, element-wise multiply: $\nu = \phi \odot \xi(\gamma)$. | Similar performance. |
| **Matrix** | $\gamma$ produces a square matrix $\Xi(\gamma)$ via an FC layer; $\nu = \phi^\top \Xi$. | Slower; no clear gain. |

The Hadamard variant is identical in spirit to a FiLM modulation (scale-only) — a concrete point of contact with the project's existing FiLM agent. The paper concludes there is no universal winner and recommends `direct` as the simplest default.

#### Numerical results — three settings

1. **Square wave.** Linear function approximation, tile-coded inputs, TD(0). The Γ-net with both $\gamma$/$\tau$ inputs and loss scaling tracks the true return across all probe horizons (Fig. 2). MSE and variance dropped further with the "draw from both scales" mixture (Fig. 3).
2. **Robot arm (real hardware).** A human teleoperated a 2-DOF arm; the cumulant was joint speed of the shoulder. 30 Hz updates, $\tau$ in $[1, 100]$ steps. The Γ-net *matches or beats* a baseline per-horizon network on the displayed timescale (Fig. 4b) — the negative-transfer worry didn't materialise here.
3. **Atari Centipede @ 25M frames.** Policy evaluation under a fixed Rainbow policy (Hessel et al. 2018) in evaluation $\epsilon$-greedy mode ($\epsilon = 10^{-4}$). Ground truth is 1000 Monte-Carlo rollouts per evaluation point. The Γ-net's predictions for probe $\tau \in \{1, 2, 5, 10, 20, 40, 60, 80, 100\}$ tracked the MC truth in shape (Fig. 5), and the steady-state MSE was within a small factor of per-horizon baselines. But — important caveat — **per-horizon networks plus linear interpolation between bracketing horizons matched or *beat* the Γ-net** when ground-truth horizons were known in advance (Fig. 10).

#### Limitations as authors state them

- **Fixed discount only.** The general GVF spec allows transition-dependent discount $\gamma_{t+1} = \gamma(S_t, A_t, S_{t+1})$ (White 2017), but Γ-nets handle only a single scalar $\gamma$ per query. Extending to state-dependent termination functions is open.
- **No clear winner among scaling / embedding / sampling variants.** Beyond the broad recommendations (use both $\gamma$ and $\tau$ as input, sample from both scales), the paper repeatedly flags that the best configuration is task-dependent.
- **Interpolation baseline can win.** If the practitioner already knows the horizons of interest, a small bag of horizon-specialised networks plus linear-in-$\tau$ interpolation can outperform a Γ-net at those horizons. Γ-nets pay for queryability at *arbitrary* $\gamma$.
- **Author's revised motivation.** Originally the paper hoped Γ-nets would serve as a predictive state representation for downstream control; the discussion concludes that "multiple heads with predictions at fixed timescales and let the policy network learn to generalize over those predictions as it needed" might actually be the better path forward — foreshadowing Romoff and Fedus.

#### Where γ-nets connect to the other two papers in this batch

- The discussion section explicitly names Romoff et al. 2019 as a complement: Romoff's cascade of head differences could reduce the long-horizon variance that Γ-nets suffer from.
- Fedus et al. 2019 is cited as both complementary and a use case: Fedus uses a bag of fixed-$\gamma$ heads to integrate into hyperbolic discounting, and Sherstan note that a Γ-net could in principle serve as the same basis function with *one* network instead of a bag.

### Appendix: Section-by-section backbone (Sherstan et al. 2020)

**Step 1 — Section list (original order).** Abstract → Value Functions and Timescale (intro) → Background (MDP, TD, GVF) → Generalizing over Timescale → Experiments (Square Wave; Robot Arm; Atari Environment incl. Training, Evaluation, Plotting, Embedding Comparison, Timescale Input Comparison, Distribution Comparison, Loss Scaling, Estimation by Interpolation) → Discussion → Conclusion → References.

**Step 2/3 — Per-section content.**

- **Abstract.** Γ-nets generalise value-function estimation over timescale by providing $\gamma$ as a network input. Trained one transition at a time across many $\gamma_k$; demonstrated on a square wave, a robot arm, and Atari policy evaluation. Small accuracy cost vs. fixed-timescale predictors.
- **Value Functions and Timescale (intro).** Motivates multi-timescale prediction via GVFs and the "nexting" idea (Modayil, White, Sutton 2014). Most prior value estimation has been single-fixed-$\gamma$; explicit need for a method that supports arbitrary $\gamma$ at query time. Cites Schaul UVFA (generalises over goals) and Xu, van Hasselt, Silver 2018 (meta-learning $\gamma$).
- **Background.** Standard MDP $\langle S, A, p, R, \gamma\rangle$, return $G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \dots$, GVF substitutes cumulant $C$ for reward $R$, value $V^\pi(s) = \mathbb{E}_\pi[G_t | S_t = s]$, semi-gradient TD update on $\delta_t = C_{t+1} + \gamma V(S_{t+1}) - V(S_t)$. Function approximation $V(s; w)$.
- **Generalizing over Timescale.** Goal: predict $V$ for any valid $\gamma$. Architecture: $V(s, \gamma; w)$. Per-step training set $\Gamma_t$ of $\gamma_k$. TD error eq. (1). $\Gamma_t$ choice matters because of the $\tau = 1/(1-\gamma)$ non-linearity. Recommends giving both $\gamma$ and $\tau$ as input and possibly sampling from both. Loss-scaling derivation: predict $f(s, \gamma_k) = \mathbb{E}[\sum_t \gamma_k^t (1-\gamma_k) C_{t+1}]$, recover $V = f/(1-\gamma_k)$, scaled TD error.
- **Experiments: Square Wave.** 100-step square wave $\{-1, 1\}$, tile-coded, TD(0) with LFA. Default: both inputs, $|\Gamma_t| = 6$, pinned $\tau \in \{1, 100\}$, four additional draws (two each $\gamma$- and $\tau$-uniform), loss scaling on. Plots: input comparison, distribution comparison, scaling comparison.
- **Experiments: Robot Arm.** Operator joysticks a 2-DOF arm around a wire maze; cumulant = shoulder/elbow joint speed; 30 Hz. Same LFA + TD(0) setup. Γ-net matches or beats the per-horizon baseline.
- **Experiments: Atari.** Rainbow policy frozen at 25M frames as data generator; Γ-net is a 5-layer MLP $[512, 256, 128, 16, 1]$ over the last conv-feature layer $\phi$. Prioritised replay; n-step returns; $|\Gamma_t| = 8$ with $\tau \in \{1, 100\}$ pinned. Probe horizons $\tau = \{1, 2, 5, 10, 20, 40, 60, 80, 100\}$. Ground truth: 2000 MC rollouts per evaluation point. Tested embedding methods (direct/`l_embed`/Hadamard/matrix), input choice ($\gamma$/$\tau$/both), distribution ($\gamma$/$\tau$/both), loss scaling on/off, interpolation baseline.
- **Discussion.** Reliable for both reward and sensorimotor prediction. Practical recommendations: direct embedding; both $\gamma$ and $\tau$ inputs; sample from both scales; loss scaling situational. Open: transition-dependent $\gamma$; connection to Romoff (cascade) and Fedus (hyperbolic / auxiliary tasks). Reconsidered original motivation: multiple fixed-$\gamma$ heads may actually be preferable for representation learning.
- **Conclusion.** Γ-nets are a simple, single-network technique for multi-horizon value estimation, complementary to UVFAs and SR; promising for predictive-state-representation and lifelong-learning research.

**Step 4 — Most relevant sections expanded.** Generalizing-over-Timescale (architecture, eq. 1, $\gamma$/$\tau$ duality, loss-scaling derivation) and the Atari embedding-comparison sub-experiment (because the Hadamard `hl_embed` variant is structurally identical to a FiLM-style scale modulation, which is the project's existing conditioning mechanism). Expanded forms are folded into the Phase 2 sections above.

---

## 2. Romoff et al. 2019 — Separating Value Functions across Time-Scales (TD(Δ))

**Venue:** ICML 2019 (Proceedings of the 36th International Conference on Machine Learning, PMLR 97).
**Authors:** Joshua Romoff*, Peter Henderson*, Ahmed Touati, Emma Brunskill, Joelle Pineau, Yann Ollivier (MILA / McGill, FAIR, Stanford).
**PDF:** `docs/project/references/Gamma/sources/Romoff et al. 2019 - Separating value functions across time-scales.pdf` (54 pages; ~10-page main body + appendices with proofs and per-game results).
**Code:** `github.com/facebookresearch/td-delta`.

### Phase 1 — Foundational overview (undergrad-level)

#### Introduction

Long-horizon value estimation in deep reinforcement learning is hard for a specific reason: when the discount $\gamma$ is close to 1, the target $G_t = \sum_k \gamma^k r_{t+k+1}$ accumulates noise from many future rewards, and the variance of the target balloons. The standard workaround is to use a smaller $\gamma$ — but that *biases* the agent toward short-sighted behaviour and can destroy performance in tasks (Atari, OpenAI Five) where the "real" goal lives at a long horizon.

Romoff et al. observe a simple fact: a value function at large $\gamma$ already *contains* the information that would be captured by a value function at any smaller $\gamma'$ — because $V_\gamma$ is a strictly more inclusive sum than $V_{\gamma'}$. So instead of training $V_\gamma$ as a single monolithic estimator, why not learn it as an **ordered cascade of deltas**?

$$
V_{\gamma_z}(s) \;=\; V_{\gamma_{z-1}}(s) \;+\; W_z(s), \qquad W_z(s) \;:=\; V_{\gamma_z}(s) - V_{\gamma_{z-1}}(s).
$$

Each $W_z$ is a separate (small) value head with its own discount, its own learning rate, its own $k$-step return length. The cheap short-horizon head $W_0 = V_{\gamma_0}$ converges fast, and the deeper-horizon heads only need to learn the **incremental** information beyond the next-shorter head's prediction. The authors call the method **TD($\Delta$)** — TD with deltas.

#### Key findings

- **Theoretical equivalence.** Under specific symmetric hyperparameter choices (equal learning rate, $\lambda_z \gamma_z = \lambda \gamma$ for all $z$, and consistent initialisation), TD($\Delta$) with linear function approximation is **exactly equivalent** to standard TD($\lambda$). So the decomposition is "free" in the equivalence regime — and the benefits come from *breaking* the symmetry (using shorter $k$-step returns / smaller $\lambda$ for the shorter-horizon heads, which is principled because shorter horizons admit lower-variance estimation).
- **Theoretical variance reduction.** A bias-variance analysis (Theorem 4) extending Kearns & Singh 2000 shows that when $k_z$ values are allowed to differ across heads, the *variance* term in the error bound on $\sum_z \Delta^z_t$ can be strictly smaller than the variance term for monolithic phased TD($\lambda$), at the cost of a controlled "bias introduction" term coming from compounding bias through the cascade. The rule of thumb that drops out: pick $k_z \approx 1/(1-\gamma_z)$ so that $\gamma_z^{k_z}$ is bounded by a constant ($\approx 1/e$); double the effective horizon each step ($\gamma_{z+1} \approx (\gamma_z + 1)/2$).
- **Empirical wins on dense-reward Atari.** Plugged into PPO (the authors' main empirical vehicle), TD($\Delta$) matches or significantly outperforms standard PPO on 6/9 dense-reward "Hard" Atari games (Qbert, MsPacman, Hero, Frostbite, BankHeist, Amidar; Alien borderline). It loses on 2 sparse-reward games (Zaxxon, WizardOfWor) — fixable by tuning $\gamma_0$ upward.
- **Free interpretability.** Plotting the individual $W_z$ traces along a single Atari rollout (Fig. 2 in the paper) shows that the long-horizon $W_Z$ drops smoothly toward a "lost life" event many steps before it happens, while short-horizon $W_0$ tracks moment-to-moment reward. The decomposition exposes what each timescale "knows" — useful for diagnosis and for anytime / partial-deployment scenarios.

#### Initial takeaway

TD($\Delta$) is the **horizon-cascade** answer to multi-horizon value estimation: keep value heads separate but arrange them so each head only learns what's *new* at its horizon. It's strictly more flexible than monolithic TD ($\Delta$ contains TD as a special case), it admits the per-head tuning that monolithic TD cannot, and it gives variance reduction for free in the right regime. The cost is more outputs to manage and a sensible $\{\gamma_z\}$ schedule — and the recommended doubling schedule mostly removes the tuning burden.

### Phase 2 — Graduate-level deep dive

#### Background: monolithic TD with function approximation

For an MDP $(\mathcal{S}, \mathcal{A}, P, r)$ and policy $\pi$, the discounted value at $\gamma$ is

$$
V^\pi_\gamma(s) \;=\; \mathbb{E}_\pi\!\left[\sum_{t=0}^\infty \gamma^t r_t \,\Big|\, s_0 = s\right].
$$

It is the fixed point of the Bellman operator $T^\pi V_\gamma = r^\pi + \gamma P^\pi V_\gamma$. With function approximation $\hat V_\gamma(s; \theta) = \langle \theta_\gamma, \phi(s)\rangle$ and one-step TD error $\delta_t^\gamma = r_t + \gamma \hat V_\gamma(s_{t+1}) - \hat V_\gamma(s_t)$, the GAE / $\lambda$-return target is

$$
G_t^{\gamma, \lambda} \;=\; \hat V_\gamma(s_t) + \sum_{k=0}^\infty (\lambda \gamma)^k \delta_{t+k}^\gamma, \qquad A(s_t) = \sum_{k=0}^\infty (\lambda \gamma)^k \delta_{t+k}^\gamma.
$$

The advantage $A$ feeds the standard actor-critic policy loss $L(\omega) = \mathbb{E}[-\log \pi(a, s; \omega)\, A(s)]$ and, with the clipping trick, the PPO loss.

#### The TD(Δ) decomposition

Pick an ordered ladder of discounts $\Delta := (\gamma_0, \gamma_1, \dots, \gamma_Z)$ with $\gamma_0 < \gamma_1 < \dots < \gamma_Z$, each defining a value function $V_{\gamma_z}$. Define the **delta functions**

$$
W_z \;:=\; V_{\gamma_z} - V_{\gamma_{z-1}}, \qquad W_0 \;:=\; V_{\gamma_0}.
$$

By construction

$$
\boxed{\;V_{\gamma_z}(s) \;=\; \sum_{i=0}^{z} W_i(s).\;}
$$

The key derivation is that **each $W_z$ has its own Bellman-like equation**. Starting from the standard Bellman equations for $V_{\gamma_z}$ and $V_{\gamma_{z-1}}$:

$$
\begin{aligned}
W_z(s_t) &= V_{\gamma_z}(s_t) - V_{\gamma_{z-1}}(s_t) \\
&= \mathbb{E}\!\left[(r_t + \gamma_z V_{\gamma_z}(s_{t+1})) - (r_t + \gamma_{z-1} V_{\gamma_{z-1}}(s_{t+1}))\right] \\
&= \mathbb{E}\!\left[\gamma_z\, (W_z(s_{t+1}) + V_{\gamma_{z-1}}(s_{t+1})) - \gamma_{z-1}\, V_{\gamma_{z-1}}(s_{t+1})\right] \\
&= \mathbb{E}\!\left[(\gamma_z - \gamma_{z-1})\, V_{\gamma_{z-1}}(s_{t+1}) + \gamma_z\, W_z(s_{t+1})\right].
\end{aligned}
$$

So $W_z$ satisfies a **Bellman-like recursion with decay factor $\gamma_z$ and "pseudo-rewards" $(\gamma_z - \gamma_{z-1}) V_{\gamma_{z-1}}(s_{t+1})$** — i.e., each $W_z$ looks like its own RL problem, where the reward is supplied by the value function of the next-shorter horizon. Crucially: $V_{\gamma_{z-1}}(s_{t+1}) = \sum_{i \le z-1} W_i(s_{t+1})$, so all heads can be trained from a single forward pass through the cascade — no separate $V_{\gamma_{z-1}}$ network needed beyond the heads themselves.

#### Multi-step TD(Δ): unrolling the cascade

The single-step version generalises to a $k_z$-step return. By iteratively unrolling the recursion:

$$
W_z(s_t) \;=\; \mathbb{E}\!\left[\sum_{i=1}^{k_z - 1} (\gamma_z^i - \gamma_{z-1}^i)\, r_{t+i} \;+\; (\gamma_z^{k_z} - \gamma_{z-1}^{k_z})\, V_{\gamma_{z-1}}(s_{t+k_z}) \;+\; \gamma_z^{k_z}\, W_z(s_{t+k_z})\right].
$$

Each $W_z$ accumulates a *difference of geometric weights* on the rewards plus a bootstrap that splits into "what the shorter-horizon estimator already knows" $(V_{\gamma_{z-1}})$ and "what is genuinely new at this horizon" $(W_z)$. **The same trajectory rolls out all $W_z$ at once**, and the only per-head choices are the bootstrap depth $k_z$ and the head's learning rate $\alpha_z$.

The TD($\lambda, \Delta$) variant defines per-head $\lambda$-returns with their own TD errors

$$
\delta_t^z := (\gamma_z - \gamma_{z-1})\, \hat V_{\gamma_{z-1}}(s_{t+1}) + \gamma_z\, \hat W_z(s_{t+1}) - \hat W_z(s_t),
$$

and target

$$
G_t^{z, \lambda_z} := \hat W_z(s_t) + \sum_{k=0}^\infty (\lambda_z \gamma_z)^k \delta_{t+k}^z.
$$

Note this generalises naturally: when $\lambda_z = 1$ and $\gamma_z = \gamma_Z$ identical across heads, the cascade collapses to monolithic TD($\lambda$).

#### Policy-gradient integration: TD(λ, Δ) with GAE

For the PPO-based experiments, the policy uses an aggregated advantage built from the **summed** value head $\sum_z \hat W_z = \hat V_{\gamma_Z}$ and the largest discount $\gamma_Z$:

$$
A^\Delta(s_t) := \sum_{k=0}^{T-1} (\lambda_Z \gamma_Z)^k \delta_{t+k}^\Delta, \qquad \delta_{t+k}^\Delta := r_t + \gamma_Z \sum_z \hat W_z(s_{t+1}) - \sum_z \hat W_z(s_t).
$$

This is exactly the standard PPO advantage with $V_\gamma$ replaced by the cascade-summed estimator. The policy update is unchanged (eq. 4 of the paper). Algorithm 2 in the paper shows the full PPO-TD($\lambda, \Delta$) loop.

#### Theoretical results

**Theorem 1 (equivalence under symmetry).** With linear function approximation $\hat V_\gamma(s) = \langle \theta^\gamma, \phi(s)\rangle$, $\hat W_z(s) = \langle \theta^z, \phi(s)\rangle$, equal learning rates $\alpha_z = \alpha$, equal effective decay $\lambda_z \gamma_z = \lambda \gamma$ for all $z$, and initialisation $\sum_z \theta_0^z = \theta_0^\gamma$, the TD($\lambda, \Delta$) update is **exactly** equivalent to TD($\lambda$):

$$
\sum_{z=0}^Z \theta_t^z \;=\; \theta_t^\gamma \qquad \forall t.
$$

Proof by induction; the inductive step requires showing that $\sum_z \delta_k^z = \delta_k^\gamma$ — a telescoping calculation in which the cascade's pseudo-rewards $(\gamma_z - \gamma_{z-1}) V_{\gamma_{z-1}}$ cancel against the bootstrap shifts to recover the standard one-step TD error.

**Theorem 2 (operator contraction beyond λ = 1).** The TD($\lambda$) operator $T_\lambda V = V + (I - \lambda \gamma P)^{-1}(TV - V)$ is a max-norm contraction for any $\lambda \in [0, \tfrac{1+\gamma}{2\gamma})$, with contraction coefficient $\gamma |1 - \lambda| / (1 - \lambda \gamma)$. This matters because the equivalence condition $\lambda_z \gamma_z = \lambda \gamma$ implies $\lambda_z = \lambda \gamma / \gamma_z > \lambda$ for $\gamma_z < \gamma$, potentially $> 1$ — but Theorem 2 confirms that as long as $\lambda_z < (1 + \gamma_z) / (2 \gamma_z)$, the per-head operator is still well-defined and contractive.

**Theorem 4 (bias-variance bound).** For phased TD($\Delta$) under the assumption $\gamma_0 \le \gamma_1 \le \dots \le \gamma_Z = \gamma$ and $k_0 \le k_1 \le \dots \le k_Z = k$, with $\epsilon = \sqrt{2 \log(2k/\delta)/n}$, with probability $1 - \delta$:

$$
\sum_{z=0}^Z \Delta_t^z \;\le\; \underbrace{\epsilon \frac{1 - \gamma^k}{1 - \gamma} \;+\; \epsilon \sum_{z=0}^{Z-1} \frac{\gamma_z^{k_{z+1}} - \gamma_z^{k_z}}{1 - \gamma_z}}_{\text{variance reduction}} \;+\; \underbrace{\sum_{z=0}^{Z-1}\sum_{u=0}^z (\gamma_z^{k_z} - \gamma_z^{k_{z+1}}) \Delta_{t-1}^u \;+\; \gamma^k \sum_{z=0}^Z \Delta_{t-1}^z}_{\text{bias introduction}}.
$$

The first square bracket is the variance term inherited from Kearns & Singh's monolithic bound, plus a **correction** that can be made *negative* by choosing $k_z < k_{z+1}$ — i.e., shorter bootstrap depths at shorter horizons reduce variance. The second square bracket is the bias compounding from the cascade; it is bounded by $\sum_u \Delta_{t-1}^u$ and disappears at the equivalence regime $k_z = k$. The take-away rule of thumb: pick $k_z \approx 1/(1 - \gamma_z)$ so that $\gamma_z^{k_z} \le 1/e$ is bounded, then double the horizon at each step ($\gamma_{z+1} = (\gamma_z + 1)/2$) for a logarithmic number of heads.

#### Empirical results

**Tabular ring MDP.** Same 5-state ring MDP as Kearns & Singh 2000. The authors compare TD($\Delta$) with the doubling schedule against single-estimator TD across $\gamma_Z \in \{0.75, 0.875, 0.9375, 0.96875, 0.984375, 0.992, 0.996\}$ (effective horizons $\{4, 8, 16, 32, 64, 125, 250\}$). For matched $k$ across all heads, the two are statistically identical (as Theorem 1 promises); for tailored $k_z$, TD($\Delta$) is statistically equal to or significantly better than the single-estimator baseline in every cell tested, with the gap growing for larger $\gamma_Z$ and $k$.

**Dense-reward Atari (Bellemare et al. 2016 "Hard" games).** Two variants:
1. **PPO-TD($\lambda, \Delta$):** $\gamma_Z = 0.99$, doubling schedule, $\lambda_z$ set such that $\lambda_z \gamma_z = \lambda_Z \gamma_Z$ (per Theorem 1) — i.e., still in the equivalence regime.
2. **PPO-TD($\hat \lambda, \Delta$):** as above but capping $\lambda_z \le 1$ — i.e., breaking equivalence in favour of more aggressive bootstrapping at shorter horizons.

| Game | PPO | PPO-TD(λ,Δ) | PPO-TD($\hat\lambda$,Δ) | Reward density |
|---|---|---|---|---|
| Zaxxon | 7366 ± 223 | 396 ± 210 | 3291 ± 812 | 1.15 |
| WizardOfWor | 3408 ± 193 | 2118 ± 138 | 2440 ± 89 | 1.07 |
| Qbert | 11735 ± 387 | **13428 ± 333** | **13092 ± 430** | 12.26 |
| MsPacman | 1888 ± 111 | **2273 ± 67** | **2241 ± 78** | 13.27 |
| Hero | 21038 ± 972 | **29074 ± 512** | **29014 ± 764** | 13.46 |
| Frostbite | 294 ± 5 | 292 ± 7 | 304 ± 21 | 5.04 |
| BankHeist | 1190 ± 3 | 1183 ± 13 | 1166 ± 5 | 6.30 |
| Amidar | 575 ± 54 | **731 ± 30** | 672 ± 45 | 4.63 |
| Alien | 1315 ± 70 | 1606 ± 112 | 1663 ± 113 | 11.33 |

(Bold = significantly better than baselines; reward density = non-zero-reward frequency per 100 steps under the baseline policy.) The pattern is sharp: TD($\Delta$) wins on dense-reward games and loses on the two sparsest. The authors hypothesise that the cascade's myopic bias slows initial learning when rewards are rare, which can be fixed by raising $\gamma_0$ to reduce the compounding-bias term in Theorem 4. Fig. 3 of the paper confirms that pushing $\gamma_0$ up (using only $\{0.3, 0.99\}$ or $\{0.92, 0.99\}$ as the ladder) recovers parity with PPO on Zaxxon / WizardOfWor.

#### Interpretability: per-head traces (Fig. 2)

A single Qbert rollout, plotted as $W_0, W_1, \dots, W_Z$ alongside the instantaneous reward, shows the longest-horizon head $W_Z$ falling smoothly toward a "lost life" event many timesteps ahead of the event itself, while $W_0$ tracks immediate reward. **The decomposition is not just an estimation trick — it lets you read off which horizon is responsible for which part of the value.** The authors flag this for production / deployment use: a practitioner can inspect whether the policy's prediction at horizon $z$ is "worth the cost" before deciding to run further inference. They also note it gives the algorithm an *anytime* character — at any time during training, the partial sum $\sum_{i \le z'} W_i$ for the converged-so-far heads is a valid approximation to $V_{\gamma_{z'}}$, even if larger-horizon heads have not converged yet.

#### Engineering recipe

The recommended-defaults checklist that drops out of the paper:

1. Pick the target $\gamma_Z$ as you would for a normal value function.
2. Build the ladder by halving the effective horizon downward: $\gamma_{z} = (\gamma_{z+1} + 1)/2$ down to a sensible $\gamma_0$.
3. Set $k_z \approx 1/(1 - \gamma_z)$.
4. Set $\lambda_z$ either to satisfy $\lambda_z \gamma_z = \lambda_Z \gamma_Z$ (safe / theoretically equivalent at convergence) or capped at $1$ (more aggressive variance reduction; works better empirically on Atari).
5. Plug into any TD-based algorithm (TD, Sarsa, Q-learning, PPO, A2C); replace the single value head with a $(Z+1)$-output linear (or small MLP) head.

#### Limitations as authors state them

- **Sparse-reward regime.** The compounding bias from short-horizon heads slows early learning when rewards are rare; mitigation requires tuning $\gamma_0$ upward.
- **Schedule choice.** The doubling schedule works "well enough" without tuning but is not always optimal; the authors flag meta-gradient (Xu, van Hasselt, Silver 2018) as a complementary tool for adaptive ladder choice.
- **Algorithmic complexity.** Naïve implementation is $O(Z^2)$ per step in the multi-step return; the authors note it can be made $O(Z)$ by caching $\hat V$ at each $\gamma_z$.
- **Not all benefit comes from the decomposition itself.** Some of the empirical wins are attributable to per-head tuning rather than the cascade structure per se; the PPO+ baseline (same architecture, single value loss) controls for this.

#### Where TD(Δ) connects to the other two papers

- **Contrast with Sherstan (γ-nets).** Both produce a multi-horizon estimator; γ-nets condenses the cascade into one network with $\gamma$ as an input, TD($\Delta$) keeps separate heads. γ-nets queries arbitrary $\gamma$ at test time; TD($\Delta$) only the $Z+1$ explicit horizons. The Romoff discussion explicitly notes that the *cascade structure* is a way to "bootstrap" the long-horizon learning off the shorter horizons — something γ-nets does not do.
- **Contrast with Fedus (hyperbolic).** Fedus also keeps separate heads but **integrates** $\sum_z V_{\gamma_z}$ to recover a hyperbolic discount; Romoff **sums** the *differences* $\sum_z W_z = V_{\gamma_Z}$ to recover the largest discount. Both architectures look similar at the network level — a parallel set of value heads at different $\gamma$ — but the *target* of the sum and the *parameterisation* of each head differ.

### Appendix: Section-by-section backbone (Romoff et al. 2019)

**Step 1 — Section list (original order).** Abstract → 1. Introduction → 2. Related Work → 3. Background and Notation → 4. TD(Δ) (4.1 Single-step; 4.2 Multi-step; 4.3 TD(λ,Δ); 4.4 TD(λ,Δ) with GAE) → 5. Analysis (5.1 Equivalence; 5.2 Reduced-k analysis) → 6. Experiments (6.1 Tabular; 6.2 Dense Atari; 6.3 Tuning/Ablation) → 7. Discussion → Conclusion → Appendices (Reproducibility checklist; B. Proofs; further experiments).

**Step 2/3 — Per-section content.**

- **Abstract.** Argues that finite-horizon-episodic RL (e.g. Atari) is better optimised at large $\gamma$, but large-$\gamma$ value targets have high variance. Proposes TD($\Delta$), which decomposes the value function into delta estimators $W_z = V_{\gamma_z} - V_{\gamma_{z-1}}$; each $W_z$ satisfies a Bellman-like equation with $V_{\gamma_{z-1}}$ as pseudo-reward. Demonstrates theoretical and empirical gains over standard TD in dense-reward MDPs.
- **Introduction.** Discounting trades off planning horizon vs. learning ability. Existing approaches: curriculum on $\gamma$ (OpenAI 2018, Prokhorov & Wunsch 1997, François-Lavet 2015), meta-gradient (Xu et al. 2018). Observation: a large-$\gamma$ value already encompasses smaller time-scales, so decompose into deltas that can each be learned with their own bias-variance trade-off and the shorter horizons bootstrap the longer.
- **Related work.** Closest: Fedus 2019 (hyperbolic), Sherstan 2018 (γ-nets / GVFs), Sutton 2011 (Horde), Sutton 1995 (TD models), Feinberg & Shwartz 1994 (mixture of two $\gamma$s), Reinke et al. 2017 (average-return imitation). All learn ensembles at different $\gamma$, but **none use short-horizon estimates to train the long-horizon estimate** — the distinguishing feature of TD($\Delta$). Also touches HRL (Dietterich, Russell-Zimdars, van Seijen) and meta-gradient.
- **Background and notation.** Standard MDP, monolithic discounted value $V^\pi_\gamma$, Bellman operator $T^\pi V_\gamma = r^\pi + \gamma P^\pi V_\gamma$, one-step TD error, GAE/$\lambda$-return advantage, actor-critic policy loss, PPO clipping objective. Notation will drop the $\pi$ superscript subsequently.
- **TD(Δ).** Definitions $W_z = V_{\gamma_z} - V_{\gamma_{z-1}}$, $W_0 = V_{\gamma_0}$, $V_{\gamma_z} = \sum_{i \le z} W_i$. Derives the Bellman-like equation for $W_z$ (eq. 8): $W_z(s_t) = \mathbb{E}[(\gamma_z - \gamma_{z-1}) V_{\gamma_{z-1}}(s_{t+1}) + \gamma_z W_z(s_{t+1})]$. Recommended default ladder: $\gamma_{z+1} = (\gamma_z + 1)/2$ (doubling effective horizon) until $\gamma_Z$ reached. Then derives multi-step (eq. 9) and $\lambda$-return (eq. 12) variants. PPO integration uses an aggregate advantage $A^\Delta$ on $\sum_z \hat W_z$ at the largest horizon $\gamma_Z$, $\lambda_Z$.
- **Analysis.** Theorem 1: equivalence to TD($\lambda$) under symmetric hyperparameters and matched initialisation; proof by induction with telescoping sum. Theorem 2: TD($\lambda$) operator is contractive for $\lambda < (1+\gamma)/(2\gamma)$, so $\lambda_z > 1$ is admissible. Theorem 3 (Kearns & Singh 2000, restated): bias-variance bound for phased $k$-step TD. Theorem 4: extended bound for phased TD($\Delta$) showing strictly negative variance correction when $k_z < k_{z+1}$, at cost of compounding bias term. Practical rule: $k_z = 1/(1 - \gamma_z)$, $\gamma_z$ doubling.
- **Experiments — Tabular.** 5-state ring MDP (Kearns & Singh 2000); 7 settings of $\gamma_Z$ giving effective horizons $\{4, 8, 16, 32, 64, 125, 250\}$. With $k_z$ all equal, exactly matches single estimator (Theorem 1); with tailored $k_z$, strictly outperforms (or ties) the single estimator in all cases, with the gap growing in $\gamma_Z$ and $k$.
- **Experiments — Dense Atari.** PPO-TD($\lambda, \Delta$) and PPO-TD($\hat\lambda, \Delta$) (cap $\lambda_z \le 1$) on the 9 "Hard" dense-reward Atari games of Bellemare 2016. Doubling ladder from $\gamma_Z = 0.99$. Significant wins on Qbert, MsPacman, Hero, Amidar; losses on Zaxxon, WizardOfWor (sparse-reward). PPO+ (same architecture, single value loss) ablation rules out architecture-only explanation. Fig. 2 shows interpretable per-head traces aligned to in-game events (lost life).
- **Experiments — Tuning/Ablation.** Pushing $\gamma_0$ up (using only $\{0.3, 0.99\}$ or $\{0.92, 0.99\}$ as the ladder; labelled `ppoDelta3` and `ppoDelta12`) recovers parity with PPO on Zaxxon / WizardOfWor — confirming the Theorem-4 prediction that the compounding-bias term dominates when $\gamma_0$ is too small in sparse-reward MDPs.
- **Discussion.** Three claimed benefits: scalability (heads can be distributed across machines), per-head tuning (learning rate / $k_z$ / $\lambda_z$), and interpretability (read off the per-horizon contributions). Anytime character: $\sum_{i \le z'} W_i$ is a valid approximation for any converged sub-ladder. Extends naturally to Sarsa, Q-learning, etc.
- **Conclusion.** TD($\Delta$) is a "drop-in" addition to any TD-based method; the recommended doubling ladder yields gains without tuning; further tuning improves results further.

**Step 4 — Most relevant sections expanded.** Section 4 (the derivation of the $W_z$ Bellman-like equation, the multi-step unrolling, and the PPO advantage construction) and Section 5 (Theorem 1 equivalence + Theorem 4 bias-variance bound — these are the load-bearing theoretical results). Both are folded into the Phase 2 sections above with full step-by-step derivations.

---

## 3. Fedus et al. 2019 — Hyperbolic Discounting and Learning over Multiple Horizons

**Venue:** arXiv preprint `arXiv:1902.06865` (concurrent with Romoff 2019).
**Authors:** William Fedus, Carles Gelada, Yoshua Bengio, Marc G. Bellemare, Hugo Larochelle (Google Brain / Mila).
**PDF:** `docs/project/references/Gamma/sources/Fedus et al. 2019 - Hyperbolic discounting and learning over multiple horizons.pdf` (28 pages including appendices).

### Phase 1 — Foundational overview (undergrad-level)

#### Introduction

Standard reinforcement learning agents discount future reward **exponentially**: a reward $r$ at time $t$ is worth $\gamma^t r$ to the agent right now. The exponential choice is convenient — it makes the value function satisfy the Bellman equation, and the Bellman equation is what enables TD-style algorithms (Q-learning, DQN, etc.). The trouble is that humans and animals don't appear to discount this way. Across psychology, behavioural economics, and neuroscience, the measured discount curves are **hyperbolic**:

$$
d_k(t) \;=\; \frac{1}{1 + k t},
$$

which falls off fast initially but tails off slowly — quite different from $\gamma^t$. Animals robustly show "preference reversals" (a hyperbolic-curve signature) that exponential discounting forbids by Strotz's 1955 result.

Fedus et al. ask: can we build a TD-based agent that *acts* with hyperbolic discounting, without giving up the algorithmic guarantees of standard Q-learning? Their answer leans on a mathematical identity. Hyperbolic discounting equals an **integral over exponentials**:

$$
\boxed{\;\frac{1}{1 + k t} \;=\; \int_0^1 \gamma^{k t}\, d\gamma.\;}
$$

So a hyperbolically-discounted value is the integral of a continuum of exponentially-discounted values, each at a different $\gamma$. They approximate that integral with a Riemann sum over a finite set of $\gamma$ values, train a Q-head for each, and aggregate. The architecture is one shared body that outputs $n_\gamma$ parallel Q-heads at $\gamma_0 < \gamma_1 < \dots < \gamma_N$.

The paper has a second, semi-independent contribution. They find — surprisingly — that **even when the agent's behaviour policy doesn't use hyperbolic discounting at all**, the multi-head architecture *as an auxiliary task* significantly improves a state-of-the-art Rainbow agent on Atari. Predicting value at multiple horizons turns out to be a useful representation-learning signal in its own right.

#### Key findings

- **Hyperbolic-from-exponentials identity.** $1/(1+kt) = \int_0^1 \gamma^{kt} d\gamma$ — so multiple exponential Q-functions can combine into a hyperbolic one.
- **General weighting lemma.** Any discount function $d(t)$ expressible as $\int_0^1 w(\gamma) \gamma^t d\gamma$ admits the same trick: $Q^{H,d}_\pi(s,a) = \int_0^1 w(\gamma) Q^{H,\gamma}_\pi(s,a) d\gamma$. Table 1 in the paper gives $w(\gamma)$ for exponential, hyperbolic, and uniform-prior hazards.
- **Hazard interpretation (Sozou 1998).** A constant hazard rate $\lambda$ on the environment gives the agent a per-step survival $s(t) = e^{-\lambda t} = \gamma^t$ — i.e., the standard RL discount factor is equivalent to a Dirac-delta prior on hazard rate. A non-degenerate prior $p(\lambda)$ — e.g., exponential prior — implies a non-exponential discount (hyperbolic for exponential prior on $\lambda$). So hyperbolic discounting is the *Bayesian-rational* response to *uncertainty about the environment's hazard rate*.
- **Pathworld validation.** In a hand-designed environment where the agent picks one of $N$ paths and is subjected to a sampled-per-episode hazard, hyperbolic agents track the theoretical-value-per-path curve with MSE ≈ 0.002, while every tested exponential-$\gamma$ baseline misses by 0.5–2.8 — and crucially, robustness extends to cases where the agent's hazard prior is wrong by a factor of 2 in $k$.
- **Multi-horizon auxiliary task (the second big finding).** In Atari, the *aggregation-over-$\gamma$* part of the architecture (acting hyperbolically) doesn't matter much — `Hyper-Rainbow` (hyperbolic behaviour policy) and `Multi-Rainbow` (multi-$\gamma$ heads, but greedy w.r.t. one $\gamma$) perform near-identically. **The lift comes from learning the multi-horizon Q-heads as an auxiliary task** — better representation, propagated through the shared CNN trunk. Multi-Rainbow wins on 14/19 randomly-sampled Atari games over Rainbow.

#### Initial takeaway

Fedus et al. give the project two things. (1) A principled way to build a non-exponential-discounting agent on top of any TD method — relevant if "the modulator should bend the discount curve" becomes a design goal. (2) A free auxiliary-task improvement: even if downstream policy uses a single $\gamma$, training extra Q-heads at many $\gamma$ improves the shared representation. This second finding is the most actionable take-away for an agent that is already FiLM-conditioned on a modulator — the extra heads cost ~$n_\gamma$ affine projections off the shared body, no extra deep computation, and the auxiliary signal is independent of whether the modulator's job is horizon control or anything else.

### Phase 2 — Graduate-level deep dive

#### Hazard rate and the discount function (the Sozou 1998 derivation)

Define the **survival function** $s(t) = P(\text{agent is alive at time } t)$, and the **hazard rate** as the negative log-derivative of survival:

$$
h(t) \;=\; -\frac{d}{dt} \ln s(t).
$$

A risk-neutral agent should value a future reward $r_t$ at $v(r_t) = s(t)\, r_t$ — discount by survival probability.

**Known constant hazard $\to$ exponential discount.** If $h(t) = \lambda$ (constant), the ODE $\lambda = -d \ln s/dt$ has solution $s(t) = e^{-\lambda t}$. Identifying with $\gamma^t$ gives $\gamma = e^{-\lambda}$. So exponential discounting in RL is *equivalent to* the prior "constant known hazard rate $\lambda = -\ln \gamma$".

**Uncertain hazard $\to$ non-exponential discount.** If the agent maintains a prior $p(\lambda)$ over hazard rates, then the expected survival marginalises:

$$
s(t) \;=\; \int_0^\infty p(\lambda)\, e^{-\lambda t}\, d\lambda.
$$

This is exactly the Laplace transform of $p$. For an exponential prior $p(\lambda) = (1/k) e^{-\lambda/k}$, the Laplace transform evaluates to

$$
s(t) \;=\; \frac{1}{1 + k t} \;\equiv\; \Gamma_k(t)
$$

— the hyperbolic discount. **So hyperbolic discounting is what falls out of Bayes-marginalisation over hazard rates when the prior is exponential.** Other priors give other discount functions (uniform $\to (1/(kt))(1 - e^{-kt})$; gamma prior $\to (1/(1+bt))^c$).

#### The exponential-weighting condition (Lemma 5.1)

In a hazardous MDP $\langle \mathcal{S}, \mathcal{A}, R, P, H, d \rangle$ — where $H$ is the hazard distribution sampled per episode and $d(t)$ the discount function — define

$$
Q^{H,d}_\pi(s,a) \;=\; \mathbb{E}_{\lambda \sim H} \mathbb{E}_{\pi, P_\lambda}\!\left[\sum_{t=0}^\infty d(t)\, R(s_t, a_t) \,\Big|\, s_0 = s, a_0 = a\right]
$$

with hazard-modified transition $P_\lambda(s' | s, a) = e^{-\lambda} P(s' | s, a)$.

**Lemma 5.1.** If there exists a weighting function $w : [0,1] \to \mathbb{R}$ such that $d(t) = \int_0^1 w(\gamma) \gamma^t d\gamma$, then

$$
\boxed{\;Q^{H, d}_\pi(s, a) \;=\; \int_0^1 w(\gamma)\, Q^{H, \gamma}_\pi(s, a)\, d\gamma.\;}
$$

Proof (one line in the paper): substitute $d$ and swap the sum/integral, valid whenever the geometric series converges. The lemma is **the load-bearing identity** of the paper — it converts the problem of estimating a single non-exponentially-discounted Q-function into the problem of estimating a continuum of exponentially-discounted Q-functions, each computable by standard TD.

#### Two concrete weightings for hyperbolic discount

| Identity | $w(\gamma)$ | Aggregation |
|---|---|---|
| $\Gamma_k(t) = \int_0^1 \gamma^{kt} d\gamma$ | implicit (re-parameterise as $\gamma' = \gamma^k$) | $Q^\Gamma_\pi = \int_0^1 Q^{(\gamma^k)^t}_\pi d\gamma$ |
| $\Gamma_k(t) = \int_0^1 (1/k) \gamma^{1/k + t - 1} d\gamma$ | $(1/k) \gamma^{1/k - 1}$ | $Q^\Gamma_\pi = \int_0^1 (1/k) \gamma^{1/k - 1} Q^{\gamma^t}_\pi d\gamma$ |

The second form follows the Laplace-transform derivation in $p(\lambda) = (1/k) e^{-\lambda/k}$ and is what the paper uses for the practical algorithm. Table 1 in the paper gives $w(\gamma)$ for delta, exponential, and uniform priors over hazard.

#### Approximation: from integral to Riemann sum

Practical agents discretise the integral with $n_\gamma$ heads at $G = \{\gamma_0, \gamma_1, \dots, \gamma_{n_\gamma}\}$ chosen to emphasise the larger values of $\gamma$ (which is where most of the integral mass lives at horizons of interest):

$$
Q^\Gamma_\pi(s, a) \;\approx\; \sum_{\gamma_i \in G} (\gamma_{i+1} - \gamma_i)\, w(\gamma_i)\, Q^{\gamma_i}_\pi(s, a).
$$

Each $Q^{\gamma_i}_\pi$ is a separate exponentially-discounted Q-function, learned with standard TD updates against its own target with $\gamma = \gamma_i$. Architecturally (Fig. 9 of the paper) this is **one shared trunk** (CNN body $h(s)$) that branches into $n_\gamma$ small affine heads:

$$
Q^{(i)}(s, a) \;=\; f\!\left(W_i\, h(s) + b_i\right), \qquad f = \text{ReLU},
$$

so the extra cost per head is one affine projection — much cheaper than running $n_\gamma$ separate networks. The Hyper-Rainbow agent reuses the entire Rainbow stack (distributional value head, prioritised replay, n-step returns) and just multiplies the output head $n_\gamma$ times.

#### Validation on Pathworld

**Pathworld** (Fig. 5 in the paper) is a single-decision multi-armed-bandit-like MDP: $N$ paths of length $d(i) = i^2$ and reward $r(i) = i$. The agent picks one path; per-step hazard is sampled from $H = (1/k) e^{-\lambda/k}$ at the start of the episode, so longer paths give bigger rewards at higher cumulative risk. Train under hyperbolic discount $\Gamma_k$, evaluate undiscounted on the hazardous environment.

Results (Tables 2–4 of the paper):
- Hyperbolic agent with matched $k$: MSE 0.002 vs. theoretical-value curve.
- Best-tuned exponential agent ($\gamma = 0.975$): MSE 0.566 — chooses correct argmax path but misses the *shape* of the value-vs-path curve, so it would fail on any task where the argmax is not the only thing that matters.
- Hyperbolic agent with mismatched $k$ (off by factor 2): MSE 0.493 — still much better than every exponential baseline.
- Hyperbolic agent under *uniform* hazard prior (functional-form mismatch): MSE 0.235 — still beats exponential.

**Conclusion:** hyperbolic discounting is robust to mis-specification of the hazard prior, both in magnitude $k$ and in functional shape. Exponential discounting is fragile to it.

#### The Atari finding and the multi-horizon-auxiliary ablation

`Hyper-Rainbow` consists of two architectural changes vs. baseline Rainbow:
1. **Behaviour policy.** Acts greedy w.r.t. the hyperbolic Q-value $Q^\Gamma$ (Riemann sum over heads).
2. **Multi-head training.** Learns Q-values at $n_\gamma$ different $\gamma$ simultaneously off the shared trunk.

On 19 randomly-chosen Atari games, Hyper-Rainbow beats Rainbow on 14. To attribute the lift, they construct `Multi-Rainbow`: same multi-head architecture, but the behaviour policy still acts greedy w.r.t. the single largest-$\gamma$ head — i.e., learns multi-horizon but acts single-horizon. **Multi-Rainbow matches Hyper-Rainbow nearly game-for-game.** The win is therefore not coming from hyperbolic action selection — it's coming from the auxiliary-task pressure of having to predict Q at many horizons through the shared representation.

This is consistent with the auxiliary-task literature (UNREAL: Jaderberg et al. 2016; Lample & Chaplot 2017): more prediction targets force the representation to encode features that generalise. Multi-horizon prediction is just a particularly natural set of targets — each $\gamma_i$ samples a different time-horizon of the same underlying reward stream.

Further ablation (Fig. 12): Multi-C51 (C51 + multi-horizon auxiliary) improves on C51 in 9/10 randomly-chosen games. The multi-horizon auxiliary task plays *well* with most components of Rainbow but does not currently play well with prioritised replay (4/10 games negatively impacted) — the authors flag this as a TD-error-prioritisation aggregation problem (the implementation averages TD errors across heads, which may be suboptimal) and defer to future work (Appendix E preliminary results).

#### Hazardous MDP framework and equivalence theorems

Beyond the constructive algorithm, the paper formalises the hazardous-MDP setting:

**Equivalence (Section 4.1).** $Q^{\delta(0), \gamma^t}_\pi(s, a) = Q^{\delta(-\ln \gamma), 1}_\pi(s, a)$ for all $\pi, s, a$. In words: discounting future rewards by $\gamma^t$ in a *non-hazardous* MDP is *exactly equivalent* to not discounting at all but acting in an MDP with constant hazard rate $\lambda = -\ln \gamma$. So discounting can be interpreted as "make the policy robust to a per-step survival risk you don't actually know is there".

**Hyperbolic analogue (Appendix A).** $Q^{\delta(0), \Gamma_k}_\pi(s, a) = Q^{p_k, 1}_\pi(s, a)$ where $p_k(\lambda) = (1/k) e^{-\lambda/k}$. Hyperbolic discounting in a non-hazardous MDP $=$ undiscounted optimisation in an MDP where hazard is drawn from an exponential prior. This is the "robustness" interpretation of hyperbolic discounting: train at high $\gamma$ but consume hyperbolically, and the resulting policy is robust to a wide range of hazard rates without needing to know which one is true.

#### Limitations as authors state them

- **Hyperbolic Q-values are unbounded for unbounded MDPs.** The integral $\int_0^1 \gamma^t d\gamma$ summed over all $t$ converges only for episodic / non-infinite MDPs; the paper restricts to that setting throughout.
- **Choice of $G$ and $n_\gamma$.** The Riemann-sum quadrature error depends on how $G$ is spaced. The paper uses a hand-picked spacing biased toward large $\gamma$ but acknowledges this is unprincipled.
- **Prioritised-replay aggregation.** Current implementation averages TD errors across heads to form a single priority; this loses 4/10 games versus the same agent without prioritised replay. Better aggregation schemes (max, weighted by per-head loss) are flagged for future work.
- **Action selection from hyperbolic Q.** Selecting $\arg\max_a Q^\Gamma(s, a)$ requires summing $n_\gamma$ heads at each candidate action — modest cost but non-trivial in large-action settings.

#### Where Fedus connects to the other two papers

- **Vs. Sherstan (γ-nets).** Both want value at many horizons; γ-nets folds the bag of $\gamma$ into a single network with $\gamma$ as input; Fedus keeps separate heads (much cheaper at training time since affine heads share the trunk). γ-nets queries arbitrary $\gamma$; Fedus only the discrete $G$.
- **Vs. Romoff (TD($\Delta$)).** Both keep separate value heads at $\gamma_0 < \dots < \gamma_Z$. Romoff sums *differences* $\sum_z W_z = V_{\gamma_Z}$ to recover the **largest discount**. Fedus sums *weighted heads* $\sum_i w(\gamma_i) Q^{\gamma_i}$ to recover a **non-exponential discount** ($V_{\text{hyp}}$). Romoff explicitly uses short-horizon heads to bootstrap long-horizon heads; Fedus's heads are *parallel* and independent. The two are structurally compatible — one could in principle train Fedus's bag-of-heads using Romoff's cascade-bootstrapped TD update.
- **The auxiliary-task framing is the bridge to the project's interests.** A FiLM-modulated agent that already conditions its computation on an interoceptive modulator $m$ can take Fedus's auxiliary heads "for free": add $n_\gamma$ affine projections, train against TD targets with the corresponding $\gamma_i$, and reap the representation-learning benefit without touching the behaviour policy. This is the lowest-cost route to importing multi-horizon ideas into the current architecture.

### Appendix: Section-by-section backbone (Fedus et al. 2019)

**Step 1 — Section list (original order).** Abstract → 1. Introduction → 2. Related Work → 3. Belief of Risk Implies a Discount Function (3.1 Known hazard → exponential; 3.2 Uncertain hazard → non-exponential) → 4. Hazard in MDPs (4.1 Equivalence) → 5. Computing Hyperbolic Q-Values from Exponential Q-Values (5.1 Hyperbolic; 5.2 General weighting lemma) → 6. Approximating Hyperbolic Q-Values (6.1 Riemann-sum) → 7. Pathworld Experiments → 8. Atari Experiments → 9. Multi-Horizon Auxiliary Task Results (9.1 Ablation studies) → 10. Discussion / Conclusion → Appendices (A. Hyperbolic-hazard equivalence; B. Discount-function derivations; C. Architecture; D. Riemann-sum details; E. Prioritisation ablation; F. Atari per-game results; G. Hyperparameters).

**Step 2/3 — Per-section content.**

- **Abstract.** Two contributions: (i) a TD-compatible algorithm for hyperbolic (and other non-exponential) discounting, by Q-learning over a bag of $\gamma$ and aggregating; (ii) surprise finding that multi-$\gamma$ learning is a strong auxiliary task that improves Rainbow.
- **Introduction.** Standard RL uses single exponential $\gamma$; this implies a Dirac-delta prior on environment hazard rate. Empirically, humans/animals discount hyperbolically (Mazur, Ainslie, Green & Myerson). Hyperbolic discounting was thought TD-incompatible (Daw & Touretzky 2000) but the field has reconsidered (Maia 2009; Alexander & Brown 2010; Kurth-Nelson & Redish 2009 µAgents). This paper proposes the principled bridge and finds the auxiliary-task bonus.
- **Related Work.** Hyperbolic discounting in economics (Sozou 1998 — the foundation; Dasgupta & Maskin 2005). Behavioural RL and neuroscience (Montague, Schultz, Sutton & Barto on TD; Maia, Kurth-Nelson & Redish on hyperbolic). Multi-horizon RL (Horde, γ-nets, TD($\Delta$); concurrent with Romoff). Auxiliary tasks (UNREAL, Lample & Chaplot, Mirowski).
- **Belief of Risk Implies a Discount Function.** Hazard rate definitions, survival ODE, Laplace-transform derivation of $s(t)$ from $p(\lambda)$. Delta prior gives exponential ($s(t) = e^{-\lambda t} = \gamma^t$). Exponential prior gives hyperbolic ($s(t) = 1/(1+kt) = \Gamma_k(t)$). Other priors give other functional forms (Fig. 2 reproduces from Sozou).
- **Hazard in MDPs.** Extended MDP tuple $\langle \mathcal{S}, \mathcal{A}, R, P, H, d \rangle$. Per-episode hazard $\lambda \sim H$, modified transition $P_\lambda(s' | s,a) = e^{-\lambda} P(s' | s,a)$, general discount $d(t)$. Equivalence theorem 4.1: $Q^{\delta(0), \gamma^t} = Q^{\delta(-\ln \gamma), 1}$ — discount=hazard.
- **Computing Hyperbolic Q-Values from Exponential.** Identity $\Gamma_k(t) = \int_0^1 \gamma^{kt} d\gamma$ (Fig. 3 visualises the integrals at $t = 0, 1, 2, 3$). Lemma 5.1: general exponential-weighting decomposition $d(t) = \int w(\gamma) \gamma^t d\gamma \Rightarrow Q^d = \int w(\gamma) Q^\gamma d\gamma$. Alternative hyperbolic weighting $w(\gamma) = (1/k)\gamma^{1/k - 1}$.
- **Approximating Hyperbolic Q-Values.** Riemann-sum quadrature over a finite $G$; each $Q^{\gamma_i}$ trained with standard TD. Aggregation in Eq. 26. Architecture: shared trunk + $n_\gamma$ affine heads.
- **Pathworld.** Single-decision multi-armed-bandit-like MDP with sampled-per-episode hazard. Hyperbolic agent tracks theoretical value with MSE $\approx 0.002$; best exponential baseline 0.566; robustness to mis-specified $k$ (0.493) and to uniform-prior mismatch (0.235).
- **Atari.** Rainbow + multi-head trunk. Hyper-Rainbow beats Rainbow on 14/19 games.
- **Multi-Horizon Auxiliary Task Results.** Multi-Rainbow (auxiliary heads, single-$\gamma$ behaviour) ≈ Hyper-Rainbow on Atari → the gain is from auxiliary-task representation pressure, not from hyperbolic action selection. Multi-C51 wins 9/10 games over C51. Multi-Rainbow does not stack well with prioritised replay (4/10 games hurt) — flagged as a TD-error-aggregation issue.
- **Discussion / Conclusion.** Hyperbolic discounting is mathematically tractable in deep RL via the integral identity; the bag-of-$\gamma$ architecture is a strong drop-in auxiliary task even when hyperbolic action selection is not used. Future work: meta-gradient $\gamma$ schedules, better priority aggregation, application to non-Atari domains.

**Step 4 — Most relevant sections expanded.** Section 5 (the integral identity $\Gamma_k(t) = \int_0^1 \gamma^{kt} d\gamma$ and the general weighting lemma) and Section 9 (the multi-horizon-as-auxiliary-task ablation). Both are folded into the Phase 2 sections above. Section 3's hazard-prior derivation is included because it gives the project a Bayesian-coherent reading of "what does discounting mean" — relevant if the modulator is later interpreted as encoding hazard belief.

---

## 4. Cross-paper synthesis

This is a brief cross-cutting note, not a separate review.

### Three routes to multi-horizon value, side-by-side

| Aspect | Sherstan (γ-nets) | Romoff (TD(Δ)) | Fedus (hyperbolic / multi-γ aux) |
|---|---|---|---|
| Architecture | One network; $\gamma$ as input | $Z+1$ heads in cascade; $W_z = V_{\gamma_z} - V_{\gamma_{z-1}}$ | $n_\gamma$ heads in parallel; affine projections off shared trunk |
| Aggregation | $V(s, \gamma)$ at query time | $V_{\gamma_Z} = \sum_z W_z$ | $V^\Gamma = \sum_i w(\gamma_i) Q^{\gamma_i}$ (or any other Lemma-5.1 weighting) |
| Inter-head dependency | None (single network) | **Sequential** — long-horizon target uses short-horizon estimate as pseudo-reward | None — heads independent |
| Variance handling | $(1-\gamma)$ loss scaling | Per-head $k_z$, $\lambda_z$ tuning + Theorem-4 bias-variance trade-off | None explicit; relies on aggregation |
| Queryable $\gamma$ | Any $\gamma \in [0,1)$ | Only the $Z+1$ ladder $\gamma_z$ | Only the bag $G$ |
| Empirical scope | LFA + Atari policy evaluation | PPO + Atari control | Rainbow + Atari control |
| Headline benefit | Compact, single-network, arbitrary-$\gamma$ queries | Variance reduction at long horizons via short-horizon bootstrapping | Multi-horizon prediction as a representation-learning auxiliary task |
| Headline cost / loss | Slight accuracy loss vs. per-head specialist | Compounding bias if $\gamma_0$ too small in sparse-reward MDPs | Mild prioritised-replay incompatibility |

### Recurring connections and tensions

- **All three keep "the bag of $\gamma$" as a foundational object.** They differ only in what they do with it: γ-nets compresses it into one function with $\gamma$ as an input; Romoff cascades it into differences; Fedus integrates it through a weighting kernel. The three operations — compression, differencing, integration — are not mutually exclusive (Sherstan's discussion explicitly hopes Γ-nets could one day serve as Fedus's basis function via one network).
- **Construction of $\Gamma_t / \{\gamma_z\} / G$ is universally the elephant.** Each paper finds that the choice of which $\gamma$ to train on dominates the empirical results. Sherstan recommends sampling from both $\gamma$- and $\tau$-uniform; Romoff recommends doubling the effective horizon; Fedus recommends hand-picked spacing biased toward large $\gamma$. None of these are derived from theory.
- **Auxiliary-task framing is the broadest take-home.** Fedus's ablation isolating the multi-horizon-as-auxiliary effect *applies retroactively* to γ-nets and TD($\Delta$): both architectures effectively train a network to predict at many horizons through a shared body, which is exactly the auxiliary-task signal Fedus identifies. The implication is that **any multi-horizon architecture** plausibly delivers a representation-learning bonus on top of whatever its primary motivation is.
- **None of the three implements modulator-conditioned $\gamma$.** All three treat $\gamma$ (or the bag thereof) as a hyperparameter fixed at design time. For the project, the natural next move — "let the interoceptive modulator $m$ continuously bend either $\gamma$ (γ-nets-style input) or the aggregation weights $w(\gamma; m)$ (Fedus-style) — is unimplemented in any of these papers. The Sozou hazard-prior framing in Fedus gives the cleanest biological story (modulator $\sim$ hazard prior $\sim$ implicit discount), but the implementation route is not derived.

### For the project — three actionable recommendations

These are recommendations for delegation, not implementation. Code changes should be planned by `senior-developer` if pursued.

1. **Cheapest path: multi-horizon auxiliary task on the existing FiLM agent.** Add $n_\gamma \approx 5$–$10$ affine value heads off the existing shared body, each trained against a TD target with a different $\gamma_i$. Behaviour policy unchanged. The Fedus paper shows this is a near-free representation-learning upgrade in 14/19 Atari games. Risk: modest; downside: increased output dimension. This is the route most compatible with the current architecture.
2. **Mid-cost path: γ-as-input (Sherstan-style) for the head, modulator-as-input via FiLM (current).** Concatenate $\gamma$ (and $\tau$) into the value head's input alongside the existing FiLM-modulated trunk. The modulator can shift the *attended* $\gamma$ at evaluation time by setting which $\gamma$ is queried. Risk: higher tuning cost; need to handle the $\gamma$/$\tau$ sampling distribution.
3. **High-cost / high-reward path: hazard-prior-conditioned aggregation (Fedus + modulator).** Make the modulator's output the parameter $k$ of the hyperbolic discount, so the policy continuously bends between "myopic, high hazard" and "far-sighted, low hazard" with the modulator. This is the most biologically motivated route (it matches Doya-style serotonin-as-temporal-horizon-control directly). Risk: requires the aggregation kernel $w(\gamma; k)$ to be implemented and differentiable; the integral $\int w(\gamma; k) Q^\gamma d\gamma$ must be amenable to whatever loss is used downstream. Highest potential payoff, highest implementation cost.

---

