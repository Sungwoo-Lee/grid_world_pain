# Learning Rate — Literature Review C: Meta-Gradient Reinforcement Learning

**Corpus theme:** RL-specific instances of "meta-gradient on a hyperparameter." Two papers, processed strictly one at a time. Both treat a quantity that an RL algorithm normally hand-tunes — the per-feature step size, or the discount factor / trace-decay parameter that defines the return — as a *learnable* meta-parameter whose value is adjusted online by gradient descent on some meta-objective.

---

## Question / Purpose / Context (plain-language entry point)

**Why we built this collection.** Reinforcement-learning algorithms are full of "magic numbers." Two of the most important are:

1. **The step size** (often called $\alpha$ or *learning rate*) — how big a correction the agent makes after each piece of experience. Too large and the value estimates wobble or explode; too small and the agent learns at a crawl. Crucially, in temporal-difference (TD) learning the agent is updating its predictions toward *its own bootstrapped estimates of the future*, which means the target itself is non-stationary — the "right" step size therefore changes over time, and may be different for different input features (a pixel at the corner of the screen has nothing to do with the return; an interoceptive heart-rate channel might be everything). A single global $\alpha$ cannot express that.
2. **The return-shaping parameters** — primarily the discount $\gamma$ (how far into the future the agent cares) and the trace-decay $\lambda$ (how aggressively the agent bootstraps off its own value estimates rather than waiting for real rewards). These define *what target the agent is trying to predict*. They are usually picked once and frozen for the rest of training, but the best value can depend on what stage of learning the agent is in (myopic early, far-sighted late), or even on what state the agent is currently in (signal vs. noise periods).

**What "meta-gradient" means.** Once you decide that some hyperparameter $\eta$ is a learnable quantity, you need a learning signal for it. The trick — going back to Sutton's 1992 IDBD — is to take the gradient of *some performance loss* with respect to $\eta$, even though $\eta$ only enters the parameter update *indirectly* (through the size of each gradient step, or through what return the loss measures against). The chain rule still works: differentiate through the inner update. The result is a "gradient of a gradient" — a **meta-gradient** — and you do ordinary gradient descent on $\eta$ using it. The phrase "stochastic meta-descent" is Schraudolph's name for the same idea applied per-feature.

**The two papers, in one sentence each.**

- **Kearney, Veeriah, Travnik, Sutton, Pilarski (2018) — TIDBD.** Generalize Sutton's IDBD step-size adapter from supervised learning to TD learning, keeping per-feature step sizes that can both grow and shrink. The thing the meta-gradient adapts is the **learner's step size** (one $\alpha_i$ per feature). The target the agent is chasing is *fixed* by the user.
- **Xu, van Hasselt, Silver (2018) — Meta-Gradient RL.** Treat the *return* itself as a parametric function $g_\eta(\tau)$ — with $\eta = \{\gamma, \lambda\}$, the discount and trace-decay — and adapt $\eta$ online via meta-gradient on a separate policy-gradient meta-objective. The thing the meta-gradient adapts is the **target the agent chases**. The learner's step size is fixed by the user.

**Why this matters for the project.** Both papers are precedents for "a modulator signal influences something the optimizer or the target computation depends on." TIDBD modulates the *learner* (per-feature step sizes); Xu modulates the *target* (the return). Together they bracket the design space: a neuromodulator-like signal could in principle act at either end — speed up or slow down learning for particular features (TIDBD-like), or stretch / shorten the horizon over which value is integrated (Xu-like). The math infrastructure — semi-gradient through the TD error in TIDBD, chain-rule through one inner update in Xu — is what any project-level mechanism would inherit.

---

## Table of Contents

- [Question / Purpose / Context (plain-language entry point)](#question--purpose--context-plain-language-entry-point)
- [Paper 1 — Kearney et al. (2018), TIDBD](#paper-1--kearney-et-al-2018-tidbd-adapting-temporal-difference-step-sizes-through-stochastic-meta-descent)
  - [Phase 1 — Foundational Overview](#phase-1--foundational-overview-tidbd)
  - [Phase 2 — Graduate-Level Deep Dive](#phase-2--graduate-level-deep-dive-tidbd)
  - [Appendix: Section-by-Section Backbone](#appendix-section-by-section-backbone-tidbd)
- [Paper 2 — Xu, van Hasselt, Silver (2018), Meta-Gradient Reinforcement Learning](#paper-2--xu-van-hasselt-silver-2018-meta-gradient-reinforcement-learning)
  - [Phase 1 — Foundational Overview](#phase-1--foundational-overview-meta-gradient-rl)
  - [Phase 2 — Graduate-Level Deep Dive](#phase-2--graduate-level-deep-dive-meta-gradient-rl)
  - [Appendix: Section-by-Section Backbone](#appendix-section-by-section-backbone-meta-gradient-rl)
- [Cross-paper synthesis — TIDBD modulates the learner, Xu modulates the target](#cross-paper-synthesis--tidbd-modulates-the-learner-xu-modulates-the-target)

---

## Paper 1 — Kearney et al. (2018), TIDBD: Adapting Temporal-Difference Step-sizes Through Stochastic Meta-descent

**PDF:** `docs/project/references/Learning_Rate/sources/Kearney et al. 2018 - TIDBD - Adapting temporal-difference step-sizes through stochastic meta-descent.pdf`
**Venue:** Submitted to NeurIPS 2017 (arXiv:1804.03334)
**Authors:** Alex Kearney, Vivek Veeriah, Jaden B. Travnik, Richard S. Sutton, Patrick M. Pilarski (University of Alberta, RLAI Lab)
**Lineage:** Sutton (1992) IDBD → Schraudolph (1999) local gain adaptation → Dabney (2014) SID/NOSID → **this paper (TIDBD)**.

### Phase 1 — Foundational Overview (TIDBD)

#### Introduction

In TD learning the agent updates an estimated value $V(s)$ toward a target $R + \gamma V(s')$ — a *bootstrap*, meaning the target itself is built from the agent's current value function. The size of every such correction is scaled by a step size $\alpha$. Two facts make the choice of $\alpha$ awkward in TD:

1. **Bootstrapping makes the target non-stationary.** Even if the world is stationary, the agent's own evolving value estimates make the regression target drift, so a step size that worked early may be too large or too small later.
2. **Not all input features carry equal information.** With linear function approximation $V(s) = w^\top \phi(s)$, an input dimension that's just noise should receive almost no weight update; a relevant feature should receive a strong one. A single global $\alpha$ cannot express that distinction — it would have to compromise.

Existing TD step-size adapters could do one of these but not both: HL($\lambda$) and AlphaBound used a single decreasing scalar; RMSProp could go per-feature but Kearney et al. show it performs poorly on TD; Dabney's SID/NOSID could grow and shrink but stayed scalar. The paper's pitch: extend Sutton's IDBD (which already does per-feature growing-and-shrinking step sizes for *supervised* learning) to TD, by deriving the meta-gradient through the **TD error** rather than through a supervised error. The resulting algorithm is TIDBD (TD-IDBD, "tid-bid").

#### Key findings

- **Tabular grid-world (5×5).** For *every* tested initial $\alpha$ there exists a meta-step-size $\theta$ such that TIDBD beats vanilla TD, with lower variance. Even in this small stationary problem, where vectorization gives no leverage, adapting the step size online helps.
- **Mountain-Car with 10 random "distractor" features added (tile-coded, 1001-dim).** TIDBD assigns *large* step sizes to the genuine position / velocity features and step sizes ≈ 0 to the random ones. This is the paper's "representation learning by step-size adaptation" claim: separating signal from noise as a side-effect of online $\alpha_i$ tuning.
- **Real-world robot gripper-position prediction.** TIDBD with *replacing* traces beats vanilla TD, AlphaBound, and RMSProp-on-TD across every tested $\lambda$, especially low $\lambda$. The sensitivity to the meta-parameter $\theta$ is smaller than vanilla TD's sensitivity to $\alpha$ — i.e. TIDBD is easier to tune than the thing it replaces, even though it doesn't fully eliminate hyperparameter tuning.

#### Initial takeaway

If you accept that "feature relevance varies" and "TD bootstrapping makes step-size tuning non-stationary," then a per-feature, two-way-adaptive step size is the right object — not a scalar, not monotone-decreasing. TIDBD is the cleanest way to get that, and it doubles as a primitive form of representation learning because irrelevant features get suppressed.

### Phase 2 — Graduate-Level Deep Dive (TIDBD)

#### Setup: Markov reward process and the TD prediction problem

An MRP is the tuple $\langle \mathcal{S}, p, r, \gamma \rangle$ with state set $\mathcal{S}$, transition kernel $p(s' \mid s)$, reward $r(s, s')$, and discount $0 \le \gamma \le 1$. The target is the value function

$$
v^*(s) := \mathbb{E}\left[ G_t \mid S_t = s \right], \qquad G_t := \sum_{i=1}^{\infty} \gamma^{i-1} R_{t+i}.
$$

TIDBD approximates $v^*$ by a linear function

$$
V(s \mid w) = w^\top \phi(s),
$$

with $w \in \mathbb{R}^n$ and feature vector $\phi(s) \in \mathbb{R}^n$. The TD($\lambda$) prediction error and eligibility trace are

$$
\delta_t = R_{t+1} + \gamma\, w^\top \phi(s_{t+1}) - w^\top \phi(s_t), \qquad z_{t,i} = \gamma\lambda\, z_{t-1,i} + \phi_i(s_t).
$$

The vanilla TD($\lambda$) update is $w_i \leftarrow w_i + \alpha \delta_t z_{t,i}$ with a *single* scalar $\alpha$.

#### The IDBD parameterization, ported to TD

TIDBD's first move is to keep IDBD's exponential parameterization of a per-feature step size:

$$
\alpha_i = e^{\beta_i}, \qquad \beta_i \in \mathbb{R}.
$$

Two consequences. First, $\alpha_i > 0$ automatically. Second, additive updates to $\beta_i$ produce multiplicative ("geometric") updates to $\alpha_i$ — a 0.1-sized $\beta_i$ change is a $\sim 10\%$ change in $\alpha_i$ regardless of where $\alpha_i$ currently sits. This is what makes a single meta-step-size $\theta$ behave sensibly across step sizes spanning orders of magnitude.

#### Algorithm 1 (TIDBD($\lambda$))

At each step, given $(s, s', R)$:

1. $\delta \leftarrow R + \gamma\, w^\top \phi(s') - w^\top \phi(s)$
2. For each feature $i$:
   - $\beta_i \leftarrow \beta_i + \theta\, \delta\, \phi_i(s)\, H_i$ &nbsp;&nbsp; *(meta-update)*
   - $\alpha_i \leftarrow e^{\beta_i}$
   - $z_i \leftarrow \gamma\lambda\, z_i + \phi_i(s)$
   - $w_i \leftarrow w_i + \alpha_i\, \delta\, z_i$
   - $H_i \leftarrow H_i\, [1 - \alpha_i\, \phi_i(s)\, z_i]^{+} + \alpha_i\, \delta\, z_i$ &nbsp;&nbsp; *(memory-trace update)*

The novel pieces are the **meta-update on $\beta_i$** and the **auxiliary trace $H_i$**. Intuition for the meta-update: $\beta_i$ moves in the direction $\delta\, \phi_i(s)\, H_i$, where $H_i$ summarizes how previous changes to $w_i$ accumulated. If the current weight update $\delta\, \phi_i(s)$ is *aligned* with the trace of past updates $H_i$, the learner is repeatedly correcting in the same direction — that's a sign the step size has been too small, so grow $\beta_i$. If they oppose (over-shoot), shrink $\beta_i$.

#### Derivation: TIDBD as stochastic meta-descent

The starting point is the meta-objective $\tfrac{1}{2}\delta^2$ — the squared TD error — and the goal is to descend it in $\beta_i$:

$$
\beta_i(t+1) = \beta_i(t) - \tfrac{1}{2}\theta\, \frac{\partial \delta^2(t)}{\partial \beta_i}.
$$

Expand the chain rule through the weights:

$$
\frac{\partial \delta^2(t)}{\partial \beta_i} = \sum_j \frac{\partial \delta^2(t)}{\partial w_j(t)} \, \frac{\partial w_j(t)}{\partial \beta_i}.
$$

**First approximation (IDBD-style diagonal).** Assume the dominant effect of changing $\beta_i$ is on $w_i$, not on $w_{j \ne i}$. Drop the off-diagonal terms:

$$
\beta_i(t+1) \approx \beta_i(t) - \tfrac{1}{2}\theta\, \frac{\partial \delta^2(t)}{\partial w_i(t)} \, \frac{\partial w_i(t)}{\partial \beta_i}.
$$

**Second approximation (semi-gradient).** The TD target $R + \gamma V(\phi(s_{t+1}))$ depends on $w$ too, so $\partial \delta / \partial w$ is the difference of two terms and gives a *biased* gradient (Barnard, 1993). TIDBD follows Sutton & Barto (1998) and uses the **semi-gradient** — only differentiate the predicted-value term $-w_i \phi_i(t)$, treating the target as constant. Concretely:

$$
\frac{\partial \delta(t)}{\partial w_i(t)} \approx -\phi_i(t).
$$

Hence

$$
\beta_i(t+1) \approx \beta_i(t) - \theta\, \delta(t) \cdot (-\phi_i(t)) \cdot \frac{\partial w_i(t)}{\partial \beta_i} = \beta_i(t) + \theta\, \delta(t)\, \phi_i(t)\, H_i(t),
$$

where we have defined

$$
H_i(t) := \frac{\partial w_i(t)}{\partial \beta_i}.
$$

That's Algorithm 1's $\beta$-update line.

#### Deriving the $H_i$ update

$H_i$ is the *sensitivity* of $w_i$ to its own meta-parameter $\beta_i$. Since $w_i(t+1) = w_i(t) + e^{\beta_i(t+1)} \delta(t) z_i(t)$,

$$
H_i(t+1) = \frac{\partial w_i(t+1)}{\partial \beta_i} = H_i(t) + e^{\beta_i(t+1)} \delta(t) z_i(t) + e^{\beta_i(t+1)}\, \frac{\partial \delta(t)}{\partial \beta_i}\, z_i(t) + e^{\beta_i(t+1)}\, \frac{\partial z_i(t)}{\partial \beta_i}\, \delta(t).
$$

Two sub-pieces.

**Piece A: $\partial \delta / \partial \beta_i$.** Using the same diagonal-only approximation as before:

$$
\frac{\partial \delta(t)}{\partial \beta_i} = \frac{\partial}{\partial \beta_i}\left[ -V(\phi(t)) \right] = \frac{\partial}{\partial \beta_i}\left[ -\sum_j w_j \phi_j(t) \right] \approx \frac{\partial}{\partial \beta_i}\left[-w_i \phi_i(t)\right] = -H_i(t)\, \phi_i(t).
$$

**Piece B: $\partial z_i / \partial \beta_i$.** From $z_i(t+1) = \gamma\lambda\, z_i(t) + \phi_i(t)$,

$$
\frac{\partial z_i(t+1)}{\partial \beta_i} = \gamma\lambda\, \frac{\partial z_i(t)}{\partial \beta_i} = 0,
$$

since traces are initialized to zero. The trace's sensitivity to $\beta_i$ stays zero forever — a useful simplification.

Plugging both pieces back in and using $\alpha_i(t+1) = e^{\beta_i(t+1)}$:

$$
H_i(t+1) \approx H_i(t) + \alpha_i(t+1)\, \delta(t)\, z_i(t) - \alpha_i(t+1)\, H_i(t)\, \phi_i(t)\, z_i(t) = H_i(t)\bigl[1 - \alpha_i(t+1)\, \phi_i(t)\, z_i(t)\bigr] + \alpha_i(t+1)\, \delta(t)\, z_i(t).
$$

The final step in the algorithm is to **bound the bracket from below by zero**, $[\,\cdot\,]^{+}$, to prevent $H_i$ from flipping sign when $\alpha_i$ becomes large enough to over-shoot the decay term — this is a stability hack, not a derivation step, but it's what stops the meta-trace from oscillating.

#### Why the semi-gradient choice matters

The "true" gradient would also differentiate $\partial V(\phi(s_{t+1}))/\partial w$ in the bootstrap target. Carrying that term gives Barnard's true-gradient TD, which is more theoretically sound but slower and less robust in practice. TIDBD inherits the standard RL trade-off: semi-gradient bootstrapping is biased but well-behaved on the kinds of features and trajectories TD agents encounter. The cost is that no convergence theorem from supervised IDBD carries over directly — Kearney et al. are explicit that TIDBD is an empirical extension whose stability is supported by experiment, not proof.

#### Stability considerations under bootstrapping

Two interactions to flag:

1. **Accumulating vs. replacing traces.** With accumulating $z_i$, a state visited many times in quick succession piles up trace mass, so the effective update size is large; combined with TIDBD's correlation-based meta-update, this creates a feedback loop where $\beta_i$ grows too fast, $\alpha_i$ explodes, and the agent diverges. Replacing traces, which cap $z_i$ at $\phi_i$ on revisit, are much more stable.
2. **Meta-step-size $\theta$.** The paper sweeps $\theta \in (0, 0.2)$ in the grid world and $\theta \in (0, 0.02)$ in the robot task. Below the threshold TIDBD ≈ TD; above it the meta-loop oscillates. There's still no $\theta$-free version — only a recommendation that AutoStep-style normalization (Mahmood et al. 2012) might mitigate sensitivity in future work.

#### Limitations and proposed extensions (per the authors)

- **Meta-parameter $\theta$ still has to be chosen.** TIDBD shifts the tuning burden from $\alpha$ to $\theta$, and is less sensitive to its tuning knob than vanilla TD is to $\alpha$, but does not eliminate the knob. Combining TIDBD with AutoStep's normalization is named as the obvious next step.
- **Prediction only.** All three experiments are prediction tasks; control (i.e., agents acting to maximise return) is left as future work.
- **Linear function approximation only.** The diagonal approximation and the $H_i$ trace are derived assuming $V = w^\top \phi$. Generalizing to nonlinear function approximation (deep networks) would need either Schraudolph-style local gain adaptation per-parameter or a different meta-gradient formulation — TIDBD does not provide that path.

### Appendix: Section-by-Section Backbone (TIDBD)

**Abstract.** Introduces TIDBD as a vector-step-size TD adapter that generalizes IDBD; positions it against HL($\lambda$), AlphaBound (scalar only), RMSProp-on-TD (works but bad on TD), and Dabney's SID/NOSID (scalar). Three results promised: stationary tabular improvement, irrelevant-feature suppression, robot-task gains over AlphaBound + RMSProp.

**§1 — Step-size adaptation in TD learning.** Frames two design axes: scalar vs. vector step size; monotonically decreasing vs. two-way adaptive. Argues TD's bootstrapping introduces nonstationarity that demands the two-way option. Surveys prior step-size adapters and explains why none meets both criteria.

**§2 — Markov reward processes.** Standard MRP definitions $\langle \mathcal{S}, p, r, \gamma \rangle$, value function as discounted return $G_t = \sum \gamma^{i-1} R_{t+i}$. Motivates prediction tasks via robot servomotor / gripper-position anticipation.

**§3 — TIDBD (algorithmic implementation).** Algorithm 1 with $\beta$-update, $\alpha = e^{\beta}$, eligibility trace $z$, weight update $w$, and the auxiliary memory trace $H$. Intuition: correlated updates ⇒ grow $\beta$; anti-correlated ⇒ shrink. Accumulating-vs-replacing-traces distinction flagged.

**§4 — TIDBD derivation.** Starts from $\partial \delta^2 / \partial \beta_i$ meta-descent; applies (i) diagonal approximation across $j$, (ii) semi-gradient bootstrapping (Barnard 1993; Sutton & Barto 1998), (iii) defines $H_i \equiv \partial w_i / \partial \beta_i$, (iv) derives $H_i$-update with sub-pieces $\partial \delta / \partial \beta_i = -H_i \phi_i$ and $\partial z_i / \partial \beta_i = 0$; (v) positive-bounding $[1 - \alpha_i \phi_i z_i]^+$ as the final stability step.

**§5 — Does TIDBD outperform ordinary TD?** Grid-world (5×5, Sutton & Barto 1998), $\lambda = 0$, $\gamma = 0.99$, 21 $\theta$ values in $(0, 0.2)$, 15000 steps × 30 runs. Result: for every initial $\alpha$ there exists a $\theta$ such that TIDBD beats vanilla TD on MSVE with lower variance.

**§6 — Can TIDBD perform representation learning?** Mountain-Car with SARSA(0)-trained driver policy used to define an MRP; tile-coded state of size 1001 including 10 random "distractor" features. TIDBD grows $\alpha_i$ on the real features (position, velocity) and keeps random-feature $\alpha_i$ near zero — interpreted as primitive feature selection.

**§7 — How robust is TIDBD?** Real-world robot-gripper prediction (replicating van Seijen et al. 2016; data from Edwards et al. 2016). Compares TIDBD vs. ordinary TD vs. AlphaBound vs. TD-with-RMSProp across $\lambda \in [0, 1]$. Cumulative absolute prediction error over 24 user-data trials. TIDBD with replacing traces wins at every $\lambda$, especially low $\lambda$. TIDBD with accumulating traces degrades at high $\lambda$ due to the trace-meta-loop feedback discussed in Phase 2.

**§8 — Conclusion, limitations, future work.** Limitations: $\theta$ tuning still required; prediction-only experiments; accumulating-trace instability at high $\lambda$. Future directions: AutoStep-style normalization for $\theta$-robustness, control tasks, better trace integration.

**References.** Notable lineage: Sutton 1992 (IDBD), Schraudolph 1999 (local gain adaptation), Dabney 2014 (SID/NOSID), Mahmood et al. 2012 (AutoStep), van Seijen et al. 2016 (true online TD), Edwards et al. 2016 (robot dataset).

---

## Paper 2 — Xu, van Hasselt, Silver (2018), Meta-Gradient Reinforcement Learning

**PDF:** `docs/project/references/Learning_Rate/sources/Xu et al. 2018 - Meta-gradient reinforcement learning.pdf`
**Venue:** NeurIPS 2018 (32nd Conference on Neural Information Processing Systems, Montréal)
**Authors:** Zhongwen Xu, Hado van Hasselt, David Silver (DeepMind)
**Lineage:** Sutton (1992) IDBD / online cross-validation → Schraudolph (1999) → Andrychowicz et al. (2016) L2L → Finn et al. (2017) MAML → **this paper** (single-lifetime meta-gradient on return parameters); contemporaneous with Zheng, Oh, Singh (2018) on intrinsic-reward meta-learning.

### Phase 1 — Foundational Overview (Meta-Gradient RL)

#### Introduction

Standard deep-RL algorithms (A2C, IMPALA, DQN, Rainbow) bake the *return* $G$ they chase into the loss function as a constant — a black-box function of the discount $\gamma$ and trace-decay $\lambda$ which the user picks once and never revisits. But the "right" return depends on what stage of learning the agent is in (it is well-known that low-$\gamma$ early helps; high-$\gamma$ late is needed) and on the local statistics of the state being visited (a noisy transition should bootstrap; a deterministic one should accumulate). Xu et al.'s thesis is that the return parameters $\eta = \{\gamma, \lambda\}$ should be *learned online*, by a separate meta-gradient that asks "did the last update improve performance under a long-horizon meta-objective?"

Crucially, this is **not** multi-task meta-learning à la MAML. There is one task, one lifetime, one stream of experience. Meta-learning means *adapting a hyperparameter within a single training run*. Sutton's 1992 "online cross-validation" idea is the precedent — apply your update on sample $\tau$, then *measure performance of the updated parameters on the next sample $\tau'$*, and back-propagate that performance signal into the hyperparameter.

#### Key findings

- **Generic chain-rule recipe.** Define an update $\theta' = \theta + f(\tau, \theta, \eta)$; define a held-out meta-objective $\bar{J}(\tau', \theta', \bar{\eta})$ that uses *fixed* reference meta-parameters $\bar{\eta}$ (typically a long-sighted $\bar{\gamma} = 1, \bar{\lambda} = 1$); take $\partial \bar{J} / \partial \eta$ via the chain rule. A single extra backward pass — through one inner update — does the job, when you approximate the gradient-accumulation matrix $A = I + \partial f / \partial \theta$ by $A = 0$ (i.e. ignore the recursion).
- **Illustrative chain MRPs.** State-dependent $\gamma$ and $\lambda$ adapt correctly: $\gamma$ goes low in "noise" states and high in "signal" states; $\lambda$ goes low in states whose value is well-known and high in noisier states. The meta-gradient finds the right per-state values automatically.
- **Atari at scale, 57 games, 200M frames, IMPALA backbone.**
  - IMPALA baseline (γ = 0.995): median human-normalised score 211.9% (human-starts) / 257.1% (no-ops).
  - Meta-gradient adapting $\gamma$ alone (cross-validated $\bar{\gamma} = 0.995$): **267.9% / 275.5%**.
  - Meta-gradient adapting $\{\gamma, \lambda\}$ jointly: **292.9% / 287.6%**.
  - Both numbers beat Rainbow (153% / 223%) at the same frame budget.
- **The UVFA-style conditioning trick is essential.** Adapting $\gamma$ without conditioning the value function and policy on (an embedding of) $\gamma$ collapses performance to 183% — below baseline. The reason: as $\eta$ drifts, the value function trained against the old $\eta$ is stale. Feeding $\eta$ as an input lets $v_\theta(s, e_\eta)$ and $\pi_\theta(s, e_\eta)$ adapt their predictions in step.

#### Initial takeaway

The paper makes "online cross-validation of $\eta$" tractable at Atari scale by (i) approximating the recursive gradient with a single-step ($A = 0$) shortcut, (ii) conditioning the value and policy networks on $\eta$ to absorb non-stationarity, and (iii) reusing each n-step trajectory as its own held-out validation set. The headline lesson: *the return is not a fixed object — treating it as a learnable function of the agent's experience yields large gains on hard, long-horizon problems*.

### Phase 2 — Graduate-Level Deep Dive (Meta-Gradient RL)

#### The generic meta-gradient framework

Let the agent parameters be $\theta$ and the meta-parameters be $\eta$. An update function $f$ produces post-update parameters

$$
\theta' = \theta + f(\tau, \theta, \eta),
$$

from a trajectory $\tau = \{S_t, A_t, R_t, \ldots\}$. The meta-objective $\bar{J}(\tau', \theta', \bar{\eta})$ measures how well $\theta'$ performs on a held-out trajectory $\tau'$, using *fixed reference* meta-parameters $\bar{\eta}$. The meta-gradient is, by chain rule,

$$
\frac{\partial \bar{J}(\tau', \theta', \bar{\eta})}{\partial \eta} = \frac{\partial \bar{J}(\tau', \theta', \bar{\eta})}{\partial \theta'} \cdot \frac{d\theta'}{d\eta}.
$$

#### Computing $d\theta' / d\eta$ — the trace and its approximation

Because successive parameter updates accumulate additively, the total derivative $d\theta'/d\eta$ obeys a recursion (Williams & Zipser 1989). Differentiating $\theta' = \theta + f(\tau, \theta, \eta)$:

$$
\frac{d\theta'}{d\eta} = \frac{d\theta}{d\eta} + \frac{\partial f(\tau, \theta, \eta)}{\partial \theta}\, \frac{d\theta}{d\eta} + \frac{\partial f(\tau, \theta, \eta)}{\partial \eta} = \left( I + \frac{\partial f(\tau, \theta, \eta)}{\partial \theta} \right) \frac{d\theta}{d\eta} + \frac{\partial f(\tau, \theta, \eta)}{\partial \eta}.
$$

Writing $z = d\theta/d\eta$ and $z' = d\theta'/d\eta$, this is the recursion

$$
z' = A\, z + \frac{\partial f(\tau, \theta, \eta)}{\partial \eta}, \qquad A = I + \frac{\partial f(\tau, \theta, \eta)}{\partial \theta}.
$$

Three problems with using $A$ exactly. First, $A$ is $n \times n$ where $n$ is the agent's parameter count — for IMPALA's deep ResNet this is millions. Second, evaluating $A$ at every step is $O(n^2)$ memory and $O(n^2)$ compute. Third, the formula assumes $\eta$ stays fixed over the accumulation horizon, which it does not.

**The practical fix.** Choose one of three increasingly aggressive approximations.

1. **Diagonal approximation (Sutton 1992; Schraudolph 1999):** replace $\partial f / \partial \theta$ by a diagonal estimate $\hat{\partial f}/\hat{\partial \theta}$. Cheap, but loses cross-parameter influence.
2. **Decay the trace into the past:** $A \leftarrow \mu(I + \partial f/\partial \theta)$ with $\mu \in [0, 1]$. Damps the assumption that $\eta$ has been fixed forever.
3. **Single-step approximation $A = 0$** (equivalently $\mu = 0$): keep only the *direct* effect of $\eta$ on a single update; ignore the recursion. This is what the paper uses for the Atari experiments. It reduces the whole machinery to one extra backward pass.

With $A = 0$:

$$
z' = \frac{\partial f(\tau, \theta, \eta)}{\partial \eta},
$$

and the meta-update is

$$
\Delta \eta = -\beta\, \frac{\partial \bar{J}(\tau', \theta', \bar{\eta})}{\partial \theta'}\, z',
$$

where $\beta$ is the meta-learning rate. One backward pass to get $\partial \bar{J} / \partial \theta'$ (the standard policy-gradient grad), one backward pass to get $\partial f / \partial \eta$ (differentiating through the *return computation* inside the inner-loop loss), one dot product. Cheap.

#### The return as a parametric function

The agent's return $g_\eta(\tau)$ is rewritten as an *explicit* function of $\eta$, so it is differentiable. Two canonical forms.

**n-step return** ($\eta = \{\gamma\}$):

$$
g_\eta(\tau_t) = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \ldots + \gamma^{n-1} R_{t+n} + \gamma^n v_\theta(S_{t+n}).
$$

**$\lambda$-return** ($\eta = \{\gamma, \lambda\}$), recursively:

$$
g_\eta(\tau_t) = R_{t+1} + \gamma\, (1 - \lambda)\, v_\theta(S_{t+1}) + \gamma \lambda\, g_\eta(\tau_{t+1}).
$$

The recursive form shows the **gating interpretation**: $\gamma = 0$ terminates the return (collect only $R_{t+1}$), $\lambda = 0$ bootstraps immediately (collect $R_{t+1} + \gamma v_\theta(S_{t+1})$), $\gamma = \lambda = 1$ continues onto the next step with no bootstrap. Adapting $\eta$ is literally adjusting the per-step bootstrap-vs-rollout balance.

To avoid the constraint $\eta \in [0, 1]$ being violated by SGD updates, Xu et al. parameterize the meta-parameters as logits, $\eta = \sigma(x)$ with the logistic $\sigma(x) = 1/(1 + e^{-x})$. The meta-learner stores and updates $x$; the algorithm sees $\eta$.

#### Worked instantiation 1: TD($\lambda$) prediction

Inner-loop objective:

$$
J(\tau, \theta, \eta) = \bigl(g_\eta(\tau) - v_\theta(S)\bigr)^2, \qquad \frac{\partial J}{\partial \theta} = -2\, \bigl(g_\eta(\tau) - v_\theta(S)\bigr)\, \frac{\partial v_\theta(S)}{\partial \theta},
$$

with the standard TD($\lambda$) **semi-gradient** treating $g_\eta$ as constant w.r.t. $\theta$. The inner-loop update is

$$
f(\tau, \theta, \eta) = -\frac{\alpha}{2}\frac{\partial J}{\partial \theta} = \alpha\, \bigl(g_\eta(\tau) - v_\theta(S)\bigr)\, \frac{\partial v_\theta(S)}{\partial \theta}.
$$

The crucial derivative is

$$
\frac{\partial f(\tau, \theta, \eta)}{\partial \eta} = -\frac{\alpha}{2}\, \frac{\partial^2 J(\tau, \theta, \eta)}{\partial \theta\, \partial \eta} = \alpha\, \frac{\partial g_\eta(\tau)}{\partial \eta}\, \frac{\partial v_\theta(S)}{\partial \theta}.
$$

The first factor $\partial g_\eta(\tau) / \partial \eta$ is just the derivative of the $\lambda$-return w.r.t. $\gamma$ or $\lambda$ — a closed-form expression in $R_{t+i}$ and $v_\theta(S_{t+i})$. Modern autodiff frameworks compute it for free.

The held-out meta-objective is the mean-squared error of the updated $\theta'$ against a *long-sighted, unbiased* reference return $g_{\bar{\eta}}(\tau')$ with $\bar{\eta} = \{\bar{\gamma} = 1, \bar{\lambda} = 1\}$:

$$
\bar{J}(\tau', \theta', \bar{\eta}) = \bigl(g_{\bar{\eta}}(\tau') - v_{\theta'}(S')\bigr)^2, \qquad \frac{\partial \bar{J}}{\partial \theta'} = -2\, \bigl(g_{\bar{\eta}}(\tau') - v_{\theta'}(S')\bigr)\, \frac{\partial v_{\theta'}(S')}{\partial \theta'}.
$$

The agent uses (potentially short-sighted, biased) $\eta$ during the inner update; the meta-gradient judges the result against the long-sighted ideal $\bar{\eta}$ — that's the cross-validation logic.

#### Worked instantiation 2: A2C control

The A2C semi-gradient combines actor, critic, and entropy terms:

$$
-\frac{\partial J(\tau, \theta, \eta)}{\partial \theta} = \bigl(g_\eta(\tau) - v_\theta(S)\bigr)\, \frac{\partial \log \pi_\theta(A \mid S)}{\partial \theta} + b\, \bigl(g_\eta(\tau) - v_\theta(S)\bigr)\, \frac{\partial v_\theta(S)}{\partial \theta} + c\, \frac{\partial H(\pi_\theta(\cdot \mid S))}{\partial \theta},
$$

with policy-loss coefficient implicit, value-loss coefficient $b$, entropy coefficient $c$. The inner-loop update is $f(\tau, \theta, \eta) = -\alpha\, \partial J / \partial \theta$, and

$$
\frac{\partial f(\tau, \theta, \eta)}{\partial \eta} = \alpha\, \frac{\partial g_\eta(\tau)}{\partial \eta}\, \left[ \frac{\partial \log \pi_\theta(A \mid S)}{\partial \theta} + b\, \frac{\partial v_\theta(S)}{\partial \theta} \right].
$$

The entropy term does not depend on $\eta$ and drops out. The meta-objective for control is a **policy-gradient objective on a held-out trajectory**:

$$
\frac{\partial \bar{J}(\tau', \theta', \bar{\eta})}{\partial \theta'} = \bigl(g_{\bar{\eta}}(\tau') - v_{\theta'}(S')\bigr)\, \frac{\partial \log \pi_{\theta'}(A' \mid S')}{\partial \theta'},
$$

i.e. "how much would I want to push $\theta'$ in the direction of higher returns under the unbiased reference $\bar{\eta}$?" If the inner update pushed $\theta$ in a direction that aligns with that, $\partial \bar{J}/\partial \theta' \cdot z'$ is large and positive — the meta-gradient nudges $\eta$ to do more of what just happened.

#### UVFA-style conditioning of $v_\theta$ and $\pi_\theta$ on $\eta$

If $\eta$ drifts during training, the value function $v_\theta$ trained against $g_\eta$ becomes stale: a critic that learned $\gamma = 0$ values is wrong about $\gamma = 1$ values. The paper's fix borrows from Schaul et al.'s Universal Value Function Approximator: feed $\eta$ as an input.

$$
v_\theta^\eta(S) = v_\theta\bigl([S; e_\eta]\bigr), \qquad \pi_\theta^\eta(S) = \pi_\theta\bigl([S; e_\eta]\bigr),
$$

where $e_\eta$ is an embedding of $\eta$ (a small MLP), $[\cdot ; \cdot]$ is concatenation, and crucially **the gradient does not flow through $\eta$ via the embedding** — only the embedding network's parameters are updated, not $\eta$ itself via this route. The agent thus implicitly learns a family of value functions / policies indexed by $\eta$, and shifting $\eta$ across this family is a cheap inference operation rather than a slow re-fitting.

The ablation in §3.2 makes the necessity of this trick concrete: stripping the $\eta$-embedding drops Atari median score from 267.9% to 183%, **below the baseline** of 211.9%. Without conditioning, the meta-gradient signal is noisier than its benefit.

#### Production-scale implementation details (§1.5)

- **A2C → IMPALA off-policy correction.** The exact return $g_\eta(\tau)$ uses V-trace (Espeholt et al. 2018), itself a differentiable function of $\gamma$ and $\lambda$.
- **RMSProp instead of SGD in the inner update.** Substituted via automatic differentiation (Appendix C.2). The meta-gradient still passes correctly because RMSProp's update is a differentiable function of $\eta$.
- **Trajectory reuse.** Each n-step trajectory is reused twice — once as $\tau$ for the inner update, once as $\tau'$ for the meta-objective — so no extra data is needed. (Cf. Zheng et al. 2018, who do the same; the paper notes this can be problematic in highly stochastic domains because of correlated noise.)
- **Logit parameterization** $\eta = \sigma(x)$ enforces $\eta \in (0, 1)$ during free SGD on $x$.
- **Meta-batch size and meta-LR $\beta$** are tuned on six Atari games, as is standard practice for Atari hyperparameter selection in the IMPALA / Rainbow / Dueling Networks literature.

#### Comparison to TIDBD's derivation

Both papers do the *same maneuver* — differentiate a TD-style loss through an inner update to get the meta-gradient — but apply it to opposite ends:

| | TIDBD | Xu et al. |
|---|---|---|
| Hyperparameter $\eta$ being adapted | step size $\alpha_i = e^{\beta_i}$, per feature | discount $\gamma$ and trace $\lambda$, scalar (or per-state in illustrative MRP) |
| Inner-loop loss | $\tfrac{1}{2}\delta^2$ (TD error squared) | A2C objective (actor + critic + entropy) |
| Where $\eta$ enters | scales the parameter update | shapes the return inside the parameter update |
| Recursive trace | $H_i = \partial w_i / \partial \beta_i$, kept as accumulated state | $z = d\theta/d\eta$, kept as accumulated state in principle, **set to zero in practice** ($A = 0$ shortcut) |
| Semi-gradient choice | yes, on the TD bootstrap target | yes, on the inner-loop $g_\eta$ |
| Diagonal approximation | yes, $\partial w_j / \partial \beta_i \approx 0$ for $j \ne i$ | embedded in the choice $A = 0$ |
| Held-out meta-objective | none — same TD error used for inner and meta | yes — held-out trajectory $\tau'$, fixed reference $\bar{\eta}$ |

Xu's paper is essentially TIDBD's stochastic-meta-descent move generalized to (a) deep nonlinear function approximation, (b) the return parameters rather than the step size, and (c) a proper held-out cross-validation meta-objective, with the trade-off that the recursive trace gets thrown away ($A = 0$).

#### Limitations and open questions (per the authors)

- **The $A = 0$ approximation is opportunistic.** It drops everything except the direct one-step effect of $\eta$. The diagonal and decay alternatives are mentioned but not benchmarked. In principle, longer-horizon $\eta$ dependencies are not captured.
- **Trajectory reuse may be problematic in highly stochastic environments** (the paper credits Zheng et al. 2018 with the same caveat) because the inner-update noise and the meta-objective noise are then correlated.
- **State-dependent $\eta$ was tried but did not help on Atari** — only on the illustrative chain MRPs. Why a state-conditioned $\gamma$ doesn't pay off at scale is left open.
- **Best meta-learning-rate $\beta$ and meta-batch size are still tuned by hand**, on a six-game subset. The agent does not adapt its own meta-meta-parameters.
- **Single-lifetime scope.** Multi-task generalization (à la MAML) is explicitly out of scope; the paper is about within-lifetime hyperparameter adaptation.

### Appendix: Section-by-Section Backbone (Meta-Gradient RL)

**Abstract.** The return is the chief design choice in RL ($\gamma$, $\lambda$, $n$-step horizon, off-policy correction, even the rewards themselves), and is usually fixed by hand. Proposes a gradient-based meta-learner that adapts the return *online during a single lifetime* and achieves SOTA on 57 Atari games at 200M frames.

**§(Intro).** Discusses the role of the return as a proxy for the unknown true value function. Surveys design choices: $\gamma$ controls horizon (low-$\gamma$ optimization is easier but myopic); $\lambda$ trades bias for variance via bootstrap depth; n-step returns, off-policy corrections, target networks, emphatic weightings, reward clipping, intrinsic rewards are all sub-cases of "what return to chase." Many of these have been hand-scheduled in prior work.

**§1 — Meta-Gradient RL Algorithms.** Generic recipe:
- Update function $\theta' = \theta + f(\tau, \theta, \eta)$.
- Meta-objective $\bar{J}(\tau', \theta', \bar{\eta})$ on a *held-out* trajectory using *fixed* reference $\bar{\eta}$.
- Chain-rule meta-gradient $\partial \bar{J}/\partial \eta = (\partial \bar{J}/\partial \theta')(d\theta'/d\eta)$.
- Online accumulation of $z = d\theta/d\eta$ via $z' = (I + \partial f/\partial \theta) z + \partial f/\partial \eta$.
- Three approximations: diagonal $\hat{\partial f}/\hat{\partial \theta}$, decay $\mu(I + \partial f/\partial \theta)$, or single-step $A = 0$. Atari experiments use $A = 0$.
- SGD on meta-parameter: $\Delta \eta = -\beta\, (\partial \bar{J}/\partial \theta')\, z'$.

**§1.1 — Applying Meta-Gradients to Returns.** Define $g_\eta(\tau_t)$ as n-step return (with $\eta = \{\gamma\}$) or $\lambda$-return (with $\eta = \{\gamma, \lambda\}$). Gating interpretation: $\gamma = 0$ terminates, $\lambda = 0$ bootstraps, $\gamma = \lambda = 1$ continues. Both forms fully differentiable in $\eta$.

**§1.2 — Meta-Gradient Prediction.** Walk-through for TD($\lambda$):
- Inner: $J(\tau, \theta, \eta) = (g_\eta(\tau) - v_\theta(S))^2$, semi-gradient.
- $\partial f/\partial \eta = \alpha\, (\partial g_\eta/\partial \eta)\, (\partial v_\theta/\partial \theta)$.
- Meta-objective: MSE with $\bar{\eta} = \{1, 1\}$ as long-sighted reference.

**§1.3 — Meta-Gradient Control.** Walk-through for A2C:
- Inner: A2C combined objective with policy-grad, value-grad, entropy.
- Meta-objective: held-out policy-gradient objective with reference $\bar{\eta}$.
- Crucially, the meta-objective is on the *policy*'s performance, not the critic's MSE — this aligns the meta-signal with what the agent ultimately optimizes.

**§1.4 — Conditioned Value and Policy Functions.** The non-stationarity-in-$\eta$ problem and the UVFA-style fix. Concatenate $\eta$-embedding $e_\eta$ with state input; train $e_\eta$ by backprop but block gradient flow back to $\eta$ from this path. Establishes the **critical ablation hook**: stripping $e_\eta$ collapses performance.

**§1.5 — Meta-Gradient RL in Practice.** Implementation details for Atari:
- A2C objective summed over an n-step trajectory.
- RMSProp inner optimizer (no momentum), still differentiable.
- IMPALA's V-trace off-policy correction (Appendix C.1).
- Each trajectory reused for inner update and meta-validation (Appendix C.3).
- Automatic differentiation does the second-derivative work (Appendix C.2).

**§2 — Illustrative Examples.** Two chain MRPs:
- 10-step alternating "signal" / "noise" states: $R = +0.1$ in odd states, $R \sim \mathcal{N}(0, 1)$ in even states. Meta-gradient drives $\gamma$ low in even (noise) states, high in odd (signal) states.
- 9-step MRP with random reward followed by its negation: $R_{t+1} = -R_t$. Meta-gradient drives $\lambda$ low in "well-known-value" states and high in "noisier" states.
- State-dependent $\eta$ adapts correctly over $\sim 2000$ episodes; final state-conditioned values are clearly bimodal between signal and noise states.

**§3 — Deep RL Experiments.** IMPALA backbone, deep ResNet (Espeholt et al. 2018), 200M frames, 57 Atari games, ALE benchmark, median human-normalized score (Nair et al. 2015).

**§3.1 — Experiment Setup.** Two evaluation protocols (human starts, no-op starts). Configuration matched to Espeholt et al. 2018 for fair comparison. Meta-LR $\beta$, meta-batch size, embedding dim for $e_\eta$ tuned on six Atari games. Logit parameterization $\eta = \sigma(x)$.

**§3.2 — Experiment Results.**
- Four IMPALA variants: $\eta = \emptyset$ (baseline), $\eta = \{\lambda\}$, $\eta = \{\gamma\}$, $\eta = \{\gamma, \lambda\}$.
- Two baseline discounts: $\gamma = 0.99$ (Espeholt's original) and $\gamma = 0.995$ (manually tuned and better; Appendix D.1).
- Median normalized scores in Table 1:
  - Baseline IMPALA: 144.4% / 211.9% (human-starts, $\gamma = 0.99$ / $0.995$) and 191.8% / 257.1% (no-ops).
  - Meta-$\{\lambda\}$: 156.6% / 214.2% and 185.5% / 246.5% (small effect, $\lambda$ alone is not the lever).
  - Meta-$\{\gamma\}$: 233.2% / 267.9% and 280.9% / 275.5%.
  - Meta-$\{\gamma, \lambda\}$: 221.6% / **292.9%** and 242.6% / 287.6%.
- 30–80% relative improvement over baseline; absolute SOTA at the time.
- **Ablation: no $\eta$-embedding** (i.e., no UVFA conditioning) drops meta-$\{\gamma\}$ at $\bar{\gamma} = 0.995$ from 267.9% to 183% — **below baseline**.
- Comparison to Rainbow at 200M frames: Rainbow 153% / 223%, meta-gradient 292.9% / 287.6%. Caveat: many architectural differences (network depth, off-policy correction).

**§4 — Related Work.**
- Meta-learning: Schmidhuber 1987 (self-referential GP), Hochreiter et al. 2001 (RNN meta-learners), Andrychowicz et al. 2016 / Wichrowska et al. 2017 (learned optimizers), Duan et al. 2016 / Wang et al. 2016a (RL² recurrent meta-policies), Finn et al. 2017a (MAML). All are multi-task; this paper is single-task.
- Zheng et al. 2018 (NeurIPS, contemporaneous) does the same chain-rule on a *learned intrinsic reward* added to the external reward; does not condition $v_\theta$ on $\eta$, reuses samples without separation — flagged as potentially problematic in stochastic domains.
- LR adaptation: Sutton 1992 (IDBD / online cross-validation, original idea), Schraudolph 1999 (nonlinear extension), Maclaurin et al. 2015 (reverse-mode hyperparameter), Pedregosa 2016 (approximate-gradient hyperopt), Franceschi et al. 2017 (forward/reverse). All supervised-learning-flavoured.
- $\lambda$-scheduling: Singh & Dayan 1998 (bias/variance analysis), Kearns & Singh 2000 (upper bounds → schedule), Downey & Sanner 2010 (Bayesian model averaging), Konidaris et al. 2011 (TD$_\gamma$ as parameter-free estimator), White & White 2016 (greedy local-MSE-minimizing $\lambda$). All exploit i.i.d. assumptions that don't hold in deep-RL trajectories.

**§5 — Conclusion.** The meta-gradient framework generalizes beyond $\{\gamma, \lambda\}$ to any differentiable component of the return — including the reward function and the update rule itself. Hyperparameter tuning has plagued RL for decades; treating hyperparameters as meta-parameters of a differentiable update is a path to self-tuning agents that can adapt to novel environments online.

**Appendices (not all extracted in the body of this review).** A: pseudo-code. B: hyperparameters (matched to Espeholt et al. 2018 plus meta-LR $\beta$, meta-batch size, embedding dim). C.1: V-trace off-policy correction. C.2: automatic differentiation through the inner update. C.3: trajectory-reuse implementation. D.1: $\gamma = 0.995$ tuning evidence. E.1: per-game improvement table. E.2: per-game $\gamma$ / $\lambda$ adaptation traces.

---

## Cross-paper synthesis — TIDBD modulates the learner, Xu modulates the target

The two papers occupy the **two endpoints** of the design space for "meta-gradient on an RL hyperparameter."

| Axis | TIDBD (Kearney et al. 2018) | Meta-Gradient RL (Xu et al. 2018) |
|---|---|---|
| **What the meta-gradient adapts** | the **learner's step size** $\alpha_i = e^{\beta_i}$, per feature | the **return** $g_\eta$, via $\eta = \{\gamma, \lambda\}$ |
| **Where $\eta$ enters the inner-loop update** | scales the gradient: $w_i \leftarrow w_i + \alpha_i\, \delta\, z_i$ | shapes the target inside the gradient: $\delta = g_\eta - v_\theta$ |
| **Meta-objective** | same TD error $\tfrac{1}{2}\delta^2$ as inner loss (no held-out validation) | held-out trajectory $\tau'$ with fixed reference $\bar{\eta}$ (true online cross-validation) |
| **Recursive trace through inner updates** | $H_i = \partial w_i / \partial \beta_i$, accumulated explicitly | $z = d\theta/d\eta$, approximated by $A = 0$ (single-step) in deep-RL practice |
| **Function approximator** | linear $V = w^\top \phi$ | deep nonlinear $v_\theta, \pi_\theta$ (IMPALA ResNet) |
| **Non-stationarity-in-$\eta$ fix** | none needed (linear weights track $\alpha_i$ changes) | UVFA-style conditioning $v_\theta^\eta = v_\theta([S; e_\eta])$ — **necessary**, ablation drops score below baseline |
| **Per-feature / per-state granularity** | per-feature $\alpha_i$ is the *point* of the method | per-state $\eta$ works on toy MRPs but doesn't pay off at Atari scale |
| **Empirical scale demonstrated** | tabular grid-world, Mountain-Car (1001 tile features), real-robot gripper prediction | Atari 57-game suite, 200M frames, IMPALA-scale ResNet |
| **Headline empirical claim** | Beats vanilla TD / AlphaBound / RMSProp-on-TD; performs basic representation learning via $\alpha_i$ growth on signal features and $\alpha_i \to 0$ on noise features | Median human-normalised Atari score 292.9% (vs IMPALA baseline 211.9%, Rainbow 153%) at fixed 200M-frame budget |

**The unified picture.** Pick any RL update of the form $\theta' = \theta + \alpha\, g(\tau, \theta, \eta)$. The step-size $\alpha$ controls **how big a step** the learner takes; the return-shaper $\eta$ inside $g$ controls **what direction** the step is taken in. TIDBD adapts the former; Xu adapts the latter. Both use the same fundamental tool — differentiate through the inner update by the chain rule, with semi-gradient and diagonal-style approximations to keep the computation tractable.

**Implications for the project's "modulator influences optimizer or target" framing.**

- A neuromodulator-like signal that gates **learning rate** acts at TIDBD's level: it scales the inner-loop weight update, possibly per-feature or per-channel. The math infrastructure to learn such a gating online — should the modulator's gain become a meta-parameter rather than a hand-tuned scalar — is essentially TIDBD with $\beta_i$ replaced by a network output. The auxiliary trace $H_i$ would need to be carried (or approximated diagonally / set to zero, as Xu does).
- A neuromodulator-like signal that gates **the return** — discount, trace decay, even the reward shape — acts at Xu's level: it changes what the agent's loss is *aimed at*, leaving the optimizer alone. The math infrastructure is the more general meta-gradient framework with chain rule through one inner update and a held-out meta-objective. The UVFA-style $\eta$-embedding ablation is a strong warning: if the modulator's value is going to drift, the value function and policy must be *told what value the modulator is currently at*, or the meta-loop will hurt more than it helps.

The rest of the **Learning_Rate** corpus extends this picture in three other directions, all relevant to but separate from these two RL-specific papers:

- **L2O foundations** (Andrychowicz et al. 2016 *Learning to learn by gradient descent by gradient descent*; Harrison et al. 2022 *A closer look at learned optimization*; Flennerhag et al. 2019 *Warped Gradient Descent*; Li et al. 2017 *Meta-SGD*) — fully learned, RNN-based optimizers that subsume both step size and update direction into a network output; multi-task meta-learning rather than single-lifetime.
- **Direct LR adaptation** (Baydin et al. 2017 *Hypergradient descent*) — the supervised-learning analogue of Xu's chain-rule shortcut, applied to the global learning rate itself.
- **Gated / hypernetwork architectures** (Ha et al. 2016 *HyperNetworks*; Dauphin et al. 2016 *Gated Conv Nets*; Srivastava et al. 2015 *Highway Networks*, *Training Very Deep Networks*) — the architectural-modulation cousin of the meta-gradient family, where a context vector (rather than a meta-gradient) decides which weights / features get amplified.

Reading order recommendation: TIDBD first (clean linear derivation, the *origin* of stochastic meta-descent on a step size); Xu second (modern, deep, control-flavoured, generalizes the trick to the target side). Both are concise (≤ 9 pages of body content) and self-contained.

---

*Review complete. Both papers processed sequentially: TIDBD (9 pages, ~25k chars extracted) and Xu et al. 2018 (12 pages, ~37k chars extracted). Phase 1 / Phase 2 / Section-by-section backbone supplied for each; cross-paper synthesis included as the closing section.*
