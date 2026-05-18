# Gamma Reading List — Non-Exponential Discounting (D)

**Topic:** discount-function structure in RL — specifically, whether departures from the geometric form $\gamma^t$ can be made consistent with Bellman-style machinery.
**Scope of this file:** one paper, **Schultheis, Rothkopf, Koeppl (NeurIPS 2022) — *Reinforcement Learning with Non-Exponential Discounting***.
**Source:** `docs/project/references/Gamma/sources/Schultheis et al. 2022 - Reinforcement learning with non-exponential discounting.pdf`

This file is one of several "Gamma reading list" reviews in `docs/project/references/Gamma/`. It is concerned with **changing the functional form of the discount**, not with conditioning a single geometric discount on $\gamma$ as an input (UVFA / $\gamma$-Nets / $\gamma$-Models) or with decomposing a single geometric discount across timescales (TD($\Delta$)). Cross-references to those papers appear in the "Connections" section.

## Table of Contents

- [Purpose (entry point)](#purpose-entry-point)
- [Phase 1 — Foundational Overview (undergrad-level)](#phase-1--foundational-overview-undergrad-level)
  - [Why does standard RL use $\gamma^t$?](#why-does-standard-rl-use-gammat)
  - [What's wrong with $\gamma^t$?](#whats-wrong-with-gammat)
  - [The paper's proposal in one sentence](#the-papers-proposal-in-one-sentence)
  - [Initial takeaway](#initial-takeaway)
- [Phase 2 — Graduate Deep Dive](#phase-2--graduate-deep-dive)
  - [2.1 From discount factor to survival function](#21-from-discount-factor-to-survival-function)
  - [2.2 The hyperbolic discount as a Gamma mixture](#22-the-hyperbolic-discount-as-a-gamma-mixture)
  - [2.3 Time-dependent value function](#23-time-dependent-value-function)
  - [2.4 Generalized HJB equation](#24-generalized-hjb-equation)
  - [2.5 Convergence theorem for hyperbolic discounting](#25-convergence-theorem-for-hyperbolic-discounting)
  - [2.6 Contraction? — what replaces the Banach fixed-point story](#26-contraction--what-replaces-the-banach-fixed-point-story)
  - [2.7 Discrete-time Bellman equation: $\gamma$ becomes $\lambda(t) = S(t+1)/S(t)$](#27-discrete-time-bellman-equation-gamma-becomes-lambdat--st1st)
  - [2.8 Time-inconsistency and preference reversal](#28-time-inconsistency-and-preference-reversal)
  - [2.9 Collocation solver for the HJB PDE](#29-collocation-solver-for-the-hjb-pde)
  - [2.10 Inverse RL: recovering $S(t)$ from switching times](#210-inverse-rl-recovering-st-from-switching-times)
  - [2.11 Experiments — investment and line tasks](#211-experiments--investment-and-line-tasks)
- [Connections to the rest of the $\gamma$-reading list](#connections-to-the-rest-of-the-gamma-reading-list)
- [Limitations and open work](#limitations-and-open-work)
- [Appendix — Section-by-section backbone](#appendix--section-by-section-backbone)

---

## Purpose (entry point)

Why does this paper exist on the reading list? Almost every reinforcement-learning paper picks a single number — usually $\gamma = 0.99$ — and multiplies the reward $t$ steps in the future by $\gamma^t$ to value it less than reward right now. That single choice is doing three jobs at once: (i) keeping the infinite-horizon sum finite, (ii) telling the agent to prefer sooner over later, and (iii) tacitly assuming a constant per-step probability of "the world ends and you stop collecting reward." The exponential form $\gamma^t = \exp(-\lambda t)$ — what economists call **geometric** or **exponential** discounting — is mathematically convenient because it is the only memoryless discount: from the agent's point of view at any time $t$, the future "looks the same" regardless of the absolute clock value. That memorylessness is exactly what lets us write the Bellman equation $V(s) = \max_a \mathbb{E}[r + \gamma V(s')]$ with $\gamma$ a constant and prove it is a contraction with a unique fixed point.

But humans and animals don't seem to discount this way. Given a choice between $\$100$ today and $\$110$ tomorrow people often pick today, yet given a choice between $\$100$ in 30 days and $\$110$ in 31 days the same people pick the larger-later option. That is called **preference reversal**, and exponential discounting cannot produce it: under $\exp(-\lambda t)$ shifting both options forward by the same amount leaves the ratio of their discounted values unchanged. The classical fix in psychology is **hyperbolic discounting**, $S(t) = 1/(1 + kt)$ — fat-tailed compared to the exponential, much less aggressive at long horizons.

Schultheis et al. ask: **can the standard RL toolkit — Bellman / HJB equations, policy optimization, even inverse RL — be derived for an arbitrary discount function $S(t)$, not just $\exp(-\lambda t)$, while staying mathematically honest?** Their answer is yes, in continuous time, at the cost of (a) carrying time $t$ as an explicit state variable so the value function becomes $V^*(x, t)$, (b) replacing the constant $\lambda$ on the LHS of the HJB equation with a time-varying hazard rate $\alpha(t)$, and (c) accepting that the resulting PDE has to be solved numerically — they use neural-network function approximation with a collocation method. The paper is the most direct theoretical statement on the reading list of *what it costs* to leave geometric discounting behind. The rest of this review unpacks that derivation, the well-definedness conditions (hyperbolic discounting is finite-valued only when its decay exponent $\alpha_0 > 1$), the time-inconsistency this introduces, and how the paper relates to Fedus 2019 (multi-$\gamma$ approximation), Sherstan 2020 ($\gamma$ as an input), and the UVFA/USFA family.

---

## Phase 1 — Foundational Overview (undergrad-level)

### Why does standard RL use $\gamma^t$?

The reinforcement-learning objective is some version of "maximize expected total reward." In an episodic task with bounded length this is a finite sum and no discounting is needed. In an infinite-horizon task without discounting the sum can diverge: an agent that collects reward $1$ forever has infinite return, and so does another agent that wastes the first thousand steps then collects $1$ forever — the objective cannot tell them apart. Discounting fixes this by multiplying the reward $t$ steps in the future by some factor $d(t)$ with $d(0) = 1$ and $d(t) \to 0$ as $t \to \infty$, making the sum $\sum_t d(t) r_t$ converge.

Picking $d(t) = \gamma^t$ for some $\gamma \in (0, 1)$ has three advantages that explain its dominance:

1. **Finite sum.** $\sum_t \gamma^t$ converges for any bounded reward.
2. **Recursion.** $\gamma^{t+1} = \gamma \cdot \gamma^t$, which is exactly what lets you write the Bellman equation $V(s) = r(s) + \gamma \, \mathbb{E}[V(s')]$ with a *constant* $\gamma$ in front of the next-state value. Without geometric structure, the recursion picks up a time-dependent coefficient.
3. **Contraction.** The Bellman operator $T V = r + \gamma P V$ contracts in sup-norm with modulus $\gamma$, so iterating it converges to a unique $V^*$ from any starting $V$. This is the formal backbone of every value-iteration, policy-iteration, Q-learning, TD($\lambda$), DQN, and PPO convergence result.

Three is the load-bearing one. (1) and (2) hold for any discount whose ratio $d(t+1)/d(t)$ is constant — which is the geometric form by definition.

The geometric discount also has an intuitive interpretation in terms of **survival**: if at every time step there is a constant per-step probability $1 - \gamma$ that the agent "dies" or the episode terminates, then the probability of still being alive $t$ steps from now is $\gamma^t$. Discounting and termination are then the same thing, and $\gamma$ is the survival function of a **memoryless** termination time (geometric distribution in discrete time, exponential in continuous time).

### What's wrong with $\gamma^t$?

Two things, one behavioral and one theoretical.

**Behavioral.** Humans and animals are not memoryless discounters. The flagship empirical phenomenon is **preference reversal**: given a choice between a smaller reward soon and a larger reward later, subjects who pick the smaller-sooner option when both rewards are near in time *switch* to the larger-later option when both rewards are pushed far into the future by the same amount. Mathematically, the ratio
$$
\frac{d(T)}{d(T + \Delta)}
$$
depends on $T$ under hyperbolic discounting (it shrinks toward $1$ as $T$ grows) but is constant under exponential discounting. A subject whose preference between the two rewards depends on this ratio will exhibit reversal only under non-exponential discounting.

The shape that best fits human and animal data is **hyperbolic**: $d(t) = 1/(1 + kt)$, or its generalized form $1/(1 + t/\beta)^{\alpha}$. Compared to the exponential, the hyperbolic discount falls fast at first and then flattens, so the relative value of a reward two years out vs. three years out is much closer to $1$ than under any exponential fit.

**Theoretical.** Even if you do not care about modeling humans, the exponential form encodes a strong assumption: *constant, known* hazard rate. Sozou (1998) — repeatedly cited by Schultheis et al. and the rest of the reading list — showed that if instead you assume a *constant but unknown* hazard rate $\lambda$ with a Bayesian prior over $\lambda$, then marginalizing yields a non-exponential (in fact hyperbolic, for the right prior) expected discount. Hyperbolic discounting is the Bayes-optimal response to ignorance about how risky the world is. So even a "rational" agent does not get to pick $\exp(-\lambda t)$ if its hazard-rate uncertainty is non-trivial.

A second theoretical point: discounting is what makes the infinite-horizon RL objective well-defined, but **the choice of $\gamma$** silently picks an effective horizon $\sim 1/(1-\gamma)$. There is no principled reason to expect that the right horizon is the same at every state, every task, every life-stage of the agent.

### The paper's proposal in one sentence

Replace the constant discount $\gamma$ (equivalently the constant hazard $\lambda$) with an arbitrary survival function $S(t)$, accept that the value function now depends on time as well as state ($V^*(x, t)$ rather than $V^*(x)$), and derive a Hamilton–Jacobi–Bellman PDE whose LHS coefficient is the *time-varying* hazard rate $\alpha(t) = -S'(t)/S(t)$ instead of the constant $\lambda$.

### Initial takeaway

- A general-discount-function theory exists in continuous time. The price is that $V$ becomes $V(x, t)$ rather than $V(x)$, and the Bellman recursion becomes a PDE that has to be solved with function approximation rather than fixed-point iteration over a tabular operator.
- Hyperbolic discounting in particular is well-defined (finite-valued) only when its decay exponent $\alpha_0 > 1$ — otherwise the integral diverges and the standard objective is ill-posed. This is a non-trivial constraint that does not exist for exponential discounting (any $\gamma \in [0,1)$ works).
- The classical Banach-fixed-point contraction argument is replaced by a viscosity-solution existence argument from the optimal-control PDE literature (Fleming & Soner). Different tool, similar role.
- Time becomes a first-class input to the value function. This is the same architectural move as in $\gamma$-Nets (Sherstan 2020), $\gamma$-Models (Janner 2020), and UVFA / USFA (Schaul 2015, Borsa 2018) — all of which condition the value function on an extra continuous input — except that here the extra input is *clock time* rather than $\gamma$, goal, or task.

---

## Phase 2 — Graduate Deep Dive

### 2.1 From discount factor to survival function

The paper recasts the discount factor as a survival function. Let $T \in \mathbb{R}_{\geq 0}$ be a random termination time. Define
$$
F(t) = \mathbb{P}(T \leq t), \qquad S(t) = 1 - F(t) = \mathbb{P}(T > t).
$$
$S(0) = 1$, $S$ monotone decreasing, $\lim_{t \to \infty} S(t) = 0$. The **hazard rate** is
$$
\alpha(t) \;=\; \lim_{\Delta t \to 0} \frac{1}{\Delta t} \, \mathbb{P}(t \leq T < t + \Delta t \mid T \geq t) \;=\; -\frac{S'(t)}{S(t)},
$$
so equivalently
$$
S(t) \;=\; \exp\!\left( -\int_0^t \alpha(\tau) \, d\tau \right). \tag{1}
$$
A **constant** hazard $\alpha(t) = \lambda$ recovers $S(t) = \exp(-\lambda t)$, which is the unique memoryless survival function. Any non-constant $\alpha(t)$ breaks memorylessness — the conditional distribution of remaining lifetime $T - t \mid T > t$ now depends on $t$.

Schultheis et al. treat reward valuation through the survival function: a reward $r$ received at time $t$ is worth $L(r, t) = S(t) \cdot r$, with $S$ interpreted as "probability the reward is still available at $t$." This is the standard Sozou framing — discounting is rationalized as risk of premature termination.

### 2.2 The hyperbolic discount as a Gamma mixture

If the hazard rate is constant but its value $\lambda$ is unknown with a Gamma prior $\lambda \sim \mathrm{Gamma}(\alpha_0, \beta_0)$, the marginal survival function is hyperbolic. Appendix G of the paper derives this explicitly:
$$
S(t; \alpha_0, \beta_0) \;=\; \int_{\lambda} \exp(-\lambda t) \, p(\lambda) \, d\lambda \;=\; \int_{\lambda} \exp(-\lambda t) \, \frac{\beta_0^{\alpha_0} \lambda^{\alpha_0 - 1} \exp(-\beta_0 \lambda)}{\Gamma(\alpha_0)} \, d\lambda.
$$
Combine the exponentials:
$$
= \int_{\lambda} \frac{\beta_0^{\alpha_0} \lambda^{\alpha_0 - 1} \exp(-(\beta_0 + t) \lambda)}{\Gamma(\alpha_0)} \, d\lambda \;=\; \frac{\beta_0^{\alpha_0}}{(\beta_0 + t)^{\alpha_0}} \underbrace{\int_{\lambda} \mathrm{Gamma}(\lambda; \alpha_0, \beta_0 + t) \, d\lambda}_{=1}
$$
$$
\boxed{S(t; \alpha_0, \beta_0) \;=\; \frac{1}{\left(\frac{t}{\beta_0} + 1\right)^{\alpha_0}}.} \tag{2}
$$
This is the **generalized hyperbolic** (Mazur-style) form. The special case $\alpha_0 = 1$ recovers the Mazur form $S(t) = 1/(1 + t/\beta_0)$, and $\alpha_0 \to \infty$ with $\beta_0 = \alpha_0 / \lambda$ recovers the exponential $\exp(-\lambda t)$ in the limit (sharp prior, no uncertainty).

The Bayesian posterior at time $t$ conditional on survival up to $t$ is $p(\lambda \mid t) = \mathrm{Gamma}(\alpha_0, \beta_0 + t)$, so the **posterior expected hazard rate** is
$$
\alpha(t) \;=\; \mathbb{E}[\lambda \mid T > t] \;=\; \frac{\alpha_0}{\beta_0 + t}. \tag{3}
$$
This is the hazard rate that goes into the HJB equation. It **decreases** with $t$: the longer you have survived, the safer you infer the world to be. This is the source of preference reversal.

### 2.3 Time-dependent value function

In continuous-time optimal control with state SDE
$$
dX(t) = f(X(t), u(t), t) \, dt + G(X(t), u(t), t) \, dW(t),
$$
the standard exponential-discounted value function is time-stationary:
$$
V^*(x) = \max_{u_{[t, \infty)}} \mathbb{E}\!\left[\int_t^\infty \exp(-\lambda(\tau - t)) R(X(\tau), u(\tau)) \, d\tau \,\big|\, X(t) = x \right]. \tag{4}
$$
The key technical step in the paper is to replace $\exp(-\lambda(\tau - t))$ with the **conditional survival ratio** $S(\tau)/S(t)$ — the probability of being alive at time $\tau$, given alive at time $t$:
$$
\boxed{V^*(x, t) \;=\; \max_{u_{[t, \infty)}} \mathbb{E}\!\left[\int_t^\infty \frac{S(\tau)}{S(t)} \, R(X(\tau), u(\tau), \tau) \, d\tau \,\big|\, X(t) = x \right].} \tag{6}
$$
Two structural consequences:
- $V^*$ now has explicit time argument $V^*(x, t)$, because the conditional survival ratio depends on $t$ (memorylessness is gone).
- $R$ may be time-dependent without extra cost, since we are already carrying $t$ as an argument.

### 2.4 Generalized HJB equation

The derivation follows the standard recipe — split the integral, apply Itô, take $\Delta t \to 0$ — but the factor that turns into the LHS coefficient is now the survival ratio, not the constant $\lambda$. Splitting:
$$
V^*(x, t) = \max_{u_{[t, t + \Delta t]}} \mathbb{E}\!\left[ \int_t^{t + \Delta t} \frac{S(\tau)}{S(t)} R \, d\tau + \frac{S(t + \Delta t)}{S(t)} V^*(X(t + \Delta t), t + \Delta t) \,\Big|\, X(t) = x \right].
$$
First term goes to $R(x, u, t) \, \Delta t + o(\Delta t)$. Second term, by Taylor + Itô:
$$
V^*(X(t + \Delta t), t + \Delta t) \;=\; V^*(X(t), t) \;+\; \int_t^{t + \Delta t} \!\!\left[ V^*_x f + V^*_t + \tfrac{1}{2} \mathrm{tr}\{V^*_{xx} G G^\top\} \right] d\tau \;+\; \int_t^{t + \Delta t} V^*_x G \, dW(\tau) \;+\; o(\Delta t).
$$
Subtract $V^*(x, t)$ from both sides, divide by $\Delta t$. The LHS coefficient becomes
$$
\frac{1 - S(t + \Delta t)/S(t)}{\Delta t} \;=\; \frac{1}{\Delta t} \cdot \frac{S(t) - S(t + \Delta t)}{S(t)} \;\xrightarrow{\Delta t \to 0}\; -\frac{S'(t)}{S(t)} \;=\; \alpha(t).
$$
The Brownian integral has zero expectation and dies. The result is the **generalized HJB equation**:
$$
\boxed{\alpha(t) V^*(x, t) \;=\; \max_{u \in \mathcal{U}} \!\left[ R(x, u, t) \;+\; V^*_t(x, t) \;+\; V^*_x(x, t) f(x, u, t) \;+\; \tfrac{1}{2} \mathrm{tr}\{V^*_{xx}(x, t) G(x, u, t) G(x, u, t)^\top\} \right].} \tag{7}
$$
Three observations:

1. **LHS coefficient is $\alpha(t)$, not a constant.** Constant hazard recovers the standard HJB (Eq. 4) with $\alpha(t) = \lambda$.
2. **Explicit $V^*_t$ term.** Under exponential discounting and time-stationary dynamics, $V^*_t = 0$ and the term disappears. Here it generally does not — $V$ varies in $t$ as the agent's effective horizon shifts under the changing hazard.
3. **Optimal action is time-dependent.** The argmax of the RHS depends on $t$ through both $V^*$ and $R$. Define $Q(x, u, t) := R + V^*_t + V^*_x f + \tfrac{1}{2}\mathrm{tr}\{V^*_{xx} G G^\top\}$, so $\pi^*(x, t) = \arg\max_u Q(x, u, t)$. The optimal policy is **non-stationary** even when the dynamics and reward are stationary. This is the formal expression of preference reversal — the same $(x)$ admits different optimal actions at different $t$.

### 2.5 Convergence theorem for hyperbolic discounting

Unlike exponential discounting (any $\gamma < 1$ gives a finite value function provided $R$ is bounded), hyperbolic discounting has a sharp well-definedness threshold:

**Theorem 1 (Schultheis et al.).** *Consider the hyperbolic discount function $S(t; \alpha_0, \beta_0) = 1/(t/\beta_0 + 1)^{\alpha_0}$.*

*(I) If $R(x, u, t)$ is bounded above and $\alpha_0 > 1$, the value function in Eq. (6) is well-defined (finite).*

*(II) If $R(x, u, t)$ is bounded below and $\alpha_0 \leq 1$, the value function is not well-defined (becomes infinite).*

**Proof sketch (Appendix A).** Let $r_{\sup} = \sup R$. From Eq. (6),
$$
V^*(x, t) \;\leq\; \frac{r_{\sup}}{S(t)} \int_t^\infty S(\tau) \, d\tau \;=\; \frac{r_{\sup}}{S(t)} \int_t^\infty \frac{d\tau}{(\tau/\beta_0 + 1)^{\alpha_0}} \;\leq\; \frac{r_{\sup}}{S(t)} \int_t^\infty \frac{d\tau}{(\tau/\beta_0)^{\alpha_0}} \;=\; \frac{\beta_0^{\alpha_0} r_{\sup}}{S(t)} \int_t^\infty \tau^{-\alpha_0} d\tau.
$$
The tail integral
$$
\int_t^\infty \tau^{-\alpha_0} d\tau \;=\; \left[ \frac{\tau^{1 - \alpha_0}}{1 - \alpha_0} \right]_t^\infty
$$
is finite iff $\alpha_0 > 1$. Part II is the same argument with $r_{\inf}$ from below.

**Practical consequence.** The Mazur form $S(t) = 1/(1 + kt)$ corresponds to $\alpha_0 = 1$ and is *exactly at the boundary* of ill-definedness — it produces a finite value function only when $R$ has zero supremum (no upper bound on integrated reward). In experiments the paper uses $\alpha_0 = 3$ (investment task) and $\alpha_0 = 5$ (line task), comfortably in the regime where Theorem 1(I) applies.

### 2.6 Contraction? — what replaces the Banach fixed-point story

A natural question for the project context is whether the Bellman / HJB operator under non-exponential discounting is still a contraction in some norm. The paper does **not** invoke contraction; it does not need to. The classical exponential-RL contraction proof uses the fact that the Bellman operator $T V = r + \gamma \, \mathbb{E}_P[V]$ contracts in sup-norm with modulus $\gamma < 1$. With time-varying hazard $\alpha(t)$ there is no single contraction modulus, and the operator analog — the HJB PDE viewed as a fixed-point equation — does not contract in any standard function-space norm uniformly in $t$.

The paper instead appeals to the **viscosity-solution theory of Fleming and Soner**: existence and uniqueness of a viscosity solution to the HJB PDE under regularity conditions on $f, G, R, S$ (continuity, polynomial growth, Lipschitz in $X$, bounded derivatives). This is a weaker but standard substitute. The cost is that one cannot directly transfer tabular value-iteration convergence proofs; one has to argue about PDE-residual minimization instead.

**Implication for the rest of the reading list.** Methods that retain geometric structure — Fedus 2019's mixture-of-exponential-Q-heads, Sherstan 2020's $\gamma$-as-input, Janner 2020's $\gamma$-models — all keep some form of geometric contraction available (each constituent $\gamma$-head still contracts at rate $\gamma$). Schultheis's method does not — its theoretical guarantee is at the PDE-solution level, not at the operator level.

### 2.7 Discrete-time Bellman equation: $\gamma$ becomes $\lambda(t) = S(t+1)/S(t)$

Appendix C of the paper gives the discrete-time analog. With objective
$$
J(u_0, u_1, \ldots) \;=\; \mathbb{E}\!\left[ \sum_{\tau = 0}^\infty S(\tau) R(X_\tau, u_\tau, \tau) \right],
$$
and value function
$$
V(x, t) \;=\; \max_{u_t, u_{t+1}, \ldots} \mathbb{E}\!\left[ \sum_{\tau = t}^\infty \frac{S(\tau)}{S(t)} R(X_\tau, u_\tau, \tau) \,\Big|\, X_t = x \right],
$$
the Bellman recursion is
$$
\boxed{V(x, t) \;=\; \max_u \!\left\{ R(x, u, t) \;+\; \lambda(t) \, \mathbb{E}\!\left[ V(X_{t+1}, t + 1) \mid X_t = x \right] \right\}, \qquad \lambda(t) := \frac{S(t+1)}{S(t)}.}
$$
This is the right form to compare with discrete-time references in the rest of the reading list. **The effective discount $\lambda(t)$ is now time-varying.** For exponential discounting $S(t) = \gamma^t$ gives $\lambda(t) = \gamma$ (recovers standard discrete Bellman). For hyperbolic discounting $S(t) = 1/(1 + t/\beta_0)^{\alpha_0}$:
$$
\lambda(t) \;=\; \frac{(1 + t/\beta_0)^{\alpha_0}}{(1 + (t+1)/\beta_0)^{\alpha_0}} \;=\; \left( \frac{1 + t/\beta_0}{1 + (t+1)/\beta_0} \right)^{\alpha_0}.
$$
At $t = 0$ this is $1 / (1 + 1/\beta_0)^{\alpha_0}$, and as $t \to \infty$ it tends to $1$. So the effective discount **grows toward $1$ over time** — the agent becomes more long-sighted the longer it has survived, which is again the Bayesian interpretation: hazard uncertainty resolves toward "the world is safe."

### 2.8 Time-inconsistency and preference reversal

A policy $\pi$ is **time-consistent** if the optimal action at $(x, t)$ as computed at time $t' < t$ matches the optimal action at $(x, t)$ as computed at time $t$ itself. Exponential discounting is the unique discount function under which the optimal policy is time-consistent — this is Strotz's (1955) classical result.

Under non-exponential discounting, time-consistency fails. The agent at $t' = 0$ commits to a plan that prescribes, say, "spend at $t = 10$, invest at $t = 100$." But when the agent actually reaches $t = 10$ and re-solves the HJB equation, it may now prefer to invest. This is **dynamic inconsistency** — the agent's preferences over the same options change with the passage of time.

Schultheis et al. handle this implicitly by treating the HJB solution $V^*(x, t)$ as the *commitment* solution: the policy that is optimal viewed from $t' = 0$. This is the "naive sophisticate" path in the time-inconsistency literature: solve once, follow. The alternative — a Markov-perfect equilibrium where the agent at every $t$ solves the HJB with itself as the future actor — would give a different solution (see Björk & Murgoci 2014, cited in the paper). The paper does not develop the equilibrium variant; it sticks with the commitment solution, which is the cleanest from a PDE-solving standpoint.

For the project context this distinction matters: if a non-exponentially-discounted RL agent is being deployed online and re-solved as time progresses, it will deviate from any plan made at $t = 0$ unless additional commitment machinery is added.

### 2.9 Collocation solver for the HJB PDE

The forward HJB problem in Eq. (7) is a second-order nonlinear PDE in $(x, t)$. The paper solves it by **collocation with a neural-network ansatz** — the same recipe as Sirignano & Spiliopoulos (2018, DGM) and Lutter et al. (2020). Define the PDE residual
$$
E(V, x, t) \;:=\; -\alpha(t) V(x, t) \;+\; \max_u \!\left[ R(x, u, t) \;+\; V_t \;+\; V_x f(x, u, t) \;+\; \tfrac{1}{2} \mathrm{tr}\{V_{xx} G G^\top\} \right]. \tag{8}
$$
A correct $V$ satisfies $E(V, x, t) = 0$ for all $(x, t)$. Approximate $V^*(x, t) \approx V_\psi(x, t)$ with a neural network parameterized by $\psi$ and minimize
$$
\mathcal{L}(\psi) \;=\; \sum_{i = 1}^N E(V_\psi, \hat{x}_i, \hat{t}_i)^2
$$
over samples $(\hat{x}_i, \hat{t}_i)$. Required derivatives $V_\psi$, $V_{\psi,x}$, $V_{\psi,t}$, $V_{\psi,xx}$ are obtained by autodiff through the network. **Algorithm 1** in the paper is the standard inner-loop: sample $(\hat{x}, \hat{t})$, compute forward and derivative passes, evaluate $E$, backprop $\mathcal{L}$ into $\psi$, repeat.

Two implementation tricks worth noting:

**Time reparametrization** (Appendix D). Since $t$ is unbounded but neural-network inputs should be bounded, reparametrize
$$
y(t) \;=\; 1 - \exp(-\lambda t) \;\in\; [0, 1),
$$
feed $y$ to the network as $\tilde{V}(x, y)$, and recover the time derivative by chain rule:
$$
V_t(x, t) \;=\; \tilde{V}_y(x, y(t)) \cdot y_t(t), \qquad y_t(t) \;=\; \lambda \exp(-\lambda t).
$$
The choice of $\lambda$ in $y(t)$ is a sampling-density hyperparameter (the paper uses $\lambda = 0.2$); it does *not* enter the HJB equation itself.

**Horizon annealing**. The HJB residual loss has multiple stationary points (well-known issue in PDE-solver literature). The paper anneals from short to long horizon by starting with a high effective $\alpha_0$ offset and linearly decreasing it over training. Concretely they add $+50$ to $\alpha_0$ at initialization and ramp it down to $0$ over $50{,}000$ episodes. This is analogous to curriculum or $\gamma$-annealing in standard RL.

### 2.10 Inverse RL: recovering $S(t)$ from switching times

Beyond the forward problem the paper solves the **inverse** problem: given observed action-switching events $\mathcal{D} = \{(x_i, u^-_i, u^+_i, t_i)\}_{i = 1}^N$ from a human or animal subject, recover the parameters $\theta$ of the discount function $S(t; \theta)$.

The key idea is that at the moment of a switch from $u^-$ to $u^+$, the two action values are momentarily equal:
$$
Q(x, u^-, t) \;=\; Q(x, u^+, t).
$$
The IRL objective is then
$$
F(x, u^-, u^+, t) \;=\; \left[ Q(x, u^-, t) - Q(x, u^+, t) \right]^2,
$$
which expands to
$$
F \;=\; \left[ R(x, u^+, t) - R(x, u^-, t) \;+\; V_x(x, t) (f(x, u^+, t) - f(x, u^-, t)) \;+\; \tfrac{1}{2} \mathrm{tr}\{V_{xx}(x, t) (G^+ {G^+}^\top - G^- {G^-}^\top)\} \right]^2. \tag{9}
$$
Note that $V$ here is the solution of the forward HJB at the *current* $\theta$ — so $F$ depends on $\theta$ both directly and indirectly through $V$.

The gradient $dF/d\theta$ requires $V_\theta$ — how the value function changes with the discount parameters. This is computed via the **forward sensitivity method**: differentiating the residual $E(V, x, t) = 0$ (Eq. 8) with respect to $\theta$ yields another PDE for $V_\theta$:
$$
0 \;=\; \frac{dE}{d\theta} \;=\; E_\theta \;+\; E_V V_\theta \;+\; E_{V_t} (V_\theta)_t \;+\; E_{V_x} (V_\theta)_x \;+\; E_{V_{xx}} (V_\theta)_{xx} \;=:\; H(V, V_\theta, x, t). \tag{11}
$$
**Algorithm 2** solves this second PDE the same way as Algorithm 1 (collocation + neural-net ansatz for $V_\theta$), then evaluates
$$
\frac{dF}{d\theta} \;=\; F_\theta \;+\; F_V V_\theta \;+\; F_{V_x} (V_\theta)_x \;+\; F_{V_{xx}} (V_\theta)_{xx} \tag{10}
$$
to produce a gradient for $\theta$ that is then used in an outer optimization loop.

This is the **first IRL method that recovers properties of $S(t)$ itself** rather than the reward function $R$ — a useful piece of methodological cleavage. Standard IRL (Ng & Russell 2000, Ziebart 2008, IQ-Learn 2021) assumes the discount fixed and infers $R$; this paper assumes $R$ fixed and infers $S$.

### 2.11 Experiments — investment and line tasks

The paper validates both algorithms on two toy continuous-time problems.

**Investment problem.** State $x = (x_b, x_i) \in [0, 1]^2$: account balance and interest rate. Two actions, $\{\mathrm{spend}, \mathrm{invest}\}$. Spending pays immediate reward at rate $0.1$ but leaves balance unchanged; investing increases balance at rate $0.1$ and pays no immediate reward; in both modes the agent receives ongoing interest reward $R_x = x_b \cdot x_i$. Interest rate $x_i$ follows a Gaussian SDE. Hyperbolic prior $\alpha_0 = 3$, $\beta_0 = 1$.

Observed behavior under the learned $V^*(x, t)$: at small $t$ the agent prefers to *spend* (collect immediate reward before the world ends), then preference *reverses* and the agent prefers to *invest* once enough time has elapsed that the hazard rate has dropped below the threshold. The exponential-discount baseline ($\lambda = 3$) produces a flat, time-independent policy with no reversal — the obvious comparison.

**Line problem.** State $x \in [-1, 1]$, actions $\{\mathrm{left}, \mathrm{stay}, \mathrm{right}\}$. Reward structure has a small reward at $x \approx 0.5$ and a much larger reward at $x \approx -1$. Hyperbolic prior $\alpha_0 = 5$, $\beta_0 = 1$.

Observed behavior: the agent initially moves right toward the small-sooner reward, parks there, and after enough time has passed (hazard dropped) moves left to chase the large-later reward. This is the canonical "smaller-sooner / larger-later" preference-reversal task, generalized to continuous control.

For both problems the IRL algorithm correctly recovers the $(\alpha_0, \beta_0)$ parameters used to generate simulated switching data — gradient field points toward the truth, $F$ values minimize near the truth.

The experiments are **proof-of-concept only** — both state spaces are 1–2D, action spaces are 2–3 discrete actions, and there is no comparison to deep-RL benchmarks. The contribution is theoretical.

---

## Connections to the rest of the $\gamma$-reading list

Schultheis is the most theoretically rigorous member of the reading list on the question "what happens to RL when you change the shape of the discount function." The other reading-list papers address either *approximations* of hyperbolic discounting under standard machinery (Fedus, Alexander & Brown) or *conditioning* a single geometric discount on $\gamma$ as input (Sherstan, Janner, UVFA/USFA). The contrast is sharp and worth tabulating.

| Paper | Discount form | Time / $\gamma$ enters value function how? | Bellman / contraction? | Captures preference reversal? |
|---|---|---|---|---|
| **Schultheis 2022** (this paper) | arbitrary $S(t)$, focus hyperbolic | $V(x, t)$ — clock time as explicit input | HJB PDE; viscosity-solution existence (no Banach contraction) | **Yes**, by construction |
| **Fedus 2019** [17] | mixture of exponentials approximating hyperbolic | $V_\gamma(x)$ for many fixed $\gamma$; combined post-hoc | Each $V_\gamma$ satisfies standard Bellman, contracts at rate $\gamma$ | Approximate, **static** (no $V_t$) — Schultheis explicitly calls this out |
| **Sherstan 2020** ($\gamma$-Nets) | exponential, single $\gamma$ per query | $V(s, \gamma)$ — discount as input | Standard Bellman per $\gamma$ | No (geometric per $\gamma$) |
| **Janner 2020** ($\gamma$-models) | exponential, generative state-occupancy model | $\mu_\gamma^\pi(s' \mid s)$ as a $\gamma$-conditioned distribution | Standard Bellman (TD on occupancy) | No |
| **Romoff 2019** (TD($\Delta$)) | exponential, decomposed across nearby $\gamma_k$ | $V = \sum_k W_k$, each $W_k$ a TD-bootstrapped $\gamma_k$-residual | Standard Bellman per $W_k$ | No |
| **UVFA Schaul 2015 / USFA Borsa 2018** | exponential | $V(s, g)$ / $\psi(s, a, w)$ — goal/task as input | Standard Bellman per goal | No |
| **Alexander & Brown 2010** [16] | hyperbolic, via state-dependent $\gamma$ | indirect, through $V$ magnitude | Standard TD; modified update | Approximate; cannot match a specified $S(t)$ |

The structural distinction:

- **Conditioning approaches** (Sherstan, Janner, UVFA, USFA) add an extra continuous input to $V$ but keep the underlying discount geometric per slice. They inherit standard Bellman contraction per slice.
- **Mixture approaches** (Fedus, Romoff) reconstruct a non-geometric effective discount as a *linear combination* of geometric ones. They are static at the slice level — no $V_t$ — so they cannot reproduce time-dependent preference reversal as a dynamic phenomenon. Fedus's method matches a hyperbolic discount function in *shape*, not in *time-dependent-policy* behavior.
- **State-dependent-$\gamma$ approaches** (Alexander & Brown, van Hasselt et al. 2019 [57]) make $\gamma$ depend on the current value or state. Schultheis criticizes these in §2 (Related work): "as time is only considered indirectly through the magnitude of the value function, these approaches cannot be used for finding the solution to a given specific discount function or eliciting the discount function from data."
- **Schultheis 2022** is the only one that derives the equation $V$ satisfies under a *specified* arbitrary $S(t)$ in a formally correct continuous-time way. The cost is the PDE.

**Project-level implication.** If the project's interest is in modeling *human-like* time preferences (preference reversal, risk-uncertain hazard) under construct-valid pain dynamics, Schultheis is the right theoretical anchor. If the interest is in *implementing* a tractable non-geometric discount inside an existing deep-RL stack, Fedus's mixture-of-Q-heads is the practical recipe (and is closer to the auxiliary-task framing in much of the lab's recent work). The two are not in conflict — Fedus is an approximation of the regime Schultheis characterizes exactly.

**Hand-off note.** If the project wants to actually implement non-exponential discounting inside the current Recurrent-PPO or DreamerV3 codebase, the implementation choice is not "Schultheis vs. Fedus" at the algorithm-design level — Schultheis's algorithm is continuous-time model-based and would need substantial adaptation. The pragmatic route is Fedus-style multi-$\gamma$ Q-heads as an architectural change; Schultheis is the theoretical reference for what that architecture is approximating. This is a `senior-developer` decision; flagging it here for the record.

---

## Limitations and open work

From the paper's own §6 plus my reading:

1. **Finite action space.** The maximization over $u$ in the HJB equation is evaluated by enumeration. Continuous control would need either a strictly convex action cost (allowing closed-form max as in Lutter et al. 2020) or a Lipschitz-continuous control approximation (Kim et al. 2021).
2. **Model-based only.** Requires known dynamics $f, G$. Model-free extension via TD-error learning under general discount functions is named as future work; the natural starting point is Alexander & Brown's [16] generalization, but Schultheis flags that one as theoretically deficient for inferring $S(t)$.
3. **Curse of dimensionality.** Collocation in high-dimensional $(x, t)$ scales poorly; the authors suggest adaptive collocation (DGM) or advantage updating (Baird 1994).
4. **Commitment vs. equilibrium.** The paper solves the commitment HJB. The time-inconsistent equilibrium version (Björk & Murgoci) is not derived.
5. **IRL caveats.** Switching states are assumed to be exact and noise-free; only switching *times* are noisy. The dependence of switching state on $\theta$ is neglected.
6. **No jump-diffusion.** Stock-style discontinuous returns (Merton 1976) are flagged as a straightforward extension but not carried out.
7. **No connection to RL contraction / convergence rates.** The paper gives existence (via viscosity solutions) but not rate-of-convergence results for the collocation iterates.

---

## Appendix — Section-by-section backbone

This appendix preserves the paper's original section order. The Phase 1 / Phase 2 syntheses above regroup and rephrase this material thematically.

### §1 Introduction
- Empirical observation: humans and animals prefer sooner rewards (rate-discounting). Behavior often fit better by hyperbolic than exponential discounting; the signature phenomenon is preference reversal.
- Discounting in RL serves three purposes: (i) make infinite-horizon sums converge, (ii) encode preference for earlier reward, (iii) interpretable as probability of termination (Sutton & Barto, Puterman).
- Most RL uses exponential discounting; psychology / economics literature on non-exponential discounting has remained mostly disjoint.
- Paper's contributions: (a) HJB equation for general discount function; (b) well-definedness conditions for the hyperbolic case; (c) collocation-based deep-learning solver; (d) inverse-RL algorithm for recovering $S(t)$ from data; (e) two simulated tasks.

### §2 Related Work
- Continuous-time optimal control: Stratonovich, Kushner-Dupuis, Fleming-Soner, Pontryagin; Doya 2000 for RL formulation.
- HJB solution methods: linearization (Jacobson, Tassa-Erez), path integrals (Kappen, Theodorou-Buchli-Schaal), collocation (Simpkins-Todorov), neural-net function approximation (Tassa-Erez 2007, Sirignano-Spiliopoulos / DGM, Han-Jentzen-Weinan, Lutter et al., Alt-Schultheis-Koeppl).
- Non-exponential discounting in economics / psychology: Strotz, Frederick et al., Mazur, Andersen et al.; methods of limits, adjusting-amount procedures, Bayesian-adaptive estimators (Cavagnaro, Chang).
- Rationalizations: Sozou's uncertain-hazard model, Takahashi's log-time perception, Ray & Bossaerts's biological-clock argument.
- Semi-MDPs (Howard, Korolyuk, Ross, Bradtke-Duff): general transition-time distributions but discrete state.
- Quasi-hyperbolic discounting in MDPs (Jaśkiewicz-Nowak, Björk-Murgoci): captures preference reversal but limited functional form.
- Closest relatives: **Fedus 2019 [17]** — mixture of exponential Q-heads, static $V$, no preference reversal in time. **Alexander & Brown 2010 [16]** and **van Hasselt et al. 2019 [57]** — state/value-coupled $\gamma$, indirect time-dependence, cannot recover a specified $S(t)$.
- IRL: Ng-Russell, Abbeel-Ng, Ziebart MaxEnt, IQ-Learn — all infer $R$, not $S$.

### §3 Background
- **§3.1 Survival analysis.** Random termination time $T$, CDF $F(t)$, survival $S(t) = 1 - F(t)$, hazard rate $\alpha(t) = -S'(t)/S(t)$, Eq. (1) $S(t) = \exp(-\int_0^t \alpha\, d\tau)$. Constant $\alpha = \lambda$ is the only memoryless case.
- **§3.2 Discounting and preference reversal.** Reward at $t$ valued at $L(r, t) = S(t) r$. Constant hazard $\to$ exponential discount $\to$ time-consistent preferences. Bayesian prior on $\lambda$ via $\mathrm{Gamma}(\alpha_0, \beta_0)$ yields hyperbolic $S(t)$ in Eq. (2); posterior expected hazard Eq. (3): $\alpha(t) = \alpha_0 / (\beta_0 + t)$.
- **§3.3 Optimal control.** Standard stochastic optimal control: state SDE $dX = f \, dt + G \, dW$, exponential-discounted value function Eq. (4), classical HJB. $V^*$ depends on $x$ only.

### §4 Reinforcement Learning with general discount function
- **§4.1 Technical requirements.** Strong-unique-solution of SDE: $f, G$ Lipschitz-continuous and at-most-linear-growth in $X$. Viscosity-solution existence: $f, G$ with bounded continuous derivatives in $X, t$; $R, S$ continuous, polynomially-bounded. Integral in $V^*$ converges. **Theorem 1** (proven in App A): hyperbolic discounting yields well-defined $V^*$ iff $\alpha_0 > 1$ (when $R$ bounded above).
- **§4.2 HJB equation.** Recursive value-function split, Taylor + Itô expansion, take $\Delta t \to 0$, identify hazard rate $\alpha(t)$ on LHS. Eq. (7): $\alpha(t) V^*(x, t) = \max_u [R + V^*_t + V^*_x f + \tfrac{1}{2}\mathrm{tr}\{V^*_{xx} G G^\top\}]$. Define $Q(x, u, t)$ as the RHS without max; optimal policy $\pi^*(x, t) = \arg\max_u Q$.
- **§4.3 Solving the HJB equation.** Collocation: residual $E(V, x, t)$ in Eq. (8), neural-net $V_\psi$, sample $(\hat{x}_i, \hat{t}_i)$, minimize $\sum_i E(V_\psi, \hat{x}_i, \hat{t}_i)^2$ via autodiff. Reparametrize unbounded $t$ as $y = 1 - \exp(-\lambda t)$. **Algorithm 1**.
- **§4.4 Inverse RL.** Given $\mathcal{D} = \{(x_i, u^-_i, u^+_i, t_i)\}$ from observed switching, infer $\theta$ in $S(t; \theta)$. At switch: $Q(x, u^-, t) = Q(x, u^+, t)$. Objective $F = (\Delta Q)^2$ in Eq. (9). Gradient via forward sensitivity: residual $E$ derivative $\to$ PDE for $V_\theta$ in Eq. (11); $dF/d\theta$ via Eq. (10). **Algorithm 2**.

### §5 Experiments
- **§5.1 Tasks.** Investment problem (balance + interest rate, spend vs. invest, $\alpha_0 = 3$, $\beta_0 = 1$). Line problem (1D position, left/stay/right, asymmetric reward landscape, $\alpha_0 = 5$, $\beta_0 = 1$).
- **§5.2 Results.** Learned $V(x, t)$ rises with $t$ (hazard drops $\to$ effective horizon grows). Preference reversal visible in policy maps (spend $\to$ invest; right $\to$ left). Exponential baselines show flat-in-$t$ policy with no reversal. IRL gradient field correctly points toward ground-truth $(\alpha_0, \beta_0)$; $F$ landscape minimizes near truth.

### §6 Conclusion
- Established: well-definedness conditions for hyperbolic-discount RL; HJB equation for general $S(t)$; collocation solver; inverse-RL machinery.
- Limitations: finite action space; known model required; slow at high state-dim; commitment-only HJB; switching-state-$\theta$ dependence neglected; no jump-diffusion.
- Future: continuous control via Lutter / Tassa-Erez machinery; model-free analog along Alexander-Brown lines; adaptive collocation / advantage updating for scale; human experiments.

### Appendix A. Convergence proof (Theorem 1).
- Part I: $\alpha_0 > 1 \Rightarrow$ tail integral $\int_t^\infty \tau^{-\alpha_0} d\tau$ finite, $V^*$ bounded.
- Part II: $\alpha_0 \leq 1$, lower bound on $V^*$ diverges via same integral.

### Appendix B. Full HJB derivation.
Recursive split, Itô on $V^*(X(t + \Delta t), t + \Delta t)$, divide by $\Delta t$, take limit, identify $\alpha(t)$.

### Appendix C. Discrete-time Bellman equation.
$V(x, t) = \max_u \{R(x, u, t) + \lambda(t) \mathbb{E}[V(X_{t+1}, t+1)]\}$ with $\lambda(t) = S(t+1)/S(t)$ — time-varying effective discount.

### Appendix D. Value-function approximation and collocation.
$y = 1 - \exp(-\lambda t)$ reparametrization; chain-rule for $V_t = \tilde V_y y_t$; horizon annealing — start $\alpha_0$ offset high (short horizon), decrease over training.

### Appendix E. Experiment details.
State spaces, dynamics models, reward functions, hyperbolic-prior parameters for both tasks.

### Appendix F. Hyperparameters.
Two-hidden-layer Sigmoid MLP; collocation batch $10{,}000$; $125{,}000$ episodes investment / $100{,}000$ episodes line; $\alpha_0$ offset $50 \to 0$ over $50{,}000$ episodes; Adam, lr $0.003$; baseline exponential $\lambda = $ prior mean ($3$ or $5$).

### Appendix G. Derivation of hyperbolic discount from Gamma-mixed exponential.
Closed-form integration showing $S(t; \alpha_0, \beta_0) = (1 + t/\beta_0)^{-\alpha_0}$.

### Appendix H. Discount factor as transition to terminal state.
Standard MDP-with-terminal-state framing: continuous-time termination rate $\lambda(t)$; CCDF $P(X(t) \neq \Upsilon) = S(t)$; recovers exponential CCDF for constant $\lambda$.

---

**Status:** Phase 1 + Phase 2 + backbone-appendix complete for Schultheis et al. 2022. Ready for cross-paper synthesis when companion reviews (Fedus, Sherstan, Janner, UVFA, USFA, Romoff, Alexander-Brown) are added to this `Gamma` topic folder.
