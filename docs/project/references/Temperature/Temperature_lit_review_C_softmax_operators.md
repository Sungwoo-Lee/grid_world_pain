# Temperature — Literature Review C: Softmax / Entropy Operators in RL

**Author:** literature-reviewer
**Started:** 2026-05-19
**Scope:** Mathematical structure of temperature and entropy operators in reinforcement learning — when softmax-style operators are principled (contraction-preserving, monotone, smoothness-inducing) versus when they merely "add noise."

---

## Why this collection exists (plain-language entry point)

Reinforcement learning agents repeatedly ask the same question at every state: "given my current estimate of how good each action is, what should I do next?" The cleanest mathematical answer is the **max operator** — pick the action with the highest estimated value. The max sits at the center of the **Bellman equation**, which is the recursive bookkeeping rule that says "the value of being here equals the immediate reward plus the (discounted) value of being in the best place I can reach next." When an RL algorithm repeatedly applies this rule — value iteration, Q-learning, SARSA — it is iterating a *Bellman operator* on its value table. The reason these algorithms work at all is a single mathematical fact: the max-based Bellman operator is a **contraction** under the supremum norm. Each application shrinks the gap between the current value estimate and the true optimal value by at least a factor of $\gamma$ (the discount). Banach's fixed-point theorem then guarantees: there is exactly one fixed point, and the algorithm converges to it.

So far so clean. The trouble is that the hard max is greedy — it commits 100% probability to the current best action and 0% to everything else. That kills exploration and makes the policy non-differentiable, which is fatal for gradient-based methods. The natural fix is to replace the max with a **soft** alternative: an operator that returns something *between* the max and the mean, parameterized by a "temperature" knob that interpolates between "almost-max" (low temperature, sharp) and "almost-mean" (high temperature, flat). The Boltzmann softmax — $\mathrm{boltz}_\beta(X) = \sum_i x_i \, e^{\beta x_i} / \sum_i e^{\beta x_i}$, the inverse-temperature-weighted average of values — is the field's default choice and shows up everywhere from SARSA exploration to policy-gradient output layers to inverse RL.

Here is the catch that motivates this whole review folder: **the Boltzmann softmax is *not* a contraction.** Plugging it into a Bellman backup in place of the max does not in general give you a unique fixed point, and the iteration can move *away* from any fixed point it has. The user's project conditions a temperature head on a neuromodulator-like signal, so the question "when is putting temperature inside a Bellman backup principled, and when is it just heuristic noise?" is structural, not cosmetic. The three papers reviewed here are the canonical mathematical answer:

1. **Asadi & Littman (2017)** — proves the failure of naive Boltzmann and proposes **mellowmax**, a log-sum-exp-style operator that *is* a non-expansion at every temperature.
2. **Song et al. (2018)** — pushes back on the Asadi narrative: shows the softmax Bellman operator, *despite* not being a contraction, can have benefits the contraction-only story misses (faster early convergence, error tolerance), and re-examines when the multi-fixed-point problem actually bites.
3. **Ahmed et al. (2018)** — moves to the **policy-gradient** side: shows entropy regularization smooths the optimization landscape, makes saddle points easier to escape, and connects empirical PG convergence to the entropy of the parametrized policy. This is *operator-level* entropy (in the Bellman backup) versus *gradient-level* entropy bonus (in the PG loss) — a distinction the project must keep clean.

Read in order, these three papers tell the story of how temperature/entropy moved from "useful exploration heuristic" to "object of formal analysis with provable consequences," and where the dust has not yet fully settled.

---

## Table of Contents

1. [Asadi & Littman 2017 — An Alternative Softmax Operator for RL](#paper-1--asadi--littman-2017--an-alternative-softmax-operator-for-reinforcement-learning)
   - [Phase 1 (undergrad)](#phase-1--foundational-overview-undergraduate-level)
   - [Phase 2 (graduate, full LaTeX)](#phase-2--graduate-level-deep-dive)
   - [Appendix: Section-by-Section Backbone](#appendix-1-section-by-section-backbone--asadi--littman-2017)
2. [Song et al. 2018 — Revisiting the softmax Bellman operator](#paper-2--song-et-al-2018--revisiting-the-softmax-bellman-operator-new-benefits-and-new-perspective)
   - [Phase 1 (undergrad)](#phase-1--foundational-overview-undergraduate-level-1)
   - [Phase 2 (graduate, full LaTeX)](#phase-2--graduate-level-deep-dive-1)
   - [Appendix: Section-by-Section Backbone](#appendix-2-section-by-section-backbone--song-et-al-2018)
3. [Ahmed et al. 2018 — Understanding the impact of entropy on policy optimization](#paper-3--ahmed-et-al-2018--understanding-the-impact-of-entropy-on-policy-optimization)
   - [Phase 1 (undergrad)](#phase-1--foundational-overview-undergraduate-level-2)
   - [Phase 2 (graduate, full LaTeX)](#phase-2--graduate-level-deep-dive-2)
   - [Appendix: Section-by-Section Backbone](#appendix-3-section-by-section-backbone--ahmed-et-al-2018)
4. [Cross-paper synthesis](#cross-paper-synthesis)

---

## Paper 1 — Asadi & Littman 2017 — An Alternative Softmax Operator for Reinforcement Learning

**Citation:** Asadi, K. & Littman, M. L. (2017). An Alternative Softmax Operator for Reinforcement Learning. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70.

**PDF:** `docs/project/references/Temperature/sources/Asadi and Littman 2017 - An Alternative Softmax Operator for Reinforcement Learning.pdf`

### Phase 1 — Foundational Overview (Undergraduate-Level)

**Introduction (plain-English).** SARSA, Q-learning, and value iteration are all *bootstrapping* algorithms — they update an estimate of how good a state is by looking at the (estimated) value of the next state. The classical version uses the max over the next state's actions: "I assume I will play optimally from tomorrow onward." The reason this iteration is guaranteed to converge to a single, correct answer is that the max-based Bellman update is a **non-expansion** in the supremum norm — applying it never makes two value tables further apart than they already were. Combined with the discount factor $\gamma < 1$, the update is actually a **contraction**, and Banach's fixed-point theorem says: there is exactly one solution and you will find it.

In practice, however, we often want a *soft* version of the max — one that does not commit all probability to a single action, so that the agent keeps exploring, can be trained by gradient descent, and assigns non-zero probability to every action (useful in inverse RL, where "this action was impossible" is a destructive assumption). The default soft alternative is the **Boltzmann softmax**, $\mathrm{boltz}_\beta(X) = \sum_i x_i \cdot e^{\beta x_i} / \sum_i e^{\beta x_i}$, with inverse-temperature parameter $\beta$. At $\beta \to \infty$ it acts like max; at $\beta \to 0$ it acts like the average.

**Key Findings.**
1. **Boltzmann softmax is not a non-expansion.** The authors construct a 2-state, 2-action MDP (Figure 1 in the paper) on which generalized value iteration (GVI) with Boltzmann softmax at $\beta = 16.55$ has **two distinct fixed points**. SARSA-Boltzmann on this MDP bounces forever between the two fixed points and never stabilizes (Figure 2). The vector field of GVI updates (Figure 5) shows updates can move *away* from the fixed points — the operator is genuinely expanding in places.
2. **A non-expansive alternative exists: mellowmax.** Define
   $$\mathrm{mm}_\omega(X) \;=\; \frac{1}{\omega} \log\!\left( \frac{1}{n} \sum_{i=1}^n e^{\omega x_i} \right).$$
   This is a normalized log-sum-exp. It satisfies all four properties the authors lay out for an "ideal softmax": it approximates max for large $\omega$ (Property 1), it is a non-expansion for all $\omega$ (Property 2), it is differentiable everywhere (Property 3), and it never assigns zero probability to a non-maximizing action (Property 4). Mellowmax replaces Boltzmann inside a Bellman backup without breaking convergence.
3. **A policy that pairs with mellowmax.** Mellowmax is a value operator; to *act*, one needs a distribution over actions. The authors derive the **maximum entropy mellowmax policy** by maximizing Shannon entropy subject to the constraint that the expected $Q$ value under the policy equals $\mathrm{mm}_\omega(Q(s,\cdot))$. The solution is a Boltzmann distribution $\pi(a|s) \propto e^{\beta(s) \hat{Q}(s,a)}$ — but with a **state-dependent** inverse-temperature $\beta(s)$ found by a root-finding procedure. This is the headline algorithmic device: keep the familiar Boltzmann-form *behavior* policy, but solve for $\beta(s)$ at every state so the operator-level non-expansion still holds.
4. **Experimental validation.** On 200 random MDPs, GVI under Boltzmann fails to terminate on 8 and has multiple fixed points on 3; GVI under mellowmax has 0 failures and converges faster on average. On the multi-passenger taxi domain, SARSA-mellowmax matches SARSA-Boltzmann in average return (no exploration cost from the stability fix). On Lunar Lander with REINFORCE, replacing the Boltzmann output layer with the maximum-entropy mellowmax policy improves peak return.

**Initial Takeaway.** The contraction property is not a technicality — it is what makes value iteration converge at all. If you bolt a soft operator into a Bellman backup without checking non-expansion, you may converge to garbage, oscillate, or move steadily *away* from any fixed point. Mellowmax is the clean fix: it has the same "soft max" shape Boltzmann does, with a single scalar knob that interpolates max ↔ mean, but it is built from log-sum-exp rather than from a softmax-weighted *average of inputs*, and that single change buys back the non-expansion property. The state-dependent-$\beta$ Boltzmann policy is the price you pay to use it as a *behavior* policy — it is no longer a fixed-$\beta$ Boltzmann, but a Boltzmann whose temperature is recomputed at every state to keep operator-level non-expansion intact.

### Phase 2 — Graduate-Level Deep Dive

#### Setup: four properties of an "ideal" softmax operator

The authors define a softmax operator as a parameterized family $S_\theta: \mathbb{R}^n \to \mathbb{R}$ that interpolates between $\max$ and $\mathrm{mean}$. The ideal properties are:

1. **Approximates max:** some parameter setting recovers $\max$ arbitrarily closely.
2. **Non-expansion in $\|\cdot\|_\infty$:** $|S_\theta(X) - S_\theta(Y)| \le \max_i |x_i - y_i|$ for all $X, Y$. This is the property that, when composed with the discount factor $\gamma < 1$, yields a $\gamma$-contraction Bellman operator and hence a unique fixed point.
3. **Differentiable in its inputs.**
4. **No starvation:** never assigns zero probability to any action (the associated policy has full support).

The authors first survey the standard candidates:
- $\max(X)$: satisfies (1, 2) but not (3, 4) — non-differentiable and ignores non-maximizers.
- $\mathrm{mean}(X) = \frac{1}{n}\sum_i x_i$: satisfies (2, 3, 4) but not (1) — never maximizes.
- $\mathrm{eps}_\varepsilon(X) = \varepsilon \, \mathrm{mean}(X) + (1-\varepsilon) \max(X)$: convex combination of two non-expansions, hence (2); but (3) fails because the max piece is non-differentiable.
- $\mathrm{boltz}_\beta(X) = \sum_i x_i \, e^{\beta x_i} \big/ \sum_i e^{\beta x_i}$: differentiable, satisfies (1) as $\beta \to \infty$, satisfies (4), but **fails (2)**.

The Boltzmann failure is the load-bearing fact for the rest of the paper.

#### The Boltzmann counterexample (Section 2 & 4)

The authors construct an explicit MDP (Figure 1) with two states $s_1, s_2$, two actions $a, b$, $\gamma = 0.98$. The structure (action $a$ in $s_1$ self-loops with small probability and yields small reward; action $b$ leads to $s_2$ with reward branching) is chosen so that the GVI fixed-point equation $\hat{Q} = \mathcal{T}_{\mathrm{boltz}_\beta} \hat{Q}$ has **two** roots at $\beta = 16.55$. Figure 4 plots both fixed points as a function of $\beta$; for a range of $\beta$ values, two co-existing fixed points trace out two distinct branches. Figure 5 plots the GVI vector field in $(\hat{Q}(s_1,a), \hat{Q}(s_2,b))$ space: for some initial conditions (including points *between* the two fixed points), the GVI update arrow points *away* from both fixed points. SARSA-Boltzmann (Algorithm 1) on this MDP exhibits perpetual oscillation (Figure 2).

This is the first published explicit MDP showing two distinct Boltzmann fixed points. The general fact that Boltzmann is not a non-expansion was known from Littman (1996), but no concrete counterexample with multiple fixed points had been published before.

#### The mellowmax operator (Section 5)

The proposed operator is
$$\boxed{\;\mathrm{mm}_\omega(X) \;=\; \frac{1}{\omega} \log\!\left( \frac{1}{n} \sum_{i=1}^n e^{\omega x_i} \right)\;}$$

with parameter $\omega \in \mathbb{R} \setminus \{0\}$. This is a particular instantiation of the **quasi-arithmetic mean** with generator $\phi(x) = e^{\omega x}$; it also arises in information-theoretic regularization of policies under a KL-divergence cost (Todorov 2006; Rubin et al. 2012; Fox et al. 2016) and earlier in power engineering (Safak 1993).

Compare this to Boltzmann:
- $\mathrm{boltz}_\beta(X) = \sum_i x_i \cdot \frac{e^{\beta x_i}}{\sum_j e^{\beta x_j}}$ — a **softmax-weighted average of the inputs $x_i$**.
- $\mathrm{mm}_\omega(X) = \frac{1}{\omega} \log\!\left(\frac{1}{n}\sum_i e^{\omega x_i}\right)$ — a **normalized log-sum-exp of the inputs**.

These are *different functions*. Boltzmann puts the softmax weights on the input values themselves; mellowmax exponentiates the inputs, averages, then logs. The latter is what makes the non-expansion proof go through.

#### Proof: mellowmax is a non-expansion (Section 5.1)

**Claim.** $|\mathrm{mm}_\omega(X) - \mathrm{mm}_\omega(Y)| \le \max_i |x_i - y_i|$ for all $X, Y \in \mathbb{R}^n$ and all $\omega \ne 0$.

**Proof (re-derivation, following the paper).** Let $\Delta_i = x_i - y_i$ for $i \in \{1, \ldots, n\}$. Let $i^* = \arg\max_i \Delta_i$ (assume unique, WLOG). Assume $\omega > 0$ and, by symmetry, that $x_{i^*} - y_{i^*} \ge 0$. Then

$$
\mathrm{mm}_\omega(X) - \mathrm{mm}_\omega(Y)
= \frac{1}{\omega}\log\!\left(\frac{1}{n}\sum_i e^{\omega x_i}\right) - \frac{1}{\omega}\log\!\left(\frac{1}{n}\sum_i e^{\omega y_i}\right)
= \frac{1}{\omega}\log\!\left(\frac{\sum_i e^{\omega x_i}}{\sum_i e^{\omega y_i}}\right).
$$

Write $x_i = y_i + \Delta_i$, so $e^{\omega x_i} = e^{\omega y_i} e^{\omega \Delta_i}$. Then

$$
\mathrm{mm}_\omega(X) - \mathrm{mm}_\omega(Y)
= \frac{1}{\omega}\log\!\left(\frac{\sum_i e^{\omega y_i} e^{\omega \Delta_i}}{\sum_i e^{\omega y_i}}\right).
$$

This is a $\log$ of a *probability-weighted average* of $e^{\omega \Delta_i}$ where the weights $p_i = e^{\omega y_i}/\sum_j e^{\omega y_j}$ are nonnegative and sum to one. Since $e^{\omega \Delta_i} \le e^{\omega \Delta_{i^*}}$ for every $i$ (the index $i^*$ achieves the max),

$$
\frac{\sum_i e^{\omega y_i} e^{\omega \Delta_i}}{\sum_i e^{\omega y_i}}
\;\le\; e^{\omega \Delta_{i^*}}.
$$

Taking $\log$ and dividing by $\omega > 0$:

$$
\mathrm{mm}_\omega(X) - \mathrm{mm}_\omega(Y) \;\le\; \frac{1}{\omega} \cdot \omega \, \Delta_{i^*} \;=\; \Delta_{i^*} \;=\; \max_i (x_i - y_i) \;\le\; \max_i |x_i - y_i|.
$$

The reverse inequality is identical with $X$ and $Y$ swapped, giving $|\mathrm{mm}_\omega(X) - \mathrm{mm}_\omega(Y)| \le \max_i |x_i - y_i|$. For $\omega < 0$ the inequality direction flips at the division step but the same bound holds. $\blacksquare$

**Why the analogous argument fails for Boltzmann.** For $\mathrm{boltz}_\beta$, the perturbation $X \to Y = X + \Delta$ changes both the input values *and* the softmax weights — so the difference $\mathrm{boltz}_\beta(X) - \mathrm{boltz}_\beta(Y)$ has a term from $\Delta_i$ flowing through the values and a coupled term from the softmax weights themselves redistributing. The two terms can constructively interfere, and the bound $\le \max_i |\Delta_i|$ does *not* hold. The 2-state counterexample of Section 4 is the concrete witness.

#### Limits and derivatives (Section 5.2, 5.3, 5.4)

**Max limit.** Let $m = \max_i x_i$ and $W = |\{i : x_i = m\}|$ (the number of "winners"). Then

$$
\lim_{\omega \to \infty} \mathrm{mm}_\omega(X)
= \lim_{\omega \to \infty} \frac{1}{\omega}\log\!\left(\frac{1}{n}\sum_i e^{\omega x_i}\right)
= \lim_{\omega \to \infty} \frac{1}{\omega}\log\!\left(\frac{1}{n} e^{\omega m} \sum_i e^{\omega(x_i - m)}\right).
$$

The inner sum $\sum_i e^{\omega(x_i - m)} \to W$ as $\omega \to \infty$ (winners contribute 1, losers contribute 0). So

$$
\lim_{\omega \to \infty} \mathrm{mm}_\omega(X) = \lim_{\omega \to \infty} \frac{1}{\omega}\!\left[\omega m + \log\frac{W}{n}\right] = m = \max(X). \quad\blacksquare
$$

Similarly $\omega \to -\infty$ gives $\min(X)$.

**Mean limit ($\omega \to 0$).** Both numerator and denominator of $\mathrm{mm}_\omega(X) = \frac{1}{\omega}\log\!\left(\frac{1}{n}\sum_i e^{\omega x_i}\right)$ go to zero, so apply L'Hôpital with respect to $\omega$:

$$
\lim_{\omega \to 0} \mathrm{mm}_\omega(X)
\stackrel{\mathrm{L'H}}{=} \lim_{\omega \to 0} \frac{\frac{1}{n}\sum_i x_i e^{\omega x_i}}{\frac{1}{n}\sum_i e^{\omega x_i}}
= \frac{\frac{1}{n}\sum_i x_i}{\frac{1}{n}\sum_i 1}
= \mathrm{mean}(X). \quad\blacksquare
$$

So $\mathrm{mm}_\omega$ smoothly interpolates $\min \;\leftarrow\; (\omega \to -\infty) \;\cdots\; \mathrm{mean} \;(\omega = 0)\; \cdots\; (\omega \to +\infty) \;\rightarrow\; \max$.

**Derivative in $x_i$.** Differentiate $\mathrm{mm}_\omega$ with respect to $x_i$:

$$
\frac{\partial \mathrm{mm}_\omega(X)}{\partial x_i}
= \frac{1}{\omega} \cdot \frac{\omega e^{\omega x_i}}{\sum_j e^{\omega x_j}}
= \frac{e^{\omega x_i}}{\sum_j e^{\omega x_j}} \;\ge\; 0.
$$

Two interpretations: (i) the gradient is non-negative, so mellowmax is monotone non-decreasing in each input — desirable for a Bellman backup; (ii) the gradient is exactly the *Boltzmann distribution weight* with $\beta = \omega$ — i.e., the gradient of mellowmax is the Boltzmann distribution, even though mellowmax itself is not Boltzmann. This is the connection that the maximum-entropy policy section exploits.

**Derivative in $\omega$.** With $n_\omega(X) = \log(\frac{1}{n}\sum_i e^{\omega x_i})$ and $d_\omega(X) = \omega$, $\mathrm{mm}_\omega = n_\omega / d_\omega$, and the quotient rule gives a closed form. The relevant fact is that $\mathrm{mm}_\omega$ is smooth in $\omega$ except at $\omega = 0$ (where the operator is defined by the limiting value, $\mathrm{mean}(X)$).

#### Maximum-entropy mellowmax policy (Section 6)

Mellowmax is a *value* operator; for action selection we need a distribution. The natural choice is the maximum-entropy distribution consistent with the operator's value:

$$
\pi_{\mathrm{mm}}(s) = \arg\min_\pi \;\sum_{a \in \mathcal{A}} \pi(a|s) \log \pi(a|s)
\quad\text{s.t.}\quad
\sum_a \pi(a|s) \hat{Q}(s,a) = \mathrm{mm}_\omega(\hat{Q}(s,\cdot)),
\;\;\pi(a|s) \ge 0,\;\sum_a \pi(a|s) = 1.
$$

(The original paper writes $\min \sum \pi \log \pi$, which is $-H(\pi)$; minimizing this maximizes Shannon entropy. The sign convention here matches the paper.)

This is a convex problem (the entropy is concave, the constraints are linear). Apply Lagrange multipliers:

$$
\mathcal{L}(\pi, \lambda_1, \lambda_2) = \sum_a \pi(a|s)\log\pi(a|s) - \lambda_1\!\left(\sum_a \pi(a|s) - 1\right) - \lambda_2\!\left(\sum_a \pi(a|s)\hat{Q}(s,a) - \mathrm{mm}_\omega(\hat{Q}(s,\cdot))\right).
$$

Setting $\partial \mathcal{L}/\partial \pi(a|s) = 0$:

$$
\log \pi(a|s) + 1 - \lambda_1 - \lambda_2 \hat{Q}(s,a) = 0
\;\Longrightarrow\;
\pi(a|s) = \exp(\lambda_1 - 1 + \lambda_2 \hat{Q}(s,a)).
$$

Normalizing across actions absorbs $\lambda_1 - 1$ into the partition function, yielding

$$
\boxed{\;\pi_{\mathrm{mm}}(a|s) = \frac{e^{\beta(s) \hat{Q}(s,a)}}{\sum_{a' \in \mathcal{A}} e^{\beta(s) \hat{Q}(s,a')}}, \quad\text{where } \beta(s) = \lambda_2(s) \text{ solves the constraint.}\;}
$$

The constraint $\sum_a \pi(a|s) \hat{Q}(s,a) = \mathrm{mm}_\omega(\hat{Q}(s,\cdot))$ becomes, after substituting and using the shift trick:

$$
\sum_a e^{\beta(s)(\hat{Q}(s,a) - \mathrm{mm}_\omega(\hat{Q}(s,\cdot)))}\!\left(\hat{Q}(s,a) - \mathrm{mm}_\omega(\hat{Q}(s,\cdot))\right) = 0.
$$

The authors prove this equation in $\beta$ has a unique root: as $\beta \to +\infty$, the term for the best action dominates and the function is positive; as $\beta \to -\infty$, the term for the worst action dominates and the function is negative; the derivative shows the function is monotonically increasing. So Brent's method (or any 1-D root finder) gives $\beta(s)$ in $O(1)$ per state.

**Interpretation.** The mellowmax operator is parameterized by a *single fixed* $\omega$. The maximum-entropy mellowmax *policy* is then a Boltzmann distribution with a **state-dependent** inverse temperature $\beta(s)$ that adapts so the policy's expected $\hat{Q}$ matches $\mathrm{mm}_\omega$. From the policy's external shape, it looks just like Boltzmann softmax — but the temperature is no longer a free hyperparameter; it is determined per-state by the constraint that the underlying operator is a non-expansion. This is the central algorithmic device.

#### Convergence guarantee (Section 6, closing)

The (expected) SARSA update under $\pi_{\mathrm{mm}}$ is

$$
\mathbb{E}_{\pi_{\mathrm{mm}}}[r + \gamma \hat{Q}(s',a') \mid s, a]
= \mathcal{R}(s,a) + \gamma \sum_{s'} \mathcal{P}(s,a,s') \underbrace{\sum_{a'} \pi_{\mathrm{mm}}(a'|s') \hat{Q}(s',a')}_{= \,\mathrm{mm}_\omega(\hat{Q}(s',\cdot)) \text{ by constraint}}.
$$

That is, the target equals the GVI update (Eq. 1) with $\bigotimes = \mathrm{mm}_\omega$. Because $\mathrm{mm}_\omega$ is a non-expansion, the GVI operator $\mathcal{T}_{\mathrm{mm}_\omega}: \hat{Q} \mapsto \mathcal{R} + \gamma \mathcal{P} \mathrm{mm}_\omega(\hat{Q})$ is a $\gamma$-contraction in $\|\cdot\|_\infty$ (composing a non-expansion with $\gamma$-scaling), hence Banach gives a unique fixed point, and stochastic-approximation arguments (Singh et al. 2000, Littman & Szepesvári 1996) give SARSA-mellowmax tabular convergence to it.

#### Experimental results (Section 7)

1. **Random MDPs (200 trials).** Sample $|\mathcal{S}| \in \{2, \ldots, 10\}$, $|\mathcal{A}| \in \{2, \ldots, 5\}$, randomly populate $\mathcal{R}$ and $\mathcal{P}$. Stop GVI after 1000 iterations if not converged.

   | Operator | Failed to terminate | Multiple fixed points | Avg iterations |
   |---|---|---|---|
   | $\mathrm{boltz}_\beta$ | 8 / 200 | 3 / 200 | 231.65 |
   | $\mathrm{mm}_\omega$ | 0 / 200 | 0 / 200 | 201.32 |

   Mellowmax is strictly better on all three measures.

2. **Multi-passenger taxi (Dearden et al. 1998).** SARSA-mellowmax matches SARSA-Boltzmann on average return (both significantly above SARSA-$\varepsilon$-greedy). The stability fix does not cost exploration efficiency.

3. **Lunar Lander (OpenAI Gym).** REINFORCE with a 16-unit hidden layer and an output layer that uses the maximum-entropy mellowmax policy (treating pre-activations as $\hat{Q}$ values) outperforms REINFORCE with a Boltzmann output layer at peak performance.

#### Numerical stability

The shift identity
$$
\frac{1}{\omega}\log\!\left(\frac{1}{n}\sum_i e^{\omega x_i}\right) = c + \frac{1}{\omega}\log\!\left(\frac{1}{n}\sum_i e^{\omega(x_i - c)}\right)
$$
holds for any $c$. Choosing $c = \max_i x_i$ avoids overflow — the standard log-sum-exp trick.

#### Significance for the project's temperature-conditioning question

Mellowmax answers one half of the user's project question precisely:

- **Operator-level temperature is principled when the operator is a non-expansion.** Mellowmax is; fixed-$\beta$ Boltzmann is not.
- **A modulator that conditions $\omega$ (or $\beta(s)$ in the maximum-entropy mellowmax policy) is still on solid ground at every state** — because the non-expansion holds for all $\omega \in \mathbb{R} \setminus \{0\}$, conditioning $\omega$ on any signal (modulator, state, time) does not break the contraction, as long as the resulting operator at each state is mellowmax-with-some-$\omega$. Note the policy-level $\beta(s)$ is already state-dependent in the max-entropy mellowmax policy; that is a feature, not a bug.
- **What it does *not* answer:** the policy-gradient case, where temperature appears in the loss (as an entropy bonus) rather than in a Bellman backup. That is the domain of Ahmed et al. (2018), reviewed below.

### Appendix 1: Section-by-Section Backbone — Asadi & Littman 2017

This appendix preserves the paper's original section order. Phase 1 / Phase 2 above re-organize this content by theme; the backbone is the audit trail.

**Abstract.** A softmax operator interpolates between max and mean. Boltzmann softmax is prone to misbehavior. The authors propose an alternative softmax operator that (a) is a non-expansion (ensuring convergent learning/planning), (b) is differentiable, and (c) is the basis of a SARSA variant that computes a Boltzmann policy with a *state-dependent* temperature parameter.

**Section 1 — Introduction.** Defines four operators on $X = (x_1, \ldots, x_n)$: $\max$, $\mathrm{mean}$, $\mathrm{eps}_\varepsilon = \varepsilon\,\mathrm{mean} + (1-\varepsilon)\max$, and $\mathrm{boltz}_\beta = \sum_i x_i \cdot e^{\beta x_i} / \sum_i e^{\beta x_i}$. Lists four desirable properties: approximates max (P1), non-expansion (P2), differentiable (P3), no starvation (P4). Notes $\max$ fails P3 and P4; $\mathrm{mean}$ fails P1; $\mathrm{eps}_\varepsilon$ fails P3; $\mathrm{boltz}_\beta$ fails P2.

**Section 2 — Boltzmann Misbehaves.** Runs SARSA-Boltzmann (Algorithm 1) on the 2-state MDP of Figure 1 with $\gamma = 0.98$, $\alpha = 0.1$, $\beta = 16.55$. Figure 2 shows the value estimates never stabilize. Notes this is, to the authors' knowledge, the first tabular-SARSA example of failure to converge under Boltzmann.

**Section 3 — Background.** Standard MDP definitions $\langle \mathcal{S}, \mathcal{A}, \mathcal{R}, \mathcal{P}, \gamma \rangle$, Bellman optimality. Introduces *Generalized Value Iteration (GVI)*, Algorithm 2, parameterized by an arbitrary operator $\bigotimes$:
$$
\hat{Q}(s,a) \leftarrow \mathcal{R}(s,a) + \gamma \sum_{s'} \mathcal{P}(s,a,s') \bigotimes_{a'} \hat{Q}(s',a'). \quad\text{(Eq. 1)}
$$
States that GVI converges to a unique fixed point if $\bigotimes$ is a non-expansion in $\|\cdot\|_\infty$. $\max$, $\mathrm{mean}$, $\mathrm{eps}_\varepsilon$ are non-expansions; $\mathrm{boltz}_\beta$ is not.

**Section 4 — Boltzmann Has Multiple Fixed Points.** Figure 4: for a range of $\beta$, GVI under $\mathrm{boltz}_\beta$ on the Figure 1 MDP has two distinct fixed points (red, blue branches). Figure 5: vector field of GVI updates at $\beta = 16.55$; for some initial conditions the updates move *away* from the fixed points. SARSA-Boltzmann oscillates between the two fixed points.

**Section 5 — Mellowmax and its Properties.** Defines $\mathrm{mm}_\omega(X) = \frac{1}{\omega}\log\!\left(\frac{1}{n}\sum_i e^{\omega x_i}\right)$. Notes it is a quasi-arithmetic mean and arises in KL-regularized RL.

- 5.1 **Non-expansion proof.** Re-derived in Phase 2 above.
- 5.2 **Maximization limit.** $\omega \to +\infty$ gives $\max$; $\omega \to -\infty$ gives $\min$.
- 5.3 **Derivatives.** $\partial \mathrm{mm}_\omega / \partial x_i = e^{\omega x_i} / \sum_j e^{\omega x_j} \ge 0$ (the Boltzmann weight). Closed form for $\partial \mathrm{mm}_\omega / \partial \omega$ via quotient rule.
- 5.4 **Mean limit ($\omega \to 0$).** L'Hôpital gives $\mathrm{mm}_0(X) = \mathrm{mean}(X)$.

**Section 6 — Maximum Entropy Mellowmax Policy.** Maximum-entropy distribution constrained to $\mathbb{E}_\pi[\hat{Q}] = \mathrm{mm}_\omega(\hat{Q})$, solved via Lagrange multipliers. Result: $\pi_{\mathrm{mm}}(a|s) \propto e^{\beta(s) \hat{Q}(s,a)}$ with state-dependent $\beta(s)$ found by Brent's method. Uniqueness of $\beta(s)$ proven by monotonicity. Notes the policy has Boltzmann form but with adaptive temperature; SARSA under $\pi_{\mathrm{mm}}$ is a stochastic GVI under $\mathrm{mm}_\omega$ and hence converges.

**Section 7 — Experiments on MDPs.**
- 7.1 *Random MDPs.* 200 random MDPs; mellowmax beats Boltzmann on failure rate (0 vs 8), multi-fixed-point rate (0 vs 3), average iterations (201 vs 232).
- 7.2 *Multi-passenger taxi (Dearden et al. 1998).* SARSA-mellowmax matches SARSA-Boltzmann; both beat SARSA-$\varepsilon$-greedy.
- 7.3 *Lunar Lander.* REINFORCE with maximum-entropy mellowmax output layer beats REINFORCE-Boltzmann at peak.

**Section 8 — Related Work.** Inverse RL (Bayesian IRL Ramachandran & Amir 2007; natural-gradient IRL Neu & Szepesvári 2007; MLE IRL Babes et al. 2011) — all use Boltzmann; mellowmax is a candidate replacement. Linearly solvable MDPs (Todorov 2006); G-learning (Fox et al. 2016) — both feature mellowmax-like operators arising from KL regularization. Fox et al. 2016 uses the operator in off-policy updates with $\varepsilon$-greedy behavior; this paper does on-policy SARSA.

**Section 9 — Conclusion and Future Work.** Proposes mellowmax as a drop-in for Boltzmann throughout RL. Future directions: bound the sub-optimality of GVI fixed points; extend to function approximation (where non-expansion is even more important per Gordon 1995); gradient-based IRL exploiting mellowmax's convexity.

---

## Paper 2 — Song et al. 2018 — Revisiting the softmax Bellman operator: New benefits and new perspective

**Citation:** Song, Z., Parr, R. E., & Carin, L. (2019). Revisiting the Softmax Bellman Operator: New Benefits and New Perspective. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, PMLR 97. (arXiv version 2018.)

**PDF:** `docs/project/references/Temperature/sources/Song et al. 2018 - Revisiting the softmax Bellman operator - New benefits and new perspective.pdf`

### Phase 1 — Foundational Overview (Undergraduate-Level)

**Introduction (plain-English).** Asadi & Littman (2017) made a clean case against the Boltzmann softmax Bellman operator: it is not a contraction, value iteration under it can have multiple fixed points, and a clean alternative (mellowmax) exists. The story seemed settled. Song, Parr, and Carin (2018/2019) revisit it and complicate it. Their headline empirical finding: when you take a deep Q-network (DQN) — the classic deep-RL value-based agent — and *replace* the max inside its target network with the Boltzmann softmax (giving a variant they call "S-DQN"), it works **better** than vanilla DQN on the Atari benchmark suite. It also beats Double DQN, the standard fix for the overestimation bias problem that motivated all this work in the first place. And it does so even though the softmax Bellman operator is the same one Asadi & Littman correctly identified as non-contractive.

So the puzzle is: a "broken" operator (non-contractive, sub-optimal at convergence) yields superior practical performance. Why?

**Key Findings.**

1. **Bounded sub-optimality of the softmax Bellman operator.** Even though $\mathcal{T}_{\mathrm{soft}}$ (softmax Bellman, $\tau \to \infty$ is max) is not a contraction, Song et al. prove that iterating it from any initial $Q_0$ stays within a *bounded* distance from the true optimum $Q^*$. The bound is $\frac{\gamma(m-1)}{1-\gamma} \cdot \max\!\left\{\frac{1}{\tau+2}, \frac{2 Q_{\max}}{1+e^{\tau}}\right\}$ where $m = |\mathcal{A}|$. So "non-contraction" does not mean "diverges"; it means "ends up close to the truth, but not exactly there."
2. **Exponential convergence to the standard Bellman operator in $\tau$.** $\mathcal{T}_{\mathrm{soft}} \to \mathcal{T}$ as $\tau \to \infty$ — and the rate of convergence is **exponential** in $\tau$. So setting $\tau$ moderately large (in their experiments, $\tau \in \{1, 5, 10\}$) already puts $\mathcal{T}_{\mathrm{soft}}$ very close to $\mathcal{T}$ for the purposes of the Bellman fixed point, while still keeping the softness that the authors will argue helps.
3. **Overestimation bias reduction — analytically.** Under the same noise assumptions as the DDQN paper (van Hasselt et al. 2016a) — namely, $Q_t(s,a) = V^*(s) + \varepsilon_a$ for i.i.d. mean-zero noise $\varepsilon_a$ — the softmax Bellman operator's overestimation error is *strictly smaller* than the max operator's, for every $\tau \in [0, \infty)$. They give explicit lower and upper bounds for the reduction, and prove that the overestimation error grows monotonically in $\tau$ (more max-like ⇒ more overestimation).
4. **Gradient-noise reduction.** Smaller overestimation $\Rightarrow$ smaller $|Q_\theta(s,a) - \mathcal{T} Q_{\theta^-}(s,a)|$ TD error $\Rightarrow$ smaller gradient $\nabla_\theta L$. S-DQN and S-DDQN show numerically lower $\ell_2$ gradient norm than DQN/DDQN on Atari, particularly dramatically on Asterix (where vanilla DQN's gradient explodes due to runaway overestimation; S-DQN does not).
5. **Empirical performance.** On six Atari games (Q*Bert, Ms. Pacman, Crazy Climber, Breakout, Asterix, Seaquest), S-DQN and S-DDQN consistently match or outperform DQN/DDQN, and S-DQN often beats DDQN — i.e., the softmax operator alone delivers overestimation reduction comparable to or better than the engineered double-Q trick.
6. **A direct comparison with mellowmax.** They run mellowmax in the same off-policy DQN harness (foregoing Asadi's expensive per-state Brent root-finder). Softmax matches or beats mellowmax on 5 of 6 games (Asterix is the exception). Table 3 summarizes trade-offs across the three operators: max has no overestimation reduction but no tuning; mellowmax has both contraction and overestimation reduction but needs a root-finder to produce a policy; softmax has overestimation reduction and a directly-usable policy but lacks the contraction property.

**Initial Takeaway.** The Asadi-Littman result that "Boltzmann is not a contraction" is mathematically correct, but its *practical implication* in the function-approximation regime — i.e., when $\hat{Q}$ is a neural network, not a lookup table — turns out to be more nuanced. In that regime, the dominant failure mode is not "fails to converge to the unique fixed point of an exact Bellman backup"; it is **overestimation bias** from the max compounded with neural-network noise, and the resulting gradient explosion. The softmax Bellman operator hurts you on the first axis (no contraction) but helps you on the second (less overestimation), and in deep RL the second dominates. So the "right" operator depends on what regime you are in: tabular → contraction matters most → use mellowmax or max; deep-RL function approximation → overestimation matters most → softmax is competitive (and Song et al. argue, often better).

This is the first paper to identify that the operator-choice question has a *function-approximation* twist that the tabular analysis misses.

### Phase 2 — Graduate-Level Deep Dive

#### Setup and notation

The standard Bellman operator and the softmax Bellman operator are:

$$
(\mathcal{T} Q)(s,a) = R(s,a) + \gamma \sum_{s'} P(s' | s, a) \max_{a'} Q(s', a'), \quad\text{(Eq. 2)}
$$

$$
(\mathcal{T}_{\mathrm{soft}} Q)(s, a) = R(s,a) + \gamma \sum_{s'} P(s' | s, a) \underbrace{\sum_{a'} \frac{\exp[\tau Q(s', a')]}{\sum_{\bar{a}} \exp[\tau Q(s', \bar{a})]} Q(s', a')}_{\equiv\, \mathrm{sm}_\tau(Q(s', \cdot))}. \quad\text{(Eq. 3)}
$$

So $\mathrm{sm}_\tau(Q(s,\cdot))$ is exactly Asadi & Littman's $\mathrm{boltz}_\tau(Q(s,\cdot))$ — a softmax-weighted average of the $Q$ values themselves. The mellowmax operator (Asadi & Littman 2017) for comparison:

$$
\mathrm{mm}_\omega(Q(s,\cdot)) = \frac{1}{\omega} \log\!\left(\frac{1}{m}\sum_{a'} e^{\omega Q(s, a')}\right). \quad\text{(Eq. 4)}
$$

Both $\mathrm{sm}_\tau$ and $\mathrm{mm}_\omega$ converge to $\mathrm{mean}$ as the temperature parameter $\to 0$ and to $\max$ as it $\to \infty$, and both are continuous in their parameters with support $[0, \infty)$.

Notation conventions throughout the paper: $m = |\mathcal{A}|$, $R_{\min}/R_{\max}$ are the reward extremes, $Q_{\max} = R_{\max}/(1-\gamma)$ is the max possible Q-value, and

$$
\hat{\delta}(s) \;\triangleq\; \sup_Q \max_{i, j} \big| Q(s, a_i) - Q(s, a_j) \big|
$$

is the largest possible action-value spread at state $s$ over all $Q$ that arise during value iteration.

#### Lemma 2: bounding the softmax–max gap

Sorting $Q(s, a_{[1]}) \ge Q(s, a_{[2]}) \ge \cdots \ge Q(s, a_{[m]})$ and introducing $\delta_i(s) = Q(s, a_{[1]}) - Q(s, a_{[i]}) \ge 0$ (with $\delta_1(s) = 0$), the gap between $\max$ and $\mathrm{sm}_\tau$ is:

$$
\max_a Q(s, a) - \mathrm{sm}_\tau(Q(s, \cdot)) = \frac{\sum_{i=2}^m \exp[-\tau \delta_i(s)] \, \delta_i(s)}{1 + \sum_{i=2}^m \exp[-\tau \delta_i(s)]}.
$$

**Upper bound** (used for the performance bound and convergence proof in Section 3): using the inequality $\sum_i \frac{x_i}{1 + \sum_j y_j} \le \sum_i \frac{x_i}{1 + y_i}$ for nonneg sequences,

$$
0 \le \max_a Q(s, a) - \mathrm{sm}_\tau(Q(s, \cdot)) \le (m-1) \max\!\left\{\frac{1}{\tau + 2}, \frac{2 Q_{\max}}{1 + e^\tau}\right\}.
$$

The two cases inside the $\max$ correspond to $\delta_i(s) \le 1$ (where $\frac{\delta_i}{1 + e^{\tau \delta_i}} \le \frac{1}{\tau + 2}$ via Taylor expansion of $e^{\tau \delta_i}$ around 0) and $\delta_i(s) > 1$ (where the same ratio is bounded by $\frac{2 Q_{\max}}{1 + e^\tau}$ using $\delta_i \le 2 Q_{\max}/(1 - \gamma)$).

**Lower bound:** $\max_a Q(s,a) - \mathrm{sm}_\tau(Q(s,\cdot)) \ge \frac{\hat{\delta}(s)}{m \, \exp[\tau \hat{\delta}(s)]}$.

Both bounds **decay to zero exponentially fast in $\tau$**, which is the heart of the "$\mathcal{T}_{\mathrm{soft}} \to \mathcal{T}$ at exponential rate" result (Theorem 3, part II).

#### Theorem 3: performance bound for $\mathcal{T}_{\mathrm{soft}}$

**Statement.** Let $\mathcal{T}^k Q_0$ and $\mathcal{T}_{\mathrm{soft}}^k Q_0$ denote $k$ applications of $\mathcal{T}$ and $\mathcal{T}_{\mathrm{soft}}$ respectively to an initial $Q_0$. Then:

**(I)** For every $(s, a)$,

$$
\limsup_{k \to \infty} \mathcal{T}_{\mathrm{soft}}^k Q_0(s, a) \le Q^*(s, a),
$$

$$
\liminf_{k \to \infty} \mathcal{T}_{\mathrm{soft}}^k Q_0(s, a) \ge Q^*(s, a) - \frac{\gamma(m-1)}{1-\gamma}\max\!\left\{\frac{1}{\tau + 2}, \frac{2 Q_{\max}}{1 + e^\tau}\right\}.
$$

**(II)** $\mathcal{T}_{\mathrm{soft}}^k Q_0 \to \mathcal{T}^k Q_0$ at an **exponential rate in $\tau$**: $|\mathcal{T}^k Q_0 - \mathcal{T}_{\mathrm{soft}}^k Q_0|$ decays exponentially in $\tau$.

**Proof sketch (upper bound, Part I).** By induction on $k$.

- *Base case* ($k = 1$): $(\mathcal{T} Q_0 - \mathcal{T}_{\mathrm{soft}} Q_0)(s, a) = \gamma \sum_{s'} P(s' | s, a) [\max_{a'} Q_0(s', a') - \mathrm{sm}_\tau(Q_0(s', \cdot))] \ge 0$ pointwise, since $\max \ge \mathrm{sm}_\tau$ always.

- *Inductive step:* if $\mathcal{T}^l Q_0 \ge \mathcal{T}_{\mathrm{soft}}^l Q_0$, then $\mathcal{T}^{l+1} Q_0 = \mathcal{T} \mathcal{T}^l Q_0 \ge \mathcal{T} \mathcal{T}_{\mathrm{soft}}^l Q_0 \ge \mathcal{T}_{\mathrm{soft}} \mathcal{T}_{\mathrm{soft}}^l Q_0 = \mathcal{T}_{\mathrm{soft}}^{l+1} Q_0$. The first inequality uses monotonicity of $\mathcal{T}$ (which holds because $\max$ is monotone in its arguments), and the second uses $\mathcal{T} \ge \mathcal{T}_{\mathrm{soft}}$ pointwise. Taking $k \to \infty$ in the inductive bound gives $\limsup \mathcal{T}_{\mathrm{soft}}^k Q_0 \le Q^*$.

**Proof sketch (lower bound, Part I).** Conjecture $\mathcal{T}^k Q_0 - \mathcal{T}_{\mathrm{soft}}^k Q_0 \le \sum_{j=1}^k \gamma^j \zeta$ where $\zeta = \sup_Q \max_{s} [\max_a Q(s,a) - \mathrm{sm}_\tau(Q(s, \cdot))]$ is the worst-case single-step softmax–max gap, bounded by Lemma 2. Induct on $k$; the geometric series $\sum_{j=1}^k \gamma^j = \frac{\gamma(1 - \gamma^k)}{1 - \gamma}$ gives the asymptotic bound $\frac{\gamma}{1-\gamma} \zeta$. Substituting Lemma 2's upper bound on $\zeta$ yields the stated lower bound on $\liminf \mathcal{T}_{\mathrm{soft}}^k Q_0$.

**Proof sketch (Part II — exponential rate).** From the proof of the lower bound (Eq. A8 in their supplement), the gap is bounded by

$$
\mathcal{T}^k Q_0(s,a) - \mathcal{T}_{\mathrm{soft}}^k Q_0(s,a) \le \frac{\gamma (1 - \gamma^k)}{1 - \gamma} \sum_{i=2}^m \frac{\delta_i(s)}{1 + e^{\tau \delta_i(s)}}.
$$

Using $\frac{\delta_i(s)}{1 + e^{\tau \delta_i(s)}} \le \frac{\delta_i(s)}{e^{\tau \delta_i(s)}}$ and letting $i^* \le m$ be the smallest index with $\delta_{i^*}(s) > 0$ (so $\delta_i(s) \ge \delta_{i^*}(s) > 0$ for $i \ge i^*$):

$$
\sum_{i=i^*}^m \frac{\delta_i(s)}{e^{\tau \delta_i(s)}} \le \frac{1}{e^{\tau \delta_{i^*}(s)}} \sum_{i=i^*}^m \delta_i(s).
$$

This is exponentially small in $\tau$, completing Part II. $\blacksquare$

**Interpretation.** The contraction property of $\mathcal{T}$ tells you that $\mathcal{T}^k Q_0 \to Q^*$. The non-contraction of $\mathcal{T}_{\mathrm{soft}}$ does *not* mean $\mathcal{T}_{\mathrm{soft}}^k Q_0$ diverges — it means the iterate sits in a *bounded region* around $Q^*$, with the bound shrinking exponentially in $\tau$. This is much weaker than "the iteration may wander arbitrarily far from $Q^*$." A reader who took the Asadi-Littman result to mean "Boltzmann is dangerous" should update toward "Boltzmann sits in a bounded neighborhood of $Q^*$, with neighborhood radius controllable by $\tau$."

#### Theorem 4: overestimation bias reduction

This is the result that *explains the empirical wins*.

**Setup (van Hasselt et al. 2016a assumptions).** Assume:
- **(A1)** There exists a state value $V^*(s)$ such that $Q^*(s, a) = V^*(s)$ for every action — i.e., at the optimum, all actions are equivalent. (This is the standard worst-case assumption for analyzing overestimation.)
- **(A2)** The estimation noise is additive: $Q_t(s, a) = V^*(s) + \varepsilon_a$ with $\varepsilon_a$ i.i.d., mean zero, symmetric.

**Overestimation error under max:**

$$
\mathbb{E}\!\left[\max_a Q_t(s, a) - \max_a Q^*(s, a)\right] = \mathbb{E}\!\left[\max_a \varepsilon_a\right] \ge 0,
$$

with strict inequality whenever $m \ge 2$ and the noise has any spread. This is the Thrun-Schwartz / van Hasselt overestimation bias.

**Overestimation error under softmax:**

$$
\mathbb{E}\!\left[\mathrm{sm}_\tau(Q_t(s, \cdot)) - V^*(s)\right]
= \mathbb{E}\!\left[\sum_a \frac{e^{\tau Q_t(s,a)}}{\sum_{\bar{a}} e^{\tau Q_t(s, \bar{a})}} \, (Q_t(s,a) - V^*(s))\right]
= \mathbb{E}\!\left[\sum_a \frac{e^{\tau \varepsilon_a}}{\sum_{\bar{a}} e^{\tau \varepsilon_{\bar{a}}}} \, \varepsilon_a\right]. \quad\text{(Eq. A9)}
$$

The last expression is the softmax-weighted average of mean-zero noises with weights that themselves depend on the noises. Because $\mathrm{sm}_\tau \le \max$ pointwise, this quantity is $\le \mathbb{E}[\max_a \varepsilon_a]$, giving Part (I): **softmax overestimation $\le$ max overestimation for every $\tau \ge 0$**.

**Part (II) — explicit bounds on the reduction:** the reduction $\mathbb{E}[\max_a \varepsilon_a] - \mathbb{E}[\mathrm{sm}_\tau(\cdot)]$ lies in

$$
\left[\frac{\hat{\delta}(s)}{m \, e^{\tau \hat{\delta}(s)}}, \;(m-1)\max\!\left\{\frac{1}{\tau + 2}, \frac{2 Q_{\max}}{1 + e^\tau}\right\}\right].
$$

**Part (III) — monotonicity:** Using Lemma A3 (the softmax-weighted-input function $g_x(\tau) = \frac{\sum_i x_i e^{\tau x_i}}{\sum_i e^{\tau x_i}}$ is monotonically increasing in $\tau \ge 0$ by the Cauchy-Schwarz inequality), the overestimation error of $\mathcal{T}_{\mathrm{soft}}$ is monotonically increasing in $\tau$. At $\tau = 0$, $\mathrm{sm}_0(\cdot) = \mathrm{mean}(\cdot)$ and the noise averages to (exactly) zero. At $\tau \to \infty$, $\mathrm{sm}_\tau(\cdot) \to \max(\cdot)$ and the noise drops back into the max-overestimation regime.

So $\tau$ is now a tunable knob *trading off* Bellman-optimality (large $\tau$ is close to true $\mathcal{T}$) against overestimation (small $\tau$ has less noise leakage). The "right" $\tau$ depends on the noise level.

#### Why this matters for gradient stability (Section 5.2)

The DQN training loss is

$$
\nabla_\theta L = \mathbb{E}_{s, a, r, s'}\!\left[(Q_\theta(s, a) - \mathcal{T} Q_{\theta^-}(s, a)) \nabla_\theta Q_\theta(s, a)\right]. \quad\text{(Eq. 5)}
$$

When $Q_{\theta^-}$ is overestimated by the max operator, the TD error $Q_\theta - \mathcal{T} Q_{\theta^-}$ has high magnitude, and $\nabla_\theta Q_\theta$ chases it — producing large gradients that destabilize neural-network training (Mnih et al. 2015 already noted this). The softmax operator shrinks the overestimation, shrinks the TD error, shrinks the gradient norm, and stabilizes training. Figure 5 of the paper shows this directly on Asterix: DQN's $\|\nabla_\theta L\|_2$ explodes; S-DQN's stays bounded.

This is a *non-tabular* effect — it could not show up in Asadi-Littman's setting because they did not have a deep network. It is real and is the dominant mechanism by which softmax beats max in deep RL.

#### Table 3 — operator comparison

| Operator | Bellman-optimal? | Tuning param? | Overestimation reduction? | Policy representation? | Compatible with Double-Q? |
|---|---|---|---|---|---|
| Max | Yes | No | — | Yes | Yes |
| Mellowmax | No | Yes ($\omega$) | Yes | No (needs Brent root-find) | No |
| Softmax | No | Yes ($\tau$) | Yes | Yes | Yes |

Mellowmax and softmax are essentially equivalent for the *Bellman update* — for any fixed $Q$, there is a state-specific correspondence between $\tau$ in $\mathrm{sm}_\tau$ and $\omega$ in $\mathrm{mm}_\omega$ that makes their values equal (both are continuous interpolators between $\mathrm{mean}$ and $\max$ on $[0, \infty)$). But the mapping is state-dependent and re-shuffles as $Q$ updates, so they are not the *same* operator. The clean separation in Table 3 is that softmax already *is* a probability distribution, while mellowmax has to be combined with a max-entropy step to get one. In deep RL, where you fold the output of the operator directly into a softmax over actions, that "free" policy representation is meaningful.

#### Figure 7 — the explicit trade-off

The supplementary Figure 7 plots, as a function of $\tau$ (softmax) or $\omega$ (mellowmax):
- *Left:* approximation error to the max function — both decay exponentially, but softmax converges *faster* (smaller bound at finite $\tau$).
- *Right:* overestimation error — mellowmax reduces overestimation *more* than softmax at the same parameter value.

So softmax is closer to max (less Bellman sub-optimality) but mellowmax handles overestimation better. The paper's empirical message is that, on Atari at moderate $\tau$, the trade-off favors softmax — but the trade-off is real and the right operator may depend on the noise regime.

#### What is genuinely new vs. what is re-derivation

A reader of both Asadi & Littman 2017 and Song et al. 2018 should be clear on what each paper adds:

| Claim | Asadi & Littman 2017 | Song et al. 2018 |
|---|---|---|
| Boltzmann softmax is not a non-expansion | Re-stated from Littman 1996; provides first explicit counterexample MDP | Re-stated; not the focus |
| GVI under Boltzmann has multiple fixed points | First explicit example (Figure 4) | Not re-examined |
| Mellowmax (log-sum-exp) is a non-expansion | **Original proof** (Section 5.1) | Cited |
| Maximum-entropy mellowmax policy | **Original** (Section 6) | Cited; criticized as expensive (per-state root-find) |
| $\mathcal{T}_{\mathrm{soft}}$ has bounded sub-optimality | Open question | **New** (Theorem 3 part I) |
| $\mathcal{T}_{\mathrm{soft}} \to \mathcal{T}$ exponentially in $\tau$ | Implicit at best | **New** (Theorem 3 part II) |
| $\mathcal{T}_{\mathrm{soft}}$ reduces overestimation vs $\max$ | Not addressed | **New** (Theorem 4) |
| Empirical: S-DQN > DQN on Atari | Out of scope | **New main experiment** |
| Empirical: S-DQN ≥ DDQN on Atari | Out of scope | **New** |
| Softmax–mellowmax comparison in DQN harness | Not addressed (mellowmax used on-policy, SARSA) | **New** (Table 2) |

The Song et al. contribution is genuinely complementary, not a refutation. Asadi-Littman is the right paper for the tabular contraction question; Song et al. is the right paper for the deep-RL function-approximation regime where overestimation, not contraction, is the dominant failure mode.

#### Significance for the project's temperature-conditioning question

Song et al. sharpens the project's framing in two ways:

1. **A modulator-conditioned temperature is doing two distinct things at once:** (a) controlling how "soft" the Bellman backup is (operator-level), and (b) controlling the overestimation bias (function-approximation-level). These are coupled through $\tau$: increasing $\tau$ makes the operator closer to true $\mathcal{T}$ *and* increases overestimation. The "right" modulator output is therefore neither $\tau = 0$ nor $\tau = \infty$.
2. **In function-approximation deep RL, the contraction property may be less load-bearing than one would think from reading Asadi-Littman alone.** Softmax works in practice on Atari despite being non-contractive, because the bound on its sub-optimality decays exponentially in $\tau$. The user's project should *not* discard fixed-$\beta$ Boltzmann a priori on contraction grounds; that argument only conclusively applies in the tabular regime.

### Appendix 2: Section-by-Section Backbone — Song et al. 2018

**Abstract.** Despite the softmax Bellman operator's non-contraction, combining it with Deep Q-learning yields Q-functions with superior policies — even outperforming Double Q-learning. The paper proves: (i) the softmax Bellman operator converges to the standard Bellman operator at an exponential rate in $\tau$; (ii) the distance of its $Q$-function from optimal can be bounded; (iii) it reduces overestimation error. A comparison among operators (max, mellowmax, softmax) shows the trade-offs.

**Section 1 — Introduction.** Motivation: softmax has long been used as a differentiable approximation to max (Sutton & Barto 1998, Reverdy & Leonard 2016), but in Q-iteration its non-contraction (Littman 1996, p. 205) was thought to make it a bad choice for the *Bellman update itself*. The paper's claim: in deep Q-learning, this expectation is "surprisingly incorrect." Replacing max with softmax in DQN's target network yields better Atari scores, reduced Q-value overestimation, and reduced gradient noise — independent of exploration. The paper also re-examines mellowmax (Asadi & Littman 2017) and shows softmax is competitive.

**Section 2 — Background and Notation.** Standard MDP $\langle \mathcal{S}, \mathcal{A}, P, R, \gamma \rangle$, optimal $Q^*$, DQN training objective Eq. (1), DDQN target. Notation: $f_\tau(x) = [\exp(\tau x_1), \ldots, \exp(\tau x_m)]^T / \sum_i \exp(\tau x_i)$ for the softmax PMF, $g_x(\tau) = f_\tau^T(x) x$ for the softmax-weighted average, $m = |\mathcal{A}|$.

**Section 3 — The Softmax Bellman Operator.** Defines $\mathcal{T}_{\mathrm{soft}}$ (Eq. 3) and $\mathrm{mm}_\omega$ (Eq. 4). Notes $\mathcal{T}_{\mathrm{soft}} \to \mathcal{T}$ as $\tau \to \infty$, $\mathrm{mm}_\omega$ does not directly give a policy and needs additional steps (Asadi & Littman 2017). Their use is *off-policy* with $\varepsilon$-greedy behavior (matching DQN), not on-policy SARSA as in Asadi & Littman.

- **Definition 1 + Lemma 2:** define $\hat{\delta}(s)$ as the largest Q-value spread across actions and prove the softmax-vs-max gap is bounded by $(m-1) \max\!\left\{\frac{1}{\tau+2}, \frac{2 Q_{\max}}{1 + e^\tau}\right\}$ above and $\frac{\hat{\delta}(s)}{m e^{\tau \hat{\delta}(s)}}$ below.

- **3.1 Theorem 3 (performance bound):** $\limsup_k \mathcal{T}_{\mathrm{soft}}^k Q_0 \le Q^*$; $\liminf_k \mathcal{T}_{\mathrm{soft}}^k Q_0 \ge Q^* - \frac{\gamma(m-1)}{1-\gamma}\max\{1/(\tau+2), 2 Q_{\max}/(1 + e^\tau)\}$. Part (II): the gap $|\mathcal{T}^k Q_0 - \mathcal{T}_{\mathrm{soft}}^k Q_0|$ decays exponentially in $\tau$.

**Section 4 — Main Experiments.** Six Atari games (Q*Bert, Ms. Pacman, Crazy Climber, Breakout, Asterix, Seaquest); 200 training epochs; 5 random seeds. Inverse-temperature grid $\tau \in \{1, 5, 10\}$; logarithmic-cooling annealing tested, no improvement vs. constant $\tau$. Figure 1 (3-row plot): S-DQN > DQN (top), S-DDQN > DDQN (middle), S-DQN > DDQN (bottom). Asterix is the dramatic case where vanilla DQN's overestimation explodes. Table 1 reports score-vs-$\tau$ trade-off: too-small $\tau$ moves further from Bellman optimality (Asterix at $\tau = 1$ underperforms); too-large $\tau$ approaches max and reintroduces overestimation. Table 2 compares max, softmax, mellowmax in the same DQN harness — softmax matches or beats mellowmax on 5/6 games (Asterix is the exception where mellowmax wins).

**Section 5 — Why Softmax Helps?**

- *5.1 Overestimation bias reduction* — Theorem 4: under van Hasselt's assumptions, softmax overestimation $\le$ max overestimation for all $\tau \ge 0$; reduction is bounded in $[\hat{\delta}(s)/(m e^{\tau \hat{\delta}(s)}), \;(m-1)\max\{1/(\tau+2), 2 Q_{\max}/(1+e^\tau)\}]$; overestimation increases monotonically in $\tau$. Figure 2 simulates standard normal noise, plots overestimation vs number of actions for $\tau \in \{0.1, 1, 10\}$, and confirms softmax (both single and double) reduces overestimation vs max (single and double). Figure 3 verifies on Atari: S-DQN, S-DDQN have smaller Q-value estimates than DQN. Figure 4 shows monotonicity of overestimation in $\tau$ within S-DDQN.

- *5.2 Gradient noise reduction* — Eq. (5) shows the DQN gradient is TD-error-times-$\nabla Q$; overestimation amplifies the TD error, hence the gradient. Reducing overestimation $\Rightarrow$ reducing gradient norm. Figure 5: S-DQN and S-DDQN have lower $\ell_2$ gradient norm and lower variance on Atari, dramatically so on Asterix. Figure 6: gradient norm grows with $\tau$ within S-DDQN, consistent with Theorem 4.

**Section 6 — A Comparison for Bellman Operators.** Table 3 (re-printed in Phase 2 above). Notes softmax and mellowmax are *Bellman-update-equivalent* in the sense that for any $Q$ there is a state-specific $\omega(\tau)$ making $\mathrm{mm}_\omega = \mathrm{sm}_\tau$ — but the mapping is state-dependent and $Q$-dependent, so the operators are still distinct. Figure 7: trade-off between approximation-to-max (softmax wins) and overestimation reduction (mellowmax wins).

**Section 7 — Related Work.** Mellowmax (Asadi & Littman 2017); log-sum-exp in linearly-solvable MDPs (Todorov 2007), G-learning (Fox et al. 2016), entropy-regularized RL (Haarnoja et al. 2017; Schulman et al. 2017; Neu et al. 2017; Nachum et al. 2017); sparse softmax via Tsallis entropy (Lee et al. 2018); DDQN (van Hasselt et al. 2016a), Averaged-DQN (Anschel et al. 2017), adaptive normalization (van Hasselt et al. 2016b), categorical DQN (Bellemare et al. 2017), other DQN variants (Wang et al. 2016, Schaul et al. 2016, Hessel et al. 2018, Dabney et al. 2018).

**Section 8 — Conclusion and Future Work.** Headline: softmax Bellman operator is a viable alternative to Double Q-learning for overestimation reduction, with the trade-off of non-contraction (bounded but non-zero sub-optimality). Future work: theoretical performance trade-off as a function of $\tau$, and an efficient annealing schedule.

**Supplementary material.** Detailed proofs (Lemma A1 bounds $Q$-values during $\mathcal{T}_{\mathrm{soft}}$-iteration; Lemma 2 derivation in full; Theorem 3 inductive arguments; Lemma A3 monotonicity of softmax-weighted-input via Cauchy-Schwarz; full proof of Theorem 4 parts I/II/III). Additional plots Figures A1–A5 across all six games, table of $\tau$ values used.

---

## Paper 3 — Ahmed et al. 2018 — Understanding the impact of entropy on policy optimization

**Citation:** Ahmed, Z., Le Roux, N., Norouzi, M., & Schuurmans, D. (2019). Understanding the Impact of Entropy on Policy Optimization. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, PMLR 97. (arXiv version 2018.)

**PDF:** `docs/project/references/Temperature/sources/Ahmed et al. 2018 - Understanding the impact of entropy on policy optimization.pdf`

### Phase 1 — Foundational Overview (Undergraduate-Level)

**Introduction (plain-English).** The first two papers in this review are about *operator-level* entropy — temperature inside a Bellman backup. This third paper is about *gradient-level* entropy: the entropy bonus that policy-gradient algorithms (REINFORCE, A3C, PPO, etc.) add to the loss function. These two uses of "entropy" look superficially similar — both push the policy away from determinism by adding a $-\sum_a \pi(a|s) \log \pi(a|s)$ term — but they live in different mathematical worlds, and confusing them is one of the easiest ways to make the project's temperature-modulator story incoherent. This paper is the canonical clarification of what entropy does on the policy-gradient side.

The standard story for why entropy regularization helps PG is "it encourages exploration" — without an entropy bonus, the policy can collapse to deterministic too quickly, before exploring enough state-action pairs, and get stuck in a bad local optimum. This is a *sampling* story: entropy keeps the policy stochastic enough that REINFORCE gradient estimates touch enough of the environment.

Ahmed, Le Roux, Norouzi, and Schuurmans push back on this conventional wisdom with a deceptively simple experiment: **what if we give the policy gradient access to the exact gradient — no Monte Carlo sampling, no variance — and see if entropy still helps?** It does. So entropy is not just a sampling-noise-mitigator. It must be doing something else, and that something else is geometric: entropy reshapes the **optimization landscape** of the policy-gradient objective, making it smoother, connecting otherwise-isolated local optima with paths of improvement, and damping curvature fluctuations along the optimization trajectory.

**Key Findings.**

1. **Entropy helps even with the exact gradient.** On a 5×5 gridworld where the policy gradient can be computed exactly (no Monte Carlo error), about 25% of random initializations of vanilla policy gradient converge to a sub-optimal policy. Adding entropy regularization (and decaying it) drops the sub-optimal fraction to zero. The variance-reduction story fundamentally cannot explain this — there is no variance.
2. **A new visualization tool: random-direction perturbation scatter plots.** Around a parameter $\theta_0$, sample many random unit directions $d$, evaluate $\Delta O_d^\pm = O(\theta_0 \pm \alpha d) - O(\theta_0)$, and plot $\Delta O^+$ vs $\Delta O^-$. The shape of the scatter plot distinguishes local optima (both negative), local minima (both positive), saddle points (mixed signs), linear regions ($\Delta O^+ \approx -\Delta O^-$), and flat regions (both $\approx 0$). Projecting onto the diagonals recovers gradient ($\Delta O^+ - \Delta O^-$) and curvature ($\Delta O^+ + \Delta O^-$) information in direction $d$, giving a histogram-style "curvature spectrum" without ever computing eigenvalues.
3. **Entropy smooths the landscape.** Around the sub-optimal gridworld solution, the un-regularized objective is in a flat region with no improvement directions; adding the entropy term reveals many directions of positive improvement. Linear interpolation between the sub-optimal and optimal policy parameters traverses a *valley* of poor solutions under the un-regularized objective, but a monotonically increasing path under the entropy-augmented one. Entropy literally rewires the local geometry to connect basins.
4. **In continuous control (MuJoCo Hopper, Walker2d, HalfCheetah), higher-entropy Gaussian policies learn faster and find better final solutions, in some environments.** Crucially, the optimal learning rate scales with entropy: $\sigma = 1.0$ tolerates a learning rate 10× larger than $\sigma = 0.1$ in Hopper. The mechanism: high-entropy policies *dampen the fluctuations of curvature along the optimization trajectory*, so a single constant learning rate can be safely chosen. Low-entropy policies have curvature that oscillates wildly between iterations, forcing the learning rate to be the smallest one that works everywhere.
5. **The effect is environment-dependent.** HalfCheetah, despite being a sibling MuJoCo task, shows almost no entropy benefit — both curvature spectrum and final solutions are essentially independent of $\sigma$. So "entropy regularization helps" is not universal; it is a property of the landscape, and one cannot a priori predict which environments will benefit.

**Initial Takeaway.** Entropy regularization in policy gradient is *not* (just) an exploration heuristic — it is a **landscape smoother**. It connects local optima, fills in flat regions with shallow gradients, and dampens curvature fluctuations. This re-explains a known phenomenon (entropy helps) with a more mechanically useful mechanism (geometry, not sampling). The implication for the user's project: a modulator that conditions the entropy *bonus weight* on a signal is doing something fundamentally different from a modulator that conditions an operator-level temperature inside a Bellman backup. The first reshapes the gradient-descent landscape; the second reshapes the fixed point of value iteration. The two effects can compound or cancel and should be tracked separately.

### Phase 2 — Graduate-Level Deep Dive

#### Setup: what entropy regularization changes in policy gradient

The standard policy-gradient objective is the expected discounted return

$$
O_{\mathrm{ER}}(\theta) \;=\; \mathbb{E}_{\pi_\theta}\!\left[\sum_{t=1}^\infty \gamma^{t-1} r_t\right]
$$

with gradient given by the policy-gradient theorem (Sutton et al. 2000):

$$
\nabla_\theta O_{\mathrm{ER}}(\theta) = \int_s d^{\pi_\theta}(s) \int_a \nabla_\theta \pi_\theta(a|s) \, Q^{\pi_\theta}(s, a) \, da \, ds.
$$

Entropy regularization augments the per-step reward with a state-conditional entropy bonus:

$$
r^\tau_t = r_t + \tau \, H(\pi(\cdot | s_t)), \quad\text{where } H(\pi(\cdot | s_t)) = \mathbb{E}_{a \sim \pi(\cdot|s_t)}[-\log \pi(a | s_t)],
$$

giving the entropy-regularized objective $O_{\mathrm{ENT}}(\theta) = \mathbb{E}_{\pi_\theta}[\sum_t \gamma^{t-1} r^\tau_t]$ and gradient

$$
\nabla_\theta O_{\mathrm{ENT}}(\theta) = \int_s d^{\pi_\theta}(s) \int_a \pi(a|s)\!\left[Q^{\tau, \pi_\theta}(s, a) \nabla_\theta \log \pi(a|s) + \tau \nabla_\theta H(\pi(\cdot | s))\right] da \, ds. \quad\text{(Eq. 3)}
$$

Here $Q^{\tau, \pi_\theta}$ is the $Q$ function under the entropy-augmented rewards. Note that **both** $O_{\mathrm{ER}}$ and $O_{\mathrm{ENT}}$ depend on $\pi_\theta$, so the entropy term reshapes the objective everywhere — not just at the explicit bonus.

This is a crucial point of contrast with the operator-level analyses of Asadi-Littman and Song et al.: those papers ask "what happens to the Bellman fixed point if I replace $\max$ with $\mathrm{softmax}$ or $\mathrm{mellowmax}$?" Ahmed et al. ask "what happens to the policy-gradient optimization landscape if I add an entropy term to the reward?" These are different questions about different objects, and the user must keep them separate.

#### The random-direction perturbation visualization

The paper's main methodological contribution. Given parameters $\theta_0$ and a small radius $\alpha$, sample directions $d$ uniformly on the unit ball, evaluate

$$
\Delta O_d^+ = O(\theta_0 + \alpha d) - O(\theta_0), \quad \Delta O_d^- = O(\theta_0 - \alpha d) - O(\theta_0),
$$

and plot $\Delta O_d^+$ vs $\Delta O_d^-$ across many directions. Geometric interpretation:

- **Local maximum:** $\Delta O_d^+ < 0$ and $\Delta O_d^- < 0$ for all $d$ — every nearby point is worse.
- **Local minimum:** $\Delta O_d^+ > 0$ and $\Delta O_d^- > 0$ — every nearby point is better.
- **Saddle point:** mixed signs — some directions go up, some go down.
- **Linear region:** $\Delta O_d^+ \approx -\Delta O_d^-$ — locally, the function is linear with non-zero gradient.
- **Flat region:** $\Delta O_d^+ \approx \Delta O_d^- \approx 0$ — no movement either way.

**Gradient–curvature decomposition.** Assume $O$ is locally quadratic, $O(\theta) \approx a^T \theta + \frac{1}{2}\theta^T H \theta$. Then by direct expansion (paper's Eq. 4–13 in the supplement, re-derived):

$$
O(\theta_0 + \alpha d) = O(\theta_0) + \alpha a^T d + \frac{\alpha^2}{2} d^T H d + \alpha \theta_0^T H d,
$$

$$
\Delta O_d^+ = \alpha a^T d + \frac{\alpha^2}{2} d^T H d + \alpha \theta_0^T H d, \qquad
\Delta O_d^- = -\alpha a^T d + \frac{\alpha^2}{2} d^T H d - \alpha \theta_0^T H d.
$$

Projecting onto the diagonals:

$$
\boxed{\;\Delta O_d^+ - \Delta O_d^- = 2 \alpha (a + \theta_0^T H)^T d = 2 \alpha \, \nabla O(\theta_0)^T d\;} \quad\text{(gradient projection, Eq. 1)}
$$

$$
\boxed{\;\Delta O_d^+ + \Delta O_d^- = \alpha^2 d^T H d\;} \quad\text{(curvature projection, Eq. 2)}
$$

So the projection onto the $x = -y$ axis is twice the directional derivative; the projection onto the $x = y$ axis is the Hessian quadratic form. By histogramming these projections over many random $d$, the authors recover an empirical *gradient spectrum* and *curvature spectrum* around $\theta_0$ — without ever computing $\nabla O$ or eigendecomposing $H$. The max/min of the curvature spectrum approximate the max/min eigenvalues of $H$.

**Why this matters.** Computing the full Hessian of a deep policy is intractable (millions of parameters squared). Computing eigenvalues is iterative and expensive. The random-direction projection is *embarrassingly parallel* (each direction is independent), needs only forward passes, and gives a low-cost picture of curvature shape. This is what enables the empirical landscape analyses in Sections 3.1 and 3.2.

#### Result 1: gridworld with exact gradient (Section 3.1)

**Setup.** 5×5 gridworld, four actions, suboptimal corner reward and optimal opposite-corner reward, linear policy $\pi(a | s_t) \propto \exp(\theta^T s_t)$ with one-hot state encoding. Replace the policy-gradient integrals with sums (the environment dynamics are known), giving **zero-variance** gradient estimates. Run gradient ascent from 100+ random $\theta_0$.

**Result.** ~25% of seeds converge to the suboptimal policy $\pi_{\mathrm{sub}}$ instead of $\pi_{\mathrm{opt}}$. The suboptimal solution sits in a **flat region** — random-direction perturbations at $\theta_{\mathrm{sub}}$ show negligible gradient and small strictly-negative curvature (Figure 4a, black circles). Linear interpolation $\theta(\alpha) = (1 - \alpha) \theta_{\mathrm{sub}} + \alpha \theta_{\mathrm{opt}}$ between the two optima passes through a *valley* of poor objective values (Figure 4b, black).

**Two interventions break the trap.** (a) Adding entropy regularization $\tau = 0.01$ to the objective. (b) Mixing the policy with uniform: $\pi_{\mathrm{eff}}(a|s) = (1 - \mathrm{Mix}) \pi(a|s) + \mathrm{Mix}/|\mathcal{A}|$. Both rescue performance. With either intervention:

- At $\theta_{\mathrm{sub}}$, the random-perturbation scatter shows **many directions of positive improvement** (Figure 4a, orange stars + blue triangles) — no longer flat.
- The interpolation between $\theta_{\mathrm{sub}}$ and $\theta_{\mathrm{opt}}$ becomes a **monotonically increasing path of improvement** (Figure 4b, orange) — no more valley.

**Why this matters.** This experiment surgically removes the sampling-noise explanation: the gradient is exact, so any benefit of entropy must be from the objective shape, not from gradient variance. Entropy is, mechanically, a landscape smoother — it fills in flat regions, connects basins, and creates improvement paths where none existed.

#### Result 2: continuous control (Section 3.2, MuJoCo Hopper / Walker2d / HalfCheetah)

**Setup.** Gaussian policy with mean $\mu_\theta(s_t) = \theta^T s_t$ and *fixed* standard deviation $\sigma$. The Gaussian entropy is $H(\mathcal{N}(\mu, \sigma^2 I_k)) = \frac{k}{2}\log(2\pi e \sigma^2)$ which depends *only* on $\sigma$. Therefore controlling $\sigma$ directly controls entropy without confounding from the policy mean. REINFORCE estimator with large batch size to control for the variance-reduction effect of larger $\sigma$ (Zhao et al. 2011). Evaluate the *deterministic* performance by setting $\sigma = 0$ in the evaluation policy.

**Finding 1: faster learning under higher entropy.** Hopper, Walker2d: $\sigma \in \{0.5, 0.75, 1.0\}$ policies learn 2–8× faster than $\sigma = 0.1$, and find final policies with 2–8× higher mean reward. HalfCheetah: a much weaker effect; all $\sigma$ converge similarly.

**Finding 2: optimal learning rate scales with entropy.** In Hopper, the best constant learning rate for $\sigma = 1.0$ ($\eta = 0.001$) is 10× larger than for $\sigma = 0.1$ ($\eta = 0.0001$). This is *not* explained by simple scalar reparametrization of the loss (the loss magnitude also changes, but in a way that does not match the LR ratio).

**Finding 3: curvature fluctuates less under higher entropy.** Track the curvature in the direction of most improvement at each iteration (90th-percentile direction, chosen for robustness to outliers). For $\sigma \in \{0.0, 0.1\}$ in Hopper, this curvature swings rapidly in time — both magnitude and sign change between adjacent iterations (Figure 6a). For $\sigma \in \{0.5, 1.0\}$, the curvature is much more stable. **Stable curvature ⇒ a single constant learning rate works ⇒ no need to set it to the worst-case-tiny value.** HalfCheetah: curvature is stable across all $\sigma$, consistent with the absence of entropy benefit there (Figure 6b).

**Finding 4: high entropy reveals improvement paths from poor final solutions.** Take a poor final solution $\theta_{\mathrm{poor}}$ from a $\sigma = 0.1$ run and a good final solution $\theta_{\mathrm{good}}$ from a $\sigma = 1.0$ run, both starting from the same random init. Random-perturbation analysis at $\theta_{\mathrm{poor}}$: under $\sigma = 0.0$, 84% of directions have detectable negative curvature ⇒ near a local optimum. Under $\sigma = 1.0$, nearly all curvatures are within sampling noise ⇒ in a *linear region*. Linear interpolation between $\theta_{\mathrm{poor}}$ and $\theta_{\mathrm{good}}$ along the high-entropy objective traces a monotonically increasing path (Figure 7b) — so from $\theta_{\mathrm{poor}}$ a direction of improvement exists in the high-entropy objective. The poor solution is genuinely stuck in the low-entropy landscape but not in the high-entropy one.

**Finding 5: this is environment-dependent.** HalfCheetah's landscape is robust to $\sigma$ — the same poor-vs-good pairing reveals no path of improvement at any $\sigma$ (Figure S17c). So entropy's smoothing effect is a property of the *(environment, parametrization)* pair, not a universal rule. The authors flag this as a caution: any claim "entropy regularization always helps" is empirically false.

#### Operator-level vs gradient-level entropy: the critical project distinction

Asadi-Littman / Song et al. and Ahmed et al. all use the word "entropy" but operate on different objects. The distinction matters for the user's modulator project.

| Aspect | Operator-level entropy (Asadi, Song) | Gradient-level entropy (Ahmed) |
|---|---|---|
| Object it modifies | Bellman backup operator $\mathcal{T}$ | Loss function gradient |
| Where it lives | Inside the value-iteration recursion | Inside the PG update |
| Effect when raised | Softer max in the backup; smaller overestimation | Smoother loss landscape; larger usable LR; connects local optima |
| Failure mode of "wrong" temperature | Wrong fixed point (e.g., multiple FPs under Boltzmann) | Wrong basin of attraction; LR forced small |
| Convergence story | Banach contraction $\to$ unique FP | Gradient descent on a non-convex surface $\to$ local optimum (sometimes good, sometimes not) |
| Effect of $\tau \to 0$ | Operator $\to$ mean | Objective $\to$ pure entropy maximization, no reward signal |
| Effect of $\tau \to \infty$ | Operator $\to$ max | Objective $\to$ pure expected reward, full PG, may collapse to greedy |

A modulator that conditions a single temperature parameter $\beta$ that is *used in both places at once* (e.g., as the softmax temperature for action selection during behavior, and as the inverse temperature in an entropy-regularized actor-critic loss) is making two simultaneous changes — and the project must track them separately or the dynamics will be uninterpretable. Ahmed et al. is the right paper to cite when the project claim is "varying the temperature reshapes the gradient landscape"; Asadi-Littman / Song et al. are the right papers when the claim is "varying the temperature preserves / breaks the contraction of the value backup."

#### Limitations of the random-perturbation visualization

The authors are explicit (supplement Section S.1.3): random-direction perturbation is statistically reliable when the *number of improvement directions* $k_2$ is comparable to the *total number of dimensions* $k_1 + k_2$. When $k_1 \gg k_2$ (most directions are flat or descending, only a few ascend), random perturbations miss the ascent directions — and so do stochastic gradients with non-trivial noise. So the technique is best for moderate-dimensional analysis (hundreds of parameters) and can be misleading in extreme high-dimensional settings where the only good directions are needle-thin manifolds.

#### Significance for the project's temperature-conditioning question

Ahmed et al. completes the picture started by Asadi-Littman and Song et al.:

- **A modulator that controls a Bellman-backup temperature** affects whether the value iteration converges (Asadi) and how biased its fixed point is (Song et al.).
- **A modulator that controls a PG entropy-bonus temperature** affects landscape smoothness, optimal learning rate, and which basin of attraction the policy lands in (Ahmed).
- These are not the same effect. The first is a property of the algorithm's *contraction structure*; the second is a property of the *gradient-descent geometry*.
- If the project conditions a single temperature in a deep-RL pipeline (e.g., the temperature of a Boltzmann actor head whose Q-value targets come from a softmax Bellman backup), it is silently conditioning *both* effects. Disentangling them requires either an ablation (operator temperature varied while gradient-temperature held fixed and vice-versa) or a clear theoretical commitment to which mechanism is doing the work.
- Ahmed et al.'s diagnostic tool (the random-direction perturbation scatter plot) is directly usable for the project: if the modulator is hypothesized to act by smoothing the landscape, the scatter plot is the right way to check it empirically.

### Appendix 3: Section-by-Section Backbone — Ahmed et al. 2018

**Abstract.** Entropy regularization is commonly used to improve policy optimization, often justified as "encouraging exploration." Using new visualizations of the optimization landscape, the authors show: (i) even with exact gradients, policy optimization is hard due to objective geometry; (ii) higher-entropy policies make the landscape smoother, connect local optima, and enable larger learning rates. Entropy serves as a regularizer, not just an explorer.

**Section 1 — Introduction.** Policy optimization is non-concave even with linear policies (Kakade 2001). Conventional view: "high variance in REINFORCE is the main issue." Authors' claim: the dominant problem is the geometry of the objective, not the noise. Contributions: (1) experimental evidence that landscape geometry, not noise, drives PG difficulty; (2) a novel random-perturbation visualization; (3) demonstration that high-entropy policies smooth the objective and enable larger LR.

**Section 2 — Approach.**
- *2.1.1 Linear interpolations.* Evaluate $O((1-\alpha) \theta_0 + \alpha \theta_1)$ for $0 \le \alpha \le 1$. Reveals 1D slices: valleys, monotone increases, barriers. Limitations (Draxler et al. 2018): the 1D slice can be misleading; isolated optima in 1D may be connected by manifolds in higher dimensions.
- *2.1.2 Random perturbations.* Sample unit-ball directions $d$; evaluate $\Delta O^\pm_d$. Classify $\theta_0$ as local max / min / saddle / linear / flat from the scatter plot signs. Project onto diagonals for gradient (Eq. 1) and curvature (Eq. 2) information. Histograms of curvature recover an approximate eigenvalue spectrum.
- *2.2 The policy optimization problem.* Standard formulation. Entropy-augmented reward $r^\tau_t = r_t + \tau H(\pi(\cdot | s_t))$. Gradient Eq. (3). Both $O_{\mathrm{ER}}$ and $O_{\mathrm{ENT}}$ depend on $\pi_\theta$, so any policy change reflects in the objective and gradient.

**Section 3 — Results.**

- *3.1 Entropy helps even with the exact gradient.* 5×5 gridworld; exact policy gradient (no Monte Carlo); two locally-optimal policies (down to $R_{\mathrm{sub}}$, right to $R_{\mathrm{opt}}$). 25% of random inits converge to $R_{\mathrm{sub}}$. At $\theta_{\mathrm{sub}}$ the un-regularized objective is flat with negligible gradient (Figure 4a, black). Linear interpolation between $\theta_{\mathrm{sub}}$ and $\theta_{\mathrm{opt}}$ traverses a valley (Figure 4b, black). Adding entropy $\tau = 0.01$ or uniform-mixing the policy reveals positive-improvement directions at $\theta_{\mathrm{sub}}$ (Figure 4a, orange/blue) and converts the inter-optima interpolation into a monotonically increasing path (Figure 4b, orange).

- *3.2 More stochastic policies induce smoother objectives.* MuJoCo Hopper, Walker2d, HalfCheetah. Gaussian policies with $\mu_\theta(s) = \theta^T s$ and fixed $\sigma \in \{0.1, 0.25, 0.5, 0.75, 1.0\}$. REINFORCE; large batch to control for variance reduction. Evaluation: deterministic ($\sigma_{\mathrm{eval}} = 0$).

  - *3.2.2 Effect on learning dynamics.* Higher $\sigma$ ⇒ faster learning and better final policy in Hopper, Walker2d. HalfCheetah: weak effect (Figure 5).
  - *3.2.3 Why high entropy learns quickly.* The best learning rate scales 10× from $\sigma = 0.1$ to $\sigma = 1.0$ in Hopper. The curvature in the direction of most improvement *fluctuates rapidly* for low-$\sigma$ optimization but is stable for high-$\sigma$ (Figure 6a). Stable curvature ⇒ a constant LR works ⇒ a larger LR is safe. HalfCheetah's curvature is stable across all $\sigma$ (Figure 6b) — consistent with no entropy benefit there.
  - *3.2.4 Reducing local optima.* Final $\theta_{\mathrm{poor}}$ from $\sigma = 0.1$ runs: 85% of directions have negative curvature (near local optimum). Re-evaluating at $\sigma = 1.0$: curvatures collapse to near-zero (linear region). Linear interpolation from $\theta_{\mathrm{poor}}$ to a $\theta_{\mathrm{good}}$ from a $\sigma = 1.0$ run traces a monotonically increasing path in the high-entropy objective (Figure 7b). HalfCheetah: no such path found (Figure S17c) — environment-dependence again.

**Section 4 — Related Work.** Visualization techniques: Goodfellow et al. 2015, Li et al. 2018a/b, Draxler et al. 2018, Fort & Scherlis 2018, Keskar et al. 2017. Entropy in PG: Williams & Peng 1991 (origin), Chaudhari et al. 2017 (entropy-SGD induces $\beta$-smooth objectives), Neu et al. 2017 (entropy regularization $\equiv$ dual optimization). Empirical PG critiques: Rajeswaran et al. 2017, Henderson et al. 2018, Ilyas et al. 2018 (PPO surrogate ascent can decouple from true ascent — landscape connection).

**Section 5 — Discussion and Future Directions.**
- *Difficulty of PG.* Even with exact gradients, PG is hard because (a) policy reparametrizations create flat plateaus (natural gradient methods like TRPO/PPO are well-motivated for this reason); (b) the landscape is problem-dependent and surprisingly heterogeneous across MuJoCo tasks.
- *Sampling strategies.* Mainstream view of entropy = exploration cannot explain HalfCheetah's behavior. Geometric view fits.
- *Smoothing.* Proposed unifying mechanism: many things that "help" PG (entropy bonus, natural gradient, batch norm, Q-value smoothing, surrogate objectives) work by smoothing the loss surface. Connection to curriculum/mollification (Chapelle & Wu 2010; Gulcehre et al. 2017).

**Supplementary material.** S.1: random-perturbation derivation (Eq. 4–13 — the gradient/curvature projection identities re-derived). S.2: exact entropy-augmented PG derivation via the Bellman equation $g^\pi = (I - \gamma P^\pi)^{-1} r^\pi$. S.3+: additional figures across all environments, seeds, and $\sigma$ values; negative examples (e.g., S17 showing pairs where smoothing fails); the gridworld decay-$\tau$ ablation (S5); curvature spectrum plots (S10–S12).

---

## Cross-paper synthesis

### The Asadi ↔ Song direct dialogue

Asadi & Littman (2017) and Song et al. (2018/2019) are in genuine, productive dialogue — Song et al. cite Asadi-Littman 11 times and reuse its operator (mellowmax) as a baseline. The dialogue resolves as follows:

| Question | Asadi answer | Song answer | Resolution |
|---|---|---|---|
| Is Boltzmann softmax a contraction? | No | No (cited) | Settled: not a contraction |
| Does GVI under Boltzmann have multiple fixed points? | Sometimes (explicit MDP) | Yes (cited) | Settled: can happen |
| Is the multi-fixed-point problem a *practical* issue for deep RL? | Implied yes (motivates mellowmax) | **No** (Theorem 3: bounded deviation, exponential rate to true Bellman in $\tau$) | **Song's contribution** — the contraction failure is bounded, not catastrophic |
| What is the dominant practical failure mode of Bellman backups in deep RL? | Multiple fixed points / oscillation | **Overestimation bias** from max + neural-network noise | **Song's contribution** — reframes the question |
| Should we use mellowmax or Boltzmann softmax in DQN-style algorithms? | Mellowmax (mathematically cleaner) | **Boltzmann softmax** (better empirical Atari scores at moderate $\tau$, has free policy representation, compatible with double-Q) | **Song's contribution** — operator choice depends on regime |
| Should we use mellowmax or Boltzmann softmax in tabular SARSA? | Mellowmax (Asadi's experiments show fewer failures) | Not the focus | **Asadi's domain** — tabular is where contraction matters most |

The crisp message: **Asadi-Littman is the right paper for the tabular contraction question; Song et al. is the right paper for the deep-RL function-approximation regime**. They are not in conflict — they are answering different questions about different operating regimes. The user's project will live in the deep-RL regime (neural-network policy + value head with a modulator-conditioned temperature) and should weight Song's reframing accordingly: the contraction story is real but not the dominant practical concern.

### How Ahmed's PG-side analysis relates to the operator-side analyses

Ahmed et al. (2018) operates on the PG side, but its mechanism connects to the operator analyses in three ways:

1. **The smoothing claim is structurally parallel.** Asadi-Littman's mellowmax and Song's softmax are *value-side smoothers*: they replace a non-smooth $\max$ with a smooth surrogate, with a temperature knob controlling the smoothing scale. Ahmed's entropy bonus is a *policy-side smoother*: it replaces a non-smooth peaked policy with a smooth higher-entropy one, with a temperature knob controlling the smoothing scale. **In both cases, temperature is a smoothing scale; in both cases, the smoothing has provable consequences (contraction / overestimation reduction in the operator case; landscape connectivity / curvature stability in the gradient case).**
2. **The "entropy regularization $\equiv$ soft $Q$-learning" equivalence (Schulman et al. 2017a, Nachum et al. 2017, Neu et al. 2017) is the bridge.** Schulman et al. proved that entropy-regularized PG is equivalent to a soft $Q$-learning algorithm whose Bellman backup uses log-sum-exp — which is **the mellowmax operator**. So the same temperature scalar that controls policy smoothness in PG (Ahmed's $\tau$) controls operator smoothness in the equivalent value-iteration view (Asadi's $\omega$). The Asadi paper notes this connection in its Section 5 (mellowmax "can also be derived from information theoretical principles as a way of regularizing policies with a cost function defined by KL divergence").
3. **Function approximation breaks clean operator analysis but does not break gradient-landscape analysis.** Song et al. argue that in deep RL the operator-level question (contraction or not) is less load-bearing than overestimation. Ahmed et al. argue that in deep RL the gradient-level question (landscape smoothness) is *the* load-bearing question. Both papers, then, point at the same conclusion: in deep RL the *temperature/entropy parameter* is primarily controlling some non-trivial geometric property of the optimization (overestimation bias on the operator side, landscape smoothness on the gradient side), not the tabular contraction properties that motivate the original analyses.

For the user's project: a single conditioned-temperature scalar in a deep-RL pipeline is doing **at least three** things simultaneously — controlling the softness of the Bellman backup (operator-side), controlling the overestimation/gradient-noise of the value head (function-approximation-side), and controlling the smoothness of the policy-gradient landscape (gradient-side). Any neural-modulator story that conditions one temperature is implicitly committing to all three. The cleanest design either separates these knobs (e.g., distinct heads for operator temperature and policy temperature) or commits explicitly to the equivalence (e.g., soft actor-critic, where the same $\alpha$ governs both — this is the Haarnoja papers, also in the Temperature corpus).

### Project guidance from the three-paper sweep

1. **Tabular regime, on-policy SARSA-like algorithm:** prefer mellowmax (Asadi) over fixed-$\beta$ Boltzmann. The contraction property is load-bearing here. If the modulator conditions $\omega$ or the state-dependent $\beta(s)$, the result is still on solid theoretical ground.
2. **Deep-RL regime with value-based learning (DQN-style):** softmax Bellman (Song) is empirically competitive and even superior to Double DQN in many cases. A modulator-conditioned $\tau$ is doing something useful (controlling overestimation/gradient-noise tradeoff). The non-contraction is acceptable since the sub-optimality is bounded and decays exponentially in $\tau$.
3. **Deep-RL regime with policy-gradient (REINFORCE / A2C / PPO-style):** the entropy-bonus story (Ahmed) is the right mental model. A modulator-conditioned entropy weight reshapes the landscape, scales the safe learning rate, and connects basins. The HalfCheetah caveat applies: do not assume the effect transfers across environments.
4. **Soft actor-critic style (SAC, on the Temperature corpus's reading list):** the same temperature appears in *both* the operator and the gradient simultaneously, by design. The Asadi/Song/Ahmed three-way distinction collapses (the equivalence is explicit). A modulator there is conditioning a single, well-defined scalar that does both jobs.

The user's project should be explicit about which regime it is in, because the "right" answer to "is conditioning temperature principled?" depends on it.

---

## Status

- **Papers processed:** 3 of 3 (Asadi & Littman 2017; Song et al. 2018; Ahmed et al. 2018).
- **Each paper has:** Phase 1 (undergrad) synthesis, Phase 2 (graduate, full LaTeX with derivations), Appendix backbone preserving original section order.
- **Cross-paper synthesis section:** added with Asadi↔Song dialogue table and Ahmed ↔ operator-side connection.
- **TOC:** auto-updated.


