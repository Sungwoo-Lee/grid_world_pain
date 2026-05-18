# Learning-Rate Adaptation — Literature Review (Shard B: Direct LR Adaptation)

## Purpose

This shard reviews the **two simplest, most direct precedents** in the project's Learning_Rate corpus for treating the learning rate as a *learned, time-varying, possibly per-parameter* variable. Both papers take essentially the same conceptual move — "the learning rate is just another parameter; differentiate the loss with respect to it and update by gradient descent" — but apply it at very different scales and with very different mechanics. Together, they form the **minimum-viable reference set** for a project that may want a neuromodulator-style signal to act on the *update step* (a temperature-on-the-update) rather than on the policy distribution (a temperature-on-the-policy).

**Why this is even a research question (Adam already exists).** A practitioner's first instinct is "use Adam and tune the learning rate once". That works, but two things still bite: (1) Adam still has a single global learning-rate hyperparameter you must hand-pick, and the optimum often spans 2–3 orders of magnitude across problems; (2) Adam's per-parameter scaling comes from a *fixed heuristic* (the running mean of squared gradients), not from anything that adapts to whether that heuristic is actually working on the loss surface you're walking on. The papers reviewed here both refuse the fixed-heuristic compromise. Baydin et al. (2017) say: differentiate the loss with respect to the global learning rate itself and run another layer of gradient descent on it — this is the **hypergradient** idea, and it works as a *one-line drop-in* on top of SGD, SGD+Nesterov, or Adam. Li et al. (2017) say: don't just learn one scalar — learn a **per-parameter vector of learning rates** (allowed to be negative, so the vector encodes *both step size and update direction*) by **meta-learning** across a distribution of tasks, then deploy that vector at test time for fast few-shot adaptation. The first is online and surgical; the second is offline-trained and architectural. They are bookends.

**What this shard does and does not cover.** It covers only the two direct-adaptation papers. The remaining four entries in the Learning_Rate corpus — Andrychowicz et al. (L2O foundations), Xu et al. (RL-specific meta-gradient), Flennerhag et al. (warped gradient descent), Harrison et al. (learned-optimizer stability) — belong to a separate "learning-to-optimize" branch and are reviewed elsewhere. The Hypernetwork / Highway / Gated-conv entries in the same source folder are architectural and reviewed under their own topics.

## Table of Contents

1. [Baydin et al. 2017 — Online learning rate adaptation with hypergradient descent](#1-baydin-et-al-2017--online-learning-rate-adaptation-with-hypergradient-descent)
   - [Phase 1 — Foundational overview](#phase-1--foundational-overview-baydin)
   - [Phase 2 — Graduate-level deep dive](#phase-2--graduate-level-deep-dive-baydin)
   - [Appendix: Section-by-section backbone](#appendix-section-by-section-backbone-baydin)
2. [Li et al. 2017 — Meta-SGD: Learning to learn quickly for few-shot learning](#2-li-et-al-2017--meta-sgd-learning-to-learn-quickly-for-few-shot-learning)
   - [Phase 1 — Foundational overview](#phase-1--foundational-overview-meta-sgd)
   - [Phase 2 — Graduate-level deep dive](#phase-2--graduate-level-deep-dive-meta-sgd)
   - [Appendix: Section-by-section backbone](#appendix-section-by-section-backbone-meta-sgd)
3. [Cross-paper connections and project framing](#cross-paper-connections-and-project-framing)

---

## 1. Baydin et al. 2017 — Online learning rate adaptation with hypergradient descent

**Citation:** Baydin, A. G., Cornish, R., Martínez Rubio, D., Schmidt, M., and Wood, F. *Online learning rate adaptation with hypergradient descent.* ICLR 2018 (arXiv:1703.04782).
**PDF:** `docs/project/references/Learning_Rate/sources/Baydin et al. 2017 - Online learning rate adaptation with hypergradient descent.pdf`

### Phase 1 — Foundational overview (Baydin)

#### What's wrong with a fixed learning rate (or even Adam)?

The learning rate $\alpha$ controls how big a step you take down the loss surface at each iteration. Pick it too high and you overshoot and diverge; pick it too low and training crawls. Bengio (2012) called it *"often the single most important hyperparameter"*. Modern adaptive optimizers (AdaGrad, RMSProp, Adam) make $\alpha$ *per-parameter* using running statistics of past gradients, which helps a lot — but they all still have a **single global learning-rate hyperparameter that you have to hand-tune by grid or random search**. That tuning is expensive (every guess requires a full training run) and brittle (the right value often shifts when you change architecture, dataset size, or batch size).

#### What hypergradient descent proposes

The fix is a one-sentence idea: **the learning rate is just another parameter, so differentiate the loss with respect to it and run gradient descent on it too**. Concretely, at iteration $t$, you ask "if I had used a slightly different learning rate at the *previous* step, would the loss now be lower?" The answer is a single dot product of two gradients that you have already computed — the current gradient and the previous gradient — and a single scalar update. The full procedure is:

1. Compute the current gradient $g_t$ (you already do this).
2. Compute the *hypergradient* — one dot product $g_t \cdot g_{t-1}$ (you already have $g_{t-1}$ in memory from last step).
3. Update $\alpha$: $\alpha_t \leftarrow \alpha_{t-1} + \beta \, (g_t \cdot g_{t-1})$, where $\beta$ is a new "meta" learning rate.
4. Do the parameter update with $\alpha_t$.

That's it — **one extra line of code** wrapping the optimizer. The same recipe applies to SGD, SGD+Nesterov momentum, and Adam with minor changes to the "what is $\nabla_\alpha u$" line. The paper rediscovers a 1998 idea by Almeida et al. that had been overlooked because it predated modern adaptive optimizers.

#### Key empirical result

On MNIST logistic regression, MNIST two-layer MLP, and CIFAR-10 VGGNet, the hypergradient variants (SGD-HD, SGDN-HD, Adam-HD) **consistently outperform their fixed-LR counterparts** across a wide range of initial $\alpha_0$. The most striking observation: even when starting from a badly-chosen $\alpha_0$, HD pulls the loss trajectory close to what a tuned $\alpha_0$ would have given. The learned $\alpha_t$ shows a characteristic pattern — a sharp initial rise (e.g., from $10^{-3}$ to $\sim 0.05$ for SGD-HD), then a decay toward a small steady value comparable to the best fixed $\alpha$ for that problem. In the grid-search heatmap (Figure 4 of the paper), as long as $\beta$ is small enough, **HD never makes things worse** — and in the limit $\beta \to 0$ it exactly recovers the original optimizer. Adam-HD beats Adam, which is remarkable because Adam is already an adaptive method.

#### Initial takeaway

A trivially small code change converts any first-order optimizer into one that auto-tunes its global learning rate online, with provable convergence (under convexity + a "transition to the underlying algorithm" extension), no extra memory cost beyond one gradient copy, and a strictly non-degrading worst case. It does *not* give per-parameter adaptation in the AdaGrad sense — the LR is still a single scalar — but it makes the choice of *that scalar's initial value* much less consequential.

### Phase 2 — Graduate-level deep dive (Baydin)

#### Setup and core derivation

Let $f : \mathbb{R}^n \to \mathbb{R}$ be the objective (typically a stochastic mini-batch loss). Standard SGD maintains parameters $\theta_t \in \mathbb{R}^n$ and learning rate $\alpha$, with update

$$
\theta_t = \theta_{t-1} - \alpha \, \nabla f(\theta_{t-1}). \tag{1}
$$

The authors ask: *what would the gradient of the loss be with respect to* $\alpha$ *itself*? Because $\alpha$ was used at the previous step to produce $\theta_{t-1}$, the chain rule gives

$$
\frac{\partial f(\theta_{t-1})}{\partial \alpha} = \nabla f(\theta_{t-1})^\top \frac{\partial \theta_{t-1}}{\partial \alpha}.
$$

The key observation: $\theta_{t-1} = \theta_{t-2} - \alpha \, \nabla f(\theta_{t-2})$, so

$$
\frac{\partial \theta_{t-1}}{\partial \alpha} = -\nabla f(\theta_{t-2}),
$$

and therefore the **hypergradient** has the strikingly simple form

$$
\boxed{\;h_t \;:=\; \frac{\partial f(\theta_{t-1})}{\partial \alpha} \;=\; -\,\nabla f(\theta_{t-1})^\top \nabla f(\theta_{t-2}) \;} \tag{2}
$$

— minus the dot product of the current and previous gradient. (The minus sign means: if the two consecutive gradients *agree in direction* (positive dot product), the hypergradient is negative, so $\alpha$ should *increase* — you've been pushing the right way and should push harder. If they *disagree* (negative dot product, i.e., you overshot and reversed direction), $\alpha$ should *decrease*. This recovers the intuition of older heuristic methods like Delta-Bar-Delta and RPROP, but as a strict gradient instead of a sign rule.)

The LR update is then

$$
\alpha_t \;=\; \alpha_{t-1} - \beta \, h_t \;=\; \alpha_{t-1} + \beta \, \nabla f(\theta_{t-1})^\top \nabla f(\theta_{t-2}), \tag{3}
$$

with $\beta$ the **hypergradient learning rate**. The parameter update of Eq. (1) is then run with $\alpha_t$ instead of a fixed $\alpha$:

$$
\theta_t = \theta_{t-1} - \alpha_t \, \nabla f(\theta_{t-1}). \tag{4}
$$

**Memory cost.** The only quantity needed beyond stock SGD is $\nabla f(\theta_{t-2})$ — one extra copy of a parameter-shaped vector. **Compute cost.** A single dot product in dimension $n$, i.e., one inner product per step. Both are negligible.

#### General-case derivation

For an arbitrary first-order update rule $\theta_t = u(\Theta_{t-1}, \alpha)$ (where $\Theta_{t-1}$ packages any history the rule uses), the same chain rule yields, in expectation over the stochastic-gradient noise,

$$
\frac{\partial \mathbb{E}\!\left[f \circ u(\Theta_{t-1}, \alpha_t)\right]}{\partial \alpha_t}
= \mathbb{E}\!\left[\tilde\nabla_\theta f(\theta_t)^\top \nabla_\alpha u(\Theta_{t-1}, \alpha_t)\right], \tag{5}
$$

where $\tilde\nabla$ denotes the stochastic estimator. Since $\theta_t$ is not yet available at the moment we want to set $\alpha_t$, we use the previous step's analogue:

$$
\boxed{\;\alpha_t \;=\; \alpha_{t-1} - \beta \, \tilde\nabla_\theta f(\theta_{t-1})^\top \nabla_\alpha u(\Theta_{t-2}, \alpha_{t-1})\;} \quad\text{(additive rule)} \tag{6}
$$

This is the **additive rule** of HD. The paper also derives a **multiplicative rule**, obtained by choosing

$$
\beta = \beta' \frac{|\alpha_{t-1}|}{\|\tilde\nabla f(\theta_{t-1})\| \, \|\nabla_\alpha u(\Theta_{t-2}, \alpha_{t-1})\|}, \tag{7}
$$

which after substitution gives

$$
\alpha_t = \alpha_{t-1}\left(1 \;-\; \beta' \frac{\tilde\nabla f(\theta_{t-1})^\top \nabla_\alpha u(\Theta_{t-2}, \alpha_{t-1})}{\|\tilde\nabla f(\theta_{t-1})\| \, \|\nabla_\alpha u(\Theta_{t-2}, \alpha_{t-1})\|}\right). \tag{8}
$$

The multiplicative rule is **invariant to rescaling** of either $\alpha$ or the gradient, and the authors note that multiplicative adaptation is generally faster than additive.

#### Variants for SGD-momentum and Adam

For each base optimizer, the only thing that changes is the quantity $\nabla_\alpha u_t$ — the partial of the *update step* with respect to $\alpha$. The paper's Algorithms 4–6 give:

- **SGD-HD** (Alg. 4): $u_t = -\alpha_t \, g_t$, so $\nabla_\alpha u_t = -g_t$. Hypergradient $h_t = g_t \cdot \nabla_\alpha u_{t-1} = -\,g_t \cdot g_{t-1}$. (Up to sign convention this matches Eq. 2.)

- **SGDN-HD** (Alg. 5, SGD + Nesterov momentum $\mu$): velocity $v_t = \mu v_{t-1} + g_t$; update $u_t = -\alpha_t (g_t + \mu v_t)$; so $\nabla_\alpha u_t = -(g_t + \mu v_t)$. Hypergradient $h_t = g_t \cdot \nabla_\alpha u_{t-1} = -g_t \cdot (g_{t-1} + \mu v_{t-1})$.

- **Adam-HD** (Alg. 6, Adam with first/second moments $m, v$ and bias-corrected $\hat m, \hat v$): update $u_t = -\alpha_t \hat m_t / (\sqrt{\hat v_t} + \epsilon)$; so $\nabla_\alpha u_t = -\hat m_t / (\sqrt{\hat v_t} + \epsilon)$. Hypergradient $h_t = g_t \cdot \nabla_\alpha u_{t-1}$.

In every case, the implementation cost over the base optimizer is: (a) store $\nabla_\alpha u_{t-1}$ between iterations (one extra parameter-shaped vector), (b) one dot product, (c) one scalar update of $\alpha$. The paper distributes this as a drop-in `torch.optim`-compatible API.

#### Empirical patterns and characteristic LR trajectory

Across all three test problems (MNIST logistic regression, MNIST 2-layer MLP, CIFAR-10 VGGNet), the learned $\alpha_t$ traces a stereotyped shape:

1. **Initial rapid rise** from $\alpha_0$ to a problem-specific peak (e.g., $\sim 0.05$ for SGD-HD/SGDN-HD; only $\sim 0.001083$ for Adam-HD since Adam's update is already scaled).
2. **Decay** to a steady-state value close to the best fixed $\alpha$ for that problem.
3. **Fluctuation** around that value.

The authors conjecture that **most of the benefit comes from getting the early-training LR right** — the trajectory implicitly does a fast warm-up the practitioner would otherwise have to hand-engineer. This motivates the convergence extension below.

#### Convergence with a "transition" extension

To get a convergence proof, the authors introduce a damping schedule. Define

$$
\gamma_t = \delta(t) \, \alpha_t + (1 - \delta(t)) \, \alpha_\infty, \qquad \delta(1) = 1, \; \delta(t) \to 0 \text{ as } t \to \infty,
$$

with $\alpha_\infty$ a chosen target steady-state LR and $\delta(t) = 1/t^2$ a typical choice. Use $\gamma_t$ (not $\alpha_t$) in the parameter update of Eq. (1). Early in training, $\gamma_t \approx \alpha_t$ (hypergradient-driven); late in training, $\gamma_t \to \alpha_\infty$ (fixed). The authors prove:

> **Theorem 5.1.** Suppose $f$ is convex and $L$-Lipschitz smooth with $\|\nabla f(\theta)\| < M$ for all $\theta$. If $\alpha_\infty < 1/L$ and $t \delta(t) \to 0$, then $\theta_t \to \theta^*$.

*Proof sketch (paraphrased).* From Eq. (3), $|\alpha_t|$ grows at most linearly in $t$: $|\alpha_t| \le |\alpha_0| + t \beta M^2$. So $\delta(t) \alpha_t \to 0$ under the hypothesis $t\delta(t) \to 0$, giving $\gamma_t \to \alpha_\infty$. For large enough $t$, $1/(L+1) < \gamma_t < 1/L$, which is a valid (possibly non-constant) LR for gradient descent on a convex $L$-smooth function — convergence then follows from the standard PL-condition / convex-smooth result (Karimi et al. 2016).

This is a *non-stochastic* result. The non-transitioning HD has no convergence proof in this paper; the authors flag both that and rate analysis as open work.

#### Sensitivity to $\beta$ and $\alpha_0$

The grid-search experiments (Figure 4) show two practically useful facts:

- **Insensitivity to $\alpha_0$ given a good $\beta$.** Once you pick a reasonable $\beta$, HD tolerates initial learning rates spanning orders of magnitude.
- **Strict non-degradation in $\beta$.** As $\beta \to 0$, HD recovers the underlying optimizer exactly. So setting $\beta$ small acts as a "tuning insurance": no harm done, possible improvement.

The recommended ranges are $\beta \sim 10^{-3}$ for SGD-HD and SGDN-HD, and $\beta \sim 10^{-7}$ to $10^{-8}$ for Adam-HD — Adam's update is already pre-scaled, so the appropriate $\beta$ is smaller.

#### Higher-order hypergradients and limitations

Section 5.2 raises the natural recursion: $\beta$ itself is now a hyperparameter, so one could apply HD again to it, and so on. The authors leave this as future work. The paper's primary limitations are: (1) convergence proof only for the transitioning variant, only convex case; (2) global scalar $\alpha$, no per-parameter adaptation in this paper (though they note the generalization is straightforward); (3) training-loss-based hypergradient — switching to validation-loss-based hypergradients (as Maclaurin et al. 2015 do) is possible but not explored.

### Appendix: Section-by-section backbone (Baydin)

**§1 Introduction.** LR is the most important hyperparameter (Bengio 2012); modern adaptive optimizers (AdaGrad, RMSProp, Adam) still have a global LR that must be tuned. Hypergradient idea: the derivative of an optimizer's update with respect to its own LR can be used to adapt that LR online. Rediscovered from Almeida et al. (1998); generalized here to modern optimizers (SGD, SGDN, Adam).

**§2 Hypergradient descent.** Apply gradient descent to the LR. Differs from Maclaurin et al. (2015) by being online per-iteration rather than propagated through a full inner training run. Derives the basic SGD-HD update: $\partial f(\theta_{t-1})/\partial \alpha = -\nabla f(\theta_{t-1}) \cdot \nabla f(\theta_{t-2})$, giving $\alpha_t = \alpha_{t-1} + \beta \nabla f(\theta_{t-1}) \cdot \nabla f(\theta_{t-2})$.

**§2.1 General derivation.** For an arbitrary update $\theta_t = u(\Theta_{t-1}, \alpha)$, the hypergradient is $\nabla_\theta f(\theta_t)^\top \nabla_\alpha u$. Since $\theta_t$ isn't yet computed when we set $\alpha_t$, use the previous step's quantities (additive rule, Eq. 6). Multiplicative rule (Eqs. 7–8) is rescaling-invariant and faster. Algorithms 1–6 lay out SGD, SGDN, Adam and their HD variants — the only change is computing and storing $\nabla_\alpha u_t$ and the LR update.

**§3 Related work.**
- *§3.1 LR adaptation:* Almeida et al. (1998) first; Plagianakos et al. (2001, 1998); Shao & Yip (2000); Stochastic Meta-Descent (Schraudolph 1999, 2006; Sutton 1992) — uses second-order Hessian-vector info; RPROP (sign-based), Delta-Bar-Delta (Jacobs 1988, sign-comparison heuristic); AdaGrad, RMSProp, vSGD, Adam (heuristic geometry estimates).
- *§3.2 Hyperparameter optimization via derivatives:* Bengio (2000), Domke (2012), Maclaurin et al. (2015 — reversible learning through full training run); HD's distinguishing move is online per-step rather than end-of-run.

**§4 Experiments.** Three problems: MNIST logistic regression, MNIST 2-layer MLP (1000 units, ReLU), CIFAR-10 VGGNet (D-config: conv-64×2 → maxpool → conv-128×2 → maxpool → conv-256×3 → maxpool → conv-512×3 → maxpool → conv-512×3 → maxpool → fc-512 → fc-10). PyTorch, Titan Xp, minibatch 128.

- *§4.1 Online tuning.* $\alpha_0 \in \{10^{-1}, ..., 10^{-6}\}$, $\beta = 10^{-4}$. For every $\alpha_0$, HD pulls loss trajectory toward optimum. Grid search (Figure 4): in non-pathological $(\alpha_0, \beta)$ regions, HD is uniformly $\ge$ baseline; $\beta \to 0$ recovers baseline exactly.
- *§4.1.1 / §4.2 / §4.3 (logistic / MLP / VGGNet):* All HD variants outperform their non-HD ancestors. Adam-HD beats Adam — most striking result, since Adam is already adaptive. Characteristic $\alpha_t$ trajectory: rapid rise from $\alpha_0$ to a problem-specific peak ($\sim 0.05$ for SGD-HD/SGDN-HD; $\sim 0.001083$ for Adam-HD), then decay.

**§5 Convergence and extensions.**
- *§5.1 Transition to underlying algorithm.* $\gamma_t = \delta(t)\alpha_t + (1-\delta(t))\alpha_\infty$; HD-like early, fixed-LR late. Theorem 5.1: $f$ convex + $L$-smooth + bounded-gradient + $\alpha_\infty < 1/L$ + $t\delta(t) \to 0$ ⇒ $\theta_t \to \theta^*$.
- *§5.2 Higher-order hypergradients.* Recursively apply HD to $\beta$; not explored.

**§6 Conclusion.** General, memory- and compute-efficient one-line drop-in. Worst-case no-harm ($\beta \to 0$ recovers baseline). Convergence proof partial; rate analysis open.

---

## 2. Li et al. 2017 — Meta-SGD: Learning to learn quickly for few-shot learning

**Citation:** Li, Z., Zhou, F., Chen, F., and Li, H. *Meta-SGD: Learning to Learn Quickly for Few-Shot Learning.* arXiv:1707.09835 (2017). Huawei Noah's Ark Lab.
**PDF:** `docs/project/references/Learning_Rate/sources/Li et al. 2017 - Meta-SGD - Learning to learn quickly for few-shot learning.pdf`

### Phase 1 — Foundational overview (Meta-SGD)

#### What's wrong with a fixed (or even Adam) learning rate — *for few-shot learning*?

This paper attacks a different problem from Baydin et al. The motivation here is not "save the practitioner from hand-tuning $\alpha$"; it is **"adapt a fresh learner to a brand-new task from only 5–20 examples in a single gradient step"**. In that regime, three things matter that fixed-LR SGD (and even Adam) get wrong:

1. **The initialization** is random — but with only 5 examples, you cannot afford to start anywhere reasonable.
2. **The update direction** is the gradient — but the gradient computed on 5 examples is very noisy and may push toward overfitting rather than generalizing.
3. **The step size** is a hand-picked scalar — but the right step size is task-dependent *and* parameter-dependent.

The fix proposed: **treat all three** — the initial parameters, the update direction, and the per-parameter step size — **as learnable quantities, trained by meta-learning across a distribution of similar tasks**. After meta-training, the meta-learner can adapt the learner to a new task in **one gradient step**.

#### What Meta-SGD proposes (in one line)

Replace the standard SGD step

$$
\theta' = \theta - \alpha \, \nabla L_T(\theta) \quad\text{(scalar }\alpha\text{, fixed)}
$$

with

$$
\theta' = \theta - \boldsymbol{\alpha} \circ \nabla L_T(\theta),
$$

where $\boldsymbol{\alpha} \in \mathbb{R}^{|\theta|}$ is a **vector** of the same shape as $\theta$, $\circ$ is element-wise multiplication, **and both $\theta$ (the initialization) and $\boldsymbol{\alpha}$ (the per-parameter step) are meta-learned across tasks**. Crucially, **entries of $\boldsymbol{\alpha}$ may be negative**, so $\boldsymbol{\alpha} \circ \nabla L_T(\theta)$ generally points in a *different direction* from the raw gradient. The "vector $\boldsymbol{\alpha}$" therefore simultaneously encodes the per-parameter learning rate (its magnitude) and the update direction (its sign and the pattern of relative magnitudes across coordinates).

#### Comparison to the two closest precedents

- **MAML (Finn et al. 2017)** uses the same inner-loop structure $\theta' = \theta - \alpha \nabla L_T(\theta)$ but with a **scalar fixed** $\alpha$; only the initialization $\theta$ is meta-learned. Meta-SGD strictly contains MAML as the special case $\boldsymbol{\alpha} = \alpha \cdot \mathbf{1}$.
- **Meta-LSTM / "Learning to learn by gradient descent by gradient descent"** (Ravi & Larochelle 2017; Andrychowicz et al. 2016) uses an LSTM to compute the per-step update, with state shared across parameters via per-coordinate independent updates. This is much higher-capacity but much harder to train.

Meta-SGD sits between these two: it has more capacity than MAML (per-parameter, signed step) but is structurally simpler than Meta-LSTM (still one gradient step, no recurrent state to backprop through).

#### Key empirical result

- **Regression (sine curves):** Meta-SGD beats MAML on every K-shot setting (e.g., 5-shot training → 5-shot testing: $0.90$ MSE vs $1.13$ MSE; 5-shot training → 20-shot testing: $0.50$ vs $0.71$).
- **Few-shot classification (Omniglot 5-way 1-shot):** Meta-SGD 99.53% vs MAML 98.7% vs Matching Nets 98.1%.
- **Few-shot classification (MiniImagenet 5-way 1-shot):** Meta-SGD 50.47% vs MAML 48.70% vs Meta-LSTM 43.44%; on the much harder MiniImagenet 20-way 5-shot, Meta-SGD 28.92% vs MAML 19.29%.
- **Reinforcement learning (2D point navigation):** Meta-SGD beats MAML on both fixed-start and varying-start tasks (average return $-8.64$ vs $-9.12$, and $-10.15$ vs $-10.71$).

The improvements are modest in absolute accuracy on easy benchmarks (Omniglot) but substantial on harder ones (MiniImagenet 20-way 5-shot: $+9.6$ points absolute over MAML).

#### Initial takeaway

If you grant that you have a distribution of related tasks (i.e., meta-training is feasible), and your goal is single-step adaptation from a handful of examples, then giving the learner a per-parameter, sign-flexible step vector is a strict capacity upgrade over MAML at modest extra cost. The whole "$\boldsymbol{\alpha}$ vector" is just one extra parameter-shaped tensor — same as adding a single bias term per layer — and it is trained by the same outer SGD that already trains $\theta$.

### Phase 2 — Graduate-level deep dive (Meta-SGD)

#### Setup and inner-loop adaptation

Let $f_\theta : \mathcal{X} \to \mathcal{Y}$ be a differentiable learner parameterized by $\theta \in \mathbb{R}^d$. For a task $T$ with training set $\text{train}(T) = \{(x_i, y_i)\}$, define the empirical task loss

$$
L_T(\theta) = \frac{1}{|T|} \sum_{(x,y) \in T} \ell(f_\theta(x), y).
$$

The standard one-step adaptation under SGD is

$$
\theta_t = \theta_{t-1} - \alpha \, \nabla L_T(\theta_{t-1}), \tag{1}
$$

with $\alpha \in \mathbb{R}_+$ a fixed scalar. The paper observes that an "optimizer" is fundamentally three choices: initialization $\theta_0$, update direction (here, $-\nabla L_T$), and learning rate $\alpha$. Standard SGD hand-picks all three. Meta-SGD proposes the **single-step inner-loop adaptation**

$$
\boxed{\;\theta' \;=\; \theta \;-\; \boldsymbol{\alpha} \circ \nabla L_T(\theta)\;} \tag{2}
$$

with two learnable meta-parameters:

- $\theta \in \mathbb{R}^d$: the **initialization** (shared across tasks at meta-test time).
- $\boldsymbol{\alpha} \in \mathbb{R}^d$: a **per-parameter vector**, same shape as $\theta$. Entries are unconstrained — in particular, $\alpha_i < 0$ is permitted, meaning that the meta-learner can request a step that moves coordinate $i$ in the direction of the gradient (ascending the per-task loss) when this helps the eventual test-set loss.

The update direction $\boldsymbol{\alpha} \circ \nabla L_T(\theta)$ is therefore *not* parallel to $\nabla L_T(\theta)$ in general — its direction is an element-wise modulation of the gradient, and its norm carries the implicit "step length". Conditional on $\boldsymbol{\alpha}$, the inner step is still fully determined by the gradient (so the method is "SGD-like"); but $\boldsymbol{\alpha}$ itself is learned, not chosen.

#### Bilevel meta-objective

Let $p(T)$ be a distribution over tasks. Each task $T \sim p(T)$ has a training partition $\text{train}(T)$ and a test partition $\text{test}(T)$ (in the meta-learning sense — both are *labeled*; "test" is the held-out portion used to score the adapted learner $\theta'$). The meta-training objective is

$$
\boxed{\;\min_{\theta,\,\boldsymbol{\alpha}} \; \mathbb{E}_{T \sim p(T)} \big[\, L_{\text{test}(T)}\!\big(\theta - \boldsymbol{\alpha} \circ \nabla L_{\text{train}(T)}(\theta)\big) \,\big]\;} \tag{3}
$$

This is a **bilevel optimization**: the *inner* problem produces $\theta' = \theta - \boldsymbol{\alpha} \circ \nabla L_{\text{train}(T)}(\theta)$ for each sampled task; the *outer* problem minimizes the resulting test loss with respect to both meta-parameters $(\theta, \boldsymbol{\alpha})$ jointly. Crucially, Eq. (3) is **differentiable in both $\theta$ and $\boldsymbol{\alpha}$** because the inner step is a single closed-form expression — no inner loop to backprop through, unlike multi-step MAML variants.

The outer update is simply outer-SGD with meta-LR $\beta$:

$$
(\theta, \boldsymbol{\alpha}) \;\leftarrow\; (\theta, \boldsymbol{\alpha}) \;-\; \beta \, \nabla_{(\theta, \boldsymbol{\alpha})} \sum_{T_i} L_{\text{test}(T_i)}\!\big(\theta'_i\big), \tag{4}
$$

where the sum runs over a meta-batch of sampled tasks (Algorithm 1 of the paper). The meta-gradient w.r.t. $\boldsymbol{\alpha}$ has the structure

$$
\nabla_{\boldsymbol{\alpha}} L_{\text{test}(T)}(\theta') \;=\; -\, \nabla L_{\text{train}(T)}(\theta) \;\circ\; \nabla L_{\text{test}(T)}(\theta'),
$$

i.e., the **element-wise product of the training-set gradient (at $\theta$) and the test-set gradient (at $\theta'$)**. Coordinates where these two gradients *agree* will receive a negative meta-gradient — pushing $\boldsymbol{\alpha}_i$ *up*, i.e., making the adaptation step larger along that coordinate. Coordinates where they *disagree* (i.e., the inner step generalized poorly on that coordinate) will receive a positive meta-gradient — pushing $\boldsymbol{\alpha}_i$ down, possibly through zero into negative territory. This is the mechanism by which Meta-SGD discovers that some coordinates should be stepped *against* the per-task gradient to avoid overfitting under few-shot data.

(One may notice a structural similarity to Baydin's hypergradient: both methods drive the LR using a *dot/product of two gradients*. The difference is the time-scale and the granularity: Baydin uses consecutive *training* gradients on a single task and updates a *scalar* $\alpha$ *online*; Meta-SGD uses train-vs-test gradients of *each task* and updates a *vector* $\boldsymbol{\alpha}$ across a *task distribution*.)

#### Reinforcement-learning variant

For RL, a task is an MDP $T = (\mathcal{S}, \mathcal{A}, q, q_0, T_h, r, \gamma)$ with state space $\mathcal{S}$, action space $\mathcal{A}$, transition $q$, initial-state distribution $q_0$, horizon $T_h$, reward $r$, and discount $\gamma$. The learner $f_\theta$ is a stochastic policy $\pi_\theta(a \mid s)$ and the task loss is the negative expected discounted return

$$
L_T(\theta) \;=\; -\,\mathbb{E}_{s_t, a_t \sim f_\theta,\, q,\, q_0}\!\left[\sum_{t=0}^{T_h} \gamma^t r(s_t, a_t)\right]. \tag{4 in paper}
$$

The inner update of Eq. (2) is unchanged but $\nabla L_T(\theta)$ is now estimated by the vanilla policy gradient on $N_1$ trajectories sampled under $\pi_\theta$. The outer update is run with TRPO (Schulman et al. 2015) on $N_2$ trajectories sampled under the adapted policy $\pi_{\theta'}$. The bilevel objective is

$$
\min_{\theta,\,\boldsymbol{\alpha}} \; \mathbb{E}_{T \sim p(T)} \big[\, L_T(\theta - \boldsymbol{\alpha} \circ \nabla L_T(\theta)) \,\big], \tag{5}
$$

(note: in RL there is no separate test set per task — the same task is sampled before and after adaptation; what is held out is the *task* itself across the $p(T)$ distribution).

#### Why the per-parameter, sign-flexible $\boldsymbol{\alpha}$ helps

The paper does not give a formal analysis but the empirical pattern across regression, classification, and RL is consistent. Three mechanisms are plausible:

1. **Per-parameter step sizes** allow the meta-learner to make large updates to layer-specific features and small updates to shared low-level features — a learned form of "layer-wise LR" widely used in practice (e.g., differential LRs in fine-tuning).
2. **Sign flexibility** allows the meta-learner to ignore or reverse spurious correlations in the few-shot training set. A particular coordinate's training-set gradient may systematically point the wrong way under few-shot data; a negative $\alpha_i$ encodes "trust this coordinate's gradient with the opposite sign".
3. **Implicit regularization.** Across the meta-batch, $\boldsymbol{\alpha}$ is shared, so its values are pinned by the *average* generalization signal across tasks. This effectively regularizes the inner step toward whatever direction helps generalize across tasks in the distribution, not just fit the current few-shot training set.

The flip side: $\boldsymbol{\alpha}$ has dimensionality $|\theta|$. For a large network, this doubles the parameter count of the meta-system, which is a meaningful cost at meta-training time (though not at meta-test time, where you just do one matmul and one element-wise multiply).

#### Where per-parameter freedom hurts

The paper does not analyze failure modes, but three are visible in practice:

- **Meta-overfitting.** With $|\theta|$ extra free parameters, $\boldsymbol{\alpha}$ can memorize the meta-training task distribution and fail to generalize to truly novel tasks.
- **Curriculum / scaling.** Meta-SGD's RL experiments are on a single 2D navigation domain. Generalization across markedly different MDP families is open; the paper flags large-scale meta-learning as future work.
- **Static $\boldsymbol{\alpha}$ at meta-test time.** Unlike Baydin's HD, $\boldsymbol{\alpha}$ is *fixed* once meta-training ends. So Meta-SGD does *not* solve the within-task online-adaptation problem; it solves the across-task one-step-adaptation problem.

#### Implementation summary (Algorithm 1, supervised case)

```
Initialize θ, α
while not done:
    Sample batch of tasks T_i ~ p(T)
    for each T_i:
        Compute L_train(T_i)(θ)
        θ'_i ← θ - α ∘ ∇L_train(T_i)(θ)
        Compute L_test(T_i)(θ'_i)
    Outer update:
        (θ, α) ← (θ, α) - β · ∇_(θ,α) Σ_i L_test(T_i)(θ'_i)
```

Algorithm 2 (RL) differs only in: inner gradient comes from vanilla PG on $N_1$ trajectories under $\pi_\theta$; outer gradient comes from TRPO using $N_2$ trajectories under $\pi_{\theta'}$.

### Appendix: Section-by-section backbone (Meta-SGD)

**§1 Introduction.** Few-shot learning challenge: standard DL learns each task from scratch with extensive SGD updates over large data. Meta-learning learns across tasks. Three meta-learner families: recurrent (LSTM), metric (Matching Nets), or optimizer-shaped (MAML, Meta-LSTM). MAML learns only init; Meta-LSTM (Ravi & Larochelle 2017, building on Andrychowicz et al. 2016) learns init + update strategy via LSTM but is hard to train. Meta-SGD: SGD-like, learns init + update direction + LR jointly, easy to train, one-step adaptation.

**§2 Related Work.**
- Generative models (Lake et al.) for few-shot.
- Metric meta-learners (Matching Nets, Siamese Nets) — for non-parametric learners.
- RNN-as-optimizer line: Hochreiter et al. (2001); Andrychowicz et al. (2016) generic LSTM-SGD; Ravi & Larochelle (2017) extend to few-shot, both init and update strategy learned but per-parameter independent (Meta-LSTM).
- MAML: SGD with learned init; α is a hyperparameter.

**§3 Meta-SGD.**

*§3.1 Meta-Learner.* Defines the standard SGD inner step (Eq. 1) and observes three optimizer ingredients: init, update direction, LR. Proposes inner step $\theta' = \theta - \boldsymbol{\alpha} \circ \nabla L_T(\theta)$ (Eq. 2). $\boldsymbol{\alpha}$ same size as $\theta$, element-wise product. Since the update direction $\boldsymbol{\alpha} \circ \nabla L_T(\theta)$ differs from the gradient $\nabla L_T(\theta)$, $\boldsymbol{\alpha}$ encodes both direction and step size. One-step adaptation only.

*§3.2 Meta-training.* Task distribution $p(T)$; each task split train(T)/test(T). Bilevel objective (Eq. 3): $\min_{\theta,\boldsymbol{\alpha}} \mathbb{E}_T[L_{test(T)}(\theta - \boldsymbol{\alpha} \circ \nabla L_{train(T)}(\theta))]$. Differentiable in both $\theta$ and $\boldsymbol{\alpha}$. Outer SGD with rate $\beta$. Algorithm 1 (supervised). RL variant: task = MDP, $L_T = -\mathbb{E}[\sum \gamma^t r_t]$ (Eq. 4), inner = vanilla PG, outer = TRPO, no separate test set (same task before/after adaptation). Bilevel RL objective (Eq. 5); Algorithm 2.

*§3.3 Related Meta-Learners.* MAML: only init learned; Meta-SGD strictly contains it as $\boldsymbol{\alpha} = \alpha \mathbf{1}$. Meta-LSTM: learns init+direction+LR but each parameter independent + LSTM-state cost — high complexity, hard to train.

**§4 Experimental Results.**

*§4.1 Regression.* Sine $y(x) = A\sin(\omega x + b)$ with $A \in [0.1,5.0]$, $\omega \in [0.8,1.2]$, $b \in [0, \pi]$. MLP 1-40-40-1, ReLU. K $\in \{5, 10, 20\}$. $\boldsymbol{\alpha}$ init uniform in $[0.005, 0.1]$ (MAML uses fixed $\alpha = 0.01$). 60k iterations, meta-batch 4. Table 1: Meta-SGD beats MAML on every (train-K, test-K) cell. After 5-shot training, on 20-shot testing: Meta-SGD MSE 0.50 vs MAML 0.71. MAML's reliance on a hand-picked $\alpha$ is brittle (changing $\alpha = 0.01 \to 0.1$ degrades MAML substantially).

*§4.2 Classification.* 4-module conv (3×3 conv → BN → ReLU → 2×2 maxpool). Omniglot: 28×28, 64 filters + FC-32. MiniImagenet: 84×84, 32 filters. N-way K-shot, N $\in \{5, 20\}$, K $\in \{1, 5\}$. Trained 60k iterations.
- Table 2 (Omniglot): Meta-SGD 99.53% (5-way 1-shot), 99.93% (5-way 5-shot), 95.93% (20-way 1-shot), 98.97% (20-way 5-shot) — narrowly beats MAML on every cell.
- Table 3 (MiniImagenet): Meta-SGD 50.47% (5-way 1-shot), 64.03% (5-way 5-shot), 17.56% (20-way 1-shot), 28.92% (20-way 5-shot) — beats MAML, Meta-LSTM, Matching Nets on every cell. Most striking gap: 20-way 5-shot, Meta-SGD 28.92% vs MAML 19.29% (+9.6 points).
- Quirk: 5-shot test results are obtained with 1-shot meta-training (cross-shot transfer better than matched training).

*§4.3 Reinforcement Learning.* 2D point navigation, two task sets (fixed start at origin; varying start in $[-0.5,0.5]^2$). Policy: MLP 2-100-100-2 mean + 2 trainable log-variance params, Gaussian action. Reward = -distance to goal. Inner: vanilla PG, $N_1 = 20$ trajectories per task; outer: TRPO, 100 meta-iterations, meta-batch 20 tasks. Meta-test: 600 tasks. Table 4: Meta-SGD beats MAML on both sets ($-8.64$ vs $-9.12$, fixed; $-10.15$ vs $-10.71$, varying).

**§5 Conclusions.** All three optimizer ingredients (init, direction, LR) learned end-to-end. One-step adaptation gives SOTA on few-shot regression / classification / RL. Future work: large-scale meta-learning (compute cost), generalization across task domains, multi-tasking meta-learners.

---

## Cross-paper connections and project framing

### The shared move and where the two papers diverge

Both papers refuse to treat the learning rate as a fixed hyperparameter and instead make it a *learnable* object updated by some flavor of gradient signal. The core mathematical move is the same:

| | Baydin (Hypergradient Descent) | Li (Meta-SGD) |
|---|---|---|
| What is learned | Scalar $\alpha \in \mathbb{R}_+$ | Vector $\boldsymbol{\alpha} \in \mathbb{R}^d$, sign-free |
| Update signal | Dot product of two *consecutive* per-task gradients: $g_t \cdot g_{t-1}$ | Element-wise product of train and test gradient *on each task*: $\nabla L_{train} \circ \nabla L_{test}$ |
| Time-scale | **Online**, within a single training run | **Offline (meta-training) then static**, across a task distribution |
| Setting | Single-task standard training | Few-shot multi-task adaptation |
| Cost over baseline | One extra gradient copy + one dot product per step | $|\theta|$ extra parameters; one extra forward+backward through inner step |
| Generalization story | Implicit; lets initial $\alpha_0$ matter less | Explicit; train/test split forces $\boldsymbol{\alpha}$ to favor generalizing directions |
| Convergence guarantee | Yes, for transitioning variant on convex $L$-smooth $f$ | None given; bilevel meta-optimization, no formal guarantee |
| Update direction relative to gradient | Same direction (scaled by $\alpha_t > 0$) | **Generally different** (Hadamard with possibly-negative vector) |

The papers are best read as **complementary bookends**, not as alternatives. Baydin solves "I have one long training run and don't want to tune $\alpha_0$". Meta-SGD solves "I have many similar short training runs and want each one to take one step from a clever starting point in a clever direction".

### Conceptual lineage

Both methods sit in a tree that traces back to Sutton (1992) gain adaptation and Schraudolph (1999, 2006) stochastic meta-descent. Baydin makes that tradition modern and tractable for SGD-Nesterov / Adam; Meta-SGD turns it into a meta-learning framework with a held-out generalization signal. Within their tree:

- **Baydin** is downstream of Almeida et al. (1998), Sutton (1992), Schraudolph's SMD, and Maclaurin et al. (2015, reversible learning). The contributions are: extension to modern optimizers, online (rather than end-of-training) updates, partial convergence theory.
- **Meta-SGD** is downstream of MAML (Finn et al. 2017) and Meta-LSTM (Ravi & Larochelle 2017, Andrychowicz et al. 2016). The contribution is: drop the LSTM, keep the one-step-SGD structure, but lift $\alpha$ from scalar hyperparameter to learnable per-parameter vector.

### Framing for the project: temperature-on-the-update vs temperature-on-the-policy

The project context here is whether a neuromodulator-style signal might act on the *update step* rather than on the *policy distribution* (i.e., as a "temperature-on-the-update" instead of a "temperature-on-the-policy"). These two papers anchor the *minimum-viable evidence* that this works:

- **Baydin establishes that** a one-dimensional learnable scalar that gates the magnitude of the parameter update is *easy to add, stable, never harmful in the small-$\beta$ limit, and helpful even on top of Adam*. If a modulator were to influence a scalar global step size during training, the Baydin construction is the cleanest mechanism with the cleanest worst-case guarantee. The hypergradient $h_t = -g_t \cdot g_{t-1}$ is also conceptually congenial to a modulator interpretation: "if my last two updates agreed, push harder; if they disagreed, pull back" maps onto reward-prediction-error / agreement-detection structures that already exist in neuromodulatory accounts.

- **Meta-SGD establishes that** a *per-parameter*, *sign-flexible* learnable step vector is a strict capacity upgrade over the scalar case, *if you have a meta-training distribution to fit it against*. If a modulator were to influence a *high-dimensional* gating of the parameter update — e.g., a per-layer or per-channel temperature — Meta-SGD is the precedent showing this works and not just adds noise. The sign-flexibility is notable: not all coordinates should be stepped along the gradient, and the meta-learner can discover which to reverse.

The key gap the rest of the corpus fills: Baydin/Li both use the gradient of the *loss* as their meta-signal. The project's modulator is a *biological* signal — it does not have to be the loss gradient. The L2O foundations (Andrychowicz), RL-specific meta-gradient (Xu et al.), and gated/hypernetwork architecture (HyperNetworks, Highway, Gated convs) papers in the same source folder collectively cover (a) how to learn the *update rule itself* (not just its scalar coefficient) and (b) how to *gate parameter-shaped quantities* with non-loss signals.

### Practical sequencing for a project that wants to act on the update step

Read in the order: **Baydin → Meta-SGD → (Andrychowicz, Xu, Flennerhag, Harrison; HyperNetworks; Highway/Gated conv)**. The first two anchor the simplest forms — what is mathematically minimal and what the empirical wins are. The remaining six in the corpus shift the question from "what coefficient gates the update?" to "what learned function produces the update?" and "how should that function be parameterized as a learnable gate?".

