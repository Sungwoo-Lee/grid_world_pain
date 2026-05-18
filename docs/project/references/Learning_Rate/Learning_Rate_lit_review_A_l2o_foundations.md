# Learning Rate — Literature Review A: L2O Foundations

## Purpose

This review covers three foundational papers in **learning-to-optimize (L2O)**: a research program that replaces hand-designed optimizers (SGD, Adam, RMSprop) with an *optimizer that is itself a learned function* — typically a small neural network whose parameters $\phi$ are trained on a distribution of optimization problems so that, at test time, the network produces parameter updates that minimize the loss faster (or to a lower value) than any fixed update rule.

**Why a fresh reader should care.** Standard deep-learning training does:
$$\theta_{t+1} = \theta_t - \alpha\,\nabla L(\theta_t),$$
where $\alpha$ is a fixed (or schedule-driven) learning rate. L2O replaces the right-hand side with a *learned* function
$$\theta_{t+1} = \theta_t + g_\phi\!\big(\nabla L(\theta_t),\,h_t\big),$$
where $h_t$ is some recurrent state carried across optimizer steps. The optimizer $g_\phi$ is trained by **meta-learning**: over a population of "inner" optimization problems, you measure how good $g_\phi$ is at minimizing each one, and you adjust $\phi$ (the optimizer's weights) by gradient descent on that meta-loss. That is the title of paper 1 — *learning to learn by gradient descent by gradient descent*: the outer gradient descent is on the optimizer; the inner gradient descent is what the optimizer is performing.

**Why this collection exists for the project.** The user is investigating whether a *modulator signal* — an interoceptive / context / neuromodulatory variable — can condition not only the agent's forward pass but the *update rule itself*. L2O is the mathematical home of that question: once the update rule is a learned function $g_\phi$, asking "what if $g$ is also conditioned on a modulator $m_t$?" is a clean, well-posed extension:
$$\theta_{t+1} = \theta_t + g_\phi\!\big(\nabla L(\theta_t),\,h_t,\,m_t\big).$$
The three papers reviewed here establish the foundations needed to reason about that extension: (1) Andrychowicz et al. 2016 introduced the coordinate-wise LSTM optimizer and the meta-training objective; (2) Flennerhag et al. 2019 (WarpGrad) reframes meta-learning as warping the loss landscape itself, which is the geometric analogue of "conditioning the update direction on context"; (3) Harrison et al. 2022 is the analytical "closer look" paper that tells us *when* learned optimizers actually generalize, what inductive biases must be baked in to make them stable, and which naive L2O designs blow up at test time.

The rest of the Learning Rate corpus (treated in companion reviews) covers direct LR adaptation (hypergradient, Meta-SGD), RL-specific meta-gradient (TIDBD, Meta-grad RL), and gated/hypernetwork/GLU architectures.

---

## Table of Contents

1. [Andrychowicz et al. 2016 — Learning to learn by gradient descent by gradient descent](#paper-1--andrychowicz-et-al-2016--learning-to-learn-by-gradient-descent-by-gradient-descent)
2. [Flennerhag et al. 2019 — Meta-learning with Warped Gradient Descent (WarpGrad)](#paper-2--flennerhag-et-al-2019--meta-learning-with-warped-gradient-descent-warpgrad)
3. [Harrison et al. 2022 — A closer look at learned optimization: Stability, robustness, and inductive biases](#paper-3--harrison-et-al-2022--a-closer-look-at-learned-optimization)
4. [Cross-paper synthesis](#cross-paper-synthesis)

---

## Paper 1 — Andrychowicz et al. 2016 — Learning to learn by gradient descent by gradient descent

- **PDF:** `docs/project/references/Learning_Rate/sources/Andrychowicz et al. 2016 - Learning to learn by gradient descent by gradient descent.pdf`
- **Venue:** NeurIPS 2016 (30th Conference on Neural Information Processing Systems, Barcelona).
- **Authors:** Andrychowicz, Denil, Gómez Colmenarejo, Hoffman, Pfau, Schaul, Shillingford, de Freitas (DeepMind / Oxford / CIFAR).

### Phase 1 — Foundational Overview (Undergraduate-Level)

**Introduction.**
Almost everything in modern machine learning has shifted from *hand-designed* to *learned* — features, representations, even network architectures. One thing remained hand-designed: the **optimizer**. SGD, Momentum, RMSprop, Adam — they are all human-engineered update rules. Andrychowicz et al. ask the obvious question: *if features can be learned, why not the optimizer itself?* They cast optimizer design as a learning problem: train a small recurrent neural network (an LSTM) to *produce* parameter updates, and pick the LSTM's weights by gradient descent on how well its updates minimize a family of training tasks. The title is a precise summary: the outer loop is gradient descent (on the optimizer's weights $\phi$); the inner loop is gradient descent (on the optimizee's weights $\theta$) — but the inner gradient descent is now performed *by* the LSTM.

**Key Findings.**
1. **A coordinate-wise LSTM** — one small LSTM applied independently to each parameter of the model being trained, with shared weights but separate hidden states — can be trained to outperform SGD, Momentum (NAG), RMSprop, and Adam on the very tasks it was trained on, with each baseline's learning rate tuned for that same task.
2. **Generalization across tasks of similar structure works well.** An LSTM optimizer trained on a 1-hidden-layer 20-unit MLP for MNIST keeps outperforming Adam when tested on (a) 40-unit MLPs, (b) 2-hidden-layer MLPs, and (c) longer training horizons (200 steps when trained on 100). It even transfers from a CIFAR-10 subset to held-out CIFAR labels.
3. **Generalization across architecture *family* fails.** The same MNIST-trained sigmoid-MLP optimizer fails to optimize a ReLU MLP — different activation, different gradient distribution, different learning dynamics. This is the first hint of a robustness problem that Paper 3 (Harrison) studies systematically.
4. **The learned optimizer behaves like momentum** when probed (Appendix B), making bigger updates than SGD/Adam but with more noise, suggestive of a shorter-time-scale momentum than Adam.

**Initial Takeaway.**
The paper is *the* foundational L2O work. It establishes:
- the **meta-loss** $L(\phi) = \mathbb{E}_f \big[\sum_t w_t f(\theta_t)\big]$ (a sum over the inner trajectory, not just the endpoint, so BPTT is informative);
- the **coordinate-wise architecture** that makes L2O scalable to networks with tens of thousands of parameters (a single small LSTM applied per-coordinate, with shared $\phi$);
- the **truncated BPTT** training recipe with one-time-step gradient-stopping through $\nabla_\theta f$ to avoid Hessian-vector products.

All later L2O work — including the other two papers in this collection — either extends, critiques, or stabilizes this template.

### Phase 2 — Graduate-Level Deep Dive

#### The meta-learning objective and the trajectory-weighted loss

For a single optimization problem $f$, the goal is to find $\theta^* = \arg\min_\theta f(\theta)$. Once the optimizer is parametric, the optimizee's final parameters $\theta^*(f, \phi)$ depend on $\phi$, so the natural meta-loss on a *distribution* over functions $f$ is the endpoint loss
$$L(\phi) = \mathbb{E}_f\big[\,f\!\big(\theta^*(f, \phi)\big)\,\big]. \tag{2}$$

This is the "right" objective but it is bad for training: gradients along the trajectory of inner updates only flow through the final step. The authors relax (2) to a **trajectory-weighted** objective with horizon $T$:
$$L(\phi) \;=\; \mathbb{E}_f\!\left[\,\sum_{t=1}^{T} w_t\, f(\theta_t)\,\right] \quad\text{where}\quad
\theta_{t+1} = \theta_t + g_t, \quad
\begin{bmatrix}g_t\\h_{t+1}\end{bmatrix} = m\big(\nabla_t, h_t, \phi\big), \tag{3}$$
with $\nabla_t \equiv \nabla_\theta f(\theta_t)$, $w_t \in \mathbb{R}_{\ge 0}$ time-step weights, and $m$ the LSTM. Setting $w_t = 1[t = T]$ recovers (2). In all experiments the authors choose $w_t = 1$ for every $t$.

**Derivation: why the trajectory weighting matters.** Apply BPTT to (3). The gradient of $L$ w.r.t. $\phi$ is
$$\frac{\partial L}{\partial \phi} \;=\; \mathbb{E}_f\!\left[\,\sum_{t=1}^T w_t\, \frac{\partial f(\theta_t)}{\partial \theta_t}\,\frac{\partial \theta_t}{\partial \phi}\,\right]
\;=\; \mathbb{E}_f\!\left[\,\sum_{t=1}^T w_t\, \nabla_t^\top \frac{\partial \theta_t}{\partial \phi}\,\right].$$
With $w_t = 1[t=T]$ only the $t=T$ term survives, and $\partial\theta_T / \partial\phi$ accumulates through the *entire* unrolled chain — long credit paths, vanishing/exploding gradients, expensive memory. With $w_t = 1$, every intermediate $\theta_t$ provides an independent learning signal, so the optimizer is rewarded for *fast* descent, not only for the endpoint. This is the standard trick that makes BPTT-through-the-learner trainable.

#### The coordinate-wise LSTM optimizer

A fully-connected RNN over an $n$-dimensional $\theta$ would have hidden state and weight count scaling with $n$ — impossible for $n \sim 10^4$ even on the MNIST MLP. Andrychowicz solve this with a **coordinate-wise factorization**: one small two-layer LSTM with 20 hidden units per layer is applied *independently to each coordinate*, with shared weights $\phi$ but separate hidden state $h^{(i)}_t$ for coordinate $i$:
$$\begin{bmatrix} g^{(i)}_t \\ h^{(i)}_{t+1} \end{bmatrix} \;=\; m_\phi\!\big(\nabla^{(i)}_t,\, h^{(i)}_t\big), \quad i = 1, \ldots, n.$$
Three consequences:
1. **Scale.** Total optimizer-parameter count is $O(1)$ in $n$ (only $\phi$, which is the LSTM's weights).
2. **Permutation invariance.** Swap coordinates of $\theta$ → the optimizer behaves identically. This is exactly the symmetry RMSprop / Adam already exploit ("each parameter has its own running second moment, computed by the same rule").
3. **Limit: no cross-coordinate correlation.** This is a *diagonal* preconditioner. Appendix D explores extensions: **GAC** (Global Averaging Cells) that share information by averaging a subset of LSTM activations across all coordinates, and **NTM-BFGS** that uses an external low-rank memory shared across coordinates, structurally mimicking BFGS's inverse-Hessian update.

#### The detached-gradient trick: $\partial \nabla_t / \partial \phi = 0$

The computational graph (Figure 2 in the paper) has two kinds of edges:
- **solid** edges: gradient flows. The optimizer state $h_t$ depends on $\phi$; the update $g_t$ depends on $\phi$; through $g_t$, the next $\theta_{t+1}$ depends on $\phi$.
- **dashed** edges: gradient is **dropped**. Specifically, the dependence of $\nabla_t = \nabla_\theta f(\theta_t)$ on $\phi$ (via $\theta_t$) is ignored.

Formally, this is the approximation
$$\frac{\partial \nabla_t}{\partial \phi} \;\stackrel{!}{=}\; 0.$$
**Why?** The full derivative would be $\frac{\partial \nabla_t}{\partial\phi} = \nabla^2 f(\theta_t)\,\frac{\partial \theta_t}{\partial\phi}$ — a Hessian-vector product per step, which is the cost L2O is trying to avoid in the first place. Dropping it makes the meta-gradient cheap (no second derivatives), at the cost of a biased meta-gradient.

#### Truncated BPTT and unrolling

In all experiments the optimizer is trained by **truncated BPTT** on Equation (3). For MNIST the inner horizon is $T = 100$ steps but the BPTT unroll is **20 steps**. The trajectory is sliced into chunks; gradients flow only inside the current chunk; the hidden state $h$ is carried across chunks (no gradient through the boundary). This is the standard recurrent-network training trick applied to the meta-graph.

The meta-optimizer (the outer optimizer training $\phi$) is **Adam** with a learning rate selected by random search; early stopping is used to avoid the "meta-overfitting" pathology where $\phi$ memorizes idiosyncrasies of the inner-task training distribution.

#### Gradient preprocessing (Appendix A)

Different coordinates' gradients can span many orders of magnitude. Neural networks regress poorly on inputs with such heterogeneous scale. The authors map each scalar gradient $\nabla$ to a 2-D feature via a log-magnitude / sign decomposition:
$$\nabla \;\mapsto\;
\begin{cases}
\big(\,\tfrac{\log|\nabla|}{p},\; \mathrm{sgn}(\nabla)\,\big), & |\nabla| \ge e^{-p}, \\
\big(\,-1,\; e^{p}\nabla\,\big), & |\nabla| < e^{-p},
\end{cases}$$
with $p = 10$. The piecewise switch avoids the $\log|\nabla| \to -\infty$ singularity at $\nabla \to 0$. This is essentially the same idea as RMSprop's second-moment normalization, lifted into the optimizer's input features. Paper 3 (Harrison) shows that inductive biases of this kind — log/sign decomposition, magnitude normalization — are exactly what separates L2O designs that generalize from those that diverge.

#### Empirical findings, summarized

| Experiment | Setting | Result |
|---|---|---|
| 10-D quadratics $f(\theta) = \lVert W\theta - y\rVert_2^2$ | 100 inner steps, 20-step BPTT | LSTM-opt beats all baselines |
| MNIST, MLP-1-20-sigmoid | 100 inner steps | LSTM-opt > Adam ≈ NAG ≈ RMSprop |
| MNIST, *test-time* 200 steps | trained on 100 steps | LSTM-opt keeps winning |
| MNIST, *test-time* 40 units, 2 layers | trained on 20 units, 1 layer | generalizes |
| MNIST, *test-time* ReLU | trained on sigmoid | **fails to generalize** |
| CIFAR-10 (conv+FC); separate LSTM per layer-type | 1000 steps | LSTM-opt > Adam; transfers to held-out labels |
| Neural-Art style transfer, 64×64 → 128×128, new style | 128 inner steps, 32-step BPTT | LSTM-opt > all baselines, even out-of-distribution |

#### Connection to the project's modulator question

Andrychowicz's coordinate-wise LSTM is exactly the abstraction the project needs to ask: *what if the optimizer is conditioned on a modulator?* The current architecture is
$$\big(g^{(i)}_t,\, h^{(i)}_{t+1}\big) = m_\phi\!\big(\nabla^{(i)}_t,\, h^{(i)}_t\big).$$
The modulator-conditioned generalization is
$$\big(g^{(i)}_t,\, h^{(i)}_{t+1}\big) = m_\phi\!\big(\nabla^{(i)}_t,\, h^{(i)}_t,\, m_t\big),$$
where $m_t$ is an interoceptive / contextual scalar (or low-D vector) broadcast across all coordinates. Three implementation routes are natural in this framework:
1. **Input concatenation:** append $m_t$ to the LSTM input — equivalent to letting the optimizer learn $\nabla \mapsto g$ as a *family* of functions indexed by $m_t$.
2. **FiLM on LSTM activations:** scale/shift the LSTM's hidden state by $\gamma(m_t), \beta(m_t)$ — modulates the optimizer's effective dynamics without enlarging $\phi$.
3. **Hypernetwork on $\phi$:** generate $\phi$ itself from $m_t$ — equivalent to a *family* of learned optimizers indexed by the modulator (very high capacity; Paper 3 would flag stability concerns).

(Code-side implementation belongs to `senior-developer`.)

### Appendix: Section-by-Section Backbone

**Abstract.** Casts optimizer design as a learning problem; LSTM-based learned optimizers outperform hand-designed competitors on quadratics, small MLPs, conv-nets on CIFAR, and Neural-Art style transfer, and generalize to similar-structure tasks.

**§1 Introduction.** Motivates the shift "hand-designed → learned" applied to the update rule. Reviews the standard SGD update $\theta_{t+1} = \theta_t - \alpha_t \nabla f(\theta_t)$ and its enhancements (momentum, Rprop, AdaGrad, RMSprop, Adam, K-FAC). Cites the No Free Lunch theorems as motivation for problem-class-specialized optimizers. Proposes the learned update
$$\theta_{t+1} = \theta_t + g_t(\nabla f(\theta_t), \phi). \tag{1}$$

**§1.1 Transfer learning and generalization.** Reframes "transfer between optimization problems" as a generalization problem in meta-learning: example points = problem instances, function to learn = the update rule. Inductive biases of the optimizer family determine which task-distributions it generalizes to.

**§1.2 A brief history and related work.** Lineage: Schmidhuber's self-referential networks (1987, 1992, 1993); Bengio et al. (1990, 1995) "learning without gradient descent by gradient descent"; Runarsson & Jonsson (2000); Cotter & Conwell (1990), Younger et al. (1999, 2001), Hochreiter et al. (2001) — meta-learning with backpropagation, the direct ancestor.

**§2 Learning to learn with recurrent neural networks.** Defines the optimizer as a recurrent network $m_\phi$ with hidden state $h_t$. States the endpoint meta-loss (Eq. 2) and the trajectory-weighted relaxation (Eq. 3). Introduces the computational graph (Fig. 2) and the detached-gradient simplification $\partial\nabla_t / \partial\phi = 0$.

**§2.1 Coordinatewise LSTM optimizer.** Coordinate-wise factorization with shared $\phi$ and per-coordinate $h^{(i)}_t$ — the central architectural choice. Two-layer LSTM with 20 hidden units per layer. Preprocessing / postprocessing of LSTM inputs and outputs is needed to handle gradient-magnitude heterogeneity (Appendix A).

**§3 Experiments.** All trained optimizers use 20-unit two-layer LSTMs, trained with truncated BPTT, meta-optimized with Adam, learning rate selected by random search, early-stopping on a validation set of fresh tasks.

- **§3.1 Quadratic functions.** $f(\theta) = \lVert W\theta - y\rVert^2_2$ with random $W \in \mathbb{R}^{10\times 10}$, $y \in \mathbb{R}^{10}$. 100 inner steps, 20-step unroll. LSTM optimizer substantially outperforms all baselines.
- **§3.2 MNIST MLP.** Base task: cross-entropy of a 1-hidden-layer 20-unit sigmoid MLP, batch size 128. LSTM-opt beats Adam/NAG/RMSprop. Generalizes to: 40 units, 2 layers, 200 steps (despite 100-step training). **Fails** to generalize to ReLU activations.
- **§3.3 CIFAR-10 conv-net.** 3 conv layers with max-pool + 32-unit FC head, ReLU + batch-norm. Two LSTMs (one for conv layers, one for FC layers) because a single coordinate-wise LSTM is insufficient for the conv/FC heterogeneity. Beats baselines on CIFAR-10, CIFAR-5, CIFAR-2. An optimizer trained only on held-out labels transfers well to the full dataset.
- **§3.4 Neural Art.** $f(\theta) = \alpha L_\text{content}(c, \theta) + \beta L_\text{style}(s, \theta) + \gamma L_\text{reg}(\theta)$. 1 style image + 1800 ImageNet content images, 64×64. LSTM-opt beats baselines at training resolution and *also* at double resolution (128×128) with a different style — strong out-of-distribution transfer.

**§4 Conclusion.** Optimizer design as a learning problem works. Learned optimizers compare favorably to hand-engineered baselines. Strong transfer across architecture variants and image resolutions in the tasks where the family of inner problems retains similar structure.

**Appendix A — Gradient preprocessing.** Log-magnitude / sign decomposition (above).

**Appendix B — Visualizations.** Trajectories of proposed updates compared with SGD and Adam (Fig. 10). Learned optimizer behaves like short-time-scale momentum; surprisingly, sometimes steps in the *opposite* direction of Adam at the same gradient.

**Appendix C — Neural Art examples.** Qualitative results.

**Appendix D — Information sharing between coordinates.** Two architectural extensions to break the diagonal-preconditioner limit of the coordinate-wise design:
- **LSTM+GAC** (Global Averaging Cells): a subset of LSTM cells per layer have their outgoing activations averaged across all coordinates. Sufficient to implement, e.g., L2 gradient clipping.
- **NTM-BFGS optimizer**: LSTM+GAC controller + external low-rank memory $M_t$ (Neural Turing Machine-style) shared across coordinates. Memory updates are restricted to low rank, mirroring BFGS's inverse-Hessian rank-2 update. The read/write operations preserve BFGS's structure but learn their parametric form.

---

## Paper 2 — Flennerhag et al. 2019 — Meta-learning with Warped Gradient Descent (WarpGrad)

- **PDF:** `docs/project/references/Learning_Rate/sources/Flennerhag et al. 2019 - Meta-learning with Warped Gradient Descent.pdf`
- **Venue:** ICLR 2020 (arXiv first appeared late 2019).
- **Authors:** Flennerhag (Manchester / Alan Turing Institute / DeepMind), Rusu, Pascanu, Visin, Yin, Hadsell.

### Phase 1 — Foundational Overview (Undergraduate-Level)

**Introduction.**
Andrychowicz-style "memory-based" L2O is *flexible* — an LSTM can in principle express any update rule — but it is *brittle*: there is no guarantee the optimizer converges, the meta-gradient is expensive when unrolling many inner steps, and out-of-distribution transfer often fails. The opposite camp, MAML-style "gradient-based meta-learning", takes the standard gradient-descent update rule as a hard inductive bias and only meta-learns the *initial parameters* $\theta_0$ that allow fast adaptation. Gradient-based methods are robust but limited: they back-propagate through the inner adaptation steps, which is computationally infeasible beyond a handful of steps ("few-shot").

WarpGrad is a third path that combines both. It keeps the standard gradient descent update $\theta \leftarrow \theta - \alpha \nabla L(\theta)$ as the inner rule, but it **meta-learns *how the loss landscape looks*** — by inserting small neural networks called **warp layers** between the layers of the task model. The warp layers transform the geometry of the loss landscape so that ordinary gradient descent — performed on the *warped* coordinates — is well-behaved across a whole distribution of tasks. The meta-objective is "trajectory-agnostic": it does **not** back-propagate through the inner adaptation steps, so WarpGrad scales to hundreds of inner steps (where MAML and its descendants cannot).

**Key Findings.**
1. **The "warp layers" are universal:** inserting any neural network $\omega$ between layers of the task-learner $h$ is equivalent (to first order) to preconditioning the gradient by a matrix $P$ that depends on the warp's Jacobian. With linear warps, you recover a known method (T-Nets / Meta-Curvature, block-diagonal preconditioning); with non-linear warps you go strictly beyond block-diagonal preconditioning — including data-dependent preconditioning.
2. **Geometric interpretation:** WarpGrad is, to first order, equivalent to Riemannian gradient descent on the task-learner's parameter manifold $\mathcal{W}$ under a meta-learned Riemann metric $G^{-1} = (D_x \Omega)(D_x \Omega)^\top$. The metric is *learned across tasks*, so it captures the geometry of the whole task distribution.
3. **Trajectory-agnostic meta-objective:** because preconditioning is a Markov-order-1 operator (the matrix $P(\theta;\phi)$ depends only on the current $\theta$, not the trajectory), the meta-objective can be written as an expectation over $(\tau, \theta)$ sampled from any inner trajectory — no back-prop through the K-step adaptation.
4. **Scales beyond few-shot:** retains MAML's inductive bias on miniImageNet / tieredImageNet 5-way 1-shot/5-shot (Warp-MAML beats MAML by 3.6–5.5 pp), and additionally scales to 640-step adaptation (multi-shot tieredImageNet) and 100-step adaptation (multi-shot Omniglot) where MAML-class methods cannot run at all.
5. **Works in RL with recurrent learners:** a Warp-RNN (hypernetwork modulating an LSTM task-learner) beats Hebbian meta-learning and L2RL on Miconi's maze-navigation task — the first time, to the authors' knowledge, a gradient-based meta-learner outperforms memory-based meta-learners on a memory-demanding RL task.
6. **Combats catastrophic forgetting:** by choosing $L_\text{meta}^\tau$ to be the average loss over current *and previous* sub-tasks, warp parameters learn to disentangle their adaptation processes — a working continual-learning instantiation.

**Initial Takeaway.**
WarpGrad is the geometric / preconditioning view of L2O. Instead of asking *"what update should I take next?"* (Andrychowicz LSTM), WarpGrad asks *"what coordinate system should I be in so that vanilla SGD already does the right thing?"* The two are first-order equivalent on a Riemannian manifold but have very different practical properties: WarpGrad inherits convergence guarantees from gradient descent and avoids the BPTT-through-adaptation bottleneck of MAML. For the project's modulator-conditioning question, WarpGrad is the cleanest mathematical setting in which "modulator-conditioned preconditioning" becomes a first-class object: a modulator $m$ enters the warp layers $\omega(\cdot;\,\phi,\,m)$, which is structurally the same as **FiLM-on-warp-layers** — exactly the architecture the project's hypothesis-development thread keeps converging on.

### Phase 2 — Graduate-Level Deep Dive

#### Where WarpGrad sits in the landscape of optimization-based meta-learners

All optimization-based meta-learners parameterize an update rule $\theta \leftarrow U(\theta;\xi)$. Gradient-based ones share an inductive bias: $U$ is built around $\nabla L$. The paper's compact taxonomy (Eqs. 2–5):
$$
\begin{aligned}
U(\theta_k;\,\theta_0)            &= \theta_k - \alpha\,\nabla L(\theta_k) && \text{MAML (only $\theta_0$ is meta-learned)} \\
U(\theta_k;\,\theta_0,\phi)       &= \theta_k - \alpha\,\mathrm{diag}(\phi)\,\nabla L(\theta_k) && \text{Meta-SGD (per-parameter LR)} \\
U(\theta_k;\,\theta_0,\phi)       &= \theta_k - \alpha\,B(\theta_k;\phi)\,\nabla L(\theta_k) && \text{Meta-Curvature (block-diagonal $B$)} \\
U(\theta_k;\,\theta_0,\phi)       &= \theta_k - \alpha\,\nabla L(\theta_k;\phi) && \text{T-Nets (linear $T$ embedded in layers)}
\end{aligned}
$$
All four meta-learn $\xi = \{\theta_0, \phi\}$ by back-propagating through the K-step inner descent — which limits them to few-shot adaptation.

WarpGrad generalises in two orthogonal directions:
- (i) **Beyond block-diagonal:** make the inter-layer projection $\omega$ a *non-linear* neural network, not a fixed linear $T$. This couples coordinates across layers and makes preconditioning *data-dependent*.
- (ii) **Drop the BPTT-through-adaptation requirement:** exploit that preconditioning is a Markov-1 operator (depends on $\theta_t$, not the past trajectory). This makes the meta-loss a trajectory-agnostic expectation over $(\tau,\theta)$.

#### Warp layers as embedded preconditioning

Let the task-learner be a stack $f = h^{(L)} \circ \cdots \circ h^{(1)}$. Insert warp layers $\omega^{(i)}$ between each pair:
$$\hat f = \omega^{(L)} \circ h^{(L)} \circ \cdots \circ \omega^{(1)} \circ h^{(1)}.$$
$\theta = \{\theta^{(i)}\}$ are the **task parameters** (updated during inner adaptation); $\phi = \{\phi^{(i)}\}$ are the **warp parameters** (fixed during inner adaptation, meta-learned across tasks). Back-prop through $\hat f$ now gives a preconditioned gradient (Eq. 6):
$$\frac{\partial L}{\partial \theta^{(i)}} \;=\; \mathbb{E}\!\left[\nabla\ell^\top \prod_{j=0}^{L-(i+1)}\big(D_x \omega^{(L-j)}\,D_x h^{(L-j)}\big)\,D_x\omega^{(i)}\,D_\theta h^{(i)}\right],$$
where $D_x$ is the Jacobian w.r.t. input and $D_\theta$ the Jacobian w.r.t. parameters. The chain of $D_x\omega^{(j)}$ Jacobians is exactly the "preconditioning matrix" $P$. When $\omega$ is linear and equals a fixed projection $T$, you recover **T-Nets** (block-diagonal $P$). When $\omega$ is a non-linear net, $P$ becomes data-dependent and *not* block-diagonal — a strict generalisation.

#### The Riemannian-geometry interpretation (the geometric "punchline")

This is the conceptual core of the paper, sketched here in detail because it is the framework in which "modulator-conditioned preconditioning" lives.

Define the reparameterisation $\Omega$: a map from a *warped* parameter space $\mathcal{P}$ to the native task manifold $\mathcal{W}$, induced by the warp layers,
$$h^{(i)}\big(x;\,\Omega(\theta;\phi)^{(i)}\big) \;=\; \omega^{(i)}\big(h^{(i)}(x;\theta^{(i)});\,\phi\big), \quad \forall x, i.$$
Let $\gamma = \Omega(\theta;\phi)$ — the native parameters as seen by the task-learner. Gradient descent in $\mathcal{P}$:
$$\Delta\theta \;:=\; \nabla(L\circ \Omega)(\theta;\phi) \;=\; [D_x\Omega(\theta;\phi)]^\top \nabla L(\gamma), \tag{7}$$
and in $\mathcal{W}$:
$$\Delta\gamma \;:=\; D_x\Omega(\theta;\phi)\,\Delta\theta \;=\; G(\gamma;\phi)^{-1}\,\nabla L(\gamma), \quad\text{where}\quad G^{-1} := [D_x\Omega][D_x\Omega]^\top. \tag{8}$$
**Key claim (Eq. 9):** to first order in the step size,
$$(L\circ\Omega)(\theta - \alpha\,\Delta\theta) \;=\; L(\gamma - \alpha\,\Delta\gamma) \;+\; \mathcal{O}(\alpha^2).$$

**Derivation sketch.** Taylor-expand both sides at $\alpha=0$:
- LHS: $(L\circ\Omega)(\theta) - \alpha\,\nabla(L\circ\Omega)(\theta)^\top \Delta\theta + \mathcal{O}(\alpha^2)$.
- RHS: $L(\gamma) - \alpha\,\nabla L(\gamma)^\top \Delta\gamma + \mathcal{O}(\alpha^2)$.
At zeroth order both equal $L(\gamma)$ (since $\gamma = \Omega(\theta;\phi)$). At first order LHS gives $\nabla(L\circ\Omega)^\top \Delta\theta = \Delta\theta^\top \Delta\theta$ (by Eq. 7); RHS gives $\nabla L^\top \Delta\gamma = \nabla L^\top G^{-1} \nabla L$. Using Eq. 8, $\Delta\gamma = D_x\Omega \, \Delta\theta$, so $\nabla L^\top \Delta\gamma = \nabla L^\top D_x\Omega\,\Delta\theta = ([D_x\Omega]^\top\nabla L)^\top \Delta\theta = \Delta\theta^\top \Delta\theta$. The two first-order terms agree, hence the $\mathcal{O}(\alpha^2)$ residual.

**Interpretation.** A descent step in $\mathcal{P}$ (the WarpGrad space) is first-order equivalent to **Riemannian gradient descent in $\mathcal{W}$ under the meta-learned metric $G$**, i.e. natural-gradient-like descent. Provided $\Omega$ is non-degenerate ($G$ non-singular), $G^{-1}$ is positive-definite, so $G$ is a valid Riemann metric and WarpGrad inherits gradient-descent convergence guarantees.

This is why the paper says: *we are not just preconditioning gradients, we are learning a manifold structure on which adaptation is natural*. Different tasks have different native loss surfaces (Figure 3 right), but in the *warped* space $\mathcal{P}$ they all look smooth and well-behaved (Figure 3 left).

#### The canonical meta-objective and its trajectory-agnostic relaxation

The ideal meta-objective (Eq. 10) — **steepest descent in $\mathcal{W}$ under the meta-learned $G$**:
$$\min_\phi\; \mathbb{E}_{L,\gamma\sim p(L,\gamma)}\Big[ L\big(\gamma - \alpha\, G(\gamma;\phi)^{-1}\nabla L(\gamma)\big)\Big]. \tag{10}$$
Operationalised (Eq. 11) via first-order equivalence (recasting in terms of $\theta$):
$$L(\phi) \;:=\; \sum_{\tau\sim p(\tau)}\sum_{\theta^\tau \sim p(\theta\mid\tau)}\; L^\tau_\text{meta}\!\Big(\theta^\tau - \alpha\,\nabla L^\tau_\text{task}(\theta^\tau;\phi);\;\phi\Big). \tag{11}$$
Two crucial decouplings:
- **$L_\text{task}$ vs $L_\text{meta}$ can differ.** Task adaptation uses $L_\text{task}$ (e.g. training set); meta-update uses $L_\text{meta}$ (e.g. validation set; or, in continual learning, the average over current + previous sub-tasks).
- **No BPTT through trajectories.** The sum is over samples $\theta^\tau$ drawn from the inner-adaptation distribution $p(\theta\mid\tau)$, not from a frozen trajectory. The meta-gradient is *independent of the number of inner steps* $K$.

A first-order approximation (Eq. 12) drops the second-order term by detaching the inner gradient with a stop-gradient operator:
$$\hat L(\phi) \;=\; \sum_\tau\sum_{\theta^\tau} L^\tau_\text{meta}\Big(\mathrm{sg}\big[\,\theta^\tau - \alpha\,\nabla L^\tau_\text{task}(\theta^\tau;\phi)\,\big];\,\phi\Big). \tag{12}$$
Unlike first-order MAML (which throws away *the whole trajectory* except the last gradient), WarpGrad-FO keeps *all* gradient terms and only discards local second-order effects. The ablation in Appendix F confirms only a small performance loss versus the exact version.

#### Sampling $\theta^\tau$ along an adaptation chain

The paper exploits a known reading of SGD as Bayesian sampling from an empirical posterior (Grant et al. 2018): each inner iterate $\theta^\tau_k$ is a sample from $p(\theta^\tau_k \mid \theta^\tau_{k-1}, \phi)$. A $K$-step adaptation generates a chain $\theta^\tau_0, \ldots, \theta^\tau_K$. Stochastic sampling from such chains defines the empirical $p(\theta\mid\tau)$ used in Eq. 11. Algorithm 1 (online) and Algorithm 2 (offline, with a replay buffer of $\theta$-samples) implement this. The offline variant — replaying samples $2000\times$ per meta-step in the Omniglot experiment — pushes test accuracy from 76.3% to 84.3%.

#### Integration with learned initialisations (Eq. 13)

Because WarpGrad takes $p(\theta\mid\tau)$ as given, it composes with any choice of prior on $\theta_0$: a multi-task solution, a learned point estimate, or a full Bayesian prior. The combined objective:
$$J(\phi, \theta_0) \;=\; L(\phi) \;+\; \lambda\,C(\theta_0), \tag{13}$$
where $C$ is any prior-objective and $\lambda$ controls its weight. Examples in the paper:
- **Warp-MAML** = WarpGrad + MAML's $C_\text{MAML}$ (few-shot ImageNet).
- **Warp-Leap** = WarpGrad + Leap's expected-trajectory-length objective (multi-shot Omniglot, tieredImageNet).
- **Warp-RNN** (RL) = Warp parameters meta-learned online while task-RNN weights are updated by A2C.

#### Empirical headlines (Section 4)

| Setting | Comparison | Result |
|---|---|---|
| miniImageNet 5-way 1-shot | Reptile / MSGD / (M)T-Net / CAVIA / MAML | Warp-MAML 52.3 ± 0.8 (best) |
| miniImageNet 5-way 5-shot | Same | Warp-MAML 68.4 ± 0.6 (best) |
| tieredImageNet 5-way 1-shot | MAML 51.7 | Warp-MAML 57.2 (+5.2 pp) |
| tieredImageNet 10-way 640-shot | Reptile 76.5, Leap 73.9 | Warp-Leap 80.4 (+3.9 pp) |
| Omniglot 20-way 100-shot | Reptile 70.8, Leap 75.5, fine-tuning 76.4 | Warp-Leap 83.6 (+8.1 pp) |
| Maze-RL recurrent | L2RL (RNN) ~125 reward; Hebbian-RNN | Warp-RNN ~160 reward in 60k episodes |
| Continual sine regression | n/a | WarpGrad maintains $\sim 10^{-2}$ loss on past sub-tasks after switching |

**Important null result (Appendix G):** WarpGrad is *not* Natural Gradient Descent. The Kronecker-factored covariance check shows Warp-Leap's preconditioning matrices are **not** approximations to the inverse Fisher; the geometry is genuinely meta-learned, not approximated curvature.

#### Connection to the project's modulator question

WarpGrad gives the project the cleanest mathematical framing of "context-conditioned learning rate / update rule":
- the natural-gradient form $\Delta\gamma = G^{-1}(\gamma;\phi)\nabla L(\gamma)$ becomes
  $$\Delta\gamma = G^{-1}(\gamma;\phi,m)\,\nabla L(\gamma)$$
  when the warp layers $\omega$ are conditioned on a modulator $m$, e.g. via FiLM scale/shift inside $\omega$:
  $$\omega^{(i)}(x;\,\phi^{(i)},m) = \gamma(m)\odot \tilde\omega^{(i)}(x;\phi^{(i)}) + \beta(m).$$
  The induced metric $G$ becomes a *function* of the modulator — i.e. the **loss-landscape geometry itself is modulator-conditioned**.
- Because the WarpGrad meta-objective is trajectory-agnostic, modulator-conditioned WarpGrad inherits the scaling property: no BPTT-through-adaptation, valid for hundreds of inner steps.
- Unlike Andrychowicz's coordinate-wise LSTM, WarpGrad warp layers naturally *couple* parameters across layers, so a modulator can shape the *off-diagonal* structure of preconditioning. This is the geometric analogue of the project's neuromodulator hypothesis: a single global signal reshapes the effective dynamics of learning across the whole agent.

(Code-side implementation belongs to `senior-developer`. The math here is the conceptual ground for that handoff.)

### Appendix: Section-by-Section Backbone

**Abstract.** Memory-based L2O (Andrychowicz) lacks inductive bias and can fail to converge; MAML-style gradient-based meta-learning requires BPTT through adaptation and is limited to few-shot. WarpGrad intersects both: meta-learns preconditioning matrices via *warp layers* interleaved in the task-learner; warp layers are meta-learned without BPTT through task-training. Empirical scope: few-shot, multi-shot, continual, RL.

**§1 Introduction.** Argues the dichotomy memory-vs-gradient-based meta-learning. Cites the L2O lineage (Andrychowicz 2016; Ravi & Larochelle 2016; Li & Malik 2016; Chen et al. 2017). Cites the MAML lineage (Finn et al. 2017; Nichol et al. 2018; Flennerhag et al. 2019 Leap). Cites the preconditioning lineage (Meta-SGD, Meta-Curvature, T-Nets) and their BPTT bottleneck. States the contribution: a trajectory-agnostic meta-objective for non-linear preconditioning via warp layers.

**§2 Warped Gradient Descent.**
- **§2.1 Gradient-based meta-learning.** Defines the MAML objective (Eq. 1) and the family of preconditioned variants (Eqs. 2–5). Pinpoints the three failure modes of BPTT-through-adaptation: cost, vanishing/exploding gradients, credit-assignment.
- **§2.2 General-purpose preconditioning.** Generalises T-Nets by replacing linear projections with universal-function-approximator warp layers $\omega$. Derives the preconditioned gradient (Eq. 6) via Jacobian chain rule.
- **§2.3 The geometry of WarpGrad.** Establishes Riemann-metric interpretation: $G^{-1} = [D_x\Omega][D_x\Omega]^\top$ (Eqs. 7, 8); first-order equivalence between $\mathcal{P}$-descent and natural-gradient-like descent in $\mathcal{W}$ (Eq. 9); canonical meta-objective (Eq. 10).
- **§2.4 Meta-learning warp parameters.** Trajectory-agnostic operational meta-objective (Eq. 11); first-order stop-gradient relaxation (Eq. 12); discussion of how $L_\text{task}$ and $L_\text{meta}$ can differ. Algorithms 1 (online, constant memory, linear in $K$) and 2 (offline, replay buffer).
- **§2.5 Integration with learned initialisations.** Composition with priors over $\theta_0$ (multi-task, point estimate, Bayesian); combined objective $J(\phi,\theta_0) = L(\phi) + \lambda C(\theta_0)$ (Eq. 13). Connection to Mirror Descent with a meta-learned dual space.

**§3 Related work.** Lineage from Schmidhuber 1987 / Bengio et al. 1991 evolutionary meta-learning; Hochreiter et al. 2001 gradient-descent meta-learning; fast vs slow weights (Hinton & Plaut 1987; Schmidhuber 1992; Ba et al. 2016; HyperNetworks); few-shot prediction-of-parameters methods; second-order optimisation (Natural Gradient, K-FAC, Hessian-free) and their stability issues.

**§4 Experiments.**
- **§4.1 Few-shot learning.** Warp-MAML on miniImageNet (5-way 1/5-shot) and tieredImageNet (5-way 1/5-shot). Convolutional task-learner (4 conv blocks, batch-norm, max-pool, ReLU) with $3\times 3$ conv warp-layers inserted after each block. Online meta-training (Alg. 1). Result: state-of-the-art improvements (Table 1).
- **§4.2 Multi-shot learning.** New tieredImageNet protocol with 640 adaptation steps (impossible for MAML). Warp-Leap with offline meta-training. Multi-shot Omniglot (20-way 100-shot). Ablation: residual / non-linear warps further improve results.
- **§4.3 Complex meta-learning.**
  - (c.1) RL maze navigation (Miconi 2018): Warp-RNN = HyperNetwork-LSTM modulating task-RNN weights; outperforms L2RL and Hebbian meta-learning. Linear warps actually *hurt* here — non-linearity is essential.
  - (c.2) Continual sine-regression (5 sub-tasks): $L_\text{meta} = $ average over current + previous sub-tasks; WarpGrad preserves performance on past sub-tasks within 1 order of magnitude.

**§5 Conclusion.** WarpGrad merges memory-based flexibility with gradient-based inductive bias. Avoids BPTT-through-adaptation. Scales to large architectures, RL, and continual learning. Open directions: understanding the relation to second-order optimisation (already shown not-Fisher in Appendix G); designing warp layers with stronger inductive biases.

**Appendix A — practical guidelines** for warp-layer design.

**Appendix B — offline meta-training algorithm** with replay buffer and per-step memory bound.

**Appendix C — Geodesic interpretation.** Warp-Leap corresponds to a joint search for a geometry in which task adaptation traces *geodesics* (shortest paths under $G$).

**Appendix D — Synthetic 2-D illustration** showing how warped surfaces (top row) are smoother than native loss surfaces (bottom row).

**Appendix E–J — experimental details** (architectures, hyper-parameters, ablations on offline vs online, on initialisation, on Natural-Gradient comparison).

**Appendix G — WarpGrad vs Natural Gradient Descent.** Empirical Kronecker-factored covariance check: Warp-Leap preconditioners are *not* approximations to the inverse Fisher. The learned geometry is genuinely meta-learned, not curvature-estimated.

---

## Paper 3 — Harrison et al. 2022 — A closer look at learned optimization

- **PDF:** `docs/project/references/Learning_Rate/sources/Harrison et al. 2022 - A closer look at learned optimization - Stability, robustness, and inductive biases.pdf`
- **Venue:** NeurIPS 2022.
- **Authors:** James Harrison, Luke Metz, Jascha Sohl-Dickstein (Google Research, Brain Team).
- **Code:** `learned_optimization/learned_optimizers/adafac_nominal.py` (Google's `learned_optimization` repository).

### Phase 1 — Foundational Overview (Undergraduate-Level)

**Introduction.**
The first two papers in this review built L2O — a powerful but practically unreliable framework. Black-box learned optimizers (LSTM-based, in the Andrychowicz lineage) do beat hand-tuned Adam *inside their meta-training distribution*, but they suffer two devastating failure modes that have kept them from broad adoption:
1. **Out-of-distribution divergence.** Apply a learned optimizer for more steps than it was meta-trained on, or to a task with a different architecture, and it often *diverges* — climbing the loss instead of descending.
2. **Unstable meta-training.** Meta-training itself is fragile: high sensitivity to random seed, long stuck phases, inconsistent progress. The standard remedy — meta-training across thousands of tasks for weeks on million-dollar hardware — is itself an impediment to research.

Harrison et al. ask a different question than the first two papers. They ask: *why are learned optimizers unstable, and what minimal inductive biases would make them stable by design?* Their tool is **dynamical-systems theory**: they study the eigenvalues of the parameter-update dynamics in a "noisy quadratic" toy problem and derive concrete conditions under which the dynamics are stable. They then translate those conditions into three architectural changes, which together produce the **STAR** (**S**tabilized **T**hrough **A**mple **R**egularization) learned optimizer.

**Key Findings.**
1. **Three inductive biases stabilize learned optimizers:**
   - **Nominal term** ("bias toward descent") — add a known-good optimizer (e.g. Adam / AggMo) to the learned optimizer's output, so the combined update is guaranteed to be a descent direction when the learned term vanishes.
   - **Weight decay on the learned optimizer's parameters** — controls the magnitude of the learned update and prevents the upper eigenvalue bound from being violated.
   - **Preconditioning at the output of the learned term** — apply an Adam-style $1/\sqrt{v}$ preconditioner to the learned update, reducing the impact of the problem Hessian on stability.
2. **A theorem characterising stability.** For the noisy quadratic with $A = I - \alpha H - PH$, stability $\rho(A) \le 1$ requires $\lambda_{\min}(P) \ge -\alpha$ (lower bound: nominal term gives stability margin) and $\lambda_{\max}(P) \le 2/\lambda_{\max}(H) - \alpha$ (upper bound: too-large learned updates cause oscillatory divergence).
3. **STAR generalizes dramatically out of meta-distribution.** Meta-trained on a 2-hidden-layer MLP on FashionMNIST for 2000 inner steps, STAR optimizes (a) CNNs on CIFAR-10, (b) ResNet-style models on 32×32 ImageNet, (c) LSTMs on LM1B, and (d) a 5-layer 256-hidden decoder-only **Transformer on LM1B** (175× more parameters, 5× more steps than meta-training). The baseline black-box optimizer (Metz et al.'s `small_fc_lopt`) diverges in all four. This is the first demonstration of effective generalization after meta-training on a *single* task.
4. **Adaptive nominal term beats blackbox-cancels-nominal.** A common implementation trick — letting a large blackbox term cancel the nominal term — looks equivalent in clean theory but is shown to be **robust-unstable**: a multiplicative-error analysis (D-stability) shows that the blackbox-cancels-nominal design has worse robust-stability margins than a magnitude-controlled nominal term.

**Initial Takeaway.**
Harrison's paper is the "engineering closure" of the L2O program. Where Andrychowicz proved the *expressive power* of learned optimizers and Flennerhag gave the *geometric framework*, Harrison delivers the *stability theory* that says which design choices make a learned optimizer trustworthy. For the project, this paper is the **stability rubric** against which any modulator-conditioned optimizer extension must be checked. A naive "add a modulator $m$ to the LSTM input" can easily blow up exactly the way Harrison's black-box baseline does — diverging when the modulator pushes the optimizer outside its meta-training distribution.

### Phase 2 — Graduate-Level Deep Dive

#### Problem statement (Section 3) — formal setup

The optimizer operates on the parameters $\phi \in \Phi \subseteq \mathbb{R}^N$ of an inner model with stochastic loss $L_t(\phi) = \mathbb{E}_{x_t}[\ell(x_t;\phi)]$. The learned optimizer $f(\cdot;\theta)$ takes input features $z_t$ (parameters, loss values, gradients, iteration index, plus optimizer hidden state) and produces an update:
$$\phi_{t+1} = \phi_t - f(z_t;\theta). \tag{2}$$
Meta-training minimises a (possibly time-weighted) sum of inner losses,
$$\hat\theta = \arg\min_{\theta\in\Theta}\, \mathcal{L}(\theta;T), \qquad \mathcal{L}(\theta;T) = \sum_{t=1}^T w_t L_t(\phi_t). \tag{3}$$
This is the same trajectory-weighted meta-loss as Andrychowicz, but the analysis below focuses on a tractable simplification: the **noisy quadratic** setting.

#### The noisy quadratic setting (Section 4) — analytical tractability

Per-step loss (Eq. 4):
$$\ell(\phi_t) = \tfrac{1}{2}(\phi_t - \xi_t)^\top H\,(\phi_t - \xi_t), \quad \xi_t \overset{\text{iid}}{\sim} \mathcal{N}(0,\Sigma_\xi),$$
with gradient $\nabla_t = H(\phi_t - \xi_t)$. The loss is wide-NN-regime accurate (cited Refs. 54–55) and standard in optimizer theory.

Consider an update with a fixed "nominal" term $g_t = \alpha\nabla_t$ (analogue of plain SGD) plus a learned dense preconditioner $P$ acting on the gradient:
$$\phi_{t+1} = \phi_t - (g_t + P\nabla_t). \tag{5}$$
Substituting gives autonomous linear dynamics (Eq. 6):
$$\phi_{t+1} \;=\; \underbrace{(I - (\alpha I + P)H)}_{=: A}\,\phi_t \;+\; \underbrace{(\alpha I + P)H}_{=: I - A}\,\xi_t. \tag{6}$$

#### Stability ↔ meta-gradient health

From Eq. 6, iterating gives $\phi_t = A^t \phi_0 + \sum_{k=0}^{t-1} A^{t-k-1}(I-A)\xi_k$, hence $\phi_t \sim \mathcal{N}(A^t\phi_0, \Sigma_t)$ with
$$\Sigma_t = \sum_{k=0}^{t-1} A^{t-k-1}(I-A)\Sigma_\xi (A^{t-k-1}(I-A))^\top. \tag{7}$$
The expected meta-loss (Eq. 8):
$$\mathcal{L}(\theta;T) = \frac{1}{T}\sum_{t=1}^T \big(\phi_0^\top (A^t)^\top H A^t \phi_0 + \mathrm{tr}\,H(\Sigma_t + \Sigma_\xi)\big).$$

**Key polynomial-degree argument.** The per-step loss at time $t$ is polynomial in $A$ of degree $2t$. Hence the gradient $\partial \mathcal{L}/\partial P$ is polynomial in the entries of $A$ of degree $2t-1$. If the **spectral radius** $\rho(A) > 1$, then both $\mathbb{E}\lVert\nabla_\theta\mathcal{L}\rVert$ and $\mathrm{Var}(\nabla_\theta\mathcal{L})$ **diverge** with horizon $T$. So:
> Instability of the inner dynamics $\phi_t$ propagates to instability of the meta-gradient, which propagates to meta-training failure.

This is the formal link between "learned optimizer is unstable inside its rollout" and "meta-training itself fails". It justifies treating inner stability as a design constraint, not just a runtime concern.

#### Theorem 1 — the central stability result (Section 4.1)

> Let $A = I - \alpha H - PH$ with $H$ symmetric positive-definite, $P$ diagonalizable with real eigenvalues, and $\alpha \ge 0$. Then $\rho(A) \le 1$ iff
> $$-\alpha \le \lambda_{\min}(P) \tag{9}$$
> $$\lambda_{\max}(P) \le \frac{2}{\lambda_{\max}(H)} - \alpha. \tag{10}$$

**Derivation sketch.** Eigenvalues of $A$ are $\lambda_i(A) = 1 - (\alpha + \lambda_i(P))\lambda_i(H)$ under the simultaneously-diagonalizable assumption ($P$ shares eigenvectors with $H$ — guaranteed in the proof; intuition holds otherwise). Stability $|\lambda_i(A)| \le 1$ gives $0 \le (\alpha + \lambda_i(P))\lambda_i(H) \le 2$. Lower bound: $\alpha + \lambda_i(P) \ge 0$ ⇒ $\lambda_{\min}(P) \ge -\alpha$ (Eq. 9). Upper bound: $\alpha + \lambda_i(P) \le 2/\lambda_i(H)$ ⇒ tightest at $\lambda_{\max}(H)$ giving $\lambda_{\max}(P) \le 2/\lambda_{\max}(H) - \alpha$ (Eq. 10).

**Interpretation:**
- **Eq. 9 — the lower-bound side: the nominal term gives margin.** If the learned $P$ had a negative eigenvalue (the learned optimizer would push *uphill* in some direction), the nominal $\alpha\nabla_t$ can absorb it as long as $\alpha + \lambda_{\min}(P) \ge 0$. Bigger $\alpha$ ⇒ more "descent insurance".
- **Eq. 10 — the upper-bound side: the nominal term costs margin.** As $\alpha$ grows, the maximum allowed $\lambda_{\max}(P)$ shrinks — i.e., the learned optimizer can only safely add small updates. The dual pressure dictates magnitude regularization of $P$ via weight decay.
- **Hitting the upper bound is realistic.** While the lower bound is exceeded by adversarial $P$ (oscillation-to-divergence), neural networks are observed to drive optimizers *predictably* into the upper-bound regime (Ref. 56) — so weight decay is essential, not optional.

#### Preconditioning at the output (Section 4.2) — Lemma 2

Standard adaptive optimizers (AdaGrad, RMSProp, Adam) approximate $H^{-1/2}$ via the diagonal second moment of gradients and apply
$$\tilde g_t = H^{-1/2} g_t.$$
**Where do you apply this preconditioner?** Harrison argues: at the **output** of the learned optimizer, not the input. Combining output-side preconditioning with the nominal-plus-learned design:
$$\phi_{t+1} = \phi_t - H^{-1/2}(\alpha I + P)\nabla_t. \tag{12}$$
**Lemma 2:** under matching conditions to Theorem 1, stability requires (Eq. 13)
$$\lambda_{\max}(P) \le \frac{2}{\sqrt{\lambda_{\max}(H)}} - \alpha.$$
If $\lambda_{\max}(H) > 1$, this upper bound is **looser** than Eq. 10 — i.e., output-side preconditioning gives the learned term more stability margin.

**Why output-side, not input-side?** Output-side normalization is robust to arbitrary *initialization* of the blackbox term — the blackbox can produce wildly mis-scaled outputs early in meta-training, and the $1/\sqrt{v}$ at the output rescales them into a controllable range. Input-side preconditioning only affects what the blackbox *sees* and gives no protection if the blackbox amplifies poorly.

#### Adaptive nominal term and the robust-stability argument (Section 4.3)

A naive shortcut: let $P = P^* - \alpha I$ and let the blackbox "cancel" the nominal term:
$$\phi_{t+1} = \phi_t - \alpha\nabla_t - (P^* - \alpha I)\nabla_t = \phi_t - P^*\nabla_t. \tag{14}$$
In clean theory this recovers the optimal $P^*$ updates and seems to remove the cost of carrying a nominal. **But:** there is always *multiplicative error* in the learned $P$. Model it as $P = \Delta\tilde P$, $\Delta \in \mathcal{D} = \{\mathrm{diag}(d) : 0 < d_i \le \bar d\}$ (Eq. 15). Then under the blackbox-cancels-nominal scheme:
$$\phi_{t+1} = \phi_t - (\alpha(I-\Delta) + \Delta P^*)\nabla_t. \tag{18}$$
For the adversarial choice $\Delta = (1-\varepsilon)I$:
$$\phi_{t+1} = \phi_t - (\alpha\varepsilon I + (1-\varepsilon)P^*)\nabla_t, \quad A = I - \alpha\varepsilon H - (1-\varepsilon)P^* H. \tag{19}$$
The excess term $\alpha\varepsilon H$ adversarially perturbs stability — for adversarial $\varepsilon$ near 1, the system can be driven to instability.

**Robust-stability conclusion (Lemma 3):** to guarantee $\rho(A) \le 1$ for all $\Delta \in \mathcal{D}$, the bounds become tighter than Theorem 1:
$$-\frac{\alpha}{\max_i d_i} \le \lambda_{\min}(\tilde P), \quad \lambda_{\max}(\tilde P) \le \frac{1}{\max_i d_i}\!\left(\frac{2}{\lambda_{\max}(H)} - \alpha\right). \tag{16, 17}$$
**Design implication:** instead of letting the blackbox cancel the nominal, *directly control* the magnitude of both $\alpha$ and $P$ via a hyperparameter-controller-style scalar magnitude head. This is the **adaptive nominal term**: an additional scalar output head on the optimizer MLP (with an exponential nonlinearity), trained jointly, that scales the nominal contribution over the course of training.

#### Non-Markovian (hidden-state) optimizers (Section 4.4)

When the optimizer carries hidden state (LSTM / momentum / EMA), the joint $(\phi, h)$-dynamics must be analyzed for stability — not just the parameter dynamics in isolation. Polyak momentum's joint analysis gives *looser* upper stability bounds than naive analysis predicts, which is consistent with momentum being a stabilizer in practice. The paper opts for **EMA / momentum-style hidden states that are stable by design** — exponential moving averages decay all transients, so the hidden-state contribution to the dynamics matrix is bounded.

#### The STAR optimizer (Section 5)

STAR applies the three insights to `small_fc_lopt` from Metz et al. 2022 — a 197-weight elementwise MLP optimizer with two heads (direction $d$ and magnitude $m$):
$$f(z_t) = \beta_1\, d_\theta(z_t)\,\exp\!\big(\beta_2\, m_\theta(z_t)\big), \tag{20}$$
with hand-set constants $\beta_1, \beta_2 = 0.001$. STAR replaces this with $f(z_t) = f_b(z_t) + f_g(z_t)$:
- **Nominal term:**
  $$f_g(z_t) = \beta_1 \exp(\beta_2 m_g(z_t))\, g(z_t), \tag{21}$$
  where $g(z_t)$ combines AggMo (Ref. 63) and Adam, $m_g$ is the nominal magnitude head with exponential nonlinearity.
- **Blackbox term:**
  $$f_b(z_t) = \frac{\beta_3\, d(z_t)}{v(z_t)}\,\exp(\beta_4\,m_b(z_t)), \tag{22}$$
  where $v(z_t)$ is the output-side preconditioner (the same $v$ as the Adam EMA gradient magnitude already on the input features — only one extra division), $m_b$ the blackbox magnitude head.
- **Heavy L2 weight decay** on the optimizer's MLP parameters: this is what enforces the small-$\lambda_{\max}(P)$ regime in (10). Output-magnitude regularization (via clipping) is also possible but worse, because it fails on out-of-distribution inputs; weight decay protects for *arbitrary* reasonably-sized inputs.

**Total parameter count.** STAR adds 5 extra MLP weights + 1 scalar β over `small_fc_lopt`'s 197 weights — essentially zero overhead. Computational overhead is dominated by the one extra division at the output, which is negligible.

#### Empirical headlines (Section 5.3, Figure 3, Figure 4)

| Setting | Baseline blackbox (`small_fc_lopt`) | STAR |
|---|---|---|
| Meta-training on MLP-FashionMNIST (2k inner steps) | Slow, unstable convergence | Faster meta-training, more stable across seeds |
| Apply to MLP-FashionMNIST for $> 2$k steps | **Diverges** outside meta-training horizon | Continues descending, ~2 orders of magnitude longer |
| 3-hidden-layer MLP w/ layer-norm on CIFAR-10 | Diverges | Trains; comparable to tuned Adam |
| Shallow ResNet on 32×32 ImageNet | Diverges | Trains |
| 256-unit LSTM language model on LM1B | Diverges | Trains |
| 5-layer 256-hidden decoder-only Transformer on LM1B | Diverges | Trains — *175× more parameters, 5× more steps than meta-training* |

The Transformer-on-LM1B generalization is unprecedented: prior L2O work required massive multi-task meta-training to attempt this kind of out-of-distribution transfer; STAR achieves it from a **single-task meta-training run on a small MLP**.

#### Connection to the project's modulator question

Harrison's paper is the **safety rubric** for the project's modulator-conditioning ambitions. Three implications for the modulator-conditioned-optimizer line of work:

1. **A modulator-conditioned optimizer must keep the nominal-term scaffolding.** If the LSTM / FiLM / hypernetwork branch is the only thing producing the update, no amount of meta-training will fix the divergence-when-out-of-distribution failure mode that Section 5 demonstrates. The architecture should be
   $$\Delta\phi_t = f_g(z_t) + f_b(z_t;\,m_t),$$
   where $f_g$ is a known-good (Adam-style) optimizer and only $f_b$ depends on the modulator $m_t$. The modulator controls the *learned correction*, not the base descent direction.

2. **A modulator changing the optimizer must also change the magnitude controller.** Theorem 1's upper bound $\lambda_{\max}(P) \le 2/\lambda_{\max}(H) - \alpha$ is *the* constraint that determines whether a modulator-driven update stays stable. The natural design: route $m_t$ into the magnitude head $m_b$ of $f_b$ — letting the modulator change the *scale* of the learned correction, with the nominal term and weight decay ensuring stability is preserved regardless of $m_t$'s value. This is structurally identical to FiLM-on-output-magnitude.

3. **The blackbox-cancels-nominal design must be avoided.** Section 4.3's robust-stability argument is especially sharp here: if the modulator's purpose is to make the optimizer "behave differently" in different contexts, then making the blackbox dominant (and have it cancel the nominal) is the maximally fragile design — exactly the multiplicative-error situation Lemma 3 says fails.

(Code-side implementation belongs to `senior-developer`. The math here delivers the stability rubric for that handoff.)

### Appendix: Section-by-Section Backbone

**Abstract.** Learned optimizers struggle with stability and out-of-distribution generalization despite massive meta-training. Uses dynamical-systems theory in the noisy-quadratic setting to characterize stability conditions. Introduces three modifications to architecture and meta-training: nominal term, weight decay, output-side preconditioning. Resulting STAR optimizer outperforms prior SOTA at matched compute, is faster to meta-train, and generalizes to a transformer with 175× more parameters and 5× more steps than meta-training MLP.

**§1 Introduction.** Two failure modes of L2O: (i) reduced performance / divergence when applied outside meta-training distribution (e.g. more steps than trained, different architecture); (ii) unstable meta-training itself (random-seed-dependent, stuck phases, inconsistent progress). Three contributions: dynamical-systems-based stability analysis; architectural/training modifications; experimental demonstration of robust generalization.

**§2 Related Work.** Lineage:
- L2O direct (Andrychowicz 2016; Wichrowska 2017; Metz 2019, 2020, 2022).
- Meta-learning broader (Schmidhuber 1987; Hospedales 2020; few-shot families).
- Adaptive hand-designed optimizers (AdaGrad, RMSProp, Adam) and their hyperparameter-tuning weakness.
- Hyperparameter-controller learned optimizers (Refs. 41–45) — inherent stability via descent-direction outputs.
- Long-computational-graph chaos in meta-training (Metz 2019); truncated zeroth-order optimization (Refs. 8, 51); reinforcement-learning-based meta-training (Refs. 45–47); fixed momentum operators for stable hidden states (Refs. 7, 9, 61).

**§3 Problem Statement.** Formal setup. Inner model parameters $\phi \in \mathbb{R}^N$, expected loss $L_t(\phi) = \mathbb{E}_{x_t}[\ell(x_t;\phi)]$ (Eq. 1). Learned optimizer update $\phi_{t+1} = \phi_t - f(z_t;\theta)$ (Eq. 2). Meta-loss $\mathcal{L}(\theta;T) = \sum_{t=1}^T w_t L_t(\phi_t)$ (Eq. 3). Limits scope to training-loss minimization (validation-loss is left to future work).

**§4 Understanding Optimizer Performance: The Noisy Quadratic Setting.**
- **§4.1 Nominal Terms Shift the Region of Stability.** Per-step quadratic loss $\ell(\phi_t) = \tfrac{1}{2}(\phi_t - \xi_t)^\top H(\phi_t - \xi_t)$ (Eq. 4). Update with nominal + learned preconditioner: $\phi_{t+1} = \phi_t - (g_t + P\nabla_t)$ (Eq. 5). Autonomous dynamics $\phi_{t+1} = A\phi_t + (I - A)\xi_t$ with $A = I - (\alpha I + P)H$ (Eq. 6). Asymptotic stability ⇔ $\rho(A) < 1$; instability of $A$ ⇒ unbounded meta-gradient. **Theorem 1**: $\rho(A) \le 1$ iff $-\alpha \le \lambda_{\min}(P)$ (Eq. 9) and $\lambda_{\max}(P) \le 2/\lambda_{\max}(H) - \alpha$ (Eq. 10). Figure 1: nominal term decreases real parts of eigenvalues; weight decay pulls $P$ toward zero (eigenvalues toward 1); preconditioning reduces dependence on $H$; combined → all eigenvalues inside unit circle.
- **§4.2 Preconditioners can Stabilize and Simplify the Design of Update Dynamics.** Adaptive preconditioners $\tilde g_t = H^{-1/2} g_t$ (Eq. 11) standard in Adam/RMSProp/AdaGrad. Output-side application of preconditioner: $\phi_{t+1} = \phi_t - H^{-1/2}(\alpha I + P)\nabla_t$ (Eq. 12). **Lemma 2**: looser upper bound $\lambda_{\max}(P) \le 2/\sqrt{\lambda_{\max}(H)} - \alpha$ (Eq. 13) when $\lambda_{\max}(H) > 1$. Argues output-side preconditioning is more robust to bad blackbox initialization than input-side.
- **§4.3 Adaptive Nominal Terms Improve Robust Stability.** The blackbox-cancels-nominal shortcut $P = P^* - \alpha I$ recovers clean-theory optimum (Eq. 14) but is fragile under multiplicative error. Disturbance set $\mathcal{D} = \{\Delta = \mathrm{diag}(d), 0 < d_i \le \bar d\}$ (Eq. 15). **Lemma 3**: robust stability requires tightened bounds (Eqs. 16, 17). Adversarial $\Delta = (1-\varepsilon)I$ shows excess instability term $\alpha\varepsilon H$ (Eqs. 18, 19). Conclusion: direct magnitude control on both $\alpha$ and $P$ (via magnitude controller head) beats blackbox-cancels-nominal.
- **§4.4 Non-Markovian Optimizers Require Joint Stability.** Optimizer hidden states (momentum, EMA, LSTM) require joint $(\phi, h)$-stability analysis. EMA/momentum-style hidden states are stable by design. Polyak momentum's joint analysis (Ref. 62) gives *looser* upper stability margins. Filtering of stochastic gradients is a desirable side effect.

**§5 Designing a Better Learned Optimizer.**
- **§5.1 New Design Features in the STAR Optimizer.** (i) Bias toward descent via AggMo+Adam nominal term; (ii) magnitude controller via exponential-nonlinearity scalar head (5 extra params); (iii) weight decay on blackbox params (controls weights, not outputs, so generalizes); (iv) preconditioner-style normalization at blackbox output; (v) EMA-style stable hidden states.
- **§5.2 Overview of the STAR Optimizer.** Built on Metz et al.'s `small_fc_lopt` (Ref. 9): 197-weight elementwise MLP, two hidden layers of width 4. Original parameterization (Eq. 20): $f(z_t) = \beta_1 d_\theta(z_t) \exp(\beta_2 m_\theta(z_t))$. STAR's two-term parameterization: $f = f_b + f_g$ with $f_g$ the nominal term (Eq. 21) and $f_b$ the blackbox term (Eq. 22) with $v(z_t)$ in the denominator for output preconditioning. Adds 5 + 1 = 6 parameters total.
- **§5.3 The STAR Optimizer Improves Performance.** Figure 3: STAR is faster to meta-train, gets better final inner-training loss, and remains stable for ~2 orders of magnitude longer than its meta-training horizon. Figure 4: meta-trained once on MLP-FashionMNIST-2000-steps, STAR generalizes to (a) CNN-LayerNorm-CIFAR10, (b) ResNet-32×32-ImageNet, (c) LSTM-LM1B, (d) Transformer-LM1B. Baseline blackbox diverges in all four; STAR is comparable to tuned Adam, sometimes substantially better.

**§6 Discussion.** Stability injection is a sufficient condition for descent-in-expectation in convex settings, and empirically generalises to non-convex training. Open questions: are these biases *optimal* for neural-network training? Connections to flat-minima literature (Refs. 81, 82). Many stability-analysis tools from the dynamical-systems literature (Refs. 74–78) remain to be exploited.

---

## Cross-paper synthesis

Reading the three papers as a sequence, a clear arc emerges that maps directly onto the project's modulator-conditioning question.

### The L2O design trilemma

| Axis | Andrychowicz 2016 | Flennerhag 2019 (WarpGrad) | Harrison 2022 (STAR) |
|---|---|---|---|
| **What is learned** | The whole update rule $g_\phi$ | A geometry (warp layers $\omega$) | A correction $f_b$ added to a nominal $f_g$ |
| **Inductive bias** | Almost none — RNN can be anything | Strong — gradient descent in a warped space, Riemannian | Strong — descent-direction nominal, weight decay, output preconditioner |
| **Meta-training cost** | High; BPTT through all inner steps | Low — trajectory-agnostic (no BPTT through adaptation) | Moderate; truncated BPTT with three stabilisers |
| **OOD generalization** | Within similar-structure tasks only; fails on activation change | Strong (Warp-Leap transfers to 640-step, recurrent, RL, continual) | Strong — single-task meta-training transfers to Transformer-LM1B |
| **Stability guarantee** | None | Yes, by Riemannian-metric construction | Yes, by eigenvalue-bound analysis on $A$ |
| **Where the modulator naturally enters** | LSTM input; FiLM on LSTM hidden state | Warp-layer FiLM / hypernet conditioning | Magnitude head $m_b$ of the blackbox term |

### Three readings of "modulator-conditioned optimizer"

The same project hypothesis — "the update rule is conditioned on an interoceptive / context signal $m_t$" — has three structurally different mathematical interpretations, one per paper:

1. **Andrychowicz reading: modulator changes the *function* the optimizer computes.**
   $$\begin{bmatrix} g^{(i)}_t \\ h^{(i)}_{t+1} \end{bmatrix} = m_\phi\!\big(\nabla^{(i)}_t,\, h^{(i)}_t,\, m_t\big).$$
   Maximally flexible, no stability guarantees, requires lots of meta-training. *Risk:* Harrison's failure modes — diverges out of distribution.

2. **Flennerhag reading: modulator changes the *geometry* the optimizer sees.**
   $$\Delta\gamma = G^{-1}(\gamma;\,\phi, m_t)\,\nabla L(\gamma),$$
   with warp layers $\omega(\cdot;\phi,m_t)$ conditioned on the modulator. Stability inherits from Riemannian-metric construction; trajectory-agnostic meta-training; scales to long adaptation. *This is the cleanest mathematical home for the project's hypothesis.*

3. **Harrison reading: modulator changes the *learned correction* on top of a known-good optimizer.**
   $$\Delta\phi_t = f_g(z_t) + f_b(z_t; m_t).$$
   The nominal $f_g$ guarantees descent; the modulator $m_t$ only changes the corrective term $f_b$ (e.g. through its magnitude head $m_b$). Stable by design; generalizes by design. *This is the cleanest engineering home for the project's hypothesis.*

In practice, readings 2 and 3 are **complementary**, not competing — a WarpGrad-style architecture (modulator-conditioned warp layers) can sit *inside* a STAR-style scaffolding (nominal Adam + output preconditioner + weight-decayed blackbox). That composition would be the "best of both worlds" instantiation of modulator-conditioned L2O.

### Open questions for the project (handoff signals)

These are not implementation plans — they are mathematical / experimental questions that future direction-setting agents (`research-postdoc`, `professor-dl-theory`, `professor-rl`) and the `senior-developer` for implementation should address:

- **Does Harrison's eigenvalue bound (Theorem 1) extend to modulator-conditioned $P$?** The dynamics become $A_t = I - \alpha H - P(m_t)H$. If $P$'s spectrum changes with $m_t$, the stability condition $\rho(A_t) \le 1$ becomes time-varying. Stability of time-varying linear systems is a well-studied problem (e.g. uniform exponential stability) — the analytical extension is non-trivial but tractable.
- **Does Flennerhag's first-order equivalence (Eq. 9) hold when $\Omega$ depends on $m_t$?** If yes, modulator-conditioned WarpGrad inherits its Riemannian-descent guarantees automatically; if no, the geometric framework must be re-derived.
- **What inner-task distribution is "rich enough" to meta-train a modulator-conditioned optimizer?** Harrison shows that a single MLP-on-FashionMNIST task suffices for stable STAR. Whether a single task suffices for a *modulator-conditioned* STAR depends on whether the modulator's role is exposed by within-task variation (e.g. mid-training context switches) or only across-task variation.

---

## Document status

- **Papers processed:** 3 of 3.
- **Total length:** ~720 lines.
- **Tmp working files:** `tmp/20260519_011645_litreview_Learning_Rate.md` (this doc's seed), `tmp/20260519_011645_{andrychowicz,flennerhag,harrison}_full.txt` (raw PDF extracts).
- **Companion reviews planned by user:** B (direct LR adaptation: hypergradient, Meta-SGD); C (RL-specific meta-gradient: TIDBD, Meta-grad RL); D (gated/hypernet/GLU architectures).
