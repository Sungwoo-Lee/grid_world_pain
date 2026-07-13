> **Per-paper review — continual-learning corpus, paper 4 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§4); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 4. Zenke et al. (2017) — Continual Learning Through Synaptic Intelligence (SI)

**PDF:** `docs/project/references/continual_learning/sources/Zenke et al. 2017 - Synaptic Intelligence.pdf`
**Venue:** ICML 2017 (PMLR 70; Stanford — Zenke, Poole, Ganguli). **Type:** algorithm + theory (supervised).
**Primer link:** the **regularization** family, "EWC's online cousin" — importance accumulated *along the whole trajectory* instead of at task end.

## Phase 1: Foundational Overview

**Introduction (plain language).** SI shares EWC's goal — protect the weights that mattered for old tasks — but changes *how importance is measured*. EWC waits until a task is finished, then does a separate pass to compute the Fisher information at the final weights. SI instead watches each weight *while* it learns and keeps a running tally of how much that weight has been *pulling the loss down* over the entire training trajectory. A weight that repeatedly contributed to reducing the loss is deemed important and gets consolidated (made stiff) when the task ends. The paper frames this as giving each synapse a richer internal state — "intelligent synapses" that are 3-dimensional (they track current value, old value, and accumulated importance) rather than a single scalar — echoing the biological fact that real synapses carry complex molecular machinery, not one number.

**Key finding.** SI matches EWC's forgetting-resistance on **split MNIST**, **permuted MNIST**, and **split CIFAR-10/100**, but computes importance **online and locally** (no separate end-of-task Fisher pass, no summing over output classes — so it scales to high-dimensional outputs where EWC's exact Fisher is expensive). A bonus empirical result: on split CIFAR, SI-consolidated networks sometimes *generalize better on new tasks* than networks trained from scratch — consolidation acts as a helpful regularizer against overfitting.

**Initial takeaway.** SI is the "trajectory-integrated" regularizer. Same penalty *shape* as EWC (a per-weight-weighted quadratic anchoring to old values), different *importance estimator* — cheap, streaming, biologically flavoured. The theory section proves that in a tractable quadratic case the SI importance reduces to the Hessian, giving it the same curvature meaning EWC's Fisher has.

## Phase 2: Graduate-Level Deep Dive

**Setup and notation.** Training traces a trajectory $\theta(t)$ in parameter space. For an infinitesimal update $\delta(t)$, the change in loss is, to first order,

$$
\mathcal{L}(\theta(t) + \delta(t)) - \mathcal{L}(\theta(t)) \approx \sum_k g_k(t)\, \delta_k(t), \qquad g_k = \frac{\partial \mathcal{L}}{\partial \theta_k}. \tag{1}
$$

So each parameter's change $\delta_k = \theta_k'(t)$ contributes $g_k(t)\,\delta_k(t)$ to the total loss change. Summing (integrating) over the whole trajectory gives a **path integral of the gradient field**:

$$
\int_{\mathcal{C}} g(\theta(t))\, d\theta = \int_{t_0}^{t_1} g(\theta(t)) \cdot \theta'(t)\, dt. \tag{2}
$$

Because the gradient is a *conservative* field, this integral equals the *net* loss change $\mathcal{L}(\theta(t_1)) - \mathcal{L}(\theta(t_0))$ regardless of path. The key move is to **decompose it per-parameter**, defining the per-weight importance $\omega_k^\mu$ for task $\mu$:

$$
\int_{t_{\mu-1}}^{t_\mu} g(\theta(t)) \cdot \theta'(t)\, dt = \sum_k \int_{t_{\mu-1}}^{t_\mu} g_k(\theta(t))\, \theta_k'(t)\, dt \;\equiv\; -\sum_k \omega_k^\mu. \tag{3}
$$

The minus sign is a convention (we care about *decreasing* the loss, so a weight that reduces loss earns positive $\omega_k^\mu$). **$\omega_k^\mu$ is the total amount by which parameter $k$ drove the loss down over task $\mu$'s training.** In practice it's a cheap running sum: at each SGD step accumulate the product of the gradient $g_k$ and the actual update $\theta_k'$. (Because SGD is noisy, this over-estimates the true $\omega_k^\mu$ — a known bias, corrected empirically by the strength parameter $c<1$ below.)

**The surrogate loss and consolidation penalty.** The problem: we want to minimize $\mathcal{L} = \sum_\mu \mathcal{L}^\mu$ over all tasks but only ever see one $\mathcal{L}^\mu$ at a time. Catastrophic forgetting is when minimizing the current $\mathcal{L}^\mu$ inadvertently raises past losses $\mathcal{L}^\nu$ ($\nu<\mu$). SI replaces the inaccessible past losses with a **quadratic surrogate** anchored at the end-of-previous-task weights $\tilde{\theta}_k = \theta_k(t_{\mu-1})$. The modified objective is

$$
\tilde{\mathcal{L}}^\mu = \mathcal{L}^\mu + c \sum_k \Omega_k^\mu \big(\tilde{\theta}_k - \theta_k\big)^2, \tag{4}
$$

where $c$ is a dimensionless strength trading old vs. new memories, and the per-parameter regularization strength is

$$
\Omega_k^\mu = \sum_{\nu < \mu} \frac{\omega_k^\nu}{(\Delta_k^\nu)^2 + \xi}, \qquad \Delta_k^\nu \equiv \theta_k(t_\nu) - \theta_k(t_{\nu-1}). \tag{5}
$$

Here $\Delta_k^\nu$ is how far weight $k$ *moved* during task $\nu$, and $\xi$ is a small damping constant preventing blow-up when $\Delta_k^\nu \to 0$. **Read Eq. (5) carefully:** importance = (loss reduction the weight achieved) divided by (distance-squared it travelled). The denominator serves two roles: (i) it makes the term carry the same units as the loss (so the penalty is dimensionally a loss), and (ii) it normalizes out how far the weight happened to move, isolating *efficiency* of loss reduction. Note $\tilde{\mathcal{L}}^\mu$ has the **exact same form as EWC's Eq. (3)** — a per-weight-weighted quadratic pulling $\theta_k$ back to a reference value — only the weighting $\Omega_k^\mu$ is trajectory-derived rather than Fisher-derived. Bookkeeping: $\omega_k$ accrues continuously during training; $\Omega_k^\mu$ and the references $\tilde{\theta}$ update only at task boundaries; $\omega_k$ resets to zero after each consolidation.

**Interpretation of the surrogate (Fig. 2).** The quadratic surrogate is *not* the Hessian-at-the-minimum quadratic. It is chosen to match three properties of the actual descent on the old task: the total loss drop $\mathcal{L}(\theta(0)) - \mathcal{L}(\theta(T))$, the net parameter motion $\theta(0)-\theta(T)$, and having its minimum at the endpoint $\theta(T)$. Those three conditions uniquely fix the surrogate quadratic that "summarizes" the whole descent trajectory.

**Theory — the SI importance recovers the Hessian.** The paper's analytic core proves that in a clean case the path-integral importance $Q$ (the matrix whose diagonal is $\omega$) equals the Hessian, giving SI the same curvature-based meaning EWC's Fisher has. Consider a quadratic error

$$
E(\theta) = \tfrac{1}{2}(\theta - \theta^*)^\top H (\theta - \theta^*), \tag{6}
$$

with minimum $\theta^*$ and Hessian $H$. Continuous-time gradient descent obeys

$$
\tau \frac{d\theta}{dt} = -\frac{\partial E}{\partial\theta} = -H(\theta - \theta^*), \tag{7}
$$

whose exact solution from initial $\theta(0)$ is

$$
\theta(t) = \theta^* + e^{-H t/\tau}\big(\theta(0) - \theta^*\big), \tag{8}
$$

with update velocity

$$
\theta'(t) = \frac{d\theta}{dt} = -\frac{1}{\tau} H\, e^{-H t/\tau}\big(\theta(0) - \theta^*\big). \tag{9}
$$

Since $g = \tau\, d\theta/dt$, the importances (Eq. 3) are the diagonal of the time-integrated outer product of the velocity:

$$
Q = \tau \int_0^\infty dt\; \frac{d\theta}{dt}\,\frac{d\theta}{dt}^{\!\top}. \tag{10}
$$

Diagonalize $H$ with eigenpairs $(\lambda_\alpha, u_\alpha)$ and let $d_\alpha = u_\alpha \cdot (\theta(0)-\theta^*)$ be the projection of the total displacement onto eigenmode $\alpha$. Substituting (9) into (10), changing to the eigenbasis, and doing the Gaussian time integral $\int_0^\infty e^{-(\lambda_\alpha + \lambda_\beta)t/\tau}dt = \tau/(\lambda_\alpha+\lambda_\beta)$ yields

$$
Q_{ij} = \sum_{\alpha\beta} u_i^\alpha\, d_\alpha\, \frac{\lambda_\alpha \lambda_\beta}{\lambda_\alpha + \lambda_\beta}\, d_\beta\, u_j^\beta. \tag{11}
$$

Note $Q$ no longer depends on the descent speed $\tau$ (it is a steady-state, time-integrated quantity).

**Three cases where $Q$ reduces to $H$.**
1. *Averaged over random initial conditions.* If the displacements $d_\alpha$ are zero-mean iid with variance $\sigma^2$, then $\langle d_\alpha d_\beta\rangle = \sigma^2 \delta_{\alpha\beta}$, and the double sum in (11) collapses (using $\tfrac{\lambda_\alpha^2}{2\lambda_\alpha} = \tfrac{\lambda_\alpha}{2}$):

$$
\langle Q_{ij}\rangle = \tfrac{1}{2}\sigma^2 \sum_\alpha u_i^\alpha \lambda_\alpha u_j^\alpha = \tfrac{1}{2}\sigma^2 H_{ij}. \tag{12}
$$

So the correlation of parameter updates, integrated over time, *is the Hessian* up to the scale factor $\sigma^2$ — and the $(\Delta_k^\nu)^2$ denominator in Eq. (5) (which averages to $\sigma^2$ at zero damping) exactly removes that scale factor. This is the theoretical justification for the normalization in Eq. (5).

2. *Diagonal Hessian.* If $H$ is diagonal, $u_i^\alpha = \delta_{\alpha i}e_i$, so eigenvalues are the diagonal entries $\lambda_i = H_{ii}$ and (11) reduces to $Q_{ij} = \delta_{ij}(d_i)^2 H_{ii}$. Normalizing by $(d_i)^2$ recovers the diagonal Hessian.

3. *Rank-1 Hessian.* If only $\lambda_1 \neq 0$, then $Q_{ij} = \tfrac{1}{2}(d_1)^2 u_i^1 \lambda_1 u_j^1 = \tfrac{1}{2}(d_1)^2 H_{ij}$. The paper flags this as the *interesting* case for continual learning: a low-rank error leaves many weight-space directions unconstrained by the current task, i.e. spare capacity for future tasks — the geometric reason weight-anchoring can work without freezing.

**Caveat the theory states.** These exact correspondences hold because for a quadratic loss $H$ is constant along the trajectory. For general losses $H$ varies along the path, so no exact SI↔Hessian↔endpoint-Fisher identity holds; but empirically SI's importance *correlates* with endpoint measures (Fisher, etc.), which the authors offer as the explanation for why SI and EWC perform comparably despite computing importance so differently.

**Experiments.** *Split MNIST* (5 tasks, each a binary digit pair, multi-head, MLP 2×256 ReLU, $\xi=10^{-3}$): with consolidation ($c=1$) old-task accuracy stays near 1; without ($c=0$) it drops to chance (Fig. 3). *Permuted MNIST* (MLP 2×2000, $\xi=0.1$, $c=0.1$ via grid search, Adam state retained across tasks): SI (blue) tracks EWC and both stay high over 10 tasks while SGD and SGD+dropout collapse (Fig. 4); importance-correlation matrices (Fig. 5) show consolidation keeps per-task important-weight sets *uncorrelated* (different weights per task), whereas fine-tuning lets second-layer importances become correlated across tasks — the mechanistic signature of forgetting. *Split CIFAR-10/100* (CNN, 6 tasks): consolidation shows no age-dependent accuracy decline; and green (consolidation) ≥ gray (from-scratch) on validation while the reverse holds on *training* accuracy — i.e. consolidation reduces overfitting, generalizing better on new tasks with limited data.

*Relevance note.* SI is the online/streaming member of the regularization family and the cleanest theoretical bridge in this shard between "importance" and "loss curvature" (its Hessian result). For any project idea that wants a *cheap, per-step* estimate of which parameters matter — computed without a separate Fisher pass — SI is the template. The path-integral / conservative-field argument is also a reusable analytic tool.

## Appendix: Section-by-Section Backbone

- **Abstract.** Intelligent synapses accumulate task-relevant information online; store new memories without forgetting; reduces forgetting while staying computationally efficient.
- **§1 Introduction.** ANNs freeze after training; retraining on shifted distributions ⇒ overfitting/forgetting. Biological synapses are complex molecular machines, not scalars. Proposal: 3D synaptic state (past value, current value, importance $\omega$); consolidate important synapses at task switch; new tasks learned by unimportant synapses.
- **§2 Prior work.** Taxonomy: (1) architectural (freezing, reduced LR, ReLU/Maxout/LWTA, dropout, progressive nets — grows with tasks); (2) functional (LwF distillation, activation-$\ell_2$ — expensive, need old-net forward passes); (3) structural (EWC — diagonal Fisher, cost linear in #outputs, limits high-dim outputs). SI is structural but online.
- **§3 Synaptic framework.** Loss-change first order Eq. (1); path integral Eqs. (2)–(3) defining $\omega_k^\mu$; online running-sum approximation; SGD noise ⇒ over-estimate. Surrogate loss Eq. (4); regularization strength Eq. (5) with $\Delta_k^\nu$ and damping $\xi$; strength $c$ ($c=1$ ideal, <1 to compensate noise); update schedule ($\omega$ continuous, $\Omega$/reference at task end, $\omega$ reset). Fig. 1/Fig. 2 surrogate intuition (3 matching conditions).
- **§4 Theoretical analysis.** Quadratic error Eq. (6); continuous-time descent Eqs. (7)–(9); $Q$ matrix Eq. (10)–(11); reductions to Hessian: averaged over inits Eq. (12) (justifies Eq. 5 normalization), diagonal Hessian Eq. (13), rank-1 Hessian Eq. (14) (interesting low-rank/spare-capacity case). Caveat: exact only for constant $H$; general case only correlational.
- **§5 Experiments.** 5.1 Split MNIST (Fig. 3, $c=0$ vs $c=1$). 5.2 Permuted MNIST (Fig. 4 vs EWC/SGD/dropout; Fig. 5 importance-correlation matrices). 5.3 Split CIFAR-10/100 (Fig. 6; consolidation prevents age-dependent decline + reduces overfitting).
- **§6 Discussion.** Similar to EWC but online + trajectory-wide; needs higher-dimensional synapses; biology of complex synapses (state-dependent plasticity, decaying tags, reversible changes). "Add intelligence to synapses" as a research direction.

---
