---
title: "How Does Batch Normalization Help Optimization?"
authors: Shibani Santurkar, Dimitris Tsipras, Andrew Ilyas, Aleksander Madry
year: 2018
venue: NeurIPS 2018
slug: santurkar_2018_batchnorm_optimization
source_pdf: sources/How Does Batch Normalization Help Optimization_.pdf
topic: FiLM
---

# Santurkar et al. 2018 — How Does Batch Normalization Help Optimization?

## Plain-English entry point

This paper asks why **batch normalization** — the technique of inserting a layer that subtracts the running mean and divides by the running standard deviation of each feature channel before re-applying a learned per-channel scale-and-shift — makes deep neural networks train faster and more stably. The original 2015 explanation by Ioffe and Szegedy was that batch norm reduces **"internal covariate shift" (ICS)**: the phenomenon where each layer's input distribution keeps changing as the layers below it update, supposedly forcing the layer to chase a moving target. That explanation was widely accepted but never really tested.

Santurkar et al. systematically demolish the ICS explanation in two stages. First, they inject **strong, time-varying noise** after the batch-norm layer — deliberately re-introducing covariate shift — and show that performance barely degrades. So distributional stability isn't load-bearing. Second, they propose a more relevant operational definition of ICS — the change in a layer's loss gradient before vs. after the lower layers are updated — and find that networks with batch norm often have **more** ICS by this measure, even while training faster.

Then they offer a new explanation: batch norm makes the **loss landscape smoother**. Concretely, it (a) tightens the **Lipschitz constant** of the loss function (the loss can't change too quickly with respect to its inputs) and (b) tightens the **beta-smoothness** (the gradient can't change too quickly either). The empirical signatures are dramatic: along the direction of any gradient step, the standard network's loss varies wildly and the gradient flips direction within small distances, while the batch-norm network has predictable, monotone behavior. They back this up with a small set of clean theorems showing the bounded loss-Lipschitzness and gradient-Lipschitzness follow from the batch-norm reparametrization, and that the same smoothing effect is produced by alternative normalization schemes (L1-norm, L∞-norm-based), implying it is the normalization-as-reparametrization, not BN's specifics, that matters. This paper is the standard citation for "why FiLM-style layers, which often sit on top of normalization, train well in the first place".

## Section-ordered backbone

### Abstract
Conventional wisdom says BatchNorm works by controlling internal covariate shift (ICS); this paper shows ICS is largely unrelated to BN's effectiveness. Instead, BN's fundamental impact is to **smooth the optimization landscape**, making gradients more predictive and stable, which permits larger learning rates and faster convergence.

### 1. Introduction
Surveys BN's empirical dominance (>6000 citations at the time) despite poor theoretical understanding. The accepted explanation is ICS reduction, but this is not concretely supported. The paper sets out to (a) test the ICS hypothesis, (b) identify what BN actually does, (c) prove theorems.

### 2. Batch Normalization and Internal Covariate Shift

**2.1 Does BN performance stem from controlling ICS?** Trains VGG on CIFAR-10 with and without BN: BN clearly wins (Figure 1a, b). But visualizing the activations' distributions over training (Figure 1c) shows the difference in distributional stability is marginal — not the clean "BN flattens distributions" story.

Killer experiment: train a network with BN, then **inject** non-zero-mean, non-unit-variance, time-varying noise *after* the BN layer. This deliberately reintroduces severe covariate shift at every step. Result: the noisy-BN network nearly matches the clean-BN network's performance (Figure 2) and *still* beats the no-BN baseline. So distributional stability of layer inputs is *not* the mechanism.

**2.2 Is BN reducing ICS at all?** Proposes a sharper, optimization-grounded definition: ICS at layer $i$ at time $t$ is the L2 difference between the gradient $G_{t,i}$ computed assuming all layers update simultaneously and the gradient $G'_{t,i}$ computed *after* the lower layers have already been updated (Definition 2.1). Under this operational definition, BN networks often show **more** ICS than standard networks. Particularly stark in deep linear networks where the standard network sees almost zero ICS yet the BN network has near-uncorrelated $G$ and $G'$ — and the BN network *still* trains drastically faster. So BN's success cannot be explained by ICS reduction.

### 3. Why Does BatchNorm Work?

**3.1 The smoothing effect.** BN reparametrizes the optimization problem to make its landscape significantly smoother: (a) the loss is more Lipschitz (changes more slowly as one moves in input space, gradient magnitudes are smaller); (b) the gradients are more Lipschitz — "effective beta-smoothness" — so the gradient at a point predicts the gradient a step away. The mechanism: when you take a step of size eta along the gradient direction, the gradient direction at the new point is still a good estimate of the actual gradient — so you can confidently take larger steps. Vanishing/exploding gradients, kink-sensitivity, learning-rate sensitivity, and initialization sensitivity all become manifestations of this smoothing.

**3.2 Empirical exploration.** Figure 4 shows three measurements as a function of training step: (a) the range of loss values along the gradient direction over a fixed step distance (huge for standard, tiny for BN); (b) the L2 distance between the gradient at the starting point and gradients at nearby points along the gradient direction (~100x smaller for BN early in training); (c) "effective beta-smoothness" (maximum gradient change over distance moved). All three favor BN by orders of magnitude, especially early in training.

**3.3 Is BN the only way to smooth the landscape?** Test alternative normalization strategies that fix the first moment but normalize by L1, L2, or L∞ norm (rather than L2 variance, which is what BN does). All match or exceed BN's training-speed gains while *not* producing Gaussian-looking distributions of layer inputs. This further breaks the ICS explanation: ICS is not even reduced by these alternatives, yet the optimization benefit is preserved. The smoothing effect is general to "natural normalization", not specific to BN's exact formulation.

### 4. Theoretical Analysis

**4.1 Setup.** Compare a linear layer $W$ with output $y = Wx$, vs. the same layer followed by a BatchNorm layer producing $\hat y = (y - \mu)/\sigma$ then $z = \gamma \hat y + \beta$. The downstream loss $\mathcal{L}$ may include arbitrary further non-linear layers. Both networks have identical $W$ and identical loss; analyze the difference made by inserting BN.

**4.2 Theoretical Results.**

- **Theorem 4.1 (loss Lipschitzness).** The squared gradient magnitude of the BN loss w.r.t. layer outputs is bounded by the squared gradient magnitude of the original loss, minus additive terms that grow with the dimension and with the correlation between the gradient and the (normalized) activations. Combined with the variance-rescaling factor $\gamma^2/\sigma^2$, BN produces an effectively flatter landscape.
- **Theorem 4.2 (gradient smoothness, the beta-smoothness theorem).** The Hessian quadratic form in the gradient direction is rescaled by $1/\sigma^2$ and *reduced* by an additive term involving the inner product between the gradient and the normalized activations. Under mild PSD conditions (locally convex loss, common with softmax+cross-entropy and piecewise-linear non-linearities) and a sign condition on the activation-gradient inner product (mild as long as the gradient points roughly toward the optimum), the BN network's gradients are *more predictive* than the standard network's.
- **Observation 4.3.** BN's reparametrization is not just a rescaling: for any $W$, there exists a BN configuration with the same activations, so all minima of the standard landscape are preserved in the BN landscape — BN doesn't change *where* the optima are, only the geometry of the path to them.
- **Theorem 4.4.** The activation-space results carry over to a minimax worst-case bound on the weight-space Lipschitzness.
- **Lemma 4.5.** BN-network optima tend to be closer (in L2) to common initializations, so BN also benefits *initialization*.

### 5. Related Work
Surveys layer norm, weight norm, instance norm, group norm — all post-BN normalization schemes — and notes that the new smoothing interpretation likely applies to all of them, opening principled normalization-design as a research direction.

### 6. Conclusion
BN works by reparametrizing to smooth the loss landscape, not by reducing internal covariate shift. The effect is theoretically grounded and generic across natural normalization schemes.

## Phase 1 — Undergraduate-level synthesis

**The key idea in one sentence.** Batch normalization makes the loss surface flatter and the gradient more reliable. The "internal covariate shift" story that motivated batch norm in 2015 turns out to be wrong — the actual benefit is smoother optimization.

**Worked example.** Imagine you are training a deep network and you measure two things at every step.
1. Take the current gradient. Take a step of size eta in that direction. How much did the loss change? With a standard network, the loss might oscillate wildly — sometimes the step lowers the loss, sometimes it raises it sharply. With batch norm, the loss change is much more controlled and predictable.
2. At the new point, measure the gradient again. With a standard network, this new gradient could point in a wildly different direction than the original gradient (sometimes near 90 degrees away). With batch norm, the new gradient is much closer to the original. The gradient field is *Lipschitz* — it doesn't change too fast across the input space.

Why does this matter? Because gradient descent is essentially "look at the local slope, take a step downhill, repeat". If the local slope is reliable as you move, you can confidently take larger steps. Large learning rates are what allow modern deep networks to train in reasonable time. Without batch norm, networks must use small learning rates or risk exploding/vanishing gradients or jumping into sharp local minima. With batch norm, large learning rates become safe.

The killer experiment that closes the case: inject random noise into the activations *after* the batch-norm layer — deliberately undoing the distributional stability batch norm provides. The network *still* trains as fast as standard batch norm. So the distributional-stability story is not the explanation. What survives the noise injection is the *reparametrization* of the loss surface — the smoothing — and that is what carries the optimization benefit.

**Why this matters for the FiLM family.** Conditional Batch Norm, Conditional Instance Norm, AdaIN, and FiLM all sit on top of (or in place of) a normalization step. The reason these methods train at all — and especially the reason FiLM has decoupled itself from the normalization step (Perez et al. 2018 show FiLM works without explicit normalization) — is fully explained by this paper: the underlying mechanism is loss-landscape smoothing, which is general to "normalize-then-affine" reparametrizations and does not depend on ICS reduction.

## Phase 2 — Graduate-level deep dive

### The standard BN equation

For input $x \in \mathbb{R}^{N \times C \times H \times W}$ during training (batch size $N$, channels $C$, spatial $H \times W$), batch normalization computes per-channel statistics

$$
\mu_c = \frac{1}{NHW} \sum_{n,h,w} x_{n,c,h,w}, \qquad
\sigma_c^2 = \frac{1}{NHW} \sum_{n,h,w} (x_{n,c,h,w} - \mu_c)^2,
$$

normalizes,

$$
\hat x_{n,c,h,w} = \frac{x_{n,c,h,w} - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}},
$$

and applies a learned per-channel affine

$$
\mathrm{BN}(x)_{n,c,h,w} = \gamma_c \hat x_{n,c,h,w} + \beta_c.
$$

At inference, population statistics replace minibatch statistics.

### Definition 2.1 (operational ICS)

Let $W^{(t)}_1, \ldots, W^{(t)}_k$ be the parameter tensors of the $k$ layers at training step $t$, and let $(x^{(t)}, y^{(t)})$ be the minibatch. Define

$$
G_{t,i} = \nabla_{W^{(t)}_i} \mathcal{L}(W^{(t)}_1, \ldots, W^{(t)}_k; x^{(t)}, y^{(t)}),
$$

i.e., layer $i$'s gradient at the *current* parameter state. Then define

$$
G'_{t,i} = \nabla_{W^{(t)}_i} \mathcal{L}(W^{(t+1)}_1, \ldots, W^{(t+1)}_{i-1}, W^{(t)}_i, W^{(t)}_{i+1}, \ldots, W^{(t)}_k; x^{(t)}, y^{(t)}),
$$

i.e., the same gradient at layer $i$, but evaluated *after* layers $1, \ldots, i-1$ have already been updated to their new values $W^{(t+1)}_*$.

ICS at layer $i$, step $t$ is

$$
\mathrm{ICS}_{t,i} = \|G_{t,i} - G'_{t,i}\|_2.
$$

This captures the "effective" cross-layer interference an optimizer feels at layer $i$. Empirically, ICS by this definition is comparable or larger in BN networks, even though they train faster.

### Theorem 4.1 — Loss Lipschitzness

Let $y_j$ be the pre-BN activation at neuron $j$, $\hat y_j$ its standardized version, $\gamma$ the BN scale, $\sigma_j$ the minibatch standard deviation of $y_j$, and $m$ the minibatch size. Let $\mathcal{L}$ be the original loss and $\hat{\mathcal{L}}$ the loss in the BN-network. Then

$$
\left\|\nabla_{y_j} \hat{\mathcal{L}} \right\|^2 \le \frac{\gamma^2}{\sigma_j^2} \left( \left\|\nabla_{y_j} \mathcal{L}\right\|^2 - \frac{1}{m} \langle \mathbf{1}, \nabla_{y_j} \mathcal{L} \rangle^2 - \frac{1}{\sqrt{m}} \langle \nabla_{y_j} \mathcal{L}, \hat y_j \rangle^2 \right).
$$

The middle term $\frac{1}{m} \langle \mathbf{1}, \nabla_{y_j} \mathcal{L} \rangle^2$ is the squared mean of the gradient over the minibatch — non-trivial because the inner product with the all-ones vector $\mathbf{1}$ has magnitude growing with $\sqrt{m}$ when the gradient has any consistent bias. The last term $\langle \nabla_{y_j} \mathcal{L}, \hat y_j \rangle^2$ is the squared correlation between the gradient and the activation — empirically nonzero. Both terms *subtract* from the bound, plus the factor $\gamma^2/\sigma_j^2 \approx 1$ in practice (since $\sigma$ is typically large after the rescaling). Net effect: $\|\nabla \hat{\mathcal{L}}\| \le \|\nabla \mathcal{L}\|$ — BN tightens the loss Lipschitz constant.

### Theorem 4.2 — Gradient smoothness (beta-smoothness)

Let $\hat g_j = \nabla_{y_j} \hat{\mathcal{L}}$ and $H_{jj} = \partial^2 \hat{\mathcal{L}} / \partial y_j \partial y_j$. Then

$$
(\nabla_{y_j} \hat{\mathcal{L}})^\top \, \frac{\partial^2 \hat{\mathcal{L}}}{\partial y_j \partial y_j} \, (\nabla_{y_j} \hat{\mathcal{L}}) \le \frac{\gamma^2}{\sigma_j^2} \left( \hat g_j^\top H_{jj} \hat g_j - \frac{1}{m \gamma} \langle \hat g_j, \hat y_j \rangle \left\|\frac{\partial \hat{\mathcal{L}}}{\partial y_j}\right\|^2 \right).
$$

The quadratic form in the gradient direction (which is the relevant Hessian term for Taylor expansion around a gradient step) is rescaled by $1/\sigma_j^2$ — reducing variance-driven sensitivity — *and* reduced by an additive term. Under the two mild conditions (a) $H_{jj} \succeq 0$ (locally convex loss — true for softmax+cross-entropy with piecewise-linear activations), and (b) $\langle \hat g_j, \hat y_j \rangle \ge 0$ (the negative gradient roughly points at the loss minimum in normalized-activation space), the BN gradient is more predictive than the standard one. Concretely, a step of size $\eta$ along $\hat g_j$ takes you to a point whose gradient is closer to $\hat g_j$ than the analogous step in the standard network.

### Observation 4.3 — Reparametrization, not rescaling

For any $W$, there exists a BN configuration $(W, \gamma, \beta)$ with $\gamma_j = \sigma_j$ that produces *identical activations* to the non-BN network. Therefore every minimum of the standard loss landscape corresponds to a minimum of the BN landscape. BN doesn't change the optima; it changes the geometry around them, smoothing the path to any given optimum.

### Theorem 4.4 — Weight-space Lipschitzness

The activation-space Lipschitz improvement translates to a worst-case bound on weight-space Lipschitzness:

$$
\hat g_j = \max_{\|X\| \le \lambda} \|\nabla_W \hat{\mathcal{L}}\|^2
\quad \implies \quad
\hat g_j \le \frac{\gamma^2}{\sigma_j^2} \left( g_j^2 - m \mu_{g_j}^2 - \lambda^2 \langle \nabla_{y_j} \mathcal{L}, \hat y_j \rangle^2 \right).
$$

That is, the same additive-reduction-plus-variance-rescaling story holds in weight space too.

### Lemma 4.5 — Initialization advantage

For weight initialization $W_0$ and optima $W^*$ (non-BN) and $\widehat{W}^*$ (BN, closest to the BN optima),

$$
\|\widehat{W}_0 - \widehat{W}^*\|^2 \le \|W_0 - W^*\|^2 - \frac{1}{\|W^*\|^2}\left(\|W^*\|^2 - \langle W^*, W_0 \rangle\right)^2,
$$

provided $\langle W_0, W^* \rangle > 0$. BN-network optima are no further from the initialization than standard-network optima — sometimes strictly closer. So BN networks both find their optima faster (smoother landscape) and have closer optima to find (favorable initialization).

### Why ICS noise injection doesn't kill BN

The killer experiment (Figure 2) injects time-varying noise of non-zero mean and non-unit variance *after* the BN layer. From the loss-landscape perspective, this is fine: the noise affects the activations, but the BN reparametrization of the *loss surface* remains intact. The gradients of the BN-reparametrized loss are still Lipschitz-bounded; the smoothness bounds in Theorem 4.2 still apply. Distributional stability of activations was a *correlate* of BN training, not a *cause*.

### Why L_p-norm-based normalization works equally well

Section 3.3 tests normalizations that fix the first moment and divide by $\|y\|_p$ for $p \in \{1, 2, \infty\}$. Under L1 normalization, the resulting activations are not Gaussian-like — yet they train as fast as BN or faster. This is consistent with the analysis: any reparametrization that *bounds* the gradient and Hessian quadratic forms via a Lipschitz-style rescaling and additive subtraction will produce the same kind of smoothing. ICS is unrelated; the smoothing is what matters.

### Connection to FiLM

FiLM (Perez et al. 2018) is the canonical example of a downstream method that depends on the substrate that this paper analyzes. FiLM layers traditionally sit immediately after a normalization step (BN, IN, or LN) so that the conditional $(\gamma_{i,c}, \beta_{i,c})$ have a clean, standardized input to operate on. Perez et al.'s ablation that "FiLM without batch normalization" still works (93.7% on CLEVR vs. 97.4% with BN) confirms FiLM's effect is conceptually distinct from BN's, but the BN-or-equivalent normalization is what makes the optimization tractable in the first place. Santurkar et al.'s paper is the standard reference for *why*.

## Connections to other papers in this corpus

- **`perez_2018_film.md`** — Perez et al. cite this paper as justification for using batch norm in their FiLM-ed ResBlocks; the BN ablation in Perez 2018 (Table 2: "no batch normalization" → 93.7%) is consistent with this paper's claim that BN provides optimization smoothing but is not the source of the conditioning effect.
- **`dumoulin_2017_cond_instance_norm.md`** — CIN sits on top of instance norm; this paper's claim that the smoothing effect generalizes to "natural normalization schemes" implies the same optimization rationale applies to instance norm.
- **`huang_belongie_2017_adain.md`** — AdaIN replaces the learned (gamma, beta) with statistics from a style image; the underlying instance-norm-as-smoother story carries over.
- **`birnbaum_2019_temporal_film.md`** and **`wisnu_2025_stsm_film.md`** — Both rely on FiLM-on-top-of-normalization architectures whose trainability is explained here.
