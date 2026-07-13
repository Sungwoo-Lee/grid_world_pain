> **Per-paper review — continual-learning corpus, paper 7 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§7); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 7. Ash & Adams (2020) — On Warm-Starting Neural Network Training

**PDF:** `docs/project/references/continual_learning/sources/Ash & Adams 2020 - On Warm-Starting Neural Network Training.pdf`
**Venue:** NeurIPS 2020. **Authors:** Jordan T. Ash (Microsoft Research NYC), Ryan P. Adams (Princeton).

## Phase 1 — Foundational Overview

**The problem in one sentence.** When new data arrive and you want to update a neural network, the intuitive, cheap thing to do is to keep yesterday's weights and just keep training on the bigger dataset — a "warm start." This paper shows that warm-starting *hurts the model's test accuracy*, even though it reaches exactly the same training accuracy as a network trained from scratch.

**The concrete demonstration.** Train an 18-layer ResNet on 50% of CIFAR-10 to convergence, then continue training it on 100% of CIFAR-10. Compare it to a fresh randomly-initialized ResNet trained on 100% of CIFAR-10. Both reach ~100% *training* accuracy, but the warm-started one generalizes several percentage points *worse* on held-out test data (e.g., 51.7% vs. 56.2% for ResNet+SGD on CIFAR-10). The gap is robust across architectures (ResNet, MLP), optimizers (SGD, Adam), and datasets (CIFAR-10, CIFAR-100, SVHN). Logistic regression — a convex model — is *not* damaged, which is the tell that this is a non-convex-optimization pathology, not a statistical one.

**Key findings.**
- The damage appears after *very little* pretraining — a few epochs, even before the pretraining phase reaches 100% accuracy. Early stopping does not save you.
- No standard fix works: batch normalization, larger/smaller batch size, larger learning rate, L2 weight decay, confidence penalties, adversarial training — all fail to close the gap while preserving the warm start's speed benefit. Warm-started models that *do* generalize well only do so by essentially "forgetting" their initialization (their converged weights end up nearly uncorrelated with the warm-start weights), which erases the time savings.
- **The fix — shrink-and-perturb (SP).** Before the next round of training, replace each weight by a shrunken-plus-noised version: multiply it by a factor $\lambda<1$ and add small Gaussian noise. This one-line trick closes the generalization gap *and* keeps the training-speed benefit of warm-starting.

**Initial takeaway.** A "head start" from prior training can be a liability, not an asset, whenever the second-round dataset is large (data-rich). The mechanism is a **gradient imbalance**: for a warm-started network, gradients from new/unseen data are much larger in magnitude than gradients from already-fit data, so the optimizer's trajectory is biased and lands in a worse-generalizing minimum. Shrink-and-perturb re-balances those gradient magnitudes without destroying the learned function. This is the supervised-learning ancestor of the whole "resets" line in RL (Nikishin 2022, D'Oro 2023) — see the project primer §2 Phase 2.

## Phase 2 — Graduate-Level Deep Dive

**Setup and the empirical gap.** Let a dataset $\mathcal{D}$ be split into halves $\mathcal{D}_1, \mathcal{D}_2$. Warm-starting fits $\theta^{(1)} = \arg\min_\theta \mathcal{L}(\theta; \mathcal{D}_1)$ to convergence, then initializes round two at $\theta^{(1)}$ and fits on $\mathcal{D}_1 \cup \mathcal{D}_2$. The random-init control fits the same union from a fresh $\theta_0 \sim \text{init}$. Both attain $\approx 100\%$ training accuracy (zero training loss on a modern over-parameterized ResNet), so the test-accuracy gap is entirely an implicit-bias / generalization phenomenon, not an optimization-failure-to-fit phenomenon. The gap is shown (Appendix Fig. 10) to be *inversely proportional to the fraction of data available in round one* — the more you pretrain, the worse the eventual generalization.

**The shrink-and-perturb operator.** At training round $t$, each learnable parameter is re-initialized as

$$\theta_i^{t} \;\leftarrow\; \lambda\, \theta_i^{t-1} \;+\; p_t, \qquad p_t \sim \mathcal{N}(0, \sigma^2), \quad 0 < \lambda < 1 .$$

In practice the perturbation $p_t$ is drawn as a *scaled freshly-initialized network* rather than i.i.d. Gaussian, so that per-layer variances match the architecture's initialization scheme. Two claims justify why this works: (i) shrinking preserves the learned hypothesis, and (ii) shrinking re-balances the gradients.

**(i) Shrinking preserves the hypothesis (Proposition 1).** Consider an $L$-layer ReLU network with no bias terms and no batch normalization. ReLU is positively homogeneous: $\text{ReLU}(\lambda z) = \lambda\,\text{ReLU}(z)$ for $\lambda>0$. Propagating a global weight-shrink by $\lambda$ through all $L$ layers, the pre-softmax logits $z(x) = f_\theta(x)$ scale as

$$f_{\lambda\theta}(x) \;=\; \lambda^{L}\, f_\theta(x) .$$

*Derivation.* Let layer $\ell$ compute $h_\ell = \text{ReLU}(W_\ell h_{\ell-1})$ with $h_0 = x$. Replacing every $W_\ell \to \lambda W_\ell$:
$h_1' = \text{ReLU}(\lambda W_1 x) = \lambda\,\text{ReLU}(W_1 x) = \lambda h_1$; inductively $h_\ell' = \text{ReLU}(\lambda W_\ell \cdot \lambda^{\ell-1} h_{\ell-1}) = \lambda^{\ell} h_\ell$. The final linear map (layer $L$) gives logits $z' = \lambda W_L h_{L-1}' = \lambda\cdot\lambda^{L-1} z = \lambda^L z$. $\square$

Because $\arg\max_c z_c = \arg\max_c \lambda^L z_c$ for $\lambda>0$, the **predicted class label is unchanged** — the hypothesis $\arg\max f_\theta(x)$ is preserved. What *does* change is the *confidence*: shrinking the logits by $\lambda^L$ pushes the softmax toward uniform, raising output entropy and hence raising the cross-entropy loss. For architectures with batch-norm (ResNet), BN's running mean/variance absorb the rescaling, so performance is essentially invariant to $\lambda$ except at extreme shrinkage (Fig. 6). For ReLU MLPs *with* bias, the property degrades gracefully — damage only appears for $\lambda < 0.6$.

**(ii) Shrinking re-balances gradients (the mechanism).** The paper's diagnosis is a gradient-magnitude imbalance. In warm-started round-two training, decompose the batch loss into contributions from already-fit data $\mathcal{D}_1$ and new data $\mathcal{D}_2$. Because the network already fits $\mathcal{D}_1$ (near-zero loss there), $\|\nabla_\theta \mathcal{L}(\theta;\mathcal{D}_1)\| \ll \|\nabla_\theta \mathcal{L}(\theta;\mathcal{D}_2)\|$. The aggregate gradient is dominated by $\mathcal{D}_2$, so the optimizer moves as if it were doing a biased single-task step — an imbalance known to be pathological in multi-task optimization (gradient surgery, Yu et al. 2020). Figure 5 measures the two gradient norms separately over round two and shows a drastic gap for warm-started models.

Shrink-and-perturb repairs this because shrinking *raises the loss on the already-fit data*: from Proposition 1, shrinking logits by $\lambda^L$ increases the cross-entropy on $\mathcal{D}_1$ from ~0 back to a nontrivial value, which restores $\|\nabla_\theta \mathcal{L}(\theta;\mathcal{D}_1)\|$ to a magnitude comparable with $\|\nabla_\theta \mathcal{L}(\theta;\mathcal{D}_2)\|$. The gradients are re-standardized *while the coarse learned structure (the argmax hypothesis) is preserved* — that combination is exactly what pure noise injection cannot achieve (noise re-balances gradients but destroys the function; Appendix Table 4).

**Relationship to weight decay.** Applying SP at *every* SGD step yields the update

$$\theta_i \;\leftarrow\; \lambda\!\left(\theta_i + \eta\,\frac{\partial \mathcal{L}}{\partial \theta_i}\right) + p ,$$

so the shrink factor $\lambda$ behaves like a weight-decay coefficient plus injected noise. But SP is *not* reducible to weight decay: Appendix Table 3 shows L2-regularized models are still vulnerable to the warm-start gap, and SP closes the gap even on top of aggressive weight decay. The regularization benefit is a small secondary effect (marginal improvement even on static data); the primary benefit is the sequential-training gradient re-balancing.

**Trade-off surface.** Figure 8 sweeps $(\lambda, \sigma)$: the bottom-left corner ($\lambda\to0$) is pure random init (best generalization, slowest), the top-right ($\lambda\to1$) is pure warm start (fast, worst generalization). Intermediate $\lambda\approx0.6$, noise scale $0.01$ recovers random-init test accuracy while retaining most of the speed; smaller $\lambda$ can even *outperform* random init. Adding the perturbation improves both time and generalization over shrink-alone.

**Pre-training / transfer connection.** SP also robustifies transfer: when pre-training on a source then fitting a fraction of a target, warm-starting helps when target data are *scarce* (few-shot regime) but crosses over to hurting when target data are *abundant* (Fig. 9). SP tracks the better of the two strategies automatically at $\lambda=0.3$, noise $10^{-4}$, removing the need to predict the crossover.

**Project relevance.** This is the canonical citation for "carrying weights forward can lose to a from-scratch baseline" — precisely the regime the project's curriculum agent hit (primer cross-links `curriculum_underperformed_baseline_plasticity_vs_budget`). SP ($\theta\leftarrow\lambda\theta+\epsilon$) is a *partial-reset* primitive whose RL descendants (Nikishin's last-layer resets, D'Oro's high-replay-ratio resets) are the corrective family in the primer §4.

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** Real ML systems ingest data piecemeal (finance, ads, recsys, active learning). Convex theory says warm-start; deep nets contradict it — warm-starting hurts generalization without hurting training accuracy. Motivates studying "when and how" plus a simple fix. Notes findings are *not* inconsistent with small-data pre-training / few-shot transfer; the claim is specifically about the *large-data* second round.
- **§2 Warm Starting Damages Generalization.** §2.1 Basic batch updating: 50%→100% two-phase protocol; Table 1 shows consistent, significant test-accuracy damage for ResNet & MLP across SGD/Adam and CIFAR-10/100/SVHN; convex LR unaffected; effect stronger on harder datasets. §2.2 Online learning: streaming CIFAR-10 in 1000-sample batches to a ResNet; warm-start trains far faster but generalization gap grows with more data (Fig. 2).
- **§3 Conventional Approaches.** §3.1 Batch size / learning rate sweeps (Fig. 3): warm-started models can match random-init accuracy only by giving up the speed benefit; well-generalizing warm-started models have low weight-correlation with their init (Appendix Fig. 11) — they "forgot." §3.2 Speed of damage (Fig. 4): only a few epochs of pretraining suffice to inflict the gap. §3.3 Regularization (L2, confidence penalty, adversarial): helps a little, does not close the gap.
- **§4 Shrink, Perturb, Repeat.** Defines SP $\theta_i^t \leftarrow \lambda\theta_i^{t-1}+p_t$. "Shrinking preserves hypotheses" (Prop 1, $\lambda^L$ logit-scaling, entropy increase). "Shrink-perturb balances gradients" (Fig. 5, the imbalance mechanism). Trade-off study $(\lambda,\sigma)$ (Figs. 7, 8). §4.1 relation to weight decay (per-step SP ≈ noisy L2, but not reducible to it). §4.2 relation to pre-training (crossover; SP tracks the better strategy, Fig. 9).
- **§5 Discussion / Related Work.** Warm-start understood for convex models only; connections to critical learning periods (Achille et al.), initialization theory, distance-from-init generalization (Nagarajan & Kolter), margin/flat-minima/complexity generalization literature, and pre-training.
- **§6 Broader Impact.** SP reduces the compute/energy cost of retraining-from-scratch ("Red AI"), democratizing online/active-learning research.

---
