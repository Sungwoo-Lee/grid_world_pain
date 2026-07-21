> **Per-paper review — in-context-learning corpus, paper 20 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§20); content is identical. Manifest: [[in_context_learning_sources]].

# 20. Nguyen & Grover 2022 — Transformer Neural Processes

**PDF:** `docs/project/references/in_context_learning/sources/Nguyen and Grover 2022 - Transformer Neural Processes.pdf` · ICML 2022 · arXiv:2207.04179

## Phase 1: Foundational Overview (Undergraduate-Level)

**The problem in plain terms.** Neural Processes (NPs) and even Attentive NPs (§3) still have two structural problems for **uncertainty-aware meta-learning** — the setting where a model must both predict accurately *and* quantify how uncertain it is, so that its uncertainty can guide decisions (Bayesian optimization, contextual bandits). Problem one: NPs use a latent variable $z$, which makes the marginal likelihood *intractable*, forcing them to optimize a variational lower bound (the ELBO) instead of the true likelihood — and lower bounds do not always give meaningful latents. Problem two: NPs *underfit*; ANPs fix the fit but tend to become *overconfident* and do poorly on sequential decision-making.

**The idea: treat it as sequence modeling.** Transformer Neural Processes (TNPs) drop the latent variable entirely and instead cast the whole problem as **autoregressive sequence modeling**, exactly like a GPT language model. You lay out the context points and target points as a sequence and train a Transformer (with a causal mask) to predict each target's label from all previous points, maximizing the *actual* conditional log-likelihood — no latent variable, no variational bound, and a very expressive predictive distribution.

**The catch and the fix.** A vanilla GPT cannot be dropped in directly: it uses positional encodings that make its output depend on the *order* of the context and target points, but an NP must be **invariant to context order** and **equivariant to target order**. TNPs fix this by (a) concatenating each $x_i$ with its $y_i$ into a single token and *removing positional encodings*, (b) a custom padding-and-masking scheme so the autoregressive structure is respected without leaking order into predictions, and (c) a Monte-Carlo symmetrization to make the prediction equivariant to target order.

**Key findings / initial takeaway.** TNPs come in three flavors trading expressivity for computation: **TNP-A** (autoregressive, most expressive), **TNP-D** (diagonal, assumes targets independent given context), **TNP-ND** (non-diagonal, models a full covariance via Cholesky). Empirically TNPs achieve state-of-the-art on 1-D meta-regression, image completion (EMNIST, CelebA), contextual bandits (wheel problem), and Bayesian optimization, beating CNP/NP/ANP and their bootstrapped variants by large margins. Takeaway: **you can get a better, likelihood-exact neural process by throwing out the latent variable and modeling the context-plus-targets as a sequence with a permutation-respecting Transformer.**

## Phase 2: Graduate-Level Deep Dive

**Background — the NP latent-variable likelihood and its intractability.** With context $C = \{x_i,y_i\}_{i=1}^m$ and targets $T=\{x_i,y_i\}_{i=m+1}^N$, the latent-variable NP likelihood is

$$
p(y_{m+1:N}\mid x_{m+1:N}, C) = \int_z p(y_{m+1:N}\mid x_{m+1:N}, z)\, p(z\mid C)\, dz, \tag{1}
$$

which is intractable, so NPs maximize the ELBO

$$
\log p(y_{m+1:N}\mid x_{m+1:N}, C) \ge \mathbb{E}_{q(z\mid C,T)}\big[\log p(y_{m+1:N}\mid x_{m+1:N}, z)\big] - \mathrm{KL}\big(q(z\mid C,T)\,\|\,p(z\mid C)\big), \tag{2}
$$

viewable as a VAE over target labels conditioned on context: the encoder $q(z\mid C,T)$ is permutation-invariant, and the decoder factorizes $p(y_{m+1:N}\mid x_{m+1:N}, z) = \prod_{i=m+1}^N p(y_i\mid x_i, z)$ (targets independent given $z$).

**The TNP objective — the amortized autoregressive conditional-predictive loss (the core contribution).** TNP treats each set of evaluations $\{x_i,y_i\}_{i=1}^N$ as an ordered sequence, segregates a random subset $(x_{1:m}, y_{1:m})$ as context, and **autoregressively** models the remaining targets:

$$
\mathcal{L}(\theta) = \mathbb{E}_{x_{1:N}, y_{1:N}, m}\big[\log p_\theta(y_{m+1:N}\mid x_{1:N}, y_{1:m})\big] \tag{4}
$$
$$
= \mathbb{E}_{x_{1:N}, y_{1:N}, m}\Bigg[\sum_{i=m+1}^N \log p_\theta(y_i\mid x_{1:i}, y_{1:i-1})\Bigg], \tag{5}
$$

where each conditional is a univariate Gaussian and $m$ is sampled uniformly to define the context/target split. **Contrast with the rest of the shard.** This is the *same* held-out conditional-log-likelihood idea as CNP (Eq. 4 of §2) and PFN (Eq. 2 of §1), but factorized **autoregressively** rather than **fully** (CNP's $\prod_{x\in T}Q(f(x)\mid O,x)$ conditions every target only on the *context*; TNP-A conditions target $i$ on the context *and all previous targets*). This chain-rule factorization is exact — no variational bound, no latent variable — and strictly more expressive because it models target-target dependencies. It is the "no latent path at all, maximally expressive deterministic prediction" corner of the design space: where CNP amortizes a factored posterior predictive and the NP/ANP latent path samples one global $z$ for coherent joint samples, TNP-A obtains joint coherence through the autoregressive chain rule instead of through a latent.

**The two desiderata (Properties 3.1, 3.2) — why a vanilla GPT fails.**
- **Property 3.1 (Context invariance):** for any permutation $\pi$ and $m\in[1,N-1]$,
$$
p_\theta(y_{m+1:N}\mid x_{m+1:N}, x_{1:m}, y_{1:m}) = p_\theta(y_{m+1:N}\mid x_{m+1:N}, x_{\pi(1):\pi(m)}, y_{\pi(1):\pi(m)}).
$$
Permuting the context must not change target predictions.
- **Property 3.2 (Target equivariance):** for any permutation $\pi$,
$$
p_\theta(y_{m+1:N}\mid x_{m+1:N}, x_{1:m}, y_{1:m}) = p_\theta(y_{\pi(m+1):\pi(N)}\mid x_{\pi(m+1):\pi(N)}, x_{1:m}, y_{1:m}).
$$
Permuting target inputs must permute predictions correspondingly.

A vanilla GPT with positional encodings violates both: positional encodings are needed to pair $x_i$ with $y_i$, but they make outputs order-dependent (breaking context invariance), and different target orderings give non-equivariant predictions.

**TNP-A: Autoregressive TNP (§3.1).** To get pairing *without* order-leakage: concatenate $x_i$ and $y_i$ into a single token (so no positional encoding is needed to associate them). For a target point $y_{i>m}$ that must depend on previous pairs *and* its own input $x_i$, introduce **auxiliary padded tokens** $x_{i>m}$ padded with a dummy $0$, appended to the sequence:

$$
\tau = \{(x_1,y_1),\dots,(x_N,y_N),(x_{m+1},0),\dots,(x_N,0)\}. \tag{6}
$$

A custom **attention mask** enforces the autoregressive order: (1) context points $(x_i,y_i)_{i=1}^m$ attend only to themselves; (2) target point $(x_i,y_i)$ for $i>m$ attends to all context and previous target points $(x_j,y_j)_{j=m+1}^{i}$; (3) padded target $(x_i,0)$ for $i>m$ attends to all context and previous targets $(x_j,y_j)_{j=m+1}^{i-1}$. This masking gives **Property 3.1 by construction** (context tokens only see each other, order-independently).

**Achieving Property 3.2 by symmetrization — the derivation.** Any function can be made equivariant to a group by averaging its evaluations over the group. Define the equivariant joint distribution as an average over the permutation group:

$$
\tilde p_\theta(y_{m+1:N}\mid x_{1:N}, y_{1:m}) = \mathbb{E}_\pi\big[\, p_\theta(y_{\pi(m+1):\pi(N)}\mid x_{\pi(m+1):\pi(N)}, x_{1:m}, y_{1:m}) \,\big]. \tag{7}
$$

Because the full permutation group is intractable to enumerate, use a **Monte-Carlo average over randomly sampled permutations**. Consequently TNP-A satisfies target equivariance only *in the limit* (as the number of sampled permutations grows), but remains tractable at training and evaluation. This is the price of TNP-A's expressivity: exact equivariance is only approached.

**TNP-D: Diagonal TNP (§3.2).** Assume targets are conditionally independent given context and inputs:

$$
p_\theta(y_{m+1:N}\mid x_{1:N}, y_{1:m}) = \prod_{i=m+1}^N p_\theta(y_i\mid x_i, x_{1:m}, y_{1:m}). \tag{8}
$$

Drop the real target pairs from the sequence and feed only context points plus padded targets $(x_i,0)_{i=m+1}^N$. This is a multivariate normal with **diagonal covariance**; equivariance (Property 3.2) holds **exactly** without permutation averaging (each target predicted independently, structurally identical to CNP's factorization but with attention). The cost is the strong independence assumption — no target-target correlations.

**TNP-ND: Non-Diagonal TNP (§3.3).** Parametrize a multivariate normal with a **full covariance**:

$$
p_\theta(y_{m+1:N}\mid x_{1:N}, y_{1:m}) = \mathcal{N}\big(y_{m+1:N}\,\big|\, \mu_\theta(x_{1:N},y_{1:m}),\ \Sigma_\theta(x_{1:N},y_{1:m})\big). \tag{9}
$$

As with TNP-D, real target pairs are removed (targets predicted jointly). Two covariance parametrizations: **Cholesky** $\Sigma = LL^\top$ ($L$ lower-triangular with positive diagonal; used in the main experiments) and **low-rank** $\Sigma = \exp(D) + AA^\top$ ($D$ diagonal, $A$ low-rank). Because the number of targets varies, $L$ cannot be output directly by a fixed-size head; instead, the last masked self-attention layer's outputs $z_{m+1:N}$ feed (a) an MLP producing means $\mu_{m+1:N}$ and (b) a second self-attention stack + projection producing vectors $h_i\in\mathbb{R}^p$. Stacking these into $H\in\mathbb{R}^{n\times p}$ (with $n=N-m$), the Cholesky factor is computed as

$$
L = \text{lower}(HH^\top), \qquad H\in\mathbb{R}^{n\times p}, \tag{10}
$$

where $\text{lower}(\cdot)$ keeps the lower triangle. This does not span all lower-triangular matrices but has two virtues: it handles an arbitrary number of targets, and its space complexity is $O(n)$ rather than $O(n^2)$. **The expressivity/tractability ladder:** TNP-A (autoregressive, most expressive, equivariance only in MC limit) $\succ$ TNP-ND (full covariance via Cholesky, exact equivariance) $\succ$ TNP-D (diagonal, cheapest, exact equivariance) — confirmed empirically (Table 1: on RBF, TNP-A 1.63 > TNP-ND 1.46 > TNP-D 1.39 log-likelihood).

**Architecture instantiation.** A GPT-style stack of masked self-attention layers with the custom mask above, no positional encodings, tokens = concatenated $(x_i,y_i)$ (or padded $(x_i,0)$). Self-attention over the full context+target sequence is what supplies the cross-attention-like context selection of ANP *and* the target-target dependencies that ANP's latent path could only approximate.

**Empirical results.** (i) *1-D regression* (train on random-hyperparameter RBF GP, test on RBF / Matérn-5/2 / Periodic): TNPs beat CNP/CANP/NP/ANP/BNP/BANP on 2/3 kernels by large margins (Table 1). (ii) *Image completion* (EMNIST, CelebA): TNP-A best (CelebA 5.82 vs ANP 2.90, BANP 3.09), strong generalization to unseen EMNIST classes (Table 2). (iii) *Contextual bandits* (wheel problem, varying difficulty $\delta$): TNPs achieve lowest cumulative regret across all $\delta$, degrading only slightly as difficulty rises (Table 3), directly demonstrating good uncertainty for sequential decision-making — the setting where ANP's overconfidence hurt. (iv) *Bayesian optimization*: TNPs optimize even the periodic-kernel functions better than baselines despite worse raw regression likelihood there.

## Appendix: Section-by-Section Backbone

- **Abstract.** NPs define distributions over functions and estimate uncertainty, but underfit and often have intractable likelihoods, limiting sequential decision-making. TNPs cast uncertainty-aware meta-learning as sequence modeling, learned via an autoregressive likelihood objective with a transformer architecture respecting context invariance and target equivariance. Knobs trade decoding-distribution expressivity vs computation. SOTA on meta-regression, image completion, contextual bandits, Bayesian optimization.
- **§1 Introduction.** Meta-learning = fast adaptation to unseen tasks from few examples; uncertainty matters for sequential decision-making. NPs = flexible function distributions, VAE-over-datasets structure, amortized functional uncertainty in latents. Two drawbacks: (1) intractable marginal likelihood → variational surrogate → possibly meaningless latents; (2) underfitting (ANP partly fixes fit but is overconfident, poor on decision-making). TNP: autoregressive conditional log-likelihood, no latents, GPT-style causal-mask transformer; but vanilla transformer lacks context invariance / target equivariance → fixes via token concatenation, padding/masking, MC symmetrization. Three variants (A/D/ND).
- **§2 Background.** §2.1 Uncertainty-aware meta-learning (unknown distribution over functions $F$; per-function $D_{\text{train}}$/$D_{\text{test}}$; joint predictive over test set, e.g. diagonal-Gaussian $\mu_j, \sigma_j$). §2.2 Neural Processes (latent likelihood Eq. 1; ELBO Eq. 2; VAE view, permutation-invariant encoder, factored decoder). §2.3 Transformers (self-attention Eq. 3; positional encodings; arbitrary-length sequences).
- **§3 Transformer Neural Processes.** Sequence-modeling framing; autoregressive objective Eqs. 4–5 (univariate-Gaussian conditionals, MC over batch and $m$). Property 3.1 context invariance, Property 3.2 target equivariance; vanilla GPT violates both via positional encodings. §3.1 TNP-A (concatenate $x_i,y_i$; padded tokens Eq. 6; three-rule attention mask; Property 3.1 by masking; Property 3.2 by symmetrization Eq. 7 with MC permutation average, exact only in limit). §3.2 TNP-D (conditional independence Eq. 8; drop target pairs, feed padded targets; diagonal covariance; exact equivariance; strong assumption). §3.3 TNP-ND (full-covariance normal Eq. 9; Cholesky $\Sigma=LL^\top$ and low-rank $\exp(D)+AA^\top$; parameterization Eq. 10 $L=\text{lower}(HH^\top)$, arbitrary target count, $O(n)$ space).
- **§4 Experiments.** §4.1 1-D regression (train random-hyperparameter RBF GP; test RBF/Matérn/Periodic; Table 1 TNPs win 2/3; A > ND > D). §4.2 Image completion (EMNIST, CelebA; Table 2 large gains, TNP-A best, unseen-class generalization; Fig. 3 crisper completions). §4.3 Contextual bandits (wheel problem, difficulty $\delta$; UCB; cumulative regret; Table 3 TNPs best across all $\delta$). §4.4 Bayesian optimization (TNPs optimize periodic functions better despite lower regression likelihood).
- **§5 Related Work.** NP variants (Conditional NP, latent NP, Attentive NP, Bootstrapping NP/BANP, Convolutional CNP); transformers for sequence modeling and few-shot; discussion of latent-free vs latent-variable NPs.
- **Appendix.** A.1 low-rank covariance results; B experimental details / additional metrics (B.1 regression, B.2 image samples, B.3 simple-regret bandit results); architecture and hyperparameter details; open-sourced code (TNP-pytorch), baselines from official BNP implementation.

---
