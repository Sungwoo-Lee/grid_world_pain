> **Per-paper review — in-context-learning corpus, paper 19 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§19); content is identical. Manifest: [[in_context_learning_sources]].

# 19. Kim et al. 2019 — Attentive Neural Processes

**PDF:** `docs/project/references/in_context_learning/sources/Kim et al. 2019 - Attentive Neural Processes.pdf` · ICLR 2019 · arXiv:1901.05761

## Phase 1: Foundational Overview (Undergraduate-Level)

**The problem in plain terms.** Neural Processes (NPs) — the latent-variable cousin of CNPs from §2 — have a nagging flaw: they **underfit**. When you show an NP some observed points and ask it to predict at *those very same input locations*, it gets them noticeably wrong (inaccurate means, over-inflated variances). Visually, in 1-D curve fitting the predicted curve misses the context dots; in face-completion the reconstructed top half looks blurry even though the model was given the top half.

**Why it underfits.** In an NP, the encoder mashes the whole context set into a **single fixed-length summary vector** by *averaging* the per-point representations. Averaging gives every context point equal weight, so when the decoder wants to predict at a query location, it cannot easily tell *which* context points are the relevant ones. The mean-aggregation is an information bottleneck.

**The fix: attention.** The paper draws an analogy to GPs, where the **kernel** measures similarity between two input locations and tells you which observed points matter for a given query. Attentive Neural Processes (ANPs) replace the mean-aggregation with a **differentiable attention** mechanism: each target query *attends to* the context points, giving more weight to the relevant ones — exactly like a learned kernel. This preserves permutation invariance (attention over a set is order-independent) while curing the bottleneck. They add **self-attention** among context points (to model interactions between them) and **cross-attention** from each target query to the contexts (to fetch a query-specific summary).

**Key findings / initial takeaway.** ANPs dramatically improve accuracy at context locations, train faster (both in iterations and wall-clock, despite attention's extra cost), and model a wider range of functions than NPs. On 1-D GP regression, dot-product and multihead attention nearly perfectly reproduce context points; on MNIST/CelebA completion, reconstructions become crisp and can even map between resolutions. The cost: complexity rises from $O(n+m)$ to $O(n(n+m))$. Takeaway: **attention is the neural analogue of a GP kernel; bolting it onto NPs fixes the underfitting that a fixed-size mean summary causes.**

## Phase 2: Graduate-Level Deep Dive

**NP background — the deterministic path and the latent path made explicit.** The NP defines an (infinite) family of conditional distributions: condition on context $(x_C, y_C) := (x_i,y_i)_{i\in C}$ to predict targets $(x_T, y_T)$, invariant to context and target ordering. The **deterministic NP** models:

$$
p(y_T \mid x_T, x_C, y_C) := p(y_T \mid x_T, r_C), \qquad r_C := r(x_C, y_C)\in\mathbb{R}^d, \tag{1}
$$

where $r$ is a deterministic, permutation-invariant aggregator (each pair through an MLP, then **mean**), and the likelihood is a Gaussian factored across targets, with mean/variance from an MLP of $x_i$ and $r_C$. This is the **deterministic path**.

The **latent-variable NP** adds a global latent $z$ to capture functional uncertainty via a complementary **latent path**:

$$
p(y_T \mid x_T, x_C, y_C) := \int p(y_T \mid x_T, r_C, z)\, q(z\mid s_C)\, dz, \tag{2}
$$

where $z$ is a factorized Gaussian parametrized by $s_C := s(x_C, y_C)$ (a second permutation-invariant aggregator), and $q(z\mid s_\varnothing) := p(z)$ is the prior. **The path distinction — the conceptual crux of this shard.** The two paths do different jobs: the **latent path** carries a *global* latent $z$ whose single draw induces correlations across *all* target predictions — it models which realization of the stochastic process we are on (global structure); the **deterministic path** carries a *query-specific* representation that models fine-grained *local* structure (which context points are near this query). Kim et al. keep the latent path free of cross-attention *precisely to preserve the global latent's correlating role* — attention is added only to the deterministic path.

**The training objective — ELBO (Eq. 3).** With both paths, parameters are learned by maximizing the evidence lower bound

$$
\log p(y_T \mid x_T, x_C, y_C) \;\ge\; \mathbb{E}_{q(z\mid s_T)}\big[\log p(y_T\mid x_T, r_C, z)\big] \;-\; D_{\mathrm{KL}}\big(q(z\mid s_T)\,\|\,q(z\mid s_C)\big), \tag{3}
$$

optimized over random context $C$ and target $T$ subsets via the reparametrization trick. **Reading the two terms:** the first is a reconstruction term (decode targets from context summary $r_C$ and a latent drawn from the *target*-conditioned posterior $q(z\mid s_T)$); the second regularizes the *context*-summary distribution $q(z\mid s_C)$ toward the *target*-summary distribution $q(z\mid s_T)$ — sensible because contexts and targets come from the same stochastic-process realization (and $C\subset T$ in practice). Note this uses $q(z\mid s_C)$ in place of the usual fixed prior $p(z)$: the "prior" is itself conditioned on the context, exactly as in the latent CNP of §2. Maximum-likelihood learning here minimizes the KL between the (consistent) conditionals of the true data-generating process and the NP's conditionals — so the NP *approximates* a consistent process without being one (it violates context-consistency).

**Attention primitives (§2.2).** Given keys/values $K\in\mathbb{R}^{n\times d_k}, V\in\mathbb{R}^{n\times d_v}$ and queries $Q\in\mathbb{R}^{m\times d_k}$:

- **Laplace (kernel) attention** — parameter-free, keys/queries are the raw $x$-coordinates:
$$
\text{Laplace}(Q,K,V) := WV\in\mathbb{R}^{m\times d_v}, \quad W_{i\cdot} := \text{softmax}\big((-\lVert Q_{i\cdot} - K_{j\cdot}\rVert_1)_{j=1}^n\big).
$$
- **Scaled dot-product attention** — similarity in a learned representation space:
$$
\text{DotProduct}(Q,K,V) := \text{softmax}\!\big(QK^\top/\sqrt{d_k}\big)V \in\mathbb{R}^{m\times d_v}.
$$
- **Multihead attention** — per-head linear projections, dot-product, concatenate, project:
$$
\text{MultiHead}(Q,K,V) := \text{concat}(\text{head}_1,\dots,\text{head}_H)W, \quad \text{head}_h := \text{DotProduct}(QW_h^Q, KW_h^K, VW_h^V).
$$
All are permutation-invariant in the key-value pairs — the property that makes attention a legal drop-in for the NP's set aggregation.

**The ANP architecture (§3) — where attention enters each path.** Two insertions:
1. **Self-attention over context points**, applied *before* mean-aggregation in *both* the deterministic and latent paths, to model interactions among context points (e.g. if many contexts overlap, the query need not weight all of them). Higher-order interactions are modeled by *stacking* self-attention layers (à la Vaswani et al. 2017).
2. **Cross-attention in the deterministic path only**: the mean-aggregation producing $r_C$ is replaced by cross-attention where each target query $x_*$ attends to the context inputs $x_C$ to produce a **query-specific** representation $r_* := r_*(x_C, y_C, x_*)$. This is precisely the mechanism letting each query attend to the context points it deems relevant.

The decoder is unchanged except $r_C \to r_*$. ANP is trained with the *same* ELBO (Eq. 3), Gaussian likelihood $p(y_i\mid x_i, r_*(x_C,y_C,x_i), z)$ and diagonal Gaussian $q(z\mid s_C)$. **If attention is uniform** (all contexts equal weight) the ANP recovers the NP — so the NP is the degenerate uniform-attention special case. Permutation invariance in the contexts is preserved. Cost rises to $O(n(n+m))$ (self-attention across $n$ contexts, plus per-target weights over all contexts), but computations are matrix multiplications parallelizable across contexts and targets, so wall-clock stays competitive.

**Why cross-attention is kept out of the latent path — the derivation of the design choice.** If cross-attention were applied in the latent path, each target would get its own local latent, destroying the *global* latent $z$ that induces cross-target correlations (the whole point of the latent path: one $z$ sample = one coherent function realization). Keeping the latent path with a single global $z$ and giving the deterministic path query-specific attention cleanly separates global structure (latent path, correlations, sample diversity) from local structure (deterministic path, accurate context reconstruction).

**Empirical results.** (i) *1-D GP regression* with random kernel hyperparameters: ANP shows much faster drop in context-reconstruction error and lower target NLL than NP, in both iterations and wall-clock. Dot-product/multihead ANP nearly perfectly predict context points; NP underfits (learns large likelihood noise to explain data); Laplace attention behaves like NP because its similarity is L1 distance in raw $x$-space rather than a learned space. Multihead smooths the non-smooth predictions of raw dot-product. Merely enlarging the NP bottleneck $d$ helps only up to a limit and never matches multihead ANP (which needs 10% of the wall-clock). (ii) *2-D image regression* (MNIST, CelebA): Stacked-Multihead ANP reconstructions are nearly indistinguishable from originals vs. blurry NP; different $z$ samples give diverse-but-coherent faces/digits (evidence $z$ models global structure); heads specialize (one attends to the target pixel, one to a nearby region, one exploits face symmetry by looking at the mirror-image side). The model can map between resolutions ($4\times4 \to 32\times32$, even $32\times32 \to 256\times256$) despite training on a single resolution, since $x$ (pixel location) lives in a continuous space.

## Appendix: Section-by-Section Backbone

- **Abstract.** NPs learn to map a context set of input-output pairs to a distribution over regression functions, with linear complexity and a wide family of conditionals. But NPs **underfit** — inaccurate predictions at observed inputs. ANPs incorporate attention so each input location attends to relevant context points; greatly improves accuracy, speeds training, expands modelable functions.
- **§1 Introduction.** Regression as a distribution over functions (Bayesian view; GPs as non-parametric example). NPs: efficient, linear-cost, arbitrary context size; but underfit (Fig. 1: inaccurate means, over-large variances at context; blurry face reconstruction). Hypothesis: mean-aggregation bottleneck gives equal weight to all context points. Fix: GP-kernel-inspired differentiable attention preserving permutation invariance. Evaluate on 1-D and 2-D regression; ANPs improve reconstruction, speed, expressiveness.
- **§2 Background.** §2.1 Neural Processes (deterministic NP Eq. 1 with $r_C$; latent NP Eq. 2 with global $z$, factorized-Gaussian $q(z\mid s_C)$, prior $q(z\mid s_\varnothing)=p(z)$; both-path model most expressive; ELBO Eq. 3 with reparametrization; properties: scalability $O(n+m)$, flexibility, permutation invariance, but no context-consistency; NP approximates conditionals of a consistent process). §2.2 Attention (key-value + query weighting, permutation invariance; Laplace kernel attention parameter-free; scaled dot-product; multihead per-head projections).
- **§3 Attentive Neural Processes.** Self-attention over contexts (both paths, models context interactions, stackable for higher-order); cross-attention in deterministic path (mean-aggregation → query-specific $r_*(x_C,y_C,x_*)$); latent path keeps global $z$ (no cross-attention, preserves target correlations / global structure vs. local structure in deterministic path). Decoder unchanged ($r_C\to r_*$); same ELBO; uniform attention recovers NP; permutation invariance preserved; complexity $O(n(n+m))$ but parallelizable, wall-clock competitive.
- **§4 Experimental Results.** 1-D GP regression (random & fixed kernel hyperparameters; context reconstruction error and target NLL vs iterations/wall-clock; NP underfits, dot-product/multihead ANP accurate, Laplace ≈ NP; bottleneck-size $d$ sweep shows raising $d$ insufficient; toy Bayesian Optimization proof-of-concept). 2-D image regression (MNIST, CelebA; NP vs Multihead ANP vs Stacked-Multihead ANP; crisp reconstructions, diverse coherent $z$ samples, head-role visualization including symmetry-exploiting head; resolution mapping $4\times4\to32\times32\to256\times256$).
- **§5 Related Work.** GPs (kernel↔attention parallel; Deep Kernel Learning; VIP); Meta-Learning (few-shot classification with attention; neural statistician / variational homoencoder with local+global latents; Vfunc); Generative Query Networks as an NP special case (x=viewpoints, y=frames).
- **§6 Conclusion & Discussion.** ANPs cure underfitting, improve accuracy/speed/expressiveness. Future: cross-attention in latent path with local+global latents (neural-statistician-style for regression); ANPs on text (stochastic fill-in-the-blank); self-attention in the decoder → an Image-Transformer-like model over arbitrary pixel orderings (targets then affect each other, so ordering/grouping matters).
- **Appendices A–E.** A: architectural details (8 heads for multihead). B: 1-D experimental details. C: fixed-kernel results, dot-product non-smoothness explanation, Bayesian Optimization analysis. D: 2-D experimental details (self-attention stacking à la Parmar et al. 2018). E: additional image results, per-head attention visualizations, resolution-mapping figures.

---
