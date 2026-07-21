> **Per-paper review — in-context-learning corpus, paper 22 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§22); content is identical. Manifest: [[in_context_learning_sources]].

# 22. Mittal et al. 2025 — Amortized In-Context Bayesian Posterior Estimation

**PDF:** `docs/project/references/in_context_learning/sources/Mittal et al. 2025 - Amortized In-Context Bayesian Posterior Estimation.pdf`
**Status:** arXiv preprint (arXiv:2502.06601v1, 10 Feb 2025). Authors: Sarthak Mittal, Niels Leif Bracher, Guillaume Lajoie, Priyank Jaini, Marcus Brubaker (Mila / RPI / Google DeepMind / York / Vector).

## Phase 1: Foundational Overview (Undergraduate-Level)

**The problem in plain words.** Suppose you have a fixed probabilistic model (a likelihood + prior) and you keep getting new datasets — new poll results, new case counts, a new geographic region. Standard Bayesian tools (MCMC, VI) force you to re-run an expensive fit *from scratch* for every new dataset. **Amortization** is the trick of training a single neural network once so that, for any new dataset $\mathcal{D}$ handed to it as context, it *instantly* outputs an approximate posterior $q_\phi(\theta\mid\mathcal{D})$ — no re-fitting. Because the datasets are unordered bags of i.i.d. points, the network should be **permutation-invariant** (order of context examples must not matter).

**The core question.** Prior in-context work mostly modeled the *predictive* distribution $p(y^*\mid x^*, \mathcal{D})$ directly. This paper instead targets the **posterior over parameters** $p(\theta\mid\mathcal{D})$, and runs a careful, controlled *bake-off* over the two things that matter:
1. **Training objective:** forward KL (= neural posterior estimation / SBI) vs reverse KL (= the VI / neural-process style).
2. **Architecture & density:** GRU vs DeepSets vs Transformer for the set-conditioner; diagonal Gaussian vs normalizing flow for the output density.

**Key findings.**
- **Reverse KL + Transformer + normalizing flow is the overall winner** for predictive problems, especially in high dimensions.
- **Forward KL suffers from mode-averaging** — on high-dimensional multimodal problems (nonlinear regression/classification) it smears mass across modes and predicts poorly. Reverse KL is mode-seeking, which turns out to help predictively at high dimension.
- **Reverse KL has a crucial practical superpower:** it can be trained on *any* dataset distribution, including *real* data, because it does not need ground-truth $(\theta,\mathcal{D})$ pairs — only the joint density. Forward KL *must* train on simulated data from the assumed model, so it breaks under **misspecification** and **sim-to-real transfer**. Reverse KL degrades gracefully and improves further when trained directly on real data.
- **Surprising architecture result:** GRUs (not even permutation-invariant) often beat DeepSets (which are). The likely reason: DeepSets' fixed pooling operator limits expressivity; GRUs can *learn* approximate invariance. Transformers beat both.
- The models trained *only on synthetic data* generalize **zero-shot to real tabular OpenML tasks**, and provide excellent warm-start initializations that converge far faster than training from the prior.

**Initial takeaway.** This is the shard's systematic "how to amortize a posterior" study. Its central lesson — **reverse KL for robustness and high-dim prediction, forward KL for capturing low-dim multimodality** — directly informs the design axis that Reuter (forward-KL/flow-matching, distribution) and Mittal-Parametric (point-vs-distribution) sit on.

## Phase 2: Graduate-Level Deep Dive

### 2.1 Setup: from per-dataset optimization to amortization

Classic approximate inference solves, *for each* $\mathcal{D}$,
$$\phi^* = \arg\min_\phi\; D\big(p(\cdot\mid\mathcal{D}),\, q_\phi(\cdot)\big). \tag{4}$$
Reverse-KL VI instantiates $D$ as $D_{R\text{-}KL}(p,q_\phi) = \mathbb{E}_{\theta\sim q_\phi}[\log \frac{q_\phi(\theta)}{p(\theta\mid\mathcal{D})}]$, equivalently maximizing the ELBO. Forward-KL/EP instantiates $D_{F\text{-}KL} = \mathbb{E}_{\theta\sim p(\cdot\mid\mathcal{D})}[\log\frac{p(\theta\mid\mathcal{D})}{q_\phi(\theta)}]$.

**Amortization** replaces the per-dataset $q_\phi(\cdot)$ with a single **conditional** model $q_\phi(\cdot\mid\mathcal{D})$ trained across many datasets:
$$\phi^* = \arg\min_\phi\; \mathbb{E}_{\mathcal{D}\sim\chi}\, D\big(p(\cdot\mid\mathcal{D}),\, q_\phi(\cdot\mid\mathcal{D})\big), \tag{9}$$
where $\chi$ is a distribution over datasets. This is the "in-context posterior estimator."

### 2.2 The two objectives — derivations that expose the forward/reverse asymmetry

**Forward KL → neural posterior estimation.** When $\chi$ is exactly the assumed model's marginal, i.e.
$$\chi(\mathcal{D}) = \int p(\theta)\prod_{x_n\in\mathcal{D}} p(x_n\mid\theta)\, d\theta, \tag{10}$$
the forward-KL objective simplifies dramatically:
$$\phi^*_{F\text{-}KL} = \arg\min_\phi\; \mathbb{E}_{\mathcal{D}\sim\chi}\,\mathbb{E}_{\theta\sim p(\cdot\mid\mathcal{D})}\!\left[\log\frac{p(\theta\mid\mathcal{D})}{q_\phi(\theta\mid\mathcal{D})}\right] \tag{11}$$
$$= \arg\min_\phi\; \mathbb{E}_{\theta}\,\mathbb{E}_{\mathcal{D}\sim p(\cdot\mid\theta)}\!\left[-\log q_\phi(\theta\mid\mathcal{D})\right]. \tag{12}$$

*Derivation of the key step (11)→(12).* Drop the $\theta$-only term $\log p(\theta\mid\mathcal{D})$ (constant in $\phi$), leaving $\mathbb{E}_{\mathcal{D}\sim\chi}\mathbb{E}_{\theta\sim p(\theta\mid\mathcal{D})}[-\log q_\phi(\theta\mid\mathcal{D})]$. The nested expectation $\mathbb{E}_{\mathcal{D}\sim\chi}\mathbb{E}_{\theta\sim p(\theta\mid\mathcal{D})}[\,\cdot\,]$ is an expectation over the **joint** $p(\theta,\mathcal{D})$. By Bayes' factorization $p(\theta,\mathcal{D}) = p(\theta)\,p(\mathcal{D}\mid\theta)$, this reverses the sampling order to: draw $\theta\sim p(\theta)$, then $\mathcal{D}\sim p(\cdot\mid\theta)$ — i.e. **ancestral sampling from the simulator**. This is exactly the tower-property/joint-sampling trick (identical in spirit to Reuter's Proposition 1). It removes any need to sample or evaluate the true posterior — the hurdle that cripples classic Expectation Propagation. **But** the change of expectation is *only valid when $\mathcal{D}$ is sampled according to the assumed $p$.** That constraint is the Achilles heel: forward KL is *locked to simulated data from the correct model*.

**Reverse KL → amortized VI, free choice of $\chi$.**
$$\phi^*_{R\text{-}KL} = \arg\min_\phi\; \mathbb{E}_{\mathcal{D}\sim\chi}\,\mathbb{E}_{\theta\sim q_\phi(\cdot\mid\mathcal{D})}\!\left[\log\frac{q_\phi(\theta\mid\mathcal{D})}{p(\theta\mid\mathcal{D})}\right] = \arg\min_\phi\; \mathbb{E}_{\mathcal{D}\sim\chi}\,\mathbb{E}_{\theta\sim q_\phi(\cdot\mid\mathcal{D})}\!\left[\log\frac{q_\phi(\theta\mid\mathcal{D})}{p(\mathcal{D},\theta)}\right]. \tag{13,14}$$

*Why the second form is trainable.* The intractable normalizer $p(\mathcal{D})$ in $p(\theta\mid\mathcal{D}) = p(\mathcal{D},\theta)/p(\mathcal{D})$ contributes only an additive $\log p(\mathcal{D})$ constant in $\phi$, so it drops out of the gradient — leaving the *unnormalized joint* $p(\mathcal{D},\theta) = p(\theta)\prod_n p(x_n\mid\theta)$, which is evaluable pointwise. **Crucially, the outer expectation is over $\mathcal{D}\sim\chi$ with $\chi$ arbitrary:** because the inner expectation samples $\theta\sim q_\phi$ (the model itself), not from the posterior, *no ground-truth $(\theta,\mathcal{D})$ pairs are needed*. This is the asymmetry: reverse KL can consume real data whose true generative process is unknown/misspecified; forward KL cannot. (The gradient of the reverse-KL inner expectation w.r.t. $\phi$ is handled by reparametrization through $q_\phi$ — a diagonal Gaussian or a normalizing flow — since both admit pathwise gradients.)

### 2.3 Density and architecture choices

- **Density $q_\phi(\cdot\mid\mathcal{D})$:** diagonal Gaussian (cheap, unimodal) or **discrete-time normalizing flow** (expressive, multimodal-capable).
- **Set-conditioner:** GRU (sequential, *not* permutation-invariant), DeepSets (invariant via fixed pooling), Transformer (invariant via attention, no positional encoding, no fixed pool).
- **Variable feature dimensions:** low-dim problems are embedded into a fixed 100-D space by **masking** unused dimensions/parameters to zero (à la TabPFN), so a *single* estimator handles, e.g., both 1-D and 50-D nonlinear regression.

### 2.4 Findings that matter for the shard's design axis

- **Forward vs reverse KL (the central result):** forward KL better captures **low-dimensional multimodality** (GMM cluster labels visibly switch under forward KL, evidencing mode coverage; reverse KL locks to one labeling). But reverse KL **wins in high dimensions** on both predictive (L2/accuracy) and sample-based ($W_2$, symmetric KL) metrics, and dominates under **misspecification / sim-to-real** (Table 4: reverse KL L2 $\approx 0.35$ vs forward KL $\approx 8$–$15$ on the same OOD transfer; "+switched data" — training reverse KL directly on real data — improves further).
- **Capacity of $q_\phi$:** normalizing flows help forward KL **substantially** but reverse KL only **marginally** — because reverse KL's mode-seeking tendency latches onto a single mode even when given multimodal capacity; forward KL without flow capacity badly *overestimates variance*.
- **Architecture:** Transformer > GRU > DeepSets. DeepSets' fixed pooling caps expressivity; GRUs *learn* approximate invariance and beat DeepSets despite lacking the inductive bias.

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** Real-world motivation (polling, COVID compartment models, cryo-EM) where the probabilistic model is fixed but data streams in — ideal for amortization. Gap: no rigorous analysis of in-context *posterior-estimation* objectives. Contributions: (1) general framework with forward/reverse KL objectives; (2) benchmark of architectures × density parametrizations × objectives; (3) OOD/misspecification/sim-to-real evaluation.
- **§2 Background.** Bayesian inference (eqs 1–3: PPD as $\mathbb{E}_{\theta\mid\mathcal{D}}[p(x^*\mid\theta)]$, Monte Carlo estimate, Bayes rule). Approximate inference: sampling (MCMC, Langevin/HMC) vs variational (eq 4). Reverse-KL VI = ELBO (eqs 5–6); forward-KL = EP (eq 7). PPD via $q_{\phi^*}$ (eq 8). Estimators & amortization: VAE encoder as canonical amortizer; NPs and SBI-NPE amortize dataset-conditioned encoders under reverse/forward KL respectively; ICL viewed as amortizing the *predictive* $p(y^*\mid x^*,\mathcal{D})$.
- **§3 Posterior Estimation from Data in Context.** Reframes per-dataset VI/EP as amortized in-context estimation (eq 9). Forward KL ⇒ NPE under constraint (10), simplifying to joint-sampling ML (eqs 11–12). Reverse KL ⇒ free $\chi$ (eqs 13–14). Discusses permutation invariance requirement (DeepSets/Transformer yes, GRU no) and $\chi$ as black-box simulator via ancestral sampling.
- **§4 Experiments.** Tasks: Gaussian mean, GMM means, (non)linear regression, (non)linear classification. Baselines: prior (Random), MLE (Optimization), Langevin, HMC. Metrics: predictive (L2/accuracy, eq 15) and sample-based ($W_2^2$ eq 16, symmetric KL). §4.1 zero-shot posterior approximation (Tables 1: fixed-dim; rev-KL+flow+transformer best). §4.2 variable feature dims via masking (Table 2). §4.3 misspecification (rev-KL robust, Table 4). §4.4 tabular OpenML benchmarks (Table 3; MAP-finetuning warm-start, Figure 2). §4.5 posterior quality (symmetric KL Table 5, $W_2^2$ Table 6).
- **§5 Discussion/Conclusion.** Forward vs reverse KL trade-off (multimodal low-D vs high-D + robustness); architecture (GRU>DeepSets, Transformer best); capacity of $q_\phi$ (flow helps forward more than reverse).

---
