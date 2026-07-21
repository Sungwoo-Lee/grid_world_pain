> **Per-paper review — in-context-learning corpus, paper 8 of 15 (original batch).**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§8); content is identical. Manifest: [[in_context_learning_sources]].

# 8. Reuter et al. 2025 — Can Transformers Learn Full Bayesian Inference In Context?

**PDF:** `docs/project/references/in_context_learning/sources/Reuter et al. 2025 - Can Transformers Learn Full Bayesian Inference in Context.pdf`
**Venue:** ICML 2025 (PMLR 267). Authors: Arik Reuter, Tim G. J. Rudner, Vincent Fortuin, David Rügamer (LMU Munich / NYU / TU Munich / Helmholtz AI / MCML).

## Phase 1 — Foundational Overview (undergraduate level)

**The problem in plain words.** Bayesian inference is the machine that, after seeing data, tells you not just a single best guess for a model's hidden parameters but a *whole probability distribution* over them — capturing how uncertain you should be, and where multiple explanations are plausible. The gold-standard way to compute this (MCMC, e.g. Hamiltonian Monte Carlo) is slow: you re-run an expensive sampler from scratch every single time new data arrives. Variational inference (VI) is faster but forces you to pick a rigid family of shapes for the answer (often a Gaussian), and it tends to collapse onto one mode and misjudge the spread.

Classic "prior-data fitted networks" (PFNs, and their tabular cousin TabPFN) already showed that a transformer can *learn to do Bayesian prediction* by training only on synthetic data drawn from a chosen prior — but they only produce a *univariate, usually discretized predictive distribution* for one output value. They do not hand you the full, high-dimensional posterior over the latent parameters.

**The core idea.** Reuter et al. ask: can we train a transformer that ingests an *entire small dataset* $x$ in its context and, in one forward pass, spits out *samples from the full posterior* $P(z\mid x)$ over the latent variables $z$ — matching what an expensive HMC run would give? They answer yes, by bolting a **generative sampler onto a PFN-style encoder**. Specifically:
- A large **TabPFN-style transformer encoder** reads the dataset $x$ and produces a representation.
- A **diffusion-transformer decoder** trained by **flow matching** (a continuous normalizing flow) turns that representation into posterior samples $z \sim P(z\mid x)$.
- Training data is *purely synthetic*: you sample $z$ from a prior, then $x$ from the likelihood, and teach the network to invert that map. No real posterior is ever needed at training time.

**Key findings.**
- Across generalized linear models (GLMs), factor analysis (FA), and Gaussian mixture models (GMMs), the ICL sampler's posterior samples are **as close to HMC's as, or closer than, several state-of-the-art VI methods** — on both synthetic and 17 real-world tabular datasets.
- The advantage is largest exactly where VI struggles: **skewed posteriors** (e.g. gamma priors) and **multimodal posteriors** (GMMs), where VI shows textbook "mode-seeking" collapse and even flow-based VI (IAF) fails to capture bimodality from only 50 data points.
- **Flow matching is essential** — ablations replacing it with a plain Gaussian parametrization or a diffusion objective degrade the fit badly.
- **Inference is fast** (single forward pass + a cheap ODE solve) but the **up-front training cost is high**, and quality degrades under out-of-distribution shift and in high latent dimension ($K = 20, 50$).

**Initial takeaway.** This paper pushes ICL from "predict the next value" to "sample the full parameter posterior." It is the shard's clearest demonstration that a transformer, trained only on simulator draws, can *replace* MCMC/VI for full Bayesian inference on realistic models — a **distribution estimator** in the purest sense, and the natural counterpoint to the Mittal "point vs distribution" debate below.

## Phase 2 — Graduate-Level Deep Dive

### 2.1 The amortization objective and the tractability trick

The learning target is the map $f_0 : \mathcal{X} \to \mathcal{M}(\mathcal{Z})$, $x \mapsto P^{z\mid x}$, where $\mathcal{M}(\mathcal{Z})$ is the space of probability measures over the latent space. The model $f_\theta(x) = Q^{z\mid x}_\theta$ is trained to minimize the expected divergence

$$R_\theta \;=\; \mathbb{E}_{x\sim p(x)}\!\left[\, d\!\left(Q^{z\mid x}_\theta,\; P^{z\mid x}\right)\right]. \tag{1}$$

$R_\theta$ is intractable because $P^{z\mid x}$ is unknown. The paper's key enabling result rewrites it into an objective over the **joint** $P^{x,z}$, from which we *can* sample:

$$\widetilde{R}_\theta \;=\; \mathbb{E}_{x,z\sim p(x,z)}\!\left[\, \mathcal{L}_d(x,z,\theta)\right]. \tag{2}$$

**Proposition 1 (equivalence condition).** If the divergence has the linear-in-$P$ form
$d(Q^{z\mid x}_\theta, P^{z\mid x}) = \int \gamma(Q^{z\mid x}_\theta)\, dP^{z\mid x}$
for some measurable functional $\gamma:\mathcal{M}(\mathcal{Z})\to\mathbb{R}$, then $R_\theta = \widetilde{R}_\theta$ with $\mathcal{L}_d(x,z,\theta) = \gamma(Q^{z\mid x}_\theta)$.

*Derivation sketch.* Starting from (1) and substituting the integral form of $d$:
$$R_\theta = \mathbb{E}_{x}\!\left[\int \gamma(Q^{z\mid x}_\theta)\, dP^{z\mid x}\right] = \mathbb{E}_{x}\,\mathbb{E}_{z\mid x}\!\left[\gamma(Q^{z\mid x}_\theta)\right] = \mathbb{E}_{x,z\sim p(x,z)}\!\left[\gamma(Q^{z\mid x}_\theta)\right],$$
where the middle step is the definition of conditional expectation and the last is the **law of total expectation** — the tower property collapses $\mathbb{E}_x \mathbb{E}_{z\mid x}$ into a single expectation over the joint. This is exactly the same tractability trick PFNs use: it is why training on simulator draws $(x,z)\sim P^{x,z}$ is equivalent to matching the true posterior.

**Worked instance — forward KL recovers max-likelihood.** Choosing $d$ to be the forward KL $D_{KL}[p(\cdot\mid x)\,\|\,q_\theta(\cdot\mid x)]$ gives $\gamma$ such that
$$\mathcal{L}_{d_{KL}}(x,z,\theta) = -\log q_\theta(z\mid x) + \text{const}.$$
So minimizing $\widetilde{R}_\theta$ under forward KL is **maximum-likelihood training of $q_\theta$ on joint samples** — precisely the PFN/NPE (neural posterior estimation) objective. This is the anchor connecting Reuter to the forward-KL branch in both Mittal papers.

### 2.2 Flow matching as the posterior parametrization

Rather than a tractable density $q_\theta(z\mid x)$, Reuter parametrizes $Q^{z\mid x}_\theta$ implicitly as a **continuous normalizing flow (CNF)**. A base $P_B = \mathcal{N}(0,I)$ is pushed forward by a conditional flow $\psi_\theta(\cdot\mid x)$:

$$Q^{z\mid x}_\theta := [\psi_\theta(\cdot\mid x)]_\sharp P_B, \qquad P^{z\mid x} \approx [\psi_\theta(\cdot\mid x)]_\sharp P_B. \tag{3}$$

The flow is defined by an ODE driven by a learned time-dependent vector field $v^\theta_{t,x}$:

$$\frac{d}{dt}\psi_{\theta,t}(z\mid x) = v^\theta_{t,x}\!\big(\psi_{\theta,t}(z\mid x)\big), \qquad \psi_{\theta,0}(z\mid x)=z, \quad 0\le t\le 1. \tag{4}$$

The initial condition makes the flow the identity at $t=0$; integrating to $t=1$ transports $P_B$ onto the posterior.

**Flow-matching loss.** Under Gaussian conditional probability paths with an optimal-transport mean/variance schedule, the discrepancy becomes

$$d_{CFM}\!\left(Q^{z\mid x}_\theta, P^{z\mid x}\right) = \mathbb{E}\!\left[\,\big\| v^\theta_{t,x}(\gamma_t(z^{(1)}\mid z^{(0)})) - (z^{(1)} - \omega z^{(0)})\big\|_2^2\,\right], \tag{5}$$

where the expectation is over $t\sim U([0,1])$, $z^{(0)}\sim P_B$, $z^{(1)}\sim P^{z\mid x}$, the interpolant is $\gamma_t(z^{(1)}\mid z^{(0)}) := (1-\omega t)z^{(0)} + t z^{(1)}$, and $\omega = 1-\sigma_{\min}$ (they set $\omega = 1-10^{-4}$). Because $d_{CFM}$ is an expectation of a per-sample loss, it satisfies Proposition 1's form, so the empirical training objective over $N$ i.i.d. joint draws is simply

$$\widehat{R}_\theta = \sum_{i=1}^N \big\| v^\theta_{t_i,x_i}(\gamma_{t_i}(z^{(1)}_i\mid z^{(0)}_i)) + z^{(1)}_i - \omega z^{(0)}_i\big\|_2^2. \tag{7}$$

At **sampling** time: draw $z^{(0)}\sim P_B$, encode $x$, then hand the decoder-defined field $(t,\nu)\mapsto v^\theta_{t,x}(\nu)$ to a numerical ODE solver integrating $0\to 1$.

*Why CNF over discrete normalizing flows or a Gaussian?* (i) CNFs place **no architectural constraint** (no invertibility/triangular-Jacobian requirement), so complex data conditioning via cross-attention is free; (ii) flow matching is **simulation-free** and more sample-efficient than diffusion; (iii) it can represent essentially arbitrary posterior shapes — the crucial property for the skewed/multimodal cases where VI fails.

### 2.3 Architecture and the point-vs-distribution positioning

The architecture couples a **TabPFN-style encoder** (reads the dataset $x$) with a **diffusion-transformer decoder** where time $t$ enters via adaptive layer-norm (adaLN) blocks initialized to identity, plus cross-attention onto the encoder output. The decoder input $(1-\omega t)z^{(0)} + t z^{(1)}$ is treated as a length-one sequence; effectively an MLP-with-conditioning + cross-attention. A separate model is trained per model-scenario (7 GLM, 6 FA, 4 GMM).

**Where this sits on the shard's central axis.** Reuter is an unambiguous **full-distribution estimator** and shows it *wins* against VI. This is in productive tension with Mittal (Point or Distribution), which finds point estimators usually win *for downstream prediction* on higher-dimensional problems. The reconciliation: (a) Reuter evaluates *posterior-sample fidelity* against HMC (C2ST / MMD / $W_2$), not just predictive loss; (b) Reuter's latent dimensions where ICL wins are modest ($K$ small) and its own high-$K$ ablation ($K=20,50$) shows the advantage *evaporates* — consistent with Mittal's high-dimensional finding. The two papers are measuring different things (fidelity vs prediction) on different regimes.

### 2.4 Evaluation and results

Three sample-based metrics vs the gold standard (analytic where available, else HMC-NUTS): **C2ST** (classifier 2-sample test ROC-AUC; 0.5 = indistinguishable, 1.0 = perfectly separable), **MMD** (maximum mean discrepancy), and **$W_2$** (empirical 2-Wasserstein). Baselines: Laplace approximation and ADVI-style VI with diagonal / full / structured Gaussian and IAF variational families.

Headline numbers (lower is better): on GLMs, ICL C2ST $0.657$ (synthetic) vs best VI $0.711$; on FA, ICL C2ST $0.568$ — remarkably near the $0.5$ floor — vs VI $\ge 0.987$; on the hardest GMMs, ICL $\approx 0.825$ C2ST vs VI $\approx 0.99$ (saturated). MMD and $W_2$ track the same ordering. Qualitatively, VI exhibits mode-seeking collapse on bimodal GMM posteriors while ICL matches HMC's marginals.

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** Motivates ICL for *full* (high-dimensional, continuous) posteriors vs PFNs' univariate predictive posteriors. Frames two pains of classic full-Bayes: slow sampling, and misspecification from rigid VI families. Five contributions: (1) a model sampling $P(z\mid x)$ with no parameter updates/parametric posterior assumptions; (2) training on synthetic joint draws $P^{x,z}$ + a general analysis framework; (3) instantiation for GLMs, GMMs, FA; (4) fidelity comparable/superior to HMC and better than VI; (5) ablations (diffusion vs flow matching, Gaussian parametrization, OOD, dimensionality).
- **§2 Related Work.** Three lenses: **ICL** (Garg et al. 2022 function classes; contrast: their work is scalar/small-scale/simulated-only, this is multivariate posteriors on real data, *unsupervised*); **Amortized inference** (VAEs/NPs amortize per-datapoint $q_\theta(z_j\mid h_\phi(x_j))$; this work amortizes *per-dataset*, a functional map over a meta-dataset $\mathcal{D}\subset(\mathcal{X}\times\mathcal{Z})^N$; no ELBO/KL used, unlike AVI); **Simulation-based inference** (SBI/NPE via normalizing flows, flow matching, transformer diffusion on the joint). Explicitly cites concurrent Mittal 2025a (amortized posterior benchmark) and 2025b (point vs distribution) — noting *they* evaluate posterior mean / predictive performance, whereas Reuter evaluates *full posteriors*.
- **§3 ICL for Full Bayesian Inference.** §3 (objective + Proposition 1, the joint-sampling tractability result). §3.1 Defining the posterior form (normalizing flows §3.1.1, CNFs + flow-matching loss §3.1.2, eqs. 3–7). §3.2 Sampling from the joint (ancestral: $z\sim P^z$ then $x\sim P^{x\mid z}$). §3.3 Architecture (TabPFN encoder + diffusion-transformer decoder, adaLN, cross-attention). §3.4 Implementing flow matching (train vs sample loops, ODE solve).
- **§4 Experiments.** Modeling scenarios (7 GLM, 4 FA, 4 GMM). 50 synthetic + 17 real-world tabular datasets (Grinsztajn suite). Metrics C2ST/MMD/$W_2$ vs analytic-or-HMC. §4.1 GLMs (ICL best, esp. skewed gamma-prior). §4.2 FA (ICL C2ST 0.568, near floor). §4.3 GMMs (hardest — discrete assignments, high-dim, non-identifiable/multimodal; ICL still best, VI mode-seeks). §4.4 Ablations: flow matching essential; dimensionality (advantage vanishes at $K=20,50$); OOD degrades with shift; predictive performance (VI competitive on point prediction, ICL competitive); MLP vs transformer encoder (transformer wins); C2ST classifier choice validated.
- **§5 Discussion.** Contributions recap. Limitations: heavy up-front training; focus on simple posteriors with tractable references; high-dim challenges both method and metrics; large contexts expensive. Outlook: any model with a conceivable generative process can be fit — potential beyond standard Bayesian methods.

---
