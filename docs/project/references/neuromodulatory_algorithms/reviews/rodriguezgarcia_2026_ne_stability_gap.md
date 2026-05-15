---
title: "Noradrenergic-inspired gain modulation attenuates the stability gap in joint training"
authors:
  - Alejandro Rodriguez-Garcia
  - Anindya Ghosh
  - Srikanth Ramaswamy
year: 2026
venue: "arXiv:2507.14056 (v2, 27 Jan 2026)"
slug: rodriguezgarcia_2026_ne_stability_gap
source_pdf: "sources/Rodriguez-Garcia et al. 2026 - Noradrenergic-inspired gain modulation attenuates the stability gap in joint training.pdf"
topic: neuromodulatory_algorithms
---

# Rodriguez-Garcia, Ghosh & Ramaswamy (2026) — Noradrenergic-inspired gain modulation attenuates the stability gap in joint training

## Plain-English entry point

When a deep network is trained on a sequence of tasks, it usually **forgets** earlier tasks once it sees later ones — this is called *catastrophic forgetting*. A more recently identified, subtler problem is the **stability gap**: even when forgetting is, in principle, fixable (because all old data is still available during *joint training*), the network's accuracy on previously learned tasks **transiently crashes** the moment a new task begins, and only recovers a few hundred iterations later. That brief crash matters in safety-critical deployments: a self-driving car or medical model that is being continually updated cannot afford a momentary 20–40% accuracy drop.

This paper claims the stability gap is not a memory problem — it is an **optimisation problem** — and proposes a fix inspired by **noradrenaline**, a neuromodulator the brain releases in bursts when it encounters surprise or unexpected uncertainty. In the brain, a noradrenergic burst transiently raises the **gain** of cortical neurons — the slope of their input-output transfer function — so they respond more strongly to new sensory evidence without changing what they are selective for. Rodriguez-Garcia et al. import that idea into deep learning as **NGM-SGD (Noradrenergic Gain-Modulated SGD)**: every neuron carries a multiplicative gain $g(t)$ that is boosted whenever the softmax output entropy (the model's uncertainty about the current sample) spikes, and decays back to a baseline $g_0$ otherwise.

The authors prove two things mathematically — (1) this gain-then-decay mechanism is equivalent to a **fast/slow two-timescale optimizer** (Hinton & Plaut 1987's fast-weights idea); (2) multiplying weights by $g$ before the forward pass reparameterises the loss landscape so that local curvature is divided by $g^2$ — i.e. *gain boosts flatten the loss surface around task transitions*. Empirically, on five standard continual-learning benchmarks (Split-MNIST, Split-CIFAR-10, Split-mini-ImageNet, Rotated-MNIST, Domain-CIFAR-100) NGM-SGD reduces the stability gap by a factor of ~2 to 15 compared to Adam, momentum-SGD, and vanilla-SGD, while keeping final accuracy competitive. This paper is the corpus's most direct *algorithmic* instantiation of the "neuromodulator as gain controller" theme.

## Section-ordered backbone

**1. Introduction.** Continual learning (CL) tries to acquire new tasks without overwriting old ones. A recently identified failure mode — the **stability gap** (De Lange et al. 2022; Hess et al. 2023) — is a *transient* accuracy drop on previously learned tasks at every task boundary, **persisting even under perfect joint training** where $\mathcal{L}_{\text{joint}} = \mathcal{L}_{\text{new}} + \mathcal{L}_{\text{old}}$ is fully available. This isolates the gap as a property of *the dynamics of optimisation*, not of memory or representational drift. Standard optimisers (momentum-SGD, Adam) implicitly carry multi-timescale dynamics yet still exhibit the gap because their momentum/inertia terms overshoot at distribution shifts. The paper proposes a fix grounded in noradrenergic gain modulation: phasic NA bursts under unexpected uncertainty raise neural gain (the slope of $f(I)$), enabling rapid sensory integration without altering selectivity (Aston-Jones & Cohen 2005; Ferguson & Cardin 2020). The hypothesis is that transient gain boosts give rise to a *fast–slow* weight decomposition and *flatten the local loss landscape*, attenuating the stability gap.

**1.1 Related work.**
- **Stability gap in CL.** *Continual evaluation* (De Lange et al. 2022) — evaluating the model every $\rho$ iterations — exposed gaps invisible to task-end evaluation. The gap persists in LLMs (Guo et al. 2024) and even under perfect joint training (Hess et al. 2023; Kamath et al. 2024). Existing mitigations (Lapacz et al. 2024 — head-only forgetting; Harun & Kanan 2023; Soutif-Cormerais et al. 2023 — weight-averaging) add architectural cost or require task-specific details.
- **Gain modulation.** Defined as the slope of a neuron's input-output curve (Ferguson & Cardin 2020). Used for attentional gating (Caroline Haimerl et al. 2022), adaptive whitening (Duong et al. 2023a,b), motor primitives (Stroud et al. 2018). Theoretical claim from Munn et al. 2021/2023, Shine 2019: NA flattens the energy landscape, ACh deepens it. Wainstein et al. 2025 showed uncertainty-driven gain modulation produces high-velocity perceptual switches in RNNs — but only modulated gains post-hoc. This paper extends to **gain dynamics during learning**.

**2. Results.**

- **2.1 Gain boosts as a multi-timescale optimizer.** For a network $y(t) = h(x(t); g(t), w(t))$ with effective weight $W_{ij}(t) = g_i(t) w_{ij}(t)$, decompose into phasic burst $\Delta g(t) = g_i(t) - g_0$ and tonic baseline $g_0$: $W_{ij}(t) = g_0 w_{ij}(t) + [g_i(t) - g_0] w_{ij}(t) = w^{\text{slow}} + w^{\text{fast}}$. The fast component exists only while gain is elevated (after a surprise), decays exponentially, and is exactly the fast-weights structure of Hinton & Plaut 1987. Proof of concept in 1-D linear model shows orange (gain-modulated) trajectory matches blue (explicit fast-slow) trajectory while purple (slow only) lags.
- **2.2 Noradrenergic uncertainty.** Uncertainty quantified as softmax-output entropy $H(y) = \sum_i \pi_i \log \pi_i$ (Wainstein et al. 2025). Gain update: $g(t+1) = \gamma g(t) + (1-\gamma) g_0 + \eta H(y)$ — exponential decay back to $g_0$ with phasic forcing from entropy.
- **Algorithm 1: NGM-SGD.** Per iteration: forward pass with $\hat W = gW$ → cross-entropy loss → SGD update $W \leftarrow W - \alpha \nabla_W \mathcal{L}$ → compute entropy → gain update.
- **2.3 Online task-agnostic CL setup.** Joint training: model retains access to *all* past data; tasks presented in online stream with fixed iterations $I_C$ per context. This isolates the stability gap from other forgetting sources.
- **2.4 NGM-SGD reduces stability gaps.** Across Split-MNIST, Split-CIFAR-10, Split-mini-ImageNet (class-incremental) and Rotated-MNIST, Domain-CIFAR-100 (domain-incremental), NGM-SGD's average stability gap (avg-SG) is the lowest by ~2× to 15× over Adam / MSGD. Largest gaps occur at $T_1 \to T_2$ when distribution doubles; MSGD's gap is 1.0 (catastrophic) vs. NGM-SGD's 0.028 on Split-MNIST. Vanilla SGD is competitive in low-noise regimes because it stays closest to the true gradient.
- **2.5 Gain boosts reduce test loss at task boundaries.** Forward-pass reparameterisation flattens effective Hessian. Unlike Adam/MSGD, NGM-SGD follows the true gradient (gain rescales magnitude, not direction), so it does not deviate from the non-increasing-loss valley.
- **2.6 Gain encodes task complexity.** Asymptotic gain level differs between class-incremental (rising baseline as each task adds new classes) and domain-incremental (stable baseline, complexity already captured by task 1) — gain is a proxy for accumulated cognitive demand.

**3. Discussion.** Paradigm shift from *what* to optimise (loss objective) to *how* to optimise (the trajectory through parameter space). Two effects: (i) implicit fast-slow timescale separation, (ii) forward-pass reparameterisation flattening the loss. Limitations: small task sets (deliberately, to isolate the gap), MLP / ResNet-18 backbones. Future: cholinergic gain-gating for sparse task segregation; combination with replay.

**4. Methods.** Datasets: Split-MNIST (5 binary tasks), Split-CIFAR-10 (5 binary tasks), Split-mini-ImageNet (5 × 10-class tasks), Rotated-MNIST ($0°, 80°, 160°$), Domain-CIFAR-100 (20 superclasses split across 3 tasks). FFNN (2 × 400 hidden, ReLU, no biases) for MNIST; slim ResNet-18 for CIFAR/ImageNet, gain on output layer only. Hyperparameters: $\gamma = 0.9$, $\eta \in \{0.5, 0.4, 0.2, 0.1, 0.1\}$ (Rot-MNIST, Split-MNIST, Split-CIFAR-10, Domain-CIFAR-100, Split-mini-ImageNet), batch 128 (MNIST) / 256 (rest), $I_C = 200$–$800$. 5 seeds per experiment.

**Appendix B (load-bearing for Phase 2).** Proves gain reparameterisation flattens curvature: for $\tilde{\mathcal{L}}(W) = \mathcal{L}(GW)$ with $G = gI$, the Hessian satisfies $H_W = g^2 H_{W_{\text{eff}}}$, so eigenvalues in effective-parameter space transform as $\lambda^{\text{eff}}_i = \lambda_i / g^2$.

## Phase 1 — Undergraduate-level synthesis

The story is short.

When you train a neural network on Task A then start training on Task B, two bad things can happen. The famous one is **catastrophic forgetting** — Task A accuracy stays low afterwards. The newer, sneakier one is the **stability gap** — even if Task A data is still in the mix (joint training), the network's Task A accuracy *briefly* tanks the moment Task B starts, and then climbs back up over a few hundred iterations. For a self-driving car or a hospital model that gets updated weekly, that brief tank is the problem.

Why does it happen? Modern optimisers like Adam and momentum-SGD carry **momentum** terms — basically the model keeps moving in a direction smoothed over recent gradients. When the data distribution suddenly shifts, that momentum points the *wrong way* and the model overshoots into a region of bad Task A performance.

The biology fix: when your brain is suddenly surprised, the locus coeruleus dumps **noradrenaline** across the cortex. NA does not change *what* neurons respond to; it changes *how strongly* they respond — it cranks up the gain. After the surprise is integrated, NA decays and gain returns to baseline.

The algorithm: give each neuron a gain $g(t)$ that is **boosted by how uncertain the model is** (measured by softmax entropy on the current batch) and decays exponentially back to a baseline $g_0$. Train with normal SGD on the *effective* weight $\hat W = g W$. Two consequences:

1. **Implicit fast-slow weights.** When $g$ spikes, the effective weight has an extra additive "fast" component that helps the model lock onto the new task; when $g$ decays, only the stable "slow" weights remain.
2. **Loss landscape flattening.** Multiplying weights by $g$ in the forward pass divides the local Hessian eigenvalues by $g^2$. A flatter loss means smaller gradient overshoots — the trajectory through parameter space is gentler.

Tested on five benchmarks (Split-MNIST, Split-CIFAR-10, Split-mini-ImageNet, Rotated-MNIST, Domain-CIFAR-100), the average stability-gap drops from ~30–40% (Adam, MSGD) to ~3–15%, with final accuracy unchanged or slightly higher.

This is a *drop-in optimiser change*. No new architecture, no extra memory buffer, no task labels. It is also compatible with replay and regularisation methods.

## Phase 2 — Graduate-level deep dive

### The stability gap, formally

Given a sequence of tasks $\{T_k\}_{k=1}^{K}$ with distributions $\{\mathcal{D}_k\}$, the per-task **stability gap** at transition $k - 1 \to k$ is the relative drop in old-task accuracy:

$$
\mathrm{SG}_k \;=\; \frac{A\bigl(\mathcal{D}_{k-1},\, F_{\theta_f}^{(k-1)}\bigr) \;-\; \min\limits_{i \in I_C} A\bigl(\mathcal{D}_{k-1},\, F_{\theta_i}^{(k)}\bigr)}{A\bigl(\mathcal{D}_{k-1},\, F_{\theta_f}^{(k-1)}\bigr)},
$$

and average stability gap $\mathrm{avg\text{-}SG} = \tfrac{1}{K-1}\sum_{k=2}^{K}\mathrm{SG}_k$. Hess et al. 2023 and Kamath et al. 2024 establish empirically that $\mathrm{SG}_k > 0$ even under perfect joint training, isolating it as an **optimisation-dynamics** artefact.

### The noradrenergic-inspired gain equation

The paper's core gain dynamics (Eq. 4 of the paper):

$$
\boxed{ \quad g(t+1) \;=\; \gamma\, g(t) \;+\; (1 - \gamma)\, g_0 \;+\; \eta\, H\!\bigl(y(t)\bigr) \quad }
$$

with
- $\gamma \in (0, 1)$ the gain-decay constant ($\gamma = 0.9$ in all experiments),
- $g_0 \ge 1$ the tonic baseline (set to $1$ everywhere),
- $\eta > 0$ the phasic-burst scale (dataset-specific, $0.1$–$0.5$),
- $H(y)$ the softmax-output **entropy** as the uncertainty proxy:

$$
H(y) \;=\; \sum_{i} \pi_i(y) \,\log \pi_i(y),
$$

(note the sign convention in the paper makes $H$ negative; the practical update uses $|H|$ or shifted). When the model is confident ($\pi$ near a one-hot), $H \to 0$ and $g$ relaxes; when the model is surprised by a new-task sample, $H$ is large and $g$ jumps.

### Gain modulation as fast–slow weight decomposition

Define the **effective weight** $W_{ij}(t) = g_i(t)\, w_{ij}(t)$. Decompose:

$$
W_{ij}(t) \;=\; g_0\, w_{ij}(t) \;+\; \bigl[g_i(t) - g_0\bigr]\, w_{ij}(t) \;\equiv\; w_{ij}^{\text{slow}}(t) \;+\; w_{ij}^{\text{fast}}(t),
$$

with $w_{ij}^{\text{slow}}(t) = g_0\, w_{ij}(t)$ tracking the tonic synaptic state and $w_{ij}^{\text{fast}}(t) = [g_i(t) - g_0]\, w_{ij}(t)$ existing only during phasic NA bursts. Because $g_i(t) \to g_0$ exponentially, $w^{\text{fast}}$ decays to zero — exactly the fast-weights idea of Hinton & Plaut 1987.

Crucially, **the weight gradient with respect to $w$ scales with $g$**:

$$
\frac{\partial \mathcal{L}}{\partial w_{ij}} \;=\; g_i(t) \cdot \frac{\partial \mathcal{L}}{\partial W_{ij}}.
$$

So during phasic bursts, the *plasticity itself* is amplified — neurons that are currently most uncertain accumulate larger weight updates. After confidence returns, $g$ relaxes and the slower consolidated $w$ remains.

### Loss-landscape flattening — the Hessian congruence proof

The paper's load-bearing theoretical claim (Appendix B). Define $\tilde{\mathcal{L}}(W) = \mathcal{L}(\Phi(W))$ with linear reparameterisation $\Phi(W) = G\, W$, $G = gI_n$. Two lemmas:

**Lemma 1 (Gradient transformation).** $\nabla_W \tilde{\mathcal{L}}(W) = G^{\top}\, \nabla_{W_{\text{eff}}} \mathcal{L}(\Phi(W)) = g\, \nabla_{W_{\text{eff}}} \mathcal{L}(W_{\text{eff}})$. Chain rule.

**Lemma 2 (Hessian congruence).** Via second-order chain rule,

$$
\nabla^{2}_{W} \tilde{\mathcal{L}}(W) \;=\; G^{\top}\, \nabla^{2}_{W_{\text{eff}}} \mathcal{L}(\Phi(W))\, G \;=\; g^{2}\, H_{W_{\text{eff}}}(W_{\text{eff}}).
$$

**Corollary (eigenvalue rescaling).** If $\{\lambda_i\}$ are the eigenvalues of $H_W(W^{\star})$ and $\{\lambda_i^{\text{eff}}\}$ those of $H_{W_{\text{eff}}}(W_{\text{eff}}^{\star})$, then for isotropic $G = g I_n$:

$$
\boxed{ \quad \lambda^{\text{eff}}_{i} \;=\; \frac{\lambda_i(W^{\star})}{g^{2}} \quad }
$$

So **raising the gain by a factor $g$ divides the effective-parameter-space curvature by $g^2$** — quadratic flattening. At a task boundary when entropy spikes and $g$ briefly jumps from $1$ to, say, $1.4$, the local curvature drops by $\sim 2\times$, the gradient step traverses a much gentler valley, and the model integrates the new distribution without overshooting Task A's optimum.

### The full NGM-SGD algorithm

From the paper's Algorithm 1:

```
Inputs: context sequence C={c_k}, iterations I_C, batch B, lr α, decay γ, baseline g_0, scale η

Initialise W <- W_init, g <- g_init

for each context c_i in C:
  for iteration i = 1 .. I_C:
    (X, Y) ~ batch from D_k
    pi = softmax(F_W(X; g))                   # forward with effective weights g*W
    L  = (1/B) * sum_j cross_entropy(pi_j, Y_j)
    W  <- W - α * grad_W L                    # standard SGD on W (NOT W_eff)
    H  = -(1/B) * sum_j sum_l pi_{j,l} log pi_{j,l}
    g  <- γ*g + (1-γ)*g_0 + η * H              # gain update
```

Notes:
- Gradient is taken w.r.t. $W$, **not** $\hat W = gW$ — the gain only enters the forward pass.
- Entropy $H$ is computed per-batch as the mean predictive-distribution entropy.
- For CNN backbones, gain is restricted to the **output (classifier) layer** to avoid disrupting shared convolutional feature maps (consistent with Lapacz et al. 2024 finding that the classification head dominates stability-gap behaviour).

### Empirical findings — main numbers (paper Table 1)

| Benchmark | Metric | NGM-SGD | MSGD | Adam | SGD |
|---|---|---|---|---|---|
| Split-MNIST | avg-SG ↓ | **0.017** | 0.267 | 0.283 | 0.034 |
| Split-CIFAR-10 | avg-SG ↓ | **0.134** | 0.425 | 0.310 | 0.241 |
| Split-mini-ImageNet | avg-SG ↓ | **0.300** | 0.409 | 0.384 | 0.332 |
| Rotated-MNIST | avg-SG ↓ | 0.054 | 0.117 | 0.092 | **0.053** |
| Domain-CIFAR-100 | avg-SG ↓ | **0.224** | 0.232 | 0.256 | 0.243 |

NGM-SGD also matches or beats baselines on avg-ACC (final accuracy), avg-min-ACC, and WC-ACC. Per-transition gaps (paper Table 2) confirm the largest improvement is at $T_1 \to T_2$: Split-MNIST $\mathrm{SG}_1$ drops from MSGD's catastrophic 1.000 to NGM-SGD's 0.028.

### Ablation: gain dynamics are necessary

Appendix C ablates the gain dynamics, keeping only $g = g_0$. The stability gap appears to improve, but for the wrong reason — without phasic bursts, only the slow component $w^{\text{slow}} = g_0\, w$ contributes to learning, so the network learns *very* slowly and Task A is never disrupted because Task B is barely acquired. NGM-SGD's reduction of the gap is therefore not a stability-via-stagnation artefact: it requires the fast component to be present.

### Computational mechanism summary

NGM-SGD is a four-line modification to vanilla SGD that:

1. Maintains a per-neuron gain scalar $g_i(t)$ updated by exponential decay + entropy-driven phasic boost.
2. Forward-passes with $\hat W = g \cdot W$, leaving the parameter gradient w.r.t. $W$ unchanged in direction.
3. Implicitly decomposes the effective weight into a fast (decaying) and slow (consolidated) component.
4. Quadratically flattens the local loss curvature at task boundaries via the Hessian-congruence identity $H_W = g^2 H_{W_{\text{eff}}}$.

The result is a biologically grounded, drop-in optimiser change that attenuates the optimisation-dynamics-induced stability gap, without buffer storage, task labels, or architectural growth. It is compatible with replay (paper Appendix H).

## Connections to other papers in this corpus

- **Direct gain-modulation lineage.** This paper sits at the centre of the corpus's *gain-modulation* sub-cluster. It cites and builds on:
  - **Ferguson & Cardin 2020** (`ferguson_cardin_2020_gain_modulation`, other batch) — the canonical cortical-gain-modulation review providing the biological definition of gain as input-output slope.
  - **Wainstein et al. 2025** (`wainstein_2025_gain_perceptual_switches`, other batch) — pupillometry + fMRI + RNN evidence that uncertainty-driven gain modulation drives perceptual switches; this paper *extends* Wainstein's RNN result to learning dynamics.
  - **Shine 2019** and **Shine et al. 2021** (`shine_2021_cellular_to_dynamics`, other batch) — the macro-scale claim that NA flattens and ACh deepens the energy landscape.
  - **Aston-Jones & Cohen 2005** — the locus-coeruleus / NA / adaptive-gain theory.
- **Conceptual frame from Kudithipudi 2022.** The paper opens by citing Kudithipudi et al. 2022 (`kudithipudi_2022_lifelong_learning`, this batch) as the canonical lifelong-learning survey. This paper instantiates the survey's "neuromodulation → overcome forgetting" cell of Figure 10 with a concrete algorithm.
- **Durstewitz 2025 companion.** Durstewitz et al. 2025 (`durstewitz_2025_neuroscience_continual_learning`, this batch) is the dynamical-systems-flavoured complementary perspective — both papers argue continual learning needs to look at *how* you optimise, not just what. The "loss flattening" effect proven here is one concrete way to operationalise Durstewitz's "near-bifurcation" framing.
- **Mei et al. 2025** — co-authored by Rodriguez-Garcia, cited as `Mei et al. 2025` in the paper (the multi-neuromodulatory-dynamics paper); maps onto Mei et al. 2022 (`mei_2022_multiscale_neuromodulation`, other batch) in this corpus.
- **Fast-slow weights / multi-timescale.** Hinton & Plaut 1987 (fast weights) is the foundational reference; Stroud et al. 2018 (motor primitives via targeted gain modulation) appears in the related-work section.
- **Other neuromodulation-in-DL papers.** Vecoven et al. 2020 (`vecoven_2020_neuromodulation_deep_nets`, other batch), Mei et al. 2022 (`mei_2022_multiscale_neuromodulation`, other batch), Wang et al. 2024 (`wang_2024_neuromodulated_meta_learning`, other batch), Lee et al. 2024 (`lee_2024_lifelong_rl_neuromodulation`, other batch), Ben-Iwhiwhu et al. 2022 (`ben-iwhiwhu_2022_context_meta_rl`, other batch), Tsuda et al. 2021 (`tsuda_2021_hypertube_shifts`, other batch), Costacurta et al. 2024 (`costacurta_2024_structured_flexibility`, other batch) all sit in the same neuromodulation-in-deep-learning thematic cell but use different mechanisms (context inference, meta-learned plasticity rules, RNN reweighting). This paper is distinguished by its **focus on the stability gap** specifically and by its **Hessian-flattening proof**.
- **Krichmar-lab gain / patience work** — Xing, Zou & Krichmar 2020 *Neuromodulated patience for robot navigation* (cited as future work in Espino et al. 2024, `espino_2024_snn_path_planning`, this batch) is the closest robotics analogue.
- **Uncertainty-as-entropy.** Using softmax entropy $H(\pi)$ as the uncertainty proxy connects to the broader Bayesian / active-inference framing in the project's `uncertainty` corpus (Yu & Dayan 2005's NA-as-unexpected-uncertainty signal).
