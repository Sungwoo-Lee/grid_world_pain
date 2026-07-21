> **Per-paper review — in-context-learning corpus, paper 36 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§36); content is identical. Manifest: [[in_context_learning_sources]].

# 36. Chen & Wang 2022 — Transformers as Meta-Learners for Implicit Neural Representations

**PDF:** `docs/project/references/in_context_learning/sources/Chen and Wang 2022 - Transformers as Meta-Learners for Implicit Neural Representations.pdf`
· ECCV 2022 · arXiv 2208.02801

## Phase 1: Foundational Overview (Undergraduate-Level)

**Background — what an INR is.** An Implicit Neural Representation (INR) stores a piece of data as a
*function*: instead of a grid of pixels, an image is a small neural net $f_\theta$ that maps a 2D
coordinate $(x,y)$ to an RGB value; a 3D scene is a NeRF mapping location+direction to colour+density.
INRs are resolution-free and compact, but *fitting* one to a given image normally means running gradient
descent from scratch — slow, and it generalizes badly from sparse observations.

**The idea.** Prior shortcuts learn a hypernetwork that squeezes each INR into a *single* latent vector —
but one vector is an information bottleneck that blurs fine detail. Gradient-based meta-learning (MetaSDF,
Learned Init) can infer the *whole* weight set but still needs test-time gradient steps. Chen & Wang
propose a **Transformer that outputs the entire INR weight set in one forward pass** — no single-vector
bottleneck, no gradient computation. Feed the image (as patch "data tokens") plus a set of learnable
"initialization tokens," and the Transformer maps the data tokens to output tokens that *are* the columns
of the INR's weight matrices.

**Key findings.** On 2D image regression (CelebA, Imagenette) and 3D view synthesis (ShapeNet NeRF), the
Transformer meta-learner *beats* the gradient-based Learned Init baseline — **without any test-time
optimization** (CelebA PSNR 31.96 vs 30.37; ShapeNet chairs 19.66 vs 18.85). More weight groups → sharper
detail (PSNR climbs monotonically from 25.6 at 1 group to 32.0 at 64). It also composes gracefully with
optional test-time refinement and with extra input views. An attention analysis shows generated weight
columns attend to *structured* image parts (nose/mouth in layer 1, forehead in layer 2, whole face in
layer 3) — the INR captures data structure without explicit supervision.

**Initial takeaway.** This is HyperTransformer's idea (paper #5) carried to a new target — generating a
data-representing neural function rather than a classifier. For the project it is the clearest statement of
the general principle: **a Transformer, viewed as a set-to-set map, is a hypernetwork whose residual
attention updates play the role of gradient-descent steps on the target weights.** The explicit "residual
attention ≈ gradient descent" formulation (below) is the tightest bridge in the shard between weight
generation and the ICL-as-optimization reading.

## Phase 2: Graduate-Level Deep Dive

**Problem setup.** Recover a signal $I:\mathcal X\to\mathcal Y$ from observations $O=\{T_i(I)\}$ (each
$T_i$ a measurement, e.g. "pixel $i$'s RGB" or "render ray $r_i$"). Represent $I$ by an INR $f_\theta$
whose weights are a set of matrices $\theta=\{W_i\in\mathbb R^{\text{in}_i\times\text{out}_i}\}_{i=0}^{m-1}$
(biases folded in). Fitting = minimize the $L_2$ loss
$$
L(\theta;O)=\frac{1}{|O|}\sum_{T_i\in O}\big\|T_i(f_\theta)-T_i(I)\big\|_2^2 .
$$

**Generalized gradient-based meta-learning.** MAML's inference is $n$ residual gradient steps from a
learned init $\theta_0$:
$$
\theta_{i+1}=\theta_i+\big(-\nabla_\theta L(\theta;O)\big|_{\theta=\theta_i}\big).
$$
Chen & Wang generalize the update to a *learned, step-specific* rule $U_{\psi_i}$ conditioned on data
$D_i$:
$$
\theta_{i+1}=\theta_i+U_{\psi_i}(\theta_i;D_i).
$$
The learnable components are now (i) an init $\theta_0$, (ii) a step count $n$, and (iii) the per-step
update rules $\{U_{\psi_i}\}$. Setting $U_{\psi_i}=-\nabla_\theta L$ recovers MAML; letting $U_{\psi_i}$
be an attention+feedforward block gives the Transformer.

**Transformer = this meta-learner (the key identification).** A Transformer's residual stream implements
exactly Eq. (3). Let $\varphi_i$ be the token states at layer $i$; then
$$
\varphi_{i+1}=\varphi_i+U_{\psi_i}(\varphi_i;d_i),
$$
where $d_i$ are the data tokens at layer $i$, $U_{\psi_i}$ is the (attention ⊕ feedforward) sublayer, the
learnable init $\varphi_0$ is the **initialization tokens**, and the token $\varphi_i$ corresponds to the
target weight $\theta_i$. The **residual connection plays the role of the $+(-\nabla_\theta)$ update in
gradient descent**, and each Transformer layer is one "meta-optimization step" whose update rule is
learned rather than fixed. (This is the same conceptual move as von Oswald et al. 2023 "Transformers learn
in-context by gradient descent," here applied to weight generation.)

**Concrete architecture (set-to-set weight decoding).**
1. **Data tokens.** Split the observation image into $P\times P$ patches (ViT-style), flatten, add a
   learnable positional embedding $e_i$, map by an FC layer: data token $=\mathrm{FC}(p_i+e_i)$. For NeRF
   view synthesis each pixel is extended to 9 channels (RGB + 3D ray origin + 3D ray direction) before
   patchifying, and multiple views simply contribute more tokens to the same set.
2. **Initialization tokens.** View each INR weight matrix $W_i$ as a set of **column vectors**; create one
   learnable init token per column (or per group — see below).
3. **Joint Transformer encoder.** Data + init tokens go in together; the encoder jointly (i) builds
   observation features via data–data attention, (ii) transfers observation knowledge to weights via
   data–init attention, and (iii) models weight–weight relations via init–init attention.
4. **Weight tokens out.** The output vectors at the init-token positions are the **weight tokens**; $m$
   independent per-layer FCs (FC$^{*}$) map them to the actual column vectors, assembling
   $\theta=\{W_i\}_{i=0}^{m-1}$.

**Training objective.** Compute $L(\theta;O)$ (Eq. 1) on the generating observations $O$; for tasks that
demand generalization (single-view synthesis) instead sample a *different* observation set $O'\ne O$ and
minimize $L(\theta;O')$, forcing the generated INR to generalize to unseen views. Purely feed-forward
backprop into the Transformer — no unrolled inner loop, no higher-order derivatives (the cost that made
MetaSDF/Learned Init "less flexible").

**Weight grouping (scalability knob).** Assigning a token per column is wasteful for large $\theta$. Group
$k$ columns and emit one token $u_{\lfloor i/k\rfloor}$ per group, then reconstitute each column as
$$
w_i=\mathrm{normalize}\big(u_{\lfloor i/k\rfloor}\cdot \bar w_i\big),
$$
where $\bar w_i$ are per-column **learnable** vectors *independent of the observation* (they keep columns
within a group distinct), and $\mathrm{normalize}$ is $L_2$ normalization. Group size $k$ trades precision
vs cost; the ablation (G = 1/4/16/64 → PSNR 25.6/27.9/29.9/32.0) shows the Transformer genuinely learns
inter-weight structure, not redundant copies.

**Results.** Image regression (Table 1): Ours > Learned Init on CelebA (31.96 vs 30.37) and Imagenette
(29.01 vs 27.07), *no test-time optimization*. View synthesis (Table 3): Ours beats Standard/Matched/
Shuffled/Learned Init on chairs/cars/lamps. Optional 100-step test-time refinement and extra views both
help further (Table 4). Attention-mask analysis (§5): weight columns attend to semantically structured
image regions → INRs exploit data structure implicitly.

**Project relevance.** Chen & Wang give the cleanest "**Transformer layer = one learned gradient step on
the target weights**" statement in the shard, generalizing MAML into an attention stack. Two importable
ideas: (1) the **residual-attention-as-optimizer** view formalizes why a conditioning module can perform
implicit adaptation without explicit gradients — the theoretical backbone for any claim that a
FiLM/hypernetwork modulator is "learning at inference"; (2) **weight grouping** is a practical recipe for
scaling weight generation (emit a few group vectors + fixed per-slot templates) that maps onto how one
might modulate a wide policy layer cheaply. Domain caveat: the target here is a data-representing INR, not
an RL policy, but the set-to-set weight-generation mechanism is domain-agnostic.

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** INRs = continuous neural functions (image: coord→RGB; NeRF: loc+dir→density+RGB);
  resolution-free/compact but need per-instance GD (slow, poor sparse generalization). Prior:
  single-vector hypernetwork (bottleneck, blurry) or gradient-based meta-learning (whole weights but
  higher-order derivatives + fixed init + test-time GD). Proposal: Transformer as hypernetwork building the
  whole INR weight set in one pass; connect to gradient-based meta-learning; analyze generated INRs.
- **§2 Related work.** INRs (DeepSDF, OccNet, IM-NET, NeRF; SIREN sine activations; Fourier features).
  Hypernetworks for INRs (single-vector latent decode; hybrid feature-map/grid methods). Meta-learning
  (MAML/Reptile; MetaSDF, Learned Init — fixed init + test-time optimization). Transformers (NLP → ViT →
  meta-learning here).
- **§3 Method.** §3.1 problem formulation: signal $I$, INR $f_\theta$ with matrix set $\theta$, observation
  set $O=\{T_i(I)\}$, $L_2$ loss (Eq.1). §3.2 motivating from gradient-based meta-learning: MAML residual
  update (Eq.2) → generalized learned update $U_{\psi_i}$ (Eq.3) → Transformer residual instantiation
  (Eq.4), residual link ≈ subtracting gradients (Fig 2). §3.3 Transformers as meta-learners: data tokens
  (ViT patches) + init tokens (one per weight column) → Transformer encoder → weight tokens → layerwise
  FC$^{*}$ → $\theta$ (Fig 3); train with $L(\theta;O)$ or $L(\theta;O')$ for generalization. §3.4 weight
  grouping: $w_i=\mathrm{normalize}(u_{\lfloor i/k\rfloor}\cdot\bar w_i)$ (Fig 4).
- **§4 Experiments.** §4.1 image regression (CelebA/Imagenette; ViT-Base-like 6-layer encoder; 5-layer
  256-hidden MLP INR; 64 groups default): Table 1 beats Learned Init; Table 2 group ablation. §4.2 view
  synthesis (ShapeNet chairs/cars/lamps; 9-channel ray-augmented views; 6-layer NeRF): Table 3 beats
  baselines + Learned Init w/o test-time opt; adaptive foreground sampling for stability; Table 4 test-time
  opt + multi-view gains.
- **§5 Does the INR exploit data structure?** Visualize last-layer weight-token→data-token attention →
  weight columns attend to structured parts (nose/mouth, forehead, whole face) (Fig 7).
- **§6 Conclusion.** Transformer hypernetwork builds INRs in one pass, beats gradient-based meta-learning,
  composable with test-time optimization; generated INRs implicitly capture data structure.

---
