> **Per-paper review — continual-learning corpus, paper 18 of 22.**
> Extracted from the [master review](../continual_learning_lit_review.md) (§18); content is identical. Field-evolution primer: [[continual_learning_field_evolution]].

# 18. Dohare et al. 2024 — Loss of Plasticity in Deep Continual Learning (Nature)

**PDF:** `docs/project/references/continual_learning/sources/Dohare et al. 2024 - Loss of Plasticity in Deep Continual Learning (Nature).pdf`
**Venue:** *Nature* 632, 768–774 (22 Aug 2024). **Authors:** Shibhansh Dohare, J. Fernando
Hernandez-Garcia, Qingfeng Lan, Parash Rahman, A. Rupam Mahmood, Richard S. Sutton (University
of Alberta / Amii).

> **Verification caveat (equations) — read before reuse.** The equations in this entry's Phase 2 — the continual-backprop **contribution-utility (Eq. 1)**, the **effective-rank (Eq. 2)**, and the **stable-rank** formula — were extracted from a two-column PDF whose tokens fragmented, and were **reconstructed from the surrounding prose**. Treat every symbol below as provisional and route it through a `math-reviewer` pass before lifting any equation verbatim into project code or docs.

## Phase 1: Foundational Overview (Dohare 2024)

**The plain-language question.** Standard deep learning uses two phases: a training phase where
weights change, then a frozen phase where the net is deployed. Natural learning is *continual* —
always learning. Does deep learning actually work if you never stop training on new data? This
Nature paper answers, definitively and at scale: **no.** Standard deep-learning methods
*gradually lose plasticity* until, after enough new tasks, they learn no better than a *shallow
(linear) network* — a total erasure of the benefit of depth.

**The demonstration (breadth is the point).** To be convincing for Nature, the result must be
systematic: many architectures, many optimizers, many hyperparameters, run long enough to
expose *long-term* decay (3–4 orders of magnitude more compute than a single training run).
They show plasticity loss on:
- **Continual ImageNet** — 1000 classes paired into ~500,000 binary tasks; task difficulty is
  *constant*, so any accuracy drop is pure plasticity loss. Backprop peaks near 88% then falls
  *below the linear baseline* by task 2000, at *all* step sizes.
- **Class-incremental CIFAR-100** — add 5 classes at a time; an incrementally-trained
  18-layer ResNet ends up 5% *worse* than a network retrained from scratch (a drop equal to
  removing batch norm).
- **Reinforcement learning** — a simulated ant robot with friction changed every 2M steps;
  standard PPO collapses after the first friction change. Even with *constant* friction, PPO's
  reward rises for ~3M steps then *collapses* — the ant fails every episode by 20M steps.

**The cure — continual backpropagation.** The fix is almost trivially simple: keep doing
backprop, but on every step **reinitialize a tiny fraction of the least-used units** back to
the initial random distribution (typically fewer than one unit per step). This continually
*injects diversity/variability* that gradient descent alone erodes. Continual backprop
*maintains plasticity apparently indefinitely* across all three settings — and, unlike Shrink-
and-Perturb (which perturbs *all* weights), it perturbs *selectively*, minimizing disruption to
what the net already knows.

**The three correlates of plasticity loss.** As backprop loses plasticity, three things happen
together: (1) the fraction of **dead/dormant units** rises; (2) the **average weight magnitude**
grows (→ ill-conditioned Hessian → slow convergence); (3) the **effective/stable rank** of the
representation drops (units become redundant / non-diverse). Continual backprop keeps all three
healthy.

**The bold claim.** *"Methods based on gradient descent are not enough — sustained deep learning
requires a random, non-gradient component to maintain variability and plasticity."* This is a
philosophical thesis, not just an algorithm: gradient descent is inherently *variability-
destroying*, so a *variation-and-selection* (evolution-like) process must be layered on top.

**Initial takeaway.** This is the field's capstone existence proof: standard deep learning
provably fails at continual learning, and the failure is *not intrinsic* — a cheap
selective-reinitialization rule fixes it. It elevates "loss of plasticity" from an RL curiosity
to a fundamental limitation of gradient-descent-based deep learning, published at the highest
visibility.

*Primer connection:* this is the primer's Phase-3 **anchor / capstone**. The three correlates
(dead units, weight magnitude, effective rank) and the continual-backprop algorithm are the
reference points the primer uses to organize the whole loss-of-plasticity literature; Shrink-
and-Perturb and L2 appear as the "partial fixes" that continual backprop completes.

## Phase 2: Graduate-Level Deep Dive (Dohare 2024)

### 3.1 The continual-backpropagation algorithm and its utility measure

Continual backprop = standard backprop + a *selective reinitialization* step per update. The
key design choice is *which* units to replace, governed by a **contribution utility**.

**Contribution utility (Eq. 1).** For the $i$-th hidden unit in layer $l$ at time $t$, the
utility is a running average (decay $\eta=0.99$) of the magnitude of its contribution to the
next layer:

$$
u_{l,i,t} \;=\; \eta\, u_{l,i,t-1} \;+\; (1-\eta)\,\lvert h_{l,i,t}\rvert \sum_{k=1}^{n_{l+1}} \lvert w_{l,i,k,t}\rvert ,
$$

where $h_{l,i,t}$ is the unit's activation and $w_{l,i,k,t}$ its outgoing weight to unit $k$ in
layer $l+1$ ($n_{l+1}$ = width of layer $l+1$). **Intuition / derivation of the form:** the
signal a unit injects into a downstream consumer is $h_{l,i}\, w_{l,i,k}$; summed over
consumers and taken in magnitude, $\lvert h_{l,i}\rvert \sum_k \lvert w_{l,i,k}\rvert$ measures
how much this unit *matters* to the layer above. A unit whose contribution is small can be
overwhelmed by other units' contributions, so it is a safe candidate for replacement. The EMA
smooths this over the data stream.

**The reinitialization rule and its safeguards.**
- **Zero out the outgoing weights.** When unit $r$ is reinitialized, its *incoming* weights are
  resampled from the initial distribution $d_l$, but its *outgoing* weights are set to **zero**.
  *Why:* zero outgoing weights mean the fresh unit does **not** perturb the currently-computed
  function — the net's output is unchanged the instant of replacement, so learned behavior is
  preserved and the unit re-earns its influence only through subsequent gradient descent.
- **Maturity threshold $m$.** A freshly reset unit has zero utility (its outputs are zeroed), so
  it would be *immediately* re-selected for replacement — a churn pathology. To prevent this,
  units are protected for $m$ updates (a unit is "mature" once its age exceeds $m$; $m=100$ in
  ImageNet, up to 1000 elsewhere).
- **Replacement rate $\rho$.** Each step, a fraction $\rho$ of *mature* units per layer is
  replaced. $\rho$ is tiny — e.g., $\rho=10^{-5}$ with a 512-unit layer gives
  $512\times10^{-5}=0.00512$ replacements/step ≈ one replacement per ~200 updates. The
  algorithm accumulates a fractional counter $c_l \mathrel{+}= n_{\text{eligible}}\times\rho$ and
  triggers a replacement whenever $c_l>1$.

**Algorithm 1 (per input $x_t$).** Forward pass → evaluate loss → backward pass (SGD or variant)
→ for each layer $l\in[1,L-1]$: increment ages; update utilities (Eq. 1); count eligible
(mature) units; accumulate $c_l \mathrel{+}= n_{\text{eligible}}\rho$; if $c_l>1$, find the
minimum-utility mature unit $r$, resample its incoming weights $w_{l-1}[:,r]\sim d_l$, zero its
outgoing weights $w_l[r,:]=0$, reset $u_l[r]=0$ and age$[r]=0$, decrement $c_l$. In mini-batch
settings the instantaneous per-batch utility can replace the running average to save compute.

This is the modern, deep-learning-compatible descendant of "generate-and-test" feature search
(Selfridge's Pandemonium 1959; Mahmood & Sutton 2013) — a **variation-and-selection process in
the space of units**, layered on continuing gradient descent.

### 3.2 The three quantitative correlates (formal definitions)

**(a) Dead / dormant units.** For ReLU: count units whose output is *zero for all* examples in a
2000-image sample taken at the start of each task. For sigmoidal activations: count units
within $\epsilon$ of an extreme (saturated) value. A dead unit has $\phi'(a)=0$ everywhere ⇒
zero gradient to its incoming weights ⇒ **frozen forever** (in permuted-MNIST, where all inputs
are non-negative, a first-layer dead ReLU can never revive). Dead units directly reduce network
capacity. (Cross-reference: Sokar et al. 2023's "dormant neuron"/ReDo is the sibling notion.)

**(b) Average weight magnitude & the Hessian condition-number argument.**
$\bar w = \frac{1}{|\theta|}\sum |\theta_i|$. Plasticity loss co-occurs with steady growth of
$\bar w$. The mechanistic link: in the second-order Taylor expansion of the loss,
$\ell(\theta+\delta)\approx \ell(\theta)+\nabla\ell^\top\delta + \tfrac12\delta^\top H\delta$,
the weights are tied to the Hessian $H$; large weights tend to yield a large **condition
number** $\kappa(H)=\lambda_{\max}/\lambda_{\min}$. Gradient descent's convergence rate is
governed by $\kappa$ — for a convex quadratic, the error contracts per step by
$\big(\frac{\kappa-1}{\kappa+1}\big)^2$, so $\kappa\gg1$ ⇒ near-unit contraction ⇒ crawling
progress. Hence weight growth ⇒ ill-conditioned Hessian ⇒ slow learning. (This is the same
sharpness/conditioning story Lyle 2023/2024 tell from the Hessian side.)

**(c) Effective rank and stable rank of the representation.** Let $\Phi\in\mathbb{R}^{n\times m}$
be the representation matrix with singular values $\sigma_1\ge\cdots\ge\sigma_q$,
$q=\max(n,m)$.

- **Effective rank (Eq. 2, Roy & Vetterli).** Normalize singular values into a distribution
  $p_k = \sigma_k / \lVert\sigma\rVert_1$, and take the exponential of their Shannon entropy:

$$
\operatorname{erank}(\Phi) \;=\; \exp\{H(p_1,\dots,p_q)\}, \qquad
H(p_1,\dots,p_q) \;=\; -\sum_{k=1}^{q} p_k \log(p_k).
$$

  It is continuous in $[1, \operatorname{rank}(\Phi)]$. **Interpretation & limits:** if one
  singular value dominates ($p_1\to1$, all others $\to0$), entropy $\to0$ and $\operatorname{erank}\to e^0=1$
  (one effective dimension). If all $q$ singular values are equal ($p_k=1/q$), entropy $=\log q$
  and $\operatorname{erank}=e^{\log q}=q$ (full diversity). So a *low* effective rank means a
  few units suffice to produce the layer output — the rest are redundant, a bad starting point
  for new tasks.
- **Stable rank (used for Fig. 2d).** For the same singular values sorted descending, the stable
  rank is the smallest $k$ capturing 99% of the (squared-)singular-value mass:

$$
\operatorname{srank}_{0.99}(\Phi) \;=\; \min\Big\{ k : \frac{\sum_{i\le k}\sigma_i}{\sum_{j\le q}\sigma_j} > 0.99 \Big\}.
$$

**The rank-collapse mechanism of plasticity loss.** Gradient-based optimization has an
*implicit bias toward low-rank solutions*. After each task the optimizer finds a low-rank
solution, which then *initializes* the next task; iterating, the representation's effective rank
**ratchets down** task after task, progressively shrinking the set of functions the network can
immediately represent at the start of a new task — a compounding loss of plasticity.

### 3.3 The problem suite (why each testbed exists)

- **Continual ImageNet.** 1000 classes × 700 images (600 train / 100 test); binary tasks from
  class pairs (~500k tasks); 3-conv + 3-FC network, 2-unit head reset to zero at each task
  boundary (privileged task-boundary info, used only here as it is standard for
  sequential-independent-task CL); SGD+momentum 0.9, cross-entropy. *Constant task difficulty*
  ⇒ any accuracy drop = plasticity loss. Linear baseline doesn't degrade (its whole net is reset
  each task).
- **Class-incremental CIFAR-100.** 18-layer ResNet with batch norm, data augmentation, L2, LR
  schedule ("base deep-learning system"); grow classes 5→100; early-stopping-style weight reset
  to best-validation checkpoint each increment; compared to *retrain-from-scratch* to factor out
  the intrinsically harder many-class problem. Correlates plotted (dormant %, stable rank).
- **Ant locomotion (RL).** PPO on a simulated ant; friction changed every 2M steps
  (non-stationary) or held constant 50M steps (stationary). Standard PPO collapses; tuned-Adam
  PPO less so but still degrades; PPO + L2 or + continual-backprop maintain plasticity. Under
  constant friction, correlates (Fig. 4) mirror supervised learning: dormant units rise, stable
  rank falls, weight magnitude grows under PPO; L2 fixes norm but drives weights *too* small
  (prevents committing to good behavior); continual backprop (with slight L2) keeps improving.
- **Online Permuted MNIST.** Cheap testbed: 800 random pixel-permutations as tasks, one online
  pass, no mini-batches; used for the in-depth correlate study; loss of plasticity robust across
  step sizes, network sizes (100–10,000 units — even the largest lose *some* plasticity), and
  task-change rates (10k–1M examples).
- **Slowly-Changing Regression (SCR).** Ultra-idealized CPU-scale problem (15 min/run). Input =
  binary vector of size $m{+}1$: $f$ slowly-changing bits (one flipped every $T$ examples),
  $m{-}f$ random bits, 1 constant bias bit. Target = fixed random *target network* with LTU
  (linear-threshold-unit) hidden layer, weights $\pm1$, threshold $\theta_i=(m{+}1)\beta - S_i$
  ($S_i$ = number of negative input weights). Target net (100 hidden units) is *more complex*
  than the learner (5 hidden units), forcing continual tracking of a moving best-approximation.
  Result: squared error rises for *all six* activations (sigmoid, tanh, ELU, leaky-ReLU, ReLU,
  Swish) — ReLU/tanh worst (to linear-baseline level), ELU less severe but still degrading ⇒
  plasticity loss is *not* an artifact of one activation.

### 3.4 What the existing partial fixes do (and don't)

- **L2 regularization.** Penalty $\lambda\lVert\theta\rVert_2^2$ keeps weight magnitude from
  growing ⇒ reduces plasticity loss substantially. But it does *not* stop dead-unit growth or
  effective-rank collapse ⇒ *partial* fix. In RL it can shrink weights *too* far (Fig. 4d),
  hurting commitment to good behavior.
- **Shrink-and-Perturb (Ash & Adams 2020).** L2 shrink + Gaussian noise to *all* weights. Caps
  weight magnitude *and* reduces dead units (noise revives them) ⇒ *almost* fully mitigates
  plasticity loss in permuted MNIST — but effective rank stays lower than continual backprop's,
  and it is sensitive to noise variance (too high ⇒ worse loss of plasticity).
- **Adam, Dropout, normalization.** Surprisingly, these popular methods *increased* plasticity
  loss in Dohare's continual settings (Extended Data Fig. 4a) — a notable tension with Lyle
  2023/2024, who find *layer* normalization strongly *helps*. (Likely reconciliation: Dohare's
  "normalization" is *online/batch* normalization in a single-pass online regime, and the
  optimizer/objective regimes differ; the papers agree that *weight-norm control* and
  *variability injection* matter.)

**The synthesis (Discussion).** During continual training, units become *dormant, overcommitted,
and similar to each other*; the network irreversibly loses diversity and thus the ability to
learn. Continual backprop restores diversity *selectively* (least-used units only) — variation
and selection in unit-space plus continuing gradient descent. The thesis: **gradient descent is
variability-destroying; sustained learning needs a random, non-gradient component.**

## Appendix: Section-by-Section Backbone (Dohare 2024)

- **Abstract.** Deep learning uses train-then-freeze; natural learning is continual. Standard
  deep-learning methods *lose plasticity* in continual settings until they learn no better than
  a shallow net. Shown on ImageNet + RL across networks/algorithms. Plasticity maintained only
  by algorithms that *continually inject diversity* — e.g. continual backprop (reinit a small
  fraction of least-used units). Gradient descent alone insufficient; need a random non-gradient
  component.
- **Intro.** Deep learning's train/deploy split; ChatGPT example; continuing to train on new
  data is usually ineffective; retrain-from-scratch costs millions. Real-world change is
  ubiquitous. Loss of plasticity first shown ~2000 in psychology; visible in recent works;
  distinct from catastrophic forgetting. Continual backprop overview; roots in generate-and-test
  (Pandemonium 1959).
- **Plasticity loss in supervised learning.** Continual ImageNet setup; backprop peaks 88% then
  falls below linear baseline by task 2000 at all step sizes (Fig. 1b). Weight-shrinking methods
  (L2, Shrink-and-Perturb) are exceptions and maintain plasticity (Fig. 1c). Class-incremental
  CIFAR-100 with 18-layer ResNet base system; incremental training ends 5% below
  retrain-from-scratch after 100 classes (Fig. 2b); dormant units rise, stable rank falls
  (Fig. 2c,d); continual backprop eliminates the loss.
- **Plasticity loss in reinforcement learning.** RL needs continual learning more (agent changes
  its own data). Harder to demonstrate rigorously. Ant with changing friction: standard PPO
  fails catastrophically (sawtooth for others); tuned-PPO better but degrades; PPO + L2 /
  continual backprop maintain plasticity (Fig. 3). Constant-friction ant: PPO collapses after
  ~3M steps; correlates mirror SL (Fig. 4).
- **Maintaining plasticity.** Adam/Dropout/normalization *worsen* plasticity loss; L2 and
  Shrink-and-Perturb help. Continual backprop: reinit small number of least-used units, zero
  outgoing weights, maturity protection; maintains plasticity across all settings, with fewer
  dormant units, high stable rank, constant weight magnitude. Variation-and-selection framing.
- **Discussion.** Deep learning fails when learning must continue (learns no better than shallow
  nets). Problem not intrinsic — Shrink-and-Perturb and especially continual backprop maintain
  plasticity indefinitely by adding continuing variability (continual backprop restricts it to
  least-used units).
- **Methods.**
  - *Specifics of continual backprop:* contribution-utility Eq. 1; zero outgoing weights;
    maturity threshold $m$; replacement rate $\rho$; Algorithm 1; mini-batch instantaneous
    utility option.
  - *Continual ImageNet details:* 1000 classes, 700 img/class; 32×32 downsampled; 3-conv+3-FC;
    2-unit head reset to zero at task change; SGD+momentum 0.9; step sizes 0.01/0.001/0.0001;
    30 runs; grid search over L2/Shrink-and-Perturb/continual-backprop hyperparameters.
  - *Class-incremental CIFAR-100:* increments of 5 classes; 200 epochs/increment (4000 total);
    LR schedule resetting each increment; validation-best weight reset; 18-layer ResNet; stable
    rank Eq.; final continual-backprop accuracy 76.13%.
  - *Robust loss in permuted MNIST:* Online Permuted MNIST (800 tasks, one online pass, no
    mini-batches); robustness across step sizes / network sizes (100–10,000) / task-change rates.
  - *Slowly-Changing Regression:* binary input ($f$ slow + $m{-}f$ random + 1 bias bit), LTU
    target network, six activations; error rises for all activations.
  - *Understanding loss of plasticity:* only the weights change over time ⇒ initial distribution
    has special properties (diversity, non-saturation, small magnitude). Three correlates: dead
    units (measure via all-zero over 2000 samples), average weight magnitude (→ Hessian
    condition number → slow convergence), effective rank Eq. 2 (entropy-of-singular-value-
    distribution; low rank ⇒ redundant units ⇒ bad starting point; low-rank ratchet across
    tasks).
  - *Existing methods:* L2 (caps norm, partial — dead units/rank still worsen); Shrink-and-
    Perturb (caps norm + revives dead units, near-complete but noise-sensitive); Adam/Dropout/
    online-normalization worsen plasticity loss.

# Adjacent Threads

**Bridges to the project's concerns.** Four papers that are not "loss of plasticity" proper but connect the literature to the project's specifics: the activation scheme (19) that Abbas repurposed as a plasticity fix, the covariance mechanism (20) behind policy-entropy collapse in PPO, the curriculum-learning framework (21) that lets the project's negative curriculum result be diagnosed precisely, and the task-agnostic continual-RL result (22) showing a carried recurrent belief-state can beat task-aware agents. Read together in the synthesis, the last three converge on one diagnosis of the project's curriculum failure.

---
