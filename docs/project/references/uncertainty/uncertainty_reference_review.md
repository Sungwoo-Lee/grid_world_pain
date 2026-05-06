---
title: "Reference Review — Heteroscedastic Uncertainty in Deep Learning (8 Papers)"
status: DRAFT v1 — all 8 papers reviewed, synthesis complete
last_updated: 2026-04-14
target_depth: graduate-level — full derivations in LaTeX
related:
  - perceptual_noise_lit_review.md (§2, SQ1)
  - ../develop/active/filim/FiLM_ENSEMBLE_SENSORY_PRECISION.md
scope: |
  Per-paper review of the eight PDFs in docs/project/references/uncertainty/.
  Each entry has three layers: (1) Basic introduction and concept, (2) Main results
  and algorithm, (3) Graduate-level deep dive with every equation from the paper in
  LaTeX and line-by-line derivations. Reviewed one paper at a time with the
  document updated after each entry.
---

# Reference Review — Heteroscedastic Uncertainty in Deep Learning

> **How to read.** Each paper's entry has a fixed 3-layer structure. Skim Layer 1 for
> orientation, Layer 2 for the practical takeaway, and Layer 3 when you need to
> reimplement the method or understand why it behaves the way it does. Equation
> numbers in Layer 3 follow the original paper's numbering where possible, with
> `(L.x)` used for derivation steps that are not in the paper itself.

## Status Dashboard

| # | Paper (year) | Status | Core contribution (one line) |
|---|--------------|--------|------------------------------|
| 1 | Kendall & Gal (2017) | DONE | Aleatoric + epistemic uncertainty in a single Bayesian DL framework |
| 2 | Lakshminarayanan et al. (2017) | DONE | Deep ensembles as a simple, non-Bayesian uncertainty estimator |
| 3 | Amini et al. (2019) | DONE | Deep Evidential Regression — NIG prior over (μ, σ²) |
| 4 | Detlefsen et al. (2019) | DONE | Reliable training for heteroscedastic variance networks |
| 5 | Romano et al. (2019) | DONE | Conformalized Quantile Regression (CQR) — distribution-free intervals |
| 6 | Stirn & Knowles (2020) | DONE | Variational variance — calibrated σ² via variational inference |
| 7 | Seitzer et al. (2022) | DONE | β-NLL: diagnosis + fix for heteroscedastic collapse |
| 8 | Meinert et al. (2023) | DONE | DER does not actually separate aleatoric from epistemic |

## Notation (used throughout)

Common symbols reused across papers to keep LaTeX consistent:

- $x \in \mathcal{X}$ — input; $y \in \mathbb{R}^d$ — target; $f_\theta(x)$ — network with weights $\theta$.
- $\mu_\theta(x)$ — predicted mean head; $\sigma^2_\theta(x)$ — predicted variance head.
- $s_\theta(x) := \log \sigma^2_\theta(x)$ — log-variance parameterization.
- $\beta_1, \beta_2$ — NIG prior parameters (Amini); $\alpha, \beta$ — generic (context dependent).
- $\mathcal{L}$ — loss; $\mathbb{E}_{p}[\cdot]$ — expectation; $\mathrm{KL}(q\|p)$ — KL divergence.

---

## 1. Kendall & Gal (2017) — *What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?*

**Venue:** NIPS 2017 (arXiv:1703.04977). **Domain:** Per-pixel semantic segmentation and
monocular depth regression.

### Layer 1 — Basic introduction and concept

The paper makes one organizing move that has shaped the entire subsequent uncertainty-in-DL
literature: it distinguishes **aleatoric** uncertainty (noise inherent in the observations —
e.g. sensor noise, labelling noise) from **epistemic** uncertainty (noise inherent in the model —
what-the-model-does-not-know, reducible by more data). The distinction is not original to the
authors (it comes from Der Kiureghian & Ditlevsen, 2009, and before that from reliability
engineering), but Kendall & Gal operationalize it in a *single deep-learning framework* that
captures both:

- **Epistemic uncertainty** is captured by a Bayesian neural network with a prior $p(\mathbf{W})$
  over the weights; the weight posterior $p(\mathbf{W}\mid \mathbf{X},\mathbf{Y})$ is approximated
  by **MC Dropout** (Gal & Ghahramani, 2016), i.e. dropout applied at both train *and* test time,
  with $T$ stochastic forward passes summarised by their mean and variance.
- **Aleatoric uncertainty** is captured by having the network output not only $\hat{y}$ but also
  a per-input **log-variance** $\hat{s} = \log \hat{\sigma}^2$; training uses a Gaussian NLL,
  which *automatically down-weights* residuals from high-predicted-variance points.

Aleatoric is further split into **homoscedastic** (constant across inputs — a single scalar
$\sigma$) and **heteroscedastic** (input-dependent — $\sigma(x)$). Only the heteroscedastic
case is the focus of the paper and of precision-modulation applications downstream.

The paper's central *empirical* observation: in vision-sized datasets, epistemic uncertainty is
largely "explained away" by the data, so aleatoric modelling does most of the practical work —
but epistemic is still essential for flagging out-of-distribution inputs (where aleatoric
stays low but epistemic rises). This motivates the combined model rather than either alone.

### Layer 2 — Main results and algorithm

**Algorithmic core.** Place a single MC-Dropout BNN with a *two-headed* output: one head predicts
the mean $\hat y$ and the other predicts the log-variance $\hat s$. Train under a Gaussian NLL
on $\hat y$ weighted by $\exp(-\hat s)$. At test time, do $T$ stochastic forward passes and
decompose the predictive variance into an epistemic term (variance across passes) and an
aleatoric term (mean predicted $\hat\sigma^2$).

**Loss (regression).** With pixel index $i$ and network output $[\hat y_i, \hat s_i] =
f^{\widehat{\mathbf W}}(x_i)$,

$$
\mathcal L_{\mathrm{BNN}}(\theta) \;=\; \frac{1}{D}\sum_{i=1}^{D}
\tfrac{1}{2}\,\exp(-\hat s_i)\,\|y_i - \hat y_i\|^2 \;+\; \tfrac{1}{2}\,\hat s_i.
\tag{K-8}
$$

This is the paper's canonical heteroscedastic regression loss (Eq. 8 in the paper) and is the
ancestor of every subsequent per-pixel / per-channel uncertainty head in the deep-learning
literature.

**Predictive variance (regression).** With $T$ MC samples producing $\{\hat y_t, \hat\sigma_t^2\}$,

$$
\mathrm{Var}(y) \;\approx\; \underbrace{\frac{1}{T}\sum_{t=1}^T \hat y_t^{2} - \Big(\frac{1}{T}\sum_{t=1}^T \hat y_t\Big)^{\!2}}_{\text{epistemic}}
\;+\; \underbrace{\frac{1}{T}\sum_{t=1}^T \hat\sigma_t^{2}}_{\text{aleatoric}}.
\tag{K-9}
$$

The decomposition is exact under the model (derivation in Layer 3 §1.3.3).

**Loss (classification).** Place a Gaussian $\mathcal N(\mathbf f_i^{\mathbf W}, (\sigma_i^{\mathbf W})^2)$
over the *pre-softmax logits*, sample $T$ logit realisations, and log-sum-exp them:

$$
\hat{\mathbf x}_{i,t} = \mathbf f_i^{\mathbf W} + \boldsymbol\sigma_i^{\mathbf W}\odot\boldsymbol\epsilon_t,
\qquad \boldsymbol\epsilon_t\sim\mathcal N(\mathbf 0,\mathbf I),
$$

$$
\mathcal L_{x} \;=\; \sum_i \log\frac{1}{T}\sum_t \exp\!\Big(\hat x_{i,t,c} - \log\sum_{c'} \exp \hat x_{i,t,c'}\Big).
\tag{K-12}
$$

The paper calls this **heteroscedastic classification** and shows it acts as *learned loss
attenuation* on ambiguous pixels — analogous to the regression case.

**Empirical result.** On CamVid, NYUv2 (segmentation) and Make3D, NYUv2-Depth (regression), the
combined model sets new SOTA. The improvement from aleatoric alone is larger than from epistemic
alone; combining the two adds a further small but consistent gain. Table 3 of the paper also
demonstrates the **aleatoric-stays-constant-but-epistemic-grows-on-OOD** behaviour that
subsequent literature (e.g. the natural-RL / distractor-benchmark line) relies on.

### Layer 3 — Graduate-level deep dive with full derivations

#### 1.3.1 The MC-Dropout variational objective (Eq. K-1)

The starting point is Gal & Ghahramani's (2016) reframing of dropout as a variational Bayesian
approximation. Let $q_\theta(\mathbf W)$ be the dropout approximating posterior (a mixture of
two Gaussians per weight, with the zero-centred component having vanishing variance — this is
Gal's Bernoulli-approximation construction). The standard variational lower bound is

$$
\log p(\mathbf Y\mid\mathbf X)
\;\geq\;
\underbrace{\int q_\theta(\mathbf W)\,\log p(\mathbf Y\mid\mathbf X,\mathbf W)\,\mathrm d\mathbf W}_{\text{expected log-likelihood}}
\;-\;
\underbrace{\mathrm{KL}\!\big(q_\theta(\mathbf W)\,\|\,p(\mathbf W)\big)}_{\text{regulariser}}.
\tag{L.1.1}
$$

With i.i.d. data the expected log-likelihood splits over the $N$ data points. Gal shows that for a
Gaussian prior $p(\mathbf W) = \mathcal N(0, \ell^{-2}\mathbf I)$ (length-scale $\ell$) and the
dropout $q_\theta$, the KL term reduces (up to constants and in the limit of small sub-Gaussian
variances) to an $\ell_2$ weight-decay penalty:

$$
\mathrm{KL}\!\big(q_\theta(\mathbf W)\,\|\,p(\mathbf W)\big) \;\approx\; \frac{1-p}{2}\,\|\theta\|^2 \;+\; \mathrm{const},
\tag{L.1.2}
$$

where $p$ is the dropout *keep probability*. Dividing the resulting objective by $N$ and flipping
signs to make it a minimisation target gives Kendall & Gal's Eq. K-1:

$$
\mathcal L(\theta, p) \;=\; -\frac{1}{N}\sum_{i=1}^{N} \log p\!\big(y_i\mid f^{\widehat{\mathbf W}_i}(x_i)\big)
\;+\; \frac{1-p}{2N}\,\|\theta\|^2.
\tag{K-1}
$$

Each minibatch sample draws one $\widehat{\mathbf W}_i\sim q_\theta$ — this is exactly what a
standard dropout forward pass does.

#### 1.3.2 Homoscedastic Gaussian likelihood → Eq. K-2

Specialise K-1 to regression with fixed observation noise $\sigma^2$. The Gaussian likelihood is

$$
p(y_i\mid f^{\widehat{\mathbf W}_i}(x_i)) \;=\; \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\Big(-\tfrac{1}{2\sigma^2}\|y_i - f^{\widehat{\mathbf W}_i}(x_i)\|^2\Big),
\tag{L.1.3}
$$

so its negative log is

$$
-\log p(y_i\mid\cdot) \;=\; \tfrac{1}{2\sigma^2}\|y_i - f^{\widehat{\mathbf W}_i}(x_i)\|^2 \;+\; \tfrac{1}{2}\log\sigma^2 \;+\; \tfrac{1}{2}\log 2\pi,
\tag{K-2}
$$

matching K-2 once the additive constant $\tfrac{1}{2}\log 2\pi$ is dropped (the paper writes "$\propto$").

#### 1.3.3 Predictive-variance decomposition → Eq. K-9

The predictive distribution under the approximate posterior is

$$
p(y^\star\mid x^\star, \mathbf X, \mathbf Y) \;\approx\; \int q_\theta(\mathbf W)\,
\mathcal N\!\big(y^\star;\, \hat y^\star(\mathbf W), \hat\sigma^{2\star}(\mathbf W)\big)\,\mathrm d\mathbf W.
\tag{L.1.4}
$$

Let $Y^\star$ denote the random output under this distribution. The law of total variance states

$$
\mathrm{Var}(Y^\star) \;=\; \underbrace{\mathbb E_{\mathbf W}[\mathrm{Var}(Y^\star\mid\mathbf W)]}_{\text{aleatoric}}
\;+\; \underbrace{\mathrm{Var}_{\mathbf W}[\mathbb E(Y^\star\mid\mathbf W)]}_{\text{epistemic}}.
\tag{L.1.5}
$$

Under the Gaussian likelihood, $\mathbb E[Y^\star\mid\mathbf W] = \hat y^\star(\mathbf W)$ and
$\mathrm{Var}(Y^\star\mid\mathbf W) = \hat\sigma^{2\star}(\mathbf W)$. Replace the intractable
expectations by MC averages over $T$ draws $\widehat{\mathbf W}_t\sim q_\theta$:

$$
\mathbb E_{\mathbf W}[\mathrm{Var}(Y^\star\mid\mathbf W)]
\;\approx\; \frac{1}{T}\sum_{t=1}^T \hat\sigma_t^{2}
\qquad\text{(aleatoric estimate)},
\tag{L.1.6}
$$

$$
\mathrm{Var}_{\mathbf W}[\mathbb E(Y^\star\mid\mathbf W)]
\;\approx\; \frac{1}{T}\sum_{t=1}^T \hat y_t^2 - \Big(\frac{1}{T}\sum_{t=1}^T \hat y_t\Big)^{\!2}
\qquad\text{(epistemic estimate)}.
\tag{L.1.7}
$$

Summing L.1.6 + L.1.7 reproduces K-9 exactly. Note two subtle points:

1. **Why the aleatoric term uses predicted variance, not residual.** Under the assumed likelihood,
   the conditional variance *is* $\hat\sigma^2(\mathbf W)$ by construction — there is no residual
   entering L.1.6. Any bias in the aleatoric head propagates directly to the decomposition; this
   is the entry point for the failure modes later dissected by Seitzer et al. (2022).
2. **Decomposition is additive, not orthogonal.** L.1.5 is an equality, but L.1.6 and L.1.7 are
   MC estimates with their own correlated noise. Treating them as "the" aleatoric and epistemic
   uncertainties is a modelling choice that Meinert et al. (2023) will later criticise in a
   different setting (Deep Evidential Regression).

#### 1.3.4 Heteroscedastic loss → Eqs. K-7 and K-8

Make $\sigma^2$ a function of input by extending the network to output $[\hat y_i, \hat\sigma_i^2]$
(Eq. K-6). The Gaussian NLL at sample $i$ is

$$
-\log p(y_i\mid\hat y_i, \hat\sigma_i^2)
\;=\; \tfrac{1}{2}\hat\sigma_i^{-2}\|y_i - \hat y_i\|^2 \;+\; \tfrac{1}{2}\log\hat\sigma_i^2 \;+\; \tfrac{1}{2}\log 2\pi.
\tag{L.1.8}
$$

Summing over the $D$ output pixels of a single image and dropping the constant gives K-7:

$$
\mathcal L_{\mathrm{BNN}}(\theta) \;=\; \frac{1}{D}\sum_i \tfrac{1}{2}\hat\sigma_i^{-2}\|y_i - \hat y_i\|^2 + \tfrac{1}{2}\log\hat\sigma_i^2.
\tag{K-7}
$$

To avoid predicting $\sigma^2$ directly (with the attendant positivity constraint and the
division-by-zero hazard), reparameterise as $s_i := \log\hat\sigma_i^2$. Then
$\hat\sigma_i^{-2} = \exp(-s_i)$ and $\log\hat\sigma_i^2 = s_i$, yielding K-8:

$$
\mathcal L_{\mathrm{BNN}}(\theta) \;=\; \frac{1}{D}\sum_i \tfrac{1}{2}\exp(-s_i)\|y_i - \hat y_i\|^2 + \tfrac{1}{2}s_i.
\tag{K-8}
$$

The log-parameterisation has three properties worth remembering, because every downstream method
(β-NLL, variational variance, evidential regression) ultimately argues with one of them:

**(a) Automatic residual attenuation.** Differentiate K-8 with respect to $\hat y_i$:

$$
\frac{\partial \mathcal L}{\partial \hat y_i} \;=\; \exp(-s_i)\,(\hat y_i - y_i).
\tag{L.1.9}
$$

A large predicted log-variance ($s_i \gg 0$) exponentially *shrinks* the mean-head gradient —
noisy samples are effectively down-weighted during training. This is the "learned loss
attenuation" the paper emphasises.

**(b) Balancing against the log-term.** Differentiate K-8 with respect to $s_i$:

$$
\frac{\partial \mathcal L}{\partial s_i} \;=\; \tfrac{1}{2}\Big[\,1 - \exp(-s_i)\|y_i - \hat y_i\|^2\,\Big].
\tag{L.1.10}
$$

Setting $\partial\mathcal L/\partial s_i = 0$ gives the optimal $s_i^\star = \log \|y_i-\hat y_i\|^2$,
i.e. the network is trained to predict the **log of the squared residual** (up to scaling). The
$+\tfrac{1}{2}s_i$ term stops the network from taking the trivial escape of sending $s_i\to\infty$
to zero out the attenuation.

**(c) The feedback loop L.1.9 × L.1.10.** When $(\hat y_i - y_i)^2$ shrinks quickly due to fitting,
$\partial\mathcal L/\partial s_i$ pushes $s_i$ *down*; a smaller $s_i$ in turn *increases* the mean
gradient via L.1.9. This feedback loop is the point of failure that Seitzer et al. (2022) will
analyse in depth (§7 below): if the mean fits too quickly, the variance head starves and
collapses.

#### 1.3.5 Heteroscedastic classification → Eqs. K-10 to K-12

The classification case places Gaussian noise on the *logits* rather than the class
probabilities. Let $\mathbf f_i^{\mathbf W}, \boldsymbol\sigma_i^{\mathbf W}$ be the mean and
diagonal log-variance outputs of the network for pixel $i$. The generative model is

$$
\hat{\mathbf x}_i \mid \mathbf W \;\sim\; \mathcal N\!\big(\mathbf f_i^{\mathbf W},\,\mathrm{diag}((\boldsymbol\sigma_i^{\mathbf W})^2)\big),
\qquad
\hat{\mathbf p}_i \;=\; \mathrm{Softmax}(\hat{\mathbf x}_i).
\tag{K-10}
$$

The expected log-likelihood at observed class $c$ is

$$
\mathcal L_{\text{exact}} \;=\; \log \mathbb E_{\hat{\mathbf x}_i\sim\mathcal N(\cdot)}\big[\hat p_{i,c}\big],
\tag{K-11}
$$

with no closed form because the softmax does not commute with the Gaussian. Use $T$-sample MC
integration with the reparameterisation trick:

$$
\hat{\mathbf x}_{i,t} \;=\; \mathbf f_i^{\mathbf W} + \boldsymbol\sigma_i^{\mathbf W}\odot\boldsymbol\epsilon_t,
\qquad\boldsymbol\epsilon_t\sim\mathcal N(\mathbf 0,\mathbf I).
$$

Then

$$
\log\mathbb E[\hat p_{i,c}]
\;\approx\; \log\frac{1}{T}\sum_t \hat p_{i,t,c}
\;=\; \log\frac{1}{T}\sum_t \exp\!\big(\hat x_{i,t,c} - \log\textstyle\sum_{c'} \exp\hat x_{i,t,c'}\big),
\tag{K-12}
$$

where we used $\hat p_{i,t,c} = \exp(\hat x_{i,t,c})/\sum_{c'}\exp(\hat x_{i,t,c'})$. The derivation
is a single log-of-expectation written with the softmax's log form — the *numerical stability*
comes from subtracting the per-sample log-sum-exp inside the log, so each $\hat p_{i,t,c}$ is
computed in log-domain before being exponentiated for the outer mean.

**Why this is "loss attenuation" in classification.** Large predicted logit-variance
$(\sigma_i^{\mathbf W})^2$ injects large $\boldsymbol\epsilon_t\odot\boldsymbol\sigma_i$ noise into
the logits. The softmax flattens toward uniform, so $\hat p_{i,t,c} \to 1/C$ irrespective of $c$;
the per-pixel NLL plateaus at $\log C$ and stops producing gradient signal for the mean head. The
mechanism is structurally identical to the exp(-s) attenuation in regression but routed through
softmax flattening instead of a multiplicative reweighting.

#### 1.3.6 What this buys for the broader lineage

Three properties of this paper's construction become load-bearing in later methods and are worth
highlighting for cross-reference:

1. **Log-variance parameterization $s = \log\sigma^2$** (L.1.8, K-8) becomes universal; every
   heteroscedastic paper reviewed below inherits this choice.
2. **Law-of-total-variance decomposition** (L.1.5, K-9) is the template Lakshminarayanan et al.
   (2017, §2) imitate with deep ensembles, and is the target Meinert et al. (2023, §8) will argue
   Deep Evidential Regression fails to deliver.
3. **The mean-vs-variance gradient asymmetry** (L.1.9 vs L.1.10) is the mechanism Seitzer et al.
   (2022, §7) will diagnose as the source of heteroscedastic collapse, and which Detlefsen et al.
   (2019, §4) will mitigate by locally-aware estimation.

The paper does not itself discuss these failure modes — it reports an empirical success on
vision tasks — but its choice of parameterisation is the exact object the rest of this review
will be debating.

---

## 2. Lakshminarayanan, Pritzel & Blundell (2017) — *Simple and Scalable Predictive Uncertainty Estimation Using Deep Ensembles*

**Venue:** NIPS 2017 (arXiv:1612.01474). **Domain:** General-purpose — regression benchmarks,
MNIST/SVHN/ImageNet classification, out-of-distribution detection.

### Layer 1 — Basic introduction and concept

Published at the *same NIPS 2017* as Kendall & Gal, this paper offers a deliberately
**non-Bayesian** alternative to MC Dropout: simply train $M$ copies of the network with
different random initialisations (and optionally adversarial training), each with a heteroscedastic
Gaussian NLL head, and average their predictions. The authors argue — and demonstrate empirically —
that this produces uncertainty estimates on par with or better than approximate BNNs, while being
trivially parallelisable and requiring essentially no modification to the standard training pipeline.

The three ingredients of the "recipe":

1. **Proper scoring rule objective.** Train each network with a proper scoring rule (Gaussian NLL
   for regression, log-likelihood / Brier for classification) so the *predictive distribution*
   itself is optimised, not just the mean.
2. **Learned variance head** (regression only). Following Nix & Weigend (1994), add a second
   output $\sigma^2_\theta(x)$ passed through softplus for positivity. This gives
   *heteroscedastic aleatoric* uncertainty per input.
3. **Ensemble of $M$ independently-initialised networks** (default $M=5$). No bagging, no
   boosting — just different random seeds. Average the predictive distributions.
4. **Optional adversarial training** via the Fast Gradient Sign Method (Goodfellow et al., 2015)
   to smooth predictive distributions around training inputs.

**Why this matters for precision modulation.** Deep Ensembles is the canonical method that
Turkoglu et al. (2022) fold into FiLM-Ensemble — the FiLM-modulated backbone is shared across
ensemble members, while the heteroscedastic head is per-member. The across-member variance
supplies *epistemic* uncertainty; the per-member $\sigma^2$ supplies *aleatoric*. So this paper
is the methodological parent of the core Phase-3 proposal in the GridWorld Pain plan.

### Layer 2 — Main results and algorithm

**Algorithmic core (Algorithm 1).** For each of $M$ networks, independently initialise weights
$\theta_m$, optionally generate one adversarial example per minibatch sample, and minimise the
proper-scoring-rule loss. At test time, combine as a uniformly-weighted mixture.

**Regression NLL head (Eq. L-1).** Each network outputs $(\mu_\theta(x), \sigma^2_\theta(x))$:

$$
-\log p_\theta(y_n\mid x_n) \;=\; \frac{\log \sigma^2_\theta(x_n)}{2} \;+\; \frac{(y_n - \mu_\theta(x_n))^2}{2\sigma^2_\theta(x_n)} \;+\; \text{const}.
\tag{L-1}
$$

This is algebraically identical to Kendall & Gal's K-7 (modulo the log-variance reparameterisation
Kendall uses in K-8). What distinguishes this paper is the *ensembling on top*.

**Ensemble combination (for regression).** The ensemble's predictive distribution is a uniform
mixture of Gaussians:

$$
p(y\mid x) \;=\; \frac{1}{M}\sum_{m=1}^M \mathcal N\!\big(y;\, \mu_{\theta_m}(x), \sigma^2_{\theta_m}(x)\big).
$$

The paper approximates this mixture by a single Gaussian with matched first two moments:

$$
\mu_\star(x) \;=\; \frac{1}{M}\sum_m \mu_{\theta_m}(x),
\qquad
\sigma^2_\star(x) \;=\; \frac{1}{M}\sum_m\!\Big[\sigma^2_{\theta_m}(x) + \mu^2_{\theta_m}(x)\Big] - \mu_\star^2(x).
\tag{L-2}
$$

The second equation is the *mixture-of-Gaussians variance formula* (derived in Layer 3 §2.3.2):
the total variance decomposes into the mean of the per-member variances (aleatoric-like) plus the
variance of the per-member means (epistemic-like).

**Adversarial training.** Given loss $\ell(\theta, x, y)$, generate
$x' = x + \epsilon\,\mathrm{sign}(\nabla_x \ell(\theta, x, y))$ and minimise the combined
$\ell(\theta, x, y) + \ell(\theta, x', y)$. Recommended $\epsilon = 0.01 \times$ input range.

**Empirical results (highlights).**

- **UCI regression.** On 10 standard regression benchmarks (Boston, Concrete, Energy, Kin8nm,
  Naval, Power, Protein, Wine, Yacht, Year MSD), deep ensembles beat both Probabilistic
  Backpropagation (Hernández-Lobato & Adams, 2015) and MC Dropout on NLL on most datasets, often
  by a large margin.
- **MNIST / SVHN classification.** Ensembles + adversarial training dominates MC Dropout on
  accuracy, NLL, and Brier score — and the gap *widens* with $M$.
- **ImageNet.** First published application of ensemble-based uncertainty to ImageNet; clear
  NLL / Brier improvements with $M$ from 1 to 10.
- **OOD detection (MNIST→NotMNIST, SVHN→CIFAR10, ImageNet dogs→non-dogs).** MC Dropout produces
  confidently-wrong predictions on out-of-distribution inputs (predictive entropy stays near 0);
  deep ensembles produce rapidly-rising entropy on OOD as $M$ grows. This is the paper's most
  visually striking result and the reason the method became the default OOD baseline.

**Calibration.** Figure 7 (supplementary) shows deep ensembles are well-calibrated on regression
tasks, while MSE-trained ensembles (which just use empirical variance across members) are
severely *under*-confident — a direct empirical demonstration that learning the variance matters.

### Layer 3 — Graduate-level deep dive with full derivations

#### 2.3.1 Proper scoring rules and why NLL is one

A scoring rule $S(p_\theta, (y, x))$ assigns a real number to a predictive distribution. Its
expected form under the true joint $q(y, x)$ is

$$
S(p_\theta, q) \;=\; \int q(y, x)\, S(p_\theta, (y, x))\, \mathrm d y\, \mathrm d x.
\tag{L.2.1}
$$

A scoring rule is **proper** iff $S(p_\theta, q) \le S(q, q)$ with equality iff $p_\theta = q$. Proper
scoring rules are the correct training objectives for probabilistic prediction because their unique
maximum (asymptotically) recovers the true conditional distribution.

**Log-likelihood is proper (Gibbs' inequality).** For $S(p_\theta, (y, x)) = \log p_\theta(y\mid x)$,

$$
\mathbb E_{q(x)q(y\mid x)}\!\big[\log p_\theta(y\mid x)\big]
\;-\; \mathbb E_{q(x)q(y\mid x)}\!\big[\log q(y\mid x)\big]
\;=\; -\,\mathbb E_{q(x)}\!\big[\mathrm{KL}(q(\cdot\mid x)\,\|\,p_\theta(\cdot\mid x))\big] \;\le\; 0,
\tag{L.2.2}
$$

with equality iff the KL vanishes everywhere, i.e. $p_\theta = q$. Hence $S(p_\theta, q) \le S(q, q)$ —
this is Gibbs' inequality.

**Brier score is proper.** For $K$-way classification with one-hot label $\delta_{k=y}$,

$$
\mathrm{BS}(p_\theta, (y, x)) \;=\; \frac{1}{K}\sum_{k=1}^K \big(\delta_{k=y} - p_\theta(y=k\mid x)\big)^2.
\tag{L.2.3}
$$

Differentiating the expected Brier with respect to $p_\theta(y=k\mid x)$ at fixed $x$ and setting to zero gives $p_\theta(y=k\mid x) = q(y=k\mid x)$ — i.e. the minimiser is the truth, so Brier is proper.
The two classical proper rules (log-likelihood, Brier) are the ones the paper uses.

#### 2.3.2 Mixture-of-Gaussians moment formula → Eq. L-2

Let $Y$ be a random variable and $M$ a discrete latent with $P(M=m) = 1/M$, with
$Y\mid M=m \sim \mathcal N(\mu_m, \sigma_m^2)$. Then by the **law of total variance**,

$$
\mathrm{Var}(Y) \;=\; \mathbb E_M[\mathrm{Var}(Y\mid M)] \;+\; \mathrm{Var}_M[\mathbb E(Y\mid M)].
\tag{L.2.4}
$$

Expanding: $\mathbb E_M[\mathrm{Var}(Y\mid M)] = \tfrac{1}{M}\sum_m \sigma_m^2$ (mean of per-member
variances) and $\mathrm{Var}_M[\mathbb E(Y\mid M)] = \tfrac{1}{M}\sum_m \mu_m^2 - (\tfrac{1}{M}\sum_m \mu_m)^2$ (variance of per-member means). Summing:

$$
\sigma_\star^2
\;=\; \underbrace{\frac{1}{M}\sum_m \sigma_m^2}_{\text{aleatoric}}
\;+\; \underbrace{\Big(\frac{1}{M}\sum_m \mu_m^2 - \mu_\star^2\Big)}_{\text{epistemic}}
\;=\; \frac{1}{M}\sum_m\!\big[\sigma_m^2 + \mu_m^2\big] \;-\; \mu_\star^2,
\tag{L-2}
$$

which is exactly the paper's L-2. **Cross-reference:** this is the same variance decomposition as
Kendall & Gal's Eq. K-9 (Layer 3 §1.3.3, L.1.6 + L.1.7) — the *source* of across-member variance is
different (dropout masks vs. independent initialisations), but the moment formula is identical.
This is the sense in which deep ensembles are "non-Bayesian but serve the same function."

#### 2.3.3 Bayesian model averaging vs. ensemble model combination

A subtle but important point argued in §2.4 of the paper. Bayesian model averaging (BMA) performs

$$
p_{\text{BMA}}(y\mid x, \mathcal D) \;=\; \int p(y\mid x, \mathbf W)\, p(\mathbf W\mid \mathcal D)\, \mathrm d\mathbf W,
\tag{L.2.5}
$$

which under a sharp posterior collapses to the *single* best weight (point prediction). A deep
ensemble, by contrast, is **not** trying to approximate the posterior: each member is a
point-estimator and the ensemble is a *genuinely new* distribution over predictions. When the
model class does not contain the true data-generating process (the usual situation), ensemble
combination has access to a *richer* hypothesis class than BMA — Minka (2000) and Clarke (2003)
make this point sharply.

For the GridWorld Pain use case: FiLM-Ensemble inherits this property. Ensemble members are
separate point estimators of *both* the mean and the variance, and the across-member spread is
interpretable as an approximation to epistemic uncertainty even though the training procedure is
purely frequentist.

#### 2.3.4 Why variance-learning strictly dominates MSE+empirical-variance

The paper's supplementary Table 2 shows that an ensemble of $M$ networks trained with MSE, using
the *empirical variance across members* as the uncertainty estimate, is catastrophically
miscalibrated (see the 80%-interval-contains-20% calibration curve on Year MSD). The reason is
structural: under squared-error training, each member converges toward the conditional mean
$\mathbb E[y\mid x]$, so the across-member variance decays at rate $O(1/M)$ toward zero and
contains *no information about the noise level* $\mathrm{Var}(y\mid x)$. Only by training each
member with a *scoring rule* that depends on predicted variance can the aleatoric term survive.

Formally, with MSE loss $\mathcal L_m = \|y - \mu_m(x)\|^2$, stationarity implies
$\mu_m^\star(x) \to \mathbb E[y\mid x]$ for *every* member, so
$\mathrm{Var}_M[\mu_m(x)] \to 0$. But the true predictive variance includes the aleatoric term
$\sigma^2(x) = \mathrm{Var}(y\mid x)$, which does not vanish; the MSE ensemble estimate therefore
underestimates total variance by exactly this term.

**Implication for Phase 3.** Any GridWorld Pain precision-head experiment that tries to read
uncertainty from across-seed variance of deterministic critics/actors will hit this failure mode.
The NLL-trained head is not optional.

#### 2.3.5 Adversarial training as variational smoothing

Adversarial training at $\epsilon$-scale approximates a smoothed likelihood

$$
\tilde p_\theta(y\mid x) \;\propto\; \int p_\theta(y\mid x + \delta)\, \mathbb 1[\|\delta\|_\infty \le \epsilon]\, \mathrm d\delta,
\tag{L.2.6}
$$

by pointing the perturbation in the *worst-case* direction rather than a random one. A first-order
Taylor expansion of the loss around $x$ gives

$$
\ell(\theta, x + \delta, y) \;\approx\; \ell(\theta, x, y) + \delta^\top \nabla_x \ell(\theta, x, y).
\tag{L.2.7}
$$

The $\ell_\infty$-ball maximiser is $\delta^\star = \epsilon\,\mathrm{sign}(\nabla_x \ell)$ — this is
FGSM. Training with $\ell(\theta, x', y)$ alongside $\ell(\theta, x, y)$ therefore penalises
*rapid changes in likelihood* along the worst-case input direction, which is exactly the local
smoothness a well-calibrated predictive distribution needs. For uncertainty specifically: if the
network's $\log \sigma^2$ head has large gradient in some input direction, adversarial training
penalises it and prevents the model from being wildly overconfident on near-training inputs.

#### 2.3.6 The softplus reparameterisation

The paper uses $\sigma^2(x) = \log(1 + \exp(g(x))) + 10^{-6}$ for a scalar head $g(x)$ rather than
the log-variance $s(x) = \log \sigma^2(x)$ used by Kendall & Gal. Both parameterise the positive
real line; the softplus version differs in its gradient behaviour near zero:

$$
\frac{\mathrm d \mathrm{softplus}(g)}{\mathrm d g} \;=\; \sigma(g) \;=\; \frac{1}{1+\exp(-g)},
\tag{L.2.8}
$$

which saturates to 0 as $g\to-\infty$ and to 1 as $g\to+\infty$. In contrast, $\sigma^2 = \exp(s)$
has gradient $\exp(s)$, unbounded above. In practice the softplus parameterisation gives a more
bounded gradient signal for small variances and is slightly more numerically stable than $\exp(s)$
for variances near 1, but it has the same failure modes under variance collapse that Seitzer et
al. (2022) will analyse — the essential issue (division by $\sigma^2$ in the gradient on $\mu$) is
independent of which positive parameterisation is used.

#### 2.3.7 Relation to what follows

Three bridges to later papers in this review:

- **To §3 (Amini et al., 2019).** Deep ensembles requires $M$ forward passes; evidential
  regression's motivation is to recover the aleatoric/epistemic split in a *single* pass. The
  variance-decomposition formula L-2 is the target evidential regression tries to reproduce
  without ensembling.
- **To §7 (Seitzer et al., 2022).** The Gaussian NLL head (L-1) is the same object Seitzer et al.
  will diagnose as prone to variance collapse. Ensembling does *not* mitigate this: each member
  can collapse independently, and the across-member variance only captures epistemic — not
  aleatoric — uncertainty. β-NLL is therefore as relevant to deep ensembles as it is to a single
  heteroscedastic network.
- **To §8 (Meinert et al., 2023).** Deep ensembles remains, in 2023, the empirical gold standard
  for the aleatoric/epistemic split — Meinert et al. use it as the *ground truth* against which
  DER's claimed decomposition is benchmarked (and found wanting).

---

## 3. Amini, Schwarting, Soleimany & Rus (2019) — *Deep Evidential Regression*

**Venue:** NeurIPS 2020 (arXiv:1910.02600, v2 Nov 2020). **Domain:** Regression (UCI benchmarks,
monocular depth estimation on NYU-Depth-v2).

### Layer 1 — Basic introduction and concept

Deep Evidential Regression (DER) asks: *can we get the aleatoric/epistemic decomposition of
Kendall & Gal (2017, §1) and deep ensembles (Lakshminarayanan et al., 2017, §2) from a* **single
deterministic forward pass**? The authors answer yes by placing a **Normal-Inverse-Gamma (NIG)
conjugate prior** directly on the Gaussian likelihood parameters $(\mu, \sigma^2)$ and training
the network to output the four NIG hyperparameters $(\gamma, \upsilon, \alpha, \beta)$.

The philosophical shift: Bayesian deep learning places a prior on *network weights* $\mathbf W$;
evidential deep learning places a prior *directly on the likelihood parameters* $\theta = (\mu, \sigma^2)$.
Because the NIG is conjugate to the Gaussian, the posterior predictive distribution is a
Student-$t$, analytically available. Uncertainty decomposes cleanly:

- $\mathbb E[\mu] = \gamma$ — the **point prediction**.
- $\mathbb E[\sigma^2] = \beta/(\alpha-1)$ — **aleatoric uncertainty** (inherent noise).
- $\mathrm{Var}[\mu] = \beta/\{\upsilon(\alpha-1)\}$ — **epistemic uncertainty** (model ignorance).

Two training terms: (i) a Student-$t$ NLL on the marginal likelihood, and (ii) an *evidence
regulariser* $|y-\gamma|\cdot(2\upsilon+\alpha)$ that penalises confident-but-wrong predictions.

**Why this matters.** DER is the canonical "single-pass" uncertainty method — no $T$ MC dropout
samples, no $M$ ensemble members. For real-time / embedded / robotics use it was a major step.
**But:** Meinert et al. (2023, §8 of this review) will argue that DER does *not* actually
recover the aleatoric/epistemic decomposition it claims — so this entry and §8 should be read
together.

### Layer 2 — Main results and algorithm

**Generative model (Eq. A-3).** Place a Gaussian prior on $\mu$ *conditional* on $\sigma^2$, and
an Inverse-Gamma prior on $\sigma^2$:

$$
(y_1,\dots,y_N) \sim \mathcal N(\mu, \sigma^2),
\qquad
\mu \sim \mathcal N(\gamma,\, \sigma^2 \upsilon^{-1}),
\qquad
\sigma^2 \sim \Gamma^{-1}(\alpha, \beta),
\tag{A-3}
$$

with $\gamma\in\mathbb R$, $\upsilon>0$, $\alpha>1$, $\beta>0$. The joint $p(\mu,\sigma^2\mid \gamma,\upsilon,\alpha,\beta)$ is the **Normal-Inverse-Gamma** density:

$$
p(\mu,\sigma^2\mid\gamma,\upsilon,\alpha,\beta)
\;=\; \frac{\beta^\alpha\sqrt{\upsilon}}{\Gamma(\alpha)\sqrt{2\pi\sigma^2}}
\Big(\tfrac{1}{\sigma^2}\Big)^{\alpha+1}
\exp\!\Big\{-\frac{2\beta + \upsilon(\gamma-\mu)^2}{2\sigma^2}\Big\}.
\tag{A-4}
$$

**Virtual-observation interpretation.** The NIG hyperparameters are pseudo-counts: the mean is
estimated from $\upsilon$ virtual observations with sample mean $\gamma$; the variance is
estimated from $\alpha$ virtual observations with sum-of-squared-deviations $2\beta$. **Total
evidence** is defined as $\Phi = 2\upsilon + \alpha$.

**Uncertainty decomposition (Eq. A-5).** First-order moments of the NIG yield:

$$
\underbrace{\mathbb E[\mu] \;=\; \gamma}_{\text{prediction}},
\qquad
\underbrace{\mathbb E[\sigma^2] \;=\; \frac{\beta}{\alpha-1}}_{\text{aleatoric}},
\qquad
\underbrace{\mathrm{Var}[\mu] \;=\; \frac{\beta}{\upsilon(\alpha-1)}}_{\text{epistemic}}.
\tag{A-5}
$$

Note $\mathrm{Var}[\mu] = \mathbb E[\sigma^2] / \upsilon$ — the epistemic variance is the
aleatoric variance divided by the virtual-observation count $\upsilon$. More virtual observations
⇒ tighter posterior on $\mu$.

**Marginal likelihood = Student-$t$ (Eq. A-7).** Integrating $(\mu,\sigma^2)$ out of the
Gaussian likelihood under the NIG prior yields, in closed form,

$$
p(y_i\mid\mathbf m) \;=\; \mathrm{St}\!\left(y_i;\, \gamma,\, \frac{\beta(1+\upsilon)}{\upsilon\,\alpha},\, 2\alpha\right),
\tag{A-7}
$$

a Student-$t$ with location $\gamma$, scale $\beta(1+\upsilon)/(\upsilon\alpha)$, and $2\alpha$
degrees of freedom.

**NLL loss (Eq. A-8).** With $\Omega := 2\beta(1+\upsilon)$,

$$
\mathcal L_i^{\mathrm{NLL}}(\mathbf w) \;=\; \tfrac{1}{2}\log\!\Big(\tfrac{\pi}{\upsilon}\Big)
\;-\; \alpha\log\Omega \;+\; \Big(\alpha+\tfrac{1}{2}\Big)\log\!\big((y_i-\gamma)^2\upsilon + \Omega\big)
\;+\; \log\!\Big(\tfrac{\Gamma(\alpha)}{\Gamma(\alpha+1/2)}\Big).
\tag{A-8}
$$

**Evidence regulariser (Eq. A-9).** To prevent the network from placing *confident* evidence on
*wrong* predictions, regularise by error-weighted total evidence:

$$
\mathcal L_i^{\mathrm R}(\mathbf w) \;=\; |y_i - \mathbb E[\mu_i]| \cdot \Phi
\;=\; |y_i - \gamma| \cdot (2\upsilon + \alpha).
\tag{A-9}
$$

**Total loss (Eq. A-10).** $\mathcal L_i = \mathcal L_i^{\mathrm{NLL}} + \lambda\,\mathcal L_i^{\mathrm R}$,
with $\lambda$ trading off fit vs. uncertainty inflation.

**Implementation.** Four output neurons per target: linear activation for $\gamma$, softplus
(with $+1$ for $\alpha$) for $(\upsilon,\alpha,\beta)$.

**Empirical highlights.**

- **UCI benchmarks (Table 1).** DER matches or beats dropout and deep ensembles on NLL on 9/9
  datasets, with inference ~3–4× faster than dropout ($n=5$) and ~3.5–4× faster than $M=5$ ensembles.
- **NYU depth estimation.** DER's calibration error (0.033) beats ensembles (0.048) and dropout
  (0.126), with ~10× fewer parameters than a 10-ensemble and ~40× faster inference than dropout
  with 50 MC samples.
- **OOD detection (ApolloScape driving images).** Entropy on OOD inflates competitively with
  ensembles, *without ever seeing OOD data during training* — a key advantage over Prior
  Networks (Malinin & Gales, 2018), which require OOD supervision.
- **Adversarial robustness.** Under FGSM perturbation, evidential entropy rises monotonically
  with perturbation scale.

### Layer 3 — Graduate-level deep dive with full derivations

#### 3.3.1 Why NIG is the right prior: conjugacy

The Gaussian likelihood with unknown mean and variance is a two-parameter exponential family.
Its **conjugate prior** — the prior class closed under Bayesian updating — is precisely the NIG.
Conjugacy is the reason DER can collapse inference into a single forward pass: the posterior
after seeing $N$ data points has the same NIG *functional form*, only with updated
hyperparameters. We will not derive the update rules here (they are standard; see Bishop 2006
Ch. 2), but they motivate the *virtual-observation* reading: the NIG hyperparameters behave
algebraically like sufficient statistics for an imaginary dataset of size $\upsilon$ / $2\alpha$.

Two structural facts follow:

1. **The NIG is a distribution over a pair** $(\mu, \sigma^2)$ — it is a *higher-order*
   distribution, i.e. a distribution over parameters of another distribution. Sampling a single
   $(\mu_j, \sigma_j^2)$ from the NIG gives a lower-order Gaussian likelihood
   $\mathcal N(\mu_j, \sigma_j^2)$, and sampling from *that* gives a data point. The paper's
   Fig. 2 illustrates this hierarchy.
2. **Higher evidence ⇒ tighter NIG**, by construction. As $\upsilon, \alpha \to \infty$ the NIG
   concentrates on a single $(\mu, \sigma^2)$ and the Bayesian posterior over parameters
   collapses to a point estimate.

#### 3.3.2 Derivation of the moments → Eq. A-5

**Mean of $\mu$.** From $\mu\mid\sigma^2 \sim \mathcal N(\gamma, \sigma^2/\upsilon)$ and the
tower rule,

$$
\mathbb E[\mu] \;=\; \mathbb E_{\sigma^2}\!\big[\mathbb E(\mu\mid\sigma^2)\big] \;=\; \mathbb E_{\sigma^2}[\gamma] \;=\; \gamma.
\tag{A.1}
$$

**Mean of $\sigma^2$** (supplementary S9). For $\sigma^2\sim\Gamma^{-1}(\alpha,\beta)$, the density is

$$
p(\sigma^2) \;=\; \frac{\beta^\alpha}{\Gamma(\alpha)}\,(\sigma^2)^{-\alpha-1}\exp(-\beta/\sigma^2),\qquad \sigma^2>0.
\tag{A.2}
$$

Change variables to $u = \beta/\sigma^2$ so $\sigma^2 = \beta/u$ and $\mathrm d\sigma^2 = -\beta/u^2\,\mathrm d u$. Then

$$
\mathbb E[\sigma^2] \;=\; \int_0^\infty \sigma^2\, p(\sigma^2)\,\mathrm d\sigma^2
\;=\; \frac{\beta^\alpha}{\Gamma(\alpha)} \int_0^\infty (\sigma^2)^{-\alpha}\exp(-\beta/\sigma^2)\,\mathrm d\sigma^2.
\tag{A.3}
$$

Substituting $u=\beta/\sigma^2$ and simplifying,

$$
\mathbb E[\sigma^2] \;=\; \frac{\beta}{\Gamma(\alpha)} \int_0^\infty u^{\alpha-2}\exp(-u)\,\mathrm d u
\;=\; \frac{\beta\,\Gamma(\alpha-1)}{\Gamma(\alpha)} \;=\; \frac{\beta}{\alpha-1},
\tag{A-5 aleatoric}
$$

using $\Gamma(\alpha) = (\alpha-1)\Gamma(\alpha-1)$, valid for $\alpha>1$.

**Variance of $\mu$** (supplementary S10–S13). Use the law of total variance:

$$
\mathrm{Var}[\mu]
\;=\; \mathbb E_{\sigma^2}\!\big[\mathrm{Var}(\mu\mid\sigma^2)\big] \;+\; \mathrm{Var}_{\sigma^2}\!\big[\mathbb E(\mu\mid\sigma^2)\big].
\tag{A.4}
$$

The first term is $\mathbb E_{\sigma^2}[\sigma^2/\upsilon] = \mathbb E[\sigma^2]/\upsilon = \beta/\{\upsilon(\alpha-1)\}$.
The second term vanishes because $\mathbb E(\mu\mid\sigma^2) = \gamma$ is constant in $\sigma^2$. Therefore

$$
\mathrm{Var}[\mu] \;=\; \frac{\beta}{\upsilon(\alpha-1)}.
\tag{A-5 epistemic}
$$

Note this derivation makes the *structural* reason for the $\upsilon$ factor clear: $\upsilon$
enters only because $\mu\mid\sigma^2 \sim \mathcal N(\gamma, \sigma^2/\upsilon)$ sets the scale of
the inner variance. This coupling between $\mu$ and $\sigma^2$ is the hinge on which the
critique of Meinert et al. (2023, §8) will turn — they will argue that because the NIG *mixes*
the two uncertainties through $\upsilon$, the claimed "decomposition" is not a genuine
separation but an artefact of the parameterisation.

#### 3.3.3 Derivation of the marginal likelihood → Eq. A-7

The goal is to evaluate

$$
p(y_i\mid\mathbf m) \;=\; \int_0^\infty \!\!\int_{-\infty}^\infty p(y_i\mid\mu,\sigma^2)\,p(\mu,\sigma^2\mid\mathbf m)\,\mathrm d\mu\,\mathrm d\sigma^2.
\tag{A-6}
$$

The Gaussian likelihood and the $\mu$-part of the NIG combine into a Gaussian in $\mu$:

$$
p(y_i\mid\mu,\sigma^2)\,p(\mu\mid\sigma^2,\gamma,\upsilon)
\;\propto\; \exp\!\Big\{-\frac{(y_i-\mu)^2}{2\sigma^2} - \frac{\upsilon(\mu-\gamma)^2}{2\sigma^2}\Big\}.
\tag{A.5}
$$

Expand the exponent and complete the square in $\mu$. Let $\mu^\star = (y_i + \upsilon\gamma)/(1+\upsilon)$
be the posterior mean. After algebra (supplementary S20 captures the intermediate):

$$
\text{exponent} \;=\; -\frac{(1+\upsilon)\,(\mu-\mu^\star)^2}{2\sigma^2} \;-\; \frac{\upsilon(y_i-\gamma)^2}{2(1+\upsilon)\sigma^2}.
\tag{A.6}
$$

The $\mu$-integral is Gaussian and yields $\sigma\sqrt{2\pi/(1+\upsilon)}$. The remaining integral in $\sigma^2$ has the form of an unnormalised inverse-gamma:

$$
p(y_i\mid\mathbf m)
\;\propto\; \int_0^\infty (\sigma^2)^{-\alpha-3/2}\exp\!\Big\{-\frac{1}{\sigma^2}\Big[\beta + \tfrac{\upsilon(y_i-\gamma)^2}{2(1+\upsilon)}\Big]\Big\}\,\mathrm d\sigma^2.
\tag{A.7}
$$

Substituting $u = [\beta + \upsilon(y_i-\gamma)^2/(2(1+\upsilon))]/\sigma^2$ reduces this to a gamma integral:

$$
p(y_i\mid\mathbf m) \;\propto\; \Big[2\beta(1+\upsilon) + \upsilon(y_i-\gamma)^2\Big]^{-(\alpha+1/2)} \cdot \Gamma(\alpha+\tfrac{1}{2}).
\tag{A.8}
$$

Collecting all constants (supplementary S22):

$$
p(y_i\mid\mathbf m)
\;=\; \frac{\Gamma(\alpha+1/2)}{\Gamma(\alpha)}\,
\sqrt{\frac{\upsilon}{\pi}}\,
\big(2\beta(1+\upsilon)\big)^{\alpha}\,
\big[\upsilon(y_i-\gamma)^2 + 2\beta(1+\upsilon)\big]^{-(\alpha+1/2)}.
\tag{A.9}
$$

Comparing to the standard Student-$t$ density

$$
\mathrm{St}(y;\mu_{\rm St},\sigma_{\rm St}^2,\nu) \;=\; \frac{\Gamma((\nu+1)/2)}{\Gamma(\nu/2)\sqrt{\nu\pi}\,\sigma_{\rm St}}
\Big[1 + \frac{(y-\mu_{\rm St})^2}{\nu\sigma_{\rm St}^2}\Big]^{-(\nu+1)/2},
\tag{A.10}
$$

matching exponents identifies $\nu = 2\alpha$, $\mu_{\rm St} = \gamma$, and $\sigma_{\rm St}^2 = \beta(1+\upsilon)/(\upsilon\alpha)$ — exactly A-7.

#### 3.3.4 Student-$t$ NLL → Eq. A-8

Take the negative log of A-7. Using the Student-$t$ density in the
$\mathrm{St}(y;\mu_{\rm St}, \sigma_{\rm St}^2, \nu)$ form with $\nu = 2\alpha$:

$$
-\log\mathrm{St}(y_i;\gamma,\sigma_{\rm St}^2,2\alpha)
\;=\; \tfrac{1}{2}\log(2\alpha\pi\sigma_{\rm St}^2) - \log\Gamma(\alpha+\tfrac{1}{2}) + \log\Gamma(\alpha)
\;+\; \big(\alpha+\tfrac{1}{2}\big)\log\!\Big(1 + \frac{(y_i-\gamma)^2}{2\alpha\sigma_{\rm St}^2}\Big).
\tag{A.11}
$$

Substituting $\sigma_{\rm St}^2 = \beta(1+\upsilon)/(\upsilon\alpha)$, writing $\Omega := 2\beta(1+\upsilon)$,
and simplifying (supplementary S26) gives the paper's form:

$$
\mathcal L_i^{\mathrm{NLL}} \;=\; \tfrac{1}{2}\log(\pi/\upsilon) - \alpha\log\Omega + (\alpha+\tfrac{1}{2})\log\!\big((y_i-\gamma)^2\upsilon + \Omega\big) + \log\!\Big(\tfrac{\Gamma(\alpha)}{\Gamma(\alpha+1/2)}\Big).
\tag{A-8}
$$

The two $\Gamma$-functions combine into a $\log\Gamma$ ratio that is smooth in $\alpha$ (it is
essentially a regularised digamma at large $\alpha$) — tractable via `scipy.special.gammaln`.

#### 3.3.5 The evidence regulariser: what it is and what it is not → Eq. A-9

The regulariser $|y_i-\gamma|\cdot(2\upsilon+\alpha)$ is *not* a KL divergence to a
non-informative prior — that would be the natural Bayesian regulariser, but the paper shows in §
S1.3 that the KL between any NIG and a *zero-evidence* NIG is **undefined**. Specifically, for a
zero-evidence NIG, either $\upsilon=0$ (causing $\log(\upsilon)$ terms to diverge) or $\alpha=0$
(causing $\log\Gamma(\alpha)$ to diverge). One can soften to an $\epsilon$-evidence prior
(Eq. S31–S32), but the result is hypersensitive to $\epsilon$.

Instead, the authors propose an **ad hoc evidence-weighted residual** regulariser: if the
network places high evidence on a prediction $\gamma$ that differs from $y_i$, scale the penalty
by both the error magnitude and the total evidence. This has two properties:

- **Correct predictions are not penalised** even at high evidence.
- **Wrong predictions with high evidence are strongly penalised**, pushing the network to
  *reduce* $\upsilon,\alpha$ (inflate uncertainty) whenever it is systematically wrong.

It does *not*, however, directly penalise a model for being overconfident on in-distribution
data when the prediction happens to be correct by luck — this is the root of Meinert et al.'s
(2023) critique that the regulariser trades epistemic for aleatoric rather than reducing total
uncertainty appropriately.

**Gradient structure.** Differentiate A-9 with respect to $(\upsilon, \alpha)$:

$$
\frac{\partial \mathcal L_i^{\mathrm R}}{\partial \upsilon} = 2|y_i-\gamma|,
\qquad
\frac{\partial \mathcal L_i^{\mathrm R}}{\partial \alpha} = |y_i-\gamma|.
\tag{A.12}
$$

So $\upsilon$ receives *twice* the pressure that $\alpha$ does — a fact that is easy to miss but
which structurally biases the trained model toward adjusting $\upsilon$ (and thus epistemic
uncertainty, via $\beta/\{\upsilon(\alpha-1)\}$) more aggressively than $\alpha$ (which controls
aleatoric via $\beta/(\alpha-1)$).

#### 3.3.6 The parameter constraints and why they matter

Recall the NIG requires $\upsilon>0$, $\alpha>1$, $\beta>0$. The paper enforces:

$$
\upsilon = \mathrm{softplus}(g_\upsilon), \qquad \alpha = 1 + \mathrm{softplus}(g_\alpha), \qquad \beta = \mathrm{softplus}(g_\beta).
\tag{A.13}
$$

The constraint $\alpha > 1$ is not cosmetic — it is what makes the aleatoric uncertainty
$\mathbb E[\sigma^2] = \beta/(\alpha-1)$ *finite*. Near $\alpha\to 1^+$, $\mathbb E[\sigma^2]\to\infty$,
which is the NIG's way of expressing "I have essentially no evidence about the variance". Hitting
this regime during training is numerically catastrophic because the NLL loss contains
$\log\Gamma(\alpha)$ which blows up. In practice, implementations clamp $\alpha$ away from 1 by a
safety margin (the reference implementation uses $\alpha \ge 1.01$).

#### 3.3.7 Why DER is efficient but controversial — forward pointers to §8

DER's *efficiency* claim is airtight: one forward pass, one loss. Its *correctness* claim — that
$\beta/(\alpha-1)$ is genuinely aleatoric and $\beta/\{\upsilon(\alpha-1)\}$ is genuinely epistemic
— rests on two premises that Meinert et al. (2023, §8 of this review) will challenge:

1. **Decomposition identifiability.** Because the two claimed uncertainties share both $\beta$ and
   $(\alpha-1)$ in the denominator, they are perfectly correlated up to the factor $1/\upsilon$.
   Meinert et al. argue this means DER cannot actually *separate* the two sources — it can only
   rescale a single underlying uncertainty signal.
2. **Regulariser effect.** The evidence regulariser (A-9) incentivises increasing
   $|y-\gamma|\cdot(2\upsilon+\alpha)$ — but the network can satisfy this by shrinking
   $\upsilon,\alpha$ and paying only a fit-term cost (via the NLL log-terms). In practice this
   means the model's "epistemic" uncertainty is largely a *learned proxy* for an error-detection
   signal, not a Bayesian posterior variance.

Readers should therefore *not* take DER's decomposition at face value without consulting §8.
That said, the paper's Table 1 / Figure 4 empirical results — especially on OOD detection and
calibration — are robust and reproducible; what is disputed is the *interpretation*, not the
numerics.

#### 3.3.8 Bridges in both directions

- **To §1 (Kendall & Gal).** Kendall & Gal's total variance (K-9) decomposes into
  model-variance + mean-variance via MC dropout; DER achieves the same decomposition
  *analytically* via the NIG moments. Both share the limitation that the claimed "aleatoric"
  term depends on the specific parametric form of the head.
- **To §2 (Lakshminarayanan et al.).** Deep ensembles and DER are the two canonical ways to get
  an epistemic signal: $M$ samples vs. one NIG. Ensembles remain the empirical gold standard;
  DER is what you use when sampling is prohibitive.
- **To §7 (Seitzer et al., β-NLL).** DER's Gaussian-likelihood parameterisation still contains
  the mean-gradient-divided-by-variance pattern (L.1.9 in the Kendall & Gal entry), so DER is
  also susceptible to variance collapse — with the extra twist that both $\upsilon$ and $\alpha$
  can collapse, not just $\sigma^2$.
- **To §8 (Meinert et al.).** Required companion reading; treat §8 as the rebuttal.

---

## 4. Detlefsen, Jørgensen & Hauberg (2019) — *Reliable Training and Estimation of Variance Networks*

**Venue:** NeurIPS 2019 (arXiv:1906.03260). **Domain:** Regression (UCI benchmarks, active
learning, VAEs).

> **Attribution note.** The initial lit-review (perceptual_noise_lit_review.md §7.1) cited this
> paper as "Skafte, Jørgensen & Hauberg (2019)" following the (erroneous) convention in some
> secondary sources. The correct first author is **Detlefsen** (Nicki S. Detlefsen); Skafte is
> the same person post-name-change in later works. The Phase 1 doc should be updated on the next
> revision.

### Layer 1 — Basic introduction and concept

This paper is the first systematic diagnosis of *why heteroscedastic variance networks* (the
Kendall & Gal / Lakshminarayanan head) under-estimate variance in practice — and a prescription
for fixing them. The central observation is deceptively simple:

> *Variance estimation is not the same task as mean estimation, so the tools that work for mean
> fitting do not necessarily generalise.*

The paper's Fig. 1 is the iconic demonstration: a two-headed network fit to the toy
heteroscedastic dataset $y = x\sin(x) + 0.3\epsilon_1 + 0.3\,x\,\epsilon_2$ has an almost-perfect
mean, but its variance head (i) severely *under-estimates* noise everywhere within the data
support, and (ii) does not *inflate* outside the data support where the mean is uncertain. This
is a general failure mode, and the paper proposes four complementary fixes bundled into the
**"Combined"** method:

- **(LS) Locality sampler.** A mini-batching scheme that samples $k$-nearest-neighbours so each
  batch has enough nearby points for a meaningful variance estimate — paired with
  Horvitz–Thompson reweighting to keep the gradient *unbiased*.
- **(MV) Mean-variance split training.** Never update mean and variance simultaneously; freeze
  one while fitting the other, alternating. This mirrors the Gaussian MLE identity that $\hat\sigma^2_{\rm MLE} = (y-\hat\mu)^2$ only has a legitimate value when $\hat\mu$ is held fixed.
- **(IG) Inverse-Gamma prior on $\sigma^2$.** Model $\sigma^2\sim\Gamma^{-1}(\alpha(x), \beta(x))$
  rather than predicting $\sigma^2$ directly. Marginalising gives a **Student-$t$** predictive
  distribution — the same object Amini et al. (2019, §3) arrive at from a different direction.
- **(EX) Extrapolation architecture.** Explicitly interpolate predicted variance toward a
  prior value $\eta$ as distance from training data increases — mimicking posterior GP behaviour.

Together these four fixes turn a catastrophic variance head into a competitive one. The paper's
significance for precision modulation is that it **explains in detail why naive heteroscedastic
heads fail** and enumerates the specific training-dynamics defects — each of which the GridWorld
Pain Phase 3 precision head would inherit unless actively mitigated.

### Layer 2 — Main results and algorithm

**Local likelihood diagnostic (Eq. D-1).** The core theoretical lens. Any continuous regression
assumes $\sigma^2(x)$ varies smoothly with $x$, so the local likelihood

$$
\log\tilde p_\theta(y_i\mid x_i) \;=\; \sum_{j=1}^N w_j(x_i)\,\log p_\theta(y_j\mid x_j)
\tag{D-1}
$$

with a kernel $w_j(x_i)$ that decays in $\|x_j-x_i\|$ captures the local density of information.
Uniform mini-batches (i.e. $w_j = 1_{j=i}$) mean that if a point $x_i$ is isolated in a batch, no
other point contributes to $\nabla_{\theta_{\sigma^2}}$ at $x_i$, so the variance update points
toward the MLE of a *single point*, which is zero. This is the **structural cause** of variance
collapse in minibatched heteroscedastic training.

**Horvitz-Thompson adjusted estimator (Eq. D-2).** The locality sampler uses $k$-nearest-neighbour
minibatches (sample $m$ seed points uniformly, then sample $n$ among each seed's $k$-NN).
Per-sample reweighting makes the stochastic gradient unbiased:

$$
\sum_{i=1}^N \Big[-\tfrac{1}{2}\log\sigma^2(x_i) - \frac{(y_i-\mu(x_i))^2}{2\sigma^2(x_i)}\Big]
\;\approx\; \sum_{x_j\in\mathcal O}\frac{1}{\pi_j}\Big[-\tfrac{1}{2}\log\sigma^2(x_j) - \frac{(y_j-\mu(x_j))^2}{2\sigma^2(x_j)}\Big],
\tag{D-2}
$$

with inclusion probability

$$
\pi_j \;=\; \frac{m}{N}\sum_{i=1}^N \frac{n}{k}\,\mathbb 1\big[j\in\mathcal O_k(i)\big],
\tag{D-3}
$$

where $\mathcal O_k(i)$ is the set of $k$-NN of $x_i$.

**IG prior → Student-$t$ predictive (Eq. D-4).** Integrating $\sigma^2 \sim \Gamma^{-1}(\alpha,\beta)$
out of the Gaussian likelihood:

$$
\log p_\theta(y_i) \;=\; \log\int \mathcal N(y_i\mid\mu_i,\sigma_i^2)\,\Gamma^{-1}(\sigma_i^2\mid\alpha_i,\beta_i)\,\mathrm d\sigma_i^2 \;=\; \log t_{\mu_i,\alpha_i,\beta_i}(y_i).
\tag{D-4}
$$

The explicit Student-$t$ density is derived in the paper's Appendix B (reproduced in §4.3.3 below).

**Extrapolation architecture (Eq. D-5).** With inducing points $\{c_i\}_{i=1}^L$ and minimum
distance $\delta(x_0) = \min_i \|c_i - x_0\|$,

$$
\hat\sigma^2(x_0) \;=\; \big(1 - \nu(\delta(x_0))\big)\,\hat\sigma^2_\theta \;+\; \eta\,\nu(\delta(x_0)),
\tag{D-5}
$$

where $\nu$ is a sigmoid warping $[0,\infty)\to[0,1]$. As $\delta\to\infty$, $\hat\sigma^2\to\eta$;
near data, $\hat\sigma^2\to\hat\sigma^2_\theta$. This recovers the high-entropy extrapolation
behaviour of posterior GPs with stationary kernels.

**Empirical highlights.**

- **UCI regression (Table 1).** Combined wins on test-set NLL on 10/13 datasets; the baseline
  two-headed NN is worst everywhere, confirming the paper's thesis.
- **Ablation (Fig. 5, Fig. 11).** LS and IG individually account for most of the gain; no
  component hurts. RMSE is essentially unchanged across variants — which proves the Combined
  method is fixing the *variance* head without perturbing the mean fit.
- **Active learning (Fig. 6, Fig. 12).** Better uncertainty ⇒ faster active learning.
- **Weather-data calibration benchmark.** Combined achieves the lowest mean-absolute error
  vs. the "true" empirical variance (0.016 vs. 0.0186 for ensembles, 0.0184 for GPs).
- **VAE application.** Replacing the decoder-variance training with the Combined method
  improves ELBO and test log-likelihood on MNIST, FashionMNIST, CIFAR10, SVHN.

### Layer 3 — Graduate-level deep dive with full derivations

#### 4.3.1 Why a single-point variance MLE doesn't exist

The Gaussian MLE identities are

$$
\hat\mu \;=\; \tfrac{1}{N}\sum_i y_i,\qquad \hat\sigma^2 \;=\; \tfrac{1}{N-1}\sum_i (y_i-\hat\mu)^2.
\tag{D.1}
$$

With $N=1$: $\hat\mu = y_1$ (trivial), but $\hat\sigma^2 = 0/0$ — undefined. Alternatively, with
$\hat\mu$ held *fixed* (i.e. not a free parameter), the variance MLE becomes $\hat\sigma^2(x_i) = (y_i-\mu(x_i))^2$ — defined but *maximally biased* because it is a single squared residual with no
averaging.

In a standard mini-batch setting where one sample $x_i$ happens to be isolated (no other batch
points near it), $\nabla_{\theta_{\sigma^2}}$ at that sample behaves exactly as if $N=1$:
the gradient points toward the zero-variance MLE. Across many batches, this induces a
**systematic downward drift** in $\sigma^2$ — empirically observed as variance collapse (the
paper's Fig. 1 and Seitzer et al. 2022 both catch this).

The **log-Gaussian-NLL gradient analysis** (the same L.1.9–L.1.10 from the Kendall & Gal entry)
compounds this: since the mean gradient is $\propto \exp(-s)$ and $s\to-\infty$ locally when a
point is isolated, *the mean update rate at that point blows up while the variance update vanishes
toward zero-variance MLE*. The iteration is unstable.

#### 4.3.2 The locality sampler: why it works and why HT weighting is necessary

Let $\mathcal O$ be a locality-sampled mini-batch. The naive stochastic gradient
$\hat g_{\rm naive} = \sum_{j\in\mathcal O} \nabla_\theta \ell_j$ is a biased estimator of the
full-data gradient $g = \sum_{j=1}^N \nabla_\theta \ell_j$, because the inclusion probabilities
$\pi_j$ are *non-uniform* (points with more neighbours in their $k$-NN cliques get over-sampled).

**Horvitz–Thompson (1952)** corrects this. The HT estimator is

$$
\hat g_{\rm HT} \;=\; \sum_{j\in\mathcal O} \frac{1}{\pi_j}\,\nabla_\theta \ell_j,
\tag{D.2}
$$

and it is *exactly unbiased* as long as every unit has $\pi_j > 0$. To see this,

$$
\mathbb E[\hat g_{\rm HT}] \;=\; \sum_{j=1}^N \mathbb E[\mathbb 1_{j\in\mathcal O}]\cdot\frac{1}{\pi_j}\nabla_\theta\ell_j
\;=\; \sum_{j=1}^N \pi_j\cdot\frac{1}{\pi_j}\nabla_\theta\ell_j \;=\; g.
\tag{D.3}
$$

**Deriving $\pi_j$ (Eq. D-3).** Consider the sampling: (a) pick $m$ seeds uniformly without
replacement → probability $m/N$ each; (b) within each seed's $k$-NN, pick $n$ uniformly → probability $n/k$ each. So the probability that $x_j$ is included *given that $x_i$ was a seed* is $(n/k)\mathbb 1[j\in\mathcal O_k(i)]$. Aggregating over all possible seeds and using inclusion-exclusion (approximating disjointly, as in the paper):

$$
\pi_j \;\approx\; \frac{m}{N}\sum_{i=1}^N \frac{n}{k}\,\mathbb 1\big[j\in\mathcal O_k(i)\big] \;=\; \frac{m}{N}\cdot\frac{n}{k}\cdot\#\{i : j\in\mathcal O_k(i)\}.
\tag{D-3}
$$

The count $\#\{i : j\in\mathcal O_k(i)\}$ is the **$k$-reverse-nearest-neighbour count** of $x_j$
— the number of points for which $x_j$ is in the $k$-NN. This requires precomputing the full
$N\times N$ distance matrix ($O(N^2 D)$), but only once.

**Why this helps variance specifically.** In the standard mini-batch, the gradient at the variance
head is a *pointwise* MLE-like signal. In the locality batch, *multiple* nearby points all contribute to $\nabla_{\theta_{\sigma^2}}$ at each sample via the smoothness of $\sigma^2(x)$. This is the "degrees of freedom" argument in the paper (footnote 4): the Gamma distribution of squared residuals under Gaussian likelihood gains informative structure once you have multiple correlated residuals rather than one.

The supplementary (Fig. 10) empirically confirms this: the **sparsity index** of the variance gradient (fraction of zero coordinates) rises from ~0.4 (uniform batch) to ~0.7 (locality sampler), and the **gradient variance** drops by an order of magnitude.

#### 4.3.3 Mean-variance split training: a theoretical argument

MV training alternates optimisation of $\theta_\mu$ with $\theta_{\sigma^2}$ held fixed, and vice versa. The rationale is directly the *MLE existence* argument (§4.3.1): $\hat\sigma^2$ exists and is well-defined *conditional on* $\mu$ being known. Since alternating fixes one, each half-step has legitimate MLE semantics.

Formally, MV is block coordinate ascent on the joint log-likelihood
$\mathcal L(\theta_\mu, \theta_{\sigma^2})$. Under standard assumptions (concavity in each block
given the other), it converges to a local maximum. The non-trivial point is that the joint
$\mathcal L$ is *not* jointly convex even in the one-dimensional linear case, so joint SGD has no
such convergence guarantee. Empirically (Fig. 5 ablation), MV improves NLL on all tested datasets
without hurting RMSE.

**Contrast with Kendall & Gal's K-8.** K-8 does joint training of $(\mu, s = \log\sigma^2)$. The
feedback loop $\partial\mathcal L/\partial s \cdot \partial\mathcal L/\partial\mu$ (L.1.9 × L.1.10
from the Kendall & Gal entry) can make this joint training unstable. MV decouples the loop.

#### 4.3.4 The inverse-Gamma prior → Student-$t$ derivation (Appendix B)

The derivation proceeds as follows. With $\sigma^2\sim\Gamma^{-1}(\alpha,\beta)$, equivalently
$1/\sigma^2 \sim \Gamma(\alpha,\beta)$ (shape–rate form, $\mathbb E[1/\sigma^2] = \alpha/\beta$), marginalise:

$$
p(y\mid\mu,\alpha,\beta) \;=\; \int_0^\infty \mathcal N(y\mid\mu,\sigma^2)\,\frac{\beta^\alpha}{\Gamma(\alpha)}(\sigma^2)^{-(\alpha+1)}\exp(-\beta/\sigma^2)\,\mathrm d\sigma^2.
\tag{D.4}
$$

Substitute $u = 1/\sigma^2$ so $\mathrm d\sigma^2 = -\mathrm d u / u^2$:

$$
p(y\mid\mu,\alpha,\beta) \;=\; \frac{\beta^\alpha}{\Gamma(\alpha)\sqrt{2\pi}}\int_0^\infty u^{\alpha-1/2}\exp\!\Big\{-u\cdot\Big[\beta + \tfrac{1}{2}(y-\mu)^2\Big]\Big\}\,\mathrm d u.
\tag{D.5}
$$

The integrand is an unnormalised $\Gamma(\alpha+\tfrac{1}{2}, \beta + \tfrac{1}{2}(y-\mu)^2)$, so

$$
p(y\mid\mu,\alpha,\beta) \;=\; \frac{\beta^\alpha}{\Gamma(\alpha)\sqrt{2\pi}}\cdot\frac{\Gamma(\alpha+\tfrac{1}{2})}{\big[\beta + \tfrac{1}{2}(y-\mu)^2\big]^{\alpha+1/2}}.
\tag{D.6}
$$

This is a *location-scaled* Student-$t$: with $\nu := 2\alpha$ degrees of freedom, location $\mu$,
and scale $\sigma_{\rm St}^2 := \beta/\alpha$, we can rewrite D.6 as

$$
p(y\mid\mu,\alpha,\beta) \;=\; \frac{\Gamma(\alpha+1/2)}{\Gamma(\alpha)\sqrt{2\pi\alpha\sigma_{\rm St}^2}}\Big[1 + \frac{(y-\mu)^2}{2\alpha\sigma_{\rm St}^2}\Big]^{-(\alpha+1/2)}.
\tag{D.7}
$$

**Cross-reference.** This is the exact same marginalisation that Amini et al. (2019, §3.3.3)
perform for DER — except Amini marginalises over *both* $\mu$ and $\sigma^2$ (using an extra Gaussian prior on $\mu$), while Detlefsen et al. marginalise only over $\sigma^2$ and keep $\mu$ as a point-estimate network output. So the Detlefsen Student-$t$ has $\nu = 2\alpha$ degrees of freedom, whereas DER's Student-$t$ also has $\nu = 2\alpha$. The two methods converge on the same predictive distribution but with different interpretations of the hyperparameters.

**Why Student-$t$ helps.** Heavy tails mean large residuals are less penalised per unit log-density. Concretely, the Student-$t$ NLL at residual $r = y-\mu$ is $\propto (\alpha+1/2)\log(1 + r^2/(2\alpha\sigma_{\rm St}^2))$, which grows *logarithmically* in $r^2$ — not linearly as the Gaussian NLL does. Robust to outliers by construction.

#### 4.3.5 The extrapolation architecture: recovering GP behaviour

The observation is geometric: for a posterior GP with stationary kernel $k(x,x') = k(\|x-x'\|)$, the posterior variance at $x_0$ satisfies

$$
\sigma^2_{\rm GP}(x_0) \to k(0) \quad\text{as } \min_i \|x_i - x_0\|\to\infty.
\tag{D.8}
$$

That is, far from training data, posterior variance reverts to the prior variance $k(0)$ — a *constant*. Detlefsen's D-5 is an explicit neural-network re-enactment of this property: blend the learned variance $\hat\sigma^2_\theta$ with a prior $\eta$ using a distance-to-inducing-points soft gate $\nu$. Specifically,

$$
\nu(x) \;=\; \mathrm{sigmoid}\big((x+a)/\gamma\big),\qquad a \approx -6.9077\,\gamma,
\tag{D.9}
$$

chosen so that $\nu(0) \approx 0$ (no blending when right at an inducing point) and $\nu(x)\to 1$
as $x\to\infty$. Both $\gamma$ (length scale) and the inducing points $\{c_i\}$ are **learned during training**.

**Connection to sparse GPs.** The inducing points $c_i$ play an analogous role to the pseudo-inputs of sparse GPs (Snelson & Ghahramani, 2006). The paper initialises them via $k$-means on the training data, then jointly optimises — a lightweight version of sparse-GP inducing-point optimisation.

**What D-5 does *not* do.** It does not change the in-distribution variance estimates — those still come from the heteroscedastic NN head. So D-5 is purely an *extrapolation* fix, orthogonal to LS / MV / IG which fix in-distribution training.

#### 4.3.6 Bridges to the rest of the review

- **To §1 (Kendall & Gal).** Detlefsen et al. explicitly diagnose the training failure mode that
  Kendall & Gal's K-8 loss induces, and show it is a *batching* problem as much as a loss problem.
- **To §2 (Deep ensembles).** Locality sampling is *complementary* to ensembling: each ensemble
  member can independently benefit from locality batches. The paper explicitly notes this in its
  discussion.
- **To §3 (DER).** The IG prior is structurally identical to DER's NIG prior; Detlefsen's
  marginalisation is a special case of DER's with the $\mu$-prior removed. Bridge: if you start
  from Detlefsen's Combined and *add* a Gaussian prior on $\mu$, you recover DER — with the same
  caveats Meinert et al. (2023, §8) will raise about decomposition identifiability.
- **To §7 (Seitzer β-NLL).** β-NLL is a *loss-level* fix for the same failure mode Detlefsen et
  al. diagnose at the *batching level*. The two are additive: β-NLL + locality sampler is
  stronger than either alone.

---

## 5. Romano, Patterson & Candès (2019) — *Conformalized Quantile Regression*

**Venue:** NeurIPS 2019 (arXiv:1905.03222). **Domain:** Distribution-free prediction intervals
for regression (11 UCI benchmarks).

### Layer 1 — Basic introduction and concept

CQR is philosophically different from the six papers reviewed above. Where Kendall & Gal,
Lakshminarayanan, Amini, Detlefsen, and (later) Stirn & Knowles / Seitzer / Meinert all ask
*how should the network parameterise its own uncertainty?*, Romano et al. ask a sharper question:
**can we obtain a prediction interval with a finite-sample coverage guarantee, regardless of
which neural network we use?**

The answer — yes, via **conformal prediction** — is built on two ingredients:

1. **Quantile regression** (Koenker & Bassett, 1978). Train two networks to output the
   $\alpha_{\rm lo}$ and $\alpha_{\rm hi}$ conditional quantiles $\hat q_{\alpha_{\rm lo}}(x),
   \hat q_{\alpha_{\rm hi}}(x)$ of $Y\mid X$ via the *pinball loss*. The plug-in interval
   $\hat C(x) = [\hat q_{\alpha_{\rm lo}}(x), \hat q_{\alpha_{\rm hi}}(x)]$ is *adaptive* to
   heteroscedasticity but has no finite-sample coverage guarantee.
2. **Split conformal calibration.** Hold out a calibration set $\mathcal I_2$. Compute per-sample
   "conformity scores" $E_i$ that measure how badly the plug-in interval mis-covers $y_i$. Shift
   the plug-in interval by the empirical $(1-\alpha)$-quantile of these scores. The resulting
   interval is **guaranteed** to satisfy $\mathbb P\{Y_{n+1}\in C(X_{n+1})\} \geq 1-\alpha$ under
   exchangeability alone.

**Why this matters for precision modulation.** CQR produces a *scalar interval width* per input
— this is a per-sample gate, not an inverse-variance precision. But for gating applications where
the semantics of "precision" reduce to "how wide is the confidence region", CQR is the
uncertainty-estimation method with the strongest theoretical guarantees in the modern deep
learning literature. It is the **one method in this review that trades parametric assumptions
for distribution-free coverage guarantees** — a strictly different point in design space.

### Layer 2 — Main results and algorithm

**Pinball loss (check function).** For a target quantile level $\alpha$,

$$
\rho_\alpha(y, \hat y) \;=\; \begin{cases} \alpha(y - \hat y) & \text{if } y - \hat y > 0,\\ (1-\alpha)(\hat y - y) & \text{otherwise}.\end{cases}
\tag{R-1}
$$

The pinball loss is the proper scoring rule for quantile estimation: its expected minimiser is exactly the $\alpha$th conditional quantile $q_\alpha(x)$.

**Ideal interval (Eq. R-2).** With $\alpha_{\rm lo} = \alpha/2, \alpha_{\rm hi} = 1-\alpha/2$,

$$
C(x) \;=\; [\,q_{\alpha_{\rm lo}}(x),\, q_{\alpha_{\rm hi}}(x)\,].
\tag{R-2}
$$

By construction $\mathbb P\{Y\in C(X)\mid X=x\} \geq 1-\alpha$ (eq. R-3), i.e. *conditionally*
valid — the strongest possible coverage statement. In practice we replace $q$'s with estimates
$\hat q$ and lose conditional validity.

**CQR conformity score (Eq. R-6).** On the calibration set,

$$
E_i \;=\; \max\!\big\{\,\hat q_{\alpha_{\rm lo}}(X_i) - Y_i,\; Y_i - \hat q_{\alpha_{\rm hi}}(X_i)\,\big\}
\tag{R-6}
$$

is *positive* when $Y_i$ falls outside the plug-in interval (by the amount of the violation) and
*negative* when $Y_i$ is comfortably inside.

**CQR interval (Eq. R-7).** Let $Q_{1-\alpha}(E, \mathcal I_2)$ be the $\lceil(1-\alpha)(|\mathcal I_2|+1)\rceil / |\mathcal I_2|$ empirical quantile of $\{E_i\}$. Then

$$
C(X_{n+1}) \;=\; \big[\,\hat q_{\alpha_{\rm lo}}(X_{n+1}) - Q_{1-\alpha}(E,\mathcal I_2),\;\; \hat q_{\alpha_{\rm hi}}(X_{n+1}) + Q_{1-\alpha}(E,\mathcal I_2)\,\big].
\tag{R-7}
$$

Inflating (or contracting) the plug-in interval symmetrically by $Q_{1-\alpha}(E,\mathcal I_2)$
produces an interval with the guaranteed coverage.

**Theorem 1 (coverage guarantee).** If $(X_i, Y_i), i=1,\dots,n+1$ are *exchangeable*, then

$$
\mathbb P\{Y_{n+1} \in C(X_{n+1})\} \;\geq\; 1-\alpha.
\tag{R-Thm1}
$$

If the conformity scores $E_i$ are almost surely distinct, the bound is nearly tight:
$\mathbb P\{Y_{n+1}\in C(X_{n+1})\} \leq 1-\alpha + 1/(|\mathcal I_2|+1)$.

**Theorem 2 (asymmetric two-tailed).** Inflate left and right tails independently with
separate quantiles $Q_{1-\alpha_{\rm lo}}(E_{\rm lo}, \mathcal I_2)$ and
$Q_{1-\alpha_{\rm hi}}(E_{\rm hi}, \mathcal I_2)$ to obtain one-sided coverage bounds of
$1-\alpha_{\rm lo}$ and $1-\alpha_{\rm hi}$ respectively.

**Empirical highlights (Table 1).** Across 11 UCI benchmarks with $\alpha=0.1$ and 20 train-test
splits (2,200 total experiments):

- Every conformal method hits the nominal 90% coverage (89.9%–90.3%).
- Raw (non-conformalised) quantile neural nets **undercover** (88.87%) and raw quantile random
  forests **overcover** (92.62%) — illustrating why conformal calibration is needed.
- CQR Random Forests and CQR Neural Net achieve the *shortest* average intervals (1.40 each),
  beating locally adaptive split conformal (1.79), standard split conformal with neural net (2.20),
  and non-conformalised quantile methods.

### Layer 3 — Graduate-level deep dive with full derivations

#### 5.3.1 Why the pinball loss is the right quantile estimator

**Claim.** $q_\alpha(x) = \arg\min_{\hat q}\, \mathbb E_{Y\sim p(\cdot\mid x)}[\rho_\alpha(Y, \hat q)]$.

**Proof.** Expand the expectation:

$$
\mathbb E[\rho_\alpha(Y,\hat q)]
\;=\; \alpha\int_{\hat q}^\infty (y-\hat q)\,p(y\mid x)\,\mathrm d y \;+\; (1-\alpha)\int_{-\infty}^{\hat q}(\hat q-y)\,p(y\mid x)\,\mathrm d y.
\tag{R.1}
$$

Differentiate with respect to $\hat q$. Using Leibniz's rule,

$$
\frac{\mathrm d}{\mathrm d\hat q}\,\mathbb E[\rho_\alpha]
\;=\; -\alpha\int_{\hat q}^\infty p(y\mid x)\,\mathrm d y \;+\; (1-\alpha)\int_{-\infty}^{\hat q} p(y\mid x)\,\mathrm d y
\;=\; -\alpha\,(1-F(\hat q)) + (1-\alpha)\,F(\hat q),
\tag{R.2}
$$

where $F$ is the conditional CDF. Setting the derivative to zero:

$$
-\alpha + \alpha F(\hat q) + F(\hat q) - \alpha F(\hat q) = 0 \quad\Rightarrow\quad F(\hat q) = \alpha,
\tag{R.3}
$$

i.e. $\hat q = F^{-1}(\alpha) = q_\alpha(x)$, as claimed. The pinball loss has an *asymmetric*
cost structure: residuals of the wrong sign are weighted differently, precisely so that the
minimiser recovers the quantile rather than the mean.

#### 5.3.2 Exchangeability: the core technical assumption

A sequence $(Z_1, \dots, Z_{n+1})$ is *exchangeable* if its joint distribution is invariant under
permutation. IID samples are exchangeable; exchangeability is strictly weaker (e.g. Pólya urn
samples are exchangeable but not IID). Every conformal prediction result in this paper relies on
exchangeability and *only* on exchangeability — this is the distribution-free guarantee.

**Key lemma.** If $(Z_1,\dots,Z_{n+1})$ are exchangeable, then for any *symmetric* function
$A$ of the first $n$ coordinates, the rank of $A(Z_1,\dots,Z_n; Z_{n+1})$ among
$\{A(Z_1,\dots,Z_n; Z_i)\}_{i=1}^{n+1}$ (the standard conformal "plug-in rank") is uniformly
distributed on $\{1,\dots,n+1\}$. This is the sole probabilistic fact needed for Theorem 1.

#### 5.3.3 Proof sketch of Theorem 1 (standard conformal argument)

Let $\{E_1, \dots, E_{|\mathcal I_2|}\}$ be the calibration-set conformity scores, and let
$E_{n+1}$ be the (unobserved) test-point conformity score defined analogously by R-6:

$$
E_{n+1} \;:=\; \max\!\big\{\,\hat q_{\alpha_{\rm lo}}(X_{n+1}) - Y_{n+1},\; Y_{n+1} - \hat q_{\alpha_{\rm hi}}(X_{n+1})\,\big\}.
\tag{R.4}
$$

**Step 1.** Because $\hat q_{\alpha_{\rm lo}}, \hat q_{\alpha_{\rm hi}}$ are trained on the *proper training set* $\mathcal I_1$ (disjoint from $\mathcal I_2$ and from $(X_{n+1}, Y_{n+1})$), and because the calibration + test samples $\{(X_i, Y_i)\}_{i\in\mathcal I_2}\cup\{(X_{n+1}, Y_{n+1})\}$ are exchangeable (conditional on $\mathcal I_1$), the scores $\{E_i\}_{i\in\mathcal I_2}\cup\{E_{n+1}\}$ are also exchangeable.

**Step 2.** By the rank lemma (§5.3.2), for any $k\in\{1,\dots,|\mathcal I_2|+1\}$,

$$
\mathbb P\{E_{n+1}\text{ has rank }k\} \;=\; \frac{1}{|\mathcal I_2|+1}.
\tag{R.5}
$$

**Step 3.** Let $Q_{1-\alpha}(E,\mathcal I_2)$ be the $\lceil(1-\alpha)(|\mathcal I_2|+1)\rceil$-th order statistic of the calibration scores. The event $\{Y_{n+1}\in C(X_{n+1})\}$ is equivalent to $\{E_{n+1} \leq Q_{1-\alpha}(E,\mathcal I_2)\}$ because:

- $Y_{n+1}\geq \hat q_{\alpha_{\rm lo}}(X_{n+1}) - Q_{1-\alpha}$ iff $\hat q_{\alpha_{\rm lo}}(X_{n+1}) - Y_{n+1} \leq Q_{1-\alpha}$.
- $Y_{n+1}\leq \hat q_{\alpha_{\rm hi}}(X_{n+1}) + Q_{1-\alpha}$ iff $Y_{n+1} - \hat q_{\alpha_{\rm hi}}(X_{n+1}) \leq Q_{1-\alpha}$.
- The $\max$ of these two residuals (R-6) is $\leq Q_{1-\alpha}$ iff *both* hold, which is $\{Y_{n+1}\in C(X_{n+1})\}$.

**Step 4.** By exchangeability (R.5), $E_{n+1}$ is *rank-uniform* among $\{E_1,\dots,E_{|\mathcal I_2|}, E_{n+1}\}$. Therefore

$$
\mathbb P\{E_{n+1}\leq Q_{1-\alpha}(E,\mathcal I_2)\} \;\geq\; \frac{\lceil(1-\alpha)(|\mathcal I_2|+1)\rceil}{|\mathcal I_2|+1} \;\geq\; 1-\alpha.
\tag{R.6}
$$

Combining Steps 3 and 4 gives Theorem 1. The second part (near-tightness when $E_i$'s are a.s. distinct) is a corollary: with distinct scores, the rank distribution is uniform on exactly $|\mathcal I_2|+1$ values, so the probability is also bounded above by $(\lceil(1-\alpha)(|\mathcal I_2|+1)\rceil+1)/(|\mathcal I_2|+1) \leq 1-\alpha + 1/(|\mathcal I_2|+1)$.

#### 5.3.4 Why the conformity score (R-6) is the "right" choice

Three alternatives one might have tried:

**(a) Absolute residual to predicted mean.** The original split-conformal choice:
$R_i = |Y_i - \hat\mu(X_i)|$. But this produces a *fixed-width* interval (paper's Eq. 5) — the calibration width $Q_{1-\alpha}(R, \mathcal I_2)$ does not depend on $X_{n+1}$. No local adaptivity.

**(b) Scaled residual.** Locally adaptive split conformal (Papadopoulos et al., 2008):
$\tilde R_i = |Y_i - \hat\mu(X_i)| / \hat\sigma(X_i)$. Produces variable-width intervals but requires fitting *two* networks ($\hat\mu$ and $\hat\sigma$) and has a subtle issue: $\hat\sigma$ is trained on the *training residuals*, which are biased (small) relative to test-time residuals. This forces $Q_{1-\alpha}(\tilde R, \mathcal I_2)$ to be larger than it should be, eroding adaptivity.

**(c) CQR signed max (R-6).** The key insight: define $E_i$ *directly* in terms of the quantile estimates so that conformalisation is a shift of the quantile-regression interval, not a rescaling of a mean-regression interval. The $\max$ handles both under- and overcoverage symmetrically: if $Y_i$ is safely inside, $E_i < 0$ (the interval was too wide); if $Y_i$ falls outside, $E_i > 0$ (the interval was too narrow). The calibration quantile $Q_{1-\alpha}(E,\mathcal I_2)$ can therefore be *negative*, meaning the plug-in interval was conservative and CQR *shrinks* it.

This last property — that CQR can *shrink* an over-conservative plug-in interval — is what produces the dramatic 2.20 → 1.40 interval-length drop observed empirically (Table 1).

#### 5.3.5 Theorem 2 derivation

The two-tailed extension (Theorem 2) replaces the two-sided maximum with independent one-sided scores:

$$
E_{\rm lo, i} \;=\; \hat q_{\alpha_{\rm lo}}(X_i) - Y_i,\qquad
E_{\rm hi, i} \;=\; Y_i - \hat q_{\alpha_{\rm hi}}(X_i).
\tag{R.7}
$$

Apply the conformal argument *separately* to each:

$$
\mathbb P\{Y_{n+1} \geq \hat q_{\alpha_{\rm lo}}(X_{n+1}) - Q_{1-\alpha_{\rm lo}}(E_{\rm lo},\mathcal I_2)\} \geq 1-\alpha_{\rm lo},
\tag{R.8}
$$

$$
\mathbb P\{Y_{n+1} \leq \hat q_{\alpha_{\rm hi}}(X_{n+1}) + Q_{1-\alpha_{\rm hi}}(E_{\rm hi},\mathcal I_2)\} \geq 1-\alpha_{\rm hi}.
\tag{R.9}
$$

By a union bound, the two-sided miscoverage is at most $\alpha_{\rm lo} + \alpha_{\rm hi} = \alpha$, hence $\mathbb P\{Y_{n+1}\in C(X_{n+1})\}\geq 1-\alpha$.

**Cost.** The union bound is typically loose, so the total coverage is usually much larger than
$1-\alpha$ — which translates into *wider* intervals than Theorem 1's symmetric conformalisation.
The paper reports CQR NN moving from length 1.40 (Theorem 1) to 1.58 (Theorem 2) at no change in
coverage. Use Theorem 2 only when asymmetric one-sided guarantees are required (e.g. safety-critical lower bounds).

#### 5.3.6 What CQR is not

Three clarifications that bear on the GridWorld Pain use case:

- **CQR is not Bayesian.** There is no prior, no posterior — only frequentist coverage. The
  "uncertainty" CQR produces is a *prediction interval width*, not a variance. It cannot be
  used as an *inverse-variance precision* in the Active Inference sense.
- **Coverage is marginal, not conditional.** Theorem 1 guarantees $\mathbb P\{Y_{n+1}\in C(X_{n+1})\}\geq 1-\alpha$, where the probability is over *both* $X_{n+1}$ and the calibration set. It does *not* guarantee $\mathbb P\{Y\in C(X)\mid X=x\}\geq 1-\alpha$ for every $x$. Recent work (Angelopoulos & Bates, 2023, tutorial; Barber et al. 2021) characterises the gap.
- **CQR is a post-hoc wrapper.** You still need a quantile network (trained with pinball loss) to start with. CQR does *not* fix a bad quantile model — it only corrects its coverage on average.

#### 5.3.7 Bridges to the rest of the review

- **To §1–4 (parametric variance heads).** Complementary, not competing. Kendall & Gal /
  Lakshminarayanan / Amini / Detlefsen give a variance or precision; CQR gives a prediction
  interval. One can even *conformalise* a Gaussian NLL head's interval via CQR: use
  $\hat q_{\alpha_{\rm lo}}(x) = \hat\mu(x) - z_{1-\alpha/2}\hat\sigma(x)$ and analogously for
  the upper bound, then apply Algorithm 1. This is often called "calibrated regression" (Kuleshov et al., 2018) and is a common final-step fix for over/under-confident NN variance heads.
- **To §6 (Stirn & Knowles).** Stirn & Knowles produce a *calibrated* $\sigma^2$; CQR produces a
  *calibrated* interval. They are two different routes to the same end goal.
- **To §7 (β-NLL) and §8 (Meinert et al.).** Orthogonal concerns: β-NLL fixes collapse in
  parametric variance training; CQR bypasses the parametric question entirely. If your
  downstream application needs a scalar gate, use β-NLL; if it needs a guaranteed coverage
  interval, use CQR.

---

## 6. Stirn & Knowles (2020) — *Variational Variance: Simple, Reliable, Calibrated Heteroscedastic Noise Variance Parameterization*

**Venue:** AISTATS 2021 (arXiv:2006.04910 v3, Oct 2020). **Domain:** Regression and VAEs (UCI
benchmarks, MNIST / CIFAR / CelebA).

### Layer 1 — Basic introduction and concept

Stirn & Knowles is the *variational* answer to the same failure mode Detlefsen et al. (§4)
diagnosed at the batching level and Seitzer et al. (§7 below) diagnose at the loss level. The
authors' diagnosis:

> *If the mean network fits nearly perfectly, $\mu(x_i)\approx y_i$, then maximising the log
> likelihood pushes the variance network toward $\sigma^2(x_i)\to 0$. And because $\sigma^{-2}$
> multiplies the mean-gradient, as $\mu$ improves, the effective learning rate on $\mu$
> **blows up** — violating the Robbins–Monro convergence criterion.*

Their fix: **treat precision $\lambda = 1/\sigma^2$ as a latent variable**, place a prior
$p(\lambda)$ on it, and do variational inference with $q(\lambda\mid x) = \mathrm{Gamma}(\lambda\mid\alpha(x), \beta(x))$. The KL term $\mathrm{KL}(q(\lambda\mid x)\,\|\,p(\lambda))$ that appears in
the ELBO acts as a **probabilistic barrier function** that prevents precision from escaping to
infinity — analogous to log-barrier constraints in convex optimisation.

A further contribution is the methodological one: moving past log-likelihood as the sole
evaluation metric and insisting on **posterior predictive checks (PPCs)** — specifically,
measuring mean-bias, mean-RMSE, variance-bias, variance-RMSE, sample-bias, and sample-RMSE
against the empirical training distribution. Many methods that win on NLL *fail* these PPCs;
Stirn & Knowles is the paper that exposes this.

**Why this matters for precision modulation.** The paper introduces *several* priors
(Gamma, VAMP, xVAMP, VBEM, VBEM∗) that differ in whether they are homo- or heteroscedastic and
whether their parameters are trainable. The VBEM∗ (trainable mixture of Gammas) emerges as the
best-performing variant. For the GridWorld Pain Phase 3 precision head, this paper is the
strongest argument that *training-time regularisation of the variance head* via a KL-to-prior
term is structurally superior to the bare Gaussian NLL — and it predates β-NLL (§7) by two years
with an essentially compatible claim.

### Layer 2 — Main results and algorithm

**Generative model for regression (Fig. 1, right).** With neural nets $\mu(x), \alpha(x), \beta(x)$,

$$
\lambda \sim p(\lambda),\qquad y \mid x, \lambda \sim \mathcal N(\mu(x),\, 1/\lambda),
\qquad q(\lambda\mid x) \;=\; \mathrm{Gamma}(\lambda\mid \alpha(x), \beta(x)).
\tag{SK-1}
$$

**Variational objective (Eq. SK-1 in the paper).**

$$
\mathcal L \;=\; \sum_{(x,y)\in\mathcal D} \underbrace{\mathbb E_{q(\lambda\mid\alpha(x),\beta(x))}\!\big[\log\mathcal N(y\mid\mu(x), 1/\lambda)\big]}_{\text{expected log-likelihood (analytic)}}
\;-\; \underbrace{\mathrm{KL}\!\big(q(\lambda\mid\alpha(x),\beta(x))\,\|\,p(\lambda)\big)}_{\text{regulariser}}.
\tag{SK-1}
$$

The inner expectation evaluates in closed form (§6.3.2 below):

$$
\mathbb E_{q(\lambda)}[\log\mathcal N(y\mid\mu, 1/\lambda)]
\;=\; \tfrac{1}{2}\!\Big(\psi(\alpha(x)) - \log\beta(x) - \log 2\pi - \tfrac{\alpha(x)}{\beta(x)}(y-\mu(x))^2\Big),
\tag{SK-2}
$$

where $\psi(\cdot)$ is the digamma function.

**Prior menu.** The paper's empirical contribution is to enumerate and compare six priors:

| Prior | Form | Heteroscedastic? | Trainable? |
|-------|------|:---:|:---:|
| Standard | $p(\lambda) = \mathrm{Gamma}(\lambda; a, b)$ with fixed $(a,b)$ | No | No |
| VAP (ablation) | $p(\lambda_i) := q(\lambda_i\mid\alpha(x_i), \beta(x_i))$ | Yes (degenerate) | — |
| VAMP | $p(\lambda) = N^{-1}\sum_j q(\lambda\mid\alpha(x_j), \beta(x_j))$ (aggregate posterior) | No | — |
| VAMP* | VAMP with trainable pseudo-inputs $u_j$ | No | Yes |
| xVAMP | $p(\lambda\mid x) = \sum_j \pi_j(x)\,q(\lambda\mid\alpha(u_j), \beta(u_j))$ | **Yes** | — |
| xVAMP* | xVAMP with trainable $u_j$ | Yes | Yes |
| VBEM | $p(\lambda\mid x) = \sum_j \pi_j(x)\,\mathrm{Gamma}(\lambda; a_j, b_j)$, fixed grid $(a_j, b_j)$ | Yes | No |
| VBEM* | VBEM with trainable $(\hat a_j, \hat b_j)$ | Yes | **Yes** |

where $\pi(x)$ is a neural-net simplex mapping producing mixture proportions.

**Empirical findings (Table 1 + Table 2, UCI regression, 20 trials per dataset).**

- **VBEM*** wins log-likelihood on 5/10 datasets and is the most balanced across the six PPC
  categories.
- **Student** (Detlefsen's Gamma-Normal parameterised Student-$t$, no KL) and **VAP** (variational
  with zero KL regularisation) both underperform — confirming that the **KL-to-prior term is the
  load-bearing ingredient**, not the Gamma-Normal form.
- **VAMP / VAMP*** fail because marginalising over $x_j$ in the prior *destroys heteroscedasticity*:
  the prior becomes approximately homoscedastic and the variational posterior is pushed toward
  uniform over training positions.

**VAE application (§4).** The same machinery applies: introduce a per-$z$ Gamma posterior over
decoder precision. The resulting model is called **V3AE**. It beats vanilla Gaussian VAE and
Takahashi et al.'s Student-$t$ VAE on ELBO and on sample-based PPCs.

### Layer 3 — Graduate-level deep dive with full derivations

#### 6.3.1 The diagnosis: why Gaussian NLL optimisation is unstable

Consider the Gaussian NLL in precision parameterisation $\lambda := 1/\sigma^2$:

$$
-\log p(y\mid\mu, \lambda) \;=\; \tfrac{1}{2}\log(2\pi/\lambda) \;+\; \tfrac{\lambda}{2}(y-\mu)^2.
\tag{SK.A}
$$

Differentiate with respect to $\mu$:

$$
\frac{\partial \mathcal L}{\partial \mu} \;=\; \lambda\,(\mu - y),
\tag{SK.B}
$$

and with respect to $\lambda$:

$$
\frac{\partial \mathcal L}{\partial \lambda} \;=\; -\,\frac{1}{2\lambda} \;+\; \tfrac{1}{2}(y-\mu)^2.
\tag{SK.C}
$$

At the stationary point $\partial\mathcal L/\partial\lambda = 0$, the MLE solution is
$\hat\lambda = 1/(y-\mu)^2$. As $\mu\to y$ (perfect mean), $\hat\lambda\to\infty$ — the
pathological zero-variance solution.

**The instability mechanism.** The gradient SK.B is proportional to $\lambda$. So as the mean
network improves and the variance network consequently predicts large $\lambda$, the update *step
size on $\mu$* grows — a violation of the Robbins–Monro diminishing-step-size criterion required
for SGD convergence. Small residuals produce *disproportionately large* mean updates, which can
destabilise training entirely. This is structurally the *same* loop as Kendall & Gal's
L.1.9–L.1.10 analysis (§1), just rewritten in precision rather than log-variance.

#### 6.3.2 Derivation of the analytic expected log-likelihood → Eq. SK-2

The inner expectation in the ELBO requires

$$
\mathbb E_{\lambda\sim\mathrm{Gamma}(\alpha,\beta)}\!\big[\log\mathcal N(y\mid\mu, 1/\lambda)\big]
\;=\; -\tfrac{1}{2}\log(2\pi) + \tfrac{1}{2}\,\mathbb E[\log\lambda] - \tfrac{1}{2}\,\mathbb E[\lambda]\,(y-\mu)^2.
\tag{SK.D}
$$

Using two standard Gamma-distribution identities (shape-rate form, $\mathbb E[\lambda] = \alpha/\beta$):

$$
\mathbb E_\lambda[\lambda] \;=\; \alpha/\beta,
\qquad
\mathbb E_\lambda[\log \lambda] \;=\; \psi(\alpha) - \log\beta.
\tag{SK.E}
$$

The $\log\lambda$ identity follows from $\mathbb E[\log\lambda] = \frac{\partial}{\partial\alpha}\log\int\lambda^{\alpha-1}\exp(-\beta\lambda)\mathrm d\lambda\big|_{\rm normalised}
= \psi(\alpha) - \log\beta$, where the digamma $\psi$ is the logarithmic derivative of the Gamma function.

Substituting SK.E into SK.D:

$$
\mathbb E_{q}[\log\mathcal N(y\mid\mu, 1/\lambda)] \;=\; \tfrac{1}{2}\big[\psi(\alpha(x)) - \log\beta(x) - \log 2\pi - \tfrac{\alpha(x)}{\beta(x)}(y-\mu(x))^2\big].
\tag{SK-2}
$$

This is exactly the paper's Eq. SK-2. Importantly, the "effective precision" in the residual term
is $\alpha(x)/\beta(x) = \mathbb E_q[\lambda]$ — so the gradient w.r.t. $\mu$ is proportional to
the *expected* precision under the variational posterior, not to any single sampled $\lambda$.

#### 6.3.3 Why the KL regulariser prevents collapse

The ELBO's KL term against a Gamma prior $\mathrm{Gamma}(a, b)$ is

$$
\mathrm{KL}\!\big(\mathrm{Gamma}(\alpha,\beta)\,\|\,\mathrm{Gamma}(a, b)\big)
\;=\; (\alpha - a)\psi(\alpha) - \log\Gamma(\alpha) + \log\Gamma(a) + a(\log\beta - \log b) + \alpha\frac{b-\beta}{\beta}.
\tag{SK.F}
$$

**Gradient with respect to $\alpha, \beta$.** As $\beta\to 0^+$ (variance $\to 0$, i.e. collapse),
the term $a\log\beta\to -\infty$ and the $\alpha(b-\beta)/\beta = \alpha b/\beta - \alpha$ term
blows up. Similarly, as $\alpha\to\infty$, $(\alpha-a)\psi(\alpha)\to\infty$ linearly. Both
divergent regimes receive *infinite penalty* from the KL term, which acts as a soft barrier
preventing the variational posterior from pathologically concentrating.

This is the *probabilistic analogue of a log-barrier function* in convex optimisation: the KL
to a fixed Gamma prior provides a smooth penalty that grows unboundedly as precision hits the
zero-variance singularity, preventing the failure mode diagnosed in §6.3.1.

**Sensitivity to prior choice.** With a **Standard** $\mathrm{Gamma}(a, b)$ prior, the barrier
is strong but *homoscedastic*: it pushes all $q(\lambda\mid x)$ toward the same mean, losing
local variance structure. This is why Standard underperforms on heteroscedastic data. The
heteroscedastic priors (xVAMP, VBEM) preserve local variance structure while still providing
collapse protection.

#### 6.3.4 The VAP ablation: why the KL term is the key

The **VAP (Variational Posterior)** prior sets $p(\lambda_i) := q(\lambda_i\mid\alpha(x_i), \beta(x_i))$ — the prior *is* the posterior, making $\mathrm{KL}(q\,\|\,p) \equiv 0$ pointwise. The ELBO
reduces to the *expected log likelihood alone*.

This is equivalent to maximum-likelihood training of the Student-$t$ marginal $\int \mathcal N(y\mid\mu, 1/\lambda)\,\mathrm{Gamma}(\lambda\mid\alpha(x), \beta(x))\mathrm d\lambda$ — because by Jensen's inequality,

$$
\mathbb E_{q(\lambda)}[\log \mathcal N(y\mid\mu, 1/\lambda)] \;\leq\; \log\mathbb E_{q(\lambda)}[\mathcal N(y\mid\mu, 1/\lambda)] \;=\; \log\mathrm{St}(y; \mu, \alpha, \beta).
\tag{SK.G}
$$

The fact that **VAP underperforms Standard/xVAMP/VBEM on PPCs** is the paper's strongest
evidence that the KL barrier, not the Gamma-Normal marginalisation (§4, Detlefsen et al.) or the
Student-$t$ form, is what makes the method work.

#### 6.3.5 xVAMP prior and the KL decomposition → Eq. SK.3

For the heteroscedastic mixture prior

$$
p(\lambda\mid x) \;=\; \sum_{j=1}^K \pi_j(x)\,q(\lambda\mid\alpha(u_j), \beta(u_j)),
$$

the KL decomposes as

$$
\mathrm{KL}(q(\lambda\mid x)\,\|\,p(\lambda\mid x))
\;=\; \mathbb E_{q(\lambda\mid x)}[\log q(\lambda\mid x)] \;-\; \mathbb E_{q(\lambda\mid x)}\!\Big[\log\sum_j \pi_j(x)\,q(\lambda\mid u_j)\Big].
\tag{SK-3}
$$

The first term is the *negative entropy* of a Gamma distribution and evaluates analytically:

$$
-\mathrm H(\mathrm{Gamma}(\alpha,\beta)) \;=\; \alpha - \log\beta + \log\Gamma(\alpha) + (1-\alpha)\psi(\alpha).
\tag{SK.H}
$$

The second term has no closed form because of the mixture structure — it is Monte-Carlo
estimated using the log-sum-exp trick for numerical stability:

$$
\mathbb E_{q(\lambda)}\!\Big[\log\sum_j \pi_j(x)\,q(\lambda\mid u_j)\Big]
\;\approx\; \frac{1}{M}\sum_{m=1}^M \log\sum_j \pi_j(x)\,q(\lambda^{(m)}\mid u_j),
\qquad \lambda^{(m)}\sim q(\lambda\mid x).
\tag{SK.I}
$$

Reparameterisation gradients are available for the Gamma distribution via the implicit reparameterisation trick (Figurnov et al., 2018), so the ELBO is fully end-to-end differentiable.

#### 6.3.6 Why VAMP fails but xVAMP works — the heteroscedasticity preservation argument

The vanilla VAMP prior $p(\lambda) = N^{-1}\sum_j q(\lambda\mid\alpha(x_j), \beta(x_j))$
*marginalises out* $x$, producing a homoscedastic prior. The KL to this prior is minimised when
all $q(\lambda\mid x)$ agree — i.e. when the variational posterior *itself* becomes homoscedastic.

The paper's toy experiment (Fig. 2) confirms this empirically: VAMP predicts nearly constant
$\mathrm{std}(y\mid x) \approx 1.8$ across the entire training interval [0, 10], even though the
true standard deviation varies from ≈ 0.3 at $x=0$ to ≈ 3.3 at $x=10$. The aggregate-posterior
prior erases heteroscedasticity by design.

**xVAMP fixes this** by making the mixture proportions $\pi_j(x)$ *input-dependent*: the prior
adapts to $x$, so the KL-minimising $q(\lambda\mid x)$ is allowed to differ across $x$. The
Gamma mixture components $q(\lambda\mid u_j)$ are shared (trained on pseudo-inputs), but the
gate $\pi(x)$ selects a different combination at each input.

#### 6.3.7 Posterior predictive checks (PPCs): a methodological contribution

The paper's evaluation protocol — six PPC metrics plus log likelihood — is a methodological
contribution the field has only partially adopted. The six metrics are:

1. **Mean Bias** — $\mathbb E[\mathbb E[y\mid x] - y]$ over validation points
2. **Mean RMSE** — $\mathrm{RMSE}(\mathbb E[y\mid x], y)$
3. **Variance Bias** — $\mathbb E[\mathrm{Var}[y\mid x] - (\mathbb E[y\mid x] - y)^2]$
4. **Variance RMSE** — $\mathrm{RMSE}(\mathrm{Var}[y\mid x], (\mathbb E[y\mid x] - y)^2)$
5. **Sample Bias** — $\mathbb E[y^\star - y]$ where $y^\star\sim p(y\mid x)$
6. **Sample RMSE** — $\mathrm{RMSE}(y^\star, y)$

Metric 3–4 are the critical diagnostic: a model can win on NLL by *overestimating* variance in
the tails (compensating for bad mean predictions) while *underestimating* it in the body. Only
variance-calibration metrics catch this.

**Implication for GridWorld Pain Phase 3.** If the Phase 3 precision head is evaluated only by RL
return, variance miscalibration can be invisible. Adding even a single variance-calibration
probe (e.g. empirical residual variance vs. predicted $\sigma^2$ on a held-out batch) would go
a long way to preventing the collapse modes this paper documents.

#### 6.3.8 Bridges to the rest of the review

- **To §1 (Kendall & Gal).** Same instability diagnosis, different fix: Kendall & Gal live with
  the Gaussian NLL and hope training works; Stirn & Knowles change the loss to a variational
  objective with a KL barrier.
- **To §3 (Amini, DER).** DER places a NIG prior over $(\mu, \sigma^2)$ and marginalises; V3AE
  places a Gamma prior over $\lambda$ alone and does variational inference. Both arrive at a
  Student-$t$ predictive, but V3AE's KL-to-prior term is what Meinert et al. (§8) will argue DER
  is missing — V3AE implements the regularisation DER's evidence regulariser (A-9) gestures at
  but does not derive from first principles.
- **To §4 (Detlefsen).** Complementary. Detlefsen's locality sampler fixes the *batching*
  failure mode; Stirn & Knowles' KL fixes the *loss-landscape* failure mode. A system that
  combines both would be strictly stronger than either.
- **To §7 (β-NLL).** β-NLL gets to the same result via a much simpler knob: reweight the NLL
  by $\sigma^{2\beta}$. Stirn & Knowles give a principled Bayesian derivation of the same
  anti-collapse regularisation. β-NLL is the operational winner; Stirn & Knowles is the
  theoretical justification.
- **To §8 (Meinert, DER critique).** Stirn & Knowles essentially argues that *any* good
  precision head needs a KL-to-prior regulariser. DER does not have one — it has the evidence
  regulariser (A-9), which Meinert et al. will show is not equivalent. Reading §8 with Stirn &
  Knowles' KL argument in mind explains *why* DER's decomposition fails.

---

## 7. Seitzer, Tavakoli, Antić & Martius (2022) — *On the Pitfalls of Heteroscedastic Uncertainty Estimation with Probabilistic Neural Networks*

**Venue:** ICLR 2022 (arXiv:2203.09168). **Domain:** Regression (UCI + image-based
object-dynamics prediction), model-based RL.

### Layer 1 — Basic introduction and concept

Seitzer et al. is the *most surgical* paper in this review: it takes a single, specific failure
mode of Gaussian NLL training and proposes a one-line fix that is trivially implementable,
provably interpolates between NLL and MSE, and empirically outperforms both.

**The failure mode (Fig. 1 of the paper).** Train a two-headed $(\mu, \sigma^2)$ network on a
simple sinusoid with tiny homoscedastic noise. Against expectation, **the network cannot fit
the left half of the domain** — the mean prediction stays locally linear while the variance
estimate inflates to match the residuals. In contrast, an MSE-trained network converges to the
optimal fit within $10^5$ updates. After $10^7$ NLL updates, RMSE is still dramatically worse
than MSE achieved in two orders of magnitude less time. The behaviour is **stable** across
optimisers, hyperparameters, and architectures.

**The diagnosis.** In the NLL gradient $\nabla_{\hat\mu}\mathcal L_{\rm NLL} \propto (\hat\mu - y)/\hat\sigma^2$, the per-point contribution is scaled by $1/\hat\sigma^2$. As the network partially fits some region, $\hat\sigma^2$ shrinks there, which *increases* the effective gradient on that region. The network is drawn ever more strongly to the already-fit region, while the poorly-fit region — which reports large $\hat\sigma^2$ — is *down-weighted* to irrelevance. This is equivalent to sampling each point with probability $\propto 1/\sigma^2(x)$, inverting the relationship that good gradient descent should have (high-error points should matter *more*, not less).

**The fix: β-NLL.** Reweight the per-point NLL by $\lfloor\hat\sigma^{2\beta}\rfloor$ (stop-gradient applied), where $\beta\in[0, 1]$ is a hyperparameter interpolating between vanilla NLL ($\beta=0$) and MSE ($\beta=1$). At $\beta=1$, the mean gradient exactly matches MSE while the variance head still receives a calibrating signal. At $\beta=0.5$ (the empirical sweet spot), the effective weighting becomes $1/\sigma$ rather than $1/\sigma^2$.

**Why this matters for precision modulation.** β-NLL is the single most actionable recipe in this
review. The code change is literally one line. Every downstream paper that uses a Gaussian NLL
head for uncertainty — DreamerV3 decoder variant, FiLM-Ensemble, the GridWorld Pain Phase 3
precision head — should use β-NLL with $\beta = 0.5$ by default, not as a rescue measure.

### Layer 2 — Main results and algorithm

**Gaussian NLL (Eq. S-1).**

$$
\mathcal L_{\rm NLL}(\theta) \;=\; \mathbb E_{X,Y}\!\Big[\,\tfrac{1}{2}\log\hat\sigma^2(X) \;+\; \frac{(Y - \hat\mu(X))^2}{2\hat\sigma^2(X)} \;+\; \text{const}\,\Big].
\tag{S-1}
$$

**NLL gradients (Eqs. S-3, S-4).**

$$
\nabla_{\hat\mu}\mathcal L_{\rm NLL} \;=\; \mathbb E_{X,Y}\!\Big[\frac{\hat\mu(X) - Y}{\hat\sigma^2(X)}\Big],
\qquad
\nabla_{\hat\sigma^2}\mathcal L_{\rm NLL} \;=\; \mathbb E_{X,Y}\!\Big[\frac{\hat\sigma^2(X) - (Y-\hat\mu(X))^2}{2\,(\hat\sigma^2(X))^2}\Big].
\tag{S-3, S-4}
$$

**Effective data distribution under NLL (Eq. S-6).** The paper's key theoretical observation.
NLL's mean-gradient can be rewritten as an MSE gradient under a *reweighted* data distribution
$\tilde P(X, Y) = Z^{-1}\,P(X, Y)/\sigma^2(X)$:

$$
\nabla_{\hat\mu}\mathcal L_{\rm NLL}(\theta) \;\propto\; \nabla_{\hat\mu}\mathbb E_{X,Y\sim\tilde P}\!\Big[\frac{(Y - \hat\mu(X))^2}{2}\Big].
\tag{S-6}
$$

This makes explicit that NLL training is MSE on an *inverse-variance-weighted* dataset — which biases away from exactly the regions that need more attention.

**β-NLL loss (Eq. S-7).**

$$
\mathcal L_{\beta\text{-NLL}} \;:=\; \mathbb E_{X,Y}\!\Big[\lfloor\hat\sigma^{2\beta}(X)\rfloor\cdot\Big(\tfrac{1}{2}\log\hat\sigma^2(X) + \frac{(Y-\hat\mu(X))^2}{2\hat\sigma^2(X)} + \text{const}\Big)\Big],
\tag{S-7}
$$

where $\lfloor\cdot\rfloor$ is the stop-gradient (used in PyTorch as `.detach()`, in TensorFlow as `tf.stop_gradient`).

**β-NLL gradients (Eqs. S-8, S-9).**

$$
\nabla_{\hat\mu}\mathcal L_{\beta\text{-NLL}} \;=\; \mathbb E_{X,Y}\!\Big[\frac{\hat\mu(X) - Y}{\hat\sigma^{2-2\beta}(X)}\Big],
\qquad
\nabla_{\hat\sigma^2}\mathcal L_{\beta\text{-NLL}} \;=\; \mathbb E_{X,Y}\!\Big[\frac{\hat\sigma^2(X) - (Y-\hat\mu(X))^2}{2\,\hat\sigma^{4-2\beta}(X)}\Big].
\tag{S-8, S-9}
$$

**Key special cases.**

| $\beta$ | Mean gradient weighting | Variance gradient weighting | Interpretation |
|:---:|:---:|:---:|---|
| 0 | $1/\sigma^2$ | $1/\sigma^4$ | Standard NLL |
| 0.5 | $1/\sigma$ | $1/\sigma^3$ | **Empirical sweet spot** |
| 1 | 1 (uniform) | $1/\sigma^2$ | MSE for mean + NLL for variance |

At $\beta=1$, the mean head behaves as if trained with MSE (no inverse-variance weighting), while
the variance head still receives a useful training signal via the residual comparison. This is
the paper's cleanest theoretical result: **you can have the mean-stability of MSE and the
variance-calibration of NLL simultaneously.**

**Empirical highlights.**

- **Sinusoid (Fig. 1, 3).** β-NLL converges to the optimal fit in comparable time to MSE,
  while NLL never converges.
- **UCI regression, ObjectSlide, MuJoCo model-based RL** (Sec. 5). β-NLL with $\beta=0.5$ beats
  NLL on both RMSE and test log-likelihood on most tasks, and is strictly better on the
  "difficult" tasks where NLL's inverse-variance weighting self-sabotages.
- **Residual histograms (Fig. 6).** NLL produces a *bimodal* residual distribution — easy points
  fit to machine precision, hard points with long-tailed errors; β-NLL produces a log-normal
  distribution (no abandoned tail).

### Layer 3 — Graduate-level deep dive with full derivations

#### 7.3.1 The inverse-variance weighting derivation → Eq. S-6

Start from the NLL mean-gradient (S-3):

$$
\nabla_{\hat\mu}\mathcal L_{\rm NLL} \;=\; \mathbb E_{X,Y\sim P}\!\Big[\frac{\hat\mu(X) - Y}{\hat\sigma^2(X)}\Big]
\;=\; \int\int P(x, y)\cdot\frac{\hat\mu(x) - y}{\hat\sigma^2(x)}\,\mathrm d x\,\mathrm d y.
\tag{S.A}
$$

Define the reweighted distribution

$$
\tilde P(x, y) \;:=\; \frac{1}{Z}\cdot\frac{P(x, y)}{\sigma^2(x)},
\qquad Z \;=\; \int\int \frac{P(x, y)}{\sigma^2(x)}\,\mathrm d x\,\mathrm d y,
\tag{S.B}
$$

normalising $\tilde P$ to unit mass. Then

$$
\nabla_{\hat\mu}\mathcal L_{\rm NLL} \;=\; Z\int\int \tilde P(x, y)\cdot(\hat\mu(x) - y)\,\mathrm d x\,\mathrm d y \;=\; Z\cdot\mathbb E_{X,Y\sim\tilde P}[\hat\mu(X) - Y].
\tag{S.C}
$$

The right-hand side is proportional to the gradient of MSE on $\tilde P$:

$$
\nabla_{\hat\mu}\mathbb E_{X,Y\sim\tilde P}\!\Big[\frac{(Y - \hat\mu(X))^2}{2}\Big] \;=\; \mathbb E_{X,Y\sim\tilde P}[\hat\mu(X) - Y].
\tag{S.D}
$$

Combining S.C and S.D gives the paper's Eq. S-6. The practical consequence is sharp: *NLL
implicitly downweights high-variance regions* — which, in function-approximation settings, is
exactly backwards because "high variance" in early training often means "poorly fit", not "high
noise".

#### 7.3.2 The self-amplification mechanism

Suppose the network has started to fit a region correctly. Then
$(y - \hat\mu(x))^2 \approx \hat\sigma^2(x)$, i.e. the predicted variance matches the squared residual. Substituting into S-3's gradient:

$$
\frac{\hat\mu(x) - y}{\hat\sigma^2(x)} \;\approx\; \frac{\hat\mu(x) - y}{(\hat\mu(x) - y)^2} \;=\; \frac{1}{\hat\mu(x) - y}.
\tag{S.E}
$$

The gradient at well-fit points scales as *inverse residual*: smaller residuals → bigger gradients
→ further improvement → even smaller residuals → even bigger gradients. Meanwhile, in a poorly-fit
region where $(y - \hat\mu)^2 \gg \hat\sigma^2$, the gradient $|\hat\mu - y|/\hat\sigma^2$ is only
*moderately* large — and shrinking relative to the well-fit region's gradient. The two regions'
gradient magnitudes diverge: **rich get richer**.

This is the "vicious cycle" the authors describe in §3.2 and visualise in Fig. 5: the virtual
sampling probability of hard regions drops from ~$10^{-3}$ (uniform) to ~$10^{-5}$ over the
course of $10^6$ updates. The model converges to a locally stable but globally wrong solution
because the gradient for exiting that solution has been driven to zero.

#### 7.3.3 Why β-NLL fixes it → gradient analysis

Apply the chain rule to S-7 with stop-gradient on $\hat\sigma^{2\beta}$. The mean gradient is

$$
\nabla_{\hat\mu}\mathcal L_{\beta\text{-NLL}}
\;=\; \mathbb E_{X,Y}\!\Big[\,\hat\sigma^{2\beta}(X)\cdot\frac{\hat\mu(X) - Y}{\hat\sigma^2(X)}\,\Big]
\;=\; \mathbb E_{X,Y}\!\Big[\frac{\hat\mu(X) - Y}{\hat\sigma^{2(1-\beta)}(X)}\Big].
\tag{S.F}
$$

The effective inverse-variance weight in the mean gradient is now $1/\sigma^{2(1-\beta)}$. At
$\beta = 0.5$, this is $1/\sigma$ — the gradient still responds to local reliability, but the
self-amplification is dramatically weaker than $1/\sigma^2$. At $\beta = 1$, the weight is 1,
and the mean head behaves exactly as MSE-trained: no inverse-variance effect at all.

Similarly, the variance gradient becomes

$$
\nabla_{\hat\sigma^2}\mathcal L_{\beta\text{-NLL}}
\;=\; \mathbb E_{X,Y}\!\Big[\frac{\hat\sigma^2(X) - (Y-\hat\mu(X))^2}{2\,\hat\sigma^{4 - 2\beta}(X)}\Big].
\tag{S.G}
$$

This still drives $\hat\sigma^2$ toward the squared residual (the numerator vanishes when
$\hat\sigma^2 = (y-\hat\mu)^2$), so calibration is preserved — just with a different scaling
in the denominator. **The stop-gradient on $\hat\sigma^{2\beta}$ is essential**: without it, the
variance head would try to adjust the *weighting* in addition to minimising the residual, which
creates a second-order coupling that β-NLL specifically avoids.

#### 7.3.4 The interpolation: why β exists as a free parameter

The paper provides two readings of $\beta$:

**Reading 1 — data-distribution reweighting.** β-NLL is MSE on distribution $\tilde P_\beta(x, y) = Z^{-1}P(x, y)\sigma^{2\beta-2}(x)$. At $\beta = 0$, this is NLL's inverse-variance-weighted distribution. At $\beta = 1$, it's the original $P$. At $\beta = 0.5$, it's $P$ weighted by $1/\sigma$ — a compromise.

**Reading 2 — optima preservation.** For any $\beta$, the NLL optimum $(\hat\mu, \hat\sigma^2) = (\mu^\star, \sigma^{2\star})$ remains a stationary point of $\mathcal L_{\beta\text{-NLL}}$ — because at that point both factors in the product vanish (mean correct, variance equal to residual). So β-NLL is an interpolation through a family of losses with the *same optimum* but different gradient trajectories to it.

The empirical $\beta = 0.5$ sweet spot is not explained theoretically — it is simply the value that balances exploration of hard regions (large $\beta$) against variance calibration (small $\beta$) best on the surveyed tasks.

#### 7.3.5 Connection to the other fixes in this review

β-NLL is *purely* a loss-level fix. It is additive to:

- **Detlefsen's locality sampler (§4)** — a batching-level fix. β-NLL changes *how each gradient step weights points*; locality sampling changes *which points are in the batch*. A system with both would be strictly stronger.
- **Stirn & Knowles' KL regularisation (§6)** — a prior-level fix. β-NLL doesn't introduce a prior; it just reweights NLL. You could in principle use β-NLL inside a variational ELBO for additional regularisation.
- **Lakshminarayanan's ensembles (§2)** — β-NLL can be applied *per ensemble member* with no modifications. Each member trains faster (because NLL's symmetry-breaking failure is averted) and the ensemble still captures epistemic uncertainty from initialisation variance.

It is *not* compatible with the evidential regression of Amini et al. (§3) out of the box,
because DER's loss is the negative log of a Student-$t$ marginal rather than a Gaussian NLL.
However, the same reweighting principle can be applied: multiply DER's NLL by
$\lfloor\mathbb E_q[\sigma^2]^\beta\rfloor = \lfloor(\beta/(\alpha-1))^\beta\rfloor$. This
is not explored in the literature reviewed here and would be a legitimate extension.

#### 7.3.6 The symmetry-breaking argument (§3.1, Fig. 3)

A secondary contribution. The initial failure mode is not purely about inverse-variance
weighting — it is also that the network's feature space is *locally linear* and *symmetric*
around zero residual early in training. The NLL objective at a locally linear fit is at a saddle
point: any direction of mean adjustment increases the loss somewhere. The authors show (Jacobian
variance metric, Eq. 5) that regions where the feature space remains flat (low Jacobian
variance) are precisely the regions the NLL-trained model fails to fit.

**Implication.** β-NLL works not only because it reweights the gradient, but also because by
routing a larger gradient through the hard regions, it *breaks the feature-space symmetry faster*.
This is why $\beta$ values greater than 0 (not just $\beta = 0.5$) all help on the sinusoid —
any upweighting of hard regions provides the necessary symmetry breaking.

#### 7.3.7 Bridges in both directions

- **To §1 (Kendall & Gal).** The K-8 loss is a special case of β-NLL with $\beta = 0$. The
  Seitzer et al. result implies Kendall & Gal's depth estimation networks (and every other
  vision paper using a heteroscedastic Gaussian NLL) would likely benefit from β-NLL with
  $\beta=0.5$.
- **To §2 (Deep ensembles).** Drop-in compatible.
- **To §3 (DER).** Requires adaptation but conceptually applicable.
- **To §4 (Detlefsen).** Complementary; both should be used together for best results.
- **To §6 (Stirn & Knowles).** Different frame (loss vs. prior) but same diagnostic — both
  identify the $1/\sigma^2$ factor in the mean gradient as the instability source.
- **To §8 (Meinert et al.).** Meinert et al. will argue that β-NLL's fix for collapse (which DER
  lacks) is part of why deep ensembles with β-NLL remain the gold standard for uncertainty
  decomposition, while DER does not.

---

## 8. Meinert, Gawlikowski & Lavin (2023) — *The Unreasonable Effectiveness of Deep Evidential Regression*

**Venue:** AAAI 2023. **Domain:** Regression — critical analysis of Deep Evidential Regression
(§3 of this review).

### Layer 1 — Basic introduction and concept

Meinert et al. is the "rebuttal" of this review. Where Amini et al. (§3) propose DER as a
single-forward-pass alternative to MC Dropout / ensembling for aleatoric + epistemic decomposition,
Meinert et al. argue — with explicit derivations and convergence analyses — that **DER's
decomposition does not mean what it claims**. Specifically:

1. **DER's loss is overparameterised.** There exists a one-dimensional family of parameter
   settings along which the NLL is *independent of the data*. Minimising DER's loss therefore
   does not uniquely identify the NIG hyperparameters — it can "minimise" by sliding along this
   unidentified direction.
2. **Empirically, DER works via a convergence-speed heuristic.** The virtual-observation count
   $\nu$ collapses rapidly toward zero in data-sparse regions, driven by the regulariser
   $|y - \gamma|\cdot\Phi$. The proxy $1/\nu$ therefore grows in data-sparse regions — which
   *looks* like epistemic uncertainty but is actually a measure of *convergence speed*, not a
   Bayesian posterior variance.
3. **The "aleatoric" quantity $\beta/(\alpha-1)$ is also wrong.** Meinert et al. show that what
   DER actually learns is the *width of the Student-$t$ posterior predictive*,
   $w_{\rm St} = \sqrt{\beta(1+\nu)/(\alpha\nu)}$. This is a much more sensible aleatoric
   proxy — and is materially different from $\beta/(\alpha-1)$ in practice.

**Proposed redefinition.** Meinert et al. propose a corrected interpretation:
$u'_{\rm al} = w_{\rm St}$ and $u'_{\rm ep} = 1/\sqrt\nu$. Under this reinterpretation, DER's
outputs become sensible heuristics — *but not* Bayesian uncertainties in the strict sense.

**Why this matters for precision modulation.** If the GridWorld Pain Phase 3 plan imports DER
to get a per-channel aleatoric/epistemic split, Meinert et al. is mandatory reading. The split
DER claims to provide is not what it provides; the quantities being used as "precision" are
convergence-speed proxies. For an RL setting where fast early convergence is desirable, this
may coincidentally work; for the long-horizon, stable-behaviour target of the project, the
heuristic interpretation is shaky ground.

### Layer 2 — Main results and algorithm

**DER recap (Eq. M-2 in paper).** With NIG hyperparameters $m = (\gamma, \nu, \alpha, \beta)$,
the Student-$t$ posterior predictive is

$$
\mathcal L_i^{\rm NIG} \;=\; \mathrm{St}_{2\alpha_i}\!\left(y_i\,\Big|\,\gamma_i,\;\frac{\beta_i(1+\nu_i)}{\nu_i\alpha_i}\right),
\tag{M-2}
$$

and the canonical DER uncertainties (Eq. M-3):

$$
u_{\rm al}^2 \;\equiv\; \mathbb E[\sigma_i^2] \;=\; \frac{\beta_i}{\alpha_i - 1},
\qquad
u_{\rm ep}^2 \;\equiv\; \mathrm{Var}[\mu_i] \;=\; \frac{\mathbb E[\sigma_i^2]}{\nu_i} \;=\; \frac{\beta_i}{\nu_i(\alpha_i - 1)}.
\tag{M-3}
$$

**The overparameterisation theorem (Eq. M-5).** DER's NLL $\mathcal L^{\rm NIG}$ is *not* uniquely
minimised in $(\nu, \beta)$: there is a one-parameter family $(\nu, \beta(\nu))$ along which
the NLL is flat:

$$
\frac{\partial}{\partial \nu_i}\,\log\mathcal L^{\rm NIG}_i \;=\; 0 \quad\text{if}\quad \beta_i(\nu_i) \;\propto\; \frac{1}{1 + \nu_i^{-1}}.
\tag{M-5}
$$

Consequently, the optimizer can minimise the loss by sending $\nu \to 0$ along the trajectory
$\beta = 1/(1 + \nu^{-1}) = \nu/(1+\nu) \to 0$ — *independent of the data*. This is the core
theoretical critique.

**Regulariser gradient (Eq. M-7).** The evidence regulariser $\mathcal L^R = \lambda|y - \gamma|\Phi$ with $\Phi = 2\nu + \alpha$ has gradient

$$
\frac{\partial \mathcal L_i^R}{\partial \nu_i} \;\propto\; \lambda\,|y_i - \gamma_i|,
\tag{M-7}
$$

so **large residuals push $\nu$ down fast** in the early epochs. In data-sparse regions, residuals are initially large (random initialisation) and $\nu$ collapses there rapidly. In data-dense regions, residuals decay faster, so $\nu$ stays moderate.

**Width-of-$t$ redefinition (Eq. M-9).**

$$
w_{\rm St} \;=\; \sqrt{\frac{\beta_i(1+\nu_i)}{\alpha_i \nu_i}}.
\tag{M-9}
$$

This is the *scale parameter* of the Student-$t$ posterior predictive — the direct analogue of $\sigma$ for a Gaussian. For moderate $\alpha$ the Student-$t$ approximates a Gaussian and $w_{\rm St}$ is therefore the natural aleatoric uncertainty.

**Proposed redefinition (Eq. M-10).**

$$
u'_{\rm al} \;\equiv\; w_{\rm St} \;=\; \sqrt{\frac{\beta_i(1+\nu_i)}{\alpha_i \nu_i}},
\qquad
u'_{\rm ep} \;\equiv\; \frac{u_{\rm ep}}{u_{\rm al}} \;=\; \frac{1}{\sqrt{\nu_i}}.
\tag{M-10}
$$

**Modified loss with $w_{\rm St}$-normalised residual (Eq. M-11).**

$$
\mathcal L_i(\omega) \;=\; -\log \mathcal L_i^{\rm NIG}(\omega) \;+\; \lambda\left|\frac{y_i - \gamma_i}{w_{\rm St}}\right|\sqrt{\Phi},
\tag{M-11}
$$

which prevents regions with genuinely high aleatoric noise from driving $\nu$ spuriously toward zero.

**Empirical highlights.**

- **Synthetic cubic regression** $y = x^3 + \mathcal N(0, 9)$, matching Amini et al.'s original experiment. On $x\in[-4, 4]$ (training) vs $x\in[-7, 7]$ (test). 50 seeds, 500 epochs.
- **Fig. 2a (SOTA DER).** DER's $u_{\rm al}$ shows a peaked structure around the middle of the training region — but the true noise is *constant* $\sigma=3$ everywhere. The peak is ~0.7, severely underestimating true noise.
- **Fig. 2b (redefined $u'_{\rm al} = w_{\rm St}$).** The peaked structure is now captured in the *aleatoric* dimension. $u'_{\rm al} \approx 3$ matches the true $\sigma$.
- **Fig. 2c (modified loss M-11).** Cleaner separation; $u'_{\rm al}$ flat at ~3, $u'_{\rm ep}$ rises in OOD regions.

### Layer 3 — Graduate-level deep dive with full derivations

#### 8.3.1 Why the Student-$t$ marginal is overparameterised → Eq. M-5

The Student-$t$ density with location $\gamma$, scale $s^2$, and $\nu_t$ degrees of freedom is

$$
\mathrm{St}_{\nu_t}(y\mid\gamma, s^2) \;\propto\; \Big[1 + \frac{(y-\gamma)^2}{\nu_t\, s^2}\Big]^{-(\nu_t+1)/2}.
\tag{M.A}
$$

DER identifies $\nu_t = 2\alpha$ and $s^2 = \beta(1+\nu)/(\nu\alpha)$. The log-density is

$$
\log \mathrm{St}_{2\alpha}(y\mid\gamma, s^2) \;=\; \text{const}(\alpha) \;-\; \tfrac{1}{2}\log s^2 \;-\; \big(\alpha + \tfrac{1}{2}\big)\log\!\Big[1 + \frac{(y-\gamma)^2}{2\alpha s^2}\Big].
\tag{M.B}
$$

**The key observation.** The log-density depends on $(\nu, \beta)$ only through $s^2 = \beta(1+\nu)/(\nu\alpha)$. Two different $(\nu, \beta)$ pairs that produce the same $s^2$ are indistinguishable from the data's perspective. Solving $s^2 = \beta(1+\nu)/(\nu\alpha) = $ const for $\beta$:

$$
\beta(\nu) \;=\; \frac{s^2\alpha\nu}{1+\nu} \;=\; \frac{s^2\alpha}{1 + \nu^{-1}},
\tag{M.C}
$$

which is exactly the paper's M-5 up to a constant. This is the "valley" in the loss landscape: along this curve, $s^2$ stays constant, so the likelihood is flat, so the NLL gradient $\partial_\nu \log \mathcal L^{\rm NIG}$ vanishes.

**Implication.** Without additional constraints, $\nu$ and $\beta$ are *not separately identifiable*. The DER regulariser $\lambda|y-\gamma|\Phi$ introduces a constraint, but:

- It depends on $\nu + \alpha/2$ (the total evidence $\Phi$), not on $\nu$ alone.
- It scales with the residual $|y - \gamma|$, which early in training reflects *fit quality*, not *data noise*.

So the regulariser's effect on $\nu$ is indirect and empirical: it pushes $\nu$ down where fits are bad. The resulting $\nu$ is a **convergence-speed proxy**, not a Bayesian posterior concentration.

#### 8.3.2 The convergence-speed interpretation (§2.1, Fig. 1)

Meinert et al. track the parameters $\gamma_i, \nu_i, \alpha_i, \beta_i$ over training and find:

- **Large residuals in the first few epochs** (in data-sparse regions) yield large gradients on $\nu$ via M-7: $\partial_\nu \mathcal L^R \propto \lambda|y-\gamma|$. So $\nu$ drops fastest there.
- As training progresses, $\gamma$ fits reduce residuals, and the gradient on $\nu$ slows down. $\nu$ stabilises at whatever value it reached.
- Consequently, **initial fit quality fixes final $\nu$**. Regions that fit slowly end up with small $\nu$, hence large $u_{\rm ep} = 1/\sqrt\nu$.

This is not Bayesian epistemic uncertainty. It is convergence-speed. For the GridWorld Pain Phase 3 setting, the implication is specific: a precision-head that reports "high epistemic uncertainty" when it merely *hasn't finished training on a region* will mislead the modulator into gating decisions that do not correspond to genuine model ignorance.

#### 8.3.3 Why $\beta/(\alpha-1)$ is the wrong aleatoric quantity (§2.3)

The Student-$t$ variance (when defined, $\alpha > 1$) is

$$
\mathrm{Var}[y\mid m] \;=\; \frac{\nu_t}{\nu_t - 2}\,s^2 \;=\; \frac{2\alpha}{2\alpha - 2}\cdot\frac{\beta(1+\nu)}{\nu\alpha} \;=\; \frac{\beta(1+\nu)}{\nu(\alpha-1)},
\tag{M.D}
$$

which is *not* $\beta/(\alpha-1)$. The quantity $\beta/(\alpha-1)$ is the *expectation of $\sigma^2$ under the NIG prior* — not the variance of the marginal Student-$t$ predictive. The two coincide only in the limit $\nu \to \infty$.

Meinert et al. argue (and Fig. 2a/b empirically confirms) that **what DER actually learns** is $w_{\rm St}^2 = \beta(1+\nu)/(\alpha\nu) = s^2$ — the scale of the Student-$t$. This is the correct aleatoric quantity.

**The numerical gap.** With $\alpha = 2$, $\beta = 1$, $\nu = 0.5$:

- DER's $u_{\rm al}^2 = \beta/(\alpha-1) = 1$.
- True Student-$t$ variance: $\beta(1+\nu)/(\nu(\alpha-1)) = 1\cdot 1.5/(0.5\cdot 1) = 3$.
- Width-of-$t$: $w_{\rm St}^2 = \beta(1+\nu)/(\alpha\nu) = 1.5 = 1.5$.

The three numbers are materially different. DER's canonical aleatoric quantity is off by a factor of 3 from the actual variance in this example.

#### 8.3.4 The "unreasonable" empirical success despite theoretical flaws

Meinert et al. are careful to note that DER **does work** in practice — hence the title. The explanation has two parts:

1. **OOD detection works via convergence speed.** DER's $1/\nu$ rises in data-sparse regions because $\nu$ didn't finish converging there. This coincides with the OOD detection goal, even though the mechanism is not Bayesian.
2. **Regions of high true noise happen to coincide with slow convergence.** When genuine aleatoric noise is high, residuals are large, $\gamma$ takes longer to converge, $\nu$ drops more. So DER's $1/\nu$ correlates with *both* OOD-ness and high-noise regions — which looks like aleatoric+epistemic uncertainty but is actually a single signal.

This is consistent with the title: DER is "unreasonably effective" because it uses a heuristic
proxy that correlates with multiple uncertainty types without actually disentangling them.

#### 8.3.5 The modified loss (M-11) and why it helps

The paper's fix replaces the bare residual $|y-\gamma|$ in the regulariser with $|y-\gamma|/w_{\rm St}$ — the *normalised* residual:

$$
\mathcal L_i \;=\; -\log \mathcal L_i^{\rm NIG} \;+\; \lambda\,\frac{|y_i - \gamma_i|}{w_{\rm St}}\sqrt{\Phi}.
\tag{M-11}
$$

**Why $w_{\rm St}$-normalisation?** In regions of genuinely high noise, large residuals are expected. Scaling the residual by the learned noise scale $w_{\rm St}$ gives a *z-score*-like quantity — roughly $|y-\gamma|/\sigma$ — so the regulariser fires only when the residual is large *relative to the learned noise level*, not just large in absolute terms.

**Empirical effect (Fig. 2c).** With M-11, $u'_{\rm al} \approx 3$ (correct) and $u'_{\rm ep}$ rises only in OOD regions. The spurious peak around $x=0$ in the canonical DER's $u_{\rm al}$ disappears — it was an artefact of convergence-speed bleeding into the aleatoric estimate.

The $\sqrt\Phi$ factor (replacing $\Phi$) is also motivated: it makes the total-evidence penalty grow as $\sqrt\nu$ rather than $\nu$, which keeps the regulariser effective without driving $\nu$ to zero as aggressively as the original.

#### 8.3.6 Implications for this review's design question

Combining Meinert et al. with earlier entries:

- **Stirn & Knowles (§6)** argues any uncertainty head needs a KL-to-prior term; DER has $\mathcal L^R$ which is *not* a KL and is therefore not the same regularisation. Meinert et al. explains *why* this matters: without a proper KL, DER has no mechanism to prevent the overparameterisation sliding.
- **Seitzer (§7)** β-NLL could be applied to DER's Student-$t$ NLL (replace Gaussian NLL with Student-$t$ NLL in the β-weighting). This would likely help DER's mean-fit convergence, but would not fix the decomposition-identifiability problem Meinert et al. identifies.
- **Lakshminarayanan (§2)** deep ensembles remain, as of Meinert et al. (2023), the gold-standard reference against which DER is found wanting. If the GridWorld Pain Phase 3 plan needs a genuine aleatoric/epistemic split, *ensembling with heteroscedastic heads* is still the safest choice.

#### 8.3.7 Bottom line

DER is a **heuristic that works**, not a Bayesian method that decomposes uncertainty. For
applications where OOD detection is the primary goal and Bayesian semantics are not load-bearing,
DER is fine (and fast). For applications that require *faithful* aleatoric/epistemic
decomposition — which Active Inference and precision weighting philosophically are —
Meinert et al. recommends either (a) using the modified loss M-11 with the redefined
proxies M-10, or (b) abandoning DER for deep ensembles with heteroscedastic heads.

This concludes the chronological review. See §Synthesis below for cross-paper design
implications for the GridWorld Pain Phase 3 plan.

---

## Synthesis — What These 8 Papers Say Together

### Three intertwined narratives

**(I) The Gaussian NLL head is broken by default.** Every paper after Kendall & Gal (§1)
— explicitly in Detlefsen (§4), Stirn & Knowles (§6), Seitzer (§7), and implicitly in
Lakshminarayanan (§2) — identifies the same failure mode: the $1/\sigma^2$ factor in the
mean-gradient creates a self-amplifying feedback loop ("rich get richer") that leaves hard
regions unfit and inflates $\sigma^2$ to compensate. This is not a subtle theoretical concern;
Seitzer's Fig. 1 sinusoid shows it catastrophically on a trivial problem.

**(II) The fixes are at four different architectural levels.**

| Level | Method | Paper | Cost |
|---|---|---|---|
| Loss | β-NLL reweighting | §7 Seitzer | 1 line of code |
| Loss + Prior | KL-to-Gamma regulariser | §6 Stirn & Knowles | Variational ELBO machinery |
| Batching | $k$-NN locality sampler | §4 Detlefsen | $O(N^2 D)$ precompute |
| Architecture | Ensembling | §2 Lakshminarayanan | $M\times$ compute |

These fixes are **complementary**, not alternatives. A maximally reliable precision head should
use β-NLL (cheap, additive) + ensembling (separate fix for epistemic) + — if data is scarce — a
KL-to-Gamma regulariser for additional barrier protection.

**(III) Single-pass aleatoric/epistemic decomposition is harder than it looks.** Kendall & Gal
(§1) decompose via MC Dropout ($T$ forward passes); Lakshminarayanan (§2) via $M$ ensembles;
Amini (§3) claims to do it analytically via the NIG posterior. Meinert (§8) proves Amini's
decomposition is not well-defined: the NIG hyperparameters are overparameterised, and what DER
learns as "epistemic" is actually a convergence-speed proxy. As of 2023, **deep ensembles with
heteroscedastic heads remain the gold standard** for genuine decomposition. CQR (§5) offers a
strictly different product — distribution-free intervals — at the cost of Bayesian semantics.

### Cross-paper mechanism table

| Mechanism | §1 KG | §2 DE | §3 DER | §4 Det | §5 CQR | §6 SK | §7 βN | §8 Me |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Gaussian NLL head | ✓ | ✓ | — | ✓ | — | ✓ (precision form) | ✓ | — |
| Student-$t$ predictive | — | — | ✓ | ✓ | — | ✓ | — | ✓ |
| Epistemic via sampling | ✓ (MCD) | ✓ (ensemble) | — | — | — | — | — | — |
| Epistemic analytic | — | — | ✓ | — | — | — | — | critiqued |
| KL-to-prior regulariser | — | — | — | — | — | ✓ | — | — |
| Distribution-free | — | — | — | — | ✓ | — | — | — |
| Diagnoses collapse | — | — | — | ✓ | — | ✓ | ✓ | — |
| Fixes collapse | — | partially | — | ✓ (batching) | — | ✓ (prior) | ✓ (loss) | ✓ (redef) |

### Recommended recipe for GridWorld Pain Phase 3

Given the §8 critique of DER and the converging diagnosis across §4, §6, §7, the recommended
architecture for a Phase 3 precision head is:

1. **Heteroscedastic head with log-variance parameterisation.** $s(x) = \log\hat\sigma^2(x)$
   (Kendall & Gal K-8), via softplus(g(x)) if preferred (Lakshminarayanan).
2. **β-NLL loss with β = 0.5.** (Seitzer.) One-line fix; do not start with bare NLL.
3. **Auxiliary target.** An observation-reconstruction or next-step-prediction auxiliary,
   not RL return alone. This gives the variance head real gradient signal (Detlefsen + Stirn &
   Knowles both argue this is load-bearing).
4. **Monitoring (mandatory, not optional).** Log γ distribution, log σ̂² distribution, and
   variance-calibration metrics (Stirn & Knowles' PPCs §6.3.7). Collapse is silent; only
   explicit monitoring catches it.
5. **Ensemble of 3–5 members for epistemic.** (Lakshminarayanan.) Use per-member heteroscedastic
   heads under β-NLL. Combine via mixture-of-Gaussians moment formula (L-2).
6. **Avoid DER.** Unless specifically using the Meinert et al. corrections (M-10, M-11),
   DER will not provide the decomposition semantics the project needs.

Distribution-free alternative: if the project can tolerate a different uncertainty semantics
(prediction interval width rather than inverse-variance precision), **conformalised** any of
the above via CQR (§5). The guarantee is weaker in form but stronger in finite-sample validity.

### Open questions not addressed by any of these 8 papers

- **Combining β-NLL with DER or Stirn & Knowles.** No published experiments; likely additive.
- **Ensemble of variational-variance models.** Stirn & Knowles did not ensemble; would likely
  further improve PPCs.
- **Conformal calibration of heteroscedastic heads in RL.** CQR has not been applied to the
  RL setting (online, non-exchangeable data); this is an active research area outside the
  scope of these 8 papers.
- **Precision modulation applications.** None of these 8 papers directly addresses the
  FiLM-Ensemble-like setting where log σ̂² is *reused as a gating signal*. This is
  architecturally specific to the Turkoglu et al. (2022) line and should be reviewed separately
  in the perceptual_noise_lit_review.md (§2, §3).

## Change Log

| Date | Paper | Change |
|------|-------|--------|
| 2026-04-14 | — | Skeleton created; chronological order locked; per-paper 3-layer template established |
| 2026-04-14 | 1 — Kendall & Gal | Full 3-layer entry: MC-Dropout VI derivation, K-1 → K-12 with line-by-line derivations, gradient analysis (L.1.9–L.1.10) connecting to downstream critiques |
| 2026-04-14 | 2 — Lakshminarayanan et al. | Full 3-layer entry: proper-scoring-rule theory (Gibbs' inequality), mixture-of-Gaussians variance decomposition, MSE vs NLL collapse argument, softplus vs log-variance, FGSM derivation, bridges to DER/β-NLL/Meinert |
| 2026-04-14 | 3 — Amini et al. (DER) | Full 3-layer entry: NIG conjugacy, moment derivations (S6–S13), Student-$t$ marginalisation (S15–S26), NLL loss derivation, evidence regulariser analysis, forward-pointers to Meinert critique |
| 2026-04-14 | 4 — Detlefsen et al. | Full 3-layer entry: local-likelihood diagnostic, HT unbiased estimator derivation, MV coordinate-ascent justification, IG → Student-$t$ marginalisation (Appendix B), GP extrapolation architecture. Attribution note on Skafte/Detlefsen name |
| 2026-04-14 | 5 — Romano et al. (CQR) | Full 3-layer entry: pinball-loss minimiser derivation, exchangeability + rank lemma, Theorem 1 proof sketch (Steps 1–4), comparison of conformity-score choices (a/b/c), Theorem 2 union bound, distinction from parametric methods |
| 2026-04-14 | 6 — Stirn & Knowles | Full 3-layer entry (rewritten after pypdf extraction workaround): Gaussian-NLL instability analysis (SK.A–SK.C), analytic expected log-likelihood derivation (SK.D–SK.E → SK-2), KL-Gamma-to-Gamma regulariser as log-barrier (SK.F), VAP ablation via Jensen (SK.G), xVAMP KL decomposition (SK-3 + SK.H–SK.I), PPC methodology |
| 2026-04-14 | 7 — Seitzer et al. (β-NLL) | Full 3-layer entry: inverse-variance-weighted data distribution derivation (S.A–S.D → S-6), rich-get-richer self-amplification (S.E), β-NLL gradient (S.F–S.G), β = 0/0.5/1 interpretation table, symmetry-breaking argument, compatibility with §1, §2, §3, §4, §6 |
| 2026-04-14 | 8 — Meinert et al. (DER critique) | Full 3-layer entry: overparameterisation derivation (M.A–M.C → M-5), convergence-speed-as-epistemic analysis, Student-$t$ variance vs $\beta/(\alpha-1)$ numerical gap (M.D), $w_{\rm St}$-normalised loss (M-11) |
| 2026-04-14 | Synthesis | Three narratives (collapse / fix-levels / decomposition-is-hard), cross-paper mechanism table, Phase 3 recipe, open questions |
