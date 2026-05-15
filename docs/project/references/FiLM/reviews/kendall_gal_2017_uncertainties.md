---
title: "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?"
authors: ["Alex Kendall", "Yarin Gal"]
year: 2017
venue: "NeurIPS 2017 (31st Conference on Neural Information Processing Systems)"
slug: kendall_gal_2017_uncertainties
source_pdf: "docs/project/references/FiLM/sources/Kendall and Gal 2017 - What uncertainties do we need in Bayesian deep learning for computer vision.pdf"
topic: FiLM
---

## Plain-English Entry Point

Deep neural networks are usually deployed as point predictors — they output a single number (regression) or a probability vector (classification), and downstream systems take that output at face value. That is dangerous when the model is wrong: the paper opens with two real disasters (a fatal Tesla Autopilot crash and a Google Photos racial mis-tagging incident) that could plausibly have been averted if the model had **flagged its own uncertainty**.

Bayesian statistics distinguishes two flavors of uncertainty that this paper teaches a deep network to expose:

- **Aleatoric uncertainty** (Latin "alea" = die roll) — the **noise inherent in the data**. Even with infinite training data, you cannot reduce it. Example: a low-resolution pixel on an object boundary in a depth map.
- **Epistemic uncertainty** (Greek "episteme" = knowledge) — the **model's ignorance about itself**. It shrinks as you collect more data. Example: a class label the model has rarely seen during training.

The authors further split aleatoric into **homoscedastic** (same noise for every input) vs **heteroscedastic** (input-dependent noise — a feature-rich image vs a blank wall). They build a single network that outputs *both* a prediction and a per-input log-variance, trained with a loss that simultaneously learns the prediction and the noise level. The same network is then made Bayesian by dropping out weights at test time and averaging over the resulting samples (**Monte Carlo dropout**), so epistemic uncertainty falls out as the variance across those samples. The headline mathematical formula — `L = ½ exp(-s) ||y - ŷ||² + ½ s`, with $s = \log \sigma^2$ — is the **anchor heteroscedastic regression loss** this project uses. The loss naturally **attenuates noisy datapoints** (high predicted noise downweights the residual term), giving the model a built-in robustness to corrupt labels.

## Section-Ordered Backbone

**Abstract.** Two kinds of uncertainty exist: aleatoric (data noise) and epistemic (model ignorance). Authors build a Bayesian deep-learning framework that captures both, derives new loss functions that can be interpreted as learned attenuation, and improves SOTA on per-pixel semantic segmentation and monocular depth regression.

**1. Introduction.** Modern deep networks lack a principled way to say "I don't know". Two motivating disasters (Tesla 2016, Google Photos 2015). The paper claims that in big-data regimes — typical for vision — aleatoric uncertainty is most useful (epistemic can be explained away), but epistemic is critical for out-of-distribution / safety-critical detection. Three contributions: (1) accurate joint capture of aleatoric and epistemic, including a novel classification approach; (2) 1–3% performance improvement from attenuation; (3) characterization of the trade-offs.

**2. Related Work.** Reviews two thrusts. **Epistemic** uncertainty in Bayesian neural networks (BNNs): place a prior on weights, do approximate inference; Monte Carlo dropout (Gal & Ghahramani) is the practical workhorse, equivalent to variational inference with a Bernoulli mixture posterior; predictive variance decomposes into observation noise + parameter variance. **Aleatoric** uncertainty: model output noise; heteroscedastic regression learns the noise as a function of $x$.

**3. Combining Aleatoric and Epistemic Uncertainty.** Make the heteroscedastic regression network Bayesian by adding dropout weights. The network has a two-headed output: predicted mean $\hat{y}$ and predicted log-variance $s = \log \hat{\sigma}^2$. The training objective becomes the heteroscedastic NLL plus weight decay; the predictive variance combines the data noise (mean predicted $\hat{\sigma}^2$ over samples) with the parameter variance (variance of predicted $\hat{y}$ over samples).

- **3.1 Combining heteroscedastic aleatoric with epistemic.** Loss: $L_{\text{BNN}}(\theta) = \frac{1}{D} \sum_i \frac{1}{2}\hat{\sigma}_i^{-2}\|y_i - \hat{y}_i\|^2 + \frac{1}{2}\log\hat{\sigma}_i^2$. Trained by predicting $s_i := \log \hat{\sigma}_i^2$ for numerical stability: $L = \frac{1}{D}\sum_i \frac{1}{2}\exp(-s_i)\|y_i - \hat{y}_i\|^2 + \frac{1}{2} s_i$.
- **3.2 Heteroscedastic uncertainty as learned attenuation.** The $\exp(-s_i)$ factor adaptively downweights noisy data; the $\frac{1}{2} s_i$ regularizer prevents the network from predicting infinite uncertainty everywhere.
- **3.3 Heteroscedastic classification.** Place Gaussian noise on the *logits* (pre-softmax), then marginalize via MC sampling. The corrupted logit becomes $\hat{x}_{i,t} = f_i^W + \sigma_i^W \epsilon_t$, $\epsilon_t \sim \mathcal{N}(0, I)$, and the loss is a numerically-stable log-sum-exp over softmax samples.

**4. Experiments.** DenseNet architecture (TensorFlow re-implementation, slightly beats the original on CamVid). Aleatoric uncertainty trained with MAP using a Laplacian prior (L1 loss); epistemic via 50 MC dropout samples ($p = 0.2$). Tested on (a) semantic segmentation — CamVid (67.5% IoU SOTA) and NYUv2 40-class — and (b) monocular depth regression — Make3D and NYUv2 Depth. Combining both uncertainties beats each alone.

**5. Analysis.**
- **5.1 Uncertainty quality.** Precision-recall curves are strictly decreasing in uncertainty for both kinds. When only one is modeled, it tries to compensate for the missing other. Calibration plots show MSE between predicted and observed frequencies improves when both uncertainties are modeled.
- **5.2 Distance from training data.** (Table 3) Aleatoric stays approximately constant as you cut training data or test on out-of-distribution data; epistemic decreases with more training data and *increases* on OOD test sets — confirming the textbook properties.
- **5.3 Real-time application.** Aleatoric is essentially free at test time (one forward pass). Epistemic requires $T$ MC samples and is expensive — DenseNet's dropout everywhere causes ~50× slowdown for 50 samples.

**6. Conclusions.** Aleatoric matters for large datasets and real-time; epistemic matters for safety-critical / OOD detection and small-data regimes. They are complementary and combine well. Open question: real-time epistemic uncertainty in deep learning.

## Phase 1 — Undergraduate-Level Synthesis

Two kinds of "I don't know" can live in a model's output, and they have **different causes and different remedies**:

1. **Aleatoric** — *data noise*. If your training images are blurry on object edges, predictions on object edges will always be noisy, no matter how much data you have. The fix is *not* more data; the fix is for the model to **report** that noise level. Heteroscedastic regression does this by adding a second output head that predicts a per-input variance $\sigma^2(x)$.
2. **Epistemic** — *model ignorance*. If you have only seen 3 examples of zebras, the model is uncertain about zebras *because it has not seen enough*. The fix is more data, or to express ignorance until data arrives. Bayesian neural networks do this by averaging predictions over many plausible weight settings (a posterior $p(W | X, Y)$). MC dropout is a tractable approximation: keep dropout on at test time, run forward $T$ times, look at the *spread* across the runs.

The key trick in this paper is the **heteroscedastic regression loss**:

$$
L = \frac{1}{2 \sigma^2(x)} \| y - \hat{y}(x) \|^2 + \frac{1}{2} \log \sigma^2(x).
$$

The first term is the usual squared error, *divided* by predicted noise. A noisy point ($\sigma^2$ large) contributes less to the loss — the model is allowed to be wrong there without penalty. But the second term (the $\log \sigma^2$) penalizes predicting *too-large* noise everywhere — that would be cheating. The two terms balance: the model is forced to predict large noise *exactly* on points it cannot fit, and only on those points.

This is **learned loss attenuation**: the model decides per-input how much to trust the label. It buys robustness to noisy/corrupt labels for free, without ever being told which labels are corrupt.

When you also turn the network Bayesian (MC dropout), the **predictive variance decomposes** into the predicted noise (aleatoric, learned by the loss above) plus the spread across MC samples (epistemic, from weight uncertainty). The takeaway: a single architecture, with one extra output head and one extra loss term, gives you both kinds of uncertainty, with no labels for uncertainty itself.

## Phase 2 — Graduate-Level Deep Dive

### Epistemic Uncertainty via MC Dropout

For a Bayesian neural network with weight prior $p(W)$, given data $(X, Y)$, we want the posterior $p(W | X, Y) \propto p(Y | X, W) p(W)$. This is intractable in general; MC dropout approximates it by a variational distribution $q^*_\theta(W)$ — a Bernoulli mixture parametrized by the dropout mask — found by minimizing $\text{KL}(q^*_\theta(W) \| p(W | X, Y))$. The resulting objective ([Gal & Ghahramani, 2016]) for $N$ data points and dropout probability $p$ is

$$
\mathcal{L}(\theta, p) = -\frac{1}{N} \sum_{i=1}^N \log p\bigl(y_i \mid f^{\widehat{W}_i}(x_i)\bigr) + \frac{1 - p}{2N} \| \theta \|^2 . \tag{1}
$$

For a Gaussian likelihood in regression,

$$
-\log p\bigl(y_i \mid f^{\widehat{W}_i}(x_i)\bigr) \propto \frac{1}{2\sigma^2} \| y_i - f^{\widehat{W}_i}(x_i) \|^2 + \frac{1}{2} \log \sigma^2 , \tag{2}
$$

with $\sigma$ the observation noise. Test-time predictions are obtained by $T$ stochastic forward passes (sampling dropout masks):

$$
\mathbb{E}[y] \approx \frac{1}{T} \sum_{t=1}^T f^{\widehat{W}_t}(x), \qquad \text{Var}(y) \approx \sigma^2 + \frac{1}{T} \sum_{t=1}^T f^{\widehat{W}_t}(x)^\top f^{\widehat{W}_t}(x) - \mathbb{E}[y]^\top \mathbb{E}[y] . \tag{3}
$$

The second and third terms together are the **sample variance across MC samples** — this is the epistemic contribution and vanishes when all sampled weights agree.

### Heteroscedastic Aleatoric Loss (the anchor formula)

Allow the noise to depend on the input: $\sigma^2 \to \sigma^2(x_i)$. With a single non-Bayesian network jointly predicting mean $f(x_i)$ and noise $\sigma(x_i)^2$ via MAP inference, the loss becomes

$$
\mathcal{L}_{\text{NN}}(\theta) = \frac{1}{N} \sum_{i=1}^N \frac{1}{2\sigma(x_i)^2} \| y_i - f(x_i) \|^2 + \frac{1}{2} \log \sigma(x_i)^2 . \tag{4}
$$

This is heteroscedastic Gaussian NLL. Note **no labels for $\sigma^2$ are needed**: the variance is learned implicitly to balance the two terms.

### Combined Bayesian + Heteroscedastic (the project's anchor formula)

Turn the heteroscedastic network into a BNN via dropout, and produce both heads from a single forward pass: $[\hat{y}, \hat{\sigma}^2] = f^{\widehat{W}}(x)$. Per-pixel loss for $D$ output pixels:

$$
\mathcal{L}_{\text{BNN}}(\theta) = \frac{1}{D} \sum_i \frac{1}{2} \hat{\sigma}_i^{-2} \| y_i - \hat{y}_i \|^2 + \frac{1}{2} \log \hat{\sigma}_i^2 . \tag{5}
$$

For numerical stability, the network outputs the **log-variance** $s_i := \log \hat{\sigma}_i^2$ rather than $\hat{\sigma}_i^2$, so the loss becomes

$$
\boxed{\;\mathcal{L}_{\text{BNN}}(\theta) = \frac{1}{D} \sum_i \frac{1}{2} \exp(-s_i)\, \| y_i - \hat{y}_i \|^2 + \frac{1}{2} s_i\;} \tag{6}
$$

(This is *the* heteroscedastic loss the project anchors on.)

**Parameterization details to be precise about:**

- $s_i$ is **log of variance**, not log of standard deviation. Some papers parameterize $s = \log \sigma$ — be careful: that version has $\exp(-2s)$, not $\exp(-s)$.
- $s$ is **unconstrained real-valued**, which is exactly why this parameterization is preferred: $\exp(-s) > 0$ for any $s \in \mathbb{R}$, so $\hat{\sigma}^2$ is automatically positive without a softplus or constraint.
- The factor $\frac{1}{2}$ is sometimes absorbed into a learning-rate or weight-decay constant; if your implementation drops it, the *ratio* of the two terms changes and the model will tune $s_i$ to a different fixed point. Keep the $\frac{1}{2}$ explicit unless you also rescale.
- The original Kendall–Gal paper actually uses an L1 (Laplacian) variant in their experiments, where the residual $\|y_i - \hat{y}_i\|^2$ becomes $|y_i - \hat{y}_i|$ and the leading $\frac{1}{2}$ is replaced — but in the project the L2 (Gaussian) form above is the standard.

**Why this loss attenuates noise.** Take derivatives w.r.t. $s_i$:

$$
\frac{\partial \mathcal{L}_i}{\partial s_i} = -\frac{1}{2} \exp(-s_i) \|y_i - \hat{y}_i\|^2 + \frac{1}{2} .
$$

Setting to zero gives the optimum $s_i^* = \log \|y_i - \hat{y}_i\|^2$ — i.e., the model wants to predict log-variance equal to the log of the squared residual. So **inputs that are hard to fit get high predicted noise**, which downweights their residual contribution. The $\frac{1}{2} s_i$ regularizer keeps the model from predicting infinite noise everywhere (large $s$ is penalized linearly).

### Predictive Variance Decomposition (Aleatoric vs Epistemic)

With $T$ MC dropout samples $\{\hat{y}_t, \hat{\sigma}_t^2\}_{t=1}^T$ from the BNN, the total predictive variance for output $y$ is

$$
\text{Var}(y) \approx \underbrace{\frac{1}{T} \sum_{t=1}^T \hat{y}_t^2 - \Bigl(\frac{1}{T} \sum_{t=1}^T \hat{y}_t\Bigr)^2}_{\text{epistemic: sample variance of means}} + \underbrace{\frac{1}{T} \sum_{t=1}^T \hat{\sigma}_t^2}_{\text{aleatoric: mean of predicted variances}} . \tag{7}
$$

This is the **law of total variance** applied to the MC ensemble:

$$
\text{Var}(y) = \mathbb{E}_W[\text{Var}(y \mid W)] + \text{Var}_W[\mathbb{E}(y \mid W)] .
$$

The first term — average noise across weight samples — is the **aleatoric** contribution; the second term — variance of the mean across weight samples — is the **epistemic** contribution.

**Empirical signature** (Section 5.2, Table 3):

- Aleatoric is roughly constant whether trained on $\frac{1}{4}$ or full data, and whether tested in- or out-of-distribution.
- Epistemic decreases with more training data and increases sharply on OOD test sets.

This is the diagnostic that lets aleatoric vs epistemic be distinguished empirically: train on subsets, measure the spread.

### Heteroscedastic Classification

For classification, "noise on the output" is placed on the **logits** (pre-softmax), not on the probability vector (since adding noise to a softmax output is awkward):

$$
\hat{x}_i \mid W \sim \mathcal{N}(f_i^W,\ (\sigma_i^W)^2), \qquad \hat{p}_i = \text{softmax}(\hat{x}_i) . \tag{8}
$$

The expected log-likelihood $\log \mathbb{E}_{\mathcal{N}(\hat{x}_i; f_i^W, (\sigma_i^W)^2)}[\hat{p}_{i, c}]$ has no closed form, so MC-sample the logits:

$$
\hat{x}_{i, t} = f_i^W + \sigma_i^W \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
$$

and use the numerically-stable

$$
\mathcal{L}_x = \sum_i \log \Bigl( \frac{1}{T} \sum_t \exp(\hat{x}_{i, t, c} - \log \sum_{c'} \exp \hat{x}_{i, t, c'}) \Bigr) . \tag{9}
$$

This is also interpretable as classification loss attenuation: confident logits stay sharp, uncertain logits get spread by the Gaussian and softmax averages them out.

## Connections to the Project

- **This is the project's core uncertainty formula.** Equation (6) — $L = \frac{1}{2} \exp(-s) \|y - \hat{y}\|^2 + \frac{1}{2} s$ with $s = \log \sigma^2$ — is the heteroscedastic regression loss that the project's interoceptive / pain-uncertainty modeling components are built on. Future readers should treat *this* paper as the canonical citation when documenting or reviewing that loss. Watch for the parameterization pitfalls flagged above (log-variance vs log-std; the $\frac{1}{2}$ factor).
- **Aleatoric vs epistemic decomposition** (equation 7) tells the project how to *report* uncertainty downstream. If the project ever decides to MC-dropout-sample the policy or value network, the same decomposition applies: aleatoric = task irreducibility (think "nociceptor noise"), epistemic = exploration target ("the agent has not visited this state enough"). Both have clear behavioral readouts and map cleanly to neuromodulator candidates (e.g., NE encoding "unexpected uncertainty" $\sim$ epistemic, ACh encoding "expected uncertainty" $\sim$ aleatoric — see professor-neuromodulation memos).
- **Connects to FiLM-ensemble (Turkoglu 2022) and the broader uncertainty survey (Gawlikowski 2023).** Kendall-Gal's MC dropout is the *workhorse* deep-Bayesian method; Turkoglu 2022 in this same corpus proposes FiLM-ensemble as a cheaper alternative that gives you $M$ "virtual" ensemble members without storing $M$ networks. Gawlikowski 2023 places Kendall-Gal squarely in the "MC dropout / Bayesian approximation" branch of the uncertainty taxonomy. If the project is comparing uncertainty methods, this paper anchors the **heteroscedastic + MC dropout baseline** that ensemble and evidential methods must beat.
- **Attenuation as a robustness mechanism.** Section 3.2 frames the heteroscedastic loss as **learned attenuation** of noisy data — exactly the kind of self-supervised robustness the project's pain-modeling components benefit from (e.g., when noxious-stimulus labels are themselves noisy, the model can flag them as high-aleatoric without explicit corruption labels).
- **Real-time caveat.** Section 5.3 notes that MC dropout for epistemic uncertainty is expensive (~50× slowdown for 50 samples on architectures with widespread dropout). For real-time RL rollouts in this project, aleatoric is essentially free (one forward pass), epistemic is not — a planning constraint to flag for `senior-developer` if epistemic is desired in the RL inner loop.
