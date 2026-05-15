---
title: "A Survey of Uncertainty in Deep Neural Networks"
authors: ["Jakob Gawlikowski", "Cedrique Rovile Njieutcheu Tassi", "Mohsin Ali", "Jongseok Lee", "Matthias Humt", "Jianxiang Feng", "Anna Kruspe", "Rudolph Triebel", "Peter Jung", "Ribana Roscher", "Muhammad Shahzad", "Wen Yang", "Richard Bamler", "Xiao Xiang Zhu"]
year: 2023
venue: "Artificial Intelligence Review"
slug: gawlikowski_2023_uncertainty_survey
source_pdf: "docs/project/references/FiLM/sources/Gawlikowski et al. 2023 - A survey of uncertainty in deep neural networks.pdf"
topic: FiLM
---

## Plain-English Entry Point

This is the **map of the territory** for uncertainty in deep neural networks — a 77-page survey that organizes a decade of work on "how should a neural network express what it does not know?" into a single taxonomy. The motivation is the same as in Kendall & Gal 2017 (see the companion review): a network that returns only a point prediction is dangerous when wrong, and modern deep learning is rife with well-known failure modes (over-confident softmax outputs, brittleness to distribution shift, adversarial-attack vulnerability, missed out-of-distribution samples).

The survey draws **five sources of uncertainty** (real-world variability, sensor noise, model-architecture choices, randomness in training, and inputs the network has never seen) and groups them into the standard two-bucket vocabulary: **data / aleatoric** uncertainty (irreducible noise) and **model / epistemic** uncertainty (model's ignorance about itself, reducible with more data). A third optional bucket — **distributional** uncertainty — separates "the data distribution is different from what I trained on" from generic model ignorance.

Its core contribution is the **four-branch taxonomy of methods** for quantifying uncertainty:

1. **Single deterministic networks** — one forward pass; uncertainty either output as an extra head (e.g., evidential / Dirichlet networks) or derived externally.
2. **Bayesian neural networks (BNNs)** — place a distribution on the weights; approximate the intractable posterior with variational inference, Markov-chain Monte Carlo (MCMC) sampling, or Laplace approximation. MC dropout (Gal & Ghahramani; Kendall & Gal) lives here.
3. **Ensembles** — train $M$ deterministic networks with different seeds; their disagreement is the model uncertainty.
4. **Test-time augmentation** — augment the test sample $M$ ways through one deterministic network; their disagreement is the uncertainty.

Plus chapters on **measuring** uncertainty (entropy, mutual information, predictive variance, expected calibration error), **calibrating** networks (temperature scaling, regularization), benchmarks, and applications in medical imaging, robotics, and earth observation. For the project, this is the high-level reference doc that **places Kendall & Gal 2017 in context** and is the canonical citation for the aleatoric/epistemic split as a *field-wide* convention rather than one paper's claim.

## Section-Ordered Backbone

**Abstract.** Surveys uncertainty estimation in DNNs. Introduces the reducible (model / epistemic) vs irreducible (data / aleatoric) split. Reviews four method families (deterministic, BNN, ensemble, test-time augmentation). Covers measures, calibration, benchmarks, and applications in medical, robotics, and earth-observation.

**1. Introduction.** Four obstacles to deploying DNNs in safety-critical settings: lack of expressiveness/transparency; inability to separate in- vs out-of-domain inputs; unreliable uncertainty estimates / overconfidence; adversarial-attack vulnerability. Lists prior surveys (Ghanem 2017, Gal 1998, Kendall 2019, Wang & Yeung 2016/2020, Abdar 2021, Hüllermeier & Waegeman 2021) and stakes its own contribution: a clear thread from uncertainty sources to applications, with practical caveats.

**2. Uncertainty in deep neural networks.** A four-step pipeline: (1) data acquisition, (2) DNN design + training, (3) inference, (4) prediction's uncertainty model. Five factors:

- *Factor I*: variability in real-world situations (distribution shift)
- *Factor II*: noise/error in measurement systems → aleatoric
- *Factor III*: errors in model structure
- *Factor IV*: errors in training procedure
- *Factor V*: errors from unknown data (OOD)

Only Factor II is aleatoric; the rest are epistemic. Section 2.4 lays out the predictive distribution:

$$
p(y^* \mid x^*, \mathcal{D}) = \int \underbrace{p(y^* \mid x^*, \theta)}_{\text{data}}\, \underbrace{p(\theta \mid \mathcal{D})}_{\text{model}}\, d\theta,
$$

and 2.4.2 introduces the optional three-way decomposition (data / distributional / model). Section 2.5 cross-cuts these as in-domain, domain-shift, and out-of-domain uncertainty by input region.

**3. Uncertainty estimation (the four branches).**

- **3.1 Single deterministic methods.** Internal approaches predict distribution parameters (Dirichlet via Prior Networks of Malinin & Gales, Evidential Networks of Sensoy et al., Mixtures of Dirichlets, normal-inverse-gamma for regression in Amini et al. 2020). External approaches add a separate uncertainty estimator on top of a frozen network (Raghu, Ramalho-Miranda, Lee-AlRegib, Oberdiek).
- **3.2 Bayesian neural networks.** Posterior $p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta) p(\theta)$; predictive integral $p(y^* \mid x^*, \mathcal{D}) = \int p(y^* \mid x^*, \theta) p(\theta \mid \mathcal{D}) d\theta$. Three families:
  - **3.2.1 Variational inference.** Approximate posterior with a tractable family $q(\theta)$; minimize $\mathrm{KL}(q \| p)$ via the ELBO. Bayes by Backprop (Blundell 2015), local reparameterization (Kingma 2015), MC dropout (Gal & Ghahramani 2016) as a special case, hierarchical priors (Wu 2018).
  - **3.2.2 Sampling.** MCMC, HMC, SGHMC, SG-Langevin dynamics. Theoretically unbiased; expensive; hard to assess convergence.
  - **3.2.3 Laplace approximation.** Fit a Gaussian to the posterior at its MAP mode using the Hessian / Fisher (Ritter 2018, KFAC). Applies to pre-trained networks; cheap; single-mode.
- **3.3 Ensembles.**
  - **3.3.1 Principles.** Train $M$ networks; average / vote; disagreement = model uncertainty.
  - **3.3.2 Single vs multi-mode.** Ensembles uniquely capture multiple loss-landscape modes (Fort et al. 2019); BNNs and deterministic methods are single-mode.
  - **3.3.3 Bringing variety.** Random init, data shuffling, bagging/boosting, augmentation, mixed architectures. Random init + shuffling alone is usually sufficient.
  - **3.3.4 Uncertainty quantification.** Deep Ensembles (Lakshminarayanan 2017): two-headed members (prediction + per-input variance); often outperform MC dropout (Gustafsson 2020, Ovadia 2019).
  - **3.3.5 Efficiency.** Pruning, ensemble distillation (Hinton 2015; Malinin 2020), sub-ensembles (shared trunk, per-member task head), batch-ensembles (Wen 2019; weights as $W \odot r_i s_i^\top$). **FiLM-ensemble (Turkoglu 2022) sits in this efficiency family.**
- **3.4 Test-time augmentation.** Augment the test sample $M$ ways through one deterministic network; aggregate. Cheap, no retraining; popular in medical imaging.
- **3.5 Real-life considerations.** Trade-offs: ensembles + TTA easy but $M\times$ inference cost; BNNs deeper-theoretic but harder to engineer; single deterministic cheapest but limited.

**4. Uncertainty measures.** For **classification**: maximum class probability $\max_k p_k$, entropy $H(p) = -\sum_k p_k \log p_k$, mutual information $\text{MI}(\theta, y \mid x, \mathcal{D}) = H[\hat{p}] - \mathbb{E}_{\theta \sim p(\theta \mid \mathcal{D})}[H[p(y \mid x, \theta)]]$ (model uncertainty), expected KL, predictive variance, OOD-specific Dirichlet-precision measures. AUROC / AUPRC for full-dataset evaluation. For **regression**: predicted standard deviation (Kendall-Gal), prediction intervals (MPIW, PICP). For **segmentation**: per-pixel + structure-level (Dice, IoU) variants.

**5. Calibration.** A network is calibrated if predicted confidence equals empirical accuracy (eqs. 37–38). Deeper networks tend to be over-confident (Guo 2017). Three method classes:

- **5.1.1 Regularization** during training: label smoothing, focal loss, mixup, entropy regularization.
- **5.1.2 Post-processing**: temperature scaling, Platt scaling, isotonic regression, Bayesian binning, Beta calibration.
- **5.1.3 Calibration via uncertainty methods**: ensembles and MC dropout improve calibration as a side effect.

**5.2 Evaluating calibration.** Expected Calibration Error (ECE), Maximum Calibration Error (MCE), Adaptive ECE, Negative Log-Likelihood, Brier Score, reliability diagrams.

**6. Datasets and benchmarks.** MNIST/CIFAR variants for in-distribution; SVHN, LSUN, Tiny-ImageNet, MNIST-C, ImageNet-C for OOD and shift; rotated MNIST and permuted MNIST for continual; UCI for regression. Lists open-source implementations under their GitHub repo.

**7. Applications.** Active learning, reinforcement learning (Gal/Ghahramani 2016, Kahn 2017, Lütjens 2019, Huang 2019), medical imaging, robotics, earth observation. Each section catalogs uncertainty-aware DNN deployments and open problems.

**8. Conclusion and outlook.** Open problems: principled prior selection for BNNs; efficient sampling at inference; rigorous OOD benchmarks beyond MNIST/CIFAR; combinations of methods (e.g., BNN + ensemble); reliable calibration under shift; uncertainty propagation through pipelines.

## Phase 1 — Undergraduate-Level Synthesis

Picture a neural network as a function with three places where uncertainty can live:

1. **In the input.** The image is blurry; the sensor is noisy; the label was wrong. → **aleatoric / data uncertainty**. Cannot be reduced even with infinite data. Best handled by giving the network a *second output head* that predicts how noisy this input is.
2. **In the weights.** The training set was too small, or hit a bad random seed, or stopped early. The network is unsure about *itself*. → **epistemic / model uncertainty**. Goes away with more data (or better training). Best handled by treating weights as random and computing the spread of predictions over many weight settings.
3. **In the test distribution.** The training set was cats and dogs; the test input is a bird. → **distributional / out-of-distribution uncertainty**. A special case of epistemic, but worth its own bucket because it triggers the safety-critical "do not trust this prediction" flag.

The survey shows **four ways to actually compute these uncertainties** in practice:

| Method family | One-line idea | Key paper | Cost |
|---|---|---|---|
| **Single deterministic** | One forward pass; the network outputs both prediction and uncertainty (e.g., a Dirichlet over class probabilities, or a variance for regression) | Sensoy 2018 (evidential), Malinin & Gales 2018 (prior networks), Kendall-Gal 2017 (heteroscedastic head) | Cheap |
| **Bayesian NN** | Put a distribution on the weights; average predictions over $M$ samples from that distribution | Blundell 2015 (Bayes-by-Backprop), Gal & Ghahramani 2016 (MC dropout), Ritter 2018 (Laplace) | Expensive at inference (M forward passes) |
| **Ensemble** | Train $M$ deterministic networks separately; their disagreement is the uncertainty | Lakshminarayanan 2017 (Deep Ensembles) | Expensive: $M \times$ memory and $M \times$ inference |
| **Test-time augmentation** | One trained network; at test time, apply $M$ different augmentations and aggregate | Ayhan & Berens 2018 | Cheap to train, $M \times$ inference |

A few practical takeaways the survey hammers home:

- **MC dropout** (a BNN special case) is the most widely-used baseline because you only need to keep dropout *on at test time* — minimal code change.
- **Deep Ensembles** often beat MC dropout, especially under distribution shift (Ovadia 2019). The trade-off is $M \times$ everything.
- **Bare softmax outputs are not calibrated** — a 90%-confident prediction is *not* right 90% of the time. Temperature scaling fixes this with one extra scalar parameter.
- **Out-of-distribution detection is harder than in-distribution uncertainty** — it requires either OOD training data, a Dirichlet-style prior network, or careful logit-magnitude analysis.

For this project, the survey is the **canonical citation for the aleatoric / epistemic split** as a *field convention* and the **reference table** for when to choose MC dropout vs ensembles vs evidential vs FiLM-ensemble (Turkoglu 2022, which slots into the "efficient ensembles" sub-section 3.3.5).

## Phase 2 — Graduate-Level Deep Dive: The Taxonomy

### Predictive Distribution and the Aleatoric / Epistemic Decomposition

Define the predictive distribution for a new input $x^*$ given training data $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ as

$$
p(y^* \mid x^*, \mathcal{D}) = \int \underbrace{p(y^* \mid x^*, \theta)}_{\text{data uncertainty}}\, \underbrace{p(\theta \mid \mathcal{D})}_{\text{model uncertainty}}\, d\theta . \tag{1}
$$

Bayes' rule gives the posterior

$$
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) p(\theta)}{p(\mathcal{D})}, \qquad p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta) p(\theta) d\theta . \tag{2}
$$

The integral $p(\mathcal{D})$ (the *evidence*) is intractable for deep networks — this is what every Bayesian approximation method is trying to avoid computing. The optional three-way decomposition (Malinin & Gales 2018) inserts a *distributional* intermediate $\mu$ (e.g., a Dirichlet over class probabilities):

$$
p(y^* \mid x^*, \mathcal{D}) = \iint \underbrace{p(y \mid \mu)}_{\text{data}}\, \underbrace{p(\mu \mid x^*, \theta)}_{\text{distributional}}\, \underbrace{p(\theta \mid \mathcal{D})}_{\text{model}}\, d\mu\, d\theta . \tag{3}
$$

This three-way split is what makes out-of-distribution detection cleaner than two-way: $p(\mu \mid x^*, \theta)$ measures how confidently the network places the input on the simplex of categorical distributions.

### Branch 1: Single Deterministic Methods (Evidential / Dirichlet)

Replace the softmax output with parameters of a Dirichlet:

$$
\text{Dir}(\mu \mid \alpha) = \frac{\Gamma(\alpha_0)}{\prod_c \Gamma(\alpha_c)} \prod_{c=1}^K \mu_c^{\alpha_c - 1}, \qquad \alpha_0 = \sum_c \alpha_c . \tag{4}
$$

The concentration parameters $\alpha = \exp(z)$ are produced by applying a strictly-positive transform to the logits $z$. Geometric reading on the $(K-1)$-simplex (Fig. 6 of the paper):

- **Sharp at a corner** $\Rightarrow$ confident class prediction;
- **Sharp at the center** $\Rightarrow$ high data uncertainty, low distributional;
- **Flat** $\Rightarrow$ high distributional (OOD).

Evidential networks (Sensoy 2018) train with a loss equal to the expected categorical cross-entropy *under* the Dirichlet, plus a KL regularizer pulling unconfident predictions toward a uniform Dirichlet. Prior networks (Malinin & Gales 2018) train with $\mathrm{KL}(\text{Dir}_{\text{sharp}} \| q_\theta)$ on in-distribution and $\mathrm{KL}(\text{Dir}_{\text{flat}} \| q_\theta)$ on OOD — requiring labeled OOD data.

For regression: deep evidential regression (Amini 2020) parameterizes a normal-inverse-gamma over $(\mu, \sigma^2)$; the network outputs $(\gamma, \nu, \alpha, \beta)$ and predictions are

$$
\mathbb{E}[\mu] = \gamma, \qquad \mathbb{E}[\sigma^2] = \frac{\beta}{\alpha - 1}, \qquad \text{Var}[\mu] = \frac{\beta}{\nu(\alpha - 1)} .
$$

### Branch 2: Bayesian Neural Networks

#### 2a. Variational inference

Pick a tractable family $q_\phi(\theta)$ (Gaussian, mean-field Gaussian, structured Gaussian, Bernoulli mixture). Minimize $\mathrm{KL}(q_\phi \| p(\theta \mid \mathcal{D}))$, equivalently maximize the Evidence Lower Bound (ELBO):

$$
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q_\phi}[\log p(\mathcal{D} \mid \theta)] - \mathrm{KL}(q_\phi(\theta) \| p(\theta)) . \tag{5}
$$

**Reparameterization** makes this differentiable: sample $\epsilon \sim \mathcal{N}(0, I)$, set $\theta = \mu_\phi + \sigma_\phi \odot \epsilon$, backprop through $\theta$. **Local reparameterization** (Kingma 2015) reduces gradient variance by sampling at the activation level rather than at the weight level.

**MC dropout** (Gal & Ghahramani 2016) is a special case: $q_\phi$ is a Bernoulli mixture over weight matrices (the rows are zeroed with probability $p$). The variational objective reduces to standard cross-entropy + L2 weight decay. Test-time inference is $T$ stochastic forward passes with dropout left on.

#### 2b. Sampling (MCMC)

Construct a Markov chain whose stationary distribution is $p(\theta \mid \mathcal{D})$. **Stochastic-gradient Hamiltonian Monte Carlo (SGHMC)** updates:

$$
\theta_{t+1} = \theta_t + v_t, \qquad v_{t+1} = (1 - \alpha)v_t - \eta \nabla_\theta U(\theta_t) + \sqrt{2 \alpha \eta} \xi_t, \quad \xi_t \sim \mathcal{N}(0, I),
$$

with $U(\theta) = -\log p(\theta \mid \mathcal{D})$ the potential energy. Unbiased and multi-modal in theory; convergence is hard to assess.

#### 2c. Laplace approximation

Fit a Gaussian to the posterior at the MAP $\theta^*$ using the Hessian of the log-posterior:

$$
p(\theta \mid \mathcal{D}) \approx \mathcal{N}(\theta^*,\ H^{-1}), \qquad H = -\nabla^2_\theta \log p(\theta \mid \mathcal{D}) \Big|_{\theta^*} . \tag{6}
$$

In practice $H$ is approximated by the Fisher information matrix (KFAC for tractability). Cheap, deterministic, applies to pre-trained networks; single-mode.

### Branch 3: Ensembles

Deep Ensembles (Lakshminarayanan 2017) train $M$ networks $f_{\theta_1}, \ldots, f_{\theta_M}$ with different random initializations and data shuffling. Each member has a two-headed regression output $(\hat{\mu}_i, \hat{\sigma}_i^2)$ and is trained with the Gaussian NLL (the same heteroscedastic loss from Kendall-Gal). Predictive mean and variance:

$$
\bar{\mu}(x) = \frac{1}{M} \sum_i \hat{\mu}_i(x), \qquad \bar{\sigma}^2(x) = \underbrace{\frac{1}{M} \sum_i \hat{\sigma}_i^2(x)}_{\text{aleatoric}} + \underbrace{\frac{1}{M} \sum_i \hat{\mu}_i(x)^2 - \bar{\mu}(x)^2}_{\text{epistemic}} . \tag{7}
$$

This is structurally identical to the Kendall-Gal Bayesian + heteroscedastic decomposition (eq. 7 of that review), but the M samples come from independently-trained networks rather than dropout masks. Ensembles uniquely cover **multiple modes** of the posterior; BNNs and deterministic methods are single-mode (Fort et al. 2019).

**Efficient ensembles** trade independence for parameter sharing:

- **Sub-ensembles** (Valdenegro-Toro 2019): single shared trunk, M independent task heads.
- **Batch-ensembles** (Wen 2019): member $i$'s weights are $W \odot F_i$ where $F_i = r_i s_i^\top$ is rank-one. Per-member parameter overhead is $n + m$ instead of $nm$.
- **FiLM-ensemble** (Turkoglu 2022, this corpus): per-member γ/β modulate a shared backbone — see `turkoglu_2022_film_ensemble.md` for the deep dive.
- **Ensemble distillation** (Hinton 2015, Malinin 2020): train one student to mimic the M-ensemble average (or the M-ensemble Dirichlet for higher-order moments).

### Branch 4: Test-Time Augmentation

For a test input $x^*$, generate $M$ augmentations $\{a_i(x^*)\}_{i=1}^M$ (rotations, crops, noise) and aggregate predictions:

$$
\bar{p}(y \mid x^*) = \frac{1}{M} \sum_{i=1}^M p_\theta(y \mid a_i(x^*)) . \tag{8}
$$

No retraining required; one fixed network; $M$ forward passes at test time. Best-suited where augmentations are natural (medical imaging).

### Measures of Uncertainty

For classification with $M$ predictions $\{p_i\}_{i=1}^M$ from any of the above:

- **Mean prediction**: $\hat{p} = \frac{1}{M} \sum_i p_i$.
- **Entropy** (total uncertainty): $H(\hat{p}) = -\sum_k \hat{p}_k \log \hat{p}_k$.
- **Mutual information** (model uncertainty):
$$
\text{MI}(\theta, y \mid x, \mathcal{D}) = H(\hat{p}) - \frac{1}{M} \sum_i H(p_i) . \tag{9}
$$
MI is zero when all members agree (no model uncertainty); the entropy of $\hat{p}$ is then identical to the average entropy per member, which is pure aleatoric.

- **Predictive variance** (regression): equation (7).

### Calibration

A network $f_\theta$ is *calibrated* if confidence equals empirical accuracy:

$$
\forall p \in [0, 1]: \quad \frac{\sum_i \sum_k y_{i,k} \mathbb{I}\{f_\theta(x_i)_k = p\}}{\sum_i \sum_k \mathbb{I}\{f_\theta(x_i)_k = p\}} \xrightarrow{N \to \infty} p . \tag{10}
$$

**Expected Calibration Error**: bin predictions by confidence, measure $|$ accuracy $-$ confidence $|$ per bin, average weighted by bin size:

$$
\text{ECE} = \sum_{b=1}^B \frac{|B_b|}{N} \bigl| \text{acc}(B_b) - \text{conf}(B_b) \bigr| . \tag{11}
$$

**Temperature scaling**: post-hoc, divide logits by a single learned scalar $T$ before softmax:

$$
\hat{p}_k(x) = \frac{\exp(z_k(x) / T)}{\sum_{k'} \exp(z_{k'}(x) / T)} .
$$

$T$ is fit on a validation set by minimizing NLL. One scalar; doesn't change accuracy; massively improves calibration on most over-confident deep networks (Guo 2017).

## Connections to the Project

- **Anchor citation for the aleatoric / epistemic split.** Kendall-Gal 2017 (in this same corpus) is *the* paper to cite for the heteroscedastic loss the project uses; Gawlikowski 2023 is the paper to cite when the project wants to establish that **the aleatoric / epistemic decomposition is field-standard**, not one paper's framing. Use Kendall-Gal for the formula; use Gawlikowski for the taxonomy.
- **Where the project's components fit.** The project's heteroscedastic-loss head (Kendall-Gal eq. 6) is a **single-deterministic / internal** method (Section 3.1). If the project moves to MC dropout, that's **Bayesian / variational** (Section 3.2.1). If it moves to FiLM-ensemble (Turkoglu 2022), that's **efficient ensemble** (Section 3.3.5). Evidential / Dirichlet (Sensoy / Amini) is a more aggressive single-deterministic alternative — worth flagging to `professor-rl-bayesian-dl` and `senior-developer` if the project ever needs OOD detection in the RL inner loop without M-fold inference cost.
- **OOD detection is the under-served capability.** Section 2.5 makes clear that aleatoric does *not* detect OOD; epistemic does. For the project's safety-critical reasoning (a pain agent encountering a novel stimulus), this is the gap a pure heteroscedastic head leaves open. The survey points to prior-network and evidential approaches as compatible single-pass alternatives.
- **Calibration matters for downstream RL.** If the policy or value function takes uncertainty as input, miscalibrated uncertainty directly biases the policy. Temperature scaling (eq. 11) is essentially free and should be a default. ECE / reliability diagrams (Section 5.2) belong in any uncertainty-aware evaluation suite the project builds.
- **Reinforcement-learning subsection (7.2).** Explicitly cites Gal/Ghahramani, Kahn 2017, Lütjens 2019, Huang 2019 as uses of uncertainty in RL (active exploration via Thompson sampling on epistemic uncertainty; safety via aleatoric thresholds). The project's neuromodulator-as-hyperparameter direction (see professor memos) plugs directly into this literature — ACh as expected uncertainty (aleatoric) and NE as unexpected uncertainty (epistemic) maps neatly onto the survey's framing.
- **Connections within the corpus.** Reads as a hub:
  - Kendall & Gal 2017 (`kendall_gal_2017_uncertainties.md`) → Section 3.1 (single-deterministic heteroscedastic) and 3.2.1 (variational / MC dropout).
  - Turkoglu et al. 2022 (`turkoglu_2022_film_ensemble.md`) → Section 3.3.5 (efficient ensembles).
  - Perez et al. 2018 (FiLM, Batch 1) → cited as a conditioning primitive that Turkoglu builds on.
  - Vaswani et al. 2017 (`vaswani_2017_attention.md`) → orthogonal to uncertainty methods but shares the "conditioning primitive" framing with FiLM.
