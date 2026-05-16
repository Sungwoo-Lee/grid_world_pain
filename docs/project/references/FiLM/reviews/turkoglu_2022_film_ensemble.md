---
title: "FiLM-Ensemble: Probabilistic Deep Learning via Feature-wise Linear Modulation"
authors: ["Mehmet Ozgur Turkoglu", "Alexander Becker", "Hüseyin Anil Gündüz", "Mina Rezaei", "Bernd Bischl", "Rodrigo Caye Daudt", "Stefano D'Aronco", "Jan Dirk Wegner", "Konrad Schindler"]
year: 2022
venue: "NeurIPS 2022 (36th Conference on Neural Information Processing Systems)"
slug: turkoglu_2022_film_ensemble
source_pdf: "docs/project/references/FiLM/sources/Turkoglu et al. 2022 - FiLM-ensemble - Probabilistic deep learning via Feature-wise Linear Modulation.pdf"
topic: FiLM
---

## Plain-English Entry Point

A **deep ensemble** is the gold standard for getting trustworthy uncertainty estimates from a neural network: you train $M$ independent copies of the same architecture from different random seeds, then look at how much they disagree on a new input — the spread of their predictions is the **epistemic** uncertainty (the model's ignorance about itself). The catch: training and storing $M$ networks costs $M\times$ memory and $M\times$ compute. For a 16-member ensemble of ResNet-18, that is a $1500\%$ parameter increase over a single network.

This paper proposes **FiLM-Ensemble**, an "implicit" ensemble that gets you $M$ different-acting networks while storing **one** shared backbone. The trick: only the **batch-normalization scale-and-shift parameters** (the $\gamma$ and $\beta$ from Feature-wise Linear Modulation) differ across ensemble members. Every BatchNorm layer holds $M$ different $(\gamma^m, \beta^m)$ tables, one per member, while all convolutional weights are shared. To get diverse members, the per-member $\gamma$ and $\beta$ are randomly initialized with a tunable Xavier gain $\rho$ — higher $\rho$ pushes members apart, lower $\rho$ collapses them to a single model.

Headline numbers: a 16-member FiLM-Ensemble adds **1.3%** parameters to a ResNet-18 (vs $1500\%$ for a naive deep ensemble), trains and infers in parallel on a single GPU by replicating the input along the batch axis, achieves **higher diversity** than the naive deep ensemble on CIFAR-10 (9.2% disagreement vs 6.8%), and **matches or beats** the explicit deep ensemble on accuracy + Expected Calibration Error on CIFAR-10/100, glaucoma classification, and DNA-sequence classification. For this project, FiLM-Ensemble is **the bridge between FiLM (Perez 2018) and ensemble-based uncertainty (Lakshminarayanan 2017)** — it shows that FiLM modulation is rich enough to instantiate functionally distinct sub-networks, not just to condition on a task ID.

## Section-Ordered Backbone

**Abstract.** Epistemic-uncertainty estimation is critical for deploying ML; deep ensembles are the standard but cost $M\times$. FiLM-Ensemble is an implicit ensemble: per-member γ and β modulate a shared backbone via batch-norm conditioning. Achieves high diversity, low overhead, near-deep-ensemble accuracy and calibration.

**1. Introduction.** Decomposes uncertainty as aleatoric vs epistemic (citing DerKiureghian-Ditlevsen 2009, Malinin-Gales 2018 for the distributional split). Reviews efficient-ensemble lineage: MC dropout, Snapshot, BatchEnsemble, MIMO, Masksembles, Hyperensembles. Identifies the open gap — none clearly outperforms the others, and all fall noticeably short of the explicit ensemble. Inspired by FiLM's success in multi-task learning (Perez 2018, Li 2018, Takeda 2021), proposes to use per-member γ/β as the ensemble's source of diversity. Three contributions: novel method, easy drop-in (replace BatchNorm with conditional BatchNorm), excellent accuracy-vs-calibration trade-off.

**2. FiLM-Ensemble.** Method.

- Recap of FiLM (Perez 2018): for layer $n$ with feature map $F_n$ of dimension $D_n$, compute $\gamma_n = g_n(z)$, $\beta_n = h_n(z)$ from a conditioning signal $z$, apply $\text{FiLM}(F_n \mid \gamma_n, \beta_n) = \gamma_n(z) \circ F_n + \beta_n(z)$ where $\circ$ is the Hadamard product over the channel dimension.
- **Key adaptation:** the conditioning variable $z$ becomes a *member index* $m \in \{1, \ldots, M\}$. Then $g_n$ and $h_n$ degenerate to look-up tables, so the model simply stores $M$ vectors $\gamma^m_n$ and $\beta^m_n$ per BatchNorm layer:

  $$
  \text{FiLM}(F_n \mid \gamma^m_n, \beta^m_n) = \gamma^m_n \circ F_n + \beta^m_n .
  $$
- **Diversity initialization** (the crucial design choice). All $\gamma^m_n$ and $\beta^m_n$ are sampled from a Xavier-uniform distribution bounded by $\pm \sqrt{3 / D_n} \cdot \rho$, where $\rho$ is a tunable gain. As $\rho \to 0$, members start identical and the ensemble collapses; as $\rho$ grows, members start more diverse, favoring calibration over raw accuracy.
- **Training.** Every input $x$ is fed through every member; predictions $y_m = f_{\theta, \gamma^m, \beta^m}(x)$ are averaged. All parameters (shared $\theta$, per-member $\gamma^m, \beta^m$) are optimized jointly with cross-entropy.
- **Inference.** Final prediction $\hat{y} = \frac{1}{M} \sum_m y_m$.
- **2.1 Implementation.** PyTorch; parallelize members by replicating inputs along the batch axis and broadcasting per-member γ/β. SGD with momentum 0.9, weight decay 5e-4, 200 epochs, batch 128, cosine-annealed LR from 0.1, $\rho = 2$ default.

**3. Experiments.** Datasets: CIFAR-10/100, Retinal Glaucoma (Diaz-Pinto), REFUGE2020 (for OOD), 6mA DNA-rice sequence. Baselines: single network, deep ensemble, MC dropout, Masksemble, MIMO, BatchEnsemble.

- **3.1 Diversity analysis.** Two metrics: disagreement score (fraction of test samples where two members disagree, eq. 6) and pairwise KL divergence (eq. 7). FiLM-Ensemble achieves *higher* diversity than the explicit deep ensemble at all $M \in \{2, 4, 8, 16\}$ on CIFAR-10/VGG-11.
- **3.2 Computational cost.** (Table 1) For 16-member ensembles: deep ensemble adds 1500% parameters; FiLM-Ensemble adds 0.9% (VGG-11) / 1.3% (ResNet-18). MIMO is faster on ResNet-18 but suffers severe accuracy loss.
- **3.3 CIFAR-10/100.** FiLM-Ensemble achieves the best implicit-method accuracy on both; only beaten by the explicit deep ensemble. CIFAR-100 ResNet-18 at $M=4$: FiLM-Ens 79.4% / ECE 0.038 vs deep ens 81.6% / 0.041. CIFAR-100 ResNet-34: FiLM-Ens 80.2% / 0.045 vs deep ens 82.0% / 0.044.
- **3.4 Retinal Glaucoma.** FiLM-Ensemble achieves best accuracy across all $M$: $M=16$ FiLM-Ens 87.8% beats deep ens 86.8%. ECE is competitive but not always best.
- **3.5 6mA DNA classification.** FiLM-Ensemble can be tuned via $\rho$ to trade accuracy for calibration; tunes to better calibration than deep ensemble at $<0.25$ pp accuracy cost.
- **3.6 OOD detection.** AUROC for distinguishing in-distribution Glaucoma from REFUGE OOD: FiLM-Ensemble at $M=16$ achieves 79.85%, beating deep ensemble (78.06%). The implicit ensemble's higher diversity translates to better OOD calibration.

**4. Related Work.**

- **4.1 Epistemic uncertainty.** Reviews variational inference (Graves, Blundell), MCMC (Welling-Teh, Chen 2014). Notes MCMC's difficulty exploring multi-modal deep loss landscapes.
- **4.2 Ensembles and sub-networks.** Cites Lakshminarayanan 2017 as the standard; reviews efficient alternatives (MC dropout, snapshot, BatchEnsemble, Masksembles, MIMO). "None clearly outperforms most others. The question how to effectively model uncertainty still remains open."
- **4.3 Feature-wise Linear Modulation.** Cites the FiLM lineage (De Vries 2017 Conditional BatchNorm; Perez 2018 FiLM for VQA; Strub 2018 multi-hop FiLM for dialogue; Huang-Belongie 2017 AdaIN for style; Dumoulin 2017 conditional instance norm for style; Yang 2018 video segmentation; Oreshkin 2018 TADAM for few-shot; Vuorio 2019 multimodal MAML; Vinyals 2019 AlphaStar). Explicitly notes Vuorio 2019 observation that "FiLM outperforms attention-based modulation in this context, and is more stable." Closes: *To our knowledge, FiLM has so far not been used for (implicit) model ensembling or uncertainty quantification.*

**5. Limitations & Future Work.** Pre-trained models are an open question (similar init → less diversity). Self-attention / Transformer architectures use layer norm not batch norm, so FiLM-Ensemble needs adaptation. Orthogonal extensions (per-member training data subsets à la MIMO, temperature scaling, data augmentation) could be combined.

**6. Conclusion.** A simple, effective implicit ensemble. Code at github.com/prs-eth/FILM-Ensemble.

## Phase 1 — Undergraduate-Level Synthesis

Here is the trick in three sentences. **First**, a deep ensemble's diversity comes from random initialization — each network's weights start in a different place and end in a different local minimum. **Second**, you can get a *similar* kind of diversity in a *single* network by holding the convolutions fixed and letting only the BatchNorm $\gamma$ and $\beta$ — two small vectors per layer per "member" — be different. **Third**, BatchNorm $\gamma$ and $\beta$ are exactly what FiLM modulates, so you can re-purpose the FiLM machinery (originally designed to condition on a task ID) to condition on a *member index*.

The picture:

- The shared backbone (all the conv weights, the matrix multiplications, the residuals) is trained once.
- Each BatchNorm layer carries an $M$-row table of $(\gamma, \beta)$ pairs.
- At training and inference time, the same input $x$ is fed $M$ times through the network with $m = 1, \ldots, M$ selecting which row of the table is used. The result is $M$ different predictions.
- The final prediction is their average; the *spread* across the $M$ predictions is the epistemic uncertainty.

**Why does this work?** BatchNorm $\gamma$ controls the per-channel scale of activations and $\beta$ controls the per-channel offset. With sufficient initial spread (the $\rho$ gain), members start with different feature emphases, and the training dynamics keep them apart — because the shared weights are optimized for the *mixture*, not for any one member, so the per-member parameters take on the role of distinguishing the members. The empirical surprise is that this is enough to make the $M$ predictions *more diverse* than $M$ independently-trained networks — at $M=16$ on CIFAR-10, FiLM-Ensemble shows 9.2% pairwise disagreement vs the deep ensemble's 6.8%.

**Practical knobs:**

- $\rho$ (the Xavier gain) trades accuracy for calibration. Bigger $\rho$ → more diverse members → better calibration / better OOD detection / slightly lower accuracy.
- $M$ (the number of members) acts like in a deep ensemble — calibration generally improves with $M$, but the gains taper off around $M = 8\text{--}16$.

**Cost reality check.** At $M = 16$ on a ResNet-18, FiLM-Ensemble adds $\sim 1.3\%$ parameters vs $1500\%$ for the explicit deep ensemble. Inference time goes up roughly $M\times$ (you still need $M$ forward passes), but those passes can be parallelized as a $M\times$-larger batch on one GPU, so wall-clock cost is much smaller than running $M$ separate networks.

## Phase 2 — Graduate-Level Deep Dive

### FiLM Recap (Perez et al. 2018)

For a feature map $F_n \in \mathbb{R}^{B \times D_n \times \cdots}$ at layer $n$ (batch $B$, channels $D_n$, spatial dims trailing), and conditioning input $z$, FiLM is

$$
\gamma_n = g_n(z) \in \mathbb{R}^{D_n}, \qquad \beta_n = h_n(z) \in \mathbb{R}^{D_n}, \tag{1}
$$

$$
\text{FiLM}(F_n \mid \gamma_n, \beta_n) = \gamma_n \circ F_n + \beta_n , \tag{2}
$$

where $\circ$ is the Hadamard product *over the channel dimension* and broadcasts over the spatial dimensions. The output is feature-map-shaped; the modulation is per-channel-affine.

### FiLM-Ensemble Reformulation

Replace the continuous conditioning $z$ with a discrete member index $m \in \{1, \ldots, M\}$. Then $g_n, h_n$ degenerate to look-up tables:

$$
\gamma^m_n, \beta^m_n \in \mathbb{R}^{D_n} \text{ stored directly as parameters}.
$$

The per-member forward pass at layer $n$ becomes

$$
\boxed{\;\text{FiLM}\!\bigl(F_n \mid \gamma^m_n, \beta^m_n\bigr) = \gamma^m_n \circ F_n + \beta^m_n\;} \tag{3}
$$

Practically, the FiLM is grafted onto an existing BatchNorm layer, replacing the BN's standard affine pair with the per-member $(\gamma^m, \beta^m)$:

$$
\text{ConditionalBN}(F_n; m) = \gamma^m_n \circ \frac{F_n - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} + \beta^m_n , \tag{4}
$$

with $\mu_B, \sigma_B^2$ the standard batch statistics (running estimates at inference). The BN normalization is shared across members; only the post-normalization affine is per-member.

### Parameter Counting

For a backbone with $L$ BatchNorm layers each of dimension $D_n$, the total per-member overhead is $\sum_{n=1}^L 2 D_n$ (factor 2 for $\gamma$ and $\beta$), so an $M$-member ensemble adds

$$
\Delta P_{\text{FiLM-Ens}} = (M - 1) \cdot 2 \sum_{n=1}^L D_n = (M-1) \cdot 2 \cdot C_{\text{BN}} ,
$$

where $C_{\text{BN}}$ is the total BatchNorm-channel count. For ResNet-18, $C_{\text{BN}} \approx 4{,}864$, so at $M = 16$, $\Delta P \approx 30 \cdot 9{,}728 \approx 290\text{K}$ — well under 2% of ResNet-18's $\sim 11\text{M}$ parameters. By contrast, the explicit deep ensemble has $\Delta P = (M-1) \cdot P_{\text{base}}$, i.e., $15 \cdot 11\text{M} \approx 165\text{M}$ extra parameters.

### Diversity Initialization

The crucial design choice is sampling $\gamma^m_n, \beta^m_n$ from a Xavier-uniform with tunable gain:

$$
\gamma^m_n, \beta^m_n \sim \mathcal{U}\!\left(-\sqrt{\frac{3}{D_n}} \rho,\ \sqrt{\frac{3}{D_n}} \rho\right) . \tag{5}
$$

The Xavier-uniform width $\sqrt{3 / D_n}$ gives unit variance for a uniform distribution; the factor $\rho$ is a multiplicative gain that controls initial spread. As $\rho \to 0$, all members initialize to (near-)zero $\gamma$ (which is *not* the usual BN init of $\gamma = 1$, so this is a real choice). The paper sets $\rho = 2$ as default; for the 6mA DNA dataset, $\rho \in \{4, 8, 16, 32\}$ controls the accuracy-calibration trade-off explicitly.

**Mechanism.** At training start, the $M$ members process features differently because each has different $\gamma^m, \beta^m$ — even with shared conv weights, the resulting per-member loss gradients on the *shared* weights are different. Over training, the shared weights are pushed toward a compromise that serves all $M$ members; the per-member $\gamma^m, \beta^m$ then specialize each member to a different niche in the loss landscape. Higher initial spread → more divergent specialization → more diverse final members. This is the *implicit* analog of explicit-ensemble diversity, which comes from random weight initialization throughout the network.

### Diversity Metrics

For two members $f_i, f_j$ on a test set of $s$ samples:

**Disagreement score:**

$$
D = \frac{2}{M(M-1)} \sum_{i=1}^M \sum_{j=i+1}^M \frac{1}{s} \sum_{k=1}^s \mathbb{1}\{f_i(x_k) \neq f_j(x_k)\} . \tag{6}
$$

**Pairwise KL:**

$$
\text{KL} = \frac{2}{M(M-1)} \sum_{i, j > i} \frac{1}{s} \sum_k \sum_c p(f_i(x_k))_c \log \frac{p(f_i(x_k))_c}{p(f_j(x_k))_c} . \tag{7}
$$

Both metrics are higher for more independent ensembles. Figure 1 of the paper shows FiLM-Ensemble outperforming the deep ensemble on both at $M \geq 8$ on CIFAR-10/VGG-11.

### Predictive Mean and Variance

For classification with softmax outputs $\sigma(\cdot)$:

$$
\hat{y}(x) = \frac{1}{M} \sum_{m=1}^M \sigma\!\bigl(f_{\theta, \gamma^m, \beta^m}(x)\bigr) , \tag{8}
$$

and predictive uncertainty (model uncertainty) by the entropy decomposition (Gawlikowski survey eq. 9):

$$
H[\hat{y}(x)] = \underbrace{\frac{1}{M} \sum_m H[\sigma(f_{\theta, \gamma^m, \beta^m}(x))]}_{\text{average aleatoric}} + \underbrace{\text{MI}}_{\text{epistemic via member disagreement}} .
$$

For regression, $\bar{\mu} = \frac{1}{M} \sum_m \hat{\mu}_m$ and $\bar{\sigma}^2 = \frac{1}{M} \sum_m \hat{\sigma}_m^2 + \frac{1}{M} \sum_m \hat{\mu}_m^2 - \bar{\mu}^2$ as in Deep Ensembles (Lakshminarayanan 2017).

### Calibration Result: Expected Calibration Error

Recall $\text{ECE} = \sum_b \frac{|B_b|}{N} | \text{acc}(B_b) - \text{conf}(B_b) |$. Table 2 (CIFAR-100, $M=4$):

| Method | Acc (ResNet-18) | ECE (ResNet-18) |
|---|---|---|
| Single | 78.0 | 0.046 |
| Deep Ensemble | **81.6** | 0.041 |
| MC-Dropout | 75.5 | 0.064 |
| MIMO | 48.0 | 0.083 |
| Masksemble | 72.5 | 0.075 |
| BatchEnsemble | 77.7 | 0.052 |
| **FiLM-Ensemble** | 79.4 | **0.038** |

FiLM-Ensemble is the best-calibrated method and the best implicit method on accuracy; only the explicit (15× parameter) deep ensemble beats it on accuracy.

### Out-of-Distribution Detection

Train on Diaz-Pinto Glaucoma, test on REFUGE2020 as OOD. Use ensemble-disagreement-based uncertainty score; AUROC measures how well the score separates ID from OOD. Table 4: FiLM-Ensemble at $M=16$ achieves **79.85%** AUROC, beating Deep Ensemble (78.06%), Masksemble (70.95%), BatchEnsemble (75.04%), and MC-Dropout (72.22%). The higher diversity (Section 3.1) directly translates to better OOD calibration here.

## Connections to the Project

- **Bridges FiLM (Perez 2018, Batch 1) and ensembles (Lakshminarayanan 2017).** This paper is the explicit answer to "can FiLM be more than a conditioning primitive?" — yes, the per-channel γ/β scale-and-shift is rich enough to instantiate $M$ functionally distinct sub-networks from one backbone. For the project, this means: if the project already uses FiLM for context-conditioning (a task ID, a neuromodulator hyperparameter), the same mechanism *for free* gives an efficient way to do model-uncertainty estimation by extending γ/β to a member axis.
- **Aleatoric / epistemic decomposition reuse.** The predictive mean and variance formulas (Phase 2 eq. 8 and the law-of-total-variance regression case) are structurally identical to Kendall-Gal 2017's MC-dropout decomposition. The project's anchor heteroscedastic loss (Kendall-Gal eq. 6) extends naturally: each member produces $(\hat{\mu}_m, \hat{s}_m)$, the per-member heteroscedastic loss is averaged in training, and inference combines the M members' predicted variances (aleatoric) with the spread across members (epistemic).
- **Cheap alternative to MC dropout in the RL inner loop.** The Gawlikowski survey (Section 5.3 of Kendall-Gal review) flags MC dropout as expensive (50× slowdown for 50 samples on architectures with dropout everywhere). FiLM-Ensemble's per-member γ/β replication can be done as one parallelized batch on a single GPU — much closer to one-shot inference cost than serial MC sampling. For real-time RL rollouts, this is the more practical uncertainty mechanism.
- **Diversity gain $\rho$ as a hyperparameter knob.** The $\rho$ tradeoff (Section 3.5) is a clean handle the project could use to interpolate between "one-effective-model" and "high-disagreement ensemble". This maps suggestively onto the project's neuromodulator-as-hyperparameter direction memo: $\rho$ controls how strongly the members specialize, analogous to how a neuromodulator might widen or narrow the population of effective policy heads.
- **Limitations relevant to the project's stack.** Section 5 flags: (i) FiLM-Ensemble's reliance on BatchNorm makes it awkward for Transformer / LayerNorm architectures (`vaswani_2017_attention.md`); (ii) pre-trained models are hard to retrofit because similar init reduces diversity. If the project moves to a Transformer policy or wants to retrofit a pre-trained encoder, neither is immediately solved by this paper — flag to `senior-developer` as design-time constraints.
- **Recommended citation pattern.** When the project documents its uncertainty pipeline:
  - Kendall-Gal 2017 → cite for the heteroscedastic loss formula (per-input aleatoric).
  - Lakshminarayanan 2017 → cite for the deep-ensemble decomposition (aleatoric + epistemic).
  - **Turkoglu 2022 (this paper)** → cite for the *efficient* implementation of that decomposition via FiLM-style member-indexed γ/β.
  - Gawlikowski 2023 → cite for the taxonomic placement of the choice.
