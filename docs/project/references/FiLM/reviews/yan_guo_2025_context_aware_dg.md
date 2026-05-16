---
title: "Context-Aware Self-Adaptation for Domain Generalization"
authors: ["Hao Yan", "Yuhong Guo"]
year: 2025
venue: "2nd AdvML Frontiers workshop at ICML (preprint Apr 2025, arXiv:2405.03064)"
slug: yan_guo_2025_context_aware_dg
source_pdf: docs/project/references/FiLM/sources/Yan and Guo 2025 - Context-Aware Self-adaptation for domain generalization.pdf
topic: FiLM
---

# Context-Aware Self-Adaptation for Domain Generalization

## Plain-English entry point

**Domain generalization (DG)** is the problem of training a model on data from several *source domains* and having it work well on a brand-new *target domain* you've never seen — without any test-time fine-tuning. For instance, train an image classifier on photos, cartoons, and sketches, and have it work on art paintings. The standard fix is to either expand training data (data augmentation), force the representation to be domain-invariant, or use meta-learning to *simulate* the generalization problem during training. The DomainBed benchmark (Gulrajani & Lopez-Paz 2021) famously showed that plain Empirical Risk Minimization (ERM) — just train on all source domains pooled — beats most DG methods, raising the question of what genuinely helps.

This paper proposes **CASA** (Context-Aware Self-Adaptation), a two-stage method that adds a small **FiLM-like adaptation module** between a pretrained feature extractor and a frozen classifier. In stage 1, a separate classifier is trained per meta-source-domain. In stage 2, a *shared* adaptation module $g$ is inserted between feature extractor and classifier; $g$ takes both the current feature vector $z$ and the **mini-batch feature mean** $\mu$ as inputs, and outputs channel-wise scales $\gamma$ and shifts $\beta$ that linearly modulate the features:

$$\text{CaFiLM}(z_c) = \gamma_c\,z_c + \beta_c,$$

where $\gamma_c, \beta_c$ are learned from $z_c$ and $\mu_c$ via a 6-parameter shared linear layer. The key invariant is that the adapted features stay in the *same vector space* as the original (so the frozen classifier still works), but are *modulated* by the batch's domain-context signal. CASA reaches state-of-the-art on DomainBed (PACS, VLCS, OfficeHome, TerraIncognita, DomainNet), beating prior best EoA / DNA / SWAD with an average accuracy of 68.8 % across all five datasets.

This is the corpus's cleanest example of **FiLM as a domain-adaptation conditioner** — the conditioning signal is "this batch's domain context, summarised by the feature-mean $\mu$", not a task ID or a question. The mechanism is essentially identical to canonical FiLM but with a clever twist: the modulation parameters $\gamma, \beta$ are learned *per-channel* from $(z_c, \mu_c)$ pairs using a *shared* 2×2 weight matrix, giving the module only 6 trainable parameters in total.

## Section-ordered backbone

**1. Introduction.** Standard models fail under distribution shift. Domain generalization seeks methods that train on multiple source domains and generalise without test-time fine-tuning. The paper proposes CASA, a two-stage method: stage-1 trains a meta-source model per meta-source domain; stage-2 adds a context-aware self-adaptation module $g$ that adapts the pretrained features to the meta-target domain using mini-batch feature mean as "domain context". The adaptation uses a **CaFiLM** mechanism (Context-aware FiLM) — channel-wise linear modulation with $\gamma, \beta$ learned from the feature dimension and the batch feature mean. Inference uses an ensemble of multiple adapted meta-source models.

**2. Related Work.** Reviews three DG categories: data augmentation (Mixup, adversarial augmentation, generation), domain-invariant representation learning (adversarial alignment, MMD, KL, contrastive losses), and learning strategies (meta-learning per Li et al. 2018a). The DomainBed paper showed ERM beats most DG methods, prompting newer methods like SWAD (flat-minima ERM), mDSDI (domain-invariant + domain-specific features), DNA (PAC-Bayes bound), EoA (ensemble of moving averages), AGFA (Fourier-amplitude synthetic data). CASA differs from source-free domain-adaptation (which assumes access to unlabeled target data) — CASA learns *to adapt* to any unseen target.

**3. Proposed Method.**
- **§3.1 Two-stage generalizable learning.** Given $d$ source domains $\{D_j\}_{j=1}^d$, construct meta-source/meta-target pairs $(S_i, T_i)$ where $S_i = \cup_{j} \delta_i(j) D_j$ picks one or more domains as meta-source and the rest become meta-target. Stage 1: train a model $h_i \circ f_i$ per meta-source $S_i$ with cross-entropy ERM. Stage 2: insert a *shared* adaptation module $g$ between $f_i$ and $h_i$, getting the combined model $h_i \circ g \circ f_i$. The pretrained $f_i, h_i$ are frozen (mostly); only $g$ is trained. Two loss terms: an **adapt** loss that pushes $h_i \circ g_{C(T_i)} \circ f_i$ to fit the meta-target $T_i$, and a **preserve** loss that ensures $h_i \circ g_{C(S_i)} \circ f_i$ still fits the meta-source $S_i$. Total objective: $\min_{\theta_g} \mathcal{L}_{\text{adapt}} + \lambda \mathcal{L}_{\text{preserve}}$.

- **§3.2.1 Context choice.** The context $C(x)$ is the **mini-batch feature mean** $\mu = \mathbb{E}_{x \in X_b} f_i(x)$. It is cheap, doesn't require global statistics, and is computed from the same mini-batch the model is processing — so inference doesn't need an extra pre-pass over the target domain. Notable conceptual aside: batch normalisation *removes* first-order statistics from features to aid generalization; CASA *uses* them as domain information for the same goal.

- **§3.2.2 Module structure (the FiLM piece).** A naive MLP that maps $z$ to a new $z'$ could leave the feature space — which would break the frozen classifier $h_i$. To keep $g(z)$ in the *same space* as $z$, the authors draw on FiLM (Perez et al. 2018) and define **CaFiLM** (Context-aware FiLM): for each feature channel $c$, learn $\gamma_c, \beta_c$ from the pair $(z_c, \mu_c)$ via a shared 2×2 linear projection, and apply $\text{CaFiLM}(z_c) = \gamma_c z_c + \beta_c$. Crucially, $A \in \mathbb{R}^{2 \times 2}$ and $b \in \mathbb{R}^{2}$ are *shared across all channels* — only 6 parameters in the entire module. Implementation: stack $z$ and $\mu$ into a 2-row matrix, apply a 1-D convolution with weight $A$ and bias $b$ along the stacking dimension, producing two output rows $\gamma, \beta$.

- **§3.3 Inference with ensemble.** After training $|\mathcal{T}|$ adapted meta-source models $\{h_i \circ g \circ f_i\}$, inference averages their probability predictions: $\hat y = \arg\max_c \mathbb{E}_i[(h_i \circ g_{C(x)} \circ f_i(x))_c]$. The ensemble is contrasted with EoA (Arpit et al. 2022): EoA averages models trained on *the same* data with different seeds/hyperparameters; CASA averages models trained on *different meta-source* slices, then adapted by the same context-aware module — more like random forests.

**4. Experiments.** DomainBed evaluation on 5 datasets: PACS, VLCS, OfficeHome, TerraIncognita, DomainNet. ResNet-50 backbone, Adam optimiser, $\lambda$ tradeoff = 0.1 for PACS, 1 for others. Three seeds per task.

- **Table 1 results.** ERM 63.3 avg → CASA 68.8 avg. CASA SOTA on PACS (89.7), VLCS (81.5), OfficeHome (73.5); near-SOTA on TerraIncognita (52.0 vs. EoA 52.3); near-SOTA on DomainNet (47.2, matches DNA). Best overall average among all baselines.
- **Ablations.** (Table 2) Ensemble without adaptation: 66.8; with adaptation but no contextual training (just inserted at meta-source training): 67.0; full CASA: 68.8 — the +1.8 comes from training $g$ with the adapt+preserve objective. (Table 3) Removing the context input from $g$ (replacing CaFiLM with a context-blind MLP) drops PACS-related accuracy from 73.5 to 72.4 avg. (Table 4) Replacing CaFiLM with an MLP that maps $z$ to a new $z'$ in arbitrary space: drops from 73.5 to 72.4 avg. Both confirm that FiLM-style same-vector-space modulation matters.

**5. Conclusion.** CASA proposes a simple two-stage approach: train meta-source models, then train a *shared* context-aware FiLM-style adaptation module to handle distribution shift to meta-target domains, while preserving meta-source performance. Ensemble at inference. SOTA on standard DomainBed benchmarks.

## Phase 1 — Undergraduate-level synthesis

**The problem.** Train a model on data from a few source domains (e.g. photos, cartoons, sketches) and have it work on a brand-new target domain (e.g. paintings) without fine-tuning at test time.

**The trick.** Train in two stages. (Stage 1) Train several models, each on a different subset of source domains — call these "meta-source" models. (Stage 2) Add a small **adapter** $g$ between each meta-source model's feature extractor and classifier, and train $g$ on *another* slice of source domains — the "meta-target" slice. Train $g$ with two losses: it should help the model fit the meta-target (adapt loss), AND it should not hurt the model's accuracy on its original meta-source (preserve loss). The result is an adapter that knows how to retune features for new domains without destroying old performance.

**The FiLM bit.** What does $g$ do? It modulates each feature channel: $z'_c = \gamma_c z_c + \beta_c$. This is exactly **FiLM** (Perez et al. 2018). The clever twist is that **$\gamma_c$ and $\beta_c$ are computed from the value $z_c$ itself AND from the average of $z_c$ over the current mini-batch** (called the "context", written $\mu_c$). The intuition: the batch mean $\mu$ summarises which domain we're currently in (paintings have different statistics than photos), so it tells the adapter how to retune the features. To keep the parameter count tiny, the projection $(z_c, \mu_c) \to (\gamma_c, \beta_c)$ is a single 2×2 weight matrix and 2-vector bias **shared across all channels** — so the whole adapter has only 6 trainable parameters.

**Why preserve the vector space?** The classifier $h_i$ on top of the adapter is frozen. If $g$ remapped features to arbitrary new vectors, $h_i$ would no longer make sense. FiLM-style linear modulation guarantees the output stays in the same space as the input — it only rescales and shifts each channel — which is why FiLM, not an MLP, is the right tool here. Table 4 confirms this: replacing CaFiLM with an MLP drops accuracy by 1.1 percentage points on OfficeHome.

**Result.** CASA reaches 68.8 % average across 5 DomainBed datasets, beating prior SOTA EoA (68.0), DNA (67.6), SWAD (66.9), MIRO (65.9). Particularly strong on the smaller datasets (PACS, VLCS, OfficeHome). Competitive on large ones (DomainNet).

## Phase 2 — Graduate-level deep dive

### 2.1 Meta-generalization setup

Given $d$ source domains $\mathcal{D} = \{D_j\}_{j=1}^d$, choose a binary selector $\delta_i \in \{0, 1\}^d$ to construct a meta-source / meta-target pair:

$$
S_i = \bigcup_{j=1}^d \delta_i(j)\,D_j, \qquad T_i = \mathcal{D} \setminus S_i.
$$

Repeating with different $\delta_i$ generates a collection of meta-tasks $\mathcal{T} = \{(S_i, T_i)\}$. The paper uses 7 such combinations for the 4-domain datasets (PACS / VLCS / OfficeHome / TerraIncognita) and 11 of the 31 possible combinations for the 6-domain DomainNet.

### 2.2 Stage-1 meta-source training

For each meta-source domain $S_i$, train a feature extractor $f_i(\cdot; \theta_{f_i})$ and classifier $h_i(\cdot; \theta_{h_i})$ to minimise ERM cross-entropy:

$$
(\theta_{f_i}^\star, \theta_{h_i}^\star) \;=\; \arg\min_{\theta_{f_i}, \theta_{h_i}}\; \mathbb{E}_{(x,y) \in S_i}\bigl[\ell_{\text{ce}}\bigl(h_i \circ f_i(x;\, \theta_{f_i}, \theta_{h_i}),\, y\bigr)\bigr].
$$

This yields $|\mathcal{T}|$ pretrained meta-source models.

### 2.3 Stage-2 adapt + preserve objective

Insert a *shared* adaptation module $g$ between $f_i$ and $h_i$ for every meta-task $i$. The meta-target adapt loss:

$$
\mathcal{L}_{\text{adapt}} \;=\; \mathbb{E}_{T_i \in \mathcal{T}}\, \mathbb{E}_{(x, y) \in T_i}\Bigl[\ell_{\text{ce}}\Bigl(h_i \circ g_{C(T_i)} \circ f_i(x;\, \theta_g, \theta_{f_i}^\star, \theta_{h_i}^\star),\, y\Bigr)\Bigr].
$$

The meta-source preserve loss:

$$
\mathcal{L}_{\text{preserve}} \;=\; \mathbb{E}_{S_i \in \mathcal{T}}\, \mathbb{E}_{(x, y) \in S_i}\Bigl[\ell_{\text{ce}}\Bigl(h_i \circ g_{C(S_i)} \circ f_i(x;\, \theta_g, \theta_{f_i}^\star, \theta_{h_i}^\star),\, y\Bigr)\Bigr].
$$

The full stage-2 objective is

$$
\min_{\theta_g}\;\; \mathcal{L}_{\text{adapt}} \;+\; \lambda\,\mathcal{L}_{\text{preserve}}, \qquad \lambda \in \{0.1\,(\text{PACS}),\, 1\,(\text{others})\}.
$$

The pretrained feature-extractor and classifier parameters $\theta_{f_i}^\star, \theta_{h_i}^\star$ are **frozen** for PACS/VLCS/OfficeHome; for the larger TerraIncognita and DomainNet they are *fine-tuned* alongside $g$ with a smaller learning rate ($5 \times 10^{-5}$ for the classifier, $10^{-3}$ for $\theta_g$).

### 2.4 The CaFiLM update rule (the load-bearing equation)

For a feature vector $z = f_i(x) \in \mathbb{R}^C$ with channels $z_c$ and a mini-batch feature mean $\mu = \mathbb{E}_{x \in X_b} f_i(x) \in \mathbb{R}^C$ with channels $\mu_c$, the **CaFiLM** module computes, for each channel $c$:

$$
\begin{bmatrix} \gamma_c \\ \beta_c \end{bmatrix} \;=\; A\,\begin{bmatrix} z_c \\ \mu_c \end{bmatrix} \;+\; b, \qquad A \in \mathbb{R}^{2 \times 2},\; b \in \mathbb{R}^{2},
$$

and then applies the FiLM modulation channel-wise:

$$
\widetilde z_c \;=\; \gamma_c \cdot z_c \;+\; \beta_c.
$$

The parameters $A$ and $b$ are **shared across all $C$ channels** — so the total parameter count of the adaptation module is $2 \times 2 + 2 = 6$. The full self-adaptation function is

$$
g_{C(x)}(z) \;=\; \mathrm{CaFiLM}(z;\,\mu) \;\in\; \mathbb{R}^C, \qquad \mu = \mathbb{E}_{x' \in X_b} f_i(x').
$$

Implementation: stack $z$ and $\mu$ into a $2 \times C$ matrix and apply a 1-D convolution (kernel size 1) with $A$ as the weight and $b$ as the bias, sliding *along* the stacking dimension — producing the two output rows $\gamma \in \mathbb{R}^C$ and $\beta \in \mathbb{R}^C$.

### 2.5 Why CaFiLM and not an MLP?

The module $g$ sits between a frozen feature extractor and a frozen classifier. If $g$ remapped features to an arbitrary new space, the frozen classifier $h_i$ would receive features that look nothing like what it was trained on, and accuracy would collapse on the meta-source domain — defeating the preserve loss. FiLM guarantees:

$$
\|g(z) - z\|_2 \;\le\; |\gamma_c - 1|\,\|z\|_2 + \|\beta\|_2,
$$

so when $A, b$ are initialised near the identity ($A \approx \mathrm{diag}(1, 0)$, $b \approx 0$), $\gamma_c \approx 1$ and $\beta_c \approx 0$, and $g(z) \approx z$ at the start of stage-2 training. The module can only smoothly *retune* features in the same space, not displace them — the right inductive bias for adapt + preserve.

Table 4 in the paper provides the empirical evidence: replacing CaFiLM with an MLP that maps $z \to z'$ drops OfficeHome average accuracy from 73.5 to 72.4. This is the cleanest "FiLM beats MLP for same-space modulation" demonstration in the corpus.

### 2.6 Ensemble inference

After stage-2, there are $|\mathcal{T}|$ adapted models $\{h_i \circ g \circ f_i\}_{i=1}^{|\mathcal{T}|}$. Inference averages predicted probabilities:

$$
\hat y \;=\; \arg\max_{c \in \{1, \dots, K\}}\; \mathbb{E}_{i \in \{1, \dots, |\mathcal{T}|\}}\Bigl[\bigl(h_i \circ g_{C(x)} \circ f_i(x)\bigr)_c\Bigr].
$$

The context $C(x)$ is computed *online* from the current test-time mini-batch, using only the same model that is making the prediction — no oracle access to the target domain.

### 2.7 Empirical signature

DomainBed (Table 1):

| Method | PACS | VLCS | OffHome | TerraIncog | DomNet | Avg |
|---|---|---|---|---|---|---|
| ERM | 85.5 | 77.5 | 66.5 | 46.1 | 40.9 | 63.3 |
| SWAD | 88.1 | 79.1 | 70.6 | 50.0 | 46.5 | 66.9 |
| EoA (prior best) | 88.6 | 79.1 | 72.5 | 52.3 | 47.4 | 68.0 |
| **CASA** | **89.7** | **81.5** | **73.5** | 52.0 | 47.2 | **68.8** |

Ablations: removing context (MLP without $\mu$) → −1.1 pp on OfficeHome; replacing CaFiLM with full MLP → −1.1 pp; removing the stage-2 adapt training entirely → −1.8 pp. The +5.5 pp over ERM splits roughly as +3.5 pp ensemble of meta-source models and +2.0 pp from the CaFiLM adaptation.

## Connections

- **`perez_2018_film.md` (B1, canonical FiLM)** — CaFiLM is *literally* Perez et al.'s FiLM with the conditioner being the mini-batch feature mean $\mu$ instead of an external question or task embedding. The paper cites Perez et al. directly in §3.2.2. **The novelty is the conditioner choice, not the modulation operator.**
- **`jang_2022_bcz.md` (this batch)** — BC-Z uses FiLM where the conditioner is a *pretrained sentence embedding* (the task); CaFiLM uses FiLM where the conditioner is the *mini-batch feature mean* (the domain). Both demonstrate FiLM's role as a low-bandwidth steering mechanism.
- **`nikulin_2023_anti_exploration_rnd.md` (this batch)** — Both papers find FiLM beats MLP / concat for the same task because of FiLM's *same-vector-space* property. In Nikulin: smoother gradient landscape for the actor. In Yan & Guo: keeps adapted features compatible with a frozen classifier. The corpus-level synthesis: **FiLM's structural invariant (modulation, not remapping) makes it the right tool whenever a downstream consumer of the features is fixed.**
- **Meta-learning cluster (Abdollahzadeh, Takeda — both in B3)** — CaFiLM's two-stage meta-training (stage 1 = meta-source ERM; stage 2 = train shared $g$ on adapt + preserve over meta-task pairs) is a *meta-learning* instantiation of FiLM. The forward link to those B3 papers is direct: they analyse FiLM as a multi-task conditioner; CASA shows FiLM as a *meta-learned* domain conditioner. **The curator should flag this as a tight connection.**
- **`turkoglu_2022_film_ensemble.md` (B3, FiLM-Ensemble)** — CASA also uses ensembling (multiple adapted meta-source models). FiLM-Ensemble uses multiple FiLM channels *within one network*; CASA uses one CaFiLM module *across multiple networks*. The two papers exemplify "FiLM + ensemble" at orthogonal granularities.
- **`gorishniy_2025_tabm.md` (this batch)** — Both papers are about parameter-efficient modulation: TabM uses 32 implicit ensemble members via BatchEnsemble adapters; CASA uses a 6-parameter shared FiLM module to adapt features across domains. Both show that **a tiny amount of multiplicative gating around a much larger backbone is enough to deliver SOTA on the corresponding benchmark.**
- **Project-side relevance** — For NMN-as-modulator: CASA is the corpus's cleanest "modulator with very few parameters that re-tunes a frozen backbone using a context signal" example. The 6-parameter shared $A, b$ across all channels is a particularly clean parametrisation if the project wants a *low-bandwidth interoceptive modulator that re-conditions a fixed policy network*. Hand-off candidate: `professor-rl-bayesian-dl` for conceptual mapping; `experiment-designer` if this becomes a candidate architecture for an experiment.
