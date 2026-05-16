---
title: "TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling"
authors: ["Yury Gorishniy", "Akim Kotelnikov", "Artem Babenko"]
year: 2025
venue: "ICLR 2025"
slug: gorishniy_2025_tabm
source_pdf: docs/project/references/FiLM/sources/Gorishniy et al. 2025 - TabM - Advancing tabular deep learning with parameter-efficient ensembling.pdf
topic: FiLM
---

# TabM: Advancing Tabular Deep Learning with Parameter-Efficient Ensembling

## Plain-English entry point

Most machine-learning problems in industry don't involve images or text — they involve **tabular data**: spreadsheets with mixed numerical and categorical columns. For decades, gradient-boosted decision trees (XGBoost, LightGBM, CatBoost) have been the practical winners; deep-learning architectures keep claiming parity but rarely deliver. This paper proposes **TabM**, a deep model that beats every tabular-DL competitor (attention-based and retrieval-based methods alike) on a 46-dataset benchmark, matches gradient-boosted trees, and runs faster than every DL alternative.

The key idea is **parameter-efficient ensembling**, specifically a variant of **BatchEnsemble** (Wen et al. 2020). A normal deep ensemble trains $k$ independent MLPs and averages their predictions — expensive in compute and memory. BatchEnsemble does this with **one** MLP whose linear layers are *modulated* by per-member learnable rank-1 vectors. Specifically, each linear layer's weight is reshaped from $W$ to $W \odot (s_i\,r_i^\top)$ for member $i$, where $r_i, s_i$ are short trainable vectors that vary across the $k$ implicit ensemble members but $W$ is *shared*. The implicit members are forwarded in parallel by stacking $k$ inputs in a batch dimension.

**Why this is in the FiLM corpus.** BatchEnsemble's mechanism — per-member channel-wise scales modulating a shared weight matrix — is **structurally identical** to FiLM. FiLM uses an *external conditioner* to produce $\gamma, \beta$ that modulate features; BatchEnsemble uses *per-member learnable* $r, s$ vectors to modulate weights. Same multiplicative gating operator, different conditioning source. The paper itself flags this connection: TabM is explicitly compared against FiLM-Ensemble (Turkoglu et al. 2022, also in this corpus), and the authors prefer BatchEnsemble because FiLM-Ensemble *requires* normalisation layers to insert its $\gamma, \beta$ — which tabular MLPs typically lack. This is a load-bearing piece of evidence that **FiLM-style multiplicative modulation is a strong inductive bias for ensembling**, and tabular data benefits from it the same way as image data.

The paper's other contribution is a fresh large-scale benchmark of tabular-DL methods (46 datasets, 4 dimensions: rank, score distribution, training time, inference throughput) that shows MLP-based models — including TabM — Pareto-dominate the attention- and retrieval-based architectures the field has been chasing.

## Section-ordered backbone

**1. Introduction.** Tabular DL has been steadily claimed to match gradient-boosted decision trees (GBDT), but the practical evidence is mixed: many claimed wins are inconsistent across datasets, and efficiency (training time, inference throughput) is often ignored. The authors revisit the question and identify **parameter-efficient ensembling** as a previously-overlooked path to strong tabular models. Their TabM = MLP + BatchEnsemble-variant + better initialisation. Main contributions: (1) a simple architecture (TabM) that competes with GBDT and beats prior tabular-DL methods; (2) a 4-axis benchmark across 46 datasets; (3) an analysis showing TabM's implicit submodels are weak individually but strong collectively.

**2. Related Work.** Positions TabM against (a) GBDT (the practical baseline), (b) attention-based tabular DL (FT-Transformer, SAINT, ExcelFormer, T2G-Former), (c) retrieval-based tabular DL (TabR, ModernNCA), (d) MLP-based tabular DL (ResNet, MLP-PLR, etc.), and (e) **parameter-efficient deep ensembles** — Lee et al. 2015 TreeNet, MIMO (Havasi 2021), BatchEnsemble (Wen 2020), Packed-Ensemble (Laurent 2023), **FiLM-Ensemble (Turkoglu 2022)**. The novelty is bringing efficient ensembling to tabular data and showing it works.

**3. TabM.**
- **§3.1 Preliminaries.** 46 datasets, mixed regression and classification, mixed random and "domain-aware" (time-aware) splits. Standard hyperparameter-tuning protocol. The reporting metric is *relative improvement over MLP* in percent.
- **§3.2 BatchEnsemble primer.** For a linear layer $l(x) = Wx + b$, BatchEnsemble defines the $i$-th member's layer as $l_i(x_i) = s_i \odot (W (r_i \odot x_i)) + b_i$, equivalently $W_i = W \odot (s_i\,r_i^\top)$. The vectors $r_i, s_i, b_i$ are per-member ("adapters"); $W$ is shared. All $k$ members are forwarded in parallel by stacking inputs $X \in \mathbb{R}^{k \times d}$, giving $l_{\text{BE}}(X) = ((X \odot R) W) \odot S + B$ where $R, S, B \in \mathbb{R}^{k \times d}$.
- **§3.3 Architecture construction.** TabM is built incrementally:
  - **MLP** — baseline.
  - **MLP×k** — traditional deep ensemble: train $k$ independent MLPs, average predictions. Already beats FT-Transformer.
  - **TabM$_{\text{packed}}$** — MLP + Packed-Ensemble, i.e. $k$ MLPs are *one* model that processes $k$ inputs in parallel but with no weight sharing. Better than MLP×k because hyperparameters and early-stopping are tuned for the *ensemble*, not per-member.
  - **TabM$_{\text{naive}}$** — MLP + naive BatchEnsemble (full $3N$ adapters $R, S, B$ per linear layer, all random-init $\pm 1$). Slightly better than TabM$_{\text{packed}}$ — weight sharing acts as regularisation.
  - **TabM$_{\text{bad}}$** — same as $_{\text{naive}}$ but with the *very first* $R$ adapter removed. Big performance drop. This isolates the first $R$ adapter as critical.
  - **TabM$_{\text{mini}}$** — keep *only* the first $R$ adapter, drop all $3N - 1$ others. Performs slightly better than $_{\text{naive}}$.
  - **TabM (final)** — full $3N$ adapters, but all multiplicative adapters except the very first $R$ are *initialised deterministically to 1* (so at init the model behaves like TabM$_{\text{mini}}$). This is the strongest variant.
- **§3.4 Practical modifications.** Shared training batches (using the same $x$ for all $k$ members) is a small-loss-large-efficiency-gain trick. Non-linear feature embeddings (piecewise-linear) help further.
- **§3.5 Summary.** Two key drivers: *simultaneous training* of implicit ensemble members (enables ensemble-aware early stopping and hyperparameter tuning) and *weight sharing* (regularises).

**4. Evaluation.** 46-dataset benchmark, 4 axes (rank, score distribution, training time, inference throughput). TabM and TabM$^\dagger_{\text{mini}}$ (with feature embeddings) hold the top two ranks among DL models. Attention-based methods (FT-Transformer, SAINT, T2G, ExcelFormer) are slower and less reliable. Retrieval-based (TabR, ModernNCA) struggle on large datasets. On the 9 domain-aware (real-world distribution-shifted) splits, TabM's lead over attention/retrieval methods widens.

**5. Analysis.** §5.1 shows the $k = 32$ implicit submodels of TabM individually overfit (worse than a single MLP) but their *mean prediction* outperforms a single MLP on both train and test loss — the canonical deep-ensemble signature, replicated with 1 MLP's worth of shared weights. §5.2 ablates: removing simultaneous training collapses performance; removing weight sharing recovers some performance. §5.3 sweeps $k \in \{1, 2, 4, 8, 16, 32, 64\}$ and finds $k = 32$ is a robust default.

**6. Discussion.** TabM is a simple, strong, fast DL baseline for tabular data. Limitations: requires more memory than a plain MLP (the $k$ parallel streams); the choice of MLP backbone is unoptimised; FiLM-Ensemble-style normalisation-layer modulation was not explored because tabular MLPs typically lack normalisation layers.

**Appendix A.1 (relevant to FiLM corpus).** The authors explicitly compare BatchEnsemble against FiLM-Ensemble and MIMO. They prefer BatchEnsemble because: (a) FiLM-Ensemble requires normalisation layers (where its $\gamma, \beta$ are inserted), which tabular MLPs don't have; (b) MIMO requires concatenating $k$ inputs (multiplying input width by $k$), which is prohibitive on high-feature-count datasets. So the corpus connection is: **FiLM-Ensemble and BatchEnsemble are both feature-wise-multiplicative ensembling methods; they differ only in where the modulation is inserted (norm-layer vs. linear-layer).**

## Phase 1 — Undergraduate-level synthesis

**The problem.** Tabular data is everywhere in industry. Gradient-boosted decision trees (GBDT) are the practical winner; deep-learning competitors are usually slower and unreliable. Existing tabular DL has chased attention and retrieval mechanisms with mixed success.

**The idea.** *Ensemble* a bunch of MLPs — but cheaply. A normal deep ensemble trains $k = 32$ MLPs separately and averages their predictions. That's expensive. **BatchEnsemble** does the same thing with **one** MLP. The trick: each linear layer's weight matrix is multiplied element-wise by a rank-1 matrix $s_i\,r_i^\top$ that varies across members, while the underlying $W$ is shared. So each "implicit member" sees a slightly different effective weight matrix, but the bulk of the parameters is shared. You can run all $k$ members in parallel by stacking $k$ copies of the input.

**Why this is a FiLM paper.** BatchEnsemble's modulation $s_i \odot (W (r_i \odot x_i))$ is *the same multiplicative gating* as FiLM's $\gamma \odot h + \beta$. The difference is just:
- **FiLM**: $\gamma, \beta$ come from an *external conditioner* (a question, a task ID, a state vector).
- **BatchEnsemble**: $r, s$ are *learnable per-member parameters*.

In both cases the multiplicative gating is what enables one network to behave like many. The paper compares its method directly against **FiLM-Ensemble** (Turkoglu et al. 2022 — also in this corpus) and notes that FiLM-Ensemble's only difference is *where* the gating is inserted: FiLM-Ensemble injects $\gamma, \beta$ at normalisation layers, BatchEnsemble injects $r, s$ at linear-layer weights.

**Result.** TabM beats every prior tabular-DL method on 46 benchmark datasets, runs as fast as a plain MLP, and matches gradient-boosted trees. Cheap ensembling via multiplicative modulation is a "low-hanging fruit" for tabular DL.

**Flag for the FiLM corpus.** TabM is FiLM-adjacent rather than canonically FiLM: it uses the same multiplicative-gating inductive bias, but the "conditioner" is just a member index, not an external context vector. The conceptual link is what makes it relevant — it's empirical evidence that the same operator the FiLM literature has built around is also the best-performing tool for the orthogonal goal of *parameter-efficient ensembling*.

## Phase 2 — Graduate-level deep dive

### 2.1 BatchEnsemble formalism (the load-bearing equation)

Consider a linear layer of a base MLP with weights $W \in \mathbb{R}^{d \times d}$ (square for notational ease) and bias $b \in \mathbb{R}^d$. A traditional deep ensemble of $k$ members has $k$ independent copies:

$$
l_i(x_i) \;=\; W_i\,x_i \;+\; b_i, \qquad i = 1, \dots, k,
$$

with parameter count $k(d^2 + d)$. BatchEnsemble factorises each $W_i$ as

$$
W_i \;=\; W \;\odot\; \bigl(s_i\, r_i^\top\bigr), \qquad r_i, s_i \in \mathbb{R}^d,
$$

so that

$$
l_i(x_i) \;=\; s_i \;\odot\; \bigl(W\,(r_i \odot x_i)\bigr) \;+\; b_i.
$$

Here $W \in \mathbb{R}^{d \times d}$ is **shared across all members**; $r_i, s_i, b_i \in \mathbb{R}^d$ are **per-member** vectors ("adapters"). Parameter count is now $d^2 + 3kd$, of which only $3kd$ scales with ensemble size — a factor of $\frac{d^2 + 3kd}{k(d^2 + d)} \approx 1/k$ overhead in the large-$d$ regime.

Members are computed *in parallel* by stacking $k$ inputs in a batch dimension. Let $X \in \mathbb{R}^{k \times d}$ store the $k$ member inputs (one row per member), and let $R, S, B \in \mathbb{R}^{k \times d}$ stack the per-member adapter vectors row-wise. Then

$$
l_{\text{BE}}(X) \;=\; \bigl((X \odot R)\, W\bigr) \;\odot\; S \;+\; B \;\in\; \mathbb{R}^{k \times d},
$$

where row $i$ of $l_{\text{BE}}(X)$ equals $l_i(x_i)$. On modern hardware, this single batched matmul costs much less than $k$ separate matmuls because the original workload typically underutilises GPU parallelism.

### 2.2 The FiLM connection (corpus-relevant)

The canonical FiLM equation (Perez et al. 2018, the corpus root) is

$$
\widetilde h \;=\; \boldsymbol{\gamma}(c) \odot h + \boldsymbol{\beta}(c),
$$

where $h$ is a hidden activation, $c$ is a conditioning vector (a task ID, a question embedding, etc.), and $\gamma(c), \beta(c)$ are externally produced channel-wise affine parameters.

BatchEnsemble's gating, written in the same style, is

$$
\widetilde h^{(i)} \;=\; s_i \odot \bigl(W (r_i \odot x)\bigr) + b_i,
$$

i.e. a **two-sided multiplicative modulation of the linear-layer computation by member-indexed scale vectors** $r_i$ (input gate) and $s_i$ (output gate), with member-indexed bias $b_i$. Up to the location of the modulation (input vs. output of the linear layer), this is structurally identical to FiLM applied with the conditioner being the member index $i$. The crucial differences are:

| | FiLM (Perez et al. 2018) | BatchEnsemble |
|---|---|---|
| **Modulator source** | External conditioner $c$ via a learnable function $\gamma(c), \beta(c)$ | Per-member learnable vectors $r_i, s_i$ (no conditioner) |
| **Number of modulations** | One per data point (varies with $c$) | $k$ per data point ($k$ implicit members) |
| **Location** | Channel-wise on hidden activations, often post-norm | Pre- and post-linear-layer multiplication on weights |
| **Goal** | Task/context conditioning | Implicit ensemble |

The corpus message: **FiLM and BatchEnsemble are two instantiations of the same multiplicative-modulation inductive bias.** TabM's success is a one-more datapoint that this inductive bias is broadly useful — here for ensembling tabular MLPs.

### 2.3 Why TabM beats naïve BatchEnsemble — the initialisation trick

The TabM ablation reveals an interesting empirical structure:

1. **TabM$_{\text{packed}}$** ($k$ independent MLPs, no weight sharing, parallel forward) beats MLP×$k$ (sequential independent ensemble) by ~0.5 % relative-to-MLP — gains come from *ensemble-aware early stopping*.
2. **TabM$_{\text{naive}}$** (BatchEnsemble with full $3N$ adapters, all random init) beats TabM$_{\text{packed}}$ by ~0.5 % more — gains come from *weight sharing as regularisation*.
3. **TabM$_{\text{bad}}$** = TabM$_{\text{naive}}$ minus the first $R$ adapter drops dramatically. **TabM$_{\text{mini}}$** = only the first $R$ adapter recovers and slightly improves on TabM$_{\text{naive}}$ — so the *very first* adapter is doing essentially all the work, by mapping the same input into $k$ distinct representation spaces *before* the first weight matrix mixes the features.
4. **TabM (final)** keeps all $3N$ adapters but initialises every multiplicative adapter *except* the first to $\mathbf{1}$ deterministically — so at $t = 0$ the model behaves like TabM$_{\text{mini}}$, and during training the remaining adapters can add expressivity. This is the best variant.

Formally, if $R^{(\ell)}_i, S^{(\ell)}_i$ denote the layer-$\ell$ adapters for member $i$, the TabM initialisation rule is

$$
R^{(1)}_i \;\sim\; \pm 1 \text{ (Rademacher)}, \qquad R^{(\ell)}_i, S^{(\ell)}_i \;=\; \mathbf 1 \text{ for } \ell \ge 2 \text{ or } \ell = 1 \text{ on } S \text{ side}, \qquad b^{(\ell)}_i \;=\; 0.
$$

The mechanism is "warm-start the ensemble at TabM$_{\text{mini}}$, then let the model unlock the remaining expressivity if it helps".

### 2.4 Why FiLM-Ensemble was not chosen

In Appendix A.1 the authors explain that FiLM-Ensemble (Turkoglu et al. 2022) was a candidate but rejected for tabular MLPs because:

- **FiLM-Ensemble injects its per-member $\gamma_i, \beta_i$ at *normalisation layers*** (batch-norm / layer-norm). Tabular MLPs typically do not have normalisation layers between blocks, so FiLM-Ensemble would need additional norm layers added — an architectural intrusion.
- BatchEnsemble's $r, s$ instead live around the *linear layer's weight matrix*, which every MLP has by definition. So BatchEnsemble is "drop-in" for any MLP-like backbone.

This is a useful note for the corpus: **FiLM-Ensemble and BatchEnsemble are interchangeable in spirit but differ in their structural assumptions** about the host architecture. TabM is an empirical reason to prefer linear-layer-weight modulation when the host architecture is plain MLP-style.

### 2.5 Empirical signature

Figure 2 / Figure 3 of the paper (46-dataset benchmark, relative improvement over MLP):

- MLP: 0.0 % baseline
- FT-Transformer: 0.4 %
- MLP×32: 1.0 %
- TabM$_{\text{packed}}$: 1.4 %
- TabM$_{\text{naive}}$: 1.9 %
- TabM$_{\text{mini}}$: 2.0 %
- **TabM: 2.2 %**
- TabM$^\dagger_{\text{mini}}$ (with feature embeddings): 2.9 %
- **TabM$^\dagger_{\text{mini}}\times 5$ (ensemble of TabM): 3.2 %**

The improvement per parameter is essentially flat across the BatchEnsemble variants — the architectural simplicity is preserved. The numbers are small per-dataset but the *consistency* across 46 datasets makes TabM Pareto-dominant on rank and the lower percentile of score distribution.

## Connections

- **`perez_2018_film.md` (B1, canonical FiLM)** — TabM's load-bearing operator is structurally a FiLM applied at linear-layer weights with the conditioner being a member-index. The connection is conceptual, not via direct citation in the FiLM tradition. **Flag this explicitly to `literature-curator`: TabM is FiLM-adjacent, not canonically FiLM, but the multiplicative-modulation inductive bias is the same.**
- **`turkoglu_2022_film_ensemble.md` (B3, FiLM-Ensemble)** — **Strong analog.** Both papers use feature-wise multiplicative modulation to instantiate $k$ implicit ensemble members in one network. TabM uses BatchEnsemble (modulation around linear layers); FiLM-Ensemble uses FiLM (modulation at normalisation layers). The TabM paper directly compares against FiLM-Ensemble in Appendix A.1 and rejects it only because tabular MLPs lack normalisation layers — a pragmatic, not principled, choice. The two methods are interchangeable in spirit.
- **`jang_2022_bcz.md` (this batch)** — Different goal but same operator family. BC-Z uses FiLM to condition a policy on an *external* task embedding; TabM uses BatchEnsemble (FiLM-with-member-index) to ensemble. Useful contrast for the curator: FiLM-style modulation can either steer a model (BC-Z) or instantiate many models in one (TabM, FiLM-Ensemble).
- **Multimodal meta-learning cluster (Abdollahzadeh, Takeda — both in B3)** — These papers analyse FiLM as a multi-task / mixed-task conditioner. TabM occupies a different point in the design space: FiLM-with-member-index *instead of* FiLM-with-task-index. The architectural primitive is the same; the conditioner identity is different.
- **Project-side relevance** — For NMN-as-modulator: TabM is the cleanest evidence in the corpus that *learnable per-member multiplicative gates around a shared linear layer* are sufficient to instantiate a high-quality ensemble. If the project wants an interoceptive modulator that *re-purposes the same policy network in multiple parallel "modes"* (e.g., one mode per interoceptive context), the TabM template — shared $W$, per-mode $r, s, b$, parallel batched forward — is a strong starting point. Hand-off candidate: `professor-rl-bayesian-dl` for the conceptual link, `senior-developer` for any implementation issue plan.
