---
title: "Revisit Multimodal Meta-Learning through the Lens of Multi-Task Learning"
authors: ["Milad Abdollahzadeh", "Touba Malekzadeh", "Ngai-Man Cheung"]
year: 2021
venue: "NeurIPS 2021 (35th Conference on Neural Information Processing Systems)"
slug: abdollahzadeh_2021_multimodal_meta
source_pdf: "docs/project/references/FiLM/sources/Abdollahzadeh et al. 2021 - Revisit multimodal meta-learning through the lens of multi-task learning.pdf"
topic: FiLM
---

## Plain-English Entry Point

**Meta-learning** is "learning to learn" — train a model on many small tasks so that, given a new task with only a handful of labeled examples (the "support set"), it can quickly adapt and classify the rest (the "query set"). The most famous algorithm in this family is **MAML** (Model-Agnostic Meta-Learning, Finn 2017): learn an initialization from which a few gradient steps yield a good task-specific model. **Few-shot classification** is the standard benchmark setup ($N$-way, $K$-shot: pick $N$ classes, give the model $K$ examples per class, ask it to classify a new query).

**Multimodal meta-learning** extends this: instead of all tasks coming from one domain (say, ImageNet sub-categories), the meta-training distribution mixes domains (Omniglot characters + ImageNet objects + Aircraft + Birds, etc.). The hope is the meta-learner can transfer skill *across* domains, the way humans learn skiing partly from skateboarding. The prior state of the art, **MMAML** (Vuorio 2019), handles this by adding a **modulation network** on top of MAML: a small auxiliary network reads the support set, produces a **task embedding** $\upsilon_T$, then generates per-task FiLM scale-and-shift parameters $\eta, \gamma$ to modulate the base network's feature maps.

This paper makes two contributions. **First**, it borrows a metric called **transference** from multi-task learning (Fifty 2020) and adapts it to meta-learning: for each pair of source task $i$ and target task $j$, compute the loss-ratio $\text{LR}_{i \to j}$ after one SGD step on task $i$. If $\text{LR} < 1$, task $i$ helped task $j$ (positive transfer); if $\text{LR} > 1$, task $i$ hurt (negative transfer). Applied during meta-training, this exposes a clear pattern: early in training, cross-mode tasks mostly hurt each other; later, transfer becomes positive. **Second**, the paper observes that FiLM modulation in MMAML is **too restrictive** — multiplying every weight in a layer's kernel by a single scalar leaves "different tasks fighting for the model capacity". They propose **Kernel Modulation (KML)**: instead of per-channel γ/β, generate a per-task **modulation matrix** $M^l(\upsilon_T, \phi)$ that multiplies the **full convolutional kernel element-wise**, plus a bias offset. This is FiLM scaled up from a channel-level affine to a per-weight residual perturbation. KML beats MMAML by substantial margins across 2/3/5-mode multimodal benchmarks and also helps unimodal few-shot. For this project, the takeaway is that **modulation richness matters**: the project's use of FiLM-style per-channel γ/β may be capacity-limited for highly diverse task distributions, and KML offers a heavier-modulation alternative within the same conceptual frame.

## Section-Ordered Backbone

**Abstract.** Two contributions: (1) a task-level transference metric to quantify cross-mode knowledge transfer in multimodal meta-learning; (2) a new modulator (Kernel Modulation, KML) inspired by hard parameter sharing in multi-task learning, motivated by re-interpreting FiLM in MMAML.

**1. Introduction.** Reviews MMAML's claim that a single multimodal meta-learner can beat per-mode specialists. Identifies the gap: no direct verification of cross-mode knowledge transfer. Two contributions stated explicitly: micro-level quantification + new multimodal meta-learner.

**2. Related Work.** Three sub-areas:

- **Few-shot learning** — metric-based (ProtoNet, Matching Net, Relation Net), optimization-based (Meta-LSTM, MAML and variants, LEO, Meta-SGD), augmentation-based, weight-generation.
- **Multi-task learning** — hard vs soft parameter sharing (Ruder 2017, Caruana 1993). Transference (Fifty 2020) measures task interaction. Negative transfer occurs when tasks "fight for capacity".
- The two literatures barely intersect; this paper bridges them.

**3. Preliminaries.** $N$-way, $K$-shot setup. Support set $S_T = \{(x_i, y_i)\}_{i=1}^{N \times K}$, query set $Q_T$. Episodic training (Vinyals 2016) mimics few-shot at training time. Multimodal: task distribution $p(\mathcal{T})$ spans multiple input/label domains.

**4. Information Transfer Among Tasks.**

- **Transference in episodic training.** Episodes implicitly cooperate to build shared features. Some cooperative, some destructive.
- **Algorithm 1 (transference metric).** At time step $t$: sample batch of tasks $\xi$, pick target task $T_j$. For each $T_i$ in $\xi$, compute updated $\theta^{t+1}_i = \theta^t - \alpha \nabla_{\theta} L_{T_i}(Q_i; \theta^t, S_i)$. Then evaluate target loss with the new params: $\text{LR}_{i \to j} = L_{T_j}(Q_j; \theta^{t+1}_i, S_j) / L_{T_j}(Q_j; \theta^t, S_j)$.
- $\text{LR}_{i \to j} < 1$ → positive transfer; $> 1$ → negative. The paper's adaptation: source from meta-train, target from meta-test, to measure *generalization*-level transfer.
- **Experiment.** 300 mini-ImageNet meta-train tasks → 1 FC100 meta-test task with ProtoNet. Histogram is bimodal; mid-training shows both positive and negative transfer; mostly negative tasks are non-animal classes, mostly positive are animal classes. Over training, negative shifts to positive.

**5. Proposed Multimodal Meta-Learner.**

- **5.1 MMAML recap.** Task encoder $\upsilon_T = h_\phi(S_T)$ (4-layer CNN over the support set) produces a task embedding. Parameter generator $\omega_T = g_\phi(\upsilon_T)$ (MLPs) produces FiLM params $\omega_T = \{\eta_T, \gamma_T\}$. The output channel $i$ of a base-network layer becomes $\hat{Y}_i = \eta_i Y_i + \gamma_i$ with $\eta_i, \gamma_i$ scalars.
- **Re-interpretation of FiLM.** For a conv layer with kernel $W_i$ and bias $b_i$, $Y_i = W_i * X + b_i$, so the FiLM modulation can be rewritten as

  $$
  \hat{Y}_i = (\eta_i W_i) * X + (\eta_i b_i + \gamma_i) ,
  $$

  i.e., FiLM is equivalent to convolving with a **modulated kernel** $\hat{W}_i = \eta_i W_i$ (the *entire* kernel scaled by one scalar) and adding a modulated bias. So per-task kernels are heavily entangled — they are just scalar multiples of one shared base kernel. This is the proposed *limitation*: different tasks have only a 1-D degree of freedom per channel to specialize, and so they "fight for the model capacity".
- **5.2 Kernel Modulation (KML).** Generate a per-task modulation **matrix** $M^l(\upsilon_T, \phi)$ matching the layer-$l$ kernel shape. Modulated kernel:

  $$
  \hat{W}^l_T = W^l \odot \bigl(J + M^l(\upsilon_T, \phi)\bigr) \tag{6}
  $$

  with $J$ the all-ones matrix, so $M$ encodes a *residual* perturbation around the base kernel. Bias offset: $\hat{b}^l_T = b^l + \Delta b^l(\upsilon_T, \phi)$.
- **Parameter-efficient generator** (Figure 4). To avoid blowing up the parameter generator, the modulation matrix is constructed as an outer product: $M^l(\upsilon_T, \phi) = g^l_{\phi_1}(\upsilon_T) \otimes g^l_{\phi_2}(\upsilon_T)$. Bias offset $\Delta b^l = g^l_{\phi_3}(\upsilon_T)$. Versus a single MLP, this reduces $g_\phi$ parameters by ~150×.
- **Generalization.** KML is presented explicitly as a **generalization of FiLM**: FiLM is the special case $M = \eta \cdot \mathbf{1}^\top$ (rank-1, constant scalar over the kernel). KML allows higher-rank perturbations of $W$.

**6. Experiments.** Datasets: Omniglot, mini-ImageNet, FC100, CUB, Aircraft. Combined into 2-mode / 3-mode / 5-mode meta-datasets. Meta-learners: ProtoNet, MAML. Baselines: per-mode specialists (Multi-MAML, Multi-ProtoNet), no-mode-info shared learner, MMAML, MProtoNet (MMAML's frame with ProtoNet substituted).

- **Table 1.** KML adds 2-5 percentage points on top of MMAML / MProtoNet across all multimodal scenarios. Best gains in 5-shot 5Mode setting: MProtoNet+KML 64.91% vs MProtoNet 59.95%.
- **Table 2.** On 2-mode (mini-ImageNet + FC100), the KML variant shows positive transfer from mini-ImageNet to FC100 (mini-ImageNet helps FC100 accuracy).
- **Transference results** (Figure 5). Compared to vanilla ProtoNet, MProtoNet+KML produces more positive-transfer source tasks at every training stage.
- **Unimodal classification** (supplementary): KML also helps even when the task distribution is single-domain.
- **RL extension.** Briefly mentioned in supplementary.

**7. Discussion and Conclusion.** Transference quantification + KML modulator. KML's improvement comes at a parameter / compute cost vs FiLM; positions as a strict generalization. Reframes multimodal meta-learning as MTL on episodes.

## Phase 1 — Undergraduate-Level Synthesis

Two ideas, both reframings of established techniques.

**Idea 1 — measure cross-task interference directly.** Meta-learning treats tasks as samples from a distribution and trains over many of them. But when those tasks come from very different domains (characters vs aircraft vs birds), some help each other and some hurt. The paper imports a metric from multi-task learning: simulate one SGD step on task $i$, then re-evaluate target task $j$'s loss. Ratio $< 1$ = task $i$ helped $j$; ratio $> 1$ = it hurt. Apply this across hundreds of source-target pairs and you get a histogram of how productive each cross-task transfer is. Empirically, early in training cross-mode tasks mostly hurt each other (negative transfer), and as training proceeds they become more productive.

**Idea 2 — FiLM is too narrow for highly multimodal task distributions; generalize it.** The prior state-of-the-art (MMAML) uses FiLM to modulate per-task: a small task-encoder produces a task embedding, a generator produces per-channel scale $\eta$ and shift $\gamma$, the base network's feature maps are then $\hat{Y}_i = \eta_i Y_i + \gamma_i$. The paper points out that **this is equivalent** to multiplying the *entire convolutional kernel* of channel $i$ by the single scalar $\eta_i$ — which leaves a one-dimensional degree of freedom per channel for each task. When the task distribution mixes characters and birds and aircraft, that's not enough room for each task to find its own niche; the tasks "fight for capacity".

**Kernel Modulation (KML)** lifts this: instead of one scalar per channel, generate a *matrix* the shape of the convolutional kernel, with one modulation entry per weight. The modulated kernel is $\hat{W}^l = W^l \odot (J + M^l)$ — base weights plus per-weight residual perturbation. To keep the generator small, the modulation matrix is built as the outer product of two small vectors (rank-1 in the matrix sense, but acting on every weight).

Headline numbers:

- 5-mode 5-shot accuracy: ProtoNet 58.91% → MProtoNet (FiLM) 59.95% → MProtoNet+KML **64.91%**, a ~5 pp jump over FiLM modulation.
- Across all 8 configurations of (2/3/5 mode, 1/5 shot), KML wins or ties.
- The transference histogram shifts toward positive transfer, confirming KML reduces inter-task fighting.

**For the project**, this paper says two useful things. (a) If you ever extend the project's task setting to truly multimodal stimuli (different physiological regimes, very different sensory domains), FiLM may not be expressive enough as a modulator — KML is a drop-in upgrade in the same conceptual frame. (b) The **transference metric** is a generic diagnostic for multi-task interference and could be applied any time the project trains a single policy across heterogeneous task variants.

## Phase 2 — Graduate-Level Deep Dive

### The Transference Metric

Let $\theta_t$ be the meta-learner parameters at step $t$, $\alpha$ the learning rate, and $T_i, T_j$ two tasks (each with support set $S$ and query set $Q$). After one SGD step on task $i$:

$$
\theta_i^{t+1} = \theta_t - \alpha \nabla_{\theta_t} L_{T_i}(Q_i; \theta_t, S_i) . \tag{1}
$$

The transference from $i$ to $j$ is the loss ratio on target task $j$:

$$
\text{LR}_{i \to j} = \frac{L_{T_j}(Q_j; \theta_i^{t+1}, S_j)}{L_{T_j}(Q_j; \theta_t, S_j)} . \tag{2}
$$

Interpretation:

- $\text{LR}_{i \to j} < 1$: gradient step on $i$ *reduced* loss on $j$ → positive transfer.
- $\text{LR}_{i \to j} > 1$: gradient step on $i$ *increased* loss on $j$ → negative transfer.
- $\text{LR}_{i \to i}$: self-transference, a baseline.

**Adaptation for meta-learning generalization.** Fifty et al. 2020 used transference only on training tasks. This paper samples source $i$ from meta-train and target $j$ from meta-test — making the metric a *generalization* measure rather than a training-loss diagnostic.

### MMAML Modulation (the Baseline)

Task encoder $h_\phi$ produces a task embedding $\upsilon_T = h_\phi(S_T)$. Parameter generator $g_\phi$ produces FiLM scale-and-shift:

$$
\omega_T = g_\phi(\upsilon_T) = \{\eta_T, \gamma_T\} . \tag{3a}
$$

For the $i$-th output channel of a base-network layer:

$$
\hat{Y}_i = \eta_i Y_i + \gamma_i, \tag{3}
$$

with $\eta_i, \gamma_i$ scalars (one per channel).

### The Re-Interpretation Lemma

A standard conv layer computes $Y_i = W_i * X + b_i$ where $W_i$ is the $i$-th kernel (shape $k \times k \times C_{\text{in}}$) and $b_i$ a scalar bias. Substituting into (3):

$$
\hat{Y}_i = \eta_i (W_i * X + b_i) + \gamma_i = (\eta_i W_i) * X + (\eta_i b_i + \gamma_i) . \tag{5}
$$

So FiLM is *equivalent* to a kernel modulation

$$
\hat{W}_i = \eta_i W_i, \qquad \hat{b}_i = \eta_i b_i + \gamma_i .
$$

This makes the limitation crisp: the modulated kernel $\hat{W}_i$ is a *uniform rescaling* of the base kernel $W_i$ — every spatial position and every input channel of $\hat{W}_i$ is forced to be the same scalar multiple of $W_i$. The per-task degree of freedom is exactly one scalar per output channel.

### Kernel Modulation (KML)

KML generalizes FiLM by giving each *weight* of the kernel its own modulation. For layer $l$:

$$
\hat{W}^l_T = W^l \odot \bigl(J + M^l(\upsilon_T, \phi)\bigr), \tag{6}
$$

$$
\hat{b}^l_T = b^l + \Delta b^l(\upsilon_T, \phi), \tag{7}
$$

where $M^l(\upsilon_T, \phi)$ has the same shape as $W^l$ and $J$ is the all-ones tensor. The Hadamard product $\odot$ is element-wise, so each weight gets multiplied by $(1 + M^l_{\text{element}})$ — a residual perturbation. The structural relationship to FiLM:

- **FiLM is a strict special case:** if $M^l_{\text{element}} = \eta_i - 1$ for every element in the $i$-th output kernel and $\Delta b^l_i = \eta_i b_i + \gamma_i - b_i$, then KML reduces exactly to MMAML's FiLM.
- **KML enlarges the per-task degree of freedom** from $O(C_{\text{out}})$ (one scalar per output channel) to $O(C_{\text{out}} \times C_{\text{in}} \times k \times k)$ (one scalar per weight), at the cost of much more generator capacity.

### Parameter-Efficient Generator (the outer-product trick)

To keep the generator from blowing up — generating one number per weight of a large CNN is expensive — KML factors $M^l$ as a rank-1 outer product:

$$
M^l(\upsilon_T, \phi) = g^l_{\phi_1}(\upsilon_T) \otimes g^l_{\phi_2}(\upsilon_T), \tag{8}
$$

with $g^l_{\phi_1}, g^l_{\phi_2}$ small MLPs producing vectors that, when outer-producted, give a matrix reshapeable to the kernel shape. So the modulation matrix has rank 1 *as a matrix*, but every element of the kernel still receives its own modulation factor.

The bias offset uses a third small MLP:

$$
\Delta b^l(\upsilon_T, \phi) = g^l_{\phi_3}(\upsilon_T) . \tag{9}
$$

Versus a single full MLP generating $|W^l|$ outputs, this factored generator reduces parameters in $g_\phi$ by ~150×.

### Forward Pass and Training

For task $T_i$, compute task embedding $\upsilon_{T_i}$ from the support set, then for each layer $l$:

1. Generate $(M^l, \Delta b^l)$ via (8), (9).
2. Compute modulated $(\hat{W}^l_{T_i}, \hat{b}^l_{T_i})$ via (6), (7).
3. Forward through the layer with these per-task parameters.

Training objective (algorithm 2):

$$
\theta \leftarrow \theta - \alpha \nabla_\theta \sum_{T_i \sim p(\mathcal{T})} L_{T_i}(Q_i; \hat{\theta}_{T_i}, S_i),
$$

with simultaneous updates of $\phi$ (task encoder) and $\varphi$ (generator).

### Why KML Increases Positive Transfer

The negative-transfer mechanism: with FiLM, every task's modulated kernel is a scalar multiple of the same base kernel — the base kernel must be a compromise that serves *all* modes. With KML, each task gets a per-weight perturbation, so the base kernel can be optimized for a "good prior" with each task's perturbation handling its specifics. Empirically (Figure 5 of the paper), the histogram of $\text{LR}_{i \to j}$ shifts left (more $< 1$) under KML.

### Hard Parameter Sharing as the Theoretical Lens

The paper recasts episodic training as multi-task learning on episodes. **Hard parameter sharing** in MTL: a shared backbone with task-specific heads — well-known to regularize and improve cross-task generalization (Caruana 1993). KML's modulated kernel $\hat{W}^l = W^l \odot (J + M^l)$ explicitly enforces hard parameter sharing on $W^l$ and adds a task-specific perturbation $M^l$, which the supplementary's "1st-Layer Shared / N-Layer Shared" experiments validate: sharing early layers (low-level features) and modulating only later layers gives the best unimodal accuracy.

## Connections to the Project

- **FiLM-as-channel-affine is a strict subset.** The re-interpretation lemma (eq. 5) — FiLM = uniform kernel rescaling — is a useful theoretical point to internalize. When the project documents "we use FiLM to modulate the policy with the neuromodulator state," the implicit claim is that **one scalar per channel is enough capacity** for the modulator to specialize behaviors. For low-dimensional modulator inputs (a few scalars representing physiology), this is plausible. For richer / more heterogeneous modulators, KML offers a drop-in expressive upgrade in the same architectural slot.
- **KML as an expressive alternative to FiLM in the project's stack.** If experiment results show the FiLM-modulated agent saturates against a baseline at high task-diversity (the kind of plateau the project's H₁a refutation-class results have shown), the natural next experiment is to swap the per-channel FiLM γ/β for KML's per-weight modulation matrix. The outer-product factorization (eq. 8) keeps the parameter count bounded.
- **Transference as a diagnostic for the project's multi-task variants.** When the project explores 2-context mixtures (e.g., a "high-tonic" and "low-tonic" pain regime, or different terrain types), Algorithm 1 of this paper is a ready-made way to ask "does training jointly help, or do the contexts hurt each other?" This is exactly the question `experiment-designer` and `experiment-analyzer` ask post-hoc; transference gives a *during-training* answer.
- **Connections within the corpus.**
  - Perez et al. 2018 (FiLM, Batch 1) → the lemma reinterprets exactly that paper's mechanism as a uniform kernel rescaling.
  - Turkoglu et al. 2022 (`turkoglu_2022_film_ensemble.md`, this batch) → uses FiLM γ/β to instantiate diverse *ensemble members*; KML's per-weight expressiveness is the opposite trade-off (more capacity per "member"/task).
  - Vuorio 2019 (cited reference [1], MMAML) → the immediate parent work; the project should cite MMAML and KML together when discussing task-conditioned modulation.
  - Vaswani 2017 (`vaswani_2017_attention.md`) → attention is a *different* conditioning primitive; KML/FiLM are both feature-affine, attention is feature-aggregation. The Vuorio 2019 finding (quoted by Turkoglu 2022, this batch) that "FiLM outperforms attention-based modulation in this context and is more stable" is the empirical comparison point.
- **RL extension caveat.** Section 6 mentions RL experiments in the supplementary but defers details. For the project's RL setting, the *meta-learning episodic* framing does not transfer cleanly (the project's environments are not few-shot in the support/query sense). The relevant abstraction is **task-conditioned RL** with KML/FiLM as the conditioning mechanism. The transference metric (eq. 2) would need re-derivation for the RL inner loop — likely as expected-return ratio rather than loss ratio.
