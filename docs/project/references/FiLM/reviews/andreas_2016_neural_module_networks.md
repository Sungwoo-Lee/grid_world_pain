---
title: "Neural Module Networks"
authors: ["Jacob Andreas", "Marcus Rohrbach", "Trevor Darrell", "Dan Klein"]
year: 2016
venue: "CVPR 2016"
slug: andreas_2016_neural_module_networks
source_pdf: "sources/Andreas et al. 2016 - Neural Module Networks.pdf"
topic: FiLM
---

# Andreas et al. 2016 — Neural Module Networks

## Plain-English entry point

This paper tackles **visual question answering (VQA)** — given an image and a natural-language question like *"What color is his tie?"* or *"Is there a red shape above a circle?"*, produce the correct answer. The authors argue that monolithic deep networks miss something important about questions: they're *compositional*. Different questions share substructure ("where is the dog?" and "where is the cat?" share the where-is operation; "where is the dog?" and "what color is the dog?" share the find-the-dog operation). A monolithic CNN-LSTM has no way to exploit that shared structure.

The proposed solution: a **Neural Module Network (NMN)** is a small library of neural-network "modules", each typed by the kind of computation it performs — `find[c]` (image → attention heatmap for concept $c$), `transform[c]` (attention → relocated attention, e.g. `transform[above]` shifts upward), `combine[c]` (two attentions → one, e.g. `combine[and]`), `describe[c]` (image + attention → label, e.g. `describe[color]`), and `measure[c]` (attention → label, e.g. `measure[is]` checks non-empty). For each question, a *separate* network is dynamically assembled from these modules by parsing the question into a small symbolic expression and mapping that expression to a tree of module instances. All instances of a given module type **share weights** across questions. The whole assembly is trained end-to-end by maximum likelihood, with no module-supervision — `find[dog]` learns to be a dog detector solely because of weight tying and the downstream task loss.

The model achieves state-of-the-art on VQA (natural images) and crushes baselines by ~25% absolute accuracy on a new synthetic **SHAPES** dataset designed to stress compositional reasoning. Critically, the model **generalizes to longer compositions than seen at training** — training on questions of size ≤ 5 modules and testing on size-6 questions works.

## Section-ordered backbone

**1. Introduction.** VQA is compositional. Existing approaches are either monolithic (CNN + RNN → classifier, no compositional structure) or symbolic (semantic parsers + logical inference, no neural representations). The paper merges both: parse the question into a symbolic structure, *use that structure to assemble a neural network*, train end-to-end. Contributions: (a) NMN architecture for discrete composition of heterogeneous neural modules; (b) practical instantiation for VQA via a Stanford-parser pipeline; (c) new SHAPES dataset for stress-testing compositionality.

**2. Motivations.** Two empirical facts: state-of-the-art performance on different vision tasks needs different network *topologies* (no single best topology), but intermediate representations transfer (ImageNet pretraining). NMN generalizes this: treat VQA as a highly multitask learning problem where each question is its own task. Predict the computation graph per question.

**3. Related work.** VQA datasets (DAQUAR, COCO-QA, VQA). Classical compositional approaches (Liang's semantic parsers); neural approaches with joint embeddings; grounding/localization via attention. Recursive neural networks build the graph from syntactic structure but apply one module repeatedly; memory networks have a fixed `find`-then-`describe` graph; NMNs allow *heterogeneous* per-example graphs with mixed message types (features, attentions, labels) — distinguishing this work.

**4. Neural module networks for VQA.**
- **4.1 Module inventory.**
  - `find[c]: image → attention`. Convolves each image position with a per-$c$ weight vector; produces a heatmap.
  - `transform[c]: attention → attention`. Two-FC-layer ReLU MLP (first FC: dim 32; second FC: same dim as input) with per-$c$ weights. E.g., `transform[above]` shifts attention upward, `transform[not]` inverts.
  - `combine[c]: attention × attention → attention`. Convolution + nonlinearity merging two attention maps. E.g., `combine[and]`, `combine[or]`.
  - `describe[c]: image × attention → label`. Computes attention-weighted average of image features, then a single FC. E.g., `describe[color]`.
  - `measure[c]: attention → label`. FC layers on a flattened attention map. E.g., `measure[is]` (existence), `measure[count]`.
- **4.2 From strings to networks.** Parse question with Stanford Parser → universal dependency representation. Filter dependencies connected to the wh-word/copula. Lemmatize. Convert to symbolic form: `what is standing in the field` → `what(stand)`, `what color is the truck` → `color(truck)`, `is there a circle next to a square` → `is(circle, next-to(square))`. Map the symbolic tree to a module tree: leaves → `find`, internal → `transform` or `combine` (arity-dependent), root → `describe` or `measure` (domain-dependent).
- **4.3 Combining with an LSTM.** An LSTM (1000 units) reads the original question and produces a sentence embedding. This is added (elementwise + FC) to the NMN's root-module output to capture syntactic regularities (e.g., plural-vs-singular distinctions wiped out by the parser) and prior-knowledge defaults (e.g., bears are brown).

**5. Training.** End-to-end MLE. Dynamic batch construction: networks with the same high-level structure (e.g., `describe[color](find[x])` for various $x$) can be batched together. Each module's per-instance weights are updated only when that instance appears in a question. ADADELTA optimizer with default settings.

**6. Experiments — compositionality (SHAPES).** New synthetic dataset: 244 unique yes/no questions, each paired with 64 different images → 15,616 question-image pairs. Questions require recognizing colors, shapes, and spatial relations among multiple objects. Results: NMN 90.6% overall vs VIS+LSTM 65.3% vs majority 63.0%. NMN remains strong on size-6 questions even when trained on ≤5 — *compositional generalization* is demonstrated.

**7. Experiments — VQA.** Real images. NMN with `find`/`combine`/`describe` modules (877 module instances, 51138 distinct layouts) combined with the LSTM question encoder. Achieves state-of-the-art on the VQA test-dev and test-standard splits.

**8. Conclusion.** NMNs offer a *discrete*-composition path to compositional generalization, complementary to *monolithic* deep-network conditioning. Modules learn task-specific specializations from end-to-end gradient signal alone — `find[dog]` becomes a dog detector with no module-level supervision.

## Phase 1 — undergraduate-level synthesis

**Key idea.** Don't build *one* network that answers every question. Build a *library of small typed networks* ("modules") and, for each question, *assemble a different network on the fly* from those modules based on the question's grammatical structure.

**Why this is different from a regular CNN-LSTM VQA model.** A monolithic VQA network has to learn separately how to answer every sentence pattern. NMN exploits the fact that *the operation `find-the-dog`* is shared between "where is the dog?" and "what color is the dog?", and *the operation `find-the-color`* is shared between "what color is the dog?" and "what color is the cat?". By tying weights across questions, the model gets compositional re-use for free.

**Setup.**
1. Define five module types. `find[c]` makes an attention map for thing $c$. `transform[c]` moves/transforms an attention (e.g. `above`). `combine[c]` merges two attentions (e.g. `and`, `or`). `describe[c]` labels an attended region (e.g. `color`). `measure[c]` labels a whole attention (e.g. `is`, `count`).
2. **Parse the question** into a small symbolic expression (e.g. *"What color is his tie?"* → `color(tie)`).
3. **Map the expression to a module tree** (leaves → `find`, root → `describe`/`measure`, internal → `transform`/`combine`).
4. **Execute** the tree on the image. Train end-to-end.
5. **Bias with question context**: an LSTM also reads the raw question and its hidden state is added to the NMN output (helps with syntactic cues the parser discards).

**Headline result.** On the SHAPES synthetic compositional-reasoning dataset, NMN 90.6% vs the strongest CNN-LSTM baseline at 65.3%. On the natural-image VQA dataset, state-of-the-art at submission time. Critically, training on shorter compositions and testing on longer compositions works — modules generalize compositionally.

**Initial takeaway.** NMN is the discrete / symbolic / sparse-composition cousin of conditional-modulation networks like FiLM. Where FiLM lets a question modulate every channel of a single big CNN smoothly (continuous routing), NMN selects a discrete subset of modules and wires them together (combinatorial routing). NMN is a direct ancestor of compositional FiLM and of the End-to-End Module Networks (Hu et al. 2017, this batch), which replace the off-the-shelf parser with a learned layout policy.

## Phase 2 — graduate-level deep dive

### Model formulation

For each training datum $(w, x, y)$ with $w$ = question, $x$ = image, $y$ = answer, the model is specified by:
- A finite collection of **modules** $\{m\}$, each with parameters $\theta_m$.
- A **layout predictor** $P: \mathcal{W} \to \mathcal{T}$ mapping a string $w$ to a typed tree $T = P(w)$.

The induced computation is

$$
p(y \mid w, x; \theta) = M_{T}\!\big(x; \{\theta_m\}_{m \in T}\big),
$$

where $M_T$ is the network obtained by executing the modules in tree order, and $\theta = \bigcup_m \theta_m$ are the weights tied across the entire training set per module *instance* (not per question).

### Module signatures

Three data types: $I$ = image, $A$ = attention (unnormalized), $L$ = label distribution.

| Module | Signature | Implementation |
|---|---|---|
| `find[c]` | $I \to A$ | $\mathrm{find}_c(x)_{p} = w_c^\top x_{p}$, per-position conv with per-$c$ kernel |
| `transform[c]` | $A \to A$ | Two-layer ReLU MLP: $\mathrm{ReLU}(W_2^c \cdot \mathrm{ReLU}(W_1^c \cdot a))$, $W_1^c: |A| \to 32$, $W_2^c: 32 \to |A|$ |
| `combine[c]` | $A \times A \to A$ | $\sigma(W_c \ast [a_1, a_2])$, conv on stacked attentions |
| `describe[c]` | $I \times A \to L$ | $\mathrm{softmax}\!\big(W_c \cdot \sum_p \tilde a_p \cdot x_p\big)$, where $\tilde a = \mathrm{softmax}(a)$ |
| `measure[c]` | $A \to L$ | FC layers on the flattened attention |

The instance index $c$ controls per-instance weights $\theta_m^c$; the type controls the function-form. So `find[dog]` and `find[cat]` share architecture but have different weights $w_{\text{dog}}, w_{\text{cat}}$.

### Layout assembly

Given the parser-derived symbolic form $\sigma(w)$ — a small expression like `color(truck)` or `is(circle, next-to(square))` — define the tree $T = P(w)$ recursively:

- **Leaf** at symbol $c$: $T_{\text{leaf}}(c) = \texttt{find[c]}$.
- **Unary internal** at symbol $c$ with child $T_{\text{ch}}$: $T(c, T_{\text{ch}}) = \texttt{transform[c]}(T_{\text{ch}})$.
- **Binary internal** at symbol $c$ with children $T_1, T_2$: $T(c, T_1, T_2) = \texttt{combine[c]}(T_1, T_2)$.
- **Root** at symbol $c$:
  - For VQA: $T_{\text{root}}(c, T_{\text{ch}}) = \texttt{describe[c]}(x, T_{\text{ch}})$.
  - For SHAPES (yes/no): $T_{\text{root}}(c, T_{\text{ch}}) = \texttt{measure[c]}(T_{\text{ch}})$.

### Combining with the LSTM question encoder

The final answer distribution combines the NMN output with an LSTM encoding of the raw question. Let $h_w \in \mathbb{R}^{1000}$ be the LSTM final hidden state, $r_T \in \mathbb{R}^{|L|}$ the NMN root-module output (a feature vector before the final softmax in the VQA branch). Then

$$
\hat y = \mathrm{softmax}\!\Big(W_y \cdot \mathrm{ReLU}\big(r_T + W_h h_w\big)\Big),
$$

where $W_h, W_y$ are learned linear projections. The role of $h_w$ is to capture syntactic regularities (e.g., plural agreement) and prior-knowledge defaults (bears are brown) that the discarded parts of the parse don't see.

### Training objective

End-to-end MLE over the training set:

$$
\mathcal{L}(\theta) = -\sum_{(w_n, x_n, y_n)} \log p\big(y_n \mid w_n, x_n; \theta\big) = -\sum_n \log \hat y_n[y_n].
$$

Because each module instance $m^c$ appears only when the relevant symbol $c$ appears in $\sigma(w)$, gradient updates are **highly unbalanced** across instances — popular instances (`find[man]`) train fast, rare ones (`find[unicorn]`) slowly. The authors use ADADELTA to give per-parameter adaptive learning rates, which empirically beats vanilla SGD.

### Compositional generalization

Because modules are *typed* and *type-checked* at assembly, novel compositions are well-formed even if unseen at training. Empirically, NMN trained only on questions whose tree has $\leq 5$ modules generalizes to test questions with 6 modules at no loss (Table 2: train-size-≤5 reaches 89.7% on size-6 questions vs 85.2% for the full-data NMN). This is the most striking result of the paper: compositional generalization is achievable with discrete, weight-tied module re-use.

### Empirical anchors

| Dataset | Method | Accuracy |
|---|---|---|
| SHAPES (overall) | Majority | 63.0% |
| SHAPES (overall) | VIS+LSTM | 65.3% |
| SHAPES (overall) | NMN | 90.6% |
| SHAPES (size 6, train ≤ 5) | NMN | 89.7% (compositional generalization) |
| VQA test-dev | NMN + LSTM | state-of-the-art at submission |

## Connections

- **`hu_2017_e2e_module_networks.md`** (this batch): direct successor. Replaces the *fixed off-the-shelf parser* used here with a **learned layout policy** (RL/REINFORCE) so the layout itself is part of the model, end-to-end. Same module-library philosophy, fully differentiable wiring.
- **`shazeer_2017_sparse_moe.md`** (this batch): MoE is the *continuous* counterpart of module networks — both pick a sparse subset of computation units per example, but MoE uses a learned dense softmax-then-top-$k$ gate while NMN uses a discrete parser tree. The "load balancing" problem in MoE has a sibling in NMN: rare module instances are undertrained, and ADADELTA partly compensates.
- **`ha_2016_hypernetworks.md`** (this batch): NMN is *symbolic discrete composition*; hypernetworks are *continuous weight generation*. They occupy opposite ends of the conditioning spectrum: NMN's question $\to$ tree $\to$ module-selection is discrete; hypernet's context $\to$ embedding $\to$ weights is continuous. Both achieve modularity (Galanti & Wolf 2020 sense).
- **`perez_2018_film.md`** (other batch — FiLM): NMN explicitly anticipates **compositional FiLM**. In FiLM, every layer of one CNN is modulated by per-channel $\gamma, \beta$ generated from the question; in NMN, a *tree* of modules is selected and chained. FiLM-on-NMN-modules would unify both — each module is FiLM-conditioned, the tree is parser-generated.
- **`galanti_wolf_2020_hypernet_modularity.md`** (this batch): provides a theoretical lens for why module networks work — capacity in the *layout/composer* is leveraged efficiently, per-task primary networks (modules) can stay small.
