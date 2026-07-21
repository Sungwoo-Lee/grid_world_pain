> **Per-paper review — in-context-learning corpus, paper 12 of 15 (original batch).**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§12); content is identical. Manifest: [[in_context_learning_sources]].

# 12. Schug et al. 2025 — Attention as a Hypernetwork

**PDF:** `docs/project/references/in_context_learning/sources/Schug et al. 2025 - Attention as a Hypernetwork.pdf`
**Venue:** ICLR 2025. **Authors:** Simon Schug, Seijin Kobayashi, Yassir Akram, João Sacramento, Razvan Pascanu (ETH Zürich / Google DeepMind).

## Phase 1 — Foundational Overview (undergraduate level)

**The question in plain terms.** Transformers can sometimes solve *new combinations* of familiar pieces — e.g. if trained on tasks that use rules A, B, C, D in various pairs, they can handle the unseen pair (A, D). This is called **compositional generalization**. Why does the attention mechanism support it? This paper answers by showing that **multi-head attention is secretly a hypernetwork**.

- A **hypernetwork** is a small network that outputs the *weights* of a second ("value") network, which then does the actual work on the input. The hypernetwork takes a short **latent code** describing "which operation to run."
- The paper's central re-derivation: for a single query token, if you look at the attention scores **across the heads** (not across the keys, as usual), that little vector of per-head scores *is* the hypernetwork's latent code. It selects a linear combination of fixed per-head "operation matrices" and applies the result to each key's value.
- So the number of heads `H` = the dimension of the hypernetwork's latent code. Each head contributes a reusable building-block operation; the attention scores mix them.

**Key findings.**
1. **The equivalence** — multi-head attention = a linear hypernetwork generating a per-(key,query) linear value network (Eqs. 2–5 below). This is a *different* decomposition from the well-known "linear attention = fast-weight programmer" (Schlag 2021, §13), which sums outer products over the **key** index; Schug sums over the **head** index.
2. **HYLA (Hypernetwork Linear Attention)** — a drop-in modification that makes the value network *nonlinear* (adds a ReLU) **without adding parameters**, by exploiting the fact that the value+output projections already form a deep-linear network. It also normalizes attention scores across heads (RMSHead) rather than across keys (softmax). HYLA improves compositional generalization.
3. **The latent code is interpretable** — on two abstract-reasoning tasks (a fuzzy-logic task and the new SRAVEN benchmark), the per-head attention-score vector for the "response" token clusters by the *subtask / rule* being applied. You can train a logistic-regression classifier on the latent code and read off which rule the network is using — even on held-out task compositions.
4. **Scaling helps** — bigger models + more data → compositional generalization emerges + latent space becomes more functionally structured. Killing the mechanism (using a single head) hurts.
5. **SRAVEN** — a new symbolic version of Raven's Progressive Matrices with parametric difficulty and controllable rule-composition splits, including a "finding correspondences" difficulty (column-wise feature permutations).

**Initial takeaway.** Multiple attention heads aren't just an engineering trick for stability — they give attention a built-in hypernetwork that composes reusable operations from a compact code. Strengthening that mechanism (HYLA) improves systematic generalization and even closes part of the linear-vs-softmax gap in language modeling. **Bridge to the project:** this is a first-principles argument that attention *is* conditional modulation — exactly the FiLM/hypernetwork family — so a FiLM or hypernetwork conditioning module is not an add-on foreign to attention, but a more explicit exposure of a mechanism attention already contains.

## Phase 2 — Graduate-Level Deep Dive

### 2.1 Setup and notation

Bold lower-case = vectors ($\mathbf{z}$), bold upper-case = matrices ($\mathbf{A}$), bold italic upper-case = learnable parameters ($\boldsymbol{W}_V$). Self-attention maps an input sequence $\mathbf{X}\in\mathbb{R}^{D\times T}$ (feature dim $D$, sequence length $T$) to outputs $\mathbf{Y}\in\mathbb{R}^{D\times T}$. There are $H$ heads with per-head dimension $D_{\text{head}}=D/H$.

For each head $h\in\{1,\dots,H\}$, project into keys and queries with head-specific matrices $\boldsymbol{W}_h^{\text{key}},\boldsymbol{W}_h^{\text{query}},\boldsymbol{W}_h^{\text{value}}\in\mathbb{R}^{D_{\text{head}}\times D}$:

$$
\mathbf{K}_h = \boldsymbol{W}_h^{\text{key}}\mathbf{X},\qquad
\mathbf{Q}_h = \boldsymbol{W}_h^{\text{query}}\mathbf{X}.
$$

The (unnormalized) head-specific attention matrix and the stacked, normalized tensor are

$$
\tilde{\mathbf{A}}_h = \frac{\mathbf{Q}_h^{\top}\mathbf{K}_h}{\sqrt{D_{\text{head}}}},
\qquad
\mathbf{A}=\sigma\!\left(\big[\tilde{\mathbf{A}}_1\ \tilde{\mathbf{A}}_2\ \cdots\ \tilde{\mathbf{A}}_H\big]\right),
\qquad
\mathbf{A}=(a_{h,q,k})_{h,q,k}\in\mathbb{R}^{H\times T\times T}.
$$

$\sigma(\cdot)$ is the normalizer: **identity** for linear attention, **softmax over the key index $k$** (per head $h$, per query $q$) for standard softmax attention.

### 2.2 The core derivation: attention as a hypernetwork (Eqs. 2–5)

Fix a single query index $q$. Let $\mathbf{x}_k\in\mathbb{R}^{D}$ be the $k$-th column of $\mathbf{X}$. Standard multi-head attention writes the output as the output-projection $\boldsymbol{W}^{\text{out}}\in\mathbb{R}^{D\times D}$ applied to the concatenation ($\bigoplus$) over heads of the per-head attention-weighted value sums:

$$
\text{MHA}_q(\mathbf{X}) \;=\; \boldsymbol{W}^{\text{out}}\bigoplus_{h=1}^{H}\sum_{k=1}^{T} a_{h,q,k}\,\boldsymbol{W}_h^{\text{value}}\mathbf{x}_k. \tag{2}
$$

**Step 1 — split the output projection over heads.** Write $\boldsymbol{W}^{\text{out}}=\bigoplus_{h=1}^{H}\boldsymbol{W}_h^{\text{out}}$ where each slice $\boldsymbol{W}_h^{\text{out}}\in\mathbb{R}^{D\times D_{\text{head}}}$. A block matrix times a stacked vector is the sum of block-times-subvector, so the concatenation collapses to a sum over heads:

$$
\text{MHA}_q(\mathbf{X}) \;=\; \sum_{h=1}^{H}\boldsymbol{W}_h^{\text{out}}\sum_{k=1}^{T} a_{h,q,k}\,\boldsymbol{W}_h^{\text{value}}\mathbf{x}_k. \tag{3}
$$

**Step 2 — swap the order of summation** ($\sum_h\sum_k = \sum_k\sum_h$) and pull the scalar $a_{h,q,k}$ out. Because $a_{h,q,k}$ is a scalar it commutes with the matrices:

$$
\text{MHA}_q(\mathbf{X}) \;=\; \sum_{k=1}^{T}\Bigg(\underbrace{\sum_{h=1}^{H}\underbrace{a_{h,q,k}}_{\text{latent code}}\,\boldsymbol{W}_h^{\text{out}}\boldsymbol{W}_h^{\text{value}}}_{\text{hypernetwork}}\Bigg)\mathbf{x}_k. \tag{4}
$$

**Step 3 — name the generated value network.** Define the per-(key,query) weight matrix

$$
\mathbf{W}_{q,k} \;:=\; \sum_{h=1}^{H} a_{h,q,k}\,\boldsymbol{W}_h^{\text{out}}\boldsymbol{W}_h^{\text{value}}\;\in\;\mathbb{R}^{D\times D},
\qquad\text{so}\qquad
\text{MHA}_q(\mathbf{X}) \;=\; \sum_{k=1}^{T}\mathbf{W}_{q,k}\,\mathbf{x}_k. \tag{5}
$$

**Reading of the result.**
- The vector $\mathbf{a}_{q,k}:=(a_{1,q,k},\dots,a_{H,q,k})^{\top}\in\mathbb{R}^{H}$ is the **latent code**: an $H$-dimensional specification of the operation to run on key $k$ for query $q$. Latent dimension = number of heads.
- The map $\mathbf{a}_{q,k}\mapsto \mathbf{W}_{q,k}=\sum_h a_{h,q,k}\,\boldsymbol{W}_h^{\text{out}}\boldsymbol{W}_h^{\text{value}}$ is a **linear hypernetwork** whose "basis operations" are the fixed rank-$\le D_{\text{head}}$ matrices $\boldsymbol{W}_h^{\text{out}}\boldsymbol{W}_h^{\text{value}}$. The *same* hypernetwork (same basis matrices) is reused for every $(q,k)$ pair — this weight-sharing is exactly what incentivizes reuse and recombination of operations, hence compositional generalization.
- The dot products $\tilde a_{h,q,k}\propto \mathbf{q}_{h,q}^{\top}\mathbf{k}_{h,k}$ that produce the code can be read as an **amortized inference** step: from the key/query content, infer which composition of operations to apply.

**Contrast with the fast-weight view (Schlag 2021).** In linear attention as a fast-weight programmer, the constructed weight matrix is $\mathbf{W}_{\text{fw}}=\sum_k \mathbf{v}_k\mathbf{k}_k^{\top}$ — a sum of outer products over the **key** index, one global matrix accumulated over the sequence. Schug's decomposition instead sums over the **head** index, producing a *distinct* $\mathbf{W}_{q,k}$ per key-query pair. The two views are complementary factorizations of the same layer.

### 2.3 HYLA — Hypernetwork Linear Attention (Eqs. 6–7)

Observation: in Eq. (4) the generated operator $\boldsymbol{W}_h^{\text{out}}\boldsymbol{W}_h^{\text{value}}$ is a *product of two linear maps* — i.e. a deep-linear network with no nonlinearity between them. HYLA inserts an element-wise nonlinearity $\phi$ (ReLU) between the value projection and the output projection, turning the generated value network into a genuine one-hidden-layer MLP **without any new parameters**:

$$
\text{HYLA}_q(\mathbf{X}) \;=\; \sum_{k=1}^{T}\left(\sum_{h=1}^{H} a_{h,q,k}\,\boldsymbol{W}_h^{\text{out}}\right)\,\phi\!\left(\sum_{h=1}^{H} a_{h,q,k}\,\boldsymbol{W}_h^{\text{value}}\,\mathbf{x}_k\right)
\;=\; \sum_{k=1}^{T}\mathbf{W}'_{q,k}\,\phi\!\left(\mathbf{W}_{q,k}\mathbf{x}_k\right), \tag{6–7}
$$

with $\phi(x)=\max(0,x)$, and the two generated matrices
$\mathbf{W}_{q,k}=\sum_h a_{h,q,k}\boldsymbol{W}_h^{\text{value}}$ (first layer, into hidden) and
$\mathbf{W}'_{q,k}=\sum_h a_{h,q,k}\boldsymbol{W}_h^{\text{out}}$ (second layer, out of hidden). Note that relative to Eq. (4) the single product $\boldsymbol{W}_h^{\text{out}}\boldsymbol{W}_h^{\text{value}}$ is *split* so the nonlinearity can sit between the two factors.

**Normalization — RMSHead.** HYLA sets $\sigma(\cdot)=\text{RMSHead}(\cdot)$: normalize the attention scores across the **head** index (locally per $(q,k)$), using RMSNorm

$$
\text{RMSNorm}(\mathbf{x}) \;=\; \frac{\mathbf{x}}{\sqrt{\tfrac{1}{n}\sum_{i=1}^{n} x_i^2}}\,,
$$

with $n=H$ and no learnable gain. Rationale: this keeps the *variance-preserving* property of standard NN initialization (Glorot & Bengio) for the hypernetwork-generated weights $\mathbf{W}_{q,k}$, stabilizing gradient-based training — because $\mathbf{W}_{q,k}$ is a code-weighted sum of the basis matrices, unbounded codes would blow up the generated weights' scale. Crucially, RMSHead is **local to each $(q,k)$**: unlike softmax-over-keys, it needs **no communication across keys**, preserving linear attention's efficiency.

### 2.4 Why the single-head case degenerates

With $H=1$, Eq. (4) becomes $\mathbf{W}_{q,k}=a_{1,q,k}\,\boldsymbol{W}_1^{\text{out}}\boldsymbol{W}_1^{\text{value}}$ — a *scalar rescaling* of one fixed operator. The hypernetwork can no longer **compose** multiple basis operations; the latent code collapses to a 1-D gain. The paper confirms empirically (SRAVEN, Fig. 5C) that $H=1$ sharply reduces OOD accuracy, directly supporting the claim that composition across heads is the load-bearing mechanism. This also reframes the classic "you can prune all-but-one head" finding (Voita 2019; Michel 2019) as **module collapse** rather than evidence that heads are redundant.

### 2.5 Tasks and empirical structure

**Fuzzy-logic task.** Scalars $x_i\in[0,1]$; Zadeh fuzzy operators $x_i\wedge x_j=\min(x_i,x_j)$, $x_i\vee x_j=\max(x_i,x_j)$, $\bar x_i=1-x_i$ (which reduce to Boolean ops on $\{0,1\}$). Each task is a disjunctive-normal-form function of $K$ terms over $L$ variables, e.g. $f=(x_1\wedge x_2\wedge x_3\wedge x_4)\vee(\bar x_1\wedge x_2\wedge \bar x_3\wedge x_4)$. The compositional split holds out a fraction of the $\binom{2^L}{K}$-style term combinations for OOD, so every *term* is seen in training but the *combination* is novel. Result: the per-head latent code of the response token is decodable (F1) into the underlying terms, and tSNE clusters by term. HYLA degrades least as the held-out fraction grows.

**SRAVEN (Symbolic Raven).** Each panel is a $K$-tuple of integers in $\{0,\dots,F-1\}$ (default $K=4$, $F=8$). $K$ of $R=8$ rules (progression, addition, difference, etc., in modular arithmetic mod $F$) govern the features across each row of a $3\times3$ matrix; the model sees 8 context panels and predicts the 9th (all $K$ sub-features must be correct). Compositional split holds out 25% of rule combinations (AB≡BA identified). **Finding correspondences**: a column-specific consistent permutation of features is applied, so the task can't be decomposed into $K$ independent single-feature tasks — the number of hypotheses explodes. Results: with enough scale all attention variants reach ~80% OOD; HYLA leads at small scale; the final-layer latent code clusters by *rule*, and cosine-similarity of average per-rule codes shows semantically related rules (addition vs. difference) sharing nearly identical codes (sign flip of operands).

**Language modeling.** 50M-param decoder-only, C4, 130B tokens: HYLA beats linear attention and approaches softmax attention — notable because softmax's edge is usually attributed to associative-recall / induction-head binding where linear attention struggles. Strengthening the hypernetwork mechanism partly closes that gap.

### 2.6 Discussion / generality

Multi-head attention is one *granularity* choice for a hypernetwork: it parameterizes **key-query-specific** value networks, then pools (sums) over keys. Other granularities are possible — a query-specific value network that subsumes the key-aggregation, or a full sequence-level operation. This connects to attention-GNNs: the message function becomes a hypernetwork subsuming attention weights, and aggregation is the sum. **Limitation:** analysis is on from-scratch models for controllability; whether pretrained large models form similarly structured latent codes is open.

## Appendix: Section-by-Section Backbone

- **Abstract.** Reformulate MHA as a hypernetwork → attention scores across heads = a composable low-dim latent code specifying key-query operations. Empirically the code predicts subtasks on unseen compositions. Making the generated value network nonlinear (HYLA) improves compositional generalization. Introduce SRAVEN.
- **§1 Introduction.** Compositional generalization = generalizing to unseen combinations of seen constituents; NNs notoriously struggle. Prior ICL mechanisms: gradient descent in-context (von Oswald, Dai, Akyürek), fast-weight programmer view (Schlag/Schmidhuber), task vectors (Hendel, Todd). Key claim: MHA ≡ hypernetwork ⇒ attention scores over heads = latent code. Contributions: reformulation, scaling enables compositional generalization + structured latent space, HYLA parameter-free nonlinear value net, SRAVEN benchmark.
- **§2 Attention as a hypernetwork.** §2.1 recap hypernetworks $h(\mathbf z;\theta)\to \boldsymbol W$ configuring $f(\mathbf x;\boldsymbol W)$; recap MHA; derive Eqs. (1)–(5): latent code $=(a_{h,q,k})_h$, hypernetwork $=\sum_h a_{h,q,k}\boldsymbol W_h^{\text{out}}\boldsymbol W_h^{\text{value}}$. Contrast with key-index outer-product fast-weight view. §2.2 HYLA (Eqs. 6–7): nonlinear value net without extra params + RMSHead normalization across heads; local, no cross-key communication.
- **§3 Fuzzy logic functions.** Zadeh operators (Eq. 8); DNF tasks (Eq. 9); compositional train/OOD split by terms; ICL over N examples; MSE on masked final target. §3.1 all models solve given enough context; HYLA degrades least with more held-out. Novel *unknown* terms are unsolvable by all. §3.2 latent code decodes to terms (logistic regression F1), tSNE clusters by term.
- **§4 SRAVEN.** §4.1 motivation (Raven, finding correspondences). §4.2 task: $K$-tuples of integers, $R=8$ rules, modular arithmetic, 8 context + 1 response panel, 25% rule-combos held out. §4.3 finding correspondences via column-wise feature permutation. §4.4 results: scaling enables compositional generalization (~80% OOD); single-head degenerates (Fig 5C); latent code clusters by rule (Fig 5B); related-rule code reuse (Fig 5E); logistic-regression rule decoding (Fig 5F); difficulty scales with $K$ (Fig 5D).
- **§5 Language modeling.** 50M decoder-only, C4/130B tokens; HYLA > linear, ≈ softmax; hypernetwork mechanism relevant at scale.
- **§6 Related work.** Role of multiple heads (prunability → reinterpreted as module collapse); compositional generalization + scale; Raven-based tasks.
- **§7 Discussion.** Granularity of hypernetwork parameterization; GNN connection (message = hypernetwork, aggregation = sum); pooling operators as design axis. Limitation: from-scratch only.

---
