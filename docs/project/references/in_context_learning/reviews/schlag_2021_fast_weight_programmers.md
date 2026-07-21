> **Per-paper review — in-context-learning corpus, paper 13 of 15 (original batch).**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§13); content is identical. Manifest: [[in_context_learning_sources]].

# 13. Schlag et al. 2021 — Linear Transformers Are Secretly Fast Weight Programmers

**PDF:** `docs/project/references/in_context_learning/sources/Schlag et al. 2021 - Linear Transformers Are Secretly Fast Weight Programmers.pdf`
**Venue:** ICML 2021. **Authors:** Imanol Schlag*, Kazuki Irie*, Jürgen Schmidhuber (IDSIA).

## Phase 1 — Foundational Overview (undergraduate level)

**The question in plain terms.** Standard transformers attend over the whole sequence, which costs time quadratic in sequence length and memory that grows with length. "Linear transformers" replace the softmax with a kernel trick so cost is linear and memory is a *fixed-size matrix*. This paper's headline: that fixed-size matrix is **not new** — it is exactly the "fast weights" idea Schmidhuber introduced in 1991. A slow network (the trained weights) learns to write key→value associations into a fast-changing weight matrix via **outer products**, and reads them back by matrix-vector multiply. So a linear transformer is "secretly" a **Fast Weight Programmer (FWP)**.

**Analogy.** Think of the fast-weight matrix as a whiteboard. Each new token writes an association (key, value) onto the board by adding an outer product $\mathbf v\otimes\mathbf k$. When a query arrives, you multiply the board by the query to read back the value whose key best matches. The trained ("slow") network only learns *how to generate* the keys, values, and queries — the whiteboard content is computed on the fly, in-context.

**Key findings.**
1. **Formal equivalence.** Linear self-attention (softmax removed / kernel-linearized) is algebraically identical, up to normalization, to the 1991–93 outer-product FWP. The accumulated attention matrix $\mathbf W^{(i)}=\sum_j \mathbf v^{(j)}\otimes\phi(\mathbf k^{(j)})$ is the fast-weight memory.
2. **Capacity limit.** Because associations are summed into a finite $d_{\text{dot}}\times d_{\text{value}}$ matrix, you can only store ~$d_{\text{dot}}$ mutually-retrievable associations before keys stop being orthogonal and retrieval returns interfering mixtures. Once sequence length exceeds $d_{\text{dot}}$ the model is in an **overcapacity regime**.
3. **The Delta rule fix.** The pure additive update can't *erase* stale associations. They replace it with a **delta-rule** update: before writing a new value, read out the value currently stored under that key, and write only the *correction*, scaled by a learned, dynamic "write strength" $\beta^{(i)}$. This is the **DeltaNet** — a linear transformer that can edit its own memory.
4. **DPFP kernel.** A new deterministic, parameter-free feature map $\phi$ that projects keys/queries up to a higher-dimensional, near-orthogonal space (raising capacity) without random sampling (unlike Performer/FAVOR+).
5. **Empirics.** Synthetic retrieval confirms the capacity cliff at $d_{\text{dot}}$; DeltaNet beats the sum-rule on memory-editing tasks, WMT14 translation, and WikiText-103 language modeling, and — with fast-weight carry-over across segments — runs unbounded context with a tiny fixed state.

**Initial takeaway.** This paper is the historical + mathematical anchor of the whole "context as weights" corpus. It shows the transformer's context can *literally be a weight matrix built by outer products* — the exact algebra a hypernetwork or FiLM layer performs, but accumulated over the sequence. **Bridge to the project:** the outer-product write $\mathbf v\otimes\mathbf k$ and the delta-rule editable-memory view are the raw material behind "attention = hypernetwork" (Schug, §12, over the head index instead of the key index) and behind the neuromodulatory / fast-plasticity framings (fast weights ≈ synaptic modulation, explicitly cited to von der Malsburg 1981).

## Phase 2 — Graduate-Level Deep Dive

### 2.1 The 1991 Fast Weight Programmer (Eqs. 1–3)

For a sequential input $\{\mathbf x^{(i)}\}_{i=1}^L$, $\mathbf x^{(i)}\in\mathbb R^{d_{\text{in}}}$, the slow net produces two activation patterns and writes their outer product into a fast-weight matrix:

$$
\mathbf a^{(i)},\mathbf b^{(i)} = \boldsymbol W_a\mathbf x^{(i)},\ \boldsymbol W_b\mathbf x^{(i)}, \tag{1}
$$
$$
\mathbf W^{(i)} = \sigma\!\left(\mathbf W^{(i-1)} + \mathbf a^{(i)}\otimes\mathbf b^{(i)}\right), \tag{2}
$$
$$
\mathbf y^{(i)} = \mathbf W^{(i)}\mathbf x^{(i)}. \tag{3}
$$

$\boldsymbol W_a,\boldsymbol W_b$ are trainable **slow** weights; $\mathbf W^{(i)}$ is the input-dependent **fast** weight (a key-value associative memory). Write = summation of an outer product (Eq. 2); retrieval = matrix-vector product (Eq. 3). In today's terminology $\mathbf a,\mathbf b$ are *values* and *keys* — "self-invented activation patterns."

### 2.2 Self-attention without softmax IS an FWP (Eqs. 4–11)

Auto-regressive self-attention projects each input into key/value/query and attends causally:

$$
\mathbf k^{(i)},\mathbf v^{(i)},\mathbf q^{(i)} = \boldsymbol W_k\mathbf x^{(i)},\ \boldsymbol W_v\mathbf x^{(i)},\ \boldsymbol W_q\mathbf x^{(i)}, \tag{4}
$$
$$
\mathbf K^{(i)}=[\mathbf K^{(i-1)},\mathbf k^{(i)}],\quad \mathbf V^{(i)}=[\mathbf V^{(i-1)},\mathbf v^{(i)}],\quad \mathbf y^{(i)}=\mathbf V^{(i)}\,\text{softmax}\big((\mathbf K^{(i)})^\top\mathbf q^{(i)}\big). \tag{5–7}
$$

**Remove the softmax** and use associativity of matrix multiplication:

$$
\mathbf y^{(i)} = \mathbf V^{(i)}(\mathbf K^{(i)})^\top\mathbf q^{(i)}
= \big(\mathbf V^{(i)}(\mathbf K^{(i)})^\top\big)\mathbf q^{(i)}
= \Bigg(\sum_{j=1}^{i}\mathbf v^{(j)}\otimes\mathbf k^{(j)}\Bigg)\mathbf q^{(i)}. \tag{8}
$$

The crucial step is the middle equality: instead of first forming the $i\times 1$ attention score vector $(\mathbf K^{(i)})^\top\mathbf q^{(i)}$ (cost grows with $i$), you first form the *fixed-size* $d_{\text{value}}\times d_{\text{key}}$ matrix $\mathbf V^{(i)}(\mathbf K^{(i)})^\top=\sum_j \mathbf v^{(j)}\otimes\mathbf k^{(j)}$. Define

$$
\mathbf W^{(i)} = \sum_{j=1}^{i}\mathbf v^{(j)}\otimes\mathbf k^{(j)}, \tag{9}
$$

which admits the recurrent, constant-memory form matching Eqs. 1–3 with $\sigma=\text{Id}$:

$$
\mathbf W^{(i)} = \mathbf W^{(i-1)} + \mathbf v^{(i)}\otimes\mathbf k^{(i)}, \qquad \mathbf y^{(i)}=\mathbf W^{(i)}\mathbf q^{(i)}. \tag{10–11}
$$

So the query plays the role of the FWP retrieval input, and the causal attention output is a read from an outer-product memory. **This is the paper's core equivalence.**

### 2.3 Kernel linearization of softmax (Eqs. 12–19)

Write softmax attention explicitly with kernel $\kappa(\mathbf k,\mathbf q)=\exp(\mathbf k\cdot\mathbf q)$:

$$
\mathbf y^{(i)} = \sum_{j=1}^{i}\frac{\mathbf v^{(j)}\,\kappa(\mathbf k^{(j)},\mathbf q^{(i)})}{\sum_{j'=1}^{i}\kappa(\mathbf k^{(j')},\mathbf q^{(i)})}. \tag{12}
$$

Replace the exponential kernel by a **factorizable** one $\kappa'(\mathbf k,\mathbf q)=\phi(\mathbf k)^\top\phi(\mathbf q)$ with feature map $\phi:\mathbb R^{d_{\text{key}}}\to\mathbb R^{d_{\text{dot}}}$. Then substituting and using linearity to pull $\phi(\mathbf q^{(i)})$ out of the sum:

$$
\mathbf y^{(i)} = \frac{\sum_{j=1}^{i}\mathbf v^{(j)}\big(\phi(\mathbf k^{(j)})^\top\phi(\mathbf q^{(i)})\big)}{\sum_{j'=1}^{i}\phi(\mathbf k^{(j')})^\top\phi(\mathbf q^{(i)})}
= \frac{\Big(\sum_{j}\mathbf v^{(j)}\otimes\phi(\mathbf k^{(j)})\Big)\phi(\mathbf q^{(i)})}{\Big(\sum_{j'}\phi(\mathbf k^{(j')})\Big)^\top\phi(\mathbf q^{(i)})}. \tag{13–14}
$$

Introduce the fast-weight matrix and the denominator accumulator:

$$
\mathbf W^{(i)}=\sum_{j=1}^{i}\mathbf v^{(j)}\otimes\phi(\mathbf k^{(j)}),\qquad
\mathbf z^{(i)}=\sum_{j=1}^{i}\phi(\mathbf k^{(j)}), \tag{15–16}
$$

giving the recurrent linear-transformer forward pass (Katharopoulos et al. 2020):

$$
\mathbf W^{(i)}=\mathbf W^{(i-1)}+\mathbf v^{(i)}\otimes\phi(\mathbf k^{(i)}),\quad
\mathbf z^{(i)}=\mathbf z^{(i-1)}+\phi(\mathbf k^{(i)}),\quad
\mathbf y^{(i)}=\frac{\mathbf W^{(i)}\phi(\mathbf q^{(i)})}{\mathbf z^{(i)}\cdot\phi(\mathbf q^{(i)})}. \tag{17–19}
$$

This is exactly an FWP (§2.1) with normalization. Hence: **the core of every linear transformer is an outer-product Fast Weight Programmer.**

### 2.4 Capacity limitation via tensor-product representations

Retrieval reads $\mathbf y^{(i)}=\mathbf W^{(i)}\phi(\mathbf q^{(i)})=\sum_j\mathbf v^{(j)}\big(\phi(\mathbf k^{(j)})\cdot\phi(\mathbf q^{(i)})\big)$. To recover $\mathbf v^{(m)}$ cleanly when querying with $\mathbf k^{(m)}$, we need $\phi(\mathbf k^{(j)})\cdot\phi(\mathbf k^{(m)})\approx\delta_{jm}$ — i.e. the projected keys must be near-orthogonal. In a $d_{\text{dot}}$-dimensional space there are at most $d_{\text{dot}}$ mutually orthogonal vectors, so storing more than $d_{\text{dot}}$ associations forces **crosstalk** and retrieval error. This is precisely Smolensky's (1990) tensor-product-representation (TPR) analysis: the fast-weight matrix is a second-order TPR of keys (roles) and values (fillers), and Smolensky's Theorems 3.1/3.3 formalize the crosstalk. **Difference from classic TPRs:** classic TPRs are hand-wired with a-priori symbolic structure; the FWP *learns* all the role/filler (key/value) vectors end-to-end. Consequence for transformers: when $L>d_{\text{dot}}$ the model is in overcapacity and the naive additive rule degrades.

### 2.5 The delta-rule update (Eqs. 20–25) — the DeltaNet

To operate in overcapacity, the memory must *edit*, not just accumulate. Given a new $(\mathbf k^{(i)},\mathbf v^{(i)})$:

$$
\bar{\mathbf v}^{(i)} = \mathbf W^{(i-1)}\phi(\mathbf k^{(i)}) \quad\text{(read current value at this key)}, \tag{20}
$$
$$
\beta^{(i)} = \sigma\big(\boldsymbol W_\beta\mathbf x^{(i)}\big)\in[0,1] \quad\text{(learned dynamic write-strength)}, \tag{21}
$$
$$
\mathbf v^{(i)}_{\text{new}} = \beta^{(i)}\mathbf v^{(i)} + (1-\beta^{(i)})\bar{\mathbf v}^{(i)} \quad\text{(convex combination)}. \tag{22}
$$

The update writes the new value and **removes** the old one at that key:

$$
\mathbf W^{(i)} = \mathbf W^{(i-1)} + \underbrace{\mathbf v^{(i)}_{\text{new}}\otimes\phi(\mathbf k^{(i)})}_{\text{write}} - \underbrace{\bar{\mathbf v}^{(i)}\otimes\phi(\mathbf k^{(i)})}_{\text{remove}}. \tag{23}
$$

Substituting Eq. 22 for $\mathbf v^{(i)}_{\text{new}}$ and simplifying the write−remove difference:

$$
\mathbf v^{(i)}_{\text{new}} - \bar{\mathbf v}^{(i)} = \beta^{(i)}\mathbf v^{(i)}+(1-\beta^{(i)})\bar{\mathbf v}^{(i)} - \bar{\mathbf v}^{(i)} = \beta^{(i)}\big(\mathbf v^{(i)}-\bar{\mathbf v}^{(i)}\big),
$$
$$
\boxed{\;\mathbf W^{(i)} = \mathbf W^{(i-1)} + \beta^{(i)}\big(\mathbf v^{(i)}-\bar{\mathbf v}^{(i)}\big)\otimes\phi(\mathbf k^{(i)})\;}, \qquad \mathbf y^{(i)}=\mathbf W^{(i)}\phi(\mathbf q^{(i)}). \tag{24–25}
$$

Equation 24 is exactly the **Widrow–Hoff delta rule** $\Delta\mathbf W\propto(\text{target}-\text{prediction})\otimes\text{input}$, with $\beta^{(i)}$ a *self-invented, input-dependent learning rate*. The FWP therefore learns to correct its own key→value map online. (The pure sum rule of Eq. 17 is the special case $\bar{\mathbf v}^{(i)}\equiv 0$, $\beta^{(i)}\equiv1$.) The paper connects this to a broader theme (elaborated further in Mittal et al., §14): the in-context state update is itself a learned optimization step.

### 2.6 Normalization variants

- **Attention normalization** (Eqs. 26–28): keep the accumulator $\mathbf z^{(i)}=\mathbf z^{(i-1)}+\phi(\mathbf k^{(i)})$ and normalize both the read-out $\bar{\mathbf v}^{(i)}$ and output $\mathbf y^{(i)}$ by $\mathbf z^{(i)}\cdot\phi(\cdot)$. Drawbacks: $\mathbf z^{(i)}$ grows unboundedly (instability), and it does not balance write vs. remove for the delta rule.
- **Sum normalization** (Eq. 29): instead normalize each feature vector by its component sum, $\phi'(\mathbf q^{(i)})=\phi(\mathbf q^{(i)})/\sum_{j=1}^{d_{\text{dot}}}\phi(\mathbf q^{(i)})_j$, applied to both keys and queries before Eqs. 20–25. Intuition: if a vector's components sum to 1, then $\mathbf W\phi'$ is a convex "attention over the columns of $\mathbf W$." Empirically sum normalization alone suffices; extra attention normalization *hurts* language-modeling perplexity, and for unbounded-context DeltaNet the attention accumulator must be removed (it blows up).

### 2.7 DPFP feature map (Eqs. 33–37)

Two failure modes motivate a new $\phi$: Katharopoulos' $\phi=\text{ELU}(x)+1$ keeps $d_{\text{dot}}=d_{\text{key}}$ (no capacity gain); Performer/FAVOR+ raises capacity but via *random* features (sampling variance). The **Deterministic Parameter-Free Projection (DPFP)** raises $d_{\text{dot}}$ deterministically by taking products of rectified coordinate pairs so that inputs falling in different orthants map to orthogonal supports. In $\mathbb R^2\to\mathbb R^4$ with $r(a)=\max(0,a)$:

$$
\phi_1(\mathbf k)=r(k_1)r(k_2),\ \phi_2(\mathbf k)=r(-k_1)r(k_2),\ \phi_3(\mathbf k)=r(k_1)r(-k_2),\ \phi_4(\mathbf k)=r(-k_1)r(-k_2). \tag{33–36}
$$

Each input has exactly one nonzero component ⇒ inputs in different quadrants are orthogonal in the projected space. General form for $\mathbf k\in\mathbb R^{d_{\text{key}}}$, index $i\in[1,2d_{\text{key}}]$, capacity knob $\nu\in\{1,\dots,2d_{\text{key}}-1\}$:

$$
\phi_{i\nu}(\mathbf k)=r\big(\big[\begin{smallmatrix}\mathbf k\\-\mathbf k\end{smallmatrix}\big]\big)_i\; r\big(\big[\begin{smallmatrix}\mathbf k\\-\mathbf k\end{smallmatrix}\big]\big)_{i+\nu}, \tag{37}
$$

giving codomain dimension $d_{\text{dot}}=2d_{\text{key}}\nu$. It is embarrassingly parallel and needs no learned parameters or random sampling; its sparsity/orthogonality goal aligns with the empirical observation that ReLU beats exp in FAVOR+.

### 2.8 Empirical highlights

- **Synthetic retrieval (Setting 1, capacity):** with $d_{\text{key}}=64$, plain Linear-Attention starts erroring at ~60 associations; DPFP-$\nu$ errors at its $d_{\text{dot}}=128/256/384$ limits; FAVOR+ never reaches zero loss; softmax is best (infinite capacity) but struggles past 500 keys. Directly validates the $\le d_{\text{dot}}$ capacity theory.
- **Setting 2 (editing):** keys re-assigned (sampled with replacement, $L=2S$); the delta rule beats the sum rule and prior FWP variants — memory editing works.
- **WMT14 En-De:** DPFP beats Linear Transformer and Performer at small $d_{\text{dot}}$ (good simplicity/performance trade-off).
- **WikiText-103:** DeltaNet (delta rule) beats sum-rule linear transformer and Performer at matched parameters; sum normalization required, extra attention normalization and absolute positional encoding both unnecessary/harmful. **Unbounded context** (Table 4): carrying the fast-weight memory across segments, DeltaNet keeps perplexity ~27–29 with a tiny 0.13M state, while the naive sum-rule linear transformer diverges (>260) — a striking demonstration of editable finite memory.

## Appendix: Section-by-Section Backbone

- **Abstract.** Formal equivalence of linearized self-attention and early-'90s fast-weight controllers; slow net learns by GD to program fast weights via additive outer products of self-invented keys/values. Infer a capacity limit; replace additive rule with a delta-rule instruction (dynamic learning rate); propose a new linearization kernel (DPFP). Experiments on synthetic retrieval, WMT14 MT, WikiText-103 LM.
- **§1 Introduction.** Transformer self-attention is quadratic time / linear memory. Linear transformers linearize softmax → constant memory, linear time. Claim: this family ≡ FWPs (Schmidhuber 1991/92/93) up to normalization. Derive capacity limit + overcapacity regime; introduce delta-rule update + dynamic learning rate; propose new φ.
- **§2 Background on FWPs.** Fast weights = input-dependent, variable weights (von der Malsburg synaptic modulation; Hinton & Plaut; Feldman dynamic connections). 1991 two-net system: slow net programs fast net via outer-product instructions (Eqs. 1–3); key-value associative memory; TPR connection (Smolensky 1990); modern revivals: hypernetworks, dynamic plasticity, dynamic convolution, lambda nets, meta-learning.
- **§3 Relation to Transformers.** §3.1 remove softmax → self-attention is an FWP (Eqs. 4–11); $\mathbf W^{(i)}=\sum_j\mathbf v^{(j)}\otimes\mathbf k^{(j)}$. §3.2 kernel linearization (Eqs. 12–19); fast weight $\mathbf W^{(i)}$ + accumulator $\mathbf z^{(i)}$; core of linear transformers = outer-product FWP.
- **§4 Analysing & improving.** §4.1 capacity: $\le d_{\text{dot}}$ orthogonal keys; overcapacity when $L>d_{\text{dot}}$; TPR theory (Smolensky Thm 3.1/3.3); FWPs learn the vectors (vs. hand-wired TPRs). §4.2 delta-rule update (Eqs. 20–25): read-modify-write with dynamic write-strength $\beta^{(i)}$; = Widrow-Hoff delta rule with learned learning rate; normalization: attention-norm (Eqs. 26–28, drawbacks) vs. sum-norm (Eq. 29, preferred).
- **§5 Linear attention functions.** §5.1 desirable φ properties (positive codomain, capacity via $d_{\text{dot}}$). §5.2 Katharopoulos ELU+1 (Eq. 30, no capacity gain). §5.3 FAVOR+ random features (Eqs. 31–32, sampling variance, $d_{\text{dot}}=2m$). §5.4 DPFP (Eqs. 33–37): deterministic, parameter-free, near-orthogonal up-projection, $d_{\text{dot}}=2d_{\text{key}}\nu$.
- **§6 Experiments.** §6.1.1 capacity confirmed on synthetic retrieval; §6.1.2 delta rule beats sum rule on editing; §6.2 WMT14 (DPFP competitive at small $d_{\text{dot}}$); §6.3 WikiText-103 (DeltaNet best; sum-norm needed; no pos-enc/attn-norm; unbounded-context DeltaNet vs. Transformer-XL, tiny state size).
- **§7 Conclusion.** Linear attention ≡ 1991 FWP; capacity analysis; delta-rule editable memory with learned learning rate; new linearization kernel; opens design space for finite-memory transformers.

---
