---
title: "Attention Is All You Need"
authors: ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar", "Jakob Uszkoreit", "Llion Jones", "Aidan N. Gomez", "Łukasz Kaiser", "Illia Polosukhin"]
year: 2017
venue: "NeurIPS 2017 (31st Conference on Neural Information Processing Systems)"
slug: vaswani_2017_attention
source_pdf: "docs/project/references/FiLM/sources/Vaswani et al. 2017 - Attention is all you need.pdf"
topic: FiLM
---

## Plain-English Entry Point

This paper introduces the **Transformer**, a sequence-to-sequence neural network architecture built entirely from **attention** mechanisms — discarding the recurrent (RNN / LSTM) and convolutional layers that previously dominated machine translation. The core operation is **self-attention**: every position in a sequence directly compares itself to every other position and forms a weighted blend of their representations. The comparisons are done by projecting each input token into three vectors — a **query** (Q), a **key** (K), and a **value** (V). The query of position *i* is matched against the keys of every position via dot product, the resulting compatibilities are softmaxed into weights, and those weights mix the value vectors into the output at *i*.

The authors stack two refinements on this primitive: (i) **scaled** dot-product attention — divide dot products by $\sqrt{d_k}$ so softmax does not saturate as dimensions grow; (ii) **multi-head attention** — run several attention operations in parallel on independently projected subspaces, then concatenate. Because the model lacks any built-in notion of sequence order, **sinusoidal positional encodings** are added to the input embeddings. The Transformer trains in a fraction of the time of prior models and sets new state-of-the-art BLEU scores on WMT 2014 English-German (28.4) and English-French (41.0). For this project, the relevance is conceptual: attention is an alternative **conditioning primitive** to FiLM — both modulate features based on context, but attention does so by reweighting positions while FiLM does so by per-feature affine scaling.

## Section-Ordered Backbone

**Abstract.** Proposes the Transformer, an attention-only encoder-decoder architecture. Beats existing models in quality, parallelism, and training cost on WMT 2014 EN-DE (28.4 BLEU) and EN-FR (41.0 BLEU).

**1. Introduction.** Recurrent models factor computation along symbol positions, creating sequential dependencies $h_t = f(h_{t-1}, x_t)$ that preclude parallelization within examples. Attention previously appeared only as an add-on to RNNs. The Transformer eliminates recurrence and convolution entirely, relying solely on attention to draw global dependencies.

**2. Background.** Compares with ByteNet, ConvS2S, and Extended Neural GPU — all reduce sequential computation via convolutions, but the number of operations relating two positions still grows with distance (linearly for ConvS2S, logarithmically for ByteNet). The Transformer reduces this to constant. Counteracts the resulting reduced effective resolution with multi-head attention. Self-attention previously used in reading comprehension, summarization, entailment; the Transformer is the first transduction model relying *entirely* on self-attention.

**3. Model Architecture.** Encoder-decoder structure. Encoder maps $(x_1, \ldots, x_n)$ to continuous representations $z = (z_1, \ldots, z_n)$. Decoder generates outputs auto-regressively. Both stacks use $N=6$ identical layers with residual connections and LayerNorm. All sub-layers and embeddings produce $d_{\text{model}} = 512$ dimensional outputs.

- **3.1 Encoder and Decoder Stacks.** Encoder layer = (multi-head self-attention) + (position-wise FFN); each wrapped with residual + LayerNorm. Decoder adds a third sub-layer that attends to the encoder output (encoder-decoder attention), and masks future positions in its self-attention to preserve auto-regressivity.
- **3.2 Attention.** Attention as a query-key-value map: output is a weighted sum of values where weights come from compatibility of query with keys.
  - **3.2.1 Scaled Dot-Product Attention.** $\text{Attention}(Q, K, V) = \text{softmax}(QK^\top/\sqrt{d_k}) V$. Scaling by $\sqrt{d_k}$ prevents softmax saturation at large $d_k$.
  - **3.2.2 Multi-Head Attention.** $h = 8$ heads with $d_k = d_v = d_{\text{model}}/h = 64$. Each head projects Q/K/V to a smaller subspace, runs attention, then results are concatenated and projected.
  - **3.2.3 Applications.** Three uses: (1) encoder-decoder attention (decoder queries, encoder keys/values); (2) encoder self-attention; (3) decoder masked self-attention.
- **3.3 Position-wise FFN.** $\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$, $d_{ff} = 2048$. Applied identically at every position but with different parameters per layer.
- **3.4 Embeddings and Softmax.** Shared input/output embedding and pre-softmax projection weights; embeddings scaled by $\sqrt{d_{\text{model}}}$.
- **3.5 Positional Encoding.** Sinusoidal: $PE_{(pos, 2i)} = \sin(pos/10000^{2i/d_{\text{model}}})$, $PE_{(pos, 2i+1)} = \cos(\cdot)$. Wavelengths form geometric progression $2\pi$ to $10000 \cdot 2\pi$. Chosen because relative offsets are linear functions of $PE_{pos}$ and to allow extrapolation to unseen sequence lengths.

**4. Why Self-Attention.** Three desiderata: (i) total computational complexity per layer; (ii) parallelism (minimum sequential operations); (iii) path length between long-range dependencies. Self-attention: $O(n^2 \cdot d)$, $O(1)$ sequential ops, $O(1)$ path length — vs recurrent $O(n \cdot d^2)$, $O(n)$, $O(n)$. Faster than recurrent when $n < d$ (typical for NMT). Bonus: attention heads are interpretable and learn syntactic/semantic structure.

**5. Training.** WMT 2014 EN-DE (4.5M pairs, 37K BPE vocab) and EN-FR (36M sentences, 32K word-piece). 8 P100 GPUs; base model = 100K steps / 12 h; big = 300K steps / 3.5 days. Adam with $\beta_1 = 0.9$, $\beta_2 = 0.98$, $\epsilon = 10^{-9}$. Learning rate schedule: $\text{lrate} = d_{\text{model}}^{-0.5} \cdot \min(\text{step}^{-0.5}, \text{step} \cdot \text{warmup}^{-1.5})$ with 4000 warmup steps. Regularization: residual dropout ($P_{\text{drop}} = 0.1$), label smoothing ($\epsilon_{ls} = 0.1$).

**6. Results.** Big Transformer: 28.4 BLEU EN-DE, 41.0 BLEU EN-FR (single-model SOTA). Ablations in Table 3: reducing $d_k$ hurts (compatibility is non-trivial); learned vs sinusoidal positional encodings give nearly identical results; dropout important; more parameters helps.

**7. Conclusion.** First sequence transduction model based entirely on attention. Plans to extend to other modalities (images, audio, video) and to investigate restricted local attention for long sequences.

## Phase 1 — Undergraduate-Level Synthesis

The key idea: instead of processing a sentence one word at a time (as RNNs do), let every word *look at* every other word in a single parallel operation. The "looking" is implemented by giving each word three vectors — what it is **asking about** (query), what it can **answer** (key), and what **information** it carries (value). The answer to each word's question is a blended view of every other word's information, where the blend is determined by how well the asker's question matches each other word's offered answer.

Three engineering choices make this practical:

1. **Scaling** — when vectors are high-dimensional, raw dot products can blow up and saturate softmax (one token gets ~all the attention; everyone else gets ~0 weight, and gradients vanish). Dividing by $\sqrt{d_k}$ keeps the scale tame.
2. **Multi-head** — instead of one attention computation in $d_{\text{model}}$-dim, do $h=8$ parallel attentions in $d_{\text{model}}/h$-dim subspaces. Different heads can attend to different relations (syntactic, semantic, positional) simultaneously without averaging them away.
3. **Positional encoding** — attention itself is permutation-equivariant: shuffle the input tokens and the output shuffles identically. To inject word order, sinusoidal patterns are *added* to each input embedding before any attention is computed.

The headline result is that this purely-attentional model beats RNN- and CNN-based machine translation systems while training far faster. The takeaway beyond translation: attention is a general-purpose mechanism for **contextual conditioning**, and dropping the sequential prior allows full GPU parallelism. For this project, the contrast with FiLM is the most useful framing: FiLM conditions one stream on another by **affine scaling per channel**; attention conditions one stream on another by **weighted aggregation across positions**. Both are conditioning primitives; they differ in what they treat as the unit of modulation.

## Phase 2 — Graduate-Level Deep Dive

### Scaled Dot-Product Attention

Given queries $Q \in \mathbb{R}^{n_q \times d_k}$, keys $K \in \mathbb{R}^{n_k \times d_k}$, and values $V \in \mathbb{R}^{n_k \times d_v}$, scaled dot-product attention is

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V. \tag{1}
$$

The matrix $QK^\top \in \mathbb{R}^{n_q \times n_k}$ holds all pairwise compatibility scores; row $i$ is softmaxed to a probability distribution over key positions; the result then weighted-averages the values. Computationally, this is two matmuls and a softmax — $O(n^2 d)$ for self-attention where $n = n_q = n_k$.

**Why scale by $\sqrt{d_k}$?** Assume each component of $q$ and $k$ is independent with mean 0 and variance 1. Then their dot product

$$
q \cdot k = \sum_{i=1}^{d_k} q_i k_i
$$

has $\mathbb{E}[q \cdot k] = 0$ and $\text{Var}(q \cdot k) = d_k$ (sum of $d_k$ independent unit-variance terms). The standard deviation grows as $\sqrt{d_k}$. Large pre-softmax logits push softmax into saturated regions where one entry dominates and gradients with respect to all other entries vanish:

$$
\frac{\partial \, \text{softmax}(z)_i}{\partial z_j} = \text{softmax}(z)_i (\delta_{ij} - \text{softmax}(z)_j) \xrightarrow{\text{softmax}(z)_i \to 1} 0 \text{ for } j \neq i.
$$

Dividing by $\sqrt{d_k}$ normalizes the logits to roughly unit variance, keeping softmax in its high-gradient regime.

### Multi-Head Attention

Rather than running one $d_{\text{model}}$-dim attention, project queries, keys, and values $h$ times in parallel to lower-dim subspaces and run attention independently per head:

$$
\begin{aligned}
\text{head}_i &= \text{Attention}(QW_i^Q,\ KW_i^K,\ VW_i^V), \\
\text{MultiHead}(Q, K, V) &= \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\,W^O,
\end{aligned} \tag{2}
$$

with $W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, $W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}$. With $h=8$ and $d_k = d_v = d_{\text{model}}/h = 64$, total compute is comparable to single-head attention at full $d_{\text{model}}$. The motivation: averaging in a single high-dim attention destroys information; per-subspace attention lets distinct heads specialize. Ablations (Table 3 rows A) show single-head attention is 0.9 BLEU worse than $h=8$; quality also drops with too many heads.

### Positional Encoding

Attention is permutation-equivariant, so position must be injected externally. The chosen encoding is sinusoidal:

$$
\begin{aligned}
PE_{(pos, 2i)} &= \sin\!\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right), \\
PE_{(pos, 2i+1)} &= \cos\!\left( \frac{pos}{10000^{2i/d_{\text{model}}}} \right).
\end{aligned} \tag{3}
$$

Dimension $i$ corresponds to a sinusoid with angular frequency $\omega_i = 10000^{-2i/d_{\text{model}}}$; wavelengths form a geometric progression from $2\pi$ (at $i=0$) to $10000 \cdot 2\pi$ (at $i = d_{\text{model}}/2$). The encoding is added (not concatenated) to the input embedding.

**Why this form?** For any fixed offset $k$, $PE_{pos+k}$ can be written as a linear function of $PE_{pos}$:

$$
\begin{aligned}
\sin(\omega_i (pos+k)) &= \sin(\omega_i\, pos)\cos(\omega_i k) + \cos(\omega_i\, pos)\sin(\omega_i k), \\
\cos(\omega_i (pos+k)) &= \cos(\omega_i\, pos)\cos(\omega_i k) - \sin(\omega_i\, pos)\sin(\omega_i k),
\end{aligned}
$$

i.e., the transformation $PE_{pos} \mapsto PE_{pos+k}$ is a fixed rotation per-frequency, with rotation angles depending only on the offset $k$ and not on $pos$. This makes relative positions easy for the model to learn via linear projections inside the attention mechanism. The geometric frequency progression covers both fine-grained adjacent-position distinctions and very coarse long-range distinctions. Empirically, learned positional embeddings perform nearly identically (Table 3 row E), but sinusoidal encodings extrapolate cleanly to sequence lengths unseen during training.

### Position-wise Feed-Forward Network

Between attention sub-layers, each position is independently processed through

$$
\text{FFN}(x) = \max(0,\ xW_1 + b_1)\, W_2 + b_2, \tag{4}
$$

with $d_{\text{model}} = 512$ and inner $d_{ff} = 2048$ (a 4× expansion). Same parameters across positions in a layer, different across layers. Equivalently described as two 1×1 convolutions.

### Complexity Comparison

For self-attention vs recurrent vs convolutional layers operating on a length-$n$, dim-$d$ sequence:

| Layer | Complexity per layer | Sequential ops | Max path length |
|---|---|---|---|
| Self-attention | $O(n^2 \cdot d)$ | $O(1)$ | $O(1)$ |
| Recurrent | $O(n \cdot d^2)$ | $O(n)$ | $O(n)$ |
| Convolutional | $O(k \cdot n \cdot d^2)$ | $O(1)$ | $O(\log_k n)$ |
| Self-attention (restricted, neighborhood $r$) | $O(r \cdot n \cdot d)$ | $O(1)$ | $O(n/r)$ |

Self-attention is faster than recurrent when $n < d$ (typical in NMT, where $n \approx 30$ and $d \approx 512$) and gives every position a one-hop path to every other position.

### Learning Rate Schedule

$$
\text{lrate} = d_{\text{model}}^{-0.5} \cdot \min\!\left(\text{step}^{-0.5},\ \text{step} \cdot \text{warmup}^{-1.5}\right), \tag{5}
$$

with $\text{warmup} = 4000$. Linear warmup followed by inverse-square-root decay; the $d_{\text{model}}^{-0.5}$ prefactor scales the schedule down for larger models.

## Connections to the Project

- **Attention vs FiLM (contrast).** Attention and FiLM are **different conditioning primitives in the same design space**. FiLM (`perez_2018_film.md` in Batch 1) conditions a feature map $F$ on context $c$ by per-channel affine modulation: $\text{FiLM}(F | c) = \gamma(c) \odot F + \beta(c)$, where $\gamma, \beta$ are produced by a conditioning network. Attention conditions one stream on another by **softmax-weighted aggregation across positions**: each output position is a convex combination of value vectors from the conditioning stream. Concretely:
  - **Granularity of modulation:** FiLM modulates *channels* (one $\gamma_c, \beta_c$ per feature map channel); attention modulates *positions* (one weight per source token, shared across all feature dimensions of that token's value).
  - **Parameter cost of conditioning:** FiLM costs $O(C)$ per modulation; cross-attention costs $O(n_q \cdot n_k)$ in compute and produces $n_q \cdot n_k$ soft "attention" coefficients.
  - **Inductive bias:** FiLM assumes the context affects *what features to amplify or suppress*; attention assumes the context provides *content to read from*. For task-conditioned RL (the project's setting), FiLM is a lighter modulation interface when the conditioning signal is low-dimensional (a task ID or a few scalar physiology variables); attention becomes attractive when the conditioning signal is itself a structured sequence.
- **Self-attention as a way to share information across sequence positions in RL.** If we extend a recurrent policy with self-attention over recent observation history, every step can directly look at any past step rather than relying on the GRU/LSTM to compress history. This is a structural alternative to FiLM-style conditioning that the project may consider if context spans variable temporal extent.
- **Multi-head decomposition resembles the "specialized modulators" framing.** Multi-head attention's per-subspace specialization echoes the project's neuromodulator-as-hyperparameter direction memo, where different modulators (ACh / NE / DA / 5-HT) can be thought of as specialized conditioning channels that act on different feature subspaces of the policy. The Transformer's empirical finding that 8 heads outperform 1 (and that quality drops with too many) is a useful prior for the question "how many modulators are useful?".
- **Recommendation for follow-up.** If the project moves to attention-based modulation (rather than FiLM), the math-reviewer and senior-developer would need to evaluate (a) the $O(n^2)$ cost during RL rollouts and (b) the positional-encoding choice (sinusoidal vs learned) — both are non-trivial design decisions that this paper enumerates the trade-offs of.
