> **Per-paper review — in-context-learning corpus, paper 11 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§11); content is identical. Manifest: [[in_context_learning_sources]].

# 11. Dai et al. 2023 — Why Can GPT Learn In-Context? (Language Models Secretly Perform Gradient Descent as Meta-Optimizers)

**PDF:** `docs/project/references/in_context_learning/sources/Dai et al. 2023 - Why Can GPT Learn In-Context (Secretly Gradient Descent).pdf`
**Venue:** ACL 2023 Findings. **Authors:** Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, Furu Wei (Peking Univ. / Tsinghua / Microsoft Research).

## Phase 1: Foundational Overview (Undergraduate-Level)

**The plain-language question.** Akyürek et al. work with a toy linear-regression transformer trained from scratch. Dai et al. ask the same "how does ICL work" question but for **real, off-the-shelf GPT models on real NLP tasks** (sentiment, topic classification, natural-language inference). Their answer: **the attention mechanism has a hidden "dual form" that is mathematically equivalent to gradient descent**, so a frozen GPT reading demonstration examples is effectively *fine-tuning itself without a backward pass*. They call the network a **meta-optimizer**: it produces **meta-gradients** from the demonstrations by ordinary forward computation, and applies them through attention.

**The core analogy.**
- **Fine-tuning:** compute gradients by back-propagation, add $\Delta W_{\mathrm{FT}}$ to the weights.
- **In-context learning:** the attention over demonstration tokens produces an implicit weight update $\Delta W_{\mathrm{ICL}}$ that plays the same role — but it is built by *forward* computation, no back-prop.

So ICL is "**implicit fine-tuning**." The two share a "dual view" of gradient descent.

**How they check it.** On six classification datasets with GPT-1.3B and GPT-2.7B, they compare ICL against a deliberately matched fine-tuning baseline (same examples, one step each, same order, only key/value projections updated) along four axes:
1. **Predictions:** ICL correctly reproduces ~85–89% of the corrections fine-tuning makes over zero-shot (their "Rec2FTP" metric).
2. **Attention-output representations** move in the *same direction* as fine-tuning's (their "SimAOU" cosine ≈ 0.19–0.23, versus ≈ 0 for random updates).
3. **Attention weights to the query** become more like a fine-tuned model's than an un-tuned model's (their "SimAM").
4. **Attention to training tokens** is rank-correlated with fine-tuning's (their "Kendall" ≈ 0.19–0.21, versus ≈ 0 for random).

**A design payoff.** If attention values are "meta-gradients," you can borrow tricks from real optimizers. They add **momentum** (an exponential moving average of attention value vectors) to attention and get consistent gains — lower language-model perplexity and +2.8 average accuracy on ICL tasks — which they treat as further evidence the meta-optimizer view is real and useful.

**Initial takeaway.** This paper is the "real-model, real-task" complement to Akyürek. It does not prove an exact algorithm (it uses an approximation — linear attention without softmax), but it gives a clean equation showing attention ≡ a gradient-descent weight update, plus behavioral evidence that ICL and fine-tuning line up. The momentum-attention result shows the analogy can *guide architecture design*.

## Phase 2: Graduate-Level Deep Dive

The algorithm-implementation claim here is the **duality between (linear) attention and a gradient-descent weight update**. This is more of a formal *analogy* than Akyürek's exact construction or Bai's approximation-with-error-bound, but the derivation is the crux.

### The dual form of a linearly-updated linear layer (Irie et al. 2022 lineage)

Start from a linear layer whose weights receive a gradient-descent update. Let $W_0, \Delta W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$ be the initial and update matrices, $x \in \mathbb{R}^{d_{\mathrm{in}}}$ the input:

$$
\mathcal{F}(x) = (W_0 + \Delta W)\,x. \tag{7}
$$

In back-propagation, $\Delta W$ is the sum of outer products of historical inputs $x_i'$ and their **error signals** $e_i = -\gamma\,g_i$ (learning rate $\gamma$ times the output gradient $g_i$):

$$
\Delta W = \sum_i e_i \otimes x_i'. \tag{8}
$$

**Derivation of the dual form.** Substitute (8) into (7):
$$
\mathcal{F}(x) = W_0 x + \Delta W x
= W_0 x + \Big(\sum_i e_i \otimes x_i'\Big) x
= W_0 x + \sum_i e_i \big(x_i'^\top x\big)
= W_0 x + \mathrm{LinearAttn}(E, X', x). \tag{9}
$$

The last step uses $(e_i \otimes x_i')\,x = e_i (x_i'^\top x)$: the outer product acting on $x$ collapses to the error vector $e_i$ scaled by the inner product $x_i'^\top x$. **Reading the result as attention:** the historical errors $E$ are the *values*, the historical inputs $X'$ are the *keys*, and the current input $x$ is the *query*. So "a linear layer trained by GD" $\equiv$ "$W_0 x$ plus a linear-attention lookup." This is the identity the whole paper hangs on.

### Applying the duality to Transformer attention (Section 3.1)

Let $x$ be the query-token representation and $q = W_Q x$ the attention query. In the ICL setting, with $X'$ the demonstration-token representations and $X$ the preceding query tokens, standard attention is

$$
\mathcal{F}_{\mathrm{ICL}}(q) = \mathrm{Attn}(V, K, q) = W_V [X'; X]\,\mathrm{softmax}\!\left(\frac{(W_K [X'; X])^\top q}{\sqrt{d}}\right). \tag{10}
$$

**Approximation step (the paper's main caveat).** Drop the softmax and scaling factor to get *relaxed linear attention*:

$$
\tilde{\mathcal{F}}_{\mathrm{ICL}}(q) \approx W_V [X'; X]\,(W_K [X'; X])^\top q
= \underbrace{W_V X (W_K X)^\top}_{\text{query part}} q + \underbrace{W_V X' (W_K X')^\top}_{\text{demonstration part}} q. \tag{11}
$$

Define $W_{\mathrm{ZSL}} := W_V X (W_K X)^\top$ — the zero-shot-learning weights, i.e. what attention would produce with *no* demonstrations. Then, running the duality (9) *in reverse* on the demonstration part:

$$
\tilde{\mathcal{F}}_{\mathrm{ICL}}(q)
= W_{\mathrm{ZSL}} q + \sum_i \big(W_V x_i'\big)\big(W_K x_i'\big)^\top q
= W_{\mathrm{ZSL}} q + \sum_i \big((W_V x_i') \otimes (W_K x_i')\big) q
= \big(W_{\mathrm{ZSL}} + \Delta W_{\mathrm{ICL}}\big) q. \tag{12}
$$

**Interpretation.** The demonstration tokens contribute a weight update $\Delta W_{\mathrm{ICL}} = \sum_i (W_V x_i') \otimes (W_K x_i')$ acting on the initial ZSL weights — exactly the structure of (7)–(8). Here $W_V X'$ plays the role of the meta-gradient values $E$, and $W_K X'$ the role of the historical inputs $X'$. So **attention to demonstrations = an implicit parameter update built by forward computation.**

The three-line summary the paper repeats: (1) pretrained GPT is a meta-optimizer; (2) it produces meta-gradients from demonstrations via forward computation; (3) attention applies those meta-gradients to build the ICL model.

### Comparison with an explicit fine-tuning update (Section 3.2)

A fine-tuning update to only the key/value projections gives, in the same relaxed-linear form,

$$
\tilde{\mathcal{F}}_{\mathrm{FT}}(q) = (W_V + \Delta W_V)\,X X^\top (W_K + \Delta W_K)^\top q = (W_{\mathrm{ZSL}} + \Delta W_{\mathrm{FT}})\,q. \tag{13}
$$

Comparing (12) and (13): both add an update to $W_{\mathrm{ZSL}}$; $\Delta W_{\mathrm{ICL}}$ comes from forward computation, $\Delta W_{\mathrm{FT}}$ from back-propagation. The paper enumerates four shared properties justifying "ICL = implicit fine-tuning": **both perform gradient descent**; **same training information** (both driven by the demonstration examples); **same causal order** (decoder-only attention + one-epoch same-order fine-tuning both prevent later examples affecting earlier); **both act on attention** (keys/values only).

### Empirical alignment metrics (Section 4)

- **Rec2FTP** (recall to fine-tuning prediction): among query examples fine-tuning fixes over ZSL, the fraction ICL also fixes: $\dfrac{N_{(\mathrm{FT}>\mathrm{ZSL}) \wedge (\mathrm{ICL}>\mathrm{ZSL})}}{N_{\mathrm{FT}>\mathrm{ZSL}}}$. Averages 85.6% (1.3B) / 89.4% (2.7B).
- **SimAOU** (similarity of attention-output updates): cosine between $h^{(l)}_{\mathrm{ICL}} - h^{(l)}_{\mathrm{ZSL}}$ and $h^{(l)}_{\mathrm{FT}} - h^{(l)}_{\mathrm{ZSL}}$. ~0.19–0.23 vs. ~0 for random updates.
- **SimAM** (similarity of attention maps to the query): ICL's pre-softmax attention weights are closer to the *post*-fine-tuning model's than to the *pre*-fine-tuning model's.
- **Kendall (ICL, FT)** rank correlation of attention to training tokens: $\dfrac{P_c - P_d}{N(N-1)/2}$ ≈ 0.19–0.21 vs. ~0 random. ($P_c, P_d$ = concordant/discordant pairs.)

### Momentum-based attention (Section 5)

Gradient descent with momentum averages past gradients:

$$
\Theta_t = \Theta_{t-1} - \gamma \sum_{i=1}^{t-1} \eta^{t-i} \nabla f_{\Theta_i}, \tag{14}
$$

with $\eta \in (0,1)$. Since attention **values** are the meta-gradients, apply an exponential moving average (EMA) to them:

$$
\mathrm{MoAttn}(V, K, q_t) = \mathrm{Attn}(V, K, q_t) + \mathrm{EMA}(V)
= V\,\mathrm{softmax}\!\Big(\tfrac{K^\top q_t}{\sqrt d}\Big) + \sum_{i=1}^{t-1} \eta^{t-i} v_i,
$$

where $v_i$ is the $i$-th value vector. Results: consistent perplexity improvement in language modeling (e.g. 15.14 → 15.02 at length 1024) and +2.8 average ICL accuracy across six datasets. The paper frames this as the analogy *paying rent* — the meta-optimizer view is not just interpretive but design-generative.

**Stated limitations.** The duality is derived for *relaxed linear attention* (no softmax); full softmax attention "may be more complex." Analysis is limited to ≤2.7B models and classification tasks.

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** ICL as meta-optimization; the dual form between attention and gradient descent; ICL = implicit fine-tuning. Four contributions: (i) the duality + meta-optimization view, (ii) ICL↔fine-tuning connection, (iii) empirical evidence, (iv) momentum-attention design.
- **§2 Background.** §2.1 ICL for classification with GPT: predict $\hat y = \arg\max_{y_j} P_M(y_j \mid C, x)$ over a candidate answer set, with template formatting $T(x,y)$ and logits $l_j = M(I) \cdot e_{y_j}$ (Eqs. 1–6). §2.2 the dual form of a GD-trained linear layer as linear attention (Eqs. 7–9), citing Aizerman 1964 and Irie et al. 2022.
- **§3 Understanding ICL as implicit fine-tuning.** §3.1 relaxed-linear approximation of attention (Eq. 10→11), definition of $W_{\mathrm{ZSL}}$, reverse-duality to $\Delta W_{\mathrm{ICL}}$ (Eq. 12), meta-optimization summary. §3.2 the matched fine-tuning baseline (Eq. 13) and four shared properties.
- **§4 Experiments.** §4.1 settings (GPT-1.3B / 2.7B from fairseq; 32 demonstrations; SGD one-epoch fine-tuning on the same examples). §4.2 six datasets (SST2/SST5/MR/Subj/AGNews/CB) with ZSL/FT/ICL accuracies (Table 1). §4.3 Rec2FTP (Table 2). §4.4 SimAOU (Table 3). §4.5 SimAM (Table 4). §4.6 Kendall correlation of attention to training tokens (Table 5).
- **§5 Momentum-based attention.** GD-with-momentum (Eq. 14); EMA over attention values; perplexity gains (Table 6) and ICL accuracy gains (Table 7).
- **§6 Related work.** Contrasts with Xie et al. (implicit Bayesian inference), Olsson et al. (induction heads), Garg et al. / Akyürek et al. / von Oswald et al. (GD in trained-from-scratch regression transformers). Positions this work as the first to analyze *off-the-shelf* GPTs on *real* tasks.
- **§7 Conclusion + Limitations.** Duality → meta-optimization → implicit fine-tuning; momentum design. Limits: relaxed-linear (softmax-free) analysis; ≤2.7B; classification only.
- **Appendices A–C.** Templates and candidate answer sets; random-seed and learning-rate grids; from-scratch LM hyperparameters (350M, 24 layers).

---
