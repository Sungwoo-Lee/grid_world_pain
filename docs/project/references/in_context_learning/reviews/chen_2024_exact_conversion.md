> **Per-paper review — in-context-learning corpus, paper 38 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§38); content is identical. Manifest: [[in_context_learning_sources]].

# 38. Chen et al. 2024 — Exact Conversion of In-Context Learning to Model Weights

**PDF:** `docs/project/references/in_context_learning/sources/Chen et al. 2024 - Exact Conversion of In-Context Learning to Model Weights.pdf`
**Venue:** ICML 2024. **Authors:** Brian K Chen, Tianyang Hu, Hui Jin, Hwee Kuan Lee, Kenji Kawaguchi (NUS / A*STAR / Huawei Noah's Ark Lab).

## Phase 1: Foundational Overview (Undergraduate-Level)

**The question in plain terms.** In-context learning (ICL) lets a language model adapt to a task by putting demonstration examples in the prompt — no fine-tuning, no weight updates. But the effect is *temporary*: it applies only to that one prompt and is thrown away afterward. Can we take those demonstration tokens and **bake them permanently into the model's weights** — exactly, not approximately, and cheaply? For **linearized-attention** transformers, this paper says yes: the whole ICL prompt collapses into a single **bias term** added to one internal matrix.

**The key structural insight.** In a linear/kernel attention layer, the value read for a query is $\phi(Q_i)^\top A$ where $A=\sum_j \phi(K_j)V_j^\top$ is a fixed-size matrix the authors call the **Key-Value matrix**. Adding demonstration (ICL) tokens to the prompt does **not** change the query features $\phi(Q_i)$ — it only *adds more terms to $A$*. So the entire influence of the context is a fixed additive contribution to $A$. Store that contribution as a bias $b_{KV}$ and you have permanently absorbed the context. This is the literal, exact version of "context = weights."

**Key findings.**
1. **Direct conversion is impossible** (you can't just tweak $W_K,W_V$ to absorb the context, because the equation fails at the zero-input prompt).
2. **Bias conversion is exact.** Add a bias to the Key-Value matrix (and, with normalization/RoPE, a second bias $b_D$ to the denominator). A closed-form update of these biases exactly reproduces the ICL model's outputs for *all* future inputs. Algorithm **ICLCA** does this in one forward pass over the ICL tokens — no gradient descent.
3. **Costs ~1% of parameters**, is composable (stack multiple contexts by adding biases) and reversible (drop the bias to recover the original model).
4. **RNN view.** Since linear transformers are RNNs, the bias is exactly a change to the **initial hidden state** $s_0$ — you initialize the recurrence as if the ICL tokens had already been processed.
5. **Approximate conversion for softmax transformers** (ICLAA): first kernel-approximate the softmax (Performers), then apply the same bias. On GPT-2 it roughly halves the logit error induced by dropping the context (16.56% → 9.17%) and the model still "remembers" the context (generates Microsoft after a Bill-Gates context prompt is removed).
6. **Exactness verified:** on linear-attention models the converted-model logits match the ICL model to ~$10^{-6}$–$10^{-7}$ relative error across 205K→1.98B params; on an induction-head task, ICL accuracy 99.95% is preserved exactly by the converted (context-free) model vs. 2.23% without.

**Initial takeaway.** This is the most literal member of the "context ↔ weights" family: it shows a demonstration prompt is *algebraically identical* to a specific weight (bias) edit in linear attention. **Bridge to the project:** it operationalizes the fast-weight / hypernetwork view (papers 1–3) — the outer-product sum $\sum_j\phi(K_j)V_j^\top$ (Schlag's fast weight $\mathbf W^{(i)}$) IS the thing you freeze into a bias, and Schug's "context generates a value network" becomes "context generates a storable weight offset." For a project using FiLM/hypernetwork conditioning, this is the proof-of-concept that a learned context code can be materialized as a permanent, composable, low-cost weight modification.

## Phase 2: Graduate-Level Deep Dive

### 4.1 Attention definitions and the Key-Value matrix

Input tokens $X=[X_1,\dots,X_N]^\top\in\mathbb R^{N\times d_{in}}$. Single-head softmax attention:

$$
O = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_K}}\right)V,\quad Q=XW_Q,\ K=XW_K,\ V=XW_V. \tag{1}
$$

Row-wise / kernel-generalized form (Katharopoulos 2020), with similarity $\text{sim}$ replaced by a kernel via feature map $\phi$:

$$
O_i^\top = \frac{\sum_{j=1}^N \text{sim}(Q_i,K_j)V_j^\top}{\sum_{j=1}^N \text{sim}(Q_i,K_j)}
\ \xrightarrow{\ \text{sim}\to\phi\ }\
O_i^\top = \frac{\phi(Q_i)^\top\sum_{j=1}^N \phi(K_j)V_j^\top}{D_1(Q_i)^\top D_2(X)}. \tag{2–3}
$$

The masked (autoregressive) version sums $j$ only up to $i$:

$$
O_i^\top = \frac{1}{D_1(Q_i)^\top D_2(X)}\,\phi(Q_i)^\top\sum_{j=1}^{i}\phi(K_j)V_j^\top. \tag{5}
$$

**Definition 3.4 — the Key-Value matrix:**

$$
A = \sum_{j=1}^N \phi(K_j)V_j^\top \quad\text{(or }\textstyle\sum_{j=1}^i\text{ when masked)}. \tag{6}
$$

**The pivotal observation:** appending ICL tokens $\{X'_i\}_{i=1}^M$ to the prompt does not change $\phi(Q_i)$ (the query features of the input tokens) — it only appends terms to $A$. Therefore the entire ICL effect is a modification of $A$, and converting $A$ converts the layer. This is the same outer-product accumulator as Schlag's fast weight $\mathbf W^{(i)}$ (paper 2, Eq. 15) with $\mathbf v\otimes\phi(\mathbf k)=\phi(K_j)V_j^\top$.

### 4.2 Why direct conversion fails (§4.1)

Naive attempt: for plain linear attention ($\phi=\text{Id}$, $D_1=D_2=1$), find weights $W_1,W_2$ absorbing the ICL contribution for all $X$:

$$
W_1^\top X^\top X W_2 = W_K^\top X^\top X W_V + \underbrace{W_K^\top X'^\top X' W_V}_{\text{constant ICL term}}. \tag{7}
$$

Vectorizing (Kronecker form):

$$
(W_2^\top\otimes W_1^\top)\,\text{vec}(X^\top X) = (W_V^\top\otimes W_K^\top)\,\text{vec}(X^\top X) + (W_V^\top\otimes W_K^\top)\,\text{vec}(X'^\top X'). \tag{8}
$$

Since $X=0$ is an admissible prompt, Eq. 8 at $X=0$ forces $(W_V^\top\otimes W_K^\top)\,\text{vec}(X'^\top X')=0$, which is false in general. (Appendix A shows the only escape is to restrict all inputs to a shifted subspace $X^\top X\in c+D$ with $c$ depending on $X$ — not on $X'$ — which cannot hold for all $X$; and even then $W_1,W_2$ are non-unique.) **Conclusion: you cannot fold the context into the existing multiplicative weights; you need free additive parameters.**

### 4.3 Bias conversion for linear attention (Theorem 4.1)

Because attention is linear in $A$, a **bias on the Key-Value matrix** is the natural free parameter. Define

$$
\text{LinAttn}(Q,K,V,b_{KV}) = Q\,(K^\top V + b_{KV}). \tag{9}
$$

**Theorem 4.1.** For any weights and prompts, appending the ICL tokens is exactly equivalent to updating the bias:

$$
\text{LinAttn}([X';X]W_Q,[X';X]W_K,[X';X]W_V,\,b_{V,K}) = \text{LinAttn}(XW_Q,XW_K,XW_V,\,b'_{V,K}) \tag{10}
$$

holds for all $X$ if

$$
\boxed{\,b'_{V,K} = b_{V,K} + W_K^\top X'^\top X' W_V\,}. \tag{11}
$$

Proof is immediate: the concatenated $K^\top V$ decomposes as $X^\top X$-part $+$ $X'^\top X'$-part (cross terms vanish because ICL tokens are the *first* $M$ positions and, under autoregressive masking, input-token queries attend to ICL keys but ICL tokens never attend to input tokens). The $X'^\top X'$ part is a constant → absorbed into the bias. No causal-mask or positional-encoding bookkeeping needed for the plain case.

### 4.4 Assumptions for general linearized models (§4.3)

For a general linearized attention model the same trick works under:
- **A4.2 Autoregressive.** ICL tokens (positions $1..M$) must not be affected by input tokens; otherwise the "context contribution" would vary with the input, breaking a single-bias conversion.
- **A4.3 Relative positional encoding.** So that inserting ICL tokens does not shift the input tokens' encodings (absolute PE would require an extra correcting linear transform). RoFormer/Llama qualify.
- **A4.4 Separation of the normalizer.** $D_2([A;B]) - D_2([B]) = D_2^*([A])$, i.e. the denominator's context contribution depends only on $A$. Satisfied by kernelized attention ($D_2(X)=\sum_j\phi(K_j)$) and RetNet ($D_1=D_2=1$).

### 4.5 Full conversion with RoPE (Theorem 4.5) — the core result

With rotary positional encoding $R^{d_K}_{\Theta,i}$ (a block-diagonal rotation by $i$ positions), masked kernel attention is

$$
O_i^\top(X) = (R^{d_K}_{\Theta,i}\phi(Q_i))^\top\left[\frac{1}{D_1(Q_i)^\top D_2(X)}\sum_{j=1}^{i} R^{d_K}_{\Theta,j}\phi(K_j)V_j^\top\right]. \tag{12}
$$

Add **two** biases — $b_{KV}$ to the Key-Value matrix and $b_D$ to the normalizer:

$$
O_i^\top(X,b_{KV},b_D) = (R^{d_K}_{\Theta,i}\phi(Q_i))^\top\left[\frac{1}{D_1(Q_i)^\top(D_2(X)+b_D)}\Big(\sum_{j=1}^{i}R^{d_K}_{\Theta,j}\phi(K_j)V_j^\top + b_{KV}\Big)\right]. \tag{13}
$$

**Theorem 4.5 (exact conversion).** Prepending ICL tokens $X'$ (length $M$) is exactly reproduced by updating the biases to

$$
\boxed{\,b'_{KV} = R^{d_K}_{\Theta,-M}\,b_{KV} + \sum_{j=1}^{M} R^{d_K}_{\Theta,\,j-M}\,\phi(K'_j)V_j'^\top\,},\qquad
\boxed{\,b'_D = b_D + D_2^*(X')\,}, \tag{14–15}
$$

so that $O_i([X';X],b_{KV},b_D) = O_i(X,b'_{KV},b'_D)$ for all $X$. (Eq. 16)

**Derivation sketch (Appendix B).** Using the two rotary identities $R^{d}_{\Theta,a+b}=R^{d}_{\Theta,a}R^{d}_{\Theta,b}$ and $R^{d\,\top}_{\Theta,m}=R^{d}_{\Theta,-m}$: when the length-$M$ context is prepended, every input token shifts to position $i+M$, so its query rotary becomes $R^{d_K}_{\Theta,i+M}$ and keys $R^{d_K}_{\Theta,j+M}$. Factor $R^{d_K}_{\Theta,i}$ out of the numerator; the query–key rotation reduces to the *relative* shift $R^{d_K\top}_{\Theta,i}R^{d_K}_{\Theta,j}=R^{d_K}_{\Theta,j-i}$, which is invariant to the global $+M$ shift for the input–input terms. The ICL–input terms carry a residual rotation $R^{d_K}_{\Theta,j-M}$ on $\phi(K'_j)V_j'^\top$, and the pre-existing bias picks up $R^{d_K}_{\Theta,-M}$ — exactly Eqs. 14–15. **Intuition:** $R^{d_K}_{\Theta,-M}b_{KV}$ shifts the previously-stored context "left by $M$" to make room, then the new ICL outer-products are appended.

**Theoretical upshot:** contextual information lives in the **Key–Value** interaction, not the usual Query–Key attention-score matrix. This opens Key-Value-matrix editing (scaling terms, MLP approximations of $A$) as a research direction.

### 4.6 ICLCA algorithm (§4.5)

**Algorithm 1 (ICLCA):** feed the ICL tokens $X'^{(0)}$ through the $L$ layers. For each **linearized attention** layer $l$: compute $Q',K',V'$ from $X'^{(l-1)}$; update $b'^{(l)}_{KV}=R^{d_K}_{\Theta,-M}b^{(l)}_{KV}+\sum_{j=1}^M R^{d_K}_{\Theta,j-M}\phi(K'^{(l-1)}_j)(V'^{(l-1)}_j)^\top$ and $b'^{(l)}_D=b^{(l)}_D+D_2^*(X'^{(l-1)})$; then propagate $X'^{(l)}$ forward (needed because deeper layers' ICL features depend on shallower conversions). All non-attention sublayers are element-wise, so ICL tokens pass through them independently. Save the biases. **Cost:** one forward pass over $M$ context tokens, no backprop.

### 4.7 Approximate conversion for softmax transformers (§5)

Softmax attention is not linear in $A$, so conversion is only approximate: kernel-approximate $\text{sim}$ by $\phi$ (Performer PRF/ORF), apply Theorem 4.5 to the approximated model, and re-inject the bias into the original softmax layer:

$$
O_i^\top = \frac{\big[\sum_{j=1}^N\text{sim}(Q_i,K_j)V_j^\top\big] + R^{d_K}_{\Theta,i}\phi(Q_i)^\top b_{KV}}{\big[\sum_{j=1}^N\text{sim}(Q_i,K_j)\big] + R^{d_K}_{\Theta,i}\phi(Q_i)^\top b_D},\quad
b_{KV}=\sum_{j=1}^M R^{d_K}_{\Theta,j-M}\phi(K'_j)V_j'^\top,\ \ b_D=\sum_{j=1}^M\phi(K'_j). \tag{17–18}
$$

Accuracy is limited only by the kernel's softmax-approximation quality — and crucially the approximation is applied **only to the $M$ ICL tokens**, not all tokens, so it is strictly better than approximating the full softmax. Algorithm ICLAA (Appendix D.1).

### 4.8 RNN view (§7)

Linear transformers are RNNs; Eq. 12 is the recurrence

$$
s_i = s_{i-1} + R^{d_K}_{\Theta,i}\phi(K_i)V_i^\top,\quad z_i = z_{i-1}+D_2^*(X_i),\quad y_i=\frac{R^{d_K}_{\Theta,i}\phi(Q_i)^\top s_i}{D_1(Q_i)^\top z_i}. \tag{19}
$$

Adding the biases is **exactly** re-initializing the hidden state:

$$
s_0 = b'_{KV}-b_{KV} = \sum_{j=1}^M R^{d_K}_{\Theta,j-M}\phi(K'_j)V_j'^\top,\qquad z_0 = b'_D-b_D = D_2(X').
$$

Because future tokens only see past hidden states, capturing the ICL prompt's hidden state suffices. This generalizes the method to non-transformer autoregressive RNNs (H3, Mamba) *if* they exhibit ICL — the context is stored in the recurrence's initial state.

### 4.9 Empirical results (§6)

- **Exactness (Table 1):** relative logit error between $M_{\text{old,ICL}}$ and $M_{\text{new,no ICL}}$ over 100 random prompts: $2.9\times10^{-7}$ (205K params) → $4.3\times10^{-6}$ (1.98B). Error grows only ~2× per 10× params — scales gracefully.
- **Induction head (Table 2):** ICL model 99.95% accuracy; without ICL 2.23%; converted context-free model 99.95% — the prompt is captured exactly.
- **Added bias helps training (§6.3):** training the biased architecture from scratch is slightly faster and solves the induction task better on its own.
- **GPT-2 approximate (§6.4):** two-step (Performer-approx softmax → ICLAA bias). Logit relative error of dropping the context improves 16.56% → 9.17%; qualitatively the converted context-free model still generates context-consistent text (e.g. "CEO of Microsoft" after a removed Bill-Gates context; California/Los Angeles; Steve Jobs/Apple).

### 4.10 Practical implications & limits (§8)

Enables: ICL-guided localized fine-tuning without catastrophic forgetting; inference savings by pre-converting long repeated prompts (system prompts, personal history) into a one-time bias; a cheap initialization for existing fine-tuning. Composable (add biases) and reversible (discard). **Limits:** exact only for linearized attention with $D_2$-separation, autoregressive, relative-PE; softmax case is approximate and bottlenecked by kernel quality; extending exact conversion to softmax attention is open.

## Appendix: Section-by-Section Backbone

- **Abstract.** For linearized transformers, ICL can be made explicit and permanent via bias terms. Prove equivalence between an ICL-prompted model and the same model with added biases. Algorithm ICLCA: exact, inexpensive conversion (no gradient updates). Extend to approximate conversion for regular (softmax) transformers; GPT-2 experiments show retained context.
- **§1 Introduction.** ICL is interpretable, no parameter updates, but temporary (per-prompt). Existing context-distillation methods are empirical, gradient-based, lack guarantees. Proposal: absorb ICL into a bias on the attention module for linearized attention; ~1% extra params, composable, reversible. Contributions: ICL captured in Key-Value interaction; exact bias conversion (linear); approximate conversion (softmax); experiments (linear exact + GPT-2 approx).
- **§2 Related work.** Transformers/LLMs; linear & kernel attention (Katharopoulos, Performer, RetNet); ICL + context distillation (Bayesian ICL: Xie, Akyurek, Garg; transformers-as-statisticians: Bai; distillation: Askell/Choi/Snell — empirical, expensive).
- **§3 Preliminaries.** §3.1 attention (Def 3.1 softmax; Eq. 2 kernel-generalized; Def 3.2 linearized with $D_1,D_2$; Def 3.3 masked; Def 3.4 Key-Value matrix $A=\sum_j\phi(K_j)V_j^\top$ — adding tokens only changes $A$). §3.2 ICL setup: input prompt $X$ (len $N$) + ICL prompt $X'$ (len $M$), concatenated $[X';X]$; tokens treated as sets.
- **§4 Exact conversion.** §4.1 direct conversion fails (Eqs. 7–8, zero-prompt contradiction; Appendix A shifted-subspace escape infeasible). §4.2 bias on Key-Value matrix (Eq. 9); Theorem 4.1 + Eq. 11 (exact for linear attention). §4.3 Assumptions 4.2–4.4 (autoregressive, relative PE, normalizer separation). §4.4 RoPE conversion (Eqs. 12–13); Theorem 4.5 + Eqs. 14–16 (two biases $b_{KV},b_D$); Key-Value interaction carries context. §4.5 Algorithm 1 ICLCA.
- **§5 Approximate conversion (softmax).** Kernel-approximate sim, apply Thm 4.5, re-inject bias (Eqs. 17–18); accuracy = kernel quality; only ICL tokens approximated; ICLAA (Appendix D.1).
- **§6 Experiments.** §6.1 exactness (Table 1, $10^{-7}$–$10^{-6}$ error, 205K–1.98B). §6.2 induction head (Table 2, 99.95% preserved vs 2.23%). §6.3 bias helps from-scratch training. §6.4 GPT-2 approx (16.56%→9.17% logit error; context-consistent generations).
- **§7 Transformers as RNNs.** Recurrence (Eq. 19); biases = re-initialized hidden state $s_0,z_0$; generalizes to H3/Mamba if they show ICL.
- **§8 Conclusion.** Exact, interpretable, cheap context→weight conversion via Key-Value bias; applications (ICL-guided fine-tuning, inference savings, fine-tune init); limits (linear exact; softmax approximate; exact softmax open).
- **Appendices.** A: naive direct-conversion infeasibility. B: proof of Theorem 4.5 (rotary identities Eqs. 22–25). C: other free-parameter choices (input encoder, FFN/ROME, scaling — bias chosen). D: GPT-J tandem-attention variant + ICLAA algorithm. E: more GPT-2 examples.
