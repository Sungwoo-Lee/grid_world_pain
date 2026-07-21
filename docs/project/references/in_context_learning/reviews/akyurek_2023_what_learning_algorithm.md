> **Per-paper review — in-context-learning corpus, paper 10 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§10); content is identical. Manifest: [[in_context_learning_sources]].

# 10. Akyürek et al. 2023 — What Learning Algorithm Is In-Context Learning? (Investigations with Linear Models)

**PDF:** `docs/project/references/in_context_learning/sources/Akyurek et al. 2023 - What learning algorithm is in-context learning.pdf`
**Venue:** ICLR 2023. **Authors:** Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, Denny Zhou (Google Research / MIT CSAIL / Stanford).

## Phase 1: Foundational Overview (Undergraduate-Level)

**The plain-language question.** When you give a large language model a few worked examples in its prompt — pairs of an input `x` and its answer `f(x)` — and it then correctly answers a new input, *how* is it doing that? The model's weights never change. So where does the "learning" happen? This paper's hypothesis: **the transformer builds a small model inside its own activations (its scratch-pad of intermediate numbers) and trains that small model on the fly**, using an algorithm it discovered during pre-training. In-context learning (ICL) would then be a familiar textbook estimator running silently inside the network.

**The testbed.** To make this checkable, the authors use the simplest interesting learning problem: **linear regression**. The true function is $f(x) = w^\top x$; the model sees a sequence of $(x_i, y_i)$ pairs and must predict $y$ for a fresh $x$. Linear regression is ideal because it has many known solution algorithms (gradient descent, ridge regression, exact least squares, the Bayesian optimum), and they give *different* answers when data are scarce or noisy — so you can tell which one the transformer is imitating.

**Three lines of evidence.**
1. **It is possible in principle (proof by construction).** They hand-build transformer weights that provably execute one step of gradient descent (needing only about `d` hidden units for a `d`-dimensional problem) and, separately, one step of exact ridge-regression updating via the Sherman–Morrison formula (needing about `d²` units). So the architecture is expressive enough to *be* these algorithms.
2. **Trained models actually behave like these algorithms.** A transformer trained from scratch on the ICL objective produces predictions that closely match ordinary least squares (OLS) on clean data, and match the **minimum-Bayes-risk ridge predictor** when the data are noisy. As you dial up depth, the model transitions through **algorithmic "phases"**: 1-layer ≈ one gradient step, 2–4 layers ≈ ridge regression, 8+ layers ≈ exact OLS.
3. **You can read the intermediate quantities out of its activations.** Using a trained "probe," they recover the moment matrix $X^\top Y$ and the least-squares weight vector $w_{\mathrm{OLS}}$ from hidden states — and $X^\top Y$ appears in early layers, $w$ in later layers, mirroring the order a real algorithm would compute them.

**Initial takeaway.** ICL — at least in this controlled linear setting — is *not* mysterious. The transformer appears to rediscover and run standard estimation algorithms. The specific algorithm it lands on depends on how much capacity (depth, width) it has and how noisy the data are. This reframes "emergent" ICL as ordinary machine learning happening one level down.

## Phase 2: Graduate-Level Deep Dive

Phase 2 foregrounds the **algorithm-implementation claim**: exactly how fixed transformer weights simulate (a) a gradient-descent step and (b) a Sherman–Morrison ridge update, and how the empirical "phase transition" and Bayesian-matching results are established.

### Setup and encoding

Function class $\mathcal{F} = \{f(x) = w^\top x : w, x \in \mathbb{R}^d\}$; squared loss $L(y, y') = (y-y')^2$. The (regularized) objective and its closed-form minimizer are

$$
\sum_i L(w^\top x_i, y_i) + \lambda \|w\|_2^2 \quad\Longrightarrow\quad w^\star = (X^\top X + \lambda I)^{-1} X^\top y .
$$

$\lambda = 0$ gives OLS; $\lambda > 0$ gives ridge. Each example is embedded as a $(d{+}1)$-dimensional token: $\tilde x_i = [0, x_i]$ and $\tilde y_i = [y_i, 0_d]$, so inputs and labels alternate in a single token stream. The ICL training objective is autoregressive:

$$
\arg\min_\theta\; \mathbb{E}_{\substack{x_1,\dots,x_n \sim p(x)\\ f \sim p(f)}}\left[\sum_{i=1}^n L\big(f(x_i),\, T_\theta([x_1, f(x_1), \dots, x_i])\big)\right].
$$

### Computational primitives (Lemma 1)

The constructions are built from four operations, each realizable by **a single decoder layer** (self-attention + feed-forward as in Eqs. 1–5 of the paper):
- `mov` — copy a block of rows from one column (token) into another (data movement across positions, done by attention).
- `mul` — interpret two row-blocks of a column as matrices $A_1$ ($a\times b$) and $A_2$ ($b\times c$) and write $A_1 A_2$ into a target block.
- `div` — divide a block by the absolute value of a scalar entry (used for the Sherman–Morrison denominator).
- `aff` — apply an affine map $W_1 h_{i:j} + W_2 h_{i':j'} + b$ to row-blocks.

That these four are single-layer-expressible is the workhorse lemma; everything else is composition.

### Theorem 1 — one gradient-descent step in $O(d)$ hidden space

Gradient descent on the ridge objective for a single example $(x_i, y_i)$ is

$$
w' = w - \alpha\,\frac{\partial}{\partial w}\Big[L(w^\top x_i, y_i) + \lambda\|w\|_2^2\Big]
   = w - 2\alpha\big(x_i x_i^\top w - y_i x_i + \lambda w\big). \tag{GD}
$$

**Derivation of the gradient.** With $L = (w^\top x_i - y_i)^2$,
$$
\frac{\partial L}{\partial w} = 2(w^\top x_i - y_i)\,x_i = 2\big(x_i x_i^\top w - y_i x_i\big),
$$
and $\partial(\lambda\|w\|^2)/\partial w = 2\lambda w$; summing and multiplying by $-\alpha$ gives (GD). (The paper writes the scalar form $x(w^\top x) - yx + \lambda w$.)

**How the layers realize (GD).** Starting from the block-Hankel input $H^{(0)}$ that stacks $y_i$, $x_i$, and the query $x_n$, the transformer executes the update as a sequence of primitive calls (reconstructed from the appendix trace):
1. `aff` with $W_1 = w$: compute the prediction $w^\top x_i$ and write it below $x_i$.
2. `aff` with $W_1 = I, W_2 = -I$: compute the residual $w^\top x_i - y_i$.
3. `mul`: form the outer-product-scaled vector $x_i(w^\top x_i - y_i)$ — this is (half) the per-example gradient.
4. `aff` with bias $b = w$ and scale $-2\alpha$: form $w' = w - 2\alpha\,x_i(w^\top x_i - y_i)$ (the $\lambda w$ term folds into the affine bias/scale).
5. `aff` with $W_1 = w'$: read out the new prediction $w'^\top x_n$ at the query column.

This needs only a **constant number of layers** and **$O(d)$ hidden width** (the largest intermediate is a $d$-vector). Stacking $n$ copies of this block gives $n$ gradient steps.

### Theorem 2 — one exact ridge update via Sherman–Morrison in $O(d^2)$ hidden space

Directly evaluating $w^\star = (X^\top X + \lambda I)^{-1} X^\top y$ requires a matrix inverse. The **Sherman–Morrison identity** turns the inverse into example-by-example rank-one updates. For invertible $A$ and vectors $u, v$:

$$
(A + u v^\top)^{-1} = A^{-1} - \frac{A^{-1} u v^\top A^{-1}}{1 + v^\top A^{-1} u}. \tag{SM}
$$

Because $X^\top X = \sum_i x_i x_i^\top$ is a sum of rank-one terms, (SM) lets you fold in one training example at a time. The single-example predictor the transformer computes is

$$
w' = \big(\lambda I + x_i x_i^\top\big)^{-1} x_i y_i
   = \left(\frac{I}{\lambda} - \frac{\tfrac{I}{\lambda} x_i x_i^\top \tfrac{I}{\lambda}}{1 + x_i^\top \tfrac{I}{\lambda} x_i}\right) x_i y_i, \tag{Ridge-SM}
$$

obtained by applying (SM) with $A = \lambda I$ (so $A^{-1} = I/\lambda$), $u = v = x_i$. The numerator $\tfrac{I}{\lambda} x_i x_i^\top \tfrac{I}{\lambda}$ is a $d \times d$ matrix, hence the $O(d^2)$ hidden-space requirement; the denominator scalar is handled by the `div` primitive. This is a single rank-one update; $n$ stacked blocks give the full $n$-example solution. The paper stresses these are **upper bounds on capacity to implement**, not statements that training discovers exactly these weights.

### What algorithm is actually learned — behavioral metrics

Two metrics quantify "which known algorithm does the trained model imitate":

**Squared prediction difference (SPD)** — agreement at the output level, algorithm-agnostic:
$$
\mathrm{SPD}(A_1, A_2) = \mathbb{E}_{\substack{D \sim p(D)\\ x' \sim p(x)}}\big(A_1(D)(x') - A_2(D)(x')\big)^2 .
$$

**Implicit linear weight difference (ILWD)** — agreement at the parameter level. For an algorithm $A$, generate its predictions on unlabeled probes $\{x'_i\}$, fit the best linear weights to those predictions,
$$
\hat w_A = \arg\min_w \sum_i \big(A(D)(x'_i) - w^\top x'_i\big)^2, \qquad
\mathrm{ILWD}(A_1, A_2) = \mathbb{E}_D \mathbb{E}_{D_{X'}} \|\hat w_{A_1} - \hat w_{A_2}\|_2^2 .
$$

**Empirical findings.**
- On **noiseless** data, a large model ($L=16, H=512, M=4$) matches **OLS** to squared error $< 0.01$ under both SPD and ILWD, and is markedly farther from $k$-NN or single-step GD. In the under-determined regime ($n < d$), it tracks the **minimum-norm** OLS solution.
- On **noisy** data, the Bayes-optimal predictor is $\hat y = \mathbb{E}[y \mid x, D]$, which for Gaussian prior $w \sim \mathcal{N}(0, \tau^2 I)$ and Gaussian noise $\sigma^2$ has the ridge form
$$
\hat w = \Big(X^\top X + \tfrac{\sigma^2}{\tau^2} I\Big)^{-1} X^\top Y, \qquad \hat y = \hat w^\top x .
$$
The transformer's best-fitting ridge parameter tracks $\sigma^2/\tau^2$ across the whole noise grid — i.e. **ICL behaves like the minimum-Bayes-risk estimator**. As $\sigma \to 0^+$ this collapses back to OLS, unifying the noiseless result.
- **Algorithmic phase transitions.** Varying depth: 1-layer ≈ one GD step (poorly, in absolute terms), 2–4 layers ≈ ridge, 8+ layers ≈ OLS. Varying hidden size shows an analogous shift. Notably, ridge-like behavior appears at $H \geq 16$ (8-D) / $H \geq 32$ (16-D) — far below the $O(d^2)$ the Sherman–Morrison construction would need, so **training finds more hidden-state-efficient solutions** than the hand construction.

### Probing for intermediate quantities

Freeze the trained model; train an auxiliary probe with position-attention:
$$
\alpha = \mathrm{softmax}(sv), \qquad \hat v = \mathrm{FF}_v\big(\alpha^\top W_v H^{(l)}\big),
$$
minimizing $\|v - \hat v\|^2$ against targets $v \in \{X^\top Y,\, w_{\mathrm{OLS}}\}$. A 2-layer MLP probe beats a linear probe (targets are encoded **non-linearly**, unlike the linear hand construction), both targets decode accurately in deep layers, and — matching how a real algorithm proceeds — the moment matrix $X^\top Y$ becomes decodable early (~layer 7) and the weight vector $w$ later (~layer 12).

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** Poses the "how, not what" question of ICL. Hypothesis: transformers encode an implicit model in activations and train it in-context. Preview of the three evidence strands; states the capacity results ($O(d)$ hidden + constant depth → one GD step; $O(d^2)$ → one ridge update).
- **§2 Preliminaries.** ICL as meta-learning reduced to ordinary supervised learning. §2.1 defines the decoder transformer: self-attention $a_i = \mathrm{Attention}(h_i; W^F, W^Q, W^K, W^V)$ with softmax heads (Eqs. 1–3), feed-forward $\mathrm{FF}(a_i; W_1, W_2)$ with GeLU nonlinearity and layer-norm (Eqs. 4–7). §2.2 gives the autoregressive ICL objective (Eq. 8). §2.3 sets up linear regression, the closed-form solution (Eq. 10), and the $(d{+}1)$-D token encoding.
- **§3 What learning algorithms can a transformer implement?** §3.1 defines the four primitives `mov`/`mul`/`div`/`aff` and Lemma 1 (each is one decoder layer). §3.2 Theorem 1: constant depth + $O(d)$ hidden space implements one GD step (Eq. 11). §3.3 Theorem 2: Sherman–Morrison identity (Eq. 13) yields a rank-one ridge update (Eq. 14) in constant depth + $O(d^2)$ hidden space. Discussion contrasts with universality results (Yun et al.; Schlag et al. linear self-attention) and notes single-step → multi-step by layer-stacking.
- **§4 What computation does an in-context learner perform?** §4.1 behavioral metrics SPD (Eq. 15) and ILWD (Eqs. 16–17). §4.2 experimental setup (hyperparameter sweep; $L=16, H=512, M=4$ best; 500k iterations; 40 pairs per context; Gaussian $w, x$). §4.3 results: ICL ≈ OLS on noiseless data (Fig. 1); ICL ≈ minimum-Bayes-risk ridge on noisy data (Eqs. 18–19, Fig. 2); algorithmic phase transitions with depth and width (Fig. 3).
- **§5 Does ICL encode meaningful intermediate quantities?** Probing methodology (Eqs. 20–21). Recovers $X^\top Y$ and $w_{\mathrm{OLS}}$ non-linearly from deep layers; $X^\top Y$ decodable earlier than $w$; control task (fixed $w=1$) probes worse, confirming non-triviality (Fig. 4).
- **§6 Conclusion.** Transformers can in theory implement multiple linear-regression algorithms, empirically implement a range of them (transitioning with capacity and noise), and expose probeable intermediates. Extensible to richer function classes via nonlinear feature layers and to large-scale LMs.
- **Appendices A–E.** Proofs of Lemma 1 and Theorems 1–2 (explicit primitive-by-primitive traces — the GD trace was used above), training hyperparameters, and probing details.

---
