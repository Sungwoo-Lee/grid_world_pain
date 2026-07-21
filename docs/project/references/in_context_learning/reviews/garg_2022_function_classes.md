> **Per-paper review — in-context-learning corpus, paper 6 of 15 (original batch).**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§6); content is identical. Manifest: [[in_context_learning_sources]].

# 6. Garg et al. 2022 — What Can Transformers Learn In-Context? A Case Study of Simple Function Classes

**PDF:** `docs/project/references/in_context_learning/sources/Garg et al. 2022 - What Can Transformers Learn In-Context.pdf` (NeurIPS 2022)

## Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction / core problem.** GPT-style models can in-context learn, but with real language it is impossible to disentangle *genuine on-the-fly learning* from *retrieving a task the model already memorized during pretraining*. Garg et al. sidestep language entirely and pose a clean, controllable question: **can a Transformer be trained from scratch to in-context learn a well-defined function class** — say, linear functions — so that, given a prompt of input-output pairs $(x_1, f(x_1), \dots, x_k, f(x_k), x_{\text{query}})$ from an *unseen* function $f$, it outputs $f(x_{\text{query}})$?

**Key idea in plain terms.** A "function class" is a family like "all linear functions in 20 dimensions." Training: repeatedly sample a random function $f$ from the class and random inputs, form a prompt, and train the Transformer to predict each output from the preceding examples (squared loss). Testing: hand it prompts from *fresh* functions it never saw and measure how close its prediction is to the truth versus known learning algorithms (least squares, Lasso, etc.). If it matches the best algorithm, the Transformer has effectively *learned a learning algorithm* and runs it in a single forward pass.

**Key findings.**
- A standard 12-layer GPT-2-style Transformer (22.4M params) trained from scratch **in-context learns linear functions as well as the optimal least-squares estimator** (error 0.02 at $k=d=20$ examples, 0.0006 at $2d$).
- The ability is **robust to distribution shift**: skewed input covariance, label noise (even reproducing the least-squares *double-descent* curve), and in-context/query inputs in different orthants — so it is not doing nearest-neighbor lookup, it genuinely regresses.
- It scales to **harder classes**: sparse linear functions (matches Lasso, beating least squares by exploiting sparsity), depth-4 decision trees (beats greedy tree learning and XGBoost), and 2-layer ReLU networks (matches a network trained by gradient descent). A model trained on 2-layer nets also in-context learns linear functions for free.
- **Model capacity and curriculum learning** matter: more parameters → lower error and higher usable dimension; a curriculum (start low-dimensional, grow) gives drastic training speed-ups. Non-trivial ICL needs relatively little data (~1k distinct functions or 100k prompts).

**Initial takeaway.** Garg et al. provide the clean *empirical* counterpart to Xie's theory: they *build* Transformers that provably in-context learn function classes, and show the learned computation matches sophisticated iterative algorithms (Lasso, gradient descent) despite running in one forward pass. This is the paper that turned "does the Transformer learn an algorithm?" into a tractable, measurable question — and set up von Oswald's next question, "*which* algorithm?"

## Phase 2: Graduate-Level Deep Dive

### 2.1 Formalizing in-context learning of a function class

Let $D_{\mathcal X}$ be a distribution over inputs and $D_{\mathcal F}$ a distribution over functions in class $\mathcal F$. A **prompt** is $P = (x_1, f(x_1), \dots, x_k, f(x_k), x_{\text{query}})$ with $x_i, x_{\text{query}}\stackrel{\text{iid}}{\sim} D_{\mathcal X}$ and $f\sim D_{\mathcal F}$. A model $M$ in-context learns $\mathcal F$ up to $\epsilon$ (w.r.t. $D_{\mathcal F}, D_{\mathcal X}$) if

$$\mathbb E_P\big[\ell\big(M(P), f(x_{\text{query}})\big)\big] \le \epsilon, \tag{1}$$

for a loss $\ell$ (squared error here). Crucially, *being able to in-context learn is a property of $M$ alone*, independent of how $M$ was produced — training $M$ is an instance of meta-learning.

### 2.2 Training objective

Sample $f\sim D_{\mathcal F}$ and inputs $x_1,\dots,x_{k+1}\sim D_{\mathcal X}$; build prompt prefixes $P^i = (x_1, f(x_1),\dots, x_i, f(x_i), x_{i+1})$. Train $M_\theta$ to minimize the average loss over all prefixes:

$$\min_\theta\ \mathbb E_P\Big[\frac{1}{k+1}\sum_{i=0}^{k}\ell\big(M_\theta(P^i),\, f(x_{i+1})\big)\Big]. \tag{2}$$

**Implementation.** Decoder-only GPT-2-family Transformer, 12 layers, 8 heads, 256-dim embeddings (22.4M params). Prompt outputs $f(x_i)$ are zero-padded to input dimension, both mapped to embedding space by learnable linear maps; a learnable linear read-out maps the predicted vector to a scalar. The architecture computes predictions for *all* prefixes $M_\theta(P^i)$ in one forward pass. Trained from scratch (no pretrained LM, no text), batch 64, 500k steps, squared-error loss.

**Curriculum learning.** Start with inputs confined to a 5-dim subspace and prompt length 11; every 2000 steps increase subspace dim by 1 and prompt length by 2, up to ambient $d$ and length $2d{+}1$. This is the decisive trick for high $d$: at $d=50$, loss barely moves in 500k steps *without* curriculum but nearly reaches optimum *with* it.

### 2.3 Linear functions: matching the Bayes-optimal estimator

$\mathcal F = \{f(x)=w^\top x\}$ in $d=20$; $x_i, x_{\text{query}}, w\sim\mathcal N(0, I_d)$. Baselines: least squares (minimum-norm fit — the optimal estimator, a lower bound), 3-nearest-neighbors, and simple averaging $\hat w = \frac1k\sum y_i x_i$. **Result (Fig. 2):** the Transformer's normalized squared error $\big(M(P)-w^\top x_{\text{query}}\big)^2/d$ tracks least squares for every $k$; at $k=d=20$, least squares hits 0 while the Transformer reaches 0.02, improving to 0.0006 at $2d$. Simple baselines are far worse, so the model encodes a *non-trivial* algorithm — and an ablation (Appendix B.7) rules out memorization.

**What function is it computing?** Fix the $k$ in-context examples and view the output as $\hat f_{w, x_{1:k}}(x_{\text{query}})$. When $k<d$ the ground truth is unrecoverable and the ideal prediction is the projection $\big(\mathrm{proj}_{x_{1:k}}(w)\big)^\top x_{\text{query}}$. Two probes:
- **Function along a random direction:** $\hat f_{w,x_{1:d}}(\lambda x)$ and $\hat f_{w,x_{1:2d}}(\lambda x)$ match ground truth; $\hat f_{w,x_{1:d/2}}(\lambda x)$ matches the *projected* ground truth (for query norm near a typical training input).
- **Local correctness via gradient:** the model is differentiable, so compute $\nabla_{x_{\text{query}}}\hat f$. It should point along $\mathrm{proj}_{x_{1:k}}(w)$. Fig. 3b shows the normalized inner product of gradient with projected $w$ is ~1 for all $k$, and with $w$ itself once $k\ge d$ (since $\mathrm{proj}_{x_{1:k}}(w)=w$ a.s. then). The model is *locally* the correct linear map.

### 2.4 Out-of-distribution robustness (evidence of a real algorithm)

Train on $D^{\text{train}}$, evaluate on shifted $D^{\text{test}}$ (and a separate query distribution $D^{\text{test}}_{\text{query}}$):
- **Skewed covariance:** inputs from $\mathcal N(0,\Sigma)$, eigenvalues $\propto 1/i^2$, norm-matched. Tracks least squares up to $k=10$ then plateaus — imperfect but still $<$ half the NN-baseline error.
- **Noisy linear regression:** $y_i = w^\top x_i + \epsilon_i$, $\epsilon_i\sim\mathcal N(0,1)$. Tracks least squares and even reproduces the **double-descent** error spike near $k=d$ — a signature of the least-squares estimator, not of nearest-neighbor.
- **Different orthants:** all in-context $x_i$ forced into one (random) orthant, query in another. Essentially unaffected (errors 0.062 at $k=20$, 0.004 at $k=40$), which a nearest-neighbor method could not achieve — confirming it is *not* doing similarity search.

### 2.5 More complex classes

- **Sparse linear ($d=20, s=3$):** matches **Lasso** (which has no closed form and needs iterative $\ell_1$-regularized minimization), beating minimum-norm least squares — the Transformer *exploits sparsity* in one forward pass.
- **Depth-4 decision trees (16 leaves):** beats greedy tree learning and XGBoost; even when the baselines are handed the sign pattern of inputs, the Transformer still wins (0.12 vs 0.31 error at 100 examples).
- **2-layer ReLU nets (100 hidden units):** matches a same-architecture net trained by Adam on the in-context examples; and the model trained on 2-layer nets *also* in-context learns linear functions (unprompted transfer).

### 2.6 What matters: capacity, curriculum, data

- **Capacity & dimension (Fig. 6):** across 3.4M / 7.6M / 22.4M params and $d\in\{10,\dots,50\}$, error falls with capacity and rises with $d$; capacity especially helps OOD (skewed covariance, orthant shift).
- **Curriculum:** essential at high $d$; without it, a long loss plateau precedes a sharp drop — reminiscent of the sudden emergence of **induction heads** (Olsson et al.) in language models.
- **Data efficiency:** non-trivial ICL with $n_p=100$k distinct prompts or $n_w=1$k distinct functions; near-unrestricted performance at $n_p=1$M / $n_w=10$k. (The main model sees 32M distinct functions over training.)

**Positioning.** Garg et al. explicitly cite Xie (Bayesian-inference framing) and Müller (PFNs) as the closest prior work, and frame their own contribution as posing ICL as *learning a function class at inference time* and empirically probing the encoded algorithm and its extrapolation.

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** ICL as learning a function class $\mathcal F$ from in-context examples (Eq. 1, def. of ICL up to $\epsilon$). Motivation: disentangle genuine learning from indexing memorized tasks. Preview: Transformers learn linear functions (∼least squares), robust to two kinds of distribution shift, and learn sparse-linear / trees / 2-layer nets.
- **§2 Training models for in-context learning.** Prompt construction; training objective over prefixes (Eq. 2); model structure (GPT-2 decoder, 12L/8H/256d, 22.4M); zero-padding of outputs; all-prefix single-pass prediction; curriculum learning.
- **§3 In-context learning of linear functions.** §3.1 matches least squares (Fig. 2); no memorization (App B.7). §3.2 what function: prefix-conditioned $\hat f$, projection when $k<d$; visualization along random direction; local correctness via gradient alignment with $\mathrm{proj}_{x_{1:k}}(w)$ (Fig. 3).
- **§4 Extrapolating beyond training distribution.** Formal train/test distribution split incl. separate query distribution. Skewed covariance; noisy regression (double descent); different orthants (Fig. 4). Concludes it learned real linear regression, not NN search.
- **§5 More complex function classes.** Sparse linear (∼Lasso); decision trees (> greedy/XGBoost); 2-layer ReLU nets (∼GD-trained net; also learns linear) (Fig. 5).
- **§6 Investigating what matters.** Problem dimension & capacity (Fig. 6); curriculum (speed-up, plateau→drop, induction-head analogy); number of distinct prompts/functions (data efficiency).
- **§7 Related work.** ICL (Xie Bayesian framing, Min et al., Chan et al., Olsson induction heads); Transformers (Müller PFNs, Nguyen & Grover TNPs); meta-learning; data-driven algorithm design.
- **§8 Discussion.** Formalized ICL of function classes; open questions on complexity, curriculum, inductive biases, reverse-engineering encoded algorithms.
- **Appendices A–B.** A architecture/training/baseline details, curriculum schedule. B additional results: query scaling robustness, full OOD, capacity plots, training variance, curriculum comparison, data-count ablations, no-memorization check (B.7).

---
