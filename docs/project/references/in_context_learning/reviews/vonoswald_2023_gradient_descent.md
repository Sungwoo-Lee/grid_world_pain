> **Per-paper review — in-context-learning corpus, paper 7 of 15 (original batch).**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§7); content is identical. Manifest: [[in_context_learning_sources]].

# 7. von Oswald et al. 2023 — Transformers Learn In-Context by Gradient Descent

**PDF:** `docs/project/references/in_context_learning/sources/von Oswald et al. 2023 - Transformers Learn In-Context by Gradient Descent.pdf` (ICML 2023, PMLR 202)

## Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction / core problem.** Garg showed Transformers *can* in-context learn like optimal algorithms; von Oswald asks the sharper follow-up: **what algorithm, mechanistically, is the forward pass running?** Their answer, for regression tasks: the Transformer is performing **gradient descent** on a small regression problem built from the in-context examples — inside its own forward pass, without any weight updates. They call this a "mesa-optimizer": an optimizer that emerges *inside* a learned network.

**Key idea in plain terms.** Normally we think of two learning loops. The *outer loop* is ordinary training: gradient descent adjusts the Transformer's weights over many tasks. The claim is that this outer training installs, inside the frozen forward pass, an *inner loop* that itself does gradient descent — one attention layer ≈ one gradient-descent step on the in-context data. So when you feed the model a prompt, it silently: (1) treats the examples as a training set, (2) takes a gradient step (or several, for a deep model) on a regression loss, and (3) reads out the resulting prediction. They prove this is *possible* with an explicit hand-built weight construction, then show that when you actually *train* such Transformers, the learned weights **match the construction**.

**Key findings.**
- **Explicit construction (Proposition 1):** a *single* linear self-attention (LSA) layer, with specific key/query/value/projection matrices, produces a token update *identical* to one gradient-descent step on a mean-squared-error regression loss.
- **Trained ≈ constructed:** when you optimize a one-layer LSA Transformer on linear regression, the learned weights (up to a scaling redundancy) coincide with the construction — verified by cosine similarity ≈1, matching predictions, and successful weight-space interpolation between trained and constructed models.
- **Depth = more steps:** $K$ stacked layers ≈ $K$ gradient steps. Trained deep Transformers actually *beat* plain gradient descent by learning an accelerated variant the authors call **GD++** (a curvature/data-preconditioning correction), and they realign perfectly with GD++.
- **Nonlinear tasks:** adding an MLP before the attention layer = gradient descent on *deep representations* (kernel regression with kernel $k(x,y)=m(x)^\top m(y)$); demonstrated on sine-wave regression.
- **Building the dataset (induction-head link):** a preceding (softmax) attention layer can *copy* neighboring tokens to assemble the $(x,y)$ token format the GD step needs — connecting the mechanism to the "induction heads" of Olsson et al.

**Initial takeaway.** This is the mechanistic capstone of the lineage: it does not just say "Transformers behave like a good learning algorithm" (Garg) or "the optimum is Bayesian" (Xie/PFN) — it identifies the *concrete computation* (gradient descent), gives a closed-form weight construction realizing it, and shows optimization *finds* that construction. It reframes in-context learning as *emergent gradient-based meta-learning in the forward pass*.

## Phase 2: Graduate-Level Deep Dive

### 2.1 Setup: linear self-attention

A standard multi-head self-attention layer updates token $e_j$ by

$$e_j \leftarrow e_j + \mathrm{SA}_\theta(j,\{e_1,\dots,e_N\}) = e_j + \sum_h P_h V_h\,\mathrm{softmax}(K_h^\top q_{h,j}), \tag{1}$$

with value/key columns $v_{h,i}=W_{h,V}e_i$, $k_{h,i}=W_{h,K}e_i$ and query $q_{h,j}=W_{h,Q}e_j$; $P_h$ is the output projection. The paper's *only* architectural departure (following Schlag et al. 2021) is to **drop the softmax**, giving the **linear self-attention (LSA)** layer:

$$e_j \leftarrow e_j + \mathrm{LSA}_\theta(j,\{e_1,\dots,e_N\}) = e_j + \sum_h P_h V_h K_h^\top q_{h,j}.$$

### 2.2 Gradient descent as a data transformation

Reference linear model $y(x)=Wx$, dataset $D=\{(x_i,y_i)\}_{i=1}^N$, squared-error loss

$$L(W) = \frac{1}{2N}\sum_{i=1}^N \|Wx_i - y_i\|^2. \tag{2}$$

One GD step with learning rate $\eta$:

$$\Delta W = -\eta\nabla_W L(W) = -\frac{\eta}{N}\sum_{i=1}^N (Wx_i - y_i)x_i^\top. \tag{3}$$

The pivotal re-interpretation: instead of updating *weights*, push the change onto the *data/targets*. After the step,

$$L(W+\Delta W) = \frac{1}{2N}\sum_{i=1}^N \|(W+\Delta W)x_i - y_i\|^2 = \frac{1}{2N}\sum_{i=1}^N \|Wx_i - (y_i - \Delta y_i)\|^2, \tag{4}$$

where $\Delta y_i = \Delta W x_i$. So **a GD step is equivalent to keeping $W$ fixed and updating each target $y_i \mapsto y_i - \Delta y_i$.** This "learning by transforming data, not weights" is exactly the form an attention layer can produce.

### 2.3 The weight construction (Proposition 1 — the mechanistic core)

Encode each in-context example as a stacked token $e_j = (x_j, y_j)\in\mathbb R^{N_x+N_y}$ for $j=1,\dots,N$, and the query as $e_{N+1}=(x_{\text{test}}, \hat y_{\text{test}})$.

**Proposition 1.** With a 1-head LSA layer, one can choose $W_K, W_Q, W_V, P$ so that the layer's update to every token equals the GD-induced dynamics

$$e_j \leftarrow (x_j, y_j) + (0, -\Delta W x_j) = (x_j, y_j) + PVK^\top q_j, \quad\text{i.e.}\quad e_j = (x_j,\, y_j - \Delta y_j),$$

*including* the test token. The explicit block-matrix construction:

$$W_K = W_Q = \begin{pmatrix} I_x & 0 \\ 0 & 0\end{pmatrix}, \qquad W_V = \begin{pmatrix} 0 & 0 \\ W_0 & -I_y\end{pmatrix}, \qquad P = \frac{\eta}{N} I,$$

where $I_x, I_y$ are identities of size $N_x, N_y$ and $W_0$ is the initial linear model. **Derivation** — substituting into the LSA update and using $\otimes$ for the outer product:

$$\begin{pmatrix} x_j \\ y_j\end{pmatrix} \leftarrow \begin{pmatrix} x_j \\ y_j\end{pmatrix} + \frac{\eta}{N} I \sum_{i=1}^N \begin{pmatrix} 0 & 0 \\ W_0 & -I_y\end{pmatrix}\begin{pmatrix} x_i \\ y_i\end{pmatrix} \otimes \begin{pmatrix} I_x & 0 \\ 0 & 0\end{pmatrix}\begin{pmatrix} x_i \\ y_i\end{pmatrix}\ \begin{pmatrix} I_x & 0 \\ 0 & 0\end{pmatrix}\begin{pmatrix} x_j \\ y_j\end{pmatrix}$$

The key/query projections keep only the $x$-parts ($W_K e_i = (x_i, 0)$, $W_Q e_j = (x_j, 0)$), so the attention weight is the inner product $x_i^\top x_j$. The value projection computes $W_V e_i = (0,\ W_0 x_i - y_i)$, i.e. the current **residual**. Collecting terms:

$$= \begin{pmatrix} x_j \\ y_j\end{pmatrix} + \frac{\eta}{N}\sum_{i=1}^N \begin{pmatrix} 0 \\ W_0 x_i - y_i\end{pmatrix}\otimes \begin{pmatrix} x_i \\ 0\end{pmatrix}\begin{pmatrix} x_j \\ 0\end{pmatrix} = \begin{pmatrix} x_j \\ y_j\end{pmatrix} + \begin{pmatrix} 0 \\ -\Delta W x_j\end{pmatrix},$$

since $\frac{\eta}{N}\sum_i (W_0 x_i - y_i)(x_i^\top x_j) = \big(\frac{\eta}{N}\sum_i (W_0 x_i - y_i)x_i^\top\big)x_j = \Delta W x_j$ by Eq. (3). The $x$-entries are untouched; only the $y$-entries absorb $-\Delta W x_j$ — exactly the target transformation of Eq. (4). $\square$

Practical notes on the construction:
- **Full vs. masked attention.** Strictly, keys/values should use only $e_1,\dots,e_N$ (exclude the query) so $\Delta W$ depends only on training data. In practice this masking can be *dropped* if $W_0\approx 0$ (small init), which makes $\Delta W$ independent of the test token anyway. Experiments confirm small-init training meets these conditions.
- **Read-out.** Initialize the query's $y$-entry as $-W_0 x_{N+1}$; then the prediction is recovered by multiplying the updated token's $y$-entry by $-1$: $-(y_{N+1}-\Delta y_{N+1}) = y_{N+1} + \Delta W x_{N+1}$. A final projection matrix (standard in Transformers) does this. A *single head* suffices.
- **Uniqueness / scaling.** Only the products $PW_V$ and $W_K W_Q$ must match; with no nonlinearity, any rescaling $PW_V s$ and $W_K W_Q / s$ is equivalent. Correcting for this lets one verify learned weights match the construction.
- **Meta-learned learning rate.** Trained across a task family, $\eta$ (folded into $P$) is *implicitly meta-learned* to the value that best reduces the average loss in a fixed number of steps.

### 2.4 Trained one-layer LSA ≈ one GD step (empirical, §3)

Train a single LSA layer to minimize the prediction error on the query token,

$$L(\theta) = \frac{1}{B}\sum_{\tau=1}^B \big\|\hat y_\theta(\{e_{\tau,i}\}_{i=1}^N, e_{\tau,N+1}) - y_{\tau,\text{test}}\big\|^2, \tag{5}$$

with fresh tasks each step (teacher $W_\tau\sim\mathcal N(0,I)$, $x_{\tau,i}\sim U(-1,1)^{n_I}$, $y_{\tau,i}=W_\tau x_{\tau,i}$; $N=n_I=10$, $n_O=1$). Because the LSA prediction is linear in $x_{\text{test}}$, $\hat y_\theta(x_{\text{test}}) = \Delta W_{\theta,D}\, x_{\text{test}}$. Compare against the control $\hat y_{\theta_{\text{GD}}}$ (one GD step from $W_0=0$, learning rate tuned by line search). Over $10^4$ validation tasks they measure (1) prediction L2 difference, (2) cosine similarity of the model *sensitivities* $\partial\hat y/\partial x_{\text{test}}$, (3) their L2 difference. **Result (Fig. 2):** near-perfect agreement — cosine ≈ 1, and a simple *scaling correction* recovers the construction so that trained TF, GD, and the interpolated weights $\theta_I=(\theta+\theta_{\text{GD}})/2$ have identical loss. OOD tests (inputs $U(-\alpha,\alpha)$, teacher scaling $\alpha W$) show the trained TF and GD degrade *together*. Repeating the LSA update (with a dampening $\lambda=0.75$) matches iterated GD (Fig. 1).

### 2.5 Depth = iterated GD, and GD++ (§3 continued)

Stack $K$ LSA layers → read out $-y_{N+1}+\sum_{k=1}^K \Delta y_{k,N+1} = y_{N+1} + \sum_{k=1}^K \Delta W_k x_{N+1}$, i.e. $K$ GD steps. Trained deep Transformers *outperform* plain $K$-step GD; the authors show they instead match **GD++**, a variant with an input **data transformation**

$$x_j \leftarrow H(X)\, x_j, \qquad H(X) = (I - \gamma X X^\top),$$

a curvature/preconditioning correction (one tuned $\gamma$; for a recurrent 2-layer LSA they realign perfectly with GD++; for a 5-layer untied model, per-layer $\eta,\gamma$ give near-zero loss and almost perfect alignment). More generally an LSA layer implements $x_j\leftarrow (I + P(X)V(X)K(X)^\top W_Q)x_j = H_\theta(X)x_j$, a *task-specific data transformation* encodable in $\theta$. Softmax SA + LayerNorm (i.e. the standard block) gives qualitatively the same alignment, though weaker than pure LSA.

### 2.6 Nonlinear regression via MLP (§3, Proposition 2)

**Proposition 2.** Given a Transformer block = an MLP $m(e)$ transforming tokens $e_j=(x_j,y_j)$ followed by an attention layer, one can construct weights implementing gradient descent on

$$\frac{1}{2N}\sum_{i=1}^N \|W m(x_i) - y_i\|^2.$$

Iterating such blocks solves **kernelized least-squares regression** with kernel $k(x,y)=m(x)^\top m(y)$ induced by the MLP. Unlike standard meta-learning (which back-props a *task-shared* embedding), the MLP here processes both inputs and targets, and its output is fed to one LSA layer implementing the GD step. Demonstrated on sine-wave regression (Fig. 4): trained TF and a meta-learned MLP + one GD output-layer step match on both initial and post-step predictions.

### 2.7 Building the dataset: copying and the induction-head link (§4, Proposition 3)

Proposition 1 assumed each token already concatenates $(x_j, y_j)$. Real prompts alternate $e_{2j}=(x_j)$, $e_{2j+1}=(0, y_j)$. **Proposition 3:** a 1-head linear or softmax attention layer (with positional encodings) can *transform* the alternating tokens into the Proposition-1 format. Empirically (Fig. 5), a 2-layer SA circuit trained on alternating tokens matches *one* GD step (not two — the first layer is spent on copying), and before the loss drops the first layer's output becomes highly sensitive to the *neighboring* token $e_{j+1}$ while staying independent of others — direct evidence of a **copying mechanism** merging input and target into one token. This required a softmax first layer (linear failed), consistent with Olsson et al.'s finding that softmax attention easily learns to copy. The authors conclude **copying (induction-head-like) is the second crucial mechanism**: copy to assemble $(x,y)$ tokens, then run GD in later layers.

### 2.8 Framing: two-timescale meta-learning / mesa-optimization

Learning Transformer weights (outer loop / slow) installs a forward-pass learning algorithm (inner loop / fast). This builds on **fast weights** (Schmidhuber 1992), shown equivalent to linear self-attention (Schlag et al. 2021), and on MAML-style "adapt only the output layer on a deep representation." The emergent-optimizer-inside-a-network phenomenon is **mesa-optimization** (Hubinger et al. 2019). The authors caution that linear-SA-only Transformers explain only a limited slice of the full ICL story; softmax, depth, and domain introduce phase transitions and complexity beyond this construction. Parallel works: Akyürek et al. 2023 (a multi-layer construction implementing one GD step *with weight decay*) and Garg et al. 2022 (performance match). This paper's distinctive claim is that **optimization actually finds the construction** (Proposition 1 verified in trained weights), and that it explains the shallow two-layer circuits studied by Olsson et al.

## Appendix: Section-by-Section Backbone

- **Abstract / §1 Introduction.** Hypothesis: training Transformers ≈ gradient-based meta-learning. Single-LSA-layer construction = one GD step; trained TFs match it → mesa-optimizers doing GD in the forward pass; GD++ curvature correction; deep representations for nonlinear tasks; induction-head parallel. Meta-learning framing (fast/slow weights, fast weights ≡ linear attention (Schlag), MAML). **Hypothesis 1** stated. Contrast with Akyürek et al. (multi-layer construction) — here a simpler single-layer one, verified to be found by optimization.
- **§2 Linear self-attention can emulate gradient descent.** SA layer (Eq. 1) → LSA (drop softmax). GD as data transformation: loss (Eq. 2), step (Eq. 3), target-update reinterpretation (Eq. 4). Token encoding $e_j=(x_j,y_j)$, query $e_{N+1}$. **Proposition 1** + five practical notes (full/masked attention & $W_0\approx0$; read-out via $-1$ multiply; uniqueness/scaling; meta-learned $\eta$; task-specific $H_\theta(X)$).
- **§3 Trained Transformers do mimic GD.** Token construction; training objective (Eq. 5); teacher-generated data. One-step: trained LSA vs constructed $\theta_{\text{GD}}$ — prediction/sensitivity metrics, interpolation, OOD, repeated update (Fig. 2). Multi-step: $K$ layers = $K$ steps; TF beats GD, matches **GD++** ($H(X)=I-\gamma XX^\top$); recurrent 2-layer and 5-layer untied (Fig. 3); softmax+LayerNorm variant. Nonlinear: **Proposition 2** (MLP + attention = GD on deep reps = kernel regression), sine-wave (Fig. 4).
- **§4 Do self-attention layers build regression tasks?** **Proposition 3** (copy alternating $(x)$ / $(0,y)$ tokens into Prop-1 format). 2-layer circuit matches one GD step; first layer becomes sensitive to neighbor token (Fig. 5) → copying mechanism; needs softmax; induction-head connection.
- **§5 Discussion.** Recap of GD hypothesis; outer-loop = meta-learning; builds on Schlag et al. delta-rule fast weights; extends to $K$-step GD and GD++; explains standard architectures; caveats on scope.
- **Appendices A.1–A.12.** A.1 Proposition 1 construction + full derivation (Eqs. 6–7 and block-matrix expansion). A.5 Proposition 3 copying construction. A.6 dampening $\lambda$. A.8 kernel regression / kernel smoothing discussion. A.9 softmax + LayerNorm results. A.10 GD++ curvature-correction details. A.11 phase transitions (seed-dependent). A.12 experimental details incl. fixed-dataset-size cycling (Fig. 16).

---
