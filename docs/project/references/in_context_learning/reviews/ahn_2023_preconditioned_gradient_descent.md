> **Per-paper review — in-context-learning corpus, paper 13 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§13); content is identical. Manifest: [[in_context_learning_sources]].

# 13. Ahn et al. 2023 — Transformers Learn to Implement Preconditioned Gradient Descent for In-Context Learning

**PDF:** `docs/project/references/in_context_learning/sources/Ahn et al. 2023 - Transformers learn to implement preconditioned gradient descent.pdf`
**Venue:** NeurIPS 2023 (arXiv:2306.00297v2).

## Phase 1: Foundational Overview (Undergraduate-Level)

**The question in plain terms.** Earlier work (von Oswald et al. 2023; Akyürek et al. 2022) showed that if you *hand-pick* the weights of a transformer, it can carry out gradient descent (GD) on a small regression problem hidden inside its input prompt. That is an **expressivity** claim: the machine is *capable* of it. But in practice nobody hand-picks weights — you train them by minimizing a loss with SGD/Adam, which is non-convex and offers no guarantee you land on the "GD-implementing" weights. Ahn et al. ask the harder question: **if you actually train the network, is the GD-implementing solution the one training prefers?** They answer yes, and more: the preferred solution is not plain GD but a *smarter* variant called **preconditioned gradient descent**.

**The testbed.** A "linear transformer" (self-attention with the softmax removed) is trained on random **linear-regression prompts**: each prompt packs $n$ labeled examples $(x^{(i)}, y^{(i)}=\langle x^{(i)}, w_\star\rangle)$ plus one unlabeled query $x^{(n+1)}$, and the network must predict $\langle w_\star, x^{(n+1)}\rangle$. The task weight $w_\star$ and the covariates change every prompt, so the only way to succeed is to *learn a regression algorithm* that reads the labeled examples and infers $w_\star$ on the fly.

**Key findings.**
- **Single layer (Theorem 1).** The *global minimum* of the training loss makes the network perform **exactly one step of preconditioned gradient descent**. The "preconditioner" is a matrix that (a) approximates the inverse of the data covariance $\Sigma^{-1}$ — which speeds up convergence when the inputs are badly scaled — and (b) adds a regularizer that grows when the number of in-context examples $n$ is small (i.e., when the data are inadequate). So the trained network doesn't just do GD; it adapts its step to *both* the data geometry *and* the sample size, resembling structural risk minimization.
- **Multiple layers (Theorems 2–4).** Under a natural sparsity restriction on the weights, an $L$-layer network's forward pass is *provably identical* to $L$ steps of preconditioned GD (Lemma 1). Theorem 2 shows a 2-layer net's optimum is GD with adaptive, coordinate-wise step sizes (Adagrad-like, but tuned to the data distribution rather than the instance). Theorem 3 shows that for deeper nets, the choice "preconditioner $\propto \Sigma^{-1}$ at every layer" (Newton-like) is a critical point. Theorem 4 relaxes the sparsity and finds a *richer* algorithm that both preconditions the gradient *and* reshapes the covariates to improve conditioning between steps — this recovers the empirically-observed **GD++** algorithm of von Oswald et al.
- **Experiments** on 3-layer nets confirm the trained weights converge to exactly these critical points, with loss going to ~0 (suggesting they are global optima).

**Initial takeaway.** This is the first rigorous *loss-landscape* result for learning algorithms via training. It reframes "transformers can do GD" (expressivity) into "training a transformer *discovers* an adaptive, preconditioned GD tuned to the task distribution" (optimization). The bonus insight: the learned preconditioner encodes distributional knowledge (covariance + sample-size-dependent regularization) that plain GD lacks — the transformer is learning a *better-than-vanilla* optimizer.

## Phase 2: Graduate-Level Deep Dive

**Architecture.** Following Schlag et al. (2021) and von Oswald et al. (2023), the softmax-free linear self-attention layer is, with reparameterization $P:=W_v$ and $Q:=W_k^\top W_q$,

$$
\mathrm{Attn}_{P,Q}(Z) = P\,Z\,M\,(Z^\top Q Z), \qquad
M := \begin{bmatrix} I_n & 0 \\ 0 & 0 \end{bmatrix}\in\mathbb{R}^{(n+1)\times(n+1)},
$$

where the mask $M$ enforces prompt asymmetry (the query's label is unknown). The input matrix stacks covariates over labels:

$$
Z_0 = \begin{bmatrix} x^{(1)} & \cdots & x^{(n)} & x^{(n+1)} \\ y^{(1)} & \cdots & y^{(n)} & 0 \end{bmatrix}\in\mathbb{R}^{(d+1)\times(n+1)}.
$$

An $L$-layer transformer applies residual updates $Z_{\ell+1} = Z_\ell + \tfrac1n \mathrm{Attn}_{P_\ell,Q_\ell}(Z_\ell)$, and the prediction is $\mathrm{TF}_L(Z_0) = -[Z_L]_{(d+1),(n+1)}$ (the minus sign matches von Oswald et al.). Training minimizes the **in-context loss**

$$
f\big(\{P_\ell,Q_\ell\}\big) = \mathbb{E}_{Z_0, w_\star}\Big[\big(\mathrm{TF}_L(Z_0) + w_\star^\top x^{(n+1)}\big)^2\Big].
$$

### Derivation 1 — the loss depends only on $(b, A)$ (single layer)

Spell out $Z_0 + \tfrac1n\mathrm{Attn}_{P,Q}(Z_0)$. Because $M$ zeros out the query column inside the sum,

$$
\Big[Z_0 + \tfrac1n P\Big(\textstyle\sum_{i=1}^n z^{(i)}z^{(i)\top}\Big) Q\, Z_0\Big]_{\text{last col}}
= \begin{bmatrix} x^{(n+1)} \\ 0 \end{bmatrix} + \tfrac1n P\Big(\sum_{i=1}^n z^{(i)}z^{(i)\top}\Big)Q\begin{bmatrix} x^{(n+1)} \\ 0\end{bmatrix}.
$$

Taking the $(d+1)$-th entry, only the **last row of $P$** (call it $b^\top$) and the **first $d$ columns of $Q$** (call it $A\in\mathbb{R}^{(d+1)\times d}$) survive. Writing $G := \tfrac1n\sum_i z^{(i)}z^{(i)\top}$, the prediction is $\tfrac1n b^\top G A x^{(n+1)}$ (up to the constant scaling absorbed into $A$), so the loss collapses to

$$
f(b,A) = \mathbb{E}_{Z_0,w_\star}\Big[\big(b^\top G A\, x^{(n+1)} + w_\star^\top x^{(n+1)}\big)^2\Big]
= \mathbb{E}\Big[\big((b^\top G A + w_\star^\top)\,x^{(n+1)}\big)^2\Big].
$$

This is the crucial reduction: the enormous parameter matrices reduce to the pair $(b, A)$.

### Derivation 2 — global minimum for isotropic data (Theorem 1, $\Sigma=I$)

Decompose over coordinates. Using $\mathbb{E}[x^{(n+1)}[j]x^{(n+1)}[j']]=\delta_{jj'}$ and writing $A=[a_1,\dots,a_d]$,

$$
f(b,A) = \sum_{j=1}^d \mathbb{E}_{Z_0,w_\star}\big[b^\top G a_j + w_\star[j]\big]^2 .
$$

Reparameterize each term via the trace identity $b^\top G a_j = \mathrm{Tr}(G\,a_j b^\top) = \langle G, b a_j^\top\rangle$ (with $\langle X,Y\rangle := \mathrm{Tr}(XY^\top)$), and define $X := b a_j^\top$ so each component becomes the **convex** quadratic

$$
f_j(X) = \mathbb{E}_{Z_0,w_\star}\big[\langle G,X\rangle + w_\star[j]\big]^2 .
$$

Its gradient is $\nabla f_j(X) = 2\,\mathbb{E}[\langle G,X\rangle G] + 2\,\mathbb{E}[w_\star[j]\,G]$. Two moment computations finish it:

- **Cross term.** With $G = \tfrac1n\sum_i\begin{bmatrix} x^{(i)}x^{(i)\top} & y^{(i)}x^{(i)} \\ y^{(i)}x^{(i)\top} & y^{(i)2}\end{bmatrix}$ and $w_\star$ symmetric ($w_\star \stackrel{d}{=} -w_\star$, so odd moments vanish), one gets $\mathbb{E}[w_\star[j]\,G] = E_{d+1,j} + E_{j,d+1}$, where $E_{i,i'}$ is the single-one indicator matrix.
- **Quadratic term.** Computing $\mathbb{E}[\langle G, E_{d+1,j}\rangle G]$ using 4th-order Gaussian moments of $x$ yields, after collecting, that the minimizer is $X_j = -\big(\tfrac{n-1}{n} + (d+2)\tfrac1n\big)^{-1} E_{d+1,j}$.

Since $f_j$ is convex, $\nabla f_j(X_j)=0$ certifies the global optimum. Stacking over $j$ recovers the isotropic optimum

$$
Q_0 = -\frac{1}{\frac{n-1}{n} + (d+2)\frac1n}\begin{bmatrix} I_d & 0 \\ 0 & 0\end{bmatrix},\qquad
P_0 = \begin{bmatrix} 0_{d\times d} & 0 \\ 0 & 1\end{bmatrix}.
$$

Up to the rescaling gauge freedom ($P_0\!\to\!\gamma P_0,\ Q_0\!\to\!\gamma^{-1}Q_0$), these are exactly von Oswald et al.'s one-step-GD parameters.

### Non-isotropic optimum and its interpretation

For $x^{(i)}\sim N(0,\Sigma)$ with $\Sigma=U\Lambda U^\top$, $\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_d)$, and $w_\star\sim N(0,I)$, the optimal preconditioner is

$$
Q_0 = -\begin{bmatrix} U\,\mathrm{diag}\!\Big(\frac{1}{\frac{n+1}{n}\lambda_i + \frac{1}{n}(\sum_k\lambda_k)}\Big)_{i}\,U^\top & 0 \\ 0 & 0\end{bmatrix}.
$$

Two structural readings:
- **As $n\to\infty$**, the top-left block $\to \Sigma^{-1}$: the transformer preconditions by the inverse covariance, accelerating convergence when $\Sigma$ is ill-conditioned. This is the "adapts to input distribution" property.
- **The $\tfrac1n\sum_k\lambda_k$ term** in each denominator acts as a **Tikhonov regularizer** whose strength grows when $n$ is small or the covariate variance is high — "adapts to the variance induced by data inadequacy," echoing structural risk minimization (Vapnik).

### Multi-layer: forward pass = preconditioned GD (Lemma 1)

Under the sparsity constraint

$$
P_i = \begin{bmatrix} 0_{d\times d} & 0 \\ 0 & 1\end{bmatrix},\qquad
Q_i = -\begin{bmatrix} A_i & 0 \\ 0 & 0\end{bmatrix},\quad A_i\in\mathbb{R}^{d\times d},
$$

Lemma 1 proves $[Z_\ell]_{(d+1),(n+1)} = -\langle x^{(n+1)}, w_\ell^{\mathrm{gd}}\rangle$ where $w_0^{\mathrm{gd}}=0$ and

$$
w_{\ell+1}^{\mathrm{gd}} = w_\ell^{\mathrm{gd}} - A_\ell\,\nabla R_{w_\star}\big(w_\ell^{\mathrm{gd}}\big),\qquad
R_{w_\star}(w) := \frac{1}{2n}\sum_{i=1}^n (w^\top x_i - w_\star^\top x_i)^2 .
$$

So each layer is one preconditioned GD step on the in-context empirical risk, with layer-dependent preconditioner $A_\ell$. This is the bridge that turns "landscape of transformer weights" into "search over $L$-step gradient algorithms."

- **Theorem 2 (2-layer, symmetric).** For isotropic $x,w_\star\sim N(0,I)$, the global optimum uses **diagonal** $A_1,A_2$ — i.e., GD with coordinate-wise adaptive step sizes (Adagrad-like), but the step sizes are tuned to the *distribution*, not the instance.
- **Theorem 3 (deep, sparse).** For $x\sim N(0,\Sigma)$, $w_\star\sim N(0,\Sigma^{-1})$ (the "distorted-view" regression scenario $\tilde x = W x$, $\Sigma=WW^\top$), the set $S = \{A_i = a_i\Sigma^{-1}\ \forall i\}$ satisfies $\inf_{A\in S}\sum_i\|\nabla_{A_i}f\|_F^2 = 0$, i.e., $S$ essentially contains critical points. This is a **Newton-like / full-matrix Adagrad** preconditioner $A_i\propto\Sigma^{-1}=\mathbb{E}[XX^\top]^{-1}$. Unlike Theorem 1, there is *no* robustness trade-off here because $w_\star$ already carries covariance $\Sigma^{-1}$. Proof technique: rewrite the in-context loss as a matrix polynomial in the layer weights; exploit distributional invariances to construct a *flow contained in $S$* whose objective decreases as fast as gradient flow; since $f$ is lower-bounded, $S$ must contain points of arbitrarily small gradient. (Subtlety: the infimum may not be attained, so $S$ contains near-critical points, not necessarily an exact zero-gradient point.)
- **Theorem 4 (deep, relaxed).** With the richer parameterization $P_i=\begin{bmatrix}B_i&0\\0&1\end{bmatrix},\ Q_i=\begin{bmatrix}A_i&0\\0&0\end{bmatrix}$ (non-symmetric $A_i,B_i$), the critical set is $\{A_i=a_i\Sigma^{-1},\ B_i=b_i I\}$. Here $A_i$ preconditions the gradient while $B_i$ **reshapes the covariates** each iteration to improve the Gram-matrix conditioning: the layer-$k$ feature update is $X_{k+1} = X_k + B_k X_k M X_k^\top A_k X_k \approx X_k(I - |a_k b_k| M X_k^\top X_k)$ — curvature correction. When $\Sigma=I$ this reduces exactly to von Oswald et al.'s **GD++**. Experiments show $\|A_0\|\le\|A_1\|\le\|A_2\|$: small step early (ill-conditioned) then larger step once $B$'s have improved conditioning.
- **Theorem 5 (ReLU attention).** With $\sigma=$ ReLU applied entrywise inside attention, a single-layer global minimizer is still characterized (isotropic data), showing the story is not brittle to the exact linearity of attention.

**Note for the project.** The core transferable idea is that *the algorithm a trained conditioner implements is dictated by the training task distribution*, and that distributional statistics (covariance, sample-size-dependent regularization) get **baked into the weights** as a preconditioner. Any code implications (e.g., using a linear-attention proxy to study our modulatory conditioning) should be routed to `senior-developer` for an `issue_plan`; this review does not plan code.

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** Distinguishes expressivity ("a transformer *can* implement GD") from the learning question ("does *training* find it?"). Positions the paper as the first loss-landscape analysis of learning algorithms via training linear transformers on random linear regression.
- **§1.1 Related works.** Turing-completeness of RNNs (Siegelmann–Sontag), neural Turing machines, algorithmic power of transformers, hidden-layer GD interpretation (Jastrzebski et al.), and concurrent single-layer analyses (Zhang et al. 2023 = paper #3 here; Mahankali et al. 2023). Their Theorem 1 structure matches Zhang et al.'s independently-derived global optimum.
- **§2 Setting.** Random linear-regression data distribution; input matrix $Z_0$; softmax-free linear self-attention $\mathrm{Attn}_{P,Q}(Z)=PZM(Z^\top QZ)$; $L$-layer residual stack; in-context loss $f$. Table 1 maps each theorem to its data model and guarantee (global minimizer vs critical point).
- **§3 Single-layer global optimum.** Theorem 1 (non-isotropic $\Sigma$, isotropic $w_\star$); the isotropic special case (7); interpretation of the preconditioner as $\Sigma^{-1}$ + sample-size regularizer.
- **§4 Multi-layer, sparse parameters.** Sparsity condition (8); Lemma 1 (forward pass = preconditioned GD); §4.1 Theorem 2 (2-layer symmetric → diagonal adaptive-step GD); §4.2 Theorem 3 (deep → $A_i\propto\Sigma^{-1}$ critical point, Newton-like); §4.3 experimental validation on 3-layer nets ($d=5,n=20$) showing $\mathrm{Dist}(\Sigma^{1/2}A_i\Sigma^{1/2}, I)\to0$.
- **§5 Beyond standard optimization.** Relaxed parameterization (11); Theorem 4 (critical set $A_i\propto\Sigma^{-1}$, $B_i\propto I$ = GD++); §5.1 experiments confirming $B_i\to I$, $A_i\to\Sigma^{-1}$, and increasing step-size norms across layers.
- **§6 Discussion.** Future directions: nonlinear attention; Theorem 5 (ReLU single-layer global minimizer).
- **Appendices A–D.** A: single-layer proofs (loss rewriting, isotropic warm-up via convex per-coordinate components $f_j$, non-isotropic, ReLU). B: multi-layer proofs (Theorem 2/3/4), including the flow-in-$S$ argument. C: auxiliary lemmas, incl. proof of Lemma 1. D: experiment details / weight visualizations.

---
