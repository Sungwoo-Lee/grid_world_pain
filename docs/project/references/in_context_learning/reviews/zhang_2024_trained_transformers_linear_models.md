> **Per-paper review — in-context-learning corpus, paper 15 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§15); content is identical. Manifest: [[in_context_learning_sources]].

# 15. Zhang et al. 2024 — Trained Transformers Learn Linear Models In-Context

**PDF:** `docs/project/references/in_context_learning/sources/Zhang et al. 2024 - Trained Transformers Learn Linear Models In-Context.pdf`
**Venue:** JMLR / arXiv:2306.09927v3.

## Phase 1: Foundational Overview (Undergraduate-Level)

**The question in plain terms.** Ahn et al. characterize the *global minimum* of the training loss (where does the good solution *sit*?). Zhang et al. answer the strictly harder dynamical question: **starting from a random initialization, does gradient descent actually reach that good solution, despite the loss being non-convex?** And once there, **how robust is the learned in-context learner to distribution shift?**

**Setup.** A single linear self-attention (LSA) layer is trained by **gradient flow** (gradient descent with infinitesimal step size) on the *population* loss of random anisotropic linear-regression prompts. Covariates $x_i\sim N(0,\Lambda)$ (possibly badly scaled), task weight $w\sim N(0,I_d)$, labels $y_i=\langle w, x_i\rangle$.

**Key findings.**
- **Global convergence despite non-convexity (Theorem 4.1).** With a "balanced" random initialization, gradient flow provably converges to a **global minimum** of the population loss, and the paper writes down the limiting weights in closed form.
- **The learned algorithm (Corollary 4.3).** At convergence the prediction is $\hat y_{\text{query}} = x_{\text{query}}^\top\,\Gamma^{-1}\big(\frac1M\sum_i y_i x_i\big)$ with $\Gamma = (1+\tfrac1N)\Lambda + \tfrac{\mathrm{tr}(\Lambda)}{N}I_d$ — one step of **preconditioned** GD, and because the preconditioner is $\approx\Lambda^{-1}$ it handles *anisotropic* data where plain one-step GD fails. Prediction error is $O(1/M + 1/N^2)$: it decays faster in the *training* prompt length $N$ than the *test* length $M$, and does *not* vanish unless $N\to\infty$ (finite training prompts leave irreducible error).
- **Robust to task & query shift, brittle to covariate shift (§4.2).** The learned learner tolerates label noise and nonlinear tasks (it computes the best linear predictor over the test prompt) and query-distribution changes — but if the test covariate distribution differs from training (e.g. features scaled by $c$), the prediction is off by exactly $c^2$: the transformer is **not** implementing scale-invariant OLS.
- **Even training on random covariances doesn't fix it (§4.3, Theorem 4.5).** If you vary $\Lambda$ across prompts, gradient flow still converges to a global minimum, but the single-LSA learner still fails covariate shift — e.g. with $\lambda_i\sim\mathrm{Exp}(1)$ the prediction shrinks to $\tfrac13\langle w, x_{\text{query}}\rangle$. Larger nonlinear GPT-2 models generalize better under covariate shift, especially trained on random covariances (but still imperfectly, and they spike when $M>N$).

**Initial takeaway.** This is the "does training actually get there" companion to Ahn et al. — an end-to-end convergence proof plus a careful **robustness audit** exposing that the elegant one-step-preconditioned-GD story is fragile: the learned algorithm secretly depends on the training covariance and is *not* covariate-shift invariant. That fragility is a cautionary tale for any conditioning mechanism trained on a fixed input statistic.

## Phase 2: Graduate-Level Deep Dive

### Model and initialization

The LSA layer (softmax removed, value/projection merged into $W^{PV}$, key/query merged into $W^{KQ}$) is

$$
f_{\mathrm{LSA}}(E;\theta) = E + W^{PV} E\,\frac{E^\top W^{KQ} E}{\rho},\qquad
E = \begin{bmatrix} x_1 & \cdots & x_N & x_{\text{query}} \\ y_1 & \cdots & y_N & 0\end{bmatrix},\ \rho=N.
$$

The prediction is the bottom-right entry. Only the last row of $W^{PV}$ and first $d$ columns of $W^{KQ}$ matter, so with the block split $W^{PV}=\begin{bmatrix}W^{PV}_{11}&w^{PV}_{12}\\(w^{PV}_{21})^\top&w^{PV}_{22}\end{bmatrix}$, $W^{KQ}=\begin{bmatrix}W^{KQ}_{11}&w^{KQ}_{12}\\(w^{KQ}_{21})^\top&w^{KQ}_{22}\end{bmatrix}$,

$$
\hat y_{\text{query}} = \big[(w^{PV}_{21})^\top\ \ w^{PV}_{22}\big]\,\frac{EE^\top}{N}\begin{bmatrix}W^{KQ}_{11}\\(w^{KQ}_{21})^\top\end{bmatrix}x_{\text{query}}.
$$

**Balanced initialization (Assumption 3.3):** $W^{PV}(0)=\sigma\begin{bmatrix}0&0\\0&1\end{bmatrix}$, $W^{KQ}(0)=\sigma\begin{bmatrix}\Theta\Theta^\top&0\\0&0\end{bmatrix}$ with $\|\Theta\Theta^\top\|_F=1$. This "balancedness" (borrowed from deep-linear-network theory) keeps the two effective layers matched along the whole trajectory and is what makes the non-convex flow tractable.

### Reduction to rank-one matrix factorization (Lemma 5.1)

The prediction is a **quadratic form** in the parameters:

$$
\hat y_{\text{query}}(E_\tau;\theta) = u^\top H_\tau u,\qquad
H_\tau = \tfrac12 X_\tau\otimes\frac{E_\tau E_\tau^\top}{N},\quad
X_\tau=\begin{bmatrix}0_{d\times d}&x_{\tau,\text{query}}\\ x_{\tau,\text{query}}^\top&0\end{bmatrix},
$$

with $u=\mathrm{Vec}(U)$, $U=\begin{bmatrix}U_{11}&u_{12}\\ u_{21}^\top& u_{-1}\end{bmatrix}$ collecting the active parameter blocks. So the loss is $\widehat L = \frac1{2B}\sum_\tau(u^\top H_\tau u - w_\tau^\top x_{\tau,\text{query}})^2$ — a rank-one matrix-factorization problem. **Non-convexity is explicit:** by the Kronecker eigenvalue rule, $\mathrm{eig}(H_\tau)=\mathrm{eig}(\tfrac12 X_\tau)\cdot\mathrm{eig}(E_\tau E_\tau^\top/N)$; the characteristic polynomial

$$
\det(\mu I - X_\tau) = \mu^{d-1}\big(\mu^2 - \|x_{\tau,\text{query}}\|_2^2\big)
$$

gives $X_\tau$ one **negative** eigenvalue a.s., so $H_\tau$ has $\ge d+1$ negative eigenvalues and $u^\top H_\tau u$ is non-convex.

### Gradient-flow dynamical system (Lemma 5.2)

Under balanced init, $u_{12}(t)=u_{21}(t)=0$ for all $t$, and the surviving coordinates obey the coupled ODEs

$$
\frac{d}{dt}U_{11} = -u_{-1}^2\,\Gamma\Lambda U_{11}\Lambda + u_{-1}\Lambda^2,\qquad
\frac{d}{dt}u_{-1} = -\mathrm{tr}\big[u_{-1}\Gamma\Lambda U_{11}\Lambda U_{11}^\top - \Lambda^2 U_{11}^\top\big],
$$

with $\Gamma=(1+\tfrac1N)\Lambda + \tfrac1N\mathrm{tr}(\Lambda)I_d$. These are exactly gradient flow on the reduced objective

$$
\tilde\ell(U_{11},u_{-1}) = \mathrm{tr}\Big[\tfrac12 u_{-1}^2\,\Gamma\Lambda U_{11}\Lambda U_{11}^\top - u_{-1}\Lambda^2 U_{11}^\top\Big],
$$

which equals $L$ up to a parameter-independent constant.

### Global minima and the learned algorithm (Lemma 5.3)

Any global minimizer of $\tilde\ell$ satisfies

$$
u_{-1}U_{11} = \Gamma^{-1} \approx_{N\to\infty} \Lambda^{-1}.
$$

Substituting back, the trained network's prediction on a fresh prompt is

$$
\hat y_{\text{query}} = \frac1M\sum_{i=1}^M y_i\, x_i^\top\Gamma^{-1}x_{\text{query}}
= w^\top\Big(\tfrac1M\sum_i x_i x_i^\top\Big)\Gamma^{-1}x_{\text{query}} \approx w^\top x_{\text{query}},
$$

i.e. **one step of $\Gamma^{-1}$-preconditioned GD**, which — unlike plain one-step GD — succeeds for anisotropic $\Lambda\ne I$ because $\Gamma^{-1}\approx\Lambda^{-1}$ cancels the Gram matrix $\frac1M\sum x_ix_i^\top\approx\Lambda$.

### PL inequality ⇒ convergence (Lemma 5.4, Theorem 4.1)

Even though $\tilde\ell$ is non-convex, balanced init keeps the trajectory in a region where a **Polyak–Łojasiewicz (PL)** inequality holds:

$$
\|\nabla\tilde\ell(U_{11}(t), u_{-1}(t))\|_2^2 \ge \mu\big(\tilde\ell(U_{11}(t),u_{-1}(t)) - \min\tilde\ell\big),\quad
\mu = \frac{\sigma^2\sqrt d\,\|\Lambda\|_{op}^2\,\mathrm{tr}(\Gamma^{-1}\Lambda^{-1})\,\mathrm{tr}(\Lambda^{-1})}{\|\Lambda\Theta\|_F^2\,[2-\sqrt d\,\sigma^2\|\Gamma\|_{op}]} > 0,
$$

valid when the initialization scale is small, $\sigma^2\|\Gamma\|_{op}\sqrt d < 2$. PL ⇒ linear convergence of the loss and, in the limit,

$$
u_{-1}(t)\to\|\Gamma^{-1}\|_F^{1/2},\qquad U_{11}(t)\to\|\Gamma^{-1}\|_F^{-1/2}\Gamma^{-1}.
$$

Translating back yields the closed-form global weights (Theorem 4.1):

$$
W^{KQ}_\ast = \begin{bmatrix}\mathrm{tr}(\Gamma^{-2})^{-1/4}\,\Gamma^{-1}&0\\0&0\end{bmatrix},\quad
W^{PV}_\ast = \begin{bmatrix}0_{d\times d}&0\\0&\mathrm{tr}(\Gamma^{-2})^{1/4}\end{bmatrix},\quad
\Gamma = (1+\tfrac1N)\Lambda + \tfrac{\mathrm{tr}(\Lambda)}{N}I_d.
$$

(For $\Lambda=I$ this matches von Oswald et al.'s construction up to the gauge scaling and the $(1+(d+1)/N)^{-1}$ factor.)

### Prediction error (Theorem 4.2)

For a test distribution $D$ over $(x,y)$ with $x\sim N(0,\Lambda)$, defining $a:=\Lambda^{-1}\mathbb{E}[xy]$ and $\Sigma:=\mathbb{E}[(xy-\mathbb{E}(xy))(xy-\mathbb{E}(xy))^\top]$,

$$
\mathbb{E}(\hat y_{\text{query}}-y_{\text{query}})^2 = \underbrace{\min_{w}\mathbb{E}(\langle w,x_{\text{query}}\rangle - y_{\text{query}})^2}_{\text{best linear predictor}}
+ \frac1M\mathrm{tr}(\Sigma\Gamma^{-2}\Lambda)
+ \frac1{N^2}\big[\|a\|^2_{\Gamma^{-2}\Lambda^3} + 2\,\mathrm{tr}(\Lambda)\|a\|^2_{\Gamma^{-2}\Lambda^2} + \mathrm{tr}(\Lambda)^2\|a\|^2_{\Gamma^{-2}\Lambda}\big].
$$

Two structural reads: (i) excess error is $O(1/M + 1/N^2)$ — faster in $N$ than $M$; (ii) even $M\to\infty$ leaves an $O(1/N^2)$ gap: **finite training-prompt length is an irreducible bottleneck.** For noiseless well-specified linear data with $w\sim N(0,I)$ and condition number $\kappa$, this collapses to $\le \frac{(d+1)\mathrm{tr}(\Lambda)}{M} + \frac{(1+2d+d^2\kappa)\mathrm{tr}(\Lambda)}{N^2}$ (Corollary 4.3).

### Distribution-shift audit (§4.2) — the fragility

From the universal identity $\hat y_{\text{query}}\approx x_{\text{query}}^\top\Lambda^{-1}(\frac1M\sum_i y_i x_i)$:
- **Task shift (tolerated).** Noisy labels $y_i=\langle w,x_i\rangle+\varepsilon_i$: the noise term $x_{\text{query}}^\top\Lambda^{-1}(\frac1M\sum\varepsilon_i x_i)\to0$ (mean-zero, independent), so $\hat y\approx\langle w, x_{\text{query}}\rangle$ for *arbitrary* $w$. Nonlinear tasks → best linear predictor.
- **Query shift (tolerated).** With $D^{\text{train}}_x=D^{\text{test}}_x$ and large $M$, $\hat y\approx x_{\text{query}}^\top\Lambda^{-1}\Lambda w = x_{\text{query}}^\top w$ for very general query distributions.
- **Covariate shift (NOT tolerated).** If test covariates are scaled by $c$, $\frac1M\sum x_ix_i^\top\approx c^2\Lambda$, so $\hat y\approx c^2 x_{\text{query}}^\top w \ne x_{\text{query}}^\top w$. OLS would be scale-invariant; the transformer is not — proving the *implemented algorithm differs from OLS* even where predictions superficially match.

### Random-covariance training doesn't rescue it (§4.3, Theorem 4.5)

Train with $\Lambda_\tau$ drawn i.i.d. (diagonal, positive, finite 3rd moment). Gradient flow still converges to a global minimum with limiting

$$
W^{KQ}_\ast \propto (\mathbb{E}\Gamma_\tau\Lambda_\tau^2)^{-1}(\mathbb{E}\Lambda_\tau^2),\qquad
\Gamma_\tau = \tfrac{N+1}{N}\Lambda_\tau + \tfrac1N\mathrm{tr}(\Lambda_\tau)I_d.
$$

For a fresh $\Lambda_{\text{new}}$, the prediction tends (as $M,N\to\infty$) to $x_{\text{query}}^\top(\mathbb{E}\Lambda_\tau^2)(\mathbb{E}\Lambda_\tau^3)^{-1}(\mathbb{E}\Lambda_\tau)\,w$; the factor $(\mathbb{E}\Lambda_\tau^2)(\mathbb{E}\Lambda_\tau^3)^{-1}(\mathbb{E}\Lambda_\tau)\ne I$ in general. With $\lambda_{\tau,i}\sim\mathrm{Exp}(1)$ ($\mathbb{E}\Lambda=I,\ \mathbb{E}\Lambda^2=2I,\ \mathbb{E}\Lambda^3=6I$), $\mathbb{E}\hat y_{\text{query}}\to\frac{2\cdot1}{6}\langle w,x_{\text{query}}\rangle=\tfrac13\langle w,x_{\text{query}}\rangle$ — a systematic $\tfrac13$ shrinkage. A single LSA layer *cannot* in-context-learn linear models across covariate distributions.

**Empirical coda.** Nonlinear GPT-2 trained on random covariances generalizes better under covariate shift (fails only at the largest scale $c=9$), but exhibits an error **spike when $M>N$** (test prompt longer than training), conjectured to stem from GPT-2's absolute positional encodings (concurrent work reports removing them helps).

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** ICL as parameter-free supervised learning; Garg et al.'s function-class framing; the open gap — *how does gradient-based training produce ICL?* Four contributions: global convergence for single-LSA; characterization of the learned algorithm & error; distribution-shift robustness; the random-covariance failure + nonlinear-GPT2 experiments.
- **§2 Related work.** GD-view (von Oswald/Akyürek), Bayesian-inference view (Xie et al.), implicit-finetuning (Dai et al.), kernel-regression (Han et al.), approximation-theoretic (Yun et al., Bai et al.), gradient-dynamics results (Jelassi et al., Li–Li–Risteski); explicit comparison to concurrent Ahn et al. (landscape vs. dynamics).
- **§3 Preliminaries.** Notation (Kronecker $\otimes$, Vec); Def 3.1 (trained on in-context examples), Def 3.2 (in-context learning of a class up to error $\eta$); softmax self-attention → LSA simplification $f_{\mathrm{LSA}}=E+W^{PV}E\,E^\top W^{KQ}E/\rho$; token embedding (3.4); prediction reduces to block components (3.6); training procedure (population loss (3.8), gradient flow (3.9)); Assumption 3.3 (balanced init).
- **§4 Main results.** §4.1 Theorem 4.1 (convergence + limiting weights); prediction identities (4.2)/(4.3); Theorem 4.2 (error decomposition, $O(1/M+1/N^2)$); Corollary 4.3. §4.2 distribution shifts (task/query tolerated, covariate not). §4.3 Def 4.4 (random covariate distributions); Theorem 4.5 (convergence + $\tfrac13$-shrinkage failure); nonlinear GPT-2 experiments (Fig 1, $M>N$ spike).
- **§5 Proof ideas.** §5.1 reduction to quadratic $u^\top H_\tau u$ / rank-one factorization + non-convexity via $X_\tau$'s negative eigenvalue (Lemma 5.1). §5.2 gradient-flow ODEs + reduced objective $\tilde\ell$ (Lemma 5.2), global-minima condition $u_{-1}U_{11}=\Gamma^{-1}$ (Lemma 5.3). §5.3 PL inequality ⇒ global convergence + limiting values (Lemma 5.4).
- **§6 Conclusion.** Summary; robustness to task/query but brittleness to covariate shift; open questions (alternative inits, deeper nets, positional-encoding spike).
- **Appendices A–E.** A: proof of Theorem 4.1 (Lemmas 5.1–5.4 in full). B: proof of Theorem 4.2. C: proof of Theorem 4.5 (random covariance). D: auxiliary results. E: experiment details.

---
