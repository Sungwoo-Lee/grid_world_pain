> **Per-paper review — in-context-learning corpus, paper 24 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§24); content is identical. Manifest: [[in_context_learning_sources]].

# 24. Kang et al. 2026 — Transformers Can Learn Posterior Predictive Distributions In-Context

**PDF:** `docs/project/references/in_context_learning/sources/Kang et al. 2026 - Transformers Can Learn Posterior Predictive Distributions In-Context.pdf`
**Status:** arXiv preprint (arXiv:2605.26713v1, 26 May 2026); to appear ICML 2026 (PMLR 306). Authors: Gyeonghun Kang, Changwoo J. Lee, Xiang Cheng (Duke).

## Phase 1: Foundational Overview (Undergraduate-Level)

**The problem in plain words.** PFNs (prior-data fitted networks) empirically produce astonishingly good uncertainty estimates: their 95% predictive intervals often match the truth. But *why* can a transformer do this? Previous theory only showed transformers can compute a **point prediction** (the posterior predictive *mean*) by implicitly running gradient descent in their forward pass. Predicting a *full distribution* is harder — you need the mean **and** the variance **and** a way to emit an actual probability density. This paper supplies the missing theory, for the tractable case of **Gaussian process (GP) regression** (where the true posterior predictive distribution is known in closed form, so you can check the transformer against ground truth).

**The core idea (a constructive proof).** Kang et al. *build*, by hand, an explicit set of transformer weights that:
1. Uses **self-attention to run an iterative solver** (Richardson iteration) that computes the GP posterior predictive **mean $\mu$ and variance $\tau$** — the moments.
2. Uses a **shallow MLP + softmax head to convert those two moments into a binned (discretized) probability distribution** — the actual density output, exactly how real PFNs emit predictions.

They then prove **error bounds**: the approximation error shrinks **exponentially in the number of attention layers $L$** (more depth = more solver iterations = tighter moments) and like **$1/C$ in the number of bins $C$** (finer discretization). Finally they explain a real, mysterious PFN behavior — **generalizing to context sizes bigger than those seen in pre-training** ("$n$-generalization") — and show that **attention normalization** (the softmax denominator) is what makes it work, acting as a **preconditioner** that keeps the solver stable as the dataset size grows.

**Key findings.**
- **Theorem 3.1:** attention implements the moment-computing recursion with error $\lesssim \exp(-(1-\rho)L)$ — exponential convergence in depth.
- **Theorem 4.1:** total end-to-end error $\text{TV}(p_n, q_\vartheta) \lesssim e^{-(1-\rho)L} + \delta_{MLP} + \tfrac{1}{C} + \varepsilon_{\text{tail}}$ — cleanly decomposed into attention/MLP/binning/truncation pieces.
- **Theorem 5.1–5.3:** without normalization the required solver step size scales like $1/n$, so a single fixed model breaks outside its pre-training range; **normalized attention = Jacobi preconditioning** fixes the spectrum independent of $n$; but bigger $n$ still needs **linearly more depth** ($L\gtrsim n\log(1/\epsilon)$ for the RBF kernel) because the Gram matrix becomes ill-conditioned.
- Simulations confirm all three: TV error drops with depth and bins; unnormalized models spike outside the pre-training range while normalized ones stay flat; deeper + wider-pretraining-range models generalize better.

**Initial takeaway.** This is the shard's **theory anchor**. It explains, mechanistically, *how* a transformer can be a **posterior-predictive-distribution estimator** (not just a point estimator) — bridging the von-Oswald/Akyürek "attention = gradient descent" line to the distributional PFN world, and giving concrete architectural prescriptions (depth, normalization, bin count) grounded in numerical-linear-algebra convergence theory.

## Phase 2: Graduate-Level Deep Dive

### 4.1 Setup: PFN output as a discretized PPD, and its population minimizer

A PFN is a transformer with $L$ self-attention blocks + an MLP head emitting $C$ logits $\ell = (\ell_1,\dots,\ell_C)$. Fixing a partition $\Gamma = \{a=\gamma_1<\dots<\gamma_{C+1}=b\}$ of an interval capturing all but $\varepsilon$ mass, the output is a **piecewise-constant density**:
$$q_\vartheta(y\mid x,\mathcal{D}_n) = \sum_{c=1}^C \frac{\mathbf{1}_{y\in(\gamma_c,\gamma_{c+1}]}}{\Delta_c}\cdot\frac{\exp(\ell_c)}{\sum_{c'}\exp(\ell_{c'})}, \qquad \Delta_c := \gamma_{c+1}-\gamma_c. \tag{2}$$
Training minimizes the truncated NLL $\mathbb{E}[-\log q_\vartheta(y\mid x,\mathcal{D}_n)]$, equivalent to minimizing the expected KL to the true PPD. The **population minimizer** $q^*$ is the truncated-and-discretized true PPD:
$$q^*(y\mid x,\mathcal{D}_n) = \sum_{c=1}^C \mathbf{1}_{y\in(\gamma_c,\gamma_{c+1}]}\cdot\frac{\int_{\gamma_c}^{\gamma_{c+1}} p_n(t\mid x,\mathcal{D}_n)\,dt}{\Delta_c(1-\varepsilon)}, \tag{3}$$
and for Lipschitz-smooth $p_n$, $\mathbb{E}[\text{TV}(p_n,q^*)] = O(|\Gamma|) + \varepsilon$ (binning + truncation error only).

**Architecture abstraction.** Tokens are columns $[Z^{(0)}]_{:,j} = [x_j^\top, y_j]^\top$ ($j\le n$) with the query token $[x_{n+1}^\top, 0]^\top$ (masked label). Layer update with residual, masked attention, and a *learnable linear skip* $S^{(l)}$:
$$Z^{(l+1)} = Z^{(l)} + S^{(l)}Z^{(l)} + \text{Attn}(Z^{(l)}; V^{(l)},K^{(l)},Q^{(l)}), \qquad \ell = W_2\,\text{act}(W_1[Z^{(L)}]_{:,n+1} + h_1) + h_2. \tag{4,5}$$

**GP target (closed form).** For $y_i = \phi(x_i)+\epsilon_i$, $\epsilon_i\sim\mathcal{N}(0,\sigma^2)$, GP prior on $\phi$ with kernel $\kappa$, the PPD is $\mathcal{N}(y;\mu(Z),\tau(Z))$ with
$$\mu(Z) = k_x^\top(G+\sigma^2 I_n)^{-1}Y, \qquad \tau(Z) = \kappa(x,x)+\sigma^2 - k_x^\top(G+\sigma^2 I_n)^{-1}k_x,$$
where $[G]_{ij}=\kappa(x_i,x_j)$ and $[k_x]_i = \kappa(x_i,x)$. **Both moments reduce to evaluating $k_x^\top(G+\sigma^2 I_n)^{-1}v$ for $v\in\{Y, k_x\}$** — the single linear-solve primitive attention must implement.

### 4.2 Theorem 3.1 — attention runs a Richardson iteration for the moments

**KRR view.** $u^*(x) = k_x^\top(G+\sigma^2 I_n)^{-1}v$ is the kernel-ridge-regression solution: $u^* = \arg\min_{u\in\mathcal{H}} \sum_i (v_i - u(x_i))^2 + \sigma^2\|u\|_\mathcal{H}^2$, and $u^*_X$ solves $(G+\sigma^2 I_n)u^*_X = Gv$ by the representer theorem.

**Richardson iteration** for this linear system (componentwise, $u^{(0)}\equiv 0$):
$$u^{(l+1)}(x_j) = (1-\eta^{(l)}\sigma^2)\,u^{(l)}(x_j) + \eta^{(l)}\sum_{i=1}^n \kappa(x_i,x_j)\big(v_i - u^{(l)}(x_i)\big). \tag{7}$$
If $0<\eta^{(l)}<2/(\lambda_1(G)+\sigma^2)$ for all $l$ and $\sum_l\eta^{(l)}=\infty$, then $u^{(L)}(x)\to k_x^\top(G+\sigma^2 I_n)^{-1}v$ as $L\to\infty$.

**Mapping to attention.** Define **unnormalized attention** $\text{Attn}_{M,\kappa}(Z;V,K,Q)_{:,j} = \sum_i \kappa(Kz_i,Qz_j)M_{ij}Vz_i$. The recursion (7) maps exactly: the **decay term $-\eta^{(l)}\sigma^2 u^{(l)}$ is implemented by the learnable skip $S^{(l)}$**, and the **kernel-weighted correction $\eta^{(l)}\sum_i\kappa(x_i,x_j)(v_i-u^{(l)}(x_i))$ is the self-attention** over context tokens. Running *both* right-hand sides $v=Y$ (for $\mu$) and $v=k_x$ (for the variance-reduction term) in parallel (stacked token dimensions) yields both moments.

**Convergence bound (Theorem 3.1):**
$$\big\| \text{TF}_{\vartheta,L} - (\mu,\tau)^\top\big\|_\infty \;\lesssim\; \exp\big(-(1-\rho)L\big), \qquad \rho := 1-\eta(\lambda_n(G)+\sigma^2)\in(0,1),$$
i.e. the transformer inherits the Richardson iteration's **geometric (exponential-in-depth) convergence**; $\rho$ is the convergence factor and $\lambda_n(G)$ the smallest Gram eigenvalue.

### 4.3 Theorem 4.1 — from moments to a density (the distributional step)

For a 1-D exponential-family target $f_\theta(y) = \exp(\langle\psi(\theta),T(y)\rangle - A(\theta))h(y)$ (Gaussian: $\psi(\mu,\tau)=(\mu/\tau,\,-1/(2\tau))$, $T(y)=(y,y^2)$), the MLP head approximates the natural-parameter map $\psi$ by universal approximation ($\|\psi-\tilde\psi\|\le\delta_{MLP}$), and a **readout matrix $\Xi$ with rows $T(\xi_c)$** (sufficient statistics at bin midpoints $\xi_c$) produces logits $(\Xi\tilde\psi(\theta))_c = \langle\tilde\psi(\theta),T(\xi_c)\rangle$. **Softmax then removes the log-normalizer $A(\theta)$ automatically** (it is an additive constant across bins), giving the piecewise-constant density. This is the key insight explaining *why real PFN MLP heads work*: the head weights encode the sufficient statistics per bin.

**End-to-end bound (Theorem 4.1):**
$$\text{TV}(p_n, q_\vartheta) \;\lesssim\; \underbrace{e^{-(1-\rho)L}}_{\text{attention/solver}} + \underbrace{\delta_{MLP}}_{\text{natural-param MLP}} + \underbrace{\tfrac{1}{C}}_{\text{binning}} + \underbrace{\varepsilon_{\text{tail}}(Z^{(0)})}_{\text{truncation}}.$$
In the practical regime where $C$, MLP width, and truncation interval are chosen large, **the attention depth is the binding constraint** — motivating §5.

### 4.4 Theorems 5.1–5.3 — why normalization and depth govern $n$-generalization

The transformer is a **single-weight solver** that must approximately solve $(G+\sigma^2 I_n)u = Gv$ for *any* $n$ within a *fixed* depth $L$.

- **Theorem 5.1 (step size $\sim 1/n$).** With $x_i\sim\mathcal{N}(0,I_d/d)$, for linear and RBF kernels $\lambda_1(G)=\Theta(n)$ w.h.p., so the admissible step size bound $\eta<2/(\lambda_1(G)+\sigma^2) = \Theta(1/n)$. A step size $\hat\eta$ tuned to $[n_{\min},n_{\max}]$ is **too conservative for $n'\ll n_{\min}$** (under-converges in $L$ steps) and **violates the bound for $n'\gg n_{\max}$** (diverges). This is the mechanism behind PFN failure outside the pre-training range.
- **Normalization = Jacobi preconditioning (Theorem 5.1 remedy).** Left-multiply by $D^{-1}$, $D=\text{diag}(G\mathbf{1}_n)$: $D^{-1}(G+\sigma^2 I_n)u = D^{-1}Gv$, same fixed point. Since $D^{-1}G$ is row-stochastic, $\lambda_1(D^{-1}(G+\sigma^2 I_n))\in[1,1+\sigma^2]$ — **bounded independent of $n$**, so a fixed step size works for all $n$. The preconditioned recursion
$$u^{(l+1)}_{pr}(x) = \Big(1-\eta\tfrac{s_x}{\sigma^2}\Big)u^{(l)}_{pr}(x) + \tfrac{\eta}{s_x}\sum_i\kappa(x_i,x)\big(v_i - u^{(l)}_{pr}(x_i)\big), \quad s_x=\textstyle\sum_i\kappa(x_i,x), \tag{14}$$
is **implemented by normalized (softmax-style) attention** $\text{Attn}^{na}_{M,\kappa}$ whose weights are divided by their sum $s_j$, plus a token-wise $1/s_{x_j}$ scaling of the skip term. So the ubiquitous softmax denominator in real transformers *is* the preconditioner enabling $n$-generalization — a striking mechanistic result.
- **Theorem 5.2 & 5.3 (bigger $n$ needs more depth).** Even preconditioned, $\text{cond}(D^{-1}(G+\sigma^2 I_n)) = \Theta(n)$ a.s. for RBF (Gram increasingly ill-conditioned as eigenvalues decay geometrically). The optimal convergence factor $\rho^* = 1-2/(\text{cond}(G+\sigma^2 I_n)+1)\to 1$ as $n$ grows, so to hit tolerance $\epsilon$ it suffices that $L\gtrsim n\log(1/\epsilon)$ — **required depth grows ~linearly in $n$**. Hence deeper stacks (and wider pre-training ranges exposing more eigenspectra) are essential for large-context PFNs — matching the empirical TabPFN-2.5/3 trend of 18–24 layers.

### 4.5 Relation to the shard's axis, and limitations

Kang is the **distribution-estimator theory** the two Mittal papers lack: it proves attention *can* emit a calibrated PPD, not merely a point. But it is honest that these are **constructive existence results** (like von Oswald / Akyürek): they show the architecture *can* realize the mechanism, not that standard PFN pre-training *does* discover it. The learnable-relaxation experiments (diagonal $Q,K$, Richardson sparsity pattern) partially bridge existence→learning by reproducing the predicted depth/bin/normalization effects. Scope: GP regression (closed-form PPD), with sketched extensions to hierarchical GPs (multi-head = one head per mixture component) and latent-GP/non-Gaussian models.

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** PFNs approximate the PPD via ICL; strong empirical UQ (95% intervals match truth; BayesOpt acquisition). Gap: no theory for *distributional* (not point) ICL. Prior theory = "attention implements gradient descent" for point prediction (Akyürek, von Oswald, Bai, Ahn, Zhang, Mahankali). Motivated by depth (TabPFN-2.5/3 use 18–24 layers), bin resolution, and $n$-generalization (TabPFNv2 pretrained to 2048, works to 10000). Three contributions: explicit construction + error bounds (Thm 3.1, 4.1); $n$-generalization mechanism via spectra/normalization (Thm 5.1–5.3); empirical validation.
- **§2 Background and Problem Setup.** PPD definition (eq 1); PFN objective as expected-KL minimization; discretized output (eq 2) and population minimizer $q^*$ (eq 3, TV bound). Transformer architecture (tokens, masked attention, learnable skip, eqs 4–5). GP regression closed-form $\mu,\tau$ reducing to $k_x^\top(G+\sigma^2 I)^{-1}v$.
- **§3 Attention Computes Moments of PPD.** KRR + representer theorem; Richardson iteration (eq 7) + convergence condition. Unnormalized attention definition (eq 8). Construction implementing both recursions in parallel; **Theorem 3.1** (exponential-in-$L$ convergence, factor $\rho$). Remark 3.2 (hierarchical-GP multi-head extension).
- **§4 From Moments to PPD.** Exponential-family densities; MLP for natural-parameter map (universal approximation, $\delta_{MLP}$); softmax discretization with sufficient-statistic readout $\Xi$ (removes $A(\theta)$). **Theorem 4.1** (TV bound = attention + MLP + $1/C$ + tail). Interpretation: depth controls moment convergence, $C$ controls discretization; justifies binned-MLP PFN practice.
- **§5 Generalizing Beyond Pretrain Sample Sizes.** Pretraining over $n\in[n_{\min},n_{\max}]$. Step size decreases with $n$ (system eq 11); **Theorem 5.1** ($\eta\sim 1/n$; two failure modes). Jacobi preconditioning (eq 13) bounds spectrum; **normalized attention implements it** (eq 14, $\text{Attn}^{na}$). **Theorem 5.2** ($\text{cond}=\Theta(n)$ for RBF); **Theorem 5.3** ($L\gtrsim n\log(1/\epsilon)$ depth requirement). Remark 5.4 (learnable models more flexible than the construction).
- **§6 Numerical Studies.** BLR (linear kernel) + RBF regression; theory/learnable/normalized parameterizations. §6.1 validates Thm 4.1 (Figure 2: log-TV linear in log-bins, power-law in log-depth). §6.2 normalized vs unnormalized (Figure 4: unnormalized spikes OOD, normalized stable). §6.3 validates Thm 5.3 (Figure 5: depth + pretraining-range improve MSE/coverage/width; iteration-limited).
- **§7 Discussion.** Constructive account for GP PPDs; extensions (hierarchical GP mixtures, latent-GP/GP-classification). Limitation: existence results, not a proof that pretraining discovers them; open problems on optimization guarantees.

---
