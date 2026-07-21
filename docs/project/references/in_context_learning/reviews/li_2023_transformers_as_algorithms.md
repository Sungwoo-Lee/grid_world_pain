> **Per-paper review — in-context-learning corpus, paper 14 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§14); content is identical. Manifest: [[in_context_learning_sources]].

# 14. Li et al. 2023 — Transformers as Algorithms: Generalization and Stability in In-Context Learning

**PDF:** `docs/project/references/in_context_learning/sources/Li et al. 2023 - Transformers as Algorithms.pdf`
**Full title:** *Transformers as Algorithms: Generalization and Stability in In-context Learning* (arXiv:2301.07067v2).

## Phase 1: Foundational Overview (Undergraduate-Level)

**The question in plain terms.** The other two papers in this shard ask *which* algorithm a trained transformer implements. Li et al. ask a complementary statistical question: **how much data do you need to train a good in-context learner, and why is a transformer a statistically sound way to learn an algorithm at all?** They formalize ICL as an **algorithm-learning problem** — the transformer is abstracted as a map from a data sequence to a prediction function — and prove finite-sample generalization bounds for the whole meta-learning pipeline.

**Setup.** During training ("multitask learning", MTL) the model sees $T$ tasks, each supplying a length-$n$ sequence fed **auto-regressively**: the model predicts example $m$ from the prompt of the first $m-1$ examples, for every $m$. Two prompt types are covered: (1) i.i.d. (input, label) pairs (supervised learning), and (2) **trajectories of a dynamical system** $x_{m+1}=f(x_m)+\text{noise}$ (so the prompt is temporally *dependent*, unlike prior ICL theory).

**Key findings.**
- **Generalization rate $1/\sqrt{nT}$.** The excess MTL risk decays like $\sqrt{\dim(\mathcal A)/(nT)}$ — you benefit both from more tasks $T$ *and* from more examples per task $n$. The hard part is the $1/\sqrt n$ factor: examples within one prompt are temporally coupled (each influences all later predictions), so a naive bound would lose the $n$. The key is **algorithmic stability** — if one in-context example only changes future predictions by $O(1/m)$, a martingale (Azuma–Hoeffding) argument recovers the $\sqrt n$.
- **Transformers are provably stable.** A $D$-layer transformer with bounded weights changes its output by at most $\frac{2}{2m-1}((1+\Gamma)e^\Gamma)^D$ when a non-final in-context token is swapped — meeting the $K/m$ stability condition. The proof hinges on a tight Lipschitz analysis of the softmax self-attention layer.
- **Transfer (unseen tasks) is governed by task complexity, not model size.** Empirically, transfer risk for $d$-dimensional linear regression aligns perfectly when plotted against $(n/d,\,T/d^2)$ — it depends on the *task* dimension $d$, not on the transformer's parameter count $\dim(\mathcal A)$ (verified across GPT-2 sizes spanning $64\times$ parameter counts). This is an **inductive-bias** phenomenon: MTL pretraining implicitly selects an algorithm lying in the span of the source tasks.
- **ICL as model selection.** With a stronger $\times n$ factor from Theorem 3.5, ICL can adaptively pick the right hypothesis class per sample size $m$ (bias–variance trade-off), achieving $\sqrt{\log H / T}$ instead of paying for the union of all classes.

**Initial takeaway.** Where Ahn/Zhang describe *what* algorithm emerges, Li et al. supply the *statistical-learning-theory scaffolding*: stability ⇒ generalization ⇒ meta-learning transfer, with matching experiments. The stability–generalization bridge is the technical heart.

## Phase 2: Graduate-Level Deep Dive

### Problem formalization

A length-$m$ prompt is $x^{(m)}_{\text{prompt}}=(z_1,\dots,z_{m-1}, x_m)$. The transformer is abstracted as an algorithm $\mathrm{Alg}\in\mathcal A$ mapping a subsequence $S^{(m)}$ to a prediction function $f^{\mathrm{Alg}}_{S^{(m)}}:\mathcal X\to\mathcal Y$, with the explicit identification

$$
\mathrm{TF}\big(x^{(m+1)}_{\text{prompt}}\big) = f^{\mathrm{Alg}}_{S^{(m)}}(x_{m+1}),\qquad
f^{\mathrm{Alg}}_{S^{(m)}}(x) := \mathrm{TF}((S^{(m)}, x)).
$$

Training is empirical risk minimization over algorithms:

$$
\widehat{\mathrm{Alg}} = \arg\min_{\mathrm{Alg}\in\mathcal A}\widehat L_{S_{\text{all}}}(\mathrm{Alg}),\quad
\widehat L_{S_{\text{all}}} = \frac1T\sum_{t=1}^T\widehat L_t,\quad
\widehat L_t = \frac1n\sum_{i=1}^n \ell\big(y_{ti}, f^{\mathrm{Alg}}_{S^{(i-1)}_t}(x_{ti})\big).
$$

The target is the **excess MTL risk** $R_{\mathrm{MTL}}(\widehat{\mathrm{Alg}}) = L_{\mathrm{MTL}}(\widehat{\mathrm{Alg}}) - \min_{\mathrm{Alg}} L_{\mathrm{MTL}}(\mathrm{Alg})$.

### Algorithmic stability (Assumption 3.1)

Let $S^j$ be $S$ with its $j$-th sample replaced. **Error stability** with constant $K$:

$$
\Big|\mathbb{E}_{(x,y)}\big[\ell(y, f^{\mathrm{Alg}}_S(x)) - \ell(y, f^{\mathrm{Alg}}_{S^j}(x))\big]\Big| \le \frac{K}{m}.
$$

The stronger **pairwise** version bounds the *difference of differences* between two algorithms by $K\rho(\mathrm{Alg},\mathrm{Alg}')/m$, enabling chaining. The $1/m$ scaling is the classical order (Bousquet–Elisseeff) needed for realistic generalization.

### Theorem 3.2 — transformers satisfy stability (with derivation of the self-attention Lipschitz bound)

For a $D$-layer transformer $X^{(i)}=\mathrm{Parallel\_MLPs}(\mathrm{ATTN}(X^{(i-1)}))$ with $\mathrm{ATTN}(X)=\mathrm{softmax}(XWX^\top)XV$, unit-ball tokens, $\|V\|\le1$, $\|W\|\le\Gamma/2$, and $1$-Lipschitz MLPs, swapping a non-final token $j<m$ gives

$$
\big|\mathrm{TF}(x^{(m)}_{\text{prompt}}) - \mathrm{TF}(x'^{(m)}_{\text{prompt}})\big| \le \frac{2}{2m-1}\big((1+\Gamma)e^{\Gamma}\big)^D,
$$

hence $K = 2L_\ell((1+\Gamma)e^\Gamma)^D$. The exponential depth dependence is tolerable (GPT-2/BERT have 12–48 layers). Two tightness facts: stability *fails* if $\Gamma$ is allowed to grow like $\log m$, and it *fails* if the perturbed token is the last one ($j<m$ is essential).

**The engine — softmax Lipschitzness (Lemma B.1).** For $x,x+\varepsilon\in\mathbb{R}^n$ with $\|x\|_\infty,\|x+\varepsilon\|_\infty\le c$,

$$
\|\mathrm{softmax}(x)\|_\infty \le \frac{e^{2c}}{n},\qquad
\|\mathrm{softmax}(x)-\mathrm{softmax}(x+\varepsilon)\|_1 \le \frac{e^{2c}\|\varepsilon\|_1}{n}.
$$

*Proof:* the $\ell_\infty$ bound follows from $\frac{e^c}{e^c + \sum_{i\ge2}e^{-c}}\le \frac{e^{2c}}{n}$. For the perturbation, the softmax Jacobian is $\mathrm{diag}(s)-ss^\top$ with $s=\mathrm{softmax}(x)$; bounding $\|(\mathrm{diag}(s)-ss^\top)\varepsilon\|_1\le e^{2c}\|\varepsilon\|_1/n$ and integrating $\delta:0\to1$ along $x+\delta\varepsilon$ gives the claim.

**Self-attention stability (Lemma B.2).** For $A=\mathrm{softmax}(XWX^\top)XV$ and its perturbed twin $\bar A$, decompose the output difference $P=\bar A - A$ into a **softmax-shift** term and a **value-input-shift** term:

$$
P = \underbrace{[\mathrm{softmax}(\bar X W\bar X^\top)-\mathrm{softmax}(XWX^\top)]XV}_{P_1} + \underbrace{\mathrm{softmax}(\bar X W\bar X^\top)\,EV}_{P_2}.
$$

Using Lemma B.1 with $|x_i^\top W x_j|\le\Gamma$: $\|P_2\|_{2,1}\le e^{2\Gamma}\|E\|_{2,1}$, and (via a $\delta\to0$ derivative argument splitting the two-sided perturbation into left/right pieces, each contributing $\Gamma e^{2\Gamma}$) $\|P_1\|_{2,1}\le 2\Gamma e^{2\Gamma}\|E\|_{2,1}$. Together,

$$
\|\bar A - A\|_{2,1} \le (2\Gamma+1)\,e^{2\Gamma}\,\|E\|_{2,1}.
$$

Iterating this contraction-like bound over $D$ layers (Lemma B.3 single-layer + Theorem B.4 multi-layer, with the MLP contraction $\|M\|\le1$ folding in) produces the $((1+\Gamma)e^\Gamma)^D$ factor, and the $1/(2m-1)$ comes from the per-token normalization ($\|\varepsilon_i\|\le C_0/n$ giving the individual-output bound $\|\bar\varepsilon_i\|\le\tfrac1n(2\Gamma+1)e^{2\Gamma}C_0$).

### Theorem 3.5 — the generalization bound

With $\mathcal A$ being $K$-stable, $\ell$ $L$-Lipschitz and $[0,1]$-valued, and covering number $\mathcal N(\mathcal A,\rho,\varepsilon)$ under the algorithm distance $\rho(\mathrm{Alg},\mathrm{Alg}')=\sup_S\frac1n\sum_i\|f^{\mathrm{Alg}}_{S^{(i-1)}}(x_i)-f^{\mathrm{Alg}'}_{S^{(i-1)}}(x_i)\|$, with prob. $\ge1-2\delta$:

$$
R_{\mathrm{MTL}}(\widehat{\mathrm{Alg}}) \le \inf_{\varepsilon>0}\Bigg\{4L\varepsilon + 2(1 + K\log n)\sqrt{\frac{\log(\mathcal N(\mathcal A,\rho,\varepsilon)/\delta)}{c\,nT}}\Bigg\}.
$$

For Lipschitz architectures $\log\mathcal N\sim\dim(\mathcal A)\log(1/\varepsilon)$, so up to logs $R_{\mathrm{MTL}}\lesssim\sqrt{\dim(\mathcal A)/(nT)}$. The stronger pairwise-stable bound (4) replaces the covering term with Dudley's entropy integral (chaining), matching Rademacher complexity at $nT$ samples.

**Proof sketch (why $1/\sqrt n$ survives temporal dependence).** Fix an algorithm and define the Doob martingale $X_{t,i}=\mathbb{E}\big[\frac1n\sum_j\ell(y_{tj}, f^{\mathrm{Alg}}_{S^{(j-1)}_t}(x_{tj}))\,\big|\,S^{(i)}_t\big]$. Revealing example $i$ changes predictions $i{+}1,\dots,n$; by stability each contributes $\le K/j$, so the martingale increment obeys $|X_{t,i}-X_{t,i-1}|\lesssim 1 + \sum_{j=i}^n K/j \lesssim 1 + K\log n$. Azuma–Hoeffding on $\frac1T\sum_t(X_{t,0}-X_{t,n})=|L_{\mathrm{MTL}}-\widehat L_{S_{\text{all}}}|$ then gives concentration at rate $(1+K\log n)/\sqrt{nT}$; a covering/chaining union bound makes it uniform over $\mathcal A$. With $M$ independent sequences per task, the rate improves to $1/\sqrt{nMT}$.

### Dynamical systems (Section 5)

The same machinery extends to prompts that are trajectories $x_{m+1}=f(x_m)+w_m$. Two stability notions combine: **system stability** (Def 5.1, $(C_\rho,\rho)$: $\|f^{(m)}(x_0)-f^{(m)}(x_0')\|\le C_\rho\rho^m\|x_0-x_0'\|$, exponential forgetting of initial condition) and **algorithmic stability for dynamics** (Assumption 5.3, a Lipschitz-style variant summing perturbations over $i=j,\dots,m$). Theorem 5.4 recovers Theorem 3.5's bound with $K$ replaced by $\bar K = 2K\frac{\bar C_\rho}{1-\bar\rho}(\bar w + \bar x/\sqrt n)$: the system's exponential stability controls how a noise perturbation propagates down the trajectory.

### Transfer learning & inductive bias (Section 4)

Transfer risk decays as $1/\mathrm{poly}(T)$ (unseen tasks induce distribution shift, unfixable by larger $n$ or $M$): $L_{\mathrm{TFR}}-L_{\mathrm{MTL}}\lesssim\sqrt{\log\mathcal N(\mathcal A,\rho,\varepsilon)/T}$. The empirical surprise: transfer risk for $d$-dim linear regression collapses onto a single curve in $(n/d, T/d^2)$ and is *independent* of $\dim(\mathcal A)$. The Bayes-optimal comparator is weighted ridge $\hat\beta=(X^\top X+\sigma^2\Sigma^{-1})^{-1}X^\top y$; estimating the task covariance $\hat\Sigma=\frac1T\sum\beta_i\beta_i^\top$ to spectral accuracy needs only $T=\Omega(d)$, yet ICL empirically needs $T\propto d^2$ (entrywise $\ell_\infty$ accuracy) — so ICL is *not* sample-optimal in $T$, but its inductive bias makes it insensitive to architecture size.

### Model selection (Section 6)

Under Hypothesis 1 (the TF can $\varepsilon^{h,m}_{\mathrm{TF}}$-approximate ERM over each class $\mathcal F_h$), ICL adaptively selects, per sample size $m$, the class minimizing $L^\star_h + \mathcal O(R_m(\mathcal F_h)) + \varepsilon^{h,m}_{\mathrm{TF}}$ (Observation 1), rather than paying the Rademacher complexity of the union $\bigcup_h\mathcal F_h$. With $H$ classes over $n$ sample sizes there are $H^n$ effective ERM algorithms (VC-dim $\sim n\log H$), and Theorem 3.5's $\times n$ factor turns the excess risk into $\sqrt{n\log H/(nT)}=\sqrt{\log H/T}$ — the extra $n$ *pays for* per-sample-size adaptivity.

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** ICL formalized as algorithm learning; three contributions: MTL generalization bounds ($1/\sqrt{nT}$), transformer stability, MTL→transfer inductive bias. Figure 2: ICL matches Bayes-optimal weighted ridge on noisy/anisotropic linear regression and beats fixed-memory least-squares on partially-observed LDS.
- **§1.1 Related work.** Garg et al. (function-class ICL), AutoML view, Bayesian-inference view (Xie et al.), GD-view (von Oswald/Akyürek), time-series learning, algorithmic stability (Bousquet–Elisseeff). Distinguishes: finite-sample bounds + temporally-dependent prompts + no mixing assumptions.
- **§2 Problem Setup.** ICL prompt notation; algorithm-hypothesis abstraction $f^{\mathrm{Alg}}_{S^{(m)}}$; MTL ERM objective; excess MTL and transfer risks.
- **§3 Generalization.** §3.1 stability (Assumption 3.1 error + pairwise; Theorem 3.2 transformer stability with $((1+\Gamma)e^\Gamma)^D/(2m-1)$ bound; tightness). §3.2 covering numbers (Def 3.3), algorithm distance (Def 3.4), Theorem 3.5 (bounds (3),(4)); martingale/Azuma proof sketch; multiple-sequences $1/\sqrt{nMT}$.
- **§4 Transfer / inductive bias.** $1/\mathrm{poly}(T)$ transfer decay; $(n/d, T/d^2)$ alignment; architecture-independence; weighted-ridge comparator; source–target distance predicts transfer (Fig 5).
- **§5 Dynamical systems.** $(C_\rho,\rho)$-stability (Def 5.1); Assumption 5.2/5.3; Theorem 5.4 (bound with $\bar K$).
- **§6 Model selection.** Hypothesis 1; Observation 1; adaptive class selection; $\sqrt{\log H/T}$ benefit of the $\times n$ factor.
- **§7 Numerical evaluations.** GPT-2 (12L/8H/256d); linear regression, LDS; stability improves with prompt length; noisy-data training aids stability.
- **Appendices A–F.** A: extra experiments (architecture-independence across tiny/small/standard GPT-2; idealized comparators). B: stability proofs (Lemmas B.1–B.7, Theorem B.4). C: MTL + transfer proofs. D: dynamical proof. E: model-selection formalization. F: related work.

---
