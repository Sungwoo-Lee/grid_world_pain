> **Per-paper review — in-context-learning corpus, paper 27 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§27); content is identical. Manifest: [[in_context_learning_sources]].

# 27. Raventós et al. 2023 — Pretraining Task Diversity and the Emergence of Non-Bayesian In-Context Learning for Regression

**PDF:** `docs/project/references/in_context_learning/sources/Raventos et al. 2023 - Pretraining task diversity and non-Bayesian in-context learning.pdf`
**Authors / venue:** Allan Raventós*, Mansheej Paul*, Feng Chen, Surya Ganguli (Stanford). NeurIPS 2023. arXiv:2306.15063.

## Phase 1: Foundational Overview (Undergraduate-Level)

**The question.** Can ICL solve *genuinely new* tasks — ones very different from anything in pretraining — or is it just "retrieving" pretraining tasks (Bayesian inference over the pretraining prior)? The answer turns on **how many distinct tasks the model saw during pretraining** (task diversity), and the paper finds a sharp **threshold**.

**The clean testbed.** Instead of messy language, they study **in-context linear regression**. A "task" is a hidden weight vector $w$; the prompt gives example pairs $(x_1,y_1),\dots,(x_k,y_k)$ with $y=w^\top x + \text{noise}$, and the transformer must predict $y$ for a new $x$. The trick: pretrain on a **finite pool of $M$ tasks** (M distinct $w$'s), then *test on brand-new $w$'s* drawn from the full Gaussian. Because it's linear regression, the two "ideal answers" are known in closed form:
- **dMMSE** (discrete minimum-MSE): the Bayes-optimal predictor whose prior is exactly the finite pretraining pool. It's great on pretraining tasks, bad on new ones.
- **Ridge regression:** the Bayes-optimal predictor whose prior is the full Gaussian over *all* tasks. It solves new tasks optimally.

So they can literally ask: does the trained transformer behave like dMMSE (memorized the pool) or like Ridge (learned to solve any task)?

**Key finding — the task-diversity threshold.**
- **Low diversity (small $M$, up to ~$2^6$):** the transformer $\approx$ dMMSE. It nails pretraining tasks but **fails on new tasks** — it is Bayesian *with respect to the narrow pretraining prior*.
- **High diversity (beyond a threshold between $2^{14}$ and $2^{15}$ tasks for the base model):** the transformer $\approx$ **Ridge** — it solves fundamentally new tasks optimally, and it does so by **deviating from** the Bayes-optimal-on-pretraining estimator (dMMSE). This deviation is the whole point: ICL of new tasks is **non-Bayesian** relative to the pretraining distribution.
- A **phase transition:** below threshold, training on *more data* (more sequences per task) pushes the model *toward* dMMSE (more overfit to the pool); above threshold, more data pushes it *toward* Ridge. In the limit of infinite sequences-per-task the crossover becomes a sharp jump.

**More findings.** (1) The threshold grows only *linearly* with problem dimension $D$ (despite the task volume growing exponentially), and the transformer's error is nearly $D$-independent while dMMSE degrades badly — the transformer scales far better than pure pretraining-Bayes. (2) **Weight decay lowers** the threshold (regularization helps ICL emerge with fewer tasks, at some accuracy cost); more layers / bigger embeddings *raise* the threshold (for small models). (3) The transformer even beats a *smoothed* dMMSE (Gaussian-blurred pool prior), so its success is not merely "couldn't memorize the pool." (4) A break in the scaling law of the early-stopping time $t^*\propto M^{\alpha}$ ($\alpha\approx0.47$ below threshold) confirms a genuine change in *learning dynamics*, not just underfitting.

**Initial takeaway.** ICL of new tasks is an **emergent, non-Bayesian** phenomenon that a theory of "Bayesian inference on the pretraining distribution" *cannot* fully explain. You need **task diversity above a threshold** — and scale of data alone (below threshold) actively *hurts*. This complements Chan et al.: Chan identifies *which distributional properties* switch ICL on; Raventós quantifies *how much task variety* is needed and shows the learned algorithm departs from the naive Bayesian prior. For the project this is the sharpest statement of "diversity, not just scale, drives generalizing in-context adaptation."

## Phase 2: Graduate-Level Deep Dive

**Data-generating process.** A task is a latent $w\in\mathbb{R}^D$. Inputs $x_k\sim\mathcal N(0,I_D)$ i.i.d.; targets $y_k = w^\top x_k + \varepsilon_k$, $\varepsilon_k\sim\mathcal N(0,\sigma^2)$. A decoder-only causal transformer $f_\theta$ sees context $S_k = (x_1,y_1,\dots,x_{k-1},y_{k-1},x_k)$ and predicts the target of $x_k$; each forward pass thus solves $K$ nested regressions (increasing context) with the same $w$.

**Two task distributions.**
- Pretraining: $T_{\text{Pretrain}} = U\{w^{(1)},\dots,w^{(M)}\}$, a *uniform* distribution over a **finite** pool, each $w^{(i)}\sim\mathcal N(0,I_D)$. Diversity is $M$.
- Ideal: $T_{\text{True}} = \mathcal N(0,I_D)$ over *all* $w$.

**Pretraining objective** (MSE averaged over the $K$ predictions):
$$ \mathcal{L}_{T_{\text{Pretrain}}}(\theta) = \mathbb{E}_{\substack{w\sim T_{\text{Pretrain}} \\ x_{1:K}\sim\mathcal N(0,I_D)\\ \varepsilon_{1:K}\sim\mathcal N(0,\sigma^2)}} \left[ \frac1K \sum_{k=1}^K \big(f_\theta(S_k) - y_k\big)^2 \right]. \tag{1} $$

**The optimal estimator is the Bayesian posterior mean.** Derivation (Appendix A.1): using the law of total expectation, the $k$-th loss term is
$$ \mathbb{E}\big[(f_\theta(S_k)-y_k)^2\big] = \mathbb{E}_{S_k}\,\mathbb{E}_{w,\varepsilon_k}\big[(f_\theta(S_k)-y_k)^2 \mid S_k\big], $$
minimized pointwise by the conditional mean $\hat y_k^{T} = \mathbb{E}_{T,\varepsilon_k}[y_k\mid S_k]$. Expanding the posterior over $w$:
$$ \mathbb{E}[y_k\mid S_k] = \int\! dw\,dy_k\; y_k\, p(y_k\mid x_k,w)\,p(w\mid X,y) = \int\! dw\; (w^\top x_k)\, p(w\mid X,y) \equiv \hat w_k^\top x_k, $$
with the **posterior-mean weight**
$$ \hat w_k = \mathbb{E}[w\mid S_k] = \frac{\int dw\; p(w)\, w\,\prod_{i=1}^{k-1} p(y_i\mid x_i,w)}{\int dw\; p(w)\,\prod_{i=1}^{k-1} p(y_i\mid x_i,w)}. \tag{6} $$
Here $X=(x_1^\top,\dots,x_{k-1}^\top)\in\mathbb R^{(k-1)\times D}$, $y=(y_1,\dots,y_{k-1})$.

**dMMSE — plug the discrete prior into Eq. (6).** With $p(w)=U\{w^{(1)},\dots,w^{(M)}\}$ and Gaussian likelihood $p(y_j\mid x_j,w)\propto\exp\!\big(-\tfrac{1}{2\sigma^2}(y_j-w^\top x_j)^2\big)$:
$$ \hat w_k^{\text{dMMSE}} = \sum_{i=1}^{M} \frac{\exp\!\Big(-\tfrac{1}{2\sigma^2}\sum_{j=1}^{k-1}(y_j - w^{(i)\top}x_j)^2\Big)} {\sum_{l=1}^{M}\exp\!\Big(-\tfrac{1}{2\sigma^2}\sum_{j=1}^{k-1}(y_j - w^{(l)\top}x_j)^2\Big)}\; w^{(i)}, \tag{2} $$
with $\hat w_1^{\text{dMMSE}} = \tfrac1M\sum_i w^{(i)}$. This is a **softmax-weighted average of the pool vectors**, weight $\propto$ likelihood of the observed context under each candidate task — a *soft nearest-task retrieval*. It cannot produce a $w$ outside the convex hull of the pool, hence fails on new tasks.

**Ridge — plug the Gaussian prior into Eq. (6).** With $p(w)=\mathcal N(0,I_D)$, the posterior is Gaussian; completing the square in the exponent (Appendix A.3),
$$ -\tfrac{1}{\sigma^2}(Xw-y)^\top(Xw-y) - w^\top w \;=\; -\tfrac{1}{\sigma^2}\big(w-\mu\big)^\top(X^\top X + \sigma^2 I_D)\big(w-\mu\big) + \text{const}, $$
so the posterior mean is the ridge solution with ridge parameter $=\sigma^2$:
$$ \hat w_k^{\text{Ridge}} = \big(X^\top X + \sigma^2 I_D\big)^{-1} X^\top y, \qquad \hat w_1^{\text{Ridge}} = 0. \tag{3} $$
(Note the paper's Eq. 8 writes $X^\top y$; the completing-the-square step uses $\mu=(X^\top X+\sigma^2 I)^{-1}X^\top y$.) This estimator generalizes to *any* $w$, hence is optimal on $T_{\text{True}}$.

**Behavioral distance metric.** To locate the transformer between the two estimators, they measure the mean-squared prediction gap under a test distribution $T$:
$$ \Delta^{T}_{\text{PT},\,\text{Ridge/dMMSE}} = \mathbb{E}_{\substack{w\sim T\\ x_{1:K},\,\varepsilon_{1:K}}} \left[ \frac{1}{KD}\sum_{k=1}^K \Big(f_\theta(S_k) - \hat y_k^{\text{Ridge/dMMSE}}\Big)^2 \right]. \tag{4} $$

**The phase-transition logic (the crux).** Increasing sequences-per-task (via batch size $256\to512\to1024$ at fixed 500k steps, or via steps $500\text{k}\to1\text{M}$ at fixed batch) probes what solution training *converges toward*:
- For $M\le 2^{10}$: $\Delta^{T_{\text{Pretrain}}}_{\text{PT,dMMSE}}$ *decreases* — more data ⇒ transformer moves **toward dMMSE** (overspecializes to the pool). Predictions also move *away* from Ridge.
- Beyond the threshold ($\sim$ between $2^{14}$ and $2^{15}$ for base PT): more data ⇒ both $\Delta^{T_{\text{Pretrain}}}_{\text{PT,Ridge}}$ and $\Delta^{T_{\text{True}}}_{\text{PT,Ridge}}$ *decrease* — the transformer moves **toward Ridge** on *all* tasks, and gets *better* at new tasks the more it trains, despite the finite pool.

Because the direction of motion (toward dMMSE vs toward Ridge) flips sign across $M^*$, the smooth crossover in the left column of Fig 2 must **sharpen into a true discontinuity as sequences-per-task $\to\infty$**. That the two independent ways of adding data (bigger batch vs more steps) give the *same* threshold, and that behavior is invariant to batch/step trade-offs at fixed sequences-per-task (Appendix D), pins sequences-per-task as the controlling variable.

**Learning-dynamics evidence it is not mere underfitting.** Define $t^*$ = the step where $\Delta^{T_{\text{True}}}_{\text{PT,Ridge}}$ is minimized (the "early-stopping time for Ridge"). For $M<M^*$, distance-to-Ridge dips then rises (the model transits *through* a Ridge-like solution en route to dMMSE), and $t^*$ obeys a power law
$$ t^* \propto M^{\alpha}, \qquad \alpha \approx 0.47 \ \ (\text{linear fit of } \log t^* \text{ vs } \log M). $$
For $M>2^{10}$ (small PT), distance-to-Ridge decreases *monotonically*, $t^*=2M$ steps — a **break in the scaling law**. Extending to 2M steps (4×) does not shift the threshold, and above-threshold learning curves match the $M=\infty$ model. **Linear-mode-connectivity** probes reinforce this: PTs trained with $M\gtrsim 2^{13}$ share a loss basin with $M=\infty$ (low loss barrier), whereas $M<2^{13}$ models are separated by large barriers — i.e. large-$M$ solutions live near the $T_{\text{True}}$-optimal solution in weight space.

**Interpolation-path diagnostic.** For pairs of seen tasks $w_i,w_j$, construct new tasks on a norm-preserving interpolation path:
$$ w_\alpha = \tfrac12\big(\lVert w_i\rVert_2 + \lVert w_j\rVert_2\big)\; \frac{\alpha w_i + (1-\alpha)w_j}{\lVert \alpha w_i + (1-\alpha)w_j\rVert_2}, \qquad \alpha\in[0,1]. \tag{5} $$
The norm-fixing avoids $\lVert w_\alpha\rVert$ collapsing near $\alpha=\tfrac12$. At the path center ($\alpha=\tfrac12$, tasks *farthest* from the pool): at $M=2^5$ the PT tracks dMMSE (fails); at $M=2^{10}$ it beats dMMSE but not yet Ridge; at $M=2^{15}$ it matches Ridge even at the center. dMMSE never solves center tasks at any $M$ in range.

**Dimension scaling.** Varying $D\in\{8,16,32\}$ at fixed SNR (scaling $\sigma^2$ and context $K=2D$), the threshold rises only **~linearly** ($\approx 2^{14},2^{15},2^{16}$) — remarkable because the Gaussian $T_{\text{True}}$ concentrates on a sphere whose covering number grows *exponentially* in $D$. At $M=2^{20}$, $\Delta^{T_{\text{True}}}_{\text{PT,Ridge}}$ is nearly $D$-independent while $\Delta^{T_{\text{True}}}_{\text{dMMSE,Ridge}}$ grows sharply with $D$: the PT's scaling **vastly outperforms** dMMSE's.

**Regularization / capacity.** Weight decay over three orders of magnitude **monotonically lowers** the threshold (Fig 7 left) — but with worse absolute performance (Fig 13) — hinting that *implicit* regularization drives the transition even without explicit weight decay. Increasing embedding dim has no effect on base-PT threshold but raises it for small PT; increasing depth raises the small-PT threshold. Base PT (higher capacity) has a higher threshold and is capacity-insensitive; small PT is still in a capacity-sensitive regime. Interpretation: capacity, up to a point, *increases* the diversity needed to trigger the transition.

**Robustness of the "learns the generative prior" claim.** Beyond threshold the PT matches the Bayes-optimal estimator for the *underlying generative model of tasks* — shown for both Gaussian and **Laplace** priors (Fig 14) — even though lower-training-loss (dMMSE-like) solutions exist. So implicit regularization steers the PT to the *true generative prior*, not the empirical pool.

**Relation to the shard / project.** This is the mechanistic, closed-form counterpart to Chan et al.: it proves (in a controlled setting where the Bayesian ideal is computable) that generalizing ICL **requires task diversity above a threshold** and is **non-Bayesian w.r.t. the pretraining prior**. For any project that wants "learn to adapt in-context to unseen conditions," the actionable message is: *diversify the task/context distribution past threshold; scaling data at low diversity is counterproductive.* It also validates the Xie et al. implicit-Bayesian-inference story only in the low-diversity regime, and refutes it as a *complete* account.

## Appendix: Section-by-Section Backbone

- **Abstract.** ICL on linear regression vs pretraining task diversity. Below a diversity threshold → transformer = Bayesian estimator with the *pretraining* prior (fails new tasks). Above → matches Ridge (Gaussian prior over all tasks), solving new tasks optimally by *deviating* from the pretraining-Bayes optimum. Explores regularization, capacity, task structure.
- **§1 Introduction.** ICL emerges from next-token pretraining, not built into architecture. Xie et al.'s Bayesian-retrieval hypothesis. But $T_{\text{Pretrain}}$ is an unrepresentative subsample of the desired $T_{\text{True}}$; would predict suboptimality on far tasks. Study linear regression because optimal estimators are computable. **Contributions:** (i) low diversity ⇒ dMMSE-like, high ⇒ Ridge-like; (ii) a threshold + phase transition in sequences-per-task limit; (iii) threshold scales ~linearly with $D$, PT error nearly $D$-independent; (iv) weight decay lowers threshold, depth/width raise it. ICL not fully explained by pretraining-Bayes.
- **§2 Problem setup.** Task = latent $w$; $x\sim\mathcal N(0,I_D)$, $y=w^\top x+\varepsilon$. Causal transformer predicts each target from preceding context (Eq 1 loss). $T_{\text{Pretrain}}=U\{w^{(1..M)}\}$; $T_{\text{True}}=\mathcal N(0,I_D)$. Optimal = posterior mean. dMMSE (Eq 2, softmax over pool). Ridge with ridge $=\sigma^2$ (Eq 3). Distance metric $\Delta$ (Eq 4).
- **§3 Experiments.** Base GPT-2-style (8L, 128-dim, 2 heads) or small (4L, 64-dim); $D=8$, $K=16$, $\sigma^2=0.25$; Adam, one-cycle LR, batch 256, 500k steps. §3.1 Threshold: low $M$ (≤$2^6$) → dMMSE, fail new tasks; high $M$ → Ridge, solve new tasks non-Bayesianly (Figs 2–3). Finite-size scaling of sequences-per-task ⇒ algorithmic phase transition (threshold $2^{14}$–$2^{15}$). Learning dynamics $t^*\propto M^{0.47}$ then breaks (Fig 4). Interpolation paths (Eq 5, Fig 5). PT beats *smoothed* dMMSE (Fig 12). §3.2 Threshold rises ~linearly with $D$; PT $D$-robust, dMMSE not (Fig 6). §3.3 Weight decay ↓ threshold; depth/embedding ↑ threshold (small PT) (Fig 7).
- **§4 Related work.** Builds on Xie et al. Bayesian ICL (validated only at low diversity). Prior linear-regression ICL (Garg, Akyürek, von Oswald) used *unlimited* diversity. Gradient-descent-in-forward-pass hypothesis. Meta-learning diversity literature. Kirsch et al. (diversity→ICL on classification). Chan et al. (burstiness/rare classes). Olsson et al. (induction heads).
- **§5 Discussion.** Algorithmic phase transition dMMSE→Ridge at intermediate diversity; threshold scales moderately with $D$; implicit regularization lets PT escape the pretraining prior. Beyond threshold PT learns the *true generative prior* (Gaussian & Laplace). Linear-mode-connectivity: large-$M$ PTs share basin with $M=\infty$. Open: what "tasks"/diversity means for language; scale alone insufficient (below threshold, more data *hurts*).
- **Appendix A** Bayesian MMSE derivation (Eq 6), dMMSE (Eq 7), Ridge completing-the-square (Eq 8). **B** hyperparameters (JAX, TPU, ~4 TPU-hrs/run; $\sigma^2$ scaled with $D$). **C–G** support figures, sequences-per-task invariance, smoothed-dMMSE construction, weight-space analyses.

---
