> **Per-paper review — in-context-learning corpus, paper 2 of 38.**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§2); content is identical. Manifest: [[in_context_learning_sources]].

# 2. Wang et al. 2023 — Large Language Models Are Latent Variable Models: Explaining and Finding Good Demonstrations for In-Context Learning

**PDF:** `docs/project/references/in_context_learning/sources/Wang et al. 2023 - Large Language Models Are Latent Variable Models.pdf` · NeurIPS 2023 · arXiv:2301.11916.

## Phase 1: Foundational Overview (Undergraduate-Level)

**Question this paper answers (plain language).** When you give a large language model (LLM) a few worked examples ("demonstrations") before your real question — e.g. two labeled movie reviews before a third unlabeled one — the model often gets the answer right without any retraining. This is *in-context learning* (ICL). But the answer is famously fragile: swap which examples you show, and accuracy swings wildly. This paper asks two linked questions: (1) *Why* does ICL work at all under ordinary language-model pretraining? and (2) Can we *pick* the best few demonstrations automatically, cheaply, and in a way that transfers across models?

**Core idea.** Treat the LLM as a **latent variable model** (like a modern neural topic model). The claim is that when the model reads a prompt, it implicitly infers a hidden "concept" or "task" variable $\theta$ — capturing both the task ("classify sentiment") and the format — and then generates its answer by conditioning on that inferred concept. Good demonstrations are the ones that most sharply point the model at the *correct* concept $\theta^d$ for task $d$.

**Key findings.**
- **Theory.** With a finite number of demonstrations, the ICL classifier can reach the **Bayes-optimal classifier** — but only if the demonstrations make the model's posterior over concepts collapse onto the true task concept $\theta^d$. This is a stronger, more practical statement than the prior HMM-based result of Xie et al. (which needed *infinitely many* demonstrations).
- **Algorithm.** A two-stage recipe: (1) **latent concept learning** — learn a handful of new "concept token" embeddings per task by prompt-tuning a *small* LM (< 1B params, e.g. GPT2-large); (2) **demonstration selection** — score each candidate example by how strongly it predicts those concept tokens, and keep the top-$k$.
- **Results.** Demonstrations chosen with a small model transfer directly to much larger models (up to 175B). Averaged over 8 GPT models and 8 text-classification datasets, the method beats uniform selection by ~12.5% relative. It also helps on GSM8K math word problems. Control experiments show the gain comes specifically from the *learned* concept tokens (random tokens give no benefit), and — unlike prior work — demonstration *order* barely matters once selection is good.

**Initial takeaway.** This is the first Bayesian/latent-variable explanation of ICL that (a) rests on realistic assumptions (no HMM, finite demonstrations, arbitrary causal direction) and (b) yields an *actually useful* algorithm validated on real LLMs. The conceptual payload for our project: a frozen sequence model can behave like a task-inferring Bayesian, and "conditioning on demonstrations" is mathematically "sharpening a posterior over latent tasks."

## Phase 2: Graduate-Level Deep Dive

### Setup and generative assumptions

We predict a discrete target $Y\in\mathcal Y$ from a token sequence $X\in\mathcal X$. A latent, *continuous* concept variable $\theta\in\Theta$ indexes the task. The paper posits a structural causal model with two possible directions, chosen per task:

$$Y = f(X,\theta,\epsilon) \quad (X\!\to\!Y\!\leftarrow\!\theta, \text{ "direct"}), \qquad X = g(Y,\theta,\epsilon)\quad (Y\!\to\!X\!\leftarrow\!\theta,\text{ "channel"}),$$

with independent noise $\epsilon$ and deterministic $f,g$. An **injective** map $T\to\Theta$ assigns each task $d$ a unique concept $\theta^d$, so every example is generated as $Y^d=f(X^d,\theta^d,\epsilon)$. Labels are projected to token space by injective $\tau^d:\mathcal Y\to\mathcal X$ (e.g. positive $\mapsto$ "positive"), and a delimiter $w^d$ separates concatenated demonstrations. With this preprocessing the LLM's ICL output probability is written $P^d_M$.

The **starting identity** for ICL marginalizes over the latent concept (Eq. 1):

$$
P^d_M(Y\mid X^d_1,Y^d_1,\dots,X^d_k,Y^d_k,X) \;=\; \int_\Theta P^d_M(Y\mid \theta,X)\,P^d_M(\theta\mid X^d_1,Y^d_1,\dots,X^d_k,Y^d_k,X)\,d\theta.
\tag{1}
$$

This is exactly **Bayesian model averaging**: the predictive is the concept-conditional predictor $P^d_M(Y\mid\theta,X)$ averaged against the posterior over concepts induced by the demonstrations. The bridging **Assumption 2.1** ties the pretrained model to the true data distribution: $P_M(X)=P(X)$ and $P^d_M(Y\mid\theta,X)\propto P(Y\mid\theta,X)$.

### Bayes-optimality of the concept-conditional predictor (Prop. 2.2)

**Claim.** For the $X\!\to\!Y\!\leftarrow\!\theta$ direction, $\arg\max_{y}P^d_M(Y=y\mid\theta^d,X)$ is the Bayes-optimal classifier.

**Derivation.** Because task $d$'s data generation is $Y=f(X,\theta^d,\epsilon)$, the *true* conditional depends on $\theta^d$ only:
$$P^d(Y\mid X) = P(Y\mid \theta^d,X).$$
By Assumption 2.1, $\arg\max_y P^d_M(Y=y\mid\theta^d,X)=\arg\max_y P(Y=y\mid\theta^d,X)$. The RHS is by definition the minimizer of misclassification probability — the Bayes classifier. $\square$

### The gap theorem: ICL ≤ Bayes, equality iff posterior collapses (Thm. 2.3)

**Claim.** The ICL classifier $C_k(X)=\arg\max_y P^d_M(Y=y\mid X^d_1,Y^d_1,\dots,X^d_k,Y^d_k,X)$ has misclassification probability $\ge$ that of the Bayes classifier $C_{\theta^d}$, with equality **iff** the concept posterior is a point mass at $\theta^d$ for *every* input:
$$\forall x\in\mathcal X,\quad P^d_M(\theta^d\mid X^d_1,Y^d_1,\dots,X^d_k,Y^d_k,X=x)=1.$$

**Step-by-step derivation.** Let $C_\theta(X)=\arg\max_y P^d_M(Y=y\mid\theta,X)$ and define the risk (0–1 loss) $R(C_\theta)=P(C_\theta(X)\ne Y)=\mathbb E_{XY}[\mathbf 1_{C_\theta(X)\ne Y}]$. For the ICL classifier $C_k$,
$$
R(C_k)=\mathbb E_{XY}[\mathbf 1_{C_k(X)\ne Y}]
=\mathbb E_X\!\Big[\sum_{y\in\mathcal Y}\big(1-P^d_M(Y=y\mid\theta^d,X)\big)\,\mathbf 1_{C_k(X)=y}\Big].
$$
The bracket is minimized pointwise by choosing, for each $X$, the label $y$ that maximizes $P^d_M(Y=y\mid\theta^d,X)$ — i.e. by $C_k(X)=C_{\theta^d}(X)$. Substituting Eq. (1) for the predictive that defines $C_k$, the only way the arg-max of the *mixture* $\int P^d_M(Y\mid\theta,X)\,P^d_M(\theta\mid\dots,X)\,d\theta$ is guaranteed to coincide with the arg-max of $P^d_M(Y\mid\theta^d,X)$ for all $X$ is when the mixing measure $P^d_M(\theta\mid\dots,X)$ places all its mass on $\theta^d$. Hence equality of risks holds iff the posterior collapses to $\theta^d$. $\square$

**Interpretation.** ICL is *never better* than knowing the true concept, and it *approaches* Bayes-optimality precisely to the extent that the demonstrations concentrate the concept posterior. This reframes "good demonstrations" as "posterior-sharpening demonstrations."

**Channel direction (Appendix A.2).** Symmetric results hold for $Y\!\to\!X\!\leftarrow\!\theta$. Prop. A.5: when labels are balanced ($P^d(Y)=1/|\mathcal Y|$), Bayes gives $\arg\max_y P^d_M(X\mid\theta^d,Y=y)$ because
$$P^d(Y\mid X)=\frac{P^d(X\mid Y)P^d(Y)}{P(X)}\propto P^d(X\mid Y)=P(X\mid\theta^d,Y).$$
Thm. A.6 is the analogous gap theorem with equality iff $P^d_M(\theta^d\mid Y^d_1,X^d_1,\dots,Y=y)=1$ for all $y$. These are the "direct" and "channel" methods of Min et al. (2022); the paper argues the better direction is task-dependent, not universally the channel.

### Method — latent concept learning (Algorithm 1)

Modeling the full posterior over $\Theta$ is intractable, so the method estimates a single optimal $\theta^d$ per task. For task $d$, add $c$ new **concept tokens** $\hat\theta^d$ to the vocabulary; freeze all LLM parameters $M$ except the new embeddings $E_{\text{new}}(\hat\theta^d)$; prompt-tune (Lester et al. 2021) by prepending these tokens to $X$ (or $Y$). The per-example loss is direction-dependent:

$$
\ell(X,Y;\hat\theta^d)=
\begin{cases}
-\log P^d_M(Y\mid \hat\theta^d,X) & X\!\to\!Y\!\leftarrow\!\theta,\\
-\log P^d_M(X\mid \hat\theta^d,Y) & Y\!\to\!X\!\leftarrow\!\theta,
\end{cases}
\qquad
L(\hat\theta^d)=\mathbb E_{X,Y}[\ell(X,Y;\hat\theta^d)].
$$

**Why the minimizer recovers the true concept (Prop. 3.1, derivation).** For the direct direction, decompose the expected loss as an entropy plus a KL:
$$
L(\hat\theta^d)=H\big(P(Y\mid\theta^d,X)\big)+\mathrm{KL}\big(P(Y\mid\theta^d,X)\,\big\|\,P^d_M(Y\mid\hat\theta^d,X)\big).
$$
The entropy term is constant in $\hat\theta^d$; the KL term is $\ge 0$ and vanishes **iff** $P^d_M(Y\mid\hat\theta^d,X)=P(Y\mid\theta^d,X)$. So at the minimum the concept-token predictor equals the true concept-conditional predictor (hence Bayes-optimal by Prop. 2.2). If $M$ is invertible (its embedding matrix is invertible, and $\theta$ is identifiable), then $\hat\theta^d=\theta^d$. $\square$

Because $\Theta$ is too large to enumerate, the concept posterior is normalized over a *sampled diverse subset of tasks* $S\subseteq T$:
$$
\hat P^d_{M'}(\hat\theta^d\mid w_{1:t})=\frac{P^d_{M'}(\hat\theta^d\mid w_{1:t})}{\sum_{t\in S}P^t_{M'}(\hat\theta^t\mid w_{1:t})}.
$$

### Method — demonstration selection (Algorithm 2)

By Thm. 2.3 we want demonstrations that maximize the concept posterior across all test inputs:
$$
\arg\max_{X^d_1,Y^d_1,\dots,X^d_k,Y^d_k}\ \mathbb E_X\big[P^d_M(\theta^d\mid X^d_1,Y^d_1,\dots,X^d_k,Y^d_k,X)\big].
$$
Two simplifications make this tractable. First, test inputs are independent of demonstrations and $P_M(X)=P(X)$, so the expectation drops $X$:
$$\mathbb E_X\big[P^d_M(\theta^d\mid \dots,X)\big]=P^d_M(\theta^d\mid X^d_1,Y^d_1,\dots,X^d_k,Y^d_k).$$
Second, assuming demonstrations are independently sampled and applying Bayes' rule to each,
$$
P^d_M(\theta^d\mid X^d_1,Y^d_1,\dots,X^d_k,Y^d_k)=\frac{\prod_{i=1}^k P^d_M(\theta^d\mid X^d_i,Y^d_i)}{P^d_M(\theta^d)^{\,k-1}}.
$$
With a uniform prior on $\theta$ the denominator is constant, so the objective factorizes and the optimum is simply the **top-$k$ examples by $\hat P^d_{M'}(\hat\theta^d\mid X^d_i,Y^d_i)$** — a per-example score, avoiding the $O(|D^d|^k)$ combinatorial search. (The independence assumption is acknowledged as a heuristic: correlated demonstrations may work well jointly but are not searched.)

### Empirical anatomy

- **Transfer:** demonstrations selected once with GPT2-large lift 8 GPT models (GPT2/GPT3 families, 6–175B), ~12.5% relative over uniform, attributed to shared pretraining corpora.
- **Cross-family:** also helps GPT-J, OPT, LLaMA at 6–7B; GSM8K (Table 1) with LLaMA-2 / ChatGPT beats uniform and similarity baselines, and *larger* selector (7B) beats smaller (1.5B).
- **Ablations:** learned concept tokens matter (random tokens ≈ random selection); $k\in\{2,4,8,16\}$ all beat baseline (best at $k{=}4$, diminishing at large $k$); $c{\approx}10$ concept tokens is a sweet spot (too few underfit, too many dilute selectivity); order-insensitivity (contrasting Lu et al. 2022) — interpreted as order capturing model-specific artifacts while content carries task info.
- **Qualitative:** t-SNE shows concept tokens for semantically similar tasks cluster; selected examples are harder/longer/more deductive, echoing "hard pretraining examples drive ICL."

### Relevance notes for the project
The paper operationalizes the abstract "ICL = Bayesian task inference" claim into (i) a clean risk-comparison theorem (ICL ≤ Bayes, gap governed by posterior concentration) and (ii) a KL-decomposition showing that a *learned low-dimensional latent code* (concept tokens) can serve as an approximate sufficient statistic for the task. For any project component that conditions a frozen policy/model on a small context to "identify a situation," this is the reference for the identifiability + posterior-collapse framing.

## Appendix: Section-by-Section Backbone

- **§1 Introduction.** ICL is powerful but sensitive to demonstration choice/format/order; mechanisms under standard LM pretraining are not understood. Prior latent-topic account (Xie et al. 2022) assumes HMM data and needs infinite demonstrations — hard to extrapolate to real language. Proposes a topic-model-style latent variable view: $P(w_{1:T})=\int_\Theta P(w_{1:T}\mid\theta)P(\theta)d\theta$, and a simplified LLM analogue $P_M(w_{t+1:T}\mid w_{1:t})=\int_\Theta P_M(w_{t+1:T}\mid\theta)P_M(\theta\mid w_{1:t})d\theta$, where $\theta$ is an approximate sufficient statistic (task + format). Empirical-Bayes: approximate optimal $\theta^d$ with a small LM. Contributions: general 3-variable causal generative process; finite-demonstration Bayes-optimality; practical, transferable selection algorithm.
- **§2 Theoretical Analysis.** §2.1 notations: discrete $Y$, sequence $X$, continuous latent $\theta$, two causal directions ($Y=f(X,\theta,\epsilon)$ / $X=g(Y,\theta,\epsilon)$), injective $T\to\Theta$, label-to-token maps $\tau^d$, delimiter $w^d$. §2.2 results: marginalization identity Eq. (1); Assumption 2.1 (pretrained ≈ true dist.); Prop. 2.2 (concept-conditional predictor is Bayes-optimal); Thm. 2.3 (ICL risk ≥ Bayes risk, equality iff posterior collapses to $\theta^d$); channel-direction analogue Eq. (2).
- **§3 Method.** Two-phase pipeline (Fig. 1). §3.1 Latent Concept Learning (Alg. 1): add $c|S|$ concept tokens, freeze $M$, prompt-tune $E_{\text{new}}$ to minimize $L(\hat\theta^d)$; Prop. 3.1 (minimizer recovers true concept, identifiable if $M$ invertible); normalize concept posterior over sampled task subset $S$. §3.2 Demonstration Selection (Alg. 2): factorize concept posterior under independence + uniform prior → select top-$k$ by per-example concept score; order argued irrelevant.
- **§4 Experiments.** 8 datasets across 5 task types (SST2, FPB, COLA, DBpedia, EmoC, EmoS, ETHOS-SO, ETHOS-R). Direction chosen per task by which gives higher random-demonstration accuracy. Defaults $k{=}4$, $c{=}10$, selector GPT2-large. Baselines: Uniform, Similar (sentence-transformer cosine). Main result: +12.5% relative over uniform across 8 GPTs (Fig. 2). Non-GPT models (Fig. 3a), GSM8K (Table 1), learned-vs-random tokens (Fig. 3b), $k$ and $c$ ablations (Fig. 4), order insensitivity, t-SNE qualitative (Fig. 5).
- **§5 Related Work.** Heuristic similarity/entropy selection; pretraining-distribution accounts (HMM, Zipfian burstiness); GD-connection mechanistic accounts; other latent-variable theories; empirical ICL studies (label-noise robustness, chain-of-thought). Positions itself as first Bayesian account verifiable on real LLMs.
- **§6 Conclusion.** LLMs as implicit topic models inferring a latent concept; two-step algorithm (extract concept tokens with small LM → select demonstrations), directly transferable across LLMs.
- **Appendix A Proofs.** A.1 direct direction (Prop. 2.2, Thm. 2.3 with 0–1 risk algebra); A.2 channel direction (Prop. A.5 balanced-label Bayes, Thm. A.6 gap theorem); A.3 method (Prop. 3.1 via $L=H+\mathrm{KL}$ decomposition, identifiability under invertibility).
- **Appendix B Experiments.** Dataset templates/label mappings (Table 2), sizes, licenses; training details (A100/V100/A6000, lr 1e-4, batch 16, 10k steps); detailed per-dataset tables; causal-direction, other-LLM, random-token results.

---
