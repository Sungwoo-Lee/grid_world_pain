# In-Context Learning — Master Literature Review

## What this document is (plain-language entry point)

**In-context learning (ICL)** is the ability of a Transformer to "learn" a new task at *inference time*, purely from examples placed in its prompt, **without any change to its weights**. Show a language model a handful of "input → answer" pairs and then a fresh input, and it completes the pattern — as if it had been trained on that task, even though not a single gradient step was taken. This document is a deep-dive review of 15 papers that try to explain *how* and *why* this works.

The corpus threads together **three complementary readings of the same phenomenon**:

1. **ICL as (implicit / amortized) Bayesian inference.** The model behaves as if it holds a *prior* over tasks and, on reading the prompt examples, computes a *posterior* — sharpening its belief about "which task is this?" and predicting by averaging over that belief. The prompt is evidence; more examples means a sharper posterior.
2. **ICL as a learning algorithm run inside the forward pass.** The frozen network, when it reads a prompt, silently *executes an optimizer* — most concretely, one step of **gradient descent** per attention layer on a small regression problem built from the in-context examples. Training the weights (the slow, outer loop) installs a *learner* (the fast, inner loop) that runs at inference.
3. **ICL as weight generation (the hypernetwork / fast-weight view).** The context is algebraically equivalent to a *set of extra weights*. A linear-attention layer accumulates the prompt into an outer-product "fast-weight" matrix; multi-head attention is literally a **hypernetwork** whose per-head attention scores are a latent code selecting which operation to run; and one can even *materialize* a whole prompt into a permanent, composable **bias edit** on one internal matrix.

These three readings are not rivals so much as different projections of one object. Reading (3) — the hypernetwork / fast-weight bridge — is the strand most directly relevant to this project's own conditioning work (see the separate `[[Hypernetwork]]` and FiLM reference topics), because it says a FiLM or hypernetwork conditioning module is *not* a foreign add-on to attention but a more explicit exposure of a mechanism attention already contains.

**Division of labor with the manifest.** The download record and index for this corpus lives in the manifest `[[in_context_learning_sources]]` — that doc tracks *which* papers exist, where the PDFs sit, and why each was pulled. **This** doc is the per-paper deep-dive: for each reviewed paper it carries a plain-language Phase 1 overview, a graduate-level Phase 2 with the full LaTeX derivations, and a section-by-section backbone. Use the manifest to see the whole corpus at a glance; use this doc to actually understand a paper.

> ### ⚠️ Scope caveat — read before assuming full coverage
>
> **This review covers the original 15-paper batch only.** The corpus was later expanded to **39 papers** (the manifest `[[in_context_learning_sources]]` lists all of them). The **24 papers added on 2026-07-21 are not yet reviewed** — they have source PDFs in `sources/` and manifest entries, but no deep-dive here. Notable un-reviewed additions include Akyürek et al. 2023, Ahn et al. 2023, Bai et al. 2023, Dai et al. 2023, Zhang et al. 2024 (the "attention = (preconditioned) gradient descent" theory cluster), the Neural-Process line (Garnelo 2018, Kim 2019, Nguyen & Grover 2022), the classic hypernetwork line (Ha 2017, von Oswald 2020, Zhmoginov 2022, Munkhdalai 2017), the fast-weight extensions (Ba 2016, Irie 2021/2022), TabPFN (Hollmann 2023), and further Bayesian-ICL theory (Panwar 2024, Raventós 2023, Wies 2023, Wang 2023). **Do not read the absence of a paper here as evidence it is outside scope** — check the manifest.

---

## Table of Contents

**Group A — Mila / Lajoie–Bengio group (ICL as compression, latent inference, amortized inference)**

1. [Elmoznino et al. 2025 — In-Context Learning and Occam's Razor](#1-elmoznino-et-al-2025--in-context-learning-and-occams-razor)
2. [Mittal et al. 2025 — Does Learning the Right Latent Variables Necessarily Improve In-Context Learning?](#2-mittal-et-al-2025--does-learning-the-right-latent-variables-necessarily-improve-in-context-learning)
3. [Hu et al. 2024 — Amortizing Intractable Inference in Large Language Models](#3-hu-et-al-2024--amortizing-intractable-inference-in-large-language-models)

**Group B — Bayesian foundations (why conditioning on examples adapts, and what computation runs)**

4. [Xie et al. 2022 — An Explanation of In-Context Learning as Implicit Bayesian Inference](#4-xie-et-al-2022--an-explanation-of-in-context-learning-as-implicit-bayesian-inference)
5. [Müller et al. 2022 — Transformers Can Do Bayesian Inference (PFNs)](#5-müller-et-al-2022--transformers-can-do-bayesian-inference-pfns)
6. [Garg et al. 2022 — What Can Transformers Learn In-Context? A Case Study of Simple Function Classes](#6-garg-et-al-2022--what-can-transformers-learn-in-context-a-case-study-of-simple-function-classes)
7. [von Oswald et al. 2023 — Transformers Learn In-Context by Gradient Descent](#7-von-oswald-et-al-2023--transformers-learn-in-context-by-gradient-descent)

**Group C — Posterior estimation (turning a Transformer into a posterior / posterior-predictive machine)**

8. [Reuter et al. 2025 — Can Transformers Learn Full Bayesian Inference In Context?](#8-reuter-et-al-2025--can-transformers-learn-full-bayesian-inference-in-context)
9. [Mittal et al. 2025 — Amortized In-Context Bayesian Posterior Estimation](#9-mittal-et-al-2025--amortized-in-context-bayesian-posterior-estimation)
10. [Mittal et al. 2025 — In-Context Parametric Inference: Point or Distribution Estimators?](#10-mittal-et-al-2025--in-context-parametric-inference-point-or-distribution-estimators)
11. [Kang et al. 2026 — Transformers Can Learn Posterior Predictive Distributions In-Context](#11-kang-et-al-2026--transformers-can-learn-posterior-predictive-distributions-in-context)

**Group D — Hypernetwork ↔ ICL (context as weights; the bridge to this project's conditioning work)**

12. [Schug et al. 2025 — Attention as a Hypernetwork](#12-schug-et-al-2025--attention-as-a-hypernetwork)
13. [Schlag et al. 2021 — Linear Transformers Are Secretly Fast Weight Programmers](#13-schlag-et-al-2021--linear-transformers-are-secretly-fast-weight-programmers)
14. [Mittal et al. 2025 — Iterative Amortized Inference: Unifying ICL and Learned Optimizers](#14-mittal-et-al-2025--iterative-amortized-inference-unifying-icl-and-learned-optimizers)
15. [Chen et al. 2024 — Exact Conversion of In-Context Learning to Model Weights](#15-chen-et-al-2024--exact-conversion-of-in-context-learning-to-model-weights)

Also below: [Cross-Paper Synthesis](#cross-paper-synthesis) (the throughlines that tie the 15 papers together — read this first for the big picture).

---

## Cross-Paper Synthesis

This section folds together the local syntheses that accompanied each processing batch into one corpus-level view. It traces the three readings named in the entry point through the 15 papers, surfaces where they agree and disagree, and flags the axis most contested within the corpus.

### Throughline 1 — ICL as Bayesian inference (the dominant lineage)

The single most-repeated claim in the corpus is that **ICL is Bayesian inference the model was never told it was doing**. The lineage builds cumulatively:

- **Xie et al. 2022** (§4) gives the founding argument: if pretraining documents carry a shared latent "concept," then to predict the next token the model must *infer the concept*, and at test time it infers the *shared prompt concept* by marginalization. Formally the in-context predictor's belief over concepts **concentrates** on the true one as the number of examples $n\to\infty$ (posterior concentration under a distinguishability condition), and error otherwise decays like $O(1/k)$ in example length $k$. ICL emerges *for free* from ordinary next-token pretraining, provided the data has latent long-range structure.
- **Müller et al. 2022** (§5, PFNs) turns the observation into a *training objective*: sample tasks from a prior, hide a label, train a Transformer to predict it. They prove minimizing this "prior-data NLL" is exactly minimizing the KL divergence to the true **posterior predictive distribution (PPD)** — so a perfectly trained **PFN *is* Bayesian**, not by accident but as the exact minimizer of an objective whose global optimum is the posterior predictive (Corollary 1.1–1.2).
- **Elmoznino et al. 2025** (§1) reframes the whole thing through **compression / Occam's razor**: the cumulative next-token loss *is* prequential code length, an upper bound on Kolmogorov complexity, so ICL is driven toward the *simplest* predictor that fits. Crucially they argue the **Kolmogorov/compression view generalizes the Bayesian view** (their Appendix H): it needs no well-defined task distribution and accommodates non-Bayesian empirics (Raventós et al. 2024). This is the conceptual umbrella under which the Bayesian lineage sits as a special case.
- The **posterior-estimation cluster** (§8–§11) pushes from "predict the next value" to "output the posterior itself." **Reuter et al. 2025** (§8) emits *full high-dimensional posterior samples* via flow matching, matching HMC and beating variational inference on skewed/multimodal posteriors. The **two Mittal posterior papers** (§9 Amortized, §10 Point-or-Distribution) run controlled bake-offs over *which objective* (forward vs reverse KL) and *whether to estimate a distribution at all*. **Kang et al. 2026** (§11) supplies the mechanistic proof that attention can compute the PPD's moments and emit a calibrated binned density.

All of these share one piece of machinery: the **joint-sampling / law-of-total-expectation trick** that turns an intractable "match the true posterior" objective into tractable training on simulator draws $(\theta,\mathcal{D})$ (Reuter's Proposition 1 ≡ Mittal's forward-KL derivation ≡ the PFN prior-data NLL).

### Throughline 2 — ICL as gradient descent (the mechanistic reading)

A second lineage asks not "what is the optimum?" but "what computation actually runs in the forward pass?":

- **Garg et al. 2022** (§6) is the clean *empirical* counterpart to Xie's theory: train a Transformer from scratch to in-context learn a well-defined function class (linear functions, sparse-linear, decision trees, 2-layer nets), and it **matches the optimal algorithm** — least squares for linear regression (including reproducing the double-descent curve), Lasso for sparse — in one forward pass, robust to distribution shift (so it is *not* doing nearest-neighbor lookup). This turned "does the Transformer learn an algorithm?" into a measurable question.
- **von Oswald et al. 2023** (§7) answers *which* algorithm: for regression, **one linear self-attention layer performs exactly one step of gradient descent** on a least-squares loss built from the in-context examples (Proposition 1 gives the explicit weight construction; trained weights are shown to *match* it, and depth = iterated GD, with trained models discovering an accelerated variant "GD++"). ICL is *emergent gradient-based meta-learning* — a "mesa-optimizer" installed by the outer training loop.

These two threads meet the Bayesian one at **Kang et al. 2026** (§11), which explicitly bridges the von-Oswald / Akyürek "attention = gradient descent" mechanism to the distributional PFN world: attention runs a Richardson iteration to compute posterior-predictive *moments*, then an MLP+softmax head turns them into a density.

### Throughline 3 — The hypernetwork ↔ ICL bridge (most relevant to this project)

This arc — the "context as weights" reading — is the sub-corpus closest to the project's hypernetwork / FiLM conditioning interests. It forms a tight chain from raw algebra to literal weight edits:

- **Schlag et al. 2021** (§13) is the historical + mathematical anchor: **linear self-attention is a Fast Weight Programmer**. Removing the softmax and using associativity, the causal attention output is a read from an accumulated **outer-product matrix** $\mathbf{W}^{(i)}=\sum_j \mathbf{v}^{(j)}\otimes\phi(\mathbf{k}^{(j)})$ — the context *is* a weight matrix, written by outer products over the **key** index. Their delta-rule extension (DeltaNet) makes that memory *editable* via a learned, input-dependent write-strength — i.e. the in-context state update is itself a learned optimization step (the Widrow–Hoff delta rule).
- **Schug et al. 2025** (§12) factors the *same* attention layer differently — over the **head** index — to reveal **multi-head attention as a hypernetwork**: the per-head attention scores $\mathbf{a}_{q,k}\in\mathbb{R}^H$ are a *latent code* that selects a linear combination of fixed per-head "operation matrices" $\boldsymbol{W}_h^{\text{out}}\boldsymbol{W}_h^{\text{value}}$ (Eqs. 2–5). Number of heads = latent-code dimension. Their HYLA variant makes the generated value network nonlinear *without adding parameters* and improves compositional generalization. This is a first-principles argument that **attention is already conditional modulation** — the FiLM / hypernetwork family — so exposing a hypernetwork conditioning module is not foreign to attention.
- **Mittal et al. 2025 (Iterative Amortized Inference)** (§14) is the abstraction layer over both: every fast-adaptation method (MAML, ICL, PFNs, hypernetworks, learned optimizers) is one template $f_\gamma(x, g_\phi(\mathcal{D}_T))$ — a shared predictor applied to a query, using a task-adaptation function that maps support data to task information. Its **parametric / implicit / explicit** taxonomy places hypernetworks (parametric — externalize a task code), ICL/PFNs (implicit — internalize in the forward pass), and neural processes (explicit) as three faces of one object, and reframes in-context adaptation as **iterative optimization**. This tells the project exactly where a FiLM/hypernetwork module sits (parametric: a learned $g_\phi$ generating conditioning parameters) versus in-context conditioning (implicit).
- **Chen et al. 2024** (§15) is the extreme, literal endpoint: for linearized attention, a whole ICL prompt collapses **exactly** into a single **bias edit** on the Key-Value matrix $A=\sum_j\phi(K_j)V_j^\top$ — the very outer-product accumulator Schlag identified. Algorithm ICLCA materializes the context into permanent, composable, reversible weights in one forward pass (no gradient descent). This is the proof-of-concept that a learned context code *can* be turned into a permanent low-cost weight modification — Schug's "context generates a value network" becomes "context generates a storable weight offset."

The chain is therefore: **Schlag** (context = outer-product fast weights) → **Schug** (same layer, factored over heads = a hypernetwork latent code) → **Mittal** (amortized-inference framing: parametric vs implicit) → **Chen** (materialize the accumulator as a literal, composable weight/bias edit). Cross-link to the project's separate `[[Hypernetwork]]` reference topic for the conditioning-architecture side of this bridge.

### The contested axis — point vs. distribution estimator

The sharpest *disagreement* in the corpus is internal to the posterior-estimation cluster (§8–§11), and it is a genuine, not merely apparent, contest:

- **Reuter et al. 2025** (§8) demonstrates a **full-distribution** estimator (flow matching) that *beats* variational inference on posterior fidelity — especially for skewed and multimodal posteriors in *modest* latent dimension.
- **Mittal (Point or Distribution)** (§10) demonstrates that **point** estimators (MLE/MAP) generally *beat* distribution estimators for *downstream prediction* — winning ~71% of 88 tasks — especially in *high* dimension and under multimodality. The argued cause is **fundamental, not an engineering gap**: parameter-space symmetries (label-switching in mixtures, weight-permutation symmetries in neural nets) create combinatorially many *equivalent* modes that are expensive to represent yet predictively redundant, so a point estimate captures the useful information far more cheaply.

These are **reconcilable**, and the reconciliation is instructive: they measure *different things on different regimes*. Reuter scores *posterior-sample fidelity* against HMC (C2ST / MMD / $W_2$) at modest latent dimension; Mittal scores *predictive loss* at high dimension. Both agree the distribution advantage **shrinks as latent dimension grows** — Reuter's own $K=20,50$ ablation shows the advantage evaporate, matching Mittal's high-dimensional finding. A secondary axis, the **amortization objective**, runs alongside: **forward KL** (= neural posterior estimation / SBI, = Reuter's flow-matching loss) is mean-seeking, over-estimates variance, and *must* train on simulator draws from the assumed model (fragile under misspecification); **reverse KL** (= amortized VI / neural-process style) is mode-seeking, can train on *arbitrary / real* data, and is robust under misspecification and better predictively in high dimension. **Symmetric KL** is a strong compromise (Mittal §10).

### Gaps and connections to the project

- **The existence-vs-learning gap.** von Oswald (§7), Kang (§11), and Chen (§15) are all **constructive** results — they prove the architecture *can* realize a mechanism (GD step, moment solver, bias edit), not that standard pretraining *discovers* it. This is the corpus's biggest open question and a caution for any project claim that a conditioning mechanism "implements" a given computation.
- **Latent inference is necessary but not sufficient.** Mittal (§2, "Does Learning the Right Latent Variables…") is the pointed negative result: even when a bottleneck provably *recovers* the true task latents (linearly decodable, causally interventable), end-to-end training fails to build a *prediction* head that uses them to extrapolate out-of-distribution. The transferable lesson for the project's conditioning work: **a clean, low-dimensional, interpretable task representation is not self-justifying** — the module that *consumes* the representation needs its own inductive biases. This complements Occam (§1): compression pushes toward simple models, but handing the model correct low-dimensional structure still does not guarantee the prediction stage exploits it.
- **Diversity buys robustness.** Hu et al. (§3) operationalizes "reasoning = amortized Bayesian inference over latents" as a *training algorithm* (GFlowNet fine-tuning), and shows distribution-*matching* beats reward-*maximization* precisely when diversity and robustness to reward misspecification matter — with the biggest gains out-of-distribution. For the project: prefer distribution-matching over reward-maximizing RL when diversity of latent samples is what buys OOD generalization.

Where these connect to the project's own program (the neuromodulation / FiLM conditioning line, its gates, and the null-result diagnosis series): the hypernetwork↔ICL bridge (Throughline 3) is the formal backing for treating attention-style conditioning and FiLM/hypernetwork conditioning as the same family; and Mittal §2's "latents recovered but unused" result is a direct warning relevant to any diagnosis of why a modulation signal is present in activations yet fails to change behavior. These links are drawn here for the reader to follow into the project's develop-side docs; this review does not itself re-run those diagnoses.

---

## Per-Paper Reviews

The 15 reviews below are preserved verbatim from the per-paper deep-dives (Phase 1 overview, Phase 2 graduate deep-dive with full derivations, and the section-by-section backbone completeness guard). Only the section numbering has been unified to 1–15 and the papers reordered by theme; no paper content has been compressed.

---

## 1. Elmoznino et al. 2025 — In-Context Learning and Occam's Razor

**Venue:** ICML 2025 (PMLR v267). **Group:** Mila – Quebec AI Institute / Université de Montréal (+ NVIDIA). Authors: Eric Elmoznino, Tom Marty, Tejas Kasetty, Leo Gagnon, Sarthak Mittal, Mahan Fathi, Dhanya Sridhar, Guillaume Lajoie.
**PDF:** `docs/project/references/in_context_learning/sources/Elmoznino et al. 2025 - In-context learning and Occam's razor.pdf`
**Code:** https://github.com/3rdCore/PrequentialCode

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction — the core problem.** Machine learning wants models that *generalize* to data they were never trained on. A long-standing heuristic for this is **Occam's razor**: among all explanations that fit the observations, prefer the simplest one, because a simple rule cannot just memorize the data — it has to capture a real pattern. The trouble is that "simplicity" is mathematically slippery. Formal notions like Kolmogorov complexity (the length of the shortest program that reproduces the data) are *uncomputable*, so in practice people only minimize training error and then bolt on indirect simplicity nudges — weight decay, architecture choices, early stopping. This paper asks: is there a training objective that minimizes error *and* complexity **directly**? Its surprising answer: yes, and you are already using it whenever you train a Transformer with next-token prediction. That objective, applied to sequences of (data → prediction) pairs, secretly performs Occam's razor.

**Key findings — the main result.** The paper's central claim is an *equivalence*: the cumulative next-token-prediction loss that trains an in-context learner is **exactly** a data-compression scheme called **prequential coding** (a.k.a. prequential MDL). Under prequential coding you compress a dataset by predicting each new point using a model trained only on the points before it, and summing up the "surprise" (negative log-likelihood) of each point. The total bill — the area under the learning curve — is an upper bound on the data's Kolmogorov complexity. Because compression and simplicity are two sides of the same coin (Minimum Description Length principle), a sequence model trained to minimize this loss is being driven to infer, from the context, the *simplest* predictor that still fits — which is precisely what generalizes. The paper backs this with controlled experiments (linear regression, sinusoid regression, a Mastermind code-breaking task, and a synthetic HMM "language" task) showing that standard ICL ("prequential ICL") generalizes better than a control that only minimizes training error ("train-risk ICL"), especially in the **low-data regime**, and that the gap widens with task difficulty.

**Initial takeaway.** ICL works well not by magic but because next-token prediction is a disguised form of optimal compression, and optimal compression implements Occam's razor. This is a **normative** account — it says what ICL *should* do — and it also exposes ICL's failure modes: current in-context learners can **underfit** (they run one fixed-depth forward pass, unlike gradient descent which can iterate for weeks), and they fail to generalize to genuinely novel tasks (a large pretrained LLM, GPT-4, did *worse* than a tiny task-specialized model — and even worse than a trivial "predict the average label" baseline — on the Mastermind task). These are concrete, principled targets for improving ICL.

### Phase 2: Graduate-Level Deep Dive

The theoretical contribution is a four-step chain that ends in an exact equivalence between the ICL training objective and prequential code length. I reconstruct each link with its derivation.

#### 2.1 Kolmogorov complexity, the 2-part code, and why NLL is compression

The Kolmogorov complexity of a string $x$ is the length of the shortest binary program that prints $x$ on a universal Turing machine $U$ and halts:

$$
K(x) = \min_{r} \{\, l(r) : U(r) = x,\; r \in \{0,1\}^* \,\}.
$$

The conditional version $K(x\mid y)$ is the shortest program that outputs $x$ given $y$ as input. A key structural fact (used implicitly throughout) is the **Symmetry of Information** theorem, holding up to an additive logarithmic term:

$$
K(x,y) = K(y\mid x) + K(x) = K(x\mid y) + K(y).
$$

Now specialize to a dataset $D = (d_1,\dots,d_n)$ with $d_i \sim p$. Grünwald's coding argument says a large dataset is optimally compressed by a **2-part code**: first transmit the generating distribution $p$, then transmit the data under it. Optimal encoding of data under a known distribution costs $-\log_2 p(D)$ bits (arithmetic coding achieves the Shannon bound), giving the paper's Equation (1):

$$
K(D) = K(p) + K(D\mid p) = K(p) - \log_2 p(D).
$$

The two terms have completely different tractability. $K(p)$ — the complexity of the *distribution* — requires enumerating all programs that output the function $p$, so it is intractable. But $K(D\mid p) = -\log_2 p(D)$ is just the **negative log-likelihood** (NLL) of the data, the workhorse ML loss. This is the load-bearing observation of the whole paper: **NLL *is* a conditional description length**, so "a model with lower training error" literally means "a model that compresses its data into fewer bits."

For a parameterized model $p_\theta$ rather than the true $p$, the equality relaxes to an inequality (Appendix A, Eq. 10):

$$
K(X\mid p_\theta) \le -\sum_{i=1}^n \log_2 p_\theta(x_i), \qquad \text{equality iff } p_\theta = p.
$$

and the dataset's own complexity is bounded by the best 2-part code (Eq. 11–13):

$$
K(X) \le K(X\mid p_\theta) + K(p_\theta), \qquad
K(X) = \arg\min_{p_\theta}\Big[ -\sum_{i=1}^n \log_2 p_\theta(x_i) + K(p_\theta) \Big].
$$

This last line is Occam's razor made mathematical: the complexity of the data equals the cost of the model that **jointly minimizes error and its own complexity**, and the minimizer is (asymptotically) the true generating process $p$ — hence generalization.

#### 2.2 Two idealized learners: maximum-likelihood vs. Occam

A learning algorithm is a map $T : \mathcal{P}(D) \to \Theta$ from datasets to model parameters. Standard maximum-likelihood training is

$$
T^{\mathrm{ml}}(D) = \arg\min_{\theta'} \; -\sum_{d\in D} \log_2 p_{\theta'}(d),
$$

which optimizes error alone. The *Occam* learner adds the model-complexity term from the 2-part code:

$$
T^{\mathrm{oc}}(D) = \arg\min_{\theta'} \Big[\, K(p_{\theta'}) - \sum_{d\in D} \log_2 p_{\theta'}(d) \,\Big].
$$

$T^{\mathrm{oc}}$ is exactly what we want but **intractable**, because $K(p_{\theta'})$ cannot be computed. Regularizers (L2 norm, restricted model class, SGD's implicit bias) are all read here as *cheap surrogates* for the missing $K(p_{\theta'})$ term. The paper's move is to sidestep $K(p_\theta)$ entirely via prequential coding, then *meta-learn* a learner $T_\phi$ that approximates $T^{\mathrm{oc}}$.

#### 2.3 Prequential coding — compression that never needs $K(p_\theta)$

Order the dataset $D = \{d_1,\dots,d_N\}$ and write $D_{1:i} = \{d_1,\dots,d_i\}$. Prequential coding compresses $D$ **online**: train $p_{\theta_1} = T(d_1)$ on one point, use it to encode $d_2$ at cost $-\log_2 p_{\theta_1}(d_2)$; then retrain $p_{\theta_2}=T(D_{1:2})$, encode $d_3$; and so on. The receiver can decode each point because it has already decoded all prior points and can re-run the (deterministic) learner $T$ to reproduce $p_{\theta_i}$. The total bill is the **prequential code length** (Eq. 4):

$$
L_{\mathrm{preq}}(D; T) = \sum_{i=0}^{N-1} -\log_2 p_{\theta_i}(d_{i+1}) \;\ge\; K(D).
$$

Geometrically this is the **area under the learning curve** (per-point loss vs. context length). Crucially it *never references* $K(p_\theta)$ — the model complexity is paid for implicitly, through how *slowly* the curve descends. A model that generalizes from few points (fast-descending curve, small area) is by definition simple; a model that needs many points is complex. Thus prequential code length operationalizes complexity via **speed of adaptation**.

**The learner must itself be encoded.** A subtlety (Appendix B): prequential coding presupposes the receiver knows $T$. Accounting for that, the honest bound is

$$
K(D) \le K(D, p_\theta) = K(D\mid T) + K(T) \le L_{\mathrm{preq}}(D; T) + K(T)
\;\Longrightarrow\; L_{\mathrm{preq}}(D; T) + K(T) \ge K(D).
$$

If $T$ is a *short* program (SGD, say), $K(T)\approx 0$ and $L_{\mathrm{preq}}$ alone bounds $K(D)$. But a meta-learned $T_\phi$ (a Transformer) is a *long* program, so $K(T_\phi)$ cannot be ignored — which motivates §2.4.

#### 2.4 Meta-learning over tasks controls $K(T_\phi)$, and ICL solves it exactly

To stop a learnable $T_\phi$ from cheating by memorizing one dataset (which would make $K(T_\phi)$ huge), train it across a **meta-dataset** $\mathcal{D} = \{D_1,\dots,D_M\}$ of $M$ tasks and minimize average prequential code length (Eq. 5):

$$
\mathcal{L}(\mathcal{D};\phi) = \sum_{i=1}^{M} L_{\mathrm{preq}}(D_i; T_\phi) \;\ge\; \sum_{i=1}^{M} K(D_i\mid T_\phi).
$$

A single $T_\phi$ that compresses *many* tasks well is forced to be a genuinely general (hence low-complexity relative to the tasks it explains) learner — this is the mechanism that keeps $K(T_\phi)$ in check without ever computing it.

The punchline: minimizing this meta-objective is **already** what a probabilistic sequence model does. Such a model predicts $F(d_t \mid D_{1:t-1})$. The decisive observation is the **dual identity** of the sequence model — it is simultaneously the learner $T_\phi$ *and* the fitted model $p_\theta$:

- $\phi$ = the network weights (e.g. Transformer parameters);
- $\theta = T_\phi(D_{1:t-1})$ = the *activations* the network produces after reading context $D_{1:t-1}$;
- so $F(d_t \mid D_{1:t-1}) = p_{T_\phi(D_{1:t-1})}(d_t)$.

The training loss is the cumulative next-token NLL,

$$
\mathcal{L}(D;\phi) = \sum_{t=1}^{N} -\log p_{T_\phi(D_{1:t-1})}(d_t),
$$

**which is identically the prequential code length of Eq. (4).** This dual nature is what makes the seemingly-daunting desiderata of §2.4 trivially satisfied: (1) the loss *is* the prequential code length; (2) evaluating $T_\phi$ costs a single forward pass; (3) it is fully differentiable in $\phi$, so meta-gradients are cheap; (4) Transformers parameterize an expressive class of learners (multi-head attention, induction heads). For a supervised task the datapoint is a pair $d_t = (x_t, y_t)$, and a query $x^*$ appended to the context yields $\hat y^*$ as the next token — so the framework covers regression/classification, not just language, and applies to non-iid sequences too (prequential coding is defined over arbitrary orderings).

#### 2.5 The critical experimental contrast: prequential vs. train-risk ICL

The theory predicts that *which token you evaluate the loss on* is what makes ICL generalize. The paper isolates this with a clean control. Both learners have the identical architecture $\hat y_q = T_\phi((x,y)_{1:j}, x_q)$; they differ only in the query:

- **Prequential ICL** (standard): $x_q$ is a **novel** input not in the context → loss = generalization error → prequential code length.
- **Train-risk ICL** (control): $x_q$ is **resampled from the context** $x_{1:j}$ → loss = training error only.

To stop train-risk ICL from trivially attending-and-copying the matching context token, both use a **bottlenecked architecture** (from Mittal et al. 2025, §2 of this review): a Transformer compresses the context to a low-dimensional $z = \mathrm{Transformer}_\phi((x,y)_{1:j})$, then a separate MLP head predicts $\hat y_q = \mathrm{MLP}_\phi(x_q, z)$.

**Findings (Figures 2–3):**
- At **large context lengths** the two are identical — with abundant data, Eq. (1)'s $-\log_2 p(D)$ error term dominates the $K(p)$ complexity term, so overfitting is not a risk.
- At **short context lengths**, prequential ICL wins, and the gap **grows with task difficulty** (small for linear regression, larger for sinusoid, largest for Mastermind; confirmed by sweeping input dimensionality in Appendix D).
- Visualizing inferred polynomial coefficients (Chebyshev basis): train-risk ICL exploits high-degree components to overfit noisy data; prequential ICL fits simpler, lower-degree functions closer to ground truth.

#### 2.6 Failure modes exposed by the theory

- **vs. SGD (Figure 2):** ICL generalizes better in low-data regimes but can be *beaten by SGD* in high-data regimes on hard tasks (Mastermind), because a Transformer's implicit fitted model has expressivity scaling with its activations ($\sim N$) and fixed compute (one forward pass), whereas an SGD-trained DNN scales with weights ($\sim N^2$) and unbounded compute. This is **underfitting**, and it is architecture-dependent — the choice of $T_\phi$ (bottlenecked Transformer vs. plain Transformer vs. Mamba-1/2) strongly changes which tasks get compressed well (Figure 4).
- **Task generalization (Figure 5):** GPT-4 on Mastermind has *higher* prequential code length than both a small task-specialized model and a naive marginal-class baseline — a large pretrained model can fail to minimize prequential code length on a genuinely novel task, even one a small MLP solves easily.

#### 2.7 Kolmogorov vs. Bayesian framing (Appendix H) — why this matters for the corpus

The dominant rival account (Xie et al. 2022, Müller et al. 2022; Groups B of this corpus) is **ICL = implicit Bayesian inference**: meta-training instills a prior $p(p_\theta\mid \mathcal{D})$ over tasks, and in-context data yields a posterior $p_D(p_\theta\mid D) \propto p(D\mid p_\theta)\,p(p_\theta\mid\mathcal{D})$, with predictions via Bayesian model averaging. Elmoznino et al. argue the **Kolmogorov/compression view generalizes the Bayesian one**:

1. It needs **no well-defined task distribution** — $D$ need not be drawn from a prior for the meta-learning argument to hold.
2. Minimizing $\sum_i L_{\mathrm{preq}}(D_i;T_\phi)$ need not induce any explicit prior (it *can*, if that is the cheapest way to compress the meta-dataset).
3. The Bayesian prior $p(p_\theta)$ is only *implicitly* defined by architecture + objective, making it hard to analyze and prone to unfalsifiable "just-so" explanations; the compression view refers only to the explicit $T_\phi$ and the measurable $L_{\mathrm{preq}}$.
4. It **accommodates non-Bayesian empirics** — e.g. Raventós et al. (2024) found that with high pretraining-task diversity, ICL does *not* converge to the Bayes-optimal predictor under the pretraining prior but to a broader prior; the compression view explains this as $T_\phi$ having insufficient capacity to encode the true diverse prior, defaulting to a simpler broad-coverage approximation.

This appendix is the conceptual hinge linking paper 1 to the rest of the ICL corpus: it positions compression/MDL as the umbrella under which the Bayesian-inference lineage (Group B) sits as a special case.

### Appendix: Section-by-Section Backbone

- **Abstract.** ICL's next-token loss = prequential coding = joint minimization of training error and model complexity → a normative Occam's-razor account of ICL, plus diagnosis of ICL's shortcomings.
- **§1 Introduction.** Generalization needs Occam's razor; formal complexity (VC dim, Kolmogorov) is uncomputable; current ML uses indirect proxies (L1/L2, inductive bias). ICL fits "models at inference" without weight updates. Claim: ICL's meta-objective ≡ prequential coding, a compression method → Occam's razor. Scope: ICL = inferring a statistical model from an in-context dataset (memory-based meta-learning).
- **§2 ICL and Occam's Razor.** Four-step argument.
  - **§2.1 Kolmogorov complexity & compression.** $K(x)$ = shortest program; 2-part code $K(D)=K(p)-\log_2 p(D)$ (Eq. 1); NLL = conditional description length; $T^{\mathrm{ml}}$ (Eq. 2) vs. intractable Occam learner $T^{\mathrm{oc}}$ (Eq. 3).
  - **§2.2 Prequential coding.** Online compress-as-you-learn; $L_{\mathrm{preq}}(D;T)=\sum -\log_2 p_{\theta_i}(d_{i+1}) \ge K(D)$ (Eq. 4) = area under learning curve; complexity ↔ speed of adaptation; must also pay $K(T)$.
  - **§2.3 Minimizing prequential code length.** Meta-dataset $\mathcal{D}$ of $M$ tasks; objective $\sum_i L_{\mathrm{preq}}(D_i;T_\phi)$ (Eq. 5); relation of $T_\phi^*$ to $T^{\mathrm{oc}}$ via MDL.
  - **§2.4 ICL minimizes prequential code length.** Four desiderata; sequence model is *both* learner $T_\phi$ (weights) and model $p_\theta$ (activations); cumulative next-token NLL ≡ Eq. 4; extends to supervised & non-iid.
  - **Summary.** Next-token prediction explicitly optimizes a compression meta-objective; explains LLMs as universal compressors (Deletang et al. 2024).
- **§3 Experiments.** Tasks: (1) linear regression $\mathbb{R}^3\!\to\!\mathbb{R}$, (2) sinusoid regression, (3) Mastermind (code-breaking multi-label classification, length-8 codes, digits 1–6), (4) HMM non-stationary "language."
  - **§3.1 vs. train-risk ICL.** Prequential vs. train-risk = novel vs. in-context query; bottlenecked architecture prevents copy-cheat; prequential wins in low-data, gap grows with difficulty (Figs 2–3).
  - **§3.2 vs. gradient-based learners (SGD).** ICL better in low-data; SGD matches/beats in high-data (Mastermind) — activations ($N$) vs. weights ($N^2$), one forward pass vs. unbounded compute. Regularized SGD still beaten in low-data (App. E).
  - **§3.3 Influence of ICL architecture.** Transformer w/ & w/o bottleneck, Mamba-1, Mamba-2; architecture strongly interacts with task (Fig 4) — next-token objective alone doesn't guarantee good compression.
  - **§3.4 Large pretrained models.** GPT-4 on Mastermind fails to minimize prequential code length — worse than small specialized model *and* naive marginal baseline (Fig 5).
- **§4 Related Work.** (i) Sequence modeling & compression (Blier & Ollivier; Deletang et al.); (ii) ICL as Bayes-optimal prediction (Xie et al., Müller et al., Akyürek et al., Raventós et al.); (iii) ICL as meta-learned optimizer (von Oswald et al. — linear Transformers do gradient descent; Ahn et al.; Garg et al.).
- **§5 Discussion & Future Work.** Shortcomings: (a) underfitting, architecture-dependent → add optimization primitives / variable compute depth; mixture-of-learners combining ICL (low-data) + SGD (high-data) (Bornschein et al. 2024); (b) task generalization is hard open-endedly → domain-specific learners, or a meta-meta ICL minimizing $K(T_\phi)$; (c) data ordering affects prequential code length → curriculum design.
- **Appendix A.** Full Kolmogorov background: Eq. (6) definition, Eq. (7) conditional, Eq. (8) symmetry of information, Eqs. (9)–(13) dataset complexity / 2-part code; relation to Shannon entropy (Kolmogorov strictly generalizes Shannon — handles the complexity of the *distribution itself*).
- **Appendix B.** Compression without a known learner: $L_{\mathrm{preq}}(D;T)+K(T)\ge K(D)$ (Eqs. 14–17); for short $T$ (SGD), $K(T)\approx 0$.
- **Appendix C.** Architectures (bottlenecked Transformer 4L/4H/256d + 5-layer MLP head; encoder–decoder no-bottleneck with dual $X$/$D$ token streams and masks $M_X,M_D$, Eqs. 18–19; Mamba-1/2), hyperparameters, Chebyshev-basis visualization, GPT-4o prompting protocol (temperature 0, logprobs), and the HMM generative process (cycles/speeds/emission groups building blocks $\xi$, task latents $\tau$; 12,228 HMMs; $|X|=50$, $|Z|=20$).
- **Appendix D.** Task difficulty (sinusoid input-dim sweep 1→10) widens the prequential-vs-train-risk gap.
- **Appendix E.** SGD regularization (early stop, L2) lowers prequential code length but still loses to prequential ICL in low-data.
- **Appendix F.** HMM comparisons: prequential ICL generalizes at all context lengths; train-risk ICL and SGD fail on non-stationary data.
- **Appendix G.** Full-sequence vs. suffix-only training: suffix-only (mimicking large-context LLM training) generalizes worse in low-data.
- **Appendix H.** Advantages over the Bayesian perspective (see §2.7 above) — compression view generalizes Bayesian view; handles no-task-distribution, out-of-domain generalization, and non-Bayesian empirics (Raventós et al. 2024).

---

## 2. Mittal et al. 2025 — Does Learning the Right Latent Variables Necessarily Improve In-Context Learning?

**Venue:** ICML 2025 (PMLR v267). **Group:** Mila – Quebec AI Institute / Université de Montréal (+ Google DeepMind). Authors: Sarthak Mittal, Eric Elmoznino, Leo Gagnon, Sangnie Bhardwaj, Guillaume Lajoie, Dhanya Sridhar. (Same core group as paper 1; this is the paper whose bottlenecked architecture paper 1 borrows for its "train-risk vs. prequential" control.)
**PDF:** `docs/project/references/in_context_learning/sources/Mittal et al. 2025 - Does learning the right latent variables improve in-context learning.pdf`

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction — the core problem.** A popular story for *why* in-context learning generalizes goes like this: many tasks are secretly generated by a few hidden "task variables" (linear regression is fully described by its coefficient vector; a board game by its rules), so a Transformer that reads some examples must be inferring those hidden variables and then using them to answer new queries. This is the **parametric** view: first infer the parameters `z` from the context, then apply a prediction rule. Its appeal is that it *should* generalize far outside the training distribution, because once you know the true `z`, prediction is exact everywhere. The competing worry is that Transformers instead take **shortcuts** — they just compare a new query to the stored examples and copy the answer of the closest match (like nearest-neighbour or kernel regression, the behaviour of "induction heads"). Shortcuts fit the training data fine but break under distribution shift. This paper asks a sharp, testable question: **if we force a Transformer to infer the true task variables, does it actually generalize better?**

**Key findings — the main result.** The authors build an **explicit model** — a minimal surgery on the Transformer that inserts an **information bottleneck**: one Transformer squeezes the whole context into a small fixed-size vector `z`, and a *separate* prediction network answers the query using only `z` (the query is forbidden from looking at the raw examples). This structurally bans the copy-the-nearest-example shortcut and forces the model to distil the task's latent variables. They compare this against the ordinary Transformer (the **implicit model**) across a broad task suite: linear/nonlinear/sinusoid regression, classification, Gaussian-process regression, a neuroscience ODE (Hodgkin–Huxley), and several compositional-reasoning tasks (Raven's Progressive Matrices, DeepMind Alchemy, a CRISPR gene-targeting dataset, and a reusable mixture-of-experts). The **surprising result**: the explicit model gives essentially **no generalization benefit**, in-distribution *or* out-of-distribution. Even more surprising, careful analysis (linear decoding, auxiliary supervision, causal interventions) shows the bottleneck **does** learn the correct latent variables — you can linearly read `z` out of it, and causally intervene on it to get correct counterfactual predictions. So the failure is not in *finding* the latents; it is that the **downstream prediction network fails to learn how to use them**. When the authors replace the prediction network with the *true* prediction function `g` as an oracle, OOD generalization jumps dramatically and most tasks are solved.

**Initial takeaway.** Learning the right latent variables is **necessary but not sufficient** for robust ICL generalization. The community's "shortcuts are the problem" narrative is incomplete: banning shortcuts and correctly inferring task latents buys you *interpretability* (you can decode and causally manipulate the latents — useful for mechanistic interpretability and scientific discovery), but *not* better performance. The real bottleneck is the prediction function — end-to-end training does not, on its own, produce a prediction head that leverages clean latents in a way that extrapolates. This is a pointed negative result that redirects effort toward inductive biases in the *prediction* stage. It also complements paper 1: paper 1 says next-token prediction pushes toward simple (compressing) models; paper 2 says even when you additionally hand the model the correct low-dimensional structure, the prediction stage still fails to exploit it OOD — an architectural limitation, not merely a shortcut-avoidance problem.

### Phase 2: Graduate-Level Deep Dive

#### 2.1 Formal setup — parametric factorization of the posterior predictive

A task is a functional mapping $g : \mathcal{X}\times\mathcal{Z} \to \mathcal{Y}$, sending input $x$ to label $y$ through a task latent $z$ — e.g. $y = z^\top x$ (deterministic linear regression) or $y \sim \mathcal{N}(\cdot;\, z^\top x, \sigma^2)$ (stochastic). Training draws context sets $D = \{(x_i,y_i)\}_{i=1}^n$ sharing $g$ but with different latent realizations $z_1, z_2, \dots$. The goal of ICL is the **posterior predictive** $p(y^* \mid x^*, D)$ for a novel query $x^*$. Crucially, unlike paper 1's autoregressive/next-token setting, here the model is **non-causal**: a fixed context $D$ predicts a *single* query $y^*$, and all tokens attend freely.

The parametric assumption is a **conditional-independence factorization**: there exists a latent $z$ such that conditioning on it renders the prediction independent of the raw context,

$$
p(y^* \mid x^*, z, D) = p(y^* \mid x^*, z).
$$

This cleanly splits ICL into two stages:
1. **Context aggregation:** infer $z$ from $D$ so that the above conditional independence holds.
2. **Predictive modeling:** learn the map $g$ that turns $(x^*, z)$ into $y^*$.

#### 2.2 The implicit vs. explicit vs. implicit-proxy models (Appendix E formalism)

**Implicit model** (standard Transformer, parameters $\phi$): models the predictive distribution directly with unconstrained access to $D$,

$$
p_\phi(y \mid x, D), \qquad \arg\max_\phi \; \mathbb{E}_{x,y,D}\big[\log p_\phi(y \mid x, D)\big].
$$

There is no forced separation; the Transformer *jointly and implicitly* performs both latent inference and prediction, and — the concern — can route the query $x^*$ directly to the context via self-attention, enabling kernel-regression-style shortcuts.

**Explicit model** (parameters $\psi$ for context aggregation, $\gamma$ for prediction): interposes a **low-dimensional bottleneck** $z_\psi(D)$,

$$
p_\gamma\big(y \mid x, z_\psi(D)\big), \qquad z_\psi(D) = \text{Transformer}_\psi(D).
$$

The predictive distribution touches $D$ **only** through $z_\psi(D)$. Because $z_\psi(D)$ is a fixed-size vector of much lower dimension than $D$, and — critically — is **invariant to the query** $x^*$, the query cannot attend to individual context points; the aggregator is forced to summarize $D$ into generative factors. Architecturally this is exactly a **Conditional Neural Process** (Garnelo et al. 2018a), $p_\theta(y^*\mid x^*, D) = p_\theta(y^*\mid x^*, z_\psi(D))$, but with a Transformer aggregator and — the novel angle — an explicit interrogation of whether $z_\psi$ recovers the *true* latent $z$.

**Implicit proxy model** (Appendix E; the key interpolating control): identical architecture to the explicit model, but the latent is allowed to depend on the query,

$$
p_\gamma\big(y \mid x, z_\psi(D, x)\big).
$$

This breaks the query–latent conditional independence — $z_\psi(D,x)$ is effectively the final attention-layer output of an implicit model, followed by an MLP head. It interpolates between explicit and implicit while keeping a bottleneck, and it is the sharpest disambiguation experiment (§2.4).

The expected trade-off (stated a priori): the explicit model should win when the true model is **parametric and low-dimensional** (few $z$ suffice, e.g. linear regression); the implicit model should win when the task is **non-parametric or very high-dimensional** — e.g. Gaussian-process regression with an RBF kernel, where the "latent" is a point in an infinite-dimensional function space that cannot fit in a finite bottleneck, and where query prediction inherently requires kernel similarities to all context points.

#### 2.3 The central negative result and its dissection

The training objective is maximum likelihood (regression → MSE, classification → cross-entropy), with **parameter-count controlled** between implicit and explicit models. Two evaluation regimes:
- **In-distribution (ID):** $z, D, x^*$ all from the training distribution.
- **Out-of-distribution (OOD):** for synthetic regression/classification, $z$ and $D$ as in training but queries $x^*$ drawn from a Gaussian with **3× the standard deviation** (extrapolation along the domain); for compositional/reasoning tasks, **held-out combinations** of latent parts (all part-values seen marginally at train time, but novel *combinations* at test — compositional generalization).

**Finding 1 — no benefit (Figures 2, 5).** ID: all models near-perfect, implicit marginally better. OOD: the explicit model provides **no significant advantage**; on synthetic regression all models degrade and the *implicit* model degrades slightly *less*; on the reusable MoE task the implicit model consistently beats the explicit one across all fractions of latent combinations seen.

To explain this, the authors pose two hypotheses: (H1) the explicit model failed to *extract* the true latent; (H2) it extracted it but failed to *use* it. They find strong evidence for **H2** via four probes:

**Probe A — auxiliary latent supervision (Figure 3, purple).** Add a supervised term forcing the bottleneck to recover the true latent up to a linear map:

$$
\mathcal{L}_{\text{aux}} = \big\| z - W z_\psi \big\|^2, \qquad W \text{ learnable},
$$

backpropagated through the aggregator alongside the task loss. Result: **no improvement** (except a minor bump on Raven's PM). If bad latent inference were the problem, this should have fixed it. It didn't → H1 rejected.

**Probe B — linear decodability without supervision (Figure 4a).** Train a *separate* linear probe to predict the true $z$ from $z_\psi$ (no gradient into the aggregator). The true latent is **accurately recovered up to linear reparameterization** — decoding from the bottleneck far outperforms decoding directly from concatenated context examples. So the aggregator learns the latent *for free*, confirming H1 is not the issue.

**Probe C — oracle prediction function (Figure 3, green).** Hard-code the true $g$ as the prediction head (e.g. for linear regression, linearly project $z_\psi$ to weight-dimension and take the dot product with $x^*$). Result: **substantially better OOD generalization**, effectively solving most tasks. This is the smoking gun for H2: the latents were fine; the *learned* prediction head was the failure point.

**Probe D — causal counterfactual intervention via Distributed Alignment Search (DAS, Figure 4b, 9).** Linear decodability alone can be a spurious correlation (a feature decodable but unused). DAS (Geiger et al. 2023b) searches for a subspace (here a 10-dim orthogonal projection $\Pi$ at the bottleneck) that, when overwritten with a source run's value, causally produces the correct *counterfactual* prediction under the swapped latent. The metric is **Interchange Intervention Accuracy (IIA)**, reported relative to a no-intervention baseline: $\frac{\text{IIA} - \text{BASELINE}}{1 - \text{BASELINE}}$. Result: the explicit model's bottleneck **can** be intervened on to yield correct counterfactuals; the implicit model **cannot** (its latents are distributed across layers/positions and are not causally clean). → explicit models are genuinely more interpretable, even though not more performant.

#### 2.4 The implicit-proxy disambiguation and boundary conditions

The implicit-proxy model (§2.2) is the decisive test: it keeps a bottleneck but lets $z_\psi(D,x)$ depend on the query. Result (Figure 11): as queries move further OOD, the implicit proxy **predicts better** than the explicit model **despite recovering the true latents worse**. This directly demonstrates that *superior latent inference does not imply superior prediction* — the prediction stage, not the inference stage, governs OOD success.

**Boundary conditions the paper is careful to note:**
- **Nonlinear (MLP) regression is an exception** — here even the oracle-$g$ explicit model performs poorly, because inferring an MLP's weight matrix from a handful of context points is too hard; the model abandons parametric inference for a non-parametric solution, and even latent decoding fails.
- **Classification vs. regression:** OOD is generally easy for classification (decision boundaries lie within the training domain and don't move outside it), so oracle prediction functions barely help; regression functions keep changing beyond the training domain, so OOD is genuinely hard and oracle-$g$ helps a lot.
- **Extreme shortcut injection (Figure 7):** training queries sampled *near* context points (incentivizing nearest-neighbour behaviour) then evaluating far from context — implicit models suffer much more under this manipulation, because explicit models distil the latent independent of the query. This is the *one* regime where the explicit inductive bias demonstrably helps.
- **Scaling (Figure 6):** latents decode more easily from the bottleneck with lower input dimensionality, longer context, and larger models — but better decoding never translates into a performance edge over the implicit model.

#### 2.5 Significance and relation to the corpus

The paper is a **construct-validity check** on the latent-variable / Bayesian-inference story of ICL (Groups A and B of this corpus). It concedes the story's premise — Transformers *can* infer the right latents — but severs the assumed link to generalization. For the project, the transferable lesson is architectural: **a clean, interpretable, low-dimensional task representation is not self-justifying**; the module that *consumes* that representation needs its own inductive biases (e.g. a known or structured prediction function) to extrapolate. It also supplies the exact bottlenecked architecture that paper 1 reuses, and it motivates the amortized-inference direction of paper 3 (better inference machinery is not enough if the generative/prediction model is weak).

### Appendix: Section-by-Section Backbone

- **Abstract.** Test whether explicitly inferring task latents (via a bottleneck that blocks attention shortcuts) improves ICL. Finding: little difference vs. standard Transformer; biasing toward task-relevant latents does not improve OOD. The bottleneck *does* extract the latents, but downstream processing fails to use them → intrinsic Transformer limitation; latent inference aids interpretability, not generalization.
- **§1 Introduction.** Parametric (infer-$z$-then-predict) vs. non-parametric shortcut (kernel-regression/induction-head) views of ICL. Attention $\approx$ kernel regression (Tsai et al. 2019, Han et al. 2023). Hypothesis to test: does banning shortcuts (parametric bias) improve ICL? Introduces implicit (standard Transformer) vs. explicit (bottlenecked) models. Contributions: (i) framework to test whether parametric ICL generalizes better OOD; (ii) analyze benefit of inferring true latents; (iii) identify the prediction-function/latent-utilization flaw.
- **§2 Notation.** Dataset $D$, inputs $x\in\mathcal X$, outputs $y\in\mathcal Y$, task map $g:\mathcal X,\mathcal Z\to\mathcal Y$; query $x^*$/target $y^*$; $\psi$ = context-aggregator params, $\gamma$ = prediction params (explicit), $\phi$ = single implicit-model params.
- **§3 Implicit vs. Explicit Inference.** Non-causal ICL over draws of context with shared $g$, varying $z$. MLE objective (Eq. 1). Decomposition into **context aggregation** (infer $z$ s.t. $p(y^*|x^*,z,D)=p(y^*|x^*,z)$) and **predictive modeling** (learn $g$). Implicit model $p_\phi(y^*|x^*,D)$ vs. explicit model $p_\gamma(y^*|x^*,z_\psi(D))$ with low-dim query-invariant bottleneck. A-priori trade-off: explicit better for low-dim parametric; implicit better for non-parametric/high-dim (GP-RBF latent is infinite-dimensional). Aim: not to build the best explicit model but to minimally remove shortcuts.
- **§4 Experiments.**
  - *Task setup:* regression (linear, nonlinear-MLP, sinusoid, GP, Hodgkin–Huxley ODE), classification (linear, nonlinear-MLP), compositional (reusable modular MoE, Alchemy, Raven's PM, Gene Targeting).
  - *Training/eval:* MSE/CE, param-count controlled; ID = same distribution; OOD = 3×-wider query variance (synthetic) or held-out latent combinations (compositional). Privileged oracles: known decoder ($g$) and known latent (auxiliary loss).
  - *Result 1:* explicit does **not** outperform implicit — ID nearly tied (implicit marginally better), OOD no explicit benefit; MoE implicit consistently better (Fig 5).
  - *Result 2:* explicit models **infer** correct latents (auxiliary loss gives no gain — Fig 3 purple; latents linearly decodable — Fig 4a) but **don't use** them; **oracle prediction $g$** yields large OOD gains (Fig 3 green) → H2 confirmed.
  - *Interpretability:* linear decoding of latents (Fig 4a) + DAS counterfactual intervention (Fig 4b) succeed for explicit, fail for implicit → interpretability + scientific-discovery value.
  - *Scaling (Fig 6):* latent decodes better with lower dim / longer context / bigger model, but no performance edge.
  - *aux_direct vs. aux_decoded (Fig 8):* forcing $\|z-z_\psi\|^2$ directly vs. through a linear decoder — neither helps OOD.
  - *Extreme shortcut injection (Fig 7):* queries near context at train, far at eval — implicit suffers more; explicit distills query-independent latent.
- **§5 Conclusion.** Learning correct latents is **not sufficient** for better ID/OOD generalization; end-to-end optimization fails to learn a prediction model that leverages the latents. Bottlenecks buy interpretability, not robustness.
- **Appendix A — Related Work.** ICL as layer-activation task parameterization; controlled-setting ICL (linear regression, HMMs, grammars, Turing machines) approaching Bayes-optimal (Xie et al.). Shortcuts (induction heads, kernel regression $p(y_q|x_q,x_{1:n})\propto\sum_i K(x_i,x_q)y_i$; task vectors). Neural Processes: CNP $p_\theta(y^*|x^*,z_\psi(D))$; NP $p_\theta(y|x^*,D)=\int p_\theta(y|x,z)p_\theta(z|D)dz$ trained via ELBO with amortized $q_\phi$. Meta-learning (inner/outer loop). Mechanistic interpretability (linear probes and their unreliability → causal methods/DAS).
- **Appendix B — Tasks (full generative specs).** Linear ($y=z^\top x$), nonlinear-MLP ($y=f_z(x)$, 64-hidden), sinusoid ($y=\sum_{i=1}^K \alpha_i\sin(2\pi\lambda_i x)$, $K=3$, $\lambda\sim U(0,5)$, $\alpha\sim U(-1,1)$), GP-RBF ($Y\sim\mathcal N(0,K(X,X))$), Hodgkin–Huxley ODE (membrane-potential dynamics; latents $z=\{\bar g_{Na},\bar g_K\}$; 6,400 conductance pairs); classification (linear/nonlinear with softmax); MoE ($y=g_{z_5}\circ\cdots\circ g_{z_1}(x)$, $L=K=5$); Alchemy (167,424 environments); Raven's PM (symbolic, 4 attributes × 40 values); Gene Targeting (Perturb-seq CRISPR, 5000-dim expression).
- **Appendix C — Model Details.** Implicit = 8-layer Transformer; explicit aggregator = 4-layer/256-d/512-MLP/4-head Transformer; predictor = 3-hidden-512 MLP *or* matched Transformer. Prompt formats with mask token $\varnothing$. Context $n\in[16,128]$. Scaling configs. **C.4 DAS:** learn projection $\Pi\in\mathbb R^{d\times10}$ at a location, intervene base with source's subspace value, train with CE against true counterfactual; evaluate via IIA relative to baseline.
- **Appendix D — Analysis.** Explicit models sufficiently uncover latents; still don't generalize better; learned downstream prediction is sub-optimal (reinforced by implicit-proxy Fig 11: worse latent inference but better prediction OOD). Classification vs. regression OOD asymmetry. Oracle-$g$ fails for nonlinear-MLP regression (latent too hard to infer). Output-dimensionality and context-aggregator-size ablations.
- **Appendix E — Mathematical Formalism.** Implicit $p_\phi(y|x,D)$ (Eq. 2); explicit $p_\gamma(y|x,z_\psi(D))$; **implicit proxy** $p_\gamma(y|x,z_\psi(D,x))$ (breaks query–latent conditional independence) — plate diagrams.

---

## 3. Hu et al. 2024 — Amortizing Intractable Inference in Large Language Models

**Venue:** ICLR 2024. **Group:** Mila – Quebec AI Institute / Université de Montréal (+ Oxford). Authors: Edward J. Hu*, Moksh Jain*, Eric Elmoznino, Younesse Kaddar, Guillaume Lajoie, Yoshua Bengio, Nikolay Malkin. (Same neighbourhood as papers 1–2 via Elmoznino/Lajoie; adds the Bengio/Malkin GFlowNet line.)
**PDF:** `docs/project/references/in_context_learning/sources/Hu et al. 2024 - Amortizing intractable inference in large language models.pdf`
**Code:** https://github.com/GFNOrg/gfn-lm-tuning

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction — the core problem.** A large language model (LLM) is trained to do one thing: predict the next token given the tokens before it (left-to-right). That makes *forward* sampling easy — start with a prompt, generate a continuation. But an enormous number of genuinely useful tasks are *not* left-to-right. **Infilling** (fill the gap between a known beginning and a known ending), **constrained generation** (produce text that satisfies some property), and — the paper's headline case — **chain-of-thought reasoning** (find the intermediate reasoning steps that best explain how a question leads to its answer) all require sampling from a *different* conditional distribution than the one the LLM was trained on. These other conditionals are **intractable**: you can *score* any candidate (evaluate its probability under the LLM), but you cannot cheaply *sample* from the posterior one token at a time. This paper's goal is to give LLMs a principled way to sample from these intractable posteriors.

**Key findings — the main result.** The authors frame chain-of-thought reasoning as a **latent-variable model**: an input `X` (question) generates an output `Y` (answer) through a hidden reasoning chain `Z`. Answering well means sampling `Z` from the **Bayesian posterior** `p(Z | X, Y) ∝ p_LM(X Z Y)` — the reasoning chains that make the correct answer likely. Rather than the usual tools (MCMC, which is slow and hard to tune for text; or reward-maximizing RL like PPO, which collapses onto a few high-reward modes), they **fine-tune the LLM itself to become an amortized sampler** of this posterior, using **GFlowNets** (Generative Flow Networks) — a class of diversity-seeking RL algorithms that train a policy to generate objects with probability *proportional to a reward*, rather than to maximize that reward. The reward here is just the LLM's own joint score `p_LM(X Z Y)`, so no external labels for `Z` are needed. A striking motivating demo: asked to "generate a random integer 0–100," a base LLM is wildly non-uniform and PPO barely fixes it (because rewarding all valid numbers equally gives zero policy gradient), while GFlowNet fine-tuning matches the uniform target almost exactly (KL from 3.37 down to ~1e-4). On real tasks the payoff is large: **+10.9%** over supervised fine-tuning on subjectivity classification with only 10 labelled examples, and it **outperforms supervised fine-tuning and PPO by 63%** on multi-step integer arithmetic with tool use — with especially big gains **out-of-distribution** (longer arithmetic expressions than seen in training).

**Initial takeaway.** The connection to in-context learning is conceptual and important: this paper shows that the *same* reasoning an LLM does "in context" (produce a chain of thought, then answer) can be recast as **Bayesian posterior inference over a latent variable**, and that you can **amortize** that inference into the weights of the model itself. The GFlowNet-tuned policy `q(Z | X)` is literally a learned, reusable inference network — it generalizes to unseen inputs `X` and, because it captures the *full diversity* of the posterior (not one mode), it enables **Bayesian model averaging**: sample several reasoning chains, answer with each, take a majority vote. Diversity is not a nicety here — it is what drives the out-of-distribution robustness and defends against reward misspecification (the "overoptimization" failure mode of RLHF-style tuning). This is the corpus's clearest statement of "ICL / reasoning = amortized Bayesian inference," and it operationalizes it as a training algorithm rather than a post-hoc interpretation.

### Phase 2: Graduate-Level Deep Dive

#### 3.1 Intractable conditionals in autoregressive LLMs

An autoregressive LLM factorizes a sequence distribution as ordered conditionals,

$$
p(w_{1:N}) = p(w_1)\,p(w_2\mid w_1)\cdots p(w_N\mid w_{1:N-1}),
$$

which makes *left-to-right* sampling trivial but *any other* conditional intractable. The paper catalogs three intractable families:
- **Tempered / contrastive sampling:** $q(Z\mid X)\propto p_{LM}(XZ)^{1/T}$ with $T<1$ (peaky, high-likelihood continuations), or the contrastive $q(Z\mid X)\propto p_{LM}(XZ)^\alpha p_{LM}(Z)^\beta$ ($\beta<0$) that down-weights generic continuations.
- **Infilling / reverse generation:** $q(Z\mid X,Y)\propto p_{LM}(XZY)$ with $X,Y$ fixed (reverse generation is the special case $X=\varnothing$).
- **Constrained generation:** $q(Z)\propto p_{LM}(Z)\,c(Z)$ for an external constraint/classifier $c$.

All are "score-easy, sample-hard": evaluating $p_{LM}$ on a full sequence is one forward pass, but the token-wise conditionals needed to *sample* from the posterior are not available.

#### 3.2 Chain-of-thought as latent-variable inference

The core reframing. For a question–answer pair $(X,Y)$, the marginal likelihood decomposes by summing over latent reasoning chains $Z$ (Eq. 1):

$$
p(Y\mid X) = \sum_{Z} p_{LM}(ZY\mid X) = \sum_{Z} p_{LM}(Y\mid XZ)\,p_{LM}(Z\mid X),
$$

with $p_{LM}(Z\mid X)$ the **conditional prior** over reasoning chains and $p_{LM}(Y\mid XZ)$ the **likelihood** of the answer given the chain. Reasoning = sampling from the **posterior** (Eq. 2):

$$
p_{LM}(Z\mid X,Y) = \frac{p_{LM}(XZY)}{\sum_{Z'} p_{LM}(XZ'Y)} \;\propto\; p_{LM}(XZY).
$$

The LVM is useful precisely when the marginal $p_{LM}(Y\mid X)$ is *harder* to model than the pieces $p_{LM}(Z\mid X)$ and $p_{LM}(Y\mid XZ)$ — i.e. a hard inference is broken into a chain of easier ones. The normalizing sum over $Z'$ (all possible reasoning chains) is what makes the posterior intractable.

**Learning the model too — variational EM.** Beyond sampling $Z$, one can fine-tune $p_{LM}$ to maximize the data likelihood of $(X,Y)$ under the LVM. Because $p_{LM}(X,Y)=\sum_Z p_{LM}(XZY)$ is intractable, use **variational EM**:
- **E-step:** draw $Z\sim p_{LM}(Z\mid X,Y)$ from the amortized posterior sampler (the GFlowNet).
- **M-step:** maximize $\mathbb{E}_{Z\sim p_{LM}(Z\mid X,Y)}\big[\log p_{LM}(XZY)\big]$ w.r.t. the LLM parameters.

Iterating = amortized inference (learn to sample the chain of thought) + supervised fine-tuning on self-generated chains. This is exactly the "+ Supervised fine-tuning" row in the subjectivity experiment.

#### 3.3 GFlowNets as amortized samplers — the SubTB objective and its derivation

A GFlowNet learns a policy that samples terminal objects $Z=z_1z_2\cdots z_n\top$ (with stop symbol $\top$) with probability **proportional to a reward** $R:\mathcal{Z}\to\mathbb{R}_{>0}$. The generative process mirrors autoregressive generation: at step $i$ sample $z_i\sim q_{GFN}(z_i\mid z_{1:i-1})$, append, repeat until $\top$. The marginal likelihood of terminating at $Z$ is

$$
q^\top_{GFN}(Z) = \prod_{i=1}^{n} q_{GFN}(z_i\mid z_{1:i-1})\, q_{GFN}(\top\mid z),
$$

and the training goal is $q^\top_{GFN}(Z)\propto R(Z)$. **Setting the reward $R(Z)=p_{LM}(XZY)\propto p_{LM}(Z\mid X,Y)$ makes the converged GFlowNet a sampler of the target posterior** — this is the crux that ties GFlowNets to the LVM of §3.2.

**Derivation of the objective (Appendix A.2 → Eq. 3).** Start from the general **subtrajectory balance (SubTB)** constraint (Madan et al. 2023) for a partial trajectory $\tau = s_m\to\cdots\to s_n$, with forward policy $P_F$, backward policy $P_B$, and state-flow function $F$ (Eq. 4):

$$
\mathcal{L}_{SubTB}(Z;\theta) = \left( \log \frac{F(s_m)\prod_{i=m}^{n-1} P_F(s_{i+1}\mid s_i)}{F(s_n)\prod_{i=m}^{n-1} P_B(s_i\mid s_{i+1})} \right)^2 .
$$

Two simplifications specialize this to autoregressive text generation:
1. **Tree-structured state space.** Left-to-right generation in a fixed order means each state has a *unique* parent, so the backward policy is trivial: $P_B(s\mid s')=1$. The denominator product drops out.
2. **Every state is terminable** (Deleu et al. 2022). At convergence $R(s_n^\top)=F(s_n)\,P_F(\top\mid s_n)$, so substitute $F(s_n)=R(s_n^\top)/P_F(\top\mid s_n)$. This **eliminates the separately-parameterized flow function** $F$ — the only learned object is the forward policy $P_F \equiv q_{GFN}$.

Summing the resulting balance condition over all partial trajectories $0\le i<j\le n$ with equal weight ($\lambda=1$) gives the final per-sequence objective (Eq. 3):

$$
\mathcal{L}(Z;\theta) = \sum_{0\le i<j\le n} \left( \log \frac{R(z_{1:i}\top)\prod_{k=i+1}^{j} q_{GFN}(z_k\mid z_{1:k-1})\, q_{GFN}(\top\mid z_{1:j})}{R(z_{1:j}\top)\, q_{GFN}(\top\mid z_{1:i})} \right)^2 .
$$

For sequence generation this SubTB objective is **equivalent to the path-consistency objective** (Nachum et al. 2017) in maximum-entropy RL (Haarnoja et al. 2017) — placing GFlowNet fine-tuning squarely in the max-ent-RL family, but with the distinguishing goal of *matching* the full reward distribution rather than maximizing expected reward.

**Off-policy training.** Because Eq. 3 can be driven to zero for *all* trajectories simultaneously (given capacity), gradients may use trajectories from *any* full-support behaviour policy. The mini-batch mixes three sources: (1) the current policy $q_{GFN}$, (2) a tempered version of it, and (3) a **replay buffer** of past trajectories — crucial for exploring the combinatorially large sequence space. Fine-tuning is done with **LoRA** (low-rank adaptation) for hardware efficiency, not full fine-tuning.

#### 3.4 Amortization, conditioning, and Bayesian model averaging (Table 1)

The GFlowNet policy is parameterized as an autoregressive LM that generates $Z$ token-by-token. Making $X$ (and optionally $Y$) an *input* to the policy is what turns per-instance inference into **amortized** inference that generalizes to unseen inputs. Two conditioning regimes:
- **Condition on $X$ only** — for reasoning/classification where each $X$ has a single correct $Y$ and $Y$ is unknown at test time. The policy $q_{GFN}(Z\mid X)$ is initialized as a copy of $p_{LM}$ conditioned on prefix $X$, then GFlowNet-tuned. Sampling $Z$ is an **inverse problem**: infer $Z$ given conditional prior $p_{LM}(Z\mid X)$ and observation $Y$ under likelihood $p_{LM}(Y\mid XZ)$. Prediction for a new $X$ is **Bayesian model averaging**: draw $Z\sim q_{GFN}(Z\mid X)$, then $Y\sim p_{LM}(Y\mid XZ)$, aggregate (majority vote). In this view the GFlowNet is a Bayesian model where $Z$ are conditionally-sampled "parameters" transforming $X$ into $Y$ — analogous to an LM cascade / deep language network.
- **Condition on both $X$ and $Y$** — for infilling, where the map $X\to Y$ is one-to-many and $Y$ is available at test time; here $Z$ itself is the object of interest, so the policy $q_{GFN}(Z\mid X,Y)$ conditions on a prompt containing both.

#### 3.5 Empirical results — why distribution-matching beats mode-seeking

- **Random-number demo (§2, Fig. 2):** the sharpest illustration. Rewarding all valid integers equally gives an *expected policy gradient of zero*, so PPO cannot un-skew the pretraining bias (Benford-law-like preference for numbers starting with '1'); it only learns validity (95.8% valid, still skewed). GFlowNet matches the uniform target (100% valid, KL $3.37\to 9.75\times10^{-5}$). Distribution-matching succeeds exactly where reward-maximization is degenerate.
- **Sentence continuation (§4.1, Fig. 3):** reward $R(Z)=p_{LM}(Z\mid X)^{1/T}$, $0<T<1$. GFlowNet samples higher-max-likelihood *and* more diverse continuations than diverse beam search — even when beam search is given **5× the compute** — because it amortizes over prompts in a single learned pass.
- **Story infilling (§4.2, Table 2):** ROCStories, generate the 4th sentence (the turning point) given sentences 1–3 and 5; $q_{GFN}(Z\mid X,Y)$ beats prompting and supervised fine-tuning on BERTScore/BLEU-4/GLEU-4/GPT-4-eval by accounting for the ending while generating.
- **Subjectivity classification (§4.3, Table 3):** low-data regime; latent rationale $Z$. GFlowNet fine-tuning beats supervised fine-tuning by large margins with few labels (e.g. **71.4% vs 64.3% at 10 examples**), and one EM step ("+ supervised fine-tuning") often helps further. At test time: sample 10 rationales, answer with each, majority-vote.
- **Arithmetic with tool use (§4.4, Table 4):** GPT-J 6B, calculator restricted to two-term expressions (planning under a limited tool). Trained on 3–4 operands; evaluated in-distribution (3–4) and OOD (5 operands). GFlowNet: **95.2% / 75.4% / 40.7%** vs supervised fine-tuning 72.1 / 19.6 / 12.8 and PPO 30.6 / 13.7 / 5.6. PPO fails via **reward over-optimization** — high-reward-but-spurious sequences that aren't even valid tool calls (the misspecified-reward pathology). Distribution-matching avoids mode collapse and is robust to reward misspecification.

#### 3.6 Significance and relation to the corpus

This paper is the corpus's **amortized-inference** anchor. Where paper 1 says next-token prediction *is* compression/Occam and paper 2 says clean latent inference isn't enough for OOD prediction, paper 3 provides a *constructive training method* for the latent-variable / Bayesian view: amortize the intractable posterior over reasoning chains into the LLM's weights via a diversity-seeking objective. The throughline "ICL / reasoning = (amortized) Bayesian inference over latents" is here made an algorithm with measurable wins, and it explicitly connects to Bayesian model averaging — the same posterior-predictive machinery invoked in Groups A/B. For the project, the transferable ideas are: (i) treat intractable posterior sampling as amortizable into a network; (ii) prefer distribution-matching (GFlowNet/max-ent) over reward-maximizing RL when *diversity* and *robustness to reward misspecification* matter; (iii) diversity of latent samples directly buys OOD generalization via averaging. **Limitations** the authors flag: experiments ≤6B params; on-policy exploration in complex-latent problems remains open; the method improves *inference*, not the LLM's underlying knowledge — so hallucination/miscalibration tied to knowledge representation are not addressed.

### Appendix: Section-by-Section Backbone

- **Abstract.** LLMs' next-token training limits tractable querying to left-to-right sampling; many tasks (continuation, infilling, constrained generation) need intractable-posterior sampling. Solution: amortized Bayesian inference via GFlowNet fine-tuning (diversity-seeking RL). Recast chain-of-thought as latent-variable modeling → data-efficient adaptation to multi-step reasoning + tool use.
- **§1 Introduction.** LLMs as knowledge stores queryable only by prefix-conditioned sampling. Reasoning as probabilistic inference; CoT as intractable posterior inference: $p(Y|X)=\sum_Z p_{LM}(Y|XZ)p_{LM}(Z|X)$ (Eq. 1); posterior $p_{LM}(Z|X,Y)\propto p_{LM}(XZY)$ intractable. MCMC (slow, hard proposals) and reward-max RL/PPO (mode collapse, overoptimization) inadequate. Amortized inference via GFlowNets: sample $\propto$ reward $p_{LM}(XZY)$. Contributions: (1) general amortized-sampling algorithm for intractable LLM posteriors; (2) probabilistic CoT fine-tuning; (3) results on continuation, NL reasoning, arithmetic+tools, infilling.
- **§2 Motivating example — random numbers.** Sample from a target given an unnormalized density. Base GPT-J skewed (50.5% valid); PPO learns validity but stays skewed (equal reward → zero gradient); GFlowNet matches uniform (KL $3.37\to 9.75\text{e-}5$). Distribution-matching vs reward-maximization.
- **§3 Fine-tuning LLMs to sample from intractable distributions.**
  - **§3.1 Problem.** Ordered-conditional factorization; intractable families — tempered/contrastive $q(Z|X)\propto p_{LM}(XZ)^{1/T}$; infilling/reverse $q(Z|X,Y)\propto p_{LM}(XZY)$; constrained $q(Z)\propto p_{LM}(Z)c(Z)$. Table 1 object glossary.
  - **§3.2 Reasoning through latent variables.** Posterior Eq. 2; LVM useful when marginal harder than pieces; variational EM (E-step: sample $Z\sim$ amortized posterior; M-step: maximize $\mathbb{E}_Z\log p_{LM}(XZY)$).
  - **§3.3 Amortized inference with GFlowNet objectives.** GFlowNet basics; $q^\top_{GFN}(Z)=\prod q_{GFN}(z_i|z_{1:i-1})q_{GFN}(\top|z)$; goal $q^\top_{GFN}\propto R$; SubTB objective (Eq. 3) ≡ path-consistency/max-ent RL; off-policy training (policy + tempered + replay buffer); reward $R(Z)=p_{LM}(XZY)$; conditioning on $X$ (reasoning, BMA prediction) vs $X,Y$ (infilling); LoRA fine-tuning.
- **§4 Empirical results.**
  - **§4.1 Sentence continuation** — OpenWebText, GPT-2 XL 1.5B; reward $p_{LM}(Z|X)^{1/T}$; beats diverse beam search on likelihood+diversity even at 5× compute (Fig. 3).
  - **§4.2 Infilling stories** — ROCStories, GPT-2 Large; $q_{GFN}(Z|X,Y)$; beats prompting + supervised FT on BERTScore/BLEU/GLEU/GPT-4 (Table 2).
  - **§4.3 Subjectivity classification** — SUBJ, instruct-GPT-J 6B, low-data; latent rationale; 71.4% vs 64.3% (SFT) at 10 examples; +EM step (Table 3).
  - **§4.4 Arithmetic step-by-step** — GPT-J 6B + calculator tool (2-term limit); train 3–4 operands, test OOD 5; GFlowNet 95.2/75.4/40.7 vs SFT and PPO; PPO overoptimizes misspecified reward (Table 4).
- **§5 Further related work.** Sampling intractable marginals (MCMC, SMC, Gibbs via masked LMs); GFlowNets as variational inference for structured Bayesian posteriors; CoT reasoning (self-consistency aggregation, STaR fine-tuning on successful chains, MCMC CoT).
- **§6 Conclusion.** GFlowNet fine-tuning gives better fidelity–diversity trade-off, sample efficiency, and generalization than MLE or reward-max RL; converts compute into test-time performance without extra data. Future: universal reasoner $q(Z|X)$ shared across tasks; better base LLM; epistemic-uncertainty quantification via multiple samples; richer (non-left-to-right) latent generative processes. **Limitations:** ≤6B params; on-policy exploration open; improves inference not knowledge (hallucination/miscalibration untouched).
- **Appendix A.1 — Glossary of RL for LLMs.** RL, policy, reward (= unnormalized posterior for GFlowNets), distribution-matching, policy-gradient methods.
- **Appendix A.2 — Learning objective.** General SubTB (Eq. 4) with $P_F$, $P_B$, flow $F$; for left-to-right generation $P_B=1$ (tree) and terminable-state substitution $F(s_n)=R(s_n^\top)/P_F(\top|s_n)$ eliminate the flow function → only $q_{GFN}$ learned; sum over subtrajectories ($\lambda=1$) → Eq. 3.
- **Appendices B–E.** Per-task details: sentence-continuation granularity rationale; infilling setup; subjectivity prompts (Table D.1/D.2); arithmetic replay-buffer seeding (50 $(X,Z,Y)$ demos + 1000 $(X,Y)$), tool-use mechanics, and PPO failure illustrations.

---

## 4. Xie et al. 2022 — An Explanation of In-Context Learning as Implicit Bayesian Inference

**PDF:** `docs/project/references/in_context_learning/sources/Xie et al. 2022 - An Explanation of In-context Learning as Implicit Bayesian Inference.pdf` (ICLR 2022, arXiv:2111.02080v6)

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction / core problem.** Large language models such as GPT-3 can "learn" a new task just by reading a *prompt* of a few input-output examples (e.g. "Albert Einstein was German \n Mahatma Gandhi was Indian \n Marie Curie was ___") and then completing it correctly — *without any change to the model's weights*. This is called in-context learning (ICL). It is puzzling because the model was never explicitly trained to learn-from-examples, and the prompt (independent examples glued end-to-end) looks nothing like the flowing natural text the model was trained on. The paper asks: what property of the *pretraining data* makes this ability appear on its own?

**Key idea in plain terms.** The authors propose that ICL is *implicit Bayesian inference*. Imagine each document in the training corpus is generated by first secretly picking a "concept" (a topic / latent theme, e.g. "wiki-biography style: name → nationality → occupation → …") and then writing text consistent with that concept. To predict the next word well during pretraining, the model must silently *infer which concept is in play* and track it across sentences. At test time, when you feed it a prompt, the model does the same thing: it infers the *shared concept* that ties the prompt examples together, and uses it to answer. The more examples you give, the more sharply the model's internal belief about the concept concentrates on the right one — so accuracy rises.

**Key findings.**
- They formalize pretraining as a **mixture of Hidden Markov Models (HMMs)**: a latent concept $\theta$ sets the HMM's transition matrix, then a document is sampled from that HMM.
- They **prove** that, despite a genuine distribution mismatch (prompts are low-probability under pretraining because they concatenate independent examples), the in-context predictor's error becomes **optimal as the number of examples $n \to \infty$**, provided a "distinguishability" condition holds — i.e., the true prompt concept is separable enough from all other concepts.
- When distinguishability fails (e.g. a continuous concept space), the error still **shrinks with the length $k$ of each example** (roughly $O(1/k)$). So *information in the inputs themselves*, not only the input→output mapping, fuels ICL.
- They build a small synthetic dataset, **GINC**, and show that Transformers *and* LSTMs trained on it reproduce real large-scale phenomena: accuracy rises with more examples and longer examples; larger models do better *even at identical pretraining loss*; example ordering matters a lot; and zero-shot can beat few-shot.

**Initial takeaway.** ICL need not be a mysterious special capability. It can *emerge for free* from ordinary next-token pretraining, as long as the data has **latent long-range structure** (a concept shared across a document). The prompt then acts as evidence, and the model "selects" the matching concept by marginalization — Bayesian inference happening implicitly inside a model that was only ever trained to predict the next token. Crucially, an ablation shows the **mixture-of-concepts structure is necessary**: training on a single concept, or on data with all transitions but no latent structure, kills ICL.

### Phase 2: Graduate-Level Deep Dive

#### 2.1 Generative model and the posterior-predictive target

**Pretraining distribution.** A latent concept $\theta \in \Theta$ (with prior $p(\theta)$) parameterizes an HMM over hidden states $h_t \in \mathcal H$ emitting observed tokens $o_t \in \mathcal O$. A length-$T$ document has marginal likelihood

$$p(o_1,\dots,o_T) = \int_{\theta\in\Theta} p(o_1,\dots,o_T \mid \theta)\, p(\theta)\, d\theta ,$$

where $p(o_1,\dots,o_T\mid\theta)$ is the HMM likelihood: $\theta$ fixes the hidden-state transition matrix.

**Prompt distribution.** A prompt concatenates $n$ i.i.d. training examples and one test input, all conditioned on a **shared prompt concept** $\theta^\*$:

$$[S_n, x_{\text{test}}] = [x_1,y_1,o^{\text{delim}}, x_2,y_2,o^{\text{delim}},\dots,x_n,y_n,o^{\text{delim}}, x_{\text{test}}] \sim p_{\text{prompt}}.$$

Each example $O_i=[x_i,y_i]$ (length $k$; input $x_i = O_i[1{:}k{-}1]$, output $y_i=O_i[k]$) is generated by: (1) draw a start hidden state $h_i^{\text{start}}$ from a prompt start distribution $p_{\text{prompt}}$; (2) roll the HMM forward under $\theta^\*$. Delimiters $o^{\text{delim}}$ separate examples.

**The object of study is the posterior-predictive distribution** — the conditional under the *pretraining* model of a completion given the prompt, marginalizing the concept:

$$p(\text{output}\mid \text{prompt}) = \int_{\text{concept}} p(\text{output}\mid \text{concept}, \text{prompt})\, p(\text{concept}\mid \text{prompt})\, d(\text{concept}). \tag{1}$$

The paper assumes the LM has fit the pretraining $p$ exactly (infinite data / capacity), so ICL reduces to characterizing $(1)$ under a *prompt* drawn from the mismatched $p_{\text{prompt}}$. The in-context predictor is $f_n(x_{\text{test}}) = \arg\max_y p(y\mid S_n, x_{\text{test}})$, evaluated by the 0-1 risk $L_{0\text{-}1}(f_n) = \mathbb E_{x_{\text{test}},y_{\text{test}}\sim p_{\text{prompt}}}[\mathbf 1[f_n(x_{\text{test}})\ne y_{\text{test}}]]$.

#### 2.2 The posterior-concentration mechanism (the heart of the paper)

Apply Bayes' rule to expand the predictor and drop the $x_{\text{test}}$-independent normalizer:

$$p(y\mid S_n,x_{\text{test}}) \propto \int_\theta p(y\mid S_n,x_{\text{test}},\theta)\, p(S_n,x_{\text{test}}\mid\theta)\, p(\theta)\, d\theta .$$

Now **divide numerator and denominator inside by the likelihood under the true concept $p(S_n,x_{\text{test}}\mid\theta^\*)$** (a constant in $\theta$), and expand over the test's start hidden state:

$$p(y\mid S_n,x_{\text{test}}) = \int_\theta \sum_{h^{\text{start}}_{\text{test}}\in\mathcal H} p(y\mid x_{\text{test}}, h^{\text{start}}_{\text{test}},\theta)\, p(h^{\text{start}}_{\text{test}}\mid S_n,x_{\text{test}},\theta)\, \exp\!\big(n\cdot r_n(\theta)\big)\, p(\theta)\, d\theta, \tag{6}$$

where the **average log-likelihood ratio** is defined as

$$r_n(\theta) = \frac{1}{n}\log \frac{p(S_n,x_{\text{test}}\mid\theta)}{p(S_n,x_{\text{test}}\mid\theta^\*)}.$$

The entire argument reduces to controlling $\exp(n\cdot r_n(\theta))$:
- For $\theta = \theta^\*$ the ratio is identically $1$.
- **Claim:** under distinguishability, $r_n(\theta) \to$ a *negative* constant for every $\theta\ne\theta^\*$, hence $\exp(n\, r_n(\theta)) = \big[p(S_n,x_{\text{test}}\mid\theta)/p(S_n,x_{\text{test}}\mid\theta^\*)\big] \to 0$.

So in the $n\to\infty$ limit, every off-target concept is exponentially suppressed and the integral collapses onto $\theta^\*$ — the model "selects" the prompt concept **by marginalization**, which is precisely implicit Bayesian model selection. Dominated convergence justifies exchanging limit and integral (probabilities bounded).

#### 2.3 Why $r_n(\theta)$ is negative — the factorization-through-delimiters trick

The technical obstacle: under the *pretraining* distribution the prompt examples are **not independent** (concatenation creates cross-example dependence), so the log-likelihood ratio does not trivially split into a sum. The fix uses the delimiter tokens to *approximately factorize* the prompt with only $O(1)$ error per example.

Assumptions that make this work:
- **Assumption 1 (Delimiter hidden states).** A subset $\mathcal D\subseteq\mathcal H$ emits $o^{\text{delim}}$ deterministically: $p(o^{\text{delim}}\mid h^{\text{delim}},\theta)=1$ for $h^{\text{delim}}\in\mathcal D$, and $=0$ otherwise. Observing a delimiter reveals the hidden state is *in* $\mathcal D$ (but not which element).
- **Assumption 2 (Bounded delimiter transitions).** $p(h^{\text{delim}}\mid h,\theta) < c_2$ for $\theta\ne\theta^\*$, $> c_1>0$ for $\theta^\*$; start prob $p(h^{\text{delim}}\mid\theta)\in[c_3,c_4]$.
- **Assumption 3 (Prompt-start shift bound).** $\max_{h^{\text{delim}}\in\mathcal D}\mathrm{TV}\big(p_{\text{prompt}}(h)\,\|\,p(h\mid h^{\text{delim}},\theta^\*)\big) < \Delta/4$, where $\Delta$ is the **margin** between the most and second-most likely prompt label.
- **Assumption 5 (Regularity).** Positive lower bounds on transition ($>c_5$), start ($\ge c_8$), and emission ($>c_6$) probabilities under $\theta^\*$, and $p(\theta)$ has full support on $\Theta$.

Writing $O_i^{\text{ex}} = [o^{\text{delim}}_{i-1}, O_i]$ (the $i$-th example plus its preceding delimiter), the likelihood factorizes as

$$p(S_n,x_{\text{test}}\mid\theta) = p(x_{\text{test}}\mid S_n,\theta)\, p(S_n\mid\theta) \approx \prod_{i=1}^n O(1)\, p(O_i\mid\theta). \tag{8}$$

The key step marginalizes over the delimiter hidden state $h^{\text{delim}}_{i-1}$. Because $p(h\mid O^{\text{ex}}_{1:i-1},\theta)=0$ for $h\notin\mathcal D$ (Assumption 1), the sum over all hidden states collapses to a sum over $\mathcal D$, and the Markov chain $O^{\text{ex}}_{1:i-1}\to h^{\text{delim}}_{i-1}\to O^{\text{ex}}_i$ decouples example $i$ from its history:

$$\prod_{i=1}^n O(1)\, p(O^{\text{ex}}_i\mid O^{\text{ex}}_{1:i-1},\theta) = \prod_{i=1}^n O(1)\sum_{h^{\text{delim}}_{i-1}\in\mathcal D} p(O_i\mid h^{\text{delim}}_{i-1},\theta)\, p(h^{\text{delim}}_{i-1}\mid O^{\text{ex}}_{1:i-1},\theta) \approx \prod_{i=1}^n O(1)\, p(O_i\mid\theta). \tag{10}$$

The delimiter "resets" the chain: knowing you just saw a delimiter tells you the hidden state is in $\mathcal D$ but not which one, and Assumption 2 bounds $p(h^{\text{delim}}_{i-1}\mid O^{\text{ex}}_{1:i-1},\theta)=O(1)$. Substituting into $r_n$:

$$r_n(\theta) \le \frac1n\Big(O(n) + \sum_{i=1}^n \log\frac{p(O_i\mid\theta)}{p(O_i\mid\theta^\*)}\Big) \to O(1) + \mathbb E_{O\sim p_{\text{prompt}}}\Big[\log\frac{p(O\mid\theta)}{p(O\mid\theta^\*)}\Big]. \tag{11}$$

The expectation is a **difference of two KL divergences**:

$$\mathbb E_{O\sim p_{\text{prompt}}}\Big[\log\tfrac{p(O\mid\theta)}{p(O\mid\theta^\*)}\Big] = \mathrm{KL}\big(p_{\text{prompt}}(O)\,\|\,p(O\mid\theta^\*)\big) - \mathrm{KL}\big(p_{\text{prompt}}(O)\,\|\,p(O\mid\theta)\big).$$

The first KL is bounded by a constant (since $p_{\text{prompt}}$ and $p(\cdot\mid\theta^\*)$ are close on one example). The second decomposes into $O(k)$ per-token negative KL terms. **If the signal (the second KL, $\sim k$ terms) exceeds the mismatch error (the $O(1)$ terms), then $r_n(\theta)<0$**, and ICL succeeds.

#### 2.4 Distinguishability condition and the main theorem

Define per-token distributions $p^j_\theta(o) := p(O[j]=o\mid O[1{:}j{-}1],\theta)$ and $p^j_{\text{prompt}}$ analogously, with

$$\mathrm{KL}_j(\theta^\* \| \theta) := \mathbb E_{O[1:j-1]\sim p_{\text{prompt}}}\big[\mathrm{KL}(p^j_{\text{prompt}} \| p^j_\theta)\big], \tag{12}$$

and mismatch error terms

$$\epsilon^\theta_{\text{delim}} := 2(\log c_2 - \log c_1) + \log c_4 - \log c_3, \qquad \epsilon^\theta_{\text{start}} := \log(1/c_8). \tag{13}$$

**Condition 1 (Distinguishability).** $\theta^\*$ is distinguishable if for all $\theta\ne\theta^\*$,

$$\sum_{j=1}^k \mathrm{KL}_j(\theta^\*\|\theta) > \epsilon^\theta_{\text{start}} + \epsilon^\theta_{\text{delim}}. \tag{14}$$

The LHS grows with example length $k$ (more per-token evidence), so **longer examples improve distinguishability**.

**Theorem 1.** Under the Section-2 assumptions and Condition 1, as $n\to\infty$,

$$\arg\max_y p(y\mid S_n,x_{\text{test}}) \to \arg\max_y p_{\text{prompt}}(y\mid x_{\text{test}}), \tag{15}$$

hence $f_n$ achieves the Bayes-optimal 0-1 risk: $\lim_{n\to\infty} L_{0\text{-}1}(f_n) = \inf_f L_{0\text{-}1}(f)$.

*Proof sketch (Appendix D).* Once $\exp(n\,r_n(\theta))\to 0$ for $\theta\ne\theta^\*$, the integral $(6)$ splits into a dominant $\theta^\*$ term plus a residual $\int_{\theta\ne\theta^\*}\epsilon_\theta(y)p(\theta)d\theta$ with $\epsilon_\theta(y)\le\epsilon/2$. **Lemma 1** shows the $\arg\max$ of the $\theta^\*$-conditioned averaged predictive equals $\arg\max_y p_{\text{prompt}}(y\mid x_{\text{test}})$: the two predictive distributions, written in matrix form as $WTv$ and $Wu$, differ by at most $\|Tv-u\|_1 = 2\,\mathrm{TV}(p_{\text{prompt}}\|\textstyle\sum_i v_i\, p(\cdot\mid h^{\text{delim}}{=}i,\theta^\*)) < \Delta/2$ (using convexity of TV and Assumption 3). Since no output probability shifts by more than $\Delta/2$ and the label margin is $\Delta$, the $\arg\max$ is preserved. The residual is $< (\Delta/2-\Delta')/2$, so it cannot flip the $\arg\max$ either. $\square$

#### 2.5 Non-distinguishable case: error decays with example length

When Condition 1 fails (set $B$ of "confusable" $\theta$), the outputs of $\theta$ and $\theta^\*$ are close in KL, so the excess error is controllable.

**Theorem 2 (continuity).** Assuming a 2nd-order Taylor expansion of the per-token KL around $\theta^\*$,

$$\mathrm{KL}_j(\theta^\*\|\theta) = \tfrac12(\theta-\theta^\*)^\top I_{j,\theta^\*}(\theta-\theta^\*) + O(\|\theta-\theta^\*\|^3), \tag{16}$$

with $I_{j,\theta^\*}$ the per-token Fisher information and worst-case condition number $\gamma_{\theta^\*} = \max_j \lambda_{\max}(I_{j,\theta^\*}) / \min_j \lambda_{\min}(I_{j,\theta^\*})$, then for $k\ge 2$,

$$\lim_{n\to\infty} L_{0\text{-}1}(f_n) \le \inf_f L_{0\text{-}1}(f) + g^{-1}\!\Big(O\Big(\tfrac{\gamma_{\theta^\*}\sup_{\theta\in B}(\epsilon^\theta_{\text{start}}+\epsilon^\theta_{\text{delim}})}{k-1}\Big)\Big), \tag{17}$$

where $g(\delta)=\tfrac12\big((1-\delta)\log(1-\delta)+(1+\delta)\log(1+\delta)\big)$ is the multiclass-logistic calibration function. Since $g^{-1}$ is roughly linear for small argument, the excess risk decays like $O(1/k)$. **Theorem 3** gives an analogous $O(1/k)$ bound without continuity when $x_{\text{test}}$ has random length in $[2,k]$.

#### 2.6 Empirical verification on GINC

GINC = uniform mixture of 5 HMM concepts, ~10M pretraining tokens, prompts with $n\in[0,64]$, $k\in\{3,5,8,10\}$; target $y_{\text{test}}=\arg\max_y p_{\text{prompt}}(y\mid x_{\text{test}})$ (so intrinsic error is 0). Findings mirror the theory and real LMs:
- Accuracy rises with both $n$ and $k$ (Transformers and LSTMs).
- **Model scale helps beyond pretraining loss**: 12- and 16-layer Transformers share validation loss 1.33 (vocab 50) yet accuracy climbs 81%→85%.
- **Ablations**: single-concept pretraining, or random-transition pretraining, both flatten ICL — the *mixture-of-concepts latent structure* is essential; unseen concepts fail to extrapolate.
- Sensitivity to example order (10–40% swing across permutations) and zero-shot-beats-few-shot both reproduced.

**Relation to Bernstein–von Mises.** The classical BvM theorem (posterior concentrates on the MLE) does not directly apply because prompt examples are dependent under $p$ and drawn from a mismatched $p_{\text{prompt}}$. This paper's contribution is essentially a BvM-type concentration *under misspecification and dependence*, engineered via the delimiter-factorization.

### Appendix: Section-by-Section Backbone

- **Abstract / §1 Introduction.** ICL emerges when pretraining docs have long-range coherence; LM must infer a latent document concept; at test time it infers a shared prompt concept. Proven for a mixture-of-HMMs pretraining distribution despite prompt/pretraining mismatch. GINC synthetic dataset reproduces scaling, order-sensitivity, zero-shot>few-shot.
- **§2 In-context learning setting.** Pretraining: concept $\theta\sim p(\theta)$ → HMM document (Eq. 2). Prompt: $n$ i.i.d. examples + test input under shared $\theta^\*$ (Eq. 3), separated by delimiters. Mismatch: prompts are low-probability concatenations. In-context predictor $f_n$ and 0-1 risk defined. **§2.1 Assumptions 1–5** (delimiter hidden states; bounded delimiter transitions; prompt-start TV bound tied to margin $\Delta$; well-specification $\theta^\*\in\Theta$; regularity lower bounds).
- **§3 Theoretical analysis.** §3.1 high-level: expand predictor via Bayes (Eqs. 5–7), define $r_n(\theta)$, show $\exp(n r_n(\theta))\to0$ off-target ⇒ concept selection. §3.2 heuristic derivation: delimiter factorization (Eqs. 8–10), $r_n$ as difference of KLs (Eq. 11). §3.3 formal: Condition 1 distinguishability (Eqs. 12–14); Theorem 1 (optimality, Eq. 15); §3.3.2 non-distinguishable — Theorem 2 (continuity, $O(1/k)$, Eqs. 16–17), Theorem 3 (random test length, Eq. 18).
- **§4 Simulations.** GINC construction; main result (accuracy ↑ with $n,k$); latent-structure ablations; unseen-concept extrapolation failure; model-size/architecture effects (scale helps at fixed loss; LSTMs > Transformers on GINC); order sensitivity; zero-shot>few-shot.
- **§5 Discussion / related work.** BvM limitations; topic models & HMMs; bridging prompt/pretraining mismatch via finetuning/calibration; meta-learning contrast (meta-learners *trained* to learn vs. ICL *emergent*); studying large-scale phenomena at small scale.
- **§6 Conclusion.** ICL cast as implicit Bayesian inference; occurs for mixture-of-HMMs pretraining.
- **Appendices A–F.** A framework details (Eqs. 19–24). B Propositions 1–2 (delimiter/example probability lower bounds). C Lemma 1 convergence (matrix-form TV bound, Eqs. 31–47). D full Theorem 1 proof (Eqs. 48–64). E/F non-distinguishable proofs, GINC details, training setup.

---

## 5. Müller et al. 2022 — Transformers Can Do Bayesian Inference (PFNs)

**PDF:** `docs/project/references/in_context_learning/sources/Muller et al. 2022 - Transformers Can Do Bayesian Inference (PFNs).pdf` (ICLR 2022, arXiv:2112.10510v7)

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction / core problem.** Bayesian methods are attractive because you can encode prior knowledge explicitly and get calibrated uncertainty, but computing the *posterior predictive distribution* (PPD) — the probability of a new label $y$ given a query $x$ and a dataset $D$ — is intractable for most interesting priors. Traditional workarounds (MCMC, variational inference) are slow and require access to non-normalized posterior densities. This paper asks: can we *train a neural network once* so that, at inference, it approximates Bayesian posterior prediction for a whole family of tasks in a **single forward pass**?

**Key idea in plain terms — Prior-Data Fitted Networks (PFNs).** The only thing you must be able to do is *sample* from your prior over tasks. The recipe:
1. Repeatedly **sample a task/function from the prior**, then **sample a small dataset** from that task.
2. **Hide one label** and train a Transformer to predict it from the rest of the (set-valued) dataset, using ordinary cross-entropy / negative-log-likelihood.
3. After training, feed the network your *real* dataset plus a test point; its output is an approximation to the true Bayesian PPD.

In other words, PFNs recast "approximate the posterior" as a plain *supervised classification problem with a set-valued input*. They call the training phase "fitting prior-data" and the forward pass "(Bayesian) inference." This is exactly in-context learning: the dataset $D$ is the prompt, the query is the test input.

**Key findings.**
- PFNs **near-perfectly mimic Gaussian Processes** (where the true PPD is available for comparison) and handle intractable cases (GPs with hyper-priors, Bayesian neural networks) with **200×–8000× speedups** over MCMC/VI.
- On small tabular datasets, a PFN with a Bayesian-neural-network prior **beats XGBoost, CatBoost, and standard BNNs** in ROC AUC and especially in calibration (Expected Calibration Error 0.025 vs 0.066–0.157), while being thousands of times faster (13 s vs 20 h for all 20 datasets).
- A tiny hand-written "random lines" prior enables competitive **5-shot 5-way Omniglot** few-shot image classification.
- They introduce a discretized regression head (the **Riemann distribution**) so a classification-style network can represent continuous PPDs.

**Initial takeaway.** PFNs turn Bayesian inference into a one-time offline training cost, then amortize it into a fast forward pass. The theoretical payoff (below) is clean: minimizing the training loss *is* minimizing the KL divergence to the true PPD, so a perfectly-trained PFN recovers exact Bayesian posterior prediction. This is the "prior-fitting objective" that grounds the whole ICL-as-inference view: ICL isn't *approximating* something Bayesian by accident — a PFN is *explicitly optimized* to be Bayesian.

### Phase 2: Graduate-Level Deep Dive

#### 2.1 The Bayesian target: the posterior-predictive distribution

For a supervised problem with latent task $t\sim p(t)$ and dataset $D=\{(x_i,y_i)\}_{i=1}^n$, Bayes gives the posterior $p(t\mid D)$, and the **PPD** for a new query is

$$p(y\mid x, D) = \int_t p(y\mid x, t)\, p(t\mid D)\, dt. \tag{1}$$

Closed form only for special priors (e.g. GPs). Classically one *first* approximates $p(t\mid D)$ (via MCMC or VI) and *then* integrates. PFNs skip instantiating the posterior entirely and model the PPD directly.

#### 2.2 The prior-fitting objective (the mechanistic core)

Let $q_\theta(y\mid x, D)$ be a model (a permutation-invariant Transformer) that takes a set-valued dataset $D$ plus a query $x$ and returns a distribution over $y$. Train it to minimize the **Prior-Data Negative Log-Likelihood**:

$$\ell_\theta = \mathbb E_{D\cup\{(x,y)\}\sim p(D)}\big[-\log q_\theta(y\mid x, D)\big], \tag{2}$$

i.e., draw a dataset of size $|D|+1$ from the prior, hold out one point $(x,y)$, and penalize the model's surprise at $y$. **Algorithm 1** just repeats: sample $D\cup\{(x_i,y_i)\}_{i=1}^m\sim p(D)$, compute $\bar\ell_\theta = \sum_{i=1}^m(-\log q_\theta(y_i\mid x_i, D))$, SGD step.

**Insight 1 (objective = expected cross-entropy to the PPD).** Expanding the expectation in $(2)$ and factoring $p(x,y,D)=p(x,D)\,p(y\mid x,D)$:

$$\ell_\theta = -\int_{D,x,y} p(x,y,D)\log q_\theta(y\mid x,D) = -\int_{D,x} p(x,D)\int_y p(y\mid x,D)\log q_\theta(y\mid x,D)$$
$$= \int_{D,x} p(x,D)\, H\big(p(\cdot\mid x,D),\, q_\theta(\cdot\mid x,D)\big) = \mathbb E_{x,D\sim p(D)}\big[H\big(p(\cdot\mid x,D),\, q_\theta(\cdot\mid x,D)\big)\big]. \tag{3–4}$$

So the loss is the **expected cross-entropy between the true PPD $p(\cdot\mid x,D)$ and the approximation $q_\theta$**, averaged over prior-sampled $(x,D)$. The inner integral over $y$ is exactly the definition of cross-entropy $H(p,q)=-\int_y p\log q$.

**Corollary 1.1 (objective = expected KL, up to a constant).** Since cross-entropy = KL + entropy,

$$\mathbb E_{x,D}\big[\mathrm{KL}\big(p(\cdot\mid x,D),\, q_\theta(\cdot\mid x,D)\big)\big] = -\mathbb E_{x,D}\!\int_y p(y\mid x,D)\log\frac{q_\theta(y\mid x,D)}{p(y\mid x,D)}$$
$$= \mathbb E_{x,D}\big[H(p, q_\theta)\big] - \mathbb E_{x,D}\big[H(p)\big] = \ell_\theta + C, \tag{5–9}$$

where $C=-\mathbb E_{x,D}[H(p(\cdot\mid x,D))]$ does **not depend on $\theta$**. Therefore *minimizing the prior-data NLL is minimizing the KL divergence to the exact PPD.* This makes $\ell_\theta$ itself a valid yardstick for how close any method comes to true Bayesian inference — used throughout the experiments. (The 2024 note credits Foong et al. 2020 §3.2 Prop. 1 with the first statement of this result.)

**Corollary 1.2 (exactness at the optimum).** If $p$ is realizable within the family $q_\theta$ (i.e. some $\theta$ gives $q_\theta=p$), then the optimum $\theta^\* \in \arg\min_\theta \ell_\theta$ satisfies

$$q_{\theta^\*}(\cdot\mid x,D) = p(\cdot\mid x,D) \quad\text{for all } x,D \text{ with } p(x,D)>0.$$

*Proof.* Cross-entropy is minimized uniquely when the two distributions are equal; the minimizer of the expected cross-entropy therefore matches the PPD pointwise wherever the prior places mass. $\square$

This is the precise sense in which **a well-trained PFN performs Bayesian inference**: not approximately by analogy, but as the exact minimizer of an objective whose global optimum *is* the posterior predictive.

**Contrast with MCMC/VI.** PFNs need only prior *samples* from $p(D)$. MCMC needs an unnormalized posterior $f(t\mid D,x)\propto p(t\mid D,x)$; VI needs density values of $p(t,D)$. Both are often hard; prior sampling usually is not.

#### 2.3 Architecture: permutation-invariant Transformer + Riemann head

- **Encoder without positional encodings** → invariant to dataset permutation (a dataset is a *set*, not a sequence). Inputs and queries are linearly projected into the embedding space; the model outputs a PPD per query, attending to the $n$ input pairs (see Fig. 2a: queries attend to the dataset but not to each other). $N=n+m$ split randomly into $n$ context and $m$ query points each step.
- **Riemann distribution** (regression head). Neural nets classify better than they regress continuous densities, so the target range is discretized into buckets $B$ chosen to be **equal-mass under prior-data**: $p(y\in b)=1/|B|$, estimated from a large prior sample. For unbounded support, the outer buckets are replaced by scaled half-normals. **Theorem 1**: a finite-support Riemann distribution can approximate any almost-everywhere-continuous (Riemann-integrable), full-support density to arbitrary precision in KL — $\forall p\,\forall\epsilon\,\exists B.\ \mathrm{KL}(q_B, p)\le\epsilon$. The proof bounds the log-density between per-bucket Darboux upper/lower sums $\hat u,\hat l$ and drives their integrated gap below $\log(1/P)$.

#### 2.4 Experiments

- **GP with fixed hyperparameters (tractable PPD).** PFN's PPD is visually indistinguishable from the exact GP posterior mean/CI; approximation improves monotonically with more meta-train datasets (500K→4M) and generalizes to 2000-example datasets.
- **GP with hyper-priors (intractable).** PFN attains lower Prior-Data NLL than MLE-II and NUTS, while being 200× faster than MLE-II and 1000–8000× faster than NUTS.
- **BNNs.** PFN matches or beats Bayes-by-Backprop SVI (1000× faster) and NUTS (10 000× faster).
- **Tabular (20 OpenML datasets, 30 train samples each).** PFN-BNN wins on mean rank ROC AUC (2.786) and ECE (0.025), Wilcoxon $p\approx3.6\text{e-}13$ vs baselines; 13 s GPU total. Notably the PFN integrates *architecture* choice into the Bayesian inference — a **prior over architectures** $A\sim p(A)$, weights $W_{i,j}\sim p_w$, i.i.d. features, forward pass to produce $(x_i, A_W(x_i))$ — yielding a posterior over architectures, not a point estimate.
- **Omniglot 5-shot 5-way.** A 55-line random-lines prior + fine-tuning gives 0.865 accuracy, on par with PACOH-NN (0.885).

### Appendix: Section-by-Section Backbone

- **Abstract / §1 Introduction.** PFNs approximate a large set of posteriors via ICL; only requirement is sampling from a prior over supervised tasks. Recast posterior approximation as supervised classification with set-valued input. 200×+ speedups; results on GP regression, BNNs, tabular, few-shot image.
- **§2 Background.** Transfer-learning framing; meta-learning / learning-to-learn; (Conditional) Neural Processes; the PPD (Eq. 1); MCMC vs VI; amortized simulation-based inference / neural posterior estimation. PFNs differ by modeling the PPD directly from prior samples without instantiating the posterior.
- **§3 PPD approximation with PFNs.** Model $q_\theta(y\mid x,D)$; Algorithm 1 (fit prior-data); Prior-Data NLL $\ell_\theta$ (Eq. 2); **Insight 1** (= expected cross-entropy, Eqs. 3–4); **Corollary 1.1** (= expected KL + const); **Corollary 1.2** (exact at optimum if realizable). Contrast to VI/MCMC requirements.
- **§4 Adapting the Transformer.** Efficient permutation-invariant encoder (no positional encodings); Riemann distribution regression head (equal-mass buckets, half-normal tails); Theorem 1 (universal density approximation in KL).
- **§5 Posterior approximation studies.** §5.1 GP fixed hyperparams (near-exact); §5.2 GP hyper-priors (beats MLE-II & NUTS, 200–8000× faster); §5.3 BNNs (beats SVI/NUTS by 1000–10000×).
- **§6 Application to tabular datasets.** GP prior and BNN-prior-over-architectures for classification; 20 OpenML datasets; PFN-BNN strongest, best calibrated, fastest (Table 1).
- **§7 Few-shot learning.** Random-lines prior for Omniglot; 5-shot 5-way; on par with SOTA after fine-tuning (Table 2).
- **§8 Conclusion & future work.** Novel priors; better architectures; scaling; amortized simulation-based inference.
- **Appendices A–H.** A proof of Corollary 1.1 (Eqs. 5–9); B proof of Corollary 1.2; C proof of Theorem 1 (Riemann density approximation via Darboux sums, Eqs. 10–18); D–H architecture, Riemann details, ablations (permutation invariance), experimental setups, priors-over-architectures details.

---

## 6. Garg et al. 2022 — What Can Transformers Learn In-Context? A Case Study of Simple Function Classes

**PDF:** `docs/project/references/in_context_learning/sources/Garg et al. 2022 - What Can Transformers Learn In-Context.pdf` (NeurIPS 2022)

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction / core problem.** GPT-style models can in-context learn, but with real language it is impossible to disentangle *genuine on-the-fly learning* from *retrieving a task the model already memorized during pretraining*. Garg et al. sidestep language entirely and pose a clean, controllable question: **can a Transformer be trained from scratch to in-context learn a well-defined function class** — say, linear functions — so that, given a prompt of input-output pairs $(x_1, f(x_1), \dots, x_k, f(x_k), x_{\text{query}})$ from an *unseen* function $f$, it outputs $f(x_{\text{query}})$?

**Key idea in plain terms.** A "function class" is a family like "all linear functions in 20 dimensions." Training: repeatedly sample a random function $f$ from the class and random inputs, form a prompt, and train the Transformer to predict each output from the preceding examples (squared loss). Testing: hand it prompts from *fresh* functions it never saw and measure how close its prediction is to the truth versus known learning algorithms (least squares, Lasso, etc.). If it matches the best algorithm, the Transformer has effectively *learned a learning algorithm* and runs it in a single forward pass.

**Key findings.**
- A standard 12-layer GPT-2-style Transformer (22.4M params) trained from scratch **in-context learns linear functions as well as the optimal least-squares estimator** (error 0.02 at $k=d=20$ examples, 0.0006 at $2d$).
- The ability is **robust to distribution shift**: skewed input covariance, label noise (even reproducing the least-squares *double-descent* curve), and in-context/query inputs in different orthants — so it is not doing nearest-neighbor lookup, it genuinely regresses.
- It scales to **harder classes**: sparse linear functions (matches Lasso, beating least squares by exploiting sparsity), depth-4 decision trees (beats greedy tree learning and XGBoost), and 2-layer ReLU networks (matches a network trained by gradient descent). A model trained on 2-layer nets also in-context learns linear functions for free.
- **Model capacity and curriculum learning** matter: more parameters → lower error and higher usable dimension; a curriculum (start low-dimensional, grow) gives drastic training speed-ups. Non-trivial ICL needs relatively little data (~1k distinct functions or 100k prompts).

**Initial takeaway.** Garg et al. provide the clean *empirical* counterpart to Xie's theory: they *build* Transformers that provably in-context learn function classes, and show the learned computation matches sophisticated iterative algorithms (Lasso, gradient descent) despite running in one forward pass. This is the paper that turned "does the Transformer learn an algorithm?" into a tractable, measurable question — and set up von Oswald's next question, "*which* algorithm?"

### Phase 2: Graduate-Level Deep Dive

#### 2.1 Formalizing in-context learning of a function class

Let $D_{\mathcal X}$ be a distribution over inputs and $D_{\mathcal F}$ a distribution over functions in class $\mathcal F$. A **prompt** is $P = (x_1, f(x_1), \dots, x_k, f(x_k), x_{\text{query}})$ with $x_i, x_{\text{query}}\stackrel{\text{iid}}{\sim} D_{\mathcal X}$ and $f\sim D_{\mathcal F}$. A model $M$ in-context learns $\mathcal F$ up to $\epsilon$ (w.r.t. $D_{\mathcal F}, D_{\mathcal X}$) if

$$\mathbb E_P\big[\ell\big(M(P), f(x_{\text{query}})\big)\big] \le \epsilon, \tag{1}$$

for a loss $\ell$ (squared error here). Crucially, *being able to in-context learn is a property of $M$ alone*, independent of how $M$ was produced — training $M$ is an instance of meta-learning.

#### 2.2 Training objective

Sample $f\sim D_{\mathcal F}$ and inputs $x_1,\dots,x_{k+1}\sim D_{\mathcal X}$; build prompt prefixes $P^i = (x_1, f(x_1),\dots, x_i, f(x_i), x_{i+1})$. Train $M_\theta$ to minimize the average loss over all prefixes:

$$\min_\theta\ \mathbb E_P\Big[\frac{1}{k+1}\sum_{i=0}^{k}\ell\big(M_\theta(P^i),\, f(x_{i+1})\big)\Big]. \tag{2}$$

**Implementation.** Decoder-only GPT-2-family Transformer, 12 layers, 8 heads, 256-dim embeddings (22.4M params). Prompt outputs $f(x_i)$ are zero-padded to input dimension, both mapped to embedding space by learnable linear maps; a learnable linear read-out maps the predicted vector to a scalar. The architecture computes predictions for *all* prefixes $M_\theta(P^i)$ in one forward pass. Trained from scratch (no pretrained LM, no text), batch 64, 500k steps, squared-error loss.

**Curriculum learning.** Start with inputs confined to a 5-dim subspace and prompt length 11; every 2000 steps increase subspace dim by 1 and prompt length by 2, up to ambient $d$ and length $2d{+}1$. This is the decisive trick for high $d$: at $d=50$, loss barely moves in 500k steps *without* curriculum but nearly reaches optimum *with* it.

#### 2.3 Linear functions: matching the Bayes-optimal estimator

$\mathcal F = \{f(x)=w^\top x\}$ in $d=20$; $x_i, x_{\text{query}}, w\sim\mathcal N(0, I_d)$. Baselines: least squares (minimum-norm fit — the optimal estimator, a lower bound), 3-nearest-neighbors, and simple averaging $\hat w = \frac1k\sum y_i x_i$. **Result (Fig. 2):** the Transformer's normalized squared error $\big(M(P)-w^\top x_{\text{query}}\big)^2/d$ tracks least squares for every $k$; at $k=d=20$, least squares hits 0 while the Transformer reaches 0.02, improving to 0.0006 at $2d$. Simple baselines are far worse, so the model encodes a *non-trivial* algorithm — and an ablation (Appendix B.7) rules out memorization.

**What function is it computing?** Fix the $k$ in-context examples and view the output as $\hat f_{w, x_{1:k}}(x_{\text{query}})$. When $k<d$ the ground truth is unrecoverable and the ideal prediction is the projection $\big(\mathrm{proj}_{x_{1:k}}(w)\big)^\top x_{\text{query}}$. Two probes:
- **Function along a random direction:** $\hat f_{w,x_{1:d}}(\lambda x)$ and $\hat f_{w,x_{1:2d}}(\lambda x)$ match ground truth; $\hat f_{w,x_{1:d/2}}(\lambda x)$ matches the *projected* ground truth (for query norm near a typical training input).
- **Local correctness via gradient:** the model is differentiable, so compute $\nabla_{x_{\text{query}}}\hat f$. It should point along $\mathrm{proj}_{x_{1:k}}(w)$. Fig. 3b shows the normalized inner product of gradient with projected $w$ is ~1 for all $k$, and with $w$ itself once $k\ge d$ (since $\mathrm{proj}_{x_{1:k}}(w)=w$ a.s. then). The model is *locally* the correct linear map.

#### 2.4 Out-of-distribution robustness (evidence of a real algorithm)

Train on $D^{\text{train}}$, evaluate on shifted $D^{\text{test}}$ (and a separate query distribution $D^{\text{test}}_{\text{query}}$):
- **Skewed covariance:** inputs from $\mathcal N(0,\Sigma)$, eigenvalues $\propto 1/i^2$, norm-matched. Tracks least squares up to $k=10$ then plateaus — imperfect but still $<$ half the NN-baseline error.
- **Noisy linear regression:** $y_i = w^\top x_i + \epsilon_i$, $\epsilon_i\sim\mathcal N(0,1)$. Tracks least squares and even reproduces the **double-descent** error spike near $k=d$ — a signature of the least-squares estimator, not of nearest-neighbor.
- **Different orthants:** all in-context $x_i$ forced into one (random) orthant, query in another. Essentially unaffected (errors 0.062 at $k=20$, 0.004 at $k=40$), which a nearest-neighbor method could not achieve — confirming it is *not* doing similarity search.

#### 2.5 More complex classes

- **Sparse linear ($d=20, s=3$):** matches **Lasso** (which has no closed form and needs iterative $\ell_1$-regularized minimization), beating minimum-norm least squares — the Transformer *exploits sparsity* in one forward pass.
- **Depth-4 decision trees (16 leaves):** beats greedy tree learning and XGBoost; even when the baselines are handed the sign pattern of inputs, the Transformer still wins (0.12 vs 0.31 error at 100 examples).
- **2-layer ReLU nets (100 hidden units):** matches a same-architecture net trained by Adam on the in-context examples; and the model trained on 2-layer nets *also* in-context learns linear functions (unprompted transfer).

#### 2.6 What matters: capacity, curriculum, data

- **Capacity & dimension (Fig. 6):** across 3.4M / 7.6M / 22.4M params and $d\in\{10,\dots,50\}$, error falls with capacity and rises with $d$; capacity especially helps OOD (skewed covariance, orthant shift).
- **Curriculum:** essential at high $d$; without it, a long loss plateau precedes a sharp drop — reminiscent of the sudden emergence of **induction heads** (Olsson et al.) in language models.
- **Data efficiency:** non-trivial ICL with $n_p=100$k distinct prompts or $n_w=1$k distinct functions; near-unrestricted performance at $n_p=1$M / $n_w=10$k. (The main model sees 32M distinct functions over training.)

**Positioning.** Garg et al. explicitly cite Xie (Bayesian-inference framing) and Müller (PFNs) as the closest prior work, and frame their own contribution as posing ICL as *learning a function class at inference time* and empirically probing the encoded algorithm and its extrapolation.

### Appendix: Section-by-Section Backbone

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

## 7. von Oswald et al. 2023 — Transformers Learn In-Context by Gradient Descent

**PDF:** `docs/project/references/in_context_learning/sources/von Oswald et al. 2023 - Transformers Learn In-Context by Gradient Descent.pdf` (ICML 2023, PMLR 202)

### Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction / core problem.** Garg showed Transformers *can* in-context learn like optimal algorithms; von Oswald asks the sharper follow-up: **what algorithm, mechanistically, is the forward pass running?** Their answer, for regression tasks: the Transformer is performing **gradient descent** on a small regression problem built from the in-context examples — inside its own forward pass, without any weight updates. They call this a "mesa-optimizer": an optimizer that emerges *inside* a learned network.

**Key idea in plain terms.** Normally we think of two learning loops. The *outer loop* is ordinary training: gradient descent adjusts the Transformer's weights over many tasks. The claim is that this outer training installs, inside the frozen forward pass, an *inner loop* that itself does gradient descent — one attention layer ≈ one gradient-descent step on the in-context data. So when you feed the model a prompt, it silently: (1) treats the examples as a training set, (2) takes a gradient step (or several, for a deep model) on a regression loss, and (3) reads out the resulting prediction. They prove this is *possible* with an explicit hand-built weight construction, then show that when you actually *train* such Transformers, the learned weights **match the construction**.

**Key findings.**
- **Explicit construction (Proposition 1):** a *single* linear self-attention (LSA) layer, with specific key/query/value/projection matrices, produces a token update *identical* to one gradient-descent step on a mean-squared-error regression loss.
- **Trained ≈ constructed:** when you optimize a one-layer LSA Transformer on linear regression, the learned weights (up to a scaling redundancy) coincide with the construction — verified by cosine similarity ≈1, matching predictions, and successful weight-space interpolation between trained and constructed models.
- **Depth = more steps:** $K$ stacked layers ≈ $K$ gradient steps. Trained deep Transformers actually *beat* plain gradient descent by learning an accelerated variant the authors call **GD++** (a curvature/data-preconditioning correction), and they realign perfectly with GD++.
- **Nonlinear tasks:** adding an MLP before the attention layer = gradient descent on *deep representations* (kernel regression with kernel $k(x,y)=m(x)^\top m(y)$); demonstrated on sine-wave regression.
- **Building the dataset (induction-head link):** a preceding (softmax) attention layer can *copy* neighboring tokens to assemble the $(x,y)$ token format the GD step needs — connecting the mechanism to the "induction heads" of Olsson et al.

**Initial takeaway.** This is the mechanistic capstone of the lineage: it does not just say "Transformers behave like a good learning algorithm" (Garg) or "the optimum is Bayesian" (Xie/PFN) — it identifies the *concrete computation* (gradient descent), gives a closed-form weight construction realizing it, and shows optimization *finds* that construction. It reframes in-context learning as *emergent gradient-based meta-learning in the forward pass*.

### Phase 2: Graduate-Level Deep Dive

#### 2.1 Setup: linear self-attention

A standard multi-head self-attention layer updates token $e_j$ by

$$e_j \leftarrow e_j + \mathrm{SA}_\theta(j,\{e_1,\dots,e_N\}) = e_j + \sum_h P_h V_h\,\mathrm{softmax}(K_h^\top q_{h,j}), \tag{1}$$

with value/key columns $v_{h,i}=W_{h,V}e_i$, $k_{h,i}=W_{h,K}e_i$ and query $q_{h,j}=W_{h,Q}e_j$; $P_h$ is the output projection. The paper's *only* architectural departure (following Schlag et al. 2021) is to **drop the softmax**, giving the **linear self-attention (LSA)** layer:

$$e_j \leftarrow e_j + \mathrm{LSA}_\theta(j,\{e_1,\dots,e_N\}) = e_j + \sum_h P_h V_h K_h^\top q_{h,j}.$$

#### 2.2 Gradient descent as a data transformation

Reference linear model $y(x)=Wx$, dataset $D=\{(x_i,y_i)\}_{i=1}^N$, squared-error loss

$$L(W) = \frac{1}{2N}\sum_{i=1}^N \|Wx_i - y_i\|^2. \tag{2}$$

One GD step with learning rate $\eta$:

$$\Delta W = -\eta\nabla_W L(W) = -\frac{\eta}{N}\sum_{i=1}^N (Wx_i - y_i)x_i^\top. \tag{3}$$

The pivotal re-interpretation: instead of updating *weights*, push the change onto the *data/targets*. After the step,

$$L(W+\Delta W) = \frac{1}{2N}\sum_{i=1}^N \|(W+\Delta W)x_i - y_i\|^2 = \frac{1}{2N}\sum_{i=1}^N \|Wx_i - (y_i - \Delta y_i)\|^2, \tag{4}$$

where $\Delta y_i = \Delta W x_i$. So **a GD step is equivalent to keeping $W$ fixed and updating each target $y_i \mapsto y_i - \Delta y_i$.** This "learning by transforming data, not weights" is exactly the form an attention layer can produce.

#### 2.3 The weight construction (Proposition 1 — the mechanistic core)

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

#### 2.4 Trained one-layer LSA ≈ one GD step (empirical, §3)

Train a single LSA layer to minimize the prediction error on the query token,

$$L(\theta) = \frac{1}{B}\sum_{\tau=1}^B \big\|\hat y_\theta(\{e_{\tau,i}\}_{i=1}^N, e_{\tau,N+1}) - y_{\tau,\text{test}}\big\|^2, \tag{5}$$

with fresh tasks each step (teacher $W_\tau\sim\mathcal N(0,I)$, $x_{\tau,i}\sim U(-1,1)^{n_I}$, $y_{\tau,i}=W_\tau x_{\tau,i}$; $N=n_I=10$, $n_O=1$). Because the LSA prediction is linear in $x_{\text{test}}$, $\hat y_\theta(x_{\text{test}}) = \Delta W_{\theta,D}\, x_{\text{test}}$. Compare against the control $\hat y_{\theta_{\text{GD}}}$ (one GD step from $W_0=0$, learning rate tuned by line search). Over $10^4$ validation tasks they measure (1) prediction L2 difference, (2) cosine similarity of the model *sensitivities* $\partial\hat y/\partial x_{\text{test}}$, (3) their L2 difference. **Result (Fig. 2):** near-perfect agreement — cosine ≈ 1, and a simple *scaling correction* recovers the construction so that trained TF, GD, and the interpolated weights $\theta_I=(\theta+\theta_{\text{GD}})/2$ have identical loss. OOD tests (inputs $U(-\alpha,\alpha)$, teacher scaling $\alpha W$) show the trained TF and GD degrade *together*. Repeating the LSA update (with a dampening $\lambda=0.75$) matches iterated GD (Fig. 1).

#### 2.5 Depth = iterated GD, and GD++ (§3 continued)

Stack $K$ LSA layers → read out $-y_{N+1}+\sum_{k=1}^K \Delta y_{k,N+1} = y_{N+1} + \sum_{k=1}^K \Delta W_k x_{N+1}$, i.e. $K$ GD steps. Trained deep Transformers *outperform* plain $K$-step GD; the authors show they instead match **GD++**, a variant with an input **data transformation**

$$x_j \leftarrow H(X)\, x_j, \qquad H(X) = (I - \gamma X X^\top),$$

a curvature/preconditioning correction (one tuned $\gamma$; for a recurrent 2-layer LSA they realign perfectly with GD++; for a 5-layer untied model, per-layer $\eta,\gamma$ give near-zero loss and almost perfect alignment). More generally an LSA layer implements $x_j\leftarrow (I + P(X)V(X)K(X)^\top W_Q)x_j = H_\theta(X)x_j$, a *task-specific data transformation* encodable in $\theta$. Softmax SA + LayerNorm (i.e. the standard block) gives qualitatively the same alignment, though weaker than pure LSA.

#### 2.6 Nonlinear regression via MLP (§3, Proposition 2)

**Proposition 2.** Given a Transformer block = an MLP $m(e)$ transforming tokens $e_j=(x_j,y_j)$ followed by an attention layer, one can construct weights implementing gradient descent on

$$\frac{1}{2N}\sum_{i=1}^N \|W m(x_i) - y_i\|^2.$$

Iterating such blocks solves **kernelized least-squares regression** with kernel $k(x,y)=m(x)^\top m(y)$ induced by the MLP. Unlike standard meta-learning (which back-props a *task-shared* embedding), the MLP here processes both inputs and targets, and its output is fed to one LSA layer implementing the GD step. Demonstrated on sine-wave regression (Fig. 4): trained TF and a meta-learned MLP + one GD output-layer step match on both initial and post-step predictions.

#### 2.7 Building the dataset: copying and the induction-head link (§4, Proposition 3)

Proposition 1 assumed each token already concatenates $(x_j, y_j)$. Real prompts alternate $e_{2j}=(x_j)$, $e_{2j+1}=(0, y_j)$. **Proposition 3:** a 1-head linear or softmax attention layer (with positional encodings) can *transform* the alternating tokens into the Proposition-1 format. Empirically (Fig. 5), a 2-layer SA circuit trained on alternating tokens matches *one* GD step (not two — the first layer is spent on copying), and before the loss drops the first layer's output becomes highly sensitive to the *neighboring* token $e_{j+1}$ while staying independent of others — direct evidence of a **copying mechanism** merging input and target into one token. This required a softmax first layer (linear failed), consistent with Olsson et al.'s finding that softmax attention easily learns to copy. The authors conclude **copying (induction-head-like) is the second crucial mechanism**: copy to assemble $(x,y)$ tokens, then run GD in later layers.

#### 2.8 Framing: two-timescale meta-learning / mesa-optimization

Learning Transformer weights (outer loop / slow) installs a forward-pass learning algorithm (inner loop / fast). This builds on **fast weights** (Schmidhuber 1992), shown equivalent to linear self-attention (Schlag et al. 2021), and on MAML-style "adapt only the output layer on a deep representation." The emergent-optimizer-inside-a-network phenomenon is **mesa-optimization** (Hubinger et al. 2019). The authors caution that linear-SA-only Transformers explain only a limited slice of the full ICL story; softmax, depth, and domain introduce phase transitions and complexity beyond this construction. Parallel works: Akyürek et al. 2023 (a multi-layer construction implementing one GD step *with weight decay*) and Garg et al. 2022 (performance match). This paper's distinctive claim is that **optimization actually finds the construction** (Proposition 1 verified in trained weights), and that it explains the shallow two-layer circuits studied by Olsson et al.

### Appendix: Section-by-Section Backbone

- **Abstract / §1 Introduction.** Hypothesis: training Transformers ≈ gradient-based meta-learning. Single-LSA-layer construction = one GD step; trained TFs match it → mesa-optimizers doing GD in the forward pass; GD++ curvature correction; deep representations for nonlinear tasks; induction-head parallel. Meta-learning framing (fast/slow weights, fast weights ≡ linear attention (Schlag), MAML). **Hypothesis 1** stated. Contrast with Akyürek et al. (multi-layer construction) — here a simpler single-layer one, verified to be found by optimization.
- **§2 Linear self-attention can emulate gradient descent.** SA layer (Eq. 1) → LSA (drop softmax). GD as data transformation: loss (Eq. 2), step (Eq. 3), target-update reinterpretation (Eq. 4). Token encoding $e_j=(x_j,y_j)$, query $e_{N+1}$. **Proposition 1** + five practical notes (full/masked attention & $W_0\approx0$; read-out via $-1$ multiply; uniqueness/scaling; meta-learned $\eta$; task-specific $H_\theta(X)$).
- **§3 Trained Transformers do mimic GD.** Token construction; training objective (Eq. 5); teacher-generated data. One-step: trained LSA vs constructed $\theta_{\text{GD}}$ — prediction/sensitivity metrics, interpolation, OOD, repeated update (Fig. 2). Multi-step: $K$ layers = $K$ steps; TF beats GD, matches **GD++** ($H(X)=I-\gamma XX^\top$); recurrent 2-layer and 5-layer untied (Fig. 3); softmax+LayerNorm variant. Nonlinear: **Proposition 2** (MLP + attention = GD on deep reps = kernel regression), sine-wave (Fig. 4).
- **§4 Do self-attention layers build regression tasks?** **Proposition 3** (copy alternating $(x)$ / $(0,y)$ tokens into Prop-1 format). 2-layer circuit matches one GD step; first layer becomes sensitive to neighbor token (Fig. 5) → copying mechanism; needs softmax; induction-head connection.
- **§5 Discussion.** Recap of GD hypothesis; outer-loop = meta-learning; builds on Schlag et al. delta-rule fast weights; extends to $K$-step GD and GD++; explains standard architectures; caveats on scope.
- **Appendices A.1–A.12.** A.1 Proposition 1 construction + full derivation (Eqs. 6–7 and block-matrix expansion). A.5 Proposition 3 copying construction. A.6 dampening $\lambda$. A.8 kernel regression / kernel smoothing discussion. A.9 softmax + LayerNorm results. A.10 GD++ curvature-correction details. A.11 phase transitions (seed-dependent). A.12 experimental details incl. fixed-dataset-size cycling (Fig. 16).

---

## 8. Reuter et al. 2025 — Can Transformers Learn Full Bayesian Inference In Context?

**PDF:** `docs/project/references/in_context_learning/sources/Reuter et al. 2025 - Can Transformers Learn Full Bayesian Inference in Context.pdf`
**Venue:** ICML 2025 (PMLR 267). Authors: Arik Reuter, Tim G. J. Rudner, Vincent Fortuin, David Rügamer (LMU Munich / NYU / TU Munich / Helmholtz AI / MCML).

### Phase 1 — Foundational Overview (undergraduate level)

**The problem in plain words.** Bayesian inference is the machine that, after seeing data, tells you not just a single best guess for a model's hidden parameters but a *whole probability distribution* over them — capturing how uncertain you should be, and where multiple explanations are plausible. The gold-standard way to compute this (MCMC, e.g. Hamiltonian Monte Carlo) is slow: you re-run an expensive sampler from scratch every single time new data arrives. Variational inference (VI) is faster but forces you to pick a rigid family of shapes for the answer (often a Gaussian), and it tends to collapse onto one mode and misjudge the spread.

Classic "prior-data fitted networks" (PFNs, and their tabular cousin TabPFN) already showed that a transformer can *learn to do Bayesian prediction* by training only on synthetic data drawn from a chosen prior — but they only produce a *univariate, usually discretized predictive distribution* for one output value. They do not hand you the full, high-dimensional posterior over the latent parameters.

**The core idea.** Reuter et al. ask: can we train a transformer that ingests an *entire small dataset* $x$ in its context and, in one forward pass, spits out *samples from the full posterior* $P(z\mid x)$ over the latent variables $z$ — matching what an expensive HMC run would give? They answer yes, by bolting a **generative sampler onto a PFN-style encoder**. Specifically:
- A large **TabPFN-style transformer encoder** reads the dataset $x$ and produces a representation.
- A **diffusion-transformer decoder** trained by **flow matching** (a continuous normalizing flow) turns that representation into posterior samples $z \sim P(z\mid x)$.
- Training data is *purely synthetic*: you sample $z$ from a prior, then $x$ from the likelihood, and teach the network to invert that map. No real posterior is ever needed at training time.

**Key findings.**
- Across generalized linear models (GLMs), factor analysis (FA), and Gaussian mixture models (GMMs), the ICL sampler's posterior samples are **as close to HMC's as, or closer than, several state-of-the-art VI methods** — on both synthetic and 17 real-world tabular datasets.
- The advantage is largest exactly where VI struggles: **skewed posteriors** (e.g. gamma priors) and **multimodal posteriors** (GMMs), where VI shows textbook "mode-seeking" collapse and even flow-based VI (IAF) fails to capture bimodality from only 50 data points.
- **Flow matching is essential** — ablations replacing it with a plain Gaussian parametrization or a diffusion objective degrade the fit badly.
- **Inference is fast** (single forward pass + a cheap ODE solve) but the **up-front training cost is high**, and quality degrades under out-of-distribution shift and in high latent dimension ($K = 20, 50$).

**Initial takeaway.** This paper pushes ICL from "predict the next value" to "sample the full parameter posterior." It is the shard's clearest demonstration that a transformer, trained only on simulator draws, can *replace* MCMC/VI for full Bayesian inference on realistic models — a **distribution estimator** in the purest sense, and the natural counterpoint to the Mittal "point vs distribution" debate below.

### Phase 2 — Graduate-Level Deep Dive

#### 2.1 The amortization objective and the tractability trick

The learning target is the map $f_0 : \mathcal{X} \to \mathcal{M}(\mathcal{Z})$, $x \mapsto P^{z\mid x}$, where $\mathcal{M}(\mathcal{Z})$ is the space of probability measures over the latent space. The model $f_\theta(x) = Q^{z\mid x}_\theta$ is trained to minimize the expected divergence

$$R_\theta \;=\; \mathbb{E}_{x\sim p(x)}\!\left[\, d\!\left(Q^{z\mid x}_\theta,\; P^{z\mid x}\right)\right]. \tag{1}$$

$R_\theta$ is intractable because $P^{z\mid x}$ is unknown. The paper's key enabling result rewrites it into an objective over the **joint** $P^{x,z}$, from which we *can* sample:

$$\widetilde{R}_\theta \;=\; \mathbb{E}_{x,z\sim p(x,z)}\!\left[\, \mathcal{L}_d(x,z,\theta)\right]. \tag{2}$$

**Proposition 1 (equivalence condition).** If the divergence has the linear-in-$P$ form
$d(Q^{z\mid x}_\theta, P^{z\mid x}) = \int \gamma(Q^{z\mid x}_\theta)\, dP^{z\mid x}$
for some measurable functional $\gamma:\mathcal{M}(\mathcal{Z})\to\mathbb{R}$, then $R_\theta = \widetilde{R}_\theta$ with $\mathcal{L}_d(x,z,\theta) = \gamma(Q^{z\mid x}_\theta)$.

*Derivation sketch.* Starting from (1) and substituting the integral form of $d$:
$$R_\theta = \mathbb{E}_{x}\!\left[\int \gamma(Q^{z\mid x}_\theta)\, dP^{z\mid x}\right] = \mathbb{E}_{x}\,\mathbb{E}_{z\mid x}\!\left[\gamma(Q^{z\mid x}_\theta)\right] = \mathbb{E}_{x,z\sim p(x,z)}\!\left[\gamma(Q^{z\mid x}_\theta)\right],$$
where the middle step is the definition of conditional expectation and the last is the **law of total expectation** — the tower property collapses $\mathbb{E}_x \mathbb{E}_{z\mid x}$ into a single expectation over the joint. This is exactly the same tractability trick PFNs use: it is why training on simulator draws $(x,z)\sim P^{x,z}$ is equivalent to matching the true posterior.

**Worked instance — forward KL recovers max-likelihood.** Choosing $d$ to be the forward KL $D_{KL}[p(\cdot\mid x)\,\|\,q_\theta(\cdot\mid x)]$ gives $\gamma$ such that
$$\mathcal{L}_{d_{KL}}(x,z,\theta) = -\log q_\theta(z\mid x) + \text{const}.$$
So minimizing $\widetilde{R}_\theta$ under forward KL is **maximum-likelihood training of $q_\theta$ on joint samples** — precisely the PFN/NPE (neural posterior estimation) objective. This is the anchor connecting Reuter to the forward-KL branch in both Mittal papers.

#### 2.2 Flow matching as the posterior parametrization

Rather than a tractable density $q_\theta(z\mid x)$, Reuter parametrizes $Q^{z\mid x}_\theta$ implicitly as a **continuous normalizing flow (CNF)**. A base $P_B = \mathcal{N}(0,I)$ is pushed forward by a conditional flow $\psi_\theta(\cdot\mid x)$:

$$Q^{z\mid x}_\theta := [\psi_\theta(\cdot\mid x)]_\sharp P_B, \qquad P^{z\mid x} \approx [\psi_\theta(\cdot\mid x)]_\sharp P_B. \tag{3}$$

The flow is defined by an ODE driven by a learned time-dependent vector field $v^\theta_{t,x}$:

$$\frac{d}{dt}\psi_{\theta,t}(z\mid x) = v^\theta_{t,x}\!\big(\psi_{\theta,t}(z\mid x)\big), \qquad \psi_{\theta,0}(z\mid x)=z, \quad 0\le t\le 1. \tag{4}$$

The initial condition makes the flow the identity at $t=0$; integrating to $t=1$ transports $P_B$ onto the posterior.

**Flow-matching loss.** Under Gaussian conditional probability paths with an optimal-transport mean/variance schedule, the discrepancy becomes

$$d_{CFM}\!\left(Q^{z\mid x}_\theta, P^{z\mid x}\right) = \mathbb{E}\!\left[\,\big\| v^\theta_{t,x}(\gamma_t(z^{(1)}\mid z^{(0)})) - (z^{(1)} - \omega z^{(0)})\big\|_2^2\,\right], \tag{5}$$

where the expectation is over $t\sim U([0,1])$, $z^{(0)}\sim P_B$, $z^{(1)}\sim P^{z\mid x}$, the interpolant is $\gamma_t(z^{(1)}\mid z^{(0)}) := (1-\omega t)z^{(0)} + t z^{(1)}$, and $\omega = 1-\sigma_{\min}$ (they set $\omega = 1-10^{-4}$). Because $d_{CFM}$ is an expectation of a per-sample loss, it satisfies Proposition 1's form, so the empirical training objective over $N$ i.i.d. joint draws is simply

$$\widehat{R}_\theta = \sum_{i=1}^N \big\| v^\theta_{t_i,x_i}(\gamma_{t_i}(z^{(1)}_i\mid z^{(0)}_i)) + z^{(1)}_i - \omega z^{(0)}_i\big\|_2^2. \tag{7}$$

At **sampling** time: draw $z^{(0)}\sim P_B$, encode $x$, then hand the decoder-defined field $(t,\nu)\mapsto v^\theta_{t,x}(\nu)$ to a numerical ODE solver integrating $0\to 1$.

*Why CNF over discrete normalizing flows or a Gaussian?* (i) CNFs place **no architectural constraint** (no invertibility/triangular-Jacobian requirement), so complex data conditioning via cross-attention is free; (ii) flow matching is **simulation-free** and more sample-efficient than diffusion; (iii) it can represent essentially arbitrary posterior shapes — the crucial property for the skewed/multimodal cases where VI fails.

#### 2.3 Architecture and the point-vs-distribution positioning

The architecture couples a **TabPFN-style encoder** (reads the dataset $x$) with a **diffusion-transformer decoder** where time $t$ enters via adaptive layer-norm (adaLN) blocks initialized to identity, plus cross-attention onto the encoder output. The decoder input $(1-\omega t)z^{(0)} + t z^{(1)}$ is treated as a length-one sequence; effectively an MLP-with-conditioning + cross-attention. A separate model is trained per model-scenario (7 GLM, 6 FA, 4 GMM).

**Where this sits on the shard's central axis.** Reuter is an unambiguous **full-distribution estimator** and shows it *wins* against VI. This is in productive tension with Mittal (Point or Distribution), which finds point estimators usually win *for downstream prediction* on higher-dimensional problems. The reconciliation: (a) Reuter evaluates *posterior-sample fidelity* against HMC (C2ST / MMD / $W_2$), not just predictive loss; (b) Reuter's latent dimensions where ICL wins are modest ($K$ small) and its own high-$K$ ablation ($K=20,50$) shows the advantage *evaporates* — consistent with Mittal's high-dimensional finding. The two papers are measuring different things (fidelity vs prediction) on different regimes.

#### 2.4 Evaluation and results

Three sample-based metrics vs the gold standard (analytic where available, else HMC-NUTS): **C2ST** (classifier 2-sample test ROC-AUC; 0.5 = indistinguishable, 1.0 = perfectly separable), **MMD** (maximum mean discrepancy), and **$W_2$** (empirical 2-Wasserstein). Baselines: Laplace approximation and ADVI-style VI with diagonal / full / structured Gaussian and IAF variational families.

Headline numbers (lower is better): on GLMs, ICL C2ST $0.657$ (synthetic) vs best VI $0.711$; on FA, ICL C2ST $0.568$ — remarkably near the $0.5$ floor — vs VI $\ge 0.987$; on the hardest GMMs, ICL $\approx 0.825$ C2ST vs VI $\approx 0.99$ (saturated). MMD and $W_2$ track the same ordering. Qualitatively, VI exhibits mode-seeking collapse on bimodal GMM posteriors while ICL matches HMC's marginals.

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** Motivates ICL for *full* (high-dimensional, continuous) posteriors vs PFNs' univariate predictive posteriors. Frames two pains of classic full-Bayes: slow sampling, and misspecification from rigid VI families. Five contributions: (1) a model sampling $P(z\mid x)$ with no parameter updates/parametric posterior assumptions; (2) training on synthetic joint draws $P^{x,z}$ + a general analysis framework; (3) instantiation for GLMs, GMMs, FA; (4) fidelity comparable/superior to HMC and better than VI; (5) ablations (diffusion vs flow matching, Gaussian parametrization, OOD, dimensionality).
- **§2 Related Work.** Three lenses: **ICL** (Garg et al. 2022 function classes; contrast: their work is scalar/small-scale/simulated-only, this is multivariate posteriors on real data, *unsupervised*); **Amortized inference** (VAEs/NPs amortize per-datapoint $q_\theta(z_j\mid h_\phi(x_j))$; this work amortizes *per-dataset*, a functional map over a meta-dataset $\mathcal{D}\subset(\mathcal{X}\times\mathcal{Z})^N$; no ELBO/KL used, unlike AVI); **Simulation-based inference** (SBI/NPE via normalizing flows, flow matching, transformer diffusion on the joint). Explicitly cites concurrent Mittal 2025a (amortized posterior benchmark) and 2025b (point vs distribution) — noting *they* evaluate posterior mean / predictive performance, whereas Reuter evaluates *full posteriors*.
- **§3 ICL for Full Bayesian Inference.** §3 (objective + Proposition 1, the joint-sampling tractability result). §3.1 Defining the posterior form (normalizing flows §3.1.1, CNFs + flow-matching loss §3.1.2, eqs. 3–7). §3.2 Sampling from the joint (ancestral: $z\sim P^z$ then $x\sim P^{x\mid z}$). §3.3 Architecture (TabPFN encoder + diffusion-transformer decoder, adaLN, cross-attention). §3.4 Implementing flow matching (train vs sample loops, ODE solve).
- **§4 Experiments.** Modeling scenarios (7 GLM, 4 FA, 4 GMM). 50 synthetic + 17 real-world tabular datasets (Grinsztajn suite). Metrics C2ST/MMD/$W_2$ vs analytic-or-HMC. §4.1 GLMs (ICL best, esp. skewed gamma-prior). §4.2 FA (ICL C2ST 0.568, near floor). §4.3 GMMs (hardest — discrete assignments, high-dim, non-identifiable/multimodal; ICL still best, VI mode-seeks). §4.4 Ablations: flow matching essential; dimensionality (advantage vanishes at $K=20,50$); OOD degrades with shift; predictive performance (VI competitive on point prediction, ICL competitive); MLP vs transformer encoder (transformer wins); C2ST classifier choice validated.
- **§5 Discussion.** Contributions recap. Limitations: heavy up-front training; focus on simple posteriors with tractable references; high-dim challenges both method and metrics; large contexts expensive. Outlook: any model with a conceivable generative process can be fit — potential beyond standard Bayesian methods.

---

## 9. Mittal et al. 2025 — Amortized In-Context Bayesian Posterior Estimation

**PDF:** `docs/project/references/in_context_learning/sources/Mittal et al. 2025 - Amortized In-Context Bayesian Posterior Estimation.pdf`
**Status:** arXiv preprint (arXiv:2502.06601v1, 10 Feb 2025). Authors: Sarthak Mittal, Niels Leif Bracher, Guillaume Lajoie, Priyank Jaini, Marcus Brubaker (Mila / RPI / Google DeepMind / York / Vector).

### Phase 1 — Foundational Overview (undergraduate level)

**The problem in plain words.** Suppose you have a fixed probabilistic model (a likelihood + prior) and you keep getting new datasets — new poll results, new case counts, a new geographic region. Standard Bayesian tools (MCMC, VI) force you to re-run an expensive fit *from scratch* for every new dataset. **Amortization** is the trick of training a single neural network once so that, for any new dataset $\mathcal{D}$ handed to it as context, it *instantly* outputs an approximate posterior $q_\phi(\theta\mid\mathcal{D})$ — no re-fitting. Because the datasets are unordered bags of i.i.d. points, the network should be **permutation-invariant** (order of context examples must not matter).

**The core question.** Prior in-context work mostly modeled the *predictive* distribution $p(y^*\mid x^*, \mathcal{D})$ directly. This paper instead targets the **posterior over parameters** $p(\theta\mid\mathcal{D})$, and runs a careful, controlled *bake-off* over the two things that matter:
1. **Training objective:** forward KL (= neural posterior estimation / SBI) vs reverse KL (= the VI / neural-process style).
2. **Architecture & density:** GRU vs DeepSets vs Transformer for the set-conditioner; diagonal Gaussian vs normalizing flow for the output density.

**Key findings.**
- **Reverse KL + Transformer + normalizing flow is the overall winner** for predictive problems, especially in high dimensions.
- **Forward KL suffers from mode-averaging** — on high-dimensional multimodal problems (nonlinear regression/classification) it smears mass across modes and predicts poorly. Reverse KL is mode-seeking, which turns out to help predictively at high dimension.
- **Reverse KL has a crucial practical superpower:** it can be trained on *any* dataset distribution, including *real* data, because it does not need ground-truth $(\theta,\mathcal{D})$ pairs — only the joint density. Forward KL *must* train on simulated data from the assumed model, so it breaks under **misspecification** and **sim-to-real transfer**. Reverse KL degrades gracefully and improves further when trained directly on real data.
- **Surprising architecture result:** GRUs (not even permutation-invariant) often beat DeepSets (which are). The likely reason: DeepSets' fixed pooling operator limits expressivity; GRUs can *learn* approximate invariance. Transformers beat both.
- The models trained *only on synthetic data* generalize **zero-shot to real tabular OpenML tasks**, and provide excellent warm-start initializations that converge far faster than training from the prior.

**Initial takeaway.** This is the shard's systematic "how to amortize a posterior" study. Its central lesson — **reverse KL for robustness and high-dim prediction, forward KL for capturing low-dim multimodality** — directly informs the design axis that Reuter (forward-KL/flow-matching, distribution) and Mittal-Parametric (point-vs-distribution) sit on.

### Phase 2 — Graduate-Level Deep Dive

#### 2.1 Setup: from per-dataset optimization to amortization

Classic approximate inference solves, *for each* $\mathcal{D}$,
$$\phi^* = \arg\min_\phi\; D\big(p(\cdot\mid\mathcal{D}),\, q_\phi(\cdot)\big). \tag{4}$$
Reverse-KL VI instantiates $D$ as $D_{R\text{-}KL}(p,q_\phi) = \mathbb{E}_{\theta\sim q_\phi}[\log \frac{q_\phi(\theta)}{p(\theta\mid\mathcal{D})}]$, equivalently maximizing the ELBO. Forward-KL/EP instantiates $D_{F\text{-}KL} = \mathbb{E}_{\theta\sim p(\cdot\mid\mathcal{D})}[\log\frac{p(\theta\mid\mathcal{D})}{q_\phi(\theta)}]$.

**Amortization** replaces the per-dataset $q_\phi(\cdot)$ with a single **conditional** model $q_\phi(\cdot\mid\mathcal{D})$ trained across many datasets:
$$\phi^* = \arg\min_\phi\; \mathbb{E}_{\mathcal{D}\sim\chi}\, D\big(p(\cdot\mid\mathcal{D}),\, q_\phi(\cdot\mid\mathcal{D})\big), \tag{9}$$
where $\chi$ is a distribution over datasets. This is the "in-context posterior estimator."

#### 2.2 The two objectives — derivations that expose the forward/reverse asymmetry

**Forward KL → neural posterior estimation.** When $\chi$ is exactly the assumed model's marginal, i.e.
$$\chi(\mathcal{D}) = \int p(\theta)\prod_{x_n\in\mathcal{D}} p(x_n\mid\theta)\, d\theta, \tag{10}$$
the forward-KL objective simplifies dramatically:
$$\phi^*_{F\text{-}KL} = \arg\min_\phi\; \mathbb{E}_{\mathcal{D}\sim\chi}\,\mathbb{E}_{\theta\sim p(\cdot\mid\mathcal{D})}\!\left[\log\frac{p(\theta\mid\mathcal{D})}{q_\phi(\theta\mid\mathcal{D})}\right] \tag{11}$$
$$= \arg\min_\phi\; \mathbb{E}_{\theta}\,\mathbb{E}_{\mathcal{D}\sim p(\cdot\mid\theta)}\!\left[-\log q_\phi(\theta\mid\mathcal{D})\right]. \tag{12}$$

*Derivation of the key step (11)→(12).* Drop the $\theta$-only term $\log p(\theta\mid\mathcal{D})$ (constant in $\phi$), leaving $\mathbb{E}_{\mathcal{D}\sim\chi}\mathbb{E}_{\theta\sim p(\theta\mid\mathcal{D})}[-\log q_\phi(\theta\mid\mathcal{D})]$. The nested expectation $\mathbb{E}_{\mathcal{D}\sim\chi}\mathbb{E}_{\theta\sim p(\theta\mid\mathcal{D})}[\,\cdot\,]$ is an expectation over the **joint** $p(\theta,\mathcal{D})$. By Bayes' factorization $p(\theta,\mathcal{D}) = p(\theta)\,p(\mathcal{D}\mid\theta)$, this reverses the sampling order to: draw $\theta\sim p(\theta)$, then $\mathcal{D}\sim p(\cdot\mid\theta)$ — i.e. **ancestral sampling from the simulator**. This is exactly the tower-property/joint-sampling trick (identical in spirit to Reuter's Proposition 1). It removes any need to sample or evaluate the true posterior — the hurdle that cripples classic Expectation Propagation. **But** the change of expectation is *only valid when $\mathcal{D}$ is sampled according to the assumed $p$.** That constraint is the Achilles heel: forward KL is *locked to simulated data from the correct model*.

**Reverse KL → amortized VI, free choice of $\chi$.**
$$\phi^*_{R\text{-}KL} = \arg\min_\phi\; \mathbb{E}_{\mathcal{D}\sim\chi}\,\mathbb{E}_{\theta\sim q_\phi(\cdot\mid\mathcal{D})}\!\left[\log\frac{q_\phi(\theta\mid\mathcal{D})}{p(\theta\mid\mathcal{D})}\right] = \arg\min_\phi\; \mathbb{E}_{\mathcal{D}\sim\chi}\,\mathbb{E}_{\theta\sim q_\phi(\cdot\mid\mathcal{D})}\!\left[\log\frac{q_\phi(\theta\mid\mathcal{D})}{p(\mathcal{D},\theta)}\right]. \tag{13,14}$$

*Why the second form is trainable.* The intractable normalizer $p(\mathcal{D})$ in $p(\theta\mid\mathcal{D}) = p(\mathcal{D},\theta)/p(\mathcal{D})$ contributes only an additive $\log p(\mathcal{D})$ constant in $\phi$, so it drops out of the gradient — leaving the *unnormalized joint* $p(\mathcal{D},\theta) = p(\theta)\prod_n p(x_n\mid\theta)$, which is evaluable pointwise. **Crucially, the outer expectation is over $\mathcal{D}\sim\chi$ with $\chi$ arbitrary:** because the inner expectation samples $\theta\sim q_\phi$ (the model itself), not from the posterior, *no ground-truth $(\theta,\mathcal{D})$ pairs are needed*. This is the asymmetry: reverse KL can consume real data whose true generative process is unknown/misspecified; forward KL cannot. (The gradient of the reverse-KL inner expectation w.r.t. $\phi$ is handled by reparametrization through $q_\phi$ — a diagonal Gaussian or a normalizing flow — since both admit pathwise gradients.)

#### 2.3 Density and architecture choices

- **Density $q_\phi(\cdot\mid\mathcal{D})$:** diagonal Gaussian (cheap, unimodal) or **discrete-time normalizing flow** (expressive, multimodal-capable).
- **Set-conditioner:** GRU (sequential, *not* permutation-invariant), DeepSets (invariant via fixed pooling), Transformer (invariant via attention, no positional encoding, no fixed pool).
- **Variable feature dimensions:** low-dim problems are embedded into a fixed 100-D space by **masking** unused dimensions/parameters to zero (à la TabPFN), so a *single* estimator handles, e.g., both 1-D and 50-D nonlinear regression.

#### 2.4 Findings that matter for the shard's design axis

- **Forward vs reverse KL (the central result):** forward KL better captures **low-dimensional multimodality** (GMM cluster labels visibly switch under forward KL, evidencing mode coverage; reverse KL locks to one labeling). But reverse KL **wins in high dimensions** on both predictive (L2/accuracy) and sample-based ($W_2$, symmetric KL) metrics, and dominates under **misspecification / sim-to-real** (Table 4: reverse KL L2 $\approx 0.35$ vs forward KL $\approx 8$–$15$ on the same OOD transfer; "+switched data" — training reverse KL directly on real data — improves further).
- **Capacity of $q_\phi$:** normalizing flows help forward KL **substantially** but reverse KL only **marginally** — because reverse KL's mode-seeking tendency latches onto a single mode even when given multimodal capacity; forward KL without flow capacity badly *overestimates variance*.
- **Architecture:** Transformer > GRU > DeepSets. DeepSets' fixed pooling caps expressivity; GRUs *learn* approximate invariance and beat DeepSets despite lacking the inductive bias.

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** Real-world motivation (polling, COVID compartment models, cryo-EM) where the probabilistic model is fixed but data streams in — ideal for amortization. Gap: no rigorous analysis of in-context *posterior-estimation* objectives. Contributions: (1) general framework with forward/reverse KL objectives; (2) benchmark of architectures × density parametrizations × objectives; (3) OOD/misspecification/sim-to-real evaluation.
- **§2 Background.** Bayesian inference (eqs 1–3: PPD as $\mathbb{E}_{\theta\mid\mathcal{D}}[p(x^*\mid\theta)]$, Monte Carlo estimate, Bayes rule). Approximate inference: sampling (MCMC, Langevin/HMC) vs variational (eq 4). Reverse-KL VI = ELBO (eqs 5–6); forward-KL = EP (eq 7). PPD via $q_{\phi^*}$ (eq 8). Estimators & amortization: VAE encoder as canonical amortizer; NPs and SBI-NPE amortize dataset-conditioned encoders under reverse/forward KL respectively; ICL viewed as amortizing the *predictive* $p(y^*\mid x^*,\mathcal{D})$.
- **§3 Posterior Estimation from Data in Context.** Reframes per-dataset VI/EP as amortized in-context estimation (eq 9). Forward KL ⇒ NPE under constraint (10), simplifying to joint-sampling ML (eqs 11–12). Reverse KL ⇒ free $\chi$ (eqs 13–14). Discusses permutation invariance requirement (DeepSets/Transformer yes, GRU no) and $\chi$ as black-box simulator via ancestral sampling.
- **§4 Experiments.** Tasks: Gaussian mean, GMM means, (non)linear regression, (non)linear classification. Baselines: prior (Random), MLE (Optimization), Langevin, HMC. Metrics: predictive (L2/accuracy, eq 15) and sample-based ($W_2^2$ eq 16, symmetric KL). §4.1 zero-shot posterior approximation (Tables 1: fixed-dim; rev-KL+flow+transformer best). §4.2 variable feature dims via masking (Table 2). §4.3 misspecification (rev-KL robust, Table 4). §4.4 tabular OpenML benchmarks (Table 3; MAP-finetuning warm-start, Figure 2). §4.5 posterior quality (symmetric KL Table 5, $W_2^2$ Table 6).
- **§5 Discussion/Conclusion.** Forward vs reverse KL trade-off (multimodal low-D vs high-D + robustness); architecture (GRU>DeepSets, Transformer best); capacity of $q_\phi$ (flow helps forward more than reverse).

---

## 10. Mittal et al. 2025 — In-Context Parametric Inference: Point or Distribution Estimators?

**PDF:** `docs/project/references/in_context_learning/sources/Mittal et al. 2025 - In-Context Parametric Inference - Point or Distribution Estimators.pdf`
**Status:** arXiv preprint (arXiv:2502.11617v1, 17 Feb 2025). Authors: Sarthak Mittal, Yoshua Bengio, Nikolay Malkin, Guillaume Lajoie (Mila / U. Edinburgh).

### Phase 1 — Foundational Overview (undergraduate level)

**The problem in plain words.** When you infer a model's parameters from data you can either report **a single best value** (a "point estimate" — maximum likelihood MLE, or maximum-a-posteriori MAP) or **a whole distribution** over the parameters (the Bayesian posterior). Bayesian theory says that, *at optimality*, using the full posterior to make predictions is the right thing to do — you average over all plausible parameter values. But that optimality assumes you can represent the posterior perfectly. In practice, amortized/in-context estimators use limited families (a Gaussian, a normalizing flow) and must generalize to new datasets, introducing an **amortization gap**. So the practical question is genuinely open: **for downstream prediction, is it better to have an in-context network output a point estimate, or a full posterior?**

**The core question, sharpened.** This is the direct sequel to the Amortized paper. It builds a *hierarchical taxonomy* of in-context estimators (Figure 1): **Frequentist** (MLE) and **Bayesian point** (MAP) on the point side; and three flavors of distribution estimators — **sample-based** (need simulator draws: forward-KL Gaussian/flow, score-based diffusion, flow-matching CNF), **variational** (need only the joint density: reverse-KL Gaussian/flow, diffusion samplers like pDEM), and **sample+variational** (symmetric KL). Then it runs all of them head-to-head on a large benchmark (88 tasks → 324 trained models per estimator) judged purely by **predictive performance**.

**Key findings.**
- **Amortized point estimators (MLE/MAP) generally win**, especially on high-dimensional tasks (e.g. inferring the weights of a 2-layer neural net). In the winner-take-all aggregate, point methods win **~71%** of tasks (roughly split MLE/MAP), vs ~9% variational, ~15% sample+variational, ~5% sample-based.
- Distribution estimators **remain competitive only in some low-dimensional problems**.
- Within Bayesian methods, the **simple Gaussian** often beat the fancier normalizing-flow and diffusion posteriors, and **symmetric KL** was the strongest posterior-training signal.
- The authors argue the culprit is **fundamental, not just an engineering gap**: multimodal parameter posteriors (from likelihood symmetries — label-switching in mixtures, weight-permutation symmetries in neural nets) have a **combinatorial explosion of equivalent modes**. Representing all these redundant modes demands ever-more posterior expressivity, yet they all give equivalent predictions — so a point estimate captures the useful information far more cheaply.

**Initial takeaway.** This is the shard's most *deflationary* and most decision-relevant paper: it says that for the *practical goal of prediction*, the elaborate machinery of in-context distribution estimation (including Reuter-style flow matching) often does not pay off versus a plain amortized point estimate — precisely because of parameter-space multimodality/symmetry. It sets the pointed counter-hypothesis against Reuter's "full-distribution wins."

### Phase 2 — Graduate-Level Deep Dive

#### 3.1 The estimator taxonomy and its training objectives

**Point estimators.** An amortized model $\theta = f(\mathcal{D};\phi)$ directly outputs the parameter. Training minimizes the (amortized) MAP or MLE objective; for MAP:
$$\mathcal{L}_{MAP}(\mathcal{D}) = \log p\big(f(\mathcal{D};\phi)\big) + \sum_{i=1}^{|\mathcal{D}|}\log p\big(y_i\mid x_i,\, f(\mathcal{D};\phi)\big). \tag{8}$$
An unbiased gradient uses a minibatch $\mathcal{B}\subset\mathcal{D}$ reweighted by $\frac{|\mathcal{D}|}{|\mathcal{B}|}$. MLE drops the prior term. Note the structural parallel to MAP $= \arg\max_\theta \log p(\theta) + \sum_i \log p(y_i\mid x_i,\theta)$ — the amortizer just *learns the argmax as a function of $\mathcal{D}$*.

**Sample-based posterior estimators (forward KL).** Given joint draws $(\theta,\mathcal{D})\sim\chi$ (requires well-specification and an exposed $\theta$), fit a conditional generative model by log-likelihood, which equals forward-KL minimization:
$$\mathbb{E}_{(\theta,\mathcal{D})\sim\chi}\!\left[-\log q_\phi(\theta\mid\mathcal{D})\right] = \mathbb{E}_{\mathcal{D}\sim\chi}\, D_{KL}\!\big(p(\theta\mid\mathcal{D})\,\|\,q_\phi(\theta\mid\mathcal{D})\big) + \text{const}. \tag{9}$$
*Derivation.* $\mathbb{E}_{(\theta,\mathcal{D})}[-\log q_\phi] = \mathbb{E}_{\mathcal{D}}\mathbb{E}_{\theta\mid\mathcal{D}}[-\log q_\phi]$; adding and subtracting $\mathbb{E}_{\theta\mid\mathcal{D}}[\log p(\theta\mid\mathcal{D})]$ (the negative entropy of the true posterior, constant in $\phi$) yields $\mathbb{E}_{\mathcal{D}}[D_{KL}(p\|q_\phi)] + \text{const}$. Instantiations: **Gaussian** (optimal solution matches first two posterior moments per $\mathcal{D}$), **normalizing flows**, **continuous-time NFs / flow-matching** (simulation-free, no architectural constraint — the Reuter objective), **score-based diffusion** (learns time-conditioned score; equivalent to a variational upper bound on eq 9).

**Variational posterior estimators (reverse KL).** No simulator draws needed; only pointwise access to the unnormalized joint $p(\theta,\mathcal{D}) = p(\theta)p(\mathcal{D}\mid\theta)$. The reverse KL is exactly optimizable and reads as **entropy-regularized maximum likelihood**:
$$D_{KL}\!\big(q_\phi(\theta\mid\mathcal{D})\,\|\,p(\theta\mid\mathcal{D})\big) = \mathbb{E}_{\theta\sim q_\phi(\theta\mid\mathcal{D})}\!\left[-\log p(\theta\mid\mathcal{D})\right] - H\!\left[q_\phi(\theta\mid\mathcal{D})\right]. \tag{10}$$
*Reading.* The first term pulls $q_\phi$ toward high-posterior-density regions (the "likelihood" pull); the entropy term $-H[q_\phi]$ resists collapse to a point. Because there is **no expectation over $\mathcal{D}$ tied to $p$**, one may optimize (10) for $\mathcal{D}$ drawn from *any* distribution, or even a single fixed $\mathcal{D}$ — the same free-$\chi$ property established in the Amortized paper. Instantiations: reverse-KL Gaussian/flow, and **diffusion samplers** (fit an unnormalized-density sampler; e.g. **pDEM**, denoising energy matching, regresses the denoiser to a biased-but-consistent Monte-Carlo score estimate of $p(\theta\mid\mathcal{D})$).

**Sample + variational.** **Symmetric KL** = equally-weighted forward + reverse KL, for Gaussian or discrete-flow $q_\phi$.

Mean-seeking vs mode-seeking is stated crisply: **forward KL is mean-seeking and over-estimates variance**; **reverse KL is mode-seeking and under-estimates variance / captures few modes**.

#### 3.2 Evaluation design — the point-vs-distribution comparison made fair

Predictions on new $x^*$ use the posterior predictive $p(y\mid x^*,\mathcal{D}) = \int p(y\mid x^*,\theta)p(\theta\mid\mathcal{D})d\theta \approx \mathbb{E}_{\theta\sim q_\phi}[p(y\mid x^*,\theta)]$ (eq 7); point estimators plug their single $\theta$ into $p(y\mid x^*,\theta)$. Two metric variants:
$$\underbrace{\mathbb{E}_{x^*,y^*,\mathcal{D}}\,\mathbb{E}_{\theta\sim q_\phi}\,\|\hat y - y^*\|^2}_{\text{expected loss (11)}} \qquad\text{vs}\qquad \underbrace{\mathbb{E}_{x^*,y^*,\mathcal{D}}\,\big\|\mathbb{E}_{\theta\sim q_\phi}\hat y - y^*\big\|^2}_{\text{ensemble loss (12)}}.$$
They report the **ensemble-based** metric (average the predictions over posterior samples, *then* score) — this is the fair, favorable-to-Bayes choice, since it uses the full posterior's averaging benefit; for point estimators the two metrics coincide (single $\theta$). This design deliberately gives distribution estimators their best shot; they still lose.

#### 3.3 Results and the "fundamental obstacle" argument

- **Aggregate ranking (Figure 2), winner-take-all over 88 tasks:** Point 71.6% (MLE 36.4% + MAP 35.2%); Sample+Variational 14.9%; Variational 9.2%; Sample-based 4.6%. Among $q_\phi$ choices: Gaussian 17.0% > Normalizing Flow 8.0% > Diffusion 3.4%. Among training signals for posteriors: Symmetric-KL Gaussian 13.6% is the strongest single posterior method.
- **Fixed-dim (Table 1) and 2-layer-NN (Table 2):** point estimators clearly superior, worse for distribution methods as dimension/nonlinearity grows; forward-KL variants collapse on high-dim classification (accuracy near chance).
- **Misspecification / OOD (Tables 4, 5):** point estimators and variational methods can retrain on the *target* distribution ("+switched data") and win; sample-based (forward-KL) methods are stuck on the assumed-model simulator and transfer poorly — the same forward/reverse asymmetry as the Amortized paper.
- **The mechanistic claim (Conclusion):** the failure of distribution estimators is tied to **multimodality from likelihood symmetries** — mixture-model identifiability (label switching) and Bayesian-NN weight-permutation symmetries produce a number of modes that **explodes combinatorially in network width**. Each redundant mode needs representation capacity, yet all give equivalent predictions; mode-connectivity results (low energy barriers modulo symmetry) suggest the posterior is "effectively simpler" than its mode count, so a point estimate loses little. The authors recommend **hybrid** approaches (subnetwork/partial Bayesian treatment; more efficient amortized variational families).

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** Frequentist (point, fixed-unknown $\theta$) vs Bayesian (distribution, random $\theta$). Point estimation dominates deep learning; trade-offs under *amortization* are poorly understood. Cites the theory-vs-practice gap in ICL (Xie/Akyürek Bayes-optimal in theory; Garg/Falck falls short in practice) and VAE/BNN evidence that better variational bounds ≠ better models, cold-posterior effects. Claim: point estimators generally outperform, esp. high-dim.
- **§2 Problem Setup.** Generative model of i.i.d. $\mathcal{D}=\{(x_i,y_i)\}\sim\chi$; well-specified vs misspecified. §2.1 Frequentist/MLE (eqs 1–3, amortized $f(\mathcal{D};\phi)\approx\theta_{MLE}$). §2.2 Bayesian/MAP (eqs 4–5; posterior concentration ⇒ MAP→MLE as $k\to\infty$); amortized $q_\phi(\theta\mid\mathcal{D})$ (eq 6). §2.3 Posterior predictive (eq 7). §2.4 Amortization by in-context estimation (transformer, no positional embeddings for permutation invariance).
- **§3 Amortized Inference Training Objectives.** §3.1 Point estimates (eq 8 MAP loss, minibatch surrogate). §3.2 Posterior estimates: §3.2.1 sample-based (eq 9 forward KL; Gaussian, NF, CNF/flow-matching, score-based diffusion). §3.2.2 variational (eq 10 reverse KL as entropy-regularized ML; diffusion samplers/pDEM; MCMC as non-amortized reference). §3.2.3 sample+variational (symmetric KL). Figure 1 taxonomy.
- **§4 Experiments.** Tasks (GM, GMM, (N)LR, (N)LC; nonlinear via NNs). Baselines (Random, True Posterior, single/multi-chain Langevin & HMC, Optimization). Metrics: expected (11) vs ensemble (12); ensemble reported. §4.1 in-distribution (Figure 2 aggregate; §4.1.1 fixed-dim Tables 1–2; §4.1.2 variable-dim via masking Table 3). §4.2 misspecification (§4.2.1 synthetic Table 4 with "+switched data"; §4.2.2 tabular OpenML Table 5).
- **§5 Conclusion.** Point > distribution for prediction, esp. high-dim/multimodal. Roots the effect in redundant-mode combinatorics (mixture identifiability, BNN symmetries, mode connectivity). Recommends hybrid amortized inference.

---

## 11. Kang et al. 2026 — Transformers Can Learn Posterior Predictive Distributions In-Context

**PDF:** `docs/project/references/in_context_learning/sources/Kang et al. 2026 - Transformers Can Learn Posterior Predictive Distributions In-Context.pdf`
**Status:** arXiv preprint (arXiv:2605.26713v1, 26 May 2026); to appear ICML 2026 (PMLR 306). Authors: Gyeonghun Kang, Changwoo J. Lee, Xiang Cheng (Duke).

### Phase 1 — Foundational Overview (undergraduate level)

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

### Phase 2 — Graduate-Level Deep Dive

#### 4.1 Setup: PFN output as a discretized PPD, and its population minimizer

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

#### 4.2 Theorem 3.1 — attention runs a Richardson iteration for the moments

**KRR view.** $u^*(x) = k_x^\top(G+\sigma^2 I_n)^{-1}v$ is the kernel-ridge-regression solution: $u^* = \arg\min_{u\in\mathcal{H}} \sum_i (v_i - u(x_i))^2 + \sigma^2\|u\|_\mathcal{H}^2$, and $u^*_X$ solves $(G+\sigma^2 I_n)u^*_X = Gv$ by the representer theorem.

**Richardson iteration** for this linear system (componentwise, $u^{(0)}\equiv 0$):
$$u^{(l+1)}(x_j) = (1-\eta^{(l)}\sigma^2)\,u^{(l)}(x_j) + \eta^{(l)}\sum_{i=1}^n \kappa(x_i,x_j)\big(v_i - u^{(l)}(x_i)\big). \tag{7}$$
If $0<\eta^{(l)}<2/(\lambda_1(G)+\sigma^2)$ for all $l$ and $\sum_l\eta^{(l)}=\infty$, then $u^{(L)}(x)\to k_x^\top(G+\sigma^2 I_n)^{-1}v$ as $L\to\infty$.

**Mapping to attention.** Define **unnormalized attention** $\text{Attn}_{M,\kappa}(Z;V,K,Q)_{:,j} = \sum_i \kappa(Kz_i,Qz_j)M_{ij}Vz_i$. The recursion (7) maps exactly: the **decay term $-\eta^{(l)}\sigma^2 u^{(l)}$ is implemented by the learnable skip $S^{(l)}$**, and the **kernel-weighted correction $\eta^{(l)}\sum_i\kappa(x_i,x_j)(v_i-u^{(l)}(x_i))$ is the self-attention** over context tokens. Running *both* right-hand sides $v=Y$ (for $\mu$) and $v=k_x$ (for the variance-reduction term) in parallel (stacked token dimensions) yields both moments.

**Convergence bound (Theorem 3.1):**
$$\big\| \text{TF}_{\vartheta,L} - (\mu,\tau)^\top\big\|_\infty \;\lesssim\; \exp\big(-(1-\rho)L\big), \qquad \rho := 1-\eta(\lambda_n(G)+\sigma^2)\in(0,1),$$
i.e. the transformer inherits the Richardson iteration's **geometric (exponential-in-depth) convergence**; $\rho$ is the convergence factor and $\lambda_n(G)$ the smallest Gram eigenvalue.

#### 4.3 Theorem 4.1 — from moments to a density (the distributional step)

For a 1-D exponential-family target $f_\theta(y) = \exp(\langle\psi(\theta),T(y)\rangle - A(\theta))h(y)$ (Gaussian: $\psi(\mu,\tau)=(\mu/\tau,\,-1/(2\tau))$, $T(y)=(y,y^2)$), the MLP head approximates the natural-parameter map $\psi$ by universal approximation ($\|\psi-\tilde\psi\|\le\delta_{MLP}$), and a **readout matrix $\Xi$ with rows $T(\xi_c)$** (sufficient statistics at bin midpoints $\xi_c$) produces logits $(\Xi\tilde\psi(\theta))_c = \langle\tilde\psi(\theta),T(\xi_c)\rangle$. **Softmax then removes the log-normalizer $A(\theta)$ automatically** (it is an additive constant across bins), giving the piecewise-constant density. This is the key insight explaining *why real PFN MLP heads work*: the head weights encode the sufficient statistics per bin.

**End-to-end bound (Theorem 4.1):**
$$\text{TV}(p_n, q_\vartheta) \;\lesssim\; \underbrace{e^{-(1-\rho)L}}_{\text{attention/solver}} + \underbrace{\delta_{MLP}}_{\text{natural-param MLP}} + \underbrace{\tfrac{1}{C}}_{\text{binning}} + \underbrace{\varepsilon_{\text{tail}}(Z^{(0)})}_{\text{truncation}}.$$
In the practical regime where $C$, MLP width, and truncation interval are chosen large, **the attention depth is the binding constraint** — motivating §5.

#### 4.4 Theorems 5.1–5.3 — why normalization and depth govern $n$-generalization

The transformer is a **single-weight solver** that must approximately solve $(G+\sigma^2 I_n)u = Gv$ for *any* $n$ within a *fixed* depth $L$.

- **Theorem 5.1 (step size $\sim 1/n$).** With $x_i\sim\mathcal{N}(0,I_d/d)$, for linear and RBF kernels $\lambda_1(G)=\Theta(n)$ w.h.p., so the admissible step size bound $\eta<2/(\lambda_1(G)+\sigma^2) = \Theta(1/n)$. A step size $\hat\eta$ tuned to $[n_{\min},n_{\max}]$ is **too conservative for $n'\ll n_{\min}$** (under-converges in $L$ steps) and **violates the bound for $n'\gg n_{\max}$** (diverges). This is the mechanism behind PFN failure outside the pre-training range.
- **Normalization = Jacobi preconditioning (Theorem 5.1 remedy).** Left-multiply by $D^{-1}$, $D=\text{diag}(G\mathbf{1}_n)$: $D^{-1}(G+\sigma^2 I_n)u = D^{-1}Gv$, same fixed point. Since $D^{-1}G$ is row-stochastic, $\lambda_1(D^{-1}(G+\sigma^2 I_n))\in[1,1+\sigma^2]$ — **bounded independent of $n$**, so a fixed step size works for all $n$. The preconditioned recursion
$$u^{(l+1)}_{pr}(x) = \Big(1-\eta\tfrac{s_x}{\sigma^2}\Big)u^{(l)}_{pr}(x) + \tfrac{\eta}{s_x}\sum_i\kappa(x_i,x)\big(v_i - u^{(l)}_{pr}(x_i)\big), \quad s_x=\textstyle\sum_i\kappa(x_i,x), \tag{14}$$
is **implemented by normalized (softmax-style) attention** $\text{Attn}^{na}_{M,\kappa}$ whose weights are divided by their sum $s_j$, plus a token-wise $1/s_{x_j}$ scaling of the skip term. So the ubiquitous softmax denominator in real transformers *is* the preconditioner enabling $n$-generalization — a striking mechanistic result.
- **Theorem 5.2 & 5.3 (bigger $n$ needs more depth).** Even preconditioned, $\text{cond}(D^{-1}(G+\sigma^2 I_n)) = \Theta(n)$ a.s. for RBF (Gram increasingly ill-conditioned as eigenvalues decay geometrically). The optimal convergence factor $\rho^* = 1-2/(\text{cond}(G+\sigma^2 I_n)+1)\to 1$ as $n$ grows, so to hit tolerance $\epsilon$ it suffices that $L\gtrsim n\log(1/\epsilon)$ — **required depth grows ~linearly in $n$**. Hence deeper stacks (and wider pre-training ranges exposing more eigenspectra) are essential for large-context PFNs — matching the empirical TabPFN-2.5/3 trend of 18–24 layers.

#### 4.5 Relation to the shard's axis, and limitations

Kang is the **distribution-estimator theory** the two Mittal papers lack: it proves attention *can* emit a calibrated PPD, not merely a point. But it is honest that these are **constructive existence results** (like von Oswald / Akyürek): they show the architecture *can* realize the mechanism, not that standard PFN pre-training *does* discover it. The learnable-relaxation experiments (diagonal $Q,K$, Richardson sparsity pattern) partially bridge existence→learning by reproducing the predicted depth/bin/normalization effects. Scope: GP regression (closed-form PPD), with sketched extensions to hierarchical GPs (multi-head = one head per mixture component) and latent-GP/non-Gaussian models.

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** PFNs approximate the PPD via ICL; strong empirical UQ (95% intervals match truth; BayesOpt acquisition). Gap: no theory for *distributional* (not point) ICL. Prior theory = "attention implements gradient descent" for point prediction (Akyürek, von Oswald, Bai, Ahn, Zhang, Mahankali). Motivated by depth (TabPFN-2.5/3 use 18–24 layers), bin resolution, and $n$-generalization (TabPFNv2 pretrained to 2048, works to 10000). Three contributions: explicit construction + error bounds (Thm 3.1, 4.1); $n$-generalization mechanism via spectra/normalization (Thm 5.1–5.3); empirical validation.
- **§2 Background and Problem Setup.** PPD definition (eq 1); PFN objective as expected-KL minimization; discretized output (eq 2) and population minimizer $q^*$ (eq 3, TV bound). Transformer architecture (tokens, masked attention, learnable skip, eqs 4–5). GP regression closed-form $\mu,\tau$ reducing to $k_x^\top(G+\sigma^2 I)^{-1}v$.
- **§3 Attention Computes Moments of PPD.** KRR + representer theorem; Richardson iteration (eq 7) + convergence condition. Unnormalized attention definition (eq 8). Construction implementing both recursions in parallel; **Theorem 3.1** (exponential-in-$L$ convergence, factor $\rho$). Remark 3.2 (hierarchical-GP multi-head extension).
- **§4 From Moments to PPD.** Exponential-family densities; MLP for natural-parameter map (universal approximation, $\delta_{MLP}$); softmax discretization with sufficient-statistic readout $\Xi$ (removes $A(\theta)$). **Theorem 4.1** (TV bound = attention + MLP + $1/C$ + tail). Interpretation: depth controls moment convergence, $C$ controls discretization; justifies binned-MLP PFN practice.
- **§5 Generalizing Beyond Pretrain Sample Sizes.** Pretraining over $n\in[n_{\min},n_{\max}]$. Step size decreases with $n$ (system eq 11); **Theorem 5.1** ($\eta\sim 1/n$; two failure modes). Jacobi preconditioning (eq 13) bounds spectrum; **normalized attention implements it** (eq 14, $\text{Attn}^{na}$). **Theorem 5.2** ($\text{cond}=\Theta(n)$ for RBF); **Theorem 5.3** ($L\gtrsim n\log(1/\epsilon)$ depth requirement). Remark 5.4 (learnable models more flexible than the construction).
- **§6 Numerical Studies.** BLR (linear kernel) + RBF regression; theory/learnable/normalized parameterizations. §6.1 validates Thm 4.1 (Figure 2: log-TV linear in log-bins, power-law in log-depth). §6.2 normalized vs unnormalized (Figure 4: unnormalized spikes OOD, normalized stable). §6.3 validates Thm 5.3 (Figure 5: depth + pretraining-range improve MSE/coverage/width; iteration-limited).
- **§7 Discussion.** Constructive account for GP PPDs; extensions (hierarchical GP mixtures, latent-GP/GP-classification). Limitation: existence results, not a proof that pretraining discovers them; open problems on optimization guarantees.

---

## 12. Schug et al. 2025 — Attention as a Hypernetwork

**PDF:** `docs/project/references/in_context_learning/sources/Schug et al. 2025 - Attention as a Hypernetwork.pdf`
**Venue:** ICLR 2025. **Authors:** Simon Schug, Seijin Kobayashi, Yassir Akram, João Sacramento, Razvan Pascanu (ETH Zürich / Google DeepMind).

### Phase 1 — Foundational Overview (undergraduate level)

**The question in plain terms.** Transformers can sometimes solve *new combinations* of familiar pieces — e.g. if trained on tasks that use rules A, B, C, D in various pairs, they can handle the unseen pair (A, D). This is called **compositional generalization**. Why does the attention mechanism support it? This paper answers by showing that **multi-head attention is secretly a hypernetwork**.

- A **hypernetwork** is a small network that outputs the *weights* of a second ("value") network, which then does the actual work on the input. The hypernetwork takes a short **latent code** describing "which operation to run."
- The paper's central re-derivation: for a single query token, if you look at the attention scores **across the heads** (not across the keys, as usual), that little vector of per-head scores *is* the hypernetwork's latent code. It selects a linear combination of fixed per-head "operation matrices" and applies the result to each key's value.
- So the number of heads `H` = the dimension of the hypernetwork's latent code. Each head contributes a reusable building-block operation; the attention scores mix them.

**Key findings.**
1. **The equivalence** — multi-head attention = a linear hypernetwork generating a per-(key,query) linear value network (Eqs. 2–5 below). This is a *different* decomposition from the well-known "linear attention = fast-weight programmer" (Schlag 2021, §13), which sums outer products over the **key** index; Schug sums over the **head** index.
2. **HYLA (Hypernetwork Linear Attention)** — a drop-in modification that makes the value network *nonlinear* (adds a ReLU) **without adding parameters**, by exploiting the fact that the value+output projections already form a deep-linear network. It also normalizes attention scores across heads (RMSHead) rather than across keys (softmax). HYLA improves compositional generalization.
3. **The latent code is interpretable** — on two abstract-reasoning tasks (a fuzzy-logic task and the new SRAVEN benchmark), the per-head attention-score vector for the "response" token clusters by the *subtask / rule* being applied. You can train a logistic-regression classifier on the latent code and read off which rule the network is using — even on held-out task compositions.
4. **Scaling helps** — bigger models + more data → compositional generalization emerges + latent space becomes more functionally structured. Killing the mechanism (using a single head) hurts.
5. **SRAVEN** — a new symbolic version of Raven's Progressive Matrices with parametric difficulty and controllable rule-composition splits, including a "finding correspondences" difficulty (column-wise feature permutations).

**Initial takeaway.** Multiple attention heads aren't just an engineering trick for stability — they give attention a built-in hypernetwork that composes reusable operations from a compact code. Strengthening that mechanism (HYLA) improves systematic generalization and even closes part of the linear-vs-softmax gap in language modeling. **Bridge to the project:** this is a first-principles argument that attention *is* conditional modulation — exactly the FiLM/hypernetwork family — so a FiLM or hypernetwork conditioning module is not an add-on foreign to attention, but a more explicit exposure of a mechanism attention already contains.

### Phase 2 — Graduate-Level Deep Dive

#### 2.1 Setup and notation

Bold lower-case = vectors ($\mathbf{z}$), bold upper-case = matrices ($\mathbf{A}$), bold italic upper-case = learnable parameters ($\boldsymbol{W}_V$). Self-attention maps an input sequence $\mathbf{X}\in\mathbb{R}^{D\times T}$ (feature dim $D$, sequence length $T$) to outputs $\mathbf{Y}\in\mathbb{R}^{D\times T}$. There are $H$ heads with per-head dimension $D_{\text{head}}=D/H$.

For each head $h\in\{1,\dots,H\}$, project into keys and queries with head-specific matrices $\boldsymbol{W}_h^{\text{key}},\boldsymbol{W}_h^{\text{query}},\boldsymbol{W}_h^{\text{value}}\in\mathbb{R}^{D_{\text{head}}\times D}$:

$$
\mathbf{K}_h = \boldsymbol{W}_h^{\text{key}}\mathbf{X},\qquad
\mathbf{Q}_h = \boldsymbol{W}_h^{\text{query}}\mathbf{X}.
$$

The (unnormalized) head-specific attention matrix and the stacked, normalized tensor are

$$
\tilde{\mathbf{A}}_h = \frac{\mathbf{Q}_h^{\top}\mathbf{K}_h}{\sqrt{D_{\text{head}}}},
\qquad
\mathbf{A}=\sigma\!\left(\big[\tilde{\mathbf{A}}_1\ \tilde{\mathbf{A}}_2\ \cdots\ \tilde{\mathbf{A}}_H\big]\right),
\qquad
\mathbf{A}=(a_{h,q,k})_{h,q,k}\in\mathbb{R}^{H\times T\times T}.
$$

$\sigma(\cdot)$ is the normalizer: **identity** for linear attention, **softmax over the key index $k$** (per head $h$, per query $q$) for standard softmax attention.

#### 2.2 The core derivation: attention as a hypernetwork (Eqs. 2–5)

Fix a single query index $q$. Let $\mathbf{x}_k\in\mathbb{R}^{D}$ be the $k$-th column of $\mathbf{X}$. Standard multi-head attention writes the output as the output-projection $\boldsymbol{W}^{\text{out}}\in\mathbb{R}^{D\times D}$ applied to the concatenation ($\bigoplus$) over heads of the per-head attention-weighted value sums:

$$
\text{MHA}_q(\mathbf{X}) \;=\; \boldsymbol{W}^{\text{out}}\bigoplus_{h=1}^{H}\sum_{k=1}^{T} a_{h,q,k}\,\boldsymbol{W}_h^{\text{value}}\mathbf{x}_k. \tag{2}
$$

**Step 1 — split the output projection over heads.** Write $\boldsymbol{W}^{\text{out}}=\bigoplus_{h=1}^{H}\boldsymbol{W}_h^{\text{out}}$ where each slice $\boldsymbol{W}_h^{\text{out}}\in\mathbb{R}^{D\times D_{\text{head}}}$. A block matrix times a stacked vector is the sum of block-times-subvector, so the concatenation collapses to a sum over heads:

$$
\text{MHA}_q(\mathbf{X}) \;=\; \sum_{h=1}^{H}\boldsymbol{W}_h^{\text{out}}\sum_{k=1}^{T} a_{h,q,k}\,\boldsymbol{W}_h^{\text{value}}\mathbf{x}_k. \tag{3}
$$

**Step 2 — swap the order of summation** ($\sum_h\sum_k = \sum_k\sum_h$) and pull the scalar $a_{h,q,k}$ out. Because $a_{h,q,k}$ is a scalar it commutes with the matrices:

$$
\text{MHA}_q(\mathbf{X}) \;=\; \sum_{k=1}^{T}\Bigg(\underbrace{\sum_{h=1}^{H}\underbrace{a_{h,q,k}}_{\text{latent code}}\,\boldsymbol{W}_h^{\text{out}}\boldsymbol{W}_h^{\text{value}}}_{\text{hypernetwork}}\Bigg)\mathbf{x}_k. \tag{4}
$$

**Step 3 — name the generated value network.** Define the per-(key,query) weight matrix

$$
\mathbf{W}_{q,k} \;:=\; \sum_{h=1}^{H} a_{h,q,k}\,\boldsymbol{W}_h^{\text{out}}\boldsymbol{W}_h^{\text{value}}\;\in\;\mathbb{R}^{D\times D},
\qquad\text{so}\qquad
\text{MHA}_q(\mathbf{X}) \;=\; \sum_{k=1}^{T}\mathbf{W}_{q,k}\,\mathbf{x}_k. \tag{5}
$$

**Reading of the result.**
- The vector $\mathbf{a}_{q,k}:=(a_{1,q,k},\dots,a_{H,q,k})^{\top}\in\mathbb{R}^{H}$ is the **latent code**: an $H$-dimensional specification of the operation to run on key $k$ for query $q$. Latent dimension = number of heads.
- The map $\mathbf{a}_{q,k}\mapsto \mathbf{W}_{q,k}=\sum_h a_{h,q,k}\,\boldsymbol{W}_h^{\text{out}}\boldsymbol{W}_h^{\text{value}}$ is a **linear hypernetwork** whose "basis operations" are the fixed rank-$\le D_{\text{head}}$ matrices $\boldsymbol{W}_h^{\text{out}}\boldsymbol{W}_h^{\text{value}}$. The *same* hypernetwork (same basis matrices) is reused for every $(q,k)$ pair — this weight-sharing is exactly what incentivizes reuse and recombination of operations, hence compositional generalization.
- The dot products $\tilde a_{h,q,k}\propto \mathbf{q}_{h,q}^{\top}\mathbf{k}_{h,k}$ that produce the code can be read as an **amortized inference** step: from the key/query content, infer which composition of operations to apply.

**Contrast with the fast-weight view (Schlag 2021).** In linear attention as a fast-weight programmer, the constructed weight matrix is $\mathbf{W}_{\text{fw}}=\sum_k \mathbf{v}_k\mathbf{k}_k^{\top}$ — a sum of outer products over the **key** index, one global matrix accumulated over the sequence. Schug's decomposition instead sums over the **head** index, producing a *distinct* $\mathbf{W}_{q,k}$ per key-query pair. The two views are complementary factorizations of the same layer.

#### 2.3 HYLA — Hypernetwork Linear Attention (Eqs. 6–7)

Observation: in Eq. (4) the generated operator $\boldsymbol{W}_h^{\text{out}}\boldsymbol{W}_h^{\text{value}}$ is a *product of two linear maps* — i.e. a deep-linear network with no nonlinearity between them. HYLA inserts an element-wise nonlinearity $\phi$ (ReLU) between the value projection and the output projection, turning the generated value network into a genuine one-hidden-layer MLP **without any new parameters**:

$$
\text{HYLA}_q(\mathbf{X}) \;=\; \sum_{k=1}^{T}\left(\sum_{h=1}^{H} a_{h,q,k}\,\boldsymbol{W}_h^{\text{out}}\right)\,\phi\!\left(\sum_{h=1}^{H} a_{h,q,k}\,\boldsymbol{W}_h^{\text{value}}\,\mathbf{x}_k\right)
\;=\; \sum_{k=1}^{T}\mathbf{W}'_{q,k}\,\phi\!\left(\mathbf{W}_{q,k}\mathbf{x}_k\right), \tag{6–7}
$$

with $\phi(x)=\max(0,x)$, and the two generated matrices
$\mathbf{W}_{q,k}=\sum_h a_{h,q,k}\boldsymbol{W}_h^{\text{value}}$ (first layer, into hidden) and
$\mathbf{W}'_{q,k}=\sum_h a_{h,q,k}\boldsymbol{W}_h^{\text{out}}$ (second layer, out of hidden). Note that relative to Eq. (4) the single product $\boldsymbol{W}_h^{\text{out}}\boldsymbol{W}_h^{\text{value}}$ is *split* so the nonlinearity can sit between the two factors.

**Normalization — RMSHead.** HYLA sets $\sigma(\cdot)=\text{RMSHead}(\cdot)$: normalize the attention scores across the **head** index (locally per $(q,k)$), using RMSNorm

$$
\text{RMSNorm}(\mathbf{x}) \;=\; \frac{\mathbf{x}}{\sqrt{\tfrac{1}{n}\sum_{i=1}^{n} x_i^2}}\,,
$$

with $n=H$ and no learnable gain. Rationale: this keeps the *variance-preserving* property of standard NN initialization (Glorot & Bengio) for the hypernetwork-generated weights $\mathbf{W}_{q,k}$, stabilizing gradient-based training — because $\mathbf{W}_{q,k}$ is a code-weighted sum of the basis matrices, unbounded codes would blow up the generated weights' scale. Crucially, RMSHead is **local to each $(q,k)$**: unlike softmax-over-keys, it needs **no communication across keys**, preserving linear attention's efficiency.

#### 2.4 Why the single-head case degenerates

With $H=1$, Eq. (4) becomes $\mathbf{W}_{q,k}=a_{1,q,k}\,\boldsymbol{W}_1^{\text{out}}\boldsymbol{W}_1^{\text{value}}$ — a *scalar rescaling* of one fixed operator. The hypernetwork can no longer **compose** multiple basis operations; the latent code collapses to a 1-D gain. The paper confirms empirically (SRAVEN, Fig. 5C) that $H=1$ sharply reduces OOD accuracy, directly supporting the claim that composition across heads is the load-bearing mechanism. This also reframes the classic "you can prune all-but-one head" finding (Voita 2019; Michel 2019) as **module collapse** rather than evidence that heads are redundant.

#### 2.5 Tasks and empirical structure

**Fuzzy-logic task.** Scalars $x_i\in[0,1]$; Zadeh fuzzy operators $x_i\wedge x_j=\min(x_i,x_j)$, $x_i\vee x_j=\max(x_i,x_j)$, $\bar x_i=1-x_i$ (which reduce to Boolean ops on $\{0,1\}$). Each task is a disjunctive-normal-form function of $K$ terms over $L$ variables, e.g. $f=(x_1\wedge x_2\wedge x_3\wedge x_4)\vee(\bar x_1\wedge x_2\wedge \bar x_3\wedge x_4)$. The compositional split holds out a fraction of the $\binom{2^L}{K}$-style term combinations for OOD, so every *term* is seen in training but the *combination* is novel. Result: the per-head latent code of the response token is decodable (F1) into the underlying terms, and tSNE clusters by term. HYLA degrades least as the held-out fraction grows.

**SRAVEN (Symbolic Raven).** Each panel is a $K$-tuple of integers in $\{0,\dots,F-1\}$ (default $K=4$, $F=8$). $K$ of $R=8$ rules (progression, addition, difference, etc., in modular arithmetic mod $F$) govern the features across each row of a $3\times3$ matrix; the model sees 8 context panels and predicts the 9th (all $K$ sub-features must be correct). Compositional split holds out 25% of rule combinations (AB≡BA identified). **Finding correspondences**: a column-specific consistent permutation of features is applied, so the task can't be decomposed into $K$ independent single-feature tasks — the number of hypotheses explodes. Results: with enough scale all attention variants reach ~80% OOD; HYLA leads at small scale; the final-layer latent code clusters by *rule*, and cosine-similarity of average per-rule codes shows semantically related rules (addition vs. difference) sharing nearly identical codes (sign flip of operands).

**Language modeling.** 50M-param decoder-only, C4, 130B tokens: HYLA beats linear attention and approaches softmax attention — notable because softmax's edge is usually attributed to associative-recall / induction-head binding where linear attention struggles. Strengthening the hypernetwork mechanism partly closes that gap.

#### 2.6 Discussion / generality

Multi-head attention is one *granularity* choice for a hypernetwork: it parameterizes **key-query-specific** value networks, then pools (sums) over keys. Other granularities are possible — a query-specific value network that subsumes the key-aggregation, or a full sequence-level operation. This connects to attention-GNNs: the message function becomes a hypernetwork subsuming attention weights, and aggregation is the sum. **Limitation:** analysis is on from-scratch models for controllability; whether pretrained large models form similarly structured latent codes is open.

### Appendix: Section-by-Section Backbone

- **Abstract.** Reformulate MHA as a hypernetwork → attention scores across heads = a composable low-dim latent code specifying key-query operations. Empirically the code predicts subtasks on unseen compositions. Making the generated value network nonlinear (HYLA) improves compositional generalization. Introduce SRAVEN.
- **§1 Introduction.** Compositional generalization = generalizing to unseen combinations of seen constituents; NNs notoriously struggle. Prior ICL mechanisms: gradient descent in-context (von Oswald, Dai, Akyürek), fast-weight programmer view (Schlag/Schmidhuber), task vectors (Hendel, Todd). Key claim: MHA ≡ hypernetwork ⇒ attention scores over heads = latent code. Contributions: reformulation, scaling enables compositional generalization + structured latent space, HYLA parameter-free nonlinear value net, SRAVEN benchmark.
- **§2 Attention as a hypernetwork.** §2.1 recap hypernetworks $h(\mathbf z;\theta)\to \boldsymbol W$ configuring $f(\mathbf x;\boldsymbol W)$; recap MHA; derive Eqs. (1)–(5): latent code $=(a_{h,q,k})_h$, hypernetwork $=\sum_h a_{h,q,k}\boldsymbol W_h^{\text{out}}\boldsymbol W_h^{\text{value}}$. Contrast with key-index outer-product fast-weight view. §2.2 HYLA (Eqs. 6–7): nonlinear value net without extra params + RMSHead normalization across heads; local, no cross-key communication.
- **§3 Fuzzy logic functions.** Zadeh operators (Eq. 8); DNF tasks (Eq. 9); compositional train/OOD split by terms; ICL over N examples; MSE on masked final target. §3.1 all models solve given enough context; HYLA degrades least with more held-out. Novel *unknown* terms are unsolvable by all. §3.2 latent code decodes to terms (logistic regression F1), tSNE clusters by term.
- **§4 SRAVEN.** §4.1 motivation (Raven, finding correspondences). §4.2 task: $K$-tuples of integers, $R=8$ rules, modular arithmetic, 8 context + 1 response panel, 25% rule-combos held out. §4.3 finding correspondences via column-wise feature permutation. §4.4 results: scaling enables compositional generalization (~80% OOD); single-head degenerates (Fig 5C); latent code clusters by rule (Fig 5B); related-rule code reuse (Fig 5E); logistic-regression rule decoding (Fig 5F); difficulty scales with $K$ (Fig 5D).
- **§5 Language modeling.** 50M decoder-only, C4/130B tokens; HYLA > linear, ≈ softmax; hypernetwork mechanism relevant at scale.
- **§6 Related work.** Role of multiple heads (prunability → reinterpreted as module collapse); compositional generalization + scale; Raven-based tasks.
- **§7 Discussion.** Granularity of hypernetwork parameterization; GNN connection (message = hypernetwork, aggregation = sum); pooling operators as design axis. Limitation: from-scratch only.

---

## 13. Schlag et al. 2021 — Linear Transformers Are Secretly Fast Weight Programmers

**PDF:** `docs/project/references/in_context_learning/sources/Schlag et al. 2021 - Linear Transformers Are Secretly Fast Weight Programmers.pdf`
**Venue:** ICML 2021. **Authors:** Imanol Schlag*, Kazuki Irie*, Jürgen Schmidhuber (IDSIA).

### Phase 1 — Foundational Overview (undergraduate level)

**The question in plain terms.** Standard transformers attend over the whole sequence, which costs time quadratic in sequence length and memory that grows with length. "Linear transformers" replace the softmax with a kernel trick so cost is linear and memory is a *fixed-size matrix*. This paper's headline: that fixed-size matrix is **not new** — it is exactly the "fast weights" idea Schmidhuber introduced in 1991. A slow network (the trained weights) learns to write key→value associations into a fast-changing weight matrix via **outer products**, and reads them back by matrix-vector multiply. So a linear transformer is "secretly" a **Fast Weight Programmer (FWP)**.

**Analogy.** Think of the fast-weight matrix as a whiteboard. Each new token writes an association (key, value) onto the board by adding an outer product $\mathbf v\otimes\mathbf k$. When a query arrives, you multiply the board by the query to read back the value whose key best matches. The trained ("slow") network only learns *how to generate* the keys, values, and queries — the whiteboard content is computed on the fly, in-context.

**Key findings.**
1. **Formal equivalence.** Linear self-attention (softmax removed / kernel-linearized) is algebraically identical, up to normalization, to the 1991–93 outer-product FWP. The accumulated attention matrix $\mathbf W^{(i)}=\sum_j \mathbf v^{(j)}\otimes\phi(\mathbf k^{(j)})$ is the fast-weight memory.
2. **Capacity limit.** Because associations are summed into a finite $d_{\text{dot}}\times d_{\text{value}}$ matrix, you can only store ~$d_{\text{dot}}$ mutually-retrievable associations before keys stop being orthogonal and retrieval returns interfering mixtures. Once sequence length exceeds $d_{\text{dot}}$ the model is in an **overcapacity regime**.
3. **The Delta rule fix.** The pure additive update can't *erase* stale associations. They replace it with a **delta-rule** update: before writing a new value, read out the value currently stored under that key, and write only the *correction*, scaled by a learned, dynamic "write strength" $\beta^{(i)}$. This is the **DeltaNet** — a linear transformer that can edit its own memory.
4. **DPFP kernel.** A new deterministic, parameter-free feature map $\phi$ that projects keys/queries up to a higher-dimensional, near-orthogonal space (raising capacity) without random sampling (unlike Performer/FAVOR+).
5. **Empirics.** Synthetic retrieval confirms the capacity cliff at $d_{\text{dot}}$; DeltaNet beats the sum-rule on memory-editing tasks, WMT14 translation, and WikiText-103 language modeling, and — with fast-weight carry-over across segments — runs unbounded context with a tiny fixed state.

**Initial takeaway.** This paper is the historical + mathematical anchor of the whole "context as weights" corpus. It shows the transformer's context can *literally be a weight matrix built by outer products* — the exact algebra a hypernetwork or FiLM layer performs, but accumulated over the sequence. **Bridge to the project:** the outer-product write $\mathbf v\otimes\mathbf k$ and the delta-rule editable-memory view are the raw material behind "attention = hypernetwork" (Schug, §12, over the head index instead of the key index) and behind the neuromodulatory / fast-plasticity framings (fast weights ≈ synaptic modulation, explicitly cited to von der Malsburg 1981).

### Phase 2 — Graduate-Level Deep Dive

#### 2.1 The 1991 Fast Weight Programmer (Eqs. 1–3)

For a sequential input $\{\mathbf x^{(i)}\}_{i=1}^L$, $\mathbf x^{(i)}\in\mathbb R^{d_{\text{in}}}$, the slow net produces two activation patterns and writes their outer product into a fast-weight matrix:

$$
\mathbf a^{(i)},\mathbf b^{(i)} = \boldsymbol W_a\mathbf x^{(i)},\ \boldsymbol W_b\mathbf x^{(i)}, \tag{1}
$$
$$
\mathbf W^{(i)} = \sigma\!\left(\mathbf W^{(i-1)} + \mathbf a^{(i)}\otimes\mathbf b^{(i)}\right), \tag{2}
$$
$$
\mathbf y^{(i)} = \mathbf W^{(i)}\mathbf x^{(i)}. \tag{3}
$$

$\boldsymbol W_a,\boldsymbol W_b$ are trainable **slow** weights; $\mathbf W^{(i)}$ is the input-dependent **fast** weight (a key-value associative memory). Write = summation of an outer product (Eq. 2); retrieval = matrix-vector product (Eq. 3). In today's terminology $\mathbf a,\mathbf b$ are *values* and *keys* — "self-invented activation patterns."

#### 2.2 Self-attention without softmax IS an FWP (Eqs. 4–11)

Auto-regressive self-attention projects each input into key/value/query and attends causally:

$$
\mathbf k^{(i)},\mathbf v^{(i)},\mathbf q^{(i)} = \boldsymbol W_k\mathbf x^{(i)},\ \boldsymbol W_v\mathbf x^{(i)},\ \boldsymbol W_q\mathbf x^{(i)}, \tag{4}
$$
$$
\mathbf K^{(i)}=[\mathbf K^{(i-1)},\mathbf k^{(i)}],\quad \mathbf V^{(i)}=[\mathbf V^{(i-1)},\mathbf v^{(i)}],\quad \mathbf y^{(i)}=\mathbf V^{(i)}\,\text{softmax}\big((\mathbf K^{(i)})^\top\mathbf q^{(i)}\big). \tag{5–7}
$$

**Remove the softmax** and use associativity of matrix multiplication:

$$
\mathbf y^{(i)} = \mathbf V^{(i)}(\mathbf K^{(i)})^\top\mathbf q^{(i)}
= \big(\mathbf V^{(i)}(\mathbf K^{(i)})^\top\big)\mathbf q^{(i)}
= \Bigg(\sum_{j=1}^{i}\mathbf v^{(j)}\otimes\mathbf k^{(j)}\Bigg)\mathbf q^{(i)}. \tag{8}
$$

The crucial step is the middle equality: instead of first forming the $i\times 1$ attention score vector $(\mathbf K^{(i)})^\top\mathbf q^{(i)}$ (cost grows with $i$), you first form the *fixed-size* $d_{\text{value}}\times d_{\text{key}}$ matrix $\mathbf V^{(i)}(\mathbf K^{(i)})^\top=\sum_j \mathbf v^{(j)}\otimes\mathbf k^{(j)}$. Define

$$
\mathbf W^{(i)} = \sum_{j=1}^{i}\mathbf v^{(j)}\otimes\mathbf k^{(j)}, \tag{9}
$$

which admits the recurrent, constant-memory form matching Eqs. 1–3 with $\sigma=\text{Id}$:

$$
\mathbf W^{(i)} = \mathbf W^{(i-1)} + \mathbf v^{(i)}\otimes\mathbf k^{(i)}, \qquad \mathbf y^{(i)}=\mathbf W^{(i)}\mathbf q^{(i)}. \tag{10–11}
$$

So the query plays the role of the FWP retrieval input, and the causal attention output is a read from an outer-product memory. **This is the paper's core equivalence.**

#### 2.3 Kernel linearization of softmax (Eqs. 12–19)

Write softmax attention explicitly with kernel $\kappa(\mathbf k,\mathbf q)=\exp(\mathbf k\cdot\mathbf q)$:

$$
\mathbf y^{(i)} = \sum_{j=1}^{i}\frac{\mathbf v^{(j)}\,\kappa(\mathbf k^{(j)},\mathbf q^{(i)})}{\sum_{j'=1}^{i}\kappa(\mathbf k^{(j')},\mathbf q^{(i)})}. \tag{12}
$$

Replace the exponential kernel by a **factorizable** one $\kappa'(\mathbf k,\mathbf q)=\phi(\mathbf k)^\top\phi(\mathbf q)$ with feature map $\phi:\mathbb R^{d_{\text{key}}}\to\mathbb R^{d_{\text{dot}}}$. Then substituting and using linearity to pull $\phi(\mathbf q^{(i)})$ out of the sum:

$$
\mathbf y^{(i)} = \frac{\sum_{j=1}^{i}\mathbf v^{(j)}\big(\phi(\mathbf k^{(j)})^\top\phi(\mathbf q^{(i)})\big)}{\sum_{j'=1}^{i}\phi(\mathbf k^{(j')})^\top\phi(\mathbf q^{(i)})}
= \frac{\Big(\sum_{j}\mathbf v^{(j)}\otimes\phi(\mathbf k^{(j)})\Big)\phi(\mathbf q^{(i)})}{\Big(\sum_{j'}\phi(\mathbf k^{(j')})\Big)^\top\phi(\mathbf q^{(i)})}. \tag{13–14}
$$

Introduce the fast-weight matrix and the denominator accumulator:

$$
\mathbf W^{(i)}=\sum_{j=1}^{i}\mathbf v^{(j)}\otimes\phi(\mathbf k^{(j)}),\qquad
\mathbf z^{(i)}=\sum_{j=1}^{i}\phi(\mathbf k^{(j)}), \tag{15–16}
$$

giving the recurrent linear-transformer forward pass (Katharopoulos et al. 2020):

$$
\mathbf W^{(i)}=\mathbf W^{(i-1)}+\mathbf v^{(i)}\otimes\phi(\mathbf k^{(i)}),\quad
\mathbf z^{(i)}=\mathbf z^{(i-1)}+\phi(\mathbf k^{(i)}),\quad
\mathbf y^{(i)}=\frac{\mathbf W^{(i)}\phi(\mathbf q^{(i)})}{\mathbf z^{(i)}\cdot\phi(\mathbf q^{(i)})}. \tag{17–19}
$$

This is exactly an FWP (§2.1) with normalization. Hence: **the core of every linear transformer is an outer-product Fast Weight Programmer.**

#### 2.4 Capacity limitation via tensor-product representations

Retrieval reads $\mathbf y^{(i)}=\mathbf W^{(i)}\phi(\mathbf q^{(i)})=\sum_j\mathbf v^{(j)}\big(\phi(\mathbf k^{(j)})\cdot\phi(\mathbf q^{(i)})\big)$. To recover $\mathbf v^{(m)}$ cleanly when querying with $\mathbf k^{(m)}$, we need $\phi(\mathbf k^{(j)})\cdot\phi(\mathbf k^{(m)})\approx\delta_{jm}$ — i.e. the projected keys must be near-orthogonal. In a $d_{\text{dot}}$-dimensional space there are at most $d_{\text{dot}}$ mutually orthogonal vectors, so storing more than $d_{\text{dot}}$ associations forces **crosstalk** and retrieval error. This is precisely Smolensky's (1990) tensor-product-representation (TPR) analysis: the fast-weight matrix is a second-order TPR of keys (roles) and values (fillers), and Smolensky's Theorems 3.1/3.3 formalize the crosstalk. **Difference from classic TPRs:** classic TPRs are hand-wired with a-priori symbolic structure; the FWP *learns* all the role/filler (key/value) vectors end-to-end. Consequence for transformers: when $L>d_{\text{dot}}$ the model is in overcapacity and the naive additive rule degrades.

#### 2.5 The delta-rule update (Eqs. 20–25) — the DeltaNet

To operate in overcapacity, the memory must *edit*, not just accumulate. Given a new $(\mathbf k^{(i)},\mathbf v^{(i)})$:

$$
\bar{\mathbf v}^{(i)} = \mathbf W^{(i-1)}\phi(\mathbf k^{(i)}) \quad\text{(read current value at this key)}, \tag{20}
$$
$$
\beta^{(i)} = \sigma\big(\boldsymbol W_\beta\mathbf x^{(i)}\big)\in[0,1] \quad\text{(learned dynamic write-strength)}, \tag{21}
$$
$$
\mathbf v^{(i)}_{\text{new}} = \beta^{(i)}\mathbf v^{(i)} + (1-\beta^{(i)})\bar{\mathbf v}^{(i)} \quad\text{(convex combination)}. \tag{22}
$$

The update writes the new value and **removes** the old one at that key:

$$
\mathbf W^{(i)} = \mathbf W^{(i-1)} + \underbrace{\mathbf v^{(i)}_{\text{new}}\otimes\phi(\mathbf k^{(i)})}_{\text{write}} - \underbrace{\bar{\mathbf v}^{(i)}\otimes\phi(\mathbf k^{(i)})}_{\text{remove}}. \tag{23}
$$

Substituting Eq. 22 for $\mathbf v^{(i)}_{\text{new}}$ and simplifying the write−remove difference:

$$
\mathbf v^{(i)}_{\text{new}} - \bar{\mathbf v}^{(i)} = \beta^{(i)}\mathbf v^{(i)}+(1-\beta^{(i)})\bar{\mathbf v}^{(i)} - \bar{\mathbf v}^{(i)} = \beta^{(i)}\big(\mathbf v^{(i)}-\bar{\mathbf v}^{(i)}\big),
$$
$$
\boxed{\;\mathbf W^{(i)} = \mathbf W^{(i-1)} + \beta^{(i)}\big(\mathbf v^{(i)}-\bar{\mathbf v}^{(i)}\big)\otimes\phi(\mathbf k^{(i)})\;}, \qquad \mathbf y^{(i)}=\mathbf W^{(i)}\phi(\mathbf q^{(i)}). \tag{24–25}
$$

Equation 24 is exactly the **Widrow–Hoff delta rule** $\Delta\mathbf W\propto(\text{target}-\text{prediction})\otimes\text{input}$, with $\beta^{(i)}$ a *self-invented, input-dependent learning rate*. The FWP therefore learns to correct its own key→value map online. (The pure sum rule of Eq. 17 is the special case $\bar{\mathbf v}^{(i)}\equiv 0$, $\beta^{(i)}\equiv1$.) The paper connects this to a broader theme (elaborated further in Mittal et al., §14): the in-context state update is itself a learned optimization step.

#### 2.6 Normalization variants

- **Attention normalization** (Eqs. 26–28): keep the accumulator $\mathbf z^{(i)}=\mathbf z^{(i-1)}+\phi(\mathbf k^{(i)})$ and normalize both the read-out $\bar{\mathbf v}^{(i)}$ and output $\mathbf y^{(i)}$ by $\mathbf z^{(i)}\cdot\phi(\cdot)$. Drawbacks: $\mathbf z^{(i)}$ grows unboundedly (instability), and it does not balance write vs. remove for the delta rule.
- **Sum normalization** (Eq. 29): instead normalize each feature vector by its component sum, $\phi'(\mathbf q^{(i)})=\phi(\mathbf q^{(i)})/\sum_{j=1}^{d_{\text{dot}}}\phi(\mathbf q^{(i)})_j$, applied to both keys and queries before Eqs. 20–25. Intuition: if a vector's components sum to 1, then $\mathbf W\phi'$ is a convex "attention over the columns of $\mathbf W$." Empirically sum normalization alone suffices; extra attention normalization *hurts* language-modeling perplexity, and for unbounded-context DeltaNet the attention accumulator must be removed (it blows up).

#### 2.7 DPFP feature map (Eqs. 33–37)

Two failure modes motivate a new $\phi$: Katharopoulos' $\phi=\text{ELU}(x)+1$ keeps $d_{\text{dot}}=d_{\text{key}}$ (no capacity gain); Performer/FAVOR+ raises capacity but via *random* features (sampling variance). The **Deterministic Parameter-Free Projection (DPFP)** raises $d_{\text{dot}}$ deterministically by taking products of rectified coordinate pairs so that inputs falling in different orthants map to orthogonal supports. In $\mathbb R^2\to\mathbb R^4$ with $r(a)=\max(0,a)$:

$$
\phi_1(\mathbf k)=r(k_1)r(k_2),\ \phi_2(\mathbf k)=r(-k_1)r(k_2),\ \phi_3(\mathbf k)=r(k_1)r(-k_2),\ \phi_4(\mathbf k)=r(-k_1)r(-k_2). \tag{33–36}
$$

Each input has exactly one nonzero component ⇒ inputs in different quadrants are orthogonal in the projected space. General form for $\mathbf k\in\mathbb R^{d_{\text{key}}}$, index $i\in[1,2d_{\text{key}}]$, capacity knob $\nu\in\{1,\dots,2d_{\text{key}}-1\}$:

$$
\phi_{i\nu}(\mathbf k)=r\big(\big[\begin{smallmatrix}\mathbf k\\-\mathbf k\end{smallmatrix}\big]\big)_i\; r\big(\big[\begin{smallmatrix}\mathbf k\\-\mathbf k\end{smallmatrix}\big]\big)_{i+\nu}, \tag{37}
$$

giving codomain dimension $d_{\text{dot}}=2d_{\text{key}}\nu$. It is embarrassingly parallel and needs no learned parameters or random sampling; its sparsity/orthogonality goal aligns with the empirical observation that ReLU beats exp in FAVOR+.

#### 2.8 Empirical highlights

- **Synthetic retrieval (Setting 1, capacity):** with $d_{\text{key}}=64$, plain Linear-Attention starts erroring at ~60 associations; DPFP-$\nu$ errors at its $d_{\text{dot}}=128/256/384$ limits; FAVOR+ never reaches zero loss; softmax is best (infinite capacity) but struggles past 500 keys. Directly validates the $\le d_{\text{dot}}$ capacity theory.
- **Setting 2 (editing):** keys re-assigned (sampled with replacement, $L=2S$); the delta rule beats the sum rule and prior FWP variants — memory editing works.
- **WMT14 En-De:** DPFP beats Linear Transformer and Performer at small $d_{\text{dot}}$ (good simplicity/performance trade-off).
- **WikiText-103:** DeltaNet (delta rule) beats sum-rule linear transformer and Performer at matched parameters; sum normalization required, extra attention normalization and absolute positional encoding both unnecessary/harmful. **Unbounded context** (Table 4): carrying the fast-weight memory across segments, DeltaNet keeps perplexity ~27–29 with a tiny 0.13M state, while the naive sum-rule linear transformer diverges (>260) — a striking demonstration of editable finite memory.

### Appendix: Section-by-Section Backbone

- **Abstract.** Formal equivalence of linearized self-attention and early-'90s fast-weight controllers; slow net learns by GD to program fast weights via additive outer products of self-invented keys/values. Infer a capacity limit; replace additive rule with a delta-rule instruction (dynamic learning rate); propose a new linearization kernel (DPFP). Experiments on synthetic retrieval, WMT14 MT, WikiText-103 LM.
- **§1 Introduction.** Transformer self-attention is quadratic time / linear memory. Linear transformers linearize softmax → constant memory, linear time. Claim: this family ≡ FWPs (Schmidhuber 1991/92/93) up to normalization. Derive capacity limit + overcapacity regime; introduce delta-rule update + dynamic learning rate; propose new φ.
- **§2 Background on FWPs.** Fast weights = input-dependent, variable weights (von der Malsburg synaptic modulation; Hinton & Plaut; Feldman dynamic connections). 1991 two-net system: slow net programs fast net via outer-product instructions (Eqs. 1–3); key-value associative memory; TPR connection (Smolensky 1990); modern revivals: hypernetworks, dynamic plasticity, dynamic convolution, lambda nets, meta-learning.
- **§3 Relation to Transformers.** §3.1 remove softmax → self-attention is an FWP (Eqs. 4–11); $\mathbf W^{(i)}=\sum_j\mathbf v^{(j)}\otimes\mathbf k^{(j)}$. §3.2 kernel linearization (Eqs. 12–19); fast weight $\mathbf W^{(i)}$ + accumulator $\mathbf z^{(i)}$; core of linear transformers = outer-product FWP.
- **§4 Analysing & improving.** §4.1 capacity: $\le d_{\text{dot}}$ orthogonal keys; overcapacity when $L>d_{\text{dot}}$; TPR theory (Smolensky Thm 3.1/3.3); FWPs learn the vectors (vs. hand-wired TPRs). §4.2 delta-rule update (Eqs. 20–25): read-modify-write with dynamic write-strength $\beta^{(i)}$; = Widrow-Hoff delta rule with learned learning rate; normalization: attention-norm (Eqs. 26–28, drawbacks) vs. sum-norm (Eq. 29, preferred).
- **§5 Linear attention functions.** §5.1 desirable φ properties (positive codomain, capacity via $d_{\text{dot}}$). §5.2 Katharopoulos ELU+1 (Eq. 30, no capacity gain). §5.3 FAVOR+ random features (Eqs. 31–32, sampling variance, $d_{\text{dot}}=2m$). §5.4 DPFP (Eqs. 33–37): deterministic, parameter-free, near-orthogonal up-projection, $d_{\text{dot}}=2d_{\text{key}}\nu$.
- **§6 Experiments.** §6.1.1 capacity confirmed on synthetic retrieval; §6.1.2 delta rule beats sum rule on editing; §6.2 WMT14 (DPFP competitive at small $d_{\text{dot}}$); §6.3 WikiText-103 (DeltaNet best; sum-norm needed; no pos-enc/attn-norm; unbounded-context DeltaNet vs. Transformer-XL, tiny state size).
- **§7 Conclusion.** Linear attention ≡ 1991 FWP; capacity analysis; delta-rule editable memory with learned learning rate; new linearization kernel; opens design space for finite-memory transformers.

---

## 14. Mittal et al. 2025 — Iterative Amortized Inference: Unifying ICL and Learned Optimizers

**PDF:** `docs/project/references/in_context_learning/sources/Mittal et al. 2025 - Iterative Amortized Inference - Unifying ICL and Learned Optimizers.pdf`
**Venue:** arXiv (Oct 2025). **Authors:** Sarthak Mittal, Divyat Mahajan, Guillaume Lajoie, Mohammad Pezeshki (Mila / Université de Montréal / FAIR at Meta).

### Phase 1 — Foundational Overview (undergraduate level)

**The question in plain terms.** Many modern methods "learn to solve new tasks fast": meta-learning (MAML), in-context learning (ICL), prompt tuning, hypernetworks, and learned optimizers. They *look* different but share a goal — reuse structure across tasks so a new task needs only a little task-specific adaptation. This paper gives **one equation that contains all of them** and a **taxonomy** with three regimes, then proposes a scalable extension.

**The running example.** Model the motion of an object on many planets. Newton's laws are shared across all planets (amortize once); only the gravitational constant differs per planet (adapt per task). Every method is just a choice of *what you amortize* (the shared bit) versus *how you read the task data at inference* (the adapt bit).

**Key findings.**
1. **Unified formula.** Every method is $f_\gamma\big(x, g_\phi(\mathcal D_T)\big)$: a **shared predictor** $f_\gamma$ (task-invariant) applied to a query $x$, using a **task-adaptation function** $g_\phi$ that maps the task's support data $\mathcal D_T$ into task-specific information $\theta_T$ (weights, prompts, latents, or context tokens).
2. **Three regimes** (Table 1):
   - **Parametric** — $f$ is fixed (a known likelihood form), $g_\phi$ learns to infer its parameters. Hypernetworks and learned optimizers live here. Task info is *externalized* into explicit parameters.
   - **Implicit** — $g$ is trivial (identity/subsample), $f_\gamma$ is a transformer that jointly ingests context + query. **This is ICL / PFNs.** Task adaptation is *internalized* in the forward pass.
   - **Explicit** — both $f_\gamma$ and $g_\phi$ learned: $g_\phi$ compresses the whole dataset into a low-dim embedding (the "gravitational constant"), $f_\gamma$ is the learned likelihood ("laws of physics"). Neural processes live here.
3. **The limitation.** Implicit/ICL is capped by context length; parametric/explicit pooling throws away fine detail. Neither scales to *large* task datasets.
4. **The fix — iterative amortized inference.** Borrow SGD's trick: don't cram the whole dataset into one forward pass; instead **refine a solution step-by-step over mini-batches**. A learned sequence model $h_\phi$ takes the current state + a fresh mini-batch and outputs a better state. This unifies learned optimizers (which refine over gradient mini-batches) with ICL/hypernetworks (which refine over observation mini-batches), and lets either or both signals drive the update.
5. **Empirics.** More refinement steps consistently help across regression, MNIST/FMNIST/ImageNet classification (incl. OOD → CIFAR), topological-order prediction, and generative modeling (GMM, alphabets). Using **observations directly often beats gradients-only** at low dimension; gradients help more at high dimension / few observations. Parametric beats explicit (joint learning is hard). For implicit, keeping the recurrent state at the **logits** (pre-softmax) works best.

**Initial takeaway.** This is the *organizing map* of the whole shard. It places "attention/ICL" (implicit), "hypernetworks" (parametric), and "neural processes" (explicit) as three faces of one $f_\gamma(x, g_\phi(\mathcal D_T))$ template, and shows the in-context adaptation is best understood as **iterative optimization**. **Bridge to the project:** it tells you exactly where a FiLM/hypernetwork conditioning module sits (parametric amortization: a learned $g_\phi$ generating the conditioning parameters $\theta_T$) versus where in-context conditioning sits (implicit), and frames both as instances of amortized inference — useful when deciding whether the project's conditioning should externalize a task code (interpretable, low-dim) or internalize it in a forward pass (expressive, expensive).

### Phase 2 — Graduate-Level Deep Dive

#### 3.1 The single-task baseline and its two failures

Standard learning minimizes true risk, approximated by empirical risk over an i.i.d. dataset $\mathcal D_T$:

$$
\theta_T^* = \arg\min_\theta\ \mathbb E_{x,y\sim p_T}\big[\mathcal L(y, f(x,\theta))\big],\qquad
\hat\theta_T = \arg\min_\theta\ \sum_{(x,y)\in\mathcal D_T}\mathcal L(y,f(x,\theta)). \tag{1–2}
$$

Two failures: (a) no systematic transfer to a *new* task; (b) it ignores cross-task shared structure — data from task $T_2$ carries inductive signal for $T_1$ but is discarded.

#### 3.2 Meta-learning, hypernetworks, learned optimizers, PFNs as one family

**MAML** learns a global initialization $\hat\theta_0$ then adapts via an inner optimizer $g_{\theta_0}$:

$$
\hat\theta_0 = \arg\min_{\theta_0}\ \mathbb E_T\,\mathbb E_{x,y,\mathcal D_T}\big[\mathcal L(y,f(x,\theta_T))\big],\quad \theta_T = g_{\theta_0}(\mathcal D_T). \tag{3}
$$

**Hypernetworks / learned optimizers** replace $\theta_0$ by learnable $\phi$: $g_\phi$ is a sequence model producing task parameters — from *observations* (hypernetwork) or from *gradients* (learned optimizer). **PFNs / example-based ICL** instead directly model the conditional predictive distribution without exposing $\theta_T$:

$$
\hat\gamma = \arg\min_\gamma\ \mathbb E_T\,\mathbb E_{x,y,\mathcal D_T}\big[\mathcal L(y, f_\gamma(x,\mathcal D_T))\big]. \tag{4}
$$

**The unification.** All of the above are

$$
\boxed{\;\min_{\gamma,\phi}\ \mathbb E_T\,\mathbb E_{x,y,\mathcal D_T}\Big[\mathcal L\big(y,\ f_\gamma\big(x,\ g_\phi(\mathcal D_T)\big)\big)\Big]\;} \tag{5}
$$

with the empirical support/query split (support $\mathcal D_T^{\text{train}}$ feeds $g_\phi$; loss on $\mathcal D_T^{\text{valid}}$):

$$
\min_{\gamma,\phi}\ \mathbb E_T\Bigg[\sum_{(x,y)\in\mathcal D_T^{\text{valid}}}\mathcal L\big(y, f_\gamma(x,\theta_T)\big)\Bigg],\quad \theta_T=g_\phi(\mathcal D_T^{\text{train}}). \tag{6}
$$

Special cases by choice of $(f,g,\text{split of }\phi\text{ vs }\gamma)$:
- **Supervised learning:** $f$ fixed, $g=$ SGD (solves Eq. 2).
- **MAML:** $f$ fixed, $g_\phi=$ SGD with learnable init $\theta_0\in\phi$.
- **Learned optimizers / hypernetworks:** $f$ fixed, $g_\phi$ a neural net. Learned optimizer is the *iterative* form $g_\phi := h_\phi^{(k)}\circ\cdots\circ h_\phi^{(1)}$, where each $h_\phi^{(i)}$ is a sequence model taking past parameters $\theta^{(0)},\dots,\theta^{(i-1)}$ and their gradients $\nabla^{(j)}=\nabla_\theta\sum_{(x,y)\in B^{(j)}}\mathcal L(y,f(x,\theta))\big|_{\theta^{(j)}}$; hypernetwork is the one-shot observation-to-parameter map.
- **ICL:** $g$ samples observations, $f_\gamma$ is a predictive sequence model with the context as observations (prompt-based ICL is the emergent LLM special case).

**Non-unique decomposition (important subtlety).** $f_\gamma(x, g_\phi(\mathcal D_T))$ does not uniquely split computation between $f$ and $g$: anything $g_\phi$ computes can be folded into $f_\gamma$. The *converse fails* because $g_\phi$ has no access to the query $x$. The convention that makes the split meaningful: **assign all query-independent computation to $g_\phi$, and query-dependent computation to $f_\gamma$.** So $f_\gamma$ sees task info $\mathcal D_T$ only through the structure $g_\phi$ imposes (pooled representation, gradients, or natural language).

#### 3.3 The taxonomy (parametric / implicit / explicit)

- **Parametric** ($f$ fixed, $g_\phi$ learnable): likelihood form is *known* (a simulator, or a categorical likelihood with a linear map); $g_\phi$ is the inference procedure discovering optimal parameters (e.g. linear coefficients or a posterior over them). Advantages: can use **gradient information** in addition to $\mathcal D_T$; yields a low-dim, interpretable task representation.
- **Implicit** ($f_\gamma$ trainable, $g$ fixed): a single model ingests query + observations and learns the output form directly (ICL, some PFNs/CNPs). $g$ is identity or subsampling. Hard to localize which activations are task vs. prediction; under assumptions the forward pass recovers algorithmic behavior (gradient descent — von Oswald 2023; causal discovery — Nichani 2024), but this is the posterior-predictive view.
- **Explicit** (both $f_\gamma,g_\phi$ trainable): $g_\phi$ gives a low-dim dataset embedding ("gravity constant"), $f_\gamma$ the shared likelihood ("physics"). Neural processes. Gets dimensionality reduction + gradient use, but pays **non-stationarity** — $f_\gamma$ and $g_\phi$ are coupled and co-adapting.

#### 3.4 Iterative amortized inference (Eqs. 7–9)

Motivation: implicit/ICL is context-length bounded; parametric/explicit compress via pooling or gradients (lossy). SGD scales by iterating over mini-batches. So make $g_\phi$ (or $f_\gamma$) an iterative, Markovian, greedy refinement.

**Parametric / explicit — refine the state $\theta$.** With subsampled batches $B^{(i)}_{\text{train}}\subset\mathcal D_T^{\text{train}}$ and learned init $\theta^{(0)}$:

$$
\theta^{(0)}\ \xrightarrow{\ h_\phi(\cdot,\,B^{(0)}_{\text{train}})\ }\ \theta^{(1)}\ \xrightarrow{\ h_\phi(\cdot,\,B^{(1)}_{\text{train}})\ }\ \cdots\ \xrightarrow{\ h_\phi(\cdot,\,B^{(k-1)}_{\text{train}})\ }\ \theta^{(k)}=:\theta_T. \tag{7}
$$

Learned optimizers are the restricted case where $h_\phi$ sees observations *only through gradients*: $h_\phi(\theta,B):=\text{Transformer}_\phi\big(\theta,\ \nabla_\theta\sum_{(x,y)\in B}\mathcal L(y,f_\gamma(x,\theta))\big)$. Because the observation→gradient map is **non-invertible**, gradient-only optimizers cannot recover general schemes (e.g. the closed-form linear-regression solution) that using raw observations can. Here $\theta$ acts as a **recurrent memory** compressing the task; prediction for $x$ given $\theta_T$ is conditionally independent of $\mathcal D_T$.

**Implicit — refine the prediction $\hat y$.** No task parameters are exposed; the recurrent state is tied to the query. $g_\phi$ just subsamples, $f_\gamma=r_\gamma$ a transformer:

$$
\hat y^{(0)}\ \xrightarrow{\ r_\gamma([x,\hat y^{(0)}],\,B^{(0)}_{\text{train}})\ }\ \hat y^{(1)}\ \xrightarrow{\ }\ \cdots\ \xrightarrow{\ r_\gamma([x,\hat y^{(k-1)}],\,B^{(k-1)}_{\text{train}})\ }\ \hat y^{(k)}. \tag{8}
$$

$\hat y^{(0)}$ is a learned init (ideally the marginal/unconditional prediction). Since $\theta_T$ is never exposed, gradient signal is hard to inject.

**Greedy single-step training (Eq. 9).** Following SGD's greediness, train $\phi,\gamma$ on single-step improvements only, stopping gradients from flowing back through prior states $\theta^{(i)}$ or $\hat y^{(i)}$:

$$
\theta^{(i)}\ \xrightarrow{h_\phi(\cdot,B)}\ \theta^{(i+1)}\to\mathcal L\big(y,f_\gamma(x,\theta^{(i+1)})\big)
\quad\text{or}\quad
\hat y^{(i)}\ \xrightarrow{r_\gamma([x,\cdot],B)}\ \hat y^{(i+1)}\to\mathcal L\big(y,\hat y^{(i+1)}\big). \tag{9}
$$

An outer loop produces the $i$-th states; $(x,y)\in\mathcal D_T^{\text{valid}}$ while $B\subseteq\mathcal D_T^{\text{train}}$ (amortizing a learner to minimize *validation* loss). All three paradigms modify causal masking for efficient parallel computation over variable-size mini-batches.

**Optional non-Markovian variant.** Relax to a history window: $h_\phi:\theta_{t-k:t},\nabla_{t-k:t},B_t\to\theta_{t+1}$. Empirically extra history barely helps (except FMNIST pretraining).

#### 3.5 Connection to the rest of the shard

This paper is the abstraction layer over Schlag (paper 2) and Schug (paper 1): Schlag's delta-rule fast-weight update $\mathbf W^{(i)}=\mathbf W^{(i-1)}+\beta^{(i)}(\mathbf v^{(i)}-\bar{\mathbf v}^{(i)})\otimes\phi(\mathbf k^{(i)})$ is precisely a *single greedy refinement step* $h_\phi(\theta^{(i)},B^{(i)})$ in the parametric regime, where the "state" $\theta$ is the fast-weight matrix and $\beta$ is a learned learning rate. Schug's per-token hypernetwork code $\theta_T=g_\phi(\mathcal D_T)$ is a *one-step parametric* amortization (generate weights from context). Chen (paper 4) is the extreme parametric endpoint: convert the implicit ICL adaptation into literal explicit weights.

#### 3.6 Empirical findings (Tables 2–6, §5)

- **All three regimes:** iterative refinement (more steps) monotonically improves ID and OOD performance across linear regression, MNIST/FMNIST/ImageNet classification, topological order, and generative (GMM/alphabet) tasks. E.g. pretrain on ImageNet → good CIFAR-10 from in-context examples alone.
- **Observations vs. gradients:** at low dimension, raw observations beat gradient-only (learned-optimizer) inference; at high dimension / few observations, gradients give a more reliable search signal. Combining both often best.
- **Parametric > explicit** even though explicit's solution class *contains* parametric's — attributed to the coupled, non-stationary joint optimization of $f_\gamma,g_\phi$.
- **Implicit state choice:** keep the recurrent state at the **logits** (pre-softmax), not the pre-MLP latents or post-softmax probs — being close to prediction helps greedy refinement.
- **Runtime:** a $K$-step method with mini-batch $B$ costs $O(KB^2)$; a one-step model over the same $KB$ data costs $O((KB)^2)$, so iterative amortization is $\sim K\times$ more efficient (and more memory-scalable) at equal data.

### Appendix: Section-by-Section Backbone

- **Abstract.** Amortized learning (meta-learning, ICL, prompt tuning, learned optimizers) reuses cross-task computation. Unified framework: methods differ in *what they amortize* (initializations / learned updates / predictive mappings) and *how they use task data at inference*. Taxonomy: parametric / implicit / explicit. Limitation: poor scaling to large datasets (context length / lossy pooling). Proposal: iterative amortized inference — step-by-step refinement over mini-batches (SGD-inspired), bridging optimization-based meta-learning and forward-pass amortization.
- **§1 Introduction.** Planet/gravity running example. Meta-learners (learned init), LLM ICL (conditioning on observations/instructions), learned optimizers (amortize optimization via gradient-conditioned updates). Two components: (1) shared task-invariant mechanism, (2) task-adaptation function. Three regimes by inductive bias. Contributions: unified framework (§2), taxonomy (§3), stochastic iterative amortization (§4).
- **§2 Unified perspective.** Eqs. 1–2 (true/empirical risk); Eq. 3 (MAML / hypernetwork / learned optimizer as $\theta_T=g_{\theta_0/\phi}(\mathcal D_T)$); Eq. 4 (PFN/ICL conditional predictive); Eq. 5–6 (unified $f_\gamma(x,g_\phi(\mathcal D_T))$). §2.1 special cases (supervised/MAML/learned-opt/hypernet/ICL). Non-unique decomposition (query-independent → $g_\phi$; query-dependent → $f_\gamma$). "Learning the learner": higher-order gradients, first-order approx, BPTT, RL, ES.
- **§3 Taxonomy.** Parametric (fixed $f$, learnable $g_\phi$; hypernets, learned optimizers, parametric-ICL; can use gradients; interpretable low-dim params). Implicit (trainable $f_\gamma$, fixed $g$; ICL/PFN/CNP; forward-pass conditioning; hard to localize task activations). Explicit (both trainable; dataset embedding + learned likelihood; neural processes; non-stationarity cost).
- **§4 Iterative amortized inference.** Motivation (context-length / pooling limits vs. SGD mini-batch scaling). Parametric/explicit refinement of $\theta$ (Eq. 7); learned-optimizer restriction (gradient-only, non-invertible obs→grad map); $\theta$ as recurrent task memory. Implicit refinement of $\hat y$ (Eq. 8), state tied to query. Greedy single-step training with stop-gradient (Eq. 9); validation-loss amortization; modified causal masking. Non-Markovian history variant.
- **§5 Experiments.** Tasks: linear regression, MNIST/FMNIST/ImageNet classification (+CIFAR OOD, Dino-v2 embeddings), topological order (SCM), MoG + alphabet generation (flow-matching). §5.1 consistent gains with steps across parametric (Table 2) / explicit (Table 3) / implicit (Tables 4–6). §5.2 analysis: gradients sub-optimal at low-dim; history barely helps; explicit < parametric (non-stationarity); best implicit state = logits; runtime $O(KB^2)$ vs $O((KB)^2)$ → $K\times$ efficiency.

---

## 15. Chen et al. 2024 — Exact Conversion of In-Context Learning to Model Weights

**PDF:** `docs/project/references/in_context_learning/sources/Chen et al. 2024 - Exact Conversion of In-Context Learning to Model Weights.pdf`
**Venue:** ICML 2024. **Authors:** Brian K Chen, Tianyang Hu, Hui Jin, Hwee Kuan Lee, Kenji Kawaguchi (NUS / A*STAR / Huawei Noah's Ark Lab).

### Phase 1 — Foundational Overview (undergraduate level)

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

### Phase 2 — Graduate-Level Deep Dive

#### 4.1 Attention definitions and the Key-Value matrix

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

#### 4.2 Why direct conversion fails (§4.1)

Naive attempt: for plain linear attention ($\phi=\text{Id}$, $D_1=D_2=1$), find weights $W_1,W_2$ absorbing the ICL contribution for all $X$:

$$
W_1^\top X^\top X W_2 = W_K^\top X^\top X W_V + \underbrace{W_K^\top X'^\top X' W_V}_{\text{constant ICL term}}. \tag{7}
$$

Vectorizing (Kronecker form):

$$
(W_2^\top\otimes W_1^\top)\,\text{vec}(X^\top X) = (W_V^\top\otimes W_K^\top)\,\text{vec}(X^\top X) + (W_V^\top\otimes W_K^\top)\,\text{vec}(X'^\top X'). \tag{8}
$$

Since $X=0$ is an admissible prompt, Eq. 8 at $X=0$ forces $(W_V^\top\otimes W_K^\top)\,\text{vec}(X'^\top X')=0$, which is false in general. (Appendix A shows the only escape is to restrict all inputs to a shifted subspace $X^\top X\in c+D$ with $c$ depending on $X$ — not on $X'$ — which cannot hold for all $X$; and even then $W_1,W_2$ are non-unique.) **Conclusion: you cannot fold the context into the existing multiplicative weights; you need free additive parameters.**

#### 4.3 Bias conversion for linear attention (Theorem 4.1)

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

#### 4.4 Assumptions for general linearized models (§4.3)

For a general linearized attention model the same trick works under:
- **A4.2 Autoregressive.** ICL tokens (positions $1..M$) must not be affected by input tokens; otherwise the "context contribution" would vary with the input, breaking a single-bias conversion.
- **A4.3 Relative positional encoding.** So that inserting ICL tokens does not shift the input tokens' encodings (absolute PE would require an extra correcting linear transform). RoFormer/Llama qualify.
- **A4.4 Separation of the normalizer.** $D_2([A;B]) - D_2([B]) = D_2^*([A])$, i.e. the denominator's context contribution depends only on $A$. Satisfied by kernelized attention ($D_2(X)=\sum_j\phi(K_j)$) and RetNet ($D_1=D_2=1$).

#### 4.5 Full conversion with RoPE (Theorem 4.5) — the core result

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

#### 4.6 ICLCA algorithm (§4.5)

**Algorithm 1 (ICLCA):** feed the ICL tokens $X'^{(0)}$ through the $L$ layers. For each **linearized attention** layer $l$: compute $Q',K',V'$ from $X'^{(l-1)}$; update $b'^{(l)}_{KV}=R^{d_K}_{\Theta,-M}b^{(l)}_{KV}+\sum_{j=1}^M R^{d_K}_{\Theta,j-M}\phi(K'^{(l-1)}_j)(V'^{(l-1)}_j)^\top$ and $b'^{(l)}_D=b^{(l)}_D+D_2^*(X'^{(l-1)})$; then propagate $X'^{(l)}$ forward (needed because deeper layers' ICL features depend on shallower conversions). All non-attention sublayers are element-wise, so ICL tokens pass through them independently. Save the biases. **Cost:** one forward pass over $M$ context tokens, no backprop.

#### 4.7 Approximate conversion for softmax transformers (§5)

Softmax attention is not linear in $A$, so conversion is only approximate: kernel-approximate $\text{sim}$ by $\phi$ (Performer PRF/ORF), apply Theorem 4.5 to the approximated model, and re-inject the bias into the original softmax layer:

$$
O_i^\top = \frac{\big[\sum_{j=1}^N\text{sim}(Q_i,K_j)V_j^\top\big] + R^{d_K}_{\Theta,i}\phi(Q_i)^\top b_{KV}}{\big[\sum_{j=1}^N\text{sim}(Q_i,K_j)\big] + R^{d_K}_{\Theta,i}\phi(Q_i)^\top b_D},\quad
b_{KV}=\sum_{j=1}^M R^{d_K}_{\Theta,j-M}\phi(K'_j)V_j'^\top,\ \ b_D=\sum_{j=1}^M\phi(K'_j). \tag{17–18}
$$

Accuracy is limited only by the kernel's softmax-approximation quality — and crucially the approximation is applied **only to the $M$ ICL tokens**, not all tokens, so it is strictly better than approximating the full softmax. Algorithm ICLAA (Appendix D.1).

#### 4.8 RNN view (§7)

Linear transformers are RNNs; Eq. 12 is the recurrence

$$
s_i = s_{i-1} + R^{d_K}_{\Theta,i}\phi(K_i)V_i^\top,\quad z_i = z_{i-1}+D_2^*(X_i),\quad y_i=\frac{R^{d_K}_{\Theta,i}\phi(Q_i)^\top s_i}{D_1(Q_i)^\top z_i}. \tag{19}
$$

Adding the biases is **exactly** re-initializing the hidden state:

$$
s_0 = b'_{KV}-b_{KV} = \sum_{j=1}^M R^{d_K}_{\Theta,j-M}\phi(K'_j)V_j'^\top,\qquad z_0 = b'_D-b_D = D_2(X').
$$

Because future tokens only see past hidden states, capturing the ICL prompt's hidden state suffices. This generalizes the method to non-transformer autoregressive RNNs (H3, Mamba) *if* they exhibit ICL — the context is stored in the recurrence's initial state.

#### 4.9 Empirical results (§6)

- **Exactness (Table 1):** relative logit error between $M_{\text{old,ICL}}$ and $M_{\text{new,no ICL}}$ over 100 random prompts: $2.9\times10^{-7}$ (205K params) → $4.3\times10^{-6}$ (1.98B). Error grows only ~2× per 10× params — scales gracefully.
- **Induction head (Table 2):** ICL model 99.95% accuracy; without ICL 2.23%; converted context-free model 99.95% — the prompt is captured exactly.
- **Added bias helps training (§6.3):** training the biased architecture from scratch is slightly faster and solves the induction task better on its own.
- **GPT-2 approximate (§6.4):** two-step (Performer-approx softmax → ICLAA bias). Logit relative error of dropping the context improves 16.56% → 9.17%; qualitatively the converted context-free model still generates context-consistent text (e.g. "CEO of Microsoft" after a removed Bill-Gates context; California/Los Angeles; Steve Jobs/Apple).

#### 4.10 Practical implications & limits (§8)

Enables: ICL-guided localized fine-tuning without catastrophic forgetting; inference savings by pre-converting long repeated prompts (system prompts, personal history) into a one-time bias; a cheap initialization for existing fine-tuning. Composable (add biases) and reversible (discard). **Limits:** exact only for linearized attention with $D_2$-separation, autoregressive, relative-PE; softmax case is approximate and bottlenecked by kernel quality; extending exact conversion to softmax attention is open.

### Appendix: Section-by-Section Backbone

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
