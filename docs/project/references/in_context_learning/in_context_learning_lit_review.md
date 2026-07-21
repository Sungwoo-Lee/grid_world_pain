# In-Context Learning — Master Literature Review

## What this document is (plain-language entry point)

**In-context learning (ICL)** is the ability of a Transformer to "learn" a new task at *inference time*, purely from examples placed in its prompt, **without any change to its weights**. Show a language model a handful of "input → answer" pairs and then a fresh input, and it completes the pattern — as if it had been trained on that task, even though not a single gradient step was taken. This document is a deep-dive review of the **38 papers** that try to explain *how* and *why* this works, grouped into the five themes the project cares about.

The corpus threads together **five complementary readings of the same phenomenon** — the five parts below:

1. **ICL as (implicit / amortized) Bayesian inference / model averaging.** The model behaves as if it holds a *prior* over tasks and, on reading the prompt examples, computes a *posterior* — sharpening its belief about "which task is this?" and predicting by averaging over that belief. The prompt is evidence; more examples means a sharper posterior.
2. **ICL as a learning algorithm run inside the forward pass.** The frozen network, when it reads a prompt, silently *executes an optimizer* — most concretely, one step of **gradient descent** per attention layer on a small regression problem built from the in-context examples. Training the weights (the slow, outer loop) installs a *learner* (the fast, inner loop) that runs at inference.
3. **Amortized Bayesian inference.** Train a network offline on many synthetic tasks so that one forward pass maps a *context set* of examples to a *predictive distribution* — the machinery behind Prior-Data Fitted Networks (PFN / TabPFN) and Neural Processes, and behind Transformers that emit full posteriors.
4. **Emergence / task-diversity conditions.** *When* does ICL appear at all, and when is it Bayesian versus not? The answer turns on data statistics (burstiness, a heavy-tailed vocabulary) and on how many distinct tasks pretraining saw (a diversity threshold).
5. **ICL as weight generation (the hypernetwork / fast-weight-programmer view).** The context is algebraically equivalent to a *set of extra weights*. A linear-attention layer accumulates the prompt into an outer-product "fast-weight" matrix; multi-head attention is literally a **hypernetwork**; and one can even *materialize* a whole prompt into a permanent, composable **weight edit**.

These readings are not rivals so much as different projections of one object. Reading (5) — the hypernetwork / fast-weight bridge — is the strand most directly relevant to this project's own conditioning work (see the separate `[[Hypernetwork]]` and FiLM reference topics): it says a FiLM or hypernetwork conditioning module is *not* a foreign add-on to attention but a more explicit exposure of a mechanism attention already contains. In particular, Ha et al.'s HyperNetworks (§28) contains the exact **row-scaling reduction that is the mathematical parent of FiLM**.

**Division of labor with the manifest.** The download record and index for this corpus lives in the manifest `[[in_context_learning_sources]]` — that doc tracks *which* papers exist, where the PDFs sit, and why each was pulled. **This** doc is the per-paper deep-dive: for each reviewed paper it carries a plain-language Phase 1 overview, a graduate-level Phase 2 with the full LaTeX derivations, and a section-by-section backbone. Use the manifest to see the whole corpus at a glance; use this doc to actually understand a paper. The manifest's five-section taxonomy is the organizing spine of this document.

> ### Scope caveat — read before assuming full coverage
>
> **This review covers all 38 on-disk papers.** The corpus catalogued in the manifest `[[in_context_learning_sources]]` is **39 papers**; the only one *not* reviewed here is **Schmidhuber (1992), "Learning to Control Fast-Weight Memories"** — it has no open-access PDF (behind the MIT-Press paywall), so it appears in Part 5 only as a short un-numbered placeholder at its manifest position. Every other paper carries a full Phase 1 + Phase 2 + backbone below.


---

## Table of Contents

**Part 1 — §1 ICL as Bayesian inference / model averaging**

1. [Xie et al. 2022 — An Explanation of In-Context Learning as Implicit Bayesian Inference](#1-xie-et-al-2022--an-explanation-of-in-context-learning-as-implicit-bayesian-inference)
2. [Wang et al. 2023 — Large Language Models Are Latent Variable Models: Explaining and Finding Good Demonstrations for In-Context Learning](#2-wang-et-al-2023--large-language-models-are-latent-variable-models-explaining-and-finding-good-demonstrations-for-in-context-learning)
3. [Panwar et al. 2024 — In-Context Learning through the Bayesian Prism](#3-panwar-et-al-2024--in-context-learning-through-the-bayesian-prism)
4. [Wies et al. 2023 — The Learnability of In-Context Learning](#4-wies-et-al-2023--the-learnability-of-in-context-learning)
5. [Elmoznino et al. 2025 — In-Context Learning and Occam's Razor](#5-elmoznino-et-al-2025--in-context-learning-and-occams-razor)
6. [Mittal et al. 2025 — Does Learning the Right Latent Variables Necessarily Improve In-Context Learning?](#6-mittal-et-al-2025--does-learning-the-right-latent-variables-necessarily-improve-in-context-learning)
7. [Hu et al. 2024 — Amortizing Intractable Inference in Large Language Models](#7-hu-et-al-2024--amortizing-intractable-inference-in-large-language-models)

**Part 2 — §2 ICL as gradient descent / learning algorithms inside transformers**

8. [Garg et al. 2022 — What Can Transformers Learn In-Context? A Case Study of Simple Function Classes](#8-garg-et-al-2022--what-can-transformers-learn-in-context-a-case-study-of-simple-function-classes)
9. [von Oswald et al. 2023 — Transformers Learn In-Context by Gradient Descent](#9-von-oswald-et-al-2023--transformers-learn-in-context-by-gradient-descent)
10. [Akyürek et al. 2023 — What Learning Algorithm Is In-Context Learning? (Investigations with Linear Models)](#10-akyrek-et-al-2023--what-learning-algorithm-is-in-context-learning-investigations-with-linear-models)
11. [Dai et al. 2023 — Why Can GPT Learn In-Context? (Language Models Secretly Perform Gradient Descent as Meta-Optimizers)](#11-dai-et-al-2023--why-can-gpt-learn-in-context-language-models-secretly-perform-gradient-descent-as-meta-optimizers)
12. [Bai et al. 2023 — Transformers as Statisticians (Provable In-Context Learning with In-Context Algorithm Selection)](#12-bai-et-al-2023--transformers-as-statisticians-provable-in-context-learning-with-in-context-algorithm-selection)
13. [Ahn et al. 2023 — Transformers Learn to Implement Preconditioned Gradient Descent for In-Context Learning](#13-ahn-et-al-2023--transformers-learn-to-implement-preconditioned-gradient-descent-for-in-context-learning)
14. [Li et al. 2023 — Transformers as Algorithms: Generalization and Stability in In-Context Learning](#14-li-et-al-2023--transformers-as-algorithms-generalization-and-stability-in-in-context-learning)
15. [Zhang et al. 2024 — Trained Transformers Learn Linear Models In-Context](#15-zhang-et-al-2024--trained-transformers-learn-linear-models-in-context)

**Part 3 — §3 Amortized Bayesian inference (PFN / TabPFN / neural processes / posterior estimation)**

16. [Müller et al. 2022 — Transformers Can Do Bayesian Inference (PFNs)](#16-mller-et-al-2022--transformers-can-do-bayesian-inference-pfns)
17. [Hollmann et al. 2023 — TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second](#17-hollmann-et-al-2023--tabpfn-a-transformer-that-solves-small-tabular-classification-problems-in-a-second)
18. [Garnelo et al. 2018 — Conditional Neural Processes](#18-garnelo-et-al-2018--conditional-neural-processes)
19. [Kim et al. 2019 — Attentive Neural Processes](#19-kim-et-al-2019--attentive-neural-processes)
20. [Nguyen & Grover 2022 — Transformer Neural Processes](#20-nguyen--grover-2022--transformer-neural-processes)
21. [Reuter et al. 2025 — Can Transformers Learn Full Bayesian Inference In Context?](#21-reuter-et-al-2025--can-transformers-learn-full-bayesian-inference-in-context)
22. [Mittal et al. 2025 — Amortized In-Context Bayesian Posterior Estimation](#22-mittal-et-al-2025--amortized-in-context-bayesian-posterior-estimation)
23. [Mittal et al. 2025 — In-Context Parametric Inference: Point or Distribution Estimators?](#23-mittal-et-al-2025--in-context-parametric-inference-point-or-distribution-estimators)
24. [Kang et al. 2026 — Transformers Can Learn Posterior Predictive Distributions In-Context](#24-kang-et-al-2026--transformers-can-learn-posterior-predictive-distributions-in-context)
25. [Mittal et al. 2025 — Iterative Amortized Inference: Unifying ICL and Learned Optimizers](#25-mittal-et-al-2025--iterative-amortized-inference-unifying-icl-and-learned-optimizers)

**Part 4 — §4 Emergence / task-diversity / latent-variable conditions for ICL**

26. [Chan et al. 2022 — Data Distributional Properties Drive Emergent In-Context Learning](#26-chan-et-al-2022--data-distributional-properties-drive-emergent-in-context-learning)
27. [Raventós et al. 2023 — Pretraining Task Diversity and the Emergence of Non-Bayesian In-Context Learning for Regression](#27-ravents-et-al-2023--pretraining-task-diversity-and-the-emergence-of-non-bayesian-in-context-learning-for-regression)

**Part 5 — §5 Hypernetworks & fast-weight programmers (attention ↔ weight generation)**

28. [Ha et al. 2017 — HyperNetworks](#28-ha-et-al-2017--hypernetworks)
29. [Ba et al. 2016 — Using Fast Weights to Attend to the Recent Past](#29-ba-et-al-2016--using-fast-weights-to-attend-to-the-recent-past)
30. [Munkhdalai & Yu 2017 — Meta Networks](#30-munkhdalai--yu-2017--meta-networks)
31. [Schlag et al. 2021 — Linear Transformers Are Secretly Fast Weight Programmers](#31-schlag-et-al-2021--linear-transformers-are-secretly-fast-weight-programmers)
32. [Irie et al. 2021 — Going Beyond Linear Transformers with Recurrent Fast Weight Programmers](#32-irie-et-al-2021--going-beyond-linear-transformers-with-recurrent-fast-weight-programmers)
33. [Irie et al. 2022 — A Modern Self-Referential Weight Matrix That Learns to Modify Itself](#33-irie-et-al-2022--a-modern-self-referential-weight-matrix-that-learns-to-modify-itself)
34. [von Oswald et al. 2020 — Continual Learning with Hypernetworks](#34-von-oswald-et-al-2020--continual-learning-with-hypernetworks)
35. [Zhmoginov et al. 2022 — HyperTransformer](#35-zhmoginov-et-al-2022--hypertransformer)
36. [Chen & Wang 2022 — Transformers as Meta-Learners for Implicit Neural Representations](#36-chen--wang-2022--transformers-as-meta-learners-for-implicit-neural-representations)
37. [Schug et al. 2025 — Attention as a Hypernetwork](#37-schug-et-al-2025--attention-as-a-hypernetwork)
38. [Chen et al. 2024 — Exact Conversion of In-Context Learning to Model Weights](#38-chen-et-al-2024--exact-conversion-of-in-context-learning-to-model-weights)

Also below: [Cross-Paper Synthesis](#cross-paper-synthesis) — the five throughlines that tie the 38 papers together; read this first for the big picture.

---

## Cross-Paper Synthesis

This section folds the per-shard and per-batch syntheses into one corpus-level view over all 38 papers. It traces five throughlines — one per manifest part — surfaces where the papers agree and disagree, and flags the axis most contested within the corpus and the strand most relevant to the project (the hypernetwork ↔ ICL bridge). Section numbers below (§N) refer to the numbered per-paper reviews.

### Throughline 1 — ICL as Bayesian inference / model averaging (the dominant lineage)

The single most-repeated claim in the corpus is that **ICL is Bayesian inference the model was never told it was doing**. The lineage builds cumulatively:

- **Xie et al. 2022** (§1) gives the founding argument: if pretraining documents carry a shared latent "concept," then to predict the next token the model must *infer the concept*, and at test time it infers the *shared prompt concept* by marginalization. The in-context predictor's belief over concepts **concentrates** on the true one as the number of examples grows (posterior concentration under a distinguishability condition). ICL emerges *for free* from ordinary next-token pretraining, provided the data has latent long-range structure.
- **Müller et al. 2022** (§16, PFNs) turns the observation into a *training objective*: sample tasks from a prior, hide a label, train a Transformer to predict it. Minimizing this "prior-data NLL" is exactly minimizing the KL divergence to the true **posterior predictive distribution (PPD)** — so a perfectly trained **PFN *is* Bayesian**, not by accident but as the exact minimizer of an objective whose global optimum is the posterior predictive.
- **Wang et al. 2023** (§2) carries the latent-variable account to *real LLMs*: an LLM behaves as a latent-variable (topic) model that infers a hidden concept $\theta$ and predicts by **Bayesian model averaging** over the concept posterior (their Eq. 1). Their gap theorem (Thm. 2.3) says the ICL classifier is never better than the Bayes-optimal concept-conditional predictor and *equals* it exactly when the demonstrations collapse the concept posterior onto the true task — reframing "good demonstrations" as "posterior-sharpening demonstrations," and yielding a practical, model-transferable demonstration-selection algorithm.
- **Panwar et al. 2024** (§3) is the most extensive *empirical* stress-test of "trained transformer ≈ Bayesian predictor": across many linear and non-linear function families, a high-capacity transformer tracks the **posterior-mean estimator (PME)** at *all* prompt lengths, even reproducing the closed-form implied weights. Its key conceptual move — the **mixture PME** $M_{\theta,\mathcal F}(P)=\sum_i\beta_i M_{\theta,F_i}(P)$ with $\beta_i$ the posterior probability of family $i$ — shows that *recognition and solution are the same integral*: there is no separate "identify the task then run its algorithm" step. It also cleanly reconciles the gradient-descent reading (Throughline 2) with the Bayesian one by proposing GD is simply the best tractable approximation to Bayes under a capacity budget.
- **Wies et al. 2023** (§4) supplies the rigorous complement: the first **PAC learnability** theory for ICL, with *finite-sample* guarantees. Its Lemma 1 makes the mechanism quantitative — the prompt likelihood-ratio of any wrong concept decays like $\exp(-k\,\Delta_{\mathrm{KL}}/2)$, so the concept posterior **re-weights onto the true task exponentially fast** in the number of examples $k$ and the inter-concept KL separation. The guarantee survives *random label flipping*, formally explaining why corrupting demonstration labels barely hurts ICL: what the prompt conveys is *which task*, not the input→label map.
- **Elmoznino et al. 2025** (§5) reframes the whole thing through **compression / Occam's razor**: the cumulative next-token loss *is* prequential code length, an upper bound on Kolmogorov complexity, so ICL is driven toward the *simplest* predictor that fits. Crucially they argue the **Kolmogorov/compression view generalizes the Bayesian view** (their Appendix H): it needs no well-defined task distribution and accommodates *non-Bayesian* empirics — precisely the high-diversity regime Raventós et al. (§27) exhibit. This is the conceptual umbrella under which the Bayesian lineage sits as a special case.
- **Mittal (Does Learning the Right Latent Variables…)** (§6) is the pointed *negative* result inside the lineage: even a bottleneck that provably *recovers* the true task latents does not, on its own, build a prediction head that uses them to extrapolate out-of-distribution — latent inference is **necessary but not sufficient**.
- **Hu et al. 2024** (§7) operationalizes "reasoning = amortized Bayesian inference over latents" as a *training algorithm* (GFlowNet fine-tuning), showing distribution-*matching* beats reward-*maximization* when diversity and robustness to reward misspecification matter — with the biggest gains out-of-distribution.
- The **posterior-estimation cluster** (§21–§24) pushes from "predict the next value" to "output the posterior itself." **Reuter et al. 2025** (§21) emits *full high-dimensional posterior samples* via flow matching, matching HMC. The **two Mittal posterior papers** (§22 Amortized, §23 Point-or-Distribution) run controlled bake-offs over *which objective* (forward vs reverse KL) and *whether to estimate a distribution at all*. **Kang et al. 2026** (§24) supplies the mechanistic proof that attention can compute the PPD's moments and emit a calibrated binned density.

All of these share one piece of machinery: the **joint-sampling / law-of-total-expectation trick** that turns an intractable "match the true posterior" objective into tractable training on simulator draws $(\theta,\mathcal{D})$ (Reuter's Proposition 1 ≡ Mittal's forward-KL derivation ≡ the PFN prior-data NLL ≡ Raventós's posterior-mean derivation).

### Throughline 2 — ICL as gradient descent / algorithms inside transformers (the mechanistic reading)

A second lineage asks not "what is the optimum?" but "what computation actually runs in the forward pass?" The chain runs from an exact construction to end-to-end training dynamics:

- **von Oswald et al. 2023** (§9) is the anchor: for regression, **one linear self-attention layer performs exactly one step of gradient descent** on a least-squares loss built from the in-context examples (explicit weight construction; trained weights *match* it; depth = iterated GD, with trained models discovering an accelerated "GD++").
- **Akyürek et al. 2023** (§10) identifies *which* algorithm by construction and by probing: fixed weights can realize one GD step ($O(d)$ hidden space) *or* an exact ridge update via Sherman–Morrison ($O(d^2)$), and a trained model transitions through algorithmic **phases** with depth (1 layer ≈ one GD step, 2–4 ≈ ridge, 8+ ≈ exact OLS), matching the **minimum-Bayes-risk ridge** predictor on noisy data. Intermediate quantities ($X^\top Y$ early, $w_{\mathrm{OLS}}$ late) are decodable from activations.
- **Dai et al. 2023** (§11) carries the reading to *off-the-shelf GPTs on real NLP tasks* via the **attention↔gradient-descent dual form**: relaxed linear attention over demonstrations equals an implicit weight update $\Delta W_{\mathrm{ICL}}$ on the zero-shot weights, so ICL is "implicit fine-tuning." The analogy pays rent: adding momentum to attention (an EMA over value vectors) improves both perplexity and ICL accuracy.
- **Ahn et al. 2023** (§13) sharpens von Oswald from *expressivity* to *optimization*: the **global minimum** of the training loss implements **preconditioned** GD whose preconditioner approximates $\Sigma^{-1}$ (the inverse data covariance) plus a sample-size-dependent regularizer; deep nets recover GD++/Newton-like preconditioning. The learned optimizer bakes in distributional statistics that vanilla GD lacks.
- **Bai et al. 2023** (§12) is the most ambitious: a single transformer implements a whole *statistician's toolbox* (ridge, Lasso, GLMs, two-layer-NN GD) via one reusable in-context-GD engine (one attention layer ≈ one GD step, error accumulating only *linearly* in steps), **plus in-context algorithm selection** — post-ICL validation and pre-ICL testing let one transformer pick the right base algorithm per input, achieving nearly-Bayes risk under *mixed* noise. It also proves you can *learn* these transformers from polynomially many pretraining sequences (the first end-to-end ICL theory).
- **Li et al. 2023** (§14) supplies the statistical-learning scaffolding: abstracting the transformer as a data→predictor *algorithm*, it proves $1/\sqrt{nT}$ generalization via **algorithmic stability** (a Doob-martingale + Azuma argument that survives temporally dependent in-context examples), and shows transfer is governed by *task complexity*, not model size.
- **Zhang et al. 2024** (§15) closes the loop with the training-dynamics convergence proof: gradient flow on a single linear-attention layer provably reaches the global minimum (via a PL inequality under balanced initialization), landing on $\Gamma^{-1}$-preconditioned GD — but its **robustness audit** exposes fragility: the learned algorithm is *not* covariate-scale-invariant OLS (it secretly depends on the training covariance), a direct caution for any conditioner trained on a fixed input statistic.
- **Garg et al. 2022** (§8) is the clean *empirical* counterpart to the whole thread: trained from scratch, a transformer **matches the optimal algorithm** for well-defined function classes (least squares — including the double-descent curve — Lasso for sparse, and 2-layer nets) in one forward pass, robust to distribution shift, so it is *not* nearest-neighbor lookup.

These meet the Bayesian thread at **Kang et al. 2026** (§24), which bridges the von-Oswald/Akyürek "attention = gradient descent" mechanism to the distributional PFN world (attention runs a Richardson iteration for posterior-predictive *moments*), and at **Panwar** (§3), which frames GD as capacity-limited Bayes.

### Throughline 3 — Amortized inference / neural processes (Part 3)

Part 3 is the family where the Bayesian objective is *baked into a network offline* so that one forward pass maps a context set to a predictive distribution. Two design axes organize it: the **objective** (a full-set factored likelihood vs an autoregressive factorization) and the **path** used to summarize context (a deterministic point-conditional embedding vs a global latent sampled once per prediction).

- **Garnelo et al. 2018** (§18, Conditional Neural Processes) is the seed: a permutation-invariant encode-aggregate-decode network trained on a held-out conditional log-likelihood — structurally the *same* amortization as the PFN loss (§16/§17), but the aggregator is a **mean**, giving a deterministic point-conditional predictor and $O(n{+}m)$ cost. Its factored likelihood cannot produce coherent joint samples, motivating the latent path.
- **Kim et al. 2019** (§19, Attentive NPs) diagnoses the mean-aggregation as an **underfitting bottleneck** and cures it with **attention** — the neural analogue of a GP kernel — added only to the deterministic (local) path, keeping a single global latent $z$ on the latent (global) path for coherent samples. This formalizes the deterministic-vs-latent path distinction that runs through the whole family.
- **Nguyen & Grover 2022** (§20, Transformer NPs) throws out the latent variable entirely and casts the problem as **autoregressive sequence modeling** (GPT-style, likelihood-exact, no variational bound), obtaining joint coherence through the chain rule instead of a latent; variants trade expressivity for cost (TNP-A ≻ TNP-ND ≻ TNP-D) and win where uncertainty must drive sequential decisions.
- **Müller/Hollmann** (§16/§17, PFN/TabPFN) are the *full-amortization* extreme: the entire posterior average is in the forward pass, with the inductive bias written *explicitly* as a synthetic-dataset generator (TabPFN's Structural-Causal-Model + BNN prior with an Occam simplicity preference) — a single fixed Transformer that replaces the fit-and-tune pipeline for small tabular classification in under a second.
- The posterior-estimation cluster then adds the **point-vs-distribution estimator axis** (Reuter §21 vs the Mittal papers §22/§23 — see the contested-axis section below), and **Mittal (Iterative Amortized Inference)** (§25) is the abstraction layer over the entire family: every fast-adaptation method (MAML, ICL, PFNs, hypernetworks, learned optimizers, neural processes) is one template $f_\gamma(x, g_\phi(\mathcal{D}_T))$, and its **parametric / implicit / explicit** taxonomy places hypernetworks (parametric), ICL/PFNs (implicit), and neural processes (explicit) as three faces of one object — the formal hinge connecting Part 3 to Part 5.

### Throughline 4 — Emergence / task-diversity conditions (Part 4)

Two papers ask *when* ICL appears and *whether it is Bayesian*:

- **Chan et al. 2022** (§26) gives the **data-side** necessary conditions: ICL emerges from the *interaction* of the transformer with naturalistic data statistics — **burstiness** (items recur in clusters), a large **long-tailed vocabulary** of rare classes, and **dynamic meaning** — and ICL/IWL (in-weights learning) coexist only under a **Zipfian marginal at exponent ≈ 1**, exactly where natural language sits. Under matched parameters, RNNs/LSTMs never achieve ICL; "attention is *not* all you need" — architecture **and** data both matter.
- **Raventós et al. 2023** (§27) quantifies the **task-diversity threshold** in a setting where the Bayesian ideal is computable (in-context linear regression). Below threshold the transformer matches **dMMSE** (Bayes-optimal w.r.t. the *finite pretraining pool*) and fails new tasks; above a sharp threshold it matches **Ridge** (the Gaussian-prior Bayes predictor) and solves genuinely new tasks — by *deviating* from the pretraining-Bayes optimum. This is the corpus's sharpest statement that generalizing ICL is **non-Bayesian w.r.t. the pretraining prior**, and that *scale alone (below threshold) actively hurts*. It validates the Xie/implicit-Bayes story (§1) only in the low-diversity regime and is the empirical anchor for Elmoznino's claim (§5) that compression generalizes Bayes.

Together they answer the "when" the mechanism papers assume: ICL is not a free gift of the architecture but is switched on by data distribution (Chan) and by task variety past a threshold (Raventós).

### Throughline 5 — The hypernetwork ↔ ICL / fast-weight bridge (the corpus's payoff for the project)

This arc — the "context as weights" reading — is the sub-corpus closest to the project's hypernetwork / FiLM conditioning interests, and it is where the corpus earns its keep for the project. Read it as a single chain from a 1990s idea to literal, storable weight edits:

- **Ba et al. 2016** (§29) and the (unreviewed) **Schmidhuber 1992** placeholder are the historical root: a **Hebbian outer-product fast-weight memory *is* attention over the past**. Ba's Eq. (4) shows $A(t)h = \eta\sum_\tau \lambda^{t-\tau} h(\tau)\,[h(\tau)^\top h]$ — a decay-weighted sum of past states weighted by dot-product similarity — i.e. content-addressable memory stored in *weights*, not activations, retrievable without gradient updates.
- **Ha et al. 2017** (§28, HyperNetworks) is the canonical *weight-generation* paper — one small network emits another's weights, end-to-end differentiable — and, **critically for this project, contains the row-scaling / element-wise-multiplication reduction (its Eqs. 7–8, 12) that is the mathematical parent of FiLM**: rather than rebuild a weight matrix from a context code, *scale its rows* by a context-derived vector, i.e. feature-wise multiplicative gain plus additive bias. This is the exact algebra the project's FiLM modulator uses; cross-link the separate `[[Hypernetwork]]` topic (which holds the primary Ha review and the FiLM corpus). The dynamic (per-timestep, input-conditioned) HyperRNN variant is a concrete mechanism by which context can *reprogram* computation without gradient steps.
- **Schlag et al. 2021** (§31) makes the equivalence exact and modern: **linear self-attention is a Fast Weight Programmer**. Dropping softmax and using associativity, the causal attention output reads from an accumulated **outer-product matrix** $\mathbf{W}^{(i)}=\sum_j \mathbf{v}^{(j)}\otimes\phi(\mathbf{k}^{(j)})$ — the context *is* a weight matrix written by outer products over the key index. Its **delta-rule** extension (DeltaNet) makes that memory *editable* via a learned input-dependent write strength — i.e. the in-context write is itself a learned optimization step (Widrow–Hoff).
- **Irie et al. 2021** (§32) and **Irie et al. 2022** (§33) push the FWP lineage into recurrence and self-reference — Recurrent FWPs (Delta RNN / Recurrent Delta Net) and a **Self-Referential Weight Matrix** that rewrites *itself* (only $W_0$ is trained). These are the two papers in the corpus that actually run fast-weight memories as the recurrent core of an **actor-critic RL agent** (Atari/IMPALA; ProcGen multi-task), and Irie 2022's **Fake-SR ablation** (self-modification on vs off, matched budget) is the exact modulation-on/off control the project should emulate.
- **Munkhdalai & Yu 2017** (§30, Meta Networks) is the "explicit hypernetwork + gradient-as-context" ancestor: it generates fast weights from *loss gradients*, and reports a load-bearing empirical fact — a base learner built from fast weights *alone* fails to converge; the fast path must be **added to a stable slow path** (layer augmentation), directly analogous to why FiLM *modulates* a base feature rather than replacing it.
- **von Oswald et al. 2020** (§34, Continual Learning with Hypernetworks) is the canonical **task-conditioned hypernetwork**: a shared weight-generator plus a per-task embedding, with a **data-free output regularizer** that lets one generator hold many context-specific behaviours without interference — the reference architecture if the grid-world agent must support several noise/context regimes from one network.
- **Zhmoginov et al. 2022** (§35, HyperTransformer) and **Chen & Wang 2022** (§36, Trans-INR) are the "Transformer-as-hypernetwork" archetypes — a Transformer reads a support/observation set and emits an entire target network's weights in one pass — and each proves the tightest local bridge to Throughline 2: **a self-attention / residual-attention layer computes one (learned) gradient-descent step on the generated weights**. Zhmoginov's "generate only the last layer suffices" is a cheap, importable design point (condition just the readout).
- **Schug et al. 2025** (§37, Attention as a Hypernetwork) factors the *same* attention layer over the **head** index to reveal **multi-head attention as a hypernetwork**: the per-head attention scores are a *latent code* selecting a linear combination of fixed per-head operation matrices (number of heads = latent-code dimension). This is the first-principles argument that **attention is already conditional modulation** — the FiLM / hypernetwork family — so exposing a hypernetwork conditioning module is not foreign to attention.
- **Chen et al. 2024** (§38, Exact Conversion of ICL to Model Weights) is the extreme, literal endpoint: for linearized attention, a whole ICL prompt collapses **exactly** into a single **bias edit** on the Key–Value matrix $A=\sum_j\phi(K_j)V_j^\top$ — the very outer-product accumulator Schlag identified. Algorithm ICLCA materializes the context into permanent, composable, reversible weights in one forward pass (no gradient descent) — the proof-of-concept that a learned context code *can* become a storable weight offset.

The chain is therefore: **Ba/Schmidhuber** (fast weights = attention over the past) → **Ha** (context generates weights; its row-scaling reduction = the parent of FiLM) → **Schlag** (linear attention = outer-product fast-weight programmer) → **Irie 2021/2022** (recurrent / self-referential FWPs, the RL demonstrations) → **Munkhdalai** (fast weights must be added to a stable base, like FiLM) → **von Oswald 2020 / Zhmoginov / Chen & Wang** (full-weight generation from a context/embedding, attention layer ≈ one GD step) → **Schug** (multi-head attention *is* a hypernetwork latent code) → **Chen 2024** (materialize the accumulator as a literal, composable weight edit). Two recurring project-relevant lessons fall out: **attention is already the FiLM/hypernetwork family** (Schug, Ha), and **context should perturb a stable base computation, not supplant it** (Munkhdalai's fast-only failure, the delta/residual updates, von Oswald's drift regularizer). Cross-link `[[Hypernetwork]]` for the conditioning-architecture side of this bridge.

### The contested axis — point vs. distribution estimator

The sharpest *disagreement* in the corpus is internal to the posterior-estimation cluster (§21–§24), and it is a genuine, not merely apparent, contest:

- **Reuter et al. 2025** (§21) demonstrates a **full-distribution** estimator (flow matching) that *beats* variational inference on posterior fidelity — especially for skewed and multimodal posteriors in *modest* latent dimension.
- **Mittal (Point or Distribution)** (§23) demonstrates that **point** estimators (MLE/MAP) generally *beat* distribution estimators for *downstream prediction* — winning ~71% of 88 tasks — especially in *high* dimension and under multimodality. The argued cause is **fundamental, not an engineering gap**: parameter-space symmetries (label-switching in mixtures, weight-permutation symmetries in neural nets) create combinatorially many *equivalent* modes that are expensive to represent yet predictively redundant.

These are **reconcilable**, and the reconciliation is instructive: they measure *different things on different regimes*. Reuter scores *posterior-sample fidelity* against HMC at modest latent dimension; Mittal scores *predictive loss* at high dimension. Both agree the distribution advantage **shrinks as latent dimension grows** — Reuter's own $K=20,50$ ablation shows the advantage evaporate, matching Mittal's high-dimensional finding. A secondary axis, the **amortization objective**, runs alongside: **forward KL** (neural posterior estimation / SBI, = Reuter's flow-matching loss) is mean-seeking and *must* train on simulator draws (fragile under misspecification); **reverse KL** (amortized VI / neural-process style) is mode-seeking, can train on arbitrary/real data, and is more robust in high dimension. **Symmetric KL** is a strong compromise (Mittal §23).

### Gaps and connections to the project

- **The existence-vs-learning gap.** von Oswald (§9), Akyürek (§10), Kang (§24), the HyperTransformer/Trans-INR "attention ≈ GD step" results (§35/§36), and Chen (§38) are all **constructive** — they prove the architecture *can* realize a mechanism, not that standard pretraining *discovers* it. Ahn (§13) and Zhang (§15) partly close this for a single linear-attention layer (the optimum *is* preconditioned GD and training *reaches* it), but Zhang's audit shows the learned algorithm is silently covariate-shift-dependent. This is the corpus's biggest open caution for any project claim that a conditioning mechanism "implements" a given computation.
- **Latent inference is necessary but not sufficient.** Mittal (§6) is the pointed negative result: even when a bottleneck provably *recovers* the true task latents (linearly decodable, causally interventable), end-to-end training fails to build a *prediction* head that uses them to extrapolate OOD. The transferable lesson: **a clean, low-dimensional, interpretable task representation is not self-justifying** — the module that *consumes* it needs its own inductive biases. This is a direct warning for any diagnosis of why a modulation signal is present in activations yet fails to change behavior.
- **Diversity, not scale, drives generalizing adaptation.** Raventós (§27) and Chan (§26) jointly say: whether a context-conditioned model *memorizes* or *generalizes* across regimes is set by task diversity (above a threshold) and data statistics (burstiness, heavy tails), and scaling data at low diversity is counterproductive. For any project component meant to "adapt in-context to unseen conditions," diversify the task/context distribution past threshold. Hu et al. (§7) adds that distribution-*matching* beats reward-*maximization* when diversity buys OOD robustness — prefer it in RL when diverse latent samples are what matter.
- **The FiLM/hypernetwork bridge is the payoff.** Throughline 5 is the formal backing for treating attention-style conditioning and FiLM/hypernetwork conditioning as the *same family* (Schug §37, Ha §28), for the "modulate, don't overwrite" design rule (Munkhdalai §30), for the "condition just the readout" cheap design point (Zhmoginov §35), and for the modulation-on/off ablation template (Irie 2022 §33's Fake-SR). These links are drawn here for the reader to follow into the project's develop-side docs (the neuromodulation / FiLM conditioning line, its gates, and the null-result diagnosis series); this review does not itself re-run those diagnoses. Any move to implement a delta-rule fast-weight memory or a task-conditioned hypernetwork head in the project's RL stack should be routed to `senior-developer` for an `issue_plan`; equation-consistency checks belong to `math-reviewer`.

---

## Per-Paper Reviews

The 38 reviews below are preserved verbatim from the per-paper deep-dives (Phase 1 overview, Phase 2 graduate deep-dive with full LaTeX derivations, and the section-by-section backbone completeness guard). Only the paper numbering (1–38), the three sub-section headers, and the thematic ordering have been unified; no paper content has been compressed. They are grouped into the manifest's five parts.


---

# Part 1 — §1 ICL as Bayesian inference / model averaging

*How conditioning on prompt examples adapts a frozen model: the papers reading ICL as (implicit) Bayesian inference over a latent task, predicting by averaging over the posterior — from the founding HMM account, through real-LLM latent-variable and empirical posterior-mean tests, to finite-sample PAC learnability and the compression/Occam umbrella.*

---

## 1. Xie et al. 2022 — An Explanation of In-Context Learning as Implicit Bayesian Inference

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

## 2. Wang et al. 2023 — Large Language Models Are Latent Variable Models: Explaining and Finding Good Demonstrations for In-Context Learning

**PDF:** `docs/project/references/in_context_learning/sources/Wang et al. 2023 - Large Language Models Are Latent Variable Models.pdf` · NeurIPS 2023 · arXiv:2301.11916.

### Phase 1: Foundational Overview (Undergraduate-Level)

**Question this paper answers (plain language).** When you give a large language model (LLM) a few worked examples ("demonstrations") before your real question — e.g. two labeled movie reviews before a third unlabeled one — the model often gets the answer right without any retraining. This is *in-context learning* (ICL). But the answer is famously fragile: swap which examples you show, and accuracy swings wildly. This paper asks two linked questions: (1) *Why* does ICL work at all under ordinary language-model pretraining? and (2) Can we *pick* the best few demonstrations automatically, cheaply, and in a way that transfers across models?

**Core idea.** Treat the LLM as a **latent variable model** (like a modern neural topic model). The claim is that when the model reads a prompt, it implicitly infers a hidden "concept" or "task" variable $\theta$ — capturing both the task ("classify sentiment") and the format — and then generates its answer by conditioning on that inferred concept. Good demonstrations are the ones that most sharply point the model at the *correct* concept $\theta^d$ for task $d$.

**Key findings.**
- **Theory.** With a finite number of demonstrations, the ICL classifier can reach the **Bayes-optimal classifier** — but only if the demonstrations make the model's posterior over concepts collapse onto the true task concept $\theta^d$. This is a stronger, more practical statement than the prior HMM-based result of Xie et al. (which needed *infinitely many* demonstrations).
- **Algorithm.** A two-stage recipe: (1) **latent concept learning** — learn a handful of new "concept token" embeddings per task by prompt-tuning a *small* LM (< 1B params, e.g. GPT2-large); (2) **demonstration selection** — score each candidate example by how strongly it predicts those concept tokens, and keep the top-$k$.
- **Results.** Demonstrations chosen with a small model transfer directly to much larger models (up to 175B). Averaged over 8 GPT models and 8 text-classification datasets, the method beats uniform selection by ~12.5% relative. It also helps on GSM8K math word problems. Control experiments show the gain comes specifically from the *learned* concept tokens (random tokens give no benefit), and — unlike prior work — demonstration *order* barely matters once selection is good.

**Initial takeaway.** This is the first Bayesian/latent-variable explanation of ICL that (a) rests on realistic assumptions (no HMM, finite demonstrations, arbitrary causal direction) and (b) yields an *actually useful* algorithm validated on real LLMs. The conceptual payload for our project: a frozen sequence model can behave like a task-inferring Bayesian, and "conditioning on demonstrations" is mathematically "sharpening a posterior over latent tasks."

### Phase 2: Graduate-Level Deep Dive

#### Setup and generative assumptions

We predict a discrete target $Y\in\mathcal Y$ from a token sequence $X\in\mathcal X$. A latent, *continuous* concept variable $\theta\in\Theta$ indexes the task. The paper posits a structural causal model with two possible directions, chosen per task:

$$Y = f(X,\theta,\epsilon) \quad (X\!\to\!Y\!\leftarrow\!\theta, \text{ "direct"}), \qquad X = g(Y,\theta,\epsilon)\quad (Y\!\to\!X\!\leftarrow\!\theta,\text{ "channel"}),$$

with independent noise $\epsilon$ and deterministic $f,g$. An **injective** map $T\to\Theta$ assigns each task $d$ a unique concept $\theta^d$, so every example is generated as $Y^d=f(X^d,\theta^d,\epsilon)$. Labels are projected to token space by injective $\tau^d:\mathcal Y\to\mathcal X$ (e.g. positive $\mapsto$ "positive"), and a delimiter $w^d$ separates concatenated demonstrations. With this preprocessing the LLM's ICL output probability is written $P^d_M$.

The **starting identity** for ICL marginalizes over the latent concept (Eq. 1):

$$
P^d_M(Y\mid X^d_1,Y^d_1,\dots,X^d_k,Y^d_k,X) \;=\; \int_\Theta P^d_M(Y\mid \theta,X)\,P^d_M(\theta\mid X^d_1,Y^d_1,\dots,X^d_k,Y^d_k,X)\,d\theta.
\tag{1}
$$

This is exactly **Bayesian model averaging**: the predictive is the concept-conditional predictor $P^d_M(Y\mid\theta,X)$ averaged against the posterior over concepts induced by the demonstrations. The bridging **Assumption 2.1** ties the pretrained model to the true data distribution: $P_M(X)=P(X)$ and $P^d_M(Y\mid\theta,X)\propto P(Y\mid\theta,X)$.

#### Bayes-optimality of the concept-conditional predictor (Prop. 2.2)

**Claim.** For the $X\!\to\!Y\!\leftarrow\!\theta$ direction, $\arg\max_{y}P^d_M(Y=y\mid\theta^d,X)$ is the Bayes-optimal classifier.

**Derivation.** Because task $d$'s data generation is $Y=f(X,\theta^d,\epsilon)$, the *true* conditional depends on $\theta^d$ only:
$$P^d(Y\mid X) = P(Y\mid \theta^d,X).$$
By Assumption 2.1, $\arg\max_y P^d_M(Y=y\mid\theta^d,X)=\arg\max_y P(Y=y\mid\theta^d,X)$. The RHS is by definition the minimizer of misclassification probability — the Bayes classifier. $\square$

#### The gap theorem: ICL ≤ Bayes, equality iff posterior collapses (Thm. 2.3)

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

#### Method — latent concept learning (Algorithm 1)

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

#### Method — demonstration selection (Algorithm 2)

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

#### Empirical anatomy

- **Transfer:** demonstrations selected once with GPT2-large lift 8 GPT models (GPT2/GPT3 families, 6–175B), ~12.5% relative over uniform, attributed to shared pretraining corpora.
- **Cross-family:** also helps GPT-J, OPT, LLaMA at 6–7B; GSM8K (Table 1) with LLaMA-2 / ChatGPT beats uniform and similarity baselines, and *larger* selector (7B) beats smaller (1.5B).
- **Ablations:** learned concept tokens matter (random tokens ≈ random selection); $k\in\{2,4,8,16\}$ all beat baseline (best at $k{=}4$, diminishing at large $k$); $c{\approx}10$ concept tokens is a sweet spot (too few underfit, too many dilute selectivity); order-insensitivity (contrasting Lu et al. 2022) — interpreted as order capturing model-specific artifacts while content carries task info.
- **Qualitative:** t-SNE shows concept tokens for semantically similar tasks cluster; selected examples are harder/longer/more deductive, echoing "hard pretraining examples drive ICL."

#### Relevance notes for the project
The paper operationalizes the abstract "ICL = Bayesian task inference" claim into (i) a clean risk-comparison theorem (ICL ≤ Bayes, gap governed by posterior concentration) and (ii) a KL-decomposition showing that a *learned low-dimensional latent code* (concept tokens) can serve as an approximate sufficient statistic for the task. For any project component that conditions a frozen policy/model on a small context to "identify a situation," this is the reference for the identifiability + posterior-collapse framing.

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** ICL is powerful but sensitive to demonstration choice/format/order; mechanisms under standard LM pretraining are not understood. Prior latent-topic account (Xie et al. 2022) assumes HMM data and needs infinite demonstrations — hard to extrapolate to real language. Proposes a topic-model-style latent variable view: $P(w_{1:T})=\int_\Theta P(w_{1:T}\mid\theta)P(\theta)d\theta$, and a simplified LLM analogue $P_M(w_{t+1:T}\mid w_{1:t})=\int_\Theta P_M(w_{t+1:T}\mid\theta)P_M(\theta\mid w_{1:t})d\theta$, where $\theta$ is an approximate sufficient statistic (task + format). Empirical-Bayes: approximate optimal $\theta^d$ with a small LM. Contributions: general 3-variable causal generative process; finite-demonstration Bayes-optimality; practical, transferable selection algorithm.
- **§2 Theoretical Analysis.** §2.1 notations: discrete $Y$, sequence $X$, continuous latent $\theta$, two causal directions ($Y=f(X,\theta,\epsilon)$ / $X=g(Y,\theta,\epsilon)$), injective $T\to\Theta$, label-to-token maps $\tau^d$, delimiter $w^d$. §2.2 results: marginalization identity Eq. (1); Assumption 2.1 (pretrained ≈ true dist.); Prop. 2.2 (concept-conditional predictor is Bayes-optimal); Thm. 2.3 (ICL risk ≥ Bayes risk, equality iff posterior collapses to $\theta^d$); channel-direction analogue Eq. (2).
- **§3 Method.** Two-phase pipeline (Fig. 1). §3.1 Latent Concept Learning (Alg. 1): add $c|S|$ concept tokens, freeze $M$, prompt-tune $E_{\text{new}}$ to minimize $L(\hat\theta^d)$; Prop. 3.1 (minimizer recovers true concept, identifiable if $M$ invertible); normalize concept posterior over sampled task subset $S$. §3.2 Demonstration Selection (Alg. 2): factorize concept posterior under independence + uniform prior → select top-$k$ by per-example concept score; order argued irrelevant.
- **§4 Experiments.** 8 datasets across 5 task types (SST2, FPB, COLA, DBpedia, EmoC, EmoS, ETHOS-SO, ETHOS-R). Direction chosen per task by which gives higher random-demonstration accuracy. Defaults $k{=}4$, $c{=}10$, selector GPT2-large. Baselines: Uniform, Similar (sentence-transformer cosine). Main result: +12.5% relative over uniform across 8 GPTs (Fig. 2). Non-GPT models (Fig. 3a), GSM8K (Table 1), learned-vs-random tokens (Fig. 3b), $k$ and $c$ ablations (Fig. 4), order insensitivity, t-SNE qualitative (Fig. 5).
- **§5 Related Work.** Heuristic similarity/entropy selection; pretraining-distribution accounts (HMM, Zipfian burstiness); GD-connection mechanistic accounts; other latent-variable theories; empirical ICL studies (label-noise robustness, chain-of-thought). Positions itself as first Bayesian account verifiable on real LLMs.
- **§6 Conclusion.** LLMs as implicit topic models inferring a latent concept; two-step algorithm (extract concept tokens with small LM → select demonstrations), directly transferable across LLMs.
- **Appendix A Proofs.** A.1 direct direction (Prop. 2.2, Thm. 2.3 with 0–1 risk algebra); A.2 channel direction (Prop. A.5 balanced-label Bayes, Thm. A.6 gap theorem); A.3 method (Prop. 3.1 via $L=H+\mathrm{KL}$ decomposition, identifiability under invertibility).
- **Appendix B Experiments.** Dataset templates/label mappings (Table 2), sizes, licenses; training details (A100/V100/A6000, lr 1e-4, batch 16, 10k steps); detailed per-dataset tables; causal-direction, other-LLM, random-token results.

---

## 3. Panwar et al. 2024 — In-Context Learning through the Bayesian Prism

**PDF:** `docs/project/references/in_context_learning/sources/Panwar et al. 2024 - In-Context Learning through the Bayesian Prism.pdf` · ICLR 2024 · arXiv:2306.04891.

### Phase 1: Foundational Overview (Undergraduate-Level)

**Question this paper answers (plain language).** If you train a transformer from scratch on a huge, controlled stream of (input, output) example-sequences drawn from some known family of functions (say, linear functions), it learns to solve *new* functions from that family at test time just by reading examples — no weight updates. This is the "learning-to-learn" (meta-ICL) setup of Garg et al. 2022. The puzzle: *what rule* has the transformer internalized? This paper's answer, tested at scale: to a strikingly good approximation, a high-capacity transformer computes the **Bayesian predictor** for the exact distribution it was pretrained on — it predicts by averaging over the posterior of possible functions given the examples.

**Two headline setups.**
- **MICL** ("meta-ICL"): pretrain on one function family (e.g. linear regression with a Gaussian prior on weights).
- **HMICL** ("hierarchical meta-ICL"): pretrain on a *mixture* of several families (e.g. sometimes a linear function, sometimes a decision tree). This is the paper's new contribution and is closer to real LLMs, which switch between many task types.

**Key findings.**
1. **Transformers solve task mixtures — and do so *without* a separate "recognize-then-solve" step.** The Bayesian view predicts this for free: the optimal predictor for a mixture is automatically a *posterior-weighted blend* of the per-family predictors, where the blend weights shift toward the correct family as more examples arrive.
2. **They match the exact Bayesian predictor at *all* prompt lengths**, not just asymptotically. Where math permits (Gaussian mixtures), the transformer's loss curves and even its *recovered implied weights* track the closed-form posterior-mean estimator (PME) almost exactly.
3. **ICL's inductive bias is inherited from the pretraining distribution, not added by the architecture.** Using Fourier-series tasks: if all frequencies appear equally in pretraining, the model shows no frequency preference; if low frequencies dominate pretraining, the model shows a *simplicity bias* (prefers low-frequency fits) — exactly what a Bayesian would do given that prior.
4. **Generalization to *unseen* function classes requires enough task diversity — and this involves *deviating* from the pure Bayesian predictor.** With few pretraining tasks the model memorizes them; past a diversity threshold it generalizes to brand-new tasks (matching a broader Bayesian predictor / Lasso).
5. **A striking "forget" phenomenon:** at intermediate task diversities, during pretraining the model *first* generalizes to the full task distribution, then later *forgets* it and reverts to memorizing the pretraining tasks.
6. **Where transformers appear to do gradient descent (low-capacity, or hard tasks), the paper hypothesizes GD is simply the best *tractable approximation* to Bayesian inference under capacity constraints** — not a competing story.

**Initial takeaway.** This is the most extensive *empirical* stress-test of "trained transformer ≈ Bayesian predictor." It supports the Bayesian prism as a unifying lens (mixtures, inductive bias, simplicity bias all fall out of it) while honestly cataloguing where and why the model deviates (limited capacity, out-of-distribution generalization, forgetting).

### Phase 2: Graduate-Level Deep Dive

#### The meta-ICL objective and the posterior-mean estimator (PME)

Let $D_X$ be an input distribution on $\mathbb R^d$, $\mathcal F$ a function family with prior $D_{\mathcal F}$. A prompt of length $p$ is
$$P=\big(x_1,f(x_1),\dots,x_p,f(x_p),x_{p+1}\big),\qquad f\sim D_{\mathcal F},\ x_i\stackrel{iid}{\sim}D_X.$$
The transformer $M_\theta$ is trained to predict $f(x_{p+1})$ under squared loss:
$$
\min_\theta\ \mathbb E_{f,x_{1:p}}\Big[\tfrac{1}{p+1}\sum_{i=0}^{p}\ell\big(M_\theta(P^i),f(x_{i+1})\big)\Big],\qquad \ell(y,y')=(y-y')^2,
$$
where $P^i=(x_1,f(x_1),\dots,x_i,f(x_i),x_{i+1})$ is the sub-prompt with $i$ examples plus the query.

**Derivation that the optimum is the posterior mean.** Condition the inner expectation on the observed sub-prompt:
$$
\min_\theta \mathbb E_{f,x_{1:i}}\,\ell\big(M_\theta(P^i),f(x_{i+1})\big)
=\mathbb E_{P^i}\Big[\min_\theta \mathbb E_f\big[\ell(M_\theta(P^i),f(x_{i+1}))\mid P^i\big]\Big].
$$
For squared loss the inner minimizer over a scalar prediction is the conditional mean, so the **Bayes-optimal ICL predictor** — the "PME" — is
$$
\boxed{\,M_\theta(P^i)=\mathbb E_{f\sim D_{\mathcal F}}\big[f(x_{i+1})\mid P^i\big]=\int p_{\mathcal F}(f\mid P^i)\,f(x_{i+1})\,df.\,}
$$
This is a **Bayesian model average**: predictor $f(x_{i+1})$ integrated against the *posterior over functions* $p_{\mathcal F}(f\mid P^i)\propto p(P^i\mid f)\,p_{\mathcal F}(f)$.

#### Hierarchical meta-ICL (HMICL) and the mixture PME

HMICL draws the family itself first: family set $\mathcal F=\{F_1,\dots,F_m\}$, sampling weights $\alpha=[\alpha_1,\dots,\alpha_m]$, $\sum_i\alpha_i=1$. Sample $F_i\sim\alpha$, then $f\sim F_i$, then the prompt. MICL is the $m{=}1$ case.

**Derivation of the mixture PME (Appendix A.1).** Take two families ($m{=}2$) for clarity; the marginal function density is $p_{\mathcal F}(f)=\alpha_1 p_1(f)+\alpha_2 p_2(f)$. Compute the posterior over functions given a prompt $P$:
$$
\begin{aligned}
p_{\mathcal F}(f\mid P)
&=\frac{p(P\mid f)\,p_{\mathcal F}(f)}{p_{\mathcal F}(P)}
=\frac{p(P\mid f)}{p_{\mathcal F}(P)}\big(\alpha_1 p_1(f)+\alpha_2 p_2(f)\big)\\
&=\underbrace{\frac{\alpha_1 p_1(P)}{p_{\mathcal F}(P)}}_{\beta_1}\cdot\frac{p(P\mid f)p_1(f)}{p_1(P)}
+\underbrace{\frac{\alpha_2 p_2(P)}{p_{\mathcal F}(P)}}_{\beta_2}\cdot\frac{p(P\mid f)p_2(f)}{p_2(P)}\\
&=\beta_1\,p_1(f\mid P)+\beta_2\,p_2(f\mid P),
\end{aligned}
$$
with $\beta_i=\alpha_i p_i(P)/p_{\mathcal F}(P)$ and $p_{\mathcal F}(P)=\alpha_1 p_1(P)+\alpha_2 p_2(P)$. Plugging into the PME integral:
$$
\boxed{\,M_{\theta,\mathcal F}(P)=\beta_1 M_{\theta,F_1}(P)+\beta_2 M_{\theta,F_2}(P).\,}\tag{Eq. 1/3}
$$

**Why this dissolves "task recognition."** The mixture predictor is a **convex combination of per-family PMEs**, with weights $\beta_i$ equal to the *posterior probability of family $i$ given the prompt*. At $k{=}0$, $\beta_i=\alpha_i$ (the prior). As examples accumulate, the likelihood ratio $p_i(P)$ drives $\beta_{\text{correct}}\!\to\!1$ and $\beta_{\text{other}}\!\to\!0$. There is no separate "identify the task, then run its algorithm" module — **recognition and solution are the same integral**. This is the paper's principal conceptual contribution and its point of departure from the "algorithm selection" reading of Bai et al. (2023).

#### Closed-form probe: Gaussian Mixture Models (GMM)

Mixture of two dense-linear-regression classes: $f:x\mapsto w^\top x$, with $w\sim\mathcal N_d(\mu_i,\Sigma_i)$, $\mu_1=(3,0,\dots,0)$, $\mu_2=(-3,0,\dots,0)$, $\Sigma_1=\Sigma_2=\Sigma^*$ (identity with top-left entry zeroed). Equivalently a mixture-of-Gaussians prior $p_M(w)=\alpha_1\mathcal N_d(\mu_1,\Sigma_1)+\alpha_2\mathcal N_d(\mu_2,\Sigma_2)$. Because each component is a Gaussian-prior linear model, each per-family PME is available in closed form (a ridge-type posterior mean), and the mixture PME is their $\beta$-weighted blend.

**Weight-probing methodology (following Akyürek et al. 2022).** Feed the model $2d$ test inputs $\{x_i'\}$, collect predictions $\{y_i'\}$, and solve the linear system for the *implied* weight vector $w_{\text{probe}}$ that the transformer is effectively applying. Comparing $w_{\text{probe}}$ to $w$, PME(T1), PME(T2), PME(GMM) gives a *mechanistic* (not just loss-level) test.

**Findings (Figs. 1, 22–25).** Transformer loss curves lie almost exactly on PME(GMM) for prompts from either component; at $k{=}d$ examples from $T_{\text{prompt}}$ all of {Transformer, PME($T_{\text{prompt}}$), PME(GMM)} hit zero error while PME($T_{\text{other}}$) worsens. The probed weights match PME(GMM) for *all* $k$. The $\beta$'s start at the priors $\alpha$ and evolve with $k$ exactly per Eq. (3) (e.g. with $\alpha_1{=}2/3$ the mixture starts closer to $T_1$). This is direct evidence the transformer simulates the *mixture* PME, not any single-component solver or OLS.

#### Simplicity bias is inherited, not intrinsic (Fourier series)

Fourier function class $f(x)=a_0+\sum_{n=1}^N a_n\cos(n\pi x/L)+\sum_{n=1}^N b_n\sin(n\pi x/L)$, feature map $\Phi_N(x)=[1,\cos(\pi x/L),\dots,\sin(N\pi x/L)]^\top$, $f(x;\Phi_N)=w^\top\Phi_N(x)$, $w\sim\mathcal N(0,I)$, $N{=}10$, $L{=}5$, $x\sim U(-L,L)$. Implied frequencies are recovered by taking the DFT of the model's predictions on a grid.

- **MICL (single class):** *no* preferred frequency. At small $k$ all frequencies get similar magnitude; at large $k$ the model correctly zeroes coefficients above the true max frequency $M$.
- **HMICL (mixture of max-frequencies $\Phi_1,\dots,\Phi_N$):** a clear **low-frequency (simplicity) bias** at small $k$, resolving to the true frequencies at large $k$. This is exactly what Bayes predicts: frequency 1 appears in *every* mixture component while frequency $N$ appears only in $\Phi_N$, so the prior — hence posterior at low $k$ — favors low frequencies. Biasing pretraining toward high frequencies produces a *complexity* bias instead. **Conclusion:** the prior's biases are reflected in the posterior; the transformer adds no inductive bias of its own.

#### Multi-task generalization and the diversity threshold (Monomials)

Degree-2 monomial regression: feature set $S\subset M=\{(i,j):1\le i,j\le d\}$, $\Phi_S(x)=(x_ix_j)_{(i,j)\in S}$, $f(x)=w^\top\Phi_S(x)$, $w\sim\mathcal N_{|S|}(0,I)$. HMICL over $K$ fixed feature sets $S_1,\dots,S_K$ (task diversity $K$), $D{=}d{=}10$, $p{=}124$; total monomials $M_{\text{tot}}=\binom{d}{2}+\binom{d}{1}=55$, total distinct classes $\binom{55}{10}\approx 3\times10^{10}$. Baselines: OLS$_S$ (oracle basis; Bayesian predictor here), OLS$_{\Phi_M}$ (all monomials), Lasso$_{\Phi_M}$ (exploits sparsity $|S|\ll M_{\text{tot}}$), and BP$_{\text{proxy}}$ (transformer trained on the full distribution, used as an approximate Bayesian predictor).

**Result (Fig. 3):** small $K$ → poor OOD generalization (memorizes pretraining classes); increasing $K$ → OOD loss approaches OLS$_{\Phi_M}$ then Lasso$_{\Phi_M}$ and BP$_{\text{proxy}}$, and ID/OOD losses converge. Generalizing to unseen classes is a *deviation* from the pure Bayesian predictor for the pretraining distribution (which would refuse to generalize and instead fit the finite pretraining set).

#### Forgetting: generalize-then-memorize (Noisy Linear Regression)

Following Raventós et al. (2023): $d{=}8$, $p{=}15$, $f(x)=w^\top x+\epsilon$, $\epsilon\sim\mathcal N(0,\sigma^2{=}0.25)$, pretraining = uniform over $K$ fixed weight vectors, full distribution = standard Gaussian. Bayesian predictors: **dMMSE** (discrete-prior posterior mean over the finite pretraining tasks) and **Ridge** (Gaussian-prior posterior mean, the full-distribution Bayesian predictor). Four regimes by $K$:
1. $2^1$–$2^3$: no generalization, no forgetting (agrees with dMMSE).
2. $2^4$–$2^6$: some generalization, no forgetting.
3. $2^7$–$2^{11}$ ("Gaussian forgetting region"): OOD loss improves to a minimum $t_{\min}$ where it *equals* ID loss and *agrees with Ridge*, then worsens; by end of training it sits between dMMSE and Ridge. The model **generalizes to the full Gaussian distribution then forgets it**.
4. $2^{12}$–$2^{20}$: full generalization, no forgetting (agrees with Ridge).

The forgetting/transition region is the same set of diversities $\{2^7,\dots,2^{11}\}$. Forgetting scales with input dimension $d$ and is robust to hyperparameters; the paper contrasts it with grokking and links it to simplicity bias.

#### Gradient descent as tractable-approximation-to-Bayes (hypothesis)

Low-capacity transformers (one/few layers) match one/few steps of GD on the squared-error objective (Akyürek et al. 2022; von Oswald et al. 2022); Garg et al. 2022 found GD-like behavior on 2-layer-NN tasks. Panwar et al. **reconcile** this with the Bayesian view: GD is proposed to be *the best approximation to Bayesian inference achievable within the model's capacity*. Support: (i) Mingard et al. (2021) — GD solutions correlate strongly with the Bayesian predictor across architectures; (ii) for convex objectives GD provably reaches the global optimum, i.e. the Bayes-optimal predictor, as steps increase; (iii) Akyürek et al. show that *increasing* transformer capacity moves behavior *toward* Bayesian inference. So GD and Bayes are not competing accounts — GD is Bayes under a compute/capacity budget.

#### Breadth of verification (§7)

Bayesian hypothesis tested across linear inverse problems (Dense, Sparse, Sign-Vector, Low-Rank, Skewed-Covariance regression) and non-linear ones (Fourier series, degree-2 monomials, random Fourier features, Haar wavelets), in both MICL and HMICL. Where PME is closed-form → compared directly; where intractable → compared to MCMC (NUTS/HMC) posterior samples; where even sampling fails → compared to strong near-optimal baselines. Both *errors* and *implied weights* agree. Order of demonstrations is verified irrelevant for these classes (consistent with exchangeable-likelihood Bayesian inference). Architectural note (Appendix A.2): removing positional encodings greatly improves length generalization for dense/sparse regression (decoder causal mask supplies implicit position info).

#### Relevance notes for the project
This is the empirical backbone for treating a trained context-conditioned model as a Bayesian task-averager. The mixture-PME derivation ($\beta_i=$ posterior task probability, prediction = $\beta$-weighted blend) is the cleanest formal statement of "no separate task-recognition step" — directly relevant to any architecture where a shared network must serve multiple regimes and switch based on context. The forgetting/diversity-threshold results are the caution: whether a context-conditioned model *memorizes* or *generalizes* across regimes is set by pretraining task diversity and can be non-monotonic in training time.

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** ICL learns functions from in-context (x, f(x)) pairs without weight updates. Ideal LM = Bayesian predictor (samples from pretraining distribution conditioned on prompt). Extends Garg et al. 2022 MICL to hierarchical HMICL (unions of families). Five contributions: (1) multi-family ICL setup; (2) high-capacity transformers do Bayesian inference (direct + weight-probe evidence); (3) ICL inductive bias inherited from pretraining distribution (Fourier simplicity bias); (4) generalization to unseen classes given task diversity; (5) study of deviations (forgetting, GD-as-approximation).
- **§2 Background.** MICL prompt/objective; squared loss. PME = posterior mean $\mathbb E_f[f(x_{i+1})\mid P^i]$ derived by conditioning + squared-loss minimizer. §2.1 HMICL sampling process + mixture PME $M_{\theta,\mathcal F}(P)=\sum_i\beta_i M_{\theta,F_i}(P)$, $\beta_i=\alpha_i p_i(P)/p_{\mathcal F}(P)$. §2.2 model/training: GPT-style decoder, 12L/8H/256d, batch 64, 500k steps, linear input embedding, curriculum learning, standard-normal inputs.
- **§3 Transformers can in-context learn task mixtures.** §3.1 GMM (two Gaussian-prior dense-regression classes, closed-form PME); weight-probing via $2d$ test inputs. Results: transformer loss ≈ PME(GMM) for all $k$; probed weights ≈ PME(GMM); $\beta$'s evolve prior→one-hot. More complex mixtures (2–3 linear inverse problems, NN + decision-tree mixtures) also work. Implication: no separate task recognition — recognition and solution intertwined (Eq. 1).
- **§4 Simplicity bias in ICL?** Fourier series MICL (single class → no frequency bias) vs HMICL (mixture → low-frequency simplicity bias at small $k$). DFT-based frequency probing. High-frequency-biased pretraining → complexity bias. Bias comes from pretraining distribution / prior, not architecture.
- **§5 Multi-task generalization.** Degree-2 monomials; OLS$_S$ (Bayesian oracle), OLS$_{\Phi_M}$, Lasso$_{\Phi_M}$, BP$_{\text{proxy}}$. Task diversity $K\in\{10,\dots,5000\}$; ID vs OOD evaluation. Larger $K$ → OOD generalization approaching Lasso/BP$_{\text{proxy}}$ but ID degrades; deviation from pretraining-distribution Bayesian predictor. Parallels Raventós et al. 2023.
- **§6 Deviations from Bayesian inference?** §6.1 forgetting: NLR setup, four regimes by $K$; Gaussian forgetting region $\{2^7..2^{11}\}$ generalizes-then-memorizes; agreement with dMMSE (small $K$) / Ridge (large $K$) / neither (transition). §6.2 GD as tractable approximation to Bayesian inference under capacity constraints (Mingard et al., convexity, Akyürek capacity trend).
- **§7 Summary of further results.** Bayesian hypothesis verified across many linear/non-linear inverse problems in MICL + HMICL; compared to exact PME / MCMC / strong baselines; errors and implied weights agree; order-independence verified.
- **§8 Conclusion.** Bayesian perspective as unifying explanation for ICL (inductive bias from pretraining, mixture solving); generalization to new tasks = apparent deviation; open questions on forgetting, GD-Bayes relation, decision trees, real-world LLMs, hallucination/jailbreaking via implied posterior.
- **Appendix A.1** PME theoretical details + full mixture-PME derivation (reproduced above). **A.2** positional-encoding removal improves length generalization. **A.3** experimental setup (Adam, curriculum tables, 32×V100, ~30k GPU-hours, 90% bootstrap CIs). **Appendices B–D** per-family setups/results (dense/sparse/sign-vector/low-rank/skewed regression; Fourier; monomials; random Fourier features; Haar wavelets), GMM PME closed forms (§C.2), NLR forgetting analysis (§D.3).

---

## 4. Wies et al. 2023 — The Learnability of In-Context Learning

**PDF:** `docs/project/references/in_context_learning/sources/Wies et al. 2023 - The Learnability of In-Context Learning.pdf` · arXiv:2303.07895 (preprint under review).

### Phase 1: Foundational Overview (Undergraduate-Level)

**Question this paper answers (plain language).** In-context learning has a paradox at its heart: you feed a frozen language model a prompt that looks *nothing* like its training data — a raw concatenation of "input → label" pairs — and yet it learns the task, with accuracy improving as you add more pairs. How can a model learn from inputs so unlike anything it saw in pretraining? And can we *prove*, with concrete finite numbers of examples, that this works?

**Core contribution.** The first **PAC (Probably Approximately Correct) learnability framework for ICL**, and the first **finite-sample-complexity guarantees**. Prior theory (Xie et al. 2022) proved ICL works only in the limit of *infinitely many* demonstrations and only for a specific HMM data model. This paper proves it works with a *polynomial* (few-shot-realistic) number of demonstrations, for *any* mixture-of-tasks pretraining distribution satisfying mild conditions, and — crucially — while accounting for *imperfect* pretraining.

**The mechanism it proves: task identification, not task learning.** The model is viewed as pretrained on an implicit **multi-task mixture**: each document is generated by first drawing a latent "concept" (task) $\phi$ from a prior, then drawing tokens from that concept's distribution. The theorem shows that concatenating a few labeled examples in the prompt acts to **re-weight the model's prior over concepts** toward the true task — the prompt's likelihood under the correct concept dominates all others *exponentially* fast in the number of examples. So ICL "uncovers" a task the model *already learned* during pretraining rather than learning a new one.

**Key findings.**
- **Definition 1** ports PAC learning to ICL: a class of downstream tasks is *in-context learnable* after pretraining if, with enough pretraining examples ($m_{\mathcal H}$) and enough in-context examples ($m_{\tilde{\mathcal H}}$), the in-context loss comes within $\epsilon$ of the Bayes error rate with probability $\ge 1-\delta$. "Efficient" = both sample sizes polynomial in $1/\epsilon,1/\delta$.
- **Lemma 1**: the likelihood ratio of the prompt under a wrong concept vs. the true concept decays as $\exp(-k\cdot \Delta_{KL}/2)$ — exponential in the number of examples $k$ and the minimal KL divergence between concepts.
- **Theorems 1–2**: with polynomially many examples the in-context predictor equals the Bayes-optimal classifier on large-margin cases and is harmless on small-margin cases, giving **efficient in-context learnability**.
- **Robustness surprise:** the guarantees survive *randomly flipping the labels* in the prompt — consistent with the empirical finding (Min et al. 2022) that corrupting demonstration labels barely hurts ICL. This directly supports "ICL is about identifying the task, not learning the input→label map."

**Initial takeaway.** This is the theoretical anchor for "ICL = latent task identification via Bayesian prior re-weighting," upgraded from asymptotic + HMM-specific to finite-sample + general-mixture + imperfect-pretraining. It formally explains why prompts that don't resemble pretraining data still work, and why label noise in demonstrations is tolerable.

### Phase 2: Graduate-Level Deep Dive

#### The PAC framework for ICL

A pretrained function $f_\theta$ is fit to a pretraining distribution $D$ over strings from alphabet $\Sigma$. For a downstream task with distribution $\tilde D$ over (input $x$, label $y$) pairs, the prompt concatenates $k$ labeled examples with a delimiter "\n":
$$p := x_1\oplus y_1\oplus\text{"\textbackslash n"}\oplus\cdots\oplus x_k\oplus y_k\oplus\text{"\textbackslash n"}.$$
Prediction picks the label maximizing the continuation likelihood $f_\theta(p\oplus x\oplus y')$. The **in-context (0–1) loss** is
$$
L_{\text{in-context},\tilde D}:=\mathbb E_{x,y\sim\tilde D}\Big[\ell_{0-1}\big(\arg\max_{y'}f_\theta(p\oplus x\oplus y'),\,y\big)\Big].
$$

**Definition 1 (in-context learnability).** $\tilde{\mathcal H}$ is in-context learnable after pretraining on $D\in\mathcal H$ if there exist a pretraining algorithm $A$ and sample-size functions $m_{\mathcal H},m_{\tilde{\mathcal H}}$ such that for all $\epsilon,\delta>0$ and all $\tilde D\in\tilde{\mathcal H}$, whenever pretraining examples $n>m_{\mathcal H}(\epsilon,\delta)$ and in-context examples $k>m_{\tilde{\mathcal H}}(\epsilon,\delta)$, with probability $\ge 1-\delta$:
$$L_{\text{in-context},\tilde D}-\text{BayesErrorRate}\le\epsilon.$$
It is **efficiently** learnable if $m_{\mathcal H},m_{\tilde{\mathcal H}}$ are polynomial in $\epsilon^{-1},\delta^{-1}$.

**Assumption 1 (imperfect but PAC-learnable pretraining).** The pretraining algorithm produces $f_\theta$ whose next-token conditionals are uniformly $\epsilon$-close in total variation to the true ones:
$$
\max_{o_1\dots o_T\in\Sigma}\big|P_D(o_T\mid o_1\dots o_{T-1})-f_\theta(o_T\mid o_1\dots o_{T-1})\big|<\epsilon,
$$
with $m_D$ polynomial. This is the key generalization over Xie et al.: pretraining need not be *perfect*, only PAC-good.

#### The latent-concept mixture and its assumptions

A length-$T$ pretraining example is generated by first sampling a concept $\phi\sim P(\phi)$ from a family $\Phi$, then sampling tokens $\sim P_\phi(x_1,\dots,x_T)$. A downstream example takes tokens $o_1,\dots,o_T\sim P_\phi$, sets $x_t=o_t$ ($t<T$), label $y=o_T$. Three mild assumptions:

- **Assumption 2 (approximate independence across the delimiter):** $\exists\, c_1\in(0,1]$ s.t. for any strings $s_1,s_2$ and concept $\phi$,
$$c_1\le\frac{P_\phi(s_1\oplus\text{"\textbackslash n"})\cdot P_\phi(s_2)}{P_\phi(s_1\oplus\text{"\textbackslash n"}\oplus s_2)}\le\frac1{c_1}.$$
(Two paragraphs separated by "\n" are near-independent *within each component* — but crucially *not* within the mixture, which is what lets the prompt re-weight the prior.)
- **Assumption 3 (bounded token likelihood):** $\exists\, c_2>0$ s.t. $P_\phi(\sigma\mid s)>c_2$ for any string $s$, character $\sigma$, concept $\phi$. Prevents zero-probability prompts from the unnatural concatenation.
- **Assumption 4 (strictly positive prior):** $\exists\, c_3>0$ s.t. $P_D(\phi)\ge c_3$ for every concept. Without it, a rare task is unrecognizable.

#### Lemma 1 — exponential concentration of the prompt likelihood ratio

**Claim.** For a downstream concept $\phi^\star$ with $\Delta_{KL}>8\log\frac1{c_1 c_2}$, there is a polynomial $m_{\tilde D}$ s.t. for $k\ge m_{\tilde D}(\epsilon,\delta)$ in-context examples, $\frac{P_\phi(p)}{P_{\phi^\star}(p)}<\epsilon$ w.p. $\ge 1-\delta$ — and this **survives random label flipping**. $m_{\tilde D}$ is polynomial in $\log\frac1\delta,\log\frac1\epsilon,\log\frac1{c_1 c_2},\frac1{\Delta_{KL}},T$.

**Derivation sketch (Appendix A).** By approximate independence (Assumption 2 applied $k$ times), the prompt factorizes up to a controlled constant:
$$
c_1^k\le\frac{\prod_{i=1}^k P_\phi(x_i\oplus y_i\oplus\text{"\textbackslash n"})}{P_\phi(p)}\le c_1^{-k}.
$$
Assumption 3 bounds the delimiter-induced drift $P_\phi(x_i\oplus y_i)\le P_\phi(x_i\oplus y_i\oplus\text{"\textbackslash n"})\le P_\phi(x_i\oplus y_i)c_2$ and the label-flip drift $c_2<\frac{P_\phi(x_i\oplus\tilde y_i)}{P_\phi(x_i\oplus y_i)}<\frac1{c_2}$, giving $(c_1 c_2^2)^k\le\frac{\prod_i P_\phi(x_i\oplus\tilde y_i)}{P_\phi(p)}\le(c_1 c_2^2)^{-k}$. Hence the log-ratio of prompt probabilities is
$$
\log\frac{P_\phi(p)}{P_{\phi^\star}(p)}\le\sum_{i=1}^k\log\frac{P_\phi(x_i\oplus\tilde y_i)}{P_{\phi^\star}(x_i\oplus\tilde y_i)}+4k\log\frac1{c_1 c_2}.
$$
Each summand is i.i.d. with expectation equal to *minus* the KL divergence:
$$
\mathbb E_p\Big[\tfrac1k\sum_{i=1}^k\log\tfrac{P_\phi(x_i\oplus\tilde y_i)}{P_{\phi^\star}(x_i\oplus\tilde y_i)}\Big]=-\mathrm{KL}(P_{\phi^\star},P_\phi).
$$
Each term is bounded ($|\cdot|\le T\log\frac1{c_2}$), so **Hoeffding's inequality** gives, w.p. $\ge 1-\delta$,
$$
\frac{P_\phi(p)}{P_{\phi^\star}(p)}<\exp\!\Big(-\tfrac k2\big(\mathrm{KL}(P_{\phi^\star},P_\phi)-8\log\tfrac1{c_1 c_2}\big)\Big),
$$
valid for $k>(\log\tfrac1\delta)(16T^2)\big(\log^2\tfrac1{c_2}\big)/\mathrm{KL}(P_{\phi^\star},P_\phi)^2$. Choosing $m_{\tilde D}$ as the max of this and $\frac{2\log\frac1\epsilon}{\Delta_{KL}-8\log\frac1{c_1 c_2}}$ yields the claim. The two ingredients: **independence** → multiplicative (hence exponential-in-$k$) effect; **KL measures log-probabilities** → exponential in $\Delta_{KL}$. $\square$

**This is the Bayesian mechanism made quantitative.** The posterior over concepts is $P_D(\phi\mid p)\propto P_D(\phi)P_\phi(p)$; Lemma 1 says every wrong concept's posterior weight decays like $e^{-k\Delta_{KL}/2}$ relative to $\phi^\star$. So the prompt **re-weights the prior onto the true task exponentially fast** — the formal content of "ICL identifies the latent task."

#### Theorem 1 — the in-context margin is preserved

Define the downstream margin $\Delta(x,y,\tilde y)=P_{\tilde D}(y\mid x)-P_{\tilde D}(\tilde y\mid x)$ and the in-context margin $\Delta(p,x,y,\tilde y)=P_D(y\mid p\oplus x)-P_D(\tilde y\mid p\oplus x)$. **Claim:** for margin $\Delta(x,y,\tilde y)>0$ and $k\ge m_{\tilde D}(\Delta/2,\delta)$, w.p. $\ge1-\delta$,
$$\Delta(p,x,y,\tilde y)>\tfrac12\Delta(x,y,\tilde y)+c_1^2-1,$$
robust to label flips.

**Derivation sketch (Appendix B).** Write the label-likelihood difference as
$$
P_D(y\mid p\oplus x)-P_D(\tilde y\mid p\oplus x)=\frac{\sum_\phi P_D(\phi)\big[P_\phi(p\oplus x\oplus y)-P_\phi(p\oplus x\oplus\tilde y)\big]}{\sum_\phi P_D(\phi)P_\phi(p\oplus x)}.
$$
Assumption 2 factorizes $P_\phi(p\oplus x\oplus y)$ into $P_\phi(p)\cdot P_\phi(x\oplus y)$ up to $c_1^{\pm1}$. Split numerator/denominator into the $\phi^\star$ term ($A,C$) and the rest ($B,D$). Lemma 1 makes $B/C$ and $D/C$ negligible (each $\lesssim\sum_{\phi\ne\phi^\star}\frac{P_D(\phi)}{P_D(\phi^\star)}\frac{P_\phi(p)}{P_{\phi^\star}(p)}$, controlled via Assumptions 3,4). The dominant ratio is $A/C=\Delta(x,y,\tilde y)+(c_1^2-1)P_\phi(y\mid x)+(c_1^{-2}-1)P_\phi(\tilde y\mid x)>\Delta(x,y,\tilde y)-1+c_1^2$. Bounding $|B/C|<\tfrac15\Delta$ and $|D/C|<\tfrac14$ gives $\Delta(p,x,y,\tilde y)>\tfrac12\Delta(x,y,\tilde y)+c_1^2-1$. So **at least half the downstream margin is preserved** in-context. $\square$

#### Theorem 2 — efficient in-context learnability

**Claim.** If every downstream margin $\Delta_{\tilde D}\ge 4(1-c_1^2)$ and the minimal inter-concept KL $>\log\frac1{c_1 c_2}$, then $\tilde{\mathcal H}$ is *efficiently* in-context learnable after pretraining on $D$.

**Derivation sketch.** Set pretraining accuracy to $8\times$ the target ICL accuracy, i.e. $\Delta_{\text{pretraining}}=\epsilon/8$. Split test examples by margin. **Case 1 (margin $\ge 8\Delta_{\text{pretraining}}$):** Theorem 1 gives in-context margin $>\tfrac12\Delta+c_1^2-1\ge2\Delta_{\text{pretraining}}$ (using $\Delta\ge\Delta_{\tilde D}>4(1-c_1^2)$), so the imperfect model $f_\theta$'s prediction equals the ground-truth in-context predictor equals the Bayes classifier — zero contribution to loss. **Case 2 (small margin $<8\Delta_{\text{pretraining}}$):** the Bayes error is already high there, so even a wrong prediction contributes $<\epsilon$. Combining, every $x$ contributes $\le\epsilon$, and both sample sizes are polynomial (from Assumption 1 and Lemma 1). $\square$

**Reading the result.** ICL succeeds by *task recognition*: once $k$ is large enough, the prompt collapses the concept posterior onto $\phi^\star$, the in-context predictor becomes the Bayes-optimal classifier on the informative (large-margin) inputs, and is provably harmless on the ambiguous ones. Imperfect pretraining is absorbed by demanding $8\times$ tighter pretraining accuracy. Label-flip robustness (carried through every step) matches the empirical "ground-truth labels barely matter" phenomenon — because what the prompt conveys is *which task*, not the input→label mapping itself.

#### Relevance notes for the project
Wies et al. is the rigorous complement to Wang et al. (real-LLM latent-variable account) and Panwar et al. (empirical Bayesian-predictor tests): it *proves*, with finite samples and imperfect models, that a mixture-of-tasks pretraining objective yields ICL-by-task-identification, with an explicit exponential ($e^{-k\Delta_{KL}/2}$) rate of posterior concentration. For the project, the transferable formal object is the **prompt-likelihood-ratio concentration bound**: any frozen context-conditioned model built on a latent-mixture generative story will identify the operative regime at a rate set by (i) number of context examples and (ii) KL separation between regimes — and will tolerate label corruption because identification, not mapping-learning, is the mechanism.

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** ICL emerged at scale (Brown et al. 2020): frozen LM specialized by concatenated examples, accuracy rising with example count, though prompts don't resemble pretraining data. No formal definition of ICL learning existed. Proposes PAC definition + first finite sample-complexity results. Frames pretraining as *implicit unsupervised multi-task learning* (GPT2 lineage): tasks are latent, not labeled, so pretraining ≠ fine-tuning. Contrasts Xie et al. 2022 (HMM-specific, infinite examples, perfect pretraining) — this work is general-mixture, polynomial-sample, imperfect-pretraining. Empirical support: label-randomization robustness (Min et al. 2022), irrelevant-text-mimicking-task effects (Webson & Pavlick; Lampinen et al.).
- **§2 A PAC learnability framework for ICL.** Prompt construction Eq. (1); in-context 0–1 loss Eq. (2); Definition 1 (in-context learnable / efficiently learnable, Eq. 3); Assumption 1 (pretraining PAC-close in TV, Eq. 4). Central question: how do frozen models learn from prompts unlike pretraining data?
- **§3 Guarantees on in-context learning.** §3.1 latent-concept hypothesis class: pretraining = mixture over concepts $\phi\sim P(\phi)$, tokens $\sim P_\phi$; downstream example = concept's tokens with last token as label; distribution drift from concatenation acknowledged. Assumptions 2 (approx. independence across "\n", Eq. 5), 3 (token likelihood lower bound $c_2$, Eq. 6), 4 (prior lower bound $c_3$). §3.2 guarantees via latent-concept inference: two-scenario proof strategy (large-margin: deviations negligible; small-margin: Bayes error high anyway). Lemma 1 (exponential likelihood-ratio decay in $k$ and $\Delta_{KL}$, label-flip robust). Theorem 1 (in-context margin $\ge$ half downstream margin). Theorem 2 (efficient in-context learnability given margin $\ge4(1-c_1^2)$ and min-KL $>\log\frac1{c_1c_2}$).
- **§4 Related Work.** PAC + distribution-dependent bounds lineage (Valiant; Benedek & Itai; etc.); prior pretraining-benefit theory learns some weights (Saunshi et al., Wei et al.) whereas this work uses frozen models. Function-class ICL works (Garg et al., Akyürek et al., Laskin et al. RL, Levine et al. inductive bias, Li et al. generalization bound) all bake demonstrations into pretraining, unlike here. Mechanistic works (induction heads Olsson et al.; GD-expressivity Akyürek/Dai/von Oswald) lack learnability guarantees. Chan et al. — data-distributional drivers, empirical. Xie et al. closest but HMM-specific, asymptotic, perfect-pretraining.
- **§5 Conclusion.** Latent-multitask pretraining explains ICL from non-resembling prompts; first finite (polynomial) sample-complexity guarantees. Open: model-size ↔ ICL-efficiency link; extension to tasks *not* in the pretraining mixture (Wei et al. 2023).
- **Appendix A — Proof of Lemma 1.** Explicit $P_\phi(p)$ factorization (Eqs. 8–11), log-ratio bound (Eq. 12), KL-expectation identity (Eq. 13), boundedness (Eq. 14), Hoeffding concentration (Eq. 15), choice of $m_{\tilde D}$.
- **Appendix B — Proof of Theorem 1.** Label-likelihood difference (Eq. 16), independence factorization (Eqs. 17–19), $A/B/C/D$ decomposition (Eqs. 20–24), triangle-inequality bounds on $B/C$ and $D/C$ (Eqs. 29–31) using Assumptions 3,4, final margin bound (Eq. 28).

---

## 5. Elmoznino et al. 2025 — In-Context Learning and Occam's Razor

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

## 6. Mittal et al. 2025 — Does Learning the Right Latent Variables Necessarily Improve In-Context Learning?

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

## 7. Hu et al. 2024 — Amortizing Intractable Inference in Large Language Models

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

# Part 2 — §2 ICL as gradient descent / learning algorithms inside transformers

*What computation actually runs in the forward pass: the papers showing a frozen transformer secretly executes a learning algorithm — most concretely one step of (preconditioned) gradient descent per attention layer — with expressivity constructions, algorithm identification and selection, optimization-landscape and training-dynamics proofs, and the empirical match to optimal estimators.*

---

## 8. Garg et al. 2022 — What Can Transformers Learn In-Context? A Case Study of Simple Function Classes

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

## 9. von Oswald et al. 2023 — Transformers Learn In-Context by Gradient Descent

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

## 10. Akyürek et al. 2023 — What Learning Algorithm Is In-Context Learning? (Investigations with Linear Models)

**PDF:** `docs/project/references/in_context_learning/sources/Akyurek et al. 2023 - What learning algorithm is in-context learning.pdf`
**Venue:** ICLR 2023. **Authors:** Ekin Akyürek, Dale Schuurmans, Jacob Andreas, Tengyu Ma, Denny Zhou (Google Research / MIT CSAIL / Stanford).

### Phase 1: Foundational Overview (Undergraduate-Level)

**The plain-language question.** When you give a large language model a few worked examples in its prompt — pairs of an input `x` and its answer `f(x)` — and it then correctly answers a new input, *how* is it doing that? The model's weights never change. So where does the "learning" happen? This paper's hypothesis: **the transformer builds a small model inside its own activations (its scratch-pad of intermediate numbers) and trains that small model on the fly**, using an algorithm it discovered during pre-training. In-context learning (ICL) would then be a familiar textbook estimator running silently inside the network.

**The testbed.** To make this checkable, the authors use the simplest interesting learning problem: **linear regression**. The true function is $f(x) = w^\top x$; the model sees a sequence of $(x_i, y_i)$ pairs and must predict $y$ for a fresh $x$. Linear regression is ideal because it has many known solution algorithms (gradient descent, ridge regression, exact least squares, the Bayesian optimum), and they give *different* answers when data are scarce or noisy — so you can tell which one the transformer is imitating.

**Three lines of evidence.**
1. **It is possible in principle (proof by construction).** They hand-build transformer weights that provably execute one step of gradient descent (needing only about `d` hidden units for a `d`-dimensional problem) and, separately, one step of exact ridge-regression updating via the Sherman–Morrison formula (needing about `d²` units). So the architecture is expressive enough to *be* these algorithms.
2. **Trained models actually behave like these algorithms.** A transformer trained from scratch on the ICL objective produces predictions that closely match ordinary least squares (OLS) on clean data, and match the **minimum-Bayes-risk ridge predictor** when the data are noisy. As you dial up depth, the model transitions through **algorithmic "phases"**: 1-layer ≈ one gradient step, 2–4 layers ≈ ridge regression, 8+ layers ≈ exact OLS.
3. **You can read the intermediate quantities out of its activations.** Using a trained "probe," they recover the moment matrix $X^\top Y$ and the least-squares weight vector $w_{\mathrm{OLS}}$ from hidden states — and $X^\top Y$ appears in early layers, $w$ in later layers, mirroring the order a real algorithm would compute them.

**Initial takeaway.** ICL — at least in this controlled linear setting — is *not* mysterious. The transformer appears to rediscover and run standard estimation algorithms. The specific algorithm it lands on depends on how much capacity (depth, width) it has and how noisy the data are. This reframes "emergent" ICL as ordinary machine learning happening one level down.

### Phase 2: Graduate-Level Deep Dive

Phase 2 foregrounds the **algorithm-implementation claim**: exactly how fixed transformer weights simulate (a) a gradient-descent step and (b) a Sherman–Morrison ridge update, and how the empirical "phase transition" and Bayesian-matching results are established.

#### Setup and encoding

Function class $\mathcal{F} = \{f(x) = w^\top x : w, x \in \mathbb{R}^d\}$; squared loss $L(y, y') = (y-y')^2$. The (regularized) objective and its closed-form minimizer are

$$
\sum_i L(w^\top x_i, y_i) + \lambda \|w\|_2^2 \quad\Longrightarrow\quad w^\star = (X^\top X + \lambda I)^{-1} X^\top y .
$$

$\lambda = 0$ gives OLS; $\lambda > 0$ gives ridge. Each example is embedded as a $(d{+}1)$-dimensional token: $\tilde x_i = [0, x_i]$ and $\tilde y_i = [y_i, 0_d]$, so inputs and labels alternate in a single token stream. The ICL training objective is autoregressive:

$$
\arg\min_\theta\; \mathbb{E}_{\substack{x_1,\dots,x_n \sim p(x)\\ f \sim p(f)}}\left[\sum_{i=1}^n L\big(f(x_i),\, T_\theta([x_1, f(x_1), \dots, x_i])\big)\right].
$$

#### Computational primitives (Lemma 1)

The constructions are built from four operations, each realizable by **a single decoder layer** (self-attention + feed-forward as in Eqs. 1–5 of the paper):
- `mov` — copy a block of rows from one column (token) into another (data movement across positions, done by attention).
- `mul` — interpret two row-blocks of a column as matrices $A_1$ ($a\times b$) and $A_2$ ($b\times c$) and write $A_1 A_2$ into a target block.
- `div` — divide a block by the absolute value of a scalar entry (used for the Sherman–Morrison denominator).
- `aff` — apply an affine map $W_1 h_{i:j} + W_2 h_{i':j'} + b$ to row-blocks.

That these four are single-layer-expressible is the workhorse lemma; everything else is composition.

#### Theorem 1 — one gradient-descent step in $O(d)$ hidden space

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

#### Theorem 2 — one exact ridge update via Sherman–Morrison in $O(d^2)$ hidden space

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

#### What algorithm is actually learned — behavioral metrics

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

#### Probing for intermediate quantities

Freeze the trained model; train an auxiliary probe with position-attention:
$$
\alpha = \mathrm{softmax}(sv), \qquad \hat v = \mathrm{FF}_v\big(\alpha^\top W_v H^{(l)}\big),
$$
minimizing $\|v - \hat v\|^2$ against targets $v \in \{X^\top Y,\, w_{\mathrm{OLS}}\}$. A 2-layer MLP probe beats a linear probe (targets are encoded **non-linearly**, unlike the linear hand construction), both targets decode accurately in deep layers, and — matching how a real algorithm proceeds — the moment matrix $X^\top Y$ becomes decodable early (~layer 7) and the weight vector $w$ later (~layer 12).

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** Poses the "how, not what" question of ICL. Hypothesis: transformers encode an implicit model in activations and train it in-context. Preview of the three evidence strands; states the capacity results ($O(d)$ hidden + constant depth → one GD step; $O(d^2)$ → one ridge update).
- **§2 Preliminaries.** ICL as meta-learning reduced to ordinary supervised learning. §2.1 defines the decoder transformer: self-attention $a_i = \mathrm{Attention}(h_i; W^F, W^Q, W^K, W^V)$ with softmax heads (Eqs. 1–3), feed-forward $\mathrm{FF}(a_i; W_1, W_2)$ with GeLU nonlinearity and layer-norm (Eqs. 4–7). §2.2 gives the autoregressive ICL objective (Eq. 8). §2.3 sets up linear regression, the closed-form solution (Eq. 10), and the $(d{+}1)$-D token encoding.
- **§3 What learning algorithms can a transformer implement?** §3.1 defines the four primitives `mov`/`mul`/`div`/`aff` and Lemma 1 (each is one decoder layer). §3.2 Theorem 1: constant depth + $O(d)$ hidden space implements one GD step (Eq. 11). §3.3 Theorem 2: Sherman–Morrison identity (Eq. 13) yields a rank-one ridge update (Eq. 14) in constant depth + $O(d^2)$ hidden space. Discussion contrasts with universality results (Yun et al.; Schlag et al. linear self-attention) and notes single-step → multi-step by layer-stacking.
- **§4 What computation does an in-context learner perform?** §4.1 behavioral metrics SPD (Eq. 15) and ILWD (Eqs. 16–17). §4.2 experimental setup (hyperparameter sweep; $L=16, H=512, M=4$ best; 500k iterations; 40 pairs per context; Gaussian $w, x$). §4.3 results: ICL ≈ OLS on noiseless data (Fig. 1); ICL ≈ minimum-Bayes-risk ridge on noisy data (Eqs. 18–19, Fig. 2); algorithmic phase transitions with depth and width (Fig. 3).
- **§5 Does ICL encode meaningful intermediate quantities?** Probing methodology (Eqs. 20–21). Recovers $X^\top Y$ and $w_{\mathrm{OLS}}$ non-linearly from deep layers; $X^\top Y$ decodable earlier than $w$; control task (fixed $w=1$) probes worse, confirming non-triviality (Fig. 4).
- **§6 Conclusion.** Transformers can in theory implement multiple linear-regression algorithms, empirically implement a range of them (transitioning with capacity and noise), and expose probeable intermediates. Extensible to richer function classes via nonlinear feature layers and to large-scale LMs.
- **Appendices A–E.** Proofs of Lemma 1 and Theorems 1–2 (explicit primitive-by-primitive traces — the GD trace was used above), training hyperparameters, and probing details.

---

## 11. Dai et al. 2023 — Why Can GPT Learn In-Context? (Language Models Secretly Perform Gradient Descent as Meta-Optimizers)

**PDF:** `docs/project/references/in_context_learning/sources/Dai et al. 2023 - Why Can GPT Learn In-Context (Secretly Gradient Descent).pdf`
**Venue:** ACL 2023 Findings. **Authors:** Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Shuming Ma, Zhifang Sui, Furu Wei (Peking Univ. / Tsinghua / Microsoft Research).

### Phase 1: Foundational Overview (Undergraduate-Level)

**The plain-language question.** Akyürek et al. work with a toy linear-regression transformer trained from scratch. Dai et al. ask the same "how does ICL work" question but for **real, off-the-shelf GPT models on real NLP tasks** (sentiment, topic classification, natural-language inference). Their answer: **the attention mechanism has a hidden "dual form" that is mathematically equivalent to gradient descent**, so a frozen GPT reading demonstration examples is effectively *fine-tuning itself without a backward pass*. They call the network a **meta-optimizer**: it produces **meta-gradients** from the demonstrations by ordinary forward computation, and applies them through attention.

**The core analogy.**
- **Fine-tuning:** compute gradients by back-propagation, add $\Delta W_{\mathrm{FT}}$ to the weights.
- **In-context learning:** the attention over demonstration tokens produces an implicit weight update $\Delta W_{\mathrm{ICL}}$ that plays the same role — but it is built by *forward* computation, no back-prop.

So ICL is "**implicit fine-tuning**." The two share a "dual view" of gradient descent.

**How they check it.** On six classification datasets with GPT-1.3B and GPT-2.7B, they compare ICL against a deliberately matched fine-tuning baseline (same examples, one step each, same order, only key/value projections updated) along four axes:
1. **Predictions:** ICL correctly reproduces ~85–89% of the corrections fine-tuning makes over zero-shot (their "Rec2FTP" metric).
2. **Attention-output representations** move in the *same direction* as fine-tuning's (their "SimAOU" cosine ≈ 0.19–0.23, versus ≈ 0 for random updates).
3. **Attention weights to the query** become more like a fine-tuned model's than an un-tuned model's (their "SimAM").
4. **Attention to training tokens** is rank-correlated with fine-tuning's (their "Kendall" ≈ 0.19–0.21, versus ≈ 0 for random).

**A design payoff.** If attention values are "meta-gradients," you can borrow tricks from real optimizers. They add **momentum** (an exponential moving average of attention value vectors) to attention and get consistent gains — lower language-model perplexity and +2.8 average accuracy on ICL tasks — which they treat as further evidence the meta-optimizer view is real and useful.

**Initial takeaway.** This paper is the "real-model, real-task" complement to Akyürek. It does not prove an exact algorithm (it uses an approximation — linear attention without softmax), but it gives a clean equation showing attention ≡ a gradient-descent weight update, plus behavioral evidence that ICL and fine-tuning line up. The momentum-attention result shows the analogy can *guide architecture design*.

### Phase 2: Graduate-Level Deep Dive

The algorithm-implementation claim here is the **duality between (linear) attention and a gradient-descent weight update**. This is more of a formal *analogy* than Akyürek's exact construction or Bai's approximation-with-error-bound, but the derivation is the crux.

#### The dual form of a linearly-updated linear layer (Irie et al. 2022 lineage)

Start from a linear layer whose weights receive a gradient-descent update. Let $W_0, \Delta W \in \mathbb{R}^{d_{\mathrm{out}} \times d_{\mathrm{in}}}$ be the initial and update matrices, $x \in \mathbb{R}^{d_{\mathrm{in}}}$ the input:

$$
\mathcal{F}(x) = (W_0 + \Delta W)\,x. \tag{7}
$$

In back-propagation, $\Delta W$ is the sum of outer products of historical inputs $x_i'$ and their **error signals** $e_i = -\gamma\,g_i$ (learning rate $\gamma$ times the output gradient $g_i$):

$$
\Delta W = \sum_i e_i \otimes x_i'. \tag{8}
$$

**Derivation of the dual form.** Substitute (8) into (7):
$$
\mathcal{F}(x) = W_0 x + \Delta W x
= W_0 x + \Big(\sum_i e_i \otimes x_i'\Big) x
= W_0 x + \sum_i e_i \big(x_i'^\top x\big)
= W_0 x + \mathrm{LinearAttn}(E, X', x). \tag{9}
$$

The last step uses $(e_i \otimes x_i')\,x = e_i (x_i'^\top x)$: the outer product acting on $x$ collapses to the error vector $e_i$ scaled by the inner product $x_i'^\top x$. **Reading the result as attention:** the historical errors $E$ are the *values*, the historical inputs $X'$ are the *keys*, and the current input $x$ is the *query*. So "a linear layer trained by GD" $\equiv$ "$W_0 x$ plus a linear-attention lookup." This is the identity the whole paper hangs on.

#### Applying the duality to Transformer attention (Section 3.1)

Let $x$ be the query-token representation and $q = W_Q x$ the attention query. In the ICL setting, with $X'$ the demonstration-token representations and $X$ the preceding query tokens, standard attention is

$$
\mathcal{F}_{\mathrm{ICL}}(q) = \mathrm{Attn}(V, K, q) = W_V [X'; X]\,\mathrm{softmax}\!\left(\frac{(W_K [X'; X])^\top q}{\sqrt{d}}\right). \tag{10}
$$

**Approximation step (the paper's main caveat).** Drop the softmax and scaling factor to get *relaxed linear attention*:

$$
\tilde{\mathcal{F}}_{\mathrm{ICL}}(q) \approx W_V [X'; X]\,(W_K [X'; X])^\top q
= \underbrace{W_V X (W_K X)^\top}_{\text{query part}} q + \underbrace{W_V X' (W_K X')^\top}_{\text{demonstration part}} q. \tag{11}
$$

Define $W_{\mathrm{ZSL}} := W_V X (W_K X)^\top$ — the zero-shot-learning weights, i.e. what attention would produce with *no* demonstrations. Then, running the duality (9) *in reverse* on the demonstration part:

$$
\tilde{\mathcal{F}}_{\mathrm{ICL}}(q)
= W_{\mathrm{ZSL}} q + \sum_i \big(W_V x_i'\big)\big(W_K x_i'\big)^\top q
= W_{\mathrm{ZSL}} q + \sum_i \big((W_V x_i') \otimes (W_K x_i')\big) q
= \big(W_{\mathrm{ZSL}} + \Delta W_{\mathrm{ICL}}\big) q. \tag{12}
$$

**Interpretation.** The demonstration tokens contribute a weight update $\Delta W_{\mathrm{ICL}} = \sum_i (W_V x_i') \otimes (W_K x_i')$ acting on the initial ZSL weights — exactly the structure of (7)–(8). Here $W_V X'$ plays the role of the meta-gradient values $E$, and $W_K X'$ the role of the historical inputs $X'$. So **attention to demonstrations = an implicit parameter update built by forward computation.**

The three-line summary the paper repeats: (1) pretrained GPT is a meta-optimizer; (2) it produces meta-gradients from demonstrations via forward computation; (3) attention applies those meta-gradients to build the ICL model.

#### Comparison with an explicit fine-tuning update (Section 3.2)

A fine-tuning update to only the key/value projections gives, in the same relaxed-linear form,

$$
\tilde{\mathcal{F}}_{\mathrm{FT}}(q) = (W_V + \Delta W_V)\,X X^\top (W_K + \Delta W_K)^\top q = (W_{\mathrm{ZSL}} + \Delta W_{\mathrm{FT}})\,q. \tag{13}
$$

Comparing (12) and (13): both add an update to $W_{\mathrm{ZSL}}$; $\Delta W_{\mathrm{ICL}}$ comes from forward computation, $\Delta W_{\mathrm{FT}}$ from back-propagation. The paper enumerates four shared properties justifying "ICL = implicit fine-tuning": **both perform gradient descent**; **same training information** (both driven by the demonstration examples); **same causal order** (decoder-only attention + one-epoch same-order fine-tuning both prevent later examples affecting earlier); **both act on attention** (keys/values only).

#### Empirical alignment metrics (Section 4)

- **Rec2FTP** (recall to fine-tuning prediction): among query examples fine-tuning fixes over ZSL, the fraction ICL also fixes: $\dfrac{N_{(\mathrm{FT}>\mathrm{ZSL}) \wedge (\mathrm{ICL}>\mathrm{ZSL})}}{N_{\mathrm{FT}>\mathrm{ZSL}}}$. Averages 85.6% (1.3B) / 89.4% (2.7B).
- **SimAOU** (similarity of attention-output updates): cosine between $h^{(l)}_{\mathrm{ICL}} - h^{(l)}_{\mathrm{ZSL}}$ and $h^{(l)}_{\mathrm{FT}} - h^{(l)}_{\mathrm{ZSL}}$. ~0.19–0.23 vs. ~0 for random updates.
- **SimAM** (similarity of attention maps to the query): ICL's pre-softmax attention weights are closer to the *post*-fine-tuning model's than to the *pre*-fine-tuning model's.
- **Kendall (ICL, FT)** rank correlation of attention to training tokens: $\dfrac{P_c - P_d}{N(N-1)/2}$ ≈ 0.19–0.21 vs. ~0 random. ($P_c, P_d$ = concordant/discordant pairs.)

#### Momentum-based attention (Section 5)

Gradient descent with momentum averages past gradients:

$$
\Theta_t = \Theta_{t-1} - \gamma \sum_{i=1}^{t-1} \eta^{t-i} \nabla f_{\Theta_i}, \tag{14}
$$

with $\eta \in (0,1)$. Since attention **values** are the meta-gradients, apply an exponential moving average (EMA) to them:

$$
\mathrm{MoAttn}(V, K, q_t) = \mathrm{Attn}(V, K, q_t) + \mathrm{EMA}(V)
= V\,\mathrm{softmax}\!\Big(\tfrac{K^\top q_t}{\sqrt d}\Big) + \sum_{i=1}^{t-1} \eta^{t-i} v_i,
$$

where $v_i$ is the $i$-th value vector. Results: consistent perplexity improvement in language modeling (e.g. 15.14 → 15.02 at length 1024) and +2.8 average ICL accuracy across six datasets. The paper frames this as the analogy *paying rent* — the meta-optimizer view is not just interpretive but design-generative.

**Stated limitations.** The duality is derived for *relaxed linear attention* (no softmax); full softmax attention "may be more complex." Analysis is limited to ≤2.7B models and classification tasks.

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** ICL as meta-optimization; the dual form between attention and gradient descent; ICL = implicit fine-tuning. Four contributions: (i) the duality + meta-optimization view, (ii) ICL↔fine-tuning connection, (iii) empirical evidence, (iv) momentum-attention design.
- **§2 Background.** §2.1 ICL for classification with GPT: predict $\hat y = \arg\max_{y_j} P_M(y_j \mid C, x)$ over a candidate answer set, with template formatting $T(x,y)$ and logits $l_j = M(I) \cdot e_{y_j}$ (Eqs. 1–6). §2.2 the dual form of a GD-trained linear layer as linear attention (Eqs. 7–9), citing Aizerman 1964 and Irie et al. 2022.
- **§3 Understanding ICL as implicit fine-tuning.** §3.1 relaxed-linear approximation of attention (Eq. 10→11), definition of $W_{\mathrm{ZSL}}$, reverse-duality to $\Delta W_{\mathrm{ICL}}$ (Eq. 12), meta-optimization summary. §3.2 the matched fine-tuning baseline (Eq. 13) and four shared properties.
- **§4 Experiments.** §4.1 settings (GPT-1.3B / 2.7B from fairseq; 32 demonstrations; SGD one-epoch fine-tuning on the same examples). §4.2 six datasets (SST2/SST5/MR/Subj/AGNews/CB) with ZSL/FT/ICL accuracies (Table 1). §4.3 Rec2FTP (Table 2). §4.4 SimAOU (Table 3). §4.5 SimAM (Table 4). §4.6 Kendall correlation of attention to training tokens (Table 5).
- **§5 Momentum-based attention.** GD-with-momentum (Eq. 14); EMA over attention values; perplexity gains (Table 6) and ICL accuracy gains (Table 7).
- **§6 Related work.** Contrasts with Xie et al. (implicit Bayesian inference), Olsson et al. (induction heads), Garg et al. / Akyürek et al. / von Oswald et al. (GD in trained-from-scratch regression transformers). Positions this work as the first to analyze *off-the-shelf* GPTs on *real* tasks.
- **§7 Conclusion + Limitations.** Duality → meta-optimization → implicit fine-tuning; momentum design. Limits: relaxed-linear (softmax-free) analysis; ≤2.7B; classification only.
- **Appendices A–C.** Templates and candidate answer sets; random-seed and learning-rate grids; from-scratch LM hyperparameters (350M, 24 layers).

---

## 12. Bai et al. 2023 — Transformers as Statisticians (Provable In-Context Learning with In-Context Algorithm Selection)

**PDF:** `docs/project/references/in_context_learning/sources/Bai et al. 2023 - Transformers as Statisticians.pdf`
**Venue:** NeurIPS 2023 (spotlight). **Authors:** Yu Bai, Fan Chen, Huan Wang, Caiming Xiong, Song Mei (Salesforce AI Research / Peking Univ. / UC Berkeley).

### Phase 1: Foundational Overview (Undergraduate-Level)

**The plain-language question.** Akyürek and Dai each show a transformer can *be* one algorithm (gradient descent, ridge). Bai et al. push much further and ask: **can one transformer act like a whole statistician — implement a toolbox of standard estimators, and, crucially, pick the right one for the data it is handed, all inside a single forward pass with no hint about which task it is facing?** They answer yes, with **provable, quantitative guarantees** (explicit bounds on layers, heads, and pretraining sample size), and they confirm it experimentally.

**Two big deliverables.**
1. **A comprehensive expressivity theory.** They prove a single transformer can run — in context — least squares, **ridge regression, Lasso** (sparse regression), **convex risk minimization for generalized linear models** (which includes logistic regression for classification), and **gradient descent on two-layer neural networks**. Each construction has mild size bounds (constant heads, logarithmically many layers for the convex cases) and achieves near-optimal prediction accuracy. Underneath them all is one reusable engine: an efficient **in-context gradient descent** construction where one attention layer ≈ one GD step, and the approximation error grows only *linearly* in the number of steps.
2. **In-context algorithm selection.** The headline conceptual contribution. A single transformer can **adaptively choose** which base algorithm to run depending on the input sequence — with no prompt telling it which. Two mechanisms:
   - **Post-ICL validation:** internally split the demonstrations into a train and a validation part, run several candidate algorithms (e.g. ridge with different regularization strengths) on the train part, score each on the validation part, and keep the best. This is textbook model selection, executed inside attention.
   - **Pre-ICL testing:** inspect a cheap summary statistic of the input to decide the task first — e.g. "are the labels all 0/1? then do logistic regression; otherwise do linear regression."

**The flagship application.** Using post-ICL validation, they build a transformer that is **nearly Bayes-optimal on noisy linear regression with *mixed* noise levels** — a harder task than any prior ICL-theory paper handled, where the optimal answer is a data-dependent mixture of ridge regressions. Experiments (Figure 2/4) show one "algorithm-selecting" transformer simultaneously matching the individual Bayes error on two different noise levels, beating any single fixed-λ ridge.

**A pretraining guarantee.** They also prove you can *learn* these transformers from **polynomially many** pretraining sequences, giving the first end-to-end ICL story: expressivity + statistical prediction power + sample complexity of pretraining.

**Initial takeaway.** This is the most ambitious of the three. Where Akyürek says "ICL ≈ a linear estimator" and Dai says "ICL ≈ implicit fine-tuning," Bai says "ICL ≈ a full adaptive statistical decision procedure, provably, and you can train it efficiently." The algorithm-selection result is the key new idea: transformers aren't locked into one algorithm — they can meta-select.

### Phase 2: Graduate-Level Deep Dive

Phase 2 foregrounds (i) the **in-context gradient-descent engine** with its linear-error-accumulation guarantee, and (ii) the **two algorithm-selection mechanisms**, since those are the load-bearing algorithm-implementation claims.

#### Architecture and encoding

The theoretical transformer uses **normalized ReLU attention** in place of softmax (shown experimentally comparable). An attention layer with $M$ heads:

$$
\widetilde H = \mathrm{Attn}_\theta(H) = H + \frac{1}{N}\sum_{m=1}^M (V_m H)\,\sigma\big((Q_m H)^\top (K_m H)\big), \qquad \sigma = \mathrm{ReLU}. \tag{1}
$$

In token form: $\widetilde h_i = h_i + \sum_{m=1}^M \frac1N \sum_{j=1}^N \sigma(\langle Q_m h_i, K_m h_j\rangle)\,V_m h_j$. An MLP layer: $\widetilde H = H + W_2 \sigma(W_1 H)$. A transformer is $L$ such (attention → MLP) blocks. The ICL instance $(D, x_{N+1})$ is encoded as

$$
H = \begin{bmatrix} x_1 & \cdots & x_N & x_{N+1} \\ y_1 & \cdots & y_N & 0 \\ p_1 & \cdots & p_N & p_{N+1} \end{bmatrix} \in \mathbb{R}^{D \times (N+1)},
$$

with positional/indicator rows $p_i$ (ones, zeros, and a train-token flag $t_i = \mathbf{1}\{i < N+1\}$), $D = \Theta(d)$. The prediction is read from the $(d{+}1, N{+}1)$ entry of the output: $\hat y_{N+1} = \mathrm{clip}_R\big((\widetilde h_{N+1})_{d+1}\big)$.

#### The in-context gradient-descent engine (Section 3.5, Theorem 13)

**Goal.** Approximate $L$ steps of GD on a convex empirical risk $\widehat L_N(w) = \frac1N \sum_{i=1}^N \ell(w^\top x_i, y_i)$, with the trajectory

$$
w^{t+1}_{\mathrm{GD}} = w^t_{\mathrm{GD}} - \eta\,\nabla \widehat L_N(w^t_{\mathrm{GD}}), \qquad \nabla \widehat L_N(w) = \frac1N \sum_{i=1}^N \partial_s \ell(w^\top x_i, y_i)\,x_i. \tag{ICGD}
$$

**Key device — one attention layer per GD step.** A single attention layer computes, at each token, the sum $-\frac{\eta}{N}\sum_{i=1}^N \partial_s\ell(\langle w^t, x_i\rangle, y_i)\,x_i$ and adds it to the current $w^t$ stored in the token, yielding $w^{t+1}$. The subtlety is the gradient's scalar factor $\partial_s\ell(s,t)$ (a nonlinear bivariate function). It is realized by a **sum of ReLUs**:

**Definition 12 (approximability by sum of relus).** $g$ is $(\varepsilon_{\mathrm{approx}}, R, M, C)$-approximable if there is $f_{M,C}(z) = \sum_{m=1}^M c_m \sigma(a_m^\top [z;1])$ with $\sum_m |c_m| \le C$, $\max_m \|a_m\|_1 \le 1$, and $\sup_{z \in [-R,R]^k} |g(z) - f_{M,C}(z)| \le \varepsilon_{\mathrm{approx}}$.

So $\partial_s\ell(s,t) \approx \sum_{m=1}^M a_m \sigma(b_m s + c_m t)$, and each ReLU term becomes one attention head: the head's query/key produce $\sigma(b_m\langle w^t, x_i\rangle + c_m y_i)$ and its value emits $-\eta a_m [x_i; 0; 0]$, so summing over heads and tokens reproduces the gradient step (this is the Figure-3 construction). For the **square loss** $\partial_s\ell = s - y$ is affine and exactly a single-ReLU expression, recovering von Oswald et al. / Akyürek as a special case; for general smooth convex $\ell$ the sum-of-relus is an approximation (hence more heads).

**Theorem 13 (Convex ICGD).** For convex $\ell$ with $\partial_s\ell$ $(\varepsilon, R, M, C)$-approximable, there is an attention-only transformer with $L+1$ layers, $\le M$ heads in the first $L$ layers, such that layer $\ell$'s output stores $\widehat w_\ell$ with

$$
\big\|\widehat w_\ell - w^\ell_{\mathrm{GD}}\big\|_2 \le \varepsilon \cdot (\ell \eta B_x),
$$

i.e. **error accumulates only linearly in the number of steps $\ell$**, and the final prediction satisfies $|\hat y_{N+1} - \langle w^L_{\mathrm{GD}}, x_{N+1}\rangle| \le \varepsilon (L\eta B_x^2)$.

**Why the error stays linear — Lemma 14 (stability of convex GD).** Suppose $f$ convex, $\nabla f$ is $L_f$-smooth on a ball of radius $R \ge 2\|w^\star\|$, and two sequences run
$$
\widehat w_{\ell+1} = \widehat w_\ell - \eta\nabla f(\widehat w_\ell) + \varepsilon_\ell \;(\|\varepsilon_\ell\| \le \varepsilon), \qquad w^{\ell+1}_{\mathrm{GD}} = w^\ell_{\mathrm{GD}} - \eta\nabla f(w^\ell_{\mathrm{GD}}),
$$
both from $0$. Then for $\eta \le 2/L_f$ and $L \le R/(2\varepsilon)$, $\|\widehat w_L - w^L_{\mathrm{GD}}\|_2 \le L\varepsilon$.

**Derivation sketch.** The GD map $w \mapsto w - \eta\nabla f(w)$ is **non-expansive** (1-Lipschitz) for convex $L_f$-smooth $f$ when $\eta \le 2/L_f$: for any $u, v$, $\|(u - \eta\nabla f(u)) - (v - \eta\nabla f(v))\| \le \|u - v\|$ (a standard co-coercivity result). Writing $\delta_\ell = \widehat w_\ell - w^\ell_{\mathrm{GD}}$ and unrolling,
$$
\|\delta_{\ell+1}\| = \big\|\underbrace{(\widehat w_\ell - \eta\nabla f(\widehat w_\ell)) - (w^\ell_{\mathrm{GD}} - \eta\nabla f(w^\ell_{\mathrm{GD}}))}_{\text{non-expansive, } \le \|\delta_\ell\|} + \varepsilon_\ell\big\| \le \|\delta_\ell\| + \varepsilon,
$$
so $\|\delta_L\| \le L\varepsilon$. Crucially, non-expansiveness prevents the per-step error $\varepsilon$ from *compounding multiplicatively* — in the non-convex two-layer-NN case (Appendix G) the map is no longer non-expansive and the bound degrades to exponential-in-$L$.

#### Base algorithms as corollaries

- **In-context ridge / least squares (Theorem 4).** For condition number $\kappa = (\beta+\lambda)/(\alpha+\lambda)$, an attention-only transformer with $L = \lceil 2\kappa \log(B_x B_w/2\varepsilon)\rceil + 1$ layers and $\le 3$ heads approximates $w^\lambda_{\mathrm{ridge}}$ to precision $\varepsilon$. (Logarithmic depth because GD on a strongly-convex quadratic converges linearly/geometrically.) Corollary 5 gives near-optimal $\widetilde O(d\sigma^2/N)$ excess risk for least squares; Corollary 6 gives nearly-Bayes risk for Gaussian-prior linear models (where the Bayes estimator *is* ridge with $\lambda = d\sigma^2/N$).
- **In-context GLM / logistic regression (Theorem 7–8, Corollary 9).** Minimize the convex integral loss $\ell(t,y) = -yt + \int_0^t g(s)\,ds$ (for logistic, $g = \sigma_{\log}$, $\ell = -yt + \log(1+e^t)$). Heads scale as $\widetilde O(1/\varepsilon^2)$ because the nonlinear gradient needs a sum-of-relus approximation. Achieves $O(d/N)$ excess risk.
- **In-context Lasso (Theorem 10–11).** Implements *proximal* gradient descent — the MLP layers realize the soft-threshold proximal operator for the non-smooth $\ell_1$ term — reaching the optimal $\widetilde O(s\log d/N)$ sparse-regression rate.
- **GD on two-layer neural nets (Theorem G.1).** A $2L$-layer transformer implements $L$ inexact GD steps on a two-layer-NN ERM; error accumulation is exponential (non-convex), as expected.

#### In-context algorithm selection (Section 4)

**Mechanism 1 — Post-ICL validation (Proposition 15, Theorem 16).** Split $D = (D_{\mathrm{train}}, D_{\mathrm{val}})$ via the positional flag ($t_i = 1$ train, $-1$ val). Run $K$ base predictors $\{f_k\}$ on $D_{\mathrm{train}}$, score each by validation loss $\widehat L_{\mathrm{val}}(f) = \frac{1}{|D_{\mathrm{val}}|}\sum_{(x_i,y_i)\in D_{\mathrm{val}}} \ell(f(x_i), y_i)$, and a **3-layer transformer** outputs a $\widehat f$ that is a convex combination of the near-best predictors $\{f_k : \widehat L_{\mathrm{val}}(f_k) \le \min_{k^\star}\widehat L_{\mathrm{val}}(f_{k^\star}) + \gamma\}$. The generalization consequence:

$$
L(\widehat f) \le \min_{k^\star} L(f_{k^\star}) + \max_{k}\big|\widehat L_{\mathrm{val}}(f_k) - L(f_k)\big| + \gamma,
$$

i.e. it competes with the best base algorithm up to a validation-concentration term. Theorem 16 instantiates this as **ridge with in-context regularization selection** over $\{\lambda_1,\dots,\lambda_K\}$.

**Flagship — nearly Bayes-optimal ICL under mixed noise (Theorem 17).** Data are drawn as $k \sim \Lambda$, $w^\star \sim \mathcal{N}(0, I_d/d)$, $y_i = \langle x_i, w^\star\rangle + \mathcal{N}(0, \sigma_k^2)$ with $K$ possible noise levels. The Bayes predictor is a data-dependent mixture of ridge regressions with $\lambda_k = d\sigma_k^2/N$ — the mixing weights depend on $D$ non-trivially. A transformer with $O(\log N)$ layers and $O(K)$ heads, built from post-ICL validation, achieves

$$
\mathbb{E}_\pi\!\left[\tfrac12(y_{N+1} - \hat y_{N+1})^2\right] \le \mathrm{BayesRisk}_\pi + O\big((\log K / N)^{1/3}\big),
$$

vanishing excess risk over Bayes as $N \to \infty$. This strictly strengthens Akyürek (who matched Bayes only at a *single fixed* noise level).

**Mechanism 2 — Pre-ICL testing (Lemma 18, Proposition 19).** Compute a summary statistic of the labels; e.g. a binary-type check $\Psi_{\mathrm{binary}}(D) = \frac1N \sum_i \psi(y_i)$ with $\psi(y) = 1$ if $y \in \{0,1\}$, tapering to $0$ otherwise. **A single 6-head attention layer implements $\Psi_{\mathrm{binary}}$ exactly.** A transformer then routes to logistic regression when labels are binary and to least squares when they are continuous — selecting the *task*, not just the hyperparameter.

#### Pretraining sample complexity (Section 5)

Given $n$ pretraining ICL instances $Z_j = (H_j, y_{N+1,j}) \sim \pi$, minimize the clipped square ICL loss (TF-ERM) over the norm ball $\Theta_{L,M,D',B}$. A chaining argument gives:

$$
L_{\mathrm{icl}}(\widehat\theta) \le \inf_{\theta \in \Theta_{L,M,D',B}} L_{\mathrm{icl}}(\theta) + O\!\left(B_y^2 \sqrt{\frac{L^2(MD^2 + DD')\iota + \log(1/\xi)}{n}}\right), \tag{Thm 20}
$$

with $\iota$ a log factor. Combining with the expressivity results yields **end-to-end** guarantees: e.g. Theorem 21 pretrains an in-context linear-regression transformer to excess risk $\widetilde O(\sqrt{\kappa^2 d^2/n} + d\sigma^2/N)$ with $O(\kappa\log(\kappa N/\sigma))$ layers, 3 heads, from **polynomially many** sequences. Analogous end-to-end theorems cover sparse regression (Thm 22), mixed-noise ridge with algorithm selection (Thm 23), and logistic regression (Thm 24).

#### Experiments (Section 6)

A 12-layer transformer (the ReLU-attention architecture from the theory), $d=20$. **Base mode:** trained per-task, it matches the best baseline on 4 of 5 tasks (linear, two noisy-linear, sparse, classification), and beats least squares / matches Lasso on the sparse task. **Mixture mode:** a single `TF_alg_select` trained on a mixture of two noise levels simultaneously approaches *both* individual Bayes errors (Figure 2/4b) and beats any fixed-λ ridge; another selects between regression and classification (Figure 4c). Single-task transformers (`TF_noise_1`, `TF_noise_2`) are optimal on their own task but suboptimal on the other — confirming the *adaptive* selection is doing real work.

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** Motivation: transformers do more than one simple algorithm. Two contributions: (1) in-context algorithm selection (a single TF adaptively picks base algorithms), (2) first comprehensive end-to-end ICL theory (expressivity + prediction power + pretraining sample complexity). Two selection mechanisms previewed (post-ICL validation, pre-ICL testing) with the mixed-noise Bayes application. §1.1 related work (Garg et al.; Akyürek; von Oswald; Xie et al. Bayesian; Li et al. model selection) and §Techniques (Bach ReLU-approximation; chaining).
- **§2 Preliminaries.** §2.1 transformer definitions: normalized-ReLU attention (Def. 1, Eq. 1), MLP layer (Def. 2), $L$-layer TF (Def. 3), norm $|||\theta|||$ (Eq. 2). §2.2 ICL setup: instance $(D, x_{N+1})$, input-sequence encoding (Eq. 3), read-out with clipping; note on decoder generalization to every-token prediction.
- **§3 Basic in-context learning algorithms.** §3.1 ridge/least squares (ICRidge, Theorem 4; Corollaries 5–6 near-optimal & nearly-Bayes). §3.2 GLMs / logistic (ICGLM, Theorems 7–8, Corollary 9). §3.3 Lasso via proximal GD (ICLasso, Theorems 10–11). §3.4 GD on two-layer NNs (Theorem G.1). §3.5 the **in-context gradient descent engine**: Definition 12 (sum-of-relus), Theorem 13 (convex ICGD, linear error accumulation), Lemma 14 (convex-GD stability), Figure 3 (one attention layer = one GD step), proximal-GD variant.
- **§4 In-context algorithm selection.** §4.1 post-ICL validation (Proposition 15, Theorem 16 ridge-λ selection). §4.1.1 nearly Bayes-optimal ICL on mixed-noise linear models (Theorem 17). §4.2 pre-ICL testing (Ψ_binary, Lemma 18, Proposition 19 regression-vs-classification routing).
- **§5 Analysis of pretraining.** §5.1 TF-ERM setup and generalization bound (Theorem 20, chaining). §5.2 end-to-end pretraining examples: linear regression (Theorem 21), sparse regression (Theorem 22), mixed-noise algorithm selection (Theorem 23), logistic regression (Theorem 24). Remark: transformer encodes no problem-specific structure beyond size.
- **§6 Experiments.** §6.1 base + mixture training modes; results matching best baselines and demonstrating algorithm selection (Figure 4). §6.2 details of the mixed-noise Bayes experiment (Figure 2).
- **Appendices A–K.** Sum-of-relus approximation lemmas (Bach), proofs of all theorems, decoder-architecture generalization (App. B), two-layer-NN GD (App. G), exact-Bayes derivation for mixed noise (App. I), pretraining proofs (App. J), experimental details (App. K).

---

## 13. Ahn et al. 2023 — Transformers Learn to Implement Preconditioned Gradient Descent for In-Context Learning

**PDF:** `docs/project/references/in_context_learning/sources/Ahn et al. 2023 - Transformers learn to implement preconditioned gradient descent.pdf`
**Venue:** NeurIPS 2023 (arXiv:2306.00297v2).

### Phase 1: Foundational Overview (Undergraduate-Level)

**The question in plain terms.** Earlier work (von Oswald et al. 2023; Akyürek et al. 2022) showed that if you *hand-pick* the weights of a transformer, it can carry out gradient descent (GD) on a small regression problem hidden inside its input prompt. That is an **expressivity** claim: the machine is *capable* of it. But in practice nobody hand-picks weights — you train them by minimizing a loss with SGD/Adam, which is non-convex and offers no guarantee you land on the "GD-implementing" weights. Ahn et al. ask the harder question: **if you actually train the network, is the GD-implementing solution the one training prefers?** They answer yes, and more: the preferred solution is not plain GD but a *smarter* variant called **preconditioned gradient descent**.

**The testbed.** A "linear transformer" (self-attention with the softmax removed) is trained on random **linear-regression prompts**: each prompt packs $n$ labeled examples $(x^{(i)}, y^{(i)}=\langle x^{(i)}, w_\star\rangle)$ plus one unlabeled query $x^{(n+1)}$, and the network must predict $\langle w_\star, x^{(n+1)}\rangle$. The task weight $w_\star$ and the covariates change every prompt, so the only way to succeed is to *learn a regression algorithm* that reads the labeled examples and infers $w_\star$ on the fly.

**Key findings.**
- **Single layer (Theorem 1).** The *global minimum* of the training loss makes the network perform **exactly one step of preconditioned gradient descent**. The "preconditioner" is a matrix that (a) approximates the inverse of the data covariance $\Sigma^{-1}$ — which speeds up convergence when the inputs are badly scaled — and (b) adds a regularizer that grows when the number of in-context examples $n$ is small (i.e., when the data are inadequate). So the trained network doesn't just do GD; it adapts its step to *both* the data geometry *and* the sample size, resembling structural risk minimization.
- **Multiple layers (Theorems 2–4).** Under a natural sparsity restriction on the weights, an $L$-layer network's forward pass is *provably identical* to $L$ steps of preconditioned GD (Lemma 1). Theorem 2 shows a 2-layer net's optimum is GD with adaptive, coordinate-wise step sizes (Adagrad-like, but tuned to the data distribution rather than the instance). Theorem 3 shows that for deeper nets, the choice "preconditioner $\propto \Sigma^{-1}$ at every layer" (Newton-like) is a critical point. Theorem 4 relaxes the sparsity and finds a *richer* algorithm that both preconditions the gradient *and* reshapes the covariates to improve conditioning between steps — this recovers the empirically-observed **GD++** algorithm of von Oswald et al.
- **Experiments** on 3-layer nets confirm the trained weights converge to exactly these critical points, with loss going to ~0 (suggesting they are global optima).

**Initial takeaway.** This is the first rigorous *loss-landscape* result for learning algorithms via training. It reframes "transformers can do GD" (expressivity) into "training a transformer *discovers* an adaptive, preconditioned GD tuned to the task distribution" (optimization). The bonus insight: the learned preconditioner encodes distributional knowledge (covariance + sample-size-dependent regularization) that plain GD lacks — the transformer is learning a *better-than-vanilla* optimizer.

### Phase 2: Graduate-Level Deep Dive

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

#### Derivation 1 — the loss depends only on $(b, A)$ (single layer)

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

#### Derivation 2 — global minimum for isotropic data (Theorem 1, $\Sigma=I$)

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

#### Non-isotropic optimum and its interpretation

For $x^{(i)}\sim N(0,\Sigma)$ with $\Sigma=U\Lambda U^\top$, $\Lambda=\mathrm{diag}(\lambda_1,\dots,\lambda_d)$, and $w_\star\sim N(0,I)$, the optimal preconditioner is

$$
Q_0 = -\begin{bmatrix} U\,\mathrm{diag}\!\Big(\frac{1}{\frac{n+1}{n}\lambda_i + \frac{1}{n}(\sum_k\lambda_k)}\Big)_{i}\,U^\top & 0 \\ 0 & 0\end{bmatrix}.
$$

Two structural readings:
- **As $n\to\infty$**, the top-left block $\to \Sigma^{-1}$: the transformer preconditions by the inverse covariance, accelerating convergence when $\Sigma$ is ill-conditioned. This is the "adapts to input distribution" property.
- **The $\tfrac1n\sum_k\lambda_k$ term** in each denominator acts as a **Tikhonov regularizer** whose strength grows when $n$ is small or the covariate variance is high — "adapts to the variance induced by data inadequacy," echoing structural risk minimization (Vapnik).

#### Multi-layer: forward pass = preconditioned GD (Lemma 1)

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

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** Distinguishes expressivity ("a transformer *can* implement GD") from the learning question ("does *training* find it?"). Positions the paper as the first loss-landscape analysis of learning algorithms via training linear transformers on random linear regression.
- **§1.1 Related works.** Turing-completeness of RNNs (Siegelmann–Sontag), neural Turing machines, algorithmic power of transformers, hidden-layer GD interpretation (Jastrzebski et al.), and concurrent single-layer analyses (Zhang et al. 2023 = paper #3 here; Mahankali et al. 2023). Their Theorem 1 structure matches Zhang et al.'s independently-derived global optimum.
- **§2 Setting.** Random linear-regression data distribution; input matrix $Z_0$; softmax-free linear self-attention $\mathrm{Attn}_{P,Q}(Z)=PZM(Z^\top QZ)$; $L$-layer residual stack; in-context loss $f$. Table 1 maps each theorem to its data model and guarantee (global minimizer vs critical point).
- **§3 Single-layer global optimum.** Theorem 1 (non-isotropic $\Sigma$, isotropic $w_\star$); the isotropic special case (7); interpretation of the preconditioner as $\Sigma^{-1}$ + sample-size regularizer.
- **§4 Multi-layer, sparse parameters.** Sparsity condition (8); Lemma 1 (forward pass = preconditioned GD); §4.1 Theorem 2 (2-layer symmetric → diagonal adaptive-step GD); §4.2 Theorem 3 (deep → $A_i\propto\Sigma^{-1}$ critical point, Newton-like); §4.3 experimental validation on 3-layer nets ($d=5,n=20$) showing $\mathrm{Dist}(\Sigma^{1/2}A_i\Sigma^{1/2}, I)\to0$.
- **§5 Beyond standard optimization.** Relaxed parameterization (11); Theorem 4 (critical set $A_i\propto\Sigma^{-1}$, $B_i\propto I$ = GD++); §5.1 experiments confirming $B_i\to I$, $A_i\to\Sigma^{-1}$, and increasing step-size norms across layers.
- **§6 Discussion.** Future directions: nonlinear attention; Theorem 5 (ReLU single-layer global minimizer).
- **Appendices A–D.** A: single-layer proofs (loss rewriting, isotropic warm-up via convex per-coordinate components $f_j$, non-isotropic, ReLU). B: multi-layer proofs (Theorem 2/3/4), including the flow-in-$S$ argument. C: auxiliary lemmas, incl. proof of Lemma 1. D: experiment details / weight visualizations.

---

## 14. Li et al. 2023 — Transformers as Algorithms: Generalization and Stability in In-Context Learning

**PDF:** `docs/project/references/in_context_learning/sources/Li et al. 2023 - Transformers as Algorithms.pdf`
**Full title:** *Transformers as Algorithms: Generalization and Stability in In-context Learning* (arXiv:2301.07067v2).

### Phase 1: Foundational Overview (Undergraduate-Level)

**The question in plain terms.** The other two papers in this shard ask *which* algorithm a trained transformer implements. Li et al. ask a complementary statistical question: **how much data do you need to train a good in-context learner, and why is a transformer a statistically sound way to learn an algorithm at all?** They formalize ICL as an **algorithm-learning problem** — the transformer is abstracted as a map from a data sequence to a prediction function — and prove finite-sample generalization bounds for the whole meta-learning pipeline.

**Setup.** During training ("multitask learning", MTL) the model sees $T$ tasks, each supplying a length-$n$ sequence fed **auto-regressively**: the model predicts example $m$ from the prompt of the first $m-1$ examples, for every $m$. Two prompt types are covered: (1) i.i.d. (input, label) pairs (supervised learning), and (2) **trajectories of a dynamical system** $x_{m+1}=f(x_m)+\text{noise}$ (so the prompt is temporally *dependent*, unlike prior ICL theory).

**Key findings.**
- **Generalization rate $1/\sqrt{nT}$.** The excess MTL risk decays like $\sqrt{\dim(\mathcal A)/(nT)}$ — you benefit both from more tasks $T$ *and* from more examples per task $n$. The hard part is the $1/\sqrt n$ factor: examples within one prompt are temporally coupled (each influences all later predictions), so a naive bound would lose the $n$. The key is **algorithmic stability** — if one in-context example only changes future predictions by $O(1/m)$, a martingale (Azuma–Hoeffding) argument recovers the $\sqrt n$.
- **Transformers are provably stable.** A $D$-layer transformer with bounded weights changes its output by at most $\frac{2}{2m-1}((1+\Gamma)e^\Gamma)^D$ when a non-final in-context token is swapped — meeting the $K/m$ stability condition. The proof hinges on a tight Lipschitz analysis of the softmax self-attention layer.
- **Transfer (unseen tasks) is governed by task complexity, not model size.** Empirically, transfer risk for $d$-dimensional linear regression aligns perfectly when plotted against $(n/d,\,T/d^2)$ — it depends on the *task* dimension $d$, not on the transformer's parameter count $\dim(\mathcal A)$ (verified across GPT-2 sizes spanning $64\times$ parameter counts). This is an **inductive-bias** phenomenon: MTL pretraining implicitly selects an algorithm lying in the span of the source tasks.
- **ICL as model selection.** With a stronger $\times n$ factor from Theorem 3.5, ICL can adaptively pick the right hypothesis class per sample size $m$ (bias–variance trade-off), achieving $\sqrt{\log H / T}$ instead of paying for the union of all classes.

**Initial takeaway.** Where Ahn/Zhang describe *what* algorithm emerges, Li et al. supply the *statistical-learning-theory scaffolding*: stability ⇒ generalization ⇒ meta-learning transfer, with matching experiments. The stability–generalization bridge is the technical heart.

### Phase 2: Graduate-Level Deep Dive

#### Problem formalization

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

#### Algorithmic stability (Assumption 3.1)

Let $S^j$ be $S$ with its $j$-th sample replaced. **Error stability** with constant $K$:

$$
\Big|\mathbb{E}_{(x,y)}\big[\ell(y, f^{\mathrm{Alg}}_S(x)) - \ell(y, f^{\mathrm{Alg}}_{S^j}(x))\big]\Big| \le \frac{K}{m}.
$$

The stronger **pairwise** version bounds the *difference of differences* between two algorithms by $K\rho(\mathrm{Alg},\mathrm{Alg}')/m$, enabling chaining. The $1/m$ scaling is the classical order (Bousquet–Elisseeff) needed for realistic generalization.

#### Theorem 3.2 — transformers satisfy stability (with derivation of the self-attention Lipschitz bound)

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

#### Theorem 3.5 — the generalization bound

With $\mathcal A$ being $K$-stable, $\ell$ $L$-Lipschitz and $[0,1]$-valued, and covering number $\mathcal N(\mathcal A,\rho,\varepsilon)$ under the algorithm distance $\rho(\mathrm{Alg},\mathrm{Alg}')=\sup_S\frac1n\sum_i\|f^{\mathrm{Alg}}_{S^{(i-1)}}(x_i)-f^{\mathrm{Alg}'}_{S^{(i-1)}}(x_i)\|$, with prob. $\ge1-2\delta$:

$$
R_{\mathrm{MTL}}(\widehat{\mathrm{Alg}}) \le \inf_{\varepsilon>0}\Bigg\{4L\varepsilon + 2(1 + K\log n)\sqrt{\frac{\log(\mathcal N(\mathcal A,\rho,\varepsilon)/\delta)}{c\,nT}}\Bigg\}.
$$

For Lipschitz architectures $\log\mathcal N\sim\dim(\mathcal A)\log(1/\varepsilon)$, so up to logs $R_{\mathrm{MTL}}\lesssim\sqrt{\dim(\mathcal A)/(nT)}$. The stronger pairwise-stable bound (4) replaces the covering term with Dudley's entropy integral (chaining), matching Rademacher complexity at $nT$ samples.

**Proof sketch (why $1/\sqrt n$ survives temporal dependence).** Fix an algorithm and define the Doob martingale $X_{t,i}=\mathbb{E}\big[\frac1n\sum_j\ell(y_{tj}, f^{\mathrm{Alg}}_{S^{(j-1)}_t}(x_{tj}))\,\big|\,S^{(i)}_t\big]$. Revealing example $i$ changes predictions $i{+}1,\dots,n$; by stability each contributes $\le K/j$, so the martingale increment obeys $|X_{t,i}-X_{t,i-1}|\lesssim 1 + \sum_{j=i}^n K/j \lesssim 1 + K\log n$. Azuma–Hoeffding on $\frac1T\sum_t(X_{t,0}-X_{t,n})=|L_{\mathrm{MTL}}-\widehat L_{S_{\text{all}}}|$ then gives concentration at rate $(1+K\log n)/\sqrt{nT}$; a covering/chaining union bound makes it uniform over $\mathcal A$. With $M$ independent sequences per task, the rate improves to $1/\sqrt{nMT}$.

#### Dynamical systems (Section 5)

The same machinery extends to prompts that are trajectories $x_{m+1}=f(x_m)+w_m$. Two stability notions combine: **system stability** (Def 5.1, $(C_\rho,\rho)$: $\|f^{(m)}(x_0)-f^{(m)}(x_0')\|\le C_\rho\rho^m\|x_0-x_0'\|$, exponential forgetting of initial condition) and **algorithmic stability for dynamics** (Assumption 5.3, a Lipschitz-style variant summing perturbations over $i=j,\dots,m$). Theorem 5.4 recovers Theorem 3.5's bound with $K$ replaced by $\bar K = 2K\frac{\bar C_\rho}{1-\bar\rho}(\bar w + \bar x/\sqrt n)$: the system's exponential stability controls how a noise perturbation propagates down the trajectory.

#### Transfer learning & inductive bias (Section 4)

Transfer risk decays as $1/\mathrm{poly}(T)$ (unseen tasks induce distribution shift, unfixable by larger $n$ or $M$): $L_{\mathrm{TFR}}-L_{\mathrm{MTL}}\lesssim\sqrt{\log\mathcal N(\mathcal A,\rho,\varepsilon)/T}$. The empirical surprise: transfer risk for $d$-dim linear regression collapses onto a single curve in $(n/d, T/d^2)$ and is *independent* of $\dim(\mathcal A)$. The Bayes-optimal comparator is weighted ridge $\hat\beta=(X^\top X+\sigma^2\Sigma^{-1})^{-1}X^\top y$; estimating the task covariance $\hat\Sigma=\frac1T\sum\beta_i\beta_i^\top$ to spectral accuracy needs only $T=\Omega(d)$, yet ICL empirically needs $T\propto d^2$ (entrywise $\ell_\infty$ accuracy) — so ICL is *not* sample-optimal in $T$, but its inductive bias makes it insensitive to architecture size.

#### Model selection (Section 6)

Under Hypothesis 1 (the TF can $\varepsilon^{h,m}_{\mathrm{TF}}$-approximate ERM over each class $\mathcal F_h$), ICL adaptively selects, per sample size $m$, the class minimizing $L^\star_h + \mathcal O(R_m(\mathcal F_h)) + \varepsilon^{h,m}_{\mathrm{TF}}$ (Observation 1), rather than paying the Rademacher complexity of the union $\bigcup_h\mathcal F_h$. With $H$ classes over $n$ sample sizes there are $H^n$ effective ERM algorithms (VC-dim $\sim n\log H$), and Theorem 3.5's $\times n$ factor turns the excess risk into $\sqrt{n\log H/(nT)}=\sqrt{\log H/T}$ — the extra $n$ *pays for* per-sample-size adaptivity.

### Appendix: Section-by-Section Backbone

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

## 15. Zhang et al. 2024 — Trained Transformers Learn Linear Models In-Context

**PDF:** `docs/project/references/in_context_learning/sources/Zhang et al. 2024 - Trained Transformers Learn Linear Models In-Context.pdf`
**Venue:** JMLR / arXiv:2306.09927v3.

### Phase 1: Foundational Overview (Undergraduate-Level)

**The question in plain terms.** Ahn et al. characterize the *global minimum* of the training loss (where does the good solution *sit*?). Zhang et al. answer the strictly harder dynamical question: **starting from a random initialization, does gradient descent actually reach that good solution, despite the loss being non-convex?** And once there, **how robust is the learned in-context learner to distribution shift?**

**Setup.** A single linear self-attention (LSA) layer is trained by **gradient flow** (gradient descent with infinitesimal step size) on the *population* loss of random anisotropic linear-regression prompts. Covariates $x_i\sim N(0,\Lambda)$ (possibly badly scaled), task weight $w\sim N(0,I_d)$, labels $y_i=\langle w, x_i\rangle$.

**Key findings.**
- **Global convergence despite non-convexity (Theorem 4.1).** With a "balanced" random initialization, gradient flow provably converges to a **global minimum** of the population loss, and the paper writes down the limiting weights in closed form.
- **The learned algorithm (Corollary 4.3).** At convergence the prediction is $\hat y_{\text{query}} = x_{\text{query}}^\top\,\Gamma^{-1}\big(\frac1M\sum_i y_i x_i\big)$ with $\Gamma = (1+\tfrac1N)\Lambda + \tfrac{\mathrm{tr}(\Lambda)}{N}I_d$ — one step of **preconditioned** GD, and because the preconditioner is $\approx\Lambda^{-1}$ it handles *anisotropic* data where plain one-step GD fails. Prediction error is $O(1/M + 1/N^2)$: it decays faster in the *training* prompt length $N$ than the *test* length $M$, and does *not* vanish unless $N\to\infty$ (finite training prompts leave irreducible error).
- **Robust to task & query shift, brittle to covariate shift (§4.2).** The learned learner tolerates label noise and nonlinear tasks (it computes the best linear predictor over the test prompt) and query-distribution changes — but if the test covariate distribution differs from training (e.g. features scaled by $c$), the prediction is off by exactly $c^2$: the transformer is **not** implementing scale-invariant OLS.
- **Even training on random covariances doesn't fix it (§4.3, Theorem 4.5).** If you vary $\Lambda$ across prompts, gradient flow still converges to a global minimum, but the single-LSA learner still fails covariate shift — e.g. with $\lambda_i\sim\mathrm{Exp}(1)$ the prediction shrinks to $\tfrac13\langle w, x_{\text{query}}\rangle$. Larger nonlinear GPT-2 models generalize better under covariate shift, especially trained on random covariances (but still imperfectly, and they spike when $M>N$).

**Initial takeaway.** This is the "does training actually get there" companion to Ahn et al. — an end-to-end convergence proof plus a careful **robustness audit** exposing that the elegant one-step-preconditioned-GD story is fragile: the learned algorithm secretly depends on the training covariance and is *not* covariate-shift invariant. That fragility is a cautionary tale for any conditioning mechanism trained on a fixed input statistic.

### Phase 2: Graduate-Level Deep Dive

#### Model and initialization

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

#### Reduction to rank-one matrix factorization (Lemma 5.1)

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

#### Gradient-flow dynamical system (Lemma 5.2)

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

#### Global minima and the learned algorithm (Lemma 5.3)

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

#### PL inequality ⇒ convergence (Lemma 5.4, Theorem 4.1)

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

#### Prediction error (Theorem 4.2)

For a test distribution $D$ over $(x,y)$ with $x\sim N(0,\Lambda)$, defining $a:=\Lambda^{-1}\mathbb{E}[xy]$ and $\Sigma:=\mathbb{E}[(xy-\mathbb{E}(xy))(xy-\mathbb{E}(xy))^\top]$,

$$
\mathbb{E}(\hat y_{\text{query}}-y_{\text{query}})^2 = \underbrace{\min_{w}\mathbb{E}(\langle w,x_{\text{query}}\rangle - y_{\text{query}})^2}_{\text{best linear predictor}}
+ \frac1M\mathrm{tr}(\Sigma\Gamma^{-2}\Lambda)
+ \frac1{N^2}\big[\|a\|^2_{\Gamma^{-2}\Lambda^3} + 2\,\mathrm{tr}(\Lambda)\|a\|^2_{\Gamma^{-2}\Lambda^2} + \mathrm{tr}(\Lambda)^2\|a\|^2_{\Gamma^{-2}\Lambda}\big].
$$

Two structural reads: (i) excess error is $O(1/M + 1/N^2)$ — faster in $N$ than $M$; (ii) even $M\to\infty$ leaves an $O(1/N^2)$ gap: **finite training-prompt length is an irreducible bottleneck.** For noiseless well-specified linear data with $w\sim N(0,I)$ and condition number $\kappa$, this collapses to $\le \frac{(d+1)\mathrm{tr}(\Lambda)}{M} + \frac{(1+2d+d^2\kappa)\mathrm{tr}(\Lambda)}{N^2}$ (Corollary 4.3).

#### Distribution-shift audit (§4.2) — the fragility

From the universal identity $\hat y_{\text{query}}\approx x_{\text{query}}^\top\Lambda^{-1}(\frac1M\sum_i y_i x_i)$:
- **Task shift (tolerated).** Noisy labels $y_i=\langle w,x_i\rangle+\varepsilon_i$: the noise term $x_{\text{query}}^\top\Lambda^{-1}(\frac1M\sum\varepsilon_i x_i)\to0$ (mean-zero, independent), so $\hat y\approx\langle w, x_{\text{query}}\rangle$ for *arbitrary* $w$. Nonlinear tasks → best linear predictor.
- **Query shift (tolerated).** With $D^{\text{train}}_x=D^{\text{test}}_x$ and large $M$, $\hat y\approx x_{\text{query}}^\top\Lambda^{-1}\Lambda w = x_{\text{query}}^\top w$ for very general query distributions.
- **Covariate shift (NOT tolerated).** If test covariates are scaled by $c$, $\frac1M\sum x_ix_i^\top\approx c^2\Lambda$, so $\hat y\approx c^2 x_{\text{query}}^\top w \ne x_{\text{query}}^\top w$. OLS would be scale-invariant; the transformer is not — proving the *implemented algorithm differs from OLS* even where predictions superficially match.

#### Random-covariance training doesn't rescue it (§4.3, Theorem 4.5)

Train with $\Lambda_\tau$ drawn i.i.d. (diagonal, positive, finite 3rd moment). Gradient flow still converges to a global minimum with limiting

$$
W^{KQ}_\ast \propto (\mathbb{E}\Gamma_\tau\Lambda_\tau^2)^{-1}(\mathbb{E}\Lambda_\tau^2),\qquad
\Gamma_\tau = \tfrac{N+1}{N}\Lambda_\tau + \tfrac1N\mathrm{tr}(\Lambda_\tau)I_d.
$$

For a fresh $\Lambda_{\text{new}}$, the prediction tends (as $M,N\to\infty$) to $x_{\text{query}}^\top(\mathbb{E}\Lambda_\tau^2)(\mathbb{E}\Lambda_\tau^3)^{-1}(\mathbb{E}\Lambda_\tau)\,w$; the factor $(\mathbb{E}\Lambda_\tau^2)(\mathbb{E}\Lambda_\tau^3)^{-1}(\mathbb{E}\Lambda_\tau)\ne I$ in general. With $\lambda_{\tau,i}\sim\mathrm{Exp}(1)$ ($\mathbb{E}\Lambda=I,\ \mathbb{E}\Lambda^2=2I,\ \mathbb{E}\Lambda^3=6I$), $\mathbb{E}\hat y_{\text{query}}\to\frac{2\cdot1}{6}\langle w,x_{\text{query}}\rangle=\tfrac13\langle w,x_{\text{query}}\rangle$ — a systematic $\tfrac13$ shrinkage. A single LSA layer *cannot* in-context-learn linear models across covariate distributions.

**Empirical coda.** Nonlinear GPT-2 trained on random covariances generalizes better under covariate shift (fails only at the largest scale $c=9$), but exhibits an error **spike when $M>N$** (test prompt longer than training), conjectured to stem from GPT-2's absolute positional encodings (concurrent work reports removing them helps).

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** ICL as parameter-free supervised learning; Garg et al.'s function-class framing; the open gap — *how does gradient-based training produce ICL?* Four contributions: global convergence for single-LSA; characterization of the learned algorithm & error; distribution-shift robustness; the random-covariance failure + nonlinear-GPT2 experiments.
- **§2 Related work.** GD-view (von Oswald/Akyürek), Bayesian-inference view (Xie et al.), implicit-finetuning (Dai et al.), kernel-regression (Han et al.), approximation-theoretic (Yun et al., Bai et al.), gradient-dynamics results (Jelassi et al., Li–Li–Risteski); explicit comparison to concurrent Ahn et al. (landscape vs. dynamics).
- **§3 Preliminaries.** Notation (Kronecker $\otimes$, Vec); Def 3.1 (trained on in-context examples), Def 3.2 (in-context learning of a class up to error $\eta$); softmax self-attention → LSA simplification $f_{\mathrm{LSA}}=E+W^{PV}E\,E^\top W^{KQ}E/\rho$; token embedding (3.4); prediction reduces to block components (3.6); training procedure (population loss (3.8), gradient flow (3.9)); Assumption 3.3 (balanced init).
- **§4 Main results.** §4.1 Theorem 4.1 (convergence + limiting weights); prediction identities (4.2)/(4.3); Theorem 4.2 (error decomposition, $O(1/M+1/N^2)$); Corollary 4.3. §4.2 distribution shifts (task/query tolerated, covariate not). §4.3 Def 4.4 (random covariate distributions); Theorem 4.5 (convergence + $\tfrac13$-shrinkage failure); nonlinear GPT-2 experiments (Fig 1, $M>N$ spike).
- **§5 Proof ideas.** §5.1 reduction to quadratic $u^\top H_\tau u$ / rank-one factorization + non-convexity via $X_\tau$'s negative eigenvalue (Lemma 5.1). §5.2 gradient-flow ODEs + reduced objective $\tilde\ell$ (Lemma 5.2), global-minima condition $u_{-1}U_{11}=\Gamma^{-1}$ (Lemma 5.3). §5.3 PL inequality ⇒ global convergence + limiting values (Lemma 5.4).
- **§6 Conclusion.** Summary; robustness to task/query but brittleness to covariate shift; open questions (alternative inits, deeper nets, positional-encoding spike).
- **Appendices A–E.** A: proof of Theorem 4.1 (Lemmas 5.1–5.4 in full). B: proof of Theorem 4.2. C: proof of Theorem 4.5 (random covariance). D: auxiliary results. E: experiment details.

---

# Part 3 — §3 Amortized Bayesian inference (PFN / TabPFN / neural processes / posterior estimation)

*Amortized conditional prediction: train a network offline so one forward pass maps a context set to a predictive distribution. Prior-Data Fitted Networks (PFN/TabPFN) and the Neural-Process line (CNP → ANP → TNP), plus Transformers that emit full posteriors — organized by the deterministic-vs-latent path and the point-vs-distribution estimator axes.*

---

## 16. Müller et al. 2022 — Transformers Can Do Bayesian Inference (PFNs)

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

## 17. Hollmann et al. 2023 — TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second

**PDF:** `docs/project/references/in_context_learning/sources/Hollmann et al. 2023 - TabPFN.pdf` · ICLR 2023 (oral) · arXiv:2207.01848

### Phase 1: Foundational Overview (Undergraduate-Level)

**The problem in plain terms.** Small tabular datasets (spreadsheet-style data: rows are samples, columns are features) are the most common data type in real machine-learning applications, yet deep learning historically loses to gradient-boosted decision trees (XGBoost, LightGBM, CatBoost) here. The usual recipe — fit a fresh model to each new dataset, then tune its hyperparameters with expensive cross-validated search — is slow. TabPFN throws that recipe out. It is a *single, already-trained* Transformer that classifies a brand-new small tabular dataset **in one forward pass in under a second, with no fitting and no tuning at all**.

**How it works, at a high level.** TabPFN is a **Prior-Data Fitted Network (PFN)**. Once, offline, it is trained on *millions of synthetic datasets* sampled from a hand-designed "prior" over what tabular data might look like. Each synthetic dataset comes with training rows and held-out test rows; the network is trained to predict the held-out labels from the training rows given in its input. Because it has seen so many synthetic worlds, it learns the *general algorithm* of "look at labeled examples, predict the query label." At deployment you feed your real dataset's labeled rows plus a test row as one big input sequence, and the network emits class probabilities. This is exactly in-context learning (ICL): the labeled rows are the prompt, no weights change.

**The key idea about the prior.** In an ordinary neural net you encode your assumptions (inductive bias) implicitly through architecture, regularization, dropout, etc. In a PFN you encode them *explicitly and directly* by writing a program that generates synthetic datasets. TabPFN's prior generates data from **Structural Causal Models (SCMs)** and **Bayesian Neural Networks (BNNs)**, with a built-in **preference for simplicity** (Occam's razor: fewer nodes / fewer parameters are more likely). The trained network then approximates the **Posterior Predictive Distribution (PPD)** — the Bayesian-optimal prediction averaged over *all* data-generating mechanisms consistent with your data, weighted by prior probability and likelihood. So a single forward pass implicitly does an infinite ensemble over causal models.

**Key findings / initial takeaway.** On 18 numerical OpenML-CC18 datasets (≤1000 training points, ≤100 features, ≤10 classes), TabPFN clearly beats individual boosted-tree methods and *matches state-of-the-art AutoML systems that were given a full hour* — while itself taking under a second (a 230× speedup on CPU, 5700× on GPU). Its errors are uncorrelated with those of tree methods, so ensembling TabPFN with AutoGluon beats everything. Predictions are smooth and well-calibrated (GP-like uncertainty growing away from the data). The headline: **for small tabular classification, a fixed pre-trained Transformer can replace the entire fit-and-tune pipeline.**

### Phase 2: Graduate-Level Deep Dive

**The amortized-conditional-predictive objective (the heart of the PFN framework).**

In the Bayesian framework for supervised learning, a **prior** defines a hypothesis space $\Phi$ over input-output relationships. Each hypothesis $\phi \in \Phi$ is a data-generating mechanism (e.g. one specific SCM). The **Posterior Predictive Distribution (PPD)** for a test point $x_{\text{test}}$, conditioned on a training set $D_{\text{train}} = \{(x_1,y_1),\dots,(x_n,y_n)\}$, is obtained by integrating over $\Phi$:

$$
p(y \mid x, D) \;\propto\; \int_{\Phi} p(y \mid x, \phi)\, p(D \mid \phi)\, p(\phi)\, d\phi .
$$

Here each hypothesis $\phi$ is weighted by its prior probability $p(\phi)$ and the likelihood $p(D\mid\phi)$ of the observed data under it — this is exactly Bayesian model averaging over data-generating mechanisms. This integral is intractable for any rich $\Phi$; the entire point of a PFN is to **amortize** it into the weights of a network $q_\theta$.

**Prior-fitting (offline training).** The prior is specified operationally as a *sampling scheme* rather than a density:

$$
p(D) = \mathbb{E}_{\phi \sim p(\phi)}\big[\, p(D \mid \phi) \,\big],
$$

i.e. first draw a mechanism $\phi \sim p(\phi)$, then draw a synthetic dataset $D = (x_i,y_i)_{i=1}^n \sim p(D\mid\phi)$. The PFN is trained to predict a held-out portion $D_{\text{test}} \subset D$ from the rest $D_{\text{train}} = D \setminus D_{\text{test}}$. For a single held-out test point $\{(x_{\text{test}}, y_{\text{test}})\} = D_{\text{test}}$, the training loss is the cross-entropy on the held-out label:

$$
\mathcal{L}_{\text{PFN}} \;=\; \mathbb{E}_{\big(\{(x_{\text{test}}, y_{\text{test}})\}\cup D_{\text{train}}\big)\sim p(D)}\big[\, -\log q_\theta(y_{\text{test}}\mid x_{\text{test}}, D_{\text{train}}) \,\big].
$$

**Why this recovers the PPD — the derivation.** This is the crux (established in Müller et al. 2022 and reused here). Consider the expected negative log-likelihood as a functional of the model $q_\theta$. Expanding the expectation over the joint prior sampling $p(D) = p(y_{\text{test}}, x_{\text{test}}, D_{\text{train}})$:

$$
\mathcal{L}_{\text{PFN}}(\theta) = -\int p(x_{\text{test}}, D_{\text{train}}) \int p(y_{\text{test}} \mid x_{\text{test}}, D_{\text{train}}) \, \log q_\theta(y_{\text{test}} \mid x_{\text{test}}, D_{\text{train}}) \, dy_{\text{test}} \, d(x_{\text{test}}, D_{\text{train}}).
$$

The inner integral is, up to a constant, the cross-entropy between the true conditional $p(y_{\text{test}}\mid x_{\text{test}}, D_{\text{train}})$ and the model $q_\theta$. Adding and subtracting the entropy of the true conditional, the inner term equals

$$
\underbrace{H\!\big(p(y\mid x,D)\big)}_{\text{constant in }\theta} + D_{\mathrm{KL}}\!\big(p(y\mid x,D)\,\|\,q_\theta(y\mid x,D)\big).
$$

Since the entropy term does not depend on $\theta$, minimizing $\mathcal{L}_{\text{PFN}}$ is equivalent to minimizing the expected KL divergence $\mathbb{E}_{x,D}\big[D_{\mathrm{KL}}(p \| q_\theta)\big] \ge 0$, which is zero iff $q_\theta(\cdot\mid x_{\text{test}}, D_{\text{train}}) = p(\cdot\mid x_{\text{test}}, D_{\text{train}})$ almost everywhere. But the *true* conditional under the joint prior sampling process **is exactly the PPD** — because $p(y_{\text{test}}\mid x_{\text{test}}, D_{\text{train}})$ marginalizes over the latent $\phi$ that generated the whole batch:

$$
p(y_{\text{test}}\mid x_{\text{test}}, D_{\text{train}}) = \int_\Phi p(y_{\text{test}}\mid x_{\text{test}}, \phi)\, p(\phi \mid D_{\text{train}})\, d\phi,
\qquad p(\phi\mid D_{\text{train}}) \propto p(D_{\text{train}}\mid\phi)p(\phi).
$$

So a sufficiently expressive $q_\theta$ trained to convergence approximates Bayesian inference over the prior *without ever representing $\phi$ or $p(\phi\mid D)$ explicitly*. This is the "deterministic-path, full-amortization" extreme of the family in this shard: there is no latent variable at inference; the entire posterior average is baked into the forward pass.

**The TabPFN prior — turning a dataset generator into an inductive bias.** TabPFN's contribution over the generic PFN is a *specific, rich prior for tabular data*:

- **Fully probabilistic hyperparameters (§4.1).** Rather than point estimates, every prior hyperparameter (e.g. average number of SCM nodes) is drawn from a distribution — e.g. a log-scaled uniform. The PPD then integrates jointly over hyperparameters *and* weights, giving a fully-Bayesian treatment that would cost linearly-many ensemble members to approximate otherwise.
- **Simplicity / Occam (§4.2).** Simpler generating graphs (few nodes, few parameters) get higher prior mass, echoing the Speed Prior and cognitive-science findings on human preference for simple explanations.
- **SCM prior (§4.3).** A **Structural Causal Model** is a set of structural assignments (mechanisms) $z_i = f_i(z_{\mathrm{PA}_G(i)}, \epsilon_i)$, where $\mathrm{PA}_G(i)$ are the parents (direct causes) of node $i$ in a DAG $G$, $f_i$ is a possibly-nonlinear deterministic function, and $\epsilon_i$ is a noise variable. To build a synthetic dataset: sample a DAG and functions, designate a subset of nodes $z_X$ as observed features and one node $z_y$ as the target, then draw $n$ samples by sampling all noise variables and propagating through the graph. Features and targets become correlated through forward and backward causation (the target may be a cause *or* an effect of features). The authors frame this as "rung 1.5" on Pearl's ladder — not causal inference, but association-based prediction that *assumes* SCM-like generative structure.
- **BNN prior (§4.4).** Mixed 50/50 with the SCM prior: sample a network architecture and weights, feed inputs through with sampled noise, use outputs as targets.
- **Regression→classification conversion (§4.5).** Scalar labels $\hat y$ become $N_c$-class labels by (i) sampling $N_c \sim p(N_c)$; (ii) sampling $N_c-1$ class-boundary values $B_i$ from the set of continuous targets; (iii) mapping $y_i \leftarrow \sum_j [\,B_j < \hat y_i\,]$ (count of boundaries below the value), then shuffling class labels to remove ordinality.

**Architecture and its ICL attention pattern.** TabPFN encodes each feature-vector-plus-label as a single token. Training tokens attend to each other (permutation-invariant set encoding — no positional encodings), and each test token attends *only to the training tokens* (not to other test tokens), so all test predictions are produced in parallel in one pass and are mutually independent given $D_{\text{train}}$. Two modifications over the base PFN: slightly altered attention masks for faster inference, and zero-padding so one fixed network handles varying feature counts. Concretely: a 12-layer Transformer trained for 18000 batches of 512 synthetic datasets each — 20 hours on 8× RTX 2080 Ti, done exactly once. Test-time ensembling averages 32 forward passes over power-transformed inputs with rotated feature/class indices (a cheap approximation to the invariances the model has not fully learned).

**Empirical claims (§5).** On 18 numerical OpenML-CC18 datasets at a 60-minute budget for competitors: TabPFN (mean rank AUC-OVO ≈ 2.94) beats LightGBM/CatBoost/XGBoost and matches or beats Auto-sklearn 2.0 and AutoGluon, at inference times of 1.3 s (CPU, no ensemble) / 0.05–0.6 s (GPU) versus ~3000 s for the AutoML baselines. TabPFN + AutoGluon ensemble is best overall (uncorrelated errors). Qualitatively (Fig. 4), decision boundaries on moons/circles/iris/wine are smooth with GP-like uncertainty inflation far from data. Surprisingly, the model **generalizes beyond the training-set sizes it saw** (trained on ≤1024, still improves up to 5000 real training points).

**Limitations.** The Transformer's attention makes runtime and memory scale **quadratically in the number of input samples**, capping practical use at small datasets; the prior is tuned for purely-numerical features (categorical/missing-value handling is weaker); classification-only (regression left to follow-ups); and the whole approach is bounded by how well the synthetic prior matches real tabular data.

### Appendix: Section-by-Section Backbone

- **Abstract.** TabPFN = trained Transformer doing supervised classification for small tabular data in <1 s, no tuning, competitive with SOTA. Performs ICL over labeled (x, f(x)) sequences; fully in the weights; set-valued input, whole test set in one forward pass. It is a PFN trained offline once to approximate Bayesian inference on synthetic datasets from a prior blending SCMs with a simplicity preference. 230× (CPU) / 5700× (GPU) speedup vs AutoML on 18 CC18 datasets; validated on 67 more.
- **§1 Introduction.** Tabular data dominated by GBDTs; deep learning underperforms. Radical change: no per-dataset fit; a single forward pass with a Transformer pre-trained on a synthetic tabular prior. Builds on PFNs. In PFNs, you design a dataset-generating algorithm to encode the prior directly. Prior blends BNNs and SCMs with Occam's-razor preference for fewer nodes/parameters. PPD = infinite ensemble over data-generating mechanisms.
- **§2 Background on PFNs.** PPD for supervised learning (Eq. 1, integration over hypotheses $\Phi$). Synthetic prior-fitting: $p(D)=\mathbb{E}_{\phi}[p(D|\phi)]$; training loss = cross-entropy on held-out synthetic examples (Eq. 2); minimizing it approximates the true PPD. Real-world inference: feed $\langle D_{\text{train}}, x_{\text{test}}\rangle$, get PPD in one forward pass, no gradient learning at test time = ICL. Architecture: Transformer, feature-vector+label as tokens, train tokens attend to each other, test tokens attend only to train tokens. Prior work (Müller 2022) did 30-example binary; here scaled to 1000 points, 10 classes, imbalance.
- **§3 The TabPFN.** PFN fitted on the new tabular prior; two architecture tweaks (attention masks for speed, zero-padding for variable features). Prior-fitting once: 12-layer Transformer, 18000 batches × 512 datasets, 20 h on 8 GPUs. Inference: single pass, plus optional 32-pass ensemble over power-transforms and index rotations.
- **§4 A Prior for Tabular Data.** §4.1 Fundamentally probabilistic models (distributions, not point estimates, over hyperparameters; mixture over BNN + SCM priors). §4.2 Simplicity (Occam / Speed Prior; simplicity = few nodes/params). §4.3 SCM prior (structural assignments $z_i=f_i(z_{\mathrm{PA}}, \epsilon_i)$; sample DAG+functions, designate feature nodes $z_X$ and target $z_y$; "rung 1.5" association-based prediction; skip explicit graph, approximate PPD directly). §4.4 BNN prior (sample architecture+weights, propagate inputs; mixed 50/50 with SCM). §4.5 Multi-class conversion (sample $N_c$, sample boundaries, indicator-sum mapping, shuffle labels).
- **§5 Experiments.** §5.1 Toy (moons/circles/iris/wine: smooth, calibrated, GP-like uncertainty). §5.2 Tabular ML tasks (18 numerical CC18 datasets ≤1000 train / 100 feat / 10 classes; baselines KNN, LogReg, XGBoost, LightGBM, CatBoost, Auto-sklearn 2.0, AutoGluon). Results: TabPFN matches 1-hour AutoML in <1 s; Fig. 5 ROC-AUC vs time budget; Table 1 mean ranks (TabPFN ≈ 2.94, +AutoGluon ≈ 2.67). Errors uncorrelated → ensemble best. Generalizes beyond trained sizes (up to 5000).
- **§6 Conclusions & Future Work.** Single Transformer replaces AutoML at 0.4 s. Limitations: small-dataset-only (quadratic scaling), numerical features, classification-only; 17 enumerated follow-ups (scaling, categorical/missing handling, regression, OOD robustness, fairness, causal interventions/counterfactuals).
- **§7 Ethics / §8 Reproducibility.** Positive carbon/accessibility impact; open-sourced code, weights, notebooks; evaluated on public OpenML benchmarks to avoid cherry-picking.
- **Appendix (A–F).** A: scaling limitations (quadratic in samples). B: additional results, inductive-bias analysis (alignment with simple SCM hypotheses), BNN-vs-mixture ablation, generalization-to-larger-sizes figure. C: prior refinements (correlated/categorical features, exponential scaling, missing values), scaling details from 30→1000. E: architecture/training details, differences from Müller 2022. F: baseline search spaces.

---

## 18. Garnelo et al. 2018 — Conditional Neural Processes

**PDF:** `docs/project/references/in_context_learning/sources/Garnelo et al. 2018 - Conditional Neural Processes.pdf` · ICML 2018 · arXiv:1807.01613

### Phase 1: Foundational Overview (Undergraduate-Level)

**The problem in plain terms.** Two families of models sit at opposite extremes. Deep neural networks are flexible and scalable but data-hungry — you train one from scratch for every new function, needing lots of points. Gaussian Processes (GPs) are the opposite: they exploit a prior to infer a function's shape from just a handful of points and give calibrated uncertainty — but they are computationally expensive (cost grows cubically, $O((n+m)^3)$) and it is hard to hand-design a good prior kernel. **Conditional Neural Processes (CNPs)** aim to get the best of both: GP-like data efficiency and uncertainty, but as a neural network trained by gradient descent, scaling linearly.

**The core idea.** A CNP looks at a **context set** of observed input-output pairs $O = \{(x_i,y_i)\}$, squeezes it into a single fixed-length summary vector $r$, and then predicts the output distribution at any query ("target") input $x$ by conditioning on $r$. Concretely: each context pair is passed through a shared neural net to get an embedding, the embeddings are **averaged** (a permutation-invariant aggregation, so order does not matter), and the average is fed together with the target input into a decoder that outputs a mean and variance. That's it — one summary, then cheap per-target predictions.

**How it's trained.** You repeatedly sample a random function (e.g. a random GP curve), give the model a *random subset* of its points as context, and ask it to predict the outputs at all points (observed and unobserved), maximizing the log-likelihood. By seeing many functions with varying amounts of context, the CNP learns a *data-driven prior*: with few context points it predicts the average over all plausible functions (with wide uncertainty); as context grows, predictions sharpen. Crucially the "prior knowledge" now comes from *training data*, not a hand-picked kernel.

**Key findings / initial takeaway.** On 1-D regression, image completion (MNIST, CelebA), and one-shot Omniglot classification, CNPs make sensible few-shot predictions with meaningful uncertainty, at $O(n+m)$ test-time cost instead of $O(nm)$ or $O((n+m)^3)$. With few context points they beat kNN and GP baselines; they are robust to context ordering. The trade-off, stated honestly by the authors: CNPs give up the mathematical guarantees of true stochastic processes (their conditionals need not be consistent with any single prior process) in exchange for flexibility and scalability. The takeaway: **a permutation-invariant encode-aggregate-decode network can amortize few-shot function prediction, learning its prior directly from data.**

### Phase 2: Graduate-Level Deep Dive

**Stochastic-process setup.** Let $O = \{(x_i, y_i)\}_{i=0}^{n-1} \subset X\times Y$ be observations and $T = \{x_i\}_{i=n}^{n+m-1} \subset X$ be target inputs. Assume outputs are a realization of a stochastic process: let $P$ be a distribution over functions $f: X\to Y$; for $f\sim P$, set $y_i = f(x_i)$. $P$ defines a joint over $\{f(x_i)\}$ and hence a conditional $P(f(T)\mid O, T)$; the task is to predict $f(x)$ for $x\in T$ given $O$. For a GP, this conditional is Gaussian with closed form but costs $O((n+m)^3)$.

**The CNP as a directly-parametrized conditional process.** A CNP is a conditional stochastic process $Q_\theta$ that defines distributions over $f(x)$ for $x\in T$, **inheriting permutation invariance** in both $O$ and $T$: for permutations $O', T'$,

$$
Q_\theta(f(T)\mid O, T) = Q_\theta(f(T')\mid O, T') = Q_\theta(f(T)\mid O', T).
$$

Invariance in $T$ is enforced by a **factored** likelihood:

$$
Q_\theta(f(T)\mid O, T) = \prod_{x\in T} Q_\theta\big(f(x)\mid O, x\big).
$$

This factorization is the "deterministic-path" signature of the family: each target is predicted independently given the context summary — which is exactly what makes the model cheap and permutation-invariant, but also what prevents it from producing *coherent joint samples* (there is no modeled covariance between targets). This limitation motivates the latent-variable extension below and, later, the whole ANP/TNP lineage.

**The encode–aggregate–decode architecture (Eqs. 1–3).** The defining characteristic is conditioning on $O$ through a **fixed-dimensional** embedding:

$$
r_i = h_\theta(x_i, y_i) \quad \forall (x_i,y_i)\in O \tag{1}
$$
$$
r = r_1 \oplus r_2 \oplus \cdots \oplus r_n \tag{2}
$$
$$
\phi_i = g_\theta(x_i, r) \quad \forall x_i\in T \tag{3}
$$

where $h_\theta: X\times Y \to \mathbb{R}^d$ is the encoder MLP, $\oplus$ is a **commutative** aggregation operator $\mathbb{R}^d\times\mathbb{R}^d\to\mathbb{R}^d$ (in practice the **mean**, $\tfrac1n\sum_i r_i$), $g_\theta: X\times\mathbb{R}^d\to\mathbb{R}^e$ is the decoder, and $\phi_i$ parametrizes the output distribution $Q_\theta(f(x_i)\mid O, x_i) = Q(f(x_i)\mid\phi_i)$. For **regression**, $\phi_i = (\mu_i, \sigma_i^2)$ parametrizes a Gaussian $\mathcal N(\mu_i, \sigma_i^2)$; for **classification**, $\phi_i$ are the logits of a categorical over $c$ classes. The mean aggregator gives two structural properties: (i) permutation invariance in $O$, and (ii) **streaming/online** updates — since $r_1\oplus\cdots\oplus r_n$ is computable in $O(1)$ from $r_1\oplus\cdots\oplus r_{n-1}$, new observations are absorbed incrementally. Overall test-time complexity is $O(n+m)$.

**The training objective — the amortized conditional-predictive loss.** Let $f\sim P$, $O=\{(x_i,y_i)\}_{i=0}^{n-1}$, and $N\sim\text{uniform}[0,\dots,n-1]$. Condition on the first $N$ points $O_N = \{(x_i,y_i)\}_{i=0}^{N}\subset O$ and minimize the negative conditional log-probability over *all* $n$ outputs (observed and unobserved):

$$
\mathcal{L}(\theta) = -\,\mathbb{E}_{f\sim P}\Bigg[\, \mathbb{E}_N\Big[\, \log Q_\theta\big(\{y_i\}_{i=0}^{n-1}\mid O_N, \{x_i\}_{i=0}^{n-1}\big)\Big]\Bigg]. \tag{4}
$$

The gradient is estimated by Monte Carlo, sampling $f$ and $N$. **Interpretation / derivation of what this trains.** Randomizing $N$ forces the model to predict well across *every* context size — small $N$ (much uncertainty) to large $N$ (little uncertainty). Scoring the *observed* points too (targets include $O_N$) gives the model a signal of the residual uncertainty inherent in $P$. Note the parallel to the PFN loss of §1: both are held-out conditional log-likelihoods under a data/function-generating distribution; the CNP samples $f\sim P$ (a GP or dataset generator), the PFN samples $\phi\sim p(\phi)$ then $D\sim p(D\mid\phi)$ — structurally identical amortization, and minimizing either drives $Q_\theta$ toward the true conditionals of the generating process (the CNP paper is careful to note that unlike the PFN framing, the learned conditionals are **not guaranteed to be mutually consistent** with any single prior process — that consistency is exactly the mathematical guarantee traded away).

**The latent-variable extension (§4.2.3) — introducing the latent path.** The factored model cannot produce coherent samples over multiple targets (no covariance). To fix this, add a latent $z$: compute the context representation $r$, which parametrizes a Gaussian over $z$; sample $z$ **once**, then decode *all* targets conditioned on that single draw. Different draws of $z$ give different globally-coherent functions. Training uses a VAE-style variational lower bound of the log-likelihood, with a **conditional prior** $p(z\mid O)$ (sees only observed pixels) and a **posterior** $p(z\mid O, T)$ (sees observed and target pixels). This is the seed of the deterministic-path vs. latent-path distinction that Kim et al. (§3) formalize: the deterministic path (Eqs. 1–3) gives per-target point predictions; the latent path gives a distribution *over functions* with target-correlated samples, and the latent CNP is explicitly described as "an approximated amortized version of Bayesian deep learning."

**Empirical results.** (i) *1-D regression* on GP-generated curves (fixed and switching kernels): CNP recovers mean and uncertainty from few context points, sharpening as context grows; the switching-kernel task is easy for CNP (just change the training set) but hard for a stationary-kernel GP. (ii) *Image completion* (MNIST, CelebA) cast as regression $f:[0,1]^2\to[0,1]^{(3)}$: with 1 non-informative context point the prediction is the dataset average; more context → image-specific detail; variance localizes to edges. CNP can condition on arbitrary pixel patterns (even unseen at training) and query at unseen resolutions/subpixels. Table 1: with few context points CNP beats kNN and GP on CelebA MSE and is order-robust. (iii) *Omniglot one-shot classification*: convolutional encoder, aggregate within each class; CNP reaches 95.3% (5-way 1-shot) at $O(n+m)$ vs $O(nm)$ for MANN/Matching Networks — comparable accuracy, cheaper.

### Appendix: Section-by-Section Backbone

- **Abstract.** DNNs excel at function approximation but train from scratch per function; GPs exploit priors for fast test-time inference but are expensive and hard to design priors for. CNPs combine both: inspired by stochastic-process flexibility, structured as NNs, trained by gradient descent. Accurate few-shot predictions, scale to large datasets. Demonstrated on regression, classification, image completion.
- **§1 Introduction.** Data-efficiency via two-phase learning (learn domain statistics, then a specific task from few points). Supervised learning as function approximation over observations/targets. Two approaches: (a) random-init parametric $g$ per task (most deep learning, limited prior sharing) vs. (b) probabilistic stochastic-process view (GPs; Bayesian inference over function space, but intractable as data grows). CNPs directly parametrize conditional distributions over functions given observations, permutation-invariant, $O(n+m)$ test-time; trained to maximize conditional likelihood of random target subsets. Emphasis: CNPs do *not* implement Bayesian inference directly and conditionals may be inconsistent.
- **§2 Model.** §2.1 Stochastic processes ($P$ over functions, conditional $P(f(T)|O,T)$; GP as example with $O((n+m)^3)$ cost). §2.2 CNPs ($Q_\theta$ permutation-invariant conditional process; factored likelihood $\prod_{x\in T}Q_\theta(f(x)|O,x)$; encode–aggregate–decode Eqs. 1–3; mean aggregator; Gaussian for regression, categorical for classification; streaming $O(1)$ update; $O(n+m)$). §2.3 Training CNPs (random $N$, loss Eq. 4 scoring both observed and unobserved, MC gradient; shifts prior-knowledge burden from analytic prior to empirical data; three summary properties — conditional-distribution model, permutation invariance, $O(n+m)$ scalability; note on latent-variable extension for coherent samples).
- **§3 Related research.** §3.1 Gaussian Processes (sparse GPs, deep GPs, deep kernel learning). §3.2 Meta-learning (GQN as a special case; PixelCNN/memory-augmented few-shot density estimation; Matching Networks as a CNP with concatenation aggregator + explicit distance kernel, $O(nm)$; neural statistician / variational homoencoder; latent CNP as amortized Bayesian DL).
- **§4 Experimental Results.** §4.1 Function regression (1-D GP data, fixed & switching kernels; 3-layer MLP encoder d=128, mean aggregate, 5-layer decoder; GP is an upper bound; CNP recovers mean+uncertainty, handles switching kernels trivially). §4.2 Image completion (regression over pixels). §4.2.1 MNIST (average-of-digits with one context point; edge-localized variance; simple active exploration by max-variance pixel selection). §4.2.2 CelebA (flexibility in context patterns and target resolutions; Table 1 CNP beats kNN/GP with few context points, order-robust). §4.2.3 Latent variable model (add $z$, sample once for coherent targets; VAE lower bound; conditional prior $p(z|O)$, posterior $p(z|O,T)$; sample diversity shrinks as observations grow). §4.3 Classification (Omniglot one-shot; conv encoder, class-wise aggregation; Table 2: 95.3% 5-way 1-shot at $O(n+m)$).
- **§5 Discussion.** CNP flexible at test time, extracts prior from training data; demonstrated across regression/classification/image completion; relations to GPs, deep learning, meta-/few-shot learning; a step toward learning reusable high-level abstractions (a trained CNP encapsulates statistics of a *family* of functions). Implementations are proofs-of-concept, extensible.

---

## 19. Kim et al. 2019 — Attentive Neural Processes

**PDF:** `docs/project/references/in_context_learning/sources/Kim et al. 2019 - Attentive Neural Processes.pdf` · ICLR 2019 · arXiv:1901.05761

### Phase 1: Foundational Overview (Undergraduate-Level)

**The problem in plain terms.** Neural Processes (NPs) — the latent-variable cousin of CNPs from §2 — have a nagging flaw: they **underfit**. When you show an NP some observed points and ask it to predict at *those very same input locations*, it gets them noticeably wrong (inaccurate means, over-inflated variances). Visually, in 1-D curve fitting the predicted curve misses the context dots; in face-completion the reconstructed top half looks blurry even though the model was given the top half.

**Why it underfits.** In an NP, the encoder mashes the whole context set into a **single fixed-length summary vector** by *averaging* the per-point representations. Averaging gives every context point equal weight, so when the decoder wants to predict at a query location, it cannot easily tell *which* context points are the relevant ones. The mean-aggregation is an information bottleneck.

**The fix: attention.** The paper draws an analogy to GPs, where the **kernel** measures similarity between two input locations and tells you which observed points matter for a given query. Attentive Neural Processes (ANPs) replace the mean-aggregation with a **differentiable attention** mechanism: each target query *attends to* the context points, giving more weight to the relevant ones — exactly like a learned kernel. This preserves permutation invariance (attention over a set is order-independent) while curing the bottleneck. They add **self-attention** among context points (to model interactions between them) and **cross-attention** from each target query to the contexts (to fetch a query-specific summary).

**Key findings / initial takeaway.** ANPs dramatically improve accuracy at context locations, train faster (both in iterations and wall-clock, despite attention's extra cost), and model a wider range of functions than NPs. On 1-D GP regression, dot-product and multihead attention nearly perfectly reproduce context points; on MNIST/CelebA completion, reconstructions become crisp and can even map between resolutions. The cost: complexity rises from $O(n+m)$ to $O(n(n+m))$. Takeaway: **attention is the neural analogue of a GP kernel; bolting it onto NPs fixes the underfitting that a fixed-size mean summary causes.**

### Phase 2: Graduate-Level Deep Dive

**NP background — the deterministic path and the latent path made explicit.** The NP defines an (infinite) family of conditional distributions: condition on context $(x_C, y_C) := (x_i,y_i)_{i\in C}$ to predict targets $(x_T, y_T)$, invariant to context and target ordering. The **deterministic NP** models:

$$
p(y_T \mid x_T, x_C, y_C) := p(y_T \mid x_T, r_C), \qquad r_C := r(x_C, y_C)\in\mathbb{R}^d, \tag{1}
$$

where $r$ is a deterministic, permutation-invariant aggregator (each pair through an MLP, then **mean**), and the likelihood is a Gaussian factored across targets, with mean/variance from an MLP of $x_i$ and $r_C$. This is the **deterministic path**.

The **latent-variable NP** adds a global latent $z$ to capture functional uncertainty via a complementary **latent path**:

$$
p(y_T \mid x_T, x_C, y_C) := \int p(y_T \mid x_T, r_C, z)\, q(z\mid s_C)\, dz, \tag{2}
$$

where $z$ is a factorized Gaussian parametrized by $s_C := s(x_C, y_C)$ (a second permutation-invariant aggregator), and $q(z\mid s_\varnothing) := p(z)$ is the prior. **The path distinction — the conceptual crux of this shard.** The two paths do different jobs: the **latent path** carries a *global* latent $z$ whose single draw induces correlations across *all* target predictions — it models which realization of the stochastic process we are on (global structure); the **deterministic path** carries a *query-specific* representation that models fine-grained *local* structure (which context points are near this query). Kim et al. keep the latent path free of cross-attention *precisely to preserve the global latent's correlating role* — attention is added only to the deterministic path.

**The training objective — ELBO (Eq. 3).** With both paths, parameters are learned by maximizing the evidence lower bound

$$
\log p(y_T \mid x_T, x_C, y_C) \;\ge\; \mathbb{E}_{q(z\mid s_T)}\big[\log p(y_T\mid x_T, r_C, z)\big] \;-\; D_{\mathrm{KL}}\big(q(z\mid s_T)\,\|\,q(z\mid s_C)\big), \tag{3}
$$

optimized over random context $C$ and target $T$ subsets via the reparametrization trick. **Reading the two terms:** the first is a reconstruction term (decode targets from context summary $r_C$ and a latent drawn from the *target*-conditioned posterior $q(z\mid s_T)$); the second regularizes the *context*-summary distribution $q(z\mid s_C)$ toward the *target*-summary distribution $q(z\mid s_T)$ — sensible because contexts and targets come from the same stochastic-process realization (and $C\subset T$ in practice). Note this uses $q(z\mid s_C)$ in place of the usual fixed prior $p(z)$: the "prior" is itself conditioned on the context, exactly as in the latent CNP of §2. Maximum-likelihood learning here minimizes the KL between the (consistent) conditionals of the true data-generating process and the NP's conditionals — so the NP *approximates* a consistent process without being one (it violates context-consistency).

**Attention primitives (§2.2).** Given keys/values $K\in\mathbb{R}^{n\times d_k}, V\in\mathbb{R}^{n\times d_v}$ and queries $Q\in\mathbb{R}^{m\times d_k}$:

- **Laplace (kernel) attention** — parameter-free, keys/queries are the raw $x$-coordinates:
$$
\text{Laplace}(Q,K,V) := WV\in\mathbb{R}^{m\times d_v}, \quad W_{i\cdot} := \text{softmax}\big((-\lVert Q_{i\cdot} - K_{j\cdot}\rVert_1)_{j=1}^n\big).
$$
- **Scaled dot-product attention** — similarity in a learned representation space:
$$
\text{DotProduct}(Q,K,V) := \text{softmax}\!\big(QK^\top/\sqrt{d_k}\big)V \in\mathbb{R}^{m\times d_v}.
$$
- **Multihead attention** — per-head linear projections, dot-product, concatenate, project:
$$
\text{MultiHead}(Q,K,V) := \text{concat}(\text{head}_1,\dots,\text{head}_H)W, \quad \text{head}_h := \text{DotProduct}(QW_h^Q, KW_h^K, VW_h^V).
$$
All are permutation-invariant in the key-value pairs — the property that makes attention a legal drop-in for the NP's set aggregation.

**The ANP architecture (§3) — where attention enters each path.** Two insertions:
1. **Self-attention over context points**, applied *before* mean-aggregation in *both* the deterministic and latent paths, to model interactions among context points (e.g. if many contexts overlap, the query need not weight all of them). Higher-order interactions are modeled by *stacking* self-attention layers (à la Vaswani et al. 2017).
2. **Cross-attention in the deterministic path only**: the mean-aggregation producing $r_C$ is replaced by cross-attention where each target query $x_*$ attends to the context inputs $x_C$ to produce a **query-specific** representation $r_* := r_*(x_C, y_C, x_*)$. This is precisely the mechanism letting each query attend to the context points it deems relevant.

The decoder is unchanged except $r_C \to r_*$. ANP is trained with the *same* ELBO (Eq. 3), Gaussian likelihood $p(y_i\mid x_i, r_*(x_C,y_C,x_i), z)$ and diagonal Gaussian $q(z\mid s_C)$. **If attention is uniform** (all contexts equal weight) the ANP recovers the NP — so the NP is the degenerate uniform-attention special case. Permutation invariance in the contexts is preserved. Cost rises to $O(n(n+m))$ (self-attention across $n$ contexts, plus per-target weights over all contexts), but computations are matrix multiplications parallelizable across contexts and targets, so wall-clock stays competitive.

**Why cross-attention is kept out of the latent path — the derivation of the design choice.** If cross-attention were applied in the latent path, each target would get its own local latent, destroying the *global* latent $z$ that induces cross-target correlations (the whole point of the latent path: one $z$ sample = one coherent function realization). Keeping the latent path with a single global $z$ and giving the deterministic path query-specific attention cleanly separates global structure (latent path, correlations, sample diversity) from local structure (deterministic path, accurate context reconstruction).

**Empirical results.** (i) *1-D GP regression* with random kernel hyperparameters: ANP shows much faster drop in context-reconstruction error and lower target NLL than NP, in both iterations and wall-clock. Dot-product/multihead ANP nearly perfectly predict context points; NP underfits (learns large likelihood noise to explain data); Laplace attention behaves like NP because its similarity is L1 distance in raw $x$-space rather than a learned space. Multihead smooths the non-smooth predictions of raw dot-product. Merely enlarging the NP bottleneck $d$ helps only up to a limit and never matches multihead ANP (which needs 10% of the wall-clock). (ii) *2-D image regression* (MNIST, CelebA): Stacked-Multihead ANP reconstructions are nearly indistinguishable from originals vs. blurry NP; different $z$ samples give diverse-but-coherent faces/digits (evidence $z$ models global structure); heads specialize (one attends to the target pixel, one to a nearby region, one exploits face symmetry by looking at the mirror-image side). The model can map between resolutions ($4\times4 \to 32\times32$, even $32\times32 \to 256\times256$) despite training on a single resolution, since $x$ (pixel location) lives in a continuous space.

### Appendix: Section-by-Section Backbone

- **Abstract.** NPs learn to map a context set of input-output pairs to a distribution over regression functions, with linear complexity and a wide family of conditionals. But NPs **underfit** — inaccurate predictions at observed inputs. ANPs incorporate attention so each input location attends to relevant context points; greatly improves accuracy, speeds training, expands modelable functions.
- **§1 Introduction.** Regression as a distribution over functions (Bayesian view; GPs as non-parametric example). NPs: efficient, linear-cost, arbitrary context size; but underfit (Fig. 1: inaccurate means, over-large variances at context; blurry face reconstruction). Hypothesis: mean-aggregation bottleneck gives equal weight to all context points. Fix: GP-kernel-inspired differentiable attention preserving permutation invariance. Evaluate on 1-D and 2-D regression; ANPs improve reconstruction, speed, expressiveness.
- **§2 Background.** §2.1 Neural Processes (deterministic NP Eq. 1 with $r_C$; latent NP Eq. 2 with global $z$, factorized-Gaussian $q(z\mid s_C)$, prior $q(z\mid s_\varnothing)=p(z)$; both-path model most expressive; ELBO Eq. 3 with reparametrization; properties: scalability $O(n+m)$, flexibility, permutation invariance, but no context-consistency; NP approximates conditionals of a consistent process). §2.2 Attention (key-value + query weighting, permutation invariance; Laplace kernel attention parameter-free; scaled dot-product; multihead per-head projections).
- **§3 Attentive Neural Processes.** Self-attention over contexts (both paths, models context interactions, stackable for higher-order); cross-attention in deterministic path (mean-aggregation → query-specific $r_*(x_C,y_C,x_*)$); latent path keeps global $z$ (no cross-attention, preserves target correlations / global structure vs. local structure in deterministic path). Decoder unchanged ($r_C\to r_*$); same ELBO; uniform attention recovers NP; permutation invariance preserved; complexity $O(n(n+m))$ but parallelizable, wall-clock competitive.
- **§4 Experimental Results.** 1-D GP regression (random & fixed kernel hyperparameters; context reconstruction error and target NLL vs iterations/wall-clock; NP underfits, dot-product/multihead ANP accurate, Laplace ≈ NP; bottleneck-size $d$ sweep shows raising $d$ insufficient; toy Bayesian Optimization proof-of-concept). 2-D image regression (MNIST, CelebA; NP vs Multihead ANP vs Stacked-Multihead ANP; crisp reconstructions, diverse coherent $z$ samples, head-role visualization including symmetry-exploiting head; resolution mapping $4\times4\to32\times32\to256\times256$).
- **§5 Related Work.** GPs (kernel↔attention parallel; Deep Kernel Learning; VIP); Meta-Learning (few-shot classification with attention; neural statistician / variational homoencoder with local+global latents; Vfunc); Generative Query Networks as an NP special case (x=viewpoints, y=frames).
- **§6 Conclusion & Discussion.** ANPs cure underfitting, improve accuracy/speed/expressiveness. Future: cross-attention in latent path with local+global latents (neural-statistician-style for regression); ANPs on text (stochastic fill-in-the-blank); self-attention in the decoder → an Image-Transformer-like model over arbitrary pixel orderings (targets then affect each other, so ordering/grouping matters).
- **Appendices A–E.** A: architectural details (8 heads for multihead). B: 1-D experimental details. C: fixed-kernel results, dot-product non-smoothness explanation, Bayesian Optimization analysis. D: 2-D experimental details (self-attention stacking à la Parmar et al. 2018). E: additional image results, per-head attention visualizations, resolution-mapping figures.

---

## 20. Nguyen & Grover 2022 — Transformer Neural Processes

**PDF:** `docs/project/references/in_context_learning/sources/Nguyen and Grover 2022 - Transformer Neural Processes.pdf` · ICML 2022 · arXiv:2207.04179

### Phase 1: Foundational Overview (Undergraduate-Level)

**The problem in plain terms.** Neural Processes (NPs) and even Attentive NPs (§3) still have two structural problems for **uncertainty-aware meta-learning** — the setting where a model must both predict accurately *and* quantify how uncertain it is, so that its uncertainty can guide decisions (Bayesian optimization, contextual bandits). Problem one: NPs use a latent variable $z$, which makes the marginal likelihood *intractable*, forcing them to optimize a variational lower bound (the ELBO) instead of the true likelihood — and lower bounds do not always give meaningful latents. Problem two: NPs *underfit*; ANPs fix the fit but tend to become *overconfident* and do poorly on sequential decision-making.

**The idea: treat it as sequence modeling.** Transformer Neural Processes (TNPs) drop the latent variable entirely and instead cast the whole problem as **autoregressive sequence modeling**, exactly like a GPT language model. You lay out the context points and target points as a sequence and train a Transformer (with a causal mask) to predict each target's label from all previous points, maximizing the *actual* conditional log-likelihood — no latent variable, no variational bound, and a very expressive predictive distribution.

**The catch and the fix.** A vanilla GPT cannot be dropped in directly: it uses positional encodings that make its output depend on the *order* of the context and target points, but an NP must be **invariant to context order** and **equivariant to target order**. TNPs fix this by (a) concatenating each $x_i$ with its $y_i$ into a single token and *removing positional encodings*, (b) a custom padding-and-masking scheme so the autoregressive structure is respected without leaking order into predictions, and (c) a Monte-Carlo symmetrization to make the prediction equivariant to target order.

**Key findings / initial takeaway.** TNPs come in three flavors trading expressivity for computation: **TNP-A** (autoregressive, most expressive), **TNP-D** (diagonal, assumes targets independent given context), **TNP-ND** (non-diagonal, models a full covariance via Cholesky). Empirically TNPs achieve state-of-the-art on 1-D meta-regression, image completion (EMNIST, CelebA), contextual bandits (wheel problem), and Bayesian optimization, beating CNP/NP/ANP and their bootstrapped variants by large margins. Takeaway: **you can get a better, likelihood-exact neural process by throwing out the latent variable and modeling the context-plus-targets as a sequence with a permutation-respecting Transformer.**

### Phase 2: Graduate-Level Deep Dive

**Background — the NP latent-variable likelihood and its intractability.** With context $C = \{x_i,y_i\}_{i=1}^m$ and targets $T=\{x_i,y_i\}_{i=m+1}^N$, the latent-variable NP likelihood is

$$
p(y_{m+1:N}\mid x_{m+1:N}, C) = \int_z p(y_{m+1:N}\mid x_{m+1:N}, z)\, p(z\mid C)\, dz, \tag{1}
$$

which is intractable, so NPs maximize the ELBO

$$
\log p(y_{m+1:N}\mid x_{m+1:N}, C) \ge \mathbb{E}_{q(z\mid C,T)}\big[\log p(y_{m+1:N}\mid x_{m+1:N}, z)\big] - \mathrm{KL}\big(q(z\mid C,T)\,\|\,p(z\mid C)\big), \tag{2}
$$

viewable as a VAE over target labels conditioned on context: the encoder $q(z\mid C,T)$ is permutation-invariant, and the decoder factorizes $p(y_{m+1:N}\mid x_{m+1:N}, z) = \prod_{i=m+1}^N p(y_i\mid x_i, z)$ (targets independent given $z$).

**The TNP objective — the amortized autoregressive conditional-predictive loss (the core contribution).** TNP treats each set of evaluations $\{x_i,y_i\}_{i=1}^N$ as an ordered sequence, segregates a random subset $(x_{1:m}, y_{1:m})$ as context, and **autoregressively** models the remaining targets:

$$
\mathcal{L}(\theta) = \mathbb{E}_{x_{1:N}, y_{1:N}, m}\big[\log p_\theta(y_{m+1:N}\mid x_{1:N}, y_{1:m})\big] \tag{4}
$$
$$
= \mathbb{E}_{x_{1:N}, y_{1:N}, m}\Bigg[\sum_{i=m+1}^N \log p_\theta(y_i\mid x_{1:i}, y_{1:i-1})\Bigg], \tag{5}
$$

where each conditional is a univariate Gaussian and $m$ is sampled uniformly to define the context/target split. **Contrast with the rest of the shard.** This is the *same* held-out conditional-log-likelihood idea as CNP (Eq. 4 of §2) and PFN (Eq. 2 of §1), but factorized **autoregressively** rather than **fully** (CNP's $\prod_{x\in T}Q(f(x)\mid O,x)$ conditions every target only on the *context*; TNP-A conditions target $i$ on the context *and all previous targets*). This chain-rule factorization is exact — no variational bound, no latent variable — and strictly more expressive because it models target-target dependencies. It is the "no latent path at all, maximally expressive deterministic prediction" corner of the design space: where CNP amortizes a factored posterior predictive and the NP/ANP latent path samples one global $z$ for coherent joint samples, TNP-A obtains joint coherence through the autoregressive chain rule instead of through a latent.

**The two desiderata (Properties 3.1, 3.2) — why a vanilla GPT fails.**
- **Property 3.1 (Context invariance):** for any permutation $\pi$ and $m\in[1,N-1]$,
$$
p_\theta(y_{m+1:N}\mid x_{m+1:N}, x_{1:m}, y_{1:m}) = p_\theta(y_{m+1:N}\mid x_{m+1:N}, x_{\pi(1):\pi(m)}, y_{\pi(1):\pi(m)}).
$$
Permuting the context must not change target predictions.
- **Property 3.2 (Target equivariance):** for any permutation $\pi$,
$$
p_\theta(y_{m+1:N}\mid x_{m+1:N}, x_{1:m}, y_{1:m}) = p_\theta(y_{\pi(m+1):\pi(N)}\mid x_{\pi(m+1):\pi(N)}, x_{1:m}, y_{1:m}).
$$
Permuting target inputs must permute predictions correspondingly.

A vanilla GPT with positional encodings violates both: positional encodings are needed to pair $x_i$ with $y_i$, but they make outputs order-dependent (breaking context invariance), and different target orderings give non-equivariant predictions.

**TNP-A: Autoregressive TNP (§3.1).** To get pairing *without* order-leakage: concatenate $x_i$ and $y_i$ into a single token (so no positional encoding is needed to associate them). For a target point $y_{i>m}$ that must depend on previous pairs *and* its own input $x_i$, introduce **auxiliary padded tokens** $x_{i>m}$ padded with a dummy $0$, appended to the sequence:

$$
\tau = \{(x_1,y_1),\dots,(x_N,y_N),(x_{m+1},0),\dots,(x_N,0)\}. \tag{6}
$$

A custom **attention mask** enforces the autoregressive order: (1) context points $(x_i,y_i)_{i=1}^m$ attend only to themselves; (2) target point $(x_i,y_i)$ for $i>m$ attends to all context and previous target points $(x_j,y_j)_{j=m+1}^{i}$; (3) padded target $(x_i,0)$ for $i>m$ attends to all context and previous targets $(x_j,y_j)_{j=m+1}^{i-1}$. This masking gives **Property 3.1 by construction** (context tokens only see each other, order-independently).

**Achieving Property 3.2 by symmetrization — the derivation.** Any function can be made equivariant to a group by averaging its evaluations over the group. Define the equivariant joint distribution as an average over the permutation group:

$$
\tilde p_\theta(y_{m+1:N}\mid x_{1:N}, y_{1:m}) = \mathbb{E}_\pi\big[\, p_\theta(y_{\pi(m+1):\pi(N)}\mid x_{\pi(m+1):\pi(N)}, x_{1:m}, y_{1:m}) \,\big]. \tag{7}
$$

Because the full permutation group is intractable to enumerate, use a **Monte-Carlo average over randomly sampled permutations**. Consequently TNP-A satisfies target equivariance only *in the limit* (as the number of sampled permutations grows), but remains tractable at training and evaluation. This is the price of TNP-A's expressivity: exact equivariance is only approached.

**TNP-D: Diagonal TNP (§3.2).** Assume targets are conditionally independent given context and inputs:

$$
p_\theta(y_{m+1:N}\mid x_{1:N}, y_{1:m}) = \prod_{i=m+1}^N p_\theta(y_i\mid x_i, x_{1:m}, y_{1:m}). \tag{8}
$$

Drop the real target pairs from the sequence and feed only context points plus padded targets $(x_i,0)_{i=m+1}^N$. This is a multivariate normal with **diagonal covariance**; equivariance (Property 3.2) holds **exactly** without permutation averaging (each target predicted independently, structurally identical to CNP's factorization but with attention). The cost is the strong independence assumption — no target-target correlations.

**TNP-ND: Non-Diagonal TNP (§3.3).** Parametrize a multivariate normal with a **full covariance**:

$$
p_\theta(y_{m+1:N}\mid x_{1:N}, y_{1:m}) = \mathcal{N}\big(y_{m+1:N}\,\big|\, \mu_\theta(x_{1:N},y_{1:m}),\ \Sigma_\theta(x_{1:N},y_{1:m})\big). \tag{9}
$$

As with TNP-D, real target pairs are removed (targets predicted jointly). Two covariance parametrizations: **Cholesky** $\Sigma = LL^\top$ ($L$ lower-triangular with positive diagonal; used in the main experiments) and **low-rank** $\Sigma = \exp(D) + AA^\top$ ($D$ diagonal, $A$ low-rank). Because the number of targets varies, $L$ cannot be output directly by a fixed-size head; instead, the last masked self-attention layer's outputs $z_{m+1:N}$ feed (a) an MLP producing means $\mu_{m+1:N}$ and (b) a second self-attention stack + projection producing vectors $h_i\in\mathbb{R}^p$. Stacking these into $H\in\mathbb{R}^{n\times p}$ (with $n=N-m$), the Cholesky factor is computed as

$$
L = \text{lower}(HH^\top), \qquad H\in\mathbb{R}^{n\times p}, \tag{10}
$$

where $\text{lower}(\cdot)$ keeps the lower triangle. This does not span all lower-triangular matrices but has two virtues: it handles an arbitrary number of targets, and its space complexity is $O(n)$ rather than $O(n^2)$. **The expressivity/tractability ladder:** TNP-A (autoregressive, most expressive, equivariance only in MC limit) $\succ$ TNP-ND (full covariance via Cholesky, exact equivariance) $\succ$ TNP-D (diagonal, cheapest, exact equivariance) — confirmed empirically (Table 1: on RBF, TNP-A 1.63 > TNP-ND 1.46 > TNP-D 1.39 log-likelihood).

**Architecture instantiation.** A GPT-style stack of masked self-attention layers with the custom mask above, no positional encodings, tokens = concatenated $(x_i,y_i)$ (or padded $(x_i,0)$). Self-attention over the full context+target sequence is what supplies the cross-attention-like context selection of ANP *and* the target-target dependencies that ANP's latent path could only approximate.

**Empirical results.** (i) *1-D regression* (train on random-hyperparameter RBF GP, test on RBF / Matérn-5/2 / Periodic): TNPs beat CNP/CANP/NP/ANP/BNP/BANP on 2/3 kernels by large margins (Table 1). (ii) *Image completion* (EMNIST, CelebA): TNP-A best (CelebA 5.82 vs ANP 2.90, BANP 3.09), strong generalization to unseen EMNIST classes (Table 2). (iii) *Contextual bandits* (wheel problem, varying difficulty $\delta$): TNPs achieve lowest cumulative regret across all $\delta$, degrading only slightly as difficulty rises (Table 3), directly demonstrating good uncertainty for sequential decision-making — the setting where ANP's overconfidence hurt. (iv) *Bayesian optimization*: TNPs optimize even the periodic-kernel functions better than baselines despite worse raw regression likelihood there.

### Appendix: Section-by-Section Backbone

- **Abstract.** NPs define distributions over functions and estimate uncertainty, but underfit and often have intractable likelihoods, limiting sequential decision-making. TNPs cast uncertainty-aware meta-learning as sequence modeling, learned via an autoregressive likelihood objective with a transformer architecture respecting context invariance and target equivariance. Knobs trade decoding-distribution expressivity vs computation. SOTA on meta-regression, image completion, contextual bandits, Bayesian optimization.
- **§1 Introduction.** Meta-learning = fast adaptation to unseen tasks from few examples; uncertainty matters for sequential decision-making. NPs = flexible function distributions, VAE-over-datasets structure, amortized functional uncertainty in latents. Two drawbacks: (1) intractable marginal likelihood → variational surrogate → possibly meaningless latents; (2) underfitting (ANP partly fixes fit but is overconfident, poor on decision-making). TNP: autoregressive conditional log-likelihood, no latents, GPT-style causal-mask transformer; but vanilla transformer lacks context invariance / target equivariance → fixes via token concatenation, padding/masking, MC symmetrization. Three variants (A/D/ND).
- **§2 Background.** §2.1 Uncertainty-aware meta-learning (unknown distribution over functions $F$; per-function $D_{\text{train}}$/$D_{\text{test}}$; joint predictive over test set, e.g. diagonal-Gaussian $\mu_j, \sigma_j$). §2.2 Neural Processes (latent likelihood Eq. 1; ELBO Eq. 2; VAE view, permutation-invariant encoder, factored decoder). §2.3 Transformers (self-attention Eq. 3; positional encodings; arbitrary-length sequences).
- **§3 Transformer Neural Processes.** Sequence-modeling framing; autoregressive objective Eqs. 4–5 (univariate-Gaussian conditionals, MC over batch and $m$). Property 3.1 context invariance, Property 3.2 target equivariance; vanilla GPT violates both via positional encodings. §3.1 TNP-A (concatenate $x_i,y_i$; padded tokens Eq. 6; three-rule attention mask; Property 3.1 by masking; Property 3.2 by symmetrization Eq. 7 with MC permutation average, exact only in limit). §3.2 TNP-D (conditional independence Eq. 8; drop target pairs, feed padded targets; diagonal covariance; exact equivariance; strong assumption). §3.3 TNP-ND (full-covariance normal Eq. 9; Cholesky $\Sigma=LL^\top$ and low-rank $\exp(D)+AA^\top$; parameterization Eq. 10 $L=\text{lower}(HH^\top)$, arbitrary target count, $O(n)$ space).
- **§4 Experiments.** §4.1 1-D regression (train random-hyperparameter RBF GP; test RBF/Matérn/Periodic; Table 1 TNPs win 2/3; A > ND > D). §4.2 Image completion (EMNIST, CelebA; Table 2 large gains, TNP-A best, unseen-class generalization; Fig. 3 crisper completions). §4.3 Contextual bandits (wheel problem, difficulty $\delta$; UCB; cumulative regret; Table 3 TNPs best across all $\delta$). §4.4 Bayesian optimization (TNPs optimize periodic functions better despite lower regression likelihood).
- **§5 Related Work.** NP variants (Conditional NP, latent NP, Attentive NP, Bootstrapping NP/BANP, Convolutional CNP); transformers for sequence modeling and few-shot; discussion of latent-free vs latent-variable NPs.
- **Appendix.** A.1 low-rank covariance results; B experimental details / additional metrics (B.1 regression, B.2 image samples, B.3 simple-regret bandit results); architecture and hyperparameter details; open-sourced code (TNP-pytorch), baselines from official BNP implementation.

---

## 21. Reuter et al. 2025 — Can Transformers Learn Full Bayesian Inference In Context?

**PDF:** `docs/project/references/in_context_learning/sources/Reuter et al. 2025 - Can Transformers Learn Full Bayesian Inference in Context.pdf`
**Venue:** ICML 2025 (PMLR 267). Authors: Arik Reuter, Tim G. J. Rudner, Vincent Fortuin, David Rügamer (LMU Munich / NYU / TU Munich / Helmholtz AI / MCML).

### Phase 1: Foundational Overview (Undergraduate-Level)

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

### Phase 2: Graduate-Level Deep Dive

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

## 22. Mittal et al. 2025 — Amortized In-Context Bayesian Posterior Estimation

**PDF:** `docs/project/references/in_context_learning/sources/Mittal et al. 2025 - Amortized In-Context Bayesian Posterior Estimation.pdf`
**Status:** arXiv preprint (arXiv:2502.06601v1, 10 Feb 2025). Authors: Sarthak Mittal, Niels Leif Bracher, Guillaume Lajoie, Priyank Jaini, Marcus Brubaker (Mila / RPI / Google DeepMind / York / Vector).

### Phase 1: Foundational Overview (Undergraduate-Level)

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

### Phase 2: Graduate-Level Deep Dive

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

## 23. Mittal et al. 2025 — In-Context Parametric Inference: Point or Distribution Estimators?

**PDF:** `docs/project/references/in_context_learning/sources/Mittal et al. 2025 - In-Context Parametric Inference - Point or Distribution Estimators.pdf`
**Status:** arXiv preprint (arXiv:2502.11617v1, 17 Feb 2025). Authors: Sarthak Mittal, Yoshua Bengio, Nikolay Malkin, Guillaume Lajoie (Mila / U. Edinburgh).

### Phase 1: Foundational Overview (Undergraduate-Level)

**The problem in plain words.** When you infer a model's parameters from data you can either report **a single best value** (a "point estimate" — maximum likelihood MLE, or maximum-a-posteriori MAP) or **a whole distribution** over the parameters (the Bayesian posterior). Bayesian theory says that, *at optimality*, using the full posterior to make predictions is the right thing to do — you average over all plausible parameter values. But that optimality assumes you can represent the posterior perfectly. In practice, amortized/in-context estimators use limited families (a Gaussian, a normalizing flow) and must generalize to new datasets, introducing an **amortization gap**. So the practical question is genuinely open: **for downstream prediction, is it better to have an in-context network output a point estimate, or a full posterior?**

**The core question, sharpened.** This is the direct sequel to the Amortized paper. It builds a *hierarchical taxonomy* of in-context estimators (Figure 1): **Frequentist** (MLE) and **Bayesian point** (MAP) on the point side; and three flavors of distribution estimators — **sample-based** (need simulator draws: forward-KL Gaussian/flow, score-based diffusion, flow-matching CNF), **variational** (need only the joint density: reverse-KL Gaussian/flow, diffusion samplers like pDEM), and **sample+variational** (symmetric KL). Then it runs all of them head-to-head on a large benchmark (88 tasks → 324 trained models per estimator) judged purely by **predictive performance**.

**Key findings.**
- **Amortized point estimators (MLE/MAP) generally win**, especially on high-dimensional tasks (e.g. inferring the weights of a 2-layer neural net). In the winner-take-all aggregate, point methods win **~71%** of tasks (roughly split MLE/MAP), vs ~9% variational, ~15% sample+variational, ~5% sample-based.
- Distribution estimators **remain competitive only in some low-dimensional problems**.
- Within Bayesian methods, the **simple Gaussian** often beat the fancier normalizing-flow and diffusion posteriors, and **symmetric KL** was the strongest posterior-training signal.
- The authors argue the culprit is **fundamental, not just an engineering gap**: multimodal parameter posteriors (from likelihood symmetries — label-switching in mixtures, weight-permutation symmetries in neural nets) have a **combinatorial explosion of equivalent modes**. Representing all these redundant modes demands ever-more posterior expressivity, yet they all give equivalent predictions — so a point estimate captures the useful information far more cheaply.

**Initial takeaway.** This is the shard's most *deflationary* and most decision-relevant paper: it says that for the *practical goal of prediction*, the elaborate machinery of in-context distribution estimation (including Reuter-style flow matching) often does not pay off versus a plain amortized point estimate — precisely because of parameter-space multimodality/symmetry. It sets the pointed counter-hypothesis against Reuter's "full-distribution wins."

### Phase 2: Graduate-Level Deep Dive

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

## 24. Kang et al. 2026 — Transformers Can Learn Posterior Predictive Distributions In-Context

**PDF:** `docs/project/references/in_context_learning/sources/Kang et al. 2026 - Transformers Can Learn Posterior Predictive Distributions In-Context.pdf`
**Status:** arXiv preprint (arXiv:2605.26713v1, 26 May 2026); to appear ICML 2026 (PMLR 306). Authors: Gyeonghun Kang, Changwoo J. Lee, Xiang Cheng (Duke).

### Phase 1: Foundational Overview (Undergraduate-Level)

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

### Phase 2: Graduate-Level Deep Dive

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

## 25. Mittal et al. 2025 — Iterative Amortized Inference: Unifying ICL and Learned Optimizers

**PDF:** `docs/project/references/in_context_learning/sources/Mittal et al. 2025 - Iterative Amortized Inference - Unifying ICL and Learned Optimizers.pdf`
**Venue:** arXiv (Oct 2025). **Authors:** Sarthak Mittal, Divyat Mahajan, Guillaume Lajoie, Mohammad Pezeshki (Mila / Université de Montréal / FAIR at Meta).

### Phase 1: Foundational Overview (Undergraduate-Level)

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

### Phase 2: Graduate-Level Deep Dive

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

# Part 4 — §4 Emergence / task-diversity / latent-variable conditions for ICL

*When does ICL appear, and is it Bayesian? The data-distributional drivers (burstiness, heavy-tailed vocabulary, dynamic meaning, the Zipf-≈1 sweet spot) and the pretraining task-diversity threshold that separates memorizing the pretraining pool (Bayesian w.r.t. the prior) from solving genuinely new tasks (non-Bayesian).*

---

## 26. Chan et al. 2022 — Data Distributional Properties Drive Emergent In-Context Learning

**PDF:** `docs/project/references/in_context_learning/sources/Chan et al. 2022 - Data Distributional Properties Drive Emergent In-Context Learning.pdf`
**Authors / venue:** Stephanie C.Y. Chan, Adam Santoro, Andrew K. Lampinen, Jane X. Wang, Aaditya K. Singh, Pierre H. Richemond, James L. McClelland, Felix Hill (DeepMind / UCL / Stanford). arXiv:2205.05055.

### Phase 1: Foundational Overview (Undergraduate-Level)

**The question.** Large transformer language models can do "in-context learning" (ICL) — solve a brand-new task from a few examples typed into the prompt, with **no weight updates**. Nobody trained them explicitly to do this; it just *appeared*. Why? This paper's answer: it is not just the transformer architecture — it is a property of the **training data itself**. Certain statistical shapes of the data (which natural language happens to have) *cause* ICL to emerge.

**The setup in plain terms.** The authors build a clean toy world from the Omniglot handwritten-character dataset (1,623 character classes, 20 images each). The model sees sequences of (image, label) pairs — 8 pairs of "context" followed by one "query" image — and must predict the query's label. They then deliberately dial specific statistical properties of the training data up and down and watch whether ICL emerges. They contrast two learning modes:
- **In-context learning (ICL):** answer the query by looking at the examples in *this* prompt (works even for characters never seen in training, whose label is randomly reassigned each sequence).
- **In-weights learning (IWL):** answer by recalling a fixed image→label mapping baked into the weights during training (works only for classes seen in training, and fails when the query class is absent from the context).

**Key findings — which data properties turn ICL on.**
1. **Burstiness.** In natural data an item tends to appear in temporal *clusters* rather than uniformly. They make a fraction of training sequences "bursty" — the query class appears 3 times in the context. More burstiness → more ICL, less IWL.
2. **Many rare classes (large vocabulary + long tail).** Going from 100 → 1,600 → 12,800 training classes (rotations/flips manufacture extra classes) increases ICL. You need **both** burstiness **and** many classes; neither alone suffices.
3. **Dynamic meaning.** (a) *Label multiplicity* — giving each class several possible labels (chosen randomly per sequence, consistent within a sequence) increases ICL. (b) *Within-class variation* — more visual variety per class (full 20-exemplar Omniglot, or added pixel noise) increases ICL. Making the generalization problem harder preferentially *hurt IWL more than ICL*, so ICL won by comparison.
4. **A tradeoff, and how to break it.** In most conditions ICL and IWL *trade off* — a single model could not hold both. The fix: train on a **Zipfian (power-law) marginal** over classes (few very-common classes, long tail of rare ones — exactly like word frequencies). At **Zipf exponent ≈ 1** (the value real languages sit at) the model keeps **both** high ICL (driven by the rare tail) **and** high IWL of the common classes. Uniform data (exponent 0) gives only ICL; very skewed data (exponent 3) gives only IWL.
5. **Architecture also matters.** Under identical data and matched parameter counts, **vanilla RNNs and LSTMs never achieved ICL** — only the transformer did. But a transformer on the *wrong* data distribution also failed. So: "attention is *not* all you need" — architecture **and** data are both necessary.

**Initial takeaway.** ICL is not a free gift of the transformer. It emerges from the *interaction* of the transformer architecture with naturalistic data statistics — burstiness, a heavy-tailed vocabulary of rare items, and dynamic/contextual meaning. Because language natively has all of these (and the magic Zipf-≈1 skew), the paper offers a mechanistic story for *why* ICL "just appears" in LLMs, and a recipe for eliciting it in non-language domains (which are usually deliberately uniformized). Cognitive tie-in: maps onto complementary-learning-systems theory (neocortex ≈ slow weights, hippocampus ≈ transformer context window).

### Phase 2: Graduate-Level Deep Dive

**Task and model formalism.** Each training sequence is a length-17 token stream: 8 image-label pairs (context) plus a query image,
$$ \big(\text{img}_1, \ell_1, \text{img}_2, \ell_2, \dots, \text{img}_8, \ell_8, \text{img}_{\text{query}}\big), $$
and the target is the query label. Images pass through a (non-pretrained) 2-block-per-group ResNet embedder; integer labels through a standard embedding table; both augmented with sinusoidal positional encodings; then fed to a **causal transformer** (12 layers, embedding dim 64, 8 heads by default). The objective is softmax cross-entropy on the *query* prediction only:
$$ \mathcal{L}(\theta) = -\,\mathbb{E}_{\text{seq}}\Big[ \log p_\theta\big(\ell_{\text{query}} \mid \text{img}_1,\ell_1,\dots,\text{img}_8,\ell_8,\text{img}_{\text{query}}\big)\Big]. $$
Crucially — unlike meta-training — **image→label mappings are fixed across training sequences**, and images/classes recur. This is the deliberate "middle ground" between standard supervised learning (fixed mappings, uniform recurrence) and few-shot meta-training (novel mappings every episode).

**Evaluation probes (design of the ICL vs IWL dissociation).**
- *ICL probe (Fig 1c):* a 4-shot 2-way few-shot sequence on **holdout** classes never seen in training; the two classes are randomly assigned to labels $\{0,1\}$ *per sequence*. Because labels are re-randomized, the only way to score above chance $=\tfrac12$ is to read the current context. Accuracy is computed restricted to the two in-context labels (so the model cannot cheat by copying any context label).
- *IWL probe (Fig 1d):* classes carry their **training labels**, but the query class is forced to be **absent from the context** (classes sampled uniformly without replacement within the sequence). No contextual support → the model must retrieve a weight-stored mapping; chance $\approx 1/1600$.

This construction is what makes the paper's claims clean: for the *training* data the two strategies give identical answers (labels are fixed), so the model's behavior on the *ambiguous* probes reveals a learned **bias**, not a correctness difference.

**Burstiness — formal instantiation.** A sequence is "bursty" iff the query class appears exactly 3 times in the context; to prevent a majority-label shortcut, a *second* class also appears 3 times. Non-bursty sequences draw the 8 context pairs i.i.d. uniformly over the class pool. The scalar control knob is $p(\text{bursty})\in\{0,0.5,0.9,1.0\}$ = fraction of bursty sequences. Empirically ICL accuracy is monotone increasing in $p(\text{bursty})$ and IWL accuracy monotone decreasing (Fig 2). A notable dynamical wrinkle: some runs *start* ICL-biased and drift toward IWL over training — ICL can be transient.

**Number of classes.** Holding $p(\text{bursty})=0.9$, ICL rises with the class count $\in\{100, 1600, 12800\}$. The 12,800-class regime is manufactured by applying the 8-element group of rotations $\{0^\circ,90^\circ,180^\circ,270^\circ\}\times$ horizontal flip, with holdout transforms excluded from training. Caveat the authors flag: Omniglot's rotational/mirror symmetries mean the ×8 augmentation *also* injects a label-multiplicity effect (same image → multiple labels), so the 12,800 result partly confounds "more classes" with "dynamic meaning."

**Dynamic meaning — two instantiations.**
- *Label multiplicity* $m\in\{1,2,5,10\}$: each class has $m$ candidate labels; the shown label is sampled per sequence but held consistent within a sequence ("one sense per discourse"). ICL increases with $m$.
- *Within-class variation:* single fixed exemplar (lowest), + resampled Gaussian pixel noise ($\sigma\in\{0.1,0.5\}$), full 20-exemplar Omniglot (highest). More variation → more ICL, bounded above by within-class generalization difficulty.

**The Zipfian sweet spot (the central quantitative result).** The marginal class distribution is
$$ p(X = x) \propto \frac{1}{x^{\alpha}}, \qquad x = \text{class rank},\ \alpha \in [0,\infty). $$
Interpretation of the limits and the interior:
- $\alpha = 0$ (uniform): every class rare → **high ICL, ~zero IWL**.
- $\alpha \to$ large (e.g. 3): a handful of classes dominate (top-3 ≈ 97% of data) → **high IWL of common classes, ICL collapses**, and even IWL of rare classes stays at chance (rare items are *never* memorized, Fig 6e).
- $\alpha \approx 1$: **sweet spot** — both ICL (on holdout) and IWL (on the 10 most-common classes) coexist at high accuracy. The long rare tail induces ICL while the common head is memorized into weights. Real languages sit at $\alpha\approx 1$ (Piantadosi 2014), a striking coincidence.

The mechanistic reading: the rare tail acts like a stream of quasi-meta-training episodes (rare + bursty ⇒ disproportionately likely to recur *within* a context window), while the common head behaves like a stationary supervised dataset the weights can memorize. Skew lets one dataset serve both regimes.

**Architecture ablation.** Matching depth, hidden size, and parameter count (Transformer-12L ≈ 831k params; LSTM-12L ≈ 628k; a 15-run log-uniform LR/warmup sweep per architecture, 90 runs total), vanilla RNN and LSTM **never exceed chance** on the ICL probe (Fig 7), while the transformer succeeds. The transformer even matches-or-beats the recurrent nets on **IWL** (Fig 8) — so the recurrent failure is *not* explainable as "RNNs are merely more IWL-biased." The authors connect the transformer's advantage to modern Hopfield / associative-memory equivalences (Ramsauer et al. 2021; Krotov & Hopfield 2021): attention over a context window is computationally akin to query-based associative retrieval, which recurrent state compression lacks.

**Why this matters for the shard's theme.** Chan et al. give the *data-side* necessary conditions for ICL: (i) burstiness, (ii) a large long-tailed vocabulary of rare classes, (iii) dynamic meaning, and (iv) — for ICL+IWL coexistence — a Zipf-≈1 marginal. These are exactly the levers a designer of *non-language* ICL curricula (e.g. RL environments, which are usually uniformized) would pull. The paper explicitly notes RL environments are typically uniform (citing Chan et al. 2022 "Zipfian environments for RL") and that this may forfeit an ICL capability.

### Appendix: Section-by-Section Backbone

- **Abstract.** ICL emerges from *data distributions*: burstiness + many rare classes + dynamic meaning. Initially ICL/IWL trade off, but a skewed Zipfian marginal lets both coexist. Naturalistic distributions elicit ICL only in transformers, not RNN/LSTM. Data **and** architecture both matter.
- **§1 Introduction.** ICL = rapid generalization from few in-context examples, no gradient updates; contrasts IWL (slow, gradient-based). Meta-learning achieved few-shot via *explicit* meta-training; here ICL is *emergent*. Two candidate causes: novel architecture (transformer) vs distributional qualities of data. Natural data is bursty, Zipfian, with dynamic/context-dependent meaning (polysemy/homonymy/synonymy) — a middle ground between supervised and meta-training data.
- **§2 Experimental Design.** §2.1 Data: Omniglot, 16-token context (8 image-label pairs) + query; labels fixed & recurring (departs from few-shot). Bursty = query class ×3 (+ a distractor class ×3); non-bursty = i.i.d. uniform. §2.2 Model: ResNet + label-embedder → causal transformer (12L, dim 64, 8 heads), softmax CE on query. §2.3 Eval: ICL probe = 4-shot-2-way on holdout classes with randomized $\{0,1\}$ labels (chance ½, scored on 2 labels); IWL probe = training labels, query class absent from context (chance 1/1600).
- **§3 Results.** §3.1 *What promotes ICL:* Burstiness↑ ⇒ ICL↑, IWL↓ (Fig 2); #classes 100→1600→12800 ⇒ ICL↑ (Fig 3, need burstiness too); label multiplicity 1→10 ⇒ ICL↑ (Fig 4); within-class variation↑ ⇒ ICL↑ (Fig 5, harder generalization hurts IWL more). §3.2 *Coexistence:* Zipfian marginal $p\propto x^{-\alpha}$ (Eq 1); sweet spot $\alpha\approx1$ holds both ICL + IWL(common); $\alpha=0$ only ICL; $\alpha$ large only IWL; rare classes never memorized (Fig 6). §3.3 *Architecture:* RNN/LSTM never reach ICL under matched params (Fig 7); transformer also ≥ on IWL (Fig 8).
- **§4 Discussion.** Data properties (burstiness, class number/rarity, dynamic meaning) promote ICL; architecture matters (transformer ≫ recurrent) but is insufficient alone. ICL/IWL "bias" framing (neither is "correct" on the ambiguous probes). Implications: mechanistic account of LLM ICL; counters "LLMs don't do genuine ICL" narrative (success on holdout classes); design non-language datasets with structured/non-uniform distributions; ties to complementary-learning-systems theory (neocortex=weights, hippocampus=context window) and infant statistical learning. Non-uniformity is dual: it hurts supervised/RL but *induces* ICL.
- **Appendix A** training details (500k steps, 16 TPU cores, Adam, warmup→inv-sqrt LR, 3–5 seeds). **B** could extend to novel labels (currently only novel classes). **C** recurrent-vs-transformer details, parameter counts, in-context eval on trained classes (~similar, slightly higher), multi-class eval (same patterns; Zipf-1 models output context labels less often but still above chance).

---

## 27. Raventós et al. 2023 — Pretraining Task Diversity and the Emergence of Non-Bayesian In-Context Learning for Regression

**PDF:** `docs/project/references/in_context_learning/sources/Raventos et al. 2023 - Pretraining task diversity and non-Bayesian in-context learning.pdf`
**Authors / venue:** Allan Raventós*, Mansheej Paul*, Feng Chen, Surya Ganguli (Stanford). NeurIPS 2023. arXiv:2306.15063.

### Phase 1: Foundational Overview (Undergraduate-Level)

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

### Phase 2: Graduate-Level Deep Dive

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

### Appendix: Section-by-Section Backbone

- **Abstract.** ICL on linear regression vs pretraining task diversity. Below a diversity threshold → transformer = Bayesian estimator with the *pretraining* prior (fails new tasks). Above → matches Ridge (Gaussian prior over all tasks), solving new tasks optimally by *deviating* from the pretraining-Bayes optimum. Explores regularization, capacity, task structure.
- **§1 Introduction.** ICL emerges from next-token pretraining, not built into architecture. Xie et al.'s Bayesian-retrieval hypothesis. But $T_{\text{Pretrain}}$ is an unrepresentative subsample of the desired $T_{\text{True}}$; would predict suboptimality on far tasks. Study linear regression because optimal estimators are computable. **Contributions:** (i) low diversity ⇒ dMMSE-like, high ⇒ Ridge-like; (ii) a threshold + phase transition in sequences-per-task limit; (iii) threshold scales ~linearly with $D$, PT error nearly $D$-independent; (iv) weight decay lowers threshold, depth/width raise it. ICL not fully explained by pretraining-Bayes.
- **§2 Problem setup.** Task = latent $w$; $x\sim\mathcal N(0,I_D)$, $y=w^\top x+\varepsilon$. Causal transformer predicts each target from preceding context (Eq 1 loss). $T_{\text{Pretrain}}=U\{w^{(1..M)}\}$; $T_{\text{True}}=\mathcal N(0,I_D)$. Optimal = posterior mean. dMMSE (Eq 2, softmax over pool). Ridge with ridge $=\sigma^2$ (Eq 3). Distance metric $\Delta$ (Eq 4).
- **§3 Experiments.** Base GPT-2-style (8L, 128-dim, 2 heads) or small (4L, 64-dim); $D=8$, $K=16$, $\sigma^2=0.25$; Adam, one-cycle LR, batch 256, 500k steps. §3.1 Threshold: low $M$ (≤$2^6$) → dMMSE, fail new tasks; high $M$ → Ridge, solve new tasks non-Bayesianly (Figs 2–3). Finite-size scaling of sequences-per-task ⇒ algorithmic phase transition (threshold $2^{14}$–$2^{15}$). Learning dynamics $t^*\propto M^{0.47}$ then breaks (Fig 4). Interpolation paths (Eq 5, Fig 5). PT beats *smoothed* dMMSE (Fig 12). §3.2 Threshold rises ~linearly with $D$; PT $D$-robust, dMMSE not (Fig 6). §3.3 Weight decay ↓ threshold; depth/embedding ↑ threshold (small PT) (Fig 7).
- **§4 Related work.** Builds on Xie et al. Bayesian ICL (validated only at low diversity). Prior linear-regression ICL (Garg, Akyürek, von Oswald) used *unlimited* diversity. Gradient-descent-in-forward-pass hypothesis. Meta-learning diversity literature. Kirsch et al. (diversity→ICL on classification). Chan et al. (burstiness/rare classes). Olsson et al. (induction heads).
- **§5 Discussion.** Algorithmic phase transition dMMSE→Ridge at intermediate diversity; threshold scales moderately with $D$; implicit regularization lets PT escape the pretraining prior. Beyond threshold PT learns the *true generative prior* (Gaussian & Laplace). Linear-mode-connectivity: large-$M$ PTs share basin with $M=\infty$. Open: what "tasks"/diversity means for language; scale alone insufficient (below threshold, more data *hurts*).
- **Appendix A** Bayesian MMSE derivation (Eq 6), dMMSE (Eq 7), Ridge completing-the-square (Eq 8). **B** hyperparameters (JAX, TPU, ~4 TPU-hrs/run; $\sigma^2$ scaled with $D$). **C–G** support figures, sequences-per-task invariance, smoothed-dMMSE construction, weight-space analyses.

---

# Part 5 — §5 Hypernetworks & fast-weight programmers (attention ↔ weight generation)

*ICL as weight generation — the bridge to the project's `[[Hypernetwork]]` and FiLM work. Outer-product fast-weight programmers (linear attention ⇔ fast weights) and full-weight hypernetwork generation, both realizing the same idea: a conditioning signal reshapes another network's computation without gradient steps. Ha's row-scaling reduction here is the mathematical parent of FiLM.*

---

## 28. Ha et al. 2017 — HyperNetworks

**PDF:** `docs/project/references/in_context_learning/sources/Ha et al. 2017 - HyperNetworks.pdf`
**Authors / venue:** David Ha, Andrew Dai, Quoc V. Le (Google Brain). ICLR 2017. arXiv:1609.09106.

### Phase 1: Foundational Overview (Undergraduate-Level)

**The core idea.** Normally a neural network's weights are learned parameters you optimize directly. A **hypernetwork** flips this: a *small* network (the "hypernetwork," analogy = genotype) **outputs the weights** of a *larger* network (the "main network," analogy = phenotype). You train them together, end-to-end, with ordinary backprop. Instead of learning a giant weight matrix directly, you learn (a) a compact embedding vector describing each layer and (b) the small hypernetwork that decodes that embedding into the full weight matrix.

**Why bother.** Two payoffs. (1) **Parameter compression via relaxed weight-sharing.** Convolutional nets have *no* weight-sharing across layers (flexible but redundant); recurrent nets impose *hard* weight-sharing across time (compact but rigid, prone to vanishing gradients). Hypernetworks sit *between*: each layer gets its own embedding (its own "personality") but all embeddings pass through the *same* shared decoder — so layers are similar-but-not-identical, and total parameters drop a lot. (2) **Input-dependent, time-varying weights** for recurrent nets: a "dynamic hypernetwork" regenerates the main RNN's weights *at every timestep*, letting the effective weights adapt to the current input and hidden state — something a normal RNN (fixed weights across the whole sequence) cannot do.

**Two flavors.**
- **Static hypernetwork** (for CNNs): each layer $j$ has a *learned* embedding $z^j$; the hypernetwork $g$ maps $z^j\to$ the layer's convolution kernel $K^j$. On CIFAR-10 a single hypernetwork generates *all 36* residual-block kernels; accuracy drops only ~1.25–1.5% while parameters shrink dramatically (e.g. 0.148M vs 2.24M params for a WRN-40-2-scale net). On MNIST a 12,544-weight kernel is reproduced from a 4-number embedding.
- **Dynamic hypernetwork** (HyperRNN/HyperLSTM): a small recurrent "HyperRNN cell" reads $(x_t, h_{t-1})$ and emits embeddings that generate the main RNN's weights *for that timestep*. On character-level language modeling (Penn Treebank, enwik8), handwriting generation (IAM), and neural machine translation (WMT En→Fr), HyperLSTM matches or beats Layer-Norm LSTM and reaches near-SOTA — **challenging the weight-sharing paradigm** for RNNs.

**Initial takeaway.** A hypernetwork is a *weight-generating* function. This is the mechanistic backbone behind "conditioning": rather than feeding a context signal *into* a fixed network, you use the context to *rewrite the network's weights*. That is exactly the abstraction the project's `Hypernetwork` and `FiLM` topics build on — FiLM (feature-wise scale/shift) turns out to be a *memory-efficient special case* of a dynamic hypernetwork (see the weight-scaling-vector trick below). For ICL specifically: a dynamic hypernetwork that regenerates weights from context is one concrete mechanism by which "in-context" information could reprogram computation without slow gradient updates.

### Phase 2: Graduate-Level Deep Dive

**Static hypernetwork for CNNs.** Store layer $j$'s kernel as $K^j\in\mathbb R^{N_{\text{in}} f_{\text{size}}\times N_{\text{out}} f_{\text{size}}}$ (for $j=1,\dots,D$ layers). The hypernetwork produces it from a learned embedding $z^j\in\mathbb R^{N_z}$:
$$ K^j = g(z^j), \qquad \forall j=1,\dots,D. \tag{1} $$
$g$ is a **two-layer linear network**. Slice the kernel into $N_{\text{in}}$ chunks $K^j_i\in\mathbb R^{f_{\text{size}}\times N_{\text{out}} f_{\text{size}}}$. First layer: $N_{\text{in}}$ projection matrices $W_i\in\mathbb R^{d\times N_z}$, biases $B_i\in\mathbb R^d$ (with $d=N_z$ chosen). Second layer: a *shared* tensor $W_{\text{out}}\in\mathbb R^{f_{\text{size}}\times N_{\text{out}} f_{\text{size}}\times d}$ and bias $B_{\text{out}}$:
$$ a^j_i = W_i z^j + B_i, \qquad K^j_i = \langle W_{\text{out}}, a^j_i\rangle + B_{\text{out}}, \qquad K^j = \big(\,K^j_1\ K^j_2\ \cdots\ K^j_{N_{\text{in}}}\,\big). \tag{2} $$
Here $\langle W_{\text{out}}, a^j_i\rangle$ is a tensor–vector contraction over the length-$d$ axis. **Why two layers, not one?** Sharing $W_{\text{out}}$ across the $N_{\text{in}}$ slices makes it far more compact: a one-layer hypernetwork would need $N_z\times N_{\text{in}}\times f_{\text{size}}\times N_{\text{out}}\times f_{\text{size}}$ parameters, whereas the two-layer version totals
$$ \underbrace{N_z D}_{\text{embeddings}} + \underbrace{d(N_z+1)N_{\text{in}}}_{\text{layer 1}} + \underbrace{f_{\text{size}} N_{\text{out}} f_{\text{size}}(d+1)}_{\text{shared layer 2}} \;\ll\; D N_{\text{in}} f_{\text{size}} N_{\text{out}} f_{\text{size}}\ \text{(direct kernels)}. $$
Varying kernel dimensions (as in ResNets, where $N_{\text{in}},N_{\text{out}}$ are integer multiples of 16) are handled by generating a **basic 16-channel kernel** and **tiling**: e.g. a $32\times64$ kernel = 8 basic kernels, each from its own embedding (Eq 3). Larger kernels therefore consume proportionally more embedding vectors. This is a **relaxed weight-sharing** interpolation between a purely recurrent (all layers identical) and purely convolutional (all layers independent) net.

**Dynamic hypernetwork — HyperRNN (full/linear form).** Baseline RNN:
$$ h_t = \phi(W_h h_{t-1} + W_x x_t + b). \tag{4} $$
In the HyperRNN the main-RNN weights *float over time*, generated from timestep embeddings $z_h, z_x, z_b$:
$$ h_t = \phi\big(W_h(z_h)\,h_{t-1} + W_x(z_x)\,x_t + b(z_b)\big),\quad \begin{aligned} W_h(z_h) &= \langle W_{hz}, z_h\rangle\\ W_x(z_x) &= \langle W_{xz}, z_x\rangle\\ b(z_b) &= W_{bz} z_b + b_0 \end{aligned} \tag{5} $$
with $W_{hz}\in\mathbb R^{N_h\times N_h\times N_z}$, $W_{xz}\in\mathbb R^{N_h\times N_x\times N_z}$. A **smaller** recurrent "HyperRNN cell" (hidden size $N_{\hat h}\ll N_h$) computes the embeddings from the concatenated $\hat x_t=[h_{t-1};x_t]$:
$$ \hat h_t = \phi(W_{\hat h}\hat h_{t-1} + W_{\hat x}\hat x_t + \hat b),\quad z_h = W_{\hat h h}\hat h_{t-1}+b_{\hat h h},\ \ z_x = W_{\hat h x}\hat h_{t-1}+b_{\hat h x},\ \ z_b = W_{\hat h b}\hat h_{t-1}. \tag{6} $$

**The memory-efficiency problem and the weight-scaling-vector trick (the FiLM bridge).** Eq. (5) constructs each weight matrix as a linear combination of $N_z$ full-size matrices — costing $N_z\times$ the memory of a basic RNN, infeasible at scale. The fix: instead of *rebuilding* $W$, **linearly scale its rows** by a weight-scaling vector $d(z)\in\mathbb R^{N_h}$ (a linear projection of $z$):
$$ W(z) = W(d(z)) = \begin{pmatrix} d_0(z)\,W_0\\ d_1(z)\,W_1\\ \vdots\\ d_{N_h}(z)\,W_{N_h} \end{pmatrix}. \tag{7} $$
This row-scaling is equivalent to an **element-wise (Hadamard) multiplication**, dropping the overhead from "$N_z\times$ memory" to "$N_z\times$ hidden units." The efficient HyperRNN becomes
$$ h_t = \phi\big(d_h(z_h)\odot W_h h_{t-1} + d_x(z_x)\odot W_x x_t + b(z_b)\big),\quad \begin{aligned} d_h(z_h)&=W_{hz}z_h\\ d_x(z_x)&=W_{xz}z_x\\ b(z_b)&=W_{bz}z_b+b_0 \end{aligned} \tag{8} $$
**This is precisely a FiLM-style feature-wise modulation**: a context-derived vector applies per-row/per-feature multiplicative gain (plus an additive bias via $b(z_b)$) to a fixed weight matrix. The paper explicitly relates it to Recurrent Batch Norm, Layer Norm, and Multiplicative-Integration RNNs — the shared theme being input-conditioned multiplicative scaling of pre-activations, but here the scaling policy is *learned and per-timestep, per-sample* rather than moment-based.

**HyperLSTM.** The base LSTM gates ($y\in\{i,g,f,o\}$):
$$ y_t = W^y_h h_{t-1} + W^y_x x_t + b^y,\qquad c_t = \sigma(f_t)\odot c_{t-1} + \sigma(i_t)\odot\phi(g_t),\qquad h_t = \sigma(o_t)\odot\phi(c_t). \tag{9} $$
Each gate's weights are made functions of gate-specific embeddings $z^y_h, z^y_x, z^y_b$ produced (with optional Layer Norm) by a smaller HyperLSTM cell (Eqs 10–11), then applied via the efficient scaling form:
$$ y_t = \text{LN}\big(d^y_h\odot W^y_h h_{t-1} + d^y_x\odot W^y_x x_t + b^y(z^y_b)\big),\quad d^y_h=W^y_{hz}z_h,\ d^y_x=W^y_{xz}z_x,\ b^y=W^y_{bz}z^y_b+b^y_0. \tag{12} $$
Initialization matters (Appendix A.2.3): scaling vectors initialized to $0.1/N_z$ (following Recurrent BN, aids gradient flow), orthogonal init for $W_h,W_x$, embedding-projection weights zeroed with biases at 1 (so the model starts near an unmodulated LSTM), recurrent dropout in the main cell only.

**Empirical highlights.**
- *CIFAR-10 (static):* Hyper Residual Net 40-2 = 7.23% error at **0.148M** params vs WRN 40-2 = 5.66% at **2.24M** — relaxed weight-sharing costs ~1.5% accuracy for ~15× fewer params.
- *Penn Treebank (char):* Layer-Norm HyperLSTM 1.250 BPC vs Layer-Norm LSTM 1.267; 2-layer Norm HyperLSTM 1.219 (near-SOTA).
- *enwik8:* Layer-Norm HyperLSTM (2048 units) 1.340 BPC (near-SOTA), faster convergence per step.
- *IAM handwriting:* HyperLSTM (900 units) log-loss −1162 beats Layer-Norm LSTM −1096; here LN did *not* combine well.
- *WMT En→Fr NMT:* dropping HyperLSTM cells into GNMT gives BLEU 40.03 (SOTA single model at the time) vs 38.95 for the LSTM baseline.

**Qualitative mechanism.** Visualizing the norm of per-timestep weight changes during generation (Figs 4, 7) shows the main RNN's weights undergo **regime changes** — large adjustments concentrated *between* words/characters, near-static *within* frequent words. So the HyperRNN cell is effectively *switching which generative model is active* moment-to-moment. Histograms of $\phi(c_t)$ (Fig 5) reveal the HyperLSTM's *dynamic* scaling policy differs qualitatively from Layer Norm's moment-based normalization (the HyperLSTM cell often runs saturated) yet reaches similar performance — and stacking the two yields further gains, implying the learned policy exceeds moment normalization.

**Relation to shard / project.** HyperNetworks is the canonical *weight-generation* paper: it formalizes "one network writes another's weights," end-to-end differentiable, and — critically for this project — shows the **row-scaling / element-wise-multiplication reduction (Eqs 7–8, 12) that is the mathematical parent of FiLM**. The dynamic (per-timestep, input-conditioned) variant is a mechanism by which contextual information can *reprogram* a network's computation without gradient updates, directly relevant to how in-context adaptation might be implemented architecturally. Lineage: Schmidhuber's fast weights (1992/93) → HyperNEAT/CPPN (evolutionary weight generation) → this end-to-end-differentiable embedding-based hypernetwork.

### Appendix: Section-by-Section Backbone

- **Abstract.** One network generates weights for another (genotype→phenotype). End-to-end backprop (vs evolutionary HyperNEAT). Relaxed weight-sharing for deep CNNs and long RNNs. Main result: non-shared LSTM weights, near-SOTA on char-LM, handwriting, NMT; respectable image recognition with fewer params.
- **§1 Introduction.** Small hypernetwork generates weights of a large main network from an embedding describing each layer. Embeddings can be *fixed & learned* (static, approx weight-sharing) or *dynamically generated* (recurrent, adapts over timesteps). Mixes well with batch/layer norm.
- **§2 Motivation & Related Work.** Inspired by evolutionary computing (search compact generator, not millions of weights): HyperNEAT/CPPN, Compressed Weight Search (DCT), DPPN, ACDC. Schmidhuber's fast weights (context-dependent weight changes, 1992/93). Related weight-prediction work (Denil, Bertinetto, De Brabandere dynamic filters). Novelty: end-to-end gradient training + application to *recurrent* nets.
- **§3 Methods.** CNN and RNN as ends of a spectrum (no-sharing vs hard-sharing); hypernetworks = relaxed sharing. §3.1 Static (Eqs 1–3): two-layer linear $g$, shared $W_{\text{out}}$ compactness, kernel tiling for varying sizes. §3.2 Dynamic HyperRNN (Eqs 4–6 full form), memory problem, weight-scaling-vector fix (Eqs 7–8, element-wise mult, ties to Recurrent BN / Layer Norm / Multiplicative RNN). HyperLSTM deferred to Appendix A.2.2.
- **§4 Experiments.** §4.1 Static on MNIST ConvNet (99.24% vs 99.28%, 4-dim embedding). §4.2 Static on WRN CIFAR-10 (36 kernels from one hypernetwork; ~1.25–1.5% accuracy cost, big param reduction; Tables 1–2). §4.3 HyperLSTM Penn Treebank (Table 3; matches/beats Layer-Norm LSTM). §4.4 enwik8 (Table 4, near-SOTA). §4.5 IAM handwriting (Table 5; HyperLSTM best, LN doesn't combine). §4.6 NMT WMT En→Fr (Table 6, SOTA single model). Weight-change visualizations (Figs 4–7).
- **§5 Conclusion.** Static + dynamic hypernetworks, end-to-end, fewer params, competitive-to-better across image/LM/handwriting/NMT.
- **Appendix.** A.1 coordinate-based hypernetwork recovers conv-like filters (but poor accuracy, 93.5%). A.2 conceptual diagrams; A.2.2 HyperLSTM equations (9–13); A.2.3 initialization details. A.3 experiment setups/hyperparameters.

---

## Schmidhuber 1992 — Learning to Control Fast-Weight Memories

*(not reviewed — no open-access PDF; MIT-Press paywall. Neural Computation 4(1), 1992. See manifest `[[in_context_learning_sources]]` §5, row 28.)*

---

## 29. Ba et al. 2016 — Using Fast Weights to Attend to the Recent Past

**PDF:** `docs/project/references/in_context_learning/sources/Ba et al. 2016 - Using Fast Weights to Attend to the Recent Past.pdf`
**Authors / venue:** Jimmy Ba (Toronto), Geoffrey Hinton (Toronto/Google Brain), Volodymyr Mnih, Joel Z. Leibo, Catalin Ionescu (DeepMind). arXiv:1610.06258.

### Phase 1: Foundational Overview (Undergraduate-Level)

**The core idea.** A standard RNN has two kinds of memory: (1) **neural activities** — the hidden state, a fast, small (capacity ~$O(H)$ for $H$ hidden units) short-term memory updated every step; and (2) **slow weights** — the connection matrices, high-capacity (~$O(H^2)$) but updated only slowly by gradient descent (long-term memory). This paper adds a **third** kind: **fast weights** — a temporary weight matrix $A$ that changes *much faster than slow weights but slower than activities*. Fast weights store a *high-capacity* ($O(H^2)$) memory of the *recent* hidden states of the current sequence, and act like **attention to the recent past** — but without storing explicit copies of past activity vectors and without a separate learned attention module.

**How it works, intuitively.** Every timestep, the fast-weight matrix decays a little and adds the **outer product** of the current hidden vector with itself (a Hebbian "fire together, wire together" write). Then, when computing the next hidden state, the network runs a brief **inner-loop settling**: the fast weights pull the new hidden state toward *past* hidden states, in proportion to how much each past state resembles the current one (their dot product), discounted by how long ago it occurred. This is mathematically **equivalent** to attending over all recent hidden vectors weighted by dot-product similarity and a decay factor — i.e. a Hebbian associative memory *is* a form of attention. Layer normalization inside the inner loop keeps the dot products from exploding/vanishing and is what makes it work robustly.

**Biological motivation.** Working memory, priming, and attention operate on 100ms–minutes timescales — too slow for pure activation dynamics (~10ms), too fast for long-term synaptic plasticity (minutes–hours). The brain plausibly holds temporary state in **short-term synaptic plasticity** (facilitation/depression, spike-timing-dependent), which is synapse-specific — hence $O(H^2)$ capacity, matching fast weights, not the $O(H)$ of activations. Fast weights are argued to be far more biologically plausible than Neural Turing Machines / Memory Networks (no explicit tape, no read/write addressing — writes are just superimposed on the same fast-changing synapse).

**Key results.** On (1) an **associative retrieval** toy task (memorize key-value pairs, then answer a queried key), fast weights dramatically beat LSTM/associative-LSTM when hidden units are few (1.81% vs ~60% error at 20 units) and converge faster; (2) **MNIST via sequential glimpses** (multi-level visual attention), fast weights match a ConvNet without weight-sharing and beat LSTM/IRNN; (3) **facial expression recognition** (Multi-PIE), fast weights beat LSTM/IRNN (though a ConvNet still wins on near-frontal faces); (4) **reinforcement learning** (partially-observable "Catch"), a fast-weights RNN learns faster than LSTM/RNN, with a larger advantage on the more memory-demanding version.

**Initial takeaway.** Fast weights are a *content-addressable, Hebbian, decaying associative memory* that frees up the hidden activities for computation instead of storage. Two connections for this shard: (a) it is the direct ancestor of the "linear-transformers-are-fast-weight-programmers" line and of dynamic weight generation (Ha's HyperRNN cites the same Schmidhuber fast-weights lineage); (b) it operationalizes "attend to the recent past by storing information in *weights* rather than *activations*" — a mechanistic route to in-context memory without gradient updates. For the project's RL angle, the Catch result shows fast weights help *memory-dependent RL policies*, and the "state vs weight equivalence for query-based access" is the same point Chan et al. invoke (hippocampus ≈ transformer context window).

### Phase 2: Graduate-Level Deep Dive

**Memory taxonomy (capacities).** Hidden activity: short-term, updated every step, capacity $O(H)$. Slow weights: $W$ (hidden→hidden) $O(H^2)$, $C$ (input→hidden) $O(IH)$, output $O(HO)$; updated at sequence end. **Fast weights $A$:** capacity $O(H^2)$, updated every step but decaying. LSTM's gating gives multiplicative *effective*-weight adjustment but still only $O(H)$ short-term capacity — fast weights lift this to $O(H^2)$.

**The fast-weight update rule (Hebbian outer-product with decay).**
$$ A(t) = \lambda\,A(t-1) + \eta\,h(t)\,h(t)^\top, \tag{1} $$
with decay rate $\lambda$ and fast-learning-rate $\eta$ (paper uses $\eta=0.5$, $\lambda=0.9$–$0.95$). Unrolling from $A(0)=0$:
$$ A(t) = \eta\sum_{\tau=1}^{t}\lambda^{t-\tau}\,h(\tau)\,h(\tau)^\top. \tag{3} $$

**Two-step hidden update with inner-loop settling.** The "preliminary" next state uses only slow weights:
$$ h_0(t+1) = f\big(W h(t) + C x(t)\big), $$
then an inner loop of $S$ steps refines it, with the slow-weight term held fixed as a **sustained boundary condition** (square brackets):
$$ h_{s+1}(t+1) = f\Big(\big[W h(t) + C x(t)\big] + A(t)\,h_s(t+1)\Big),\qquad h(t+1)=h_S(t+1). \tag{2} $$

**The key equivalence: fast weights = dot-product attention over past states.** Substitute Eq. (3) into the fast-weight term $A(t)h_s(t+1)$:
$$ A(t)\,h_s(t+1) = \eta\sum_{\tau=1}^{t}\lambda^{t-\tau}\,h(\tau)\,\big[h(\tau)^\top h_s(t+1)\big]. \tag{4} $$
Read this carefully: the bracketed term $h(\tau)^\top h_s(t+1)$ is the **scalar (dot-product) similarity** between a past hidden state $h(\tau)$ and the current one. So $A(t)h_s(t+1)$ is a **decay-weighted ($\lambda^{t-\tau}$) sum of past hidden vectors, each weighted by its similarity to the current state** — structurally identical to attention, except the "attention weights" come from *dot-product content similarity* (not a separately parameterized alignment model as in Bahdanau et al.). During the inner-loop iterations, attention *sharpens* toward past states that most attract the current state (a Hopfield-like retrieval dynamic).

**Why this equivalence matters computationally.** Eq. (4) lets you compute the fast-weight contribution **without ever materializing the $H\times H$ matrix $A$** — you just store the recent hidden vectors and take dot products. When the sequence length is smaller than $H$, $A$ is low-rank and this is cheaper. Crucially it also **rescues mini-batching**: $A$ differs per sequence (so batching over the explicit matrix is awkward), but the "compare against stored hidden vectors" form batches naturally.

**Layer-normalized fast weights (stability).** Raw dot products can vanish/explode with hidden-vector norm. Applying layer normalization $\text{LN}[\cdot]$ on *each* inner-loop iteration:
$$ h_{s+1}(t+1) = f\Big(\text{LN}\big[\,W h(t) + C x(t) + A(t)\,h_s(t+1)\,\big]\Big). \tag{5} $$
This is reported essential for robustness to the choice of $\eta,\lambda$; default runs use LN + outer-product rule with $\eta=0.5$, $\lambda=0.95$.

**Design rationale vs Hopfield / NTM.** Hopfield nets use a one-shot outer-product rule reaching ~$0.15N$ binary-vector capacity; iterative error-correcting rules (Gardner 1988) achieve more but are iterative. Ba et al. deliberately choose the **simple, non-iterative outer-product** write (capacity secondary to simplicity) with **rapid decay** so old memories fade. Versus NTM/Neural-Stack/Memory-Networks: (i) biologically implementable in synapses (no exotic tape); (ii) **no read/write addressing decision** — writes are always-on and superimposed on the same decaying synaptic component; three information sources combine each step — new input via $C$, previous state via $W$, recent-history via $A$.

**Experiments (quantified).**
- *Associative retrieval* (key-value strings like `c9k8j3f1??c`→`9`; 100k train): with few recurrent units the fast-weights RNN crushes LSTM/A-LSTM. At $R=20$ units: Fast weights **1.81%** error vs LSTM 60.81%, IRNN 62.11%, A-LSTM 60.13%; at $R=50$: FW **0%** vs LSTM 1.85%. Faster convergence too (Fig 2). Supports the hypothesis that fast weights let the RNN use its recurrent units more effectively (storage offloaded to $A$).
- *MNIST via glimpses* (multi-level attention, 24 patches of $7\times7$ at two scales; fast weights as a neurally-plausible "cache" for partial results across the coarse/fine hierarchy). At 50 features: FW **7.21%** vs LSTM 12%, IRNN 12.95%; at 200 features FW **0.85%** ≈ ConvNet 0.9%, beating LSTM 1.10%. Fast weights match a ConvNet without biologically-implausible weight sharing.
- *Facial expression* (Multi-PIE, 6 classes, 200 hidden units, 24 glimpses): FW **86.34%** vs LSTM 81.32%, IRNN 81.11%; ConvNet 88.23% still wins on near-frontal faces (the rigid predetermined glimpse policy forfeits the attention model's high-res-selective advantage).
- *RL — partially observable Catch* (A3C; blank observations after frame $M$ force memory): fast-weights RNN learns faster than LSTM/RNN, with a **larger margin on the harder ($N{=}24,M{=}5$) variant** needing more memory (Fig 5).

**Biological inner-loop implementations (Appendix B).** Method 1 (used): store post-$W$-transition inputs as sustained boundary conditions during settling. Method 2 (more biologically plausible): add the identity to $A$ so settling sustains the hidden vector. Equivalent for ReLUs when $A=0$; similar otherwise. Method 1 worked slightly better with LN.

**Relation to shard / project.** This is the *fast-weight* mechanism the shard pairs with Ha's hypernetworks — both descend from Schmidhuber's fast-weights idea (a network producing another network's context-dependent weight changes). Ba et al.'s Eq. (4) — Hebbian outer-product memory ≡ dot-product attention — is the conceptual seed later formalized as "linear transformers / attention are fast-weight programmers." For in-context learning: it shows temporary, task-specific information can be written into (fast) *weights* and retrieved by content, giving in-sequence memory **without slow gradient updates** — the same "adapt within a sequence, no weight training" spirit as ICL, and the same weight-vs-state-equivalence Chan et al. cite for the transformer's hippocampus-like context window. The RL/Catch result is the most direct tie to the project's memory-dependent-policy setting.

### Appendix: Section-by-Section Backbone

- **Abstract.** Add "fast weights" — variables slower than activities, faster than standard weights — to store temporary memories of the recent past; a neurally plausible attention-to-the-past that avoids storing copies of activity patterns.
- **§1 Introduction.** RNN has two memories: hidden activity ($O(H)$, per-step) and slow weights ($O(H^2)+O(IH)+O(HO)$, per-sequence). LSTM helps via increments + multiplicative gates but still $O(H)$ short-term. Need a third memory: higher capacity than activities, faster than slow weights. Prior suggestions (Hinton & Plaut 1987 recursion; Schmidhuber 1993 backprop-trainable) never implemented at scale.
- **§2 Physiology evidence.** Working memory/attention/priming at 100ms–minutes: too slow for activations (10ms), too fast for long-term plasticity. Brain uses short-term synaptic plasticity (facilitation via residual Ca²⁺, depression via neurotransmitter depletion, STDP) — synapse-specific ⇒ $O(H^2)$, matching fast weights.
- **§3 Fast Associative Memory.** Memories reconstructed from weights (Willshaw/Kohonen/Hopfield lineage), not stored as activity copies. Choose simple non-iterative outer-product rule with rapid decay (capacity secondary). Advantages over NTM/Neural-Stack/Memory-Net: biologically implementable, no addressing decision, always-on superimposed writes. Three info sources: input via $C$, previous state via $W$, recent history via $A$. Update rule Eq (1); unrolled Eq (3); two-step hidden update with sustained boundary condition Eq (2); **the attention equivalence Eq (4)** ($A(t)h_s$ = decay×dot-product-weighted sum of past states); enables avoiding full $A$ and enables mini-batching. §3.1 Layer-normalized fast weights Eq (5) for stability ($\eta=0.5$, $\lambda=0.95$).
- **§4 Experiments.** §4.1 Associative retrieval (Table 1, Fig 2; FW ≫ LSTM at small units). §4.2 MNIST glimpse integration / multi-level attention (Table 2; FW ≈ ConvNet, no weight-sharing). §4.3 Facial expression Multi-PIE (Table 3; FW > LSTM/IRNN, < ConvNet on frontal). §4.4 RL agents with memory, partially-observable Catch via A3C (Fig 5; FW learns faster, bigger margin on harder variant).
- **§5 Conclusion.** Attracting each new hidden state toward recent states by dot-product similarity improves RNNs across tasks; LN makes it work; a form of attention to the recent past. Implications for recursion/higher-cognition modeling: fast-weight associative memory frees neurons so connection-knowledge can be applied recursively without storing activity copies.
- **Supplementary.** A experimental details/hyperparameters (init, $\eta=0.5$, $\lambda=0.9$; A3C agents 128 ReLU + 128 recurrent cells). B two biological inner-loop settling methods (sustained boundary vs $A+I$).

---

## 30. Munkhdalai & Yu 2017 — Meta Networks

**PDF:** `docs/project/references/in_context_learning/sources/Munkhdalai and Yu 2017 - Meta Networks.pdf`
· ICML 2017 · arXiv 1703.00837

### Phase 1: Foundational Overview (Undergraduate-Level)

**The problem in plain terms.** A standard neural network needs lots of labelled data and, worse,
*forgets* old skills when you train it on something new. Humans learn a new concept from a single
example and keep their old skills. Meta Networks (**MetaNet**) is an early (2017) "learning to learn"
architecture that gives a network this rapid-adaptation ability by splitting it into two cooperating
learners with weights that change on *different time-scales*.

- A **base learner** does the actual task (e.g. classify an image). It has ordinary **slow weights**
  (trained by gradient descent) plus **fast weights** that are conjured up fresh for each input.
- A **meta learner** is the weight factory. It watches how the base learner is doing — specifically it
  reads the **loss gradients** the base learner produces on a few example images ("meta information") —
  and from those gradients it *generates* the fast weights that will make the base learner good at the
  current task.

The trick that ties them together is **layer augmentation**: each layer runs its input through *both*
the slow weights and the freshly-generated fast weights, applies a nonlinearity to each, and adds the
two activation vectors. Fast and slow weights are "feature detectors in two numeric domains" unified by
the ReLU.

**Key findings.** On the standard one-shot image benchmarks of the day (Omniglot, Mini-ImageNet) MetaNet
beat prior methods by up to 6% (49.2% on 5-way 1-shot Mini-ImageNet vs 43.6% for Matching Nets). It also
showed three properties the project should note: (i) a meta learner trained to parameterize one base
network could parameterize a *different, fixed-weight* CNN at test time; (ii) it could be trained on
5-way tasks and generalize to 20-way (and even 100-way) tasks by hot-swapping a new softmax layer that
the meta learner parameterizes on the fly; (iii) it exhibited limited **continual learning** — training
on MNIST after Omniglot could even *improve* Omniglot accuracy (reverse transfer) up to a point.

**Initial takeaway.** MetaNet is a conceptual ancestor of the whole "weight-generation" arc: it makes
the case that *loss gradients* are a useful conditioning signal, and that generating fast weights (rather
than fine-tuning slow ones) is a viable route to one-shot adaptation. Its limitation — juggling three
weight time-scales via crude additive layer augmentation — is exactly what the later outer-product FWPs
and Transformer hypernetworks streamline.

### Phase 2: Graduate-Level Deep Dive

**Weight-generation mechanism.** MetaNet uses *loss gradients as meta-information* and two learned
weight-generators. Let $b$ be the base learner with slow weights $W$, $u$ the dynamic representation
network with slow weights $Q$, and let $m$ (weights $Z$) and $d$ (weights $G$) be the fast-weight
generators. The full trainable parameter set is $\theta=\{W,Q,Z,G\}$.

*Example-level fast weights.* For each support example $x'_i$ the base learner incurs a task loss and its
gradient w.r.t. the slow weights is the meta-information:
$$
\mathcal{L}_i=\mathrm{loss}_{\text{task}}\big(b(W,x'_i),y'_i\big),\qquad
\nabla_i=\nabla_W \mathcal{L}_i,\qquad
W^{*}_i = m(Z,\nabla_i).
$$
So the generator $m$ is a neural network that maps a *gradient tensor* to a *weight tensor* $W^{*}_i$.
Crucially $m$ is applied **coordinate-wise across the gradient** (parameters $Z$ shared across
coordinates, gradients preprocessed with the Andrychowicz et al. $p=7$ rule) — the same trick used by
learned optimizers — which is what makes generating a large weight tensor tractable.

*Task-level fast weights.* A representation loss $\mathrm{loss}_{\text{emb}}$ (cross-entropy for 1-shot,
contrastive when >1 example/class) produces gradients that the generator $d$ (an LSTM over the $T$
sampled examples) summarizes into a single per-task weight $Q^{*}$:
$$
\mathcal{L}_i=\mathrm{loss}_{\text{emb}}\big(u(Q,x'_i),y'_i\big),\quad
\nabla_i=\nabla_Q \mathcal{L}_i,\quad
Q^{*}=d\big(G,\{\nabla_i\}_{i=1}^{T}\big).
$$

*Storing and reading fast weights (a soft-attention memory).* The example-level fast weights
$\{W^{*}_i\}_{i=1}^{N}$ are stored in a memory $M$, indexed by task-dependent embeddings
$r'_i=u(Q,Q^{*},x'_i)$ collected in $R$. At query time, for input $x_i$ the model embeds it
$r_i=u(Q,Q^{*},x_i)$, computes cosine-similarity attention against the index, and reads a **soft mixture
of stored fast weights**:
$$
a_i=\mathrm{attention}(R,r_i),\qquad
W^{*}_i=\mathrm{softmax}(a_i)^{\top} M .
$$
This is a differentiable, content-addressable weight memory — an idea that reappears in the FWP lineage
(where the "memory" is a single accumulating fast-weight matrix rather than a slot table).

**Layer augmentation (the fast⊕slow integration rule).** For an augmented layer with input $h$, slow
weight $W$, fast weight $W^{*}$ and nonlinearity $\phi$ (ReLU),
$$
\mathrm{AugLayer}(h)=\phi(Wh)+\phi(W^{*}h),
$$
with the final softmax layer aggregating *before* normalization. The paper found that a base learner
built from fast weights **alone** fails to converge (collapses to a constant classifier); the slow-weight
"anchor" is necessary. This is a load-bearing empirical fact for the project: **pure hypernetwork
generation of a whole layer can be unstable; a residual/additive combination with a stably-trained base
path helps** — directly analogous to why FiLM modulates (scales/shifts) a base feature rather than
replacing it.

**Training loop (Algorithm 1, condensed).** (1) Sample $T$ support examples, compute representation-loss
gradients, generate $Q^{*}=d(G,\{\nabla\})$. (2) For each of $N$ support examples, compute task-loss
gradient $\nabla_i$, generate $W^{*}_i=m(Z,\nabla_i)$, store in memory $M$, store index $r'_i$ in $R$.
(3) For each of $L$ training examples, embed, attend over $R$, read $W^{*}_i=\mathrm{softmax}(a_i)^\top M$,
accumulate $\mathcal{L}_{\text{train}}$. (4) Update $\theta=\{W,Q,Z,G\}$ by $\nabla_\theta
\mathcal{L}_{\text{train}}$. Note the outer loss backpropagates *through* the fast-weight generation, so
$m,d$ learn to produce useful weights end-to-end.

**Why this matters for the project.** MetaNet is the "explicit hypernetwork + gradient-as-context" end of
the spectrum. Two of its findings transfer directly to any modulation-based agent: (i) *gradients are an
informative conditioning signal* (relevant if a modulator were ever conditioned on a TD-error or
prediction-error signal), and (ii) *fast weights must be added to, not substituted for, a stable slow
path*. Its RL relevance is speculative in the paper ("MetaNet can readily be applied to parameterize
policies in RL") but never demonstrated — the later FWP papers (Irie 2021/2022) actually do the RL
experiments.

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** Deep nets need large data and forget; humans do one-shot + continual learning via
  meta-learning (Harlow 1949). MetaNet = base learner (task space) + meta learner (task-agnostic meta
  space) + external memory. Three weight time-scales: slow (gradient descent), task-level fast (per task),
  example-level fast (per input). Meta-information = **loss gradients**. Two losses: representation
  (embedding) loss and main task loss.
- **§2 Related Work.** One-shot learning via generative/metric methods (Lake 2015 BPL; Koch Siamese;
  Vinyals Matching Nets; Santoro MANN/NTM; Ravi&Larochelle LSTM optimizer). Meta-optimizers
  (Andrychowicz 2016). Fast weights: Hinton&Plaut 1987, Ba 2016, Schmidhuber 1992/1993. Weight
  generation: Gomez&Schmidhuber (RNN→fast weights), De Brabandere dynamic filters, Ha et al. HyperNetworks
  (slow weights for RNNs). MetaNet generates fast weights at *two* time-scales; novel **layer
  augmentation** for integration. Ties to memory-augmented NNs.
- **§3 Meta Networks.** Two modules + memory; three procedures: meta-info acquisition, fast-weight
  generation, slow-weight optimization. Algorithm 1 (one-shot SL). Task = support set + training set,
  labels consistent within task but vary across tasks.
  - **§3.1 Meta Learner.** Representation net $u$ (slow $Q$, task-fast $Q^{*}$); generators $m$
    (Eq.1, gradient→example fast weight $W^{*}_i$, MLP with weights $Z$) and $d$ (Eqs.2–4, gradient
    summary→task fast weight $Q^{*}$, LSTM with weights $G$). Contrastive-loss variant (Eqs.6–7) for
    >1 example/class. Memory read via cosine attention + softmax (Eqs.8–9).
  - **§3.2 Base Learner.** $b$ with slow $W$ + example-fast $W^{*}$; produces gradient meta-info
    (Eqs.10–11); prediction $P(\hat y_i|x_i,W,W^{*}_i)=b(W,W^{*}_i,x_i)$ (Eq.12). Can consume $r_i$
    instead of $x_i$ to share representations.
  - **§3.3 Layer Augmentation.** $\phi(Wh)+\phi(W^{*}h)$; fast-only base learner fails to converge.
- **§4 Results.** Omniglot previous split (Table 1): MetaNet 98.95/98.67/97.11/97.0 (5/10/15/20-way),
  +0.5–2% over SOTA; MetaNet− (no task-fast $Q^{*}$) close but worse; MetaNet+ (extra task-fast in base)
  hurts. Mini-ImageNet (Table 2): 49.21% 5-way 1-shot, +6%. Omniglot standard split (Table 3): 95.92%
  20-way, ~human. Generalization: N-way train / K-way test (Table 4) — trained-harder-tested-easier helps,
  10-way→100-way ≈65%; rapid parameterization of a *fixed-weight* new CNN works; meta-level continual
  learning (Omniglot→MNIST) shows reverse transfer up to ~2400 MNIST trials then mild forgetting (−1.7%).
- **§5 Discussion.** Loss gradients are a promising, problem-independent meta-info; layer augmentation is
  a bottleneck when many weight time-scales coexist; future: discover own augmentation schema, apply to RL
  policies and sequence models.
- **Appendices A–B.** Training details (CNN backends, LSTM $d$ 20 units, MLP $m$ 20 units, Adam lr 1e−3);
  MNIST out-of-domain (MetaNet 74.8% vs Matching Net 72.0%).

---

## 31. Schlag et al. 2021 — Linear Transformers Are Secretly Fast Weight Programmers

**PDF:** `docs/project/references/in_context_learning/sources/Schlag et al. 2021 - Linear Transformers Are Secretly Fast Weight Programmers.pdf`
**Venue:** ICML 2021. **Authors:** Imanol Schlag*, Kazuki Irie*, Jürgen Schmidhuber (IDSIA).

### Phase 1: Foundational Overview (Undergraduate-Level)

**The question in plain terms.** Standard transformers attend over the whole sequence, which costs time quadratic in sequence length and memory that grows with length. "Linear transformers" replace the softmax with a kernel trick so cost is linear and memory is a *fixed-size matrix*. This paper's headline: that fixed-size matrix is **not new** — it is exactly the "fast weights" idea Schmidhuber introduced in 1991. A slow network (the trained weights) learns to write key→value associations into a fast-changing weight matrix via **outer products**, and reads them back by matrix-vector multiply. So a linear transformer is "secretly" a **Fast Weight Programmer (FWP)**.

**Analogy.** Think of the fast-weight matrix as a whiteboard. Each new token writes an association (key, value) onto the board by adding an outer product $\mathbf v\otimes\mathbf k$. When a query arrives, you multiply the board by the query to read back the value whose key best matches. The trained ("slow") network only learns *how to generate* the keys, values, and queries — the whiteboard content is computed on the fly, in-context.

**Key findings.**
1. **Formal equivalence.** Linear self-attention (softmax removed / kernel-linearized) is algebraically identical, up to normalization, to the 1991–93 outer-product FWP. The accumulated attention matrix $\mathbf W^{(i)}=\sum_j \mathbf v^{(j)}\otimes\phi(\mathbf k^{(j)})$ is the fast-weight memory.
2. **Capacity limit.** Because associations are summed into a finite $d_{\text{dot}}\times d_{\text{value}}$ matrix, you can only store ~$d_{\text{dot}}$ mutually-retrievable associations before keys stop being orthogonal and retrieval returns interfering mixtures. Once sequence length exceeds $d_{\text{dot}}$ the model is in an **overcapacity regime**.
3. **The Delta rule fix.** The pure additive update can't *erase* stale associations. They replace it with a **delta-rule** update: before writing a new value, read out the value currently stored under that key, and write only the *correction*, scaled by a learned, dynamic "write strength" $\beta^{(i)}$. This is the **DeltaNet** — a linear transformer that can edit its own memory.
4. **DPFP kernel.** A new deterministic, parameter-free feature map $\phi$ that projects keys/queries up to a higher-dimensional, near-orthogonal space (raising capacity) without random sampling (unlike Performer/FAVOR+).
5. **Empirics.** Synthetic retrieval confirms the capacity cliff at $d_{\text{dot}}$; DeltaNet beats the sum-rule on memory-editing tasks, WMT14 translation, and WikiText-103 language modeling, and — with fast-weight carry-over across segments — runs unbounded context with a tiny fixed state.

**Initial takeaway.** This paper is the historical + mathematical anchor of the whole "context as weights" corpus. It shows the transformer's context can *literally be a weight matrix built by outer products* — the exact algebra a hypernetwork or FiLM layer performs, but accumulated over the sequence. **Bridge to the project:** the outer-product write $\mathbf v\otimes\mathbf k$ and the delta-rule editable-memory view are the raw material behind "attention = hypernetwork" (Schug, §12, over the head index instead of the key index) and behind the neuromodulatory / fast-plasticity framings (fast weights ≈ synaptic modulation, explicitly cited to von der Malsburg 1981).

### Phase 2: Graduate-Level Deep Dive

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

## 32. Irie et al. 2021 — Going Beyond Linear Transformers with Recurrent Fast Weight Programmers

**PDF:** `docs/project/references/in_context_learning/sources/Irie et al. 2021 - Going Beyond Linear Transformers with Recurrent Fast Weight Programmers.pdf`
· NeurIPS 2021 · arXiv 2106.06295

### Phase 1: Foundational Overview (Undergraduate-Level)

**The setup.** A "linear Transformer" is a Transformer whose softmax attention has been replaced by a
cheaper kernel, giving linear-time, constant-memory sequence processing. A 2021 insight (Schlag et al.,
Katharopoulos et al.) is that a linear Transformer is *exactly* an old idea from the '90s: a **Fast Weight
Programmer (FWP)** — a slow network that continually rewrites the weights of a fast network by adding
rank-one outer products. In existing linear Transformers, both the slow and fast nets are trivially
simple: single feedforward layers. This paper asks: *what if we make them recurrent?*

**The idea in plain terms.** Think of a small "fast" network whose weight matrix $W$ acts like a piece of
scratch memory. At every time step the "slow" network reads the input, produces a key–value pair, and
*writes* it into $W$ via an outer product; it also produces a query that *reads* from $W$ to make the
output. The authors' contribution is to add **recurrence** in two places: (i) make the fast network
itself an RNN or LSTM (so it has its own recurrent hidden state on top of the fast-weight memory), and
(ii) let the slow network's key/value/query depend on the fast net's previous output. These give
**Recurrent Fast Weight Programmers (RFWPs)** — models that behave like both Transformers and RNNs.

**Key findings.** On WikiText-103 language modelling the recurrent variants match or beat the linear
Transformer and even, when run without context truncation, beat a regular Transformer restricted to a
256-token window. On synthetic algorithmic tasks (code execution — track the values of several variables;
sequential ListOps — evaluate nested list operations) the recurrent fast nets are markedly more robust
than the plain Delta Net, especially at higher difficulty. **The result the project should care about:**
dropped in as a replacement for the LSTM in an IMPALA actor-critic on Atari 2600, the small RFWPs deliver
*large* score improvements over the LSTM baseline across many games.

**Initial takeaway.** This paper is the cleanest statement of the "linear attention ⇔ fast-weight
programming" equivalence *and* the one that pushes it into RL. For a project weighing recurrent modulated
architectures for a partially-observable grid world, the Delta RNN / Recurrent Delta Net are directly
relevant: they are constant-memory recurrent sequence models with a *content-addressable, continually
rewritten weight memory* — a plausible substrate for context-dependent behaviour.

### Phase 2: Graduate-Level Deep Dive

**General FWP formulation.** An FWP with trainable slow parameters $\theta_{\text{slow}}$ maps an input
sequence $\{x^{(t)}\}$ to outputs $\{y^{(t)}\}$ through a two-network system:
$$
\theta^{(t)}_{\text{fast}},\,q^{(t)} = \mathrm{SlowNet}\big(\{x^{(j)}\}_{j\le t},\{y^{(j)}\}_{j<t},
\{\theta^{(j)}_{\text{fast}}\}_{j<t},\{q^{(j)}\}_{j<t};\theta_{\text{slow}}\big),
$$
$$
y^{(t)} = \mathrm{FastNet}\big(\{q^{(j)}\}_{j\le t},\{y^{(j)}\}_{j<t};\theta^{(t)}_{\text{fast}}\big).
$$
The slow weights change only by gradient descent (fixed at test time); the fast weights
$\theta^{(t)}_{\text{fast}}$ can change *every step*. Because a weight matrix is too large to emit
directly, the fast weights are generated **incrementally** via an update rule:
$$
z^{(t)},q^{(t)}=\mathrm{SlowSubnet}(\cdots;\theta_{\text{slow}}),\qquad
\theta^{(t)}_{\text{fast}}=\mathrm{UpdateRule}\big(\theta^{(t-1)}_{\text{fast}},z^{(t)}\big).
$$
The update rule is the FWP's "elementary programming instruction."

**Linear Transformer as the additive-outer-product FWP.** With a kernel feature map $\phi$ (absorbed into
$k,q$ notation) the slow net emits key/value/query by fixed projections, writes a rank-one outer product,
and reads with the query:
$$
k^{(t)},v^{(t)},q^{(t)}=W_k x^{(t)},\,W_v x^{(t)},\,W_q x^{(t)},
$$
$$
W^{(t)}=W^{(t-1)}+v^{(t)}\otimes k^{(t)},\qquad
y^{(t)}=W^{(t)}q^{(t)}.
$$
This is the **sum update rule** — purely additive. It is *exactly* linearized self-attention: the fast
weight matrix $W^{(t)}=\sum_{j\le t} v^{(j)}\otimes k^{(j)}$ accumulates the key–value outer products, and
$W^{(t)}q^{(t)}=\sum_{j} v^{(j)}(k^{(j)\top}q^{(t)})$ is attention with unnormalized kernel scores. **This
equivalence is the conceptual keystone of the whole shard.**

**Delta Net — the delta-rule update.** Pure addition saturates memory capacity (keys collide). The Delta
Net (Schlag et al. 2021) replaces the write with a delta-rule correction: first *read back* the value
currently stored at key $k^{(t)}$, then write only the *residual*, scaled by a generated learning rate
$\beta^{(t)}=\sigma(W_\beta x^{(t)})$:
$$
\bar v^{(t)}=W^{(t-1)}k^{(t)},\qquad
W^{(t)}=W^{(t-1)}+\beta^{(t)}\big(v^{(t)}-\bar v^{(t)}\big)\otimes k^{(t)}.
$$
Interpretation: this is one step of online gradient descent on $\tfrac12\|W k^{(t)}-v^{(t)}\|^2$ with
step size $\beta^{(t)}$ — the model *learns to write* by error-correction, not blind accumulation. (This
"delta rule ≈ inner-loop gradient step" reading is the bridge to the [[Hypernetwork]]/ICL-as-GD papers in
§2 of the corpus.)

**Contribution 1 — recurrent fast nets.** Any architecture can be "made fast." A plain RNN layer
$h^{(t)}=\sigma(Wx^{(t)}+Rh^{(t-1)})$ becomes fast by replacing $W,R$ with delta-updated fast matrices
$W^{(t)},R^{(t)}$. The paper's concrete **Delta RNN** keeps the linear-Transformer read but adds a
recurrent fast term:
$$
y^{(t)}=W^{(t)}q^{(t)}+R^{(t)}f\big(y^{(t-1)}\big),\qquad f=\mathrm{softmax},
$$
where $R^{(t)}\in\mathbb R^{d_{\text{out}}\times d_{\text{out}}}$ is a *second* fast weight matrix,
generated by the same delta rule with its own slow weights. The softmax on the recurrent query is
required so keys/queries stay non-negative and sum to one — the stability condition the delta rule
demands. A **Delta LSTM** extends this to six fast weight matrices (best LM perplexity, but expensive:
14K words/s vs 41K for Delta RNN). A **Delta MLP** stacks $K$ fast layers
$h^{(t)}_k=W^{(t)}_k f(h^{(t)}_{k-1})$; it *underperforms* — evidence that generating all layers'
weights at once is worse than letting the slow net see intermediate activations.

**Contribution 2 — recurrent slow nets (Recurrent Delta Net, RDN).** Make the slow net's projections
depend on the fast net's previous output $y^{(t-1)}$:
$$
\begin{aligned}
k^{(t)}&=W_k x^{(t)}+R_k\tanh(y^{(t-1)}), &
v^{(t)}&=W_v x^{(t)}+R_v\tanh(y^{(t-1)}),\\
q^{(t)}&=W_q x^{(t)}+R_q\tanh(y^{(t-1)}), &
\beta^{(t)}&=\sigma\big(W_\beta x^{(t)}+R_\beta\tanh(y^{(t-1)})\big).
\end{aligned}
$$
With these four extra recurrent connections the whole system becomes a proper RNN while retaining
constant memory and linear time.

**Complexity.** All variants keep the linear Transformer's constant state size and linear time in
sequence length; the extra cost is per-step compute (custom CUDA kernels). Delta RNN and RDN are the
practical sweet spot.

**RL result (project-relevant).** Setup: IMPALA / V-trace actor-critic (Torchbeast), the large 15-layer
residual conv stem, with the single 256-node LSTM replaced by an RDN (2 layers, hidden 128, 4 heads,
FF 512) or Delta RNN. Trained per-game on standard Atari 2600. RFWPs give large *relative improvements
over the linear Transformer* and beat LSTM in most of 20 games. **Takeaway for us:** a fast-weight
recurrent memory is a viable, constant-memory drop-in for an LSTM in an on-policy actor-critic —
architecturally close to the recurrent policies used in the project's grid world, and the delta-rule
write gives a principled "context-dependent weight" mechanism that is stronger than plain additive
attention.

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** Transformers are quadratic-time and grow state linearly; problematic for
  auto-regressive / long / infinite sequences and for RL in POMDPs (still LSTM-dominated). Linearized
  attention → linear Transformers (Katharopoulos, Performer, Peng). This work: adopt the FWP view; add
  recurrence to slow and fast nets → RFWPs. Contributions: (1) study novel powerful FWPs; (2) augment
  linear Transformers with recurrence.
- **§2 Background on FWPs.**
  - **§2.1 General formulation** (Eqs.1–4): SlowNet/FastNet; incremental UpdateRule decomposition.
  - **§2.2 Linear Transformers as FWPs.** Sum update rule (Eqs.5–7, rank-one outer product = linearized
    attention). Delta Net (Eqs.8–10): delta rule with generated $\beta^{(t)}$ and read-back $\bar v^{(t)}$;
    fixes capacity problem. Multi-head version. "Other approaches": Hypernetworks scale rows of a slow
    matrix; weight compression; dynamic conv / LambdaNets / plasticity.
- **§3 FWPs with slow or fast RNNs.**
  - **§3.1 Fast-net extensions.** Making any net fast (RNN with fast $W^{(t)},R^{(t)}$). **Delta RNN**
    (Eq.12). **Delta LSTM** (6 fast matrices, App.A.2). **Delta MLP** (Eqs.13–14, deep fast net) and
    **Delta Delta Net** (Delta Net as fast net — hierarchical FWP).
  - **§3.2 Slow-net extensions.** **Recurrent Delta Net (RDN)** (Eqs.15–18): recurrent connections into
    $k,v,q,\beta$ from $y^{(t-1)}$.
  - **§3.3 Related models.** RFWPs as memory-augmented RNNs; Schmidhuber 1993 / Ba et al. 2016 recurrent
    FWP; Fast Weight Memory; Metalearned Neural Memory; residual/layernorm/multi-head deep Transformer
    scaffolding.
- **§4 Experiments.**
  - **§4.1 Language modelling (WikiText-103, Table 1).** Transformer 34.1 test ppl; Linear Transformer
    38.3; Delta Net 35.2; Delta MLP worse (36.8); Delta RNN 35.0; Delta LSTM best (33.8, 47.3M params);
    RDN 35.2. Full-context Delta RNN 32.8 / RDN 33.6 beat the 256-window Transformer. Speeds: LT 66K,
    Delta Net 63K, Delta RNN 41K, RDN 35K, Delta LSTM 14K words/s (Transformer 33K).
  - **§4.2 Code execution.** Track 3 or 5 variables. LSTM best (99.0/93.2%); Linear Transformer fails
    (0%); Delta RNN stable at 5 vars (85.1%) vs Delta Net unstable (61.4±20).
  - **§4.3 Sequential ListOps.** Depth 10/15. LSTM best at depth 10 (88.5%) but collapses at depth 15
    (24.4%); mutable-memory Transformer variants beat regular + linear Transformers; RDN 79.2% at depth 15.
  - **§4.4 RL on Atari 2600.** IMPALA/V-trace; replace LSTM with RDN or Delta RNN; large relative
    improvements over LT across ~20 games; small RDN already strong.
- **Appendices.** A: model/ablation details (incl. Delta LSTM equations, activation-function ablation,
  Delta-Delta dimensionality). B: task descriptions + extra results. C: RL hyperparameters.

---

## 33. Irie et al. 2022 — A Modern Self-Referential Weight Matrix That Learns to Modify Itself

**PDF:** `docs/project/references/in_context_learning/sources/Irie et al. 2022 - A Modern Self-Referential Weight Matrix That Learns to Modify Itself.pdf`
· ICML 2022 · arXiv 2202.05780

### Phase 1: Foundational Overview (Undergraduate-Level)

**The premise.** "A network's weight matrix is its program." Ordinary training freezes that program after
training. But the world keeps changing after training stops. A **self-referential weight matrix (SRWM)**
is a single weight matrix that keeps *rewriting itself* at runtime — in principle it can meta-learn to
learn, and even meta-meta-learn. The 1990s proposed such nets but no one made them work at scale; this
paper builds a practical version using the modern fast-weight-programmer machinery (outer products +
delta rule).

**The key move.** In a normal FWP (previous paper) there are two separate matrices: a fixed *slow* one
that decides how to update, and a *fast* one that gets updated. The SRWM **collapses these into one
matrix** $W_t$ that produces the outputs *and* produces the very key/value/query/learning-rate signals
that it then uses to modify itself. The only thing trained by gradient descent is the *initial* matrix
$W_0$; after that, $W_0$ has to encode its own self-modification algorithm.

**Key findings.** (i) On standard few-shot image classification (Omniglot, Mini-ImageNet) the SRWM is
competitive with the plain Delta Net and with generic sequence-model baselines like SNAIL — a "single
self-modifying matrix works about as well as separate slow+fast nets." (ii) In a **sequential multi-task**
setting — feed a stream of Omniglot images, then suddenly switch to Mini-ImageNet — the SRWM *adapts
itself on the fly* and beats the Delta Net (53.3 vs 50.4% total on the second task), showing genuine
runtime self-adaptation. (iii) In **multi-task RL on ProcGen** (6 procedurally-generated games trained
jointly) the SRWM is competitive/strong, and a "Fake SR" ablation (SRWM with the self-modification turned
off) isolates the benefit of self-modification.

**Initial takeaway.** The SRWM is the most extreme point on the shard's spectrum: not "generate a target
network's weights" but "a network that programs *itself*." For the project it is the strongest existing
demonstration that a *single self-modifying recurrent weight memory* can serve as the adaptive core of an
IMPALA-style RL agent in a procedurally-varying environment — the closest published analog to an agent
that must re-tune its own behaviour to shifting contexts without external gradient steps.

### Phase 2: Graduate-Level Deep Dive

**Recap: the DeltaNet backbone.** The SRWM is derived by making the DeltaNet's slow weight matrix fast /
self-referential. DeltaNet transforms $x_t\in\mathbb R^{d_{\text{in}}}$ to $y_t\in\mathbb R^{d_{\text{out}}}$:
$$
k_t,v_t,q_t,\beta_t=W_{\text{slow}}x_t,\qquad
\bar v_t=W_{t-1}\phi(k_t),
$$
$$
W_t=W_{t-1}+\sigma(\beta_t)\big(v_t-\bar v_t\big)\otimes\phi(k_t),\qquad
y_t=W_t\phi(q_t),
$$
where $\phi$ is a positive, sum-to-one map (softmax) applied on both write and read for stability, and
$W_{\text{slow}}$ is fixed (gradient-descent trained). Here the programmer $W_{\text{slow}}$ is *separate*
from the programmed $W_t$.

**The self-referential weight matrix.** Now let a *single* matrix $W_{t-1}$ play both roles. Given input
$x_t$, it emits the output **and** its own modification signals:
$$
y_t,k_t,q_t,\beta_t=W_{t-1}\phi(x_t),
$$
$$
\bar v_t=W_{t-1}\phi(k_t),\qquad
v_t=W_{t-1}\phi(q_t),
$$
$$
W_t=W_{t-1}+\sigma(\beta_t)\big(v_t-\bar v_t\big)\otimes\phi(k_t).
$$
The terminology (from Schmidhuber's original SRWM papers): $k_t$ is the **modifier-key** — the key whose
stored value must change; $q_t$ is the **analyser-query** — fed back into $W_{t-1}$ to retrieve the new
value $v_t$ to associate with the modifier-key. So the matrix *interrogates itself* ($v_t=W_{t-1}\phi(q_t)$)
to decide what to write, then delta-corrects itself. The output block $W^y_{t-1}$ has $d_{\text{out}}$ rows;
the key/query/beta blocks have $d_{\text{in}}$ (and 1) rows, so
$W_t\in\mathbb R^{(d_{\text{out}}+2 d_{\text{in}}+1)\times d_{\text{in}}}$, and the value vectors
$v_t,\bar v_t$ inherit that same row dimension so the outer-product update is shape-consistent.

**Only $W_0$ is trained.** This is the defining property: "the initial values of the SRWM $W_0$ are the
only parameters in this layer trained by gradient descent." Everything downstream — how it updates, how
fast ($\beta_t$), what it writes — is *emergent* from $W_0$ acting on the input stream. In practice they
expand the output to generate **four separate learning rates** $\beta_t\in\mathbb R^4$, one for each
sub-matrix $[W^y,W^q,W^k,W^\beta]$, and use multi-head computation.

**Why "self-referential" ⊃ "fast-weight."** Because the SRWM adapts *the way it adapts itself* per task
(the update-generating weights are themselves modified), it can specialize its learning algorithm to each
task in a multi-task stream — something the plain DeltaNet (which uses one fixed $W_{\text{slow}}$ for all
tasks) structurally cannot. This is the mechanistic reason the SRWM beats DeltaNet in the sequential
multi-task experiment but merely ties it in single-task few-shot.

**Trainability.** Despite the recursion, training uses ordinary truncated backprop-through-time — the
additive structure of the delta update ($W_t=W_{t-1}+\Delta_t$) keeps gradients tractable over a 50-step
span (the same span used for the RL agents). No second-order / bilevel optimization (contrast MAML).

**Experiments, mechanism-first.**
- *Few-shot (Table 1).* Synchronous-label episode (image+label fed together for the first $NK$ tokens,
  query fed without label). SRWM 97.4% Omniglot 1-shot, 47.0/61.4% Mini-ImageNet 1/5-shot — ≈ DeltaNet;
  the HyperTransformer (paper #5) is stronger (53.8/67.1) but is few-shot-specialized, whereas SRWM is a
  generic sequence model usable in RL.
- *Sequential multi-task (Fig 4, Table 2).* Delayed-label setting (label of input $t$ arrives at step
  $t{+}1$); concatenate an Omniglot segment then a Mini-ImageNet segment. Accuracy drops at the switch
  (step ~74) then recovers as the SRWM re-programs itself. SRWM > DeltaNet on the harder second task,
  demonstrating runtime adaptation. Large batch size was the critical hyperparameter (else the model
  collapses to a zero-shot heuristic).
- *Multi-task RL on ProcGen (Fig 5, §4.3).* Jointly train 6 easy-distribution games (Bigfish, Fruitbot,
  Maze, Leaper, Plunder, Starpilot), IMPALA/Torchbeast, 48 actors, 15-layer conv stem, memory module =
  SRWM (2 layers, hidden 128) vs LSTM / DeltaNet / feed-forward / **Fake SR** (self-modification removed —
  only the $y$ output kept). The shared $W_0$ is common to all tasks/episodes; the *effective* weight
  matrix is a function of the episode-specific input stream (Fig 5 caption). SRWM states reset only at
  episode boundaries. A memory-distribution 4-game partial-observability experiment (App C.1) confirms the
  effect.

**Project relevance.** This is the shard's headline for RL: a self-modifying weight memory as the adaptive
core of an actor-critic in a procedurally-generated environment with clean train/test level splits — the
methodological template one would copy to test "does an agent whose internal weights re-tune themselves to
context outperform a fixed-recurrent baseline" in the grid world. The **Fake-SR ablation is the exact
control design** the project should emulate (modulation-on vs modulation-off, same parameter budget) when
claiming a self-modification / modulation benefit.

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** WM = program (Schmidhuber 1990); frozen after training; environments keep evolving
  → want autonomous self-updating programs. Revisit the '90s SRWM using modern FWP/delta-rule/linear-
  Transformer tools. Evaluate in (a) few-shot SL, (b) sequential multi-task few-shot, (c) multi-task RL
  (ProcGen).
- **§2 Background on FWPs.** DeltaNet (Eqs.1–4): $W_{\text{slow}}$ emits $k,v,q,\beta$; delta update with
  read-back $\bar v_t=W_{t-1}\phi(k_t)$; $\phi$ = softmax for stability; multi-head; used as self-attention
  replacement in a Transformer with FF/layernorm/residuals. Slow programmer separate from fast programmed
  net.
- **§3 A Modern SRWM.** Self-training via self-invented key/value patterns + learning rates. SRWM
  $W_{t-1}\in\mathbb R^{(d_{\text{out}}+2d_{\text{in}}+1)\times d_{\text{in}}}$ produces
  $[y_t,q_t,k_t,\beta_t]$ (Eqs.5–8). Modifier-key $k_t$, analyser-query $q_t$. Only $W_0$ trained by GD.
  Extend to "3D+4" for four $\beta_t$; multi-head; App A full spec. Can replace any WM; main model =
  DeltaNet with Eqs.1–4 replaced by Eqs.5–8; App C.1 variant replaces only DeltaNet's slow matrix by SRWM.
- **§4 Experiments.**
  - **§4.1 Standard few-shot.** N-way K-shot episodes, sequential (Santoro/Hochreiter) approach;
    synchronous-label setting (Mishra/SNAIL, Fig 2). Table 1: SRWM ≈ DeltaNet; competitive with SNAIL,
    MAML, fwCNN-Hebb; HyperTransformer strongest. Shared Conv-4-32 / Conv-64 backends.
  - **§4.2 Sequential multi-task adaptation.** Delayed-label setting (Fig 3). Concatenate
    Omniglot+Mini-ImageNet segments, alternate order, randomized lengths. Harder to train; large batch
    size critical. Fig 4: recovery after task switch. Table 2: SRWM > DeltaNet (53.3 vs 50.4% total on
    Mini-ImageNet-second).
  - **§4.3 Multi-task RL (ProcGen).** 6 easy games jointly (Fig 5), IMPALA/Torchbeast, 48 actors,
    15-layer conv, memory module swap; baselines LSTM/DeltaNet/FF/Fake-SR; 200 train levels, 3×200 test
    splits; 300M steps; BPTT span 50; states reset at episode boundary. App C.1: 4 memory-distribution
    games (partial observability).
- **§5/§6 Related work & discussion.** Self-modifying nets lineage (Schmidhuber '92/'93; Finn MAML;
  Hochreiter learning-to-learn); recursive self-improvement framing.

---

## 34. von Oswald et al. 2020 — Continual Learning with Hypernetworks

**PDF:** `docs/project/references/in_context_learning/sources/von Oswald et al. 2020 - Continual learning with hypernetworks.pdf`
· ICLR 2020 · arXiv 1906.00695

### Phase 1: Foundational Overview (Undergraduate-Level)

**The problem.** Train a network on task 1, then task 2, then task 3 — and it *forgets* task 1
("catastrophic forgetting"). The obvious fix, keeping all past data around and rehearsing it, violates
the spirit of online/continual learning. This paper's insight: **move the forgetting problem up one
level.** Don't try to protect the network's weights directly; instead learn a small **hypernetwork** — a
"weight factory" — that takes a short **task embedding** vector $e^{(t)}$ and outputs the full weight set
$\Theta_{\text{trgt}}$ of the task's network.

**Why that helps.** To remember a task you no longer need to store its data — you only need to remember
one low-dimensional embedding $e^{(t)}$ and make sure the hypernetwork still maps that embedding to the
same good weights it used to. So "don't forget task $t$" becomes a single, cheap constraint: *keep
$f_h(e^{(t)})$ fixed.* This is enforced with a simple regularizer that requires **no past data at all** —
only the stored embeddings.

**Key findings.** Task-conditioned hypernetworks reach state-of-the-art on standard continual-learning
benchmarks (permuted/split MNIST, split CIFAR-10/100) and — remarkably — retain performance on very long
task sequences (up to ~100 tasks) with essentially no degradation, even in a **compressive regime** where
the hypernetwork has *fewer* trainable weights than the target network it generates (thanks to
"chunking" — reusing a small hypernetwork to emit the target weights piece by piece). The learned
task-embedding space shows structure and supports forward transfer. The framework also protects a
generative-replay model (the same regularizer shields a VAE's hypernetwork).

**Initial takeaway.** This is the canonical "task-conditioned hypernetwork" paper and the most direct
[[Hypernetwork]]-topic bridge in the shard. Its central lesson for the project: **conditioning weight
generation on a compact task/context embedding turns "remember a behaviour" into "remember a vector,"**
and a single output-space regularizer prevents interference between contexts. If the project ever wants
one agent to hold several context-specific policies without them bleeding into each other, this is the
reference architecture and the reference regularizer.

### Phase 2: Graduate-Level Deep Dive

**Task-conditioned hypernetwork.** Instead of learning target weights $\Theta_{\text{trgt}}$ directly,
learn a metamodel $f_h(\cdot,\Theta_h)$ whose output *is* $\Theta_{\text{trgt}}$:
$$
\Theta_{\text{trgt}}^{(t)} = f_h\big(e^{(t)},\Theta_h\big),
$$
where $e^{(t)}$ is a learned, differentiable **task embedding** (an ordinary parameter vector optimized by
backprop alongside $\Theta_h$). The target network then computes $y=f_{\text{trgt}}(x,\Theta^{(t)}_{\text{trgt}})$.

**The forgetting problem, stated.** The ideal but forbidden rehearsal loss fixes past input–output maps:
$$
\mathcal{L}_{\text{output}}^{\text{data}}=\sum_{t=1}^{T-1}\sum_{i=1}^{|X^{(t)}|}
\big\|f(x^{(t,i)},\Theta^{*})-f(x^{(t,i)},\Theta)\big\|^2 ,
$$
which requires storing all past inputs $x^{(t,i)}$. The hypernetwork replaces this with a **weight-space**
constraint that needs *no data*.

**Two-step, data-free output regularizer (the core contribution).** When learning task $T$:
1. Compute a candidate task-loss step $\Delta\Theta_h$ (one Adam step) minimizing the current task loss
   $\mathcal{L}^{(T)}_{\text{task}}=\mathcal{L}_{\text{task}}(\Theta_h,e^{(T)},X^{(T)},Y^{(T)})$.
2. Take the actual step by minimizing the total loss
$$
\mathcal{L}_{\text{total}}=\mathcal{L}_{\text{task}}\big(\Theta_h,e^{(T)},X^{(T)},Y^{(T)}\big)
+\frac{\beta_{\text{output}}}{T-1}\sum_{t=1}^{T-1}
\big\|f_h(e^{(t)},\Theta_h^{*})-f_h\big(e^{(t)},\Theta_h+\Delta\Theta_h\big)\big\|^2 .
$$
Here $\Theta_h^{*}$ = hypernetwork parameters *before* learning task $T$ (a frozen snapshot), $\Delta\Theta_h$
is treated as a constant "lookahead," and $\beta_{\text{output}}$ trades plasticity vs stability. The
regularizer says: *for every past task embedding $e^{(t)}$, the weights the hypernetwork emits must not
drift from what they were.* Memory of the past enters **only** through the stored embeddings
$\{e^{(t)}\}_{t=1}^{T-1}$ — never through past data. (A stochastic version averages over a random subset
of past tasks for efficiency; a sensitivity scan on $\beta_{\text{output}}$ is in App D.)

**Model compression via chunking.** Emitting a full deep-net weight tensor in one shot is high-dimensional.
Instead invoke a *small* hypernetwork iteratively, once per **chunk** (e.g. one layer at a time). To break
the unwanted weight-sharing this induces, introduce a learned set of **chunk embeddings**
$C=\{c_i\}_{i=1}^{N_C}$; the full target weights are the concatenation
$$
\Theta_{\text{trgt}}=\big[f_h(e,c_1),\,f_h(e,c_2),\,\dots,\,f_h(e,c_{N_C})\big],
$$
iterating over chunk embeddings with the task embedding $e$ fixed. Chunk embeddings are ordinary learned
parameters shared across tasks. This yields the **compressive regime**: the number of trainable
hypernetwork parameters can be *smaller* than the target-network size, yet the model still solves the
task (App E states a universal-approximation result for chunked nets).

**Context-free inference (unknown task identity).** The hypernetwork needs a task embedding as input, but
sometimes the task ID is unknown at test time. Three strategies:
- **HNET+ENT** — pick the embedding $e^{(t)}$ giving the lowest predictive-entropy output for the input
  (in-distribution data → peaked output). No extra learning.
- **HNET+R** — hypernetwork parameterizes a *replay* generator (VAE); mix current data with synthetic
  past data, protecting the target classifier by soft targets.
- **HNET+TIR** — add an auxiliary task-inference classifier, itself protected by hypernetwork-regularized
  synthetic replay, that predicts task ID from the input.

**CL scenarios (van de Ven & Tolias taxonomy).** CL1: task ID given. CL2: task ID unknown but not needed
(shared head). CL3: task ID must be inferred (hardest).

**Project relevance.** Two takeaways. (1) *Architecturally*, "one shared weight-generator + per-context
embedding" is exactly the design pattern behind a context-modulated policy; the project's FiLM modulator
is the affine special case ($f_h$ emits scale/shift instead of full weights), and this paper is the full
generalization with the memory analysis to justify it. (2) *The output regularizer* is the mechanism that
lets a single generator hold *multiple* context-specific behaviours without interference — a directly
importable idea if the grid-world agent must support several noise/context regimes from one network.
Caveat: this is supervised continual learning, not RL; adapting the two-step regularizer to a non-
stationary RL objective is non-trivial and would be a `senior-developer` `issue_plan` if pursued.

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** CL desiderata: no forgetting / positive backward transfer / positive forward
  transfer. Thought experiment: rehearsal with self-generated targets = multi-task upper bound but stores
  data. Change of perspective: maintain *sets of parameters* $\{\Theta^{(t)}\}$ via a metamodel
  $f_h(e,\Theta_h)$ mapping task embedding → weights; memorize one point per task. Generic w.r.t. target
  architecture; also improves generative replay.
- **§2 Model.**
  - **§2.1 Task-conditioned hypernetworks.** Weight generators (Ha 2017, Schmidhuber 1992). Output
    regularizer (Eq.1 = data-dependent baseline; Eq.2 = data-free two-step version with lookahead
    $\Delta\Theta_h$). Learned task embeddings updated by task loss, saved after each task.
  - **§2.2 Chunked hypernetworks.** Iterative chunk generation; chunk embeddings $C$; compressive regime;
    App E approximation result.
  - **§2.3 Context-free inference.** HNET+ENT (entropy), HNET+R (replay), HNET+TIR (task-inference
    classifier). CL1/CL2/CL3.
- **§3 Results.** Permuted/split MNIST, split CIFAR-10/100 (ResNet-32 target). SOTA on standard
  benchmarks; near-zero degradation on long (~100-task) sequences; compressive regime works; task-embedding
  space structure + forward transfer; hypernetwork-protected replay improves generative-replay CL.
- **Appendices B–F.** Architectures, loss functions, $\beta_{\text{output}}$ sensitivity (D),
  approximation theorem (E), GAN replay variant (F).

---

## 35. Zhmoginov et al. 2022 — HyperTransformer

**PDF:** `docs/project/references/in_context_learning/sources/Zhmoginov et al. 2022 - HyperTransformer.pdf`
· ICML 2022 · arXiv 2201.04182

### Phase 1: Foundational Overview (Undergraduate-Level)

**The idea.** For few-shot learning you get a handful of labelled "support" images and must classify new
"query" images of the same (novel) classes. HyperTransformer (HT) uses a **Transformer as a weight
factory**: feed it the support images and labels as tokens, and in a *single forward pass* it outputs all
the weights of a small convolutional network (CNN) specialized to that task. The support set never touches
the CNN via gradient descent — the Transformer *reads* it and *emits* a task-specific CNN.

**Why this design.** It **decouples the complexity of the task space from the complexity of an individual
task.** The big, high-capacity Transformer holds all the cross-task knowledge; the generated CNN can be
tiny because it only has to solve one task. This is exactly the opposite trade-off from MAML, where the
adapted model must "fit" all the meta-knowledge into the *same* number of parameters as the model itself —
a bottleneck that bites hardest for small models.

**Key findings.** For *small* generated CNNs, HT substantially outperforms MAML++ and RFS. For larger
CNNs, generating only the **final logits layer** on top of a conventionally-learned embedding already
matches state of the art — generating all layers helps only below a model-size threshold. HT is naturally
permutation-invariant (self-attention over an unordered support set), handles unbalanced/variable-size
support sets, and extends to **semi-supervised** few-shot learning by adding an "unlabeled" token — and,
tellingly, exploiting unlabeled data requires **≥2 Transformer layers**, matching a theoretical argument
that two layers can encode a nearest-neighbor label-propagation algorithm.

**Initial takeaway.** HT is the "Transformer-as-hypernetwork" archetype and the direct predecessor of
Chen & Wang's INR version (paper #6). Its most project-relevant analytical claim is that **a single
self-attention layer, with the right weights, computes one step of gradient descent on the logits-layer
cross-entropy loss** — connecting weight generation to the ICL-as-gradient-descent thread (§2 of the
corpus) and to FiLM-style conditioning.

### Phase 2: Graduate-Level Deep Dive

**Analytical framing — learning a solver.** A task $t$ has loss $L(f;t)$ and a **task description**
$\tau(t)$ (support set, or more generally any info). The weight generator is a solver $a_\phi$ that maps a
description to a model:
$$
f^{*}=a_\phi(\tau)\in\mathcal F,\qquad
\arg\min_{\phi\in\Phi}\;\mathbb E_{t\sim p(t)}\,L\big(a_\phi(\tau(t)),t\big).
$$
Few-shot learning is the special case where $\tau(t)$ = a support set of $k$ labelled samples per class in
an $n$-way task, and $L$ is the query-set loss. Because $\tau$ is *any* token set, HT covers supervised,
semi-supervised, multimodal descriptions uniformly.

**Layer-by-layer weight generation.** HT generates target CNN weights $\{\theta_\ell\}_{\ell=1}^{L}$
**autoregressively over layers**:
$$
\theta_1(\tau)\to\theta_2(\theta_1;\tau)\to\cdots\to\theta_L(\theta_{1..L-1};\tau).
$$
Each layer's generator is a Transformer that receives, per support sample $i$, a token built from three
pieces — a shared **image embedding** $s_{\phi_s}(x_i)$, an **activation embedding**
$h^\ell_{\phi_l}(z^\ell_i)$ of that layer's input activations $z^\ell_i=f_{\ell-1}(x_i;\theta_{1..\ell-1})$,
and the label $c_i$:
$$
\mathcal I_\ell=\Big\{\big(s_{\phi_s}(x_i),\,h^\ell_{\phi_l}(z^\ell_i),\,c_i\big)\Big\}_{i=1}^{kn}.
$$
The activation embedding makes each layer's weights depend on the inputs that layer actually receives (a
locality prior); the image embedding gives a global, weight-independent view shared across generators.

**Weight-slice placeholder tokens.** Alongside the sample tokens, the Transformer input is padded with
learnable **weight placeholder tokens**, each a $d$-dim vector tied to one slice of the to-be-generated
weight tensor. After self-attention over the full sequence, the outputs at the placeholder positions are
read out and assembled into the weight tensors. For a $k\times k\times n_{\text{in}}\times n_{\text{out}}$
conv kernel, either "output allocation" ($n_{\text{out}}$ tokens of size $k^2\times n_{\text{in}}$) or
"spatial allocation" ($k^2$ tokens of size $n_{\text{in}}\times n_{\text{out}}$) is used. Labels are
encoded as learned placeholder embeddings $\xi(c)$ (no semantic content — just class-slot markers); an
unlabeled token $\hat\xi$ handles the semi-supervised case.

**Self-attention ≈ one gradient step (the key derivation).** Consider generating the **logits layer** $W$.
Encode support samples as $I_k=(\xi(c_k),e_k)$ (label embedding + feature embedding) and weight slices as
placeholder tokens $(\mu^{(i)},0)$. If the query $Q_i$ produced by weight-slice token $i$ attends only to
keys $K_k$ of samples whose label $c_k$ matches $i$, and those samples' values are their embeddings $e_k$,
then self-attention *averages the embeddings of all samples with label $i$*:
$$
W_{i,\cdot}\ \sim\ \sum_{m=1}^{n} y_i^{(m)}\,e^{(m)} ,
$$
where $y^{(m)}$ is the one-hot label of sample $m$. This is precisely (i) cosine-similarity prototype
weighting **and** (ii) the result of a **single gradient-descent step** on the cross-entropy loss starting
from zero logits weights (App A). Hence a one-layer self-attention weight generator subsumes the classic
prototype/one-GD-step few-shot algorithm — and, with more layers/capacity, can learn strictly better
solvers. For **semi-supervised** learning a ≥2-layer attention first propagates labels from labelled to
similar unlabelled samples (queries/keys ∝ embeddings), then averages — a learned nearest-neighbor
algorithm, which is why the unlabeled-data benefit needs depth ≥2.

**ODE view (Appendix B).** For a one-to-one $t\mapsto\tau(t)$, the optimal $\theta(\tau)$ tracking a local
minimum of $L$ along a curve $\hat t(\gamma)$ in task space obeys
$$
\frac{d\theta}{d\gamma}=-\Big(\frac{\partial^2 L}{\partial\theta^2}\Big)^{-1}
\frac{\partial^2 L}{\partial\theta\,\partial t}\frac{d\hat t}{d\gamma},
$$
i.e. weight generation "tracks" the minimizer as the task varies — a continuous-generation ideal that HT
approximates by directly solving the empirical objective.

**Training.** Single loop: support set → HT generates CNN weights → cross-entropy on the query set →
backprop into all generator parameters $\phi$ (Transformer + image/activation feature extractors). No
nested/bilevel optimization, no unrolled inner loop — a stability/simplicity win over MAML.

**Results.** Small-model regime: HT beats MAML++/RFS on Omniglot & Mini-ImageNet across channel counts
(e.g. Omniglot 1-shot 8-channel 87.2 vs MAML++ 81.4). Large-model regime: generating only the logits
layer suffices. Semi-supervised: unlabeled data helps, but only with ≥2 layers.

**Project relevance.** HT is the concrete instantiation of "attention IS a weight generator." Its
self-attention ≈ gradient-step derivation is the mechanistic link the project should cite when arguing
that a modulation/generation module is doing *implicit optimization* over context. The **generate-only-
the-last-layer** finding is a practical, cheap design point: rather than a full hypernetwork, condition
just the readout — analogous to conditioning only a policy head. The layer-autoregressive generation and
activation-embedding locality prior are also relevant if the project ever generates more than a single
modulated layer.

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** Few-shot = extreme data scarcity; solver $a_\phi$ maps support set → model.
  Metric-based (universal embedding, capacity-limited) vs optimization-based (MAML, must fit meta-knowledge
  into model-sized $\theta_0$). HT: Transformer generates entire task-specific CNN in one pass; decouples
  task-space from task complexity; especially good for small CNNs. Semi-supervised via unlabeled token
  (needs ≥2 layers). Can generate all layers or just logits (threshold). End-to-end single-loop training.
- **§2 Related work.** Metric-based (Siamese/Matching/Proto/Relation/TADAM); optimization-based
  (MAML/Reptile/LEO); weight modulation & generation (CNAPs, LGM-Net, LEO, HyperNetworks); Transformers in
  vision & few-shot (FEAT embedding adaptation, ViT). HT generates a whole end-to-end model, single step,
  no SGD refinement (unlike LEO).
- **§3 Analytical framework.** §3.1 learning from generalized task descriptions (Eq.1). §3.2 few-shot as
  special case (support = $\tau$, query = loss); semi-supervised extension.
- **§4 HyperTransformer.** ODE tracking (Eq., App B). §4.1 few-shot model: layer-autoregressive
  generation; image + activation embeddings ($\mathcal I_\ell$); weight-slice placeholder tokens; label
  encodings $\xi(c)$ / unlabeled $\hat\xi$; output vs spatial kernel allocation; query-set cross-entropy
  training. §4.2 reasoning behind self-attention: supervised ≈ prototype averaging = one GD step (App A);
  semi-supervised = 2-layer label propagation + averaging.
- **§5 Experiments.** Omniglot/Mini-ImageNet across channel widths; small-model gains over MAML++/RFS;
  large-model logits-only sufficiency (Fig 3); §5.3 semi-supervised gains requiring ≥2 layers; generate-all
  vs generate-logits trade-off.
- **Appendix A/B.** Self-attention = GD-step derivation; ODE weight-tracking derivation.

---

## 36. Chen & Wang 2022 — Transformers as Meta-Learners for Implicit Neural Representations

**PDF:** `docs/project/references/in_context_learning/sources/Chen and Wang 2022 - Transformers as Meta-Learners for Implicit Neural Representations.pdf`
· ECCV 2022 · arXiv 2208.02801

### Phase 1: Foundational Overview (Undergraduate-Level)

**Background — what an INR is.** An Implicit Neural Representation (INR) stores a piece of data as a
*function*: instead of a grid of pixels, an image is a small neural net $f_\theta$ that maps a 2D
coordinate $(x,y)$ to an RGB value; a 3D scene is a NeRF mapping location+direction to colour+density.
INRs are resolution-free and compact, but *fitting* one to a given image normally means running gradient
descent from scratch — slow, and it generalizes badly from sparse observations.

**The idea.** Prior shortcuts learn a hypernetwork that squeezes each INR into a *single* latent vector —
but one vector is an information bottleneck that blurs fine detail. Gradient-based meta-learning (MetaSDF,
Learned Init) can infer the *whole* weight set but still needs test-time gradient steps. Chen & Wang
propose a **Transformer that outputs the entire INR weight set in one forward pass** — no single-vector
bottleneck, no gradient computation. Feed the image (as patch "data tokens") plus a set of learnable
"initialization tokens," and the Transformer maps the data tokens to output tokens that *are* the columns
of the INR's weight matrices.

**Key findings.** On 2D image regression (CelebA, Imagenette) and 3D view synthesis (ShapeNet NeRF), the
Transformer meta-learner *beats* the gradient-based Learned Init baseline — **without any test-time
optimization** (CelebA PSNR 31.96 vs 30.37; ShapeNet chairs 19.66 vs 18.85). More weight groups → sharper
detail (PSNR climbs monotonically from 25.6 at 1 group to 32.0 at 64). It also composes gracefully with
optional test-time refinement and with extra input views. An attention analysis shows generated weight
columns attend to *structured* image parts (nose/mouth in layer 1, forehead in layer 2, whole face in
layer 3) — the INR captures data structure without explicit supervision.

**Initial takeaway.** This is HyperTransformer's idea (paper #5) carried to a new target — generating a
data-representing neural function rather than a classifier. For the project it is the clearest statement of
the general principle: **a Transformer, viewed as a set-to-set map, is a hypernetwork whose residual
attention updates play the role of gradient-descent steps on the target weights.** The explicit "residual
attention ≈ gradient descent" formulation (below) is the tightest bridge in the shard between weight
generation and the ICL-as-optimization reading.

### Phase 2: Graduate-Level Deep Dive

**Problem setup.** Recover a signal $I:\mathcal X\to\mathcal Y$ from observations $O=\{T_i(I)\}$ (each
$T_i$ a measurement, e.g. "pixel $i$'s RGB" or "render ray $r_i$"). Represent $I$ by an INR $f_\theta$
whose weights are a set of matrices $\theta=\{W_i\in\mathbb R^{\text{in}_i\times\text{out}_i}\}_{i=0}^{m-1}$
(biases folded in). Fitting = minimize the $L_2$ loss
$$
L(\theta;O)=\frac{1}{|O|}\sum_{T_i\in O}\big\|T_i(f_\theta)-T_i(I)\big\|_2^2 .
$$

**Generalized gradient-based meta-learning.** MAML's inference is $n$ residual gradient steps from a
learned init $\theta_0$:
$$
\theta_{i+1}=\theta_i+\big(-\nabla_\theta L(\theta;O)\big|_{\theta=\theta_i}\big).
$$
Chen & Wang generalize the update to a *learned, step-specific* rule $U_{\psi_i}$ conditioned on data
$D_i$:
$$
\theta_{i+1}=\theta_i+U_{\psi_i}(\theta_i;D_i).
$$
The learnable components are now (i) an init $\theta_0$, (ii) a step count $n$, and (iii) the per-step
update rules $\{U_{\psi_i}\}$. Setting $U_{\psi_i}=-\nabla_\theta L$ recovers MAML; letting $U_{\psi_i}$
be an attention+feedforward block gives the Transformer.

**Transformer = this meta-learner (the key identification).** A Transformer's residual stream implements
exactly Eq. (3). Let $\varphi_i$ be the token states at layer $i$; then
$$
\varphi_{i+1}=\varphi_i+U_{\psi_i}(\varphi_i;d_i),
$$
where $d_i$ are the data tokens at layer $i$, $U_{\psi_i}$ is the (attention ⊕ feedforward) sublayer, the
learnable init $\varphi_0$ is the **initialization tokens**, and the token $\varphi_i$ corresponds to the
target weight $\theta_i$. The **residual connection plays the role of the $+(-\nabla_\theta)$ update in
gradient descent**, and each Transformer layer is one "meta-optimization step" whose update rule is
learned rather than fixed. (This is the same conceptual move as von Oswald et al. 2023 "Transformers learn
in-context by gradient descent," here applied to weight generation.)

**Concrete architecture (set-to-set weight decoding).**
1. **Data tokens.** Split the observation image into $P\times P$ patches (ViT-style), flatten, add a
   learnable positional embedding $e_i$, map by an FC layer: data token $=\mathrm{FC}(p_i+e_i)$. For NeRF
   view synthesis each pixel is extended to 9 channels (RGB + 3D ray origin + 3D ray direction) before
   patchifying, and multiple views simply contribute more tokens to the same set.
2. **Initialization tokens.** View each INR weight matrix $W_i$ as a set of **column vectors**; create one
   learnable init token per column (or per group — see below).
3. **Joint Transformer encoder.** Data + init tokens go in together; the encoder jointly (i) builds
   observation features via data–data attention, (ii) transfers observation knowledge to weights via
   data–init attention, and (iii) models weight–weight relations via init–init attention.
4. **Weight tokens out.** The output vectors at the init-token positions are the **weight tokens**; $m$
   independent per-layer FCs (FC$^{*}$) map them to the actual column vectors, assembling
   $\theta=\{W_i\}_{i=0}^{m-1}$.

**Training objective.** Compute $L(\theta;O)$ (Eq. 1) on the generating observations $O$; for tasks that
demand generalization (single-view synthesis) instead sample a *different* observation set $O'\ne O$ and
minimize $L(\theta;O')$, forcing the generated INR to generalize to unseen views. Purely feed-forward
backprop into the Transformer — no unrolled inner loop, no higher-order derivatives (the cost that made
MetaSDF/Learned Init "less flexible").

**Weight grouping (scalability knob).** Assigning a token per column is wasteful for large $\theta$. Group
$k$ columns and emit one token $u_{\lfloor i/k\rfloor}$ per group, then reconstitute each column as
$$
w_i=\mathrm{normalize}\big(u_{\lfloor i/k\rfloor}\cdot \bar w_i\big),
$$
where $\bar w_i$ are per-column **learnable** vectors *independent of the observation* (they keep columns
within a group distinct), and $\mathrm{normalize}$ is $L_2$ normalization. Group size $k$ trades precision
vs cost; the ablation (G = 1/4/16/64 → PSNR 25.6/27.9/29.9/32.0) shows the Transformer genuinely learns
inter-weight structure, not redundant copies.

**Results.** Image regression (Table 1): Ours > Learned Init on CelebA (31.96 vs 30.37) and Imagenette
(29.01 vs 27.07), *no test-time optimization*. View synthesis (Table 3): Ours beats Standard/Matched/
Shuffled/Learned Init on chairs/cars/lamps. Optional 100-step test-time refinement and extra views both
help further (Table 4). Attention-mask analysis (§5): weight columns attend to semantically structured
image regions → INRs exploit data structure implicitly.

**Project relevance.** Chen & Wang give the cleanest "**Transformer layer = one learned gradient step on
the target weights**" statement in the shard, generalizing MAML into an attention stack. Two importable
ideas: (1) the **residual-attention-as-optimizer** view formalizes why a conditioning module can perform
implicit adaptation without explicit gradients — the theoretical backbone for any claim that a
FiLM/hypernetwork modulator is "learning at inference"; (2) **weight grouping** is a practical recipe for
scaling weight generation (emit a few group vectors + fixed per-slot templates) that maps onto how one
might modulate a wide policy layer cheaply. Domain caveat: the target here is a data-representing INR, not
an RL policy, but the set-to-set weight-generation mechanism is domain-agnostic.

### Appendix: Section-by-Section Backbone

- **§1 Introduction.** INRs = continuous neural functions (image: coord→RGB; NeRF: loc+dir→density+RGB);
  resolution-free/compact but need per-instance GD (slow, poor sparse generalization). Prior:
  single-vector hypernetwork (bottleneck, blurry) or gradient-based meta-learning (whole weights but
  higher-order derivatives + fixed init + test-time GD). Proposal: Transformer as hypernetwork building the
  whole INR weight set in one pass; connect to gradient-based meta-learning; analyze generated INRs.
- **§2 Related work.** INRs (DeepSDF, OccNet, IM-NET, NeRF; SIREN sine activations; Fourier features).
  Hypernetworks for INRs (single-vector latent decode; hybrid feature-map/grid methods). Meta-learning
  (MAML/Reptile; MetaSDF, Learned Init — fixed init + test-time optimization). Transformers (NLP → ViT →
  meta-learning here).
- **§3 Method.** §3.1 problem formulation: signal $I$, INR $f_\theta$ with matrix set $\theta$, observation
  set $O=\{T_i(I)\}$, $L_2$ loss (Eq.1). §3.2 motivating from gradient-based meta-learning: MAML residual
  update (Eq.2) → generalized learned update $U_{\psi_i}$ (Eq.3) → Transformer residual instantiation
  (Eq.4), residual link ≈ subtracting gradients (Fig 2). §3.3 Transformers as meta-learners: data tokens
  (ViT patches) + init tokens (one per weight column) → Transformer encoder → weight tokens → layerwise
  FC$^{*}$ → $\theta$ (Fig 3); train with $L(\theta;O)$ or $L(\theta;O')$ for generalization. §3.4 weight
  grouping: $w_i=\mathrm{normalize}(u_{\lfloor i/k\rfloor}\cdot\bar w_i)$ (Fig 4).
- **§4 Experiments.** §4.1 image regression (CelebA/Imagenette; ViT-Base-like 6-layer encoder; 5-layer
  256-hidden MLP INR; 64 groups default): Table 1 beats Learned Init; Table 2 group ablation. §4.2 view
  synthesis (ShapeNet chairs/cars/lamps; 9-channel ray-augmented views; 6-layer NeRF): Table 3 beats
  baselines + Learned Init w/o test-time opt; adaptive foreground sampling for stability; Table 4 test-time
  opt + multi-view gains.
- **§5 Does the INR exploit data structure?** Visualize last-layer weight-token→data-token attention →
  weight columns attend to structured parts (nose/mouth, forehead, whole face) (Fig 7).
- **§6 Conclusion.** Transformer hypernetwork builds INRs in one pass, beats gradient-based meta-learning,
  composable with test-time optimization; generated INRs implicitly capture data structure.

---

## 37. Schug et al. 2025 — Attention as a Hypernetwork

**PDF:** `docs/project/references/in_context_learning/sources/Schug et al. 2025 - Attention as a Hypernetwork.pdf`
**Venue:** ICLR 2025. **Authors:** Simon Schug, Seijin Kobayashi, Yassir Akram, João Sacramento, Razvan Pascanu (ETH Zürich / Google DeepMind).

### Phase 1: Foundational Overview (Undergraduate-Level)

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

### Phase 2: Graduate-Level Deep Dive

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

## 38. Chen et al. 2024 — Exact Conversion of In-Context Learning to Model Weights

**PDF:** `docs/project/references/in_context_learning/sources/Chen et al. 2024 - Exact Conversion of In-Context Learning to Model Weights.pdf`
**Venue:** ICML 2024. **Authors:** Brian K Chen, Tianyang Hu, Hui Jin, Hwee Kuan Lee, Kenji Kawaguchi (NUS / A*STAR / Huawei Noah's Ark Lab).

### Phase 1: Foundational Overview (Undergraduate-Level)

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

### Phase 2: Graduate-Level Deep Dive

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
