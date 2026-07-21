> **Per-paper review — in-context-learning corpus, paper 1 of 15 (original batch).**
> Extracted from the [master review](../in_context_learning_lit_review.md) (§1); content is identical. Manifest: [[in_context_learning_sources]].

# 1. Elmoznino et al. 2025 — In-Context Learning and Occam's Razor

**Venue:** ICML 2025 (PMLR v267). **Group:** Mila – Quebec AI Institute / Université de Montréal (+ NVIDIA). Authors: Eric Elmoznino, Tom Marty, Tejas Kasetty, Leo Gagnon, Sarthak Mittal, Mahan Fathi, Dhanya Sridhar, Guillaume Lajoie.
**PDF:** `docs/project/references/in_context_learning/sources/Elmoznino et al. 2025 - In-context learning and Occam's razor.pdf`
**Code:** https://github.com/3rdCore/PrequentialCode

## Phase 1: Foundational Overview (Undergraduate-Level)

**Introduction — the core problem.** Machine learning wants models that *generalize* to data they were never trained on. A long-standing heuristic for this is **Occam's razor**: among all explanations that fit the observations, prefer the simplest one, because a simple rule cannot just memorize the data — it has to capture a real pattern. The trouble is that "simplicity" is mathematically slippery. Formal notions like Kolmogorov complexity (the length of the shortest program that reproduces the data) are *uncomputable*, so in practice people only minimize training error and then bolt on indirect simplicity nudges — weight decay, architecture choices, early stopping. This paper asks: is there a training objective that minimizes error *and* complexity **directly**? Its surprising answer: yes, and you are already using it whenever you train a Transformer with next-token prediction. That objective, applied to sequences of (data → prediction) pairs, secretly performs Occam's razor.

**Key findings — the main result.** The paper's central claim is an *equivalence*: the cumulative next-token-prediction loss that trains an in-context learner is **exactly** a data-compression scheme called **prequential coding** (a.k.a. prequential MDL). Under prequential coding you compress a dataset by predicting each new point using a model trained only on the points before it, and summing up the "surprise" (negative log-likelihood) of each point. The total bill — the area under the learning curve — is an upper bound on the data's Kolmogorov complexity. Because compression and simplicity are two sides of the same coin (Minimum Description Length principle), a sequence model trained to minimize this loss is being driven to infer, from the context, the *simplest* predictor that still fits — which is precisely what generalizes. The paper backs this with controlled experiments (linear regression, sinusoid regression, a Mastermind code-breaking task, and a synthetic HMM "language" task) showing that standard ICL ("prequential ICL") generalizes better than a control that only minimizes training error ("train-risk ICL"), especially in the **low-data regime**, and that the gap widens with task difficulty.

**Initial takeaway.** ICL works well not by magic but because next-token prediction is a disguised form of optimal compression, and optimal compression implements Occam's razor. This is a **normative** account — it says what ICL *should* do — and it also exposes ICL's failure modes: current in-context learners can **underfit** (they run one fixed-depth forward pass, unlike gradient descent which can iterate for weeks), and they fail to generalize to genuinely novel tasks (a large pretrained LLM, GPT-4, did *worse* than a tiny task-specialized model — and even worse than a trivial "predict the average label" baseline — on the Mastermind task). These are concrete, principled targets for improving ICL.

## Phase 2: Graduate-Level Deep Dive

The theoretical contribution is a four-step chain that ends in an exact equivalence between the ICL training objective and prequential code length. I reconstruct each link with its derivation.

### 2.1 Kolmogorov complexity, the 2-part code, and why NLL is compression

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

### 2.2 Two idealized learners: maximum-likelihood vs. Occam

A learning algorithm is a map $T : \mathcal{P}(D) \to \Theta$ from datasets to model parameters. Standard maximum-likelihood training is

$$
T^{\mathrm{ml}}(D) = \arg\min_{\theta'} \; -\sum_{d\in D} \log_2 p_{\theta'}(d),
$$

which optimizes error alone. The *Occam* learner adds the model-complexity term from the 2-part code:

$$
T^{\mathrm{oc}}(D) = \arg\min_{\theta'} \Big[\, K(p_{\theta'}) - \sum_{d\in D} \log_2 p_{\theta'}(d) \,\Big].
$$

$T^{\mathrm{oc}}$ is exactly what we want but **intractable**, because $K(p_{\theta'})$ cannot be computed. Regularizers (L2 norm, restricted model class, SGD's implicit bias) are all read here as *cheap surrogates* for the missing $K(p_{\theta'})$ term. The paper's move is to sidestep $K(p_\theta)$ entirely via prequential coding, then *meta-learn* a learner $T_\phi$ that approximates $T^{\mathrm{oc}}$.

### 2.3 Prequential coding — compression that never needs $K(p_\theta)$

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

### 2.4 Meta-learning over tasks controls $K(T_\phi)$, and ICL solves it exactly

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

### 2.5 The critical experimental contrast: prequential vs. train-risk ICL

The theory predicts that *which token you evaluate the loss on* is what makes ICL generalize. The paper isolates this with a clean control. Both learners have the identical architecture $\hat y_q = T_\phi((x,y)_{1:j}, x_q)$; they differ only in the query:

- **Prequential ICL** (standard): $x_q$ is a **novel** input not in the context → loss = generalization error → prequential code length.
- **Train-risk ICL** (control): $x_q$ is **resampled from the context** $x_{1:j}$ → loss = training error only.

To stop train-risk ICL from trivially attending-and-copying the matching context token, both use a **bottlenecked architecture** (from Mittal et al. 2025, §2 of this review): a Transformer compresses the context to a low-dimensional $z = \mathrm{Transformer}_\phi((x,y)_{1:j})$, then a separate MLP head predicts $\hat y_q = \mathrm{MLP}_\phi(x_q, z)$.

**Findings (Figures 2–3):**
- At **large context lengths** the two are identical — with abundant data, Eq. (1)'s $-\log_2 p(D)$ error term dominates the $K(p)$ complexity term, so overfitting is not a risk.
- At **short context lengths**, prequential ICL wins, and the gap **grows with task difficulty** (small for linear regression, larger for sinusoid, largest for Mastermind; confirmed by sweeping input dimensionality in Appendix D).
- Visualizing inferred polynomial coefficients (Chebyshev basis): train-risk ICL exploits high-degree components to overfit noisy data; prequential ICL fits simpler, lower-degree functions closer to ground truth.

### 2.6 Failure modes exposed by the theory

- **vs. SGD (Figure 2):** ICL generalizes better in low-data regimes but can be *beaten by SGD* in high-data regimes on hard tasks (Mastermind), because a Transformer's implicit fitted model has expressivity scaling with its activations ($\sim N$) and fixed compute (one forward pass), whereas an SGD-trained DNN scales with weights ($\sim N^2$) and unbounded compute. This is **underfitting**, and it is architecture-dependent — the choice of $T_\phi$ (bottlenecked Transformer vs. plain Transformer vs. Mamba-1/2) strongly changes which tasks get compressed well (Figure 4).
- **Task generalization (Figure 5):** GPT-4 on Mastermind has *higher* prequential code length than both a small task-specialized model and a naive marginal-class baseline — a large pretrained model can fail to minimize prequential code length on a genuinely novel task, even one a small MLP solves easily.

### 2.7 Kolmogorov vs. Bayesian framing (Appendix H) — why this matters for the corpus

The dominant rival account (Xie et al. 2022, Müller et al. 2022; Groups B of this corpus) is **ICL = implicit Bayesian inference**: meta-training instills a prior $p(p_\theta\mid \mathcal{D})$ over tasks, and in-context data yields a posterior $p_D(p_\theta\mid D) \propto p(D\mid p_\theta)\,p(p_\theta\mid\mathcal{D})$, with predictions via Bayesian model averaging. Elmoznino et al. argue the **Kolmogorov/compression view generalizes the Bayesian one**:

1. It needs **no well-defined task distribution** — $D$ need not be drawn from a prior for the meta-learning argument to hold.
2. Minimizing $\sum_i L_{\mathrm{preq}}(D_i;T_\phi)$ need not induce any explicit prior (it *can*, if that is the cheapest way to compress the meta-dataset).
3. The Bayesian prior $p(p_\theta)$ is only *implicitly* defined by architecture + objective, making it hard to analyze and prone to unfalsifiable "just-so" explanations; the compression view refers only to the explicit $T_\phi$ and the measurable $L_{\mathrm{preq}}$.
4. It **accommodates non-Bayesian empirics** — e.g. Raventós et al. (2024) found that with high pretraining-task diversity, ICL does *not* converge to the Bayes-optimal predictor under the pretraining prior but to a broader prior; the compression view explains this as $T_\phi$ having insufficient capacity to encode the true diverse prior, defaulting to a simpler broad-coverage approximation.

This appendix is the conceptual hinge linking paper 1 to the rest of the ICL corpus: it positions compression/MDL as the umbrella under which the Bayesian-inference lineage (Group B) sits as a special case.

## Appendix: Section-by-Section Backbone

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
