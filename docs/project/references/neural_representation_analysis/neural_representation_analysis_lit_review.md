---
title: "Neural Representation & Modulatory-Signal Analysis — Master Reference Review (3 papers)"
topic: neural_representation_analysis
status: in-progress
created: 2026-09-08
last_updated: 2026-09-08
papers_reviewed: 3
sources_held: 3
scope: |
  This folder collects work on **how to tell whether a neural agent's internal
  representations — and in particular a modulatory side-signal's output — are
  genuinely doing something**, using instruments finer than a performance
  comparison.

  The corpus is deliberately narrow and methods-driven. It was assembled to
  answer one question raised by this project's own results: after 32 training
  runs of a FiLM-style neuromodulator produced no measurable behavioural
  difference, is the modulator changing anything at the level of the network's
  internal representations? Papers enter this folder when they RUN a concrete,
  reimplementable analysis of internal activity or of a modulatory signal, and
  report what that analysis licensed. Papers that merely propose a modulation
  architecture and report learning curves are out of scope.

  Two of the three papers are neuromodulation architectures whose authors
  analysed their own modulatory signal; the third is a neuro-ethology-inspired
  analysis framework whose appendix carries the single most transferable
  instrument in the set.
related:
  - ../behavior_analysis/behavior_analysis_lit_review.md
  - ../modulation_in_rl/modulation_in_rl_lit_review.md
  - ../neuromodulatory_algorithms/
  - ../../critiques/nmn_input_site_grid_optimisation_dynamics.md
---

# Neural Representation & Modulatory-Signal Analysis — Master Reference Review

## 1. Plain-English entry point

This project built a small side-network — a "neuromodulator" — that watches the agent's
senses each step and, from what it sees, turns the main network's units up or down: it
emits a multiplier (a **gain**) and a shift (an **offset**) applied to the activity at up
to four places inside a reinforcement-learning agent. Thirty-two agents were trained with
various versions of this arrangement. A parallel behavioural analysis then found **no
difference between agents that had the modulator and agents that did not** — in
particular, no change in whether an injured agent seeks cover.

That result moves the question from behaviour to the inside of the network: **is the
modulator doing anything at all, even though nothing it does shows up in what the agent
does?** An in-house review sharpened the problem considerably. The numbers we currently
record for the modulator's output are averages taken over every unit, every timestep and
every parallel environment at once. A modulator that varies sharply from moment to moment
with the agent's bodily state, and a modulator that has quietly collapsed into a fixed
re-tuning of the network that never changes, **produce exactly the same recorded
numbers**. Nothing we log can tell them apart.

This document reviews three papers chosen to answer a single practical question: *what
analyses have other researchers actually run to establish that an internal signal is
genuinely context-dependent, and which of those could we run on the checkpoints already
sitting on disk?* The reading is deliberately for method rather than for results.

The three papers, in one line each:

- **Vecoven and colleagues (2020)** built essentially our architecture — a side-network
  that sets the slope and threshold of every unit in the main network — and then did three
  things with it that we have not: they plotted the modulatory signal against the hidden
  variable it was supposed to be tracking; they plotted each individual unit's gain against
  that same variable; and, most usefully, they **froze the modulator mid-episode, secretly
  changed the world, and showed the agent then confidently did the wrong thing**. That last
  experiment is a clean causal demonstration that the signal was carrying the adaptation,
  and it needs no retraining.
- **Ben-Iwhiwhu and colleagues (2022)** built a closely related architecture and asked
  whether its internal representations differ more across tasks than an unmodulated
  network's, using a standard similarity measure. Their most useful property for us is that
  this measure is **blind to a modulator that has merely rescaled everything by a constant**
  — exactly the degenerate case we cannot currently rule out. Their weaknesses are that all
  their evidence is coloured squares read by eye, with no summary number and no null
  comparison, and that they never froze or perturbed anything.
- **Simmons-Edler and colleagues (2025)** is the one to act on first. Buried in an appendix
  is an analysis in which they fit **one small statistical model per hidden unit** — 512 of
  them — predicting that unit's activity from where the agent was standing, and find that
  about 100 units are reliably tuned to location. The method does not care what the
  predictor is: swap "where the agent is standing" for "how hungry and how injured the agent
  is" and it becomes a direct test of whether our modulator's output is tuned to bodily
  state. Because it yields **one result per unit rather than one per training run**, it stays
  interpretable even though we have only a single training run per configuration — which
  is the constraint that makes most comparison methods useless to us. And the same paper
  supplies the strongest available argument for looking at all: it twice reports cases where
  two agents performed identically while doing something mechanistically different, and one
  case where a coarse population-level measure reported "nothing here" in a condition where
  the finer per-unit measure found structure.

**Verdict.** All three papers offer usable instruments, and none of them offers a
ready-made answer, because none of them ever faced our situation — a modulator that helps
neither behaviour nor performance and must be judged on its internals alone. Two papers are
silent on that case; the third gives principled grounds for saying a behavioural null is
precisely when per-unit analysis is warranted, without promising that anything will be
found. Section 5 is the operational summary: seven candidate analyses, what each costs,
what each would establish, and which two should run first.

## 2. How to read this document

- **Section 3** is the table of contents; **section 4** is the corpus manifest.
- **Section 5** is the payoff: a cross-paper inventory of every analysis method in the
  corpus, scored for transferability to a recurrent-PPO agent on a 10×10 grid with a
  27-dimensional observation, one seed per configuration, and a modulator emitting a
  per-timestep gain/offset pair.
- **Sections 6–8** are one per paper, in the order processed. Each carries **§x.2 Phase 1**
  (plain-language overview), **§x.3 Phase 2** (graduate-level, with the mathematics
  reconstructed where the paper omits it), **§x.4 Relevance to this project** (what
  transfers, what does not and why, what the paper would have concluded from a behavioural
  null, and compute cost), and **§x.5 the section-by-section backbone** preserving the
  paper's own ordering.
- Where a paper's analysis **would not apply here**, that is stated explicitly with the
  reason. Methods requiring many seeds, a differentiable world model, large-scale neural
  recordings, or a ground-truth experimenter-controlled task variable are marked out of
  reach rather than listed hopefully.
- **Nothing in this document plans code or experiments.** Several findings imply tooling
  changes — chiefly per-unit logging of activations and of the modulator's gain and offset
  at evaluation time. Those are named as such and handed to `senior-developer` for an
  `issue_plan`; they are not designed here.

## 3. Table of contents

| § | Content |
|---|---|
| [1](#1-plain-english-entry-point) | Plain-English entry point |
| [2](#2-how-to-read-this-document) | How to read this document |
| [3](#3-table-of-contents) | Table of contents |
| [4](#4-corpus-manifest) | Corpus manifest |
| [5](#5-method-inventory--what-the-corpus-actually-offers) | **Method inventory — what the corpus actually offers** |
| [6](#6-vecoven-ernst-wehenkel--drion-2020--introducing-neuromodulation-in-deep-neural-networks-to-learn-adaptive-behaviours) | **Vecoven et al. (2020)** — Introducing neuromodulation in deep neural networks |
| [7](#7-ben-iwhiwhu-dick-ketz-pilly--soltoggio-2022--context-meta-reinforcement-learning-via-neuromodulation) | **Ben-Iwhiwhu et al. (2022)** — Context meta-RL via neuromodulation |
| [8](#8-simmons-edler-badman-berg-chua-vastola-lunger-qian--rajan-2025--deep-rl-needs-deep-behavior-analysis) | **Simmons-Edler et al. (2025)** — Deep RL needs deep behavior analysis |
| [9](#9-cross-paper-synthesis) | Cross-paper synthesis — what the corpus does and does not license |

## 4. Corpus manifest

| # | Paper | Year | Venue | Version held | Pages | Role in this corpus |
|---|---|---|---|---|---|---|
| 1 | Vecoven, Ernst, Wehenkel & Drion — *Introducing neuromodulation in deep neural networks to learn adaptive behaviours* | 2020 | PLoS ONE 15(1):e0227922 | **Version of record**, open access | 13 | Nearest architectural precedent; supplies the **freeze-and-switch causal test** |
| 2 | Ben-Iwhiwhu, Dick, Ketz, Pilly & Soltoggio — *Context Meta-Reinforcement Learning via Neuromodulation* | 2022 | *Neural Networks* 152:70–79 | **arXiv preprint `2111.00134v3`** (journal version paywalled) | 44 | Supplies **CKA representational similarity**, and the only precedent for a *self-conditioned* modulator |
| 3 | Simmons-Edler, Badman, Berg, Chua, Vastola, Lunger, Qian & Rajan — *Deep RL Needs Deep Behavior Analysis* | 2025 | NeurIPS 2025 | arXiv `2506.06981v2` preprint | 48 | Supplies the **per-unit encoding GLM** — the most transferable method in the set — and the methodological case for looking past a behavioural null |

All three PDFs are held read-only under
`docs/project/references/neural_representation_analysis/sources/`.

**Version caveats.** Paper 2 is the **preprint**, not the version of record; its section,
figure and appendix numbering will not match the 10-page Elsevier article and is flagged as
such throughout §7. Paper 1's supplementary "S1 File" — which holds the exact network
architectures, the algorithm description and a proof — is **not bundled in the PDF held
here**, so layer widths and the neuromodulatory-signal dimension are unrecoverable from
this document. Paper 3 is a preprint of a paper accepted at NeurIPS 2025 and is **already
reviewed at length** for its behavioural content in
[`../behavior_analysis/behavior_analysis_lit_review.md`](../behavior_analysis/behavior_analysis_lit_review.md) §5;
§8 here covers only its neural and representational instruments and does not duplicate that
treatment.

## 5. Method inventory — what the corpus actually offers

Every analysis in the corpus that produces a testable statement about internal activity,
scored against our constraints. "Runs today" means: on saved checkpoints, with existing
trajectory stores, with no retraining and no environment change — subject only to logging
per-unit quantities at evaluation, which is a tooling change common to nearly all of them.

| # | Method | Source | What it takes in | What it puts out | Unit of analysis | Runs today? | Verdict for us |
|---|---|---|---|---|---|---|---|
| M1 | **Per-unit encoding GLM** on binned state | Simmons-Edler App. D | per-unit activity (or per-unit γ, β) + a design matrix of exclusive state-bin indicators | one fitted tuning curve and one held-out log-likelihood **per unit**; a count of state-tuned units | **the unit** (n = 128/site) | Yes, once per-unit values are logged | **Run first.** The only method whose output survives one seed per configuration |
| M2 | **Freeze-and-switch causal test** | Vecoven Fig 10 | a frozen policy; the ability to clamp γ, β and to alter context mid-episode | survival steps and behavioural measures under 5 clamp/switch conditions | the episode | Freeze: yes. Mid-episode context switch: needs a small intervention hook | **Run second.** The only *causal* instrument in the corpus; settles "functionally inert" without any baseline comparison |
| M3 | **Linear CKA, modulated vs. mean-frozen network** | Ben-Iwhiwhu §5.2.1 (adapted) | two activation matrices for **the same inputs** | one scalar in [0,1]; ≈1 means modulation changes nothing geometrically | the (run, site) pair | Needs paired samples — see §7.4 | Valuable *because* it is invariant to uniform rescaling, i.e. immune to the gauge ambiguity in our logged means |
| M4 | **Cross-context CKA** of activations, and of γ/β | Ben-Iwhiwhu §5.2.1–5.2.2 | activations binned by context | a C×C similarity matrix | the (run, site) pair | Same paired-sample caveat | Useful as a companion to M1; weak on its own without a shuffle null the source never computed |
| M5 | **Encoding scatter / R² of the modulatory signal against a named context variable** | Vecoven Figs 7A–9A (quantified) | γ, β over a rollout + candidate context channels | a per-unit R² distribution, plus a time-shuffled null | the unit | Yes | Cheap; complements M1. Their version was eyeballed — ours should not be |
| M6 | **Spectral / rank analysis of the gain matrix** | implied by Vecoven §2 and Ben-Iwhiwhu §4.1 | the mean-subtracted per-unit gain matrix over a rollout | an eigenvalue spectrum; the number of modulatory modes actually used | the (run, site) pair | Yes | One SVD. Both papers' architectures are rank-limited by design and ours is too (GRU state = 16 units), so the spectrum is directly interpretable |
| M7 | **Sign-flip and dormancy audit** | Vecoven §4 "Adaptation"; our own critique §3.1 | per-unit γ, β and pre-activations over a rollout | fraction of units whose gain crosses zero; per-unit active-probability distribution | the unit | Yes | One pass. Positions our modulator in the design space and tests the dormancy worry |
| M8 | **Linear population decoding** of state from hidden activity | Simmons-Edler §3.3, App. C | hidden states + a target variable, ridge, held-out split | RMSE vs. a chance baseline | the (run, site) pair | Yes | Include, but treat as the **coarse** instrument — the source paper's own results show it returning a null where M1 found structure |
| M9 | **Parameter-matched capacity control** | Ben-Iwhiwhu §5.3 | a widened / deepened unmodulated baseline | a performance comparison | the run | **No — requires new training** | Noted so it is not discovered late if a positive claim is ever made |
| M10 | Cross-task CKA against *task* dissimilarity | Ben-Iwhiwhu §5.2 | multiple distinct tasks | — | — | **No — we have one task family** | Out of reach; the axis does not exist in our design |
| M11 | Movement-state segmentation; insect-trajectory comparison | Simmons-Edler B.2, B.3 | long spatially extended trajectories | latent locomotor states | the trajectory | Not usefully | Out of reach on a 10×10 grid with four movement actions |

**Two prerequisites shared by almost every row**: (i) recording $\gamma_{it}, \beta_{it}$
and hidden activations **per unit per timestep at evaluation**, rather than the pooled means
we log during training; (ii) for M2 and the strong form of M3, the ability to feed the
modulator a **counterfactual context** while the main network sees the real observation.
Both are tooling changes, both are small, and both belong to `senior-developer`.


---

## 6. Vecoven, Ernst, Wehenkel & Drion (2020) — Introducing neuromodulation in deep neural networks to learn adaptive behaviours

### 6.1 Metadata and provenance

| Field | Value |
|---|---|
| **Full title** | Introducing neuromodulation in deep neural networks to learn adaptive behaviours |
| **Authors** | Nicolas Vecoven, Damien Ernst, Antoine Wehenkel, Guillaume Drion (Montefiore Institute, University of Liège) |
| **Venue** | PLoS ONE 15(1): e0227922, published 27 January 2020 |
| **DOI** | `10.1371/journal.pone.0227922` |
| **Version held** | **Version of record**, open access (CC-BY), 13 pp |
| **PDF** | `docs/project/references/neural_representation_analysis/sources/Vecoven et al. 2020 - Introducing neuromodulation in deep neural networks.pdf` |
| **Completeness caveat** | The paper repeatedly defers to an **"S1 File"** supplement for (i) the formal algorithm description, (ii) the exact per-benchmark network architectures, (iii) benchmark 2 and 3 environment details, and (iv) the proof of the optimal-Bayesian return on benchmark 1. That supplement is **not bundled in the PDF held here**. Consequently layer widths, the neuromodulatory-signal dimension `k`, learning rates and rollout lengths are **not recoverable from this document**, and no claim below depends on them. |
| **Code** | No repository URL appears in the version of record. |

### 6.2 Phase 1 — Foundational overview (undergraduate level)

**The problem.** A neural network trained by gradient descent has fixed weights once training stops. If the world then changes — the wind blows from a different direction, the reward moves to the other target — the network cannot retune itself, because retuning means changing weights and there is no gradient signal at test time. Animals do not have this problem: their nervous systems carry chemicals (dopamine, acetylcholine, noradrenaline and others) that **change how existing neurons behave** — how steeply a neuron responds to its input, and how easily it fires at all — on a timescale of seconds, without rewiring anything. This is *cellular neuromodulation*. The paper's proposal is to copy that mechanism into a deep network.

**The architecture.** Split the network in two. A **main network** does the actual job (map observation to action). A small **neuromodulatory network** watches a separate stream — here, the history of what the agent has done and what reward it got, *deliberately excluding the current observation* — and emits a single vector `z`, the artificial "neuromodulator". Every neuron in the main network then has its activation function reshaped by `z`: `z` sets that neuron's **slope** (how strongly it passes its input through) and its **offset** (where its threshold sits). Both are read out of `z` by a per-neuron linear projection. The authors call the whole thing an **NMN** — Neuro-Modulated Network.

This is, operationally, the same gain-and-offset operator this project uses: a per-unit multiplier and a per-unit additive shift applied to a pre-activation, generated by a side network from a conditioning signal. The differences are in *what the side network is allowed to see* and *how the readout is parameterised*, and both differences matter for us (§6.4).

**The test bed.** Three *meta*-reinforcement-learning benchmarks. "Meta" means: at the start of each episode the environment secretly draws a task parameter (call it α) from a distribution, and the agent has to figure out from experience which task it is in. Benchmark 1 is one-dimensional — the agent sees the target's position with an unknown constant bias α added, and must learn to subtract the bias it has inferred. Benchmark 2 is 2-D navigation where an unknown wind direction pushes every move off course. Benchmark 3 is 2-D navigation towards two targets, one worth +100 and the other −50, with which is which flipped by a hidden coin flip. In all three the only way to succeed is to *infer the hidden parameter from experience within the episode*, then act on the inference.

**Key findings.**

1. **Performance.** Against a standard recurrent network (RNN) baseline matched to within 2 % on parameter count and with identical recurrent depth, the NMN learns faster and converges higher on all three benchmarks, over 15 random seeds. Seed-to-seed variance is markedly smaller for the NMN. On benchmark 1 the NMN reaches an expected episode return of 4534 against a computed Bayes-optimal 4679 — near-optimal.
2. **The modulatory signal really does encode the hidden task.** Plotting each dimension of `z` against the ground-truth α across 1000 episodes shows `z` starting out uniform in α early in an episode (the agent does not yet know which task it is in) and becoming a smooth, systematic function of α as evidence accumulates. Two similar tasks give similar `z`. This is a *representational* result, established by directly regressing/scattering the modulatory signal against the known latent variable.
3. **The modulation is causally load-bearing.** Freezing `z` mid-episode and then secretly swapping which target is good makes the agent walk confidently into the bad target; unfreezing `z` lets it recover within the same episode. This is the clean causal demonstration that the modulatory pathway, not the main network's own recurrence, carries the adaptation.
4. **Some units switch off.** Per-neuron slope plots show that for certain values of α some neurons have slope ≈ 0, i.e. their output is a constant and they contribute nothing for that task — effectively a task-specific sub-network. Other neurons flip the *sign* of their slope, inverting their tuning.
5. **Robustness.** Swapping the saturating rectifier for a sigmoid degrades both architectures but degrades the RNN far more. Varying main-network depth (0, 1, 4 hidden layers) moves the RNN's performance a lot and the NMN's very little — the RNN matches the NMN only at zero hidden layers.

**Initial takeaway.** The paper is the earliest of the three reviewed here to do the thing this project now needs: it does not stop at "the modulated agent scores higher", it opens the modulator up and shows (a) that its output tracks the latent context, (b) that individual units' gains are organised by that context, and (c) that severing the signal breaks the behaviour it is supposed to support. The three analyses are cheap, need no extra training, and are directly reimplementable on saved checkpoints. Their limitation for us is that all three lean on an experimenter-known ground-truth context variable α, and on a task family where a behavioural difference already exists.

### 6.3 Phase 2 — Graduate-level deep dive

#### 6.3.1 The modulated activation function, and its identity with FiLM

Let $\sigma:\mathbb R\to\mathbb R$ be any scalar activation. The paper's neuromodulation-capable version, for a single main-network neuron with parameter vectors $w_s, w_b \in \mathbb R^k$ and neuromodulatory signal $z\in\mathbb R^k$, is

$$
\sigma_{\mathrm{NMN}}(x, z; w_s, w_b) \;=\; \sigma\!\big(z^{\top}(x\,w_s + w_b)\big).
$$

Expanding the inner product, since $x$ is a scalar and $x w_s + w_b \in \mathbb R^k$,

$$
z^{\top}(x\,w_s + w_b) \;=\; x\,\underbrace{\langle z, w_s\rangle}_{\textstyle \gamma(z)} \;+\; \underbrace{\langle z, w_b\rangle}_{\textstyle \beta(z)},
$$

so that

$$
\sigma_{\mathrm{NMN}}(x,z) \;=\; \sigma\big(\gamma(z)\,x + \beta(z)\big),
\qquad
\gamma(z) = \langle z, w_s\rangle,\quad \beta(z) = \langle z, w_b\rangle .
$$

**This is exactly FiLM applied to the pre-activation**, with the gain and offset produced by a *linear* readout of a shared latent $z$. Three structural facts follow immediately and all three bear on this project.

1. **The generator is affine-free.** There is no additive per-unit constant in either readout: $\gamma$ and $\beta$ are pure linear functionals of $z$. Hence $z = 0 \Rightarrow \gamma = \beta = 0$, and the neuron's output collapses to the constant $\sigma(0)$. There is no "identity at initialisation" property; the network is *born modulated*. This project's implementation is the opposite — gains initialise at 1 and offsets at 0, with learned per-unit baselines $g_i, g'_i$ added on top (see the optimisation-dynamics critique, `docs/project/critiques/nmn_input_site_grid_optimisation_dynamics.md` §1.1). The consequence is that in Vecoven's architecture the modulator *cannot* degenerate to "off" without killing the network, whereas in ours it can degenerate to "identity" silently.
2. **Rank.** Stacking $n$ main-network neurons, the vector of gains is $\gamma(z) = W_s z$ with $W_s \in \mathbb R^{n\times k}$. Because $k \ll n$ in practice, the achievable set of gain vectors is a $k$-dimensional linear subspace of $\mathbb R^{n}$: **the modulation across the whole main network has rank at most $k$.** The same holds for offsets. Any per-unit gain pattern the network can express is a linear combination of $k$ fixed "modulatory modes". This is a strong and testable structural prediction, and it is the natural object for a principal-components analysis of logged gains (§6.4).
3. **Parameter scaling.** Each neuron adds $2k$ parameters, so the modulatory overhead is $O(2kn)$ — linear in the number of *neurons*. A hypernetwork that generated weights instead would be $O(k \cdot \#\text{connections})$, i.e. linear in the number of *edges*. The paper states this explicitly as the argument for scalability against Ha et al.'s hypernetworks.

The signal itself is $z = f(c)$, with $c$ the "contextual and feedback" input and $f$ any differentiable map. **One $z$ is shared by every neuromodulated neuron in a network** — network-global signal, per-unit projection. The authors flag in the conclusion that a per-layer $z$ would be a natural extension and that they did **not** test it; that extension is what this project's multi-site grid implements, so the grid is exploring a direction this paper names and leaves open.

#### 6.3.2 The meta-RL formalism and what the modulator's input is

The meta-RL setting follows Wang et al. (2016) *Learning to reinforcement learn*. A distribution $\mathcal D$ over MDPs replaces a single MDP; at the start of episode $i$ an MDP $M \sim \mathcal D$ is drawn and the agent interacts with it for $T$ steps. Write the interaction history

$$
h_t \;=\; [x_0, a_0, r_0, x_1, \dots, a_{t-1}, r_{t-1}, x_t],
$$

and the objective as the expected discounted return over all timesteps *and episodes*. The RNN baseline feeds $h_t$ to both actor and critic. The NMN splits the stream:

$$
\underbrace{c_t = h_t \setminus x_t}_{\text{modulator input}}, \qquad \underbrace{x_t}_{\text{main-network input}} .
$$

That is, **the modulator sees the history with the current observation removed** — past states, past actions, past rewards — while the main network sees only the current observation. This is a deliberate architectural separation of "what is the situation right now" from "what kind of world am I in", and it is the single design decision that most distinguishes this paper from our own setup, where the modulator reads a *slice of the current observation* (see §6.4, self-conditioning).

Because $h_t$ grows with $t$, $f$ is instantiated as an RNN. Actor and critic are modulated **separately with no shared parameters**, on the explicit grounds that "the neuromodulatory signal might need to be different for the actor and the critic" — the same reasoning that motivates this project's per-site modulation grid.

The main network uses the **saturated ReLU**

$$
\sigma(x) \;=\; \min\big(1, \max(-1, x)\big),
$$

for every layer except the output layer, which is modulated but linear ($\sigma(x)=x$). Optimisation is A2C with generalised advantage estimation (Schulman et al. 2015) and proximal policy updates (Schulman et al. 2017).

#### 6.3.3 What a zero gain means under a saturating activation — the dormancy analogue

The paper's most-cited qualitative observation is that some neurons acquire "a scaling factor almost equal to 0, leading to a constant activation function", and that such neurons "can be pruned if the corresponding offset is added to [their] connected neurons". It is worth deriving what this actually claims, because the corresponding statement for **our** rectifier is different in an important way.

Under the saturated ReLU, a modulated neuron computes

$$
y \;=\; \operatorname{clip}\big(\gamma x + \beta,\; -1,\; +1\big).
$$

Define the neuron's **informative probability** under the empirical pre-activation distribution $p(x)$:

$$
q \;=\; \Pr_{x\sim p}\Big(\big|\gamma x + \beta\big| < 1\Big)
\;=\;
\Pr_{x\sim p}\!\left(\frac{-1-\beta}{\gamma} < x < \frac{1-\beta}{\gamma}\right) \ \ (\gamma>0).
$$

The width of that admitting interval is $2/|\gamma|$, centred at $-\beta/\gamma$. Two degenerate limits:

- $|\gamma| \to \infty$: the interval collapses, the unit becomes a **hard threshold** at $x = -\beta/\gamma$ — it transmits one bit.
- $|\gamma| \to 0$ with $|\beta| < 1$: the interval covers all of $\mathbb R$ but the *slope* vanishes, so $y \to \beta$, a **constant**. The unit transmits nothing. Downstream, a constant $\beta$ entering a linear layer with weights $u$ contributes $u\beta$ — a constant that folds into the next layer's bias, which is precisely the pruning argument.

Formally, if neuron $i$ has $\gamma_i \equiv 0$ and $\beta_i \equiv \beta_i^\star$ for **all** $z$ realised by the modulator, then for every downstream neuron $j$ with incoming weight $u_{ji}$ and bias $b_j$,

$$
\sum_{i'} u_{ji'} y_{i'} + b_j \;=\; \sum_{i' \neq i} u_{ji'} y_{i'} + \big(b_j + u_{ji}\beta_i^{\star}\big),
$$

so neuron $i$ is removable with a bias correction — exact, not approximate. If instead $\gamma_i(z) \approx 0$ only on a *subset* of contexts, the unit is context-gated and the network implements a **task-conditional sub-network**. The paper explicitly proposes measuring how disjoint those sub-networks become as tasks are made less similar, and leaves it as future work.

**The contrast with our rectifier.** For an unsaturated ReLU, $y = \max(0, \gamma x + \beta)$, and the analogous quantity is the *active* probability

$$
p_i \;=\; \Pr_{t}\big(\gamma_{it} x_{it} + \beta_{it} > 0\big),
$$

which is the diagnostic the optimisation-dynamics critique already proposes (§3.1 there). The important asymmetry: under a **saturating** activation, small $|\gamma|$ means "dead"; under a **rectifier**, small $|\gamma|$ with $\beta > 0$ means the unit becomes a near-constant *positive* output (also dead, in the dormant-unit sense of Sokar et al.), while small $|\gamma|$ with $\beta<0$ means the unit is off. So Vecoven's zero-gain observation is a genuine precedent for asking the dormancy question of a FiLM-modulated network, but the *threshold test* has to be restated for our nonlinearity rather than transplanted.

#### 6.3.4 The three representational analyses, stated so they can be reimplemented

This is the section of the paper this project is reading it for. All three analyses run **after training, on a frozen policy**, and cost only rollouts.

---

**Analysis A — Modulatory signal versus ground-truth context (Figs 7A, 8A, 9A).**

- *Input*: 1000 evaluation episodes from a trained agent. For each episode, the sampled task parameter $\alpha$ (known to the experimenter, unknown to the agent) and the modulator output $z_t \in \mathbb R^k$ at each timestep.
- *Procedure*: for a small set of timesteps $t \in \{t_1, \dots, t_m\}$ spanning early-to-late in the episode, produce a scatter of each coordinate $z_t^{(d)}$ against $\alpha$, one panel per timestep — i.e. a **time-resolved encoding plot of the modulatory signal against the latent variable**. For benchmark 2 the latent is an angle $\alpha[3]$, so the same plot is drawn in polar coordinates with each coordinate of $z$ on its own radius. For benchmark 3, $\alpha$ is five-dimensional, so the other four components are held fixed and $z$ is examined against the binary $\alpha[5]$ alone.
- *Output*: the shape of the conditional distribution $p(z_t \mid \alpha)$ as a function of $t$.
- *Conclusion licensed*: at small $t$, $p(z_t\mid\alpha)$ is essentially **uniform in $\alpha$** — the agent has no information, and (the authors argue) plays a near task-independent strategy. As $t$ grows the distribution becomes non-uniform and $z$ varies **continuously and monotonically** with $\alpha$, with the spread of $z$ values widening. At large $t$, $z$ is nearly constant *across timesteps*, from which the authors conclude the signal is "almost state-independent and serves only for adaptation". Continuity in $\alpha$ licenses the further claim that similar tasks receive similar modulation.
- *What it does not do*: no quantitative statistic is reported — no mutual information, no $R^2$, no decoding accuracy. The evidence is visual. Anyone reimplementing this should replace the eyeball with a number (§6.4).

---

**Analysis B — Per-unit gain versus ground-truth context (Figs 7B, 8B, 9B).**

- *Input*: the same rollouts; the per-neuron gain $\gamma_i(z_t) = \langle z_t, w_{s,i}\rangle$ for every neuron $i$ of one hidden layer of the main network.
- *Procedure*: plot $\gamma_i$ against $\alpha$, one trace per neuron, at a chosen timestep. In benchmark 3, where $\alpha[5]$ is binary and $k$ is larger, the authors **select the five neurons whose gain is most correlated with $\alpha[5]$** and show only those — a selection step that must be reported honestly, because choosing the top-correlated units and then showing that they correlate is not by itself evidence of population-level structure.
- *Output*: the per-unit tuning curve of the gain with respect to context.
- *Conclusions licensed*: (i) gains vary smoothly with $\alpha$, inheriting the continuity of $z$ — expected, since $\gamma_i$ is a linear function of $z$; (ii) some gains cross zero as $\alpha$ varies, **inverting the slope of the activation function**, which is a qualitatively different operation from scaling; (iii) some units have $\gamma_i \approx 0$ for particular $\alpha$, giving the context-conditional sub-network of §6.3.3.
- *Caveat inherited from the architecture*: because $\gamma = W_s z$, Analysis B contains no information not already in Analysis A up to a fixed linear map. It is a *readout* of A in units the network actually uses, not independent evidence.

---

**Analysis C — Freeze / unfreeze the modulatory signal with a mid-episode task switch (Fig 10, benchmark 3).**

This is the causal test, and the one this project should copy most directly. Within a single episode the authors run a five-phase protocol and plot the agent's trajectory (arrow per timestep) in each phase:

| Phase | Manipulation | Observed behaviour | Inference |
|---|---|---|---|
| (a) | $z$ **locked at its initial value** from the first timestep | The agent's "exploration" strategy — the policy it plays with no task information | Isolates the task-independent prior policy |
| (b) | $z$ **unlocked**, updated every timestep | Agent adapts and navigates to the correct (+100) target | Adaptation is possible |
| (c) | $z$ **re-locked** at its current, well-adapted value | Agent still navigates to the correct target, but with "slightly degraded" avoidance of the wrong one | The *adapted* value of $z$ is sufficient for the task-level behaviour; the residual degradation is attributed to $z$ additionally carrying state information |
| (d) | $z$ still locked; **the two targets' rewards are swapped** | Agent now reliably walks to the *wrong* target | The behaviour is driven by $z$, not by the main network re-inferring the task; the modulatory pathway is necessary |
| (e) | $z$ **unlocked** again | Agent re-adapts and returns to the correct target | The pathway is sufficient for recovery, within one episode |

The logic is a within-episode **necessity-and-sufficiency argument**: (c) shows a frozen adapted $z$ suffices; (d) shows that with $z$ frozen the agent cannot track a context change, so the adaptation is *carried by* $z$; (e) shows restoring the signal restores the behaviour. Crucially, phase (d) requires the ability to **change the context mid-episode while holding the agent fixed** — an intervention on the environment, not on the network.

The paper also asserts, without a supporting figure, that "freezing $z$ after adaptation will not strongly degrade the agent's performance", from which it concludes that the components of $z$ that do *not* depend on $\alpha$ are not critical.

---

#### 6.3.5 The comparison structure — what is contrasted with what

| Contrast | Detail | Controlled for |
|---|---|---|
| **NMN vs. RNN** | Same number of recurrent layers and units; parameter counts within 2 % | Capacity |
| **Across seeds** | 15 training runs per architecture per benchmark; mean ± std shown, running mean over 1000 episodes | Seed noise — and the paper's claim of *lower variance* for NMN is only meaningful because $n=15$ |
| **sReLU vs. sigmoid** | Both architectures, all three benchmarks (Fig 11) | Activation-function choice; NMN degrades less |
| **Depth 0 / 1 / 4 hidden layers** | Benchmark 1 only (Fig 12) | Architecture sensitivity; RNN matches NMN only at depth 0 |
| **NMN vs. Bayes-optimal** | Benchmark 1 only: 4534 vs 4679 expected return | An absolute ceiling, not just a relative baseline |

Note what is **absent**: there is no shuffled-modulator control (feeding $z$ from a different episode), no frozen-random-modulator control, no ablation that removes modulation from a subset of layers, and no statistical test anywhere. The comparison structure is "modulated versus unmodulated, matched capacity, many seeds", plus the within-episode freeze protocol.

#### 6.3.6 Where the paper is honest about limits

- Benchmark 3's $z$ is admitted to be **hard to interpret**: some dimensions constant in $\alpha[5]$, some correlated with it, some not converging at all — the last attributed to $z$ coding agent *state* rather than task. The authors do not attempt to separate these; they infer non-criticality of the non-converging dimensions from the freeze experiment rather than from any decomposition.
- The conclusion explicitly lists as open: per-layer rather than per-network modulatory signals; sharing activation parameters per layer (i.e. coarser granularity); non-linear parametric activations; whether continuity of $z$ in the task survives those changes; and "analysing the neuromodulatory signal to a greater depth ... with respect to different more complex tasks".
- The final paragraph concedes that "further research is certainly still needed to better characterise the NMN performances".

### 6.4 Relevance to this project

**What transfers, concretely.**

1. **Analysis A, upgraded to a number, is our missing contextual-engagement statistic.** Our critique already defines the contextual fraction $\rho_\gamma$ — the share of the total variance of the gain that is temporal rather than unit-to-unit. Vecoven's Analysis A is the complementary question: *of the temporal variance, how much is explained by a named context variable?* Concretely, for each site collect $\gamma_{it}$ over an evaluation rollout and fit a per-unit linear (or logistic, for binary context) encoding model against a candidate context vector $c_t$ — nociception level, hunger, injury state, distance-to-cover — and report the cross-validated $R^2$ distribution over units, plus the same quantity on **time-shuffled** context as a null. Vecoven never computed this; they eyeballed the scatter. The upgrade is cheap and it is what turns their figure into evidence.
2. **The rank-$k$ prediction gives a free structural diagnostic.** Their gains live in a $k$-dimensional subspace by construction. Ours are $\gamma_{it} = g_i + c_i + w_i^\top h^m_t$ with $h^m_t$ a 16-unit GRU state, so **our per-site gain matrix also has temporal rank at most 16**, on top of a static per-unit offset. That makes a PCA of the mean-subtracted gain matrix $[\gamma_{it} - \bar\gamma_i]$ immediately interpretable: the eigenvalue spectrum tells us how many modulatory modes the trained modulator actually uses out of the 16 available, and a spectrum with essentially all mass in a single component means the modulator has collapsed to a one-dimensional knob. This is a strictly stronger statement than $\rho_\gamma > 0$ and costs one SVD.
3. **Analysis C is the causal test we lack, and a version of it runs on saved checkpoints.** The freeze-at-current-value manipulation needs no retraining: replay the policy with $\gamma_t,\beta_t$ held at the value they took at some step $t_0$, and compare survival steps and the behavioural measures. Our critique's freeze-at-**mean** counterfactual is the special case $t_0 = $ "average", and by the gauge identity it is exactly equivalent to deleting the modulator. Vecoven's protocol adds the piece we do not have: **freeze, then change the context, and check the agent fails to track it.** Our environment can do this — an episode in which the agent's injury state is externally altered while the modulator is clamped is the direct analogue of swapping the good and bad targets.
4. **The per-unit gain-versus-context plot (Analysis B) is worth reproducing, with their selection step made explicit.** Reporting "the five most correlated units" is fine as an illustration provided the *population* statistic is reported alongside. Report the full distribution of per-unit $R^2$, then show exemplars.
5. **Sign-flipping gains are a phenomenon to look for.** Their Fig 7B finding — gains crossing zero and inverting a unit's tuning — is a qualitatively distinct mode of modulation from rescaling. Whether any of our units do this is answerable from logged $\gamma_{it}$ in one pass, and the answer is interesting either way: our reported gains have drifted to roughly 0.2–0.9 with no reported sign changes, which would place our modulator firmly in the "scale only" regime.

**What does not transfer, and why.**

- **The ground-truth context variable.** Every one of their analyses is anchored on $\alpha$, a scalar drawn by the experimenter and *guaranteed by construction* to be the only thing worth inferring. We have no such variable. Our nearest equivalents — body-state channels, injury, arm identity — are neither guaranteed to be the relevant latent nor independent of the observation the modulator reads. A null $R^2$ against our candidate context would therefore be ambiguous in a way theirs would not be: it could mean "the modulator is not context-dependent" or "we picked the wrong context". Mitigation is to pre-register several candidate contexts and report all of them, including the shuffled null.
- **The episode-level, converging signal.** Their $z$ *converges within an episode and then holds nearly constant* — they say so explicitly and treat it as the desired behaviour ("almost state-independent and serves only for adaptation"). Our modulator is supposed to do the opposite: track a body state that changes within the episode. So their headline signature of success (a $z$ that settles) would in our setting be the signature of the failure mode we are worried about. **Do not import their success criterion.** Import the method, invert the expected sign.
- **The self-conditioning difference.** Their modulator sees $h_t \setminus x_t$ — history *without* the current observation — while ours reads a slice of the current observation itself. Their architecture therefore cannot express "modulate on what I am seeing right now", only "modulate on what kind of world this is". A finding that their $z$ is context-dependent says nothing about whether a self-conditioned modulator will be; if anything it suggests the informational separation is doing work.
- **Meta-RL structure.** Their task distribution makes context inference *necessary for reward*, which guarantees pressure on the modulator. Our single-task-distribution gridworld provides no such guarantee, and this is one plausible mechanistic account of our behavioural null: with nothing in the environment that requires context-conditional policy switching, the modulator has no gradient pushing it towards context dependence. That account is testable — it predicts $\rho_\gamma$ near zero and a flat encoding $R^2$ — and it is a prediction the paper supports only by analogy, not by evidence.
- **Sigmoid / depth robustness studies** need retraining and are out of scope for a checkpoint-only analysis.

**What would they have concluded from a behavioural null?** The paper is **silent**. Every reported comparison has the NMN winning, so the question never arises, and no analysis is framed as diagnosing a modulator that fails to help. The nearest thing to grounds for an answer is the freeze/unfreeze protocol: it is a *within-agent* test that does not require the modulated agent to beat an unmodulated one — it asks only whether *this* agent's behaviour depends on *its own* modulatory signal. A null in phase (d) — frozen $z$, context switched, behaviour unchanged — would establish a functionally inert modulator without any reference to a baseline. That is an inference the method supports and the authors did not need to draw. Stating it as *their* conclusion would be an invention; stating it as a use of their method is fair.

**Compute and data requirements.** Trivial by our standards. All three analyses are post-training rollouts of a frozen policy: 1000 episodes on 1-D and 2-D toy environments. Applied to us: for each of 32 runs, one evaluation rollout batch with $\gamma_{it}, \beta_{it}$ recorded per site per step, plus the context channels. At 128-step windows, 128 units per site, four sites, and (say) 200 episodes, that is on the order of $10^7$ floats per run — small, and our trajectory stores already carry the observation side. The freeze counterfactual is a second rollout batch per condition with no gradient computation. Nothing here needs a GPU-hour that matters, and nothing needs retraining. The one requirement we do not currently meet is **logging $\gamma$ and $\beta$ per unit per timestep at evaluation**, rather than the pooled means we log during training; that is a tooling change, and it belongs to `senior-developer` as an `issue_plan`, not to this review.

**One-seed caveat.** Their variance claim rests on 15 seeds. We have one seed per configuration. Nothing in §6.4 items 1–5 requires multiple seeds — they are all *within-agent* representational statements about a given trained network — but any claim of the form "arm X modulates more contextually than arm Y" does, and cannot be supported by our grid as it stands.

### 6.5 Appendix: Section-by-Section Backbone

Preserving the paper's original section order.

**Abstract.** Animals adapt intentions, attention and actions to the environment; this rests on cellular neuromodulation, which dynamically controls intrinsic neuronal properties and stimulus responses in a context-dependent manner. The paper builds a deep architecture inspired by cellular neuromodulation and tests its adaptation on navigation benchmarks in a meta-RL context against state-of-the-art approaches.

**1. Introduction.** DNNs succeed but generalise and adapt poorly to unforeseen problems. In biology, adaptation is linked to neuromodulation, which acts alongside synaptic plasticity and continuously tunes neuron input/output behaviour in response to biochemical signals; it regulates properties unachievable by synaptic plasticity alone and is critical to adaptive control of continuous behaviours such as motor control. The paper introduces the **NMN** (Neuro-Modulated Network): a **main network** of neurons with parametric activation functions, plus a **neuromodulatory network** that controls those parameters. The two networks take **different inputs** — the neuromodulatory network processes feedback and contextual data, the main network processes everything else. Related work: differentiable plasticity (Miconi et al. 2018); Backpropamine, which learns a neuromodulatory signal gating *which* connections are plastic *when*; hypernetworks (Ha et al. 2016), where one network computes another's weights; and work on learned fixed activation functions.

**2. NMN architecture.** Mimics cellular neuromodulation (Drion et al. 2015) by tasking the neuromodulatory network with tuning the **slope and bias** of the main network's activation functions. Definition: $\sigma_{\mathrm{NMN}}(x,z;w_s,w_b) = \sigma(z^\top(x w_s + w_b))$ with $z\in\mathbb R^k$ the neuromodulatory signal and $w_s, w_b \in \mathbb R^k$ per-neuron parameters governing scale and offset. **All** main-network activations are replaced by neuromodulation-capable versions; **one** $z$, of free dimension $k$, is shared by all of them and computed as $z = f(c)$ where $c$ is a vector of contextual and feedback inputs. $f$ can be any DNN; if $c$ has dynamic size (more task information arriving over time) $f$ can be an RNN or a conditional neural process. Fig 1 gives the architecture and the activation-function computation graph. Parameter scaling: new parameters grow linearly with the **number of neurons**, versus linearly with the **number of connections** for a weight-generating hypernetwork — hence easier extension to very large networks.

**3. Experiments.**

*3.1 Setting.* Meta-RL (Wang et al. 2016) is the biologically motivated evaluation frame: an MDP subdivided as a distribution $\mathcal D$ over simpler MDPs. Notation $t$, $x_t$, $a_t$, $r_t$; at the start of episode $i$ an MDP $M$ is drawn from $\mathcal D$ and interacted with for $T$ steps; the agent's only information about $M$ comes from observed states and rewards. History $h_t = [x_0,a_0,r_0,x_1,\dots,a_{t-1},r_{t-1},x_t]$. Goal: maximise expected discounted reward over all timesteps and episodes.

*3.2 Training.* A2C: (1) interact and gather data, (2) update actor from critic ratings, (3) update critic towards a value function. Wang et al. model actor and critic as RNNs taking $h_t$; here both are modelled as NMNs, with modulator input $c = h_t \setminus x_t$ and main-network input $x_t$. Since $h_t$ grows, the modulator is an RNN. Fig 2 contrasts the two architectures. The main network is fully connected with **saturated ReLU** $\sigma(x)=\min(1,\max(-1,x))$, except the final (also modulated) layer with $\sigma(x)=x$; sigmoidal results are also reported in §4 and are "often appreciably inferior". RNN and NMN have the same number of recurrent layers/units and parameter counts within 2 %. Both trained by A2C with GAE and proximal policy updates. **No parameters shared between actor and critic**, motivated by the modulatory signal possibly needing to differ between them. Formal algorithm description and exact per-benchmark architectures are deferred to supplementary material.

*3.3 Benchmarks.* Three custom benchmarks with **continuous action spaces**: a toy problem and two sparse-reward navigation problems.
- *Benchmark 1* (1-D state and action). Task variable $\alpha \sim \mathcal U[-10,10]$. The agent observes a biased target position $x_t = p_t + \alpha$ where the true target $p_t \in [-5-\alpha, 5-\alpha]$. Action $a_t \in [-20,20]$; reward $10$ if $|a_t - p_t| < 1$, else $-|a_t - p_t|$. On success $p_{t+1}$ is resampled uniformly, else $p_{t+1} = p_t$ (Fig 3).
- *Benchmark 2* (2-D navigation, noisy movement). Three-dimensional $\alpha$: target at $(\alpha[1],\alpha[2])$; $\alpha[3]\sim\mathcal U[-\pi,\pi)$ is the main direction of a **wind cone** from which a perturbation $w_t$ is drawn uniformly each step. The agent observes its position relative to the target and outputs a move direction $m_t$; it is displaced by $m_t + w_t$; reward $-0.2$ per step, $+100$ on reaching the target, after which it is teleported to a uniform random position (Fig 4). (The figure caption states $-2$ where the text states $-0.2$; the discrepancy is in the paper.)
- *Benchmark 3* (2-D navigation, two targets). Five-dimensional $\alpha$: targets at $(\alpha[1],\alpha[2])$ and $(\alpha[3],\alpha[4])$; $\alpha[5]$ is a **Bernoulli** variable assigning $+100$ to one target and $-50$ to the other. The agent observes its relative position to both targets and moves along its action direction; on reaching a target it collects the reward and is teleported (Fig 5).

**4. Results.**

*Learning.* Fig 6: mean ± std of episode return over **15 seeds**, running mean over 1000 episodes, all three benchmarks. NMNs learn faster per episode and converge to better policies; NMN variance across seeds is much smaller than RNN's. Benchmark 1: the optimal Bayesian policy achieves expected return 4679 (proof in supplement); NMN reaches 4534 after 20 000 episodes — near-optimal.

*Adaptation.* Fig 7 (benchmark 1) shows, over 1000 episodes: **A** the temporal evolution of $z$ against $\alpha$; **B** per-neuron scale factors against $\alpha$ for one hidden layer; **C** per-timestep rewards. For small $t$, $z$ depends little on $\alpha$ (agent uncertain), scale factors are noisy, behaviour is poor and near task-independent. As $t$ grows, $z$ converges with clear $\alpha$-dependence and wider spread, and reward rises. At large $t$, $z$ holds constant across timesteps — "almost state-independent and serves only for adaptation". $z$ varies **continuously** with $\alpha$: similar tasks, similar signals. Per-neuron scale factors vary between negative and positive, **inverting the activation slope**; some neurons are **inactive** (scale ≈ 0, constant activation) for some $\alpha$.

Fig 8 (benchmark 2): $z$ appears to code **exclusively** for $\alpha[3]$, converging with respect to it regardless of $\alpha[1],\alpha[2]$ — plausibly because the target coordinates are not needed to compute an optimal move. Plots are projected into polar coordinates ($\alpha[3]$ is an angle; each $z$ dimension gets its own radius). Despite only noisy per-step information about $\alpha[3]$, $z$ quasi-converges slowly; scale factors vary smoothly and become highly asymmetric by timestep 100. Notably the agent performs well after ~40 steps **while $z$ has not yet converged** — good performance is possible while still gathering task information.

Fig 9 (benchmark 3): harder to interpret; $z$ codes for the task **and, in some sense, the agent's state**. With the two target positions fixed, $z$ is examined against $\alpha[5]$; adaptation is visible as very few negative rewards after 30 steps, and the agent starts navigating to the correct target after ~15 steps on average. At late timesteps $z$ *partially* converges: (i) some dimensions are **constant** in $\alpha[5]$, presumably coding $\alpha[1..4]$; (ii) some are **well correlated** with $\alpha[5]$ — panel B shows the five neurons whose scale factors are most correlated with $\alpha[5]$, with clearly different values for the two $\alpha[5]$ settings; (iii) the remainder **do not converge at all**, implying they code agent state, not task. The authors stress that **freezing $z$ after adaptation does not strongly degrade performance**, so the non-$\alpha$ features in $z$ are not critical.

Fig 10 — the freeze/unfreeze and task-switch protocol on benchmark 3, five panels: (a) $z$ locked at its initial value → the agent's task-independent "exploration" strategy; (b) $z$ unlocked → free adaptation; (c) $z$ locked at a post-adaptation value → still navigates to the correct target, but with slightly degraded avoidance of the wrong one, further suggesting $z$ codes state as well as task; (d) $z$ still locked, **targets switched** → the agent now always moves to the wrong target, since without updating $z$ there is no adaptation; (e) $z$ unlocked → adaptation recovers and it navigates correctly again.

*Robustness study — sigmoid activations.* Fig 11: sigmoids give worse or equivalent results to sReLUs for both architectures; the NMN is **more robust** to the change, the sReLU-vs-sigmoid gap being far smaller for NMN than RNN (especially benchmark 2).

*Robustness study — architecture impact.* Fig 12 (benchmark 1): main networks with 0, 1 and 4 hidden layers. RNNs **can** reach NMN performance at a particular architecture (no hidden layer) but are strongly architecture-dependent; NMNs are "surprisingly consistent" across depth.

**5. Conclusions.** A high-level import of cellular neuromodulation improves adaptive capability; the NMN beat classical RNNs on three meta-RL benchmarks. Four extension lines: (i) other adaptation-requiring problem classes, notably supervised **few-shot** meta-learning, where $c$ would be a few labelled samples and the NMN can be seen as a conditional neural process with a purpose-built conditioning connection; (ii) improving the NMN — non-linear parametric activations, spiking neurons, **sharing activation parameters per layer** (which in convolutional layers amounts to scaling filters), a **per-layer rather than per-network** neuromodulatory signal, and checking whether continuity of $z$ in the task survives; (iii) more benchmarks, and **analysing the neuromodulatory signal to a greater depth** on more complex tasks — including the observation that zero-scale neurons are exactly prunable (add their offset to downstream neurons), that units with zero scale for *all* tasks permit pruning the main network outright, and that units with zero scale for *some* tasks mean only a **sub-network** is used per task, raising the question of whether disjoint sub-networks emerge for less similar tasks; (iv) a concession that further work is needed to characterise NMN performance.

**Supporting information.** S1 File (not included in the PDF held here).

---

## 7. Ben-Iwhiwhu, Dick, Ketz, Pilly & Soltoggio (2022) — Context Meta-Reinforcement Learning via Neuromodulation

### 7.1 Metadata and provenance

| Field | Value |
|---|---|
| **Full title** | Context Meta-Reinforcement Learning via Neuromodulation |
| **Authors** | Eseoghene Ben-Iwhiwhu, Jeffery Dick (Loughborough University, UK); Nicholas A. Ketz, Praveen K. Pilly (HRL Laboratories, Malibu); Andrea Soltoggio (Loughborough) |
| **Published venue** | *Neural Networks* **152**: 70–79 (2022) |
| **Version held** | **arXiv preprint `2111.00134v3`, 26 April 2022, 44 pp**, self-described on the title page as "Preprint submitted to Neural Networks April 27, 2022". The Elsevier version of record is paywalled and is *not* the file reviewed here. |
| **Numbering caveat** | Section, figure, table and appendix numbering below is the **preprint's**. The journal article is 10 pages against the preprint's 44, so the published version almost certainly reorganises or relegates material — in particular the preprint's Appendices A–F (experimental configurations, CT-graph description, PyTorch listing, CARLA experiment, Meta-World discussion, and 8 further pages of analysis heat maps) may be supplementary or absent. **Do not cite preprint section numbers as journal section numbers.** |
| **PDF** | `docs/project/references/neural_representation_analysis/sources/Ben-Iwhiwhu et al. 2022 - Context meta-reinforcement learning via neuromodulation (preprint).pdf` |
| **Code** | `https://github.com/dlpbc/nm-metarl` — an extension of the original CAVIA (`lmzintgraf/cavia`) and PEARL (`katerakelly/oyster`) MIT-licensed implementations. A representative PyTorch layer is printed in full in Appendix C. |
| **Funding** | AFRL / DARPA contract FA8750-18-C0103 (the L2M lifelong-learning programme). |

### 7.2 Phase 1 — Foundational overview (undergraduate level)

**The problem.** Meta-reinforcement learning asks an agent to master not one task but a *family* of them, and to switch between family members after a handful of trials. Two established ways to do that: nudge the network's weights with a few gradient steps (CAVIA, built on MAML), or infer a "which task am I in" code and feed it to the network as extra input (PEARL). Both leave the policy network itself unchanged in structure, and the authors argue that is the bottleneck. Their reasoning: in an ordinary fully-connected network **every neuron has the same kind of job**, so producing a genuinely different policy means changing a great many weights at once — which a handful of gradient steps cannot do. When the tasks in the family are similar, that is fine. When they are genuinely different, the network cannot get far enough.

**The proposal.** Give the network a second population of neurons whose job is *not* to compute the answer but to **switch the first population on and off**. Each hidden layer is replaced by a "neuromodulated fully connected layer" holding two groups: standard neurons that produce the layer's output, and a smaller group of **modulatory** neurons that read the same layer input, compute their own activity, and project a multiplier onto each standard neuron. The multiplier passes through a `tanh`, so it can be positive or negative; the layer output then passes through a rectifier, so a negative multiplier turns a unit off outright. The result is that a small change in the modulatory pathway can reconfigure which units are active — a big change in the layer's function for a small change in parameters.

Note what is *not* here: **there is no additive offset**. This is gain-only modulation. The comparison to this project's gain-and-offset scheme has to be made with that in mind.

**How it is tested.** The modulated layer is dropped into two existing meta-RL algorithms — CAVIA and PEARL — and compared against the unmodulated standard layer, everything else held fixed. Environments run from easy to hard: 2-D point navigation; MuJoCo half-cheetah (run in a commanded direction; run at a commanded speed); Meta-World robot-arm manipulation (ML1, one task with parametric variation; ML45, forty-five distinct training tasks); and a configurable sparse-reward tree-navigation problem, the CT-graph, at three difficulty depths. Three seeds throughout.

**Key findings.**

1. **The benefit is conditional on task diversity.** On the easy environments (2-D navigation, both half-cheetah variants) the modulated and standard networks perform the same. On the hard ones (Meta-World ML1 and ML45, CT-graph at all three depths) the modulated network wins clearly, on both return and task-success-rate. The authors' framing: modulation helps exactly when the family's tasks require *dissimilar* representations.
2. **The advantage is not extra parameters.** A control experiment enlarges the standard network — once by widening every hidden layer, once by adding a layer — to match the modulated network's parameter count. Neither larger baseline closes the gap on CT-graph depth-4 or ML45.
3. **The representational claim.** Using a similarity measure called CKA (centred kernel alignment) the authors build task-by-task similarity heat maps of each hidden layer's activity, at five points: before any adaptation and after each of four adaptation gradient steps. On hard environments, the *modulated* network's layers become clearly dissimilar across tasks while the standard network's stay similar — the standard network fails to differentiate the tasks internally, and that is offered as the mechanism behind the performance gap.
4. **The modulatory pathway is where the diversity comes from.** Repeating the same CKA analysis on the modulatory activity itself, rather than on the layer output, shows dissimilar patterns across tasks — the authors' evidence that the modulators, not the standard neurons, generate the task-specific reconfiguration.
5. **An honest caveat buried in a footnote.** A memory-based meta-RL baseline (RL²) reached about 10 % success on ML45, "comparable to the results obtained using neuromodulatory gating". The authors also concede in the discussion that their gating mechanism is essentially the same idea as an LSTM/GRU gate applied to a feedforward net, and that no precise measure of "task similarity" or "task complexity" — the axis their whole argument is organised around — is actually defined anywhere in the paper or the wider meta-RL literature.

**Initial takeaway.** This is the paper in the corpus that most directly asks our question — *is the modulator changing anything inside the network?* — and answers it with a specific, standard, cheap tool (CKA on hidden activations, and separately on the modulatory signal itself). It also happens to be the paper whose architecture most resembles ours in one important respect the Vecoven paper does not share: **its modulator reads exactly the same input as the units it modulates**, which is the self-conditioning design we adopted and had no precedent for. Its weaknesses are equally clear and we should not copy them: the representational evidence is entirely qualitative heat maps with no summary statistic, no null distribution and no significance testing; the probe data over which similarity is computed is never stated; and the analysis is only ever deployed to *explain a performance gap that already exists*, never to adjudicate a null.

### 7.3 Phase 2 — Graduate-level deep dive

#### 7.3.1 The neuromodulated fully connected layer

Let $x$ be the layer input. The layer holds three weight matrices: $W^s$ (input → standard neurons), $W^g$ (input → modulatory neurons), $W^m$ (modulatory neurons → standard neurons). The forward pass is (paper's Eqs. 7–10):

$$
h^s = W^s x, \qquad
g = \mathrm{ReLU}(W^g x), \qquad
h^m = \tanh(W^m g), \qquad
h = \mathrm{ReLU}\big(h^s \otimes h^m\big),
$$

with $\otimes$ elementwise. A "strict" variant replaces the analogue gain by its sign (paper's Eq. 11):

$$
h = \mathrm{ReLU}\big(h^s \otimes \operatorname{sign}(h^m)\big),
$$

reported as better suited to discrete-control problems. The Appendix C listing pins down two details the equations leave open: biases are present on all three linear maps, and in the strict variant $\operatorname{sign}(0)$ is manually forced to $+1$. (The listing's `nn.Linear(nm_features, out_features, nm_features)` passes `nm_features` where PyTorch expects the boolean `bias` — harmless, since any non-zero integer is truthy, but it confirms the snippet is illustrative rather than the exact shipped code.)

**Relation to FiLM and to this project's operator.** Writing $\gamma = h^m$ and $\beta = 0$:

$$
h = \mathrm{ReLU}\big(\gamma \odot (W^s x) + \beta\big), \qquad \gamma = \tanh\!\big(W^m\,\mathrm{ReLU}(W^g x)\big) \in (-1,1)^{n}, \quad \beta \equiv 0 .
$$

So it is **FiLM with the offset deleted and the gain hard-bounded to $(-1,1)$**. Three consequences worth stating precisely.

*(i) The gain cannot exceed unity.* Because $\tanh$ saturates, $|\gamma_i| < 1$ always: the layer can only ever attenuate or sign-flip, never amplify. This is a design decision our implementation does not make — our gains are unbounded linear read-outs of a GRU state — and it is a strong regulariser. It also means the "gains fell to 0.2–0.9" observation in our own runs is, in this architecture's terms, entirely unremarkable: their gains live in that band by construction.

*(ii) The sign flip is the mechanism, and it is a hard gate.* With a rectified output,

$$
h_i = \mathrm{ReLU}(\gamma_i h^s_i) =
\begin{cases}
\gamma_i h^s_i & \text{if } \gamma_i h^s_i > 0,\\[2pt]
0 & \text{otherwise,}
\end{cases}
$$

so flipping the sign of $\gamma_i$ swaps *which half* of the standard neuron's input distribution survives the rectifier. A unit that fired on positive $h^s_i$ now fires on negative $h^s_i$ — its selectivity is inverted, not merely rescaled. This is the "dynamically turn on or off certain output in $h$" claim of §4.1, and the strict variant makes it the *only* operation available. Contrast with a positive-gain-only FiLM, which can attenuate a unit to silence but can never re-purpose it. **This project's modulator, with gains reported in $[0.2, 0.9]$ and no observed sign changes, is operating in the attenuation-only regime and has never exercised the mechanism the paper credits for its result.**

*(iii) Rank, again.* The modulatory bottleneck is small: Appendix Table A.1 gives neuromodulator sizes of 4, 8, 16 or 32 against hidden layer sizes of 100–600. Since $\gamma = \tanh(W^m g)$ with $g \in \mathbb R^{m}$ and $m \ll n$, the *pre-activation* of the gain vector lies in an $m$-dimensional subspace of $\mathbb R^{n}$ — the same rank-limitation argument as in Vecoven (§6.3.1), reached by a different route. The two papers independently converge on "a low-dimensional modulatory code broadcast through a per-unit linear projection", which makes a spectral analysis of logged gains a natural corpus-wide diagnostic (§9).

**Self-conditioning.** The modulatory input is $x$ — *the layer's own input*. There is no separate context stream, no privileged task label reaching the modulator, and (in CAVIA) the task context enters only by being concatenated to the network input, hence reaching standard and modulatory neurons alike. This is **full self-conditioning**, and it is the architectural fact most relevant to this project's design: it is the only reviewed precedent in which a modulator conditioned on the same information as the modulated stream produces a measurable benefit. Compare Vecoven, who deliberately withhold the current observation from the modulator (§6.3.2). The corpus therefore contains one paper on each side of that design choice, and both report success — which means the corpus does *not* adjudicate it.

Two further architectural facts from Appendix A: **the output layer is never modulated** (a plain linear layer in both CAVIA and PEARL variants), and in PEARL **only the actor is modulated**, never the critic or the Q/value/inference networks.

#### 7.3.2 The two meta-RL hosts, and why the analysis is done in CAVIA

*Problem setting.* Tasks $\mathcal T_i \sim p(\mathcal T)$, each an MDP $M_i = \{S, A, q, r, q_0\}$; objective over a finite horizon $H$ with discount $\gamma$:

$$
J(\pi) \;=\; \mathbb E_{q_0, q, \pi}\left[\sum_{t=0}^{H-1}\gamma^{t} r(s_t, a_t, s_{t+1})\right].
$$

*CAVIA* (Zintgraf et al. 2019) splits the policy parameters into network weights $\theta$ and **context parameters** $\phi$, the latter concatenated onto the network input and reset to $\phi_0 = 0$ at the start of each task. Inner loop (per task $i$, one gradient step during meta-training, four during meta-testing):

$$
\phi_i \;=\; \phi_0 - \alpha \nabla_{\phi} J_{\mathcal T_i}\big(\tau_i^{\mathrm{train}}, \pi_{\theta, \phi_0}\big).
$$

Outer loop, over a batch of $N$ tasks:

$$
\theta \;=\; \theta - \beta \nabla_{\theta}\,\frac{1}{N}\sum_{\tau_i \in \mathcal T} J_{\mathcal T_i}\big(\tau_i^{\mathrm{test}}, \pi_{\theta,\phi_i}\big).
$$

The outer loop uses TRPO; the inner loop uses vanilla policy gradient with GAE, with the unusually large inner learning rates of $0.5$ (2-D navigation, CT-graph) and $10.0$ (half-cheetah, Meta-World), and a linear feature baseline. Hessian-vector products for TRPO are computed by finite differences to avoid third-order derivatives.

*PEARL* (Rakelly et al. 2019) is off-policy soft-actor-critic with a probabilistic task embedding $z \sim q_\phi(z\mid c)$ inferred from collected context $c$, concatenated to actor and critic inputs. The three objectives are reproduced in the paper as

$$
\mathcal L_{\mathrm{actor}} = \mathbb E_{s\sim\mathcal B,\,a\sim\pi_\theta,\,z\sim q_\phi}\Big[D_{\mathrm{KL}}\Big(\pi_\theta(a\mid s,\bar z)\,\Big\|\,\tfrac{\exp Q_\theta(s,a,\bar z)}{Z_\theta(s)}\Big)\Big],
$$
$$
\mathcal L_{\mathrm{critic}} = \mathbb E_{(s,a,r,s')\sim\mathcal B,\,z\sim q_\phi}\big[Q_\theta(s,a,z) - \big(r + \bar V(s', \bar z)\big)\big]^2,
$$
$$
\mathcal L_{\mathrm{inference}} = \mathbb E_{\mathcal T}\Big[\mathbb E_{z \sim q_\phi(z\mid c^{\mathcal T})}\big[\mathcal L_{\mathrm{critic}} + \beta D_{\mathrm{KL}}\big(q_\phi(z\mid c^{\mathcal T})\,\|\,p(z)\big)\big]\Big],
$$

with $\bar V$ a target network, $\bar z$ indicating a stop-gradient, and $p(z)$ a unit Gaussian prior.

**Why CAVIA hosts the representational analysis.** The authors say so explicitly: CAVIA has a *single* neural component (the policy network), while PEARL has several (policy, Q, value, inference), so attributing a representational change is cleaner in CAVIA. They also note CAVIA covers both discrete and continuous environments while their PEARL experiments are continuous-only. This is a methodological point worth internalising: **representational analysis is easier to interpret in the architecture with fewer moving parts**, and our recurrent-PPO agent with a shared encoder, a GRU and separate actor/critic heads is closer to PEARL than to CAVIA in that respect.

#### 7.3.3 CKA, derived — the paper declines to, so this section does

The paper states that "describing the formulation of the CKA is outside the scope of this work" and refers the reader to Kornblith et al. (2019). Since reimplementation is the point of this review, here is the derivation.

Let $X \in \mathbb R^{n\times p_1}$ and $Y \in \mathbb R^{n\times p_2}$ be two activation matrices for **the same $n$ inputs in the same order** — $n$ probe examples, $p_1$ and $p_2$ units. Column-centre both. Define the centring matrix $H = I_n - \tfrac{1}{n}\mathbf 1\mathbf 1^{\top}$ and the linear Gram matrices $K = XX^{\top}$, $L = YY^{\top}$, both $n\times n$.

The empirical **Hilbert–Schmidt Independence Criterion** is

$$
\mathrm{HSIC}(K,L) \;=\; \frac{1}{(n-1)^2}\,\operatorname{tr}\!\big(K H L H\big),
$$

which measures dependence between the two representations by comparing their *pairwise-similarity structures*: $K_{ab}$ is how alike examples $a$ and $b$ look in representation $X$, $L_{ab}$ the same in $Y$, and HSIC is (up to centring) their inner product. HSIC is not scale-invariant, so CKA normalises it:

$$
\mathrm{CKA}(K,L) \;=\; \frac{\mathrm{HSIC}(K,L)}{\sqrt{\mathrm{HSIC}(K,K)\,\mathrm{HSIC}(L,L)}} \;\in\; [0,1].
$$

For **linear** CKA the whole thing collapses to a form that never builds an $n\times n$ matrix. With $X, Y$ already centred, $HXX^{\top}H = XX^{\top}$, so

$$
\operatorname{tr}(KHLH) = \operatorname{tr}\big(XX^{\top}YY^{\top}\big) = \operatorname{tr}\big((Y^{\top}X)^{\top}(Y^{\top}X)\big) = \big\|Y^{\top}X\big\|_F^{2},
$$

using cyclicity of the trace, and likewise $\operatorname{tr}(KHKH) = \|X^{\top}X\|_F^2$. The $1/(n-1)^2$ factors cancel in the ratio, giving

$$
\boxed{\;\mathrm{CKA}_{\mathrm{lin}}(X,Y) \;=\; \frac{\big\|Y^{\top}X\big\|_F^{2}}{\big\|X^{\top}X\big\|_F\;\big\|Y^{\top}Y\big\|_F}\;}
$$

which costs $O(n p_1 p_2)$ and a few Frobenius norms. Properties that matter here:

- **Invariant to orthogonal transformations** of either representation and to **isotropic scaling** ($X \to cX$). This is why the authors prefer it: a modulator that merely rescales a layer's activity by a single global factor will register CKA $=1$, i.e. *no representational change*. That is exactly the property we want, because it makes CKA blind to the gauge freedom the optimisation-dynamics critique identified (§2.2 there) — a uniform rescaling is invisible to it.
- **Not invariant to arbitrary invertible linear maps**, unlike CCA — which is why CKA gives sensible answers on wide layers where CCA saturates.
- **Requires paired samples.** $X$ and $Y$ must be activations *for the same inputs*. This is the binding constraint on transferring the method (§7.4).

**What the paper does not tell us.** The probe set is never specified. To compare "task 1's representation" with "task 2's representation" one must decide which $n$ inputs are pushed through both task-adapted networks, and the preprint does not say whether these are states drawn from task 1's trajectories, task 2's, a shared pool, or a random probe distribution. Since CKA's value depends materially on that choice, **this is a genuine reproducibility gap**, and anyone reimplementing must fix and report the probe set explicitly. The open-source repository is the place to resolve it.

#### 7.3.4 The two representational analyses, stated so they can be reimplemented

**Analysis A — Cross-task CKA of hidden-layer outputs (§5.2.1, Figs 6, 7, F.14–F.21).**

- *Input*: the trained CAVIA policy, adapted to each of $T$ meta-test tasks ($T = 6$ for 2-D navigation, $4$ for CT-graph depth-2, $8$ for CT-graph depth-3/4); the layer output $h$ of hidden layer 1 (and separately hidden layer 2), for both SPN and NPN.
- *Procedure*: for each **adaptation stage** $u \in \{\text{before update}, \text{grad }1, \dots, \text{grad }4\}$ and each pair of tasks $(i,j)$, compute $\mathrm{CKA}(h^{(i,u)}, h^{(j,u)})$ and render the resulting $T\times T$ matrix as a heat map. Five heat maps per layer per network — a *trajectory of representational differentiation across the inner-loop adaptation*.
- *Output*: a $T \times T$ similarity matrix per (layer, network, adaptation stage).
- *Conclusions licensed*: (i) before any gradient update, representations are similar across tasks in every setting — the shared base representation; (ii) after updates, dissimilarity emerges; (iii) on the **easy** environments both SPN and NPN differentiate (and on half-cheetah-direction the SPN differentiates *better*), matching their equal performance; (iv) on **CT-graph**, only the NPN differentiates, matching the performance gap; (v) on Meta-World, the NPN's advantage grows from ML1 to ML45.
- *What it does not do*: there is no scalar summary of a heat map, no error bar across the three seeds, no permutation null, and no test. Every claim is "the heat map looks non-uniform". A reader cannot tell whether NPN-vs-SPN differences exceed seed-to-seed variation.

**Analysis B — Cross-task CKA of the modulatory activity itself (§5.2.2, Figs 8, 9, 10, F.22 ff.).**

- *Input*: the same rollouts; the **modulatory activity $h^m$** (the gain vector) of each neuromodulated layer.
- *Procedure*: identical to Analysis A, substituting $h^m$ for $h$. Reported for 2-D navigation, CT-graph depth-2 and Meta-World ML45 in the main text, with CT-graph depth-3/4, both half-cheetah variants and ML1 in the appendix.
- *Conclusion licensed*: "non-uniformity in the heat map plots ... indicates that those layers encode diverse or dissimilar representations across task", from which "we can therefore conclude that the neuromodulatory activities, when projected onto a layer's standard neurons, produce the desired dissimilar representations across tasks."
- *Assessment*: this is the right question — **is the modulatory signal itself context-differentiated, separately from the layer it modulates?** — asked with an appropriate tool. But the inference is weaker than stated. Non-uniform cross-task CKA of $h^m$ shows the modulatory signal varies with task; it does not by itself show that this variation *causes* the layer-output dissimilarity in Analysis A, because $h$ and $h^m$ are not independent ($h = \mathrm{ReLU}(h^s \otimes h^m)$). A mediation-style argument, or a clamping intervention of the kind Vecoven run (§6.3.4, Analysis C), would be needed. **No clamping or freezing experiment appears anywhere in this paper.** That is the single largest methodological gap relative to what this project needs.

**Analysis C — Parameter-matched capacity controls (§5.3, Fig 11).**

Two enlarged standard baselines, "SPN larger width" (each hidden layer widened) and "SPN larger depth" (one extra hidden layer), sized to approximately match the NPN's parameter count, on CT-graph depth-4 and Meta-World ML45 under CAVIA. Neither recovers the NPN's performance. This is a clean and necessary control and it is the paper's strongest single result, because it rules out the most boring explanation of the effect.

#### 7.3.5 The comparison structure

| Contrast | Where | Controlled for |
|---|---|---|
| **NPN vs. SPN under CAVIA** | 8 environments (2-D nav, Ch-dir, Ch-vel, ML1, ML45, CT-graph d2/d3/d4) + CARLA | Meta-RL algorithm held fixed |
| **NPN vs. SPN under PEARL** | 5 continuous environments | Generality across meta-RL host |
| **NPN vs. parameter-matched SPN (wider / deeper)** | CT-graph d4, ML45 | Capacity |
| **Across environment complexity** | ordered easy → hard | The paper's central claim is a *interaction*: benefit grows with task dissimilarity |
| **Layer 1 vs. layer 2** | all CKA analyses repeated for both hidden layers | Depth of the effect |
| **Layer output $h$ vs. modulatory activity $h^m$** | Analyses A vs. B | Localises the source of representational diversity |
| **Adaptation stage (0–4 inner-loop steps)** | every heat map is a 5-panel row | The *dynamics* of differentiation, not just the endpoint |
| **(footnote) RL² memory baseline** | ML45 only | ~10 % success, "comparable" to NPN — an unflattering comparison the authors report anyway |

Absent: shuffled or randomised modulator controls; frozen-modulator ablations; any statistical test; more than three seeds; and any measure of task similarity, which the discussion concedes is missing from the field as a whole.

#### 7.3.6 Where the paper is honest about limits

- The discussion concedes the mechanism "is reminiscent of the gating in recurrent/memory networks (LSTMs and GRUs)", and that the strong results of memory-based meta-RL might be attributable to the same gating. The claimed advantage over recurrent variants is speed and parallelisability, not expressive power.
- "A precise measure of task complexity and similarity was not clearly outlined in this work and this is widely the case in meta-RL literature." The paper's organising axis is therefore informal.
- Meta-World numbers differ from the original benchmark paper because the reward functions were rewritten in the v2 environments; **no hyperparameter search was performed for either arm**, on the stated grounds that the comparison is like-for-like.
- The CARLA experiment (Appendix D) is explicitly labelled preliminary, with limited analysis, and excluded from the main paper.
- The conclusion restricts the claim: the framework "is most suited to problems that require rapid change in optimal representations across tasks, while its advantage is reduced when tasks can be solved using similar representations."

### 7.4 Relevance to this project

**What transfers, concretely.**

1. **Linear CKA between the modulated and the mean-frozen network is a single number for "does modulation change the representation at all", and it runs today.** This is the adaptation of Analysis A that best fits our constraints. Take one evaluation rollout; record hidden activations at a chosen site under (a) the trained agent as-is, and (b) the same agent with $\gamma_t,\beta_t$ clamped to their rollout time-averages $(\bar\gamma, \bar\beta)$ — which, by the gauge identity in our own critique, is functionally the modulator-deleted network. Both conditions see **the same observation sequence** only if the clamp is applied open-loop for a short horizon or the rollout is replayed off-policy; that pairing requirement is real and must be respected (see below). Then $\mathrm{CKA}_{\mathrm{lin}}(H_{\text{modulated}}, H_{\text{frozen}}) \approx 1$ says the contextual part of modulation does not change the representation's geometry, and CKA well below 1 says it does. Crucially, **CKA's isotropic-scaling invariance means this number is immune to the gauge freedom that makes our logged $\gamma$/$\beta$ means uninterpretable** — it cannot be fooled by a modulator that has folded itself into a constant rescaling. That property is the single most valuable thing this paper offers us.
2. **Cross-context CKA replaces cross-task CKA.** Their $T\times T$ task matrix becomes a $C \times C$ context matrix, with contexts defined by binned body state: uninjured / mildly injured / severely injured, high / low energy, and so on. Bin the timesteps of a rollout by context, sample a matched number from each bin, and compute pairwise CKA of the site's activations. The prediction under a working modulator is off-diagonal CKA noticeably below 1; under a collapsed modulator it is ≈ 1 everywhere. Run the identical analysis on a **baseline unmodulated run from the same grid** to get the contrast their Fig 7 provides, and on **time-shuffled context labels** to get the null they never computed.
3. **Analysis B — CKA on the modulatory activity itself — is directly our question.** Substituting our per-site $(\gamma_t, \beta_t)$ for their $h^m$ gives exactly "is the modulatory signal itself differentiated by context". It is the natural companion to the contextual-fraction statistic $\rho_\gamma$ our critique already proposes: $\rho_\gamma$ asks *how much* the signal varies over time, cross-context CKA asks *whether that variation is organised by context*. Both should be reported.
4. **The parameter-matched control is the design lesson.** Their strongest result is a control, not a headline. If any future comparison in this project claims the modulator helps, the widened/deepened unmodulated baseline is the comparison that will be asked for. Recording it here so it is not discovered late.
5. **The sign-flip observation reframes our own numbers.** Their mechanism *is* sign inversion under a rectifier; ours has never used it. Reporting the fraction of $(unit, timestep)$ pairs at which our $\gamma$ changes sign — almost certainly zero, given the reported $[0.2, 0.9]$ range — is a one-line result that positions our modulator in the attenuation-only corner of the design space, and gives a concrete, motivated intervention (allow or encourage negative gains) if a follow-up is wanted. That would be an `experiment-designer` / `senior-developer` job, not this review's.

**What does not transfer, and why.**

- **The paired-sample requirement of CKA is a real constraint, not a formality.** CKA compares two representations *of the same inputs*. Their setting supplies this trivially: the network changes (task-adapted parameters), the probe inputs can be held fixed. In ours the modulator is driven by the observation *stream*, so any intervention on the modulator changes the agent's actions and therefore the subsequent observations — the two conditions diverge and the samples stop being paired. Three workable fixes, in increasing cost: (i) compute CKA only over the *first* $k$ steps after the intervention, before trajectories diverge; (ii) replay a fixed recorded observation sequence through both network variants off-policy, which pairs perfectly but evaluates the network off its own state distribution; (iii) clamp the *modulator's input* to a counterfactual context while feeding the real observation to the main path, which pairs perfectly and stays on-policy for the main stream but requires a small code change to decouple the two input paths. Option (iii) is the cleanest and is the analogue of Vecoven's task-switch intervention; it is a code change and belongs in an `issue_plan`.
- **The whole "task diversity" axis.** Their central claim is an interaction between modulation and *how dissimilar the tasks are*. We have one task distribution, and the axis along which our grid varies is modulator input and injection site, not task dissimilarity. **Their result therefore predicts nothing about our grid except, uncomfortably, that a single-task-family setting is the regime in which they found modulation to give no benefit.** That is the closest thing in this corpus to a mechanistic account of our behavioural null, and it deserves to be stated in exactly those terms rather than softened.
- **Gain-only, bounded modulation.** Their $\gamma \in (-1,1)$ with $\beta \equiv 0$. Our offsets have drifted to $-1.8$ and are load-bearing. Results about *their* gain distribution do not transfer to ours, and in particular their architecture cannot exhibit the dormancy failure mode our critique worries about in the same way, because a saturating gain cannot push a unit's pre-activation arbitrarily negative — only an offset can.
- **Their gradient-adaptation axis has no counterpart in our setup.** Every heat map is a five-panel row indexed by *inner-loop gradient steps*. We have no inner loop; our modulator adapts within an episode by recurrence, not by gradient. The natural substitute axis for us is *time within an episode*, or *training checkpoint index* — and with 50 checkpoints per run we have a much richer version of the "how does differentiation develop" question than they do. Using checkpoint index as the panel axis, and asking when during training (if ever) contextual differentiation appears, is a genuinely better version of their figure that our data supports and theirs did not.
- **PEARL/CAVIA specifics** — inner/outer loops, TRPO, probabilistic task embeddings — are irrelevant to a recurrent-PPO agent and should not be imported.
- **Three seeds.** Their between-arm comparisons rest on three seeds and no test; ours would rest on one. Neither supports a confident between-arm claim, and we should not lean on theirs as precedent for making one.

**What would they have concluded from a behavioural null?** This paper gives more grounds than the other two, because **it actually contains behavioural nulls and ran the representational analysis on them anyway.** On 2-D navigation and both half-cheetah variants the modulated and standard networks performed the same; the corresponding CKA analyses show *both* networks differentiating tasks, and on half-cheetah-direction the *unmodulated* network differentiating better. Their reading, stated in the conclusion, is that the benefit "is reduced when tasks can be solved using similar representations" — i.e. **a behavioural null was interpreted as evidence that the task did not demand representational reconfiguration, not as evidence that the modulator was broken.** Two things follow for us, and they pull in opposite directions:

- *Supporting caution*: in their hands, behavioural null co-occurred with representational non-advantage across three environments. That is a weak, qualitative, $n=3$-seed correlational pattern, but it is a pattern, and it means a representational null on our side would not be surprising to these authors.
- *Supporting the search*: they never tested the converse — a case with no behavioural difference but a *large* representational difference — and their analysis is not designed to detect one. Nothing in the paper rules it out. The honest statement is that the paper's evidence is **consistent with** our behavioural null reflecting a task that does not demand context-conditional representations, and **silent** on whether an inert-looking modulator can nonetheless be representationally active.

**Compute and data requirements.** Their experiments needed Tesla K80 and RTX 2080 Ti GPUs and 500–1500 meta-training iterations per environment — but that is the *training* cost, which we do not incur. The **analysis** cost is negligible: linear CKA on an $n \times p$ activation matrix with $n$ a few thousand probe steps and $p = 128$ units is a couple of matrix products, milliseconds on CPU. Applied to our 32 runs × 50 checkpoints, a full checkpoint-by-checkpoint CKA sweep is on the order of $1600$ forward-pass rollouts plus $O(10^4)$ trivial matrix operations — hours of wall-clock at most, dominated entirely by rollout generation, and embarrassingly parallel across runs. The prerequisites we do not yet have are (a) recording hidden activations and per-unit $\gamma,\beta$ at evaluation, and (b) for option (iii) above, the ability to feed the modulator a counterfactual context. Both are tooling changes for `senior-developer`.

### 7.5 Appendix: Section-by-Section Backbone

Preserving the preprint's original section order.

**Abstract.** Meta-RL enables fast adaptation from few samples via dynamic representations in the policy network, obtained by reasoning about task context, parameter updates, or both. Obtaining rich dynamic representations beyond simple benchmarks is hard because of the burden on the policy network to accommodate different policies. The paper introduces **neuromodulation as a modular component** augmenting a standard policy network, regulating neuronal activity to produce efficient dynamic representations for task adaptation. Evaluated across discrete and continuous control environments of increasing complexity, applied to two state-of-the-art meta-RL algorithms (**CAVIA** and **PEARL**), it produces significantly better results and richer dynamic representations than the baselines.

**1. Introduction.** Meta-RL aims at rapid adaptation from few interactions. Dynamic representations can come from gradient-based (MAML), context-based (memory: RL²/Wang; recursive: SNAIL; probabilistic: PEARL), or hybrid (CAVIA) approaches; context approaches are "more interpretable as they disentangle task context from the policy network". **Stated limitation**: such approaches do not scale as complexity grows, because many diverse policies must be stored in one network; as tasks grow complex their similarity drops and the required representations become dissimilar. **Hypothesis**: standard policy networks are unlikely to produce diverse policies from a trained base representation *because all neurons have a homogeneous role or function*, so significant policy change requires widespread network change. **Speculation**: a network endowed with modulatory neurons has a significantly higher ability to modify its policy. Richness is **defined operationally as the dissimilarity of representations across tasks**. Existing neuromodulation designs gate plasticity, gate activations, or alter high-level behaviour; this work uses one principle — modulatory signals alter each layer's representation by **gating the weighted sum of input** of the standard component. Primary contribution: a neuromodulated policy network for meta-RL; modular, stackable with standard/convolutional layers. Results indicate an increasing advantage as complexity increases, and comparable performance on simpler problems. Code at `github.com/dlpbc/nm-metarl`.

**2. Related work.** *Meta-RL*: optimisation methods (MAML, Meta-SGD, Stadie, ProMP) learn good initial parameters adaptable in a few gradient steps; context-based methods aggregate few-shot experience into context variables, derived probabilistically, by recurrent memory, by recursive networks, or by combinations; hybrid methods obtain task-specific context parameters via gradient updates. *Neuromodulation*: a neuron altering the cellular activity or synaptic weights of other neurons; known biological neuromodulators DA, 5-HT, ACh, NA; Doya (2002) maps dopamine to reward-prediction error (TD error), serotonin to discount factor, acetylcholine to learning rate, noradrenaline to policy randomness. Prior applications to gradient-based RL, neuroevolutionary RL, goal-driven perception, and continual learning (Beaulieu et al. 2020, activation gating + meta-learning against catastrophic forgetting). Designs differ: plasticity gating, activation gating, direct action modification.

**3. Background.** *3.1 Problem formulation*: tasks $\mathcal T_i \sim p(\mathcal T)$, each an MDP $\{S,A,q,r,q_0\}$; objective Eq. (1), the expected discounted return over horizon $H$. *3.2 CAVIA*: a MAML extension, interpretable and less prone to meta-overfitting; policy $\pi_{\theta,\phi}$ with network parameters $\theta$ and **context parameters** $\phi$ **concatenated to the input**; inner loop updates only $\phi$ (Eq. 2), outer loop only $\theta$ (Eq. 3), over $N$ tasks per iteration with $\phi$ reset to $\phi_0$ per task. *3.3 PEARL*: off-policy, soft-actor-critic based; probabilistic task context $z$ with prior belief updated to a posterior $p(z\mid c)$ estimated by an inference network $q_\phi(z\mid c)$; $z$ concatenated to actor and critic inputs; objectives Eqs. (4)–(6) for actor, critic and inference, with target network $\bar V$, stop-gradient $\bar z$, unit-Gaussian prior, replay buffer $\mathcal B$, KL weight $\beta$.

**4. Neuromodulated network.** *4.1 Computational framework*: the neuromodulated policy network is a **stack of neuromodulated fully connected layers** (Fig 1a). Each layer holds **standard neurons** (the layer's output/representation, connected to the preceding layer by $W^s$) and **neuromodulators** (receiving input from the preceding layer via $W^g$, projecting onto standard neurons via $W^m$) (Fig 1b). The projected modulatory activity's function is the designer's choice — it could gate the plasticity of $W^s$, gate the activation of $h$, or otherwise; **this work uses an activity-gating neuromodulator** that multiplies target-neuron activity *before* the layer's non-linearity. Forward pass Eqs. (7)–(10): $h^s = W^s x$; $g = \mathrm{ReLU}(W^g x)$; $h^m = \tanh(W^m g)$; $h = \mathrm{ReLU}(h^s \otimes h^m)$. The **tanh** enables positive and negative modulatory signals, so the network can affect both the **magnitude and the sign** of target activations; with ReLU output, $h^m$ has "the intrinsic ability to dynamically turn on or off certain output in $h$". A simpler **strict-gating** variant (Eq. 11) uses $\operatorname{sign}(h^m)$ only, "shown to be suited for discrete control problems".

**5. Results and analysis.** Environments: 2-D navigation; half-cheetah direction and velocity (MuJoCo); Meta-World ML1 and ML45; and the discrete **CT-graph** (branch $b$, depth $d$). The setup investigates neuromodulation as a **complement** to meta-RL, not a competitor. SPN vs NPN under CAVIA across all environments; SPN vs NPN under PEARL in continuous environments only (SAC being continuous-control). CARLA results are in Appendix D.

*5.1 Performance.* Original CAVIA/PEARL setups followed; **for PEARL, neuromodulation was applied only to the actor**. Reported numbers are meta-testing results after meta-training; in CAVIA, meta-testing fine-tunes for **4 inner-loop gradient steps**. Metric is return or Meta-World success rate. **All results are over three seeds.**
- *5.1.1 2-D navigation* (Finn et al. 2017): navigate to a goal sampled in $[-0.5,0.5]$; reward is negative squared distance; observation is 2-D position; actions are velocity commands clipped to $[-0.1,0.1]$. Both networks perform well and comparably — expected, since the environment is simple and per-task representations are not very distinct (Figs 2a, 3a).
- *5.1.2 Half-cheetah*: direction (run forward or backward — only two unique tasks) and velocity (target speed sampled from $[0,2.0)$ or $[0,3.0)$). High-dimensional but "still simplistic"; optimal policies across tasks have similar representations. Comparable performance for SPN and NPN under both CAVIA and PEARL (Figs 2b/c, 3b/c).
- *5.1.3 Meta-World*: robotic-arm manipulation; **ML1** is one task with parametric variations, **ML45** has 45 meta-train and 5 meta-test tasks. NPN **outperforms** SPN in both, under both CAVIA and PEARL, on average return (Figs 2d/e, 3d/e) and on success rate (Fig 4). Experiments use the updated **v2** Meta-World reward functions.
- *5.1.4 CT-graph*: a sparse-reward discrete graph environment parameterised by branch $b$ and depth $d$; navigate from a start state to a randomly sampled end state; with increasing depth the action sequence grows linearly but the policy search space grows **exponentially**. Depths 2, 3, 4 used. **Significant** performance difference in favour of NPN at all three depths (Fig 5), attributed to the rich dynamic representations analysed in §5.2.

*5.2 Analysis.* CAVIA policies are analysed rather than PEARL's, because CAVIA has a single neural component (easier to analyse than PEARL's several) and covers both discrete and continuous environments. Representation similarity across tasks is measured with **centred kernel alignment (CKA)** (Kornblith et al. 2019), comparing per-layer representations of SPN and NPN across tasks. Alternatives noted and rejected: CCA (Morcos et al.), RSA (Kriegeskorte et al.), HSIC (Gretton et al.). CKA generates a similarity measure between two representations by comparing their similarity *structures*, each built from pairwise similarities between data points; it is a normalised extension of HSIC with isotropic-scaling invariance. **The formulation is explicitly declared out of scope**; CKA is chosen for robustness to random initialisation and for permitting comparison within and across networks, and because it has been used in meta-RL before (Raghu et al. 2020).
- *5.2.1 Representation similarities across tasks (layer outputs).* Heat maps per layer, one row per network, five columns: before update and after grad steps 1–4 (Figs 6, 7). Before any update, representations are similar across tasks. **2-D navigation**: both SPN (Fig 6a) and NPN (Fig 6b) show good cross-task dissimilarity — consistent with their comparable performance; the problem's simplicity makes task-distinct representations easy to obtain. **CT-graph depth-2**: the NPN's representations are **more dissimilar** across tasks than the SPN's (Figs 7a vs 7b). The argument: task-specific representations are distinct, so adaptation requires "a significant jump in the solution space"; the SPN struggles to make that jump, the NPN's dynamic alteration makes it possible. Second-hidden-layer plots in Appendix F.1.
- *5.2.2 Representation similarities of the neuromodulatory units.* "Now we ask ourselves where the representational diversity comes from. Is the neuromodulatory layer effectively contributing to rich representations...?" Same CKA procedure applied to the **neuromodulatory activities $h^m$** per layer across tasks, for 2-D navigation (Fig 8), CT-graph depth-2 (Fig 9) and ML45 (Fig 10), with more in Appendix F.4. Non-uniform heat maps, in contrast to Fig 7a, indicate those layers encode dissimilar representations across tasks; conclusion: the neuromodulatory activities, projected onto the standard neurons, produce the desired cross-task dissimilarity.

*5.3 Control experiments.* Because neuromodulators add parameters, the SPN was enlarged to approximately match the NPN's parameter count in two ways: **SPN larger width** (bigger hidden layers) and **SPN larger depth** (one additional hidden layer). Tested with CAVIA on CT-graph depth-4 and ML45 (Fig 11). **Increasing SPN size does not match NPN performance.**

**6. Discussions.** *Neuromodulation and gated recurrent networks*: the gating is "reminiscent of" LSTM/GRU gating, so memory-based meta-RL's success may itself be a gating effect; this work isolates a **simpler** gating on a feedforward MLP. Analogy drawn to decoupling attention from recurrence to build Transformers: decoupling gating from recurrence yields faster training and better parallelisation while keeping the benefit; memory approaches remain necessary for sequential data and POMDPs. Footnote: an **RL² run on ML45 reached ~10 % average meta-test success, comparable to the neuromodulated result**. *Task similarity measure and robust benchmarks*: complexity was operationalised as task dissimilarity, but **no precise measure of task complexity or similarity was defined**, and this is a field-wide gap; a good metric should capture change points in reward, state space, transition function or combinations, whereas most meta-RL benchmarks vary the reward function only; CT-graph, Meta-World and Alchemy are early steps.

**7. Conclusion and future work.** An architectural extension of standard meta-RL policy networks with neuromodulation, complementary to existing meta-RL frameworks. Significantly outperforms the standard policy network on complex problems, comparable on simpler ones. The projected modulatory activity could instead gate plasticity, or multiple neuromodulator types could share a layer; the extension could be tested with a **recurrent** meta-RL policy to enhance memory dynamics. Scope statement: "most suited to problems that require rapid change in optimal representations across tasks, while its advantage is reduced when tasks can be solved using similar representations."

**Acknowledgment.** AFRL/DARPA contract FA8750-18-C0103; thanks to Jeffrey Krichmar.

**Appendix A — Experimental configurations.** Tesla K80 and GeForce RTX 2080 GPUs. **The output layer of the neuromodulated policy network is a regular fully connected linear layer**, with only the preceding layers neuromodulated, in both CAVIA and PEARL. *A.1 CAVIA*: context variables concatenated to the policy input and reset to zero at the start of each task; one inner-loop gradient step during training, four at meta-test; ReLU throughout; Table A.1 gives per-environment iterations (500–1500), meta-batch size (20–45), trajectories per task (10–100), **context parameter count (5–100)**, 2 hidden layers everywhere, **hidden size 100–600**, and **neuromodulator size 4–32**. Outer loop TRPO (Table A.2: max KL $10^{-2}$, 10 conjugate-gradient iterations, damping $10^{-5}$, 15 line-search iterations, backtrack ratio 0.8); inner loop vanilla policy gradient with GAE, learning rate **0.5** (2-D nav, CT-graph) or **10.0** (half-cheetah, Meta-World); linear feature baseline in both loops; finite-difference Hessian-vector products; 4 sampling workers. *A.2 PEARL*: original configurations largely followed; learning rate 3e-4 for all components; KL penalty 0.1; **neuromodulation applied only to the policy (actor)**; Table A.3 gives 500–1000 iterations, 2–225 train tasks, context vector size 5–7, network size 300 (policy/Q/value), inference network 200, 3 hidden layers, neuromodulator size 4–32.

**Appendix B — CT-graph.** States: start, crash, wait, decision, end, and one end designated goal. Wait states require a wait (forward) action; decision states require a turn action; a mismatched action causes a **crash** with reward $-0.01$ and a return to start. Reaching the goal end state gives $+1.0$; otherwise reward is $0.0$ per step. Episodes terminate at a crash or at any end state. Observations are 1-D vectors with full state observability. Complexity is set by branch $b$, depth $d$, sequence length, reward function and observability level; depths 2, 3 and 4 used (Fig B.12).

**Appendix C — Implementation.** A PyTorch `NMLinear` module listing: linear maps `std`, `in_nm`, `out_nm`; activations `relu` for the modulatory hidden stage and `tanh` for the projection; an optional `gate='strict'` branch replacing the projection by its sign with $\operatorname{sign}(0)\mapsto 1$; output multiplied in place. Full implementation open-sourced; the codebase extends the original CAVIA and PEARL repositories.

**Appendix D — Additional experiments (CARLA).** Autonomous-driving environment via the MACAD OpenAI-gym wrapper; $64\times64\times3$ RGB observations, 9 discrete actions, clear-noon weather. Two tasks — *drive aggressively* and *drive passively* — defined by reward functions and sampled from a uniform task distribution; the tasks are similar but the domain is hard. Observations are compressed by a **VAE pre-trained on random-action samples and then frozen**; CAVIA context parameters are concatenated to the VAE latents. 300 iterations, 4 tasks per iteration, context size 10, 2 episodes per task before and after one inner-loop step. The NPN shows an advantage (Fig D.13b). Explicitly labelled preliminary.

**Appendix E — Discussion on Meta-World performance.** Differences from the original Meta-World paper's baselines are attributed to the **v2 reward-function rewrite**; the authors note results could improve with hyperparameter tuning for either arm, and state that **no hyperparameter search was performed**, since the goal is a like-for-like difference.

**Appendix F — Analysis plots.** *F.1* second hidden layer for 2-D navigation (Fig F.14 — both networks differentiate; both already partly differentiated before any update, underlining the environment's low complexity) and CT-graph depth-2 (Fig F.15 — only the NPN differentiates). *F.2* CT-graph depth-3 and depth-4, layers 1 and 2, both networks (Figs F.16, F.17) — the SPN still fails to differentiate, the NPN succeeds. *F.3* half-cheetah direction and velocity (Figs F.18, F.19) — both networks differentiate; **the SPN differentiates better on half-cheetah direction**, attributed to task simplicity — and Meta-World ML1 and ML45 (Figs F.20, F.21) — the NPN's advantage grows with complexity. *F.4* cross-task CKA of the **modulatory activities $h^m$** for CT-graph depths 3 and 4, both half-cheetah variants, and ML1; dissimilar representations observed throughout.

---

## 8. Simmons-Edler, Badman, Berg, Chua, Vastola, Lunger, Qian & Rajan (2025) — Deep RL Needs Deep Behavior Analysis

### 8.1 Metadata and provenance

| Field | Value |
|---|---|
| **Full title** | Deep RL Needs Deep Behavior Analysis: Exploring Implicit Planning by Model-Free Agents in Open-Ended Environments |
| **Authors** | Riley Simmons-Edler¹²*, Ryan P. Badman¹²*, Felix Baastad Berg³, Raymond Chua⁴, John J. Vastola¹², Joshua Lunger⁵, William Qian⁶², Kanaka Rajan¹²† (¹Neurobiology, Harvard Medical School; ²Kempner Institute, Harvard; ³NTNU; ⁴McGill & Mila; ⁵Toronto; ⁶Harvard Biophysics). *equal contribution; †corresponding |
| **Venue** | **39th Conference on Neural Information Processing Systems (NeurIPS 2025)** |
| **Version held** | arXiv `2506.06981v2`, 30 November 2025, 48 pp (main text 10 pp + appendices A–E) |
| **PDF** | `docs/project/references/neural_representation_analysis/sources/Simmons-Edler et al. 2025 - Deep RL needs deep behavior analysis.pdf` |
| **Code** | `https://github.com/RileySE/Craftax-Foraging/tree/foraging`. The authors state that **example scripts for the Appendix D neural-GLM analysis are in that repository** — the single most useful pointer in this review. |
| **Cross-reference** | The **same paper (same arXiv v2)** already carries a full 900-line review in this project's behaviour-analysis corpus at [`../behavior_analysis/behavior_analysis_lit_review.md`](../behavior_analysis/behavior_analysis_lit_review.md) §5, covering the environment, the behavioural motifs, path analysis, phase segmentation and the decision GLMs. **This section deliberately does not duplicate that treatment.** It is scoped to the paper's *neural / representational* instruments — the per-neuron encoding GLM of Appendix D, the ridge decoding of §3.3 and Appendix C, and the architecture contrasts those instruments are deployed across — because those, not the behavioural motifs, are what the modulator question needs. Read the two sections together. |

### 8.2 Phase 1 — Foundational overview (undergraduate level)

**The problem.** When we want to know whether a reinforcement-learning agent has learned something interesting, we almost always look at a reward curve. The authors' complaint is that a reward curve tells you *whether* the agent copes and nothing about *how* — and that neuroscience and ethology have spent a century building instruments for exactly that gap, which machine learning has barely borrowed. They set out to demonstrate the borrowing: build a naturalistic environment, train an ordinary agent in it, and then analyse that agent the way a neuroscientist analyses a mouse — filming its paths, fitting statistical models to its choices, and *recording from its units while it behaves*.

**The setup.** ForageWorld: a 96×96 procedurally generated arena built on the Craftax benchmark. Food animals wander from spawn points and deplete when eaten, so a patch must be left and revisited later; water sources are fixed; predators appear near food and chase the agent when it is in line of sight. The agent manages hunger, thirst, fatigue and health, and is rewarded simply for keeping each of those four above half its maximum — a homeostatic reward, not a task-completion one. The agent is deliberately plain: PPO with a single-layer 512-unit GRU, about 6.5 million parameters, no world model, no planner, no external memory store. Everything is logged every timestep — the full hidden state, all internal variables, positions, predator distances, the value output, the policy entropy.

**Key findings.**

1. **Behaviour.** The agent spontaneously explores in outward-spiralling, rotating loops that closely resemble a displaced desert ant's search pattern, then switches into efficient revisitation of remembered patches. Which patch it returns to is predicted, in a statistical choice model, by how recently it was visited, how many food animals were seen there, how many predator encounters happened there, and how uncertain the agent's own position estimate was there.
2. **Population-level representation.** A simple linear readout (ridge regression) of the 512-unit hidden state recovers where the agent *was* up to 50–100 steps ago and where it *will be* up to 50–100 steps ahead, in arena-centred coordinates. Body-centred coordinates cannot be decoded at all. Decoding improves over training, and a single decoder generalises across different arenas for the same agent.
3. **Single-unit representation — the part that matters most here.** In an appendix analysis, the authors fit **one statistical model per hidden unit**, 512 of them, predicting that unit's activity from the agent's current location (the arena coarsened to a 14×14 grid, one on/off predictor per cell). Roughly **100 of the 512 units are well fit by position alone** — they are, in the neuroscientist's sense, place-sensitive. Furthermore the *strength* of a unit's position response grows with distance from the arena's centre, which the authors read as a distance-accumulation signal pushing the agent to return home.
4. **The same single-unit analysis run across three different agent architectures** — the baseline; the same agent trained without its auxiliary "predict your own position" objective; and an entirely different algorithm (a parallelised Q-network with an LSTM) — finds the distance-ramp in all three. Because the pattern survives changing the objective and changing the algorithm, they argue it is a property of the *task solution* rather than of any one design.
5. **A tension the paper does not remark on.** The population decoder fails completely without the auxiliary objective — arena-centred position drops to chance (their Figure 17, "eliminating the structured spatial encoding"). Yet the single-unit encoding models in that same condition still find position-sensitive units and still show the distance ramp. In other words: **the coarser instrument reported "no representation" in a condition where the finer instrument found one.** That is, precisely, this project's situation.

**Initial takeaway.** The brief that commissioned this review originally treated this paper as the odd one out — behaviour rather than representation. That was wrong, and the correction matters. Appendix D contains the **most directly transferable method in the entire three-paper corpus**: a per-unit encoding model whose output is a *distribution over units* rather than a single number per run, which is exactly the property a project with one seed per configuration needs; whose regressors can be anything the experimenter logs, so internal body state substitutes for spatial position without altering a line of the method; and whose compute cost is negligible. Its weaknesses are specific and fixable: the criterion for calling a unit "sensitive" is never stated numerically, and there is no shuffled-label null anywhere. We should adopt the method and repair those two omissions.

### 8.3 Phase 2 — Graduate-level deep dive

#### 8.3.1 Environment, reward and agent — only what the analyses depend on

The reward is homeostatic and piecewise-constant in four physiological variables:

$$
R(s_t) \;=\; 0.1 \times \Big(1 + \operatorname{sign}(\mathrm{health}_t - 5) + \operatorname{sign}(\mathrm{food}_t - 5) + \operatorname{sign}(\mathrm{drink}_t - 5) + \operatorname{sign}(\mathrm{energy}_t - 5)\Big).
$$

Each of the four terms contributes $+0.1$ when the variable is above half its maximum and $-0.1$ below, so reward is a count of satisfied homeostatic constraints rather than a consumption bonus — the authors note this correlates strongly with survival time and discourages over-grazing. **This is structurally the same reward philosophy as this project's, and it is why the two agents are comparable objects at all.**

Observations: a $9\times11$ egocentric grid view plus an inventory vector carrying health, food, drink, energy and items, encoded by a feedforward layer before the GRU. Architecture: single-layer GRU, 512 hidden units, shared between actor and critic, with separate policy and value heads; PPO with GAE ($\gamma = 0.99$, $\lambda = 0.8$, clip $\epsilon = 0.2$, value weight 0.5, entropy weight 0.01, `tanh` activation, learning rate $2.5\times10^{-4}$, rollout horizon 64, minibatch 8192, gradient-norm clip 1.0); 3 billion environment steps per agent, 24–48 h on one A100 or H100.

Two optional components that define the architecture contrasts used later:

- **Auxiliary path-integration head.** A small fully connected head shares the GRU state $h_t$ with the policy and value heads and predicts the agent's displacement from its episode-start origin, trained jointly with an $L_2$ loss $\mathcal L_{\mathrm{aux}} = \mathbb E_t[\|\hat p_t - p_t\|_2^2]$, weighted by $W_{\mathrm{aux}}$.
- **Magnitude pruning** of the RNN core to 90 % sparsity via JaxPruner, chosen to approximate the 70–94 % sparsity of biological brains.

Table 2 lists the full per-timestep logging schema — 30-odd variables including the complete hidden state, all four physiological levels, the *timers* for hunger/thirst/fatigue/recovery, distances to nearest melee, ranged and passive entities, on-screen counts, absolute and origin-relative position, the value output, policy entropy and log-probability. This schema is worth reading as a specification, because the analyses are only possible because of it, and because it is a direct model for what our own evaluation logging should carry.

#### 8.3.2 Appendix D — the per-neuron encoding GLM, derived and specified

This is the section the review exists for. Everything in it comes from Appendix D (pp. 33–38) and the headline sentence on p. 8.

**Pipeline, step by step.**

1. **Tooling.** GLMs fitted with **NeMoS** (*Neural ModelS*), the Flatiron Institute's neural-GLM library (`github.com/flatironinstitute/nemos`).
2. **Unit of analysis: the neuron.** *One GLM is fit per hidden unit*, 512 fits per agent.
3. **Data per fit.** Each per-neuron model is fit across **multiple episodes — "typically 5–10 episodes minimum"** (the worked example in Figure 21 uses 12) — explicitly "to avoid overtraining to one arena configuration".
4. **Regressors.** The $96\times96$ arena is coarse-grained into a $14\times14$ grid of position bins. For each bin, a **binary indicator** records whether the agent is in that bin at that timestep. That is **196 regressor coefficients**, one per bin. (The paper says "196 total regressor coefficients", which implies **no separate intercept** — with an intercept plus a complete set of exclusive indicators the design matrix would be exactly rank-deficient. The paper does not state this; the inference is mine and should be checked against the released scripts.)
5. **Response preprocessing.** Hidden-unit activity is **shifted to be non-negative** (minimum zeroed) and then **normalised per neuron to $[0,1]$**, so that what is modelled is each unit's activity change relative to its own baseline, "for easier cross-neuron comparison".
6. **Train/test split.** The GLM is fit on the **first 70 % of each episode**, tested on the last 30 % (Figure 19 illustrates the split).
7. **Fit assessment.** Figure 20 plots, one point per neuron, the **difference in log-likelihood between test and train data**. "The majority of neurons fit had comparable performance between the test and training sets." Figure 21 shows one well-fit neuron's activity (black) against its 12-episode model prediction (red).
8. **Selectivity count.** "Approximately **100/512** neurons had good fit performance using just position regressors across runs, with values typically in the range of **60–120 position-sensitive neurons per training run**."
9. **Structure of the coefficients.** The 196 coefficients per neuron are clustered by **k-means into five clusters**, and the number of neurons falling in the **three largest mean-valued clusters** is plotted per position bin (Figure 22) — a spatial map of *how many* units are well fit per location. Figure 23 maps the **average coefficient magnitude** per bin, raw and Gaussian-smoothed.
10. **The headline structural result.** Figure 24 plots average coefficient magnitude against **radial distance from arena centre**, with points as the mean across neurons and error bars as 95 % confidence intervals *across neurons*. Both the *number* of strongly position-encoding neurons and the *average coefficient magnitude* increase with radial distance. Interpretation offered: "an accumulation circuit may ramp with distance from origin and pressure the agent to return to the origin the farther away it gets", partially explaining the observed periodic origin-revisiting.
11. **Architecture generalisation.** Steps 9–10 are repeated identically for **PPO-RNN without path integration** (Figures 25, 26) and for **PQN-RNN** (Figures 27, 28). The ramp appears in all three, "suggesting it is fundamental to the task solution".

**The mathematics, since the paper does not give it.** Let $y_{it} \in [0,1]$ be the normalised activity of unit $i$ at time $t$, and let $x_t \in \{0,1\}^{B}$ ($B = 196$) be the one-hot position indicator, so $x_{tb} = 1$ iff the agent occupies bin $b$ at time $t$. A GLM with link $g$ posits

$$
g\big(\mathbb E[y_{it}\mid x_t]\big) \;=\; \eta_{it} \;=\; \sum_{b=1}^{B}\theta_{ib}\,x_{tb} \;=\; \theta_{i,b(t)},
$$

where $b(t)$ is the bin occupied at time $t$. **Because the indicators are mutually exclusive and exhaustive, the linear predictor collapses to a single coefficient — the one belonging to the occupied bin.** This has a consequence worth stating plainly: the model has exactly one free parameter per bin and imposes *no* smoothness or linearity in space. Maximising the likelihood therefore decouples across bins. For a Gaussian family with identity link,

$$
\hat\theta_{ib} \;=\; \frac{1}{|\mathcal T_b|}\sum_{t \in \mathcal T_b} y_{it}, \qquad \mathcal T_b = \{t : b(t) = b\},
$$

and for the Poisson family with log link (NeMoS's canonical choice for neural data),

$$
\hat\theta_{ib} \;=\; \log\!\left(\frac{1}{|\mathcal T_b|}\sum_{t\in\mathcal T_b} y_{it}\right).
$$

Either way, **the fitted coefficient vector *is* the unit's binned tuning curve**, up to a fixed monotone transform. That is what licenses reading "average coefficient magnitude per bin" as a tuning map and plotting it against radial distance. It also means the analysis is, mathematically, a maximum-likelihood-fitted conditional-mean-per-bin — with the likelihood providing a principled goodness-of-fit score and the held-out split providing overfitting control that a raw bin-average would not have.

**Which observation family they used is not stated.** NeMoS is built around Poisson observation models for spike trains; the response here is a continuous value in $[0,1]$, for which a Poisson likelihood is a quasi-likelihood at best. Log-likelihoods are reported (Figure 20), so *some* family was chosen. **The paper does not say which, and this must be resolved from the released scripts before reimplementation.** For a bounded continuous response a Gaussian or a Beta/quasi-binomial family would be the defensible choices.

**The goodness-of-fit statistic.** Writing $\ell_i(\mathcal D) = \sum_{t\in\mathcal D}\log p(y_{it}\mid x_t; \hat\theta_i)$ for the log-likelihood of unit $i$'s fitted model on data set $\mathcal D$, Figure 20 plots

$$
\Delta\ell_i \;=\; \ell_i(\mathcal D_{\mathrm{test}}) - \ell_i(\mathcal D_{\mathrm{train}})
$$

(presumably per-datapoint-normalised, since train and test differ in size — again unstated). A unit that has overfit its 196 coefficients to the training portion shows a strongly negative $\Delta\ell_i$; a unit whose tuning is stable across the split shows $\Delta\ell_i \approx 0$. **This is a genuine and appropriate overfitting guard, and the coordinator's reading of Figure 20 is correct.**

**But it is not a null calibration, and there is no threshold.** Two omissions must be stated plainly:

- The text never gives a **numeric criterion** for "good fit performance". "Approximately 100/512" and "typically 60–120" are reported as if from a threshold that is never printed.
- **No shuffled-label, circularly-shifted or otherwise permuted null distribution appears anywhere in Appendix D.** $\Delta\ell$ controls overfitting; it does not tell you what $\ell$ a unit would achieve against a position variable that carries no real relationship to its activity. Because position is strongly temporally autocorrelated and hidden-unit activity is too, a substantial apparent fit can arise from shared slow drift alone.

Anyone reimplementing this should add both. The correct null for autocorrelated regressors is **not** an i.i.d. shuffle of the bin labels — that destroys the autocorrelation and produces an anti-conservatively narrow null. Use a **circular shift** of the regressor time series by a random offset (per episode), or a **block permutation** with blocks long relative to the regressor's autocorrelation time, and declare unit $i$ selective when its held-out log-likelihood exceeds, say, the 95th percentile of its own null distribution over ~200 shifts. That is a few extra lines and it converts "approximately 100 neurons looked good" into a calibrated count.

#### 8.3.3 The population decoding analysis (§3.3, Appendix C) — and why it is the *coarser* instrument

Record hidden states $h_t \in \mathbb R^{512}$ and ground-truth positions. For each temporal offset $\Delta t$ (past and future), fit a separate ridge decoder

$$
f(h_t) = A h_t + b, \qquad A \in \mathbb R^{2\times 512},\; b\in\mathbb R^{2},
$$

minimising

$$
\mathcal L_{\Delta t}(f) \;=\; \sum_{i=1}^{N}\big(Y^{i}_{t+\Delta t} - f(h^{i}_t)\big)^2 + \alpha\|f\|_K^2,
$$

where $Y_{t+\Delta t} = (\Delta x, \Delta y)$ is the allocentric displacement $\Delta t$ steps in the past or future. Ridge is chosen for interpretability, numerical stability and $O(Np^2)$ cost; the $\ell_2$ term is needed because some RNN units are inactive for entire episodes, making the problem ill-conditioned. Trained on the first 75 % of each episode and tested on the last 25 %, across multiple episodes, to prevent leakage through temporal continuity in $h_t$. Accuracy is RMSE against an average-displacement chance baseline. Separate decoders are fitted for allocentric (origin-relative) and egocentric (body-relative) frames.

Results: allocentric decodable to $\pm 50$–100 steps; egocentric at chance throughout; decoding improves over training; a single decoder generalises across arenas; pruned networks decode *better* than dense ones (Fig 16); past- and future-coding units overlap but separate more at longer horizons, and separate more cleanly in sparse networks (Fig 18).

**The instrument-sensitivity point.** Compare the two analyses structurally. The decoder is **one linear map from all 512 units to a 2-D continuous target** — it asks whether position is recoverable *linearly from the population*. The GLM is **512 independent maps from one binned categorical variable to one unit's activity** — it asks whether each unit's activity varies with position *in any way expressible as a per-bin mean*. The second is strictly more flexible per unit (196 free parameters, no linearity assumption) and strictly less demanding globally (no requirement that the code be linearly readable, or consistent in sign across units).

This is why the tension noted in §8.2 arises and is, on inspection, resolvable. Appendix C.1 and Figure 17 state that removing the auxiliary path-integration objective "eliminated above-chance decoding of allocentric position from the RNN state" and that the loss "was not merely a regularizer, but essential for inducing spatially interpretable internal representations." Appendix D states that the position-coefficient radial ramp "was present even without the position predicting auxiliary objective (Figures 25 and 26)", and those figures are captioned as showing "the average GLM coefficient value across **well-fit neurons** per bin" — i.e. well-fit neurons existed in that condition. **The paper never comments on the juxtaposition.** Two reconciliations are available and they are not mutually exclusive: (i) the decoder targets displacement at offsets $\pm\Delta t$ including memory and prediction, whereas the GLM targets *current* position only, so current position could survive while past/future encoding does not; (ii) the GLM's per-bin flexibility detects a nonlinear or non-monotone spatial tuning that a linear population readout cannot express. Reconciliation (ii) is the one with teeth for us, and it generalises to a rule: **a null from a linear population decoder is weak evidence of absence, because a tuned-but-nonlinear code is invisible to it and visible to a binned encoding model.** I state this as my reading of two separate claims the authors make; it is not a claim they make themselves.

#### 8.3.4 The behavioural decision GLMs and their multicollinearity diagnostics (§4.2, B.4, Fig 13) — a *different* model, deliberately kept separate

These must not be conflated with the neural GLM, and the coordinator was right to say so.

- **Model**: a logistic GLM classifying whether a patch was **chosen (1) or not (0)** at a revisitation decision point, with choices defined 50 timesteps before each patch eat event.
- **Regressors** (Fig 3): prior eat rate at that patch, drink rate / water proximity, predator encounter rate, recency of visitation, dwell time, observed cow count, and the agent's prior position-prediction error at that patch ("uncertainty").
- **Data**: a **single model fit jointly across five PPO-RNN agents, spanning 7978 revisitation decisions**, with **agent ID as a fixed effect** to absorb inter-agent variability. Fitted with `statsmodels.formula.api`.
- **Findings**: agents prefer patches with fewer prior eats, avoid patches with more predator encounters, prefer more recently visited patches, mildly prefer longer dwell times, prefer more observed cows, and prefer patches with higher prior position-prediction error. Significance stars at $p<0.05/0.01/0.001$, 95 % CI error bars.
- **Diagnostics (B.4)**: **variance inflation factors**, all $< 10$, "confirmed that none of the regressors exhibited problematic multicollinearity."

For a predictor $j$ regressed on all other predictors with coefficient of determination $R_j^2$,

$$
\mathrm{VIF}_j \;=\; \frac{1}{1 - R_j^{2}},
$$

which is the factor by which the estimated variance of $\hat\beta_j$ is inflated relative to an orthogonal design. $R_j^2 = 0.9$ gives $\mathrm{VIF}_j = 10$; the conventional alarm thresholds are 5 or 10, and the paper uses 10.

**Crucially, the VIF check belongs to the behavioural model and is not reported for the neural GLM** — and in their setup the neural GLM does not need it, because mutually exclusive one-hot bins cannot be strongly collinear. For exclusive indicators with occupancies $p_a, p_b$,

$$
\operatorname{Cov}(x_a, x_b) = -p_a p_b, \qquad \operatorname{Var}(x_b) = p_b(1-p_b),
\qquad
\operatorname{Corr}(x_a, x_b) = -\sqrt{\frac{p_a p_b}{(1-p_a)(1-p_b)}},
$$

which is small whenever occupancy is spread over many bins. **Their design avoids the collinearity problem by construction.** That fact is the key to transferring the method to a correlated predictor set (§8.4).

#### 8.3.5 The architecture-contrast structure

| Contrast | Where | What it was used to conclude |
|---|---|---|
| **PPO-RNN vs. PPO feedforward** | Fig 2 (top left), Fig 10 | Recurrence is necessary; memoryless agents loop locally and die early |
| **512 vs. 128 vs. 64 hidden units; pruned vs. dense** | Fig 2 (top left/right) | High capacity is needed to *support* sparsity; pruning costs nothing in performance and *improves* decoding (Fig 16) |
| **With vs. without auxiliary path-integration loss** | Fig 2 (bottom left), Figs 11, 17, 25, 26 | Needed for large arenas but not small; **eliminates population decoding**; but the single-unit radial ramp persists |
| **360° vs. front-facing field of view** | Fig 2 (bottom right), Figs 14, 15 | Equal final performance concealing real behavioural differences — the paper's own demonstration that a performance null hides a mechanism difference |
| **PPO-GRU vs. PQN-LSTM** | Figs 29–31, Appendix E | Comparable performance for the ~30 % of PQN runs that learned at all, similar pathing, but **PPO converges on predator *fighting* and PQN on predator *evasion*** — again, equal reward, different strategy |
| **Neural GLM across all three architectures** | Figs 22–28 | The radial coefficient ramp is present in PPO-RNN, PPO-RNN-no-path-integration and PQN-RNN → argued to be "fundamental to the task solution" rather than architecture-specific |

Two features of this structure are worth naming because they are the template for a modulated-versus-unmodulated comparison. First, **the architecture contrast is used in two opposite directions**: to show that something *differs* (decoding collapses without the auxiliary loss) and to show that something is *shared* (the radial ramp survives every manipulation). Both are informative; the second is the one usually neglected. Second, **the paper twice reports equal performance concealing different mechanisms** (front-FOV, PQN-vs-PPO), which is the paper's own strongest argument for why a behavioural null does not settle a mechanistic question.

The neural-GLM architecture comparison is, however, **qualitative**: the three conditions are compared by the *shape* of a heat map and of a radial curve, with each figure captioned as data from *a* single run. There is no across-architecture test on the count of selective units, and no such count is even reported per architecture beyond the pooled "60–120 per training run".

#### 8.3.6 Where the paper is honest about limits

- Craftax's 96×96 grid cap prevents testing whether the results scale to larger arenas.
- Tree, rock and lake positions were not logged, limiting landmark-navigation analysis — an explicit admission that the logging schema, not the method, was the binding constraint.
- Adding further RL architectures with full behavioural-neural logging is "currently non-trivial in this environment", hence the restriction to PPO and PQN with shared components.
- Only ~30 % of PQN runs learned the task at all; the rest "failed catastrophically", attributed to suboptimal exploration.
- The Appendix D analysis is explicitly framed as "a supplementary analysis" — this project is treating it as a headline method, which is a deliberate and stated re-weighting.
- The introduction concedes that common representational-similarity metrics "can yield inconsistent conclusions", citing the reproducibility literature — a relevant caution given that the Ben-Iwhiwhu paper in this same corpus rests entirely on one such metric (§7).

### 8.4 Relevance to this project

**What transfers, concretely — and this is the strongest set in the corpus.**

1. **The per-unit encoding GLM, with internal state substituted for position, is the method to run first.** The method is entirely **indifferent to what the regressors encode**: it is a per-unit generalised linear model on an arbitrary design matrix, and position enters only as a choice of columns. Two distinct deployments, both immediately available:
   - *(a) On hidden activations.* For each site's 128 units, fit one GLM of unit activity on binned internal state (satiation, felt injury / nociception, energy). Output: a count and a distribution of state-sensitive units, per run.
   - *(b) On the modulator's own output.* For each site's 128 gain coefficients, fit one GLM of $\gamma_{it}$ (and separately $\beta_{it}$) on the same design matrix. Output: **how many of the modulator's 128 output channels are actually tuned to internal state.** This is our open question, asked directly, at the level of the object we care about, with no intervention, no counterfactual rollout, no paired samples and no retraining. **It is the cheapest decisive measurement in this review.**
2. **The unit of analysis is the neuron, and that is decisive for a one-seed project. The coordinator's reading is confirmed, with one qualification.** Each run yields $n = 512$ (for them) or $n = 128$ per site (for us) independent model fits, so a *single* run produces a distribution with a mean, a spread and a confidence interval — their Figure 24 error bars are explicitly "the mean across neurons ... 95 % CI". A method whose output is one scalar per run is uninterpretable with $n=1$; this one is not. **The qualification**: units within a run are not independent replicates of *the run*, so a within-run statement ("this network has 94 state-tuned units in its actor site, with tuning strength rising with injury severity") is well supported, while a between-arm statement ("the four-site arm has more state-tuned units than the encoder-only arm") is a comparison of two point estimates and inherits all the usual one-seed weakness. The paper supplies the calibration for how weak: **their own run-to-run count varied from 60 to 120 out of 512** — a spread of a factor of two on nominally identical training. Any between-arm difference smaller than that should not be believed from single runs.
3. **The joint-binning trick solves our multicollinearity problem, and it is already in their design.** The coordinator's concern is right: satiation and felt injury will be correlated over an episode, and B.4's VIF diagnostic belongs to a *different* model and is not applied to the neural GLM. But notice what their spatial design actually is — $x$ and $y$ are themselves two continuous variables that are heavily correlated over a trajectory, and the authors never regress on $x$ and $y$ as separate linear terms. They **jointly bin the pair into a 14×14 grid of mutually exclusive indicators**, which (i) makes collinearity structurally negligible (§8.3.4), (ii) imposes no functional form, and (iii) yields a readable 2-D tuning map. **Do exactly the same with (satiation, injury)**: a $K\times K$ grid of joint internal-state bins, one indicator each. Collinearity disappears by construction; the output is a 2-D internal-state tuning map per unit, directly analogous to a place field; and the correlation between the two variables becomes a *coverage* problem (some cells rarely occupied) rather than an *identifiability* problem. Report the occupancy map alongside, as they do — their Figures 22 and 23 explicitly annotate bins the agent never visited, and ours will have empty corners for the same reason (e.g. "severely injured and fully satiated" may be rare). Where a linear-in-state term is wanted instead, then and only then does B.4's VIF check become the required diagnostic, with the same $<10$ convention.
4. **Add the null the paper omits.** Declare a unit state-sensitive when its held-out log-likelihood beats the 95th percentile of a null built by **circularly shifting the state regressor within each episode** (not i.i.d. shuffling — internal state is far more slowly varying than position, so an i.i.d. shuffle would give an absurdly permissive null). Report both the count of selective units and the *full distribution* of held-out log-likelihood minus its null median, so that a null result is visibly a null rather than a threshold artefact. Keep their train/test split (they used 70/30 within episode; their decoding analysis used 75/25) and their multi-episode pooling, which is the guard against fitting one arena — for us, one grid layout or one initial condition.
5. **Their architecture contrast is a direct template for modulated versus unmodulated.** Run the identical GLM pipeline on a modulated run and its unmodulated counterpart from the same grid and compare (i) the count of state-selective units, (ii) the distribution of tuning strength, (iii) the shape of tuning against a graded state variable (their radial-distance plot becomes "coefficient magnitude versus injury severity"). Their own logic supports both outcomes being publishable: a *difference* localises what modulation adds; an *identity* — the pattern present in both arms, as their radial ramp was present in all three of theirs — argues the structure is a property of the task rather than of the mechanism, which given our behavioural null would be the honest and interesting negative result.
6. **The logging schema (Table 2) is a specification worth copying.** Their analyses are possible only because the hidden state, every internal variable, every timer, every distance-to-entity and the value and entropy outputs are all logged per timestep. Our evaluation logging currently does not carry per-unit activations or per-unit $\gamma,\beta$. That gap, not any conceptual difficulty, is what stands between us and running this analysis today.
7. **"Equal performance conceals different mechanisms" is this paper's central methodological claim and it is directly the situation we are in.** They show it twice with clean examples — front-facing versus 360° field of view (identical final performance, different exploration), and PPO versus PQN (comparable survival, one fights predators and one evades them). A behavioural null between our modulated and unmodulated agents is therefore, on this paper's evidence, *not* grounds for concluding the modulator changes nothing; it is grounds for looking at the units.

**What does not transfer, and why.**

- **Everything spatial.** Their headline finding — position-sensitive units, a distance-from-origin ramp, allocentric-versus-egocentric decoding, past/future planning horizons — is about a 96×96 arena with a stable global reference frame and episodes running to tens of thousands of steps. Our 10×10 grid with 128-step rollout windows supports nothing analogous: with 100 cells and short windows there is no meaningful radial structure, and "planning horizon" at our timescale is a different question. **Import the instrument, discard the variable.**
- **The auxiliary path-integration head and the pruning sweep** both require retraining and are out of scope for a checkpoint-only analysis.
- **Five-seed behavioural comparisons and the 7978-decision pooled behavioural GLM.** Their decision GLM pools five agents with agent ID as a fixed effect. We have one seed per configuration, so the equivalent pooling would be across configurations — which confounds the very factor under study. The *behavioural* GLM therefore transfers much less well than the *neural* one, and this is the one place where the coordinator's original "behaviour, not representation" framing was pointing at something real.
- **The bayesmove movement-state segmentation and conditional-inference-tree analysis (B.2)** are built for long, spatially extended trajectories with meaningful turning angles and step sizes. On a 10×10 grid with four movement actions this is unlikely to yield three interpretable locomotor states. Marked as low-value for us.
- **Comparison to insect search patterns (B.3)** is a qualitative analogy with no quantitative instrument attached.
- **The environment itself.** ForageWorld is a genuinely appealing target — homeostatic reward, depleting patches, predators, partial observability, and *already instrumented for exactly these analyses with public code*. It is not a method that transfers to our environment; it is an alternative platform. Noting it as such and going no further.

**What would they have concluded from a behavioural null?** This paper is the only one of the three that gives a direct, evidenced answer, and the answer favours looking harder. Their explicit thesis is that reward curves and aggregate performance "offer limited insight into how agents solve structured tasks", and they demonstrate the point twice with **behavioural nulls that concealed real mechanistic differences** — the front-FOV agents whose "final performance conceals behavioral differences", and the PQN agents whose comparable survival hid a completely different predator strategy. On the representational side, the no-path-integration condition supplies the sharpest case: a *coarse* instrument (linear population decoding) returned a flat null in a condition where a *fine* instrument (per-unit encoding GLM) still found structure. Translated to our situation: **the absence of a behavioural difference between modulated and unmodulated agents is, on this paper's stated position, exactly the circumstance in which per-unit analysis is warranted, and the absence of a difference in a pooled or population-level statistic is not sufficient grounds for concluding the modulator is inert.** The one thing this does *not* license is the converse claim — nothing in the paper shows that a representational difference will be found, only that the search is the methodologically correct response. And the reconciliation of Figure 17 with Figures 25–26 is my inference from two of their claims, not a conclusion they draw.

**Compute and data requirements.** The *fitting* is trivial and the coordinator's expectation is confirmed: 512 GLMs of 196 coefficients each, over a few tens of thousands of timesteps, is seconds to a couple of minutes on CPU per agent; the closed-form structure derived in §8.3.2 means each fit is essentially a set of per-bin means with a likelihood evaluation. For us the numbers are smaller still — 128 units per site. Adding a 200-shift circular-shift null multiplies the cost by 200 and it remains cheap, and it parallelises perfectly.

What is *not* cheap in their pipeline, and where the real cost sits for us:

- **Their training** — 3 billion timesteps per agent, 24–48 h on an A100/H100 — is irrelevant to us; ours is already spent.
- **Rollout generation and logging** is the actual bottleneck. Their per-fit requirement of 5–10 episodes translates for us into enough evaluation rollouts to populate every internal-state bin adequately. This is the number that should be estimated before committing: with a $K \times K$ joint state grid and a target of a few hundred timesteps per occupied bin, the required rollout volume scales with $K^2$ and with how skewed our state occupancy is. **Pilot on one run and one site before scaling to 32 runs × 4 sites × 50 checkpoints.**
- **The 50-checkpoints-per-run axis is an opportunity their design lacks.** They report decoding accuracy at only a handful of training stages and note their error bars are inflated because early-training agents die too soon to supply timesteps. We have 50 checkpoints per run, which supports a much finer "when during training does state tuning appear, if it ever does" curve than anything in the paper — and if the answer is "never", that curve is itself the result.
- **Storage**: per-unit activations and per-unit $\gamma,\beta$ at 128 units × 4 sites × 128 steps × a few hundred episodes is on the order of $10^7$–$10^8$ floats per run. Compressible, but it should be sized with measured bytes rather than raw arithmetic before a 32-run sweep is committed.

Both prerequisites — evaluation-time per-unit logging, and a state-binning utility — are tooling changes and belong to `senior-developer` as an `issue_plan`. This review does not plan them.

### 8.5 Appendix: Section-by-Section Backbone

Preserving the paper's original section order.

**Abstract.** Understanding DRL agent behaviour as task and agent sophistication increase requires more than reward-curve comparison, yet behavioural-analysis methods remain underdeveloped in DRL. The authors apply neuroscience and ethology tools to DRL agents in **ForageWorld**, a novel, complex, partially observable environment capturing sparse depleting resource patches, predator threats and spatially extended arenas, and use it as a platform for **joint behavioural and neural analysis**. Contrary to common assumptions, model-free RNN-based agents exhibit structured, planning-like behaviour purely through emergent dynamics, without explicit memory modules or world models. The tools are distilled into a general analysis framework linking behavioural and representational features to diagnostic methods.

**1. Introduction.** Complex, naturalistic, open-ended tasks create a need for tools to understand both behaviour and neural dynamics. ML has adopted neural-representation-analysis tools from neuroscience, but mature *behavioural* analysis methods remain underused. RL evaluation still focuses on reward curves and aggregate performance. Contributions: (1) ForageWorld; (2) adaptation of path analysis, **generalized linear models**, and recurrent-state decoding for joint behavioural-neural analysis; (3) evidence that model-free RNN agents show planning-like behaviour without world models; (4) a released reproducible pipeline and open-source codebase. Five motif classes analysed: exploration, decision-making, learning dynamics, internal representations, and (implicitly) capacity.

**Table 1 — the reusable analysis framework.** *Goal inference* → decision GLMs, in-context learning study, behaviour phase segmentation. *Memory span* → RNN state decoding, memory-module ablations. *Planning horizon* → future decoding, auxiliary-loss analysis for predictive objectives. *Spatial structure* → position occupancy entropy, tracking revisitation, path analysis. *Network capacity* → network size / pruning sweeps, comparing performance curves. **Representations** → **decoding, GLMs, encoding profiles of task variables**.

**2. Background.** *2.1* Generalisation is critical yet many RL algorithms fail beyond training tasks, renewing interest in open-ended tasks requiring rich exploration, subgoal discovery and long-horizon reasoning. *2.2* Open-ended RL benchmarks have not attracted sustained neuroscience interest, partly because few reflect how animals explore, plan and decide. Recent neuroscience-to-ML work has focused on **representational similarity**, not joint behaviour-and-neural analysis; **common similarity metrics can yield inconsistent conclusions**. Neural activity alone is often insufficient to explain computation. The authors adopt a **behaviour-first** approach.

**3. Methods.** *3.1 ForageWorld*: built on Craftax; cows have an arena-wide count limit, spawn at fixed points and diffuse, causing temporary localized depletion and forcing periodic patch revisitation; water maintains hydration; predators appear near food patches and pursue on line of sight; agents manage hunger, thirst, fatigue and health; fatigue recovers only through sleep (permitted below 50 % energy, immobilising the agent); starvation and thirst reduce health; death at zero health; arenas procedurally generated per episode. **Reward is survival-oriented (Eq. 1)** — $0.1\times$ the count of physiological variables above half maximum. Per-timestep logging of neural states, actions and environment variables (Table 2), supporting held-out evaluation. *3.2 Architectures*: **PPO with a GRU core (PPO-RNN)**; an optional auxiliary head predicting position relative to the first-timestep origin (**path integration**) sharing the RNN hidden state with policy and value heads, trained with an $L_2$ loss (Eq. 2) weighted by $W_{\mathrm{aux}}$; **JaxPruner magnitude pruning to 90 % sparsity**, motivated by 70–94 % biological sparsity, with pruned agents performing comparably and sometimes decoding better. *3.3 Decoding*: record $h_t$ and ground-truth $(x_t,y_t)$; at each $t$ train a decoder predicting allocentric displacement $\Delta t$ steps into past or future from $h_t \in \mathbb R^{512}$; separate decoders for allocentric and egocentric frames; **ridge regression** for interpretability; trained on the first 75 % of each episode and evaluated on the last 25 %; RMSE against an average-displacement baseline; **all behavioural analyses in held-out test arenas with frozen weights**.

**4. Results.** *4.1 Structured exploration and revisitation*: trajectory visualisation, behavioural segmentation and spatial entropy show phase-like foraging; agents generate outwardly spiralling, azimuthally rotating loops that expand to cover the arena, resembling insect search; exploration persists during revisitation and supports predator avoidance by revealing shortcuts. *4.2 Multi-objective foraging*: after exploration, agents shift to direct revisitation of remembered patches and periodically return to the arena origin. Consumption rates for expert agents: eat 0.011 per step (~once every 95 steps), drink 0.034, sleep 0.262. **Patch-choice GLM (Fig 3)**: choices defined 50 timesteps before each patch eat event; agents prefer patches with fewer prior eats, show no water-proximity preference, avoid patches with more predator encounters, prefer more recently visited patches, mildly prefer longer dwell time, prefer patches with more observed cows, and prefer patches with higher prior position-prediction error (uncertainty); significance stars and 95 % CIs; full output and VIF in Fig 13. *4.3 Emergence over training*: staged learning; an early stationary "fishing" strategy near the origin; an abrupt multi-metric shift after ~20 000 training iterations bringing longer-distance travel, better food/water trade-offs, a sharp rise in tool-making and gradual predator-defence gains. Ablations: **front-facing field of view** produces earlier, denser local exploration, with **final performance concealing behavioural differences**; **PQN-LSTM** performs comparably to PPO-GRU with similar learning histories and pathing but much lower predator-killing rates — PPO converges on fighting, PQN on evasion. *4.4 Recurrent state supports memory and planning*: allocentric position decodable from the current RNN state up to **50–100 timesteps into past and future**; **~100/512 neurons position-sensitive at the single-neuron level by GLM analysis**, with **GLM coefficient magnitude increasing with radial distance from origin**, suggesting a distance-accumulation circuit modulating origin-revisitation (Appendix D); decoding accuracy improves over training; **egocentric orientation could not be reliably decoded**. *4.5 Generalisation and modularity*: pruned networks decode moderately better; **removing the auxiliary position objective eliminates above-chance decoding**, implying the associated performance loss stems from an inability to encode position under the PPO objective alone; past- and future-position encoding rely on overlapping but modular subpopulations that diverge at longer horizons; a single decoder generalises across episodes, attributed to the preserved 96×96 layout and global orientation providing a stable allocentric reference frame.

**5. Discussion.** Two findings: ForageWorld induces behaviour–neural motifs relevant to planning and memory, and the toolkit reveals them where simple performance metrics cannot. Challenges the view that sophisticated planning and memory require explicit world models, mammalian-like brains or symbolic memory; claims the first systematic evidence that such behaviour emerges in complex naturalistic environments without world models. Architectural choices — recurrence, pruning, auxiliary losses — significantly influence both performance *and* interpretability; the auxiliary objective is a "win-win". Agents with a few hundred recurrent units may be useful models for small nervous systems; cross-arena decoder generalisation suggests a stable, compass-like encoding of space. Disputes the idea that DRL models are too complex for neuroscience-style analysis: "If neuroscience tools cannot explain DRL agents with full behavioral and neural access, their utility for understanding the brain is limited." *Limitations*: the 96×96 grid cap; failure to log tree/rock/lake locations, limiting landmark-navigation study; the difficulty of instrumenting further RL architectures with full behavioural-neural logging.

**Appendix A — Task and training details.** Design balances complexity, transparency to analysis, JAX-accelerated training efficiency, biological plausibility and neuroscientist accessibility; explicitly positioned against both n-armed-bandit foraging tasks (too simple) and Minecraft-scale tasks (too costly and hard to instrument). *A.1* Procedural generation: fixed 96×96 grid, Perlin-noise resource patches, line-of-sight-breaking obstacles, agent starts at centre. *A.2* Episodes cap at 100 000 timesteps or end at zero health, which is the typical outcome. *A.3* Observation: $9\times11$ egocentric grid view plus an inventory vector (health, food, water, fatigue, items), encoded by a feedforward layer before the RNN. *A.4* Logging: full recurrent hidden state, internal reward components, trajectories, environment state and decisions each timestep (Table 2). *A.5* Hyperparameters: PPO with GAE, 3 billion timesteps per agent (hundreds of thousands of episodes), rollout horizon 64, minibatch 8192, learning rate 0.00025, Adam, gradient-norm clip 1.0, single A100/H100 per run converging in 24–48 h, ≥24 GB memory, logging every 2048 PPO iterations (~134 M timesteps), checkpoints selected on held-out arena performance. *A.6* Architecture: PPO clipped surrogate (Eq. 3); single-layer **GRU with 512 hidden units** shared between actor and critic with separate heads; ~6.5 M trainable parameters; follows the Craftax implementation. *A.7* Future directions: curriculum learning; LLM foraging tests (noting transformers would be harder to analyse than RNNs); multi-agent support.

**Appendix B — Interpreting agent behaviour.** *B.1* Representative trajectories across three episodes showing early loops, transition, and full-episode revisitation (Fig 6). *B.2* **Behavioural phase segmentation**: the `bayesmove` package identifies three latent movement states from turning angle and step size over a 7-timestep moving window; conditional inference trees identify which task variables predict transitions; states correspond to short-, mid- and long-range navigation and are differentially associated with predator presence, eating and positional uncertainty (Fig 7). *B.3* Comparison to ant and bee search patterns (Fig 8). *B.4* **GLM outputs and multicollinearity checks**: each GLM classifies whether a patch was chosen at a revisitation point from historical variables; **fitted jointly across five PPO-RNN agents over 7978 revisitation decisions with agent ID as a fixed effect**, using `statsmodels.formula.api`; uncertainty, predator encounter rate and recency emerged as significant, dwell time and drink rate less so; **VIF scores all < 10** (Fig 13). *B.5* Policy entropy declines then slowly rises again late in training. *B.6* Memory and predator ablations: memoryless agents become trapped in local loops and survive far less (Fig 10); agents unable to damage predators show similar movement and survival, indicating **evasion rather than combat is the primary strategy** (Fig 12).

**Appendix C — Interpreting agent representations.** *C.1* Removing the auxiliary path-integration objective **eliminated above-chance allocentric decoding** (Fig 17): the loss "was not merely a regularizer, but essential for inducing spatially interpretable internal representations". Pruned networks decode better than dense (Fig 16). *C.2* Ridge-regression decoder weights aligned by neuron ID (Fig 18): overlapping populations code past and future, with overlap diminishing at longer offsets and sparse networks showing clearer separation and sparser weight maps. *C.3* Ridge with $\ell_2$ penalty (some RNN units are inactive for whole episodes, making decoding ill-conditioned); linear model $f(h_t) = Ah_t + b$ with $A \in \mathbb R^{2\times512}$; one decoder per offset $\Delta t$; $O(Np^2)$ cost; **75/25 within-episode train/test split** to prevent leakage from temporal continuity; decoders trained across multiple episodes.

**Appendix D — Generalized linear model for predicting neural activity from current position.** Framed as "a supplementary analysis" of how position is represented at the single-neuron level. **NeMoS** used for fitting. **One GLM per neuron**, with **5–10 episodes minimum per model fit** to avoid over-training to one arena configuration. The 96×96 arena is coarse-grained into **14×14 position bins**, with **a binary indicator per bin — 196 total regressor coefficients**. Neural data is shifted to non-negative values and **normalised to 0–1 per neuron** to express activity change relative to that neuron's baseline. **Fit on the first 70 % of each episode, tested on the last 30 %** (Fig 19); most neurons show comparable test and train performance (**Fig 20**: test-minus-train log-likelihood, one model per data point, PPO-RNN baseline run). **~100/512 neurons had good fit performance from position regressors alone, typically 60–120 per training run**; an example good fit across 12 episodes is Fig 21. Coefficients are **k-means clustered into five clusters**, with neurons in the three largest mean-valued clusters counted per bin (Fig 22); average coefficient per bin mapped raw and Gaussian-smoothed (Fig 23). **Both the number of strongly position-encoding neurons and the average coefficient magnitude increase with radial distance from the arena origin** (Fig 24; means across neurons, 95 % CI error bars) — read as a possible accumulation circuit ramping with distance and pressuring origin return. The trend holds in **PPO-RNN without path integration** (Figs 25, 26) and in **PQN-RNN** (Figs 27, 28), "suggesting it is fundamental to the task solution". **Example scripts are in the paper's GitHub repository.**

**Appendix E — Parallelized Q-networks (PQN).** PQN adapted to ForageWorld using the LSTM architecture and default hyperparameters from the source work, sharing environment and training hyperparameters with PPO except where noted. An orthogonal test of whether a different algorithm produces different behaviour. PQN-LSTM matched PPO-GRU **for the subset of runs that learned**, but across the two successful exploration-hyperparameter configurations **only ~30 % of PQN runs performed comparably**, the rest failing catastrophically in a way never seen in PPO — attributed to suboptimal exploration locking in early, reliable but suboptimal behaviour. Successful runs showed similar learning histories and pathing but **much lower predator-killing rates**, indicating PPO converges on fighting and PQN on evasion.

---

## 9. Cross-paper synthesis

### 9.1 What the three papers agree on, without having coordinated

**A low-dimensional modulatory code, broadcast through a per-unit linear projection.** Vecoven's gains are $\gamma = W_s z$ with $z \in \mathbb R^{k}$; Ben-Iwhiwhu's are $\gamma = \tanh(W^m g)$ with $g \in \mathbb R^{m}$, $m \in \{4,8,16,32\}$ against hidden widths of 100–600. Neither paper remarks on the consequence, which is that **the set of achievable per-unit gain patterns is a subspace of dimension at most $k$ (or $m$)**. Our own modulator has the same structure — a 16-unit GRU state read by per-unit linear heads — so our per-site gain matrix has temporal rank at most 16 on top of a static per-unit baseline. This makes method **M6** (a singular-value decomposition of the mean-subtracted gain matrix) an immediately interpretable diagnostic: it counts how many of the available modulatory modes the trained modulator actually uses. A spectrum with essentially all its mass in one component means the modulator has collapsed to a single scalar knob — a strictly stronger and more informative statement than "the contextual fraction is small".

**The mechanism both architectures credit is gating, not scaling.** Vecoven report gains crossing zero and inverting a unit's tuning, and gains at zero making a unit constant and prunable. Ben-Iwhiwhu build their entire method on a `tanh` gain whose *sign* determines which half of a unit's input survives the rectifier, and offer a "strict" variant in which only the sign is used. **Our modulator, with gains reported in the range 0.2–0.9 and no observed sign changes, has never exercised this mechanism.** It is operating in the attenuation-only corner of the design space. That is a one-line measurement (method **M7**) and it reframes our own numbers: it would place our result not as "the modulator failed" but as "the modulator never used the operation the literature credits for the effect".

**Every paper that reports a benefit reports it where context matters for reward.** Vecoven's benchmarks make inferring a hidden task parameter *necessary* to score; Ben-Iwhiwhu's benefit appears only where tasks are dissimilar enough to demand different representations, and explicitly vanishes where they are not. Our environment supplies a single task family. This is the corpus's most uncomfortable and most honest contribution to our behavioural null: **the regime in which both neuromodulation papers report no benefit is the regime we are in.** It is a mechanistic hypothesis, not a proof, and it is testable — it predicts a small contextual fraction and flat encoding statistics — but it should be said out loud rather than softened.

### 9.2 Where the corpus splits

**Self-conditioning.** Vecoven give the modulator the interaction history *with the current observation removed*, an explicit informational separation between "what is the situation now" (main network) and "what kind of world is this" (modulator). Ben-Iwhiwhu give the modulator *the layer's own input* — full self-conditioning, the design we adopted. Both report success. **The corpus therefore does not adjudicate the choice**, and the earlier project-level suspicion that self-conditioning is a liability finds neither support nor refutation here.

**Whether the modulatory signal should be constant or varying.** Vecoven treat a signal that converges within an episode and then holds steady as the *success* condition — "almost state-independent and serves only for adaptation". Our modulator is supposed to do the opposite: track a body state that changes within the episode. **Import their method, invert their expected sign.** Reading their criterion straight across would have us celebrating exactly the collapse we are trying to detect.

**Whether representational analysis can precede a performance difference.** Ben-Iwhiwhu deploy CKA only to explain a gap that already exists. Simmons-Edler deploy per-unit analysis specifically to find what performance metrics miss, and demonstrate twice that equal performance can conceal different mechanisms. Our situation requires the second stance.

### 9.3 The instrument-sensitivity ladder — the corpus's most useful single lesson

Ordering the corpus's instruments by how much structure they require before they will report anything:

1. **Aggregate behavioural comparison** (our current evidence) — requires a difference in what the agent does. Reported null.
2. **Pooled modulator statistics** (our current logging) — requires a difference in means over units, timesteps and environments. Provably cannot distinguish contextual modulation from a constant re-parameterisation (the gauge identity in our own optimisation-dynamics critique).
3. **Linear population decoding** (method M8) — requires the variable to be linearly readable from the population. Simmons-Edler's Figure 17 shows this returning a flat null.
4. **Representational similarity, CKA** (M3, M4) — requires a change in the *geometry* of the representation, and is deliberately blind to uniform rescaling. Sensitive to things (3) misses; still a single scalar per comparison.
5. **Per-unit encoding models** (M1) — requires only that some individual unit's activity varies with the variable, in any way expressible as a per-bin mean. Simmons-Edler's Appendix D finds position-tuned units in the very condition where (3) reported nothing.
6. **Causal intervention** (M2) — does not measure a correlation at all; asks whether behaviour depends on the signal. Does not require any baseline agent.

The lesson is that **a null at one rung is weak evidence about the rungs below it**, and our behavioural null sits on rung 1. Simmons-Edler make this argument by demonstration and it is the corpus's strongest methodological claim. It does not promise that anything will be found lower down. It establishes that not having looked is not the same as having looked.

### 9.4 What the corpus does *not* license

- **It does not license the claim that a representational difference will be found.** Only one paper contains behavioural nulls at all (Ben-Iwhiwhu, on three easy environments), and in those cases the representational analysis *also* showed no advantage for the modulated network. That is weak, qualitative, three-seed, correlational evidence, but it is the only direct evidence in the corpus and it points the other way.
- **It does not license between-arm comparisons from our grid.** Vecoven use 15 seeds, Ben-Iwhiwhu 3, Simmons-Edler 5 for behaviour and report a **60-to-120-out-of-512 run-to-run spread** in their own selective-unit count. We have one seed per configuration. Within-run, per-unit statements are supported; "arm A modulates more contextually than arm B" is not.
- **It does not supply a validated null distribution for any of its encoding analyses.** Vecoven's context-dependence evidence is a scatter plot read by eye; Ben-Iwhiwhu's is a heat map read by eye; Simmons-Edler's per-unit GLM has a genuine overfitting guard (a held-out log-likelihood comparison) but **no shuffled or shifted null and no stated selectivity threshold**. Every one of these needs a null added before it can support a negative conclusion, and for slowly varying internal-state regressors that null must be a circular shift or block permutation rather than an i.i.d. shuffle.
- **It does not contain a single frozen-modulator or shuffled-modulator control outside Vecoven's Figure 10.** For a field organised around the claim that modulation matters, the causal evidence in this corpus rests on one five-panel figure in one 2020 paper.

### 9.5 Recommended reading order

For someone coming to this cold with our question in hand: **§8.3.2** (the per-unit GLM, specified and derived) → **§6.3.4 Analysis C** (the freeze-and-switch protocol) → **§7.3.3** (CKA derived, and why its scaling invariance matters to us) → **§5** (the method inventory) → **§9.3** (the sensitivity ladder). The three Phase 1 sections are the plain-language route for anyone who wants the argument without the mathematics.

### 9.6 Handoffs

Two consequences of this review imply work that is explicitly **not** planned here:

- **Tooling** — evaluation-time logging of per-unit activations and per-unit $\gamma, \beta$, and a hook to feed the modulator a counterfactual context independently of the main network's input. Nearly every method in §5 depends on the first; methods M2 and the strong form of M3 depend on the second. → `senior-developer`, as an `issue_plan`.
- **Analysis design** — pre-registering the candidate context variables, the binning scheme, the null construction and the selectivity threshold *before* the numbers are looked at, since a corpus this thin on nulls makes post-hoc thresholding a live hazard. → `experiment-designer`, with `plan-reviewer` on the resulting plan.

A third item is a judgement call for the user rather than a handoff: **whether a representational-analysis result, positive or negative, is worth the tooling investment given that the behavioural question has already returned a null.** §9.1's third point — that both neuromodulation papers report no benefit in precisely our single-task regime — is the argument on the sceptical side, and it deserves to be weighed before the work is commissioned. That is a `pi`-level focus-versus-explore call.

