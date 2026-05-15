---
title: "Flexible multitask computation in recurrent networks utilizes shared dynamical motifs"
authors: ["Laura Driscoll", "Krishna Shenoy", "David Sussillo"]
year: 2022
venue: "bioRxiv 2022.08.15.503870 (preprint)"
slug: driscoll_2022_dynamical_motifs
source_pdf: "sources/Driscoll et al. 2022 - Flexible multitask computation in recurrent networks utilizes shared dynamical motifs.pdf"
topic: neuromodulatory_algorithms
---

# Driscoll, Shenoy & Sussillo 2022 — Flexible multitask computation in recurrent networks utilizes shared dynamical motifs

## Plain-English entry point

This paper asks a question that sits at the boundary of neuroscience and AI: when a single neural network has to perform many cognitive tasks (e.g., remember a direction, decide whether two inputs match, respond opposite to a stimulus), how does it reuse machinery across tasks instead of building each task from scratch? The authors train one recurrent neural network (RNN) — a brain-inspired network whose units feed back on each other — on **15 cognitive tasks** drawn from monkey neurophysiology (memory tasks, pro/anti saccade tasks, integration, categorization). Then, instead of just reporting accuracy, they crack the network open using **dynamical-systems analysis**: they find the network's "fixed points" — states where activity is stationary or near-stationary — and see how those fixed points form geometric structures called **attractors** (states the network gets pulled toward) and **decision boundaries** (states that push activity in opposite directions). The key discovery: tasks that share a *computational subpart* (say, "remember an angle") share the *same geometric structure* in state space — what they call a **dynamical motif** (e.g., a ring attractor for circular memory, two-point attractors for category memory, an unstable fixed point for a decision boundary). When the rule input changes, the same underlying scaffold is reused with small tweaks. Lesioning a cluster of units breaks only the tasks that depend on the motif those units implement. New tasks can be learned 100× faster by training only the rule-input weights, leaving the motifs intact. This matters because it gives a concrete, mechanistic model of **compositionality** — how a network builds complex behavior by reusing simple parts — and proposes "dynamical motif" as a new level of organization between the single neuron and the whole network, testable in biological recordings.

## Section-ordered backbone

**Introduction.** Humans are flexible: they can learn an "anti" version of a known task quickly. Compositional models say complex computations are built from reusable elementary parts (subtasks), but the neural substrate is unclear. Yang et al. 2019 showed that one RNN trained on 20 tasks develops clustered representations; Driscoll et al. ask *why* and *how*. They use dynamical-systems analysis (Sussillo & Barak 2013) — finding fixed points and locally linearizing — to read out the network's computational primitives.

**Single-task networks (Fig. 1).** A 200-unit softplus RNN with diagonal-initialized $W_{\text{rec}}$, L2 regularization, trained by BPTT on MemoryPro (remember stimulus direction, respond after delay). Fixed-point analysis reveals: (a) Context period — one stable fixed point + a ring of fixed points. (b) Stimulus period — stimulus-dependent fixed points organize an orthogonal stimulus representation. (c) Memory period — the ring becomes a **ring attractor** that holds the angle. (d) Response period — a new output-potent ring attractor emerges. **Input-interpolation** (continuously sweep input $u(\alpha) = (1-\alpha)u_{\text{memory}} + \alpha u_{\text{response}}$) shows fixed points slide smoothly from memory to response location.

**Two-task networks (Fig. 2).** Train one RNN on MemoryPro + MemoryAnti. Rule-input interpolation reveals **shared fixed points** across tasks: same context-period fixed point, two stable + one unstable fixed point in the stimulus period (the unstable one is the "anti" inversion mechanism), shared ring attractor in memory and response. The same ring is reused; only its orientation relative to $W_{\text{out}}$ changes.

**15-task networks (Fig. 3).** Define a **variance matrix** $V_{u, p}$ = variance of unit $u$ across stimulus conditions during task period $p$, normalized across all task periods. Hierarchical clustering shows block structure: clusters of units active for "delay-related" periods, "anti-related" periods, "modality-2-stimulus" periods, "categorization" periods, etc. Block structure is robust to architecture/hyperparameter choices (softplus vs tanh, 200 vs 1500 units, with/without noise). Trained networks correlate more strongly with each other than with untrained, confirming structure emerges from learning, not initialization. Examples: two category tasks share two-point attractors; multiple continuous-memory tasks share a ring attractor.

**Shared stimulus-period motifs (Fig. 4).** Tasks with similar stimulus computations (e.g., both category tasks) have initial conditions (end-of-context state) **close in state space** and trajectories at stimulus onset point in similar directions. Tasks without shared motifs (DelayAnti vs ReactPro) have separated initial conditions, evolve in non-overlapping subspaces, and show **bifurcations** during rule-input interpolation — the fixed point relevant to one task jumps discontinuously to the other task's fixed point.

**Lesion modularity (Fig. 5).** Zero the output of one unit cluster from the variance matrix. Effect: only tasks whose computation depends on that cluster's motif lose performance; others are intact. E.g., lesioning the "delay" cluster (cluster c) impairs all delay-period tasks but leaves reaction-timed tasks unaffected. The cause is geometric — the lesion destroys the relevant attractor, not just removes representational capacity.

**Fast transfer learning (Fig. 6).** After pre-training on 14 of 15 tasks, train *only* the one-hot rule-input weight (length $N_{\text{rec}}$ vector) for the held-out task (MemoryAnti). The network learns in a fraction of the original training. The minimal sufficient pre-training set is the union of all needed motifs: pre-training on {DelayAnti, MemoryPro} (which contain the "anti stimulus" and "delayed memory" motifs) is enough to learn MemoryAnti at full speed from just the rule input. Pre-training on tasks that *lack* the anti motif (two pro tasks) is much slower; no pre-training fails.

**Discussion.** The findings recommend a **two-stage lifelong-learning strategy**: (1) early, train recurrent weights to acquire dynamical motifs; (2) late, freeze recurrent weights and train new input vectors to compose motifs into new tasks. This avoids catastrophic forgetting. Findings are experimentally testable: the variance matrix can be computed from neural recordings (BOLD-like). Limitations: noise-corrupted but otherwise idealized inputs; no cell types; not biological learning rules.

## Phase 1 — Undergraduate-level synthesis

**The key idea.** Think of an RNN's activity as a ball rolling on a hilly landscape — fixed points are valleys (stable) or hilltops (unstable), attractors are basins. The shape of this landscape *is* the computation. Driscoll et al. show that when one RNN learns many tasks, it doesn't carve a new landscape for each task; it reuses **motifs** (ring valleys for circular memory, paired valleys for binary memory, hilltops for decision boundaries), and the task rule just shifts the ball's starting position so it rolls into the right basin.

**The experimental setup.** Train one 200-unit RNN on 15 standard cognitive tasks (memory of an angle, choose direction, anti-direction, categorize, integrate noisy evidence) using backpropagation through time. After training, freeze the network. For each task and each task period (context, stimulus, memory, response), find all the fixed points by minimizing $\|F(h, u) - h\|$. Examine the geometry: where do trajectories go? What attracts them? Then probe by smoothly interpolating one task's rule input toward another's and watch how fixed points morph. Finally, lesion clusters of units, and try transfer-learning new tasks by training only the rule input.

**The result.** (1) Tasks that share a subcomputation share the actual geometric structure that implements it — same ring attractor for any task needing continuous memory, same pair of attractors for any task needing binary memory. (2) Rule input acts like a switch that selects which motifs the network engages. (3) Lesions are surgically modular: kill the anti-cluster, lose only anti-tasks. (4) New tasks built from old motifs can be learned by training only ~200 rule-input weights, ~100× faster than full training. (5) This gives an experimentally falsifiable claim about brain modularity: brain regions or cell clusters should implement specific dynamical motifs reused across tasks.

## Phase 2 — Graduate-level deep dive

### 2.1 The RNN model (eqs. 1–3)

The continuous-time RNN is:

$$
\tau \frac{d\mathbf{h}}{dt} = -\mathbf{h}(t) + F\!\left(W_{\text{rec}}\, \mathbf{h}(t) + W_{\text{in}}\, \mathbf{u}(t) + \mathbf{b}_{\text{rec}}\right),
$$

with output

$$
\mathbf{z}(t) = W_{\text{out}}\, \mathbf{h}(t) + \mathbf{b}_{\text{out}},
$$

and softplus activation

$$
F(\mathbf{h}) = \ln\!\left(1 + \exp(\mathbf{h})\right).
$$

Inputs $\mathbf{u}(t) \in \mathbb{R}^{20}$ comprise (1) fixation (1d), (2) stimulus (4d = two 2d vectors of $A\sin\theta, A\cos\theta$), (3) rule (15d one-hot). Outputs $\mathbf{z}(t) \in \mathbb{R}^3$: fixation/response + 2d sin/cos response direction. $N_{\text{rec}} = 200$. Loss is squared error.

### 2.2 Fixed-point analysis (Sussillo & Barak 2013)

For frozen inputs $\mathbf{u}^* $, a fixed point $\mathbf{h}^*$ satisfies $\dot{\mathbf{h}} = 0$, i.e.,

$$
\mathbf{h}^* = F(W_{\text{rec}}\, \mathbf{h}^* + W_{\text{in}}\, \mathbf{u}^* + \mathbf{b}_{\text{rec}}).
$$

Found by minimizing the squared norm $q(\mathbf{h}) = \tfrac{1}{2}\,\| -\mathbf{h} + F(W_{\text{rec}}\mathbf{h} + W_{\text{in}}\mathbf{u}^* + \mathbf{b}_{\text{rec}})\|^2$ via gradient descent from many random initializations. Local dynamics around each fixed point are read from the Jacobian:

$$
J(\mathbf{h}^*) = \frac{1}{\tau}\Big[-I + \text{diag}\!\big(F'(W_{\text{rec}}\mathbf{h}^* + W_{\text{in}}\mathbf{u}^*)\big)\, W_{\text{rec}}\Big].
$$

Eigenvalues $\lambda$ of $J$ classify the fixed point: $\Re(\lambda) < 0$ all dimensions → stable (attractor); $\Re(\lambda) > 0$ some dimensions → unstable (saddle / decision boundary); $\Re(\lambda) \approx 0$ along a continuous manifold → line / ring attractor (slow modes that preserve a continuous variable).

### 2.3 Input interpolation

For two input conditions $\mathbf{u}_A$, $\mathbf{u}_B$:

$$
\mathbf{u}(\alpha) = (1 - \alpha)\, \mathbf{u}_A + \alpha\, \mathbf{u}_B, \quad \alpha \in [0, 1],
$$

fixed points are tracked as $\alpha$ varies. Smooth motion → shared motif; bifurcation (discontinuous jump) → different motifs.

### 2.4 The dynamical-motif catalog

| Motif | Geometric structure | Computation |
|---|---|---|
| Stable fixed point | Point attractor | Default state / context anchor |
| Ring of fixed points | $S^1$ continuous attractor | Memory of a circular variable (angle) |
| Two-point attractor pair | Two stable + one unstable saddle between | Binary categorization (cue $< \pi$ vs $> \pi$) |
| Unstable saddle | Hilltop with stable/unstable manifolds | Decision boundary |
| Rotation | Complex-conjugate eigenpair with $\Re(\lambda)$ near 0 | Continuous transformation (e.g., anti-inversion $\theta \mapsto \theta + \pi$) |
| Input amplification | Direction along which $W_{\text{in}}$ aligns with leading $J$-eigenvectors | Stimulus-driven trajectory |

A **ring attractor** is mathematically a 1-parameter family of fixed points $\{\mathbf{h}^*(\phi) : \phi \in [0, 2\pi)\}$ with Jacobian $J(\mathbf{h}^*(\phi))$ having exactly one eigenvalue at $\Re(\lambda) = 0$ (tangent to ring) and all others $\Re(\lambda) < 0$. Memorized angle $\phi$ is preserved against perturbations transverse to the ring.

### 2.5 Variance matrix and cluster identification

For unit $u$ and task period $p$:

$$
V_{u, p} = \frac{\text{Var}_{s}\!\left[ \overline{h_{u, t}(s, p)} \right]}{\max_{p'}\, \text{Var}_{s}\!\left[ \overline{h_{u, t}(s, p')} \right]},
$$

where $s$ indexes stimulus conditions, $\overline{h_{u,t}}$ is the unit's time-averaged activity within period $p$, and the normalization is across periods (column-wise). Hierarchical clustering of rows and columns of $V$ reveals modular block structure → units cluster by motif, task periods cluster by computation type. The matrix is the RNN analog of an fMRI activation map.

### 2.6 Lesion experiment

Set $\mathbf{h}_u(t) = 0$ for all $u \in \mathcal{C}$ (cluster) throughout the trial. Performance change on task $T$:

$$
\Delta P(T, \mathcal{C}) = P_{\text{lesioned}}(T) - P_{\text{intact}}(T).
$$

Modular result: $|\Delta P(T, \mathcal{C})|$ is large only for tasks whose dynamical motif uses cluster $\mathcal{C}$, near zero elsewhere. The cause is that removing units flattens the attractor manifold those units implemented; tasks not depending on that attractor are unaffected.

### 2.7 Fast transfer learning

The rule input enters as $W_{\text{in}}\, \mathbf{u}_{\text{rule}}$. The "rule weight vector" $\mathbf{w}_{\text{rule}} \in \mathbb{R}^{N_{\text{rec}}}$ is the column of $W_{\text{in}}$ associated with the one-hot rule index. After freezing $W_{\text{rec}}, W_{\text{out}}, W_{\text{in, stim}}, W_{\text{in, fix}}$, only $\mathbf{w}_{\text{rule}}$ is trained:

$$
\mathbf{w}_{\text{rule}}^{(t+1)} = \mathbf{w}_{\text{rule}}^{(t)} - \eta\, \nabla_{\mathbf{w}_{\text{rule}}} \mathcal{L}.
$$

Because the rule input enters additively through $W_{\text{in}}\mathbf{u}$, modifying $\mathbf{w}_{\text{rule}}$ effectively translates the network's operating point in input-bias space, which *selects* a fixed-point configuration without altering the *family* of fixed points (which is determined by $W_{\text{rec}}$). Transfer succeeds iff the required dynamical motifs already exist in $W_{\text{rec}}$ from prior training — a clean, falsifiable claim about what makes pre-training useful.

### 2.8 Connection to neuromodulation

Driscoll et al. do not invoke neuromodulators by name, but the rule-input mechanism is mathematically a *bias* that switches the network between motif configurations — the analog of a context-dependent modulator. The contrast with Tsuda 2021 is precise:

| Mechanism | What changes | Effect |
|---|---|---|
| Driscoll rule input | Input bias $W_{\text{in}}\mathbf{u}_{\text{rule}}$ | Selects fixed-point sub-configuration; motifs reused |
| Tsuda weight scaling | $W_{\text{rec}} \mapsto f_{nm}\,W_{\text{rec}}$ | Reshapes the entire fixed-point family / Jacobian eigenspectrum |
| Costacurta multiplicative gain $g$ | $W_{\text{rec}} \mapsto W_0 + \sum_k g_k W_k$ | Continuous low-rank reshaping of dynamics |

Both Driscoll's rule input and a neuromodulator pursue the same compositional goal — flexibility via a low-dimensional control input — but via different parametrizations.

## Connections to other corpus papers

- **`tsuda_2021_activity_hypertubes.md`** (this batch) — closest neighbor: Tsuda also studies one RNN solving multiple behaviors. Tsuda's "hypertubes" and Driscoll's "shared motifs" are complementary lenses on the same phenomenon, differing in mechanism (weight-scaling neuromodulator vs. rule-input bias).
- **`costacurta_2024_structured_flexibility.md`** (this batch) — explicit low-rank decomposition of $W_{\text{rec}}$ as a function of neuromodulatory inputs; provides a structured-flexibility account that subsumes Driscoll's input-bias mechanism.
- **`shine_2021_cellular_to_dynamics.md`** (this batch) — Driscoll's "rule input shifts fixed points" is the population-level analog of Shine's "neuromodulator shifts the gain function operating point". Shine provides the biophysical grounding for the input-bias / gain-shift abstraction.
- **`wainstein_2025_gain_perceptual_switches.md`** (this batch) — Wainstein's pupil-driven RNN gain modulation likewise reshapes fixed-point structure; both papers cite the same Sussillo dynamical-systems toolkit.
- **`yang_2019_task_representations.md`** (other batch, Yang et al. 2019, Nature Neurosci) — Driscoll's ref. 20; direct predecessor and the source of the 15-task suite. Driscoll provides the dynamical-systems explanation for Yang's empirical clustering.
- **`mante_2013_context_pfc.md`** (other batch) — Driscoll's ref. 10; Mante introduced the input-driven recurrent-dynamics framework Driscoll extends.
- **`mastrogiuseppe_ostojic_lowrank.md`** (other batch) — Driscoll's refs. 15, 29; low-rank RNN theory linking connectivity to computations, the formal basis for Costacurta 2024.
- **`duncker_driscoll_2020_organize_dynamics.md`** (other batch) — Driscoll's ref. 18; method for organizing RNN dynamics by task-computation for continual learning.
- **`russo_2018_motor_cortex_untangled.md`** (other batch) — Driscoll's ref. 16; cited as evidence that biological motor cortex also embeds untangled population responses, consistent with shared-motif organization.
- **`kudithipudi_2022_lifelong_learning.md`** (other batch) — Driscoll's two-stage learning hypothesis (slow recurrent → fast input) is exactly the lifelong-learning architecture Kudithipudi et al. argue for biologically.
