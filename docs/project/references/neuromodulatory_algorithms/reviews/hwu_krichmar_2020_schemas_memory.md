---
title: "A neural model of schemas and memory encoding"
authors: ["Tiffany Hwu", "Jeffrey L. Krichmar"]
year: 2020
venue: "Biological Cybernetics 114:169–186"
slug: hwu_krichmar_2020_schemas_memory
source_pdf: "sources/Hwu and Krichmar 2020 - A neural model of schemas and memory encoding.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

This paper builds a small brain-inspired neural network that learns spatial maps the way rats do in the classic Tse et al. (2007) experiment. In that experiment rats learn a "schema" — a layout of food wells in an arena, each well containing a specific flavor. Once a rat is familiar with the schema, swapping two of the wells for new food/flavor pairs is learned almost overnight, not after weeks. This is the hallmark of *schema-consistent learning*: new information fitted into an existing context is absorbed fast.

The authors model this with two cooperating processing streams. An **indexing stream** through the medial prefrontal cortex (mPFC) and hippocampus (HPC) — split into a dorsal HPC (dHPC) and a ventral HPC (vHPC) — produces sparse "index" patterns that say *which* context is active. A **representation stream** uses Contrastive Hebbian Learning (a biologically plausible alternative to backpropagation, where the network alternates between a "free" run and a "clamped" run with the target pinned, and updates weights based on the difference between the two) to learn the cue→action map. Two neuromodulator-like modules detect **novelty** (anti-Hebbian, weights *decay* on familiar stimuli) and **schema familiarity** (Hebbian + sigmoid, weights *grow* as a schema is repeated). Their product controls how many extra replay "epochs" the network does per day — the model's stand-in for hippocampal replay during quiet wake/sleep.

The headline claim: a network that gates the representation stream by hippocampal indices and *amplifies replay when novelty meets a familiar schema* reproduces Tse et al.'s rapid schema-consistent learning, the HPC lesion deficit, and avoids catastrophic forgetting across multiple schemas — without backprop and without storing past examples explicitly.

## Section-ordered backbone

### Abstract
Rats rapidly assimilate new paired associations (PAs) into a familiar spatial schema (Tse et al. 2007). The authors construct a biologically plausible neural network in which an **indexing stream** (mPFC, vHPC, dHPC) and a **representation stream** (cue, mPFC, AC, action) cooperate to learn context-dependent spatial tasks. A neuromodulator combining novelty and familiarity governs replay-driven rapid encoding. The model reproduces Tse et al.'s behavior, including HPC-lesion deficits, and offers a route to mitigate catastrophic forgetting in artificial networks.

### 1. Introduction
Frames the stability–plasticity dilemma and catastrophic forgetting. Reviews:
- **Complementary Learning Systems (CLS)** — fast hippocampal binding plus slow neocortical consolidation (McClelland et al. 1995; Kumaran et al. 2016).
- **Hippocampal indexing theory** (Teyler & DiScenna 1986) — HPC stores indices into neocortical activity patterns.
- **SLIMM framework** (van Kesteren et al. 2012) — for familiar schemas, mPFC inhibits HPC; for novel stimuli, HPC activates.
- **Eichenbaum (2017)** — theta synchronization between mPFC and medial temporal lobe is gated by the thalamic nucleus reuniens; vHPC encodes context, dHPC encodes specific episodes.
- **Neuromodulation** — basal forebrain (BF) drives attention/cortical processing; locus coeruleus (LC) drives novelty-triggered single-trial learning in HPC and modulates theta/gamma (Wagatsuma et al. 2018; Walling et al. 2011).

### 2. Methods and tasks
**2.1 Tse et al. (2007) summary.** Rats train 20 days on a 6-PA schema (a preexisting schema). On day 21, two wells are swapped for new PAs (new PAs within a preexisting schema), learned rapidly. HPC lesions block further learning of yet more new PAs but spare retention. A second, novel schema (Schema B) can be learned by controls but not by HPC-lesioned rats.

**2.2 Contrastive Hebbian Learning (CHL).** Multilayer network with bidirectional weights $W_k$ (forward) and $\gamma W_k^T$ (feedback). Three phases per epoch:
- **Free phase:** input fixed, network settles to $\check{x}_k$ (Eq. 1).
- **Clamped phase:** both input and output fixed, settles to $\hat{x}_k$.
- **Weight update:** $\Delta W_k = \eta_{CHL}(\hat{x}_k \hat{x}_{k-1}^T - \check{x}_k \check{x}_{k-1}^T)$ (Eq. 2).

CHL is biologically more plausible than backprop (Movellan 1991). The paper uses the symmetric-weights version for simplicity (Detorakis et al. 2018 offer asymmetric variants).

### 3. Neural model
**3.1 Indexing stream.** Context pattern (5×5 grid of food-well presence) → mPFC via $x_k(t)=f_k(W_k x_{k-1}(t-1))$ (Eq. 3) with ReLU $f(x)=\max(x,0)$ (Eq. 4); hard winner-take-all (WTA). mPFC weights trained by normalized Hebbian rule $\Delta W = \eta_{pattern} x_k x_{k-1}^T$ (Eq. 5) with $w_i = w_i/\lVert w\rVert$ (Eq. 6). vHPC indexes mPFC; dHPC indexes triplets of (context, cue, action) using larger $\eta_{indexing}$. Matches Eichenbaum's dorsal/ventral functional split.

**3.2 Representation stream.** Cue + mPFC → AC (association cortex, multimodal layer) → action. Bidirectional weights trained by CHL. vHPC gates AC via a static inhibitory weight matrix initialized to $w_{inh}$ everywhere, with a random fraction $P$ of weights zeroed (sparse holes in the inhibition pattern, fraction $P$ controlled by an experimenter parameter ∈ [0,1]). For a winning vHPC neuron, only the AC neurons sitting at its zeroed positions become active — schema-specific subsets of AC carry each task.

**3.3 Novelty and schema familiarity.** A neuromodulatory area has two submodules.
- **Novelty** (dHPC → submodule, weights $w_{novelty}$ initialized high): updates by anti-Hebbian rule $\Delta W = -\eta_{indexing}\, x_{novelty} x_{dHPC}^T$ (Eq. 7). Repeated exposures → long-term depression (LTD) → novelty drops.
- **Familiarity** (mPFC → submodule, weights $w_{fam}$ initialized near zero): updates by Hebbian rule (Eq. 5), transfer is a shifted sigmoid $f(x) = 1/(1+e^{-s(x-x_{shift})})$ (Eq. 8) — bimodal, so familiarity only "switches on" past a training threshold.
- **Neuromodulator** = product: $\text{neuromodulator} = x_{novelty} \cdot x_{familiarity}$ (Eq. 9). High only when both novelty and schema familiarity coincide.
- **Replay control**: number of training epochs per trial $= e_{default} + \text{neuromodulator} \cdot e_{boost}$ (Eq. 10).

**Trial structure.** Per trial: many epochs, each with an indexing phase, CHL free phase, CHL clamped phase. After $e_{settle}$ epochs the running max neuromodulator value sets total epochs via Eq. 10.

**3.4 Experimental design.** 20 simulated rats. 5×5 arena, 18 cue neurons, 10 mPFC, 40 AC, 5 vHPC, 40 dHPC. Parameters listed in Table 1 ($e_{boost}=1000$, $e_{default}=600$, $e_{settle}=20$, $w_{inh}=-10$, $w_{fam}=10^{-4}$, $w_{novelty}=1$, sigmoid gain $s=200$, $x_{shift}=0.03$, $P=0.3$). Replicates Tse et al. Experiments 1 and 2.

### 4. Results
**4.1 Experiment 1.** Schema A learned over 20 trials; on trial 21, two new PAs added — learned in a single trial (~200% dig-time ratio, matching Tse et al.). HPC-lesioned and control groups both retain Schema A and the first set of new PAs. When yet another pair is swapped on trial 23, controls keep learning but HPC-lesioned cannot — confirming the role of HPC indexing in driving the CHL clamping cycle.

**4.2 Experiment 2.** Schema B is learned by controls but not by HPC-lesioned rats. Schema A is preserved across the Schema B training. Returning to Schema A retrains quickly with no catastrophic forgetting.

**4.3 Neural activity.** mPFC neuron traces show one neuron consistently wins for Schema A and continues to potentiate; a different mPFC neuron wins for Schema B; returning to Schema A re-activates the original winner. vHPC neurons rise/fall together because they all read the winning mPFC neuron. dHPC switches winners every epoch (different triplet each time).

**4.4 Effects of neuromodulation.** Familiarity rises step-like (sigmoid). Novelty starts high, falls fast (anti-Hebbian LTD). Product spikes only on trial 1 (introduction) and trial 21 (new PAs in familiar schema). A control with flat epoch counts can also learn — but the neuromodulated network reaches equal performance with far fewer total epochs (~13,875 vs 16,800 / 33,600), saving training time.

**4.5 HPC sparsity ($P$) and forgetting.** $P=0$: no learning. $P=1$ (dense): each schema learned but immediately forgotten when a new one starts. Intermediate $P$ trades capacity vs. interference — $P=0.3$ is a sweet spot.

**4.6 mPFC size and forgetting.** Surprising result: with **non-overlapping** schemas, mPFC size = 1 produces zero catastrophic forgetting (all schemas share a single neuron because no information collides). With **overlapping** schemas, mPFC = 1 fails — multiple neurons are needed for distributed schema codes.

### 5. Discussion
Highlights:
- **5.1 Hippocampal indexing** modularizes representations and reduces overwrite.
- **5.2 Neuromodulation / novelty** — the LC is a candidate biological correlate; familiarity + novelty combination parallels Yu & Dayan (2005) on uncertainty.
- **5.3 mPFC–HPC interaction** — supports the cooperative (Preston & Eichenbaum 2013) over the SLIMM-style inhibitory account.
- **5.4 Spatial navigation** — links to dorsal–ventral place-cell specificity (Jung et al. 1994).
- **5.5 Future experimental tests** — LC/BF deactivation, severed mPFC–HPC connections, artificial gating patterns onto neocortex post-HPC-lesion.
- **5.6 ML applications** — context-based gating as a continual-learning trick; the group reports applying the architecture to a household-robot scene understanding task.

## Phase 1 — Undergraduate-level synthesis

**Key idea.** The brain doesn't relearn from scratch every time you put your keys in a new place — it slots the new information into a *schema* (a familiar context like "my apartment") so learning is fast. The authors model that with a small artificial brain. Two ideas drive the design:

1. **Indexing** — the hippocampus produces a small label ("this is Schema A") that tells the rest of the network which subset of neurons should be active for this context. Like a Post-it tag on a memory.
2. **Neuromodulated replay** — a single chemical signal (their stand-in for noradrenaline / acetylcholine) goes up *only* when something *new* happens *inside a familiar context*. When it goes up, the network internally rehearses the new information many more times that day — modeling hippocampal replay during quiet wakefulness and sleep.

**Setup.** A 5×5 grid arena. Each "food well" is a (location, flavor) pair. Train 20 days on six wells (Schema A). On day 21, swap two wells for new wells. On day 22, surgically remove the hippocampus in half the simulated rats. On day 23 swap two more wells. Then introduce a totally new arena (Schema B). Test by cueing the network with a flavor and reading off the action neuron that fires most strongly — its activity proportion stands in for the rat's dig time at that well.

**Result.**
- The intact network learns Schema A gradually (matching rats), then absorbs the new wells in *one* trial when the schema is already familiar.
- HPC-lesioned networks keep what they already knew but cannot encode further new information.
- Three sequential schemas are learned without catastrophic forgetting *if* the hippocampal-to-cortex connections are sparse (intermediate $P$); dense connections cause overwrite.
- The neuromodulator saves training time: the same final performance reached in roughly 60–80% of the total epochs a flat-replay baseline needs.

**Concrete instantiation.** Imagine the network's mPFC as a row of 10 light bulbs. Schema A trains a single bulb to glow brightly whenever the kitchen layout is shown. When the kitchen layout changes slightly (two wells swapped), the same bulb stays on but dims a bit, *and the novelty module screams*. The product of "this is the kitchen, I know it" × "but something is new here!" turns the replay knob up to ~10×, and overnight the network rehearses the changed wells until they fit. When the layout changes wholesale (Schema B), a different bulb takes over — and because the hippocampus gates a separate subset of the AC for that bulb, learning Schema B doesn't disturb Schema A.

## Phase 2 — Graduate-level deep dive

### Core dynamics

The network alternates between three phases per epoch. Let layer $k$ have activations $x_k(t) \in \mathbb{R}^{n_k}$, forward weight matrix $W_k \in \mathbb{R}^{n_k \times n_{k-1}}$, and feedback scaling $\gamma$.

**Free phase recurrence** (input clamped, network relaxes):

$$
x_k(t) = f_k\!\left(W_k\, x_{k-1}(t-1) + \gamma\, W_{k+1}^T\, x_{k+1}(t-1)\right), \qquad k=1,\dots,L.
$$

Iterated for $T_s$ steps to fixed point $\check{x}_k$.

**Clamped phase recurrence** (input and target both fixed); produces fixed point $\hat{x}_k$.

**Contrastive Hebbian update** (Eq. 2):

$$
\Delta W_k = \eta_{CHL}\!\left(\hat{x}_k \hat{x}_{k-1}^T - \check{x}_k \check{x}_{k-1}^T\right), \qquad k = 1,\dots,L.
$$

**Equivalence with backpropagation.** Movellan (1991) showed that for small $\gamma$ in a continuous Hopfield network, the CHL update is a finite-difference estimate of $-\partial E / \partial W_k$ for the squared-error energy. Sketch: the clamped fixed point $\hat{x}$ satisfies $\partial E_{clamped}/\partial x_k = 0$; the free fixed point $\check{x}$ satisfies $\partial E_{free}/\partial x_k = 0$; the difference of outer products $\hat{x}_k \hat{x}_{k-1}^T - \check{x}_k \check{x}_{k-1}^T$ in the limit $\gamma\to 0^+$ is proportional to the gradient of the loss with respect to $W_k$, so CHL approximates backprop without requiring an explicit error signal traveling backwards along one-way synapses. This makes the model biologically plausible while retaining gradient-style learning.

### Indexing stream

Pure feedforward (no feedback) within the indexing stream:

$$
x_k(t) = f_k\!\left(W_k\, x_{k-1}(t-1)\right), \qquad f_k(x) = \max(x, 0).
$$

A hard winner-take-all picks $i^\star = \arg\max_i x_k^{(i)}$ and sets all other components to zero. Weight update is **normalized Hebbian**:

$$
\Delta W_k = \eta_{pattern}\, x_k\, x_{k-1}^T, \qquad w_i \leftarrow \frac{w_i}{\lVert w\rVert_2}.
$$

The normalization (Eq. 6) prevents unbounded weight growth and implements a form of competitive learning: only the winning neuron's incoming weights are updated and renormalized. In steady state each mPFC neuron's weight vector $w_i$ converges to the centroid of the context-pattern cluster it wins on.

### Sparse gating from vHPC to AC

For each simulated rat, the static inhibitory matrix $M^{inh} \in \mathbb{R}^{n_{AC} \times n_{vHPC}}$ is built once at initialization:

$$
M^{inh}_{ij} = \begin{cases} 0 & \text{with probability } P \\ w_{inh} & \text{with probability } 1-P \end{cases}, \qquad w_{inh} = -10.
$$

When vHPC neuron $j$ wins (its activation is large positive; others are zero by WTA), AC neuron $i$ receives effective input $\approx M^{inh}_{ij} \cdot x_{vHPC}^{(j)}$. If $M^{inh}_{ij} = w_{inh}$, AC neuron $i$ is strongly suppressed; if $M^{inh}_{ij} = 0$, AC neuron $i$ is free to be driven by the upstream representation stream. Different winning vHPC neurons gate disjoint random subsets of AC, implementing context-conditioned routing without ever changing $M^{inh}$.

### Neuromodulator dynamics

**Novelty submodule** (dHPC → novelty). Initial weight $w_{novelty} = 1$ (high). Activity computed feedforward (Eq. 3) then anti-Hebbian:

$$
\Delta W_{novelty} = -\eta_{indexing}\, x_{novelty}\, x_{dHPC}^T.
$$

After $n$ repeated co-activations of a fixed dHPC triplet, the weight $w$ obeys roughly $w(n+1) = w(n) - \eta_{indexing}\, x_{novelty}\, x_{dHPC}^{(j)}$; iterated, $w(n) \to 0$ exponentially (LTD).

**Familiarity submodule** (mPFC → familiarity). Initial weight $w_{fam} \approx 10^{-4}$ (near zero). Hebbian update (Eq. 5) drives weight up monotonically with repeated wins; the transfer is the shifted sigmoid:

$$
f(x) = \frac{1}{1+\exp\!\left(-s\,(x - x_{shift})\right)}, \qquad s=200,\ x_{shift}=0.03.
$$

With such a high gain $s$, the sigmoid is a near-step function with switch point at $x_{shift}$. Familiarity behaves bimodally: ≈0 until enough Hebbian potentiation pushes the input past 0.03, then ≈1 thereafter.

**Combined neuromodulator** (Eq. 9):

$$
\nu = x_{novelty} \cdot x_{familiarity}.
$$

This is **AND-like in the high regime**: $\nu \approx 1$ requires *both* an unfamiliar dHPC index *and* a familiar schema. New environment (Schema B introduction): $x_{familiarity}=0$, so $\nu \approx 0$. Stale familiar environment (no new PAs): $x_{novelty} \approx 0$, so $\nu \approx 0$. The exception window — new PAs slotted into an existing schema — is the only regime where both are high simultaneously.

**Replay control** (Eq. 10):

$$
\text{epochs} = e_{default} + \nu_{\max}\cdot e_{boost},
$$

where $\nu_{\max}$ is the running maximum of $\nu$ during the first $e_{settle}$ epochs of the trial. With $e_{default}=600$, $e_{boost}=1000$, peak replay tops out at 1600 epochs.

### Why intermediate vHPC→AC sparsity ($P$) works

Let $S$ be the number of schemas to be encoded and $n_{AC}=40$ the AC layer width. Each winning vHPC neuron unmasks $\approx P\cdot n_{AC}$ AC neurons. For $S$ schemas to be separable, expected overlap of unmasked sets between any two schemas is roughly $P^2\cdot n_{AC}$, so:

- $P=0$: no AC neurons free $\Rightarrow$ no learning.
- $P=1$: all AC neurons free for every schema $\Rightarrow$ full overlap, full interference, full catastrophic forgetting.
- $P=0.3$ with $n_{AC}=40$: each schema gets ~12 AC neurons, pairwise overlap ~3.6 — enough representational capacity per schema, enough disjointness across schemas.

This is the network's analog of Cover's hyperplane-separability argument adapted to gating sparsity.

### Why an mPFC of size 1 paradoxically avoids forgetting for non-overlapping schemas

With $n_{mPFC}=1$, all schemas share the same mPFC neuron, hence the same vHPC neuron (WTA on a single neuron is trivial), hence the same fixed AC mask. If the schemas have disjoint cues and disjoint actions, the AC and action layers' CHL-trained weights occupy disjoint cue–action subspaces and can coexist without interference. If schemas share cues but differ in target actions, the same AC neurons must learn contradictory targets and the network collapses — explaining the overlap-vs-non-overlap result in §4.6.

### Parameter summary (from Table 1)

| Symbol | Value | Meaning |
|---|---|---|
| $T_s$ | 5 | Settling time steps per phase |
| $\eta_{indexing}$ | 0.1 | Learning rate for vHPC, dHPC, novelty |
| $\eta_{pattern}$ | $10^{-4}$ | Learning rate for mPFC, familiarity (slow consolidation) |
| $\eta_{CHL}$ | $10^{-3}$ | CHL learning rate for representation stream |
| $\gamma$ | $10^{-3}$ | Feedback scaling in CHL recurrence |
| $e_{boost}$ | 1000 | Maximum replay-epoch boost |
| $e_{default}$ | 600 | Baseline replay epochs per trial |
| $e_{settle}$ | 20 | Epochs over which $\nu_{\max}$ is recorded |
| $w_{inh}$ | $-10$ | vHPC→AC inhibitory weight |
| $w_{fam}$ | $10^{-4}$ | Initial mPFC→familiarity weight |
| $w_{novelty}$ | 1 | Initial dHPC→novelty weight |
| $s$ | 200 | Sigmoid gain on familiarity |
| $x_{shift}$ | 0.03 | Sigmoid threshold |
| $P$ | 0.3 | Fraction of vHPC→AC inhibitory weights zeroed |

## Connections

**This paper relates to:**

- **[doya_2002_metalearning_neuromodulation](doya_2002_metalearning_neuromodulation.md)** — Doya's framework assigns one neuromodulator per RL meta-parameter (DA = TD error, ACh = memory time scale, NE = exploration, 5-HT = discounting). Hwu & Krichmar's "neuromodulator" is closest to Doya's NE/ACh axis: it gates *replay rate* and *encoding strength* based on novelty + familiarity. They do not adopt Doya's one-to-one chemical mapping but cite the same Krichmar (2008) framework.
- **[krichmar_hwu_2022_design_principles_neurorobotics](krichmar_hwu_2022_design_principles_neurorobotics.md)** — companion design-principles paper from the same group; Hwu & Krichmar 2020 is one of the worked examples cited for "principle: neuromodulation for context-dependent learning".
- **[alonso_krichmar_2023_sparse_quantized_hopfield](alonso_krichmar_2023_sparse_quantized_hopfield.md)** — different mechanism but same continual-learning agenda: SQHN also uses sparsity (winner-take-all + quantization) to avoid catastrophic forgetting in an online setting.
- **[zou_2020_neuromodulated_attention](zou_2020_neuromodulated_attention.md)** — Zou et al. (with Krichmar as senior author) extend the novelty/familiarity gating idea to attention in uncertain perceptual domains; Hwu & Krichmar acknowledge Xinyun Zou in the Acknowledgements as collaborator on the precursor Telluride 2017 project.
- **[xing_2020_neuromodulated_patience](xing_2020_neuromodulated_patience.md)** and **[xing_2022_neuromodulation_rl_environment_changes](xing_2022_neuromodulation_rl_environment_changes.md)** — sister-lab papers on neuromodulated RL meta-parameters; share the "Krichmar 2008 framework" citation.

**Forward citations expected in this corpus.** Later papers in the Krichmar program — including Krichmar & Hwu 2022, Alonso & Krichmar 2023, and Espino et al. 2024 — reference this work as the canonical demonstration that mPFC-HPC indexing plus novelty/familiarity neuromodulation reproduces Tse et al. (2007). Cross-corpus papers on lifelong RL (Lee et al. 2024; Ben-Iwhiwhu et al. 2022; Vecoven et al. 2020) sit in the same "neuromodulation against catastrophic forgetting" cluster but use the Doya-style RL meta-parameter framing rather than CHL.

**Foundational priors cited but outside this corpus.** McClelland et al. 1995 (CLS), Teyler & DiScenna 1986 (indexing theory), Tse et al. 2007 (target experiment), van Kesteren et al. 2012 (SLIMM), Eichenbaum 2017 (mPFC–HPC theta), Yu & Dayan 2005 (uncertainty + neuromodulation), Aston-Jones & Cohen 2005 (LC adaptive gain), Movellan 1991 (CHL).
