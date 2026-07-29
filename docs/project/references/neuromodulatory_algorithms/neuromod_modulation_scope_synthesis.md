---
title: "Modulation Scope and Granularity — Cross-Paper Synthesis (neuromodulatory_algorithms corpus)"
topic: neuromodulatory_algorithms
status: curated
created: 2026-07-28
last_updated: 2026-07-28
papers_synthesised: 41
related:
  - neuromodulatory_algorithms_lit_review.md
  - neuromodulatory_algorithms_synthesis.md
  - neuromodulatory_algorithms_predictions_synthesis.md
  - ../FiLM/film_modulation_granularity_synthesis.md
scope: |
  Cross-paper synthesis over the 41-paper computational-neuromodulation corpus,
  answering one architectural question: at what scope, and at which sites, does
  the biological/algorithmic neuromodulation literature actually apply its
  modulatory signal, and how (if at all) does it group the modulated units?
  Uses the taxonomy defined in ../FiLM/film_modulation_granularity_synthesis.md
  so the two halves compose. Adds two corpus-specific refinements (support vs.
  resolution; what is modulated) that the FiLM half does not need.
---

# Modulation Scope and Granularity — Cross-Paper Synthesis (neuromodulation corpus)

## 1. Plain-English entry point

Some neural networks contain a small side-network — a "modulator" — whose job is not
to answer the question but to **change how the rest of the network computes**: turn
some units' responsiveness up, others down, make the network more or less decisive.
This document asks a purely mechanical question of the 41 papers we have collected on
brain-inspired modulation: **how many separate modulation numbers exist, and which
units does each one reach?** One number for the entire network? One per individual
neuron? One shared by a block of neurons?

A companion document asked the same question of a different 31-paper corpus — the
"FiLM" and hypernetwork literature from mainstream deep learning, where a side-network
emits a *gain* (multiply) and an *offset* (add) for each feature. That corpus answered
almost unanimously: **one gain and one offset per individual feature or neuron**, at
many layers.

**This corpus answers the opposite way, and that is the headline.** The neuroscience
literature's default is a **single global number for the whole network** — one
"dopamine level", one "arousal level", one "gain" — because that is what a real
neuromodulator physically is: a chemical sprayed into a wide volume of tissue that
every neuron in reach is bathed in. Roughly two-thirds of the papers here that build a
working modulator use exactly one scalar. Where the field *does* get selective, it is
almost never by giving each neuron its own number; it is by choosing **which
population** the single number reaches (only the excitatory cells, only the output
layer, only one 10 % patch), or by letting each neuron carry its own **learned
sensitivity** — a "receptor" — so that one broadcast number lands differently on
different cells.

Our own agent's modulator sits **between the two literatures**. Its shared 16-unit
recurrent side-network is conventional in both. Its habit of emitting a whole
128-number vector is normal for FiLM and unusual for neuroscience. Its practice of
tying blocks of neurons to one shared number (the `grouping_size` sweep) turns out to
have exactly **one** precedent in the whole reference library — AlKilany & Goodman
2025, in this corpus — and, correcting the FiLM half, that precedent matches our design
*more* closely than previously recorded: they too keep a per-neuron learned baseline
and group only the dynamic part, and they say in the methods why. Details, tables, and
four explicit judgments follow.

---

## 2. Taxonomy, plus two refinements this corpus forces

Axis labels are identical to
[`../FiLM/film_modulation_granularity_synthesis.md` §2](../FiLM/film_modulation_granularity_synthesis.md),
so the two halves compose.

**Axis 1 — Granularity.** How many distinct modulation values exist per modulated
tensor.

| Cell | Name | Definition |
|---|---|---|
| **(a)** | Node-wise / per-unit | one value per individual neuron / activation |
| **(b)** | Per-channel | one value per conv feature map (collapses to (a) in dense layers) |
| **(c)** | Grouped | one value shared across a *group* of units; anything strictly between (a)/(b) and (e) |
| **(d)** | Layer-scalar | a single value for an entire layer |
| **(e)** | Network-wise / global | one value for the whole network |

**Axis 2 — Placement / sites.** Which layers or populations are modulated, how many
injection points, early vs. late, inside recurrent machinery or not.

**Axis 3 — Parameterisation across sites.** One shared generator with per-site heads,
vs. a separate generator per site, vs. one signal broadcast unchanged everywhere.
"Layer-wise" in the Axis-3 sense ("each layer has its own parameter set") is
distinguished throughout from Axis 1d ("one scalar per layer").

**Axis 4 — Grouping / sharing schemes.** Explicit grouping, tying, cell-type or
population partitioning, low-rank factorisation. Grouping unit, how the size is chosen,
the stated motivation, and any empirical ablation.

### 2.1 Refinement 1 — *support* is not *resolution*

The FiLM corpus never needed this distinction because in that corpus every modulated
layer is modulated in full. This corpus does need it, because its papers vary two
different things and both get called "targeted neuromodulation":

- **Resolution** — how many *distinct values* the modulation signal carries. This is
  Axis 1. AlKilany & Goodman's group size $G$ varies resolution: at $G{=}1$ there are
  200 distinct values, at $G{=}200$ there is one.
- **Support** — *which subset of units* the signal reaches at all, with the untouched
  units left at baseline. Tsuda et al. 2021's "10 % subpopulation" varies support: the
  resolution is always one scalar; what changes is how many neurons feel it.

Both look like "narrowing the modulation", and the corpus's two headline scope
ablations sit on *different* one of these axes. Conflating them produces a false
consensus; §6.3 keeps them apart.

### 2.2 Refinement 2 — *what* is modulated varies, and scope co-varies with it

The FiLM corpus modulates one thing: layer activations. This corpus modulates at
least seven distinct target types — activations, recurrent weights, intrinsic
biophysical parameters, learning rates, exploration temperature, plasticity gates, and
graph/structure hyperparameters — and the answer to "what granularity?" is largely
determined by *which* of those is being modulated. A learning rate is a scalar because
learning rates are scalars, not because the field made a granularity choice.
§7 tabulates this explicitly; it is the single most important thing to hold in mind
when reading the counts below.

---

## 3. Axis 1 — Granularity, per paper

Papers are grouped by cell. "Off-axis" marks papers whose modulator is indexed by
something other than units (tasks, goals, modes).

### 3.1 Cell (e) — one global scalar for the whole network

| Paper | The signal | What it multiplies / sets | Evidence |
|---|---|---|---|
| [Tsuda et al. 2021](reviews/tsuda_2021_activity_hypertubes.md) | one scalar $f_{nm}$ | **every recurrent weight**: $\tilde W = f_{nm} W$ | $N=200$ Dale's-law RNN; $f_{nm}\in[0.5,9]$; the corpus's minimal modulator |
| [Wainstein et al. 2025](reviews/wainstein_2025_gain_perceptual_switches.md) | one scalar $g(t)$ | sigmoid **slope** of every unit: $r_t = 1/(1+e^{-g(t)x_t})$ | "Gain $g$ enters everywhere uniformly (single scalar across the network)" |
| [Rodriguez-Garcia et al. 2026](reviews/rodriguezgarcia_2026_ne_stability_gap.md) | one scalar $g$ | incoming weights of each gain-modulated neuron, $\hat W = gW$ | **See §3.5 caveat** — notated per-neuron, driven by a batch-scalar entropy, therefore numerically global |
| [Osman et al. 2024](reviews/osman_2024_hopfield_arousal.md) | one scalar $\alpha$ | divides the whole recurrent matrix: $\tfrac{1}{\alpha}My$ | bifurcation at $\alpha^\star=\lambda_{\max}(M)$ requires $\alpha$ be a scalar |
| [Cox & Krichmar 2009](reviews/cox_krichmar_2009_neuromodulation_robot_controller.md) | one scalar $nm(t) = 10\cdot\overline{(\text{BF}+\text{Raphe}+\text{VTA})}$ | **extrinsic and inhibitory** synaptic inputs only; intrinsic excitatory left at $nm{=}1$ | an Axis-4 partition by *connection type*, not by unit — see §6.2 |
| [Krichmar 2013](reviews/krichmar_2013_neurorobotic_anxiety_curiosity.md) | one scalar per modulator (tonic DA, tonic 5-HT, summed ACh/NE) | tonic term added to target neurons' input current; ACh/NE multiplies frontal lateral inhibition + modulator→frontal weights | modulators have *no* per-unit index; the only structure is which projection each targets |
| [Xing et al. 2020](reviews/xing_2020_neuromodulated_patience.md) | one 5-HT scalar | patience term in a Bayesian wait/quit rule | "a single scalar 5-HT patience parameter" |
| [Doya 2002](reviews/doya_2002_metalearning_neuromodulation.md) | four scalars $\alpha,\beta,\gamma,\delta$ | learning rate / inverse temperature / discount / TD error | explicitly framed as *global* meta-parameters |
| [Lee et al. 2024](reviews/lee_2024_lifelong_rl.md) | $\alpha(s,a) = E/(E{+}A)$, $\beta(s)=1/\langle E\rangle$ | learning rate, softmax temperature | state-dependent but not unit-indexed |
| [Wang et al. 2025 NEST](reviews/wang_2025_nest_hypergraph.md) | two scalars $\alpha,\beta$ from two small MLPs | hypergraph clustering threshold, shortcut probability | the corpus's lightest modulator |
| [L'Haridon & Cañamero 2023](reviews/lharidon_canamero_2023_stress_pain.md) | one cortisol scalar | multiplies damage: $\text{pain} = c_{\text{cortisol}}\cdot\text{damage}$ | closest construct analogue to this project |
| [Cañamero 1997](reviews/canamero_1997_motivations_emotions.md) and school ([Blanchard 2006](reviews/blanchard_canamero_2006_affect_modulated.md), [Cos 2010](reviews/cos_2010_affordances_consummatory.md), [Lones 2013](reviews/lones_canamero_2013_epigenetic_hormones.md)/[2018](reviews/lones_2018_hormone_epigenetic.md), [Lewis 2016](reviews/lewis_canamero_2016_hedonic_pleasure.md), [Khan 2022](reviews/khan_canamero_2022_social_buffering.md)) | one scalar concentration per hormone | perceived physiological state $\tilde x_i = x_i - \kappa_{i,k}h_k$; ART-1 vigilance $\rho$; behaviour intensity; drive gains | scalar hormone × **per-target learned sensitivity** — see §6.2, the corpus's closest thing to a "receptor" scheme |
| [Hwu & Krichmar 2020](reviews/hwu_krichmar_2020_schemas_memory.md) | one scalar $\nu = x_{\text{novelty}}\cdot x_{\text{familiarity}}$ | number of replay epochs, $e_{\text{default}} + \nu e_{\text{boost}}$ | modulates a *training-schedule* quantity, so a scalar is forced |
| [Alonso & Krichmar 2023](reviews/alonso_krichmar_2023_sparse_quantized_hopfield.md) | one decaying scalar $\epsilon = \alpha/(t{+}\alpha)$ | neuron-growth threshold | a global schedule, not a per-unit signal |

### 3.2 Cell (a) — per-unit vector

| Paper | The signal | Generator output | Note |
|---|---|---|---|
| [Vecoven et al. 2020](reviews/vecoven_2020_neuromod_dnn.md) | effective slope $z^\top w_s$ and bias $z^\top w_b$, **per neuron** | a **$k$-dim global vector $z$**, shared across all neurons | Cell (a) *signal* carried by a cell-(e) *generator*, decoded by per-neuron learned weights $(w_s, w_b)$. Structurally a **rank-$k$ factorisation** of a per-neuron $(\gamma,\beta)$. The single most important architecture in this corpus for our purposes — see §5.1 |
| [Ben-Iwhiwhu et al. 2022](reviews/beniwhiwhu_2022_context_meta_rl.md) | $h_m = \tanh(W_m\,\mathrm{ReLU}(W_g x))$, element-wise | one modulator population **per layer** | $h = \mathrm{ReLU}(h_s \otimes h_m)$; per-unit gain including sign flip |
| [Tambaş et al. 2025](reviews/tambas_2025_krotov_hopfield_rbm.md) | ternary $g_\nu(I)\in\{+1,-\Delta,0\}$ per hidden unit | rank ordering over the whole layer | per-unit *output*, global *computation*; modulates plasticity, not activation |
| [AlKilany & Goodman 2025](reviews/alkilany_goodman_2025_snn_dynamic_sensory.md) at $G{=}1$ | five biophysical parameters per LIF neuron | one MLP (or SNN) | the $G{=}1$ end of the corpus's only granularity sweep |

### 3.3 Cell (c) — grouped / factorised

| Paper | Grouping unit | Range | Status |
|---|---|---|---|
| **[AlKilany & Goodman 2025](reviews/alkilany_goodman_2025_snn_dynamic_sensory.md)** | contiguous block of $G$ neurons share one modulator output | $G \in \{1,2,5,10,20,50,100,200\}$ over a 200-unit hidden layer | **implemented and swept** — the corpus's only true Axis-1 grouping |
| [Costacurta et al. 2024](reviews/costacurta_2024_structured_flexibility.md) | rank-1 *modes* of a low-rank recurrent matrix, $W_x(t)=\sum_k s_k(z(t))\,\ell_k r_k^\top$ | $K \in \{1,3\}$ over $N\approx100$ units | implemented; grouping in a **learned basis**, not in unit index |
| [Mei et al. 2022](reviews/mei_2022_multiscale_neuromod.md) Scale 2 | cell-type populations $\mathcal{P}_1,\dots,\mathcal{P}_K$, $h_j = g_k(\mathbf m_t)\sigma(W_j x + b_j)$ for $j\in\mathcal{P}_k$ | — | **proposal only**; the field's clearest statement that grouped modulation is the biologically right form |
| [Ferguson & Cardin 2020](reviews/ferguson_cardin_2020_gain_modulation.md) | PV⁺ / SST⁺ / VIP⁺ interneuron classes; cortical layers | — | biology, not an algorithm: receptor expression *is* the grouping |
| [Shine et al. 2021](reviews/shine_2021_cellular_to_dynamics.md) | cortical layer × receptor subtype ($\alpha_1$ in layers I–III, $\alpha_2$ in II–IV, muscarinic infragranular, nicotinic granular) | — | biology; the diffuse signal is decoded into a layer-partitioned effect |

### 3.4 Finer than (a), and off-axis

| Paper | Granularity | Target |
|---|---|---|
| [Wang et al. 2024 NeuronML](reviews/wang_2024_neuromod_meta.md) | **per-weight** mask $M\in[0,1]^d$, $\theta_M = M\odot\theta$ | network structure (which units are on per task) |
| [Kolouri et al. 2019](reviews/kolouri_2019_attention_plasticity.md) | **per-synapse** importance $\gamma_{ji}^l$ | plasticity (how freely each weight may change) |
| [Zou et al. 2020](reviews/zou_2020_neuromodulated_attention.md) | ACh is a $K{=}4$ vector over **goals**; NE a scalar | goal prior + reset |
| [Xing et al. 2022](reviews/xing_2022_neuromodulation_rl_environment_changes.md) | ACh is a $K$-vector over **stored tasks**; NE a scalar | policy-slot selection |
| [Driscoll et al. 2022](reviews/driscoll_2022_dynamical_motifs.md) | rule-input weight vector $\mathbf w_{\text{rule}}\in\mathbb{R}^{N_{\text{rec}}}$ | additive input bias, not a modulator per se |
| [Espino et al. 2024](reviews/espino_2024_snn_path_planning.md) | per-edge axonal delays $D_{ij}$; neuromodulation flagged as *future work* | path cost |

Reviews with no modulator of their own — [Avery & Krichmar 2017](reviews/avery_krichmar_2017_models_neuromodulation.md), [Krichmar & Hwu 2022](reviews/krichmar_hwu_2022_design_principles_neurorobotics.md), [Kudithipudi et al. 2022](reviews/kudithipudi_2022_lifelong_learning.md), [Durstewitz et al. 2025](reviews/durstewitz_2025_neuroscience_continual_learning.md), [Chiba & Krichmar 2020](reviews/chiba_krichmar_2020_self_monitoring.md), [Cañamero 2005](reviews/canamero_2005_emotion_understanding.md), [Scarinzi & Cañamero 2022](reviews/scarinzi_canamero_2022_affective_interactions.md) — are excluded from the counts.

### 3.5 A load-bearing caveat: Rodriguez-Garcia's gain is a scalar, not a vector

The per-paper review describes NGM-SGD as maintaining "a per-neuron gain scalar
$g_i(t)$", and the paper's own prose says the gain "multiplicatively scales the
incoming weights of each output neuron $i$". But the update rule (paper Eq. 4, and
Algorithm 1 line 9) is

$$
g \leftarrow \gamma g + (1-\gamma) g_0 + \eta H, \qquad
H = -\tfrac{1}{B}\sum_{j=1}^{B}\sum_{l} \pi_{j,l}\log \pi_{j,l},
$$

where $H$ is the **batch-mean predictive entropy — a single number**. Every $g_i$ is
initialised identically and driven by the identical forcing term, so they remain equal
for all time. The theory section confirms it: the reparameterisation is
$\Phi(W)=GW$ with $G = gI$, "isotropic". **NGM-SGD is cell (e).** It is written in
per-neuron notation because the *biological story* is per-neuron; the *implementation*
is one global scalar. This distinction matters for us because NGM-SGD is otherwise the
corpus's closest algorithmic cousin to our $\gamma$ head.

### 3.6 Verdict on Axis 1

**The field's convention is cell (e), a single global scalar — the exact opposite of
the FiLM corpus's per-channel/per-unit convention.** Of the ~24 corpus papers that
instantiate a modulator with definable granularity, **15 use one global scalar (or one
scalar per named modulator)**; **4 emit a per-unit vector**; **2 implement grouped or
factorised modulation**; **2 go finer than per-unit** (per-weight); the remainder are
indexed by tasks or goals rather than by units.

Two things follow.

1. **The biological framing does what you would expect.** Papers that take the anatomy
   seriously — diffuse projections, volume transmission, a nucleus of a few thousand
   cells innervating all of cortex — converge on one number. Mei et al. 2022 name this
   explicitly: Vecoven-style broadcast modulation "is closer to *volume* signalling
   (the modulator $z$ broadcasts to all neurons)", in contrast to synapse-targeted
   modulation.
2. **But almost every paper then reintroduces selectivity somewhere else**, and this is
   the corpus's real structural signature. The global scalar is decoded locally: by
   per-neuron learned receptor weights (Vecoven), by receptor-subtype and cell-class
   expression (Ferguson & Cardin, Shine), by which projection each nucleus targets
   (Krichmar 2013), by connection type (Cox & Krichmar), or by which subpopulation is
   modulated at all (Tsuda). **"Global broadcast + local decoding" is the neuro
   convention, not "one number does one thing everywhere".**

**Cell (c) proper — a contiguous block of units tied to one shared dynamic value — is
attested exactly once.** A regex scan of all 41 source PDFs for grouping vocabulary
(`group size`, `grouping size`, `spatial grouping`, `group of G neurons`, `groups of
neurons`, `shared across neurons/units/channels`, `granularit*`, `per-neuron`,
`subpopulation`, `cell-type specific`) returns `spatial grouping` / `group size` /
`grouping size` **only in AlKilany & Goodman 2025**. (Khan & Cañamero 2022's four
`group size` hits refer to the size of the *agent society*, not a neuron group.) The
biological analogue — `subpopulation` and `cell-type specific` — is by contrast
well attested: 51 hits in Tsuda, 7 in Mei, 3 in Ferguson & Cardin, plus Costacurta,
Driscoll, Shine, and Lee.

**Cell (d), the layer-scalar, is unattested here too**, exactly as in the FiLM corpus.
No paper in either corpus emits one number per layer as its modulation signal.

---

## 4. Axis 2 — Placement and number of sites

| Paper | Sites | Where | Sweep / ablation? |
|---|---|---|---|
| Vecoven et al. 2020 | **every hidden neuron of both actor and critic** | activation function of every layer (saturated ReLU), output layer linear | depth swept (0/1/4 hidden layers): NMN roughly insensitive to depth, RNN baseline is not |
| Ben-Iwhiwhu et al. 2022 | **every fully-connected layer** of the policy | pre-activation, before the ReLU | actor only; critic modulation tried and found *unstable* in PEARL — a genuine placement finding |
| Rodriguez-Garcia et al. 2026 | **MLP**: all hidden + output neurons. **ResNet-18**: output/classifier layer only | incoming weights, forward pass only | **Yes, and it is reasoned, not swept** — see §4.1 |
| AlKilany & Goodman 2025 | all 200 recurrent LIF neurons | five *intrinsic* parameters: $v_{\text{th}}$, $v_r$, $U_0$, $\tau_m$, $\tau_s$ | modulator architecture swept (ANN vs. SNN); $K$ and $G$ swept; layer subset not swept |
| Costacurta et al. 2024 | the recurrent matrix of the output subnetwork | $K$ rank-1 components' singular values | $K$ swept (rank-1 vs. rank-3); rank-3 extrapolates better |
| Tsuda et al. 2021 | recurrent weights of a chosen subpopulation | rows/columns of $W$ | **Yes — the corpus's support ablation.** 100/90/50/10/0 %, and excitatory-only vs. inhibitory-only |
| Wainstein et al. 2025 | every unit's activation slope | sigmoid gain | static-$g$ sweep and dynamic-$\gamma$ sweep; no site sweep |
| Cox & Krichmar 2009 | **extrinsic + inhibitory synapses network-wide**; intrinsic excitatory exempt | synaptic input term | lesion study of *which nucleus*, not which site |
| Krichmar 2013 | modulator→frontal projections, frontal lateral inhibition, target state neurons | additive tonic + multiplicative ACh/NE gate | lesion of OFC→DA and mPFC→5-HT projections |
| Zou 2020 / Xing 2022 | a modulator layer *on top of* a frozen perception/RL backbone; nothing inside it | goal prior / policy-slot selector | ablation of ACh vs. NE (both necessary) |
| Wang et al. 2024 | every parameter (mask over $\theta$) | multiplicative mask | ablation of the three mask constraints |
| Kolouri et al. 2019 | every synapse | plasticity regulariser | — |
| Cañamero school | perception (ART-1 vigilance $\rho$), motivation gain, behaviour intensity, perceived body state | several named scalars, no network layers | robot-tier ablation (basic / hormonal / epigenetic) |

### 4.1 Verdict on Axis 2 — and the corpus's one explicit placement argument

**There is no field-level convention on the number of sites, because most papers only
have one site to choose.** When the modulated object is a learning rate, a discount
factor, or a hormone-modulated drive, "placement" does not arise. Among the papers
that *do* modulate a network's internals, the modal choice is "everywhere in the
modulated subnetwork" (Vecoven, Ben-Iwhiwhu, AlKilany, Wainstein, Tsuda at 100 %).

The corpus's one substantive, *stated* placement argument is
**Rodriguez-Garcia et al. 2026**, and it is worth quoting because it directly
contradicts the FiLM corpus's habit:

> "Convolutional networks mirror the brain's hierarchy of feature detectors, but unlike
> biological neurons, where each has its own input-output curve, CNN feature maps use a
> shared filter, so uniform scaling cannot match sensitivity at the neuron-level and
> may disrupt useful feature learning. Thus, we restricted gain modulation to the
> output layer, where it could regulate input sensitivity without disrupting feature
> selectivity."

Two things are being said. First, **per-neuron gain is the right form for a dense
layer and the wrong form for a conv layer** — precisely inverting the FiLM
literature's assumption that a conv channel is the natural modulation unit. Second, in
a deep backbone they **withdrew modulation from all but the last layer**, and reported
that head-only modulation still attenuated the stability gap (corroborating Łapacz et
al. 2024's finding that the classification head dominates stability-gap behaviour).
This is the corpus's only "fewer sites is better" result, and its mechanism —
*don't modulate a stage whose feature selectivity you need to leave alone* — is the
same mechanism behind the FiLM corpus's two placement dissents (AdaIN's single site;
Yan & Guo's frozen downstream classifier).

**Second placement finding, weaker but directly relevant to us: Ben-Iwhiwhu et al.
found modulating the critic unstable** and modulate only the actor in PEARL. Our design
modulates the shared trunk feeding both heads. No paper in either corpus modulates a
value head successfully; one paper reports trying and stopping.

---

## 5. Axis 3 — Parameterisation across sites

| Paper | Generator | Per-site parameters | Form |
|---|---|---|---|
| **Vecoven et al. 2020** | **one** recurrent network over interaction history → $z\in\mathbb{R}^k$ | **yes — per *neuron*, not per layer**: $(w_s,w_b)\in\mathbb{R}^k$ each | $\sigma\big((z^\top w_s)x + (z^\top w_b)\big)$ |
| Ben-Iwhiwhu et al. 2022 | **one modulator population per layer** ($W_g$, $W_m$) | inherent | $h=\mathrm{ReLU}(h_s\otimes\tanh(W_m\mathrm{ReLU}(W_g x)))$ |
| AlKilany & Goodman 2025 | **one** 2-layer MLP (or one recurrent LIF layer) | **yes** — one output group per modulated biophysical parameter | $p \leftarrow m$ or $p \leftarrow p + m$ |
| Costacurta et al. 2024 | **one** slow modulator subnetwork $z(t)$ | **yes** — one gate $s_k$ per rank-1 mode via $A_z, b_z$ | $s = \sigma(A_z z + b_z)$ |
| Rodriguez-Garcia et al. 2026 | none (a scalar recursion on batch entropy) | none — broadcast unchanged | $g \leftarrow \gamma g + (1-\gamma)g_0 + \eta H$ |
| Tsuda et al. 2021 / Wainstein 2025 / Osman 2024 | none | none — broadcast unchanged | one externally set or ODE-driven scalar |
| Cañamero school (Cañamero 1997; Lones 2018) | one gland per hormone | **yes — a learned per-target "receptor sensitivity"** $\text{Sens}_h$ | $\text{Drive}_i = \dfrac{\text{Sens}_i E_h^i}{\text{Sens}_t E_h^t}$, with $\text{Sens}_i(t{+}1)=\text{Sens}_i(t)^{E_h^i/\sigma}$ |
| Krichmar 2013 | one neuron per modulator | per-projection weights | $-1.0$ for 5-HT→DA, mPFC→5-HT, OFC→DA |
| Wang et al. 2025 NEST | **two separate MLPs**, one per output ($\Pi_\alpha$, $\Pi_\beta$) | inherent | $\alpha=\Pi_\alpha(C)$, $\beta=\Pi_\beta(F_a)$ |
| Zou 2020 / Xing 2022 | none (hand-designed multiplicative update rules) | one ACh entry per goal/task | $\mathrm{ACh}_i \leftarrow \mathrm{ACh}_i\cdot ch_{\{\text{correct,wrong}\}}$ |

### 5.1 Verdict on Axis 3 — the receptor pattern is this corpus's real contribution

**"One shared generator, per-site output heads" is present but is not the modal design
here**; it appears in AlKilany (per modulated biophysical parameter), Costacurta (per
rank-1 mode), and — with the site being a *neuron* rather than a layer — Vecoven.
The modal design in this corpus is simply **broadcast unchanged**, because the modal
signal is a scalar.

The structurally interesting pattern, and the one that has no counterpart in the FiLM
corpus, is what we can call the **receptor pattern**: a *low-dimensional global
carrier* plus a *per-target learned sensitivity* that decodes it. It appears three
times, in three literatures that do not cite each other:

- **Vecoven et al. 2020 (deep RL).** $z \in \mathbb{R}^k$ is global; each neuron holds
  $(w_s, w_b) \in \mathbb{R}^k$; the neuron's effective gain is $z^\top w_s$. This is
  mathematically a **rank-$k$ factorisation of a per-neuron $\gamma$ vector**: a
  128-neuron layer with $k{=}8$ has 8 degrees of freedom of *dynamic* variation
  distributed across 128 neurons by fixed learned directions. Vecoven notes the
  parameter cost is $2kN$, linear in neurons rather than quadratic in connections.
- **Cañamero school (affective robotics).** Hormone concentration is a scalar; each
  target holds a $\text{Sens}_h$ that is itself plastic (Lones et al. 2018:
  $\text{Sens}_i(t{+}1)=\text{Sens}_i(t)^{E_h^i/\sigma}$). Six qualitatively different
  robot phenotypes — pushing, ambush, hibernation, stalking — emerge purely from
  *receptor-sensitivity* divergence under a fixed hormone architecture.
- **Ferguson & Cardin 2020 / Shine et al. 2021 (biology).** The ligand is diffuse; the
  *receptor expression profile* is cell-class- and layer-specific, and this is what
  makes one broadcast signal produce opposite effects in different populations (ACh
  depolarises VIP⁺, disinhibits pyramidal dendrites, suppresses L4 excitatory cells but
  enhances L2/3 and L5). Shine's concentration-dependent $\alpha_2$-then-$\alpha_1$
  recruitment is a receptor scheme producing the Yerkes–Dodson inverted U from a single
  monotone signal.

**This is the design pattern the neuro literature would recommend to us and that the
FiLM literature does not contain.** It is worth naming because it is not the same thing
as grouping. Grouping ties $G$ neurons to *the same* value; the receptor pattern gives
every neuron a *different* value drawn from a $k$-dimensional subspace. Both reduce the
dynamic degrees of freedom from $N$ to something smaller; only grouping forces
neighbouring units to be identical.

---

## 6. Axis 4 — Grouping, population partitioning, multi-modulator target assignment

### 6.1 The one true grouping scheme: AlKilany & Goodman 2025, deepened

Our code cites this paper for "branched output heads with spatial grouping". Here is
what the source actually says, from Methods §3.3.2 and the Results:

**The scheme.** "In spatial grouping we use the same modulator output for a group of
$G$ neurons." Figure 1C: at $G{=}1$ "every output of the modulator modifies a single
parameter of a single neuron"; at $G>1$ "each output modifies a single parameter of $G$
neurons identically." Hidden layer is 200 recurrent LIF neurons; $G$ is swept over
$\{1, 2, 5, 10, 20, 50, 100, 200\}$ — i.e. from strictly per-neuron (cell a) all the
way to a single value for the whole layer (cell e).

**Stated motivation — biological realism of the anatomical range, not efficiency.**
The Results state it plainly: "The effect of neuromodulators can be as specific as a
single neuron, or may impact a small or large group of neurons. We therefore tested the
effect on performance when neurons are grouped into smaller or larger groups." The
Introduction frames the same point: neuromodulators "can change properties of small to
large groups of cells". A *secondary*, engineering motivation appears in the
Discussion: because the model works "at a range of spatial and temporal scales", "these
properties allow us to customise the model for optimal efficiency on each specific
design of neuromorphic device." So: **biological plausibility first, hardware
flexibility second. Regularisation is never mentioned.**

**The result — flat, with an explicit hedge.** "At least in the tasks and networks we
tried, spatially extended neuromodulation appeared to be about equally effective as
highly specific neuromodulation (fig. 2), although this may be different for different
tasks." Figure 2E plots accuracy against $G$ on Spiking Heidelberg Digits (spoken-digit
recognition) for the addition-coupling ANN modulator, with a mean curve and a
full-range band; the curve does not trend meaningfully across the whole $1 \to 200$
ladder. **The paper reports no advantage to coarsening and no cost to it.** It does not
support a "coarser is better" hypothesis.

**The second grouping axis, $K$ (temporal), is orthogonal and better-explored.** The
modulator runs every $K$ timesteps, receiving the input spikes summed over the previous
$K$ steps and holding its output constant for the next $K$. $K$ is swept from per-step
up to ~400 ms of interval. Unlike $G$, **$K$ matters and its optimum is
dataset-dependent** (best around 50–200 ms on some datasets, per-step on others) —
there is a real trade-off between integration time and update frequency. Our modulator
fires every environment step ($K{=}1$) and this axis has never been examined for us.

**The substitution-vs-addition coupling result, and why it constrains the grouping
result.** Two coupling rules are defined: *substitution* $p \leftarrow m$ (the
modulator emits the whole parameter value) and *addition* $p \leftarrow p + m$ (the
modulator emits a correction on top of the neuron's own learned parameter). Across
SHD/SSC/DVS, **substitution generally outperforms addition** — the paper's headline
coupling finding, and the one our design does *not* follow. But the $G$-sweep was run
**only on the addition branch**, and the methods say exactly why:

> "For comparability, we only use this for the ANN addition-based modulator. Parameter
> values at the start of the simulation are a learnable parameter, and heterogeneous
> (different for each neuron). However, an increase or decrease in a parameter value
> from the modulator network during a trial will be shared between a group of $G$
> neurons. **This ensures that the network remains fully heterogeneous (which would not
> be possible with the ANN replacement method.)**"

**This is our exact decomposition, with its rationale stated.** A learnable,
per-neuron, heterogeneous **static baseline**, plus a **grouped dynamic increment**
added on top — and the stated reason for combining them is that grouping the
*substituted* value would destroy per-neuron heterogeneity, whereas grouping the
*increment* preserves it. Our `nnx.Param` per-neuron baselines plus
`baseline + jnp.repeat(raw, grouping_size)` implement the same thing for the same
structural reason.

**This corrects the FiLM synthesis.** Its judgment 3 states that the composite —
"per-neuron static resolution with deliberately coarser dynamic resolution" — "appears
nowhere in either corpus, and I did not find it in the neuromodulation corpus either."
It is in the neuromodulation corpus, in the methods section of the very paper our code
cites. The composite is **not** novel; what remains unattested is our *motivation* for
it (regularisation) and the finding we hope to get (coarsening improves performance).

### 6.2 Population partitioning — the biological analogue of grouping

This is where the neuro corpus is genuinely rich, and it is *not* the same thing as
Axis-1 grouping. Six distinct partitioning schemes appear:

| Paper | Partition | How chosen | Motivation | Result |
|---|---|---|---|---|
| **Tsuda et al. 2021** | contiguous or random **subpopulations** of the RNN, 10 %–100 %; and separately **excitatory-only vs. inhibitory-only** | arbitrary (size swept); the E/I split is anatomically motivated ("just as some neuromodulators affect neurons in a cell-type specific manner… with corresponding receptors") | demonstrate that real neuromodulators range from "tightly localized" to "broadcast widely" | **support size 100 %→10 % all work**; E-only and I-only both work; nine *non-overlapping* 10 % subpopulations store nine distinct behaviours in one network. See §6.3 for the multi-behaviour result |
| **Cox & Krichmar 2009** | **connection type**: extrinsic + inhibitory synapses are modulated ($nm = 10\bar a$); intrinsic excitatory are not ($nm=1$) | anatomically motivated (thalamocortical vs. intracortical) | produce signal-to-noise sharpening: make sensory drive dominate associational background | lesion study confirms modulator-specific behaviour collapse; SNR drops under lesion |
| **Avery & Krichmar 2017** (reviewing Deco & Thiele; Avery et al. 2014) | thalamocortical vs. intracortical drive | anatomical | ACh raises $I_{TC}/I_{IC}$ — "the cortex listens to the world rather than to its own ongoing thought" | the same partition as Cox & Krichmar, independently derived |
| **Krichmar 2013** | which **projection** each modulator targets: Object→DA, Light→5-HT, Bump→both; OFC→DA, mPFC→5-HT | anatomically motivated | reproduce open-field anxiety/curiosity pharmacology | the *asymmetry* of the two lesions is predicted from where each sits relative to the 5-HT→DA opponency |
| **Ferguson & Cardin 2020** | PV⁺ (perisomatic) / SST⁺ (dendritic) / VIP⁺ (disinhibitory) interneuron classes; cortical layers | anatomical (receptor expression) | explain *how* a diffuse ligand produces specific, sometimes opposite, gain effects | ACh→VIP⁺⊣SST⁺⊣pyramidal-dendrite is the canonical disinhibition circuit |
| **Mei et al. 2022** Scale 2 | excitatory vs. inhibitory populations, $g_k(\mathbf m_t)$ per population | proposed | disinhibition; "cell-type-specific neuromodulation" as one of four scales a DNN should implement | proposal; cites Tsuda 2021 as the exemplar |

**How partitions are chosen: overwhelmingly anatomically, never learned.** Not one
paper in the corpus *learns* its partition. Tsuda's subpopulations are drawn
arbitrarily and then swept for size; every other partition is imported from
neuroanatomy (E/I, interneuron class, projection target, thalamocortical vs.
intracortical). **No paper in this corpus learns which units should share a modulation
signal.** That is a real gap, and it is the gap a learned-grouping variant of our
design would fill.

### 6.3 Papers that ablate scope, and what they found

Exactly **three**, and they sit on three different axes. Keeping them apart is
essential.

1. **AlKilany & Goodman 2025 — resolution ($G$), full support.** All 200 neurons are
   modulated; the number of distinct values falls from 200 to 1. **Result: flat.**
   Explicitly hedged as possibly task-dependent. This is the *only* experiment in the
   entire reference library — either corpus — that varies modulation resolution while
   holding everything else fixed.
2. **Tsuda et al. 2021 — support (which units), fixed resolution of one scalar.**
   100 % / 90 % / 50 % / 10 % / 0 % of the network. **Result on the 2-behaviour task:
   flat from 100 % down to 10 %** (0 % is the negative control and fails). **Result on
   the 9-behaviour task: narrow support wins outright.** Extended Data Fig. 6 reports
   that whole-network modulation with nine different scalar *levels* supports 3
   behaviours (5/5 networks) but fails entirely at 4, 5, and 9 behaviours (0/5 each),
   whereas nine non-overlapping 10 % subpopulations learn the full 9-behaviour task
   (8/10 networks). Larger and overlapping subpopulations "less consistently learn the
   full task". **This is the corpus's one result where restricting scope strictly
   improves capacity** — and the mechanism is interference between behaviours competing
   for the same modulated substrate, which is a *capacity* argument, not a
   regularisation argument.
3. **Rodriguez-Garcia et al. 2026 — sites (which layers), fixed global resolution.**
   Not a sweep but a reasoned restriction: all hidden+output neurons for a dense MLP,
   output layer only for a ResNet-18 backbone, on the grounds that uniform scaling of a
   shared conv filter "may disrupt useful feature learning". Head-only modulation still
   attenuated the stability gap.

Two further near-misses: **Costacurta et al. 2024** sweeps the rank $K$ of the
modulated basis (rank-3 extrapolates better than rank-1 on the timing task — *richer
wins*), and **Vecoven et al. 2020** sweeps main-network depth (NMN insensitive, RNN
baseline sensitive) but not modulation scope.

### 6.4 Multi-modulator designs and how targets get assigned

Nine corpus papers run more than one modulator. The assignment rule is always one of
three, and never learned.

| Paper | Modulators | Target assignment | Each modulator's scope |
|---|---|---|---|
| **Doya 2002** | DA, 5-HT, NA, ACh | **one algorithmic meta-parameter each**: DA=$\delta$ (TD error), 5-HT=$\gamma$ (discount), NA=$\beta$ (inverse temperature), ACh=$\alpha$ (learning rate). Derived from matching each modulator's *behavioural* signature to each parameter's *algebraic* role | each is a **global scalar**; Fig. 9 predicts a cross-effect matrix among them |
| **Lee et al. 2024** | ACh, NA (DA implicit) | Doya's assignment, refined through Yu & Dayan: ACh↔expected uncertainty→$\alpha$, NA↔unexpected uncertainty→$\beta$. 5-HT **dropped** for lack of a convergent normative theory | global scalars, state-dependent |
| **Avery & Krichmar 2017** | DA, 5-HT, ACh, NA | presents Doya's mapping *and* the rival "decisiveness" view (all four do the same downstream thing — SNR sharpening — and differ only in triggers). Flags that the same parameter $\alpha$ has been assigned to 5-HT (Balasubramani 2015) *and* NA (Nassar 2012) | global; explicitly says the mapping is "task- and context-dependent, not a fixed identity" |
| **Cox & Krichmar 2009** | DA (VTA), 5-HT (Raphe), ACh (BF) | by **trigger**, not by target: same downstream effect (multiply extrinsic+inhibitory input), different eliciting event (Good button vs. Bad button vs. attentional load) | one summed global scalar; specificity comes entirely from *which colour channel is co-active at burst time* |
| **Krichmar 2013** | DA, 5-HT, ACh/NE | by **projection**: Object→DA, Light→5-HT, Bump→both; ACh/NE gates frontal competition. Plus 5-HT⊣DA opponency and frontal⊣modulator cognitive control | tonic and phasic scalars per modulator |
| **Zou 2020 / Xing 2022** | ACh, NE | by **function**: ACh carries a per-goal / per-task confidence vector, NE a scalar reset detector. Ablations show **both are necessary** (ACh alone → no switching; NE alone → random guessing) | ACh is a $K$-vector over goals/tasks; NE is a scalar |
| **Cañamero school** | 3 endocrine hormones + 1 neurohormone (Lones 2018); cortisol (L'Haridon); oxytocin + cortisol (Khan) | by **homeostatic variable**: one hormone per physiological deficit; the neurohormone $D_1$ gates avoidance and speed | scalar concentration × per-target receptor sensitivity |
| **Shine et al. 2021** | ACh, NA, DA, 5-HT | by **which parameter of the population gain function** each reshapes: height, slope, threshold, temporal aperture | global release, layer- and receptor-partitioned effect |
| **Mei et al. 2022** | ACh, NA, DA, 5-HT | by **scale**: hyperparameters, cell types, weights, dendritic compartments. Recommends *combining* rather than choosing | mixed; Scale 2 is explicitly grouped |

**Verdict.** Assignment is by **trigger** (Krichmar school), by **algebraic role**
(Doya school), by **homeostatic variable** (Cañamero school), or by **gain-function
parameter** (Shine). Never learned, and the corpus's two schools openly disagree:
Doya says one modulator carries one distinct parameter; Krichmar 2008 says all
modulators do the same thing downstream and differ only in what sets them off. Avery &
Krichmar 2017 presents both as legitimate and does not resolve it. **Multi-target
modulation from a shared source is therefore normal in this corpus. Multi-target
modulation from a shared *learned generator* is not** — the assignments here are all
hand-designed.

---

## 7. What is modulated, and how scope co-varies with it

This is the axis the FiLM corpus does not have, and it explains most of the granularity
counts in §3.

| Target type | Papers | Typical granularity | Why |
|---|---|---|---|
| **Activation gain / slope** | Vecoven 2020; Ben-Iwhiwhu 2022; Wainstein 2025; Rodriguez-Garcia 2026; Shine 2021; Ferguson & Cardin 2020 | **(a) per-unit** when a generator exists; **(e) global** when the paper is a pure gain-ODE | this is the one target where a per-unit vector is natural, and it is the target our $\gamma$/$\beta$ heads hit |
| **Recurrent weights** | Tsuda 2021 ($f_{nm}W$); Costacurta 2024 ($LS(z)R^\top$); Osman 2024 ($M/\alpha$) | **(e) global scalar**, or **(c) low-rank** | a weight matrix has $N^2$ entries; a per-entry modulator is a hypernetwork, which these papers deliberately avoid for tractability |
| **Intrinsic biophysical parameters** (threshold, reset, rest, $\tau_m$, $\tau_s$) | AlKilany & Goodman 2025 | **(a)→(c)→(e)**, swept | the only corpus paper that treats granularity as a free variable |
| **Learning rate** | Doya 2002; Lee 2024; ACh in Avery & Krichmar 2017 | **(e) global scalar** | a learning rate is a scalar by construction |
| **Exploration temperature** | Doya 2002 ($\beta$=NA); Lee 2024 ($\beta(s)=1/\langle E\rangle$); Zou 2020 (softmax over ACh, $\beta{=}0.7$); Xing 2022 ($\beta{=}2$) | **(e) global scalar** — always | there is exactly one action distribution, so there is exactly one temperature. **No paper anywhere in the library emits a per-unit temperature.** |
| **Plasticity gate** | Cox & Krichmar 2009 (Heaviside gate $nm>2$ on BCM); Kolouri 2019 (per-synapse $\gamma$); Tambaş 2025 (per-unit rank gate); Hwu & Krichmar 2020 (replay epochs) | **(e) global gate** ×  **per-synapse importance** | the gate is global; the *importance* it gates is per-weight — a two-level scheme with no analogue in FiLM |
| **Recurrent gating / memory** | Costacurta 2024 (proves the modulator ≡ an LSTM forget gate); Ben-Iwhiwhu 2022 (notes the resemblance) | **(c) per-mode** / **(a) per-unit** | Costacurta's $s_k(z(t))\odot w_{t-1}$ *is* a forget gate at rank-$K$ resolution |
| **Structure / routing** | Wang 2024 (per-weight mask); Wang 2025 NEST (two graph scalars); Driscoll 2022 (rule input) | **per-weight** or **2 scalars** | bimodal; nothing in between |
| **Perceptual threshold / perceived state** | Cañamero 1997 (ART-1 vigilance $\rho$); L'Haridon 2023 (pain = cortisol × damage) | **(e) global scalar** × per-variable receptor | homeostatic architectures have no hidden units to index |

**The pattern:** granularity is not a free design choice in most of this corpus; it is
determined by the target. Our project modulates **three** of these target types at once
— activations (per-unit-ish), a recurrent gate bias (per-unit-ish), and an action
temperature (necessarily scalar) — and this is the first cross-target comparison that
makes that visible. Note the consequence for our sweep: **the temperature head is
outside the `grouping_size` axis entirely** (it emits one number by construction), so
whatever the grouping screen finds cannot apply to it.

---

## 8. Where this project sits — the four judgments

### 8.1 Our design, in taxonomy terms

From `src/models/neuromodulator.py`, class `NeuromodulatorRNN`:

- **Generator (Axis 3):** **one** shared GRU cell (`mod_hidden_size: 16`) reading the
  raw observation vector.
- **Sites (Axis 2):** **six heads, four target objects, three target types** —
  $\gamma$ and $\beta$ for a unimodal sensory-encoder stage; $\gamma$ and $\beta$ for a
  multimodal-hub stage; a bias added into the task GRU's update gate (clipped to
  $[-2,2]$); a scalar action temperature (softplus, offset $+0.5$, clipped).
- **Granularity (Axis 1):** cell **(c)**. Each $\gamma$/$\beta$/memory head emits
  $\lceil 128/g \rceil$ raw values, expanded by
  `jnp.repeat(raw, grouping_size)[:128]`. The sweep spans $g{=}1$ (cell a) to $g{=}128$
  (cell e). The temperature head is a single unit and is not grouped.
- **Decomposition (Axis 4):** a full 128-dim **per-neuron learned baseline**
  (`nnx.Param`, zero-initialised) is added to every expanded signal. Static component
  always per-neuron; only the dynamic component is grouped.

### 8.2 Judgment 1 — Where does our design sit in the neuro literature's conventions?

**A learned, observation-driven, near-per-unit vector modulator is a *minority* design
in this corpus, and the majority design is a single global scalar. But the minority is
the part of the corpus we actually descend from, and within it our design is
conventional in three of four respects.**

Break it into four sub-claims.

- **Is a *learned* modulator normal here? Yes, but only recently.** Roughly half the
  corpus hand-designs its modulator dynamics (Cañamero school hormones, Krichmar-lab
  ACh/NE multiplicative update rules, Doya's meta-parameters, Zou/Xing's threshold
  rules). Learned modulator subnetworks are the post-2020 wing: Vecoven, Ben-Iwhiwhu,
  Costacurta, AlKilany, Wang 2024/2025. We are firmly in that wing.
- **Is an *observation-driven* modulator normal? No — this is our clearest
  divergence.** The deep-RL wing's stated design principle is that the modulator should
  receive **context, not raw observation**. Vecoven is explicit and structural about
  it: the main network gets $x_t$, the modulator gets $c_t = h_t \setminus x_t$ — the
  history *minus* the current observation. Ben-Iwhiwhu feeds a task-context vector
  ($\phi$ from CAVIA, $z$ from PEARL). AlKilany feeds recent *hidden-layer spiking
  activity plus current parameter values*, i.e. the modulated network's own state.
  Costacurta feeds a context one-hot to the modulator subnet and nothing else.
  Rodriguez-Garcia feeds output entropy. **Our modulator reads the raw observation
  through a GRU** — closest to Wainstein/Rodriguez-Garcia in spirit (an internal
  uncertainty-like signal) but structurally unlike all of them, because nothing in the
  corpus feeds a modulator the same input the main network already sees. Whether that
  is a defect depends on whether the GRU's recurrence is doing the context-extraction
  the corpus considers essential; that is an empirical question nobody has posed.
- **Is a *per-unit-ish vector* normal? It is the minority form (4 of ~24), but it is
  the form used by every paper whose modulated object is an activation.** The scalar
  majority is largely papers modulating scalars (learning rates, discounts,
  temperatures, hormone-gated drives). Restricted to "papers that modulate a hidden
  layer's activations with a learned generator", the per-unit vector is the *norm*:
  Vecoven, Ben-Iwhiwhu, and AlKilany-at-$G{=}1$ all do it. **So our vector output is
  normal for what we modulate, and the "neuro uses scalars" headline should not be
  read as a criticism of it.**
- **Is our *lack of a receptor layer* normal? No.** Vecoven gets per-neuron variety
  from a $k$-dim global carrier decoded by per-neuron learned weights; we get it from
  $\lceil 128/g \rceil$ independent linear outputs. At $g{=}1$ our head is a
  $16 \to 128$ linear map, which *is* a rank-16 factorisation and therefore closer to
  Vecoven than it first appears — a fact worth stating in any write-up, because it
  means our $g{=}1$ condition is **not** 128 free degrees of freedom.

**Recommendation.** Present our modulator as a member of the Vecoven→Ben-Iwhiwhu→
Costacurta→AlKilany lineage (learned modulator subnetwork acting on activations),
not as a member of the Doya→Krichmar→Cañamero lineage (hand-designed global scalars),
and be explicit that the raw-observation input is a departure the lineage would
question.

### 8.3 Judgment 2 — Is our multi-head / multi-target design attested?

**Multi-target modulation from a shared source is thoroughly attested. Multi-target
modulation from a shared *learned* generator with per-target heads is attested twice.
Our specific mix — activations + a recurrent gate + an action temperature — is not
attested anywhere, and the temperature head is the odd one out.**

Three levels of attestation:

1. **Multi-target from one source: standard.** Doya's four modulators each own a
   different algorithmic parameter; Cañamero's hormones each own a different drive;
   Krichmar 2013's ACh/NE gates both frontal lateral inhibition *and* modulator→frontal
   projections. Cox & Krichmar's single $nm$ scalar simultaneously (i) multiplies
   extrinsic input, (ii) multiplies inhibitory input, and (iii) gates BCM plasticity
   above a threshold — three target types from one number.
2. **Multi-target from one *learned generator with per-target heads*: exactly two
   papers.** **AlKilany & Goodman 2025** — one MLP with one output group per modulated
   biophysical parameter (threshold, reset, rest, $\tau_m$, $\tau_s$): five heads, same
   generator, exactly our layout. **Costacurta et al. 2024** — one modulator subnet
   $z(t)$ with one gate $s_k$ per rank-1 mode.
3. **One modulator per target type: also common**, and it is the Krichmar-lab default
   (a separate ACh module and a separate NE module in Zou 2020 and Xing 2022, with
   different state shapes — a $K$-vector and a scalar). Wang 2025 NEST uses two
   *separate* MLPs for its two scalars. **Vecoven et al. 2020 deliberately splits**:
   actor and critic each get their **own independent NMN with no shared modulator**, on
   the argument that "the modulatory signals for policy vs. value may differ" — an
   argument the paper concedes is intuitive but unproven.

**The gap.** Nothing in the corpus drives an **action-selection temperature** from the
same learned generator that drives **activation gains**. Temperature modulation exists
(Doya's $\beta$=NA; Lee's $\beta(s)$; the softmax temperatures in Zou and Xing) but
always as a stand-alone scalar computed from an uncertainty statistic, never as one
head among several on a shared trunk. Likewise, modulating a **recurrent gate bias** is
attested only obliquely: Costacurta proves that a rank-$K$ modulator of a low-rank
recurrent matrix *is* an LSTM forget gate, which makes our memory head a coarse
relative of their $s_k$ — but they modulate the recurrent *weights*, we add a *bias to
the update gate*, and no paper compares the two.

**Verdict: our layout is a legitimate extrapolation with one clean precedent
(AlKilany's five-head shared modulator), but the specific act of yoking a policy
temperature to the same trunk as perceptual gain is unprecedented and worth flagging as
a design claim rather than a convention.** Vecoven's actor/critic split is the corpus's
one recorded worry in this direction, and Ben-Iwhiwhu's "modulating the critic was
unstable" is the one recorded failure.

### 8.4 Judgment 3 — Is our grouping sweep biologically motivated by anything beyond AlKilany?

**Yes — but the biological motivation supports *population targeting*, not *contiguous
block tying*, and the distinction matters.**

What the corpus genuinely supports:

- **Diffuse-to-focal is a real anatomical range.** Tsuda: "neuromodulators are released
  in specific regions — some tightly localized, others broadcast widely — to influence
  local and global neural output." AlKilany's Introduction: modulators "change
  properties of small to large groups of cells". Mei's Scale 2 makes cell-type-specific
  modulation a first-class design target. **So the *existence* of grouped modulation is
  well grounded biologically.** Our sweep is not an arbitrary hyperparameter; it spans
  a range the biology actually occupies.
- **A shared modulator acting on a coherent population is the biological norm, not the
  exception.** Ferguson & Cardin's interneuron classes, Shine's layer-specific receptor
  topography, Cox & Krichmar's extrinsic/intrinsic split, Krichmar 2013's
  projection-specific targeting — every one of these is "many units, one signal".
- **Restricting scope can strictly help.** Tsuda's 9-behaviour result (§6.3) is the
  corpus's only case where narrowing beats global, and it is a clean, quantitative
  capacity argument: whole-network modulation supports 3 behaviours and fails at 4+;
  nine 10 % subpopulations support all 9.

What the corpus does **not** support:

- **Contiguity.** Every biological partition in the corpus is *functional* — E vs. I,
  interneuron class, cortical layer, projection target. Not one is "neurons 0–7 share a
  value, neurons 8–15 share another". Our `jnp.repeat` grouping is contiguous in an
  arbitrary index order that carries no functional meaning, since the hidden units of a
  trained GRU have no canonical ordering. **Biologically, our grouping is the *arbitrary
  partition* control condition, not the biologically motivated one.** AlKilany's is too
  — and they are honest that the choice is about spanning the anatomical range, not
  about matching a specific circuit.
- **Regularisation.** No paper in the corpus offers grouping as a regulariser or reports
  an over-powered modulator. The nearest thing is Rodriguez-Garcia's *reduction* of
  modulation sites in a conv backbone to avoid disrupting feature selectivity, and
  Ben-Iwhiwhu's decision not to modulate the critic because it was unstable — two
  instances of "modulating less, for stability reasons", both about *placement*, not
  granularity.
- **A prediction about the outcome.** AlKilany reports flat and hedges; Tsuda reports
  flat down to 10 % support on the easy task. Neither predicts improvement.

**Verdict: the sweep is well-motivated as a *biological-range* sweep and poorly
motivated as a *regularisation* sweep. If the screen returns a positive result, the
most defensible framing is "arbitrary-partition grouping acts as a capacity
constraint", with Tsuda's multi-behaviour interference result as the mechanism to
argue from — not AlKilany's flat curve. A cheap, biologically better-motivated variant
worth naming for `experiment-designer`: partition by *function* rather than by index —
e.g. tie the units that feed the policy head separately from those that feed the value
head — which is the corpus's actual convention and which no paper has tested in a
learned-modulator setting.**

### 8.5 Judgment 4 — Does this corpus support or undercut "finer is better"?

**It substantially undercuts it, but not by finding that coarser is better — by finding
that the whole granularity axis is much flatter than the FiLM corpus implies, and that
the FiLM corpus's evidence was about *capacity*, not *granularity*.**

Set out the evidence side by side.

| Evidence | Corpus | Axis actually varied | Result |
|---|---|---|---|
| Abdollahzadeh 2021 (KML) per-channel vs. per-weight | FiLM | granularity of the modulation signal | finer wins (~5 pp) |
| Beck 2023 full-weight hypernet vs. FiLM $(\gamma,\beta)$ | FiLM | *what is generated* (all weights vs. affine only) | richer wins (42.9 % vs. 25.5 %) |
| Nikulin 2023 FiLM vs. bilinear | FiLM | *rank* of the conditioning map | tie on quality, FiLM wins on cost |
| **AlKilany 2025 $G$ sweep** | **neuro** | **granularity, cleanly** | **flat, $G{=}1 \to 200$** |
| **Tsuda 2021 subpopulation sweep** | **neuro** | **support** | **flat 100 %→10 % (2-behaviour); narrow wins outright (9-behaviour)** |
| Costacurta 2024 rank $K$ | neuro | rank of the modulated basis | richer (rank-3) extrapolates better |
| Rodriguez-Garcia 2026 site restriction | neuro | number of modulated layers | fewer sites at least as good in a conv backbone |
| Tsuda 2021 whole-network scalar | neuro | granularity at the extreme | **a single scalar × all weights is sufficient to store 9 distinct behaviours** and to reproduce fly pharmacology |

Three observations.

1. **Only one of the FiLM corpus's three "finer wins" results is actually a granularity
   comparison.** Beck's is a capacity comparison (generate everything vs. generate an
   affine); Nikulin's is a rank comparison and the *coarser* option wins on cost.
   Abdollahzadeh's is genuine — and it is a *few-shot meta-learning* result whose stated
   mechanism ("one degree of freedom per channel per task, so heterogeneous tasks fight
   for capacity") is about **many tasks competing**, not about single-task performance.
   Notice that Tsuda's 9-behaviour result has the *same* mechanism — interference
   between behaviours sharing a modulated substrate — and reaches the *opposite*
   conclusion, because Tsuda's fix is to give each behaviour its own *disjoint
   subpopulation* rather than to give every unit its own value. **These are the same
   phenomenon with two different remedies: more resolution, or more separation.**
2. **The neuro corpus's existence proofs are extraordinarily coarse.** Tsuda stores nine
   behaviours in one 200-unit RNN with a *single scalar*. Wainstein reproduces human
   perceptual switching with a *single scalar*. Osman derives a clean bifurcation from a
   *single scalar*. A corpus in which one number reshapes a whole network's dynamics
   cannot be reconciled with "finer is always better" — it says the useful modulation is
   often genuinely low-dimensional.
3. **The two corpora do not actually disagree about mechanism; they disagree about
   regime.** FiLM's finer-is-better evidence comes from *supervised many-task*
   settings where the modulator must express many distinct target functions. The neuro
   corpus's flat-in-granularity evidence comes from *single-task* settings (SHD speech,
   Go/No-Go) where the modulator's job is to set an operating point. **Our grid-world
   agent is closer to the second regime than the first** — one environment, one task,
   the modulator setting a moment-to-moment operating point — which is a reason to
   expect flatness rather than a strong granularity effect.

**Verdict: revise the FiLM synthesis's "all three granularity ablations found
finer-is-better" to "the two corpora together contain five scope ablations; two find
finer-is-better in many-task supervised settings, two find flat in single-task
settings, and one finds narrower-support-wins under multi-behaviour interference." The
literature's honest position is that scope effects are regime-dependent, and our
regime is the one where flatness has been reported.**

---

## 9. Agreements and disagreements with the FiLM half

**Agreements.**

- **Cell (c) grouping is essentially unattested, and AlKilany & Goodman 2025 is the sole
  precedent in the entire reference library.** Independently confirmed here by a
  full-text regex scan of all 41 neuromod PDFs: `spatial grouping` / `group size` /
  `grouping size` appear in that paper and nowhere else.
- **Cell (d), the layer-scalar, is unattested in both corpora.** No paper in either
  emits one number per layer as its modulation signal.
- **"Modulate less where a downstream stage's selectivity must be preserved" is a real,
  cross-corpus placement finding.** FiLM's version is AdaIN's single site and Yan &
  Guo's frozen classifier; neuro's version is Rodriguez-Garcia's head-only restriction
  in a conv backbone and Ben-Iwhiwhu's refusal to modulate the critic.

**Disagreements / corrections.**

1. **The FiLM half's judgment 3 is wrong on a point of fact.** It states that the
   composite "per-neuron static baseline + coarser grouped dynamic component" "appears
   nowhere in either corpus, and I did not find it in the neuromodulation corpus
   either". It is in **AlKilany & Goodman 2025, Methods §3.3.2**, with the rationale
   stated (grouping the increment preserves per-neuron heterogeneity; grouping a
   substituted value would not). Our decomposition is **not novel**; it is a faithful
   reproduction of the cited paper's design.
2. **"Finer is better" does not survive contact with this corpus.** See §8.5. Two
   further scope ablations exist here; both find flat, and one finds narrower-is-better.
   The FiLM claim should be scoped to many-task supervised regimes.
3. **The FiLM half's framing "the field's convention is per-channel/per-unit" is
   corpus-specific, not field-wide.** In the neuromodulation literature the convention
   is a single global scalar (~15 of ~24 implementing papers). Any paper we write should
   say *which* literature's convention it is departing from.
4. **The FiLM half has no analogue of the receptor pattern** (§5.1) — a low-dimensional
   global carrier decoded by per-target learned sensitivities. This corpus has three
   independent instances, and it is arguably the design the biology recommends. It is
   also, quietly, what our $g{=}1$ condition already is (a rank-16 map from a 16-unit
   GRU to 128 outputs), which nobody has noted.
5. **The FiLM half's Axis-2 conclusion "inject at many depths including early ones"
   does not transfer.** That conclusion rests on two conv-vision sweeps (De Vries's
   stage-wise sweep, Nikulin's `film_full`). The one comparable neuro result runs the
   other way in a conv backbone, and the neuro corpus's dense/recurrent papers modulate
   everything without ablating. **There is no cross-corpus support for "modulate at many
   depths" in a recurrent RL policy.**

---

## 10. What this corpus does NOT settle

1. **Whether coarsening a modulation signal ever *improves* performance.** The corpus's
   one clean granularity sweep (AlKilany $G$) reports flat and hedges that it "may be
   different for different tasks". Tsuda's narrow-support win is on a *different axis*
   (support, not resolution) and under a *different pressure* (nine behaviours competing
   for one substrate). **Our grouping screen's pre-registered hypothesis has no support
   in either corpus.** It is genuinely open — which makes it worth running, and makes
   any positive result worth scrutinising hard.
2. **Whether a modulator can be over-powered.** No paper in this corpus reports a
   modulation-capacity-induced instability. The two closest things — Ben-Iwhiwhu's
   unstable critic modulation and Rodriguez-Garcia's "uniform scaling… may disrupt
   useful feature learning" — are both about *where* modulation is applied, not *how
   many degrees of freedom* it has. Our motivating observation (per-neuron $g{=}1$ led
   early then destabilised late with the temperature knob railed) is unexplained by
   anything in the literature.
3. **Whether an arbitrary contiguous partition behaves like a functional one.** Every
   biologically motivated partition in the corpus is functional; ours is index-order
   contiguous. Nobody has compared arbitrary against functional grouping. If they differ,
   our screen measures the arbitrary case only.
4. **How to learn a partition.** No paper in the corpus learns which units should share
   a modulation signal. Wang 2024's learned per-task mask is the closest (it learns
   *which* units are active, not which share a value), and Vecoven's per-neuron
   $(w_s, w_b)$ is the closest continuous relaxation. This is a field-level gap, not a
   gap in our coverage.
5. **Whether modulating a recurrent cell's gate bias is a good idea.** Costacurta proves
   a rank-$K$ recurrent-weight modulator is an LSTM forget gate; we add a bias to a GRU
   update gate. Different operations, never compared, in either corpus.
6. **What the right input to a modulator is.** Vecoven withholds the current observation
   on principle; AlKilany feeds the modulated network's own activity; Rodriguez-Garcia
   and Wainstein feed an entropy statistic; Lee feeds an uncertainty decomposition; the
   Cañamero school feeds homeostatic deficit. We feed the raw observation. There is no
   head-to-head comparison anywhere, and the choice is at least as consequential as
   `grouping_size`.
7. **Whether the multi-head design creates cross-head interference.** Doya's Fig. 9
   predicts a full cross-effect matrix among four modulators (5-HT⊣NA, 5-HT⊣ACh,
   DA-variability⊣ACh, and so on) precisely because the parameters interact
   algebraically. Our six heads share a 16-unit GRU state and therefore *cannot* vary
   independently. Nobody has studied what a shared low-dimensional trunk does to
   multi-target modulation.
8. **Whether the per-neuron baseline and grouped dynamic component interact badly.**
   AlKilany introduced the decomposition to *preserve heterogeneity*, not to isolate a
   granularity effect, and never checked whether the baseline absorbs the dynamic
   signal. Inherited untested. (This is the same confound the FiLM half raised in its
   §8 item 5; it is not resolved by finding the precedent.)

---

## 11. Connections to project gates, hypotheses, and phases

- **Phase 2 (FiLM variant characterisation)** is the phase this document serves. §8.4
  says the grouping axis is biologically real but our *contiguous* instantiation is the
  arbitrary-partition control; §8.5 says the field's expectation should be flatness in
  our regime.
- **H1 (perceptual amplification)** predicts $\gamma$ rises on nociceptive-relevant
  features after injury — a *feature-selective* claim. The tension the FiLM half flagged
  stands and sharpens here: at $g{=}128$ the dynamic component is a single global
  scalar, so H1's selectivity survives only in the static per-neuron baseline. This
  corpus adds a reason to care: **the receptor pattern (§5.1) says feature selectivity
  under a low-dimensional carrier lives in the per-target sensitivities, not in the
  carrier**, so at any $g$ the right place to read H1 may be the *baseline plus head
  weights*, not the emitted signal.
- **H2 (memory persistence)** rides on the memory gate-bias head. Costacurta 2024 is the
  corpus's only formal treatment of modulation-as-forget-gate and shows the modulator
  sets the *time constant of each dynamical mode*; that is the right analytical frame for
  H2 and is currently unused in our docs.
- **G1 / G2** ([project_plan.md §4](../../project_plan.md)) are survival-step thresholds
  and are not directly informed by this synthesis.
- **Null-result diagnosis series v1–v8** ([`docs/develop/INDEX.md`](../../../develop/INDEX.md)):
  this corpus supplies two candidate explanations the series has not exploited.
  **(a) Modulator input.** Every learned modulator in this corpus withholds the raw
  observation from the modulator and feeds it context, history, its own activity, or an
  uncertainty statistic instead; we feed it the observation. If the modulator has no
  information the main network lacks, a null result is the expected outcome — and this
  is testable cheaply by swapping the modulator's input. **(b) Site selection.** The
  corpus's one reasoned placement argument (Rodriguez-Garcia) and its one recorded
  placement failure (Ben-Iwhiwhu's critic) both say *which* stage you modulate matters
  more than how finely. Our five injection points were chosen by function and never
  ablated. Both of these are better-supported next axes than granularity.

---

## 12. Cross-references

- FiLM + Hypernetwork half of this question (shared taxonomy, shared verdict format):
  [`../FiLM/film_modulation_granularity_synthesis.md`](../FiLM/film_modulation_granularity_synthesis.md)
- Master index over the 41 per-paper neuromodulation reviews:
  [`neuromodulatory_algorithms_lit_review.md`](neuromodulatory_algorithms_lit_review.md)
- Thematic + historical synthesis (clusters A–G, 1997→2026 timeline, cross-cluster
  matrix). Its §5 Question 6 — "is the right level of modulation per-neuron, per-layer,
  per-rank-1-component, or global?" — is the question this document answers:
  [`neuromodulatory_algorithms_synthesis.md`](neuromodulatory_algorithms_synthesis.md)
- Corpus-derived predictions for v3:
  [`neuromodulatory_algorithms_predictions_synthesis.md`](neuromodulatory_algorithms_predictions_synthesis.md)
- FiLM ↔ neuromodulation mechanism mapping:
  [`../FiLM/film_neuromod_integration_synthesis.md`](../FiLM/film_neuromod_integration_synthesis.md)
- The grouping precedent, in full:
  [`reviews/alkilany_goodman_2025_snn_dynamic_sensory.md`](reviews/alkilany_goodman_2025_snn_dynamic_sensory.md)
- The receptor pattern's deep-learning instance:
  [`reviews/vecoven_2020_neuromod_dnn.md`](reviews/vecoven_2020_neuromod_dnn.md)
- The support ablation:
  [`reviews/tsuda_2021_activity_hypertubes.md`](reviews/tsuda_2021_activity_hypertubes.md)
- The placement argument:
  [`reviews/rodriguezgarcia_2026_ne_stability_gap.md`](reviews/rodriguezgarcia_2026_ne_stability_gap.md)
- Project algorithm spec (H1–H5, injection points):
  [`docs/develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md`](../../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md)
- The live grouping experiment:
  [`docs/experiments/active/basic_curriculum/NMN_FILM_GROUPING_SCREEN.md`](../../../experiments/active/basic_curriculum/NMN_FILM_GROUPING_SCREEN.md)

### Recommended follow-ups

- **`experiment-designer`** — two cheap, better-supported axes than granularity:
  (i) a **modulator-input** ablation (raw observation vs. observation-plus-history vs.
  the modulated network's own hidden state), which is the one design choice the entire
  learned-modulator wing of this corpus agrees on and we depart from; (ii) a
  **functional-partition** grouping variant (tie units by which downstream head they
  serve, rather than by contiguous index), which is what every biologically motivated
  partition in the corpus actually does.
- **`plan-reviewer`** — the grouping screen's interpretation. §8.5 shows the two scope
  ablations that exist both report *flat*, and §10 item 8 shows the per-neuron baseline
  confound is inherited and untested. A flat result would therefore be the literature's
  expected outcome and would not by itself distinguish "grouping is inert" from "the
  baseline absorbed the signal".
- **`literature-curator`** (this agent, next pass) — apply the corrections in §9 to
  [`../FiLM/film_modulation_granularity_synthesis.md`](../FiLM/film_modulation_granularity_synthesis.md)
  judgment 3 and §6.2, appended as a signed cross-corpus note rather than an in-place
  rewrite.
- **`literature-reviewer`** — no new pull is needed. The corpus is saturated on Axes 1–3.
  The Axis-4 gap (learned grouping) is a gap in the *field*.
