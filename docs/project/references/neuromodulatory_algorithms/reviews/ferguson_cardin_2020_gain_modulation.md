---
title: "Mechanisms underlying gain modulation in the cortex"
authors: "Katie A. Ferguson, Jessica A. Cardin"
year: 2020
venue: "Nature Reviews Neuroscience 21, 80–92"
slug: "ferguson_cardin_2020_gain_modulation"
source_pdf: "sources/Ferguson and Cardin 2020 - Mechanisms underlying gain modulation in the cortex.pdf"
topic: "neuromodulatory_algorithms"
---

## Plain-English entry point

This is a *biology* review article. Unlike the other papers in this corpus, it makes no claims about deep learning. It is included here because it provides the *neuroscientific substrate* that every "neuromodulation in DNN" paper appeals to. The question it asks: in the mammalian cortex, *how, at the level of cells and circuits, does the brain change a neuron's gain*? "Gain" here has a precise meaning — it's the *slope* of a neuron's input–output (I/O) curve, i.e., how strongly the neuron's firing rate responds to a given change in input. A high-gain neuron is sensitive (small input change → big firing-rate change); a low-gain neuron is sluggish. Gain modulation is the act of *changing this slope without changing the neuron's selectivity* — the neuron still prefers the same stimulus, but its responsiveness goes up or down. This is exactly what the Vecoven 2020 NMN, the Ben-Iwhiwhu 2022 NPN, the Mei 2022 "scale 1/2" framework, and the Lee 2024 Doya-DaYu agent all try to *imitate* algorithmically.

Why it matters as part of this corpus: the review distinguishes **multiplicative (gain)** modulation from **additive (offset)** modulation, surveys the cellular mechanisms (synaptic fluctuations, conductance changes, GABAergic inhibition), enumerates the biological "input streams" that drive gain modulation (sensory context, attention, arousal, locomotion, learning, and four classical neuromodulators — acetylcholine, noradrenaline, serotonin, dopamine), and reviews the role of three GABAergic interneuron classes (parvalbumin-positive, somatostatin-positive, vasoactive-intestinal-peptide-positive) as the *circuit-level mediators* of gain modulation. It is the canonical reference for "what is actually happening biologically when we say a neuromodulator modulates gain".

## Section-by-section backbone

### Abstract & Introduction
Gain modulation = active regulation of a neuron's input-sensitivity (the slope of its I/O curve) without altering selectivity. It is driven by attention, arousal, motor activity, neuromodulators. Mechanisms converge on a common set: GABAergic inhibition, synaptically driven $V_m$ fluctuations, conductance changes, biophysical changes. The cortex implements gain modulation through these mechanisms; GABAergic interneurons (especially PV+, SST+, VIP+) are central. The review covers attention, locomotion, arousal, neuromodulation, and the role of interneurons.

### Box 1 — Divisive versus additive modulation
Two orthogonal types of I/O transformation:
- **Multiplicative (gain) modulation**: rescales the I/O slope without changing the rheobase (threshold). Input gain (input axis scaling) vs. response gain (output axis scaling).
- **Additive (offset) modulation**: shifts the I/O curve along an axis without changing slope, *altering* stimulus selectivity. May produce *subtractive* effects through threshold non-linearities ("iceberg effect").

Normalization is a special case of gain modulation where responses are divided by the summed activity of a local population.

### Multiple modes of gain control
Gain is dynamically regulated by sensory context (e.g., contrast-invariant orientation tuning), behavioural state, attention, learning. Key examples:
- **Contrast invariance** in V1 — gain rescaled across contrast levels while orientation selectivity stays fixed.
- **Attention** in primate V1/V4 — both contrast gain and response gain enhanced for attended stimuli.
- **Learning** — gain increases specifically in cells tuned to learned orientation (V1).
- **Locomotion** — increases V1 gain in mice but *decreases* A1 gain — area-specific.

### Cellular mechanisms — Synaptic input regulation
Three mechanisms for gain control at single-cell level:
1. **Synaptically driven $V_m$ fluctuations** — under balanced excitation–inhibition, these create a power-law relationship between mean $V_m$ and firing rate. Tightly regulated by behavioural state (arousal → desynchronisation → wider fluctuations).
2. **Membrane conductance changes** — these produce *additive*, not multiplicative, shifts.
3. **Depolarization state** — also additive.

So multiplicative gain modulation specifically requires changes in $V_m$ fluctuation amplitude, often driven by synaptic inhibition.

### Inhibitory regulation of neural sensitivity
**Three GABAergic interneuron classes:**
- **PV+ (parvalbumin)**: fast-spiking, target perisomatic + axonic regions of pyramidal neurons. Mediate divisive/subtractive gain modulation.
- **SST+ (somatostatin)**: low-threshold spiking, target dendrites. Mediate dendritic gain modulation; involved in adaptation.
- **VIP+ (vasoactive intestinal peptide)**: sparse, dendrite-targeting. Mainly inhibit *other interneurons* (especially SST+), producing disinhibition.

**Disinhibition pathway**: VIP+ → SST+ → pyramidal. Increasing VIP+ activity → reduced SST+ activity → less dendritic inhibition on pyramidal → increased response gain. This is the canonical microcircuit for state-dependent gain enhancement (arousal, locomotion).

### Behavioural state regulation
- Locomotion drives ↑ visual response gain via VIP+ activation in mouse V1.
- Locomotion drives ↓ auditory response gain in A1.
- Pupil dilation (arousal proxy) correlates with locus coeruleus firing.
- Locomotion vs. pure arousal engage different mechanisms (Figure 2c): locomotion → depolarization + ↑ spontaneous + ↑ evoked; pure arousal → ↓ spontaneous but ↑ evoked → ↑ SNR.

### Neuromodulatory control
- **Acetylcholine (ACh)**: from basal forebrain. Acts on nAChRs and mAChRs across excitatory and inhibitory neurons. Multifaceted effects: nicotine ↑ V1 gain in macaque; optogenetic ACh release desynchronizes spiking and ↑ visual perceptual performance without changing overall firing. mAChRs reduce Ca²⁺ channels, ↑ excitability via Ca²⁺-dependent K⁺ channels, enhance bursting. Layer-specific: ACh suppresses excitatory neurons in L4 but enhances them in L2/3 and L5.
- **Noradrenaline (NA)**: from locus coeruleus. β-adrenergic receptors ↑ excitability; α-adrenergic receptors ↓ synaptic excitation. NA reduces spontaneous firing but enhances evoked → ↑ SNR. Locomotion-induced V1 gain increases *require* noradrenergic transmission; blocking NA receptors abolishes them.
- **Serotonin (5-HT)**: predominantly *reduces* gain. Local serotonin in macaque V1 ↓ excitatory response gain. Effects via heterogeneous receptor expression.
- **Dopamine (DA)**: D1 receptors in DLPFC modulate response amplitude and selectivity for working memory; FEF D1 receptors enhance V4 response amplitude/selectivity (top-down attention).

### Interneurons as targets of neuromodulation
The key insight: most neuromodulator effects on gain are *mediated by their effects on interneurons*. VIP+ express nAChRs strongly and are directly activated by cholinergic input. SST+ get mAChR-mediated depolarization. 5-HT acts on 5-HT₃A receptors of L1 and VIP+ interneurons, and on 5-HT₂A receptors of PV+ interneurons. α-adrenergic receptors on SST+ and a subset of PV+ interneurons. So neuromodulators ultimately reshape the *inhibitory landscape*, which then sets the gain of excitatory neurons.

### Functions of gain modulation
Gain modulation enables:
1. **Adaptation across dynamic ranges** — neurons maintain sensitivity to small changes across a wide intensity range.
2. **Stimulus-feature separation** — gain-invariance to intensity allows decoding of other features (orientation, frequency).
3. **Attractor dynamics and winner-take-all** — state-dependent gain increases promote convergence to stable representations under noise.
4. **Computational efficiency** — distributed stimulus representations, optimized decoders, reduced redundancy under resource constraints.

### Future directions / open questions
- Heterogeneity of gain modulation across cell populations and layers.
- Reliability and repeatability of gain modulation across repeated trials.
- Population-level vs. single-cell gain interactions.
- Cellular mechanisms of interactions between neuromodulators (GPCR convergence).

## Phase 1 — Undergraduate-level synthesis

**The key idea.** Every neuron has an input–output curve: as you increase the input current, the firing rate goes up, eventually saturating. The *slope* of that curve is called *gain*. A high-gain neuron is twitchy; a low-gain neuron is muted. Importantly, the *position* of the curve along the input axis is *selectivity* (what the neuron responds to), and gain modulation changes the *slope* without changing *selectivity* — the neuron still likes vertical bars (say), but it's now twitchier or more muted in how strongly it responds to them.

The cortex constantly does this. When you pay attention to something, the neurons coding for it get a gain boost. When you start moving (locomotion), V1 neurons get a gain boost (so you see better while running). When you're highly aroused, the signal-to-noise ratio in your sensory cortex goes up. The cellular machinery: a combination of (a) the noisy "synaptic chatter" around a neuron, which smooths out its firing-rate response and creates a power-law relationship to mean voltage; (b) the *total conductance* of the cell, which can shift the curve additively; and (c) crucially, GABAergic *inhibition*.

The brain uses three main classes of inhibitory interneurons — PV+ (around the soma), SST+ (on the dendrites), and VIP+ (which inhibits other interneurons) — to set the gain of every pyramidal neuron. Neuromodulators (ACh, NA, 5-HT, DA) act in part by targeting these interneuron classes — acetylcholine activates VIP+ which silences SST+ which lifts dendritic inhibition off pyramidal neurons, etc. This is the "disinhibition" circuit that mediates state-dependent gain enhancement.

**The experimental setup (the review surveys many).** Optogenetic activation/inhibition of specific interneuron classes; recording I/O curves of pyramidal cells; behavioural manipulations (locomotion, arousal, attention) paired with electrophysiology; cell-type-specific receptor knockouts.

**The result.** Gain modulation is multiplicative, mediated by GABAergic inhibition (mostly SST+ for dendritic gain, PV+ for somatic), shaped by neuromodulators that act primarily through their effects on interneuron populations. ACh, NA, DA tend to enhance gain (often by disinhibition); 5-HT tends to reduce gain.

**Why this matters for the rest of the corpus.** Every DNN paper in this corpus that uses the phrase "neuromodulation" is implicitly claiming a relationship to *this* biology. Vecoven 2020's $\sigma(z^\top (xw_s + w_b))$ explicitly modulates the *slope* of the activation function — that is, *gain*. Ben-Iwhiwhu 2022's $h_s \otimes h_m$ multiplicative gating is literally response-gain modulation. Mei 2022's four-scale framework explicitly cites multiplicative gain. The Ferguson & Cardin review is what these papers are biologically *trying to be analogous to*.

## Phase 2 — Graduate-level deep dive

### Formal definition — gain modulation vs. additive modulation

Let a neuron's firing rate $r$ be a function of total input current $I$ via I/O function $r = f(I)$. **Multiplicative gain modulation** is

$$r = f(g \cdot I), \qquad g \in \mathbb{R}_{>0}$$

(input gain) or

$$r = g \cdot f(I)$$

(response gain). Selectivity (the *shape* of $f$) is preserved; the slope changes. **Additive modulation** instead shifts the curve:

$$r = f(I + c) \quad \text{or} \quad r = f(I) + c$$

This changes the *position* of the curve and therefore alters the input threshold for any given response — i.e., it changes effective selectivity.

The crucial distinction in network terms: multiplicative modulation enables *multimodal integration* (combining two streams of input through scaling) without altering tuning; additive modulation cannot.

### Iceberg effect and divisive→subtractive conversion

A *divisive* (multiplicative with $g < 1$) modulation of subthreshold $V_m$ can produce a *subtractive* effect on spiking output if the spike threshold $V_{thr}$ is below $V_m$. Imagine cutting $V_m$ by a factor $g$: subthreshold responses to non-preferred stimuli now fall below $V_{thr}$ entirely, sharpening the tuning curve. Mathematically: if $V_m(\theta)$ is the membrane response to stimulus $\theta$, and the spike-rate is $r(\theta) = \lfloor V_m(\theta) - V_{thr} \rfloor_+$, then divisive scaling $V_m \to gV_m$ gives

$$r_g(\theta) = \lfloor gV_m(\theta) - V_{thr} \rfloor_+$$

For stimulus values where $V_m < V_{thr}/g$, $r_g = 0$. The tuning curve over $\theta$ becomes narrower — sharper selectivity.

This is the "iceberg effect" — a single mechanism (divisive synaptic inhibition) produces *both* gain modulation *and* selectivity sharpening, depending on whether you measure subthreshold or spike output.

### Synaptic-fluctuation mechanism

Under balanced excitation/inhibition, synaptically driven $V_m$ fluctuations create a *power-law* relationship between mean $V_m$ and mean firing rate:

$$r \propto (V_m - V_{thr})^\beta_+, \qquad \beta > 1$$

(typically $\beta \approx 2$ from in vivo data). The exponent $\beta$ and the prefactor depend on the *variance* of $V_m$ fluctuations: larger fluctuations → larger effective $\beta$ → steeper gain. The classic result (Hansel & van Vreeswijk 2002; Murphy & Miller 2003; Mitchell & Silver 2003): a global change in synaptic noise level can multiplicatively scale firing-rate responses without shifting thresholds.

This is *the* biological substrate that Vecoven 2020's $z^\top w_s$ effective slope mimics — except Vecoven's mechanism is a direct rescaling of the activation function, not an emergent property of noise.

### GABAergic shunting inhibition and gain

GABAergic synaptic input at or near a neuron's resting potential opens chloride channels with reversal potential close to $V_{rest}$. The synaptic current is small (no driving force), but the *conductance increase* shunts excitatory inputs at the same site, reducing the effective input–output gain of those inputs. Mathematically, the postsynaptic potential from an excitatory input becomes

$$V_{psp} = \frac{I_{exc}}{G_{leak} + G_{inh}}$$

where $G_{inh}$ is the (large) inhibitory conductance. Increasing $G_{inh}$ multiplicatively reduces $V_{psp}$ — divisive gain modulation. Combined with synaptic-fluctuation noise (which provides the power-law), shunting inhibition can produce purely multiplicative gain control (the Chance, Abbott & Reyes 2002 / Mitchell & Silver 2003 mechanism).

### The interneuron map

| Interneuron | Marker | Target | Synaptic dynamics | Gain effect |
|---|---|---|---|---|
| PV+ | Parvalbumin | Perisomatic + axonic | Fast, depressing | Divisive ± subtractive (response gain); fast adapting |
| SST+ | Somatostatin | Dendrites | Slower, sustained | Dendritic divisive; adaptation-mediated gain |
| VIP+ | Vasoactive intestinal peptide | Other interneurons (esp. SST+) | Disinhibitory | Net gain *increase* via SST+ suppression |

### The disinhibition circuit (Figure 1 / canonical)

$$\text{ACh / arousal} \;\to\; \text{VIP+} \;\dashv\; \text{SST+} \;\dashv\; \text{Pyramidal dendrite} \;\to\; \text{response gain ↑}$$

where $\dashv$ denotes inhibition. The signs flip through the chain: more ACh → more VIP+ → less SST+ → less dendritic inhibition on pyramidal → higher pyramidal gain.

### Neuromodulator-specific receptor pharmacology (Table-style summary)

**Acetylcholine:**
- nAChR (nicotinic, ionotropic): strongly depolarises VIP+ interneurons, enhances pyramidal-to-SST+ synapses.
- mAChR (muscarinic, GPCR): depolarises SST+, regulates K⁺ channels in pyramidal neurons (Ca²⁺-dependent K⁺ ↑ excitability; SK channels ↓ L5 activity).

**Noradrenaline:**
- α-adrenergic: depolarises SST+ and subset of PV+ interneurons; reduces synaptic excitation via α2.
- β-adrenergic: increases excitability of expressing pyramidal neurons.
- Locomotion-induced V1 gain *requires* NA — blocking NA receptors hyperpolarises pyramidals and abolishes the gain increase.

**Serotonin:**
- 5-HT₃A on L1 and VIP+ interneurons.
- 5-HT₂A on PV+ interneurons.
- Net effect on V1 excitatory neurons: gain *reduction*.

**Dopamine:**
- D1 in DLPFC: working-memory selectivity.
- D1 in FEF (frontal eye field): enhances V4 response amplitude — top-down attention.

### How the DNN papers in this corpus map onto Ferguson & Cardin

| DNN paper | Biological analogue claimed | Match quality |
|---|---|---|
| Vecoven 2020 (NMN) | $z$ modulates slope/bias of activation = response gain | Tight — directly mimics gain modulation |
| Ben-Iwhiwhu 2022 (NPN) | $h_m$ multiplicative gating of $h_s$ = response gain (with sign-flip) | Tight |
| Mei 2022 (multiscale framework) | All four scales (hyperparameter, cell-type, weight, dendrite) | Comprehensive map |
| Wang 2024 (NeuronML) | Per-task structural mask = "regional activation" | Loose — closer to anatomical pathway selection than cellular gain |
| Lee 2024 (Doya-DaYu) | Hyperparameters set by uncertainty estimates analogous to ACh/NA function | Functional mapping, not cellular-mechanism mapping |
| Wang 2025 (NEST) | Two scalars $\alpha, \beta$ for graph construction | Loose — no per-neuron gain |

Ferguson & Cardin is therefore the *closest fit* as biological reference for Vecoven 2020 and Ben-Iwhiwhu 2022 (the activation-modulation pair) and a *partial fit* for Mei 2022 (which generalises). It's a *weak fit* for Wang 2024, Lee 2024, and Wang 2025, which use the word "neuromodulation" with looser correspondence.

### Critical scrutiny — what the review *cannot* yet tell us

1. **Direction of causality** between single-neuron gain modulation and population-level encoding is not established (the authors flag this explicitly).
2. **Reliability and repeatability** of gain modulation under repeated identical conditions is poorly characterised — most studies are single-trial-averaged.
3. **Interaction of multiple neuromodulators** at the GPCR / second-messenger level (convergence to small numbers of pathways) is largely unexplored.
4. **State-dependent inhibitory dynamics** in inhibition-stabilized networks: theory predicts certain regimes; experimental verification is sparse.
5. **Heterogeneity within cell classes**: not all PV+ cells are alike; not all SST+ cells are alike. Sub-class-specific gain effects need finer-resolution tools.

For any DNN paper claiming biological inspiration from gain modulation, these unresolved issues mean the analogy is at best to a *currently-best-guess* mechanism, not a verified one.

## Connections

This is the *biological foundation* paper for the whole corpus. Every other paper claims, implicitly or explicitly, that its mechanism is "inspired by" or "analogous to" cortical gain modulation. The review provides the substance behind those claims.

- **[vecoven_2020_neuromod_dnn](vecoven_2020_neuromod_dnn.md)** — Vecoven's $\sigma_{NMN}(x, z; w_s, w_b) = \sigma(z^\top (x w_s + w_b))$ is a direct algorithmic analogue of multiplicative response gain modulation (slope rescaling without selectivity change). Ferguson & Cardin provides the cellular-mechanism story for *why* this is biologically meaningful (synaptic fluctuations + shunting inhibition).
- **[ben-iwhiwhu_2022_context_meta_rl](beniwhiwhu_2022_context_meta_rl.md)** — Ben-Iwhiwhu's $h = \text{ReLU}(h_s \otimes h_m)$ multiplicative gating with $h_m \in [-1, 1]$ via $\tanh$ is response-gain modulation with sign flip. Closely matches the biological substrate.
- **[mei_2022_multiscale_neuromod](mei_2022_multiscale_neuromod.md)** — Mei et al. explicitly cite Ferguson & Cardin-style gain modulation as the substrate for their Scale 1 (hyperparameter) and Scale 2 (cell-type-specific) tiers. The cell-type-specific tier is *direct* mapping to PV+/SST+/VIP+ functional differentiation.
- **[lee_2024_lifelong_rl](lee_2024_lifelong_rl.md)** — Lee et al. cite NA's role in cortical gain (Aston-Jones & Cohen) which Ferguson & Cardin reviews in detail. The link to NEST's hyperparameter modulation is via the NA-as-tonic-gain story.
- **[wang_2024_neuromod_meta](wang_2024_neuromod_meta.md)** — looser link; the "brain activates different regions" framing is closer to anatomical pathway selection than to cellular gain modulation.
- **[wang_2025_nest_hypergraph](wang_2025_nest_hypergraph.md)** — loosest link; modulates graph-construction hyperparameters, not neuronal gain.
- **Doya 2002** (in this corpus) — provides the high-level RL-to-neuromodulator mapping that Ferguson & Cardin's cellular review grounds. The two are complementary: Doya at the algorithmic level, Ferguson & Cardin at the circuit level.
- **Aston-Jones & Cohen 2005** — NA adaptive-gain theory; Ferguson & Cardin reviews the cellular evidence supporting it (β-adrenergic ↑ excitability; locomotion-induced V1 gain requires NA).
- **Yu & Dayan 2005** — ACh = expected uncertainty, NA = unexpected uncertainty; Ferguson & Cardin reviews the cellular mechanisms by which ACh and NA exert their effects.
- **Avery & Krichmar 2017** — review of neuromodulatory systems; sister review to Ferguson & Cardin from the systems perspective.
- **Shine et al. 2021** ("Computational models link cellular mechanisms of neuromodulation to large-scale neural dynamics") — explicit attempt to bridge from Ferguson & Cardin cellular detail to large-scale dynamics; a natural follow-up.
