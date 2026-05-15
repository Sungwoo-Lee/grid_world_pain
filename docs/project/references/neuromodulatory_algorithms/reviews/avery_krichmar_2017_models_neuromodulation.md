---
title: "Models of Neuromodulation"
authors: ["Michael C. Avery", "Jeffrey L. Krichmar"]
year: 2017
venue: "Chapter 27 in Computational Models of Brain and Behavior, ed. A.A. Moustafa, John Wiley & Sons"
slug: avery_krichmar_2017_models_neuromodulation
source_pdf: "sources/Avery and Krichmar 2017 - Models of Neuromodulation.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

This is a **review chapter** (not an original empirical paper) from the Krichmar lab summarising the state of computational modelling of the four major ascending neuromodulator systems of the mammalian brain — **dopamine (DA), serotonin (5-HT), acetylcholine (ACh), and noradrenaline (NA)** — and how they jointly support attention, decision making, learning, and memory. It is the natural "lit-review" entry point for any project that wants to invoke a "neuromodulator-inspired" controller, because it lays out **both** the experimental findings (what each system *does* in animals and humans) **and** the canonical computational models (TD learning, actor-critic, Bayesian update, signal-to-noise sharpening) that each system has been mapped to.

The four mappings the chapter advocates:

1. **DA: reward prediction error and "wanting".** Phasic DA bursts after unexpected reward, pause on expected-but-omitted reward — Schultz, Dayan, Montague 1997. Maps directly onto the TD-error term $\delta$ in actor-critic reinforcement learning.
2. **5-HT: temporal discounting, harm-aversion, anxiety, and impulsivity.** Maps onto the discount factor $\gamma$ in TD learning (Doya 2002), and onto a *cost* critic in actor-critic when paired with a DA reward critic. ATD (acute tryptophan depletion) lowers 5-HT and reproducibly increases impulsivity, risk-taking, and harm-aversion failure in humans.
3. **ACh: attentional effort and gain modulation.** Originates in the basal forebrain (substantia innominata / nucleus basalis and medial septum / diagonal band). The basal forebrain has an **incremental** pathway (SI/nBM → cortex; turns attention *on* to relevant stimuli) and a **decremental** pathway (MS/VDB → cortex; turns attention *off* via latent inhibition / habituation). ACh acts as a *gain* knob on top-down inputs — when ACh is high, sensory thalamocortical inputs dominate; when low, intracortical recurrence dominates.
4. **NA: novelty, network reset, and unexpected uncertainty.** Locus coeruleus (LC) fires phasically for unexpected salient events and is the substrate of Yu and Dayan's (2005) "unexpected uncertainty" signal. LC also shows an inverted-U relationship between tonic firing and task performance — too little tonic LC = inattentive; too much tonic LC = distractible; optimal tonic LC + phasic responsiveness = peak performance (the Yerkes-Dodson curve made neural).

Two cross-cutting models the chapter then synthesises:
- **Doya's metalearning theory (2002)** — the cleanest one-modulator-per-parameter mapping: DA = TD error $\delta$, 5-HT = discount factor $\gamma$, ACh = learning rate $\alpha$, NA = inverse-temperature $\beta$ in Softmax action selection.
- **Krichmar's "decisiveness" framework (2008)** — all modulators do the *same* thing downstream (SNR sharpening to make the agent more decisive), and differ only in their *triggers* (cost, surprise, reward, effort).

For the present project, this is the single most useful chapter to read first: it names the modulators, gives the canonical computational interpretations, and points at the rest of the Krichmar-lab evidence corpus.

## Section-ordered backbone

### Introduction
The four ascending modulator systems originate in small subcortical nuclei (thousands of neurons in rodent, tens of thousands in human) and project broadly. They track distinct environmental signals — risks, rewards, novelty, effort, social cooperation — and influence attention, decision making, learning, and memory. Figure 27.1 places the anatomy: VTA / SNc (DA), Raphe (5-HT), Locus Coeruleus (NA), basal forebrain (ACh), with projections to PFC / ACC, neocortex, striatum, hippocampus, amygdala, and nucleus accumbens.

### Dopaminergic System
Schultz et al.'s (1997) canonical result: dopaminergic neurons in VTA and SNc carry a **reward prediction error** signal. With pre-conditioning, an unexpected reward causes phasic bursts. After conditioning, the burst transfers to the predictive cue and the actual reward elicits no net change. If reward is omitted after a predicting cue, the neurons dip below baseline. The bursts activate D1 receptors in the basal-ganglia direct pathway (reinforcement); the dips strengthen the D2 indirect pathway (inhibition). Moustafa et al. (2013) model how levodopa enhances stimulus-response learning while DA agonists impair it, via D1/D2 receptor effects.

The chapter then introduces **TD learning** explicitly with the actor-critic equations (Eq. 27.1, reproduced in Phase 2 below). DA = $\delta$, the TD error. Other modulators have been proposed to map to other TD parameters: 5-HT to learning rate $\alpha$ (Balasubramani et al. 2015), NA to $\alpha$ (Nassar et al. 2012) — already at this point the chapter signals that the mapping is not unique.

The dopamine-circuit subsection notes that VTA/SNc receive input from the lateral habenula (negative-reward signal; Matsumoto and Hikosaka 2007) and the pedunculopontine tegmental nucleus (positive-reward signal; Hong and Hikosaka 2014). DA projects out to striatum, thalamus, amygdala, hippocampus, PFC — making it a "feedback" hub. The chapter then describes the Avery and Krichmar (2015) model of D2 receptors in PFC gating working memory, with the suggestion that improper D2 activation in PFC produces schizophrenia-like noise.

### Serotonergic System
**5-HT and impulsivity.** Doya (2002) proposed 5-HT = temporal-discount $\gamma$. Empirical support: Miyazaki et al. (2011) showed Raphe-5-HT firing tracks rats' willingness to wait for delayed reward; ATD (acute tryptophan depletion) in humans shortens delay-discounting horizons (Schweighofer et al. 2008, Tanaka et al. 2007, 2009).

**5-HT and harm-aversion.** ATD in the Ultimatum Game causes subjects to reject more unfair offers (Crockett et al. 2008). ATD in Go/No-Go and Reinforced Categorization decreases punishment-induced inhibition (Crockett et al. 2009, 2012). The Asher / Zaldivar / Krichmar Hawk-Dove actor-critic (2010, see Fig. 27.4) implements **two critics**: a reward critic (DA) and a cost critic (5-HT). Lesioning the 5-HT critic produces aggressive, hawk-like play; this matches ATD humans.

**5-HT and anxiety.** Manipulations of 5-HT1A and 5-HT2A receptors in mice elevate anxiety in the open-field test (Heisler et al. 1998, Weisstaub et al. 2006). 5-HTTLPR polymorphism is associated with depression risk (Jasinska et al. 2012).

### Dopamine-Serotonin Opponency
Daw, Kakade, and Dayan (2002) and Boureau and Dayan (2011) reviewed empirical and theoretical evidence for DA / 5-HT opposition: DA invigorates reward-seeking, 5-HT withdraws and avoids punishment. Daw et al. (2002) proposed tonic 5-HT = average reward rate, tonic DA = average punishment rate, phasic 5-HT = future-punishment prediction error — but empirical evidence for these mappings has been hard to find. Asher, Zaldivar, and Krichmar (2010) showed that direct opponency is *not* necessary — having different modulators handle different sensory events produces *behavioural* opponency without explicit modulator–modulator inhibition. The Krichmar (2013) neurorobot (`krichmar_2013_neurorobotic_anxiety_curiosity.md` in this corpus) implements the explicit 5-HT→DA inhibition version.

### Cholinergic System
ACh originates in (i) substantia innominata / nucleus basalis (SI/nBM) projecting to neocortex; (ii) medial septum / vertical limb of the diagonal band (MS/VDB) projecting to hippocampus and cingulate. Oros, Chiba, Nitz, and Krichmar (2014) dissociated these two pathways into an **incremental** (SI/nBM-driven, increase attention to relevant stimulus) and **decremental** (MS/VDB-driven, decrease attention to no-longer-relevant stimulus) cholinergic effect. Lesioning MS/VDB disrupted latent inhibition and increased perseveration in extinction (Fig. 27.6).

Deco and Thiele (2011) and Avery, Dutt, and Krichmar (2014) modelled the cellular mechanism: ACh reduces firing-rate adaptation, enhances thalamocortical input, *reduces* lateral intracortical connectivity strength, and increases inhibitory drive. The Avery model adds a thalamic-reticular disinhibition account: basal forebrain disinhibits the sensory thalamus via inhibitory projections to TRN.

### Noradrenergic System
LC originates in the brain stem; projects to nearly every cortical and subcortical region except basal ganglia. LC has *tonic* (slow arousal) and *phasic* (short bursts for salient / novel / task-relevant stimuli) modes. **Inverted-U** between tonic LC and task performance (Aston-Jones and Cohen 2005, Fig. 27.7): too low = inattentive; too high = distractible; optimal = phasically responsive at moderate tonic baseline.

**Network reset.** Bouret and Sara (2005): phasic LC induces a large-scale reconfiguration of brain network activity, "resetting" the agent's task set. Hermans et al. (2011) showed stress (which activates LC) reconfigures functional connectivity, and the reconfiguration is dampened by adrenergic-receptor blockade.

**Unexpected uncertainty.** Yu and Dayan (2005) proposed phasic LC bursts encode *unexpected* uncertainty — a "surprise" signal that says "the prior must be updated, not just refined". Avery, Nitz, Chiba, and Krichmar (2012) implemented a neural-network version where NA surprise modulates Hebbian learning rate *and* gain on sensory input.

### Universal Models of Neuromodulation

**Doya's metalearning (2002)**. The neat one-modulator-per-parameter scheme: TD error $\delta$ = DA; discount factor $\gamma$ = 5-HT; learning rate $\alpha$ = ACh; inverse-temperature $\beta$ (exploration-exploitation) = NA. Equations 27.2–27.4 reproduced below.

**Krichmar's decisiveness framework (2008)**. All four modulators share the same downstream effect — *sharpen SNR in cortical-thalamic circuits so the agent becomes decisive* — but differ in their triggers (DA = reward, 5-HT = cost, ACh = effort, NA = surprise; see Fig. 27.8A). Phasic modulator activity → exploitive / decisive; tonic / absent modulator activity → exploratory / curious. The Cox–Krichmar 2009 neurorobot (`cox_krichmar_2009_neuromodulation_robot_controller.md`) tested this framework empirically.

### Conclusions
Most models still focus on one or two modulators. Future models should consider the interactions. Receptor-expression surveys (Zaldivar and Krichmar 2013, using Allen Mouse Brain Atlas) show that the SI/nBM and VTA carry receptors for *all four* modulators — implying that the modulators are themselves modulating each other, not operating in parallel.

## Phase 1 — Undergraduate-level synthesis

**The four-modulator orchestra.** The mammalian brain has four big "spray" systems — small clumps of neurons deep in the brain that release a chemical signal across most of the cortex. They are the **dopaminergic** (DA), **serotonergic** (5-HT), **cholinergic** (ACh), and **noradrenergic** (NA) systems. Each has a job, but they all share a similar trick: when they fire, they *sharpen* the brain's response to whatever is happening right now.

**What each modulator does.**
- **Dopamine** is the *reward surprise* signal. When something unexpectedly good happens, dopamine spikes; when an expected good thing fails to happen, dopamine dips. This dual response is exactly the **reward prediction error** in reinforcement learning. Dopamine is also the "wanting" signal — the urge to chase a reward, not the pleasure of getting it.
- **Serotonin** is the *patience, caution, and anxiety* signal. Animals with low serotonin (e.g., humans given a tryptophan-depletion drink) become impulsive, risk-taking, and harm-insensitive. Serotonin sets the brain's *willingness to wait* (the discount factor in RL) and its *willingness to absorb a cost* for a delayed reward.
- **Acetylcholine** is the *attention effort* signal, with two flavours. One pathway (from the nucleus basalis) turns attention *on*; another (from the medial septum) turns attention *off*. Together they let the brain pay attention to new things and let go of stale things.
- **Noradrenaline** is the *novelty and surprise* signal. The locus coeruleus (a tiny brainstem nucleus) fires phasically when something unexpected happens, and this firing causes a brain-wide "reset" — old habits are suspended, the brain becomes receptive to new patterns. Too much tonic noradrenaline and you're a distractible mess; too little and you're zoned out; just right and you're sharp and responsive — the famous "Yerkes-Dodson inverted-U".

**The two big theories of how the four fit together.**

*Doya 2002.* Reinforcement learning has four parameters. Each modulator owns one:
- $\delta$ (TD error) = DA
- $\gamma$ (discount factor) = 5-HT
- $\alpha$ (learning rate) = ACh
- $\beta$ (exploration / exploitation) = NA

*Krichmar 2008 (preferred by this chapter's authors).* All four modulators do the same thing — *sharpen the signal-to-noise ratio in their downstream targets* — but they fire for different reasons (DA for reward, 5-HT for threat, ACh for effort, NA for novelty). When a modulator fires phasically, the brain becomes decisive; when modulators are quiet (tonic-only), the brain is exploratory.

**The worked example.** The Hawk-Dove game (Asher, Zaldivar, Krichmar 2010). Two players. Each picks "escalate" (try to seize the resource) or "display" (cooperate). If both escalate, both get hurt. If one escalates and one displays, the escalator wins everything. If both display, they share. The Krichmar architecture uses two critics — a DA *reward* critic that tracks expected gain, and a 5-HT *cost* critic that tracks expected injury. With both critics intact, the simulated agent reaches a balanced strategy. **Lesion the 5-HT critic** and the agent becomes pure hawk — it escalates everything, just like ATD-treated humans in the Ultimatum Game.

## Phase 2 — Graduate-level deep dive

### Temporal-Difference (TD) learning and the actor-critic mapping

The chapter reproduces the standard actor-critic in Eq. 27.1:

$$
\delta_t = r_{t+1} + \gamma \, V(s_{t+1}) - V(s_t),
$$

$$
V(s_t) \leftarrow V(s_t) + \alpha \, \delta_t,
$$

$$
P(a_t \mid s_t) \leftarrow P(a_t \mid s_t) + \alpha^{*} \, \delta_t,
$$

where $r_{t+1}$ is the observed reward at time $t+1$, $V(s_t)$ is the state value, $\gamma$ is the discount factor, $\alpha$ is the learning rate (for $V$), and $\alpha^*$ is the policy learning rate. The TD error $\delta_t$ is the **reward prediction error** that Schultz et al. (1997) identified as the dopamine signal: positive when reward exceeds prediction, negative when reward falls short.

### Doya's metalearning identification

In the metalearning interpretation (Doya 2002), the chapter restates four equations:

$$
\delta_t = r_t + \gamma \, V(s_{t+1}) - V(s_t) \quad \text{(DA)},
$$

$$
\Delta w_t = \alpha \, \delta_t \, \nabla_{w} V(s_t) \quad \text{(ACh sets } \alpha\text{)},
$$

$$
p(a_t \mid s_t) = \frac{\exp(\beta \, Q(s_t, a_t))}{\sum_{i=1}^{N} \exp(\beta \, Q(s_t, a_i))} \quad \text{(NA sets } \beta\text{)},
$$

with **5-HT** identified with the discount factor $\gamma$ (in the first equation) — so that low 5-HT collapses to low $\gamma$, i.e., short planning horizon, i.e., impulsivity. Note that the Softmax inverse-temperature $\beta$ is the noradrenergic "exploration/exploitation" knob: high $\beta$ → sharp Softmax → exploitive; low $\beta$ → flat Softmax → exploratory. This is consistent with the LC inverted-U: optimal performance at intermediate $\beta$.

### Why the Doya mapping is not the only one

The chapter explicitly flags that two papers map the *same* learning-rate parameter $\alpha$ onto different modulators:
- **Balasubramani et al. (2015)** maps $\alpha$ onto 5-HT.
- **Nassar et al. (2012)** maps $\alpha$ onto pupil-linked NA arousal.

Both have empirical support. The chapter's reading: the mapping of modulator-to-parameter is task- and context-dependent, not a fixed identity.

### The two-critic actor-critic with explicit 5-HT cost

The Hawk-Dove actor-critic (Asher, Zaldivar, Krichmar 2010; Zaldivar et al. 2010) has

$$
\delta_R(t) = r(t) + \gamma \, V_R(s_{t+1}) - V_R(s_t),
$$

$$
\delta_C(t) = c(t) + \gamma \, V_C(s_{t+1}) - V_C(s_t),
$$

where $V_R$ is the reward value (DA / VTA) and $V_C$ is the cost value (5-HT / Raphe). The actor's policy is updated by *both* errors, with weighting set by the relative tonic levels of DA and 5-HT. Lesion of $V_C$ (5-HT) removes cost sensitivity and produces pure hawk play; lesion of $V_R$ (DA) removes reward sensitivity and produces pure dove play. The architecture is the cleanest formal statement of **dopamine-serotonin opponency without direct inhibition** — the opponency is functional, not anatomical.

### Cholinergic gain on top-down input

Deco and Thiele (2011) and Avery et al. (2014) propose ACh acts on cortical circuits via:
- Reduction of firing-rate adaptation in pyramidal neurons.
- Enhancement of thalamocortical (extrinsic) input.
- Reduction of intracortical (intrinsic) recurrent excitation.
- Increase in inhibitory drive (via muscarinic receptors on inhibitory interneurons).

Mathematically the net effect can be written as a gain ratio: if $I_{\text{TC}}$ is thalamocortical drive and $I_{\text{IC}}$ is intracortical recurrence, then ACh increases $I_{\text{TC}} / I_{\text{IC}}$, which in a competitive-attractor cortical model means external input dominates over internal pattern completion — i.e., the cortex listens to the world rather than to its own ongoing thought. This is structurally identical to the modulator-gating trick in Cox and Krichmar 2009, where neuromodulator activity multiplies *extrinsic* inputs by 10 and leaves intrinsic inputs at 1.

### Yu-Dayan unexpected-uncertainty model

Yu and Dayan (2005) propose:
- **Expected uncertainty** ($\sigma_E$): known variability within the current model. Tracked by ACh.
- **Unexpected uncertainty** ($\sigma_U$): the model is wrong, the world has changed. Tracked by NA.

In their Bayesian inference scheme, expected uncertainty drives ordinary Bayesian updating of the parameters of the current model, while unexpected uncertainty triggers a *change of model* — a network reset. Avery et al. (2012) implement this as a neural network where an NA surprise signal modulates both the Hebbian learning rate and the gain on sensory inputs. **Computational reading**: NA is not a TD error; it is a *meta*-level signal that says "the TD machinery is using the wrong state representation, switch context".

### The Krichmar 2008 decisiveness framework

The unifying claim: the *common downstream effect* of all four modulators is

$$
\text{SNR sharpening at the cortical level} \;\Rightarrow\; \text{decisive winner-take-all action selection}.
$$

The *triggers* differ: DA for value-laden positive events, 5-HT for value-laden negative events, ACh for attentional-effort demand, NA for novelty / surprise. Tonic modulation sets the exploration-exploitation baseline; phasic modulation produces *moment-of-decision* exploitation. The Cox–Krichmar (2009) neurorobot is the engineering instantiation.

### Relevance flag for the present project

For a pain-modulated RL grid-world project this chapter is the **canonical reference list**. Concrete takeaways:
1. **Choose your mapping deliberately.** A pain-modulated agent can plausibly be read as (i) modulating $\alpha$ (faster updates when in pain — ACh-like), (ii) modulating $\gamma$ (shorter horizon when in pain — 5-HT-like), (iii) injecting a TD-error scaling (pain *is* a TD-error contribution — DA-like with negative sign), or (iv) modulating $\beta$ (lower exploration when in pain — NA-like inverted-U). The chapter shows that *each of these has empirical / modelling precedent in the published literature*, and that the field has not converged on one mapping.
2. **The two-critic actor-critic is the clean architecture.** A separate reward critic and cost / pain critic, with the policy weighted by the relative tonic levels, is the design pattern with the most explicit modelling history.
3. **Phasic vs. tonic dynamics matter.** Phasic = decision-sharpening; tonic = context / regime-setting. Any modulator dial in the present project should have both an event-driven spike component and a slowly-evolving baseline component. (Krichmar 2013 — `krichmar_2013_neurorobotic_anxiety_curiosity.md` — is the engineering exemplar of this split.)
4. **The "inverted-U" empirical pattern** for tonic NA is a likely shape for tonic pain modulation too — a small amount of pain may sharpen attention to relevant stimuli (adaptive), while too much pain may produce distractibility / chronic-pain phenotypes (maladaptive). Worth designing a parameter sweep that tests this shape.

## Connections

- **`cox_krichmar_2009_neuromodulation_robot_controller.md`** — the empirical instantiation of the "decisiveness" framework reviewed here.
- **`krichmar_2013_neurorobotic_anxiety_curiosity.md`** — the empirical instantiation of the DA / 5-HT opponency and frontal cognitive-control reviewed here.
- **`chiba_krichmar_2020_self_monitoring.md`** — the explicit follow-up extending self-monitoring to internal states.
- **`doya_2002_metalearning_neuromodulation.md`** (in this corpus) — the theoretical anchor: the one-modulator-per-parameter mapping the chapter discusses as "Doya 2002".
- **Yu and Dayan (2005)** ("Uncertainty, neuromodulation, and attention", *Neuron*) — the expected / unexpected uncertainty paper this chapter cites repeatedly. The project's separate uncertainty corpus (`docs/project/references/uncertainty/`) intersects this chapter heavily.
- **Aston-Jones and Cohen (2005)** ("An integrative theory of locus coeruleus-norepinephrine function") — the LC inverted-U / adaptive-gain reference. The single most important NA reference for the field.
- **Schultz, Dayan, Montague (1997)** — the canonical DA = TD error paper. Cited in every model in this chapter.
- **`xing_krichmar_2020_patience_navigation.md`**, **`xing_krichmar_2022_environment_neuromodulation.md`**, **`zou_krichmar_2020_neuromodulated_attention.md`** (corpus) — Krichmar-lab successors that build on this chapter's framework in modern RL settings.
- **`ben-iwhiwhu_2022_context_meta_rl_neuromodulation.md`**, **`vecoven_2020_neuromodulation_deep_nn.md`**, **`wang_2024_neuromodulated_metalearning.md`**, **`lee_2024_lifelong_rl_neuromodulation.md`** (corpus) — modern deep-learning instantiations of the same identification scheme.
- **Cañamero school (`khan_canamero_2022_social_buffering.md`, `lharidon_canamero_2023_stress_pain.md`)** — parallel European programme on hormone-modulated action selection; uses cortisol / oxytocin rather than DA / 5-HT but the design pattern (modulator gating action-selection threshold) is shared.
