---
title: "A neurorobotic platform to test the influence of neuromodulatory signaling on anxious and curious behavior"
authors: ["Jeffrey L. Krichmar"]
year: 2013
venue: "Frontiers in Neurorobotics, vol. 7, art. 1"
slug: krichmar_2013_neurorobotic_anxiety_curiosity
source_pdf: "sources/Krichmar 2013 - A neurorobotic platform to test the influence of neuromodulatory signaling on anxious and curious behavior.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

This paper builds on **Cox and Krichmar (2009)** but tackles a sharper question: *what computational mechanism causes a real robot to behave anxiously in a new environment, gradually get curious, but flip back to anxious after a startling event — and to do all of that with a small set of dials that look like neuromodulator levels?*

The platform is an **iRobot Create** (the Roomba dev kit), nicknamed **CarlRoomba**, in a roughly 3.7 m² plywood arena with a cardboard "novel object" in the middle and a docking-station "nest" in one corner. The arena is dark; halfway through each 4-minute trial the room lights flash on for 10 s (a stressor proxy, since rodents prefer the dark). This is the classic rodent **open-field test** of anxiety — robotised.

The controller is a small neural network with five kinds of region:
- **Event neurons** (binary): Object detected (laser), Light detected (camera brightness), Bump (contact).
- **State neurons**: WallFollow, FindHome, OpenField, ExploreObject (four canned behaviours).
- **Dopamine (DA)** neuron — fires for objects and bumps; codes reward / novelty / curiosity.
- **Serotonin (5-HT)** neuron — fires for lights and bumps; codes risk / anxiety / harm-aversion.
- **ACh/NE** neurons — one per event type; act as an *attentional filter* that habituates to repeated events but lets surprising events through.
- **Frontal cortex** neurons — split into **OFC** (orbitofrontal, projects to DA with inhibition, gates curious-state neurons) and **mPFC** (medial prefrontal, projects to 5-HT with inhibition, gates anxious-state neurons).

The new modelling moves relative to Cox and Krichmar 2009: (i) **tonic** modulator levels are added (long-lasting context-setting signals, not just phasic bursts); (ii) **5-HT inhibits DA** (serotonergic-dopaminergic opponency); (iii) **OFC and mPFC inhibit their respective modulators** as a cognitive-control "this event has been handled" feedback. The experiments are then four parameter manipulations — high 5-HT, low 5-HT, high DA, low DA — plus lesions of the OFC→DA and mPFC→5-HT projections. Each manipulation produces a robot phenotype that maps qualitatively onto a known rodent or human pharmacology (e.g., 5-HTTLPR short allele → CarlRoomba with high tonic 5-HT; ATD-treated humans → CarlRoomba with low tonic 5-HT and elevated curiosity).

The take-away: a **two-modulator opponency + a frontal cognitive-control gate** is enough to reproduce the open-field "wary → curious → wary-after-stress → curious-again" phenotype, and to predict pharmacological / lesion phenotypes that match published rodent and human data.

## Section-ordered backbone

### Abstract
The vertebrate neuromodulatory systems are critical for value-laden responses. DA controls reward / curiosity; 5-HT controls anxiety / harm-aversion; ACh and NE filter noise. Frontal cortex exerts cognitive control over the modulators. A neural network embedded on an iRobot Create is tested in an open-field paradigm and compared qualitatively to rodent behaviour. The model tests the hypothesis that high 5-HT produces withdrawn behaviour by suppressing DA, that high DA or low 5-HT produces curiosity, and that frontal control of the modulators is necessary for coping with stress.

### Introduction
Frontal cortex projects to all four classical modulators (Briand et al. 2007). The orbitofrontal cortex (OFC) projects to the DA system (VTA) and monitors expected reward; the medial prefrontal cortex (mPFC) projects to the 5-HT system (Raphe) and monitors expected cost / stress (Jasinska et al. 2012). Previous Krichmar work (2012; Cox and Krichmar 2009) had only phasic modulators; this paper adds tonic dynamics, 5-HT→DA inhibition, and frontal→modulator inhibition.

### Methods — Robot control
iRobot Create + URG-04-LX laser range finder + netbook running Matlab. Three event detectors: object (laser, 12–30° wide, <1 m), light (camera brightness > 50%), bump (Create bump sensors or laser <20 cm). Four canned behaviours: WallFollow, FindHome (covers and docks to Roomba dock), OpenField (drive toward openest region of the laser scan), ExploreObject (drive toward narrow laser feature). Simulation cycle ≈ 1 s.

### Methods — Neural simulation
Three event neurons (binary), four state neurons, one DA neuron, one 5-HT neuron, three ACh/NE neurons (one per event), state-set OFC and mPFC neurons. Connectivity:
- Each OFC and mPFC neuron has all-to-all excitatory (+1.0) and inhibitory (−1.0) connections to every other frontal neuron.
- OFC for OpenField and ExploreObject → DA with weight −1.0 (inhibitory).
- mPFC for WallFollow and FindHome → 5-HT with weight −1.0 (inhibitory).
- Modulator → frontal weights = 5.
- Event → modulator weights = 0.5; event → AChNE weights = 1.
- Specific assignments: Object → DA (novelty / wanting), Light → 5-HT (risk), Bump → both DA and 5-HT (ambiguous valence).
- 5-HT → DA weight = −1.0 (opponency).

All neurons except event neurons use a logistic sigmoid activation. Synaptic input has four parts: a baseline $b$, the weighted summed presynaptic activity, a self-persistence term $p \cdot n_i(t-1)$, and a tonic modulator term. ACh/NE projects multiplicatively onto the frontal lateral inhibitions and onto the modulator→frontal connections (the attentional-filter mechanism). ACh/NE weights from events are *depressing* — each event makes the wire weaker, and weights return slowly to baseline (time constant $\tau = 25$).

Tonic levels of DA and 5-HT facilitate (grow) every time an event passes the ACh/NE filter, and otherwise return slowly toward unit baseline. Initial tonic 5-HT = 2.0, initial tonic DA = 1.0 (so CarlRoomba starts the trial anxious).

Action selection: the most active state neuron is chosen if its activity > 0.67 (which produces a state switch roughly every 12–15 s); otherwise the previous state continues.

### Methods — Experimental paradigm
3.7 m² open-field arena, dark, cardboard object in centre, dock in corner. CarlRoomba placed at the dock facing the centre. Trial runs 240 s; at ≈ 120 s the lab lights flash on for 10 s (the stressor). Each parameter setting run 5 times with different RNG seeds.

### Results — Control condition (intact model)
$\tau_{\text{DA}} = \tau_{5\text{HT}} = 50$. Behaviour: anxious early (WallFollow, FindHome), increasingly curious by ≈ 60 s, anxiety spike at the 120 s light flash, recovery to curious behaviour by ≈ 200 s. AChNE habituates to repeated bump events. This matches the Fonio et al. (2009) rodent progression: nest → wall → centre.

### Results — High 5-HT
$\tau_{5\text{HT}} = 150$ (5-HT decays much more slowly). CarlRoomba responds normally to the light flash but **never recovers** — anxiety dominates the remainder of the trial. Maps onto 5-HT1A-knockout mice (Heisler et al. 1998), which have elevated 5-HT and spend less time in the centre.

### Results — Low 5-HT
$\tau_{5\text{HT}} = 1$. Phasic 5-HT response to light still occurs, but tonic 5-HT decays immediately. Object and bump events after the stressor trigger exploration. Maps onto acute tryptophan depletion (ATD) in humans (Crockett et al. 2008; Robinson et al. 2010): reduced harm aversion, increased risk taking.

### Results — High DA
$\tau_{\text{DA}} = 150$. Stress response still present (5-HT still spikes on light), but CarlRoomba is much quicker to return to curiosity, and ventures into the centre even during/right after the stressor. Maps onto cocaine in rats (Carey et al. 2008): elevated locomotion and novel-object exploration.

### Results — Low DA
$\tau_{\text{DA}} = 1$. CarlRoomba's 5-HT system dominates; behaviour stays withdrawn (wall-follow / find-home) with occasional object-driven curiosity flashes. Bump events trigger anxiety, not exploration. Maps onto reduced effort-for-reward in DA-blocked rats (Denk et al. 2005) and the COMT-Met polymorphism risk-aversion in humans (Roussos et al. 2008).

### Results — mPFC→5-HT lesion
Removes the cognitive-control "this stressor is handled" inhibition of 5-HT. After the light flash, anxious behaviour dominates the whole remainder of the trial. Maps onto Amat et al. (2005): mPFC inactivation in rats during tailshock loses the ability to control the stress response.

### Results — OFC→DA lesion
Removes the analogous cognitive-control inhibition of DA. CarlRoomba perseverates in curious behaviour and in 50% of trials does not respond to the light flash at all. The asymmetry — DA lesion abolishes only some, while 5-HT lesion abolishes all — is attributed to the 5-HT→DA inhibition, which keeps DA in check even when OFC is gone.

### Discussion
Three sub-themes:
1. **5-HT and risk-averse behaviour** — long-tonic-$\tau$ 5-HT reproduces the short-allele 5-HTTLPR / 5-HT1A-knockout anxiety phenotype; short-tonic-$\tau$ 5-HT reproduces the ATD reduced-harm-aversion phenotype.
2. **DA and risk-taking** — long-tonic-$\tau$ DA reproduces cocaine-induced novelty-seeking; short-tonic-$\tau$ DA reproduces COMT-Met cautious-decision phenotype.
3. **Frontal cortex and cognitive control** — OFC and mPFC each provide a "this event has been handled" inhibitory feedback to their target modulator; the asymmetric effect of their lesions is explained by the 5-HT→DA opponency.

### Related work
Cited contrast: Doya 2002 / 2008 mapping of four modulators to four TD-learning parameters (learning rate, temperature, discount factor, TD error). The Krichmar architecture does not implement TD learning explicitly; it uses tonic modulator dynamics + frontal gating instead.

## Phase 1 — Undergraduate-level synthesis

**The setup.** A Roomba in a 4 m² plywood box. There is a docking station in one corner (its "nest") and a cardboard pillar in the middle (a "novel object"). The room is dark. Four minutes per trial. Halfway through, the lab lights flash on for ten seconds — that is the stressor.

**The brain.** A tiny neural network. Two modulator neurons matter most:
- **Dopamine (DA)** — gets excited by interesting objects. When DA is high, the Roomba is curious: it drives into the open area, approaches the cardboard pillar, investigates.
- **Serotonin (5-HT)** — gets excited by scary stimuli (the light flash, an unexpected bump). When 5-HT is high, the Roomba is anxious: it follows walls, finds its dock, hides.

**Two extra ingredients on top of Cox and Krichmar 2009.**
1. Each modulator has a *tonic* level — a long-lasting average. A burst of 5-HT during the light flash slowly raises the tonic 5-HT background, which biases all subsequent decisions toward anxiety until the background decays away.
2. A piece of "frontal cortex" sits above each modulator and **inhibits it** after the event is handled. So when the Roomba has already turned away from the light, the mPFC says "OK, threat dealt with" and shuts off the 5-HT — restoring curiosity.

**The four pharmacology dials.** By changing only the *decay time constant* of one modulator, the authors mimic four real human / rodent conditions:
- Slow 5-HT decay → like a person with the anxiety-prone 5-HTTLPR allele → Roomba stays scared after the light, never recovers.
- Fast 5-HT decay → like tryptophan-depleted humans → Roomba shrugs off the light and explores even during the flash.
- Slow DA decay → like a person on cocaine → Roomba is dominantly curious; explores even during the stressor.
- Fast DA decay → like the COMT-Met polymorphism in humans → Roomba is cautious; 5-HT dominates and it never leaves the wall.

**The two lesions.** Cut the mPFC→5-HT wire (remove "the stressor was handled" feedback) → Roomba is permanently anxious after the flash. Cut the OFC→DA wire (remove "the curiosity was satisfied" feedback) → Roomba is permanently curious and stops noticing the flash on half the trials.

**The big idea.** A small set of *time constants* on opposing modulators, plus a *frontal "event handled" gate*, is enough to capture a rich behavioural repertoire and to make the right predictions across multiple real pharmacological / genetic interventions in animals and humans.

## Phase 2 — Graduate-level deep dive

### Sigmoidal activation

For all non-event neurons:

$$
n_i(t) = \frac{1}{1 + e^{-g \, I_i(t)}},
$$

with gain $g = 2$ for frontal-cortex and modulator neurons, $g = 10$ for ACh/NE (sharper response). Event neurons are binary.

### Synaptic input with persistence and tonic modulator

$$
I_i(t) = b + \sum_j n_j(t) \, w_{ji}(t) + p \, n_i(t-1) + \mathrm{tonic}_{nm}(t),
$$

with baseline $b = -1$ for DA and 5-HT, $b = -0.5$ for ACh/NE, $b \sim \mathrm{Uniform}(-1, 0)$ (per time step) for OFC/mPFC. The persistence $p = 0.25$ for frontal, $p = 0.5$ for ACh/NE, $p = 0$ for DA and 5-HT (modulators have no self-persistence; tonic stores the slow component instead). The tonic term is only present for DA and 5-HT.

### ACh/NE attentional-filter modulation of frontal input

For frontal-cortex neurons only, an extra contribution is added to $I_i$:

$$
I_i(t) \mathrel{+}= \sum_j \mathrm{AChNE}(t-1) \, n_{\text{fctx},j}(t-1) \, w_{\text{inh}, ji}(t-1) + \sum_k \mathrm{AChNE}(t-1) \, n_{\text{nm},k}(t-1) \, w_{\text{nm}, ki}(t-1),
$$

where $\mathrm{AChNE}(t)$ is the *sum* of all activity in the ACh/NE neurons, $n_{\text{fctx},j}$ are other frontal neurons, $n_{\text{nm},k}$ are modulator neurons, $w_{\text{inh}}$ are lateral inhibitions, and $w_{\text{nm}}$ are modulator-to-frontal weights. This is the cleanest formal statement of the "attentional gating" trick: **ACh/NE multiplicatively gates both the lateral competition within frontal cortex *and* the strength of modulator input into it**. When AChNE is high (a novel event just fired), the lateral inhibitions are strong (sharp winner-take-all in frontal) and the modulator influence on frontal is strong (the value-laden bias is dominant). When AChNE is habituated (a stale repeated event), the frontal lateral inhibitions weaken and the modulator influence is muted — i.e., the agent ignores stale events.

### Event-to-AChNE depressing synapse

$$
w_{ji}(t) = \begin{cases}
p \cdot w_{ji}(t-1), & \text{if } e_j = 1,\\[2pt]
w_{ji}(t-1) + \dfrac{1 - w_{ji}(t-1)}{\tau}, & \text{otherwise},
\end{cases}
$$

with depression factor $p = 0.25$ per event and time constant $\tau = 25$. Each time the event fires, the weight is *multiplied* by 0.25 (a quarter); between events, it slowly relaxes back toward 1 with time constant 25 cycles. This implements rate-dependent habituation: a high-rate event drives its AChNE weight to zero (no longer attention-getting); an isolated rare event sees an almost-full weight (highly attention-getting).

### Tonic-modulator dynamics

$$
\mathrm{tonic}_i(t) = \begin{cases}
p \cdot \mathrm{tonic}_i(t-1), & \text{if } \mathrm{AChNE}_j > 0.5,\\[2pt]
\mathrm{tonic}_i(t-1) + \dfrac{1 - \mathrm{tonic}_i(t-1)}{\tau}, & \text{otherwise},
\end{cases}
$$

with **facilitation factor $p = 1.25$** (note: $>1$, the opposite sign of the AChNE rule) and time constant $\tau$ — *this* is the dial the four pharmacology conditions tune. When a salient event passes the AChNE filter (AChNE > 0.5), the tonic level for the modulator is multiplied by 1.25, raising the long-term baseline. Between events, the tonic level relaxes back toward 1 with time constant $\tau$. So:
- Large $\tau$ → modulator carries its boost for a long time → context-setting bias persists (anxious or curious phenotype "stuck").
- Small $\tau$ → modulator boosts decay rapidly → behaviour returns to neutral quickly after each event.

The four phenotype experiments are pure $\tau$-sweeps:
- **Control**: $\tau_{\text{DA}} = \tau_{5\text{HT}} = 50$. Balanced.
- **High 5-HT**: $\tau_{5\text{HT}} = 150$. Slow 5-HT decay → permanent anxiety after the light flash.
- **Low 5-HT**: $\tau_{5\text{HT}} = 1$. Fast 5-HT decay → curiosity dominates.
- **High DA**: $\tau_{\text{DA}} = 150$. Slow DA decay → curiosity dominates even during the stressor.
- **Low DA**: $\tau_{\text{DA}} = 1$. Fast DA decay → 5-HT dominates by default → anxiety.

### The 5-HT→DA opponency and frontal cognitive-control gating

The two pieces that make this model qualitatively different from Cox and Krichmar 2009 are:
1. **5-HT directly inhibits DA** (weight −1.0), so a 5-HT spike pulls DA down. This means a stressor (light flash) doesn't just *add* anxiety; it *subtracts* curiosity. The behavioural switch is therefore sharp.
2. **Frontal cortex inhibits its target modulator after handling the event** (mPFC→5-HT and OFC→DA, both weight −1.0). This is the "cognitive control" closed-loop. Mathematically it is a negative-feedback regulator on tonic levels — once the event has been routed into the appropriate state neuron (WallFollow / FindHome for 5-HT; OpenField / ExploreObject for DA), the corresponding frontal neuron fires, which subtracts from the modulator's input current, which lowers the modulator's tonic. The closed-loop reduces persistence of the stress / curiosity drive *exactly when* the agent has already responded to it.

### Action selection threshold

The maximally active state neuron is selected if its activity > 0.67, else the current state persists. This threshold was tuned so that new behaviours are selected roughly 4–5 times per minute of trial time.

### The asymmetric lesion result and what it predicts

Lesion mPFC→5-HT removes the cognitive-control inhibition on 5-HT. Tonic 5-HT runs away after the stressor, and because 5-HT directly inhibits DA, curiosity is *also* suppressed. The robot is permanently anxious. Lesion OFC→DA removes the cognitive-control inhibition on DA, but the 5-HT system can still inhibit DA reactively when a stressor arrives, so on roughly half of trials the stressor still produces a brief anxiety phase. The author predicts this asymmetry directly from the architectural difference: **mPFC sits above the upstream of the opponency; OFC sits above the downstream of it**.

### Relevance flag for the present project

For a pain-modulated RL agent in a grid world this paper is the cleanest existing demonstration of:
1. **Tonic vs. phasic modulator separation** — short-time-constant phasic spike for the event itself, long-time-constant tonic background for the *context-setting* bias that biases future decisions.
2. **Opponency between two modulator channels** — anxiety vs. curiosity, with one channel multiplicatively inhibiting the other. Directly translatable to "pain vs. reward" or "interoceptive distress vs. extrinsic reward".
3. **Cognitive-control gating from a higher area** — a closed-loop subtraction that *terminates* a tonic boost when the agent has acted on it. This is the design pattern that prevents the modulator from running away after the eliciting event has been handled.
4. **Sweep-of-time-constants methodology** — varying only $\tau$ on one channel reproduces published rodent and human pharmacological / genetic phenotypes. The methodological lesson is that a *small parameter sweep on the modulator dynamics* is the right diagnostic for an affective / interoceptive architecture, not a wholesale architecture change.

## Connections

- **`cox_krichmar_2009_neuromodulation_robot_controller.md`** — the predecessor. Cox and Krichmar 2009 provided the BCM-gated learning rule and the SNR-sharpening interpretation; the present paper adds tonic dynamics, opponency, and frontal cognitive-control.
- **`avery_krichmar_2017_models_neuromodulation.md`** — the review paper from the same lab; this 2013 paper is one of the canonical exemplars discussed there.
- **`chiba_krichmar_2020_self_monitoring.md`** — Krichmar's later self-monitoring framework; extends cognitive control to internal-state monitoring.
- **Krichmar 2012** ("A biologically inspired action selection algorithm based on principles of neuromodulation", IJCNN) — the immediate predecessor cited as Krichmar 2012; introduced the AChNE attentional filter and depressing-event-weight rule used here.
- **Doya 2002** ([`doya_2002_metalearning_neuromodulation.md`](doya_2002_metalearning_neuromodulation.md), in this corpus) — alternative theoretical mapping of the four modulators to four TD-learning parameters; the Krichmar architecture is the structural rival.
- **Khamassi et al. 2011** (cited in Discussion) — combined DA RL with noradrenergic exploration parameter on neurorobots; a closely related programme to this one.
- **Cañamero school papers** in this corpus (`khan_canamero_2022_social_buffering.md`, `lharidon_canamero_2023_stress_pain.md`, `scarinzi_canamero_2022_affective_interactions.md`) — the parallel European programme on hormone-modulated action selection. The structural commitments (gain modulation, threshold modulation, hysteresis) are the same; the *labels* (cortisol / oxytocin vs. 5-HT / DA) and the *triggers* (homeostatic deficit vs. exteroceptive value) differ.
- **`zou_krichmar_2020_neuromodulated_attention.md`** (corpus) — direct intellectual descendant: uses an OFC / DA / 5-HT-style architecture in a deeper RL setting.
- **`xing_krichmar_2020_patience_navigation.md`** and **`xing_krichmar_2022_environment_neuromodulation.md`** (corpus) — Krichmar-lab successors that apply neuromodulated RL to navigation.
