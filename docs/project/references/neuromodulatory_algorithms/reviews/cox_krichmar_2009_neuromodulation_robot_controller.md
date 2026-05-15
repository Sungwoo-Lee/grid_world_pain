---
title: "Neuromodulation as a Robot Controller: A Brain-Inspired Strategy for Controlling Autonomous Robots"
authors: ["Brian R. Cox", "Jeffrey L. Krichmar"]
year: 2009
venue: "IEEE Robotics & Automation Magazine, vol. 16, no. 3, pp. 72–80, September 2009"
slug: cox_krichmar_2009_neuromodulation_robot_controller
source_pdf: "sources/Cox and Krichmar 2009 - Neuromodulation as a robot controller.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

This is the **foundational engineering paper** for the Krichmar-school programme: *let's build a wheeled robot whose entire controller is a simulated mini-brain of neuromodulator nuclei, and see if the resulting behaviour is decisive and goal-appropriate.* The robot is CARL-1 — a two-wheeled mobile robot with a camera, IR obstacle sensors, and a wireless link to a workstation running a 6,700-neuron mean-firing-rate neural model with about 1.3 million synapses.

The biological premise is now standard but was the operational claim of this paper: **the four main neuromodulator systems of the mammalian brain do different things**:
- **Dopamine (DA)** released from the **ventral tegmental area (VTA)** signals reward / reward-prediction.
- **Serotonin (5-HT)** released from the **Raphe nucleus** signals risk / threat / anxiety.
- **Acetylcholine (ACh)** released from the **basal forebrain (BF)** sets *attentional effort* — how much computational resource to throw at the current decision.
- **Noradrenaline (NA)** from the **locus coeruleus** signals novelty / saliency (not modelled here, but acknowledged).

What unites them, despite their different triggers, is the *computational effect*: **phasic (short-burst) neuromodulator release sharpens the signal-to-noise ratio (SNR) of downstream neural populations**, biasing them toward decisive winner-take-all responses. The same architecture can therefore approach (Find) green panels and flee (Flee) red panels — same neural fabric, two opposite behaviours, gated by *which* neuromodulator fires.

The robot is trained by an operator: when CARL-1 sits on a green panel, the operator presses a "good" button (excites VTA, inhibits Raphe); when on a red panel, presses "bad" (excites Raphe, inhibits VTA). A modified BCM Hebbian-style learning rule, *gated* by neuromodulator activity above a threshold, wires the colour-to-action mappings into the synapses. After 10 good + 10 bad events the robot is on its own. **Result**: it reliably approaches green and flees red; **lesion** the VTA and Find collapses; lesion the Raphe and Flee collapses; lesion *only* BF and behaviour is essentially preserved; but lesion BF *together with* VTA or Raphe and behaviour collapses below the single-lesion level — i.e., **ACh is a compensatory amplifier that becomes essential under load**.

This paper is the template from which Krichmar's later neurorobotic work (Krichmar 2013, Xing et al., Zou et al., Chiba and Krichmar 2020) descends — the architecture, the choice of nuclei, the BCM-with-neuromodulator-gating rule, and the lesion methodology.

## Section-ordered backbone

### Introduction
Neuromodulators are chemical transmitters with strong, lasting effects on behaviour. Four major systems — NA, 5-HT, DA, ACh — project broadly from small subcortical nuclei to most of the brain, are reciprocally connected with cognitive areas (amygdala, frontal cortex, hippocampus), and share a common downstream effect: **sharpening of target neural networks toward a winner-take-all response**. The authors propose a computational framework with two premises: (i) the common effect of these systems is *decisive action when conditions demand it, exploratory behaviour otherwise*; (ii) the systems differ in what *triggers* them — 5-HT by risk, ACh by attentional demand, DA by reward anticipation, NA by novelty. A robot controller built on this design should learn context-appropriate actions, focus on important stimuli, ignore distractors, and respond decisively.

### Methods — Robot and apparatus
CARL-1 is a Cognitive Anteater Robotics Lab build: 10″ × 8.5″ two-wheeled base with a CCD camera on a pan-tilt mount, IR obstacle sensors, Wi-Fi to a workstation. Environment: 10 × 10 ft enclosure with eight floor light panels; four corners cyan / green / magenta / red, settable from the workstation.

### Methods — Neural architecture
6,700 neurons, ≈ 1.3 M synapses. Components:
- **Visuomotor area** — four subareas (cyan, green, magenta, red), each 15 × 20 = 300 retinotopic neurons.
- **Neuromodulatory systems** — three 100-neuron pools: BF (ACh), Raphe (5-HT), VTA (DA).
- **Action areas** — Find (100 neurons), Flee (100 neurons).
- **Behaviour drivers** — Good (100 neurons), Bad (100 neurons), set by the operator's GUI buttons.

Connectivity:
- Within a visuomotor subarea (e.g., red→red): Gaussian neighbourhood, std 5 neurons, initial weights 0.8–1.0.
- Across visuomotor subareas (e.g., red↔green): 25% sparse, weights 0.8–1.0 excitatory or −0.8 to −1.0 inhibitory — i.e., *mutual inhibition between colour channels*.
- Visuomotor → neuromodulatory: 10% sparse, weights 0.05–0.10, plastic.
- Behaviour drivers → modulators / actions: dense, weight ±200 (very strong) — Good excites VTA and Find, inhibits Raphe and Flee; Bad does the opposite.
- VTA → Find: all-to-all. Raphe → Flee: all-to-all. BF → both Raphe and Flee: all-to-all (initial weights 0.1–0.2).

### Methods — Neuronal dynamics
Mean firing rate model with persistence (one-pole IIR) and logistic activation; synaptic input is gated by a global neuromodulator level $nm(t)$ that scales extrinsic and inhibitory inputs by ten times the average activity of BF + Raphe + VTA (intrinsic excitatory connections always see $nm = 1$).

### Methods — Synaptic plasticity
Weights from visuomotor to modulator and action areas are updated by a **neuromodulator-gated BCM rule**: a Heaviside gate $H_{NM}$ activates learning only when $nm > 2$, and a sliding threshold $\theta_{BCM}$ (Bienenstock–Cooper–Munro 1982) determines whether the update is potentiating or depressing. Weights also decay slowly back to their initial value to prevent over-learning.

### Methods — Action selection
Three-state controller: random exploration (default), Find (approach), Flee (move away). Switching happens when the difference between Find and Flee averaged activity exceeds 0.75. During exploration the camera pans and turning rate is proportional to pan position. During Find/Flee the most salient colour is chosen by a Softmax over visuomotor area mean activities. Find: orient toward; Flee: orient away.

### Methods — Training protocol
Operator drives the robot near a colour panel and presses Good (green) or Bad (red); ten trials of each, then unsupervised testing for 7,500 simulation cycles (8–10 s panel-shuffle period). Ten distinct "subjects" trained — same hardware, different random connectivity seeds.

### Results — Behaviour
All ten subjects reliably approach green and flee red. Behavioural trajectory is *non-trivial*: gaze fixation first, then body turn, then approach (or retreat). The Find behaviour ends with the camera tilting down onto the panel; the Flee behaviour gives "the impression of an animal warily eyeing a threatening object as it slowly backed away".

### Results — Lesions
- Lesion **VTA** → Find responses collapse, Flee unaffected (p < 0.0005 Wilcoxon).
- Lesion **Raphe** → Flee responses collapse, Find unaffected (p < 0.0005 Wilcoxon).
- Lesion **BF** alone → no significant effect on either behaviour.
- Lesion **BF + VTA** → Find abolished; **BF + Raphe** → Flee abolished. The double lesion is *worse than the single lesion*, suggesting BF normally provides a compensatory amplification that becomes essential when the primary modulator is impaired.

### Results — Signal-to-noise ratio
Authors define SNR as ratio of target-colour visuomotor activity to total colour activity during the relevant behaviour. Lesions of the primary modulator (VTA for Find, Raphe for Flee) significantly reduce SNR; double lesions (BF + primary) reduce it further (all p ≪ 0.0001, t-test).

### Results — Correlation of modulator with colour
Raphe activity correlates strongly with red activity ($r = 0.951$), is uncorrelated with green ($r = -0.099$) or magenta ($r = -0.237$) or cyan ($r = -0.319$). VTA correlates strongly with green ($r = 0.980$) and is approximately uncorrelated or weakly anti-correlated with the other colours. The neuromodulator response is specific to value-laden stimuli.

### Discussion
Three sub-discussions tie the experiment to the neuroscience literature:
- **DA and wanting**: VTA is for value-laden approach. Berridge's distinction between wanting and liking; Redgrave and Gurney on dopamine in action discovery.
- **5-HT and risk**: Raphe activity codes threats; in primates 5-HT is involved in social anxiety / threat (Watson et al. 2009).
- **ACh and attentional effort**: BF is the source of cortical ACh; BF lesion is the canonical lesion for impaired *increased* attentional effort, which is why BF alone doesn't impair simple behaviour but BF + primary modulator does.

### Conclusion / Neurorobot approach
The neurorobotics philosophy: real environments are richer and noisier than simulated ones, "understanding through building" (Pfeifer and Bongard 2007). Other neuromodulator-robot work cited as related: Alexander and Sporns (DA reward anticipation), Prescott et al. (basal-ganglia model), Doya and Uchibe (Cyber Rodent), Garforth et al. (attentional selection). The Cox–Krichmar contribution is the specific mechanism — *phasic modulator burst amplifies extrinsic inputs ten-fold and gates plasticity above a threshold*.

## Phase 1 — Undergraduate-level synthesis

**The setup.** A wheeled robot, a camera, a 10-foot arena with coloured floor panels. A computer simulates a mini-brain with 6,700 neurons and 1.3 million synapses.

**The mini-brain has four kinds of region.**
- *Eye regions* (one per colour): light up when the robot sees that colour.
- *Modulator regions* (three of them): VTA = "good signal", Raphe = "bad signal", BF = "pay attention" signal.
- *Action regions*: Find (approach) and Flee (run away).
- *Teacher buttons*: Good / Bad — clicked by the human during training.

**Training.** Human walks the robot near a green panel and clicks Good; near a red panel and clicks Bad. Each click jolts the modulator regions, which jolts the action regions, which strengthens the wires from "I see this colour" to "perform this action" — but only *when the modulator is strongly active* (this is the gating).

**Testing.** Robot is on its own. Each time it sees green, the VTA fires, sharpens the green signal, suppresses everything else, and the Find region fires, and the robot approaches. Each time it sees red, the Raphe fires, sharpens the red signal, suppresses everything else, and the Flee region fires, and the robot retreats. The interesting trick is that this is *the same neural fabric* — only the modulator differs.

**The lesions tell us what each piece does.**
- Cut out VTA → robot stops approaching green (treats it as neutral).
- Cut out Raphe → robot stops fleeing red (treats it as neutral, dangerously risky).
- Cut out BF alone → no obvious effect.
- Cut out BF + VTA → Find collapses *more* than VTA alone. So BF is a *compensatory* "extra attention" amplifier; you don't see it until you have already broken the primary modulator.

**The big idea.** Different neuromodulators carry different *meanings* (good / bad / pay-attention), but they have the *same mechanical effect*: amplify the relevant signal, suppress the rest, and lock in the learning that wires the next response. This is the *gain-control* view of neuromodulation that the rest of the Krichmar programme builds on.

## Phase 2 — Graduate-level deep dive

### Mean-firing-rate neuron with persistence

Each neuron $i$ has activity $s_i(t) \in [0,1]$. The update is

$$
s_i(t) = \rho_i \, s_i(t-1) + (1 - \rho_i) \, \sigma\!\big( -0.1 \, I_i(t) \big)^{-1},
$$

written more cleanly as

$$
s_i(t) = \rho_i \, s_i(t-1) + (1 - \rho_i) \cdot \frac{1}{1 + e^{-0.1 \, I_i(t)}},
$$

where $\rho_i$ is per-neuron *persistence* (visuomotor neurons $\rho = 0.5$; all others $\rho = 0.1$) and the logistic sigmoid maps synaptic input to a firing rate. The persistence term makes the neuron a low-pass filter on its input.

### Synaptic input with global neuromodulator gain

$$
I_i(t) = \sum_j nm(t-1) \, w_{ij}(t-1) \, s_j(t-1),
$$

where $nm$ is the *current level of neuromodulator at synapse $ij$*. The trick: $nm$ is set to **ten times the combined average activity of BF, Raphe, and VTA** for *extrinsic and inhibitory* connections, and $nm = 1$ for intrinsic excitatory connections. So:

$$
nm_{\text{extrinsic}}(t) = 10 \cdot \frac{\bar{a}_{\text{BF}}(t) + \bar{a}_{\text{Raphe}}(t) + \bar{a}_{\text{VTA}}(t)}{3},
$$

where $\bar{a}_X$ is the mean firing rate of neural area $X$. A phasic modulator burst (high $\bar{a}$) therefore *multiplies* extrinsic and inhibitory inputs by a large factor, while leaving intrinsic excitatory inputs unchanged. This is the SNR-sharpening mechanism: under a burst, sensory inputs (extrinsic) suddenly dominate associational background (intrinsic), and inhibitory connections (also extrinsic in their treatment) suddenly *win* the lateral competition.

### Neuromodulator-gated BCM plasticity

For visuomotor-to-modulator and visuomotor-to-action synapses:

$$
\Delta w_{ij}(t) = \underbrace{\varepsilon \left( w_{ij}(0) - w_{ij}(t-1) \right)}_{\text{decay to initial}} + \underbrace{\delta \, H_{NM}(nm) \, s_j(t-1) \left( s_i(t-1) - \theta_{BCM} \right)}_{\text{neuromodulator-gated BCM term}},
$$

with $\varepsilon = 10^{-5}$, $\delta = 10^{-3}$, $H_{NM}(nm) = \mathbb{1}\{nm > 2\}$ (a Heaviside step requiring $nm > 2$ to enable plasticity), and the sliding threshold

$$
\Delta \theta_{BCM} = 0.001 \left( s_i(t)^2 - \theta_{BCM} \right),
$$

i.e., $\theta_{BCM}$ tracks the squared postsynaptic activity over time — a neuron that has been very active will need an even more active postsynaptic signal to potentiate further (Bienenstock, Cooper, and Munro 1982). The decay-to-initial term acts as a slow forgetting function preventing runaway potentiation.

**Why this matters.** The combination "BCM rule × Heaviside-gated by modulator activity" implements **a learning rule that only fires when the brain is paying attention to a value-laden event** — the gate is closed during exploration (when BF + Raphe + VTA are all quiet) and open only during reward / threat / saliency. This is qualitatively the same trick that appears in eligibility-trace RL: only credit-assign at outcome time, not at every time step.

### Action selection via Softmax over visuomotor activities

In Find / Flee mode, the most salient colour is chosen by

$$
P_c = \frac{e^{5 a_c}}{\sum_{i=1}^{4} e^{5 a_i}},
$$

with $a_c$ the average activity of visuomotor area $c$. Temperature is effectively $1/5$, giving a fairly sharp Softmax. The camera then saccades to the centroid of activity in the chosen area; pan position drives turning rate; tilt position drives forward velocity (positive for Find, negative for Flee).

### SNR metric

$$
\mathrm{SNR} = \frac{\mathrm{vis}_{\text{tgt}}}{\sum_{i=1}^4 \mathrm{vis}_i},
$$

with $\mathrm{vis}_{\text{tgt}}$ the average activity of green during Find or red during Flee, and the denominator the total visuomotor activity. Modulator lesions move this ratio significantly downward.

### The mechanism the paper claims

Three claims, in order of increasing strength:
1. **Phasic modulator activity sharpens SNR** — supported by the SNR metric and lesion-induced reductions.
2. **Specificity of effect derives from triggering specificity, not from downstream specificity** — VTA fires for green because the Good button connects to it; Raphe fires for red because the Bad button connects to it; in both cases the downstream effect is the same SNR sharpening, applied to whichever colour channel happens to be co-active at the moment of the burst.
3. **BF is a *compensatory* amplifier** — necessary only under load (concurrent failure of the primary modulator).

### Relevance flag for the present project

This paper is the **original Krichmar template** for *gain-modulating a sensory channel by a value-laden modulator signal*. Three transferable structural commitments:
1. **Modulator effect is multiplicative on extrinsic inputs**, leaving intrinsic associational background unchanged. The same multiplicative-gain motif appears in FiLM, hypernet conditioning, and in L'Haridon's cortisol × damage = pain rule. In a pain-modulated RL setting, the natural reading is "interoceptive modulator multiplies the nociceptive channel before the policy sees it", not "modulator adds a bias".
2. **Plasticity is gated by modulator activity above a threshold**. In a deep-RL setting this maps onto eligibility-trace / TD-error-gated updates; the paper provides a biologically motivated cleanup of why this gating is the natural design.
3. **Multiple modulators with different *triggers* but the *same downstream effect*** — DA for reward, 5-HT for threat. In a multi-affect RL agent (pain + curiosity + reward), this licenses a design where each affect has its own trigger circuit but a shared SNR-sharpening downstream.

## Connections

- **`avery_krichmar_2017_models_neuromodulation.md`** — the explicit review-and-models paper from the same lab, building on this 2009 architecture.
- **`krichmar_2013_neurorobotic_anxiety_curiosity.md`** — direct successor: same architecture, applied to the 5-HT / NA anxiety vs. curiosity trade-off.
- **`chiba_krichmar_2020_self_monitoring.md`** — the meta-cognitive extension: same neuromodulator architecture but with self-monitoring of internal state to gate exploration / exploitation.
- **Krichmar 2008** ("The neuromodulatory system — A framework for survival and adaptive behavior in a challenging world", *Adaptive Behavior*) — cited as [1]; the theoretical companion to this paper, with the same nuclei-and-functions taxonomy.
- **Doya 2002** ([`doya_2002_metalearning_neuromodulation.md`](doya_2002_metalearning_neuromodulation.md), in this corpus) — independent and earlier RL-theoretic mapping: ACh = action-value learning rate, NA = inverse temperature, DA = TD error, 5-HT = discount factor. The Cox–Krichmar paper is the engineering instantiation; Doya is the theoretical framework.
- **Doya and Uchibe (2005) Cyber Rodent** — cited as [23]; same kind of full-stack neurorobotic experiment but with battery-pack-driven exploration / exploitation switching.
- **Bienenstock, Cooper, and Munro (1982)** — cited as [12]; the BCM plasticity rule that the modulator gates here.
- **Aston-Jones and Cohen (2005)** — cited as [5]; the locus-coeruleus adaptive-gain theory that motivates the SNR-sharpening interpretation.
- **Yu and Dayan (2005)** ("Uncertainty, neuromodulation, and attention") — cited as [9]; the *uncertainty* angle on the same nuclei. The present project's uncertainty corpus (`docs/project/references/uncertainty/`) descends from this line.
- **Cañamero school (`khan_canamero_2022_social_buffering.md`, `lharidon_canamero_2023_stress_pain.md`)** — same problem (modulating action selection via hormonal state) but with a homeostatic-deficit reading of the modulator rather than a value-prediction reading.
