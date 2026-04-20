# Computational Models of Perceptual Decision-Making in Pain: A Tutorial on Hypervigilance

> **Purpose**: Tutorial and introductory document for the Interoceptive AI research project.
> Covers foundational concepts through graduate-level computational models, with a focus on hypervigilance as a central phenomenon.
> **Audience**: Undergraduate foundations (Part I) through graduate-level theory (Parts II-III) and implementation (Part IV).
> **Last updated**: 2026-04-16

---

## Table of Contents

### Part I: Foundations (Undergraduate Level)
1. [What is Perceptual Decision-Making?](#1-what-is-perceptual-decision-making)
2. [What is Pain? — Beyond the Alarm Bell](#2-what-is-pain--beyond-the-alarm-bell)
3. [Connecting the Two — Pain as a Perceptual Decision](#3-connecting-the-two--pain-as-a-perceptual-decision)

### Part II: Computational Frameworks (Graduate Level)
4. [Bayesian Pain Inference](#4-bayesian-pain-inference)
5. [Drift-Diffusion Models of Pain Perception](#5-drift-diffusion-models-of-pain-perception)
6. [Defensive Decision-Making and Threat Computation](#6-defensive-decision-making-and-threat-computation)
7. [Pain as Optimal Control — The LQG Framework](#7-pain-as-optimal-control--the-lqg-framework)
8. [Active Inference and the Chronic Pain Trap](#8-active-inference-and-the-chronic-pain-trap)

### Part III: Hypervigilance — A Unified Computational Account
9. [Defining Hypervigilance Computationally](#9-defining-hypervigilance-computationally)
10. [From Theory to Engineering — Computational Phenotyping](#10-from-theory-to-engineering--computational-phenotyping)

### Part IV: Implementation in the Grid-World Pain Environment
11. [The Grid-World Pain Platform — Architecture Overview](#11-the-grid-world-pain-platform--architecture-overview)
12. [Mapping Computational Pain Models to the Grid-World](#12-mapping-computational-pain-models-to-the-grid-world)
13. [Implementation Roadmap — Hypervigilance in Silico](#13-implementation-roadmap--hypervigilance-in-silico)

### [Annotated Bibliography](#annotated-bibliography)

---

## 1. What is Perceptual Decision-Making?

Every moment, your brain faces a fundamental problem: the world is uncertain, but you must act. Perceptual decision-making is the process by which the brain converts noisy, ambiguous sensory information into categorical choices — "is that shape a predator or a bush?", "is this sensation painful or benign?", "should I approach or flee?"

### 1.1 The Brain as a Statistician

Consider a simple example. You are walking through a forest and hear a rustling sound. The sound could be caused by wind (harmless) or by a snake (dangerous). Your brain receives a noisy auditory signal and must decide which hypothesis is true. This is a **statistical inference** problem: given uncertain evidence, which explanation is more likely?

The formal framework for this is **Signal Detection Theory (SDT)** (Green & Swets, 1966). Given two competing hypotheses $h_1$ (wind) and $h_2$ (snake), and noisy sensory evidence $e$, the brain computes a **likelihood ratio**:

$$
l_{12}(e) = \frac{P(e \mid h_2)}{P(e \mid h_1)}
$$

This ratio compares how likely the observed evidence is under each hypothesis. The decision rule is: choose "snake" if $l_{12}(e) \geq \beta$, where $\beta$ is a **criterion** that depends on the observer's goals:

- **Maximize accuracy** (equal priors): $\beta = 1$ — choose whichever hypothesis the evidence favors.
- **Account for prior beliefs** (unequal priors): $\beta = P(h_1) / P(h_2)$ — if snakes are rare, require stronger evidence.
- **Account for costs**: if mistaking a snake for wind is deadly, lower $\beta$ to err on the side of caution. This produces a **liberal criterion** — exactly what we will later call hypervigilance.

### 1.2 Evidence Accumulation Over Time

In reality, evidence does not arrive all at once. It trickles in over time — you hear the rustling for several seconds, each moment providing a noisy sample. The brain must *accumulate* this evidence before committing to a decision.

The mathematical framework for this is the **Sequential Probability Ratio Test (SPRT)** (Wald, 1947). Instead of deciding from a single sample, the brain keeps a running tally — a **decision variable** $y_n$ that sums up the evidence so far:

$$
y_n = \sum_{i=1}^{n} \log \frac{P(e_i \mid h_2)}{P(e_i \mid h_1)}
$$

Each new piece of evidence $e_i$ nudges the tally toward one hypothesis or the other. Accumulation continues until $y_n$ crosses one of two **bounds** — thresholds that trigger a commitment to one choice. This produces a beautiful tradeoff:

- **High bounds** = more evidence required = slower but more accurate decisions.
- **Low bounds** = less evidence required = faster but more error-prone decisions.

This is the **speed-accuracy tradeoff**, a universal feature of biological decision-making (Gold & Shadlen, 2007).

### 1.3 The Random-Dot Motion Paradigm

The canonical laboratory task for studying perceptual decision-making is the **random-dot motion (RDM)** task. A monkey watches a field of dots; some fraction (the "coherence") move in a consistent direction while the rest move randomly. The monkey must decide: left or right?

Single-unit recordings in macaques revealed the neural implementation of the SPRT:

1. **Sensory neurons in area MT** encode the *momentary evidence* — their firing rate reflects the direction and strength of the motion signal.
2. **Downstream neurons in area LIP** *accumulate* this evidence over time — their firing rate ramps up like the decision variable $y_n$.
3. **A stereotyped firing-rate threshold** triggers the saccade — this is the neural implementation of the SPRT bound (Gold & Shadlen, 2007).

The key insight: the brain literally computes the mathematical integrals predicted by statistical decision theory. This is not a metaphor — specific neurons implement specific terms in the equations.

### 1.4 Why This Matters for Pain

The framework of perceptual decision-making — noisy evidence, accumulation, thresholds, priors, and costs — applies far beyond vision. As we will see, pain perception is a perceptual decision: the brain must infer, from noisy nociceptive signals, whether the body is damaged and how severely. Hypervigilance, the amplification of pain-related signals, is a specific configuration of this decision-making machinery — one where priors are pessimistic, thresholds are low, and costs are asymmetric.

---

## 2. What is Pain? — Beyond the Alarm Bell

### 2.1 The Classical View and Its Failures

The traditional "alarm bell" model treats pain as a straightforward readout of tissue damage: nociceptors in the skin detect a noxious stimulus, send a signal up the spinal cord, and the brain registers "ouch." Pain intensity should be proportional to stimulus intensity — more damage, more pain.

This model is empirically wrong. Consider the following well-documented phenomena:

- **Placebo analgesia**: A sugar pill described as a painkiller genuinely reduces pain. No change in the stimulus, yet the experience changes (Wiech, 2016).
- **Nocebo hyperalgesia**: Warning someone that a procedure will be extremely painful makes the identical stimulus hurt more.
- **Distraction**: Soldiers report not noticing severe battlefield injuries until combat ends. Cognitive load reduces pain.
- **Catastrophizing**: Ruminating about pain ("this will never get better") amplifies the experience far beyond what nociceptive input would predict.
- **Chronic pain without tissue damage**: Many chronic pain patients have no detectable peripheral pathology — the alarm is ringing, but there is no fire.
- **Painless injury**: Congenital insensitivity to pain (CIP) patients sustain severe tissue damage without any pain experience.

These observations force a radical conclusion: **pain is not a sensation — it is a perception constructed by the brain** (Wiech, 2016). The nociceptive signal is just one input among many. The brain combines it with expectations, context, attention, memory, and goals to produce the experience we call pain.

### 2.2 Three Kinds of Pain — Three Computational Problems

Seymour, Crook & Chen (2023) draw a critical distinction between three types of pain that correspond to fundamentally different computational problems:

| Type | Timescale | Computational Role | Example |
|------|-----------|-------------------|---------|
| **Acute nociception** | Milliseconds to seconds | Fast feedforward reflex to minimize ongoing damage | Touching a hot stove → withdrawal |
| **Post-injury pain** | Hours to weeks | Feedback controller for recuperation — suppresses movement, promotes rest, reallocates energy | Limping after a sprained ankle |
| **Chronic pain** | Months to years | System failure — the controller is stuck in a protective mode despite tissue healing | Back pain persisting years after the original injury has healed |

Acute pain is the alarm bell. It works well and is computationally simple — a fast reflex loop from nociceptor to spinal cord to muscle.

Post-injury pain is something entirely different. It is not signaling ongoing damage; it is *managing recovery*. The brain must infer a hidden variable — the state of tissue healing — from noisy, indirect signals, and must decide how much to protect versus how much to resume normal activity. This is an **optimal control problem under uncertainty**.

Chronic pain is what happens when this control system breaks down. The controller becomes stuck in "injured" mode, continuing to enforce protection long after the tissue has healed.

### 2.3 The Puzzle of Hypervigilance

Hypervigilance to pain is the phenomenon where an organism becomes *more* sensitive to pain-related signals after injury — not less. This seems paradoxical from a simple alarm-bell perspective: if the alarm has already sounded, why turn up the volume?

Crombez, Van Damme & Eccleston (2005) define hypervigilance precisely as **"an unintentional and efficient process that emerges when the threat value of pain is high, the fear system is activated, and the individual's current concern is to escape and avoid pain."** It is specifically an *attentional* phenomenon — distinct from peripheral/central sensitization mechanisms that produce hyperalgesia and allodynia.

The attentional manifestations include:

- **Attentional capture**: pain-related stimuli automatically seize attention, even when irrelevant to the current task.
- **Difficulty disengaging**: once attention is captured by pain, it is hard to redirect it to other goals.
- **Interpretive bias**: ambiguous sensations are more likely to be interpreted as painful.
- **Scanning**: persistent monitoring of the body for threat signals.

> **Important distinction.** Hyperalgesia (amplified pain to noxious stimuli) and allodynia (pain from non-noxious stimuli) are separate central sensitization mechanisms, not manifestations of hypervigilance per se (Crombez et al., 2005). However, they are computationally related: sensitization increases the sensory precision $\Pi_{\text{obs}}$ for nociceptive channels, which feeds into the attentional system and can promote hypervigilant scanning. The two processes co-occur clinically and reinforce each other, but they operate at different levels of the computational hierarchy.

From a computational perspective, hypervigilance is not a malfunction — it is an *adaptive strategy* that makes perfect sense in a world where the cost of underestimating injury vastly exceeds the cost of overestimating it (Seymour et al., 2023). An organism that underestimates a wound and resumes foraging too early risks reopening the injury, infection, and death. An organism that overestimates and rests an extra day merely loses one day of foraging.

The problem arises when this adaptive strategy fails to disengage. Hypervigilance that persists beyond the healing period is the computational substrate of the transition from acute to chronic pain. Understanding this transition — *why* the system gets stuck and *how* to unstick it — is the central goal of this tutorial.

### 2.4 Comparative Evidence: Hypervigilance is Ancient

Hypervigilance is not a human peculiarity or even a mammalian one. Cephalopods (octopus, cuttlefish, squid) — separated from mammals by ~500 million years of evolution — show strikingly similar post-injury behavior (Seymour et al., 2023):

- Peripheral nociceptors exhibit threshold reduction and increased spontaneous firing after injury.
- Wound-directed grooming and guarding persists for *days*.
- These behaviors are modulated by opioid-class agonists.

This cross-phylum convergence is strong evidence that post-injury hypervigilance is an ancient, optimal-control solution that has been independently reinvented because the underlying computational problem — managing recovery from tissue damage under uncertainty — is universal.

---

## 3. Connecting the Two — Pain as a Perceptual Decision

### 3.1 Pain Perception as Bayesian Inference — The Intuitive Version

Section 1 introduced the brain as a decision-maker that combines noisy evidence with prior beliefs. Section 2 showed that pain is a constructed percept, not a simple readout. The connection is direct: **pain perception is a perceptual decision** governed by the same Bayesian machinery.

Here is the intuitive picture. Your brain maintains a *belief* about whether your body is in danger. This belief is not based solely on what the nociceptors are reporting right now — it also incorporates:

- **Prior expectations**: What do you expect to feel? If you have been told "this injection will hurt," you expect pain.
- **Context**: Are you in a hospital (safe) or a war zone (dangerous)?
- **Attention**: Are you focused on the sensation or distracted by a conversation?
- **History**: Have you experienced this kind of stimulus before? What happened last time?

The brain combines all of these into a single *posterior belief* — the best estimate of the true pain state given everything the brain knows. This posterior belief *is* the pain experience.

### 3.2 A Simple Worked Example

Imagine a patient is about to receive a heat stimulus on their arm. The brain has two sources of information:

1. **Prior expectation** ($\mu_{\text{prior}}$): Based on the doctor's warning, the patient expects a pain intensity of **7 out of 10**.
2. **Sensory evidence** ($o$): The nociceptors report an actual intensity of **4 out of 10**.

If pain were a simple readout, the patient would feel 4/10. But under Bayesian inference, the brain combines prior and evidence, weighted by their respective *reliabilities* (precisions):

- If the patient is **very confident in the warning** (high prior precision) and **uncertain about the sensory signal** (low observation precision, e.g., the stimulus is ambiguous), the experience will be pulled toward the prior: maybe **6/10**.
- If the patient **trusts the sensory signal** (high observation precision, e.g., clear sharp stimulus) and **is unsure about the warning** (low prior precision), the experience will be pulled toward the evidence: maybe **4.5/10**.

The key insight: **the same physical stimulus produces different pain experiences depending on the relative precision of prior and evidence**. This single principle explains placebo (low-pain prior pulls experience down), nocebo (high-pain prior pulls experience up), distraction (lowers sensory precision → experience drifts toward prior), and attention (raises sensory precision → experience tracks the stimulus more closely).

### 3.3 Why Expectations Can Override Sensation

The most dramatic demonstration of this principle is the case where expectations become so strong that they *completely override* sensory input. In the Bayesian framework, this happens when the prior precision becomes infinitely large relative to the observation precision:

- If $\text{Prior precision} \gg \text{Observation precision}$: the brain *ignores the sensory signal entirely* and the experience equals the expectation.

This is not just a theoretical curiosity. It is the computational account of:

- **Phantom limb pain**: the limb is gone, but the brain's prior on "damaged limb" is so strong that it generates pain without any peripheral input.
- **Placebo surgery**: sham operations (skin incision only, no actual repair) can produce genuine pain relief — the prior "I have been fixed" overwrites the nociceptive signal.
- **Chronic pain without pathology**: the prior on "injured" is so precise and so resistant to updating that the brain cannot accept incoming evidence of recovery.

This is the formal substrate of what clinicians call *catastrophizing* — and, as we will develop in detail, it is the core computational mechanism of pathological hypervigilance.

### 3.4 Preview of What Comes Next

Parts I has given us the conceptual foundation:

1. The brain is a statistical decision-maker that accumulates noisy evidence against prior beliefs.
2. Pain is a constructed percept — the output of this inference process applied to interoceptive (body-state) signals.
3. Hypervigilance is what happens when this machinery is configured for maximum sensitivity to threat: strong priors, low thresholds, and asymmetric costs.

Part II will formalize each of these ideas mathematically. We will derive the exact equations for Bayesian pain inference (Section 4), show how the drift-diffusion model captures pain decisions (Section 5), embed pain in the broader framework of defensive decision-making (Section 6), develop the optimal-control theory of post-injury pain (Section 7), and unify everything under active inference (Section 8).

---

## 4. Bayesian Pain Inference

This section formalizes the intuitive picture from Section 3 into a rigorous mathematical framework. Wiech (2016) frames pain as **"perception as inference"** within a **predictive coding** framework, where prior information generates expectations about future perception and interprets sensory input. The mathematical formalization below draws on the Bayesian and predictive-coding literature she cites (Friston, 2010; Buchel et al., 2014) to state this framework in precise terms.

> **Attribution note.** Wiech (2016) uses the terms "perception as inference" and "predictive coding" rather than "Bayesian inference." The Gaussian precision-weighted equations below are the standard mathematical formalization of the framework she describes, reconstructed from the literature she cites.

### 4.1 The Generative Model

Let $x$ denote the latent bodily/pain state (the "true" level of tissue damage or danger) and $o$ the noisy nociceptive observation (the signal arriving from peripheral nociceptors). By Bayes' theorem:

$$
P(x \mid o) = \frac{P(o \mid x)\, P(x)}{P(o)}
$$

- $P(x)$ is the **prior** — the brain's expectation about $x$ before observing the current nociceptive signal.
- $P(o \mid x)$ is the **likelihood** — how probable the observed signal is, given a true state $x$.
- $P(x \mid o)$ is the **posterior** — the brain's updated belief, which determines the pain experience.

### 4.2 Gaussian Closed Form and Precision-Weighting

Assume Gaussian distributions (a standard simplification that captures the essential structure):

$$
P(x) = \mathcal{N}(\mu_{\text{prior}}, \sigma_{\text{prior}}^2), \qquad P(o \mid x) = \mathcal{N}(x, \sigma_{\text{obs}}^2)
$$

The posterior is also Gaussian, with mean:

$$
\boxed{\;
\mu_{\text{post}} = \frac{\sigma_{\text{obs}}^2}{\sigma_{\text{obs}}^2 + \sigma_{\text{prior}}^2}\, \mu_{\text{prior}} + \frac{\sigma_{\text{prior}}^2}{\sigma_{\text{obs}}^2 + \sigma_{\text{prior}}^2}\, o
\;}
$$

**Derivation.** The posterior of two Gaussians is obtained by multiplying the PDFs:

$$
P(x \mid o) \propto \exp\!\left(-\frac{(o - x)^2}{2\sigma_{\text{obs}}^2}\right) \exp\!\left(-\frac{(x - \mu_{\text{prior}})^2}{2\sigma_{\text{prior}}^2}\right)
$$

Completing the square in $x$, the exponent becomes:

$$
-\frac{1}{2}\left(\frac{1}{\sigma_{\text{obs}}^2} + \frac{1}{\sigma_{\text{prior}}^2}\right)\!\left(x - \mu_{\text{post}}\right)^2 + \text{const}
$$

where

$$
\mu_{\text{post}} = \frac{\frac{1}{\sigma_{\text{obs}}^2}\, o + \frac{1}{\sigma_{\text{prior}}^2}\, \mu_{\text{prior}}}{\frac{1}{\sigma_{\text{obs}}^2} + \frac{1}{\sigma_{\text{prior}}^2}}
$$

Defining **precision** $\Pi = 1/\sigma^2$, this simplifies to the precision-weighted form:

$$
\mu_{\text{post}} = \frac{\Pi_{\text{obs}}\, o + \Pi_{\text{prior}}\, \mu_{\text{prior}}}{\Pi_{\text{obs}} + \Pi_{\text{prior}}}
$$

and the posterior precision is:

$$
\Pi_{\text{post}} = \Pi_{\text{obs}} + \Pi_{\text{prior}}
$$

**Interpretation.** The posterior pain experience is a weighted average of expectation and evidence, where the weights are the *precisions* (inverse variances) — i.e., the *reliabilities* — of each source. Whichever source is more precise (less noisy) dominates the percept.

### 4.3 Prediction Error and the Kalman-Form Update

An equivalent and computationally more revealing formulation uses the **prediction error** $\delta$:

$$
\delta = o - \mu_{\text{prior}}
$$

This is the discrepancy between what was expected and what was observed. The posterior can then be written as:

$$
\mu_{\text{post}} = \mu_{\text{prior}} + \underbrace{\frac{\Pi_{\text{obs}}}{\Pi_{\text{obs}} + \Pi_{\text{prior}}}}_{K}\, \delta
$$

The term $K$ is the **Kalman gain** — the fraction of the prediction error that is absorbed into the updated belief. It ranges from 0 (ignore the error, trust the prior) to 1 (fully trust the new evidence).

This is the **predictive coding** formulation of pain (Wiech, 2016; Friston, 2010):

1. The brain generates a **prediction** $\mu_{\text{prior}}$ about the expected nociceptive input.
2. The actual input $o$ is compared against this prediction, generating a **prediction error** $\delta$.
3. The prediction error is *precision-weighted* by the Kalman gain $K$ and used to update the belief.
4. The updated belief $\mu_{\text{post}}$ is the pain experience.

### 4.4 Attention and Expectation as Precision Modulators

The power of this framework lies in how it explains cognitive modulation of pain through a single mechanism — **precision modulation**:

| Cognitive Factor | Precision Effect | Consequence for Pain |
|:---|:---|:---|
| **Attention to pain** | $\Pi_{\text{obs}} \uparrow$ | $K \to 1$; pain tracks the stimulus closely; amplification |
| **Distraction** | $\Pi_{\text{obs}} \downarrow$ | $K \to 0$; pain drifts toward expectation; analgesia if prior is low |
| **Strong expectation of pain** | $\Pi_{\text{prior}} \uparrow$ with high $\mu_{\text{prior}}$ | $K \to 0$; pain locks onto the expectation; nocebo hyperalgesia |
| **Strong expectation of relief** | $\Pi_{\text{prior}} \uparrow$ with low $\mu_{\text{prior}}$ | $K \to 0$; pain locks onto low expectation; placebo analgesia |
| **Uncertainty about expectations** | $\Pi_{\text{prior}} \downarrow$ | $K \to 1$; pain becomes more stimulus-driven |

### 4.5 Neural Mapping

The computational variables map onto identifiable neural substrates (Wiech, 2016):

| Computational Variable | Neural Substrate |
|:---|:---|
| Likelihood $P(o \mid x)$ | S1, S2, posterior insula, lateral thalamus |
| Top-down prior $P(x)$ | DLPFC, rostral ACC |
| Descending precision / analgesia gate | ACC $\to$ PAG $\to$ spinal dorsal horn |
| Prediction error $\delta$ and valuation | vmPFC, NAcc (mesolimbic circuit) |
| Attentional precision $\Pi_{\text{obs}}$ | Salience network: anterior insula, mid-cingulate |
| Reappraisal / generative model change | vmPFC $\to$ NAcc (mesolimbic valuation) |

A critical finding from multivariate fMRI decoding: **cognitive self-regulation does not primarily modulate the Neurologic Pain Signature** (the S1/S2/thalamus pattern that tracks nociceptive intensity). Instead, it acts through **vmPFC-NAcc connectivity** — the mesolimbic valuation system. This means reappraisal changes the *generative model* (the prior and its precision), not the early sensory encoding (Wiech, 2016).

### 4.6 Hypervigilance as Precision Dysregulation

In this Bayesian framework, hypervigilance has a precise formal definition:

$$
\text{Hypervigilance} = \begin{cases}
\Pi_{\text{prior}} \uparrow \text{ with } \mu_{\text{prior}} \text{ biased toward "injury"} \\
\Pi_{\text{obs}} \uparrow \text{ for nociceptive channels (attentional amplification)} \\
K \to 0 \text{ for recovery signals, } K \to 1 \text{ for threat signals}
\end{cases}
$$

The first component (inflated prior precision on injury) means the brain is highly confident that the body is damaged and resistant to updating. The second (amplified sensory precision for nociceptive channels) means that any pain-related signal is given maximum weight. The third (asymmetric Kalman gain) means the system selectively absorbs evidence consistent with injury while discounting evidence of recovery.

In the **limit of infinite prior precision** ($\Pi_{\text{prior}} \to \infty$):

$$
\mu_{\text{post}} \to \mu_{\text{prior}}
$$

The brain ignores the sensory signal entirely. The experience is determined by the expectation alone. This is the computational signature of chronic pain decoupled from peripheral pathology — and the formal endpoint of pathological hypervigilance.

> **Attribution note.** Wiech (2016) describes chronic pain in terms of "suboptimal learning," "delayed updating," and "change-resistant mental representations of pain" — not in the explicit precision-parameter language used here. The formal translation of catastrophizing into overly precise priors ($\Pi_{\text{prior}} \to \infty$) comes from the broader active-inference literature (Friston, 2010; Edwards et al., 2012), not directly from Wiech's 2016 paper. However, the conceptual mapping is direct: "change-resistant mental representations" correspond formally to high-precision priors that resist updating.

---

## 5. Drift-Diffusion Models of Pain Perception

Section 4 described pain as a static Bayesian inference. But pain perception unfolds over *time* — the brain accumulates evidence before committing to a percept. The **Drift-Diffusion Model (DDM)** formalizes this temporal dimension and provides a link between computational theory and measurable behavioral data (reaction times, accuracy) that the static Bayesian model cannot.

### 5.1 DDM Basics

The DDM (Ratcliff & McKoon, 2008) models the decision process as a noisy evidence accumulation. A **decision variable** $Z(t)$ starts at some initial value and drifts stochastically toward one of two response boundaries:

$$
dZ = v\, dt + \sigma\, dW_t, \qquad Z(0) = z
$$

where:
- $v$ = **drift rate** — the average rate of evidence accumulation, reflecting signal strength or stimulus discriminability.
- $\sigma$ = **diffusion coefficient** — the noise in the evidence stream.
- $z$ = **starting point** — the initial bias of the accumulator before evidence begins.
- $a$ = **boundary separation** — the distance between the two decision thresholds.
- $t_0$ = **non-decision time** — encoding and motor execution time.

A decision is made when $Z(t)$ first hits either the upper boundary ($a$) or the lower boundary ($0$). The *which boundary* determines the choice; the *when* determines the reaction time.

**Speed-accuracy tradeoff.** The boundary separation $a$ controls the tradeoff:
- Large $a$: requires more evidence, slower but more accurate.
- Small $a$: requires less evidence, faster but more error-prone.

### 5.2 Two Routes of Expectation Influence on Pain

Wiech (2016) identifies two mathematically distinct ways that cognitive expectations can influence pain-related perceptual decisions:

| Mechanism | DDM Parameter | Effect | Behavioral Signature |
|:---|:---|:---|:---|
| **Altered sensory processing** | Drift rate shift $\Delta v$ | Changes the rate at which evidence accumulates | Symmetric RT changes; sensitivity ($d'$) shift |
| **Altered perceptual decision** | Starting point shift $\Delta z$ | Biases the accumulator closer to one boundary | Faster expectation-congruent *errors*; criterion ($\beta$) shift |

The critical empirical finding: when participants are cued to expect high or low pain before receiving a stimulus, **expectation predominantly shifts the starting point $z$, not the drift rate $v$** (Wiech, 2016).

**Derivation of the starting-point shift interpretation.** If expectation of high pain shifts the starting point $z$ closer to the "painful" boundary, then:
- Less evidence is needed to reach the "painful" threshold → faster "pain" responses.
- But when the actual stimulus is low-intensity, the accumulator starts near the wrong boundary and must traverse most of the boundary separation → slower correct rejections.
- Crucially, *errors* (reporting pain when there is none) become faster, not slower — the diagnostic signature of a bias rather than a sensitivity change.

Drift rate changes ($\Delta v$) were only observed in one specific condition: when participants expected high pain but received a low-intensity stimulus — i.e., when the expectation was maximally incongruent with the evidence (Wiech, 2016). This suggests that strong prior violations can engage additional sensory processing, but the default route of expectation influence is decisional.

### 5.3 Connecting DDM to Bayesian Inference

The DDM and Bayesian models are not competing accounts — they are complementary views of the same process:

- **Starting point $z$** corresponds to the **log prior odds** $\log[P(h_{\text{pain}})/P(h_{\text{no pain}})]$. A strong prior on pain shifts $z$ toward the pain boundary — exactly the starting-point bias observed empirically.
- **Drift rate $v$** corresponds to the **sensory evidence strength** — the nociceptive signal quality, which maps onto $\Pi_{\text{obs}}$ in the Bayesian framework.
- **Boundary separation $a$** corresponds to the **evidence criterion** — how much posterior confidence the brain requires before committing. This maps onto the speed-accuracy tradeoff controlled by the context (urgent threat vs. safe environment).

In the perceptual decision framework of O'Connell & Kelly (2021), the Centro-Parietal Positivity (CPP) — an EEG signal that builds to a fixed amplitude before each perceptual report — is the neural implementation of the DDM accumulator. The same logic applies to pain: a neural decision variable accumulates nociceptive evidence to a threshold, with the starting point and threshold modulated by cognitive context.

### 5.4 HDDM: Individual Differences and Clinical Phenotyping

The **Hierarchical Drift-Diffusion Model (HDDM)** (Wiecki, Sofer & Frank, 2013) extends the DDM to estimate parameters at both the group and individual level using hierarchical Bayesian inference:

$$
\theta_j \sim \mathcal{N}(\mu_\theta, \sigma_\theta^2), \qquad \mu_\theta \sim p(\mu_\theta), \quad \sigma_\theta \sim p(\sigma_\theta)
$$

where $\theta_j = (v_j, a_j, z_j, t_{0,j})$ are the DDM parameters for subject $j$, drawn from group-level distributions. The choice and RT data for each trial are modeled as:

$$
(c_{ij}, \mathrm{RT}_{ij}) \sim \mathrm{WFPT}(a_j, v_{ij}, z_j, t_{0,j})
$$

where WFPT is the Wiener First Passage Time distribution — the analytic likelihood of the DDM.

This is the tool that enables **computational phenotyping** of pain patients (Mahajan & Seymour, 2025). Instead of vague diagnostic labels, each patient is characterized by a vector of computational parameters: How fast do they accumulate nociceptive evidence ($v$)? How much evidence do they require before reporting pain ($a$)? How biased is their starting point toward pain ($z$)? These parameters are interpretable, individual-specific, and measurable.

### 5.5 Hypervigilance in DDM Terms

Hypervigilance maps onto a specific DDM parameter configuration:

1. **Starting point biased toward pain**: $z > a/2$ — the accumulator begins closer to the "pain" boundary. Less evidence is needed to trigger a pain response.
2. **Lowered boundary separation**: $a \downarrow$ — the system requires less total evidence before committing. Faster, more impulsive pain decisions.
3. **Possibly elevated drift rate for nociceptive channels**: $v \uparrow$ — nociceptive evidence is accumulated faster, reflecting attentional amplification.

Together, these produce the behavioral profile of hypervigilance: faster pain detection, lower pain thresholds, more false alarms (reporting pain when none is present), and faster expectation-congruent errors.

Under **threat imminence**, the DDM boundary collapses further via an urgency signal $u(t)$:

$$
a(t) = a_0 - u(t)
$$

This urgency-gating mechanism (Cisek, 2021; Tashjian et al., 2021) explains the transition from slow, deliberative pain evaluation (high $a$, model-based reasoning) to fast, reflexive pain responses (low $a$, Pavlovian defense) as perceived danger increases. Safety computations from the anterior vmPFC can suppress $u(t)$, restoring deliberative processing — but when safety signals fail (as in chronic pain), urgency remains high and the system stays in hypervigilant mode.

---

## 6. Defensive Decision-Making and Threat Computation

Pain does not exist in isolation — it is embedded in a broader system of defensive decision-making. This section reviews the computational architecture of threat processing and safety computation, showing how hypervigilance emerges as a specific failure mode within this architecture.

### 6.1 The Six-Level Taxonomy of Defensive Behaviour

LeDoux & Daw (2018) replace the monolithic concept of "fear" with a **computational hierarchy** of defensive strategies, each with a distinct algorithm and neural circuit:

| Level | Algorithm | Update Rule | Neural Substrate |
|:---|:---|:---|:---|
| 1. Reflexes (startle) | Innate stimulus-response | None (genetic prior) | Brainstem |
| 2. Fixed reaction patterns (freezing, flight) | Species-specific defense reactions (SSDRs) | None (selected by amygdala) | LA $\to$ CeA $\to$ PAG |
| 3. Habits (instrumental) | Model-free Q-learning / cached S-R | $Q(s,a) \leftarrow Q + \alpha\,[r + \gamma\max_{a'} Q(s',a') - Q]$ | Dorsolateral striatum, BA $\to$ NAcc |
| 4. Action-outcome (goal-directed) | Model-based RL (tree search) | Bellman backup over learned $T, R$ | mPFC, hippocampus |
| 5. Deliberative actions (implicit) | Model-based + prospection | Planning without conscious access | Prefrontal |
| 6. Deliberative actions (explicit) | Model-based + metacognition | Conscious planning + self-model | Lateral PFC, frontopolar |

> **Attribution note.** LeDoux & Daw (2018) classify levels 1-2 as "innate reactions" (covering the traditional Pavlovian category) and levels 3-6 as "instrumental behaviours" of increasing complexity. The terms "Pavlovian habits" and "instrumental habits" as separate levels are a simplification used in some secondary literature; the original paper treats habits as a single instrumental level.

The critical insight: **different levels dominate under different conditions**. Under immediate threat (predator lunging), control collapses to levels 1-2 (reflexes, freezing). Under distal, uncertain threat, control rises to levels 5-6 (planning, deliberation). The variable that governs this transition is **threat imminence** — the spatial and temporal proximity of danger.

The transition is mediated by **arbitration based on posterior uncertainty** (Daw et al., 2005):

$$
w_{\text{MB}} = \frac{1/\sigma^2_{\text{MB}}}{1/\sigma^2_{\text{MB}} + 1/\sigma^2_{\text{MF}}}, \qquad Q_{\text{net}} = w_{\text{MB}}\, Q_{\text{MB}} + (1 - w_{\text{MB}})\, Q_{\text{MF}}
$$

Under threat imminence, model-based search becomes too slow (its variance $\sigma^2_{\text{MB}}$ explodes), so $w_{\text{MB}} \to 0$ and control collapses onto model-free habits, then onto innate reflexes.

### 6.2 The Pavlovian Bias: Why Hypervigilance Promotes Freezing

A key computational feature of defensive behaviour is the **Pavlovian-instrumental interaction**. Pavlovian aversive values bias policy toward behavioural inhibition regardless of what instrumental learning suggests:

$$
Q_{\text{net}}(s,a) = w_{\text{inst}}\, Q_{\text{inst}}(s,a) + w_{\text{pav}}\, V_{\text{pav}}(s)\, \kappa(a)
$$

where $\kappa(a) > 0$ for "withhold/inhibit" actions and $\kappa(a) < 0$ for "approach/act." When the Pavlovian threat value $V_{\text{pav}}$ is strongly negative, the system rewards inhibition — producing freezing or guarding even when the optimal strategy is to flee or seek food.

This is directly relevant to pain hypervigilance: a hypervigilant agent with high Pavlovian threat values will tend toward **behavioral inhibition** (rest, guard, avoid movement), exactly the protective strategy that Seymour et al. (2023) identify as the precursor to chronic pain via information restriction.

### 6.3 Safety as a Distinct Computation

Tashjian, Zbozinek & Mobbs (2021) make the critical argument that safety is **not** simply the absence of threat ($\text{safety} \neq 1 - \text{threat}$). Safety is an independent, actively computed decision variable.

Let the threat vector be $\mathbf{T} = (I, V_T, U_T)$ (imminence, threat value, threat uncertainty) and the self-oriented vector be $\mathbf{O} = (\Pi, E, K)$ (available policies, prior experience, perceived control). Safety is:

$$
S(t) = f\bigl(\mathbf{T}(t),\, \mathbf{O}(t)\bigr)
$$

The key property: $\partial S / \partial K > 0$ even at fixed $\mathbf{T}$ — **perceived control decouples safety from objective threat**. This is why being inside a shark cage (high objective threat, high control) feels safer than hearing an unexplained noise at night (low objective threat, no control).

The full utility of an action integrates both:

$$
U(a) = w_1\, V_{\text{reward}}(a) + w_2\, V_S(a) - w_3\, V_{\text{threat}}(a)
$$

Safety enters as a **positive** utility term, not a negated threat.

### 6.4 Threat Variables: A Dimensional Account

Levy & Schiller (2021) decompose threat processing into five dissociable computational variables:

| Variable | Meaning | Formal Expression |
|:---|:---|:---|
| Value $V_{\text{harm}}$ | Magnitude of potential harm | Utility units |
| Probability / risk | Known $P(\text{harm})$ | Variance $\sigma^2$ |
| Ambiguity | Second-order uncertainty over $P$ | Dispersion $A$ |
| Imminence | Spatial/temporal proximity | Distance $d$; hyperbolic: $V_{\text{subj}}(d) = V_0 / (1 + k\,d)$ |
| Controllability | Action-vs-omission divergence | $C = P(\text{safe} \mid \text{action}) - P(\text{safe} \mid \text{no action})$ |

A combined subjective utility of a defensive action integrates all five dimensions:

$$
U(a) = -P(\text{harm} \mid a)\, V_{\text{harm}} - \alpha\, \sigma^2(a) - \beta\, A(a) - \gamma\, D_{\text{imm}}(d) + \eta\, C(a)
$$

> **Attribution note.** Levy & Schiller (2021) discuss each variable independently throughout the paper but do not present them as a formal five-item list or print a combined utility equation. The equation above is a synthesis constructed from the individual formalisms they describe, following standard decision-theoretic conventions.

Individual-difference vectors $(\alpha, \beta, k, \eta)$ constitute a **computational fingerprint** of anxiety and hypervigilance. In particular:
- **Extreme ambiguity aversion** ($\beta \gg 0$): penalizes any situation where threat probability is unknown — the "better safe than sorry" heuristic pushed to pathological extremes.
- **Flattened generalization gradient**: threat value transfers broadly across contexts ($c$ parameter low in Shepard's exponential generalization law), so novel situations are treated as threatening.

### 6.5 Hypervigilance as a Failure of Safety Computation

Combining the above frameworks, hypervigilance can be understood as a **dual failure**:

1. **Overactive threat computation**: high $V_{\text{harm}}$, high ambiguity aversion $\beta$, broad threat generalization (low $c$ in Shepard gradient), and elevated Pavlovian bias toward inhibition.
2. **Underactive safety computation**: low perceived control $K$, poor safety generalization (high $c_S$ — safety learned in one context does not transfer), and weak anterior vmPFC integration.

In the DDM terms of Section 5, overactive threat raises the urgency signal $u(t)$, collapsing the boundary and forcing fast reflexive decisions. Underactive safety fails to suppress $u(t)$, preventing the restoration of deliberative processing. The agent is trapped in a low-boundary, high-urgency, inhibition-biased regime — the computational architecture of hypervigilance.

The neural architecture supports this dual-failure account:
- **Anterior vmPFC**: safety hub; integrates self-model variables (control, experience) with threat; outputs to ventral striatum for exploratory behavior.
- **Posterior vmPFC**: anticipated-threat hub; co-activates with amygdala and PAG for defensive output.
- **BNST (bed nucleus of the stria terminalis)**: handles sustained, uncertain threat (anxiety) — the neural substrate of chronic hypervigilance under ambiguity.

---

## 7. Pain as Optimal Control — The LQG Framework

Sections 4-6 treated pain as a perceptual inference problem: given noisy signals, what is the state of the body? This section, following Seymour, Crook & Chen (2023), extends the picture to **control**: given the inferred body state, what should the organism *do* about it? Seymour et al. present a control-systems framework in which the brain acts as an optimal controller for an injured body. They write out the deterministic state-space equations and a general cost-rate function, and discuss feedforward vs. feedback control, Kalman filtering, and the exploration-exploitation dilemma.

> **Attribution note.** Seymour et al. (2023) do not use the term "LQG" (Linear-Quadratic-Gaussian) and print the state-space equations *without* the stochastic noise terms ($w$, $v$). The full LQG formulation below — including noise, the Kalman-Bucy filter with its Riccati equation, and the quadratic cost with explicit $Q$ and $R$ matrices — is the standard control-theoretic extension of their framework, reconstructed from the references they cite. This extension is warranted because their conceptual discussion (partial observability, optimal estimation, cost-based control) maps directly onto LQG, but the reader should note that the specific equations are not printed in the original paper.

### 7.1 The Plant-Sensor-Controller Model

The framework treats the injured body as a **plant** to be controlled:

- **Plant**: the biological tissue undergoing injury and healing.
- **Sensor**: nociceptors and other afferent channels that report on tissue state.
- **Controller**: the CNS, which infers the tissue state and selects protective actions.

Let $x(t) \in \mathbb{R}^n$ be the **latent injury/healing state vector** (e.g., inflammation level, mechanical integrity, neural sensitization). Let $u(t) \in \mathbb{R}^m$ be the **control input** (guarding, resting, descending analgesia). Let $y(t) \in \mathbb{R}^N$ be the **nociceptive/afferent observation**.

The state-space dynamics are:

$$
\dot{x}(t) = A\, x(t) + B\, u(t) + w(t), \qquad w \sim \mathcal{N}(0, W)
$$

$$
y(t) = C\, x(t) + D\, u(t) + v(t), \qquad v \sim \mathcal{N}(0, V)
$$

**Matrix interpretations:**
- $A$: natural healing dynamics. If all eigenvalues of $A$ have negative real parts ($\Re(\lambda_i(A)) < 0$), the tissue heals spontaneously.
- $B$: how actions modulate healing. Rest may accelerate healing ($B$ negative for rest action on inflammation state); movement may decelerate it.
- $C$: nociceptor transduction — how the hidden injury state maps to observed signals. **Central sensitization increases $C$**: the same tissue state produces a louder nociceptive signal.
- $D$: efference copy — the controller's knowledge of how its own actions affect the sensory signal.
- $W$: process noise (biological variability in healing).
- $V$: observation noise (nociceptor unreliability).

### 7.2 Kalman-Bucy State Estimation

Because the true injury state $x(t)$ is hidden, the brain must estimate it. The optimal estimator under the Gaussian-linear assumptions is the **Kalman-Bucy filter**:

$$
\dot{\hat{x}}(t) = A\,\hat{x}(t) + B\,u(t) + L\bigl(y(t) - \hat{y}(t)\bigr), \qquad \hat{y}(t) = C\,\hat{x}(t) + D\,u(t)
$$

The **Kalman gain** $L$ determines how strongly the estimator responds to the *innovation* (the discrepancy between predicted and observed sensory input):

$$
L = \Sigma\, C^\top V^{-1}
$$

where $\Sigma$ is the steady-state error covariance, solving the **filtering Riccati equation**:

$$
A\Sigma + \Sigma A^\top - \Sigma C^\top V^{-1} C \Sigma + W = 0
$$

**Connection to Section 4.** The Kalman gain $L$ is the continuous-time, multivariate generalization of the scalar Kalman gain $K = \Pi_{\text{obs}} / (\Pi_{\text{obs}} + \Pi_{\text{prior}})$ from the Bayesian pain inference. The structure is identical: the estimator corrects its prediction by a precision-weighted fraction of the prediction error.

**Physiological interpretation of $L$:**
- $L$ increases with nociceptor sensitivity (larger $C$, central sensitization) → the estimator over-weights incoming signals → prediction errors are amplified → the brain updates its injury estimate more aggressively upward. This is the control-theoretic substrate of **hyperalgesia and allodynia**.
- $L$ decreases when observation noise is high (large $V$) → the estimator trusts the prior more → the brain may fail to detect genuine changes in tissue state.

### 7.3 LQG Optimal Control for Recuperation

The controller's objective is to drive the injury state $x$ toward zero (healed) while minimizing the cost of protective behavior:

$$
\boxed{\;
J = \mathbb{E}\!\left[\int_0^\infty \bigl(x^\top Q\, x + u^\top R\, u\bigr)\, dt\right]
\;}
$$

- $Q \succeq 0$: the **biological cost of being injured** — metabolic drain, immune cost, fitness loss, vulnerability to predators.
- $R \succ 0$: the **opportunity cost of protection** — foraging foregone, mating foregone, energy spent on guarding postures.

By the **separation principle**, the optimal control uses the Kalman estimate:

$$
u^*(t) = -K\, \hat{x}(t)
$$

where $K = R^{-1} B^\top P$ and $P$ solves the **control Riccati equation**:

$$
A^\top P + P A - P B R^{-1} B^\top P + Q = 0
$$

**Interpretation:** When $\|\hat{x}\|$ is large (the brain believes the injury is severe), $u^*$ is large (strong guarding, rest, analgesia). As $\hat{x} \to 0$ (the brain believes healing is progressing), $u^* \to 0$ and normal foraging resumes.

### 7.4 Chronic Pain as Control Failure: Information Restriction

Seymour et al. (2023) explicitly describe **four** mechanisms that seed information restriction, illustrated in their Figure 4 ("Information restriction model of chronic pain"). All four are observed clinically:

**Failure 1: Maladaptive learning.** Avoidance reduces access to information about whether actions still cause pain. The agent stops sampling the environment, so it never learns that the injury has healed. This is compounded by altered reward/punishment valuation — punishment sensitivity increases post-injury, further discouraging exploratory behavior.

This is a self-reinforcing loop:
$$
\text{Guarding} \to \text{No movement} \to \text{No sensory feedback} \to \text{Innovation} = 0 \to \hat{x} \text{ stuck at "injured"} \to \text{More guarding}
$$

**Failure 2: Maladaptive model.** An incomplete generative model of afferent input fails to account for central and peripheral sensitization. Because the brain lacks a perfect "efference copy" of these facilitatory processes, the amplified nociceptive signals are interpreted as evidence of greater injury rather than as a gain artefact. This adds noise and uncertainty that biases inference toward persistent injury, because under-estimating harm carries a much higher evolutionary cost than over-estimating it.

**Failure 3: Maladaptive integration.** Nerve lesions, amputations, or other structural changes create persistent incongruent multisensory integration — the brain receives conflicting signals it cannot reconcile within its existing model. This irreducible model-mismatch prevents the system from converging on a coherent "healed" estimate.

**Failure 4: Maladaptive priors.** Incorrect or pessimistic cognitive beliefs and expectancies ("it will never get better," low perceived controllability) act as overly strong Bayesian priors that downplay the importance of incoming sensory information signaling recovery.

These four failure modes feed each other: avoidance restricts data (Failure 1), restricted data cannot correct the incomplete model (Failure 2) or resolve sensory conflict (Failure 3), and pessimistic priors (Failure 4) further discount whatever recovery evidence does arrive. This positive-feedback trap is the formal origin of the acute-to-chronic pain transition.

### 7.5 Meta-Control: Context-Dependent Tuning

Seymour et al. (2023) explicitly include a "meta-control" layer in their hierarchical model (their Figure 1a), implemented by the anterior insula and vmPFC, which regulates the lower control loops based on context — setting homeostatic priorities and balancing exploration against risk-seeking. In our LQG formalization, this corresponds to rewriting $Q$ and $R$ based on context:

> **Attribution note.** Seymour et al. discuss meta-control conceptually with a general cost-rate function $\text{Cost}[a] = \int_0^T \rho(s(t), a(t))\, dt + \rho_T$. The specific translation into $Q$ and $R$ matrices below is inferred from standard optimal control theory, not explicitly stated in their paper.

- In a safe shelter with social support: $R \uparrow$ (opportunity cost of guarding is low — it is fine to rest), $Q$ may be lowered (the urgency of healing is less acute).
- In an open, predator-rich environment: $R \downarrow$ (the opportunity cost of guarding is very high — staying still means being eaten), which drives the controller to resume movement despite injury.

**Neuromodulatory implementation:**
- **Opioids** modulate the Kalman gain $L$ (descending analgesia reduces the innovation signal) and the observation matrix $C$ (stress-induced analgesia suppresses nociceptor gain). During acute threat, opioid-mediated suppression of $L$ allows threat-escape behavior to override protection.
- **Dopamine** sets the effective $Q/R$ ratio and controls action vigor. Post-injury downshift of mesolimbic dopamine raises the effective $R$ on foraging — enforcing rest and low motivation (Gershman et al., 2024).

---

## 8. Active Inference and the Chronic Pain Trap

Active inference (Friston, 2010; Smith, Friston & Whyte, 2022) provides the most general framework in this tutorial — one that subsumes Bayesian inference (Section 4), the DDM (Section 5), threat/safety computations (Section 6), and LQG control (Section 7) as special cases. Its central claim is that both perception and action serve a single objective: **minimizing surprise** (equivalently, maximizing model evidence). When applied to pain, this framework produces the most precise account of why hypervigilance becomes self-sustaining and how chronic pain is an **active-inference trap**.

### 8.1 The Free-Energy Principle — Core Ideas

An agent maintains a **generative model** $p(o, s)$ of how hidden states $s$ in the world produce observations $o$. The agent cannot compute the true posterior $p(s \mid o)$ exactly (this requires the intractable marginal $p(o)$), so it maintains an approximate posterior $q(s)$ and minimizes the **variational free energy**:

$$
F = \underbrace{D_{\text{KL}}\bigl[q(s) \,\|\, p(s \mid o)\bigr]}_{\geq\, 0} - \log p(o)
$$

Since the KL divergence is non-negative, $F \geq -\log p(o)$, so minimizing $F$ simultaneously:
1. Makes $q(s)$ a better approximation to the true posterior (**perception**).
2. Maximizes the marginal likelihood $p(o)$ — i.e., makes observations consistent with the generative model.

**Action** enters by changing future observations. The agent selects actions (policies $\pi$) that are expected to minimize free energy in the future.

### 8.2 Expected Free Energy and the Risk-Ambiguity Decomposition

For evaluating future policies, the relevant quantity is the **Expected Free Energy (EFE)**, $G_\pi$:

$$
\boxed{\;
G_\pi = \underbrace{D_{\text{KL}}\bigl[q(o \mid \pi) \,\|\, p(o \mid C)\bigr]}_{\text{risk}} + \underbrace{\mathbb{E}_{q(s \mid \pi)}\!\left[H[p(o \mid s)]\right]}_{\text{ambiguity}}
\;}
$$

- **Risk** (first term): the divergence between the observations expected under policy $\pi$ and the agent's **preferred observations** $p(o \mid C)$. Minimizing risk drives the agent toward outcomes it prefers — reward-seeking, homeostasis, avoiding tissue damage.
- **Ambiguity** (second term): the expected entropy of the observation model — how uncertain the mapping from states to observations is. Minimizing ambiguity drives the agent to seek information, reducing uncertainty about the world.

Policy selection minimizes total EFE:

$$
\pi^* = \arg\min_\pi G_\pi
$$

The EFE beautifully captures the **exploration-exploitation tradeoff**: the risk term drives exploitation (seek preferred outcomes), while the ambiguity term drives exploration (seek information).

### 8.3 Pain Preferences and the Generative Model

In the context of pain, the generative model encodes:

- **Hidden states $s$**: injury severity, healing stage, environmental threat level.
- **Observations $o$**: nociceptive input, proprioceptive feedback, visual/auditory signals.
- **Preferred observations $p(o \mid C)$**: the agent prefers observations consistent with a healthy, uninjured body — low nociceptive input, full mobility, no threat signals.

The preference prior $C$ heavily penalizes nociceptive observations:

$$
p(o \mid C) \propto \exp(C(o)), \qquad C(o_{\text{pain}}) \ll C(o_{\text{no pain}})
$$

This means any policy that predicts future pain observations generates high risk in the EFE. The agent will therefore strongly prefer policies that avoid predicted pain.

### 8.4 The Exploration-Exploitation Dilemma in Injury Recovery

Here is where the framework reveals the vulnerability of the pain system. Consider an agent that has been injured and is in the recovery phase:

**Exploitation (minimize risk):** The agent selects policies that minimize predicted nociceptive observations. These are protective policies — rest, guard, avoid movement. They are immediately effective: the agent avoids pain-triggering observations.

**Exploration (minimize ambiguity):** The agent should also select policies that provide informative observations about the healing state. These are *active* policies — test the injured limb, attempt movement, probe the boundary of pain. They increase short-term risk (movement may produce nociceptive input) but decrease ambiguity (they provide data about whether healing has progressed).

The trap is that **exploitation directly undermines exploration**:

1. Guarding suppresses movement.
2. Without movement, nociceptive prediction errors vanish — not because the injury is healed, but because the system has stopped sampling the relevant observations.
3. Without prediction errors, the generative model cannot update — it remains stuck on the "injured" hypothesis.
4. The stuck model continues to predict pain, which drives continued guarding.

This is Seymour et al.'s (2023) **information restriction hypothesis** expressed in active-inference terms.

### 8.5 Chronic Pain as a Self-Confirming Attractor

Formally, the chronic pain state is a **local minimum of EFE** in which:

1. The prior belief over hidden states $q(s)$ is concentrated on "injured" with very high precision.
2. All policies that would provide corrective information (movement, exploration) have high risk because they predict nociceptive observations that diverge from preferred observations $p(o \mid C)$.
3. The ambiguity term is technically high (the agent is uncertain about whether it has healed), but it is overwhelmed by the risk term — the agent would rather stay uncertain and avoid pain than accept short-term pain to gain information.

The system has fallen into a **dark-room problem** on interoception. The classic dark-room problem asks: why doesn't an agent that minimizes surprise just sit in a dark room forever? The answer for exteroceptive agents is that they have preferences for rich sensory experience ($C$ encodes preferences for food, social interaction, etc.). But for interoceptive pain channels, the preferences are *monotonically negative* — there is no interoceptive observation the agent actively seeks. The dark room (complete avoidance of all nociceptive input) is genuinely the EFE minimum under a stuck injury prior.

### 8.6 Hypervigilance in Active-Inference Terms

Hypervigilance maps onto the following parameter configuration:

1. **Excessive prior precision on injury states**: $q(s = \text{injured})$ has very high precision (low variance). The agent is extremely confident it is injured and resistant to evidence otherwise.
2. **Elevated preference precision**: $C(o_{\text{pain}})$ is very strongly negative. Even mild nociceptive observations are catastrophically costly.
3. **Risk-dominated EFE**: The risk term dwarfs the ambiguity term, so the agent never selects exploratory policies. All behavior is defensive.
4. **Collapsed epistemic action**: The agent has effectively abandoned the information-seeking component of its policy. It no longer probes its environment to test whether the injury has healed.

The catastrophizing literature maps directly onto this: catastrophizing = excessive prior precision + elevated preference precision, producing an agent that is locked into risk-minimization and cannot engage in the epistemic actions that would allow recovery (Wiech, 2016).

### 8.7 Connecting Active Inference to the Other Frameworks

Active inference is not a competing framework — it *contains* the others:

| Framework | Active Inference Equivalent |
|:---|:---|
| Bayesian inference (Section 4) | Perception as variational inference: minimizing $F$ w.r.t. $q(s)$ |
| DDM (Section 5) | Evidence accumulation is belief updating; boundary crossing is policy commitment when EFE difference exceeds threshold |
| Safety/threat (Section 6) | Safety $\approx$ low risk in EFE; threat $\approx$ high risk; controllability $\approx$ availability of low-risk policies |
| LQG control (Section 7) | LQG is the continuous, Gaussian, quadratic-cost special case of EFE minimization; Kalman filter is the Gaussian variational posterior |

The value of active inference is that it provides a **single, principled objective** (minimize EFE) that naturally captures the tension between risk and ambiguity — the tension that is at the heart of chronic pain and hypervigilance.

---

## 9. Defining Hypervigilance Computationally

The preceding sections developed five computational frameworks, each illuminating different aspects of pain. This section synthesizes them into a **unified computational account of hypervigilance** — showing that what clinicians observe as a single phenomenon is, computationally, a coherent pattern of parameter deviations that spans every level of the decision-making hierarchy.

### 9.1 Clinical Phenomenology

Before the formal account, a brief grounding in what hypervigilance looks like clinically. Following Crombez, Van Damme & Eccleston (2005), hypervigilance is strictly an *attentional* process — "an unintentional and efficient process that emerges when the threat value of pain is high." Its core manifestations are:

- **Attentional capture**: pain-related cues (words, images, bodily sensations) automatically seize attention, disrupting ongoing tasks.
- **Difficulty disengaging**: once attention is captured by pain, it is hard to redirect it.
- **Interpretive bias**: ambiguous stimuli are more likely to be classified as threatening or painful.
- **Body scanning**: persistent monitoring of the body for threat signals.
- **Behavioral inhibition**: avoidance of pain-relevant activities, even when objectively safe (Van Damme et al., 2010).
- **Persistent anxiety**: a generalized expectation of pain that pervades daily life.

Hypervigilance co-occurs with but is computationally distinct from **central sensitization** (hyperalgesia, allodynia), which operates at the sensory-gain level rather than the attentional level. In the computational framework, sensitization increases $\Pi_{\text{obs}}$ for nociceptive channels; hypervigilance amplifies the *use* of that information in decision-making (prior precision, boundary, urgency). The two reinforce each other but are dissociable.

These manifestations form a coherent pattern that every computational framework captures from its own vantage point.

### 9.2 Hypervigilance Across the Computational Stack

| Framework | Normal (Adaptive) | Hypervigilant (Maladaptive) | Key Parameter |
|:---|:---|:---|:---|
| **Bayesian inference** (Section 4) | Prior and observation precision balanced; posterior tracks tissue state accurately | $\Pi_{\text{prior}} \uparrow$ on injury; $\Pi_{\text{obs}} \uparrow$ for nociceptive channels; asymmetric $K$ (absorbs threat, rejects recovery) | Prior precision $\Pi_{\text{prior}}$, Kalman gain $K$ |
| **DDM** (Section 5) | Starting point centered ($z = a/2$); boundary appropriate for context | Starting point biased toward pain ($z > a/2$); boundary lowered ($a \downarrow$); urgency elevated | Starting point $z$, boundary $a$, urgency $u(t)$ |
| **Threat/Safety** (Section 6) | Safety computed and gates defensive release; threat generalization bounded | Safety undercomputed (low $K_{\text{control}}$); threat overgeneralized (low $c$ in Shepard gradient); extreme ambiguity aversion ($\beta \gg 0$) | Safety $S(t)$, generalization $c$, ambiguity aversion $\beta$ |
| **LQG control** (Section 7) | Kalman gain $L$ calibrated; cost balance $Q/R$ appropriate; information flows freely | Four failure modes: maladaptive learning (avoidance restricts data), maladaptive model ($C$ inflated without efference copy), maladaptive integration (sensory conflict), maladaptive priors ($\hat{x}$ stuck) | Observation matrix $C$, gain $L$, information flow, prior precision |
| **Active inference** (Section 8) | Risk and ambiguity balanced in EFE; agent explores to resolve uncertainty | Risk dominates EFE; prior precision on injury $\to \infty$; epistemic action collapsed; dark-room trap | Prior precision, risk/ambiguity balance |

### 9.3 The Unifying Theme: Precision Dysregulation

Across all five frameworks, hypervigilance reduces to a single computational motif: **precision dysregulation** — the mis-calibration of confidence weights in the brain's inference and control machinery.

In the Bayesian framework, precision is explicit ($\Pi_{\text{prior}}$, $\Pi_{\text{obs}}$). In the DDM, it maps onto the starting point and boundary (which encode prior confidence and evidence requirements). In threat computation, it appears as the generalization gradient and ambiguity aversion (which encode confidence in threat and safety estimates). In LQG, it is the Kalman gain and the observation matrix (which encode sensor reliability and state-estimation confidence). In active inference, it is the precision on beliefs and preferences (which determine the risk-ambiguity balance).

The convergence is not accidental. These frameworks are mathematically nested:

$$
\text{Active Inference} \supset \text{LQG} \supset \text{Bayesian Inference} \equiv \text{DDM (limiting case)}
$$

Precision dysregulation at the active-inference level propagates down through all the special cases, producing the full clinical syndrome of hypervigilance.

### 9.4 A Formal Definition

We can now state a formal computational definition of hypervigilance:

> **Hypervigilance** is a state of the agent's generative model in which:
> 1. The precision of the injury/threat prior is elevated ($\Pi_{\text{prior}}^{\text{injury}} \gg \Pi_{\text{prior}}^{\text{safe}}$), encoding strong confidence in ongoing danger.
> 2. The precision of nociceptive observations is elevated ($\Pi_{\text{obs}}^{\text{noc}} \uparrow$), amplifying the gain on pain-related sensory channels.
> 3. The precision of safety/recovery evidence is reduced ($\Pi_{\text{obs}}^{\text{recovery}} \downarrow$), discounting signals of healing.
> 4. The expected free energy is dominated by risk, suppressing epistemic (exploratory) policy selection.
>
> The combined effect is a **self-reinforcing attractor**: the agent avoids the very observations that could correct its injury belief, locking the generative model into a persistent threat state that drives continued protective behavior.

This definition is:
- **Measurable**: each parameter maps onto observable behavioral or neural quantities (DDM parameters from HDDM fitting, Kalman gain from model-based fMRI, prior precision from confidence reports).
- **Individualized**: the pattern of parameter deviations constitutes a computational fingerprint unique to each patient.
- **Mechanistic**: it specifies not just *what* is wrong but *why* — the causal chain from precision dysregulation to behavioral inhibition to information restriction to persistence.

---

## 10. From Theory to Engineering — Computational Phenotyping

The computational account of hypervigilance (Section 9) is not merely theoretical — it is the specification for a **clinical engineering programme**. Mahajan & Seymour (2025) argue that pain neuroscience must become *pain engineering*, with a tight loop between forward models (simulate and predict) and reverse models (fit to data and decode).

### 10.1 Forward Engineering: The Generative Stack

Forward engineering builds mechanistic models from first principles and uses them to generate predictions:

1. **Bayesian pain inference** (Section 4): generates predictions about pain perception given prior and sensory precision parameters.
2. **RL with aversive prediction error**: with pain magnitude as negative reward, generates predictions about avoidance learning and threat conditioning.

$$
\delta_t^{\text{RPE}} = r_t + \gamma\, \max_{a'} Q(s_{t+1}, a') - Q(s_t, a_t)
$$

3. **LQG controller** (Section 7): generates predictions about protective behavior given injury dynamics and cost parameters.
4. **Model Predictive Control (MPC)**: extends LQG to finite-horizon optimization with explicit constraints:

$$
\min_{u_{t:t+H}} \sum_{k=t}^{t+H} \left(\hat{x}_{k|t}^\top Q\, \hat{x}_{k|t} + u_k^\top R\, u_k\right)
$$

subject to $\hat{x}_{k+1|t} = A\hat{x}_{k|t} + Bu_k$ and safety constraints $u_{\min} \leq u_k \leq u_{\max}$. The first action $u_t^*$ is executed, then the optimization is re-run at $t+1$ (receding horizon).

### 10.2 Reverse Engineering: Fitting Latent Computations

Reverse engineering fits the forward models to individual patient data, extracting the computational parameters that characterize each person's pain system:

**HDDM for perceptual parameters.** As described in Section 5.4, hierarchical Bayesian estimation of DDM parameters from choice-RT data yields subject-level drift ($v_j$), boundary ($a_j$), starting point ($z_j$), and non-decision time ($t_{0,j}$).

**Linear Dynamical System identification via EM.** For the LQG model (Section 7), the system matrices $(A, B, C, D, W, V)$ can be estimated from multimodal time-series (neural, behavioral, autonomic) using Expectation-Maximization:

- **E-step**: run Kalman smoother with current parameters to obtain posterior state estimates $\hat{x}_t$ and covariances.
- **M-step**: closed-form updates, e.g.:

$$
C^{\text{new}} = \left(\sum_t y_t\, \hat{x}_t^\top\right)\left(\sum_t (\hat{\Sigma}_t + \hat{x}_t \hat{x}_t^\top)\right)^{-1}
$$

Iterate until convergence. This reverse-engineers the patient's injury dynamics ($A$), nociceptor sensitivity ($C$), and control effectiveness ($B$) from observed data.

**Variational autoencoders for nonlinear latent axes.** When the linear assumption is inadequate, VAEs with encoder $q_\phi(z \mid x)$ and decoder $p_\theta(x \mid z)$ discover low-dimensional manifolds tracking pain trajectories, providing continuous computational biomarkers (Mahajan & Seymour, 2025).

### 10.3 Computational Phenotyping of Hypervigilance

Once parameters are fit, each patient's **computational phenotype** is a vector:

$$
\Theta_j = \bigl(\Pi_{\text{prior},j},\, \Pi_{\text{obs},j},\, \alpha_j,\, \gamma_j,\, A_j,\, B_j,\, C_j,\, K_j\bigr)
$$

This replaces vague clinical labels like "catastrophizer" with machine-readable parameter deviations. A hypervigilant phenotype would show:
- Elevated $\Pi_{\text{prior}}$ (strong injury prior)
- Elevated $C$ (sensitized observation matrix)
- Biased starting point $z_j > a_j/2$ (DDM bias toward pain)
- Low boundary $a_j$ (impulsive pain decisions)

### 10.4 Digital Twins and Closed-Loop Intervention

A **digital twin** is a patient-specific POMDP parameterized by $\Theta_j$. Any candidate intervention $\mathcal{I}$ can be simulated in silico:

$$
\widehat{\Delta}_\mathcal{I} = \mathbb{E}_{\Theta_j}\!\left[\widehat{\text{Pain}}_{t+\Delta t}(\mathcal{I}) - \widehat{\text{Pain}}_{t+\Delta t}(\text{no intervention})\right]
$$

For **closed-loop neuromodulation** (e.g., spinal cord stimulation), the stimulation device becomes a real-time MPC controller:

$$
\min_{u_{t:t+H}} \sum_{k=t}^{t+H} \left((\hat{x}_k - x^{\text{target}})^\top Q\, (\hat{x}_k - x^{\text{target}}) + u_k^\top R\, u_k\right)
$$

subject to safety constraints. The state estimate $\hat{x}_t$ comes from the Kalman/VAE pipeline, and the stimulation parameters are optimized in real time.

This is the translational bridge: the computational theory of hypervigilance (Sections 4-9) becomes the specification for a *device stack* — sensors that estimate the patient's computational state, controllers that optimize intervention, and digital twins that predict outcomes before any physical treatment is applied.

---

## 11. The Grid-World Pain Platform — Architecture Overview

This section describes the **GridWorld Pain** environment — an RL research platform for interoceptive AI — and how its architecture maps onto the computational frameworks developed in Sections 4-10. The platform is implemented in JAX with full `vmap` parallelism (128-1000+ environments), providing the computational substrate for studying hypervigilance in silico.

### 11.1 Environment Overview

The agent inhabits a 10x10 grid populated with:
- **Food resources** (4): provide nutrition when consumed.
- **Danger zones** (8): cause injury on contact (damage sampled from [min, max]).
- **Predators** (3): active threat agents with patrol/hunt/return state machine, Manhattan pursuit, stamina, and attack delay.
- **Obstacles** (rocks: 12, bushes: 20): blocking/concealment.
- **Neutral animals** (rabbits: 5): non-threatening entities.

The agent has 6 actions: move in 4 cardinal directions, rest, and eat.

### 11.2 The Body Model — Latent States

The agent maintains three homeostatic variables that serve as the **latent state** in the control-theoretic sense of Section 7:

| Variable | Role | LQG Analogue |
|:---|:---|:---|
| **Injury** ($x_{\text{inj}}$) | Tissue damage level; increases from dangers/predators/obstacles; recovers via exponential streak-based healing | Primary state variable $x(t)$ in the LQG plant |
| **Nutrition** ($x_{\text{nut}}$) | Energy reserves; depleted by metabolic cost, replenished by eating food | Resource state; enters the $Q$ matrix cost |
| **Satiation** ($x_{\text{sat}}$) | Derived from nutrition: $\text{sat} = \text{max\_sat} \times (N/\text{max\_N})^k$ | Secondary state; set-point deviation enters the drive |

**Healing dynamics** correspond to the $A$ matrix (natural recovery):
- Injury recovery is exponential, accelerated by consecutive rest steps (`rest_streak`): the longer the agent rests, the faster it heals.
- A smoothing ring buffer (`injury_buffer`) implements temporal averaging of injury signals.

**Termination** occurs via: starvation (nutrition = 0), overeating, or lethal injury — encoding the fundamental survival constraint that makes the agent's decision problem non-trivial.

### 11.3 The Sensory System — Observations as Nociceptive Channels

The agent receives a 9-modality observation vector, directly analogous to the nociceptive and exteroceptive channels in the computational pain literature:

| # | Sensor | Dim | Computational Analogue |
|:-:|:---|:---|:---|
| 1 | **Injury** | 1 | Interoceptive nociception — the agent's own damage state (normalized $x_{\text{inj}} / x_{\text{max}}$) |
| 2 | **Nutrition** | 1 | Interoceptive hunger signal |
| 3 | **Satiation** | 1 | Interoceptive satiety signal |
| 4 | **Extero Nociception** | 1 | Exteroceptive nociception — maximum intensity from danger/predator/obstacle overlap and collision |
| 5 | **Olfaction** | 5 | Chemical gradient sensing — distance-weighted sum of entity properties |
| 6 | **Collision** | varies | Somatosensory — binary Manhattan diamond of obstacle/boundary contacts |
| 7 | **Proprioception** | action_dim | Motor efference copy — one-hot of last action |
| 8 | **Visual** | varies | Exteroceptive — 8-channel Manhattan diamond (terrain, entities) |
| 9 | **Location** | 2 | Spatial self-localization (normalized coordinates) |

### 11.4 Perceptual Noise — Precision as a Configurable Parameter

The **perceptual noise system** (`apply_perceptual_noise()` in `sensor.py`) implements per-modality Gaussian noise with three modes:

| Mode | Formula | Computational Analogue |
|:---|:---|:---|
| 0: None | No noise | Perfect observation ($\Pi_{\text{obs}} = \infty$) |
| 1: Constant | $o' = o + \epsilon$, $\epsilon \sim \mathcal{N}(0, \sigma_{\text{base}}^2)$ | Fixed observation precision ($\Pi_{\text{obs}} = 1/\sigma_{\text{base}}^2$) |
| 2: State-dependent | $o' = o + \epsilon$, $\epsilon \sim \mathcal{N}(0, [\sigma_{\text{base}} (1 + \kappa \cdot x_{\text{inj}}/x_{\text{max}})]^2)$ | **Injury-dependent precision** — the $C$ matrix sensitization from Seymour's LQG |

Mode 2 is the most important for hypervigilance research: it couples the observation noise to the injury state, so that **being injured changes the quality of sensory information**. The parameter $\kappa$ (`injury_scale`) controls the strength of this coupling.

Per-modality clipping (`clip_min`, `clip_max`) and modality-specific $\sigma_{\text{base}}$ values allow fine-grained control over which sensory channels are affected and how strongly.

### 11.5 Agent Architectures

Three architectures are available, each with different relevance to the computational pain frameworks:

**RecurrentPPO** (primary): Hierarchical observation encoder with per-sensor MLPs, body hub (intero + nociception + collision), association hub (proprioception + olfaction), and fusion layer, followed by GRU/LSTM recurrence. The hierarchical structure mirrors the brain's sensory processing hierarchy.

**DreamerV3** (experimental): World model with RSSM (GRU deterministic + categorical stochastic), posterior $q(z \mid h, o)$ and prior $p(z \mid h)$. The KL divergence between posterior and prior is a direct analogue of the **prediction error** in predictive coding. The 15-step imagination rollouts for actor/critic training implement a form of **model-based planning** similar to the Bellman backups in Section 6.4.

**Neuromodulation** (experimental): `HierarchicalNeuromodulator` with percept pathway (satiation, nutrition, injury, collision $\to$ $\phi_p$) and memory pathway (prior hidden state $\to$ $\phi_m$), producing multiplicative gates ($\gamma_{\text{unimodal}}$, $\gamma_{\text{body}}$, $\gamma_{\text{association}}$). These gates are the closest existing analogue to **precision-weighting** in the Bayesian framework.

### 11.6 Reward Structure — The Cost Function

The reward implements two modes that correspond to different formulations of the LQG cost:

**Survival mode**: $r = +1.0$ for eating food, $-\text{death\_penalty}$ for death. This is a sparse reward that encodes the most basic $Q$ matrix: only terminal states matter.

**Homeostatic mode**: $r = \text{prev\_drive} - \text{curr\_drive}$, where:

$$
\text{drive} = \left\| \begin{pmatrix} \text{satiation} - \text{setpoint} \\ \text{injury} \end{pmatrix} \right\|_2
$$

This is a continuous cost that penalizes deviations from homeostatic set-points — a direct implementation of the LQG cost $x^\top Q\, x$ where $Q$ weights the importance of each homeostatic variable and the set-point defines the target state $x^* = 0$.

The `eating_reward_penalty` and `metabolic_cost` serve as the $R$ matrix — the opportunity cost of actions.

---

## 12. Mapping Computational Pain Models to the Grid-World

This section provides a detailed mapping from each computational framework to specific components of the GridWorld Pain architecture, identifying where the frameworks are already implemented, where partial implementations exist, and where extensions are needed.

### 12.1 Bayesian Inference $\to$ Perceptual Noise

**Framework concept**: The posterior pain experience is a precision-weighted combination of prior expectation and sensory observation (Section 4.2).

**Grid-world mapping**: The perceptual noise system (Section 11.4, Mode 2) implements the *observation side* of this inference. When `injury_scale` ($\kappa$) is positive, higher injury increases observation noise, which in the Bayesian framework corresponds to **lowering sensory precision** $\Pi_{\text{obs}}$:

$$
\sigma_{\text{obs}}^2 = \sigma_{\text{base}}^2 \cdot (1 + \kappa \cdot x_{\text{inj}}/x_{\text{max}})^2 \quad \Rightarrow \quad \Pi_{\text{obs}} = \frac{1}{\sigma_{\text{obs}}^2} \downarrow \text{ as injury } \uparrow
$$

This means that an injured agent receives noisier observations, which should push its internal beliefs toward its priors (the GRU/RSSM hidden state). This is a form of **state-dependent precision modulation** — the sensory reliability changes with the body state.

**Important subtlety**: In the clinical literature, central sensitization *increases* sensory precision for nociceptive channels (amplifying pain signals). The current noise model does the opposite — it *decreases* precision (adds noise) when injured. This could model the *non-nociceptive* channels (vision, olfaction become less reliable when injured due to attentional capture by pain), while nociceptive channels (injury sensor, extero nociception) could use a different $\kappa$ with the opposite sign. The per-modality noise configuration already supports this differential treatment.

### 12.2 Prediction Error $\to$ World Model (DreamerV3)

**Framework concept**: Pain perception is driven by prediction errors — the discrepancy between expected and observed nociceptive input (Section 4.3).

**Grid-world mapping**: The DreamerV3 RSSM generates explicit predictions and prediction errors:

- **Prior** $p(z_t \mid h_t)$: the world model's prediction of the next latent state *before* seeing the observation.
- **Posterior** $q(z_t \mid h_t, o_t)$: the updated belief *after* incorporating the observation.
- **KL divergence** $D_{\text{KL}}[q \| p]$: the prediction error — how much the observation surprised the model.

This KL divergence is already computed and minimized during world model training. It is the direct analogue of the nociceptive prediction error $\delta = o - \mu_{\text{prior}}$ from Section 4.3.

The **observation decoder loss** (how well the model reconstructs the observation from the latent state) provides a per-modality measure of prediction quality. High reconstruction error on nociceptive channels specifically indicates that the model failed to predict pain-related inputs — a computational signature of surprising injury events.

### 12.3 Precision-Weighting $\to$ Neuromodulation

**Framework concept**: Precision weights determine how strongly different information channels influence the posterior (Section 4.4). Attention amplifies sensory precision; expectation amplifies prior precision.

**Grid-world mapping**: The `HierarchicalNeuromodulator` produces **multiplicative gates** ($\gamma_{\text{unimodal}}$, $\gamma_{\text{body}}$, $\gamma_{\text{association}}$) that scale the output of each encoder phase:

$$
\text{encoded\_phase} \leftarrow \text{encoded\_phase} \times \gamma_{\text{phase}}
$$

These gates are the most direct analogue of precision-weighting in the architecture:

- $\gamma_{\text{body}}$ scales the body hub output (intero + nociception + collision) — this is the **interoceptive precision** parameter.
- $\gamma_{\text{unimodal}}$ scales individual sensor encodings — this is the **per-modality sensory precision**.
- $\gamma_{\text{association}}$ scales the association hub (proprioception + olfaction) — this is the **cross-modal association precision**.

The **percept pathway** ($\phi_p$), which takes (satiation, nutrition, injury, collision) as input, computes the modulation signal from the current body state. This means the interoceptive state *directly controls* the precision of sensory processing — exactly the mechanism by which injury should modulate attention in the hypervigilance account.

The **memory pathway** ($\phi_m$), which takes the prior hidden state as input, provides a *history-dependent* modulation. This captures the idea that precision is not only a function of the current state but also of prior beliefs and expectations.

### 12.4 LQG Cost $\to$ Reward Structure

**Framework concept**: The LQG objective (Section 7.3) balances the cost of being injured ($Q$) against the cost of protection ($R$).

**Grid-world mapping**: The homeostatic reward implements this directly:

$$
r_t = \underbrace{(\text{drive}_{t-1} - \text{drive}_t)}_{\text{reduction in } x^\top Q\, x} - \underbrace{\text{eating\_penalty}}_{\text{action cost } u^\top R\, u}
$$

where $\text{drive} = \|(\text{sat} - \text{setpoint},\, \text{injury})\|_2$.

The implicit $Q$ matrix weights injury and satiation-deviation equally (both enter the L2 norm). The $R$ matrix is represented by `metabolic_cost` (the cost of moving) and `eating_reward_penalty` (the cost of eating actions). `death_penalty` corresponds to an infinite terminal cost for reaching absorbing injury/starvation states.

The **rest action** is the grid-world analogue of guarding: it has zero metabolic cost, accelerates healing (reduces $x_{\text{inj}}$), but produces no food and exposes the agent to nearby threats without moving away. The agent must learn the optimal rest-vs-forage tradeoff — exactly the $Q/R$ balance in the LQG framework.

### 12.5 Active-Inference Trap $\to$ Information Restriction

**Framework concept**: Guarding restricts sensory feedback, preventing the model from detecting recovery (Section 8.4-8.5).

**Grid-world mapping**: When the agent selects the rest action:

1. **No movement occurs** → the visual and collision observation channels report the same local scene every step → no new spatial information.
2. **No food interaction** → olfaction gradients are static → no new chemical information.
3. **Injury decreases** (healing) → but the agent's injury sensor reports the same declining trajectory regardless of action choice.

The key question is whether the agent's world model can detect healing *without* actively probing the environment. In the current architecture:

- The **injury sensor** directly reports the normalized injury level, so the model always has access to the recovery signal. This partially breaks the information-restriction trap.
- However, the **perceptual noise** in Mode 2 means that the injury sensor's own reliability decreases when injured (noise increases with $\kappa \cdot x_{\text{inj}}$). If noise is high enough, the recovery signal is drowned out.
- More importantly, the agent cannot learn **what is safe to approach** by resting. It must actually move toward food sources (past dangers, through predator territories) to learn that certain paths are navigable. This spatial knowledge is the grid-world analogue of the proprioceptive feedback that Seymour et al. (2023) identify as critical for injury recovery.

Thus, the information-restriction mechanism operates at the *environmental learning* level: a resting agent loses information about the changing spatial threat landscape, even if it maintains information about its own body state.

---

## 13. Implementation Roadmap — Hypervigilance in Silico

### 13.1 Current State: What Already Exists

The GridWorld Pain platform already implements several components of the hypervigilance framework:

| Component | Implementation | Status |
|:---|:---|:---|
| State-dependent sensory noise | `apply_perceptual_noise()` Mode 2: $\sigma \propto (1 + \kappa \cdot \text{injury})$ | Implemented |
| Per-modality noise control | Independent $\sigma_{\text{base}}$, $\kappa$, clip per sensor | Implemented |
| Multiplicative precision gates | `HierarchicalNeuromodulator` $\gamma$ gates | Implemented (experimental) |
| Interoceptive state input to gates | Percept pathway: (sat, nut, inj, collision) $\to \phi_p$ | Implemented |
| History-dependent modulation | Memory pathway: prior hidden $\to \phi_m$ | Implemented |
| Homeostatic reward (LQG cost) | $r = \text{prev\_drive} - \text{curr\_drive}$ | Implemented |
| Rest action (guarding) | Action 4; zero movement, accelerated healing | Implemented |
| World model with prediction errors | DreamerV3 RSSM: $D_{\text{KL}}[q \| p]$ | Implemented (experimental) |

### 13.2 Gap Analysis: What is Missing

To fully model hypervigilance as defined in Section 9.4, several extensions are needed:

**Gap 1: Injury-history-dependent precision (not just current injury).**
The current noise model scales with *instantaneous* injury. But hypervigilance is characterized by *persistent* sensitization — the system remains hypervigilant even after injury begins to heal. This requires a precision parameter that tracks **cumulative injury history**, not just the current level.

*Proposed mechanism*: Add an `injury_history` state variable that accumulates injury exposure with slow decay:

$$
h_{t+1} = (1 - \lambda)\, h_t + \text{injury\_increment}_t
$$

Then use $h_t$ (not just $x_{\text{inj},t}$) to modulate noise: $\sigma = \sigma_{\text{base}} \cdot (1 + \kappa \cdot h_t / h_{\text{max}})$. The decay rate $\lambda$ controls how quickly hypervigilance fades after injury resolution — slow decay = persistent hypervigilance = chronic pain vulnerability.

**Gap 2: Asymmetric precision for nociceptive vs. non-nociceptive channels.**
Hypervigilance involves *amplified* nociceptive precision but *reduced* non-nociceptive precision (attentional capture redirects processing resources). The current system applies noise uniformly or with the same $\kappa$ sign.

*Proposed mechanism*: Use negative $\kappa$ for nociceptive channels (injury, extero_noc: noise *decreases* with injury → precision *increases*) and positive $\kappa$ for non-nociceptive channels (visual, olfaction: noise *increases* with injury → precision *decreases*). This creates the attentional capture pattern: injured agent has sharp pain sensing but blurred environmental awareness.

**Gap 3: Adaptive decision boundary / urgency mechanism.**
The DDM account (Section 5.5) requires an adjustable decision boundary that collapses under threat. The current architecture has no explicit decision threshold — the actor network directly outputs action probabilities.

*Proposed mechanism*: Introduce a **temperature parameter** in the actor's softmax that is modulated by the neuromodulator:

$$
\pi(a \mid s) = \text{softmax}(z / \tau), \qquad \tau = f(\phi_p, \phi_m)
$$

Low $\tau$ (high urgency, hypervigilant) → peaked action distribution → deterministic, reflexive behavior. High $\tau$ (low urgency, safe) → flat distribution → exploratory behavior. The neuromodulator already computes a `temperature = sigmoid(phi_p)` signal — this could be wired to the actor.

**Gap 4: Meta-controller for Q/R reweighting.**
The LQG framework (Section 7.5) requires a meta-controller that adjusts the $Q/R$ balance based on context. Currently, the reward structure is fixed at initialization.

*Proposed mechanism*: Make the homeostatic drive weights context-dependent:

$$
\text{drive}_t = \left\| \begin{pmatrix} w_{\text{sat}}(c_t) \cdot (\text{sat} - \text{setpoint}) \\ w_{\text{inj}}(c_t) \cdot \text{injury} \end{pmatrix} \right\|_2
$$

where $c_t$ is a context signal (e.g., proximity to shelter, recent predator encounters) and $w_{\text{sat}}, w_{\text{inj}}$ are learned or rule-based weights. In a safe context, $w_{\text{inj}}$ decreases (healing is less urgent, foraging more important). Under threat, $w_{\text{inj}}$ increases (protection dominates).

**Gap 5: Explicit belief-state tracking for injury estimation.**
The Kalman filter framework (Section 7.2) requires the agent to maintain an explicit estimate of the injury state under uncertainty. The current GRU hidden state implicitly tracks this, but there is no explicit Bayesian inference step.

*Proposed mechanism*: For the DreamerV3 architecture, the RSSM already maintains a latent state that should encode injury beliefs. The key question is whether the world model learns to distinguish "still injured but resting" from "healed and resting" — both produce similar observation sequences (low nociceptive input, static visual field). Training the world model with long rollouts that span full injury-recovery cycles would encourage this distinction.

### 13.3 Experimental Design: Ablation Studies

To isolate the contribution of each hypervigilance mechanism, a systematic ablation study is proposed:

| Experiment | Manipulation | Prediction |
|:---|:---|:---|
| **Baseline** | No noise, no neuromodulation | Agent learns optimal forage/rest balance; no hypervigilance |
| **+Noise (constant)** | Mode 1: fixed $\sigma$ per modality | Reduced performance; no injury-dependent behavior change |
| **+Noise (state-dep)** | Mode 2: $\sigma \propto (1 + \kappa \cdot \text{injury})$ | Agent becomes more cautious post-injury; performance recovery slower |
| **+Injury history** | Gap 1: $\sigma \propto (1 + \kappa \cdot h_t)$ | Persistent caution even after healing; slower return to baseline foraging |
| **+Asymmetric $\kappa$** | Gap 2: negative $\kappa$ for noc, positive for non-noc | Amplified injury avoidance; reduced environmental exploration when injured |
| **+Neuromodulation** | Full `HierarchicalNeuromodulator` | Learned precision gates; expect injury-dependent gating patterns |
| **+Temperature** | Gap 3: neuromodulator controls actor temperature | Reflexive (low-entropy) behavior under injury; exploratory under health |
| **Full hypervigilance** | All of the above | Complete syndrome: persistent caution, attentional capture, behavioral inhibition, delayed recovery |

The key metrics are:
- **Survival steps**: how long the agent survives (primary performance metric).
- **Rest-after-injury ratio**: proportion of rest actions in the N steps after an injury event (measures guarding behavior).
- **Recovery time**: steps from injury to resumption of full foraging (measures information restriction).
- **Neuromodulator $\gamma$ trajectories**: how the precision gates evolve during and after injury (measures learned precision modulation).
- **World model KL divergence**: prediction error magnitude during and after injury (measures surprise dynamics).

---

## Annotated Bibliography

### Core References (from the 13-paper review)

1. **Gold, J. I. & Shadlen, M. N. (2007).** The neural basis of decision making. *Annual Review of Neuroscience*, 30, 535-574.
   Foundational review establishing that perceptual decisions are implemented by neural circuits that accumulate noisy evidence (in LIP) from sensory neurons (in MT) to a stereotyped firing-rate bound. Introduces the SPRT/DDM framework as the computational theory of decision-making. [Sections 1, 5]

2. **Hanks, T. D. & Summerfield, C. (2017).** Perceptual decision making in rodents, monkeys, and humans. *Neuron*, 93(1), 15-31.
   Cross-species review showing that evidence accumulation to bound is conserved across rodents, primates, and humans. Introduces the Centro-Parietal Positivity (CPP) as the human EEG correlate of the decision variable. Important for establishing the universality of the decision framework. [Sections 1, 5]

3. **O'Connell, R. G. & Kelly, S. P. (2021).** Neurophysiology of human perceptual decision-making. *Annual Review of Neuroscience*, 44, 495-519.
   Establishes the CPP as a supramodal, neurally-informed DDM signal in humans. Hierarchical Bayesian DDM fitting allows decomposition of individual differences into drift, bound, and starting-point components. [Sections 5, 10]

4. **Cisek, P. (2021).** An evolutionary perspective on the neural basis of decision making. In *The Neural Basis of Mentalizing* (pp. 3-22). Springer.
   Proposes that decision-making evolved from affordance competition with urgency-gating, not from a dedicated "decision center." The urgency signal that collapses boundaries under time pressure is key to understanding hypervigilance as boundary collapse under threat. [Sections 5, 6]

5. **Fleming, S. M. (2024).** Metacognition and confidence: A review and synthesis. *Annual Review of Psychology*, 75, 241-268.
   Reviews metacognitive monitoring (confidence, meta-$d'$) as a second-order inference about the quality of first-order decisions. Individual differences in interoceptive meta-$d'$ predict vulnerability to chronic pain via miscalibrated confidence in body-state inference. [Sections 4, 10]

6. **Gershman, S. J., Uchida, N., & Bhatt, M. (2024).** Explaining dopamine through prediction errors and beyond. *Nature Neuroscience*.
   Extends the reward-prediction-error model of dopamine to include average-reward rate, precision, and context-dependent RPE. Relevant for understanding neuromodulatory control of the $Q/R$ ratio and action vigor in post-injury pain. [Sections 7, 10]

7. **Smith, R., Friston, K. J., & Whyte, C. J. (2022).** A step-by-step tutorial on active inference and its application to empirical data. *Journal of Mathematical Psychology*, 107, 102632.
   The most accessible technical introduction to active inference (POMDP + VFE/EFE). Provides the formal machinery (generative model, variational posterior, expected free energy) that Section 8 builds upon. [Section 8]

8. **LeDoux, J. E. & Daw, N. D. (2018).** Surviving threats: Neural circuit and computational implications of a new taxonomy of defensive behaviour. *Nature Reviews Neuroscience*, 19(5), 269-282.
   Replaces "fear" with a 6-level computational hierarchy from reflexes to deliberation, each with distinct RL algorithms and neural circuits. The Pavlovian-instrumental interaction and model-free/model-based arbitration under threat are central to understanding why hypervigilance promotes behavioral inhibition. [Section 6]

9. **Tashjian, S. M., Zbozinek, T. D., & Mobbs, D. (2021).** A decision architecture for safety computations. *Trends in Cognitive Sciences*, 25(5), 342-354.
   Argues that safety is a distinct, actively computed decision variable (not inverted threat), encoded in anterior vmPFC. Perceived control decouples safety from objective threat. Hypervigilance is reframed as a failure of safety computation, not just overactive threat detection. [Section 6]

10. **Levy, I. & Schiller, D. (2021).** Neural computations of threat. *Trends in Cognitive Sciences*, 25(4), 274-287.
    Unifies threat processing across five computational variables (value, probability, ambiguity, imminence, controllability). The individual-difference vector $(\alpha, \beta, k, \eta)$ provides the computational fingerprint framework for anxiety and hypervigilance phenotyping. [Section 6]

11. **Wiech, K. (2016).** Deconstructing the sensation of pain: The influence of cognitive processes on pain perception. *Science*, 354(6312), 584-587.
    The key paper linking perceptual decision-making to pain. Establishes pain as Bayesian inference with precision-weighted prediction errors. Demonstrates that expectation shifts DDM starting point (not drift rate). Identifies chronic pain as an active-inference trap with overly precise injury priors. [Sections 3, 4, 5, 9]

12. **Seymour, B., Crook, R. J., & Chen, Z. S. (2023).** Post-injury pain and behaviour: A control theory perspective. *Nature Reviews Neuroscience*.
    The foundational paper for the LQG framework of pain. Post-injury pain is a recuperation controller; chronic pain is control failure via information restriction. Cross-species cephalopod evidence demonstrates the evolutionary conservation of the control strategy. [Sections 2, 7, 8, 9]

13. **Mahajan, P. & Seymour, B. (2025).** Forward and reverse engineering the pain system: From computational neuroscience to neuro-engineering. *PAIN*.
    Closes the theoretical-to-clinical loop. Forward models (Bayesian inference, RL, LQG, MPC) + reverse engineering (HDDM, LDS-EM, VAE) + closed-loop neuromodulation = computational pain medicine. Introduces computational phenotyping and digital twins for personalized treatment. [Section 10]

### Additional Recommended References

14. **Buchel, C., Geuter, S., Sprenger, C., & Eippert, F. (2014).** Placebo analgesia: A predictive coding perspective. *Neuron*, 81(6), 1223-1239.
    Provides experimental evidence for the predictive-coding account of placebo analgesia, showing that expectations modulate pain through precision-weighted prediction errors in the descending pain modulatory system. Directly supports the Bayesian framework of Section 4.

15. **Ratcliff, R. & McKoon, G. (2008).** The diffusion decision model: Theory and data for two-choice decision tasks. *Neural Computation*, 20(4), 873-922.
    The definitive technical reference for the DDM. Provides the mathematical derivations (Wiener first-passage time distributions, speed-accuracy tradeoff equations) and extensive model-fitting methodology. Essential background for Section 5.

16. **Wiecki, T. V., Sofer, I., & Frank, M. J. (2013).** HDDM: Hierarchical Bayesian estimation of the drift-diffusion model in Python. *Frontiers in Neuroinformatics*, 7, 14.
    Introduces the HDDM software package for hierarchical Bayesian DDM fitting. The methodology enables reliable individual-difference parameter estimation even with limited trial counts — the practical tool for computational phenotyping of pain patients (Section 10).

17. **Friston, K. (2010).** The free-energy principle: A unified brain theory? *Nature Reviews Neuroscience*, 11(2), 127-138.
    The original statement of the free-energy principle. All perception and action serve to minimize variational free energy (equivalently, maximize model evidence). Provides the theoretical foundation for the active-inference account of chronic pain (Section 8).

18. **Crombez, G., Van Damme, S., & Eccleston, C. (2005).** Hypervigilance to pain: An experimental and clinical analysis. *Pain*, 116(1-2), 4-7.
    Defines hypervigilance to pain and reviews the experimental evidence: lowered thresholds, attentional capture, interpretive bias, difficulty disengaging. The clinical phenomenology that the computational frameworks of this tutorial aim to explain. [Sections 2, 9]

19. **Van Damme, S., Legrain, V., Vogt, J., & Eccleston, C. (2010).** Keeping pain in mind: A motivational account of attention to pain. *Neuroscience & Biobehavioral Reviews*, 34(2), 204-213.
    Proposes a motivational framework for attention to pain: pain captures attention because it signals threat to current goals. The degree of attentional capture depends on pain intensity, novelty, unpredictability, and threat value — all of which map onto precision and threat variables in the computational framework.

20. **Vlaeyen, J. W. S., Crombez, G., & Linton, S. J. (2016).** The fear-avoidance model of pain. *Pain*, 157(8), 1588-1598.
    The dominant clinical model of chronic pain transition: pain catastrophizing → fear → avoidance → disability → more pain. The computational frameworks in this tutorial formalize this cycle as precision dysregulation → Pavlovian bias → information restriction → stuck inference.

21. **Pezzulo, G., Rigoli, F., & Friston, K. (2015).** Active inference, homeostatic regulation and adaptive behavioural control. *Progress in Neurobiology*, 134, 17-35.
    Extends active inference to homeostatic regulation, showing how interoceptive inference and allostatic control emerge from the same free-energy minimization objective. Directly relevant to the grid-world's homeostatic reward structure and the connection between interoception and pain.

22. **Todd, J., Sharpe, L., Johnson, A., Nicholson Perry, K., Colagiuri, B., & Dear, B. F. (2015).** Towards a new model of attentional biases in the development, maintenance, and management of pain. *Pain*, 156(9), 1589-1600.
    Proposes a multi-stage model of attentional bias in pain: initial orienting, engagement, and disengagement as separate processes with distinct computational signatures. Relevant for designing per-modality precision modulation in the grid-world neuromodulator.

23. **Daw, N. D., Niv, Y., & Dayan, P. (2005).** Uncertainty-based competition between prefrontal and dorsolateral striatal systems for behavioral control. *Nature Neuroscience*, 8(12), 1704-1711.
    The model-free/model-based arbitration framework based on posterior uncertainty. Under threat imminence, model-based control cedes to model-free habits — the computational mechanism for the shift from deliberative to reflexive behavior in hypervigilance.

24. **Green, D. M. & Swets, J. A. (1966).** *Signal detection theory and psychophysics*. New York: Wiley.
    The founding text of SDT. Provides the mathematical basis for the likelihood ratio, criterion, ROC analysis, and sensitivity measures ($d'$) that underpin the entire perceptual decision-making framework.

25. **Wald, A. (1947).** *Sequential analysis*. New York: Wiley.
    Introduces the Sequential Probability Ratio Test (SPRT) — the optimal sequential binary hypothesis test. The SPRT is the mathematical backbone of evidence-accumulation models, with the DDM as its continuous-time Gaussian limit.
