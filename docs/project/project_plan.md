---
title: "EPISODE — Project Direction"
last_updated: 2026-05-26
---

# EPISODE — Project Direction

**EPISODE** — **E**mergence of **P**ain **I**n **S**imulated **O**rganismic & **D**ynamic **E**nvironments.

> **What this document is.** Direction context for the project. It states the conceptual framing, the audiences, the framework, and the two papers the project produces. It is **not** a stage plan, a list of pre-registered hypotheses, a record of current state, or a record of past work. It is a stable reference. Subtasks operate inside this frame; this frame does not change as subtasks land.

## 1. The thesis: pain is more than nociception

Pain is not nociception.

**Nociception** is a peripheral damage signal — fast, mechanical, reflex-driving. A force sensor on a robot, or a pain receptor in skin, is a nociceptor.

**Pain** is the cognitive and perceptual processing that sits on top of that signal — the reweighting of perception, memory, and behavior in response to the damage signal *and to the context the signal arrives in*. Without that processing layer, an organism (or robot) has the alarm bell but not the protective repertoire that goes with it.

Pain science established this distinction decades ago. Animals with congenital absence of nociception still have the perceptual / mnemonic / behavioral correlates of pain abolished — they lose much more than the reflex. Animals with intact nociception but disrupted cognitive processing (chronic pain, placebo analgesia) show the inverse: nociceptive input is unchanged, the protective behavior is not.

The foundational claim of this project is that this distinction is computationally meaningful and reproducible in a closed-loop embodied agent. **Pain-like behavior requires more than a damage signal; it requires a cognitive/perceptual layer that reweights perception, memory, and policy in response to damage history and context.** The project's job is to specify what that layer looks like algorithmically and to demonstrate, in a controlled simulation, that the behavioral signatures of pain emerge when that layer is added on top of nociception, and not before.

## 2. Why this matters — three audiences across two papers

### 2.1 Neuroscience (Paper 1)

Computational pain science has a strong verbal-model tradition — predictive-coding accounts of chronic pain, interoceptive inference, hypervigilance and attentional bias, fear-avoidance, descending opioid modulation, the chronic-pain controller-failure account. These theories specify what pain *is* computationally, but they are rarely instantiated in closed-loop embodied form. They live as block diagrams, equations on a slide, or fits to neural data — not as agents you can deploy.

This project provides a computational testbed in which those theories become *runnable*. The pain-science researcher can write down a precision-weighting account, a fear-avoidance account, a chronic-controller account, and ask: *what does this look like in an agent that has to survive?* Behavioral signatures, perceptual signatures, and the algorithmic primitives that produce them all become measurable on a shared substrate.

### 2.2 Robotics (Paper 1)

Robotic pain is dominated by sensor design — better force sensors, better torque sensors, more compliant skin, finer-grained nociceptive arrays. This work is good and necessary, but it mirrors a stage that pain science left behind in the 20th century: it treats pain *as a sensor problem*.

The pain-science distinction (§1) says this is incomplete. A robot equipped with the best possible nociceptive sensor still lacks the protective behavioral repertoire that comes from cognitive / perceptual processing of that signal — context-sensitive avoidance, graded recovery, drive-conflict management, hypervigilance after damage history. The project's contribution to robotics is to make the missing layer explicit, demonstrate it in a closed-loop agent, and offer a reference architecture for what the cognitive layer looks like on top of any nociceptive sensor stack.

The argument to robotics is not "your sensors are wrong" — it is "your sensors are necessary but not sufficient, and here is what sufficiency looks like".

### 2.3 Machine learning (Paper 2)

In biology, neuromodulation is increasingly understood as *parameter modulation*: ascending modulators (acetylcholine, noradrenaline, dopamine, serotonin, opioid) emit slow context-conditioned signals that reshape how downstream populations process input, retain memory, and select actions.

In machine learning, three subfields use parameter-modulation primitives but rarely speak to one another:

- **Perceptual modulation** — FiLM, AdaIN, conditional batch / layer norm, attention-as-modulation, hypernetworks. A context signal reshapes how the network processes input.
- **Hyperparameter modulation** — Doya-2002–style: the modulator outputs *the learning algorithm's own hyperparameters* (temperature, learning rate, time discount, exploration bonus) as a learnable function of context.
- **Continual learning** — context-conditioned gating and modulator architectures (HyperNet, PathNet-family, modulator-gated experts) used to prevent catastrophic forgetting across task / distribution shifts.

A neuromodulation-inspired modulator architecture is a single algorithmic primitive that touches all three. The contribution of Paper 2 is to make this connection explicit — formally framing the three as instances of one conditional-architecture class, and demonstrating that a shared modulator architecture serves all three.

## 3. The framework: four behavioral categories × three levels of analysis

The project's work is organised in a 4 × 3 matrix. Any subtask is locatable in it.

### 3.1 Four behavioral categories of pain-like behavior

Each category is a distinct empirical signature that exceeds what a pure-nociception agent (a reflex-only damage-avoider with no cognitive layer) produces.

| Category | Plain-English definition | What makes it pain-like (vs. pure-nociception) |
|---|---|---|
| **Avoidance** | The agent moves away from damage sources and damage-associated contexts. | Generalises from acute damage events to *context-conditioned predictions* of damage — not reflex withdrawal at the moment of contact. |
| **Recovery** | After damage, the agent's behavior and perception return toward baseline along a graded time course. | The recovery trajectory is context-modulated and structured by the cognitive layer — not a fixed sensor reset. |
| **Managing conflict needs** | When protective avoidance conflicts with other drives (hunger, exploration, task completion), the agent resolves the trade-off in a context-sensitive way. | A pure-nociception agent cannot weigh competing drives against damage history; cognitive modulation is what allows the trade-off to be made coherently. |
| **Hypervigilance** | After damage, perceptual and behavioral sensitivity to threat-related signals changes — selectively elevated and persistent. | Channel-selective, context-modulated, outlasting the immediate input — not a uniform fear response. |

The four categories together constitute the case for "pain ≠ nociception". Each individually is a familiar phenomenon; the *joint* demonstration that all four follow from a single cognitive/perceptual layer added on top of nociception is the contribution.

### 3.2 Three levels of analysis

Each category is studied at three levels. The same agent, the same training run, can be read at any of the three.

| Level | What you measure | What it answers |
|---|---|---|
| **Behavioral** | Action distributions, survival, time-to-recover, foraging-vs-avoidance switching, threat-channel-selective response, trajectory shape around damage events. | *Does the agent act pain-like by behavioral criteria?* |
| **Perceptual** | Internal representation changes — channel-selective gain, gate activity, attention to threat features, modulator state trajectories, latent representations of damage history. | *Does the agent's network process perceptual information differently after damage, in the way pain theory predicts?* |
| **Algorithmic** | What class of algorithm the modulator implements — its place in the FiLM / hypernetwork family, its connection to Doya-style hyperparameter modulation, its connection to continual-learning modulator gating. | *What is the modulator, computationally, and what other problems does this class of algorithm address?* |

The behavioral level is what a non-mechanistic reviewer cares about. The perceptual level is the mechanism. The algorithmic level is what generalises out of this project to the rest of machine learning.

## 4. Paper 1 — Nature Machine Intelligence

**Headline claim.** Pain is more than nociception, and a closed-loop embodied agent exhibits pain-like behavior across all four categories when a neuromodulation-inspired cognitive/perceptual layer is added on top of the nociceptive signal — and does not exhibit those signatures when the cognitive layer is absent.

**Audiences.** Computational neuroscience (pain theorists, predictive-coding researchers, interoceptive-inference researchers) and embodied / cognitive robotics (researchers building protective behavior for physical systems).

**Calibration — perspective-plus-pilot.** Paper 1 does not claim to have exhaustively solved any of the four categories. It demonstrates the framework: each category receives a clean behavioral-level demonstration; at least one or two carry clean perceptual-level mechanism. The contribution is to make the framework concrete, runnable, and credible enough that the field is moved to pursue it further — not to deliver saturation-level treatment of any single category.

**Algorithmic commitment — minimal and shared.** Paper 1 uses one shared neuromodulation-inspired modulator architecture, focused on the *perceptual modulation* injection site (the FiLM-variant family), applied uniformly across all four categories. Hyperparameter modulation and continual-learning modulator gating are deliberately not used in Paper 1; those belong to Paper 2.

This minimality is a feature, not a limitation. The argument is stronger when one modulator architecture produces all four behavioral signatures than when each category requires bespoke architecture.

## 5. Paper 2 — NeurIPS

**Headline claim.** Neuromodulation-inspired modulator architectures — FiLM variants understood as conditional-architecture / hypernetwork objects — are a single algorithmic primitive that unifies three subfields of machine learning currently treated separately: perceptual modulation, hyperparameter modulation, and continual learning.

**Audience.** Machine learning (representation learning, meta-reinforcement learning, continual learning, conditional / modular architectures).

**Calibration — perspective-plus-pilot.** Paper 2 does not claim to beat specialised state-of-the-art in any of the three subfields. The unifying claim is carried by the formal framing (modulation as a hypernetwork-class object), supported by clean pilot demonstrations of the same modulator architecture serving each of the three roles, with at least one of the three carrying a tighter empirical comparison to anchor the claim.

**The three subfields and what the modulator does in each.**

- *Perceptual modulation.* The same modulator architecture as in Paper 1, in its perceptual-injection role: shaping how the encoder processes input as a function of internal state and context.
- *Hyperparameter modulation.* In the Doya 2002 lineage: the modulator emits the learning algorithm's own hyperparameters (policy temperature, exploration coefficient, time discount, learning rate-like quantities) as learnable, context-conditioned outputs.
- *Continual learning.* The modulator as a context-conditioned gating mechanism: distinct context regimes activate distinct modulator states, which preserve performance on prior tasks while allowing new tasks to be learned.

**Connection to Paper 1.** The modulator architecture is shared between the two papers. Paper 1's biological grounding gives Paper 2's "neuromodulation-inspired" framing its meaning; Paper 2's algorithmic-class generalisation gives Paper 1's reference architecture its broader significance. Neither paper is parasitic on the other — each makes a complete contribution to its own audience — but they reinforce one another.

## 6. Out of scope

The following are deliberately not addressed by this project. Reviewer questions about them are answered by acknowledging them as scope boundaries, not by overreaching the claims.

- **Subjective pain / qualia.** The project makes no claim that the agent feels pain, or that pain-like behavior implies pain experience. The contribution is computational and behavioral.
- **Clinical translation.** No claim that the modulator architecture corresponds to a specific clinical syndrome (chronic pain, fibromyalgia, neuropathic pain, CRPS) at a diagnostic or therapeutic level.
- **Exhaustive cracking of any one category.** Paper 1 is perspective-plus-pilot; a saturation-level deep dive into hypervigilance, or into recovery dynamics, or into any single category, is a follow-up project, not part of this program.
- **Exhaustive cracking of any one ML subfield.** Paper 2's unifying claim is at the framing-and-pilot level. Beating specialised state-of-the-art on any one of perceptual modulation, hyperparameter modulation, or continual learning is out of scope.
- **Affective-vs-sensory dissociation in pain.** The classical pain-science distinction between sensory-discriminative and affective-motivational dimensions of pain is not operationalised here.
- **Real-robot deployment.** The reference architecture is demonstrated in simulation. Sim-to-real transfer of the cognitive layer is a separate engineering project.

## 7. How this document is used

This document is a *direction context*. It states the conceptual framing, the audiences, the framework, and the two papers. It does not specify:

- Phases, gates, or exit criteria.
- Run budgets, node allocations, or seed counts.
- Architectural details (specific FiLM variants, injection sites, training losses, modulator hidden sizes).
- The current state of the project, or any record of past work.
- Hypotheses pre-registered for any specific experiment.

Those concrete artefacts are produced and maintained in subtask documents:

- `docs/develop/` — implementation plans, architectural specifications, technical depth.
- `docs/experiments/` — experimental designs, training-result analyses.
- `docs/project/` (beyond this file) — concept memos and research-direction memos from the project's researcher agents.
- `docs/pi/` — portfolio-level focus-vs-explore calls and strategic logs.
- `docs/memory/` and `docs/diary/` — session-level memory and event log.

Each subtask document should be locatable in the 4 × 3 matrix of §3 and should declare which paper (§4 or §5) it contributes to. When any Claude Code agent is consulted on a subtask in this project, this document is the frame the subtask operates inside.

The frame is stable. The work is not.
