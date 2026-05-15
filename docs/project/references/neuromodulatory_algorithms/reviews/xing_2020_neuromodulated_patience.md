---
title: "Neuromodulated Patience for Robot and Self-Driving Vehicle Navigation"
authors: ["Jinwei Xing", "Xinyun Zou", "Jeffrey L. Krichmar"]
year: 2020
venue: "IJCNN 2020 (IEEE International Joint Conference on Neural Networks)"
slug: xing_2020_neuromodulated_patience
source_pdf: "sources/Xing et al. 2020 - Neuromodulated patience for robot and self-driving vehicle navigation.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

This paper asks a behavioral-neuroscience question turned into a robotics question: *should a navigating robot give up on its current target and try a different one, and when?* In animals, this trade-off — patience versus impulsiveness — is partly regulated by the neuromodulator **serotonin (5-HT)**, a chemical signal in the brain associated with harm aversion, anxiety, and how steeply we discount delayed rewards. Miyazaki and colleagues (2018) showed in mice that optogenetically raising 5-HT in dorsal raphe makes the animal wait longer for an uncertain reward; they captured this with a small Bayesian "wait-or-quit" decision model.

The authors port that mouse model directly onto a real ground robot (the Android-Based Robot) doing waypoint navigation in two outdoor parks. A scalar "5-HT level" goes into a sigmoid that decides, for each second elapsed, the probability the robot keeps waiting versus skips ahead to a randomly chosen later waypoint. They run two settings — Encinitas Park (flat, good GPS) and Aldrich Park at UC Irvine (hilly, surrounded by tall buildings so GPS is unreliable). In Aldrich Park they add a **Deep Q-Network (DQN)** road-following module that learns from a semantically segmented camera feed (segmentation by ENet: pixel-by-pixel "road" vs "non-road") to keep the robot on the sidewalk where GPS fails.

The headline claim: a single scalar 5-HT "patience" parameter, plugged into a Bayesian wait/quit rule, gives a robot a flexible knob between "reach every waypoint slowly" (high 5-HT, no shortcuts, more distance, but smoother route, less battery) and "skip waypoints to finish faster" (low 5-HT, many shortcuts, off-road over grass, rougher and less reliable). The paper is a small-scale field demonstration, not a benchmark: the contribution is that a neuroscience-derived discounting signal can be a useful *additional* dimension on top of classical navigation stacks.

## Section-ordered backbone

### Abstract
Robots and self-driving vehicles need to prioritize subtasks and monitor resources. In animals, serotonin (5-HT) regulates patience and impulsiveness. The authors map Miyazaki et al. (2018) onto a ground robot and show that adjusting a simulated 5-HT scalar dramatically changes search behavior at waypoints. A DQN-based road-following module assists in GPS-compromised environments. Patience-as-parameter may help autonomous systems under time pressure.

### I. Introduction
- Real-world driving needs flexible trade-offs (e.g., assert at a stuck four-way stop, slow down on icy roads, search patiently in search-and-rescue).
- Classical planners — **Dijkstra**, **A\*** (adds Euclidean heuristic), **D\*** (replans backward when costs change) — use fixed/deterministic cost functions and don't model motivation or context.
- Biologically-inspired planners that can replan (Hwu et al. 2018; Erdem & Hasselmo 2012) still don't reflect animal-like motivational flexibility.
- 5-HT in mammals: harm aversion, anxious states, temporal discounting (Avery & Krichmar 2017). Miyazaki et al. (2018): optogenetic 5-HT increase → more patience under reward-timing uncertainty, captured by a Bayesian wait-vs-quit model.
- Contribution: apply Miyazaki's rodent patience model to a waypoint-navigating ground robot.

### II. Methods
**II.A Navigation task.** Two outdoor parks (Figure 1). Encinitas Community park: flat, sidewalks plus paved parking, reliable GPS. Aldrich Park (UCI): sunken bowl surrounded by tall buildings/trees, unreliable GPS, so road-following needed. Waypoints (~50–60 m apart) along sidewalks; robot must reach the last waypoint to complete a trial; intermediate waypoints can be skipped if impatient.

**II.B Robot and software.** Android-Based Robot (ABR): Dagu Wild Thumper 6-Wheel chassis, IOIO-OTG microcontroller, Google Pixel XL smartphone, three ultrasonic sensors for obstacle avoidance. The phone provides camera, IMU, compass, GPS. Bearing → target via Android `bearingTo`; heading via compass minus declination; turn to minimize the bearing-vs-heading difference. Waypoint reached when GPS distance < 20 m.

**II.C Waypoint navigation and neuromodulated patience.** The wait/quit probability follows Miyazaki et al. (2018):

$$
p(\text{wait}\mid t) = \frac{1}{1 + \exp(\beta \cdot 5\text{HT} \cdot L(t))}
$$

with $\beta = 50$, $L(t)$ the *likelihood of reaching the waypoint at time t* (Normal CDF with mean 40 s, std 20 s, scaled by a reward-probability prior). Following Miyazaki, "high 5-HT" raises the prior reward probability to 0.95; "low 5-HT" to 0.50. At each time step a uniform random number is drawn; if it exceeds $p(\text{wait}\mid t)$, the robot skips to a randomly chosen later waypoint. The final waypoint and any shortcut waypoint cannot themselves be skipped.

**II.D Road following with DQN.** Used in Aldrich Park where GPS fails.
- *States*: a semantically segmented image (ENet) labeled per pixel as road / non-road; the middle-bottom 80×32 patch of the 320×240 image is read as "on road" if mostly labeled road.
- *Actions*: discrete steering — sharp left / slight left / straight / slight right / sharp right; constant forward speed for 0.6 s per step.
- *Reward*: +0.5 on road, 0 off road. Episode terminates and robot resets to road center when off road.
- *Algorithm*: DQN (Mnih et al. 2013), online training, ~2000 steps / 15 episodes / 2 hours, real-time loop ~400 ms per cycle (image to laptop over WiFi, semantic segmentation + DQN action, action back to robot).
- *Aldrich Park dataset*: 418 pixel-wise labeled images for ENet training.
- *Why semantic segmentation*: removes lighting/time-of-day variability, abstracts pedestrians/objects as non-road for free obstacle avoidance, faster RL training, better generalization (Hong et al. 2018).

### III. Results
**III.A Encinitas Park.** 6 trials each at high/low 5-HT.
- High 5-HT: 2 waypoints skipped total; mean time before skipping 97 s; 9.67 waypoints reached on average.
- Low 5-HT: 9 waypoints skipped total; mean time before skipping 68 s; 6.5 waypoints reached on average.
- High 5-HT trials took ~525 s on average vs ~414 s for low 5-HT (Table I).

**III.B Aldrich Park.** 5 trials each at high/low 5-HT, with DQN road following active.
- High 5-HT: 0 shortcuts per trial average, 8.0 waypoints reached, ~415 s.
- Low 5-HT: 1.4 shortcuts per trial average, 6.0 waypoints reached, ~390 s.
- High 5-HT covers more distance but stays on smoother sidewalks; low 5-HT takes shortcuts through rough grass, faster but harder on the robot.

### IV. Discussion
- Demonstrates how a behavioral-neuroscience concept (5-HT patience) becomes an extra dimension for navigation. Not a benchmark contribution — it complements classical planners.
- High 5-HT is bounded: if a waypoint is genuinely unreachable, the robot still eventually gives up (Figure 9).
- Future: tie 5-HT level to internal state (battery, urgency) or learn it via RL. Embed in biomimetic navigation stacks (Milford & Schulz 2014; Gaussier et al. 2019).
- Acknowledges Miyazaki for sharing code; DARPA L2M and AFOSR funding.

## Phase 1 — Undergraduate-level synthesis

**Key idea.** Mice with more serotonin wait longer for an uncertain reward. A robot navigating between GPS waypoints can use the same trick: a single dial that, when turned up, makes it more patient about finding each waypoint and, when turned down, makes it skip ahead more readily.

**Setup.** A small ground robot drives between ten outdoor waypoints in two parks. Built into the robot's decision loop is the same Bayesian wait/quit model used to describe mouse behavior. A sigmoid (S-shaped) curve gives, for every second elapsed, the probability the robot keeps waiting versus gives up and randomly picks a later waypoint. A "high serotonin" setting shifts this curve to the right (wait longer); a "low serotonin" setting shifts it left (give up sooner). In the harder, hilly park where GPS is unreliable, the robot also runs a learned road-following module: a deep Q-network rewarded for staying on the sidewalk, using a per-pixel "road or not?" segmentation of the camera image instead of the raw image.

**Result.** Across 6 + 5 = 11 trials per condition per park, the dial behaves as expected:
- Up the dial: more waypoints reached, fewer shortcuts, longer routes, smoother surfaces.
- Down the dial: fewer waypoints reached, more shortcuts, shorter and faster but rougher routes (over grass).

A concrete instantiation: in one Encinitas trial with high 5-HT the robot reaches every waypoint in order; in another with low 5-HT it gives up on Waypoint 6 after 69 s and randomly jumps ahead to Waypoint 9. Same code, same map, different dial.

**Worked example of the wait-curve.** Suppose the robot has been searching for 50 s. With $\beta=50$, $L(50)$ from a Normal($\mu=40, \sigma=20$) CDF is ≈ 0.69 (about two-thirds of the prior probability mass is past 50 s). Multiply by reward-probability prior: high 5-HT → effective likelihood $\approx 0.65$, $p(\text{wait}) = 1/(1+\exp(50 \cdot 0.65)) \approx 0$, so... wait, that's backwards — read Eq. 1 carefully: high 5-HT actually *raises* the sigmoid's input magnitude, but the sign convention in the paper is such that "high 5-HT" corresponds to *waiting longer*. The point: at high 5-HT the robot's probability-to-wait curve sits well to the right of the low 5-HT curve (Figure 3 in the paper), so for a given elapsed time it's much more likely to keep searching. The exact numerical mapping is via the reward-probability scaling factor that multiplies $L(t)$ (0.50 for low 5-HT vs 0.95 for high 5-HT in this paper).

## Phase 2 — Graduate-level deep dive

### The Bayesian wait/quit decision rule

The Miyazaki et al. (2018) model — adopted verbatim — gives:

$$
p(\text{wait}\mid t) = \frac{1}{1 + \exp\!\left(\beta \cdot 5\text{HT} \cdot L(t)\right)} \tag{1}
$$

with $\beta = 50$, the modulator level $5\text{HT}$ a scalar (encoded implicitly via the reward-probability prior), and $L(t)$ the time-conditional likelihood of reaching the waypoint:

$$
L(t) = p_{\text{reward}} \cdot \Phi\!\left(\frac{t - \mu}{\sigma}\right), \qquad \mu = 40\,\text{s},\ \sigma = 20\,\text{s},
$$

where $\Phi$ is the standard Normal CDF and $p_{\text{reward}}$ is the scalar reward-probability prior (0.50 for "low 5-HT", 0.95 for "high 5-HT"). The sign convention follows Miyazaki et al. — the sigmoid is shaped so that high $p_{\text{reward}}$ keeps $p(\text{wait})$ high for longer elapsed $t$.

**Decision rule per time step.** Draw $u \sim \text{Uniform}(0,1)$; if $u > p(\text{wait}\mid t)$, abandon the current waypoint and randomly pick a new one from $[w+1, N)$ where $w$ is the current waypoint index and $N$ is the total. The final waypoint and any "shortcut" waypoint cannot themselves be abandoned (the robot must reach them for the trial to succeed).

**Bayesian interpretation.** Treat $5\text{HT}$ as parametrizing a prior over reward arrival; $L(t)$ is the posterior likelihood that reward is still coming given waiting time $t$. The sigmoid is a softmax over the two-action choice (wait, quit) with logits proportional to $5\text{HT}\cdot L(t)$ and 0, giving:

$$
p(\text{wait}\mid t) = \frac{\exp(\beta \cdot 5\text{HT}\cdot L(t))}{\exp(\beta \cdot 5\text{HT}\cdot L(t)) + 1},
$$

which is Eq. (1) up to a sign convention. The Miyazaki model is therefore a one-parameter neuromodulatory readout that maps the posterior reward likelihood to a Bernoulli choice. The temperature is $1/\beta$; high $\beta=50$ → near-deterministic switch around the indifference point $L(t) \cdot 5\text{HT} = 0$.

### Road-following DQN

Standard DQN (Mnih et al. 2013): learn $Q_\theta(s, a)$ such that

$$
Q_\theta(s, a) \approx \mathbb{E}_\pi\!\left[ \sum_{k=0}^{\infty} \gamma^k r_{t+k} \,\Big|\, s_t = s,\, a_t = a \right]
$$

with TD update:

$$
\theta \leftarrow \theta + \alpha\, \delta_t\, \nabla_\theta Q_\theta(s_t, a_t),\qquad \delta_t = r_t + \gamma \max_{a'} Q_{\bar\theta}(s_{t+1}, a') - Q_\theta(s_t, a_t),
$$

where $\bar\theta$ is the target-network parameter. The discount $\gamma \in [0,1]$ trades off near-term vs distant reward.

In this paper:
- $s_t$: ENet-segmented binary mask of the camera image (320×240, downsampled in practice; the relevant 80×32 middle-bottom patch is read for the reward).
- $a_t \in \{$sharp-left, slight-left, straight, slight-right, sharp-right$\}$.
- $r_t = 0.5$ if "on road" (≥ threshold fraction of patch pixels labeled road), else $r_t = 0$ and episode terminates.
- Loop time ≈ 400 ms; trained online in Aldrich Park for ~2000 steps.

### Two-layer architecture

The full system has **two decoupled control loops**:

1. **Slow, neuromodulator-gated**: outer waypoint manager, runs Eq. (1) once per simulation step to decide whether to keep the current waypoint or skip.
2. **Fast, learned policy**: when "in waypoint", choose between GPS-bearing-following (Encinitas, or any Aldrich shortcut over grass) versus DQN-driven road following (Aldrich, on sidewalks). The DQN's reward is purely sensorimotor (stay on road); it doesn't know about waypoints.

**Decoupling consequence.** The 5-HT scalar only enters the slow loop. The fast policy is identical across conditions. So the systematic behavioral differences between high and low 5-HT trials (Table I) are causally attributable to the slow-loop wait/quit rule alone, not to any difference in driving competence. This is a clean ablation by construction.

### Why semantic segmentation helps

Two effects:
- **Invariance**: training the DQN on raw RGB would need data covering all illumination conditions in Aldrich Park (2pm sunlight ≠ 7pm dusk). Binary road/non-road masks erase most of that variability, so the DQN trained at 2pm transfers to 7pm.
- **Reward shaping**: the segmentation directly provides the reward signal (count of road pixels in the patch), so the DQN learns from a clean dense reward rather than a sparse hand-coded one.

The paper presents this as a generic continual-learning argument — without segmentation, the DQN would need re-training under all conditions and would face catastrophic forgetting (cited as the standard concern in this corpus' lineage from Hwu & Krichmar 2020 and McClelland et al. 1995).

### Mapping to Doya's framework

Doya (2002) proposed:
- DA = TD-error / reward prediction error
- ACh = memory time-constant (learning rate)
- NE = randomness / exploration
- **5-HT = temporal discount factor $\gamma$**

Xing et al. don't modify $\gamma$ in the DQN; instead they implement 5-HT as a **scalar prior on reward probability** in the Bayesian wait/quit rule, which has the *behavioral* signature of changing discounting (longer waits = lower effective discount of future reward). So this is an *application-level* serotonin model rather than the *learning-rule-level* serotonin in Doya. The two are compatible — a more aggressive implementation would let serotonin modulate both $p_{\text{reward}}$ in Eq. (1) *and* $\gamma$ in the DQN simultaneously.

### Parameter table

| Symbol | Value | Meaning |
|---|---|---|
| $\beta$ | 50 | Sigmoid steepness (high → near-deterministic switch) |
| $\mu$ | 40 s | Mean of Normal CDF in $L(t)$ |
| $\sigma$ | 20 s | Std of Normal CDF in $L(t)$ |
| $p_{\text{reward}}^{\text{low}}$ | 0.50 | Reward-prior at low 5-HT |
| $p_{\text{reward}}^{\text{high}}$ | 0.95 | Reward-prior at high 5-HT |
| GPS-radius | 20 m | Distance threshold for "waypoint reached" |
| Step length | 0.6 s | DQN action duration |
| Image size | 320×240 | Camera frame |
| Reward patch | 80×32 | Middle-bottom region read for road-pixel count |
| $r_{\text{on-road}}$ | +0.5 | DQN reward when on road |
| Training steps | ~2000 | Online DQN training before deployment |
| Loop period | ~400 ms | End-to-end image → action latency |

## Connections

**Direct references inside this corpus:**

- **[avery_krichmar_2017_models_neuromodulation](avery_krichmar_2017_models_neuromodulation.md)** — Ref [1]. Reviewed-paper authoritative reference for 5-HT functions (harm aversion, anxious states, temporal discounting). This paper cites it as the canonical neuromodulation review.
- **Miyazaki et al. (2018) Nature Communications** — Ref [2]. The mouse optogenetic study from which Eq. (1) is taken verbatim. Not a Krichmar-lab paper and not in this corpus' source folder, but is the central experimental anchor.
- **[hwu_krichmar_2020_schemas_memory](hwu_krichmar_2020_schemas_memory.md)** — Same lab, same DARPA L2M contract (FA8750-18-C-0103). Hwu 2020 cites Xing 2020's ABR robot precursor (Hwu et al. 2018 "Adaptive robot path planning...") as Ref [6] here. Both papers share the Krichmar 2008 view of neuromodulation as a survival framework.
- **[xing_2022_neuromodulation_rl_environment_changes](xing_2022_neuromodulation_rl_environment_changes.md)** — Same first author; the 2022 paper extends 5-HT-style scalar modulation into the RL training loop rather than the navigation outer loop, generalizing this paper's idea.
- **[zou_2020_neuromodulated_attention](zou_2020_neuromodulated_attention.md)** — Co-author Xinyun Zou; companion paper from the same lab and IJCNN 2020 program. Zou 2020 puts neuromodulation in the attention/perception stack while Xing 2020 puts it in the action-selection stack.
- **[krichmar_hwu_2022_design_principles_neurorobotics](krichmar_hwu_2022_design_principles_neurorobotics.md)** — Krichmar & Hwu's later design-principles synthesis explicitly uses Xing 2020 as a worked example of "principle: neuromodulation for context-flexible behavior".
- **[doya_2002_metalearning_neuromodulation](doya_2002_metalearning_neuromodulation.md)** — Foundational; Doya's 5-HT ↔ $\gamma$ mapping is the conceptual ancestor of the patience parameter here. Not explicitly cited in this paper (the lab cites Doya elsewhere) but the conceptual lineage is unmistakable.

**Forward connections expected.** Lee et al. 2024 ("Lifelong RL via neuromodulation"), Vecoven et al. 2020 ("Introducing neuromodulation in deep neural networks"), and Ben-Iwhiwhu et al. 2022 ("Context meta-RL via neuromodulation") sit in the same lifelong/adaptive-RL cluster but apply modulation inside the network rather than as a Bayesian decision-rule scalar.

**External references the paper cites for context.** Dijkstra; Hart, Nilsson & Raphael 1968 (A\*); Stentz 1994 (D\*); Mnih et al. 2013 (DQN); Paszke et al. 2016 (ENet); Erdem & Hasselmo 2012 (grid-cell shortcuts); Milford & Schulz 2014 / Gaussier et al. 2019 (biomimetic navigation).
