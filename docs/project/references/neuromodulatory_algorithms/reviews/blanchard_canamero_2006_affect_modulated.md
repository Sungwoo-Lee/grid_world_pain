---
title: "Developing Affect-Modulated Behaviors: Stability, Exploration, Exploitation or Imitation?"
authors: ["Arnaud J. Blanchard", "Lola Cañamero"]
year: 2006
venue: "Adaptive Systems Research Group, University of Hertfordshire (workshop paper; cited in subsequent Cañamero-school literature as Blanchard & Cañamero 2006)"
slug: blanchard_canamero_2006_affect_modulated
source_pdf: "sources/Blanchard and Cañamero 2006 - Developing affect-modulated behaviors - Stability, exploration, exploitation or imitation.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

This paper is a clean, small-scale **real-robot demonstration** that a *single* control architecture — built around two scalar quantities the authors call **well-being** and **affect** — can autonomously switch between four behavioral modes: **seeking stability** (staying near a safe / familiar zone), **exploration** (cautiously moving into novelty), **exploitation** (opportunistically continuing a movement that just paid off), and **low-level imitation** (mirroring a caretaker's motion when the situation feels safe). The robot is a Koala wheeled robot whose only useful sensor is an infrared distance reading to a "caretaker" in front of it. Stimulation of its side sensors counts as a reward and raises an internal scalar called *well-being* $W_b$; the robot also tracks a target distance to the caretaker (the *desired sensation* $\overline{S_d}$) using an averaging rule with a learning rate that itself depends on $W_b$. The clever part is the affect scalar $A_f$, which measures how close the current sensation is to the desired sensation in **sensory space**. A high $A_f$ value (the robot feels safe) flips the sign of the *motivation to continue* $M_c$ so the robot **amplifies** any perceived movement (curious exploration or imitation); a low $A_f$ flips the sign back so the robot **opposes** any perceived movement (stability seeking, withdrawal). On top of this, the variation of well-being over different *time scales* gives a **pleasure** signal $P_l$ that interrupts the current mode when something good or bad suddenly happens. The architecture is shown to produce the cautious-approach gait of a small animal exploring a new object, opportunistic reward-following, and low-level imitation of a moving caretaker — all without explicit mode-switch logic. The paper is important to the Cañamero corpus because it demonstrates that the *affect-as-modulator-of-sign-of-motivation* idea can replace discrete behavior-selection switches.

## Section-ordered backbone

### Abstract
Exploring the environment is essential for autonomous agents to learn and consolidate experience, but it is risky. A trade-off must therefore exist between seeking stability, exploration, imitation of other agents' novel actions, and taking advantage of opportunities. The paper presents a Perception–Action robotic architecture that achieves this trade-off through modulatory mechanisms based on *well-being* and *affect*. Implemented and tested on a Koala robot.

### Section 1 — Introduction
Motivates the gap left by the authors' earlier imprinting-based architecture (Blanchard & Cañamero 2005a), in which the robot stayed safely near a caretaker but could discover nothing new. Argues that endowing a robot with autonomous exploration poses three problems: (1) generating all four behavior types from one architecture, (2) autonomously switching among them, (3) achieving a good balance. Frames the work inside the Perception–Action (PerAc) tradition (Prinz 1997; Gaussier & Zrehen 1995), where perception and action are tightly coupled and action is a side effect of trying to achieve or correct a perception (a homeostatic-control view per Ashby 1952). Distinguishes *sensation* (raw sensor input) from *perception* (interpretation including associated actions/affordances). Differentiates from classical behavior-selection literature: switches are not over discrete behaviors but emerge from continuous modulation by well-being and affect, defined as:

- **Well-being** $W_b \in [0,1]$: viability of the internal state, i.e. distance of internal variables from their ideal values — Dunn's *endogenous* component of comfort.
- **Affect** $A_f \in [0,1]$: evaluation of safety/goodness of a situation based on the familiarity (frequency) and past well-being (pleasantness) of the associated sensation — Dunn's *exogenous* component.

### Section 2 — Seeking Stability
Reproduces the imprinting + adaptation mechanism from Blanchard & Cañamero 2005a. The Koala robot is placed in front of a caretaker. The only relevant sensation is distance to the caretaker $S_d$ (from infrared sensors). Stimulation of the lateral IR sensors raises $W_b$ (which otherwise decays). The robot learns the *desired sensation* $\overline{S_d}$ as the running average of past sensations, weighted by well-being:

$$
\overline{S_d}(t) = \overline{S_d}(t-1) + \eta(t) \cdot \big( S_d(t) - \overline{S_d}(t-1) \big), \qquad \eta(t) = \tfrac{1}{t} \ \text{(plain average)},
$$

then upgraded to a well-being-weighted rate $\eta(t) = W_b(t) / \widetilde{W_b}(t)$, with $\widetilde{W_b}$ the cumulative well-being. To avoid the saturation problem (as $\widetilde{W_b}$ grows, learning halts), the authors define **multiple time scales** $k$ with rates $\eta_k(t) = \big( W_b(t)/\widetilde{W_b}(t) \big)^k$, $k \in \{0.2, 0.4, \ldots, 2.0\}$. Each time scale $k$ has its own desired sensation $\overline{S_d}_k$. The mismatch $\Delta d_k(t) = S_d(t) - \overline{S_d}_k(t)$ defines a *perceived action* $P_{a,k}(t) = \epsilon^k \cdot \Delta d_k(t)$ — the action that *would* have produced the mismatch, with $\epsilon^k$ scaling action-magnitude with the time scale. With $M_{c,k} = -1$ (motivation to oppose), the executed action becomes

$$
A_{c,k}(t) = P_{a,k}(t) \cdot M_{c,k}(t),
$$

producing a robot that drives toward $\overline{S_d}_k$ from any side — pure stability-seeking.

### Section 3 — Exploration
The stability-seeking architecture stays static in absence of caretaker motion. To break the symmetry the authors introduce an *apathy* variable $A_p$ that is high when $M_c$ is small (no strong motivation either way), and use it to gate an innate forward-motion drive $B_e$:

$$
A_e(t) = \big( A_e(t-1) + B_e(t) \big) \cdot A_p(t), \qquad A_p(t) = e^{-r \cdot M_c(t)^2},
$$

with $r$ a decay-rate parameter. Then the key affect formulation: a low *perceived action* magnitude means the robot is close to its desired sensation, hence safe, hence affect is high. They define

$$
A_{f,k}(t) = e^{-s \cdot P_{a,k}(t)^2},
$$

with $s$ a decay rate. The *openness to the world* $O_{p,k}(t) = A_{f,k}(t) - q$, with $q \in [0,1]$ a *timorousness* parameter, is positive when the robot feels safe enough to amplify perceived actions and negative when it should oppose them. They equate $M_{c,k} = O_{p,k}$, so when the robot is in a familiar zone $M_c > 0$ and small forward motions are *amplified* (curious exploration); when it drifts into unfamiliar territory $A_f$ falls, $M_c$ flips negative, and the robot retreats. The emergent behavior is **cautious approach** — move forward, stop, wait, move forward, retreat, etc. — for which the paper shows trajectory plots at $q = 0, 0.5, 1.0$ (lower $q$ → more confident; higher $q$ → more hesitant).

### Section 4 — Exploitation and Interruption-Related Behaviors
To respond to sudden well-being changes (e.g., a side-touch reward suddenly appears), the architecture is extended with a *pleasure* signal. The averaged well-being at each time scale $k$ is

$$
\overline{W_b}_k(t) = \overline{W_b}_k(t-1) + \eta_k \cdot \big( W_b(t) - \overline{W_b}_k(t-1) \big),
$$

and pleasure (the deviation of current $W_b$ from its time-scale-$k$ baseline) is

$$
P_{l,k}(t) = W_b(t) - \overline{W_b}_k(t), \qquad P_{l,k} \in [-1, 1].
$$

The motivation to continue is then

$$
M_{c,k}(t) = O_{p,k}(t) + P_{l,k}(t).
$$

Positive $P_{l,k}$ (sudden reward) amplifies any ongoing perceived action, producing *opportunism*; negative $P_{l,k}$ (sudden danger) cancels or reverses it, producing *avoidance*. Three experiments with reward boxes verify: (exp 1) no reward → cautious exploration as in §3; (exp 2) reward at robot's front → robot accelerates toward it; (exp 3) reward already next to robot → exploration would cost the reward, so the robot stays.

### Section 5 — Low-level Imitation
With no extra machinery, when a caretaker moves in front of the robot, the unexpected sensation drops $A_f$ if the caretaker moves far, but if the robot is already in its familiar zone and well-being is steady or rising, $M_c$ is positive, so any **perceived caretaker-induced motion** is *amplified*: the robot moves *toward* a caretaker that approaches and *away from* one that recedes, i.e. low-level imitation. If the caretaker moves into the avoidance zone, the behavior flips sign and the robot avoids. The paper contrasts this *amplification* view of imitation with prior work that treated imitation as error-reduction (Andry et al. 2003; Demiris & Johnson 2003) — here, imitation is the **process of amplifying an unexpected sensation**, controlled by affect.

### Section 6 — Conclusion and Perspectives
Recap: a single PerAc architecture, modulated by well-being and affect, autonomously produces stability-seeking, exploration, exploitation, avoidance, and low-level imitation. Future directions: richer perceptual feature space beyond distance-only sensors; deeper investigation of biological plausibility, in particular similarity to dopaminergic circuits in behavior selection.

## Phase 1 — undergraduate-level synthesis

**The plain idea.** Don't build a robot with a list of behaviors (one for explore, one for retreat, one for follow) and a brain that switches between them. Instead build a robot with *two emotional dials*:
- *well-being* — am I in good internal shape right now?
- *affect* — does my current sensation match what I've learned is "the safe sensation"?

Both dials are continuous numbers between 0 and 1. The robot constantly computes a third signal, the *motivation to continue*, which says: should I *amplify* or *oppose* the motion that would explain my current sensation? When affect is high (safe), motivation-to-continue is positive, so the robot *adds* to whatever motion fits the perceptual mismatch — that gives curiosity, exploration, and following the caretaker. When affect is low (unfamiliar), motivation-to-continue is negative, so the robot *subtracts* — that gives stability-seeking and avoidance. A separate *pleasure* signal — the moment-to-moment change in well-being — gives an extra positive or negative kick whenever something good or bad just happened, generating opportunism and danger-avoidance reflexes.

**The setup.** A Koala robot (a small four-wheeled robot from K-Team) faces a "caretaker" (an object that triggers the lateral IR sensors when nearby). The only sensor the controller uses is *distance to the caretaker* via infrared. Side IR contact gives the robot well-being (a reward). The controller is the affect-modulated PerAc architecture described above, with ten time-scale-$k$ copies, $k \in \{0.2, 0.4, \ldots, 2.0\}$. Parameters tested: $r = 1$, $s \in \{0.1, 0.001\}$, $q \in \{0, 0.5, 0.75, 1.0\}$.

**The results.** Plots in the paper show:
- Cautious-approach trajectories that depend on $q$: low $q$ → confident smooth advance; high $q$ → hesitant, partly retreating advance.
- With a reward at the front, the robot accelerates toward it (exp 2). With the reward already nearby, the robot keeps it instead of exploring (exp 3).
- With a moving caretaker, the robot tracks the caretaker when within the familiar zone (imitation) and flips to avoidance once outside (imitation breaks down on the safe-to-unsafe boundary).

**Why it matters.** The architecture demonstrates that **a sign-flip on a single modulator** — driven by an affect/well-being readout of the agent-environment match — can replace discrete behavior-selection logic. This is a small but cleanly worked example of the *neuromodulation*-family stance that Cañamero advocates in `canamero_2005_emotion_understanding.md`.

## Phase 2 — graduate-level deep dive

### State, sensation, and well-being

Let $W_b(t) \in [0,1]$ be the robot's well-being scalar — a function of internal (physiological) state that rises when lateral IR sensors are stimulated and decays otherwise. Let $S_d(t)$ be the (front) IR-distance sensation. The architecture maintains ten time-scale-indexed copies of all dynamic variables, $k \in K = \{0.2, 0.4, \ldots, 2.0\}$.

### Desired sensation via well-being-weighted Rescorla–Wagner

The classical Rescorla–Wagner (1972) trial-averaging rule with rate $\eta(t) = 1/t$ produces a running mean of $S_d$. The authors generalize it by replacing the constant rate with a *well-being-weighted* rate:

$$
\eta_k(t) = \left( \frac{W_b(t)}{\widetilde{W_b}(t)} \right)^k, \qquad \widetilde{W_b}(t) = \sum_{\tau \le t} W_b(\tau),
$$

and update the time-scale-$k$ desired sensation by

$$
\overline{S_d}_k(t) = \overline{S_d}_k(t-1) + \eta_k(t) \cdot \big( S_d(t) - \overline{S_d}_k(t-1) \big). \qquad (1)
$$

This is the paper's Equation (1) generalized to a $k$-indexed family. Two limits clarify the role of $k$:

- **$k \to 0^+$:** $\eta_k(t) \to 1$ (since $a^0 = 1$), so $\overline{S_d}_k(t) \to S_d(t)$ — purely short-term tracking. The "desired sensation" at this time scale is whatever the robot is sensing right now.
- **$k \to +\infty$:** $\eta_k(t) \to 0$ for any $W_b/\widetilde{W_b} < 1$, so $\overline{S_d}_k(t) \to \overline{S_d}_k(0)$ — purely long-term memory dominated by initial conditions.

For finite $k$, learning rate scales monotonically with the *relative* size of current $W_b$ vs. lifetime cumulative $\widetilde{W_b}$. The same $\eta_k$ is reused (Eq. 8 in the paper) to update the average well-being at each time scale,

$$
\overline{W_b}_k(t) = \overline{W_b}_k(t-1) + \eta_k(t) \cdot \big( W_b(t) - \overline{W_b}_k(t-1) \big). \qquad (8)
$$

### Perceived action: inverse model in PerAc

For each $k$, the *sensory mismatch* is

$$
\Delta d_k(t) = S_d(t) - \overline{S_d}_k(t).
$$

The *perceived action* is the action that would have produced $\Delta d_k$ in one step, with a time-scale-dependent gain:

$$
P_{a,k}(t) = \epsilon^k \cdot \Delta d_k(t), \qquad (2)
$$

where $\epsilon \in (0,1)$ ensures larger $k$ → smaller gain. This is the *inverse model* of the PerAc framework — a hand-tuned mapping from perceptual residuals to motor commands, replacing the babbling-learned sensorimotor association of Andry et al. 2003.

### Affect, apathy, openness

The paper's central modulator, *affect*, is

$$
A_{f,k}(t) = \exp\!\big( -s \cdot P_{a,k}(t)^2 \big) \in (0, 1], \qquad (6)
$$

a Gaussian-shaped *familiarity-in-action-space* kernel of width $1/\sqrt{s}$. When the inverse model needs little or no motion to reach $\overline{S_d}_k$, $A_{f,k} \to 1$ (highly safe / familiar). When the inverse model demands large motion, $A_{f,k} \to 0$ (unfamiliar). Note that $A_{f,k}$ lives in *sensory-action* space, not pure sensory space — it depends on $P_{a,k}$ which is the gain-scaled mismatch.

*Openness to the world*:

$$
O_{p,k}(t) = A_{f,k}(t) - q, \qquad q \in [0,1]. \qquad (7)
$$

The sign of $O_{p,k}$ depends on whether $A_{f,k}$ exceeds the *timorousness* threshold $q$. For $q = 0$, $O_{p,k}$ is always non-negative — the robot always amplifies. For $q = 1$, $O_{p,k}$ is always non-positive — the robot always opposes. Intermediate $q$ creates a *familiarity-threshold-gated* sign flip: amplify when the current sensation is more familiar than threshold, oppose when less.

*Apathy*:

$$
A_p(t) = \exp\!\big( -r \cdot M_c(t)^2 \big) \in (0,1], \qquad (5)
$$

a Gaussian kernel on the motivation-to-continue scalar. When $M_c$ is near zero (no strong sign either way), $A_p \approx 1$ — the robot is bored. When $|M_c|$ is large, $A_p \to 0$.

*Innate exploration drive*:

$$
A_e(t) = \big( A_e(t-1) + B_e(t) \big) \cdot A_p(t), \qquad (4)
$$

with $B_e$ an innate forward-motion bias. The product with $A_p$ gates the drive: high-apathy → drive grows; sufficient motivation → drive collapses.

### Pleasure as well-being prediction error

For each $k$, *pleasure* is

$$
P_{l,k}(t) = W_b(t) - \overline{W_b}_k(t) \in [-1, 1]. \qquad (9)
$$

This is structurally a *well-being prediction error* — the analogue of a reward-prediction-error signal in the dopamine-as-RPE literature, at multiple temporal horizons. The authors flag this connection explicitly in the conclusion ("we would like to further investigate … its similarity with the brain dopaminergic circuits").

### Motivation to continue

The final modulator is

$$
M_{c,k}(t) = O_{p,k}(t) + P_{l,k}(t). \qquad (10)
$$

The sign of $M_{c,k}$ decides amplification ($M_{c,k} > 0$) versus opposition ($M_{c,k} < 0$). The four canonical behaviors then map cleanly to the sign and source of $M_c$:

| Behavior | Driven by | Sign of $M_c$ |
|---|---|---|
| Seeking stability | $O_{p,k} < 0$ (low affect) | $M_c < 0$ — oppose the perceived motion that would have produced $\Delta d_k$ |
| Cautious exploration | $O_{p,k} > 0$ (safe), $A_p \approx 1$ (no other drive) | $M_c > 0$ — amplify innate forward bias $B_e$ |
| Opportunistic exploitation | $P_{l,k} \gg 0$ (sudden $W_b$ jump) | $M_c > 0$ — amplify the action that just paid off |
| Avoidance | $P_{l,k} \ll 0$ (sudden $W_b$ drop) | $M_c < 0$ — reverse the action that just hurt |
| Low-level imitation | Caretaker motion injects $\Delta d_k$; if still $A_f$ high → $M_c > 0$ | Amplify the perceived motion induced by caretaker |

### Final action computation

Per time scale $k$, the executed action is

$$
A_{c,k}(t) = P_{a,k}(t) \cdot M_{c,k}(t), \qquad (3)
$$

and the robot's actual motor command is presumably (the paper does not state it explicitly but Figures 3, 4, 6, 9 imply) an average across the ten time scales:

$$
A_c(t) = \tfrac{1}{|K|} \sum_{k \in K} A_{c,k}(t).
$$

This decomposition — running the same modulator pipeline at multiple temporal horizons and averaging — is the architectural device that lets a *single* equation give both fast withdrawal reflexes (from short-$k$ time scales) and slow imprinting-style stability (from long-$k$ time scales).

### Pseudocode

```
Initialize: for each k in K: Sd_bar_k = 0, Wb_bar_k = 0, Ae = 0
Parameters: s, q, r, epsilon, K = {0.2, 0.4, ..., 2.0}, B_e (innate forward bias)

At each t:
  Read W_b(t)          # internal physiology
  Read S_d(t)          # IR distance to caretaker
  cumulative_Wb += W_b(t)
  for each k in K:
    eta_k = (W_b(t) / cumulative_Wb) ** k
    Sd_bar_k  += eta_k * (S_d(t) - Sd_bar_k)
    Wb_bar_k  += eta_k * (W_b(t) - Wb_bar_k)
    P_a_k      = (epsilon ** k) * (S_d(t) - Sd_bar_k)
    A_f_k      = exp(-s * P_a_k ** 2)
    O_p_k      = A_f_k - q
    P_l_k      = W_b(t) - Wb_bar_k
    M_c_k      = O_p_k + P_l_k
  M_c_total = mean_k(M_c_k)
  A_p = exp(-r * M_c_total ** 2)
  A_e = (A_e + B_e) * A_p                 # apathy-gated exploration drive
  for each k in K:
    A_c_k = P_a_k * M_c_k + A_e            # combine inverse-model and exploration drive
  Execute action mean_k(A_c_k)
```

(The paper's text and Figure 9 suggest $A_e$ enters as an additive bias to the action stream rather than to $M_c$ itself; the schematic captures the spirit of the architecture.)

### Mathematical commentary

Two structural choices are worth highlighting:

1. **Sign-flip via Gaussian-of-mismatch minus threshold.** $A_{f,k} - q$ implements a *smooth* approximation to a sigmoidal indicator function on whether $|P_{a,k}|$ is below or above some critical magnitude. Tuning $q$ shifts the cross-over; tuning $s$ sharpens it. This is the smooth analogue of "is the world familiar enough to play?"

2. **Pleasure as RPE on well-being.** $P_{l,k} = W_b - \overline{W_b}_k$ is exactly the temporal-difference structure familiar from RL — except that the "value function" being differenced is the running average of well-being rather than expected discounted reward. The conclusion's gesture toward dopamine is therefore well-founded: dopaminergic systems are widely modelled as carrying reward-prediction-error signals that affect action vigor (Niv-Daw-Joel-Dayan 2007) — exactly the role $P_{l,k}$ plays here in modulating $M_c$. This same well-being-RPE-like construct reappears, in more refined form, in `lewis_canamero_2016_hedonic_pleasure.md`.

## Connections

- **Direct ancestor.** `canamero_1997_motivations_emotions.md` — the homeostatic-physiology backbone and the idea of emotion/affect as a modulator of action selection. Blanchard & Cañamero replace the Abbott motivation-list with a single $W_b$ scalar and a single $A_f$ scalar but inherit the architectural philosophy.
- **Methodological framing.** `canamero_2005_emotion_understanding.md` — this 2006 paper is a concrete worked example of the *designed-emotion / neuromodulation-family* stance that Cañamero 2005 advocates over the *circuit-family* alternative.
- **Affordance-learning successor.** `cos_2010_affordances_consummatory.md` — Cos et al. 2010 takes the same well-being-grounded reward idea and extends it to learn object *affordances* over multiple consummatory behaviors.
- **Hedonic refinement of the pleasure signal.** `lewis_canamero_2016_hedonic_pleasure.md` — Lewis & Cañamero 2016 critique the equation "pleasure = change in well-being" (effectively the $P_l$ definition here) and propose a more careful hedonic-quality formulation that decouples pleasure from need.
- **Hormone-driven lifetime-adaptation successor.** `lones_canamero_2013_epigenetic_hormones.md` and `lones_2018_hormone_epigenetic.md` — keep the well-being / affect modulator architecture and add a second, slower hormone-driven plasticity layer.
- **Cross-corpus link.** The closing remark about dopaminergic circuits in behavior selection points to the same dopamine-as-modulator literature that Cox & Krichmar 2009 and Avery & Krichmar 2017 (other batches) develop more explicitly. The curator should cross-link.
