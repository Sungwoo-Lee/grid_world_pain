---
title: "Learning Affordances of Consummatory Behaviors: Motivation-Driven Adaptive Perception"
authors: ["Ignasi Cos", "Lola Cañamero", "Gillian M. Hayes"]
year: 2010
venue: "Adaptive Behavior 18(3–4), 285–314, SAGE (International Society for Adaptive Behavior)"
slug: cos_2010_affordances_consummatory
source_pdf: "sources/Cos et al. 2010 - Learning Affordances of Consummatory Behaviors - Motivation-Driven Adaptive Perception.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

This paper takes Gibson's psychology-of-perception idea of an **affordance** — the things an object lets you do (a fruit "affords" eating, a tree "affords" shade) — and shows how a small wheeled robot can *learn* affordances directly from the **hormonal feedback** that its own simulated body gives after a successful consummatory action. The agent is a Khepera robot with a gripper, controlled by an architecture with three pieces. (1) A **synthetic physiology** of three homeostatic variables — *nutrition* (food in blood), *stamina* (energy), and *restlessness* (need to engage) — each with an ideal set point and a viability range; when a variable drifts from its set point a corresponding **drive** (hunger, fatigue, curiosity) rises. (2) A **Growing-When-Required (GWR) self-organizing network** that clusters the visual snapshots of nearby objects into topological nodes — adding a new node when the current input is "underrepresented" by the existing map. (3) An **affordance network**: a layer of synapses $\chi_{kf}$ connecting each GWR node $f$ to each behavior $k$ (eat / rest / interact). Whenever the agent picks a behavior and executes it next to an object, the resulting *sudden* change in a homeostatic variable triggers a **hormonal release** $S$ that Hebbian-reinforces the synapse $\chi_{kf}$ from the currently active node to the just-executed behavior. Over many random interactions, $\chi_{kf}$ converges on the *true* affordance distribution — small objects afford eating, medium objects afford resting, large objects afford interacting — and the agent then picks behaviors using an **affordance × drive** product, beating a pure motivation-driven (drive-only) baseline on a *physiological stability* metric. The paper is significant in the corpus because it shows that **hormonal feedback** can play the role of reward in a Hebbian/REINFORCE-style affordance-learning rule — and that doing so binds *perception, motivation, and action* into a single adaptive ecology rather than treating them as separable modules.

## Section-ordered backbone

### Abstract
Introduces a formalization of the dynamics between sensorimotor interaction and homeostasis, integrated in a single architecture to learn object affordances of consummatory behaviors. Describes the principles necessary to learn grounded knowledge in the context of an agent and its surrounding environment, then tests in an embodied, situated (simulated) robot. Learned affordances are dynamically redefined depending on object similarity, resource availability, and the rhythms of internal physiology — e.g., if a resource becomes scarce, the value of its effect rises.

### Section 1 — Introduction
Locates the work in Gibson's (1986) affordance theory: perception as directed potentiality-for-action, not as a fixed input-output stage. Reviews neurophysiological support for the simultaneous representation of multiple potential actions on the frontoparietal cortical loops (Cisek & Kalaska 2005; Song & Nakayama 2007). Argues that adaptation work has focused on action selection while neglecting the dual question — what *signals* tie behaviors to environmental elements? The answer proposed: **the physiological effect of a successful behavioral interaction**. If an object affords the currently-executed behavior, that interaction compensates a homeostatic variable; the resulting fluctuation can be used to learn the object's perceptual signature → behavior mapping. The continuous-learning view also handles novelty (fruits never seen before are recognized as edible via similarity) and *internal-state-dependent revaluation* (eating becomes more rewarding when food gets scarce, even at unchanged caloric content).

### Section 2 — Background and Related Research
Reviews the affordance literature: original Gibson 1986, Turvey 1992 (affordances as properties of the environment), Stoffregen 2003, Şahin et al. 2007 (a three-perspective formalization — agent / environment / observer). Distinguishes *representationalist* and *Gibsonian* readings (Chemero & Turvey 2007). Cites Spier & McFarland's 1996 "learning without cognition" framework as the methodological closest relative. Notes use of affordances in HCI (St. Amant 1999), in cognitive neuroscience as cortical loops (Fagg & Arbib 1998; Cisek 2007), and as imitation primitives (Nehaniv & Dautenhahn 1998). The paper positions itself in the *agent's-perspective* slot of Şahin et al.'s formalization.

### Section 3 — Motivation-Based Affordances
Sets up the conceptual scaffolding: homeostasis (Cannon 1929; Maturana & Varela 1980), drives as urges to compensate (Hull 1943), motivation/reinforcement as two directions of the same neuropsychological substrate (Bindra 1969), reward as the common currency for behavior selection (McFarland & Sibly 1975; Redgrave, Prescott & Gurney 1999), and physiological stability as the survival metric (Ashby 1965). Cites the basal-ganglia reinforcement-comparison hypothesis (Houk, Adams & Barto 1995; Sutton & Barto 1981) and the neuromodulatory-phenomena framing (Fellous 2001, 2004) as the neural plausibility scaffolding for the architecture they will propose.

### Section 4 — The Model

**4.1 Overview.** Three concurrent processes: artificial physiology (homeostatic dynamics + drives), GWRN clustering of visual snapshots, affordance learning via hormonal-reinforced synapses.

**4.2 Artificial Physiology.** Each homeostatic variable $V_i$ obeys

$$
\dot V_i(t) = -\frac{V_i(t)}{\tau_i} + \sum_k \alpha(b_{ki}) \cdot \delta\!\left( t - t_{j(k)} \right), \qquad (\text{Eq. 1})
$$

with $\tau_i$ the natural decay constant, $\alpha(b_{ki})$ the impulsive effect of behavior $b_k$ on variable $i$, and $\delta$ the Dirac delta firing at the moment $t_{j(k)}$ of successful execution. The analytical solution is

$$
V_i(t) = V_0 \, e^{-t/\tau_i} + \sum_k \alpha(b_{ki}) \cdot u\!\left( t - t_{j(k)} \right) e^{-(t - t_{j(k)})/\tau_i}, \qquad (\text{Eq. 2})
$$

with $u$ the Heaviside step. Three variables are used in the experiments: *nutrition* (compensated by `eat`), *stamina* (compensated by `rest`), *restlessness* (compensated by `interact`). Each has an optimal set point and viability bounds. Drives are defined as

$$
D_m(t) = \sum_l a_{lm} \big( V_{\text{opt},l} - V_l(t) \big) + \sum_l b_{lm} \dot V_l, \qquad (\text{Eq. 3})
$$

with $a_{lm}$ coefficients tying variable $l$ to drive $m$ and $b_{lm}$ coefficients tying its rate of change to the drive. The paper simplifies to the case where each drive depends on a single variable, linearly, with $b_{lm}=0$. The behaviors form a coarse-grained subsystem layer with a winner-takes-all disinhibition. Two arbitration policies are compared: **motivation-driven** (pick the behavior tied to the strongest drive) and **affordance × drive** (pick $\arg\max_k \chi_{kf^*} \cdot \overline{D}_k$, multiplying the learned affordance value by the current drive intensity).

**4.3 GWRN for Sensory Perception.** Snapshots are normalized 64×1 vectors of light intensity. The Growing-When-Required network (Marsland, Shapiro & Nehmzow 2002) is used with three parameters: activity threshold $a_T$, habituation threshold $h_T$, and maximum synapse age $\text{age}_{\max}$. For each input $\vec x_q$, the closest two nodes $i,j$ are identified, a synapse grown, node activities computed by

$$
a_n = \exp\!\left( -\big\lVert \vec\omega_n - \vec x_q \big\rVert^2 \right), \qquad n \in \{1, \ldots, N\}, \qquad (\text{Eq. 4})
$$

then either a new node is inserted (if no node fits well — activity below $a_T$ — and habituation $u$ is below $h_T$) or the winner and adjacent nodes drag toward $\vec x_q$ by amounts proportional to mismatch (Eqs. 5–6). Habituation decays per Eq. 7–8 with normalization $\kappa_r, \kappa_b$ and time constants $\tau_r, \tau_b$. Synapses age and are pruned if older than $\text{age}_{\max}$.

**4.4 Learning Affordances.** A second set of *functional* synapses $\chi_{kf}$ connects each GWRN node $f$ to each behavior $k$, initialized at small random values. After execution of behavior $k$ that produced a homeostatic-variable jump:

$$
\chi_{kf} \leftarrow \chi_{kf} + \gamma \, S, \qquad (\text{Eq. 9})
$$

with $\gamma$ a small positive learning rate and $S$ the hormonal-response value (Eq. 10 below). Only the synapse linking the active node $f$ to the just-executed behavior $k$ is updated; all synapses also slow-decay to prune disuse.

**4.5 Hormonal Reinforcement.** The hormone $S$ obeys

$$
\dot S = -\frac{S}{\tau_s} + \sum_i \beta_i \sum_n \delta\!\big( t - t_n \big), \qquad t_n = \{ t : \dot V_i > X^* \}, \qquad (\text{Eq. 10})
$$

with $\tau_s$ the hormonal time constant, $\beta_i$ a per-variable normalization, and $t_n$ the moments at which a homeostatic variable's derivative exceeds the *sudden-change* threshold $X^*$. Because $\tau_s \ll \tau_i$, $S$ rises sharply on a successful interaction and decays quickly to zero — a Dirac-like teaching signal that gates the Hebbian update in Eq. 9.

### Section 5 — Experiments

**Setup.** Simulated Khepera robot in WebotsTM, with a gripper for grasping. Behaviors: `wander` (random exploration until an object is in IR range), `eat` (only succeeds on objects ≤ gripper width 0.04), `rest` (succeeds on medium objects, ≥ 0.04), `interact` (succeeds on large objects, ≥ 0.07; in the *abundant* environment all objects support `interact`). Two environments — *simple* (two object types) and *complex* (sizes spread continuously). Two distributions — *abundant* and *scarce*. Two physiology decay regimes — slow ($\tau = 10^{-4}$/s) and fast ($\tau = 10^{-3}$/s).

**Metrics.** GWRN fitting error (Eq. 11):

$$
\sigma_k = \mathbb{E}\!\left\{ \sum_{f=0}^{M-1} \sigma_{\chi_{kf}} \right\}.
$$

Physiological stability (Eq. 12):

$$
\text{Physiological Stability} = \mathbb{E}\!\left\{ \tfrac{1}{N} \sum_{m=0}^{N-1} D_m \right\}.
$$

Lower is better. Behaviorally, the agent is asked to keep this small under varying environments.

**5.1 Simple environment.** Parameter sweeps fix $a_T \in [0.5, 0.9]$, $h_T \in [0.01, 1.0]$, $\text{age}_{\max} \in [5, 40]$. Findings: fitting error decreases 0.03 → 0.01 as $a_T$ rises 0.5 → 0.9; plateau at $\text{age}_{\max} > 20$; $h_T$ has little effect.

**5.2 Complex environment.** Three sub-experiments:
- Slow decay × scarce affordances: drives drop and physiological stability rises as the GWRN network grows; for the *affordance × drive* policy the curve plateaus around $\sim 15$ nodes. The *motivation-driven* baseline is much worse — it ignores object affordances and so wastes interactions.
- Fast decay × scarce affordances: motivation-driven policy is catastrophic — drives saturate near 1.0 (close to the lethal boundary); affordance × drive policy still tracks the same downward trend but at a higher floor.
- Fast decay × abundant affordances: gap closes — motivation-driven is now adequate because successful interactions are easy to find.

**5.3 Theoretical formalization.** Derives an analytical lethal-boundary condition. For the most-critical variable $i$, the time to lethality in the absence of compensation is

$$
t_{\max} = -\tau_i \log\!\left( \frac{V_i^*}{V_i^* + b_{ki}} \right), \qquad (\text{Eq. 13})
$$

with $V_i^*$ the lethal threshold and $b_{ki}$ the per-interaction compensatory effect. Assuming compensatory interactions are a Poisson process with rate $\lambda_i$ (and that all interactions succeed), the survival probability is

$$
P_{\text{survival}} = P\{T < t_{\max}\} = 1 - e^{-\lambda_i t_{\max}}, \qquad (\text{Eq. 14})
$$

so to survive with probability $\ge p$,

$$
\lambda_i \ge -\frac{\log(1-p)}{t_{\max}} = -\frac{\log(1-p)}{-\tau_i \log\!\big( V_i^* / (V_i^* + b_{ki}) \big)}, \qquad (\text{Eq. 15})
$$

i.e., scarce environments (small $\lambda_i$) and fast decay (small $\tau_i$) collude to push the agent below the survival threshold — quantitatively explaining the dramatic difference between the *fast-decay × scarce* and *fast-decay × abundant* outcomes.

### Section 6 — Discussion
Contrasts with Spier & McFarland 1996/1997 *drk* model (which learned $d$-values via a delta rule but disregarded perception). Three differences: hormonal release indirectly controls learning and gives a *valency assessment* (Ackley & Littman 1991) — eating is good if hungry, bad if sated; the architecture is framed by Ashby's physiological stability; perception is non-trivial (GWRN + affordance network). Discusses external vs. internal effect: future extension to learn affordances via *both* the visual change in the environment and the internal physiological response. Limitations: snapshot-based vision (not optic flow), no claim of biological faithfulness, simulator-only experiments, univocal variable-drive-behavior mapping.

### Section 7 — Conclusion
Reiterates the architecture (GWRN + affordance network + hormonal reinforcement of consummatory effects), the two-metric assessment (fitting error + physiological stability), and the headline finding: the **affordance × drive** policy beats motivation-driven baseline on physiological stability under scarce and fast-decay regimes. The paper argues that perception and motivation should be designed *together*, not separately.

## Phase 1 — undergraduate-level synthesis

**The plain idea.** Give a robot a body with three hunger-like drives (hunger, fatigue, curiosity). Give it three things it can do — `eat`, `rest`, `interact`. Drop it in a world full of objects of various sizes. Some of those objects let `eat` succeed (small ones the gripper can grasp), some let `rest` succeed (medium-sized), some let `interact` succeed (large or any in an abundant world). The robot doesn't know which object affords which behavior. Whenever the robot tries something and it works — when its blood-sugar / energy / engagement actually rises — its body releases a brief "hormone" $S$ that lasts much less time than the slow physiology. That hormone, which only fires when the homeostatic variables are actively being compensated, drives a Hebbian update: strengthen the link from "what I was looking at" → "what I just did". After enough random trials, the robot has learned, for every object cluster, which behavior is afforded. Now combine that learned affordance map with the current drive — pick the behavior that is *both* afforded by the nearby object *and* needed by the current physiology — and the robot survives much longer.

**The setup.** A simulated Khepera robot in Webots with infrared sensors and a camera. Vision is reduced to 64-pixel intensity snapshots normalized to $[0,1]$. The robot's vision is clustered by a *Growing-When-Required* network — a self-organizing map that adds a node whenever no existing node matches the input well enough. The affordance layer is a separate set of synapses $\chi_{kf}$ from GWRN nodes to behaviors. After every behavior attempt, if the body rewards it with a hormone $S$, the relevant synapse is strengthened.

**The result.** In a scarce environment with fast-decaying physiology — the hardest condition — the standard *motivation-driven* policy (always do whatever your most-urgent drive demands) is catastrophic, with drives pinned near death. The *affordance × drive* policy — multiply the learned affordance value $\chi_{kf}$ by the current drive intensity $D_m$ before picking — keeps the agent alive much longer. In an abundant environment with slow physiology, the difference shrinks because both policies have many chances to succeed. The paper's analytic Poisson model (Eq. 15) explains *why* — the survival condition is $\lambda_i \ge -\log(1-p)/t_{\max}$.

**Concrete worked example.** Object size 0.03 m. The GWRN clusters this into a node $f_3$. The agent (hungry) executes `eat` near $f_3$; it succeeds; nutrition jumps by $\alpha = 0.3$; the hormone $S$ spikes; the synapse $\chi_{\text{eat}, f_3}$ grows by $\gamma S$. Subsequent encounters with size-0.03 objects increasingly pick `eat`. Object size 0.06 → cluster $f_5$. Agent attempts `eat` → fails (object too big for gripper) → no hormone → no update. Agent attempts `rest` → succeeds → $\chi_{\text{rest}, f_5}$ grows. Over many interactions the agent autonomously partitions object space by affordance.

## Phase 2 — graduate-level deep dive

### Continuous-time physiology

The synthetic physiology is a linear leak driven by impulsive consumption:

$$
\dot V_i(t) = -\frac{V_i(t)}{\tau_i} + \sum_k \alpha(b_{ki}) \sum_{j(k)} \delta\!\big( t - t_{j(k)} \big),
$$

where $\alpha(b_{ki}) \in \mathbb{R}_+$ is the per-success kick of behavior $b_k$ on variable $V_i$, and $\{t_{j(k)}\}_j$ are the moments of successful executions of $b_k$. Solving by variation of constants:

$$
V_i(t) = V_0 \, e^{-t/\tau_i} + \sum_k \alpha(b_{ki}) \sum_{j(k)} u\!\big( t - t_{j(k)} \big) \, e^{-\big( t - t_{j(k)} \big)/\tau_i}.
$$

In experiments: $\alpha_{\text{eat}, \text{nutrition}} = 0.3$, $\alpha_{\text{rest}, \text{stamina}} = 0.2$, $\alpha_{\text{interact}, \text{restlessness}} = 0.1$. Decay-rate regimes: slow $\tau_i = 10^{-4}/s$, fast $\tau_i = 10^{-3}/s$.

### Drives

$$
D_m(t) = \sum_l a_{lm} \big( V_{\text{opt},l} - V_l(t) \big) + \sum_l b_{lm} \dot V_l.
$$

Restricted to a single-variable linear case ($b_{lm} = 0$, only one nonzero $a_{lm}$ per $m$):

$$
D_m(t) = a_m \big( V_{\text{opt},m} - V_m(t) \big),
$$

clamped to $[0,1]$ in practice.

### Behavior arbitration

Let $f^*(t) = \arg\max_f a_f(t)$ be the currently active GWRN node and let $\chi_{kf^*}(t) \in [0,1]$ be the affordance synapse weight.

**Motivation-driven baseline:**

$$
k^*(t) = \arg\max_m D_m(t).
$$

**Affordance × drive policy:**

$$
k^*(t) = \arg\max_k \big[ \chi_{k f^*(t)}(t) \cdot D_{m(k)}(t) \big],
$$

where $m(k)$ is the drive that behavior $k$ compensates. The product encodes Bindra's (1969) unified motivation-reinforcement substrate.

### GWRN clustering

Inputs $\vec x_q \in [0,1]^{64}$. Existing nodes $\{\vec\omega_n\}_{n=1}^N$. Activity:

$$
a_n = \exp\!\big( -\lVert \vec\omega_n - \vec x_q \rVert^2 \big).
$$

(The original GWR network in Marsland, Shapiro & Nehmzow 2002 uses a linear metric; Cos et al. replace it with the squared-Euclidean Gaussian to collapse near-duplicates and separate distant nodes more crisply.) Habituation $u_n(t)$ decays per

$$
u_r(t) = u_0 \cdot \tfrac{1}{\kappa_r} \cdot \big( 1 - \exp(-\kappa_r t / \tau_r) \big),
$$

(equivalent expression for adjacent nodes with $\kappa_b, \tau_b$). The two thresholds gate node insertion:

- If $a_r < a_T$ AND $u_r > h_T$: insert a new node $o$ at midpoint between the best-match $r$ and $\vec x_q$; replace the $r$-$g$ synapse with two new ones.
- Else: drag the winner and its $L$ neighbors toward $\vec x_q$ by

$$
\Delta \vec\omega_r = e_b \cdot u_b \cdot (\vec x_q - \vec\omega_r), \qquad \Delta \vec\omega_b = e_b \cdot u_b \cdot (\vec x_q - \vec\omega_r) \ \text{ for } b \in \{1, \ldots, L\}.
$$

Synapse ages are decremented on use, pruned past $\text{age}_{\max}$. Lonely nodes are pruned.

### Hormonal teaching signal

The hormone $S$ is generated by sudden compensatory jumps in physiology:

$$
\dot S = -\frac{S}{\tau_s} + \sum_i \beta_i \sum_n \delta(t - t_n), \qquad t_n = \big\{ t : \dot V_i(t) > X^* \big\},
$$

with $\tau_s \ll \tau_i$. Sharp rises in any homeostatic variable trigger an $S$ pulse that decays before the next variable update. Integrating between two pulses,

$$
S(t) = \sum_n \beta_i \, u(t - t_n) \, e^{-(t - t_n)/\tau_s},
$$

i.e., a short-window exponential teaching signal that gates the Hebbian update.

### Affordance learning rule

For the synapse $\chi_{kf}$ from the currently active node $f$ to the just-executed behavior $k$:

$$
\chi_{kf}(t+\Delta t) = \chi_{kf}(t) + \gamma \, S(t) \, \mathbb{1}\{k\text{ was just executed}\} \mathbb{1}\{f\text{ is the active node}\} - \delta_{\text{decay}} \, \chi_{kf}(t),
$$

with $\delta_{\text{decay}}$ a slow uniform decay that prunes unused synapses. Because $S \neq 0$ only when $\dot V_i > X^*$ (a successful compensation), the rule is a *post-synaptic-reward gated* Hebbian update — structurally a three-factor Hebbian rule of the form

$$
\Delta \chi_{kf} \propto \underbrace{a_f}_{\text{pre}} \cdot \underbrace{\mathbb{1}\{k^* = k\}}_{\text{post}} \cdot \underbrace{S}_{\text{neuromodulator}},
$$

the canonical biological-plausibility template that the broader neuromodulation-as-learning-modulator literature (Doya 2002; Fellous 2004; Schultz, Dayan & Montague 1997) formalizes. The hormone $S$ is the *reward* (or in dopamine-as-RPE language, a reward-prediction-like signal, though the paper does not subtract a baseline).

### Survival analysis (Eqs. 13–15)

Worst-case variable: lifetime in absence of compensation is

$$
V_i(t_{\max}) = V_0 \, e^{-t_{\max}/\tau_i} = V_i^* \quad \Rightarrow \quad t_{\max} = \tau_i \log\!\big( V_0 / V_i^* \big).
$$

The paper writes the equivalent expression in terms of the *per-interaction* gain $b_{ki}$ that just lifts the variable above $V_i^*$:

$$
t_{\max} = -\tau_i \log\!\left( \frac{V_i^*}{V_i^* + b_{ki}} \right). \qquad (\text{Eq. 13})
$$

Modeling compensatory interactions as a homogeneous Poisson process of rate $\lambda_i$,

$$
P_{\text{survival}} = P\{T < t_{\max}\} = P\{n > 0\} = 1 - e^{-\lambda_i t_{\max}}. \qquad (\text{Eq. 14})
$$

Solving for the minimum interaction rate to survive with probability $p$:

$$
\lambda_i \ge -\frac{\log(1-p)}{t_{\max}} = -\frac{\log(1-p)}{-\tau_i \log\!\big( V_i^* / (V_i^* + b_{ki}) \big)}. \qquad (\text{Eq. 15})
$$

This survival-rate inequality is the analytical justification for why the *fast-decay × scarce-affordance* condition is lethal under the motivation-driven baseline: $\lambda_i$ (rate of successful encounters under random exploration) falls below the inequality's right-hand side, so $P_{\text{survival}} \to 0$. The affordance × drive policy raises $\lambda_i$ — picking *afforded* behaviors at every encounter rather than wasting trials on non-afforded ones — and pushes the agent back above the threshold.

### Pseudocode of the architecture

```
Initialize GWRN with two random nodes; chi_kf at small random; V_i at set-points; S = 0.
Parameters: tau_i, tau_s, alpha_{k,i}, beta_i, X*, a_T, h_T, age_max, gamma, decay_chi
Behavior repertoire: B = {eat, rest, interact}; wander as default

At every dt:
  Physiology:
    for each variable i: V_i ← V_i - V_i dt / tau_i      # leak
    S        ← S - S dt / tau_s
    for each i with dV_i/dt > X*: S ← S + beta_i        # hormone pulse on rapid compensation
  Perception (only when within IR range of an object):
    snapshot vec_x   = 64-pixel light intensities, normalized
    GWRN_step(vec_x, a_T, h_T, age_max)                  # node insertion / drag / pruning
    f*  = argmax_n a_n                                   # active node
  Drives:
    for each m: D_m = max(0, a_m * (V_opt_m - V_m))
  Behavior arbitration (only on object encounter):
    if policy == 'motivation_driven':
        k* = argmax_m D_m
    else:  # 'affordance_x_drive'
        k* = argmax_k (chi[k, f*] * D_{m(k)})
    execute behavior b_{k*} on the object
    if execution succeeds:                              # object affords k*
        V_{i(k*)} += alpha[k*, i(k*)]                   # impulsive bump
        # hormone S now spikes via the dV_i/dt > X* trigger
  Learning rule (every dt):
    for each (k, f): chi[k, f] ← (1 - decay_chi) * chi[k, f]
    if (object encountered, k* just executed, S > 0):
        chi[k*, f*] ← chi[k*, f*] + gamma * S
  Mortality: if any V_i < V_i_lethal: agent dies; episode ends
```

### Mathematical commentary

Three structural choices repay reflection:

1. **The hormone as a Dirac-like teaching signal.** Because $\tau_s \ll \tau_i$, $S$ is essentially a pulse — non-zero only in the immediate aftermath of a successful interaction. This makes the Hebbian update Markov in time-rate: only the most-recent active node and most-recent behavior get credit. The architecture deliberately avoids eligibility traces, simplifying the credit-assignment problem at the cost of giving up multi-step credit. This is a design choice — the paper does not claim it would scale to delayed-reward tasks.

2. **The product affordance × drive is exactly a *value-of-information* objective.** $\chi_{kf}$ is a learned conditional probability that the object affords $k$; $D_m$ is the urgency of needing the variable that $k$ compensates. The product is the *expected utility* of attempting $k$ given the current state — the agent is playing a one-step bandit. This connects the architecture to standard RL even though the paper presents it as a Hebbian variant.

3. **The Poisson-survival analysis is a closed-form lower bound on lifespan.** Eq. 15 gives an a-priori survival condition without any reference to the agent's learning algorithm — only $\tau_i$ (physiology), $b_{ki}$ (behavior potency), $\lambda_i$ (environment richness), and $p$ (desired survival probability). This factorization is the cleanest articulation in the corpus of the *environment-vs-physiology dependency* that all subsequent Cañamero-school papers worry about — especially `lewis_canamero_2016_hedonic_pleasure.md`, where the connection between hedonic structure and survival is rethought.

## Connections

- **Direct architectural ancestor.** `canamero_1997_motivations_emotions.md` — same homeostatic-physiology + drives + winner-takes-all behavior arbitration. Cos et al. add the affordance-learning layer and the hormonal teaching signal.
- **Affect-modulated PerAc cousin.** `blanchard_canamero_2006_affect_modulated.md` — also uses a hormone-like / well-being-derived modulator to bias action selection, but is concerned with *amplifying vs. opposing* a single inverse-model output rather than learning a *what-affords-what* map.
- **Methodological framing.** `canamero_2005_emotion_understanding.md` — the 2005 review explicitly cites Cos-Aguilera, Cañamero & Hayes 2003 as one of the *initial solutions* to grounding pain / pleasure signals in homeostatic variation; the present paper is the publication-grade development of that 2003 conference workshop.
- **Pleasure-as-decoupled-from-need successor.** `lewis_canamero_2016_hedonic_pleasure.md` — Lewis & Cañamero 2016 take issue with the equation "physiology compensation → hormone → reinforcement" used here and propose a more refined hedonic structure that allows pleasure to occur when the agent is *not* in deficit.
- **Lifetime-adaptation successor.** `lones_canamero_2013_epigenetic_hormones.md` and `lones_2018_hormone_epigenetic.md` — keep the hormone-as-modulator idea but add a much slower, lifetime-scale plasticity layer in which hormones alter *parameters* of the homeostatic + drive system itself, not just synaptic weights.
- **Cross-corpus links** (likely in other batches). The three-factor Hebbian / neuromodulated-RL formulation in Eq. 9 is the bridge to broader literature: Doya 2002 (metalearning and neuromodulation), Avery & Krichmar 2017 (models of neuromodulation), Cox & Krichmar 2009 (neuromodulation as a robot controller), and the modern descendants Wang et al. 2024 (Neuromodulated Meta-Learning), Lee et al. 2024 (Lifelong RL via neuromodulation). The curator should cross-link.
