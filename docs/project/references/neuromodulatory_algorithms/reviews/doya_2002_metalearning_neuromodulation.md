---
title: "Metalearning and neuromodulation"
authors: ["Kenji Doya"]
year: 2002
venue: "Neural Networks 15:495–506 (Special Issue on Computational Models of Neuromodulation)"
slug: doya_2002_metalearning_neuromodulation
source_pdf: "sources/Doya 2002 - Metalearning and neuromodulation.pdf"
topic: neuromodulatory_algorithms
---

## Plain-English entry point

This is the **foundational paper for the entire "neuromodulators as RL meta-parameters" research program** that runs through every other paper in this corpus. The question Doya asks: animals can learn flexibly across wildly different environments, while artificial reinforcement learning (RL) systems require humans to hand-tune knobs like the learning rate, the exploration noise, and the reward-discount factor for every new task. The brain must have a built-in way of *automatically* adjusting these knobs — a *metalearning* mechanism. Doya proposes that the **four major ascending neuromodulatory systems** — broadcast neurochemicals that project diffusely from brainstem nuclei to large cortical territories — are exactly that mechanism. Each one carries one of the four RL meta-parameters.

The mapping, in one table:

| Neuromodulator | Origin | RL role |
|---|---|---|
| **Dopamine (DA)** | Substantia nigra compacta + VTA | **TD error** $\delta(t)$ — the reward prediction error |
| **Serotonin (5-HT)** | Dorsal raphe | **Discount factor** $\gamma$ — how far into the future to predict reward |
| **Noradrenaline (NA / NE)** | Locus coeruleus | **Inverse temperature** $\beta$ — randomness vs. determinism in action selection |
| **Acetylcholine (ACh)** | Basal forebrain (nucleus basalis of Meynert) | **Learning rate** $\alpha$ — speed of memory update |

Doya develops the hypothesis in three layers. First, he reviews the textbook actor-critic RL algorithm, naming the four global metaparameters $\alpha, \beta, \gamma$, and the TD error $\delta$. Second, he goes through each neuromodulator one at a time, summarizing the experimental data (Schultz's monkey dopamine recordings, depletion / SSRI studies for serotonin, Aston-Jones LC recordings for noradrenaline, Hasselmo's hippocampal acetylcholine work) and showing that each modulator's behavioral signature lines up with its proposed RL role. Third, he predicts **how the four modulators should interact** — facilitatory and inhibitory cross-effects derived from the algebra of the RL update equations (Fig. 9 in the paper).

The headline impact: every subsequent paper in this corpus that frames a neuromodulator as a tunable scalar inside a learning algorithm — Avery & Krichmar 2017; Krichmar 2013; Xing 2020 (5-HT for patience / discounting); Zou 2020 (ACh+NE for attention); Xing 2022 (ACh+NE for lifelong RL); Vecoven 2020; Ben-Iwhiwhu 2022; Lee 2024; Wang 2024 — descends from this paper's four-modulator framework.

## Section-ordered backbone

### Abstract
Proposes a computational theory of the four major ascending neuromodulators as carriers of global RL meta-parameters: DA = TD error, 5-HT = time scale of reward prediction (discount factor), NA = randomness in action selection, ACh = learning rate.

### 1. Introduction
- Neuromodulators (DA, 5-HT, NA, ACh) historically associated with general arousal; recent molecular advances enable a more specific account.
- Many RL applications require humans to tune meta-parameters (learning speed, exploration noise, discount horizon). These are called *meta-parameters* or *hyperparameters*. Statistical learning theory has Bayesian / risk-minimization stories for some of them, but RL practice still relies on heuristic tuning.
- Brains are robust and flexible across environments → the brain must have a built-in metalearning mechanism that adjusts meta-parameters online.
- Hypothesis: the ascending neuromodulatory systems (Fig. 1) are *the* media for metalearning, coordinating distributed learning modules in the brain.

### 2. Reinforcement learning algorithm
Standard Markov Decision Process $(S, A, T, R)$, actor-critic architecture (Fig. 2). The critic learns the state-value function $V(s)$:
$$
V(s(t)) = \mathbb{E}[r(t+1) + \gamma\,r(t+2) + \gamma^2\,r(t+3) + \dots], \tag{1}
$$
with discount factor $\gamma \in [0, 1]$. Bellman consistency $V(s(t-1)) = \mathbb{E}[r(t) + \gamma V(s(t))]$ (Eq. 2). Define the **TD error**:
$$
\delta(t) = r(t) + \gamma V(s(t)) - V(s(t-1)). \tag{3}
$$
The critic uses $\delta$ as its update: $\Delta V(s(t-1)) \propto \delta(t)$ (Eq. 4).

**2.2 Action value function and policy.** Action values $Q(s, a) = \mathbb{E}[r(t+1) + \gamma V(s(t+1)) | a(t) = a]$ (Eq. 7), updated by $\Delta Q(s(t-1), a(t-1)) \propto \delta(t)$ (Eq. 8). Policy follows Boltzmann selection:
$$
P(a_i | s) = \frac{\exp(\beta Q(s, a_i))}{\sum_{j=1}^{m} \exp(\beta Q(s, a_j))}, \tag{5}
$$
with **inverse temperature** $\beta$. $\beta = 0 \Rightarrow$ random; $\beta \to \infty \Rightarrow$ argmax (Eq. 6).

**2.3 Global learning signal and metaparameters.** Value and action-value functions parametrized by linear basis expansions $V(s) = \sum_j v_j b_j(s)$, $Q(s, a) = \sum_k w_k c_k(s, a)$ (Eqs. 9, 10). Updates:
$$
\Delta v_j = \alpha\,\delta(t)\,b_j(s(t-1)), \qquad \Delta w_k = \alpha\,\delta(t)\,c_k(s(t-1), a(t-1)). \tag{11, 12}
$$
The TD error $\delta$ is the **global learning signal**; the **meta-parameters** are $\alpha$ (learning rate), $\beta$ (inverse temperature), $\gamma$ (discount factor).

### 3. Hypothetical roles of neuromodulators
The central hypotheses, restated:
1. **DA signals the TD error $\delta$.**
2. **5-HT controls the discount factor $\gamma$.**
3. **NA controls the inverse temperature $\beta$.**
4. **ACh controls the learning rate $\alpha$.**

**3.1 Dopamine and reward prediction.** Schultz et al. (1997) monkey recordings: DA neurons fire on unexpected reward early in learning; after learning, they fire to the predictive cue and not to the reward; on omitted reward, DA dips below baseline (Fig. 4). All three signatures are exactly those of the TD error $\delta(t)$. DA also reinforces actions (electrical stimulation = reinforcement; addictive drugs increase DA). At the cellular level, DA modulates striatal cortico-striatal synaptic plasticity, reversing the direction of Hebbian plasticity depending on DA level (Reynolds & Wickens 2001). The basal-ganglia model (Fig. 5; Houk et al. 1995; Montague et al. 1996): striatum patch compartment encodes $V(s)$, striatum matrix encodes $Q(s, a)$; SNc DA neurons compute $\delta$ from striatal projections; DA modulates cortico-striatal plasticity to implement Eqs. 11, 12. Fig. 6 shows two mechanisms for TD-error computation in the BG circuit: (a) slow GABA-B-mediated direct path provides $V(s(t-1))$, fast indirect path provides $V(s(t))$; (b) alternative form $\delta(t) = r(t) - (1-\gamma)V(s(t)) + (V(s(t)) - V(s(t-1)))$ (Eq. 13).

**3.2 Serotonin and time scale of reward prediction.** The discount factor $\gamma$ matters most when immediate and long-term outcomes conflict (Fig. 7). Higher $\gamma$ → harder to predict but more far-sighted; lower $\gamma$ → easier but myopic. 5-HT data:
- 5-HT depletion → impulsive choice of small immediate over larger delayed reward in rats (Mobini et al. 2000; Rahman et al. 2001).
- SSRIs effective against depression (Wang & Licinio 2001). Depression is plausibly modeled as low $\gamma$ → all actions look unrewarding → "do nothing" optimal.
- 5-HT 2A receptor facilitates working memory in PFC (Williams et al. 2002).
- **Mechanistic prediction**: in Fig. 6, the *balance of direct vs. indirect pathways* through the BG sets the effective discount factor. 5-HT modulates this balance via projections from dorsal raphe to striatum and SNc/VTA.
- Alternative hypothesis (Daw, Kakade, Dayan 2002): 5-HT represents the predicted average reward $\bar r$ in the average-reward TD form $\delta(t) = r(t) - \bar r + V(s(t)) - V(s(t-1))$ (Eq. 14).

**3.3 Noradrenaline and randomness of action selection.** Higher $\beta$ → sharper Boltzmann softmax → more deterministic (Fig. 8). The exploration-exploitation trade-off requires online tuning of $\beta$. NA / LC data:
- LC activated in urgent / aversive situations (arousal).
- Aston-Jones et al. (1994): phasic LC response correlates with correct response in attentional tasks.
- LC modulation sharpens neural tuning by increasing input-output gain (Gilzenrat et al. 2002; Servan-Schreiber et al. 1990; Usher et al. 1999).
- Amphetamine (raises NA) → stereotyped behavior (over-exploitation).
- **Mechanistic prediction**: GP neurons have high spontaneous firing whose inhibitory dynamics may implement a stochastic roulette wheel for action selection. NA at GP would modulate this stochasticity.

**3.4 Acetylcholine and memory update.** Learning rate $\alpha$ must be initially large then decay for fast accurate learning (Murata et al. 2002; Sutton 1992 delta-bar-delta). ACh data:
- ACh modulates synaptic plasticity in hippocampus, cortex, striatum.
- Loss of cholinergic neurons in nucleus basalis of Meynert → memory disorders (Alzheimer's).
- Hasselmo's hippocampal model: high ACh → memory *storage* mode; low ACh → memory *retrieval* mode (Hasselmo & Bower 1993; Hasselmo & Schnell 1994).
- Dayan & Yu (2002) — ACh controls top-down vs bottom-up flow based on prediction-mismatch.
- Striatal cholinergic interneurons respond to potentially rewarding cues (Aosaki et al. 1994; Shimo & Hikosaka 2001) and modulate DA-dependent cortico-striatal plasticity (Partridge et al. 2002).

### 4. Dynamic interactions of neuromodulators
Fig. 9 lists predicted interactions a through g, derived from the algebra of the RL equations:

**4.1 (Arrow a)** $\delta$ depends on $\gamma$ via Eq. 3: 5-HT should *facilitate* DA when $V(s(t)) > 0$ and *inhibit* DA when $V(s(t)) < 0$.

**4.2 (Arrows b, c, d)** Large $\gamma$ → high TD-error variance → 5-HT should decrease in response to high DA variability (b). When learning long-horizon outcomes, agent shouldn't commit too quickly → high 5-HT should inhibit NA (c) and ACh (d).

**4.3 (Arrows e, f)** When agent is near-optimal (high or very-low $V$), $\beta$ should increase → NA depends on $V$ (e). State-dependent control: high $\text{Var}_a Q(s, a)$ → reduce NA (f), so that variability remains.

**4.4 (Arrow g)** Frequent oscillation in $\delta$ (sign-flipping) → learning rate too large → DA variability inhibits ACh (g) — analogous to Sutton's delta-bar-delta rule.

### 5. Conclusion
A unified theory of neuromodulators as RL meta-parameter carriers. The same global modulator can have different effects in different brain regions, depending on local receptor types — explaining the diversity of modulator receptors (Marder & Thirumalai 2002). Future work: amygdala, hippocampus, sensory neuromodulation. The framework gives a theoretical basis for understanding emotion, designing therapies for psychiatric disorders, and building "human-like" artificial agents.

## Phase 1 — Undergraduate-level synthesis

**Key idea.** Reinforcement learning has four control knobs that have to be set right for a learning system to work:
- $\alpha$ — how quickly the system updates its beliefs (learning rate).
- $\beta$ — how strictly it picks the best action vs. exploring random alternatives.
- $\gamma$ — how far into the future it cares about reward.
- $\delta$ — the running "I got more (or less) than I expected" signal that drives the actual learning.

In an artificial RL agent, a human programmer hand-tunes $\alpha, \beta, \gamma$. The brain has to tune them automatically and in real time — otherwise an animal in a dangerous situation would use the same learning rate as an animal in a safe one, which is biologically suicidal. Doya's claim: the brain uses **four broadcast chemicals**, each of which carries one of the four knobs.

**The four-way mapping.**

- **Dopamine** carries $\delta$ — the reward prediction error. Schultz's classic monkey experiments show DA neurons firing when a monkey gets a *better-than-expected* reward and falling silent when it gets a *worse-than-expected* reward. That is exactly the TD error.
- **Serotonin** carries $\gamma$. Animals depleted of serotonin become impulsive — they choose small immediate rewards over big delayed ones. SSRIs (which raise serotonin) help with depression. Depression looks like an animal stuck with low $\gamma$: nothing in the future is worth waiting for.
- **Noradrenaline** carries $\beta$. The locus coeruleus, the source of brain NA, fires sharply during attention-demanding moments and is associated with arousal. When you're focused, NA is high and your actions are deterministic; when you're relaxed, NA is low and you explore.
- **Acetylcholine** carries $\alpha$. ACh from the basal forebrain controls how aggressively the hippocampus rewrites its synapses. When ACh is high, you're storing new memories; when ACh is low, you're recalling old ones.

**Setup.** This is not an experimental paper; it's a theoretical synthesis. Doya doesn't run animals or simulations of his own — he assembles experimental findings from across two decades of monkey, rat, and human neuroscience, and shows that each one is consistent with the corresponding RL knob.

**Result.** A coherent framework in which:
- Every cellular fact about each neuromodulator maps onto an algorithmic role.
- The interactions between modulators (predicted in Fig. 9) follow algebraically from the RL update equations.
- Pathologies of mood and attention (depression, OCD, ADHD, Alzheimer's) become tunings of the meta-parameter dial.

**Concrete instantiation: a rat at a Y-maze.** It chooses left, gets a treat. DA spikes (positive $\delta$) → the cortico-striatal synapse for "in this state, go left" is strengthened (controlled by $\alpha$ via ACh). Next trial, the rat does it again. DA is now quiet because the reward is now expected. If the experimenter swaps the reward to the right arm, DA dips on the next "go left" → the synapse weakens. If the rat is starving (high cost of being slow), NA is high → it always takes whichever arm currently has the higher Q-value. If the rat just ate (low cost of being slow), NA is low → it explores. If the rat has been doing this maze for months and the maze is familiar (low novelty), ACh is low → the synapses change slowly, preserving long-term knowledge. Doya's four modulators control all four of these dials simultaneously, automatically.

## Phase 2 — Graduate-level deep dive

### The actor-critic algorithm and the four meta-parameters

Standard MDP $(S, A, T, R)$. Policy $\pi(a|s)$ either deterministic ($a = G(s)$) or stochastic. Transition either deterministic ($s_{t+1} = F(s_t, a_t)$) or stochastic Markovian. Reward $r_{t+1} \in \mathbb{R}$.

**State value function** (Eq. 1):
$$
V(s(t)) = \mathbb{E}\!\left[\sum_{k=1}^{\infty} \gamma^{k-1}\, r(t+k)\right], \qquad \gamma \in [0, 1].
$$

**Bellman consistency** (Eq. 2):
$$
V(s(t-1)) = \mathbb{E}\!\left[r(t) + \gamma V(s(t))\right].
$$

**TD error** (Eq. 3):
$$
\delta(t) = r(t) + \gamma V(s(t)) - V(s(t-1)).
$$

**Critic update** (Eq. 4): $\Delta V(s(t-1)) = \alpha\,\delta(t)$.

**Action-value** (Eq. 7): $Q(s(t), a) = \mathbb{E}[r(t+1) + \gamma V(s(t+1)) | a(t) = a]$.

**Policy** (Eq. 5): Boltzmann softmax with inverse temperature $\beta$:
$$
P(a_i | s) = \frac{\exp(\beta\,Q(s, a_i))}{\sum_{j=1}^{m} \exp(\beta\,Q(s, a_j))}.
$$

**Limits.** $\beta = 0 \Rightarrow$ uniform random; $\beta \to \infty \Rightarrow$ argmax (Eq. 6) $a(t) = \arg\max_a Q(s(t), a)$.

**Function approximation** (Eqs. 9, 10): $V(s) = \sum_j v_j b_j(s)$, $Q(s, a) = \sum_k w_k c_k(s, a)$.

**Weight updates** (Eqs. 11, 12):
$$
\Delta v_j = \alpha\,\delta(t)\,b_j(s(t-1)), \qquad \Delta w_k = \alpha\,\delta(t)\,c_k(s(t-1), a(t-1)).
$$

The four meta-parameters: $\alpha$ (learning rate), $\beta$ (inverse temperature), $\gamma$ (discount), $\delta$ (TD error / global learning signal).

### Derivation of the alternative TD error (Eq. 13)

Doya derives an alternative form that supports the basal-ganglia direct-vs-indirect-pathway mechanism. Start from Eq. 3:
$$
\delta(t) = r(t) + \gamma V(s(t)) - V(s(t-1)).
$$
Add and subtract $V(s(t))$:
$$
\delta(t) = r(t) + (\gamma - 1) V(s(t)) + V(s(t)) - V(s(t-1)).
$$
Rewrite $(\gamma - 1) = -(1 - \gamma)$:
$$
\delta(t) = r(t) - (1 - \gamma)\,V(s(t)) + \big(V(s(t)) - V(s(t-1))\big). \tag{13}
$$

Three additive terms:
- $r(t)$ — the immediate reward (excitatory).
- $-(1 - \gamma)V(s(t))$ — an *immediate inhibition* proportional to current state value, scaled by $(1 - \gamma)$ which vanishes as $\gamma \to 1$.
- $V(s(t)) - V(s(t-1))$ — the temporal *change* in value (TD-of-value).

The temporal change term needs a delay; the indirect pathway through GPe-STN can provide that delay. The $(1-\gamma)$ inhibition needs to come from the striatum's direct projection — and 5-HT modulating the strength of the indirect-vs-direct balance is exactly the proposed mechanism for $\gamma$ control.

**Connection to average-reward RL.** Mahadevan (1996) average-reward formulation replaces $(1 - \gamma)V(s(t))$ with the running average reward $\bar r$:
$$
\delta(t) = r(t) - \bar r + V(s(t)) - V(s(t-1)). \tag{14}
$$
Daw, Kakade & Dayan (2002) propose that 5-HT represents $\bar r$ — the chronic *aversive* expected reward against which DA's phasic signal is compared. This is an alternative to Doya's "5-HT = $\gamma$" hypothesis; both are framed in this paper.

### How $\gamma$ shapes the value function — the variance argument

Large $\gamma$ makes the value function more informative (it predicts farther into the future) but also makes it harder to learn reliably: $\text{Var}[V(s)]$ grows roughly as $\sigma_r^2 / (1 - \gamma^2)$ for a stationary reward sequence with per-step variance $\sigma_r^2$. As $\gamma \to 1$, the variance diverges. So:
- Too-small $\gamma$: low variance, high bias (myopic).
- Too-large $\gamma$: high variance, low bias (far-sighted but noisy).

Doya's Fig. 7 scenario: in (a) a small $\gamma$ makes the cumulative future reward $V$ negative for a costly long-run-positive action; the optimal policy rejects it (depression-like). In (b) a large $\gamma$ makes $V$ positive; the optimal policy takes it. This makes 5-HT depletion → low $\gamma$ → impulsive / "depressed" behavior consistent with both the Mobini et al. (2000) data and clinical observations.

**Prediction (Fig. 9b)**: high variability in $\delta$ → inhibitory feedback onto 5-HT (lower $\gamma$ to reduce variance). This is the algorithmic counterpart of a homeostatic loop on the prediction-horizon dial.

### Softmax temperature and the exploration/exploitation trade-off

The Boltzmann policy probability ratio for two actions:
$$
\frac{P(a_1)}{P(a_2)} = \exp\!\left(\beta(Q(s, a_1) - Q(s, a_2))\right).
$$
For a Q-difference $\Delta Q = Q(s, a_1) - Q(s, a_2)$, $P(a_1)$ is a sigmoid in $\beta\,\Delta Q$ (Fig. 8 in the paper plots this for $\beta = 0.1, 1, 10$). As $\beta \to \infty$ the sigmoid becomes a step function — argmax.

Choosing $\beta$ adaptively is the *exploration-exploitation* problem. Standard heuristics:
- **Annealing**: $\beta(t) = \beta_0 \cdot t / t_0$ — increase over time.
- **State-dependent** (Ishii, Yoshida, Yoshimoto 2002): $\beta$ depends on $\text{Var}_a Q(s, a)$ — high uncertainty over Q-values → low $\beta$ → exploration.
- **Performance-dependent**: $\beta$ depends on $V(s)$ — agent in a high-value state or a critically low-value state should commit; ambiguous state should explore (Fig. 9e).

NA in LC fits this dial: phasic LC → high $\beta$ → sharp, deterministic action; tonic LC at moderate level → low $\beta$ → exploration. Aston-Jones & Cohen (2005) adaptive-gain theory of LC is the explicit elaboration of this mapping.

### Learning-rate scheduling and ACh

For SGD-style updates, the right learning rate matters:
- Constant $\alpha$: never converges (Sutton 1988 stochastic approximation conditions $\sum \alpha_t = \infty$, $\sum \alpha_t^2 < \infty$ violated).
- $\alpha_t = c/t$: converges but slow in stationary settings, too slow in non-stationary.
- Delta-bar-delta (Sutton 1992): adapt $\alpha_t$ based on the *sign* of recent gradients — if gradient sign keeps flipping, $\alpha$ is too large; reduce. If sign is consistent, $\alpha$ may be too small; increase.

In Doya's mapping, $\delta(t)$ is the analog of the gradient, and ACh is the analog of $\alpha$. Prediction (Fig. 9g): frequent sign-flips of $\delta$ (DA fluctuations) → inhibit ACh. The Sutton delta-bar-delta rule is reinterpreted as a chemical loop on the basal-forebrain ACh system.

### Modulator interactions (Fig. 9 in the paper)

| Arrow | Effect | Algebraic origin |
|---|---|---|
| a | 5-HT $\to$ DA, sign depends on $V(s(t))$ | Eq. 3: $\delta$ contains $\gamma V(s(t))$; 5-HT scales $\gamma$ |
| b | DA-variability inhibits 5-HT | $\gamma$-large $\Rightarrow$ Var$(\delta)$ large $\Rightarrow$ scale $\gamma$ back |
| c, d | 5-HT inhibits NA and ACh | Long-horizon learning requires small $\beta$, $\alpha$ |
| e | $V(s)$ excites NA | Near-optimal or near-disastrous states $\Rightarrow$ commit |
| f | High $\text{Var}_a Q$ inhibits NA | State-dependent exploration |
| g | DA variability inhibits ACh | Delta-bar-delta on learning rate |

Each of these arrows is a *testable experimental prediction*. The hypotheses are framed so that comparing chronic SSRI animals to chronic LC-lesioned animals to chronic ACh-depleted animals to chronic DA-depleted animals should produce measurable, distinguishable RL signatures. Many of the subsequent neuromodulator-as-meta-parameter papers in this corpus (Belkaid & Krichmar 2020; Krichmar 2013; Avery et al. 2012) are direct empirical or computational tests of these arrows.

### The four-modulator framework in one equation

If we let $\theta = (v, w)$ collect all RL weights, the brain's metalearning loop is:
$$
\theta_{t+1} = \theta_t + \underbrace{\alpha(\text{ACh}_t)}_{\text{learning rate}} \cdot \underbrace{\delta(\text{DA}_t)}_{\text{TD error}} \cdot \phi(s_t, a_t),
$$
with action selected by:
$$
a_t \sim \exp\!\left(\underbrace{\beta(\text{NA}_t)}_{\text{inverse temperature}} \cdot Q_\theta(s_t, \cdot)\right),
$$
and value defined by:
$$
V_\theta(s_t) = \mathbb{E}\!\left[r_{t+1} + \underbrace{\gamma(\text{5-HT}_t)}_{\text{discount}}\, V_\theta(s_{t+1})\right].
$$

All four meta-parameters are now *functions of slow-varying neuromodulator levels*. The neuromodulators themselves are *output variables* of higher-level controllers (Fig. 9 cross-effects) that monitor the RL agent's trajectory and update their own dials accordingly. This is Doya's full proposal: the brain is a four-modulator hierarchical metalearning system.

### Why "metalearning" not "learning"

The four neuromodulators do not themselves *learn the value function* — they *learn how to learn*. The actor-critic loop in cortico-striatal circuits learns $V$ and $Q$. The neuromodulator loop sits on top, monitoring statistics of the actor-critic loop, and adjusts $\alpha, \beta, \gamma$ in response to those statistics. This is a meta-RL view: an outer optimization over hyperparameters running in parallel with an inner optimization over policy weights. The framework anticipates by ~20 years the modern deep-meta-RL literature (Wang, Kurth-Nelson et al. 2018; Duan et al. 2016; Finn et al. 2017).

### Parameter table

| Symbol | Meaning | Modulator | Behavioral signature when raised |
|---|---|---|---|
| $\alpha$ | Learning rate / time-constant | ACh | Faster encoding, more storage, less retrieval |
| $\beta$ | Inverse temperature / exploration | NA | More deterministic, less exploratory |
| $\gamma$ | Discount factor / horizon | 5-HT | More patience, less impulsivity |
| $\delta$ | TD error / reward prediction error | DA | Reinforcement of recent action |

## Connections

**Direct references inside this corpus:**

- **[avery_krichmar_2017_models_neuromodulation](avery_krichmar_2017_models_neuromodulation.md)** — the 15-year-later review that extends Doya's framework with newer experimental data and interactions between neuromodulators.
- **[krichmar_hwu_2022_design_principles_neurorobotics](krichmar_hwu_2022_design_principles_neurorobotics.md)** — cites Doya (2002) explicitly for the four-modulator meta-parameter mapping in §3.2 ("besides the dopaminergic reward system, there are multiple neuromodulators that signal different value types (Doya, 2002; Krichmar, 2008)").
- **[xing_2020_neuromodulated_patience](xing_2020_neuromodulated_patience.md)** — operationalizes the 5-HT = $\gamma$ hypothesis on a robot navigation task. The Miyazaki et al. (2018) mouse experiment underlying Xing 2020 is itself the direct empirical test of Doya's 5-HT proposal.
- **[zou_2020_neuromodulated_attention](zou_2020_neuromodulated_attention.md)** — operationalizes the ACh/NE axis (a refinement of Yu & Dayan 2005, which is itself a descendant of Doya 2002).
- **[xing_2022_neuromodulation_rl_environment_changes](xing_2022_neuromodulation_rl_environment_changes.md)** — operationalizes ACh + NE for lifelong RL, with ACh tracking expected uncertainty and NE driving network reset.
- **[hwu_krichmar_2020_schemas_memory](hwu_krichmar_2020_schemas_memory.md)** — uses a "novelty × familiarity" product as a single neuromodulator (closest analog to NE / ACh in Doya's scheme).
- **[chiba_krichmar_2020_self_monitoring](chiba_krichmar_2020_self_monitoring.md)** — self-monitoring framework with neuromodulators as the action-regulatory layer; direct extension of Doya's value-system view.
- **[shine_2021_computational_models_neuromodulation](shine_2021_computational_models_neuromodulation.md)** — cellular-mechanism account of how neuromodulators implement Doya-style meta-parameter gating.

**Forward citations expected.** *Every* later paper in this corpus that frames a neuromodulator as a tunable scalar inside a learning algorithm descends from Doya 2002:
- Vecoven et al. 2020 (neuromodulation in DNNs)
- Ben-Iwhiwhu et al. 2022 (context meta-RL via neuromodulation)
- Lee et al. 2024 (lifelong RL via neuromodulation)
- Wang et al. 2024 (neuromodulated meta-learning)
- Mei et al. 2022 (multiscale neuromodulatory systems in DNNs)
- Tsuda et al. 2021 (neuromodulators shifting RNN activity hypertubes)
- Wainstein et al. 2025 (gain neuromodulation for perceptual switches)
- Costacurta et al. 2024 (structured flexibility via neuromodulation in RNNs)
- Rodriguez-Garcia et al. 2026 (noradrenergic-inspired gain modulation)
- Tambaş et al. 2025 (neuromodulation in Krotov-Hopfield)
- AlKilany & Goodman 2025 (neuromodulation in spiking nets)
- Durstewitz et al. 2025 (continual learning + neuromodulation)
- Osman et al. 2024 (Hopfield + arousal modulation)
- Kudithipudi et al. 2022 (biological underpinnings of lifelong learning)

The conceptual genealogy is unmistakable.

**External anchors cited in Doya 2002.** Schultz, Dayan & Montague 1997 (DA = TD error); Montague, Dayan & Sejnowski 1996 (mesencephalic DA framework); Houk, Adams & Barto 1995 (BG reinforcement learning); Sutton & Barto 1998 (RL textbook); Yu & Dayan 2002 (ACh uncertainty); Daw, Kakade & Dayan 2002 (DA/5-HT opponency, average-reward 5-HT); Hasselmo & Bower 1993, Hasselmo & Schnell 1994 (ACh storage/retrieval modes); Aston-Jones et al. 1994 (LC attention); Gilzenrat et al. 2002 (NA gain modulation); Ishii et al. 2002 (state-dependent inverse-temperature control); Mobini et al. 2000 (5-HT depletion → impulsivity); Williams et al. 2002 (5-HT 2A in working memory); Reynolds & Wickens 2001 (DA-modulated cortico-striatal plasticity); Marder & Thirumalai 2002 (cellular neuromodulator effects).
