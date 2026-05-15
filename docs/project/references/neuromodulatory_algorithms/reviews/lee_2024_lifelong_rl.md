---
title: "Lifelong Reinforcement Learning via Neuromodulation"
authors: "Sebastian Lee, Samuel Liebana, Claudia Clopath, Will Dabney"
year: 2024
venue: "arXiv:2408.08446 (Imperial College London / Oxford / Google DeepMind)"
slug: "lee_2024_lifelong_rl"
source_pdf: "sources/Lee et al. 2024 - Lifelong reinforcement learning via neuromodulation.pdf"
topic: "neuromodulatory_algorithms"
---

## Plain-English entry point

This paper sits in a different niche from most neuromodulation-in-DL work. The dominant pattern in this corpus is to *engineer* a network whose internal activations or weights are modulated by a learned signal (Vecoven 2020, Ben-Iwhiwhu 2022, Mei 2022, Wang 2024). Lee et al. instead ask: can we use established *neuroscience-of-neuromodulation* theory directly to choose the **hyperparameters** of a reinforcement-learning agent on the fly? Specifically, they pick three classical RL hyperparameters — the learning rate $\alpha$, the inverse softmax temperature $\beta$ (i.e., the exploration–exploitation knob), and the discount factor $\gamma$ — and ask: in animal brains, which neuromodulators are believed to set these knobs, what do those neuromodulators encode, and can we measure analogous quantities in a deep RL agent and use them to set $\alpha, \beta, \gamma$ dynamically?

Why it matters: most RL algorithms use *fixed* hyperparameters tuned by grid search, which breaks when the environment is non-stationary (e.g., reward distributions change unpredictably across "blocks" of experience). The paper proposes a **four-component framework** (Figure 1) for mapping neuromodulators to hyperparameters: (I) hypothesise a neuromodulator-to-hyperparameter link; (II) identify what that neuromodulator signals in the brain; (III) measure an analogous quantity in the agent–environment loop; (IV) wire that measurement into a functional form for the hyperparameter. They instantiate the framework on **acetylcholine (ACh) ↔ learning rate** (Marshall et al.: ACh signals expected uncertainty, so the learning rate gets the *fraction of uncertainty that is aleatoric*) and **noradrenaline (NA) ↔ inverse temperature** (Dayan & Yu: NA signals unexpected uncertainty, so $\beta$ becomes the inverse of average epistemic uncertainty over actions). They call this the **Doya-DaYu agent** (named after Doya 2002 + Dayan & Yu). They evaluate it on a non-stationary multi-armed bandit and on a proposed mouse-physiology experiment that would close the loop back to neuroscience.

## Section-by-section backbone

### Abstract
Adaptation across tasks (continual, meta, lifelong, multi-task RL) needs some adaptation mechanism. Animal neuromodulation provides one. The paper introduces a framework to integrate neuromodulation theory into adaptive RL, instantiates it on ACh and NA, validates on a non-stationary multi-armed bandit, and proposes a closed-loop neuroscience experiment to test the framework's predictions.

### 1. Introduction
Modern RL relies on heuristics and per-domain hyperparameter tuning ($\epsilon$-greedy with linear decay; grid-searched $\alpha$, $\gamma$). Meta-RL approaches (MAML, meta-gradient, MAESN) bring their own hyperparameters and high-variance gradient estimates. The authors propose a *neuroscience-grounded* alternative: pick adaptive hyperparameters from theories of biological neuromodulation.

### 2. Background
RL formalism: MDP $\langle S, A, P, R, \gamma, p_0 \rangle$; value function $V^\pi(s) = \mathbb{E}[\sum_t \gamma^t r_t \mid s_0 = s]$; optimal policy $\pi^*$. Non-stationarity is modelled as a POMDP with hidden contexts. Neuromodulation is described as a "functionally diverse set of systems" that exert global control of neural circuits.

### 3. Neuromodulation for adaptive RL — the four-component framework

**Component I — What do neuromodulatory systems do?** Recapitulate Doya 2002's mappings:
- **Dopamine (DA) ↔ TD-error / RPE** (reward prediction error)
- **Noradrenaline (NA) ↔ inverse temperature $\beta$** (exploration–exploitation via softmax)
- **Acetylcholine (ACh) ↔ learning rate $\alpha$**
- **Serotonin (5-HT) ↔ discount factor $\gamma$**

**Component II — What do neuromodulatory systems signal?**
- **DA & RPEs**: dopamine encodes $\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$.
- **ACh & NA & Uncertainties**: Yu & Dayan propose ACh signals *expected* uncertainty, NA signals *unexpected* uncertainty. Expected uncertainty = noise within a known context; unexpected uncertainty = surplus due to context change. Empirical support: Lawson et al. 2021 (NA blockade hampers learning under unexpectedness); Iglesias et al. 2013 (basal forebrain cholinergic activity ∝ uncertainty); Marshall et al. 2016 (ACh balances within- vs. between-context uncertainty attribution).
- **5-HT**: implicated in stress / anxiety / motivation; no clean normative theory of control.

**Component III — Measuring analogous signals in agent–environment interaction.** Expected/unexpected uncertainty translate to *aleatoric* (data noise) / *epistemic* (model parameter uncertainty). From Clements et al. 2019:

$$E(s, a) = \mathbb{E}_{i \sim \text{Unif}(1, N)}\!\left[\,\mathbb{V}_{\theta \sim P(\theta \mid \mathcal{D})}(y_i(\theta; s, a))\,\right] \tag{2}$$

$$A(s, a) = \mathbb{V}_{i \sim \text{Unif}(1, N)}\!\left[\,\mathbb{E}_{\theta \sim P(\theta \mid \mathcal{D})}(y_i(\theta; s, a))\,\right] \tag{3}$$

where $E$ is *epistemic* uncertainty (variance over model parameters, averaged across quantiles), $A$ is *aleatoric* uncertainty (variance over return quantiles, with parameters integrated out), $y_i$ is the $i$-th return-distribution quantile estimate. Practical estimation via ensembles (Osband et al. 2016).

**Component IV — Functional forms back to hyperparameters.**

$$\alpha(s, a) = \frac{E(s, a)}{E(s, a) + A(s, a)}, \qquad \beta(s) = \frac{1}{\langle E(s, \hat a)\rangle_{\hat a}} \tag{4}$$

Higher epistemic uncertainty (relative to aleatoric) → higher learning rate (because the model itself is what's wrong); lower mean epistemic uncertainty over actions → higher $\beta$ → more exploitation (because the agent is more confident).

### 4. The Doya-DaYu agent
Concrete instantiation of the framework. Mappings:
- $\alpha \leftrightarrow$ ACh $\leftrightarrow$ Uncertainty balance (Marshall 2016)
- $\beta^{-1} \leftrightarrow$ NA $\leftrightarrow$ Unexpected uncertainty (Dayan & Yu 2002)
- DA ↔ RPE is implicit in the TD update.

For aleatoric (expected) uncertainty in the tabular case they use the *reliability index* (White 1988):

$$\text{Var}(G(a)) \leftarrow \text{Var}(G(a)) + \alpha_G\,[\delta^2 - \text{Var}(G(a))] \tag{5}$$

For epistemic (unexpected) uncertainty, train an ensemble; use the across-ensemble variance of mean values.

**Multi-armed bandit experiment.** Non-stationary $k$-armed bandit ($k=5$, $N$ contexts of length $M$, switch prob $p=0.4$). Compare Doya-DaYu vs. Discounted-UCB and Boltzmann. Results (Figure 3): Doya-DaYu's regret per context starts higher (it's exploring after each switch) but ends lower; the learning rate and temperature visibly spike at context switches and decay; cumulative regret is lower; oracle variants (where the true aleatoric and epistemic uncertainties are provided) do strictly better, validating the design.

### 5. Feedback to neuroscience experiments
The paper proposes the framework can run in reverse: take an artificial model of learning, treat it as a theory of the brain, design experiments on (Task, Agent-parameters, Measurements) axes. Two branches:
- **Exploratory**: from biology to ML.
- **Confirmatory**: from ML model to brain prediction to experiment.

**Concrete confirmatory proposal (Figure 5)**: a two-armed bandit task for mice (lick ports with stochastic reward, contingencies reverse when proficient). Optogenetically stimulate NA neurons in locus coeruleus (LC) at the "go" cue using both excitatory and inhibitory opsins (Carter et al. 2010). GRAB-NA sensors (Feng et al. 2019) confirm LC effect at VTA targets. Striatal electrophysiology (neuropixel) records value-related activity. Hypothesis: phasic NA stimulation should systematically modulate choice entropy (lower phasic NA → more random behaviour). The protocol counterbalances stimulation across reversal types so stimulation does not corrupt learning.

### 6. Discussion
Compares the framework to other adaptive-hyperparameter approaches (meta-RL, intrinsic motivation, etc.) and argues none has yet posed an abstract framework for using neuromodulatory mechanisms to design RL agents. Notes the framework is in-principle extensible beyond tabular (Appendix G sketches a deep RL extension with a navigational environment).

## Phase 1 — Undergraduate-level synthesis

**The key idea.** Imagine you're tuning a reinforcement-learning agent. You have three knobs: how fast it updates its value estimates (learning rate $\alpha$), how much it explores vs. exploits (inverse softmax temperature $\beta$), and how much it discounts future rewards ($\gamma$). Most RL papers tune these by grid search and freeze them. But what if the environment keeps changing — every few hundred steps the reward arms get re-shuffled? Then you'd want $\alpha$ to *spike* right after a change (so the agent learns the new contingencies fast) and decay once it's relearned them. You'd want $\beta$ to *drop* right after a change (so the agent explores), then rise (so it exploits the new optimum).

The authors notice that the brain *already has* a mechanism for this: chemical signals called neuromodulators that modulate neural circuits on the fly. There's an established hypothesis (Doya 2002) that acetylcholine acts like a learning rate, noradrenaline like an inverse exploration temperature, and dopamine like a TD error. And there's a separate hypothesis (Yu & Dayan 2002, Marshall 2016) that acetylcholine actually signals *expected* uncertainty (variance you already know about given the current context) while noradrenaline signals *unexpected* uncertainty (extra variance suggesting the context has changed).

The paper combines these two threads: it (a) measures aleatoric (≈ expected) and epistemic (≈ unexpected) uncertainty in the agent using an ensemble; (b) sets $\alpha = E / (E + A)$ — when the model's own uncertainty dominates, learn fast — and $\beta^{-1} = \langle E\rangle$ — when the model is uncertain on average, explore. They call this the **Doya-DaYu agent**.

**The experimental setup.** A 5-armed bandit where the reward distributions reshuffle randomly (with prob 0.4 every 500 steps). Compared against Discounted-UCB and Boltzmann.

**The result.** Doya-DaYu loses some reward right after each context switch (it's busy figuring out what changed), but its cumulative regret across many switches is lower than the baselines — its adaptive $\alpha$ and $\beta$ track the context shifts. Oracle versions (where the true uncertainties are provided directly) do even better, validating the design rationale.

**The second result.** They propose a mouse experiment: stimulate noradrenergic neurons in locus coeruleus with light during a similar two-armed reversal-learning task and check whether the *choice entropy* changes the way Doya-DaYu predicts.

## Phase 2 — Graduate-level deep dive

### The four-component framework formally

Let $\mathcal{N}$ be a neuromodulator. The framework's mapping is:

$$\mathcal{N} \xrightarrow{\text{(I)}} h \xrightarrow{\text{(II)}} q_{\text{bio}} \xrightarrow{\text{(III)}} q_{\text{art}} \xrightarrow{\text{(IV)}} f_h(q_{\text{art}})$$

where $h$ is an RL hyperparameter, $q_{\text{bio}}$ is the biological signal $\mathcal{N}$ encodes, $q_{\text{art}}$ is its agent-side analogue, and $f_h$ is a clipped/normalised function returning a valid hyperparameter value.

### The Doya 2002 mapping with annotations

Equation (1) of the paper (annotated Q-learning update + softmax policy):

$$Q(s, a) \leftarrow Q(s, a) + \underbrace{\alpha}_{\text{ACh}} \big[\, r + \underbrace{\gamma}_{5\text{-HT}} \max_{a'} Q(s', a') - Q(s, a)\,\big]_{\text{DA ↔ RPE}}$$

$$P(a_i \mid s) = \frac{\exp(\underbrace{\beta}_{\text{NA}}\, Q(s, a_i))}{\sum_{j=1}^{|A|} \exp(\beta\, Q(s, a_j))}$$

### Aleatoric and epistemic uncertainty

From Clements et al. 2019, given a model $f_\theta$ with posterior $P(\theta \mid \mathcal{D})$ and quantile estimates $\{y_i(\theta; s, a)\}_{i=1}^N$ of the return distribution:

$$E(s, a) = \mathbb{E}_{i \sim \text{Unif}(1, N)}\big[\, \text{Var}_{\theta \sim P(\theta \mid \mathcal{D})}(y_i(\theta; s, a))\,\big]$$

$$A(s, a) = \text{Var}_{i \sim \text{Unif}(1, N)}\big[\, \mathbb{E}_{\theta \sim P(\theta \mid \mathcal{D})}(y_i(\theta; s, a))\,\big]$$

Read these carefully:
- $E$: for each *quantile* $i$, compute the variance of $y_i$ *over parameter posterior* (model uncertainty about that quantile), then average over quantiles. This is **epistemic**: it shrinks as data accumulates.
- $A$: for each *parameter sample*, compute the mean of $y_i$ over quantiles (the predicted return for that parameter), then take the variance of this mean *across quantiles*. This is **aleatoric**: it captures intrinsic return spread.

### The hyperparameter functional forms

$$\alpha(s, a) = \frac{E(s, a)}{E(s, a) + A(s, a)}$$

Interpretation: $\alpha$ is the fraction of total uncertainty that is *epistemic*. When the model is mostly uncertain about *its own parameters* (rather than about intrinsic noise), the gradient signal is informative — learn fast. When uncertainty is mostly aleatoric, weight updates are noisy — learn slowly. This is structurally identical to a Kalman gain (the paper notes this in footnote 1).

$$\beta(s) = \frac{1}{\langle E(s, \hat a)\rangle_{\hat a}}$$

Interpretation: average epistemic uncertainty across actions, then take its reciprocal. When epistemic uncertainty is high on average, $\beta$ is low → softmax is flatter → more exploration. Note the use of **tonic** NA mapping here: the original Doya 2002 / Aston-Jones & Cohen 2005 phasic-NA story would invert this; the authors explicitly cite Berridge & Waterhouse 2003 and Tervo et al. 2014 for the tonic-mode argument.

### Tabular aleatoric estimator — White 1988

$$\text{Var}(G(a)) \leftarrow \text{Var}(G(a)) + \alpha_G\,[\delta^2 - \text{Var}(G(a))]$$

This is an online second-moment estimator (analogue of the standard incremental-mean estimator but for variance), with $\delta = r - \hat r(a)$ the immediate reward prediction error and $\alpha_G$ a separate (constant) variance-estimator learning rate. Under stationary conditions, $\text{Var}(G(a)) \to$ true aleatoric variance — this is Sakaguchi & Takano's *reliability index*.

### Epistemic estimator via ensemble

For each action $a$, train $K$ heads $\{Q_k(s, a)\}_{k=1}^K$ (or $K$ independent agents). The epistemic uncertainty is then approximated as

$$\hat E(s, a) = \text{Var}_k\big[\mathbb{E}[Q_k(s, a)]\big]$$

i.e., the spread of ensemble mean predictions.

### Oracle uncertainty (Lahlou et al. 2021)

For benchmarking, the paper computes an *oracle* epistemic uncertainty by subtracting the (known) aleatoric uncertainty from the total error magnitude:

$$E^*(s, a) = U(s, a) - A^*(s, a) \tag{6}$$

where $U$ is total uncertainty (estimated from a large sample), $A^*$ is the ground-truth aleatoric variance (known by construction in the bandit). The "full oracle" agent uses $A^*$ and $E^*$ in Eq. 4 directly; the "aleatoric oracle" uses $A^*$ but estimates $E$ from ensembles.

### Bandit-task results

Setup: $k=5$, switch prob $p=0.4$ at the end of each $M=500$-step block, arm means $\sim U[-5, 5]$, arm SDs $\sim U[0.001, 2]$. 210 random seeds. Comparators: Discounted-UCB (Garivier & Moulines 2008) and a fixed-temperature Boltzmann.

Reported findings (Figure 3):
- Doya-DaYu's *per-context* average regret is initially higher than baselines (it explores after the switch) but lower at the end of each block.
- Learning rate and temperature visibly spike at switch points and then decay, recapitulating the predicted behaviour.
- Cumulative regret is lower than baselines, especially when more switches accumulate.
- Final-context regret is significantly lower than baselines (one-way ANOVA + Tukey HSD; $p < 0.001$ vs. Boltzmann; $p < 0.01$ vs. D-UCB).

### Confirmatory neuroscience proposal

Two-armed bandit reversal-learning task for mice with: (1) optogenetic phasic NA-neuron stimulation in LC at the "go" cue using excitatory/inhibitory opsins; (2) GRAB-NA fiber photometry recordings in VTA to confirm physiological-level NA changes; (3) optionally, neuropixel recordings in striatum for action-value circuits. Stimulation protocol spans pre-reversal-proficient through post-reversal-proficient epochs, counterbalanced across reversal directions. Hypothesis: phasic NA stimulation should modulate choice entropy as predicted by Doya-DaYu (low phasic NA → higher choice entropy).

### Strengths and limitations

Strengths:
- The framework is *abstract* and *replaceable*: components (I)–(IV) can be swapped (e.g., replace Yu & Dayan's ACh/NA theory with Marshall 2016's, or replace the Clements et al. uncertainty estimator with a Bayesian deep ensemble).
- The cross-validation with oracle agents establishes that the gap between Doya-DaYu and ideal lies in uncertainty *estimation*, not in the functional forms.
- The confirmatory neuroscience experiment is concrete, falsifiable, and budget-realistic.

Limitations:
- Tabular bandit only; the deep-RL extension is in the appendix and not empirically validated.
- The serotonin mapping is dropped because no convergent normative theory exists. So the "framework" actually only instantiates 2 of Doya's 4 mappings (plus the implicit DA-RPE).
- The functional forms in Eq. 4 are first-order ("as simple as possible") — more sophisticated forms (e.g., bounded, with hyperparameter floors) are flagged as future work.

## Connections

- **Doya 2002** (Doya K., "Metalearning and neuromodulation") — the foundational reference; the entire Component I draws from it. Doya 2002 is also in this corpus.
- **Yu & Dayan 2005**, **Dayan & Yu 2002**, **Dayan & Yu 2006** — the ACh = expected uncertainty / NA = unexpected uncertainty theory. Foundational for Components II–IV.
- **Marshall et al. 2016** — alternative ACh theory (balance of within- vs. between-context uncertainty); the basis for the learning-rate functional form.
- **[vecoven_2020_neuromod_dnn](vecoven_2020_neuromod_dnn.md)** — addresses the same lifelong-RL adaptation problem but through a *modulated activation function* rather than modulated hyperparameters. Complementary.
- **[ben-iwhiwhu_2022_context_meta_rl](beniwhiwhu_2022_context_meta_rl.md)** — addresses the same problem through *modulator subnetworks per layer*. Complementary.
- **[mei_2022_multiscale_neuromod](mei_2022_multiscale_neuromod.md)** — Lee 2024 instantiates Mei's "Scale 1: hyperparameter reconfiguration" slice.
- **[wang_2024_neuromod_meta](wang_2024_neuromod_meta.md)** — fellow 2024 neuromodulated-RL paper; very different mechanism (structural mask vs. hyperparameter modulation).
- **Avery & Krichmar 2017** — cited as ref for neuromodulatory systems review; in this corpus.
- **Dabney et al. 2020** — distributional code for value in DA-based RL; co-author Dabney is on this paper. Methodologically related (quantile-based return distribution).
- **Aston-Jones & Cohen 2005** — adaptive gain theory of LC; the canonical reference for NA's exploration role.
- **Berridge & Waterhouse 2003**, **Tervo et al. 2014** — tonic-NA references; the basis for the inverted $\beta^{-1}$ mapping.
- **[ferguson_cardin_2020_gain_modulation](ferguson_cardin_2020_gain_modulation.md)** — cortical gain modulation review; provides biological substrate for NA-driven gain changes.
