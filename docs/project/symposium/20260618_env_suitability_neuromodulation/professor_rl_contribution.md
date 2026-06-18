---
title: "Professor-RL contribution — are the four neuromodulator→hyperparameter behavioural signatures actually MEASURABLE in the pain grid world, and what is the cleanest RL probe for each?"
symposium_session: 20260618_env_suitability_neuromodulation
role: professor-rl
date: 2026-06-18
builds_on: 00_env_capability_grounding.md
knobs:
  - "NA → effective temperature (exploration / policy entropy)"
  - "ACh → effective learning rate (volatility-driven plasticity)"
  - "DA → effective TD-error gain (reward-variance sensitivity)"
  - "5-HT → effective discount (patience / effective horizon)"
verdicts:
  - "NA → temperature: MEASURABLE-AS-IS (survival mode; config sweep over threat)"
  - "ACh → learning-rate: MEASURABLE-WITH-CAVEAT for the agent-internal probe; NEEDS-ENV-WORK for the behavioural volatility signature"
  - "DA → TD-gain: MEASURABLE-WITH-CONFIG, but only in homeostatic mode and only on the COST side — survival mode cannot express reward-variance"
  - "5-HT → discount: NO env work needed for the value-function probe; behavioural version NEEDS-ENV-WORK (reward-delay queue)"
---

# Professor-RL contribution: measurability of the four behavioural signatures

## §1 — Plain-English entry point

Three other professors are arguing whether our pain grid world can test the four classic "neuromodulator sets a learning knob" ideas from Doya (2002): noradrenaline ↔ how exploratory the agent is, acetylcholine ↔ how fast it relearns when the world changes, dopamine ↔ how sensitive it is to risky-vs-safe payoffs, serotonin ↔ how patient it is. The shared grounding memo ([[00_env_capability_grounding]]) already gave a per-knob "can the environment physically express this?" verdict (one green, two amber, one red). **My job is the next question down: even where the environment can express the manipulation, can we actually MEASURE the behavioural fingerprint with a clean reinforcement-learning probe, and what is that probe?**

That distinction matters because an environment can support a manipulation and still make the read-out unmeasurable or confounded. My headline verdicts, in plain terms:

- **Noradrenaline → exploration (temperature): measurable as-is.** The inverse-U "performance peaks at moderate arousal" curve is cleanly read off **survival steps** (the project's only sanctioned performance metric) plotted against how close the predator gets. No new code.
- **Acetylcholine → learning rate: half measurable, half not.** The agent-internal gradient probe (does the network rotate its updates when the regime changes?) works today. But the *behavioural* "learning rate jumps after a surprise" signature genuinely needs the environment to change its rules at a scheduled moment — which it cannot do yet.
- **Dopamine → risk sensitivity: measurable, but only one way.** Our food reward is a fixed `+1` that cannot carry variance, so the textbook "risky big reward vs. safe small reward" task is impossible in survival mode. The clean version lives in **homeostatic reward mode**, where reward is a continuous body-improvement signal — and even there the variance must be injected through the **damage/cost** side, not the food side. I give the exact probe.
- **Serotonin → patience (discount): you do not need delayed reward to measure it.** The "effective horizon" of the agent's value function can be read directly off the critic at training time. Delayed reward is only needed for the *behavioural* version, and that needs a new reward-delay buffer in the environment, which I argue is worth building but is not on the critical path.

Below, per knob: a verdict tag (**measurable-as-is / measurable-with-config / needs-env-work**), the cleanest probe, which reward mode it needs, and how "performance" is scored. The math for each estimator is in §3. I argue strictly from the grounding memo's facts.

A standing constraint I respect throughout: **performance is survival steps, never cumulative reward.** Cumulative reward is mode-dependent and, in homeostatic mode, is a transformed drive signal that does not compare across configs; survival steps is the invariant the project scores on.

---

## §2 — Per-knob measurability and probes

### 2.1 NA → effective temperature — **MEASURABLE-AS-IS**

**What the signature is.** Two claims: (i) a Yerkes-Dodson inverse-U — task return rises then falls as inferred arousal increases; (ii) policy entropy decreases monotonically in the modulator's effective-temperature gain. Both are standard RL read-outs and both are obtainable with no env change.

**Reward mode: survival mode (`use_homeostatic_reward=False`).** This is the cleaner substrate for the inverse-U, and I disagree with any instinct to reach for homeostatic mode here. The reason is the *shape of the curve we are trying to draw*. Yerkes-Dodson is a statement about **task performance** vs. arousal. In survival mode, performance IS survival steps and the reward (`+1`/eat, `−death_penalty`/death) is a clean, mode-stable proxy whose argmax aligns with survival. In homeostatic mode the per-step reward is a drive-delta whose scale shifts with setpoint and injury weighting, so "return" stops being a stable performance axis and contaminates the y-axis of the very curve we want. Use survival mode; score the y-axis as survival steps directly.

**How performance is measured.** y-axis = **mean survival steps** per episode (the sanctioned metric), evaluated at fixed policy across a sweep. Do NOT use cumulative reward as the performance axis — even in survival mode, an agent that eats aggressively and dies early can out-score on cumulative `+1`s while surviving fewer steps. Survival steps is the inverse-U's correct dependent variable.

**How "inferred arousal" is operationalised.** The env logs `dist_to_pred` (Manhattan distance to the nearest predator; grounding memo §4.1, predator detection at `07_predator_ai.md:180-185`). Define an arousal proxy that is *monotone decreasing in distance and gated on detection*:
$$
a_t \;=\; \mathbb{1}[\text{predator active}]\cdot \exp\!\big(-\,d_t / \lambda\big), \qquad d_t = \texttt{dist\_to\_pred}_t,
$$
with $\lambda$ a length scale on the order of the detection radius. This is a *measured* arousal, not an imposed one (the grounding memo's honest gap, §4.1) — the agent can flee and lower its own $a_t$. That is fine for the inverse-U **provided we sweep the curve across configs**, not within one episode: vary `detection_range` and predator `count`/`damage` across a config grid so the *episode-mean* arousal $\bar a$ tiles the x-axis from "no threat" to "saturating threat." Then plot mean survival steps against $\bar a$. The inverse-U, if present, falls out of the across-config envelope.

**The entropy claim — the cleaner, more direct probe.** The inverse-U is an indirect, emergent read-out. The *direct* test of "the modulator acts as a temperature" is the entropy claim: across the same config grid, regress **per-step policy entropy** $H[\pi(\cdot\mid s_t, c_t)]$ on the modulator's effective-temperature output (or on $a_t$ as its driver). A genuine temperature modulator produces $\partial H / \partial \tau_{\text{eff}} > 0$ monotonically. This is the load-bearing measurement; the inverse-U is corroboration. Both come free from the rollout logger.

**Identifiability caveat (RL-side, not architectural).** Be explicit that "effective temperature" here means *the entropy of the realised action distribution*, which our PPO policy already exposes. If the modulator only shifts the entropy-bonus weight, that is a temperature in the literal Gibbs sense. If instead it reshapes the logits non-uniformly (sharpening some actions, flattening others), the scalar "temperature" summary is lossy — report the full entropy AND the top-1 action probability so a non-uniform reshaping is not silently summarised as a temperature change. This is the cheap insurance against an underspecified temperature claim.

**Verdict: MEASURABLE-AS-IS.** Survival mode; y-axis = survival steps; x-axis = episode-mean arousal proxy from `dist_to_pred` swept across a threat config grid; primary read-out = entropy-vs-gain monotonicity, corroborated by the survival-steps inverse-U.

---

### 2.2 ACh → effective learning rate — **split verdict**

The grounding memo (§4.2) is right that the env's only changing-statistics today is i.i.d. per-episode predator/resource re-sampling, with no scheduled change-point. The two halves of the ACh signature land on opposite sides of that fact.

**Half A — the agent-internal gradient probe: MEASURABLE-WITH-CAVEAT (largely env-independent).** The stratified-direction gradient probe (within-stratum gradient *preserved*, across-stratum gradient *rotated*) is a measurement of the network's update geometry, not of the environment's reward structure. You can run it on whatever data the env produces. With **i.i.d. per-episode resampling** you can still define strata post-hoc — partition collected episodes by their sampled predator parameters (e.g. high-detection vs. low-detection draws) — and ask whether the modulator, conditioned on interoceptive state $c$, rotates the policy-gradient direction across strata while preserving it within. Formally, with per-stratum mean policy gradients $g_k = \mathbb{E}_{s\sim\text{stratum }k}[\nabla_\theta \log\pi(a\mid s,c)\,\hat A]$, the probe measures cross-stratum cosine $\cos(g_j, g_k)$ vs. within-stratum cosine. This is a valid measurement under i.i.d. resampling.

**The caveat is an RL-validity one, and it is important: i.i.d. resampling is not volatility.** A learning-rate *change* is a temporal-difference concept — it is defined by how the agent updates *after a surprise relative to a previously stable expectation*. With i.i.d. draws there is no "previously stable expectation": every episode is an independent sample from a fixed distribution, so the Bayes-optimal learning rate is **constant**, and any LR-like modulation the agent shows is responding to the *within-episode* sampled value, not to *volatility*. So the i.i.d. probe can demonstrate **context-conditioned gradient steering** (the modulator reshapes updates by $c$) but it **cannot demonstrate volatility-driven learning-rate adaptation** in the Behrens/ACh sense, because the environment has no volatility for the agent to track. Calling the i.i.d. result a "volatility" result would be the underspecified-claim trap.

**Half B — the behavioural learning-rate-shift signature: NEEDS-ENV-WORK.** The canonical ACh paradigm requires a **change-point**: a stable block where one contingency holds, then a switch, so you can measure the post-switch update speed (the behavioural learning rate) and show it rises with inferred volatility. The grounding memo §4.2 is correct that neither a within-episode scheduled switch nor a cross-episode block structure exists. **The minimal RL-valid design is a blocked schedule, and it does require new env code** — but it is small. Two options, in increasing fidelity:

1. **Cross-episode block schedule (cheaper).** A scheduler that holds predator parameters *fixed* for a block of $N$ episodes (stable block), then switches to a different fixed setting for the next block, repeating. This makes "volatility" a controllable, low-vs-high block-switch frequency. Measurable signature: post-switch, the agent's behavioural policy converges to the new optimum faster when the modulator detects the regime change. This is a scheduler keyed on an episode counter — modest new code in the reset path.
2. **Within-episode change-point (higher fidelity, more code).** Predator behaviour switches at a scheduled `current_step`, so the LR adaptation happens *within* the agent's recurrent trajectory and can be read against the recurrent hidden state. More faithful to the online-learning framing but touches the step loop.

**Connection to project hypothesis H5 (modulator timescale).** Whichever schedule is built, the block/switch frequency must be **commensurate with the modulator's timescale and the GAE/recurrent horizon**. If the regime switches faster than the GAE credit-assignment window ($\sim 1/(1-\lambda\gamma)$ steps) or faster than the recurrent state's effective memory, the agent cannot in principle track it and the LR signature is unmeasurable for reasons that have nothing to do with the modulator. The block length must exceed the credit-assignment horizon. This is exactly the H5 interaction the project flagged — flag it to `experiment-designer` as a binding constraint on the schedule.

**Reward mode.** Either mode works for the gradient probe. For the behavioural change-point version, **survival mode** keeps the post-switch performance read-out (survival steps) clean, same argument as 2.1.

**Verdict: agent-internal gradient probe = MEASURABLE-WITH-CAVEAT today (but only proves context-conditioned steering, not volatility-tracking); behavioural volatility signature = NEEDS-ENV-WORK (a block scheduler, minimal but real new code), with an H5-driven constraint that block length exceed the credit-assignment horizon.**

---

### 2.3 DA → effective TD-error gain — **MEASURABLE-WITH-CONFIG (homeostatic mode only; cost-side variance)**

This is the knob I most own and the one where the grounding memo's "PARTIAL" needs the most RL-specific sharpening. Three facts decide it:

**Fact 1 — survival mode cannot express reward variance, full stop.** Food reward is fixed `+1`/eat (`06_reward_and_termination.md:28`), and I confirmed `food_nutrition_gain` is a single config scalar (default 6), not a distributional range — the only per-resource stochasticity in the env is on the *olfactory signature* (`res_property_std`), which is a **smell-channel** noise, not a payoff noise (`08_resources_and_obstacles.md:60-61`). So in survival mode every food is worth exactly `+1` and there is **no reward-magnitude variance to be sensitive to.** A risky-vs-safe *food* task is not constructible in survival mode without new code. Rule it out.

**Fact 2 — homeostatic mode CAN express reward variance, on the cost side.** In homeostatic mode the per-step reward is the drive-delta
$$
r_t \;=\; \underbrace{D(\text{sat}_{t-1},\text{inj}_{t-1})}_{\text{prev drive}} \;-\; \underbrace{D(\text{sat}_t,\text{inj}_t)}_{\text{curr drive}}, \qquad D(\text{sat},\text{inj})=\big\|[\,\text{sat}-\text{setpoint},\ \text{inj}\,]\big\|.
$$
A predator/trap contact draws damage uniformly from `[min,max]` (`07_predator_ai.md:275-276`), so injury — and therefore the **injury component of the drive-delta reward** — is genuinely stochastic per contact. A region with `damage:[20,20]` is a deterministic-cost path; a region with `damage:[0,40]` is a *mean-matched, high-variance* cost path. **This is a real reward-variance manipulation, and it is config-only.** This is the substrate for the risky-vs-safe task.

**Fact 3 — and this is the subtle RL point — distributional RL (IQN) measures the return distribution correctly even when the variance lives in the COST.** A quantile critic learns the law of the return $Z^\pi(s) = \sum_t \gamma^t r_t$. The return is a *signed* quantity; a damage event is a negative-reward event; variance in damage is variance in $r_t$ and hence variance in $Z$. The distributional Bellman operator
$$
\mathcal{T}^\pi Z(s) \;\overset{D}{=}\; r(s,a) + \gamma\, Z(s'), \qquad a\sim\pi,\ s'\sim P,
$$
is agnostic to the *sign* of $r$ — it propagates the full law regardless of whether the dispersion enters through positive (food) or negative (damage) reward. So IQN's quantile-emission curves will widen, skew, and shift across the satiation×threat context grid in exactly the way the DA-as-TD-gain story predicts, **with cost-side variance.** There is one genuine asymmetry to respect: cost-side variance produces a **left-skewed (downside) return distribution** (rare large injuries), whereas the textbook risky-reward task produces right skew. A risk-sensitive (CVaR / exponential-utility) modulator will therefore manifest as **avoidance of the high-variance damage region**, i.e. left-tail aversion — which is, if anything, the *more* pain-relevant phenotype for this project. Do not treat the sign asymmetry as a defect; report which tail the modulator reweights.

**Cleanest probe.** A two-region, mean-matched-cost foraging task in **homeostatic mode**:
- Two spatial zones, each with food, reachable by the agent (use `spawn_area` separation, grounding §4.3).
- Zone S (safe): predator/trap `damage:[c,c]` — deterministic cost $c$.
- Zone R (risky): `damage:[c-\delta,\,c+\delta]` — *same mean* $c$, variance $\delta^2/3$.
- Equal expected food access in both zones (match food count/respawn so the mean drive-reduction is matched; only the cost variance differs).

**Read-out (two layers):**
1. **Distributional layer (the direct TD-gain measurement).** Fit/inspect an IQN critic; plot quantile-emission curves $\hat Z_\tau(s,c)$ as a function of the context $c=(\text{satiation, recent injury})$. The DA-as-gain signature is that the **spread/skew of the emitted quantiles shifts with $c$** — e.g. when recently injured, the modulator should up-weight the lower quantiles (heightened downside sensitivity). This is the per-context return-distribution-shape claim, measured correctly.
2. **Behavioural layer (the choice read-out).** **Zone-occupancy fraction and survival steps**, NOT cumulative reward, as functions of $c$. The risk-sensitivity phenotype is: does the agent's safe-vs-risky zone preference shift with interoceptive context (more risk-averse when satiation is high so it has slack; more risk-seeking when starving so it must gamble)? Score it as **survival steps under each zone-preference regime** plus the zone-occupancy fraction. This keeps the sanctioned metric and reads risk preference as a behavioural choice.

**Why homeostatic mode is mandatory here.** Survival mode's `{0,+1}` reward has no continuous magnitude to carry variance; only homeostatic mode's drive-delta does. This is a hard reward-mode requirement for the DA knob, and it should be stated as such to `experiment-designer`.

**Failure mode to flag.** If `food_nutrition_gain` and damage scales are mismatched, the drive-delta reward can be dominated by the hunger term, drowning the injury-variance signal the probe depends on. The two-region task must be tuned so the injury-component variance is a non-trivial fraction of the return variance — otherwise IQN learns a near-deterministic $Z$ and the quantile curves do not move. This is a config-tuning burden, not a code gap.

**Verdict: MEASURABLE-WITH-CONFIG, in homeostatic mode only, with variance injected on the cost side via `damage:[min,max]`. IQN measures the return-distribution shape correctly despite the variance being in the cost; the phenotype is downside/left-tail reweighting, which is the pain-relevant direction. Survival mode cannot test this knob.**

---

### 2.4 5-HT → effective discount — **value-function probe needs NO env work; behavioural probe NEEDS-ENV-WORK**

The grounding memo marks reward-delay NOT SUPPORTED (red). As the RL voice, my contribution is to separate two genuinely different measurements that the symposium is at risk of conflating.

**Is literal reward delay necessary to measure an effective-discount signature? No — not for the value-function version.** The effective discount of an agent is a property of its **critic**, and the critic can be interrogated directly without ever delaying a reward. The proposed test is the slope of $\log|\Delta V|$ against delay $k$. There are two ways to obtain that:

**Probe A — direct critic-horizon probe (NO env work).** Train at a fixed behavioural discount $\gamma_{\text{train}}$. Then measure the modulator's *effective* discount by probing the learned value function's sensitivity to temporally-displaced reward. Concretely, for a state from which a reward is reachable in $k$ steps, the optimal value contribution scales as $\gamma_{\text{eff}}^{k}$, so
$$
\log\big|V(s_k)\big| \;\approx\; \text{const} + k\,\log \gamma_{\text{eff}} \;\;\Longrightarrow\;\; \gamma_{\text{eff}} = \exp\!\Big(\tfrac{\partial \log|V|}{\partial k}\Big).
$$
Estimate $\gamma_{\text{eff}}(c)$ by regressing $\log|V(s)|$ on the (already-logged) **graph distance from $s$ to the nearest food** across many states, stratified by interoceptive context $c$. The 5-HT-as-discount signature is then: **does $\gamma_{\text{eff}}$ shift with $c$** (e.g. more patient / longer horizon when satiation is high, more myopic when starving)? This is measurable **today, with no env change**, in either reward mode — because we are reading the critic's internal horizon, not the agent's behaviour under delayed reward. I regard this as the primary, cheap, on-the-critical-path probe and recommend the symposium adopt it as the working test for the 5-HT knob.

**Caveat on Probe A.** It measures the *representational* horizon the critic learned, which is a function of $\gamma_{\text{train}}$ AND the modulator's reshaping. To attribute a $\gamma_{\text{eff}}$ shift to the modulator (not to reward sparsity or value-function approximation error), hold $\gamma_{\text{train}}$ and the food layout fixed and vary only $c$; the *differential* $\partial \gamma_{\text{eff}}/\partial c$ is the clean, modulator-attributable quantity. Also: distance-to-food must be a faithful temporal-displacement axis, so use the env's graph distance, not Euclidean.

**Probe B — behavioural delay-discounting (NEEDS-ENV-WORK).** The *behavioural* version — the agent actually choosing between a sooner-smaller and a later-larger reward, the classic intertemporal-choice phenotype — does require literal reward delay, and the grounding memo §4.4 is right that the env cannot deliver it. The two delay-named mechanisms (`res_reg_delay` respawn timer; the interoceptive nociception kernel) are availability and sensor delays respectively and would **confound** delay with availability/perception — using either would be an invalid discount probe (a point worth stating loudly, since both are tempting).

**What new env mechanism would make Probe B work, and is it worth it?** A **reward-delay queue on `EnvState`**: a fixed-length ring buffer (length = max delay $K$) into which the `ate_food` bonus is enqueued at the eating step and from which the entry at offset $k$ is added to `reward` $k$ steps later. The grounding memo correctly notes the existing injury ring buffer (`05_body_homeostasis.md:118-138`) is a working template, so this is structurally modest. With it, you build a sooner-small-vs-later-large food choice and read the indifference point, then test whether the indifference point (hence the behavioural discount) shifts with $c$.

**My recommendation on worth.** **Build Probe A now; defer Probe B.** Probe A answers the scientific question ("does the modulator shift the agent's effective horizon as a function of interoceptive state?") with zero env work and no confounds. Probe B adds the *behavioural* intertemporal-choice phenotype, which is more compelling for a cognitive-neuro audience (CCN/COSYNE) but is not needed to falsify the core claim. So the reward-delay queue is worth building *if and only if* the project wants the behavioural-choice phenotype as a headline contribution; for a first-pass falsification test it is off the critical path. I would gate the env-work decision on whether the target venue is RL-algorithmic (Probe A suffices) or cognitive-behavioural (Probe B adds value).

**Reward mode.** Probe A: either mode (it reads the critic). Probe B: **survival mode** preferred, because the discrete `+1` food bonus is exactly the quantity to delay; delaying a continuous drive-delta is semantically muddier (you would be delaying a body-state-change signal, which is ill-defined). This is a notable point — the reward-delay queue is *cleaner in survival mode*, the opposite of the DA knob, which is *cleaner in homeostatic mode*. The two knobs want opposite reward modes; a single experiment cannot test both simultaneously.

**Verdict: $\gamma_{\text{eff}}$-from-critic probe (Probe A) = MEASURABLE-AS-IS, no env work, either mode, on the critical path. Behavioural delay-discounting (Probe B) = NEEDS-ENV-WORK (a reward-delay ring buffer on `EnvState`, survival mode), worth building only if the behavioural intertemporal-choice phenotype is a headline goal.**

---

## §3 — Estimator and identifiability notes (the RL math, collected)

A few cross-cutting RL-mechanics points the per-knob probes depend on.

**GAE / return-estimator choice interacts with three of the four knobs.** All four read-outs sit downstream of the advantage/return estimator the project's PPO uses. Specifically:
- The **5-HT/discount probe (A)** reads $V$, so it inherits the bias of whatever bootstrapping the critic uses; a high-$\lambda$ GAE critic and an MC critic will report different $\gamma_{\text{eff}}$ for the same policy. **Report the estimator alongside $\gamma_{\text{eff}}$**; do not compare $\gamma_{\text{eff}}$ across runs with different $\lambda$.
- The **DA/IQN probe** is a *distributional* critic and is therefore a different return estimator than the scalar GAE critic the rest of the pipeline uses. Layering an IQN head changes which return law is being learned; if the IQN head and the scalar critic disagree, that is a return-estimator/return-distribution *mismatch*, not necessarily a modulator effect. Train the IQN head as the value head, not as a bolt-on, to avoid the mismatch.
- The **ACh/volatility block length** must exceed the GAE credit-assignment horizon $\sim 1/(1-\lambda\gamma)$ (§2.2).

**Advantage normalisation and the temperature knob.** The NA/entropy read-out can be polluted by per-batch advantage normalisation: if advantages are renormalised per batch, the *effective* gradient signal magnitude that drives entropy is decoupled from the raw reward scale, so an apparent entropy change could be a normalisation artefact rather than a modulator effect. Log raw and normalised advantage statistics when running the entropy regression.

**The temperature-head identifiability question the project flagged (Phase 2).** Whether the PPO temperature head is "approximating a risk-sensitive policy vs. only acting as an entropy regulariser" is decidable by *combining* the §2.1 and §2.3 probes: a pure entropy regulariser shifts $H[\pi]$ but leaves the IQN quantile-emission shape's *context-dependence* flat; a genuine risk-sensitive policy shifts the quantile reweighting with $c$. Running both probes on the same modulator output is the cleanest way to disambiguate — neither probe alone settles it. This is the strongest reason to build the DA/IQN probe even though the temperature knob is the green one.

---

## §4 — Bottom line, and which v8-null cause each speaks to

Mapping to the project's four causes of the v8 null result (project_plan §4): the **DA/IQN** and **5-HT/critic-horizon** probes both bear on the *return-estimator / value-target* family of causes — if the v8 null was partly a value-target problem, these probes are diagnostic, not just confirmatory. I flag that explicitly per the symposium charter.

| Knob | Measurability verdict | Reward mode | Performance metric | New code? |
|---|---|---|---|---|
| **NA → temperature** | MEASURABLE-AS-IS | Survival | Survival steps (inverse-U); policy entropy (primary) | No |
| **ACh → learning rate** | Gradient probe: with-caveat (proves steering, not volatility). Behavioural: NEEDS-ENV-WORK | Survival | Post-switch survival steps; gradient cosine | Block scheduler (small) for behavioural version |
| **DA → TD-gain** | MEASURABLE-WITH-CONFIG (homeostatic only, cost-side variance) | **Homeostatic** | Zone-occupancy + survival steps; IQN quantile curves | No (config + IQN head) |
| **5-HT → discount** | Probe A (critic horizon): MEASURABLE-AS-IS. Probe B (behavioural): NEEDS-ENV-WORK | A: either; B: **survival** | Probe A: $\gamma_{\text{eff}}$ slope. Probe B: indifference point | Probe A: no. Probe B: reward-delay ring buffer |

**One unavoidable cross-knob fact:** the DA knob *requires* homeostatic mode and the 5-HT behavioural knob *requires* survival mode. They cannot be co-tested in one experiment. The symposium should plan **two reward-mode tracks**, not one.

---

## §5 — Hand-offs

**Next steps**
- **`experiment-designer`** — (1) NA inverse-U: a threat config grid (sweep `detection_range`, predator `count`/`damage`) in survival mode, logging `dist_to_pred`, policy entropy, survival steps. (2) DA risky-vs-safe: a two-zone, mean-matched-cost homeostatic-mode config (Zone S `damage:[c,c]`, Zone R `damage:[c−δ,c+δ]`), tuned so injury-variance is a non-trivial share of return variance. (3) 5-HT Probe A: fixed `γ_train` + fixed food layout, sweep only interoceptive context. **Binding constraint:** any ACh block schedule must use block length > credit-assignment horizon $1/(1-\lambda\gamma)$ (H5 interaction).
- **`senior-developer`** — two contained env-work items, both off the first-pass critical path: (1) a cross-episode **block scheduler** for the ACh behavioural volatility signature (keyed on an episode counter in the reset path); (2) a **reward-delay ring buffer** on `EnvState` for the 5-HT behavioural Probe B (injury ring buffer is the template, survival mode). Recommend building neither until the value-function/critic probes (5-HT Probe A) and config-only probes (NA, DA) have been run — they may answer the question without the env work.
- **`professor-dl-theory`** — the IQN-head-vs-scalar-critic *architectural* coupling (how the distributional head attaches to the FiLM/hypernet modulator) is yours; I have specified only how the return law it learns behaves under cost-side variance.
- **`professor-bayesian-nn`** — if the DA/IQN quantile head is to also carry aleatoric/epistemic disentanglement (separating cost-variance from model uncertainty in the return distribution), that calibration question is yours.

---

*Contribution by professor-rl, 2026-06-18, for symposium `20260618_env_suitability_neuromodulation`. Built on [[00_env_capability_grounding]]. Every verdict argued from the grounding memo's cited environment facts; reward-mode requirements and the survival-steps-only performance rule are respected throughout.*
