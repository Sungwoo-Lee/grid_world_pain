---
title: "Environment capability grounding — can the pain grid world test the four Doya-2002 neuromodulator → RL-hyperparameter functions?"
symposium_session: 20260618_env_suitability_neuromodulation
role: research-postdoc (grounding pass)
date: 2026-06-18
knobs:
  - "NA → effective temperature (exploration / policy entropy)"
  - "ACh → effective learning rate (volatility-driven plasticity)"
  - "DA → effective TD-error gain (reward-variance sensitivity)"
  - "5-HT → effective discount (patience / effective horizon)"
inputs:
  - docs/environment/ENVIRONMENT_SUMMARY.md
  - docs/environment/05_body_homeostasis.md
  - docs/environment/06_reward_and_termination.md
  - docs/environment/07_predator_ai.md
  - docs/environment/08_resources_and_obstacles.md
  - docs/environment/09_sensors_and_observation.md
  - docs/environment/02_config_schema.md
downstream: four professor contributions (this symposium) build on this grounding
---

# Environment capability grounding for the four-neuromodulator symposium

## §1 — Plain-English entry point

This memo is the **shared ground-truth** for a four-professor symposium asking one question: *can our existing pain grid world environment actually test the four classic neuromodulator-to-learning-knob functions proposed by Doya (2002)?* Those four functions are:

1. **Noradrenaline (NA) → effective temperature** — how exploratory vs. greedy the agent's policy is. To test it, the environment must be able to **vary threat / arousal** in a graded way (e.g. a predator that comes closer or further).
2. **Acetylcholine (ACh) → effective learning rate** — how fast the agent updates beliefs when the world changes. To test it, the environment must be able to **change its own statistics** within or across episodes (volatility / regime change).
3. **Dopamine (DA) → effective TD-error gain** — how strongly reward-prediction-errors drive learning, which shows up as sensitivity to **reward-magnitude variance**. To test it, the environment must offer **two paths differing in payoff variance** (a steady small reward vs. a risky large one).
4. **Serotonin (5-HT) → effective discount** — how patient the agent is (how far into the future it values reward). To test it, the environment must be able to **deliver reward at a tunable delay** after the action that earned it.

Important framing the project has settled on: the modulator does **not** literally set these hyperparameters. Instead it reads the agent's **interoceptive state `c`** — its satiation (fullness), recent pain/injury, and optionally fatigue — and induces conditioning (FiLM / hypernet) that produces *behavioural signatures* we then measure *as if* they were those hyperparameters. So the real question for each knob is narrower: **can the environment express the behavioural signature at all?**

My job here is a blunt per-knob capability verdict — **SUPPORTED natively**, **PARTIAL** (reachable by editing YAML config only, no new code), or **NOT SUPPORTED** (needs new code in `src/environment/`) — each grounded in a file:line citation. The headline: three of the four required manipulations are at least reachable today; **the reward-delay manipulation that 5-HT/discount needs is NOT SUPPORTED** — the environment pays reward on the same step the food is eaten and has no delay primitive. I do not propose architecture here; that is the professors' job next.

---

## §2 — What the environment is

GridWorld Pain is a JAX-based reinforcement-learning environment in which a single agent forages on a 2-D grid while regulating an internal body. Two pressures compete every step: **external reward-seeking** (find and eat food) and **internal homeostatic regulation** (avoid starving, avoid injury). The agent dies if nutrition reaches zero or accumulated injury reaches its maximum, and otherwise the episode runs to a step cap (`ENVIRONMENT_SUMMARY.md:9`, termination codes at `06_reward_and_termination.md:198-204`).

The agent never sees the world state directly — it receives a flat observation vector assembled from up to ten sensors, split into **interoceptive** channels (injury, nutrition, satiation, a delayed "tonic" nociception trace) and **exteroceptive** channels (contact pain, smell/olfaction, collision, vision, location) (`09_sensors_and_observation.md:9`, layout table at `ENVIRONMENT_SUMMARY.md:129-140`). The interoceptive channels are the source of the modulator input `c` (catalogued in §5). The world also contains **mobile predators** that hunt the agent with a stamina-limited pursuit state machine (`07_predator_ai.md:145-219`), **neutral animals** that wander harmlessly, **obstacles** (rocks that block, bushes that hide), and **resources** (food, and "hiding-predator" trap tiles that deal damage) (`08_resources_and_obstacles.md:11-16`).

Reward comes in two configurable modes (`06_reward_and_termination.md:11-16`): **survival mode** (`use_homeostatic_reward=False`) gives `+1` on eating food and `-death_penalty` at episode end; **homeostatic mode** (`use_homeostatic_reward=True`) gives a dense per-step "drive reduction" signal — the agent is rewarded for moving toward a healthy setpoint (less hungry, less injured) (`06_reward_and_termination.md:113-117`, `calculate_drive` at `05_body_homeostasis.md:241-245`). Performance is always scored in **survival steps**, never cumulative reward (`06_reward_and_termination.md:16`). Nearly everything described above is parameterised in YAML and loaded into a Flax `EnvParams` struct; whether a manipulation is config-reachable or needs new code is decided by what the loader exposes (`02_config_schema.md:9-13`).

---

## §3 — Capability table

| Knob (Doya 2002) | Required environmental manipulation | Verdict | Mechanism that does / could do it | File:line evidence | What's missing |
|---|---|---|---|---|---|
| **NA → effective temperature** (exploration / entropy) | (a) graded threat / arousal | **SUPPORTED** | Predator hunt FSM with config-tunable detection radius, patrol box, stamina, damage, count; threat proximity varies continuously and is observable via smell + vision + contact pain | `07_predator_ai.md:180-185` (detection), `07_predator_ai.md:387-398` (config keys), `02_config_schema.md:228-240` (per-episode distributional ranges) | Nothing structural; threat is *emergent* (agent-position-dependent), not a directly-set scalar — graded levels come from config sweeps across configs |
| **ACh → effective learning rate** (volatility / regime change) | (b) world statistics change within/across episodes | **PARTIAL** | Per-episode uniform re-sampling of five predator parameters (detection, stamina, recovery, hunt-threshold, lose-interest) at every reset; resource respawn position + chemical re-draw | `07_predator_ai.md:89-100` (per-episode sampling), `08_resources_and_obstacles.md:165-167` (respawn re-draw) | No **within-episode** scheduled regime switch and no **cross-episode block structure** (e.g. "stable block then volatile block"). Re-sampling is i.i.d. per episode, not a controllable volatility schedule. Building blocked/switching volatility needs new code |
| **DA → effective TD-error gain** (reward-variance sensitivity) | (c) two paths differing in reward variance | **PARTIAL** | Per-contact uniform damage range `[min,max]` on predators/traps/obstacles; food net nutrition gain; spatial layout lets two regions differ in payoff statistics | `07_predator_ai.md:275-276` (damage `[min,max]`), `08_resources_and_obstacles.md:191-194` (resource damage draw), `05_body_homeostasis.md:50-59` (food gain) | Reward **variance** is reachable only indirectly: food reward itself is fixed (`+1` per eat, `06_reward_and_termination.md:28`); variance must be engineered via the *cost* side (variable damage) or spatial layout, and there is no single "reward σ" knob. A clean low-var-vs-high-var two-path task needs careful config design but no new code |
| **5-HT → effective discount** (patience / horizon) | (d) tunable reward **delay** | **NOT SUPPORTED** | — (no mechanism delivers reward N steps after the earning action) | `06_reward_and_termination.md:28` (`+1` paid same step as `ate_food`); `res_reg_delay` is **respawn** delay not reward delay (`08_resources_and_obstacles.md:170-172`) | A reward-delay buffer (queue the reward, release it k steps later) does not exist. The only "delays" in the env are food **respawn** timing and the **interoceptive nociception** kernel — neither defers the reward signal. Needs new env code |

---

## §4 — Per-knob detail

### 4.1 NA → effective temperature — SUPPORTED

**Signature to express:** a Yerkes-Dodson inverse-U in task performance against inferred arousal under threat, and policy-entropy changes as threat varies.

**Mechanism.** The environment already produces graded threat. Predator detection is a pure Manhattan-distance check `dist <= hunt_detect` that flips the predator into active pursuit (`07_predator_ai.md:180-185`); as the agent moves nearer or further, the threat it experiences varies continuously. Threat intensity is shaped by config: `detection_range`, `patrol_area`, `max_stamina`, `damage [min,max]`, `attack_delay`, `lose_interest_multiplier`, and predator `count` are all YAML keys (`07_predator_ai.md:387-398`). Threat is also *perceptible* to the agent — predators emit a smell channel, light up visual channel 5, and deliver contact nociception (`09_sensors_and_observation.md:158`, `:495`) — so "inferred arousal" has real observational support.

**Config / code that carries it.** Per-config: a predator entry under `environment.entities:` (or legacy `environment.predators:`), with `detection_range` and `spawn_area`/`patrol_area` controlling proximity. Per-episode graded arousal can also use distributional ranges, e.g. `detection_range: [3, 7]` (`02_config_schema.md:228-240`).

**Honest gap.** Threat is **emergent and agent-controlled**, not a scalar the experimenter sets directly each step — the agent can flee to lower its own threat. Graded "arousal levels" therefore come from sweeping across configs (small vs. large detection radius, near vs. far patrol), not from a single in-episode arousal dial. This is adequate for an inverse-U sweep but the professors should note that arousal is a *measured* quantity, not an *imposed* one.

### 4.2 ACh → effective learning rate — PARTIAL

**Signature to express:** learning-rate shifts driven by volatility / regime change; a stratified gradient probe distinguishing stable vs. volatile regimes.

**Mechanism.** Two sources of changing statistics exist. First, at **every episode reset** the five predator behavioural parameters are re-sampled from uniform `[low, high]` ranges (`07_predator_ai.md:89-100`) — so across episodes the predator's detection, stamina, recovery, hunt-threshold, and lose-interest vary. Second, **resource respawn** re-samples both position (within `res_spawn_area`) and chemical signature on each regeneration (`08_resources_and_obstacles.md:165-167`), so the smell-to-location mapping drifts within an episode if `res_property_std > 0`.

**Config / code that carries it.** Widening the distributional ranges (`detection_range: [lo, hi]`, etc.) and setting `res_property_std > 0` are pure YAML edits (`02_config_schema.md:228-240`, `08_resources_and_obstacles.md:180-185`).

**Honest gap.** The re-sampling is **i.i.d. per episode**, not a *controllable volatility schedule*. A canonical ACh/volatility paradigm needs either a **within-episode scheduled switch** ("predator behaviour changes at step 200") or a **blocked cross-episode design** ("a stable block of episodes followed by a volatile block"), so that learning rate before vs. after a change-point can be compared. Neither block-structure nor an in-episode change-point scheduler exists in the loader or step loop. Achieving a true regime-change probe needs new env code (a scheduled parameter switch keyed on `current_step` or an episode counter). Verdict PARTIAL: the *ingredients* of variability are config-reachable, but a *designed* volatility manipulation is not.

### 4.3 DA → effective TD-error gain — PARTIAL

**Signature to express:** per-context return-distribution shape, and a choice between a low-variance steady-reward path and a high-variance large-reward path.

**Mechanism.** The variance lever the environment exposes is the **per-contact uniform damage range** `[min, max]`: predators (`07_predator_ai.md:275-276`), trap resources (`08_resources_and_obstacles.md:191-194`), and obstacles all sample fresh damage on every hit. A region with `damage: [10, 10]` delivers deterministic cost; a region with `damage: [0, 40]` delivers the same mean with high variance. Combined with spatial layout (two `spawn_area`-separated zones) and food placement, two paths with matched mean but different return variance are constructible.

**Config / code that carries it.** All damage ranges, spawn areas, and food gains are YAML (`02_config_schema.md:790-816`, `05_body_homeostasis.md:50-59`).

**Honest gap.** The **food reward itself is fixed** at `+1` per eat in survival mode (`06_reward_and_termination.md:28`) — there is no per-resource reward-magnitude variance knob on the *gain* side; variance has to be injected through the *cost/damage* side or through stochastic food availability (respawn timing). There is no single "reward σ" parameter, so a clean distributional-RL two-path task requires deliberate config engineering (and homeostatic mode, where the drive-reduction signal carries continuous magnitude, may be the better substrate than survival mode's `{0, +1}`). No new code is required, so PARTIAL rather than NOT SUPPORTED — but the professors should not assume a turnkey risky-vs-safe arm.

### 4.4 5-HT → effective discount — NOT SUPPORTED

**Signature to express:** a reward-delay perturbation that shifts the effective horizon — measured as the slope of `log|ΔV|` against delay `k`.

**Mechanism.** None. When the agent eats, the reward is assigned on **the same step**: `reward_extrinsic = jnp.where(ate_food, 1.0, 0.0)` (`06_reward_and_termination.md:28`); in homeostatic mode the drive-reduction reward is likewise computed from the *current* step's body update (`06_reward_and_termination.md:113-117`). There is no buffer that holds an earned reward and releases it `k` steps later.

**The tempting-but-wrong candidates.** Two "delay"-named mechanisms exist and neither is a reward delay. (1) `res_reg_delay` is the **respawn** timer — how long after a food is consumed before a new one appears (`08_resources_and_obstacles.md:170-172`); it changes food *availability*, not the *latency between earning action and reward delivery*. (2) The **interoceptive nociception kernel** convolves past injury into a delayed *perceptual* signal (`09_sensors_and_observation.md:79-103`) — it delays a *sensor*, not the reward. Using either as a discount probe would confound delay with availability or perception.

**Honest gap and what it needs.** A genuine reward-delay manipulation requires new code in `src/environment/`: a small per-step reward queue on `EnvState` (e.g. a ring buffer parallel to the injury buffer) that defers the `ate_food` bonus by a config-set number of steps before adding it to `reward`. This is a modest addition structurally (the injury ring buffer at `05_body_homeostasis.md:118-138` is a working template) but it is genuinely absent today. Verdict: **NOT SUPPORTED** — do not assume the 5-HT/discount function can be tested without an env change.

---

## §5 — Interoceptive-state-`c` inventory

`c` is the modulator input common to all four knobs. The environment exposes the following interoceptive channels in the observation vector (authoritative order from `get_observation_breakdown`, `ENVIRONMENT_SUMMARY.md:129-140`, `09_sensors_and_observation.md:26-37`). Indices below assume the default config with all interoceptive sensors enabled.

| Obs index | Channel | Enabled by (YAML / `EnvParams` flag) | Range | What it carries | Ground-truth source |
|---|---|---|---|---|---|
| `[0]` | **Injury** | `injury_observable` (gateable; can be hidden) | `[0, 1]` | `injury_level / max_injury` — accumulated, recoverable physical damage | `05_body_homeostasis.md:23`, formula `09_sensors_and_observation.md:60` |
| `[1]` | **Nutrition** | `nutrition_observable` (gateable; can be hidden) | `[0, 1]` | `nutrition / max_nutrition` — objective energy store, decays each step | `05_body_homeostasis.md:43-66`, formula `09_sensors_and_observation.md:61` |
| `[2]` | **Satiation** | always on | `[0, 1]` | `satiation / max_satiation` — subjective fullness, power-law of nutrition | `05_body_homeostasis.md:72-92`, formula `09_sensors_and_observation.md:62` |
| `[3]` | **Interoceptive Nociception** | `interoceptive_nociception_enabled` | `[0, 1]` | delayed, alpha-kernel-convolved "tonic pain" trace of past injury (or passthrough injury if convolution off) | `09_sensors_and_observation.md:79-107`, buffer at `05_body_homeostasis.md:188-204` |

Notes for the professors:

- **The two pain-relevant `c` channels are distinct.** Index `[3]` (interoceptive nociception) is a *delayed, smoothed* function of internal injury via the `nociception_history_buffer` + alpha kernel (peak lag at `interoceptive_kernel_tau`); index `[0]` (injury) is the *instantaneous* internal damage level. There is also an **exteroceptive** nociception channel at `[4]` (`nociception_enabled`) — phasic contact pain — but that is exteroceptive, not part of `c` (`09_sensors_and_observation.md:145-174`).
- **Hideability.** Injury `[0]` and nutrition `[1]` can be gated *out* of the observation (`injury_observable=False` / `nutrition_observable=False`), forcing the agent to infer body state from satiation and the delayed nociception trace (`05_body_homeostasis.md:36-39`). This matters if a professor wants `c` to be partially latent.
- **"Recent pain/injury" for the modulator** is best read from `[3]` (the convolved trace) — it is the channel the env explicitly designed as a temporally-integrated injury percept. **"Satiation"** is `[2]`. **"Fatigue"** has *no dedicated channel*; the closest proxy is `rest_streak` (an internal `EnvState` int, `05_body_homeostasis.md:27`) which is **not** in the observation vector — exposing fatigue as a `c` channel would need new sensor code.

---

## §6 — Bottom line

**Testable today, by config alone:** the **NA → effective-temperature** function (SUPPORTED) — graded threat already exists via the predator hunt FSM and its YAML-tunable detection/patrol/damage parameters; an inverse-U sweep is a config sweep.

**Reachable but needing deliberate config design (no new code):** the **ACh → learning-rate** function (PARTIAL — variability ingredients exist but a *controlled* volatility schedule, blocked or within-episode change-point, does not, so a clean regime-change probe needs new code) and the **DA → TD-error-gain** function (PARTIAL — reward-*cost* variance is config-reachable via `damage [min,max]` and spatial layout, but there is no single reward-σ knob and food reward itself is fixed, so a risky-vs-safe two-path task must be engineered carefully).

**Not testable without new env code:** the **5-HT → effective-discount** function (NOT SUPPORTED) — reward is paid on the same step the action earns it, and the two delay-named mechanisms (`res_reg_delay`, the interoceptive kernel) are respawn-timing and sensor-delay respectively, neither a reward delay. A per-step reward-delay queue on `EnvState` would close the gap and the existing injury ring buffer is a ready template, but it is genuinely absent today.

So: one knob green, two amber, one red. The professors should build their contributions on these assumptions and **not** silently assume the discount/5-HT function is testable as-is.

---

*Grounding pass by research-postdoc, 2026-06-18. Every verdict is cited to the environment reference docs under `docs/environment/`; no architecture is proposed here. The four professor contributions for symposium `20260618_env_suitability_neuromodulation` build on this memo.*
