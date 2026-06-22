---
title: "dreamer_srl 3-stage size+entity curriculum vs from-scratch on the 10×10 full task"
topic: continual_learning
status: active
created: 2026-06-09
last_updated: 2026-06-18  # Budget-ablation T1–T4 results filled (§8): floor = T3 71.5k; T4 under-trained; curriculum = same plateau, ~45% fewer episodes vs from-scratch (single seed)
phase: 2
wandb_tag: "dreamer_v3_dsrl_curric3_*"
develop_link: docs/develop/active/diagnosis/dreamer_hypervigilance_learning_failure.md
cross_links:
  - docs/experiments/active/continual_learning/DREAMER_CURRICULUM_FOOD_THEN_PREDATOR.md
  - docs/experiments/active/dreamer_srl_v2/ep10m_3cell_termination_analysis.md
---

# dreamer_srl 3-stage size+entity curriculum vs from-scratch on the 10×10 full task

## 1. Purpose (plain-language entry point)

We want to know whether **teaching the agent in three escalating steps** beats **throwing it straight at the hard task**.

The agent is **dreamer_srl** — a from-scratch JAX re-implementation of DreamerV3, a model-based reinforcement learner that first learns a compact predictor of the world ("world model") and then trains its policy inside imagined roll-outs of that predictor. Performance is measured in **survival steps** (how many steps the agent stays alive out of a 500-step episode cap), never cumulative reward.

The hard task is a **10×10 grid survival game**: find and eat food while avoiding a moving predator that hunts you, four static ambush hazards ("hiding predators") that kill on contact, and mild-damage rocks — with hiding bushes you can duck into to break the hunter's line of sight, and harmless rabbits wandering around as distractors. When dreamer_srl is trained on this task **from scratch**, three independent runs all climbed slowly to about **200 survival steps after roughly two days of wall-clock and ~47,000 episodes, and were still rising when stopped** (see the companion analysis of the three parity runs). A separate diagnosis showed that on the *small* 5×5 version of this task, an earlier DreamerV3 build got **stuck near 30 survival steps and never escaped** — its policy stayed essentially random because the death signal was too sparse for the world model's reward and "am-I-still-alive" heads to learn from.

**The curriculum idea**: instead of starting on the full 10×10 task, start the same agent in a **small, simple 5×5 world** (food + one static ambush hazard + rocks), then add a **gentle, slowed-down moving hunter and hiding bushes** (still 5×5), then finally **scale up to the full 10×10 task with rabbits and the hunter restored to full strength**. The agent's neural-network weights carry across each transition unchanged — only the world it acts in changes. The single pre-registered question this experiment answers:

The Stage-2 hunter is deliberately a **"bridge" chaser**: it moves slowly, does about a third of the damage, and has a tighter detection range, so the agent's first encounter with a moving predator is survivable rather than instantly lethal. Full predator strength is **deferred to Stage 3**. This makes the difficulty ramp smoother and — per the diagnosis of why the small-grid agent got stuck — gives the world model's "am-I-still-alive" head a chance to learn from occasional, recoverable death events before it faces the dense, fast lethality of the full task.

> **Does a dreamer_srl agent that warms up through the 3-stage curriculum reach a higher final survival-step plateau on the 10×10 full task than a dreamer_srl agent trained on that task from scratch, and does it do so without losing seed-stability — measured across 5 seeds?**

This doc owns the **config set and stage schedule**. The **continual-learning engine that reads them** is being ported in parallel by `senior-developer`; the launch command below depends on that port landing. This doc does **not** touch `src/`.

## 2. Hypothesis & Predicted Outcomes

**Hypothesis (H1).** Warming up on the small, simple stages lets the world model's reward head and continuation ("alive/dead") head accumulate clean eating-reward and death-terminal samples *before* the agent faces the full task, so the curriculum agent reaches a **higher and/or faster-converging** survival plateau on the 10×10 full task than the from-scratch baseline.

**Confirmation (pre-registered).** The curriculum is judged to **help** if, on the Stage-3 full task, the curriculum arm's final survival (mean over the last 2,000 episodes, averaged across 5 seeds) exceeds the from-scratch baseline's final survival by **≥ 20 survival steps with non-overlapping 95% confidence intervals**, AND the curriculum arm's seed spread at that checkpoint is **≤ the baseline's** (no stability cost). "Final" is taken at matched cumulative **environment-step** budget on Stage 3, not matched episodes (episode counts differ because episode lengths differ between arms).

**Refutation (pre-registered).** The curriculum is judged to **not help** (H1 refuted) if either: (a) the curriculum arm's Stage-3 final survival is **within ±20 steps** of the baseline (null — the warm-up bought nothing), or (b) the curriculum arm is **worse** than the baseline (negative transfer — the small-grid warm-up taught geometry-specific habits that hurt on 10×10). A **catastrophic-forgetting** signature — survival collapsing at the Stage-2→Stage-3 boundary and not recovering to baseline within the Stage-3 budget — also counts as refutation of H1 (and is logged as its own failure mode, §6).

**Predicted shape of a positive result.** Survival rises on Stage 1, **drops at each stage boundary** (the task just got harder), then recovers; the Stage-3 recovery overshoots the from-scratch curve and plateaus higher. A flat Stage-1 trace near ~30 survival steps that never rises would instead reproduce the documented small-grid stuck-policy failure and predicts a null/negative final result.

## 3. Launch Manifest (system-of-record)

All rows share **wandb-group `dreamer_srl_curriculum`** and **wandb-job-type `prod`**. The curriculum arm is the new continual run; the baseline arm is the existing from-scratch 10×10 run pattern, re-run here at matched seeds for a clean paired comparison. Tag = wandb-name on every row. Actual columns (`Node`, `GPU`, `Launched at`, `WandB run ID`, `Log path`) are filled by `training-runner` at launch.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Curric s42 | planned | C-42 | `dreamer_v3_dsrl_curric3_s42` | dreamer_srl_curriculum | prod | 42 | — | — | — | — | — |
| Curric s43 | planned | C-43 | `dreamer_v3_dsrl_curric3_s43` | dreamer_srl_curriculum | prod | 43 | — | — | — | — | — |
| Curric s44 | planned | C-44 | `dreamer_v3_dsrl_curric3_s44` | dreamer_srl_curriculum | prod | 44 | — | — | — | — | — |
| Curric s45 | planned | C-45 | `dreamer_v3_dsrl_curric3_s45` | dreamer_srl_curriculum | prod | 45 | — | — | — | — | — |
| Curric s46 | planned | C-46 | `dreamer_v3_dsrl_curric3_s46` | dreamer_srl_curriculum | prod | 46 | — | — | — | — | — |
| Base s42 | planned | B-42 | `dreamer_v3_dsrl_fromscratch10x10_s42` | dreamer_srl_curriculum | prod | 42 | — | — | — | — | — |
| Base s43 | planned | B-43 | `dreamer_v3_dsrl_fromscratch10x10_s43` | dreamer_srl_curriculum | prod | 43 | — | — | — | — | — |
| Base s44 | planned | B-44 | `dreamer_v3_dsrl_fromscratch10x10_s44` | dreamer_srl_curriculum | prod | 44 | — | — | — | — | — |
| Base s45 | planned | B-45 | `dreamer_v3_dsrl_fromscratch10x10_s45` | dreamer_srl_curriculum | prod | 45 | — | — | — | — | — |
| Base s46 | planned | B-46 | `dreamer_v3_dsrl_fromscratch10x10_s46` | dreamer_srl_curriculum | prod | 46 | — | — | — | — | — |

Tags are unique across the manifest. Seeds 42–46 are matched between the two arms for a paired comparison.

> **Baseline-reuse note.** The three already-completed from-scratch parity cells (seed 42 only) are *not* used as the baseline here — they predate this design and used the single-seed `01-interoNocicept.yaml` env config without the curriculum's identical fingerprint guarantee at seeds 43–46. The `Base s4x` rows re-run the from-scratch arm at the same 5 seeds against the **same Stage-3 config** (`03_10x10_full_task.yaml`), so the comparison is config-identical end-task. If the user prefers to fold in the existing seed-42 cell as a 6th baseline point, that is a launch-time decision, not a design change.

### 3.1 Configs to Produce (designer-only, pre-launch)

| Run | Stage env configs (`--configs-dir`) | Schedule (`--continual-schedule`) | Agent config |
|---|---|---|---|
| Curric s42–s46 | `configs/experiment/dreamer_srl_curriculum/{01_5x5_food_hide_rock, 02_5x5_food_hide_chase_rock_bush, 03_10x10_full_task}.yaml` | `configs/continual/dreamer_srl_3stage_size_curriculum.yaml` | `configs/models/dreamer_srl/01_food_only_buf256k.yaml` |
| Base s42–s46 | single-stage `--config configs/experiment/dreamer_srl_curriculum/03_10x10_full_task.yaml` (no `--configs-dir`) | n/a | `configs/models/dreamer_srl/01_food_only_buf256k.yaml` |

All four config files above are authored by this doc and exist on disk. The agent config is the existing XS / 256k-replay preset that the running parity cells use — **unchanged**, so the two arms differ only in env exposure, not in agent hyperparameters.

## 4. Experimental Design

- **Independent variable**: training-exposure schedule — `{curriculum (3-stage), from-scratch (Stage-3 only)}`.
- **Dependent variable (primary)**: survival steps on the 10×10 full task (`03_10x10_full_task.yaml`), mean over the last 2,000 episodes per seed, averaged across seeds, with 95% CI.
- **Dependent variables (secondary, diagnostic only)**: per-stage survival trajectory; world-model reward-head MAE on positive vs. negative reward samples; continuation-head accuracy; actor entropy (the diagnosis metrics that flagged the stuck-policy failure). Cumulative reward is **not** a headline metric.
- **Controls / fixed factors (pinned and named)**:
  - Agent config: `configs/models/dreamer_srl/01_food_only_buf256k.yaml` (XS preset, 256k replay) — identical for both arms.
  - `--num-envs 16`, the same parallelism the parity cells used.
  - **Modality fingerprint identical across all three stages AND both arms** — see §"Modality fingerprint match" below.
  - Noise **off** (`perceptual_noise.enabled: false`) in every stage — this experiment is about exposure schedule, not noise robustness.
  - Stage-3 task is **byte-value-identical** to the from-scratch baseline task, so the two arms are evaluated on exactly the same environment.
- **Seeds**: 5 per arm (42–46). The expected effect (curriculum transfer) is plausibly **marginal** — the from-scratch baseline already learns, just slowly — so 5 seeds (not 3) are used to resolve a ~20-step difference against the ~140-step single-episode noise band documented for these runs.
- **Sample size / compute estimate**: at ~40 env-steps/sec (num_envs=16), the curriculum's three stages total roughly 2.5–3.5M (Stage 1) + 2.5–5M (Stage 2) + the Stage-3 bulk env-steps; the from-scratch baseline reached ~7.4M env-steps in ~50h. Budget per run is a comparable multi-day window; 10 runs total. Exact wall-clock depends on how fast survival (and thus episode length) rises — longer episodes mean fewer episodes per env-step.

## 5. Analysis Plan (pre-specified)

- **Primary statistic**: mean survival steps over the last 2,000 episodes per seed → mean ± 95% CI across the 5 seeds, per arm, at matched Stage-3 cumulative env-step budget.
- **Effect-size threshold**: curriculum counts as an improvement iff Δ(curriculum − baseline) ≥ **20 survival steps** with non-overlapping 95% CIs (per §2).
- **Temporal-evolution check (mandatory)**: plot survival vs. cumulative env-step for both arms across the whole run, with the two stage boundaries (Stage-1→2 at episode 15,000; Stage-2→3 at episode 75,000) marked. The window of interest is the **first 2M env-steps of Stage 3** (does the curriculum arm recover above baseline faster?) and the **final 2M env-steps** (plateau comparison).
- **Time-locked boundary analysis**: at each stage boundary, measure the survival drop and the recovery time (env-steps to return to pre-boundary survival). Pre-specified window: 500k env-steps each side of each boundary.
- **Diagnostic cross-check**: at the end of Stage 1, confirm actor entropy has fallen below the uniform-policy level (ln 6) and reward-head positive-sample MAE has dropped — if not, the Stage-1 warm-up failed to escape the documented stuck-policy regime and any null Stage-3 result is attributable to that, not to the curriculum idea.

## Modality fingerprint match (cross-stage weight-transfer guarantee)

Weight transfer across stages only works if the agent's observation vector and sensor/noise layout are identical at every stage — otherwise the encoder's input dimension changes and the carried weights are meaningless. This was verified by **live-loading each stage config through the real env loader** (`load_env_params`) and resetting the env:

- **Observation vector**: fixed **27-dim** in all three stages, with byte-identical per-modality breakdown — Satiation 1, Interoceptive-Nociception 1, Extero-Nociception 1, Olfaction 5, Collision 5, Proprioception 6, Visual 8. The visual patch is sized by `visual_sensor_range`, **not** by grid height/width, so the 5×5 → 10×10 scale-up does **not** change the observation shape. This is the empirical proof the curriculum is shape-safe.
- **Noise layout**: the `perceptual_noise.modalities` block has the identical 10-entry order (injury, nutrition, satiation, interoceptive_nociception, extero_nociception, olfaction, collision, proprioception, visual, location) → internal length-13 padded noise arrays, identical across stages. Noise is disabled in all stages, but the *layout* still has to match for the modality indices to line up.
- **Body + sensory blocks**: every parsed key/value under `body:`, `sensory:`, and `perceptual_noise:` is value-identical across the three stages and value-identical to the running from-scratch baseline env config (`configs/experiment/hypervigilance/01-interoNocicept.yaml`), confirmed by a comment-stripped diff. The only differences between stage files are entity counts/params and grid height/width — exactly the dimensions the curriculum is meant to vary.

## Entity → v2.0 schema mapping

The intended entities map onto the current v2.0 ("unified animal entity") schema as follows. Mapping was verified against the running dreamer_srl env config and the v2.0 config-schema reference.

| Spec entity | v2.0 schema home | Encoding | Stages present |
|---|---|---|---|
| food | `environment.resources`, `type: food` | `res_type` 0 | 1, 2, 3 |
| hiding predator (static lethal ambush) | `environment.resources`, `type: hiding_predator` | `res_type` 1; static, damage [15,45], nociception 0.9 | 1, 2, 3 |
| chasing predator (mobile hunter) | `environment.predators[]` (legacy section, auto-projected) | class `predator`, behaviour `hunt` — **slowed "bridge" form in Stage 2, full strength in Stage 3** (see bridge-chaser note below) | 2 (bridge), 3 (full) |
| rabbit (neutral animal / distractor) | `environment.neutral_animals[]` (legacy section, auto-projected) | class `neutral`, behaviour `wander` | **3 only** |
| rock (mild-damage obstacle) | `environment.obstacles[]` | `blocking: false`, damage [1,5] | 1, 2, 3 |
| bush (line-of-sight break) | `environment.obstacles[]`, `hides_agent: true` | hides agent from hunter | 2, 3 |

The legacy `predators:` / `neutral_animals:` dual-section form is **still accepted** by the v2.0 loader (auto-projected into the unified animal array), and is what the running dreamer_srl baseline uses — so the stage configs use it too, for maximal byte-similarity to the baseline. **No spec entity is inexpressible in the current schema** (see §"Schema flags" — none found).

### Stage-2 bridge-chaser note (user-locked)

The chasing predator enters at Stage 2 in a **slowed, lower-damage, shorter-detection "bridge" form** rather than at full strength, so the agent's first exposure to a moving hunter is a gentle introduction. Full strength is restored at **Stage 3** (`03_10x10_full_task.yaml`, untouched). The bridge values follow the prior `dreamer_curriculum/02_predator_slow.yaml` precedent. The deltas, all on the single `environment.predators[0]` entry of `02_5x5_food_hide_chase_rock_bush.yaml`, are:

| Field | Stage 2 (bridge) | Stage 3 (full) | Effect of the bridge |
|---|---|---|---|
| `move_interval` | **8** | 1 | hunter moves ~8× less often → slower pursuit |
| `attack_delay` | **8** | 3 | more warning steps before a strike lands |
| `damage` | **[5.0, 15.0]** | [15.0, 45.0] | ~1/3 damage → a hit is survivable, not instant death |
| `detection_range` | **2** | 5 | tighter detection cone → fewer / later engagements |

**Rationale.** A full-strength hunter on a tiny 5×5 grid kills almost immediately, which floods the world model with deaths and reproduces the dense-lethality regime the diagnosis blamed for the stuck policy. The bridge chaser keeps death events **occasional and recoverable**, letting the continuation ("alive/dead") head and the reward head's heavy-negative tail get sampled cleanly before Stage 3 turns the hunter back up. This smooths the cross-stage difficulty ramp without introducing any new entity or sensor.

**Fingerprint safety.** All four bridge fields are **dynamic predator-behaviour parameters** — none belongs to the cross-stage modality fingerprint (sensor flags, body block, perceptual-noise block, observation layout). The edit was load-checked: all three stage configs load cleanly through `load_env_params` (no `ValueError`), and the `sensory` + `body` + `perceptual_noise` blocks remain **byte-identical across Stages 1, 2, and 3**, so the **27-dim observation and the weight-transfer guarantee are unchanged**.

## Per-stage budget justification

Throughput anchor: dreamer_srl XS at num_envs=16 runs at **~40 env-steps/sec** (the three running parity cells logged 40.2 / 41.7 / 41.1 SPS), and reached **~47,000 episodes / ~7.4M env-steps in ~50h** on the 10×10 full task (~157 env-steps/episode averaged over the run). The schedule is on **episodes** (the engine contract), so per-stage episode budgets are chosen from per-stage episode-length expectations:

- **Stage 1 — 15,000 episodes** (cumulative 15,000). Benign 5×5 world; once eating is learned, episodes run long (toward the 500-step cap / starvation, few deaths) → ~150–250 steps/ep → ~2.5–3.5M env-steps. Enough for the reward head to sample the eating tail repeatedly and the actor to leave uniform, before any moving threat. Anchored to the 15–16k Stage-1 budgets used by the two prior dreamer curricula.
- **Stage 2 — 60,000 episodes** (cumulative 75,000; the longest stage by episode count). A moving hunter on a tiny 5×5 grid kills fast early → short episodes (~20–50 steps), so a high episode count is needed to feed the continuation/reward heads enough death-terminal samples (the under-sampled signal the diagnosis flags). ~2.5–5M env-steps; yield rises as bush-hiding is learned.
- **Stage 3 — 685,000 episodes** (cumulative **760,000** = total). The bulk stage and the measurement stage. The cap is deliberately loose — far above the ~47k episodes the from-scratch baseline consumed — because runs are evaluated / stopped long before the cap. On 10×10 the task yields ~60 steps/ep early rising to ~200 late.

Checkpoint frequencies `[7500, 30000, 100000]`: 2 checkpoints in each short stage (mid + end, to capture the post-warm-up and post-bridge weights for offline probing), coarse 100k cadence on the long Stage 3 to bound disk use while still resolving the survival inflection.

## Budget ablation (T1–T4)

**What this is, in plain language.** The original schedule above gave Stage 1 a 15,000-episode budget and capped the whole curriculum at 760,000 episodes. Live training data shows that was far too generous: on Stage 1 (the small 5×5 world with food, one static ambush hazard, and rocks) the agent's survival **flattens out by about episode 1,000** — confirmed on the first running curriculum seed (the run logged under the short ID `crvnmbo8`). So the agent spends ~14,000 episodes on a stage it has already mastered, and the cap is roughly **10× longer than the curriculum actually needs**. Rather than guess a single tighter number, we run **four progressively tighter budget schedules side by side** — T1 (loosest of the four, the agreed starting point) through T4 (most aggressive, Stage 1 sitting right at the observed plateau) — and read off the smallest budget that still works.

**Held constant across all four.** Same three stage env configs (`configs/experiment/dreamer_srl_curriculum/` — `01`/`02`/`03`, untouched), same agent config, **seed 42 held constant across all four**, noise off. The four schedules differ **only** in `episode_boundaries` and `checkpoint_frequencies`. All four run **in parallel on node 114, GPUs 0–3** (one schedule per GPU).

**The four schedules** (S1 = Stage-1 span, etc.; spans are the per-stage deltas of the cumulative boundaries):

| Variant | Schedule file (`configs/continual/`) | `episode_boundaries` | S1 / S2 / S3 spans | Total eps | `checkpoint_frequencies` |
|---|---|---|---|---|---|
| **T1** (loosest) | `dreamer_srl_3stage_curric_T1.yaml` | `[3000, 13000, 163000]` | 3,000 / 10,000 / 150,000 | 163,000 | `[1500, 5000, 25000]` |
| **T2** | `dreamer_srl_3stage_curric_T2.yaml` | `[2000, 9000, 109000]` | 2,000 / 7,000 / 100,000 | 109,000 | `[1000, 3500, 20000]` |
| **T3** | `dreamer_srl_3stage_curric_T3.yaml` | `[1500, 6500, 71500]` | 1,500 / 5,000 / 65,000 | 71,500 | `[750, 2500, 13000]` |
| **T4** (most aggressive) | `dreamer_srl_3stage_curric_T4.yaml` | `[1000, 4000, 44000]` | 1,000 / 3,000 / 40,000 | 44,000 | `[500, 1500, 8000]` |

**Per-stage budget logic** (shared by all four; only the magnitude tightens from T1 to T4):
- **Stage 1** — the food + static-hazard 5×5 warm-up. Budget is set to the **~1,000-episode plateau plus a shrinking margin**: 3× margin (T1), 2× (T2), 1.5× (T3), down to no real margin (T4, 1,000 eps right at the plateau). This is the stage the live data says was over-budgeted, so it absorbs most of the tightening.
- **Stage 2** — the **threat-learning bridge** (slowed "bridge" chaser + bushes, still 5×5). Held proportionally larger than Stage 1 because its job is to feed the world model's continuation ("alive/dead") and reward heads enough *occasional, recoverable* death-terminal samples before the full task — the under-sampled signal the diagnosis blamed for the stuck policy.
- **Stage 3** — the **full 10×10 target task** and the measurement stage. Carries the bulk of the budget in every variant; the cap stays deliberately loose (runs are evaluated at matched environment-step budget, then stopped, well before the cap).

**Validation (all four files).** Each schedule was load-checked through PyYAML and verified against the schedule contract: `episode_boundaries` strictly increasing and length 3 (= the three stage configs); `checkpoint_frequencies` length 3, every entry > 0 and ≤ its stage span. **All four PASS** every check (T1 → 2/2/6 checkpoints per stage; T2 → 2/2/5; T3 → 2/2/5; T4 → 2/2/5).

**Pre-registered read.** The winner is **the smallest total-episode budget (i.e. the most aggressive of T1–T4) that still reaches the Stage-3 survival-step plateau without degrading versus the looser budgets** — concretely, whose Stage-3 final survival is statistically indistinguishable from (or better than) the next-looser variant's. If T4 holds the plateau, it sets the new minimum viable budget; if survival drops off at some Tn, the looser neighbour Tn−1 is the floor. This ablation fixes the curriculum's budget before the multi-seed curriculum-vs-from-scratch comparison (§2–§5) is launched at scale.

## Schema flags

**None.** Every entity in the locked spec is expressible in the current v2.0 schema with no new keys: food and the static hiding predator are `resources` entries, the chasing predator is a `predators[]` entry, the rabbit is a `neutral_animals[]` entry, and rock/bush are `obstacles` entries (bush via the existing `hides_agent: true` flag). All three stage configs load cleanly through `load_env_params` with the mandatory-key loader (no `ValueError`), and produce the identical 27-dim observation. No schema-affecting change is required, so no `developer` routing is needed for the env side. (The only code dependency is the **continual engine port** itself — the `--configs-dir` + `--continual-schedule` reader — which `senior-developer` owns.)

## 6. Failure-Mode Catalog (pre-decided)

- **Stage-1 stuck policy** (actor entropy stays at ln 6, survival flat ~30): reproduces the documented small-grid failure. This **refutes H1** for the curriculum-as-built, and is attributed to the warm-up-contamination risk the user accepted (hiding predator on from step 0), **not** to a bad run. Diagnostic: actor entropy + reward-head positive-MAE at end of Stage 1.
- **Catastrophic forgetting at a boundary** (survival collapses at Stage-2→3 and does not recover to baseline within the Stage-3 budget): **refutes H1**, logged as negative transfer. Distinguished from a normal boundary dip by the recovery-time analysis (§5).
- **Training instability** (NaN, value-head explosion, KL blow-up): **refutes the run, not the hypothesis** — re-launch the affected seed. If ≥ 2 of 5 seeds diverge, escalate as a stability finding (the agent config, shared with the baseline, may need attention) rather than scoring the experiment.
- **Saturation at the 500-step cap** on Stage 1 (episodes routinely hit max_steps): this is **expected and fine** on the benign stage — it is not a ceiling problem, it means the agent survives. Only a Stage-3 plateau well below 500 is the scientific outcome of interest.
- **Insufficient Stage-3 horizon** (both arms still rising at the cap, as the baseline parity cells were): if the curriculum and baseline curves have not plateaued, report the comparison **at matched env-step budget** and flag that the plateau question is horizon-limited — do not declare a null from non-converged curves.
- **Seed noise drowning a ~20-step effect**: the 5-seed design targets this; if the 95% CIs still overlap at a Δ near 20, the pre-registered verdict is **null** (not "promising, add seeds") unless the user explicitly authorizes a seed extension.

## 7. Intended launch command shape (depends on the in-flight code port)

The continual engine that reads `--configs-dir` + `--continual-schedule` is being ported by `senior-developer`; these commands are valid **once that port lands**. Curriculum arm (one per seed, seed swapped):

```
python train.py \
  --agent_config configs/models/dreamer_srl/01_food_only_buf256k.yaml \
  --configs-dir configs/experiment/dreamer_srl_curriculum \
  --continual-schedule configs/continual/dreamer_srl_3stage_size_curriculum.yaml \
  --num-envs 16 \
  --seed 42 \
  --device cuda:0 \
  --wandb-group dreamer_srl_curriculum \
  --wandb-job-type prod \
  --wandb-name dreamer_v3_dsrl_curric3_s42 \
  --tag        dreamer_v3_dsrl_curric3_s42
```

From-scratch baseline arm (single-stage, no `--configs-dir`):

```
python train.py \
  --agent_config configs/models/dreamer_srl/01_food_only_buf256k.yaml \
  --config configs/experiment/dreamer_srl_curriculum/03_10x10_full_task.yaml \
  --num-envs 16 \
  --seed 42 \
  --device cuda:0 \
  --wandb-group dreamer_srl_curriculum \
  --wandb-job-type prod \
  --wandb-name dreamer_v3_dsrl_fromscratch10x10_s42 \
  --tag        dreamer_v3_dsrl_fromscratch10x10_s42
```

(`training-runner` owns the real launch via `run_command.py` and the agent train script; the `python` here is illustrative of the argument shape. WandB project/entity are left unset so the repo defaults apply.)

## 8. Results / Analysis / Conclusions

### 8.0 Plain-language verdict (budget ablation T1–T4)

**What was tested.** Four copies of the same 3-stage curriculum agent (identical seed 42, identical configs, noise off) were trained side-by-side, differing **only** in how many episodes each stage was allowed to run before the world changed. The four budgets ranged from the loosest (**T1**, 163,000 total episodes) down to the most aggressive (**T4**, 44,000 total episodes), with **T2** (109k) and **T3** (71.5k) in between. The pre-registered question: *what is the smallest total budget that still lets the agent reach the hard 10×10 task's survival-step ceiling?* (Survival steps = how many of the 500-step episode the agent stays alive; higher is better.)

**Headline finding.** The smallest budget that **still reaches the survival ceiling is T3 — 71,500 total episodes**. T3 climbed to about **214 survival steps** on the final 10×10 task and was flat there, statistically indistinguishable from the looser budgets T1 (~210) and T2 (~221). The most aggressive budget, **T4 (44k), fell short at ~188 survival steps — but the curves show this is because T4 was cut off while still climbing, not because it hit a low ceiling**. T4's survival rose monotonically right up to its final episode (its very last segment was its highest), so 44k episodes is simply **too few to finish learning the hard task**, not a budget that converges to a worse answer. So: **T3 (71.5k) is the new minimum viable budget; T4 (44k) is under-trained.**

**Curriculum vs. starting on the hard task directly.** This is the experiment's real scientific question (§2). The from-scratch agent — the same architecture trained straight on the 10×10 task with no warm-up — plateaus at about **220 survival steps after roughly 50,000 episodes, and stays flat through 190,000 episodes**. The curriculum agents reach **the same ceiling (~215, within seed-noise of 220), not a higher one** — so the curriculum does **not** clear the pre-registered "≥ 20 steps higher" bar for a plateau win. **What the curriculum buys instead is sample efficiency**: the T3 budget reaches 200 survival steps in about **27,000 total episodes versus the from-scratch agent's ~50,000** — roughly **45% fewer episodes to the same competence**. The warm-up stages (where survival saturates at the 500-step cap within ~1,000–1,500 episodes because the small worlds are easy) cost little and front-load the world-model's reward and alive/dead heads with clean samples, so Stage 3 starts from a better place (~150 survival steps on the first hard-task episodes vs. the from-scratch agent's ~115).

**Caveat — two cells were stopped early, and this is a single seed.** T1 and T2 were terminated partway through their long Stage-3 budgets (~47% and ~84% through Stage 3 respectively), so their plateau numbers are read off curves that had already flattened but were not run to their own caps — they corroborate T3's plateau rather than independently confirm a higher one. The entire ablation is **seed 42 only**; it fixes the *budget* for the curriculum, and the multi-seed curriculum-vs-from-scratch comparison (§2–§5, the 5-seed paired design) is still the test that will deliver the confirmed verdict on H1. **World-model health is clean in every cell** (continuation-head accuracy ~0.998, reward-head error in a tight 0.70–0.82 band, no loss explosion) — in particular **T4 did not collapse**; it is genuinely under-trained, not broken.

### 8.1 Results

**Manifest of analyzed runs** (budget-ablation cells, not the §3 multi-seed manifest, which is still `planned`). All read from local `wandb/run-*/` binary logs via the datastore reader — no web API. Working file: `tmp/20260618_152500_curric3_budget_ablation.md`.

| Cell | WandB id | Schedule | Total eps | End-state | Stage-3 plateau (survival steps) |
|---|---|---|---|---|---|
| T1 | `p8cf3c8r` | `[3000, 13000, 163000]` | 163k | terminated ep ~82,960 (~47% through S3) | ~210 (flat, still inching up) |
| T2 | `tpt85vjj` | `[2000, 9000, 109000]` | 109k | terminated ep ~92,561 (~84% through S3) | ~221 (plateaued) |
| T3 | `k1os88e9` | `[1500, 6500, 71500]` | 71.5k | **completed** ep 71,501 | **~214 (plateaued)** |
| T4 | `l3uq9zyd` | `[1000, 4000, 44000]` | 44k | **completed** ep 44,000 | ~188 (**still climbing at cap**) |
| Baseline (from-scratch) | `w9as1qe3` | direct on `01-interoNocicept.yaml`, seed 42 | — | ran to ep 190,669 | ~220 (plateaued) |

**Per-stage survival (Episode/Steps), all cells.** Stages 1 and 2 (the benign 5×5 worlds) **saturate near the 500-step cap** in every cell — mean survival 440–490 — which is the *expected* outcome flagged in the failure-mode catalog (§6, "Saturation at the 500-step cap on Stage 1 … is expected and fine"). The scientific signal is entirely in Stage 3.

| Cell | Stage 1 mean | Stage 2 mean | Stage 3 first→plateau | Stage 2→3 boundary drop |
|---|---|---|---|---|
| T1 | 443 | 472 | 150 → 210 | −259 steps |
| T2 | 433 | 445 | 137 → 221 | −280 steps |
| T3 | 415 | 437 | 141 → 214 | −277 steps |
| T4 | 397 | 415 | 120 → 188 (rising) | −263 steps |

**Stage-3 plateau, binned (robust to single-episode noise).** Note: T3's *single last log-point* reads 183.6, but that is one noisy episode-average; the last 10% of T3's Stage-3 log binned together is **214**, flat. The plateau numbers below are binned means over the final ~7k Stage-3 episodes.

| Cell | Stage-3 plateau (binned) | Shape at termination |
|---|---|---|
| T1 | 210 | flat, marginal upward creep |
| T2 | 221 | flat (plateaued) |
| T3 | **214** | flat (plateaued) |
| T4 | 188 | **monotonic rise — not yet plateaued** |
| Baseline | 220 | flat (plateaued by ~ep 50k, stable to ep 190k) |

**Sample efficiency (total episodes to reach a survival threshold).**

| Cell | Total eps to 200 survival | Total eps to 210 survival |
|---|---|---|
| T1 | 58,784 | 67,267 |
| T2 | 43,197 | 48,102 |
| T3 | **27,123** | 47,592 |
| T4 | never (capped while climbing) | never |
| Baseline | 49,659 | 58,688 |

T3 reaches 200 survival steps in **~45% fewer total episodes** than the from-scratch baseline (27k vs 50k); even T2 reaches 210 in 48k vs the baseline's 59k.

**World-model health (plateau tail) — collapse check.**

| Cell | reward-head MAE | MAE (pos rew) | MAE (neg rew) | continuation acc | model loss | actor entropy |
|---|---|---|---|---|---|---|
| T1 | 0.749 | 0.243 | 1.045 | 0.997 | 2.873 | 0.182 |
| T2 | 0.700 | 0.235 | 0.970 | 0.998 | 2.983 | 0.245 |
| T3 | 0.716 | 0.232 | 0.994 | 0.998 | 2.841 | 0.169 |
| T4 | 0.817 | 0.218 | 1.186 | 0.998 | 2.714 | 0.157 |
| Baseline | 0.658 | 0.216 | 0.903 | 0.998 | 2.758 | 0.234 |

No collapse in any cell. Continuation ("alive/dead") accuracy is ~0.998 everywhere. T4's slightly higher negative-reward MAE (1.19) and lower actor entropy are consistent with it being **earlier in training** (the reward-head's heavy-negative death tail is less converged), not with a broken model.

**Final-task termination cause.** `Episode/Term_MaxSteps` fraction (the share of episodes that survive to the 500-step cap) is only **0.04–0.09** in every cell — the agent at plateau survives ~200 steps but rarely the full 500. Final per-episode reward is **−312 to −317** across all cells including the baseline (reward is a secondary diagnostic here, not the headline; survival steps are the metric).

### 8.2 Analysis

**Temporal evolution (mandatory).** Every cell follows the predicted shape (§2): survival rises fast on Stage 1, saturates at the cap through Stages 1–2, **drops sharply (~260–280 steps) at the Stage-2→Stage-3 boundary** as the world jumps from benign 5×5 to the full 10×10 task, then climbs back over the Stage-3 budget. Stage-3 climbs:

- **T1** (longest S3 budget): 150 → 171 → 187 → 204 → 210, flattening in the last two bins. Killed at ~47% of its Stage-3 budget but already at the plateau band.
- **T2**: 137 → 178 → 208 → 214 → 221, plateaued in the last three bins. Killed at ~84%, clearly converged.
- **T3** (completed): 141 → 197 → 189 → 201 → 211 → 214, flat at the end. **Reaches the plateau within its budget.**
- **T4** (completed): 120 → 138 → 159 → 170 → 176 → 188, **strictly increasing, last bin highest**. T4 ran out of Stage-3 budget *before* the curve turned over.

This temporal read is what overturns the surface impression that "T4 underperforms at ~183". T4's low number is **horizon-limited (under-trained)**, exactly the "Insufficient Stage-3 horizon" failure mode in §6 — its curve had not plateaued, so its endpoint must not be read as a converged plateau. The honest statement is: *44k episodes is not enough for this curriculum to finish the hard task*, and the smallest budget that **is** enough is T3's 71.5k.

**Budget-ablation verdict (pre-registered read, §"Budget ablation").** The rule was: the winner is the smallest total budget whose Stage-3 final survival is statistically indistinguishable from (or better than) the next-looser variant. T3 (214) vs T2 (221) and T1 (210) are within ~10 steps, i.e. inside the seed/episode noise band (plateau-tail SD ≈ 11–13 steps) → **T3 holds the plateau**. T4 (188, and still rising) is **not** indistinguishable from T3 — it is ~26 steps lower and non-converged → T4 drops below the floor. **Floor = T3 (71,500 episodes).**

**Curriculum vs. from-scratch (the §2 success criterion).** Held against the pre-registered confirmation bar — curriculum final survival ≥ 20 steps above baseline with non-overlapping CIs:

- **Plateau height: NOT met.** Curriculum plateau (~214 for T3, ~210–221 across cells) is **within seed-noise of the from-scratch ~220**, not ≥ 20 steps above. On the height criterion this is the §2 *null* outcome — "the warm-up bought nothing extra in final survival."
- **Sample efficiency: a clear win** (not the pre-registered metric, but the operative finding). T3 reaches 200 survival in ~27k total episodes vs the baseline's ~50k (~45% fewer), and the curriculum agent enters Stage 3 at ~150 survival vs the from-scratch agent's ~115 at matched early episodes — the warm-up front-loads the world-model's reward and continuation heads so the hard-task climb starts higher and converges sooner.

So the curriculum's value is **"same ceiling, reached faster,"** i.e. **(c) fewer total episodes** of the three options posed in the brief, **not (a) higher**. This is a meaningful efficiency result but it does **not** satisfy the doc's pre-registered "higher plateau" definition of success — and it is established on a **single seed**, so it is a *budget-setting* result that motivates the 5-seed comparison, not a confirmed answer to H1.

**Boundary / catastrophic-forgetting check (§6).** The Stage-2→3 drop is large (~270 steps) but it is a *difficulty-jump* drop (the task genuinely got harder — 5×5 benign → 10×10 with four hiding predators, a full-strength chaser, and rabbits), not catastrophic forgetting: survival **recovers above the pre-boundary baseline-equivalent and climbs to the from-scratch ceiling** within the Stage-3 budget for T1/T2/T3. No "collapse-and-fail-to-recover" signature → catastrophic forgetting **not** triggered.

**Task-matching caveat.** T3 and the baseline share seed 42, agent config (`01_food_only_buf256k.yaml`), and `--num-envs 16`. The baseline trained on `configs/experiment/hypervigilance/01-interoNocicept.yaml`; the curriculum's Stage 3 used `03_10x10_full_task.yaml`. The design doc asserts (Modality-fingerprint section) these two are byte-value-identical on the env body/sensory/noise blocks and identical task entities. I did **not** re-diff the two YAMLs in this analysis; the comparison rests on that designer-verified claim. If the byte-identity ever lapsed, the plateau comparison would need re-checking.

### 8.3 Conclusions

1. **Budget floor = T3 (71,500 episodes).** T3 reaches the Stage-3 survival plateau (~214 steps) within budget and is statistically indistinguishable from the looser T1/T2. This is the new minimum viable curriculum budget — a ~10× reduction from the original 760k-episode schedule and well below the loosest ablation arm.
2. **T4 (44k) is too tight — by under-training, not by a structural ceiling.** Its survival curve was still rising monotonically at termination (last segment highest, ~188). 44k episodes does not give Stage 3 enough horizon to converge; this is the §6 "Insufficient Stage-3 horizon" failure mode, not a model defect. World-model health confirms T4 is sound (continuation acc 0.998, no loss blow-up), just early.
3. **On H1 (§2): the height bar is not cleared on a single seed.** The curriculum reaches the **same** ~220-step plateau as the from-scratch baseline, not ≥ 20 steps higher → the pre-registered *plateau-height* criterion reads **null** here. The genuine win is **sample efficiency** — the same competence in ~45% fewer episodes (T3 ~27k vs baseline ~50k to 200 survival), with a higher Stage-3 starting point from the warm-up. This is the operative motivation for the curriculum, but it is **not** the doc's pre-registered definition of success and rests on **seed 42 only**.
4. **This ablation does its job: it fixes the curriculum budget (T3, 71.5k) before scale-up.** The confirmed answer to H1 still requires the **5-seed paired curriculum-vs-from-scratch comparison** (§3 manifest, all rows currently `planned`). That comparison should now adopt the **T3 schedule** as the curriculum budget and be read against both criteria — the pre-registered height bar **and** an episodes-to-plateau efficiency comparison (recommend pre-registering the latter, since it is where the effect actually lives).
5. **No failure modes triggered** beyond the benign expected ones: Stage-1/2 cap-saturation (expected, §6), and T4's horizon-limit. No stuck policy (actor entropy fell well below ln 6 ≈ 1.79 — all cells at 0.16–0.25), no catastrophic forgetting, no world-model collapse, no training instability.

**Status note.** This is a **single-seed budget ablation**, not the multi-seed H1 test. The §3 manifest remains the system-of-record for the confirmed curriculum-vs-from-scratch verdict and is untouched by this analysis.

### 8.4 Recommendations / follow-ups

- **Adopt the T3 schedule** (`configs/continual/dreamer_srl_3stage_curric_T3.yaml`, 71.5k total) as the curriculum budget for the 5-seed §3 launch.
- **Pre-register an episodes-to-plateau (sample-efficiency) endpoint** alongside the existing plateau-height criterion in §5, since the single-seed data says the effect lives in efficiency, not ceiling height.
- **Optionally re-run T4 (44k) to higher budget on one seed** only if confirming the "under-trained, not low-ceiling" reading is wanted before committing — the temporal curve already shows this clearly, so this is low priority.

