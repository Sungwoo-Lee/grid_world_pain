---
title: "Evaluation-Path Correctness Diagnosis (independent bug hunt)"
topic: issues
status: active
created: 2026-07-04
last_updated: 2026-07-06
---

# Evaluation-Path Correctness Diagnosis

## Purpose (plain-language entry point)

This document is an independent bug hunt over the project's **evaluation path** — the code
that takes a saved training checkpoint, rebuilds the environment and the agent, replays
episodes, and produces the numbers (survival steps, rewards), CSV stats, recordings, and
videos that the project's conclusions rest on. A prior audit already fixed several bugs in
this exact area (wrong-stage environments, crashing model rebuilds, silent half-random
restores, invisible noise in videos); this pass verifies those fixes held and hunts for
anything they missed. The verdict up front: **the recent fixes are correct and complete for
their stated scope**, and the core numbers (survival steps and rewards from the standard
eval paths) are sound. But the hunt found one significant new bug: **when the "interoceptive
nociception" body-pain sensor is enabled (it is on by default), the per-step evaluation CSV
writes every sensor column under the wrong name** — a silent one-column shift that makes any
per-sensor analysis of those CSVs (e.g. the noise-diagnostics tool) quietly wrong. Several
smaller issues follow: an off-by-one survival count in one corner of the parallel evaluator,
a random-key-reuse bug that only bites if stochastic evaluation is ever switched on, two
evaluation config knobs that are validated but never actually used, and decode mistakes in
the qualitative trajectory-story tool.

Scope reviewed: `src/utils/evaluation_core.py`, `scripts/eval/eval_rollout.py`, root
`evaluation.py`, `src/utils/eval_recording.py`, `src/utils/visualization.py`,
`src/utils/episode_logging.py`, `src/environment/renderer.py` (+`renderer_v2.py` usage),
`scripts/eval/trajectory_story.py`, `scripts/eval/render_recordings.py`, plus the callers in
`train.py` and `main.py` and the noise machinery in `src/environment/sensor.py`.

---

## Finding 1 — Eval stats CSV: silent one-column shift when interoceptive nociception is enabled

**Severity: High (silently wrong analysis numbers; encode/decode-drift class) — NEW**

- `src/utils/evaluation_core.py:204-222` (obs headers) and `:226-243` (true_* headers)

`_write_episode_stats` builds CSV headers by iterating `get_observation_breakdown(params)`
and emitting a header per sensor — but the branch list handles `"Satiation"`, `"Nutrition"`,
`"Injury"`, `"Extero Nociception"`, `"Olfaction"`, `"Collision"`, `"Location"`, `"Visual"`,
`"Proprioception"` and **omits `"Interoceptive Nociception"`**, which
`get_observation_breakdown` (`src/environment/sensor.py:363-364`) emits between Satiation and
Extero Nociception whenever `interoceptive_nociception_enabled` is true — and it is **true by
default** (`configs/environment/default.yaml:217`).

What happens: the header list is one column shorter than the observation vector, and row
values are written **positionally** (`evaluation_core.py:109-111`:
`for i in range(min(num_obs_headers, len(obs_vec))): row.append(obs_vec[i])`). The CSV stays
structurally valid (no error, column counts match), but from the interoceptive-nociception
slot onward every value sits under the previous sensor's name:

- `obs_intero_satiation` → correct (satiation)
- `obs_noc` → actually holds **interoceptive nociception**
- `obs_olf_0` → actually holds **extero nociception**
- `obs_olf_1..7`, `obs_coll_*`, … → all shifted one left
- the last element of the observation vector is silently dropped.

The `true_*` block has the identical omission, so noise-diagnostics comparisons
(`scripts/verification/analyze_noise_diagnostics.py:19-22` pairs `obs_noc`↔`true_noc` etc.)
compute internally consistent but **mislabeled** per-modality noise statistics — e.g. the
"Extero Nociception" noise row is really the interoceptive channel, and the olfaction sigma
estimate blends nociception into it.

Concrete failure scenario: any training run on the default environment with
`stats_during_training: true` (the standard in-training stats pass, `train.py:2509-2515`)
writes shifted CSVs; anyone reading a per-sensor column — noise diagnostics, ad-hoc pandas
analysis, the `action_scatter` pipeline if it ever touches obs columns — gets wrong numbers
with no error. This is a live recurrence of the project's documented silent encode/decode
drift class (KNOWN_BUGS "Silent encode/decode layout drift", "Noise painted on wrong sensory
channel") — but this specific instance is **not** in the registry. Note `build_sensory_viz`
(`sensor.py:432-441`) handles the sensor correctly, so **videos are fine; only the CSVs are
shifted**.

Suggested fix: add `"Interoceptive Nociception"` to both header loops (it maps naturally to
`obs_intero_nociception` / `true_intero_nociception`), or better, generate headers
generically from the breakdown so a future sensor cannot repeat this.

---

## Finding 2 — Parallel eval: survival steps undercounted by 1 and recordings missing frame 0 when stats recording is off

**Severity: Med (survival steps are the project's primary metric; only one caller combination hits it) — NEW**

- `src/utils/evaluation_core.py:534` (step-0 seeding gated on `record_stats`)
- `src/utils/evaluation_core.py:578-596` (per-step buffer appends, unconditional)
- `src/utils/evaluation_core.py:606` (`episode_lengths.append(len(slot_rewards[i]) - 1)`)
- `src/utils/evaluation_core.py:663-675` (refill re-seeds step-0 **unconditionally**)

The parallel ("episode-ticket") evaluator seeds each slot's buffers with a step-0 placeholder
(`action=-1, reward=0.0`) **only when `record_stats` is true**, but per-step appends always
happen, and the episode length is always computed as `len(slot_rewards) - 1` (assuming the
placeholder exists). When a finished slot is refilled mid-run, the new episode's buffers get
the step-0 placeholder **unconditionally**.

Consequences when `record_stats=False` and `num_envs > 1`:

1. **First-wave episodes undercount survival by exactly 1 step**; refilled episodes count
   correctly — an inconsistent mix inside one eval result.
2. If `render_video=True` too, first-wave recordings **lack the initial frame**: rendered
   frame "step 0" is actually the state after the first action, labeled with a real action
   instead of the `None` sentinel; refilled episodes include frame 0 — visibly inconsistent
   videos, and a genuine frame-vs-stated-step off-by-one.

Live caller: `main.py:92-103` (sandbox random-agent video, `record_stats` defaults to
`config.get('testing.record_stats', False)` with **no** eval-defaults merge, `num_envs` from
`training.num_envs`). The standard paths are immune: train.py's video pass is single-env, its
stats pass has `record_stats=True`, and `evaluation.py` merges `configs/evaluation/default.yaml`
which forces `record_stats: true`. Suggested fix: seed the step-0 placeholder unconditionally
at initial reset (matching the refill path), which makes the `-1` always correct and the
recordings uniform.

Related nit: when neither stats nor video is requested, the parallel path still accumulates
full per-step state buffers per slot (memory only, no correctness impact).

---

## Finding 3 — eval_rollout: PRNG key reused every step (stochastic mode only)

**Severity: Med (dormant today — deterministic is the default — but primed to bite) — NEW**

- `scripts/eval/eval_rollout.py:98` and `:201` (`policy_fn(state, carry, rng_key, ...)` inside the step loop)
- `scripts/eval/eval_rollout.py:79/:176` (the same `rng_key` used for `jax_reset`)

`_run_episode` / `_run_episode_with_recording` pass the **same, never-split episode key** to
`policy_fn` on every step. In deterministic mode the key is discarded (argmax), so today's
default (`behavior_measures.eval_policy_mode: deterministic`,
`configs/environment/default.yaml:243`) is unaffected. But in `stochastic` mode
`get_action_and_value_nnx(..., key=key)` samples with the **identical key each step** (and
the same key already consumed by `jax_reset`): action sampling degenerates to the same
uniform draw applied to each step's logits — heavily correlated, near-deterministic
trajectories masquerading as stochastic ones, violating the project's PRNG convention (never
reuse a sub-key; always advance the main key).

This matters now because the fresh memory insight
`20260704_2014_deterministic_probe_significance_inflates` concludes that near-deterministic
probes inflate p-values and effect sizes — the natural remedy (switch the probe to
stochastic eval) would silently produce *still*-near-deterministic rollouts through this bug,
reproducing the exact statistical artifact the switch was meant to fix.

Suggested fix: thread the key (`key, sub = jax.random.split(key)` per step) through the
rollout loops and pass `sub` to the policy.

---

## Finding 4 — `behavior_measures.eval_obs_noise` and `eval_max_steps` are validated but dead

**Severity: Med (silent config no-op; the eval-noise knob directly touches the frozen-probe protocol) — NEW (known *class*: "dead config key silently ignored")**

- `src/environment/config_loader.py:172,218-219` (parsed + strictly validated: `"training" | "zero" | "custom"`)
- `scripts/eval/eval_rollout.py:854` (written into `metadata.json`), but **no code path acts on it**
- `scripts/eval/eval_rollout.py:572` (`max_steps` read from `environment.max_steps`, never from `bm_cfg.eval_max_steps`)

`eval_obs_noise` is loaded with `get_mandatory`, rejected if not one of three allowed values,
recorded into eval metadata — and then completely ignored: `policy_fn`
(`eval_rollout.py:750`) always calls `get_observation(state, params)` with training noise.
An experimenter setting `eval_obs_noise: zero` to run a noise-free probe gets a noisy
environment, with metadata.json *asserting* the noise mode they asked for. Note the memory
insight `frozen_probe_eval_match_sensory_renderer` establishes "training" as the correct
default behavior — the default is right; the knob lying about the alternatives is the bug.
`eval_max_steps` is the same pattern: parsed and validated, never consumed (the env's
`max_steps` always wins). Suggested fix: either implement both knobs (thread
`apply_noise=False` when `zero`) or delete them from the schema; do not leave validated
no-ops.

---

## Finding 5 — trajectory_story: step-0 action decodes as "Eat" and the action column lags one step

**Severity: Med-Low (qualitative story reads only; actively used via the trajectory-story skill) — NEW**

- `scripts/eval/trajectory_story.py:136` (`AM[int(acts[t])]` in `cmd_dump`) and `:196` (`cmd_obs`)

The recording schema (written by both `evaluation_core.py` and `eval_rollout.py`) stores
`actions[0] = -1` as a "no action yet" sentinel for the initial frame, and `actions[t]` (t≥1)
as the action that **produced** snapshot t. `render_recordings.py:70` decodes this correctly
(`action_t = None if < 0`). `trajectory_story.py` does not:

1. At t=0, `AM[int(-1)]` **wraps to the last element** of the action map — the step-by-step
   dump shows the agent performing "Eat" (or "Left" if eat is disabled) at step 0 of every
   episode. Fabricated data in the microscope tool.
2. For t≥1, the dump prints `acts[t]` next to `A[t]` (the agent's position at t) under the
   header "act" — but `acts[t]` is the move that *led into* position t, not the action taken
   there. A story reading "at (3,4) the agent moved Up" is off by one step; the action taken
   *at* `A[t]` is `acts[t+1]`. This is exactly the recorder-writes-one-thing /
   consumer-decodes-another drift class flagged in the hunt brief.

---

## Finding 6 — trajectory_story summary: survival overcounted by 1, hardcoded death threshold, misleading %reach-max

**Severity: Low (summary printout of a qualitative tool) — NEW**

- `scripts/eval/trajectory_story.py:87` — `lens.append(T)` where `T = len(snapshots)` = steps + 1
  (includes the initial frame), so the printed "survival: mean/median" **overcounts survival
  steps by exactly 1** relative to the recorder's own `length` field and the project's
  survival-step convention elsewhere.
- `:92` — deaths counted via `inj[-1] >= 99`, hardcoding `max_injury≈100`; wrong for any env
  where max_injury differs, and misses starvation deaths entirely (label says "injury-death
  eps", but a reader skims it as deaths).
- `:101` — `%reach-max=(lens>=lens.max()).mean()` compares against the **batch's own longest
  episode**, not `max_steps`; if no episode reaches the limit this still prints a nonzero
  "reach-max" fraction.

---

## Finding 7 — render_recordings: `--skip-existing` + `--concat` drops episodes from the consolidated video; `--cleanup-per-episode` then deletes the only copies

**Severity: Low-Med (video completeness on re-runs; the exact flag combo evaluation_core auto-render uses) — NEW**

- `scripts/eval/render_recordings.py:118-127` (tasks exclude skipped episodes; early return when all exist)
- `:146-158` (`frame_generator` iterates **tasks**, not `episode_files`)
- `:163-169` (cleanup deletes per-episode MP4s for **all** `episode_files`)

The consolidated `eval_<pct>.mp4` is concatenated only from the episodes rendered *in this
invocation*. `evaluation_core.py:303-310` invokes exactly `--concat --skip-existing
--cleanup-per-episode`. Failure scenario: an auto-render dies partway (e.g. worker OOM — the
recordings are preserved and the user re-runs), the re-run skips the already-rendered
episodes, so the consolidated video **silently omits them**, and cleanup then deletes every
per-episode MP4 — including the omitted ones, whose frames now exist nowhere. If *all*
episodes were already rendered, the script early-returns before `--concat`, so no
consolidated file is produced at all and `evaluation_core.py:322` silently skips the WandB
upload. Suggested fix: build the concat list from `episode_files` (reading existing MP4s),
not from `tasks`.

---

## Finding 8 — evaluation.py: `wandb_login` is undefined — `--wandb-run-path` silently never connects

**Severity: Low (deprecated script; silent feature loss, not a crash) — NEW**

- `evaluation.py:309` calls `wandb_login(quiet=True)`, which is **never imported** (the file
  imports `wandb` but not `wandb_login` from `src.utils.wandb_utils`). The resulting
  `NameError` is swallowed by the surrounding `except Exception` (`:315-316`), printing
  "WandB init failed: name 'wandb_login' is not defined" and continuing without WandB — so
  `--wandb-run-path` can never upload from this script.

---

## Verification of the prior fix cluster (all four verified)

| Fix | Verdict |
|---|---|
| **a3ab4cc** — eval_rollout resolves the checkpoint's own stage | ✅ Correct and conservative. Fires only when `schedule.yaml` exists AND `--config` is the run's own stage-0 `config.yaml`; reads the checkpoint's saved `stage` field (not a boundary recompute); explicit `--config` always wins; out-of-range / missing-stage cases raise. No regression found. |
| **863052f** — evaluation.py `--all` per-checkpoint stage | ✅ Correct. Restores the full payload first, reads `restored['stage']`, loads the matching `stage_XX_*.yaml` (stage dumps are fully resolved at train time — `train.py:615-619` — so the raw `yaml.safe_load` at `evaluation.py:353` is safe re: `extends:`). One inconsistency: the `_load_eval_config` helper's docstring (`:130-134`) claims per-checkpoint stage configs are "prepared identically" through it, but the `:353` call site loads raw. Materially harmless today (eval defaults carry only `testing.*` keys and the stage config feeds only `load_env_params`), but the helper and call site should agree before the difference grows teeth. Nit, not a regression. |
| **2ad9104** — model-rebuild key list + Dreamer object type + restore-completeness assertion | ✅ Correct. Both rebuild sites read the whole `agent.modulation` dict (`evaluation.py:381-383, 421-423`); Dreamer is built via `DreamerTrainer(..., Config, obs_breakdown, modulation_config)` and `trainer.agent` satisfies `generic_inference`'s `(logits, value, h_new, mod_info)` contract (`dreamer_v3_nnx.py:679-735`, incl. `initial_state`). `_assert_full_restore` is present at both restore sites and fed by `_merge_restored_into_module_state`. Residual (minor, not a regression): the merge checks *coverage* but not *shapes* (eval_rollout's restore does check shapes), and extra checkpoint keys are silently ignored — only reachable with a hand-edited config. |
| **80d3b70** — eval video records the clean true observation | ✅ Correct for all practical paths. `record_true_obs` gates on `render_video and params.perceptual_noise_enabled` (`evaluation_core.py:186-188`); the single-env recorder gets `true_obs` at step 0 and every step; the parallel path computes it per slot per step. Noise is a pure function of `state.key` (`sensor.py:294` `fold_in(state.key, 999)`), so recorded noisy obs are bit-identical to what the policy saw — no double-draw drift. Only gap: the Finding-2 corner (parallel + `record_stats=False`) omits the step-0 true-obs entry along with everything else. |

Also checked, no issue found: **frozen-probe env match** (eval_rollout's policy consumes
`get_observation` with the training noise profile from the same stage params — matches memory
`frozen_probe_eval_match_sensory_renderer`); **hidden-state hygiene** (h reset per episode in
all paths; parallel refill resets the slot's h via `initial_state(1)`); **auto-reset
boundaries** (`ParallelEnv.step` does not auto-reset — `wrapper.py:23-33` — so no metric ever
spans a reset; the ticket design segments buffers correctly, and terminal observations are
genuine terminal obs, not reset obs); **survival-step edge cases** (the death step counts as
a survived step; truncation at `max_steps` counts fully; single-env and stats-pass parallel
agree — the only divergent counters are Finding 2's corner and Finding 6's +1); **seed
comparability** (same seed → same episode key sequence across checkpoints);
**renderer_v2.py** (imported nowhere in `src/`/`scripts/` — not part of the eval path);
**episode_logging.py** (straight lift of train.py's WandB fan-out; NaN-skipping means are
consistent with the guarded ratio metrics). Cosmetic: `renderer.py:619` `f"{step or '--'}"`
displays `--` for step 0; the intero fallback at `renderer.py:592-601` reads
`state.nociception_history_buffer`, which recording snapshots don't carry — currently
unreachable from eval (sensory_data always supplied) but a latent AttributeError if anyone
renders a snapshot without sensory_data. DreamerV3 eval samples its latent with a fixed
`PRNGKey(0)` every step (`dreamer_v3_nnx.py:710-711`) — intentional-looking determinism,
noted for awareness.

Not re-reported (known): L4 CLI model-size flags absent from saved config → wrong-size
rebuild on re-eval; M1/M2 episode-end rule (undecided); `evaluation.py` being
deprecated/demo-only in general.

---

## Verdict

**The evaluation path's core loop is sound and the recent fix cluster held** — stage
resolution, model restore, noise-consistent observations, terminal-obs handling, and
survival counting on the standard paths are all correct. The area is **not fully clean**,
though: the interoceptive-nociception CSV column shift (Finding 1) is a live, silent,
default-on data-corruption bug in the eval stats artifact and should be fixed before any
per-sensor CSV analysis is trusted; Findings 2-4 are pre-armed traps on paths one
configuration flip away from live; Findings 5-7 degrade the qualitative tooling that the
team explicitly leans on when aggregates look suspicious.

Reviewed by: code-reviewer (Fable 5 independent diagnosis pass, 2026-07-04)

---

## Fix plan pointer + refinement (appended 2026-07-06 by senior-developer)

**Finding 1 (H8) has an approved fix plan**:
[[fix_plan_h8h9_eval_output_correctness]] (WP-E, covers H8 + 07's H9).

**Refinement to Finding 1, ground-truthed against real run CSVs during planning**: the
claim "the CSV stays structurally valid (no error, column counts match)" holds only for
envs **without obstacle entities**. Obstacle headers are named `obs_entity_{i}_r/c` and
match the `startswith("obs_")` predicate in `_write_episode_stats`
(`evaluation_core.py:74`), inflating `num_obs_headers`; combined with the missing
intero-noc header, obstacle-bearing envs (including the **default** env) write the FULL
observation vector under a header one name short — each data row is one field **longer**
than the header (verified: header 171 / row 172 on
`results/JAX_RecurrentPPO/20260703-154633_rppo_basic05v02_relentstam_n108/stats/4200006/`),
so `pandas.read_csv` misassigns **every** column via an implicit index, not just the
sensor block. Obstacle-free envs corrupt as originally described (same-width shift, last
element dropped; verified header 83 / row 83 on the 20260627 film_g4 run). Both modes and
their old-file remap recipes are documented in the fix plan's consumer analysis.
