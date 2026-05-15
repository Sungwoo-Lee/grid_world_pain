---
title: "Metric-schema migration plan: dreamer-srl → unified grid_world_pain schema"
topic: dreamer
status: active
created: 2026-05-15
last_updated: 2026-05-15
---

# Metric-schema migration plan: dreamer-srl → unified grid_world_pain schema

> **Companion audit:** the read-only inventory of every metric currently
> logged by the three drivers — and the gap analysis this plan closes —
> lives at [`METRIC_SCHEMA_AUDIT.md`](METRIC_SCHEMA_AUDIT.md) (commit
> `3aadca7`). Read §3 (master table), §5 (BM toolkit), §7 (recommended
> unified schema), and §8 (concrete edit list) for the context this plan
> assumes.
>
> **Implementation owner:** the [`developer`](.claude/agents/developer.md)
> agent will execute this plan one commit-chunk at a time. The
> [`senior-developer`](.claude/agents/senior-developer.md) verifies after
> each chunk lands. **No implementation happens inside this plan
> document.**

---

## 1. Context (plain-language entry point)

**What this plan is.** A 7-commit, chunked migration that takes the newly
parity-validated dreamer-srl driver (the freshly rebuilt Dreamer-V3 port
under `src/algorithms/dreamer_srl/`, currently logging to its own Weights
& Biases dashboard folder `grid_world_pain_dreamer_srl_smoke` with
sheeprl-style key names like `Game/ep_len_avg`, `Rewards/rew_avg`,
`Loss/observation_loss`) and **merges it into the master `grid_world_pain`
project** so its runs sit side-by-side with the original JAX Dreamer-V3
and Recurrent-PPO runs on the same dashboards.

**What that requires.** Two things must change in lockstep:

1. **dreamer-srl gains ~70 WandB keys it currently does not log.** The
   master project's dashboards live on **24 fixed `Episode/*` keys** (the
   survival-step + behavior-event family — `Episode/Steps`,
   `Episode/Reward`, `Episode/FoodEaten`, `Episode/PredatorHits`,
   `Episode/MeanDistFood`, `Episode/Term_MaxSteps`, etc.) plus a
   **per-tag fan-out** of distance metrics (one extra key per predator
   tag and per neutral tag declared in the env config) plus the **~30-key
   Behavior-Measure toolkit** (per-class + per-tag interrupted-feeding,
   bush-dive, and eat-under-threat-ratio measures used by the project's
   hypervigilance + sameprop experiments). On top of that, the dreamer-srl
   trainer's internal loss dict only returns ~10 keys; the master Dreamer
   trainer returns ~25 (the extra ones are `WorldModel/*` quality probes
   like reward-MAE and continue-head accuracy, and `Behavior/*` rollout
   statistics like imagined-return mean and value-MAE). dreamer-srl must
   match.
2. **The episode-aggregation rhythm has to flip.** dreamer-srl currently
   writes one WandB row per done env at the moment of episode end (sheeprl
   convention); the master project's dashboards expect one row per
   `log_interval` iterations averaged over every episode that finished in
   that window (Dreamer convention). The user picked **per-iteration
   mean** so the new dreamer-srl runs are visually drop-in compatible
   with the existing JAX Dreamer-V3 panels.

**Symmetric speed metric.** Once dreamer-srl is on the unified schema, the
plan also adds a single new key — `Time/sps_env` (steps-per-second
throughput in env-step units) — to **both** the original JAX Dreamer
branch and the Recurrent-PPO branch of `train.py`. Throughput is currently
only readable on the dreamer-srl runs; the user wants it on all three
drivers so the project's training-speed dashboards work uniformly.

**User's binding choices (from the audit's §9 Open Questions, all
answered).** **Q1** = full BM toolkit port (~200 LOC). **Q2** = full
per-tag fan-out for the distance keys (~30 LOC). **Q3** = per-iteration
mean aggregation (the Dreamer pattern, not the sheeprl-raw pattern).
**Q4** = port the extra `WorldModel/*` + `Behavior/*` trainer-side keys
(~40 LOC + 1) **but defer the eval rollout loop** to a separate future
plan. **Q5** = yes, `Time/sps_env` lands in **all three** drivers, not
just rPPO.

**Estimated total scope.** ~280 LOC of additions, almost all in
`src/algorithms/dreamer_srl/dreamer_srl_main.py` (the driver), with ~40
LOC inside the dreamer-srl trainer (`src/algorithms/dreamer_srl/train.py`)
and 2 single-line edits in `train.py` (the master driver, for the
`Time/sps_env` symmetry). The plan lands in **7 separate commits** —
chunked so a network drop or harness timeout never costs more than one
commit's worth of work. Each commit is independently verifiable (pytest
+ offline_check + a short smoke run).

---

## 2. Implementation order — 7 chunked commits

Each commit below is **self-contained, separately verifiable, and small
enough that pytest + offline_check + a 200–2000-iter smoke run cover it
in under ten minutes**. If any chunk regresses, revert only that single
commit (see §6 Rollback). Do not cascade further chunks onto a broken
foundation.

### Commit 1 — WandB project switch + minimal key renames

**Goal.** Land the smallest blast-radius change first: dreamer-srl now
writes into the master `grid_world_pain` WandB project, and its two
top-level episode keys (`Game/ep_len_avg` and `Rewards/rew_avg`) are
renamed to `Episode/Steps` and `Episode/Reward` respectively. **Keep the
existing raw-per-episode aggregation in this commit** — the per-iteration
switch happens in Commit 2. This isolates the dashboard-routing change
from the aggregation-semantics change so each can be debugged in
isolation if either misbehaves.

**Files touched.**

- `configs/logger/wandb.yaml` — read-only check. Already says
  `project: "grid_world_pain"` (audit §1, file L5). No edit needed; the
  developer should confirm this still holds at start of the commit and
  abort with a note if it does not.
- `src/algorithms/dreamer_srl/dreamer_srl_main.py`:
  - **L175–L177** (argparse `--wandb-project` default): change
    `default="grid_world_pain_dreamer_srl_smoke"` →
    `default="grid_world_pain"`.
  - **L307–L329** (wandb.init): add explicit `entity="sungwoolee"`
    matching `configs/logger/wandb.yaml:L4` so the new dreamer-srl runs
    land under the same user account as the existing JAX Dreamer runs.
  - **L455** (per-episode log): rename keys —
    ```python
    # BEFORE:
    wandb.log({"Game/ep_len_avg": ep_len, "Rewards/rew_avg": ep_rew},
              step=policy_step)
    # AFTER:
    wandb.log({"Episode/Steps": ep_len, "Episode/Reward": ep_rew},
              step=policy_step)
    ```
    (Aggregation is still raw-per-done-env at this commit; that switches
    in Commit 2.)

**Verification.**

- `pytest tests/algorithms/dreamer_srl/ -q` must remain **49/49 PASS**.
- The dreamer-srl offline check (the parity self-test that lives under
  the same tests folder, count of 17 inside the v2-CP10 manifest) must
  remain **17/17 PASS**.
- A 200-iter smoke run on
  `configs/experiment/dreamer_curriculum/01_food_only.yaml` with
  `--wandb-name commit1_smoke` should produce a run inside the
  `grid_world_pain` project — confirm via `wandb-summary.json` or the run
  URL printed at startup.

### Commit 2 — Per-iteration episode aggregation (audit Q3)

**Goal.** Switch dreamer-srl's episode logging from "one row per done env
at env-step time" to "one row per `log_interval` iterations averaging
over every episode that finished in the window." This is the Dreamer +
rPPO convention encoded in `train.py:L1394–L1437` (rPPO) and
`train.py:L1710–L1753` (Dreamer); the dreamer-srl driver becomes
structurally parallel to those blocks.

**Files touched.**

- `src/algorithms/dreamer_srl/dreamer_srl_main.py`:
  - **After L370** (after `log_every` is computed): add the buffer
    `iteration_episodes: list[dict] = []`. (Mirrors `train.py:L1062`,
    where the same buffer is initialized in the master driver.)
  - **L451–L458** (the done-env episode-log block — currently calls
    `wandb.log` directly): rewrite so the per-env episode rolls into the
    buffer instead of logging. Replace with:
    ```python
    for i in dones_idxes:
        ep_len = int(episode_lengths[i]) + 1
        ep_rew = float(episode_rewards[i]) + float(rewards[i])
        ep_data = {'r': ep_rew, 'l': ep_len}
        iteration_episodes.append(ep_data)
        if not args.quiet:
            print(f"[iter {iter_num}] episode done: env={i} ep_len={ep_len} ep_rew={ep_rew:.3f}")
    ```
    The `total_episodes_completed` counter must also be tracked in the
    driver scope — initialize at `0` near L364 and increment by
    `len(dones_idxes)` inside the dones block. (Mirror
    `train.py:L1349` and `train.py:L1651`.)
  - **At L588 area** (the existing `if last_losses and ... log_every` block
    — the place where the `Loss/*` log dict is built): just **before**
    constructing the `log_dict`, insert the per-iteration episode log
    block. Compute it from `iteration_episodes` using `np.mean` /
    `np.min` / `np.max`:
    ```python
    if use_wandb and iteration_episodes:
        rewards = [ep['r'] for ep in iteration_episodes]
        lengths = [ep['l'] for ep in iteration_episodes]
        ep_log = {
            "Episode/Reward":     float(np.mean(rewards)),
            "Episode/Reward_Min": float(np.min(rewards)),
            "Episode/Reward_Max": float(np.max(rewards)),
            "Episode/Steps":      float(np.mean(lengths)),
            "Episode/Number":     total_episodes_completed,
        }
        wandb.log(ep_log, step=policy_step)
        iteration_episodes = []  # clear for next window
    ```
    (Mirrors `train.py:L1397–L1404` rPPO and `train.py:L1713–L1720`
    Dreamer.)
  - **At L308–L329** (post-`wandb.init`): add the `wandb.define_metric`
    block from `train.py:L625–L634` so `Episode/*` keys plot against
    `Episode/Number` not the auto-step. Verbatim:
    ```python
    wandb.define_metric("iteration")
    wandb.define_metric("timesteps")
    wandb.define_metric("Episode/Number")
    wandb.define_metric("*", step_metric="timesteps")
    wandb.define_metric("Episode/*", step_metric="Episode/Number")
    ```
    (The `loss/*` / `modulator/*` / `behavior/*` / `stage/*` define lines
    from `train.py:L630–L634` are NOT needed in dreamer-srl: the driver
    uses uppercase `Loss/*` / `Behavior/*` / `WorldModel/*` and has no
    continual-learning stages.)

**Verification.**

- `pytest tests/algorithms/dreamer_srl/ -q` → **49/49 PASS** (no new
  tests, no regression).
- Offline check → **17/17 PASS**.
- 2000-iter smoke run on `01_food_only.yaml` with `--num-envs 4` (so
  several episodes finish per log window). On WandB the **`Episode/*`
  panel must show one row every `log_every` iterations** (not one per
  done env). Confirm via `wandb-summary.json` step count under
  `Episode/Number`.

### Commit 3 — Episode/* extras: behavior-event + distance + termination keys (audit §3a fixed family)

**Goal.** Land the 22 additional fixed `Episode/*` keys that the
behavior-event accumulation populates in the master driver — every key
in the audit's §3a table except the 5 already covered in Commit 2
(`Episode/Reward`, `Episode/Reward_Min`, `Episode/Reward_Max`,
`Episode/Steps`, `Episode/Number`).

The 19 fixed keys this commit adds: `Episode/FoodEaten`,
`Episode/PredatorHits`, `Episode/DangerHits`, `Episode/RestCount`,
`Episode/Collisions`, `Episode/TotalDamage`, `Episode/DamagePredator`,
`Episode/DamageDanger`, `Episode/DamageObstacle`, `Episode/MeanDistFood`,
`Episode/MeanDistPredator`, `Episode/MeanDistRabbit`,
`Episode/MeanDistHidingPredator`, `Episode/RabbitHits`,
`Episode/HidingPredatorHits`, `Episode/Term_MaxSteps`,
`Episode/Term_Starvation`, `Episode/Term_Overeating`, `Episode/Term_Injury`.

**Mechanism.** Each per-env step in dreamer-srl returns an `infos` dict
from `env.step(...)` at `dreamer_srl_main.py:L420`. The master driver
extracts `BEHAVIOR_KEYS` (10 keys, `train.py:L943–L945`),
`BEHAVIOR_DIST_KEYS` (4 keys, `train.py:L946–L947`), and
`termination_reason` from this dict every step. dreamer-srl currently
extracts only `termination_reason` (`dreamer_srl_main.py:L433`); this
commit grafts in the rest.

**Files touched.**

- `src/algorithms/dreamer_srl/dreamer_srl_main.py`:
  - **Near L356** (after `episode_lengths` / `episode_rewards` are
    initialized): add per-env behavior-event accumulators identical to
    `train.py:L943–L950`:
    ```python
    BEHAVIOR_KEYS = ['ate_food', 'hit_predator', 'hit_hiding_predator',
                     'hit_neutral', 'event_collided', 'rested',
                     'damage', 'damage_predator', 'damage_hiding_predator',
                     'damage_obstacle']
    BEHAVIOR_DIST_KEYS = ['dist_to_food', 'dist_to_pred',
                          'dist_to_neutral', 'dist_to_hiding_predator']
    episode_behavior  = {k: np.zeros(num_envs, dtype=np.float32) for k in BEHAVIOR_KEYS}
    episode_dist_sums = {k: np.zeros(num_envs, dtype=np.float32) for k in BEHAVIOR_DIST_KEYS}
    ```
  - **Inside the env-step block after L420** (right after `infos` is
    received): add per-env per-step accumulation, mirroring
    `train.py:L1327–L1332`:
    ```python
    info_np = {k: np.asarray(infos[k]) for k in (BEHAVIOR_KEYS + BEHAVIOR_DIST_KEYS + ['termination_reason'])}
    for k in BEHAVIOR_KEYS:
        episode_behavior[k] += info_np[k]
    for k in BEHAVIOR_DIST_KEYS:
        episode_dist_sums[k] += info_np[k]
    ```
  - **Inside the done-env loop at L451** (the same loop already
    rewritten in Commit 2): populate `ep_data` with all behavior + dist
    + termination_reason fields before appending to
    `iteration_episodes`. Mirror `train.py:L1354–L1359`:
    ```python
    ep_data = {'r': ep_rew, 'l': ep_len}
    for k in BEHAVIOR_KEYS:
        ep_data[k] = float(episode_behavior[k][i])
    for k in BEHAVIOR_DIST_KEYS:
        ep_data[k] = float(episode_dist_sums[k][i] / max(ep_len, 1))
    ep_data['termination_reason'] = int(info_np['termination_reason'][i])
    iteration_episodes.append(ep_data)
    # zero accumulators for this env
    for k in BEHAVIOR_KEYS:
        episode_behavior[k][i] = 0.0
    for k in BEHAVIOR_DIST_KEYS:
        episode_dist_sums[k][i] = 0.0
    ```
  - **Inside the per-iteration episode-log block** (the one introduced
    in Commit 2 at L588 area): expand the `ep_log` dict with the 15
    behavior + distance keys + the 4 termination-reason fraction keys.
    Mirror `train.py:L1406–L1428` verbatim (the `np.mean([...])`
    formulae are identical for rPPO and Dreamer; dreamer-srl uses the
    same dict-construction block):
    ```python
    if 'ate_food' in iteration_episodes[0]:
        ep_log.update({
            "Episode/FoodEaten":     float(np.mean([ep['ate_food']             for ep in iteration_episodes])),
            "Episode/PredatorHits":  float(np.mean([ep['hit_predator']         for ep in iteration_episodes])),
            "Episode/DangerHits":    float(np.mean([ep['hit_hiding_predator']  for ep in iteration_episodes])),
            "Episode/RestCount":     float(np.mean([ep['rested']               for ep in iteration_episodes])),
            "Episode/Collisions":    float(np.mean([ep['event_collided']       for ep in iteration_episodes])),
            "Episode/TotalDamage":   float(np.mean([ep['damage']               for ep in iteration_episodes])),
            "Episode/DamagePredator":float(np.mean([ep['damage_predator']      for ep in iteration_episodes])),
            "Episode/DamageDanger":  float(np.mean([ep['damage_hiding_predator'] for ep in iteration_episodes])),
            "Episode/DamageObstacle":float(np.mean([ep['damage_obstacle']      for ep in iteration_episodes])),
            "Episode/MeanDistFood":  float(np.mean([ep['dist_to_food']         for ep in iteration_episodes])),
            "Episode/MeanDistPredator":       float(np.mean([ep['dist_to_pred']                for ep in iteration_episodes])),
            "Episode/MeanDistRabbit":         float(np.mean([ep['dist_to_neutral']             for ep in iteration_episodes])),
            "Episode/MeanDistHidingPredator": float(np.mean([ep['dist_to_hiding_predator']     for ep in iteration_episodes])),
            "Episode/RabbitHits":             float(np.mean([ep['hit_neutral']                 for ep in iteration_episodes])),
            "Episode/HidingPredatorHits":     float(np.mean([ep['hit_hiding_predator']         for ep in iteration_episodes])),
        })
        term_reasons = [ep['termination_reason'] for ep in iteration_episodes]
        for code, name in [(1, 'MaxSteps'), (2, 'Starvation'), (3, 'Overeating'), (4, 'Injury')]:
            ep_log[f"Episode/Term_{name}"] = float(np.mean([1.0 if r == code else 0.0 for r in term_reasons]))
    ```

**Caveats / gotchas.**

- The dreamer-srl driver currently does not extract `agent_in_bush` from
  `infos`. That key is needed only by the BM toolkit (Commit 5) — leave it
  alone in this commit.
- `info_np['termination_reason']` is already read at `dreamer_srl_main.py:L433`
  for the terminated-vs-truncated split. Keep that existing read for
  CP7-P2 correctness; the new read inside the BEHAVIOR_KEYS bundle is
  redundant-but-cheap (same array). The implementing agent can DRY this
  with a single `info_np` dict scoped above L433 if they want — but the
  redundant read is acceptable since the cost is one numpy view.

**Verification.**

- pytest 49/49 + offline_check 17/17.
- Smoke run on `01_food_only.yaml` for ≥2000 iters. Inspect the WandB
  run's `output.log` and `wandb-summary.json`; confirm **all 24 fixed
  `Episode/*` keys** from §3a (the 5 from Commit 2 + the 19 from this
  commit) are present.
- Sanity check: `Episode/Term_MaxSteps` + `Episode/Term_Starvation` +
  `Episode/Term_Overeating` + `Episode/Term_Injury` should sum to ~1.0
  per row (every episode ended in exactly one of the four ways).

### Commit 4 — Per-tag fan-out for distance keys (audit Q2)

**Goal.** Add the `Episode/MeanDistRabbit_<tag>` /
`Episode/MeanDistPredator_<tag>` per-tag fan-out for dashboards that
slice distance by predator or rabbit tag (used heavily in the
10×10 hypervigilance + sameprop study families).

**Mechanism.** The env's `infos` dict carries two per-tag arrays:
`dist_per_predator` of shape `[B, num_predator_tags]` and
`dist_per_neutral` of shape `[B, num_neutral_tags]`. Both are populated
only when the env config declares `params.predator_tags` /
`params.neutral_tags` (a possibly-empty tuple). The master driver
extracts them at `train.py:L1300–L1301` (rPPO) and `train.py:L1598–L1599`
(Dreamer batched), accumulates per-env per-tag sums (`train.py:L957–L958`),
and at episode-end records `mean_dist_rabbit_<tag>_raw` /
`mean_dist_predator_<tag>_raw` per-tag (`train.py:L1362–L1369`,
matching `train.py:L1637–L1644` for Dreamer). The iteration-mean fan-out
helper `_append_per_tag_means` (`train.py:L1046–L1059`) then groups
per-tag scalars by tag, computes per-episode mean across matching
instances, then iteration-mean across episodes, and writes
`Episode/MeanDistRabbit_<tag>` / `Episode/MeanDistPredator_<tag>` keys.

**Files touched.**

- `src/algorithms/dreamer_srl/dreamer_srl_main.py`:
  - **Near the new behavior-event init block from Commit 3**: also pull
    in tags + allocate per-tag accumulators. Mirror
    `train.py:L953–L958`:
    ```python
    neutral_tags  = tuple(env_params.neutral_tags)
    predator_tags = tuple(env_params.predator_tags)
    num_neutral_for_log  = len(neutral_tags)
    num_predator_for_log = len(predator_tags)
    episode_dist_per_neutral_sums  = np.zeros((num_envs, num_neutral_for_log),  dtype=np.float32)
    episode_dist_per_predator_sums = np.zeros((num_envs, num_predator_for_log), dtype=np.float32)
    ```
    (`env_params` is the variable name used at `dreamer_srl_main.py:L240`;
    use it consistently.)
  - **Inside the env-step block** (next to the new behavior-event sums
    added in Commit 3): also accumulate per-tag distance sums when the
    tag families are non-empty. Mirror `train.py:L1340–L1341`:
    ```python
    if num_neutral_for_log > 0 and 'dist_per_neutral' in infos:
        episode_dist_per_neutral_sums  += np.asarray(infos['dist_per_neutral'])
    if num_predator_for_log > 0 and 'dist_per_predator' in infos:
        episode_dist_per_predator_sums += np.asarray(infos['dist_per_predator'])
    ```
  - **Inside the done-env loop** (after the existing `ep_data` fill from
    Commit 3): write per-tag means into `ep_data`. Mirror
    `train.py:L1361–L1369`:
    ```python
    ep_l_safe = max(ep_len, 1)
    if num_neutral_for_log > 0:
        means = episode_dist_per_neutral_sums[i] / ep_l_safe
        for j, tag in enumerate(neutral_tags):
            ep_data[f'mean_dist_rabbit_{tag}_raw'] = float(means[j])
    if num_predator_for_log > 0:
        means = episode_dist_per_predator_sums[i] / ep_l_safe
        for j, tag in enumerate(predator_tags):
            ep_data[f'mean_dist_predator_{tag}_raw'] = float(means[j])
    # zero per-tag accumulators for this env
    if num_neutral_for_log  > 0: episode_dist_per_neutral_sums[i, :]  = 0.0
    if num_predator_for_log > 0: episode_dist_per_predator_sums[i, :] = 0.0
    ```
  - **Inside the per-iteration episode-log block** (the one expanded in
    Commit 3): import + call the master driver's `_append_per_tag_means`
    helper. The cleanest path is to **lift the helper out of
    `train.py:L1046–L1059` into a shared module** so dreamer-srl can
    reuse it without copy/paste. Add a new module
    `src/utils/episode_logging.py` containing `_append_per_tag_means`,
    update `train.py` to import from there (the function body remains
    identical — this is a refactor, not a logic change), and import +
    call from `dreamer_srl_main.py`:
    ```python
    from src.utils.episode_logging import append_per_tag_means
    ...
    append_per_tag_means(ep_log, iteration_episodes, neutral_tags,
                          'mean_dist_rabbit',   'Episode/MeanDistRabbit')
    append_per_tag_means(ep_log, iteration_episodes, predator_tags,
                          'mean_dist_predator', 'Episode/MeanDistPredator')
    ```
    Alternative if the cross-file refactor feels risky in one commit:
    copy the function body verbatim into `dreamer_srl_main.py` as a
    private helper `_append_per_tag_means_local` and leave the master
    driver's copy untouched — this avoids the train.py edit at the
    cost of a TODO. **Recommended path: shared module** (cleaner, one
    function, two callers).

**Caveats / gotchas.**

- The env-step call at `dreamer_srl_main.py:L420` returns `infos` from
  `env.step(...)`. Confirm via a one-line print at the top of the loop
  that `dist_per_predator` and `dist_per_neutral` are present in
  `infos` keys when the env config declares non-empty tags
  (e.g., `04_pred_neut_only.yaml`). If not, the env wrapper at
  `src/environment/wrapper.py` may need to expose them — surface this
  as a separate finding back to senior-developer before continuing.
- For a smoke target that has tags, use `01_food_only.yaml` (no tags
  → no per-tag keys → confirms no-tag path works) and then a
  multi-tag env (recommend
  `configs/experiment/dreamer_curriculum/04_pred_neut_only.yaml` if it
  exists; otherwise pick any env with non-empty predator_tags/neutral_tags
  in `env_params`).

**Verification.**

- pytest 49/49 + offline_check 17/17.
- **Two** smoke runs:
  1. `01_food_only.yaml` (no tags): per-tag keys absent from
     `wandb-summary.json` — fan-out family is silently skipped. Fixed
     `Episode/*` keys still all present.
  2. A multi-tag env: `Episode/MeanDistRabbit_<tag>` keys present for
     each declared neutral tag; `Episode/MeanDistPredator_<tag>` keys
     present for each declared predator tag.

### Commit 5 — Behavior-Measure toolkit v1 full port (audit Q1)

**Goal.** Wire the full BM toolkit into dreamer-srl: all 16 fixed
per-class keys + per-tag fan-out (4 keys per predator tag, 4 per neutral
tag), totalling ~30 keys on a typical 2-predator-tag + 2-neutral-tag env.
The toolkit's logic already lives in the reusable
[`src/behavior/accumulators.py`](src/behavior/accumulators.py) module
(audit §5; the module exposes `make_bm_state`, `bm_step_update`,
`bm_reset_env`, `bm_finalise_episode`, `bm_wandb_keys`). This commit
only adds **four hook-site calls** in the driver.

**Files touched.**

- `src/algorithms/dreamer_srl/dreamer_srl_main.py`:
  - **Top of file imports**: add
    ```python
    from src.behavior.accumulators import (
        make_bm_state, bm_step_update, bm_reset_env,
        bm_finalise_episode as _bm_finalise_episode_shared,
    )
    from src.behavior.config import load_behavior_measure_cfg
    ```
    The exact import path for `load_behavior_measure_cfg` should match
    the master driver's import (`train.py` imports it from
    `src.behavior.config` or wherever it lives — the developer should
    `grep -n "load_behavior_measure_cfg" train.py` and copy the same
    path).
  - **Near the per-tag init block from Commit 4**: load BM config +
    construct `_bm_state`. Mirror `train.py:L962–L982`:
    ```python
    bm_cfg = load_behavior_measure_cfg(env_cfg)  # or wherever the merged config lives
    bm_enabled = bm_cfg is not None and bm_cfg.enabled
    if bm_enabled:
        bm_R = float(bm_cfg.cue_radius)
        bm_K = int(bm_cfg.obs_window)
        _bm_state = make_bm_state(
            num_envs=num_envs,
            num_predator_tags=num_predator_for_log,
            num_neutral_tags=num_neutral_for_log,
            bm_R=bm_R,
            bm_K=bm_K,
        )
    else:
        _bm_state = None
    ```
  - **Inside the env-step block, AFTER `infos` is read but BEFORE the
    done-env handling**: call `bm_step_update` with the per-step info
    bundle. Mirror `train.py:L1339–L1342`:
    ```python
    if bm_enabled:
        _bm_info_t = {
            'ate_food':       np.asarray(infos['ate_food']),
            'agent_in_bush':  np.asarray(infos['agent_in_bush']),
        }
        if 'dist_per_predator' in infos:
            _bm_info_t['dist_per_predator'] = np.asarray(infos['dist_per_predator'])
        if 'dist_per_neutral' in infos:
            _bm_info_t['dist_per_neutral']  = np.asarray(infos['dist_per_neutral'])
        bm_step_update(_bm_state, _bm_info_t, dones.astype(bool))
    ```
    Note: `agent_in_bush` is a NEW info-dict key the env exposes and
    dreamer-srl has not previously extracted. The master driver
    explicitly extracts it at `train.py:L1303`. If `agent_in_bush` is
    not present in `infos`, the env's `get_observation` or step
    function may need to expose it — surface this as a separate
    finding to senior-developer if so.
  - **Inside the done-env loop** (after `ep_data` is filled with the
    behavior-event + per-tag distance keys from Commits 3 & 4): call
    `_bm_finalise_episode_shared` to add per-class + per-tag raw BM
    scalars to `ep_data`. Mirror `train.py:L1371–L1372`:
    ```python
    if bm_enabled:
        ep_data.update(_bm_finalise_episode_shared(
            _bm_state, i, predator_tags, neutral_tags,
        ))
    ```
    And after the existing per-env reset block from Commit 4, add:
    ```python
    if bm_enabled:
        bm_reset_env(_bm_state, i)
    ```
  - **Inside the per-iteration episode-log block**: add a call to the
    `_bm_log_wandb` helper from `train.py:L1008–L1044`. Same
    refactor-vs-copy decision as Commit 4 — recommend lifting
    `_bm_log_wandb` + `_append_per_measure_mean` into
    `src/utils/episode_logging.py` and importing in both `train.py`
    and `dreamer_srl_main.py`:
    ```python
    from src.utils.episode_logging import bm_log_wandb, append_per_tag_means
    ...
    if bm_enabled:
        bm_log_wandb(ep_log, iteration_episodes, predator_tags, neutral_tags)
    ```

**Caveats / gotchas.**

- The BM toolkit's `bm_step_update` mutates the shared `_bm_state` in
  place via numpy. It is **NOT** thread-safe or vmap-safe; the
  dreamer-srl driver's env-step loop is sequential per-iteration (no
  vmap), so this is fine — but flag it in the implementation report.
- Behavior-measure cue radius and observation-window K are
  config-driven from `env_cfg['behavior_measures']`. The merged
  `env_cfg` (built at `dreamer_srl_main.py:L191–L193`) inherits the
  default config first then the experiment-specific override — confirm
  that `behavior_measures:` is propagated through the merge before
  Commit 5 lands. If the default config does not declare
  `behavior_measures:`, only experiments that explicitly opt in will
  get BM keys (intended behavior, matching train.py).
- Some downstream consumers expect the BM keys to be ABSENT when the
  config does not enable the toolkit (backwards-compat). The
  `if bm_enabled:` gate must wrap **every** BM call site (init,
  step-update, finalise, reset, log) — same pattern as train.py.

**Verification.**

- pytest 49/49 + offline_check 17/17.
- Smoke run on a BM-enabled env (one of the hypervigilance or sameprop
  configs that declares `behavior_measures:` — recommended:
  `configs/experiment/hypervigilance/<latest round>.yaml`, the
  developer should pick one with `behavior_measures.enabled: true`).
  Confirm in `wandb-summary.json`:
  - 8 fixed keys per class × 2 classes = 16 fixed keys
    (`Episode/InterruptedFeedingRate_predator`, `_rabbit`, etc.)
  - 4 per-tag keys per predator tag + 4 per neutral tag
    (e.g. `Episode/EatUnderThreatRatio_predator_<tag>`)
- Sanity check: `Episode/InterruptedFeedingDenominator_<class>` should
  be a non-negative integer (count of candidate feeding events); if
  the agent never feeds under threat, this is `0` and the rate is
  `nan` (filtered out by `_append_per_measure_mean`).

### Commit 6 — WorldModel/* + Behavior/* trainer extras (audit Q4)

**Goal.** Match the master Dreamer trainer's ~25-key metrics dict by
exposing the missing 16 keys from the dreamer-srl trainer's loss
computation. The list (audit §3b, "JAX-Dreamer ✅ but dreamer-srl ❌"
rows):

- `WorldModel/loss_dyn_kl`, `WorldModel/loss_rep_kl` — split of the
  current `WorldModel/loss_kl` (currently aliased to
  `Loss/state_loss`) into its two KL components.
- `WorldModel/model_reward_mae`, `WorldModel/model_reward_mae_pos`,
  `WorldModel/model_reward_mae_neg` — reward-head quality probes.
- `WorldModel/model_latent_entropy` — posterior latent entropy.
- `WorldModel/model_cont_acc` — continue-head classification accuracy.
- `Behavior/loss_actor_policy`, `Behavior/loss_actor_entropy` —
  actor-loss decomposition into policy + entropy terms.
- `Behavior/mean_return`, `Behavior/mean_norm_return`,
  `Behavior/mean_value`, `Behavior/mean_advantage`,
  `Behavior/mean_entropy`, `Behavior/value_mae` — imagined-rollout
  statistics.

**Files touched.**

- `src/algorithms/dreamer_srl/train.py`:
  - **L753–L760** (`wm_aux` dict inside `wm_loss_fn`): split the
    existing `kl_loss_mean` into its two components and add reward-MAE
    + latent-entropy + continue-accuracy. The signals are already
    computed in `wm_loss_fn` (see L743–L749: `kl_dyn`, `kl_repr`,
    `dyn_loss`, `repr_loss`); just expose the per-element means and
    the new model_* probes. Mirror
    `src/models/dreamer_v3_trainer.py:L277–L290`:
    ```python
    aux = {
        "wm_outputs":       wm_outputs,
        "kl_mean":          kl_raw.mean(),
        "kl_loss_mean":     kl_loss_2d.mean(),
        "dyn_kl_mean":      dyn_loss.mean(),   # NEW
        "rep_kl_mean":      repr_loss.mean(),  # NEW
        "reward_loss_mean": reward_loss_unreduced.mean(),
        "obs_loss_mean":    obs_loss_unreduced.mean(),
        "cont_loss_mean":   cont_loss_unreduced.mean(),
        # NEW probes — see src/models/dreamer_v3_trainer.py:L267-L275
        "reward_mae":       <compute from wm_outputs and rewards target>,
        "reward_mae_pos":   <same, conditioned on rew > 0>,
        "reward_mae_neg":   <same, conditioned on rew < 0>,
        "latent_entropy":   <-jnp.sum(q_dist * jax.nn.log_softmax(q_logits), axis=-1).mean() where q_logits = post_logits>,
        "cont_acc":         <jnp.mean((nnx.sigmoid(cont_pred) > 0.5) == cont_target.astype(bool))>,
    }
    ```
    The exact tensor names (`q_logits`, `cont_pred`) must match those
    inside `wm_loss_fn` — the developer reads
    `dreamer_v3_trainer.py:L260–L275` and adapts the same expressions
    to the dreamer-srl trainer's local variable names. **No logic
    change** — only readouts of tensors already in scope.
  - **Inside the actor-loss block at L897–L902**: expose policy +
    entropy decomposition of the actor objective and the
    imagined-rollout statistics. Mirror
    `src/models/dreamer_v3_trainer.py:L468–L478`. The current
    `actor_policy_loss` returned from `nnx.value_and_grad(actor_loss_fn)`
    is the **total** actor loss — split out the policy term and the
    entropy term as separate scalars by replacing the loss-fn's return
    with a tuple `(loss, aux_dict)` and switching to `has_aux=True`.
    The aux dict carries:
    ```python
    {
      "loss_actor_policy":  jnp.mean(-log_probs * advantage * discount_weights),
      "loss_actor_entropy": jnp.mean(-ENTROPY_SCALE * entropy * discount_weights),
      "mean_return":        jnp.mean(lambda_returns),
      "mean_norm_return":   jnp.mean(norm_returns),
      "mean_value":         jnp.mean(baseline),
      "mean_advantage":     jnp.mean(advantage),
      "mean_entropy":       jnp.mean(entropy),
      "value_mae":          jnp.mean(jnp.abs(baseline - jax.lax.stop_gradient(lambda_returns))),
    }
    ```
    Source-cite these inside the dreamer-srl trainer as
    `# Mirrors src/models/dreamer_v3_trainer.py:L468-L478`.
  - **L935–L944** (the `losses` dict returned by `make_train_step`):
    extend with the new keys, renamed to match the master trainer's
    namespace expectations. The driver-side prefix sort in Commit 6
    (next bullet) routes them to `WorldModel/*` vs `Behavior/*` by
    name prefix, so the keys must START with `loss_dyn`, `loss_rep`,
    `model_`, `loss_actor`, `mean_`, or `entropy`:
    ```python
    losses = {
        # World-model
        "world_model_loss":   wm_total_loss,
        "observation_loss":   wm_aux["obs_loss_mean"],
        "reward_loss":        wm_aux["reward_loss_mean"],
        "state_loss":         wm_aux["kl_loss_mean"],
        "continue_loss":      wm_aux["cont_loss_mean"],
        "loss_dyn_kl":        wm_aux["dyn_kl_mean"],
        "loss_rep_kl":        wm_aux["rep_kl_mean"],
        "model_reward_mae":   wm_aux["reward_mae"],
        "model_reward_mae_pos": wm_aux["reward_mae_pos"],
        "model_reward_mae_neg": wm_aux["reward_mae_neg"],
        "model_latent_entropy": wm_aux["latent_entropy"],
        "model_cont_acc":     wm_aux["cont_acc"],
        # Behavior
        "value_loss":         critic_value_loss,
        "policy_loss":        actor_policy_loss,        # total actor loss (kept for sheeprl-name compat)
        "loss_actor_policy":  actor_aux["loss_actor_policy"],
        "loss_actor_entropy": actor_aux["loss_actor_entropy"],
        "mean_return":        actor_aux["mean_return"],
        "mean_norm_return":   actor_aux["mean_norm_return"],
        "mean_value":         actor_aux["mean_value"],
        "mean_advantage":     actor_aux["mean_advantage"],
        "mean_entropy":       actor_aux["mean_entropy"],
        "value_mae":          actor_aux["value_mae"],
        # Bookkeeping
        "moments_invscale":   moments_invscale,
    }
    ```

- `src/algorithms/dreamer_srl/dreamer_srl_main.py`:
  - **L591–L602** (the existing `log_dict` block): replace the flat
    `Loss/*` mapping with a **prefix-sort** routing that mirrors
    `train.py:L1797–L1810`. The structure:
    ```python
    log_dict = {
        "Params/effective_replay_ratio": cumulative_grad_steps / max(policy_step, 1),
        "Time/sps_env":                  sps_env,
        "Diagnostic/moments_invscale":   float(last_losses.get("moments_invscale", float("nan"))),
    }
    for mk, mv in last_losses.items():
        v = float(mv)
        if mk.startswith('loss_actor') or mk.startswith('mean_') or mk == 'entropy' \
                or mk in ('value_mae',):
            log_dict[f"Behavior/{mk}"] = v
        elif mk.startswith('loss_model') or mk.startswith('loss_recon') or \
             mk.startswith('loss_kl') or mk.startswith('loss_rew') or \
             mk.startswith('loss_cont') or mk.startswith('loss_dyn') or \
             mk.startswith('loss_rep')  or mk.startswith('model_'):
            log_dict[f"WorldModel/{mk}"] = v
        elif mk in ('world_model_loss', 'observation_loss', 'reward_loss',
                    'state_loss', 'continue_loss'):
            # legacy sheeprl-style alias — re-namespace under WorldModel
            # to match the master schema
            alias = {
                'world_model_loss': 'loss_model',
                'observation_loss': 'loss_recon',
                'reward_loss':      'loss_rew',
                'state_loss':       'loss_kl',
                'continue_loss':    'loss_cont',
            }[mk]
            log_dict[f"WorldModel/{alias}"] = v
        elif mk in ('value_loss', 'policy_loss'):
            alias = {'value_loss': 'loss_critic', 'policy_loss': 'loss_actor'}[mk]
            log_dict[f"Behavior/{alias}"] = v
        elif mk == 'moments_invscale':
            pass  # already added at top of dict
        else:
            log_dict[mk] = v
    ```
    This dual-mode mapping (prefix-sort + explicit-alias) buys backwards
    compat against the sheeprl key names while enabling forward parity
    with the master Dreamer schema. Source-cite each branch with the
    corresponding `train.py` line range.

**Caveats / gotchas.**

- The actor-loss refactor (returning aux dict) changes the
  `actor_loss_fn` signature. The grad call site at
  `dreamer_srl/train.py:L901–L902` must update from
  `(actor_policy_loss, actor_grads) = nnx.value_and_grad(...)` to
  `(actor_policy_loss, actor_aux), actor_grads = nnx.value_and_grad(actor_loss_fn, has_aux=True)(actor)`.
  Confirm that `nnx.value_and_grad` returns the same outer-tuple shape
  as `jax.value_and_grad` with `has_aux=True` — if not, adjust.
- The model_reward_mae signals are computed exactly as in
  `dreamer_v3_trainer.py:L260–L268`; one straight copy of those four
  expression lines into `wm_loss_fn` is sufficient. No new tensors
  enter scope.
- This commit changes the loss-function return shapes but **not** the
  numerical loss value. All gradient computations are bit-identical to
  the pre-Commit-6 path. Confirm by running pytest after the change —
  if any of the dreamer-srl parity tests fail, the refactor introduced
  a bug.

**Verification.**

- pytest 49/49 + offline_check 17/17 (parity tests must pass — if any
  fail, the new readouts have leaked into the loss value, which is a
  bug not a feature).
- Smoke run; confirm in `wandb-summary.json` **all 16 new keys**:
  `WorldModel/loss_dyn_kl`, `WorldModel/loss_rep_kl`,
  `WorldModel/model_reward_mae`,
  `WorldModel/model_reward_mae_pos`, `WorldModel/model_reward_mae_neg`,
  `WorldModel/model_latent_entropy`, `WorldModel/model_cont_acc`,
  `Behavior/loss_actor_policy`, `Behavior/loss_actor_entropy`,
  `Behavior/mean_return`, `Behavior/mean_norm_return`,
  `Behavior/mean_value`, `Behavior/mean_advantage`,
  `Behavior/mean_entropy`, `Behavior/value_mae`, and the
  alias-rewrites `WorldModel/loss_model` / `loss_recon` / `loss_rew` /
  `loss_kl` / `loss_cont` and `Behavior/loss_critic` / `loss_actor`.
- Sanity: `WorldModel/model_cont_acc` should be in [0, 1] (it's a
  binary-classification accuracy).

### Commit 7 — Time/sps_env in master Dreamer + rPPO branches (audit Q5)

**Goal.** Add the throughput metric `Time/sps_env` (env-steps-per-second)
to **both** the rPPO branch and the Dreamer branch of the master driver,
so all three drivers report wall-clock training speed in a single
common key for cross-run dashboards.

**Files touched.**

- `train.py`:
  - **At L1481 area** (the rPPO step-aggregate `wandb_logs.update(...)`
    block — the one already containing `timesteps`, `iteration`, and
    `_stage_tag()`): add
    ```python
    "Time/sps_env": global_step / max((datetime.now() - start_time).total_seconds(), 1e-9),
    ```
    inside the dict literal at L1481–L1485. `start_time` is initialized
    at `train.py:L1152`; `global_step` is updated at L1317–L1318.
  - **At L1786 area** (the Dreamer step-aggregate `wandb_logs` dict
    initial construction — the dict already contains `"timesteps"`,
    `"iteration"`, and `"Params/effective_replay_ratio"`): add the
    same one-line addition:
    ```python
    "Time/sps_env": global_step / max((datetime.now() - start_time).total_seconds(), 1e-9),
    ```
    Insert as a fourth key in the dict literal at L1785–L1788.

**Caveats / gotchas.**

- Both branches use `datetime.now()` for wall-clock; the existing
  `start_time = datetime.now()` at L1152 captures the training-loop
  start, NOT the program start. This is correct for throughput — the
  dreamer-srl driver's `t_start = time.time()` at L372 plays the same
  role. The numbers will be directly comparable.
- The `1e-9` floor avoids a divide-by-zero on iteration 1 (first
  log_interval row). Same pattern as dreamer-srl L590.

**Verification.**

- pytest 49/49 + offline_check 17/17 (these are dreamer-srl tests; no
  regression expected since this commit only touches train.py).
- A short smoke run via the original `train.py` driver (any rPPO or
  Dreamer config — `configs/experiment/easy_baseline.yaml` or similar):
  confirm `Time/sps_env` appears in the WandB run's `wandb-summary.json`
  for both algorithm branches.

---

## 3. File-by-file edit manifest

A consolidated index of every line range that will be touched, mapped to
the commit that touches it and the source citation it mirrors.

| File | Commit | Lines | Mirrors |
|------|-------:|-------|---------|
| `configs/logger/wandb.yaml` | 1 | (read-only confirm) | already correct (audit §1 L5) |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | 1 | L175–L177 (argparse default), L307–L329 (`wandb.init` entity), L455 (key rename) | sheeprl@33b6366 → grid_world_pain master |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | 2 | L308–L329 (add `define_metric`), ~L370 (init `iteration_episodes`), L451–L458 (rewrite done-env block), L588 area (add per-iter `Episode/*` log block) | `train.py:L625-L634` (define), `train.py:L1062` (init), `train.py:L1349-L1377` (done-env), `train.py:L1397-L1404` (per-iter log) |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | 3 | ~L356 (BEHAVIOR_KEYS init), L420 area (info_np extract + per-step accum), L451–L458 (expand ep_data), L588 area (expand `ep_log`) | `train.py:L943-L950, L1297-L1303, L1327-L1332, L1354-L1359, L1406-L1428` |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | 4 | ~L356 (per-tag init), L420 area (per-tag step-accum), L451 (per-tag ep_data fields + reset), L588 area (fan-out call) | `train.py:L953-L958, L1300-L1301, L1340-L1341, L1361-L1369, L1430-L1433` |
| `src/utils/episode_logging.py` (NEW) | 4 (+5) | new module, ~40 LOC | extracts `train.py:L1000-L1059` (`_append_per_measure_mean` + `_append_per_tag_means` + `_bm_log_wandb`) |
| `train.py` | 4 (+5) | replace 3 local helpers with `from src.utils.episode_logging import ...` | no logic change; lift-and-import refactor |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | 5 | top imports (BM toolkit), ~L356 (BM init), L420 area (`bm_step_update`), L451 (`bm_finalise_episode` + `bm_reset_env`), L588 area (`bm_log_wandb` call) | `train.py:L962-L998, L1339-L1342, L1371-L1372, L1390-L1391, L1434-L1436` (rPPO Site 1), `train.py:L1644-L1647, L1662-L1663, L1750-L1752` (Dreamer Site 2) |
| `src/algorithms/dreamer_srl/train.py` | 6 | L753–L760 (`wm_aux` extras), ~L260–L290 area (compute new model_* probes inside `wm_loss_fn`), L897–L902 (actor-loss has_aux refactor), L935–L944 (extend `losses` dict) | `src/models/dreamer_v3_trainer.py:L260-L290 + L468-L478` |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | 6 | L591–L602 (replace flat `Loss/*` map with prefix-sort routing) | `train.py:L1797-L1810` |
| `train.py` | 7 | L1481 area (rPPO add `Time/sps_env`), L1786 area (Dreamer add `Time/sps_env`) | mirrors `dreamer_srl_main.py:L590, L600` |

**Tests to add** (during Commits 3, 5, 6 — bundle into the same commit
as the feature):

- `tests/algorithms/dreamer_srl/test_metric_schema.py` (NEW, ~80 LOC) —
  a smoke test that launches dreamer-srl on `01_food_only.yaml` with
  `--total-steps 200 --num-envs 4 --no-wandb`, captures the log_dict
  that would have been sent to WandB (by mocking `wandb.log`), and
  asserts the key-set against an expected manifest. Run once per
  commit boundary as part of the smoke verification step.

---

## 4. Verification protocol per commit

The developer runs the same three-step check after each of the seven
commits. **No commit is signed off as "done" until all three pass.**

| Step | Command | Expected |
|------|---------|----------|
| Unit tests | `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest tests/algorithms/dreamer_srl/ -q` | 49/49 PASS (may grow by 1 if `test_metric_schema.py` is added in this commit) |
| Offline parity check | the dreamer-srl offline-check harness (the 17-test parity manifest from v2-CP10) | 17/17 PASS |
| WandB smoke run | run dreamer-srl on `01_food_only.yaml` for 200–2000 iters with `--num-envs 4 --wandb-name commit<N>_smoke` | run lands in `grid_world_pain` project; expected keys from this commit appear in `wandb-summary.json` |

The smoke run's expected-key manifest grows commit-by-commit:

- **After Commit 1:** `Episode/Steps`, `Episode/Reward` present (raw,
  per-done-env).
- **After Commit 2:** `Episode/Number`, `Episode/Reward_Min`,
  `Episode/Reward_Max` added; cadence flips to per-`log_every`.
- **After Commit 3:** +19 fixed behavior + termination keys.
- **After Commit 4:** +per-tag distance keys on a multi-tag env.
- **After Commit 5:** +16 fixed + per-tag BM toolkit keys on a
  BM-enabled env.
- **After Commit 6:** +16 trainer-side `WorldModel/*` + `Behavior/*`
  extras.
- **After Commit 7:** `Time/sps_env` lands on **master** Dreamer +
  rPPO smoke (separate driver — `train.py` runs).

If any of the three checks fails for a given commit, the commit does
NOT land; revert the chunk per §6.

---

## 5. Final smoke verification (after all 7 commits)

The single end-to-end smoke that demonstrates the full migration is
healthy.

**Command (from repo root, single line — break only for readability):**

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python \
    src/algorithms/dreamer_srl/dreamer_srl_main.py \
    --env-config   configs/experiment/dreamer_curriculum/01_food_only.yaml \
    --agent-config configs/dreamer_srl/01_food_only.yaml \
    --total-steps 2000 --num-envs 4 --seed 0 \
    --wandb-project grid_world_pain \
    --wandb-name dreamer_srl_v3_schema_migration_smoke_s0
```

For the BM-toolkit and per-tag fan-out smoke, repeat with a multi-tag,
BM-enabled env config (the implementing agent picks the canonical
hypervigilance config that has `behavior_measures.enabled: true` and
non-empty `predator_tags` / `neutral_tags`; recommended:
`configs/experiment/hypervigilance/<latest round>.yaml`).

**Expected-key manifest** (a Python dict the developer pastes into a
helper script + parses against the WandB run's `wandb-summary.json`):

```python
EXPECTED_KEYS = {
    # Commit 2 + 3
    "Episode/Reward", "Episode/Reward_Min", "Episode/Reward_Max",
    "Episode/Steps", "Episode/Number",
    "Episode/FoodEaten", "Episode/PredatorHits", "Episode/DangerHits",
    "Episode/RestCount", "Episode/Collisions",
    "Episode/TotalDamage", "Episode/DamagePredator",
    "Episode/DamageDanger", "Episode/DamageObstacle",
    "Episode/MeanDistFood", "Episode/MeanDistPredator",
    "Episode/MeanDistRabbit", "Episode/MeanDistHidingPredator",
    "Episode/RabbitHits", "Episode/HidingPredatorHits",
    "Episode/Term_MaxSteps", "Episode/Term_Starvation",
    "Episode/Term_Overeating", "Episode/Term_Injury",
    # Commit 6 (after WorldModel + Behavior fan-out)
    "WorldModel/loss_model", "WorldModel/loss_recon",
    "WorldModel/loss_rew", "WorldModel/loss_kl",
    "WorldModel/loss_cont", "WorldModel/loss_dyn_kl",
    "WorldModel/loss_rep_kl", "WorldModel/model_reward_mae",
    "WorldModel/model_reward_mae_pos", "WorldModel/model_reward_mae_neg",
    "WorldModel/model_latent_entropy", "WorldModel/model_cont_acc",
    "Behavior/loss_critic", "Behavior/loss_actor",
    "Behavior/loss_actor_policy", "Behavior/loss_actor_entropy",
    "Behavior/mean_return", "Behavior/mean_norm_return",
    "Behavior/mean_value", "Behavior/mean_advantage",
    "Behavior/mean_entropy", "Behavior/value_mae",
    # Params + Time + Diagnostic
    "Params/effective_replay_ratio", "Time/sps_env",
    "Diagnostic/moments_invscale",
}
# Plus per-tag fan-out on a multi-tag env:
#   Episode/MeanDistRabbit_<tag>     for each neutral_tag
#   Episode/MeanDistPredator_<tag>   for each predator_tag
# Plus BM toolkit on a BM-enabled env:
#   Episode/{Interrupted,Bush,Eat}*_{predator,rabbit}            (16 keys)
#   Episode/...rate_predator_<tag>                                (4/tag)
#   Episode/...rate_rabbit_<tag>                                  (4/tag)
```

Verification: load `wandb-summary.json`, compute
`set(EXPECTED_KEYS) - set(summary.keys())` — must be empty. Report any
missing key as a smoke FAIL. The dev posts the WandB run URL + the
diff (empty or otherwise) into the Implementation Report.

---

## 6. Rollback / failure handling

If any one of the seven commits regresses pytest, offline_check, or
the smoke-key manifest:

1. **Stop.** Do NOT proceed to the next commit.
2. **Revert that commit only** via `git revert <SHA>` (creates a NEW
   commit that undoes the broken one; safer than `git reset --hard`,
   which would also wipe any partial progress on later commits the
   developer might be staging in another worktree).
3. **Surface back to senior-developer** with: the failing test name,
   the failing-key list, the full WandB run URL (so the dashboard can
   be inspected), and the diff of the offending commit.
4. Senior-developer triages: is the failure a logic bug in the chunk,
   or a missing precondition (e.g., `agent_in_bush` not exposed by the
   env)? The fix lands as either (a) an addendum to this MIGRATION_PLAN
   doc + a follow-up commit, or (b) a separate plan if the root cause
   is upstream of dreamer-srl.

**Hard rule.** Never `git reset --hard` past the chunk boundary —
gitignored data (results/, claude_data/, wandb/) is preserved by
`reset --hard` but lost by any `git clean -x` follow-up, and the
project has had data loss from exactly that pattern before (see
CLAUDE.md "Git safety"). Use `git revert` for rollback.

---

## 7. Migration completion checklist

The developer marks each box as each chunk lands and passes its three
verifications:

- [ ] **Commit 1** — WandB project switch + minimal key renames. pytest
      49/49, offline_check 17/17, smoke run lands in `grid_world_pain`.
- [ ] **Commit 2** — Per-iteration episode aggregation. Smoke shows one
      `Episode/Number` row per `log_every` iters (not per done env).
- [ ] **Commit 3** — Episode/* behavior + termination extras (+19 keys).
      Smoke shows all 24 fixed Episode/* keys.
- [ ] **Commit 4** — Per-tag fan-out for distance keys.
      `Episode/MeanDistRabbit_<tag>` / `..Predator_<tag>` present on
      multi-tag smoke env.
- [ ] **Commit 5** — Full BM toolkit port. All 16 fixed per-class BM
      keys + per-tag fan-out present on BM-enabled smoke env.
- [ ] **Commit 6** — WorldModel/* + Behavior/* trainer extras (+16
      keys). All listed in §5 manifest are present.
- [ ] **Commit 7** — `Time/sps_env` in master Dreamer + rPPO branches.
      Visible on a smoke run of `train.py` (not dreamer-srl).
- [ ] **Final smoke** — full §5 manifest passes against
      `wandb-summary.json` on a fresh end-to-end run.

---

## 8. Hand-off

After all 7 commits land and the final smoke passes:

1. **`developer` reports back** with:
   - The 7 commit hashes (one per chunk).
   - The final smoke-run WandB URL.
   - The list of EXPECTED_KEYS that landed vs. missed (must be all
     landed, none missed, for sign-off).
   - Any deviations from this plan (e.g., the cross-file refactor in
     Commit 4 was harder than expected and split into two sub-commits)
     filled into the Implementation Report below.
2. **`senior-developer` verifies** the full migration (one round) by:
   - Reading the seven commit diffs end-to-end.
   - Spot-checking the smoke run on the WandB dashboard (do the
     `Episode/*` panels look right side-by-side with an old JAX-Dreamer
     run on the same project?).
   - Filling the Verification Report below.
3. **User authorizes the relaunches.** Once verified, the user can
   relaunch the 5×5+pred and 10×10 hypervigilance runs (tracked as
   tasks #28 + #29) with the new schema. The dashboards from these
   new runs will be directly comparable to the existing JAX-Dreamer
   and rPPO history.

---

## Implementation Report

> **Implemented by**: _(to be filled by `developer` agent)_
> **Date**: _(yyyy-mm-dd)_

_(One subsection per commit. Each subsection records: commit SHA, exact
file/lines touched, test outcomes, smoke run URL, any deviation from the
plan + why.)_

### Commit 1
- SHA:
- Files:
- pytest:
- offline_check:
- smoke URL:
- deviations:

### Commit 2
…

### Commit 3
…

### Commit 4
…

### Commit 5
…

### Commit 6
…

### Commit 7
…

### Final smoke
- URL:
- EXPECTED_KEYS diff against wandb-summary.json:

---

## Verification Report

> **Verified by**: `senior-developer`
> **Date**: _(yyyy-mm-dd)_

| Commit | File(s) touched | Status | Notes |
|--------|-----------------|:------:|-------|
| 1 | wandb.yaml, dreamer_srl_main.py | | |
| 2 | dreamer_srl_main.py | | |
| 3 | dreamer_srl_main.py | | |
| 4 | dreamer_srl_main.py, episode_logging.py, train.py | | |
| 5 | dreamer_srl_main.py | | |
| 6 | dreamer_srl/train.py, dreamer_srl_main.py | | |
| 7 | train.py | | |

**Conclusion**: _(one-line summary)_

---
