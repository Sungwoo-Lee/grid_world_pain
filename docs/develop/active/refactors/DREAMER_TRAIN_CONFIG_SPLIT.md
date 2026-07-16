---
title: "Give Dreamer its own train config (configs/train/dreamer_srl.yaml)"
topic: refactors
status: active
created: 2026-07-17
last_updated: 2026-07-17
---

# Give Dreamer its own train config (`configs/train/dreamer_srl.yaml`)

> **Status**: PLANNED
> **Opened**: 2026-07-17
> **Related**: [[TWO_LEVEL_LOGGING_REDESIGN]], [[CONFIG_LAYERING_AND_EXPERIMENT_REORG]], [docs/environment/CONFIG_GUIDE.md](../../../environment/CONFIG_GUIDE.md)

---

## Context

The project trains two different learning algorithms — **recurrent PPO** (the "rPPO" trainer) and **Dreamer** (a model-based agent). Both read the same shared file of training defaults, which holds things like how often to save a model checkpoint and how often to write a row to the training dashboard. rPPO also has its *own* extra file that layers algorithm-specific values on top of the shared one; Dreamer has no such file.

The result is that the shared file is quietly not shared at all: several of its values are **Dreamer's**, and the file says so out loud — one line is literally commented "Dreamer's value", and a whole comment block announces "These are the DREAMER values". Dreamer gets the right numbers today only because it happens to be the one trainer that reads the shared file without an override layer. That works right up until a third trainer arrives, or until someone edits the shared file believing it is neutral.

**This plan gives Dreamer the same structure rPPO already has**: a new file, `configs/train/dreamer_srl.yaml`, holding Dreamer's algorithm-specific values, merged on top of the shared defaults by the Dreamer trainer. The shared file then keeps only what genuinely belongs to *both*.

**Nothing about how any run behaves may change.** A parallel session is actively running rPPO experiments right now and may relaunch at any moment; rPPO's resolved settings must come out identical to the byte. Dreamer's must too. This is a pure structural move — same numbers, better-owned homes.

**The one rule this refactor must not break**: one setting, the number of episodes averaged into each dashboard point (`logging.episode.smoothing_episodes: 5000`), must be *identical* for both algorithms or their learning curves are not comparable — one would look smoother than the other for reasons having nothing to do with the agent. Today that is guaranteed **structurally**: the value lives only in the shared file, and rPPO's override file deliberately declines to redeclare it, so both algorithms are forced to inherit the same number. The new Dreamer file **must decline in exactly the same way**. If both algorithm files declared it, the guarantee would degrade from "impossible to break" to "a comment asks you not to break it" — and the two copies would eventually drift.

---

## Analysis

### A. Where the shared file is read

| Reader | Site | Merges rPPO override? | Reads any Dreamer-specific key? |
|---|---|---|---|
| rPPO trainer | `train.py:309-314` | Yes, `train.py:316-328`, gated on `agent.algorithm == "RecurrentPPO"` | n/a — it overrides them all |
| Dreamer trainer, **single-config** mode | `dreamer_srl_main.py:514-521` | No | **Yes** — `checkpoint_frequency`, `max_checkpoints_to_keep` are `get_mandatory` at `dreamer_srl_main.py:598-599` |
| Dreamer trainer, **curriculum** (`--configs-dir`) mode | `dreamer_srl_main.py:119-126` (`_load_stage_env_cfg`) | No | **Yes** — same `get_mandatory` at L598-599 runs on the stage-0 config |
| Random-agent sandbox | `main.py:45-48` | No | **No** (verified: no reference to any moved key) |
| Dreamer probe eval | `scripts/eval/dreamer_srl_probe_eval.py:145-149` | No | **No** (verified) |
| Five eval tests | `tests/algorithms/dreamer_srl/test_eval_{rollout,recording}.py`, `test_render_upload.py`, `tests/scripts/test_eval_stats_csv_columns.py`, `test_evaluation_model_rebuild.py` | No | **No** — all eval-path, none read the moved keys |

**Consequence**: only the two Dreamer trainer merge sites need the new file. `main.py`, `probe_eval`, and the tests are untouched by this refactor. **The curriculum site is a real hazard** (hazard #4 in the brief): `_load_stage_env_cfg` builds each stage's config through its *own* copy of the merge list, and the `get_mandatory('training.checkpoint_frequency')` at L598-599 runs on the stage-0 config in **both** modes. Miss that site and every curriculum Dreamer run dies at startup with a missing-key `ValueError`.

### B. Key-by-key ownership table

This is the core of the plan. "Moves" = leaves `configs/train/default.yaml`, lands in `configs/train/dreamer_srl.yaml`.

| Key | Current value in `default.yaml` | rPPO's resolved value | Dreamer's resolved value | Verdict | Justification |
|---|---|---|---|---|---|
| `logging.episode.smoothing_episodes` | 5000 | 5000 (inherited) | 5000 (inherited) | **SHARED — stays, and is declared in NEITHER algo file** | The universality rule. Shared inheritance *is* the enforcement mechanism. See Context. |
| `training.checkpoint_frequency` | 10000 *(labelled "Dreamer's value")* | 200000 (`recurrent_ppo.yaml:19`) | 10000 | **MOVES** | rPPO fully overrides it; the value in the shared file is Dreamer's and always was. |
| `training.max_checkpoints_to_keep` | 20 | `null` (`recurrent_ppo.yaml:20`) | 20 | **MOVES** | Same — rPPO fully overrides. |
| `logging.episode.interval_episodes` | 200 | 4000 (`recurrent_ppo.yaml:30`) | 200 | **MOVES** | Per-algo row volume; already commented "PER-ALGO". rPPO fully overrides. |
| `logging.step.smoothing_iters` | 200 | 100 (`recurrent_ppo.yaml:34`) | 200 | **MOVES** | Per-algo; losses are never cross-compared, so no universality rule. rPPO fully overrides. |
| `logging.step.interval_iters` | 100 | 50 (`recurrent_ppo.yaml:36`) | 100 | **MOVES** | Per-algo. rPPO fully overrides. |
| `training.log_interval` | 10 | 500 (`recurrent_ppo.yaml:18`) | inert | **STAYS** — see §C | Deprecated legacy fallback, inert whenever the `logging:` block is present (which it always is). Moving it buys nothing and touches a dead path. |
| `training.log_accumulate` | true | true | true | **SHARED — stays** | Deprecated companion of `log_interval`; same reasoning. |
| `training.num_envs` | 128 | 128 (or `--num-envs`) | **never read** — see §D | **STAYS** (comment-only clarification) | rPPO-only in practice; Dreamer takes it from the CLI. Moving it risks a live rPPO session. |
| `training.episodes` / top-level `episodes` | 100 | 100 (always CLI-overridden) | 100 (always CLI-overridden) | **SHARED — stays** | Smoke-test placeholder; identical semantics for both. |
| `training.video_during_training` | true | true | true | **SHARED — stays** | Same behaviour both algorithms. |
| `training.stats_during_training` | false | false | false | **SHARED — stays** | Same. |
| `training.auto_analysis` | false | false | false | **SHARED — stays** | Same. |
| `training.eval_video_episodes` | 3 | 3 | 3 | **SHARED — stays** | Same. |
| `training.eval_stats_episodes` | 100 | 100 | 100 | **SHARED — stays** | Same. |
| `training.eval_stats_num_envs` | 10 | 10 | 10 | **SHARED — stays** | Same (`dreamer_srl_main.py:597` reads it). |
| `training.seed` / top-level `seed` | 42 | 42 | 42 | **SHARED — stays** | Same. |
| `training.device` | "auto" | "auto" | "auto" | **SHARED — stays** | Same. |
| top-level `tag` | "default" | "default" | "default" | **SHARED — stays** | Same. |

**Backward-compat proof for rPPO (hazard #2)**: every one of the five moved keys is *unconditionally redeclared* in `configs/train/recurrent_ppo.yaml`, which merges **above** `default.yaml`. Removing them from the lower layer therefore cannot change rPPO's resolved value — the upper layer was already winning. Verified line by line against `recurrent_ppo.yaml:16-36`. This holds **provided the rPPO gate always fires** — verified below.

**rPPO gate is safe**: every YAML under `configs/models/recurrent_ppo/` declares an `algorithm:` key (checked exhaustively; zero misses). So `train.py:323`'s `agent.algorithm == "RecurrentPPO"` gate always fires for real rPPO runs, and the override layer is always present.

### C. `training.log_interval` — assessment

Dreamer resolves it at `dreamer_srl_main.py:1250-1253` with the precedence `--log-interval` CLI > `agent_cfg.training.log_interval` > `env_cfg.training.log_interval` > **hardcoded fallback 50**. So `default.yaml`'s `log_interval: 10` currently beats the code's own 50 for Dreamer — but it is **inert**, because the two-level `logging:` block is always present and takes precedence (`dreamer_srl_main.py:1222-1232`, `rolling_logging.py:53-56`), and a `[DEPRECATION]` warning fires if it is ever consulted.

**Recommendation: leave `log_interval: 10` and `log_accumulate: true` in `default.yaml`.** Rationale: (a) it is a deprecated back-compat fallback, not a live per-algo knob, so it does not belong in a per-algo file that documents live tuning decisions; (b) moving it would change *nothing* for either algorithm; (c) simplicity-first — this refactor should touch the smallest surface that removes the pollution. If someone later deletes the deprecated path entirely, both lines go at once, which is the right granularity.

### D. `training.num_envs: 128` — decision (hazard #1)

**Decision: leave it in `default.yaml`. Do NOT copy it into `dreamer_srl.yaml`. Add a clarifying comment only.**

- Dreamer **never reads `training.num_envs`**. Verified: `dreamer_srl_main.py:533` is `num_envs = args.num_envs` — a required CLI value, full stop. Only `training.eval_stats_num_envs` is read from config (L597).
- Therefore `128` is not a Dreamer landmine in the "wrong default gets used" sense — Dreamer physically cannot pick it up and OOM on it. It is a **documentation** landmine: a reader of the shared file reasonably assumes it applies to both.
- rPPO **does** read it (`train.py:496`, `config.get_mandatory('training.num_envs')` when `--num-envs` is absent). A parallel session is actively running rPPO (basic04 size sweep, nodes 107/108/110) and may relaunch at any time. Moving or changing this value is the single highest-risk edit available here.
- Copying `num_envs: 16` into `dreamer_srl.yaml` would be **actively harmful**: it would create a key that *looks* authoritative but that the Dreamer trainer ignores, so editing it would silently do nothing. That is a worse trap than the one being fixed.

The fix is a comment on the existing line marking it rPPO-only and noting Dreamer's CLI ownership. Value untouched, so rPPO is unaffected.

### E. Gating recommendation (merge conditionally or not?)

**Recommendation: merge `configs/train/dreamer_srl.yaml` UNCONDITIONALLY in `dreamer_srl_main.py`. Do not add an `agent.algorithm == "DreamerV3"` gate.**

Three reasons, the second decisive:

1. **The gate has no job to do.** `dreamer_srl_main.py` is the Dreamer entry point and only ever runs Dreamer. rPPO's gate exists because `train.py` is shared across algorithms — there, the gate is load-bearing. Here it would be ceremony that only *looks* symmetric.
2. **The gate would be actively unsafe.** Two Dreamer agent configs — `configs/models/dreamer_srl/agent_xs.yaml` and `configs/models/dreamer_srl/01_food_only_smoke.yaml` — do **not** declare `agent.algorithm`. Under a gate, those runs would skip the merge; and since `checkpoint_frequency` / `max_checkpoints_to_keep` have *moved out* of `default.yaml`, they would then hit `get_mandatory` at L598-599 and die with a missing-key `ValueError`. The gate converts a working smoke test into a crash.
3. **The gate adds ordering coupling.** It requires loading `agent_cfg` before the env-config merge; today `agent_cfg` loads *after* (`dreamer_srl_main.py:526`). The curriculum path would need the agent config threaded down into `_load_stage_env_cfg`, which currently takes only paths. Real complexity for zero benefit.

Symmetry of *structure* (each algorithm owns a train config) is the goal; symmetry of *mechanism* (a gate) is not, because the mechanisms answer different questions. This asymmetry is documented in the new file's header so the next reader does not "fix" it.

---

## Implementation Plan

### Design

1. Create `configs/train/dreamer_srl.yaml` with the five moved keys plus a header explaining why it exists, its precedence, the deliberate `smoothing_episodes` omission, and why there is no gate.
2. Delete those five keys from `configs/train/default.yaml` and de-Dreamer-ify its comments.
3. Merge the new file at **both** Dreamer merge sites, immediately after `configs/train/default.yaml` and before eval/viz — precedence mirroring rPPO's: above `train/default.yaml`, below the env config and agent config.
4. Prove resolved configs are unchanged for both algorithms, before and after.

**Precedence check.** In rPPO, `recurrent_ppo.yaml` merges above `train/default.yaml` and below `evaluation/default.yaml` (`train.py:316-336`). Placing `dreamer_srl.yaml` in the same slot is correct and safe: eval/viz defaults declare none of the five moved keys, so slot order relative to them cannot matter — but matching rPPO keeps the two trainers readable side by side.

### File Changes

#### `configs/train/dreamer_srl.yaml` (NEW FILE — exact contents)

```yaml
# Dreamer-specific training defaults.
#
# WHY THIS FILE EXISTS
# --------------------
# configs/train/default.yaml is shared by BOTH rPPO (train.py) and dreamer
# (src/algorithms/dreamer_srl/dreamer_srl_main.py). It used to carry Dreamer's
# per-algo values directly — which "worked" only because Dreamer was the sole
# consumer without an override layer, and made the shared file dishonest (it
# was labelled "Dreamer's value"). This file is Dreamer's override layer, exactly
# symmetric with configs/train/recurrent_ppo.yaml.
#
# PRECEDENCE
# ----------
# Merged ABOVE configs/train/default.yaml but BELOW the env --env-config and the
# --agent-config (and below CLI flags like --log-interval), so these remain
# overridable defaults. rPPO never loads this file.
#
# NO ALGORITHM GATE — DELIBERATE ASYMMETRY WITH rPPO
# --------------------------------------------------
# train.py gates recurrent_ppo.yaml on agent.algorithm == "RecurrentPPO" because
# train.py is shared across algorithms. dreamer_srl_main.py is the Dreamer entry
# point and only ever runs Dreamer, so it merges this file unconditionally.
# A gate would ALSO break agent_xs.yaml and 01_food_only_smoke.yaml, which do not
# declare agent.algorithm. Do not add a gate.
# See docs/develop/active/refactors/DREAMER_TRAIN_CONFIG_SPLIT.md
training:
  checkpoint_frequency: 10000   # checkpoint + eval video every 10,000 episodes
  max_checkpoints_to_keep: 20   # rotate; keep the 20 most recent

# Two-level logging: Dreamer's PER-ALGO overrides on top of configs/train/default.yaml.
# NOTE the deliberate omission: logging.episode.smoothing_episodes is NOT set here.
# It is inherited from default.yaml so that Dreamer and rPPO necessarily share it —
# that inheritance is the mechanism that enforces the universality rule (both curves
# must average the same number of episodes per point or they are not comparable).
# configs/train/recurrent_ppo.yaml omits it for the identical reason. Do not add it
# to either file: two copies would silently drift.
logging:
  episode:
    # PER-ALGO (row volume). Dreamer runs ~131k eps/24h -> ~655 rows/24h.
    # 200 < smoothing (5000) -> rolling windows overlap.
    interval_episodes: 200
  step:
    # PER-ALGO. Losses are never cross-compared between algorithms, so no
    # universality rule. 200 iters = 3200 env-steps of loss averaging at num_envs=16.
    smoothing_iters: 200
    # < smoothing -> overlapping rolling windows.
    interval_iters: 100
```

#### `configs/train/default.yaml` (lines 1-51 — remove Dreamer-only keys, de-Dreamer-ify comments)

```yaml
# BEFORE (lines 1-14):
# Default Training Configuration
#
# Config-owns-values convention: num_envs, seed, and checkpoint_frequency are
# CONFIG-OWNED — standard launches must NOT pass --num-envs / --seed /
# --checkpoint-frequency on the CLI; this file (and, for RecurrentPPO,
# configs/train/recurrent_ppo.yaml layered on top — see that file's header)
# is authoritative. A CLI flag for one of these is only ever an intentional,
# flagged-to-the-user deviation, never copied boilerplate. `episodes` is the
# opposite: it is passed explicitly per run (see note below) because the real
# budget genuinely varies by run (smoke=100 / standard=10M / extended=100M).
training:
  episodes: 100  # smoke-test safety placeholder only — real runs always pass --episodes explicitly
  num_envs: 128
  checkpoint_frequency: 10000  # Dreamer's value (RecurrentPPO overrides to 200000 via configs/train/recurrent_ppo.yaml)
  max_checkpoints_to_keep: 20

# AFTER (lines 1-14):
# Default Training Configuration — SHARED by rPPO and Dreamer.
#
# Only ALGORITHM-NEUTRAL values belong here. Per-algo values live in the two
# override layers, each merged ABOVE this file by its own trainer:
#   configs/train/recurrent_ppo.yaml  <- train.py (gated on agent.algorithm)
#   configs/train/dreamer_srl.yaml    <- src/algorithms/dreamer_srl/dreamer_srl_main.py
# See docs/develop/active/refactors/DREAMER_TRAIN_CONFIG_SPLIT.md
#
# Config-owns-values convention: num_envs, seed, and checkpoint_frequency are
# CONFIG-OWNED — standard launches must NOT pass --num-envs / --seed /
# --checkpoint-frequency on the CLI; this file and the per-algo layer above it
# are authoritative. A CLI flag for one of these is only ever an intentional,
# flagged-to-the-user deviation, never copied boilerplate. `episodes` is the
# opposite: it is passed explicitly per run (see note below) because the real
# budget genuinely varies by run (smoke=100 / standard=10M / extended=100M).
training:
  episodes: 100  # smoke-test safety placeholder only — real runs always pass --episodes explicitly
  # rPPO-ONLY IN PRACTICE: train.py:496 reads this; dreamer_srl_main.py:533 takes
  # num_envs from the REQUIRED --num-envs CLI flag and never reads this key.
  # Dreamer runs 16 and would OOM at 128 — but cannot pick this up. Left here (not
  # moved) because live rPPO runs depend on it. Do not copy it into dreamer_srl.yaml:
  # a key Dreamer ignores would look authoritative and silently do nothing.
  num_envs: 128
```

Then, still in `default.yaml`:

- **Delete** the `checkpoint_frequency: 10000` and `max_checkpoints_to_keep: 20` lines (old L14-15) — now in `dreamer_srl.yaml`.
- **Keep unchanged**: `video_during_training`, `stats_during_training`, `auto_analysis`, `eval_video_episodes`, `eval_stats_episodes`, `eval_stats_num_envs`, `seed`, `device`, `log_interval`, `log_accumulate`, top-level `episodes` / `seed` / `tag`.
- **Replace** the `logging:` block header comment (old L33-36) and delete the three per-algo keys:

```yaml
# BEFORE (lines 33-51):
# Two-level logging. These are the DREAMER values: dreamer_srl_main.py reads this file
# and does NOT load configs/train/recurrent_ppo.yaml, which overrides the per-algo knobs.
# Every knob names its own unit. smoothing = NOISE (how many samples averaged per point);
# interval = VOLUME (how often a row is written). They are independent on purpose.
logging:
  episode:
    # UNIVERSAL — must be IDENTICAL in every algorithm. ...
    smoothing_episodes: 5000
    # PER-ALGO (row volume). Dreamer runs ~131k eps/24h -> ~655 rows/24h. < smoothing -> overlap.
    interval_episodes: 200
  step:
    # PER-ALGO. Losses are never cross-compared between algorithms, so no universality rule.
    # 200 iters = 3200 env-steps of loss averaging at num_envs=16.
    smoothing_iters: 200
    # < smoothing -> overlapping rolling windows.
    interval_iters: 100

# AFTER (lines 33-44):
# Two-level logging — the UNIVERSAL knob only. Per-algo knobs (interval_episodes,
# smoothing_iters, interval_iters) live in configs/train/recurrent_ppo.yaml and
# configs/train/dreamer_srl.yaml, each merged above this file by its own trainer.
# Every knob names its own unit. smoothing = NOISE (how many samples averaged per point);
# interval = VOLUME (how often a row is written). They are independent on purpose.
logging:
  episode:
    # UNIVERSAL — must be IDENTICAL in every algorithm. This is the ONLY thing that makes
    # a Dreamer curve and an rPPO curve equally noisy and therefore comparable.
    # NEITHER per-algo file declares it — both inherit from here, and that inheritance IS
    # the enforcement mechanism. Do not add it to either. Do not add a CLI flag for it.
    # 5000 eps ~= 3-4% of a run; first point lands ~16 min in (measured, see the doc).
    smoothing_episodes: 5000
```

Note the `logging.step:` sub-block disappears entirely from `default.yaml` — both of its keys move. That is fine: `rolling_logging.py:66-68` triggers the two-level path if **any** `logging.*` key is present, and the per-algo layer always supplies the rest.

#### `src/algorithms/dreamer_srl/dreamer_srl_main.py` — merge site 1: single-config (lines 512-521)

```python
# BEFORE:
        env_cfg = get_default_config()  # seed base (archived/standalone configs rely on this)
        # Merge Training defaults (training.checkpoint_frequency etc.)
        # Ported from train.py:L297-L327
        for _cfg_rel in [
            'configs/train/default.yaml',
            'configs/evaluation/default.yaml',
            'configs/visualization/default.yaml',
        ]:

# AFTER:
        env_cfg = get_default_config()  # seed base (archived/standalone configs rely on this)
        # Merge Training defaults. Ported from train.py:L297-L336.
        # configs/train/dreamer_srl.yaml is Dreamer's per-algo override layer — it MUST
        # come immediately after train/default.yaml (above it in precedence), mirroring
        # rPPO's train.py:316-328. It is merged UNCONDITIONALLY (no agent.algorithm gate):
        # this entry point only ever runs Dreamer, and a gate would break the agent configs
        # that omit agent.algorithm (agent_xs.yaml, 01_food_only_smoke.yaml).
        # See docs/develop/active/refactors/DREAMER_TRAIN_CONFIG_SPLIT.md
        for _cfg_rel in [
            'configs/train/default.yaml',
            'configs/train/dreamer_srl.yaml',
            'configs/evaluation/default.yaml',
            'configs/visualization/default.yaml',
        ]:
```

#### `src/algorithms/dreamer_srl/dreamer_srl_main.py` — merge site 2: curriculum (lines 118-126, `_load_stage_env_cfg`)

**Do not miss this one.** `--configs-dir` runs build their stage configs here and never touch site 1; `get_mandatory('training.checkpoint_frequency')` at L598-599 runs on the stage-0 config in both modes, so omitting this makes every curriculum Dreamer run fail at startup.

```python
# BEFORE:
    cfg = get_default_config()  # seed base (archived/standalone stage configs rely on this)
    for rel in [
        'configs/train/default.yaml',
        'configs/evaluation/default.yaml',
        'configs/visualization/default.yaml',
    ]:

# AFTER:
    cfg = get_default_config()  # seed base (archived/standalone stage configs rely on this)
    # Keep this list in lockstep with the single-config merge below (L514-521):
    # dreamer_srl.yaml is Dreamer's per-algo override layer and must sit directly
    # above train/default.yaml in both modes.
    for rel in [
        'configs/train/default.yaml',
        'configs/train/dreamer_srl.yaml',
        'configs/evaluation/default.yaml',
        'configs/visualization/default.yaml',
    ]:
```

#### `docs/environment/CONFIG_GUIDE.md` (Maintenance Contract)

This refactor adds a file to the config system and changes the layering story, so the guide must be updated in the **same change** per its own Maintenance Contract. Required edits:

- Wherever the layering order is documented, record the new layer: `env defaults → train/default.yaml → train/<algo>.yaml → evaluation/default.yaml → visualization/default.yaml → env config → agent config → CLI`.
- Document the general rule now visible in two instances: **`configs/train/default.yaml` holds algorithm-neutral values only; each trainer layers `configs/train/<algo>.yaml` on top.**
- Document the **`smoothing_episodes` inheritance invariant**: the universal logging knob is declared *only* in `default.yaml` and in *neither* per-algo file, and that omission is load-bearing.
- Note the gating asymmetry (rPPO gated because `train.py` is shared; Dreamer unconditional because its entry point is single-algorithm).
- Check whether `docs/environment/02_config_schema.md` enumerates `configs/train/` files; if so, add `dreamer_srl.yaml` there too.

**No `scripts/` file is added, moved, renamed, or deleted**, so `SCRIPTS_DEPENDENCY_MAP.md` needs no update.

#### Files explicitly NOT changed

`train.py`, `configs/train/recurrent_ppo.yaml`, `main.py`, `scripts/eval/dreamer_srl_probe_eval.py`, and all five eval tests. Any diff touching these is out of scope and will be flagged in verification.

### Backward-compatibility verification (mandatory, blocking)

The claim to prove: **rPPO's and Dreamer's resolved configs are byte-identical before and after.** Do this with a before/after dump of the *actual resolved config object*, not by re-reading YAML by hand.

**Step 1 — BEFORE any edit**, capture both baselines. Write a throwaway dumper to `tmp/` that reproduces each trainer's merge chain verbatim and prints every resolved key sorted, then run it on the **pre-change** tree:

- **rPPO chain** — replicate `train.py:306-350`: `get_default_config()` → `configs/train/default.yaml` → `configs/train/recurrent_ppo.yaml` (gate on `agent.algorithm == "RecurrentPPO"`) → `configs/evaluation/default.yaml` → `configs/logger/wandb.yaml` → `configs/visualization/default.yaml`. Use a real rPPO agent config from `configs/models/recurrent_ppo/` — pick one the live basic04 sweep is using.
- **Dreamer single-config chain** — replicate `dreamer_srl_main.py:511-524`, with `configs/models/dreamer_srl/01_food_only_buf256k.yaml`.
- **Dreamer smoke chain** — same, with `configs/models/dreamer_srl/01_food_only_smoke.yaml` (**the agent config with no `agent.algorithm` key** — this is the case a gate would have broken; it must resolve identically).
- **Dreamer curriculum chain** — replicate `_load_stage_env_cfg` (L118-129) on any existing `--configs-dir` stage directory.

Save each to `tmp/YYYYMMDD_HHMMSS_dreamer_train_cfg_split/<chain>_before.txt`.

**Step 2 — after the edits**, rerun the same dumper (updated to the new merge lists) and save `<chain>_after.txt`.

**Step 3 — diff.** `diff <chain>_before.txt <chain>_after.txt` must be **empty for all four chains**. Paste the four `diff` invocations and their (empty) output into the Implementation Report. A non-empty diff on *any* chain is a **blocker** — report it, do not "fix" it by adjusting the expected value.

**Step 4 — spot-check the four values that matter most**, and state them explicitly in the report:

| Chain | `training.checkpoint_frequency` | `training.max_checkpoints_to_keep` | `logging.episode.smoothing_episodes` | `logging.episode.interval_episodes` |
|---|---|---|---|---|
| rPPO | 200000 | `None` | **5000** | 4000 |
| Dreamer (any of the three) | 10000 | 20 | **5000** | 200 |

The `smoothing_episodes` column must read **5000 on every row**. That is the universality rule, measured rather than asserted.

**Step 5 — smoke run.** Launch a short Dreamer smoke run to confirm nothing crashes at startup (`get_mandatory` at L598-599 is the failure point if a merge site was missed). Per the known budget gotcha, single-config mode reads the budget from `env_cfg.training.*` (default 100 episodes → instant exit), so pass `--episodes` and `--log-interval` explicitly on the CLI.

**Step 6 — test suite.** Run the five eval tests that merge `train/default.yaml` (`tests/algorithms/dreamer_srl/test_eval_rollout.py`, `test_eval_recording.py`, `test_render_upload.py`, `tests/scripts/test_eval_stats_csv_columns.py`, `tests/scripts/test_evaluation_model_rebuild.py`). Analysis says none read a moved key, so all five must pass **untouched** — if any fails, the analysis was wrong; report it rather than editing the test.

**No speed measurement needed**: this is config-layering only, changing one extra YAML parse at startup. Say so explicitly in the report.

**Delete the `tmp/` dumper when done** — it is a verification scaffold, not a deliverable.

### Out of scope

- Changing `num_envs` anywhere (see §D).
- Removing the deprecated `log_interval` / `log_accumulate` path (see §C).
- Adding an `agent.algorithm` key to `agent_xs.yaml` / `01_food_only_smoke.yaml`. They work fine unconditionally; adding it would be an unrelated config edit while a parallel session is live. **Note it to the user, do not do it.**
- Any change to rPPO's config or trainer.

## Checkpoints

- [x] **CP1** — Before touching anything: all four "before" dumps captured to `tmp/` and non-empty. Done.
- [x] **CP2** — `configs/train/dreamer_srl.yaml` created; the string `smoothing_episodes` appears exactly once, inside the omission comment (`# NOTE the deliberate omission: logging.episode.smoothing_episodes is NOT set here.`); zero `key:` occurrences.
- [x] **CP3** — `configs/train/default.yaml` no longer declares `checkpoint_frequency`, `max_checkpoints_to_keep`, `interval_episodes`, `smoothing_iters`, or `interval_iters` as keys (only in prose comments), and still declares `smoothing_episodes: 5000`, `num_envs: 128`, `log_interval: 10`.
- [x] **CP4** — Both Dreamer merge sites updated: single-config (now L513-529) and curriculum `_load_stage_env_cfg` (now L119-130). See "CP4 grep discrepancy" note below.
- [x] **CP5** — All four `diff before/after` runs are empty. See evidence below.
- [x] **CP6** — Spot-check table filled with measured values; `smoothing_episodes` = 5000 on every row.
- [x] **CP7** — Dreamer smoke run ran to full completion (20/20 episodes) on node 111 GPU 0, no missing-key `ValueError`, and printed the expected `Two-level logging active` banner.
- [x] **CP8** — The five eval test files (20 test functions total) pass, unmodified.
- [x] **CP9** — `CONFIG_GUIDE.md` updated with a new §7 "Training-config layering". `02_config_schema.md` checked — it does not enumerate `configs/train/` files (it's entirely about `EnvParams`/`configs/environment/`), so no edit needed there per the plan's own conditional instruction.
- [x] **CP10** — `tmp/20260717_dreamer_train_cfg_split/` (dumper + smoke-run scratch) deleted after evidence was captured into this report. `git status` shows only the intended files (plus pre-existing unrelated diffs from before this session: `docs/diary/2026-07-15.md`, `train_command-agent.sh`, `docs/develop/INDEX.md` regenerated by the plan-authoring session, and untracked `docs/diary/2026-07-02.md` / `2026-07-09.md` / `ncdu_260713`).

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-07-17

### Summary of changes, file-by-file

1. **`configs/train/dreamer_srl.yaml`** (NEW) — created verbatim per the plan's File Changes section: `training.checkpoint_frequency: 10000`, `training.max_checkpoints_to_keep: 20`, `logging.episode.interval_episodes: 200`, `logging.step.smoothing_iters: 200`, `logging.step.interval_iters: 100`. Header documents why the file exists, precedence, and the deliberate no-gate / no-`smoothing_episodes` decisions. `smoothing_episodes` appears only inside the omission comment — never as a declared key.
2. **`configs/train/default.yaml`** — removed the five Dreamer-only keys (`checkpoint_frequency`, `max_checkpoints_to_keep`, `logging.episode.interval_episodes`, `logging.step.smoothing_iters`, `logging.step.interval_iters`); de-Dreamer-ified the file header and the `logging:` block header comment; added the rPPO-only clarifying comment on `training.num_envs: 128` (value untouched); the `logging.step:` sub-block now disappears entirely from this file (both its keys moved) — confirmed inert per the plan's note (`rolling_logging.py` triggers two-level logging on any `logging.*` key, and the per-algo layer always supplies the rest, proven by the live smoke-run banner below).
3. **`src/algorithms/dreamer_srl/dreamer_srl_main.py`** — two merge-site edits:
   - Single-config merge (was L512-521, now L513-529): added `'configs/train/dreamer_srl.yaml'` immediately after `'configs/train/default.yaml'` in the merge list, plus an explanatory comment on the no-gate decision.
   - Curriculum merge, `_load_stage_env_cfg` (was L118-126, now L119-130): identical addition, plus a "keep in lockstep" comment pointing at the single-config site.
   - **Incidental fix (not in the plan's File Changes, flagged here rather than silently applied elsewhere):** one pre-existing comment at (now) L1254 referenced `configs/train/default.yaml:15` (where `log_interval: 10` used to sit) — this line number shifted to 35 as a direct consequence of the key removals above. Corrected the line number in the same file already being edited; no other stale line-number references to the moved keys were found (`grep -rn "train/default.yaml:" src/algorithms/dreamer_srl/dreamer_srl_main.py train.py` returns only this one hit, now correct).
4. **`docs/environment/CONFIG_GUIDE.md`** — added new §7 "Training-config layering (`configs/train/`)" (renumbering old §7 Pointers → §8) documenting: the full merge order across both trainers, the algorithm-neutral-only rule for `default.yaml`, the rPPO-gated vs. Dreamer-unconditional merge asymmetry and why, and the `smoothing_episodes` inheritance invariant. `02_config_schema.md` checked (`grep -n "configs/train"` → 0 hits) — it does not enumerate `configs/train/` files, so per the plan's own conditional ("if it enumerates... add it there too") no edit was needed.

### Backward-compatibility verification (four-chain diff)

Built a throwaway dumper (`tmp/20260717_dreamer_train_cfg_split/dump_configs.py`, deleted after use) that replicates each trainer's merge chain verbatim (rPPO: `train.py:306-350`; Dreamer single-config: `dreamer_srl_main.py:511-524`; Dreamer curriculum: `_load_stage_env_cfg` L118-129) and dumps every resolved key, sorted, to a flat `key = value` text file.

Chains used:
- **rPPO single-config**: `configs/models/recurrent_ppo/recurrent_ppo_M.yaml` (declares `agent.algorithm: "RecurrentPPO"`, so the rPPO gate fires).
- **Dreamer single-config**: `configs/models/dreamer_srl/01_food_only_buf256k.yaml` + `configs/environment/experiment/basic/01-slow_predator_5x5.yaml`.
- **Dreamer smoke**: `configs/models/dreamer_srl/01_food_only_smoke.yaml` (no `agent.algorithm` key — the exact case a gate would have broken) + same env config.
- **Dreamer curriculum**: `_load_stage_env_cfg` on `configs/environment/experiment/archive/dreamer_srl_curriculum/01_5x5_food_hide_rock.yaml`.

Ran the dumper before any edit (`before`) and after all edits (`after`, with `configs/train/dreamer_srl.yaml` included in the merge list):

```
$ diff rppo_single_before.txt rppo_single_after.txt        ; exit=0  (empty)
$ diff dreamer_single_before.txt dreamer_single_after.txt  ; exit=0  (empty)
$ diff dreamer_smoke_before.txt dreamer_smoke_after.txt    ; exit=0  (empty)
$ diff dreamer_curriculum_before.txt dreamer_curriculum_after.txt ; exit=0  (empty)
```

**All four diffs are empty — rPPO's and Dreamer's resolved configs are byte-identical before and after, on every path (including the smoke config that has no `agent.algorithm`, confirming the no-gate decision does not break it).**

### Step-4 spot-check table (measured, after the change)

| Chain | `training.checkpoint_frequency` | `training.max_checkpoints_to_keep` | `logging.episode.smoothing_episodes` | `logging.episode.interval_episodes` |
|---|---|---|---|---|
| rPPO single-config | 200000 | `None` (`null`) | **5000** | 4000 |
| Dreamer single-config | 10000 | 20 | **5000** | 200 |
| Dreamer smoke (no `agent.algorithm`) | 10000 | 20 | **5000** | 200 |
| Dreamer curriculum | 10000 | 20 | **5000** | 200 |

`smoothing_episodes` = 5000 on every row, measured from the resolved `Config` object, not asserted. Values are identical to the `before` dump on all four chains (see diffs above).

### Dreamer smoke run

Launched on node 111 GPU 0 (confirmed idle via `gpu_status.py` first): `01_food_only_smoke.yaml` (the no-`agent.algorithm` agent config) against `01-slow_predator_5x5.yaml`, `--episodes 20 --num-envs 4 --log-interval 5 --no-wandb --quiet --debug`. Startup log:

```
[dreamer-srl] eval config: video_during_training=True, eval_video_episodes=3, stats_during_training=False, checkpoint_frequency=10000, viz_fps=5, auto_render=True, max_checkpoints_keep=20
...
[WARN] --log-interval is IGNORED: this config uses the two-level `logging:` block.
[dreamer-srl] Two-level logging active: episode(smoothing=5000, interval=200) step(smoothing=200, interval=100)
```

`checkpoint_frequency=10000` and `max_checkpoints_keep=20` resolved correctly from the new `dreamer_srl.yaml` merge — no `ValueError` at the `get_mandatory` calls (L598-599, the failure point a missed merge site would have hit). The banner's four numbers (`smoothing=5000, interval=200, smoothing=200, interval=100`) match the spot-check table exactly. The run continued to full completion: `[dreamer-srl] Done. Total time: 163.3s (3.6 env-steps/s, 20 episodes completed)`.

### Test suite (CP8)

```
$ /home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest -q \
    tests/algorithms/dreamer_srl/test_eval_rollout.py \
    tests/algorithms/dreamer_srl/test_eval_recording.py \
    tests/algorithms/dreamer_srl/test_render_upload.py \
    tests/scripts/test_eval_stats_csv_columns.py \
    tests/scripts/test_evaluation_model_rebuild.py
20 passed, 23 warnings in 55.85s
```

All warnings are pre-existing `DeprecationWarning`s unrelated to this change (legacy `predators:`/`neutral_animals:` config keys, Flax `.value` accessor). No test edits were needed — confirms the analysis that none of the five eval-path files read a moved key.

### Speed measurement

**Skipped — this is config-layering only** (one extra YAML file parsed at startup, before the training loop begins). No hot-path code (env step, forward/backward, vmap/jit boundary, observation pipeline) was touched. The Dreamer smoke run above (3.6 env-steps/s at `num_envs=4`, a debug/eval-heavy smoke config) is not a speed benchmark and should not be read as one.

### Deviations from the plan

1. **CP4 grep discrepancy (cosmetic, not a defect).** The plan's checkpoint states `grep -n "configs/train/dreamer_srl.yaml" src/algorithms/dreamer_srl/dreamer_srl_main.py` should return exactly 2 hits. Implemented verbatim per the plan's own File Changes text, the actual count is **3**: the two functional merge-list entries (confirmed separately via `grep -n "^\s*'configs/train/dreamer_srl.yaml',"` → exactly 2 hits, at the intended lines) plus one explanatory comment at the single-config site that itself contains the literal string `configs/train/dreamer_srl.yaml` (copied verbatim from the plan's own AFTER text for that site). The curriculum site's comment deliberately omits the `configs/train/` prefix so it doesn't double-count — the plan's two AFTER snippets are simply inconsistent with each other on this point. Functionally correct (2 real merge sites, confirmed structurally and empirically via the diffs and smoke run); flagging the discrepancy rather than silently editing the plan's checkpoint text.
2. **One incidental one-line comment fix** (L1254 stale line-number reference), described above — same file already in scope, purely a comment, no behavior change.

No other deviations. `train.py`, `configs/train/recurrent_ppo.yaml`, `main.py`, `scripts/eval/dreamer_srl_probe_eval.py`, and all five eval tests were confirmed untouched (`git diff --stat` on each returns empty). `num_envs`, the deprecated `log_interval`/`log_accumulate` path, and `agent.algorithm` additions to `agent_xs.yaml`/`01_food_only_smoke.yaml` were left alone per the plan's Out of Scope section.

### Blockers / follow-up items

None. All ten checkpoints pass.

## Verification Report

> **Verified by**: [agent/person]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `configs/train/dreamer_srl.yaml` | NEW — 5 per-algo keys, no `smoothing_episodes` | | |
| `configs/train/default.yaml` | Remove 5 Dreamer keys; de-Dreamer-ify comments; `num_envs` comment | | |
| `src/algorithms/dreamer_srl/dreamer_srl_main.py` | +1 merge entry × 2 sites | | |
| `docs/environment/CONFIG_GUIDE.md` | Layering + invariant documented | | |

**Conclusion**: [one-line summary]
