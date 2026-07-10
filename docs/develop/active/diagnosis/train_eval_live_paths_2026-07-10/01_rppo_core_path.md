---
title: "Live-path audit step 2/4 — rPPO core path end-to-end (train.py + evaluation.py)"
topic: diagnosis
status: active
created: 2026-07-10
last_updated: 2026-07-10
---

# rPPO core path — end-to-end plumbing audit (train.py + evaluation.py)

## Purpose (plain-language entry point)

This is step 2 of a four-step audit of the code paths the project actually trains and
evaluates with. It walks the **main agent's (Recurrent PPO) plumbing** end-to-end: the
script that launches training (`train.py`, freshly slimmed by the archival of the abandoned
in-house Dreamer — the rPPO path is now essentially its whole purpose) and the older
checkpoint-evaluation script (`evaluation.py`). Prior audits already covered the learning
math, the config system, and the evaluation internals; this pass hunts what none of them
owned — the *glue*: how episodes are counted and attributed, how the agent's recurrent
memory is carried across iterations, curriculum-stage switches, and resumes, whether what
training **saves** matches what evaluation **rebuilds**, and whether the Dreamer archival
left damage behind.

**Headline:** the learning path itself is sound, but **curriculum (multi-stage) runs log
their per-entity behavior metrics under the wrong entity names after a stage switch** —
live today in the double-return curriculum, where the predator's distance curve is
silently attributed to a different predator from stage 2 onward. A second real gap:
resuming a single-config run pairs the checkpoint's saved recurrent memory with
freshly-reset environments. Everything else found is Low-severity hygiene. 12 new
findings (1 Med, 1 Med-Low, 10 Low/nit), 6 known issues re-confirmed, and an explicit
sound-list below.

Scope fence honored: NMN/modulation internals (step 3 — a flow-through map is provided
below), GAE/MC return math (H4-audited), config loader internals, `evaluation_core.py`
internals (area-06), and the collector reset-key PRNG reuse (known, scheduled with A1)
were **not** re-derived.

---

## NEW findings

### N1 — Per-tag / behavior-measure accumulators are sized and named from stage-0 params and never rebuilt at stage transitions (Med)

**Where:**
- capture: `train.py:945-950` (`neutral_tags = tuple(params.neutral_tags)`, `predator_tags`,
  `num_*_for_log`, `episode_dist_per_*_sums` arrays) and `train.py:965-972`
  (`make_bm_state(num_predator_tags=..., num_neutral_tags=...)`)
- not refreshed at: the in-loop stage-transition block `train.py:1129-1174` (rebuilds
  `params`/`env`, wipes accumulator *values*, but keeps stage-0 tag lists and array widths)
  and the H2 continual-resume rebuild `train.py:1086-1105` (same omission; tags are captured
  at line 945, *before* the resume rebuild runs)
- consumers using the stale roster: extraction gates `train.py:1215-1216`
  (`if num_neutral_for_log > 0: info_np['dist_per_neutral'] = ...`), accumulation
  `train.py:1248-1251`, per-episode finalisation `train.py:1277-1287`
  (`_bm_finalise_episode(i, ...)` → stage-0 `predator_tags`/`neutral_tags` names),
  WandB fan-out `train.py:1345-1351` and `_append_per_tag_means` / `_bm_log_wandb`
  (`train.py:1000-1051`)
- missing guard: the continual pre-flight validation `train.py:552-581` checks obs/action
  dims and the sensor-modality fingerprint across stages, but **not** the entity roster
  (predator/neutral counts and tag names).

**What happens.** `EnvParams` is rebuilt per stage, and `jax_step`'s `step_info` arrays
(`dist_per_predator`, `dist_per_neutral`) are sized by the *current* stage's roster — but
every host-side per-tag structure keeps stage-0's shape and names. Three failure modes:

1. **Tag-name drift (live today).** In the live curriculum
   `configs/continual/nmn_double_return_stages/`, the predator's tag is `"full"` in stage
   `01_active_predator.yaml` (line 153) but `"TL"` in `02_passive_predator.yaml` (line 188).
   From the first stage switch onward, the passive TL-quadrant predator's distance and
   behavior-measure numbers are logged under the stage-0 keys
   (`Episode/MeanDistPredator_full`, `..._predator_full` BM keys). The values are
   positionally correct; the *names* are wrong — exactly the silent
   encode/decode-drift class the registry keeps re-finding. Any per-tag WandB analysis of
   a continual run (the double-return design is *about* TL/BR quadrant behavior) reads
   mislabeled curves with no error.
2. **Silent metric loss (0 → N).** A curriculum whose stage 0 has no predators
   (`num_predator_for_log = 0`) never extracts `dist_per_predator` in any later stage —
   predator distance metrics and M1/M2 predator measures are simply absent for the
   predator stages, with no warning.
3. **Crash (N → M, N ≥ 1).** If a later stage has a *different nonzero* entity count,
   `episode_dist_per_predator_sums += info_np['dist_per_predator'][t]`
   (`train.py:1249-1251`) is a `(num_envs, N) += (num_envs, M)` broadcast → `ValueError`
   on the first post-transition iteration. (`bm_step_update` itself is guarded with
   `min(...)` — `src/behavior/accumulators.py:188-192` — so the BM side degrades silently
   instead.)

**Status: NEW** (the 07-04 rPPO audit verified the stage-transition block wipes and resets
correctly, but only for the structures it rebuilds; the roster staleness was not covered;
no KNOWN_BUGS row exists). Training math is unaffected — `dist_per_*` feeds logging only —
hence Med, not High; but note the name-drift mode silently corrupts precisely the per-tag
telemetry the double-return study reads.

**Suggested direction:** either (a) re-derive `neutral_tags`/`predator_tags`/array widths +
rebuild `_bm_state` inside the stage-transition block (and the H2 resume block), or
(b) extend the pre-flight validation at `train.py:552-581` to require an identical
`(predator_tags, neutral_tags)` roster across stages, matching the spirit of the
fingerprint check. (b) is smaller and makes the invariant explicit.

### N2 — Single-config resume pairs the checkpoint's mid-rollout recurrent state with freshly-reset environments (Med-Low)

**Where:** `train.py:1069` (`h_state = restored_meta['h_state']`) vs `train.py:735`
(`env_state, obs = env.reset(env_key, num_envs)` — the only env init on the single-config
path; `env_state` is **not** part of the checkpoint payload, `train.py:1982-1992`).

**What happens.** The checkpoint saves `h_state` captured mid-rollout, but not the
environment state it corresponds to. On a single-config `--load-checkpoint` resume the
environments are freshly reset, yet the restored stale `h_state` is kept — so every env's
first post-resume episode starts with recurrent (and modulator) memory of a different,
now-nonexistent episode, while `episode_returns`/`episode_lengths` say "fresh episode".
The continual-resume path does this right (`train.py:1100-1101` re-inits `h_state` after
the rebuild). Transient — the trainer's per-done reset cleans each env after its first
episode ends — but it contaminates the first window(s) of every resume and contradicts the
hidden-state-reset discipline verified elsewhere.

**Status: NEW.** The H1 fix plan ([[fix_plan_h1h2h3_resume_config]], "Design decisions"
line 296) explicitly preserved this as "existing behavior, untouched" without flagging the
fresh-env mismatch; since resume only became functional with `9ee2a30`, no historical run
was affected. **Suggested:** re-init `h_state = model.initial_state(num_envs)` on the
single-config resume path too (and stop saving/restoring `h_state`, or keep it only as a
compat field), since without `env_state` in the payload the restored value can never be
correctly paired.

### N3 — Orphaned `bm_drive_batch` import (archival leftover) (Low)

`train.py:70` imports `bm_drive_batch` from `src.behavior.accumulators`; its only caller
(the Dreamer batch path, H10) was deleted with the NNX archival (`fd7af84`). Confirmed:
exactly one occurrence in the file (the import itself). Matches the archival verifier's
flag. Remove on next touch.

### N4 — Unknown `agent.algorithm` spins the training loop forever (Low)

`train.py` validates `DreamerV3` (fail-fast stub, `train.py:456-463`) but has no terminal
`else` in the algorithm-init chain (`train.py:759-922`) nor in the loop body
(`train.py:1201/1415/1601/1807`). Any other string (typo'd `RecurrentPP0`, a future name)
reaches the `while` loop at `train.py:1120` and spins forever — no branch runs,
`total_episodes_completed` never advances, no error, no output. Add an
`else: raise ValueError(...)` at phase 6.

### N5 — `--debug` traceback on during-training eval failure is imported but never printed (Low)

`train.py:2071-2074`: the eval `except` prints one warning line; the `if args.debug:`
branch does `import traceback` and nothing else — `traceback.print_exc()` is missing, so
debug mode adds zero information on eval failures. Pre-existing (verified present before
the archival commit); never registered.

### N6 — `Time/sps_env` is wildly inflated after a resume (Low)

`train.py:1400`: SPS = `global_step / (now - start_time)`. `global_step` is restored from
the checkpoint (`train.py:1071`) but `start_time` is this process's launch time — the first
post-resume logs report absurd steps-per-second until fresh steps dominate. Cosmetic; use a
process-local step counter for the rate.

### N7 — `iteration_episodes` not cleared at a stage transition; old-stage episodes logged under the new stage tag (Low)

The stage-transition block (`train.py:1129-1174`) wipes per-env accumulators but not the
iteration-level `iteration_episodes` list (cleared only by the log-window rule at
`train.py:1198`). With the live rPPO defaults (`configs/train/recurrent_ppo.yaml`:
`log_interval: 500`, accumulate mode), a transition mid-window blends up to 500 iterations
of old-stage episodes into the first post-transition WandB row, which carries the **new**
stage's `stage/index`/`stage/name` tag (`train.py:1318`). Consistent with the documented
"drift is accepted" stance, but worth one line in the block if per-stage attribution ever
matters.

### N8 — evaluation.py never passes `wandb_enabled` into the evaluator — `--wandb-run-path` uploads are dead even past the known NameError (Low)

`evaluation.py:429-432` calls `evaluate_jax_checkpoint(...)` without `wandb_enabled=True`;
the evaluator's uploads are gated on it (`evaluation_core.py:243`, `:357`). So even if the
KNOWN `wandb_login` NameError (`evaluation.py:310`, registry row "Old eval script…/
wandb_login") were fixed, this script still could never upload. Residual on a known row;
deprecated script.

### N9 — `evaluation.py --seed 0` silently ignored (Low)

`evaluation.py:250`: `seed = args.seed or config.get_mandatory('testing.seed')` — a `0`
seed is falsy and falls back to the config. `train.py` handles the same flag correctly with
`is not None` (`train.py:520`). Same latent pattern at `evaluation.py:251` for
`--episodes 0` (degenerate anyway).

### N10 — evaluation.py's eval/vis default paths are cwd-relative (Low)

`evaluation.py:245-246` uses `"configs/evaluation/default.yaml"` /
`"configs/visualization/default.yaml"` relative to the invoker's cwd, guarded by
`os.path.exists` → invoked from outside the repo root the merge is **silently skipped**.
Mostly masked because train-time already merged those keys into the saved `config.yaml`,
but it is the soft-fail-on-missing-file pattern H3 was about. `train.py` anchors the same
paths on `__file__` (`train.py:304,326,334,342`).

### N11 — Unused imports in evaluation.py (nit)

`evaluation.py:43` `glob`, `:44` `re`, `:47` `jnp`, `:58` `ParallelEnv`, `:59` `jax_step`
are all unused (only `jax_reset`, `get_observation`, `get_observation_breakdown` of that
block are used). Remove on next touch.

### N12 — Post-archival cosmetic leftovers in train.py (nit)

- `train.py:683`: `wandb.define_metric("WorldModel/*", ...)` — no live metric family
  matches (Dreamer-only); harmless dead pattern.
- `train.py:253,275`: `--profile` help/trace-dir still named
  `dreamer_v3_vs_rppo_profile`.
- `train.py:1425/1441/1618/1631`: redundant local re-imports of `jax_step`/`jax_reset`
  (already imported at `train.py:75`).
- `train.py:2044`: `import subprocess, sys` inside `main` shadows module-level `sys`
  (no earlier use of `sys` inside `main`, so currently harmless — but an easy future
  `UnboundLocalError`).

## KNOWN issues re-confirmed still present (registry/area-audit rows; not re-counted)

| Known row | Where confirmed | Note |
|---|---|---|
| WandB define_metric case mismatch — rPPO logs lowercase `loss/*`, `modulator/*` while patterns register uppercase | `train.py:680-683` vs `train.py:1366-1394` | Unchanged; loss/modulator curves still fall to the `*` catch-all (timesteps axis). Area-01 F9. |
| `--total-timesteps` accepted, echoed to WandB, ignored as budget | `train.py:499`, `train.py:1120` | Unchanged. Area-01 F4. |
| Graceful shutdown discards progress since last periodic save | `train.py:1121-1122`, `2076-2083` | `stop_requested` breaks with no final checkpoint. Area-01 F9. |
| `main.last_checkpoint_save` function-attribute state | `train.py:1973-1979` | Unchanged; new benign interplay: first post-resume iteration always triggers an immediate checkpoint + eval (attr starts at 0 while episode count is large). |
| evaluation.py `wandb_login` NameError swallowed | `evaluation.py:310,316-317` | Unchanged (see N8 for the additional layer). Area-06 F8. |
| DQN/DRQN never checkpoint | `train.py:1981-1992` (`ckpt_data` empty) | Unchanged; restore now *loudly* unsupported for them (`train.py:1076-1079`) — an improvement. Area-01 F9. |

Registry-hygiene note: the open **L4** row's Dreamer arm ("CLI overrides not saved") is
moot post-archival — its only remaining live residue is `--total-timesteps`.

## Map of NMN (modulation) flow-through points — for step 3

Treated as opaque pass-through here; these are every touchpoint in the two audited files:

| Point | Where | What flows |
|---|---|---|
| Config read | `train.py:769-771` / `evaluation.py:382-384` | `config.get('agent.modulation')`; both None it when `type` is null — identical logic, single source (saved config.yaml) |
| Model construction | `train.py:782-792` / `evaluation.py:386-396` | `modulation_config`, `observation_breakdown` (from params), `encoding_config` = whole `agent` dict |
| Hidden state | `train.py:821, 1101, 1172`; template `src/utils/checkpoint_restore.py:48` | `model.initial_state(num_envs)` bundles modulator hidden with task RNN hidden; saved/restored as opaque `h_state` |
| Runtime telemetry | `train.py:1222` (`trajectories.mod_info`) → `1225-1230` (debug print), `1375-1394` (WandB `modulator/*`; `beta_*` only when type ∈ {PreActivation, FiLM}), `1410-1411` (tqdm temperature) | z_unimodal/z_multimodal/z_memory/temperature stats |
| Gradient telemetry | aux index 4 of the loss tuple (`recurrent_ppo_trainer.py:342`) → `train.py:1362` → `modulator/grad_norm` (`train.py:1377`) | note area-02's 4c: extraction is `except: pass`-guarded in the trainer |
| Checkpoint | `train.py:1984` | modulator params are ordinary members of `nnx.state(model, nnx.Param)` — no special handling anywhere |
| Eval | `evaluation.py:386-396` + `evaluation_core` `model.initial_state` | no other NMN-specific logic on the eval side of these files |

## Explicitly checked and found sound (non-findings)

- **H1 restore module** (`src/utils/checkpoint_restore.py`, `9ee2a30`): restore-to-target
  with the exact save-payload mirror; fatal on missing steps, structure, shape, or dtype
  mismatch (a num-envs-changed resume fails loudly via the `h_state` template); model +
  optimizer updated in place; counters returned. Matches the regression test.
- **H2 continual-resume rebuild** (`train.py:1086-1105`): schedule-derived stage trusted
  over the checkpoint field (with a visible notice), env rebuilt unconditionally, `h_state`
  re-initialized, key split correctly. (Roster staleness N1 is the one thing it misses.)
- **Checkpoint save ↔ restore ↔ eval contract**: save payload (`train.py:1982-1992`) =
  restore target (`checkpoint_restore.py:45-54`) key-for-key; `evaluation.py` needs only
  `restored['model']` + `restored['stage']` — both present.
- **Same-quantity derivations agree across the two files**: `action_dim`
  (`4 + rest + eat`: `train.py:740` / `evaluation.py:368`), `input_dim` (batched
  `obs.shape[-1]` vs single-env `obs.shape[0]` — equal), `observation_breakdown`
  (stage-0 vs checkpoint-stage params — equal under the fingerprint invariant
  `train.py:552-581`), `hidden_size`/`rnn_type`/`activation`/`encoding_config`/
  `modulation_config` (all from the saved config, which train.py populates *before* the
  dump at `train.py:609-611`, including CLI overrides — the L4 fix holds for rPPO).
- **Call contract into `evaluate_jax_checkpoint`**: the evaluator takes its environment
  exclusively from `params` and reads only `testing.*` / `visualization.*` from `config`
  (`evaluation_core.py:256,277,284,328,333`) — so train.py's continual-mode combination
  (stage-0 `config`, current-stage `params`, `train.py:2015-2036`) is safe; video pass is
  single-env and stats pass has `record_stats=True`, avoiding area-06's Finding-2 corner.
- **`train_iteration` call contract**: `nnx.jit(train_iteration, static_argnums=(6,))` —
  arg 6 is the hashable `PPOConfig` NamedTuple; 6-tuple return matches the unpack at
  `train.py:1204-1206`; loss aux 5-tuple indices 0-4 match `train.py:1358-1362`.
- **h_state lifecycle (non-resume)**: carried through the jitted iteration, per-env reset
  on done inside the trainer, re-zeroed at stage transitions — consistent with the 07-04
  audit's read.
- **Episode accounting**: returns/lengths persist across iteration boundaries (episodes
  spanning windows counted correctly); `termination_reason` read at the done step
  `[t][i]` (`train.py:1274`) pre-reset; per-episode reset complete (behavior sums, per-tag
  sums, BM state); BM ordering is the H10-correct step-update → finalise → reset
  (`train.py:1253-1257` → `1287` → `1306`); Site-1's un-cast float `ate_food` is safe
  (`bm_step_update` does `.astype(bool)` internally, `accumulators.py:171-172`).
- **Stage-swap counter skew (dreamer C1 analogue)**: not present — the transition block
  runs at iteration top, resets env + counters *before* any collection, so no phantom step.
- **Checkpoint cadence**: floor-to-multiple update of the save marker is correct across
  multi-boundary iterations; save uses episode count as the Orbax step — consistent with
  evaluation.py's numeric-dir discovery (`isdigit()` also filters Orbax tmp dirs).
- **evaluation.py stage resolution** (`863052f`) and **restore-merge machinery**
  (`2ad9104`: peel → merge → `_assert_full_restore`): regression-glance clean; the only
  change since the 07-04 verification is the archival's DreamerV3 fail-fast branch
  (`evaluation.py:415-425`), which is correct and deliberately reachable for old results
  dirs. Known residuals (coverage-not-shape check; raw `yaml.safe_load` at `:354` vs
  `_load_eval_config`) unchanged, already on record in area-06.
- **Archival damage sweep**: both files byte-compile; no dangling references to removed
  Dreamer symbols (`Ratio`, `ReplayBuffer`, `DreamerTrainer`, …); the DreamerV3 stubs in
  both files raise with a clear pointer to `dreamer_srl`; the rPPO path's own code is
  untouched by the archival diff apart from the whitelist guard. Leftovers are cosmetic
  only (N3, N12).
- **Shutdown paths**: signal handler → finish-iteration → break → wandb.finish +
  checkpointer.close — orderly (the missing final save is the KNOWN row, not new).

## Verdict

**The rPPO core path trains, saves, and re-evaluates the right thing on the standard
single-config path.** The H1/H2 resume rebuild is solid, the save/restore/eval contracts
are mutually consistent, and the archival left the live path clean apart from cosmetic
orphans. Two genuine gaps remain, both in the *continual/resume* corners: the stage-0
entity-roster freeze (N1) silently mislabels per-entity telemetry in the project's live
double-return curriculum — fix or fence before the next continual analysis leans on
per-tag curves — and the single-config resume should stop reusing a recurrent state whose
world no longer exists (N2). Everything else is hygiene.

Reviewed by: code-reviewer (live-path audit step 2/4, Fable 5, 2026-07-10)
