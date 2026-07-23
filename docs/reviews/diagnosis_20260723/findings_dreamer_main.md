# Diagnosis findings — src/algorithms/dreamer_srl/dreamer_srl_main.py (2026-07-23)

Reviewer: JAX RL correctness pass on the dreamer_srl entry point (~2096 lines).
Scope: config plumbing, env construction, curriculum/continual handling, eval
scheduling, WandB step alignment, checkpoint cadence, PRNG seeding.
Known-bugs context honored: OPEN rows not re-reported (depth added where allowed);
FIXED rows regression-checked below. File:line refs against branch v3.0 worktree.

---

## Finding 1 — [P2] Eval videos silently dropped from WandB (backward-step rejection)

**Where:** dreamer_srl_main.py:1718-1726 -> src/algorithms/dreamer_srl/eval.py:566-573
(_render_and_upload) -> src/utils/wandb_utils.py:113 (wandb.log(log_dict, step=step)).

**Claim:** Training-time eval videos are uploaded with step=checkpoint_pct where
checkpoint_pct = total_episodes_completed (an EPISODE count). The run's WandB step
counter is driven by the training-loop logs, which pass step=policy_step (an
ENV-STEP count; lines 1071 and 2042). Episodes complete after many env steps, so
total_episodes_completed << policy_step, the video log is a backward step, and
WandB rejects it ("Tried to log to step N < current step M" — the file's own
comment at 1728-1731 confirms this wandb version warns-and-drops on backward
steps). The MP4 exists on disk, but eval/video and eval/checkpoint_episode never
appear in the run.

**Failure scenario:** Any run with training.video_during_training + auto_render
(defaults in configs/train/default.yaml). First checkpoint at 10,000 episodes is
millions of env steps in: upload_video(step=10000) vs internal step in the
millions -> dropped. Every subsequent video likewise.

**Why rPPO does not hit this:** train.py never passes step= to wandb.log (all
call sites step-less; internal counter stays small), so upload_video(step=
<episode count>) is a FORWARD step there. Dreamer-specific interaction between
explicit step=policy_step logging and the shared upload_video(step=checkpoint_pct)
convention inherited from evaluation_core.py:359.

**Fix direction:** In _render_and_upload, call upload_video with step=None (or
pass policy_step); rely on eval/checkpoint_episode in the payload for x-axis.

**Related minor (same family):** the stage-transition log (1655-1662) and the two
Eval/* logs (1733-1740, 1758-1765) are step-less amid explicit-step logs; at
num_envs=1 this advances the internal counter past policy_step and drops the next
1-3 training rows after each eval/stage event. Benign at default num_envs=16.

---

## Finding 2 — [P2] training.eval_stats_num_envs is a dead (but mandatory) key on the Dreamer path

**Where:** dreamer_srl_main.py:617 (only read; never referenced again — grep over
the whole file). Stats pass at 1746-1757 calls the single-env sequential
dreamer_srl_eval_rollout, not dreamer_srl_eval_rollout_batched (whose only
callers are scripts/eval/eval_rollout.py and tests).

**Claim:** Read via get_mandatory (a config lacking it hard-fails) yet has zero
effect: the checkpoint-triggered stats eval always runs eval_stats_episodes
episodes one at a time in one env. The same key is live on the rPPO path
(train.py:2129), so tuning eval_stats_num_envs silently does nothing on Dreamer.

**Failure scenario:** eval_stats_episodes: 100 at max_steps 3000 blocks the
training process for up to 300k sequential env steps per checkpoint regardless of
eval_stats_num_envs; user raises the key, sees no speedup.

**Fix direction:** Route the stats pass through dreamer_srl_eval_rollout_batched
(num_envs=eval_stats_num_envs) or stop reading the key on this path.

---

## Finding 3 — [P2] Agent-config env.num_envs is a dead key (silent config trap)

**Where:** all configs/models/dreamer_srl/*.yaml carry `env: { num_envs: 1 }`
(e.g. 01_food_only.yaml:123-124, "XS default (sheeprl env/default.yaml:L1)").
dreamer_srl_main.py:553 resolves num_envs from CLI --num-envs or
env_cfg.get_mandatory('training.num_envs') only; build_agent reads only
cfg['algo'][...] keys. Nothing reads the agent YAML's env.num_envs.

**Claim:** A sheeprl-heritage key sits in every Dreamer agent config looking
exactly like the parallel-env knob and editing it does nothing — the real value
comes from configs/train/dreamer_srl.yaml:36 (training.num_envs: 16) merged on
the ENV config side. New instance of the "config-boundary silent failures"
family (the KNOWN_BUGS row lists rPPO dead keys; this one is unrecorded).

**Failure scenario:** User sets env.num_envs: 4 in the agent YAML for a debug
run, gets 16 envs; per-env buffer capacity (buffer.size // num_envs, line 749)
and prefill length (line 558) are 4x off from what they computed.

**Fix direction:** Delete the dead env: block from the agent configs, or fail
fast in main() if agent_cfg env.num_envs exists and differs.

---

## Finding 4 — [P2] Episodes-mode "env-step cap" is advertised but never enforced; --total-steps alias contradicts its help

**Where:** dreamer_srl_main.py:570-580 (budget resolution), 1284-1286 (banner
prints "env-step cap: {total_timesteps}"), 1334 (loop condition), 823-824
(WandB config total_steps/total_timesteps).

**Claim:** In episode mode the loop condition is total_episodes_completed <
episodes only — total_timesteps is never checked, despite the banner calling it
an "env-step cap" and WandB recording it. Additionally, --episodes N
--total-steps M together take the line-572 branch where total_timesteps =
env_step_override — so the IGNORED override value is what gets printed and
stored as the run's timestep budget, while --total-steps's help text (434-436)
says it is used only when --episodes is not passed.

**Failure scenario:** A degenerate agent surviving ~max_steps every episode:
--episodes 100000 at max_steps 3000, num_envs 16 can consume up to 4.8B env
steps with no step-level kill switch while the operator believes a cap exists;
analysis reading wandb.config.total_timesteps as the budget gets a number the
run never respected.

**Fix direction:** Enforce policy_step < total_timesteps as an AND condition in
episode mode, or relabel it "estimate" and make --episodes/--total-steps
mutually exclusive.

---

## Finding 5 — [P2] Curriculum stage swap never rebinds env_cfg-derived state: BM toolkit, eval config, budget math stay stage-0

**Where:** swap block dreamer_srl_main.py:1569-1662 rebinds env_params, env, tag
rosters, and BM *state*, but:
- bm_cfg / bm_enabled / bm_R / bm_K (1005-1016) are resolved once from stage-0's
  merged config and reused at the swap's make_bm_state (1643-1650) — a stage
  YAML that enables/disables behavior_measures or changes cue_radius/obs_window
  is silently ignored for the rest of the run.
- Checkpoint-triggered evals pass config=env_cfg (1710, 1750) — stage-0's config
  — into dreamer_srl_eval_rollout, used for visualization.icons + source_path in
  write_run_meta (eval.py:82-90). Stage-specific icons render wrong in
  post-swap eval videos.
- env_max_steps (563) feeding the curriculum total_timesteps estimate (569) is
  stage-0's environment.max_steps.
- Eval cadence knobs (video_during_training etc., 613-622) are frozen at stage 0
  — arguably intended as global policy, but stage-YAML overrides of them are
  silently dead.

**Failure scenario:** 2-stage curriculum where stage 1 introduces a predator and
turns on behavior_measures to measure avoidance: bm_enabled stays False (stage 0
had no BM block) -> all BM metrics absent exactly for the stage that needed
them, no warning.

**Evidence:** bm_cfg = load_behavior_measure_cfg(env_cfg) runs once pre-loop
(1005); swap step 5 (1632-1650) rebuilds only the state with captured bm_R/bm_K;
env_cfg assigned once (513/539), never reassigned.

**Fix direction:** At the swap re-derive bm_cfg/bm_enabled/bm_R/bm_K from
schedule.stage_configs[_new_stage] and pass that stage config to the evals; or
fail fast at schedule build if stage configs diverge on these blocks (mirroring
the modality-fingerprint check).

---

## Finding 6 — [P2] Legacy logging path with --no-wandb leaks iteration_episodes unboundedly

**Where:** dreamer_srl_main.py:1475-1476 (append when logging_cfg is None),
1983-1985 (the only clear site, gated `if logging_cfg is None and use_wandb and
iteration_episodes:`).

**Claim:** On the legacy path (config with no logging: block — only standalone/
archived configs, since configs/train/default.yaml declares one) combined with
--no-wandb, every finished episode's dict is appended and never cleared:
unbounded memory growth.

**Failure scenario:** Long archived-config run with --no-wandb: millions of
episodes x ~30-key dict each -> multi-GB RSS creep, eventual node OOM blamed on
JAX.

**Fix direction:** Clear iteration_episodes at the log gate regardless of
use_wandb.

---

## Finding 7 — [P2, root-cause depth on KNOWN OPEN "eval curve mixes two estimators"]

**Where:** dreamer_srl_main.py:1733-1740 (video-pass log), 1758-1765 (stats-pass
log); eval.py:97 (key = jax.random.PRNGKey(seed)).

Depth added to the known row (not a new finding):
1. Both passes log the IDENTICAL keys Eval/MeanReward / Eval/MeanLength with the
   SAME timesteps: policy_step value and no pass label — on the timesteps axis
   the N=3 and N=100 estimators land at the same x, producing the sawtooth.
   One-line fix: distinct keys (Eval/Video* vs Eval/Stats*) or log stats only.
2. The two estimators are NESTED, not independent: both rebuild PRNGKey(args.seed)
   and split identically per episode, and the actor is deterministic argmax — so
   the video pass's 3 episodes are bit-identical to the first 3 of the stats
   pass's 100. The video pass adds zero statistical information.
3. Every checkpoint reuses the same fixed seed=args.seed, so eval initial
   conditions are identical across all checkpoints (paired eval — likely
   intended variance reduction; the eval seed is also the training seed, so it
   is not held-out in any seed sense; no env reset-key collision with training
   because ParallelEnv.reset splits its key per env, wrapper.py:18).

---

## Finding 8 — [P2, low] Stage-boundary checkpoint is evaluated on the NEW stage's env

**Where:** swap block (1569-1662) runs before the checkpoint+eval block
(1672-1765) within the same dones iteration.

**Claim:** When an episode boundary coincides with a checkpoint multiple
(common: boundaries are typically multiples of the checkpoint frequency), the
checkpoint holding pure old-stage weights is saved with stage=<new stage> and
immediately evaluated with the NEW stage's env_params — the Eval point at the
boundary measures a never-trained env attributed to the old stage's final
checkpoint. stage=<new> is defensible for resume semantics; the eval attribution
is misleading (survival-step discontinuity at boundaries that is an ordering
artifact, not learning).

**Mitigating context:** rPPO (train.py) orders swap-then-checkpoint the same way
— shared quirk, not a Dreamer regression.

**Fix direction:** Run the boundary checkpoint+eval before applying the swap, or
tag the Eval row with stage/index.

---

## Fixed-bug regression check (code side)

| FIXED row | Verdict | Evidence |
|---|---|---|
| Curriculum-swap counter skew (C1) | intact | _advance_episode_counters(..., stage_swapped=) no-ops on swap (380-406; flag set 1614; call 1783-1784); test_episode_metrics.py present |
| Continual resume ran wrong stage's world | n/a here / intact | dreamer_srl_main.py has NO resume CLI path (grep: none); checkpoints persist stage=current_stage (1689) so a future resume can restore the right stage; the fixed bug lived in train.py's live path (untouched) |
| Gamma typo in 18 configs | intact | all 18 configs/models/dreamer_srl/*.yaml carry gamma: 0.996996996996997 (uniform; grep/uniq) |
| H5 terminal step_data reset | intact | _reset_terminal_step_data (359-377) called at 1524; second reset_data row written 1499-1508 |
| Prefill counted in env steps (P6) | intact | derive_prefill (utils.py:62-82) = sheeprl // num_envs + off-by-one; ratio gate uses policy_step - prefill_steps * num_envs (1799) |
| extends: ignored by loader | intact for env configs | single-config (539) and per-stage (133) paths both route through load_env_config (resolves extends). Note: agent configs use plain Config.load_yaml (541) on BOTH Dreamer and rPPO (train.py:397) — consistent project-wide, no regression; no dreamer agent config currently uses extends |
| Grad clipping / buffer write-head / episode-logging leak / recompile storms | intact at driver level | clipped optims via make_optim_tx(..., CLIP_NORM) (734-736); per-env EnvIndependentSequentialReplayBuffer with buffer_size // num_envs + validate_per_env_capacity fail-fast (749-779); fixed-width masked Player resets (251-271); fixed-width reset_data (1499-1508); constant-shape autoreset key split (1547-1548); constant _G scan length (1224) |

## Reviewed but clean

- Prefill/train gate boundary: iter_num <= learning_starts (random) overlapping
  iter_num >= learning_starts (train) at the boundary iteration is line-for-line
  sheeprl parity (dreamer_v3.py:558/660), not an off-by-one.
- Player state at prefill->policy handoff (h0 while mid-episode, is_first=0)
  matches sheeprl's identical structure.
- Scan-path PRNG: main key enters _scan_grad_steps and is rebound from the
  returned carry (1914-1925) — no key reuse; legacy path splits per grad step;
  auto-reset keys freshly split per done-iteration at constant shape.
- Polyak in scan path: tau=1.0 at step_idx==0, % target_update_freq gate —
  parity with legacy loop and sheeprl.
- Quantized _G grad steps vs exact Ratio cadence: declared deviation D-015.
- Eval observations DO carry perceptual noise: get_observation applies noise
  internally (sensor.py:290-292, apply_noise=True default) — no train/eval obs
  distribution mismatch; eval.py's "no noise API" comment refers only to
  true-obs recording.
- Checkpoint cadence math (// freq crossing with last_ckpt_episode) handles
  multi-episode iterations and per-stage frequency changes correctly.
- No final checkpoint if the run ends off-cadence — but rPPO (train.py:2098,
  periodic only) behaves identically; shared latent quirk, project-level note.
- Single-config budget from env_cfg.training.* — known documented behavior.
- WandB define_metric routing (uppercase prefixes, Episode/Number axis, stage/*
  on Episode/Number) matches the actually-logged keys.
- num_envs = args.num_envs or get_mandatory(...) — or-short-circuit hazard
  explicitly guarded by default=None (documented in-line).
- Modality-fingerprint pre-flight across stages (660-712) correctly fails fast
  on obs/action-dim or sensor-semantic drift.
- _emit_episode_row closure over rebindable neutral_tags/predator_tags —
  correct late-binding by design (documented CLOSURE HAZARD, 1027-1033).
- step_data["obs"] view-aliasing across the swap is handled (re-sync 1593-1601).
