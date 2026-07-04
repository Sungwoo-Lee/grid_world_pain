---
title: "Fable 5 Re-Diagnosis — Combined Report (2026-07-04)"
topic: issues
status: active
created: 2026-07-04
last_updated: 2026-07-04
---

# Fable 5 Re-Diagnosis — Combined Report

## What this is (plain-language entry point)

This is the **combined result of an independent, second-opinion bug hunt over the
project's main training and evaluation code**, run on 2026-07-04. The first
whole-pipeline audit (earlier the same day, by a different model — Opus) found and fixed a
large bug cluster (see [[KNOWN_BUGS]]). The user then asked for the diagnosis to be
**re-done from scratch on the Fable 5 model**: seven parallel Fable 5 reviewers each took one
area of the pipeline — the training entry point and config system, the recurrent-PPO
learner, the environment core, the two Dreamer training stacks, the evaluation path, and
the behavior measures — read the code line by line, verified the earlier fixes, and hunted
for new bugs. Each wrote its own report (files `01_…` through `07_…` in this folder); this
document is the roll-up.

**Headline:** every previously-landed fix was re-verified as correct in the code it
touched — nothing regressed. But the sweep found **10 new High-severity bugs** the first
audit missed, clustering into three themes: **(1) every checkpoint-resume path is broken**
(rPPO restore silently keeps random weights; continual resume rebuilds the wrong-stage
world; Dreamer checkpoints omit optimizer state), **(2) episode/window boundaries are
mishandled in the learners** (rPPO value targets are cut off at every 128-step rollout
window; Dreamer's replay buffer bleeds each episode's terminal reward into the next
episode's first row), and **(3) silent data-layout drift struck again** (the eval stats CSV
mislabels every sensor column; the offline interrupted-feeding metric is structurally
always zero). One PRNG-hygiene bug was found **independently by three of the seven
reviewers** at the same line — the strongest possible static confirmation.

**Good news:** fresh (non-resumed) rPPO training launches on current configs are sound in
their core math — sequence handling, hidden-state resets, loss functions, and the
modulator wiring all check out; the environment core itself is clean. The damage is
concentrated in resume paths, window edges, Dreamer data plumbing, and analysis outputs.

---

## 1. Verification of the earlier (Opus-era) fixes

All fixes from the v3.0 audit and the historical list that fall inside the seven scopes
were re-checked. **All are correct and complete in the files they touched.** Three carry
material caveats about *scope*:

| Fix | Verdict | Caveat |
|---|---|---|
| `22c73ba` config `extends:` resolution | ✅ correct, all train paths | — |
| `ef0fd25` timeout ≠ death (env reward) | ✅ | simultaneous death+timeout resolves to death, correct |
| `3c60f6f` rPPO truncation bootstrap | ✅ correct for GAE | **zero live effect** — all 13 live rPPO configs use MC mode, whose window-edge handling is itself broken (H4 below). The KNOWN_BUGS row overstates live coverage. |
| `926c2c3`, `8c1ad2f` plain-PPO fixes | ✅ | dead path (no live config) |
| `5b093bf` Dreamer continue-head | ✅ end-to-end | dormant trap if `overeating_death` is ever enabled (M-cluster below) |
| `f5df600`+`1703a4c` two-hot layout | ✅ round-trip verified | — |
| `75976e2` CLI flags persisted | ✅ for rPPO/PPO/DQN/DRQN | **inverted for DreamerV3**: value saved to config but trainer built from raw YAML (M1 below). Also: KNOWN_BUGS row L4 is stale — this fix did cover more than the row says. |
| `a3ab4cc`, `863052f`, `2ad9104`, `80d3b70` eval fixes | ✅ all four | minor residuals noted in `06_evaluation_path.md` |
| `3e1e53e` M1/M2 same-moment counting | ✅ in `accumulators.py` | **not propagated** to the offline replay sibling in `eval_rollout.py` (M-cluster below) |
| `e5e1155` zero-denominator NaN guard | ✅ in file touched | same class recurs un-guarded in `eval_rollout.py:388` |
| `3634887` ghost predators, `db8bd03` resource slots, `7ff8d1f` int ranges, `84014e4` step-0 gate | ✅ all | one info-array residue from the ghost fix (L below) |
| `0119e87` persistent compile, Option-S upload, masked reset (dreamer_srl) | ✅ intact | — |
| v1 REINFORCE resampling bug | ✅ not regressed | — |

Latent row `20260509_1536` (Orbax↔NNX restore skew) is **confirmed real** — it is H1 below,
now with mechanism and empirical proof.

---

## 2. High-severity findings (10, all NEW)

"High" = distorts what a live run learns, silently trains/evaluates the wrong thing, or
corrupts the numbers a study consumes.

| # | Finding | Where | What happens | Report |
|---|---------|-------|--------------|--------|
| H1 | **rPPO `--load-checkpoint` never restores** | `train.py:1141-1200` | Restored param paths mismatch live NNX paths on 100% of leaves; `ValueError` swallowed by blanket `except`; training silently continues **from random weights**. Empirically proven. Confirms latent memory `20260509_1536`. | 01 |
| H2 | **Continual resume runs the wrong-stage world** | `train.py:1138-1239` | Resume restores `current_stage`, suppressing the stage-transition env rebuild → stage-N counters on a **stage-0 environment**. Live today for DreamerV3 resume. | 01 |
| H3 | **Typo'd `--config` path trains on the default env** | `src/utils/config.py:30-33` + `train.py:381` | Bad path → one warning, empty config, `default.yaml` merged underneath bypasses mandatory-key protection; run proceeds with no error. | 01 |
| H4 | **rPPO MC returns cut off at every 128-step window edge** | `recurrent_ppo_trainer.py:90-106,315-324` | Live MC mode computes returns per rollout window with carry=0 and **no bootstrap at the edge** (episodes run to 500). Every live rPPO run's value targets are position-dependently biased. | 02 |
| H5 | **dreamer_srl: first buffer row of each episode carries the previous episode's terminal reward + death flag** | `dreamer_srl_main.py:1242-1262` | sheeprl's post-done `step_data` reset block is missing → reward-head bias at episode starts, continue head trained "done" on fresh states, imagination from post-death states discount-zeroed. Three-line fix. | 04 |
| H6 | **DreamerV3-NNX: (obs, action) one-step misalignment** | `dreamer_v3_trainer.py:211` vs `dreamer_v3_nnx.py:105-144` | World-model training pairs `a_t` with `embed_t` while inference/imagination pair `a_{t-1}` → inverse-dynamics leak into the prior, train/inference mismatch, action effects one step late in imagination. | 05 |
| H7 | **DreamerV3-NNX: buffer wrap splices two envs mid-sequence** | `dreamer_v3_trainer.py:997,1035` + `dreamer_v3.yaml:14` | `buffer_capacity` 1,000,000 % `sequence_length` 128 = 64 → after the first wrap (>1M env steps), sampled sequences cross env boundaries with no `is_first`. | 05 |
| H8 | **Eval stats CSV: every sensor column silently shifted one left** | `evaluation_core.py:204-243` | Header omits "Interoceptive Nociception" (on by default) while values are positional → `obs_noc` holds intero-noc, `obs_olf_0` holds extero-noc, last column dropped. Name-based consumers get wrong per-modality numbers, no error. Known silent-drift class. | 06 |
| H9 | **Offline replay's interrupted-feeding rate is structurally always 0.0** | `eval_rollout.py:341-350` | M1 resolved *before* `steps_since_eat` update → condition can never fire. This offline path (`online_replay.json`) is what current studies consume. | 07 |
| H10 | **DreamerV3 batch path corrupts behavior metrics at episode boundaries** | `train.py:1655-1716` | Whole T-step measure loop runs before per-done finalize/reset → post-death contamination, lost opening steps, all-NaN on double-done. Dormant (online measures off by default). | 07 |

## 3. Medium-severity findings (selected; full detail in the per-area reports)

**Checkpoint / resume family (extends H1–H2):**
- Dreamer checkpoints (train.py path) omit **all optimizer states, the target critic, and the Moments normalizer**; restore also swallows exceptions → silent random-weight continuation (`train.py:2461-2471`, `1198-1200`). Sibling of the known dreamer_srl A2 row — where, additionally, **no restore path exists at all** (`checkpoint.py:84-96`).
- DreamerV3 CLI persistence inverted: saved config lies about `--lr`/`--hidden-size`; eval rebuilds the wrong architecture (`train.py:500-510` vs `:838`).
- SIGINT/SIGTERM exits without a final checkpoint save; DQN/DRQN never checkpoint at all.

**PRNG hygiene (triple-confirmed):**
- `recurrent_ppo_trainer.py:216` — reset-key aliasing: reset key at step *t* byte-identical to the scan carry key at *t+1*; per-env reset keys collide with *t+2* action keys. Found independently by agents 02, 03, and 07 with matching proofs. Determinism intact; one-line fix; **stream-breaking** (parity fixtures must be regenerated).
- Env core: one shared `damage_key` for resource/animal/obstacle rolls → rank-correlated multi-source damage (`core.py:571-643`).
- DreamerV3-NNX: model and behavior phases share an `rng`; action-sample and latent-sample reuse one key per step.
- Eval: `eval_rollout.py:98/201` reuses one episode key for every policy step — harmless in the deterministic default, but makes `eval_policy_mode: stochastic` near-deterministic — exactly the mode memory `20260704_2014` (deterministic probes inflate significance) is pushing toward.

**Dreamer recipe drift (undeclared deviations from sheeprl):**
- dreamer_srl observation loss: extra `symlog()` on decoder output + extra 0.5 factor (`train.py:704-712`); **no gradient clipping** anywhere (sheeprl: 1000/100/100); `learning_starts` counted in iterations not env steps (16× longer prefill); replay-ratio remainder dead code; episode logging double-counts terminal rewards.
- DreamerV3-NNX: `get_action` hardcodes `is_first=0` → RSSM/modulator state never reset at episode boundaries during collection; eval `__call__` skips `symlog` on observations and samples the posterior with `PRNGKey(0)` every step (distorts all during-training Dreamer eval); no validation that `collect_interval % sequence_length == 0` (the advertised `collect_interval: 1` would scramble sequences); KL/log-probs/entropy on raw logits while sampling uses 1% unimix.

**Config / dead-key recurrences:**
- Perceptual-noise `mode` typos and misspelled modality keys silently map to "no noise" (`config_loader.py:1506-1527`).
- `behavior_measures.eval_obs_noise` and `eval_max_steps`: mandatory, validated, written to metadata — **never acted on**; `eval_obs_noise: zero` silently runs the noisy env (`eval_rollout.py:572/854`).
- `--total-timesteps` silently ignored whenever episodes > 0 (always, in practice).

**Behavior-measure correctness (beyond H9/H10):**
- New M2 onset overwriting a pending one erases an achieved bush dive → BushDiveRate biased down (empirically: true 0.5 measured 0.0) (`accumulators.py:265-271`).
- Offline/online M1 denominators counted at different moments (`3e1e53e` not propagated) — cross-checks can't reconcile.
- Offline eat-under-threat yields ~1e9 instead of NaN on no-safe-eats (`eval_rollout.py:388-392`).
- Continual stage transition wipes but never rebuilds BM state for a changed entity roster (crash or silent mislabel).

**Environment latent traps (consumer-side, inert in live configs):**
- Instant death with `with_injury:false` gets no termination-reason code → every `reason >= 2` consumer (rPPO GAE, both Dreamer continue heads) treats a real death as truncation — **new load-bearing consequence** since the truncation fixes.
- `overeating_death:true` sets reason=3 on non-terminal full-stomach steps → mid-episode bootstrap zeroing / continue-head "death" on live steps. Triage of the old memory row is now complete: the flag can literally never terminate an episode.

**Eval / tooling:**
- Parallel eval undercounts first-wave episodes by 1 step and drops frame 0 when `record_stats=False` + `num_envs>1` (live via `main.py` sandbox); refilled episodes correct → inconsistent within a run (`evaluation_core.py:534-663`).
- `trajectory_story.py` mis-decodes its own recordings: "Eat" printed at step 0 (`-1` sentinel → `AM[-1]`), action column lags one step; `render_recordings.py` decodes the same schema correctly.
- `render_recordings.py --concat` + `--skip-existing` re-runs omit already-rendered episodes from the consolidated video and `--cleanup-per-episode` then **deletes their only MP4s**; all-exist → WandB upload silently skipped.

## 4. Low / nits

Kept in the per-area reports: phantom-distance info arrays for parked ghost slots (03),
orphaned `grid_world.py` renderer duplicate + dead `dreamer_v3_network.py` linen code
(03/05), saturated-spawn silent (0,0) teleport (03), WandB `define_metric` case mismatch +
`job_type` never passed (01), `evaluation.py` WandB NameError swallowed (06), M2 onset
convention for threats spawning inside the cue radius (07), target-critic init not a copy
(05), config `merge` list aliasing (01), renderer step-0 "--" display (06), stale
comments/docstrings (02/03).

## 5. Cross-cutting themes

1. **Resume is systematically broken across all three trainers** (H1, H2, Dreamer/dreamer_srl optimizer omissions, exception-swallowing restores). Until fixed, treat every `--load-checkpoint` / continual-resume result as suspect; fresh launches are fine.
2. **The silent encode/decode-drift class keeps recurring** (H8 CSV shift, H9 offline metric, trajectory_story decode, non-propagated fixes `3e1e53e`/`e5e1155`). The registry already names this class; it has now bitten in four new places.
3. **Error-swallowing `except` blocks turned three real failures silent** (H1 restore, evaluation.py WandB, mod_grad_norm). A sweep for blanket excepts on load-bearing paths would pay for itself.
4. **`termination_reason >= 2` is a fragile contract** — three consumers inherit two latent env quirks; any future config flipping `with_injury` or `overeating_death` walks into it.
5. **Fable 5 vs Opus:** the seven reviewers confirmed every Opus-era fix and additionally surfaced 10 Highs, largely by (a) empirically executing suspect paths (H1, H9, PRNG proofs) and (b) diffing ports against their upstream recipe (H5, gradient clipping, symlog deviations).

## 6. Suggested priority order (not yet planned — needs routing)

1. **H5** (dreamer_srl buffer bleed — 3-line fix) and **H4** (rPPO window bootstrap) — they bias *every live training run* of their stack.
2. **H1 + H2 + Dreamer checkpoint omissions** — one "fix resume" work package.
3. **H8 + H9** — analysis outputs currently consumed by studies are wrong.
4. **PRNG aliasing** (one-line, but schedule with A1 fixture regeneration since it breaks streams).
5. **H6/H7** before the next result-bearing DreamerV3-NNX run.
6. Med/dead-key/eval items per the per-area reports.

## 7. Per-area reports

| Report | Scope | Highs |
|---|---|---|
| [[01_train_entry_config]] | `train.py`, config loader/save, checkpointing | H1 H2 H3 |
| [[02_rppo_stack]] | rPPO trainer/network, modulator, plain PPO | H4 |
| [[03_env_core]] | env step/reward/termination, sensing, entities | — (2 Med) |
| [[04_dreamer_srl]] | dreamer_srl package | H5 |
| [[05_dreamer_v3_nnx]] | DreamerV3-NNX trainer/networks + train.py loop | H6 H7 |
| [[06_evaluation_path]] | live eval, eval_rollout, recording/rendering | H8 |
| [[07_behavior_measures]] | accumulators, episode metrics, drivers | H9 H10 |

**Registry:** [[KNOWN_BUGS]] (update via `bug-curator`, not by hand) · **Prior audit:** [[v3_pipeline_correctness_diagnosis]] · **Handoff:** [[OPEN_WORK_HANDOFF]]
