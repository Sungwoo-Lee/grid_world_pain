---
title: "Diagnosis 07 — Behavior-Measures Subsystem (M1/M2/M5, episode metrics, distance aggregator)"
topic: issues
status: active
created: 2026-07-04
last_updated: 2026-07-04
---

# Diagnosis 07 — Behavior-Measures Subsystem

## Purpose (plain-language entry point)

This is an independent bug hunt of the code that turns raw training/evaluation steps into
the project's **behavior numbers** — "how often was feeding interrupted by a nearby threat"
(M1), "how often did the agent dive into a bush after a threat appeared" (M2), and "does the
agent eat less when threatened" (M5) — plus the per-episode reward/damage tallies and mean
distances that go to WandB. These numbers feed *analysis and experiment verdicts*, not the
training gradient, so a bug here silently distorts what we conclude an agent is doing, not
what it learns.

A prior audit already fixed several bugs here; this pass **verifies those fixes** and hunts
for new ones. Verdict up front: the core accumulator module (`src/behavior/accumulators.py`)
and its per-step drivers in `train.py` (RecurrentPPO/PPO/DQN/DRQN) and `dreamer_srl` are
sound, and all four prior fixes are correctly in place **on the files they touched**. But I
found **one serious new driver bug** (the DreamerV3 batch path in `train.py` mixes steps
from *after* an episode's end into that episode's numbers), **one silently-dead measure**
(the offline replay used by the current eval protocol can *never* report an interrupted
feeding — its rate is always 0), and several smaller divergences where the offline
cross-check tool implements *different* counting rules than the online accumulator it is
supposed to cross-check — i.e., two prior fixes were complete where applied but were **not
propagated** to the offline sibling.

Per the task brief, the known-undecided **episode-end rule for in-progress M1/M2 events** is
NOT re-reported here; findings below are all distinct from it.

---

## Fix verification (the four fixes I was asked to confirm)

| Fix | Commit | Verdict |
|---|---|---|
| M1/M2 numerator & denominator counted at the same moment | `3e1e53e` | ✅ **Correct and complete in `accumulators.py`** — per-class and per-tag denominators are both incremented at event-record time (`accumulators.py:242,253-257` for M1; `:269,281-285` for M2); numerators resolve against the stored record-time tag index. ⚠️ **Not propagated** to the offline replay in `scripts/eval/eval_rollout.py`, which still counts the M1 denominator at resolution time (Finding 4). |
| Zero-denominator NaN guard for EatUnderThreatRatio | `e5e1155` | ✅ **Correct in `accumulators.py`** — both per-class (`:395`) and per-tag (`:346`) guards are mathematically sound (`eat_safe>0 ⇒ safe_steps>0`). ⚠️ **The same bug class recurs** in the offline replay's ratio (`eval_rollout.py:388-392`), which divides by `max(p_s, 1e-9)` and explodes to ~1e9 instead of NaN (Finding 5). |
| Continual stage-transition NameError → shared reset loop | (memory `20260622_1704`) | ✅ **Verified** — `train.py:1258-1260` loops `_bm_reset_env(_i)` over all envs at the transition, delegating to the shared `bm_reset_env`, which resets every BMState field including K-buffer ages, tag indices, and prev-threat flags. `dreamer_srl_main.py:1368-1375` goes further and *rebuilds* the BMState for the new stage's tag roster — `train.py` does not (Finding 6). |
| Online measures off by default | `9eacf82` | ✅ **Verified** — `configs/environment/default.yaml:234` has `behavior_measures.enabled: false`; the loader still parses the full block; all five `train.py` call sites and the `dreamer_srl` site are gated on `bm_enabled`. |

---

## Finding 1 — DreamerV3 batch driver contaminates episodes across the done boundary

**Severity: High** (analysis numbers; path currently mitigated — see below) · **NEW**

**Where:** `train.py:1655-1666` (BM step loop) vs `train.py:1669-1716` (done handling), the
"Site 2" DreamerV3 batch path.

**What happens.** The DreamerV3 branch collects a whole batch of `T = num_steps`
(collect-interval) steps per iteration, then drives the behavior-measure state machine with
a loop over **all T steps first** (`for t2 in range(T2): _bm_step_update(...)`,
`train.py:1659-1666`), and only **afterwards** walks the done events to finalise and reset
per-env state (`_bm_finalise_episode(i)` at `:1699`, `_bm_reset_env(i)` at `:1715`). The
`done_mask` argument passed into `bm_step_update` is documented as unused
(`accumulators.py:157-160`) — the contract is that the **caller** must reset at each done
boundary, which every other site (RecurrentPPO `:1384/1433`, DQN `:1946/1990`, DRQN
`:2145/2189`, PPO `:2319/2363`, and `dreamer_srl_main.py:1172/1278`) honors by interleaving
step-update and reset inside a per-step loop. Site 2 does not.

**Concrete failure scenario.** Batch of T=128 steps; env 3's episode ends at t=17.

1. The BM loop applies steps t=18..127 — the **new** episode's threat exposure, eats, and
   bush dives — into env 3's still-uncleared accumulators.
2. `_bm_finalise_episode(3)` then reports the *old* episode's M1/M2/M5 rates computed over
   old-episode **plus** 110 new-episode steps (M5 threat/safe step counts, eat counts, any
   M1/M2 events recorded after t=17 all leak in).
3. `_bm_reset_env(3)` then wipes everything — so the new episode **loses** its first 110
   steps of counts.
4. If env 3 finishes a *second* episode inside the same batch, the second
   `_bm_finalise_episode(3)` call runs on the fully-reset (empty) state → all denominators 0
   → all-NaN rates → the episode is silently dropped from the WandB means (NaN-skipped by
   `_append_per_measure_mean`, `train.py:1050-1056`).

Also note: the M1/M2 aging clocks (`m1_candidate_age`, `m2_onset_age`) tick across the done
boundary, so a pre-death feeding-interruption candidate can be resolved using
`m1_steps_since_eat` values that mix two different episodes.

**Mitigation in current practice:** online measures are off by default (`9eacf82`), and live
Dreamer work runs through `dreamer_srl_main.py`, whose driver is correct. But any config
that re-enables `behavior_measures.enabled: true` on a `train.py` DreamerV3 run gets
corrupted M1/M2/M5 numbers with no error.

**Suggested fix direction (for `developer`):** move the `_bm_step_update` call inside a
per-step loop that also handles that step's dones (finalise + reset) before advancing to
t+1 — i.e., restructure Site 2 to the Site 1 (RecurrentPPO) pattern. The episode-scalar
accumulators at Site 2 already segment correctly via `curr_start` slicing; only the BM state
machine is wrong.

---

## Finding 2 — M2 loses an already-achieved bush dive when a new onset overwrites a pending one

**Severity: Med** · **NEW** (distinct from both the fixed denominator-timing bug and the
undecided episode-end rule — this is a mid-episode numerator loss)

**Where:** `src/behavior/accumulators.py:265-271` (onset overwrite resets
`m2_in_bush_seen`), vs resolution at `:292-304`.

**What happens.** M2 keeps **one** pending-onset slot per (env, class). When a new
class-level onset fires while an old onset is still inside its K-step window, the code
overwrites the slot and resets `m2_in_bush_seen[env, c] = False` (`:271`). If the agent had
**already dived into a bush** during the old onset's window (`m2_in_bush_seen == True`), that
successful dive is erased: the old onset was counted into the denominator at record time,
but its numerator credit can now never fire. `BushDiveRate` is biased **downward** exactly in
the scenes where threats repeatedly re-approach.

**Empirically confirmed** (K=10, R=5, single env, single predator): predator enters radius
at t1 (onset A), agent dives into bush at t3, predator leaves at t5, agent exits bush,
predator re-enters at t7 (onset B overwrites A), never dives again →
`m2_onsets = 2, m2_dives = 0`. The behaviorally correct tally is `dives = 1` (onset A *was*
followed by a dive within K steps): measured rate 0.0 vs true 0.5.

**Why the M1 analog is benign** (checked, not a finding): an M1 candidate overwrite requires
`ate_food` this step, which resets `m1_steps_since_eat` to 0, so the overwritten candidate
would necessarily have resolved as *not-interrupted* — dropping its resolution contributes
exactly what resolution would have (0 to the numerator, denominator already counted).

**Also note an internal ordering inconsistency:** M1 ages/resolves *before* recording new
candidates (resolution-first, `:213-232` before `:234-257`), while M2 detects/overwrites new
onsets *before* aging/resolving (record-first, `:263-285` before `:291-304`). So an M2 onset
that would resolve (age K) on the very step a new onset fires is overwritten *before* it can
resolve — the loss window is the full K steps inclusive. Aligning M2 to resolution-first
ordering would fix the boundary case; crediting `m2_in_bush_seen` at overwrite time (resolve
the old onset immediately as a dive) would fix the general case.

---

## Finding 3 — Offline replay can NEVER report an interrupted feeding (rate is always 0)

**Severity: High** (this is the eval-protocol path that current studies actually consume) · **NEW**

**Where:** `scripts/eval/eval_rollout.py:341-350` (`_compute_online_replay`, M1 block).

**What happens.** The offline replay ages and resolves the M1 candidate **before** updating
`steps_since_eat` for the current step:

```python
# M1 — age FIRST (age-first ordering matches train.py)   <- comment is wrong in one detail
if cand_age >= 0:
    cand_age += 1
    if cand_age >= K:
        totals[f"m1_candidates_{prefix}"] += 1
        if steps_since_eat >= K:          # reads the value as of step t-1
            totals[f"m1_interrupted_{prefix}"] += 1
        cand_age = -1
steps_since_eat = 0 if ate[t] else steps_since_eat + 1    # updated AFTER resolution
```

The online accumulator updates `steps_since_eat` **before** aging
(`accumulators.py:211` before `:215-232`). A candidate is recorded at an eat step
(`steps_since_eat := 0`), so by the resolution step (age = K) the *pre-update* value read by
the offline code is at most **K−1** — the test `steps_since_eat >= K` is unsatisfiable.
`interrupted_feeding_rate_*` in `online_replay.json` is therefore **0.0 whenever any
candidates exist** (NaN otherwise), regardless of the agent's actual behavior.

**Empirically confirmed** on an identical synthetic episode (eat once under threat at t2,
threat always in radius, never eat again): online accumulator → `candidates=1,
interrupted=1`; offline replay logic (verbatim ordering) → `candidates=1, interrupted=0`.

**Blast radius.** `online_replay.json` is labeled a "sanity cross-check", but it is consumed
by real analyses (e.g. the round-26 hypervigilance design doc compares `online_replay M2`
values against thresholds, and the discrimination-measures grounding doc plans to *extend*
`_compute_online_replay` for conditional M1/M2/M5 analysis). Since `9eacf82` explicitly moved
measurement to the offline eval protocol, this off-by-one silently zeroes a headline measure
on the now-primary path. Any past conclusion of the form "no interrupted feeding in
condition X" that came from `online_replay.json` is unsupported until recomputed.

---

## Finding 4 — Offline replay implements different denominator rules than the fixed online accumulator

**Severity: Med** · **NEW as a code divergence** (its episode-end facet overlaps the KNOWN-undecided rule; the overwrite facet and the online/offline mismatch do not)

**Where:** `scripts/eval/eval_rollout.py:345` (M1 denominator at resolution time) and `:353`
(overwrite without counting), vs `accumulators.py:242` (record time); also
`eval_rollout.py:357` (M2 denominator at record time — internally inconsistent with the same
function's own M1 choice).

**What happens.** Commit `3e1e53e` fixed the online accumulator so all M1/M2 denominators
count at event-**record** time. The offline replay was not updated in step:

- Offline M1 counts a candidate into the denominator only when it **resolves** (survives the
  full K-step window); candidates overwritten within K steps or pending at episode end are
  never counted at all. Online counts every recorded candidate.
- Offline M2 counts onsets at record time (matching online) — so the two measures inside the
  same offline function follow different conventions.

**Concrete failure scenario.** The same recorded episodes produce different M1 denominators
online vs offline whenever candidates are overwritten or an episode ends mid-window — so the
"sanity cross-check" flags phantom discrepancies (or, combined with Finding 3, fails to flag
real ones). Whichever episode-end rule the user eventually picks (the undecided row), the
offline replay must be updated in the same change or the cross-check stays meaningless.

---

## Finding 5 — Offline eat-under-threat ratio explodes to ~10⁹ instead of NaN when there are no safe eats

**Severity: Med** · **NEW** (recurrence of the `e5e1155` zero-denominator class in a file the fix didn't touch)

**Where:** `scripts/eval/eval_rollout.py:388-392`.

**What happens.** The offline ratio is `p_t / max(p_s, EPS)` with `EPS = 1e-9`, guarded only
by `ts > 0 and ss > 0`. `ss > 0` does **not** imply `p_s > 0`: an agent that never eats in
safety (e.g. a strongly threat-suppressed or non-eating agent) has `p_s = 0`, and the ratio
becomes `p_t · 10⁹` — a finite, astronomically large number that poisons any mean/plot it
enters, unlike the online accumulator which correctly emits NaN (skipped downstream). Same
data, two paths, two answers.

**Fix direction:** replicate the online guard (`ss > 0 and eat_safe > 0`, else NaN).

---

## Finding 6 — `train.py` continual transitions don't rebuild per-tag state for a changed entity roster

**Severity: Low** (loud crash, not silent, for the count-change case; silent mislabeling for the rename case; no current schedule appears to change rosters) · **NEW**

**Where:** `train.py:1003-1008` (tag roster + per-tag arrays sized once, from stage-0
params) and `train.py:1238-1260` (transition wipes values but keeps shapes/names), vs
`dreamer_srl_main.py:1357-1375` (rebuilds `make_bm_state` and per-tag arrays for the new
roster).

**What happens.** The stage-consistency probe (`train.py:551-576`) validates obs/action dims
and a sensor-modality fingerprint, but **not** the predator/neutral entity roster. If a later
stage has a different number of predators/neutrals, the per-step accumulation
`episode_dist_per_predator_sums += info_np['dist_per_predator'][t]` (`train.py:1377-1378`)
raises a shape `ValueError` mid-training (loud). If the counts match but the tags differ,
BM per-tag and mean-distance metrics are silently logged under the **stage-0 tag names**
(and `bm_step_update`'s `min(num_predator_for_log, shape[1])` clamping at
`accumulators.py:188,247` silently aliases extra entities onto the last stage-0 tag slot).
`dreamer_srl_main.py` already handles this correctly; `train.py` should either rebuild (as
dreamer_srl does) or extend the stage probe to reject roster changes explicitly.

---

## Finding 7 — Episode-start "onset" when a threat spawns already inside the cue radius

**Severity: Low** (semantics note, not a state-machine defect) · **NEW**

**Where:** `src/behavior/accumulators.py:142` (`bm_reset_env` clears `m_prev_threat_in_R`)
+ `:265` (onset = rising edge of the class-level in-radius flag).

**What happens.** After an episode reset, the previous-threat flag is False, so if the new
episode spawns a threat already within `cue_radius` of the agent (random placement permits
this), step 1 registers an M2 "threat onset" that is not an approach event — the agent had
no pre-onset baseline to react from. This inflates the M2 denominator with
spawn-configuration events. The offline replay (`eval_rollout.py:322`, `prev_in_R = False`)
has the same convention, so online/offline agree; flagging so the convention is a documented
choice rather than an accident. (The `_detect_threat_onsets` window index for motif
clustering shares it too, `eval_rollout.py:260-261`.)

---

## Non-findings (checked and clean)

- **Per-step drivers at Sites 1/3/4/5 and dreamer_srl**: step-update → finalise → reset
  ordering interleaved correctly at every done; step-`t` info and done-`t` flags come from
  the same `jax_step` (pre-auto-reset — verified in `recurrent_ppo_trainer.py:196-258`, and
  in `dreamer_srl_main.py:1129-1172` where the reset runs after the BM update). No
  cross-episode leakage on these paths.
- **`bm_reset_env` completeness**: all 16 counters + ages + tag indices + `m2_in_bush_seen`
  + both prev-threat flags + `m1_steps_since_eat` are reset — no field survives an episode
  boundary on the interleaved paths.
- **Dtypes**: event counters int64 (per-episode, reset each episode), `episode_counter`
  int64 lifetime, ages int32, distance sums float32 per-episode — no overflow path.
- **Zero-division guards** in `accumulators.py` (class + tag), `episode_metrics.py` (±inf
  reward min/max replaced at finalise), `distance_aggregator.py` (`max(step_count, 1)`) —
  all sound.
- **Logging-boundary handling**: `iteration_episodes` reset/log cadence
  (`train.py:1325-1326`, `:1436`) never double-counts an episode; BM pending state correctly
  persists across iteration boundaries within an episode.
- **Aggregation convention**: WandB numbers are NaN-skipped means of *per-episode rates*
  (`_append_per_measure_mean`), not pooled counts. This weights a 1-candidate episode
  equally with a 20-candidate episode — a documented, deliberate choice
  (`accumulators.py:411-423`), noted here only so analysts don't misread it as pooled.
- **sheeprl bridge** (`pytorch_agents/.../grid_world_pain.py:129-220`): single-env,
  correctly ordered step→finalise→reset; `episode_counter` correctly survives resets.

### Side observation (outside this subsystem — PRNG hygiene in the rPPO rollout)

`recurrent_ppo_trainer.py:216` derives `reset_key, _ = jax.random.split(key)` but returns the
un-advanced `key` in the scan carry (`:258`). Since the next step recomputes
`key, act_key = jax.random.split(key)` (`:184`) from the same input, **step t's `reset_key` is
byte-identical to step t+1's carry key**, and (because `split(k, 2)` and `split(k, N)` share
their first outputs) env 0/1's per-env reset keys at step t coincide with step t+1's
key/act_key. Reproducibility holds, but auto-reset randomness is not independent of the
subsequent action-sampling stream — a violation of the project's key-reuse convention. Not a
behavior-measures bug; handing to whoever owns the JAX-correctness lens
(`diag_v3_pipeline_jax.md` audited `core.py` threading but not this trainer-side splice).

---

## Verdict

The accumulator core and the per-step (interleaved) drivers are **sound**, and all four
prior fixes are correctly in place where they were applied. The area is **not fully sound**,
on two axes: (1) the `train.py` DreamerV3 batch driver violates the accumulator's
reset-at-done contract and mixes episodes (Finding 1 — dormant only because online measures
default off); and (2) the **offline eval-protocol replay has drifted from the fixed online
semantics** — most seriously, its interrupted-feeding rate is structurally pinned to zero
(Finding 3), plus a denominator-rule mismatch (Finding 4) and an exploding ratio (Finding 5).
Since the project's measurement strategy now leans on the offline path, Findings 3–5 are the
ones that can distort live study conclusions today and should be fixed first; Finding 2
(M2 dive lost on onset overwrite) biases BushDiveRate downward on both online and offline
paths and should ride along with whatever episode-end rule the user decides.

Reviewed by: code-reviewer (independent diagnosis, Fable 5 session, 2026-07-04)
