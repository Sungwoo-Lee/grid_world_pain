---
title: "Review: H6+H7 DreamerV3-NNX world-model fixes (arrival-obs storage + buffer-wrap alignment)"
topic: dreamer_v3_nnx
status: active
created: 2026-07-05
last_updated: 2026-07-05
---

# Review: H6+H7 DreamerV3-NNX world-model fixes

## Verdict

**APPROVE-WITH-NITS.** The uncommitted change fixes two real training-corruption bugs in the DreamerV3 stack. The first (H6): the world model was being trained with each observation paired to the action the agent chose *after* seeing it — leaking one step of future information — while at act time the model receives the action that *led to* the observation. The fix makes data collection store the observation an action *produces* (sensed after the environment step but before the auto-reset, so the death/timeout observation is kept), which makes the stored rows match the model's step contract with no shift needed at training time. The second (H7): the replay buffer's ring capacity was not a multiple of the training-sequence length, so once the buffer wrapped, sampled "sequences" would splice two different environments' trajectories mid-window; the fix floors the capacity to a multiple of the sequence length at construction and refuses capacities below one block. I verified the PRNG semantics (the extra sensing pass consumes no keys and reproduces bit-identically what the policy sees at the next step), walked the index alignment across training, inference, and imagination, checked episode boundaries, checked the rounding arithmetic against the pre-existing rounding in `train.py`, and confirmed the new tests are content-level and non-vacuous. No blockers. Three nits and one pre-existing concern (already documented as out of scope in the plan) below.

Reviewed against plan doc `docs/develop/active/issues/diag_fable5_20260704/fix_plan_h6h7_dreamer_v3_world_model.md` (§A3) and diagnosis `05_dreamer_v3_nnx.md` rows H6/H7. Scope: `git diff HEAD -- src/models/dreamer_v3_trainer.py` (+50/−1) plus the two new test files. Other in-flight packages in the tree were not reviewed.

## Findings

| Severity | Location | Issue | Suggested fix |
|---|---|---|---|
| 🟡 concern (pre-existing, out of scope) | `src/models/dreamer_v3_trainer.py:566` vs. new comment at `:661-672` | `get_action` hard-codes `is_first = jnp.zeros((B, 1))`, ignoring the `is_first` flag the collect loop sets on the carry (`:697`). The new comment's claim "train-time pairing equals inference-time pairing (get_action)" therefore holds **mid-episode only**: at the first inference step after an auto-reset, the RSSM carries the previous episode's stale latent and stale `prev_action` into the new episode, while training resets the latent at `is_first` rows. Additionally, under the arrival convention the post-reset first observation of each episode never enters replay at all (the model trains on `õ_1`, the arrival of the first action, not `o_0`). Both are acknowledged in the plan (Finding 4 declared out of scope at plan line ~138; the reset-obs gap at line ~111) and are **not introduced or worsened by this diff**. | None required for this diff. Ensure Finding 4 (`get_action` never resets on episode boundary) has a row in the Known Bugs registry via `bug-curator` so it isn't lost; consider softening the `:670` comment to "for all non-reset steps" (the plan's own §A4 wording, line 104). |
| 🟢 nit | `src/models/dreamer_v3_trainer.py:673-675` | Redundant compute: for non-done envs, `obs_arrival` at step *t* is bit-identical to the `obs` sensed at step *t+1* (`:647`) — the arrival state is the same pytree with the same `state.key`, and `get_observation` is a pure function of it. One `get_observation` vmap per scan step could be saved by carrying the arrival obs forward and re-sensing only reset envs. The simple double-sense is a defensible trade (branch-free, keeps the fix one block); cost is a cheap sensor assembly, not an encoder pass. | Optional; leave as-is unless collect profiling says otherwise. |
| 🟢 nit | `src/models/dreamer_v3_trainer.py:1075` | Pre-existing off-by-one adjacent to H7: `sample()` requires `size > sequence_length` **strictly**, so a legally-constructed one-block buffer (`capacity == sequence_length`, which passes the new `rounded_capacity <= 0` raise) can never serve a sample. Harmless in live configs (capacity ≫ one block); noting because the new raise message says the buffer "must be at least one sequence_length", implying one block is usable. | Optional: change `<=` to `<` at `:1075`, or note the two-block practical minimum in the raise message. Do not fix in this diff. |
| 🟢 note (intended behavior change) | `src/models/dreamer_v3_trainer.py:989-993` + `train.py:864-866` | Interaction with the positive-buffer path: `train.py:864` pre-floors `pos_cap` with the identical formula, so the new `__init__` rounding is idempotent there (no double-rounding drift, no spurious log line). Behavior change: `positive_buffer_capacity < sequence_length` previously produced a capacity-0 buffer that would fail later and cryptically (`% 0` in `add_batch`); it now raises `ValueError` at construction. This is an improvement consistent with the no-fallback-defaults rule. | None. |

## Detailed audit

### 1. PRNG semantics of the new sensing pass (H6) — correct

- `get_observation(state, params)` derives its noise key as `jax.random.fold_in(state.key, 999)` (`src/environment/sensor.py:294`) and **consumes no keys** — it is a pure function of the state. The extra call in the new `dreamer_sense_arrival` block (`dreamer_v3_trainer.py:674`) therefore neither advances nor reuses any stream; reproducibility (same initial key + same actions → same episode) is preserved.
- `jax_step` splits `state.key` 6-way (`core.py:505`) and writes the advanced main key into the returned state (`core.py:837`), so `next_state_raw.key ≠ state.key` — the arrival obs draws fresh noise, distinct from the departure obs's noise, as it must.
- **Bit-identity with what the policy consumed**: for non-done envs, the next scan iteration's carry is `next_state_raw` unchanged (the `select_done` merge at `:687-690` picks the non-reset branch), so the `obs` sensed at `:647` next iteration is `get_observation` of the *same* state with the *same* key — bit-identical to the stored `obs_arrival`. The buffer stores exactly the noisy observation the policy acted from at the following step. For done envs, the stored terminal obs (from `next_state_raw`) is never consumed by the policy (which sees the reset obs) — by design; that row exists to keep the death observation in world-model training (plan §A3, Candidate-2 rejection).

### 2. Index alignment walk (H6) — correct

Notation: within an episode, `s_0 →(a_1)→ s_1 →(a_2)→ s_2 …`; `õ_t = get_observation(s_t)` is the arrival obs of `a_t`; `r_t`, `done_t` describe the transition into `s_t`.

- **Storage (new convention)**: scan iteration *t* senses `o = õ_{t-1}` at `:647`, `get_action` picks `a_t` from it, `jax_step` produces `s_t`; row *t* stores `(obs = õ_t, action = a_t, reward = r_t, terminal = done_t, term_reason_t, is_first_t = done_{t-1})` (`:700-724`). Only `obs` changed meaning; all other fields keep their pre-fix index — confirmed field-by-field in the diff.
- **Training scan**: `env_inputs = (action, is_first)` fed unshifted (`:216`); `RSSM.step(prev_state, embed(õ_t), a_t, is_first_t)` advances the GRU with `concat(stoch, a_t)` *before* the posterior reads `embed` (`dreamer_v3_nnx.py:121-135`). So the action input occupies the "action leading into this observation" slot — and under arrival storage `a_t` *is* the action that produced `õ_t`. Pairing correct.
- **Inference (`get_action`)**: `rssm.step(prev_state, embed(current obs), prev_state['prev_action'], …)` at `:594-596`, with `prev_action` set to the last chosen action at `:611`. At collection step *t+1* the current obs is `õ_t` and `prev_action` is `a_t` — the identical `(a_t, embed(õ_t))` pair training consumes at row *t*. Train ≡ inference mid-episode, as claimed.
- **Reward/continue heads**: `posterior_t` (state "at" `õ_t`) predicts `r_t` (arrival reward of `a_t`) and continue-target from `term_reason_t` — the standard DreamerV3 arrival alignment; the death penalty and continue-0 target sit on the row that carries the death observation. Correct.
- **Imagination**: `imagine_step` advances the GRU with the actor's new action from `feat_t` then produces the prior — same slot semantics; consistent with both.

### 3. Episode boundaries (H6) — correct, with one documented asymmetry

- Terminal row *K*: stores `(õ_K = death/timeout obs, a_K, r_K, done=1, term_reason)` — self-consistent, sensed pre-reset. Verified by the boundary test.
- Row *K+1* (first row of the new episode): stores `(õ'_1 = arrival of the new episode's first action a'_1, a'_1, r'_1, is_first = 1)` — `is_first` flows from `next_d_state['is_first'] = done` (`:697`) through the carry into `d_state.get('is_first')` at `:705`, including across `collect_sequence` chunk boundaries (the dreamer state persists in `train.py`). In the training scan, `is_first = 1` zeroes `deter`/`stoch` (`dreamer_v3_nnx.py:117-119`) but — correctly — **not** the action: under arrival storage `a'_1` is legitimate causal conditioning (chosen from `o'_0`, which precedes `õ'_1`), exactly the plan's §115 reasoning. Had `RSSM.step` masked the action sheeprl-style, this convention would be broken; it doesn't, so it isn't.
- The stale prev-action is thereby ignored at boundaries in *training*. At *inference* it is not (Finding 4, `:566`) — see the 🟡 row above; pre-existing and declared out of scope by the plan.
- No H5-class bleed: reward/terminal/term_reason for row *K+1* come from the post-reset step's own `jax_step` outputs; the boundary test pins this.

### 4. H7 rounding arithmetic — correct

- `rounded = (capacity // sequence_length) * sequence_length`: exact multiples pass through unchanged (no log, no behavior change); `capacity < sequence_length` (incl. 0 and negatives) → `rounded <= 0` → loud `ValueError`. No off-by-one at exact multiples.
- Sufficiency: `sample()` reads windows starting at `block_idx * seq_len` with `block_idx < size // seq_len`, so a window never crosses the array end on its own; the splice came purely from the write index drifting off the seq-aligned grid after `idx = (idx + num_items) % capacity` wraps. Writes are always multiples of `seq_len` (`train.py:492-493` forces collect `num_steps == agent.sequence_length`, and `add_batch` receives `B × T` env-major rows), so with capacity now a multiple of `seq_len` the write index stays on the block grid forever. The supplementary test pins exactly this invariant.
- `train.py:864` positive-buffer pre-rounding: identical formula applied first → the constructor rounding is a no-op there. No double-rounding conflict (🟢 note above for the sub-one-block raise).

### 5. JIT / vmap hygiene — clean

- The new sensing pass is a pure traced `jax.vmap(get_observation, in_axes=(0, None))` inside the already-jitted scan — identical pattern to the existing `:647` call. `EnvParams` broadcast (`in_axes=None`), env axis 0: matches project vmap convention. No host-device syncs, no shape variance, no new static args, no recompile triggers. `jax.named_scope` is trace-transparent. The H7 `print` executes in un-jitted `__init__` — no tracer leak.

### 6. Tests — content-level and non-vacuous

- `test_dreamer_collect_arrival_alignment.py`: ground truth by exact replay of the recorded actions (valid because `jax_step` is deterministic given `EnvState.key`), with (a) an explicit non-vacuity guard that departure ≠ arrival obs at *every* compared step (satiation decrement guarantees it), (b) the core arrival assertion, (c) index-stability guards pinning that reward/terminal did **not** move, (d) a precondition assert that the death-free fixture really produced no `done` in the window (replay would silently diverge past one). The boundary test deterministically forces a timeout at row 3 (`max_steps=4`), checks the terminal row stores the pre-reset arrival obs, checks flag placement (`terminal`/`term_reason` on row K, `is_first` on row K+1), and guards the H5 bleed class. Acceptable known gap, acknowledged in the test docstring: row K+1's *obs content* is unverifiable from outside the scan (the reset state is drawn from the scan's internal key stream).
- `test_dreamer_replay_buffer_wrap.py`: the no-splice test is genuinely content-level — every transition of block *b* carries `obs = float(b)`, block ids are unique across all 30 written blocks, so any mid-window splice necessarily changes the obs channel; 50 batches × 32 windows after multiple wraps. The pre-fix failure mode (`112 % 100 = 12 ≡ 4 mod 8`) is arithmetic-checked in the docstring and demonstrably red pre-fix per the plan's verification log. The alignment test re-asserts the write-grid invariant after every `add_batch`. Both non-vacuous.

## Conventions audit

| Convention | Status |
|---|---|
| Pytree immutability (`_replace` / functional updates only) | ✅ (`.at[].set` on buffer arrays; no in-place pytree mutation) |
| JIT (no new static args, no recompile triggers, no host syncs) | ✅ |
| vmap (axis 0 = env, `EnvParams` broadcast, renderer untouched) | ✅ |
| PRNG threading (no key reuse/consumption in new sensing pass; main key advance unchanged) | ✅ |
| Sensor / observation-breakdown sync (no sensor added/renamed) | ✅ n/a |
| Config protocol (no new YAML keys; runtime rounding intentionally leaves config untouched; loud `ValueError` on degenerate capacity) | ✅ |

## Conclusion

APPROVE-WITH-NITS — both fixes are semantically correct and JAX-clean; the only substantive caveat (inference never resets the RSSM at episode boundaries, `get_action:566`) is pre-existing, explicitly out of scope in the plan, and should be tracked in the Known Bugs registry.

Reviewed by: code-reviewer
