---
title: "Fix plan — WP-D: DreamerV3-NNX world-model data pipeline (H6 obs/action misalignment, H7 buffer-wrap splice)"
topic: issues
status: active
created: 2026-07-06
last_updated: 2026-07-06
---

# Fix plan — WP-D: DreamerV3-NNX world-model data pipeline (H6 + H7)

> **Status**: PLANNED
> **Opened**: 2026-07-06
> **Related**: [[05_dreamer_v3_nnx]] (Findings 1 & 2 — source of record), [[00_combined_diagnosis]] (§2 rows H6, H7), [[KNOWN_BUGS]] (open rows for both), [[sheeprl_training_howto]] (locates the vendored sheeprl reference checkout)

---

## Context

This plan fixes the two remaining High-severity data-pipeline bugs in the DreamerV3 world-model agent trained by `train.py` (the "learn a model of the world, then practice inside your own imagination" algorithm). Both were found by the 2026-07-04 Fable 5 re-diagnosis.

**Bug 1 (H6 — action paired with the wrong timestep).** The world model is trained with each observation paired to the action the agent chose *after seeing it*, while at act time (and inside imagination) the model receives the action *that led to* the observation. The model therefore learns a cheating shortcut — the action leaks information about the observation it is supposed to predict — and is then used under the opposite convention, so imagined action effects arrive one step late. This plausibly explains the project's long-standing "Dreamer underperforms Recurrent PPO despite excellent model losses" pattern. Crucially, the obvious repair (shift the action array at training time) turns out to be **wrong in this codebase** — it would silently erase all death events from world-model training (§Analysis A3). The correct fix is one line of semantics in the data-collection loop: record the observation an action *produces* (including the death observation that auto-reset currently discards) instead of the observation the action was chosen from.

**Bug 2 (H7 — replay buffer splices two environments after wrap).** The replay buffer's capacity (1,000,000 transitions) is not divisible by the 128-step training-sequence length (remainder 64). Once a run exceeds ~1M environment steps and the ring buffer wraps, every write lands mis-aligned with the fixed sampling grid, and sampled "sequences" increasingly contain half of one environment's trajectory glued to half of another's, with no episode-boundary marker. The fix is to round the capacity down to a multiple of the sequence length at buffer construction — exactly what the sibling "positive buffer" already does.

**Consequence for science:** both fixes change what DreamerV3-NNX trains on. Post-fix runs are **not comparable** to any pre-fix run.

Only H6 and H7 are in scope. Everything else from the DreamerV3-NNX report is explicitly fenced out (§Scope fence).

---

## Analysis

### A1. The canonical convention, pinned (sheeprl checkout + paper)

Reference: the vendored upstream checkout at `vendor/sheeprl/` (see [[sheeprl_training_howto]] for provenance; this is the code whose drop-in run empirically outperformed our stack).

**Storage convention** — `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:82-90` documents it as a diagram, and the env loop implements it:

```
# Actions:           a0       a1       a2      a4
#                    ^ \      ^ \      ^ \     ^
#                   /   \    /   \    /   \   /
#                  /     v  /     v  /     v /
# Observations:  o0       o1       o2       o3
# Rewards:       0        r1       r2       r3
# Dones:         0        d1       d2       d3
# Is-first       1        i1       i2       i3
```

Buffer row *t* holds: observation `o_t`; action `a_t` chosen **at** `o_t` (outgoing); reward `r_t` received **arriving at** `o_t` (i.e. produced by `a_{t-1}`); done `d_t` = "`o_t` is a terminal observation". Rewards/dones are written one env-step *after* the obs/action of the same row (`dreamer_v3.py:586-587` adds the row with obs+action; `:628-637` writes the *next* row's rewards/terminated from the step result). Row 0 of an episode has reward 0, done 0, is_first 1.

**Terminal rows are stored.** On done, sheeprl adds an extra row carrying the episode's **final (death/timeout) observation** together with the arrival reward and done flag and a zero action (`dreamer_v3.py:639-650`, `reset_data`), then zeroes the next row's reward/done and sets its is_first (`:652-656`). So the death observation, death reward, and terminated=1 all sit on one self-consistent row.

**Training-time pairing** — `dreamer_v3.py:102-104`:

```python
# Given how the environment interaction works, we remove the last actions
# and add the first one as the zero action
batch_actions = torch.cat((torch.zeros_like(data["actions"][:1]), data["actions"][:-1]), dim=0)
```

so the RSSM step at index *i* consumes **`a_{i-1}` together with `embed(o_i)`** — the action that *led into* the observation. `dreamer_v3.py:100` additionally forces `is_first[0,:] = 1` on every sampled sequence. Inside the cell, `RSSM.dynamic` (`vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:396-435`) resets the recurrent state on is_first (`:427-428`, state ← learned initial; the action input is *not* masked — sheeprl relies on the stored zero action), advances the GRU with `cat(posterior, action)` (`:432`), and only *then* reads the embedded observation in the representation model (`:434`). Rewards (`:180`) and continue targets (`:167-168`) are consumed **unshifted** — they are already arrival-convention in storage, and the recurrent feature at index *i* contains `a_{i-1}` through the GRU, so "reward caused by `a_{i-1}`, arriving at `o_i`" is predictable from it.

**Paper**: DreamerV3 (Hafner et al., 2023, arXiv:2301.04104), world-model Eq. (3): sequence model `h_t = f_φ(h_{t-1}, z_{t-1}, a_{t-1})`; encoder `z_t ~ q_φ(z_t | h_t, x_t)`. The action entering the recurrent update at time *t* is **`a_{t-1}`**, and reward/continue predictors target the reward/continuation associated with arrival at `x_t`.

**Pinned convention:** *at every consumption site, the action fed to the RSSM alongside `embed(x_t)` is the action whose execution produced `x_t`; the reward/continue targets at index t describe arrival at `x_t`.*

### A2. What our stack does (three call sites disagree)

| Site | Code | Action paired with `embed_t` / state | Verdict |
|---|---|---|---|
| Inference / collection | `src/models/dreamer_v3_trainer.py:557,589-591` (`get_action`: `rssm.step(prev_state, embed, prev_action, …)`) | `a_{t-1}` (`prev_state['prev_action']`) | canonical |
| Imagination | `:385-395, 410-416` (`imagine_step(prev_state, action)`; reward/continue/value read at `next_feat` — the **arrival** state) | source-state action drives transition; heads read at arrival | canonical |
| **World-model training** | `:199-219`, inputs built at **`:211`** `env_inputs = (action, is_first)` — no shift | **`a_t`** (chosen *at* `obs_t`, after seeing it) | **wrong** |

Root cause of the training-side pairing: `collect_sequence` records, at row *t*, the **pre-step** observation together with the action chosen from it (`:673-675`: `'obs': obs, 'action': one_hot(action_idx)`), and `train.py:1585-1600` flattens those rows into the buffer unshifted. `RSSM.step` (`src/models/dreamer_v3_nnx.py:105-144`) has the same internal structure as sheeprl's `dynamic` — is_first mask on `deter`/`stoch` only (`:117-119`), GRU advanced with `concat(stoch, action)` (`:121-128`) *before* the posterior reads `embed` (`:133-135`) — so its `action` argument is unambiguously "the action leading into this observation" (`a_{t-1}`-slot). Training violates that; inference and imagination honor it.

Our storage convention therefore differs from sheeprl's in exactly two coupled ways: (i) reward/terminal/term_reason at row *t* are **outgoing** (produced by row *t*'s own action, arriving one step later), not arrival; (ii) **no terminal-observation row exists** — auto-reset (`:650-663`) replaces the post-death state before the next row's obs is sensed, so the death observation never enters the buffer. `is_first` is already arrival-convention (`:670,678`: row *t*'s is_first = previous step's done).

### A3. Why the "obvious" train-time shift is wrong here (load-bearing derivation)

Two candidate train-time-only fixes at `:211`, both rejected:

**Candidate 1 — shift actions only** (`prev_actions = concat(zeros[:, :1], action[:, :-1])`, feed with unshifted rewards/terminals). The recurrent feature at index *t* then contains `a_{t-1}` but the reward head's target at *t* is still `r_t` — the reward produced by **`a_t`**, which is no longer anywhere in the feature. The reward and continue heads degrade from action-conditioned predictors to policy-averaged ones ("expected reward from this state under the data-collection policy"), and in imagination the immediate reward of an imagined action is no longer credited to that action. This is precisely Finding 1's consequence 3 ([[05_dreamer_v3_nnx]]): the heads only fit their same-index targets today *because* the feature (wrongly) contains `a_t`. Rejected.

**Candidate 2 — shift the whole set** (actions, reward, terminal, term_reason all shifted right by one; arrival convention à la Hafner). For non-boundary rows this is exactly canonical. But at an episode boundary, the arrival row of the *fatal* transition would be the next episode's first row — whose observation is the **fresh reset observation** (the death observation was discarded by auto-reset). Two sub-cases, both fatal:
- *Unmasked*: the reward head is trained to predict the −100 death reward, and the continue head to predict "dead", from a fresh-spawn observation with a freshly reset recurrent state. This is exactly the H5 bug class ("first buffer row of each episode carries the previous episode's terminal reward + death flag") that was just fixed in the sibling `dreamer_srl` stack. Corruption.
- *Masked at is_first rows*: the shifted death reward and death continue-target are the **only** rows where `reward = −100` and `continue-target = 0` occur (every `termination_reason ≥ 2` row's arrival is an is_first row). Masking them removes every death event from world-model training: the continue head trains on all-ones targets, imagination never truncates at death, the −100 penalty vanishes from the reward head. The agent could no longer learn death avoidance through imagination at all. Catastrophic.

**Conclusion:** with one buffer row per env step and auto-reset discarding the terminal state, no train-time re-indexing can be canonical *and* keep the death signal. sheeprl squares this circle by writing an extra terminal-observation row (`reset_data`, A1). The equivalent for our fixed-shape `lax.scan` collection is to change **which observation a row stores**.

### A4. Chosen fix for H6 — record the *arrival* observation at collection time

**Change:** in `collect_sequence` (`src/models/dreamer_v3_trainer.py:631-699`), record at row *t* the observation **produced by** row *t*'s action — sensed from `next_state_raw` (the post-step, **pre-reset** state, so the death/timeout observation is retained) — instead of the pre-step observation the policy acted from. **Every other field keeps its current index**: `action`, `reward`, `terminal`, `termination_reason`, `is_first`, and all behavior-info arrays are untouched.

Resulting row semantics: row *t* = (obs `õ_t` = arrival of `a_t`, action `a_t`, reward `r_t` arriving with `õ_t`, terminal "`õ_t` is terminal", reason for that termination, is_first "`õ_{t-1}` ended an episode"). This is **exactly sheeprl's post-shift per-index pairing** (A1): the stored action *is* the prev-action for the stored obs, so the training scan at `:211` needs **no shift at all** — the existing `env_inputs = (action, is_first)` becomes correct as-is. Verification against all consumers:

| Consumer | Under new semantics | Status |
|---|---|---|
| RSSM training scan (`:199-219`) | GRU gets `a_t` before posterior reads `embed(õ_t)` — the action that produced the obs. Prior no longer sees the inverse-dynamics leak; matches paper Eq. 3 | fixed |
| Reward head (`:240-242`) | target `r_t` = reward produced by `a_t`, which sits inside `feat_t` via the GRU; death rows now pair −100 with the actual death observation | fixed (and strengthened) |
| Continue head (`:251-253`, commit `5b093bf` semantics) | `compute_continue_target(term_reason_t)` untouched; the death/timeout target now pairs with the arrival feature that actually encodes the terminal observation | preserved, strengthened |
| Decoder recon (`:236-237`) | reconstructs `õ_t` from `feat_t` — same-index, still consistent; the model now also learns what death states look like | fine |
| Inference (`get_action`, `:557,589`) | feeds (`embed(current obs)`, action that led to it) — for all non-reset steps the current obs *is* `õ_t` of the previous step's row: train ≡ inference | fixed |
| Imagination (`:370-431`) | reward/continue/value already read at the **arrival** state (`next_feat`); heads now trained in arrival convention — fully coherent; **no change needed** | fixed by consistency |
| Lambda returns / discount weights (`:441-452`) | indexing untouched, semantics now globally consistent | no change |
| `train.py` buffer flatten (`:1585-1600`), stats (`:1623-1624`), behavior measures (`:1637-1650`) | consume `reward`/`terminal`/info fields, whose indices are unchanged; `obs` used only for shape + buffer write | no change |
| Recurrent-state carry across collect calls, stage transitions | untouched | no change |

Accepted residuals (document in code comment):
- The **reset observation itself never enters replay** (one obs per episode; sheeprl stores it as its own is_first row, which our one-row-per-step scan cannot afford — we keep the load-bearing death row instead). Distributionally it is near-identical to the first-arrival obs one step later. The policy still *acts* on reset observations during collection; the encoder sees them at inference only. Negligible.
- At an is_first row, the action input is the new episode's **real first action** (a causally valid input — it precedes the obs), not sheeprl's stored zero. The RSSM state is still reset by the is_first mask (`dreamer_v3_nnx.py:117-119`). No leak: the action was chosen from an observation *earlier* than the one being predicted.

**Considered and rejected (record for the developer — do NOT add these):**
- *Zeroing the action input inside `RSSM.step` on is_first rows* (suggested in Finding 1's sketch): under the new storage the incoming action at is_first rows is legitimate conditioning; zeroing would discard it and desync from `get_action` (which never zeroes). sheeprl only zeroes because its is_first row stores a zero action.
- *Forcing `is_first[:, 0] = 1` at train time* (sheeprl `:100`): a no-op here — our initial scan carry is all-zeros (`rssm.initial`), and the is_first mask multiplies `deter`/`stoch`, which are already zero at row 0. sheeprl needs it because its initial recurrent state is learned. Don't add dead code.
- *Making reward/continue heads take the action as explicit input* (`r(s,a)` variant): fixes credit assignment but deviates architecturally from canonical DreamerV3, changes head input dims (checkpoint break), and requires imagination/eval rewiring. Strictly dominated by the chosen fix.

Cost: one extra `vmap(get_observation)` per collection step (sensing runs twice per step: pre-step for the policy, post-step for the record). Collection is a small fraction of Dreamer wall-time (training scans dominate); the mandatory speed check (below) confirms.

### A5. H7 — buffer wrap splices two environments

`ReplayBuffer` (`src/models/dreamer_v3_trainer.py:952-1014`) stores env-major 128-step blocks and samples only at offsets that are multiples of `sequence_length` from index 0 (`:1034-1041`). Writes wrap at `% capacity` (`:997`, `:1013`). Each `add_batch` writes `num_envs × collect_interval` items (a multiple of 128 in live configs), but the live capacity `1,000,000` (`configs/models/dreamer_v3/dreamer_v3.yaml:14`) satisfies `1,000,000 % 128 = 64`: after the first wrap the write grid is shifted 64 slots relative to the fixed sampling grid, so sampled windows contain steps 64–127 of env A's block followed by steps 0–63 of env B's block — a hard mid-sequence teleport with no `is_first` marker, which the RSSM trains through as a real transition. Each further wrap shifts by another 64; corruption grows toward 100% of the buffer. Runs under 1M env steps are unaffected.

Precedent: the positive buffer is already rounded at `train.py:862-865` (`pos_cap = (pos_cap // seq_len) * seq_len`) — the constraint was understood there and missed for the main buffer (`train.py:850-856` passes raw `buffer_capacity`).

**Decision (made by the parent session, recorded here):** fix by **runtime rounding in code** — floor capacity to a multiple of `sequence_length` — **no YAML edit** (`configs/` is `experiment-designer` territory; runtime rounding keeps the fix code-side). Log the effective capacity whenever rounding changes it.

**Placement:** the single choke point is `ReplayBuffer.__init__` (`:953`), which both construction sites (`train.py:850` main, `train.py:866` positive) flow through. This honors the intent behind the parent's cited sites (`:997`, `:1035` — the wrap and sampling lines whose contract the rounding restores): rounding once at construction makes every `% capacity` and every seq-aligned sample start provably safe, provided writes stay multiples of `sequence_length` (true in live configs; the missing `collect_interval % sequence_length` validation is Finding 6, out of scope, still open). The existing `train.py:864` pre-rounding becomes redundant but harmless — leave it (surgical-changes rule).

Effective live capacities after fix: main `1,000,000 → 999,936` (7,812 blocks of 128); positive buffer unchanged (already rounded). Note: `_scan_train_gpu` receives buffer capacity as a static arg — the developer must confirm every capacity consumer reads `buffer.capacity` *post-rounding* (Checkpoint 6).

### A6. Interaction check with landed fixes (required)

- **Continue-head timeout fix (`5b093bf`)** — no conflict. `compute_continue_target` and the `termination_reason` threading are untouched; `term_reason` stays at the same row index. The H6 fix *strengthens* the fix's intent: the death/timeout target now pairs with a feature that encodes the actual terminal observation instead of the pre-step one. The regression test `tests/models/test_dreamer_continue_truncation.py` is pure-function-level and unaffected.
- **Two-hot layout fix (`f5df600` + `1703a4c`)** — no interaction. Encode/decode layout is orthogonal to data alignment; no encode/decode call site changes; −100 remains exactly representable. The reward head simply retrains on correctly paired targets.
- **H5 fix class (dreamer_srl buffer bleed)** — the chosen H6 design was selected specifically to *avoid* introducing this class here; the boundary regression test below asserts no reward/flag bleed into a new episode's first row.
- **Not touched / not worsened**: Finding 4 (`get_action` hardcodes `is_first=0` — collection-time state never resets at episode boundaries) exists identically before and after this fix; out of scope.

---

## Implementation Plan

### Design

Two independent, minimal changes in one file, plus two new test files:

1. **H6**: in `collect_sequence`, sense the arrival observation from `next_state_raw` (after the env step, before the auto-reset selection) and store *it* in the transition's `obs` field. Add convention-pinning comments at the three consumption sites so nobody "fixes" it back.
2. **H7**: floor `capacity` to a multiple of `sequence_length` in `ReplayBuffer.__init__`, raise if the result is zero, log when rounding changes the value.

**Order of work (bug-triage discipline):** write both regression test files FIRST, run them against unmodified code, confirm every designated pre-fix-failing test is red for the predicted reason, then implement, then confirm green.

### File Changes

#### 1. `src/models/dreamer_v3_trainer.py` — `collect_sequence` (H6; current lines ~631-699)

```python
# BEFORE (inside scan_fn):
            # 3. Step Environment
            with jax.named_scope("dreamer_env_step"):
                action_idx = action_idx.astype(jnp.int32)
                next_state_raw, reward, done, info = jax.vmap(
                    jax_step, in_axes=(0, 0, None))(state, action_idx, params)

            # 4. Auto-Reset
            ...
            transition = {
                'obs': obs,
                'action': jax.nn.one_hot(action_idx, ...),

# AFTER:
            # 3. Step Environment
            with jax.named_scope("dreamer_env_step"):
                action_idx = action_idx.astype(jnp.int32)
                next_state_raw, reward, done, info = jax.vmap(
                    jax_step, in_axes=(0, 0, None))(state, action_idx, params)

            # 3b. Sense the ARRIVAL observation (post-step, PRE-reset).
            # H6 fix (WP-D): the buffer row for this step stores the observation
            # PRODUCED BY this step's action — including the terminal (death/
            # timeout) observation that the auto-reset below would otherwise
            # discard — NOT the observation the policy acted from. This makes
            # the stored (obs, action) pair match the RSSM's step contract
            # (action = the action leading INTO the observation; see
            # dreamer_v3_nnx.py RSSM.step and sheeprl dreamer_v3.py:82-104),
            # so the training scan consumes rows unshifted and train-time
            # pairing equals inference-time pairing (get_action).
            # See docs/develop/active/issues/diag_fable5_20260704/
            # fix_plan_h6h7_dreamer_v3_world_model.md
            with jax.named_scope("dreamer_sense_arrival"):
                obs_arrival = jax.vmap(get_observation, in_axes=(0, None))(
                    next_state_raw, params)

            # 4. Auto-Reset  (unchanged)
            ...
            transition = {
                'obs': obs_arrival,   # H6: arrival obs (was: pre-step `obs`)
                'action': jax.nn.one_hot(action_idx, ...),
```

Everything else in the transition dict, the auto-reset block, the carry handling (`next_d_state['is_first'] = done`), and the return signature: **unchanged**. The pre-step `obs` variable is still needed (it feeds `get_action`) — do not remove it.

Also update the `collect_sequence` docstring to state the row convention explicitly (one short paragraph: "row t stores the observation resulting from row t's action; reward/terminal/term_reason at row t describe that same transition; is_first marks rows whose predecessor ended an episode").

#### 2. `src/models/dreamer_v3_trainer.py` — convention comment at the training scan (H6; current line ~211)

No functional change. Immediately above `env_inputs = (action, is_first)` add:

```python
                # Convention (H6 fix): batch rows store the ARRIVAL observation
                # with the action that produced it, so `action[t]` is already the
                # prev-action for `embed[t]` — feed UNSHIFTED. Do not add a
                # sheeprl-style shift here; the shift is baked into storage at
                # collect_sequence. See fix_plan_h6h7_dreamer_v3_world_model.md.
```

#### 3. `src/models/dreamer_v3_trainer.py` — `ReplayBuffer.__init__` (H7; current lines 952-957)

```python
# BEFORE:
class ReplayBuffer:
    def __init__(self, capacity=10_000, sequence_length=16, obs_dim=33, action_dim=4, device="gpu"):
        self.capacity = capacity
        self.sequence_length = sequence_length

# AFTER:
class ReplayBuffer:
    def __init__(self, capacity=10_000, sequence_length=16, obs_dim=33, action_dim=4, device="gpu"):
        # H7 fix (WP-D): writes wrap at `% capacity` (add_batch) while sample()
        # only reads at offsets that are multiples of sequence_length from 0.
        # If capacity is not a multiple of sequence_length, the first wrap
        # shifts the write grid relative to the sampling grid and sampled
        # sequences splice two envs mid-window with no is_first marker.
        # Floor capacity to a multiple of sequence_length (runtime rounding —
        # config YAML intentionally untouched). Mirrors the positive-buffer
        # precedent at train.py (pos_cap rounding).
        rounded_capacity = (capacity // sequence_length) * sequence_length
        if rounded_capacity <= 0:
            raise ValueError(
                f"ReplayBuffer capacity ({capacity}) must be at least one "
                f"sequence_length ({sequence_length})."
            )
        if rounded_capacity != capacity:
            print(f"[ReplayBuffer] capacity {capacity} is not a multiple of "
                  f"sequence_length {sequence_length}; rounded down to "
                  f"{rounded_capacity} (H7 fix — keeps sample windows "
                  f"env-aligned after buffer wrap).")
        self.capacity = rounded_capacity
        self.sequence_length = sequence_length
```

All array allocations below already use `capacity` — change them to use `self.capacity` (or a local `capacity = rounded_capacity` rebind; developer's choice, but every subsequent use of the raw argument must see the rounded value).

#### 4. `tests/models/test_dreamer_collect_arrival_alignment.py` (NEW — H6 regression tests)

Follow the docstring style of `tests/models/test_dreamer_continue_truncation.py` (plain-language context header). Construction: build the env from the default config (`src/utils/config.get_default_config()` + `load_env_params`), pick/override an env config **in the test** so that (a) no death can occur in the test window (e.g. no predators; starvation horizon ≫ window) and (b) for the boundary test, `max_steps` is small (e.g. 4) so a timeout done fires deterministically. Build `DreamerTrainer` the way `train.py:838-841` does (env-probe `input_dim`/`action_dim`, agent config from `configs/models/dreamer_v3/dreamer_v3.yaml` loaded via `Config`, or a minimal inline config dict providing every `get_mandatory` key — developer's choice; keep model sizes small if configurable). B=1 env, CPU is fine.

Ground-truth replay is exact because `jax_step(state, action, params)` (`src/environment/core.py:500`) is deterministic given the state (the PRNG lives in `EnvState.key`, `src/environment/state.py:89`) — replay the recorded actions from the same initial state up to (not across) the first done.

- **`test_stored_obs_is_arrival_of_stored_action`** — MUST FAIL PRE-FIX.
  1. `s_0` = vmapped `jax_reset` (B=1, fixed key); run `trainer.collect_sequence(env_state, params, num_steps=6, key)`.
  2. Precondition: assert `transitions['terminal']` is all-zero in the window (config guarantees it).
  3. Replay: `s_{t+1} = jax_step(s_t, argmax(transitions['action'][t, 0]), params)`, collecting `r_t^replay` and `obs_arr_t = get_observation(s_{t+1}, params)` and `obs_dep_t = get_observation(s_t, params)`.
  4. Non-vacuity guard: assert `obs_arr_t` differs from `obs_dep_t` for every compared `t` (the interoceptive satiation channel decrements each step, so consecutive observations always differ).
  5. **Core assertion (train ≡ inference pairing)**: for every `t`, `transitions['obs'][t, 0] == obs_arr_t` (allclose). Rationale spelled out in the test docstring: the training scan (`dreamer_v3_trainer.py:211`) feeds the row's own `(obs, action)` pair to `RSSM.step`, while inference (`get_action`, `:557,589`) feeds `(current obs, the action that produced it)`; the row pair equals an inference pair iff the stored obs is the arrival obs of the stored action. Pre-fix the stored obs is `obs_dep_t` → fails at step 4→5.
  6. Index-stability guards (pass pre- AND post-fix; pin that only `obs` moved): `transitions['reward'][t, 0] == r_t^replay`, `transitions['terminal'][t, 0] == done_t^replay`.

- **`test_terminal_row_keeps_arrival_obs_and_no_bleed_into_next_episode`** — MUST FAIL PRE-FIX (via assertion a).
  Same setup with `max_steps = 4`, `num_steps = 6`; timeout done fires at window step `k = 3`.
  - (a) **fails pre-fix**: `transitions['obs'][k, 0]` equals the *terminal arrival* observation `get_observation(jax_step(s_k, a_k, params))` — the pre-reset state — and does **not** equal the departure obs `get_observation(s_k)`.
  - (b) boundary flags: `transitions['terminal'][k, 0] == 1`, `transitions['termination_reason'][k, 0] == 1` (timeout), `transitions['is_first'][k+1, 0] == 1`.
  - (c) H5-class bleed guard (invariant; passes pre- and post-fix, pins that the fix never regresses into the dreamer_srl H5 bug): `transitions['terminal'][k+1, 0] == 0` and `transitions['termination_reason'][k+1, 0] == 0` — the new episode's first row carries no leftover death/timeout flags.

#### 5. `tests/models/test_dreamer_replay_buffer_wrap.py` (NEW — H7 regression tests)

Pure `ReplayBuffer` unit tests, no env or trainer needed. Use `device="gpu"` (JAX arrays run fine on the CPU backend) with explicit PRNG keys for deterministic sampling.

- **`test_capacity_rounded_down_to_sequence_multiple`** — MUST FAIL PRE-FIX: `ReplayBuffer(capacity=100, sequence_length=8, obs_dim=1, action_dim=2).capacity == 96`.
- **`test_capacity_below_sequence_length_raises`** — MUST FAIL PRE-FIX (no exception today): `pytest.raises(ValueError)` for `capacity=5, sequence_length=8`.
- **`test_wrapped_buffer_never_splices_two_envs_mid_sequence`** — MUST FAIL PRE-FIX; content-level, not arithmetic.
  1. `buf = ReplayBuffer(capacity=100, sequence_length=8, obs_dim=1, action_dim=2)` (pre-fix: stays 100; post-fix: rounds to 96).
  2. Write 30 env-major blocks of 8 (in `add_batch` calls of 2 blocks = 16 items each, mimicking `num_envs × collect_interval` writes), where every transition in block *b* has `obs = float(b)` (a block-identity channel) and `is_first = 1` only on each block's first slot — enough to wrap the buffer several times.
  3. Sample repeatedly (e.g. 50 batches × 32 sequences, fresh subkeys); for **every** sampled sequence assert the obs channel is constant across all 8 steps (single-env content) — or, stricter, that any change point coincides with `is_first == 1` (there will be none within a block).
  4. Pre-fix: after the first wrap (`idx` lands at `112 % 100 = 12`, i.e. ≡ 4 mod 8), 8-aligned sample windows straddle two blocks → obs channel changes mid-window with no is_first → red. Post-fix (capacity 96): grid stays aligned forever → green.
- **`test_write_index_stays_block_aligned_across_wrap`** (supplementary arithmetic pin, fails pre-fix): after each `add_batch` in the scenario above, `buf.idx % buf.sequence_length == 0`.

### Config keys

None added, removed, or changed. **No file under `configs/` is touched** (H7 is runtime rounding by design; see A5).

### Test plan

Interpreter: `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python`.

1. **Red phase**: add both test files, run
   `…/python -m pytest tests/models/test_dreamer_collect_arrival_alignment.py tests/models/test_dreamer_replay_buffer_wrap.py -v`
   against unmodified source. Expect exactly the five designated tests red (both H6 tests; all three primary H7 tests + the supplementary one), each for the predicted reason. Record the pre-fix failure output in the Implementation Report.
2. **Fix**, re-run: all green.
3. **Regression sweep**: `…/python -m pytest tests/ -x -q` (or the project's usual invocation). **Known-red baseline (pre-existing, do NOT chase, do NOT count against this change): 4× A1 parity fixtures + 3× stale-config (`b093023`) + 1× dreamer_srl offline-WM smoke.** Any *new* red beyond these eight must be investigated before reporting.
4. **Smoke run**: a short DreamerV3 training (a few hundred env steps + a couple of train steps, existing dreamer config, CPU or a free lab GPU) to confirm the collection scan compiles, the buffer constructs with the logged `999,936` rounding message, losses are finite, and no shape errors. This is not a learning-quality check.
5. **Speed check (mandatory)**: measure collection/iteration wall time (s/it over ≥100 iterations after warm-up, same machine/config/seed) before and after. H6 adds one `vmap(get_observation)` per collection step (sensing ×2 on the collection path only). Expected impact well under 5% of total iteration time; record before/after numbers in the Implementation Report. Per verification policy: >5% slowdown warrants discussion, >15% blocks.

### Comparability caveat (must be echoed in the Implementation Report)

Both fixes change the data DreamerV3-NNX trains on (H6 changes the meaning of every buffer row; H7 changes long-run buffer content). **Post-fix runs are not comparable to any pre-fix DreamerV3-NNX run** — Dreamer-vs-rPPO comparisons and modulated-Dreamer conclusions must be re-established on post-fix runs. No checkpoint-format change (no array shapes change), but pre-fix checkpoints embed the wrong convention and must not be resumed for result-bearing work.

### Scope fence

**In scope:** H6 and H7 only, as specified above.
**Explicitly OUT of scope** (all recorded in [[05_dreamer_v3_nnx]] / [[KNOWN_BUGS]]; do not fix "while you're in there"):
- Finding 4 — `get_action` hardcodes `is_first = 0` (collection-time state never resets at episode boundaries);
- Finding 5 — eval `__call__` skips `symlog` and reuses `PRNGKey(0)`;
- Finding 3 — Dreamer checkpoints omit optimizer/target-critic/Moments state;
- Finding 9 — KL/log-prob/entropy on raw logits vs 1% unimix sampling;
- Finding 6 — `collect_interval % sequence_length` validation (note: H7's safety *assumes* writes are multiples of `sequence_length`, true in all live configs; F6 remains the guard gap);
- Finding 8 — PRNG hygiene; Finding 7 — overeating/continue interaction; Finding 10 nits.

## Checkpoints

- [x] 1. Both new test files written first and confirmed RED on unmodified code, each failing for the predicted reason (paste the assertion messages into the Implementation Report). — 6 designated tests red pre-fix, messages in Implementation Report.
- [x] 2. H6: in the modified `collect_sequence`, the pre-step `obs` still feeds `get_action`, and `obs_arrival` is sensed from `next_state_raw` (NOT from `final_env_state` — sensing after reset-selection would silently reintroduce the bug for boundary rows). — confirmed in diff; step 2 (get_action from pre-step `obs`) untouched; sensing block sits between env step and auto-reset.
- [x] 3. H6: only the `'obs'` entry of the transition dict changed; diff shows `action/reward/terminal/is_first/termination_reason` and all info arrays untouched; no changes to `get_action`, `imagine_step`, `model_loss_fn` logic (comment only), `dreamer_v3_nnx.py`, or `train.py`. — confirmed via `git diff` (single-line transition change + comment blocks only).
- [x] 4. H6: `collect_sequence` still compiles under `nnx.jit` with no new static args and the scan carry structure is unchanged (no retrace loop). — signature/static_argnums untouched; tests + 30-episode smoke run compile and run cleanly.
- [x] 5. H7: constructing the live config's buffer prints the rounding message with effective capacity 999,936; positive-buffer construction (already pre-rounded at `train.py:864`) prints nothing. — smoke log has exactly ONE `[ReplayBuffer] capacity 1000000 ... rounded down to 999936` line.
- [x] 6. H7: grep all readers of buffer capacity (`.capacity`, the static `b_cap` arg of `_scan_train_gpu` and its call site, the mixture-sampler window math) and confirm every one reads the post-rounding value; no site caches the raw config number. — the raw config number flows ONLY into `ReplayBuffer.__init__` (train.py:851); `_scan_train_gpu`'s `b_cap`/`pos_cap` come from `buffer.capacity`/`positive_buffer.capacity` at the call site (trainer :853/:841/:846 pre-fix numbering), CPU mixture path reads `buffer.capacity` (:913/:924), train.py:1835 utilization reads `positive_buffer.capacity`. All post-rounding.
- [x] 7. Full test suite: no new failures beyond the 8-test known-red baseline. — 405 passed / 8 failed (exactly the baseline) / 494 skipped; plus one PRE-EXISTING collection error from a parallel work package's uncommitted test (`tests/algorithms/dreamer_srl/test_terminal_step_data_reset.py` imports a not-yet-landed symbol from their WIP `dreamer_srl_main.py`) — not caused by, and not touched by, this change.
- [x] 8. Smoke training run completes with finite losses; speed before/after recorded (same hardware/config/seed, ≥100 post-warm-up iterations). — see Implementation Report.

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-07-06

### What was implemented (file-by-file)

1. **`src/models/dreamer_v3_trainer.py` — `collect_sequence` (H6).** Added the "3b. Sense the ARRIVAL observation" block between the env step and the auto-reset: `obs_arrival = vmap(get_observation)(next_state_raw, params)` (post-step, PRE-reset — death/timeout obs retained), under a `dreamer_sense_arrival` named scope. Transition dict now stores `'obs': obs_arrival`; every other field keeps its index. Pre-step `obs` still feeds `get_action` and was not removed. Docstring updated with the row convention.
2. **`src/models/dreamer_v3_trainer.py` — training scan (H6, comment only).** Convention-pinning comment above `env_inputs = (action, is_first)`: rows are already prev-action-paired, feed UNSHIFTED, do not add a sheeprl-style shift. No functional change; `dreamer_v3_nnx.py` and `train.py` untouched.
3. **`src/models/dreamer_v3_trainer.py` — `ReplayBuffer.__init__` (H7).** Capacity floored to a multiple of `sequence_length` (local rebind, so all array allocations below see the rounded value); `ValueError` if the result is < 1 block; one-line log when rounding changes the value. No YAML touched.
4. **`tests/models/test_dreamer_collect_arrival_alignment.py` (NEW).** Two H6 regression tests per plan §File Changes 4, incl. the non-vacuity guard (arrival ≠ departure at every compared step via the satiation decrement), the exact-replay ground truth (deterministic `jax_step`, replay stops at the first done), index-stability guards (reward/terminal unmoved), and the H5-class no-bleed boundary guard.
5. **`tests/models/test_dreamer_replay_buffer_wrap.py` (NEW).** Four H7 unit tests per plan §File Changes 5, incl. the content-level no-splice-after-wrap assertion (block-identity obs channel must be constant within every sampled window across 50 batches × 32 sequences after multiple wraps).

### Red phase (pre-fix, unmodified source) — all 6 designated tests red for the predicted reason

`pytest tests/models/test_dreamer_collect_arrival_alignment.py tests/models/test_dreamer_replay_buffer_wrap.py -v` → **6 failed** (48.8s):

- `test_stored_obs_is_arrival_of_stored_action` — `AssertionError: step 0: stored obs is NOT the arrival observation of the stored action (H6: it is the pre-step/departure obs …)` — failed at the core assertion; the non-vacuity guard passed first, so the failure is meaningful.
- `test_terminal_row_keeps_arrival_obs_and_no_bleed_into_next_episode` — `AssertionError: terminal row does not store the terminal arrival observation (H6: auto-reset discarded the episode-ending observation)` — assertion (a) as designated; boundary-flag checks (b) passed pre-fix.
- `test_capacity_rounded_down_to_sequence_multiple` — `AssertionError: capacity must be rounded down to a multiple of sequence_length (8); got 100`.
- `test_capacity_below_sequence_length_raises` — `Failed: DID NOT RAISE <class 'ValueError'>`.
- `test_wrapped_buffer_never_splices_two_envs_mid_sequence` — `AssertionError: sampled sequence splices two env blocks mid-window (H7): offending windows (block-id channel rows): [[22,22,22,22,23,23,23,23], [23,23,23,23,24,24,24,24], …]` — the content-level splice, visible exactly as predicted.
- `test_write_index_stays_block_aligned_across_wrap` — `AssertionError: write index 12 is not a multiple of sequence_length 8` (first wrap: 112 % 100 = 12, as derived in §A5).

(Plan's "exactly the five designated tests" is an arithmetic slip — its own list names 2 H6 + 3 primary H7 + 1 supplementary = **6**; all 6 were red.)

### Post-fix results

- New tests: **6/6 pass**.
- Existing dreamer tests (`test_dreamer_continue_truncation.py`): **5/5 pass**.
- Full suite (`pytest tests/ -q --continue-on-collection-errors`): **405 passed, 8 failed, 494 skipped** — the 8 failures are exactly the known-red baseline (4× A1 parity gates `test_unified_parity[observability_gates_S1–S4]`, 3× stale-config FileNotFoundError in `test_inactive_animal_offgrid`/`test_truncation_not_death` (b093023), 1× `test_dreamer_srl_offline_wm_test::test_offline_wm_smoke`). One additional PRE-EXISTING collection error, `tests/algorithms/dreamer_srl/test_terminal_step_data_reset.py` (ImportError on `_reset_terminal_step_data` from a parallel work package's uncommitted WIP in `dreamer_srl_main.py`), is not caused by this change and was left alone; `--continue-on-collection-errors` was used so the rest of the suite still ran. **No new red.**

### Smoke run (Checkpoint 5/8 evidence)

`train.py --config configs/environment/experiment/basic/05-sensory_noise_10x10.yaml --agent_config configs/models/dreamer_v3/dreamer_v3.yaml --num-envs 4 --episodes 30 --no-wandb` on node 102 GPU 1 (RTX 4090): exit 0 in 2m20s. Log (`tmp/20260706_wpD_smoke.log`) contains exactly **one** rounding line — `[ReplayBuffer] capacity 1000000 is not a multiple of sequence_length 128; rounded down to 999936 (H7 fix — keeps sample windows env-aligned after buffer wrap).` — i.e. main buffer rounded, positive buffer (pre-rounded at train.py:864) silent. Collection scan compiled; gradient steps ran; losses finite (`Loss=L: 8.93, Rew=-138.16, R_MAE=8.249, Ent=1.71`); no shape errors.

### Speed check (mandatory — H6 adds one `vmap(get_observation)` per collect step)

Benchmark: `tmp/20260706_190500_wpD_speed_bench.py` — live-size model (`dreamer_v3.yaml` unmodified), live env config, `num_envs=16`, `collect_sequence` of 128 steps, 3 warm-up calls then 100 timed calls, node 102 GPU 1 (RTX 4090), fixed seeds. "Before" was measured by temporarily reverting exactly the two functional H6 lines in the working tree (sensing block + `'obs'` entry), then restoring them (restore verified by re-running all 11 dreamer tests green afterwards).

| Variant | Samples (ms per 128-step collect call) | Mean |
|---|---|---|
| BEFORE (pre-fix) | 120.15, 118.91, 127.53 | **122.2 ms/it** (~16,800 SPS) |
| AFTER (post-fix) | 126.75, 121.74, 126.26, 126.13 | **125.2 ms/it** (~16,360 SPS) |

**Collection-path delta: +3.0 ms per collect call ≈ +2.5%** (run-to-run noise band is ±5 ms, so the delta is within noise; worst-case single-sample pairing +7%). This is the collection call in isolation — the strictest possible view. In a full training iteration (collection + replay_ratio-scaled gradient steps; smoke run: ~4.7 s/it total), the +3 ms collection delta is **≪ 1% of iteration time**. Below the 5% discussion threshold on total training speed; flagged here anyway per protocol so the verifier can judge.

### Comparability caveat (echoed from the plan, as required)

Both fixes change what DreamerV3-NNX trains on (H6 changes the meaning of every buffer row; H7 changes long-run buffer content). **Post-fix runs are not comparable to any pre-fix DreamerV3-NNX run.** Dreamer-vs-rPPO comparisons and modulated-Dreamer conclusions must be re-established on post-fix runs. No checkpoint-format change, but pre-fix checkpoints embed the wrong convention and must not be resumed for result-bearing work.

### Deviations

None functional. Notes: (i) the plan counts "five" designated red tests but its own list enumerates six — all six were red pre-fix and green post-fix; (ii) the full suite required `--continue-on-collection-errors` because a parallel work package's uncommitted test file currently breaks pytest collection (detailed above); (iii) the H6 test fixture uses `configs/environment/experiment/archive/dreamer_curriculum/01_food_only.yaml` as the death-free env, per the plan's "pick/override an env config in the test" latitude.

### Blockers / follow-ups

None for this package. Out-of-scope findings (get_action `is_first=0`, eval symlog skip, checkpoint omissions, unimix, `collect_interval % sequence_length` validation) remain open in [[KNOWN_BUGS]] / [[05_dreamer_v3_nnx]] per the scope fence. Post-verification handoff (bug-curator rows for H6/H7) per the plan's final section.

**Implemented by: developer**

## Verification Report

> **Verified by**: senior-developer
> **Date**: 2026-07-06

**Plain-language verdict:** both fixes are implemented exactly as planned and verified independently. The Dreamer data collector now stores, in each buffer row, the observation an action *produced* (including the death/timeout observation that auto-reset used to discard), so the world model trains under the same (observation, action) pairing it is used with at act time. The replay buffer now rounds its capacity down to a whole number of training sequences (1,000,000 → 999,936 in the live config), so after the ring buffer wraps, sampled sequences can no longer glue two different environments' trajectories together. All 6 new regression tests pass, the full suite shows zero new failures beyond the known-red baseline, and the speed cost is negligible.

### File-by-file

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/models/dreamer_v3_trainer.py` — `collect_sequence` | H6: arrival-obs sensing block (3b) + `'obs': obs_arrival` + docstring convention | ✅ | Diff is surgical: 50 insertions / 1 deletion, only 2 functional lines. Sensing reads `next_state_raw` (post-step, PRE-reset — Checkpoint 2 satisfied, NOT `final_env_state`); sits between step and auto-reset; pre-step `obs` still feeds `get_action` (line 647→652). Only the `'obs'` entry of the transition dict changed — `action/reward/terminal/is_first/termination_reason` and all info arrays keep their index (Checkpoint 3). |
| `src/models/dreamer_v3_trainer.py` — training scan ~:211 | H6: convention-pinning comment only | ✅ | `env_inputs = (action, is_first)` UNSHIFTED, no functional change — matches §A4's "no shift needed" derivation. |
| `src/models/dreamer_v3_trainer.py` — `ReplayBuffer.__init__` | H7: floor to sequence multiple, raise below one block, log on change | ✅ | Local rebind `capacity = rounded_capacity` precedes ALL array allocations; `self.capacity` post-rounding. Checkpoint 6 re-verified by grep: the raw config number enters only at `train.py:851`; every consumer (`b_cap`/`pos_cap` at trainer :868–881, :940, :951; CPU mixture path; `train.py:1835` utilization) reads `buffer.capacity`/`positive_buffer.capacity` post-rounding. |
| `tests/models/test_dreamer_collect_arrival_alignment.py` (NEW) | 2 H6 regression tests | ✅ | Matches plan §File Changes 4 point-for-point: exact deterministic replay (PRNG inside `EnvState.key`), per-step non-vacuity guard checked BEFORE the core assertion, index-stability guards, boundary-flag + H5-class no-bleed guards. Non-vacuity guard judged sound (see below). |
| `tests/models/test_dreamer_replay_buffer_wrap.py` (NEW) | 4 H7 unit tests | ✅ | Pure-unit, real `add_batch`/`sample` API, env-major 16-item writes mimicking live shape; content-level splice test + arithmetic pin. |
| `src/models/dreamer_v3_nnx.py`, `configs/` | — | ✅ | Untouched, as required. |
| `train.py` | — | ✅ | Carries only the parallel H10/BM package's hunks (`bm_drive_batch`, :65/:1635–1745); the Dreamer buffer flatten (:1585–1600) and both `ReplayBuffer` construction sites are untouched by any in-flight change. Not counted against this package. |

### Independent checks performed (beyond diff review)

1. **Convention re-derived against the vendored sheeprl checkout.** Read `vendor/sheeprl/.../dreamer_v3.py:82-104` (storage diagram + zero-prepend train-time shift), `:586-587`/`:628-656` (arrival-written rewards/dones; `reset_data` terminal-obs row), `agent.py:396-435` (`RSSM.dynamic`). Confirmed: our post-fix row *t* = (arrival obs `õ_t`, action `a_t` that produced it) fed **unshifted** is index-for-index identical to sheeprl's post-shift pairing (`embed(o_i)` with `a_{i-1}`) at every non-boundary index, and matches paper Eq. 3. Reward/continue targets consumed unshifted are arrival-convention in both stacks.
2. **is_first-row action semantics.** sheeprl's RSSM consumes a **zero** action on is_first rows — doubly: the `reset_data` row stores a zero action *and* `dynamic` masks it (`agent.py:425` `action = (1 - is_first) * action`). Ours consumes the new episode's **real first action**, with `deter`/`stoch` reset by the mask (`dreamer_v3_nnx.py:117-119`; the action is not masked). Verified causally sound and self-consistent: that action was chosen from the reset observation *before* `õ_t` existed (no future leak — legitimate conditioning, strictly more informative than sheeprl's zero), and it keeps train ≡ inference (`get_action` never zeroes `prev_action`). Accepted-residual documentation in §A4 is accurate for our code. ⚠️ **Plan erratum (doc-only, no code impact):** §A1's clause "the action input is *not* masked — sheeprl relies on the stored zero action" is wrong about sheeprl — `agent.py:425` does mask it. §A4's rejection of zeroing inside our `RSSM.step` still stands on its independent grounds (would discard valid conditioning + desync from `get_action`), so no change requested.
3. **H7 red-phase reproduced independently** (no working-tree modification): extracted `HEAD`'s pre-fix `ReplayBuffer` to `tmp/20260706_prefix_dreamer_v3_trainer.py` and ran it — capacity(100, seq 8) stays 100; capacity=5 constructs without raising; after 7×16-item writes the write index is **12** (112 % 100), off the 8-aligned sampling grid — exactly the §A5 prediction and the implementer's reported failure messages.
4. **H6 red-phase verified by construction:** the non-vacuity guard (arrival ≠ departure obs at every compared step, via the per-step satiation decrement) is asserted *before* the core assertion and passed in my green run — which logically entails the pre-fix code (which stored the departure obs) must fail the core assertion. Implementer's pasted pre-fix messages match the tests' exact assertion strings. Guard judged non-vacuous and well-placed.
5. **Tests re-run:** `pytest tests/models/ -v` (incl. both new files) → **28/28 pass**, including all 5 existing `test_dreamer_continue_truncation.py` tests. Full suite (`pytest tests/ -q --continue-on-collection-errors`) → **407 passed, 8 failed, 494 skipped**: the 8 reds are exactly the known baseline (4× A1 parity gates, 3× stale-config `b093023`, 1× dreamer_srl offline-WM smoke). **Zero new failures.** (407 vs the implementer's 405: the parallel dreamer_srl package's collection error has since resolved — its 2 tests now collect and pass; unrelated to this package.)
6. **Smoke evidence checked:** `tmp/20260706_wpD_smoke.log` contains exactly **one** `[ReplayBuffer] capacity 1000000 … rounded down to 999936` line (main buffer rounded, positive buffer silent, per Checkpoint 5); 30 episodes completed, losses finite.
7. **Interaction check (§A6):** continue-head fix `5b093bf` — `compute_continue_target(term_reason)` at trainer :256-258 untouched, `term_reason` index unchanged; the death/timeout target now pairs with a feature encoding the actual terminal observation (strengthened, as claimed). Two-hot layout `f5df600` — `to_twohot` call site :246 untouched, orthogonal. No conflict.

### Speed verdict

✅ **No regression.** Isolated collect-call delta +3.0 ms (+2.5%), inside the reported ±5 ms run-to-run noise band; same hardware (node 102 RTX 4090), same config/seeds, 100 timed calls after warm-up — methodology sound (before-measurement via temporary revert of exactly the 2 functional lines, restore verified by re-running all 11 dreamer tests). Against the plan's threshold the relevant denominator is total training iteration time (~4.7 s/it in the smoke run): the delta is ≈0.06%, far below the 5% discussion threshold the plan pre-accepted for this path.

**Conclusion**: ✅ **PASS.** Implementation matches the plan exactly (2 functional changes + comments in one file, 2 new test files, nothing else); all 8 checkpoints hold under independent re-verification; one doc-only erratum noted in the plan's §A1 sheeprl citation (no code impact). Ready to commit. Post-verification handoff per plan: bug-curator to close H6/H7 rows; pre-fix DreamerV3-NNX runs are superseded (comparability caveat stands).

**Verified by: senior-developer**

---

**Post-verification handoff:** ask `bug-curator` to update the H6 and H7 rows in [[KNOWN_BUGS]] (fixed, link commit + this plan). Notify `experiment-designer` / the user that any queued result-bearing DreamerV3-NNX launches should wait for this fix and that pre-fix Dreamer baselines are superseded.
