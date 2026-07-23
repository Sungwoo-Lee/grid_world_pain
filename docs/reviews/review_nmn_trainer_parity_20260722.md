---
title: "NMN/FiLM rPPO variant — inheritance of trainer/env truncation & bootstrap fixes"
topic: models
status: active
reviewer: code-reviewer
created: 2026-07-22
last_updated: 2026-07-23
audited_doc: "working tree on branch v3.0 (HEAD d0585a6): src/models/recurrent_ppo_trainer.py, recurrent_ppo_network.py, neuromodulator.py, modulated_gru_cell.py, train.py (rPPO branch), src/environment/core.py"
---

# Review: does the neuromodulated (NMN/FiLM) RecurrentPPO variant correctly inherit the recent trainer/environment fixes?

## Verdict (plain language)

**CONFIRMED — the modulated and unmodulated agents share one code path for everything the
three recent fixes touched, so the neuromodulation variant inherits all three fixes with no
silent divergence.** The three fixes are: (1) when an episode ends because time ran out
(rather than the agent dying), the learning target keeps its estimate of future reward
instead of wrongly zeroing it; (2) the death-penalty reward fires only on real death, not on
surviving to the step limit; and (3) Monte-Carlo return targets at the edge of each 128-step
training window are seeded with the critic's estimate of what comes next, instead of
pretending the future is worth zero. All three live in code that never branches on whether
the neuromodulator is present: the return/advantage math consumes only reward, value, and
episode-boundary arrays, and every network forward pass — including the two "bootstrap"
value estimates the fixes introduced — goes through the model's single `__call__`, which
internally handles the modulated hidden state (a `(task_h, mod_h)` pair) as an opaque
pytree. I traced every place a plain hidden state could have been fabricated where a pair
is needed, and found none.

Two pre-existing issues were found along the way, **neither introduced by the fixes and
neither modulation-specific**: a random-key aliasing bug in the rollout loop (the key used
for episode-reset placement is byte-identical to the next step's master key — a convention
violation with possible statistical correlation, though reproducibility still holds), and a
latent inconsistency in the GAE branch that would mis-handle the known "overeating sets a
death code without ending the episode" env quirk — currently unreachable because every live
config uses MC returns and no config enables overeating death.

**Bottom line: no fix is required before continuing NMN training runs.** The two flagged
concerns should be queued as backlog items (details below).

## Scope and method

Read-only review of the current v3.0 working tree. I traced the full update cycle
(collection scan → return/advantage computation → PPO loss re-unroll → gradient step) in
`src/models/recurrent_ppo_trainer.py`, the shared network `src/models/recurrent_ppo_network.py`,
the side-network `src/models/neuromodulator.py`, the gate-bias cell
`src/models/modulated_gru_cell.py`, the rPPO branch of `train.py`, and the env-side fix in
`src/environment/core.py`. Two claims were verified empirically in the project conda env:
JAX key-split prefix aliasing (`split(k,2) == split(k,N)[:2]` → `True`) and
`'modulator' in <nnx.State>` membership (works; the grad-norm probe is live, not dead code).
Intended fix semantics cross-checked against
[[fix_plan_h4_mc_window_bootstrap]] and [[FIX_TRUNCATION_TREATED_AS_DEATH]] and the tests
`tests/models/test_gae_truncation.py`, `tests/models/test_mc_window_bootstrap.py`.

## Verdict per numbered check

| # | Check | Verdict |
|---|---|---|
| 1 | Update-step parity (rollout → returns → loss → grad) | **CONFIRMED CORRECT** |
| 2 | Hidden-state reset of `(task_h, mod_h)` at episode boundaries and BPTT window starts | **CONFIRMED CORRECT** |
| 3 | Values-next / bootstrap forward threads the true pre-reset `(task_h, mod_h)` | **CONFIRMED CORRECT** |
| 4 | BPTT re-unroll parity incl. temperature in both log-prob computations | **CONFIRMED CORRECT** |
| 5 | Edge-gate logic in `compute_mc_returns` / `compute_gae` | **CONFIRMED CORRECT (MC)**; **RISK 🟡 (GAE, latent, unreachable in live configs)** |
| 6 | JAX recompilation / PRNG / vmap-scan axes | **CONFIRMED CORRECT** except **RISK 🟡 pre-existing PRNG key aliasing (modulation-independent)** |
| 7 | `mod_info` buffer isolation and None-vs-struct pytree safety | **CONFIRMED CORRECT** |

## Findings table

| Severity | Location | Issue | Suggested fix |
|---|---|---|---|
| 🟡 concern (pre-existing, both variants) | `src/models/recurrent_ppo_trainer.py:209` + `:244` | PRNG key aliasing: `key, act_key = split(key)` (209) advances the carry, but `reset_key, _ = split(key)` (244) re-splits the *already-carried* key without advancing it. Since `jax.random.split(k, 2) == jax.random.split(k, N)[:2]` (verified empirically), step *t*'s env-0 reset key is byte-identical to step *t+1*'s master key, and env-1's reset key equals step *t+1*'s `act_key`. Reset-placement randomness is therefore key-aliased with the subsequent action/master stream. Reproducibility (same init key + actions → same episode) still holds; this is a statistical-correlation / convention violation ("never reuse a sub-key"), identical in modulated and unmodulated runs. | Change line 244 to `key, reset_key = jax.random.split(key)` and carry the advanced key. Note: changes the sampled stream → breaks bit-exact replay of existing runs; land in a designated reproducibility-break window and note it in the run-comparison docs. |
| 🟡 concern (latent, currently unreachable) | `src/models/recurrent_ppo_trainer.py:78` (with mask built at `:374`) | `compute_gae` zeroes the value bootstrap on `terminated` **without** AND-ing with `done`. `terminateds` comes from `termination_reason >= 2`, and the known env quirk (`overeating_death=True` sets reason=3 without setting `done` — `src/environment/core.py:707-708` vs `update_body`'s done at `core.py:117-126`, which omits the overeating condition) means a *continuing* episode step could get its bootstrap zeroed mid-episode while the GAE accumulation chain keeps running — an incoherent hybrid. `compute_mc_returns` explicitly guards this at the window edge (`edge_death = done AND terminated`, trainer:114) but the GAE path has no equivalent guard. Unreachable today: every live rPPO config sets `return_mode: "MC"` and no config in `configs/` sets `overeating_death: true`. | Gate the delta term on `terminated AND done` (mirror of trainer:114), or fix the upstream env quirk so reason=3 implies done. Record in KNOWN_BUGS as a sibling of the overeating quirk. |
| 🟢 nit | `src/environment/core.py:706` | `reason = where(new_nutrition <= 0.0, 2, reason)` is applied unconditionally, even when `params.with_nutrition` is False (in which case `update_body` never decays nutrition and never sets done from it). A no-nutrition config whose initial nutrition were ≤ 0 would mislabel every step as terminated=starvation in the trainer's `terminateds` mask. Pathological config space only. | Guard with `if params.with_nutrition:` for symmetry with `update_body`. |
| 🟢 nit | `src/models/recurrent_ppo_network.py:327` | LSTM + modulation silently drops Injection B: `nnx.LSTMCell` is called without `gate_bias`, so the modulator's `z_memory` head (and its params) are dead weight under `rnn_type: "LSTM"`. Not active — both live modulated configs pin `rnn_type: "GRU"` (e.g. the FiLM config comments "required for gate-bias injection"). | Either raise at construction when `modulation_enabled and rnn_type == "LSTM"`, or document the intentional A+C-only LSTM mode. |
| 🟢 nit | `src/models/recurrent_ppo_trainer.py:326-335` | The modulator grad-norm probe (`'modulator' in grads`) works in the current flax version (verified), but the bare `except: pass` means any future flax `State` API change would silently freeze `modulator/grad_norm` at 0.0 — a monitoring blind spot, not a training bug. | Narrow the except to `(KeyError, TypeError)` and/or log a one-time warning. |

## Detailed trace per check

### 1. Update-step parity — CONFIRMED CORRECT

There is exactly one trainer. The only modulation-aware trainer/train-loop code is:
`mod_info` field in the rollout `Transition` (`recurrent_ppo_trainer.py:35`, populated
at `:282`); the grad-norm probe (`:326-335`); and pure logging in `train.py` (`:1318-1326`
debug print, `:1477-1496` WandB `modulator/*` metrics, `:1512-1513` progress-bar
temperature). None of these touch rewards, dones, advantages, targets, loss masking, or
the gradient path. Everything the three fixes changed is consumed identically:

- `compute_gae` (`trainer:55-88`) and `compute_mc_returns` (`trainer:90-131`) take only
  `rewards / values / values_next / dones / terminateds / bootstrap_value` arrays; the
  `terminateds` mask is built identically for both branches from
  `step_info.termination_reason >= 2` (`trainer:361`, `:374`).
- Death-penalty gating is entirely env-side (`core.py:696`, `:731-737`) — upstream of any
  agent code.
- Modulator parameters are ordinary `nnx.Param`s inside the model
  (`neuromodulator.py:97-127`), so `nnx.Optimizer(model, ..., wrt=nnx.Param)`
  (`train.py:825-832`) optimizes them; `nnx.value_and_grad` over the whole model
  (`trainer:320`) differentiates through the FiLM/γβ path, the GRU gate bias, and the
  temperature division. Intentional differences only.

### 2. Hidden-state reset — CONFIRMED CORRECT

`_h_reset_on_done` (`trainer:137-145`) is a `jax.tree_util.tree_map` over the *whole*
hidden-state pytree — for the modulated model that is `(task_h, mod_h)` (task_h itself a
2-tuple for LSTM), so every leaf, including the modulator GRU state, is zeroed on done.
Applied at both required sites:

- **Collection**, post-step: `final_h = _h_reset_on_done(h_new, done)` (`trainer:258`),
  where `h_new = (task_h_new, mod_h_new)` from the model call (`recurrent_ppo_network.py:342`).
- **BPTT re-unroll inside the loss**: `h_reset = _h_reset_on_done(h_new, done)`
  (`trainer:164`) with identical post-step placement — train/inference parity holds.

Window-start state: the collection scan emits the *pre-forward carry* per step
(`(trans, h_state)`, `trainer:286`), and `h_init = _h_get_first_timestep(h_states)`
(`trainer:387`, helper `:147-149`) is again a generic tree_map, so the tuple structure
survives into `PPOBatch.h_init`; the per-env vmap uses `in_axes` `h_init=0`
(`trainer:315`), which broadcasts the axis over all tuple leaves.

Zeros-consistency: `NeuromodulatorRNN.initial_state` returns zeros
(`neuromodulator.py:182-186`), and task states are zeros
(`recurrent_ppo_network.py:367-384`) — so a zeros-reset equals a fresh init; no divergence.
The stage-transition reset (`train.py:1263-1265`), checkpoint-resume env rebuild
(`train.py:1137-1138`), and restored `h_state` from checkpoints (saved with full tuple
structure, `train.py:2087-2095`, restored `:1102-1110`) all go through
`model.initial_state(num_envs)` / the saved pytree, which include `mod_h`.

### 3. Bootstrap value with modulation — CONFIRMED CORRECT

- **GAE per-step `next_value`** (`trainer:234-238`): forward on the observation of the
  TRUE pre-auto-reset `next_state`, with `h_new` — which for the modulated model is the
  full `(task_h_new, mod_h_new)` pair belonging to the pre-reset trajectory. The vmap uses
  `h_axes = _h_vmap_axes(last_h_state)` (`trainer:194`, helper `:133-135`), a pytree of 0s
  matching whatever structure the model uses — no plain-array assumption anywhere.
- **MC window-edge bootstrap** (`trainer:298-304`): the scan carry exports
  `(next_state, h_new)` of the last step as `(boot_state, boot_h)` (`trainer:286`,
  `:288-291`) — both pre-auto-reset, `boot_h` deliberately *not* done-reset (the reset copy
  `final_h` is a separate carry slot). Correct: on an edge real-death the un-reset `boot_h`
  value is discarded anyway by the edge gate; on timeout/mid-cut it is exactly the state
  that would carry into the continuation.
- The only fabricated zeros object is `next_value = jnp.zeros_like(value)` in MC mode
  (`trainer:240`) — a *value array*, not a hidden state; no shape/semantics hazard.
- The symlog input transform lives inside `model.__call__`
  (`recurrent_ppo_network.py:308`), so bootstrap forwards and collection forwards see
  identical preprocessing.

### 4. BPTT re-unroll / temperature parity — CONFIRMED CORRECT

Both log-prob computations go through the same `model.__call__`, which applies
`logits = logits / mod_output.temperature` (`recurrent_ppo_network.py:337`) *before*
returning: collection via `get_action_and_value_nnx` (`recurrent_ppo_network.py:388-399`,
`log_softmax(logits)[action]`), update via `ppo_loss_fn`'s scan (`trainer:156-159`). The
importance ratio (`trainer:173`) is therefore temperature-consistent — old and new
log-probs both include the (re-unrolled, current-params) temperature. The modulator is
re-unrolled from `h_init` inside every epoch's loss with the same post-step reset
semantics as collection (check 2), giving proper BPTT gradient flow into the modulator
through all three injections (FiLM γ/β in the encoder, `gate_bias=z_memory` in
`ModulatedGRUCell` — `recurrent_ppo_network.py:330`, `modulated_gru_cell.py:67-70` — and
temperature). Eval-mode argmax (`recurrent_ppo_network.py:393`) is invariant to the
positive scalar temperature, as intended.

### 5. Edge-gate logic — CONFIRMED CORRECT (MC); latent 🟡 in GAE

`compute_mc_returns` (`trainer:114-130`), reverse scan with carry initialised to the
bootstrap, checked case by case at the edge step t = T−1:

- **Edge real death** (`done=1, terminated=1`): `edge_death=1` → `dones_for_reset[-1]=1`
  → carry zeroed before adding `r_{T-1}` → return = reward only. Correct.
- **Edge timeout** (`done=1, terminated=0`): `edge_death=0` → gate cleared → return =
  `r + γ·V(s')` where V(s') is the pre-reset timed-out state's value. Correct truncation
  semantics (matches 3c60f6f).
- **Mid-episode window cut** (`done=0`): gate untouched → bootstrap flows. Correct.
- **Overeating quirk at the edge** (`terminated=1, done=0` — episode actually continues):
  the `AND done` guard keeps the bootstrap. Correct, and explicitly commented
  (`trainer:111-113`).
- **Mid-window boundaries**: only index −1 is rewritten, so mid-window resets on merged
  `done` are unchanged (`trainer:121`); mid-window timeouts are deliberately finite-horizon
  (documented `trainer:100-102`) — a design choice, not a regression.

All seven cases in `tests/models/test_mc_window_bootstrap.py` cover exactly these
branches, including the overeating quirk and old-behaviour reproduction.

`compute_gae` (`trainer:76-88`): `delta = r + γ·V(s')·(1−terminated) − V(s)` with
accumulation reset on `(1−done)` — correct for the three main cases and covered by
`tests/models/test_gae_truncation.py`. The gap: `terminated` is not AND-ed with `done`,
so the overeating quirk would zero a *mid-episode* bootstrap while the accumulation chain
continues (findings table, row 2). Latent — MC everywhere, `overeating_death` nowhere.

### 6. JAX-specific — CONFIRMED CORRECT (one pre-existing 🟡)

- **Recompilation**: `modulation_enabled` is a plain Python attribute baked into the nnx
  graphdef; `return_mode`/`rnn_type` arrive via the static config arg
  (`nnx.jit(train_iteration, static_argnums=(6,))`, `train.py:854`; trace-time comparison
  documented at `trainer:195-198`). One trace per model configuration; no traced-value
  Python branching found in the hot path.
- **vmap/scan axes**: `mod_h` is threaded through `jax.lax.scan` as an ordinary carry leaf
  (`trainer:286-291`); batch vmaps derive axes from the actual pytree
  (`_h_vmap_axes`) rather than assuming an array. Correct.
- **PRNG**: per-env action keys are properly per-env (`trainer:210`); the modulator
  consumes no keys (deterministic GRU), so modulation adds no PRNG surface. The
  reset-key/master-key aliasing (findings table, row 1) is real, empirically confirmed,
  pre-existing, and bit-identical across both variants — it cannot cause
  modulated-vs-unmodulated divergence, and same-key + same-actions replay determinism
  still holds.

### 7. `mod_info` buffer isolation — CONFIRMED CORRECT

`PPOBatch` (`trainer:45-53`) carries no `mod_info`; the loss recomputes everything from
`obs/actions/dones/h_init`, so stored modulator outputs cannot leak into the loss.
For the unmodulated model `mod_info` is `None` — a registered empty pytree node — so
`Transition(..., mod_info=None)` scans/stacks cleanly and `train.py:1318` guards on
`mod_info is not None`. The structure difference between modulated (ModulatorOutput of
arrays) and unmodulated (None) runs is fixed at trace time per run — it forces a separate
compilation per model configuration (expected) but can never mix structures within a run.

## Conclusion

The NMN/FiLM variant inherits all three fixes through the shared code path — no silent
divergence found; two pre-existing, modulation-independent concerns flagged for backlog
(PRNG reset-key aliasing; GAE overeating-quirk gate).

Reviewed by: code-reviewer

---

## Implementation Report (2026-07-23)

All five findings-table fixes were implemented on branch v3.0, user-approved including the
reproducibility-breaking PRNG one. Plain-language summary: the rollout loop no longer reuses
a random sub-key (which had correlated episode-reset placement with the next step's random
stream), the GAE advantage path now uses the same "real death = episode ended AND the reason
was death" gate the MC path already had, a mislabeled-starvation edge case is guarded, an
unsupported LSTM+modulator combination now fails loudly at construction instead of silently
disabling one of the modulator's three injection paths, and a monitoring probe no longer
swallows every exception.

### File-by-file

| Fix | File | Change | Commit |
|---|---|---|---|
| 1 (PRNG aliasing) | `src/models/recurrent_ppo_trainer.py` (rollout scan, auto-reset key) | `reset_key, _ = split(key)` → `key, reset_key = split(key)`; advanced key flows into the scan carry (verified: the carry returns the local `key`). **Reproducibility break noted in commit body.** | `b8eb286` |
| 2 (GAE quirk gate) | `src/models/recurrent_ppo_trainer.py` (`compute_gae`) | `real_death = done * terminated` gates the delta bootstrap; accumulation reset stays on `done`; docstring updated. Numerically a no-op for all current configs (`overeating_death=false` everywhere → terminated implies done). | `218a366` |
| 3a (starvation label) | `src/environment/core.py:706` | Starvation `reason=2` stamp wrapped in `if params.with_nutrition:` — verified static (`struct.field(pytree_node=False)`, `state.py:242`), so the Python branch is trace-safe, same style as the adjacent `overeating_death` guard. | `8334d89` |
| 3b (LSTM+modulation) | `src/models/recurrent_ppo_network.py` (`__init__`) | `ValueError` at construction for `modulation_enabled and rnn_type == "LSTM"`, next to the FiLMNoNorm check. | `8334d89` |
| 3c (bare except) | `src/models/recurrent_ppo_trainer.py` (grad-norm probe) | `except:` → `except (KeyError, TypeError):`; no callbacks/logging added in the jitted path. | `8334d89` |

### Tests

- **New:** `tests/models/test_gae_truncation.py::test_bootstrap_retained_on_overeating_quirk_mid_episode`
  (quirk case: `terminated=1, done=0` mid-window must retain the bootstrap). Bug-fix discipline
  followed: **failed pre-fix** (bootstrap wrongly zeroed), **passes post-fix**.
- **New:** `tests/models/test_network_construction.py` (3 tests) — LSTM+modulation must raise
  (**DID NOT RAISE pre-fix**, passes post-fix), plus GRU+modulation and unmodulated-LSTM controls.
- **Baseline:** targeted suites (`test_gae_truncation`, `test_mc_window_bootstrap`,
  `test_truncation_not_death`) — 19 passed after repairing a stale config path (below).
- **Full `tests/models/` + `tests/env/`:** 220 passed, 508 skipped, 4 failed —
  **all 4 failures pre-existing and unrelated**: `test_unified_parity.py::test_parity[...observability_gates_S1–S4]`
  fail identically on a clean pre-fix worktree at d0585a6 (verified) — stale recorded fixtures
  (agent_pos mismatch at step 0), needs separate triage. A 5th failure
  (`test_inactive_animal_offgrid.py`) was a stale config path and is fixed (below).

### Speed check

Same box (local 4090, cuda:1), same config (`basic/03-random_init_10x10` env +
`recurrent_ppo/recurrent_ppo.yaml` agent, GRU/MC, config seed 42), same budget
(`--episodes 0 --total-timesteps 2000000 --no-wandb --quiet`), sequential runs:

- **Before** (worktree at d0585a6): 134.90 s wall → **14,826 SPS** (end-to-end incl. compile)
- **After** (HEAD): 130.77 s wall → **15,294 SPS** (+3.2%, within run-to-run noise)

No regression. Expected — op count is unchanged (Fix 1 rearranges which split output feeds
the carry; Fix 2 adds one multiply in a once-per-iteration scan not exercised in MC mode).

### Deviations from plan (all flagged, none silent)

1. **Two extra test-only commits** repairing stale paths left by the basic-ladder re-level
   (b093023), which had broken 3 `tests/env/` tests at baseline with `FileNotFoundError`:
   `ee8b981` (truncation test → `basic/05-sensory_noise`), `79775e5` (offgrid test →
   `basic_curriculum/04-far_sight_predator`). Path + docstring only; test logic untouched.
   The re-level commit's "no live config references a moved path" claim missed these two tests.
2. The 4 observability-gates parity failures are **left unfixed** — pre-existing, fixture
   regeneration is a behavioral judgment outside this plan; recommend `senior-developer` /
   `bug-curator` triage.

### Commits (in order, branch v3.0, not pushed)

1. `ee8b981` test(env): point truncation test at renamed basic/05 sensory-noise config
2. `218a366` fix(trainer): gate GAE death bootstrap on done AND terminated (overeating-quirk guard)
3. `b8eb286` fix(trainer): advance carried PRNG key when drawing the auto-reset key **[reproducibility break]**
4. `8334d89` fix(models): three review nits — starvation label guard, LSTM+modulation guard, narrowed except
5. `79775e5` test(env): point offgrid-parking test at basic_curriculum far-sight config

Live runs are unaffected (they run from already-loaded code); no configs or launch scripts touched.

Implemented by: developer
