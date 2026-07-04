---
title: "Independent diagnosis — Recurrent-PPO training stack (trainer / network / modulator / train.py loop)"
topic: issues
status: active
created: 2026-07-04
last_updated: 2026-07-04
---

# Independent diagnosis: the Recurrent-PPO training stack

## Purpose (plain-language entry point)

This is an independent bug hunt over the code that trains the project's main agent — the
recurrent PPO learner: how it collects experience from the parallel environments, how it
scores that experience (returns, advantages), how the recurrent network and its
neuromodulator ("FiLM"/gate-bias/temperature) process observations, and how the training
loop in `train.py` stitches it together. A prior audit (2026-07-04) already found and fixed
a cluster of bugs here; this pass **verifies those fixes** and hunts for anything they missed.

Two findings matter. **First (new, live path):** every live training run scores its
experience with within-window "Monte Carlo" returns that are cut off at the end of each
128-step collection window with no estimate of the future beyond it — so steps near the
window's end systematically look better/worse than they are, biasing what the critic and
policy learn on **every** live run. The recently landed "keep the future-value estimate on
timeout" trainer fix only repaired the *other* (GAE) scoring mode, which **no live config
uses** — the bug registry row can mislead readers on this. **Second (new, live path):** the
experience collector reuses random-number keys, so the randomness that places entities in a
freshly reset environment is byte-identical to the randomness that samples actions two steps
later — a real (if statistically subtle) correlation. Everything else examined — including
all four landed fixes in scope — checks out. Details, evidence, and severity below.

**Scope reviewed:** `src/models/recurrent_ppo_trainer.py`, `recurrent_ppo_network.py`,
`neuromodulator.py`, `modulated_gru_cell.py`, `modulated_layer_norm_gru_cell.py`,
`ppo_trainer.py`, `ppo_network.py`, and the rPPO rollout/logging loop in `train.py`
(~lines 758–825, 1328–1483). Read against `KNOWN_BUGS.md` so known rows are not re-reported.

---

## Verification of previously landed fixes (asked-for check)

| Fix | Commit | Verdict |
|---|---|---|
| rPPO GAE truncation bootstrap (`compute_gae` terminated-mask + pre-reset `next_value`) | `3c60f6f` | ✅ **Correct and complete for its (GAE) scope** — but see Finding 2: zero effect on live runs, all of which use MC mode |
| Plain-PPO timeout bootstrap + per-step pre-reset `next_value` | `926c2c3` | ✅ Correct (`ppo_trainer.py:150-161`, `226-235`) |
| Plain-PPO advantage baseline (V(s_t), not neighbour-shifted) | `8c1ad2f` | ✅ Correct (`ppo_trainer.py:66-68`: `delta = r + γ·V'·(1-term) − V(s_t)`; `targets = A + V`) |
| Env-side timeout reward (no −100 on survival) | `ef0fd25` | (env file out of scope) Trainer-side interplay verified: MC returns no longer ingest a −100 at timeout — this env fix is the **only** part of the truncation cluster that live rPPO runs actually feel |
| Temperature-clip floor effectively 0.5 (cosmetic) | `5caa0df` | ✅ Comment present at `neuromodulator.py:169-173`; behaviour unchanged, known |
| FiLM shared gain across senses | — | Confirmed **not re-reported** (registry: intended) |

Detail on `3c60f6f` correctness: the delta bootstrap `γ·V(s′)·(1−terminated)` is gated on
real death only (`termination_reason ≥ 2`), the accumulation reset stays gated on merged
`done` (`recurrent_ppo_trainer.py:76-88`, `330-338`), and `next_value` is computed **before**
auto-reset from the true next observation with the un-reset hidden state `h_new`
(`recurrent_ppo_trainer.py:206-212`) — exactly right for a continuation value. The MC/GAE
gating is a static, trace-time branch as documented.

---

## Finding 1 — MC-mode (the live path) truncates every value/advantage target at the rollout-window boundary with no bootstrap

**Severity:** High (systematic bias in the learning signal of every live rPPO run; longstanding, shared by all historical runs, so run-to-run comparability is preserved)
**Where:** `src/models/recurrent_ppo_trainer.py:90-106` (`compute_mc_returns`, reverse scan with initial carry `0.0`), consumed at `:315-324`
**Status: NEW** (never documented as a correctness caveat; the 2026-07-04 truncation cluster fixed the GAE path only)

**What happens.** Every live rPPO config sets `return_mode: "MC"` (grep: all 13
`configs/models/recurrent_ppo/*.yaml`) and `sequence_length: 128`, while episodes run up to
`max_steps: 500`. `compute_mc_returns` is applied per 128-step collection window with the
reverse-scan carry initialised to `0.0` and **no value bootstrap at the window edge**. So the
"return" credited to a step only sums rewards up to the end of the current window: a step at
window position 0 integrates up to 128 steps of future reward, a step at position 120
integrates at most 8. The critic is trained on these window-truncated targets
(`targets = returns`), and advantages (`returns − V`) inherit the same position-dependent bias.
Because window position is not part of the state, the critic cannot explain the bias away —
it lands directly in the advantages the policy ascends.

**Concrete failure scenario.** In the survival task per-step rewards are predominantly
negative (homeostatic drives); a step 5 steps before the window edge gets a return that is
missing ~4 windows of future negative reward, so it looks systematically "better" than an
identical state sampled early in a window. Every iteration, ~the last quarter of each window
contributes distorted advantages. This is the same *family* of defect as the fixed Finding B
Part 2 (future value thrown away at an artificial boundary) — but at **every** rollout
boundary, not just episode timeouts, and on the **live** path.

**Relation to the landed fix (flagged prominently).** The registry row "Value estimate
dropped on timeout — FIXED (`3c60f6f`, rPPO trainer)" is accurate for the GAE branch but can
mislead: no live config exercises that branch, so on live runs the trainer-side behaviour is
unchanged — MC mode has *no* bootstrap mechanism at all, at timeouts **or** window edges. The
fix's own implementation report says this plainly ("Part 2 leaves the 6 live runs
byte-identical"); the registry row does not. Recommend: (a) a KNOWN_BUGS clarification, and
(b) a decision — either accept window-truncated MC as a deliberate finite-horizon design
(documented), or move the live configs to the now-correct GAE path / add a `V(s_window_end)`
bootstrap to the MC carry (one-line change: initialise the reverse-scan carry with the
already-available `next_value[-1]` instead of `0.0`, masked by `done[-1]`).

**Sibling nit (Low, same branch):** MC targets are z-normalized per iteration
(`returns = (returns − mean)/std`), so the critic's target scale is nonstationary and
`advantages = normalized_returns − raw_V` mixes scales in early training. Long-standing
"PyTorch parity" design; noted, not counted as a bug.

**Sibling note (Low):** MC treating a *timeout* as end-of-return is defensible under the
project's finite-horizon survival objective (episode really does end at 500), even though it
contradicts the infinite-horizon philosophy the GAE fix adopted. Flagged as a philosophical
inconsistency, not a defect.

## Finding 2 — PRNG key reuse in the rollout collector: env-reset keys collide with the future action-sampling key chain

**Severity:** Medium (live path, every step of every rPPO run; statistical correlation, not a determinism break)
**Where:** `src/models/recurrent_ppo_trainer.py:216` (`reset_key, _ = jax.random.split(key)` — the carried key is **not** advanced)
**Status: NEW** (long-standing — predates the v3.0 audit; the prior JAX review covered env/eval PRNG, not this trainer loop). The plain-PPO sibling does it correctly (`ppo_trainer.py:163`: `reset_key, key = jax.random.split(key)`).

**What happens.** In `scan_fn` the collector derives `reset_key = split(key)[0]` but returns
the *un-advanced* `key` as the carry. The next step then computes its new main key as
`split(key)[0]` — the **same value**. Combined with JAX's split-prefix property
(`split(k, 2)[i] == split(k, N)[i]`, verified `True` under the project's
`threefry_partitionable` config), this produces exact collisions, reproduced empirically
with the code's own pattern (interpreter session, 2026-07-04):

- `reset_key` at step *t* **==** the carried main key at step *t+1*;
- env 0's per-env reset key at step *t* **==** `reset_key` at step *t+1* (each step's reset-key set is nested inside the previous one);
- env 1's per-env reset key at step *t* **==** `act_key` at step *t+2*; and `jax_reset`'s five internal sub-keys (agent placement / entity placement / body / property) for that reset are **byte-identical** to the per-env action-sampling keys of envs 0–4 at step *t+2* (all five equalities verified `True`).

**Concrete failure scenario.** When env 1's episode resets at step *t*, the random draws
that place its agent, food, and predators are the *same* random draws that pick the actions
of envs 0–4 two steps later. Episode-initialisation randomness and action-exploration
randomness are therefore correlated across the whole batch, every step (the reset branch is
computed unconditionally; it is *applied* on `done`). Expected practical magnitude is small,
but it is a genuine violation of the "never reuse a key" rule the project's own conventions
encode, it quietly couples exploration to environment layout, and it is a one-line fix:
`key, reset_key = jax.random.split(key)` (or `fold_in` a constant), matching the plain-PPO
sibling. Reproducibility (same seed → same run) is unaffected.

## Finding 3 — GAE `terminateds` mask inherits the latent `termination_reason` quirks

**Severity:** Low (GAE branch only — no live config uses it; trigger conditions are themselves latent env bugs)
**Where:** `src/models/recurrent_ppo_trainer.py:330` (`terminateds = termination_reason >= 2`), same pattern `ppo_trainer.py:226`
**Status: NEW interaction of KNOWN root causes** (registry rows: "over-eating never actually ends the episode"; "termination-reason unreliable when a body system is off")

**What happens.** The new real-death mask trusts `termination_reason ≥ 2`. Two known env
quirks break its contract: (a) `overeating_death=True` sets `termination_reason=3` **without**
`done=True` — in GAE mode such a step would have its bootstrap zeroed (`delta = r − V`)
*mid-episode* while the advantage chain keeps running, corrupting nearby advantages; (b) the
reason code is unreliable when a body system is disabled. Harmless today (GAE off the live
path; `overeating_death` rare), but if either the GAE path is revived (see Finding 1's
recommendation) or the env quirks are fixed independently, this mask's assumptions should be
re-checked in the same change. Worth a one-line comment at both mask sites.

## Finding 4 — Low-severity nits (no live-training impact found)

| # | Where | Note |
|---|---|---|
| 4a | `src/models/modulated_gru_cell.py:8, 76` | Docstring claims "functionally identical to nnx.GRUCell" when unmodulated — the update-gate polarity is actually **inverted** vs Flax (`h_new = (1−u)·h + u·n` here; `(1−z)·n + z·h` in Flax). It's a benign reparameterization (sign flip of gate weights), and the **sign semantics match the design doc** (NEUROMODULATION_ALGORITHM.md H2: negative `z_memory` → gate→0 → retention — true in this convention). No functional bug; docstring could mislead a future numerical-parity test. |
| 4b | `src/models/recurrent_ppo_network.py:303, 327` | `__call__` docstring says it returns a 3-tuple; it returns 4 (`logits, value, h_new, mod_info`). Also: with `rnn_type: "LSTM"` + modulation enabled, the gate-bias injection (Injection B) is **silently skipped** — all live modulated configs use GRU ("required for gate-bias injection" comment), but a config error would degrade silently rather than raise. |
| 4c | `src/models/recurrent_ppo_trainer.py:284-293` | `mod_grad_norm` extraction is wrapped in a bare `except: pass` — if the `'modulator' in grads` lookup semantics ever change with a Flax upgrade, the logged modulator-gradient norm silently becomes 0.0 forever. Logging-only. |
| 4d | `src/models/recurrent_ppo_trainer.py:126-161` | Entropy bonus is computed on temperature-scaled logits, so `ent_coef` can be partially satisfied by pushing the temperature head to its ceiling instead of genuine policy diversity. Consistent with the (intended) temp-ceiling study designs (`tempceil5/10` configs); informational only. |

---

## Sub-surfaces verified correct (explicit negatives)

- **obs/action/reward/done alignment** in the rollout buffer: `obs_t` is the pre-action
  observation of `state_t`; `value_t = V(obs_t)`; `reward_t/done_t` from stepping with
  `action_t`; `termination_reason` read at the done step in `train.py:1401`. No off-by-one.
- **Hidden-state reset discipline**: reset **after** the step's forward, gated on that step's
  `done`, identically in collection (`recurrent_ppo_trainer.py:230`) and loss replay
  (`:137-141`), for the full PyTree — task GRU/LSTM *and* modulator hidden both reset. The
  stored per-step hidden is the *pre*-forward state, so `h_init = h_states[0]` replays the
  sequence exactly.
- **GAE bootstrap hidden state**: `next_value` uses `h_new` (un-reset) on the true next
  observation pre-auto-reset — correct continuation semantics (`:206-212`).
- **vmap axes**: env batch on axis 0 everywhere; `EnvParams` broadcast (`in_axes=None`);
  loss vmap over env axis 1 of time-major arrays with `h_init` axis 0 (`:273-274`) — correct.
- **Advantage normalization order** (GAE): targets computed from raw advantages *before*
  normalization (`:339-340`). No padded steps exist (fixed-length windows), so nothing is
  normalized over padding.
- **PRNG elsewhere**: per-env action keys from a dedicated `act_key` branch (`:184-185`) —
  distinct per env, advanced each step; plain-PPO collector fully clean; top-level `train.py`
  key threading clean.
- **Symlog consistency**: applied inside `model.__call__` (`recurrent_ppo_network.py:308`),
  so collection, loss replay, bootstrap forward, and eval all see identical preprocessing.
- **Modulator math**: FiLM `γ·x+β` with pass-through init (γ-bias 1.0, β 0.0, zero baselines);
  PreActivation and Multiplicative forms match the design doc; grouping repeat/slice maps
  group *i* to neurons `[i·g, (i+1)·g)`; memory clamp applied (`neuromodulator.py:167`);
  temperature head shape `(...,1)` broadcasts correctly over the action dim.
- **No stale jit closures**: `env_params` is a traced argument (static fields participate in
  the trace hash, so stage transitions correctly recompile); `config` is a static argnum;
  nothing baked into closures that changes at runtime.
- **`done` vs `terminated` desync**: none found in this stack (trainer derives `terminateds`
  from `termination_reason` per step; never stores a conflicting copy).
- **Stage-transition mechanics** (`train.py:1224-1301`): env rebuilt, hidden state re-zeroed,
  accumulators wiped — consistent.

## Conventions audit checklist

pytree/immutability ✅ · JIT static discipline ✅ · vmap axes ✅ · **PRNG ❌ (Finding 2)** ·
sensor/obs-breakdown sync ✅ (untouched, layout stable) · config protocol ✅
(`get_mandatory` throughout the rPPO init path)

## Verdict

The rPPO stack is **structurally sound** — sequence handling, hidden-state resets,
loss math, modulator wiring, and all four landed fixes in scope are correct. Two live-path
issues remain: the **MC return window-truncation bias (Finding 1)** is the most consequential
open correctness question in this area — it needs an explicit accept-or-fix decision, plus a
registry clarification that `3c60f6f` does not touch live runs — and the **collector's PRNG
key reuse (Finding 2)** deserves its one-line fix at the next convenient commit. Nothing
found suggests live runs are learning from *wrong* data (the env-side reward fix `ef0fd25`
carries the headline correction); both findings are *bias/hygiene* defects, not corruption.

Reviewed by: `code-reviewer` (independent diagnosis pass, session 2026-07-04)
