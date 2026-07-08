---
title: "Code review — WP-NNX DreamerV3-NNX recipe-alignment fixes (F1–F5, F7)"
topic: dreamer_sheeprl_parity
status: active
created: 2026-07-08
last_updated: 2026-07-08
---

# Review: WP-NNX recipe-alignment fixes (uncommitted diff)

## Verdict

**APPROVE-WITH-NITS.** The seven-part fix package that aligns the in-house
DreamerV3 agent (the JAX/Flax-NNX world-model agent trained by `train.py`)
with the canonical DreamerV3 recipe (the vendored sheeprl reference) is
correct in all its gradient-structure, indexing, and PRNG-threading details —
every recipe site was verified line-by-line against the vendored sheeprl
source, and all 17 new regression tests plus 11 pre-existing Dreamer tests
pass. One genuine deviation remains: the replay-ratio bookkeeping does not
subtract the random-action prefill the way sheeprl does, so the **first**
training iteration fires a one-time burst of "catch-up" gradient steps. For
the everyday configs (whose ratio values were rescaled 128× down) the burst is
~4 steps — noise. For the full-intensity parity config (the one that mirrors
sheeprl's exact recipe, ratio 1.0) the burst is ~1,500 gradient steps on a
buffer of ~1,500 mostly-random transitions, where sheeprl itself would do ~4.
That should be fixed before that specific config is ever launched. Everything
else is nits or already-registered pre-existing issues.

Scope reviewed: `src/models/dreamer_v3_trainer.py` (main diff),
`train.py` (Dreamer dispatch region), `tests/models/*` (6 new test files +
2 shared helpers). `src/models/dreamer_v3_nnx.py` confirmed **untouched**
(empty diff). The parallel WP-SRL package (`src/algorithms/dreamer_srl/*`),
config-value diffs (`configs/`), and doc diffs were out of scope per the
task instruction.

## Findings

| # | Sev | Location | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | 🟡 concern | `train.py:1827` + `src/models/dreamer_v3_util.py:206-212` | **Prefill not subtracted from the Ratio argument.** sheeprl computes `ratio_steps = policy_step - prefill_steps * policy_steps_per_iter` (vendor `dreamer_v3.py:661`) before calling `Ratio`, so its first call sees ~one iteration's worth of steps. `train.py` passes raw `global_step`, and this project's `Ratio.__call__` first-call branch returns `int(step * ratio)` — a one-time backlog burst of ≈ `replay_ratio × (learning_starts + num_envs×collect_interval)` gradient steps at the first post-prefill iteration, plus a permanent `+replay_ratio×learning_starts` offset in cumulative accounting (`Params/effective_replay_ratio`). Rescaled live configs (`replay_ratio 0.00390625`, `learning_starts 1024`): burst ≈ 4–6 steps, negligible. `dreamer_v3_sheeprl_matched.yaml` (`replay_ratio 1.0`): burst ≈ 1,536 sequential gradient steps on a ~1,536-transition random-prefill buffer — a materially different training start from sheeprl in the exact config built to demonstrate parity. | Pass `global_step - learning_starts` to `ratio_scaled_updates` (mirroring vendor `:661`), or prime the `Ratio` instance's `_prev` at prefill completion. |
| 2 | 🟡 concern | `src/models/dreamer_v3_trainer.py:473-474,498-499,533,546` | **Continue = sigmoid probability; recipe uses Bernoulli mode (hard 0/1).** sheeprl binarizes imagined continues (`Independent(BernoulliSafeMode(...)).mode`, vendor `dreamer_v3.py:246`) before they enter both the λ-return and the cumulative discount weights; the in-house code feeds soft sigmoid probabilities into the same two consumers (soft geometric decay instead of a hard cutoff at p=0.5). **Pre-existing — not introduced by this diff** — but it rides the exact lines F4/F5 modified and is absent from the deviation register (`06_nnx_recipe_deviation_register.md` has no row for it). | Add a register row (declare-or-fix decision belongs to senior-developer); a fix would be `cont = (sigmoid(...) > 0.5).astype(f32)` at the two scan sites. |
| 3 | 🟢 nit | `src/models/dreamer_v3_trainer.py:524-530` | **`v_start` is dead compute and the F4 comment overstates its role.** `all_vals[0]` is never read: `compute_lambda_values` consumes only `values[1:]` and `values[-1]`, and the actor baseline comes from `v_pred_logits`, not `all_vals`. This matches sheeprl (its λ-computation also never uses v′₀), and XLA will DCE it — but the comment "v_start from the ONLINE critic (sheeprl D:244)" implies the F4 source switch here has an effect; it doesn't. | Drop the `v_start`/`all_vals[0]` computation (pass `vals` + explicit bootstrap) or correct the comment to say the entry is structurally unused. |
| 4 | 🟢 nit | `src/models/dreamer_v3_trainer.py:687,697` and `:457,465` / `:493,494` | **PRNG sub-key reuse (pre-existing, already registered as K7).** `get_action` reuses the same `key` for the RSSM posterior sample and the action sample; `scan_imag` reuses `key` for `dist.sample` and `imagine_step`. Not introduced or worsened by this diff — the new F7 prefill threading (`act_key`/`rand_key` from separate splits) is clean. Confirming the register row stays open. | No action in this diff; owned by KNOWN_BUGS K7. |
| 5 | 🟢 nit | `train.py:1538` vs vendor `dreamer_v3.py:559` | **One-iteration boundary difference in the prefill window.** In-house switches to policy actions in the same iteration training begins (`global_step < learning_starts` evaluated pre-collection); sheeprl keeps random actions through the iteration where training starts (`iter_num <= learning_starts`). One iteration of overlap; immaterial. | None needed. |
| 6 | 🟢 nit | `src/models/dreamer_v3_trainer.py:122,420-423` vs vendor `dreamer_v3.py:678` | **No first-update hard copy of online→target critic** (sheeprl uses `tau=1` on gradient step 0; here the target starts as an independent random init and is only ever EMA-blended). F4 makes this slightly more relevant: the slow-critic regularizer briefly pulls the online critic toward a random target. Benign while `zero_init_reward_critic: true` (both heads output identical constants at init). Already noted in the register (K6 note on row K2–K7). | No action in this diff; already registered. |

## Per-fix verification detail

### F1 — episode-boundary reset in `get_action` (`dreamer_v3_trainer.py:643-664,802-807`)
- `is_first` read via `prev_state.get('is_first', jnp.zeros((B,1)))` — traced data
  in the scan carry, dict-key presence resolved at trace time; **no recompile
  trigger**. Staged as `done[...,None].astype(float32)` (`:807`), carried
  consistently (initial carries set `is_first=ones` at `train.py:887,1285` and
  `collect_sequence:736`) — carry pytree structure stable across scan steps.
- `prev_action * (1.0 - is_first)`: `(B, act_dim) × (B, 1)` broadcast, float32
  both sides — correct, no dtype surprise.
- `mod_h` reset via `jnp.where(is_first > 0.5, initial_state(B), mod_h)`:
  `(B,1)` bool vs `(B, mod_hidden)` — both modulator classes'
  `initial_state` return a single `(B, mod_hidden)` array (not a tuple), so the
  broadcast is well-formed (`neuromodulator.py:375-379`). The `where` form
  (vs mask-multiply) is the right future-proof choice.
- Complementarity with the untouched `RSSM.step` mask (`dreamer_v3_nnx.py`
  `deter/stoch × (1−is_first)`) verified: the RSSM masks latents, `get_action`
  handles the two things the RSSM cannot (`prev_action`, `mod_h`). No
  double-reset conflict.
- Back-compat: callers that never stage `is_first` (eval paths) get the old
  behavior exactly (default zeros). Eval-path reset remains open as K2 —
  unchanged by this diff.

### F2 — stop-gradient sites (`dreamer_v3_trainer.py:454,490,555,571`)
Verified against vendor `dreamer_v3.py:219,240,273,286,307`:
- Actor input feats detached at creation in both scan branches (`:454,:490`);
  `rollouts['feat']` therefore stores detached feats, and the critic-loss
  re-detach at `:555` mirrors vendor `:307` (double-sg harmless).
- Log-prob consumes the detached action (`:571`, vendor `:286`) — kills the
  straight-through ∇probs term.
- **Not over-detached**: actor logits path (`actor(sg(feat))` → log_probs,
  entropy) live; critic path (`critic(sg(feat))` → both CE terms) live;
  advantage/discount/λ-targets sg'd as the recipe requires. Confirmed
  empirically by the new tests (critic-loss grad w.r.t. actor exactly zero;
  actor-loss grad w.r.t. world model exactly zero **with** a non-vacuity
  control that the actor itself does receive gradient; actor grad bit-matches
  a detached replica on the same PRNG stream).
- The modulated branch's `mod_input`/`imagine_step` consume the undetached
  action — same as the recipe's imagination chain; no live path to the loss
  survives (all downstream consumers sg'd or non-grad-arg modules).

### F3 — obs loss mean→sum (`dreamer_v3_trainer.py:54-65,256`)
`jnp.mean(jnp.sum(square, axis=-1))` — sum over the feature/event dim only,
mean over batch & time; matches sheeprl's `SymlogDistribution` log-prob (sum
over event dims, no ½ factor). Input is symlog-space on both sides (batch obs
symlog'd at `train_step:157`). `[T,B]`-mean preserved. Only one recon-loss
site exists in the file. ✅

### F4 — online-critic bootstrap + slow-critic regularizer (`:477,:502,:528,:550-562`)
- Scan `val` and `v_start` now from the **online** `critic` (vendor `:244`);
  grep confirms **no residual `target_critic` bootstrap reads** — the target
  appears only at construction (`:122`), the EMA update (`:420-423`, gradient-
  free `tree_map`, intact), and the regularizer (`:557`).
- No gradient flows from the λ-return side back into the online critic:
  `lambda_returns` consumers are `target_twohot` (sg, `:556`), `advantage`
  (sg, `:567`), and non-grad metrics/moments. Verified structurally and by
  the corrupt-the-target-critic bit-identity test.
- Slow-reg gradient reaches the **online critic only**: `target_critic` is not
  an `nnx.grad` argnum (accessed via `self`), and `slow_twohot` is sg'd
  belt-and-braces. The two-hot CE toward the target critic's symexp'd
  predictions, weighted by `discount_weights`, matches vendor `:307-316`
  term-for-term, including summing both CE terms before the weighted mean.

### F5 — true-continue first-row override (`:378-382,:539-548`)
Index-by-index match with vendor `:247-248,260` confirmed, including the
`/γ` cancellation: vendor `discount[t] = cumprod(continues*γ)/γ` with
`continues[0]=1−terminated` equals in-house
`cumprod(concat([true_cont0, conts[:-1]·γ]))` exactly (row 0 = true_cont0;
row t = true_cont0·γᵗ·∏ᵢ₌₁..ₜ cont(sᵢ)). `true_cont0` reshaped `(B,T)→(B*T,)`
with the same flattening as `start_state` ✅, sg'd ✅, and correctly **excluded**
from the λ-return continues (vendor uses `continues[1:]` there too) ✅.
`compute_continue_target` reuse means timeout rows (reason 1) keep weight 1 —
consistent with the project's truncation-vs-death convention.

### F7 — replay-ratio semantics + prefill (`train.py:844-851,1530-1538,1814-1827`; `collect_sequence:709-763`)
- Per-env-step accounting: grad steps/iter = `ratio × num_envs × collect_interval`,
  which equals sheeprl's per-policy-step semantics at steady state ✅
  (**except the first-call backlog — Finding 1**).
- `random_actions` is static (argnum 6, `self` counted at 0 — consistent with
  the pre-existing `num_steps` at 3); flips exactly once → one extra compile,
  as advertised. `bool(...)` coercion prevents a traced-bool leak into the
  static arg. ✅
- Prefill PRNG: `rand_key` from a fresh split after `act_key`; no collision
  with the policy key stream; `get_action`'s internal stream untouched. ✅
- `next_d_state['prev_action']` overwritten with the executed random action —
  RSSM belief stays paired with reality (vendor player analog). ✅
- `agent.learning_starts` read via `get_mandatory(..., int)` (signature
  supports the type arg, `config.py:67`), scoped inside the Dreamer dispatch
  branch (no crash for other algorithms), and present in **all** files under
  `configs/models/dreamer_v3/` (grep -L empty). Configuration Protocol ✅.

### `_behavior_loss` extraction (`:397-403,:427-620`)
Line-by-line comparison against the removed closure: identical except the six
documented F2/F4/F5 deltas plus the new `loss_critic_slow_reg` metric. All
formerly-closed-over values (`start_state`, `h_mod_start`, `moments_*`,
`true_cont0`) are threaded explicitly; the `else: h_mod_start = None` branch
correctly covers the non-modulated path (previously the name was only defined
under modulation). `true_cont0=None` default reproduces pre-F5 `row0=1`
semantics exactly, as documented. No stale-closure or aliasing hazard:
`self` is still closed over for `wm`/`target_critic`, same as before, and
neither is a grad argnum.

### Tests (6 new files, 2 helpers)
- **Discriminative power**: the `zero_init_reward_critic: false` override is
  justified in-file and is correct — with the live zero-init, critic/reward
  heads output constants at init, making the U1 leak paths and the U2
  bootstrap-source swap invisible; random heads expose them. Non-vacuity
  controls present in every file (garbage-carry-with-is_first=0 control,
  actor-gets-gradient sanity, all-alive-batch nonzero-loss control,
  degenerate-policy control, forward-value replica agreement guard).
- **Exact-zero assertions** (`leaf == 0.0`) are the right tolerance:
  `stop_gradient` yields structural zeros, not approximate ones — exact
  equality is strictly more discriminative and cannot flake.
- The replica helper carries an explicit lockstep-maintenance contract naming
  F4/F5 — good.
- One gap, consistent with Finding 1: `test_ratio_per_env_step_accumulation`
  starts calling `Ratio` from step ~0, so it never exercises the
  post-prefill first-call backlog; a fix for Finding 1 should extend it.
- Executed: **17/17 new tests pass** (118 s CPU); **11/11 pre-existing**
  Dreamer tests (`collect_arrival_alignment`, `continue_truncation`,
  `replay_buffer_wrap`) pass — no regression in the touched paths.

## Conventions audit

| Convention | Status | Note |
|---|---|---|
| Pytree & immutability | ✅ | No in-place pytree mutation; state dicts rebuilt per step; sg used, not mutation |
| JIT recompilation | ✅ | `random_actions` static by design, flips once (2 compiles total); `is_first` stays traced |
| vmap & batch | ✅ | Env vmaps unchanged; `EnvParams` broadcast (`in_axes=None`) throughout |
| PRNG threading | ✅ (new) / 🟡 (pre-existing) | New F7 threading clean; pre-existing intra-step sub-key reuse remains open as K7 |
| Sensor/obs-breakdown sync | n/a | No sensor or observation-layout change in scope |
| Config protocol | ✅ | `agent.learning_starts` mandatory, typed, branch-scoped, present in all 27 dreamer configs |

## Conclusion

Approve with nits: fix the Ratio prefill-backlog burst (Finding 1) before any
launch of the full-intensity `dreamer_v3_sheeprl_matched` config; register or
fix the continue-probability-vs-mode deviation (Finding 2); everything else is
recipe-exact and well-tested.

Reviewed by: code-reviewer
