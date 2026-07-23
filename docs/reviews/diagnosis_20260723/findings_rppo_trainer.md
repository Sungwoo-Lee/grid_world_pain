# Diagnosis findings — src/models/recurrent_ppo_trainer.py (2026-07-23)

Reviewer: JAX/Flax RL correctness pass on the recurrent PPO loss/returns/update machinery.
Scope: `src/models/recurrent_ppo_trainer.py` (all 422 lines), interface verification against
`src/models/recurrent_ppo_network.py`, env-side `termination_reason` / `update_body` semantics in
`src/environment/core.py:55-131, 690-745`, live configs under `configs/models/recurrent_ppo/` and
`configs/environment/`, and the regression tests `tests/models/test_gae_truncation.py` +
`tests/models/test_mc_window_bootstrap.py` (run: 18/18 pass in the project conda env).
Cross-checked against `docs/reviews/review_nmn_trainer_parity_20260722.md` and
`docs/reviews/review_h4_mc_window_bootstrap.md` to avoid duplication. Report-only; no code touched.

---

## Finding 1 — P1: MC window-edge bootstrap is in the wrong units (normalized-space critic value folded into raw-reward returns)

**Severity:** P1 — biased value targets in every live rPPO config (all live configs set `return_mode: "MC"`), strongest exactly in death-heavy training phases.

**Location:** `src/models/recurrent_ppo_trainer.py:374-380` (MC branch of `train_iteration`), interacting with `compute_mc_returns` at `:95-136`, value loss at `:184`, and the bootstrap forward at `:311-314`.

**Claim (plain English).** The H4 fix seeds the Monte-Carlo return scan with the critic's estimate of future return at the 128-step window edge. But the critic is trained (line 184, MSE) against **per-window z-scored returns** (line 378: `returns = (returns - mean)/(std + 1e-7)`; line 379: `targets = returns`), so its outputs converge to O(1) normalized-space numbers. That raw critic output is then used at line 375 / inside `compute_mc_returns` (lines 130-135) as the reverse-scan carry **added directly onto raw, un-normalized rewards** (`ret = reward + gamma * ret`). The bootstrap is therefore in the wrong units: it should contribute the raw future return `G_future`, but it contributes approximately `(G_future − μ)/σ` where μ, σ are the (evolving) per-window return statistics.

**Concrete failure scenario.** `configs/environment/default.yaml:182` sets `death_penalty: 100` and `configs/models/recurrent_ppo/recurrent_ppo_L.yaml:6` sets `gamma: 0.95`. In any window containing deaths, raw discounted returns span O(−100·γ^k … +1), so the batch std σ is ≫ 1 (order 10–100). The critic, trained on z-scored targets, outputs O(1). The window-edge seed is then ~σ× too small in magnitude (plus a −μ/σ offset) — i.e. in exactly the death-heavy regimes where window-edge credit matters most, the H4 bootstrap contributes almost nothing and the original position-dependent window-edge bias (steps near the edge credited with ~none of their true future) substantially re-appears. Conversely, if a quiet window ever has σ < 1, the bootstrap is inflated. With γ=0.95, γ^(T−t) is non-negligible for roughly the last 30–50 steps of each 128-step window (~25–40% of all training targets). The 7 H4 unit tests cannot catch this: they feed synthetic numbers and never close the loop through the normalized-target-trained critic. Neither `review_h4_mc_window_bootstrap.md` nor the 2026-07-22 parity review mentions normalization (grep: zero hits), so this is unflagged.

**Evidence (code path).** (a) Lines 374-376: `compute_mc_returns` vmapped over `(trajectories.reward, trajectories.done, terminateds, bootstrap_value, gamma)` — rewards are raw env rewards. (b) Lines 130-135: carry initialised with `bootstrap_value` and folded via `ret = reward + gamma * ret`. (c) Lines 378-379: the summed returns are z-scored and become `targets`. (d) Line 184: `value_loss = 0.5 * mean((new_values - targets)^2)` — the only training signal the critic gets, so `V ≈ (G − μ)/σ` at convergence. (e) Lines 311-314: `bootstrap_value = v_boot.squeeze(-1)` — the raw critic output, no de-normalization. The GAE branch (lines 386-396) is by contrast internally consistent: raw-scale values, raw-scale bootstrap, advantages normalized only after targets are formed.

**Fix direction (one line).** De-normalize the seed with tracked raw-return statistics (`μ + σ·V`, using running or previous-window stats), or drop per-window return normalization in MC mode and normalize advantages instead (aligning with the GAE branch).

---

## Finding 2 — P2 (latent; verifies a recorded LATENT row): instant-death mode produces real deaths with `termination_reason < 2`, silently defeating both death gates

**Severity:** P2 — not reachable in live configs (every live env config inherits `with_injury: true` from `configs/environment/default.yaml:147`; only archived configs override), but the GAE return variant just went live (`recurrent_ppo_gae.yaml`, commit 38cc1b3), which raises the cost if an instant-death config is ever revived.

**Location:** consumer at `src/models/recurrent_ppo_trainer.py:373` and `:386` (`terminateds = termination_reason >= 2`); producer at `src/environment/core.py:117-126` (`update_body`: `with_injury=False` ⇒ `done = damage > 0`, instant death) vs `core.py:704-710` (reason codes: 2 requires `with_nutrition` + nutrition ≤ 0, 3 requires `overeating_death`, 4 requires `new_injury >= max_injury` — but with `with_injury=False`, `new_injury = prev_injury` stays 0 forever and never reaches `max_injury`).

**Claim.** In `with_injury: false` configs, a predator hit kills the agent (`done=True`, death penalty fires via `real_death`) but `termination_reason` stays 0 (or 1 if it coincides with the step limit). The trainer's real-death mask `termination_reason >= 2` then classifies that real death as a non-death: (a) GAE (line 83): `real_death = done * terminated = 0`, so `delta` keeps `gamma * next_value` — the critic bootstraps future value out of a corpse state; (b) MC edge gate (line 119): `edge_death = done AND terminated = False`, so a death exactly at the window edge keeps its bootstrap instead of zeroing it. Mid-window MC resets are unaffected (gated on `done` alone).

**Concrete failure scenario.** Revive any archived `with_injury: false` config (several exist under `configs/environment/experiment/archive/`) with the now-live GAE agent config: every predator-death step trains the critic toward `r_death + γ·V(dead-state)` instead of `r_death`, systematically inflating values near predators and weakening avoidance — directly corrupting the survival-steps objective.

**Evidence.** Traced `update_body` (`core.py:117-126`), the reason stamping (`core.py:704-710`, incl. the 8334d89 `with_nutrition` guard), and both trainer mask sites. This confirms and concretizes the KNOWN_BUGS **LATENT** row "termination-reason unreliable when a body system is off" — recorded there, now verified with the exact mechanism and trainer-side blast radius. Not a new bug; requesting the bug-curator row be upgraded from LATENT to verified/open with this trace.

**Fix direction.** Stamp a dedicated reason code for instant death in `core.py` (e.g. reuse 4 whenever `update_body`'s death fires, regardless of `with_injury`), or derive `terminateds` in the trainer from `real_death` exported through `info` instead of reverse-engineering it from reason codes.

---

## Finding 3 — P2: MC-mode critic chases a nonstationary per-window affine target

**Severity:** P2 — quality/variance issue, deliberate legacy design ("PyTorch parity" comment), but now load-bearing because Finding 1 hangs off it.

**Location:** `src/models/recurrent_ppo_trainer.py:377-379`.

**Claim.** Returns are z-scored with the **current window's** batch mean/std, so the critic's regression target is a different affine transform of raw returns every iteration (μ_w, σ_w drift as the policy improves and death frequency changes). The critic can never converge to a stationary function; its error floor is the iteration-to-iteration wobble of (μ, σ). This also injects noise into MC advantages (`returns_norm − value`, line 380 — value lags one normalization regime behind) and is the root cause enabling Finding 1.

**Evidence.** Lines 377-380; contrast with the GAE branch which keeps targets in raw units and normalizes only advantages (line 396).

**Fix direction.** Same as Finding 1's second option: normalize advantages, not returns (single change resolves Findings 1 and 3 together); if return normalization must stay for parity, use running (EMA) stats rather than per-window stats.

---

## Fixed-bug regression check

| Fix | Status | Verification |
|---|---|---|
| **H4 — MC-return bootstrap at the 128-step window edge** | **Present & mechanically correct** (with the Finding-1 units caveat) | `compute_mc_returns` (`:95-136`) seeds the reverse scan with `bootstrap_value`; edge gate `edge_death = done[-1] AND terminated[-1]` (`:119-120`) zeroes only on real edge death, retains on timeout/mid-episode cut. Bootstrap forward (`:311-314`) uses `boot_state`/`boot_h` carried out of the scan as the **pre-auto-reset** `(next_state, h_new)` of the last step (carry slots 4/5, set at `:296` from the raw `jax_step` output before the `:262-265` reset selection). vmap axes `(1,1,1,0,None)/out 1` correct for `(T,B)` + `(B,)`. `tests/models/test_mc_window_bootstrap.py`: pass. **However**, the bootstrap's practical effect is degraded by the normalization units mismatch (Finding 1) — the fix is correct in isolation but not in composition with the MC target pipeline. |
| **GAE timeout value-drop** | **Present & correct** | Per-step `next_value` computed on the true pre-reset next state with un-reset `h_new` (`:239-243`), stored in `Transition.next_value` (`:293`), consumed by `compute_gae` (`:391-393`). Timeout step: `real_death=0` ⇒ delta keeps `γ·V(s')`; accumulation chain still cut by `(1-done)` (`:84`). No auto-reset fresh-episode value can leak in. |
| **218a366 — GAE death bootstrap gated on done AND terminated** | **Present & correct** | `:82-83`: `real_death = done * terminated`; `delta` uses `(1 - real_death)`; reset stays on `(1 - done)` (`:84`). Mirrors the MC edge gate; guards the overeating quirk (reason 3 without done ⇒ bootstrap retained, chain uncut — correct for a continuing episode). `tests/models/test_gae_truncation.py`: pass. Note Finding 2 is the inverse hole (done without reason ≥ 2), env-side, latent. |
| **b8eb286 — carried PRNG key advanced when drawing the auto-reset key** | **Present & correct** | `:254`: `key, reset_key = jax.random.split(key)` advances the carry (was `reset_key, _ = split(key)` aliasing step t's reset key with step t+1's master key per the 07-22 parity review). Per-step consumption is now: `:214` split → `act_key`, `:254` split → `reset_key`; both sub-keys fanned out with independent `split(·, B)`; no sub-key reused; `final_key` returned from the carry and threaded onward by `train_iteration`. |

---

## Reviewed but clean

- **PPO ratio/clip math** (`:178-181`): `exp(new − old)`, two-sided clip, `-mean(min(surr1, surr2))` — standard and correct; old log-probs come from the identical tempered-softmax path.
- **Value-target detachment**: `advantages`/`targets` are computed in `train_iteration` from concrete rollout arrays **before** `nnx.value_and_grad` (`:330`), so they are constants inside the loss — no `stop_gradient` needed, none missing.
- **Entropy term** (`:165, :187-189`): computed from the actual (temperature-scaled) policy logits; sign convention correct (`+ ent_coef * (−entropy)` maximizes entropy). Temperature receiving entropy-bonus gradient is a legitimate property of entropy-regularizing the true policy (Injection C is part of the policy; `temp_clip` bounds it) — not a bug.
- **Loss re-unroll parity** (`:159-175`): the loss scan reproduces collection exactly — same `h_init` (pre-step h at t=0, extracted from stacked `h_states[0]`, `:152-154, :399`), same post-step reset `_h_reset_on_done(h_new, done)` on merged done in both places (`:169` vs `:268`) — so epoch-0 ratios are exactly 1. Symlog obs compression lives inside the shared `model.__call__`, applied identically in both passes.
- **Hidden-state PyTree handling**: `_h_vmap_axes` / `_h_reset_on_done` / `_h_get_first_timestep` are structure-agnostic (GRU array, LSTM tuple, modulated `(task_h, mod_h)` pair); broadcast reshape at `:148` correct for both batched (B,) and scalar done. No plain hidden state fabricated where a pair is needed (confirms parity-review checks 2/3).
- **vmap/scan axes**: trajectories `(T,B,·)` with `in_axes=1`, `h_init` `(B,·)` with 0 (`:325`, `:374`, `:391`); scan over T inside per-env loss; `out_axes` restore `(T,B)`. All verified.
- **Interface with `recurrent_ppo_network.py`**: model returns a 4-tuple `(logits, value, h_new, mod_info)` on both modulated and unmodulated paths (`recurrent_ppo_network.py:351, :373`); all three trainer call sites unpack 4 (`:161, :242, :313`); `mod_info=None` is a valid empty pytree under vmap/scan.
- **No minibatching** — full-batch, full-window updates; no sequence-continuity break to get wrong. Epoch loop reuses frozen batch with per-epoch recomputed ratios (standard PPO). Stale `h_init` across epochs is the standard truncated-BPTT-PPO approximation, not a defect.
- **MC mid-window timeout finite-horizon treatment** (return reset to 0 on merged done, `:126`): documented deliberate (docstring `:104-107`; H4 review nit 2) — not re-reported.
- **`PPOBatch.values` unused** in the loss (no value clipping implemented): dead field, harmless; noting for awareness only.
- **Overeating quirk handling in MC** (`:119-120`): mid-window resets are gated on `done` alone, so reason-3-without-done cannot corrupt mid-window returns; edge gate handles the edge case. Correct.
- **`mod_grad_norm` probe** (`:336-347`): narrowed `except (KeyError, TypeError)` landed in 8334d89 (parity-review nit); `'modulator' in grads` verified live in that review. Clean.
- **Reason-code precedence** (`core.py:704-710`): death codes stamped after the truncation code, so death-at-the-buzzer is classified as death (`terminated=1`) — correct for both return modes.
- **Known/OPEN items not re-reported**: `lr_critic` dead (train.py), continual-stage accumulators (train.py), `auto_reset_step()` key-arg (env), overeating-never-ends-episode (env). PRNG aliasing and GAE overeating gate from the 07-22 parity review are confirmed fixed (b8eb286, 218a366) rather than re-reported.
