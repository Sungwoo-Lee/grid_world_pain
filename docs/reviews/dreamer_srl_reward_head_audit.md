---
title: dreamer-srl v2 reward head + imagination roll-out audit
topic: dreamer_srl_v2
status: active
created: 2026-05-18
last_updated: 2026-05-18
---

# Dreamer-SRL reward-head correctness audit

## Plain-language entry point

This audit checks the dreamer-srl v2 code paths that compute reward predictions — both during training (the reward head learning from replay-buffer samples) and during imagination (the world-model rollout that drives the actor's policy gradient). The audit was commissioned because the empirical data shows the world model systematically under-fits negative rewards (the mean absolute error on negative rewards is 2.2-6.4× higher than on positive rewards across every long 10×10-hypervigilance run we have). Phase 1's anchor analysis ([`REWARD_HEAD_ASYMMETRY_ANALYSIS.md`](../experiments/active/dreamer_srl_v2/REWARD_HEAD_ASYMMETRY_ANALYSIS.md)) framed three hypotheses for why dreamer-srl underperforms the recurrent-PPO baseline (191 vs 227-239 episode-survival steps); this audit covers H1 (reward-head asymmetry as the gap driver) and partially H2 (imagination-horizon compounding). The bottom line: no bug, but the pos/neg asymmetry has an intrinsic component baked into the symlog encoding, and the in-buffer class imbalance is the likely dominant remaining failure mode.

## Verdict (one paragraph)

The reward head + imagination roll-out code in `src/algorithms/dreamer_srl/` is, by line-by-line inspection against sheeprl@33b6366, semantically faithful to the upstream reference. **No red (bug) finding surfaced.** The reward head is the live head everywhere (no detached imagination copy); twohot encode → log_prob → consumption is numerically stable (log_softmax via `logsumexp`); replay sampling is uniform; the actor objective uses the threaded rollout actions (v1's H1 bug stays fixed). One **artefact in the aggregation of `model_reward_mae` (`_total`)** is worth surfacing for the Phase 1 doc — `_total` is unconditional mean dominated by the near-zero bucket while `_pos`/`_neg` are masked subsets, so the three numbers are **not** statistically comparable and the magnitude of the pos/neg asymmetry as currently logged is real but **not literally** a buffer-mass-weighted mean of the two halves. The imagination roll-out does NOT feed predicted reward into next-step latents, so reward-head error does not compound across horizons by itself; long-horizon reward MAE compounding (the Z2 finding from original Dreamer) must therefore be inherited solely via *latent* drift — Phase 2b's offline-WM diagnostic is the right tool to confirm this.

## Q1 — Twohot numerical stability on deep-negative tail — ✅ correct

- `src/algorithms/dreamer_srl/loss.py:120` — `bins = linspace(-20, +20, 255)` in **symlog** space (confirmed by docstring triple-consistency contract and grid endpoints).
- `src/algorithms/dreamer_srl/loss.py:201` — `x = symlog(x)` is applied to the target BEFORE bin lookup. symlog(-355) ≈ −5.875 maps to bin index ≈ 89.7 (well inside [0, 254]; bin spacing ≈ 0.1575). No boundary clamp, no NaN risk.
- `src/algorithms/dreamer_srl/loss.py:242` — `log_pred = self.logits - logsumexp(self.logits, axis=-1, keepdims=True)` — this is the stable log-softmax form, NOT naive `log(softmax(...))`. Gradients are well-defined.
- `src/algorithms/dreamer_srl/loss.py:143` — `mean` property: `symexp(sum(probs * bins, ...))`. symexp on a tightly-peaked distribution is well-behaved (exp on a bounded symlog-space mean ≈ −5.9 → real-space ≈ −362, no overflow).
- **Intrinsic asymmetry note (📎 not a bug, just physics):** because the grid lives in symlog space, the real-space MAE is intrinsically larger for large-|reward| samples than for small-|reward| ones — any non-zero softmax mass spread across two adjacent bins gets stretched exponentially by symexp on the way back to real space. The pos rewards (≤ +1 typical) sit near bin 127 where symexp is locally linear; the neg rewards (down to −355+) sit far out where symexp is locally exponential. So **a chunk of the 2.2-6.4× pos/neg MAE gap is expected from the encoding itself**, not the head's learning capacity.

## Q2 — Imagination uses LIVE reward head (no stop-grad copy) — ✅ correct

- `src/algorithms/dreamer_srl/train.py:831` — `predicted_rewards_logits = jax.vmap(world_model.reward_model)(imag_flat)`. This is the **live** `world_model.reward_model`, NOT a detached / EMA / target copy.
- The latent argument `imag_flat` is `jax.lax.stop_gradient`-ed via line 803-804 (`posteriors`, `recurrent_states`), so the imagined latents do not back-propagate into the WM, but the reward head is called fresh on the imagined latents.
- For Q4 purposes this is the correct sheeprl-faithful design (sheeprl dreamer_v3.py:L244 uses `world_model.reward_model(imagined_trajectories)` similarly). No "EMA reward head" exists in either codebase.

## Q3 — Negative-reward batch weighting / sampling — ✅ uniform, no bias against negatives

- **Buffer sampling**: `src/algorithms/dreamer_srl/buffers.py:284-289` — `self._rng.integers(0, len(valid_idxes), ...)` is uniform over all valid time indices. No prioritization, no reward-magnitude weighting. Sequences may straddle done-boundaries (the buffer rebuilds RSSM state via `is_first` inside the window).
- **Per-step loss weighting in `wm_loss_fn`** (train.py:700-751): `total = (kl_regularizer * kl_loss_2d + obs_loss_unreduced + reward_loss_unreduced + cont_loss_unreduced).mean()`. The `.mean()` is unweighted across `[T, B]` — equal weight to every timestep, including high-frequency near-zero rewards and rare large-negative rewards.
- **Twohot loss reduction**: `loss.py:245` — `(target * log_pred).sum(axis=self.dims)` with `dims=(-1,)` for `dims=1`, then averaged. No per-bin reweighting. The twohot target is a 2-bin interpolation regardless of reward sign.
- **PRNG bias toward early-frame samples**: `buffers.py:282-285` constructs `valid_idxes` excluding only the chunk near `self._pos`. There is no oversampling of episode-start frames. Sequences are sampled per-env and the env index is sampled per-sequence (line 344), so PRNG threading inside the buffer is sound.
- **Combined effect**: negatives are NOT downweighted at the data-pipeline or loss-reduction stages. The under-fit of negatives cannot be blamed on biased sampling; it must be searched for in (a) class imbalance × MLE loss (negatives are rare in the buffer, so the head's posterior is dominated by the near-zero mode), or (b) the symlog-encoding asymmetry from Q1.

## Q4 — Per-horizon reward-MAE compounding mechanism — 📎 matches upstream sheeprl ref, ✅ no extra v2 mechanism

- `agent.py:1785-1809` (`WorldModel.imagine` loop): step h+1's latent is built from `(recurrent_state, prior_flat_new)` where `prior_flat_new` comes from `rssm._transition(recurrent_state, ...)`. The predicted REWARD at step h is **never** fed back into the recurrent state, the prior, or the actor input.
- `train.py:828-848`: rewards/values/continues are predicted AFTER the rollout, in one batch shot via `jax.vmap(world_model.reward_model)(imag_flat)` over the full `[H+1, BT]` trajectory.
- Therefore the only mechanism by which reward MAE compounds across the horizon is **latent drift** — the prior at step h+1 is conditioned on the (imagined, imperfect) prior at step h, accumulating posterior-vs-prior divergence. The reward head then maps the drifted latent through a fixed mapping; its per-horizon error tracks the latent-quality decay, not a feedback loop through the reward head itself.
- **This is identical in structure to upstream sheeprl** (dreamer_v3.py:L235-L244 — same `rssm.imagination()` rollout, same post-hoc reward/value prediction). The Z2 finding from original Dreamer (0.18 → 3.05 across h=5→50) was attributed to this latent-drift × reward-head mechanism, and dreamer-srl v2 inherits the same pattern without modification.
- **Phase 2b dependency**: empirical per-horizon reward MAE compounding cannot be verified from static code review — it requires running the offline-WM diagnostic on a checkpoint. The diagnostic plan that `senior-developer` is drafting (forward-reference: [`docs/develop/active/dreamer_srl_v2/offline_wm_diagnostic_port_plan.md`](../develop/active/dreamer_srl_v2/offline_wm_diagnostic_port_plan.md) or similar) should report `model_reward_mae` at h ∈ {1, 5, 10, 20, 50}. If v2 shows the same 0.18 → 3.05 shape as Z2, the conclusion is "inherited from upstream sheeprl, not a v2 regression". If v2 is **flatter** than Z2, the parallel-env + zero-init combination is helping; if it is **steeper**, that is a new finding to investigate.

## Q5 — `_total` vs `_pos` vs `_neg` aggregation — ⚠️ concern (artefact, not a bug)

- `train.py:758-762`:
  ```
  rew_mae       = jnp.mean(jnp.abs(rew_pred - rew_target))               # ALL elements
  pos_mask      = (rew_target >  0.01)
  neg_mask      = (rew_target < -0.01)
  rew_mae_pos   = sum(|err| * pos_mask) / (sum(pos_mask) + 1e-8)         # POS subset only
  rew_mae_neg   = sum(|err| * neg_mask) / (sum(neg_mask) + 1e-8)         # NEG subset only
  ```
- `_total` averages over **all** `[T, B, 1]` elements, including the large "near-zero" bucket where `|reward| ≤ 0.01` (per-step penalty, idle steps). The near-zero bucket is by far the most populous in a hypervigilance batch.
- `_pos` and `_neg` are **conditional means** on disjoint subsets that exclude the near-zero bucket.
- **Algebraic relation**: `_total = (N_pos * _pos + N_neg * _neg + N_zero * _zero) / (N_pos + N_neg + N_zero)`. So `_total ≈ _neg` (= 0.76 close to 1.02) is consistent with: N_neg is small but `_neg` is large, AND N_zero is huge but `_zero` is small — the near-zero MAE dominates by mass but is itself low, so `_total` lands between `_zero` (small) and `_neg` (large), weighted by the dominant `_zero` mass. It is NOT a buffer-weighted mean of just `_pos` and `_neg`.
- **Anti-pattern impact**: the headline statement in Phase 1's analysis — "`rew_MAE_total = 0.76` closer to `neg=1.02` than to `pos=0.31`" — is correct as a numerical observation but should not be interpreted as evidence that "negatives drive the total". The total is mostly the near-zero bucket's MAE (probably around 0.5-0.7), which itself contains the asymmetric tail mass. The "real" pos/neg asymmetry is the `_pos` vs `_neg` ratio directly (3.3×), and the `_total` is uninformative about that ratio.
- **Verdict**: not a bug — the aggregation matches the legacy v1 trainer (`src/models/dreamer_v3_trainer.py:261-267`) exactly, so cross-codebase comparisons are sound. But for Phase 1's headline framing, `_total` should be either (a) split into `_zero` + `_pos` + `_neg` to make the bucket weights explicit, or (b) dropped in favor of just the `_pos`/`_neg` ratio. Worth a 1-line caveat in the anchor doc.

## Q6 — JAX-specific gotchas — ✅ clean

- **vmap axis correctness**: `train.py:831, 845, 850` use `jax.vmap(model)(imag_flat)` where `imag_flat` has shape `[H_plus_1 * BT, latent_dim]`. The vmap is over axis 0 — the flattened time-batch axis. Outputs are reshaped back to `[H+1, BT, ...]`. No leading axis collision with the H+1 / BT axes. ✓
- **PRNG threading in the actor's stochastic action sampling inside imagination**: `agent.py:1786` splits `key, k_rssm, k_act = jax.random.split(key, 3)` per imagination step (Python `for` loop, H_plus_1 iterations). Each `k_act` then feeds `actor(new_latent, k_act)` which calls `jax.random.gumbel(key, shape=logits.shape)` at `agent.py:1539` — `logits.shape = [BT, action_dim]`, so the gumbel call produces `BT * action_dim` independent noise samples from one key (correct — different positions in the output array get different streams). The reward head does NOT consume a PRNG key — it is deterministic conditional on the latent. v1's H1 resample bug does not transfer to the reward head because the reward head is not stochastic.
- **`imagined_actions` identity rule**: `train.py:826` extracts `imag_outputs["imagined_actions"]` from the rollout (the actions Gumbel-sampled at rollout time); `train.py:889` stop-gradients them; `train.py:891-915` recomputes log-probs via `actor.forward_logits` (deterministic, no PRNG). This matches the v1 fix [[20260518_1512_reinforce_resampling_bug_imag_action_threading]]. The reward head plays no analogous role (no resample to worry about).
- **`@struct.dataclass` mutation**: `MomentsState` (utils.py:174) is a `@flax.struct.dataclass`. `moments_update` (utils.py:217) constructs a NEW `MomentsState(low=..., high=...)` and returns it — no in-place mutation. The training loop receives `new_moments` from `one_train_step` and passes it back next call. ✓
- **JIT recompile triggers**: `make_train_step` captures `horizon` as a Python int in the closure (train.py:646); `H_plus_1 = horizon + 1` is also a Python int used to bound a Python `for h in range(H_plus_1)` loop inside `actor_loss_fn` (train.py:903) and `WorldModel.imagine` (agent.py:1785). The loop is unrolled at trace time. If `horizon` changes between train and imagination paths, the JIT cache invalidates — but the code never does this. The `bins=255` is fixed (config-supplied at build_agent time). ✓
- **Other**: I did not find a `jnp.where` masking a traced value where a static one would be safer. No global state, no impure dependencies.

## Recommended next checks (cannot cover from static review)

1. **Phase 2b offline-WM diagnostic** must report per-horizon `model_reward_mae` (Q4 closure). Specifically, splits at h ∈ {1, 5, 10, 20, 50} alongside `_pos` / `_neg` decomposition. If v2's shape mirrors Z2's 0.18 → 3.05, Q4 is closed as "inherited upstream". If not, dig into latent-drift metrics (`Imagined/state_drift` KL between imagined prior at step h and posterior at step h on the same trajectory).
2. **Reward-mass histogram audit** for one 10×10 hypervigilance run: count `N_zero`, `N_pos`, `N_neg` in the replay buffer at end of training. If the near-zero bucket is >95% of mass, the `_total` metric is mostly tracking near-zero MAE and should be flagged as misleading in WandB dashboards.
3. **Per-bin twohot histogram** at end of training: which bins are the reward head learning probability mass on? If the head puts most mass on bins 125-130 (symlog ≈ 0) regardless of input latent, the head has collapsed to "predict zero" and the negative tail is being ignored at training time — a class-imbalance-driven failure mode that Q3 cannot detect (uniform sampling is correct but uniform sampling × highly imbalanced support gives an MLE optimum near zero for the unconditional bias term).
4. **Two-term critic loss target leak check**: `compute_critic_loss` (train.py:310-322) passes raw `lambda_values` and `target_critic_values`. Both are stop-gradient'd. But the `target_critic_values` are themselves computed from the live posterior latent (train.py:962-965) — if there's an off-by-one between which timestep's latent feeds the target critic vs. the online critic, the EMA self-reg term would systematically bias the critic. From my reading both critics consume `sg_latents_h = imagined_latents[:-1]` (same slice), so this is fine — but it's worth a unit test asserting `target_critic(sg_latents_h)` and `critic(sg_latents_h)` are called on identical inputs.

## Cross-links

- Phase 1 anchor: [`docs/experiments/active/dreamer_srl_v2/REWARD_HEAD_ASYMMETRY_ANALYSIS.md`](../experiments/active/dreamer_srl_v2/REWARD_HEAD_ASYMMETRY_ANALYSIS.md)
- Phase 2b offline-WM diagnostic plan (forward reference): [`docs/develop/active/dreamer_srl_v2/offline_wm_diagnostic_port_plan.md`](../develop/active/dreamer_srl_v2/offline_wm_diagnostic_port_plan.md)
- Settled insights cited: [[20260518_1511_dreamer_srl_v2_parity_pass_outperform]], [[20260518_1512_reinforce_resampling_bug_imag_action_threading]], [[20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry]], [[20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt]], [[20260509_1534_wm_reward_head_localized_failure_a1]]
- Prior reviews for cross-context: `docs/reviews/dreamer_srl_v2_cp5_imagined_returns_review.md` (§P1: live critic in imagination, same pattern verified here for reward head), `docs/reviews/dreamer_srl_v2_cp3_actor_objective_review.md` (§A1+A2: action threading discipline that Q6 confirmed extends correctly here)

## Files audited

- `src/algorithms/dreamer_srl/loss.py` (Q1, Q2)
- `src/algorithms/dreamer_srl/train.py` (Q2, Q3, Q4, Q5, Q6)
- `src/algorithms/dreamer_srl/agent.py` (Q4, Q6)
- `src/algorithms/dreamer_srl/buffers.py` (Q3)
- `src/algorithms/dreamer_srl/utils.py` (Q1, Q6)
- `src/algorithms/dreamer_srl/dreamer_srl_main.py` (Q6 config path)

Reviewed by: `code-reviewer` (delegated; audit transcribed by top-level Claude into this file).
