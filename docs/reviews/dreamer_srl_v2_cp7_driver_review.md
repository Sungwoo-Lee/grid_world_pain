---
title: "v2-CP7 — dreamer-srl driver re-audit (H2 hypothesis-locus)"
topic: dreamer
status: active
created: 2026-05-14
last_updated: 2026-05-14
---

# v2-CP7 — dreamer-srl driver re-audit

## Verdict (plain-language entry point)

**What this review is.** A fresh line-by-line re-audit of the dreamer-srl training-loop driver
([`src/algorithms/dreamer_srl/dreamer_srl_main.py`](../../src/algorithms/dreamer_srl/dreamer_srl_main.py))
against the canonical sheeprl driver
([`vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L361-L765`](../../vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py)),
performed without trusting any v1 review under `docs/reviews/dreamer_srl_v3_cp9_*`.
The driver is the **H2 hypothesis surface** in the v2 plan — H2 says that the parity-failed
3-seed run (collapsed to the random-policy floor) is rooted in `is_first` flag mis-propagation,
prefill contamination, or the boundary debt-repayment burst inherited from v1 deviation D-014.
This memo is the v2 CP7 deliverable; v2-CP2 (grad-parity methodology) and v2-CP8 (wrapper
re-audit) run in parallel.

**Headline verdict.** **❌ FAIL — substantive H2-class divergence found.**

The driver is missing sheeprl's **second buffer write at episode boundaries** (the `reset_data`
row written into the per-env-subset of done envs at `dreamer_v3.py:L641-L650`). Sheeprl writes
TWO rows when an episode ends: row `t` with the action that produced the done, then a SECOND
row at the same per-env-subset positions carrying the **real terminal observation** (from
`real_next_obs[dones_idxes]`) with `is_first=0`. The dreamer-srl driver does not do this — it
overwrites the done envs' next obs in place via a manual `jax_reset` (`dreamer_srl_main.py:L470-L481`)
and writes only the **new-episode start obs** on the next iteration as `is_first=1`. As a
consequence the buffer **never sees the terminal observation** of any episode; the dynamic-learning
sequence-sampler therefore trains the WM to predict a post-reset obs as if it were the natural
successor of the action that caused the termination. On a 1024-prefill, 1024-grad-step burst
(D-014), this contamination is amplified into the very first wave of WM updates.

The H2 hypothesis is now elevated from MED to **HIGH confidence**: there is a real, structural,
substantive bug at the driver-buffer boundary, distinct from the substrate-class deviations
already logged.

Two additional P-class findings (P2, P3) and three F-findings (F1–F3) are detailed below. The
D-014 boundary-burst concern raised in the brief is a **real overtraining hazard** on the
prefill buffer (P3), independent of P1.

---

## Findings table

| ID | Severity | Site (dreamer-srl) | Site (sheeprl) | Issue | Suggested fix |
|---|---|---|---|---|---|
| **P1** | 🔴 **blocker** | `dreamer_srl_main.py:L443-L483` | `dreamer_v3.py:L639-L650` | **Missing second buffer write at done-boundaries.** Sheeprl writes a `reset_data` row containing `real_next_obs[dones_idxes]` with `is_first=0` AFTER the env step that produced the done. The dreamer-srl driver writes no such second row — instead it overwrites `next_obs` in place via a manual `jax_reset` (L468-L481) and only writes the post-reset obs on the NEXT iteration as `is_first=1`. The buffer therefore never contains the terminal observation, and the `is_first=0` "this is the env's natural response to the done-causing action" row is missing. This corrupts the dynamic-learning sequence by gluing the post-reset obs onto the action that caused termination. | Port sheeprl's two-write pattern. Requires extending `buffers.SequentialReplayBuffer.add` to accept an optional `indices: list[int]` parameter that writes only the named per-env slots (matching sheeprl `rb.add(reset_data, dones_idxes, ...)` at `dreamer_v3.py:L650`). Then after env step, when `dones_idxes` is non-empty, write a second row carrying `real_next_obs[dones_idxes]` + `is_first=0` into only those env slots. |
| **P2** | 🔴 **blocker** | `dreamer_srl_main.py:L420-L432` | `dreamer_v3.py:L589-L592, L635-L637` | **`terminated` and `truncated` are conflated.** The dreamer-srl driver pulls `dones_jax` from `env.step` and assigns the SAME tensor to both `terminated_np` and `truncated_np` (L430-L431, comment at L425-L429 explicitly says "we treat done as terminated for CP9 ... fine because food-only has no death events"). Sheeprl reads `terminated` and `truncated` as separate signals from gym (`dreamer_v3.py:L589`), writes them separately into the buffer, and uses **only `terminated`** at `dreamer_v3.py:L247` for the §S5 true-continue splice (`true_continue = (1 - data["terminated"])`). On a food-only NoPred substrate every `done` is a max-steps truncation, NOT a real termination — so the buffer's `terminated` column should be all zeros, but in dreamer-srl it is 1.0 at every episode boundary. Consequence: at every done boundary the §S5 true-continue splice sees `(1 - 1) = 0` and tells the WM "this episode genuinely ended" — but the agent didn't die, it just hit the step limit. The λ-return at that step's bootstrap is then zeroed instead of carrying the value estimate forward, breaking value-target learning at every episode boundary. This is a hard bug on the food-only substrate that **directly suppresses the value signal** during training. | Read `terminated` separately from `truncated` from the env (the env exposes `states.terminated` per the comment at L427). On food-only: `terminated = states.terminated` (all zeros), `truncated = dones & ~terminated` (all dones). Then the §S5 splice consumes a zero `terminated` column and correctly carries the value bootstrap across episode boundaries. |
| **P3** | 🔴 **blocker** | `dreamer_srl_main.py:L496-L501` | `dreamer_v3.py:L660-L661` (and D-014 entry) | **D-014 boundary burst overtrains on sparse prefill buffer.** D-014 was approved at v1-CP9b as "substrate-class" with the claim that long-run replay ratio is identical and only boundary debt distribution differs. The brief flags this concern empirically: at iter `learning_starts=1024` with `replay_ratio=1`, the Ratio class returns 1024 grad steps in one shot. The buffer at that moment contains exactly 1024 transitions per env (`buffer._pos == 1024`). Each grad step samples `batch_size=16` sequences of `seq_len=64` — i.e. `16 × 64 = 1024` transitions, the entire buffer's worth. With `num_envs=1` this means **every grad step samples nearly the same data** — the prefill buffer is exhausted in one batch, and all 1024 grad steps therefore train on essentially the same 1024 transitions with only random offsets / env-indices reshuffled. This is textbook overtraining on the prefill data. Sheeprl's smeared variant fires the same 1024 grad steps spread across iters `[learning_starts, 2*learning_starts-1]` — so by the time the 512th grad step fires, the buffer has grown to 1024+512 = 1536 entries, giving 1.5× the sampling diversity. The v1 approval ("long-run replay ratio is identical") elided the per-batch-diversity argument. | Port sheeprl's `ratio_steps = policy_step - prefill_steps * policy_steps_per_iter` formula at the train-gate. This requires computing `prefill_steps = learning_starts - int(learning_starts > 0)` (sheeprl `dreamer_v3.py:L511`) and subtracting it from `policy_step` before passing to the `Ratio` scheduler. Reverts D-014 to the smeared sheeprl form. |
| **F1** | 🟡 concern | `dreamer_srl_main.py:L448-L452` | `dreamer_v3.py:L617` | **`Game/ep_len_avg` is per-episode raw, not a `MeanMetric` aggregate.** Confirmed: the driver logs the raw per-episode `ep_len` under the WandB key `Game/ep_len_avg` at every done event (L451), not a windowed mean. Sheeprl uses a `torchmetrics.MeanMetric` aggregator that windows over `metric.log_every` (`dreamer_v3.py:L615-L617`). The WandB scalar in dreamer-srl is therefore the *latest* episode-end value at each log step, not a moving average. The v1 parity verdict ("collapsed to random-policy floor") is mathematically still valid because the FLOOR signal is unmistakable, but the post-hoc reading of "is the curve trending up?" was misled. | Wrap the `ep_len` logging in a `MeanMetric`-equivalent windowed aggregator (e.g. accumulate in a `collections.deque(maxlen=window)` and log the mean). |
| **F2** | 🟡 concern | `dreamer_srl_main.py:L501` | `dreamer_v3.py:L663` | **Train-gate carries an extra `buffer._pos >= seq_len` guard not present in sheeprl.** Sheeprl gates only on `iter_num >= learning_starts and ratio(ratio_steps) > 0`. dreamer-srl adds `and buffer._pos >= seq_len`, which is a defensive check — but with `learning_starts=1024` and `seq_len=64`, the buffer is guaranteed to have `_pos >= seq_len` at iter `learning_starts`, so the guard is inactive on the parity config and benign. Worth flagging because it could mask a buffer-fill bug at `learning_starts=0` configurations like CP9's D-012. | Either remove the guard (matching sheeprl) or assert `buffer._pos >= seq_len` at the gate to surface fill-rate bugs. |
| **F3** | 🟡 concern | `dreamer_srl_main.py:L104-L152` (`Player.get_actions`) | `agent.py:L596-L691` (`PlayerDV3.get_actions`) | **`is_first` is passed to Player.get_actions every step but the value sourced from `is_first_next` is only set at done-boundaries.** This is correct in steady state, but at the very first iteration of training (post-prefill, `iter_num == learning_starts + 1`), the player's first call reads the `is_first` array set at L359 (`np.ones((num_envs, 1))`). That's correct — but during prefill (`iter_num <= learning_starts`) the player is bypassed entirely (uniform-random branch at L401-L408), so `is_first` is consumed by the player for the FIRST time at iter `learning_starts + 1` carrying whatever value was last assigned. On an episode-no-done iteration this is `is_first_next = zeros` (L436), which is correct for a mid-episode step. On an episode-with-done iteration this is `is_first_next` with 1s at the done indices (L460), also correct. But there is no guarantee that iter `learning_starts + 1` corresponds to a known episode-boundary state. Worth a runtime assertion that the `is_first` flag handed to the player at the post-prefill transition matches the buffer's `is_first` row at `buffer._pos - 1`. | Add an integration test: at iter `learning_starts + 1`, assert `is_first[i] == step_data["is_first"][:, i]` for every env i. |
| **Nit** | 🟢 nit | `dreamer_srl_main.py:L425-L429` | n/a | The comment at L427 ("we treat done as terminated for CP9 ...") is now misleading — the parity-launch is at `learning_starts=1024`, not the CP9 smoke. The comment should be removed or updated. | Update or remove the stale CP9 comment. |
| **Nit** | 🟢 nit | `dreamer_srl_main.py:L498` | `dreamer_v3.py:L661` | The inline comment at L387-L399 documents the D-014 boundary behaviour but does not explicitly mention the overtraining-on-sparse-buffer hazard surfaced in P3. | Add a one-line forward-pointer to P3 of this review. |

---

## Conventions audit checklist

Standard JAX/Flax conventions are out of scope for a driver — the driver is intentionally
NumPy-heavy on the env side, JAX-heavy on the train-step side. The checklist is adapted to the
sheeprl-port surface.

| Convention | Status | Notes |
|---|---|---|
| `is_first` written into buffer at row AFTER done (sheeprl pattern) | ❌ | P1: dreamer-srl writes `is_first=1` on the next iteration but is missing the intermediate `reset_data` write at the done boundary. |
| Random-action prefill matches sheeprl (uniform sample from action space) | ✅ | Verified: `dreamer_srl_main.py:L401-L408` uses `jax.random.randint(0, action_dim)` then one-hot encodes; matches `dreamer_v3.py:L563-L571`'s `np.array(envs.action_space.sample())` + one-hot. |
| Actor NOT called during prefill | ✅ | Confirmed: prefill branch at L401 bypasses `player.get_actions`. No PRNG-stream drift (the player's recurrent state is untouched during prefill; `init_states` is called once at startup and again only on per-env dones). |
| Prefill transitions added to buffer | ✅ | Confirmed: `buffer.add(step_data, validate_args=False)` at L415 runs in every iter regardless of the prefill gate. |
| WM observe/loss NOT called during prefill | ✅ | Confirmed: train-gate `if iter_num >= learning_starts` at L496 fires for the first time at iter `learning_starts` (NOT `learning_starts + 1`), matching sheeprl L660. (Both substrates have an off-by-one with the "<= learning_starts" prefill gate at L401 — the iter where `iter_num == learning_starts` runs BOTH the random-action prefill AND the first train step. This is faithful to sheeprl.) |
| Train-gate uses `>=` (not `>`) | ✅ | L496 matches sheeprl L660. |
| Polyak update fires BEFORE train() | ✅ | Confirmed: L513-L524 runs BEFORE the `train_step(...)` call at L537-L541. Initial `tau=1` at `cumulative_grad_steps == 0` matches sheeprl L678 (`tau = 1 if cumulative_per_rank_gradient_steps == 0 else cfg.algo.critic.tau`). |
| Polyak EMA formula matches sheeprl | ✅ | dreamer-srl: `new_target = (1.0 - tau) * target + tau * online` (L521). Sheeprl: `tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)` (L680). Algebraically identical. |
| Replay-ratio (smeared vs burst) at boundary | ❌ | P3: dreamer-srl's D-014 burst overtrains on sparse prefill buffer; sheeprl's smear is materially different. |
| `terminated` vs `truncated` semantics | ❌ | P2: dreamer-srl conflates the two; corrupts §S5 true-continue splice. |
| `ep_len_avg` aggregation matches sheeprl | ❌ | F1: dreamer-srl logs raw per-episode, sheeprl uses `MeanMetric`. |
| Actor `_uniform_mix` applied before sampling | ✅ | Confirmed: `agent.py:L1517` applies `_uniform_mix` to raw logits before sampling at L1521; matches sheeprl `agent.py:L832`. |
| Actor uses straight-through gradient estimator | ✅ | Confirmed: `agent.py:L1521-L1526` implements `hard - sg(soft) + soft` straight-through; equivalent to PyTorch's `OneHotCategoricalStraightThrough.rsample()` at sheeprl `agent.py:L834`. |
| `sg(action)` before `log_prob(action)` in actor forward | ✅ | Confirmed: `agent.py:L1535` applies `jax.lax.stop_gradient(actions)` before the `log_prob`-equivalent computation at L1533-L1535. (§S7 sg(action) discipline IS satisfied in the actor forward path. H1 lives elsewhere — the actor objective construction in `train.py:compute_actor_objective` is v2-CP3's surface, not v2-CP7's.) |
| Mandatory config reads via `get_mandatory` | ✅ | All 22 reads at `dreamer_srl_main.py:L197-L227` use `agent_cfg.get_mandatory(...)`; no fallback defaults. |
| PRNG key threading (split before use, no key reuse) | ✅ | Confirmed: `key, k_player = jax.random.split(key)` (L400), `key, k_autoreset = jax.random.split(key)` (L468), `key, k_train = jax.random.split(key)` (L536). No double-use of the same sub-key. |
| Pure-functional pytree updates | ⚠ | The `Ratio`, `Player`, and `moments_init`-returned state are mutated in-place at the Python level. This is consistent with sheeprl's pattern and falls inside the substrate-class boundary (D-001 / D-011); not a finding. |

---

## H2 surface conclusion

**H2 is confirmed at HIGH confidence.** Two of the three H2 sub-hypotheses are structurally
substantiated by the driver code:

1. **`is_first` propagation IS wrong** — but not in the way the brief anticipated. The flag
   itself is set correctly on the iteration AFTER a done (matches sheeprl's `step_data["is_first"][:, dones_idxes] = 1.0`
   pattern at L656). The bug is the **missing second buffer write** — sheeprl's `reset_data` row carrying
   the real terminal obs with `is_first=0` (`dreamer_v3.py:L649-L650`) has no analog in dreamer-srl,
   so the buffer's sequence at every episode boundary glues `(obs_t, action_t, reward_t)` directly to
   `(obs_{reset+1}, action_{reset+1}, reward_{reset+1})` without the intervening
   `(real_next_obs_t, action_zero, reward_t, is_first=0)` row. The RSSM's dynamic-learning at this
   boundary therefore sees a corrupted (action_t → obs_{reset+1}) transition, which the WM is
   forced to model. **(P1)**

2. **`terminated` vs `truncated` IS conflated** — and this is independent of P1 but compounds it.
   On a food-only NoPred substrate, every `done` is a truncation (max-steps), but the driver writes
   `terminated=1.0` at every done. Through the §S5 true-continue splice
   (`dreamer_v3.py:L247`: `true_continue = (1 - data["terminated"])`), the value-target's bootstrap
   is zeroed at every episode boundary — exactly the steps where value learning matters most.
   This systematically suppresses value-signal propagation and is **plausibly sufficient on its
   own to drive the policy to the random floor** (the actor's REINFORCE term consumes `advantage =
   λ-return - baseline`; if λ-return is zeroed at every boundary, advantage is dominated by the
   `-baseline` term and the actor's gradient is decorrelated from reward). **(P2)**

3. **D-014 boundary burst** — the v1 approval rationale ("long-run replay ratio is identical")
   missed the per-batch sampling diversity argument. On a 1024-prefill buffer with 1024 grad steps
   in one burst, every grad step samples nearly the entire buffer. The 1024 grad steps train on
   essentially the same data, overfitting the prefill distribution before any post-prefill data
   has been collected. Sheeprl's smear gives 1.5× the buffer growth between the first and last of
   the 1024 grad steps, materially changing the per-batch sampling distribution.
   **(P3)**

Any one of P1, P2, P3 is independently a candidate root cause for the v1 parity failure. They are
mutually compounding: P2's value-signal suppression at boundaries is amplified by P1's corrupted
boundary transitions, and P3's overtraining ensures the WM internalises both corruptions before
any clean signal arrives.

The "actor straight-through + `sg(action)`" sub-hypothesis under H1 is **not the driver's
problem** — `agent.py:Actor.__call__` correctly implements `hard - sg(soft) + soft` with
`sg(actions)` before `log_prob`. H1's `_uniform_mix` and entropy/REINFORCE concerns live in
`train.py:compute_actor_objective` (v2-CP3's surface).

---

## Conclusion

**❌ FAIL.** Three P-class findings explain v1's parity failure: a missing `reset_data` buffer
write at episode boundaries (P1), a `terminated`/`truncated` conflation that zeros the value
bootstrap at every episode end on food-only (P2), and the D-014 boundary burst overtraining on
the sparse prefill buffer (P3). H2 is elevated from MED to HIGH confidence. P1 and P2 are both
**structural** bugs (not substrate-class drift); P3 is a substrate-class deviation that the v1
approval rationale failed to fully evaluate. P1 requires extending `buffers.SequentialReplayBuffer.add`
to accept a per-env-index parameter; P2 requires reading `terminated` separately from `truncated`
from the env; P3 requires porting sheeprl's `prefill_steps` subtraction at the train-gate.

Reviewed by: code-reviewer
