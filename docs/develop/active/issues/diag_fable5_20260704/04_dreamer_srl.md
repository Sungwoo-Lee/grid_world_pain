---
title: "Independent correctness diagnosis — dreamer_srl algorithm package"
topic: issues
status: active
created: 2026-07-04
last_updated: 2026-07-04
---

# Dreamer-SRL correctness diagnosis (independent bug hunt, 2026-07-04)

## What this document is (plain-language entry point)

This is an independent, line-by-line correctness audit of the project's from-scratch
JAX/Flax-NNX port of the DreamerV3 world-model agent (the "dreamer_srl" package under
`src/algorithms/dreamer_srl/`). The port claims line-for-line fidelity to a vendored
PyTorch reference implementation (sheeprl), so the audit compared every training-relevant
code path against that reference: how experience transitions are stored and sampled from
the replay buffer, how the world model / actor / critic losses are computed, how the
imagination rollout is seeded, how random-number keys are threaded, and whether evaluation
preprocesses observations the same way training does. Previously-found-and-fixed bugs
(the action-resampling bug, the recompile storm, the per-step CPU→GPU upload) were checked
for regression, not re-reported.

**Headline finding:** the port drops one small but load-bearing block from the reference —
when an episode ends, the reference zeroes out the reward and "episode ended" flags before
writing the next episode's first row into the replay buffer; the port does not. As a result,
every episode's *first* stored step carries the *previous* episode's final reward and death
flag. This mislabels the reward and continue heads at every episode start and, after a
death, silently zeroes the learning signal for every imagined rollout that starts from an
episode-start state. Several smaller undeclared deviations (missing gradient clipping,
a halved observation loss, replay-ratio quantization, episode-metric bleed) are also
documented below. The core loss/return math itself checks out against the reference.

Scope audited: `agent.py`, `dreamer_srl_main.py`, `train.py`, `loss.py`, `buffers.py`,
`utils.py`, `checkpoint.py`, `eval.py` (all read in full), cross-checked against
`vendor/sheeprl/sheeprl/algos/dreamer_v3/` and the shipped configs in
`configs/models/dreamer_srl/`.

---

## Finding 1 — Episode-start buffer rows inherit the previous episode's terminal reward and death flag

**Severity: High · NEW**
`src/algorithms/dreamer_srl/dreamer_srl_main.py:1242-1262` (missing block; compare
`vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py` "Reset already inserted step data")

**What happens.** At a done boundary the driver correctly writes the second
("reset_data") buffer row with the true terminal observation, and sets
`step_data["is_first"][:, dones_idxes] = 1.0` for the next row. But sheeprl *also* zeroes
the staged `step_data["rewards"]`, `["terminated"]`, and `["truncated"]` for the done
envs at this point:

```python
# sheeprl dreamer_v3.py (vendored) — MISSING from the JAX port:
step_data["rewards"][:, dones_idxes]    = np.zeros_like(...)
step_data["terminated"][:, dones_idxes] = np.zeros_like(...)
step_data["truncated"][:, dones_idxes]  = np.zeros_like(...)
```

The port never does this (`dreamer_srl_main.py:1259-1262` touches only `is_first`).
`step_data["rewards"]`/`["terminated"]` were last set at lines 1179-1181 to the *terminal*
step's values, so at the top of the next iteration `buffer.add(step_data)` writes a row:
`(obs = fresh reset obs, is_first = 1, reward = previous episode's terminal reward,
terminated = previous episode's terminated flag)`.

**Concrete failure scenario.** One poisoned row per episode per env, three effects:

1. **Reward head bias at episode starts.** The world model's reward head is trained to
   predict the previous episode's terminal reward when looking at a fresh-reset latent.
   In stages with a death penalty, that is a large negative target attached to a benign
   starting state — systematically, every episode.
2. **Continue head mislabeled after deaths.** When the previous episode ended in death
   (`terminated=1`, i.e. `termination_reason >= 2`), the continue target
   `1 - terminated = 0` teaches the continue head that fresh episode-start states are
   terminal.
3. **Imagination from episode-start states is discount-zeroed after deaths.** The §S5
   true-continue splice (`train.py:472-479`) sets `continues[0] = 1 - batch["terminated"]`
   for every imagination rollout. Rollouts whose start row is a poisoned episode-start
   row get `true_continue = 0`, so `compute_discount` zeroes the *entire* imagined
   trajectory's weight — actor and critic learn nothing from post-death reset states,
   for the whole run.

In the current food-only configs all episode ends are truncations (`terminated=0`), so
only effect 1 fires (duplicated last-step reward). In any predator/starvation stage,
all three fire. Fix is the three-line zeroing block sheeprl has.

---

## Finding 2 — Observation reconstruction loss deviates from the reference in two ways (extra symlog + halved weight)

**Severity: Med · NEW (undeclared deviation)**
`src/algorithms/dreamer_srl/train.py:704-712`

**What happens.** The reference (`vendor/sheeprl/sheeprl/utils/distribution.py:177-192`,
`SymlogDistribution.log_prob`) computes `-(sum((decoder_output − symlog(target))²))` — the
raw decoder output is *itself* the symlog-space prediction, and there is **no ½ factor**.
The port computes:

```python
obs_log_prob = -0.5 * jnp.sum((_symlog(reconstructed_obs) - _symlog(obs_target)) ** 2, axis=-1)
```

Two departures: (a) `symlog` is applied to the decoder output too, i.e. the decoder is
trained as a raw-space predictor squashed at loss time rather than a symlog-space
predictor; (b) the extra `0.5` halves the observation loss relative to the KL, reward,
and continue terms in the summed world-model loss (`train.py:751`).

**Concrete failure scenario.** No crash and no wrong basin — the objective is
self-consistent (reconstructions are consumed nowhere else) — but the ELBO term balance
silently differs from the reference recipe the whole project is benchmarking against:
reconstruction is under-weighted 2×, and gradients through the decoder are additionally
scaled by symlog's slope. For this environment's ~[0,1]-normalized observations the
symlog is near-linear so (a) is minor; (b) is a uniform 2× down-weight. Neither is in
`docs/develop/active/dreamer_srl_v1/DEVIATION_LOG.md` (checked D-001…D-014), and the
grad-parity tests cover actor/critic paths, not this term. The in-file comment claims
"Normal(symlog(pred), 1)" — the reference is an MSE distribution, not a unit Normal;
the comment's model of the reference is what introduced both departures.

---

## Finding 3 — No gradient clipping on any of the three optimizers

**Severity: Med · NEW (undeclared deviation)**
`src/algorithms/dreamer_srl/dreamer_srl_main.py:637-639`

**What happens.** sheeprl clips gradients by global norm before every optimizer step:
world model at 1000, actor at 100, critic at 100
(`vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml:52,127,154`;
`dreamer_v3.py:193-197, 300-302, 320-324`). The port builds plain
`optax.adam(lr, eps=eps)` for all three — no `optax.clip_by_global_norm` anywhere in the
package, and no clipping keys in the dreamer_srl configs.

**Concrete failure scenario.** A single bad batch (e.g. a rare large-reward transition
landing near a two-hot bin edge, or an early KL spike) produces an unclipped gradient
step that can knock the actor or critic into a divergent region; the reference recipe
is explicitly protected against this. Silent stability deviation from the recipe being
parity-tracked; not logged in DEVIATION_LOG.

---

## Finding 4 — Replay-ratio remainder carry is dead code; effective ratio silently quantized

**Severity: Med (config-dependent; exact at current canonical configs) · NEW**
`src/algorithms/dreamer_srl/dreamer_srl_main.py:1043-1044, 1523-1530`

**What happens.** Fix 2 (constant scan length) replaces the Ratio scheduler's owed step
count with a fixed `_G = max(1, int(replay_ratio * num_envs))` and tracks the shortfall
in `_grad_step_remainder` — but the remainder is only ever accumulated
(`_grad_step_remainder += n_grad_steps - _G`); no code ever adds an extra scan step when
it reaches +1 or skips one at −1, despite the comment saying it does.

**Concrete failure scenario.** Whenever `replay_ratio * num_envs` is not an integer, the
executed replay ratio permanently deviates from the configured one: e.g.
`replay_ratio=0.5, num_envs=3` → 1 grad step per iteration executed vs 1.5 owed (−33%
forever, remainder grows unboundedly). With the canonical `num_envs=16` and
`replay_ratio` 1 or 0.5 the product is integral and the path is exact — which is why
this hasn't bitten yet. The `Params/effective_replay_ratio` log key makes drift
observable after the fact, but nothing corrects it.

---

## Finding 5 — Episode metrics bleed across the boundary (+1 step, + predecessor's terminal reward)

**Severity: Med (analysis/logging numbers only; training unaffected) · NEW**
`src/algorithms/dreamer_srl/dreamer_srl_main.py:1264-1266` vs `:1500-1501`

**What happens.** Inside the done block the driver resets `episode_lengths[i] = 0` and
`episode_rewards[i] = 0.0`. But later in the *same* iteration the unconditional

```python
episode_lengths += 1
episode_rewards += rewards.astype(np.float32)
```

re-counts the terminal step and re-adds the terminal reward — now credited to the *new*
episode.

**Concrete failure scenario.** For every episode after the first per env,
`Episode/Steps` in WandB overcounts true length by 1 and `Episode/Reward` includes the
previous episode's final-step reward (in death stages: the death penalty leaks into the
next episode's logged return). Cross-algorithm comparisons of these keys against rPPO
runs are skewed by exactly this amount. Fix: mask the increments (`~dones`) or move
them above the done block.

---

## Finding 6 — `learning_starts` interpreted as iterations, not env steps (num_envs× longer prefill than the reference)

**Severity: Low-Med (silent recipe drift, directionally benign) · NEW**
`src/algorithms/dreamer_srl/dreamer_srl_main.py:1110` (gate `iter_num <= learning_starts`)

**What happens.** sheeprl converts the config value to iterations:
`learning_starts = cfg.algo.learning_starts // (num_envs * world_size)`
(`dreamer_v3.py:508-510`). The port compares `iter_num` (one iteration = `num_envs` env
steps) against the raw config value. With `learning_starts: 1024` and `num_envs=16`, the
random-action prefill runs 1024 *iterations* = **16,384 env steps**, 16× the reference
semantics; the D-014 debt-repayment burst at the train gate is correspondingly 16× larger.
At `num_envs=1` behaviour matches. Not harmful to learning (more diverse seeding), but a
silent semantic difference from the sheeprl recipe and from what the config value claims,
and it burns budget on parity runs.

---

## Finding 7 — Train gate skips owed gradient steps while the ring buffer's write head is below seq_len

**Severity: Low · NEW**
`src/algorithms/dreamer_srl/dreamer_srl_main.py:1532`

**What happens.** The gate is `n_grad_steps > 0 and buffer._pos >= seq_len`. After the
ring buffer wraps, `_pos` restarts near 0 while `_full=True`; `sample()` itself handles
the full case correctly (`buffers.py:405-421`), but the gate doesn't check `_full`, so
up to `seq_len − 1` iterations after every wrap silently skip their gradient steps (and
the Ratio debt for them is consumed by the quantized path, so it is never repaid).
With `buffer.size: 1e6` and `seq_len: 64` this is ~0.006% of iterations — negligible
today, but the correct gate is `buffer._full or buffer._pos >= seq_len`.

---

## Finding 8 — `terminated = (termination_reason >= 2)` trusts a reason code that can fire without `done` (overeating quirk interaction)

**Severity: Low (currently inert — all dreamer configs set `overeating_death: false`) · NEW consumer-side note on a KNOWN env row**
`src/algorithms/dreamer_srl/dreamer_srl_main.py:1142-1144` · env side `src/environment/core.py:704-711`

**What happens.** The driver derives the death flag purely from
`infos['termination_reason'] >= 2`, independent of `dones`. The Known-Bugs registry
records that with `overeating_death: true` the env sets `termination_reason=3` without
setting `done=True`. Under that config the driver would write `terminated=1` rows
*mid-episode* (no reset_data row, no is_first follow-up): the continue head gets "done"
labels on live states and the §S5 splice discount-zeroes imagination from those rows.
Today every dreamer env config (and `configs/environment/default.yaml`) sets
`overeating_death: false`, and the `if params.overeating_death:` guard compiles the
reason-3 branch out — so this is a latent trap, not a live bug. If overeating death is
ever enabled, either the env quirk must be fixed first or the driver should derive
`terminated` as `dones & (reason >= 2)`.

---

## Finding 9 — Checkpoints save model weights only (KNOWN — scope confirmed)

**Severity: Med · KNOWN (A2 in the handoff; registry row "Dreamer-srl checkpoints drop the optimizer's momentum on save")**
`src/algorithms/dreamer_srl/checkpoint.py:84-96`

Scope confirmation from this audit: the checkpoint stores `nnx.state(module, nnx.Param)`
for world model / actor / critic / target critic, plus moments, PRNG key, and counters.
It omits **all three Adam optimizer states** (wm/actor/critic first+second moments),
the **Ratio scheduler state**, and the **replay buffer**. Additionally there is currently
**no restore path at all** in the driver (`save_checkpoint`'s own docstring notes restore
wiring is out of scope), so the practical blast radius today is limited to future
resume/continual work — matching the registry's assessment. No new severity change.

---

## Verified clean (checked explicitly, no issue found)

- **Two-hot encode/decode symmetry** (`loss.py:94-245`): bins stored in symlog space,
  `symlog` on targets in `log_prob`, `symexp` only at `mean`/`mode`; cross-weight
  interpolation matches sheeprl (the historical v1 bin-space bug has not recurred).
- **Lambda returns** (`utils.py:124-166`, `train.py:396-493`): bootstrap at horizon from
  `values[-1:]`, continuation masking via `continues[1:] * gamma`, §S5 true-continue
  splice, and §S6 `cumprod/gamma` discount all match the reference, including the
  `[1:]` / `[:-1]` slicing.
- **KL balance stop-gradients** (`train.py:743-748`): `sg(posterior)‖prior` for the
  dynamics term, `posterior‖sg(prior)` for the representation term; free-nats floor
  applied per-element before the mean. Correct.
- **Critic two-term loss + target-critic timing** (`train.py:227-324, 960-982`;
  `dreamer_srl_main.py:964-1021`): both NLL terms present, targets stop-gradient'd,
  Polyak update fires before the train step with tau=1 hard copy at step 0; the scan-path
  `jnp.where`-masked Polyak is equivalent to the legacy path.
- **Imagination start states** (`train.py:803-816`): posteriors and recurrent states are
  stop-gradient'd before imagination (matches sheeprl's `.detach()`); actor/critic losses
  run on `stop_gradient(imagined_latents)`.
- **v1 REINFORCE resampling bug not regressed** (`train.py:888-953`): actor loss
  recomputes logits via `forward_logits` on stop-gradient'd latents and pairs them with
  the stop-gradient'd *rollout* actions — no fresh sample, no PRNG in the loss.
- **PRNG threading**: per-step key splits in `observe`/`imagine`/`dynamic`, per-grad-step
  splits in the scan carry, main-loop key advanced everywhere; no key reuse found.
- **Action/observation off-by-one**: buffer rows store `(obs_t, a_t, r_t)` with `r_t` the
  arrival reward; `action_shift` (§S2) feeds `a_{t-1}` to the RSSM step at `t`. Correct.
- **Buffer sequence sampling** (`buffers.py:305-558`): full-buffer valid-start-index
  exclusion around the write head is correct; one env column per sequence; sequences may
  cross episode boundaries by design and `is_first` (§S1 force-set + §S4 reset) handles it.
- **Eval vs training preprocessing** (`eval.py`): both go through
  `get_observation(state, params)` with the same noise semantics (noise lives inside
  `get_observation`, `apply_noise=True` default in both paths); encoder symlog identical.
  Deterministic argmax at eval is intentional.
- **JIT shape stability**: `_scan_grad_steps` compiled once (graphdefs in closure, fixed
  `_G`-length scan, fixed-width masked resets in `Player.init_states` and
  `buffer.add(done_mask=...)`); no shape-varying inputs found on the hot path. The
  persistent-compile and Option-S fixes are intact.
- **Moments normalizer**: `max: 1.0` in configs matches the vendored sheeprl config
  (invscale floor = 1); EMA-then-use ordering matches.

---

## Verdict

The numerical core of the port — RSSM, losses, returns, stop-gradient placement, PRNG
discipline, buffer indexing — is faithful to the reference and the historical v1 bugs
have not regressed. The area is **not yet sound for result-bearing runs**, for one
reason above all: **Finding 1** (episode-start rows inheriting the previous episode's
terminal reward/death flag) is a true reference divergence that mislabels the reward and
continue heads at every episode boundary and suppresses behavior learning from
post-death start states; it should be fixed (three lines) before any further training,
and it is cheap to fix. Findings 2 and 3 are undeclared recipe deviations that should
either be fixed or ratified into the DEVIATION_LOG before parity claims are made;
Findings 4–7 are correctness papercuts worth batching into the same fix pass.

Reviewed by: code-reviewer (independent diagnosis, Fable 5, 2026-07-04)
