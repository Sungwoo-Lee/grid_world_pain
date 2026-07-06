---
title: "Independent correctness diagnosis — DreamerV3-NNX training stack (train.py path)"
topic: issues
status: active
created: 2026-07-04
last_updated: 2026-07-06
---

# DreamerV3-NNX training stack — independent bug hunt (2026-07-04)

## Purpose (plain-language entry point)

This document is an independent correctness audit of the **DreamerV3 world-model agent that
`train.py` trains** — the "learn a model of the world, then practice inside your own imagination"
algorithm — covering its loss functions, its imagination rollouts, its replay buffer, its data
collection loop, and its checkpointing. (The separate `dreamer_srl` package is audited by another
agent.) A prior project-wide audit already fixed two bugs here: the "will the episode continue?"
predictor no longer treats running out of time as dying, and the reward-encoding grid was widened
so a −100 death penalty is no longer silently squashed to −20. **Both of those fixes check out as
correct and complete** — I re-derived and numerically re-tested them.

The hunt did, however, find new problems. The most serious: **the world model is trained with each
action paired to the wrong timestep** — it learns "which observation goes with the action the agent
chose *after seeing it*" instead of "which observation *results from* the previous action" — a
one-step misalignment that quietly corrupts what imagination practises on. Second: the replay
buffer's capacity (1,000,000) is **not divisible by its sequence length (128)**, so once a long run
wraps the buffer, sampled training sequences begin splicing two different environments' trajectories
together mid-sequence with no boundary marker. Several medium findings follow (evaluation feeds the
network un-compressed observations; checkpoints omit optimizer/target-critic/normalizer state;
the agent's memory is never actually reset at episode boundaries during collection).

**Scope:** `src/models/dreamer_v3_trainer.py`, `src/models/dreamer_v3_nnx.py`,
`src/models/dreamer_v3_util.py`, `src/models/dreamer_v3_network.py`, and the DreamerV3 dispatch /
collection / checkpoint portions of `train.py`. Read against `docs/develop/active/issues/KNOWN_BUGS.md`
(2026-07-04) so known rows are not re-reported.

---

## Verification of prior fixes (requested)

### ✅ Continue-head timeout fix (commit `5b093bf`) — CORRECT and COMPLETE

- `compute_continue_target` (`src/models/dreamer_v3_trainer.py:54-67`) maps
  `termination_reason >= 2` → target 0.0 (real death: starvation/overeating/injury) and reason 1
  (timeout) → target 1.0. Matches env codes emitted at `src/environment/core.py:703-711`.
- The continue loss (`dreamer_v3_trainer.py:251-253`) uses **only** `term_reason`; the old
  `batch['terminal']` is now used only for a diagnostic probe metric (`:315-322`).
- `termination_reason` is threaded end-to-end: recorded per step in `collect_sequence`
  (`dreamer_v3_trainer.py:696`), flattened and stored on both GPU and CPU buffer paths
  (`train.py:1561, 1607`), carried through `ReplayBuffer.add_batch`/`sample`
  (`dreamer_v3_trainer.py:1003, 1049`), through all three mixture-sampling pools and their
  fallbacks (`:748, 769, 781, 795, 805, 931`), and into the CPU path (`:939`).
- Timeout still sets `done=True` for auto-reset and `is_first` on the next step, so the RSSM state
  reset behavior is unchanged. Imagination has no timeout concept, so no further masking is needed.
- **One dormant caveat** — see Finding 7 (overeating interaction). It does not affect any current
  config (`overeating_death: false` everywhere) but should be noted next to the fix.

### ✅ Reward two-hot layout fix (commits `f5df600` + `1703a4c`) — CORRECT and COMPLETE in this stack

- `to_twohot`/`from_twohot` (`src/models/dreamer_v3_util.py:19-101`) use the identical bin grid on
  encode and decode for either flag value; I numerically verified the round-trip with the project
  interpreter: under `paper_canonical_twohot_bins=True`, −100 → −100.00001 and +100 → +100.00001
  (float32 exact); under the legacy `False` layout the documented ±20 clipping reproduces.
- The trainer passes the flag explicitly at **all nine** encode/decode call sites
  (`dreamer_v3_trainer.py:241, 284, 390, 395, 413, 416, 439, 456, 461`), it is `get_mandatory` in
  the trainer (`:93, :105`), and live configs set it `true`
  (`configs/models/dreamer_v3/dreamer_v3.yaml:48`). The eval-side decode
  (`dreamer_v3_nnx.py:721`) reads the same flag from the agent config, which the trainer always
  populates. No layout drift found between any encode site and any decode site.

---

## Finding 1 — 🔴 High — World-model training pairs each action with the wrong timestep (train/inference/imagination convention mismatch)

**Files:** `src/models/dreamer_v3_trainer.py:199-219` (training scan), versus
`src/models/dreamer_v3_trainer.py:557,589` (inference) and `src/models/dreamer_v3_nnx.py:105-144`
(RSSM semantics). **NEW.**

**What happens.** `RSSM.step` advances the GRU with `concat(stoch_{t-1}, action)` *before* the
posterior reads `embed_t` (`dreamer_v3_nnx.py:121-135`) — so the `action` argument is the action
*leading into* step t, i.e. it must be **a_{t−1}**. The three call sites disagree on what is passed:

- **Inference / collection** (`get_action`, `dreamer_v3_trainer.py:557,589`): passes
  `prev_state['prev_action']` = **a_{t−1}**. Correct Hafner convention.
- **Imagination** (`imagine_step` in `scan_imag`): action sampled at the source state → **a_i**
  drives the transition to z_{i+1}. Correct convention.
- **World-model training** (`model_loss_fn` scan, `dreamer_v3_trainer.py:211`
  `env_inputs = (action, is_first)` — no shift anywhere): passes `batch['action'][t]` = the action
  chosen **at obs_t, after seeing it** (`collect_sequence` stores obs and the action chosen from it
  at the same index, `:673-675`; `train.py:1554` adds them unshifted).

Canonical implementations shift the action sequence right by one before the RSSM scan (sheeprl
prepends a zero action and drops the last; Hafner's `obs_step(prev_state, prev_action, embed, is_first)`
receives shifted actions). This repo does not.

**Consequences.**
1. The prior is trained to predict `z_t` *conditioned on the action the policy chose after seeing
   obs_t* — an inverse-dynamics leak: the action carries information about the observation it is
   supposed to predict (e.g. the model can "predict" a nearby predator from the fact that the agent
   fled). In imagination the action fed is the *source-state* action, so the learned
   leak-calibrated transition is applied under the wrong convention — action effects arrive one
   step late, and chosen actions steer imagined states toward "states where the policy picks this
   action" rather than "states this action causes".
2. Train/inference mismatch: at collection time the GRU receives a_{t−1} with embed_t, but was
   trained on a_t with embed_t — the filtering distribution the actor acts on is systematically
   off-distribution relative to both training conventions.
3. The reward/continue heads only fit their same-index targets (r_t, death-by-a_t) because feat_t
   contains a_t through the GRU — meaning the correct fix must shift **actions, reward, terminal,
   and term_reason as a set** (Hafner alignment: at index t store a_{t−1}, the reward received
   *arriving* at obs_t, and whether obs_t is terminal), not just the action array.

**Concrete failure scenario.** World-model replay metrics (recon/reward/continue accuracy) look
excellent while imagined rollouts respond to the *previous* action; actor gradients reinforce
action-state correlations instead of causal effects. Symptom class: Dreamer learns slowly or
plateaus despite low model losses — consistent with this project's long-standing "Dreamer
underperforms rPPO" history.

**Suggested fix direction.** Shift actions right by one at the training scan (zero for the first
element and after `is_first`), realign reward/terminal/term_reason targets to arrival-at-obs
convention, and zero the action input when `is_first=1` in `RSSM.step` (it currently masks only
`deter`/`stoch`, `dreamer_v3_nnx.py:117-119`). This is a semantics change: old checkpoints and
old replay data are not comparable across the fix.

## Finding 2 — 🔴 High (long runs) — Replay buffer capacity not a multiple of sequence length: sampled sequences splice two envs after the first wrap

**Files:** `src/models/dreamer_v3_trainer.py:997,1013` (`add_batch` wraps at `% capacity`),
`:1035-1041` and `:741,762,774` (sampling assumes blocks aligned to multiples of `sequence_length`
from index 0); `configs/models/dreamer_v3/dreamer_v3.yaml:14` (`buffer_capacity: 1000000`) with
`sequence_length: 128`. **NEW.**

**What happens.** The buffer stores env-major 128-step runs and samples only at 128-aligned
offsets, relying on writes staying 128-aligned. Each `add_batch` call writes `num_envs × 128`
transitions (a multiple of 128), but 1,000,000 % 128 = **64**. On the first wrap, `idx` lands at
an offset ≡ 64 (mod 128) and every subsequent write grid is shifted half a block relative to the
fixed sampling grid. From then on, a sampled "sequence" from any rewritten region contains steps
64–127 of env A's chunk followed by steps 0–63 of env B's chunk — a hard teleport mid-sequence with
**no `is_first` marker**, which the RSSM trains through as if it were a real transition. Each
further wrap shifts by another 64, and the corrupted fraction of the buffer grows toward 100%.

Notably, the sibling positive buffer **is** rounded to a multiple of `sequence_length`
(`train.py:864`), showing the constraint was understood there but missed for the main buffer.

**Concrete failure scenario.** Any run longer than 1M env steps (typical multi-million-step runs
qualify) silently degrades: world-model dynamics get trained on spliced trajectories, recon/KL
losses tick up mid-run with no code change, and behavior learned from imagination inherits the
corrupted dynamics. Runs shorter than 1M env steps are unaffected; stage transitions reset
`idx=0` (`train.py:1268`) and restore alignment.

**Suggested fix direction.** Round `capacity` down to a multiple of `sequence_length` in
`ReplayBuffer.__init__` (one line), or validate and raise. Optionally also assert
`(num_envs*collect_interval) % sequence_length == 0` at setup (see Finding 6).

## Finding 3 — 🟡 Med — DreamerV3 checkpoints omit optimizer state, target critic, and return-normalizer state

**Files:** `train.py:2461-2471` (save), `train.py:1130-1140` (restore). **NEW** (sibling of the
KNOWN open row "dreamer_srl checkpoints drop the optimizer's momentum on save", which covers the
*other* dreamer package — this is the same defect class on the `train.py` DreamerV3 path, and
wider).

**What happens.** The checkpoint stores only `wm`/`actor`/`critic` `nnx.Param` state plus loop
counters. Not saved: the three Adam optimizer states (`model_opt`, `actor_opt`, `critic_opt`),
the EMA `target_critic`, and the `Moments` return-normalization EMA (`low`/`high`). The rPPO
branch by contrast saves its optimizer (`train.py:2453`). On `--load-checkpoint` resume, Adam
momenta restart from zero, the target critic restarts from a **fresh random network**, and the
return scale restarts from (0, 1) — value targets and advantage scaling are distorted for the
first few thousand gradient steps after every resume. Also: the restore is wrapped in a broad
`except Exception` that prints and **continues training from random weights** on failure
(`train.py:1198-1200`), and the Dreamer restore branch does no architecture check (the KNOWN
latent row "checkpoint restore may not map onto model" covers the Orbax-vs-NNX skew risk).

## Finding 4 — 🟡 Med — RSSM (and modulator) state is never actually reset at episode boundaries during collection: `get_action` hardcodes `is_first = 0`

**Files:** `src/models/dreamer_v3_trainer.py:561` (`is_first = jnp.zeros((B, 1))` inside
`get_action`), `:670` (`collect_sequence` dutifully sets `next_d_state['is_first'] = done` — but
`get_action` never reads it). **NEW.**

**What happens.** During data collection, when an episode ends and the env auto-resets, the
Dreamer belief state (`deter`, `stoch`, `prev_action`, and `mod_h` for modulated runs) carries
over into the new episode — the mask in `RSSM.step` (`dreamer_v3_nnx.py:117-119`) never fires
because `get_action` always passes zeros. The `is_first` flag is stored correctly in the buffer
(training *does* reset at boundaries), so this is a collect/train mismatch: the policy that
gathers data acts, at the start of every episode, on a belief state still containing the previous
episode (including its death), while training and imagination assume boundary-reset latents.
Stage transitions are unaffected (state explicitly re-initialized, `train.py:1292-1299`).
Data remains valid MDP transitions; the damage is off-distribution policy inputs and degraded
early-episode behavior/exploration. Fix: read `prev_state['is_first']` in `get_action` (it is
already maintained by both `collect_sequence` and `train.py`), and also reset `mod_h` under the
same mask for modulated runs.

## Finding 5 — 🟡 Med — Eval inference path (`DreamerV3Agent.__call__`) skips `symlog` on observations (and reuses `PRNGKey(0)` every step)

**Files:** `src/models/dreamer_v3_nnx.py:679-716` (no `symlog(x)` before encoder/modulator;
`key=None → PRNGKey(0)` at `:710-711`), driven by `src/utils/evaluation_core.py:35`
(`model(x, h)` — never passes a key) via `train.py:2495` (`model=trainer.agent`). **NEW.**

**What happens.** Training (`train_step`, `dreamer_v3_trainer.py:143`) and collection
(`get_action`, `:558`) always encode `symlog(obs)`. The evaluation entry point
`DreamerV3Agent.__call__` encodes the **raw** observation. Every during-training video/stats
evaluation of a Dreamer checkpoint therefore runs the encoder, modulator, and RSSM
out-of-distribution — mildly for small-magnitude channels (symlog(1)=0.69) but substantially for
larger-magnitude interoceptive/distance channels. Eval-only (training itself is unaffected), but
it systematically distorts every Dreamer eval metric, video, and any frozen-probe comparison run
through this path. Additionally the posterior latent is sampled with the identical `PRNGKey(0)` at
every step (deterministic, correlated bits), a minor secondary deviation. Fix: `x = symlog(x)` at
the top of `__call__`, and have `generic_inference` pass a key through.

## Finding 6 — 🟡 Med (latent config foot-gun) — `collect_interval` ≠ multiple of `sequence_length` silently violates the buffer's env-major contract; `collect_interval: 1` is advertised as "canonical"

**Files:** `src/models/dreamer_v3_trainer.py:980-993` (`add_batch` docstring: caller must pass
env-major, one env's `sequence_length` run per block), `train.py:1553` (env-major flatten of a
`(T=collect_interval, B)` chunk), `configs/models/dreamer_v3/dreamer_v3.yaml:6` (comment: "1 =
sheeprl-style (canonical)"). **NEW.**

**What happens.** With the live setting `collect_interval == sequence_length == 128`, each
128-block is exactly one env's contiguous run — correct. But nothing validates this. With
`collect_interval: 1` (explicitly suggested by the config comment), each `add_batch` writes one
step per env in env order, so a sampled 128-slot "sequence" is actually 128 *different envs' single
steps* interleaved — total sequence scramble, no error raised. Any non-multiple (e.g. 100) splices
envs mid-block similarly. Fix: assert `collect_interval % sequence_length == 0` (or build the
promised interleaving support) at setup.

## Finding 7 — 🟡 Med (dormant; interaction with the 5b093bf fix) — If `overeating_death` is ever enabled, the continue head trains "death" on steps where the episode does not end

**Files:** `src/models/dreamer_v3_trainer.py:54-67` (`term_reason >= 2` ⇒ continue-target 0) ×
`src/environment/core.py:707-708` (reason 3 set whenever `satiation >= max_satiation`) versus
`core.py:117-126` (`update_body`'s `done` has **no overeating branch** — the KNOWN latent env row
"over-eating never actually ends the episode"). **NEW interaction** (both parents known
individually; the combination is not recorded).

**What happens.** With `overeating_death: true`, every step where satiation sits at the cap emits
`termination_reason == 3` with `done == False` — potentially long stretches of steps. The old
continue target (`1 − terminal`) ignored these; the new `term_reason >= 2` target trains
continue = 0 ("death") on steps where the episode demonstrably continues, teaching imagination to
truncate value bootstraps whenever the agent is full. **Currently dormant**: every config in
`configs/` sets `overeating_death: false`. Flag it in the fix doc so enabling the knob doesn't
silently poison Dreamer's continue head; the real fix is the env-side one (make reason 3 actually
terminate, or gate the reason on death).

## Finding 8 — 🟢 Low — PRNG hygiene: same `rng` feeds both loss phases; same key reused for action-sample and latent-sample within a step

**Files:** `src/models/dreamer_v3_trainer.py:352` and `:516-520` (identical `rng` passed to
`model_loss_fn` and `behavior_loss_fn`); `:405-410` (`dist.sample(key)` and
`rssm.imagine_step(..., key)` share one key per imagination step); `:589-600` (`get_action`
reuses one key for posterior sampling and action sampling); `:758,785` (`key_recent`/`key_pos`
each used for two `randint` draws in the mixture sampler). **NEW.**

**What happens.** I verified with the project interpreter that `jax.random.split(k, 2)` is a
prefix of `split(k, N)` — so the model-scan key tree and the imagination key tree overlap (one
imagination step key *equals* the seed of the entire RSSM-scan key set), and within steps the
same bits drive two different categorical draws (correlated action/latent samples). None of this
breaks reproducibility (same seed → same run) and the practical bias is small, but it violates
the never-reuse-a-key rule and is cheap to fix (`rng_model, rng_behavior = split(rng)` in
`train_step`; one extra split per imagination step).

## Finding 9 — 🟢 Low — KL and actor log-probs/entropy computed on raw logits while sampling uses 1% unimix

**Files:** `src/models/dreamer_v3_util.py:108-114` (unimix applied to `probs` only),
`src/models/dreamer_v3_trainer.py:259-269` (KL from raw `log_softmax`), `:467-470` (actor
`log_probs`/entropy from raw logits). **NEW (deviation, not a defect per se).**

Canonical DreamerV3 defines the categorical distributions *with* unimix and computes KL,
log-probabilities, and entropy under that mixed distribution; here the sampled actions/latents
come from the unimix distribution but the gradients are computed under the raw one (slight
actor-gradient bias for near-deterministic policies; ST-estimator backward path also uses unimix
probs while KL uses raw logits). Low impact at unimix=0.01; worth aligning when Finding 1 is
touched.

## Finding 10 — 🟢 Low — Assorted nits

- **Target critic initialized as a fresh random net**, not a copy of the online critic
  (`dreamer_v3_trainer.py:108`); with `zero_init_reward_critic: true` both output zeros at init
  and the 0.02 EMA converges quickly — immaterial today, but becomes a real init mismatch if
  zero-init is ever turned off. (EMA timing itself is correct: update after each critic step,
  τ=0.02, `:530-533`.)
- **`src/models/dreamer_v3_network.py` is dead legacy code** (flax.linen, argmax "sampling" at
  `:35` mislabeled as ST sampling) — no importers anywhere in `src/`, `scripts/`, or `train.py`.
  Remove-on-touch candidate; do not confuse it with the live `dreamer_v3_nnx.py`.
- **CPU fallback path**: `_sample_mixture_cpu` uses `config.get(...)` with silent defaults
  (`dreamer_v3_trainer.py:894-897`), violating the no-fallback protocol, and uses the unseeded
  global `np.random` (non-reproducible); `train_multiple_cpu` re-creates its `@nnx.jit` closure
  every call → recompiles every iteration (`:870`). Live configs use `buffer_device: "gpu"`, so
  these bite only the fallback.
- **`self.step_count`** (`dreamer_v3_trainer.py:138`) is written once and never used.
- Replay sequences are not forced to start with `is_first=1` (sheeprl forces
  `is_first[:,0]=1`); here the zero init-carry makes the deter/stoch reset equivalent at t=0, so
  the only residue is the unzeroed action input — subsumed by Finding 1's fix.

---

## What was checked and found sound

- **KL balance & stop-gradients** (`dreamer_v3_trainer.py:259-278`): dyn = KL(sg(post)‖prior)×0.5,
  rep = KL(post‖sg(prior))×0.1 — exact paper form and scales.
- **Free bits**: `max(KL, 1.0)` applied per state after summing latent groups — matches Hafner.
- **Lambda returns** (`:28-51, 441-444`): H rewards, H+1 values with target-critic bootstrap at
  the horizon, `continues × γ` inside the recursion, reverse scan — correct; the death penalty is
  counted in the same step whose continue flag zeroes the future (reward and continue are read at
  the same imagined state), so no penalty is dropped at imagined deaths.
- **Discount weights** (`:450-452`): cumprod starting at 1, stop-gradient, applied to actor,
  entropy, and critic terms — canonical.
- **Return normalization** (`Moments`, util `:138-184`): 5th/95th percentile EMA, decay 0.99,
  `invscale = max(1, high−low)`, offsets cancel in the advantage; read outside `nnx.grad` before
  the update — matches sheeprl. One-update lag vs. Hafner's update-then-normalize is a benign
  ordering choice.
- **Actor loss**: REINFORCE on stored logits and stored actions (no resampling-at-loss-time bug),
  advantage stop-gradded, entropy bonus with configured scale.
- **symlog/symexp symmetry** and two-hot encode/decode symmetry — numerically verified.
- **`_scan_train_gpu` static/traced argument split** — buffer size, positive size, and write index
  are traced (no per-size recompile); capacities/slots/graphdef static — correct.
- **Stage transitions**: replay + positive buffers cleared (also restores block alignment), Dreamer
  recurrent state rebuilt (`train.py:1262-1299`).
- **Mixture sampler fallbacks** (empty positive buffer, short recent window) — shape-safe and
  gate on the right scalars.

## Verdict

**Not sound as-is for its scientific purpose, despite both prior fixes being correctly landed.**
The loss mathematics (KL balance, free bits, lambda returns, return normalization, EMA critic) are
faithful to DreamerV3, and the truncation-vs-death and two-hot fixes are complete. But Finding 1
(action/timestep misalignment in world-model training) is a structural correctness bug touching
every DreamerV3-NNX run ever trained on this path, and Finding 2 silently corrupts any run past
1M environment steps. Until those two are fixed, Dreamer-vs-rPPO comparisons and modulated-Dreamer
conclusions drawn from this stack carry a systematic confound. Findings 3–5 (checkpoint resume,
collection-time state reset, eval symlog) are worth fixing in the same pass; 6–10 are guards and
hygiene.

Reviewed by: code-reviewer (independent diagnosis pass, Fable 5, 2026-07-04)

---

## Fix plan pointer (2026-07-06)

Findings 1 (H6) and 2 (H7) now have an approved fix plan (work package WP-D):
[[fix_plan_h6h7_dreamer_v3_world_model]]. Note: the plan's analysis shows Finding 1's
sketched fix direction ("shift actions right at the training scan and realign
reward/terminal/term_reason") would erase all death events from world-model training in
this codebase (auto-reset never stores the terminal-arrival observation); the chosen fix
instead records the arrival observation in `collect_sequence`, which reproduces sheeprl's
post-shift pairing with the death row retained. Findings 3–10 remain open.
