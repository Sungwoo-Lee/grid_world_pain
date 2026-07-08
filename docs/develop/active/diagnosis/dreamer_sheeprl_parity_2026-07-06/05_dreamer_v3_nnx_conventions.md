---
title: "Parity/convention audit — DreamerV3-NNX stack (src/models) vs. the canonical DreamerV3 recipe"
topic: diagnosis
status: active
created: 2026-07-06
last_updated: 2026-07-08
---

# DreamerV3-NNX stack vs. the sheeprl/paper DreamerV3 recipe (area 5 of 5)

## Purpose (plain-language entry point)

This document audits the project's **in-house DreamerV3 implementation** — the world-model agent
that `train.py` trains, written directly in JAX/Flax-NNX under `src/models/` — against the
**canonical DreamerV3 recipe** (the vendored sheeprl reference implementation plus the Hafner 2023
paper). Unlike the separate `dreamer_srl` package (audited in area 4), **this stack never claimed
to be a faithful port of sheeprl**, so the question here is not "is the port faithful" but:
**where does this implementation deviate from the standard recipe, and which of those deviations
plausibly hurt training?** Every deviation found is put on the record and classified: matches the
recipe (parity), differs but provably does the same thing (equivalent by design), already recorded
in the bug registry (known-open), a **new undeclared deviation with plausible training impact**,
or cosmetic.

Headline: the loss *arithmetic* (KL balancing, free bits, two-hot reward coding, λ-returns,
return normalization, gradient clipping at 1000/100/100) matches the recipe, and the recently
landed fixes (arrival-observation pairing, buffer-capacity rounding, continue-head timeout
handling, two-hot bin grid) check out. But the audit found **six new undeclared deviations**, the
most serious being that the imagination phase **omits the reference's stop-gradients**: the
critic's loss and later-step action probabilities leak gradients back into the actor through the
sampled-action chain, so the actor is not trained with the pure REINFORCE gradient the recipe
prescribes (empirically confirmed with a minimal JAX probe). Also material: the observation
reconstruction loss is down-weighted by a factor of ≈ the observation dimension relative to the
recipe; the λ-return bootstrap uses the slow "target" critic where the recipe uses the live one
(and drops the recipe's slow-critic regularizer); and the `replay_ratio` knob, despite sharing
sheeprl's name, trains ~128× less per environment step than the same number means in sheeprl.

**Scope (ours):** `src/models/dreamer_v3_trainer.py`, `src/models/dreamer_v3_nnx.py`,
`src/models/dreamer_v3_util.py`, DreamerV3 sections of `train.py`, live config
`configs/models/dreamer_v3/dreamer_v3.yaml`.
**Reference:** `vendor/sheeprl/sheeprl/algos/dreamer_v3/{agent.py, dreamer_v3.py, loss.py,
utils.py}`, `vendor/sheeprl/sheeprl/utils/distribution.py`, and
`vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml`.
**Baseline:** [[05_dreamer_v3_nnx]] (2026-07-04 area report) and
`docs/develop/active/issues/KNOWN_BUGS.md`; fixes `bfb3780` (H6 arrival-obs + H7 capacity
rounding), `5b093bf` (continue head), `f5df600`+`1703a4c` (two-hot) are treated as landed and were
re-verified here.

---

## Classification table (complete)

Severity applies only to UNDECLARED rows. "ours" = `src/models/dreamer_v3_trainer.py` (T),
`src/models/dreamer_v3_nnx.py` (N), `src/models/dreamer_v3_util.py` (U), `train.py` (M);
"ref" = sheeprl `dreamer_v3.py` (D), `agent.py` (A), `loss.py` (L), `utils.py` (V), algo yaml (Y).

| # | Item | Ours | Ref | Class |
|---|------|------|-----|-------|
| P1 | Gradient clipping 1000 (WM) / 100 (actor) / 100 (critic) | T:113-136 | Y:52,127,154; D:193-199,300-303,320-326 | PARITY |
| P2 | Adam eps 1e-8 (WM) / 1e-5 (actor, critic); WM lr 1e-4 | T:117,125,133 | Y:111-114,140-143,157-160 | PARITY |
| P3 | KL balance: dyn 0.5 × KL(sg(post)‖prior), rep 0.1 × KL(post‖sg(prior)); regularizer 1.0 | T:261-283 | L:64-75 | PARITY (form; but see K3 for the logits it runs on) |
| P4 | Free bits: sum over 32 latent groups, then max(·, 1.0), then scale | T:277-281 | L:68-74 | PARITY |
| P5 | Reward loss: two-hot cross-entropy ≡ `TwoHotEncodingDistribution.log_prob`; 255 bins | T:245-247 | D:164; distribution.py:253-276 | PARITY |
| P6 | Two-hot bin grid (post-fix): linspace(−20, 20, 255) in symlog space, encode = decode | U:19-101; yaml:48 | distribution.py:237 | PARITY |
| P7 | Continue loss: BCE, weight 1.0; target = "not real death" (truncation ≠ death) | T:54-67,256-258 | D:167-168 (1 − terminated) | PARITY (ours strictly finer via `termination_reason`) |
| P8 | λ-return recursion and index pairing (returns for imag. states 0..H−1; rewards/continues at arrival states 1..H) | T:28-51,439-449 | V:66-77; D:251-256 | PARITY |
| P9 | Actor objective: REINFORCE with detached advantage + entropy bonus 3e-4 | T:465-478; yaml:40 | D:279-297; Y:119 | PARITY (form; see U1 for missing detaches) |
| P10 | Moments: decay 0.99, 5/95 percentiles, invscale = max(1, high−low); offset cancels in advantage | U:138-184; T:369-373,528 | V:40-63; Y:132-137 | PARITY |
| P11 | Horizon 15, λ 0.95, γ 0.997 (≈ paper 0.996997) | T:21-23 | Y:11-13 | PARITY |
| P12 | Imagination start states detached from WM training graph | T:361-367 | D:203-204 | PARITY |
| P13 | Output-head init: reward + critic zero-init (knob on); actor/continue heads not zero-init | N:502-511,585-591,636-640; yaml:44 | A:1170-1176 | PARITY |
| P14 | GRU reset gate applied to candidate (knob on); dual-LayerNorm GRU cell | N:47-70; yaml:54 | models.py LayerNormGRUCell | PARITY |
| P15 | Target-critic EMA τ = 0.02 per gradient step | T:534-538 | D:673-680; Y:151-152 | PARITY (see K6 for init; see U2 for what the target critic is *used* for) |
| P16 | symlog compression of observations at train + collection; symexp decode of scalar heads | T:143,563; U:6-17 | A:150; sheeprl symlog utils | PARITY (eval path exception = K2) |
| E1 | Action/observation pairing: ours stores the **arrival** observation with the action that produced it (H6 fix); sheeprl stores pre-step obs and shifts actions right by one at train time | T:618-731 (esp. 661-701), T:211-216 | D:82-104 (diagram), D:104 | EQUIVALENT-BY-DESIGN (same (a_{t−1}, o_t) pairing; residual gap = K1's "reset obs never stored") |
| E2 | No forced `is_first[:, 0] = 1` on sampled sequences | T:216-224 | D:100 | EQUIVALENT-BY-DESIGN **given zeros init carry** (N:97-103); stops being equivalent if a learnable initial state (C3) is ever added |
| E3 | Train-time RSSM masks deter/stoch at `is_first` but does **not** zero the action | N:117-121 | A:425 | EQUIVALENT-BY-DESIGN at train time (under the arrival convention the action at an `is_first` row is the new episode's own first action, never stale). Collection-time reset remains broken = K1 |
| E4 | Moments: use-previous-then-update ordering (one-step lag) | T:369-373,528 | D:276 (update-then-use) | EQUIVALENT-BY-DESIGN (benign lag; baseline report concurs) |
| K1 | Collection never resets belief state: `get_action` hardcodes `is_first = 0` and carries stale `prev_action` / `mod_h` through episode resets; reset observation never enters the buffer | T:566 (zeros), T:562, T:594-596; maintained-but-unread flag at T:697 | A:643-659 (`player.init_states` zeroes actions + resets latents per done env), D:657 | KNOWN-OPEN (registry row) — exact state confirmed; 2-line fix shape: in `get_action` read `prev_state['is_first']`, mask `prev_action` (and `mod_h`) with `(1 − is_first)` |
| K2 | Eval entry point `DreamerV3Agent.__call__` skips `symlog` on the observation and uses `PRNGKey(0)` every step (deterministic, correlated latent sampling); `is_first` hardcoded 0 | N:679-716 (key at 710-711, is_first at 715); driven via train.py:2485,2500 | A:150 (encoder symlogs), V:94-139 (test uses player) | KNOWN-OPEN — confirmed unchanged |
| K3 | Unimix placement: sampling uses 1% unimix but KL, actor log-probs, and entropy are computed on **raw** logits; sheeprl bakes unimix into the logits (prior, posterior, actor) so every downstream quantity uses the mixed distribution | U:108-114 (probs only); T:261-283 (KL raw), T:470-475 (actor raw) | A:437-449 (`_uniform_mix` on RSSM logits), A:832-845 (actor) | KNOWN-OPEN — confirmed unchanged |
| K4 | Checkpoint omits the 3 Adam states, the target critic, and the Moments EMA (also: `Ratio` internal counter, replay buffer) | M:2451-2461 (save), M:1130-1200 (restore) | D:741-755 saves world/actor/critic/target-critic + all 3 optimizers + moments + ratio + buffer | KNOWN-OPEN — confirmed unchanged |
| K5 | No validation that `collect_interval % sequence_length == 0`; config comment still advertises `collect_interval: 1` as "sheeprl-style (canonical)", which would interleave 128 different envs into one "sequence" | yaml:6; T:980-1002 (docstring contract only) | n/a (sheeprl buffer is per-env) | KNOWN-OPEN — confirmed unchanged |
| K6 | Target critic initialized as a fresh random network, never copied from the online critic; sheeprl does a τ=1 copy on the first update | T:108 | D:678 (`tau = 1 if cumulative_steps == 0`) | KNOWN-OPEN — confirmed unchanged (benign only while `zero_init_reward_critic: true`) |
| K7 | PRNG hygiene: same `rng` feeds model and behavior loss; one key reused for posterior-sample + action-sample in `get_action` | T:357+524 (same rng), T:594+605 | n/a | KNOWN (baseline Finding 8) — confirmed unchanged |
| U1 | **Missing stop-gradients in imagination** (3 sites): (i) critic loss consumes `rollouts['feat']` non-detached → critic-loss gradients flow into the **actor** through the straight-through action chain; (ii) actor input `feat` non-detached → later-step log-prob/entropy gradients backprop through imagined dynamics into earlier action logits (recipe uses pure REINFORCE for discrete, no dynamics backprop); (iii) `log_probs` uses the non-detached ST action sample → extra spurious ∇probs term | T:460 (i); T:379-381, 411-413 (ii); T:470-472 (iii) | D:307 (`critic(imagined_trajectories.detach())`), D:219,240,273 (`actor(...detach())`), D:286 (`log_prob(imgnd_act.detach())`) | **UNDECLARED — High** (actor gradient ≠ recipe's; empirically probed, §U1 below) |
| U2 | λ-return bootstrap values come from the **target** critic; recipe uses the **online** critic for all imagination values and instead adds a slow-critic **regularizer term** to the critic loss (`− log_prob(target_values)`), which ours lacks entirely | T:400,421,444 (target_critic in rollout + v_start); T:459-463 (no regularizer) | D:244 (online critic), D:307-316 (regularizer) | **UNDECLARED — Med** (DreamerV2-style value learning; slower/different value propagation) |
| U3 | Observation loss is `mean` over feature dims; recipe **sums** over event dims (symlog-MSE log-prob) → reconstruction term under-weighted by ≈ obs_dim (~40×) relative to reward/continue/KL | T:242 | L:61; distribution.py SymlogDistribution (sum over event dims) | **UNDECLARED — Med** (world-model loss balance materially off-recipe) |
| U4 | `replay_ratio` semantics: same knob name as sheeprl but counts gradient steps per **sequence** (`global_step // num_steps`), i.e. 1/128 of sheeprl's per-env-step meaning at `collect_interval=128`; and there is **no random-action prefill / learning_starts** — training starts after ~one collect with policy actions from the untrained net (gate: `buffer.size > max(2·batch, seq_len)` = 128 transitions) | M:1804-1809; yaml:5 (0.5) | Y:16 (replay_ratio 1), Y:17 + D:510-511,563 (1024-step random prefill) | **UNDECLARED — Med** (≈8 replayed steps per env step vs. sheeprl's ≈512 at equal knob value; early data from a random *network*, not the random *policy* the recipe prescribes) |
| U5 | Imagination discount weights omit the start state's **true continue**: rollouts that start at terminal replay rows keep weight 1 (ours never consults replay `terminal` in imagination); sheeprl sets `continues[0] = 1 − terminated` from the batch, zeroing those trajectories | T:455-457 | D:247-248,260 | **UNDECLARED — Low-Med** (terminal-start rollouts train actor/critic on post-death imaginations) |
| U6 | Hierarchical decoder ends with a **LayerNorm on the reconstruction output** (per-sensor-group normalization before slicing); recipe decoders end with a plain Linear. Live config uses `encoding_mode: hierarchical`, so every reconstruction passes it | N:224-227 (`DreamerGroupedMLP` final LN), used at N:434,451 | A:274-278 (MLPDecoder heads = bare `nn.Linear`) | **UNDECLARED — Low-Med** (constrains recon outputs; learnable affine partially compensates) |
| C1 | LayerNorm eps 1e-6 (flax default) vs. recipe 1e-3 everywhere | N (all `nnx.LayerNorm`, no eps arg; probed) | Y:28-35 | cosmetic |
| C2 | Init: `hafner_init` gives effective std ≈ 0.77/√fan_in (multiplies by the 0.8796 truncation factor instead of dividing; fan_in instead of fan_avg → for square layers ≈ 0.77× vs recipe ≈ 1.0×, probed); heads that sheeprl gives `uniform_init(1.0)` (actor head, transition/representation outputs, decoder heads) get trunc-normal here | U:220-230; N:196-197 | V:143-162; A:1170-1180 | cosmetic-to-low |
| C3 | No learnable initial recurrent state: zeros init (and zero initial posterior) vs. recipe `tanh(learned param)` + initial posterior from the transition model (`learnable_initial_recurrent_state: True`) | N:97-103 | A:382-394; Y:54 | cosmetic-to-low (interacts with E2) |
| C4 | No pre-GRU projection LayerNorm: `img_in` = Linear + SiLU; recipe = Linear + LN + SiLU before the GRU | N:83,121-123 | A:309-317 | cosmetic |
| C5 | `agent.unimix: 0.01` YAML key is **dead** — no call site passes it; `OneHotDist` hardcodes its own 0.01 default (grep-verified) | yaml:43; U:108 | — | cosmetic (config-protocol nit; harmless while both are 0.01) |
| C6 | Actor/critic lr 3e-5 vs recipe 8e-5 (declared in YAML, but a silent recipe deviation); sequence_length 128 vs sheeprl 64; batch 16 = 16 | yaml:9-10,4,3 | Y:141,158 | cosmetic (declared knobs) |
| C7 | Sequence sampling only at block-aligned offsets (each transition appears in exactly one window phase, sequence starts always at collection-window boundaries) + mixture positive/recent pools — the latter an intentional, declared off-recipe extension ("DreamerV4-inspired") | T:1065-1099, 762-822; yaml:16-22 | sheeprl SequentialReplayBuffer samples arbitrary per-env offsets | cosmetic-to-low (sound post-H7; less window diversity than recipe) |
| C8 | HORIZON/GAMMA/LAMBDA/FREE_NATS/scales hardcoded as module constants (not config); `Ratio` internal counter not checkpointed (subsumed by K4) | T:17-23; U:187-217 | Y:11-13; D:750 | cosmetic |

---

## Details on the new undeclared findings

### U1 — Missing stop-gradients in behavior learning (High)

sheeprl detaches imagined latent states everywhere the actor and critic consume them
(`dreamer_v3.py:219,240,273,307`) and detaches the imagined action inside the log-prob
(`dreamer_v3.py:286`). Ours detaches only the *start* states (T:361-367) and the advantage
(T:468). Because `OneHotDist.sample` is a straight-through estimator (U:130 —
`onehot − sg(probs) + probs`), a differentiable path runs from every later imagined quantity back
through sampled actions into the actor's logits. Three consequences, all inside
`behavior_loss_fn` (T:375-518), whose grads are taken w.r.t. `(actor, critic)` jointly (T:521):

1. `loss_critic` is built on `critic(rollouts['feat'])` (T:460) with `feat` non-detached →
   **the critic's loss gradients update the actor** (recipe: zero, by detach).
2. `actor(feat)` at step t (T:379-381 modulated / T:411-413 plain) with `feat_t` depending on ST
   actions a_{<t} → later-step log-prob and entropy terms backprop **through the world-model
   dynamics** into earlier action probabilities. DreamerV3's discrete-action recipe is explicit
   that the actor learns from REINFORCE only (dynamics backprop is the *continuous* option).
3. `log_probs = Σ actions · log_softmax(logits)` (T:470-472) with `actions` the non-detached ST
   sample → an extra `log π · ∇probs` term rides on the REINFORCE gradient.

**Empirical probe** (project interpreter, 2026-07-08): a minimal replica — ST sample feeding a
stand-in downstream loss — yields a **nonzero gradient w.r.t. the action logits** (e.g.
`[-0.347, -0.148, 0.098, 0.397]`), and the ours-vs-detached log-prob gradients differ
(`[-0.2486, -0.2511, -0.2514, 0.7510]` vs `[-0.2138, -0.2363, -0.2612, 0.7113]` on the same key).
The leak paths are real, not just structural speculation. Magnitude on a full run is
**unverifiable without an ablation** (listed below), but the actor's training signal is
categorically not the recipe's.

Fix direction: `feat = jax.lax.stop_gradient(feat)` where stored into `step_info` /
fed to `actor`, and `actions = jax.lax.stop_gradient(rollouts['action'])` in the log-prob —
mirroring sheeprl's three detach sites. (Keep the ST sample gradient *inside* the RSSM/imagination
itself, exactly as sheeprl does — only the actor/critic consumption points detach.)

### U2 — Target critic used for λ-bootstrap; slow-critic regularizer missing (Med)

Ours: every imagined value (`rollouts['value']`, T:400/421) and the start value `v_start` (T:444)
come from `self.target_critic`; the online critic appears only as the actor baseline and the
critic-loss prediction. sheeprl/Hafner: **all** imagination values come from the online critic
(D:244), and the slow critic enters only as an extra critic-loss term
`− qv.log_prob(target_values)` (D:307-316), which ours does not have. Net effect: value targets
propagate through a 0.02-EMA-delayed network (slower credit assignment), while the recipe's
anti-overfitting regularizer is absent. The 07-04 baseline report described the λ-return
*arithmetic* as sound (it is) — the online-vs-target sourcing and the missing regularizer were
not previously recorded anywhere.

### U3 — Reconstruction loss under-weighted by ≈ obs_dim (Med)

`loss_recon = mean((recon − obs)²)` over batch, time, **and feature dims** (T:242). The recipe's
observation loss is a symlog-MSE **log-probability summed over feature dims** (L:61 via
`SymlogDistribution`), then averaged over batch/time only. With this project's ~40-60-dim
observation vectors, ours weights reconstruction ≈ 40-60× lower relative to the reward, continue,
and KL terms than the recipe does. Combined with U6 this shifts what the latent state is forced
to encode. One-line fix: `jnp.mean(jnp.sum(jnp.square(recon - obs), axis=-1))`.

### U4 — `replay_ratio` is 1/128 of sheeprl's meaning; no random prefill (Med)

`train_steps = Ratio(0.5)(global_step // num_steps)` (M:1809) counts **sequences**, so at
`collect_interval = 128` one unit of "replay ratio" here does 1/128 of the gradient work the same
number does in sheeprl (Y:16, per policy step). In replayed-transitions terms: ours ≈ 8 replayed
steps per env step vs ≈ 512 for sheeprl's shipped 0.5-with-their-semantics on this batch/seq
geometry. This is a deliberate throughput choice (comment at M:1806-1808) but nowhere recorded as
a ~64× training-intensity deviation from the recipe — highly relevant to the long-standing
"Dreamer underperforms rPPO" question. Secondarily, sheeprl prefills 1024 steps with
`action_space.sample()` before any training (Y:17, D:563); ours has no prefill phase — the first
buffer fills from the untrained policy network and training starts once 128 transitions exist
(M:1804).

### U5 — Discount weights ignore the start state's true continue (Low-Med)

sheeprl overrides the imagination's step-0 continue with the replay batch's real
`1 − terminated` (D:247-248), so imagined rollouts launched from death rows carry zero weight in
both actor and critic losses (D:260,297,316). Ours starts `discount_weights` at 1 for every start
state (T:455) and never consults replay `terminal` during imagination — rollouts imagined from
terminal rows are fully weighted.

### U6 — Hierarchical decoder output passes a LayerNorm (Low-Med)

`DreamerGroupedMLP` appends `nnx.LayerNorm(output_dim)` after its final grouped Linear
(N:224-227). The hierarchical decoder (live mode, yaml:65) uses it as the per-sensor
reconstruction head (N:434), so raw reconstructions are per-sample normalized across each padded
sensor-group vector before slicing (N:451-456). Recipe decoders end with a bare Linear
(A:274-278). The learnable per-feature affine restores expressiveness in principle, but the
normalization couples features within a group and pins the pre-affine scale — a nonstandard
constraint sitting directly on the reconstruction target. The comment "similar to Encoder.body"
(N:226) suggests the encoder block was copied without noticing decoders differ.

---

## Prior fixes re-verified (spot checks)

- **H6 arrival-obs (`bfb3780`)** — storage convention (T:618-731) and the unshifted training scan
  (T:211-216) jointly reproduce sheeprl's post-shift (a_{t−1}, o_t) pairing, including retention
  of the terminal arrival observation; reward/terminal/is_first row semantics line up (E1, E3).
  The one residual gap is the never-stored reset observation — already carried by registry row K1.
- **H7 capacity rounding (`bfb3780`)** — floor-to-multiple + guard at T:991-1002; all three
  sampling pools and the CPU path index block-aligned against the rounded capacity.
- **Continue head (`5b093bf`)** and **two-hot grid (`f5df600`+`1703a4c`)** — re-checked at
  T:54-67/256-258 and U:19-101; consistent flag threading at all nine encode/decode sites.

## Verdict

The stack is a **recognizable DreamerV3** — every headline hyperparameter and loss formula the
paper names (clipping, KL balance, free bits, two-hot, λ-returns, percentile return
normalization, entropy scale) is present and correct in form. But it is **not the recipe** in
three places that plausibly matter for learning quality: the actor's gradient is contaminated by
undetached imagination paths (U1), value learning is DreamerV2-flavored (U2), and the world-model
loss balance plus effective training intensity sit far from recipe values (U3, U4). Together with
the still-open collection-reset gap (K1), these are the prime suspects to close before attributing
any Dreamer-vs-rPPO performance gap to the algorithm itself. U1 and U3 are each a few lines to
fix; U2 and U4 are small but semantics-changing (pre/post-fix runs not comparable).

**Unverifiable within this audit (needs runs or is out of reference scope):**
1. Magnitude of U1/U3/U5 effects on trained policy quality — needs an ablation run pair.
2. Whether U4's low training intensity explains the historical Dreamer-vs-rPPO gap — needs a
   replay-ratio-matched run (note the existing `dreamer_replay_ratio_sweep.md` predates H6/H7).
3. All modulator-enabled paths (FiLM/PreActivation/Multiplicative, `mod_h` threading) — project
   extensions with no sheeprl analog; audited only for internal consistency.
4. The CPU buffer path (`buffer_device: "cpu"`) — not exercised by any live config; baseline
   Finding 10's nits (silent `config.get` defaults, unseeded `np.random`, per-call re-jit) stand.

Reviewed by: code-reviewer (parity/convention audit, area 5 of 5, 2026-07-08)

---

**Fix routing (2026-07-08):** U1–U6 + K1's collection-reset fix are packaged as work package
WP-NNX — plan with per-fix file changes, red-pre-fix regression tests, and the U4/U6 decision
recommendations: [[fix_plan_nnx_parity]].
