---
title: "Dreamer vs sheeprl — Master Parity Comparison (Fable 5 re-audit)"
topic: diagnosis
status: active
created: 2026-07-06
last_updated: 2026-07-08
---

# Dreamer vs sheeprl — Master Parity Comparison

## What this is (plain-language entry point)

This is the **master record of a from-scratch re-comparison of the project's two Dreamer
implementations against the vendored sheeprl reference** (the upstream PyTorch DreamerV3 the
project treats as ground truth), run 2026-07-06→08 on the Fable 5 model at the user's request.
A comparison had been done before (see [[SHEEPRL_REFERENCE_AUDIT]] and [[DEVIATION_LOG]]),
but this audit re-derived everything rather than trusting the old record. Our implementations
are JAX, so line-identical code is impossible — the standard applied is **numerical/algorithmic
equivalence**, with every difference classified as: exact parity, provably-equivalent-by-design,
a declared deviation, an **undeclared deviation with training impact** (the dangerous kind),
or cosmetic. Five reviewers each audited one area, with paired torch/JAX numerical probes on
shared fixtures wherever cheap; their complete item-by-item tables (including the
"checked-and-equal" rows) are the five reports beside this file.

**Headline:** the cores are healthier than feared — λ-returns, the REINFORCE actor loss, the
Moments return-normalizer, KL balance, and the two-hot machinery are **bit-identical or
within float32 noise** of sheeprl in the dreamer_srl port. But the audit confirmed and newly
found **11 undeclared deviations with training impact** across the two stacks — topped by a
multi-env replay-buffer corruption (garbage rows punched into innocent envs' columns at every
reset), missing gradient clipping on all three dreamer_srl optimizers, an actor-gradient
stop-gradient leak in the NNX stack, a 2× under-weighted reconstruction loss, and a γ constant
that is silently wrong in all 19 live configs. None of these are in the deviation log — the
log itself needs maintenance.

**The two stacks fail in complementary ways.** dreamer_srl (the declared sheeprl port) has
near-perfect *behaviour-learning* parity but broken *loss weighting*, *no clipping*, and
*driver/buffer* defects. DreamerV3-NNX (in-house, never a port) has clipping and loss plumbing
mostly right but leaky *gradient routing*, DreamerV2-style value learning, and ~1/128 of the
recipe's replay intensity.

---

## 1. Method

- Reference: `vendor/sheeprl/sheeprl/algos/dreamer_v3/` (agent/dreamer_v3/loss/utils, 2,396
  lines) + `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml`. The port declares fidelity
  to sheeprl@33b6366.
- Ours: `src/algorithms/dreamer_srl/` (port; areas 1–4) and `src/models/dreamer_v3_*` (in-house
  NNX stack; area 5), plus the LIVE config values (19 dreamer_srl YAMLs, dreamer_v3.yaml).
- Every area re-derived from code; prior audit/deviation-log used only to classify
  declared-vs-undeclared. Empirical probes: paired torch (sheeprl_bridge env) / JAX fixtures.
- Baseline includes this week's fixes: H5 `b1dd90a`, H6+H7 `bfb3780` — both re-verified at
  parity by the relevant auditors.

## 2. Verified-parity summary (the good news)

| Area | Report | Items | PARITY | Equiv-by-design | Notable bit-identical probes |
|---|---|---|---|---|---|
| 1 RSSM + WM architecture | [[01_rssm_architecture]] | 52 | 34 | 6 | Lever-A suite 9/9 (GRU, transition/representation, is_first resets, action shift) |
| 2 WM losses | [[02_world_model_losses]] | 23 | 13 | 3 | KL 1.4e-6, reward two-hot 7.2e-6, continue 1.2e-7 vs torch |
| 3 Actor-critic + returns | [[03_actor_critic_returns]] | 30 | 22 | 3 | λ-returns diff 0.0, REINFORCE actor loss diff 0.0, Moments chain 0.0 |
| 4 Training loop + replay | [[04_training_loop_replay]] | ~25 | 14 | — | Ratio accounting, buffer wrap/sample, H5 done-block sheeprl-faithful |
| 5 DreamerV3-NNX vs recipe | [[05_dreamer_v3_nnx_conventions]] | ~40 | 16 | 4 | Clipping 1000/100/100 at exact parity; H6 arrival-obs equivalence re-verified |

## 3. Undeclared deviations with training impact — dreamer_srl (the port)

Ranked. **None have DEVIATION_LOG rows** — every kept row below needs one (or a fix).

| # | Severity | Deviation | Ours vs sheeprl | Training effect | Fix shape |
|---|---|---|---|---|---|
| P1 | 🔴 High | **No gradient clipping, all 3 optimizers** | `dreamer_srl_main.py:658-660` plain adam vs clip-by-norm 1000/100/100 (`dreamer_v3.py:193-197,300-302,320-324`) | Unbounded steps; WM loss empirically spikes to ~1e29–1e31 in live smokes | 3× `optax.chain(clip_by_global_norm(N), adam)` |
| P2 | 🔴 High (multi-env) | **Shared positive-buffer write-head punches hole rows** | one shared head vs `EnvIndependentReplayBuffer` | Garbage rows written into NON-done envs' columns at every reset (probe: `[21,31,0,41]`); also holds num_envs× intended capacity. Dormant at num_envs=1, active multi-env | per-env write-heads or masked add |
| P3 | 🔴 High | **Obs loss: extra 0.5 → reconstruction under-weighted exactly 2×** (probe 0.500000) + decoder trained in real space (extra symlog on output; grad scale 0.23–0.50×) | `train.py:704-712` vs `SymlogDistribution` | Recon-vs-KL/reward/continue balance off by half; gradient geometry distorted | call the already-existing faithful `reconstruction_loss` (`loss.py:424-582` — imported, never used) |
| P4 | 🟡 Med | **γ typo in all 19 live configs** | `0.996840347` vs `0.996996996996997`, false citation comment | ~5% shorter credit horizon (316.5 vs 333 steps) in every run; breaks strict-parity baselines | one-line config fix (or declare) |
| P5 | 🟡 Med | **Episode metric bleed** | `main.py:1264-1266` vs `:1500-1501` re-increment after done-reset | Every logged episode: +1 survival step, + predecessor's terminal reward — biases the project's headline metric | reorder counters |
| P6 | 🟡 Med (multi-env) | **learning_starts counted in iterations** | `main.py:1110` vs sheeprl per-env-step | Prefill 1024×num_envs env-steps (16× at 16 envs); silent recipe drift | divide by num_envs |
| P7 | 🟡 Med (fractional ratio) | **Replay-ratio remainder dead code** | `main.py:1043,1523-1530` | `replay_ratio=0.5` silently forced to 1.0 at low env counts; remainder never corrects | wire `_grad_step_remainder` or delete + declare |
| P8 | 🟢 Low | Train gate ignores `buffer._full` | `main.py:1532` | Up to seq_len−1 skipped grad steps after each ring wrap | check `_full` |
| P9 | 🟢 Low | RSSM output-linear init distribution shape (trunc-normal vs Hafner uniform, same std) | `agent.py:703-714` vs `agent.py:1173-1174` | Init-time only | optional |

Cosmetic (see area reports): dead faithful `reconstruction_loss` (= the P3 fix), missing `tol=1e-8`
clamp, entropy `+1e-8` epsilon (declared), dead decoder config keys, vestigial classes, stale comments.

**Log maintenance:** D-014 row is stale (code now subtracts `learning_starts*num_envs`); the
open KNOWN_BUGS clipping row names only the WM — broaden to all three optimizers.

## 4. Undeclared deviations with training impact — DreamerV3-NNX (vs the recipe)

This stack never declared sheeprl fidelity; rows below are recipe deviations worth fixing or
consciously keeping. Known-open registry rows (is_first=0 at collection, eval no-symlog +
PRNGKey(0), unimix-vs-raw-logits, checkpoint omissions, collect_interval validation, target-critic
init) were all confirmed still present — see [[05_dreamer_v3_nnx_conventions]].

| # | Severity | Deviation | Where | Training effect |
|---|---|---|---|---|
| U1 | 🔴 High | **Stop-gradient leaks into the actor** — sheeprl detaches imagined feat/actions at 3 sites; we detach none | `dreamer_v3_trainer.py:379-381,411-413,460,470-472` vs `dreamer_v3.py:219,240,273,286,307` | Critic-loss + dynamics-backprop gradients contaminate the actor; not pure REINFORCE (empirically confirmed probe) |
| U2 | 🟡 Med | λ-returns bootstrap from the **target** critic (recipe: online) + slow-critic regularizer absent | `trainer:400/421/444` vs `dreamer_v3.py:244,314-315` | DreamerV2-style value learning |
| U3 | 🟡 Med | Obs loss `mean` over feature dims (recipe: sum) | `trainer:242` vs `loss.py:61` | Reconstruction under-weighted ~40–60× (obs_dim) |
| U4 | 🟡 Med | `replay_ratio` counted per-sequence (recipe: per env step) + no random prefill | `train.py:1809` | ~1/128 of the recipe's replay intensity at equal config value |
| U5 | 🟢 Low-Med | Imagination discount weights start at 1, no true-continue override | `trainer:455` vs `dreamer_v3.py:247-248,260` | Rollouts from terminal rows fully weighted |
| U6 | 🟢 Low-Med | Decoder output ends in LayerNorm (recipe: bare Linear) | `dreamer_v3_nnx.py:224-227` | Nonstandard constraint on the recon head |

Also cosmetic-tier: LayerNorm eps 1e-6 vs 1e-3, init std ~0.77×, dead `agent.unimix` YAML knob
(hardcoded 0.01), lr 3e-5 vs 8e-5, hardcoded γ/λ/H.

## 5. Cross-stack observations

1. **Complementary failure profiles** (see headline). Anyone porting fixes between stacks should
   note a deviation absent in one stack is often present in the other — e.g. clipping (srl ✗ /
   NNX ✓), obs-loss weighting (srl 2× / NNX 40–60×), stop-gradients (srl ✓ / NNX ✗).
2. **The dangerous deviations are all in glue, not math.** Every bit-identity probe on the core
   equations passed; the impact rows live in loss *assembly*, optimizer *wrapping*, buffer
   *write paths*, and config *constants* — exactly the places line-by-line porting attention lapses.
3. **Comparability:** fixing P1/P3/P4 (srl) or U1–U4 (NNX) changes training. Post-fix runs will
   not be comparable to pre-fix runs — same situation as the H4–H7 fixes. Batch the fixes to
   minimize the number of comparability epochs.
4. **num_envs=1 masked two of the worst rows** (P2, P6 dormant there): the original parity pass
   ran single-env, so "parity-passed" history does not certify multi-env correctness.

## 6. Suggested fix packaging (not yet routed)

1. **WP-srl-1 (small, high leverage):** P1 clipping + P3 obs loss (call the existing faithful fn)
   + P4 γ + P5 metric bleed — one package, one comparability break.
2. **WP-srl-2 (multi-env correctness):** P2 buffer hole rows + P6 learning_starts + P7/P8 — before
   any multi-env dreamer_srl launch.
   → **Routed 2026-07-08** (user approved "fix all"): items 1+2 merged into one work package
   **WP-SRL** — P1/P2/P3/P5/P6/P7/P8 fixed, P9 declared, D-014 refreshed —
   plan: [[fix_plan_srl_parity]]. P4 (γ) is carved out to the parallel **WP-GAMMA**
   (experiment-designer, config-side). WP-SRL + WP-GAMMA form ONE dreamer_srl comparability
   epoch: post-fix runs are not comparable to any earlier run.
3. **WP-nnx-1:** U1 stop-gradients + U3 obs-loss sum (few lines each, biggest leverage) + the
   2-line is_first collection fix (known-open) — before the next result-bearing NNX run.
4. **WP-nnx-2 (deliberate recipe alignment):** U2 online-critic bootstraps + slow regularizer,
   U4 replay-ratio semantics + prefill — bigger, decide consciously.
   → **Routed 2026-07-08**: items 3+4 merged into one work package with per-item
   recommendations (U4 cost-neutral rescale, U6 fix-now) — plan: [[fix_plan_nnx_parity]].
5. **Log maintenance:** add D-rows for every kept deviation; refresh D-014; broaden the clipping row.

## 7. Reports

[[01_rssm_architecture]] · [[02_world_model_losses]] · [[03_actor_critic_returns]] ·
[[04_training_loop_replay]] · [[05_dreamer_v3_nnx_conventions]] ·
[[06_nnx_recipe_deviation_register]] (NNX kept/fixed/open dispositions after WP-NNX)

Prior record: [[SHEEPRL_REFERENCE_AUDIT]] (v2 era) · [[DEVIATION_LOG]] (v1 era) ·
Registry: [[KNOWN_BUGS]] · This week's fix cluster: [[00_combined_diagnosis]]
