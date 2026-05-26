---
title: "Why does dreamer-srl v2 still trail recurrent-PPO on 10×10 hypervigilance? Ranked hypotheses after the buffer-size sweep"
topic: dreamer_srl_v2
status: active
created: 2026-05-23
last_updated: 2026-05-23
phase: post_buffer_sweep_diagnosis
wandb_tag: "dreamer_srl_v2_postfix_XS_envs_16_4M_*_buf*"
cross_links:
  - docs/experiments/active/dreamer_srl_v2/REWARD_HEAD_ASYMMETRY_ANALYSIS.md
  - docs/experiments/active/dreamer_srl_v2/HYPERPARAM_SEARCH_10X10.md
---

# Why does dreamer-srl v2 still trail recurrent-PPO on 10×10 hypervigilance?

## 1. Question (plain-language entry point)

Two reinforcement-learning agents are being compared on a survival benchmark: a 10×10 grid called *hypervigilance* in which the agent must avoid four hidden predators while gathering food. The headline number is **survival steps** — how many environment steps the agent stays alive (max 500). The two agents are:

- **rPPO** — recurrent PPO, a model-free policy-gradient agent that learns directly from real environment reward. The current production run survives **227-239 steps**, training on roughly **2 billion environment steps** over **about 24 hours**.
- **dreamer-srl v2** — a JAX re-implementation of DreamerV3, a *model-based* agent that first learns a *world model* (a compact predictor of future observations, rewards, and episode termination) and then trains its policy on *imagined* roll-outs of that world model. The best run we have ever produced (the cell with a 256 000-transition replay buffer, henceforth "**buf256k**") survives **189 steps** after **4 million environment steps** and **about 29 hours** of wall-clock.

So dreamer-srl uses **roughly 500× fewer real environment steps** and reaches **about 80 % of rPPO's survival**. The user asks: are we still under-trained, under-tuned, or is something deeper wrong?

This note ranks **5 candidate explanations** for the residual gap, in descending expected information value. Each row gives a one-sentence rationale and a one-sentence test. The headline finding is below the table; the detailed evidence is in §3.

**Headline finding.** The single strongest signal — visible directly in the buffer sweep — is that **the two best Dreamer cells are still on a steep upward trajectory at the 4 M-step cap** (`buf500k` rose +25 ep_steps in its last training quartile; `buf256k` rose +19). The baseline cell with the larger buffer has already plateaued. This is a textbook under-trained signature in the cells that matter, so **"insufficient environment-step budget for the buffer regime that actually works"** is the leading hypothesis. The reward-head pos/neg asymmetry remains a structural concern (already documented and persists in this sweep) but is now a secondary lens, not the leading lens.

---

## 2. TL;DR — Ranked hypotheses

Confidence reflects how directly the existing 5-run sweep speaks to the hypothesis. "How to test" lists the cheapest follow-up that would shift confidence one bucket.

| # | Hypothesis | Rationale (one line) | How to test (one line) | Confidence |
|---|---|---|---|---|
| **1** | **Under-trained at 4 M env-steps** for the small-buffer regime | `buf500k` and `buf256k` are still rising +18 to +25 ep_steps per quartile at the cap; baseline (1 M buf) has plateaued | Extend `buf256k` (or both top cells) to 8 M or 12 M env-steps, same seed, same config | **High** |
| **2** | **Imagination-horizon × reward-head bias** under-fits the cost of dying | Reward-head pos/neg MAE ratio is 5-10× across all 5 cells; KL losses are stable (no blow-up), so the rep is learning, but the *reward head* is the bottleneck — and a 15-step imagined return compounds that error | (a) Inspect per-horizon reward MAE (already requested as a metric earlier); (b) ablation: shorten `horizon` from 15 → 8 and re-run buf256k; (c) double the reward-head capacity | **High** |
| **3** | **Hyperparameter coverage gap** — learning-rate / batch / replay-ratio never swept at the new buffer setting | We changed buffer size by 4× and observed a large effect; we have NOT touched `actor_lr=8e-5`, `wm_lr=1e-4`, `batch_size=16`, `seq_len=64`, `replay_ratio=1` since the parity baseline | One-shot Latin-hypercube of 4-6 cells: `{lr/2, lr×2} × {batch=32} × {replay_ratio=0.5, 2}` at fixed buf256k | **Medium** |
| **4** | **DreamerV3 is genuinely sample-efficient but per-step expensive** — gap is a budget artefact, not an algorithm defect | At 38-40 SPS on H100, 4 M env-steps is ~28 h; rPPO at ~2 B steps in 24 h is in a fundamentally different sample regime. The DreamerV3 paper's own framing ("low V100-days but low SPS") matches this; we may simply not be running long enough in wall-clock either | Per-wall-hour comparison: plot ep_steps vs wall-time (not env-steps) for both agents; if Dreamer's slope is still positive at hour 28 it confirms (1) | **Medium** |
| **5** | **Implementation residual** — a subtle JAX/Flax port bug surviving parity tests | The parity test was on food-only (positive-reward-only) where the reward head's positive MAE is fine; the asymmetry only manifests on hypervigilance. A latent bug in the negative-reward path (e.g., symlog/bin discretization, target-network update under tracing) could still be live | Offline world-model diagnostic on a buf256k checkpoint: feed real trajectories, measure per-step reward prediction vs. ground truth conditioned on injury events | **Low–Medium** |

Two non-promoted items: (a) **environment-specific disadvantage** — partial observability via four hidden predators + 27-dim vector obs is exactly the regime DreamerV3 was designed for, so we don't promote this; (b) **network-size ceiling** — XS at 256-wide already exceeded sheeprl-XS, and the HP-search doc found S/M did not help at this env-step budget, so we don't promote this either, but it would re-enter the picture once (1) is settled.

---

## 3. Evidence

### 3.1. Buffer sweep snapshot at 4 M env-steps (the source data)

All five cells were launched with identical config except `buffer_size`. All ran on node 114, seed 42, 16 parallel envs, XS model preset. See [PARITY_LAUNCH_V2.md](./PARITY_LAUNCH_V2.md) for launch details.

| Cell | Buffer | Wall (h) | `Episode/Steps` end | last-5 mean | Trajectory (q1→q4) | Δ last quartile |
|---|---:|---:|---:|---:|---|---:|
| baseline | 1 000 000 | 28.0 | 156.2 | 156.2 | 101 → 139 → 164 → 154 | **−10** (plateaued/declining) |
| buf500k | 500 000 | 26.9 | 175.5 | 192.4 | 86 → 137 → 161 → 186 | **+25** (still rising) |
| **buf256k** | 256 000 | 28.8 | **189.5** | 183.7 | 103 → 132 → 160 → 179 | **+19** (still rising) |
| buf100k | 100 000 | 31.0 | 165.6 | — | 92 → 143 → 140 → 151 | +11 |
| buf50k | 50 000 | 30.0 | 160.1 | — | 93 → 122 → 139 → 147 | +8 |

The shape "smaller buffer → faster early learning, still rising at cap" is consistent across the four non-baseline cells.

### 3.2. World-model diagnostics at 4 M env-steps

| Cell | `model_reward_mae_pos` | `model_reward_mae_neg` | ratio (neg/pos) | `loss_kl` | `loss_dyn_kl` | `loss_rep_kl` | `model_cont_acc` |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 0.128 | 0.619 | **4.8×** | 1.72 | 1.43 | 0.286 | 0.999 |
| buf500k | 0.172 | 0.849 | **4.9×** | 1.76 | 1.47 | 0.294 | 0.998 |
| buf256k | 0.096 | 1.009 | **10.5×** | 1.56 | 1.30 | 0.260 | 0.997 |
| buf100k | 0.286 | 1.326 | **4.6×** | 1.82 | 1.51 | 0.303 | 0.996 |
| buf50k | 0.274 | 0.831 | **3.0×** | 1.73 | 1.44 | 0.288 | 0.998 |

- The reward-head asymmetry (already documented in [REWARD_HEAD_ASYMMETRY_ANALYSIS.md](./REWARD_HEAD_ASYMMETRY_ANALYSIS.md)) **persists in all 5 cells**, ratio 3-10×.
- KL is stable everywhere (1.5-1.8); no blow-up signature.
- Continuation accuracy is essentially perfect everywhere.
- Note `buf256k` has the *worst* asymmetry ratio (10.5×) yet the *best* survival — counter-evidence to a simple "reward asymmetry directly bottlenecks survival" claim. This is why hypothesis #2 has to specifically appeal to the **horizon × asymmetry interaction** rather than per-step asymmetry alone.

### 3.3. Actor / value diagnostics

| Cell | `mean_entropy` end | `mean_entropy` mid | `value_mae` | `mean_return` |
|---|---:|---:|---:|---:|
| baseline | 0.161 | — | 8.34 | 13.29 |
| buf500k | 0.146 | 0.162 | 5.97 | 13.14 |
| buf256k | 0.140 | 0.153 | 6.87 | 5.12 |
| buf100k | 0.154 | — | 4.03 | 7.54 |
| buf50k | 0.160 | — | 5.59 | 3.10 |

- Entropy has **not collapsed** (~0.14-0.16, well above the 1e-3 collapse threshold).
- `loss_actor_entropy ≈ -4×10⁻⁵` everywhere, so the ent_coef=3e-4 regularizer is active but not dominant.
- Value MAE in the buf500k / buf256k cells is 6-7 — high relative to the realized return scale, consistent with "value head has not yet converged".

### 3.4. Why hypothesis #1 outranks the others

The buffer sweep was launched as a *learning-lever* probe (does sample diversity hurt the world model?) and unexpectedly produced what looks like a *budget* result: the cells that win are still climbing. That is the cheapest, highest-information next test we can possibly run, and it directly addresses the user's question "are we under-tested or under-trained" — the answer the data is currently giving is **under-trained, in the regime that actually works**. Until that extension run lands, no other hypothesis can be cleanly tested, because we don't know whether their current effect sizes are floor or ceiling.

---

## 4. Recommended next step

Single experiment, single seed extension of the two top cells:

- Continue **buf256k** and **buf500k** from their final checkpoints to **8 M env-steps** (or **12 M** if compute allows) on node 114, same config, same seed.
- If feasible, add a **single seed=7 replica of buf256k** to start gathering seed-dispersion evidence (the entire sweep is currently single-seed, so we have *no* confidence interval on the 189.5 headline).
- Wall-clock estimate: 28 h current → ~56 h to 8 M, ~84 h to 12 M.

If the extension confirms hypothesis #1 (slope stays positive past 4 M and the gap to rPPO closes), the priority queue shifts to: (1) seed-dispersion on the extended cell, (2) horizon ablation (hypothesis #2) at the new ceiling. If the extension shows a plateau at 4-6 M (hypothesis #1 refuted at this config), the priority shifts to (2) and (3) immediately.

---

## 5. Methods

Five runs (see [PARITY_LAUNCH_V2.md](./PARITY_LAUNCH_V2.md) for full launch manifest):

- `rgupbwc6` — baseline (`buffer_size=1_000_000`)
- `j3vrkci0` — buf500k
- `gu63ncqg` — buf256k (current Dreamer best on this env)
- `3951cwyx` — buf100k
- `aaithzu8` — buf50k

All on node 114 (RTX 6000 Ada), seed 42, 16 parallel envs, XS model preset, `replay_ratio=1`, `horizon=15`, `seq_len=64`, `batch_size=16`, `actor_lr=8e-5`, `wm_lr=1e-4`, `ent_coef=3e-4`. Algorithm: dreamer-srl v2 post-JointTrainer refactor (commit `3e4732641`).

End-of-training metrics extracted from local `wandb/run-*/files/wandb-summary.json` (read-only). Trajectory quartile means and entropy mid-points extracted via one-shot `wandb.Api()` history pulls (20-sample summary) — used here strictly to validate the "still-rising" signal; not used for primary numbers.

rPPO production reference 227-239 ep_steps / ~24 h / 10 M episodes is taken as user-supplied context, not independently verified in this analysis.

## 6. Metrics requested

If the dev pipeline can land them, two metrics would sharpen the next round:

| Metric | Why now | Where it'd live | Cost |
|---|---|---|---|
| Per-imagination-step reward MAE (`WorldModel/imag_reward_mae_h{1..15}`) | Hypothesis #2 explicitly predicts compounding along the horizon; we cannot test it without per-horizon error | dreamer-srl behavior trainer, after imagination roll-out | Moderate (vector of length 15 per step) |
| Per-wall-hour `Episode/Steps` snapshot for both Dreamer and rPPO | Direct test of hypothesis #4; eliminates the env-step axis-mismatch | Already implicit in `wandb` `_runtime` field, just needs a panel | Cheap |

## 7. Related issues

- The buffer-size sweep doc itself is the immediate upstream.
- Reward-head asymmetry analysis ([REWARD_HEAD_ASYMMETRY_ANALYSIS.md](./REWARD_HEAD_ASYMMETRY_ANALYSIS.md)) — hypothesis #2 above is the post-sweep refinement of that doc's H2.
- TODO: this is Mode B post-hoc — if hypothesis #1 confirms after extension, a pre-registered "Dreamer-vs-rPPO at matched wall-clock" comparison should be authored by `experiment-designer`.
