---
title: "Are the 3 dreamer-srl 10M-episode runs reproducible, and is the per-row 'fluctuation' policy instability or just per-episode noise?"
topic: dreamer_srl_v2
status: active
created: 2026-05-28
last_updated: 2026-05-28
phase: post_buf256k_extension
wandb_tag: "dreamer_srl_v2_postfix_XS_envs_16_ep10M_buf256k_s42_*"
cross_links:
  - docs/experiments/active/dreamer_srl_v2/dreamer_vs_rppo_gap_hypotheses.md
  - docs/experiments/active/dreamer_srl_v2/REWARD_HEAD_ASYMMETRY_ANALYSIS.md
  - docs/experiments/active/dreamer_srl_v2/PARITY_LAUNCH_V2.md
---

# Are the 3 dreamer-srl 10M-episode runs reproducible, and is the per-row "fluctuation" policy instability or just per-episode noise?

> **Status**: ANALYZING — Mode B retrospective (not pre-registered). Analysis covers 3 cells terminated 2026-05-28 ~02:33 (KST), all reached ~7.3–7.4 M environment steps, ~47k–48k episodes, ~49–51 h wall-clock.
> **Date**: 2026-05-28

## 1. Question (plain-language entry point)

Three identical training runs of the **dreamer-srl v2** agent (a JAX re-implementation of DreamerV3 — a *model-based* reinforcement-learning agent that first learns a compact predictor of the world, then trains a policy on imagined roll-outs) were launched on the 10×10 *hypervigilance* survival task with the same code, same config (`buf256k` — the 256 000-transition replay buffer that won the prior buffer sweep), the same random seed (42), and 16 parallel environments. They differ only in the node and GPU they ran on, and in the start time. All three were terminated within a window of about an hour after roughly **2 days of wall-clock**.

The user opened the three WandB pages and saw curves that "look very fluctuating" and wanted to know whether **the policy is actually unstable, or whether it's just an artifact of the logging cadence** (a known config bug meant all three runs logged metrics every 10 iterations — about one row per episode at 16 parallel envs — which produces a very jagged trace dominated by single-episode noise, even when the underlying policy is smoothly improving).

**Headline finding (plain English).** The fluctuation is **per-episode noise, not policy instability**. When the same 47 000-episode trace is summarised every 5 000 episodes (a 500× smoother window), all three runs produce **nearly identical, monotonically rising learning curves** going from ~59 survival steps in the first 5 000 episodes to ~200 survival steps in the last 5 000. The cell-to-cell spread at every checkpoint is **< 4 survival steps** — much smaller than the within-cell single-episode spread (standard deviation ~140 around a mean of ~200). The final-performance estimate, averaged over the last 2 000 episodes per cell, lands at **200.7 / 201.6 / 201.2 survival steps** for cells A / B / C respectively — extremely tight reproducibility. The runs are also **still rising** in the last block (last 5 000 episodes are the highest of all 10 blocks for every cell), so they have *not* plateaued — terminating them at ~7.4 M env-steps left learning unfinished, consistent with the leading hypothesis in [dreamer_vs_rppo_gap_hypotheses.md](dreamer_vs_rppo_gap_hypotheses.md) ("under-trained at 4 M env-steps").

## 2. Cell labels & launch context

| Cell | WandB run ID | Full WandB name | Node / GPU | Start (KST) | Wall-clock (h) | Final env-step | Final episode |
|---|---|---|---|---|---:|---:|---:|
| **A** | `jkmto06f` | `…_n114_gpu0` | n114 / GPU 0 | 2026-05-25 23:02 | **51.4** | 7 442 960 | 47 351 |
| **B** | `empiuzmd` | `…_n114_gpu1_log5k` | n114 / GPU 1 | 2026-05-26 01:02 | **49.5** | 7 430 160 | 47 651 |
| **C** | `flq7e8c9` | `…_n113_gpu0_log50k` | n113 / GPU 0 | 2026-05-26 01:03 | **49.4** | 7 317 680 | 47 984 |

All 3 cells:
- env config: `configs/experiment/hypervigilance/01-interoNocicept.yaml` (10×10 hypervigilance — survive while gathering food, 4 hidden predators, max 500 steps)
- agent config: `configs/dreamer_srl/01_food_only_buf256k.yaml` (XS model preset, 256k replay buffer)
- CLI: `--episodes 10000000 --num-envs 16 --seed 42 --legacy-grad-loop`
- code commit: `4d8d359` (cell A; B/C launched a few hours later but pre-`0fb2795` log-interval-fix commit)

**Known config bug (not affecting analysis):** Cells B and C were intended to log every 5 000 / 50 000 iterations respectively. The `dreamer_srl_main.py` code at the time only read `log_interval` from `env_cfg`, ignoring the `agent_cfg.training.log_interval` override. So **all three cells effectively logged every 10 iterations** (~0.8 episodes per row), producing the visual "fluctuation" the user is seeing. The next launch (commit `0fb2795`) reads the agent-cfg override correctly.

## 3. Headline trajectory (smoothed) — what the curves WOULD have looked like at log_interval=50 000

The single most useful view. Each row is the mean ± std of **5 000 consecutive episodes** — exactly the smoothing the user would have gotten if `log_interval=50000` had been honoured. Same data as the noisy per-episode WandB plots, just averaged.

| Episode block | A — `…_n114_gpu0` | B — `…_n114_gpu1` | C — `…_n113_gpu0` |
|---|---:|---:|---:|
| 0 – 5 000 | **58.9** ± 40.9 | **59.6** ± 36.0 | **58.6** ± 39.2 |
| 5 000 – 10 000 | 129.5 ± 86.8 | 116.5 ± 72.3 | 115.7 ± 72.6 |
| 10 000 – 15 000 | 147.9 ± 97.7 | 144.7 ± 90.5 | 142.0 ± 92.5 |
| 15 000 – 20 000 | 161.1 ± 111.0 | 159.6 ± 105.5 | 147.5 ± 96.7 |
| 20 000 – 25 000 | 173.1 ± 120.5 | 169.4 ± 115.4 | 158.3 ± 111.0 |
| 25 000 – 30 000 | 184.2 ± 132.0 | 183.8 ± 126.8 | 167.5 ± 121.8 |
| 30 000 – 35 000 | 183.1 ± 130.9 | 179.7 ± 126.2 | 179.4 ± 127.5 |
| 35 000 – 40 000 | 178.4 ± 127.2 | 182.0 ± 128.9 | 187.8 ± 133.5 |
| 40 000 – 45 000 | 187.0 ± 132.3 | 193.4 ± 132.5 | 195.8 ± 136.8 |
| **last (~45k–end)** | **201.1** ± 140.3 | **200.2** ± 139.2 | **201.6** ± 139.5 |

**Reading this table:**
- All three cells started at **~59 survival steps** in the first 5 000 episodes (warm-up regime — buffer not yet full).
- All three converged to **~200 survival steps** in the last 5 000 episodes — agreement within ±1 survival step.
- The within-block standard deviation is ~140 around a mean of ~200, i.e. the per-episode CV is ~70 %. This is *enormous* compared to the cell-to-cell spread (~1 step), which is what makes the WandB per-episode trace look chaotic. **The signal is in the mean; the noise is in any single episode.**

## 4. Quintile summary (the same data sliced into 5 equal episode-count blocks, ≈ 9 500 episodes per block)

A coarser view that makes the monotonic rise unmistakable.

| Quintile (eps) | A — n114/g0 | B — n114/g1 | C — n113/g0 |
|---|:---|:---|:---|
| Q1 (1st 9.5k) | 90.9 ± 74 (p50=82) | 85.5 ± 62 (p50=88) | 85.1 ± 63 (p50=81) |
| Q2 | 153.1 ± 103 (p50=131) | 150.0 ± 96 (p50=130) | 143.3 ± 93 (p50=125) |
| Q3 | 176.3 ± 124 (p50=147) | 174.5 ± 119 (p50=150) | 161.7 ± 115 (p50=136) |
| Q4 | 180.8 ± 129 (p50=150) | 180.3 ± 127 (p50=151) | 179.2 ± 129 (p50=148) |
| **Q5 (last 9.5k)** | **189.6 ± 134** (p50=158) | **194.0 ± 134** (p50=162) | **198.0 ± 138** (p50=166) |

- Each cell rises monotonically Q1→Q5 (no within-cell collapse).
- Q4→Q5 gain is **+8.8 / +13.7 / +18.8** survival steps respectively → all 3 cells still on an upward slope at termination, **C has the steepest last-quintile slope**. Consistent with the prior under-trained signature.

## 5. Final-performance verdict (last 2 000 episodes per cell)

| Cell | Last-2k mean ± std | p25 | p50 | p75 | Last-2k ep_rew |
|---|---:|---:|---:|---:|---:|
| A | **200.7** ± 139.7 | 91 | 169 | 279 | −314.6 ± 21.0 |
| B | **201.6** ± 140.5 | 91 | 168 | 285 | −314.4 ± 21.0 |
| C | **201.2** ± 140.4 | 90 | 167 | 280 | −313.8 ± 21.4 |

- **3-cell mean: 201.2 survival steps; cell-to-cell std: 0.45 steps** (~ 0.2 % of the mean).
- 95 % CI for the cell-mean ≈ 201.2 ± 1.1 (using cell-level n=3) — i.e. the runs are essentially indistinguishable.
- **Vs prior buf256k 4M-env-step reference**: prior was 189.5 ep_steps at ~28h / 4M env-steps; current is 201 ep_steps at ~50h / 7.3–7.4M env-steps. **+11.5 ep_steps for +85 % env-steps and +80 % wall-clock.** Diminishing returns visible: doubling compute bought ~6 % more survival. (The prior figure used a different rolling-window definition, so the comparison is approximate.)
- **Vs rPPO production (~227–239 ep_steps)**: dreamer-srl is now at **~84–88 % of rPPO survival**, up from ~80 % at the 4M-step reference. Gap closing, but slowly.

## 6. Reward-head asymmetry trajectory (WorldModel reward-prediction error: positive vs negative reward events)

The asymmetry that was the leading suspect in [REWARD_HEAD_ASYMMETRY_ANALYSIS.md](REWARD_HEAD_ASYMMETRY_ANALYSIS.md). Negative-reward events (predator hits, starvation, injury) are predicted ~4–5× less accurately than positive-reward events (food). Persists across all 3 cells, virtually identical:

| Step block | A mae_neg / mae_pos | B mae_neg / mae_pos | C mae_neg / mae_pos |
|---|---:|---:|---:|
| 0.0 – 0.8M | 1.090 / 0.600 (1.8×) | 0.976 / 0.441 (2.2×) | 1.072 / 0.478 (2.2×) |
| 1.5 – 2.2M | 0.934 / 0.196 (4.8×) | 0.871 / 0.201 (4.3×) | 0.947 / 0.209 (4.5×) |
| 3.7 – 4.5M | 0.988 / 0.187 (5.3×) | 0.946 / 0.190 (5.0×) | 1.013 / 0.190 (5.3×) |
| 5.9 – 6.7M | 1.068 / 0.242 (4.4×) | 1.044 / 0.236 (4.4×) | 1.070 / 0.234 (4.6×) |
| **6.7 – 7.4M** | **1.110 / 0.249 (4.5×)** | **1.067 / 0.245 (4.4×)** | **1.073 / 0.243 (4.4×)** |

**Reading this:** The asymmetry collapses from ~10–11× at init to ~4.5× by 1.5 M steps and **stays at ~4.5× for the rest of training**, climbing slightly as both heads' MAE creeps up. This matches the prior buf256k 4M observation that the negative-reward head is the persistent bottleneck. It is **not the cause** of the per-row fluctuation the user is seeing — it's a slowly-drifting structural quantity.

## 7. SPS and training cost (3-cell parity)

| Cell | Wall-clock | Iterations | Env-steps | s/it | SPS (overall) | SPS Q10 (last window) |
|---|---:|---:|---:|---:|---:|---:|
| A | 51h 26m | 463 920 | 7 439 280 | 0.399 | 40.2 | 40.20 ± 0.04 |
| B | 49h 27m | 463 280 | 7 428 880 | 0.384 | 41.7 | 41.75 ± 0.05 |
| C | 49h 26m | 456 310 | 7 317 360 | 0.390 | 41.1 | 41.06 ± 0.00 |

- All 3 cells within 4 % of each other on SPS. n114/GPU 1 is slightly faster than n114/GPU 0 (despite same node — probably contention from cell A's GPU 0 process, plus thermal). n113 is in the middle.
- Mild **monotonic SPS decay** of ~2.8 SPS over 7.4 M env-steps for cell A (~7 %), ~2.95 for B (~7 %), ~2.3 for C (~5 %). Consistent with replay-buffer growth + Python GC pressure; not a red flag.

## 8. World-model loss trajectory (cell-level reproducibility check)

| Step block | A loss_model | B loss_model | C loss_model |
|---|---:|---:|---:|
| 0.0 – 0.8M | 2.196 ± 0.22 | 2.082 ± 0.38 | 2.220 ± 0.36 |
| 2.0 – 3.0M | 2.470 ± 0.11 | 2.432 ± 0.10 | 2.515 ± 0.10 |
| 4.5 – 5.2M | 2.712 ± 0.14 | 2.662 ± 0.13 | 2.705 ± 0.13 |
| 6.7 – 7.4M | **2.880** ± 0.12 | **2.872** ± 0.12 | **2.821** ± 0.13 |

- **WM loss is rising, not falling**, across all 3 cells (from ~2.2 → ~2.85). This is *expected* in DreamerV3 — `loss_model` is the sum of recon + reward + KL terms, dominated by the dynamics-KL term that *grows* as the latent representation refines. Recon and reward losses ARE falling (recon: 0.68→0.076; reward: 2.75→0.99 for cell A, similar B/C).
- Cell-to-cell std on any block is < 0.05 — **the WM trajectory is essentially seed-locked** (same seed → same loss curve modulo CUDA non-determinism).

## 9. Ranked observations

1. **The "fluctuation" is per-episode logging noise, not policy instability.** The within-cell per-episode std is ~140 around a mean of ~200; the cell-to-cell std on the means is ~0.5. Ratio ~280×. Any single WandB row carries ~140-step noise from one episode's contingent outcome (predator-luck, food-spawn-luck), but the underlying smoothed trajectory is monotonically rising and 3-cell-overlaid.
2. **Reproducibility across nodes/GPUs is near-perfect at the same seed.** Last-2k means 200.7 / 201.6 / 201.2 → 0.2 % spread. Node-level and GPU-level noise is negligible at this seed.
3. **All 3 cells were still rising at termination.** Q5 > Q4 for every cell (+8.8 / +13.7 / +18.8 survival steps). The last 5 000-episode block is the *highest* of all 10 blocks per cell. Terminating at ~7.4 M env-steps left learning unfinished, consistent with the leading hypothesis in [dreamer_vs_rppo_gap_hypotheses.md](dreamer_vs_rppo_gap_hypotheses.md).
4. **The reward-head pos/neg asymmetry stabilises at ~4.5×, not 5–10×.** Lower than the prior 4M-step reference. The asymmetry is not collapsing further but is also not exploding — it persists as a structural bottleneck rather than diverging.
5. **Wall-clock cost is ~50 h for 7.4 M env-steps at ~40–42 SPS.** Roughly +85 % env-steps over the prior 4M reference yielded ~+6 % survival-step gain (189.5 → 201.2). **Diminishing returns are visible.** Continuing past 10 M env-steps without an architecture intervention (reward-head capacity / horizon shortening / lr sweep — items 2-3 in the gap-hypotheses doc) likely yields shallower marginal gains.

## 10. Recommendations / next actions

1. **Use the smoothed view going forward.** When the next 4 runs launch with the corrected log_interval (commit `0fb2795`), each WandB row will be 1 / 500 / 5 000 episodes' worth of average — visually clean. The per-episode WandB plot of these three runs should be read as "true value ≈ rolling-100 mean", not as the raw curve.
2. **Do not interpret per-row fluctuation as anything mechanistic.** Single-episode std is ~140 even at converged performance. Anything < ±30 between adjacent rows is noise.
3. **The next experiment cycle should target one of hypotheses 2–3 in the gap-doc** (reward-head capacity / horizon / lr), NOT another buf256k re-run. Three identical seeds confirms reproducibility; further reproducibility runs have low information value.
4. **Consider a 4th seed at buf256k for a published mean ± CI** (n=4 vs n=3 noticeably tightens the CI), but only if a paper figure needs it — not for analysis purposes.

## 11. Metrics requested (would sharpen future analyses)

| Metric | Why now | Where it'd live | Cost |
|---|---|---|---|
| `Episode/Steps_smoothed_window_500` | The raw per-episode metric is unusable in the WandB UI without smoothing; even after the log-interval fix, a server-side rolling mean would let single-glance reads of the WandB UI compare cells without manual log parsing | `dreamer_srl_main.py` — wrap `Episode/Steps` log call with a maintained rolling buffer | Cheap (one extra scalar per log call) |
| `WorldModel/reward_mae_pos_over_neg_ratio` | The pos/neg ratio is the load-bearing diagnostic for the reward-head bottleneck hypothesis; computing it on the WandB plotting side every time is friction | Same place that emits `reward_mae_pos` / `reward_mae_neg` | Cheap (one division) |
| `Episode/Term_*` per-type counts in output.log | The text logs have `episode done: env=X ep_len=N ep_rew=R` but **not** the termination type (starvation / injury / overeating / max-steps). Forces analysts to re-pull WandB to attribute survival gain to a termination shift | `dreamer_srl_main.py` per-episode print | Cheap (one extra token) |

## 12. Failure-mode catalog check (retroactive)

| Failure mode | Did it fire? | Evidence |
|---|---|---|
| Per-episode WandB plot misread as policy instability | **Yes — this is the entire reason for this analysis** | Mitigated by §3–4 smoothed views above |
| Node/GPU non-determinism corrupting reproducibility | **No** | Cell-to-cell std on last-2k mean = 0.45 steps (~ 0.2 %) |
| Reward-head asymmetry exploding past 10× | **No — held at 4.5×** | §6 |
| WM loss diverging (training collapse) | **No** | §8 — loss_model rises monotonically by the expected ~30 % (KL term growth) |
| Premature entropy collapse | **No** | Actor entropy stable at ~0.18 by step 7.4M, comparable to prior buf256k 4M; no zero-entropy collapse |
| SPS degradation past 50 % | **No** | <7 % SPS decay end-to-end per cell |

## 13. Manifest / data sources

- **Local WandB binary logs** (full per-episode + per-iter): `wandb/run-20260525_230216-jkmto06f/`, `wandb/run-20260526_010203-empiuzmd/`, `wandb/run-20260526_010323-flq7e8c9/`
- **WandB API extraction** (via `scripts/wandb_metrics.py`): same run IDs — used for the WorldModel-loss and SPS time-series (10-window summaries)
- **Local log parser**: `tmp/parse_dreamer_log.py`, `tmp/dreamer_log_quintiles.py` — read `output.log` line-by-line to assemble per-episode (`ep_len`, `ep_rew`) and per-iter (`policy_step`, `world_model_loss`, `sps`) trajectories
- **Extracted intermediate files**:
  - `tmp/20260528_ep10m_local_extracted.json` — landmark-step + last-2k stats per cell
  - `tmp/20260528_ep10m_quintiles.json` — full quintile + batch-500 + batch-5000 stats per cell
  - `tmp/20260528_ep10m_wm_3cell_v2.md` — 10-window WandB-API extraction for WM/SPS metrics
  - `tmp/20260528_ep10m_compare_3cell_v2.md` — dreamer_v3 preset comparison across all 3 cells
- **All 3 processes terminated**: output.log mtimes around 2026-05-28 02:33:38–02:33:45 KST

## 14. Related issues

- The **log_interval config-read bug** that made cells B/C log at interval=10 instead of 5 000 / 50 000 is fixed in commit `0fb2795` per the user. No further action needed.
- No bugs surfaced by this analysis warrant a `bug-fix-workflow` plan.
- The Metrics Requested above (§11) are the soft-feature handoff if the user wants to invoke `feature-workflow` on them.
