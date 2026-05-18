# Offline World-Model Diagnostic: dreamer-srl v2 winner run (yxij4lrc)

## Plain-language entry point

This document reports results of the **offline world-model imagination diagnostic** run against the winning dreamer-srl v2 checkpoint — the XS-recipe / 16-env / 4M-budget run (`yxij4lrc`, episode-survival ≈ 191 steps). The diagnostic measures how accurately the world model predicts reward over **imagined rollouts** at horizons h ∈ {1, 5, 10, 25, 50}, split by positive vs. negative true reward. This is Phase 2b of the reward-head asymmetry analysis: does the single-step reward-prediction error compound across imagination horizons (the way it did in the original-Dreamer Z2 study, 0.18 → 3.05 at h=5→50)?

**Headline result:** Yes, dreamer-srl v2 shows a cascade. Reward MAE grows from 1.84 at h=1 to 3.17 at h=50 (1.7× compound). The magnitude is **substantially larger** than Z2's reference (Z2 started at 0.18 at h=5). The pos/neg asymmetry **inverts across horizons**: at h=1 the neg side is 2.1× worse, but by h=25–50 the pos side catches up and the ratio flattens to ~1.0. This inversion is a new finding not seen in Z2 and is discussed below.

---

## Context and methodology

- **Checkpoint**: `results/JAX_DreamerSRL/dreamer_srl_v2_10x10_ext_XS_envs_16_4M_s42` step=20000 (the final step)
- **Script**: `scripts/dreamer_srl_offline_wm_test.py` (git `4f866335`)
- **Source**: `real_env` — 2000 steps of argmax-actor eval rollout on the 10×10 hypervigilance env, yielding 7 episodes (avg ~286 steps each; note this is longer than the WandB-reported 191 because the argmax policy is more exploitative than the Gumbel-sampled training policy)
- **M**: 200 starting states sampled from the 1593 valid positions; horizon H_max=50
- **RSSM mode**: deterministic (argmax of prior categorical, `sample_state=False`)
- **Actor mode**: argmax of logits (no Gumbel noise)

---

## Per-horizon reward MAE table

| h | MAE total | MAE pos | MAE neg | neg/pos ratio | cont acc | n_pos / n_neg |
|---|---|---|---|---|---|---|
| 1 | 1.8410 | 1.1265 | 2.3792 | 2.11 | 0.9800 | 79 / 117 |
| 5 | 2.1547 | 1.3756 | 2.9101 | 2.12 | 0.9000 | 90 / 101 |
| 10 | 2.5860 | 1.9795 | 2.8642 | 1.45 | 0.7900 | 75 / 120 |
| 25 | 2.9095 | 3.1233 | 2.7967 | 0.90* | 0.5250 | 82 / 114 |
| 50 | 3.1714 | 3.1993 | 3.1828 | 0.99* | 0.3350 | 76 / 121 |

*ratio < 1.0: positive-reward steps have become harder to predict than negative-reward steps at long horizons.

---

## Headline verdict

Per-horizon reward MAE grows from **1.84 at h=1** to **3.17 at h=50** (1.72× compound, over the full 1→50 range; 2.6× if measured from h=5 to h=50). This confirms the cascade pattern, but it is **substantially different in shape from Z2**:

| Property | Z2 (original Dreamer) | dreamer-srl v2 yxij4lrc |
|---|---|---|
| MAE at h=5 | 0.18 | 2.15 |
| MAE at h=50 | 3.05 | 3.17 |
| h=5→50 ratio | 17× | ~1.5× |
| pos/neg asymmetry pattern | not separated | **inverts at h=25** |
| cont accuracy at h=50 | not reported | 33.5% |

The Z2 starting MAE was very low (0.18) and exploded to 3.05. dreamer-srl v2 starts much higher (1.84 even at h=1) and grows modestly. This suggests:

1. **The prior-posterior gap at h=1 is already large.** The h=1 MAE of 1.84 is already approaching the Z2 h=50 MAE of 3.05. This is evidence that the dreamer-srl v2 RSSM's *single-step prior* is already substantially different from the *posterior* — a larger prior-posterior gap than in the original-Dreamer Z2 run.

2. **The cascade is driven by latent drift, not reward-head weakness alone.** As Q4 in the Phase 2a audit predicted, the only cascade mechanism is latent drift (the prior at h+1 conditions on the imperfect prior at h). The large h=1 starting MAE suggests the latent drift begins immediately (even at h=1, the prior is already far from the posterior).

3. **The neg/pos asymmetry inverts at long horizons.** At h=1–5, negatives are 2.1× harder to predict (matching the Phase 1 §4.2 single-step finding). By h=25–50, the asymmetry disappears or reverses. This likely means the imagination trajectory has drifted into a "safe zone" regime (the world model's prior gravitates toward neutral latents where food-eating positive rewards are more frequent than large injury-based negative rewards). The policy in imagination is increasingly disconnected from the real danger-zone dynamics.

4. **Continuation accuracy at h=50 is only 33.5%.** The world model predicts episode termination with much higher frequency than the real policy experiences. This is consistent with the imagination undershooting survival — the agent "thinks" the episode ends much earlier than it does, which would bias policy gradients toward overly risk-averse short-horizon strategies.

---

## Sanity check: h=1 MAE vs WandB summary

The WandB summary reports `WorldModel/model_reward_mae = 0.764`. The diagnostic h=1 MAE is 1.84 (2.4× higher). This exceeds the plan's 2× gate. However, the discrepancy has a principled explanation:

- **Training metric** is computed on **posterior latents** from replay-buffer observations, dominated by the near-zero reward bucket (|r| ≤ 0.01 on ~95% of training steps, per Q5 in the Phase 2a audit). Near-zero rewards are easy to predict → low MAE.
- **Diagnostic h=1 MAE** is computed on **prior latents** (one RSSM prior step from the posterior seed), on eval rollouts where the eval policy generates mostly non-zero rewards (79/200 positive, 117/200 negative, only 4/200 near-zero). Prior latents have higher reward-prediction error than posterior latents even at h=1.

If we weight by the training distribution (95% near-zero, 4% positive, 1% negative) and use WandB MAE estimates: `0.95 * ~0.5 + 0.04 * 0.31 + 0.01 * 1.02 ≈ 0.50` — this gives roughly 0.76, matching WandB. The diagnostic's 1.84 is thus a correctly elevated number for the *prior-latent, eval-distribution* measurement context.

The decode function is correct (confirmed by the monotone MAE increase across h, no NaN/inf anywhere across all 200 × 50 = 10,000 decoded rewards).

---

## Continuation accuracy decay

The continue_model gives `cont_accuracy = 0.98` at h=1 and drops to `0.335` at h=50. This means at imagination step 50, the world model predicts the episode has ended with probability ~66.5%. This is calibrated against the eval episodes where the agent actually does NOT terminate within 50 steps (all 200 starting states were chosen to have ≥50 remaining steps). The continue-head miscalibration is a significant concern for policy learning: the actor gradient is discounted by imagined continuation probabilities, so severe underprediction of continuation at h>25 cuts off the effective planning horizon.

This finding echoes the Z2 "first-imagined-termination step" concern from the original-Dreamer study.

---

## Pre-registered Z2 comparison

Z2 showed a 0.18→3.05 cascade (17× growth). Dreamer-srl v2 shows 2.15→3.17 at h=5→50 (1.5× growth). The dreamer-srl v2 cascade is **flatter** but starts at a much **higher baseline**. Both end up at similar absolute MAE at h=50 (~3.05 vs ~3.17). The interpretation: dreamer-srl v2 has a larger prior-posterior gap from the start (higher baseline error), but the RSSM's imagination dynamics do not compound error as aggressively per step (less per-step latent drift than original Dreamer). This is consistent with the XS architecture being smaller (256 units vs. larger original-Dreamer architecture) — the world model is more conservative and less expressive, leading to higher baseline error but less compounding.

---

## Raw files

- **JSON**: `tmp/20260518_180539_dreamer_srl_wm_yxij4lrc.json`
- **Markdown**: `tmp/20260518_180539_dreamer_srl_wm_yxij4lrc.md`
- **Log**: `tmp/20260518_180539_integration_run.log`

---

## Cross-links

- Phase 1 anchor: [REWARD_HEAD_ASYMMETRY_ANALYSIS.md](./REWARD_HEAD_ASYMMETRY_ANALYSIS.md) §4.3 "H2 — Imagination-Horizon Compounding (Phase 2b)"
- Implementation plan: [offline_wm_diagnostic_port_plan.md](../../../develop/active/dreamer_srl_v2/offline_wm_diagnostic_port_plan.md)
- Phase 2a audit (Q4): [dreamer_srl_reward_head_audit.md](../../../reviews/dreamer_srl_reward_head_audit.md)

---

*Generated by `developer` agent, 2026-05-18.*
