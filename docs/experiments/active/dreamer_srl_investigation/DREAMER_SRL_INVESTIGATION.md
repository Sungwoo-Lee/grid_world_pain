---
title: "Dreamer-SRL learning-quality snapshot: data-starved, gradient-starved, or healthy-but-slow? (H2b)"
topic: dreamer_srl_investigation
status: active
created: 2026-07-27
last_updated: 2026-07-27
wandb_tag: "dsrl_b0{3,4}_{M,XS}_dp1 / rppo_b0{3,4}_mc_dp1_n110"
---

# Dreamer-SRL Learning-Quality Snapshot (H2b) — Mode B, retroactive, runs in progress

> **Status**: ANALYZING (runs still training — ~22 h snapshot, NOT final)
> **Date**: 2026-07-27
> **Author**: experiment-analyzer (H2b axis of the five-way parallel dreamer_srl investigation)
> **Mode**: **Mode B — post-hoc.** No pre-registered predictions existed for this question; conclusions are correspondingly weaker than a pre-registered design would allow.
> **Related**: [[dreamer_srl_h1_speed_investigation]] (throughput axis), [[dreamer_srl_faithfulness_review]], [[dreamer_srl_settings_regime_critique]], [[gridworld_vs_dreamerv3_benchmarks_difficulty]] — sibling docs from the same parallel investigation, some in progress at time of writing.

---

## 1. Research Question

Our world-model agent (a DreamerV3-style learner, "Dreamer-SRL") is visibly learning to survive in the grid-world — but slowly in wall-clock terms, while a model-free recurrent PPO baseline blasts through a hundred times more environment experience in the same day. This snapshot asks: **from the training curves alone, WHY does Dreamer learn the way it does here?** Three candidate diagnoses:

1. **Data-starved per gradient step** — the model wants more fresh experience than it gets; its losses show it has already squeezed what it can from the data.
2. **Gradient-starved per datum** — experience arrives faster than the learner digests it; each collected step gets too few weight updates, so raising the replay ratio (more gradient steps per environment step) would speed learning per episode.
3. **Healthy-but-slow** — the learner is digesting its data fine and improving steadily; the binding constraint is simply how slowly it collects environment steps (throughput), not how it uses them.

We examine four live Dreamer runs (two environments — the "random-init 10×10" and "jump-attack 10×10" survival worlds, each with a Medium and an Extra-Small world model) about 22 hours into training, against two matched recurrent-PPO baselines launched the same day under the same reward-decay regime. A positive answer for any diagnosis must come from the *shape* of the curves (world-model loss vs. data seen, value-error lag, small-vs-large-model learning per episode), not from endpoints alone.

**Verdict preview:** healthy-but-slow, throughput-bound — with a mild gradient-starvation margin confined to the critic and to rare negative-reward (predator-contact) events. Details in §5–6.

---

## 2. Experimental Design (retroactive framing)

This is a snapshot of runs launched for other purposes (the decay-power-1.0 relaunch ladder), not a designed experiment. The de-facto factors:

### 2.1 Independent variables (de facto)

| Variable | Values | Rationale |
|----------|--------|-----------|
| Algorithm | Dreamer-SRL vs. recurrent PPO | model-based vs. model-free reference |
| Environment | basic03 "random-init 10×10" vs. basic04 "jump-attack 10×10" | two predator worlds, same size, both draw predator count from U{0,1,2} |
| Dreamer model size | M (640 dense units) vs. XS (256 dense units) | capacity vs. throughput trade |

### 2.2 Controlled variables (verified from each run's SAVED config — see §3)

- `decay_power: 1.0` in all six runs (fresh-skepticism check passed; none of the pre-2026-07-26 `decay_power 2.0` runs are mixed in).
- `replay_ratio: 0.0625` in all four Dreamer runs (logged effective replay ratio confirms 0.0615→0.0625 over training). With batch 16 × sequence 64 = 1024 frames per gradient step, this is **train_ratio 64**: each collected frame is replayed ~64× on average — the *data-rich / low-replay* end of the DreamerV3 paper's range.
- `num_envs: 128`, `max_steps: 500`, seq 64, batch 16, horizon 15 in all Dreamer runs; `num_envs: 128` in both rPPO runs.
- Seeds: Dreamer CLI `--seed 0` (all four); rPPO config seed 42 (both). **Single seed per cell — no seed-dispersion estimate is possible in this snapshot.**

### 2.3 Confounds & limitations

| Confound | Affected | Severity | Mitigation |
|----------|----------|----------|------------|
| Runs still training (~22 h in) | all | High | framed as snapshot; no "final" claims |
| One seed per cell | all | High | no CI possible; treat all deltas as point estimates |
| Episode metrics are **rolling windows** (smoothing 5000 episodes, log interval 200) | all | Med | curves are heavily smoothed; sudden transitions appear as ramps; matched-point reads inherit ±ε from window lag |
| Predator count drawn U{0,1,2} → ~1/3 of episodes have **no predator** → survival is a bimodal mixture | all | Med | **no per-episode data is reachable**: WandB logs only window means/std/min/max; training-time recordings are 3 *fixed-seed* eval episodes per checkpoint (identical env draws every checkpoint — verified in the run log), so a predator-conditioned decomposition is impossible from logged data. Stated prominently; see Metrics Requested §7. Window `Steps_Std ≈ 1.5×` mean (e.g. 186 vs 124) with Min=1/Max=500 is consistent with the mixture. |
| rPPO launched 45 min after Dreamer | wall-clock comparison | Low | noted; rates computed per-hour |
| Dreamer M and XS on different GPUs/nodes | throughput comparison | Low | SPS taken from each run's own log |

---

## 3. Run Manifest (verified)

All configs read from the run's **saved** `results/.../models/` copy, not from the live `configs/` tree.

| Run name | WandB ID | State @ snapshot | decay_power | seed | replay_ratio | num_envs | Model | Results dir |
|---|---|---|---|---|---|---|---|---|
| dsrl_b03_M_dp1 | `qf721efb` | running, 175.4k eps / 16.7M steps | 1.0 | 0 | 0.0625 | 128 | M (640u) | `results/JAX_DreamerSRL/20260726-042618_dsrl_b03_M_dp1` |
| dsrl_b03_XS_dp1 | `fwze090k` | running, 267.0k eps / 24.9M steps | 1.0 | 0 | 0.0625 | 128 | XS (256u) | `results/JAX_DreamerSRL/20260726-042620_dsrl_b03_XS_dp1` |
| dsrl_b04_M_dp1 | `lco3j3w8` | running, 169.8k eps / 16.0M steps | 1.0 | 0 | 0.0625 | 128 | M (640u) | `results/JAX_DreamerSRL/20260726-042624_dsrl_b04_M_dp1` |
| dsrl_b04_XS_dp1 | `g148xn28` | running, 255.8k eps / 25.1M steps | 1.0 | 0 | 0.0625 | 128 | XS (256u) | `results/JAX_DreamerSRL/20260726-042624_dsrl_b04_XS_dp1` |
| rppo_b03_mc_dp1_n110 | `rmosum4z` | running, 16.28M eps / 2.72B steps | 1.0 | 42 | — | 128 | rPPO | `results/JAX_RecurrentPPO/20260726-051209_rppo_b03_mc_dp1_n110` |
| rppo_b04_mc_dp1_n110 | `y5eqvx89` | running, 17.00M eps / 2.69B steps | 1.0 | 42 | — | 128 | rPPO | `results/JAX_RecurrentPPO/20260726-051300_rppo_b04_mc_dp1_n110` |

Dreamer runs launched 2026-07-26 04:26, rPPO 05:12 (KST). Snapshot pulled 2026-07-27 ~02:40. Working extractions: `tmp/h2b_csv/*.csv`, `tmp/20260727_023544_h2b_analysis_output.txt`.

---

## 4. Results

Headline metric is **survival steps** throughout (project convention). All values from WandB histories (rolling-window means unless noted).

### 4.1 Primary: survival at three exchange rates (basic03)

Three honest ways to compare a slow-collecting model-based learner against a fast model-free one; they disagree, and the disagreement IS the finding.

| Comparison point | Dreamer M | Dreamer XS | rPPO | Read |
|---|---|---|---|---|
| **Matched EPISODES** (~175k eps) | **123.4** steps, food 24.7 | 120.6, food 25.1 | 50.3, food 3.6 | Dreamer **2.4×** better per episode |
| **Matched ENV STEPS** (~16.5M steps) | **123.4** (@175k eps) | 125.4 (@203k eps) | 59.6 (@360k eps) | Dreamer **~2.1×** better per env step |
| **Matched WALL-CLOCK** (~22 h) | 123.4 | 133.2 (@267k eps) | **179.8** (@16.3M eps) | rPPO **1.35×** ahead per hour |

basic04 (jump-attack), same pattern:

| Comparison point | Dreamer M | Dreamer XS | rPPO |
|---|---|---|---|
| Matched EPISODES (~170k) | **116.6**, food 23.1 | 123.1, food 26.4 | 40.8, food 1.8 |
| Matched ENV STEPS (~16M) | 116.6 | **126.1** | 55.0 |
| Matched WALL-CLOCK (~22 h) | 116.6 | 139.9 | **166.1** |

**Exchange rates (basic03, per wall-clock hour):** rPPO collects ~126M env steps/h (~35.8k SPS) vs. Dreamer M ~0.76M/h (210 SPS) and XS ~1.13M/h (313 SPS) — a **113–170× collection-throughput gap**. In episodes: rPPO ~757k eps/h vs. Dreamer 7.9–12.0k eps/h (**63–96×**). Dreamer's ~2.1–2.4× sample-efficiency advantage is real but two orders of magnitude too small to survive the throughput gap; rPPO passes Dreamer's current survival level within its first ~2–3 h of training.

Verification against prior claims (2026-07-26 matched-episode analysis): Dreamer M 120.2 vs rPPO 51.9 @170k eps → reproduced here as 123.4 vs 50.3 @175k; food 24 vs 4 → 24.7 vs 3.6; predator-only hits/100 steps 0.82 (M endpoint) and 0.34 (rPPO plateau) → reproduced (0.82, 0.32). Consistent.

### 4.2 World-model health (temporal, windowed by env steps)

Dreamer M, basic03 (8 equal windows over 0→16.7M steps); XS and basic04 shapes are qualitatively identical (full tables in `tmp/20260727_023544_h2b_analysis_output.txt`):

| Window (M steps) | WM total loss | recon | dyn-KL | rep-KL | reward-MAE | reward-MAE(neg) | cont-acc | latent entropy |
|---|---|---|---|---|---|---|---|---|
| 0–2.1 | 2.211 | 0.052 | 1.162 | 0.232 | 0.969 | 1.213 | 0.995 | 1.06 |
| 6.3–8.4 | 1.881 | 0.031 | 0.975 | 0.195 | 0.337 | 0.465 | 0.999 | 1.16 |
| 14.6–16.7 | 1.867 | 0.027 | 0.976 | 0.195 | 0.275 | 0.376 | 0.999 | 1.16 |

- **WM total loss plateaus by ~8M env steps** (−0.7% over the last 8M). KL terms flat, continuation accuracy saturated at 0.999, latent entropy stable ~1.16 (no collapse, no blow-up).
- **Reconstruction still creeps down** (−13% over the last 8M) — slow refinement, not underfitting.
- **The one still-moving WM error is reward prediction on negative rewards**: reward-MAE(neg) −19% over the last 8M and still falling. Negative rewards are dominated by rare predator-contact events — the WM's residual error is concentrated exactly on the rare, high-stakes transitions.
- **M vs XS**: XS runs systematically higher losses (WM total 2.04 vs 1.87; recon 0.037 vs 0.027; reward-MAE(neg) 0.74 vs 0.38 at comparable data) — the smaller model fits the data measurably worse, yet (see §4.1) survives just as long. WM fit quality above the M-level is not translating into behavior.

### 4.3 Actor/critic health (temporal)

Dreamer M basic03, same windows:

| Window (M steps) | policy entropy (nats) | value MAE | mean advantage | mean return (norm'd λ) | return-normalizer scale |
|---|---|---|---|---|---|
| 0–2.1 | 0.168 | 9.95 | −0.028 | −18.2 | 123.4 |
| 6.3–8.4 | 0.117 | 5.64 | −0.005 | −0.76 | 88.4 |
| 14.6–16.7 | 0.131 | 4.50 | −0.002 | +2.26 | 69.7 |

- **Entropy**: low (0.12–0.17 vs. max ln 6 ≈ 1.79 for 6 actions) but stable — near-deterministic, not collapsing further, and ticking *up* late. XS is more exploratory and **rising** late in training (basic03: 0.24→0.36; basic04: 0.26→0.49) — no premature collapse anywhere; if anything the XS actor is re-opening exploration as returns improve.
- **Value MAE still falling steadily** (−20% over the last 8M steps in M; XS 4.16→3.11) — the critic clearly lags the world model: the WM plateaued at ~8M steps while value error keeps dropping through the snapshot.
- **Return normalizer** (the advantage divisor, `max(1/1e8, p95−p5)` of EMA return percentiles): smooth decay 123→70 as the return distribution tightens and shifts up from strongly negative. Healthy; no scale explosion/collapse. Mean imagined return crosses from −18 to positive.

### 4.4 Learning dynamics (episode-level, rolling windows)

Dreamer survival curves are **monotonically rising with no plateau** at snapshot end in all four runs (M b03: 48→120 over 175k eps, last-window slope still ~+3 steps/21k eps; XS b03: 41→133, still ~+7/33k). rPPO is essentially **plateaued** (b03: 176→179 over the last 6M episodes; b04 flat at ~166–167, last window slightly down).

**A capacity-dependent take-off:** XS shows a delayed breakthrough at ~100k episodes (basic03: window means 67→94 between 70–136k eps; basic04: 66→107 between 99–130k eps, with food/episode jumping 9.6→21.7) — the same food-seeking transition M makes by ~26–48k episodes. The larger model reaches the breakthrough ~2–3× earlier in episode count, but **after** the transition both sizes converge to the same trajectory (b03 @175k: M 123.4 vs XS 120.6; b04: XS actually ahead, 123.1 vs 116.6). Capacity buys transition speed, not the ceiling.

rPPO train-side health (context): entropy flat at ~0.68, value loss flat ~0.23, grad norm 0.38→0.25 — a converged/plateaued learner making no further per-episode progress.

### 4.5 Terminal-reason decomposition over training (composition artifact made rigorous)

Shares of episode terminations (Injury = killed by damage, Starvation, MaxSteps = survived the 500-step cap):

| Run @ point | Term_Injury | Term_Starvation | Term_MaxSteps | hits/100 steps (all predators) |
|---|---|---|---|---|
| dsrl_b03_M early (5–26k eps) | 0.64 | 0.35 | 0.01 | 4.81 |
| dsrl_b03_M @175k | 0.68 | 0.17 | 0.15 | 2.85 |
| dsrl_b03_XS @267k | 0.71 | 0.11 | 0.18 | 2.80 |
| rppo_b03 @176k eps (matched) | 0.45 | 0.55 | 0.00 | 3.16 |
| rppo_b03 @16.3M eps (plateau) | 0.42 | 0.34 | 0.25 | 1.61 |
| dsrl_b04_M @170k | 0.69 | 0.17 | 0.14 | 3.03 |
| rppo_b04 @168k eps (matched) | 0.41 | 0.59 | 0.00 | 2.45 |
| rppo_b04 @17.0M eps (plateau) | 0.48 | 0.28 | 0.24 | 1.66 |

The rigorous statement of the composition artifact: **Dreamer's higher injury share (0.68 vs 0.45 at matched episodes) is not worse predator handling — it is solved starvation.** At matched episodes Dreamer eats 24.7 food/episode vs rPPO's 3.6, so rPPO episodes end by starvation (0.55) before predators can matter. Exposure-normalized, Dreamer at matched episodes takes *fewer* predator hits per 100 steps than rPPO (2.85 vs 3.16 on basic03). Both algorithms drive hits/100 down over training (Dreamer 4.8→2.8 and still falling; rPPO plateau 1.6) — rPPO's plateau avoidance is better per-step, but it needed ~100× more episodes to get there. Term_MaxSteps (full 500-step survivals) grows in lock-step with survival for both, confirming the bimodal mixture: means mix ~0-predator full survivals with predator deaths (window Steps_Std ≈ 186–205 against means of 117–182, Min 1 / Max 500 pinned in every window).

---

## 5. Analysis — the key tension quantified

**Is Dreamer data-starved per gradient step?** No. At replay ratio 0.0625 each collected frame already receives ~64 gradient exposures (0.0625 grad-steps/env-step × 1024 frames/grad-step), and the WM's total loss has plateaued on the data it sees (−0.7% over the last 8M env steps) with saturated continuation accuracy and stable KLs. A learner starved of data per gradient step would show WM loss still dropping steeply per datum; it doesn't.

**Is it gradient-starved per datum (would higher replay ratio help)?** Only at the margin. Two error signals are still clearly declining while the WM sits flat: **value MAE** (−20%/8M steps) and **reward-MAE on rare negative events** (−19%/8M steps). More gradient steps per datum would plausibly accelerate exactly these two — the critic and the rare-event reward head — and thus buy *some* per-episode speedup. But the strongest counter-evidence is the M-vs-XS pair: XS fits the data measurably worse at every loss (WM 2.04 vs 1.87), gets the identical 64 replays per frame, and still reaches the **same survival at matched episodes** (120.6 vs 123.4 @175k on basic03; ahead on basic04). If gradient-per-datum (or capacity) were binding, the better-fit larger model should hold a durable behavioral lead — it doesn't past the take-off.

**Verdict: healthy-but-slow, throughput-bound.** All four Dreamer runs are improving monotonically with no plateau, healthy actor entropy (no collapse; XS rising), a smoothly decaying return normalizer, and a WM that fits its data. Learning tracks **episodes experienced**: at matched wall-clock, XS leads M purely by collecting 1.5× more episodes (313 vs 210 SPS). The binding constraint is the 113–170× env-step collection gap to rPPO — a systems/throughput problem (the H1 axis, [[dreamer_srl_h1_speed_investigation]]), not a learning-quality problem. Raising replay ratio is a second-order lever (critic + rare-event refinement); raising SPS is the first-order one.

Caveats on this verdict: single seed per cell; rolling-window smoothing hides fast transitions; runs unfinished; per-episode (predator-conditioned) decomposition impossible from current logs — the bimodal survival mixture means all means understate behavior in the ~2/3 of episodes that actually contain predators.

---

## 6. Conclusions (snapshot — subject to revision when runs finish)

1. **Regime call: healthy-but-slow.** WM converged-and-tracking, actor/critic healthy, survival rising without plateau; the bottleneck is data collection throughput, not data digestion.
2. **Dreamer is ~2.1–2.4× more sample-efficient than rPPO** (per episode and per env step) on both predator worlds, but rPPO's ~126M env steps/hour vs Dreamer's ~0.8–1.1M makes rPPO 1.35× ahead at equal wall-clock and it reached Dreamer's current level within ~3 h.
3. **Model capacity buys take-off speed, not the ceiling**: M reaches the food-seeking breakthrough ~2–3× earlier in episodes; XS converges to the same (or better) survival after its ~100k-episode take-off, at 1.5× the collection rate.
4. **The remaining learning margin is rare-event-shaped**: the only still-declining WM error is reward prediction on rare negative (predator-contact) transitions, and the critic still lags the WM. A modest replay-ratio increase or prioritized replay of negative-reward sequences is the most defensible learning-side intervention to test — as a pre-registered follow-up, not a mid-run tweak.
5. **Composition artifact confirmed and quantified**: Dreamer's high injury-termination share is a consequence of solving starvation; exposure-normalized predator hits favor Dreamer at matched episodes (2.85 vs 3.16 /100 steps) and rPPO only at its 100×-more-episodes plateau (1.6).

**TODO (Mode B → Mode A):** if the replay-ratio question matters strategically, re-run as a pre-registered design (rr ∈ {0.0625, 0.25, 1.0} × ≥3 seeds, XS model, basic03) with per-predator-count eval — surfaced to the user.

---

## 7. Metrics Requested

| Subfield | Request 1 | Request 2 |
|---|---|---|
| **Metric** | `Episode/Steps_hist` or per-episode terminal log keyed by spawned predator count (0/1/2) | `WorldModel/reward_mae` computed separately on replayed frames ≤N steps before a negative-reward event |
| **Why now** | Survival on basic03/04 is a bimodal mixture (U{0,1,2} predators); rolling means cannot be predator-conditioned, so all cross-algorithm survival claims carry an unquantified composition term. Training-time recordings reuse 3 fixed eval seeds and cannot substitute. | The only still-improving WM error is reward-MAE(neg); a pre-event-window version would show directly whether the WM anticipates predator contact or merely explains it post hoc — the key quantity for judging whether more replay would help behavior. |
| **Where it'd live** | `src/utils/episode_logging.py` (episode accumulator already tracks terminal reasons) | `src/algorithms/dreamer_srl/loss.py` (reward head already splits pos/neg MAE) |
| **Cost** | cheap (3 extra scalars per window, conditioned on spawn count) | moderate (needs event-relative masking over the sampled sequence) |

---

## 8. Related Issues / Cross-links

- Throughput axis (why 200–310 SPS): [[dreamer_srl_h1_speed_investigation]] (`docs/develop/active/diagnosis/dreamer_srl_h1_speed_investigation.md`, sibling, in progress).
- Implementation faithfulness vs sheeprl: [[dreamer_srl_faithfulness_review]] (`docs/reviews/dreamer_srl_faithfulness_review.md`, sibling, in progress).
- Settings-regime critique (replay ratio / train_ratio placement vs the paper): [[dreamer_srl_settings_regime_critique]] (`docs/project/critiques/dreamer_srl_settings_regime_critique.md`, sibling, in progress).
- Difficulty framing vs published DreamerV3 benchmarks: [[gridworld_vs_dreamerv3_benchmarks_difficulty]] (`docs/project/critiques/gridworld_vs_dreamerv3_benchmarks_difficulty.md`, sibling, in progress).
- No code bugs surfaced by this analysis. The fixed-seed training-time eval (3 identical env draws per checkpoint) is by design post-commit b228117 but limits eval curves to 3 env configurations — noted for the record, not filed as a bug.

---

## Appendix — extraction provenance

- Histories pulled via WandB API (entity `sungwoolee`, project `grid_world_pain`) on 2026-07-27 ~02:40 KST; CSVs in `tmp/h2b_csv/`; extraction scripts `tmp/20260727_023544_h2b_extract*.py`; full windowed tables `tmp/20260727_023544_h2b_analysis_output.txt`.
- Saved-config verification: `results/.../models/agent_config.yaml` + `env_config.yaml` (Dreamer), `results/.../models/config.yaml` (rPPO) — all six show `decay_power: 1.0`, `num_envs: 128`; Dreamer `replay_ratio: 0.0625`.
- Return-normalizer semantics verified against `src/algorithms/dreamer_srl/utils.py` (invscale = `max(1/1e8, p95 − p5)` of EMA return percentiles, sheeprl convention).
