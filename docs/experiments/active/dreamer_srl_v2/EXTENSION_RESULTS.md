---
title: "Dreamer-SRL v2 — extension training comparison (5×5+predator and 10×10 hypervigilance) vs sheeprl baselines"
topic: dreamer
status: active
created: 2026-05-15
last_updated: 2026-05-15
wandb_tag: dreamer_srl_v2_extension
phase: post-hoc-comparison
cross_links:
  - docs/experiments/active/dreamer_srl_v2/PARITY_LAUNCH_V2.md
  - docs/develop/active/dreamer_srl_v2/IMPLEMENTATION_PLAN.md
  - docs/diary/2026-05-12.md
  - docs/diary/2026-05-14.md
---

# Dreamer-SRL v2 — extension training comparison (5×5+predator and 10×10 hypervigilance) vs sheeprl baselines

## 1. Context (plain language, ~280 words)

**The question.** The JAX rebuild of the DreamerV3 model-based reinforcement-learning agent (the "dreamer-srl" track) passed its parity gate on 2026-05-15 on the easiest grid-world task — a 5×5 food-only world with no predator — saturating the survival-step cap (501) on every seed (see [PARITY_LAUNCH_V2.md](./PARITY_LAUNCH_V2.md)). The natural next question is: does the same rebuild, with the same XS-size (smallest preset) hyperparameters, **scale to the two harder tasks the project actually wants to publish on**? Those are (a) the **5×5 + predator environment** ("predator interval 3, food nutrition gain 18" — a small grid where a hostile NPC patrols and the agent must eat food while staying alive), and (b) the **10×10 hypervigilance environment** ("interoceptive nociception" task — a larger grid with four hidden predators in bushes, interoceptive pain signals smoothed over time via a delay kernel, and the published research direction's headline task). For each task we compare a single dreamer-srl seed against the existing **sheeprl baseline** — sheeprl being the reference PyTorch DreamerV3 implementation, the parity target. The headline metric is **survival steps** (`Game/ep_len_avg`, mean episode length in env steps); the 5×5 env caps at 500 steps and the 10×10 env caps at 500 steps too, so the absolute survival ceiling is the same for both tasks. The verdict window is the last 20% of training (env-steps 160,000 to 200,000).

**The headline.** Dreamer-srl **underperforms sheeprl on both extension tasks**. On 5×5+predator, dreamer-srl ends the last-20% window at survival 235.15 steps vs sheeprl's 268.35 — a gap of ~33 survival steps, or 12% of the ceiling. On 10×10 hypervigilance, dreamer-srl ends at 69.39 vs sheeprl's 106.19 — a gap of ~37 survival steps, or 35% of sheeprl's plateau. Both dreamer-srl trajectories are still climbing at end-of-training (5×5+pred more steeply than 10×10), so part of the gap may be a learning-rate-of-progress issue rather than an outcome ceiling. But on a fair-apples comparison at the **same step budget (200k env steps)**, sheeprl is the better policy on both harder tasks.

**Why this matters.** The parity verdict was on the easiest task; this comparison stress-tests whether the rebuild generalizes. The two most likely explanations for the gap, in ranked order, are **(H1) the launch invocation used `--num-envs 1` for dreamer-srl vs `num_envs=4` for sheeprl** — a 4× difference in env-collection parallelism that affects sample diversity, replay-buffer freshness, and the prefill phase — and **(H2) capacity ceiling on 10×10** (the model is XS-size and the 10×10 task has 27 observation channels vs the 5×5 food-only's 19, so the model may be under-parameterized for the harder visual / multi-predator structure). The two are **not mutually exclusive** and recommendations below address both.

**Why this is post-hoc.** No pre-registered hypothesis existed for these two runs — they were launched as extension probes of the parity-gate-passing dreamer-srl rebuild. Conclusions here are weaker than for a pre-registered design; the recommended next experiments (§7) re-frame them as proper pre-registered comparisons.

---

## 2. Setup

**The 4 runs compared.** All on 200,000 env steps (the 5x5+pred sheeprl baseline is 199.5k actual env-steps but logs through 200k bucket; 10×10 sheeprl is the same). The dreamer-srl runs were launched at 2026-05-15 14:42 (this comparison fires after both completed cleanly).

| Run label | Algorithm | WandB ID | WandB URL | num_envs | Env config | Agent config | Seed | Wall-clock |
|---|---|---|---|---|---|---|---|---|
| **5×5+pred, dreamer-srl** | dreamer-srl v2 (JAX) | [`xvrtjlat`](https://wandb.ai/sungwoolee/grid_world_pain/runs/xvrtjlat) | [link](https://wandb.ai/sungwoolee/grid_world_pain/runs/xvrtjlat) | 1 | `configs/experiment/basic/01-5X5_PredInterval3_NutGain18.yaml` | `configs/dreamer_srl/01_food_only.yaml` (XS, `learning_starts=1024`) | 42 | 12249 s (3.40 h) |
| **5×5+pred, sheeprl** | sheeprl dreamer_v3 XS (PyTorch) | [`dbh64fcy`](https://wandb.ai/sungwoolee/grid_world_pain_sheeprl_test/runs/dbh64fcy) | [link](https://wandb.ai/sungwoolee/grid_world_pain_sheeprl_test/runs/dbh64fcy) | 4 | (same task) | sheeprl `algo=dreamer_v3_XS`, `algo.learning_starts=1024` | 42 | 46676 s (12.97 h) |
| **10×10 intero, dreamer-srl** | dreamer-srl v2 (JAX) | [`405f0555`](https://wandb.ai/sungwoolee/grid_world_pain/runs/405f0555) | [link](https://wandb.ai/sungwoolee/grid_world_pain/runs/405f0555) | 1 | `configs/experiment/hypervigilance/01-interoNocicept.yaml` | `configs/dreamer_srl/01_food_only.yaml` (XS, `learning_starts=1024`) | 42 | 12235 s (3.40 h) |
| **10×10 intero, sheeprl** | sheeprl dreamer_v3 XS (PyTorch) | [`yt1uts22`](https://wandb.ai/sungwoolee/grid_world_pain_sheeprl_test/runs/yt1uts22) | [link](https://wandb.ai/sungwoolee/grid_world_pain_sheeprl_test/runs/yt1uts22) | 4 | (same task) | sheeprl `algo=dreamer_v3_XS`, `algo.learning_starts=1024` | 42 | 46253 s (12.85 h) |

**Local log paths** (the authoritative source for this analysis):
- dreamer-srl 5×5+pred: `wandb/run-20260515_144208-xvrtjlat/files/output.log`
- dreamer-srl 10×10 intero: `wandb/run-20260515_144213-405f0555/files/output.log`
- sheeprl 5×5+pred: `logs/runs/dreamer_v3/gwp_5x5_pred/wandb/run-20260512_153005-dbh64fcy/files/output.log` + cloud history
- sheeprl 10×10 intero: `logs/runs/dreamer_v3/gwp_10x10_intero/wandb/run-20260512_153005-yt1uts22/files/output.log` + cloud history

**Extraction artifacts**:
- Sheeprl ep_len_avg + SPS time-series (40 points per run, 5k-step granularity): [`tmp/20260515_dreamer_srl_extension_baselines.json`](../../../../tmp/20260515_dreamer_srl_extension_baselines.json)
- Dreamer-srl per-episode parse (1860 + 3074 episodes): [`tmp/20260515_dreamer_srl_extension_dreamersrl_traj.json`](../../../../tmp/20260515_dreamer_srl_extension_dreamersrl_traj.json)

**Two important caveats up front.**
1. Each comparison cell is **n=1 seed** per side. Per-seed variance is unknown — particularly important because the parity launch surfaced a "seed-0 took 71k env-steps to break free vs ~14.7k for other seeds" reproducibility-of-trajectory issue (see [PARITY_LAUNCH_V2.md §8](./PARITY_LAUNCH_V2.md#8-diagnostic--why-did-seed-0-take-71k-env-steps-to-break-free)). The gaps reported here could shift with more seeds.
2. **Wall-clock is not comparable** — the sheeprl baselines ran ~4× longer (12.9 h vs 3.4 h) primarily because sheeprl's PyTorch SPS sat around 600 env steps/sec vs dreamer-srl's 16 SPS. The comparison here is **fixed-step-budget, not fixed-wall-clock**: both reach env-step 200,000. Throughput differences do not enter the policy-quality comparison directly.

---

## 3. Results — per-run trajectories and side-by-side comparison

### 3.1 5×5+predator side-by-side (40k-step windows)

The headline metric is `Game/ep_len_avg` (mean episode length, max 500 steps before truncation). Sheeprl logs this every 5,000 env steps (a sample averaged across the 4 parallel envs); dreamer-srl logs an event per episode-end, which we bucket into 40k-step windows below. The pre-registered last-20% window is env-steps 160,000–200,000.

| Window (env-steps) | dreamer-srl (`xvrtjlat`) | sheeprl (`dbh64fcy`) | Gap (sheeprl − dreamer-srl) |
|---|---|---|---|
| 0–40k | **55.07** (n=739 episodes, max 135) | **60.56** (n=7 samples, max 83.72) | +5.49 |
| 40k–80k | **81.91** (n=492, max 423) | **136.34** (n=8, max 225.27) | +54.43 |
| 80k–120k | **150.85** (n=267, max 501) | **197.82** (n=8, max 243.52) | +46.97 |
| 120k–160k | **210.01** (n=192, max 501) | **240.99** (n=8, max 324.41) | +30.98 |
| **160k–200k (last-20%)** | **235.15** (n=170, max 501) | **268.35** (n=9, max 338.87) | **+33.20** |

**Trajectory shape comparison.**
- **dreamer-srl** — strong S-curve / climbing: 55 → 82 → 151 → 210 → 235. The Δ between consecutive windows is decelerating in the last window (only +25 from window 4 to window 5), suggesting the climb is starting to flatten but **has not plateaued**.
- **sheeprl** — also climbing but with more noise per window: 61 → 136 → 198 → 241 → 268. Within the last window, individual 5k-step samples reach 338.87 (the run-wide max). The climb is also still going up — the per-sample evidence at gs=195k shows 338.87 followed by 284.94 at gs=200k, hinting at high within-window variance rather than a fixed asymptote.

**Per-sample max comparison.** Both sides occasionally produce near-cap episodes. Dreamer-srl hits the 500-step env cap in 39 individual episodes across the last 120k env-steps (sheeprl's 5k-window averages mask whether sheeprl ever hits 500 in a single episode — the per-window means don't go above 339, so likely rarer).

### 3.2 10×10 hypervigilance side-by-side (40k-step windows)

| Window (env-steps) | dreamer-srl (`405f0555`) | sheeprl (`yt1uts22`) | Gap (sheeprl − dreamer-srl) |
|---|---|---|---|
| 0–40k | **54.44** (n=747 episodes, max 106) | **55.79** (n=7 samples, max 69.19) | +1.35 |
| 40k–80k | **67.11** (n=606, max 111) | **76.25** (n=8, max 95.69) | +9.14 |
| 80k–120k | **68.51** (n=592, max 111) | **95.01** (n=8, max 103.26) | +26.50 |
| 120k–160k | **74.58** (n=544, max 131) | **103.75** (n=8, max 118.35) | +29.17 |
| **160k–200k (last-20%)** | **69.39** (n=585, max 185) | **106.19** (n=9, max 116.46) | **+36.80** |

**Trajectory shape comparison.**
- **dreamer-srl** — early plateau followed by stagnation: 54 → 67 → 69 → 75 → 69. The agent finds a basic survival policy in the first 40k env-steps and then **stops improving** — windows 2, 3, 5 are statistically indistinguishable. The within-window max climbs from 111 (windows 2/3) to 185 (window 5), suggesting **occasional late-training escape attempts** (the agent sometimes survives much longer) but the per-episode mean does not rise. This is a "found a local-optimum survival strategy, can't break out of it" signature.
- **sheeprl** — gentle S-curve, climbing through training but flattening: 56 → 76 → 95 → 104 → 106. The 80k → 200k change is only +11 steps (105 → 116 max sample), so sheeprl is plateauing too, just at a higher ceiling.

**Note the diary entry's "plateaued at ~95 by 60k" from 2026-05-12 was actually undersold.** Sheeprl continued to climb past that partial-window observation, ending the last-20% window at 106.19. So the dreamer-srl ↔ sheeprl gap on 10×10 is wider than the diary estimate suggested (36.80, not ~25).

### 3.3 Training-throughput health (rules out one diagnostic hypothesis)

| Metric | dreamer-srl 5×5+pred | dreamer-srl 10×10 intero | sheeprl 5×5+pred | sheeprl 10×10 intero |
|---|---|---|---|---|
| Steady-state SPS (last window) | 16.3 | 16.3 | 600.2 | 579.9 |
| Wall-clock | 3.40 h | 3.40 h | 12.97 h | 12.85 h |
| Final world-model loss | 1.84 | 2.08 | 1.98 | 2.27 |

**Read.** The dreamer-srl 10×10 run sustained the **identical 16.3 SPS** as the 5×5+pred run — i.e. **no SPS regression** despite the harder env, the 27-channel observation, and the BM-toolkit per-tag logging changes that landed earlier in May. This rules out H3 ("per-step overhead from the recent metric migration is silently slowing 10×10 training"). The 10×10 sheeprl run sustained 580 SPS vs the 5×5+pred sheeprl run's 600 SPS — also flat. Throughput is not the bottleneck.

The world-model loss is slightly higher on 10×10 for both algos (more visual / multi-predator structure to reconstruct), consistent with task difficulty rather than a code-side problem.

---

## 4. Headline verdict

**Dreamer-srl v2 underperforms its sheeprl reference on both extension tasks, at fixed step budget and matched seed.**

| Task | dreamer-srl last-20% mean | sheeprl last-20% mean | Gap (steps) | Gap (% of sheeprl) |
|---|---|---|---|---|
| 5×5+predator | 235.15 | **268.35** | −33.20 | −12.4% |
| 10×10 hypervigilance | 69.39 | **106.19** | −36.80 | −34.7% |

This is the **first observed performance gap between the JAX rebuild and the PyTorch reference** since the v2-CP9 fix bundle landed. The parity gate (food-only NoPred) closed with the rebuild saturating at the env cap and matching sheeprl with zero variance. On harder tasks, the rebuild is now **clearly behind** the reference on one task (10×10, gap = 35% of sheeprl's plateau) and **noticeably behind** on the other (5×5+pred, gap = 12% of sheeprl's plateau).

**The parity claim is task-conditional.** "Dreamer-srl matches sheeprl" is true on food-only NoPred; "dreamer-srl matches sheeprl on the publication-track tasks" is **not yet true**.

**Important nuance.** Both dreamer-srl trajectories are **still climbing at end-of-training**, sharply on 5×5+pred (windows 4→5: +25 steps) and weakly on 10×10 (windows 4→5: actually −5 steps, but the per-window max climbed). Sheeprl is plateauing on both tasks. So part of the gap could close with more training steps — but at the **pre-registered 200k step budget**, dreamer-srl loses.

---

## 5. Diagnostic — three ranked hypotheses for the gap

### H1 — `num_envs` mismatch in the launch invocation (HIGH confidence)

The sheeprl baselines ran with `num_envs=4`; the dreamer-srl extension runs both ran with `--num-envs 1`. This was confirmed by reading `wandb/.../wandb-metadata.json` for each dreamer-srl run (`args: ['--env-config', ..., '--num-envs', '1', ...]`) and `wandb/.../files/config.yaml` for each sheeprl run (`num_envs: 4`).

**Why this matters mechanically (4 effects):**
1. **Effective prefill is 4× larger for sheeprl.** Both use `learning_starts=1024`, but sheeprl's prefill is `1024 × 4 = 4096` random-action transitions vs dreamer-srl's 1024. The world model has 4× more diverse early data to fit before gradient updates begin.
2. **Effective replay-buffer freshness is 4× higher for sheeprl.** At any given moment, sheeprl's buffer contains 4 envs' worth of recent transitions while dreamer-srl's contains 1. Imagined rollouts inside the actor loop are sampled from this buffer; broader within-batch state diversity helps the actor avoid local minima.
3. **Effective exploration is 4× wider for sheeprl per env-step.** Each parallel env explores a different stochastic action sequence; the world model fits over all 4 in parallel. The 10×10 task's plateau signature ("found one survival policy, can't escape") is exactly the kind of failure that a 4× wider exploration pool helps with.
4. **The food-only parity gate did not exercise this dimension.** The parity launch on food-only NoPred *also* used `--num-envs 1` (per [PARITY_LAUNCH_V2.md §2](./PARITY_LAUNCH_V2.md#2-setup)), and it saturated. So the 1-env recipe was demonstrated to work on the easiest task — but on tasks where exploration matters more (predator dynamics, 4-predator multi-tag, bushes-as-hiding-cover), the 1-env recipe may now be the binding constraint.

**Test for H1.** Re-run dreamer-srl 5×5+pred with `--num-envs 4` matched to sheeprl. If the gap closes substantially (dreamer-srl reaches ~265+ in the last-20% window), H1 is the explanation. The dreamer-srl driver supports `--num-envs N` already, so this is a config-only re-launch. A separate SPS×num_envs sweep ([SPS_NUM_ENVS_SWEEP_V2.md](./SPS_NUM_ENVS_SWEEP_V2.md)) already confirmed `--num-envs 4` is the throughput sweet spot on this implementation (3× SPS vs `--num-envs 1`), so the H1 re-run is also wall-clock-cheaper than the current 1-env recipe.

### H2 — XS capacity ceiling on the 10×10 task (MEDIUM-HIGH confidence, specific to 10×10)

The XS preset (256-wide dense layers, 256-wide recurrent state, 1 MLP layer) was chosen to match sheeprl's smallest preset for the parity gate. On the food-only NoPred 5×5 task (19-channel obs), it saturates trivially. The 5×5+pred task adds predator dynamics + bushes + olfaction over a 5-cell grid (still small). The 10×10 hypervigilance task is **qualitatively harder**:
- Grid area is 4× larger (100 cells vs 25)
- 4 hidden-predator tags, each with its own spawn area and bush coverage
- The visual sensor is enabled (it's disabled on 5×5+pred — `visual_sensor_enabled: false`), adding multi-tile observation structure
- Interoceptive nociception is enabled with a 12-step alpha-kernel convolution, adding temporal delay structure that the world model must capture

Sheeprl plateaus at survival 106 on this task at XS — **the task itself is hard at XS capacity**. The diary entry from 2026-05-14 noted the user has already been considering bigger sizes (the `gwp_5x5_pred_M_s42_n113` run from 2026-05-14 confirms M-size is being tested elsewhere). H2's prediction is that the 10×10 task **fundamentally requires more capacity** and neither algorithm reaches the true ceiling at XS.

The dreamer-srl-specific lower number (69 vs 106) within this regime is then **H1's signature on top of H2's capacity ceiling**: sheeprl's 4-env exploration finds a slightly higher local optimum than dreamer-srl's 1-env exploration, but neither is escaping the capacity ceiling.

**Test for H2.** Run both dreamer-srl and sheeprl at M-size (or L) on the 10×10 intero task. If both jump from ~100 to a substantially higher number (say 180+), capacity ceiling at XS was the binding constraint. If only sheeprl jumps and dreamer-srl stays low, H1 is binding on dreamer-srl independently of H2.

### H3 — A code-side residual bug surfaces on harder tasks (LOW-MEDIUM confidence)

The v2-CP9 fix bundle resolved the v1 parity-launch failure (Fix 1–5; see [PARITY_LAUNCH_V2.md §1](./PARITY_LAUNCH_V2.md#1-context-plain-language-280-words)). All five fixes were verified by 11 gradient-bit-identity tests against the sheeprl reference. But:
- The tests are at module / function granularity, not at full-training trajectory granularity
- A residual bug that's a no-op on the easy task and a big problem on the hard task is a known anti-pattern in RL rebuilds (the v1 verdict carries this exact failure mode pre-Fix-3)
- One candidate: the **continuation prediction** (whether the agent dies on this step). On food-only NoPred, deaths are starvation deaths spaced 100+ steps apart. On 5×5+pred, deaths are predator-attack deaths that can occur on any step. On 10×10 hypervigilance, deaths are mostly predator-attack with rare starvation. If the JAX driver's `terminated` vs `truncated` split (Fix 2) interacts with the `continue` prediction differently when deaths are dense, the gradient signal could be biased on these tasks.

**Test for H3.** Compare the **continue-loss curves** between dreamer-srl 5×5+pred and sheeprl 5×5+pred. If dreamer-srl's continue-loss does not converge to the same final value as sheeprl's (within ~10%), H3 is implicated. Also worth checking: the **moments_invscale floor-fraction** signature from the v1 parity failure (the canonical "flat returns" indicator); dreamer-srl 5×5+pred steady-state shows `moments_invscale ≈ 74` at iter 200k — well above the 1.0 safe floor, so this v1-style signature is not present, but a similar-flavored failure could exist on the policy side.

**Why H3 is lower-confidence than H1.** H1 has a **direct, falsifiable mechanism**: `num_envs=1 vs 4`. H3 is an unspecified residual bug; the parity launch was an exhaustive forward+gradient parity test, and all 5 known bugs were fixed. The base rate of "everything passed parity tests but a residual bug still bites you on the next task" is non-zero but lower than "an obvious launch-time hyperparameter mismatch is the explanation".

### Hypothesis ranking summary

| Hypothesis | Confidence | Test cost | Test priority |
|---|---|---|---|
| **H1 — `num_envs=1` vs sheeprl's `=4`** | HIGH | 1 dreamer-srl re-run at `--num-envs 4`, 5×5+pred | **First** |
| **H2 — XS capacity ceiling on 10×10** | MEDIUM-HIGH (10×10 only) | 2 runs at M-size (dreamer-srl + sheeprl), 10×10 | **Second** |
| **H3 — residual algo bug surfacing on hard tasks** | LOW-MEDIUM | continue-loss + moments_invscale curve comparison | Third (cheap to look at first) |

---

## 6. Recommendations

The three actionable next moves, ordered by leverage:

### 6.1 Immediate, cheap: re-run dreamer-srl on 5×5+pred at `--num-envs 4` (H1 test)

Re-launch a single dreamer-srl seed with the **only change being `--num-envs 4`** to match sheeprl exactly. Same env config, same agent config, same seed (42), same total steps (200,000). This is the **single highest-leverage experiment available** because:
- It tests the most likely explanation
- It is config-only — no code changes
- It costs ~3.4 h wall-clock (one node, one GPU)
- If the gap closes (dreamer-srl reaches ~265+ in the last-20% window), H1 is confirmed and the recommendation propagates to all future dreamer-srl launches: **default to `--num-envs 4` on harder tasks unless there's a specific reason not to**
- If the gap does not close, H1 is falsified and attention moves to H2 / H3

**Pre-registered pass criterion**: dreamer-srl last-20% mean ≥ 245 (i.e. within 90% of sheeprl's 268.35). Hand-off to `experiment-designer` to formalize the comparison.

### 6.2 Medium-term: M-size sweep on 10×10 hypervigilance (H2 test)

The user has been considering this (the `gwp_5x5_pred_M_s42_n113` run from 2026-05-14 indicates an M-size 5×5+pred sheeprl run already exists). Extend this idea by running **both** dreamer-srl and sheeprl at M-size on the 10×10 intero task, same 200k step budget. This directly tests whether XS is the binding constraint on the 10×10 plateau. If yes, the publication track moves to M-size for hypervigilance (and dreamer-srl's M-size implementation gets stress-tested in the same comparison). Hand-off to `experiment-designer`.

### 6.3 Optional diagnostic: continue-loss + moments_invscale curve comparison (H3 test)

Cheap — no new training. Extract `Loss/continue_loss` and `Diagnostic/moments_invscale` time-series from `xvrtjlat` and `dbh64fcy` and overlay. If dreamer-srl's curves are within 10% of sheeprl's on the same task at the same step, H3 is downweighted. If divergent, surface to `senior-developer` for a targeted code audit. This is a `experiment-analyzer` follow-up, no new wall-clock budget needed.

### 6.4 Do not block on this for downstream work

The publication track is on the hypervigilance task, which is **already known to be capacity-limited** at XS for sheeprl too. The 10×10 gap should not block downstream neuromodulation-integration work that needs a backbone — but `senior-developer` should be aware that **the backbone's claimed parity is task-conditional** and that downstream comparisons need to be done at the size at which the backbone matches sheeprl on the task in question.

---

## 7. Implications — does this block the publication track?

**Short answer: no, but it changes the bookkeeping.** The parity verdict ([PARITY_LAUNCH_V2.md §7](./PARITY_LAUNCH_V2.md#7-verdict)) closed at PASS-OUTPERFORM on food-only NoPred. That verdict stands — the rebuild really does match sheeprl on that task. What this comparison adds is a **second-tier qualifier**: "matches sheeprl on the easiest task, underperforms sheeprl on the publication-track tasks at the current launch recipe".

The publication-track work (perceptual noise + neuromodulation on the hypervigilance task) was always going to use **sheeprl** as the primary algorithm path, not dreamer-srl, because sheeprl already had a working multi-month track record on this task. The dreamer-srl rebuild was being built as a JAX-side parity asset that could (a) run faster on JAX hardware and (b) be the platform for JAX-side custom modifications (FiLM gating, NMN integration, etc.).

**Three concrete adjustments to the project's next moves**:

1. **The "dreamer-srl is shippable" claim from PARITY_LAUNCH_V2.md §10.1 needs the task-conditional qualifier added.** Future references to the parity-pass should read "parity established on food-only NoPred; gaps observed on harder tasks pending H1 / H2 follow-up". `senior-developer` to update [IMPLEMENTATION_PLAN.md](../../../develop/active/dreamer_srl_v2/IMPLEMENTATION_PLAN.md) accordingly when closing v2-CP12.

2. **The natural next experiment is H1's test** (§6.1). If the H1 re-run at `--num-envs 4` closes the 5×5+pred gap to within 10%, the dreamer-srl-track JAX rebuild is fully shippable for that task and the publication track can use it. If it does not, the dreamer-srl track is **demoted to a methodological asset** (gradient-tested JAX implementation, useful for FiLM / NMN integration experiments) and the publication backbone stays on sheeprl.

3. **The M-size sweep on hypervigilance is the more publication-relevant experiment** of the two recommendations. The 10×10 gap is the bigger gap (35% vs 12%) and the hypervigilance task is the publication target. If M-size closes the gap for both algorithms simultaneously, that's a useful capacity finding for the publication itself, not just a diagnostic for the rebuild.

**What the project does NOT need to do**: revisit the food-only parity verdict, re-audit the v2-CP9 fix bundle, or block any downstream work on these results. The parity claim was honest for the task it tested. The next questions are recipe-level (num_envs, size) before they are code-level.

---

## 8. Metrics Requested

No new logging needed for the H1 / H2 tests. The current logging surface (`Game/ep_len_avg`, `Time/sps_env`, `Loss/continue_loss`, `Diagnostic/moments_invscale`, `Loss/world_model_loss`, `Loss/policy_loss`) is sufficient.

**Optional convenience** — would be useful but not blocking: a single config-loader print at training-start summarizing the **effective hyperparameter overlap with the sheeprl reference** (`num_envs`, `learning_starts`, `replay_ratio`, `horizon`, `gamma`, `unimix` on the same line). The `--num-envs 1` vs `=4` mismatch was sitting in the launch invocation and the wandb-metadata for two days before anyone noticed it; a startup print would have caught it earlier. If accepted, `senior-developer` plans → `developer` implements (one-line print, ~10 lines of code).

---

## 9. Related Issues

- **H1 follow-up (highest priority)**: `experiment-designer` formalizes a pre-registered 1-vs-4 num_envs comparison on 5×5+pred. The 5×5+pred re-run with `--num-envs 4` is the test.
- **H2 follow-up**: `experiment-designer` formalizes an M-size comparison (dreamer-srl + sheeprl) on 10×10 intero. This dovetails with the user's existing M-size sweep direction (the `gwp_5x5_pred_M_s42_n113` 2026-05-14 run).
- **H3 follow-up (cheap)**: `experiment-analyzer` runs a continue-loss + moments_invscale curve overlay on `xvrtjlat` vs `dbh64fcy` — no new training, just extraction. Surface to `senior-developer` only if curves diverge.
- **v2-CP12 verdict closure update**: `senior-developer` to add the task-conditional qualifier to the parity claim when closing v2-CP12.
- **Launch-time hyperparam print** (optional logging convenience): `senior-developer` plans → `developer` implements.

---

## 10. Cross-references

- **PARITY_LAUNCH_V2** (the food-only NoPred parity verdict — the headline "matches sheeprl" claim): [PARITY_LAUNCH_V2.md](./PARITY_LAUNCH_V2.md)
- **v2 IMPLEMENTATION_PLAN**: [docs/develop/active/dreamer_srl_v2/IMPLEMENTATION_PLAN.md](../../../develop/active/dreamer_srl_v2/IMPLEMENTATION_PLAN.md)
- **Diary 2026-05-12** (sheeprl baseline diary entries with partial-window observations that this doc supersedes with full-window data): [docs/diary/2026-05-12.md](../../../diary/2026-05-12.md)
- **Diary 2026-05-14** (the M-size 5×5+pred sheeprl run that motivates the H2 follow-up): [docs/diary/2026-05-14.md](../../../diary/2026-05-14.md)
- **Sheeprl 5×5+pred baseline**: WandB [`dbh64fcy`](https://wandb.ai/sungwoolee/grid_world_pain_sheeprl_test/runs/dbh64fcy)
- **Sheeprl 10×10 intero baseline**: WandB [`yt1uts22`](https://wandb.ai/sungwoolee/grid_world_pain_sheeprl_test/runs/yt1uts22)
- **Dreamer-srl 5×5+pred extension**: WandB [`xvrtjlat`](https://wandb.ai/sungwoolee/grid_world_pain/runs/xvrtjlat)
- **Dreamer-srl 10×10 intero extension**: WandB [`405f0555`](https://wandb.ai/sungwoolee/grid_world_pain/runs/405f0555)
- **Extraction artifacts**: [`tmp/20260515_dreamer_srl_extension_baselines.json`](../../../../tmp/20260515_dreamer_srl_extension_baselines.json), [`tmp/20260515_dreamer_srl_extension_dreamersrl_traj.json`](../../../../tmp/20260515_dreamer_srl_extension_dreamersrl_traj.json)
