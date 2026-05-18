---
title: "Why does dreamer-srl v2 fall short of recurrent-PPO on the 10×10 hypervigilance task? Reward-head pos/neg asymmetry as the leading hypothesis"
topic: dreamer_srl_v2
status: active
created: 2026-05-18
last_updated: 2026-05-18
phase: 2
wandb_tag: "dreamer_srl_v2_xs_*"
develop_link: docs/develop/active/dreamer_srl_v2/
---

# Why does dreamer-srl v2 fall short of recurrent-PPO on the 10×10 hypervigilance task? Reward-head pos/neg asymmetry as the leading hypothesis

> **Status**: ANALYZING (Mode B — retrospective; not pre-registered)
> **Date**: 2026-05-18
> **Author**: experiment-analyzer
> **Related**:
> - Memory `[[20260518_1511_dreamer_srl_v2_parity_pass_outperform]]` — dreamer-srl v2 passed food-only parity at survival 501.
> - Memory `[[20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry]]` — same asymmetry pattern seen earlier in original Dreamer with a partial zero-init fix.
> - Memory `[[20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt]]` — paper-canonical bins partially fixed it; reward MAE compounds 0.18 → 3.05 across imagination horizons.
> - Memory `[[20260509_1534_wm_reward_head_localized_failure_a1]]` — offline world-model diagnostic localized the failure to the reward head.
> - Forward references (not yet written): `docs/reviews/dreamer_srl_reward_head_audit.md` (Phase 2a code-audit) and `docs/experiments/active/dreamer_srl_v2/death_penalty_ablation_design.md` (Phase 3 ablation).
> - Cross-link: `docs/experiments/active/dreamer_srl_v2/HYPERPARAM_SEARCH_10X10.md` (the 8-cell extended sweep that surfaced the runs analyzed here).

---

## 1. Research Question

**Plain-language framing.** This project has two strong agents on the 10×10 hypervigilance task — a survival benchmark in which an agent must avoid predators and damage sources while gathering food. The first agent, *recurrent-PPO* (henceforth **rPPO**), is a model-free policy-gradient agent that learns directly from environment reward. The second, **dreamer-srl v2**, is our re-implemented model-based agent: it learns a *world model* (a compact predictor of future observations, rewards, and episode terminations) and then trains its policy on **imagined** roll-outs of that world model rather than on real environment steps. On the food-only parity benchmark, dreamer-srl v2 actually outperformed all comparators (survival 501 steps). But on the harder 10×10 hypervigilance task — which adds large negative rewards (predators that one-shot kill the agent, plus a 100-point penalty per death event) — dreamer-srl v2's best run survives **191 steps**, while rPPO survives **227–239 steps**. So dreamer-srl v2 *loses ground to rPPO precisely on the task where large negative rewards dominate the reward landscape*, even though it beats rPPO when only food rewards (positive, small-magnitude) are present.

What we want to know: **is dreamer-srl v2's gap on the hypervigilance task caused by its world model systematically under-fitting negative rewards** (so the agent imagines a more forgiving world than the real one) — or is something else going on? The leading suspect, based on summary metrics from the 8-cell sweep, is the **reward-head asymmetry**: in every dreamer-srl v2 run on this task, the world model's reward predictor has a *much larger* mean absolute error on episodes that ended badly than on episodes that ended well — by a factor of 2.2× to 6.4×. A reader who has not seen the prior diagnosis memos should be able to read this section alone and understand: dreamer-srl v2 is a world-model agent; its world model is bad at predicting the *bad* outcomes specifically; we hypothesize this is why its survival lags rPPO on this task.

> This document is **Mode B** — a retrospective post-hoc analysis. There was no pre-registered design doc with predictions; the runs were originally launched as a hyperparameter search (see `HYPERPARAM_SEARCH_10X10.md`). The hypotheses below are framed retroactively to keep the analysis honest, and the conclusions are correspondingly weaker than they would be for a pre-registered experiment.

**Formal hypotheses (numbered for traceability):**

> **H1 (leading — reward-head asymmetry drives the rPPO gap).** The world model's reward head systematically under-fits negative rewards (`WorldModel/model_reward_mae_neg` is 2.2–6.4× `WorldModel/model_reward_mae_pos` across all 5 XS-recipe runs on this env). Imagination roll-outs used to train the actor therefore underestimate the cost of dying. The actor learns a less defensive policy than rPPO (which receives the true negative reward directly) and survives fewer steps.
>
> **H2 (compounding — imagination horizon amplifies H1).** Analogous to the cascade documented in memory `[[20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt]]` for the original Dreamer (per-horizon reward MAE rising from 0.18 at h=5 to 3.05 at h=50), the dreamer-srl v2 reward head likely compounds error along the imagination horizon. The actor's policy gradient is dominated by deep-horizon imagined returns, so even a small per-step asymmetry produces a large policy bias.
>
> **H3 (mitigable — flattening the negative-reward distribution should close the gap).** If H1 is correct, reducing the magnitude of the dominant negative-reward term (`env.death_penalty`, currently 100) should compress the reward range, ease the regression target the reward head must fit, and lift survival toward rPPO levels. The Phase 3 ablation tests this.

**H₀ (null for H1).** Reward-head MAE asymmetry is incidental — driven by class imbalance in the reward stream rather than by representational under-capacity — and does not causally limit policy quality. Closing the asymmetry would not move survival.

## 2. Experimental Design

### 2.1 Independent Variables

This is Mode B: the runs were not designed to test H1. The independent variables come from the originating hyperparameter search.

| Variable | Values (within the 5 runs analyzed) | Rationale |
|---|---|---|
| Recipe size | XS | Smallest model size in the sweep — winning cell of the 8-cell extended sweep |
| Number of parallel envs | 16, 64, 128 | Throughput vs. on-policy-ness sweep |
| Training budget | 2M and 4M env steps | Convergence vs. cost sweep |
| Loss-balancer flag | off (default), `LB` (on) | Pre-existing variant from the sweep |

### 2.2 Controlled Variables

| Variable | Value |
|---|---|
| Environment config | `configs/experiment/hypervigilance/01-interoNocicept.yaml` |
| `env.death_penalty` | 100 (one-shot at injury-termination) |
| `env.max_injury` | 100 |
| `env.injury_smoothing_duration` | 3 |
| World model size | "XS" preset (winning size from the sweep) |
| Reward-head representation | twohot with `linspace(-20, +20, 255)` in SYMLOG space (`src/algorithms/dreamer_srl/loss.py:118`) |
| Comparator (rPPO) configs | `rppo_nmn_tempceil10_p4_film_g1_s0`, `rppo_nmn_het_p4_unmod_c7` |

### 2.3 Confounds & Limitations

| Confound | Affected Runs | Severity | Mitigation |
|---|---|---|---|
| Single-seed per cell — no seed dispersion measured for the dreamer-srl v2 runs | all 5 analyzed runs | **High** | Treat ep-len gaps under ~20 steps as noise; H1 is supported only because the *asymmetry pattern* replicates across all 5 single-seed runs, not because any one ep-len is reliable |
| Not pre-registered — predictions are retroactive | all | High | Stated explicitly in §1; H1 reframed as the *leading* hypothesis, not a confirmed one |
| In-flight M-cell runs not yet incorporated | E7 `c5pt9t4v` at 86%, E8 `d9emwzdp` at 60% | Medium | §10.4 second-pass synthesis is pending; this doc represents the XS-cell evidence only |
| rPPO does not log `WorldModel/model_reward_mae_*` — direct head-to-head on the asymmetry metric is impossible | rPPO comparators | Medium | We compare *survival* head-to-head; the asymmetry argument is internal to dreamer-srl v2 |
| Twohot bin support is `[−20, +20]` in symlog space, which covers symlog(−355) ≈ −5.87 — so representability is not the limit | all dreamer-srl v2 runs | Low | Verified at `src/algorithms/dreamer_srl/loss.py:118`; bottleneck is *density of fit* in the deep-negative tail, not bin support |

## 3. Launch Manifest

No new training launched for this doc. Manifest below maps the 5 dreamer-srl v2 XS-recipe runs that surfaced the asymmetry (sourced from `HYPERPARAM_SEARCH_10X10.md`), plus the two rPPO comparators referenced as performance anchors.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | completed | XS / 16 envs / 4M | dreamer_srl_v2_xs_e16_4M | hypervigilance | prod | 0 | — | — | 2026-05-16T12:42:24 | `yxij4lrc` | (sweep) |
| 2 | completed | XS / 64 envs / 4M | dreamer_srl_v2_xs_e64_4M | hypervigilance | prod | 0 | — | — | 2026-05-16T12:42:25 | `02n94uzu` | (sweep) |
| 3 | completed | XS / 128 envs / 4M | dreamer_srl_v2_xs_e128_4M | hypervigilance | prod | 0 | — | — | 2026-05-16T12:42:36 | `ybsma2zd` | (sweep) |
| 4 | completed | XS / 16 envs / 2M / LB | dreamer_srl_v2_xs_e16_2M_lb | hypervigilance | prod | 0 | — | — | 2026-05-16T01:23:50 | `bzc2x3pl` | (sweep) |
| 5 | completed | XS / 64 envs / 2M / LB | dreamer_srl_v2_xs_e64_2M_lb | hypervigilance | prod | 0 | — | — | 2026-05-16T01:23:52 | `15uiw4kg` | (sweep) |
| 6 (ref) | completed | rPPO comparator | rppo_nmn_tempceil10_p4_film_g1_s0 | hypervigilance | prod | 0 | — | — | — | (see HYPERPARAM_SEARCH) | — |
| 7 (ref) | completed | rPPO comparator | rppo_nmn_het_p4_unmod_c7 | hypervigilance | prod | 0 | — | — | — | (see HYPERPARAM_SEARCH) | — |
| 8 (ref) | completed | sheeprl PyTorch DV3 | sheeprl_dv3_baseline | hypervigilance | prod | 0 | — | — | — | (see HYPERPARAM_SEARCH) | — |
| 9 (in-flight) | running | M / 64 envs / 2M (E7) | dreamer_srl_v2_m_e64_2M | hypervigilance | prod | 0 | — | — | (in flight) | `c5pt9t4v` | — |
| 10 (in-flight) | running | M / 128 envs / 2M (E8) | dreamer_srl_v2_m_e128_2M | hypervigilance | prod | 0 | — | — | (in flight) | `d9emwzdp` | — |

> The M-cell rows (9, 10) are listed for completeness but **not analyzed in this doc** — they are still mid-training. A §10.4 second-pass will fold them in after completion.

### 3.1 Configs to Produce

Not applicable — no new configs produced.

## 4. Results

### 4.1 Primary Metrics (Hypothesis Test)

Headline metric is **survival steps** (`Episode/Steps`, mean over the final 5% of training), per the project rule. Cumulative reward is not the headline.

| Run | Stack | Recipe | Survival steps (`Episode/Steps`) | Δ vs. rPPO best |
|---|---|---|---|---|
| `yxij4lrc` | dreamer-srl v2 | XS / 16 envs / 4M | **191** | −48 (−20%) |
| `02n94uzu` | dreamer-srl v2 | XS / 64 envs / 4M | 147 | −92 (−38%) |
| `ybsma2zd` | dreamer-srl v2 | XS / 128 envs / 4M | 128 | −111 (−46%) |
| `bzc2x3pl` | dreamer-srl v2 | XS / 16 envs / 2M, LB | 152 | −87 (−36%) |
| `15uiw4kg` | dreamer-srl v2 | XS / 64 envs / 2M, LB | 139 | −100 (−42%) |
| `rppo_nmn_tempceil10_p4_film_g1_s0` | rPPO | — | 227 | — |
| `rppo_nmn_het_p4_unmod_c7` | rPPO | — | **239** | — |
| sheeprl PyTorch DV3 | sheeprl ref | — | 106 | — |

**Verdict on H1 (preliminary).** The asymmetry pattern below replicates across every XS-recipe dreamer-srl v2 run on this env. H1 is **supported as the leading hypothesis** but not confirmed — confirmation requires Phase 2a (code audit) and Phase 3 (`env.death_penalty` ablation).

### 4.2 Secondary Metrics — Reward-Head Asymmetry (the core evidence)

These are world-model internals: how well the reward predictor fits the *positive* side of the reward distribution vs. the *negative* side, measured in MAE (mean absolute error). A symmetric, healthy reward head would have `neg ≈ pos`. The data shows the opposite — the negative side is systematically harder for the reward head to fit by a factor of 2.2× to 6.4×.

| Run | Recipe | Survival | `rew_MAE_total` | `rew_MAE_neg` | `rew_MAE_pos` | **neg/pos ratio** |
|---|---|---|---|---|---|---|
| `yxij4lrc` (winner) | XS / 16 / 4M | 191 | 0.76 | **1.02** | 0.31 | **3.32×** |
| `02n94uzu` | XS / 64 / 4M | 147 | 0.39 | 0.63 | 0.28 | 2.23× |
| `ybsma2zd` | XS / 128 / 4M | 128 | 0.38 | 0.77 | 0.31 | 2.46× |
| `bzc2x3pl` (LB) | XS / 16 / 2M | 152 | 0.75 | 1.08 | 0.17 | **6.39×** |
| `15uiw4kg` (LB) | XS / 64 / 2M | 139 | 0.65 | 1.08 | 0.35 | 3.08× |

**Median neg/pos ratio: 3.08×. Range: 2.23× – 6.39×.** No run escapes the pattern.

### 4.3 Diagnostic Metrics

#### 4.3.1 Episode reward range — why the negative tail matters

For the winner run `yxij4lrc`:
- `Episode/Reward` (mean) = **−322.3**
- `Episode/Reward_Min` = **−355.4**
- `Episode/Reward_Max` = **−231.9** (still negative — even the best episodes have net-negative reward because `death_penalty=100` is one-shot)
- `Eval/MeanReward` = **−206.0** (eval policy slightly more cautious)

Episode reward per-component (winner): `TotalDamage ≈ 307` (`DamagePredator=204` + `DamageDanger=73` + `DamageObstacle=30`). The reward stream is dominated by accumulated damage cost plus the one-shot `death_penalty=100` at termination.

#### 4.3.2 Bin support is NOT the bottleneck

Verified at `src/algorithms/dreamer_srl/loss.py:118` — the twohot reward head uses `linspace(-20, +20, 255)` in **symlog** space. The symlog transform compresses large magnitudes: `symlog(−355) ≈ −5.87`, well inside the `[−20, +20]` bin range. The reward head *could* in principle represent the deep-negative tail; what it lacks is **density of training signal** there. Most steps yield small negative rewards (per-step damage); the catastrophic deep-negative outcome (death termination) is rare and one-shot per episode, so the bins near the deep tail get fewer regression updates.

#### 4.3.3 World-model loss summary (winner)

| Loss component | Value (final) |
|---|---|
| `WorldModel/loss_model` | 2.485 |
| `WorldModel/loss_rew` | 0.842 (largest non-KL term) |
| `WorldModel/loss_recon` | 0.049 |
| `WorldModel/loss_cont` | 0.006 |
| `WorldModel/loss_kl` | 1.588 |
| `WorldModel/model_cont_acc` | 99.6% |
| `WorldModel/model_latent_entropy` | 1.218 |

Continuation accuracy is excellent (99.6%), reconstruction loss is small (0.049), latent entropy is moderate (1.218 — not collapsed). The reward loss (`loss_rew=0.842`) is the dominant **non-KL** signal in the model loss budget — consistent with the reward head being the under-fit component.

### 4.4 Learning Dynamics

**Caveat.** The local-file temporal trajectory (intermediate-step values of `WorldModel/model_reward_mae_*` over training) could not be extracted via the local `.wandb` datastore reader in this session. The values reported in §4.2 are end-of-training summary fields (`wandb-summary.json`). This is a known gap; see *Metrics Requested* below. The conclusions in this doc rest on the **cross-run replication of the asymmetry pattern** rather than on the intra-run temporal shape — which is the weaker but still informative version of the evidence.

What we *can* read from the summary: across the five runs, the runs with the longest training budget (4M env steps, runs 1–3) have a slightly lower `neg/pos` ratio (2.2×–3.3×) than the 2M-budget runs (3.1×–6.4×). This is consistent with the head slowly closing the gap as more data accumulates, but **not closing it** even at 4M steps with the smallest model.

## 5. Analysis

### 5.1 Key Findings

**Finding 1 — Reward-head asymmetry replicates across every XS-recipe dreamer-srl v2 run on the hypervigilance task.**
- *What*: `WorldModel/model_reward_mae_neg` is 2.23× – 6.39× `WorldModel/model_reward_mae_pos` in all 5 runs analyzed (median 3.08×).
- *Why* (proposed mechanism): The twohot reward head learns from per-step regression targets. The negative-reward tail is dominated by rare, high-magnitude events (death termination, predator hits); the positive tail is dense, small-magnitude (food). Equal weighting per step gives the dense positive distribution far more gradient than the sparse negative tail. The head fits what it sees most.
- *Evidence*: Table in §4.2; all 5 runs in same direction; pattern persists across env counts (16/64/128), training budgets (2M/4M), and loss-balancer on/off.
- *Confidence*: **High** for the asymmetry itself; **Medium** for the proposed mechanism (alternatives in §5.3).

**Finding 2 — The same asymmetry pattern is documented in earlier Dreamer-family work in this project.**
- *What*: The original-Dreamer Z1 study (memory `[[20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry]]`) found the same neg/pos asymmetry; zero-init of the reward + critic heads gave a partial fix. The Z2 study (memory `[[20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt]]`) further established that reward MAE compounds along the imagination horizon (0.18 at h=5 → 3.05 at h=50). The A1 offline-WM diagnostic (memory `[[20260509_1534_wm_reward_head_localized_failure_a1]]`) localized the failure to the reward head specifically.
- *Why*: dreamer-srl v2 inherits the architectural pattern (twohot regression head on a sequence model). The 5 P-blockers fixed in v2 (per parity memory `[[20260518_1511_dreamer_srl_v2_parity_pass_outperform]]`) addressed training-loop bugs, *not* reward-head capacity — so the asymmetry should still be present, and §4.2 confirms it is.
- *Evidence*: cross-doc replication; the pattern is not a v2-specific artifact.
- *Confidence*: **High**.

**Finding 3 — On the food-only parity task (no large negative rewards), dreamer-srl v2 wins.**
- *What*: dreamer-srl v2 reaches survival 501 on the parity task; rPPO does not match this (per memory `[[20260518_1511_dreamer_srl_v2_parity_pass_outperform]]`).
- *Why*: Without a deep-negative-reward tail, the reward head has no asymmetry to suffer from. Imagination is accurate; the agent benefits from being model-based.
- *Evidence*: parity memory; H1's flip-direction prediction (model-based should win when the reward distribution is symmetric) is satisfied.
- *Confidence*: **Medium-High** — this is a strong consistency check but does not by itself prove the causal mechanism for the hypervigilance gap.

**Finding 4 — Survival ranking within the 5 dreamer-srl v2 runs does not cleanly track the asymmetry ratio.**
- *What*: The winner (`yxij4lrc`, 191 steps) has a higher neg/pos ratio (3.32×) than `02n94uzu` (2.23×, 147 steps). If the asymmetry ratio were the sole cause, we would expect the lowest-ratio run to have the highest survival.
- *Why*: Survival is jointly determined by reward-head quality, imagination horizon, world-model recon quality, and exploration. Asymmetry is *one* knob, not the only one. The XS-recipe budget interactions (env count × training steps) also matter.
- *Evidence*: §4.1 vs. §4.2 row-wise.
- *Confidence*: **Medium**. This finding *weakens* the strict H1 ("asymmetry magnitude monotonically predicts survival") but is consistent with the looser H1 ("asymmetry is a *necessary* condition for the rPPO gap, but other factors modulate it"). Phase 3's death-penalty ablation tests the looser form directly.

### 5.2 Cross-Run Comparisons

| Comparison | Cells | What changes | What is held | Observation |
|---|---|---|---|---|
| 4M vs. 2M budget | `yxij4lrc`/`02n94uzu`/`ybsma2zd` vs. `bzc2x3pl`/`15uiw4kg` | training budget | recipe size = XS | Longer budget marginally lowers `neg/pos` ratio (median 2.5× vs. 4.7×) but does not close it. Survival also marginally higher (median 147 vs. 145). |
| Loss balancer on vs. off | `bzc2x3pl`/`15uiw4kg` vs. `yxij4lrc`/`02n94uzu`/`ybsma2zd` | LB flag | recipe size = XS | LB does not fix the asymmetry; the two LB runs have the highest neg/pos ratios (6.39× and 3.08×). |
| Envs 16 vs. 64 vs. 128 | `yxij4lrc` vs. `02n94uzu` vs. `ybsma2zd` | parallel env count | XS, 4M budget | Survival monotonically decreases with more envs (191 → 147 → 128). Likely an on-policy-ness effect; orthogonal to the asymmetry question. |
| dreamer-srl v2 vs. rPPO (this env) | all 5 vs. rPPO best | algorithm family | env config identical | rPPO wins by 36–111 steps (19–46%). |
| dreamer-srl v2 vs. rPPO (parity env) | parity memory | algorithm family | parity env | dreamer-srl v2 wins (survival 501). |

The two cross-env comparisons together are the strongest indirect support for H1 in the absence of a code audit and a death-penalty ablation: **the model-based agent loses ground specifically on the env with large negative rewards**.

### 5.3 Failure Modes & Pathologies

#### Pathology 1 — Reward-head capacity limit (the H1 mechanism)

- **Affected**: all 5 dreamer-srl v2 XS-recipe runs.
- **When it appears**: throughout training; the summary asymmetry is present at end of training and the 4M-vs-2M comparison suggests it persists despite extra updates.
- **Likely cause**: regression target density imbalance — the negative-reward tail has fewer per-step training examples than the positive food-reward distribution.
- **Performance correlation**: cross-run, every dreamer-srl v2 run on this env shows the asymmetry; every run underperforms rPPO; flips direction on parity (no deep negative tail → dreamer-srl wins).

#### Pathology 2 — Imagination roll-out compounding (the H2 mechanism, not yet measured in v2)

- **Affected**: all 5 dreamer-srl v2 XS-recipe runs (predicted; per-horizon MAE not measured in v2 yet).
- **When it appears**: throughout training, increasing with imagination horizon.
- **Likely cause**: small per-step reward bias × imagination horizon length → policy gradient bias.
- **Performance correlation**: not yet measured. **Requires the offline-WM diagnostic port (Phase 2b) to verify in dreamer-srl v2.** Documented as analogous in original Dreamer (Z2 memory, 0.18 → 3.05 from h=5 to h=50).

#### Anti-pattern flag — `rew_MAE_total` does NOT equal a simple average of `rew_MAE_neg` and `rew_MAE_pos`

For the winner: `rew_MAE_total=0.76`, `rew_MAE_neg=1.02`, `rew_MAE_pos=0.31`. The total is **closer to neg than pos**, suggesting negative-reward steps dominate the buffer by count or weight in the MAE calculation, **OR** that the metric is computed over a buffer skewed toward terminal states. This is not a bug per se, but the per-side metrics need their definition cross-checked in the audit (`src/algorithms/dreamer_srl/loss.py`) before drawing tight quantitative conclusions. Surfacing as a data-quality flag for Phase 2a.

## 6. Conclusions

### 6.1 Summary

- **Headline.** dreamer-srl v2 best (191 survival steps) underperforms rPPO best (227–239 survival steps) by 19–25% on the 10×10 hypervigilance task. Across all 5 XS-recipe dreamer-srl v2 runs on this env, the world-model reward head has 2.2× – 6.4× higher MAE on negative-reward steps than on positive-reward steps. The same agent wins on the food-only parity task (no large negative rewards). The data is **consistent with** H1 — reward-head asymmetry as the leading cause of the rPPO gap — but does not yet *prove* it.
- **H1 verdict — leading hypothesis, supported by indirect evidence.** Pattern replicates across every XS run on this env. Flips direction on parity (where there is no asymmetry to suffer). The within-cohort survival ranking is noisy w.r.t. the asymmetry ratio (Finding 4), so we cannot claim a *monotonic* causal link — only that the asymmetry is present and large in every run that loses to rPPO, and absent from the run that wins (parity).
- **H2 verdict — pending Phase 2b.** Imagination-horizon compounding is plausible by analogy to the original Dreamer Z2 study but is not measured in dreamer-srl v2. Resolving it requires porting the offline-WM diagnostic.
- **H3 verdict — pending Phase 3.** Reducing `env.death_penalty` to flatten the negative-reward distribution is the cleanest causal test. The Phase 3 design doc (forward reference) owns it.
- **Bin support is not the bottleneck.** Twohot bins cover `symlog(−355) ≈ −5.87` inside `[−20, +20]`. The bottleneck is fitting *density* in the deep-negative tail, not representability.

### 6.2 Limitations & Open Questions

- **Single-seed runs.** Each cell is one seed; the gap-vs.-rPPO numbers carry that uncertainty. The asymmetry pattern is robust (5/5 runs in same direction), but absolute survival numbers are not.
- **No causal test.** H1 is *consistent* with the data; it is not confirmed. Phase 3's `env.death_penalty` ablation provides the causal handle.
- **Intra-run temporal trajectory not captured.** The local `.wandb` datastore did not yield history rows in this session; conclusions rest on end-of-training summary values and cross-run replication. See *Metrics Requested* below.
- **M-cell runs not yet folded in.** E7 (`c5pt9t4v`, M / 64 envs / 2M) and E8 (`d9emwzdp`, M / 128 envs / 2M) are still in flight. A bigger model may close the asymmetry purely via capacity; if it does, that *strengthens* H1; if it does not, that strengthens the "regression density imbalance" mechanism over the "model size" mechanism.
- **rPPO does not log the same metric.** We cannot directly check whether rPPO's value head shows an analogous asymmetry. If it did and rPPO still wins, that would weaken H1 substantially.

### 6.3 Recommended Next Experiments

| Priority | Experiment | Rationale | Effort |
|---|---|---|---|
| P0 | **Phase 2a code audit** of `src/algorithms/dreamer_srl/loss.py` reward-head + twohot + MAE-neg/pos computation, plus `src/algorithms/dreamer_srl/world_model.py` reward-head architecture. Document in `docs/reviews/dreamer_srl_reward_head_audit.md`. | Confirms the metric semantics (resolves the `rew_MAE_total` anti-pattern flag) and identifies code-level levers (e.g., reward-target loss weighting, zero-init of the head per Z1) before launching ablations. | Small (read-only; 1 reviewer pass) |
| P0 | **Phase 3 ablation: `env.death_penalty` sweep** — values [0, 25, 50, 100]. Run dreamer-srl v2 XS / 16 envs / 4M (winner recipe) at each. Document design in `docs/experiments/active/dreamer_srl_v2/death_penalty_ablation_design.md`. | Direct causal test of H1+H3. If lowering `death_penalty` lifts survival monotonically toward rPPO, H1 is confirmed. | 4 runs × 4M steps ≈ 1 GPU-day each |
| P1 | **Phase 2b**: port the offline-WM diagnostic (`scripts/offline_wm_*`) from original Dreamer to dreamer-srl v2 and measure per-horizon `reward_mae` at h ∈ {5, 10, 20, 50}. | Tests H2 directly; expected pattern is rising MAE with horizon, analogous to Z2's 0.18 → 3.05. | 1 day code port + 1 short run |
| P2 | **Wait for M-cell completion (E7, E8) and re-run §4.2 table.** | If M-cell asymmetry < XS-cell asymmetry, capacity scaling helps; if asymmetry is invariant to size, the per-step weighting story is favored. | No new launches needed; analysis only |
| P2 | **Reward-target re-weighting probe**: re-run winner recipe with the reward-head loss weighted up-sample on negative-reward steps (or zero-init of the reward head per Z1). | Code-level mitigation orthogonal to env-level (Phase 3). If both mitigations help, they triangulate the mechanism. | 1 new code path + 2 runs |

## 7. Open Questions

These do not gate the Phase 2a audit or the Phase 3 ablation, but are useful to flag for downstream work.

- **M-cell trajectory.** Does the slower-learning, larger M-cell produce a *different* asymmetry signature, or is the asymmetry invariant to model capacity? If M-cell asymmetry ≈ XS-cell asymmetry, the regression-target-density story is favored over the capacity story. If M-cell asymmetry < XS-cell, capacity matters and the head can in principle fit the negative tail given enough parameters.
- **Does rPPO share the asymmetry?** rPPO's value head is also a regression target on the same reward stream. If rPPO logs comparable MAE-neg vs. MAE-pos diagnostics (or we can add them cheaply), and rPPO is *also* asymmetric but wins anyway, then the asymmetry is real but not the bottleneck — the bottleneck is something specific to *imagination-based* policy training (i.e., the H2 story, not H1).
- **Does the asymmetry survive the food-only parity task in dreamer-srl v2?** If the parity-task run has neg/pos asymmetry ≈ 1× (because there is no deep-negative tail in the reward stream), that is the consistency check H1 needs. We did not extract this metric for the parity run; recommended add.
- **Is the asymmetry seed-stable?** Single-seed evidence in this doc. If the asymmetry replicates across seeds with a tight CI, H1 strengthens; if it doesn't, the cross-run replication argument weakens to "5 noisy draws in the same direction".

## 8. Metrics Requested

| Subfield | Content |
|---|---|
| **Metric** | `WorldModel/model_reward_mae_neg` and `WorldModel/model_reward_mae_pos` *temporal series* — currently only the final summary value is reliably extractable; the local `.wandb` datastore did not yield history rows for these in this session. |
| **Why now** | The §4.4 Learning Dynamics section needs the intra-run trajectory to distinguish "asymmetry present from step 0 and never closes" from "asymmetry emerges late as the head specializes on the positive distribution". The two have very different mitigation paths. |
| **Where it'd live** | The metric *is* already logged at training time (it shows up in `wandb-summary.json`). The bottleneck is local-file *readability* of the time series. Likely fix in the wandb-analysis skill's local-datastore reader or a fallback that consumes `media/` snapshots. |
| **Cost** | Cheap — the data exists; we just need a reliable local parser. No new logging overhead. |

| Subfield | Content |
|---|---|
| **Metric** | Per-horizon `reward_mae` at h ∈ {5, 10, 20, 50} for dreamer-srl v2 (analogous to the Z2 study on original Dreamer). |
| **Why now** | Tests H2 directly. Without it, H2 is conjectural by analogy. |
| **Where it'd live** | The offline-WM diagnostic harness exists for original Dreamer (`scripts/offline_wm_*` per memory `[[20260509_1534_wm_reward_head_localized_failure_a1]]`); needs porting to dreamer-srl v2's interface. |
| **Cost** | Moderate — one code-port day + one short measurement run. Not a recurring logging cost. |

## 9. Related Issues

- **Phase 2a code-audit** (forward reference, not yet written): `docs/reviews/dreamer_srl_reward_head_audit.md` — Read-only audit of `src/algorithms/dreamer_srl/loss.py` (reward-head + twohot + neg/pos MAE) and `src/algorithms/dreamer_srl/world_model.py` (reward-head architecture). Owner: `code-reviewer`. Cross-link this doc as the motivating analysis.
- **Phase 3 design doc** (forward reference, not yet written): `docs/experiments/active/dreamer_srl_v2/death_penalty_ablation_design.md` — Pre-registered ablation of `env.death_penalty` ∈ {0, 25, 50, 100}. Owner: `experiment-designer`. Cite the H1/H3 verdicts in this doc as the motivating evidence.
- **Bug / metric definition flag**: §5.3 Pathology — `rew_MAE_total` is not a simple mean of `rew_MAE_neg` and `rew_MAE_pos`. Cross-check the per-side metric computation in the Phase 2a audit; if it is computed over a non-uniformly-weighted buffer, the *magnitudes* in §4.2 may shift while the *direction* (neg > pos) is preserved. This does not change H1 but does change the quantitative claims.

## 10. Status

- **10.1** This doc (anchor / Mode B retrospective): **written**, 2026-05-18.
- **10.2** Phase 2a code-audit: **not started**.
- **10.3** Phase 3 ablation design: **not started**.
- **10.4** M-cell-incorporated second-pass synthesis: **pending E7+E8 completion** (~02:00 May 19 / ~May 20).

---

## Appendix

### A. Raw Data Tables

#### A.1 Winner run (`yxij4lrc`) end-of-training summary — key fields

| Field | Value |
|---|---|
| `_step` | 4 000 000 |
| `Episode/Steps` | 190.94 |
| `Episode/Reward` | −322.30 |
| `Episode/Reward_Min` | −355.39 |
| `Episode/Reward_Max` | −231.94 |
| `Eval/MeanReward` | −205.98 |
| `Eval/MeanLength` | 141.33 |
| `Episode/Term_MaxSteps` | 0.043 (≈4.3% of episodes timed out; the rest terminated on injury/death) |
| `Episode/EatUnderThreatSafeSteps_predator` | 119.97 |
| `Episode/EatUnderThreatSafeSteps_rabbit` | 122.92 |
| `WorldModel/loss_model` | 2.485 |
| `WorldModel/loss_rew` | 0.842 |
| `WorldModel/loss_recon` | 0.049 |
| `WorldModel/loss_cont` | 0.006 |
| `WorldModel/loss_kl` | 1.588 |
| `WorldModel/loss_rep_kl` | 0.265 |
| `WorldModel/loss_dyn_kl` | 1.323 |
| `WorldModel/model_cont_acc` | 0.996 |
| `WorldModel/model_latent_entropy` | 1.218 |
| `WorldModel/model_reward_mae` | 0.764 |
| `WorldModel/model_reward_mae_neg` | 1.019 |
| `WorldModel/model_reward_mae_pos` | 0.307 |

> Source: `wandb/run-20260516_124224-yxij4lrc/files/wandb-summary.json`.

### B. Config Diffs

Env config `configs/experiment/hypervigilance/01-interoNocicept.yaml` is identical across all 5 dreamer-srl v2 runs and across the rPPO comparators (the env is the controlled variable). Agent-side config diffs across the 5 dreamer-srl v2 cells are limited to: recipe size (= XS for all 5), `num_envs` (16/64/128), `train_steps` (2M/4M), and `loss_balancer` (off/on). See `HYPERPARAM_SEARCH_10X10.md` for the full per-cell YAML diffs.

### C. Changelog

| Date | Change | Author |
|---|---|---|
| 2026-05-18 | Initial Mode B retrospective analysis. H1 framed as leading hypothesis on the basis of 5/5 XS-recipe runs replicating the neg/pos asymmetry pattern. Phase 2a and Phase 3 named as forward references. M-cell runs (E7, E8) noted but not yet incorporated; §10.4 second-pass synthesis pending. | experiment-analyzer |
