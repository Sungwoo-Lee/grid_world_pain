---
title: "NMN Noise-Heterogeneity Sweep — Why does FiLM not beat the unmodulated LN baseline?"
topic: hypervigilance
status: active
created: 2026-05-07
last_updated: 2026-05-07T23:33
phase: 1
wandb_tag: "rppo_nmn_het_<profile>_<arch>_c<cell>"
develop_link: docs/develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md
---

# NMN Noise-Heterogeneity Sweep — Why does FiLM not beat the unmodulated LN baseline?

> **Status**: RUNNING (8/10 cells launched; Runs 1–2 blocked — node 101 missing JAX in grid_world_pain conda env)
> **Date**: 2026-05-07
> **Author**: experiment-designer
> **Related**:
> - [NMN_PERFORMANCE_DIAGNOSIS_v8](../../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md) §6.3 P4 — the recommendation this experiment executes.
> - [NMN_ARCHITECTURE_REVIEW](../../../develop/active/neuromodulation/NMN_ARCHITECTURE_REVIEW.md) §2 — gate mechanism that selective heterogeneity is supposed to drive.
> - [G1' channel-rank diagnostic](../g1_prime_diagnostic/G1_PRIME_CHANNEL_RANK_DIAGNOSTIC.md) — runs in parallel; this sweep's P5 is engineered to guarantee G1' passes.

> **One-line scope.** Sweep the noise heterogeneity of the v8 controlled preset across 5 σ profiles × {FiLM g1, Unmod LN}, single seed each (10 cells), to test whether *any* heterogeneity profile produces a survival gain for FiLM beyond the seed-variance floor (~0.44 steps per v7) — and if not, rule out channel-rank degeneracy as the cause of the v8 null result.

---

## 1. Research Question

Under what (if any) noise-heterogeneity profile does a FiLM-modulated RecurrentPPO agent (LN, grouping_size = 1) exceed the unmodulated LN baseline by more than the seed-variance floor (~0.44 survival steps per v7's 2-seed estimate), when the mean σ across noise-bearing modalities is held constant at the v8 baseline value (0.140) and only the max-σ / min-σ ratio across those modalities is varied?

**Independent variable:** noise-profile heterogeneity ratio R = σ_max / σ_min across the 5 noise-bearing modalities {satiation, interoceptive_nociception, extero_nociception, olfaction, visual}, varied geometrically:

| Profile | R | mean σ (bearing) | Description |
|---|---|---|---|
| P1 | 2.0  | 0.140 | flat anchor (v8 §2.2 control, byte-identical to `01-interoNocicept_noise.yaml`) |
| P2 | 2.75 | 0.140 | mild |
| P3 | 5.0  | 0.140 | moderate |
| P4 | 9.67 | 0.140 | high |
| P5 | 18.0 | 0.140 | extreme — guarantees G1' (max/min over all 10 modalities = ∞ on the σ=0 floor) |

**Dependent variable (primary):** steady-state survival steps over the last 20% of training (windows 8–10 of 10), averaged across episodes within the window.

**Why MC, not GAE** — v8 §4.1.5 showed MC outperforms GAE under noise across the board (+19 to +71 steps). v8 §5.1 Finding 5 also showed the modulator amplifies the MC-GAE gap *against itself for GAE*, so any noise-heterogeneity benefit is most likely to surface under MC. The v8 P0 question ("does FiLM ever help") is answered most strongly under MC.

---

## 2. Hypothesis & Predicted Outcomes

### 2.1 Hypotheses (pre-registered)

> **H₀ (null):** No noise-heterogeneity profile produces a FiLM-vs-Unmod survival gain greater than the v7 seed-variance floor (Δ_SS ≤ 0.44 steps). v8's null result generalises across the heterogeneity gradient.

> **H₁a (heterogeneity-favoured profile):** *At least one* profile (P3, P4, or P5) produces FiLM_SS − Unmod_SS > 5 steps (≈ 11× v7's seed-variance floor). The architecture has latent value that the v8 baseline σ profile happened not to reveal.

> **H₁b (monotonic gradient):** The FiLM-vs-Unmod survival difference Δ_SS(R) is monotonically non-decreasing in R across P1 → P5. Heterogeneity is a graded selective pressure, not a threshold effect.

> **H₁c (v8-confirming, FiLM-failing):** FiLM never beats Unmod on any profile (Δ_SS ≤ 0.44 across all 5 cells). Combined with P5 structurally ruling out channel-rank degeneracy (max/min σ across all modalities > 50, far above any reasonable G1' threshold), this strongly localises the FiLM failure to the *modulator architecture*, not the environment.

### 2.2 What confirms vs. refutes — pre-specified

| Outcome | Predicate | Verdict |
|---|---|---|
| **H₁a confirmed** | max(FiLM_SS − Unmod_SS) over P3–P5 > 5 steps | FiLM has latent value; P4 of v8 §6.3 was a real lead. Trigger H₁b check. |
| **H₁b confirmed** (given H₁a) | Δ_SS(P1) ≤ Δ_SS(P2) ≤ Δ_SS(P3) ≤ Δ_SS(P4) ≤ Δ_SS(P5) (monotone non-decreasing, ties allowed within ±1 step) | Heterogeneity is the controlling variable. Direct follow-up: 3-seed replication on P5. |
| **H₁a confirmed but H₁b refuted** | non-monotone Δ_SS(R) — e.g. P3 wins, P4/P5 lose | An effect exists but is not heterogeneity-driven. Halt; route to senior-developer for diagnosis (could be "sweet spot" R, could be coincidence). |
| **H₁c confirmed (= H₁a refuted)** | max(FiLM_SS − Unmod_SS) over P1–P5 ≤ 0.44 steps; AND Unmod survival monotone or flat (rules out env breaking) | v8's null is preset-robust. The architecture is the bottleneck, not the noise landscape. Hand off to senior-developer for §6.3 P3 (attention/Bayesian modulator) — FiLM's specific limitation (static γ, β) is the structural cause. |
| **Refutation by env breakage** | Unmod survival drops > 30 steps on P5 vs P1 (i.e. P5 is so noisy in one channel that the task itself breaks) | Sweep confounded — heterogeneity is conflated with task difficulty even at matched mean σ. Revisit profile design with a ceiling-clipped σ_max. |
| **Refutation by training instability** | Either run NaNs / gradient explodes / Term_Starvation > 90% / Term_Injury > 50% (a regime not seen in v8) | Single-seed run cannot disambiguate "bad seed" from "bad config". Mark cell as `inconclusive`; do NOT count toward H₁a / H₁c. Hand to senior-developer if ≥ 2 cells fail this way. |

### 2.3 Effect-size threshold rationale

The 5-step Δ_SS threshold for H₁a is set ~11× v7's 0.44-step seed-variance floor. With **single-seed cells**, anything below ~5 steps is plausibly seed noise. The threshold deliberately exceeds the v7 v7-NoNoise seed range *and* a reasonable Bayesian inflation for single-cell noise. If the user prefers a different threshold, change it before launch — it is not extractable post-hoc.

---

## 3. Experimental Design

### 3.1 Independent Variables

| Variable | Values | Rationale |
|---|---|---|
| Noise profile (P1..P5) | 5 levels with R = {2.0, 2.75, 5.0, 9.67, 18.0}, geometric spacing on log R | Geometric spacing gives roughly equal log-spacing on the variable of interest (R), matching the gates' structural sensitivity (sigmoid input scale). |
| Architecture | {FiLM g1, Unmod LN} | g1 chosen because v8 §4.1.3 showed g1 has the highest gate diversity (gamma_multi_std = 1.41 for MC) — the configuration *most likely* to exploit heterogeneity if any can. Larger gSizes attenuate gates and would conflate "FiLM can't" with "g16/g32 specifically can't". |

5 × 2 = 10 cells. **No factorial extension** to GAE or larger groupings is in scope; v8 §4.1.5 + §4.1.3 both argued these directions are dominated.

**Why not a third architecture (e.g. PreActivation g1)?** v8 §6.3 P1 already lists PreActivation under noise as a separate priority. Adding it would push to 15 cells, exceeding the 10-cell budget the user specified.

### 3.2 Controlled Variables

Pinned across all 10 cells. Any deviation invalidates the heterogeneity comparison:

```yaml
# Environment (env body identical to v8 §2.2)
environment:
  height: 10, width: 10, max_steps: 500
  random_start_pos: true
  resources: 2 food (TL) + 2 food (BR)         # v8 §2.2
  predators: 4 hiding_predator (1 per quadrant) + 1 patrolling
  obstacles: 12 rocks (3 per quadrant), 10 bushes (5 TR + 5 BL)
  neutrals: 2 rabbits

body:
  food_nutrition_gain: 6              # v8 §2.2
  death_penalty: 100
  use_homeostatic_reward: true
  injury_smoothing_duration: 3

sensory:
  injury_observable: false            # canonical regime
  nutrition_observable: false
  interoceptive_nociception_enabled: true   # alpha-kernel (tau=3.0, length=12)
  interoceptive_convolution_enabled: true

agent:
  algorithm: RecurrentPPO
  return_mode: MC                     # v8 §4.1.5 — MC dominates under noise
  use_layer_norm: true                # v7/v8 — LN is the performance driver
  rnn_type: GRU
  activation: relu
  encoding_mode: hierarchical
  hidden_size: 128
  sequence_length: 128
  K_epochs: 4
  lr_actor: 0.0005, lr_critic: 0.0001
  entropy_coef: 0.01, eps_clip: 0.1
  num_envs: 128                       # v8 §2.2

# FiLM cells only:
agent.modulation:
  type: FiLM
  mod_hidden_size: 16
  grouping_size: 1
  percept_bias_init: 3.0
  memory_bias_init: 0.0
  temp_clip: [0.5, 3.0]
  memory_clip: [-2.0, 2.0]
```

**Mean σ pinning.** Across P1–P5, the mean σ over the 5 bearing modalities {satiation, intero_noc, extero_noc, olfaction, visual} is held to **0.140 exactly** (= the P1 baseline mean). This is the experiment's central design lever: total information loss is matched, only the *spread* changes. v8 §4.1.4 already showed total noise hurts (-69 to -77 steps); we do not want to re-confirm that.

**Same noise-bearing modality set across profiles.** All 5 profiles assign σ > 0 to the same 5 modalities and σ = 0 to the same 5. We do not move the noise *carrier* across profiles, only the carrier's *amplitude distribution*. This keeps the obs-channel topology stable.

### 3.3 Sample Size & Run Identification

- **10 cells total**: 5 profiles × 2 architectures.
- **1 seed per cell** (seed = 0). The user explicitly chose max-breadth, no replication; the design respects that.
- **Total compute estimate.** rPPO on this env runs at ~5.2M steps/h on a single GPU (v8 anchor); 10M episodes × 500 max-steps × 128 num_envs is the same scale as v8. v8 baselines reached ~12–19M episodes at the analysis cutoff, with a 10M-episode budget. Allocating same: 10 × 10M = 100M episodes total, parallelised across 5 nodes (each 2 GPUs), so wallclock per cell ~24-36 h. **Each cell runs in parallel; all 10 run concurrently.**
- **Why single-seed is acceptable here.** Single-seed cells can refute H₁a (a clean null across 10 cells with no R-trend is informative — it removes heterogeneity from the candidate-cause list) but cannot *confirm* H₁a alone — confirmation routes to a 3-seed replication on the winning profile.
- **Why single-seed is risky here.** v7 reported 0.44-step seed variance, but that was at fixed σ profile. Profile-dependent seed variance is unmeasured. The 5-step Δ_SS threshold for H₁a is set defensively wide for this reason; the H₁c null criterion (Δ_SS ≤ 0.44) is set at the v7 floor and is correspondingly fragile under single-seed.

### 3.4 Confounds & Limitations

| Confound | Affected | Severity | Mitigation / Decision |
|---|---|---|---|
| **Single seed per cell.** v7's 0.44-step seed-variance floor is at fixed profile; profile-dependent variance is unknown. A 4-step difference on one profile could be seed noise. | All 10 | High | Pre-registered: H₁a requires Δ_SS > 5 steps (≈ 11× the v7 floor) to count as a hit; H₁c (the null) is set at the v7 floor (≤ 0.44) and is fragile. If H₁c-by-near-miss (e.g., max Δ_SS = 0.6), declare *inconclusive*, not refute. |
| **Noise-config index-mismatch bug** (v8 §2.3 row 4). | Originally flagged as Med-High in v8. **Designer audit (this doc):** read `src/environment/config_loader.py:426–471` + `src/environment/sensor.py:222–235`. The current code maps modality name → index by YAML key order via `params.noise_modality_order`, then `sensor.py` does name-keyed lookup `idx = modality_map[sensor_name]`. **The bug is not present in current code.** | All 10 | **Decision: option (a) — proceed.** The original bug is patched; even if a residual mismatch exists at clip-min/max bounds (which are also name-keyed), all 10 runs are equally affected. Relative comparisons across cells hold. We do not halt for a P0 fix. |
| **Mean-σ pinning is approximate** at the displayed precision (0.140 ± rounding). | All 10 | Low | Profiles re-checked: P1 sum = 0.70, P2 = 0.70, P3 = 0.70, P4 = 0.70, P5 = 0.70 (all integer cents → exact). |
| **σ on σ=0 floor channels (collision, proprioception, location, injury, nutrition) is the same across all 5 profiles** — so the *full* max/min ratio is ∞ for every profile. The "heterogeneity gradient" only varies over the 5 bearing modalities. | All 10 | Low — by-design | This is correct: σ=0 channels carry no learnable gating signal (no per-channel gradient on σ=0), so they are "free" against heterogeneity as a learnable variable. The gradient on R that the agent's FiLM head can reason over is *only* over the bearing modalities. |
| **FiLM g1 may not be the right architectural choice.** v8 found gamma_multi_std (gate diversity) is highest at g1, but it's possible coarser groupings handle heterogeneity differently. | FiLM cells | Med | Pre-registered: this is the best single FiLM choice given v8 evidence. If the sweep returns H₁a with non-monotone shape, follow-up should re-run on g4 or g16. Out of this experiment's scope. |
| **G1' diagnostic runs in parallel** (separate Phase-0 work). If G1' returns RATIO ≤ 2 on the canonical preset, this entire sweep's premise is undercut — it would mean *no* profile in the gradient is observably heterogeneous through the encoder. | All 10 | Med | Decision: launch independently. P5 is structurally constructed (max/min over all 10 modalities = ∞ on the σ=0 floor; max/min over bearing = 18×) so even if G1' fails on P1 (the canonical preset), P5 is virtually certain to pass. The sweep is its own diagnostic of the heterogeneity dimension. |
| **10M episodes may be too few for FiLM to converge under the most heterogeneous profiles**. v8's MC FiLM g1 was at ~15M episodes when reported (≈ converged window 5+), but this sweep budgets 10M. | FiLM cells, P3–P5 especially | Med | Mitigation: temporal evolution check (§5) — if FiLM is still climbing at window 10 on the heterogeneous profiles while Unmod has plateaued, declare the cell *inconclusive for H₁a* (pre-empted: Δ_SS could grow beyond threshold), not a confirmed null. |

### 3.5 Bug-handling decision (called out at user request)

**Option (a): proceed** — the v8-flagged "noise-config index-mismatch bug" is **not present in the current source** (`src/environment/config_loader.py:432–471` + `src/environment/sensor.py:222–235` use name-keyed mapping via `params.noise_modality_order`, not positional). Current code routes the configured σ to the configured modality. The v8 doc's confound row is stale; this audit closes it.

This audit was done by reading the loader and sensor code; if the user wants an independent verification before launch, route to `code-reviewer` (5-minute scope: confirm `noise_modality_order` invariants).

If the bug *had* been present and fatal, option (b) — halt and route to senior-developer — would have been mandatory because the heterogeneity gradient depends on σ landing on the labelled modality. P5's "visual σ = 0.36, satiation σ = 0.02" claim is meaningless under positional misalignment.

---

## 4. Launch Manifest

System-of-record for all 10 runs. **Designer columns locked.** Runner fills `Status / Launched at / WandB run ID / Log path` at launch time. Node + GPU pre-assigned by user.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | blocked | p1_unmod | `rppo_nmn_het_p1_unmod_c1` | nmn_noise_heterogeneity | prod | 0 | 101 | cuda:0 | — | — | — |
| 2 | blocked | p1_film_g1 | `rppo_nmn_het_p1_film_g1_c2` | nmn_noise_heterogeneity | prod | 0 | 101 | cuda:1 | — | — | — |
| 3 | running | p2_unmod | `rppo_nmn_het_p2_unmod_c3` | nmn_noise_heterogeneity | prod | 0 | 102 | cuda:0 | 2026-05-07T23:21:09 | q2o3vydr | logs/20260507_232109.log |
| 4 | running | p2_film_g1 | `rppo_nmn_het_p2_film_g1_c4` | nmn_noise_heterogeneity | prod | 0 | 102 | cuda:1 | 2026-05-07T23:21:30 | oh4aws0e | logs/20260507_232130.log |
| 5 | running | p3_unmod | `rppo_nmn_het_p3_unmod_c5` | nmn_noise_heterogeneity | prod | 0 | 103 | cuda:0 | 2026-05-07T23:21:53 | 6eqjr62y | logs/20260507_232153.log |
| 6 | running | p3_film_g1 | `rppo_nmn_het_p3_film_g1_c6` | nmn_noise_heterogeneity | prod | 0 | 103 | cuda:1 | 2026-05-07T23:22:12 | tp0vz8gb | logs/20260507_232212.log |
| 7 | running | p4_unmod | `rppo_nmn_het_p4_unmod_c7` | nmn_noise_heterogeneity | prod | 0 | 104 | cuda:0 | 2026-05-07T23:22:31 | cicjphwf | logs/20260507_232231.log |
| 8 | running | p4_film_g1 | `rppo_nmn_het_p4_film_g1_c8` | nmn_noise_heterogeneity | prod | 0 | 104 | cuda:1 | 2026-05-07T23:22:50 | ay78aocv | logs/20260507_232250.log |
| 9 | running | p5_unmod | `rppo_nmn_het_p5_unmod_c9` | nmn_noise_heterogeneity | prod | 0 | 105 | cuda:0 | 2026-05-07T23:23:09 | mb0huxq1 | logs/20260507_232309.log |
| 10 | running | p5_film_g1 | `rppo_nmn_het_p5_film_g1_c10` | nmn_noise_heterogeneity | prod | 0 | 105 | cuda:1 | 2026-05-07T23:23:28 | rq7voqt8 | logs/20260507_232328.log |

**Tag pattern:** `rppo_nmn_het_<profile>_<arch>_c<cell>`.
- `<profile>` ∈ {p1, p2, p3, p4, p5}
- `<arch>` ∈ {unmod, film_g1}
- `c<cell>` ∈ {c1..c10} — included so that `pgrep -af '<TAG>'` is unambiguous even if a tag substring is reused elsewhere.

All Tags are unique within the manifest. Tag = wandb-name per project memory; runner copies verbatim. wandb-group = `nmn_noise_heterogeneity` (= the new dir under `configs/experiment/`).

### 4.1 Configs to Produce (designer scope — written by this doc)

| Run | Cell | Config (env) | Config (agent) |
|---|---|---|---|
| 1 | p1_unmod   | `configs/experiment/nmn_noise_heterogeneity/p1_flat.yaml`     | `configs/models/recurrent_ppo_nmn_het_unmod.yaml` |
| 2 | p1_film_g1 | `configs/experiment/nmn_noise_heterogeneity/p1_flat.yaml`     | `configs/models/recurrent_ppo_nmn_het_film_g1.yaml` |
| 3 | p2_unmod   | `configs/experiment/nmn_noise_heterogeneity/p2_mild.yaml`     | `configs/models/recurrent_ppo_nmn_het_unmod.yaml` |
| 4 | p2_film_g1 | `configs/experiment/nmn_noise_heterogeneity/p2_mild.yaml`     | `configs/models/recurrent_ppo_nmn_het_film_g1.yaml` |
| 5 | p3_unmod   | `configs/experiment/nmn_noise_heterogeneity/p3_moderate.yaml` | `configs/models/recurrent_ppo_nmn_het_unmod.yaml` |
| 6 | p3_film_g1 | `configs/experiment/nmn_noise_heterogeneity/p3_moderate.yaml` | `configs/models/recurrent_ppo_nmn_het_film_g1.yaml` |
| 7 | p4_unmod   | `configs/experiment/nmn_noise_heterogeneity/p4_high.yaml`     | `configs/models/recurrent_ppo_nmn_het_unmod.yaml` |
| 8 | p4_film_g1 | `configs/experiment/nmn_noise_heterogeneity/p4_high.yaml`     | `configs/models/recurrent_ppo_nmn_het_film_g1.yaml` |
| 9 | p5_unmod   | `configs/experiment/nmn_noise_heterogeneity/p5_extreme.yaml`  | `configs/models/recurrent_ppo_nmn_het_unmod.yaml` |
| 10 | p5_film_g1 | `configs/experiment/nmn_noise_heterogeneity/p5_extreme.yaml`  | `configs/models/recurrent_ppo_nmn_het_film_g1.yaml` |

All 7 configs (5 env + 2 agent) are NEW. No schema additions; all keys read by existing `config_loader.py` + `train.py:204-280` paths.

### 4.2 Reference launch command (template the runner adapts per row)

For Run 1 (p1_unmod, node 101, cuda:0):
```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/experiment/nmn_noise_heterogeneity/p1_flat.yaml \
  --agent_config configs/models/recurrent_ppo_nmn_het_unmod.yaml \
  --num-envs 128 \
  --episodes 10000000 \
  --checkpoint-frequency 100000 \
  --seed 0 \
  --device cuda:0 \
  --log-interval 10 \
  --wandb-group nmn_noise_heterogeneity \
  --wandb-job-type prod \
  --wandb-name "rppo_nmn_het_p1_unmod_c1" \
  --tag "rppo_nmn_het_p1_unmod_c1"
```

For each other row, swap (config, agent_config, device, wandb-name, tag) per §4.1 and the manifest's GPU column, and SSH/run on the assigned node per the manifest's Node column.

---

## 5. Analysis Plan (Pre-Specified)

### 5.1 Primary statistic

For each cell, **steady-state survival steps (SS_SS)** = mean episodic survival over the **last 20% of episodes** (windows 8–10 of 10 in v8's window scheme, i.e. ~last 2M of 10M episodes). This matches v8's analysis convention exactly.

For each profile P_k, **the cell-level effect** is:
$$\Delta_{\text{SS}}(P_k) = \mathrm{SS}_{\text{SS}}(\text{FiLM}, P_k) - \mathrm{SS}_{\text{SS}}(\text{Unmod}, P_k)$$

Single-seed → no within-cell CI. Across-profile pattern is the multi-cell evidence.

### 5.2 Effect-size thresholds (pre-registered)

| Predicate | Threshold | Decision |
|---|---|---|
| H₁a (any-profile gain) | max_k Δ_SS(P_k) > **5 steps** AND on profiles P3–P5 | confirm H₁a; trigger 3-seed replication on the winner |
| H₁b (monotone-in-R) | Δ_SS(P_1) ≤ Δ_SS(P_2) ≤ Δ_SS(P_3) ≤ Δ_SS(P_4) ≤ Δ_SS(P_5), tolerance ±1 step on each pairwise comparison | confirm H₁b; report as "heterogeneity is the controlling axis" |
| H₁c (full null) | max_k \|Δ_SS(P_k)\| ≤ **0.44 steps** AND Unmod survival in P5 within 30 steps of P1 | confirm H₁c (= refute H₁a robustly) |
| Inconclusive band | 0.44 < max_k Δ_SS(P_k) ≤ 5 steps | inconclusive single-seed; recommend 3-seed replication on the leading profile |

### 5.3 Temporal evolution check (mandatory per project conventions)

Per project_plan convention, end-of-training snapshots alone do not constitute analysis. For each cell, plot SS by window (10 equal-width bins over training) and check:

1. **Convergence by window 8.** If FiLM cell still climbing > 5 steps from window 8 → 10, mark *inconclusive for H₁c* (the null could disappear with more steps).
2. **Mid-training divergence.** If FiLM > Unmod in windows 1–4 but Unmod overtakes by window 7+, that is *prima facie* evidence the modulator helps early but not asymptotically — note as a separate finding (§5.3 of analysis-phase fill-in).
3. **Gate dynamics over training.** For FiLM cells, plot gamma_multi_mean and gamma_multi_std over windows. Heterogeneous profiles should produce *more diverse* gates if the architecture is exploiting heterogeneity. v8 §4.3.2 reported g1 gamma_multi_std = 1.41 on P1; if P5 produces gamma_multi_std > 1.5 (or a pattern of channel-selective near-zero gates on the noisy modalities), that is supporting evidence even when survival is null.

### 5.4 Diagnostic fingerprints per profile

Beyond survival, log and report the following per cell at SS:

| Metric | Source | What it diagnoses |
|---|---|---|
| `gamma_multi_mean`, `gamma_multi_std` | NMN logger (existing) | Whether FiLM gate is differentiating across modalities under heterogeneity (should rise from P1 → P5 if architecture is exploiting). FiLM cells only. |
| `gamma_uni_mean`, `gamma_uni_std` | NMN logger (existing) | Same for unimodal injection. |
| `temp_mean`, `temp_min`, `temp_max` | NMN logger (existing) | v8 found MC FiLM saturates temp at 3.0 ceiling; track whether heterogeneity changes this. |
| `loss/value`, `loss/grad_norm`, `loss/entropy` | trainer (existing) | Standard loss diagnostics. |
| Term_Injury / Term_Starvation / Term_MaxSteps | trainer (existing) | Decomposition of survival outcome. v8 §4.1.1 used these to characterise *how* agents die. |
| FoodEaten, TotalDamage | trainer (existing) | Behavioural decomposition; differentiates "starve from no-food" vs "die from predator/rocks". |

All of these are already logged by the existing pipeline; **no new metrics requested for this experiment.**

### 5.5 Cross-cell comparisons (specific tests)

1. **Δ_SS(R) trend** — primary plot. x-axis = log R; y-axis = Δ_SS. 5 points + 95%-trimmed range. Annotate the 5-step H₁a threshold.
2. **Unmod_SS(R) trend** — control plot. Should be flat or slightly declining (mean σ pinned, but very heterogeneous σ_max may hurt the unmodulated agent too if the noisy channel is necessary). If Unmod drops > 30 steps P1 → P5, sweep is confounded (cf. §3.4 row 6).
3. **gamma_multi_std(R) trend** — FiLM cells only. If the architecture is exploiting heterogeneity at all, this should rise with R. If it's flat, the modulator is *not* learning differential per-channel weighting even given the opportunity.
4. **temp_mean(R) trend** — FiLM cells only. If FiLM hits the temp ceiling on every profile, the temperature head is structurally non-functional under MC (already a concern from v8 §4.3.3). If it falls below 3.0 on some profile, that profile is engaging temperature modulation in a way the v8 baseline did not.

### 5.6 Reporting format

Fill §4–§6 of this doc post-launch, following [docs/TEMPLATES/training_analysis.md](../../../TEMPLATES/training_analysis.md). Hand to `experiment-analyzer` for the actual fill-in.

---

## 6. Failure-Mode Catalog (pre-registered)

| Failure mode | Pre-registered interpretation |
|---|---|
| **All 10 cells converge to ≈ v8 §4.1.1 values** (Unmod_SS ≈ 283, FiLM_g1_SS ≈ 282 on every profile) | H₁c confirmed strongly. v8's null is preset-robust. **Hand off to senior-developer** for §6.3 P3 (architecture redesign — attention or Bayesian filter). The v8 doc's P4 lead is closed: heterogeneity is not the missing ingredient. |
| **Δ_SS(R) is monotone-positive and crosses the 5-step threshold at P3 or P4** | H₁a + H₁b confirmed. Trigger `experiment-designer` to author a **3-seed replication on the winning profile** under MC + g1 (and possibly g4). Survival benefit must replicate before architectural claims can be made. |
| **Δ_SS spikes at one mid-gradient profile (e.g. P3) but is null at P4 and P5** | H₁a confirmed but H₁b refuted. Plausible explanations: (i) coincidental seed effect, (ii) "sweet spot" where heterogeneity helps but extreme heterogeneity drowns out the carrier modality the agent relies on. **Hand back to designer** for a denser sweep around the spike. Single-seed status means this could equally be coincidence. |
| **Δ_SS(R) is monotone-positive but small (max < 5 steps)** | Inconclusive. The shape suggests an effect but the size cannot be distinguished from seed noise at single-seed. Recommend 3-seed replication on P5 to disambiguate. |
| **NaN / gradient explosion in any cell** | Mark `inconclusive` for that cell; do NOT count toward H₁a/c. v8 §5.3 had GAE gradient spikes (1.1M); MC was clean. If MC produces NaN here on a heterogeneous profile, the noise σ_max may be hitting a numerically unstable regime via state-dependent scaling (`σ * (1 + 1.5 * injury / 100)` → at injury = 100, σ_eff = 2.5σ; for visual P5 σ=0.36, σ_eff_max = 0.9 — well within numerical bounds). If NaN occurs, route to senior-developer; it would suggest a value-loss explosion under genuinely heterogeneous noise, not a sweep-design bug. |
| **Term_Starvation > 90%** in any FiLM cell while same Unmod cell stays at v8-typical ~65% | The modulator is *actively breaking* foraging on that profile (worse than v8 found; v8 had Term_Starvation 64–69% across all configs). This is FiLM-specific pathology in the heterogeneous regime — note as a finding even if survival shape supports H₁c. |
| **Both Unmod and FiLM tank on P5** (Unmod_SS_P5 < 220) | P5 is too extreme; the visual σ_max = 0.36 is breaking the task even for the unmodulated agent. Sweep is confounded between "heterogeneity helps" and "task is broken". **In this case, P5 must be dropped from the analysis** and the inference is restricted to P1–P4. |
| **MC temp saturates at 3.0 ceiling on every FiLM cell** | Confirms v8 §4.3.3 finding (temp head is non-functional under MC). Independent of survival outcome, this is a finding. Hand to senior-developer with the v8-noted suggestion to raise `temp_clip` ceiling. |
| **All 10 cells fail to reach 10M episodes** within the wallclock budget (e.g. 36 h) | Operational, not scientific. Re-allocate compute or accept reduced steps; declare any cell <8M episodes inconclusive for the convergence question. |

---

## 7. Metrics Requested (none for this experiment)

All diagnostics in §5 are logged by the existing pipeline. **No code changes are needed** to run this experiment. The configs in §4.1 use only existing schema keys (verified against `src/utils/config.py`, `src/environment/config_loader.py`, `train.py:204–280`, `src/models/recurrent_ppo_network.py:204–276`).

---

## 8. Hand-offs

### 8.1 env-config-auditor (next agent)

Audit the 5 env configs and 2 agent configs in §4.1 for:
- `perceptual_noise.enabled: true` in all 5 env configs.
- σ-block matches the profile labels in §1 (P1: 0.10/0.10/0.10/0.20/0.20 on bearing channels; P5: 0.02/0.04/0.10/0.18/0.36; etc.).
- Mean σ across the 5 bearing modalities = 0.140 in every profile (sum = 0.70).
- All 10 modality keys in the same order across all 5 configs (canonical YAML order).
- `agent.use_layer_norm: true` in both agent configs.
- `agent.modulation.type: null` in `recurrent_ppo_nmn_het_unmod.yaml`.
- `agent.modulation.type: "FiLM"`, `grouping_size: 1`, `mod_hidden_size: 16`, percept_bias_init: 3.0`, `temp_clip: [0.5, 3.0]`, `memory_clip: [-2.0, 2.0]` in `recurrent_ppo_nmn_het_film_g1.yaml`.
- `agent.return_mode: "MC"` in both agent configs.
- All 10 manifest tags are unique strings.
- `env-config-auditor` may also confirm the §3.5 bug-handling decision by reading `src/environment/config_loader.py:432–471` + `src/environment/sensor.py:222–235` and verifying the name-keyed mapping.

### 8.2 user (gating step)

After auditor passes, the user authorises launch. The runner does not auto-launch; user explicitly approves.

### 8.3 training-runner

Launches each manifest row using §4.2's command template (per-row substitutions per §4.1 + manifest Node/GPU columns). One `run_command.py` invocation per row. Per project memory: post-launch `pgrep -af '<TAG>'` on each node to verify exactly one PID per tag — duplicates are halt-conditions. Uses /tmp CIFS-bypass on each node per current convention.

### 8.4 experiment-analyzer (post-training)

After all 10 cells reach the SS window (~window 8 of 10), pulls:
- Steady-state survival per cell
- Δ_SS(R) trend
- gamma_multi_std(R) for FiLM cells
- temp_mean(R) for FiLM cells
- Term_* breakdown per cell
- v8-comparison table (this sweep's P1 cells should match v8's MC_Unmod_Noise_LN ≈ 283.64 and MC_FiLM_Noise_g1 ≈ 282.80 within ~5 steps; large deviation on P1 is a replication failure).

Then fills §4–§6 of this doc against the §2.2 / §6 pre-registered predicates.

### 8.5 experiment-designer (results phase, looped back)

Designer (this agent) returns to fill §6 Conclusions and §6.3 Next Experiments based on the analyzer's output, applying the §2.2 + §6 confirmation/refutation rules without ambiguity (decisions were locked before launch).

---

## 9. Cross-references

- [NMN_PERFORMANCE_DIAGNOSIS_v8](../../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md) — anchor diagnosis. §6.3 P4 is the hypothesis this sweep tests.
- [NMN_ARCHITECTURE_REVIEW](../../../develop/active/neuromodulation/NMN_ARCHITECTURE_REVIEW.md) — structural account of why heterogeneity *should* matter for FiLM.
- [G1' channel-rank diagnostic](../g1_prime_diagnostic/G1_PRIME_CHANNEL_RANK_DIAGNOSTIC.md) — runs in parallel; this sweep's P5 structurally guarantees G1' passes.
- [project_plan §Open work items](../../../project/project_plan.md) — Phase 1 anchor for "noise landscape determines whether the AI fixed point is non-degenerate." This sweep is one of the cheapest tests of that question on the NMN side.
- [docs/TEMPLATES/training_analysis.md](../../../TEMPLATES/training_analysis.md) — template this doc follows.

---

## Appendix

### A. Profile-construction arithmetic

Bearing modalities (5): satiation, interoceptive_nociception, extero_nociception, olfaction, visual.
Σ-pinning constraint: sum of 5 σ values = 0.70 (mean = 0.140).

| Profile | σ_satiation | σ_intero_noc | σ_extero_noc | σ_olfaction | σ_visual | sum | mean | min | max | R = max/min |
|---|---|---|---|---|---|---|---|---|---|---|
| P1 | 0.10 | 0.10 | 0.10 | 0.20 | 0.20 | 0.70 | 0.140 | 0.10 | 0.20 | 2.00 |
| P2 | 0.08 | 0.10 | 0.12 | 0.18 | 0.22 | 0.70 | 0.140 | 0.08 | 0.22 | 2.75 |
| P3 | 0.05 | 0.08 | 0.12 | 0.20 | 0.25 | 0.70 | 0.140 | 0.05 | 0.25 | 5.00 |
| P4 | 0.03 | 0.06 | 0.12 | 0.20 | 0.29 | 0.70 | 0.140 | 0.03 | 0.29 | 9.67 |
| P5 | 0.02 | 0.04 | 0.10 | 0.18 | 0.36 | 0.70 | 0.140 | 0.02 | 0.36 | 18.00 |

log R sequence: 0.69, 1.01, 1.61, 2.27, 2.89 — roughly equally spaced (Δ log R = 0.32, 0.60, 0.66, 0.62), so the gradient is approximately log-uniform.

For all profiles the σ=0 channels (injury, nutrition, collision, proprioception, location) keep σ=0 with `injury_noise_scale` matching the v8 baseline; mode is unchanged from v8 §2.2. The full max/min ratio over all 10 modalities is technically ∞ for every profile (because the σ=0 channels are present).

### B. Config Diffs

The 5 env configs differ only in the `perceptual_noise.modalities.<channel>.sigma` lines (5 lines per config). The 2 agent configs differ only in the `agent.modulation` block (and the descriptive header comment).

P1 env config is byte-equivalent to `configs/experiment/hypervigilance/01-interoNocicept_noise.yaml` modulo formatting/comments.

### C. Changelog

| Date | Change | Author |
|---|---|---|
| 2026-05-07 | Initial design + 7 configs (5 env + 2 agent) + 10-row launch manifest | experiment-designer |
