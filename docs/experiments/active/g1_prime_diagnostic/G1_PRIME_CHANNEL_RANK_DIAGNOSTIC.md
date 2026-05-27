---
title: "G1' Channel-Rank Non-Degeneracy Diagnostic — Go/No-Go on the Canonical Noise Preset"
topic: g1_prime_diagnostic
status: active
created: 2026-05-07
last_updated: 2026-05-07
phase: 1
wandb_tag: "g1prime_lnbaseline_<preset>_s<seed>"
develop_link: docs/project/directions/architecture_for_pain_computation.md
---

# G1' Channel-Rank Non-Degeneracy Diagnostic — Go/No-Go on the Canonical Noise Preset

> **Status**: COLLECTING (design only — no runs launched, no code written)
> **Date**: 2026-05-07
> **Author**: experiment-designer
> **Related**:
> - [Cross-professor synthesis (Hybrid Option A + B-on-H5)](../../../project/ideas/nature_mi_paper_framing_synthesis.md) §1.5
> - [RL-BDL §5.2 — the cheaper diagnostic](../../../project/directions/architecture_for_pain_computation.md#52-the-cheaper-diagnostic)
> - [Bayesian-brain memo §3.2(a) — G1 restated as channel-rank non-degeneracy](../../../project/concepts/active_inference_hypervigilance.md#a-heterogeneous-noise-landscape--phase-1)
> - Anchor v8 LayerNorm baseline diagnosis: [NMN_PERFORMANCE_DIAGNOSIS_v8](../../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md)

> **One-line scope.** A one-hour wallclock measurement on a *trained* LayerNorm baseline checkpoint that decides whether the canonical noise preset has the channel-wise residual-variance heterogeneity that the active-inference fixed point requires. Failure gates the entire architectural program until the noise preset is retuned.

---

## 1. Research Question

Under the canonical noise preset, does an unmodulated LayerNorm-baseline RecurrentPPO agent's encoder produce **per-modality observation-reconstruction residuals whose maximum-to-minimum MSE ratio under high injury (`state.injury_level > 0.5 * max_injury`) exceeds 2 across the modality set** {Injury, Nutrition, Satiation, Interoceptive Nociception, Extero Nociception, Olfaction, Collision, Proprioception, Visual, Location}, when measured by a frozen-policy / lightly-fit reconstruction head?

> **H₀** (null, refutation outcome): max-min per-modality MSE ratio under `injury > 0.5` is ≤ 2 — the canonical noise preset is *channel-rank degenerate* and the AI fixed point under this preset is degenerate. **No further architectural work (Phase 2/3 modulator + precision head) is licensed under this preset.**
> **H₁** (alternative, confirmation outcome): max-min per-modality MSE ratio under `injury > 0.5` exceeds 2 — the preset has the heterogeneity required to make a precision-weighting modulator measurable. The architectural program proceeds.

The hypothesis is a **single scalar threshold check**, not a survival comparison. Survival steps remains the project's headline metric for downstream policy comparisons; **here the headline metric is the per-modality MSE ratio**, by design (per the synthesis memo §1.5 and RL-BDL §5.2).

### 1.1 Why this is a go/no-go gate, not a "result"

This experiment is *not* an architecture comparison. It is a **measurement of the environment** (re-derived through the encoder's residuals). The Bayesian-brain §3.2(a) statement is theoretically sharp: if the per-channel residual variance ratio under high injury is degenerate, the AI fixed point is degenerate, and *no Phase-3 architecture* — modulator, precision head, or otherwise — can break that degeneracy. Conversely, if the ratio is non-degenerate, the noise landscape is *necessary-and-as-yet-unfalsified* for the AI account; it does not by itself confirm the architectural program will work, but it removes the cheapest mode of failure.

**Failure (ratio ≤ 2) → halt the architectural program. Retune the noise preset (Phase 1') before re-running this diagnostic. Loop until pass.**

---

## 2. Experimental Design

### 2.1 Independent Variables

This is a single-cell measurement. There is no across-cell independent variable.

| Variable | Values | Rationale |
|---|---|---|
| (none — single cell) | — | Diagnostic is a one-time measurement on the canonical noise preset. |

### 2.2 Controlled Variables

Everything is fixed.

```yaml
# Environment / noise (the canonical preset — see §6 for the preset-identification halt)
config: configs/experiment/<canonical_preset>.yaml   # see §6

# Agent (LayerNorm baseline, NO modulator)
agent_config: configs/models/recurrent_ppo/recurrent_ppo.yaml
agent.use_layer_norm: true
agent.modulation.type: null
agent.encoding_mode: hierarchical
agent.rnn_type: GRU
agent.activation: relu
agent.return_mode: MC          # v8 used MC; consistent with v7/v8 prior

# Training horizon (for the policy that produces the frozen encoder)
total-timesteps: 10_000_000     # 10M, matches v7/v8 baseline horizon for steady-state

# Frozen-head MLP (the measurement device)
g1_prime_logger:
  enabled: true
  head_hidden_dims: [128, 128]   # two-layer MLP per RL-BDL §5.2
  head_input: "encoder_recurrent_hidden"  # h_task = recurrent hidden state of the frozen policy
  head_target: "obs_t+1"         # next-step full-obs vector; per-modality MSE computed against per-modality slice
  head_fit_steps: 1_000_000      # 1M env steps with frozen encoder; head-only gradient
  head_fit_lr: 0.001
  measure_steps: 200_000         # held-out 200k steps for MSE measurement (no head update)
  injury_bin_high_threshold: 0.5    # > 0.5 * max_injury counted as "high injury"
  injury_bin_low_threshold: 0.1     # < 0.1 * max_injury counted as "low injury"
  modality_breakdown_source: "src/environment/sensor.py::get_observation_breakdown"
  ratio_threshold: 2.0              # the go/no-go threshold
```

### 2.3 Confounds & Limitations

| Confound | Affected | Severity | Mitigation |
|---|---|---|---|
| **No published trained LayerNorm baseline checkpoint** under the canonical preset for this rewrite scope. v8 baselines (`ptje3he2`, `auuuayj6`) used the same sigma profile but were single-seed, and may pre-date sensor-channel additions. | All | **High — blocking** | Surface to user before launch (see §6, §7). If checkpoint unavailable, the diagnostic requires first training the LN baseline (1-3 seeds × 10M steps), which inflates wallclock from "1h" to "≈1 day". This is an unavoidable precondition. |
| Per-modality dimensionality is heterogeneous (Olfaction = 5; Visual ~ several dozen; Injury = 1). A naive sum-of-squared-errors over a high-dim modality will look "noisier" than a 1-D modality independent of true reliability. | All | High | **Use mean per-channel MSE within a modality, not summed.** I.e. for modality $i$ with dims $d_i$, $\mathrm{MSE}_i = \frac{1}{d_i} \mathbb{E}\bigl[\|\hat{o}_i - o_i\|_2^2\bigr]$. Spec'd in §3 logger contract. |
| Frozen-policy choice excludes any feedback from injury → policy → injury loop. The LN baseline does generate injury through behaviour, but a head fit on its own data window may sample an off-distribution low-injury / high-injury split if the policy rarely visits high-injury states. | All | Medium | Require ≥ 1k transitions per injury bin in the measurement window; if not met, extend `measure_steps` or reject the run as inconclusive (not a falsification). Logger pre-registers this minimum. |
| Reconstruction MSE on **noisy** observations conflates encoder representational capacity with the noise floor itself. The "true target" is the noise-free observation; the agent only ever sees the noisy one. | All | Medium | Use **noise-free** $o^{\text{true}}_{t+1}$ as the head target (computed from the env state with `apply_noise=False` — already supported by `sensor.py::get_observation`). The head's job is to recover the clean signal from a representation built on noisy input. This is the *theoretically correct* target for the AI residual variance: reliability $1/\Sigma_i$ refers to the noise-free signal, not the noisy observation. **Implementation note for `developer`:** the env already provides `get_observation(..., apply_noise=False)`; the logger calls it once per step alongside the standard noisy obs. |
| Head random-init vs. fit window: results may depend on the lightly-trained head. | All | Low–Med | Pre-register: report MSE both at head-init (fully random head — sanity check that the encoder *can* be linearly probed) **and** after the 1M-step fit. Threshold is applied to the post-fit numbers; the random-init numbers are diagnostic context. |
| Single-seed measurement of a single-seed-trained policy. The LN baseline's encoder is itself a sample. | All | Medium | Run 3 seeds (see §2.4); report mean ± range across seeds; pass criterion is **median seed's ratio > 2** (robust to one outlier). |
| The `injury_observable: false` regime hides the underlying injury level from the policy. Policy never sees `state.injury_level` directly — only `interoceptive_nociception` (alpha-kernel-convolved). | All | Low | Bin the measurement on the **hidden** `state.injury_level`, not on the observed channel. The bin is a *labelling* operation outside the agent loop and is an honest function of the env state. |

### 2.4 Sample Size & Run Identification

- **3 seeds** (0, 1, 2) on the LayerNorm baseline. Three seeds is the minimum for a "median ratio > 2" robust pass criterion; the synthesis memo §1.5 and the project's seed-count convention support ≥ 3.
- Each seed: one 10M-step LN policy training run, then one head-fit + measure pass. **If a recent v8 LN-baseline checkpoint can be reused (see §6 dependency)**, the policy-training step is skipped; only the head-fit + measure pass runs (~1 hour each).
- Total wallclock estimate (rPPO at ~5.2M steps/hour, single GPU):
  - **With reusable LN checkpoint:** 3 × ~1 h head-fit + measure ≈ 3 h.
  - **Without reusable LN checkpoint:** 3 × (10M / 5.2M ≈ 2 h policy train + 1 h head pass) ≈ 9 h. Parallelisable across 3 GPUs to ~3 h.

---

## 3. Launch Manifest

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | planned | g1prime_lnbaseline | `rppo_g1prime_lnbaseline_<preset>_s0` | g1_prime | diagnostic | 0 | — | — | — | — | — |
| 2 | planned | g1prime_lnbaseline | `rppo_g1prime_lnbaseline_<preset>_s1` | g1_prime | diagnostic | 1 | — | — | — | — | — |
| 3 | planned | g1prime_lnbaseline | `rppo_g1prime_lnbaseline_<preset>_s2` | g1_prime | diagnostic | 2 | — | — | — | — | — |

> The placeholder `<preset>` in the tag is replaced when §6 is resolved. Tag and wandb-name are **identical strings** per row, locked here. Designer-locked, not runner-overrideable.

> Single Cell because the diagnostic varies only seed. Three seeds supports a "median ratio > 2" robust pass criterion.

> **wandb-job-type** = `diagnostic` rather than `prod` — these runs are not headline experiments; they are a one-shot environment characterisation.

### 3.1 Configs to Produce (designer-only, pre-launch)

| Run | Config (env) | Config (agent) |
|-----|--------------|----------------|
| 1 | `configs/experiment/<topic>/<canonical_preset>.yaml` (re-used; canonical preset name pending §6 resolution) | `configs/models/recurrent_ppo/recurrent_ppo_g1prime.yaml` (new — adds `g1_prime_logger:` block; otherwise identical to `recurrent_ppo.yaml`) |
| 2 | (same as Run 1) | (same as Run 1) |
| 3 | (same as Run 1) | (same as Run 1) |

**Why a new agent config rather than CLI flags.** The frozen-head logger introduces ~10 mandatory keys (`g1_prime_logger.enabled`, MLP dims, head fit steps, injury thresholds, ratio threshold, etc.). Per project convention, mandatory keys live in YAML and are loaded with `config.get_mandatory()`. CLI flags would dilute the no-fallback-defaults rule. The new agent config is a thin extension of `recurrent_ppo.yaml` plus the logger block.

**Designer will produce these YAMLs only after** (a) the canonical preset ambiguity in §6 is resolved by the user, and (b) the developer has implemented the `g1_prime_logger` schema and reads via `get_mandatory()` (see §7 File Changes).

---

## 4. Results

### 4.1 Primary Metrics (Hypothesis Test)

(To be filled after runs.)

| Run | Seed | max MSE / min MSE (high inj, post-fit) | argmax modality | argmin modality | Pass? (>2) |
|---|---|---|---|---|---|
| 1 | 0 | — | — | — | — |
| 2 | 1 | — | — | — | — |
| 3 | 2 | — | — | — | — |
| **Median** | — | **—** | — | — | **—** |

> **Verdict on H₁** — to be filled. Pass criterion: median seed's ratio > 2.

### 4.2 Secondary Metrics

(To be filled after runs.)

- Per-modality MSE absolute values (high-inj bin) — diagnoses *which* channels carry the information.
- Per-modality MSE absolute values (low-inj bin) — sanity check that the head reconstructs at all.
- Ratio MSE(high inj) / MSE(low inj) per modality — directly tests state-dependence (Bayesian-brain §3.2(a) condition (b)).
- Head loss curve over the 1M-step fit — diagnoses head training pathology (NaN, plateau too high).
- Sample counts per bin — checks the ≥ 1k-transitions-per-bin minimum.

### 4.3 Diagnostic Metrics

To be filled. Specifically: per-modality MSE matrix (modality × injury_bin), random-init-head MSE for context.

### 4.4 Learning Dynamics

To be filled (head-fit loss curve, per-modality fit quality across the 1M-step window).

---

## 5. Analysis

### 5.1 Failure-Mode Catalog (pre-registered, per the experiment-designer profile)

| Failure mode | Pre-registered interpretation |
|---|---|
| **Ratio ≤ 2 across all 3 seeds** (median ≤ 2) | **Refutes H₁ on this preset.** Halt architectural program. Retune the canonical noise preset. *Not* a refutation of the project — the project's H1.5 claim ("the noise landscape determines whether the AI fixed point is non-degenerate") is itself supported by this outcome. |
| **Ratio > 2 in median, but < 2 in 1 seed** | **Pass.** Note the seed-variance in §4.3 as a flag; the rate at which a *trained* encoder lands on a non-degenerate residual structure is itself preset-stable (most seeds) but not guaranteed (one outlier). Architectural program proceeds. |
| **Head fit fails to converge (NaN, plateau equal to random-init MSE)** within the 1M-step head-fit window | **Run is inconclusive — not a refutation.** The encoder is producing features the head cannot decode; this is a *measurement-device* failure, not an environment failure. Retry with a wider head ([256, 256, 256]) or longer fit (5M). If still fails after one retry per seed, the encoder representation is genuinely impenetrable to a 2-layer MLP — escalate to senior-developer for a richer probe. |
| **Both injury bins underpopulated** (< 1k transitions in either bin during measurement) | **Run inconclusive.** Extend `measure_steps` to 1M; if still underpopulated, the LN baseline policy avoids one regime entirely (e.g. dies before reaching `injury > 0.5`, or never recovers below 0.1). This is a behavioural signal worth reporting in §4.3 but does not refute or confirm H₁. |
| **All modalities have ~equal MSE in the high-inj bin (max-min ratio < 1.5)** | **Refutes H₁ strongly.** Same action as ratio ≤ 2 (halt program), but also flags: the *encoder is collapsing channels* — possibly a LayerNorm-induced uniformisation. Note for the senior-developer hand-off (per RL-BDL §1.3, the FiLMNoNorm path may be needed to *measure* the heterogeneity even if the noise preset has it). |
| **Ratio > 2 but driven entirely by one trivial channel** (e.g. Visual with hundreds of dims dominates because dimensions are reported as raw sum, not mean) | **Implementation bug.** Means the §2.3 "use mean per-channel MSE within a modality" rule was not respected. Fix the logger and rerun — not a science result. |
| **High-inj ratio < 2 but low-inj ratio > 2** | **Refutes the *state-dependence* premise (Bayesian-brain §3.2(a)(b)) more than the heterogeneity premise.** This is a different failure than "uniform noise" — it says reliability is heterogeneous but injury-decoupled. Halt; retune the `injury_noise_scale` parameters specifically (not σ_base). |

### 5.2 Cross-Run Comparisons

(Filled post-runs.) Cross-seed comparison is the only one in scope.

### 5.3 Failure Modes & Pathologies

(Filled post-runs.)

---

## 6. Canonical Noise Preset — Identification (HALT for User)

> **STATUS: HALT.** Per the experiment-designer profile and the user's brief, "if multiple candidates exist, halt and surface to me — do not pick silently." Multiple candidates exist; the doc proceeds with a placeholder, and the user must resolve before any config is written.

### 6.1 What "the canonical noise preset" refers to in the synthesis memo

The synthesis memo §1.5 and project_plan.md §Phase 1 both reference "the canonical noise preset" — a single, currently-unspecified set of `perceptual_noise.modalities[*].sigma` and `injury_noise_scale` values that:

- Is heterogeneous across modalities (per-modality σ values differ).
- Is state-dependent on injury (`mode: state_dependent`, nonzero `injury_noise_scale` on the threat-relevant channels).
- Has been argued to drop the LayerNorm baseline's survival by a meaningful, seed-stable margin vs. the noise-free condition.

**The project_plan as written says this preset is the *output* of a Phase 1 sweep, not an input.** No file in `configs/` is currently named "canonical".

### 6.2 Candidates I find in `configs/experiment/`

The following YAMLs have `perceptual_noise.enabled: true`. Sigma profiles differ across them; only one of them is referenced anchor-style in any prior diagnosis doc.

| Path | Notes |
|---|---|
| `configs/experiment/hypervigilance/01-interoNocicept_noise.yaml` | **Strongest candidate.** Its σ profile matches verbatim the v8 NMN diagnosis [§2.2 controlled config](../../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md): injury/nutrition σ=0.0, satiation/intero/extero noc σ=0.1, olfaction σ=0.2, collision/proprio σ=0, visual σ=0.2, location σ=0; injury_noise_scale=1.5 on state-dependent channels. v8's LN baselines (`ptje3he2`, `auuuayj6`) trained on this preset. |
| `configs/experiment/2X2_area.yaml` | A 2×2 grid variant; not on the rung-3 ladder per synthesis §1.3. Not a candidate for the canonical Phase-1 preset. |
| `configs/experiment/labmeeting/basic-03-noise.yaml`, `basic-03-prop75-noise.yaml`, `basic-03-prop75_std3-noise.yaml`, `basic-03-prop75_std4-noise.yaml`, `basic-04-predRange*-noise.yaml` | Lab-meeting/exploratory variants. Different obs schemas (some pre-date the `interoceptive_nociception` channel). Not anchored in any current diagnosis doc. |

### 6.3 Recommendation to the user

**Recommend `configs/experiment/hypervigilance/01-interoNocicept_noise.yaml` as the canonical preset for this diagnostic**, on three grounds:

1. It is the only noise-enabled config that matches the v8 diagnosis's σ profile, which is the most recent baseline characterisation (March 2026).
2. Its modality schema includes the full `interoceptive_nociception` + `extero_nociception` split that the synthesis memo §1.4 G2' fingerprint requires downstream.
3. It is filed under `hypervigilance/` — the topic that the architectural-program rewrite is anchored to.

**But this is the user's call.** The diagnostic itself is preset-agnostic; the choice is which environment we are characterising. **No configs are written until the user confirms.**

---

## 7. File Changes (developer scope, not designer scope)

The diagnostic requires schema additions that are not yet read by `src/utils/config.py` or downstream loaders. Per the experiment-designer profile: **the designer does not modify code or invent schema in configs that the loader doesn't yet consume.** This section is the spec the developer (via senior-developer planning) must implement *before* any config in §3.1 is written.

### 7.1 New mandatory config keys (under `agent` block of the agent YAML)

All loaded with `config.get_mandatory()` — no defaults, no fallbacks. Missing key → `ValueError`.

| Key | Type | Mandatory if `g1_prime_logger.enabled: true` | Purpose |
|---|---|---|---|
| `g1_prime_logger.enabled` | bool | Always | Master switch. When `false`, logger is fully disabled and the rest of the keys are not parsed. |
| `g1_prime_logger.head_hidden_dims` | list[int] | yes | MLP hidden dims of the reconstruction head. RL-BDL §5.2 spec is `[128, 128]`. |
| `g1_prime_logger.head_input` | str | yes | Which feature stream the head reads. Must be one of `{"encoder_recurrent_hidden", "encoder_pre_recurrent", "raw_obs"}`. RL-BDL §5.2 says *the recurrent hidden state* (the agent's `h_task`); spec'd as `encoder_recurrent_hidden`. |
| `g1_prime_logger.head_target` | str | yes | What the head reconstructs. Must be `"obs_t+1_clean"` (= `get_observation(state_{t+1}, params, apply_noise=False)`). |
| `g1_prime_logger.head_fit_steps` | int | yes | Number of env steps over which only the head's parameters update (encoder frozen). 1_000_000 per RL-BDL §5.2 budget. |
| `g1_prime_logger.head_fit_lr` | float | yes | Adam LR for the head; head is not on the policy optimiser. |
| `g1_prime_logger.measure_steps` | int | yes | Held-out steps for MSE measurement (no head update). 200_000 default; logger must enforce ≥ 1k transitions per injury bin. |
| `g1_prime_logger.injury_bin_high_threshold` | float | yes | Float in [0, 1]; the gate on `state.injury_level / max_injury` for the "high injury" bin. |
| `g1_prime_logger.injury_bin_low_threshold` | float | yes | Float in [0, 1]; gate for "low injury" bin. |
| `g1_prime_logger.ratio_threshold` | float | yes | Pass/fail threshold on max-MSE / min-MSE per modality in the high-inj bin. 2.0 per RL-BDL §5.2. |
| `g1_prime_logger.min_transitions_per_bin` | int | yes | Minimum transitions required in each bin to count as a valid measurement; below this, run is "inconclusive." 1000 default. |

Loader location: extend `src/utils/config.py` consumption sites under `train.py` (the agent-config loader). The logger module itself lives in `src/models/g1_prime_logger.py` (new module) — out of designer scope; included here only because the design depends on its existence.

### 7.2 New code modules (developer scope)

| Module | Purpose | Approx. size |
|---|---|---|
| `src/models/g1_prime_logger.py` | Frozen-head MLP, head-fit loop, per-modality MSE accumulator, injury-bin gate, ratio computation, WandB logging keys. | ~150-200 LOC |
| Extension of `train.py` | Load checkpoint of frozen LN baseline; instantiate logger; switch from policy-training mode → head-fit mode → measurement mode based on phase counter. | ~50 LOC |
| `scripts/g1_prime_pass_fail.py` | Post-hoc analysis script — pulls per-modality MSE from local WandB summary files, computes max/min ratio per seed, emits pass/fail vs threshold. **Must read** the modality breakdown from `src/environment/sensor.py::get_observation_breakdown` to map the flat obs vector indices to modality names. | ~80 LOC |

### 7.3 Spec for `scripts/g1_prime_pass_fail.py`

The analysis script developer specification (designer's wording; developer owns implementation):

```
Inputs:
  --runs <wandb_run_id_1> <wandb_run_id_2> <wandb_run_id_3>   3 seed run ids
  --threshold 2.0                                              from config (sanity-checked against ratio_threshold key)

Behaviour:
  1. For each run, open its local WandB summary (wandb/run-*/files/wandb-summary.json or via wandb.Api().run().summary).
  2. Pull per-modality MSE values logged by g1_prime_logger:
     - g1_prime/per_modality_mse_high_inj/<modality_name>      (post-fit, high-inj bin)
     - g1_prime/per_modality_mse_low_inj/<modality_name>       (post-fit, low-inj bin)
     - g1_prime/n_transitions_high_inj                         (sample count)
     - g1_prime/n_transitions_low_inj                          (sample count)
  3. For each run, compute:
     ratio_high = max(mse_high_inj) / min(mse_high_inj)        across modalities
     ratio_low  = max(mse_low_inj)  / min(mse_low_inj)         across modalities
     ratio_high_over_low_per_modality = mse_high_inj[i] / mse_low_inj[i]
  4. Validate:
     - n_transitions_high_inj >= min_transitions_per_bin (else mark inconclusive)
     - n_transitions_low_inj  >= min_transitions_per_bin
  5. Emit:
     - per-run table (one row per seed): seed, ratio_high, ratio_low, argmax modality, argmin modality, status
     - aggregated: median ratio_high across seeds, pass/fail vs threshold
     - per-modality state-dependence: ratio_high_over_low for each modality, listed
  6. Exit code:
     - 0  if median ratio_high > threshold (PASS — architecture program proceeds)
     - 1  if median ratio_high <= threshold (FAIL — preset must be retuned)
     - 2  if any seed inconclusive (sample-count failure or NaN)

Output format: human-readable table to stdout; --json optional for machine-readable.
```

The script does **not** train or evaluate; it parses logged scalars only. WandB API is used in offline mode (read local files, not network).

### 7.4 No designer-side configs until 7.1–7.3 are merged

Per the experiment-designer profile §"Schema-affecting changes are NOT in scope": the configs in §3.1 will not be written until `g1_prime_logger.*` keys are read via `get_mandatory()` in the loader. **Hand-off chain:** this design doc → senior-developer plan (small) → developer implements 7.1–7.3 → designer authors the §3.1 YAMLs → env-config-auditor pre-flight → user authorises launch → training-runner.

---

## 8. Frozen-Head Implementation Choice — (a) vs (b)

The user's brief asked for a choice between two implementations. **The diagnostic adopts (a):** *the head trains from scratch on the existing trained LayerNorm baseline checkpoint (frozen policy)*.

### 8.1 Why (a), in one sentence

Because RL-BDL §5.2 explicitly specifies "the head is a *measurement device*, not a trained component" and "trained on a small held-out window with the encoder frozen" — interpretation (a) is the literal reading; interpretation (b) would conflate the measurement of the AI fixed point with the dynamics of an unconverged policy.

### 8.2 Why not (b), in detail

Interpretation (b) — head trains alongside fresh policy with stop-grad on the policy — would, in 1 hour wallclock (~5M env steps for rPPO), produce:
- A policy that is **not at steady state** (rPPO LN baseline reaches steady state at ~10M+ env steps in v7/v8, well beyond the 1h horizon).
- A head that is fitting to a *moving target* (the encoder is changing under policy gradient).

Per-modality MSE under (b) at 1h would be a function of *both* the noise preset's heterogeneity *and* the policy's incomplete training. The diagnostic's purpose is to characterise the noise preset; (b) confounds that with policy training dynamics. The 1h wallclock RL-BDL §5.2 cites cannot apply to (b).

### 8.3 What "~1 hour wallclock" applies to

Under (a):
- **0 minutes of policy training** — reuses an existing trained LN-baseline checkpoint.
- **~12 minutes** of head-fit (1M env steps × ~5.2M steps/h = ~12 min on a single GPU at the rPPO measured rate; head is a 2-layer MLP — much smaller than the policy update, so the marginal cost on top of env stepping is small).
- **~3 minutes** of measurement (200k steps).
- **Total ≈ 15-30 minutes per seed** (with margin); 3 seeds sequential ≈ 1 hour.

The 1h budget RL-BDL cites is **per noise preset, per checkpoint, head-fit + measurement only**. If a checkpoint must first be trained, that cost is *separate* and is *not* the "1 hour" figure.

### 8.4 The hidden precondition this surfaces

**(a) requires a usable LN baseline checkpoint to exist.** This is the highest-priority blocking dependency in this design. See §6.3: v8 baselines `ptje3he2` (GAE) and `auuuayj6` (MC) are candidates if their checkpoints are still on disk. If they are not, or if the obs schema has changed since v8 (the project added `interoceptive_nociception` somewhere along the way — check whether the v8 checkpoint's encoder input dim still matches the current `01-interoNocicept_noise.yaml` obs vector), then a **fresh LN-baseline policy training** is needed before the diagnostic can run, which inflates wallclock to ~1 day total.

The senior-developer / developer hand-off must confirm checkpoint availability and schema compatibility before declaring the diagnostic "ready to launch."

---

## 9. Hand-offs

### 9.1 Senior-developer (next agent)

Author a small implementation plan covering §7.1–7.3 (the `g1_prime_logger.*` schema, the `g1_prime_logger.py` module, the `g1_prime_pass_fail.py` script, the train.py extension). Not invasive — all additive. Gate on user authorisation of the canonical preset (§6).

### 9.2 Developer

Implements §7.1–7.3 per the senior-developer plan. Confirms checkpoint reuse path or notifies that a fresh LN-baseline training is required.

### 9.3 Env-config-auditor

After the §3.1 configs are written (post-developer), audit them for:
- `perceptual_noise.enabled: true` in the canonical preset.
- `agent.use_layer_norm: true`, `agent.modulation.type: null`.
- All `g1_prime_logger.*` keys present.
- `injury_observable: false` (the canonical regime).
- Modality breakdown in `sensor.py` matches the breakdown the analysis script will read.

### 9.4 Training-runner

Launches the 3 seeded runs after env-config-auditor passes and the user authorises. Fills the actual columns of §3 manifest in place.

### 9.5 Experiment-analyzer / experiment-designer (for results phase)

After runs complete, the analyzer (or this designer) fills §4, §5.1's actual outcomes vs. pre-registered failure modes, §6 conclusions. The pass/fail outcome of `scripts/g1_prime_pass_fail.py` is the single load-bearing line.

---

## 10. Cross-references

- [project_plan.md §Phase 1](../../../project/project_plan.md) — the canonical noise preset's role in the architectural program.
- [synthesis memo §1.5](../../../project/ideas/nature_mi_paper_framing_synthesis.md) — G1 → G1' restatement.
- [RL-BDL memo §5.2](../../../project/directions/architecture_for_pain_computation.md) — diagnostic spec.
- [Bayesian-brain memo §3.2(a)](../../../project/concepts/active_inference_hypervigilance.md) — theoretical statement of channel-rank non-degeneracy.
- [NMN_PERFORMANCE_DIAGNOSIS_v8](../../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md) — anchor v8 LN baselines (`ptje3he2`, `auuuayj6`) candidate-checkpoints.
- [training_analysis template](../../../TEMPLATES/training_analysis.md) — doc structure.

---

## Appendix

### A. Open Questions Surfaced for the User

1. **Canonical preset name (§6).** `01-interoNocicept_noise.yaml` recommended; user confirms or supplies an alternative.
2. **Reusable LN-baseline checkpoint (§8.4).** Are `ptje3he2` / `auuuayj6` (v8) on-disk and schema-compatible? If not, allocate +1 day for fresh LN-baseline training under (a).
3. **3 seeds vs. more.** Synthesis recommends ≥ 3 for marginal effects; this is a go/no-go threshold check, not a marginal-effect study, so 3 is defensible. Confirm.
4. **Reconstruction target — clean obs $o^{\text{true}}_{t+1}$ vs. noisy $o_{t+1}$.** Pre-registered as clean (§2.3 mitigation row). Note the synthesis memo and RL-BDL §5.2 do not specify; this is a designer choice tied to the AI theoretical statement (residual variance is on noise-free signal). User can override.
5. **Failure-loop policy.** If H₁ is refuted, the user re-tunes the noise preset and re-runs this diagnostic. How many retune-iterations are budgeted before the *project itself* halts and the noise-landscape question is escalated to the professor team? This design pre-registers ≤ 3 retune iterations as a soft bound; user confirms.

### B. Changelog

| Date | Change | Author |
|---|---|---|
| 2026-05-07 | Initial design | experiment-designer |
