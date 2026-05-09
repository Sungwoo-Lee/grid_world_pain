---
title: "NMN continual double-return probe — does the modulator resist catastrophic forgetting and reuse subnetworks?"
topic: hypervigilance
status: active
created: 2026-05-09
last_updated: 2026-05-09
phase: 0.5
wandb_tag: "rppo_nmn_cont_dr_{mod,unmod}_s0"
develop_link: docs/develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md
supersedes: null
superseded_by: null
---

# NMN continual double-return probe — does the modulator resist catastrophic forgetting and reuse subnetworks?

> **Status**: PLANNED — configs and manifest locked 2026-05-09. Awaiting env-config-auditor pass and user authorisation before launch.
> **Date**: 2026-05-09
> **Author**: experiment-designer
> **Related**:
> - Lock memo (postdoc synthesis v2): [`nmn_meta_continual_synthesis_v2.md`](../../../project/ideas/nmn_meta_continual_synthesis_v2.md)
> - Direction memo (professor-neuromodulation): [`nmn_continual_lifelong_probe.md`](../../../project/directions/nmn_continual_lifelong_probe.md)
> - Triage anchor: [`20260509_1517_nmn_meta_continual_pivot.md`](../../../project/triage/20260509_1517_nmn_meta_continual_pivot.md)
> - Sister design (meta side, same overnight launch window): [`NMN_META_2x3_MIXTURE_PROBE.md`](NMN_META_2x3_MIXTURE_PROBE.md)
> - Anchor diagnosis: [`NMN_PERFORMANCE_DIAGNOSIS_v8.md`](../../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md)
> - Mechanism finding behind the new agent config (`temp_clip = [0.5, 5.0]`): memory insight `20260509_1410_nmn_temp_head_natural_target_3_to_5`

> **One-line scope.** Two cells (modulated vs. unmodulated, single seed each) on a 5-stage abrupt active->passive->active->passive->active predator schedule. Tests whether the FiLM modulator attenuates the post-switch survival dip and resists catastrophic forgetting on the return stages. Sister experiment to the meta 2x3 probe — together they answer "is the architecture broken or is the v8 question hard?".

---

## 1. Research Question

The project trains an agent to survive in a small grid world (find food, avoid predators, track injury and hunger). Two architectures are being compared this week: a **plain agent** (a standard recurrent network — one brain) and a **modulated agent** (the same brain plus an extra small "modulator" head sitting on top, which can dynamically reweight how the main brain reads its senses depending on context).

This week's earlier work — the noise-heterogeneity sweep and the temperature-ceiling rerun (see [study summary](../../summaries/20260509_1421_nmn_comparison_study.md)) — closed both noise-related explanations of why the modulated agent has not been beating the plain one. The user is now pivoting to a different test: instead of asking the modulator to handle noisy senses, ask it to handle **a world that changes between training stages**. Does the modulator help the agent retain what it learned in stage 1 after stage 2 overwrites the policy, and re-use what it built when the world cycles back? This is the "continual learning" regime where the academic literature consistently reports gains for modulator-style architectures (notably Lee 2024 on a Doya-DaYu agent, and Rodriguez-Garcia 2026 on entropy-driven gain bursts).

The experiment uses the project's already-shipped continual-schedule machinery to train one agent across **five stages** that alternate between two predator behaviours: an **active predator** (full hunt mode, identical to the canonical `01-interoNocicept_sameProp.yaml` baseline) and a **passive predator** (hunt mode is structurally unreachable; the predator stays in a random patrol — kinematically rabbit-like, identical to `02-sameProp_R2_passivePredator.yaml`). The schedule is **active → passive → active → passive → active** with abrupt boundaries (no gradual interpolation in this round). The double return to "active" is the load-bearing piece: it lets us measure whether forgetting is **monotone** (each return retains less than the previous one, the signature of plain interference) or **non-monotone** (the second return reaches or exceeds the first, the signature of a re-usable context-specific subnetwork à la Tsuda 2021 hypertubes — what the modulator-architecture literature claims is the architectural contribution).

**Important caveat that must be in the user's mind before reading any positive result.** This experiment runs **one seed per cell**. The earlier temperature-ceiling rerun discovered ±4.4-step seed-to-seed noise in the project's survival numbers; under that noise, **a single unlucky seed could mislead the verdict in either direction**. The single-seed choice is deliberate (the user is spending compute across 10 cells in one overnight window and prioritised breadth over within-cell statistical power), but the read-out from this experiment is necessarily **directional, not statistically definitive**. If the headline contrast (modulated vs. unmodulated forgetting) sits within ±5 survival steps, the experiment is inconclusive on its own and a follow-up multi-seed replication is required.

**Plain-language outcomes:** "Positive" = the modulated agent forgets less on the return stages **and** the modulator's hidden state shows a measurable transient at the boundaries (mechanism check, not just survival number). "Refutation" = same forgetting as plain agent and no boundary transient — the architecture is failing the floor test.

### 1.1 Formal hypotheses

> **H₀** (null — *the modulator does not help on continual*): On the 5-stage active-passive-active-passive-active schedule, modulated and unmodulated agents show statistically indistinguishable forgetting curves. Specifically, the catastrophic-forgetting contrast $\Delta\Delta_{\text{forget}} \triangleq \Delta\text{forget}^{\text{mod}} - \Delta\text{forget}^{\text{unmod}}$ falls within $\pm 5$ survival-steps (one seed-noise floor), AND the modulator's hidden-state Mahalanobis distance from stage-1 mean does not exceed 2σ in the first 50 episodes of stage 2.

> **H₁a** (Rodriguez-Garcia stability-gap attenuation): On stage 2 (the first switch), the modulated agent's post-switch survival dip $d^{\text{mod}}_2$ is at most half the unmodulated agent's: $d^{\text{mod}}_2 \leq \frac{1}{2} d^{\text{unmod}}_2$. The modulator-state transient at the boundary is detectable (Mahalanobis > 2σ within 50 ep), and $\sigma(z_{\text{uni}})$ rises toward 1, $\tau_\pi$ moves away from 1, both transiently. Mechanism: gain spike flattens the loss-landscape Hessian during the switch, eigenvalues scale $\lambda \to \lambda/g^2$ (Rodriguez-Garcia 2026 §I.5).

> **H₁b** (Lee-style catastrophic-forgetting resistance): At end of stage 3 (first return), the modulated agent's recovered survival on the active task is meaningfully closer to its stage-1 asymptote than the unmodulated agent's. Specifically, $|\Delta\text{forget}^{\text{mod}}| \leq \frac{1}{2} |\Delta\text{forget}^{\text{unmod}}|$, where $\Delta\text{forget} \triangleq \overline{\text{survival}}_{\text{stage 3 final 200ep}} - \overline{\text{survival}}_{\text{stage 1 final 200ep}}$.

> **H₁c** (Tsuda hypertube reusable subnetwork): The second return (stage 5) exceeds or equals the first return (stage 3) in retained survival on the active task: $\overline{\text{survival}}_{\text{stage 5}} \geq \overline{\text{survival}}_{\text{stage 3}}$, on the modulated agent **only** (the unmodulated agent is predicted to show monotone decay across both returns). H₁c is the strongest possible architecture-vitality signal in this experiment because it directly contrasts pure interference (the unmodulated null) with reusable subnetwork structure (the modulator-positive claim).

### 1.2 What confirms vs. refutes — pre-specified

| Outcome | Predicate | Verdict |
|---|---|---|
| **H₁a confirmed** (stability-gap) | At stage 2, $d^{\text{mod}}_2 \leq \frac{1}{2} d^{\text{unmod}}_2$ AND mod_h Mahalanobis > 2σ within first 50 ep AND $\sigma(z_{\text{uni}})$ shows transient toward 1 | The modulator engages at the switch boundary; the Hessian-flattening mechanism (Rodriguez-Garcia 2026 §I.5) is *operational*. Fold into the case for keeping the FiLM family in the architecture roadmap. |
| **H₁b confirmed** (catastrophic-forgetting) | $|\Delta\text{forget}^{\text{mod}}| \leq \frac{1}{2} |\Delta\text{forget}^{\text{unmod}}|$ at end of stage 3 | The modulator builds context-dependent state that survives the stage 2 interference. The Lee-2024 mechanism (§G.5 in NEUROMODULATION_ALGORITHM.md) is partially-instantiated even without the precision head. **Do not block on Phase 0 for continual evidence.** |
| **H₁c confirmed** (Tsuda hypertube) | Stage 5 modulated survival >= stage 3 modulated survival AND unmodulated stage 5 < stage 3 (monotone decay on the unmod arm) | The modulator's hidden state encodes a re-usable context attractor — the architecturally-decisive signal in this probe (per professor-neuromodulation §3 magnitude target). Headline finding for the architecture-vitality narrative. |
| **H₀ confirmed (full null)** | $|\Delta\Delta_{\text{forget}}| \leq 5$ steps AND no detectable mod_h transient at any boundary AND no `\tau_\pi` excursion | Floor test failed: even the cheapest stability-gap mechanism (gain heads moving at the boundary) is not engaged. Combined with a sister-experiment null on the meta probe, this is the project's signal that pre-Phase-0 is not a viable starting architecture. **Hand to senior-developer for Phase 0 plan (T/P split + precision head per project_plan §3.2 / §3.4).** |
| **Partial null (H₀ on survival, but mod_h transient detected)** | $|\Delta\Delta_{\text{forget}}| \leq 5$ steps BUT mod_h Mahalanobis > 2σ at switches | The modulator *moves* but the movement does not cash out into survival. Mechanism is engaged, downstream pipeline (encoder gates, policy temperature) is not productively using it. Same hand-off as full null but with the modulator-uses-the-signal claim preserved — a softer but useful finding. |
| **Refutation by training instability** | Either cell NaNs, value/grad explodes, Term_Starvation > 90%, or the schedule transition triggers a JIT failure | Single cell marked inconclusive; if both cells fail, the schedule itself or the env-rebuild path on a 5-stage continual run is at fault. Halt and report. |

### 1.3 Why a single seed (and what that costs)

- Five lab nodes, ten GPU slots total, ten cells planned for the overnight launch. Two cells assigned to this experiment (modulated, unmodulated). One seed each.
- The earlier temp-clip rerun discovered ±4.4-step seed-to-seed noise on this env at this scale. Under that noise, **the only confidently-readable verdicts from a single-seed contrast are large-effect ones** — H₁a's "half the dip" and H₁c's "stage 5 >= stage 3" directly produce signed comparisons that survive ±4.4-step noise; H₁b's "half the forgetting" requires the unmodulated $\Delta\text{forget}$ to be ≥ ~10 steps for the contrast to be readable.
- The trade-off the user accepted: breadth (cover both meta and continual probes plus 6 specialists in one overnight) over depth (multi-seed within one probe). If the headline contrast is borderline (within ±5 steps), this experiment surfaces a follow-up request for a 3-seed continual replication.

---

## 2. Experimental Design

### 2.1 Independent Variables

| Variable | Values | Rationale |
|---|---|---|
| `agent.modulation.type` | {FiLM, null} | The treatment-vs-control contrast: does the modulator help. |
| Schedule stage | {1: active, 2: passive, 3: active, 4: passive, 5: active} | 5-stage abrupt double-return per professor-neuromodulation §4. The two returns to active enable the monotone-vs-non-monotone forgetting test (H₁c). |

### 2.2 Controlled Variables

Pinned across both cells:

```yaml
# Schedule (locked to one schedule YAML across both cells)
configs/continual/nmn_double_return.yaml          # episode_boundaries [1500, 3000, 3700, 4400, 5100]
configs/continual/nmn_double_return_stages/       # 5 stage YAMLs (alphabetic order, train.py:156)
# Stage env params byte-identical to:
#   01,03,05  -> 01-interoNocicept_sameProp.yaml (active, sameProp olfactory matched)
#   02,04     -> 02-sameProp_R2_passivePredator.yaml (passive, sameProp olfactory matched)

# Agent — shared hyperparameters (v8 §2.2 anchor) across both cells
agent:
  algorithm: RecurrentPPO
  return_mode: MC
  use_layer_norm: true
  rnn_type: GRU
  activation: relu
  encoding_mode: hierarchical
  hidden_size: 128
  sequence_length: 128
  K_epochs: 4
  lr_actor: 0.0005, lr_critic: 0.0001
  entropy_coef: 0.01, eps_clip: 0.1

# Training horizon
total_episodes: 5100   # = sum of stage budgets (5 stages of 1500/1500/700/700/700)
num_envs: 128 (default)
seed: 0 (per cell)

# Noise: NOISE OFF (perceptual_noise.enabled: false) — this probe tests
# context-conditioning across env *structure* changes, not noise heterogeneity.
# The earlier study's heterogeneity sweep already separated those questions.
```

### 2.3 Confounds & Limitations

| Confound | Affected Runs | Severity | Mitigation |
|---|---|---|---|
| Single seed per cell (±4.4-step seed noise per memory insight `20260509_1410`); 1 unlucky seed could mislead the verdict | both | **High** | Pre-registered effect-size thresholds (50% reduction, not parity) for H₁a/H₁b — large effects survive seed noise. If contrast within ±5 steps, mark inconclusive and request 3-seed follow-up. |
| Stage transition triggers a `ParallelEnv` rebuild + JIT recompile (train.py:1098-1100); recompile artefact could spuriously create a "stability gap" | both | Med | Recompile happens identically for both cells; the *contrast* (mod vs. unmod) cancels out the artefact. Absolute dip depths reported with the recompile-time annotated. |
| Recurrent state and Dreamer buffers reset on transition (train.py:1141-1151); modulator hidden state ALSO reset | both | Med | Symmetric reset; the post-reset adaptation curve is itself the H₁a measurement target. |
| `perceptual_noise.enabled: false` — clean obs throughout. The modulator's primary literature-claimed function (sensory reweighting) is *not* under test; only the context-detection function is. | both | **Low (by design)** | Sister experiment ([NMN_META_2x3_MIXTURE_PROBE](NMN_META_2x3_MIXTURE_PROBE.md)) tests context-detection in a different regime; together they triangulate. The earlier noise-heterogeneity sweep already covered the sensory-reweighting question. |
| Episode budget per stage (1500 / 1500 / 700 / 700 / 700) was chosen by professor-neuromod §4 estimate, not measured. If stage 1 has not converged at 1500 ep, $\Delta\text{forget}$ is contaminated by stage-1-undertraining artefact. | both | Med | The earlier project's RPPO survival typically asymptotes by ~3M training steps (~ 2-3k episodes at 500 max_steps), so 1500 ep may be early-asymptotic. Analyzer must report stage-1-final-200ep mean as a fraction of the global maximum to flag any convergence issue. |
| `mod_h` Mahalanobis distance requires the modulator's per-step hidden state to be logged. The training run logs `mod_h` as a metric vector if architecture is FiLM (per existing diagnostic plumbing); if not, the H₁a mechanism check cannot be performed. | mod cell | Low | Pre-flight: `experiment-analyzer` confirms `mod_h` logging is enabled before launch. The unmod cell trivially does not need this. |

## 3. Launch Manifest

System-of-record. `experiment-designer` filled the planned columns (Tag, Cell, Seed, wandb-group/job-type); `training-runner` will fill the actuals at launch. Single seed per cell per the §1.3 trade-off.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | running | continual_modulated | rppo_nmn_cont_dr_mod_s0 | nmn_continual_double_return | prod | 0 | 101 | cuda:0 | 2026-05-09T18:27:03 | 9wckyb5k | logs/20260509_182703.log |
| 2 | planned | continual_unmodulated | rppo_nmn_cont_dr_unmod_s0 | nmn_continual_double_return | prod | 0 | 101 | cuda:1 | — | — | — |

### 3.1 Configs to Produce

| Run | Config (env / schedule) | Config (agent) |
|-----|--------------------------|----------------|
| 1 | `--configs-dir configs/continual/nmn_double_return_stages/` `--continual-schedule configs/continual/nmn_double_return.yaml` | `configs/models/recurrent_ppo_nmn_film_g1_tempceil5.yaml` |
| 2 | (same) | `configs/models/recurrent_ppo_nmn_het_unmod.yaml` (existing) |

**New files this experiment produces:**

- `configs/continual/nmn_double_return.yaml` — schedule (5 boundaries / 5 ckpt freqs).
- `configs/continual/nmn_double_return_stages/01_active_predator.yaml` (header + body byte-identical to `01-interoNocicept_sameProp.yaml`).
- `configs/continual/nmn_double_return_stages/02_passive_predator.yaml` (header + body byte-identical to `02-sameProp_R2_passivePredator.yaml`).
- `configs/continual/nmn_double_return_stages/03_active_predator.yaml` (= `01_active_predator.yaml` body, header notes stage 3).
- `configs/continual/nmn_double_return_stages/04_passive_predator.yaml` (= `02_passive_predator.yaml` body, header notes stage 4).
- `configs/continual/nmn_double_return_stages/05_active_predator.yaml` (= `01_active_predator.yaml` body, header notes stage 5).
- `configs/models/recurrent_ppo_nmn_film_g1_tempceil5.yaml` — new canonical FiLM g1 modulated agent config with `temp_clip: [0.5, 5.0]`.

Per CLAUDE.md "Git Safety": NAS does not support symlinks, so the 5 stage YAMLs are literal copies. If the canonical active/passive baselines change, env-config-auditor must update the stage YAMLs in lock-step.

The unmodulated agent config (`recurrent_ppo_nmn_het_unmod.yaml`) is REUSED unchanged from the prior heterogeneity sweep.

---

## 4. Analysis Plan (pre-registered)

### 4.1 Primary statistic

Per H₁a/H₁b/H₁c, three primary statistics:

1. **Stability-gap dip depth** $d_k$ for $k \in \{2, 3, 4, 5\}$ (each stage transition):
   $$ d_k = \overline{\text{survival}}_{\text{stage } k-1, \text{ final 200 ep}} - \min_{t \in [t_{k}, t_{k}+200\text{ ep}]} \overline{\text{survival}}_t. $$
   Report $d_k^{\text{mod}}$ vs. $d_k^{\text{unmod}}$ for each $k$.

2. **Forgetting** $\Delta\text{forget}^{\text{return}}_r$ for return $r \in \{3, 5\}$:
   $$ \Delta\text{forget}^{\text{return}}_r = \overline{\text{survival}}_{\text{stage } r, \text{ final 200 ep}} - \overline{\text{survival}}_{\text{stage 1, final 200 ep}}. $$
   Always $\leq 0$. Smaller absolute value = less forgetting.

3. **Hypertube reuse**: the difference $\overline{\text{survival}}_{\text{stage 5}} - \overline{\text{survival}}_{\text{stage 3}}$. Positive on the modulated arm = H₁c confirmed (per professor-neuromodulation §4(b)). Sign is the load-bearing read-out.

### 4.2 Effect-size thresholds (locked)

Per §1.1 / §1.2:

- H₁a positive: $d_2^{\text{mod}} \leq \frac{1}{2} d_2^{\text{unmod}}$.
- H₁b positive: $|\Delta\text{forget}^{\text{return}}_3|^{\text{mod}} \leq \frac{1}{2} |\Delta\text{forget}^{\text{return}}_3|^{\text{unmod}}$.
- H₁c positive: stage-5 retention >= stage-3 retention on the mod arm.
- H₀ confirmed: all three contrasts within ±5 survival steps (single seed-noise floor).

### 4.3 Temporal evolution (mandatory per project rules)

Plot survival vs. episode for both cells across all 5 stages, with stage boundaries marked. Plot `mod_h` Mahalanobis distance from stage-1 mean and $\sigma(z_{\text{uni}})$ / $\tau_\pi$ traces (mod cell only) on the same x-axis. Boundary-locked transients in the modulator-state metrics within the first 50 episodes of each new stage are the H₁a mechanism arm.

### 4.4 Cross-correlation

`mod_h` Mahalanobis spike (within first 50 ep of stage 2/3/4/5) cross-correlated with $d_k^{\text{mod}}$ — does the magnitude of the modulator's boundary-detection transient predict the size of the survival dip attenuation? Pre-specified: report the correlation; do not interpret unless |r| > 0.5 (single-seed; noisy).

---

## 5. Failure-Mode Catalog (pre-decided)

| Failure mode | Reading |
|---|---|
| Either cell NaNs | Run fails; cell marked inconclusive. If both fail, the 5-stage-rebuild path itself is at fault — halt and route to developer for diagnosis. |
| Stage 1 fails to reach asymptote (final-200ep survival < 50% of max) | Stage budget too short; $\Delta\text{forget}$ unreadable. Mark inconclusive; rerun with longer stage 1 if user authorises. |
| Stage 2 / 4 (passive) survival is much higher than stage 1/3/5 (active) — this is expected, not a failure; just makes the "stability gap" on the *easier* stage smaller in absolute terms. The contrast is normalised per stage. | Treat the contrast (mod vs. unmod) as the read-out, not the absolute dip depth. |
| `mod_h` not logged on the modulated cell | H₁a mechanism check unavailable; survival arm of H₁a still readable; H₁b/H₁c still readable. Mark mechanism arm inconclusive; do not invalidate the survival arms. |
| Headline contrast within ±5 survival steps on all three primary statistics (H₀ confirmed within seed-noise floor) | Real null; 1 unlucky seed cannot rescue this. Hand to senior-developer for Phase 0 plan. |
| Headline contrast within ±5 steps on H₁a/H₁b but |r| > 0.5 on the mod_h-vs-dip cross-correlation | "Modulator moves but doesn't matter" — soft null with mechanism preserved. Useful for shaping Phase 0 (the modulator is engaged but downstream-redundant; precision head + T/P split needed to make the engagement productive). |
| Single-seed contrast says "modulated is WORSE" by > 5 steps on either H₁a or H₁b | Either the modulator is genuinely harmful in continual (a publishable negative finding) OR the seed is unlucky in the worst direction. **Cannot disambiguate from this experiment alone**; pre-register the request for a 3-seed continual replication if this outcome lands. |

## 6. Conclusions

*(Filled after analysis by `experiment-analyzer` per the v2 synthesis hand-off chain.)*

### 6.1 Summary
*(blank)*

### 6.2 Limitations & Open Questions
*(blank)*

### 6.3 Recommended Next Experiments
*(blank)*

---

## 7. Hand-offs (named-agent routing)

1. **`env-config-auditor`** (BEFORE launch). Audit:
   - the new schedule YAML (`configs/continual/nmn_double_return.yaml`) for schema correctness (`continual.episode_boundaries` strictly increasing, `checkpoint_frequencies` parallel and > 0).
   - all 5 stage YAMLs in `configs/continual/nmn_double_return_stages/` — confirm `01/03/05` body byte-matches `01-interoNocicept_sameProp.yaml`, `02/04` body byte-matches `02-sameProp_R2_passivePredator.yaml`, except for the prepended header comments.
   - the new agent config `configs/models/recurrent_ppo_nmn_film_g1_tempceil5.yaml` for parity with `recurrent_ppo_nmn_het_film_g1.yaml` modulo the `temp_clip` change.
   - obs↔noise invariants (no env-spec changes that would shift modality dimensions).
2. **User authorisation** — review the auditor pass + this design doc; greenlight the launch.
3. **`training-runner`** (after authorisation). Launch the two cells per the manifest:
   - Cell 1 (modulated) on **node 101 GPU 0** with `--tag rppo_nmn_cont_dr_mod_s0 --wandb-name rppo_nmn_cont_dr_mod_s0 --wandb-group nmn_continual_double_return --wandb-job-type prod --seed 0 --configs-dir configs/continual/nmn_double_return_stages/ --continual-schedule configs/continual/nmn_double_return.yaml --agent_config configs/models/recurrent_ppo_nmn_film_g1_tempceil5.yaml`.
   - Cell 2 (unmodulated) on **node 101 GPU 1** with `--tag rppo_nmn_cont_dr_unmod_s0 --wandb-name rppo_nmn_cont_dr_unmod_s0 --wandb-group nmn_continual_double_return --wandb-job-type prod --seed 0 --configs-dir configs/continual/nmn_double_return_stages/ --continual-schedule configs/continual/nmn_double_return.yaml --agent_config configs/models/recurrent_ppo_nmn_het_unmod.yaml`.
   - Per project memory rule `feedback_runner_post_launch_pgrep`: verify exactly one PID per tag after launch.
4. **`experiment-analyzer`** (after training completes). Fill §4 / §5 / §6 of this doc per the analysis plan above. Cross-reference with the sister meta-probe doc (NMN_META_2x3_MIXTURE_PROBE) for the joint architecture-vitality verdict.
5. **`experiment-designer`** (return). Fill §6 Conclusions per the locked H₀/H₁a/H₁b/H₁c predicates once the analyzer has reported.
6. **`senior-developer`** (conditional, only if H₀ confirmed full null on both this experiment AND the meta sister): plan Phase 0 architecture upgrades (T/P split + precision head per project_plan §3.2 / §3.4).

## 8. Metrics Requested (none new)

The experiment can be evaluated cleanly with metrics already logged: per-episode survival (always logged), `mod_h` per-step (logged for FiLM architectures via existing diagnostic plumbing), `sigma(z_uni)` / `tau_pi` / `gamma_multi` (logged), stage transition markers (logged at train.py:1156-1160). No `feature-workflow` developer touch needed for this experiment.
