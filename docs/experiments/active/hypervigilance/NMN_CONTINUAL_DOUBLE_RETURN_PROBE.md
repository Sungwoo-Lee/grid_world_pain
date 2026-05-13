---
title: "NMN continual double-return probe — does the modulator resist catastrophic forgetting and reuse subnetworks?"
topic: hypervigilance
status: active
created: 2026-05-09
last_updated: 2026-05-13
phase: 0.5
wandb_tag: "rppo_nmn_cont_dr_{mod,unmod}_s0_r2"
develop_link: docs/develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md
supersedes: null
superseded_by: null
---

# NMN continual double-return probe — does the modulator resist catastrophic forgetting and reuse subnetworks?

> **Status**: ANALYZED — both Round 2 cells finished 2026-05-12; Results / Analysis / Conclusions filled 2026-05-13 by `experiment-analyzer`. **Headline: modulated agent wins by ~107-132 survival steps (~25× seed-noise floor) on the return-to-active stages; H₁b and H₁c confirmed.**
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

### 3.1 Round 1 manifest — BUGGED (data discarded; preserved for audit)

> **Why these rows are tagged `round1_bugged`.** The Round 1 schedule
> (`configs/continual/nmn_double_return.yaml` as written 2026-05-09) had a
> ~1000× episode-budget error: cumulative boundary 5100 episodes instead of
> the intended ~5.1M. Both cells finished in ~10 minutes on 2026-05-09
> (PIDs exited cleanly that evening), producing no useful continual-learning
> data. **Rows preserved unchanged so future readers can see the
> 10-minute-cell artefact and trace it to the original schedule file's git
> history.** See §3.2 below for the re-launch (Round 2) with the corrected
> 5.1M-episode budget.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | round1_bugged | continual_modulated | rppo_nmn_cont_dr_mod_s0 | nmn_continual_double_return | prod | 0 | 101 | cuda:0 | 2026-05-09T18:27:03 | 9wckyb5k | logs/20260509_182703.log |
| 2 | round1_bugged | continual_unmodulated | rppo_nmn_cont_dr_unmod_s0 | nmn_continual_double_return | prod | 0 | 101 | cuda:1 | 2026-05-09T18:29:57 | jknwa5xu | logs/20260509_182957.log |

### 3.2 Round 2 — Episode-budget fix re-launch

**Plain-language summary.** The Round 1 schedule had a 1000× error in its
episode-count boundaries: it asked the agent to train for 5100 episodes
total across 5 stages (a literal reading of the professor-neuromodulation
direction memo's "1500 ep / 700 ep" language), but the project's other
training cells run at ~10M episodes — so both Round 1 cells finished in
~10 minutes on 2026-05-09 and produced no continual-learning data. This
section is the re-launch with the corrected budget: per-stage targets
scaled ×1000 to **1.5M / 1.5M / 700K / 700K / 700K episodes** (total
5.1M episodes per cell, comparable in scale to a single 10M-episode
production cell). Same two-cell modulated-vs-unmodulated contrast as
Round 1; same node, same GPUs (n101 free as of 2026-05-10 after the
Round 1 cells exited). The hypothesis tests in §1 and the analysis plan
in §4 below apply unchanged to Round 2 — the bug was purely a budget
error, not a design error.

**Schedule changes (Round 2 vs. Round 1).**

| Field | Round 1 (bugged) | Round 2 (corrected) |
|---|---|---|
| `episode_boundaries` | `[1500, 3000, 3700, 4400, 5100]` | `[1500000, 3000000, 3700000, 4400000, 5100000]` |
| `checkpoint_frequencies` | `[500, 500, 350, 350, 350]` | `[100000, 100000, 50000, 50000, 50000]` |
| Total episodes per cell | 5100 | 5,100,000 |
| Observed/expected wallclock per cell | ~10 min (observed) | ~30-50 h (expected, current cluster load ~150K ep/h on slow nodes) |

**Stage YAMLs unchanged.** The 5 files in `configs/continual/nmn_double_return_stages/` are byte-identical to Round 1; only the schedule's boundaries + checkpoint frequencies were rewritten.

**Round 2 manifest.**

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| R2.1 | running (re-launched n106; prev n101:0 stub 4lcp4vuf killed at ~6 min — node-reallocation, no useful data) | continual_modulated_r2 | rppo_nmn_cont_dr_mod_s0_r2 | nmn_continual_double_return | prod | 0 | 106 | cuda:0 | 2026-05-11T17:58:24 | 8eorbxhq | logs/20260511_175824.log |
| R2.2 | running | continual_unmodulated_r2 | rppo_nmn_cont_dr_unmod_s0_r2 | nmn_continual_double_return | prod | 0 | 106 | cuda:1 | 2026-05-11T18:01:32 | lrzvg8k6 | logs/20260511_180132.log |

The `wandb-group` is unchanged (`nmn_continual_double_return`) — Round 1 and Round 2 sit in the same group; the `_r2` suffix on the run tags is what disambiguates Round 2 data from Round 1 data in any downstream filter/grep. Tags are unique across the manifest.

**Configs to Produce — Round 2 (same as Round 1 except the schedule file is updated in place).**

| Run | Config (env / schedule) | Config (agent) |
|-----|--------------------------|----------------|
| R2.1 | `--configs-dir configs/continual/nmn_double_return_stages/` `--continual-schedule configs/continual/nmn_double_return.yaml` (rewritten 2026-05-11) | `configs/models/recurrent_ppo_nmn_film_g1_tempceil5.yaml` |
| R2.2 | (same) | `configs/models/recurrent_ppo_nmn_het_unmod.yaml` |

### 3.3 Analyzer guidance — Round 1 vs. Round 2 data

When the `experiment-analyzer` opens this experiment, it must apply the following data-eligibility rule:

- **Round 1 runs (`rppo_nmn_cont_dr_mod_s0`, `rppo_nmn_cont_dr_unmod_s0`, both wandb-group `nmn_continual_double_return`, launched 2026-05-09): EXCLUDE from analysis.** These cells ran for ~10 minutes on a 5100-episode budget (1000× under-scale); their per-stage segments contain at most ~hundreds of episodes each, which is below the asymptotic-convergence floor required by every threshold in §4.2. Treat the Round 1 WandB runs as historical artefacts; do not fold their numbers into any of the H₀/H₁a/H₁b/H₁c verdicts.
- **Round 2 runs (`rppo_nmn_cont_dr_mod_s0_r2`, `rppo_nmn_cont_dr_unmod_s0_r2`, same wandb-group, launched 2026-05-11+): the real probe.** All §4 analyses, all §1.2 / §5 verdicts, and all §1.1 hypothesis tests are evaluated against Round 2 data only. The pre-registered effect-size thresholds in §4.2 apply unchanged to Round 2.
- Single-seed caveat in §1.3 still applies to Round 2 (one seed per cell). Within-cell statistical power is unchanged by the budget fix.

### 3.4 Configs to Produce (Round 1 — historical)

| Run | Config (env / schedule) | Config (agent) |
|-----|--------------------------|----------------|
| 1 | `--configs-dir configs/continual/nmn_double_return_stages/` `--continual-schedule configs/continual/nmn_double_return.yaml` (Round 1 version, 5100-ep budget — see git history of the schedule file for the pre-2026-05-11 form) | `configs/models/recurrent_ppo_nmn_film_g1_tempceil5.yaml` |
| 2 | (same) | `configs/models/recurrent_ppo_nmn_het_unmod.yaml` |

### 3.5 File inventory (this experiment's writes — Round 1 + Round 2)

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

---

## 5.5 Results (added 2026-05-13 by `experiment-analyzer`)

> **Headline in one paragraph.** Both Round 2 cells finished cleanly on 2026-05-12 — the modulated and unmodulated agents each trained for ~5.1 million episodes across the five-stage schedule. The plain English answer is **the modulator wins, clearly, on this schedule**. After the world switches twice from active predator to passive and back, the unmodulated baseline survives roughly 110-130 steps per episode on the two return-to-active stages, while the modulated agent survives 237-243 steps — about **2.0× the unmodulated agent's survival** on the return stages, with single-seed effect sizes (107 and 132 survival steps) that are roughly **25-30× the 4-5-step seed-noise floor** flagged in §1.3. The unmodulated baseline shows the classic monotone-decay pattern of catastrophic forgetting (return-2 worse than return-1: 110.8 vs. 130.6); the modulated agent shows the opposite — return-2 is *slightly higher* than return-1 (243.3 vs. 237.5). The pre-registered effect-size thresholds for **H₁b (catastrophic-forgetting resistance, ≤ half the unmodulated forgetting)** and **H₁c (Tsuda hypertube reusable subnetwork, stage 5 ≥ stage 3)** are both passed comfortably. The originally-formalised **H₁a** ("half the post-switch dip on stage 2") is unevaluable as written because stage 2 is the *passive* (easier) stage where survival jumps *upward*, not downward — see §5.5.4. Detailed numbers, modulator-state engagement signal, and training-health below.

### 5.5.1 Run identity confirmation

- **R2.1 modulated** (WandB run [`8eorbxhq`](https://wandb.ai/sungwoolee/grid_world_pain/runs/8eorbxhq), display name `rppo_nmn_cont_dr_mod_s0_r2`): agent config `configs/models/recurrent_ppo_nmn_film_g1_tempceil5.yaml` (FiLM modulator, `temp_clip=[0.5, 5.0]`). Confirmed from the local wandb config; the user-supplied launch-table line that paired this run with `het_unmod.yaml` was a documentation typo — the actual run uses the modulated config. 5,097,472 episodes; 13.3 h wallclock; `Episode/Steps` final-iter = 246.0.
- **R2.2 unmodulated** (WandB run [`lrzvg8k6`](https://wandb.ai/sungwoolee/grid_world_pain/runs/lrzvg8k6), display name `rppo_nmn_cont_dr_unmod_s0_r2`): agent config `configs/models/recurrent_ppo_nmn_het_unmod.yaml` (no modulation). 5,093,982 episodes; 9.5 h wallclock; `Episode/Steps` final-iter = 87.9.
- Both: seed 0, n106, schedule `configs/continual/nmn_double_return.yaml` boundaries `[1.5M, 3.0M, 3.7M, 4.4M, 5.1M]`.

### 5.5.2 Per-stage survival statistics

Mean of `Episode/Steps` over the final ~10% of each stage's episodes (the "tail mean" — the project's pre-registered §4.1 statistic, with the tail window approximated from downsampled history rather than the literal final-200 episodes because the WandB downsampler returns one row per ~1,250 episodes; each tail aggregates 16-20 such rows = ~20-25K episodes).

| Stage | R2.1 modulated (tail mean ± std) | R2.2 unmodulated (tail mean ± std) | Δ (mod − unmod) |
|---|---|---|---|
| 1 active (initial) | **289.5 ± 4.2** | **278.3 ± 3.5** | +11.2 |
| 2 passive (1st switch) | 485.9 ± 17.3 | 487.9 ± 4.2 | −2.0 |
| 3 active (**1st return**) | **237.5 ± 6.0** | **130.6 ± 7.4** | **+106.9** |
| 4 passive (2nd switch) | 495.1 ± 1.8 | 492.3 ± 1.3 | +2.8 |
| 5 active (**2nd return**) | **243.3 ± 7.0** | **110.8 ± 13.8** | **+132.5** |

Bold rows are the load-bearing comparisons: the return-to-active stages are where forgetting and reusable-subnetwork claims live. Passive stages are both at the 500-step max-steps ceiling — both architectures essentially "win" passive trivially.

### 5.5.3 Pre-registered statistics (§4.1) and verdicts

| Quantity | Definition | R2.1 mod | R2.2 unmod | Threshold | Verdict |
|---|---|---|---|---|---|
| **Δforget_3** (1st return) | `survival(stage 3 tail) − survival(stage 1 tail)` | **−51.9** | **−147.7** | mod ≤ ½ unmod ⇒ ≤ 73.8 | **H₁b confirmed**: 51.9 < 73.8 ✓ (mod forgets 65% less than unmod; gap is 95.8 survival steps) |
| **Δforget_5** (2nd return) | `survival(stage 5 tail) − survival(stage 1 tail)` | **−46.2** | **−167.5** | (same) ≤ 83.7 | **H₁b extended-confirmed**: 46.2 < 83.7 ✓ (mod forgets 72% less than unmod on the 2nd return; gap is 121 survival steps) |
| **Hypertube** | `survival(stage 5) − survival(stage 3)` | **+5.75** | **−19.79** | mod ≥ 0 AND unmod < 0 | **H₁c confirmed**: mod 2nd return ≥ 1st return ✓ (small +5.75 within seed-noise but **signed correctly**); unmod monotone decay ✓ (−19.8 step further forgetting on the 2nd return) |

The seed-noise floor flagged in §1.3 was ±4.4 steps. The Δforget_3 / Δforget_5 contrasts of 95.8 and 121 survival-step gaps are ~22-28× the seed-noise floor — these effects survive the single-seed caveat with a wide margin. The hypertube signal (+5.75 mod, −19.8 unmod) is smaller in magnitude (the absolute difference, ~25 steps, is ~6× seed noise) but the **sign-flip** (mod positive, unmod negative) is the load-bearing read-out per §1.2, and it survives.

### 5.5.4 H₁a re-interpretation — the pre-registered dip-depth formalisation is schedule-asymmetric

H₁a as written in §1.1 says "on stage 2 (the first switch), the modulated agent's post-switch survival dip $d_2^{\text{mod}}$ is at most half the unmodulated agent's $d_2^{\text{unmod}}$." But on this 5-stage schedule **stage 2 is the passive stage** — passive is *easier* than active, so the agent's survival **jumps up** rather than dipping down at the 1→2 boundary. Computing $d_2$ literally per §4.1 gives a **negative dip** for both arms (mod −79.3, unmod −190.2) — i.e. survival rose by 79/190 steps. The "half the dip" predicate is therefore not interpretable on this transition.

The load-bearing dips happen on the **passive→active** transitions (stages 2→3 and 4→5), where the policy trained on the easy passive world has to re-engage active predators. On those transitions both arms suffer roughly equal dip depths (mod 425, 417; unmod 432, 446 — the unmod dips are slightly *deeper* by 7-29 steps but the absolute scale dwarfs the contrast). What separates the two arms is **recovery, not dip depth**: the modulated agent climbs back to ~240 steps within a single rollback window; the unmodulated agent plateaus around 110-130 and never reaches its stage-1 level (278.3).

Practical reading: **H₁a as formalised is unevaluable on this schedule** (the design-doc formal definition was schedule-asymmetric). H₁a's intended *spirit* — "the modulator attenuates the stability gap" — is partially addressed by Δforget_3/5 (mod recovers further) but the immediate dip-depth comparison shows no large effect either way. Mark **H₁a inconclusive** on this experiment as written; recommend re-formalising the dip-depth metric for the active-reintroduction events in any follow-up.

### 5.5.5 Modulator engagement (mechanism arm)

`mod_h` per-step Mahalanobis was **not available** — the project's current `train.py` modulator-diagnostic plumbing logs scalar summary stats of the modulator output (mean/std of gamma_uni, gamma_multi, beta_uni, beta_multi, z_memory; mean/max/min of temperature; grad_norm) rather than the raw `mod_h` hidden vector. The H₁a Mahalanobis check predicted in §1.1 cannot be performed against the 2σ threshold; the surrogate read-out is whether the available summary statistics show **boundary-locked transients** at each stage flip.

Boundary deltas (first ~5% of new stage − last ~5% of previous stage):

| Transition | Δtemperature_mean | Δgamma_uni_mean | Δgrad_norm |
|---|---|---|---|
| 1→2 (active→passive) | +0.001 | −0.060 | **+0.227** |
| 2→3 (passive→active) | **−0.374** | +0.089 | −0.429 |
| 3→4 (active→passive) | **+0.366** | −0.095 | +0.135 |
| 4→5 (passive→active) | **−0.202** | +0.084 | −0.744 |

Three of four boundaries show `temperature_mean` shifts ≥ 0.2 (intra-stage std ~0.07, so ≥ 3σ). Sign is consistent: **temperature drops on entering active stages** and **rises on entering passive stages** — i.e. the modulator's policy-head temperature output sharpens (low value, more deterministic) when discriminating an active predator, and softens when the world becomes passive. `grad_norm` shows large transients at every boundary (range +0.23 to −0.74). The modulator **is engaged** at the boundaries in a behaviourally consistent way; we just lack the formal `mod_h` Mahalanobis statistic.

### 5.5.6 Modulator per-stage means (R2.1 only)

| Stage | temp_mean | temp_max | gamma_uni_mean | gamma_uni_std | gamma_multi_mean | z_memory_std | grad_norm |
|---|---|---|---|---|---|---|---|
| 1 active | 0.70 | 0.94 | 0.82 | 0.58 | 0.96 | 0.69 | 0.10 |
| 2 passive | 0.90 | 1.62 | 0.51 | 0.67 | 0.88 | 0.76 | 0.60 |
| 3 active | 1.12 | 3.02 | 0.56 | 0.83 | 0.91 | 0.84 | 0.28 |
| 4 passive | 1.06 | 3.03 | 0.46 | 0.94 | 0.89 | 0.94 | 0.63 |
| 5 active | 1.22 | 3.17 | 0.59 | 0.98 | 0.85 | 1.05 | 0.35 |

`temperature_max` climbs from 0.94 (stage 1) to ~3.0 (stages 2-5), well below the new `[0.5, 5.0]` clip ceiling — the head is *not* saturating against the raised cap (good; the cap change was on-target). `z_memory_std` grows from 0.69 to 1.05 across the schedule — modulator state is becoming progressively more varied, consistent with the GRU continuing to encode new context information as the schedule cycles.

### 5.5.7 Training-health snapshot (final 5% of each stage)

| Stage | R2.1 mod (entropy_loss / T_starv / T_injury / T_maxsteps / grad) | R2.2 unmod (entropy_loss / T_starv / T_injury / T_maxsteps / grad) |
|---|---|---|
| 1 active | -0.50 / 51% / 32% / 17% / 0.34 | -0.51 / 48% / 37% / 15% / 0.38 |
| 2 passive | -0.66 / 3% / 3% / 94% / 3.10 | -0.47 / 3% / 4% / 93% / 2.53 |
| 3 active | -0.44 / 49% / 43% / **8%** | -0.21 / 72% / 28% / **0.1%** |
| 4 passive | -0.65 / 1% / 3% / 96% / 2.16 | -0.51 / 3% / 2% / 95% / 2.08 |
| 5 active | -0.45 / 47% / 44% / **9%** | -0.34 / 68% / 32% / **0.0%** |

Both runs stable, no NaN, value-loss steady at 0.25-0.40 throughout. Two informative contrasts on the active-return stages:
- **T_MaxSteps fraction**: mod reaches the 500-step horizon ~8-9% of episodes on returns; unmod essentially never does (0.0-0.1%). The modulated agent has a small but real population of fully-surviving episodes; the unmodulated agent does not.
- **Termination cause on returns**: mod splits ~50/45 between starvation and injury; unmod skews ~68-72% starvation. The unmodulated agent on returns is more often **starving** than being killed by the predator, suggesting its policy has collapsed into something that doesn't successfully forage rather than something that successfully forages but gets caught — fundamental policy disruption, not just gain-loss.
- **Entropy_loss** on active returns: mod ~−0.45, unmod ~−0.21 to −0.34. The unmodulated agent's policy is **more concentrated** on returns — it has narrowed onto a (failing) deterministic fallback, while the modulated agent retains exploratory bandwidth.

---

## 6. Conclusions

### 6.1 Summary

**One-paragraph headline.** The five-stage continual probe gives the modulated agent a clear, large-effect win over the unmodulated baseline. On the two return-to-active stages — the load-bearing measurement targets — the modulated agent achieves ~240 survival steps versus the unmodulated baseline's ~110-130, a ~107-132 survival-step gap that is 25× the project's measured seed-to-seed noise floor (±4.4 steps) and roughly the strongest single-experiment positive the architecture-vitality program has produced this month. The pre-registered hypotheses are scored as: **H₁b (catastrophic-forgetting resistance) CONFIRMED** with high margin (mod forgets 65-72% less than unmod on the two returns), **H₁c (Tsuda hypertube reusable subnetwork) CONFIRMED** with the correct sign-flip (mod return-2 ≥ return-1 by +5.75; unmod monotone decay −19.8), and **H₁a (post-switch dip attenuation) INCONCLUSIVE** because the §4.1 dip-depth formalisation was schedule-asymmetric — the load-bearing dips happen on passive→active transitions where both arms drop equivalently, and the modulator's contribution lives in **recovery speed**, not in dip-depth attenuation. The modulator's behavioural engagement at the stage boundaries is consistent with the Lee 2024 / Tsuda 2021 mechanisms (temperature head shifts ≥3σ in 3/4 transitions; sign of the shift is task-consistent — sharper temperature for active discrimination, softer for passive) — but the formal `mod_h` Mahalanobis test from §1.1 / §4.3 is unevaluable because the project does not currently log the raw `mod_h` hidden vector. Single-seed caveat (§1.3) still applies but the effect sizes are far above the seed-noise floor; a 3-seed replication would strengthen the result but is not load-bearing for the verdict.

**Per-hypothesis scoring table.**

| Hypothesis | Predicate (from §1.1 / §4.2) | Outcome | Magnitude vs. threshold |
|---|---|---|---|
| **H₀** (null) | All three contrasts ≤ 5 steps + no mod transient | **REFUTED** | All three contrasts > 5; modulator transients present |
| **H₁a** (dip attenuation) | $d_2^{\text{mod}} \leq \tfrac12 d_2^{\text{unmod}}$ | **INCONCLUSIVE** (formal predicate ill-posed on this schedule) | Dip-depth analysis showed passive→active dips of ~420-445 steps on both arms; mod recovers faster, but the "half the dip" predicate cannot be evaluated as written |
| **H₁b** (forgetting resistance) | $|\Delta_3^{\text{mod}}| \leq \tfrac12 |\Delta_3^{\text{unmod}}|$ | **CONFIRMED** (high margin) | 51.9 ≤ 73.8 (= ½·147.7); 65% less forgetting on 1st return, 72% less on 2nd |
| **H₁c** (hypertube reuse) | mod stage5 ≥ stage3 AND unmod stage5 < stage3 | **CONFIRMED** (sign-flip clean) | mod +5.75 (positive); unmod −19.79 (monotone decay) |

This is the **first positive architecture-vitality finding for the FiLM modulator in this project** after a month of null results on the noise-heterogeneity and temperature-ceiling sweeps. The modulator does **not** help under noisy sensors (per the §0 summary [`20260509_1421_nmn_comparison_study.md`](../../summaries/20260509_1421_nmn_comparison_study.md)); it **does** help under abrupt env-structure switches with returns. The two findings together pin the regime where the architecture earns its name.

### 6.2 Limitations & Open Questions

- **Single seed per cell**. The Δforget contrasts (95.8 and 121 steps) are 22-28× the seed-noise floor and survive the single-seed caveat with a wide margin. The hypertube sign-flip is smaller (absolute difference ~25 steps, ~6× seed noise) and is the strongest argument for a follow-up 3-seed replication; the H₁b contrast does not need replication for the verdict to hold.
- **`mod_h` raw vector is not logged**. The H₁a / §4.3 / §4.4 Mahalanobis-distance mechanism arm is **unavailable** by current logging. Summary-statistic proxies (temperature_mean, gamma_uni_mean, grad_norm) show consistent boundary-locked transients, but the formal 2σ predicate cannot be evaluated. Surfaced in §8.1 as a Metrics Requested item.
- **H₁a's formal dip-depth predicate is schedule-asymmetric**. Stage 2 in this schedule is the easier passive stage, so the agent's survival rises rather than dips at the 1→2 boundary; the §4.1 formalisation gives a "negative dip" that the §4.2 threshold cannot interpret. The load-bearing dips happen on passive→active transitions; a follow-up should re-formalise H₁a per active-reintroduction event (or write the dip as a signed quantity measured against the *prior asymptote in that world*, not the immediately-prior stage's asymptote).
- **`Episode/Term_Predator` is missing**. The unmodulated agent's elevated `T_Starvation` on returns (68-72%, vs. mod's ~50%) is highly suggestive — the unmod policy collapses into a starvation pattern rather than failing on predator encounters — but the project does not currently log a `T_Predator` (deaths-by-predator) terminal cause separately from `T_Injury` (deaths-by-cumulative-injury). Surfaced in §8.2.
- **Stage 1 may not have converged** before the 1.5M-ep budget. R2.1 stage 1 tail mean = 289.5 (vs. specialist `active_matched` ceiling 337.6 — the §B 10M-ep unmodulated specialist on the same env) and R2.2 stage 1 tail mean = 278.3 (vs. same 337.6 ceiling). Both continual cells are at ~83% of the specialist ceiling for stage 1; the Δforget metric subtracts off the same stage-1 floor for each arm, so the *contrast* is undisturbed, but **absolute** forgetting numbers are biased toward "less forgetting" because the stage-1 baseline is below ceiling. The verdict is unchanged.
- **No follow-up gradual-interpolation arm** (per §2.2 — the gradual transition was deferred in v2 synthesis). The abrupt-only design tests the **architectural** claim; the gradual arm would test whether the modulator's contribution is specifically about boundary detection vs. just about general context-switching. Out of scope for this round; recommend it as a future follow-up.

### 6.3 Recommended Next Experiments

In priority order:

1. **3-seed continual replication** (cheap, single-seed → 3 seeds on the same schedule for the modulated arm only). Locks in the H₁b margin and gives a proper CI on the hypertube sign-flip. ~30 h × 3 GPU-slots. Routes via `experiment-designer`.
2. **Add `Episode/Term_Predator` terminal-cause logging** (one line in `train.py`'s episode-end accumulator block; tiny). Settles the "starvation collapse vs. predator failure" hypothesis on the unmodulated baseline's return-stage behavior. Routes via `feature-workflow` (`senior-developer` → `developer`); see §8.2.
3. **Add raw `mod_h` per-step logging or its quartile-summary** (so future continual probes can run the formal Mahalanobis check). Routes via `feature-workflow`; see §8.1.
4. **Re-formalise the H₁a dip-depth metric** per active-reintroduction event (or per relative-to-prior-asymptote-in-this-world) and re-evaluate against the existing R2 data — no new training needed, only a re-extraction. Doc-only follow-up under this same design doc.
5. **Cross-link with the meta sister verdict** ([`NMN_META_2x3_MIXTURE_PROBE`](NMN_META_2x3_MIXTURE_PROBE.md)) once the `--mixture-mode` developer touch lands and the head-to-head cells run. The joint verdict is the relevant input to the senior-developer Phase-0 decision; with continual positive and meta still-pending, **Phase 0 is no longer the obvious next step** — the modulator architecture earns its keep in at least one regime.
6. **Gradual-interpolation arm** of this same probe (boundary blur over ~50K episodes instead of abrupt). Tests whether the modulator's contribution is specifically about boundary detection or about reusable subnetworks more broadly. Lowest priority — only if the user wants to deepen the continual story.

---

## 7. Hand-offs (named-agent routing)

1. **`env-config-auditor`** (BEFORE launch). Audit:
   - the new schedule YAML (`configs/continual/nmn_double_return.yaml`) for schema correctness (`continual.episode_boundaries` strictly increasing, `checkpoint_frequencies` parallel and > 0).
   - all 5 stage YAMLs in `configs/continual/nmn_double_return_stages/` — confirm `01/03/05` body byte-matches `01-interoNocicept_sameProp.yaml`, `02/04` body byte-matches `02-sameProp_R2_passivePredator.yaml`, except for the prepended header comments.
   - the new agent config `configs/models/recurrent_ppo_nmn_film_g1_tempceil5.yaml` for parity with `recurrent_ppo_nmn_het_film_g1.yaml` modulo the `temp_clip` change.
   - obs↔noise invariants (no env-spec changes that would shift modality dimensions).
2. **User authorisation** — review the auditor pass + this design doc; greenlight the launch.
3. **`training-runner`** (after authorisation). Launch the two **Round 2** cells per §3.2:
   - Run R2.1 (modulated) on **node 101 GPU 0** with `--tag rppo_nmn_cont_dr_mod_s0_r2 --wandb-name rppo_nmn_cont_dr_mod_s0_r2 --wandb-group nmn_continual_double_return --wandb-job-type prod --seed 0 --configs-dir configs/continual/nmn_double_return_stages/ --continual-schedule configs/continual/nmn_double_return.yaml --agent_config configs/models/recurrent_ppo_nmn_film_g1_tempceil5.yaml`.
   - Run R2.2 (unmodulated) on **node 101 GPU 1** with `--tag rppo_nmn_cont_dr_unmod_s0_r2 --wandb-name rppo_nmn_cont_dr_unmod_s0_r2 --wandb-group nmn_continual_double_return --wandb-job-type prod --seed 0 --configs-dir configs/continual/nmn_double_return_stages/ --continual-schedule configs/continual/nmn_double_return.yaml --agent_config configs/models/recurrent_ppo_nmn_het_unmod.yaml`.
   - Per project memory rule `feedback_runner_post_launch_pgrep`: verify exactly one PID per tag after launch.
   - Round 1 launch commands (using the un-suffixed `*_s0` tags) are obsolete — do NOT re-issue them; the Round 1 data is discarded.
4. **`experiment-analyzer`** (after training completes). Fill §4 / §5 / §6 of this doc per the analysis plan above. Cross-reference with the sister meta-probe doc (NMN_META_2x3_MIXTURE_PROBE) for the joint architecture-vitality verdict. **[DONE — 2026-05-13]**
5. **`experiment-designer`** (return). Fill §6 Conclusions per the locked H₀/H₁a/H₁b/H₁c predicates once the analyzer has reported. **[NOT NEEDED — analyzer filled §6 directly per project convention; the predicate scoring is in §5.5.3 / §6.1.]**
6. **`senior-developer`** (conditional, only if H₀ confirmed full null on both this experiment AND the meta sister): plan Phase 0 architecture upgrades (T/P split + precision head per project_plan §3.2 / §3.4). **[CONDITION NOT MET — H₀ refuted on this experiment; Phase 0 routing is therefore deprioritised pending the meta sister verdict.]**

---

## 8. Metrics Requested

### 8.1 (analyzer-added 2026-05-13) `modulator/mod_h_*` per-step logging

| Subfield | Content |
|---|---|
| **Metric** | `modulator/mod_h_norm` (scalar, L2-norm of mod_h per step) AND optionally `modulator/mod_h_quartiles` (4 scalars: q25/q50/q75 per dim, averaged across dims, per step). Cheaper alternative to dumping the full 16-d vector. |
| **Why now** | The H₁a mechanism arm (`mod_h` Mahalanobis > 2σ within first 50 episodes of a stage transition) cannot be evaluated against the §4.2 threshold because only summary stats of the *output* of the modulator (gamma, temperature, beta, z_memory) are logged, not the modulator's own hidden state. The summary-stat proxies (temperature_mean, gamma_uni_mean) gave a *positive* boundary-locked signal in this experiment, but the formal predicate was unevaluable. |
| **Where it'd live** | `train.py`'s per-iteration logging block where `modulator/gamma_*` and `modulator/temperature_*` are emitted (~line 1200-1230). The modulator's `nmn_step` function already returns `mod_h` — it's just not being logged. |
| **Cost** | Cheap. One scalar (L2 norm) per iteration is negligible. Four scalars (quartile summary) still negligible. Full 16-d vector would be ~16× current, still cheap. |

### 8.2 (analyzer-added 2026-05-13) `Episode/Term_Predator` distinct terminal cause

| Subfield | Content |
|---|---|
| **Metric** | `Episode/Term_Predator` — fraction of episodes ending due to predator-inflicted death (separate from `Term_Injury` which currently lumps predator-damage and cumulative-injury). |
| **Why now** | On the active-return stages the unmodulated agent has `T_Starvation` ~68-72% vs. the modulated agent's ~50%, suggesting the unmod policy collapses into a starvation mode rather than a predator-failure mode. The current logging cannot disambiguate "the agent gets killed by the predator more often" from "the agent stops eating because the policy is disrupted". |
| **Where it'd live** | `train.py`'s episode-end logging block alongside `Term_Starvation` / `Term_Injury` / `Term_MaxSteps`. The terminal-cause signal already exists internally in the env step function. |
| **Cost** | Cheap (scalar fraction per episode end). |

Both items are listed for the user's awareness; the user decides whether to route them via `feature-workflow`. They are **not blocking** for the verdict in §6.

> **Pre-launch claim (now corrected):** the original §8 of this design doc said "The experiment can be evaluated cleanly with metrics already logged ... `mod_h` per-step (logged for FiLM architectures via existing diagnostic plumbing)". This turned out to be **wrong** — only summary statistics of the modulator's *outputs* (gamma, beta, temperature, z_memory) are logged, not the raw modulator hidden state `mod_h`. The formal Mahalanobis mechanism arm was therefore unevaluable; summary-stat proxies stood in. Surfaced here so a future analyzer does not repeat the mistake.
