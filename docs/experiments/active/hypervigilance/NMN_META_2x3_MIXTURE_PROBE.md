---
title: "NMN meta 2x3 mixture probe — does the modulator factorise context across two manipulation axes?"
topic: hypervigilance
status: active
created: 2026-05-09
last_updated: 2026-05-09T18:50
phase: 0.5
wandb_tag: "rppo_nmn_meta_{2x3_mod,2x3_unmod,spec_*}_s0"
develop_link: docs/develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md
supersedes: null
superseded_by: null
---

# NMN meta 2x3 mixture probe — does the modulator factorise context across two manipulation axes?

> **Status**: 6 of 8 cells running (Runs 3–8, specialist arm); 2 head-to-head cells (Runs 1–2, n102) deferred to round 2 pending `--mixture-mode` developer touch via senior-developer.
> **Date**: 2026-05-09
> **Author**: experiment-designer
> **Related**:
> - Lock memo (postdoc synthesis v2): [`nmn_meta_continual_synthesis_v2.md`](../../../project/ideas/nmn_meta_continual_synthesis_v2.md)
> - Direction memo v2 (professor-rl-bayesian-dl): [`nmn_meta_context_conditioning_v2.md`](../../../project/directions/nmn_meta_context_conditioning_v2.md)
> - Direction memo v1 (still anchored): [`nmn_meta_context_conditioning.md`](../../../project/directions/nmn_meta_context_conditioning.md)
> - Triage anchor: [`20260509_1517_nmn_meta_continual_pivot.md`](../../../project/triage/20260509_1517_nmn_meta_continual_pivot.md)
> - Sister design (continual side, same overnight launch window): [`NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md`](NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md)
> - Calibration anchor (matched-context corner-camping): memory insight `20260509_1532_sameprop_round2_truncated_verdict`
> - Mechanism finding behind the new agent config (`temp_clip = [0.5, 5.0]`): memory insight `20260509_1410_nmn_temp_head_natural_target_3_to_5`

> **One-line scope.** Eight cells (2 head-to-head + 6 specialists, single seed each). The two head-to-head cells (modulated vs. unmodulated) train on a uniform-per-episode-reset random mixture over six contexts arranged as a 2-by-3 cross product: predator behaviour {active, passive} × olfactory assignment {matched, distinct, swapped}. The six specialists train on each individual cell of the cross-product as ceiling references. Headline test: does the modulator's hidden state factorise across the two manipulation axes (P3 in the v2 direction memo §1.3)?

---

## 1. Research Question

The project trains an agent to survive in a small grid world (find food, avoid predators, manage hunger and injury). Two architectures are being compared this week: a **plain agent** (a standard recurrent network — one brain) and a **modulated agent** (the same brain plus an extra small "modulator" head sitting on top, which can dynamically reweight how the main brain reads its senses depending on context).

This experiment asks the modulated agent to do something the academic literature on neuromodulator networks consistently reports as their core function: **identify which of several possible "worlds" it is in from observations alone, and gate its policy on that identity** — without ever being told which world it is in. The Ben-Iwhiwhu 2022a NPN paper showed this can give roughly 2× success-rate gain on Meta-World ML45. We are testing whether our particular FiLM-modulated agent has the same property in our grid world.

The world has **two manipulation axes**, deliberately. Axis one is the **predator's behaviour**: an *active* predator (full hunt mode — chases the agent when it gets too close) versus a *passive* predator (hunt mode is structurally unreachable; the predator just patrols its quadrant — kinematically rabbit-like, but still inflicts damage on contact). Axis two is the **olfactory assignment** — which one-hot smell vector each animal emits:

- **matched** — predator and rabbit both emit `[0,1,0,0,0]`. There is no olfactory cue at all; the agent has to discriminate from the predator's *movement signature* (recurrent state).
- **distinct-canonical** — predator emits `[0,1,0,0,0]`, rabbit emits `[0,0,1,0,0]`. The two are linearly separable on smell alone; reading olfactory is sufficient.
- **swapped** — predator emits `[0,0,1,0,0]`, rabbit emits `[0,1,0,0,0]`. Olfactory now actively *misleads* — an agent that learned "smell channel 2 = predator" on the distinct context will, on the swapped context, react to the rabbit as if it were a predator.

The cross-product is six contexts. The head-to-head cells train **one** agent on a uniform random mixture over all six (the agent never knows which context the current episode is in; it must infer from observations). The six specialist cells train one agent per context as a ceiling reference.

**The headline test (factorisation, "P3").** A modulator that builds a context representation by memorising a 6-bit code can solve the mixture trivially without learning anything generalisable about *how* the world varies. A modulator that builds a *factorised* representation — one subspace for "is the predator active or passive", another subspace for "what does the olfactory channel mean here" — has learned something more useful: it can compose. The 6×6 CKA Gram matrix on the modulator's hidden state at end-of-training will tell us which: if off-diagonal CKA between contexts that share the predator-behaviour axis matches off-diagonal CKA between contexts that share the olfactory axis, the modulator factorises (the v2 direction memo's P3 statistic, $|\bar\Delta_{\text{CKA}}^{\text{factor}}| < 0.10$). If they are very different, the modulator encodes one axis well and the other badly. **This factorisation property is, to the best of the prior literature search, not directly demonstrated in any of the project's reference papers** — a positive result is novel, and a null is still informative (the modulator is doing context coding but not the *kind* of context coding the literature implies).

**Important caveat that must be in the user's mind before reading any positive result.** This experiment runs **one seed per cell**. The earlier temperature-ceiling rerun discovered ±4.4-step seed-to-seed noise in the project's survival numbers; under that noise, **a single unlucky seed could mislead the verdict in either direction**. The single-seed choice is deliberate (the user is spending compute across 10 cells in one overnight window and prioritised breadth over within-cell statistical power), but the read-out from this experiment is necessarily **directional, not statistically definitive**. If the modulated-vs-unmodulated head-to-head sits within ±5 survival steps on the swapped contexts, the experiment is inconclusive on its own and a follow-up multi-seed replication is required. The 6×6 CKA Gram statistic is more robust to seed noise (it summarises representation geometry across many samples), so the P3 test is somewhat protected; the survival arm is not.

**Plain-language outcomes:** "Positive" = the modulated agent survives meaningfully longer than the plain agent on the swapped contexts (where olfactory misleads), AND the modulator's hidden state factorises across the two manipulation axes. "Refutation" = parity survival on swapped, **and** failure of the factorisation statistic — the modulator does not even build the representational structure literature predicts.

### 1.1 Formal hypotheses

> **H₀** (full null): On the load-bearing swapped contexts (`(active, swapped)` and `(passive, swapped)`), modulated and unmodulated mixture-trained agents have indistinguishable survival ($|\Delta\text{survival}^{\text{mod}-\text{unmod}}|_{\text{swap}} \leq 5$ steps), AND the modulator's 6×6 CKA Gram matrix on hidden states fails the within-vs-between contrast ($\bar\Delta_{\text{CKA}} < 0.05$). Architecture is failing the floor test in this regime.

> **H₁a** (P1 — context-distinguishability of $h^{\text{mod}}$): The modulator's 6×6 CKA Gram matrix shows within-context similarity meaningfully higher than between-context similarity, $\bar\Delta_{\text{CKA}} \geq 0.05$ (averaging across the 6 contexts). The modulator builds context-distinguishable hidden states. Necessary but not sufficient for an architecture-vitality positive.

> **H₁b** (P3 — factorisation across the two manipulation axes): Conditional on H₁a, the off-diagonal CKA gaps between contexts that share predator-behaviour but differ on olfactory match those between contexts that share olfactory but differ on predator-behaviour: $|\bar\Delta_{\text{CKA}}^{\text{factor}}| < 0.10$. The modulator does not just encode "which of 6"; it encodes the underlying 2-D structure of the manipulation space. Strongest positive result of this experiment.

> **H₁c** (swapped-context survival win): On the load-bearing swapped contexts, modulated agents survive longer than unmodulated by an effect-size that survives single-seed noise: $\Delta\text{survival}^{\text{mod}-\text{unmod}}_{\text{swap}} > +5$ steps on at least one of `(active, swapped)`, `(passive, swapped)`. The factorised representation cashes out into behavioural advantage on the contexts where olfactory misleads.

### 1.2 What confirms vs. refutes — pre-specified

| Outcome | Predicate | Verdict |
|---|---|---|
| **H₁a + H₁b + H₁c all confirmed** | $\bar\Delta_{\text{CKA}} \geq 0.05$ AND $\|\bar\Delta_{\text{CKA}}^{\text{factor}}\| < 0.10$ AND $\Delta\text{survival}_{\text{swap}}^{\text{mod-unmod}} > 5$ on at least one swap context | Headline architecture-vitality positive. Modulator factorises and the factorised representation is behaviourally useful. **Vindicates the FiLM family pre-Phase-0**; reframes the v8/Experiment-1/Experiment-2 sensory-modulation null as a "hard problem" rather than an architecture failure. |
| **H₁a + H₁b confirmed, H₁c null** | $\bar\Delta_{\text{CKA}} \geq 0.05$ AND $\|\bar\Delta_{\text{CKA}}^{\text{factor}}\| < 0.10$ BUT $\|\Delta\text{survival}_{\text{swap}}\| \leq 5$ | Representation arm wins, behavioural arm null. The modulator builds factorised state but the downstream policy does not exploit it. Same pattern as the earlier study's "head fully unleashed but no behavioural gain" — softer architecture-vitality positive; hand to senior-developer for the redesign of the modulator-to-policy connection. |
| **H₁a confirmed, H₁b null** | $\bar\Delta_{\text{CKA}} \geq 0.05$ but $\|\bar\Delta_{\text{CKA}}^{\text{factor}}\| > 0.15$ | Modulator builds context-distinguishable state but doesn't factorise — encodes one axis (likely the easier one, predator-behaviour from movement signature) better than the other. **The 16-dim modulator GRU is under-capacitised for 2-axis factorisation** (per v2 memo §6.1 — argues weakly for the T/P split per project_plan §3.2). Hand to senior-developer with a specific architectural reading. |
| **H₀ confirmed (full null)** | $\bar\Delta_{\text{CKA}} < 0.05$ AND $\|\Delta\text{survival}_{\text{swap}}\| \leq 5$ | Modulator does not build a context code at all. Combined with a sister-experiment null on the continual probe, this is the project's signal that pre-Phase-0 is not a viable starting architecture. Hand to senior-developer for Phase 0 (T/P split + precision head). |
| **Specialists fail to separate** | Specialist survival on the 6 cells does not span a meaningfully wider range than the head-to-head agents | The contexts are not actually task-conflicting (Risk A in v2 memo §9). Mark experiment design degenerate; the 2×3 mix is not a real architecture-vitality test on this env. |
| **Refutation by training instability** | Either head-to-head cell NaNs, value/grad explodes, or the mixture sampler under-represents one context such that effective N per context falls below ~5,000 episodes | Marked inconclusive for the affected predicate. If the mixture sampler is the issue, route to developer; if instability, halt and report. |

### 1.3 Why a single seed per cell, and what the matched-context corner-camping caveat means

- Eight cells across this experiment in one overnight launch — single seed per cell to fit the 10-GPU budget alongside the continual sister experiment.
- ±4.4-step seed-to-seed noise per memory insight `20260509_1410` — same caveat as the continual sister, reproduced here so each doc is self-contained: **1 unlucky seed could mislead the survival verdict in either direction**.
- **Matched-context survival is contaminated by corner-camping** per memory insight `20260509_1532_sameprop_round2_truncated_verdict`. On the active+matched cell of Round 2, the agent achieved survival not by discriminating predator from rabbit but by parking itself in the BR corner where one rabbit and the food spawn but no predator. Survival numbers on the matched contexts (`(active, matched)`, `(passive, matched)`) are *uninterpretable as discrimination* without per-quadrant occupancy; the analyzer must report quadrant occupancy on every matched-context cell and treat any matched-context modulated-vs-unmodulated contrast as null unless camping rates are equalised. **This is why the swapped contexts carry the falsifying weight** (corner-camping does not help on swap — the rabbit corner now smells like predator).
- The 6×6 CKA Gram statistic (P1 / P3 arm) is more robust to seed noise than survival because it averages over many samples of `mod_h` per context. The factorisation statistic is the stronger arm of the read-out.

---

## 2. Experimental Design

### 2.1 Independent Variables

| Variable | Values | Rationale |
|---|---|---|
| `agent.modulation.type` | {FiLM, null} | The treatment-vs-control contrast on the head-to-head pair (cells 1, 2). |
| Predator behaviour `m` | {active, passive} | Axis one of the 2-axis manipulation space. |
| Olfactory assignment `p` | {matched, distinct-canonical, swapped} | Axis two. Matched = no olfactory cue; distinct = sufficient olfactory; swapped = misleading olfactory (the load-bearing context). |
| Context sampling | uniform random per episode reset over the 2×3 = 6 cell cross-product | Per professor-rl/bdl v2 memo §5.1: not curriculum, not stratified. Requires the `--mixture-mode` developer touch (see §2.6). |

The 6 specialist cells (3-8) train on individual `(m, p)` configs without mixture sampling — single-context training is the standard config-only path.

### 2.2 Controlled Variables

```yaml
# Six per-context env configs (head-to-head agents sample uniformly across these per reset;
# specialists train on one each)
configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml   # (active, matched)   — REUSED, existing
configs/experiment/nmn_meta_2x3_mixture/active_distinct.yaml        # (active, distinct)  — NEW
configs/experiment/nmn_meta_2x3_mixture/active_swapped.yaml         # (active, swapped)   — NEW
configs/experiment/hypervigilance/02-sameProp_R2_passivePredator.yaml  # (passive, matched) — REUSED, existing
configs/experiment/nmn_meta_2x3_mixture/passive_distinct.yaml       # (passive, distinct) — NEW
configs/experiment/nmn_meta_2x3_mixture/passive_swapped.yaml        # (passive, swapped)  — NEW

# Body / sensors / noise — IDENTICAL across all 6 env configs (env-config-auditor enforces).
# perceptual_noise.enabled: false (clean obs).
# properties_std: [0,0,0,0,0] across all olfactory channels (sameProp regime; risk D in
#   v2 memo §9 pre-empted — no leakage of context via olfactory marginal noise).

# Agent — shared hyperparameters (v8 §2.2 anchor) across the head-to-head pair
# (the 6 specialists also use these hyperparameters except for the specialist
# unmodulated arm; see §3.1 below)
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

# Training horizon — ~10M training steps per cell; ~12-14h on lab nodes
total_timesteps: 10_000_000  # head-to-head cells (richer mix)
# specialists may run shorter (single context = faster convergence) — see §3.1
num_envs: 128 (default)
seed: 0 (per cell)
```

### 2.3 Confounds & Limitations

| Confound | Affected Runs | Severity | Mitigation |
|---|---|---|---|
| Single seed per cell (±4.4-step seed noise per insight `20260509_1410`); 1 unlucky seed could mislead the verdict | all 8 | **High** for survival arm; **Med** for CKA arm (CKA averages across many samples) | Pre-registered effect-size thresholds (>5-step survival win on swap, $|\bar\Delta_{\text{CKA}}^{\text{factor}}| < 0.10$). Inconclusive contrasts surface a follow-up multi-seed request. |
| Matched-context survival contaminated by corner-camping (per insight `20260509_1532`) | `(active, matched)`, `(passive, matched)` cells (and the matched fraction of the head-to-head mix) | **High** for matched-cell survival interpretation | Analyzer must report per-quadrant occupancy on every matched-context cell. Survival differences on matched cells are uninterpretable without camping equalisation. **Swap is the load-bearing context.** |
| Mixture sampler is a NEW code path (`--mixture-mode`, half-day developer touch) | head-to-head cells (1, 2) | Med | Pre-launch: developer's implementation routed via senior-developer planning; sanity-tested with a small N short run before launch. Per-context episode counts logged so analyzer can verify uniform sampling. |
| Per-context episode count under-representation (e.g., if mixture is implemented incorrectly and the agent sees one context 3× as often as another) | head-to-head cells | Med | Logged per-context episode count; analyzer sanity-checks it before drawing conclusions. |
| 6×6 CKA Gram matrix requires `mod_h` to be logged with context labels | mod head-to-head, mod specialists | Low | Existing diagnostic plumbing logs `mod_h`; the *context label per episode* is what the new mixture machinery must also log. Pre-flight check by env-config-auditor. |
| Specialist ceiling references run only 1 seed each — if a specialist is unlucky, it could falsely depress the ceiling for its context | specialists 3-8 | Low | Specialists are sanity references, not the headline. The 6×6 CKA Gram + head-to-head survival drive the verdict regardless. |
| The active+matched cell (cell 3) and passive+matched cell (cell 6) reuse the existing `01-interoNocicept_sameProp.yaml` and `02-sameProp_R2_passivePredator.yaml` env configs verbatim. The 4 new configs (active+distinct, active+swapped, passive+distinct, passive+swapped) modify only the olfactory `properties` field and (for passive variants) the predator behaviour fields. | all | Low | env-config-auditor verifies the diff scope. |

### 2.4 Mixture-mode resolution (Option A: half-day developer touch)

The user offered three ways to resolve the meta-side mixture-mode blocker (the v2 synthesis flagged that `train.py` needs to switch the env config at every episode reset):

- **Option (a)**: half-day `senior-developer` plan + `developer` touch to add a `--mixture-mode` flag.
- **Option (b)**: config-only "alternating-episode workaround" if the existing continual loader supports a YAML list of configs.
- **Option (c)**: defer the 2 head-to-head meta cells; fill 102:0 + 102:1 with extra specialists or a 3rd continual seed.

After reading `train.py` lines 130-200 and `src/environment/wrapper.py`:

The continual schedule machinery requires **strictly increasing cumulative episode boundaries** (`stage_for_episode(episode)` returns the stage index for an episode count) and loads **one stage YAML per file** from `glob.glob(*.yaml)` sorted alphabetically. There is no mechanism for a single file to be reused across multiple stages. A workaround that cycles through 6 contexts in tiny blocks would require duplicating each context-YAML N times into the configs-dir (e.g., 6 contexts × 50 blocks × 50-ep-stages = 300 stage YAMLs). Beyond aesthetics, this is **stratified sampling**, not uniform-per-reset sampling — the v2 direction memo §5.1 explicitly rules stratified out because it gives the agent a deterministic context-detection signal from episode index that lets it bypass the modulator.

Furthermore, `ParallelEnv(params)` builds JIT-compiled vmapped step/reset functions closed over a single `EnvParams` struct. Per-episode-reset switching requires either (a) an `EnvParams` with a `config_id` slot routed via `lax.switch` inside `jax_reset` / `jax_step`, or (b) maintaining N envs and routing per-env-per-reset to one of them. Both are real changes; the (a) workaround does not deliver them.

**Verdict: option (a)**. The mixture-mode flag is the right path. **Routing**: the user invokes `feature-workflow` (`senior-developer` plans → `developer` implements) to add the `--mixture-mode` flag and the per-context episode-count logging BEFORE this experiment launches. This experiment's launch is gated on that touch landing. The continual sister experiment (single-stage-train no mixture needed at the launch boundary, just the existing continual schedule machinery) is **not** gated — it can launch before the mixture flag lands.

A minimal scope for the developer touch (so the senior-developer planning has somewhere to start):

- Accept a new CLI flag `--mixture-mode <configs-dir>` mutually exclusive with `--config` and `--configs-dir`.
- At every env reset (whether it's an episode boundary or a parallel-env reset), sample one of N pre-loaded `EnvParams` structs uniformly at random and use it for that env's reset.
- Log per-episode `context_id` (the index into the alphabetic ordering of the configs-dir) to WandB so the analyzer can stratify metrics by context.
- The existing `--continual-schedule` path is untouched.

Implementation hint (not a plan): a JAX-friendly path uses `lax.switch(key_idx, branches)` over per-context `jax_reset` closures, with `key_idx = jax.random.randint(reset_key, ...)`. The N step functions can stay distinct (or use the same `lax.switch` trick) since the env's dynamics also depend on `EnvParams`. This is `developer`'s call to scope; the design doc surfaces the constraint, not the implementation.

### 2.5 Why the 4 new env configs (and not 6)

`(active, matched)` and `(passive, matched)` are byte-identical to the existing `01-interoNocicept_sameProp.yaml` and `02-sameProp_R2_passivePredator.yaml` in the project's hypervigilance config family. Reusing them keeps the matched-context cells consistent with the project's prior matched-olfactory work (Rounds 1 and 2 sameProp). The 4 new configs cover only the `(distinct, swapped)` × `(active, passive)` cross-product not previously instantiated.

### 2.6 Why $|\mathcal{P}| = 3$, not 2

Per professor-rl/bdl v2 memo §2.2: a 2×2 mixture (active/passive × matched/swapped) can be solved by a 2-bit context code without any factorisation, leaving P3 (the headline test) unfalsifiable. Three olfactory levels force the modulator into a richer encoding where the factorisation property is visible in the off-diagonal CKA structure. Going beyond 3 (e.g., adding distinct-canonical-2 with rabbit on e4) hits diminishing diagnostic returns at marginal cost.

---

## 3. Launch Manifest

System-of-record. `experiment-designer` filled the planned columns; `training-runner` will fill the actuals at launch. Single seed per cell.

| Run | Status | Cell | Tag (= wandb-name) | wandb-group | wandb-job-type | Seed | Node | GPU | Launched at | WandB run ID | Log path |
|-----|--------|------|--------------------|-------------|----------------|------|------|-----|-------------|--------------|----------|
| 1 | planned | 2x3_mix_modulated | rppo_nmn_meta_2x3_mod_s0 | nmn_meta_2x3_mixture | prod | 0 | 102 | cuda:0 | — | — | — |
| 2 | planned | 2x3_mix_unmodulated | rppo_nmn_meta_2x3_unmod_s0 | nmn_meta_2x3_mixture | prod | 0 | 102 | cuda:1 | — | — | — |
| 3 | running | spec_active_matched | rppo_nmn_meta_spec_active_matched_s0 | nmn_meta_2x3_mixture | prod | 0 | 103 | cuda:0 | 2026-05-09T18:33:51 | p9g5kjx3 | logs/20260509_183351.log |
| 4 | running | spec_active_distinct | rppo_nmn_meta_spec_active_distinct_s0 | nmn_meta_2x3_mixture | prod | 0 | 103 | cuda:1 | 2026-05-09T18:36:05 | iktjhpmm | logs/20260509_183605_rppo_nmn_meta_spec_active_distinct_s0.log |
| 5 | running | spec_active_swapped | rppo_nmn_meta_spec_active_swapped_s0 | nmn_meta_2x3_mixture | prod | 0 | 104 | cuda:0 | 2026-05-09T18:39:09 | 2p5zgdk4 | logs/20260509_183909.log |
| 6 | running | spec_passive_matched | rppo_nmn_meta_spec_passive_matched_s0 | nmn_meta_2x3_mixture | prod | 0 | 104 | cuda:1 | 2026-05-09T18:42:09 | 958mba24 | logs/20260509_184209.log |
| 7 | running | spec_passive_distinct | rppo_nmn_meta_spec_passive_distinct_s0 | nmn_meta_2x3_mixture | prod | 0 | 105 | cuda:0 | 2026-05-09T18:45:13 | z5dfkzw5 | logs/20260509_184513.log |
| 8 | running | spec_passive_swapped | rppo_nmn_meta_spec_passive_swapped_s0 | nmn_meta_2x3_mixture | prod | 0 | 105 | cuda:1 | 2026-05-09T18:48:36 | 44rumz7m | logs/20260509_184836.log |

### 3.1 Configs to Produce

| Run | Config (env / mixture) | Config (agent) |
|-----|-------------------------|----------------|
| 1 | `--mixture-mode configs/experiment/nmn_meta_2x3_mixture/` (after dev-touch lands; see §2.4) | `configs/models/recurrent_ppo_nmn_film_g1_tempceil5.yaml` |
| 2 | (same) | `configs/models/recurrent_ppo_nmn_het_unmod.yaml` (existing) |
| 3 | `--config configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml` (existing) | `configs/models/recurrent_ppo_nmn_het_unmod.yaml` (existing) |
| 4 | `--config configs/experiment/nmn_meta_2x3_mixture/active_distinct.yaml` | `configs/models/recurrent_ppo_nmn_het_unmod.yaml` (existing) |
| 5 | `--config configs/experiment/nmn_meta_2x3_mixture/active_swapped.yaml` | `configs/models/recurrent_ppo_nmn_het_unmod.yaml` (existing) |
| 6 | `--config configs/experiment/hypervigilance/02-sameProp_R2_passivePredator.yaml` (existing) | `configs/models/recurrent_ppo_nmn_het_unmod.yaml` (existing) |
| 7 | `--config configs/experiment/nmn_meta_2x3_mixture/passive_distinct.yaml` | `configs/models/recurrent_ppo_nmn_het_unmod.yaml` (existing) |
| 8 | `--config configs/experiment/nmn_meta_2x3_mixture/passive_swapped.yaml` | `configs/models/recurrent_ppo_nmn_het_unmod.yaml` (existing) |

**Specialist arm-choice note.** The 6 specialists use the **unmodulated** baseline architecture, not the modulated one. Specialists are *ceiling references* — they show "what's achievable on this env structure if you just train one agent per context cleanly". Whether the modulated agent could match a per-context-trained modulated specialist is a question the head-to-head cells answer (the modulated mixture-trained agent's per-context survival should be comparable to a modulated specialist's; that is what context-conditioning *means*). Adding per-context modulated specialists would double the cell count without changing the headline read-out.

**New files this experiment produces** (in `configs/`):

- `configs/experiment/nmn_meta_2x3_mixture/active_distinct.yaml` — (active, distinct-canonical olfactory).
- `configs/experiment/nmn_meta_2x3_mixture/active_swapped.yaml` — (active, swapped olfactory). LOAD-BEARING.
- `configs/experiment/nmn_meta_2x3_mixture/passive_distinct.yaml` — (passive, distinct).
- `configs/experiment/nmn_meta_2x3_mixture/passive_swapped.yaml` — (passive, swapped). LOAD-BEARING.
- `configs/models/recurrent_ppo_nmn_film_g1_tempceil5.yaml` — new canonical FiLM g1 modulated agent config with `temp_clip: [0.5, 5.0]` (also used by the continual sister experiment).

The matched cells (`01-interoNocicept_sameProp.yaml`, `02-sameProp_R2_passivePredator.yaml`) are **REUSED** unchanged from the project's existing hypervigilance config family. The unmodulated agent config is REUSED unchanged.

---

## 4. Analysis Plan (pre-registered)

### 4.1 Primary statistics

Per H₁a/H₁b/H₁c, three primary statistics:

1. **6×6 CKA Gram on $h^{\text{mod}}$ at end-of-training** (mod head-to-head cell only, run 1):
   $$ G_{ij} = \mathrm{CKA}\!\left(H^{\text{mod}}_{c_i},\, H^{\text{mod}}_{c_j}\right), \qquad c_i, c_j \in \mathcal{M} \times \mathcal{P}, $$
   with two summary statistics (per v2 direction memo §4.1):
   $$ \bar\Delta_{\text{CKA}} \triangleq \mathbb{E}_i[G_{ii}] - \mathbb{E}_{i \neq j}[G_{ij}], $$
   $$ \bar\Delta_{\text{CKA}}^{\text{factor}} \triangleq \mathbb{E}[G_{(m_1, p),\,(m_2, p)}] - \mathbb{E}[G_{(m, p_1),\,(m, p_2)}]. $$

2. **Per-context survival** for the head-to-head cells (modulated and unmodulated), stratified by `(m, p)`. The load-bearing contrast is `(active, swapped)` and `(passive, swapped)`: $\Delta\text{survival}^{\text{mod-unmod}}_{m, \text{swap}}$.

3. **Specialist ceiling reference** (6 cells): per-context survival for an unmodulated specialist trained only on that context. Compared against the head-to-head cells' per-context survival as a "fraction of ceiling" read-out.

### 4.2 Effect-size thresholds (locked, pre-registered per v2 synthesis §5.5)

- H₁a positive: $\bar\Delta_{\text{CKA}} \geq 0.05$.
- H₁b positive: $|\bar\Delta_{\text{CKA}}^{\text{factor}}| < 0.10$ (the factorisation success threshold per v2 memo §4.1).
- H₁c positive: $\Delta\text{survival}^{\text{mod-unmod}}_{m, \text{swap}} > +5$ steps on at least one of `(active, swapped)`, `(passive, swapped)`.
- H₀ confirmed: $\bar\Delta_{\text{CKA}} < 0.05$ AND $|\Delta\text{survival}_{\text{swap}}| \leq 5$.

### 4.3 Calibration (per v2 synthesis §6 — must thread through analysis)

For every matched-olfactory cell (`(active, matched)`, `(passive, matched)`, and the matched-fraction of the head-to-head mix): report **per-quadrant occupancy** alongside survival. Specifically, for each cell, log the fraction of episode steps spent in each of TL, TR, BL, BR (using the agent's grid position). If the matched-context modulated and unmodulated arms camp at similar rates, matched-context survival differences are noise; if they camp differently, the modulator is doing different *escape* policies, not different *context-conditioning* — distinct phenomena that look the same in raw survival.

### 4.4 Temporal evolution (mandatory per project rules)

Plot survival vs. training step for every cell. For the head-to-head cells, also plot per-context survival (stratified into the 6 (m, p) bins) on the same axes. The temporal arm catches: (a) early-learning artefacts (e.g., is the modulated cell starting from a worse stage that just hasn't caught up?); (b) context-imbalance during training (does one context dominate early gradient updates?); (c) modulator-internal trajectories (gamma_multi, sigma(z_uni), tau_pi over training).

### 4.5 Per-cue clamp ablation (post-training, optional — only if developer touch lands in time)

Per professor-rl/bdl v2 memo §3, a follow-up eval pass (`--obs-clamp olfactory`, `--mod-clamp`) on the trained mod head-to-head agent produces a 6×4 table of per-context survival under {free, olf-clamp, mod-clamp, both-clamp}. The signed predictions in v2 memo §3 ($\rho^{\text{mod}}$ largest on swap, $\rho^{\text{olf}}$ negative on swap) are the strongest possible disambiguator of "modulator does context-conditioning" vs. "modulator uses olfactory shortcut". This eval pass requires a separate developer touch (the `--obs-clamp` and `--mod-clamp` flags on `evaluate.py`) — **out of scope for this experiment's launch**, but pre-registered here as the natural follow-up if the headline result is mixed (e.g., H₁a + H₁b confirmed but H₁c null → the clamp matrix tells us why).

---

## 5. Failure-Mode Catalog (pre-decided)

| Failure mode | Reading |
|---|---|
| Mixture sampler under-represents one context (per-context episode count differs by > 20% from uniform) | Mark the head-to-head cells inconclusive; route to developer for sampler bug. |
| Specialists fail to separate (per-context survival range < 10 steps across the 6 specialists) | The contexts are not actually task-conflicting; the 2×3 mix is degenerate on this env. Mark experiment design itself flagged; do NOT use it to make architecture-vitality claims. |
| `(active, swapped)` specialist achieves survival floor (< 30 steps) | Swap is too hard for any architecture — this is Risk B in v2 memo §9. Drop swap from the falsifying read-out and re-interpret as a 2×2 (matched/distinct) mix. Result is a weaker but cleaner P1-only test. |
| `mod_h` not logged with context labels | 6×6 CKA Gram unavailable; H₁a/H₁b inconclusive. Survival arm (H₁c) still readable. |
| Headline contrast within ±5 steps on H₁c AND CKA arm null | Real null. Hand to senior-developer for Phase 0. |
| Headline contrast within ±5 steps on H₁c BUT CKA arm positive (factorisation present, no behavioural cash-out) | Soft positive on representation; null on behaviour. Hand to senior-developer for the modulator-to-policy connection. The clamp ablation follow-up (§4.5) is the cleanest disambiguator here. |
| Single-seed contrast says "modulated is WORSE on swap" by > 5 steps | Same disambiguation problem as the continual sister: cannot distinguish "modulator is genuinely harmful" from "unlucky seed" without replication. Pre-register the request for a 3-seed swap replication. |
| Either head-to-head cell NaNs or the mixture sampler segfaults / leaks | Halt; route to developer. |

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

1. **`senior-developer` + `developer`** (BEFORE launch, gates the head-to-head cells). Plan and implement the `--mixture-mode` flag per §2.4. The continual sister experiment ([NMN_CONTINUAL_DOUBLE_RETURN_PROBE](NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md)) is NOT gated on this and can launch first. Specialists 3-8 are also NOT gated on this and can launch first (they use single-config training).
2. **`env-config-auditor`** (BEFORE launch, parallel with #1). Audit:
   - the 4 new env configs in `configs/experiment/nmn_meta_2x3_mixture/` for parity (body / sensors / noise byte-identical across all 6 contexts; differ ONLY in olfactory `properties` and predator behaviour fields).
   - obs↔noise invariants (no env-spec changes shifting modality dimensions across the 6 contexts; if any shift, the mixture sampler at episode reset breaks).
   - the new agent config `configs/models/recurrent_ppo_nmn_film_g1_tempceil5.yaml` for parity with `recurrent_ppo_nmn_het_film_g1.yaml` modulo the `temp_clip` change.
3. **`experiment-analyzer` (pre-launch, parallel)**. Run the 6×6 ΔCKA pre-check on existing v8 trajectories per the v2 direction memo §4.1 — the cheap measurement-only probe that runs without GPU spend. The result is informational (the lock-menu Option C launches anyway), but a v8-positive ΔCKA strengthens the prior on H₁a/H₁b and a v8-negative caveats the launch.
4. **User authorisation** — review #1 (mixture-mode lands), #2 (auditor pass), and this design doc; greenlight launch.
5. **`training-runner`** (after authorisation). Launch all 8 cells per the manifest. Pre-flight: confirm node 102/103/104/105 conda envs (per memory rule `feedback_runner_node_env_preflight`); verify exactly one PID per tag post-launch (per `feedback_runner_post_launch_pgrep`).
6. **`experiment-analyzer`** (after training completes). Fill §4 / §5 / §6 of this doc per the analysis plan above. **Cross-reference with the continual sister experiment's verdict for the joint architecture-vitality conclusion.** Per §4.3, report per-quadrant occupancy on every matched-context cell.
7. **`experiment-designer`** (return). Fill §6 Conclusions per the locked H₀/H₁a/H₁b/H₁c predicates once the analyzer has reported.
8. **`senior-developer`** (conditional, only if H₀ confirmed full null on both this experiment AND the continual sister): plan Phase 0 architecture upgrades (T/P split + precision head per project_plan §3.2 / §3.4). The v2 memo §6 specifically argues the precision head case is *tightened* by a swap-context null because the swap context is precisely where down-weighting a misleading cue (the precision head's job) is load-bearing.

## 8. Metrics Requested (one new logging requirement)

| Subfield | Content |
|---|---|
| **Metric** | `episode/context_id` — per-episode integer label indicating which of the 6 (m, p) contexts the episode is currently sampling from. Zero-cost scalar per episode end. |
| **Why now** | The 6×6 CKA Gram (H₁a/H₁b) requires `mod_h` to be stratifiable by context, and the per-context survival stratification (H₁c) requires the same. Without this label, the mixture-mode head-to-head cells produce un-stratifiable WandB rows. |
| **Where it'd live** | `train.py`'s episode-end logging block (~line 1230 onwards) — alongside the existing per-episode survival / behavior accumulators. Likely needs to be passed through from the new `--mixture-mode` machinery (the env-reset callback knows which context it just sampled). |
| **Cost** | Cheap (scalar, per episode end). |

This metric is part of the same `--mixture-mode` developer touch in §2.4 — it is not a separate ask, but listed here so the analyzer's data-availability constraints are explicit. The user's `feature-workflow` invocation for the mixture-mode plan should include this.
