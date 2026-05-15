---
title: "NMN as scaling-and-shifting implementation of RL-hyperparameter modulation (v2 — after professor review)"
status: draft
audience: user, pi, experiment-designer, senior-developer
last_updated: 2026-05-16
supersedes: docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation.md
related:
  - docs/project/directions/nmn_meta_context_conditioning_v2.md
  - docs/project/ideas/20260515_continual_learning_probe_program.md
  - docs/experiments/summaries/20260513_0321_nmn_comparison_study.md
empirical_anchor: .claude-memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md
revision_summary: "v2 incorporates feedback from professor-neuromodulation and professor-rl-bayesian-dl on v1. Five load-bearing fixes applied: (1) Tsuda 2021 re-cited as multiplicative-weight, not additive-shift; (2) Ferguson & Cardin 2020 Box 1c added as additive-arm anchor; (3) Rodriguez-Garcia 2026 differentiation written into §3 (gradient-level vs forward-activation-level); (4) Lee et al. 2024 promoted to closest precedent in §4; (5) Doya/Yu-Dayan ACh branch resolved (chose Doya-α). Single-mod_h bottleneck escalated from open question to prerequisite. Falsifier resolved as combined (b′ + d) two-stage test."
---

# NMN as scaling-and-shifting implementation of RL-hyperparameter modulation (v2)

## 1. What this direction proposes

This memo redirects the project's neuromodulator-network work away from "the **NMN** (a small side-network that watches state and emits scaling/shifting signals into the main policy) helps **continual learning**" toward a stronger claim: **the NMN's algorithm — multiplicatively scale and additively shift the policy's internal activations — is simultaneously the cellular-level account of biological neuromodulation (gain control on target neurons) and the behaviour-level account of how RL hyperparameters (learning rate, exploration temperature, discount) get modulated state-by-state**. The bridge is a **FiLM** layer (feature-wise linear modulation: $h \mapsto \gamma(c) \odot h + \beta(c)$). Our anchor — the modulator beat the unmodulated baseline by ~25× the seed-noise floor on the second return to a previously-seen stage ([memory insight 20260513_0014](../../../.claude-memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md)) — is reread as a readout of a state-dependent hyperparameter shift, not as continual-learning resistance.

## 2. The two lineages

| Lineage | Mechanism | Anchors |
|---|---|---|
| **Doya — RL hyperparameters** | One scalar per system retunes a knob. DA ↔ learning rate / TD error; **ACh ↔ learning rate $\alpha$** (Doya 2002 §3.4 — chosen over the out-of-corpus Yu & Dayan precision branch); NA ↔ inverse temperature; 5-HT ↔ discount. | Doya 2002; Lee 2024; Ben-Iwhiwhu 2022; Wang 2024. |
| **Neural gain — scale and shift** | Multiplicative arm scales weights or slope. Additive arm shifts the operating point (rheobase — the current at which a neuron starts firing). | Multiplicative: Tsuda 2021 (weight scaling; hypertube is a *consequence*), Costacurta 2024, Rodriguez-Garcia 2026. Additive: Ferguson & Cardin 2020 Box 1c. Also Vecoven 2020. |

v1 mis-cited Tsuda 2021 as additive; it is multiplicative-on-weights. Ferguson & Cardin Box 1c is the additive anchor.

## 3. The unification claim

A single FiLM layer on the policy's activations is **both stories at once**. $\gamma$ rescales the downstream slope (gain modulation); $\beta$ shifts the operating point. The same scalars re-parameterise effective RL hyperparameters: scaling actor logits is an inverse-temperature change (NA); scaling encoder pre-fusion activations is an ACh-as-learning-rate change (Doya-α, same branch as Lee 2024); shifting the GRU update gate is a memory-stability change. The architecture already injects at all three sites (A encoder, B memory, C policy temperature) — simultaneously a multi-site gain controller and a multi-channel hyperparameter modulator.

**Scalar-to-population bridge.** `mod_h` is the Doya-side scalar — one nucleus, one tone; the FiLM read-out matrix is the receptor-density-analog fan-out. Per-feature $\gamma, \beta \in \mathbb{R}^d$ is the population dimension biology already commits to.

**Differentiation from Rodriguez-Garcia 2026.** Rodriguez-Garcia modulates the **gradient** (optimiser-level — gain reparameterises learning rate as $\lambda \to \lambda / g^2$); this project modulates the **forward activation** (PPO's Adam untouched). Both invoke noradrenergic gain at orthogonal points; the two could combine later.

**What `mod_h` is likely doing today.** Under the single-recurrent-state design, `mod_h` is most likely a Noradrenaline / inverse-temperature signal with two passenger heads.

**Prerequisite — the T/P split.** The unification is non-identifiable without the **Phase 0 T/P split** (two recurrent states with separable phasic and tonic timescales). A single `mod_h` forces correlated movement across knobs biology dissociates. This memo escalates the T/P split from open question to prerequisite.

## 4. What this redirects the project away from

The continual-learning win is one behavioural readout of a hyperparameter shift, not evidence that NMN helps continual learning per se. The closest published precedent is **Lee et al. 2024 §3.1 + Figure 1**, which proposes ACh and NA as modulators of RL hyperparameters in a non-stationary bandit with hand-coded functional forms. The project's contribution is **Lee's framework extended to a multi-site FiLM substrate, with the mapping emerging from optimisation**, in sequential RL (PPO) rather than bandits.

## 5. What the new direction predicts

- **(a) State-dependent exploration shift.** At a regime change the modulator's burst coincides with a policy-entropy spike; effective temperature rises for tens of episodes then relaxes. With the T/P split, this splits into phasic-burst and tonic-shift tests.
- **(b′) Combined falsifier — multiplicative/additive dissociation plus $|\delta|$-tracking.** Freeze $\gamma=1$ at Injection C; measure (i) whether regime-change recovery degrades and the entropy spike vanishes, and (ii) whether the surviving $\beta$ tracks TD-error magnitude $|\delta|$ or only the regime label. Symmetrically freeze $\beta=0$. Three outcomes: $\gamma$-freeze damaging AND $\beta$ tracks $|\delta|$ → unification holds at C. Both freezes equally damaging AND neither tracks $|\delta|$ → modulator is a context detector (Vecoven-style), unification walked back. $\beta$-freeze more damaging → hypertube-shift dominates, gain-control holds.
- **(c) Doya-channel attribution from per-site clamps.** A clamp degrades **learning-rate-sensitive** behaviours (Doya-α); C clamp degrades exploration; B clamp degrades protected-memory tasks.
- **(e) No win where the testbed cannot dissociate the stories.** Steady-state heterogeneous-noise environments need no hyperparameter movement, so the v8 null is expected and converts retroactively into evidence for the unified frame.

## 6. Hand-offs

- **`senior-developer`** — confirm Phase 0 T/P split is on the critical path (now prerequisite); plan $\gamma \geq 0$ / softplus if "gain" is the headline word.
- **`experiment-designer`** — re-derive the probe program around (b′) as falsifier and (c) as channel-attribution probe.
- **`pi`** — portfolio call on whether this is the headline paper vs. the meta-context-conditioning v2 alternative.

## 7. Links

- **Empirical anchor**: [memory insight 20260513_0014](../../../.claude-memory/memories/nmn_diagnosis/20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin.md); [study report](../../experiments/summaries/20260513_0321_nmn_comparison_study.md).
- **Superseded**: [v1](20260516_nmn_scaling_shifting_as_hyperparameter_modulation.md) — carries the full professor reviews (§8) and convergence note (§9) for the audit trail.
- **Reference corpus**: `docs/project/references/neuromodulatory_algorithms/sources/`.
- **Adjacent**: [`nmn_meta_context_conditioning_v2.md`](nmn_meta_context_conditioning_v2.md); [continual-learning probe program](../ideas/20260515_continual_learning_probe_program.md) (re-scoped on landing).
