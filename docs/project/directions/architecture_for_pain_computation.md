---
title: Architecture for Pain Computation — Direction Memo
topic: architecture / FiLM / precision head / factorial ablation
status: draft
author: professor-rl-bayesian-dl
created: 2026-05-07
last_updated: 2026-05-07
closes_gaps: G-B (architecture side), G-C, factorial-table for Claim 3
hand_offs: experiment-designer, senior-developer
---

# Architecture for Pain Computation — Direction Memo

> **One-line summary.** The smallest architectural commitment that defends Claims 1, 2, 3 of the postdoc memo and that escapes the active-inference FP-1 fixed point is **FiLMNoNorm at Injection A + a *decoupled* per-modality precision head trained by Kendall–Gal heteroscedastic reconstruction, with no multiplicative precision gate on $\gamma$**. The current Phase-3 plan in [PRECISION_MODULATION_ARCHITECTURE.md](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md) (the $\hat{\pi}\,\cdot\,\text{FiLM} + (1-\hat{\pi})\,\cdot\,\text{bypass}$ blend at line 138 + line 347) is *more* ambitious than the construct claim requires and introduces an identifiability hazard between $\gamma$ and $\hat{\pi}$ that I recommend the project remove until a cleaner null is in hand. Under this minimum architecture, the factorial × controls × seeds for the headline experiment fits in **≈ 88 runs**; the rung ladder is feasible with the *same* architecture across rungs *only if* the precision head's per-modality grouping is preserved across all rungs (no per-rung head re-tuning).
>
> Anchors: [project_plan.md §4 (the four causes), §G1, §G2, Phase 1, Phase 3](../project_plan.md); [FILM_MODULATION_PLAN.md](../../develop/active/filim/FILM_MODULATION_PLAN.md); [FiLM_ENSEMBLE_SENSORY_PRECISION.md §§2.4, 3.1, 4.4, 6](../../develop/active/filim/FiLM_ENSEMBLE_SENSORY_PRECISION.md); [PRECISION_MODULATION_ARCHITECTURE.md §§2.2–2.4, 3.3, 4.2–4.5](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md); [NMN_PERFORMANCE_DIAGNOSIS_v8.md §§4.1, 5.1](../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md); pain-modeling memo [pain_vs_nociception_construct.md §§2.2, 3, 5](../concepts/pain_vs_nociception_construct.md); Bayesian-brain memo [active_inference_hypervigilance.md §§3.2, 4.1](../concepts/active_inference_hypervigilance.md).

---

## 1. The architectural choices on the table — and the smallest commitment that defends Claims 1–3

### 1.1 What the construct + AI guardians have already pruned

Three things from the prior two memos are now load-bearing constraints, not options:

1. **Heterogeneous, state-dependent noise (Phase 1)** — the AI precondition for non-degeneracy. Without it the optimal $\Pi_s$ is uniform and the v8 null *is* the correct solution ([active_inference_hypervigilance.md §3.1](../concepts/active_inference_hypervigilance.md)).
2. **A per-channel heteroscedastic gradient on the modulator (Phase 3)** — the only practical route to the non-uniform AI fixed point under RL ([active_inference_hypervigilance.md §3.2(b)](../concepts/active_inference_hypervigilance.md)).
3. **Channel selectivity as the load-bearing dissociation metric** — the magnitude $\Delta\gamma_{\mathcal{T}\setminus\mathcal{N}}$, not the post-injury time-locked rise per se, is what dissociates the modulator from C1 (1-D EMA broadcast) ([pain_vs_nociception_construct.md §2.3 FP-1](../concepts/pain_vs_nociception_construct.md); [active_inference_hypervigilance.md §3.2(c)](../concepts/active_inference_hypervigilance.md)).

Read those three together: the architectural job of Phase 3 is *only* to put a per-channel heteroscedastic gradient on the modulator that can express channel-selective $\gamma$. Anything beyond that is luxury. Anything *before* that — including the precision-gating blend — is identifiability hazard.

### 1.2 The four candidates I evaluated

| Candidate | What it is | Verdict |
|---|---|---|
| **(i) Static FiLM (current Phase 2 default)** | $\gamma, \beta$ from $h_{\text{mod}}$, identity-init, no auxiliary loss, post-LayerNorm | Insufficient. v8 documents the failure: $\gamma$ collapses to $\approx 1$, gates bypassed ([NMN_PERFORMANCE_DIAGNOSIS_v8.md §4.1.3](../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md)). FP-1 cause 1: no precision gradient. |
| **(ii) FiLMNoNorm + per-modality heteroscedastic precision head, *decoupled*** | $\gamma, \beta$ from $h_{\text{mod}}$, no LayerNorm at the FiLM call site, separate per-modality $\hat{s}_i = \log \hat{\pi}_i$ head trained by Kendall–Gal NLL on observation reconstruction. **Crucially, $\hat{\pi}$ does *not* gate $\gamma$ multiplicatively.** | **Recommended.** |
| **(iii) Precision-gated FiLM (current Phase-3 plan)** | (ii) plus the blend $\pi_{\text{gate}} \cdot x_{\text{mod}} + (1-\pi_{\text{gate}}) \cdot x_{\text{bypass}}$ at the encoder ([PRECISION_MODULATION_ARCHITECTURE.md §§2.2, 4.3](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md), eq. on lines 138 and 347) | **Stop short of this** until (ii) clears G2 with the AI fingerprint. Rationale in §1.4. |
| **(iv) DreamerV3 heteroscedastic decoder route** | (ii) re-implemented on the world-model decoder; the precision head lives on the RSSM latent ([FiLM_ENSEMBLE_SENSORY_PRECISION.md §6](../../develop/active/filim/FiLM_ENSEMBLE_SENSORY_PRECISION.md), [PRECISION_MODULATION_ARCHITECTURE.md Track B §2.2](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md)) | Cross-check, not the primary route. Defer until PPO route closes G2. |

### 1.3 Why FiLMNoNorm rather than FiLM (with LayerNorm)

The existing FiLM-with-LayerNorm variant places LayerNorm *immediately before* the modulated affine ([FILM_MODULATION_PLAN.md §"Where to Add LayerNorm"](../../develop/active/filim/FILM_MODULATION_PLAN.md), lines 92–104). LayerNorm normalises over the 128 features within each modality before $\gamma$ acts. From the AI fixed-point perspective this is **counterproductive**: per-channel residual variance is exactly the quantity the heteroscedastic precision head needs to *measure*. Pre-normalising it across the 128 hidden units of a modality strips the magnitude information that distinguishes "this modality's prediction error is large" from "this modality's hidden representation has high spread" — which are different statistics.

Concretely, the AI prediction is $\hat{\pi}_i^* = 1/\mathbb{E}[\varepsilon_{i,t}^2]$ on the *observation-space* error ([active_inference_hypervigilance.md §1.2](../concepts/active_inference_hypervigilance.md), with the project's per-modality grouping in [PRECISION_MODULATION_ARCHITECTURE.md §2.3](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md)), so the precision head reads $\varepsilon$ at the *output* of the encoder reconstruction, not at the FiLM call site. LayerNorm at the FiLM site is therefore not in the loss path — it's an additional whitening layer between the modulator's $\gamma$ and the features it should be controlling. v8's MC-FiLM_g16 result (gates collapse to 0.46, std 0.23 — [NMN_PERFORMANCE_DIAGNOSIS_v8.md §4.1.3](../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md)) is consistent with LayerNorm flattening the magnitude variance the modulator was supposed to express.

Two further reasons to drop LayerNorm at the FiLM site:

- **Identity-init under FiLMNoNorm is cleaner.** $\gamma$ init at 1.0 with $\beta$ init at 0.0 gives $\gamma \cdot x + \beta = x$ at init — true pass-through, on raw features. With LayerNorm in front you get $\gamma \cdot \mathrm{LN}(x) + \beta = \mathrm{LN}(x) \neq x$. The v7/v8 diagnosis attributed survival benefit to LayerNorm itself, not to FiLM ([NMN_PERFORMANCE_DIAGNOSIS_v8.md §4.1.2](../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md)). Keeping LayerNorm in the *backbone* (the existing `LayerNorm` baseline path) is fine; what we are removing is its presence *inside* the modulated branch where it whitens the signal $\gamma$ is supposed to weight.
- **FP-1 mitigation.** Under FiLMNoNorm the magnitude carried by $\gamma$ is the same magnitude the modulator can put residual variance against. The optimisation problem is one quantity, not two coupled ones.

The tradeoff is well-known: FiLMNoNorm has worse training stability than FiLM (Perez et al. 2018; [FILM_MODULATION_PLAN.md "Why FiLM Normalization Matters"](../../develop/active/filim/FILM_MODULATION_PLAN.md)). The mitigation, which Phase 3 already implies, is the heteroscedastic loss: the precision head provides a strong, well-shaped gradient on per-channel scale that compensates for FiLMNoNorm's looser magnitude regulation. This is also the *theoretically correct* coupling — the magnitude regularisation is supposed to come from the prior (the data, via the precision head), not from a normalisation operator that erases the prior.

### 1.4 Why I recommend *removing* the precision-gating blend from Phase 3 (for now)

The Phase-3 architecture in [PRECISION_MODULATION_ARCHITECTURE.md §§2.2, 4.3](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md) implements the precision head's $\hat{\pi}_i$ as a *multiplicative gate* on the FiLM-modulated branch, blending against an unmodulated bypass:

$$
x_{\text{out}} \;=\; \pi_{\text{gate}}^{(i)} \cdot \mathrm{ReLU}(\gamma_i \cdot x + \beta_i) \;+\; (1 - \pi_{\text{gate}}^{(i)}) \cdot \mathrm{ReLU}(x), \qquad \pi_{\text{gate}}^{(i)} = \sigma(\log \hat{\pi}_i).
$$

This is architecturally elegant — it implements precision-weighted gain control as a soft mixture of experts — but it creates a **non-identifiability hazard** between $\gamma_i$ and $\hat{\pi}_i$: any product $\pi_{\text{gate}}^{(i)} \cdot \gamma_i$ at fixed $\beta_i$ is achievable by infinitely many $(\pi_{\text{gate}}^{(i)}, \gamma_i)$ combinations. Two pathologies follow:

1. **Channel-selectivity can hide in $\hat{\pi}$ alone**, in $\gamma$ alone, or be split. The pain-modeling memo's central dissociation metric ($\Delta\gamma_{\mathcal{T}\setminus\mathcal{N}} \geq 0.2$, [pain_vs_nociception_construct.md §2.3](../concepts/pain_vs_nociception_construct.md)) is then ambiguous — the team can claim selectivity on the *effective* gain $\pi_{\text{gate}} \cdot \gamma$ even when raw $\gamma$ is flat. Reviewers will see this. The AI-derived selectivity prediction ([active_inference_hypervigilance.md §4.1](../concepts/active_inference_hypervigilance.md), $\Delta\gamma \propto \log(\Sigma_{\mathcal{N}}/\Sigma_{\mathcal{T}})$) is on $\gamma$, not on $\pi \cdot \gamma$.

2. **The C1 dissociation argument weakens.** The Bayesian-brain memo's spine of Claim 2(b) ([active_inference_hypervigilance.md §3.2(c)](../concepts/active_inference_hypervigilance.md)) is that a 1-D EMA broadcast cannot produce channel-selective gain. Under the blend, even C1 can produce channel-selective *effective* gain by routing through $\pi_{\text{gate}}$ if the precision head sees per-channel reconstruction error — because $\pi_{\text{gate}}^{(i)}$ is *itself* per-channel. The "1-D" argument was always about $\gamma$. The blend smuggles per-channel structure back in via $\hat{\pi}$.

The recommendation: **train the precision head as a parallel auxiliary that emits $\hat{\pi}_i$ for *measurement and analysis*, but do not multiplicatively gate $\gamma$ on $\hat{\pi}$ in the first phase**. The precision head's gradient *will* still flow through the shared modulator GRU $h_{\text{mod}}$ to all the FiLM heads — that is the mechanism by which the AI fixed point is reached ([active_inference_hypervigilance.md §3.2(b)](../concepts/active_inference_hypervigilance.md), the heteroscedastic loss as the missing $\partial F/\partial s_i$ gradient). Selectivity emerges in $\gamma$ itself, where it is identifiable, and the C1 dissociation argument is preserved.

If — and only if — that decoupled architecture clears G2 with channel-selective $\gamma$, then it is meaningful to ask whether adding the multiplicative blend further improves survival. Until then, the blend is a confound on the construct claim.

This is the *less than the current plan proposes* note the prompt asked me to flag. I am not recommending the project drop precision gating forever — I am recommending it not be the *first* test, because the first test has to be legible to the construct claim.

### 1.5 The smallest architectural commitment, formally

| Component | Specification | Provenance |
|---|---|---|
| **Injection A — $\gamma, \beta$ heads** | FiLMNoNorm. $\gamma$ unconstrained, init bias $1.0$; $\beta$ unconstrained, init bias $0.0$. No LayerNorm in the modulated branch. | [FILM_MODULATION_PLAN.md "FiLMNoNorm" rows](../../develop/active/filim/FILM_MODULATION_PLAN.md) |
| **Injection B — $z_{\text{memory}}$** | Existing gate-bias on task-GRU update gate. Unchanged. Tied to $h_{\text{mod}}$. | [NEUROMODULATION_ALGORITHM.md H2](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md) |
| **Injection C — temperature (PPO) / reward scale (DreamerV3)** | Existing. Unchanged. Tied to $h_{\text{mod}}$. | [NEUROMODULATION_ALGORITHM.md H3](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md) |
| **Shared modulator GRU** | Existing single GRU producing $h_{\text{mod}}$. Hidden size 16 (current default). | [NEUROMODULATION_ALGORITHM.md H4](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md) |
| **Precision head** | Two-layer MLP on $h_{\text{mod}}$ → (a) reconstructed $\hat{o}_{t+1}$ over `obs_dim`, (b) per-modality $\hat{s}_i = \log \hat{\pi}_i$ (9 outputs). | [PRECISION_MODULATION_ARCHITECTURE.md §2.3](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md) |
| **Auxiliary loss** | Kendall–Gal per-modality NLL: $L_{\text{prec}} = \sum_{i=1}^{9} \tfrac{1}{2} \exp(\hat{s}_i) \cdot \overline{\varepsilon_i^2} - \tfrac{1}{2} \hat{s}_i$, $\overline{\varepsilon_i^2}$ = per-modality MSE on $o_{t+1} - \hat{o}_{t+1}$, normalised by $d_i$. | [PRECISION_MODULATION_ARCHITECTURE.md eq. 4.4](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md) |
| **Loss weight** | $L_{\text{total}} = L_{\text{PPO}} + \lambda_{\text{prec}} \cdot L_{\text{prec}}$, sweep $\lambda \in \{0.1, 0.5, 1.0\}$ small grid. | [PRECISION_MODULATION_ARCHITECTURE.md §6.1](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md) |
| **Coupling** | Gradient flow from $L_{\text{prec}}$ into $h_{\text{mod}}$ → all four FiLM heads. **No multiplicative gate of $\hat{\pi}$ on $\gamma$.** | This memo §1.4 |

Closest published precedent: **Kendall & Gal 2017** for the heteroscedastic head; **Perez et al. 2018** for FiLMNoNorm (their original FiLM, with conditioning network); **Turkoglu et al. 2022 (FiLM-Ensemble)** for the architectural pattern of per-context FiLM modulation paired with predicted variance. The closest thing to what we are proposing is **Kendall & Gal's heteroscedastic auxiliary** routed through a shared recurrent context (the modulator GRU) that *also* drives a static FiLM affine on perception — there is, to my knowledge, no published RL paper that does exactly this. The novelty contribution is precisely "static FiLM driven by a recurrent context whose context is shaped by a per-modality heteroscedastic NLL on observation reconstruction."

### 1.6 What this minimum architecture buys, against the three construct criteria

- **(a) Escapes FP-1.** With Phase-1 heterogeneous, state-dependent noise + the heteroscedastic gradient, the modulator has a non-degenerate $\partial F/\partial s_i \neq 0$ to descend; channel-selective $\gamma$ is the unique optimum. ✓
- **(b) Produces the four-property AI fingerprint** ([active_inference_hypervigilance.md §4.1](../concepts/active_inference_hypervigilance.md)): sign (any, given the loss structure), lag $\le \min(5, 2\tau_{\text{mod}})$ (set by GRU recurrent timescale, 5 is consistent with `mod_hidden_size = 16`), channel selectivity in $\gamma$ (identifiable because $\hat{\pi}$ does not gate $\gamma$), within-trial cross-domain $r$ (forced by shared $h_{\text{mod}}$). ✓
- **(c) Lesionable along A/B/C × shared-core factorial without re-tuning.** Each injection's parameters are independent; lesioning any one ablates a single forward path. Shared-core lesion is replacing $h_{\text{mod}}$ with three independent per-injection GRUs of comparable hidden size — also surgically scoped. **No hyperparameter retuning is required** because the precision head's loss is local to its own gradient, and removing $\gamma$ leaves $\beta$ + $z_{\text{memory}}$ + temperature paths untouched. ✓

---

## 2. Factorial ablation for Claim 3, with C0/C1/C2 controls — the full run budget

The factorial × controls × seeds is the budget the paper has to defend. I give numbers.

### 2.1 Cells of the experiment

| Family | Cells | Count |
|---|---|---|
| **Injection-site lesion factorial** (the H4 / Claim 3 spine) | $\{\}, \{A\}, \{B\}, \{C\}, \{A,B\}, \{A,C\}, \{B,C\}, \{A,B,C\}$ — i.e., the $2^3 = 8$ subsets of injections, with $\{\}$ being all three on (the modulated agent) | 8 |
| **Shared-core lesion** | Same architecture but $h_{\text{mod}}$ split into three independent per-injection GRUs of equal hidden size (8 each, sum = current 16+) — H4 negative test | 1 |
| **Construct controls (pain-modeling)** | C0 (reflex baseline; no modulator), C1 (1-D trainable EMA broadcast — see §4 below), C2 (full modulator, interoceptive channels masked in modulator input) | 3 |
| **Total cells per environment rung × λ_prec setting** | | **12** |

### 2.2 Seeds, λ\_prec, rungs

- **Seeds.** The pain-modeling memo's within-trial $r \ge 0.3$ and $\Delta\gamma \ge 0.2$ thresholds, plus the AI memo's selectivity contrast (C1 dissociation) and timescale ordering ([active_inference_hypervigilance.md §4](../concepts/active_inference_hypervigilance.md)), need a *seed-level effect-size estimate* with paired-design power. With Cohen $d \ge 0.5$ at $\alpha = 0.05$, two-sided, the conventional sample size is ~5 seeds per cell for paired comparisons. **5 seeds per cell** is the floor; **8 seeds** gives clean bootstrap CIs for the cross-correlation $r$. I recommend **8 seeds for the headline cell (full modulator) and 5 for every other cell**, on the canonical rung. Total: $11 \cdot 5 + 1 \cdot 8 = 63$ runs at the headline rung.
- **$\lambda_{\text{prec}}$ sweep.** $\{0.1, 0.5, 1.0\}$ on the *full modulator only*, on the canonical rung. 3 cells × 5 seeds = 15 runs; the chosen $\lambda$ is then locked for the rest of the factorial. Folded in below.
- **Rungs.** §3 below argues for 3 rungs on the construct-mandatory path. The factorial runs only on the **canonical rung (rung 3)**; the other rungs run *only the headline cell + C0 + C1* (3 cells × 3 rungs × 5 seeds = 45 runs, cross-rung).

### 2.3 The total

| Component | Cells × seeds | Runs |
|---|---|---|
| $\lambda_{\text{prec}}$ sweep on canonical rung (full modulator) | 3 × 5 | 15 |
| Factorial + shared-core + C0/C1/C2 on canonical rung at locked $\lambda$ | 11 × 5 + 1 × 8 | 63 |
| Rung-2 and rung-5 transfer (headline + C0 + C1 only) | 3 × 2 × 5 | 30 |
| **Total headline budget** | | **108 runs** |

The 108 number assumes 5 seeds for everything but the headline modulator. If the user can afford 8 seeds across the canonical-rung factorial, the number rises to ~150. If the user is willing to ship with 3 seeds (publishable but at the lower edge — bootstrap CIs widen), it falls to ~70. I'd recommend **108 as the target**.

### 2.4 Where the budget bites

- **The $r \ge 0.3$ threshold is the most seed-hungry metric.** Within-trial cross-correlation of three signals around event windows has high per-seed variance; the bootstrap CI on the difference between the modulated agent and C2 (the AI memo's pre-registrable contrast, [active_inference_hypervigilance.md §4.1](../concepts/active_inference_hypervigilance.md)) needs ≥ 8 seeds for a meaningful margin call. *This is where the 8-seed splurge on the headline cell pays.*
- **Factorial doubles → triples the runs.** The double-lesion cells $\{A,B\}, \{A,C\}, \{B,C\}$ are construct-essential (they are what test "the *combination* of injections is more than the sum") and should not be cut.
- **Cross-rung budget is dominated by transfer testing.** If the user can accept "platform demo" instead of "platform validation," the cross-rung budget can be cut to rung 1 + rung 5 (just the chronicity rung) and still defend Claim 5 in skeleton form. Saves 15 runs.
- **GAE vs MC.** v8 ([NMN_PERFORMANCE_DIAGNOSIS_v8.md §4.1.2](../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md)) shows GAE is *actively destabilised* by the modulator under noise. Run the headline factorial in **MC only**; report GAE as supplementary for one or two cells. This is already common in the diagnosis series and saves ~50% of the budget vs. running both.

### 2.5 What this buys

The 108-run budget supports:

- **Claim 3 (factorial).** All 8 lesion subsets + shared-core, with effect-size estimates on $\Delta\gamma_{\mathcal{T}\setminus\mathcal{N}}$, on the cross-domain $r$, and on survival. Reviewer-defensible.
- **Claim 1 (nociception ≠ pain).** C0 vs full modulator on survival and on the four-property fingerprint. C2 dissociates "architecture only" from "interoception-conditioned modulation."
- **Claim 2(b) (clinical signature).** The C1 dissociation on channel selectivity — the spine of the Bayesian-brain memo's Claim 2(b) ([active_inference_hypervigilance.md §3.2(c)](../concepts/active_inference_hypervigilance.md)).
- **Claim 5 (platform).** Cross-rung transfer of the headline cell + C0 + C1 on rungs 1 and 5 (or 1, 3, 5). Demonstrates the architecture is platform-stable.

What it does *not* buy without further runs: the regional-contrast experiment for AI-vs-heteroscedastic-BDL ([active_inference_hypervigilance.md §2.3](../concepts/active_inference_hypervigilance.md)), or the gap-G-G modulator-state-freeze transfer probe at full strength (which I treat in §3 below).

---

## 3. The rung ladder under fixed architecture — feasibility and minimum set

Claim 5 ("the platform is the contribution") survives only if the architecture in §1.5 trains successfully across rungs *without per-rung surgery*. I assess feasibility component by component.

### 3.1 Which architectural components are rung-stable

| Component | Sensitivity to rung change | Verdict |
|---|---|---|
| FiLMNoNorm ($\gamma, \beta$ heads) | Insensitive — the FiLM dimension is set by encoder hidden size, which is rung-stable as long as the obs vector's modality structure is preserved. | Stable. |
| Shared modulator GRU | Insensitive — GRU hidden size 16 is independent of grid size, predator count, etc. | Stable. |
| **Per-modality precision head** | **Sensitive to the modality count.** If a rung changes the obs vector's modality structure (e.g., adds a new sensor at rung 4), the head's 9 outputs must change. *This is the single retuning hazard.* | **Stable iff the rung ladder does not introduce new modalities.** See §3.3. |
| Heteroscedastic loss $\lambda_{\text{prec}}$ | Mildly sensitive — the relative magnitude of $L_{\text{prec}}$ vs $L_{\text{PPO}}$ changes with the survival distribution, which changes with rung. | Mildly sensitive; locking $\lambda$ from rung 3 to other rungs is a defensible commitment but should be checked. |

### 3.2 Minimum rung set

Ranking by what each rung is necessary for:

| Rung | Description | Construct-mandatory? | Why |
|---|---|---|---|
| **Rung 1** | Empty grid + injury, no predators, no food | **Yes** | Sets the C0/reflex floor for Claim 1; tests pure pain-following-injury reflex without confound from drive conflict. Pain-modeling memo §5.1 condition (i). |
| Rung 2 | + food, drive conflict | Optional | Tests FP-3 (reward-shaping equivalent); valuable but not load-bearing. |
| **Rung 3** | + predators with heterogeneous threat profile, **+ heterogeneous noise (canonical preset)** | **Yes** | The canonical experimental setting where AI fingerprint and channel selectivity are tested. Pain-modeling memo §5.1 condition (ii); AI memo §3.2(a). This is where the 63-run factorial happens. |
| Rung 4 | + heterogeneous sensory noise scaled differently | Optional | Already a parameter sweep within rung 3; promoting to a separate rung is methodology-paper flavour, not a construct requirement. |
| **Rung 5** | + slow injury dynamics (long recovery window) | **Yes** | The H5 chronicity test. The modulator-timescale ordering $\tau_A < \tau_B < \tau_C$ ([active_inference_hypervigilance.md §4.2](../concepts/active_inference_hypervigilance.md)) is only legible with a recovery window $\gg$ episode length. Pain-modeling memo §5.1 condition (iv). |

**Recommended minimum rung set: {1, 3, 5}.** This matches the pain-modeling memo's §5.1 verdict. Rungs 2 and 4 enrich the methodology figure but are not on the construct-claim critical path. The cross-rung budget in §2.3 above (rungs 2 and 5 in addition to canonical rung 3, 30 runs) should be re-allocated as **rungs 1 and 5 plus the headline rung 3**, which is the same 30-run budget but better targeted.

### 3.3 The per-rung architecture re-tuning question (gap G-G strict version)

The Bayesian-brain memo's gap-G-G (modulator-state-freeze transfer probe across rungs, [active_inference_hypervigilance.md §5](../concepts/active_inference_hypervigilance.md), level 3) requires the modulator's representation to *generalise* across rungs without surgery. Under §1.5's architecture, the strict feasibility test is:

- **Same encoder dim → same FiLM tensor shape → same precision head output dim.** Stable as long as the obs vector's modality structure is preserved across rungs. The 9 modalities (`injury, nutrition, satiation, extero_noc, olfaction, collision, proprioception, visual, location` — [PRECISION_MODULATION_ARCHITECTURE.md §2.3 table](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md)) are present at rungs 1, 3, 5 by construction; rungs 2 and 4 also preserve them. **Verdict: feasible.**
- **Same per-modality breakdown for the heteroscedastic loss.** Stable for the same reason.
- **One free hyperparameter that legitimately varies across rungs:** $\lambda_{\text{prec}}$. The team should report whether locking $\lambda_{\text{prec}}$ at the rung-3 value gives transfer; if not, the *amount* of $\lambda$ rescaling required is itself a finding (and a small one, defensible). Re-tuning *only* $\lambda_{\text{prec}}$ does not compromise the platform claim; re-tuning the head architecture would.

**Marginal training cost per rung under §1.5 architecture.** Each rung run is approximately the same wall-clock cost as the current Phase-2 baseline (no architectural change increases compute beyond ~0.5% — the precision head's two-layer MLP is dwarfed by the encoder/GRU). The cost is dominated by environment generation and seed count, not by architecture. **Marginal architecture cost per rung ≈ 0.** This is what makes the platform claim defensible.

---

## 4. C1 architectural specification — making the construct guardian's "richer nociception" baseline real

The pain-modeling memo proposes C1 as "a trainable EMA of nociception wired into the same A/B/C injection sites" ([pain_vs_nociception_construct.md §3](../concepts/pain_vs_nociception_construct.md)). The Bayesian-brain memo argues C1 cannot in principle produce channel-selective $\gamma$ ([active_inference_hypervigilance.md §3.2(c)](../concepts/active_inference_hypervigilance.md)). A weak C1 makes that dissociation cheap; a strong C1 is the construct test the paper needs. Here is the strongest C1 the construct allows.

### 4.1 Parameterisation

Replace the modulator GRU with a single trainable scalar EMA on the nociceptive channel:

$$
m_t \;=\; (1 - \alpha) \cdot m_{t-1} \;+\; \alpha \cdot o_{\text{noc}, t}, \qquad \alpha = \sigma(\theta_\alpha) \in (0, 1),
$$

where $\theta_\alpha$ is a learned scalar parameter (sigmoid-gated to keep $\alpha \in (0,1)$). At init, $\sigma(\theta_\alpha) \approx 0.1$ to give a meaningful temporal envelope; the optimisation can move it.

This is the strongest 1-D temporal envelope of nociception possible under the "richer nociception" framing — the decay rate is learned, not fixed, and it sees only $o_{\text{noc}}$, not the rest of the obs vector. Adding more inputs (e.g. injury) violates the "1-D EMA of nociception" definition; making the dynamics non-linear (e.g. a 1-D RNN) makes it not an EMA and weakens the dissociation argument.

### 4.2 Broadcast pattern at A/B/C

To make C1 the *strongest* possible 1-D mimicry, broadcast $m_t$ through the same FiLM heads used by the modulated agent:

| Injection | Broadcast | Strongest C1 form |
|---|---|---|
| **A** | $\gamma_i = w^A_{\gamma,i} \cdot m_t + b^A_{\gamma,i}$, $\beta_i = w^A_{\beta,i} \cdot m_t + b^A_{\beta,i}$, with $w^A, b^A$ learned vectors of shape (encoder hidden size,) | Per-channel *intercept* $b^A_{\gamma,i}$ allows static channel ranking; per-channel *slope* $w^A_{\gamma,i}$ allows static *direction* of channel ranking change. **But the *time-varying* component of $\gamma_i(t) - b^A_{\gamma,i}$ is a constant scalar across channels, $w^A_{\gamma,i} \cdot m_t$.** This is the algebraic content of "1-D broadcast." |
| **B** | $z_{\text{memory},t} = w^B \cdot m_t + b^B$ | Same envelope, single-dim. |
| **C** | $T_{\pi,t} = w^C \cdot m_t + b^C$ (clipped as in current modulator) | Same envelope, single-dim. |

Crucially, the per-channel *post-injury time course* of $\gamma_i(t)$ in C1 is forced to be a scalar multiple of $m_t$ across channels. **Two channels cannot rise at different rates** under C1 — only at different *amounts*. This is what the AI memo's §3.2(c) argument rules out.

### 4.3 Hyperparameters that maximise C1 strength

To ensure the dissociation-vs-modulator is meaningful and not a strawman:

- **Learnable $\alpha$**, init at $\sigma^{-1}(0.1)$ → trainable from very-fast to very-slow. Don't fix it.
- **All FiLM head weights $(w^A, b^A, w^B, b^B, w^C, b^C)$ trainable end-to-end through the RL loss + the same heteroscedastic auxiliary $L_{\text{prec}}$ used by the modulated agent, with the same $\lambda_{\text{prec}}$ schedule.** The C1 control should benefit from the precision auxiliary as much as the modulator; otherwise the dissociation could be attributed to "C1 has no auxiliary loss," which is not the intended ablation.
- **Same modulator hidden size budget.** The modulator GRU has ~$16 \cdot 32$ params (input + recurrent + output projections); C1 should match this with extra capacity in the FiLM heads (more learned $w, b$ vectors, deeper read-out from $m_t$ to $\gamma$ if necessary). The intent is "C1 is not handicapped on parameter count."
- **Same seed count** as the modulated agent (8 seeds, headline cell). Don't undersample C1.

### 4.4 What C1 can and cannot reproduce, by construction

Per the Bayesian-brain memo's argument, this strongest-C1 *can* reproduce:

- The temporal envelope of post-injury $\gamma_{\mathcal{T}}$ rise and fall (because $m_t$ has the right timescale).
- The within-trial zero-lag correlation across A/B/C (because all three are scalar functions of the same $m_t$ — in fact, $r$ should be 1.0 modulo the nonlinearities, *higher* than the modulated agent's). **This actually predicts that the cross-domain correlation alone is *not* a discriminating metric.** That's the Bayesian-brain memo's point: the AI fingerprint requires *channel selectivity*, which C1 cannot fake.
- A monotone post-injury decline in survival relative to C0 in the right direction.

C1 *cannot* reproduce:

- $\gamma_{\mathcal{T}}(t) - b^A_{\gamma, \mathcal{T}}$ rising at a different *rate* from $\gamma_{\mathcal{N}}(t) - b^A_{\gamma, \mathcal{N}}$. The post-injury *change* is forced to be a scalar broadcast.
- Channel-selective response to channel-specific reliability shifts. If injury raises olfaction noise but not visual noise, C1 cannot rebalance $\gamma_{\text{olf}}$ vs $\gamma_{\text{vis}}$ because $m_t$ does not see channel-specific prediction error.

The dissociation metric is therefore $\Delta_t \gamma_{\mathcal{T}}^{(i)} - \Delta_t \gamma_{\mathcal{T}}^{(j)}$ for two threat channels with *different* injury-scaled noise profiles ($\alpha = 2.0$ for olfaction, $\alpha = 3.0$ for visual — [PRECISION_MODULATION_ARCHITECTURE.md §2.3 table](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md)). The modulator can produce a non-zero difference; C1 cannot. This is the precise pre-registrable metric for the C1 dissociation.

---

## 5. The AI-restated G1 from an RL/BDL perspective — and a cheaper diagnostic

The Bayesian-brain memo restates G1 as: "$G1$ holds iff the AI fixed point $s^*$ under the canonical preset has a non-trivial channel ranking, $\max_i s_i^* - \min_i s_i^* > \tau$" measured from per-channel residual variances of an unmodulated LayerNorm baseline ([active_inference_hypervigilance.md §3.2(a)](../concepts/active_inference_hypervigilance.md)).

### 5.1 Is per-channel residual-variance non-degeneracy the right measurable property?

Yes, with two refinements.

**It is the right *target* property.** Per-channel residual variance is exactly the empirical quantity whose inverse is $\hat{\pi}_i^*$. If it is degenerate (uniform), the AI fixed point is degenerate, and Phase 2/3 are guaranteed to produce the v8 null *regardless of architecture*. The Bayesian-brain memo's restatement is theoretically clean.

**The two refinements an RL/BDL practitioner would add:**

1. **The residual variance has to be measured on the *task* (an RL agent's observation reconstruction), not on a stationary distribution.** A baseline LayerNorm agent's encoder does not reconstruct $o$; it produces task-relevant features. The "per-channel residual variance" the AI fixed point cares about is on a reconstruction head's prediction error, not on raw $o$. To measure it requires *running the precision head architecturally*. So the Bayesian-brain memo's "no architecture changes" is slightly optimistic — the head needs to exist, even if its gradient is not coupled to the encoder.
2. **State-conditional, not marginal.** The non-degeneracy that matters is *under high injury*, not averaged over the episode. The full check is: $\max_i s_i^*(\text{injury} = \text{high}) - \min_i s_i^*(\text{injury} = \text{high}) > \tau$. Under uniform per-channel noise this is automatic from the Phase-1 noise preset's structure, but the metric the team should log is the conditional one.

### 5.2 The cheaper diagnostic

The cheapest possible architectural variant of the AI G1 check, which I recommend:

- **Run the LayerNorm baseline as currently configured** (no modulator).
- **Attach a *frozen-weights* reconstruction head** ($h_{\text{task}} \to \hat{o}_{t+1}$, two-layer MLP, *not* trained — random-init, or trained on a small held-out window with the encoder frozen). The head is a *measurement device*, not a trained component. It does not feed into the loss.
- **Compute per-channel MSE on $o_{t+1} - \hat{o}_{t+1}$, conditioned on `injury > 0.5` and `injury < 0.1`.** Take the per-channel ratio.
- **G1 (AI version) holds iff** $\max_i \mathrm{MSE}_i(\text{high inj}) / \min_i \mathrm{MSE}_i(\text{high inj}) > \tau$ for some $\tau \ge 2$ (a 2× ratio is a loose lower bound for $\Delta s^* \ge \log 2 \approx 0.7$). The 2× threshold can be replaced with the AI memo's theory-derived $\Delta\gamma \propto \log(\Sigma_{\mathcal{N}}/\Sigma_{\mathcal{T}})$ ([active_inference_hypervigilance.md §4.1](../concepts/active_inference_hypervigilance.md)).

This is a single measurement, one run per Phase-1 noise candidate, reusing the LayerNorm baseline's already-trained encoder. No additional training. No coupled gradient. **Cheaper than the full Phase-3 precision head and gives the same go/no-go information about whether the noise preset has the AI-required heterogeneity.**

The team should run this *before* Phase 2 finishes, and ideally as part of the Phase-1 noise sweep. It will tell them whether any of the candidate noise presets actually produce a non-degenerate AI fixed point. If none of them do, the noise preset has to be retuned regardless of what the modulator does.

**Recommendation.** Add this measurement protocol to the experiment-designer's Phase-1 deliverable. It is a single seed × per-noise-preset addendum, costs ≈ 1 wallclock-hour per preset, and resolves the Bayesian-brain memo's §3.2(a) precondition without spending a full architectural run. The senior-developer hand-off is small: a frozen reconstruction head and a per-modality MSE logger.

---

## 6. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| FiLMNoNorm training instability without LayerNorm in the modulated branch | Medium | Heteroscedastic loss provides magnitude regularisation; if instability persists, fall back to LayerNorm in the *backbone* (current encoder LayerNorm path) but keep the modulated branch unnormalised. |
| Precision head training $L_{\text{prec}}$ overwhelms PPO gradient at large $\lambda$ | Medium | Start $\lambda = 0.1$; sweep up only as long as PPO survival remains above the v8 baseline. |
| C1 baseline trained jointly with $L_{\text{prec}}$ might inadvertently produce per-channel structure via $h_{\text{mod}}$-equivalent feedback | Low–Med | C1 has no $h_{\text{mod}}$; the precision head reads only $m_t$ in C1. Verify in the audit pass: check that C1's precision head input dim = 1, not 16. |
| Cross-rung transfer fails due to $\lambda_{\text{prec}}$ rescaling needs | Low | Defensible report; small per-rung $\lambda$ adjust does not compromise the platform claim, only the maximally-strong cross-rung-no-tuning sub-claim. |
| Reviewer challenges the FiLMNoNorm choice as cherry-picking the variant that worked | Low | The Phase-2 plan is to characterise *all* relevant FiLM variants ([project_plan.md Phase 2](../project_plan.md#phase-2--reproduce-the-null-then-probe-the-film-variants)). Run FiLM, FiLMNoNorm, PreActivation, Multiplicative on the canonical noise preset; report the variant chosen for Phase 3 with the diagnostic that justified it. |
| The "remove precision-gating blend for now" recommendation is rejected by the user | Low | The recommendation in §1.4 is testable: include a single "with-blend" cell at the headline factorial run (1 cell × 5 seeds = 5 extra runs). If the blend dominates, reverse the recommendation. The construct dissociation argument is what is at stake, not the architectural future. |

---

## 7. Summary of the architectural commitment

| Question | Answer |
|---|---|
| Smallest architectural commitment defending Claims 1–3? | **FiLMNoNorm at Injection A + decoupled per-modality heteroscedastic precision head + existing $h_{\text{mod}}$ shared GRU**. No precision-gating blend (yet). |
| Does it escape FP-1? | Yes, conditional on Phase-1 heterogeneous, state-dependent noise (the AI memo's necessary condition). |
| Does it produce the four-property AI fingerprint? | Yes, by construction — $\hat{\pi}$ does not gate $\gamma$, so channel selectivity is identifiable on $\gamma$ directly. |
| Is it lesionable along A/B/C × shared-core without re-tuning? | Yes — $\lambda_{\text{prec}}$ is the only rung-level free knob. |
| Total run budget for the headline experiment? | **~108 runs** (factorial + controls + λ sweep on canonical rung + cross-rung transfer to rungs 1 and 5). |
| Minimum rung set? | **{1, 3, 5}**. Matches pain-modeling memo §5.1. |
| C1 architecture? | Trainable scalar EMA on $o_{\text{noc}}$, broadcast through per-injection learned linear read-outs $w \cdot m_t + b$; same auxiliary loss; same seed budget. The strongest 1-D mimic that the construct definition allows. |
| AI-restated G1 — operational form? | Per-channel residual MSE ratio on a frozen-head LayerNorm baseline, conditioned on high-injury vs. low-injury. Costs ≈ 1 hour per noise preset; resolves the AI precondition before Phase 2. |
| One thing the user should change about the plan, given this audit? | **Decouple the precision head from the FiLM gate (drop the `pi_gate * x_mod + (1-pi_gate) * x_bypass` blend) for the first Phase-3 run.** This makes channel selectivity identifiable on $\gamma$, which is what the construct dissociation needs. The blend can be added back as a follow-up if the simpler architecture clears G2. |

---

## 8. Next steps (hand-offs)

1. **`experiment-designer`** — translate §2 (run budget) and §3.2 (rung set {1, 3, 5}) into the Launch Manifest. Pre-register the C1 dissociation metric of §4.4 (cross-channel rate-of-rise difference $\Delta_t \gamma_{\mathcal{T}}^{(i)} - \Delta_t \gamma_{\mathcal{T}}^{(j)}$) and the AI memo's five pre-registrable quantities ([active_inference_hypervigilance.md §4.3](../concepts/active_inference_hypervigilance.md)). Add the §5.2 cheap diagnostic to the Phase-1 noise sweep deliverable.
2. **`senior-developer`** — small architectural plan covering: (a) FiLMNoNorm activation in the existing encoder ([FILM_MODULATION_PLAN.md "FiLMNoNorm" branch](../../develop/active/filim/FILM_MODULATION_PLAN.md), already implemented), (b) decoupled per-modality precision head as a separate `nnx.Module` with its own NLL on the obs reconstruction, *no multiplicative gate on $\gamma$*, (c) C1 as a new `modulation.type = "EMAControl"` with a 1-D scalar state and per-injection linear read-outs, (d) frozen-head reconstruction logger for §5.2. None of (a)–(d) is invasive; (b) and (c) are new modules but contained.
3. **`professor-neuromodulation`** (in progress, parallel to this memo) — should comment on whether the timescale ordering $\tau_A < \tau_B < \tau_C$ in [active_inference_hypervigilance.md §4.2](../concepts/active_inference_hypervigilance.md) is biologically plausible under the single-GRU $h_{\text{mod}}$, or requires a multi-timescale modulator (which would be a *bigger* architectural commitment than §1.5 — relevant if the H5 chronicity claim demands it).
4. **Postdoc** — cross-professor synthesis after the neuromodulation memo lands. Resolves whether the §1.5 minimum is sufficient for level 2 ("AI hypervigilance") or whether a multi-timescale extension is needed for level 3 ("pain computation" with cross-rung transfer).
5. **User** — the decision pending: lock the §1.5 architecture as the Phase-3 starting point, or run the §1.4 "with-blend" cell in parallel (5 extra runs, +5%). I recommend the former.

---

## 9. References (papers anchoring §§1–4)

Already in the project's reference set or flagged for addition.

- Perez, E., Strub, F., De Vries, H., Dumoulin, V., Courville, A. (2018). "FiLM: Visual Reasoning with a General Conditioning Layer." *AAAI.* — origin of FiLM, FiLM-without-norm tradeoffs.
- Kendall, A., Gal, Y. (2017). "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" *NeurIPS.* — heteroscedastic NLL formulation; the §1.5 auxiliary.
- Turkoglu, M.O. et al. (2022). "FiLM-Ensemble: Probabilistic Deep Learning via Feature-wise Linear Modulation." *NeurIPS.* — closest published precedent for per-context FiLM with predicted variance.
- Friston, K. (2023). "Computational psychiatry: from synapses to sentience." *Phil. Trans. B.* — precision dynamics $\dot{\Pi}_s \propto \beta - \varepsilon_s^2$; the gradient form of which is the Kendall–Gal NLL.
- Vecoven, N. et al. (2020). "Introducing neuromodulation in deep neural networks to learn adaptive behaviours." *PLoS ONE.* — the NMN architecture this project extends.
- Hafner, D. et al. (2023). "Mastering Diverse Domains through World Models" (DreamerV3). — Track-B substrate for §1.2 candidate (iv).
- Costacurta, J. et al. (2024). "Structured flexibility in recurrent networks." — rank-1 motif scaling, theoretical context for FiLM as low-rank modulation.
