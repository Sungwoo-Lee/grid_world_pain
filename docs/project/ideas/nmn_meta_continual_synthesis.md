# Synthesis — NMN architecture-vitality probe (meta + continual)

**Author**: research-postdoc
**Date**: 2026-05-09
**Mode**: Cross-professor synthesis (Mode 3)
**Inputs**:
- Triage: [`docs/project/triage/20260509_1517_nmn_meta_continual_pivot.md`](../triage/20260509_1517_nmn_meta_continual_pivot.md)
- prof-RL/BDL: [`docs/project/directions/nmn_meta_context_conditioning.md`](../directions/nmn_meta_context_conditioning.md) (Candidate A1, gated on a measurement-only ΔCKA probe)
- prof-Neuromod: [`docs/project/directions/nmn_continual_lifelong_probe.md`](../directions/nmn_continual_lifelong_probe.md) (Candidate D, modified to 5-stage double-return; run now)

**Project anchors**:
- `project_plan.md` §4 (four causes of v8 null), §5 phase ordering (Phase 0 engineering preconditions → Phase 1 noise reshape → Phase 2 FiLM probe → Phase 3 precision/T-P split → Phase 4 hypervigilance readout); the meta/continual probe inserts as a **Phase 0.5 architecture-vitality check** that gates Phase 3, not as a replacement for it.
- `NMN_ARCHITECTURE_REVIEW.md` §2.1–§2.5 (architecture contract: branched, recurrent, separate, grouped + per-modality-broadcast gates).
- This week's NMN summary: [`20260509_1421_nmn_comparison_study.md`](../../experiments/summaries/20260509_1421_nmn_comparison_study.md). Headline: even with temperature ceiling unblocked and noise heterogenised, the modulated agent does not beat the unmodulated baseline on survival; the bottleneck is *downstream* of the temperature head; ±4–5 step seed noise rules out single-seed comparisons going forward.

---

## 1. What each professor contributed

**prof-RL/BDL (memo §2–§6).** Frames the architecture-vitality question as two **necessary properties** the modulator must satisfy for any (b)-frame meta-RL win:
- **P1 — context-distinguishability of $h^{\text{mod}}$**: $\Delta_{\text{CKA}} \triangleq \mathrm{CKA}(H^{\text{mod}}_A, H^{\text{mod}}_A) - \mathrm{CKA}(H^{\text{mod}}_A, H^{\text{mod}}_B) \geq 0.05$.
- **P2 — alignment of branched-head injection direction with the encoder's context-discriminative subspace**: a strictly stronger condition than NPN's, because the project's heads are *separate-recurrent* + grouped + spatially-broadcast (review §2.4.1), unlike NPN's intra-layer feedforward gates.
- **Decision tool**: the three-way table (CKA × clamp × survival) — survival alone is uninterpretable. Specifically, (CKA-positive, clamp-positive, survival-null) is *redundancy*, not failure, and motivates Phase 0.
- **Most-load-bearing claim**: $\Delta_{\text{CKA}}$ is computable **now, on existing v8 logged trajectories**, with no new training. If $\Delta_{\text{CKA}} < 0.05$ on v8, P1 already fails on the current modulator family, and Candidate A1 is predicted-null in advance.

**prof-Neuromod (memo §1–§5).** Frames the architecture-vitality question as "can the current single-GRU modulator implement the two published continual-RL mechanisms":
- **Rodriguez-Garcia 2026 stability-gap attenuation** (Hessian flattening via gain $g$, $\lambda \to \lambda/g^2$): the project's `z_unimodal` (multiplicative mode, bounded by $\sigma(\cdot) \in (0,1)$) and `temperature` head together expose this primitive — but with magnitude bounded above by 1, roughly half the available range an unbounded $g \geq 1$ would have.
- **Lee 2024 Doya-DaYu boundary-detection** (epistemic/aleatoric ratio $\alpha = E/(E+A)$): the current modulator **cannot** natively compute this — there is no ensemble for $E$, no aleatoric teaching signal for $A$. Phase 0's precision head is the substrate-level minimum that recovers the Lee mechanism.
- **Most-load-bearing claim**: a null on Candidate D under the current architecture has **two readable explanations** — single-GRU timescale collapse (→ T/P split) and missing aleatoric teaching signal (→ precision head). A null *defines what Phase 0 must repair* rather than indicting the architecture wholesale; a positive *vindicates* the floor pre-Phase-0 and saves 2–3 weeks of engineering. **Run now beats block.**
- **Schedule modification**: replace the triage's 3-stage active→passive→active with a **5-stage abrupt-double-return** (1→2→1→2→1). The double-return discriminates monotone-forgetting (unmodulated baseline) from non-monotone-with-higher-second-return (Tsuda hypertube signature; review §K.4) — the cleanest possible architecture-vitality signal at zero engineering cost. Optional gradual-vs-abrupt arm contingent on a small developer interpolator touch.

## 2. Where the two memos converge (and the project should treat as load-bearing)

1. **Both treat the current pre-Phase-0 architecture as worth probing despite the v8 null.** Neither professor argues to block on Phase 0 first as the default. RL/BDL gates on a no-cost analyzer probe; Neuromod runs the schedule directly. Neither says "engineer first, measure later."
2. **Both name a *mechanism check* alongside the survival number.** RL/BDL's clamp-ablation (P2 falsifier) and Neuromod's boundary-locked Mahalanobis distance + $\sigma(z_{\text{uni}})$ / $\tau_\pi$ transient. Survival alone is uninterpretable to either of them; this is the same epistemic stance, applied to two different experiments.
3. **Both identify the same Phase 0 levers as the most load-bearing repairs** if their probe nulls: T/P split (timescale-separation; project_plan §3.2) and precision head (aleatoric teaching signal; project_plan §3.4). Neither mentions opioid head (§3.3) — that is reserved for the Phase 4 hypervigilance/H5 question, not the architecture-vitality question.
4. **Both treat $\Delta_{\text{CKA}}$ / $h^{\text{mod}}$-trace as the canonical "is the modulator doing context work" diagnostic.** RL/BDL uses it on existing trajectories pre-experiment; Neuromod uses it on the boundary transient post-experiment. The same probe answers two distinct questions; nothing prevents both being computed.

## 3. Where they diverge (the user must choose)

1. **Compute timing.** RL/BDL **gates** on a free analyzer-only probe before any new training; Neuromod **launches** Candidate D in parallel because (a) it's config-only on already-shipped infrastructure, (b) ~36–48 cell-hr (one overnight on 4–6 nodes), and (c) a null is informative. The two are *not* contradictory — they target different experiments — but they imply different orderings under finite compute.
2. **What constitutes "the cleanest test."** RL/BDL: the task GRU's 128-dim hidden state could already carry context, so the cleanest contrast (A1 active vs. passive predator, where the discriminator is *temporal*) is precisely the one that resists redundancy with the GRU baseline. Neuromod: the cleanest contrast is the *double-return* schedule, because it discriminates plain interference from Tsuda-hypertube reusable subnetworks. These are orthogonal questions on the same architecture.
3. **What a null *means*.** RL/BDL: a CKA-null on existing v8 data is a hard stop — escalate to senior-developer for Phase 0 engineering. Neuromod: a survival-null on Candidate D *defines* what Phase 0 fixes (single-GRU timescale collapse vs. missing aleatoric signal); not a stop. The user's appetite for "the diagnostic null" vs. "stop-and-engineer" is the core divergence.

## 4. Reconciled recommendation: a single sequence

The two professors' experiments are **not in tension** — they answer different sub-questions of "is the architecture broken or is the v8 question hard?" RL/BDL answers "does the modulator build context-distinguishable state at all"; Neuromod answers "does the modulator's gain-form intervention buy a continual-RL-shaped win." A project-rational ordering uses RL/BDL's no-cost gate to *triage* before spending compute, but does not let it block Neuromod's overnight run if compute is available tonight:

**Step 1 (no compute, ~30 min–few hours).** `experiment-analyzer` runs RL/BDL §6 measurement-only ΔCKA probe on existing v8 logged rollouts. Two outcomes:
- **Step 1a — ΔCKA ≥ 0.05.** P1 holds on current modulators. Both downstream experiments remain on the table.
- **Step 1b — ΔCKA < 0.05.** P1 fails on current modulators. Predicts Candidate A1 will null in advance. Neuromod's Candidate D *still runs* — its mechanism (Hessian-flattening via gain) is not strictly downstream of P1, and its null disambiguates which Phase 0 lever (T/P split vs. precision head) is binding. But the *interpretation* of any positive D result must caveat that the modulator is not building NPN-style context state.

**Step 2 (overnight, ~36–48 cell-hr; ≤ 12 cells).** `experiment-designer` locks Candidate D — Neuromod's 5-stage abrupt-double-return schedule (drop the optional gradual arm in round 1; it requires a small developer interpolator touch). Hand off to `training-runner`. This is **independent of Step 1's outcome** — D is config-only, the existing `--continual-schedule` machinery accepts the YAML, and a null is informative either way.

**Step 3 (gated on Step 1).** *If* ΔCKA ≥ 0.05, `experiment-designer` locks Candidate A1 (active vs. passive predator mix; 12 cells; one overnight on 4–6 nodes) for the next compute window. The two `--mixture-mode` and `--mod-clamp` flags (RL/BDL §8) are a half-day developer touch each — schedule them in parallel with the Step 2 run.

**Step 4 (gated on Step 1b OR Step 2 null + Step 3 null).** `senior-developer` writes the Phase 0 engineering plan (T/P split + precision head); both professors converge on these as the load-bearing repairs.

This sequence respects **project_plan.md §5 phase ordering** because it inserts the architecture-vitality probe as a Phase 0.5 *between* the v8 null (Phase 2 prefix) and the Phase 3 factorial. It does not delay Phase 4. It costs at most one overnight before yielding a real signal.

## 5. The single user lock-in question

Per the agent profile, this synthesis ends with the AskUserQuestion-ready menu the parent (top-level Claude) will show the user before any handoff.

**Question to the user**: "We have two professor recommendations and finite cluster-time tonight. Which sequence do you want?"

| Option | What happens | Compute tonight | What we learn |
|---|---|---|---|
| **A — Probe-then-decide.** | `experiment-analyzer` runs the ΔCKA probe on v8 logs first (30 min–few hours), report comes back, then we lock A1 / D / Phase 0 from the result. | None. | Whether P1 holds on the current modulators. Cheapest information per compute. |
| **B — Run D in parallel.** | `experiment-analyzer` ΔCKA probe **and** `experiment-designer` lock Candidate D fire **simultaneously**; D launches tonight on 4–6 nodes overnight; ΔCKA report lands by morning. | ~36–48 cell-hr (one overnight, ≤ 12 cells). | Both the P1 status *and* the continual-RL probe result, by tomorrow morning. The Neuromod-recommended path; treats a D-null as Phase-0-defining rather than indicting. |
| **C — Run D + A1 in parallel** (compute-aggressive). | Same as B, plus lock Candidate A1 immediately without waiting on ΔCKA. Requires the half-day developer touches for `--mixture-mode` / `--mod-clamp` *or* the alternating-episode workaround. | ~72–96 cell-hr (two overnight grids). | All three signals (P1, D, A1) by ~36 hours from now. Aggressive; risks burning compute on A1 if ΔCKA < 0.05 predicts A1-null. |
| **D — Block on Phase 0 first.** | Skip the vitality probe entirely; route to `senior-developer` for the T/P split + precision head plan; defer meta/continual until Phase 0 ships. | None tonight; 2–3 weeks of engineering. | A clean test of the headline architecture, but no data for 2–3 weeks. Neither professor recommends this; it is the "conservative" option the user might still want for project-management reasons. |

**Recommended**: **Option B**. RL/BDL's gate is free and Neuromod's experiment is config-only on already-shipped infra; the two run cleanly in parallel and yield two independent architecture-vitality signals by morning. Option C is reasonable only if the user wants to commit two nights of compute and is willing to absorb the half-day developer touch. Option A is reasonable if the user wants to be cautious with compute. Option D is appropriate only if the user has independent reasons to engineer first.

## 6. Open question for the user (cannot be resolved from the memos)

**Is tonight a compute-spending night or an audit-only night?** Both professors recommend running, but neither has visibility into the lab's current cluster availability or the user's own bandwidth for tonight. The choice between Option A (audit-only) and Option B (one overnight grid) is the user's call, not the professors'. If tonight is audit-only, we run the ΔCKA probe and lock D for tomorrow night; if tonight is compute-spending, we fire B in parallel.

A secondary open question: **does the user want the optional gradual-vs-abrupt arm of Neuromod's schedule?** It needs a small developer interpolator touch (~half-day) and roughly 1.5× the Candidate D budget. Drop it in round 1 unless the user wants it explicitly; the double-return is the architecturally-decisive piece.

---

## 7. Cross-references

- This memo's parent triage: [`docs/project/triage/20260509_1517_nmn_meta_continual_pivot.md`](../triage/20260509_1517_nmn_meta_continual_pivot.md) §7 names this synthesis as the next-step deliverable.
- Both professor memos are filed under `docs/project/directions/`, which is professor scope (architecture-decision recommendations); this synthesis is filed under `docs/project/ideas/` per research-postdoc scope.
- After user lock, downstream hand-offs (`experiment-analyzer`, `experiment-designer`, `senior-developer`) are named in §4 above. Per the research-postdoc agent profile, this synthesis does NOT spawn them — the parent (top-level Claude) does, after the user picks an option in §5.
