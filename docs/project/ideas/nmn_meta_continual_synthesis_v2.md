# Synthesis v2 — NMN architecture-vitality probe (meta + continual), post-olfactory-perturbation

**Author**: research-postdoc
**Date**: 2026-05-09
**Mode**: Cross-professor synthesis (Mode 3, refresh)
**Supersedes**: nothing — append-only synthesis layer. v1 [`nmn_meta_continual_synthesis.md`](./nmn_meta_continual_synthesis.md) stays as a historical snapshot of pre-olfactory-perturbation thinking.

**Inputs (refreshed)**:
- v1 synthesis: [`nmn_meta_continual_synthesis.md`](./nmn_meta_continual_synthesis.md) (still valid; only §5 lock-in menu and §4 sequence have been updated by what follows).
- prof-RL/BDL **v2**: [`nmn_meta_context_conditioning_v2.md`](../directions/nmn_meta_context_conditioning_v2.md) — the meta task moves from 2-context to **2×3 = 6-context**, a new factorisation property P3 enters, and the falsifying measurement scales from a scalar Δ_CKA to a 6×6 Gram matrix with two summary stats.
- prof-RL/BDL v1 (still on the record): [`nmn_meta_context_conditioning.md`](../directions/nmn_meta_context_conditioning.md).
- prof-Neuromod **(unchanged)**: [`nmn_continual_lifelong_probe.md`](../directions/nmn_continual_lifelong_probe.md).
- Triage anchor: [`20260509_1517_nmn_meta_continual_pivot.md`](../triage/20260509_1517_nmn_meta_continual_pivot.md).
- Round 2 sameProp memory: `docs/memory/memories/hypervigilance/20260509_1532_sameprop_round2_truncated_verdict.md`.

**Project anchors** (unchanged from v1): `project_plan.md` §4 / §5 (Phase 0–4 ordering); `NMN_ARCHITECTURE_REVIEW.md` §2.1–§2.5; `NEUROMODULATION_ALGORITHM.md` §1.4 (H1–H5), §B.6 (NPN), §G (Lee), §I (Rodriguez-Garcia), §K (Tsuda).

---

## TL;DR — What this memo is about (read this first)

The project trains an AI agent in a small grid world to survive — find food, avoid predators, manage hunger. Two architectures are being compared:

- a **plain agent** (a standard recurrent neural network — one brain), and
- a **modulated agent** (the same brain plus an extra small head sitting on top that can dynamically adjust how the main brain reads its senses, depending on context).

The modulated version is supposed to do better in conditions that benefit from dynamic reweighting. We are testing whether it actually does.

**What just happened.** Two experiments this week tested the modulated agent under noisy senses. Both said: the modulator does not help, even after we removed a configuration cap that was holding it back. This memo is the second version (v2) of a synthesis pivoting to the regimes where the academic literature consistently reports modulators should help:

- **Meta-learning** — the world is one of several possible types each episode; the agent must figure out which type from observations and adapt. The agent is never told which type it's in.
- **Continual learning** — the world changes between long training stages; the agent must not forget what it learned in earlier stages when the world switches back.

If the modulator helps in either of these, the architecture is fine and noisy-senses is just hard. If it fails in both, the architecture itself needs to be redesigned.

**What this memo recommends.** Run a free pre-check on data we already have tonight (no GPU spend), in parallel with a continual-learning experiment that fires overnight on 4-6 lab nodes (≈ 36-48 GPU-hours). If the pre-check confirms the modulator is at least *trying* to do context-detection, fire the meta-learning experiment tomorrow night. This is **Option B** in the lock-in menu in §4. Three options not recommended (pre-check-only, run-everything-aggressive, block-on-architectural-redesign) are also there for comparison.

The technical body that follows uses project-internal shorthand (`v1`, `Δ_CKA`, `P3`, `Candidate A1'`, `Phase 0`, etc.). Cross-references at the end of the memo.

---

## 1. What changed since v1

Two specific deltas on the **meta** side; **continual unchanged**.

**Δ-1 — The meta task scales from 2 to 6 contexts.** v1's Candidate A1 was a single 2-context mix (active vs. passive predator under matched olfactory). The user asked the prof to also vary the *olfactory* properties — predator/rabbit smell-swap, distinct-canonical, and matched. The v2 memo accepts and structures this as the cross-product

$$
\mathcal{C} \;=\; \mathcal{M} \times \mathcal{P} \;=\; \{\text{active}, \text{passive}\} \times \{\text{matched}, \text{distinct-canonical}, \text{swapped}\},
$$

with **uniform sampling at episode reset** (not curriculum, not stratified). Total contexts: 6. Cell count refits to 10 cells (2 modulated × 2 seeds + 2 unmod × 2 seeds + 6 specialist controls × 1 seed) — a wash with v1's 12 cells, both fitting under the project's "easy first" ≤ 10–12 ceiling. Three new YAMLs needed (~1 hr config-only); `01-interoNocicept_sameProp.yaml` already covers the matched cell.

**Δ-2 — The Δ_CKA gate scales from scalar to a 6×6 Gram matrix with two summary stats.** v1's gate was $\Delta_{\text{CKA}} \geq 0.05$, a single number. v2 keeps that as $\bar\Delta_{\text{CKA}}$ (P1, generalised — the within–between contrast averaged over the Gram) and adds a **factorisation statistic** $\bar\Delta_{\text{CKA}}^{\text{factor}}$, which compares off-diagonal CKA between contexts that share $p$ but differ on $m$ vs. those that share $m$ but differ on $p$. P3 (the new factorisation property) requires $|\bar\Delta_{\text{CKA}}^{\text{factor}}| \approx 0$. P3 ⊃ P1: a P1-positive that fails P3 means the modulator builds a context code but does not factorise across the two manipulation axes. The probe still runs cheaply on existing v8 trajectories — the cost in *compute* is unchanged; the cost in *analyzer work* grows modestly (a 6×6 Gram with two summary stats instead of one scalar).

**Continual side unchanged.** Prof-Neuromod's 5-stage abrupt-double-return on the existing `--continual-schedule` machinery, biological-plausibility verdict ("partial — Rodriguez-Garcia mechanism feasible, Lee mechanism gated on Phase 0 precision head"), and "run-now beats block" recommendation all carry forward from v1.

A new diagnostic primitive enters that is *only* available because of Δ-1: the **per-cue clamp ablation** (v2 memo §3). With both olfactory and movement cues in play across contexts, the agent can solve some sub-mixes via olfactory shortcut. v2 restores diagnostic cleanness not by restricting cues but by measuring per-cue causal contribution at eval — olfactory-clamp vs. modulator-state-clamp, producing a *signed* prediction matrix that distinguishes "modulator does context-conditioning" from "modulator uses olfactory shortcut". This *strengthens* the meta side's interpretive power; it does not weaken or contradict v1's clamp-ablation framing, it generalises it.

## 2. Refreshed side-by-side professor positions

**Where they still agree** (load-bearing for the project, unchanged from v1 §2):

1. Both still treat the current pre-Phase-0 architecture as **worth probing** despite the v8 null. Neither argues to block on Phase 0 first.
2. Both still name a *mechanism check* alongside the survival number. Prof-RL/BDL v2 expands this from "the clamp ablation" to "the per-cue clamp matrix"; prof-Neuromod keeps the boundary-locked Mahalanobis distance + $\sigma(z_{\text{uni}})$ / $\tau_\pi$ transient. Same epistemic stance, two experiments.
3. Both still identify **T/P split** and **precision head** as the load-bearing Phase 0 levers. v2 *tightens* this case: the swap context makes the precision head specifically load-bearing (the agent must down-weight a misleading cue), and the 16-dim modulator GRU is now likely under-capacitised for the 2-axis factorisation P3 demands, weakly raising the case for the T/P split's two recurrent streams. Opioid head still out of scope for both probes.
4. Both still treat $h^{\text{mod}}$ structure as the canonical "is the modulator doing context work" diagnostic. v2's 6×6 Gram matrix is the same probe scaled.

**Where they still diverge** (the user must choose, unchanged from v1 §3):

1. **Compute timing.** Prof-RL/BDL v2 still gates the meta launch on the (now richer) measurement-only probe; prof-Neuromod still launches Candidate D in parallel because the schedule is config-only on already-shipped infra and a null is informative. Not contradictory — different experiments — but they imply different orderings under finite compute.
2. **What constitutes "the cleanest test."** Prof-RL/BDL v2: the 2×3 mix is *strictly stronger* than v1's 2-context mix, because it tests the factorisation property P3. Prof-Neuromod: the cleanest contrast is still the double-return schedule, discriminating monotone-forgetting from Tsuda-hypertube reusable subnetworks. Orthogonal questions on the same architecture.
3. **What a null *means*.** Prof-RL/BDL v2: a $\bar\Delta_{\text{CKA}} < 0.05$ on v8 logs is still a hard stop → Phase 0. A *positive* $\bar\Delta_{\text{CKA}}$ but a *negative* $\bar\Delta_{\text{CKA}}^{\text{factor}}$ is a new soft stop (predict partial clamp-ablation cleanness; run anyway but caveat). Prof-Neuromod: a survival-null on Candidate D still *defines* what Phase 0 fixes, not a stop.

**Does the v2 update change the relative cost-effectiveness of the two candidates?** Marginally, but not enough to flip the v1 verdict. v1 had A1 at 12 cells × ~13h ≈ 156 cell-hours; v2 has A1' at 10 cells × ~14h ≈ 140 cell-hours. D is unchanged at 6–12 cells × 17h ≈ 100–200 cell-hours. So the *cost* numbers are essentially unchanged. What did change is the **information density per cell-hour on the meta side**: v2 tests P3, not just P1; v2 produces a per-cue signed prediction matrix; v2 generates strictly stronger evidence per overnight than v1 would. The continual side's information density is unchanged. The meta side has gotten *cheaper per bit of evidence*, which weakly tilts the recommendation toward firing it sooner — but the question of whether to gate it on the probe or run it in parallel with D is still a function of the user's compute appetite, not of v2's redesign.

## 3. Refreshed recommended next-step sequence

The v1 sequence still works structurally. Step 1 is unchanged in cost (the 6×6 Gram still runs on existing v8 trajectories with no new training; the analyzer work is modestly heavier but still hours-scale). Step 2 launches D in parallel with the probe. Step 3 launches v2 meta gated on the probe outcome. Step 4 is the Phase 0 fallback.

**The v2-specific question**: should the v2 meta launch be **further gated** on the probe (Option A/B) or **launched alongside D in parallel** without the probe (Option C, compute-aggressive)?

I recommend **keeping the probe-gate on the v2 meta launch**. Three reasons:

1. The **factorisation stat is informative on null** — if $|\bar\Delta_{\text{CKA}}^{\text{factor}}| > 0.15$ on v8 logs, we predict the v2 meta will partially-null on the per-cue ablation matrix (modulator builds context code but does not factorise). That prediction is itself a publishable finding (no factorisation in NPN-style modulators is, per the v2 memo §1.3, *not directly demonstrated* in the reference corpus); we can still launch but the launch is now informed.
2. The **probe cost remains negligible** (analyzer work, no compute), so gating loses nothing.
3. The continual side fires regardless of probe outcome (D's mechanism is not strictly downstream of P1 / P3); D's run-tonight value is *unaffected* by the v2 update.

Together: 240–340 cell-hours = 2–3 nights on 6 nodes, well within "easy parts first". The probe + D fire tonight; v2 meta fires the night after if the probe greenlights it.

## 4. Refreshed lock-in menu

The v1 had four options (A/B/C/D — probe-only / probe+D / probe+D+A1 aggressive / block-on-Phase-0). The user neither answered nor explicitly rejected; the menu is still fresh. The v2 update tweaks cell counts (10 not 12 for meta) and swaps A1→A1' (the 2×3 mix); the menu structure is preserved.

| Option | What happens | Compute tonight | What we learn |
|---|---|---|---|
| **A — Probe-then-decide.** | `experiment-analyzer` runs the **6×6 Gram-matrix Δ_CKA + factorisation** probe on v8 logs first (hours, analyzer-only). Report comes back, then we lock A1' / D / Phase 0 from the result. | None. | Whether P1 holds *and* whether P3 holds on current modulators. Cheapest information per compute; the v2 update *adds* a measurable signal here (factorisation) over v1. |
| **B — Run D in parallel.** *(Recommended.)* | 6×6 Gram-matrix probe **and** Candidate D fire **simultaneously**; D launches tonight on 4–6 nodes overnight; probe report lands by morning. v2 meta locks tomorrow night if probe greenlights. | ~36–48 cell-hr (one overnight, ≤ 12 cells for D). | Both the P1/P3 status *and* the continual-RL probe result, by morning. Treats a D-null as Phase-0-defining. **No change from v1's recommendation.** |
| **C — Run D + v2 meta in parallel** (compute-aggressive). | Same as B, plus lock the v2 meta (2×3 mix) immediately without waiting on the probe. Requires the half-day developer touches for `--mixture-mode` / `--mod-clamp` *plus* the new `--obs-clamp <modality>` flag (~1.5 days total) *or* the alternating-episode workaround for the mix. | ~140–190 cell-hr (D + v2 meta, two overnight grids). | All three signals (P1/P3, D, v2 meta) by ~36–48 hours. Aggressive; risks burning compute on the v2 meta if Δ_CKA-bar < 0.05 predicts a meta-null in advance. **The v2 update mildly raises the cost of being wrong here** (3 flags not 2 for the developer touch) but also raises the upside (per-cue clamp matrix yields a publishable signed-prediction read-out even on partial nulls). |
| **D — Block on Phase 0 first.** | Skip the vitality probe; route to `senior-developer` for the T/P split + precision head plan (precision head specifically tightened-up by v2's swap context); defer meta/continual until Phase 0 ships. | None tonight; 2–3 weeks of engineering. | A clean test of the headline architecture, but no data for 2–3 weeks. Neither professor recommends this; it remains the "conservative" option. v2 weakly strengthens this option's prior (the under-capacitised 16-dim modulator argument), but not enough to flip the verdict. |

**Recommended**: **Option B**, unchanged from v1. The v2 update enriches the *content* of what the probe and the eventual meta launch tell us, but does not change which option dominates. The user picks A if compute-cautious, B if a one-night-on-6-nodes spend is fine, C if two nights are fine and the developer can land the 1.5-day flag work in parallel, D if there are independent project-management reasons to engineer first.

## 5. Open questions

**Carried forward from v1 §6** (unresolved; user's call):

1. **Compute-spending night vs. audit-only night.** Tonight's cluster availability and user bandwidth determine whether we fire B or A. Neither professor has visibility into this.
2. **Gradual-vs-abrupt arm of D's schedule.** Needs a small developer interpolator touch (~half-day) and ~1.5× D's budget. Drop in round 1 unless the user wants it explicitly; the double-return is the architecturally-decisive piece.

**New from v2:**

3. **Is the distinct-canonical olfactory case redundant with sameProp Round 1 data?** The user's question said "swap" specifically, but the v2 memo §2.2 argues 3 olfactory levels (matched / distinct-canonical / swapped) are strictly more diagnostic than 2 (matched / swapped) — the third level disambiguates "modulator encodes which axis" from "modulator encodes a 2-bit context code". Distinct-canonical also doubles as a sanity-check ceiling (where olfactory alone is sufficient, an architecture-blind agent should solve it). However, sameProp Round 1 already produced data for the matched case. Should the v2 launch include a fresh matched cell (consistent within-experiment statistics) or rely on Round 1 + Round 2 for the matched arm and only train new cells on distinct + swap? Recommend the former (fresh, consistent cell within a uniform mix) — the matched cell within the joint mix produces *different* training dynamics than matched in isolation, so reusing the Round 1 / 2 matched data is not actually a like-for-like substitute.
4. **Two seeds is the floor; is it the target?** Round 2 sameProp showed ±4–5 step seed noise *and* a sign flip on Cell C between R1 and R2 plausibly attributable to early-learning artifact. v2's 10-cell budget commits 2 seeds per modulated cell. The analyzer must check seed convergence before drawing v2 conclusions; if 2 seeds disagree by > ~5 survival steps, escalate to a third seed (adds 4 cells). This is a ~1.5×-budget contingency the user should be aware of going in.
5. **Do we want to formalise P3 as a registered prediction?** The v2 memo §1.3 notes that P3 is *not* directly demonstrated in any reference paper — a v2-positive on the factorisation stat would be a novel finding, but it also raises the prior probability of a v2-null. Pre-registering the P3 prediction (e.g., publishing $|\bar\Delta_{\text{CKA}}^{\text{factor}}| < 0.10$ as the success threshold *before* the probe runs) makes a positive much more credible. Cheap; the analyzer just has to pre-commit. Recommend including in the design doc.

## 6. Round 2 sameProp calibration — must thread through v2 analysis

The Round 2 sameProp partial verdict (`20260509_1532_sameprop_round2_truncated_verdict`) reports MeanDistRabbit ≈ 3.84 and MeanDistPredator ≈ 7.70 on Cell A1 (passive predator, matched olfactory): the agent under matched olfactory **does not avoid the rabbit at all** and operationally lives in the BR rabbit-and-food corner for 482/500 steps. Δ = +3.86 in matched olfactory is *uninterpretable* without per-quadrant occupancy because of corner-camping.

For v2 this matters in two ways and the analyzer must flag both:

- **Matched-context survival in any v2 cell must be interpreted with the camping-rate flag.** The analyzer comparing modulated vs. unmodulated under (m, matched) must report per-quadrant occupancy alongside survival. If both arms corner-camp at similar rates, matched-context survival differences are noise; if they camp differently, the modulator is doing different *escape* policies, not different context-conditioning — distinct phenomena that look the same in raw survival.
- **Swap is the load-bearing context.** Under olfactory swap, corner-camping is *not* a viable strategy: camping with the rabbit means camping with the apparent-predator (will either flee or die), and camping in the corner that smells like predator is also unstable. The agent must either ignore olfactory or correctly invert it. Both routes reveal the modulator's contribution. The v2 memo §7 makes this point; the synthesis re-emphasises it because it *is* the load-bearing reason the v2 memo's olfactory enrichment is worth doing — without swap, the matched case alone would just produce more uninterpretable corner-camping data.

In short: matched-context numbers in v2 cells are sanity checks at best; the swap-context numbers carry the falsifying weight.

---

## 7. Cross-references

- This memo's parent triage: [`docs/project/triage/20260509_1517_nmn_meta_continual_pivot.md`](../triage/20260509_1517_nmn_meta_continual_pivot.md).
- v1 synthesis (historical): [`nmn_meta_continual_synthesis.md`](./nmn_meta_continual_synthesis.md). v1's §5 menu is superseded by §4 of this memo; v1's §1–§4 still hold structurally for the unchanged continual side.
- After user lock, downstream hand-offs (`experiment-analyzer`, `experiment-designer`, `developer`, `senior-developer`) are named in v2 memo §8.1. Per the research-postdoc agent profile, this synthesis does NOT spawn them — the parent (top-level Claude) does, after the user picks an option in §4.
