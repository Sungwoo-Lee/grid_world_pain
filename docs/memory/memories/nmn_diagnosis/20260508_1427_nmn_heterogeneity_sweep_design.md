---
id: 20260508_1427_nmn_heterogeneity_sweep_design
date: 2026-05-08
time: "14:27"
folder: nmn_diagnosis
tags: [nmn, hypervigilance, design, decision]
summary: "Designed a 10-cell NMN noise-heterogeneity sweep (5 profiles × 2 architectures × 1 seed) that pins mean σ at 0.140 across profiles and varies only the max/min σ ratio R = 2.0 → 18.0, so total information loss is matched and only heterogeneity varies."
related: ["20260508_1426_v8_noise_bug_refuted"]
session_origin: claude_code
session_label: "nmn_noise_heterogeneity_sweep_launch_2026-05-07/08"
importance: medium
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/c7ee226b-2162-4e7a-95e9-257a5b19d713.jsonl
raw_completeness: full
---

# NMN noise-heterogeneity sweep design — matched-mean σ + R-gradient

## Key conclusion
The v8 NMN diagnosis showed FiLM ≤ Unmod under perceptual noise, but the noise preset was relatively flat across modalities. v8 §6.3 P4 hypothesized that *heterogeneous* noise (some modalities very noisy, others clean) might create the selective pressure FiLM needs. To test this without confounding "more heterogeneity" with "more total noise," the sweep design pins **mean σ at 0.140 across all 5 profiles** and varies only the max/min σ ratio R from 2.0 (P1, ≈ v8 baseline) to 18.0 (P5, extreme). Each profile is paired with both FiLM g1 and Unmod LN architectures (10 cells total), single seed each — max breadth, no replication, relying on v7's ~0.44-step seed-variance estimate as the floor for "real" effects.

## Evidence, measurements, facts
- 5 noise profiles, all with σ-bearing-sum = 0.700 (mean = 0.140) verified to floating-point precision: P1 R=2.00, P2 R=2.75, P3 R=5.00, P4 R=9.67, P5 R=18.00 (geometric monotone).
- Floor channels (injury, nutrition, collision, proprioception, location) σ = 0 across all 5 profiles — controls match v8.
- 2 architectures: `recurrent_ppo_nmn_het_unmod.yaml` (Unmod LN, MC return) and `recurrent_ppo_nmn_het_film_g1.yaml` (FiLM grouping_size=1, MC, LN). FiLM g1 was v8's best modulated config, so the "best case for NMN."
- Pre-registered hypotheses with confirmation criteria: H₀ (no profile yields NMN > Unmod by > seed-variance floor), H₁a (extreme heterogeneity P5 rescues NMN), H₁b (gradient is monotonic in R), H₁c (NMN never beats Unmod, confirms v8 across all profiles).
- 10M episodes per cell. Surprise finding (separate, not yet captured): training is ~13 hours per cell, not the days v8 expected — likely shorter mean episode length under moderate-to-extreme heterogeneity.
- WandB group: `nmn_noise_heterogeneity`. Run IDs locked at design time per project's launch-manifest contract.

## Decisions and actions
- **Matched-mean σ** chosen over matched-total-noise-variance because variance is non-linear in σ and would have made the gradient non-monotone in any meaningful sense. Linear σ-sum was the simplest invariant that future readers could verify by eye.
- **Geometric R-gradient {2, 2.75, 5, 9.67, 18}** chosen so adjacent profiles differ by ~2× in R, preserving log-linear spacing without wasting cells at the low end.
- **MC over GAE** — v8 showed MC has cleaner dynamics and FiLM hurts GAE more than MC; testing under MC is the "best case" for FiLM, so a null result there is stronger.
- **Single seed × 10 configs** chosen over 5×2 or 2×5 because v7 established seed-variance is small (~0.44 steps), and breadth of profile coverage was deemed more diagnostic than statistical replication. Caveat documented in §4 of the design doc.
- **G1' channel-rank diagnostic** is running in parallel — by design, P5 of this sweep should pass G1' a priori (max σ / min σ > 5 → > 2 channel-rank threshold), so a P5 null result rules out channel-rank degeneracy as the explanation.
- Bug-handling decision (a): note v8's flagged noise-index-mismatch bug as a confound, after auditing and refuting it. See companion insight `20260508_1426_v8_noise_bug_refuted`.

## Open questions and follow-ups
- Awaiting all 10 cells to converge (cells 3–4 done in ~13h, cells 1–2 will be last at ~17–22h). When done: spawn `experiment-analyzer` to fill in §7 Results / §8 Conclusions of the design doc and adjudicate H₀ / H₁a / H₁b / H₁c.
- If H₀ holds (NMN ≤ Unmod across all profiles): this strongly reinforces the "structural" explanation in `NMN_ARCHITECTURE_REVIEW.md` §6 (sigmoid gain ceiling, no input projection on modulator GRU). Next step would be P3 (architecture fix) rather than further noise-profile sweeps.
- If H₁a holds at P5 only: the heterogeneity threshold matters; future precision-modulator work should target the P5-style profile, not v8's P1.

## References
- Design doc: `docs/experiments/active/hypervigilance/NMN_NOISE_HETEROGENEITY_SWEEP.md`.
- Configs: `configs/experiment/nmn_noise_heterogeneity/p{1..5}_*.yaml`, `configs/models/recurrent_ppo_nmn_het_{unmod,film_g1}.yaml`.
- Anchors: `docs/develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md` §6.3, `docs/develop/active/neuromodulation/NMN_ARCHITECTURE_REVIEW.md` §6, `docs/experiments/active/g1_prime_diagnostic/G1_PRIME_CHANNEL_RANK_DIAGNOSTIC.md`.
- Companion insight: `20260508_1426_v8_noise_bug_refuted`.
- Commit: `0ed4942 feat(experiment): ✨ NMN noise-heterogeneity sweep (10-cell P4 follow-up to v8)`.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume c7ee226b-2162-4e7a-95e9-257a5b19d713` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view). Shared JSONL with companion insight `20260508_1426_v8_noise_bug_refuted`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
