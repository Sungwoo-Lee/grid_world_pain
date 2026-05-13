# _topic_index.md — `nmn_diagnosis` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `nmn_diagnosis` topic.

**Folder definition**: NMN performance diagnosis findings
**Insights**: 9
**Last updated**: 2026-05-13

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-05-13 | 00:17 | `20260513_0017_mod_h_logging_gap_blocks_cka_precheck` | R2 continual probe's modulator-engagement Mahalanobis check is unevaluable because the project never logs the raw modulator hidden vector — only summary statistics (mean, std). Three new WandB metrics requested: `modulator/mod_h_norm`, `Episode/Term_Predator`, `Episode/Occupancy_*`. Without `eval/h_mod_samples` hook + `mod_h_norm`, no CKA pre-check or representation-similarity test is possible on existing or future NMN runs. |
| 2026-05-13 | 00:16 | `20260513_0016_h1a_half_the_dip_predicate_schedule_asymmetric` | R2 continual probe's H₁a 'half-the-dip stability gap' predicate is schedule-asymmetric and methodologically malformed: the design assumed stage 2 is a stress dip, but in the active→passive→active→passive→active schedule, stage 2 is the easier passive stage, so survival rises rather than dips. The modulator's actual contribution lives in recovery speed at passive→active boundaries, not in dip-depth attenuation. H₁a is INCONCLUSIVE, not refuted — the predicate needs re-formalisation. |
| 2026-05-13 | 00:14 | `20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin` | R2 continual sister-pair probe (5.1M ep, 5-stage active↔passive schedule) delivers the first clearly-positive architecture-vitality finding for the FiLM modulator: H₁b (catastrophic-forgetting resistance) and H₁c (Tsuda-style reusable subnetwork) both CONFIRMED at ~25× the seed-noise floor. Modulator beats unmodulated baseline by +107 steps on the 1st return-to-active stage and +132 steps on the 2nd; modulator's 2nd return ≥ 1st (+5.75), baseline shows monotone decay (−19.79). |
| 2026-05-09 | 14:10 | `20260509_1410_nmn_temp_head_natural_target_3_to_5` | MC FiLM temperature head's natural output target sits in [3.0, 5.0), profile-dependently — head wants ~3.0 on P3 (R=5), ~4.7 on P4 (R=9.67), ~4.5 on P5 (R=18). Old [0.5, 3.0] ceiling was binding on P4/P5 (head pinned at 3.0) but NOT on P3 (head naturally settled there). For future NMN configs: temp_clip [0.5, 5.0] minimum to avoid binding at R ≥ 9. |
| 2026-05-09 | 14:09 | `20260509_1409_nmn_tempceil10_verdict_h1b_confirmed_p4` | 5-cell temp_clip[0.5,10.0] re-run confirms H1b on P4 (improvement +6.37 over old-ceiling P4f, but FiLM still trails Unmod by −2.55); H1a/H0/H1c refuted; P3 3-seed disambiguation refuted (σ=±4.40, parent's +0.08 was noise). Reframes parent verdict: 'FiLM-worse everywhere' was PARTIALLY a hyperparameter artefact, not purely structural — but freeing the ceiling is not sufficient for FiLM to overtake Unmod. |
| 2026-05-08 | 20:04 | `20260508_2004_profile_dependent_temp_saturation_mc_film` | MC FiLM temp_max saturates at the temp_clip 3.0 ceiling for P3f/P4f/P5f (R ≥ 5) but stays at 1.50–2.30 for P1f/P2f. Saturation is profile-dependent at R ≈ 5; raising temp_clip ceiling and re-running P3–P5 is the minimum-viable architectural test before any FiLM redesign. |
| 2026-05-08 | 20:03 | `20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse` | 10-cell NMN noise-heterogeneity sweep refutes H1a/H1b/H1c — FiLM g1 is consistently 5–13 steps WORSE than Unmod LN at every R profile (gap narrows but never crosses zero). Closes v8 §6.3 P4 lead: heterogeneity is not the missing ingredient, the architecture is the bottleneck. |
| 2026-05-08 | 14:27 | `20260508_1427_nmn_heterogeneity_sweep_design` | Designed a 10-cell NMN noise-heterogeneity sweep (5 profiles × 2 architectures × 1 seed) that pins mean σ at 0.140 across profiles and varies only the max/min σ ratio R = 2.0 → 18.0, so total information loss is matched and only heterogeneity varies. |
| 2026-05-08 | 14:26 | `20260508_1426_v8_noise_bug_refuted` | v8 NMN-diagnosis doc warns of a noise-config index-mismatch bug, but a code re-read shows the current loader is name-keyed — bug is not present, v8's relative comparisons hold and absolute σ values are correct. |

---

## Change history

- 2026-05-13: Added 3 insights from the NMN R2 continual + 6-specialist analyzer verdict session: `20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin` (first positive FiLM finding — H₁b + H₁c confirmed at ~25× seed-noise floor on the 5-stage continual schedule), `20260513_0016_h1a_half_the_dip_predicate_schedule_asymmetric` (H₁a predicate is methodologically malformed for the schedule asymmetry — modulator wins via recovery speed, not dip-depth attenuation), `20260513_0017_mod_h_logging_gap_blocks_cka_precheck` (raw modulator hidden vector never logged — Mahalanobis / CKA tests unevaluable; 3 metrics + 1 artifact hook requested). No new tags promoted (all reused: nmn, film, hypervigilance, decision, learned_lesson, design, meta, training_runner).
- 2026-05-09: Added insights `20260509_1409_nmn_tempceil10_verdict_h1b_confirmed_p4` (temp_clip[0.5,10.0] re-run verdict — H1b confirmed on P4; partial-binder reframing of parent sweep) and `20260509_1410_nmn_temp_head_natural_target_3_to_5` (MC FiLM temperature head's natural target [3.0, 5.0); profile-dependent; canonical FiLM config should adopt [0.5, 5.0] default).
- 2026-05-08: Added insights `20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse` (sweep verdict — H1a/b/c refuted, FiLM-worse direction) and `20260508_2004_profile_dependent_temp_saturation_mc_film` (profile-dependent temp saturation at R ≥ 5; actionable mechanism finding).
- 2026-05-08: Added insight `20260508_1427_nmn_heterogeneity_sweep_design` (heterogeneity sweep design).
- 2026-05-08: Folder created. Added insight `20260508_1426_v8_noise_bug_refuted` (v8 bug refutation).
