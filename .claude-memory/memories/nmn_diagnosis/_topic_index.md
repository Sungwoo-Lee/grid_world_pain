# _topic_index.md — `nmn_diagnosis` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `nmn_diagnosis` topic.

**Folder definition**: NMN performance diagnosis findings
**Insights**: 4
**Last updated**: 2026-05-08

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-05-08 | 20:04 | `20260508_2004_profile_dependent_temp_saturation_mc_film` | MC FiLM temp_max saturates at the temp_clip 3.0 ceiling for P3f/P4f/P5f (R ≥ 5) but stays at 1.50–2.30 for P1f/P2f. Saturation is profile-dependent at R ≈ 5; raising temp_clip ceiling and re-running P3–P5 is the minimum-viable architectural test before any FiLM redesign. |
| 2026-05-08 | 20:03 | `20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse` | 10-cell NMN noise-heterogeneity sweep refutes H1a/H1b/H1c — FiLM g1 is consistently 5–13 steps WORSE than Unmod LN at every R profile (gap narrows but never crosses zero). Closes v8 §6.3 P4 lead: heterogeneity is not the missing ingredient, the architecture is the bottleneck. |
| 2026-05-08 | 14:27 | `20260508_1427_nmn_heterogeneity_sweep_design` | Designed a 10-cell NMN noise-heterogeneity sweep (5 profiles × 2 architectures × 1 seed) that pins mean σ at 0.140 across profiles and varies only the max/min σ ratio R = 2.0 → 18.0, so total information loss is matched and only heterogeneity varies. |
| 2026-05-08 | 14:26 | `20260508_1426_v8_noise_bug_refuted` | v8 NMN-diagnosis doc warns of a noise-config index-mismatch bug, but a code re-read shows the current loader is name-keyed — bug is not present, v8's relative comparisons hold and absolute σ values are correct. |

---

## Change history

- 2026-05-08: Added insights `20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse` (sweep verdict — H1a/b/c refuted, FiLM-worse direction) and `20260508_2004_profile_dependent_temp_saturation_mc_film` (profile-dependent temp saturation at R ≥ 5; actionable mechanism finding).
- 2026-05-08: Added insight `20260508_1427_nmn_heterogeneity_sweep_design` (heterogeneity sweep design).
- 2026-05-08: Folder created. Added insight `20260508_1426_v8_noise_bug_refuted` (v8 bug refutation).
