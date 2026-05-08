# _topic_index.md — `nmn_diagnosis` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `nmn_diagnosis` topic.

**Folder definition**: NMN performance diagnosis findings
**Insights**: 2
**Last updated**: 2026-05-08

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-05-08 | 14:27 | `20260508_1427_nmn_heterogeneity_sweep_design` | Designed a 10-cell NMN noise-heterogeneity sweep (5 profiles × 2 architectures × 1 seed) that pins mean σ at 0.140 across profiles and varies only the max/min σ ratio R = 2.0 → 18.0, so total information loss is matched and only heterogeneity varies. |
| 2026-05-08 | 14:26 | `20260508_1426_v8_noise_bug_refuted` | v8 NMN-diagnosis doc warns of a noise-config index-mismatch bug, but a code re-read shows the current loader is name-keyed — bug is not present, v8's relative comparisons hold and absolute σ values are correct. |

---

## Change history

- 2026-05-08: Added insight `20260508_1427_nmn_heterogeneity_sweep_design` (heterogeneity sweep design).
- 2026-05-08: Folder created. Added insight `20260508_1426_v8_noise_bug_refuted` (v8 bug refutation).
