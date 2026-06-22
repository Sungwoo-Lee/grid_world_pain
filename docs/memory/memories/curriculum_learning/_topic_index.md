# _topic_index.md — `curriculum_learning` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `curriculum_learning` topic.

**Folder definition**: Curriculum/continual training
**Insights**: 1
**Last updated**: 2026-06-22

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-06-22 | 17:48 | [20260622_1748_basic_curriculum_overtraining_collapse_and_intervals](20260622_1748_basic_curriculum_overtraining_collapse_and_intervals.md) | From-scratch rPPO survival-step curves on the basic curriculum: the EASY levels (static-forage L0, slow-predator L1) learn fast (~0.2M episodes) but then CATASTROPHICALLY COLLAPSE from over-training (L0 ~3.8M, L1 ~9.0M) to a degenerate single-action 'spam one direction -> starve' policy with entropy ~0 (rPPO has no entropy floor / early stop). Stable levels converge L2 ~1.4M, L3 ~1.2M, L4 ~3.6M. The 5-stage continual curriculum budgets [1M,1M,2M,2M,4M] were set by capping each easy stage BELOW its measured collapse onset, so the curriculum's short early stages dodge the instability by design. |

---

## Related indexes

| Index | Path | Holds |
|---|---|---|
| Topic registry | [../../ROOT_INDEX.md](../../ROOT_INDEX.md) | All topic-folder metadata |
| Tag dictionary | [../_global_tags.md](../_global_tags.md) | All active tags |
