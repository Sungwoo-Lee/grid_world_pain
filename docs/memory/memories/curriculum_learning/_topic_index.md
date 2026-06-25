# _topic_index.md — `curriculum_learning` folder

> One-line entry per insight, reverse-chronological (newest at top).
> Read this file when the user's question narrows to the `curriculum_learning` topic.

**Folder definition**: Curriculum/continual training
**Insights**: 3
**Last updated**: 2026-06-24

---

## Insights (newest first)

| Date | Time | ID | Summary |
|---|---|---|---|
| 2026-06-24 | 05:17 | [20260624_0517_continual_failure_is_plasticity_loss_not_budget](20260624_0517_continual_failure_is_plasticity_loss_not_budget.md) | Per current RL literature, the continual curriculum's deficit is primarily LOSS OF PLASTICITY (a damaged-network problem), not under-training — so ~100x more target-level training is unlikely to beat the baseline on its own. The field has shifted from catastrophic forgetting (backward/stability failure) to loss of plasticity (forward/plasticity failure: the network loses the ability to learn new tasks). Fixes target trainability not memory: entropy floor / ReDo / stop recurrent-reset / shrink-and-perturb. Long-L4 run is best read as a diagnostic. |
| 2026-06-24 | 05:16 | [20260624_0516_curriculum_underperformed_baseline_negative_transfer](20260624_0516_curriculum_underperformed_baseline_negative_transfer.md) | The 5-stage continual curriculum (RecurrentPPO, weights carried forward, recurrent state hard-reset at each boundary) UNDERPERFORMED from-scratch single-task training: no stage learned faster, the fast-predator stage showed negative transfer (~221 vs ~418 from-scratch), warm-starting ACCELERATED the easy-stage entropy collapse (stage 1 degenerated to entropy~0 inside its 1M cap, far earlier than the ~9M from-scratch onset), and end-of-curriculum far-sight (~200-300) tied the from-scratch baseline (~261) despite 6M extra episodes. The curriculum's transfer bet was net-NEGATIVE. |
| 2026-06-22 | 17:48 | [20260622_1748_basic_curriculum_overtraining_collapse_and_intervals](20260622_1748_basic_curriculum_overtraining_collapse_and_intervals.md) | From-scratch rPPO survival-step curves on the basic curriculum: the EASY levels (static-forage L0, slow-predator L1) learn fast (~0.2M episodes) but then CATASTROPHICALLY COLLAPSE from over-training (L0 ~3.8M, L1 ~9.0M) to a degenerate single-action 'spam one direction -> starve' policy with entropy ~0 (rPPO has no entropy floor / early stop). Stable levels converge L2 ~1.4M, L3 ~1.2M, L4 ~3.6M. The 5-stage continual curriculum budgets [1M,1M,2M,2M,4M] were set by capping each easy stage BELOW its measured collapse onset, so the curriculum's short early stages dodge the instability by design. |

---

## Related indexes

| Index | Path | Holds |
|---|---|---|
| Topic registry | [../../ROOT_INDEX.md](../../ROOT_INDEX.md) | All topic-folder metadata |
| Tag dictionary | [../_global_tags.md](../_global_tags.md) | All active tags |
