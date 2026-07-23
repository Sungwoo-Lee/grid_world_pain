---
id: 20260723_1909_nnx_abandoned_archived_stack_confusion
date: 2026-07-23
time: "19:09"
folder: dreamer_diagnosis
tags: [dreamer, decision, learned_lesson, meta]
summary: "DreamerV3-NNX is the abandoned stack (2026-05-12 pivot); dreamer_srl is live but has no NMN hooks and no offline eval. A full fix-batch was misdirected onto NNX before a git-history check caught it; NNX was then archived to a TRACKED archive/ path (never legacy/, which is gitignored)."
related: ["20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned", "20260723_1910_sheeprl_parity_drift_lives_in_glue"]
session_origin: claude_code
session_label: "Fable 5 re-diagnosis + parity fixes + NNX archival + live-path inspection"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/07436809-982c-4ed5-b1fa-e84c030f5fd6.jsonl
raw_completeness: full
---

# Two Dreamers: NNX abandoned, dreamer_srl live — check stack-liveness before routing Dreamer work

## Key conclusion

The project has TWO Dreamer implementations and it is easy to work on the wrong one. **dreamer_srl** (`src/algorithms/dreamer_srl/`, own entry point `dreamer_srl_main.py`) is the LIVE one — the sheeprl port, actively developed and launched. **DreamerV3-NNX** (`src/models/dreamer_v3_*`, the `train.py --algorithm DreamerV3` path) was ABANDONED by the 2026-05-12 PI pivot ([[20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned]]) after a 5x survival gap survived months of debugging. During this session a large fix batch (the WP-NNX sheeprl-recipe alignment, commits `32c67ca`/`7304e75`) was spent aligning the ABANDONED NNX stack — because no session checked the pivot history first; the user caught it. NNX was then archived out of the live tree. Two live-Dreamer capability gaps were also confirmed: dreamer_srl has **zero NMN/modulation hooks** (the NMN port promised at the pivot never happened) and **no offline behavioral eval** (`eval_rollout.py` raises `NotImplementedError`; checkpoints have no restore consumer).

## Evidence, measurements, facts

- Git cadence: `dreamer_v3_*` last substantive feature commit 2026-05-11; everything after is bug-fix. `dreamer_srl/` first commit 2026-05-13, continuous development since, all recent cluster launches.
- Misdirected effort: WP-NNX commits `32c67ca` + `7304e75` (recipe alignment U1-U5/U4/is_first) landed on the abandoned stack before the history check.
- Archival: NNX moved to `src/models/archive/dreamer_v3_nnx/` (5 modules incl. `modulated_layer_norm_gru_cell.py`; live rPPO-NMN's `modulated_gru_cell.py` stays) + `configs/models/archive/dreamer_v3_nnx/` (8 configs). `train.py` shrank ~502 lines behind a fail-fast "use dreamer_srl" ValueError stub; `evaluation.py` branch raises the same. Commit `fd7af84` (renames swept into `2a7c6f9` by a parallel session), registry `84664e4`.
- **Gitignore trap (reusable)**: repo-root `legacy/` is gitignored AND `.gitignore` carries a `*legacy*` pattern — ANY path containing "legacy" silently untracks. Archive to a `archive/`-named tracked path instead (matches `configs/environment/experiment/archive/` + `docs/develop/archive/` precedent). Verified with `git check-ignore`.
- Archived tests opt-in via `GWP_RUN_ARCHIVED_NNX_TESTS=1`; suite stayed at the 8 known-red baseline (920 collected post-archive); rPPO smoke proved the live path untouched.

## Decisions and actions

- Archived DreamerV3-NNX; live Dreamer is dreamer_srl. Any `src/models/dreamer_v3_*` or `configs/models/dreamer_v3/` reference is now legacy.
- Banked the routing rule to the built-in auto-memory (`project_dreamer_stack_status.md` + MEMORY.md index row): check which stack is live before routing any Dreamer fix/audit.
- KNOWN_BUGS registry gained a "Dreamer neuromodulation does not exist in the live stack" planning-hazard row and archived-stack annotations (commit `84664e4`).

## Open questions and follow-ups

- Offline dreamer_srl eval: build it (feature package) or accept in-driver-only? Capability-gap decision pending.
- NMN-on-srl port: an unstarted ~600-900 line port; readiness memo written (`docs/develop/active/diagnosis/train_eval_live_paths_2026-07-10/03_...`). PI-worthy direction call — it defines what "Dreamer NMN" means for the paper.
- The WP-NNX fixes now live in the archive; if NNX is ever revived they are a head-start, else moot.

## References

- Pivot decision: [[20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned]]; parity-audit lesson: [[20260723_1910_sheeprl_parity_drift_lives_in_glue]].
- Archive plan: `docs/develop/active/diagnosis/dreamer_sheeprl_parity_2026-07-06/archive_plan_dreamer_v3_nnx.md`; inspection: `docs/develop/active/diagnosis/train_eval_live_paths_2026-07-10/00_combined_inspection.md`.
- Commits: `fd7af84` (archive), `84664e4` (registry), `2a7c6f9` (renames).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 07436809-982c-4ed5-b1fa-e84c030f5fd6` or `python scripts/claude/claude_jsonl_to_md.py <jsonl> /tmp/20260723_1909.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260723_1910_sheeprl_parity_drift_lives_in_glue]] (dreamer_diagnosis, 2026-07-23) — A from-scratch Fable-5 re-audit of both Dreamers vs sheeprl found the math cores
<!-- END BACKLINKS -->
