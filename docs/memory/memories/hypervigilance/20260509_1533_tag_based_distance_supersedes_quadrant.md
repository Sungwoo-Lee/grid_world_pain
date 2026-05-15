---
id: 20260509_1533_tag_based_distance_supersedes_quadrant
date: 2026-05-09
time: "15:33"
folder: hypervigilance
tags: [hypervigilance, design, decision, meta]
summary: "Tag-based per-instance distance design (optional `tag: <string>` per entity in YAML, source code geometry-agnostic) supersedes the originally-planned quadrant-hardcoded approach. `MeanDistRabbit_<tag>` vs `MeanDistPredator_<tag>` (same tag) is a more direct disambiguator for quadrant-vs-class avoidance than `QuadrantOccupancy_*` would have been."
related: ["20260509_1532_sameprop_round2_truncated_verdict", "20260509_1534_synthetic_smoke_masks_dict_assembly_bugs"]
session_origin: claude_code
session_label: "hypervigilance Round 2 partial-verdict + per-tag metrics ship"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/89124f20-6e84-466c-ab10-50a9651c68c8.jsonl
raw_completeness: full
---

# Tag-based per-instance distance supersedes quadrant-hardcoded design

## Key conclusion

The first-pass §7 metrics plan committed `EnvParams.neutral_quadrant_idx` computed at config-load via `>= H//2 / >= W//2` and emitted `Episode/MeanDistRabbit_{TL,TR,BL,BR}` plus `Episode/QuadrantOccupancy_{TL,TR,BL,BR}` — all anchored to a 10×10 grid-half-split assumption hard-coded in `core.py` and `config_loader.py`. The user rejected this as too brittle for future experimental layouts and asked for a minimal alternative using "different name or adding tag to animals and predators." The revised design adds an optional `tag: <string>` field per entity instance in YAML, kept off-JIT in a static `tuple[str, ...]` on the env config; source code emits unreduced fixed-shape `info['dist_per_neutral']` / `info['dist_per_predator']` arrays and lets `train.py`'s WandB fan-out attach the tags as suffixes. Geometry never enters source code. As a side effect, `QuadrantOccupancy_*` becomes redundant — `MeanDistRabbit_TL` vs `MeanDistPredator_TL` (both tagged TL when both spawn in the same quadrant) is a more direct disambiguator for "agent avoids predator class" vs "agent avoids predator quadrant" than per-quadrant fractional occupancy could ever be.

## Evidence, measurements, facts

- Original plan (rejected): commit `40bcc1d`, 1007 lines, 6 file changes, `neutral_quadrant_idx: jax.Array` field, 4 fixed-quadrant accumulator counters, hard-coded grid-half-split.
- Revised plan: commit `18bde6f`, 909 lines (−98), same 6 files but no new `EnvParams` field (static Python tuple instead), no quadrant geometry anywhere in source code, symmetric coverage for predators and neutrals.
- Implementation commit: `0a73613` (`feat(hypervigilance): ✨ per-tag per-instance distance logging (Round-2 §7)`).
- T5 smoke output (RPPO, 16 envs, 200 ep, real `train.py`): `MeanDistRabbit_TL=8.319`, `MeanDistRabbit_BR=3.434`, `MeanDistPredator_TL=9.559` in episode 1 — confirms the cross-tag asymmetry is observable from random-init (different quadrants → different distances).
- Same-tag reduction default: **mean** over instances sharing a tag (matches existing aggregated `Episode/MeanDistRabbit` semantics; alternative `min` deferred unless a future config relies on nearest-instance semantics).
- Default tag if YAML omits it: `f"idx{i}"` — fully backwards-compatible; `01-interoNocicept_sameProp.yaml` (Round 1) loads without modification and emits `Episode/MeanDistRabbit_idx0`, `_idx1`, etc.
- Tag character set: `[A-Za-z0-9_-]+`, validated at config-load with `ValueError` on violation (no `/` to avoid WandB nested namespace, no `.` or whitespace).
- 7/7 unit tests pass (T1 tags propagate, T2 default tags, T3 distances per-instance, T4 invalid tag char raises, plus 3 existing).
- The Cell A1 disambiguator: in the Round-2 A1 config, predator is in TL (`tag: TL`) and rabbits are in TL+BR (`tag: TL` for the TL rabbit, `tag: BR` for the BR rabbit). If the agent is class-discriminating, `MeanDistPredator_TL` should be much greater than `MeanDistRabbit_TL` (avoiding the predator but tolerating the rabbit in the same cell). If the agent is quadrant-camping, `MeanDistPredator_TL` ≈ `MeanDistRabbit_TL` (both far because the agent avoids TL entirely). Quadrant occupancy could only show "agent avoids TL" — same answer for both hypotheses.

## Decisions and actions

- Plan supersedes itself in place — original quadrant-hardcoded version (`40bcc1d`) was never implemented; rewritten before any source code was changed.
- Implementation merged on `v1.3` (`0a73613`) + RPPO Site 1 fix (`6d3d382`) + Dreamer Site 2 transition-dict shape verification (`8d75379`).
- `QuadrantOccupancy_*` metric dropped from scope; the per-tag distances cover the same disambiguation more directly.
- Both Round-1 and Round-2 hypervigilance configs now carry `tag` fields (`TL`/`BR`/`full`); developer added them as part of the implementation (a minor scope deviation — configs are normally `experiment-designer`'s domain — but the edits are mechanical and unblock the smoke test, so the deviation was accepted).
- Round 2 re-launch (Round 2.5) can now produce class-vs-quadrant disambiguation directly from the WandB log — no further tooling needed before re-launch.

## Open questions and follow-ups

- Should `hiding_predators` get tags too? Plan deferred this (different YAML surface). If Cell A1 analysis flags hiding-predator-class avoidance as a separate confound, this becomes load-bearing.
- The default-tag fallback `f"idx{i}"` works but is unfriendly at WandB read-time. Consider a one-line warning at config-load when ANY entity instance has the default — not blocking, but a hint to the experiment-designer.
- The "user rejected hard-coded geometry, asked for tags" pattern recurs in this codebase. Other places with implicit grid-half-split assumptions (e.g., quadrant-restricted spawn logic) may eventually merit similar tag-based generalization.

## References

- Plan doc: `docs/develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md` (revised version at HEAD; original was `40bcc1d`).
- Sibling insight (this session): `20260509_1532_sameprop_round2_truncated_verdict` — the experimental finding that motivated this metric design.
- Sibling insight (this session): `20260509_1534_synthetic_smoke_masks_dict_assembly_bugs` — the wiring bug encountered during implementation.
- Predecessor insight: `20260508_1445_sameprop_discriminating_channels` — channel-ranking memo that informs why per-tag distance is the right disambiguator (movement signature is the dominant pre-contact cue, so per-tag distance directly measures whether the agent is responding to it).
- Implementation commit: `0a73613` (`feat(hypervigilance): ✨ per-tag per-instance distance logging (Round-2 §7)`).
- Commits in commit graph: `40bcc1d` (original plan, rejected) → `18bde6f` (revised plan) → `0a73613` (implementation) → `4a3b80d` (impl report) → `6d3d382` (Site 1 fix) → `e9d5745` (verification appendix) → `8d75379` (Dreamer Site 2 verification appendix).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 89124f20-6e84-466c-ab10-50a9651c68c8` or `python scripts/claude_jsonl_to_md.py claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/89124f20-6e84-466c-ab10-50a9651c68c8.jsonl /tmp/20260509_1533.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
