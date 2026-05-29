---
id: 20260529_1826_lazy_import_schema_drift_first_call_crash
date: 2026-05-29
time: "18:26"
folder: dreamer_diagnosis
tags: [dreamer, learned_lesson, refutation, meta, design]
summary: "4 dreamer-srl cells crashed at episode 10000 (first checkpoint) with AttributeError: 'EnvState' object has no attribute 'animal_pos'. Root cause: src/utils/eval_recording.py is lazy-imported AT first checkpoint, so a parallel session that pulled the CP6 EnvState unification (predator_pos / neutral_pos → animal_pos) mid-training silently re-wrote the lazy-import target. The interpreter had the OLD EnvState class cached in memory but loaded the NEW eval_recording.py from disk — schema mismatch surfaces only at the lazy-call site, ~7h into training."
related: ["20260518_1737_wandb_post_crash_frozen_state_misread", "20260528_1647_bg_isolation_subagent_bypass"]
session_origin: claude_code
session_label: "dreamer-srl v2 post-fix relaunch + WandB/config-save parity"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/7962c4de-7ac9-4c9a-9958-c22a36fd45c7.jsonl
raw_completeness: full
---

# Lazy import + mid-training branch update = schema-drift crash at first call site

## Key conclusion

Four dreamer-srl training cells launched 2026-05-28 03:30 KST all crashed at episode 10000 — the first checkpoint trigger, ~7h into the run — with `AttributeError: 'EnvState' object has no attribute 'animal_pos'` raised inside `src/utils/eval_recording.py:_snapshot_state`. Diagnosis: `eval_recording.py` is imported lazily — `from src.utils.eval_recording import EpisodeRecorder, write_run_meta` lives INSIDE the function body at `src/algorithms/dreamer_srl/eval.py:68`, not at module top. Between training start and first checkpoint, a parallel session pulled the CP6 EnvState unification (predator_pos / neutral_pos → unified animal_pos). The interpreter had the OLD `EnvState` class in memory (no `animal_pos` attr); the lazy import at ~10:42 KST loaded the NEW `eval_recording.py` from disk (which reads `state.animal_pos`). The schema mismatch surfaced ONLY at the first lazy-call site, not on training start — silent for 7 hours, then 4 cells died simultaneously. Not a real code bug; a disk-vs-memory drift hazard induced by lazy imports interacting with mid-training git pulls.

## Evidence, measurements, facts

- Crash signature (identical across all 4 cells): `AttributeError: 'EnvState' object has no attribute 'animal_pos'` in `src/utils/eval_recording.py:37` at `np.asarray(state.animal_pos)`.
- Trigger: episode 10000 = first checkpoint = first call to `eval()` = first call to the lazy import.
- Lazy-import call site: `src/algorithms/dreamer_srl/eval.py:68` — `from src.utils.eval_recording import ...` inside function body.
- CP6 unification: split `predator_pos`/`neutral_pos` → single `animal_pos: jnp.ndarray  # [N, 2]` at `src/environment/state.py:44` (commits 2026-05-28).
- Renderers slice by class via `select_by_class` — works against post-CP6 schema only.
- Fix (passive — for the 4 crashed cells): relaunch on fully-unified v2.0 main. Fresh interpreter has consistent post-CP6 EnvState; lazy import loads the matching `eval_recording.py`. The 4 post-fix cells (w9as1qe3, x0ahp61j, jx3obafg, h7aa37na) launched 2026-05-29 02:09 KST and are not exposed to the mid-training drift.

## Decisions and actions

- For the 4 crashed cells: no code change — the drift was external (parallel session's commit) and the schema is now consistent on disk.
- Lesson for future: avoid lazy imports for modules that read attributes off classes already imported at top-level — the disk-vs-memory consistency assumption breaks under concurrent git pulls.
- Alternative mitigations (not adopted, but considered): (a) hoist `eval_recording.py` import to module top of `eval.py` so the schema lands at training-start (would have failed loudly at boot instead of 7h in — better failure mode); (b) freeze the working tree at run-start by checking out a SHA and using a worktree per run.
- Stronger lesson for cross-session work: parallel sessions pulling on `v1.3` or `main` while a long training is mid-flight on the same checkout can mutate any lazy-loaded files. The "always work in a worktree" pattern documented in [[20260528_1647_bg_isolation_subagent_bypass]] applies here too — but for TRAINING (not just edits).

## Open questions and follow-ups

- Is there a way to make `eval_recording.py` import the class explicitly (not duck-typed) so the mismatch becomes a clearer error message ("schema version X but model expects Y") instead of a generic AttributeError? Lower priority.
- Audit other lazy imports under `src/algorithms/dreamer_srl/` for the same hazard pattern — anywhere a function-body import reads class attributes off something defined at top-level.

## References

- Diagnosis trace was inline in this session (no separate plan doc).
- Related insight: [[20260528_1647_bg_isolation_subagent_bypass]] (parallel-session edit isolation pattern — same family). [[20260518_1737_wandb_post_crash_frozen_state_misread]] (also a "silent-for-N-hours then surfaces" failure mode, different root cause).
- Source paths: `src/utils/eval_recording.py:37`, `src/algorithms/dreamer_srl/eval.py:68`, `src/environment/state.py:44`.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 7962c4de-7ac9-4c9a-9958-c22a36fd45c7` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
