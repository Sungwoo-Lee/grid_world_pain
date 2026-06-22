---
id: 20260622_1704_continual_bm_transition_nameerror_refactor_drift
date: 2026-06-22
time: "17:04"
folder: cluster_ops
tags: [learned_lesson, decision, meta]
summary: "The continual stage-transition crash (NameError: m1_candidates) was introduced by commit 89ade04 (2026-05-12) — a delete-heavy refactor that migrated train.py's behavior-measure accumulators to a BMState object but missed one of six reset sites (the continual stage-transition wipe added the day before). Latent ~40 days because the trigger (continual + behavior_measures + an actual stage transition) was exactly the path the continual-learning review left UNVERIFIED. Fixed with the canonical per-env _bm_reset_env loop + a fast regression smoke."
related: ["20260529_1826_lazy_import_schema_drift_first_call_crash"]
session_origin: claude_code
session_label: "v3.0 config overhaul -> basic curriculum -> continual schedule + BMState stage-transition bug fix"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/96e71c7b-dc03-44c9-a98c-1c2acc86e0d9.jsonl
raw_completeness: full
---

# Continual stage-transition NameError: orphaned flat-array reference from a delete-heavy refactor

## Key conclusion
The first real curriculum (continual-learning) run crashed at its first stage boundary with `NameError: name 'm1_candidates' is not defined` (`train.py` stage-transition block). Root cause: commit **89ade04** (2026-05-12, "delegate BM accumulation to shared src/behavior modules") was a delete-heavy refactor (-264/+25 on train.py) that moved every behavior-measure accumulator from flat per-env arrays (`m1_candidates`, `m2_onsets`, ...) into a `BMState` object (`_bm_state`) and updated 5 of 6 reset/use sites — but **missed the continual stage-transition wipe block** (added the day before by `bf4f446`). That one orphaned block kept referencing arrays that no longer exist in scope. The bug stayed latent ~40 days because it only fires when **continual mode + behavior_measures enabled + an actual stage transition** all coincide — the exact path the continual-learning review had explicitly left UNVERIFIED. Fix: replace the stale 16-line block with `for i in range(num_envs): _bm_reset_env(i)` (the canonical per-env BMState reset already used at episode end), plus a fast regression test.

## Evidence, measurements, facts
- **Commit timeline** (git blame + `git log -S`): `bf4f446` (2026-05-11 04:04, "wire M1/M2/M5 online accumulators at all 5 training sites") added the flat arrays AND the stage-transition wipe — correct at the time. `c7f12eb` (2026-05-12 18:34:43) added `src/behavior/accumulators.py` (`BMState`, `make_bm_state`, `bm_reset_env`). `89ade04` (2026-05-12 18:34:**51**, 8 s later, same session) migrated train.py to `_bm_state` and deleted the flat arrays everywhere **except** the stage-transition wipe.
- **Crash**: `train.py:1223`, `m1_candidates[:, :] = 0`, fired at `[STAGE] 0:00-static_predator_5x5 -> 1:01-slow_predator_5x5 at ep=1000010` on the basic-curriculum continual run. Stage 0 trained the full 1.0M episodes fine; the transition logic itself worked; only the accumulator wipe crashed.
- **Why latent**: the continual review (`docs/develop/active/continual_learning/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md`) marked Ckpt 5/6/8/9/10/11 as "not verified — require full-length runs … left for the user's first real curriculum run." Ckpt 5 was literally "observing behavioral metrics across transition." Nothing exercised the path until this run.
- **Fix verified**: `bm_reset_env(bm, i)` resets all 24 per-env fields the old block reset (per-class + per-tag M1/M2/M5 counters, `*_age`/`*_idx` to -1, `m2_in_bush_seen`/`m_prev_threat_in_R*` to False, `m1_steps_since_eat`). Regression test `tests/training/test_continual_bm_transition.py` runs a 2-stage `num_envs=2 --num-steps 10` CPU schedule with `boundaries=[6,12]` so a transition fires in ~20 s; confirmed RED on pre-fix (NameError) → GREEN after (1 passed). Single-config training unaffected.
- Curriculum relaunched after the fix (WandB `mpql5i25`, node 106 GPU 0); the ep-1.0M transition is the real proof point.

## Decisions and actions
- **Fix**: stage-transition BM wipe now calls the canonical `_bm_reset_env(i)` per env (no hard-coded field names to drift). This keeps the site in sync with the episode-end reset by construction.
- **Prevention 1 (refactor hygiene)**: after a delete-heavy refactor that removes a symbol, `git grep <symbol>` the whole file/repo — a `git grep m1_candidates train.py` would have shown the orphan immediately.
- **Prevention 2 (known-unverified paths)**: a rarely-run code path flagged "needs a full-length run to verify" should get a **fast smoke** instead (this one: 2 stages, `num_envs=1`, `boundary=6`, behavior_measures on → fires a transition in seconds). Converted review Ckpt 5 into exactly that test.
- Same failure class as [[20260529_1826_lazy_import_schema_drift_first_call_crash]] (schema mismatch that only surfaces at the first call site of a rarely-run path).

## Open questions and follow-ups
- **Commit hygiene**: the fix landed in `97e689c`, whose message ("render scalar lists inline in saved config YAMLs") describes a *parallel session's* config refactor — the developer's `git add train.py` swept up the parallel session's in-flight train.py edits (the git-lock parallel-session contamination pattern). NOT untangled: rewriting history is unsafe with active parallel sessions on the NAS. The fix code is correct and committed; the test is cleanly in `f7ba033`.
- Continual review still has Ckpt 6/9/10/11 (DreamerV3 buffer clear, save+resume of `current_stage`) unverified — candidates for the same fast-smoke treatment.

## References
- Culprit: commit `89ade04`; introducer of the wipe: `bf4f446`; BMState module: `c7f12eb`. Fix: `97e689c` (train.py hunk; message misattributed). Test: `f7ba033`.
- Review that flagged the gap: [`CONTINUAL_LEARNING_CONFIG_SCHEDULE`](../../../develop/active/continual_learning/CONTINUAL_LEARNING_CONFIG_SCHEDULE.md) (Ckpt 5/8/10 unverified).
- Same failure class: [[20260529_1826_lazy_import_schema_drift_first_call_crash]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 96e71c7b-dc03-44c9-a98c-1c2acc86e0d9` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/20260622_1704_continual_bm_transition_nameerror_refactor_drift.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260623_0144_inactive_resource_slots_revive_respawn]] (env_entities, 2026-06-23) — Blocker: a per-episode activation mask must be threaded through the respawn/rege
<!-- END BACKLINKS -->
