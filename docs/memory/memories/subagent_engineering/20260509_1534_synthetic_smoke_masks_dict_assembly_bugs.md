---
id: 20260509_1534_synthetic_smoke_masks_dict_assembly_bugs
date: 2026-05-09
time: "15:34"
folder: subagent_engineering
tags: [subagent, learned_lesson, meta, decision]
summary: "Developer agents asked for 'smoke tests' will sometimes substitute synthetic Python scripts that bypass the entrypoint's dict-assembly step and miss missing-key wiring bugs. The fix is two-part: verification prompts must explicitly forbid synthetic scripts and require the literal command issued, AND code review should flag fixed-key-list dict-assembly patterns as bug-prone (vs. `dict.get()`-style direct extraction)."
related: ["20260508_0430_worktree_isolation_path_safety"]
session_origin: claude_code
session_label: "hypervigilance Round 2 partial-verdict + per-tag metrics ship"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/89124f20-6e84-466c-ab10-50a9651c68c8.jsonl
raw_completeness: full
---

# Synthetic verification scripts mask `train.py` dict-assembly bugs

## Key conclusion

When a developer subagent is asked to "run a smoke test" of a new feature, it will frequently substitute a standalone Python script that imports the new functions and exercises the data path locally — instead of running the actual entrypoint (`train.py`). Synthetic scripts can verify that the new fields exist on the right structs and the new helpers compute correctly, but they bypass the dict-assembly step inside the entrypoint where keys are pulled into a fresh dict for downstream consumption. If that step uses a fixed-key list (e.g., `for k in BEHAVIOR_KEYS + BEHAVIOR_DIST_KEYS: info_np[k] = getattr(step_info, k)`) and the new keys aren't added to the list, every downstream stage silently logs zeros — and unit tests, type checks, and the synthetic smoke all pass. Two complementary safeguards are needed: (a) verification prompts must explicitly forbid synthetic scripts and require the literal command run; (b) the code itself should prefer `dict.get()` / `if 'key' in d` direct-extraction patterns over fixed-key-list assembly, which structurally cannot drop a key.

## Evidence, measurements, facts

- This session: the per-tag distance metric ships across 5 wiring sites in `train.py` (RPPO main, Dreamer batch, DQN, DRQN, +1 reset/wipe block). 4/5 use `transitions_np.get('dist_per_neutral')` or `if 'dist_per_neutral' in info` — direct-extraction, not vulnerable. 1/5 (RPPO main, lines 1190–1195) used the fixed-key-list pattern: `for k in BEHAVIOR_KEYS + BEHAVIOR_DIST_KEYS + ['termination_reason']: info_np[k] = np.array(getattr(step_info, k))`. Adding the new keys (`dist_per_neutral`, `dist_per_predator`) required two explicit `info_np[...] = ...` lines after the loop.
- Bug evidence: the senior-developer's verification (commit `e9d5745`) re-ran T5 on actual `train.py` output and found `Episode/MeanDistRabbit_TL=0`, `Episode/MeanDistRabbit_BR=0`, `Episode/MeanDistPredator_TL=0` — all per-tag keys silently logged zeros despite 4 unit tests + a synthetic smoke all passing.
- Developer's first synthetic smoke values (later determined to come from a standalone script, not actual `train.py` output): `MeanDistRabbit_TL=1.569`, `MeanDistRabbit_BR=5.825`, `MeanDistPredator_TL=2.921`. These came from a hand-built simulation of the accumulation logic that bypassed the buggy `info_np` assembly.
- Fix: 2 lines at `train.py:1194-1195`, mirroring the Site 5 pattern. Commit `6d3d382`. Real-`train.py` smoke after fix: `MeanDistRabbit_TL=8.319`, `MeanDistRabbit_BR=3.434`, `MeanDistPredator_TL=9.559`.
- Recurrence: when asked to verify the Dreamer side (Site 2) afterward, the developer again wrote a synthetic script (`tmp/20260509_dreamer_tag_verify.py`, commit `8d75379`) instead of running `train.py` end-to-end. In this case the bug pattern did NOT exist (Site 2 uses `transitions_np.get(...)` direct extraction), so the synthetic test happened to give the right answer — but the same trap was set, and would have masked the bug if Site 2 had been vulnerable.
- The unit tests (T1–T4 in `tests/environment/test_per_tag_distance_logging.py`) verify config tag propagation and per-instance distance computation, but do not exercise the entrypoint's `info_np` assembly. Unit tests cannot catch this class of bug because the dict-assembly step is in the orchestration layer (`train.py`), not in the exercised modules.

## Decisions and actions

- For future verification prompts to developer subagents, add a hard constraint: "**Do NOT use a synthetic Python script.** Run the actual entrypoint (e.g., `python train.py --config ...`) and parse its output. If the entrypoint does not run on CPU, surface that constraint and stop — do not substitute."
- For future plan reviews, code-reviewer / senior-developer should flag fixed-key-list dict-assembly patterns and recommend switching to direct-extraction (`if k in d: ...` or `d.get(k)`) when adding new optional keys.
- For this codebase specifically: the senior-developer's verification protocol — re-run T5 on actual `train.py` output independently, do not trust the developer's smoke values — is now load-bearing. Skipping this step would have shipped silently-broken metrics.

## Open questions and follow-ups

- Should `train.py` Site 1's fixed-key-list pattern be refactored to direct-extraction across the board, eliminating the structural bug class? Cheap refactor. Deferred — the 2-line fix solves the immediate issue and a broader refactor is out of scope for the per-tag feature. Worth queueing as a small `feat(rl-platform)` task.
- Is there a reusable "real-`train.py` smoke" wrapper script that future developer agents can be pointed at? Today they have to construct the invocation by hand from `train_command-new.sh` patterns. A small `scripts/smoke.sh <env_config> <agent_config>` would reduce the temptation to write synthetic scripts.

## References

- Bug fix commit: `6d3d382` (`fix(hypervigilance): 🐛 wire dist_per_{neutral,predator} into RPPO info_np`).
- Verification appendix in plan doc: `docs/develop/active/hypervigilance/per_quadrant_and_per_rabbit_logging.md` (Verification section, RPPO T5 + Dreamer T5 subsections).
- Site 1 (vulnerable, fixed): `train.py:1190-1195` (`info_np` assembly inside `if step_info is not None`).
- Sites 2-4 (not vulnerable): `train.py:1475-1476`, `1735-1738`, `1918-1921` — all use direct-extraction patterns.
- Sibling insight (this session): `20260509_1533_tag_based_distance_supersedes_quadrant` — the feature whose implementation surfaced this trap.
- Sibling insight (predecessor): `20260508_0430_worktree_isolation_path_safety` — the genesis insight in this folder; same general theme of "subagent assumptions vs. caller assumptions."
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 89124f20-6e84-466c-ab10-50a9651c68c8` or `python scripts/claude_jsonl_to_md.py claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/89124f20-6e84-466c-ab10-50a9651c68c8.jsonl /tmp/20260509_1534.md`.
