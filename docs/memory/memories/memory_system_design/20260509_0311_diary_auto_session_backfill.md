---
id: 20260509_0311_diary_auto_session_backfill
date: 2026-05-09
time: "03:11"
folder: memory_system_design
tags: [meta, design, decision, memory]
summary: "`scripts/diary_append.py` now lazy-creates a Sessions row in the daily diary when an event arrives from a session that never called `session-start`. The full UUID is resolved by scanning `~/.claude/projects/<encoded>/<UUID>.jsonl` (local) and `claude_data/.claude/projects/<encoded>/<UUID>.jsonl` (NAS-synced). Convention preserved: full UUID only in the Sessions table; 8-char prefix in Events / Training runs. The Sessions table now self-heals — every session that produces any event eventually gets a copy-paste-ready full-UUID anchor."
related: ["20260508_0429_memorize_skill_design_and_ship", "20260508_0447_recall_skill_design_and_ship", "20260509_0309_cluster_py_consolidation"]
session_origin: claude_code
session_label: "container_image_rebuild_evaaa_to_episode_v1_2026-05-08"
importance: medium
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/b40582ae-43df-48c4-b3f0-03b539bebae8.jsonl
raw_completeness: full
---

# Diary auto-backfill — Sessions table self-heals when sessions miss `session-start`

## Key conclusion
The diary's design assumed every Claude session would call `session-start` at the beginning of work, populating the Sessions table with a full-UUID anchor row that (a) is the only place to find the full UUID for `claude --resume <UUID>` and (b) cross-references back from any 8-char-prefix entry in Events / Training runs. That assumption was violated on 2026-05-08: 5 of 6 active sessions produced events without ever calling `session-start`, so their full UUIDs were nowhere in the diary. After a one-time backfill, the script was patched to self-heal: `cmd_event` and `cmd_training_start` now call a new `ensure_session_row(text, time, session_token)` helper before inserting their own row. The helper extracts the parent prefix from `<prefix>` or `<prefix>/<role>`, resolves it to a full UUID by scanning Claude Code's project directories (local first, then synced), and inserts a placeholder Sessions row with label `(auto — session-start was not called)` and summary `Auto-created on first event.` if no row containing that UUID exists. Idempotent on repeat events from the same session. Skips silently when the UUID can't be resolved (brand-new session, JSONL not yet flushed) or the token is malformed.

## Evidence, measurements, facts
- 2026-05-08 diary inventory before the fix: Sessions table had 1 row (f3ab7f37, the only session that called `session-start`). Events table referenced 6 distinct session prefixes (f3ab7f37, c7ee226b, 34028a88, b12b1fea, b40582ae, ab858a54). 5 of 6 had no Sessions row.
- One-shot backfill `/tmp/fix_diary_sessions.py`: rolled back 48 over-eager full-UUID cells (an earlier mistake on my part where I had expanded prefixes in Events and Training runs too) and added 5 missing Sessions rows in 2026-05-08.md (`b12b1fea` continuation labeled `(prev-day 21:34) | 15:13`; the other 4 with `(open)` end time and the actual earliest event time).
- Going-forward fix in `scripts/diary_append.py`:
  - New helper `resolve_full_uuid(prefix)` scans `~/.claude/projects/<encoded>/<UUID>.jsonl` first, then `claude_data/.claude/projects/<encoded>/<UUID>.jsonl`. Returns the first 36-char-or-longer match starting with the prefix.
  - New helper `ensure_session_row(text, time, session_token)` extracts `prefix = session_token.split("/")[0]`, validates it's 8 hex chars, calls `resolve_full_uuid`, checks for an existing row by full-UUID substring search, and inserts a placeholder via the existing `insert_row_at_top` if missing.
  - `cmd_event(args, type_label)` (handles `implemented` / `verified` / `insight`) calls `text = ensure_session_row(text, time, sess)` immediately after `sess = resolve_session(args.session)` and before the row is built.
  - `cmd_training_start(args)` calls `ensure_session_row` at the same point.
- Verification: 4 dry-run cases pass — anchored session (f3ab7f37) → no-op, un-anchored session (random UUID) → row added, malformed tokens (`''`, `unknown`, `xyz`, `short`, `NotHexAtAll`) → all skipped, and end-to-end with a temporary `2099-12-31.md` and `CLAUDE_CODE_SESSION_ID` set to a known UUID → Sessions row + Events row appear together with consistent UUIDs.
- The placeholder row reads `| <time> | (open) | <full-UUID> | (auto — session-start was not called) | Auto-created on first event. |  |` so any human scanning the table can immediately tell which sessions were properly anchored vs. lazily backfilled. Label/summary can be edited by hand later if a session deserves a richer description.
- Convention reaffirmed: full UUIDs ONLY in the Sessions table; Events and Training runs continue to use 8-char prefixes (compactness for many rows). The user's earlier correction (when I had over-applied full UUIDs to all tables) made this convention explicit.

## Decisions and actions
- Edited `/media/nas01/projects/Interoceptive-AI/grid_world_pain/scripts/diary_append.py`:
  - Added `_PROJECT_ENCODED` constant for the `~/.claude/projects/<encoded>` directory name.
  - Added `resolve_full_uuid(prefix)` function.
  - Added `ensure_session_row(text, time, session_token)` function.
  - Patched `cmd_event` to call `ensure_session_row` before row insertion.
  - Patched `cmd_training_start` to call `ensure_session_row` before row insertion.
- Did NOT change the convention for what goes in which table — the user explicitly clarified that Sessions=full-UUID, Events/Training=8-char prefix is the right design.
- Did NOT change the agent-profile `:0:8` patterns; sub-agents still pass `--session "${CLAUDE_CODE_SESSION_ID:0:8}/<role>"` and the diary script uses that verbatim for Events/Training (correct), but extracts the parent prefix to anchor the Sessions table (the new behavior).

## Open questions and follow-ups
- Deduplication is by full-UUID substring search, not exact column match. If the full UUID happens to appear in another column (e.g. as part of a link path or a free-form Notes line), `ensure_session_row` would falsely conclude the session is anchored. Low probability but worth knowing; could tighten to "in column 3 of the Sessions table" if it ever bites.
- A brand-new session whose JSONL hasn't been flushed to disk yet (the file is opened on first user message) would silently skip the row creation; the next event from the same session re-tries. In practice the JSONL is flushed within milliseconds of the session opening, so this race is academic.
- For `note` events (the `cmd_note` subcommand), the auto-backfill is NOT triggered because notes don't have a session column — they're free-form bullets. This is intentional but worth flagging if `cmd_note` ever grows a session field.

## References
- File patched: `scripts/diary_append.py` (helpers `resolve_full_uuid` and `ensure_session_row`; calls inserted in `cmd_event` and `cmd_training_start`).
- Companion sibling insights: `20260508_0429_memorize_skill_design_and_ship` (the broader memory-system context), `20260508_0447_recall_skill_design_and_ship`.
- Related session-context insight: `20260509_0309_cluster_py_consolidation` (cluster.py's `rollout` orchestrator now reuses a single password across stages — same self-healing principle applied to a different kind of repeated state).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume b40582ae-43df-48c4-b3f0-03b539bebae8` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).
