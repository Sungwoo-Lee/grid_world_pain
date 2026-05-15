---
id: 20260508_0315_claude_memory_system_genesis
date: 2026-05-08
time: "03:15"
folder: memory_system_design
tags: [memory, design, decision, meta]
summary: "The in-repo `docs/memory/` layer coexists with the built-in auto-memory MEMORY.md; the two layers split by insight density — short typed rules stay in the built-in layer, multi-section session insights go in `docs/memory/`. Raw conversation archives are local-only and gitignored."
related: []
session_origin: claude_code
session_label: "claude-memory genesis"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/f3ab7f37-218c-463b-ba24-e555d496dec1.jsonl
raw_completeness: full
---

# Claude session-memory system: design decisions and genesis

## Key conclusion

The in-repo `docs/memory/` layer is created alongside (not replacing) the Claude Code built-in auto-memory at `~/.claude/.../memory/MEMORY.md`. The two layers split by **insight density**: short typed rules that any agent must obey on every invocation stay in the built-in layer; multi-section session insights with rationale, decisions, follow-ups, and raw-conversation traceability go in `docs/memory/`. Raw conversation archives are local-only (gitignored by default) — the insight file (with `raw_source` field) is committed; the archive itself is not.

## Evidence, measurements, facts

- Design plan: `docs/develop/active/meta/claude_memory_system_design.md` (status: active, created 2026-05-08).
- Reference implementation inspected: `tmp/Claude-memory/` (Korean operating manual, different project — Mac mini MCP/Cloudflare setup). Ported design to English; dropped attachments system and MCP-tool table.
- Four user decisions recorded in the plan (resolved 2026-05-08):
  - **Coexist** with built-in MEMORY.md (not replace it).
  - **Location**: `docs/memory/` at repo root.
  - **Scope**: core insight files committed; raw archives local-only.
  - **Language**: English throughout (not Korean as in the reference).
- Five open questions resolved by the user before implementation:
  1. **Helper script**: ship `scripts/claude_jsonl_to_md.py` now — JSONL → markdown one-shot for "save full raw" capture path.
  2. **Archive git policy**: `_archive/raw_conversations/*.md` gitignored; `.gitkeep` tracked. Archives are local-only.
  3. **Slash command `/memorize`**: no custom Claude Code slash-command wiring; treat `/memorize` and other phrases as plain natural-language trigger phrases.
  4. **Seed timing**: seed AND immediately write first insight (this file) to validate the full capture pipeline end-to-end.
  5. **Built-in MEMORY.md cross-link**: pointer lives only in project root `CLAUDE.md` (option A). No cross-link in built-in MEMORY.md to avoid drift.
- Five existing built-in MEMORY.md entries confirmed to stay put (all fit "short typed rule another agent must obey on every invocation"):
  - `feedback_training_runner_inputs.md`
  - `feedback_launch_manifest.md`
  - `feedback_runner_post_launch_pgrep.md`
  - `feedback_runner_cifs_bypass.md`
  - `feedback_runner_node_env_preflight.md`

## Decisions and actions

- Created `docs/memory/` skeleton at repo root: `CLAUDE.md`, `ROOT_INDEX.md`, `memories/_global_tags.md`, `TEMPLATES/insight.md`, `_archive/raw_conversations/.gitkeep`, `.trash/.gitkeep`, `memories/memory_system_design/_topic_index.md`, and this genesis insight file.
- Created `scripts/claude_jsonl_to_md.py` — converts Claude Code transcript JSONL to chronological markdown export for the raw-archive capture path.
- Appended "Session memory" section to project root `CLAUDE.md` routing future-Claude to `docs/memory/CLAUDE.md` and `ROOT_INDEX.md`.
- Added `.gitignore` rules: `_archive/raw_conversations/*.md` (with `!.gitkeep` exception) and `.trash/*` (with `!.gitkeep` exception).
- `ROOT_INDEX.md` populated with `memory_system_design` folder row (1 insight, top tags: `[memory, design, decision]`).
- `_global_tags.md` populated with initial active tags: `memory`, `design`, `decision`, `meta`.

## Open questions and follow-ups

None — the plan resolved all five open questions before implementation.

## References

- Plan doc: `docs/develop/active/meta/claude_memory_system_design.md`
- Project root pointer: `CLAUDE.md` (Session memory section)
- Built-in auto-memory: `~/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/memory/MEMORY.md`
- Helper script: `scripts/claude_jsonl_to_md.py`
- Why a new folder: first insight in this layer; "memory_system_design" mirrors the reference layout's same-named topic and is the natural home for any future insight about this layer's own evolution. No existing folder to match against (this is the first folder).
- `raw_source` points at the synced JSONL on NAS (`claude_data/.claude/projects/.../<UUID>.jsonl`). Push via `./sync-agent-data.sh claude push` to update the NAS copy; on another node, `./sync-agent-data.sh claude pull` first, then `claude --resume <UUID>` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot view). Backfilled (originally `none`) when /memorize Step 7 design changed to point at the synced JSONL.
