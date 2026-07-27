---
id: 20260508_0447_recall_skill_design_and_ship
date: 2026-05-08
time: 04:47
folder: wiki_system_design
tags: [wiki, design, decision, skill]
summary: "Designed and shipped the /wiki-read Claude Code skill (.claude/skills/wiki-read/), completing the /wiki-write ↔ /wiki-read capture-recall pair; default behavior is a 10-most-recent flat reverse-chrono list, overriding the operating manual §8 time-grouped default in favor of a simpler default."
related: ["20260508_0315_claude_memory_system_genesis", "20260508_0429_memorize_skill_design_and_ship"]
session_origin: claude_code
session_label: "wiki-read skill rollout"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/f3ab7f37-218c-463b-ba24-e555d496dec1.jsonl
raw_completeness: full
---

# `/wiki-read` skill design and ship

## Key conclusion

Built `.claude/skills/wiki-read/` and shipped at commit `3705ece`. The LLM Wiki is now bidirectional: `/wiki-write` writes, `/wiki-read` reads. The skill defaults to a flat reverse-chronological list of the 10 most recent insights (no time bucketing) instead of the operating manual §8's time-grouped format — the user explicitly chose the simpler default at design time. Time-grouped output (Today / Yesterday / This week / Earlier) remains available when the user's phrasing includes a range hint ("today", "yesterday", "this week", "earlier", "all"). Technical mode (raw indexes, folder names, tag dictionary) triggered by "show ROOT_INDEX", "list folders", "audit", "tag dictionary".

## Evidence, measurements, facts

- **Skill location**: `.claude/skills/wiki-read/SKILL.md` (project-local, ships with the repo).
- **Decisions captured at design-time** (locked via AskUserQuestion):
  - Name: `wiki-read` (mirrors the operating manual's vocabulary; pairs cleanly with `/wiki-write`).
  - Default scope: **10 most recent flat list, no time grouping** — explicit override of operating-manual §8's time-grouped default. Time grouping is opt-in via range-hint phrases.
  - Mode coverage: **both natural-language and technical** (technical triggered by jargon keywords per §8 mode-selection table).
  - Drill-down: **yes** — after listing, the skill offers "Reply with the ID, the date+time, or a phrase." On follow-up, opens the specific insight (Key conclusion + Decisions + Open questions; Evidence/References on "more"). Matches the lazy-load §9 philosophy: read L2 indexes for listing, L3 for drill-down, L4 raw archive only on explicit user confirmation.
- **First-invocation validation**: `/wiki-read` produced the expected 3-insight flat list with folder definitions in italics (e.g., *"Subagent + worktree usage gotchas"*) instead of raw folder names — confirming the natural-language mode hides the L0–L4 / frontmatter / fragmentation jargon per the operating manual §8 "do not expose" rules.
- **Lazy-load discipline confirmed**: the skill's listing path reads only `ROOT_INDEX.md` + each topic's `_topic_index.md`. Individual insight files are read only on drill-down. Raw archives at `_archive/raw_conversations/` are never loaded by `/wiki-read` without explicit user confirmation + the §10 large-file warning protocol.
- **Hard rules adopted from `/wiki-write`**: repo-relative paths only, never read `~/.claude/.../memory/MEMORY.md` (different layer, harness's job), all output English, no emojis unless asked.
- **No new tags introduced**: reused `memory`, `design`, `decision`, `skill` from the active set.

## Decisions and actions

- Shipped `.claude/skills/wiki-read/` with `SKILL.md` and `evals/evals.json` (3 eval prompts: default-flat, time-grouped-yesterday, technical-query). No eval-loop run this iteration — user opted to ship directly, matching the pattern from the `/wiki-write` rollout.
- The skill's default behavior diverges from the operating manual §8 (time-grouped) in favor of flat-list. Operating manual §8 still describes the time-grouped format faithfully — it is invoked when the user gives a range hint. No edit to §8 needed.
- Companion to `/wiki-write`: now `/wiki-write` (capture) and `/wiki-read` (read) are the two slash commands that frame the LLM Wiki. Both are in `.claude/skills/` and registered in the available-skills list.

## Open questions and follow-ups

- Should `/wiki-read` accept structured arguments (e.g., `/wiki-read topic=wiki_system_design`, `/wiki-read n=20`)? Currently the skill parses natural-language hints in the prompt; a structured-args path would be more deterministic but adds surface. Defer until friction emerges.
- Should there be an "audit" subcommand of `/wiki-read` that proactively checks the §6 fragmentation thresholds (≥10 folders or 30 days since last audit) and surfaces a merge proposal? Currently `/wiki-read` only reports audit state when the user asks; it does not auto-trigger an audit. Could be its own skill (`/audit-memory`) or a flag.
- The pre-existing `regen_dev_index.py` validation failure (`per_entity_avoidance_logging.md` carries `status: implemented`) is still pending a separate fix — unchanged from the prior insight.

## References

- Skill: `.claude/skills/wiki-read/SKILL.md`
- Eval definitions: `.claude/skills/wiki-read/evals/evals.json`
- Operating manual: `docs/llm_wiki/CLAUDE.md` §8 (natural-language recall format), §9 (lazy-load), §10 (large-file warning protocol)
- Companion skill: `.claude/skills/wiki-write/SKILL.md`
- Paired insight: `20260508_0429_memorize_skill_design_and_ship` (write counterpart)
- Genesis: `20260508_0315_claude_memory_system_genesis`
- Ship commit: `3705ece`
- `raw_source` points at the synced JSONL on NAS (`claude_data/.claude/projects/.../<UUID>.jsonl`). Push via `./sync-agent-data.sh claude push` to update the NAS copy; on another node, `./sync-agent-data.sh claude pull` first, then `claude --resume <UUID>` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot view). Backfilled (originally `none`) when /wiki-write Step 7 design changed to point at the synced JSONL.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
