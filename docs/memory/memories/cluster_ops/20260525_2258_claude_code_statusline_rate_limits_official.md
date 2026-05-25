---
id: 20260525_2258_claude_code_statusline_rate_limits_official
date: 2026-05-25
time: "22:58"
folder: cluster_ops
tags: [meta, learned_lesson, decision]
summary: "Claude Code's statusline JSON now exposes `rate_limits.five_hour.used_percentage`, `.resets_at`, and the same `seven_day` fields — the official, no-auth path to showing the same Pro/Max usage numbers as claude.ai/settings/usage. Obsoletes ccusage / accessToken-polling for usage display."
related: ["20260508_1826_statusline_jq_ifs_pct"]
session_origin: claude_code
session_label: "statusline rate_limits + 2-line layout + refreshInterval"
importance: medium
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/ce2e1549-b2e8-4b67-8de0-694c7a814b2b.jsonl
raw_completeness: full
---

# Claude Code statusline — `rate_limits.*` is the official Pro/Max usage path

## Key conclusion
Claude Code now pipes `rate_limits.five_hour.used_percentage`, `rate_limits.five_hour.resets_at`, and the matching `seven_day` fields directly into the statusline script's stdin JSON. These are the same numbers shown on `claude.ai/settings/usage` — no auth, no cookie, no scraping, no Node tool required. This obsoletes (for display purposes) three earlier community approaches: local transcript scanning with a static price table, the `ccusage` Node tool, and accessToken-polling against an undocumented Anthropic endpoint (e.g., the jtbr gist). Pair it with the `refreshInterval` settings.json field so the reset-countdown text ticks live during idle periods — the rate-limit percentages themselves only change once per assistant round-trip, because they are populated from the most recent API response.

## Evidence, measurements, facts
- Official docs (https://code.claude.com/docs/en/statusline, "Available data" table) document `rate_limits.five_hour.{used_percentage, resets_at}` and `rate_limits.seven_day.{used_percentage, resets_at}` as fields in the stdin JSON. `used_percentage` is 0–100; `resets_at` is Unix epoch seconds.
- Same docs ("When it updates" section): the statusline script re-runs after each new assistant message, after `/compact`, when permission mode changes, and when vim mode toggles. Updates debounced at 300 ms. The optional `refreshInterval` field re-runs the script every N seconds in addition to events (min 1).
- Field presence: `rate_limits` appears **only** for Claude.ai subscribers (Pro/Max) and **only after the first API response in the session**. Each of `five_hour` and `seven_day` may be independently absent. Defensive handling: omit line 2 entirely when both are missing.
- Refresh semantics: even with `refreshInterval: 30`, the rate-limit **percentages** don't move between API turns (they come from the last server response); only the **countdown text** (`resets ... in Xh`) re-renders against current wall-clock.
- Alternatives evaluated and rejected for the usage-display purpose:
  - **Local transcript scan** (built `~/.claude/usage_helper.py`, deleted) — aggregated `~/.claude/projects/**/*.jsonl` `usage` blocks, applied a static Opus/Sonnet/Haiku price table. Token counts exact, $ approximate, no rate-limit % (no server data).
  - **ccusage** (https://ccusage.com/guide/statusline) — Node tool; requires npm/node install; duplicates what Claude Code now does natively.
  - **jtbr gist** (https://gist.github.com/jtbr/4f99671d1cee06b44106456958caba8b) — polls an undocumented Anthropic endpoint with stored accessToken; token expires silently; 180 s cache.
- Community color convention (jtbr gist, ohugonnot/claude-code-statusline): green <45 %, yellow 45–70 %, bright-red >70 %. Implemented in `~/.claude/statusline_render.py`.
- Final rendered output (2-line):
  ```
  vncuser@docker-102:/media/nas01/projects/Interoceptive-AI/grid_world_pain  [Opus 4.7 | ctx:42% | sess:$1.23 | +17/-3 | 3m05s]
  5h: 23.5% (resets 03:23, in 4h)    wk: 67.2% (resets Thu 22:53, in 72h)
  ```

## Decisions and actions
- Wrote `~/.claude/statusline_render.py` — single Python pass over stdin JSON, emits 3 stdout lines (line1_right content, line2 with ANSI-coloured rate_limits, cwd).
- Wrote thin `~/.claude/statusline-command.sh` wrapper — adds the bash-style left half (`user@host:cwd` in PS1 colors), reads helper's 3 lines via successive `IFS= read -r`.
- Added `"refreshInterval": 30` to `~/.claude/settings.json` under the `statusLine` block — so the reset-countdown text ticks down during idle (e.g., while waiting on background subagents).
- Deleted `~/.claude/usage_helper.py` and `/tmp/claude_usage_cache.json` (the dead local-aggregator path).
- Extends [[20260508_1826_statusline_jq_ifs_pct]]: the earlier insight established "use the pre-calculated JSON fields, don't compute"; this one applies the same pattern to the newly-available rate-limit fields.

## Open questions and follow-ups
- None. The official path is stable; future statusline work can build on the same JSON contract.

## References
- Official statusline docs: https://code.claude.com/docs/en/statusline (Available data table + "When it updates" + `refreshInterval` field).
- Community guide with color convention and progress-bar mechanics: https://gist.github.com/jtbr/4f99671d1cee06b44106456958caba8b
- Full-featured community implementation using the same fields: https://github.com/ohugonnot/claude-code-statusline
- Node-tool alternative (now redundant for display): https://ccusage.com/guide/statusline
- Original feature-request issue that landed `rate_limits` in the JSON: https://github.com/anthropics/claude-code/issues/20636
- Local files (machine-local, not in this repo): `~/.claude/statusline-command.sh`, `~/.claude/statusline_render.py`, `~/.claude/settings.json`. These are user-specific and live outside the project tree; a fresh clone on a new node would re-create them from this insight.
- Predecessor insight: [[20260508_1826_statusline_jq_ifs_pct]] — same pattern (prefer pre-calculated JSON fields over self-computation).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume ce2e1549-b2e8-4b67-8de0-694c7a814b2b` (re-enter the session) or `python scripts/claude_jsonl_to_md.py claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/ce2e1549-b2e8-4b67-8de0-694c7a814b2b.jsonl /tmp/20260525_2258_claude_code_statusline_rate_limits_official.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
