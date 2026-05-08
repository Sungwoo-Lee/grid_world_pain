---
id: 20260508_1826_statusline_jq_ifs_pct
date: 2026-05-08
time: "18:26"
folder: cluster_ops
tags: [meta, learned_lesson, decision]
summary: "Three lessons from configuring ~/.claude/statusline-command.sh on docker-102: jq is not installed (use the conda python3), bash `read` with `IFS=$'\\t'` collapses consecutive tabs (use `|` instead), and Claude Code's statusline JSON already pre-calculates `context_window.used_percentage` so no transcript parsing is needed."
related: []
session_origin: claude_code
session_label: "statusline configuration on docker-102"
importance: medium
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/ab858a54-a914-4747-938a-f45e365f082f.jsonl
raw_completeness: full
---

# Claude Code statusline script — three docker-102 gotchas (jq, IFS, ctx%)

## Key conclusion
The Claude Code `statusline-setup` agent's default script uses `jq` and a tab field-separator, both of which silently fail on this lab container: `jq` is not installed, and bash's `read` collapses adjacent tabs because tab is IFS-whitespace, which shifted `cwd` into the `used_percentage` slot whenever the model had no context info yet. The working recipe is: parse the statusline JSON with `/home/vncuser/miniconda3/bin/python3 -c '...'`, emit fields joined by `|`, and read directly from the pre-calculated `context_window.used_percentage` field that Claude Code already pipes in — no transcript parsing required.

## Evidence, measurements, facts
- `which jq` returns "jq not found" on docker-102 (verified 2026-05-08); `/home/vncuser/miniconda3/bin/python3` is available.
- Claude Code's statusline JSON includes a `context_window` object with: `total_input_tokens`, `total_output_tokens`, `context_window_size`, `used_percentage` (pre-calculated, `null` until first API call), `remaining_percentage`, and a nested `current_usage` block (`input_tokens`, `output_tokens`, `cache_creation_input_tokens`, `cache_read_input_tokens`). `used_percentage` is computed from input tokens only (output tokens excluded). Source: `claude-code-guide` agent confirmation, agentId `af64b56551862d46c`.
- Bash `read` rule (man bash): when `IFS` value is whitespace (space, tab, newline), sequences of those characters in the input are treated as a single delimiter and empty fields are NOT preserved. With `IFS=$'\t'` and input `Claude Opus 4.7\t\t/tmp`, three vars get `Claude Opus 4.7`, `/tmp`, `""` instead of the intended `Claude Opus 4.7`, `""`, `/tmp`.
- Fix verified across four input shapes (model+ctx, model+null-ctx, no-model+no-ctx, ctx=99.9): `vncuser@docker-102:/…/grid_world_pain  [Claude Opus 4.7 | ctx:43%]`, etc. Final script at `/home/vncuser/.claude/statusline-command.sh` ~30 lines.
- Statusline `settings.json` at `/home/vncuser/.claude/settings.json` — `statusLine.command` already pointed at the script; only the script needed fixing.

## Decisions and actions
- Replaced `jq` with inline `python3` via the absolute conda interpreter path. The script does NOT use `python3` from PATH (project rule per CLAUDE.md).
- Switched the field separator from `\t` to `|` and read with `IFS='|'`. `|` is non-whitespace so `read` preserves empty fields between separators.
- Read `context_window.used_percentage` directly; do NOT compute from transcript even though the JSON also exposes the raw token counts.
- Render context % only when `used_percentage` is non-null — gracefully omit the `ctx:N%` block before the first API call instead of showing `ctx:%` or `ctx:0%`.
- Format: left section `\033[01;32muser@host\033[00m:\033[01;34mcwd\033[00m` (matches bash PS1 colors); right section `[model | ctx:N%]` joined by two spaces.

## Open questions and follow-ups
- The `statusline-setup` agent itself defaults to a `jq`-based script and an over-confident "ctx will work" claim; if anyone re-runs `/statusline` on this container, they will hit the same bug. Possible follow-up: brief the agent (via an entry in its `.claude/agents/` profile or a project-level note) that this machine has no `jq` so it should default to `python3` here. Not urgent — a one-time fix.
- `jq` could also just be installed (`apt-get install jq` or `conda install -c conda-forge jq`); deferred because the python3 path is already enforced project-wide.

## References
- Agent referenced: `claude-code-guide` (agentId `af64b56551862d46c`) — confirmed the `context_window` schema and the input-tokens-only formula for `used_percentage`.
- Project rule for python invocation: `CLAUDE.md` → "Conda env" section (use `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python` for project Python; for tooling outside the project env, the base conda python at `/home/vncuser/miniconda3/bin/python3` is acceptable for shell scripts that just need stdlib JSON parsing).
- Statusline script: `/home/vncuser/.claude/statusline-command.sh` (machine-local, not in this repo).
- Statusline settings: `/home/vncuser/.claude/settings.json` → `statusLine.command`.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume ab858a54-a914-4747-938a-f45e365f082f` (re-enter the session) or `python scripts/claude_jsonl_to_md.py claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/ab858a54-a914-4747-938a-f45e365f082f.jsonl /tmp/20260508_1826_statusline_jq_ifs_pct.md` (one-shot markdown view).
