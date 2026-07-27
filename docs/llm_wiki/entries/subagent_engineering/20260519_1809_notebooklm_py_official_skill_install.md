---
id: 20260519_1809_notebooklm_py_official_skill_install
date: 2026-05-19
time: "18:09"
folder: subagent_engineering
tags: [skill, learned_lesson, decision, meta]
summary: "When a Python/CLI tool has a community-authored Claude Code skill in `.claude/skills/`, check whether the official upstream package already ships its own self-installing skill before keeping the community one. `notebooklm-py` exposes `notebooklm skill install --scope project --target claude` which drops a version-stamped maintained skill into `.claude/skills/`. Prefer that path — upgrades flow automatically with `pip install -U`."
related: []
session_origin: claude_code
session_label: "notebooklm settings — community skill → official notebooklm-py package"
importance: medium
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/88af08ab-5360-43ee-bece-a7673f10b25e.jsonl
raw_completeness: full
---

# Prefer the upstream package's bundled skill over a community-authored Claude Code skill

## Key conclusion

When a CLI tool ships its own Claude Code skill (e.g., `notebooklm skill install`), prefer that over keeping a community-authored wrapper in `.claude/skills/`. The bundled skill is **version-stamped to the package**, **auto-updates** when the user runs `pip install -U`, and is **a thin wrapper over the package's CLI** — so the skill's behaviour stays in sync with the underlying tool by construction. The lesson generalises: before assuming a Claude Code skill must be hand-authored or kept "as-is" from some early community version, look for a `<tool> skill install` / `<tool> agent show` / equivalent self-install entry point in the upstream CLI's help output.

## Evidence, measurements, facts

- **Triggering session**: user asked to install `https://github.com/teng-lin/notebooklm-py` "instead of using the skill" — referring to a community-authored unofficial skill at `.claude/skills/notebooklm/` (22 files, 4.5 MB, included a 137 KB PNG, its own Python `scripts/` dir, and a self-managed `.venv` pattern).
- **Discovery**: `notebooklm --help` revealed two relevant subcommand groups — `notebooklm skill {install, show, status, uninstall}` and `notebooklm agent show {codex|claude}`. The `skill install --scope project --target claude` command drops a single `SKILL.md` file into `.claude/skills/notebooklm/SKILL.md`, version-stamped (`<!-- notebooklm-py v0.4.1 -->`) and authored by the same upstream maintainer as the CLI.
- **Diff scale**: the swap removed 22 files (4242 lines) and added 1 file (729 lines). Commit `bc4b4e9` on branch `v1.4`. The new `SKILL.md` is 28 KB.
- **Architecture difference**: community skill required `python .claude/skills/notebooklm/scripts/run.py <name.py> ...` (its own venv + Playwright install handled inside the skill); official skill calls the `notebooklm` CLI on PATH, which lives in a dedicated venv at `~/.local/notebooklm-py/` (kept separate from the project's `grid_world_pain` conda env to avoid polluting it with Playwright + Chromium).
- **Generalisation**: the same pattern likely exists for other CLI tools (`gh extension`, `wandb`, etc.) — the package owner is the right place to look for the canonical skill, not Claude Code community catalogues.

## Decisions and actions

- Wiped the unofficial community skill at `.claude/skills/notebooklm/` (full directory and the unused-but-documented `~/.claude/skills/notebooklm/data/` state dir — neither was ever authenticated on this machine).
- Installed `notebooklm-py` v0.4.1 in a dedicated venv at `~/.local/notebooklm-py/` (`python -m venv` + `pip install "notebooklm-py[browser]"` + `playwright install chromium`). Kept out of the project conda env on purpose — Playwright + headless Chromium is a developer-tooling dep, not project code.
- Ran `notebooklm skill install --scope project --target claude` to drop the official `SKILL.md` into `.claude/skills/notebooklm/`.
- Updated `.claude/agents/literature-reviewer.md` line 44 (the only line in the agent profile that referenced skill-specific syntax) to note the new package provenance and document the underlying CLI commands (`login` / `list` / `use` / `ask` / `summary`) for cases where the agent wants to bypass the skill abstraction.
- Committed the swap as a single `chore(notebooklm): 🔄 ...` commit (`bc4b4e9`). Per-package self-installing skills do not need to live in dedicated commits — a chore commit with `skill install` as one of the bullets is sufficient.

## Open questions and follow-ups

- User still needs to run `~/.local/notebooklm-py/bin/notebooklm login` once interactively to authenticate against their Google account (browser-visible OAuth). Not automatable.
- Future upgrade flow: `~/.local/notebooklm-py/bin/pip install -U notebooklm-py && notebooklm skill install --scope project --target claude` — the `skill install` is idempotent and overwrites `SKILL.md` with the new version. Worth folding into a tiny `scripts/upgrade_notebooklm.sh` if used more than once.
- Other CLI tools in the project's tooling stack (e.g., `gh`, `wandb`, `graphify`) should be checked for their own bundled-skill entry points the next time a community skill is in play.

## References

- Replaced skill: ex-`.claude/skills/notebooklm/` (community, unofficial) → current `.claude/skills/notebooklm/SKILL.md` (official, from `notebooklm-py` v0.4.1).
- Upstream: https://github.com/teng-lin/notebooklm-py (MIT).
- Touched in this session: `.claude/agents/literature-reviewer.md` (line 44 surgical edit).
- Commit: `bc4b4e9` on `v1.4`.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `claude --resume 88af08ab-5360-43ee-bece-a7673f10b25e`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
