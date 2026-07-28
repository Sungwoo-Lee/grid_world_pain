---
id: 20260728_1643_bulk_rename_component_path_and_domain_term_traps
date: 2026-07-28
time: "16:43"
folder: wiki_system_design
tags: [wiki, learned_lesson, meta, design]
summary: "Two traps in a repo-wide mechanical rename, both of which a naive sed would have gotten wrong. (1) Scripts build paths from COMPONENTS — ROOT / 'docs' / 'memory' — which a search for the string 'docs/memory' cannot see; all four wiki tools would have silently pointed at a directory that no longer existed. (2) A prose sweep must exclude domain-term corpora: docs/project/references/ is full of 'episodic memory' / 'working memory' from the literature. Fix: path rules everywhere, prose rules guarded, built-in-memory references frozen behind sentinels."
related: ["20260516_1434_cross_phase_generator_backlinks_strip", "20260703_1507_train_py_ignores_extends_drops_layers", "20260728_1642_llm_wiki_rename_ends_memory_collision"]
session_origin: claude_code
session_label: "LLM Wiki rename + consult gate"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/dec28250-837d-46e8-b4c9-5ce845ae7698.jsonl
raw_completeness: full
---

# A repo-wide rename has two traps a string-sed cannot see

## Key conclusion

Renaming the in-repo memory layer to the LLM Wiki touched 276 files, and two failure modes would have silently survived a straightforward find-and-replace. The first is that **Python builds paths from components**: the wiki tools declared `DEFAULT_ROOT = ROOT / "docs" / "memory"`, so a search for the string `docs/memory` returned nothing for those lines, and all four tools would have kept pointing at a directory that no longer existed — failing only at runtime, and only for whoever ran them next. The second is that **"memory" is a real word in this repo's subject matter**: the literature reviews under `docs/project/references/` discuss episodic memory, working memory, and fast-weight memory, none of which have anything to do with this system.

## Evidence, measurements, facts

- The component-path lines the string sweep missed, found only by a follow-up grep for quoted path fragments (`grep -n '"memories"\|"memory"' scripts/claude/*.py`):
  - `regen_wiki_links.py:12`, `lint_wiki.py:16`, `regen_wiki_graph.py:21`: `ROOT / "docs" / "memory"`
  - `open_conversation.py:21`: `ROOT / "docs" / "memory" / "memories"`
  - `snapshot_code_graph.py:31`: `REPO_ROOT / "docs" / "memory" / "code_snapshots"`
  - plus `memory_root / "memories"` inside three `collect_insights()` functions and `lint_wiki.py:403`.
- The prose pass was skipped for any path under `docs/project/references/`; path rules still applied there, so citations stayed correct.
- References to Claude's OWN built-in layer were frozen behind sentinels before the prose rules ran and restored afterwards — `MEMORY.md`, `auto-memory`, `built-in memory`, `~/.claude/.../memory/`. Verified afterwards that "the auto-memory layer holds the rule" survived verbatim while "the memory layer" (referring to ours) converted.
- Two regexes used negative lookbehind so command tokens converted but English did not: `(?<![A-Za-z])/memorize\b` and `(?<![A-Za-z])/recall\b` — these correctly skip "encoding/recall" and "recognition/recall" in the cognitive-science docs.
- Files with no extension are invisible to an extension-filtered `git ls-files` sweep — `.gitignore` (which held `docs/memory/.trash/*`) had to be fixed in a separate pass.
- NAS I/O forced the approach: a full-tree Python pass over 1905 files timed out twice at 2 min. Pre-filtering with one `git grep -lIE '<alternation>'` cut the candidate set to 329 files and made the pass finish in seconds.

## Decisions and actions

- Rename script structure that worked: **path rules applied everywhere; prose rules applied everywhere except domain-term corpora; protected tokens frozen behind sentinels for the duration of the prose pass**. Dry-run first with per-rule hit counts, inspect the risky rules' actual matches, then apply.
- After any bulk rename of a tool's home directory, **grep for quoted path fragments** (`"memory"`, `"memories"`) separately from the slash-joined string, then run every renamed tool end-to-end. Passing `--check` on the real tree is the only proof the default root is right.
- Same family as [[20260703_1507_train_py_ignores_extends_drops_layers]]: a config/path assumption that looks verified because the check shares the broken assumption. Here the check was "grep for `docs/memory`" and the broken code did not contain that string.
- Also adjacent to [[20260516_1434_cross_phase_generator_backlinks_strip]] — scripts that scan or generate document bodies need their own audit whenever the document tree moves.

## Open questions and follow-ups

- No automated guard exists against the component-path trap. A cheap future check: a test that asserts each wiki tool's `DEFAULT_ROOT` exists on disk.

## References

- Commit `46f0310`; rename script preserved at the job tmp path used during the session (local-only, not in the repo).
- The rename decision this trap was hit during: [[20260728_1642_llm_wiki_rename_ends_memory_collision]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume dec28250-837d-46e8-b4c9-5ce845ae7698` (re-enter the session) or `python scripts/claude/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260728_1642_llm_wiki_rename_ends_memory_collision]] (wiki_system_design, 2026-07-28) — The project's in-repo session-insight layer and Claude Code's built-in auto-memo
<!-- END BACKLINKS -->
