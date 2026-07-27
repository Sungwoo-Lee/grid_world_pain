---
id: 20260528_0215_foam_excludes_required_not_search_exclude
date: 2026-05-28
time: "02:15"
folder: cluster_ops
tags: [meta, learned_lesson, decision]
summary: "Foam doesn't honor VSCode's search.exclude; needs its own foam.files.ignore. Adding it dropped this project's VSCode cold start from 329,724 ms (5.5 min) to 3,729 ms (3.7 s) — 88× speedup."
related: ["20260516_1433_graphifyy_integration_cheatsheet", "20260528_0216_foam_wikilinks_filename_as_id_sparse_aliases"]
session_origin: claude_code
session_label: "EPISODE project_plan rewrite + Foam wikilink convention + VSCode startup fix"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/e386e7cb-3b2e-4604-abc2-921d1b6c973a.jsonl
raw_completeness: full
---

# Foam excludes — foam.files.ignore is required; search.exclude is not honored

## Key conclusion

VSCode's `search.exclude` and `files.watcherExclude` do **not** cover the Foam extension's wikilink indexer — Foam has its own scope-control setting, `foam.files.ignore`. On a NAS-backed remote-SSH workspace with 300+ markdown files plus PDFs in references and many gitignored bulk directories, omitting `foam.files.ignore` lets Foam walk everything (including PDFs and worktrees) and pushes its activation time into minutes. Adding `foam.files.ignore` mirroring the existing exclude lists (plus PDF source dirs and the diary tree) dropped this project's Foam activation from 5.5 minutes to under 4 seconds.

## Evidence, measurements, facts

- VSCode startup profiler (`Developer: Startup Performance`) confirmed the bottleneck deterministically:
  - **Before fix**: `foam.foam-vscode` Finish Activate = **329,724 ms** (≈ 5.5 min). Next slowest extension was GitHub.copilot-chat at 5,978 ms.
  - **After fix**: `foam.foam-vscode` Finish Activate = **3,729 ms** (88× speedup). Overall workbench-ready time dropped from 8,214 ms → 4,974 ms.
- Two earlier settings — `files.watcherExclude` and `search.exclude` (added in commit `19d223a`) — alone did NOT fix Foam, even though they fixed VSCode's native file watcher CPU. CPU dropped from 57.9% → 22.2% but cold start was still minutes. Foam ignored both.
- The directories Foam was walking unnecessarily, by size: `wandb/` 123 GB, `results/` 43 GB, `.claude/worktrees/` 1.1 GB (3 worktrees × ~370 MB each), `claude_data/` 382 MB, `docs/project/references/<topic>/sources/` (PDFs), `docs/diary/` (append-only, ~daily file).
- Setup: remote-SSH session from M2 Pro Mac (16 GB, 0.19 GB free at snapshot) to lab machine via NAS-mounted filesystem. Extension Host runs on remote; file reads cross the NAS mount. Small-file read latency dominates.
- Fix landed in `.vscode/settings.json` commit `4416c13`. Three exclude layers configured: `files.watcherExclude` (OS watcher), `search.exclude` (VSCode search), `foam.files.ignore` (Foam indexer). Each subsystem reads only its own setting.
- Foam's behaviour: it has its own DocumentLinkProvider + markdown-it plugin that registers with VSCode but maintains an internal workspace index, repopulated on cold start by walking files. No on-disk persistence — every VSCode launch rebuilds from scratch. Incremental updates during a session are fast (milliseconds per file change); cold start is the bottleneck.

## Decisions and actions

- `.vscode/settings.json` is the canonical home for both VSCode-native excludes AND extension-specific excludes — keep them aligned. The three layers overlap but each affects a different subsystem; configure all three.
- `foam.files.ignore` includes two patterns NOT in `search.exclude`: `**/docs/project/references/**/sources/**` (PDFs that graphify also tripped on with "wrong pointing object" parser errors) and `**/docs/diary/**` (high churn, low wikilink value).
- Future cold-start diagnosis pattern: run `Developer: Startup Performance` → scan "Finish Activate" column; any extension > a few seconds is a candidate. Foam's documented per-extension setting is the first thing to check before blaming VSCode or NAS.

## Open questions and follow-ups

- VSCode's Copilot Chat is now the slowest extension at 6.4 s — disabling it would shave that off cold start if Copilot is not used. Diminishing returns from here on this workspace; ~5 s cold start is near the NAS-read floor.
- Mac's memory pressure (0.19 GB free of 16 GB) probably contributed to the original 5.5 min observation; the post-fix 3.7 s is still measured on the same memory-pressured Mac, so the win is real even under pressure.

## References

- Companion convention insight (project-wide doc linking): [[20260528_0216_foam_wikilinks_filename_as_id_sparse_aliases]]
- Prior cluster_ops insight on graphify (related external tool, code-graph not doc-graph): [[20260516_1433_graphifyy_integration_cheatsheet]]
- Commits: `19d223a` (initial watcher/search excludes; partial fix), `4416c13` (foam.files.ignore; full fix).
- VSCode startup profile snapshots (in conversation): before-fix `Foam=329,724ms`, after-fix `Foam=3,729ms`.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume e386e7cb-3b2e-4604-abc2-921d1b6c973a` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260528_0216_foam_wikilinks_filename_as_id_sparse_aliases]] (wiki_system_design, 2026-05-28) — Project-wide doc-linking convention adopted: [[filename]] wikilinks for new cros
<!-- END BACKLINKS -->
