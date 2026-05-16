---
id: 20260516_1433_graphifyy_integration_cheatsheet
date: 2026-05-16
time: "14:33"
folder: cluster_ops
tags: [meta, learned_lesson, decision]
summary: "To use Graphify in any project conda env: install `graphifyy` (double-y; single-y on PyPI is squatted and unaffiliated), invoke `graphify update <path>` (NOT bare positional which prints help), expect output at `<scanned_path>/graphify-out/` (NOT cwd). Wrapper scripts must fall back to `Path(sys.executable).parent / 'graphify'` when `shutil.which()` fails — the conda env's bin is not on shell PATH when invoking the env's Python directly without activating the env. `.gitignore` must use `**/graphify-out/` (recursive) to ignore per-scope outputs."
related: ["20260516_1431_v2_three_role_architecture", "20260516_1432_karpathy_graphify_adaptation_rationale"]
session_origin: claude_code
session_label: "memory v2 build complete + graphify integration + bridge"
importance: medium
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/a06843e3-ec5e-4850-9b54-75f95633989b.jsonl
raw_completeness: full
---

# Graphify integration cheat sheet — graphifyy package, `graphify update`, per-scope output

## Key conclusion

Four non-obvious gotchas when wiring Graphify (https://github.com/safishamsi/graphify) into a project's tooling: (1) the PyPI package name is `graphifyy` (double-y) — the `graphify` package on PyPI is squatted and unaffiliated; the installed binary is `graphify` (single y); (2) the CLI's "build graph" command is `graphify update <path>`, NOT bare `graphify <path>` (which prints help); (3) Graphify writes output to `<scanned_path>/graphify-out/`, NOT the cwd — so `graphify update src/` lands artifacts at `src/graphify-out/`; (4) wrapper scripts invoked via the conda env's Python without activating the env need to fall back to `Path(sys.executable).parent / 'graphify'` because `shutil.which()` looks at shell PATH only.

## Evidence, measurements, facts

- Initial Phase F stub assumed `pip install graphify` and `graphify <path>`; both wrong. Stub fell through to no-op because `shutil.which("graphify")` returned None even after the install (also wrong because of `uv`/`pipx` unavailable in this env).
- Correct install: `/home/vncuser/miniconda3/envs/grid_world_pain/bin/pip install graphifyy` → 28 tree-sitter language packs + graphifyy 0.8.5; binary lands at `/home/vncuser/miniconda3/envs/grid_world_pain/bin/graphify`.
- Verified on the project's `src/`: 735 nodes, 1009 edges, 59 communities, 0 LLM tokens (tree-sitter only). God-nodes correctly identified `main()`, `OneHotDist`, `DreamerNeuromodulatorRNN`, `ModulatedLayerNormGRUCell`, `SiLU`, `get_observation()`, `DreamerTrainer`, `ParallelEnv`, `NeuromodulatorRNN`, `RSSM`.
- Other Graphify commands that work after `update`: `graphify explain "X"` (node + neighbors), `graphify path "A" "B"` (shortest-path between two nodes, warns on ambiguous matches), `graphify query "natural language question"` (BFS over the graph with a token budget), `graphify watch <path>` (rebuild on changes), `graphify install` (register with a host IDE — Claude Code / Codex / Cursor / etc.).
- `.gitignore` rule must be `**/graphify-out/` (recursive). The naive `graphify-out/` at repo root only ignores the root-level dir; outputs at `src/graphify-out/`, `docs/graphify-out/` etc. would slip through. Verified with `git check-ignore src/graphify-out` → matches.
- Optional LLM-extraction phase (`graphify extract`) requires `ANTHROPIC_API_KEY` / `GEMINI_API_KEY` / `OPENAI_API_KEY` / Ollama / Bedrock. Not used in v2 — tree-sitter pass is sufficient for code structure.

## Decisions and actions

- Adopted `graphify update src` as the standard invocation in `scripts/regen_code_graph.py`. Wrapper checks `shutil.which("graphify")` first, then `Path(sys.executable).parent / "graphify"` as fallback, then prints stub install instructions and exits 0 (so hooks/pre-commit don't break).
- The wrapper does NOT install graphifyy automatically — that's a per-machine user choice. The stub prints the install command verbatim.
- Agent profiles (`.claude/agents/code-reviewer.md`, `senior-developer.md`) document the existence of `src/graphify-out/GRAPH_REPORT.md` and fall back to grep/Read when absent.
- Built `scripts/snapshot_code_graph.py <label>` to capture point-in-time copies into `docs/memory/code_snapshots/` (separate from the live gitignored output).

## Open questions and follow-ups

- When would the LLM-extraction mode (`graphify extract` + an API key) actually add value over the free tree-sitter pass? Probably for docs / PDFs / image extraction, not code. Worth revisiting if `docs/project/references/` PDFs ever need to be queryable.
- `graphify install --platform claude` would write a `graphify` section into project `CLAUDE.md` + a PreToolUse hook. Not done in v2 — would tie the project tighter to one tool than warranted at this point. Revisit if the code-side wiki becomes load-bearing.
- The `--scope all` mode of `scripts/regen_code_graph.py` (scan entire repo, not just `src/`) was implemented but not tested; would produce a giant graph including docs and configs.

## References

- Graphify GitHub: https://github.com/safishamsi/graphify
- Project wrapper: `scripts/regen_code_graph.py`
- Snapshot wrapper: `scripts/snapshot_code_graph.py`
- v2 design doc, Feature 4c: [docs/develop/active/meta/claude_memory_system_v2_design.md](../../../../docs/develop/active/meta/claude_memory_system_v2_design.md)
- Companion insight (memory-side adaptation): [[20260516_1432_karpathy_graphify_adaptation_rationale]]
- Companion insight (architecture): [[20260516_1431_v2_three_role_architecture]]
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume a06843e3-4` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
