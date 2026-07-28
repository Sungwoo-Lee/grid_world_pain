---
id: 20260728_1645_wiki_search_subagent_rejected_on_economics
date: 2026-07-28
time: "16:45"
folder: subagent_engineering
tags: [subagent, wiki, decision, design, meta]
summary: "Considered and rejected a dedicated wiki-search sub-agent that would answer topic queries so main Claude never loads wiki content. Two reasons. (1) Arithmetic: a sub-agent's boot floor is ~36 KB (root CLAUDE.md 27 KB + agent profile 8.8 KB) against a mean entry of 12.5 KB, so delegation only pays above ~3 entries — for a targeted lookup it costs MORE than reading directly. (2) Reachability: sub-agents have no Agent tool, so it would be invisible to developer / plan-reviewer / code-reviewer / experiment-designer — the exact walk-back bug-curator just took in 87a2e4d. Verdict: keep the 2 KB gate for targeted work, use existing Explore for fan-out, and tighten topic-index rows instead."
related: ["20260528_1647_bg_isolation_subagent_bypass", "20260728_1643_subagents_cannot_delegate_dead_instructions", "20260728_1644_wiki_pull_gate_widened_before_any_task"]
relations: ["extends:20260728_1643_subagents_cannot_delegate_dead_instructions"]
session_origin: claude_code
session_label: "LLM Wiki rename + consult gate"
importance: high
status: settled
valid_until: 2027-01-31
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/dec28250-837d-46e8-b4c9-5ce845ae7698.jsonl
raw_completeness: full
---

# A wiki-search sub-agent loses on arithmetic and on reachability

## Key conclusion

The idea was appealing: a sub-agent that takes a topic question, reads around the wiki in its own context, and hands back a short answer — so the main conversation never pays for the reading. It does not work as the primary retrieval path, for two independent reasons. **Delegation has a fixed boot cost of roughly 36 KB** before the agent reads anything, against a mean wiki entry of 12.5 KB — so a targeted lookup costs more delegated than done directly. And **sub-agents cannot spawn sub-agents** in this project, so a wiki-search agent would be unreachable from exactly the agents that most need it. It stays worth doing for one narrow shape: broad fan-out across many entries.

## Evidence, measurements, facts

- Measured boot floor for a sub-agent, before any reading: root `CLAUDE.md` **26,928 bytes** + a typical agent profile (`bug-curator.md`) **8,750 bytes** = **~36 KB**, plus tool schemas.
- Measured wiki costs: `ROOT_INDEX` Active-folders table **2,062 B**; topic indexes **10,927–24,489 B**; mean entry **12,514 B** across 184 entries.
- **Targeted lookup** (gate + one topic index + one entry) ~32 KB read directly. Delegated: ~36 KB boot + the same ~32 KB read inside the sub-agent + ~2 KB returned. Net loss.
- **Crossover: ~36 KB boot / ~12.5 KB per entry ≈ 3 entries.** Above that, delegation wins.
- **Fan-out** ("what have we learned about Dreamer?" — 25 entries in `dreamer_diagnosis`) ≈ 312 KB. Delegated, the main context pays ~38 KB instead of ~314 KB — roughly 8x.
- Reachability: only top-level Claude and `senior-developer` hold the `Agent` tool. A parallel session captured the same constraint from the opposite direction — four agent profiles carried an unexecutable "consult `bug-curator`" instruction — in [[20260728_1643_subagents_cannot_delegate_dead_instructions|extends]]. `developer`, `code-reviewer`, `plan-reviewer`, `env-config-reviewer`, `experiment-designer`, `training-runner` do not.
- **Empirical precedent from this repo, one commit earlier**: `87a2e4d` ("reachable bug registry") rewrote `plan-reviewer` to grep `KNOWN_BUGS.md` directly, adding "**You cannot spawn `bug-curator`** — sub-agents have no `Agent` tool". `bug-curator`'s serve mode is the exact design pattern proposed here, and it had just been walked back for this reason.
- Topic-index rows average ~600 bytes each (`cluster_ops` 24,489 B / 40 rows = 612; `hypervigilance` 578; `dreamer_diagnosis` 669).

## Decisions and actions

- **No new agent type.** The pull gate from [[20260728_1644_wiki_pull_gate_widened_before_any_task]] stays the default path for targeted work — it is cheaper than any delegation can be.
- **For genuine fan-out, use the existing `Explore` / `general-purpose` agent** rather than a new type. What it needs is not a new profile but the wiki's semantics in the spawn prompt: a generic searcher will quote a `status: superseded` entry as current truth and ignore `valid_until` expiry. That is a correctness bug, not an efficiency one, and the right home for it is a delegation rule + prompt template inside the `/wiki-read` skill, so the semantics travel with the delegation.
- **The real bottleneck is upstream**: topic-index rows at ~600 B each are 5x longer than a scannable index needs (~120 B — one line, enough to decide open-or-skip). Tightening them makes every drill-down cheaper and pushes the delegation threshold further out, with no new moving part. Queued, not yet done.
- Second-order risk noted, beyond raw context size: a summarising sub-agent compresses lossily against a query it did not write, and if it drops the caveat that mattered, the caller cannot tell. Silence looks identical to "nothing relevant found".
- Related sub-agent capability boundary: [[20260528_1647_bg_isolation_subagent_bypass]].

## Open questions and follow-ups

- **Revisit trigger (scale, not date):** a dedicated agent earns its keep at roughly 400+ entries (topic indexes stop being readable at all), or when cross-topic synthesis — "what do we know about X across all ten folders" — becomes a routine query shape that no single topic index can serve. Neither holds at 184 entries in 10 folders.
- `valid_until: 2027-01-31` because the arithmetic is pinned to measured file sizes (root `CLAUDE.md` 27 KB, mean entry 12.5 KB) that drift as the repo grows. Re-measure before relying on the crossover figure.

## References

- Commit `87a2e4d` (the `bug-curator` reachability walk-back that supplied the precedent).
- `.claude/agents/plan-reviewer.md` pass 5; `.claude/agents/bug-curator.md` "How You Get Invoked".
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume dec28250-837d-46e8-b4c9-5ce845ae7698` (re-enter the session) or `python scripts/claude/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260728_1642_llm_wiki_rename_ends_memory_collision]] (wiki_system_design, 2026-07-28) — The project's in-repo session-insight layer and Claude Code's built-in auto-memo
- [[20260728_1644_wiki_pull_gate_widened_before_any_task]] (wiki_system_design, 2026-07-28) — The wiki is pull-only and was gated on 'when wiki work is requested', so it was 
<!-- END BACKLINKS -->
