---
id: 20260509_1621_multi_agent_research_chain_v2_pattern
date: 2026-05-09
time: "16:21"
folder: subagent_engineering
tags: [subagent, design, decision, learned_lesson]
summary: "Multi-agent research chains (postdoc triage → professor directions → postdoc synthesis) handle mid-chain user expansion via append-only sibling versioning, NOT in-place revision. Worked example: NMN meta/continual pivot — postdoc + 2 professors produced v1 memos, user asked for olfactory perturbation, the relevant professor was re-spawned with a v2 task and wrote nmn_meta_context_conditioning_v2.md alongside the v1; postdoc then wrote nmn_meta_continual_synthesis_v2.md alongside the v1. v1 memos preserved as historical snapshots."
related: ["20260509_1537_professor_analysis_resets_exotic_investigation", "20260509_1620_documentation_framing_policy"]
session_origin: claude_code
session_label: "nmn_meta_continual_pivot_2026-05-09"
importance: medium
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/6bbe7739-79ae-4486-b230-d8b7b8263893.jsonl
raw_completeness: full
---

# Multi-agent research chains use append-only sibling versioning under mid-chain user expansion

## Key conclusion
Multi-agent research-direction chains (`research-postdoc` triage → 1+ `professor-*` directions → `research-postdoc` synthesis) handle **mid-chain user expansion** via append-only sibling versioning at every layer, NOT in-place revision. Concrete worked example from this session: the postdoc + `professor-rl-bayesian-dl` + `professor-neuromodulation` produced v1 memos for the NMN meta/continual pivot. The user then said *"why don't you perturb the olfactory properties as well? — multiple property sets — update the plan with proper feedback from professor agents"*. The pattern that emerged: (1) re-spawn the relevant professor (here, `professor-rl-bayesian-dl`) with a v2 task that explicitly cites their v1 and asks what changes; (2) the professor writes `<topic>_v2.md` at a sibling path, NOT overwriting v1; (3) re-spawn the postdoc to write `<synthesis>_v2.md` at a sibling path; (4) v1 memos remain on disk as historical snapshots; v2 memos cross-link back to v1 with explicit "what changed" framing. The agent profiles do NOT need an additional rule to enforce this — the project's broader append-only versioning convention (in `docs/experiments/summaries/README.md` and the `.claude-memory/` operating manual) was simply applied transitively to research-chain memos. Future agent-manager routing plans for "mid-chain user expansion" should default to this pattern; the cost of a v2 sibling memo is one professor invocation + one postdoc invocation, well within budget.

## Evidence, measurements, facts
- v1 memos written: `docs/project/triage/20260509_1517_nmn_meta_continual_pivot.md` (postdoc, ~31 KB); `docs/project/directions/nmn_meta_context_conditioning.md` (prof-RL/BDL, ~19 KB); `docs/project/directions/nmn_continual_lifelong_probe.md` (prof-neuromod, ~22 KB); `docs/project/ideas/nmn_meta_continual_synthesis.md` (postdoc, ~14 KB).
- User mid-chain expansion request: olfactory perturbation (predator/rabbit smell-swap, multiple property sets) on top of the existing active/passive predator dimension.
- v2 memos written under the pattern: `docs/project/directions/nmn_meta_context_conditioning_v2.md` (prof-RL/BDL, ~27 KB); `docs/project/ideas/nmn_meta_continual_synthesis_v2.md` (postdoc, ~17 KB). Continual side memo unchanged because the user's expansion was meta-side only — the unchanged memo is referenced by the v2 synthesis without re-spawning prof-neuromod.
- v2 memos contain explicit "Δ-1, Δ-2" change-from-v1 sections at the top so a reader can navigate v1 → v2 without re-reading v1 in full.
- Each v2 memo's frontmatter or §1 explicitly states "**Supersedes**: nothing — append-only synthesis layer. v1 stays as a historical snapshot."
- Cross-references: v2 synthesis cites v1 synthesis as `[nmn_meta_continual_synthesis.md](./nmn_meta_continual_synthesis.md)`; v2 prof-RL/BDL memo cites v1 prof-RL/BDL memo. v1 memos do NOT need to be edited to point forward to v2 (broken chronology); the v2 chain navigates back via `related:` frontmatter.
- Cost estimate: 2 professor re-spawns (~5-10 min each) + 1 postdoc re-spawn (~5 min) + manual orchestration overhead ≈ 30 min for the v1 → v2 turn. Lower than re-running v1 from scratch.

## Decisions and actions
- **Append-only at every layer**: triage memo (rare to re-version), directions memos (re-version per professor when their domain is expanded), synthesis memo (re-version when ANY input memo changes). v1 memos preserved.
- **Naming convention**: `<topic>_v2.md` sibling. NOT `<topic>_2026-05-09T16-00.md` (timestamp), NOT `<topic>.md` (overwrite). The `_v2` suffix is human-scannable; the timestamp lives in the file's frontmatter `date` field.
- **Trigger for re-spawn**: user explicitly asks for an expansion that touches the relevant memo's domain. Mild rephrasings or clarifications do NOT trigger a v2 — they go in the user's reply or in a follow-up `_addendum.md`.
- **Synthesis must always re-version when any input changes**: the synthesis memo names input memos by path; if an input gains a v2 sibling, the synthesis cannot pretend the v1 input is still authoritative. This is the load-bearing rule.
- **Continual side opt-out**: when the user's expansion is one-domain-only (here: meta-side olfactory perturbation, no continual change), only the affected professor is re-spawned. The other side's v1 memo carries forward unchanged into the v2 synthesis. Saves a re-spawn.

## Open questions and follow-ups
- **v3+ versioning**: not yet exercised. If a third user-expansion lands, do we go to `_v3` or fold into `_v2`? Default per the convention: `_v3` sibling. The version index lives in the filename, not in a single doc.
- **Triage memo re-versioning**: the triage memo was NOT re-versioned in this session — the user's expansion was small enough that the v2 synthesis incorporated the triage's framing without amendment. If a user expansion fundamentally changes the triage's frame disambiguation (e.g., rules out a candidate experiment entirely), the triage probably needs a `_v2` too. This case has not yet occurred.
- **Memory-layer parallel**: should `.claude-memory/memories/<topic>/<id>.md` follow the same `_v2` pattern, or its existing `supersedes:` / `superseded_by:` frontmatter? Memory-layer convention is the latter (frontmatter, no `_v2` suffix). Research-direction memos in `docs/project/` use the former (`_v2` suffix). Two valid conventions for two layers; do NOT cross-pollinate.
- **Agent-manager codification**: should `agent-manager`'s routing-plan output explicitly check "is this a v1→v2 chain re-spawn?" and surface the append-only convention to the parent? Probably yes — would make the pattern visible at planning time, not just at execution time. Out of scope for this insight; flag for future agent-manager profile update.

## References
- Triage memo (v1, unchanged): `docs/project/triage/20260509_1517_nmn_meta_continual_pivot.md`.
- Prof-RL/BDL v1 → v2: `docs/project/directions/nmn_meta_context_conditioning.md` → `_v2.md`.
- Prof-neuromod (v1 only, unchanged): `docs/project/directions/nmn_continual_lifelong_probe.md`.
- Postdoc synthesis v1 → v2: `docs/project/ideas/nmn_meta_continual_synthesis.md` → `_v2.md`.
- Companion routing-pattern insight (this session's other agent-routing capture): `20260509_1537_professor_analysis_resets_exotic_investigation` — different routing pattern (professor-before-platform-developer) but same agent-routing layer.
- Companion policy insight (this session): `20260509_1620_documentation_framing_policy` — the project-wide append-only convention this pattern instantiates for research-chain memos.
- Project's broader append-only conventions: `docs/experiments/summaries/README.md` (re-summaries write fresh dated files); `.claude-memory/CLAUDE.md` §11 (filename conventions).
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 6bbe7739-79ae-4486-b230-d8b7b8263893` (re-enter the session) or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md` (one-shot markdown view).
