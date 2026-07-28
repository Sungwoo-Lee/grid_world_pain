---
id: 20260728_1647_agent_jargon_rename_vs_gloss_drift_check
date: 2026-07-28
time: "16:47"
folder: subagent_engineering
tags: [subagent, meta, decision, design]
summary: "Replaced software jargon across all agent profiles with plain words (blocker/concern/nit -> Critical/Moderate/Low; pre-mortem, blast radius, footgun, shard, retroactive, ...). The reusable rule: before renaming a term, count its occurrences in written OUTPUT — terms confined to profiles get replaced, terms living in produced docs (backbone: 108 files) get a gloss instead, or the rename desyncs history."
related: ["20260728_1644_plan_reviewer_and_analysis_verdict_gate"]
session_origin: claude_code
session_label: "agent-team model tiering + plan-reviewer + jargon pass"
importance: medium
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/50dd88a1-cbee-4a05-a3ad-a27865c94b92.jsonl
raw_completeness: full
---

# Agent-profile vocabulary — rename what is private, gloss what is published

## Key conclusion

Agent profiles had accumulated software-culture jargon that a domain researcher cannot decode ("blocker", "nit", "pre-mortem", "footgun", "blast radius", "shard", "retroactive"), which violates the project's own plain-language documentation rule at the exact place it matters most — the labels agents attach to their findings. The non-obvious part of fixing it: **a term's cost depends on where it lives.** A term used only inside `.claude/agents/` can be replaced freely; a term that agents have already written into hundreds of output documents cannot, because renaming it in the instructions desyncs the corpus. The check is mechanical — count occurrences in `docs/` before deciding — and it flipped the decision on three of the sixteen terms.

## Evidence, measurements, facts

- Severity scale, unified across all four reviewers: 🔴 **Critical** (fix before going further) / 🟡 **Moderate** (likely costs a re-run) / 🟢 **Low** (cosmetic) / ❓ **Open** (an assumption nobody has verified yet). Each reviewer now reproduces that legend verbatim in its report, so the label carries its definition instead of needing a glossary.
- The scale also fixed a silent inconsistency: `math-reviewer` had been using 🔴 wrong / 🟡 ambiguous while the others used blocker / concern / nit, so a "🔴" meant different things depending on which agent wrote it — making findings non-comparable across two reviews of the same work.
- Replaced outright (profile-only terms): pre-mortem → advance failure check; blast radius → side effects; footgun → easy-to-misuse trap; doc rot → docs going stale; scope creep → unrequested scope growth; load-bearing → critical; system-of-record → authoritative record; soft-split rule → leave-old-docs-in-place rule; shard/sharding → batch/batching; retroactive → unplanned / after the fact.
- Kept with a gloss because the term lives in produced documents: **backbone** (108 files under `docs/project/references/`) → profiles say "section-by-section summary" and both literature agents carry a terminology note telling them to keep writing "backbone" inside those files; **Mode A / Mode B** (13 docs) → always rendered `Mode A (planned)` / `Mode B (unplanned)`; **Verification Protocol** (14 docs) → heading now states the question it answers, cross-references read "plan-adherence check".
- Left alone deliberately: genuine field vocabulary (ablation, sweep, pre-registered, regression test, construct validity, latent bug, pre-flight, manifest, canonical), and `professor-dl-theory`'s "load-bearing", which there means a framing doing real work rather than being decorative — flattening it to "critical" would have lost the point.
- Two mechanical traps hit during the sweep: a substring search reported "nit" 48 times (matching *definite*, *monitor*, *unit*) and "rot" 44 times (*protocol*, *prototype*), so counts must use word boundaries; and `\bshard\b` missed `sharded`, `Sharding`, and `_shardN`, so a residual sweep needs case-insensitive substring matching after the word-boundary pass.
- Glossing introduced its own defect: mechanically inserting "(retroactive)" produced the stutter "Mode B (retroactive) — retroactive hypothesis frame". Any gloss pass needs a read-back for duplication.

## Decisions and actions

- Before renaming any project-wide term: `grep -rl '<term>' docs/ | wc -l`. High count in written output ⇒ gloss, do not rename.
- Prefer a label that carries its own definition over a "better" word — a better word still has to be learned once; a word printed next to its meaning does not.
- User-facing choice was collected via structured options with rendered previews rather than by proposing one wording; the first two proposals were rejected as still unintuitive, and the chosen set (Critical/Moderate/Low) was one the assistant had ranked lowest.

## Open questions and follow-ups

- The plan verdict line still reads SOUND / SOUND WITH CONCERNS / NOT READY, which uses "concerns" after that word was retired as a severity name. Left pending user preference.

## References

- Commits: `7d7ea79` (severity scale), `5d6f53a` (jargon sweep), `febc42b` (retroactive → unplanned).
- Rule source: project `CLAUDE.md` §Documentation framing. Related: [[20260728_1644_plan_reviewer_and_analysis_verdict_gate]].

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
