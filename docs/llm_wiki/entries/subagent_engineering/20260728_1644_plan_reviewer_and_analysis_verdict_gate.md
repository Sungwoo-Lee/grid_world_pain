---
id: 20260728_1644_plan_reviewer_and_analysis_verdict_gate
date: 2026-07-28
time: "16:44"
folder: subagent_engineering
tags: [subagent, design, decision, meta]
summary: "Added `plan-reviewer` — an adversarial reviewer that inspects a plan before anyone builds, and (new) an analysis verdict before anyone believes it. Closes the team's one unreviewed artefact: experiment-analyzer's conclusions previously went straight to `pi`, which asks whether to continue, not whether the inference holds. Reviewer overlap is kept deliberately; sequencing is now `agent-manager`'s explicit job."
related: ["20260728_1643_subagents_cannot_delegate_dead_instructions"]
session_origin: claude_code
session_label: "agent-team model tiering + plan-reviewer + jargon pass"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/50dd88a1-cbee-4a05-a3ad-a27865c94b92.jsonl
raw_completeness: full
---

# `plan-reviewer` — gating plans, and the analysis verdicts nobody was checking

## Key conclusion

The team had four review surfaces but only three were covered: a plan's *procedure* had no reviewer at all, and an analysis's *inference* had none either. `plan-reviewer` now covers both. The analysis gate is the higher-stakes half: a wrong plan costs a rerun, whereas a wrong verdict becomes a claim in a paper — and the only agent that previously ran after `experiment-analyzer` was `pi`, which owns portfolio scope ("deepen, pivot, or shelve?"), not whether the evidence supports the conclusion. Reviewer coverage across the team **deliberately overlaps**; that is treated as independent double-checking, not redundancy to remove.

## Evidence, measurements, facts

- Four reviewers, five objects: a drafted plan and an analysis verdict (`plan-reviewer`), equations vs. the cited paper (`math-reviewer`), a code diff vs. JAX/Flax conventions (`code-reviewer`), YAML vs. schema + critical-settings registry (`env-config-reviewer`).
- `plan-reviewer` runs 8 passes: verifiability (including circular verification — verifying with the same code path suspected of being broken), project-rule compliance, side effects + how to undo them, assumptions and ordering, prior-art collision, experiment specifics (designs only), empirical-claim soundness (verdicts only), and cost of being wrong.
- Pass 7 (empirical-claim soundness) checks: effect size vs. seed spread, survival-steps-not-reward, temporal evolution rather than endpoints, confounds carried from the design, whether the verdict covers **every** Launch Manifest row or silently drops crashed runs, whether the pre-registered refutation criterion was honoured or softened after seeing data, at least one alternative explanation, and symmetric suspicion of underpowered nulls.
- It earned its keep within an hour of existing: a parallel session ran it on the async-checkpoint-video-render plan and it returned SOUND WITH CONCERNS (0 Critical, 4 Moderate, 3 Low), catching two silent-upload races — the drain path waits for a render but never uploads it, so the final video of every run would never reach WandB; and a dispatch-over-unpolled-completion race that the source idiom is immune to because it discovers results by globbing files rather than from in-memory state.
- Reporting is hybrid: findings always inline with a one-line verdict; a file under `docs/reviews/plan_<topic>.md` only when a Critical finding exists. Analyses use a different verdict vocabulary (CONCLUSION SUPPORTED / WITH CAVEATS / NOT SUPPORTED BY THE EVIDENCE SHOWN), with the last phrased as a claim about the argument, not about the opposite being true.

## Decisions and actions

- Overlap is a feature: no reviewer is told to skip a check because another "owns" it — the same rule can hold in the plan and be violated in the code. `get_mandatory` discipline is checked from three angles on purpose.
- `agent-manager` gained a "Reviewer Sequencing (you own this)" section: upstream-first (plan → config → code, since a finding is cheapest to fix earliest), same-stage reviewers in parallel, no reviewer spawned before its object exists, duplicate findings treated as corroboration, disagreements surfaced to the user rather than arbitrated, and the review budget stated so the user can cut one.
- Shared facts are single-sourced: the latent-bug set lives in `ENVIRONMENT_SUMMARY.md` §Cross-Doc Clarifications + the registry, not restated in reviewer profiles (see [[20260728_1643_subagents_cannot_delegate_dead_instructions]] for the delegation trap this created).
- Analysis gate wired into both analyzer flows (planned and unplanned) ahead of `pi`.

## Open questions and follow-ups

- The analysis gate has not yet run on a real verdict — its value is argued, not demonstrated.
- The agent is still named `plan-reviewer` although it now also reviews analyses; renaming was judged more churn than clarity.

## References

- Commits: `54c374f` (creation), `c7e6693` (overlap policy + single-sourcing), `87a2e4d` (analysis gate).
- Profile: `.claude/agents/plan-reviewer.md`; flows in `docs/AGENT_PLAYBOOK.md` §Reviewer Coverage & Overlap.
- Worked example of its output: `docs/develop/active/refactors/ASYNC_CHECKPOINT_VIDEO_RENDER.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260728_1647_agent_jargon_rename_vs_gloss_drift_check]] (subagent_engineering, 2026-07-28) — Replaced software jargon across all agent profiles with plain words (blocker/con
<!-- END BACKLINKS -->
