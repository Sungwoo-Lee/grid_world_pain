# `docs/pi/` — Principal Investigator (PI) Workspace

## Purpose

This folder is the home of the **PI agent** ([.claude/agents/pi.md](../../.claude/agents/pi.md)) — the portfolio-level strategic layer for this academic project. The PI's job is to keep the project paper-shaped: the team operates under finite GPU-weeks and finite calendar time toward a publishable contribution, which means we cannot afford to chase every promising thread, but we also cannot tunnel on a single thread to the point of missing the better paper. The PI surfaces that focus-vs-explore dilemma at major decision points, presents the user with 2–4 candidate paths plus their costs and benefits, and logs the user's decision so the rest of the team has continuity.

The PI does not implement, design experiments, or do domain-deep research. It commits *what work is worth doing*; the other agents do the work.

## Layout

| Path | What lives here |
|---|---|
| [`PORTFOLIO.md`](PORTFOLIO.md) | The active publication-track ledger — the 1–2 (occasionally 3) papers the project is currently building toward, plus the explore-buffer reserve. Updated by the PI whenever a call changes the portfolio. Single source of truth for "what are we writing right now". |
| [`calls/YYYY-MM-DD_<topic>.md`](calls/) | Per-decision call logs. Each file documents one strategic question, the candidate paths considered, the user's verbatim decision, and the hand-off to the downstream agent. Append-only — older calls stand as the project's strategic history. |
| `roadmap/<topic>.md` (when needed) | Optional roadmap snapshots — multi-week views of the active tracks' next milestones, written when a portfolio change calls for one. Most calls do not need a roadmap; create on demand. |

## When the PI gets invoked

Read [.claude/agents/pi.md](../../.claude/agents/pi.md) for the full trigger list. Short version:

- **Manual:** "/pi", "what does PI think?", "PI feedback on `<doc>`".
- **Proactive (via `agent-manager`):**
  - Before launching a multi-run experiment (after `experiment-designer` + `env-config-auditor` clean, before `training-runner`).
  - After `experiment-analyzer` finishes a multi-run comparison.
  - When `senior-developer` drafts a roadmap-level (multi-week) plan.
  - When `research-postdoc` proposes a new direction.
  - When `literature-curator` surfaces a new theme.

The PI is **NOT** invoked for bug fixes, single-config tweaks, doc-only edits, single-paper literature reviews, or one-off ad-hoc launches.

## How a call works (for readers)

1. The PI reads the trigger context (the plan / design / analysis that brought it in) and the project frame (`project_plan.md`, recent `calls/`, `PORTFOLIO.md`).
2. The PI sketches 2–4 candidate paths with costs and benefits.
3. The PI surfaces the choice to the **user** via `AskUserQuestion`. **The user makes the final call** — the PI does not.
4. The PI writes a call log under `calls/YYYY-MM-DD_<topic>.md` capturing the question, options, decision, rationale, and hand-off.
5. The PI names the downstream agent (`experiment-designer` / `senior-developer` / `experiment-analyzer` / `research-postdoc`) for the parent (top-level Claude) to spawn.
6. The PI appends a `note` row to the daily diary via the `/diary` skill so parallel sessions see the call.

## Cross-references

- Agent profile: [.claude/agents/pi.md](../../.claude/agents/pi.md)
- Project frame: [docs/project/project_plan.md](../project/project_plan.md)
- Routing playbook: [docs/AGENT_PLAYBOOK.md](../AGENT_PLAYBOOK.md)
- Project rules and agent table: [CLAUDE.md](../../CLAUDE.md)
