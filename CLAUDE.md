## Agent Team

Delegate to the matching agent — read its profile in `.claude/agents/` for full responsibilities and tool scope.

| Agent | Model | Role | Scope |
|---|---|---|---|
| [senior-developer](.claude/agents/senior-developer.md) | opus | Plan, analyze, WandB analysis, verify | `docs/` |
| [developer](.claude/agents/developer.md) | sonnet | Implement approved plans, test, report | full code |
| [code-reviewer](.claude/agents/code-reviewer.md) | opus | JAX/Flax/vmap/PRNG correctness review | `docs/reviews/` |
| [math-reviewer](.claude/agents/math-reviewer.md) | opus | Verify equations match cited papers | `docs/reviews/` |
| [experiment-designer](.claude/agents/experiment-designer.md) | opus | Hypothesis-driven ablation design (G1/G2/H1–H5) | `docs/experiments/` |
| [literature-reviewer](.claude/agents/literature-reviewer.md) | opus | Per-paper review (backbone + Phase 1/2 LaTeX) | `docs/literature/` |
| [literature-curator](.claude/agents/literature-curator.md) | opus | Cross-paper synthesis, TOC, thematic regrouping | `docs/literature/` |

**Default code-change flow:** senior-developer plans → user approves → developer implements (uncommitted) → senior-developer verifies. Codified in the `feature-workflow` skill.

### Workflow Skills (`.claude/skills/`)

| Skill | Pattern |
|---|---|
| `feature-workflow` | plan → approve → implement → verify |
| `bug-fix-workflow` | root-cause → approve → fix + regression test → verify |
| `training-experiment-workflow` | design new experiment OR analyze existing runs |
| `parallel-literature-review` | shard corpus → K reviewers → merge (≥10 papers) |

---

## Project-Wide Rules

- **Survival-step evaluation.** Performance is measured in survival steps, never cumulative reward.
- **No fallback defaults.** Critical configs use `config.get_mandatory('key')`; missing key → `ValueError`. New keys listed in the plan's File Changes section.
- **Templates.** Plans go in `docs/` using [issue_plan](docs/TEMPLATES/issue_plan.md) (what to build/fix) or [training_analysis](docs/TEMPLATES/training_analysis.md) (what happened — hypothesis-driven). Cross-link related docs both directions.
- **Working files.** Intermediate results to `tmp/YYYYMMDD_HHMMSS_<topic>.md`, written after each step.
- **Parallelism.** OK for independent tasks; sequential for hand-off chains; shell loops beat parallel agents for mechanical batch work.
