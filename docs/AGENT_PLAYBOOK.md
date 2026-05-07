# Agent Playbook

The orchestration layer for the project's agent team. Read by `agent-manager` on every spawn; readable by other agents (and the user) for reference. Each agent profile in `.claude/agents/` says *what that agent does*; this doc says *who comes after whom and when*.

Keep this doc small. Anything agent-specific belongs in the agent's profile. Only **cross-cutting orchestration patterns** live here.

## Default Flows

### Adding a new feature / capability to the platform

```
senior-developer (plan, docs/develop/)        ← User approves plan
        ↓
developer       (implement, dirty tree)
        ↓
senior-developer (Verification Protocol)
        +
env-config-auditor (parallel) — only if the diff touches configs/, src/environment/,
                                or any sensor / observation-breakdown / perceptual-noise code.
                                Audit goes to docs/reviews/config_<plan>.md.
```

Plan-first is non-negotiable: the user approves the *plan* before any code changes. Developer leaves the working tree dirty so verification sees the exact diff.

### Fixing a bug

```
senior-developer (root cause + fix plan)      ← User approves diagnosis + fix
        ↓
developer       (regression test FIRST → fix → confirm test passes)
        ↓
senior-developer (Verification Protocol; confirms regression test is meaningful)
```

Regression test must fail pre-fix and pass post-fix — see `.claude/agents/developer.md`'s Bug-Fix Discipline section. If the plan's "root cause" is actually a project-design issue, escalate to the feature flow rather than expanding the bug-fix scope.

### Designing and running a training experiment

```
experiment-designer (design doc + configs + LAUNCH MANIFEST §3)
                    Manifest has one row per run with planned:
                      Tag, wandb-name, wandb-group, wandb-job-type, Seed
                    Plus §3.1 mapping each row to env+agent config YAMLs.
        ↓
env-config-auditor  (audit; resolve all 🔴 before launch)
        ↓
USER APPROVAL  +  collect target node + GPU index per run (AskUserQuestion)
        ↓
training-runner     (one spawn per manifest row)
                    Receives: node, GPU, plan_doc, run_id.
                    Reads manifest row → applies planned tag/wandb-* values.
                    Launches via run_command.py.
                    Updates the manifest row in place: Status=running,
                      Node, GPU, Launched at, WandB run ID, Log path.
        ↓
USER waits for training (and updates Status=completed/failed/cancelled per row)
        ↓
experiment-analyzer (Mode A; reads manifest as authoritative run inventory;
                     fills Results / Analysis / Conclusions of the same doc)
```

The Launch Manifest is the **system-of-record** binding experimental cells to WandB folders. Three agents share it with **strict column ownership**:

| Agent | Reads | Writes |
|---|---|---|
| `experiment-designer` | — | Planned columns (Run, Cell, Tag, wandb-group, wandb-job-type, Seed) + §3.1 |
| `training-runner` | Planned columns (to apply at launch) | Actual columns (Status, Node, GPU, Launched at, WandB run ID, Log path) of the row it launched |
| `experiment-analyzer` | Whole table (run discovery) | — (analyzer never writes the manifest) |

`training-runner` does NOT pick its own node/GPU — there is no monitoring source available to it. The agent-manager (or the user, in a direct invocation) collects node + GPU upfront and passes them in the spawn prompt. Bundling this with the launch-approval step is cheaper than letting the runner spawn, halt, and require a re-spawn.

For **one-off launches with no plan doc** (ad-hoc / debugging), the runner falls back to its default tag/wandb-name convention (see its profile §3a Path B). The convention is parallel to (and consistent with) the designer's manifest convention, so ad-hoc and planned runs interleave cleanly in WandB.

If an experiment-analyzer or experiment-designer adds a `## Metrics Requested` section, the user reviews and (if accepted) escalates to the feature flow to add the logger.

### Analyzing existing runs (no prior design)

```
experiment-analyzer (Mode B — retroactive hypothesis frame, full doc to docs/experiments/)
```

If the analysis surfaces a bug → fork to bug-fix flow. If a missing metric → fork to feature flow.

### Reviewing a corpus of papers

| Corpus size | Approach |
|---|---|
| < 10 papers | Single `literature-reviewer`. |
| 10–20 papers | 2–3 parallel `literature-reviewer` instances. |
| 20–40 papers | 4–5 parallel. |
| 40+ papers | 5–6 parallel (cap). |

Sharding rules for the parallel case:
- Shard by file count, not topic. K = ceil(N / 5), capped at 6.
- Each reviewer writes to its own `tmp/lit_review_shard_<i>.md`.
- After all shards complete, merge into `docs/project/references/<topic>/<topic>_lit_review.md` (the master review is co-located with the source PDFs; `<topic>` is the exact name of the source-PDF subfolder under `docs/project/references/`). Dedupe at merge (a paper may have ended up in two shards).
- Per-paper processing inside each shard stays sequential — the 4-step backbone needs focused attention per paper.
- For thematic regrouping after the merge, hand off to `literature-curator`.

## Cross-Cutting Constraints

These apply across multiple flows; the manager (or invoking agent) enforces them:

- **env-config-auditor in parallel** with senior-developer's verification, whenever the diff touches configs/, src/environment/, or sensor/observation/noise code. Two orthogonal checks: SD verifies plan adherence; auditor verifies env↔config soundness.
- **WandB analysis uses local files only.** `wandb/run-YYYYMMDD_HHMMSS-<id>/` — never the WandB web API.
- **Survival steps as headline metric** for any RL evaluation. Cumulative reward is at best a secondary diagnostic.
- **Temporal evolution mandatory** in any training analysis — not just end-of-training snapshots.
- **No fallback defaults** for critical config params — `config.get_mandatory()` everywhere, missing key → `ValueError`. Plans list new keys explicitly.
- **Frontmatter contracts**: develop tree is auto-INDEX'd via `scripts/regen_dev_index.py`; experiments tree is by-convention only (no INDEX). Don't run `regen_dev_index.py` for `docs/experiments/`.
- **Soft-split rule**: pre-existing experiment-shaped docs under `docs/develop/active/{hypervigilance,noise,diagnosis}/` are NOT migrated retroactively. Read as reference, write new docs to `docs/experiments/active/<topic>/`.
- **Dirty working tree for verification.** `developer` does NOT commit; `senior-developer` reads the uncommitted diff. Once committed, the verification signal is lost.

## Anti-Patterns to Catch

The manager (and any invoking agent) should detect and push back on these before spawning sub-agents:

- **"Analysis but no design exists"** — user asks for analysis of pre-registered hypothesis but there's no design doc. Push back: do you want Mode B post-hoc (weaker), or should we design first?
- **"Bug fix but the plan rewrites the design"** — root cause is a project-design issue, not a localized bug. Escalate to feature flow.
- **"Skip user approval because the plan looks obvious"** — there is no such thing as an obviously correct plan. Always pause at user-approval gates.
- **"Implementer also verifies"** — never. The verifier is a separate spawn.
- **"Commit before verify"** — never. The dirty diff IS the verification signal.
- **"Bundle unrelated cleanups into a bugfix"** — verifier will (rightly) flag as out-of-scope. Surface this and split into separate plans.
- **"Parallel reviewers within a single paper"** — fragmented reviews. Per-paper stays sequential; parallelism is at the corpus level only.
- **"Ad-hoc edit to configs/ during a launch"** — `training-runner` halts on config issues; route to `experiment-designer` to fix.
- **"Editing `train_command-new.sh`"** — that's the user's manual launch script. The agent owns `train_command-agent.sh` and never touches the user's.

## When to Parallelize

- Genuinely independent work: yes (e.g., math-reviewer + code-reviewer concurrently after a math-heavy implementation; N experiment-analyzer instances on N independent ablation cells).
- Mechanical batch work (same extraction across 32 runs): use a shell loop, not N parallel agents. Cheaper, simpler, easier to debug.
- Hand-off chains: sequential. Plan → implement → verify cannot be parallelized.

## Failure-Recovery Patterns

- **`training-runner` halts on missing config** → route to `experiment-designer`, then re-launch.
- **`training-runner` halts on missing node/GPU** → orchestration bug, not a runner bug. The caller (agent-manager or user) should have supplied node + GPU in the spawn prompt. Collect them via `AskUserQuestion` and re-spawn — but treat this as a one-off; the proper path is to ask upfront, before the first spawn.
- **`training-runner` halts on a plan-driven launch with `Status: running` or `completed` already** → the row was launched once already. Ask the user whether to re-launch (creates a duplicate WandB run) or pick a different row.
- **`experiment-analyzer` finds a manifest row with `Status: planned` or `running`** → run not ready. Skip and surface to user; do not block the rest of the analysis.
- **`env-config-auditor` flags 🔴** → resolve before any launch; loop back through `experiment-designer` if the design itself is wrong.
- **`developer`'s regression test passes pre-fix** → wrong test; back to `senior-developer` to revise the plan.
- **`developer` finds a file that needs changing but isn't in the plan** → halt, surface in Implementation Report; don't silently expand scope. `senior-developer` decides at verification.
- **WandB analysis surfaces a bug** → fork to bug-fix flow; cross-link both directions.
- **WandB analysis surfaces a missing metric** → fork to feature flow; the experiment may need re-running.

## Maintenance

- This doc replaces the prior workflow skills (`feature-workflow`, `bug-fix-workflow`, `training-experiment-workflow`, `parallel-literature-review`). Their procedural knowledge has been folded into the relevant agent profiles plus this orchestration layer.
- When you add a new agent, update this doc *only if* it introduces a new cross-cutting flow or constraint. Agent-internal procedures stay in the agent's profile.
- When an agent's scope changes, check whether any flow above needs updating.
