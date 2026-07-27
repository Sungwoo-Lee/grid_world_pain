---
aliases: [agent_playbook]
---

# Agent Playbook

The orchestration layer for the project's agent team. Read by `agent-manager` whenever it is producing a routing plan; readable by other agents (and the user) for reference. Each agent profile in `.claude/agents/` says *what that agent does*; this doc says *who comes after whom and when*. The `agent-manager` translates a request into a routing plan using this doc; the **parent (top-level Claude) executes the plan** by spawning the named sub-agents.

Keep this doc small. Anything agent-specific belongs in the agent's profile. Only **cross-cutting orchestration patterns** live here.

## Default Flows

### Adding a new feature / capability to the platform

```
[pi consultation — only for roadmap-level / multi-week plans]
        ↓
senior-developer (plan, docs/develop/)
        ↓
plan-reviewer   (pre-mortem on the plan)      ← User approves plan (blockers resolved first)
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

`pi` is invoked **only** when the planned scope is roadmap-level (multi-week) or when the plan is platform-only without a clear paper hook — to confirm the work fits the active publication tracks before `senior-developer` invests planning effort. Routine feature adds and bug fixes skip the PI.

### Fixing a bug

```
senior-developer (root cause + fix plan)
        ↓
plan-reviewer   (pre-mortem on the fix plan)  ← User approves diagnosis + fix
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
plan-reviewer       (pre-mortem on the design: controls, seeds / statistical power,
                     confounds, pre-registered refutation criteria, GPU feasibility,
                     budget wiring)
        ↓
env-config-auditor  (audit; resolve all 🔴 before launch)
        ↓
pi                  (portfolio-level focus-vs-explore call; AskUserQuestion;
                     log under docs/pi/calls/)
                    Skipped for one-off ad-hoc launches; required for any
                    multi-run / multi-day experiment series.
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
        ↓
pi                  (post-comparison: deepen / pivot / shelve?
                     AskUserQuestion; log under docs/pi/calls/)
                    Required after any multi-run comparison; user decides next move.
```

The Launch Manifest is the **system-of-record** binding experimental cells to WandB folders. Three agents share it with **strict column ownership**:

| Agent | Reads | Writes |
|---|---|---|
| `experiment-designer` | — | Planned columns (Run, Cell, Tag, wandb-group, wandb-job-type, Seed) + §3.1 |
| `training-runner` | Planned columns (to apply at launch) | Actual columns (Status, Node, GPU, Launched at, WandB run ID, Log path) of the row it launched |
| `experiment-analyzer` | Whole table (run discovery) | — (analyzer never writes the manifest) |

`training-runner` does NOT pick its own node/GPU — there is no monitoring source available to it. The `agent-manager` flags this as an `[ASK USER UPFRONT]` item in its routing plan, and the **parent (top-level Claude)** collects node + GPU before spawning the runner and passes them in the spawn prompt. Bundling this with the launch-approval step is cheaper than letting the runner spawn, halt, and require a re-spawn.

For **one-off launches with no plan doc** (ad-hoc / debugging), the runner falls back to its default tag/wandb-name convention (see its profile §3a Path B). The convention is parallel to (and consistent with) the designer's manifest convention, so ad-hoc and planned runs interleave cleanly in WandB.

If an experiment-analyzer or experiment-designer adds a `## Metrics Requested` section, the user reviews and (if accepted) escalates to the feature flow to add the logger.

### Analyzing existing runs (no prior design)

```
experiment-analyzer (Mode B — retroactive hypothesis frame, full doc to docs/experiments/)
        ↓
pi                  (only if the analysis spans 3+ runs or changes a track-level
                     question; otherwise skip)
```

If the analysis surfaces a bug → fork to bug-fix flow. If a missing metric → fork to feature flow.

### New-direction proposal from a researcher

```
research-postdoc OR literature-curator OR a professor
        ↓
pi                  (does this fit an active track, open a new one, or get shelved?
                     AskUserQuestion; log under docs/pi/calls/)
        ↓
USER decision → hand off to experiment-designer / senior-developer /
                another researcher per the user's pick
```

The PI is the gate between researcher-generated direction memos and downstream commitment of GPU-weeks or engineering-weeks.

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
- After all shards complete, merge into `docs/project/references/<topic>/<topic>_lit_review.md`. Each topic folder follows a fixed layout: review docs at the topic root, raw source PDFs/.txt extracts inside a `sources/` subfolder (`docs/project/references/<topic>/sources/*.pdf`). `<topic>` is the exact name of the source-PDF subfolder under `docs/project/references/`. Dedupe at merge (a paper may have ended up in two shards).
- Per-paper processing inside each shard stays sequential — the 4-step backbone needs focused attention per paper.
- For thematic regrouping after the merge, hand off to `literature-curator`.

## Cross-Cutting Constraints

These apply across multiple flows. The `agent-manager` flags them as preconditions in its routing plan; the **parent** enforces them at spawn time. Other invoking agents (when bypassing the manager for single-agent tasks) enforce them directly.

- **PI consultation at major decision points.** The `agent-manager` flags `pi` as a canonical step before launching a multi-run experiment (after `env-config-auditor` clean, before `training-runner`), after `experiment-analyzer` finishes a multi-run comparison, when `senior-developer` drafts a roadmap-level plan, and when `research-postdoc` / `literature-curator` propose a new direction. `pi` uses `AskUserQuestion` to surface the focus-vs-explore call; the user makes the final pick. `pi` does NOT run for bug fixes, single-config tweaks, doc-only edits, single-paper literature reviews, or one-off ad-hoc launches.
- **env-config-auditor in parallel** with senior-developer's verification, whenever the diff touches configs/, src/environment/, or sensor/observation/noise code. Two orthogonal checks: SD verifies plan adherence; auditor verifies env↔config soundness.
- **WandB analysis uses local files only.** `wandb/run-YYYYMMDD_HHMMSS-<id>/` — never the WandB web API.
- **Survival steps as headline metric** for any RL evaluation. Cumulative reward is at best a secondary diagnostic.
- **Temporal evolution mandatory** in any training analysis — not just end-of-training snapshots.
- **No fallback defaults** for critical config params — `config.get_mandatory()` everywhere, missing key → `ValueError`. Plans list new keys explicitly.
- **Frontmatter contracts**: develop tree is auto-INDEX'd via `scripts/claude/regen_dev_index.py`; experiments tree is by-convention only (no INDEX). Don't run `regen_dev_index.py` for `docs/experiments/`.
- **Soft-split rule**: pre-existing experiment-shaped docs under `docs/develop/active/{hypervigilance,noise,diagnosis}/` are NOT migrated retroactively. Read as reference, write new docs to `docs/experiments/active/<topic>/`.
- **Dirty working tree for verification.** `developer` does NOT commit; `senior-developer` reads the uncommitted diff. Once committed, the verification signal is lost.

## Anti-Patterns to Catch

The `agent-manager` flags these in its routing plan; the parent (or any directly-invoking agent) pushes back on them before spawning sub-agents:

- **"Analysis but no design exists"** — user asks for analysis of pre-registered hypothesis but there's no design doc. Push back: do you want Mode B post-hoc (weaker), or should we design first?
- **"Bug fix but the plan rewrites the design"** — root cause is a project-design issue, not a localized bug. Escalate to feature flow.
- **"Skip user approval because the plan looks obvious"** — there is no such thing as an obviously correct plan. Always pause at user-approval gates.
- **"Skip the PI on a multi-week / multi-paper-shaped commitment"** — the PI's whole purpose is the focus-vs-explore call at portfolio scope. A roadmap-level plan or a multi-run experiment series that goes straight from `experiment-designer` to `training-runner` without a PI call is a missed opportunity to ask "is this the right thing to spend GPU-weeks on?". Conversely, **invoking `pi` for a one-line config tweak or a routine bug fix** is also wrong — the PI declines and tells the user.
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
- **`training-runner` halts on missing node/GPU** → orchestration bug, not a runner bug. The parent should have supplied node + GPU in the spawn prompt (the `agent-manager`'s routing plan flags this as an upfront ask). Collect them via `AskUserQuestion` and re-spawn — but treat this as a one-off; the proper path is to ask upfront, before the first spawn.
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
