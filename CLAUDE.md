## Agent Team

Delegate to the matching agent — read its profile in `.claude/agents/` for full responsibilities and tool scope.

| Agent | Model | Role | Scope |
|---|---|---|---|
| [agent-manager](.claude/agents/agent-manager.md) | opus | Plan multi-agent flows; route + sequence + parallelize. Returns a routing plan to the parent (top-level Claude), which spawns the sub-agents | returns plan; writes nothing; does NOT spawn |
| [pi](.claude/agents/pi.md) | opus | Principal Investigator — portfolio-level focus-vs-explore calls at major decision points. Surfaces 2–4 candidate paths via `AskUserQuestion`; user decides; PI logs the call. Manual + proactive triggers (pre-launch, post-analysis, roadmap-level plan, new-direction proposal) | `docs/pi/` (primary); cross-process feedback append-allowed under any `docs/` subtree |
| [senior-developer](.claude/agents/senior-developer.md) | opus | Platform-development planning + post-impl verification | `docs/develop/` |
| [developer](.claude/agents/developer.md) | sonnet | Implement approved plans, test, report | full code |
| [code-reviewer](.claude/agents/code-reviewer.md) | opus | JAX/Flax/vmap/PRNG correctness review | `docs/reviews/` |
| [math-reviewer](.claude/agents/math-reviewer.md) | opus | Verify equations match cited papers | `docs/reviews/` |
| [env-config-auditor](.claude/agents/env-config-auditor.md) | sonnet | YAML/env soundness, obs↔noise sync, pre-flight before training | `docs/reviews/` |
| [experiment-designer](.claude/agents/experiment-designer.md) | opus | Experiment design + config generation | `configs/`, `docs/experiments/active/<topic>/` |
| [experiment-analyzer](.claude/agents/experiment-analyzer.md) | opus | Post-hoc training-result analysis (WandB, run comparisons) | `docs/experiments/active/<topic>/` |
| [training-runner](.claude/agents/training-runner.md) | sonnet | Pre-flight check + launch training on lab nodes (101–114) via `run_command.py`; configs are read-only | `train_command-new.sh` |

### Researchers

Domain-expert agents that generate mathematical concepts, literature reviews, and publication-direction memos for the project. Their **primary write home is `docs/project/`** (their own standalone memos), but they may also **append cross-process feedback** to any doc under `docs/` — including `docs/develop/`, `docs/experiments/`, `docs/reviews/`, `docs/pi/` — when invited to comment on an in-flight plan, design, analysis, or strategic call. They **never** edit `src/`, `configs/`, or `scripts/`. Cross-process feedback must always **append** (never silently rewrite) and be signed with a clear "Feedback from `<agent-name>`" header so the original author's voice stays distinct. Recommendations that imply code or experiment changes are handed off (named) to `senior-developer`, `experiment-designer`, etc.

| Agent | Model | Role | Primary write home (under `docs/project/`) |
|---|---|---|---|
| [research-postdoc](.claude/agents/research-postdoc.md) | opus | First responder for open-ended research questions; triages to professors or writes first-pass synthesis | `ideas/`, `triage/` |
| [professor-bayesian-brain](.claude/agents/professor-bayesian-brain.md) | opus | Perceptual decision making, predictive coding, active inference, Bayesian decision theory | `concepts/`, `directions/`, `critiques/` |
| [professor-pain-modeling](.claude/agents/professor-pain-modeling.md) | opus | Computational pain science; construct-validity guardian for "pain-like" claims | `concepts/`, `directions/`, `critiques/` |
| [professor-rl-bayesian-dl](.claude/agents/professor-rl-bayesian-dl.md) | opus | RL, Bayesian deep learning, FiLM / hypernet / conditional architectures | `concepts/`, `directions/`, `critiques/` |
| [professor-neuromodulation](.claude/agents/professor-neuromodulation.md) | opus | Computational models of ascending modulatory systems (ACh / NE / DA / 5-HT / opioid); biological-plausibility guardian | `concepts/`, `directions/`, `critiques/` |
| [literature-reviewer](.claude/agents/literature-reviewer.md) | opus | Per-paper review (backbone + Phase 1/2 LaTeX) of PDFs in `docs/project/references/<topic>/sources/` | `references/<topic>/<topic>_lit_review.md` (review at topic root, source PDFs in `sources/`) |
| [literature-curator](.claude/agents/literature-curator.md) | opus | Cross-paper synthesis, TOC, thematic regrouping of existing master reviews | `references/<topic>/<topic>_lit_review.md`, `references/<topic>/<topic>_synthesis.md` |

**Researcher routing:** open-ended research questions ("what direction?", "is there a connection between …?", "give me ideas") default to `research-postdoc`, which triages and either writes a first-pass synthesis or hands off to one or more professors / literature agents. Direct invocation is fine when the question is unambiguously in one agent's domain (e.g., "review these PDFs" → `literature-reviewer`, "regroup the lit review by theme" → `literature-curator`).

**Default routing:** for any task involving 2+ agents in sequence — feature add, bug fix, training experiment, literature review of 10+ papers, post-impl verification — spawn `agent-manager` to get a routing plan, then **spawn the named sub-agents yourself** (top-level Claude executes the plan; the manager has no `Agent` tool). Single-agent tasks (e.g., "audit this config", "review this PR", "ask a professor", "/pi") bypass the manager and route directly.

**PI consultation (proactive):** at the major decision points named in [.claude/agents/pi.md](.claude/agents/pi.md) — pre-launch of a multi-run experiment, post-analysis of a multi-run comparison, a roadmap-level plan, a new research-direction proposal — the `agent-manager` flags PI consultation as a canonical step in its routing plan; the parent then spawns `pi`, which surfaces the focus-vs-explore trade-off to the user via `AskUserQuestion`. The user makes the final call. The PI does not run for bug fixes, single-config tweaks, or one-off launches.

The canonical flows, parallelism heuristics, cross-cutting constraints, and anti-patterns are documented in [docs/AGENT_PLAYBOOK.md](docs/AGENT_PLAYBOOK.md). The manager reads this whenever it produces a routing plan; other agents may reference it for context.

---

## Working Principles

- **Think before coding.** State assumptions; if uncertain, ask. If multiple interpretations exist, surface them — don't pick silently. If a simpler approach exists, say so.
- **Ask actively, not always.** Use the `AskUserQuestion` tool whenever a decision could reasonably go more than one way and picking silently risks rework — ambiguous scope, unstated constraints, multiple plausible interpretations, or trade-offs the user should own (perf vs. simplicity, breaking change vs. shim, which file to touch). Not every task needs a question, but err on the side of asking rather than guessing. Batch related questions into one prompt; don't drip them.
- **Simplicity first.** Minimum code that solves the stated problem. No speculative features, abstractions for single-use code, configurability that wasn't requested, or error handling for impossible scenarios. If 200 lines could be 50, rewrite.
- **Surgical changes.** Every changed line should trace to the request. Don't "improve" adjacent code, refactor what isn't broken, or restyle to your preference. Remove orphans your changes created; don't delete pre-existing dead code unless asked — mention it instead.
- **Goal-driven execution.** Reframe tasks as verifiable goals before starting ("fix the bug" → "write a test that reproduces it, then make it pass"). For multi-step work, state a short plan with a verification check per step so you can loop without re-asking.

---

## Project-Wide Rules

- **Conda env.** All Python is run inside the `grid_world_pain` conda env. Invoke the interpreter directly at `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python <script>`. Do not use `conda run` / `conda activate`, and never invoke the system `python3`. (Per-experiment exception: the `sheeprl_bridge` env exists for the upstream-sheeprl drop-in test under [`docs/develop/active/diagnosis/sheeprl_training_howto.md`](docs/develop/active/diagnosis/sheeprl_training_howto.md). Same explicit-interpreter rule — never `conda run`.)
- **Launching training on lab nodes.** Always use `./run_command.py <node> "<command>"` — never raw SSH. The wrapper handles port 1800, SSH multiplexing, `nohup`, and timestamped logs under `<PROJECT_ROOT>/logs/<timestamp>.log` (NAS-shared, visible from any node and from the launcher). It deliberately does **not** `cd` to the project root or activate any conda env — the bash command you pass in owns both. So every launch script must start with:
    ```bash
    #!/bin/bash
    set -euo pipefail
    cd /media/nas01/projects/Interoceptive-AI/grid_world_pain
    # ... then call the explicit interpreter:
    /home/vncuser/miniconda3/envs/<env>/bin/python <script.py> <args>
    ```
    Reusable workloads go in `scripts/launch_*.sh` (see `scripts/launch_sheeprl.sh` as a template — takes `<config-yaml> <gpu> <tag> [steps]`). The 2026-05-12 refactor moved `cd`/conda out of the wrapper after raw-SSH launches landed in `$HOME/` and died silently. Past templates: `train_command-new.sh` (user-edited), `train_command-agent.sh` (training-runner-edited), `scripts/launch_sheeprl.sh` (sheeprl).
- **Survival-step evaluation.** Performance is measured in survival steps, never cumulative reward.
- **No fallback defaults.** Critical configs use `config.get_mandatory('key')`; missing key → `ValueError`. New keys listed in the plan's File Changes section.
- **Templates.** Plans go in `docs/` using [issue_plan](docs/TEMPLATES/issue_plan.md) (what to build/fix) or [training_analysis](docs/TEMPLATES/training_analysis.md) (what happened — hypothesis-driven). Cross-link related docs both directions.
- **Develop docs.** Files under `docs/develop/` carry YAML frontmatter (`title, topic, status, created, last_updated`) per [FRONTMATTER_CONTRACT.md](docs/develop/active/meta/FRONTMATTER_CONTRACT.md). `docs/develop/INDEX.md` is auto-generated by `scripts/regen_dev_index.py` — do not hand-edit. When superseding a doc, set `status: superseded`, link with `supersedes:` / `superseded_by:`, and use `git mv` to move it under `archive/` so history is preserved.
- **Working files.** Intermediate results to `tmp/YYYYMMDD_HHMMSS_<topic>.md`, written after each step.
- **Parallelism.** OK for independent tasks; sequential for hand-off chains; shell loops beat parallel agents for mechanical batch work.
- **Auto-commit.** Standing authorization to `git add` and `git commit` without asking, at logical sub-task boundaries — one coherent change per commit, separate commits for unrelated work (do not batch). Skip if the change is incomplete (failing tests, half-done implementation). Stage specific files by name (never `git add -A` / `.`). Match the existing message style — check `git log` for the project's conventional-commit + gitmoji format. Never commit secrets, never use `--no-verify`, never push.
- **Git safety (DO NOT WIPE GITIGNORED DATA).** This repo lives on a NAS that **does not support symlinks**, so the standard "move data outside the repo and symlink in" protection is unavailable. Gitignored data (`results/`, `claude_data/`, `wandb/`, `logs/`, `tmp/`, `legacy/`, `antigravity_data/`, etc.) is therefore destroyed unrecoverably by any aggressive `git clean`. Hard rules:
  - **Never** use `git clean -x` / `-X` / `-fdx` / `-fdX` — the `-x`/`-X` flag deletes gitignored files. Allowed: `git clean -fd` (untracked-only; gitignored survives). Always preview with `git clean -fdn` (dry-run) first, and surface the listed paths to the user before executing the real clean.
  - **Never** force-checkout / force-switch between branches without first checking whether the destination branch tracks paths that are currently untracked locally — `git checkout -f <branch>` will overwrite an untracked file at a path tracked on the destination.
  - **Avoid** `git stash -u` followed by `git stash drop` — `-u` carries untracked into the stash, and `drop` then loses them. Use `git stash list` + `git stash show -u` to confirm before any drop.
  - **`git reset --hard` is safe** for gitignored data (only resets tracked files); the danger is in `git clean -x` that often follows it.
  - **Before any merge / rebase / branch switch / non-trivial git operation**, snapshot critical untracked data: `cp -a results /tmp/results-bk-$(date +%s)` (or `claude_data`, `wandb`, etc., as relevant). The cost is one second; the alternative is re-training.
  - **Past incident**: the `results/` directory (training outputs across `JAX_RecurrentPPO` / `JAX_DreamerV3` / `JAX_Sandbox` / `JAX_PPO`) was lost during a failed merge when destructive cleanup followed. Recovery required re-training. This rule exists because of that incident.

---

## Documentation framing

Every plan, design, analysis, summary, review, and direction document the project produces must lead with a **plain-language interpretation** that a reader without prior context can follow. Docs in this project get re-read by future-Claude, parallel sessions, the user months later, and (eventually) external collaborators or reviewers — so a doc whose entry point requires the reader to already know `H₁a` / `Δ_SS` / `f96lhxpe` / `01-interoNocicept_sameProp.yaml` is a doc that gates re-readers behind a context tax.

### The rule

The **first body section** of any plan / design / analysis / summary / review / direction doc is a plain-English entry point — title in the family of *Question*, *Purpose*, *Context*, *Headline finding*, *Verdict*, *Study question*, or equivalent. Its 200-word job is to tell a fresh reader **what the doc is about, why it exists, and what it's claiming**.

In that entry-point section:

- **Translate every cited result on first mention.** "The modulator did not beat the baseline (H₁a refuted)", not just "H₁a refuted". The English first; the symbol after, in parens.
- **No bare WandB run IDs** like `f96lhxpe` — link through the design doc or memory insight that names them.
- **No bare config paths** like `configs/models/recurrent_ppo_nmn_het_film_g1.yaml` — describe what the config does ("the FiLM agent config with the modulator's temperature ceiling raised from 3.0 to 10.0") and link the path elsewhere.
- **No bare predicate / shorthand names** (`H₁a`, `H₀`, `Δ_SS`, `Cand. A1`, `Phase 0`, `T/P split`, etc.) without a one-clause translation.
- **Concrete examples beat abstract claims.** If the doc proposes "a 2-context mixture", show what the two contexts ARE — not just "context A vs. context B".

Symbolic / numerical / file-path / equation-heavy detail belongs in **later sections** of the same doc — Methods, Manifest, Links, Derivations, Tables, Appendix. The rule is about the *entry point*, not the whole document. Math-heavy memos still earn their math; they just have to introduce their question without it.

### The check

Open the document, read the first ~200 words. Could a fresh reader who has not seen the prior memos / design docs / commit history understand what this doc is about, why it exists, and what it's claiming? If yes, it passes. If no, rewrite the entry point.

### Where this rule is encoded

- **Templates**: [docs/TEMPLATES/issue_plan.md](docs/TEMPLATES/issue_plan.md) (Context section) and [docs/TEMPLATES/training_analysis.md](docs/TEMPLATES/training_analysis.md) (Research Question section) enforce the structure at the point new docs are authored.
- **Agent profiles**: every doc-producing agent (`senior-developer`, `experiment-designer`, `experiment-analyzer`, `research-postdoc`, the four professors, the three reviewers, `literature-reviewer`, `literature-curator`) carries a one-line reference to this rule in its profile.
- **Worked example**: the `summarize-study` skill ([.claude/skills/summarize-study/SKILL.md](.claude/skills/summarize-study/SKILL.md)) encodes this rule in its strictest form, with a concrete check-and-replace table — read it for a worked example of what good output looks like.

The rule applies to every new doc going forward. **Existing docs are not retroactively rewritten** unless a reader is actively confused by one.

---

## Session memory

This project carries two memory layers; future-Claude must know which one to write to.

- **Built-in auto-memory** at `~/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/memory/MEMORY.md` — short typed rules another agent must obey on every invocation, with a sibling `feedback_*.md` per rule. User-and-machine-local; not under git.
- **In-repo session memory** at `.claude-memory/` — multi-section session insights with rationale, decisions, follow-ups, and an optional raw-conversation archive. Version-controlled with the repo.

When memory work is requested, read `.claude-memory/CLAUDE.md` (operating manual) and `.claude-memory/ROOT_INDEX.md` (topic registry) before capturing or recalling. The division-of-labor decision rule and capture triggers live in `.claude-memory/CLAUDE.md`; the design rationale and worked routing examples live in [docs/develop/active/meta/claude_memory_system_design.md](docs/develop/active/meta/claude_memory_system_design.md).

---

## Diary protocol

The project keeps a daily event log at [docs/diary/YYYY-MM-DD.md](docs/diary/) — a single-glance status board across all parallel Claude sessions. **Update it whenever a notable event fires, even without an explicit user request.** The diary is the cross-session index; missing entries create invisible gaps when another session looks at "what's happening today".

The mechanism is the `/diary` skill ([.claude/skills/diary/SKILL.md](.claude/skills/diary/SKILL.md)), which calls `scripts/diary_append.py` (flock-protected, safe under parallel calls). Subcommand-to-event mapping:

| When this happens | Call this subcommand |
|---|---|
| Top-level Claude session begins a multi-step task | `session-start --label … --summary … [--link plan-or-doc]` |
| Same session wraps up | `session-end --label … --commits "<hashes>"` |
| `developer` reports an implementation complete | `implemented --subject … --link <commit-hash-or-plan-doc>` |
| `senior-developer` reports a verification complete | `verified --subject … --link <plan-doc>` |
| `/memorize` writes 1+ insights | already chained — `/memorize` Step 8 calls `diary_append.py insight ...` once per insight |
| `training-runner` launches a training | `training-start --tag … --node … --gpu … --cell … --wandb … --doc <design-doc>` |
| `experiment-analyzer` finishes an analysis | `training-done --tag <same-as-start> --result "<one-line>" --analysis <analysis-doc>` |
| Multi-step session is wrapping up (in addition to `session-end`) | `progress-report --title … --what-this-did … --headline … --whats-next … --sources …` |

Pass **raw values** (commit hashes, repo-relative paths) to `--link` / `--doc` / `--analysis` / `--commits` — the script auto-formats them as `commit `<hash>`` or `[stem](relative-path)`. Tags must match between `training-start` and `training-done`; session labels must match between `session-start` and `session-end`.

Every row carries a **Session** column showing which Claude session wrote it. The Sessions table carries the **full UUID** (copy-paste into `claude --resume <UUID>` to resume the session in a new terminal); Events and Training runs carry the **8-char prefix** for compactness — cross-reference back to the Sessions row to recover the full UUID. Top-level Claude calls inherit both formats automatically from `$CLAUDE_CODE_SESSION_ID`. Sub-agents must pass `--session "${CLAUDE_CODE_SESSION_ID:0:8}/<role>"` explicitly so rows read `f3ab7f37/developer`, `f3ab7f37/training-runner`, etc. — the slash separator makes parent→sub-agent lineage visible at a glance. Each agent profile under `.claude/agents/` documents its exact `--session` value.

Agent profiles in `.claude/agents/` carry their own diary-update reminder for their specific subcommand. Top-level Claude is responsible for `session-start` / `session-end`. The `/memorize` chain handles `insight` rows automatically.
