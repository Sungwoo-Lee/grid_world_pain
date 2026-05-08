# _global_tags.md — Global tag dictionary

> Check this file before inventing a new tag. If a suitable tag already exists, use it exactly as written.
> Tag drift leads to missed recall — use the canonical form.

**Last updated**: 2026-05-08

---

## Active tags

| Tag | First-use insight ID | Definition |
|---|---|---|
| `memory` | `20260508_0315_claude_memory_system_genesis` | Relates to the `.claude-memory/` layer itself |
| `design` | `20260508_0315_claude_memory_system_genesis` | Architecture or system-design decision |
| `decision` | `20260508_0315_claude_memory_system_genesis` | A specific choice made with rationale |
| `meta` | `20260508_0315_claude_memory_system_genesis` | About the project tooling / infrastructure, not the science |
| `skill` | `20260508_0429_memorize_skill_design_and_ship` | A Claude Code skill (under `.claude/skills/`) — design, ship, eval |
| `learned_lesson` | `20260508_0430_worktree_isolation_path_safety` | Post-mortem finding or debugging conclusion |
| `worktree` | `20260508_0430_worktree_isolation_path_safety` | Git worktree usage, isolation, lifecycle |
| `subagent` | `20260508_0430_worktree_isolation_path_safety` | Claude Code Agent-tool subagents — spawning, isolation, prompts |
| `nmn` | `20260508_1426_v8_noise_bug_refuted` | Neuromodulatory network (NMN) architecture / diagnosis |
| `noise` | `20260508_1426_v8_noise_bug_refuted` | Observation noise, noise heterogeneity, perceptual-noise configs |
| `hypervigilance` | `20260508_1427_nmn_heterogeneity_sweep_design` | Hypervigilance experiments and results |
| `training_runner` | `20260508_1428_node_env_recovery_recipe` | Training-runner agent, launch workflow, lab-node infra |
| `dreamer` | `20260508_1431_diagnostic_battery_refutes_four_fixes` | DreamerV3 model/config decisions, failure-mode diagnoses |

---

## Tag-writing rules

- English only.
- snake_case, singular form preferred (e.g., `decision` not `decisions`, `tradeoff` not `trade-offs`).
- No hierarchical slashes (e.g., use `nmn` not `model/nmn`).
- ≥ 1 tag per insight; aim for 2–4.
- Before adding a new tag: scan this table. If a close match exists, use it.
- When adding a new tag: append a row to the Active tags table with the insight ID where it first appeared.

---

## Starter-candidate tags (project vocabulary)

These are not yet active — they become active when first used in an insight. Reference this list to avoid inventing near-duplicate tags.

| Candidate | Intended scope |
|---|---|
| `film` | FiLM-gating / conditional modulation |
| `precision` | Precision-weighting, noise sensitivity |
| `rl` | Reinforcement learning algorithm decisions |
| `wandb` | WandB logging, run tracking |
| `tradeoff` | Explicit tradeoff between two approaches |

---

## Change history

- 2026-05-08: Promoted `dreamer` from candidate to active during the dreamer hypervigilance investigation memorize session (4 insights, reconciled with the parallel NMN session — my IDs bumped to 1431–1434).
- 2026-05-08: Promoted `nmn`, `noise`, `hypervigilance`, `training_runner` from candidates to active during NMN heterogeneity-sweep capture session (3 insights into `nmn_diagnosis` + `cluster_ops`).
- 2026-05-08: Activated `skill` (new), `learned_lesson` (promoted from candidate), `worktree` (new), `subagent` (new). Added during `/memorize` skill rollout capture session.
- 2026-05-08: Created. Initial active tags: `memory`, `design`, `decision`, `meta` (from genesis insight).
