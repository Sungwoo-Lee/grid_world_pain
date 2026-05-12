# _global_tags.md — Global tag dictionary

> Check this file before inventing a new tag. If a suitable tag already exists, use it exactly as written.
> Tag drift leads to missed recall — use the canonical form.

**Last updated**: 2026-05-12

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
| `film` | `20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse` | FiLM-gating / conditional modulation (architecture, hyperparameters, behaviour under noise) |
| `refutation` | `20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse` | Negative-result insight: a pre-registered hypothesis is refuted (or a candidate cause is ruled out) |

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
| `precision` | Precision-weighting, noise sensitivity |
| `rl` | Reinforcement learning algorithm decisions |
| `wandb` | WandB logging, run tracking |
| `tradeoff` | Explicit tradeoff between two approaches |

---

## Change history

- 2026-05-12: 1 insight from the behavior-measure toolkit v1 application to R2.5 (`20260512_1428_sameprop_class_discriminating_defence_event_level`) reused existing tags `hypervigilance`, `learned_lesson`, `decision` — no new tags promoted.
- 2026-05-11: 2 insights from the Z2 verdict + diagnostic-bug-fix session (`20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt`, `20260511_1535_encode_decode_flag_mismatch_silent_class_bug`) reused existing tags `dreamer`, `learned_lesson`, `decision`, `refutation`, `meta` — no new tags promoted.
- 2026-05-10: 3 insights from the dreamer sheeprl-comparison + zero-init cascade session (`20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry`, `20260510_2240_reference_impl_compare_only_act_intersections`, `20260510_2241_residual_error_pattern_directs_next_fix`) reused existing tags `dreamer`, `learned_lesson`, `decision`, `refutation`, `subagent`, `meta` — no new tags promoted.
- 2026-05-10: 1 insight from the hypervigilance Round 2.5 launch + analysis session (`20260510_2237_sameprop_round25_no_class_avoidance`) reused existing tags `hypervigilance`, `refutation`, `learned_lesson`, `decision` — no new tags promoted.
- 2026-05-09: 3 insights from the NMN meta/continual pivot session (`20260509_1619_summarize_study_skill_design_and_ship`, `20260509_1620_documentation_framing_policy`, `20260509_1621_multi_agent_research_chain_v2_pattern`) reused existing tags `memory`, `design`, `decision`, `skill`, `meta`, `subagent`, `learned_lesson` — no new tags promoted.
- 2026-05-09: 4 insights from the dreamer conventional-fixes battery session (`20260509_1534_wm_reward_head_localized_failure_a1`, `20260509_1535_conventional_fixes_battery_verdict_predator_refute`, `20260509_1536_train_py_checkpoint_restore_nnx_skew`, `20260509_1537_professor_analysis_resets_exotic_investigation`) reused existing tags `dreamer`, `learned_lesson`, `decision`, `refutation`, `hypervigilance`, `meta`, `training_runner`, `subagent` — no new tags promoted.
- 2026-05-09: 3 insights from the hypervigilance Round 2 partial-verdict + per-tag metrics ship session (`20260509_1532_sameprop_round2_truncated_verdict`, `20260509_1533_tag_based_distance_supersedes_quadrant`, `20260509_1534_synthetic_smoke_masks_dict_assembly_bugs`) reused existing tags `hypervigilance`, `refutation`, `learned_lesson`, `design`, `decision`, `meta`, `subagent` — no new tags promoted.
- 2026-05-09: 2 insights from the temp_clip[0.5,10.0] re-run analyzer chain (`20260509_1409_nmn_tempceil10_verdict_h1b_confirmed_p4`, `20260509_1410_nmn_temp_head_natural_target_3_to_5`) reused existing tags `nmn`, `hypervigilance`, `film`, `refutation`, `learned_lesson` — no new tags promoted.
- 2026-05-09: 4 insights from the cluster-ops scripting consolidation session (`20260509_0309_cluster_py_consolidation`, `20260509_0310_bash_ic_alias_over_ssh`, `20260509_0311_diary_auto_session_backfill`, `20260509_0312_node_num_hostname_in_bashrc`) reused existing tags `meta`, `decision`, `learned_lesson`, `training_runner`, `design`, `memory` — no new tags promoted.
- 2026-05-08: 2 insights from the NMN heterogeneity-sweep analyzer chain (`20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse`, `20260508_2004_profile_dependent_temp_saturation_mc_film`) promoted `film` from candidate to active and added new active tag `refutation`. Existing tags `nmn`, `hypervigilance`, `learned_lesson` reused.
- 2026-05-08: 1 insight on Claude Code statusline script gotchas (`20260508_1826_statusline_jq_ifs_pct`) reused existing tags `meta`, `learned_lesson`, `decision` — no new tags promoted.
- 2026-05-08: 1 insight on scoping the lab SSH config block to vncuser reused existing tags `meta`, `decision`, `learned_lesson`, `training_runner` — no new tags promoted.
- 2026-05-08: 3 insights from the evaaa→episode container rebuild session reused existing tags `meta`, `decision`, `learned_lesson`, `training_runner` — no new tags promoted.
- 2026-05-08: Promoted `dreamer` from candidate to active during the dreamer hypervigilance investigation memorize session (4 insights, reconciled with the parallel NMN session — my IDs bumped to 1431–1434).
- 2026-05-08: Promoted `nmn`, `noise`, `hypervigilance`, `training_runner` from candidates to active during NMN heterogeneity-sweep capture session (3 insights into `nmn_diagnosis` + `cluster_ops`).
- 2026-05-08: Activated `skill` (new), `learned_lesson` (promoted from candidate), `worktree` (new), `subagent` (new). Added during `/memorize` skill rollout capture session.
- 2026-05-08: Created. Initial active tags: `memory`, `design`, `decision`, `meta` (from genesis insight).
