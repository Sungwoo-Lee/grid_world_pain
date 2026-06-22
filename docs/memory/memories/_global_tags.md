# _global_tags.md — Global tag dictionary

> Check this file before inventing a new tag. If a suitable tag already exists, use it exactly as written.
> Tag drift leads to missed recall — use the canonical form.

**Last updated**: 2026-06-22

---

## Active tags

| Tag | First-use insight ID | Definition |
|---|---|---|
| `memory` | `20260508_0315_claude_memory_system_genesis` | Relates to the `docs/memory/` layer itself |
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
| `config` | `20260619_0111_config_v3_extends_layering_default_base` | Config loader / layering / schema / authoring system |

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
- 2026-06-22: 1 insight into the new `curriculum_learning` folder (`20260622_1748_basic_curriculum_overtraining_collapse_and_intervals`) reused existing tags `learned_lesson`, `decision` — no new tags promoted.
- 2026-06-22: 2 insights from the dreamer_srl basic-curriculum recompile-storm session (`20260622_1746_dreamer_srl_recompile_storm_done_count` into `dreamer_diagnosis`, `20260622_1747_dreamer_srl_single_config_budget_source` into `cluster_ops`) reused existing tags `dreamer`, `learned_lesson`, `decision`, `meta`, `training_runner` — no new tags promoted.
- 2026-06-22: 4 insights from the olfactory-ambiguity / hunger-gated training program session (`20260622_1744_olfactory_uncertainty_two_knob_design`, `20260622_1745_discrimination_weak_lethality_masks_gating`, `20260622_1746_olfactory_decay_power_distal_cue_strength`, `20260622_1747_hypervigilance_asymmetric_stakes_scarcity`) reused existing tags `hypervigilance`, `design`, `decision`, `refutation`, `learned_lesson`, `noise` - no new tags promoted.
- 2026-06-22: 3 insights from the conflict/hypervigilance behavior-probe session (`20260622_1744_hypervig_probe_hg10_overgeneralizes_threat`, `20260622_1745_frozen_probe_eval_match_sensory_renderer`, `20260622_1746_start_injury_dead_needs_random_range`) reused existing tags `hypervigilance`, `learned_lesson`, `decision`, `config` — no new tags promoted.
- 2026-06-22: 1 insight into existing `cluster_ops` (`20260622_1704_continual_bm_transition_nameerror_refactor_drift`) reused existing tags `learned_lesson`, `decision`, `meta` — no new tags promoted.
- 2026-06-19: 4 insights from the v3.0 config-system overhaul session (`20260619_0111_config_v3_extends_layering_default_base`, `20260619_0112_configurable_visual_properties_and_std`, `20260619_0113_configurable_initial_state_ranges`, `20260619_0114_config_guide_maintenance_contract`) promoted new tag `config`. Existing tags `design`, `decision`, `meta`, `learned_lesson` reused. New folder `config_system` opened.
- 2026-06-16: 1 insight from the testbed-search reframe (`20260616_1514_experimental_env_as_behavior_platform`) reused existing tags `hypervigilance`, `design`, `decision` — no new tags promoted.
- 2026-06-16: 1 insight from the autonomous discrimination-metric testbed search (`20260616_0142_discrimination_is_spatial_encounter_artifact`) reused existing tags `hypervigilance`, `refutation`, `learned_lesson`, `decision` — no new tags promoted.
- 2026-06-15: 1 insight from the predator-rabbit measurement-direction rethink + testbed-charter session (`20260615_1612_testbed_isolation_makes_means_honest`) reused existing tags `hypervigilance`, `design`, `decision`, `learned_lesson` — no new tags promoted.
- 2026-06-09: 1 insight from the predator-only control session (`20260609_1747_avoidance_is_post_contact_not_preemptive`, supersedes `20260609_1719`) reused existing tags `hypervigilance`, `refutation`, `learned_lesson` — no new tags promoted.
- 2026-06-09: 3 insights from the docs/environment v2.0 re-sync + JAX-tutorial conversion session (`20260609_1724_verbatim_embed_fidelity_diff_check`, `20260609_1725_env_docs_tutorial_primer_pattern`, `20260609_1726_doc_audit_surfaces_latent_bugs`) reused existing tags `subagent`, `learned_lesson`, `meta`, `decision`, `design` — no new tags promoted.
- 2026-06-09: 4 insights from the chasing-rabbit (R4) behaviour deep-dive session (`20260609_1719_predator_discrimination_visual_count_elimination`, `20260609_1720_chasing_rabbit_avoidance_damage_driven`, `20260609_1721_aggregate_stats_hide_conditional_behavior`, `20260609_1722_renderer_no_neutral_icon_and_attack_delay_ride`) reused existing tags `hypervigilance`, `learned_lesson`, `decision`, `refutation`, `meta`, `design` — no new tags promoted. 3 into `hypervigilance`, 1 into `env_entities`.

- 2026-05-29: 3 insights from the dreamer-srl v2 post-fix relaunch + WandB/config-save parity + log_interval-cadence session (`20260529_1824_dreamer_srl_wandb_spread_and_config_save_parity`, `20260529_1825_log_interval_anchored_rows_per_session`, `20260529_1826_lazy_import_schema_drift_first_call_crash`) reused existing tags `dreamer`, `meta`, `learned_lesson`, `decision`, `design`, `refutation`, `training_runner` — no new tags promoted. 2 into `cluster_ops`, 1 into `dreamer_diagnosis`.
- 2026-05-29: 1 insight from the v2.0 env_entities CP1-CP6 ship + R3 predator-distributional design session (`20260529_1823_unified_animal_entity_v2_0_arch`) reused existing tags `design`, `decision`, `learned_lesson`, `meta` — no new tags promoted. New folder `env_entities` opened (genesis insight).
- 2026-05-28: 1 insight from the v2.0 env_entities refactor session (`20260528_1647_bg_isolation_subagent_bypass`) reused existing tags `worktree`, `subagent`, `learned_lesson`, `meta`, `decision` — no new tags promoted. Extends `20260519_1810_bg_isolation_blocks_edit_not_bash` with the sub-agent-bypass finding.
- 2026-05-28: 4 insights from the EPISODE project_plan rewrite + Foam wikilink convention + VSCode startup fix session (`20260528_0215_foam_excludes_required_not_search_exclude`, `20260528_0216_foam_wikilinks_filename_as_id_sparse_aliases`, `20260528_0217_episode_direction_4x3_framework_two_papers`, `20260528_0218_git_lock_parallel_session_contamination`) reused existing tags `meta`, `learned_lesson`, `decision`, `memory`, `design`, `nmn` — no new tags promoted.
- 2026-05-25: 1 insight from the Claude Code statusline / `rate_limits.*` session (`20260525_2258_claude_code_statusline_rate_limits_official`) reused existing tags `meta`, `learned_lesson`, `decision` — no new tags promoted.
- 2026-05-19: 2 insights from the notebooklm community-skill → official `notebooklm-py` package swap session (`20260519_1809_notebooklm_py_official_skill_install`, `20260519_1810_bg_isolation_blocks_edit_not_bash`) reused existing tags `skill`, `learned_lesson`, `decision`, `meta`, `worktree`, `subagent` — no new tags promoted.
- 2026-05-19: 3 insights from the dreamer-srl v2 perf-diagnosis + JAX-Dreamer perf-retrofit history session (`20260519_1507_dreamer_srl_v2_cpu_buffer_regression`, `20260519_1508_dreamer_jax_perf_retrofit_4_phases`, `20260519_1509_nnx_lax_scan_split_merge_pattern`) reused existing tags `dreamer`, `learned_lesson`, `design`, `decision`, `meta` — no new tags promoted.
- 2026-05-18: 6 insights from the dreamer-srl v2 parity + 10×10 hyperparameter search session (`20260518_1511_dreamer_srl_v2_parity_pass_outperform`, `20260518_1512_reinforce_resampling_bug_imag_action_threading`, `20260518_1513_production_recipe_xs_16_4m_hypervigilance`, `20260518_1514_num_envs_vs_budget_interaction`, `20260518_1515_m_paradox_resolution_slow_learner`, `20260518_1516_wandb_log_dict_timesteps_key`) reused existing tags `dreamer`, `hypervigilance`, `decision`, `learned_lesson`, `refutation`, `meta`, `design` — no new tags promoted.
- 2026-05-16: 9 insights from the multi-round planning session (v3 → v4 → concept memo + math audit → lineage investigation → 4-professor symposium → v5) reused existing tags `nmn`, `film`, `design`, `decision`, `learned_lesson`, `meta`, `refutation`, `worktree`, `subagent` — no new tags promoted. 6 into `nmn_diagnosis`, 3 into `subagent_engineering`.
- 2026-05-16: 5 insights from the memory v2 build + graphify integration + bridge session (`20260516_1431_v2_three_role_architecture`, `20260516_1432_karpathy_graphify_adaptation_rationale`, `20260516_1433_graphifyy_integration_cheatsheet`, `20260516_1434_cross_phase_generator_backlinks_strip`, `20260516_1435_worktree_baseref_and_propagation`) reused existing tags `memory`, `design`, `decision`, `meta`, `worktree`, `learned_lesson` — no new tags promoted.
- 2026-05-13: 3 insights from the dreamer-srl v3 CP1 closure session (`20260513_2308_strong_strategy_validates_on_cp1`, `20260513_2309_merge_path_manifest_tripwire`, `20260513_2310_orphan_memory_branch_rewrite`) reused existing tags `dreamer`, `learned_lesson`, `decision`, `meta`, `design`, `memory` — no new tags promoted.
- 2026-05-13: 1 insight from the JAXVectorEnv v1+v2 spike closure session (`20260513_1417_jax_vmap_no_speedup_tiny_env`) reused existing tags `dreamer`, `learned_lesson`, `refutation`, `decision`, `meta` — no new tags promoted.
- 2026-05-13: 5 insights from the NMN R2 continual + 6-specialist analyzer verdict session (`20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin`, `20260513_0015_active_swapped_geq_matched_reframes_meta`, `20260513_0016_h1a_half_the_dip_predicate_schedule_asymmetric`, `20260513_0017_mod_h_logging_gap_blocks_cka_precheck`, `20260513_0018_train_py_orphan_render_workers_on_sigint`) reused existing tags `nmn`, `film`, `hypervigilance`, `decision`, `learned_lesson`, `design`, `meta`, `training_runner`, `refutation` — no new tags promoted.
- 2026-05-12: 3 insights from the dreamer-srl plan + PI pivot to sheeprl-direct session (`20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned`, `20260512_1755_pytorch_agents_pip_dep_layout`, `20260512_1756_pip_install_namespace_shadow_numpy_cap`) reused existing tags `dreamer`, `decision`, `learned_lesson`, `refutation`, `meta`, `design`, `training_runner` — no new tags promoted.
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
