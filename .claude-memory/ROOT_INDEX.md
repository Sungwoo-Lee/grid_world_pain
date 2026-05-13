# ROOT_INDEX.md — `.claude-memory/` topic folder registry

> Authoritative list of every topic folder under `memories/`.
>
> Read this file before classifying a new insight. Folder definitions here are the matching surface — if a new insight does not match any definition verbatim, the new-folder justification protocol applies (see CLAUDE.md, "Fragmentation safeguards").

**Last updated**: 2026-05-13
**Active folders**: 6
**Total insights**: 53
**Last audit**: (none)

---

## Active folders

| Folder | Definition (1 line) | Insights | Last update | Top tags |
|---|---|---|---|---|
| `memory_system_design` | Claude memory system's own design decisions | 6 | 2026-05-09 | [memory, design, decision, skill, meta] |
| `subagent_engineering` | Subagent + worktree usage gotchas | 6 | 2026-05-10 | [meta, learned_lesson, worktree, subagent, decision] |
| `nmn_diagnosis` | NMN performance diagnosis findings | 9 | 2026-05-13 | [nmn, hypervigilance, film, learned_lesson, design, meta, training_runner, refutation] |
| `dreamer_diagnosis` | DreamerV3 failure investigation | 8 | 2026-05-13 | [dreamer, decision, learned_lesson, refutation, meta] |
| `cluster_ops` | Lab cluster ops and env mgmt | 17 | 2026-05-13 | [meta, training_runner, learned_lesson, decision, design] |
| `hypervigilance` | Hypervigilance experiments | 7 | 2026-05-13 | [hypervigilance, design, learned_lesson, decision, refutation, meta] |

---

## Folder naming rules

- English snake_case, no hyphens, no camelCase, ≤ ~3 words.
- 1-line definition ≤ ~30 characters describing what the folder is for.
- New folder requires a "Why a new folder" justification in the insight body.

---

## Audit policy

Surface a merge proposal to the user when:
- Active folder count ≥ 10, or
- Last audit older than 30 days.

---

## Related indexes

| Index | Path | Holds |
|---|---|---|
| This file | `.claude-memory/ROOT_INDEX.md` | Topic-folder metadata |
| Tag dictionary | `.claude-memory/memories/_global_tags.md` | All active tags |
| Topic indexes | `.claude-memory/memories/<folder>/_topic_index.md` | One-line summary per insight in that folder |

---

## Change history

- 2026-05-13: Captured 1 insight into existing `dreamer_diagnosis` from the JAXVectorEnv v1+v2 spike closure session: `20260513_1417_jax_vmap_no_speedup_tiny_env` — JAX-vmap parallel env over 5x5 NoPred gridworld delivers no speedup vs SyncVectorEnv (v1 CPU: 1.01x at N=4; v2 GPU: 0.47x at N=4); Python-JAX boundary dominates microsecond env-step compute; only DLPack zero-copy bridge could plausibly win (separate fresh plan). Env-install chain side-effects captured (cudnn 9.10.2.21, nvcc 12.9.86, jax downgrade to 0.9.0.1, torch 2.5 to 2.8 upgrade). No new tags promoted (all reused: dreamer, learned_lesson, refutation, decision, meta).
- 2026-05-13: Captured 5 insights from the NMN R2 continual + 6-specialist analyzer verdict session: 3 into existing `nmn_diagnosis` (`20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin` — first positive FiLM finding, H₁b + H₁c confirmed at ~25× seed-noise floor on the 5-stage continual schedule, +107/+132 steps on the two return-to-active stages; `20260513_0016_h1a_half_the_dip_predicate_schedule_asymmetric` — H₁a predicate is methodologically malformed for the active↔passive schedule shape, modulator wins via recovery speed not dip-depth; `20260513_0017_mod_h_logging_gap_blocks_cka_precheck` — raw modulator hidden vector never logged, Mahalanobis / CKA tests unevaluable, 3 metrics + 1 artifact hook requested), 1 into existing `hypervigilance` (`20260513_0015_active_swapped_geq_matched_reframes_meta` — specialist ceiling table inverts design-time intuition, active_swapped > active_matched, reframes upcoming meta head-to-head as a CKA-factorisation test), 1 into existing `cluster_ops` (`20260513_0018_train_py_orphan_render_workers_on_sigint` — train.py SIGINT leaves render_recordings.py multiprocessing pool workers orphaned, two-call terminate_command.py workaround). No new tags promoted (all reused: nmn, film, hypervigilance, decision, learned_lesson, design, meta, training_runner, refutation).
- 2026-05-12: Captured 3 insights from the dreamer-srl plan + PI pivot to sheeprl-direct session: 1 into existing `dreamer_diagnosis` (`20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned` — after 3-reviewer ✅ PASS on a 1033-line JAX re-implementation plan, user pivoted via PI call to sheeprl PyTorch direct; meta-lesson: static review does not predict integration-layer execution success, the cascade-debugging history was a stronger prior), 2 into existing `cluster_ops` (`20260512_1755_pytorch_agents_pip_dep_layout` — third-party RL frameworks integrate as pip-installed git-pinned deps + extensions in sibling in-repo package, not `tmp/` clones; `20260512_1756_pip_install_namespace_shadow_numpy_cap` — two pip-install gotchas during node-114 env rebuild: outer/inner namespace shadow `editable_mode=compat` and sheeprl@33b6366 spurious numpy<2.0 cap `--no-deps`). No new tags promoted (all reused: dreamer, decision, learned_lesson, refutation, meta, design, training_runner).
- 2026-05-12: Captured 1 insight into existing `hypervigilance` from the behavior-measure toolkit v1 application to R2.5 checkpoints: `20260512_1428_sameprop_class_discriminating_defence_event_level` — refines (not supersedes) the prior `_no_class_avoidance` verdict. Under sameProp the agent IS class-discriminating at the event level (bush-dive rate +37 pp predator vs rabbit; eat-under-threat 0.75× near predator vs 1.19× near rabbit; per-tag rabbit_TL/rabbit_BR within noise), even though mean distances dissolve this signal. Two-level interpretation: spatially class-blind, behaviourally class-discriminating. No new tags promoted (all reused: hypervigilance, learned_lesson, decision).
- 2026-05-11: Captured 2 insights from the Z2 verdict + diagnostic-bug-fix session: 1 into existing `dreamer_diagnosis` (`20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt` — paper-canonical bins H2 partial fix; cumulative cascade −54%; new long-horizon-compounding residual selects GRU reset gate as next candidate), 1 into existing `cluster_ops` (`20260511_1535_encode_decode_flag_mismatch_silent_class_bug` — silent failure-mode class when knob-gated encode/decode change ships without updating auxiliary tooling; cross-check against training-time logger as trip-wire). No new tags promoted (all reused: dreamer, learned_lesson, decision, refutation, meta).
- 2026-05-10: Captured 3 insights from the dreamer sheeprl-comparison + zero-init cascade session: 1 into existing `dreamer_diagnosis` (`20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry` — Z1 H2 partial fix; pos/neg reward-MAE asymmetry directs next candidate), 2 into existing `subagent_engineering` (`20260510_2240_reference_impl_compare_only_act_intersections` — document-by-default rule for reference-impl comparators; `20260510_2241_residual_error_pattern_directs_next_fix` — residual-error pattern dictates next fix in iterative cascades). No new tags promoted (all reused: dreamer, learned_lesson, decision, refutation, subagent, meta).
- 2026-05-10: Captured 1 insight into existing `hypervigilance` from the Round 2.5 launch + analysis session: `20260510_2237_sameprop_round25_no_class_avoidance` (both pre-registered hypotheses refuted at 10M ep — Cell A1 H₀ confirmed at 75× margin via per-tag Δ_TL=+0.004; Cell C aggregated Δ=−0.53 sign-flipped, lands in Inverted band; original sameProp survey effect decomposes into two confounds with no genuine class-conditional avoidance under matched smells; provisional pending Round 2.6 seed 44 for Cell C). No new tags promoted (all reused: hypervigilance, refutation, learned_lesson, decision).
- 2026-05-09: Captured 3 insights from the NMN meta/continual pivot session: 2 into existing `memory_system_design` (`20260509_1619_summarize_study_skill_design_and_ship` — third documentation layer + skill that automates it; `20260509_1620_documentation_framing_policy` — project-wide plain-language entry-point rule promoted from one skill to all doc-producing surfaces), 1 into existing `subagent_engineering` (`20260509_1621_multi_agent_research_chain_v2_pattern` — append-only sibling versioning under mid-chain user expansion). No new tags promoted (all reused: memory, design, decision, skill, meta, subagent, learned_lesson).
- 2026-05-09: Captured 4 insights from the dreamer conventional-fixes battery session: 2 into existing `dreamer_diagnosis` (`20260509_1534_wm_reward_head_localized_failure_a1` — offline WM diagnostic localizes failure to the reward head; `20260509_1535_conventional_fixes_battery_verdict_predator_refute` — top-2 conventional causes refuted on predator), 1 into existing `cluster_ops` (`20260509_1536_train_py_checkpoint_restore_nnx_skew` — latent train.py orbax-vs-NNX bug), 1 into existing `subagent_engineering` (`20260509_1537_professor_analysis_resets_exotic_investigation` — routing pattern). No new tags promoted (all reused: dreamer, learned_lesson, decision, refutation, hypervigilance, meta, training_runner, subagent).
- 2026-05-09: Captured 3 insights from the hypervigilance Round 2 partial-verdict + per-tag metrics ship session: 2 into existing `hypervigilance` (`20260509_1532_sameprop_round2_truncated_verdict`, `20260509_1533_tag_based_distance_supersedes_quadrant`), 1 into existing `subagent_engineering` (`20260509_1534_synthetic_smoke_masks_dict_assembly_bugs`). No new tags promoted (all reused: hypervigilance, refutation, learned_lesson, design, decision, meta, subagent).
- 2026-05-09: Captured 2 insights into existing `nmn_diagnosis` from the temp_clip[0.5,10.0] re-run analyzer chain: `20260509_1409_nmn_tempceil10_verdict_h1b_confirmed_p4` (verdict — H1b confirmed on P4, partial-binder reframing) and `20260509_1410_nmn_temp_head_natural_target_3_to_5` (head natural target [3.0, 5.0); canonical FiLM config should adopt [0.5, 5.0]). No new tags (all reused: nmn, hypervigilance, film, refutation, learned_lesson).
- 2026-05-09: Captured 4 insights from the cluster-ops scripting consolidation session: 3 into existing `cluster_ops` (`20260509_0309_cluster_py_consolidation`, `20260509_0310_bash_ic_alias_over_ssh`, `20260509_0312_node_num_hostname_in_bashrc`) and 1 into existing `memory_system_design` (`20260509_0311_diary_auto_session_backfill`). No new tags (all reused: meta, decision, learned_lesson, training_runner, design, memory).
- 2026-05-08: Captured 2 insights into existing `nmn_diagnosis` from the heterogeneity-sweep analyzer chain: `20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse` (verdict — H1a/b/c refuted, FiLM consistently worse than Unmod) and `20260508_2004_profile_dependent_temp_saturation_mc_film` (actionable mechanism — temp saturation at R ≥ 5). Promoted `film` from candidate to active and added new active tag `refutation`.
- 2026-05-08: Captured 1 insight into existing `cluster_ops`: `20260508_1826_statusline_jq_ifs_pct` (Claude Code statusline script gotchas: no `jq` on docker-102, bash `read` IFS-whitespace collapse, and the pre-calculated `context_window.used_percentage` field). No new tags (all reused: meta, learned_lesson, decision).
- 2026-05-08: Captured 1 insight into existing `cluster_ops`: `20260508_1717_ssh_config_match_user_scoping` (scoping the lab SSH config block to vncuser via `Match user`). No new tags (all reused).
- 2026-05-08: Captured 3 insights into existing `cluster_ops` from the evaaa→episode container rebuild session: `20260508_1637_nas_automount_fstab_actimeo`, `20260508_1638_container_slimdown_recipe`, `20260508_1639_ssh_credentials_in_shared_image`. No new tags (all reused: meta, decision, learned_lesson, training_runner).
- 2026-05-08: Captured 3 insights from the hypervigilance/sameProp Round 1 + Round 2 design session: `20260508_1444_sameprop_round1_finding_and_confound` and `20260508_1445_sameprop_discriminating_channels` into the new `hypervigilance` folder, `20260508_1446_cifs_race_node112_recurrence` into existing `cluster_ops`. No new tags promoted (all reused: hypervigilance, design, learned_lesson, decision, meta, training_runner).
- 2026-05-08: Captured 4 insights: 2 into the new `dreamer_diagnosis` folder (`20260508_1431_diagnostic_battery_refutes_four_fixes`, `20260508_1432_probe_refutes_imagined_death_absence`), 2 into the existing `cluster_ops` folder (`20260508_1433_cifs_bypass_for_run_command`, `20260508_1434_terminate_command_key_auth_refactor`). Promoted candidate tag `dreamer` to active. (Reconciled with a parallel session that captured 3 insights at HHMM 1426–1428; my IDs bumped to 1431–1434 to avoid collision.)
- 2026-05-08: Captured 3 insights: `20260508_1426_v8_noise_bug_refuted` and `20260508_1427_nmn_heterogeneity_sweep_design` into the new `nmn_diagnosis` folder, `20260508_1428_node_env_recovery_recipe` into the new `cluster_ops` folder. Promoted candidate tags `nmn`, `noise`, `hypervigilance`, `training_runner` to active.
- 2026-05-08: Captured 1 insight: `20260508_0447_recall_skill_design_and_ship` into `memory_system_design` (no new tags; all reused).
- 2026-05-08: Captured 2 insight(s): `20260508_0429_memorize_skill_design_and_ship` into `memory_system_design`, `20260508_0430_worktree_isolation_path_safety` into the new `subagent_engineering` folder.
- 2026-05-08: Created. Added `memory_system_design` folder (genesis insight: `20260508_0315_claude_memory_system_genesis`).
