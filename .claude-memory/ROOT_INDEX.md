# ROOT_INDEX.md — `.claude-memory/` topic folder registry

> Authoritative list of every topic folder under `memories/`.
>
> Read this file before classifying a new insight. Folder definitions here are the matching surface — if a new insight does not match any definition verbatim, the new-folder justification protocol applies (see CLAUDE.md, "Fragmentation safeguards").

**Last updated**: 2026-05-09
**Active folders**: 6
**Total insights**: 30
**Last audit**: (none)

---

## Active folders

| Folder | Definition (1 line) | Insights | Last update | Top tags |
|---|---|---|---|---|
| `memory_system_design` | Claude memory system's own design decisions | 4 | 2026-05-09 | [memory, design, decision, skill] |
| `subagent_engineering` | Subagent + worktree usage gotchas | 2 | 2026-05-09 | [meta, learned_lesson, worktree, subagent, decision] |
| `nmn_diagnosis` | NMN performance diagnosis findings | 6 | 2026-05-09 | [nmn, hypervigilance, film, refutation, learned_lesson] |
| `dreamer_diagnosis` | DreamerV3 failure investigation | 2 | 2026-05-08 | [dreamer, hypervigilance, decision, learned_lesson] |
| `cluster_ops` | Lab cluster ops and env mgmt | 12 | 2026-05-09 | [meta, training_runner, learned_lesson, decision] |
| `hypervigilance` | Hypervigilance experiments | 4 | 2026-05-09 | [hypervigilance, design, learned_lesson, decision, refutation, meta] |

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
