---
title: "Experiment history — May–July 2026 (chronological campaign review)"
topic: comparison
status: active
created: 2026-07-29
last_updated: 2026-07-29
aliases: [experiment-history-may-jul-2026, campaign-timeline-2026q3]
---

# Experiment history — May through July 2026

## What this document is (read this first)

This is a **portfolio-level timeline**, not a single-run analysis. It reviews every experiment
campaign the project ran between **1 May and 29 July 2026**, in the order it happened, and links
each campaign to the design/analysis doc that owns it and to the reusable lessons ("wiki insights")
it produced. It is sourced from the daily diary (58 files) and the project's in-repo LLM Wiki. A
collaborator should be able to open it cold and understand what was tried, when, and what we learned.

Performance is always reported in **survival steps** (how many timesteps the agent stays alive),
never cumulative reward — that is the project's standing convention.

Across these three months, six research threads ran in parallel:

1. **Hypervigilance / danger discrimination** — does an agent that has learned to survive a predator
   actually *recognise danger in advance*, or does it only *react to pain after being hit*? Tested by
   giving a predator and a harmless rabbit the *same smell* ("sameProp") and by injury/hunger-gated
   perception experiments.
2. **The neuromodulator (NMN / FiLM) architecture question** — does a small "context" network that
   rescales the agent's hidden units (a FiLM modulator, a stand-in for brain neuromodulators like
   noradrenaline) help it survive better than a plain baseline?
3. **The in-house DreamerV3 (`dreamer_srl`) vs recurrent-PPO survival gap** — our model-based agent
   kept trailing the simpler model-free baseline; a months-long diagnosis to find out why.
4. **The interoceptive behavior-measure & probe platform** — a reusable toolkit that measures
   *what the agent does* (bush-diving, eat-under-threat, foraging timing) rather than just how long
   it survives, plus the eval-during-training pipeline that logs those measures live.
5. **Environment & curriculum engineering** — the `basic00`–`basic07` difficulty ladder, per-episode
   randomisation of entity counts and predator behaviour, and features like the predator jump/pounce
   and the bush "perfect refuge".
6. **Config-system correctness & cluster/agent tooling** — the v3.0 config overhaul, the silent
   `decay_power` drift that quietly changed every experiment's sensory world, the scripts
   reorganisation, GPU-status tooling, and the two big "fresh-eyes" codebase diagnoses.

Threads 1–4 are the science; threads 5–6 are the platform that repeatedly turned out to *be* the
finding (several "the agent can't discriminate" or "Dreamer fails" conclusions were later traced to
a config, a metric, or a logging artifact, not the agent).

---

## Timeline

### May 2026 — NMN verdicts, the DreamerV3 diagnosis, and the sheeprl pivot

| Date | Campaign | What was tested | Key finding | Docs | Insights | Tags |
|---|---|---|---|---|---|---|
| 05-07→05-08 | **NMN noise-heterogeneity sweep** | 10 cells (5 noise profiles × plain-LayerNorm vs FiLM modulator) — does the modulator help when sensory-noise levels differ across channels? | **Refuted.** FiLM is consistently 5–13 survival-steps *worse* than the plain baseline at every profile. | [NMN_NOISE_HETEROGENEITY_SWEEP](../active/hypervigilance/NMN_NOISE_HETEROGENEITY_SWEEP.md) | [[20260508_2003_nmn_heterogeneity_sweep_verdict_film_worse]], [[20260508_2004_profile_dependent_temp_saturation_mc_film]] | `rppo_nmn_het_p{1..5}_{unmod,film_g1}_c*` |
| 05-08→05-09 | **NMN temp-clip ceiling rerun** | Raise the modulator's temperature ceiling 3.0→10.0 — was the earlier null a hyperparameter cap? | **Partial.** Modulator closes most of the gap on the high-heterogeneity profile but still trails the baseline by ~2.5 steps. Head's natural target sits in [3.0, 5.0). | [NMN_TEMP_CLIP_CEILING_RERUN](../active/hypervigilance/NMN_TEMP_CLIP_CEILING_RERUN.md) | [[20260509_1409_nmn_tempceil10_verdict_h1b_confirmed_p4]], [[20260509_1410_nmn_temp_head_natural_target_3_to_5]] | `rppo_nmn_tempceil10_p{3,4,5}_film_g1_s*` |
| 05-07→05-09 | **DreamerV3 failure diagnosis** | 4-experiment refutation chain (diagnostic battery + imagined-death probe + conventional-fixes battery + offline world-model test) on why our DreamerV3 collapses. | Failure **localised to the world-model reward head**; four candidate fixes and the "imagines no death" hypothesis all refuted. | [dreamer_v3_diagnosis (summary)](20260509_1555_dreamer_v3_diagnosis.md), [DREAMER_CONVENTIONAL_FIXES_BATTERY](../active/dreamer_diagnosis/DREAMER_CONVENTIONAL_FIXES_BATTERY.md) | [[20260508_1431_diagnostic_battery_refutes_four_fixes]], [[20260508_1432_probe_refutes_imagined_death_absence]], [[20260509_1534_wm_reward_head_localized_failure_a1]], [[20260509_1535_conventional_fixes_battery_verdict_predator_refute]] | `dreamer_conv_{NoPred,Pred}_rr06_*` |
| 05-07→05-10 | **SameProp rabbit-avoidance R1–R2.5** | Give predator and rabbit identical smell; does the agent still keep further from the predator? Round-1 relog (2 seeds) → Round 2 → Round 2.5 (10M ep). | **Both hypotheses refuted** at the mean-distance level — no genuine class-conditional spatial avoidance under matched smells; Cell A1 corner-camps. | [sameprop study (re-summary)](20260510_2253_sameprop_rabbit_avoidance_study.md), [sameprop_round25_design](../active/hypervigilance/sameprop_round25_design.md) | [[20260508_1444_sameprop_round1_finding_and_confound]], [[20260509_1532_sameprop_round2_truncated_verdict]], [[20260510_2237_sameprop_round25_no_class_avoidance]] | `hypervigilance-round2*`, `hypervigilance-round25-*` |
| 05-10→05-11 | **DreamerV3 fix cascade (Z1/Z2)** | sheeprl reference-impl comparison surfaced deviations; zero-init reward+critic (Z1) then paper-canonical two-hot bins (Z2). | Both fired the reward-MAE test **partially**; cumulative cascade cut reward error −54% (0.39→0.18); GRU reset-gate queued next. | [dreamer_v3_fix_cascade (summary)](20260511_1559_dreamer_v3_fix_cascade.md) | [[20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry]], [[20260511_1534_z2_paper_bins_h2_partial_cascade_closure_attempt]] | `dreamer_zinit_*`, `dreamer_twohotrng_*` |
| 05-09→05-13 | **NMN vitality probes** | 6 single-world "specialist" ceilings + a 2-run continual sister pair (5-stage active↔passive schedule). Caught a 1000× episode-budget bug and re-launched (R2). | **First clearly-positive FiLM finding:** under the continual schedule the modulator beats the baseline by +107/+132 steps on the two return-to-active stages (~25× the seed-noise floor). | [nmn_comparison (re-summary)](20260513_0321_nmn_comparison_study.md), [NMN_CONTINUAL_DOUBLE_RETURN_PROBE](../active/hypervigilance/NMN_CONTINUAL_DOUBLE_RETURN_PROBE.md), [NMN_META_2x3_MIXTURE_PROBE](../active/hypervigilance/NMN_META_2x3_MIXTURE_PROBE.md) | [[20260513_0014_nmn_r2_continual_h1b_h1c_confirmed_high_margin]], [[20260513_0016_h1a_half_the_dip_predicate_schedule_asymmetric]], [[20260513_0015_active_swapped_geq_matched_reframes_meta]] | `rppo_nmn_cont_dr_{mod,unmod}_s0_r2`, `rppo_nmn_meta_spec_{active,passive}_{matched,distinct,swapped}_s0` |
| 05-11→05-14 | **SameProp behavior-measure toolkit v1 + R2.6** | Apply the new event-level measures (M1 interrupted-feeding, M2 bush-dive, M5 eat-under-threat, M7 motif clustering) to R2.5 checkpoints; re-launch two seeds. | **Two-level verdict:** spatially class-blind, but *event-level* class-discriminating — bush-dive rate +37 pp near predator vs rabbit; eat suppressed near predator. R2.6 first crashed, then seed-locked. | [sameprop (closing re-summary)](20260521_1546_sameprop_rabbit_avoidance_study.md), [behavior_measure_toolkit_v1_design](../active/behavior_measures/behavior_measure_toolkit_v1_design.md) | [[20260512_1428_sameprop_class_discriminating_defence_event_level]], [[20260518_1735_sameprop_a1_seed45_corner_camping_refuted]] | `hypervigilance-round26-{A1,C}-seed4{4,5}` |
| 05-12 | **PI pivot: JAX Dreamer rebuild → sheeprl-direct bridge** | A 1033-line JAX re-implementation of sheeprl's DreamerV3 passed 3-reviewer review, but a 5× survival evidence gap (sheeprl ~500 vs in-house JAX ~106) prompted a decision call. | **Abandoned the JAX rebuild**; adopted sheeprl PyTorch directly via an in-repo `pytorch_agents/` bridge package; 50k-step parity landed at survival ~399. | [sheeprl_bridge/METRIC_PARITY_PLAN + SPS_COMPARISON](../active/sheeprl_bridge/SPS_COMPARISON_JAX_VS_SHEEPRL.md) | [[20260512_1754_sheeprl_direct_pivot_jax_dreamer_abandoned]], [[20260512_1755_pytorch_agents_pip_dep_layout]] | `sheeprl_*`, `jax_sheeprl_matched_n{4,16}_s0` |
| 05-13→05-21 | **dreamer-srl v2 + v3 rebuild, hyperparameter search** | After the pivot, a *second* pivot back to a JAX rebuild done under a strict deviation-prevention discipline; v2 reached parity, v3 rebuilt checkpoint-by-checkpoint; 10×10 hyperparameter search. | **v2 PASSES-and-outperforms** sheeprl on food-only (501 vs ~500); v1's root bug was REINFORCE re-sampling at loss-time. **Production recipe:** XS size / 16 envs / 4M steps → survival ~184 (+74% over baseline). | [dreamer_srl_v2/PARITY_LAUNCH_V2](../active/dreamer_srl_v2/PARITY_LAUNCH_V2.md), [HYPERPARAM_SEARCH_10X10](../active/dreamer_srl_v2/HYPERPARAM_SEARCH_10X10.md) | [[20260518_1511_dreamer_srl_v2_parity_pass_outperform]], [[20260518_1512_reinforce_resampling_bug_imag_action_threading]], [[20260518_1513_production_recipe_xs_16_4m_hypervigilance]], [[20260513_2308_strong_strategy_validates_on_cp1]] | `dreamer_srl_v2_postfix_*`, dreamer-srl v3 CP ports |
| 05-13 | **JAX vectorised-env spike** | Does a JAX-vmap parallel env beat sheeprl's SyncVectorEnv on our tiny 5×5 grid? | **No speedup** (CPU 1.01×, GPU 0.47× at N=4) — the Python↔JAX boundary dominates microsecond env steps. Spike closed. | (SPS memos) | [[20260513_1417_jax_vmap_no_speedup_tiny_env]] | `jaxvec_smoke_n4*` |

Late May was also the **v2.0 env-entity refactor** (predator + neutral animals unified into one entity
class with a static class tag and per-episode distributional sampling of 5 behavioural fields), shipped
across 86 configs with byte-parity — the substrate the June per-episode-variance work built on
([[20260529_1823_unified_animal_entity_v2_0_arch]]).

### June 2026 — the config overhaul, environment features, and the predator/rabbit danger-discrimination verdict

| Date | Campaign | What was tested | Key finding | Docs | Insights | Tags |
|---|---|---|---|---|---|---|
| 06-09→06-12 | **Predator vs rabbit danger discrimination (control battery)** | Four controls (chasing rabbit, matched-aggression, predator-only, single-pred-vs-rabbit lethal contact): does the agent recognise danger or just react to pain? | **Pain-consequence-driven, not anticipatory.** With rabbits removed the agent does *not* pre-empt the lone predator; avoidance is post-contact. The visual-count "elimination" pre-emption story was refuted by the control. | [predator_rabbit_discrimination (summary)](20260612_1625_predator_rabbit_discrimination.md), [single_pred_rabbit_disengage](../active/hypervigilance/single_pred_rabbit_disengage.md) | [[20260609_1720_chasing_rabbit_avoidance_damage_driven]], [[20260609_1747_avoidance_is_post_contact_not_preemptive]], [[20260609_1721_aggregate_stats_hide_conditional_behavior]] | `recurrent_ppo_08-singlePredRabbit_disengage_s42` |
| 06-09→06-11 | **dreamer_srl 3-stage size+entity curriculum** | Grow model size + entity count in 3 stages vs from-scratch on the 10×10 task; budget-ablation cells T1–T4. | Curriculum reaches the **same plateau (~210–221 survival) at ~45% fewer episodes** (single seed); T3 sets the budget floor at 71.5k, T4 under-trained. | [DREAMER_SRL_3STAGE_SIZE_CURRICULUM](../active/continual_learning/DREAMER_SRL_3STAGE_SIZE_CURRICULUM.md) | — | `dreamer_srl_curric3_{T1..T4,size}_s42` |
| 06-15→06-16 | **Discrimination-metric testbed search** | Autonomous overnight search for an isolated world where predator-vs-rabbit "discrimination" is a trustworthy scalar metric. | The bush-dive "discrimination" is a **spatial-encounter artifact** — a known class-blind agent reproduces the +0.31 gap. Reframe: the experimental env is a *behavior-measurement platform*, not a discrimination oracle. | [testbed_solo_validation_results](../active/hypervigilance/testbed_solo_validation_results.md), [predator_rabbit_testbeds](../active/hypervigilance/predator_rabbit_testbeds.md) | [[20260615_1612_testbed_isolation_makes_means_honest]], [[20260616_0142_discrimination_is_spatial_encounter_artifact]], [[20260616_1514_experimental_env_as_behavior_platform]] | (frozen-checkpoint evals, no training) |
| 06-17→06-19 | **v3.0 config-system overhaul** | Make `default.yaml` the canonical base with opt-in `extends:` deep-merge layering; expose init-state ranges + config-driven visual properties. | Shipped; experiment configs become sparse opt-in files. Wired a `CONFIG_GUIDE.md` maintenance contract into 5 agents. (Later found: the *trainer's* loader did **not** resolve `extends:` — a latent bug, see July.) | [basic_curriculum](../active/basic_curriculum/basic_curriculum.md) | [[20260619_0111_config_v3_extends_layering_default_base]], [[20260619_0113_configurable_initial_state_ranges]], [[20260619_0114_config_guide_maintenance_contract]] | — |
| 06-19→06-24 | **basic curriculum ladder (rPPO, from-scratch)** | Convergence + collapse behaviour of levels basic00 (static forage) → basic04 (far-sighted predator), plus the 5-stage continual curriculum. | Easy levels converge ~0.2M then **over-train-collapse** to a degenerate entropy≈0 policy (~3.8M/9M); stable levels plateau ~261–428. The **continual curriculum underperformed from-scratch (negative transfer)** — diagnosed as loss-of-plasticity, not under-training. | [basic_curriculum_convergence](../active/basic_curriculum/basic_curriculum_convergence.md), [basic_curriculum_continual_result](../active/basic_curriculum/basic_curriculum_continual_result.md) | [[20260622_1748_basic_curriculum_overtraining_collapse_and_intervals]], [[20260624_0516_curriculum_underperformed_baseline_negative_transfer]], [[20260624_0517_continual_failure_is_plasticity_loss_not_budget]] | `rppo_basic0{0..4}_*_n11{3,4}`, `rppo_basic_curriculum_n106` |
| 06-19→06-22 | **Hunger-gated olfactory-ambiguity program (hg01–hg10)** | Split "predator-vs-rabbit smell distinguishability" into two orthogonal knobs (mean separation × per-episode noise); 10-run sweep; scarcity + linear-decay variants. | Pre-contact discrimination stays **weak (max +0.21 vs the 0.5-cell bar)** with 47–69% death — a clean null; "lethality masks gating." Olfactory `decay_power` (distance falloff) identified as the distal-cue-strength knob. | [20260619_hunger_gated_step1_discrimination_onset](../active/hypervigilance/20260619_hunger_gated_step1_discrimination_onset.md), [20260620_hunger_gated_step1_linear_olfactory_decay](../active/hypervigilance/20260620_hunger_gated_step1_linear_olfactory_decay.md) | [[20260622_1744_olfactory_uncertainty_two_knob_design]], [[20260622_1745_discrimination_weak_lethality_masks_gating]], [[20260622_1746_olfactory_decay_power_distal_cue_strength]], [[20260622_1747_hypervigilance_asymmetric_stakes_scarcity]] | `rppo_hg0{1..10}_*`, `rppo_hvs_{scarce,abundant}_s42` |
| 06-22→06-30 | **Per-episode environment-variance + entity features** | Randomise entity counts per episode (JAX static-shape masking), predator behavioural params, predator jump/pounce, bush "perfect refuge" (blocks animals + hides agent). | Shipped with byte-parity discipline. rPPO is recompile-immune; **Dreamer hit a recompile storm** (arrays sized to the variable done-count) — fixed with fixed-width masked reset. Caught a "ghost predator" render bug on inactive slots. | [DREAMER_SRL_3STAGE_SIZE_CURRICULUM](../active/continual_learning/DREAMER_SRL_3STAGE_SIZE_CURRICULUM.md) (crash log) | [[20260623_0143_per_episode_count_variance_masking]], [[20260623_1616_rppo_reset_recompile_immune]], [[20260622_1746_dreamer_srl_recompile_storm_done_count]], [[20260624_0516_bush_blocks_animals_movement_toggle]], [[20260629_1723_ghost_predator_inactive_slots_render]], [[20260703_0343_predator_jump_pounce_mechanism]] | `dsrl_basic0{0..5}_*` |
| 06-23→06-24 | **NMN FiLM vs plain — long level-4 run** | Does the FiLM modulator beat a plain baseline on a long (50M ep) basic04 run? | **Modulator did NOT beat plain** (276 vs 290 survival at matched budget); temperature railed at the ceiling and the run crashed late (~34M ep). | [nmn_film_vs_plain_longL4](../active/basic_curriculum/nmn_film_vs_plain_longL4.md) | — | `rppo_basic_curriculum_longL4`, `rppo_nmn_film_curric_longL4_n114` |
| 06-27→07-02 | **NMN FiLM grouping screen (g1–g128)** | Sweep the FiLM "grouping" factor g ∈ {1,2,4,…,128} on the long level-4 / basic05-all task — does group size matter? | Screen run; feeds the plain-vs-NMN comparison later re-run under corrected `decay_power`. | [NMN_FILM_GROUPING_SCREEN](../active/basic_curriculum/NMN_FILM_GROUPING_SCREEN.md) | — | `rppo_nmn_film_g{1..128}_{screen,curric_longL4,basic05all}_s42` |
| 06-29→06-30 | **Interoceptive behavior-measure study (probe platform)** | Discover *measures* of foraging + avoidance as a function of nutrition + injury, trajectory-first, on a frozen in-distribution checkpoint. New `behavior_measures` topic + reusable heatmap tooling. | Hunger sets foraging **timing not path**; the flee-to-cover reflex needs **both** approach motion **and** recognisable olfaction; cover-seeking (bush-diving) is **late-emerging** (early models run/kite instead). Injury is not pre-emptive. | [interoceptive_behavior_measure_study](../active/behavior_measures/interoceptive_behavior_measure_study.md) | [[20260630_1715_behavior_measure_study_method_and_tooling]], [[20260630_1716_foraging_hunger_timing_fixed_opening]], [[20260630_1717_avoidance_reflex_needs_motion_and_olfaction]], [[20260630_1718_cover_use_late_emerging_run_vs_hide]] | (frozen ckpt 8900007 probes) |
| 06-30 | **dreamer_srl basic training-speed fix** | Why did all 5 basic-curriculum Dreamer runs hang for hours / OOM? | The gradient `lax.scan` had **no enclosing `@jax.jit`**, so the 76k-HLO train_step recompiled every iteration — the multi-hour hang. Fixed with a persistent jit. | [basic_curriculum](../active/basic_curriculum/basic_curriculum.md) | [[20260630_1720_dreamer_srl_train_step_jit_compile_once]], [[20260630_1721_per_episode_variance_dreamer_recompile_safe]] | `dsrl_basic0{0..5}_*` |

### July 2026 — the `extends:` bug, behavior-metric datasets, the Fable diagnoses, decay_power, and the Dreamer investigation

| Date | Campaign | What was tested | Key finding | Docs | Insights | Tags |
|---|---|---|---|---|---|---|
| 07-03 | **train.py `extends:` drop bug** | Why did the noise/random-init basic07 runs behave like they had no noise? | **SEVERE latent bug:** the trainer loads configs with a plain-YAML loader that never resolves `extends:`, so inherited layers (noise, random-init) silently dropped to default. Latent since v3.0. Promoted the "verify actual saved state" methodology rule. | [basic_curriculum](../active/basic_curriculum/basic_curriculum.md) | [[20260703_1507_train_py_ignores_extends_drops_layers]], [[20260703_1508_eval_video_drops_true_obs_no_noise_contrast]], [[20260703_0344_attack_range_float_threshold_gotcha]] | `rppo_basic0{5,6,7}_*`, `rppo_basic_ladder6_*` |
| 07-04→07-06 | **Avoidance result-tables + videos (basic05/06/07 + NMN wave)** | Generate reader-facing avoidance result tables + eval videos across the noise/random-init training wave. | Established the noise-matched frozen-probe protocol (probe must carry the agent's exact training noise block); near-deterministic probes **inflate p-values AND Cohen's d** — judge by effect size + magnitude + overlap. | [model_comparison_statistics_tutorial](../active/behavior_measures/model_comparison_statistics_tutorial.md) | [[20260704_2012_noise_matched_frozen_probe]], [[20260704_2013_probe_rerun_stale_checkpoint_contamination]], [[20260704_2014_deterministic_probe_significance_inflates]] | ladder6 checkpoints |
| 07-05→07-10 | **Fable-5 re-diagnosis + sheeprl parity fix batch** | A fresh-model ("Fable 5") from-scratch re-audit of both Dreamers + sheeprl; fix all High findings (H1–H10). | Math cores **bit-identical**, but **11 undeclared training-impact deviations live in the glue**, not the math; the empirical re-audit found a *new bug class* the read-only pass missed. Confirmed DreamerV3-NNX is the **abandoned** stack; `dreamer_srl` is live but has no NMN hooks. | [sheeprl_bridge/SPS_COMPARISON_JAX_VS_SHEEPRL](../active/sheeprl_bridge/SPS_COMPARISON_JAX_VS_SHEEPRL.md) | [[20260723_1909_nnx_abandoned_archived_stack_confusion]], [[20260723_1910_sheeprl_parity_drift_lives_in_glue]], [[20260723_1912_fresh_empirical_reaudit_finds_new_bug_class]], [[20260723_1911_rppo_nmn_config_boundary_traps]] | (audit; sheeprl parity runs) |
| 07-08→07-10 | **128-env rPPO relaunch (basic00–05, 100M)** | Re-run the basic ladder at 128 envs after finding a CLI `--num-envs 16` was silently shadowing the config's 128. | All prior ladder6 runs had trained at 16 envs, not 128; 128 is ~8× faster. Established the "config-owns-values" convention + a resume path needing matching env count. | [basic_curriculum](../active/basic_curriculum/basic_curriculum.md) | [[20260710_1632_num_envs_cli_override_config_owns_values]], [[20260710_1633_rppo_resume_needs_matching_num_envs]], [[20260710_1635_bush_hiding_metastable_dwell_measure]] | `rppo_basic0{0..5}_*_128env_*`, `rppo_basic03_randinit_cont100M_n108` |
| 07-13→07-21 | **Behavior-metric datasets + basic03/04 size sweeps** | Comprehensive dataset: 6 levels × 12 conditions × 11 measures; model-size sweep (128/XS/S/M/L/XL) on basic03 and basic04, rPPO + Dreamer; batched-eval enabler. | Batched Dreamer eval 6.4×; near-zero Dreamer bush-dwell is a **genuine "not learned yet"**, verified by trajectory inspection, not a bug. Eval sweep is CPU-bound, not NAS-bound. | [dreamer_vs_rppo_survival/dreamer_srl_vs_rppo_survival_speed](../active/dreamer_vs_rppo_survival/dreamer_srl_vs_rppo_survival_speed.md) | [[20260721_0422_batched_dreamer_eval_rng_and_unify]], [[20260721_0424_dreamer_low_dwell_real_trajectory_verified]], [[20260721_0421_eval_sweep_cpu_bound_not_nas]] | `rppo_b0{3,4}_sz{128,XS,S,M,L,XL}_*`, `basic0{3,4}_size_sweep` |
| 07-21→07-22 | **basic04 variant + GAE-vs-MC sweeps** | 8 basic04 variants (slow-move, short-jump, low-damage, all-combined, attack-success-rate 0.3/0.7, combined 0.3/0.7) + a GAE-vs-Monte-Carlo return-mode arm. | Feeds the difficulty-axis and return-mode comparisons; the GAE/MC arm later discarded and re-run under corrected `decay_power`. | (variant configs) | — | `rppo_b04v0{1..8}_*_128env_*`, `rppo_b0{3,4}_gae_128env_n106` |
| 07-23→07-24 | **Full-codebase Fable diagnosis (13-unit) + 3 fix tracks** | 13-unit sub-agent diagnosis over env/rPPO/dreamer_srl/eval/configs; land config-layer, eval-logging, and Dreamer-eval-telemetry fixes. | No P0, 10 live P1s, ~65 P2s. Config layer now **fails loudly** instead of silently disabling noise/swapping worlds; **Dreamer eval videos reach WandB for the first time**. Also: predator-mixture (U{0,1,2}) makes the raw survival curve a **bimodal mixture** — a size verdict is unreliable until split. | [dreamer_srl_vs_rppo_survival_speed](../active/dreamer_vs_rppo_survival/dreamer_srl_vs_rppo_survival_speed.md), [dreamer_srl_v2/dreamer_vs_rppo_gap_hypotheses](../active/dreamer_srl_v2/dreamer_vs_rppo_gap_hypotheses.md) | [[20260723_1913_predator_mixture_inflates_survival_metric]], [[20260723_1916_train_ratio_replay_ratio_conversion]], [[20260723_1915_two_level_logging_and_config_layering]], [[20260723_1914_dreamer_noise_is_logging_granularity_artifact]] | `dsrl_b04_M_128env_*`, `rppo_nmn_g32_b0{3,4}_{gae,mc}_*` |
| 07-24 | **Eval-during-training shipped** | Async on-node behavior-probe subprocess at every Nth checkpoint → live `Probe/*` series in WandB (bush-dwell / survival / spatial-spread × predator/none/rabbit). | Verified live on the node-110 basic04 100M run; the trainer is the sole WandB writer, results keyed by checkpoint episode. | [interoceptive_behavior_measure_study](../active/behavior_measures/interoceptive_behavior_measure_study.md) | [[20260710_1634_behavior_probe_eval_speed_parallel_batched_fdsafe]] | `rppo_b04_experimenteval_128env_100M_n110` |
| 07-26 | **decay_power silent-drift caught + config governance** | Audit of saved run configs after odd sensory behaviour. | The olfactory distance-falloff exponent `sensory.decay_power` had **silently drifted 1.0→2.0** inside an unrelated Feb-2026 "scale to 10×10" commit — every recent run trained in the wrong sensory world. Reverted to 1.0; created `CONFIG_CRITICAL_SETTINGS.md` registry + change-log guardrail. Relaunched a 2×2 Dreamer size grid + an 8-run rPPO ladder under 1.0. | [CONFIG_CRITICAL_SETTINGS](../../environment/CONFIG_CRITICAL_SETTINGS.md) | [[20260726_0415_decay_power_silent_drift_reverted_1p0]], [[20260726_0416_critical_settings_registry_changelog_guardrail]], [[20260726_0419_experiment_eval_config_layer_selector]], [[20260726_0418_dreamer_imagination_dream_strip_contact_anchor]] | `dsrl_b0{3,4}_{M,XS}_dp1`, `rppo_{b0{1..4},nmn_g32_b0{1..4}}_mc_dp1_*` |
| 07-27 | **Dreamer 5-perspective "fresh-eyes" investigation + bins±6/rr arms** | Five parallel Fable investigations (speed / faithfulness / curves / hyperparameter regime / task difficulty); then a single-variable arm chain. | **Implementation clean; agent is sample-efficient (2.4× rPPO at matched episodes) but throughput-bound; XS≈M** (the 20.7M-param model is oversized). Standout suspect: the **critic's two-hot value grid at symlog ±20 gives ~52 survival-steps per bin** — the whole task spans ~1.2 bins. Shipped configurable bin range (D-017); launched ±20 control vs **bins±6** at two replay ratios. Sidelines: rPPO trains at γ=0.95 (a myopic survival surrogate); the task is "different in kind" vs the DreamerV3 suite. | [dreamer_srl_investigation/SYNTHESIS_20260727](../active/dreamer_srl_investigation/SYNTHESIS_20260727.md), [DREAMER_SRL_INVESTIGATION](../active/dreamer_srl_investigation/DREAMER_SRL_INVESTIGATION.md) | [[20260727_0537_dreamer_five_perspective_investigation_verdict]], [[20260727_0538_critic_twohot_bin_resolution_bottleneck]], [[20260727_0539_task_different_in_kind_reward_algebra]], [[20260727_0540_rppo_gamma095_myopic_survival_surrogate]], [[20260727_0541_total_steps_footgun_dreamer_resume_path]], [[20260727_0542_eval_seed_testing_seed_config_owned_flags]] | `dsrl_b03_XS_bins6_rr0p0625` (arm1), `dsrl_b03_XS_bins6_rr0p25` (arm2), `dsrl_b0{3,4}_{M,XS}_dp1` (controls) |
| 07-28→07-29 | **Hierarchical-modality encoder arms + agent-team maintenance** | On top of the bins±6 winner, two hierarchical-encoding designs (`hier_heads`, `hier_mirror`); plus a curriculum arm. Agent-team: added a plan/analysis reviewer, re-cut model tiers, renamed the in-repo memory layer to the "LLM Wiki". | Arms training (current tail — no verdict yet). | [dreamer_srl_v2/hierarchical_modality_encoder_plan](../../develop/active/dreamer_srl_v2/hierarchical_modality_encoder_plan.md) | [[20260728_1644_plan_reviewer_and_analysis_verdict_gate]], [[20260728_1642_llm_wiki_rename_ends_memory_collision]] | `dsrl_b03_XS_bins6_rr0p25_hier_{heads,mirror}`, `dsrl_curric123_hier_mirror` |

---

## Thread summaries

### Thread 1 — Hypervigilance / danger discrimination
**Arc:** *Does the agent recognise danger, or only react to pain?* Started (early May) with the "sameProp"
manipulation — predator and rabbit given the same smell. Mean-distance avoidance was **refuted**
(no class-conditional spatial avoidance), but the behavior-measure toolkit exposed an **event-level**
discrimination (bush-diving + eat-suppression near the predator, +37 pp). June's control battery
(chasing-rabbit, predator-only, matched-aggression) delivered the crisp verdict: avoidance is
**pain-consequence-driven, not anticipatory** — the agent forages until it is hit. The discrimination
"metric" turned out to be a spatial-encounter artifact, which **reframed the whole experimental
environment as a behavior-measurement platform** rather than a discrimination oracle. The hunger-gated
olfactory-ambiguity program (hg01–hg10) tried to *force* pre-contact discrimination via asymmetric
stakes and scarcity, but it stayed a clean null ("lethality masks gating"). **Current state:** the
honest measurement is event-level and post-contact; the injury-gated olfactory-noise route (basic06)
is the untested next hypothesis for eliciting genuine hypervigilance.

### Thread 2 — NMN / FiLM neuromodulator architecture
**Arc:** Does a FiLM "context" network beat a plain baseline? The **noise-heterogeneity sweep refuted
it** (5–13 steps *worse*); the temp-clip rerun recovered most of the gap but never won on the
single-task setting. The **one clear win** came under a **continual active↔passive schedule** (+107/+132
steps on return-to-active, ~25× seed noise) — the modulator helps with *recovery/plasticity*, not
steady-state ceiling. A long 50M-ep basic04 head-to-head again showed **no win (276 vs 290)** with the
temperature railing at its ceiling. The FiLM grouping screen (g1–g128) and the plain-vs-NMN ladder were
re-launched under corrected `decay_power` in late July. A recurring blocker: the raw modulator hidden
vector is **never logged**, so the CKA/Mahalanobis "is the modulator actually engaging" pre-check
remains un-computable. **Current state:** positive only in the continual regime; steady-state null;
config-boundary traps (dead `lr_critic`, silent Multiplicative fallback) found in the July audit.

### Thread 3 — In-house DreamerV3 (`dreamer_srl`) vs recurrent-PPO
**Arc:** The longest thread. May's diagnosis localised collapse to the **world-model reward head**; a
fix cascade (zero-init, paper bins) cut reward error −54% but didn't close survival. A PI call pivoted
to **sheeprl-direct**, then back to a **disciplined JAX rebuild (v2/v3)** that reached parity and beat
sheeprl on food-only (501 vs 500) — v1's root bug was **REINFORCE re-sampling at loss-time**. June/July
fixed a recompile storm, a jit-once hang, and a 128-env speed cliff. The July Fable audits found the
math cores bit-identical but **deviations in the glue**, and that "Dreamer fails at predators" was
partly a **metric artifact** (predator-count mixture) and a **logging-granularity artifact** (Dreamer's
curve only *looked* noisier). The 5-perspective investigation (07-27) reframed the whole gap:
implementation is clean, the agent is **2.4× more sample-efficient than rPPO per episode** but ~100×
slower in wall-clock on an **oversized** network, and the likely remaining bottleneck is the **critic's
two-hot bin resolution** (~52 survival-steps per bin). **Current state:** bins±6 and replay-ratio arms
training to settle the causal question; hierarchical-encoder arms on top.

### Thread 4 — Interoceptive behavior-measure & probe platform
**Arc:** Grew out of the hypervigilance work when mean distances proved too blunt. The toolkit-v1
(M1/M2/M5/M7 event measures) shipped in May and immediately paid off (event-level sameProp
discrimination). June's study formalised foraging/avoidance measures vs nutrition/injury on frozen
checkpoints and produced durable behavioural findings (hunger sets timing; flee needs motion + olfaction;
cover-use is late-emerging). July industrialised it: an ~8.8× faster probe eval, batched Dreamer eval,
a 6-level × 12-condition × 11-measure dataset, and — the capstone — **eval-during-training** logging
live `Probe/*` behaviour curves to WandB at every checkpoint. **Current state:** the platform is the
project's main scientific instrument; the statistics tutorial warns against deterministic-probe
significance inflation.

### Thread 5 — Environment & curriculum engineering
**Arc:** The v2.0 unified-animal refactor (late May) enabled per-episode distributional sampling; June
added per-episode entity-count variance, per-episode predator params, the predator jump/pounce, and the
bush perfect-refuge. The `basic00`–`basic07` ladder is the standard task suite. Two hard lessons: easy
levels **over-train-collapse** to entropy≈0, and the **continual curriculum underperforms from-scratch**
(loss-of-plasticity, not budget). rPPO is recompile-immune to per-episode variance; Dreamer needed
fixed-width masked resets. **Current state:** ladder + variants are the workhorse; the predator-count
mixture is a known survival-metric confound awaiting a conditioned metric.

### Thread 6 — Config-system correctness & cluster/agent tooling
**Arc:** The v3.0 `extends:` layering overhaul (June) was elegant but hid a **severe latent bug** — the
*trainer* never resolved `extends:`, silently dropping noise/random-init layers (found July 3). Then a
CLI flag was found silently shadowing `num_envs` (128→16), and finally `sensory.decay_power` was found
**silently drifted 1.0→2.0** across every recent run. Each discovery hardened the platform:
"config-owns-values", "verify the actual saved state", and the `CONFIG_CRITICAL_SETTINGS.md`
registry + change-log guardrail wired into 5 agents. Two Fable "fresh-eyes" diagnoses (07-23) landed
~12 verified fixes and made the config layer **fail loud**. Cluster-side: scripts reorganisation with a
dependency map, heterogeneous-GPU tooling (`gpu-status`), and the run_command.py launch discipline.
**Current state:** the governance scaffolding now exists precisely because these silent drifts each
invalidated a batch of runs.

---

## Gaps & caveats

- **Owning-doc drift:** the June-20 diary rows for the hunger-gated linear-decay and scarcity campaigns
  point at `hunger_gated_lindecay/` and `hypervig_scarcity/` folders that **no longer exist** — the
  surviving owning docs are `hypervigilance/20260620_hunger_gated_step1_linear_olfactory_decay.md` and
  `hypervigilance/20260620_hypervig_scarcity_olfactory_ambiguity.md` (linked above).
- **Several 128-env / variant / GAE-MC waves (July 8–22)** were launched from `train_command-agent.sh`
  blocks with a bare `-` in the diary Doc column (no dedicated design doc) — they are grouped here under
  their campaign but do not each own an analysis doc.
- **The decay_power=1.0 relaunch (07-26) and all arms after it are still training** — no closing verdict
  exists yet for the bins±6, replay-ratio, hierarchical-encoder, or clean plain-vs-NMN ladder campaigns.
- **NMN modulator-engagement** remains unverifiable: the raw modulator hidden vector was never logged
  in any run (a standing metrics request since May 13).

---

## Appendix A — Model-tag index

Grouped by campaign; tags rendered as inline code. Season abbreviations: `rppo` = recurrent-PPO,
`dsrl`/`dreamer_srl` = in-house JAX DreamerV3, `hg` = hunger-gated, `b0N`/`basic0N` = curriculum level,
`dp1` = corrected `decay_power` 1.0.

| Tag family | Campaign | Purpose (one line) |
|---|---|---|
| `hypervigilance-sameprop-relog-seed4{2,3}` | SameProp R1 | Reproduce the R1 rabbit-vs-predator distance asymmetry across seeds. |
| `hypervigilance-round2{,5,6}-{A1,C}-seed4{2..5}` | SameProp R2/R2.5/R2.6 | Matched-smell class-conditional avoidance; A1 = corner-camping cell, C = discriminator cell. |
| `rppo_nmn_het_p{1..5}_{unmod,film_g1}_c*` | NMN heterogeneity sweep | FiLM vs plain across 5 noise-heterogeneity profiles. |
| `rppo_nmn_tempceil10_p{3,4,5}_film_g1_s*` | NMN temp-clip rerun | Raise modulator temperature ceiling 3→10; test if the null was a cap. |
| `rppo_nmn_cont_dr_{mod,unmod}_s0_r2` | NMN continual double-return | 5-stage active↔passive schedule; the one clear FiLM win. |
| `rppo_nmn_meta_spec_{active,passive}_{matched,distinct,swapped}_s0` | NMN specialist ceilings | Single-world survival ceilings (6 worlds) as meta references. |
| `rppo_nmn_film_g{1..128}_{screen,curric_longL4,basic05all}_s42` | FiLM grouping screen | Sweep FiLM grouping factor g on long level-4 / basic05-all. |
| `rppo_nmn_film_curric_longL4_n114`, `rppo_basic_curriculum_longL4` | NMN vs plain long L4 | 50M-ep head-to-head; modulator did not beat plain (276 vs 290). |
| `rppo_nmn_g32_b0{1..4}_{mc,gae}_dp1_*` | Plain-vs-NMN dp1 ladder | Clean decay_power-1.0 relaunch of the NMN g32 comparison. |
| `dreamer_conv_{NoPred,Pred}_rr06_*` | Dreamer conventional-fixes battery | Test top conventional causes (reward scale, replay ratio) — refuted on predator. |
| `dreamer_zinit_*`, `dreamer_twohotrng_*` | Dreamer fix cascade Z1/Z2 | Zero-init reward+critic; paper-canonical two-hot bins. |
| `sheeprl_*`, `jax_sheeprl_matched_n{4,16}_s0` | sheeprl bridge / parity / SPS | sheeprl-direct DreamerV3 bridge + JAX-vs-sheeprl speed comparison. |
| `dreamer_srl_v2_postfix_*` | dreamer-srl v2 parity + hyperparameter search | v2 rebuild; XS/16/4M production recipe (~184 survival). |
| `dreamer_srl_curric3_{T1..T4,size}_s42` | dreamer_srl 3-stage size curriculum | Grow size+entities vs from-scratch; ~45% fewer episodes to same plateau. |
| `recurrent_ppo_08-singlePredRabbit_disengage_s42` | Predator-vs-rabbit control | Single predator + rabbit, lethal contact; danger-recognition control. |
| `rppo_basic0{0..4}_*`, `rppo_basic_curriculum_n106` | basic ladder (from-scratch + continual) | Convergence/collapse per level; continual = negative transfer. |
| `rppo_hg0{1..10}_*`, `rppo_hvs_{scarce,abundant}_s42` | Hunger-gated olfactory ambiguity | Two-knob smell-distinguishability sweep; clean discrimination null. |
| `dsrl_basic0{0..5}_*` | Dreamer basic ladder | Dreamer on the basic levels; recompile-storm + jit-once fixes. |
| `rppo_basic0{5,6,7}_*`, `rppo_basic_ladder6_*` | noise / random-init / jump ladder | basic05 all-combined noise, basic06 injury-gated noise, basic07 jump-attack. |
| `rppo_basic0{0..5}_*_128env_*`, `*_cont100M_*` | 128-env relaunch | Re-run the ladder at the intended 128 envs (was silently 16). |
| `rppo_b0{3,4}_sz{128,XS,S,M,L,XL}_*`, `basic0{3,4}_size_sweep` | Model-size sweeps | rPPO + Dreamer size sweep on basic03/04 for the survival dataset. |
| `rppo_b04v0{1..8}_*_128env_*` | basic04 difficulty variants | Slow-move / short-jump / low-damage / attack-success / combined variants. |
| `rppo_b0{3,4}_gae_128env_n106`, `rppo_nmn_g32_b0{3,4}_{gae,mc}_*` | GAE-vs-MC return mode | Return-mode arm (discarded + re-run under dp1). |
| `dsrl_b04_M_128env_*`, `dsrl_b0{3,4}_M_rr*` | Dreamer-vs-rPPO survival/speed | The head-to-head sample-efficiency comparison. |
| `rppo_b04_experimenteval_128env_100M_n110` | Eval-during-training | First live `Probe/*` behaviour curves in WandB. |
| `dsrl_b0{3,4}_{M,XS}_dp1` | decay_power-1.0 Dreamer 2×2 grid | Clean-sensory-world size grid; XS≈M verdict. |
| `dsrl_b03_XS_bins6_rr0p{0625,25}` | Critic two-hot bins±6 arms | Test whether tighter value-bin resolution unblocks learning. |
| `dsrl_b03_XS_bins6_rr0p25_hier_{heads,mirror}`, `dsrl_curric123_hier_mirror` | Hierarchical-encoder arms | Hierarchical modality encoding on the bins±6 winner (current tail). |

---

## Links

- **Diary backbone:** `docs/diary/2026-05-07.md` … `docs/diary/2026-07-28.md` (58 files) — `## Training runs`, `## Progress reports`, `## Events`.
- **LLM Wiki:** [`docs/llm_wiki/ROOT_INDEX.md`](../../llm_wiki/ROOT_INDEX.md) (session-by-session `## Change history`) + the 10 `_topic_index.md` folder indexes.
- **Prior study summaries** (this folder): NMN comparison, SameProp rabbit-avoidance, DreamerV3 diagnosis/fix-cascade, predator-rabbit discrimination — see [README index](README.md).
- **PI calls:** `docs/pi/calls/2026-05-12_dreamer_backend.md`, `docs/pi/calls/2026-07-27_dreamer_investigation_disposition.md`.
</content>
</invoke>
