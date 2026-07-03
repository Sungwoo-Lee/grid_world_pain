#!/bin/bash
set -euo pipefail

# This script is launched via `run_command.py` which no longer cds to the
# project root or activates a conda env. Both responsibilities live here.
cd /media/nas01/projects/Interoceptive-AI/grid_world_pain

#
# train_command-agent.sh
# ----------------------
# Launch script edited ONLY by the `training-runner` agent.
# The user's manual launch script is `train_command-new.sh` — the agent never touches that.
#
# ---------------------------------------------------------------------------
# train.py CLI arguments (full list — agent fills the relevant ones below)
# ---------------------------------------------------------------------------
# Required:
#   --agent_config <path>     Path to agent/model config YAML (e.g., configs/models/dreamer_v3/dreamer_v3.yaml)
#
# Common (env / training):
#   --config <path>           Base env config YAML (single-stage)
#   --configs-dir <dir>       Directory of stage YAMLs for continual learning (mutually exclusive with --config)
#   --continual-schedule <p>  Schedule YAML (required when --configs-dir is used)
#   --episodes <int>          Number of episodes
#   --total-timesteps <int>   Total timesteps (overrides --episodes if set)
#   --seed <int>              Random seed
#   --num-envs <int>          Parallel envs
#   --num-steps <int>         Steps per iteration (rollout length)
#   --hidden-size <int>       Hidden layer size
#   --lr <float>              Learning rate (overrides agent config)
#   --device <str>            'cuda:N' / 'gpu' / 'cpu'
#   --no-satiation            Disable satiation
#   --no-overeating-death     Disable death by overeating
#   --checkpoint-frequency N  Save checkpoint every N evals
#   --load-checkpoint <path>  Resume from checkpoint
#   --results-dir <path>      Custom results directory
#
# WandB filtering (see convention block below):
#   --wandb-project <str>     Default: 'grid_world_pain' (leave unset; default applies)
#   --wandb-entity <str>      Default: 'sungwoolee'      (leave unset; default applies)
#   --wandb-group <str>       Experiment family
#   --wandb-job-type <str>    Operational category
#   --wandb-name <str>        Run display name in WandB web
#   --wandb-resume-id <str>   Resume an existing WandB run by id
#   --no-wandb                Disable WandB logging entirely
#
# Logging / dev:
#   --tag <str>               Project-internal tag (drives results/JAX_<algo>/<ts>_<tag>/ and logs/<ts>_<tag>.log)
#   --log-interval <int>      WandB logging interval (iterations)
#   --log-accumulate / --no-log-accumulate
#                             Accumulate episode metrics across the log interval (default: accumulate)
#   --quiet                   Suppress stdout / progress bar
#   --debug                   Verbose step-by-step progress
#   --profile                 jax.profiler trace; forces --no-wandb + --quiet
#
# ---------------------------------------------------------------------------
# WandB-field convention (the agent fills all four below)
# ---------------------------------------------------------------------------
#   --wandb-group     experiment family — top dir under configs/environment/experiment/
#                       e.g. 'basic', 'hypervigilance', 'noise'
#
#   --wandb-job-type  operational category. Default 'prod'.
#                     Set to 'debug' / 'test' / 'pilot' / 'ablation' only when the user says so.
#                     (Algorithm is already filterable via Config.agent.algorithm —
#                      job-type is reserved for ops metadata.)
#
#   --wandb-name      run display name in WandB web. Format:
#                       <algo>_<config_stem>_n<node>            (single seed)
#                       <algo>_<config_stem>_s<seed>_n<node>    (seed override)
#                     e.g. 'dreamer_v3_00-5X5_NoPred_n113'
#
#   --tag             identical to --wandb-name. Drives local paths:
#                       results/JAX_<algo>/<ts>_<tag>/
#                       logs/<ts>_<tag>.log
#                     and shows up as Config.tag in the WandB run config.
#
# Defaults from configs/logger/wandb.yaml: project=grid_world_pain, entity=sungwoolee.
# The agent leaves --wandb-project and --wandb-entity unset so those defaults apply.
# ---------------------------------------------------------------------------

# hunger_gated_lindecay sweep — 2026-06-20 through 2026-06-22
# 10 recurrent_ppo runs across nodes 101/102/103/110/106/108, seed 42, ~10M episodes, fresh-init.
# wandb-group: hunger_gated_lindecay, job-type: prod
# Launched via CIFS-bypass /tmp scripts; this file is the audit record.
#
# Run 01: rppo_hg01_s0_sig0_dp1_s42      — node 108, cuda:0  (held anchor; launched 2026-06-22)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/01-s0_sig0.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:0 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg01_s0_sig0_dp1_s42 \
#   --tag rppo_hg01_s0_sig0_dp1_s42
#
# Run 02: rppo_hg02_s0.05_sig0_dp1_s42   — node 101, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/02-s0.05_sig0.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:0 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg02_s0.05_sig0_dp1_s42 \
#   --tag rppo_hg02_s0.05_sig0_dp1_s42

# ---------------------------------------------------------------------------
# basic_curriculum continual-learning run — 2026-06-22
# RecurrentPPO, 5-stage schedule (00→04 basic configs), ~10M total steps
# (1M/1M/2M/2M/4M), node 106 cuda:0, wandb-group: basic_curriculum
# Re-launch after BMState stage-transition NameError fix in train.py
# ---------------------------------------------------------------------------
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --configs-dir configs/environment/experiment/basic \
#   --continual-schedule configs/continual/basic_curriculum_schedule.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 \
#   --device cuda:0 \
#   --wandb-group basic_curriculum \
#   --wandb-job-type prod \
#   --wandb-name rppo_basic_curriculum_n106 \
#   --tag rppo_basic_curriculum_n106

# ---------------------------------------------------------------------------
# basic_curriculum long-L4 continual run — 2026-06-23
# RecurrentPPO (unmodulated), 5-stage schedule (longL4 variant), stage 4 runs ~1B eps
# (1M/1M/2M/2M then stage-4 runs from 6M to 1B — manually stopped by user).
# Node 114, cuda:1, wandb-group: basic_curriculum
# CIFS-bypass: launched via /tmp script — this file is the audit record.
# ---------------------------------------------------------------------------
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --configs-dir configs/environment/experiment/basic \
#   --continual-schedule configs/continual/basic_curriculum_schedule_longL4.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 \
#   --device cuda:1 \
#   --wandb-group basic_curriculum \
#   --wandb-job-type prod \
#   --wandb-name rppo_basic_curriculum_longL4 \
#   --tag rppo_basic_curriculum_longL4

# ---------------------------------------------------------------------------
# basic_curriculum long-L4 continual run — FiLM/NMN modulated — 2026-06-24
# RecurrentPPO + NMN FiLM modulator (per-neuron γ/β, temp_clip [0.5, 5.0])
# 5-stage longL4 schedule: stages 0-3 = 1M/1M/2M/2M eps; stage 4 far-sight
# runs from 6M to 1B eps (manually stopped). Comparison target: unmod run oq2vvh8g.
# Node 114, cuda:2, wandb-group: basic_curriculum
# CIFS-bypass: launched via /tmp script — this file is the audit record.
# ---------------------------------------------------------------------------
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --configs-dir configs/environment/experiment/basic \
#   --continual-schedule configs/continual/basic_curriculum_schedule_longL4.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g1_tempceil5.yaml \
#   --num-envs 128 \
#   --device cuda:2 \
#   --wandb-group basic_curriculum \
#   --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_curric_longL4_n114 \
#   --tag rppo_nmn_film_curric_longL4_n114

# ---------------------------------------------------------------------------
# basic standalone — random-init 10x10 — 2026-06-27
# RecurrentPPO (unmodulated), single-config from scratch, 10M steps.
# 05-random_init_10x10: per-episode randomised predator/food/bush/rock counts,
# random start nutrition [0,100] + injury [0,100], full-grid spawn, clean smell.
# Standalone (NOT continual). Level 4 basic converged ~3.6M steps → 10M gives margin.
# checkpoint_frequency in episodes: ~1000 eps ≈ 200k steps at ~200 avg steps/ep.
# Node 112, cuda:0, wandb-group: basic
# CIFS-bypass: launched via /tmp script — this file is the audit record.
# ---------------------------------------------------------------------------
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/05-random_init_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 16 \
#   --total-timesteps 10000000 \
#   --checkpoint-frequency 1000 \
#   --device cuda:0 \
#   --wandb-group basic \
#   --wandb-job-type prod \
#   --wandb-name rppo_basic05_randinit_n112 \
#   --tag rppo_basic05_randinit_n112

# ---------------------------------------------------------------------------
# NMN FiLM grouping_size screen — 2026-06-27
# RecurrentPPO + NMN FiLM modulator, 8-point group-count curve (grouping_size 1→128)
# Env: 04-far_sight_predator_10x10 (far-sight L4, from scratch, no curriculum)
# Budget: 10M episodes, num_envs=128, seed=42, checkpoint_frequency=100000
# temp_clip: [0.5, 10.0] (non-binding ceiling; de-confounds temperature rail)
# Nodes 106–109, 2 GPUs each; wandb-group: basic_curriculum, job-type: prod
# Design doc: docs/experiments/active/basic_curriculum/NMN_FILM_GROUPING_SCREEN.md
# CIFS-bypass: launched via /tmp scripts — this file is the audit record.
# ---------------------------------------------------------------------------
# Run 1: rppo_nmn_film_g1_screen_s42   — node 106, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-far_sight_predator_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g1_screen.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --log-interval 10 --device cuda:0 --seed 42 \
#   --wandb-group basic_curriculum --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g1_screen_s42 \
#   --tag rppo_nmn_film_g1_screen_s42
#
# Run 2: rppo_nmn_film_g2_screen_s42   — node 106, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-far_sight_predator_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g2_screen.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --log-interval 10 --device cuda:1 --seed 42 \
#   --wandb-group basic_curriculum --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g2_screen_s42 \
#   --tag rppo_nmn_film_g2_screen_s42
#
# Run 3: rppo_nmn_film_g4_screen_s42   — node 110, cuda:0 (reassigned from 107; 107 had no NAS mount)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-far_sight_predator_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g4_screen.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --log-interval 10 --device cuda:0 --seed 42 \
#   --wandb-group basic_curriculum --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g4_screen_s42 \
#   --tag rppo_nmn_film_g4_screen_s42
#
# Run 4: rppo_nmn_film_g8_screen_s42   — node 110, cuda:1 (reassigned from 107; 107 had no NAS mount)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-far_sight_predator_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g8_screen.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --log-interval 10 --device cuda:1 --seed 42 \
#   --wandb-group basic_curriculum --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g8_screen_s42 \
#   --tag rppo_nmn_film_g8_screen_s42
#
# Run 5: rppo_nmn_film_g16_screen_s42  — node 108, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-far_sight_predator_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g16_screen.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --log-interval 10 --device cuda:0 --seed 42 \
#   --wandb-group basic_curriculum --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g16_screen_s42 \
#   --tag rppo_nmn_film_g16_screen_s42
#
# Run 6: rppo_nmn_film_g32_screen_s42  — node 108, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-far_sight_predator_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g32_screen.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --log-interval 10 --device cuda:1 --seed 42 \
#   --wandb-group basic_curriculum --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g32_screen_s42 \
#   --tag rppo_nmn_film_g32_screen_s42
#
# Run 7: rppo_nmn_film_g64_screen_s42  — node 109, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-far_sight_predator_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g64_screen.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --log-interval 10 --device cuda:0 --seed 42 \
#   --wandb-group basic_curriculum --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g64_screen_s42 \
#   --tag rppo_nmn_film_g64_screen_s42
#
# Run 8: rppo_nmn_film_g128_screen_s42 — node 109, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-far_sight_predator_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g128_screen.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --log-interval 10 --device cuda:1 --seed 42 \
#   --wandb-group basic_curriculum --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g128_screen_s42 \
#   --tag rppo_nmn_film_g128_screen_s42
#
# Run 03: rppo_hg03_s0.1_sig0_dp1_s42    — node 101, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/03-s0.1_sig0.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:1 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg03_s0.1_sig0_dp1_s42 \
#   --tag rppo_hg03_s0.1_sig0_dp1_s42
#
# Run 04: rppo_hg04_s0.25_sig0_dp1_s42   — node 102, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/04-s0.25_sig0.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:0 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg04_s0.25_sig0_dp1_s42 \
#   --tag rppo_hg04_s0.25_sig0_dp1_s42
#
# Run 05: rppo_hg05_s0.5_sig0_dp1_s42    — node 102, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/05-s0.5_sig0.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:1 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg05_s0.5_sig0_dp1_s42 \
#   --tag rppo_hg05_s0.5_sig0_dp1_s42
#
# Run 06: rppo_hg06_s0.1_sig0.2_dp1_s42  — node 103, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/06-s0.1_sig0.2.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:0 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg06_s0.1_sig0.2_dp1_s42 \
#   --tag rppo_hg06_s0.1_sig0.2_dp1_s42
#
# Run 07: rppo_hg07_s0.25_sig0.2_dp1_s42 — node 103, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/07-s0.25_sig0.2.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:1 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg07_s0.25_sig0.2_dp1_s42 \
#   --tag rppo_hg07_s0.25_sig0.2_dp1_s42
#
# Run 08: rppo_hg08_s0.5_sig0.2_dp1_s42  — node 110, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/08-s0.5_sig0.2.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:0 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg08_s0.5_sig0.2_dp1_s42 \
#   --tag rppo_hg08_s0.5_sig0.2_dp1_s42
#
# Run 09: rppo_hg09_s0.1_sig0.4_dp1_s42  — node 110, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/09-s0.1_sig0.4.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:1 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg09_s0.1_sig0.4_dp1_s42 \
#   --tag rppo_hg09_s0.1_sig0.4_dp1_s42
#
# Run 10: rppo_hg10_s0.5_sig0.4_dp1_s42  — node 106, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/hunger_gated_lindecay/10-s0.5_sig0.4.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --device cuda:1 \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --seed 42 \
#   --wandb-group hunger_gated_lindecay \
#   --wandb-job-type prod \
#   --wandb-name rppo_hg10_s0.5_sig0.4_dp1_s42 \
#   --tag rppo_hg10_s0.5_sig0.4_dp1_s42

# ---------------------------------------------------------------------------
# basic standalone — random-init 10x10 — RE-LAUNCH 2026-06-27
# PRIOR LAUNCH (same date) used --total-timesteps 10000000 which caused exit in ~45s
# (single-config mode reads episodes default = 100 → ran 100 eps and stopped).
# FIX: replaced with --episodes 10000000 (episode-budget convention).
# RecurrentPPO (unmodulated), single-config from scratch, 10M episodes.
# 05-random_init_10x10: per-episode randomised predator/food/bush/rock counts,
# random start nutrition [0,100] + injury [0,100], full-grid spawn, clean smell.
# Standalone (NOT continual). Level 4 basic converged ~3.6M steps → 10M gives margin.
# Node 112, cuda:0, wandb-group: basic
# Supersedes defunct WandB run k9wyijj3.
# CIFS-bypass: launched via /tmp script — this file is the audit record.
# ---------------------------------------------------------------------------
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/05-random_init_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --checkpoint-frequency 100000 \
#   --device cuda:0 \
#   --log-interval 50 \
#   --wandb-group basic \
#   --wandb-job-type prod \
#   --wandb-name rppo_basic05_randinit_n112 \
#   --tag rppo_basic05_randinit_n112

# ---------------------------------------------------------------------------
# basic standalone — random-init 10x10 + FULLY RANDOMISED PREDATOR — 2026-06-30
# RecurrentPPO (unmodulated), single-config from scratch, 10M episodes.
# 05-random_init_10x10 (commit 033c255): predator/rabbit count 0-2, random start
# nutrition/injury, AND per-episode predator behaviour — detection_range [1,7],
# move_interval [1,3], attack_delay [1,3]. Fully randomised predator behaviour
# distinguishes this run from the prior basic-05 run on node 112 (WandB ga5fkr1q).
# Standalone (NOT continual). Episode budget 10M (manual stop convention).
# Node 110, cuda:0, wandb-group: basic
# CIFS-bypass: launched via /tmp script — this file is the audit record.
# ---------------------------------------------------------------------------
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/05-random_init_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --checkpoint-frequency 100000 \
#   --device cuda:0 \
#   --log-interval 50 \
#   --wandb-group basic \
#   --wandb-job-type prod \
#   --wandb-name rppo_basic05_randpred_n110 \
#   --tag rppo_basic05_randpred_n110

# ---------------------------------------------------------------------------
# basic05_variants — 4-run predator-pressure sweep — 2026-06-30
# RecurrentPPO (unmodulated), single-config (standalone, NOT continual), 10M episodes each.
# Four variants of 05-random_init_10x10 testing increased predator difficulty:
#   v1 (01-more_hiding_predators): more ambush predators (count_high 4→12) — node 108 cuda:0
#   v2 (02-relentless_stamina):    predator max_stamina [30,150] (sometimes chases to starvation) — node 108 cuda:1
#   v3 (03-fast_move_interval):    predator move_interval fixed 1 (always full-speed) — node 109 cuda:0
#   v4 (04-all_combined):          all three factors combined — node 109 cuda:1
# num_envs=16, checkpoint_frequency=100000, log_interval=50
# wandb-group: basic05_variants, job-type: prod
# CIFS-bypass: launched via /tmp scripts — this file is the audit record.
# NOTE: Runs 1 and 2 (node 108) were SKIPPED on initial launch (2026-06-30) — nas01
#   CIFS share not mounted on node 108 at that time. User remounted nas01 on node 108
#   (confirmed 52T free). RE-LAUNCHED 2026-06-30 via CIFS-bypass /tmp scripts.
# ---------------------------------------------------------------------------
# Run 1: rppo_basic05v1_hiding_n108 — node 108, cuda:0 — RE-LAUNCHED 2026-06-30
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic05_variants/01-more_hiding_predators.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --checkpoint-frequency 100000 \
#   --device cuda:0 \
#   --log-interval 50 \
#   --wandb-group basic05_variants \
#   --wandb-job-type prod \
#   --wandb-name rppo_basic05v1_hiding_n108 \
#   --tag rppo_basic05v1_hiding_n108
#
# Run 2: rppo_basic05v2_stamina_n108 — node 108, cuda:1 — RE-LAUNCHED 2026-06-30
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic05_variants/02-relentless_stamina.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --checkpoint-frequency 100000 \
#   --device cuda:1 \
#   --log-interval 50 \
#   --wandb-group basic05_variants \
#   --wandb-job-type prod \
#   --wandb-name rppo_basic05v2_stamina_n108 \
#   --tag rppo_basic05v2_stamina_n108

# Run 3: rppo_basic05v3_fastmove_n109 — node 109, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic05_variants/03-fast_move_interval.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --checkpoint-frequency 100000 \
#   --device cuda:0 \
#   --log-interval 50 \
#   --wandb-group basic05_variants \
#   --wandb-job-type prod \
#   --wandb-name rppo_basic05v3_fastmove_n109 \
#   --tag rppo_basic05v3_fastmove_n109

# Run 4: rppo_basic05v4_all_n109 — node 109, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic05_variants/04-all_combined.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --checkpoint-frequency 100000 \
#   --device cuda:1 \
#   --log-interval 50 \
#   --wandb-group basic05_variants \
#   --wandb-job-type prod \
#   --wandb-name rppo_basic05v4_all_n109 \
#   --tag rppo_basic05v4_all_n109

# ---------------------------------------------------------------------------
# Run 5: rppo_basic05v5_allnoise_n110 — node 110, cuda:1 — 2026-07-02
# basic05_variants/05-all_combined_noise: extends variant-04 (all predator-pressure
# factors combined: more hiding predators, relentless stamina, fast move-interval)
# AND ADDS level-06 injury-gated sensory noise on olfaction (0.15 -> 0.75 sigma at
# max injury) + mild visual noise. Interoception kept clean so pain-gate stays
# reliable. The hardest hypervigilance probe in the basic05_variants family.
# RecurrentPPO (unmodulated), single-config from scratch (standalone, NOT continual).
# num_envs=16, episodes=10000000, checkpoint_frequency=100000, log_interval=50.
# GPU 0 on node 110 busy with rppo_basic05_randpred_n110 (PID 1154); GPU 1 free.
# CIFS-bypass: launched via /tmp script — this file is the audit record.
# ---------------------------------------------------------------------------
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic05_variants/05-all_combined_noise.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --checkpoint-frequency 100000 \
#   --device cuda:1 \
#   --log-interval 50 \
#   --wandb-group basic05_variants \
#   --wandb-job-type prod \
#   --wandb-name rppo_basic05v5_allnoise_n110 \
#   --tag rppo_basic05v5_allnoise_n110

# ---------------------------------------------------------------------------
# basic standalone — level 07 jump/attack 10x10 — 2026-07-03
# RecurrentPPO (unmodulated), single-config from scratch (standalone, NOT continual).
# 07-jump_attack_10x10: extends basic/06 (all predator-pressure factors combined +
# injury-gated olfactory noise), ADDS predator jump/pounce (teleport onto an
# un-hidden agent within attack_range when cooldown is up; stochastic hit/miss).
# The toughest basic level — bush refuge becomes the only reliable defense.
# num_envs=16, episodes=10000000 (episode-budget convention), checkpoint_frequency=100000,
# log_interval=50.
# Node 113, cuda:0 (both 113 GPUs confirmed free RTX 4090; nas01 mounted, 52T free).
# CIFS-bypass: launched via /tmp script — this file is the audit record.
# NOTE: superseded as the active block below — this run (tag rppo_basic07_jump_n113,
# WandB u1tyn8xk) remains ALIVE on node 113 cuda:0; left untouched by the 2026-07-03 launch.
# ---------------------------------------------------------------------------
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/07-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --checkpoint-frequency 100000 \
#   --device cuda:0 \
#   --log-interval 50 \
#   --wandb-group basic \
#   --wandb-job-type prod \
#   --wandb-name rppo_basic07_jump_n113 \
#   --tag rppo_basic07_jump_n113

# ---------------------------------------------------------------------------
# basic standalone — level 07 + wider jump reach (attack_range 2-or-3) — 2026-07-03
# RecurrentPPO (unmodulated), single-config from scratch (standalone, NOT continual).
# 06-jump_range_2to3: basic level 07 (jump/pounce predator) with attack_range widened
# to [2,4] -> per-episode pounce reach randomised to 2 or 3 cells (vs. level 07's
# narrower reach). Tests whether wider ambush range further stresses bush-refuge
# reliance as the agent's primary defense.
# num_envs=16, episodes=10000000 (episode-budget convention), checkpoint_frequency=100000,
# log_interval=50.
# Node 113, cuda:1 (GPU 0 busy with basic/07 run u1tyn8xk; GPU 1 confirmed free RTX 4090;
# nas01 confirmed mounted, 52T free).
# CIFS-bypass: launched via /tmp script — this file is the audit record.
# ---------------------------------------------------------------------------
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic05_variants/06-jump_range_2to3.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --num-envs 16 \
  --episodes 10000000 \
  --checkpoint-frequency 100000 \
  --device cuda:1 \
  --log-interval 50 \
  --wandb-group basic05_variants \
  --wandb-job-type prod \
  --wandb-name rppo_basic07_jumpreach23_n113 \
  --tag rppo_basic07_jumpreach23_n113

# ---------------------------------------------------------------------------
# basic standalone — level 07 jump/attack 10x10 — RE-LAUNCH after extends: fix — 2026-07-03
# RecurrentPPO (unmodulated), single-config from scratch (standalone, NOT continual).
# 07-jump_attack_10x10: extends basic/06 (all predator-pressure factors combined +
# injury-gated olfactory noise) + random-init nutrition/injury + predator jump/pounce.
# RE-LAUNCH RATIONALE: the config-loader `extends:` chain was previously not
# resolved by train.py, so inherited layers (noise + random-init + all-combined +
# jump) were silently dropped. Fixed upstream; this run verifies the fix by
# checking the saved config.yaml for perceptual_noise.enabled=true and
# random_start_injury=true.
# num_envs=16, episodes=10000000 (episode-budget convention), checkpoint_frequency=100000,
# log_interval=50.
# Node 108, cuda:0 (confirmed free RTX 3090; nas01 mounted 52T free; JAX GPU-compile
# check passed jax 0.9.0.1).
# CIFS-bypass: launched via /tmp script — this file is the audit record.
# ---------------------------------------------------------------------------
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic/07-jump_attack_10x10.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --num-envs 16 \
  --episodes 10000000 \
  --checkpoint-frequency 100000 \
  --device cuda:0 \
  --log-interval 50 \
  --wandb-group basic \
  --wandb-job-type prod \
  --wandb-name rppo_basic07_jump_v2_n108 \
  --tag rppo_basic07_jump_v2_n108

# ---------------------------------------------------------------------------
# NMN FiLM grouping_size screen — CONTINUAL (CURRICULUM) RE-LAUNCH 2026-06-27
# RecurrentPPO + NMN FiLM modulator, 8-point group-count curve (grouping_size 1→128)
# REPLACES the from-scratch single-env screen (terminated same day).
# Env: 5-stage basic curriculum (00→04 only), via frozen dir basic_curriculum/
#   (basic_curriculum_schedule_longL4 expects 5 stages; basic/ has 6 after
#    05-random_init_10x10 was added 2026-06-27 → fix: use dedicated basic_curriculum/ dir).
# Budget: schedule-driven (no --episodes); checkpoints from schedule.
# num_envs=128, seed=42, log-interval=10. Matches rppo_nmn_film_curric_longL4_n114 exactly,
# varying only agent_config + tag + node/GPU.
# Nodes 106 (cuda:0/1), 110 (cuda:0/1), 108 (cuda:0/1), 109 (cuda:0/1)
# wandb-group: basic_curriculum, job-type: prod
# Design doc: docs/experiments/active/basic_curriculum/NMN_FILM_GROUPING_SCREEN.md
# CIFS-bypass: launched via /tmp scripts — this file is the audit record.
# ---------------------------------------------------------------------------
# Run 1: rppo_nmn_film_g1_curric_longL4_s42   — node 106, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --configs-dir configs/environment/experiment/basic_curriculum \
#   --continual-schedule configs/continual/basic_curriculum_schedule_longL4.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g1_screen.yaml \
#   --num-envs 128 --seed 42 --log-interval 10 \
#   --device cuda:0 \
#   --wandb-group basic_curriculum --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g1_curric_longL4_s42 \
#   --tag rppo_nmn_film_g1_curric_longL4_s42
#
# Run 2: rppo_nmn_film_g2_curric_longL4_s42   — node 106, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --configs-dir configs/environment/experiment/basic_curriculum \
#   --continual-schedule configs/continual/basic_curriculum_schedule_longL4.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g2_screen.yaml \
#   --num-envs 128 --seed 42 --log-interval 10 \
#   --device cuda:1 \
#   --wandb-group basic_curriculum --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g2_curric_longL4_s42 \
#   --tag rppo_nmn_film_g2_curric_longL4_s42
#
# Run 3: rppo_nmn_film_g4_curric_longL4_s42   — node 110, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --configs-dir configs/environment/experiment/basic_curriculum \
#   --continual-schedule configs/continual/basic_curriculum_schedule_longL4.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g4_screen.yaml \
#   --num-envs 128 --seed 42 --log-interval 10 \
#   --device cuda:0 \
#   --wandb-group basic_curriculum --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g4_curric_longL4_s42 \
#   --tag rppo_nmn_film_g4_curric_longL4_s42
#
# Run 4: rppo_nmn_film_g8_curric_longL4_s42   — node 110, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --configs-dir configs/environment/experiment/basic_curriculum \
#   --continual-schedule configs/continual/basic_curriculum_schedule_longL4.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g8_screen.yaml \
#   --num-envs 128 --seed 42 --log-interval 10 \
#   --device cuda:1 \
#   --wandb-group basic_curriculum --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g8_curric_longL4_s42 \
#   --tag rppo_nmn_film_g8_curric_longL4_s42
#
# Run 5: rppo_nmn_film_g16_curric_longL4_s42  — node 108, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --configs-dir configs/environment/experiment/basic_curriculum \
#   --continual-schedule configs/continual/basic_curriculum_schedule_longL4.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g16_screen.yaml \
#   --num-envs 128 --seed 42 --log-interval 10 \
#   --device cuda:0 \
#   --wandb-group basic_curriculum --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g16_curric_longL4_s42 \
#   --tag rppo_nmn_film_g16_curric_longL4_s42
#
# Run 6: rppo_nmn_film_g32_curric_longL4_s42  — node 108, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --configs-dir configs/environment/experiment/basic_curriculum \
#   --continual-schedule configs/continual/basic_curriculum_schedule_longL4.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g32_screen.yaml \
#   --num-envs 128 --seed 42 --log-interval 10 \
#   --device cuda:1 \
#   --wandb-group basic_curriculum --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g32_curric_longL4_s42 \
#   --tag rppo_nmn_film_g32_curric_longL4_s42
#
# Run 7: rppo_nmn_film_g64_curric_longL4_s42  — node 109, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --configs-dir configs/environment/experiment/basic_curriculum \
#   --continual-schedule configs/continual/basic_curriculum_schedule_longL4.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g64_screen.yaml \
#   --num-envs 128 --seed 42 --log-interval 10 \
#   --device cuda:0 \
#   --wandb-group basic_curriculum --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g64_curric_longL4_s42 \
#   --tag rppo_nmn_film_g64_curric_longL4_s42
#
# Run 8: rppo_nmn_film_g128_curric_longL4_s42 — node 109, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --configs-dir configs/environment/experiment/basic_curriculum \
#   --continual-schedule configs/continual/basic_curriculum_schedule_longL4.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g128_screen.yaml \
#   --num-envs 128 --seed 42 --log-interval 10 \
#   --device cuda:1 \
#   --wandb-group basic_curriculum --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g128_curric_longL4_s42 \
#   --tag rppo_nmn_film_g128_curric_longL4_s42

# ---------------------------------------------------------------------------
# NMN FiLM grouping_size screen — basic05_variants/04-all_combined (single-env,
# NOT continual) — 2026-07-02
# RecurrentPPO + NMN FiLM modulator, 8-point group-count curve (grouping_size 1->128).
# Env: basic05_variants/04-all_combined (all three predator-pressure factors
# combined: more hiding predators count_high 4->12, predator max_stamina
# [30,150], predator move_interval fixed 1). Harder than the basic_curriculum
# far-sight screen (this env family has never been curriculum-staged).
# Pre-flight: env config resolves/builds cleanly, obs_dim=27 action_dim=6
# (matches expectation); all 8 agent configs verified byte-identical except
# modulation.grouping_size (and header comments).
# Cards are 11GB RTX 2080 Ti (nodes 101/103/104/105), smaller than the 24GB
# cards used for the earlier far-sight screen at --num-envs 128. Smoke-tested
# --num-envs 128 on node 101 (200-episode --no-wandb dry run): peak GPU memory
# well under 11GB, no OOM -> used 128 uniformly across all 8 runs.
# Budget: 10M episodes, seed=42, checkpoint_frequency=100000, log_interval=50.
# Nodes 101, 103, 104, 105 (2 GPUs each); wandb-group: basic05_variants, job-type: prod
# CIFS-bypass: launched via /tmp scripts — this file is the audit record.
# ---------------------------------------------------------------------------
# Run 1: rppo_nmn_film_g1_basic05all_s42   — node 101, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic05_variants/04-all_combined.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g1_screen.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --log-interval 50 --device cuda:0 --seed 42 \
#   --wandb-group basic05_variants --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g1_basic05all_s42 \
#   --tag rppo_nmn_film_g1_basic05all_s42
#
# Run 2: rppo_nmn_film_g2_basic05all_s42   — node 101, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic05_variants/04-all_combined.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g2_screen.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --log-interval 50 --device cuda:1 --seed 42 \
#   --wandb-group basic05_variants --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g2_basic05all_s42 \
#   --tag rppo_nmn_film_g2_basic05all_s42
#
# Run 3: rppo_nmn_film_g4_basic05all_s42   — node 103, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic05_variants/04-all_combined.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g4_screen.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --log-interval 50 --device cuda:0 --seed 42 \
#   --wandb-group basic05_variants --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g4_basic05all_s42 \
#   --tag rppo_nmn_film_g4_basic05all_s42
#
# Run 4: rppo_nmn_film_g8_basic05all_s42   — node 103, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic05_variants/04-all_combined.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g8_screen.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --log-interval 50 --device cuda:1 --seed 42 \
#   --wandb-group basic05_variants --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g8_basic05all_s42 \
#   --tag rppo_nmn_film_g8_basic05all_s42
#
# Run 5: rppo_nmn_film_g16_basic05all_s42  — node 104, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic05_variants/04-all_combined.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g16_screen.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --log-interval 50 --device cuda:0 --seed 42 \
#   --wandb-group basic05_variants --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g16_basic05all_s42 \
#   --tag rppo_nmn_film_g16_basic05all_s42
#
# Run 6: rppo_nmn_film_g32_basic05all_s42  — node 104, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic05_variants/04-all_combined.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g32_screen.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --log-interval 50 --device cuda:1 --seed 42 \
#   --wandb-group basic05_variants --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g32_basic05all_s42 \
#   --tag rppo_nmn_film_g32_basic05all_s42
#
# Run 7: rppo_nmn_film_g64_basic05all_s42  — node 105, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic05_variants/04-all_combined.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g64_screen.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --log-interval 50 --device cuda:0 --seed 42 \
#   --wandb-group basic05_variants --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g64_basic05all_s42 \
#   --tag rppo_nmn_film_g64_basic05all_s42
#
# Run 8: rppo_nmn_film_g128_basic05all_s42 — node 105, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic05_variants/04-all_combined.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g128_screen.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --log-interval 50 --device cuda:1 --seed 42 \
#   --wandb-group basic05_variants --wandb-job-type prod \
#   --wandb-name rppo_nmn_film_g128_basic05all_s42 \
#   --tag rppo_nmn_film_g128_basic05all_s42
