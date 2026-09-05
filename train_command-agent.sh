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
# Config-owns-values convention
# ---------------------------------------------------------------------------
# --num-envs, --seed, and --checkpoint-frequency are CONFIG-OWNED —
# configs/train/default.yaml is authoritative for these three values (Dreamer
# reads it directly; RecurrentPPO additionally layers configs/train/recurrent_ppo.yaml
# on top, overriding checkpoint_frequency to 200000 and log_interval to 500 —
# both already correct in-config). Standard launches must NOT pass these
# three flags on the CLI; do so only as an intentional, one-off deviation,
# and flag it to the user when you do (this is the root-cause fix for the
# --num-envs 16 bug that silently trained 6 runs at the wrong parallelism).
#
# --episodes MUST always be passed explicitly on every launch — the config's
# episodes: 100 is a smoke-test safety placeholder, not a real budget.
#
# Minimal config-owned launch form:
#   /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#     --config <env_config.yaml> \
#     --agent_config <agent_config.yaml> \
#     --episodes <int> \
#     --device cuda:N \
#     --log-interval <int> \
#     --wandb-group <str> --wandb-job-type <str> \
#     --wandb-name <str> --tag <str>
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
#
# --- (previous active block preserved below the new one; see history above) ---
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic05_variants/06-jump_range_2to3.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 16 \
#   --episodes 10000000 \
#   --checkpoint-frequency 100000 \
#   --device cuda:1 \
#   --log-interval 50 \
#   --wandb-group basic05_variants \
#   --wandb-job-type prod \
#   --wandb-name rppo_basic07_jumpreach23_n113 \
#   --tag rppo_basic07_jumpreach23_n113

# ---------------------------------------------------------------------------
# basic04 size sweep — XS/S/M/L/XL (NO 128-baseline; that's already running on
# node 106) — 2026-07-15
# RecurrentPPO size-variant agent configs (hidden_size 256/512/1024/2048/4096,
# mirrors DreamerV3 XS/S/M/L/XL presets), single-config from scratch (standalone,
# NOT continual). Env: basic/04-jump_attack_10x10 (jump/pounce predator,
# attack_range [2,3], no sensory noise, random_start_injury true).
# num_envs=16, episodes=100000000, checkpoint_frequency=100000, log_interval=50.
# wandb-group: basic04_size_sweep, job-type: prod.
# Pack-node-first: 107 (XS/S), 108 (M/L), 110:0 already used elsewhere -> 110 not
# used here per the launch plan (XL goes to 110:0 per plan; verify no collision
# at launch time). All GPUs confirmed idle + NAS mounted + JAX GPU-compile check
# passed (jax 0.9.0.1) on 107/108/110 prior to launch.
# CIFS-bypass: launched via /tmp scripts — this file is the audit record.
# ---------------------------------------------------------------------------
# Run XS: rppo_b04_szXS_16env_n107 — node 107, cuda:0 — LAUNCHED (WandB uxcu4wid)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_XS.yaml \
#   --num-envs 16 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group basic04_size_sweep --wandb-job-type prod \
#   --wandb-name rppo_b04_szXS_16env_n107 --tag rppo_b04_szXS_16env_n107
#
# Run S: rppo_b04_szS_16env_n107 — node 107, cuda:1 — LAUNCHED (WandB 4p93s8ev)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_S.yaml \
#   --num-envs 16 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group basic04_size_sweep --wandb-job-type prod \
#   --wandb-name rppo_b04_szS_16env_n107 --tag rppo_b04_szS_16env_n107
#
# Run M: rppo_b04_szM_16env_n108 — node 108, cuda:0 — LAUNCHED (WandB jwy6opzk)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_M.yaml \
#   --num-envs 16 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group basic04_size_sweep --wandb-job-type prod \
#   --wandb-name rppo_b04_szM_16env_n108 --tag rppo_b04_szM_16env_n108
#
# Run L: rppo_b04_szL_16env_n108 — node 108, cuda:1 — LAUNCHED (WandB r62tqyzh)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_L.yaml \
#   --num-envs 16 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group basic04_size_sweep --wandb-job-type prod \
#   --wandb-name rppo_b04_szL_16env_n108 --tag rppo_b04_szL_16env_n108
#
# Run XL: rppo_b04_szXL_16env_n110 — node 110, cuda:0 (confirmed idle immediately
# pre-launch; 0 MiB used, no compute processes)
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo_XL.yaml \
  --num-envs 16 --episodes 100000000 --checkpoint-frequency 100000 \
  --device cuda:0 --log-interval 50 \
  --wandb-group basic04_size_sweep --wandb-job-type prod \
  --wandb-name rppo_b04_szXL_16env_n110 --tag rppo_b04_szXL_16env_n110

# ---------------------------------------------------------------------------
# basic standalone — level 06 sensory noise 10x10 — RE-LAUNCH after extends: fix — 2026-07-03
# RecurrentPPO (unmodulated), single-config from scratch (standalone, NOT continual).
# 06-sensory_noise_10x10: extends basic/05 (random-init nutrition/injury + all
# combined predator-pressure factors), ADDS injury-gated olfactory sensory noise.
# RE-LAUNCH RATIONALE: the config-loader `extends:` chain was previously not
# resolved by train.py, so the inherited noise layer was silently dropped. Fixed
# upstream; this run verifies the fix by checking the saved config.yaml for
# perceptual_noise.enabled=true, olfaction injury_noise_scale=4.0, and
# random_start_injury=true.
# num_envs=16, episodes=10000000 (episode-budget convention), checkpoint_frequency=100000,
# log_interval=50.
# Node 110, cuda:0 (confirmed free RTX 3090; nas01 mounted 52T free; JAX GPU-compile
# check passed jax 0.9.0.1).
# CIFS-bypass: launched via /tmp script — this file is the audit record.
# ---------------------------------------------------------------------------
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic/06-sensory_noise_10x10.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --num-envs 16 --episodes 10000000 --checkpoint-frequency 100000 \
  --device cuda:0 --log-interval 50 \
  --wandb-group basic --wandb-job-type prod \
  --wandb-name rppo_basic06_noise_n110 --tag rppo_basic06_noise_n110

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

# ---------------------------------------------------------------------------
# basic05_variants/06 — jump-REACH comparison to basic/07 — RE-LAUNCH after
# extends: fix — 2026-07-03
# RecurrentPPO (unmodulated), single-config from scratch (standalone, NOT continual).
# 06-jump_range_2to3.yaml: extends basic/07-jump_attack_10x10 (all predator-pressure
# factors + injury-gated olfactory noise + random-init + jump/pounce), REDECLARES
# only the predator's attack_range to [2,4] so the per-episode pounce reach is
# uniformly 2 OR 3 cells (vs. basic/07's fixed reach). Same config as the earlier
# rppo_basic07_jumpreach23_n113 entry above but on GPU 0 (not 1) with the
# convention-matching tag, launched after the config-loader extends: fix so the
# inherited noise + random-init layers are no longer silently dropped.
# num_envs=16, episodes=10000000 (episode-budget convention), checkpoint_frequency=100000,
# log_interval=50.
# Node 113, cuda:0 (confirmed free RTX 4090; nas01 mounted 52T free; JAX GPU-compile
# check passed jax 0.9.0.1).
# CIFS-bypass: launched via /tmp script — this file is the audit record.
# ---------------------------------------------------------------------------
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic05_variants/06-jump_range_2to3.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --num-envs 16 --episodes 10000000 --checkpoint-frequency 100000 \
  --device cuda:0 --log-interval 50 \
  --wandb-group basic --wandb-job-type prod \
  --wandb-name rppo_basic07_jumpreach_n113 --tag rppo_basic07_jumpreach_n113

# ---------------------------------------------------------------------------
# basic standalone — 05-random_init_10x10 (all-combined predator pressure +
# random-init nutrition/injury) — RE-LAUNCH after extends: fix — 2026-07-03
# RecurrentPPO (unmodulated), single-config from scratch (standalone, NOT continual).
# 05-random_init_10x10: extends environment/default (sensory/noise/behavior_measures
# inherited); random-init nutrition/injury + all predator-pressure factors folded
# in directly in this file (more hiding predators, relentless stamina, fast
# move-interval) — proper random init merged with variant-04 on 2026-07-02.
# RE-LAUNCH RATIONALE: the config-loader `extends:` chain was previously not
# resolved by train.py, so the inherited `environment/default` layer was silently
# dropped. Fixed upstream; this run verifies the fix by checking the saved
# config.yaml for random_start_injury=true and predator count_high=2,
# move_interval=[1,1] (proves the all-combined + random-init layers are present).
# num_envs=16, episodes=10000000 (episode-budget convention), checkpoint_frequency=100000,
# log_interval=50.
# Node 109, cuda:0 (confirmed free RTX 3090; nas01 mounted 52T free; JAX GPU-compile
# check passed jax 0.9.0.1).
# CIFS-bypass: launched via /tmp script — this file is the audit record.
# ---------------------------------------------------------------------------
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic/05-random_init_10x10.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --num-envs 16 --episodes 10000000 --checkpoint-frequency 100000 \
  --device cuda:0 --log-interval 50 \
  --wandb-group basic --wandb-job-type prod \
  --wandb-name rppo_basic05_alcomb_n109 --tag rppo_basic05_alcomb_n109

# ---------------------------------------------------------------------------
# basic05_variants/02 — relentless predator stamina — RE-LAUNCH after extends:
# fix — 2026-07-03 — PACKED onto node 108 GPU 1 (GPU 0 already runs basic/07,
# left untouched)
# RecurrentPPO (unmodulated), single-config from scratch (standalone, NOT continual).
# 02-relentless_stamina.yaml: extends basic05_variants/... -> basic/05-random_init_10x10
# (random-init nutrition/injury + all-combined predator-pressure factors), REDECLARES
# only the predator's max_stamina to [30,150] (some episodes spawn a relentless
# chaser that can drive the agent to starvation) and move_interval [1,3].
# RE-LAUNCH RATIONALE: supersedes the pre-fix 2026-06-30 run tagged
# rppo_basic05v2_stamina_n108 (same config, same node/GPU) — that run predates the
# config-loader extends: fix, so inherited layers (random_start_injury, all-combined
# predator pressure) were silently dropped. Verifies via saved config.yaml:
# random_start_injury=true, predator move_interval=[1,3], max_stamina=[30,150].
# num_envs=16, episodes=10000000 (episode-budget convention), checkpoint_frequency=100000,
# log_interval=50.
# Node 108, cuda:1 (GPU 0 busy with basic/07 jump run, PID confirmed via nvidia-smi
# 91% util; GPU 1 confirmed free 0% util/5MiB; nas01 mounted 52T free; JAX GPU-compile
# check passed jax 0.9.0.1).
# CIFS-bypass: launched via /tmp script — this file is the audit record.
# ---------------------------------------------------------------------------
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic05_variants/02-relentless_stamina.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --num-envs 16 --episodes 10000000 --checkpoint-frequency 100000 \
  --device cuda:1 --log-interval 50 \
  --wandb-group basic --wandb-job-type prod \
  --wandb-name rppo_basic05v02_relentstam_n108 --tag rppo_basic05v02_relentstam_n108

# ---------------------------------------------------------------------------
# basic05_variants/03 — fixed fast predator move_interval — PACKED onto node 109
# GPU 1 (GPU 0 already runs basic/05 all-combined, left untouched) — 2026-07-03
# RecurrentPPO (unmodulated), single-config from scratch (standalone, NOT continual).
# 03-fast_move_interval.yaml: extends basic/05-random_init_10x10 (random-init
# nutrition/injury + all-combined predator-pressure factors), REDECLARES only the
# predator's move_interval to a fixed [1,1] (always full-speed, no slow-predator
# episodes) and max_stamina [30,30].
# Verifies via saved config.yaml: random_start_injury=true, predator
# move_interval=[1,1], max_stamina=[30,30].
# num_envs=16, episodes=10000000 (episode-budget convention), checkpoint_frequency=100000,
# log_interval=50.
# Node 109, cuda:1 (GPU 0 busy with rppo_basic05_alcomb_n109, 78% util; GPU 1
# confirmed free 0% util/63MiB; nas01 mounted 52T free; JAX GPU-compile check
# passed jax 0.9.0.1).
# CIFS-bypass: launched via /tmp script — this file is the audit record.
# ---------------------------------------------------------------------------
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic05_variants/03-fast_move_interval.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --num-envs 16 --episodes 10000000 --checkpoint-frequency 100000 \
  --device cuda:1 --log-interval 50 \
  --wandb-group basic --wandb-job-type prod \
  --wandb-name rppo_basic05v03_fastmove_n109 --tag rppo_basic05v03_fastmove_n109

# ---------------------------------------------------------------------------
# FRESH re-leveled 6-level basic ladder — 2026-07-04
# Ladder re-leveled per commit b093023 (jump moved before noise; [2,3] default
# attack_range; variants retired into the main basic/ ladder). All 6 runs are
# RecurrentPPO (unmodulated), single-config from scratch (standalone, NOT
# continual), num_envs=16, episodes=10000000 (episode-budget convention),
# checkpoint_frequency=100000, log_interval=50.
# 00 static / 01 slow / 02 pred+rabbit / 03 random-init all-combined /
# 04 jump/pounce (no noise) / 05 sensory noise (full stack).
# Pack-node-first: 106 (00,01), 108 (02,03), 109 (04,05); node 110 left free;
# node 107 EXCLUDED (no NAS mount).
# CIFS-bypass: launched via /tmp scripts — this file is the audit record.
# ---------------------------------------------------------------------------
# Run 00: rppo_basic00_static_n106 — node 106, cuda:0
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic/00-static_predator_5x5.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --num-envs 16 --episodes 10000000 --checkpoint-frequency 100000 \
  --device cuda:0 --log-interval 50 \
  --wandb-group basic --wandb-job-type prod \
  --wandb-name rppo_basic00_static_n106 --tag rppo_basic00_static_n106

# Run 01: rppo_basic01_slow_n106 — node 106, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/01-slow_predator_5x5.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 16 --episodes 10000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group basic --wandb-job-type prod \
#   --wandb-name rppo_basic01_slow_n106 --tag rppo_basic01_slow_n106
#
# Run 02: rppo_basic02_predrabbit_n108 — node 108, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/02-predator_and_rabbit_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 16 --episodes 10000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group basic --wandb-job-type prod \
#   --wandb-name rppo_basic02_predrabbit_n108 --tag rppo_basic02_predrabbit_n108
#
# Run 03: rppo_basic03_randinit_n108 — node 108, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/03-random_init_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 16 --episodes 10000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group basic --wandb-job-type prod \
#   --wandb-name rppo_basic03_randinit_n108 --tag rppo_basic03_randinit_n108
#
# Run 04: rppo_basic04_jump_n109 — node 109, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 16 --episodes 10000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group basic --wandb-job-type prod \
#   --wandb-name rppo_basic04_jump_n109 --tag rppo_basic04_jump_n109
#
# Run 05: rppo_basic05_noise_n109 — node 109, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/05-sensory_noise_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 16 --episodes 10000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group basic --wandb-job-type prod \
#   --wandb-name rppo_basic05_noise_n109 --tag rppo_basic05_noise_n109

# ---------------------------------------------------------------------------
# basic03_randinit RESUME-AND-EXTEND — 10M -> 100M episode budget — 2026-07-08
# RecurrentPPO (unmodulated), resumes the COMPLETED basic03_randinit run
# (results/JAX_RecurrentPPO/20260704-230444_rppo_basic03_randinit_n108/models,
# latest checkpoint step 10000016) and extends the SAME single-task config
# (basic/03-random_init_10x10) to a 100M-episode budget. No curriculum, no
# multi-task change — user will stop the run manually when satisfied.
# --num-envs 16 is MANDATORY: the checkpoint was trained with 16 parallel envs;
# train.py's global default (128) causes a fatal Orbax shape-mismatch on
# restore. Confirmed by a prior resume-correctness gate test with this exact
# arg shape (weights + optimizer + counters all restore correctly).
# --checkpoint-frequency 100000 (finer than default) so behavior can be probed
# across many intermediate checkpoints.
# Node 108, cuda:1 (confirmed free via nvidia-smi immediately pre-launch: 0%
# util, 5 MiB used; GPU 0 busy at 97% util with an unrelated run — untouched).
# CIFS-bypass: launched via /tmp script — this file is the audit record.
# ---------------------------------------------------------------------------
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --config configs/environment/experiment/basic/03-random_init_10x10.yaml \
  --load-checkpoint results/JAX_RecurrentPPO/20260704-230444_rppo_basic03_randinit_n108/models \
  --num-envs 16 \
  --episodes 100000000 \
  --checkpoint-frequency 100000 \
  --device cuda:1 \
  --wandb-group basic \
  --wandb-job-type prod \
  --wandb-name rppo_basic03_randinit_cont100M_n108 \
  --tag rppo_basic03_randinit_cont100M_n108

# ---------------------------------------------------------------------------
# 6-level basic ladder RE-LAUNCH FROM SCRATCH at --num-envs 128 — 2026-07-08
# The originals (Run 00-05 above, 2026-07-04 block) were wrongly launched at
# --num-envs 16; intended setting is 128. TERMINATED (SIGINT) prior to this
# relaunch: rppo_basic00_static_n106 (PID 4846, node 106 cuda:0),
# rppo_basic01_slow_n106 (PID 5144, node 106 cuda:1),
# rppo_basic02_predrabbit_n108 (PID 1522844, node 108 cuda:0),
# rppo_basic03_randinit_cont100M_n108 (PID 2360259, node 108 cuda:1 — the
# checkpoint-resume/100M-budget variant from the block directly above).
# All 6 below are FRESH from-scratch runs (no --load-checkpoint), num_envs=128,
# episodes=10000000, checkpoint_frequency=100000, log_interval=50.
# Pack-node-first: 109 (00,01), 110 (02,03) — both fully free (4 GPUs);
# 106 (04) and 108 (05) — freed up by the termination above.
# GPU-compile pre-flight (jax 0.9.0.1, matmul+block_until_ready) passed on all
# 4 nodes; nvidia-smi confirmed all 8 target GPUs free immediately pre-launch.
# CIFS-bypass: launched via /tmp scripts — this file is the audit record.
# ---------------------------------------------------------------------------
# Run 00: rppo_basic00_static_128env_n109 — node 109, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/00-static_predator_5x5.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group basic --wandb-job-type prod \
#   --wandb-name rppo_basic00_static_128env_n109 --tag rppo_basic00_static_128env_n109
#
# Run 01: rppo_basic01_slow_128env_n109 — node 109, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/01-slow_predator_5x5.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group basic --wandb-job-type prod \
#   --wandb-name rppo_basic01_slow_128env_n109 --tag rppo_basic01_slow_128env_n109
#
# Run 02: rppo_basic02_predrabbit_128env_n110 — node 110, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/02-predator_and_rabbit_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group basic --wandb-job-type prod \
#   --wandb-name rppo_basic02_predrabbit_128env_n110 --tag rppo_basic02_predrabbit_128env_n110
#
# Run 03: rppo_basic03_randinit_128env_n110 — node 110, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/03-random_init_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group basic --wandb-job-type prod \
#   --wandb-name rppo_basic03_randinit_128env_n110 --tag rppo_basic03_randinit_128env_n110
#
# Run 04: rppo_basic04_jump_128env_n106 — node 106, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group basic --wandb-job-type prod \
#   --wandb-name rppo_basic04_jump_128env_n106 --tag rppo_basic04_jump_128env_n106
#
# Run 05: rppo_basic05_noise_128env_n108 — node 108, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/05-sensory_noise_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 10000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group basic --wandb-job-type prod \
#   --wandb-name rppo_basic05_noise_128env_n108 --tag rppo_basic05_noise_128env_n108

# ---------------------------------------------------------------------------
# 6-level basic ladder — RESUME-AND-EXTEND levels 03/04/05 — 10M -> 100M
# episode budget — 2026-07-09
# RecurrentPPO (unmodulated), resumes the just-finished 128-env from-scratch
# runs (Run 03/04/05 in the block directly above) from their ~10M-episode
# checkpoints and extends the SAME single-task config to a 100M-episode
# budget. No curriculum, no config change — user will stop manually.
# Levels 00/01/02 are left untouched (still running / not extended).
# --num-envs 128 is MANDATORY — matches the checkpoints' h_state batch dim
# (confirmed via each checkpoint's saved config.yaml: num_envs: 128).
# Latest checkpoint steps confirmed on disk: basic03=10000029,
# basic04=10000005, basic05=10000167.
# Nodes 106 (cuda:0/1) and 108 (cuda:0) — all confirmed free via nvidia-smi
# immediately pre-launch (0% util); JAX GPU-compile check (jax 0.9.0.1,
# matmul + block_until_ready) passed on both nodes.
# CIFS-bypass: launched via /tmp scripts — this file is the audit record.
# ---------------------------------------------------------------------------
# Run 03 resume: rppo_basic03_randinit_128env_100M_n106 — node 106, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/03-random_init_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --load-checkpoint results/JAX_RecurrentPPO/20260708-193853_rppo_basic03_randinit_128env_n110/models \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 \
#   --wandb-group basic --wandb-job-type prod \
#   --wandb-name rppo_basic03_randinit_128env_100M_n106 \
#   --tag rppo_basic03_randinit_128env_100M_n106
#
# Run 04 resume: rppo_basic04_jump_128env_100M_n106 — node 106, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --load-checkpoint results/JAX_RecurrentPPO/20260708-193852_rppo_basic04_jump_128env_n106/models \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 \
#   --wandb-group basic --wandb-job-type prod \
#   --wandb-name rppo_basic04_jump_128env_100M_n106 \
#   --tag rppo_basic04_jump_128env_100M_n106
#
# Run 05 resume: rppo_basic05_noise_128env_100M_n108 — node 108, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/05-sensory_noise_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --load-checkpoint results/JAX_RecurrentPPO/20260708-193853_rppo_basic05_noise_128env_n108/models \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 \
#   --wandb-group basic --wandb-job-type prod \
#   --wandb-name rppo_basic05_noise_128env_100M_n108 \
#   --tag rppo_basic05_noise_128env_100M_n108

# ---------------------------------------------------------------------------
# basic03 MODEL-SIZE SWEEP — 6 runs — 2026-07-13
# RecurrentPPO, 6-point hidden_size curve (128/256/512/1024/2048/4096) on the
# same env config basic/03-random_init_10x10 (all predator-pressure factors +
# random start nutrition/injury, no jump/pounce, clean smell).
# num_envs=16, episodes=100000000 (100M budget), checkpoint_frequency=100000,
# log_interval=50. All sizes fit a 24GB RTX 3090 at num_envs=16 (XL peaks ~18GB,
# pre-checked before launch).
# Pack-node-first: 107 (cuda:0,1), 108 (cuda:0,1), 109 (cuda:0,1); node 110 left free.
# wandb-group: basic03_size_sweep, job-type: prod
# CIFS-bypass: launched via /tmp scripts, one at a time — this file is the audit record.
# ---------------------------------------------------------------------------
# Run 1 (128, baseline): rppo_b03_sz128_n107 — node 107, cuda:0
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic/03-random_init_10x10.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --num-envs 16 --episodes 100000000 --checkpoint-frequency 100000 \
  --device cuda:0 --log-interval 50 \
  --wandb-group basic03_size_sweep --wandb-job-type prod \
  --wandb-name rppo_b03_sz128_n107 --tag rppo_b03_sz128_n107
#
# Run 2 (XS/256): rppo_b03_szXS_n107 — node 107, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/03-random_init_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_XS.yaml \
#   --num-envs 16 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group basic03_size_sweep --wandb-job-type prod \
#   --wandb-name rppo_b03_szXS_n107 --tag rppo_b03_szXS_n107
#
# Run 3 (S/512): rppo_b03_szS_n108 — node 108, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/03-random_init_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_S.yaml \
#   --num-envs 16 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group basic03_size_sweep --wandb-job-type prod \
#   --wandb-name rppo_b03_szS_n108 --tag rppo_b03_szS_n108
#
# Run 4 (M/1024): rppo_b03_szM_n108 — node 108, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/03-random_init_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_M.yaml \
#   --num-envs 16 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group basic03_size_sweep --wandb-job-type prod \
#   --wandb-name rppo_b03_szM_n108 --tag rppo_b03_szM_n108
#
# Run 5 (L/2048): rppo_b03_szL_n109 — node 109, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/03-random_init_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_L.yaml \
#   --num-envs 16 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group basic03_size_sweep --wandb-job-type prod \
#   --wandb-name rppo_b03_szL_n109 --tag rppo_b03_szL_n109
#
# Run 6 (XL/4096): rppo_b03_szXL_n109 — node 109, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/03-random_init_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_XL.yaml \
#   --num-envs 16 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group basic03_size_sweep --wandb-job-type prod \
#   --wandb-name rppo_b03_szXL_n109 --tag rppo_b03_szXL_n109

# ---------------------------------------------------------------------------
# basic04 (jump) MODEL-SIZE SWEEP — same 6-point hidden_size curve, RELAUNCH on
# basic/04-jump_attack_10x10 — 2026-07-14
# PHASE 1 (done first): the 6 basic03_size_sweep runs above (rppo_b03_sz128_n107,
# szXS_n107, szS_n108, szM_n108, szL_n109) were killed via SIGINT and confirmed
# gone; rppo_b03_szXL_n109 had ALREADY died on its own at 2026-07-14 07:27 from a
# node-109 GPU1 hardware fault (CUDA_ERROR_LAUNCH_FAILED flood in
# logs/20260713_185426.log, PID 3376504).
# PHASE 2: same 6 sizes (128/256/512/1024/2048/4096), same env family swapped to
# basic/04-jump_attack_10x10 (extends basic/03 directly, adds predator jump/pounce:
# attack_range [2,3], attack_success_rate 0.5; no sensory noise; random_start_injury
# inherited true). num_envs=16, episodes=100000000, checkpoint_frequency=100000,
# log_interval=50. wandb-group: basic04_size_sweep, job-type: prod.
# BLOCKER: node 109 is NOT usable right now — nvidia-smi reports
# "Unable to determine the device handle for GPU1: Unknown Error", and a fresh JAX
# process cannot even cuInit() on GPU0 either (CUDA_ERROR_UNKNOWN, falls back to
# CPU) — the whole node's CUDA driver stack looks poisoned by the GPU1 fault.
# Requires a reboot / driver reset outside training-runner scope. Only the 4 runs
# below (107:0, 107:1, 108:0, 108:1 — sizes 128/XS/S/M) were launched; szL_n109 and
# szXL_n109 are PENDING, surfaced to the user for a node-109-recovery or
# reassignment decision.
# Nodes 107/108 confirmed idle (GPU0+GPU1 both 0% util / ~0 MiB) and JAX
# GPU-compile check passed (jax 0.9.0.1, real matmul) before launch.
# CIFS-bypass: launched via /tmp scripts, one at a time — this file is the audit record.
# ---------------------------------------------------------------------------
# Run 1 (128, baseline): rppo_b04_sz128_n107 — node 107, cuda:0
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --num-envs 16 --episodes 100000000 --checkpoint-frequency 100000 \
  --device cuda:0 --log-interval 50 \
  --wandb-group basic04_size_sweep --wandb-job-type prod \
  --wandb-name rppo_b04_sz128_n107 --tag rppo_b04_sz128_n107
#
# Run 2 (XS/256): rppo_b04_szXS_n107 — node 107, cuda:1
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo_XS.yaml \
  --num-envs 16 --episodes 100000000 --checkpoint-frequency 100000 \
  --device cuda:1 --log-interval 50 \
  --wandb-group basic04_size_sweep --wandb-job-type prod \
  --wandb-name rppo_b04_szXS_n107 --tag rppo_b04_szXS_n107
#
# Run 3 (S/512): rppo_b04_szS_n108 — node 108, cuda:0
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo_S.yaml \
  --num-envs 16 --episodes 100000000 --checkpoint-frequency 100000 \
  --device cuda:0 --log-interval 50 \
  --wandb-group basic04_size_sweep --wandb-job-type prod \
  --wandb-name rppo_b04_szS_n108 --tag rppo_b04_szS_n108
#
# Run 4 (M/1024): rppo_b04_szM_n108 — node 108, cuda:1
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo_M.yaml \
  --num-envs 16 --episodes 100000000 --checkpoint-frequency 100000 \
  --device cuda:1 --log-interval 50 \
  --wandb-group basic04_size_sweep --wandb-job-type prod \
  --wandb-name rppo_b04_szM_n108 --tag rppo_b04_szM_n108
#
# ---------------------------------------------------------------------------
# Run 5/6 REASSIGNMENT — node 109 unusable, moved to node 110 — 2026-07-14
# Node 109 confirmed hardware/CUDA-faulted (GPU1 "Unknown Error", GPU0 also
# fails cuInit()) — unusable, left untouched. Reassigned to node 110
# (RTX 3090 x2), confirmed free (0 MiB / 0% util both GPUs), nas01 mounted
# (74T free), JAX GPU-compile check passed (jax 0.9.0.1, real matmul).
# Same settings as the 4 already running (128/XS on 107, S/M on 108):
# num_envs=16, episodes=100000000, checkpoint_frequency=100000, log_interval=50.
# Launched ONE AT A TIME via CIFS-bypass /tmp scripts — this file is the audit record.
# ---------------------------------------------------------------------------
# Run 5 (L/2048): rppo_b04_szL_n110 — node 110, cuda:0
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo_L.yaml \
  --num-envs 16 --episodes 100000000 --checkpoint-frequency 100000 \
  --device cuda:0 --log-interval 50 \
  --wandb-group basic04_size_sweep --wandb-job-type prod \
  --wandb-name rppo_b04_szL_n110 --tag rppo_b04_szL_n110
#
# Run 6 (XL/4096): rppo_b04_szXL_n110 — node 110, cuda:1
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo_XL.yaml \
  --num-envs 16 --episodes 100000000 --checkpoint-frequency 100000 \
  --device cuda:1 --log-interval 50 \
  --wandb-group basic04_size_sweep --wandb-job-type prod \
  --wandb-name rppo_b04_szXL_n110 --tag rppo_b04_szXL_n110

# ---------------------------------------------------------------------------
# basic04_variants — 4-run predator-pressure DIFFICULTY SWEEP (default/original
# RecurrentPPO size, hidden_size 128) — 2026-07-21
# RecurrentPPO (unmodulated, default size — NOT any XS/S/M/L/XL variant),
# single-config from scratch (standalone, NOT continual), 100M-episode budget.
# Four variants of basic/04-jump_attack_10x10, each softening exactly one
# predator knob relative to the harsh baseline (attack_range [2,3], damage
# 15-120, move_interval [1,1] fixed full-speed):
#   v01 (01-slow_move_interval):  move_interval [1,1] -> [1,3] (predator sometimes slower)
#   v02 (02-short_attack_range):  attack_range [2,3] -> [1,2] (shorter pounce reach)
#   v03 (03-reduced_damage):      damage 15-120 -> 15-80 (no one-shot kill; max_injury=100)
#   v04 (04-all_combined):        all three softenings combined
# num_envs=128, episodes=100000000, checkpoint_frequency=100000, log_interval=50.
# Memory: default 128-size model at 128 envs peaks ~1.4GB on a 24GB RTX 3090 (verified fine).
# wandb-group: basic04_variants, job-type: prod.
# Pack-node-first: 111 (v01 cuda:0, v02 cuda:1), 112 (v03 cuda:0, v04 cuda:1).
# Both nodes/GPUs confirmed idle (0% util) + NAS mounted (74T free) + JAX
# GPU-compile check passed (jax 0.9.0.1, real matmul on GPU) before launch.
# Launched ONE AT A TIME via CIFS-bypass /tmp scripts — this file is the audit record.
# ---------------------------------------------------------------------------
# Run v01: rppo_b04v01_slowmove_128env_n111 — node 111, cuda:0 — LAUNCHED (WandB 78nirb3q, PID 44147)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic04_variants/01-slow_move_interval.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group basic04_variants --wandb-job-type prod \
#   --wandb-name rppo_b04v01_slowmove_128env_n111 --tag rppo_b04v01_slowmove_128env_n111
#
# Run v02: rppo_b04v02_shortjump_128env_n111 — node 111, cuda:1 — LAUNCHED (WandB ugxsegwy, PID 47014)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic04_variants/02-short_attack_range.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group basic04_variants --wandb-job-type prod \
#   --wandb-name rppo_b04v02_shortjump_128env_n111 --tag rppo_b04v02_shortjump_128env_n111
#
# Run v03: rppo_b04v03_lowdmg_128env_n112 — node 112, cuda:0 — LAUNCHED (WandB lhy6be5b, PID 685263)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic04_variants/03-reduced_damage.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group basic04_variants --wandb-job-type prod \
#   --wandb-name rppo_b04v03_lowdmg_128env_n112 --tag rppo_b04v03_lowdmg_128env_n112
#
# Run v04: rppo_b04v04_allcomb_128env_n112 — node 112, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic04_variants/04-all_combined.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group basic04_variants --wandb-job-type prod \
#   --wandb-name rppo_b04v04_allcomb_128env_n112 --tag rppo_b04v04_allcomb_128env_n112

# ---------------------------------------------------------------------------
# basic04_variants — 4 NEW difficulty variants (05-08), default rPPO size — 2026-07-21
# RecurrentPPO (unmodulated, default size — hidden_size 128, recurrent_ppo.yaml),
# single-config from scratch (standalone, NOT continual), 100M-episode budget.
# Four MORE variants of basic/04-jump_attack_10x10 (companions to v01-v04 already
# running on 111/112), each probing the pounce HIT RATE (attack_success_rate) axis
# while deliberately keeping the predator's move_interval at [1,1] (fast chase) —
# per the 2026-07-21 diary finding that a fast predator preserves bush-hiding
# behaviour, whereas slowing it (as v04-all_combined does) suppresses hiding to ~8%:
#   v05 (05-attack_success_030): asr 0.5 -> 0.3 (softer hit rate only)
#   v06 (06-attack_success_070): asr 0.5 -> 0.7 (harder hit rate only)
#   v07 (07-combined_move1_asr030): attack_range [2,3]->[1,2], damage 15-120->15-80,
#                                   asr 0.5->0.3 — ALL combined but move_interval STAYS [1,1]
#   v08 (08-combined_move1_asr070): same as v07 but asr 0.5->0.7
# num_envs=128, episodes=100000000, checkpoint_frequency=100000, log_interval=50.
# wandb-group: basic04_variants, job-type: prod.
# Pack-node-first: 106 (v05 cuda:0, v06 cuda:1), 107 (v07 cuda:0, v08 cuda:1).
# 108 deliberately left free. 111/112 (v01-v04) untouched. 109 faulted (not used).
# 110/113/114 in use by others (not used).
# Both nodes' GPUs confirmed idle (0 MiB / 0% util) + NAS mounted (74T free) + JAX
# GPU-compile check passed (jax 0.9.0.1, real matmul on GPU) on 106 and 107 before launch.
# Launched ONE AT A TIME via CIFS-bypass /tmp scripts, per the LAUNCH->WAIT->VERIFY
# protocol (60s wait + pgrep-only verification) — this file is the audit record.
# ---------------------------------------------------------------------------
# Run v05: rppo_b04v05_asr030_128env_n106 — node 106, cuda:0 — LAUNCHED (WandB bpb8wja7, PID 3168139)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic04_variants/05-attack_success_030.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group basic04_variants --wandb-job-type prod \
#   --wandb-name rppo_b04v05_asr030_128env_n106 --tag rppo_b04v05_asr030_128env_n106
#
# Run v06: rppo_b04v06_asr070_128env_n106 — node 106, cuda:1 — LAUNCHED (WandB 7lwxbf2j, PID 3168391)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic04_variants/06-attack_success_070.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group basic04_variants --wandb-job-type prod \
#   --wandb-name rppo_b04v06_asr070_128env_n106 --tag rppo_b04v06_asr070_128env_n106
#
# Run v07: rppo_b04v07_comb030_128env_n107 — node 107, cuda:0 — LAUNCHED (WandB unvd1rbw, PID 3229788)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic04_variants/07-combined_move1_asr030.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group basic04_variants --wandb-job-type prod \
#   --wandb-name rppo_b04v07_comb030_128env_n107 --tag rppo_b04v07_comb030_128env_n107
#
# Run v08: rppo_b04v08_comb070_128env_n107 — node 107, cuda:1 — LAUNCHED (WandB n9jtaqi7, PID 3230039)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic04_variants/08-combined_move1_asr070.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group basic04_variants --wandb-job-type prod \
#   --wandb-name rppo_b04v08_comb070_128env_n107 --tag rppo_b04v08_comb070_128env_n107

# ---------------------------------------------------------------------------
# GAE return-mode variant — basic/03 and basic/04 — 2026-07-22
# RecurrentPPO with recurrent_ppo_gae.yaml (return_mode: GAE, everything else
# identical to the default MC-return recurrent_ppo.yaml: hidden_size 128,
# default fc/actor/critic layer sizes). Compares GAE(lambda) returns vs. the
# usual Monte-Carlo return baseline used across the rest of the project.
# basic/03-random_init_10x10: random start nutrition/injury, all-combined
#   predator pressure, NO jump/pounce (attack_range [0,0]), no sensory noise.
# basic/04-jump_attack_10x10: extends basic/03, ADDS predator jump/pounce
#   (attack_range [2,3], attack_success_rate 0.5), no sensory noise.
# num_envs=128, episodes=100000000, checkpoint_frequency=100000, log_interval=50.
# Node 106, cuda:0 (basic/03) + cuda:1 (basic/04) — both confirmed FREE via
# gpu_status.py; GPU-compile preflight passed (jax 0.9.0.1) on node 106.
# wandb-group: rppo_gae, job-type: prod. Launched ONE AT A TIME per instruction.
# CIFS-bypass: launched via /tmp scripts — this file is the audit record.
# ---------------------------------------------------------------------------
# Run 1: rppo_b03_gae_128env_n106 — node 106, cuda:0
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic/03-random_init_10x10.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo_gae.yaml \
  --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
  --device cuda:0 --log-interval 50 \
  --wandb-group rppo_gae --wandb-job-type prod \
  --wandb-name rppo_b03_gae_128env_n106 --tag rppo_b03_gae_128env_n106

# Run 2: rppo_b04_gae_128env_n106 — node 106, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_gae.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group rppo_gae --wandb-job-type prod \
#   --wandb-name rppo_b04_gae_128env_n106 --tag rppo_b04_gae_128env_n106

# ---------------------------------------------------------------------------
# NMN-g32 vs plain head-to-head — basic/03 + basic/04 x {MC, GAE} — 2026-07-23
# RecurrentPPO + NMN FiLM modulator (grouping_size=32, temp_clip [0.5,10.0]),
# single-config from scratch (standalone, NOT continual), 100M-episode budget.
# 4-run 2x2: env {basic/03 no-jump, basic/04 jump/pounce} x return_mode {MC, GAE}.
# Agent configs: recurrent_ppo_nmn_film_g32_screen.yaml (MC) and its byte-identical
# GAE sibling recurrent_ppo_nmn_film_g32_screen_gae.yaml (commit 54792eb).
# Comparison targets: the plain-rPPO baselines rppo_b03_gae_128env_n106 /
# rppo_b04_gae_128env_n106 on node 106 (same env configs, same launch params).
# num_envs=128 (config default, passed explicitly to match baselines),
# episodes=100000000, checkpoint_frequency=100000 (matches baselines; deviates
# from recurrent_ppo.yaml's 200000 by explicit instruction), log_interval=50.
# wandb-group: nmn_g32_vs_plain_b0304, job-type: prod.
# Pack-node-first: 112 (b03 MC cuda:0, b03 GAE cuda:1), 111 (b04 MC cuda:0,
# b04 GAE cuda:1). All 4 GPUs confirmed idle (0 compute procs) + NAS mounted
# (73T free) on both nodes + JAX GPU-compile check passed (jax 0.9.0.1, real
# matmul, both CudaDevices visible) on 111 and 112 pre-launch.
# SMOKE: 5-min foreground dry run of b03 x g32-MC on 112:0 (--no-wandb, tag
# smoke_nmn_g32_b03_delete_me) passed — FiLM g32 enabled, obs 27 / act 6, JIT +
# checkpoint clean, ~800 it/s; debris deleted.
# Launched ONE AT A TIME via CIFS-bypass /tmp scripts per the LAUNCH->WAIT->VERIFY
# protocol (--no-tail, 60s wait, pgrep-only verify) — this file is the audit record.
# ---------------------------------------------------------------------------
# Run 1: rppo_nmn_g32_b03_mc_n112 — node 112, cuda:0 — LAUNCHED (WandB tgilz3eu, PID 3320979, log 20260723_210707.log)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/03-random_init_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g32_screen.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group nmn_g32_vs_plain_b0304 --wandb-job-type prod \
#   --wandb-name rppo_nmn_g32_b03_mc_n112 --tag rppo_nmn_g32_b03_mc_n112
#
# Run 2: rppo_nmn_g32_b03_gae_n112 — node 112, cuda:1 — LAUNCHED (WandB 34frib5j, PID 3321084, log 20260723_210712.log)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/03-random_init_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g32_screen_gae.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group nmn_g32_vs_plain_b0304 --wandb-job-type prod \
#   --wandb-name rppo_nmn_g32_b03_gae_n112 --tag rppo_nmn_g32_b03_gae_n112
#
# Run 3: rppo_nmn_g32_b04_mc_n111 — node 111, cuda:0 — LAUNCHED (WandB 48cb7q5l, PID 2810347, log 20260723_210715.log)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g32_screen.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group nmn_g32_vs_plain_b0304 --wandb-job-type prod \
#   --wandb-name rppo_nmn_g32_b04_mc_n111 --tag rppo_nmn_g32_b04_mc_n111
#
# Run 4: rppo_nmn_g32_b04_gae_n111 — node 111, cuda:1 — LAUNCHED (WandB j9gs52dh, PID 2810517, log 20260723_210721.log)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g32_screen_gae.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group nmn_g32_vs_plain_b0304 --wandb-job-type prod \
#   --wandb-name rppo_nmn_g32_b04_gae_n111 --tag rppo_nmn_g32_b04_gae_n111

# ---------------------------------------------------------------------------
# basic04 100M relaunch — during-training experiment-eval opt-in via config
# layer (replaces the now-removed --experiment-eval CLI flag) — 2026-07-25
# RecurrentPPO (unmodulated, default size), single-config from scratch
# (standalone, NOT continual). Env: basic/04-jump_attack_10x10 (jump/pounce
# predator, no sensory noise, random_start_injury true).
# --eval-config configs/evaluation/experiment_on.yaml opts into the
# during-training behavior-probe eval (extends evaluation/default, sets
# experiment.during_training.enabled: true -> conditions=all, episodes=30,
# log_measures=[bush_dwell, survival_steps], every_n_checkpoints=1). Dispatches
# an on-node CPU subprocess at every checkpoint, logs Experiment/* to WandB.
# Feature already verified end-to-end; this run's purpose is provenance parity
# with the committed config-layer mechanism (vs. the removed CLI flag).
# num_envs=128, episodes=100000000, checkpoint_frequency=100000, log_interval
# default (config-owned) -- matches the established convention for every other
# basic/04-jump_attack_10x10 launch in this file.
# Node 110, cuda:0 (confirmed idle 0 MiB/0% util, no compute procs; NAS mounted
# 72T free; JAX GPU-compile check passed jax 0.9.0.1, real matmul on GPU).
# CIFS-bypass: launched via /tmp script — this file is the audit record.
# ---------------------------------------------------------------------------
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
  --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
  --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 --device cuda:0 \
  --eval-config configs/evaluation/experiment_on.yaml \
  --wandb-group basic04_experimenteval --wandb-job-type prod \
  --wandb-name rppo_b04_experimenteval_128env_100M_n110 --tag rppo_b04_experimenteval_128env_100M_n110

# ---------------------------------------------------------------------------
# decay_power-1.0 corrected baselines — 4-run relaunch — 2026-07-26
# CONTEXT: sensory.decay_power had silently drifted 1.0->2.0 (commit def81c1,
# 2026-02-27) and was reverted to the intended 1.0 in commit d6240f5 (2026-07-26
# 03:48; see docs/environment/CONFIG_CRITICAL_SETTINGS.md). Every earlier rPPO
# run — including the 2026-07-23 NMN-g32-vs-plain b03/b04 2x2 on nodes 111/112 —
# trained under the wrong decay_power=2.0. These four runs are the corrected
# re-launch of that same 2x2 design (env {basic/03, basic/04} x agent {plain,
# NMN FiLM g32}), now under decay_power=1.0. All prior six rPPO runs on nodes
# 106/111/112 were terminated by the user before this relaunch; those GPUs
# confirmed free. Nodes 101/103/104/105 belong to a colleague (untouched);
# node 114 hosts a parallel Dreamer grid (untouched).
# All 4: RecurrentPPO, return_mode=MC, single-config from scratch (standalone),
# num_envs=128, episodes=100000000, seed=0 (default — not overridden).
# wandb-group: rppo_baseline_dp1, job-type: prod.
# GPUs confirmed idle (0 MiB/0%, no compute procs) + NAS mounted (72T free) on
# both 110 and 113 + JAX GPU-compile check passed (jax 0.9.0.1, real matmul on
# GPU) on both nodes pre-launch. Launched ONE AT A TIME via CIFS-bypass /tmp
# scripts per the LAUNCH->WAIT->VERIFY protocol (--no-tail, 60s wait,
# pgrep-only verify) — this file is the audit record.
# ---------------------------------------------------------------------------
# Run 1: rppo_b03_mc_dp1_n110 — node 110, cuda:0 — plain rPPO, basic/03 (no jump)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/03-random_init_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_baseline_dp1 --wandb-job-type prod \
#   --wandb-name rppo_b03_mc_dp1_n110 --tag rppo_b03_mc_dp1_n110
#
# Run 2: rppo_b04_mc_dp1_n110 — node 110, cuda:1 — plain rPPO, basic/04 (jump/pounce)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group rppo_baseline_dp1 --wandb-job-type prod \
#   --wandb-name rppo_b04_mc_dp1_n110 --tag rppo_b04_mc_dp1_n110
#
# Run 3: rppo_nmn_g32_b03_mc_dp1_n113 — node 113, cuda:0 — NMN FiLM g32, basic/03
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/03-random_init_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g32_screen.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_baseline_dp1 --wandb-job-type prod \
#   --wandb-name rppo_nmn_g32_b03_mc_dp1_n113 --tag rppo_nmn_g32_b03_mc_dp1_n113
#
# Run 1: rppo_b03_mc_dp1_n110 — LAUNCHED (node 110, cuda:0, PID 893609)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/03-random_init_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_baseline_dp1 --wandb-job-type prod \
#   --wandb-name rppo_b03_mc_dp1_n110 --tag rppo_b03_mc_dp1_n110
#
# Run 2: rppo_b04_mc_dp1_n110 — LAUNCHED (node 110, cuda:1, PID 893864)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group rppo_baseline_dp1 --wandb-job-type prod \
#   --wandb-name rppo_b04_mc_dp1_n110 --tag rppo_b04_mc_dp1_n110
#
# Run 3: rppo_nmn_g32_b03_mc_dp1_n113 — LAUNCHED (node 113, cuda:0, PID 4037725)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/03-random_init_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g32_screen.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_baseline_dp1 --wandb-job-type prod \
#   --wandb-name rppo_nmn_g32_b03_mc_dp1_n113 --tag rppo_nmn_g32_b03_mc_dp1_n113
#
# Run 4: rppo_nmn_g32_b04_mc_dp1_n113 — LAUNCHED (node 113, cuda:1, PID 4038066)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g32_screen.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group rppo_baseline_dp1 --wandb-job-type prod \
#   --wandb-name rppo_nmn_g32_b04_mc_dp1_n113 --tag rppo_nmn_g32_b04_mc_dp1_n113

# ---------------------------------------------------------------------------
# decay_power-1.0 corrected baselines — SECOND 4-run relaunch — 2026-07-26
# Same family as the rppo_baseline_dp1 4-run relaunch above; extends the plain-
# vs-NMN-g32 split to basic/01 + basic/02. Plain rPPO on node 107, NMN FiLM g32
# on node 108 (mirrors the 110/113 plain/NMN split above).
# basic/01-slow_predator_5x5: 5x5 grid, 1 slow hunt predator (move_interval 3,
#   no jump), 2 static hiding predators, no rabbit.
# basic/02-predator_and_rabbit_10x10: 10x10 grid, 1 fast hunt predator
#   (move_interval 1, no jump), 4 hiding predators, 1 wandering rabbit.
# All 4: RecurrentPPO, return_mode=MC, single-config from scratch (standalone),
# num_envs=128, episodes=100000000, seed=42 (config default — not overridden,
# per explicit instruction). wandb-group: rppo_baseline_dp1, job-type: prod.
# checkpoint_frequency=100000, log_interval=50 (mirrors the 110/113 pair).
# Nodes 107/108 pre-flighted by the user (NAS mounted, both GPUs idle, 0
# compute procs) + JAX GPU-compile check passed (jax 0.9.0.1, real matmul on
# GPU) on both nodes pre-launch by this agent. Launched ONE AT A TIME via
# CIFS-bypass /tmp scripts per the LAUNCH->WAIT->VERIFY protocol (--no-tail,
# 60s wait, pgrep-only verify) — this file is the audit record.
# ---------------------------------------------------------------------------
# Run 1: rppo_b01_mc_dp1_n107 — node 107, cuda:0 — plain rPPO, basic/01 (5x5, slow predator)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/01-slow_predator_5x5.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_baseline_dp1 --wandb-job-type prod \
#   --wandb-name rppo_b01_mc_dp1_n107 --tag rppo_b01_mc_dp1_n107
#
# Run 2: rppo_b02_mc_dp1_n107 — node 107, cuda:1 — plain rPPO, basic/02 (10x10, predator+rabbit)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/02-predator_and_rabbit_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group rppo_baseline_dp1 --wandb-job-type prod \
#   --wandb-name rppo_b02_mc_dp1_n107 --tag rppo_b02_mc_dp1_n107
#
# Run 3: rppo_nmn_g32_b01_mc_dp1_n108 — node 108, cuda:0 — NMN FiLM g32, basic/01
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/01-slow_predator_5x5.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g32_screen.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_baseline_dp1 --wandb-job-type prod \
#   --wandb-name rppo_nmn_g32_b01_mc_dp1_n108 --tag rppo_nmn_g32_b01_mc_dp1_n108
#
# Run 1: rppo_b01_mc_dp1_n107 — LAUNCHED (node 107, cuda:0, PID 1403622)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/01-slow_predator_5x5.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_baseline_dp1 --wandb-job-type prod \
#   --wandb-name rppo_b01_mc_dp1_n107 --tag rppo_b01_mc_dp1_n107
#
# Run 2: rppo_b02_mc_dp1_n107 — LAUNCHED (node 107, cuda:1, PID 1403874)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/02-predator_and_rabbit_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group rppo_baseline_dp1 --wandb-job-type prod \
#   --wandb-name rppo_b02_mc_dp1_n107 --tag rppo_b02_mc_dp1_n107
#
# Run 3: rppo_nmn_g32_b01_mc_dp1_n108 — LAUNCHED (node 108, cuda:0, PID 4009642)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/01-slow_predator_5x5.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g32_screen.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_baseline_dp1 --wandb-job-type prod \
#   --wandb-name rppo_nmn_g32_b01_mc_dp1_n108 --tag rppo_nmn_g32_b01_mc_dp1_n108
#
# Run 4: rppo_nmn_g32_b02_mc_dp1_n108 — LAUNCHED (node 108, cuda:1, PID 4009895)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/02-predator_and_rabbit_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g32_screen.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group rppo_baseline_dp1 --wandb-job-type prod \
#   --wandb-name rppo_nmn_g32_b02_mc_dp1_n108 --tag rppo_nmn_g32_b02_mc_dp1_n108

# ---------------------------------------------------------------------------
# rppo_bushrefuge — 4-run bush-as-physical-refuge ladder — 2026-08-04
# ---------------------------------------------------------------------------
# WHAT THIS TESTS (plain language): the bush in this project has always concealed the
# agent from predator detection, but predators could still walk THROUGH a bush cell.
# The new configs/environment/experiment/basic_bushrefuge/ ladder (commit 76711de) is a
# byte-for-byte sibling of configs/environment/experiment/basic/ with exactly ONE key
# changed — the bush obstacle now carries `blocks_animals: true`, so predators and the
# wandering rabbit cannot MOVE INTO a bush cell while the agent still enters freely and
# stays concealed. The bush becomes a true physical refuge. Question: does that change
# hiding behaviour?
#
# Plain rPPO (unmodulated, modulation.type: null), single-config from scratch
# (standalone, NOT continual). Settings deliberately identical to the 2026-07-26
# `rppo_baseline_dp1` batch so the two families are directly comparable:
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 --log-interval 50
#   seed: config default 42 (NOT overridden on the CLI)
#
# NEW vs. the 2026-07-26 batch: --eval-config configs/evaluation/experiment_on.yaml
#   turns on the during-training behaviour probe so Experiment/bush_dwell and
#   Experiment/survival_steps plot live on WandB. A bush-refuge manipulation with no
#   bush-dwell curve is not worth running, so this flag is mandatory on all four runs.
#
# Deviations flagged to the user pre-launch:
#   - --checkpoint-frequency 100000 overrides the rPPO config-owned 200000
#     (configs/train/recurrent_ppo.yaml) — intentional, for baseline comparability.
#   - --num-envs 128 is redundant (config-owned value is already 128) but passed
#     explicitly to mirror the baseline batch's command line exactly.
#
# Pre-flight: nodes 102 + 113 both NAS-mounted (71T free), JAX 0.9.0.1 GPU-compile
# check passed on both, all four RTX 4090 GPUs idle. sensory.decay_power resolves to
# 1.0 = registry canonical (docs/environment/CONFIG_CRITICAL_SETTINGS.md).
# wandb-group: rppo_bushrefuge, job-type: prod
# CIFS-bypass: launched via /tmp scripts — this file is the audit record.
# ---------------------------------------------------------------------------
# Run 1: rppo_bushrefuge_b01_n102 — node 102, cuda:0 — 5x5, slow predator (move_interval 3)
# LAUNCHED 2026-08-04, PID 3514180, launcher log logs/20260804_040530.log
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge/01-slow_predator_5x5.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_bushrefuge --wandb-job-type prod \
#   --wandb-name rppo_bushrefuge_b01_n102 --tag rppo_bushrefuge_b01_n102
#
# Run 2: rppo_bushrefuge_b02_n102 — node 102, cuda:1 — 10x10, fast predator + wandering rabbit
# LAUNCHED 2026-08-04, PID 3514932, launcher log logs/20260804_040708.log
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge/02-predator_and_rabbit_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group rppo_bushrefuge --wandb-job-type prod \
#   --wandb-name rppo_bushrefuge_b02_n102 --tag rppo_bushrefuge_b02_n102
#
# Run 3: rppo_bushrefuge_b03_n113 — node 113, cuda:0 — 10x10 random-init + all-combined pressure
# LAUNCHED 2026-08-04, PID 1315, launcher log logs/20260804_040843.log
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge/03-random_init_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_bushrefuge --wandb-job-type prod \
#   --wandb-name rppo_bushrefuge_b03_n113 --tag rppo_bushrefuge_b03_n113
#
# Run 4: rppo_bushrefuge_b04_n113 — node 113, cuda:1 — 10x10 jump/pounce (attack_range [2,3], 50% hit)
# LAUNCHED 2026-08-04, PID 1692, launcher log logs/20260804_041012.log
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group rppo_bushrefuge --wandb-job-type prod \
#   --wandb-name rppo_bushrefuge_b04_n113 --tag rppo_bushrefuge_b04_n113

# ---------------------------------------------------------------------------
# rppo_restprem — 10-arm REST-PREMIUM sweep — 2026-08-10
# ---------------------------------------------------------------------------
# WHAT THIS TESTS (plain language): in this project injury heals ONLY when the agent
# takes the Rest action, so an injured agent freezes and heals wherever it happens to
# be standing — it does NOT travel to cover first. Measured on the bush-refuge runs:
# while injured the agent rests ~86% of the time but sits in a bush only 1-3% of it.
# This sweep asks whether making an UNINTERRUPTED rest streak valuable is enough to
# push an injured agent to walk to the refuge bush (which predators cannot enter) and
# complete the heal somewhere safe — i.e. whether injury can be made to INCREASE cover
# use instead of suppressing it.
#
# THE SINGLE VARIABLE is the "streak premium": how much more healing you get on the
# 10th consecutive Rest than on the 1st Rest after an interruption.
#   recovery = recovery_base_rate * (1 + recovery_accel_rate)^(rest_streak - 1)
# The 10 arms sweep that premium log-spaced from 1x (flat, no continuity incentive)
# to 129962x. recovery_base_rate is SOLVED per arm so every arm still sheds injury 70
# in ~13-15 rest steps — the injured window is matched, only the continuity gradient
# differs. a01 = the zero-premium anchor; a03 (38x) reproduces the current default.
#
#   arm  premium      base        accel
#   a01  1x           5.0         0.0
#   a02  ~2x          ...         ...
#   a03  38x          0.12        0.5     <- current/default regime
#   a04-a08  (log-spaced between)
#   a09  19683x       2.9e-05     2.0     <- WATCH: very small base rate
#   a10  129962x      2.1e-06     2.7     <- WATCH: very small base rate
#
# Base task: configs/environment/experiment/basic_bushrefuge/04-jump_attack_10x10
# (bush blocks_animals, jump/pounce predator attack_range [2,3]) — each arm `extends:`
# it and overrides ONLY body.recovery_base_rate + body.recovery_accel_rate.
# Configs committed at 5e170a1; the agent did not modify them.
#
# Plain rPPO (unmodulated), single-config from scratch (standalone, NOT continual).
# Flags deliberately IDENTICAL to the 2026-08-04 rppo_bushrefuge batch so the
# restpremium arms are directly comparable to that ladder:
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 --log-interval 50
#   --eval-config configs/evaluation/experiment_on.yaml   (bush_dwell / survival probe)
#   seed: config default 42 (NOT overridden on the CLI)
#
# Deviations flagged to the user pre-launch (inherited from the bushrefuge precedent):
#   - --checkpoint-frequency 100000 overrides the rPPO config-owned 200000
#     (configs/train/recurrent_ppo.yaml) — intentional, for cross-batch comparability.
#   - --num-envs 128 is redundant (config-owned value is already 128) but passed
#     explicitly to mirror the bushrefuge batch's command line exactly.
#
# Pre-flight 2026-08-10: nodes 106/107/108/110 NAS-mounted (nas01, 68T free), all 8
# RTX 3090 GPUs idle (<=58 MiB, no compute procs), JAX 0.9.0.1 GPU-compile check passed
# on all four. sensory.decay_power resolves to 1.0 = registry canonical
# (docs/environment/CONFIG_CRITICAL_SETTINGS.md).
# NODE 109 FAILED PRE-FLIGHT: reachable, but nas01 is NOT mounted (only nas02 + nas03;
# /media/nas01 empty). Arms a07 + a08 were therefore NOT launched — see block below.
# wandb-group: rppo_restprem, job-type: prod
# CIFS-bypass: launched via /tmp scripts — this file is the audit record.
# ---------------------------------------------------------------------------
# Arm a01: rppo_restprem_a01_n106 — node 106, cuda:0 — premium 1x (base 5.0, accel 0.0)
# LAUNCHED 2026-08-10, PID 3906803, launcher log logs/20260810_185746.log, WandB szje7o9w
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium/04-restprem_a01.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_restprem --wandb-job-type prod \
#   --wandb-name rppo_restprem_a01_n106 --tag rppo_restprem_a01_n106
#
# Arm a02: rppo_restprem_a02_n106 — node 106, cuda:1
# LAUNCHED 2026-08-10, PID 3906972, launcher log logs/20260810_185754.log (SHARED/garbled — see note), WandB 96xmquu3
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium/04-restprem_a02.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group rppo_restprem --wandb-job-type prod \
#   --wandb-name rppo_restprem_a02_n106 --tag rppo_restprem_a02_n106
#
# Arm a03: rppo_restprem_a03_n107 — node 107, cuda:0 — 38x, reproduces current default
# LAUNCHED 2026-08-10, PID 3560982, launcher log logs/20260810_185754.log (SHARED/garbled), WandB 2na5mqbl
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium/04-restprem_a03.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_restprem --wandb-job-type prod \
#   --wandb-name rppo_restprem_a03_n107 --tag rppo_restprem_a03_n107
#
# Arm a04: rppo_restprem_a04_n107 — node 107, cuda:1
# LAUNCHED 2026-08-10, PID 3561022, launcher log logs/20260810_185754.log (SHARED/garbled), WandB idv8vkjl
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium/04-restprem_a04.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group rppo_restprem --wandb-job-type prod \
#   --wandb-name rppo_restprem_a04_n107 --tag rppo_restprem_a04_n107
#
# Arm a05: rppo_restprem_a05_n108 — node 108, cuda:0
# LAUNCHED 2026-08-10, PID 1521914, launcher log logs/20260810_185754.log (SHARED/garbled), WandB f6u0z3mo
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium/04-restprem_a05.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_restprem --wandb-job-type prod \
#   --wandb-name rppo_restprem_a05_n108 --tag rppo_restprem_a05_n108
#
# Arm a06: rppo_restprem_a06_n108 — node 108, cuda:1
# LAUNCHED 2026-08-10, PID 1521919, launcher log logs/20260810_185754.log (SHARED/garbled), WandB m7pm9xqm
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium/04-restprem_a06.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group rppo_restprem --wandb-job-type prod \
#   --wandb-name rppo_restprem_a06_n108 --tag rppo_restprem_a06_n108
#
# Arms a07 + a08 — REASSIGNED node 109 -> node 111 by the user, 2026-08-10.
# The original 109 assignment was BLOCKED: node 109 has no nas01 CIFS mount (only
# nas02 + nas03 present; /media/nas01 is an empty directory), so the project tree is
# unreachable there. Node 111 re-verified at reassignment time: nas01 mounted (67T
# free), both RTX 3090s idle (28 / 196 MiB, 0% util), no train.py running.
# Tags renamed _n109 -> _n111 to reflect the actual node.
# Launched SEQUENTIALLY with a ~60 s gap: the first 8 arms ran 8 cold JIT compiles
# concurrently over CIFS, which is why they took ~45 min to reach the first step;
# staggering avoids compounding that. Each got an EXPLICIT --log path via
# run_command.py --log, so neither collides with the other's launcher log.
#
# Arm a07: rppo_restprem_a07_n111 — node 111, cuda:0
# LAUNCHED 2026-08-10 20:24, PID 1971, launcher log logs/20260810_restprem_a07_n111.log, WandB 70erj7yy
# (commented 2026-08-16: still running on 111:1 lineage; superseded as the live block
#  by the NO-HIDING-PREDATOR batch below)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium/04-restprem_a07.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_restprem --wandb-job-type prod \
#   --wandb-name rppo_restprem_a07_n111 --tag rppo_restprem_a07_n111
#
# Arm a08: rppo_restprem_a08_n111 — node 111, cuda:1
# LAUNCHED 2026-08-10 20:25 (~60 s after a07, staggered), PID 2187,
# launcher log logs/20260810_restprem_a08_n111.log, WandB wezpfd69
# (commented 2026-08-16 — see note on a07 above)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium/04-restprem_a08.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group rppo_restprem --wandb-job-type prod \
#   --wandb-name rppo_restprem_a08_n111 --tag rppo_restprem_a08_n111
#
# Arm a09: rppo_restprem_a09_n110 — node 110, cuda:0 — 19683x (base 2.9e-05) WATCH ITEM
# LAUNCHED 2026-08-10, PID 1017751, launcher log logs/20260810_185754.log (SHARED/garbled), WandB evrn1amy
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium/04-restprem_a09.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_restprem --wandb-job-type prod \
#   --wandb-name rppo_restprem_a09_n110 --tag rppo_restprem_a09_n110
#
# Arm a10: rppo_restprem_a10_n110 — node 110, cuda:1 — 129962x (base 2.1e-06) WATCH ITEM
# LAUNCHED 2026-08-10, PID 1017791, launcher log logs/20260810_185755.log, WandB vauz72ni
#
# LAUNCH-TIME NOTES (2026-08-10):
#  - LAUNCHER-LOG COLLISION: run_command.py names the log logs/<UTC-second>.log on the
#    SHARED NAS. Six arms (a02..a06, a09) launched inside the same second and therefore
#    all redirect into logs/20260810_185754.log, whose contents are interleaved/garbled.
#    Training is unaffected (each run has its own WandB run + results dir), but that
#    launcher log is not readable. Pass --log explicitly on future batch launches.
#  - --log-interval 50 is IGNORED by these configs (they use the two-level `logging:`
#    block: logging.episode.interval_episodes=4000, logging.step.interval_iters=50).
#    Kept on the command line only to mirror the bushrefuge batch byte-for-byte.
#  - SLOW STARTUP: with experiment.during_training enabled, startup walks the whole
#    results/eval tree over CIFS before the first GPU step. At T+23 min all 8 procs were
#    alive with CPU time climbing and the directory walk visibly advancing, but GPU util
#    was still 0%. Expected-slow, not a hang.
# (commented 2026-08-10: the live block is now the a07/a08 node-111 pair above)
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium/04-restprem_a10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group rppo_restprem --wandb-job-type prod \
#   --wandb-name rppo_restprem_a10_n110 --tag rppo_restprem_a10_n110
#
# ===========================================================================
# NO-HIDING-PREDATOR rest-premium sweep (10 arms) — LAUNCHED 2026-08-16
# ---------------------------------------------------------------------------
# Replicate of the 2026-08-10 rest-premium sweep with the 2-12 ambush
# `hiding_predator` resources REMOVED (resources restated food-only). Tests
# whether "moving is dangerous" is what stopped injured agents from travelling
# to the refuge bush. WandB group: rppo_restpremNH (new group).
#
# Configs: configs/environment/experiment/basic_bushrefuge_restpremium_nohide/
#          committed cfc0293, verified food-only; recovery curves, bush-refuge
#          blocks_animals and jump/pounce all inherited from the parent arms.
#
# LAUNCH-TIME NOTES (2026-08-16):
#  - EXPLICIT --log PER RUN this time: logs/20260816_restpremNH_<arm>_n<node>.log.
#    The 2026-08-10 batch collided six runs into logs/20260810_185754.log because
#    run_command.py defaults to logs/<UTC-second>.log on the shared NAS.
#  - STAGGERED ~60 s apart. Eight concurrent cold JIT compiles took ~54 min on
#    2026-08-10; staggering cut it to ~44 min.
#  - GPUs 108:1 and 111:1 DELIBERATELY EXCLUDED — the 2026-08-10 arms a06/a08 were
#    still finishing there (94.9M / 98.9M of 100M) at launch time.
#  - DEVIATION FROM CONFIG-OWNS-VALUES: --num-envs 128 and --checkpoint-frequency
#    100000 are passed explicitly to mirror the parent sweep byte-for-byte so the
#    two sweeps stay comparable. Flagged to the user at launch.
#  - --log-interval 50 is IGNORED by these configs (they use the two-level
#    `logging:` block). Kept only to mirror the parent batch byte-for-byte.
#  - Launched via the CIFS-bypass pattern: each block was mirrored to a unique
#    /tmp script on its target node and run through run_command.py --no-tail.
# ===========================================================================
#
# Arm a01: rppo_restpremNH_a01_n106 — node 106, cuda:0
# LAUNCHED 2026-08-16, PID 1550979, launcher log logs/20260816_restpremNH_a01_n106.log, WandB oloh6yt3
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium_nohide/04-restprem_nohide_a01.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_restpremNH --wandb-job-type prod \
#   --wandb-name rppo_restpremNH_a01_n106 --tag rppo_restpremNH_a01_n106
#
# Arm a02: rppo_restpremNH_a02_n106 — node 106, cuda:1
# LAUNCHED 2026-08-16, PID 1551189, launcher log logs/20260816_restpremNH_a02_n106.log, WandB o2ze4gji
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium_nohide/04-restprem_nohide_a02.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group rppo_restpremNH --wandb-job-type prod \
#   --wandb-name rppo_restpremNH_a02_n106 --tag rppo_restpremNH_a02_n106
#
# Arm a03: rppo_restpremNH_a03_n107 — node 107, cuda:0
# LAUNCHED 2026-08-16, PID 1204293, launcher log logs/20260816_restpremNH_a03_n107.log, WandB ob3rkm8u
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium_nohide/04-restprem_nohide_a03.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_restpremNH --wandb-job-type prod \
#   --wandb-name rppo_restpremNH_a03_n107 --tag rppo_restpremNH_a03_n107
#
# Arm a04: rppo_restpremNH_a04_n107 — node 107, cuda:1
# LAUNCHED 2026-08-16, PID 1204507, launcher log logs/20260816_restpremNH_a04_n107.log, WandB vk5upgay
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium_nohide/04-restprem_nohide_a04.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group rppo_restpremNH --wandb-job-type prod \
#   --wandb-name rppo_restpremNH_a04_n107 --tag rppo_restpremNH_a04_n107
#
# Arm a05: rppo_restpremNH_a05_n108 — node 108, cuda:0
# LAUNCHED 2026-08-16, PID 3161807, launcher log logs/20260816_restpremNH_a05_n108.log, WandB 1atfgt5p
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium_nohide/04-restprem_nohide_a05.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_restpremNH --wandb-job-type prod \
#   --wandb-name rppo_restpremNH_a05_n108 --tag rppo_restpremNH_a05_n108
#
# Arm a06: rppo_restpremNH_a06_n110 — node 110, cuda:0
# LAUNCHED 2026-08-16, PID 2838529, launcher log logs/20260816_restpremNH_a06_n110.log, WandB 7jgvntqe
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium_nohide/04-restprem_nohide_a06.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_restpremNH --wandb-job-type prod \
#   --wandb-name rppo_restpremNH_a06_n110 --tag rppo_restpremNH_a06_n110
#
# Arm a07: rppo_restpremNH_a07_n110 — node 110, cuda:1
# LAUNCHED 2026-08-16, PID 2838740, launcher log logs/20260816_restpremNH_a07_n110.log, WandB 45jsidrg
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium_nohide/04-restprem_nohide_a07.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group rppo_restpremNH --wandb-job-type prod \
#   --wandb-name rppo_restpremNH_a07_n110 --tag rppo_restpremNH_a07_n110
#
# Arm a08: rppo_restpremNH_a08_n111 — node 111, cuda:0
# LAUNCHED 2026-08-16, PID 1800371, launcher log logs/20260816_restpremNH_a08_n111.log, WandB edlxgeum
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium_nohide/04-restprem_nohide_a08.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_restpremNH --wandb-job-type prod \
#   --wandb-name rppo_restpremNH_a08_n111 --tag rppo_restpremNH_a08_n111
#
# Arm a09: rppo_restpremNH_a09_n112 — node 112, cuda:0 — recovery_base_rate 2.9e-05 WATCH ITEM
# LAUNCHED 2026-08-16, PID 532510, launcher log logs/20260816_restpremNH_a09_n112.log, WandB neta4235
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium_nohide/04-restprem_nohide_a09.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:0 --log-interval 50 \
#   --wandb-group rppo_restpremNH --wandb-job-type prod \
#   --wandb-name rppo_restpremNH_a09_n112 --tag rppo_restpremNH_a09_n112
#
# Arm a10: rppo_restpremNH_a10_n112 — node 112, cuda:1 — recovery_base_rate 2.1e-06 WATCH ITEM
# LAUNCHED 2026-08-16, PID 532722, launcher log logs/20260816_restpremNH_a10_n112.log, WandB ff0r7qrs
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic_bushrefuge_restpremium_nohide/04-restprem_nohide_a10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo.yaml \
#   --eval-config configs/evaluation/experiment_on.yaml \
#   --num-envs 128 --episodes 100000000 --checkpoint-frequency 100000 \
#   --device cuda:1 --log-interval 50 \
#   --wandb-group rppo_restpremNH --wandb-job-type prod \
#   --wandb-name rppo_restpremNH_a10_n112 --tag rppo_restpremNH_a10_n112
#
# ---------------------------------------------------------------------------
# POST-LAUNCH FINDING (2026-08-16, T+1h40m) — STARTUP EVAL-TREE WALK IS THE
# DOMINANT STARTUP COST AND IS GROWING RUN-OVER-RUN.
#
# All 10 arms launched cleanly (1 PID each, distinct WandB run, no errors), but
# at T+1h40m NONE had reached the first training step; GPU util 0% on all ten
# with only a ~280 MiB CUDA context allocated. The processes are NOT hung:
# CPU time climbs steadily (a01 05:29 -> 17:18) and `wchan` is `wait_for_response`
# (CIFS network wait).
#
# Root cause: with experiment.during_training enabled, startup walks
# results/eval/ before the first GPU step. That tree now contains
#   results/eval/avoidance/metrics_history_rppo_gae/_scratch/{b03_gae,b04_gae}/
# = 24 conditions x ~723 checkpoint dirs, each with nested
# <ckpt>/models/<ckpt>/episodes/ levels -> O(1e5) directory entries to stat over
# CIFS, from an UNRELATED July-22 experiment. Sampling /proc/<pid>/fd confirmed
# the walk advancing (avoid_pred_inj70 -> avoid_rabbit_inj70 -> b03_gae/...).
# Ten concurrent runs all walking the same CIFS tree compounds it.
#
# This is why 2026-08-10 took ~54 min and 2026-08-16 exceeds 1h40m: the cost
# scales with accumulated eval scratch output, not with the run itself.
#
# NOT actioned here (out of training-runner scope, and results/ is gitignored
# data that must not be casually deleted). Surfaced to the user for a decision:
# prune/archive the _scratch tree, or bound the startup walk in code.
# ---------------------------------------------------------------------------

# ===========================================================================
# return_mode comparison (MC / MC_FIXED / GAE) — basic/04, 5 seeds each — 2026-09-03
# ===========================================================================
# 15 plain-RecurrentPPO runs measuring, in SURVIVAL STEPS, whether PPO's mainstream
# return-normalisation convention beats this project's historical one on the jump level.
#
#   MC       (configs/models/recurrent_ppo/recurrent_ppo_cmp_mc.yaml)
#            Monte-Carlo returns z-scored, used as BOTH critic target and advantage.
#            The project's historical mode; byte-identical to prior behaviour.
#   MC_FIXED (configs/models/recurrent_ppo/recurrent_ppo_cmp_mcfixed.yaml)
#            SAME MC returns, used RAW as the critic target, ADVANTAGES normalised
#            instead — the convention 9/9 surveyed mainstream PPO libraries use.
#            NEW code path, added today in af4047ac + 98b94d99.
#   GAE      (configs/models/recurrent_ppo/recurrent_ppo_cmp_gae.yaml)
#            GAE(lambda), raw critic target, normalised advantages. Unchanged.
#
# The three agent configs differ from each other ONLY in `return_mode` (verified by
# comment-stripped diff), and the MC arm is identical to recurrent_ppo.yaml.
# modulation.type: null in all three — a modulator would confound an estimator comparison.
# Env for all 15: basic/04-jump_attack_10x10 as it currently resolves, no overrides.
# 10x10 grid, obs_dim 27, action_dim 6, max_steps 500 (confirmed at smoke-test startup).
#
# wandb-group: return_mode_cmp, job-type: prod
#
# ---------------------------------------------------------------------------
# CLI-flag deviations from the config-owns-values convention (flagged to the user)
# ---------------------------------------------------------------------------
#   --num-envs 128            NOT a deviation. configs/train/recurrent_ppo.yaml already
#                             sets num_envs: 128; the flag is redundant but agrees.
#   --checkpoint-frequency 50000  DEVIATION. rPPO's config layer owns 200000. Caller asked
#                             for 50000 => 4x more checkpoints + eval videos per run.
#   --seed 42..46             DEVIATION by design. Config owns seed: 42; this is the
#                             5-seed sweep that the experiment is built on.
#   --log-interval 50         NO-OP. train.py:731 prints
#                             "[WARN] --log-interval is IGNORED: this config uses the
#                             two-level `logging:` block". The effective value comes from
#                             configs/train/recurrent_ppo.yaml logging.step.interval_iters,
#                             which is ALREADY 50 — so the requested cadence is what runs.
#                             Kept on the CLI for parity with the caller's spec.
#
# ---------------------------------------------------------------------------
# Smoke test (MC_FIXED only — the one untested-in-training code path)
# ---------------------------------------------------------------------------
# Ran to completion on node 104 cuda:1 (RTX 2080 Ti, 11 GB — the tightest-memory card
# in this batch) at the full --num-envs 128, --no-wandb, --results-dir tmp/... :
#   "RNN Type: GRU, Activation: relu, Return Mode: MC_FIXED" / "Neuromodulation: DISABLED"
#   JIT compiled, 163 iterations, 100,032 episodes, ~20-39 it/s, no OOM,
#   value loss fell monotonically 1442.9 -> ~109 (raw-return units, as MC_FIXED expects),
#   no NaN, 2 checkpoints + 2 eval videos rendered, clean "Training complete".
# Debris deleted afterwards. MC and GAE need no smoke test (unchanged code).
#
# ---------------------------------------------------------------------------
# Launched via CIFS-bypass /tmp scripts, strictly ONE AT A TIME (run_command.py is not
# parallel-safe: concurrent calls share an SSH control socket and can return the wrong
# node's PIDs). This file is the audit record; the 15 blocks below are the exact
# command lines, kept commented because only one block can ever be active.
# ---------------------------------------------------------------------------

# Run 01: rppo_cmp_mc_s42 — node 101, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mc.yaml \
#   --episodes 1000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 50000 \
#   --log-interval 50 \
#   --seed 42 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp --wandb-job-type prod \
#   --wandb-name rppo_cmp_mc_s42 --tag rppo_cmp_mc_s42

# Run 02: rppo_cmp_mc_s43 — node 101, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mc.yaml \
#   --episodes 1000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 50000 \
#   --log-interval 50 \
#   --seed 43 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp --wandb-job-type prod \
#   --wandb-name rppo_cmp_mc_s43 --tag rppo_cmp_mc_s43

# Run 03: rppo_cmp_mc_s44 — node 103, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mc.yaml \
#   --episodes 1000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 50000 \
#   --log-interval 50 \
#   --seed 44 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp --wandb-job-type prod \
#   --wandb-name rppo_cmp_mc_s44 --tag rppo_cmp_mc_s44

# Run 04: rppo_cmp_mc_s45 — node 103, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mc.yaml \
#   --episodes 1000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 50000 \
#   --log-interval 50 \
#   --seed 45 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp --wandb-job-type prod \
#   --wandb-name rppo_cmp_mc_s45 --tag rppo_cmp_mc_s45

# Run 05: rppo_cmp_mc_s46 — node 104, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mc.yaml \
#   --episodes 1000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 50000 \
#   --log-interval 50 \
#   --seed 46 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp --wandb-job-type prod \
#   --wandb-name rppo_cmp_mc_s46 --tag rppo_cmp_mc_s46

# Run 06: rppo_cmp_mcfixed_s42 — node 104, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mcfixed.yaml \
#   --episodes 1000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 50000 \
#   --log-interval 50 \
#   --seed 42 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp --wandb-job-type prod \
#   --wandb-name rppo_cmp_mcfixed_s42 --tag rppo_cmp_mcfixed_s42

# Run 07: rppo_cmp_mcfixed_s43 — node 105, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mcfixed.yaml \
#   --episodes 1000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 50000 \
#   --log-interval 50 \
#   --seed 43 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp --wandb-job-type prod \
#   --wandb-name rppo_cmp_mcfixed_s43 --tag rppo_cmp_mcfixed_s43

# Run 08: rppo_cmp_mcfixed_s44 — node 105, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mcfixed.yaml \
#   --episodes 1000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 50000 \
#   --log-interval 50 \
#   --seed 44 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp --wandb-job-type prod \
#   --wandb-name rppo_cmp_mcfixed_s44 --tag rppo_cmp_mcfixed_s44

# Run 09: rppo_cmp_mcfixed_s45 — node 106, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mcfixed.yaml \
#   --episodes 1000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 50000 \
#   --log-interval 50 \
#   --seed 45 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp --wandb-job-type prod \
#   --wandb-name rppo_cmp_mcfixed_s45 --tag rppo_cmp_mcfixed_s45

# Run 10: rppo_cmp_mcfixed_s46 — node 106, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mcfixed.yaml \
#   --episodes 1000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 50000 \
#   --log-interval 50 \
#   --seed 46 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp --wandb-job-type prod \
#   --wandb-name rppo_cmp_mcfixed_s46 --tag rppo_cmp_mcfixed_s46

# Run 11: rppo_cmp_gae_s42 — node 107, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_gae.yaml \
#   --episodes 1000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 50000 \
#   --log-interval 50 \
#   --seed 42 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp --wandb-job-type prod \
#   --wandb-name rppo_cmp_gae_s42 --tag rppo_cmp_gae_s42

# Run 12: rppo_cmp_gae_s43 — node 107, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_gae.yaml \
#   --episodes 1000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 50000 \
#   --log-interval 50 \
#   --seed 43 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp --wandb-job-type prod \
#   --wandb-name rppo_cmp_gae_s43 --tag rppo_cmp_gae_s43

# Run 13: rppo_cmp_gae_s44 — node 108, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_gae.yaml \
#   --episodes 1000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 50000 \
#   --log-interval 50 \
#   --seed 44 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp --wandb-job-type prod \
#   --wandb-name rppo_cmp_gae_s44 --tag rppo_cmp_gae_s44

# Run 14: rppo_cmp_gae_s45 — node 108, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_gae.yaml \
#   --episodes 1000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 50000 \
#   --log-interval 50 \
#   --seed 45 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp --wandb-job-type prod \
#   --wandb-name rppo_cmp_gae_s45 --tag rppo_cmp_gae_s45

# Run 15: rppo_cmp_gae_s46 — node 109, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_gae.yaml \
#   --episodes 1000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 50000 \
#   --log-interval 50 \
#   --seed 46 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp --wandb-job-type prod \
#   --wandb-name rppo_cmp_gae_s46 --tag rppo_cmp_gae_s46

# ---------------------------------------------------------------------------
# LAUNCH RECORD — all 15 verified live, exactly one PID per tag, 2026-09-03T16:48:20
# Run  Tag                       Arm       Seed  Node:GPU  PID      WandB     Log
# 1    rppo_cmp_mc_s42           mc        42    101:0     196840   rk3huayr  logs/20260903_163031.log
# 2    rppo_cmp_mc_s43           mc        43    101:1     197637   x2jxkpq8  logs/20260903_163215.log
# 3    rppo_cmp_mc_s44           mc        44    103:0     197326   4dnvtpqf  logs/20260903_163316.log
# 4    rppo_cmp_mc_s45           mc        45    103:1     197912   9kjjj5t8  logs/20260903_163418.log
# 5    rppo_cmp_mc_s46           mc        46    104:0     201684   mx8orfd1  logs/20260903_163519.log
# 6    rppo_cmp_mcfixed_s42      mcfixed   42    104:1     202279   k7tu4ch0  logs/20260903_163621.log
# 7    rppo_cmp_mcfixed_s43      mcfixed   43    105:0     199549   tifzlhcq  logs/20260903_163722.log
# 8    rppo_cmp_mcfixed_s44      mcfixed   44    105:1     200169   0t83gfxv  logs/20260903_163824.log
# 9    rppo_cmp_mcfixed_s45      mcfixed   45    106:0     4406     x6me4z6a  logs/20260903_163925.log
# 10   rppo_cmp_mcfixed_s46      mcfixed   46    106:1     5704     kqjxkgwu  logs/20260903_164027.log
# 11   rppo_cmp_gae_s42          gae       42    107:0     13718    g20s999t  logs/20260903_164128.log
# 12   rppo_cmp_gae_s43          gae       43    107:1     14286    x36zqm96  logs/20260903_164229.log
# 13   rppo_cmp_gae_s44          gae       44    108:0     4808     s1lpg7o8  logs/20260903_164330.log
# 14   rppo_cmp_gae_s45          gae       45    108:1     5428     pwn54tw7  logs/20260903_164431.log
# 15   rppo_cmp_gae_s46          gae       46    109:0     20183    0ghnqaly  logs/20260903_164534.log
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# return_mode_cmp_10m — MC / MC_FIXED / GAE return-estimator comparison — 2026-09-04
# 10M-EPISODE RE-RUN of the 15-run 1M comparison launched 2026-09-03 (block above).
# The 1M results are KEPT; these runs use DISTINCT tags carrying a `10m` marker
# (rppo_cmp10m_*) so nothing overwrites results/JAX_RecurrentPPO/*_rppo_cmp_*.
#
# 3 arms x 5 seeds (42-46). Env identical for all 15:
#   configs/environment/experiment/basic/04-jump_attack_10x10.yaml (as it resolves;
#   no overrides). decay_power resolves to the canonical 1.0 from environment/default.
# The three agent configs are byte-identical except for one line, `return_mode`:
#   recurrent_ppo_cmp_mc.yaml       return_mode "MC"       (z-scored MC as target AND advantage)
#   recurrent_ppo_cmp_mcfixed.yaml  return_mode "MC_FIXED" (raw MC target, normalised advantages)
#   recurrent_ppo_cmp_gae.yaml      return_mode "GAE"      (GAE(lambda), raw target, normalised adv)
# All three: algorithm RecurrentPPO, modulation.type null (a modulator would confound
# an estimator comparison).
#
# TWO DELIBERATE CHANGES FROM THE 1M LAUNCH — both to match the project's standard 10M
# baseline (the 14-arm sensor-ladder study) rather than scaling the short-run settings up:
#   --episodes 10000000        (was 1000000)  — the sensor-ladder budget, so these results
#                              sit on comparable footing with that study.
#   --checkpoint-frequency 200000 (was 50000) — at 10M this gives 50 checkpoints/run, the
#                              sensor-ladder cadence. Keeping 50000 would have produced 200
#                              checkpoints AND 200 eval-video renders per run, 3000 across
#                              the batch.
#
# CONFIG-OWNED-VALUES NOTE: --num-envs 128 and --checkpoint-frequency 200000 exactly match
# the values configs/train/recurrent_ppo.yaml already carries, so they are redundant rather
# than deviating. --seed is a genuine, intended deviation: this is a 5-seed sweep and each
# row needs its own seed. --log-interval 50 is a knowing no-op on this config (the two-level
# `logging:` block governs and already uses 50) — passed for command-line consistency; the
# resulting [WARN] is expected and not a problem.
#
# PRE-FLIGHT (2026-09-04, all 8 nodes): nas01 CIFS mounted (192T, 41T free) on every node;
# every assigned GPU idle (no compute processes); JAX GPU-compile check passed on all 8
# (jax 0.9.0.1, real 4x4 matmul JIT on platform=gpu, version-matched across the cluster);
# zero train.py processes cluster-wide; no results dir matching *cmp10m* pre-existed.
# No smoke test: all three code paths incl. MC_FIXED ran 1M episodes to completion on these
# same configs and nodes hours earlier.
#
# wandb-group: return_mode_cmp_10m, job-type: prod
# run_command.py is NOT parallel-safe (shared SSH control socket can return another node's
# PIDs) -> launched STRICTLY ONE AT A TIME, each verified by pgrep before the next starts.
# CIFS-bypass: launched via per-run /tmp scripts — this file is the audit record.
# ---------------------------------------------------------------------------

# Run 1: rppo_cmp10m_mc_s42 — node 101, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mc.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 42 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_mc_s42 --tag rppo_cmp10m_mc_s42

# Run 2: rppo_cmp10m_mc_s43 — node 101, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mc.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 43 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_mc_s43 --tag rppo_cmp10m_mc_s43

# Run 3: rppo_cmp10m_mc_s44 — node 103, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mc.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 44 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_mc_s44 --tag rppo_cmp10m_mc_s44

# Run 4: rppo_cmp10m_mc_s45 — node 103, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mc.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 45 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_mc_s45 --tag rppo_cmp10m_mc_s45

# Run 5: rppo_cmp10m_mc_s46 — node 104, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mc.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 46 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_mc_s46 --tag rppo_cmp10m_mc_s46

# Run 6: rppo_cmp10m_mcfixed_s42 — node 104, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mcfixed.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 42 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_mcfixed_s42 --tag rppo_cmp10m_mcfixed_s42

# Run 7: rppo_cmp10m_mcfixed_s43 — node 105, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mcfixed.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 43 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_mcfixed_s43 --tag rppo_cmp10m_mcfixed_s43

# Run 8: rppo_cmp10m_mcfixed_s44 — node 105, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mcfixed.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 44 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_mcfixed_s44 --tag rppo_cmp10m_mcfixed_s44

# Run 9: rppo_cmp10m_mcfixed_s45 — node 106, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mcfixed.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 45 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_mcfixed_s45 --tag rppo_cmp10m_mcfixed_s45

# Run 10: rppo_cmp10m_mcfixed_s46 — node 106, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mcfixed.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 46 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_mcfixed_s46 --tag rppo_cmp10m_mcfixed_s46

# Run 11: rppo_cmp10m_gae_s42 — node 107, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_gae.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 42 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_gae_s42 --tag rppo_cmp10m_gae_s42

# Run 12: rppo_cmp10m_gae_s43 — node 107, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_gae.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 43 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_gae_s43 --tag rppo_cmp10m_gae_s43

# Run 13: rppo_cmp10m_gae_s44 — node 108, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_gae.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 44 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_gae_s44 --tag rppo_cmp10m_gae_s44

# Run 14: rppo_cmp10m_gae_s45 — node 108, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_gae.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 45 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_gae_s45 --tag rppo_cmp10m_gae_s45

# Run 15: rppo_cmp10m_gae_s46 — node 109, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_gae.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 46 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_gae_s46 --tag rppo_cmp10m_gae_s46

# ---------------------------------------------------------------------------
# LAUNCH RECORD — all 15 verified live, exactly one PID per tag, 2026-09-04T17:54:13
# Cluster-wide sweep confirmed exactly 15 'cmp10m' processes (2 per node except 109:1)
# and ZERO surviving 1M-set processes. All 15 1M result dirs (*_rppo_cmp_*) intact.
# Run  Tag                        Arm       Seed  Node:GPU  PID      WandB     Log
# 1    rppo_cmp10m_mc_s42         mc        42    101:0     271088   xy7nic92  logs/20260904_173800.log
# 2    rppo_cmp10m_mc_s43         mc        43    101:1     271676   cmqugy51  logs/20260904_173901.log
# 3    rppo_cmp10m_mc_s44         mc        44    103:0     269196   quydmpd7  logs/20260904_174002.log
# 4    rppo_cmp10m_mc_s45         mc        45    103:1     269769   ymbhe3qp  logs/20260904_174104.log
# 5    rppo_cmp10m_mc_s46         mc        46    104:0     275814   uqnl1scm  logs/20260904_174205.log
# 6    rppo_cmp10m_mcfixed_s42    mcfixed   42    104:1     276428   se72bm9i  logs/20260904_174306.log
# 7    rppo_cmp10m_mcfixed_s43    mcfixed   43    105:0     273660   ukte3hbu  logs/20260904_174408.log
# 8    rppo_cmp10m_mcfixed_s44    mcfixed   44    105:1     274271   ef2pg37t  logs/20260904_174509.log
# 9    rppo_cmp10m_mcfixed_s45    mcfixed   45    106:0     79245    b09iehij  logs/20260904_174610.log
# 10   rppo_cmp10m_mcfixed_s46    mcfixed   46    106:1     80124    vpabnrrz  logs/20260904_174711.log
# 11   rppo_cmp10m_gae_s42        gae       42    107:0     87229    4sqc0lsd  logs/20260904_174813.log
# 12   rppo_cmp10m_gae_s43        gae       43    107:1     88075    haw7hl3e  logs/20260904_174914.log
# 13   rppo_cmp10m_gae_s44        gae       44    108:0     79124    whdabu5w  logs/20260904_175015.log
# 14   rppo_cmp10m_gae_s45        gae       45    108:1     79970    4ug6okvu  logs/20260904_175116.log
# 15   rppo_cmp10m_gae_s46        gae       46    109:0     59573    96ej3ngm  logs/20260904_175218.log
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# return_mode_cmp_10m — FOURTH ARM: GAE_NORM — 2026-09-04
# Adds the missing cell of the 2x2 (estimator x normalisation scheme) to the
# 15-run MC / MC_FIXED / GAE batch launched at 17:54 today (block above).
#   MC        Monte-Carlo estimator, MATCHED scale   (z-scored target, raw residual adv)
#   MC_FIXED  Monte-Carlo estimator, SPLIT scale     (raw target, normalised adv)
#   GAE       GAE(lambda) estimator, SPLIT scale
#   GAE_NORM  GAE(lambda) estimator, MATCHED scale   <-- THIS ARM
# Hypothesis: MC's ~3.5x survival advantage at 1M comes from the MATCHED SCALE,
# not from the Monte-Carlo estimator. If so, GAE_NORM tracks MC; if the estimator
# is what matters, GAE_NORM tracks GAE.
#
# IDENTICAL to the 15 in-flight runs in every respect except the agent config,
# the seed, the tag and the node/GPU. Verified: the gaenorm agent config's `agent`
# block differs from recurrent_ppo_cmp_gae.yaml in exactly ONE key —
# return_mode "GAE" -> "GAE_NORM". No `extends:`; return_mode is in the trainer's
# LEGAL_RETURN_MODES and validates. modulation.type null (a modulator would confound).
# Env: basic/04-jump_attack_10x10.yaml, no overrides; decay_power resolves to the
# canonical 1.0 from environment/default.yaml (CONFIG_CRITICAL_SETTINGS registry).
#
# CONFIG-OWNED-VALUES NOTE (unchanged from the 17:54 block): --num-envs 128 and
# --checkpoint-frequency 200000 exactly match configs/train/recurrent_ppo.yaml, so they
# are redundant rather than deviating — passed for byte-parity with the other three arms.
# --seed IS a genuine intended deviation (5-seed sweep). --log-interval 50 is a knowing
# no-op on this config; the resulting [WARN] is expected.
#
# PRE-FLIGHT (2026-09-04, nodes 110/111/112): nas01 CIFS mounted (192T, 41T free) on all
# three; all five assigned GPUs FREE (RTX 3090, 0% util, <=64 MiB residual, no compute
# processes); zero train.py processes on all three nodes; JAX GPU-compile check passed on
# all three (jax 0.9.0.1, real 4x4 matmul JIT on platform=gpu — version-matched to the
# cluster and to the 15 in-flight runs); no results dir matching *gaenorm* pre-existed.
#
# wandb-group: return_mode_cmp_10m (same group as the other three arms), job-type: prod
# run_command.py is NOT parallel-safe (shared SSH control socket can return another node's
# PIDs) -> launched STRICTLY ONE AT A TIME, each verified by pgrep before the next starts.
# CIFS-bypass: launched via per-run /tmp scripts — this file is the audit record.
# ---------------------------------------------------------------------------

# Run 16: rppo_cmp10m_gaenorm_s42 — node 110, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_gaenorm.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 42 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_gaenorm_s42 --tag rppo_cmp10m_gaenorm_s42

# Run 17: rppo_cmp10m_gaenorm_s43 — node 110, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_gaenorm.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 43 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_gaenorm_s43 --tag rppo_cmp10m_gaenorm_s43

# Run 18: rppo_cmp10m_gaenorm_s44 — node 111, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_gaenorm.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 44 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_gaenorm_s44 --tag rppo_cmp10m_gaenorm_s44

# Run 19: rppo_cmp10m_gaenorm_s45 — node 111, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_gaenorm.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 45 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_gaenorm_s45 --tag rppo_cmp10m_gaenorm_s45

# Run 20: rppo_cmp10m_gaenorm_s46 — node 112, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_gaenorm.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 46 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_gaenorm_s46 --tag rppo_cmp10m_gaenorm_s46

# ---------------------------------------------------------------------------
# LAUNCH RECORD — all 5 verified live, exactly one PID per tag, 2026-09-04T19:07
# Cluster sweep of 110/111/112 found exactly 5 train.py processes (2+2+1), no duplicates.
# Ground truth: every run's saved models/config.yaml reads `return_mode: GAE_NORM` and
# `group: return_mode_cmp_10m`. All 5 observed stepping on GPU (util 89-99%, ~5.5 GB).
# Run  Tag                        Arm       Seed  Node:GPU  PID    WandB     Log
# 16   rppo_cmp10m_gaenorm_s42    gaenorm   42    110:0     4845   26mwmoc9  logs/20260904_185745.log
# 17   rppo_cmp10m_gaenorm_s43    gaenorm   43    110:1     5834   ac522oud  logs/20260904_185917.log
# 18   rppo_cmp10m_gaenorm_s44    gaenorm   44    111:0     17873  0mtwmifc  logs/20260904_190034.log
# 19   rppo_cmp10m_gaenorm_s45    gaenorm   45    111:1     18830  87a1r9ld  logs/20260904_190152.log
# 20   rppo_cmp10m_gaenorm_s46    gaenorm   46    112:0     4943   1v91itww  logs/20260904_190311.log
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# return_mode_cmp_10m — FIFTH ARM: MC_RAW — 2026-09-04
# Completes the estimator x scale-scheme comparison begun with the 15-run
# MC / MC_FIXED / GAE batch (17:54) and the 5-run GAE_NORM batch (18:57):
#   MC        Monte-Carlo estimator, MATCHED scale at ~1   (z-scored target, raw residual adv)
#   MC_FIXED  Monte-Carlo estimator, SPLIT scale           (raw target, separately z-scored adv)
#   GAE       GAE(lambda) estimator, SPLIT scale
#   GAE_NORM  GAE(lambda) estimator, MATCHED scale at ~1
#   MC_RAW    Monte-Carlo estimator, MATCHED scale at ~24  <-- THIS ARM (nothing rescaled)
# Question this arm answers: the first four arms cannot separate "critic target and
# advantage must share UNITS" from "the advantage must land near spread 1", because
# every matched-scale arm is matched AT 1. MC_RAW is matched at the LARGE raw scale.
# If matching is what matters, MC_RAW tracks MC (138.8 mean survival steps at 1M);
# if landing near spread 1 is what matters, MC_RAW tracks MC_FIXED (40.0) / GAE (41.9).
#
# IDENTICAL to the 20 sibling runs in every respect except the agent config, the seed,
# the tag and the node/GPU. Verified pre-flight: recurrent_ppo_cmp_mcraw.yaml has NO
# `extends:`, its `agent` block differs from recurrent_ppo_cmp_mcfixed.yaml in exactly
# ONE key (return_mode "MC_FIXED" -> "MC_RAW"), and MC_RAW is in the trainer's
# LEGAL_RETURN_MODES (src/models/recurrent_ppo_trainer.py:193) with a real implemented
# branch (line 444: raw MC returns as target, raw `returns - value` residual as
# advantage, no normalisation line). modulation.type null (a modulator would confound).
# Env: basic/04-jump_attack_10x10.yaml, no overrides; decay_power resolves to the
# canonical 1.0 from environment/default.yaml (CONFIG_CRITICAL_SETTINGS registry).
#
# CONFIG-OWNED-VALUES NOTE (unchanged from the 17:54 and 18:57 blocks): --num-envs 128
# and --checkpoint-frequency 200000 exactly match configs/train/recurrent_ppo.yaml, so
# they are redundant rather than deviating — passed for byte-parity with the other four
# arms. --seed IS a genuine intended deviation (5-seed sweep). --log-interval 50 is a
# knowing no-op on this config (the `logging:` block supersedes it); the [WARN] is expected.
#
# PRE-FLIGHT (2026-09-04, nodes 102/105/106/109/112): nas01 CIFS mounted (192T, 41T free)
# on all five; all five assigned GPUs FREE (102:0 RTX 4090; 105:0 RTX 2080 Ti; 106:1,
# 109:1, 112:1 RTX 3090 — 0% util, <=133 MiB residual, no compute processes); JAX
# GPU-compile check passed on all five (jax 0.9.0.1, real 4x4 matmul JIT on platform=gpu
# — version-matched to the cluster and to the 20 sibling runs); no results dir matching
# *mcraw* pre-existed.
# NOTE ON WHY 105:0 AND 106:1 ARE FREE: rppo_cmp10m_mcfixed_s43 (105:0) and
# rppo_cmp10m_mcfixed_s46 (106:1) COMPLETED their full 10M-episode budget earlier today
# ("Training complete." in logs/20260904_174408.log and logs/20260904_174711.log) — they
# did not crash. Their diary rows still read `running`.
# Deliberately spread across single free GPUs on otherwise-occupied nodes so that nodes
# 113 and 114 stay whole for heavier jobs — NOT consolidated.
#
# wandb-group: return_mode_cmp_10m (same group as the other four arms), job-type: prod
# run_command.py is NOT parallel-safe (shared SSH control socket can return another node's
# PIDs) -> launched STRICTLY ONE AT A TIME, each verified by pgrep before the next starts.
# CIFS-bypass: launched via per-run /tmp scripts — this file is the audit record.
# ---------------------------------------------------------------------------

# Run 21: rppo_cmp10m_mcraw_s42 — node 105, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mcraw.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 42 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_mcraw_s42 --tag rppo_cmp10m_mcraw_s42

# Run 22: rppo_cmp10m_mcraw_s43 — node 106, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mcraw.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 43 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_mcraw_s43 --tag rppo_cmp10m_mcraw_s43

# Run 23: rppo_cmp10m_mcraw_s44 — node 109, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mcraw.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 44 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_mcraw_s44 --tag rppo_cmp10m_mcraw_s44

# Run 24: rppo_cmp10m_mcraw_s45 — node 112, cuda:1
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mcraw.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 45 \
#   --device cuda:1 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_mcraw_s45 --tag rppo_cmp10m_mcraw_s45

# Run 25: rppo_cmp10m_mcraw_s46 — node 102, cuda:0
# /home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
#   --config configs/environment/experiment/basic/04-jump_attack_10x10.yaml \
#   --agent_config configs/models/recurrent_ppo/recurrent_ppo_cmp_mcraw.yaml \
#   --episodes 10000000 \
#   --num-envs 128 \
#   --checkpoint-frequency 200000 \
#   --log-interval 50 \
#   --seed 46 \
#   --device cuda:0 \
#   --wandb-group return_mode_cmp_10m --wandb-job-type prod \
#   --wandb-name rppo_cmp10m_mcraw_s46 --tag rppo_cmp10m_mcraw_s46

# ---------------------------------------------------------------------------
# LAUNCH RECORD — all 5 verified TRAINING, exactly one PID per tag, 2026-09-04T22:06
# Verification beyond process-existence: every run's episode counter advanced between
# two samples ~25 s apart (~400k -> ~430k of 10,000,000 episodes at ~1000-1350 it/s,
# finite losses, no NaN), and every assigned GPU is resident and busy (80-99% util,
# 4.5-5.6 GB). Node 102's loose pgrep shows 2 hits only because this Claude container
# runs ON node 102 — the exact `bin/python train.py` match returns exactly 1.
# Ground truth: every run's saved models/config.yaml reads `return_mode: MC_RAW`,
# `algorithm: RecurrentPPO`, `modulation.type: null`, `num_envs: 128`, `decay_power: 1.0`.
# KNOWN SNAPSHOT ARTIFACT (pre-existing, NOT introduced here): the saved config.yaml
# records `seed: 42` and `episodes: 100` for ALL FIVE — but so do all 20 sibling runs
# (mc/mcfixed/gae/gaenorm), which demonstrably ran to 10M episodes. The saved YAML keeps
# the pre-CLI-override values for those two fields. The per-run seed IS applied: the
# trainer's own startup banner prints seed 42/43/44/45/46 respectively, and the tqdm
# total reads 10,000,000. Verify seed/episodes from the banner or WandB, not from the
# saved config.yaml.
# Run  Tag                      Arm     Seed  Node:GPU  PID      WandB     Log
# 21   rppo_cmp10m_mcraw_s42    mcraw   42    105:0     442997   j0vuwph7  logs/20260904_215843.log
# 22   rppo_cmp10m_mcraw_s43    mcraw   43    106:1     256258   ls7riwt1  logs/20260904_215901.log
# 23   rppo_cmp10m_mcraw_s44    mcraw   44    109:1     139018   yo02nvg9  logs/20260904_215911.log
# 24   rppo_cmp10m_mcraw_s45    mcraw   45    112:1     39113    37oqpt8k  logs/20260904_215922.log
# 25   rppo_cmp10m_mcraw_s46    mcraw   46    102:0     1921862  o6yac86p  logs/20260904_215932.log
# ---------------------------------------------------------------------------
