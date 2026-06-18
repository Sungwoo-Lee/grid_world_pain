---
snapshot_label: v3_config_system_overhaul
captured: 2026-06-19 01:15
source_commit: c62334f
scope: src
graphify_version: 0.8.5
session_id: 96e71c7b-dc03-44c9-a98c-1c2acc86e0d9
---

# Code-graph snapshot — `v3_config_system_overhaul`

Captured `2026-06-19 01:15` from commit `c62334f`.

To compare with the current code state: `python scripts/regen_code_graph.py && diff src/graphify-out/GRAPH_REPORT.md docs/memory/code_snapshots/20260619_0115_v3_config_system_overhaul.md`.

To re-enter the originating conversation: `python scripts/open_conversation.py 96e71c7b-dc03-44c9-a98c-1c2acc86e0d9` (if the session was a `/memorize` invocation).

---

# Graph Report - src  (2026-06-19)

## Corpus Check
- 44 files · ~75,783 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 834 nodes · 1133 edges · 73 communities (54 shown, 19 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 156 edges (avg confidence: 0.65)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `c62334fb`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]
- [[_COMMUNITY_Community 53|Community 53]]
- [[_COMMUNITY_Community 54|Community 54]]
- [[_COMMUNITY_Community 55|Community 55]]
- [[_COMMUNITY_Community 56|Community 56]]
- [[_COMMUNITY_Community 57|Community 57]]
- [[_COMMUNITY_Community 58|Community 58]]
- [[_COMMUNITY_Community 59|Community 59]]
- [[_COMMUNITY_Community 60|Community 60]]
- [[_COMMUNITY_Community 61|Community 61]]
- [[_COMMUNITY_Community 62|Community 62]]
- [[_COMMUNITY_Community 63|Community 63]]
- [[_COMMUNITY_Community 64|Community 64]]
- [[_COMMUNITY_Community 65|Community 65]]
- [[_COMMUNITY_Community 66|Community 66]]
- [[_COMMUNITY_Community 67|Community 67]]
- [[_COMMUNITY_Community 68|Community 68]]
- [[_COMMUNITY_Community 69|Community 69]]
- [[_COMMUNITY_Community 70|Community 70]]
- [[_COMMUNITY_Community 71|Community 71]]
- [[_COMMUNITY_Community 72|Community 72]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 28 edges
2. `OneHotDist` - 23 edges
3. `DreamerNeuromodulatorRNN` - 22 edges
4. `ModulatedLayerNormGRUCell` - 18 edges
5. `get_observation()` - 14 edges
6. `SiLU` - 14 edges
7. `load_env_params()` - 13 edges
8. `SequentialReplayBuffer` - 13 edges
9. `ParallelEnv` - 12 edges
10. `render_jax_state_v2()` - 12 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `train_step()`  [INFERRED]
  algorithms/dreamer_srl/dreamer_srl_main.py → models/dreamer_v3_trainer.py
- `_DictState` --uses--> `ParallelEnv`  [INFERRED]
  utils/evaluation_core.py → environment/wrapper.py
- `ContinualSchedule` --uses--> `ParallelEnv`  [INFERRED]
  algorithms/dreamer_srl/dreamer_srl_main.py → environment/wrapper.py
- `Player` --uses--> `ParallelEnv`  [INFERRED]
  algorithms/dreamer_srl/dreamer_srl_main.py → environment/wrapper.py
- `_run_parallel_env_eval()` --calls--> `ParallelEnv`  [INFERRED]
  utils/evaluation_core.py → environment/wrapper.py

## Communities (73 total, 19 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.14
Nodes (22): _load_animals(), load_env_params(), _normalise_tag(), _one_hot_list(), _parse_noise_config(), Configuration loader for JAX Environment.  Translates YAML config files into JAX, Return a valid metric-suffix string.  Empty / missing → f'idx{idx}'.      The ta, Load behavior_measures: from the YAML config.      Returns None if the top-level (+14 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (28): ModulatedGRUCell, ModulatedGRUCell: A GRU cell with external gate-bias injection for neuromodulati, GRU cell with optional external gate-bias injection on the update gate.      The, Forward pass.          Args:             carry: Previous hidden state h_{t-1}, s, ModulatorOutput, NeuromodulatorRNN, Neuromodulatory networks with branched heads for RecurrentPPO and DreamerV3.  Im, Forward pass.          Args:             obs: Observation vector, shape (..., in (+20 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (44): DreamerModulatorOutput, Output from the DreamerV3 neuromodulator's branched heads., Compute head outputs from modulator hidden state., Observation mode: called during RSSM step() and get_action()., Imagination mode: called during RSSM imagine_step()., collect_trajectories(), compute_gae(), compute_mc_returns() (+36 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (30): BernoulliSafeMode, IndependentBernoulli, mean(), dreamer-srl loss utilities — TwoHotEncoding, BernoulliSafeMode, and reconstructi, Initialize the distribution.          Ported from sheeprl@33b6366:sheeprl/utils/, Log-probability of target x under the two-hot distribution.          Ported from, Bernoulli distribution with a safe-mode property, matching sheeprl's BernoulliSa, Initialize the BernoulliSafeMode distribution.          Ported from sheeprl@33b6 (+22 more)

### Community 4 - "Community 4"
Cohesion: 0.1
Nodes (17): ActorCritic, DreamerV3Agent, Container for the full agent with configurable architecture., Args:             obs_dim: Observation vector dimension.             act_dim: Ac, WorldModel, DreamerTrainer, Pre-sample `num_batches` batches at once (for CPU mode batched JIT)., Inference method with optional neuromodulation.          Args:             obs: (+9 more)

### Community 5 - "Community 5"
Cohesion: 0.09
Nodes (21): bm_finalise_episode(), _bm_finalise_tag(), bm_finalise_to_wandb_keys(), bm_reset_env(), bm_step_update(), bm_wandb_keys(), BMState, build_episode_log_dict() (+13 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (21): compute_lambda_values(), moments_init(), moments_update(), MomentsState, prepare_obs(), Ratio, dreamer-srl utility functions — leaf utilities ported from sheeprl@33b6366.  All, Compute lambda-return targets via reverse recursion.      Arguments match sheepr (+13 more)

### Community 7 - "Community 7"
Cohesion: 0.18
Nodes (14): get_observation(), Manhattan Collision Sensor (checks OOB and blocking obstacles)., Assembles the full observation vector, including noise if enabled., Assembles the full observation vector, including noise if enabled., Normalized Agent Location Sensor., Vectorized Resource Sensor (Chemical signature gradient)., Continuous Phasic Nociceptor: Detects contact with hiding predators, animals, an, Tonic interoceptive pain. Two modes (selected statically at JIT time):      - Co (+6 more)

### Community 8 - "Community 8"
Cohesion: 0.08
Nodes (16): empty(), dreamer-srl replay buffer — SequentialReplayBuffer (in-memory only).  Ported fro, Add data to the replay buffer (ring-buffer semantics).          Ported from shee, Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L105-L106, Add data to the replay buffer (ring-buffer semantics).          Ported from shee, Sample elements from the replay buffer in a sequential manner.          Ported f, Sample elements from the replay buffer in a sequential manner.          Ported f, Sequential replay buffer — in-memory, pure-numpy, no memmap.      Ported from sh (+8 more)

### Community 9 - "Community 9"
Cohesion: 0.14
Nodes (25): _card_title(), draw_action_pod(), draw_categorical_pod(), draw_intensity_pod(), draw_offline_card(), draw_spectrum_pod(), draw_vital_card(), _hbar() (+17 more)

### Community 10 - "Community 10"
Cohesion: 0.07
Nodes (33): calculate_drive(), _hunt_step(), jax_reset(), jax_step(), move_agent(), place_in_area(), Updates resource timers and regeneration., Updates predator states and positions, considering obstacles. (+25 more)

### Community 11 - "Community 11"
Cohesion: 0.18
Nodes (9): Training-time world-model forward over a [T, B] sequence.          Ported from s, Recurrent State-Space Model (RSSM) for DreamerV3.      Ported from sheeprl@33b63, Compute prior logits and stochastic state from the recurrent state.          Por, Compute posterior logits and stochastic state from hx + obs embedding., Apply uniform mixing (unimix) to flatten logits.          Ported from sheeprl@33, Sample or return mode of the Categorical stochastic state.          Ported from, Return deterministic initial recurrent state and initial posterior.          Por, One-step RSSM dynamic: §S4 reset + GRU + transition + representation.          P (+1 more)

### Community 12 - "Community 12"
Cohesion: 0.1
Nodes (24): draw_boresight_diamond(), draw_categorical_visual(), draw_dual_capsule_bar(), draw_pod_frame(), _load_icons(), JAX Environment Renderer.  Provides rendering utilities that convert JAX EnvStat, Draws a professional capsule-style progress bar showing reality vs perception., Draws a professional capsule-style progress bar showing reality vs perception. (+16 more)

### Community 13 - "Community 13"
Cohesion: 0.11
Nodes (22): draw_boresight_diamond(), draw_categorical_visual(), draw_dual_capsule_bar(), draw_pod_frame(), _load_icons(), JAX Environment Renderer.  Provides rendering utilities that convert JAX EnvStat, Draws a professional capsule-style progress bar showing reality vs perception., Draws a professional capsule-style progress bar showing reality vs perception. (+14 more)

### Community 14 - "Community 14"
Cohesion: 0.14
Nodes (15): combine_frame_and_activations(), plot_learning_curves(), plot_q_table(), plot_q_table_conventional(), plot_q_table_injury(), Visualization utilities for GridWorld Pain environment. Includes Q-table plottin, Visualizes the Q-table policy and values at different satiation levels., Visualizes a 3D Q-table (Conventional Mode). (+7 more)

### Community 15 - "Community 15"
Cohesion: 0.14
Nodes (13): dist_finalise_episode(), dist_reset_env(), dist_step_update(), dist_wandb_keys(), DistState, make_dist_state(), Per-tag distance running-mean aggregator.  Extracted from ``train.py`` (lines 12, Return per-episode mean-distance dict for env slot *i*.      Keys match the JAX- (+5 more)

### Community 16 - "Community 16"
Cohesion: 0.14
Nodes (13): episode_finalise_episode(), episode_reset_env(), episode_step_update(), episode_wandb_keys(), EpisodeAccumulatorState, make_episode_state(), Per-episode scalar accumulators for the 20 Episode/* WandB keys.  Mirrors the pa, Factory — returns a zero-initialised EpisodeAccumulatorState. (+5 more)

### Community 17 - "Community 17"
Cohesion: 0.15
Nodes (10): ContinueHead, MLPDecoder, MLP decoder from latent state to reconstructed observation.      # Ported from s, Forward pass: latent → MLP body → output head → obs_dim.          Args:, Continue (discount) head — Bernoulli logit over latent state.      # Ported from, Forward pass: latent → MLP body → scalar logit.          Args:             laten, init_weights(), Hafner truncated-normal weight initializer for Linear layers.      Returns a ker (+2 more)

### Community 18 - "Community 18"
Cohesion: 0.25
Nodes (4): EnvParams, ParallelEnv, Steps all environments in parallel., Vectorized Environment Wrapper for JAX.

### Community 19 - "Community 19"
Cohesion: 0.16
Nodes (15): evaluate_jax_checkpoint(), generic_inference(), Runs deterministic evaluation episodes using the JAX model.     When num_envs >, Runs deterministic evaluation episodes using the JAX model.     When num_envs >, Generic inference helper that works with both RecurrentPPO and DreamerV3 NNX mod, Generic inference helper that works with both RecurrentPPO and DreamerV3 NNX mod, Original single-env loop: one episode at a time., Original single-env loop: one episode at a time. (+7 more)

### Community 20 - "Community 20"
Cohesion: 0.16
Nodes (7): Decoder, DreamerGroupedLinear, DreamerGroupedMLP, DreamerObservationDecoder, Applies independent linear layers to N groups in parallel using einsum.     Uses, DreamerV3-compatible grouped MLP with LayerNorm and SiLU., Symmetric observation decoder for DreamerV3.     Matches the structure of Dreame

### Community 21 - "Community 21"
Cohesion: 0.17
Nodes (7): Single step transition with optional neuromodulation.          Args:, Dynamics-only step with optional neuromodulation (for imagination).          Arg, Returns initial RSSM state (and modulator state if enabled)., RSSM, SiLU, OneHotDist, One-Hot Categorical Distribution with Straight-Through Estimator and Unimix.

### Community 22 - "Community 22"
Cohesion: 0.15
Nodes (9): dreamer_srl_eval_rollout(), Deterministic single-env eval rollout for dreamer-srl.      Mirrors evaluate_jax, EpisodeRecorder, Episode recording format for offline (post-hoc) video rendering.  Each episode i, Exactly the fields render_jax_state reads. Keep in lockstep with renderer.py., Accumulates per-step data for one episode, then writes a single file., Accumulates per-step data for one episode, then writes a single file., _snapshot_state() (+1 more)

### Community 23 - "Community 23"
Cohesion: 0.16
Nodes (8): DreamerObservationEncoder, Hierarchical observation encoder for DreamerV3.     Replicates the Grouped Encod, Processes the observation through the hierarchy (pre-activation)., Configurable World Model matching PyTorch architecture.          Args:, DreamerNeuromodulatorRNN, Recurrent neuromodulatory network for DreamerV3 with dual input modes.      Oper, Lazily initialize the imagination projection layer.          Called after constr, Returns the initial modulator hidden state (zeros).

### Community 24 - "Community 24"
Cohesion: 0.19
Nodes (10): from_twohot(), Ratio, Cyclic inverse of symlog: sign(x) * (exp(|x|) - 1)., Computes the number of gradient steps to perform based on a target ratio      of, Converts a scalar to a Two-Hot distribution (soft discretization).     Used for, Symmetric logarithmic function: sign(x) * log(|x| + 1).     Used to compress the, Converts logits from Two-Hot distribution back to scalar (expectation).     Retu, symexp() (+2 more)

### Community 25 - "Community 25"
Cohesion: 0.15
Nodes (9): drqn_loss_fn(), Performs a single DRQN update step., A recurrent replay buffer for storing and sampling sequences., Adds a batch of transitions., Samples a batch of sequences.          Note: This is a simple implementation tha, Computes the DRQN loss with burn-in., RecurrentReplayBuffer, RecurrentTransition (+1 more)

### Community 26 - "Community 26"
Cohesion: 0.17
Nodes (9): dqn_loss_fn(), A simple Replay Buffer for JAX using JNP arrays., Adds a batch of transitions to the buffer., Samples a batch of transitions from the buffer., Computes the DQN loss (MSE)., Performs a single DQN update step., ReplayBuffer, Transition (+1 more)

### Community 27 - "Community 27"
Cohesion: 0.18
Nodes (9): action_shift(), build_agent(), MLPEncoder, agent.py — neural-network modules for the dreamer-srl v3 rebuild.  This module p, MLP encoder for single-key vector observations (no CNN — gridworld is vector obs, Forward pass: symlog(obs) → MLP hidden layers → dense_units output.          Arg, # NOTE: sheeprl concatenates hx FIRST, then input., Shift actions by one time-step: prepend zeros, drop the last action.      Ported (+1 more)

### Community 28 - "Community 28"
Cohesion: 0.2
Nodes (10): dreamer_srl eval_rollout — deterministic evaluation episodes.  Mirrors evaluate_, Subprocess-render recordings → consolidated MP4, then WandB-upload.      Near-ve, _render_and_upload(), Uploads an image (plot) to WandB.          Args:         image_path (str): Path, Uploads a video to WandB.          Args:         video_path (str): Path to the v, Attempts to login to WandB using a shared API key file if available.     The fil, suppress_output(), upload_image() (+2 more)

### Community 29 - "Community 29"
Cohesion: 0.2
Nodes (5): Encoder, Configurable Encoder matching PyTorch architecture.          The body produces a, Standard forward pass (no modulation)., Forward pass with neuromodulation (Injection A).                  Applies modula, Inference step for evaluation/rollouts.         Args:             x: Observation

### Community 30 - "Community 30"
Cohesion: 0.13
Nodes (9): LayerNormGRUCell, Args:             hidden_size: Hidden dimension (= deter_dim in RSSM)., Configurable Decoder matching PyTorch architecture.         Args:             in, hafner_init(), Hafner initialization: Truncated normal with stddev = scale / sqrt(fan_in)., ModulatedLayerNormGRUCell, ModulatedLayerNormGRUCell: A LayerNorm GRU cell with external gate-bias injectio, LayerNorm GRU cell with optional external gate-bias injection on the update gate (+1 more)

### Community 31 - "Community 31"
Cohesion: 0.2
Nodes (6): DRQNNetwork, get_action_drqn_nnx(), Forward pass for a single step., Returns the initial hidden state., Epsilon-greedy action selection for DRQN., Deep Recurrent Q-Network using Flax NNX.

### Community 32 - "Community 32"
Cohesion: 0.22
Nodes (6): ActorCriticMLP, get_action_and_value_ppo_nnx(), Apply the configured activation function., Forward pass to compute logits and value., Helper for inference with NNX., Actor-Critic MLP using Flax NNX with separate actor and critic networks.

### Community 33 - "Community 33"
Cohesion: 0.32
Nodes (5): Actor, Discrete actor — MLP body + categorical head over actions.      # Ported from sh, Apply unimix smoothing to logits — sheeprl Actor._uniform_mix (L839-L845)., Return post-unimix logits for a given latent, WITHOUT sampling.          Used by, Forward pass: latent → MLP → logits → action + log_prob + entropy.          Port

### Community 34 - "Community 34"
Cohesion: 0.25
Nodes (5): DQNNetwork, get_action_dqn_nnx(), Forward pass to compute Q-values., Epsilon-greedy action selection for DQN., Deep Q-Network using Flax NNX.

### Community 35 - "Community 35"
Cohesion: 0.18
Nodes (10): main(), Player, Get actions from the current observation.          Runs encoder + RSSM dynamic s, Training-loop driver — port of sheeprl dreamer_v3.py:L361-L765 main()., Stateful inference wrapper for a single-env dreamer agent.      Maintains (recur, Reset player recurrent + posterior state.          Sheeprl PlayerDV3.init_states, Get actions from the current observation.          Runs encoder + RSSM dynamic s, Training-loop driver — port of sheeprl dreamer_v3.py:L361-L765 main(). (+2 more)

### Community 36 - "Community 36"
Cohesion: 0.43
Nodes (4): __call__(), Decoder, RSSM, WorldModel

### Community 37 - "Community 37"
Cohesion: 0.22
Nodes (8): _build_continual_schedule(), ContinualSchedule, _load_stage_env_cfg(), dreamer_srl_main.py — training-loop driver for dreamer-srl v3.  Ports vendor/she, Discover stage YAMLs + load + validate the schedule file.      Ported from train, Describes a multi-stage curriculum: one env config + schedule per stage.      Po, Return the stage index that owns `episode` (0-based).          Mirrors train.py:, Build a full env Config for one stage: defaults + stage YAML.      Mirrors dream

### Community 38 - "Community 38"
Cohesion: 0.33
Nodes (3): LayerNormGRUCell, Single-step GRU-LN forward pass.          Args:             x  : input  [B, inpu, GRU cell with LayerNorm applied after the fused input projection.      Ported fr

### Community 39 - "Community 39"
Cohesion: 0.33
Nodes (3): MLP, Configurable MLP with LayerNorm + SiLU.          Args:             zero_init_out, Configurable Actor-Critic matching PyTorch architecture.         Args:

### Community 40 - "Community 40"
Cohesion: 0.4
Nodes (3): CriticHead, Single-layer output head for the critic with zero-init output linear.      # Por, Forward pass: linear projection of the final hidden state.          Args:

### Community 41 - "Community 41"
Cohesion: 0.4
Nodes (3): FullMLPHead, Full MLP head: mlp_layers hidden layers (Linear+LN+SiLU) + output linear.      #, Forward pass: hidden layers → output linear.          Args:             x: [...,

### Community 42 - "Community 42"
Cohesion: 0.4
Nodes (3): Single-layer output head for the reward model with zero-init output linear., Forward pass: linear projection of the final hidden state.          Args:, RewardHead

### Community 43 - "Community 43"
Cohesion: 0.4
Nodes (3): Composite world model: encoder + RSSM + decoder + reward head + continue head., Imagination rollout for behavior learning.          Ported from sheeprl@33b6366:, WorldModel

### Community 44 - "Community 44"
Cohesion: 0.24
Nodes (5): Config, get_default_config(), load_yaml(), Retrieves a value from the configuration. Raises ValueError if the key is missin, Merges another configuration dictionary or Config object into this one.

### Community 59 - "Community 59"
Cohesion: 0.2
Nodes (9): load_checkpoint(), make_checkpoint_manager(), Orbax checkpointing helpers for dreamer-srl.  Mirrors the checkpoint pattern in, Load a dreamer-srl checkpoint saved at `episode`.      Returns the raw pytree di, Load a dreamer-srl checkpoint saved at `episode`.      Returns the raw pytree di, Create an Orbax CheckpointManager under <results_dir>/checkpoints/.      Ported, Save dreamer-srl state to an Orbax checkpoint at the given episode count.      D, Save dreamer-srl state to an Orbax checkpoint at the given episode count.      D (+1 more)

### Community 60 - "Community 60"
Cohesion: 0.22
Nodes (9): apply_perceptual_noise(), build_sensory_viz(), get_observation_breakdown(), Applies vectorized, state-dependent Gaussian noise based on modality-specific mo, Applies vectorized, state-dependent Gaussian noise based on modality-specific mo, Returns a dict of {sensor_name: dimension} for observation components., Returns a dict of {sensor_name: dimension} for observation components., Build the sensory_data list consumed by renderer.render_jax_state.      If true_ (+1 more)

### Community 61 - "Community 61"
Cohesion: 0.25
Nodes (7): _append_per_measure_mean(), append_per_tag_means(), bm_log_wandb(), episode_logging.py — shared per-iteration episode-level WandB helpers.  Lifted v, Mean the same per-episode raw scalar (skipping NaN) across the window.      Lift, Group ep_data raw per-tag scalars by tag, mean across instances + episodes., Append all BM WandB keys to ep_log from the iteration's episodes.      Lifted fr

### Community 62 - "Community 62"
Cohesion: 0.25
Nodes (5): compute_lambda_values(), Lambda-return calculation with global GAMMA.     rewards: (H, B)     values: (H+, Fully JIT-compiled training loop for GPU buffer.                  Samples and tr, _scan_train_gpu(), train_step()

### Community 63 - "Community 63"
Cohesion: 0.33
Nodes (6): get_visual_offsets(), Generates Manhattan diamond offsets in a consistent order., Generates Manhattan diamond offsets in a consistent order., Matmul-optimized Visual Sensor (simplified object recognition)., Matmul-optimized Visual Sensor (configurable per-entity appearance vectors)., sense_visual()

### Community 64 - "Community 64"
Cohesion: 0.4
Nodes (5): BehaviorMeasureCfg, load_behavior_measure_cfg(), Validated configuration for the behavior-measure toolkit v1., Validated configuration for the behavior-measure toolkit v1., Load behavior_measures: from the YAML config.      Returns None if the top-level

### Community 65 - "Community 65"
Cohesion: 0.5
Nodes (4): load_env_config(), Recursively resolve `extends:` chains and return a merged Config.      Rules:, Resolve a config file to a fully-merged Config, honouring ``extends:``.      - A, _resolve_extends()

### Community 66 - "Community 66"
Cohesion: 0.5
Nodes (3): _DictState, Adapter that lets EpisodeRecorder._snapshot_state read from a dict slot buffer., Adapter that lets EpisodeRecorder._snapshot_state read from a dict slot buffer.

## Knowledge Gaps
- **386 isolated node(s):** `Vectorized Environment Wrapper for JAX.`, `Steps all environments in parallel.`, `vmapped step with automatic reset for finished environments.`, `JAX Environment Renderer.  Provides rendering utilities that convert JAX EnvStat`, `Loads icons from assets directory based on config.` (+381 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **19 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Community 35` to `Community 0`, `Community 65`, `Community 64`, `Community 3`, `Community 37`, `Community 6`, `Community 5`, `Community 8`, `Community 59`, `Community 10`, `Community 7`, `Community 44`, `Community 18`, `Community 22`, `Community 27`, `Community 28`, `Community 61`, `Community 62`?**
  _High betweenness centrality (0.549) - this node is a cross-community bridge._
- **Why does `train_step()` connect `Community 62` to `Community 35`?**
  _High betweenness centrality (0.347) - this node is a cross-community bridge._
- **Why does `DreamerNeuromodulatorRNN` connect `Community 23` to `Community 1`, `Community 2`, `Community 4`, `Community 39`, `Community 20`, `Community 21`, `Community 29`, `Community 30`?**
  _High betweenness centrality (0.229) - this node is a cross-community bridge._
- **Are the 20 inferred relationships involving `main()` (e.g. with `get_default_config()` and `load_env_config()`) actually correct?**
  _`main()` has 20 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `OneHotDist` (e.g. with `SiLU` and `LayerNormGRUCell`) actually correct?**
  _`OneHotDist` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `DreamerNeuromodulatorRNN` (e.g. with `SiLU` and `LayerNormGRUCell`) actually correct?**
  _`DreamerNeuromodulatorRNN` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `ModulatedLayerNormGRUCell` (e.g. with `SiLU` and `LayerNormGRUCell`) actually correct?**
  _`ModulatedLayerNormGRUCell` has 14 INFERRED edges - model-reasoned connections that need verification._