---
snapshot_label: v2_memory_build_complete
captured: 2026-05-16 13:22
source_commit: 0dab341
scope: src
graphify_version: 0.8.5
session_id: a06843e3-ec5e-4850-9b54-75f95633989b
---

# Code-graph snapshot — `v2_memory_build_complete`

Captured `2026-05-16 13:22` from commit `0dab341`.

To compare with the current code state: `python scripts/regen_code_graph.py && diff src/graphify-out/GRAPH_REPORT.md docs/llm_wiki/code_snapshots/20260516_1322_v2_memory_build_complete.md`.

To re-enter the originating conversation: `python scripts/open_conversation.py a06843e3-ec5e-4850-9b54-75f95633989b` (if the session was a `/wiki-write` invocation).

---

# Graph Report - src  (2026-05-16)

## Corpus Check
- 44 files · ~66,755 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 735 nodes · 1009 edges · 59 communities (46 shown, 13 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 143 edges (avg confidence: 0.65)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `ee45d2a3`
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

## God Nodes (most connected - your core abstractions)
1. `main()` - 24 edges
2. `OneHotDist` - 23 edges
3. `DreamerNeuromodulatorRNN` - 22 edges
4. `ModulatedLayerNormGRUCell` - 18 edges
5. `SiLU` - 14 edges
6. `get_observation()` - 13 edges
7. `DreamerTrainer` - 12 edges
8. `ParallelEnv` - 11 edges
9. `NeuromodulatorRNN` - 11 edges
10. `RSSM` - 11 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `train_step()`  [INFERRED]
  algorithms/dreamer_srl/dreamer_srl_main.py → models/dreamer_v3_trainer.py
- `_DictState` --uses--> `ParallelEnv`  [INFERRED]
  utils/evaluation_core.py → environment/wrapper.py
- `Player` --uses--> `ParallelEnv`  [INFERRED]
  algorithms/dreamer_srl/dreamer_srl_main.py → environment/wrapper.py
- `_run_parallel_env_eval()` --calls--> `ParallelEnv`  [INFERRED]
  utils/evaluation_core.py → environment/wrapper.py
- `main()` --calls--> `ParallelEnv`  [INFERRED]
  algorithms/dreamer_srl/dreamer_srl_main.py → environment/wrapper.py

## Communities (59 total, 13 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.05
Nodes (40): load_checkpoint(), make_checkpoint_manager(), Orbax checkpointing helpers for dreamer-srl.  Mirrors the checkpoint pattern in, Load a dreamer-srl checkpoint saved at `episode`.      Returns the raw pytree di, Create an Orbax CheckpointManager under <results_dir>/checkpoints/.      Ported, Save dreamer-srl state to an Orbax checkpoint at the given episode count.      D, save_checkpoint(), main() (+32 more)

### Community 1 - "Community 1"
Cohesion: 0.06
Nodes (28): ModulatedGRUCell, ModulatedGRUCell: A GRU cell with external gate-bias injection for neuromodulati, GRU cell with optional external gate-bias injection on the update gate.      The, Forward pass.          Args:             carry: Previous hidden state h_{t-1}, s, ModulatorOutput, NeuromodulatorRNN, Neuromodulatory networks with branched heads for RecurrentPPO and DreamerV3.  Im, Forward pass.          Args:             obs: Observation vector, shape (..., in (+20 more)

### Community 2 - "Community 2"
Cohesion: 0.06
Nodes (39): collect_trajectories(), compute_gae(), compute_mc_returns(), ppo_loss_fn(), PPOBatch, Collects parallel trajectories using jax.lax.scan., Performs a single PPO update step., Performs one full PPO iteration (collect + N epochs). (+31 more)

### Community 3 - "Community 3"
Cohesion: 0.06
Nodes (30): BernoulliSafeMode, IndependentBernoulli, mean(), dreamer-srl loss utilities — TwoHotEncoding, BernoulliSafeMode, and reconstructi, Initialize the distribution.          Ported from sheeprl@33b6366:sheeprl/utils/, Log-probability of target x under the two-hot distribution.          Ported from, Bernoulli distribution with a safe-mode property, matching sheeprl's BernoulliSa, Initialize the BernoulliSafeMode distribution.          Ported from sheeprl@33b6 (+22 more)

### Community 4 - "Community 4"
Cohesion: 0.1
Nodes (17): ActorCritic, DreamerV3Agent, Container for the full agent with configurable architecture., Args:             obs_dim: Observation vector dimension.             act_dim: Ac, WorldModel, DreamerTrainer, Pre-sample `num_batches` batches at once (for CPU mode batched JIT)., Inference method with optional neuromodulation.          Args:             obs: (+9 more)

### Community 5 - "Community 5"
Cohesion: 0.08
Nodes (22): bm_finalise_episode(), _bm_finalise_tag(), bm_finalise_to_wandb_keys(), bm_reset_env(), bm_step_update(), bm_wandb_keys(), BMState, make_bm_state() (+14 more)

### Community 6 - "Community 6"
Cohesion: 0.08
Nodes (21): compute_lambda_values(), moments_init(), moments_update(), MomentsState, prepare_obs(), Ratio, dreamer-srl utility functions — leaf utilities ported from sheeprl@33b6366.  All, Compute lambda-return targets via reverse recursion.      Arguments match sheepr (+13 more)

### Community 7 - "Community 7"
Cohesion: 0.13
Nodes (22): apply_perceptual_noise(), build_sensory_viz(), get_observation(), get_observation_breakdown(), get_visual_offsets(), Generates Manhattan diamond offsets in a consistent order., Matmul-optimized Visual Sensor (simplified object recognition)., Applies vectorized, state-dependent Gaussian noise based on modality-specific mo (+14 more)

### Community 8 - "Community 8"
Cohesion: 0.11
Nodes (9): empty(), dreamer-srl replay buffer — SequentialReplayBuffer (in-memory only).  Ported fro, Add data to the replay buffer (ring-buffer semantics).          Ported from shee, Sample elements from the replay buffer in a sequential manner.          Ported f, Sequential replay buffer — in-memory, pure-numpy, no memmap.      Ported from sh, Internal: retrieve samples given batch_idxes of shape [B*N, seq_len].          P, Sample using pre-computed time start indices and env indices.          This meth, Ported from sheeprl@33b6366:sheeprl/data/buffers.py:L105-L106 (+1 more)

### Community 9 - "Community 9"
Cohesion: 0.22
Nodes (18): _card_title(), draw_action_pod(), draw_categorical_pod(), draw_intensity_pod(), draw_offline_card(), draw_spectrum_pod(), draw_vital_card(), _hbar() (+10 more)

### Community 10 - "Community 10"
Cohesion: 0.16
Nodes (16): calculate_drive(), jax_step(), move_agent(), place_in_area(), Updates resource timers and regeneration., Updates predator states and positions, considering obstacles., Updates neutral animal positions (random patrol)., Orchestrates a full environment step in JAX. (+8 more)

### Community 11 - "Community 11"
Cohesion: 0.18
Nodes (9): Training-time world-model forward over a [T, B] sequence.          Ported from s, Recurrent State-Space Model (RSSM) for DreamerV3.      Ported from sheeprl@33b63, Compute prior logits and stochastic state from the recurrent state.          Por, Compute posterior logits and stochastic state from hx + obs embedding., Apply uniform mixing (unimix) to flatten logits.          Ported from sheeprl@33, Sample or return mode of the Categorical stochastic state.          Ported from, Return deterministic initial recurrent state and initial posterior.          Por, One-step RSSM dynamic: §S4 reset + GRU + transition + representation.          P (+1 more)

### Community 12 - "Community 12"
Cohesion: 0.17
Nodes (15): draw_boresight_diamond(), draw_categorical_visual(), draw_dual_capsule_bar(), draw_pod_frame(), _load_icons(), JAX Environment Renderer.  Provides rendering utilities that convert JAX EnvStat, Draws a professional capsule-style progress bar showing reality vs perception., Draws a modular Telemetry Pod frame. (+7 more)

### Community 13 - "Community 13"
Cohesion: 0.17
Nodes (15): draw_boresight_diamond(), draw_categorical_visual(), draw_dual_capsule_bar(), draw_pod_frame(), _load_icons(), JAX Environment Renderer.  Provides rendering utilities that convert JAX EnvStat, Draws a professional capsule-style progress bar showing reality vs perception., Draws a modular Telemetry Pod frame. (+7 more)

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
Cohesion: 0.15
Nodes (7): EnvParams, EnvState, auto_reset_step(), ParallelEnv, Steps all environments in parallel., vmapped step with automatic reset for finished environments., Vectorized Environment Wrapper for JAX.

### Community 19 - "Community 19"
Cohesion: 0.2
Nodes (12): _DictState, evaluate_jax_checkpoint(), generic_inference(), Runs deterministic evaluation episodes using the JAX model.     When num_envs >, Adapter that lets EpisodeRecorder._snapshot_state read from a dict slot buffer., Generic inference helper that works with both RecurrentPPO and DreamerV3 NNX mod, Original single-env loop: one episode at a time., Parallel env evaluation with episode-ticket design: only effective_num_envs run; (+4 more)

### Community 20 - "Community 20"
Cohesion: 0.16
Nodes (7): Decoder, DreamerGroupedLinear, DreamerGroupedMLP, DreamerObservationDecoder, Applies independent linear layers to N groups in parallel using einsum.     Uses, DreamerV3-compatible grouped MLP with LayerNorm and SiLU., Symmetric observation decoder for DreamerV3.     Matches the structure of Dreame

### Community 21 - "Community 21"
Cohesion: 0.17
Nodes (7): Single step transition with optional neuromodulation.          Args:, Dynamics-only step with optional neuromodulation (for imagination).          Arg, Returns initial RSSM state (and modulator state if enabled)., RSSM, SiLU, OneHotDist, One-Hot Categorical Distribution with Straight-Through Estimator and Unimix.

### Community 22 - "Community 22"
Cohesion: 0.16
Nodes (8): dreamer_srl_eval_rollout(), Deterministic single-env eval rollout for dreamer-srl.      Mirrors evaluate_jax, EpisodeRecorder, Episode recording format for offline (post-hoc) video rendering.  Each episode i, Exactly the fields render_jax_state reads. Keep in lockstep with renderer.py., Accumulates per-step data for one episode, then writes a single file., _snapshot_state(), write_run_meta()

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
Cohesion: 0.22
Nodes (5): LayerNormGRUCell, Args:             hidden_size: Hidden dimension (= deter_dim in RSSM)., Configurable Decoder matching PyTorch architecture.         Args:             in, hafner_init(), Hafner initialization: Truncated normal with stddev = scale / sqrt(fan_in).

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
Cohesion: 0.25
Nodes (5): DreamerModulatorOutput, Output from the DreamerV3 neuromodulator's branched heads., Compute head outputs from modulator hidden state., Observation mode: called during RSSM step() and get_action()., Imagination mode: called during RSSM imagine_step().

### Community 36 - "Community 36"
Cohesion: 0.43
Nodes (4): __call__(), Decoder, RSSM, WorldModel

### Community 37 - "Community 37"
Cohesion: 0.29
Nodes (4): ModulatedLayerNormGRUCell, ModulatedLayerNormGRUCell: A LayerNorm GRU cell with external gate-bias injectio, LayerNorm GRU cell with optional external gate-bias injection on the update gate, Forward pass.          Args:             x: Input vector (projected to hidden_si

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
Cohesion: 0.5
Nodes (4): jax_reset(), Resolve entity position overlaps via single-pass sequential scan.          Uses, Functional reset for the JAX environment.          Uses 4-phase overlap-free ent, resolve_overlaps_global()

## Knowledge Gaps
- **308 isolated node(s):** `Vectorized Environment Wrapper for JAX.`, `Steps all environments in parallel.`, `vmapped step with automatic reset for finished environments.`, `Configuration loader for JAX Environment.  Translates YAML config files into JAX`, `Validated configuration for the behavior-measure toolkit v1.` (+303 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **13 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Community 0` to `Community 3`, `Community 5`, `Community 6`, `Community 7`, `Community 8`, `Community 44`, `Community 18`, `Community 22`, `Community 27`, `Community 28`?**
  _High betweenness centrality (0.511) - this node is a cross-community bridge._
- **Why does `train_step()` connect `Community 5` to `Community 0`?**
  _High betweenness centrality (0.342) - this node is a cross-community bridge._
- **Why does `DreamerNeuromodulatorRNN` connect `Community 23` to `Community 1`, `Community 35`, `Community 4`, `Community 39`, `Community 20`, `Community 21`, `Community 29`, `Community 30`?**
  _High betweenness centrality (0.241) - this node is a cross-community bridge._
- **Are the 19 inferred relationships involving `main()` (e.g. with `get_default_config()` and `load_env_params()`) actually correct?**
  _`main()` has 19 INFERRED edges - model-reasoned connections that need verification._
- **Are the 18 inferred relationships involving `OneHotDist` (e.g. with `SiLU` and `LayerNormGRUCell`) actually correct?**
  _`OneHotDist` has 18 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `DreamerNeuromodulatorRNN` (e.g. with `SiLU` and `LayerNormGRUCell`) actually correct?**
  _`DreamerNeuromodulatorRNN` has 14 INFERRED edges - model-reasoned connections that need verification._
- **Are the 14 inferred relationships involving `ModulatedLayerNormGRUCell` (e.g. with `SiLU` and `LayerNormGRUCell`) actually correct?**
  _`ModulatedLayerNormGRUCell` has 14 INFERRED edges - model-reasoned connections that need verification._