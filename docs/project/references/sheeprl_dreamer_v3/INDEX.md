---
title: "Sheeprl DreamerV3 — Reference Implementation Walkthrough"
generated: 2026-05-12
status: reference
purpose: "Line-by-line walkthrough of sheeprl's DreamerV3 implementation, organized in file-and-line order. Source for re-implementing as `dreamer-srl` in our codebase."
---

# Sheeprl DreamerV3 — Reference Implementation Walkthrough

> **Purpose.** This folder contains a complete, line-by-line walkthrough of sheeprl's DreamerV3 implementation — every function, every class, every module-level constant, in the exact line order they appear in the source. The intent is to serve as the canonical reference for re-implementing the same algorithm in our codebase under the name `dreamer-srl` (sheeprl-replicated DreamerV3), with the same hyperparameters and network structures.

> **Why this exists.** The 2026-05-11 sheeprl drop-in smoke run ([jzgkcep4](https://wandb.ai/sungwoolee/grid_world_pain_sheeprl_test/runs/jzgkcep4)) showed that stock sheeprl reaches survival ~500 on our food-only NoPred task, vs ~106 for our DreamerV3 baseline. The cleanest path forward is to replicate sheeprl's implementation directly rather than continue walking the per-fix cascade. This walkthrough is the source material for that replication.

> **What this is NOT.** Not a paper-to-code mapping (that lives in [`../../concepts/dreamer_v3_implementation.md`](../../concepts/dreamer_v3_implementation.md)). Not a comparison against our codebase (the deviation list lives in the implementation reference §6). Not a re-implementation plan (that comes next). Just a faithful transcription with brief per-function explanations.

> **Reading order.** Start with [`dreamer_v3.md`](dreamer_v3.md) (the main training loop) — every other file is reached through its imports. The dependency graph is in §3 below.

---

## §1. Files in this walkthrough

Coverage check: every per-file doc's section count equals its source `grep -cE "^def |^class |^    def "` exactly.

| File | Source | Lines | Defs/Classes | Doc sections | Doc lines |
|---|---|---|---|---|---|
| [`dreamer_v3.md`](dreamer_v3.md) | `sheeprl/algos/dreamer_v3/dreamer_v3.py` | 780 | 2 | 2 ✓ | 849 |
| [`agent.md`](agent.md) | `sheeprl/algos/dreamer_v3/agent.py` | 1236 | 39 | 39 ✓ | 1632 |
| [`loss.md`](loss.md) | `sheeprl/algos/dreamer_v3/loss.py` | 88 | 1 | 1 ✓ | 133 |
| [`utils.md`](utils.md) | `sheeprl/algos/dreamer_v3/utils.py` | 235 | 10 | 10 ✓ | 364 |
| [`evaluate.md`](evaluate.md) | `sheeprl/algos/dreamer_v3/evaluate.py` | 57 | 1 | 1 ✓ | 94 |
| [`distribution.md`](distribution.md) | `sheeprl/utils/distribution.py` | 416 | 57 | 57 ✓ | 921 |
| [`buffers.md`](buffers.md) | `sheeprl/data/buffers.py` | 1180 | 55 | 55 ✓ | 1363 |
| [`models.md`](models.md) | `sheeprl/models/models.py` | 525 | 38 | 38 ✓ | 889 |
| [`utils_core.md`](utils_core.md) | `sheeprl/utils/utils.py` | 313 | 23 | 23 ✓ | 545 |
| **Total** | | **4,830** | **226** | **226** ✓ | **6,790** |

---

## §2. Per-file one-paragraph summaries

### [`dreamer_v3.md`](dreamer_v3.md)

`dreamer_v3.py` is the training entry point. It contains two functions: `train()` — one full gradient update over a `(sequence_length, batch_size)` replay slice, encompassing the world-model phase (RSSM rollout with the dynamic-learning loss from paper Eq. 4 — reconstruction + KL on prior/posterior + reward + continue heads), the behaviour-learning phase (`horizon`-step imagination from detached posteriors, lambda-return computation via `compute_lambda_values`, percentile-normalised actor update via `Moments` with REINFORCE for discrete actions and reparameterised gradients for continuous, plus entropy bonus from paper Eq. 11), and the critic phase (two-hot NLL of lambda-returns plus EMA-target regulariser from paper Eq. 10); and `main()` — the orchestrating loop that builds vectorised gymnasium envs (with `RestartOnException`), constructs world/actor/critic/target-critic/player via `build_agent`, hydra-instantiates three optimisers, sets up `EnvIndependentReplayBuffer` of `SequentialReplayBuffer`s (with optional memmap and resume-from-checkpoint), runs the env-step / gradient-step interleave under `Ratio`-controlled replay scheduling with Polyak-EMA target-critic updates, logs to fabric, checkpoints via `fabric.call("on_checkpoint_coupled", ...)`, and optionally runs a final evaluation and MLflow model registration. The file is the canonical reference point for how the DreamerV3 paper's Eqs. 4/10/11 are wired together in PyTorch + Lightning Fabric.

### [`agent.md`](agent.md)

`agent.py` defines every network and the factory that wires them together. The architectural centerpiece is `RSSM` — a recurrent state-space model where `(h_t, z_t)` is split into a deterministic part `h_t` (produced by a `LayerNormGRUCell` from previous `z, a`) and a stochastic part `z_t` (32 categoricals of 32 classes each, sampled with straight-through gradients and regularised by a 1% uniform mixture per `unimix=0.01`). The posterior `q(z_t | h_t, x_t)` is corrected by observation embedding; the prior `p(z_t | h_t)` predicts it ahead of time and is trained via KL. `MultiEncoder` and `MultiDecoder` wrap CNN+MLP modality-specific encoders/decoders. `Actor` produces actions through configurable distributions (discrete categorical, scaled-normal, tanh-normal). `PlayerDV3` is the rollout-time wrapper that carries `(h, z)` state across env steps with weights tied to the trainable modules. `build_agent` is the single entry point that consumes the Hydra config, instantiates every sub-network, applies Hafner-style truncated-normal init (with critical zero-init of the reward and critic heads' final layers — cascade target #27), wraps everything in Fabric for multi-device training, deep-copies a target critic, and aliases the player's parameters to the trainable ones.

### [`loss.md`](loss.md)

`loss.py` contains the single function `reconstruction_loss` — the combined world-model objective implementing Eq. 5 of Hafner et al. 2023 by combining five terms: (1) the multimodal decoder NLL summed across observation keys (MSE for pixels, symlog-Gaussian for vectors), (2) the reward predictor's symlog-two-hot cross-entropy, (3) a KL-balanced latent KL split into a dynamic term (β=0.5, posterior detached) that trains the prior and a representation term (β=0.1, prior detached) that trains the encoder — both floored by `max(·, free_nats=1.0)` to prevent posterior collapse, (4) a Bernoulli BCE on the continue/discount predictor, and (5) a mean reduction with outer `kl_regularizer`. Returns a 6-tuple (total + 5 per-term means) so the training loop can both back-prop the aggregate scalar and log each component.

### [`utils.md`](utils.md)

`dreamer_v3/utils.py` is the algorithm-specific helpers module. It defines (a) `AGGREGATOR_KEYS` and `MODELS_TO_REGISTER` (metric + checkpoint registries); (b) `Moments` — a buffer-only running 5th/95th-percentile EMA tracker that implements DreamerV3's scale-invariant actor normalization; (c) `compute_lambda_values` — backward TD(λ) recursion that produces the critic regression target and actor advantage from imagined rewards, values, and continue masks; (d) `prepare_obs` — dict-obs preprocessor that pushes numpy obs to fabric device, rescales image keys from uint8 to `[-0.5, 0.5]`, reshapes to `(T=1, B, ...)`; (e) `test` — no-grad single-episode evaluator driving `PlayerDV3`; (f) two weight initializers (`init_weights` truncated-normal Xavier-fan-avg, `uniform_init_weights(scale)` factory whose closure stamps a scale onto head Linears — `scale=0.0` is the zero-init for reward+critic heads); (g) `log_models_from_checkpoint` for MLflow registration.

### [`evaluate.md`](evaluate.md)

`evaluate.py` is the algorithm-registered top-level entry point for offline evaluation of a trained checkpoint. It spins up a Fabric logger, builds a single test-mode env worker via `make_env`, validates that the observation space is a `gym.spaces.Dict` (required because DreamerV3's encoder splits inputs by CNN/MLP keys), derives `actions_dim` and `is_continuous` from the action space, calls `build_agent` to reconstruct the modules from the checkpoint state (discarding the four trainable modules and keeping only the `PlayerDV3` inference wrapper), and hands that `player` to `test` from `utils.py` with `greedy=False` for stochastic-policy evaluation. No training, no replay buffer, no optimizer — purely "load → build inference agent → roll out."

### [`distribution.md`](distribution.md)

`distribution.py` is a leaf utility module (no internal sheeprl imports beyond `symlog`/`symexp` from `utils.py`) holding the entire probabilistic-distribution toolkit. The four load-bearing exports are: (1) `TwoHotEncodingDistribution` — two-hot encoding over a `symexp(linspace(-20, 20, K))` bin grid used by reward and critic heads to turn unbounded scalar regression into stable cross-entropy classification; (2) `SymlogDistribution` — symlog-space MSE/L1 loss for observation reconstruction; (3) `MSEDistribution` — plain unit-variance Gaussian for already-conditioned obs; (4) `BernoulliSafeMode` — numerically safe `mode` Bernoulli for the continue head. Two important secondaries: `OneHotCategoricalStraightThroughValidateArgs` provides the straight-through gradient estimator (`samples + (probs - probs.detach())`) that makes DreamerV3's discrete RSSM latents and discrete actions trainable end-to-end, and a `register_kl`-decorated kl-divergence function (line 405) provides the analytic KL between posterior and prior categorical latents. The `TruncatedStandardNormal`/`TruncatedNormal` block (lines 25–147) is upstream `torch_truncnorm` code carried for compatibility with continuous-action variants but unused by the default discrete DreamerV3 path.

### [`buffers.md`](buffers.md)

`buffers.py` is the entire replay-buffer module. DreamerV3's main loop instantiates an `EnvIndependentReplayBuffer` with `buffer_cls=SequentialReplayBuffer`: one `SequentialReplayBuffer` per parallel env, each running a `[buffer_size, 1, ...]` ring buffer over transition dicts (`observations`, `actions`, `rewards`, `terminated`, `truncated`, `is_first`). Each gradient step calls `sample_tensors(batch_size, sequence_length=per_rank_sequence_length, ...)` — the wrapper dispatches via `np.bincount` to draw per-env subsets, each sub-buffer samples contiguous length-L windows that **do NOT respect episode boundaries** (a chunk can straddle a `done`), and the trainer relies on the stored `is_first` flag to reset RSSM recurrent state at the correct intra-chunk step. Memmap mode lets the entire buffer be checkpointed (`<memmap_dir>/env_i/<key>.memmap` files) and reloaded transparently. `EpisodeBuffer` is an alternative (variable-length, episode-bounded, with optional `prioritize_ends`) that is **not** what DreamerV3's main loop uses, but is fully documented for completeness.

### [`models.md`](models.md)

`models.py` is sheeprl's generic, algorithm-agnostic network toolbox. The image encoder branch is built from `CNN` (with `LayerNormChannelLast` and `nn.SiLU` per Hafner spec), and the image decoder is the symmetric `DeCNN`. `MultiEncoder` wraps `CNN` + `MLP` into a dict-aware encoder; `MultiDecoder` does the inverse. The RSSM's recurrent core is `LayerNormGRUCell` parameterised with the local `LayerNorm` wrapper (`eps=1e-3`) — Hafner's fused-gate GRU with the `sigmoid(update - 1)` "keep-old-state" bias. All vector heads (reward, continue, value, policy, prior, posterior) are instances of `MLP`, with `nn.SiLU` activation and the local `LayerNorm`; each terminal `nn.Linear` is the target of the `uniform_init_weights(0.0)` zero-init (cascade target #27). `NatureCNN` is not used by DreamerV3 (it's there for the PPO/SAC family).

### [`utils_core.md`](utils_core.md)

`sheeprl/utils/utils.py` (output filename `utils_core.md` to disambiguate from `dreamer_v3/utils.py`) is the generic, algorithm-agnostic utility module. The load-bearing items for DreamerV3 are: (a) **`Ratio`** — the stateful replay-ratio scheduler whose `__call__` decides how many gradient updates to fire per env step, with fractional-debt bookkeeping via the `_prev += repeats / _ratio` accumulator; (b) `symlog`/`symexp` plus `two_hot_encoder`/`two_hot_decoder` — the numerical-stability backbone (paper Eq. 9); (c) `save_configs` + `unwrap_fabric` for the checkpoint pipeline. Also carries: `dotdict` Hydra-config wrapper, `gae` (used by PPO-family, not DreamerV3), `init_weights` Kaiming init, `normalize_tensor`, `polynomial_decay`, `print_config`, `safetanh`/`safeatanh` for tanh-squashed policies.

**Notable absence**: there is no shared `polyak_update` helper here — DreamerV3's EMA target-critic update is implemented inline in the main loop in [`dreamer_v3.md`](dreamer_v3.md).

---

## §3. Dependency graph (who imports whom)

```
dreamer_v3.py (main loop)
  ├── agent.py (WorldModel, build_agent, PlayerDV3)
  │     ├── models.py (MLP, CNN, MultiEncoder, MultiDecoder, LayerNormGRUCell, LayerNorm variants)
  │     ├── distribution.py (TwoHotEncodingDistribution, SymlogDistribution, MSEDistribution, BernoulliSafeMode, OneHotCategoricalStraightThroughValidateArgs)
  │     ├── utils.py (Moments, uniform_init_weights for zero-init heads)
  │     └── utils_core.py (some shared utilities)
  ├── loss.py (reconstruction_loss)
  │     └── distribution.py (TwoHot + Symlog + Bernoulli + analytic-KL for log_prob)
  ├── utils.py (Moments, compute_lambda_values, prepare_obs, test, init helpers)
  │     ├── distribution.py (TwoHot for compute_lambda_values' two-hot critic decoding)
  │     └── agent.py (PlayerDV3 for `test`)
  ├── buffers.py (EnvIndependentReplayBuffer wrapping SequentialReplayBuffer)
  └── utils_core.py (Ratio scheduler, save_configs, unwrap_fabric)

evaluate.py (top-level eval)
  ├── agent.py (build_agent, PlayerDV3)
  └── utils.py (test)
```

---

## §4. Cross-reference: where each cascade candidate lives in the walkthrough

| Cascade # | Description | Lives in | Read this section |
|---|---|---|---|
| **#2** | Paper-canonical two-hot bins (`symexp(linspace(-20, 20, K))`) | `distribution.py` (`TwoHotEncodingDistribution`) | [`distribution.md`](distribution.md) — the bin grid is computed in `__init__` of the distribution class. Also in `utils_core.py` as `two_hot_encoder`/`two_hot_decoder` standalone functions. |
| **#27** | Zero-init reward + critic head output Linears | `agent.py` (`build_agent`) + `utils.py` (`uniform_init_weights`) + `models.py` (`MLP` terminal Linear) | [`agent.md`](agent.md) `build_agent` final phase; [`utils.md`](utils.md) `uniform_init_weights(0.0)` factory; [`models.md`](models.md) `MLP` terminal Linear that receives the init. |
| **#28** | GRU reset gate applied in candidate hidden-state update | `models.py` (`LayerNormGRUCell`) | [`models.md`](models.md) — fused-gate GRU with `sigmoid(update - 1)` keep-old-state bias; the canonical paper formula. |
| **#29** | Critic self-EMA regularization (slow-target additional loss) | `dreamer_v3.py` (`train`, critic phase) | [`dreamer_v3.md`](dreamer_v3.md) — value_loss includes both the lambda-return regression AND a self-EMA term, see the critic-phase block in `train`. |
| **#30** | RSSM prior + posterior heads have a hidden layer | `agent.py` (`transition_model`, `representation_model` MLP construction) | [`agent.md`](agent.md) — `RSSM.__init__` builds `transition_model` and `representation_model` as `MLP(..., hidden_sizes=[hidden_size])` not bare Linear. |

---

## §5. Hyperparameters

The walkthrough above documents the code. The hyperparameter values that drive it live in YAML configs at `tmp/sheeprl/sheeprl/configs/algo/dreamer_v3*.yaml` (model size variants XS/S/M/L/XL) and `configs/exp/dreamer_v3*.yaml` (per-environment recipes). The 2026-05-11 smoke run used `dreamer_v3_XS` which inherits from `dreamer_v3.yaml` (XL by default) and overrides `dense_units: 256`, `mlp_layers: 1`, `recurrent_state_size: 256`, and the two transition/representation hidden sizes. The exact hyperparameter mapping is not in scope for this folder — see the eventual `dreamer-srl` plan doc.

---

## §6. Conventions used in each per-file document

- **Line order, not function order.** Every section header carries the line number from the source, so `## Line 123 — function_name` corresponds to a specific point in the file.
- **Every `def` and `class` appears.** Counts match `grep -cE "^def |^class |^    def " <source>` exactly (verified in §1 above).
- **Full code blocks** for every function/class, exactly as in the source. Methods inside classes get their own sections with `## Line X — ClassName.method_name` headings.
- **3–6 line explanation** after each code block: what it does, role in DreamerV3, non-obvious math, cross-refs.
- **Cross-references** between files use markdown links to peer files (e.g., `[WorldModel](agent.md)`).

---

## §7. Next step after this reference lands

A `senior-developer` plan under `docs/develop/active/dreamer_srl/` will use this walkthrough as source material to draft the `dreamer-srl` re-implementation. Target: bit-identical algorithm semantics with the same network shapes and hyperparameters, implemented in our JAX/Flax stack so it can plug into our existing training pipeline (`train.py`) and config system. The walkthrough's line-order organization means the implementer can simply walk file by file, function by function, translating each in place.
