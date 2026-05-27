---
title: "dreamer-srl: Sheeprl-Replicated DreamerV3 in JAX/Flax — Implementation Plan"
topic: dreamer
status: superseded
created: 2026-05-12
last_updated: 2026-05-12
phase: 2
revision: v2
---

> **Superseded by**: [`docs/pi/calls/2026-05-12_dreamer_backend.md`](../../../pi/calls/2026-05-12_dreamer_backend.md) — PI call picked Option 1 (sheeprl-direct, minimal bridge) over rebuilding DreamerV3 in JAX. The bridge plan that replaces this work is at [`docs/develop/active/sheeprl_bridge/IMPLEMENTATION_PLAN.md`](../../active/sheeprl_bridge/IMPLEMENTATION_PLAN.md).
> The `superseded_by:` frontmatter field is omitted because the successor lives under `docs/pi/`, not `docs/develop/` — the develop INDEX validator only resolves successors within its own subtree.

# dreamer-srl: Sheeprl-Replicated DreamerV3 in JAX/Flax — Implementation Plan

> **Status**: PLANNED
> **Opened**: 2026-05-12
> **Related**: [Sheeprl walkthrough INDEX](../../../project/references/sheeprl_dreamer_v3/INDEX.md) · [Sheeprl drop-in diagnosis](../diagnosis/sheeprl_drop_in_test.md)

---

## Context

Our in-house Dreamer baseline (the algorithm registered in `train.py` as `DreamerV3`, source under `src/models/dreamer_v3_*.py`) survives only ~106 environment steps on a 5×5 food-only NoPred task (no predators, agent dies if it starves). A drop-in run of the community PyTorch implementation **sheeprl** on the same task — the standard published `dreamer_v3_XS` size — survives ~500 steps (i.e. runs out the env's 500-step time-limit clock without ever starving). That is the same task, the same env, the same reward function. The algorithm is doing different work. The diagnosis run that established this gap is the WandB run [`jzgkcep4`](https://wandb.ai/sungwoolee/grid_world_pain_sheeprl_test/runs/jzgkcep4) (see [the diagnosis doc](../diagnosis/sheeprl_drop_in_test.md) for the full analysis).

Over the past week we have been walking a per-fix cascade — comparing our code line by line against sheeprl and shipping the differences one at a time. Two fixes have landed (paper-canonical two-hot reward bins, zero-init reward and critic head output layers); three remain (GRU reset gate, critic slow-target self-EMA loss, RSSM hidden layers on prior/posterior heads). The cascade has been slow and the cumulative risk is high: even after all five fixes, latent differences elsewhere in the implementation may still leave us short of sheeprl's regime.

This plan is the user's decision to stop walking the cascade and instead **re-implement sheeprl's DreamerV3 directly** as a new JAX/Flax algorithm called `dreamer-srl` (sheeprl-replicated DreamerV3). The existing JAX DreamerV3 stays untouched as a control. The goal is bit-identical algorithm semantics with the same network shapes and hyperparameters as sheeprl, plugged into our existing JAX training pipeline so it can use our env, our config loader, our WandB logging, and (eventually) our neuromodulation hooks. No NMN, no FiLM, no precision modulation in this plan — pure replication. The success gate is mean survival ≥ ~500 steps on food-only NoPred over 3 seeds within 2× sheeprl's wall-clock. That parity gate is exactly what `jzgkcep4` passed.

## Analysis

### Why a new module, not a patch

The current `src/models/dreamer_v3_*.py` family has accreted: NMN injection points, mixture sampling, hierarchical encoding, modulation knobs, an `apply_gru_reset_gate` toggle, two-hot bin variants, and FiLM hooks. Patching it toward sheeprl-fidelity while preserving all those features (or quarantining them behind flags) is the path that produced the 5-fix cascade in the first place. The cleaner path — and the path the user has chosen — is a parallel module with no inherited cruft: copy the algorithm verbatim from sheeprl, get parity, then in a separate later plan re-introduce the modulation hooks one at a time on top of a known-good base.

### Where the current Dreamer ends, where dreamer-srl begins

The existing JAX DreamerV3 stack:

| File | Lines | Role |
|---|---:|---|
| `src/models/dreamer_v3_network.py` | 80 | tiny — config struct/dataclass only |
| `src/models/dreamer_v3_nnx.py` | 735 | the WorldModel / RSSM / Actor / Critic / Player in NNX |
| `src/models/dreamer_v3_trainer.py` | 1023 | training loop, loss, sequence collection, replay, modulator wiring |
| `src/models/dreamer_v3_util.py` | 230 | `Ratio`, two-hot helpers, lambda-return, init weights |
| **Total** | **2068** | |

The sheeprl reference (PyTorch source, all line counts from `wc -l`):

| File | Source path | Lines |
|---|---|---:|
| `dreamer_v3.py` | `tmp/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py` | 780 |
| `agent.py` | `tmp/sheeprl/sheeprl/algos/dreamer_v3/agent.py` | 1236 |
| `loss.py` | `tmp/sheeprl/sheeprl/algos/dreamer_v3/loss.py` | 88 |
| `utils.py` | `tmp/sheeprl/sheeprl/algos/dreamer_v3/utils.py` | 235 |
| `models.py` | `tmp/sheeprl/sheeprl/models/models.py` | 525 |
| `distribution.py` | `tmp/sheeprl/sheeprl/utils/distribution.py` | 416 |
| `buffers.py` | `tmp/sheeprl/sheeprl/data/buffers.py` | 1180 |
| `utils_core.py` | `tmp/sheeprl/sheeprl/utils/utils.py` | 313 |
| **Total** | | **4773** |

The walkthrough at `docs/project/references/sheeprl_dreamer_v3/` documents every `def` and `class` in those files in source-line order; the `dreamer-srl` implementer translates file-by-file using that walkthrough as the map.

### Scaffolding to reuse from `train.py` and `src/`

The existing JAX scaffolding is large and most of it is reused unchanged. Re-implementing it would be wasted work and would risk silent drift from the rest of the project. Specifically, **dreamer-srl reuses**:

- **Env interface.** `src/environment/core.py` (`jax_step`, `jax_reset`), `src/environment/sensor.py` (`get_observation`), `src/environment/wrapper.py` (`ParallelEnv`), `src/environment/config_loader.py` (`load_env_params`). Sheeprl's `make_env` + `gym.vector.SyncVectorEnv` is replaced by our JAX-native `ParallelEnv` — that is the whole point of porting to JAX in the first place.
- **Config loader.** `src/utils/config.py` (`Config`, `Config.load_yaml`, `Config.get_mandatory`). No Hydra. Sheeprl's `cfg.algo.world_model.recurrent_model.recurrent_state_size` becomes `agent.get_mandatory('rssm.recurrent_state_size')` (or similar — flattened to our YAML conventions).
- **WandB logging.** `src/utils/wandb_utils.py` (`wandb_login`); the existing per-iteration logging block in `train.py` (the metric-flush sections around lines 2635–2700) is reused — only the metric *names* change to match sheeprl's (`Loss/world_model_loss`, `Game/ep_len_avg`, etc., per `AGGREGATOR_KEYS` in `tmp/sheeprl/sheeprl/algos/dreamer_v3/utils.py:20-36`).
- **Checkpointing.** Orbax block in `train.py` lines ~263–267, ~541. No memmap replay (sheeprl-specific; out of scope).
- **CLI / argparse.** The whole `train.py` `--config` / `--algorithm` / `--seed` / `--wandb-name` / `--total-timesteps` / `--num-envs` flow. Adds one new value to the `--algorithm` enum.

dreamer-srl **does NOT reuse** anything from `src/models/dreamer_v3_*.py`. Those four files stay untouched. The new module is a clean parallel tree, free of NMN/FiLM/mixture-sampling/hierarchical-encoding/modulation hooks.

### The five cascade items and where they live in dreamer-srl

The walkthrough §4 names five known-divergent items between our current Dreamer and sheeprl. All five are paper-canonical and ON by default in stock sheeprl. In dreamer-srl every one of these is correct **from the first line**:

| # | Item | dreamer-srl call site | Sheeprl reference |
|---|---|---|---|
| **#2** | Two-hot bins as `linspace(-20, +20, 255)` stored in *symlog space* (bin centers in real space are `symexp(bins)`, accessed only via `mean`/`mode`) | `loss.py` reward/critic head decoding inside the `TwoHotEncoding` distribution class | `tmp/sheeprl/sheeprl/utils/distribution.py:185-260` (`TwoHotEncodingDistribution`); critical: `self.bins = linspace(...)` at line 237 lives in symlog space |
| **#27** | Zero-init reward + critic head output linear (`uniform_init_weights(0.0)`) | `agent.py` `build_agent` final init phase | `tmp/sheeprl/sheeprl/algos/dreamer_v3/agent.py:1170-1180` |
| **#28** | GRU candidate-state uses reset gate: `cand = tanh(reset * cand_proj)` | `models.py` `LayerNormGRUCell.__call__` | `tmp/sheeprl/sheeprl/models/models.py` — `LayerNormGRUCell` (lookup via INDEX) |
| **#29** | Critic value loss = `−qv.log_prob(λ-target.detach()) − qv.log_prob(target_critic_value.detach())` (slow-target self-EMA term) | `train.py` critic phase | `tmp/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:313-316` |
| **#30** | RSSM transition_model + representation_model each have one hidden layer (`hidden_sizes=[hidden_size]`), not bare Linear | `agent.py` `build_agent` (`MLP(... hidden_sizes=[transition_hidden_size] ...)`) | `tmp/sheeprl/sheeprl/algos/dreamer_v3/agent.py:1021-1051` |

## Training-loop semantics (silent omissions called out 2026-05-12)

The three reviewers (professor-rl-bayesian-dl, math-reviewer, code-reviewer) collectively flagged a set of training-loop mechanics that sheeprl's `dreamer_v3.py:train()` does silently but which the v1 plan did not name. These are NOT optional — Checkpoint 8 forward-parity fails without them. Listed here once so the developer reads them globally, then re-cited at the file-by-file change locations.

**S1. `is_first[0] = 1` force-set on every sampled chunk** (professor-rl-bayesian-dl #1; `dreamer_v3.py:133`).
Inside `one_train_step`, BEFORE the dynamic-learning rollout, unconditionally overwrite the first timestep's `is_first` to 1:
```python
batch["is_first"] = batch["is_first"].at[0].set(1.0)   # force-set first timestep
```
This guarantees the RSSM recurrent + stochastic carry reset to the learned initial state at the first step of every replay chunk, regardless of what was stored at that index in the buffer. Without this, the posterior at t=0 is computed against an arbitrary `h₀`, not against the learned `tanh(h̄₀)`, and the bug is silent (losses still decrease, just to a different basin).

**S2. Prepend-zero-action shift** (professor-rl-bayesian-dl #2; `dreamer_v3.py:137`).
Inside `one_train_step`, BEFORE the dynamic-learning rollout, shift actions by one step — the action consumed at time `t` is the action taken at `t-1`:
```python
shifted_actions = jnp.concatenate(
    [jnp.zeros_like(actions[:1]), actions[:-1]], axis=0
)   # action-shift: action at t is action taken at t-1
```
Feed `shifted_actions`, not raw `actions`, into the RSSM rollout. The first-step action becomes `0`, which combined with `is_first=1` (from S1) gives a clean reset. With the wrong shift the prior `p_φ(z_t | h_t)` predicts the wrong distribution.

**S3. `learning_starts` random-action prefill phase** (professor-rl-bayesian-dl #3; `dreamer_v3.py:604-617, 706`).
The YAML key `agent.learning_starts: 1024` is loaded but the v1 plan did not wire it. Two wirings required:

*(a) Collection branch.* In `collect_step` (or the top-level `train.py` action-selection switch), branch on whether the policy has crossed `learning_starts`:
```python
if iter_num <= learning_starts and not resuming_from_checkpoint:
    actions = uniform_random_action(action_dim, num_envs, key)   # uniform prefill
else:
    actions = player.get_actions(obs, h, z, key)                 # policy
```

*(b) Gradient-step gate.* In the top-level `train.py` outer loop, gate `one_train_step` calls on `iter_num >= learning_starts`. The `Ratio` scheduler's `pretrain_steps=0` default is correct — sheeprl XS does not do a pretrain burst, only the random-action prefill.

Note: sheeprl's `prefill_steps = learning_starts - int(learning_starts > 0)` off-by-one (`dreamer_v3.py:557`) means `learning_starts=1024` actually starts training at step 1024 (not 1023). Sheeprl counts in **iterations** (`iter_num = policy_step / policy_steps_per_iter`, where `policy_steps_per_iter = num_envs * action_repeat = 4 * 1 = 4` for us), so `learning_starts=1024` = 256 iterations. Document this in the YAML comment so the developer doesn't off-by-one.

**S4. `is_first` reset uses arithmetic-mask, resets THREE quantities** (code-reviewer #3, math-reviewer 🟡 #3; `agent.py:423-429`).
Sheeprl's `RSSM.dynamic` resets via the arithmetic-mask form (NOT `jnp.where`), and resets three quantities (action, recurrent_state, posterior) — not two:
```python
# is_first shape: [B, 1] (trailing singleton from buffer; see buffers.py contract)
action = (1.0 - is_first) * action                                 # ZEROED
initial_recurrent, initial_posterior = self.get_initial_states((batch_size,))
recurrent_state = (1.0 - is_first) * recurrent_state + is_first * initial_recurrent
# Posterior MUST be reshape-flattened from [B, S, D] to [B, S*D] BEFORE masking:
posterior = posterior.reshape(*posterior.shape[:-2], -1)           # [B, S*D]
initial_posterior = initial_posterior.reshape(*posterior.shape)    # [B, S*D]
posterior = (1.0 - is_first) * posterior + is_first * initial_posterior
```
This is the arithmetic-mask form `(1 - is_first) * x + is_first * init`, NOT `jnp.where(is_first[..., None], init, current)`. They are numerically equivalent but the arithmetic form matches sheeprl's source 1:1 and is the convention to follow for bit-identity.

**S5. True-continue splice at imagination index 0** (math-reviewer 🟡 #2; `dreamer_v3.py:246-248`).
In the actor / behaviour-learning sub-phase, BEFORE calling `compute_lambda_values`, splice the observed `(1 - data["terminated"])` at imagination index 0 of the predicted-continues tensor:
```python
true_continue_step0 = (1.0 - data["terminated"][0])   # observed continue at step 0
predicted_continues = continue_dist.mode               # predicted for steps 1..H
continues = jnp.concatenate(
    [true_continue_step0[None], predicted_continues[1:]], axis=0
)
```
Without this, the imagined-trajectory step-0 continue is the *predicted* continue head output, which deviates from sheeprl.

**S6. Discount weighting on actor AND critic losses** (math-reviewer 🟡 #4, #5; `dreamer_v3.py:297, 316`).
Both losses are multiplied element-wise by the cumulative-discount mask `disc = cumprod(continues * gamma, axis=0) / gamma` (which gives `disc[0]=1`), `[:-1]`-sliced, gradient-detached:
```python
discount = jax.lax.stop_gradient(jnp.cumprod(continues * gamma, axis=0) / gamma)
# Actor (REINFORCE):
policy_loss = -jnp.mean(
    discount[:-1] * (objective + ent_coef * entropy[:-1])
)   # objective = log_prob(sg(a)) * sg(advantage); both shape [H, B]
# Critic (two-hot NLL + EMA self-reg):
value_loss = jnp.mean(
    (-qv.log_prob(jax.lax.stop_gradient(lambda_target))
     - qv.log_prob(jax.lax.stop_gradient(target_critic_value)))
    * discount[:-1].squeeze(-1)
)
```
Sheeprl `dreamer_v3.py:292-293` wraps the `discount = cumprod(...) / gamma` line in `with torch.no_grad():` — the JAX equivalent is `jax.lax.stop_gradient` on the result. Both `[:-1]` slices are critical: the last imagined step has no log_prob target.

**S7. Advantage `low`-offset cancellation** (math-reviewer 🟡 #6; `dreamer_v3.py:276-279`).
Sheeprl normalises lambda-values and baseline via `Moments`:
```python
normed_lambda   = (lambda_values   - offset) / invscale
normed_baseline = (baseline_values - offset) / invscale
advantage = normed_lambda - normed_baseline
```
The `offset = Moments.low` algebraically cancels in the subtraction, so the effective advantage is `(G^λ - v) / max(1, high - low)`. This is helpful for the implementer to know — but the implementation must STILL apply per-term normalisation (the cancellation is algebraic, not coded). Also note: `Moments` is used **only** for actor normalisation; the critic regression target is the **un-normalised** `lambda_values` (sheeprl `dreamer_v3.py:314` uses raw `lambda_values.detach()`, NOT normed).

**S8. Free-nats floor is per-element, applied BEFORE the mean** (math-reviewer 🟡 #7; `loss.py:100-106`).
The `max(·, free_nats)` floor is element-wise over the `[T, B]` KL tensor, NOT applied to the post-mean scalar:
```python
# dyn_loss and repr_loss start as [T, B] KL tensors
dyn_loss  = kl_dynamic * jnp.maximum(dyn_loss,  kl_free_nats)    # [T, B], per-element floor
repr_loss = kl_representation * jnp.maximum(repr_loss, kl_free_nats)
# Then reduce:
kl_total = (dyn_loss + repr_loss).mean()
```
A literalist might implement `max(mean(dyn_loss), 1.0)` instead — that's wrong.

**S9. `Independent(BernoulliSafeMode, 1)` wrap on continue head; `dims=1` on observation decoder** (professor-rl-bayesian-dl #10; `dreamer_v3.py:200, 279`; `dreamer_v3.py:186-193`).
- Continue distribution is wrapped as `Independent(BernoulliSafeMode(logits), 1)` — re-interprets the trailing axis as the event dim, so `log_prob` returns `[T, B]` per (T,B) cell, not `[T, B, 1]`. Same wrap applies in the imagination loop (`continues = Independent(BernoulliSafeMode(...), 1).mode`).
- Observation decoder output is wrapped as `MSEDistribution(mode, dims=len(shape[2:]))` or `SymlogDistribution(mode, dims=...)`. For a single-key MLP observation of shape `[T, B, F]`, `dims=1` — the trailing F axis is summed by `log_prob`.

The math is invariant for our single-dim case (output_dim=1 on the continue head), but the implementation must name the wrap correctly so future multi-dim outputs work.

**S10. Continue target is `1 - terminated` (no γ multiplier)** (professor-rl-bayesian-dl §2 borderline; `dreamer_v3.py:201`).
Sheeprl's `loss.py` docstring says `(1 - dones) * γ` but the actual call site uses just `1 - terminated`. The code is authoritative — use `1 - terminated`, no γ multiplier on the target.

## Implementation Plan

### Design

Five new Python modules under `src/algorithms/dreamer_srl/`, plus a config family under `configs/models/dreamer_srl/`. The training entry (`train.py`) gets one new branch in the algorithm dispatch (≈30 lines added; nothing removed). No existing file under `src/models/`, `src/environment/`, `src/utils/`, or `configs/models/` is modified.

**The five new modules:**

```
src/algorithms/
└── dreamer_srl/
    ├── __init__.py           # exports public symbols only
    ├── agent.py              # WorldModel, RSSM, Actor, Critic, Player, build_agent
    ├── loss.py               # reconstruction_loss (paper Eq. 4) + critic/actor losses (Eqs. 10/11)
    ├── utils.py              # Moments, compute_lambda_values, prepare_obs, init weights
    ├── buffers.py            # SequentialReplayBuffer (in-memory, no memmap)
    └── train.py              # one_train_step + collect-step orchestration; called by top-level train.py
```

**Translation order** (matches the walkthrough §3 dependency graph — translate leaves first, then trunks):

1. **`utils.py`** — `symlog`/`symexp`, `init_weights` (Hafner truncated-normal), `uniform_init_weights(scale)`, `compute_lambda_values`, `Moments`, `Ratio`, `prepare_obs`.
2. **Distributions** (folded into `loss.py` or a small inline module — sheeprl puts them in `utils/distribution.py` but they are leaf utilities) — `TwoHotEncoding`, `Symlog` log-prob, `MSE` log-prob, `BernoulliSafeMode.log_prob`, straight-through categorical sampler.
3. **`buffers.py`** — `SequentialReplayBuffer` (one per parallel env; sample `[T, B]` contiguous windows that **may straddle dones** — RSSM resets on `is_first`).
4. **`agent.py`** — `LayerNormGRUCell` (with fix #28), `MLP` (Hafner-init terminal linear), `MultiEncoder`/`MultiDecoder` (MLP-only; no CNN — our env is vector-obs), `RSSM` (with fix #30 hidden layers on prior/posterior), `Actor`, `Critic`, `Player`, `build_agent` (with fix #27 zero-init on reward + critic terminal linears).
5. **`loss.py`** — `reconstruction_loss` (KL-balanced posterior/prior, free-nats floor, reward NLL, observation NLL, continue BCE). Critic and actor losses live in `train.py` because they entangle imagination state and target-critic.
6. **`train.py`** (the module, not the top-level entry) — `one_train_step(state, batch, key) -> (new_state, metrics)` doing dynamic-learning rollout, imagination, lambda returns, actor / critic / world-model gradient steps, EMA target-critic update. Pure-functional, JIT-compatible.

**Integration point** in top-level `train.py`: a new branch `elif algorithm == "dreamer-srl":` that imports `from src.algorithms.dreamer_srl.train import build_state, one_train_step, collect_step`, mirroring the existing `JAX_DreamerV3` flow (also NNX-based). Sub-module instantiation uses `nnx.Rngs(init_key)` exactly as `train.py:784` does for the current Dreamer. The interleave between env-step and gradient-step is controlled by `Ratio` from `utils.py`, exactly as in sheeprl's main loop. No torch shadows.

### File Changes

> Sheeprl source lines below refer to `tmp/sheeprl/sheeprl/` paths. JAX/Flax hazards listed per file are the same hazards the project's `code-reviewer` agent enforces. **The current `src/models/dreamer_v3_*.py` and `configs/models/dreamer_v3*.yaml` are NOT modified.**

#### CREATE `src/algorithms/__init__.py` (new, empty file)

Marks `src/algorithms` as a package. Empty file with a single comment line is fine.

#### CREATE `src/algorithms/dreamer_srl/__init__.py` (new, ~10 lines)

Re-export the public API used by top-level `train.py`:

```python
"""dreamer-srl: sheeprl-replicated DreamerV3 in JAX/Flax."""
from src.algorithms.dreamer_srl.train import build_state, one_train_step, collect_step
from src.algorithms.dreamer_srl.agent import build_agent
from src.algorithms.dreamer_srl.buffers import SequentialReplayBuffer
__all__ = ["build_state", "one_train_step", "collect_step", "build_agent", "SequentialReplayBuffer"]
```

#### CREATE `src/algorithms/dreamer_srl/utils.py` (new, ~250 lines)

| Symbol | Sheeprl source | JAX/Flax translation hazard |
|---|---|---|
| `symlog(x) = sign(x) * log(|x|+1)`, `symexp(x) = sign(x) * (exp(|x|)-1)` | `tmp/sheeprl/sheeprl/utils/utils.py` (functions `symlog`, `symexp`) | None — pure elementwise. Use `jnp.sign`, `jnp.log1p`. Will be JIT-traced — make sure no Python branching on values. |
| `init_weights(in_features, out_features, key) -> kernel` (Hafner truncated normal, fan-avg) | `tmp/sheeprl/sheeprl/algos/dreamer_v3/utils.py:142-166` | PyTorch's `nn.init.trunc_normal_` is in-place mutation; in Flax/JAX we return a sampled array. The `0.87962566103423978` divisor (the Hafner constant — corrects truncated-normal variance back to unit variance) is hard-coded. **Use the full-precision constant `0.87962566103423978`, NOT the truncated `0.8796` from existing `src/models/dreamer_v3_util.py:hafner_init`** (code-reviewer aux observation). The shape arg must be static for JIT. |
| `uniform_init_weights(given_scale)(in_features, out_features, key) -> kernel` (cascade fix #27 when `given_scale=0.0`) | `tmp/sheeprl/sheeprl/algos/dreamer_v3/utils.py:170-186` | Same as above. `given_scale=0.0` gives `limit=0`, hence `uniform(-0, 0) = 0` everywhere — i.e. **zero-init**. Document this in a docstring so a reader understands "zero-init" is encoded as `given_scale=0.0` and not as a separate code path. |
| `compute_lambda_values(rewards, values, continues, lmbda=0.95)` | `tmp/sheeprl/sheeprl/algos/dreamer_v3/utils.py:66-77` | **Tightened 2026-05-12 (professor-rl-bayesian-dl #9)**. Backward TD(λ) recursion. The mathematical contract:<br>$$G^{\lambda}_t = r_t + c_t \cdot \bigl[(1-\lambda)\, v_{t+1} + \lambda\, G^{\lambda}_{t+1}\bigr], \quad G^{\lambda}_T = v_T$$<br>where `continues = mask * gamma` is **pre-multiplied by the caller** (sheeprl `dreamer_v3.py:284-289` passes `continues[1:] * gamma`). The function's signature must therefore take `continues` ALREADY multiplied by γ — do NOT bake γ into `compute_lambda_values` itself.<br><br>**Bootstrap and trim contract** (subtle):<br>• Bootstrap: `vals[0] = values[-1:]` — the **terminal value seed** is `values[-1]`, NOT the next-step value at the boundary.<br>• Returned tensor length = `len(continues)` (= horizon `T`), NOT `len(continues) + 1`. JAX `lax.scan(reverse=True)` naturally produces `T+1` outputs (initial carry + each scanned step); the trailing `[:-1]` slice is essential to match sheeprl's return shape.<br>• Sheeprl's `compute_lambda_values` body returns a tensor of length `T`; the caller at `dreamer_v3.py:284-289` then further `[:-1]`-slices the result to align with `baseline = predicted_values[:-1]`. Do not double-slice.<br><br>JAX equivalent: `jax.lax.scan` with `reverse=True` over the sequence axis. Inside the scan, carry is a single scalar value `vals[-1]`; at each step emit `interm[t] + continues[t] * lmbda * carry`. The final stacked output (after the trailing `[:-1]` trim) is the returns sequence. Length must be static for tracing — pass `horizon` as a compile-time int.<br><br>**Add a numpy Python-loop reference implementation to the docstring** and assert the JAX `lax.scan` output matches to `1e-6` (Checkpoint 1). **Remove the v1 plan's claim that the existing `src/models/dreamer_v3_util.py:compute_lambda_values` is "line-for-line identical to sheeprl's"** — verify, do not assert. |
| `Moments` (running 5th/95th percentile EMA) — **pure-functional `flax.struct.dataclass`, NOT `nnx.Variable` mutation** | `tmp/sheeprl/sheeprl/algos/dreamer_v3/utils.py:40-63` | **Decision locked 2026-05-12 (code-reviewer 🔴 #8)**: `Moments` is implemented as a `@flax.struct.dataclass` with a **pure-functional `update` returning a new pytree**, NOT an NNX module with `nnx.Variable` mutation.<br><br>**DO NOT pattern-match on `src/models/dreamer_v3_util.py:Moments`** — that class uses `nnx.Variable` and mutates `self.low.value` in-place inside `update` (lines 172-173). The pure-functional `flax.struct.dataclass` form is required because it makes state flow explicit through `one_train_step`'s return value (better JIT correctness, no fragile NNX-update nesting).<br><br>**Correct sheeprl-canonical form**:<br>```python<br>@flax.struct.dataclass<br>class MomentsState:<br>    low: jnp.ndarray   # scalar, init 0.0<br>    high: jnp.ndarray  # scalar, init 0.0<br><br>def moments_update(<br>    state: MomentsState, x: jnp.ndarray,<br>    decay: float = 0.99, p_low: float = 0.05, p_high: float = 0.95,<br>    max_: float = 1.0,<br>) -> Tuple[MomentsState, jnp.ndarray, jnp.ndarray]:<br>    x_flat = x.ravel().astype(jnp.float32)<br>    low_p  = jnp.quantile(x_flat, p_low)<br>    high_p = jnp.quantile(x_flat, p_high)<br>    new_low  = decay * state.low  + (1 - decay) * low_p<br>    new_high = decay * state.high + (1 - decay) * high_p<br>    invscale = jnp.maximum(1.0 / max_, new_high - new_low)<br>    return MomentsState(low=new_low, high=new_high), new_low, invscale<br>```<br>Return signature is `(new_moments_state, offset, invscale)` — three values, NOT `(low, invscale)` (the existing `dreamer_v3_util.py` return).<br><br>`torch.quantile` → `jnp.quantile`. **Hazard:** `fabric.all_gather` is sheeprl-specific (distributed training); we run single-device so just operate on the local batch — drop `all_gather`. Semantic consequence is per-rank-local percentiles, identical to sheeprl XS at single-rank.<br><br>**`moments_max = 1.0`** (sheeprl `dreamer_v3.yaml:134`, overrides class default `1e8`): the invscale **floor is `1/max_ = 1.0`**, meaning when the λ-spread is tiny the advantage is NOT re-scaled larger than the spread itself. Document this in `agent_xs.yaml` comment — see Risks §11 below. |
| `class Ratio` (replay-ratio scheduler with fractional-debt accumulator) | `tmp/sheeprl/sheeprl/utils/utils.py:259-291` | Stateful Python object. **Hazard:** if used inside a JIT'd function, the fractional `_prev` accumulator becomes a stale captured value. Solution: keep `Ratio` as a **Python-side** (uncompiled) object, called outside `jax.jit` boundaries — exactly as the current `src/models/dreamer_v3_util.py` `Ratio` is used (`train.py:789`). Document this in a docstring. The semantics is "integer number of gradient steps owed at this env step"; bookkeep with floats internally to avoid drift. |
| `prepare_obs(obs_dict, num_envs) -> Dict[str, jnp.ndarray]` | `tmp/sheeprl/sheeprl/algos/dreamer_v3/utils.py:80-91` | **Replicates sheeprl `utils.py:171-183` MLP path** (professor-rl-bayesian-dl #4). For each MLP obs key, write `jnp.asarray(v).reshape(1, num_envs, -1)`. **The leading `T=1` axis is mandatory** — the RSSM consumes `[T, B, ...]` and the encoder + RSSM expect that contract. If skipped, the encoder will silently broadcast and produce a wrong-shape embedding. Sheeprl's image-rescale code path (uint8 → [-0.5, 0.5]) is not needed here — our env is vector-obs. |

**JAX/Flax discipline for this file:**
- Every function pure (no globals, no in-place mutation of arrays).
- `@flax.struct.dataclass` for any stateful object that crosses a JIT boundary (`Moments`).
- `Ratio` stays Python-side (not crossed by JIT).
- `init_weights` returns kernel arrays; do not stash an RNG inside the function — pass `key` explicitly and split inside if needed.

**Reuse from existing code?** The current `src/models/dreamer_v3_util.py` has a `Ratio`, a `compute_lambda_values`, a `Moments`, and a `hafner_init`. **Do NOT import from there** — replicate the sheeprl implementation directly, to avoid coupling and to ensure the cascade-tainted current implementation does not contaminate `dreamer-srl`. The current `Ratio` is line-for-line identical to sheeprl's so the duplication is small. The current `compute_lambda_values` **must be verified against sheeprl's line-by-line** before relying on either — the v1 plan asserted line-for-line identity; this v2 says verify, do not assert. The current `Moments` uses `nnx.Variable` mutation (wrong pattern — see `Moments` row above) and the current `hafner_init` uses the truncated constant `0.8796` instead of `0.87962566103423978` (wrong precision — see `init_weights` row above). Neither is bit-identical to sheeprl; both must be replicated fresh.

#### CREATE `src/algorithms/dreamer_srl/buffers.py` (new, ~200 lines)

A bare-minimum **in-memory** SequentialReplayBuffer, one ring per parallel env.

| Symbol | Sheeprl source | JAX/Flax translation hazard |
|---|---|---|
| `class SequentialReplayBuffer(buffer_size, n_envs, obs_keys, obs_shapes, action_dim)` | `tmp/sheeprl/sheeprl/data/buffers.py` — `SequentialReplayBuffer` class (≈lines 600–1000; consult walkthrough `buffers.md`) | **No JIT-side state**: the buffer is Python-side, holds numpy or device-resident jnp arrays in a ring. The `add(step_data)` and `sample(batch_size, seq_len, n_samples) -> dict_of_jnp_arrays` calls happen outside the JIT region. `step_data` shape is `[1, n_envs, ...]`. Sampled batch shape is `[seq_len, batch_size, ...]` per key. |
| `class EnvIndependentReplayBuffer(buffer_size, n_envs, buffer_cls=SequentialReplayBuffer)` | `tmp/sheeprl/sheeprl/data/buffers.py` — `EnvIndependentReplayBuffer` | Wrapper that owns one `SequentialReplayBuffer` per env and uses `np.bincount` to spread `n_samples` across envs. **Important rule from the walkthrough**: "sample contiguous length-L windows that DO NOT respect episode boundaries (a chunk can straddle a done), and the trainer relies on the stored `is_first` flag to reset RSSM recurrent state at the correct intra-chunk step." dreamer-srl MUST preserve this exact contract. |

**Explicitly NOT implemented** (per non-goals): `EpisodeBuffer`, memmap mode, `prioritize_ends`, the `from_numpy` toggle (we always pass JAX arrays). The buffer is in-memory only.

**Storage shape decision** (single decision, documented here so `developer` does not silently pick wrong): keep the buffer as **numpy ring arrays on CPU**, convert to `jnp` on sample. This mirrors sheeprl exactly and avoids GPU-pressure from a multi-GB replay. (Our current `src/models/dreamer_v3_trainer.py:ReplayBuffer` supports `buffer_device: "gpu"` for zero-copy — we do NOT inherit that knob; sheeprl's reference does not have it, and replicating sheeprl is the goal.) Add this as a YAML key: `agent.buffer_device: cpu` (mandatory; no fallback).

**Required transition keys** (must match sheeprl exactly; cf. `dreamer_v3.py:539-546`):
- `observations` (dict of MLP keys; for us a single key, see config section below)
- `actions` (one-hot for discrete: shape `[1, n_envs, action_dim]`)
- `rewards` (shape `[1, n_envs, 1]`)
- `terminated` (uint8 `[1, n_envs, 1]`)
- `truncated` (uint8 `[1, n_envs, 1]`)
- `is_first` (uint8 `[1, n_envs, 1]`)

**Reuse from existing code?** No. The current `src/models/dreamer_v3_trainer.py` `ReplayBuffer` mixes sampling concerns with mixture-sampling positive-reward sub-buffers and a buffer-device toggle. Re-implement clean.

#### CREATE `src/algorithms/dreamer_srl/agent.py` (new, ~700 lines)

The biggest file. Translate sheeprl's `agent.py` (1236 lines) and the `LayerNormGRUCell` + `MLP` + `LayerNorm` machinery from `tmp/sheeprl/sheeprl/models/models.py` (the relevant blocks; the rest of `models.py` is CNN, which we skip).

| Symbol | Sheeprl source | JAX/Flax translation hazard |
|---|---|---|
| `class LayerNorm(eps=1e-3)` | `tmp/sheeprl/sheeprl/models/models.py` — `LayerNorm` wrapper | One-liner: `nnx.LayerNorm(num_features, epsilon=1e-3, rngs=rngs)`. **Locked decision (2026-05-12)**: dreamer-srl is implemented in `flax.nnx` to match the existing JAX Dreamer (`src/models/dreamer_v3_nnx.py`). Rationale: side-by-side diff against current Dreamer reads as a true sheeprl-vs-ours diff, not a dialect diff. Mutable-state pitfalls (carry slots, EMA target params, replay pointers) handled per the `code-reviewer` JAX/NNX correctness conventions in `docs/environment/ENVIRONMENT_SUMMARY.md`. |
| `class LayerNormGRUCell` (with cascade fix #28 — reset gate applied to candidate) | `tmp/sheeprl/sheeprl/models/models.py:351-403` — `LayerNormGRUCell` | **⚠️ CRITICAL: ONE Linear + ONE LayerNorm fused-gate, NOT two of each** (code-reviewer 🔴 #2). Sheeprl uses **one** `nn.Linear(input_size + hidden_size, 3 * hidden_size)` and **one** `LayerNorm(3 * hidden_size)` applied to the joint projection of `[hx, input]`. **DO NOT pattern-match on `src/models/dreamer_v3_nnx.py:18-70:LayerNormGRUCell`** — that class uses **two** `nnx.Linear` (`dense_ih`, `dense_hh`) + **two** `nnx.LayerNorm` (PyTorch `nn.GRUCell` convention with separate input-to-hidden and hidden-to-hidden projections). Different parameter count, different LayerNorm coupling — silently breaks Checkpoint 8.<br><br>**Correct fused-gate code sketch (sheeprl-canonical)**:<br>```python<br>class LayerNormGRUCell(nnx.Module):<br>    def __init__(self, input_size, hidden_size, rngs):<br>        self.linear = nnx.Linear(<br>            input_size + hidden_size, 3 * hidden_size,<br>            use_bias=True, kernel_init=hafner_init(), rngs=rngs)<br>        self.layer_norm = nnx.LayerNorm(<br>            3 * hidden_size, epsilon=1e-3, rngs=rngs)<br>    def __call__(self, x_input, hx):<br>        x = jnp.concatenate([hx, x_input], axis=-1)<br>        x = self.layer_norm(self.linear(x))<br>        # ORDER: (reset, cand, update) — sheeprl models.py:399<br>        reset, cand, update = jnp.split(x, 3, axis=-1)<br>        reset = jax.nn.sigmoid(reset)<br>        cand  = jnp.tanh(reset * cand)              # cascade fix #28<br>        update = jax.nn.sigmoid(update - 1.0)       # keep-old-state bias<br>        return update * cand + (1 - update) * hx<br>```<br><br>**Chunk order is `(reset, cand, update)`**, NOT `(reset, update, cand)` — code-reviewer #2. Sheeprl `models.py:399`: `reset, cand, update = torch.chunk(x, 3, -1)`. Bit-identical replication of a pretrained sheeprl checkpoint (Checkpoint 8) requires matching the column order so a transferred `linear.weight` slice maps to the right gate.<br><br>**Hazard:** PyTorch `LayerNormGRUCell` consumes `(input, hidden)` and returns `next_hidden`; JAX form is `(carry, x) -> (new_carry, new_carry)` for `lax.scan` compatibility. **Hazard:** vmap axes — the cell operates on `[B, H]`. In NNX, batch axis is handled implicitly (`nnx.Linear` broadcasts over leading dims); over the sequence axis the trainer uses `nnx.scan` (or `lax.scan` after `nnx.split`). Do not double-vmap. |
| `class MLP(input_dim, output_dim, hidden_sizes, activation=SiLU, norm=LayerNorm)` | `tmp/sheeprl/sheeprl/models/models.py` — `MLP` | Plain MLP. Terminal Linear is the target of `uniform_init_weights(scale)` for output heads. Activation: `nn.silu` / `jax.nn.silu`. Each hidden layer = `Dense -> LayerNorm -> SiLU`. **Hazard:** Flax `Dense` defaults to LeCun-normal kernel init — we override to Hafner truncated-normal via `kernel_init=hafner_trunc_normal(in, out)`. Pass `use_bias=False` when LayerNorm follows (`bias` is redundant — sheeprl does this at `models.py` and `agent.py` Line 1026 with `layer_args={"bias": representation_ln_cls == nn.Identity}`). |
| `class MLPEncoder(input_dims, output_dim, mlp_layers, dense_units, layer_norm, activation)` | `tmp/sheeprl/sheeprl/models/models.py` — `MLPEncoder` (the dict-encoder branch) | For us, the encoder is a single-key (`state`) MLP. `output_dim` derived from `dense_units` (256 in XS). |
| `class MLPDecoder(latent_state_size, output_dims, mlp_layers, dense_units, layer_norm, activation)` | `tmp/sheeprl/sheeprl/models/models.py` — `MLPDecoder` | Symmetric. Output Linear gets `uniform_init_weights(1.0)`. |
| `class MultiEncoder(cnn_encoder=None, mlp_encoder)` / `class MultiDecoder` | `tmp/sheeprl/sheeprl/models/models.py` | For us the CNN branch is always `None`. Keep the dict-aware structure so future image-obs is one config-line away. |
| `class RecurrentModel(input_size, recurrent_state_size, dense_units, layer_norm)` | `tmp/sheeprl/sheeprl/algos/dreamer_v3/agent.py:281-341` | Sheeprl wraps the GRU in a pre-projection MLP (one dense_units layer of LayerNorm+SiLU) then `LayerNormGRUCell`. The pre-projection is mandatory (Hafner spec). |
| `class RSSM(recurrent_model, representation_model, transition_model, distribution_cfg, discrete=32, unimix=0.01, learnable_initial_recurrent_state=True)` | `tmp/sheeprl/sheeprl/algos/dreamer_v3/agent.py:344-499` | Cascade fix #30 lives in `build_agent` where `representation_model` and `transition_model` are constructed as `MLP(hidden_sizes=[hidden_size])` — one hidden layer, not bare Linear. RSSM exposes `dynamic(posterior, recurrent_state, action, embedded_obs, is_first)`, `imagination(prior, recurrent_state, action)`, `get_initial_states(batch_shape)`.<br><br>**`is_first` handling — arithmetic-mask form, THREE quantities reset** (code-reviewer 🔴 #3; sheeprl `agent.py:423-429`). See Training-loop semantics §S4 above for the full rationale. The implementation MUST use:<br>```python<br>def dynamic(self, posterior, recurrent_state, action, embedded_obs, is_first):<br>    # Shapes: is_first [B, 1], action [B, A], posterior [B, S, D], recurrent_state [B, H_rec]<br>    action = (1.0 - is_first) * action                              # S4: zeroed<br>    initial_recurrent, initial_posterior = self.get_initial_states((batch_size,))<br>    recurrent_state = (1.0 - is_first) * recurrent_state + is_first * initial_recurrent<br>    posterior = posterior.reshape(*posterior.shape[:-2], -1)        # [B, S*D]<br>    initial_posterior = initial_posterior.reshape(*posterior.shape) # [B, S*D]<br>    posterior = (1.0 - is_first) * posterior + is_first * initial_posterior<br>    # ... rest of dynamic (recurrent step, posterior compute via embedded_obs)<br>```<br>**Use the arithmetic-mask form `(1 - is_first) * x + is_first * init`, NOT `jnp.where(is_first[..., None], init, current)`** — they are numerically equivalent but sheeprl uses the arithmetic form and the buffer stores `is_first` as `[T, B, 1]` with the trailing singleton already baked in (see `buffers.py` storage shape contract), so no extra `[..., None]` broadcast is needed.<br><br>**Hazard:** the learnable initial recurrent state is a `nn.Parameter` (sheeprl) → NNX `self.initial_recurrent_state = nnx.Param(jnp.zeros((recurrent_state_size,)))` declared in `__init__`. The recurrent state is initialised by applying `tanh` to the parameter (`agent.py:392`), broadcast over the batch — do not skip the tanh.<br><br>**`get_initial_states` is deterministic — uses transition `mode`, NOT a sample, NO PRNG consumed** (code-reviewer 🔴 #4; sheeprl `agent.py:391-394, 593`):<br>```python<br>def get_initial_states(self, batch_shape):<br>    initial_recurrent = jnp.tanh(self.initial_recurrent_state.value)         # [H_rec]<br>    initial_recurrent = jnp.broadcast_to(<br>        initial_recurrent, (*batch_shape, recurrent_state_size))<br>    # Transition's MODE (uniform-mix-applied softmax), NOT a sample. NO key consumed.<br>    prior_logits = self.transition_model(initial_recurrent)<br>    prior_logits = self._uniform_mix(prior_logits)<br>    initial_posterior = jax.nn.softmax(prior_logits, axis=-1)   # mode, not sample<br>    return initial_recurrent, initial_posterior<br>```<br>Why deterministic: `get_initial_states` is called at every `is_first=1` step (start of every episode); a non-deterministic initial posterior would break the `is_first` reset's determinism. Sheeprl's `compute_stochastic_state(..., sample_state=False)` returns `.mode` (uniform-mix-applied softmax), not `.sample`. **DO NOT** pass a PRNG key to `get_initial_states`. |
| `_uniform_mix(logits)` (the 1% uniform mixture of categorical latent) | `tmp/sheeprl/sheeprl/algos/dreamer_v3/agent.py:437-449` | Pure tensor op. Hazard: `probs_to_logits` (a PyTorch util) is just `logits = log(probs)` — replicate with `jnp.log` plus small clipping for numerical safety. |
| `compute_stochastic_state(logits, discrete=32, sample=True)` — straight-through one-hot categorical sampler | (helper called in RSSM) | The straight-through trick: forward = one-hot sample; backward = `probs + (one_hot_sample - probs).stop_gradient()`. In JAX: `one_hot = jax.nn.one_hot(jax.random.categorical(...), num_classes); ste = probs + jax.lax.stop_gradient(one_hot - probs)`. **Hazard:** double-check the gradient flow direction — sheeprl uses `samples + (probs - probs.detach())`, which is **gradient through probs** with **value of samples**. Reading `tmp/sheeprl/sheeprl/utils/distribution.py` `OneHotCategoricalStraightThroughValidateArgs` confirms it: `self.has_rsample = True; def rsample(self, sample_shape) -> samples + (self.probs - self.probs.detach())`. Replicate verbatim. |
| `class Actor` (discrete only — `MinedojoActor` skipped) | `tmp/sheeprl/sheeprl/algos/dreamer_v3/agent.py:694-846` | Multi-headed for `MultiDiscrete` (we use single-`Discrete`, so single head). Each head outputs unnormalized logits; sampling uses unimix + categorical-with-straight-through (same as RSSM). `mlp_heads` get `uniform_init_weights(1.0)` (sheeprl `build_agent:1171`); the MLP body gets `init_weights` Hafner-trunc-normal. **Hazard:** action-clip / continuous-action paths — we skip continuous entirely (see non-goals). |
| `class Critic` (just an `MLP` with `bins=255` output) | `tmp/sheeprl/sheeprl/algos/dreamer_v3/agent.py` — critic is built in `build_agent:1154-1166` | A bare `MLP`, terminal Linear zero-init'd via `uniform_init_weights(0.0)` (sheeprl `build_agent:1172`). |
| `Player` — rollout-time function namespace (NOT a module) that carries `(h, z)` across env steps | `tmp/sheeprl/sheeprl/algos/dreamer_v3/agent.py:596-693` | **Decision locked 2026-05-12 (code-reviewer 🔴 #6)**: dreamer-srl `Player` is **interpretation (B) — a thin function namespace, NOT an NNX module with its own params**. The PyTorch Player holds its own deep-copies of encoder/RSSM/actor with weights **aliased** to the trainable modules. In JAX/Flax there is no aliasing — params live in a separate pytree.<br><br>**Implementation form** — `Player` is a set of free functions (`encode_obs`, `act`, `player_initial_state`) that take the current trainable `WorldModel` / `Actor` NNX modules as arguments. The `(h, z)` rollout state lives in `DreamerSrlState` as `player_recurrent_state` + `player_stochastic_state` + `player_action`. The `collect_step` function takes the current `WorldModel` and `Actor` modules (which already hold the trainable params via NNX). **No sync step needed; no aliasing; no deep-copy.**<br><br>```python<br># In train.py:<br>@flax.struct.dataclass<br>class DreamerSrlState:<br>    world_model: nnx.GraphState<br>    actor: nnx.GraphState<br>    critic: nnx.GraphState<br>    target_critic: nnx.GraphState<br>    world_opt_state: optax.OptState<br>    actor_opt_state: optax.OptState<br>    critic_opt_state: optax.OptState<br>    moments_state: MomentsState<br>    player_recurrent_state: jnp.ndarray   # [B, H_rec]<br>    player_stochastic_state: jnp.ndarray  # [B, S, D]<br>    player_action: jnp.ndarray            # [B, A]<br>    train_step: jnp.ndarray<br>    env_step: jnp.ndarray<br>```<br>Reject interpretation (A) where `Player` is a free-standing NNX module with its OWN params plus an `nnx.update(player_state, source_state)` sync step — that re-introduces aliasing risk and adds a sync cost for no benefit. |
| `class WorldModel(encoder, rssm, observation_model, reward_model, continue_model)` | `tmp/sheeprl/sheeprl/algos/dreamer_v3/agent.py` (used implicitly; `WorldModel` is a `nn.Module` container, defined ~earlier in agent.py) | Container `nnx.Module` with sub-modules as attributes (mirrors PyTorch `nn.Module`). |
| `def build_agent(actions_dim, is_continuous, cfg, obs_space)` | `tmp/sheeprl/sheeprl/algos/dreamer_v3/agent.py:935-1242` | The bulk constructor. **Critical**: the final `if cfg.algo.hafner_initialization` block at `agent.py:1170-1180` is **the bit-identity gate**. The implementer must translate every line of that block exactly. In particular:<br>• `critic.model[-1].apply(uniform_init_weights(0.0))` — cascade fix #27 critic.<br>• `world_model.reward_model.model[-1].apply(uniform_init_weights(0.0))` — cascade fix #27 reward.<br>• `rssm.transition_model.model[-1].apply(uniform_init_weights(1.0))` — non-zero scale=1 init on terminal Linear (NOT zero-init for these; the transition head is not a value/reward head).<br>• `rssm.representation_model.model[-1].apply(uniform_init_weights(1.0))`.<br>• `actor.mlp_heads.apply(uniform_init_weights(1.0))`.<br>• `world_model.continue_model.model[-1].apply(uniform_init_weights(1.0))`.<br>• decoder heads `apply(uniform_init_weights(1.0))`. |

**JAX/Flax discipline for this file:**
- All modules are `flax.nnx.Module` subclasses, instantiated with `rngs: nnx.Rngs` in `__init__` (same pattern as `src/models/dreamer_v3_nnx.py:18` `LayerNormGRUCell.__init__`). Each module's `__call__(self, x, *, key=None)` is pure with respect to its weights; weights live on `self` as `nnx.Linear`/`nnx.LayerNorm`/`nnx.Param` attributes. **EXCEPT for `LayerNormGRUCell`** — see warning in the row above; the existing `src/models/dreamer_v3_nnx.py:LayerNormGRUCell` uses TWO Linears + TWO LayerNorms (PyTorch convention) but sheeprl uses ONE + ONE (fused-gate convention). Re-write the cell clean per the sheeprl-canonical sketch.
- **PRNG threading**: every `__call__` that samples (RSSM, Actor) takes a `key` argument explicitly. Never grab an RNG from a module attribute or a global — split keys at the caller and pass down. `nnx.Rngs` is used only at module **construction** (Hafner init / zero-init), not at forward call.
- `vmap` axes: the model operates on `[T, B, ...]` shapes. The standard pattern is `nnx.scan` (or pure-functional `lax.scan` after `nnx.split` / `nnx.merge`) over `T` with batch axis `B` implicit. Avoid mixing scan and vmap in the same dimension.
- **`lax.scan` carry signature for RSSM dynamic learning** (code-reviewer 🔴 #1; sheeprl `dreamer_v3.py:205-244`). The scan body MUST follow this contract:
  - **Carry**: `(recurrent_state: [B, H_rec], posterior: [B, S, D])` — posterior shape is `[B, S, D]` UNFLATTENED, because the loss needs the unflattened form for KL computation. Sheeprl reshapes posterior internally via `posterior.view(*posterior.shape[:-2], -1)` only at line 428 (inside `dynamic`'s `is_first` masking — see S4 above).
  - **Scan input** (per-T slice): `(batch_action[t]: [B, A], embedded_obs[t]: [B, E], is_first[t]: [B, 1])`.
  - **Scan output** (per-T): `(recurrent_state, posterior, posterior_logits, prior_logits)` — `posterior` is collected unflattened at `[B, S, D]` for KL, then flattened to `[T, B, S*D]` for latent_states concatenation at sheeprl `dreamer_v3.py:245`.
  ```python
  def _rssm_dynamic_scan_body(carry, inputs):
      recurrent_state, posterior = carry          # [B, H_rec], [B, S, D]
      action, embed, is_first = inputs            # [B, A], [B, E], [B, 1]
      recurrent_state, posterior, prior, posterior_logits, prior_logits = \
          rssm.dynamic(posterior, recurrent_state, action, embed, is_first)
      new_carry = (recurrent_state, posterior)
      outputs = (recurrent_state, posterior, posterior_logits, prior_logits)
      return new_carry, outputs

  (final_recurrent, final_posterior), (recurrent_states, posteriors, posteriors_logits, priors_logits) = \
      nnx.scan(_rssm_dynamic_scan_body, ...)(initial_carry, (batch_actions, embedded_obs, is_first))
  ```
  Without this, the developer is likely to invent a different carry shape (e.g. pre-flattened posterior) and silently regress KL computation.
- **Pytree containers** (`WorldModel`, `RSSMState`): `WorldModel` is itself an `nnx.Module`; `RSSMState` (the recurrent-state carry) is a `@flax.struct.dataclass` so it can flow through `lax.scan` cleanly.
- **No mutation of pytrees in-place.** Every `update_*` returns a new pytree. NNX modules expose state via `nnx.split` / `nnx.merge` when JIT compilation needs explicit state separation (see `src/models/dreamer_v3_trainer.py` for the existing pattern).
- **Polyak / EMA target critic**: the target critic is a second `Critic` instance, updated by `nnx.update(target, jax.tree.map(lambda t, s: tau*s + (1-tau)*t, nnx.state(target), nnx.state(source)))`. Do NOT mutate `.kernel` in-place; do NOT alias parameters across the two instances.
- **JIT recompilation triggers**: any shape that should be static (`batch_size`, `sequence_length`, `horizon`, `num_envs`, `action_dim`, `recurrent_state_size`, `stochastic_size`, `discrete_size`) goes through the `Config.get_mandatory` path **at module construction time** (closed over as Python ints), not through array shapes at call time. Document this in each module's docstring.

**Reuse from existing code?** No direct imports from `src/models/dreamer_v3_nnx.py`. The current implementation diverges on cascade items #28 and #30, which is the whole reason we are re-implementing. Even `LayerNormGRUCell` in `src/models/modulated_layer_norm_gru_cell.py` carries a modulation injection point — re-write the cell clean.

#### CREATE `src/algorithms/dreamer_srl/loss.py` (new, ~250 lines)

| Symbol | Sheeprl source | JAX/Flax translation hazard |
|---|---|---|
| `class TwoHotEncoding(logits, bins=255, low=-20, high=20)` | `tmp/sheeprl/sheeprl/utils/distribution.py` — `TwoHotEncodingDistribution` (lines ~185-260; check walkthrough `distribution.md`) | **Corrected 2026-05-12 (math-reviewer 🔴 #1)**: bin grid is `linspace(-20, +20, 255)` in **symlog space** (sheeprl `distribution.py:237` stores `self.bins = torch.linspace(low, high, K)`). The `symexp` is applied lazily inside `mean` / `mode` via `transbwd` — NOT to the stored grid. `log_prob(target)`: push `target` through `symlog`, find bracketing symlog-space bins, take soft cross-entropy with `logits`. `mean(self)`: softmax over logits, dot with symlog-space bin centers, then `symexp` to return real-space scalar. **Hazard**: this is the load-bearing distribution. Verify against sheeprl line-by-line at Checkpoint 5 — and crucially, DO NOT store `self.bins = symexp(linspace(...))`, that breaks `log_prob`. |
| `SymlogDistribution.log_prob(target)` | `tmp/sheeprl/sheeprl/utils/distribution.py` — `SymlogDistribution` | `−0.5 * (mean − symlog(target))²` summed over event dims. |
| `MSEDistribution.log_prob(target)` | `tmp/sheeprl/sheeprl/utils/distribution.py` — `MSEDistribution` | `−0.5 * (mean − target)²` summed. |
| `BernoulliSafeMode.log_prob(target)` and `.mode`, **wrapped as `Independent(BernoulliSafeMode, 1)`** | `tmp/sheeprl/sheeprl/utils/distribution.py` — `BernoulliSafeMode`; sheeprl `dreamer_v3.py:200, 279` | Standard BCE-with-logits log-prob; mode is `(logits > 0).astype(float)`. **The continue distribution MUST be wrapped as `Independent(BernoulliSafeMode(logits), 1)`** (professor-rl-bayesian-dl #10) — re-interprets the trailing axis as the event dim, so `log_prob` returns `[T, B]` per (T,B) cell rather than `[T, B, 1]`. Same wrap applies in the imagination loop: `continues = Independent(BernoulliSafeMode(...), 1).mode`. For our case (single-dim Bernoulli, output_dim=1), the wrap reduces to a sum over a size-1 axis — same scalar — but the wrap is the only line that keeps the math right if output_dim ever changes. Same `dims`-control applies to the observation decoder: wrap decoder outputs as `MSEDistribution(mode, dims=1)` or `SymlogDistribution(mode, dims=1)` for our `[T, B, F]`-shape observations (sheeprl `dreamer_v3.py:186-193`). |
| `categorical_kl(post_logits, prior_logits)` | `tmp/sheeprl/sheeprl/utils/distribution.py` line 405 area — `register_kl(OneHotCategoricalStraightThroughValidateArgs, OneHotCategoricalStraightThroughValidateArgs)` | Analytic KL between two categorical distributions: `sum(softmax(p) * (log_softmax(p) - log_softmax(q)))`. Apply over the `discrete` axis, reshape `[T, B, 32, 32]`, sum the inner two axes after KL on the last. **Hazard:** `log_softmax` is numerically stabler than `log(softmax)`. |
| `reconstruction_loss(po, observations, pr, rewards, priors_logits, posteriors_logits, kl_dynamic=0.5, kl_representation=0.1, kl_free_nats=1.0, kl_regularizer=1.0, pc, continue_targets, continue_scale_factor=1.0) -> 6-tuple` | `tmp/sheeprl/sheeprl/algos/dreamer_v3/loss.py` (the whole file — 88 lines) | The KL-balancing trick: `dyn_loss = kl(post.detach, prior)` and `repr_loss = kl(post, prior.detach)`. In JAX, `.detach()` is `jax.lax.stop_gradient(...)`.<br><br>**Free-nats floor is PER-ELEMENT, applied BEFORE the mean** (math-reviewer 🟡 #7; §S8). Sheeprl `loss.py:100-101`:<br>```python<br># dyn_loss and repr_loss start as [T, B] KL tensors<br>dyn_loss  = kl_dynamic * jnp.maximum(dyn_loss,  kl_free_nats)    # element-wise<br>repr_loss = kl_representation * jnp.maximum(repr_loss, kl_free_nats)<br># Then reduce:<br>kl_total = (dyn_loss + repr_loss).mean()<br>```<br>**DO NOT** implement `max(mean(dyn_loss), 1.0)` — that's wrong.<br><br>**Continue target is `1 - terminated` (no γ multiplier)** — sheeprl `dreamer_v3.py:201` uses `continues_targets = 1 - data["terminated"]`. The sheeprl docstring says `(1 - dones) * γ` but the code is authoritative — follow the code (§S10).<br><br>The total is mean over `[T, B]`. Returns `(total, kl_mean, state_loss, reward_loss, observation_loss, continue_loss)`. |

**Critic and actor losses live in `train.py` (the module), not here**, because they are entangled with imagination state, target-critic params, and lambda-return computation.

**JAX/Flax discipline for this file:**
- Distributions are NOT classes with mutable state — they are dataclasses with `(logits|mean, ...)` fields and pure functions `log_prob(self, target) -> array`. No object-oriented `torch.distributions.Independent` wrapper; just sum over event dims explicitly.
- `jax.lax.stop_gradient` not `.detach()`.
- All `log_prob` returns shape `[T, B]` (reduced over event dims). The aggregate sums those into a scalar at the very end.

#### CREATE `src/algorithms/dreamer_srl/train.py` (new, ~500 lines — the algorithm's own train module, **not** the top-level `train.py`)

| Symbol | Sheeprl source | JAX/Flax translation hazard |
|---|---|---|
| `@flax.struct.dataclass class DreamerSrlState`: `world_model_params, actor_params, critic_params, target_critic_params, world_opt_state, actor_opt_state, critic_opt_state, moments_state, train_step, env_step` | (mostly state-tracking — not a single sheeprl symbol) | Pure pytree of mutable training state. Carry across `lax.scan` or across iterations of the top-level loop. |
| `build_state(rngs, cfg, obs_space, action_dim) -> DreamerSrlState` | `tmp/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:435-470` (the part of `main` that builds WM/actor/critic/target/optimisers/moments) | One-shot constructor. Calls `build_agent` from `agent.py`, runs `init_weights` and the Hafner-init block, instantiates the optax optimisers (replace sheeprl's `hydra.utils.instantiate(adam, ...)` with `optax.adam(lr=..., eps=...)`), allocates `Moments` state. **Mandatory config keys** — every one a `Config.get_mandatory` call, no defaults: `agent.world_model_lr` (1e-4 in XS), `agent.actor_lr` (8e-5), `agent.critic_lr` (8e-5), `agent.world_model_eps` (1e-8), `agent.actor_eps` (1e-5), `agent.critic_eps` (1e-5). |
| `one_train_step(state, batch, key) -> (new_state, metrics)` | `tmp/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:48-358` (the `train` function — 310 lines) | This is the big one. **First, apply S1 (force-set `is_first[0]=1`) and S2 (prepend-zero-action shift) on the batch** — see Training-loop semantics §S1, §S2. Then three sub-phases:<br><br>1. **Dynamic learning** (`dreamer_v3.py:106-200`): roll the RSSM forward over `[T, B]` using `lax.scan` per the carry signature in `agent.py` discipline above. Compute posteriors/priors, decode obs/reward/continue, call `reconstruction_loss`, take a gradient step on the world-model params. Free-nats floor is per-element BEFORE mean (§S8).<br><br>2. **Behaviour learning / actor** (`dreamer_v3.py:202-304`): from detached posteriors, imagine `horizon=15` steps forward using `lax.scan` (carry: `(prior, recurrent_state)`, output: latent state, action). Apply **S5 true-continue splice at imagination step 0**. Compute the discount mask:<br>```python<br>discount = jax.lax.stop_gradient(<br>    jnp.cumprod(continues * gamma, axis=0) / gamma<br>)   # disc[0] = 1; replicates sheeprl's `with torch.no_grad():` (dreamer_v3.py:259, 292-293)<br>```<br>Compute lambda-returns via `compute_lambda_values` — note the `[1:]` indexing on the call site (`dreamer_v3.py:284-289`: `compute_lambda_values(predicted_rewards[1:], predicted_values[1:], continues[1:] * gamma, lmbda)`). Compute baseline-normed advantage via `Moments` per §S7:<br>```python<br>moments_state, offset, invscale = moments_update(state.moments_state, lambda_values)<br>normed_lambda   = (lambda_values   - offset) / invscale<br>normed_baseline = (baseline_values - offset) / invscale<br>advantage = normed_lambda - normed_baseline   # offset cancels algebraically<br>```<br>For discrete actions, REINFORCE with discount-weighted entropy (§S6):<br>```python<br>objective = log_prob_actions * jax.lax.stop_gradient(advantage)   # [H, B]<br>policy_loss = -jnp.mean(<br>    discount[:-1] * (objective + ent_coef * entropy[:-1])<br>)<br>```<br>Take an actor gradient step.<br><br>3. **Critic** (`dreamer_v3.py:306-327`): two-hot NLL of `λ-target.detach()`, **plus** `−qv.log_prob(target_critic_value.detach())` — that's cascade fix #29, the slow-target EMA regulariser. **The critic regression target uses the UN-normalised `lambda_values`**, NOT the normed form (§S7). Discount-weighted (§S6):<br>```python<br>value_loss = jnp.mean(<br>    (-qv.log_prob(jax.lax.stop_gradient(lambda_values))<br>     - qv.log_prob(jax.lax.stop_gradient(target_critic_value)))<br>    * discount[:-1].squeeze(-1)<br>)<br>```<br>Take a critic gradient step. **Hazard:** the `[1:]` indexing on the lambda-values block (`dreamer_v3.py:251-256`) trims the first step which has no defined value-target; replicate exactly. **Hazard:** the `discount = cumprod(continues * gamma, axis=0) / gamma` computation is the unrolling-aware weighting; gradient must NOT flow through it — use `jax.lax.stop_gradient` (code-reviewer 🟢 #11), NOT `with jax.disable_jit()` or any other pattern. |
| `collect_step(state, env_state, env_params, key, player_state, iter_num, learning_starts) -> (new_env_state, new_player_state, step_data, key)` | `tmp/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:550-657` (the env-interaction block of `main`) | Carry the `(h, z)` rollout state; encode obs; **branch on `iter_num` vs `learning_starts`** (§S3) — for the first 1024 iterations sample uniform-random one-hot actions, after that sample via Actor. Step env via `jax_step` (our env, not gymnasium); record `step_data` dict for buffer.add. **Hazard:** the `is_first` handling: when an env terminates, the next step's `is_first` is 1 — track this across the parallel-env axis via `jnp.where(done, 1, 0)`. Sheeprl's `RestartOnException` is a robustness wrapper around gym — we drop it (our JAX env doesn't raise mid-step).<br><br>**Random-action prefill wiring** (§S3):<br>```python<br>def collect_step(..., iter_num, learning_starts):<br>    use_random = (iter_num <= learning_starts)  # static-when-iter_num-Python-side<br>    if use_random:<br>        # Uniform sample over the discrete action space; one-hot encode<br>        action_idx = jax.random.randint(<br>            sub_key, (num_envs,), 0, action_dim<br>        )<br>        action = jax.nn.one_hot(action_idx, action_dim)<br>    else:<br>        action = player_act(world_model, actor, obs, h, z, sub_key)<br>    # ... env step, buffer add<br>```<br>Note `iter_num <= learning_starts` is sheeprl's `dreamer_v3.py:604` convention. Also gate the gradient-step call in the top-level loop on `iter_num >= learning_starts` (sheeprl `dreamer_v3.py:706`). |
| `polyak_update(target_params, source_params, tau) -> new_target_params` | `tmp/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:720-726` (inlined; no helper function) | `target = tau * source + (1 - tau) * target`. Apply via `jax.tree.map`. Triggered every `per_rank_target_network_update_freq` (=1) gradient step. First call uses `tau=1.0` (hard copy); subsequent use `tau=0.02` (sheeprl XS default).<br><br>**Cadence: Polyak fires BEFORE `one_train_step`, not after** (code-reviewer 🟡 #7; sheeprl `dreamer_v3.py:720-726`). The top-level training loop:<br>```python<br>for _ in range(per_rank_gradient_steps):  # K updates per env step<br>    if state.train_step % target_update_freq == 0:<br>        tau = jnp.where(state.train_step == 0, 1.0, tau_default)<br>        target_params_new = jax.tree.map(<br>            lambda src, tgt: tau * src + (1 - tau) * tgt,<br>            nnx.state(critic), nnx.state(target_critic),<br>        )<br>        nnx.update(target_critic, target_params_new)<br>    state, metrics = one_train_step(state, batch[i], key)<br>    state = state.replace(train_step=state.train_step + 1)<br>```<br>The condition is on `cumulative_per_rank_gradient_steps`, NOT on `train_step` or `env_step`. **`train_step` semantics in `DreamerSrlState` = "cumulative gradient steps applied since training began"**, NOT "calls to `one_train_step` since restart". If `replay_ratio=K>1` runs multiple `one_train_step` calls per env step, the counter must be incremented K times. |

**JAX/Flax discipline for this file:**
- `one_train_step` is `@jax.jit`-able. All inputs (state, batch, key) and outputs (new_state, metrics) are pytrees of jnp arrays. Python-side flow control (the three sub-phases) is fine — JIT traces it once at the right shapes.
- `optax` for optimisers — `optax.adam(lr, eps=1e-8)` replaces `hydra.utils.instantiate(cfg.algo.world_model.optimizer)`. Use `optax.chain(optax.clip_by_global_norm(1000.0), optax.adam(...))` for the world-model `clip_gradients: 1000.0` and `optax.chain(optax.clip_by_global_norm(100.0), optax.adam(...))` for actor/critic (`100.0`). Also pass `weight_decay=0` explicitly to optax (sheeprl `dreamer_v3.yaml` optimizer blocks set `weight_decay: 0`).
- **PRNG sub-key threading** (code-reviewer 🟡 #5). Per `one_train_step` invocation:
  - T sub-keys for RSSM `compute_stochastic_state` during dynamic learning (carried INSIDE the scan via per-step keys passed as scan inputs).
  - H+1 sub-keys for prior categorical during imagination.
  - H+1 sub-keys for actor categorical during imagination.
  - Total ≈ 2(H+1) + T ≈ 30 + 64 = ~94 sub-keys per gradient step at XS settings.
  ```python
  key, rssm_key, img_prior_key, img_actor_key = jax.random.split(key, 4)
  rssm_keys       = jax.random.split(rssm_key,       T)          # [T, 2]; scan INPUT
  img_prior_keys  = jax.random.split(img_prior_key,  H + 1)
  img_actor_keys  = jax.random.split(img_actor_key,  H + 1)
  # These keys-per-step arrays are scan INPUTS, NOT carries.
  ```
  **Do NOT** reuse the same key inside the scan body (would give all scan steps the same sample stream — Z3-class bug). **Do NOT** use a Python-side loop with a counter (forces JIT retrace on every step). Never carry a key in `state` — pass it as a separate argument so JIT does not retrace on each call.
- **NNX**, matching the existing JAX Dreamer scaffolding (`src/models/dreamer_v3_nnx.py`). Keep `dreamer-srl` `flax.nnx` + `optax` + `jax`. Rationale: side-by-side diff against the current Dreamer reads as a true sheeprl-vs-ours diff, not a Flax-dialect diff. Decision locked 2026-05-12 (was Risks §1; now RESOLVED).
- **`Ratio` stays Python-side** (uncompiled). The top-level `train.py` calls `ratio(env_step) -> num_grad_steps`, then runs `one_train_step` in a Python loop that count of times.

#### CREATE `configs/models/dreamer_srl/` family (new directory, three files)

Two-tier config: one model file holding the network sizes / training hyperparameters (the dreamer-srl analog of `configs/models/dreamer_v3/dreamer_v3.yaml`), one experiment file pointing at the food-only task. We DO NOT create per-stage curriculum configs in this plan — parity gate is food-only only.

##### CREATE `configs/models/dreamer_srl/agent_xs.yaml` (new, ~70 lines)

Direct YAML transcription of `tmp/sheeprl/sheeprl/configs/algo/dreamer_v3_XL.yaml` + `dreamer_v3_XS.yaml` overrides. Flattened to single-document YAML (no Hydra). Every key gets fetched via `Config.get_mandatory` in the trainer — no fallbacks.

```yaml
# dreamer-srl agent config — matches sheeprl dreamer_v3_XS defaults exactly.
# Source: tmp/sheeprl/sheeprl/configs/algo/dreamer_v3_XL.yaml (defaults)
#       + tmp/sheeprl/sheeprl/configs/algo/dreamer_v3_XS.yaml (overrides)
# DO NOT add new keys without a corresponding sheeprl reference line.
#
# Per-rank-* keys: sheeprl uses "per_rank_*" as a distributed-training prefix.
# We run single-device so per_rank=global. Keep sheeprl names verbatim for
# parity (code-reviewer 🟡 #10 decision: verbatim sheeprl names, document
# the prefix meaning here).

# === Distribution config (top-level, matches sheeprl distribution/default.yaml) ===
# professor-rl-bayesian-dl #5
distribution:
  type: "auto"                    # sheeprl distribution/default.yaml; auto → "discrete" for our Discrete env
  validate_args: false            # sheeprl distribution/default.yaml:1

agent:
  algorithm: "dreamer-srl"

  # === Training recipe ===
  gamma: 0.996996996996997        # sheeprl dreamer_v3_XL.yaml:11
  lmbda: 0.95                     # sheeprl dreamer_v3_XL.yaml:12
  horizon: 15                     # sheeprl dreamer_v3_XL.yaml:13
  replay_ratio: 1                 # sheeprl dreamer_v3_XL.yaml:16
  learning_starts: 1024           # sheeprl dreamer_v3_XL.yaml:17
                                  # Counted in ITERATIONS (= policy_step / policy_steps_per_iter
                                  # = policy_step / (num_envs * action_repeat) = policy_step / 4).
                                  # Off-by-one: prefill = learning_starts - 1 (sheeprl dreamer_v3.py:557).
                                  # Wired into BOTH collection branch (random-action prefill) AND
                                  # gradient-step gate (sheeprl dreamer_v3.py:604-617, 706). See §S3.
  per_rank_pretrain_steps: 0      # sheeprl dreamer_v3_XL.yaml:18
  per_rank_sequence_length: 64    # sheeprl exp config picks 64; mandatory here
  per_rank_batch_size: 16         # sheeprl exp default; mandatory here
  unimix: 0.01                    # sheeprl dreamer_v3_XL.yaml:40
  hafner_initialization: true     # sheeprl dreamer_v3_XL.yaml:41

  # === Network sizes — XS profile ===
  dense_units: 256                # sheeprl dreamer_v3_XS.yaml:5
  mlp_layers: 1                   # sheeprl dreamer_v3_XS.yaml:6 — encoder/decoder/reward/continue/critic/actor MLP hidden-layer count
  dense_act: "silu"               # sheeprl dreamer_v3_XL.yaml:38 (torch.nn.SiLU → jax.nn.silu)
  layer_norm_eps: 1.0e-3          # sheeprl dreamer_v3_XL.yaml:34-35

  # === World model ===
  world_model:
    discrete_size: 32             # sheeprl dreamer_v3_XL.yaml:45
    stochastic_size: 32           # sheeprl dreamer_v3_XL.yaml:46
    kl_dynamic: 0.5               # sheeprl dreamer_v3_XL.yaml:47
    kl_representation: 0.1        # sheeprl dreamer_v3_XL.yaml:48
    kl_free_nats: 1.0             # sheeprl dreamer_v3_XL.yaml:49
    kl_regularizer: 1.0           # sheeprl dreamer_v3_XL.yaml:50
    continue_scale_factor: 1.0    # sheeprl dreamer_v3_XL.yaml:51
    clip_gradients: 1000.0        # sheeprl dreamer_v3_XL.yaml:52
    decoupled_rssm: false         # sheeprl dreamer_v3_XL.yaml:53 — explicit for transparency (Risk #8 out-of-scope)
    learnable_initial_recurrent_state: true   # sheeprl dreamer_v3_XL.yaml:54
    recurrent_state_size: 256     # sheeprl dreamer_v3_XS.yaml:11
    transition_hidden_size: 256   # sheeprl dreamer_v3_XS.yaml:13 (cascade fix #30 — non-zero!)
    representation_hidden_size: 256   # sheeprl dreamer_v3_XS.yaml:15 (cascade fix #30 — non-zero!)
    world_model_lr: 1.0e-4        # sheeprl dreamer_v3_XL.yaml:112
    world_model_eps: 1.0e-8       # sheeprl dreamer_v3_XL.yaml:113
    world_model_weight_decay: 0.0 # sheeprl dreamer_v3_XL.yaml — AdamW default

    # === Reward head (sheeprl dreamer_v3_XL.yaml:99-101) ===
    # professor-rl-bayesian-dl #7
    reward_model:
      bins: 255                   # sheeprl dreamer_v3_XL.yaml:100
      dense_act: "silu"           # inherits algo.dense_act
      mlp_layers: 1               # XS — same as algo.mlp_layers
      dense_units: 256            # XS — same as algo.dense_units
      layer_norm: true            # inherits algo.mlp_layer_norm
    reward_low: -20.0             # cascade fix #2 — symlog-space bin grid
    reward_high: 20.0             # cascade fix #2 — symlog-space bin grid

    # === Continue head (sheeprl dreamer_v3_XL.yaml:103-108) ===
    # professor-rl-bayesian-dl #7
    discount_model:
      learnable: true             # sheeprl dreamer_v3_XL.yaml:104
      dense_act: "silu"           # sheeprl dreamer_v3_XL.yaml:105
      mlp_layers: 1               # XS — 1 hidden layer
      dense_units: 256            # XS — same as algo.dense_units
      layer_norm: true

  # === Actor ===
  actor:
    ent_coef: 3.0e-4              # sheeprl dreamer_v3_XL.yaml:119
    # Std params (sheeprl dreamer_v3_XL.yaml:120-122). Unused on discrete path
    # but kept for signature parity with Actor.__init__ (professor-rl-bayesian-dl #6).
    init_std: 2.0                 # sheeprl dreamer_v3_XL.yaml:122 (unused on discrete path)
    min_std: 0.1                  # sheeprl dreamer_v3_XL.yaml:120
    max_std: 1.0                  # sheeprl dreamer_v3_XL.yaml:121
    action_clip: 1.0              # sheeprl dreamer_v3_XL.yaml:129
    clip_gradients: 100.0         # sheeprl dreamer_v3_XL.yaml:127
    actor_lr: 8.0e-5              # sheeprl dreamer_v3_XL.yaml:141
    actor_eps: 1.0e-5             # sheeprl dreamer_v3_XL.yaml:142
    actor_weight_decay: 0.0       # sheeprl dreamer_v3_XL.yaml — AdamW default
    moments_decay: 0.99           # sheeprl dreamer_v3_XL.yaml:133
    moments_max: 1.0              # sheeprl dreamer_v3_XL.yaml:134
                                  # NOT the upstream default 1e8 — XL deliberately overrides.
                                  # Effect: invscale floor is 1/max_ = 1.0, so when the lambda-value
                                  # spread is tiny the advantage is NOT re-scaled larger than the
                                  # spread itself. See professor-rl-bayesian-dl #11 + §S7.
    moments_percentile_low: 0.05  # sheeprl dreamer_v3_XL.yaml:136
    moments_percentile_high: 0.95 # sheeprl dreamer_v3_XL.yaml:137

  # === Critic ===
  critic:
    bins: 255                     # sheeprl dreamer_v3_XL.yaml:153
    clip_gradients: 100.0         # sheeprl dreamer_v3_XL.yaml:154
    tau: 0.02                     # sheeprl dreamer_v3_XL.yaml:152 (Polyak EMA)
    per_rank_target_network_update_freq: 1   # sheeprl dreamer_v3_XL.yaml:151
    critic_lr: 8.0e-5             # sheeprl dreamer_v3_XL.yaml:158
    critic_eps: 1.0e-5            # sheeprl dreamer_v3_XL.yaml:159
    critic_weight_decay: 0.0      # sheeprl dreamer_v3_XL.yaml — AdamW default
    critic_low: -20.0             # symmetric with reward bins (cascade fix #2; symlog-space)
    critic_high: 20.0

  # === Player (sheeprl exposes some sizes via cfg.player) ===
  player:
    discrete_size: 32             # sheeprl-implicit — inherits world_model.discrete_size

  # === Encoder / decoder keys ===
  mlp_keys:
    encoder: ["state"]
    decoder: ["state"]
  cnn_keys:
    encoder: []
    decoder: []

  # === Buffer ===
  buffer_size: 1_000_000          # sheeprl default
  buffer_device: "cpu"            # mandatory — see buffers.py design note
```

**MLP sizing table — "which network reads which `mlp_layers` / `dense_units`"** (professor-rl-bayesian-dl #8). For sheeprl XS profile (`mlp_layers=1`, `dense_units=256`):

| Network | `mlp_layers` consumed | hidden_dim | terminal init scale | comment |
|---|---|---|---|---|
| Encoder | top-level (1) | top-level (256) | uniform_init_weights(1.0) | `MLPEncoder`, single MLP key `state` |
| Decoder | top-level (1) | top-level (256) | uniform_init_weights(1.0) | `MLPDecoder` |
| Recurrent pre-proj | hardcoded (1) | top-level (256) | Hafner init | The `RecurrentModel` MLP before the GRU |
| Transition (prior) | hardcoded (1) | `transition_hidden_size` (256) | uniform_init_weights(1.0) | Cascade fix #30 |
| Representation (post) | hardcoded (1) | `representation_hidden_size` (256) | uniform_init_weights(1.0) | Cascade fix #30 |
| Reward head | `world_model.reward_model.mlp_layers` (1) | `world_model.reward_model.dense_units` (256) | **uniform_init_weights(0.0)** | Cascade fix #27 — zero-init |
| Critic | top-level (1) | top-level (256) | **uniform_init_weights(0.0)** | Cascade fix #27 — zero-init |
| Continue head | `world_model.discount_model.mlp_layers` (1) | `world_model.discount_model.dense_units` (256) | uniform_init_weights(1.0) | |
| Actor trunk | top-level (1) | top-level (256) | uniform_init_weights(1.0) | Heads get `uniform_init_weights(1.0)` (sheeprl `build_agent:1171`) |

##### CREATE `configs/models/dreamer_srl/01_food_only.yaml` (new, ~30 lines — points at the existing env config + agent config)

```yaml
# dreamer-srl parity-gate experiment config.
# Composes the food-only NoPred env (5×5, no predator, 500-step truncation)
# with the dreamer-srl XS agent config.

defaults_from: configs/experiment/dreamer_curriculum/01_food_only.yaml
# (or whatever defaults-merging mechanism src/utils/config.py uses;
#  developer confirms by reading get_default_config() + cfg.merge() pattern
#  documented in docs/develop/active/diagnosis/sheeprl_drop_in_test.md
#  Implementation Report §1 — same merge dance the sheeprl bridge uses.)

agent_config: configs/models/dreamer_srl/agent_xs.yaml

training:
  num_envs: 4                     # match the sheeprl jzgkcep4 run for parity
  log_interval: 1
  log_accumulate: true

episodes: ???                     # filled by total_steps / (env_max_steps * num_envs)
# OR set total_steps directly: 200_000 (sheeprl jzgkcep4 ran 200k policy steps)
total_steps: 200_000

# === Inheritance from food-only env config ===
# The merge brings in environment.height/width=5×5, max_steps=500,
# resources (food count=1, hiding_predator count=0), action set (6 actions),
# rewards, sensors. We do not redefine those here.
```

**Note for `developer`** (code-reviewer 🟡 #9 — decision locked 2026-05-12): the project does not have a `defaults_from`-style merge mechanism in `src/utils/config.py` (only `Config.merge(other)`). **Selected option (c)**: drop the `defaults_from:` key; the top-level `train.py` `dreamer-srl` branch adds a 5-line loader helper:
```python
# In train.py dreamer-srl branch, before build_state:
agent_cfg = Config.load_yaml(config.get_mandatory('agent_config'))
config.merge(agent_cfg)
```
No new YAML-level mechanism; no change to `src/utils/config.py`. The `01_food_only.yaml` shown above keeps `agent_config: configs/models/dreamer_srl/agent_xs.yaml` as the wiring; the `defaults_from:` line above is **dropped** (left in the example for context but DO NOT implement it — strike it from the implementation YAML).

##### CREATE `configs/models/dreamer_srl/__init__.py` — not needed; `configs/` is YAML-only.

#### EDIT `train.py` (lines around 430-460 and 770-820 — ADD a `dreamer-srl` branch; do NOT modify the existing `DreamerV3` branch)

The edit is purely additive: a new `elif algorithm == "dreamer-srl":` branch alongside the existing `DreamerV3` branch. Approximate addition: 80–120 lines, all new.

```python
# train.py — around line 430 (the algorithm allow-list)
# BEFORE:
if algorithm not in ("RecurrentPPO", "DreamerV3"):
    raise ValueError(...)

# AFTER:
if algorithm not in ("RecurrentPPO", "DreamerV3", "dreamer-srl"):
    raise ValueError(
        f"Continual learning (--configs-dir) is only supported for "
        f"RecurrentPPO, DreamerV3, and dreamer-srl, got algorithm='{algorithm}'. "
        "Use Option A: restrict to supported algorithms at startup.")
```

```python
# train.py — around line 456 (the algorithm-specific hyperparameter read)
# Add a new branch under the existing DreamerV3 branch:
elif algorithm == "dreamer-srl":
    num_steps = args.num_steps or config.get_mandatory('agent.per_rank_sequence_length')
    hidden_size = args.hidden_size or config.get_mandatory('agent.world_model.recurrent_state_size')
    lr = args.lr or config.get_mandatory('agent.world_model.world_model_lr')
```

```python
# train.py — around line 773 (the algorithm dispatch)
# Add a new branch under the existing DreamerV3 branch (NOT replacing it):
elif algorithm == "dreamer-srl":
    from src.algorithms.dreamer_srl import build_state, one_train_step, collect_step
    from src.algorithms.dreamer_srl.buffers import SequentialReplayBuffer
    from src.algorithms.dreamer_srl.utils import Ratio  # the dreamer-srl Ratio, NOT the existing one

    # No modulation read — dreamer-srl is pure replication (see plan §non-goals).

    key, init_key = jax.random.split(key)
    state = build_state(init_key, agent_config, obs_space, action_dim)
    ratio_scaled_updates = Ratio(config.get_mandatory('agent.replay_ratio'),
                                 pretrain_steps=config.get_mandatory('agent.per_rank_pretrain_steps'))

    buffer = SequentialReplayBuffer(
        buffer_size=config.get_mandatory('agent.buffer_size'),
        n_envs=num_envs,
        obs_keys=["state"],
        obs_shapes={"state": (input_dim,)},
        action_dim=action_dim,
    )

    # Player state lives in dreamer-srl's own state pytree, not as a separate dreamer_state.
    # Initial state at first env reset:
    player_state = state.player_initial_state(num_envs)

# (Then in the collection / training loop ~lines 1736-2000, add a parallel
#  branch that calls collect_step + one_train_step in the same shape as the
#  existing DreamerV3 branch but with the dreamer-srl APIs. This is the
#  bulk of the integration; the implementer matches the existing DreamerV3
#  block's structure 1:1.)
```

**Discipline for the `train.py` edit:**
- **Surgical.** Every new line traces to the dreamer-srl algorithm. Do not refactor the existing `DreamerV3` branch.
- **No removal.** The existing `DreamerV3` branch stays bit-identical.
- The new branch's metric names match sheeprl's `AGGREGATOR_KEYS` (`Loss/world_model_loss`, `Game/ep_len_avg`, etc.) — that lets the analysis run reuse the same WandB queries the diagnosis run used for `jzgkcep4`.

#### CREATE `scripts/dreamer_srl_offline_check.py` (new, ~150 lines — optional but recommended)

A small forward-pass-and-compare harness. Loads the sheeprl-trained checkpoint at `logs/runs/dreamer_v3/grid_world_pain/.../checkpoint/ckpt_200000_0.ckpt` (the `jzgkcep4` checkpoint, on disk), loads a dreamer-srl checkpoint at the same training step, feeds both the same `(s, a, r, d)` batch of size 4, and prints the elementwise diff for: encoder embed, RSSM `(h_t, post_logits, prior_logits)`, reward-head logits, critic logits, world-model loss components.

Used in Step 11 of the verification protocol below. Threshold for "bit-identical enough": all `|diff|.max()` ≤ `1e-4` (allowing for fp32 round-off across PyTorch ↔ JAX).

**Hazard:** PyTorch and JAX have slightly different floating-point reduction orderings. A 1e-6 tolerance would be too tight. 1e-4 is the project's standing threshold for cross-framework forward-pass parity.

### File Changes — summary table

| File | Status | Lines | Notes |
|---|---|---:|---|
| `src/algorithms/__init__.py` | NEW | ~3 | Package marker |
| `src/algorithms/dreamer_srl/__init__.py` | NEW | ~10 | Public re-exports |
| `src/algorithms/dreamer_srl/utils.py` | NEW | ~250 | `symlog`, init weights, lambda-return, `Moments`, `Ratio`, `prepare_obs` |
| `src/algorithms/dreamer_srl/buffers.py` | NEW | ~200 | In-memory `SequentialReplayBuffer` + `EnvIndependentReplayBuffer` |
| `src/algorithms/dreamer_srl/agent.py` | NEW | ~700 | `LayerNormGRUCell` (fix #28), `MLP`, encoder/decoder, `RSSM` (fix #30), `Actor`, `Critic`, `Player`, `build_agent` (fix #27) |
| `src/algorithms/dreamer_srl/loss.py` | NEW | ~250 | Distributions + `reconstruction_loss` |
| `src/algorithms/dreamer_srl/train.py` | NEW | ~500 | `build_state`, `one_train_step`, `collect_step`, target-critic Polyak update, critic loss (fix #29) |
| `configs/models/dreamer_srl/agent_xs.yaml` | NEW | ~70 | Sheeprl XS hyperparameter mapping |
| `configs/models/dreamer_srl/01_food_only.yaml` | NEW | ~30 | Food-only env + agent config |
| `train.py` | EDIT | +80–120, −0 | New `dreamer-srl` algorithm branch (lines ~430, ~456, ~773, ~1319, ~1461, ~1736, ~2000, ~2635 — additive) |
| `scripts/dreamer_srl_offline_check.py` | NEW (optional) | ~150 | Cross-framework forward-pass parity check |
| **Total** | | **~2240 new + ~100 edits in `train.py`** | |

**Explicitly NOT modified:**

- `src/models/dreamer_v3_*.py` (all 4 files) — the existing JAX Dreamer is the control.
- `configs/models/dreamer_v3*.yaml` (all 7 files) — the existing Dreamer configs.
- `src/environment/*` — env interface is reused as-is.
- `src/utils/config.py` — config loader is reused as-is.
- `scripts/regen_dev_index.py` — no new topic added (we use `topic: dreamer`).
- Any agent or analyzer file under `src/`.

## Checkpoints (verification checks **during** implementation)

Each checkpoint is a **forward numerical check** against sheeprl source where feasible. The implementer (`developer` agent) runs each before moving on; result goes in the Implementation Report.

- [ ] **Checkpoint 1 — `utils.py` forward parity.** Feed `x = jnp.array([-100, -1, 0, 1, 100])` through `symlog → symexp` and confirm round-trip `|x - symexp(symlog(x))|.max() < 1e-5`. Feed `(rewards, values, continues, lmbda=0.95)` of shape `[16, 4]` through `compute_lambda_values` and compare against a `numpy` Python-loop reference implementation in the same file's docstring — both must match to `1e-6`.
- [ ] **Checkpoint 2 — `agent.py` `LayerNormGRUCell` cascade fix #28.** Construct a single cell with input_size=8, hidden=16, fixed seed, feed `x=ones(2, 8)`, `h=zeros(2, 16)`. Assert that the candidate-hidden uses the reset gate: with `apply_gru_reset_gate = False` (a temporary debug flag for this check, then removed), the output differs by `> 1e-3` from the canonical version. With it on, the output matches the formula `(1-update) * h + update * tanh(reset * cand_proj)`. After the check, **delete the debug flag** — the canonical version is the only version.
- [ ] **Checkpoint 3 — `agent.py` `build_agent` cascade fix #27.** After init, dump the reward-head terminal Linear kernel and the critic terminal Linear kernel. Both must be exactly zero: `jnp.abs(kernel).max() == 0.0`. The biases must also be exactly zero. (Sheeprl's `uniform_init_weights(0.0)` gives `limit=0`, so `uniform(-0, 0) = 0` — verify this is what we get.)
- [ ] **Checkpoint 4 — `agent.py` cascade fix #30.** Dump `rssm.representation_model` and `rssm.transition_model` and assert each is a 2-layer MLP (one hidden Dense → LayerNorm → SiLU → terminal Dense to `stochastic_size * discrete_size = 1024`). Not a bare terminal Dense.
- [ ] **Checkpoint 5 — `loss.py` two-hot distribution cascade fix #2.** Construct `TwoHotEncoding(logits, bins=255, low=-20, high=20)`. **CORRECTED 2026-05-12 (math-reviewer 🔴 #1)**: sheeprl stores `self.bins = linspace(-20, 20, 255)` in **symlog space** (see `tmp/sheeprl/sheeprl/utils/distribution.py:237`). The `symexp` is applied lazily only at consumption sites (`mean` / `mode` via `transbwd`), NOT to the stored bin grid. Required assertions:<br>• `bins[0] = -20.0`, `bins[127] = 0.0`, `bins[254] = +20.0` — the **symlog-space** grid endpoints.<br>• The real-space bin centers `symexp(bins)` should give `symexp(bins[0]) ≈ -4.85e8`, `symexp(bins[127]) = 0.0`, `symexp(bins[254]) ≈ +4.85e8` — but these live behind `mean`/`mode` accessors, not on `self.bins`.<br>• `log_prob(target)` pushes `target` through `symlog` first (since `self.bins` lives in symlog space). Verify with `target=jnp.array([0.5])`: `symlog(0.5) ≈ 0.405`, which falls between `bins[127]=0.0` and `bins[128]=0.157` — the two-hot weights should reflect that.<br>• Feed `target=jnp.array([0.5])` and decode the predicted mean from a uniform-logits input; the mean should be approximately `0.0` (uniform prior over symmetric bins). **DO NOT** implement `self.bins = symexp(linspace(-20, 20, 255))` — that would break `log_prob` because the target is symlog-encoded before bin lookup.
- [ ] **Checkpoint 6 — `train.py` critic loss cascade fix #29.** Construct a critic loss with random `qv_logits`, `lambda_target`, `target_critic_value`. Assert that the loss contains BOTH `−qv.log_prob(λ_target.detach())` AND `−qv.log_prob(target_critic_value.detach())`, not just the first term. Switch off the second term (temporary debug flag for the check) and confirm the loss decreases by exactly the second contribution. After the check, **delete the debug flag**.
- [ ] **Checkpoint 7 — `train.py` Polyak update.** On the first call, `tau=1.0` (hard copy): `target_params == source_params` after the call. On the second call, `tau=0.02`: `target_params == 0.02 * source + 0.98 * target` — check elementwise on every leaf.
- [ ] **Checkpoint 8 — end-to-end forward parity (the bit-identity check).** Run `scripts/dreamer_srl_offline_check.py` with the `jzgkcep4` checkpoint and a freshly-initialised dreamer-srl model at the same param count. **Initialise from the same RNG seed.** Compare:
  - Encoder embedding diff `< 1e-4`
  - RSSM `recurrent_state` diff `< 1e-4` after 1 dynamic step
  - Reward-head logits diff `< 1e-4`
  - Critic logits diff `< 1e-4`
  - World-model loss components (obs / reward / state / continue / total) — each pairwise diff `< 1e-4`.
  If this check passes, the implementation is bit-aligned. If not, the differences localise which file still diverges.
- [ ] **Checkpoint 2b — action shift (§S2 test).** Assert that the actions fed to the RSSM rollout are `jnp.concatenate([zeros_like(actions[:1]), actions[:-1]], axis=0)`, not raw `actions`. Construct a batch with known `actions = arange(0, T*B).reshape(T, B, A)`; assert `shifted[0] == 0` and `shifted[1:] == actions[:-1]`.
- [ ] **Checkpoint 4b — RSSM `is_first` reset (§S1+§S4 test).** Feed a sequence with `is_first = [1, 0, 0, ..., 0]` and a different sequence with `is_first = [0, 0, ..., 0]` AFTER applying the force-set §S1. Assert that on step 0 the RSSM recurrent state is exactly `tanh(initial_recurrent_state.value)` in the first case (post-S1, this is BOTH cases since S1 forces is_first[0]=1 regardless). Also assert the three-quantity reset (`action`, `recurrent_state`, posterior — see §S4) applies the arithmetic-mask form, NOT `jnp.where`.
- [ ] **Checkpoint 9 — dry-run on food-only NoPred, 5,000 env steps.** Launch the parity command (see §Parity verification protocol below) with `total_steps: 5000`. Confirm: no NaN in any loss, `Loss/world_model_loss` is decreasing from step 1000→5000, `Game/ep_len_avg` is being logged (non-NaN). Survival need not be saturated at this point; we're just confirming the loop runs end-to-end.
- [ ] **Checkpoint 9b — random-action prefill (§S3 test).** Assert that on policy steps `0..1023` (iteration `0..255` for our `num_envs=4`) the actions in the replay buffer have entropy `≈ log(action_dim)` (uniform), and from step 1024 onward the entropy starts to decrease (policy begins acting). Also assert that no gradient step fires before `iter_num >= learning_starts`.
- [ ] **Checkpoint 10 — wall-clock budget.** Time 5,000 env steps with `num_envs=4`, `replay_ratio=1`. Compute extrapolated wall-clock for 200k steps. Compare against `jzgkcep4`'s 12.5 hours. If dreamer-srl takes `> 25 hours` (2× of sheeprl), stop and report — the JAX implementation has a speed regression and the parity-time gate fails. If `≤ 25 hours`, proceed to full parity run.

## Parity-verification protocol (after Checkpoints 1–10 pass)

This is the **gate** for "dreamer-srl is done". It is not a hyperparameter sweep; the gate has a fixed criterion and a fixed budget.

### The gate

> Three seeds of dreamer-srl, running `configs/models/dreamer_srl/01_food_only.yaml`, mean survival ≥ ~500 over the last 50,000 policy steps (40k–200k window, by analogy with `jzgkcep4`'s saturation window) and wall-clock ≤ 25 hours per seed on the same node sheeprl's `jzgkcep4` ran on (node 114, RTX 6000 Ada).

### Launch commands

Launched via the `training-runner` agent. Three runs in parallel (one per seed), each on a different GPU on node 114 (or distributed across nodes 113/114 if needed — see Risks §3 below for the GPU contention plan):

```bash
# Seed 1 — analogous to the JAX_DreamerV3 launch pattern used in `train_command-agent.sh`
XLA_PYTHON_CLIENT_PREALLOCATE=false CUDA_VISIBLE_DEVICES=0 \
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --algorithm dreamer-srl \
  --config configs/models/dreamer_srl/01_food_only.yaml \
  --total-timesteps 200000 \
  --num-envs 4 \
  --seed 0 \
  --wandb-name dreamer_srl_parity_s0 \
  --device gpu:0

# Seed 2 — same with --seed 1, CUDA_VISIBLE_DEVICES=1, --wandb-name dreamer_srl_parity_s1
# Seed 3 — same with --seed 2, CUDA_VISIBLE_DEVICES=2 (or node 113), --wandb-name dreamer_srl_parity_s2
```

WandB project: `grid_world_pain_dreamer_srl_parity` (NEW project — keeps these runs separate from both `grid_world_pain` (in-house Dreamer) and `grid_world_pain_sheeprl_test` (sheeprl drop-in)).

### Pass / fail decision rule

| Outcome | Survival (mean over seeds, 40-200k window) | Wall-clock (max over seeds) | Verdict |
|---|---|---|---|
| **Pass** | ≥ 480 (i.e. ≥ 96% of the 500-step cap, allowing for one seed bobbling at 460-480) | ≤ 25 hours per seed | dreamer-srl is at parity. Proceed to neuromodulation port plan (separate doc). |
| **Marginal pass** | 450 ≤ mean < 480, all three seeds ≥ 400 | ≤ 30 hours per seed | At-the-edge parity. Run one extra seed; if it lifts the mean ≥ 480, pass. If not, marginal fail. |
| **Fail (survival)** | mean < 450, or any seed < 300 | (any) | Component-by-component diagnosis. See "What to do on fail" below. |
| **Fail (speed)** | (any) | > 30 hours per seed | Speed regression. Profile with `jax.profiler.start_trace` on a 5,000-step slice; compare to the existing `tmp/20260506_*_dreamer_v3_vs_rppo_profile/` baseline. Decide whether to ship despite the regression (only if survival passes) or to rework `one_train_step`'s JIT structure. |

### What to do on fail (survival)

**No hyperparameter tweaks.** The parity gate is bit-identical replication, so hyperparameter "fixes" hide the real bug rather than expose it. Instead, **diagnose by component**:

1. Re-run Checkpoint 8 (the offline forward-pass parity check) at the failing seed's 10k, 50k, 100k checkpoint. Whichever component first crosses the `1e-4` threshold names the diverging file.
2. If everything in Checkpoint 8 stays under `1e-4` but survival still fails, the divergence is in the **training loop** (gradient flow, optimiser state, target-critic update timing), not in the forward pass. Open a follow-up issue plan focused on `train.py` + optimiser semantics.
3. If only one component diverges (e.g. critic logits), the bug is localised to that file. Open a focused fix plan (NOT a full re-implementation) in the dreamer_srl folder, supersede this one.

**Do NOT** add new YAML knobs to "make it work". Do NOT bring back any of the cascade-era flags (`apply_gru_reset_gate`, `paper_canonical_twohot_bins`, `zero_init_reward_critic`). dreamer-srl has those flags ON by definition — they are not knobs.

## Implementation order (step-by-step for `developer`)

1. **Flax-API call resolved**: NNX (Risks §1, RESOLVED 2026-05-12). Read `src/models/dreamer_v3_nnx.py` and `src/models/dreamer_v3_trainer.py` for the project's existing NNX conventions (module construction with `nnx.Rngs`, state split/merge for JIT, EMA via `nnx.state` / `nnx.update`). Do NOT re-debate this choice during implementation — flag any blocker to senior-developer instead.
2. **Confirm config merge mechanism** (config section above). Read `train.py:78` and `src/utils/config.py:Config.merge` to determine the exact way two YAML files (env + agent) are composed at startup. Add a one-line note to the plan if a small helper is needed.
3. **Implement `utils.py`** + run Checkpoint 1.
4. **Implement `loss.py`** (distributions + `reconstruction_loss`) + run Checkpoint 5.
5. **Implement `buffers.py`** + a small ringbuffer round-trip test (not a numbered checkpoint, just a sanity smoke).
6. **Implement `agent.py`** + run Checkpoints 2, 3, 4.
7. **Implement `train.py` (the module)** + run Checkpoints 6, 7.
8. **Implement `scripts/dreamer_srl_offline_check.py`** + run Checkpoint 8.
9. **Edit `train.py` (the top-level)** to add the `dreamer-srl` branch. Wire the `learning_starts` random-action prefill (§S3) BOTH at action selection AND at the gradient-step gate. Run Checkpoints 9 + 9b (5,000-step dry-run + prefill entropy assertion).
10. **Time check** — Checkpoint 10.
11. **Launch the three-seed parity run** (Parity verification protocol above) via `training-runner`.
12. **Hand off to `experiment-analyzer`** when all three seeds finish. Analyzer writes the parity-verdict report under `docs/experiments/active/diagnosis/` (NOT under `docs/develop/`).

## Non-goals (out of scope; do NOT do these)

1. **No NMN / FiLM / precision-modulation hooks.** dreamer-srl is pure sheeprl replication. No injection sites, no identity-pass-through stubs, no commented-out FiLM. The NMN port lives in a separate later plan, after parity is reproduced.
2. **No continuous-action support.** Sheeprl's `TruncatedNormal` distribution block (`tmp/sheeprl/sheeprl/utils/distribution.py` lines 25–147) is dead code per the walkthrough. Do not translate it. Discrete-only.
3. **No MLflow registration.** Sheeprl's `log_models_from_checkpoint` (its `utils.py:189-235`) is MLflow-specific. We use WandB. Drop entirely.
4. **No memmap replay.** Sheeprl's `EnvIndependentReplayBuffer` supports memmap-on-disk for huge buffers; we run with `buffer_size=1_000_000` transitions which fits in RAM. In-memory only.
5. **No Hydra.** Sheeprl uses `hydra.utils.instantiate` for optimisers, layer-norms, activations. We use plain Python lookups (a small `dispatch` dict in `train.py`) and `optax.adam(...)` directly.
6. **No `gym.vector.SyncVectorEnv`.** Our env is JAX-native via `ParallelEnv` (`src/environment/wrapper.py`); SyncVectorEnv is replaced by vectorisation over the batch axis of `jax_step`.
7. **No `RestartOnException`.** Our JAX env does not raise mid-step; the wrapper is unnecessary.
8. **No `fabric.all_gather`.** Single-device; drop distributed-training code.
9. **No `EpisodeBuffer`.** SequentialReplayBuffer only.
10. **No hyperparameter sweeps.** Parity gate is mean ≥ ~500 over 3 seeds. No grid-search on `replay_ratio`, `learning_starts`, `kl_free_nats`, etc.
11. **No task extension to predators (`02_predator_slow.yaml`, `03_predator_full.yaml`).** Food-only NoPred is the sole parity gate.
12. **No modification to `src/models/dreamer_v3_*.py` or `configs/models/dreamer_v3*.yaml`.** The existing Dreamer is the control.
13. **No diagnostic reward-MAE port.** Sheeprl-side reward-MAE was deferred in the diagnosis run; remains deferred here.

## Risks and open questions

1. **Flax API: NNX (RESOLVED 2026-05-12).** The user's directive was "closer to current Dreamer." The current Dreamer is NNX-based end-to-end (`src/models/dreamer_v3_nnx.py:4` `from flax import nnx`; `train.py:784` `rngs=nnx.Rngs(init_key)`), so dreamer-srl is also NNX. Side-by-side comparison against the current Dreamer reads as a true sheeprl-vs-ours diff, not a Flax-dialect diff. Recurrent PPO (`src/models/recurrent_ppo_network.py`) uses `flax.linen`; dreamer-srl does NOT follow that precedent. The mutable-state pitfalls flagged in the original draft (PRNG threading at forward, EMA target update, replay-pointer state) are mitigated by explicit `nnx.split` / `nnx.merge` at JIT boundaries — see `src/models/dreamer_v3_trainer.py` for the project's existing pattern. `code-reviewer` will audit the NNX-specific correctness during implementation review.

2. **`Ratio` semantics in our env-step counter (open)**. Sheeprl's `Ratio` counts `policy_step / world_size`. Our top-level `train.py` already has a `Ratio` for the existing Dreamer at `train.py:789` (`Ratio(config.get_mandatory('agent.replay_ratio'))`), called with what argument exactly? Read `train.py` around line 789–800 and the `Ratio` call site in the training loop to confirm. The dreamer-srl `Ratio` should be called with the same env-step semantics; if there is any difference, the fractional-debt accumulator drifts and replay-ratio is silently wrong. **Resolution**: the developer reads the existing `train.py` and dreamer-srl's `utils.py:Ratio` call site side-by-side at Step 7.

3. **GPU contention on node 114 for the 3-seed parity run (open)**. Node 114 has the RTX 6000 Ada used for `jzgkcep4`. The auto-memory note "Runner must verify one PID post-launch" warns that triple-launching on the same GPU has caused duplicate-PID incidents (2026-05-07). **Resolution**: three different `CUDA_VISIBLE_DEVICES` values if node 114 has 3 GPUs; otherwise split across nodes 113 and 114. The `training-runner` agent confirms this at launch time.

4. **Gradient-flow translation of `samples + (probs - probs.detach())` (RESOLVED 2026-05-12)**. ✅ math-reviewer signed off 2026-05-12; `probs + jax.lax.stop_gradient(one_hot_sample - probs)` is gradient-identical to PyTorch form `samples + (probs - probs.detach())`. Both yield forward `= one_hot_sample` and `∂/∂probs = 𝟙`. Derivation appendix in `review_math.md` §A. No further review needed.

5. **Hafner-init constant `0.87962566103423978` (RESOLVED, with precision warning)**. This is the constant that corrects truncated-normal variance (range `[-2σ, 2σ]`) back to unit variance. Hard-coded in sheeprl `utils.py:149` and `utils.py:159`. Replicate verbatim; do not redo the derivation. **⚠️ Use the full-precision constant `0.87962566103423978`**, NOT the truncated `0.8796` in the existing `src/models/dreamer_v3_util.py:hafner_init` (code-reviewer aux observation 2026-05-12). The 4-decimal truncation is what causes Hafner-init-bearing layers to be slightly off-spec.

6. **`Moments` percentile EMA initial value (resolved-in-plan)**. Sheeprl initialises `low = high = 0.0` (`utils.py:53-54`). After the first call, `low` and `high` get the EMA update with `decay=0.99`. For ~100 first calls the values are biased toward 0 — that is sheeprl's behaviour and we replicate it exactly.

7. **Walkthrough gap — sequence-length truncation off env episodes (flagged)**. The walkthrough notes that `SequentialReplayBuffer.sample` returns contiguous-T windows that **may straddle episode boundaries**, and the trainer relies on stored `is_first` for RSSM reset (see `buffers.md`). The walkthrough does **not** transcribe the exact `bincount`-based per-env subset sampling logic in `EnvIndependentReplayBuffer.sample_tensors`. If the implementer finds an ambiguity, raise it as a new open question and read the source at `tmp/sheeprl/sheeprl/data/buffers.py:EnvIndependentReplayBuffer.sample_tensors` directly. Do not silently invent.

8. **Walkthrough gap — `decoupled_rssm` branch (flagged)**. The default config has `decoupled_rssm: false`; we replicate that. The `DecoupledRSSM` subclass at `agent.py:501-595` is **not** translated in dreamer-srl. If the walkthrough's `agent.md` covers it line-for-line, that translation is unused; if it does not, no gap. **Implementer skips it entirely.**

9. **WandB metric-name collision with the existing Dreamer (resolved-in-plan)**. The existing JAX Dreamer logs metrics under names like `dreamer_v3/world_model_loss` (or whatever pattern `train.py` uses for it). Sheeprl's `AGGREGATOR_KEYS` (`Loss/world_model_loss`, `Game/ep_len_avg`, …) is what we want for dreamer-srl, so the analyzer can reuse the `jzgkcep4` extraction code. **The new WandB project name (`grid_world_pain_dreamer_srl_parity`) keeps the runs apart;** no collision.

10. **No-fallback-defaults audit (resolved-in-plan, flagging)**. Every new key in `configs/models/dreamer_srl/agent_xs.yaml` must be read via `Config.get_mandatory` in the trainer. Specifically: `agent.gamma`, `agent.lmbda`, `agent.horizon`, `agent.replay_ratio`, `agent.learning_starts`, `agent.per_rank_pretrain_steps`, `agent.per_rank_sequence_length`, `agent.per_rank_batch_size`, `agent.unimix`, `agent.hafner_initialization`, `agent.dense_units`, `agent.mlp_layers`, `agent.layer_norm_eps`, `agent.world_model.*` (every leaf), `agent.actor.*` (every leaf), `agent.critic.*` (every leaf), `agent.mlp_keys.*`, `agent.cnn_keys.*`, `agent.buffer_size`, `agent.buffer_device`, `agent.player.discrete_size`, `distribution.type`, `distribution.validate_args`. **No `config.get('key', default)`** in dreamer-srl code. This is the project-wide rule (`CLAUDE.md` "No fallback defaults").

11. **`Moments` `max_=1.0` semantic effect (flagging — professor-rl-bayesian-dl #11)**. Sheeprl `dreamer_v3.yaml:134` sets `max: 1.0`, overriding the class default `1e8`. The semantic is: `invscale = max(1/max_, high - low)`, so with `max_=1.0` the **invscale floor is `1.0`**, meaning when the lambda-value spread is tiny the advantage is NOT re-scaled larger than the spread itself. The walkthrough calls this a "ceiling" but it's actually a floor on the invscale (= a ceiling on the scale-down factor). Document precisely in the YAML so the developer does not silently set it back to `1e8`.

12. **`Ratio` env-step counter semantics (open — code-reviewer 🟢 concern; same as Risk #2)**. The `Ratio` scheduler MUST stay Python-side (uncompiled). The plan §`utils.py` row says so but does not enforce. Resolution: the developer reads `train.py` around the existing `Ratio` call site (line ~789) and dreamer-srl's `utils.py:Ratio` call site side-by-side at Step 7 to confirm same env-step convention. Cross-reference Risk §2.

13. **No-import-from-existing-Dreamer rule (code-reviewer 🟢 concern)**. dreamer-srl does NOT import any module from `src/models/dreamer_v3_*` without an explicit ✓ from the senior-developer. Existing classes that look reusable but **must be re-implemented** to avoid silent drift: `LayerNormGRUCell` (wrong matmul structure — see `agent.py` row), `Moments` (wrong mutation pattern — see `utils.py` row), `hafner_init` (truncated constant — see Risks §5), `compute_lambda_values` (v1 plan asserted line-for-line identity; v2 requires verification). Replay-buffer sampling explicitly forbids the episode-boundary-respecting variant (sheeprl's contract is straddle-permissive — see `buffers.py` row).

14. **`per_rank_*` config-key naming convention (code-reviewer 🟡 #10, decision locked 2026-05-12)**. Keep sheeprl's verbatim names (`per_rank_sequence_length`, `per_rank_batch_size`, `per_rank_pretrain_steps`, `per_rank_target_network_update_freq`) — better for parity, and the analyzer reads both projects with the same key. The YAML header now documents the meaning: "per_rank_*" is sheeprl's distributed-training prefix; we run single-device so per_rank=global.

## Links

- [Sheeprl walkthrough INDEX](../../../project/references/sheeprl_dreamer_v3/INDEX.md) — the canonical source for every per-file translation
- [Sheeprl drop-in test diagnosis](../diagnosis/sheeprl_drop_in_test.md) — why this plan exists, including the `jzgkcep4` analysis
- [Cascade re-summary 2026-05-11](../../../experiments/summaries/20260511_1559_dreamer_v3_fix_cascade.md) — the 5-fix cascade we are stopping in favour of this re-implementation
- [`docs/develop/active/dreamer/dreamer_v3_implementation.md`](../dreamer/dreamer_v3_implementation.md) §9 — sheeprl-comparison section in the existing dreamer doc (background on the 5 cascade items)
- [`CLAUDE.md`](../../../../CLAUDE.md) — project-wide rules (conda env, no fallback defaults, git safety, no in-place destructive cleanups)
- Sheeprl source on disk: `tmp/sheeprl/sheeprl/algos/dreamer_v3/{dreamer_v3,agent,loss,utils,evaluate}.py`, `tmp/sheeprl/sheeprl/models/models.py`, `tmp/sheeprl/sheeprl/utils/{distribution,utils}.py`, `tmp/sheeprl/sheeprl/data/buffers.py`
- Sheeprl XS hyperparameters: `tmp/sheeprl/sheeprl/configs/algo/dreamer_v3_XS.yaml` (deltas) + `tmp/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml` (XL defaults — sheeprl calls this `dreamer_v3_XL` colloquially in the file header but the file itself is `dreamer_v3.yaml`)

## Implementation Report

> **Implemented by**: [agent/person]
> **Date**: [date]

<!-- Filled by the implementing agent after the Implementation order steps complete.
     Describe what was done, any deviations from the plan, blockers, and any changes
     to the file changes table. Speed-change measurements (Checkpoint 10) go here. -->

## Verification Report

> **Verified by**: [agent/person]
> **Date**: [date]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/algorithms/__init__.py` | NEW | | |
| `src/algorithms/dreamer_srl/__init__.py` | NEW | | |
| `src/algorithms/dreamer_srl/utils.py` | NEW | | |
| `src/algorithms/dreamer_srl/buffers.py` | NEW | | |
| `src/algorithms/dreamer_srl/agent.py` | NEW | | |
| `src/algorithms/dreamer_srl/loss.py` | NEW | | |
| `src/algorithms/dreamer_srl/train.py` | NEW | | |
| `configs/models/dreamer_srl/agent_xs.yaml` | NEW | | |
| `configs/models/dreamer_srl/01_food_only.yaml` | NEW | | |
| `train.py` | EDIT (additive) | | |
| `scripts/dreamer_srl_offline_check.py` | NEW (optional) | | |

**Checkpoints status:**

- [ ] Checkpoint 1 — `utils.py` forward parity
- [ ] Checkpoint 2 — `LayerNormGRUCell` fix #28
- [ ] Checkpoint 2b — Action-shift (§S2)
- [ ] Checkpoint 3 — `build_agent` fix #27 (zero-init heads)
- [ ] Checkpoint 4 — RSSM hidden layers fix #30
- [ ] Checkpoint 4b — RSSM `is_first` reset (§S1+§S4)
- [ ] Checkpoint 5 — Two-hot bin grid fix #2 (symlog space)
- [ ] Checkpoint 6 — Critic self-EMA fix #29
- [ ] Checkpoint 7 — Polyak target update
- [ ] Checkpoint 8 — End-to-end forward parity (`scripts/dreamer_srl_offline_check.py`)
- [ ] Checkpoint 9 — 5,000-step dry-run on food-only NoPred
- [ ] Checkpoint 9b — Random-action prefill (§S3)
- [ ] Checkpoint 10 — Wall-clock budget (≤ 2× sheeprl's 12.5 h)

**Parity-gate run** (3 seeds, food-only NoPred):

- Seed 0 WandB:
- Seed 1 WandB:
- Seed 2 WandB:
- Mean survival (40-200k window):
- Max wall-clock:
- Verdict:

**Conclusion**: [one-line summary]

<!-- For ⚠️/❌ items, add detailed sections below the table with:
     root cause, affected lines, and recommended fix.
     If the parity gate fails, follow the "What to do on fail (survival)"
     branch of the parity verification protocol — open a focused fix plan,
     do NOT add hyperparameter knobs. -->

---

<!--
NEW ISSUES: If a new issue is discovered during implementation/verification:
- If closely related: append as "## Issue #2: [title]" below with the same template sections.
- If independent: create a separate document and cross-reference.

NOTES:
- This plan supersedes the cascade approach for the parity question. The cascade
  doc itself (`docs/develop/active/dreamer/dreamer_v3_implementation.md` §9) remains
  active as a reference for the 5 known divergences, but no further cascade cells
  will be cut.
- The NMN port plan is a separate follow-up: open a new doc under
  `docs/develop/active/dreamer_srl/NMN_PORT_PLAN.md` (or `dreamer_srl/` folder)
  AFTER this plan's Verification Report shows ✅ on all checkpoints and the
  parity gate.
-->

---

## Review by `professor-rl-bayesian-dl` (algorithm + Bayesian-DL audit, 2026-05-12)

**Verdict: DEVIATIONS FOUND — 11 items** (all mechanical edits, no algorithmic re-think needed).

**What was checked.** Whether the plan replicates sheeprl's DreamerV3 algorithm faithfully — RSSM split, two-hot heads, Moments percentile EMA, KL-balanced split with free-nats floor, λ-return recursion, EMA target critic with cascade #29 self-EMA reg, Hafner init with cascade #27 zero-init of reward/critic terminal Linears, all five cascade items at their correct call sites, and the env-step ↔ gradient-step interleave under `Ratio`.

**What the reviewer confirmed correct.** All five cascade items (#2, #27, #28, #29, #30) are placed and named correctly. The non-goal walls (no NMN/FiLM/precision-modulation, no continuous actions, no MLflow, no memmap, no Hydra) are clean — no injection points sneak in. No arbitrary algorithmic additions.

**Three silent omissions in the training step (these would break Checkpoint-8 forward-parity).**
1. **`is_first[0]=1` force-set** on the first timestep of every replay chunk — sheeprl forces the boundary reset regardless of stored value. The plan does not name this.
2. **Prepend-zero-action shift** — action at index `t` in the replay chunk is the action taken AT `t-1`, not at `t`. The plan does not name the shift.
3. **`learning_starts` random-action prefill phase** — the YAML key is loaded but the plan never wires it into either the collection branch or the gradient-step gate.

**Eight YAML/hyperparameter coverage gaps.** Missing `distribution.type` / `validate_args`, actor `init_std` / `min_std` / `max_std`, `discount_model.learnable` flag, continue/reward-head MLP sizing keys, an ambiguous λ-return claim that doesn't fix `discount_factor` and `lmbda` to their sheeprl values, missing `Independent(BernoulliSafeMode, 1)` wrap on the continue head, `Moments` `max_=1.0` floor semantics (normaliser denominator must be `max(1, q95 - q05)`, not `q95 - q05` directly).

**Companion file**: [`review_professor_rl_bayesian_dl.md`](review_professor_rl_bayesian_dl.md) — full numbered deviation list with plan-location ↔ walkthrough-location ↔ proposed-correction per item.

---

## Review by `math-reviewer` (equation + numerical-constant audit, 2026-05-12)

**Verdict: DEVIATIONS FOUND — 7 items** (1 🔴 critical, 6 🟡 under-described). All five Hafner paper equations (Eq. 4, 5, 9, 10, 11) and all five cascade items are mathematically correct in the plan's prose and YAML. 32 numerical constants in the YAML cross-checked line-by-line against `tmp/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml` and `dreamer_v3_XS.yaml` — no drift, no arbitrary constants.

**🔴 Critical: Checkpoint-5 verification spec is wrong** (must fix before any implementation runs). The plan instructs the developer to verify `TwoHotEncoding.bins ≈ symexp(linspace(-20, 20, 255))` with `bins[0] ≈ -4.85e8`. Sheeprl actually stores `self.bins = linspace(-20, 20, 255)` in **symlog space**; the `symexp` is applied lazily in `mean` / `mode` via `transbwd`. A developer following the plan's Checkpoint 5 literally would store the wrong bin grid, and `log_prob` (which pushes target through `symlog` first) would silently miss its expected bins. Fix: change Checkpoint 5 to verify `bins[0] = -20`, `bins[127] = 0`, `bins[254] = 20`, with `symexp` applied at consumption sites only.

**🟡 Six load-bearing details under-described** (won't break the math by themselves, but the implementer should not have to re-derive them):
1. **True-continue splice at imagination index 0** (`dreamer_v3.py:246-248`) — at imagination step 0 the data-side continue probability is spliced in instead of the model-side prediction.
2. **Action zeroing on `is_first`** (`agent.py:425`) — RSSM's first-step action input is zeroed where `is_first=1`.
3. **Discount weighting on actor loss** (`dreamer_v3.py:297`) — actor loss multiplied by cumulative-discount mask along the imagination horizon.
4. **Discount weighting on critic loss** (`dreamer_v3.py:316`) — same cumulative-discount mask applied to critic regression.
5. **`low`-offset algebraic cancellation in the advantage** — the percentile-EMA `low` cancels out in `(R - low) / max(1, high - low) - 0` due to baseline subtraction; plan should note this so the implementer does not "fix" the missing offset.
6. **Per-element-then-mean ordering of the free-nats floor** — floor applied per-(B, T) element BEFORE reducing across dims, not after the mean.

**Risks §4 (straight-through estimator JAX translation) — SIGNED OFF.** `probs + jax.lax.stop_gradient(one_hot_sample - probs)` is gradient-identical to sheeprl's `samples + (probs - probs.detach())`. Plan's translation is correct.

**Companion file**: [`review_math.md`](review_math.md) — full deviation list with cited paper equations, sheeprl source lines, and LaTeX-form proposed corrections.

---

## Review by `code-reviewer` (JAX / Flax-NNX correctness audit, 2026-05-12)

**Verdict: DEVIATIONS FOUND — 11 items** (8 blockers, 3 concerns). Plan is structurally sound but underspecified at exactly the NNX-translation seams where a developer pattern-matching on the existing `src/models/dreamer_v3_nnx.py` would silently produce a non-bit-identical replica.

**Four blockers the developer would otherwise stumble on.**
1. **`LayerNormGRUCell` fused-gate structure** — sheeprl uses **one** `nn.Linear` + **one** `LayerNorm` on the concatenated `[hx, input]` (the paper's fused gate). The existing `src/models/dreamer_v3_nnx.py:LayerNormGRUCell` uses **two** `nnx.Linear` and **two** `nnx.LayerNorm` (the PyTorch `nn.GRUCell` convention with separate input-to-hidden and hidden-to-hidden projections). A developer pattern-matching the existing class will silently fail Checkpoint 8. Plan must name this explicitly.
2. **`is_first` reset is arithmetic-mask form**, not `jnp.where`: `(1 - is_first) * x + is_first * init`. AND three things are reset, not two — `action`, `recurrent_state`, AND `posterior` — with the posterior pre-flattened before masking.
3. **`get_initial_states` uses the transition model's `mode`, not a sample** — fully deterministic, no PRNG consumed at initial-state construction.
4. **`Moments` should be a `flax.struct.dataclass`**, not the `nnx.Variable`-mutation form used in `src/models/dreamer_v3_util.py`. The plan must specify a pure-functional update returning a new `Moments` pytree, not in-place attribute mutation.

**Four additional blockers**: `lax.scan` carry signature underspecified for the RSSM rollout, PRNG sub-key threading convention not pinned, `Player` aliasing semantics ambiguous (plan says "always pass current params" — correct, but does not specify whether this is via `nnx.split`/`merge` or a separate state pytree), Polyak cadence not specified (per gradient step vs. per N steps).

**Three concerns**: existing-code-reuse claims should be made stricter (don't import any module from `src/models/dreamer_v3_*` without an explicit ✓ from the senior-developer); `Ratio` scheduler should be Python-side (uncompiled) — plan says so but does not enforce; replay-buffer sampling semantics should explicitly forbid the episode-boundary-respecting variant from leaking in.

**Companion file**: [`review_code.md`](review_code.md) — full numbered deviation list with NNX code-pattern proposed corrections per item.

---

## Summary across all three reviews

| Reviewer | Total deviations | Critical / Blocker | Verdict |
|---|---|---|---|
| `professor-rl-bayesian-dl` | 11 | 3 (training-step omissions) | Deviations found |
| `math-reviewer` | 7 | 1 🔴 (Checkpoint-5 spec) | Deviations found; straight-through estimator SIGNED OFF |
| `code-reviewer` | 11 | 8 (NNX pattern traps) | Deviations found |
| **Total** | **29** | **12** | **All addressable as pre-implementation edits** |

**No arbitrary additions found by any reviewer** — every deviation is a *missing* piece or an *underspecified* piece, not an invented one. The non-goal walls held.

**Next step**: senior-developer folds all 29 deviation corrections into a v2 of the plan (mechanical edits, no algorithmic re-think). After v2 lands and a quick re-audit confirms zero residual deviations, `developer` can begin implementation.

---

## v2 Changes Applied (2026-05-12)

All 29 deviations flagged by the three reviewers have been folded into the body of this plan. Mechanical edits only — no algorithmic re-think. The reviewer sections above remain as the historical v1 audit record.

### `professor-rl-bayesian-dl` (11 items, all applied)

1. **Item 1 — `is_first[0] = 1` force-set** — applied at: Training-loop semantics §S1 (new subsection above File Changes), cited in `train.py` `one_train_step` row.
2. **Item 2 — Prepend-zero-action shift** — applied at: Training-loop semantics §S2, cited in `train.py` `one_train_step` row + new Checkpoint 2b.
3. **Item 3 — `learning_starts` random-action prefill** — applied at: Training-loop semantics §S3, `collect_step` row, Implementation order Step 9, YAML comment block on `learning_starts`, new Checkpoint 9b.
4. **Item 4 — `prepare_obs` tightening** — applied at: `utils.py` `prepare_obs` row (replaced "no image rescale" with explicit `[T=1, B, F]` contract note).
5. **Item 5 — Distribution config (`type`, `validate_args`)** — applied at: `configs/models/dreamer_srl/agent_xs.yaml` (new top-level `distribution:` block).
6. **Item 6 — Actor std hyperparameters (`init_std`, `min_std`, `max_std`)** — applied at: `agent_xs.yaml` actor block (kept for signature parity with comment noting "unused on discrete path").
7. **Item 7 — `discount_model.learnable` + reward/continue head MLP sizing** — applied at: `agent_xs.yaml` (new `reward_model:` and `discount_model:` sub-blocks under `world_model`).
8. **Item 8 — Encoder/decoder/recurrent network sizing keys** — applied at: new "MLP sizing table" section after the YAML block, naming which network reads which `mlp_layers`/`dense_units` for each of 9 networks.
9. **Item 9 — `compute_lambda_values` math signature** — applied at: `utils.py` `compute_lambda_values` row (bootstrap and trim contract, γ-pre-multiplied by caller, length-T return, docstring numpy reference loop requirement).
10. **Item 10 — `Independent(BernoulliSafeMode, 1)` wrap + decoder `dims=1`** — applied at: `loss.py` `BernoulliSafeMode` row + Training-loop semantics §S9.
11. **Item 11 — `Moments` `max_=1.0` semantic + per-rank-local note** — applied at: `agent_xs.yaml` `moments_max` comment block + Risks §11 (new).

### `math-reviewer` (7 items: 1 🔴, 6 🟡; all applied)

1. **🔴 #1 — Checkpoint 5 two-hot bin grid lives in symlog space** — applied at: Checkpoint 5 (rewritten with explicit `bins[0]=-20, bins[127]=0, bins[254]=20` assertions + warning against `symexp(linspace)` storage), cascade table row #2 (rewritten), `loss.py` `TwoHotEncoding` row (rewritten).
2. **🟡 #2 — True-continue splice at imagination step 0** — applied at: Training-loop semantics §S5, cited in `train.py` `one_train_step` row.
3. **🟡 #3 — Action zeroing on `is_first`** — applied at: Training-loop semantics §S4 (the three-quantity reset), `agent.py` RSSM row.
4. **🟡 #4 — Discount weighting on actor loss** — applied at: Training-loop semantics §S6, `train.py` `one_train_step` row.
5. **🟡 #5 — Discount weighting on critic loss** — applied at: Training-loop semantics §S6, `train.py` `one_train_step` row (also confirms critic regression target uses UN-normalised `lambda_values`).
6. **🟡 #6 — Advantage `low`-offset cancellation** — applied at: Training-loop semantics §S7, `train.py` `one_train_step` row.
7. **🟡 #7 — Free-nats floor element-wise BEFORE mean** — applied at: Training-loop semantics §S8, `loss.py` `reconstruction_loss` row.

**Risks §4 (straight-through estimator)** — flipped to RESOLVED 2026-05-12 with explicit `✅ math-reviewer signed off` and the gradient-identity proof reference.

### `code-reviewer` (11 items: 8 🔴, 3 🟡; all applied)

1. **🔴 #1 — `lax.scan` carry signature for RSSM dynamic learning** — applied at: `agent.py` JAX/Flax discipline section (new sub-bullet with carry/input/output spec + code sketch).
2. **🔴 #2 — `LayerNormGRUCell` ONE Linear + ONE LayerNorm, gate order `(reset, cand, update)`, warning against existing 2+2 pattern** — applied at: `agent.py` `LayerNormGRUCell` row (replaced with explicit warning block + corrected fused-gate code sketch + chunk-order callout).
3. **🔴 #3 — `is_first` reset is arithmetic-mask, NOT `jnp.where`; three quantities (action + recurrent + posterior); posterior pre-flattened** — applied at: Training-loop semantics §S4, `agent.py` RSSM row.
4. **🔴 #4 — `get_initial_states` uses transition `mode`, NO PRNG, fully deterministic** — applied at: `agent.py` RSSM row (new `get_initial_states` code sketch).
5. **🟡 #5 — PRNG sub-key threading discipline (~94 keys per step at XS)** — applied at: `train.py` JAX/Flax discipline section (new sub-bullet with split-count math + code sketch + Z3-bug warning).
6. **🔴 #6 — `Player` is interpretation (B) — function namespace, NOT NNX module with own params** — applied at: `agent.py` `Player` row (rewritten with `DreamerSrlState` pytree showing `(h, z)` carry + explicit "reject interpretation (A)").
7. **🟡 #7 — Polyak fires BEFORE `one_train_step`; `train_step` = cumulative gradient steps** — applied at: `train.py` `polyak_update` row (cadence code sketch + ordering callout).
8. **🔴 #8 — `Moments` is `flax.struct.dataclass`, NOT `nnx.Variable` mutation; warning against existing class** — applied at: `utils.py` `Moments` row (rewritten with pure-functional code sketch + explicit warning against existing `dreamer_v3_util.py:Moments`).
9. **🟡 #9 — `defaults_from:` is not a real Config mechanism; pick option (c) loader helper** — applied at: `01_food_only.yaml` "Note for developer" section (decision locked to option (c) with 5-line helper code).
10. **🟡 #10 — `per_rank_*` config-key naming verbatim or rename** — applied at: Risks §14 (new — decision locked to keep verbatim sheeprl names) + YAML header comment.
11. **🟢 #11 — `lax.stop_gradient` JAX form for `discount`** — applied at: `train.py` `one_train_step` row (explicit `jax.lax.stop_gradient(jnp.cumprod(...) / gamma)` code).

**Auxiliary observations also applied** (cited by code-reviewer §"Auxiliary observations"):

- Hafner constant precision (`0.87962566103423978` not `0.8796`) — applied at: `utils.py` `init_weights` row + Risks §5.
- No-import-from-existing-Dreamer rule — applied at: new Risks §13.
- `learning_starts` counted in iterations (sheeprl convention) — applied at: YAML comment block on `learning_starts`.

### Cross-cutting additions

- **Training-loop semantics section** — new subsection above File Changes, holding §S1–§S10 globally so the developer reads all silent omissions in one place before file-by-file work.
- **MLP sizing table** — new section after `agent_xs.yaml` (resolves professor #8 ambiguity about which network reads which sizing knob).
- **Checkpoints 2b, 4b, 9b** — new directly-test-the-omission checkpoints (action shift, `is_first` reset, random-action prefill).
- **Risks §11–§14** — new entries (Moments-max effect, Ratio Python-side enforcement, no-import-from-existing-Dreamer rule, per_rank naming convention).

## Senior-Developer Notes on v2 Pass

No NEW issues beyond what the three reviewers flagged were surfaced during this mechanical edit pass. Every change above traces to a numbered reviewer item or a reviewer auxiliary observation. The plan is now ready for a quick re-audit by the same three reviewers to confirm zero residual deviations; if clean, `developer` can begin Step 1 of the Implementation order.

---

## v2 Re-audit Verdicts (2026-05-12)

The same three reviewers re-audited the v2 plan. **All three signed off ✅ PASS.** 29/29 deviations resolved at developer-visible locations. No new errors introduced by the v2 mechanical edit pass.

| Reviewer | v1 deviations | v2 status | New errors | Companion file | Verdict |
|---|---|---|---|---|---|
| `professor-rl-bayesian-dl` | 11 | 11 ✅ | 0 | [`review_professor_rl_bayesian_dl_v2.md`](review_professor_rl_bayesian_dl_v2.md) | ✅ PASS — ready for implementation |
| `math-reviewer` | 7 (1 🔴 + 6 🟡) | 7 ✅ | 0 | [`review_math_v2.md`](review_math_v2.md) | ✅ PASS — mathematically faithful; 🔴 Checkpoint-5 bin-grid fix verified at three sites |
| `code-reviewer` | 11 (8 blockers + 3 concerns) | 11 ✅ | 0 | [`review_code_v2.md`](review_code_v2.md) | ✅ PASS — implementation-ready; 4 highest-risk silent-pattern-match traps each carry explicit ⚠️ warnings |

**Residual nits noted (none blocks implementation)**:

- (`professor-rl-bayesian-dl` v2) `learning_starts` unit inconsistency — text variously implies policy-steps vs. iterations. Developer should reconcile in Implementation Step 9 against sheeprl `dreamer_v3.py` semantics (sheeprl uses policy-steps).
- (`math-reviewer` v2) Lambda-return LaTeX at line ≈248 has `v_{t+1}` where it should read `v_t`. The mandatory numpy reference loop in the docstring is correct and will catch the typo at Checkpoint 1.
- (`code-reviewer` v2) Three pseudocode nits: `collect_step` JIT-boundary unstated (Python-side per existing Dreamer's `Ratio` precedent), `batch_size` vs `batch_shape` parameter-name inconsistency in `get_initial_states` sketch, `WorldModel` ↔ `DreamerSrlState` `nnx.split` / `nnx.merge` boundary unstated (follow existing `dreamer_v3_trainer.py` pattern).

**Status**: plan is implementation-ready. The next step is for `developer` to begin Implementation Step 1 (read existing NNX conventions in `src/models/dreamer_v3_nnx.py` + `dreamer_v3_trainer.py`) and follow the file-by-file checkpoint protocol.
