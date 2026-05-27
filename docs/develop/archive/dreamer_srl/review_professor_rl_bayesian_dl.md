---
title: "dreamer-srl plan review — algorithm-level audit against sheeprl walkthrough"
topic: dreamer
status: superseded
created: 2026-05-12
last_updated: 2026-05-12
reviewer: professor-rl-bayesian-dl
audited_doc: docs/develop/archive/dreamer_srl/IMPLEMENTATION_PLAN.md
---

> **Superseded by**: [`docs/pi/calls/2026-05-12_dreamer_backend.md`](../../../pi/calls/2026-05-12_dreamer_backend.md) — PI pivot to sheeprl-direct (Option 1) shelves the dreamer-srl plan this review audits.

# dreamer-srl plan review (algorithm + Bayesian-DL scope)

## Purpose

This memo audits the `dreamer-srl` re-implementation plan at [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)
against the line-by-line walkthrough of community sheeprl's DreamerV3 at
[`docs/project/references/sheeprl_dreamer_v3/`](../../../project/references/sheeprl_dreamer_v3/).
The user's directive is **bit-identical replication** of sheeprl's DreamerV3 — no missing parts, no
arbitrary additions, no algorithm-level "improvements." The existing in-house JAX DreamerV3
(`src/models/dreamer_v3_*.py`) is left alone as a control. This review is at the algorithm /
architecture / hyperparameter-mapping level only — JAX/NNX correctness goes to `code-reviewer`,
equation-vs-paper faithfulness goes to `math-reviewer`.

What the plan does well: every one of the five known divergence items from the cascade
("cascade #2 paper-canonical two-hot bins", "cascade #27 zero-init reward+critic heads",
"cascade #28 GRU reset-gate gates the candidate", "cascade #29 critic self-EMA regulariser",
"cascade #30 RSSM prior/posterior with a hidden layer") is correctly named and placed in the
right module. The RSSM math, the KL-balancing math, the imagination + λ-return + Moments-normalised
actor loop, the Polyak EMA target critic, the Hafner Xavier-fan-avg truncated-normal init with
the 0.87962566103423978 constant, the unimix categorical-collapse guard, the straight-through
estimator, and the free-nats KL floor are all named correctly. The non-goal list correctly walls
off continuous actions, MLflow, memmap, Hydra, `gym.vector`, `RestartOnException`,
`fabric.all_gather`, and `EpisodeBuffer`, plus the prohibition on NMN/FiLM/precision hooks.

**Where the plan diverges from the walkthrough — eleven items below.** None of them are
algorithm-breaking on their own, but several are silent omissions that the implementer would
otherwise default-fill in a way that breaks bit-identity at the Checkpoint-8 forward-parity gate.
The most important ones are: (a) the missing `is_first[0] = 1` forced-set inside the training
step, (b) the missing prepend-zero-action shift inside the training step, (c) the missing
`learning_starts` random-action prefill before policy collection, (d) the `prepare_obs`
behaviour that the plan reads as a no-op when in fact it does an MLP-key reshape, (e) two
hyperparameter omissions from `agent_xs.yaml` (`actor.init_std`, `actor.min_std`, `actor.max_std`,
the `distribution.validate_args`-equivalent or `distribution.type` field, and the
`agent.world_model.discount_model.learnable` flag), and (f) one structural error in the
existing `compute_lambda_values` description that, if replicated verbatim, would compute a
slightly different λ-return than sheeprl does.

The headline verdict appears at the bottom (§Verdict). Eleven items below should be folded
into the plan before any code is written.

---

## §1. Numbered deviation list

### Item 1 — Missing: `is_first[0] = 1` forced-set inside the training step

- **Deviation type**: Missing
- **Plan location**: `IMPLEMENTATION_PLAN.md` §File Changes → CREATE `src/algorithms/dreamer_srl/train.py` → row `one_train_step` (lines 234 of the plan). The plan describes "Dynamic learning ... roll the RSSM forward over `[T, B]`" but does not name the boundary trick.
- **Walkthrough location**: [`dreamer_v3.md` line 133](../../../project/references/sheeprl_dreamer_v3/dreamer_v3.md) `data["is_first"][0, :] = torch.ones_like(data["is_first"][0, :])`.
- **What's wrong**: Inside `train()`, sheeprl unconditionally overwrites the first time-step's
  `is_first` to all ones — this guarantees the RSSM's recurrent and stochastic carry both get
  reset to the learned initial state at the first step of every sampled chunk, irrespective of
  what `is_first` was stored as in the buffer for that index. Without this line, the RSSM
  carry at t=0 is whatever zero-tensor or stale value the trainer initialised, and the
  posterior at t=0 is computed against an arbitrary `h₀`, not against the learned
  $\tanh(\bar h_0)$. The first-step posterior NLL and KL would diverge from sheeprl by an
  amount that grows with sequence length (since the error propagates through the GRU). This
  is one of the more failure-mode-prone omissions because the bug is silent — losses still
  go down, just to a different basin.
- **Proposed correction**: Add an explicit step in `one_train_step` of `src/algorithms/dreamer_srl/train.py`
  before the dynamic-learning rollout:
  $$\texttt{batch["is\_first"]} = \texttt{batch["is\_first"]}.\texttt{at}[0].\texttt{set}(1.0)$$
  and document it as "force-set is_first at t=0 — replicates `dreamer_v3.py:133`."

### Item 2 — Missing: prepend-zero-action shift inside the training step

- **Deviation type**: Missing
- **Plan location**: `IMPLEMENTATION_PLAN.md` §File Changes → CREATE `src/algorithms/dreamer_srl/train.py` → row `one_train_step`. The plan describes the dynamic-learning sub-phase as "roll the RSSM forward over `[T, B]` using `lax.scan`" but does not name the action shift.
- **Walkthrough location**: [`dreamer_v3.md` line 137](../../../project/references/sheeprl_dreamer_v3/dreamer_v3.md):
  ```python
  batch_actions = torch.cat((torch.zeros_like(data["actions"][:1]), data["actions"][:-1]), dim=0)
  ```
- **What's wrong**: The actions used inside the dynamic-learning rollout are shifted by one
  step — at time $t$, the RSSM consumes the action that *led to* observation $o_t$ (i.e.
  $a_{t-1}$), not the action *chosen at* $o_t$. Sheeprl achieves this by prepending a zero action
  and dropping the last. Without this, the RSSM is fed actions one step ahead of where the
  paper formulation specifies. The first-step action becomes $\mathbf 0$, which combined with
  $\texttt{is\_first}=1$ becomes $\mathbf 0 \cdot \mathbf 0$ — a clean reset. This is
  semantically load-bearing, not cosmetic: with the wrong shift, the prior $p_\phi(z_t \mid
  h_t)$ predicts the wrong distribution.
- **Proposed correction**: In `one_train_step` of `src/algorithms/dreamer_srl/train.py`, before
  the scan, compute
  $$\texttt{shifted\_actions} = \texttt{jnp.concat}\bigl[\texttt{jnp.zeros\_like}(\texttt{actions}[:1]),\; \texttt{actions}[:-1]\bigr]_{\text{axis}=0}$$
  and feed `shifted_actions` (not `actions`) into the RSSM rollout. Document as
  "replicates `dreamer_v3.py:137`."

### Item 3 — Missing: `learning_starts` random-action prefill phase

- **Deviation type**: Missing
- **Plan location**: `IMPLEMENTATION_PLAN.md` §File Changes → `collect_step` row (line 235) and §"EDIT `train.py`" branch (line 392ff). The plan reads `agent.learning_starts: 1024` from YAML but never uses it.
- **Walkthrough location**: [`dreamer_v3.md` lines 604–617, 706](../../../project/references/sheeprl_dreamer_v3/dreamer_v3.md). Sheeprl's `main()` has:
  ```python
  if iter_num <= learning_starts and cfg.checkpoint.resume_from is None:
      real_actions = actions = np.array(envs.action_space.sample())  # random
  else:
      real_actions = actions = player.get_actions(...)               # policy
  ```
  and gradient updates only fire `if iter_num >= learning_starts:` (line 706).
- **What's wrong**: For the first 1024 policy steps, sheeprl uses uniform-random actions
  to prefill the replay buffer before the policy is ever queried. Gradient updates also do
  not fire until `iter_num >= learning_starts`. The plan's `collect_step` description shows
  only `player.get_actions(...)` — no random-action branch — and the YAML's
  `learning_starts: 1024` is loaded but not wired into either the collection branch or the
  gradient-step gate. The omission means dreamer-srl starts training a stale-init policy
  against itself for the first 1024 steps, biasing replay toward the random-init policy's
  trajectories and changing the world-model's curriculum compared to sheeprl. On a fast-saturating
  task like food-only NoPred (500-step cap, sheeprl saturates by ~50k steps), this may not
  destroy parity outright, but it will move the survival trajectory off the sheeprl reference
  curve.
- **Proposed correction**: In `collect_step` and/or the top-level `train.py` branch
  (whichever owns the action-selection switch), add:
  $$a_t \;=\; \begin{cases} \text{uniform sample from } \mathcal A & \text{if policy\_step} < \texttt{learning\_starts}\\ \pi_\theta(\cdot \mid h_t, z_t) & \text{otherwise} \end{cases}$$
  And gate the gradient-step call on `policy_step >= learning_starts`. Document in plan
  §Implementation order Step 9 (the `train.py` edit) as a required addition. The `Ratio`
  scheduler's `pretrain_steps=0` (default) is already correct — sheeprl XS does not do a
  pretrain burst, just the random-action prefill.

### Item 4 — Wrong value: `prepare_obs` described as a no-op for MLP keys

- **Deviation type**: Wrong Value (description does not match what sheeprl does)
- **Plan location**: `IMPLEMENTATION_PLAN.md` §File Changes → `prepare_obs` row in `utils.py` (line 144). Quoted: *"For us: only the MLP path matters (our env is vector-obs), so just `jnp.asarray(obs).reshape(1, num_envs, -1)`. No image rescale code path."*
- **Walkthrough location**: [`utils.md` lines 171–183](../../../project/references/sheeprl_dreamer_v3/utils.md) and the live source: `torch_obs[k] = torch_obs[k].view(1, num_envs, -1)` for non-CNN keys.
- **What's wrong**: The plan is actually correct in intent — the reshape `(1, num_envs, -1)`
  *is* what sheeprl does for MLP keys. But the framing "no image rescale" reads as "we don't
  need to do anything," which downplays the load-bearing reshape. The leading `T=1` axis
  is the RSSM's input contract — the encoder and RSSM expect `[T, B, ...]`. If the
  implementer skips the reshape, the encoder will silently broadcast and produce a wrong-shape
  embedding. Flag this in the plan to be safe.
- **Proposed correction**: Tighten the description to *"Replicates `utils.py:171-183`. For
  each MLP obs key, write `jnp.asarray(v).reshape(1, num_envs, -1)`. The leading `T=1` axis
  is mandatory — the RSSM consumes `[T, B, ...]`."*

### Item 5 — Missing: actor distribution-type config (`distribution.type` and validate_args)

- **Deviation type**: Missing
- **Plan location**: `IMPLEMENTATION_PLAN.md` §"CREATE `configs/models/dreamer_srl/agent_xs.yaml`" (line ~253). The actor block has `ent_coef`, `action_clip`, `clip_gradients`, etc., but no `distribution.type` and no `distribution.validate_args` keys.
- **Walkthrough location**: [`agent.md` line 1066](../../../project/references/sheeprl_dreamer_v3/agent.md):
  ```python
  self.distribution = distribution_cfg.get("type", "auto").lower()
  ```
  and `tmp/sheeprl/sheeprl/configs/distribution/default.yaml`:
  ```yaml
  validate_args: False
  ```
  with `cfg.distribution` passed into `Actor.__init__(..., distribution_cfg=cfg.distribution)`
  in `agent.py` line 1511 (also into `RSSM(distribution_cfg=cfg.distribution, ...)` line 1426).
- **What's wrong**: Sheeprl reads a top-level `cfg.distribution` group (not under `algo`) that
  provides at minimum `validate_args: False` and, in the actor's `__init__`, a
  `type: "auto"` (defaults to `"discrete"` for discrete envs via `Actor.__init__`'s
  `if self.distribution == "auto": ... self.distribution = "discrete"` branch — `agent.md`
  line 1074). The plan does not surface this key, so a developer reading the plan will not
  know whether to set `type` explicitly to `"discrete"` or to mimic the `"auto"` fallback.
  For a single-`Discrete` env this is a no-op — the actor always uses `OneHotCategoricalStraightThrough`
  for discrete actions — but the plan should still name the key for transparency and so the
  no-fallback-defaults rule (CLAUDE.md) is satisfied.
- **Proposed correction**: Add to `configs/models/dreamer_srl/agent_xs.yaml`:
  ```yaml
  # === Distribution config (matches sheeprl distribution/default.yaml) ===
  distribution:
    type: "auto"                # sheeprl auto → discrete for our Discrete env
    validate_args: false        # sheeprl distribution/default.yaml:1
  ```
  And in `build_state` / `build_agent`, read `agent.distribution.type` and
  `agent.distribution.validate_args` via `config.get_mandatory` per §Risks 10.

### Item 6 — Missing: actor std hyperparameters (`init_std`, `min_std`, `max_std`)

- **Deviation type**: Missing
- **Plan location**: `IMPLEMENTATION_PLAN.md` §"CREATE `configs/models/dreamer_srl/agent_xs.yaml`" actor block (lines 302–311). Has `ent_coef`, `action_clip`, `clip_gradients`, `lr`, `eps`, `moments_*`. Missing `init_std`, `min_std`, `max_std`.
- **Walkthrough location**: [sheeprl `dreamer_v3.yaml` lines 120–122](../../../../tmp/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml) — the YAML carries `min_std: 0.1`, `max_std: 1.0`, `init_std: 2.0`. [`agent.md` lines 1053–1098](../../../project/references/sheeprl_dreamer_v3/agent.md) — `Actor.__init__` stores all three and uses them inside `Actor.forward` for the continuous branch.
- **What's wrong**: Strictly speaking, these three keys are only consumed in the continuous
  branch (`scaled_normal` / `normal` / `tanh_normal`), and dreamer-srl is discrete-only per
  Non-goal #2. *If* `Actor.__init__` is faithfully translated, it still reads these three
  arguments from config and stores them on `self.init_std / self.min_std / self.max_std` even
  for the discrete path — sheeprl does. Omitting them from `agent_xs.yaml` would fail the
  `Config.get_mandatory` calls inside the translated `Actor.__init__`. Either (a) keep them
  in YAML for symmetry with sheeprl, or (b) prune them from the translated `Actor.__init__`
  signature with an explicit comment ("discrete-only — these args dropped from sheeprl
  signature"). Pick one and name it.
- **Proposed correction**: Add to `configs/models/dreamer_srl/agent_xs.yaml` actor block (preferred —
  preserves bit-identity with the sheeprl signature):
  ```yaml
    init_std: 2.0                 # sheeprl dreamer_v3.yaml:122 (unused on discrete path; kept for signature parity)
    min_std: 0.1                  # sheeprl dreamer_v3.yaml:120
    max_std: 1.0                  # sheeprl dreamer_v3.yaml:121
  ```

### Item 7 — Missing: `discount_model.learnable` flag

- **Deviation type**: Missing
- **Plan location**: `IMPLEMENTATION_PLAN.md` §"CREATE `configs/models/dreamer_srl/agent_xs.yaml`" world_model block (lines 281–299). Missing.
- **Walkthrough location**: [sheeprl `dreamer_v3.yaml` lines 103–108](../../../../tmp/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml):
  ```yaml
    discount_model:
      learnable: True
      dense_act: ${algo.dense_act}
      mlp_layers: ${algo.mlp_layers}
      ...
  ```
- **What's wrong**: Sheeprl exposes `world_model.discount_model.learnable: True`, plus the
  continue-head MLP sizing keys (`dense_act`, `mlp_layers`, `layer_norm`, `dense_units`) that
  default-interpolate from the top-level algo settings. The plan elides these — it has
  `world_model.continue_scale_factor` but not the network-sizing for the continue head. Per
  the walkthrough, the continue head is an `MLP(latent_state_size → 1, hidden_sizes=[dense_units]*mlp_layers)`
  with the same dense_units as the rest of the world model. Without these YAML keys the
  developer would invent sizings.
- **Proposed correction**: Add to `configs/models/dreamer_srl/agent_xs.yaml` world_model block:
  ```yaml
    discount_model:
      learnable: true               # sheeprl dreamer_v3.yaml:104
      dense_act: "silu"             # sheeprl dreamer_v3.yaml:105 (inherits dense_act)
      mlp_layers: 1                 # sheeprl XS dense — 1 hidden layer
      dense_units: 256              # sheeprl XS — same as the rest
  ```
  Also do the same for the reward_model network sizing (currently the plan only carries
  `reward_model.bins`, `reward_low`, `reward_high`, `world_model_lr`, `world_model_eps`,
  but not `dense_act`, `mlp_layers`, `dense_units` — these are needed because in sheeprl
  the reward head also has `mlp_layers=5` (or `mlp_layers=1` in XS) hidden layers, not
  zero. See `agent.md` line 1468: `hidden_sizes=[world_model_cfg.reward_model.dense_units] * world_model_cfg.reward_model.mlp_layers`.

### Item 8 — Missing: encoder/decoder/recurrent network sizing keys

- **Deviation type**: Missing
- **Plan location**: `IMPLEMENTATION_PLAN.md` §"CREATE `configs/models/dreamer_srl/agent_xs.yaml`" (lines 274–294). The plan has top-level `dense_units: 256`, `mlp_layers: 1`, `dense_act: "silu"`, `layer_norm_eps: 1.0e-3`, and `world_model.recurrent_state_size: 256`, `world_model.transition_hidden_size: 256`, `world_model.representation_hidden_size: 256`. Missing the per-component sub-trees that sheeprl uses.
- **Walkthrough location**: [sheeprl `dreamer_v3.yaml` lines 56–108](../../../../tmp/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml):
  ```yaml
  world_model:
    encoder:
      mlp_layers: ${algo.mlp_layers}
      dense_units: ${algo.dense_units}
      dense_act: ${algo.dense_act}
      ...
    recurrent_model:
      recurrent_state_size: 4096    # 256 in XS
      dense_units: ${algo.dense_units}
      layer_norm: ${algo.mlp_layer_norm}
    transition_model:
      hidden_size: 1024              # 256 in XS
      dense_act: ${algo.dense_act}
      layer_norm: ${algo.mlp_layer_norm}
    representation_model:
      hidden_size: 1024              # 256 in XS
      dense_act: ${algo.dense_act}
      layer_norm: ${algo.mlp_layer_norm}
    observation_model:               # = decoder
      mlp_layers: ${algo.mlp_layers}
      dense_units: ${algo.dense_units}
      dense_act: ${algo.dense_act}
      mlp_layer_norm: ${algo.mlp_layer_norm}
    reward_model:
      dense_act: ${algo.dense_act}
      mlp_layers: ${algo.mlp_layers}
      layer_norm: ${algo.mlp_layer_norm}
      dense_units: ${algo.dense_units}
      bins: 255
    discount_model: { ... }
  ```
- **What's wrong**: The plan flattens these per-component knobs into top-level `dense_units` /
  `mlp_layers`. That's defensible because sheeprl's YAML uses `${algo.dense_units}`
  interpolation to share them anyway, but the developer must still wire each network's
  *number of hidden layers* to the right value. In sheeprl XS, the **encoder, decoder,
  reward, continue heads all have `mlp_layers=1` hidden layer**, the recurrent model has
  one pre-projection layer, and the **transition/representation models have one hidden layer**
  (cascade fix #30 — already named in the plan). The plan's `mlp_layers: 1` at the top
  level *should* propagate to all of these, but it is not explicit which networks consume
  the top-level vs. their own override.
- **Proposed correction**: Either expand the YAML to mirror sheeprl's nested structure
  (preferred for bit-identity) or add an explicit table to the plan saying:
  | Network | `mlp_layers` consumed | hidden_dim | comment |
  |---|---|---|---|
  | Encoder | top-level (1) | top-level (256) | `MLPEncoder` only — single MLP key `state` |
  | Decoder | top-level (1) | top-level (256) | `MLPDecoder` only |
  | Recurrent pre-proj | (hardcoded 1) | top-level (256) | The `RecurrentModel` MLP before the GRU |
  | Transition (prior) | (hardcoded 1) | `transition_hidden_size` (256) | Cascade fix #30 |
  | Representation (post) | (hardcoded 1) | `representation_hidden_size` (256) | Cascade fix #30 |
  | Reward head | top-level (1) | top-level (256) | Terminal Linear gets `uniform_init_weights(0.0)` |
  | Critic | top-level (1) | top-level (256) | Terminal Linear gets `uniform_init_weights(0.0)` |
  | Continue head | top-level (1) | top-level (256) | Terminal Linear gets `uniform_init_weights(1.0)` |
  | Actor trunk | top-level (1) | top-level (256) | Heads get `uniform_init_weights(1.0)` |
  Pick the table approach and embed it in §File Changes. The current plan is ambiguous on
  reward/continue/decoder MLP sizing.

### Item 9 — Wrong description: the existing `compute_lambda_values` is "line-for-line identical to sheeprl's"

- **Deviation type**: Wrong Value (specifically: ambiguous claim that could mislead the implementer)
- **Plan location**: `IMPLEMENTATION_PLAN.md` §"CREATE `src/algorithms/dreamer_srl/utils.py`" (line 152). Quoted: *"The current `Ratio` is line-for-line identical to sheeprl's so the duplication is small; the current `compute_lambda_values` we should verify against sheeprl's line-for-line before relying on either."*
- **Walkthrough location**: [`utils.md` line 156](../../../project/references/sheeprl_dreamer_v3/utils.md):
  ```python
  def compute_lambda_values(rewards, values, continues, lmbda=0.95):
      vals = [values[-1:]]
      interm = rewards + continues * values * (1 - lmbda)
      for t in reversed(range(len(continues))):
          vals.append(interm[t] + continues[t] * lmbda * vals[-1])
      ret = torch.cat(list(reversed(vals))[:-1])
      return ret
  ```
- **What's wrong**: Two subtleties in this 7-line function that the plan must replicate
  exactly. (a) The recursion's bootstrap is `vals[0] = values[-1:]` (i.e., the **terminal
  value seed** is `values[-1]`, not the next-step value at the boundary). (b) The final
  `[:-1]` slice drops the bootstrap entry. In `dreamer_v3.py:284-289`, this function is
  called with `predicted_rewards[1:]`, `predicted_values[1:]`, `continues[1:] * gamma`,
  *and* the trainer further indexes `[:-1]` on the resulting `lambda_values` to align with
  `baseline = predicted_values[:-1]`. The plan's description of the JAX translation as
  `lax.scan` with `reverse=True` is correct in spirit, but the implementer must mirror the
  *exact* bootstrap and the *exact* trailing `[:-1]` — sheeprl returns a tensor of length
  `len(continues)` (== horizon), not `len(continues) + 1`. JAX `scan` with `reverse=True`
  naturally produces `len(continues) + 1` outputs (the initial carry + each scanned step),
  so the trimming step is essential. Also note: `continues * gamma` is computed by the
  *caller* in sheeprl, so `lmbda` is the only weight inside the function — the plan should
  not bake γ into `compute_lambda_values`'s signature.
- **Proposed correction**: Add to `utils.py`'s docstring a math statement:
  $$\textstyle G^{\lambda}_t \;=\; r_t \;+\; c_t \,\bigl[\,(1-\lambda)\, v_{t+1} \;+\; \lambda\, G^{\lambda}_{t+1}\bigr], \qquad G^{\lambda}_{T} \;=\; v_T$$
  where `continues = mask * gamma` is pre-multiplied by the caller, and the returned tensor
  has length $T$ (the input length, **not** $T+1$). The implementer should add a numpy
  reference loop in the docstring (per Checkpoint 1) and assert the JAX `lax.scan` output
  matches it to `1e-6`. Also remove the phrase *"line-for-line identical to sheeprl's"* about
  the project's current `compute_lambda_values` — verify, do not assert.

### Item 10 — Missing: handling of `Independent(BernoulliSafeMode, 1)` wrapping on continue head

- **Deviation type**: Missing
- **Plan location**: `IMPLEMENTATION_PLAN.md` §"CREATE `src/algorithms/dreamer_srl/loss.py`" (line 217). The plan defines `BernoulliSafeMode.log_prob(target)` as standard BCE-with-logits but does not call out the **`Independent(..., 1)` wrap** sheeprl applies.
- **Walkthrough location**: [`dreamer_v3.md` line 200](../../../project/references/sheeprl_dreamer_v3/dreamer_v3.md):
  ```python
  pc = Independent(BernoulliSafeMode(logits=world_model.continue_model(latent_states)), 1)
  ```
  and line 279: same wrap in the imagination loop.
- **What's wrong**: `Independent(BernoulliSafeMode(...), 1)` re-interprets the trailing axis
  as the **event dim**, so `log_prob` returns a scalar per `(T, B)` cell rather than per
  `(T, B, 1)`. Without the `Independent` wrap, the BCE sum is over the wrong axis and the
  continue loss is `1×` the per-element value instead of a sum-over-event. For our case
  (single-dim Bernoulli, output_dim=1), this is a one-axis distinction that happens to
  produce the same scalar — but only because the trailing axis has size 1. If the continue
  head's `output_dim` is ever changed (e.g., multi-step continue prediction), the
  `Independent` wrap is the only line that keeps the math right. Document explicitly so the
  implementer's translation reads the same shape contract.
- **Proposed correction**: In `loss.py`, document that the continue distribution
  $p_c$ is wrapped as $\text{Independent}_{1}(\text{Bernoulli-safe-mode}(\text{logits}))$, and that
  its `log_prob(target)` reduces the trailing event dim with a sum, returning a `[T, B]`
  array. Add the same wrap requirement to the imagination loop in `train.py`'s critic phase
  (`continues = Independent(BernoulliSafeMode(...), 1).mode`, `dreamer_v3.md` line 279).
  Same comment applies to the **observation reconstruction loss**: sheeprl wraps decoder
  outputs as `MSEDistribution(mode, dims=len(...shape[2:]))` or
  `SymlogDistribution(mode, dims=len(...shape[2:]))` (`dreamer_v3.md` lines 186–193). The
  `dims` argument controls how many trailing axes get summed by `log_prob`. For a
  single-key MLP observation of shape `[T, B, F]`, this is `dims=1`. Plan must name this.

### Item 11 — Missing: `Moments` initial-bias warning, plus dropped `max_` reference value

- **Deviation type**: Missing (cosmetic — flagging for transparency)
- **Plan location**: `IMPLEMENTATION_PLAN.md` §Risks 6, "`Moments` percentile EMA initial value (resolved-in-plan)". Quoted: *"For ~100 first calls the values are biased toward 0 — that is sheeprl's behaviour and we replicate it exactly."*
- **Walkthrough location**: [`utils.md` lines 110–124](../../../project/references/sheeprl_dreamer_v3/utils.md):
  ```python
  def __init__(self, decay=0.99, max_=1e8, percentile_low=0.05, percentile_high=0.95):
      self._max = torch.tensor(max_)
      ...
      self.register_buffer("low", torch.zeros(()))
      self.register_buffer("high", torch.zeros(()))
  ```
  and `forward`: `invscale = torch.max(1 / self._max, self.high - self.low)`.
- **What's wrong**: The sheeprl `Moments` default `max_` is `1e8`, but the plan's
  `agent_xs.yaml` has `moments_max: 1.0` (line 309). The walkthrough at `utils.md` documents
  `max_=1e8` as the **default**, but the sheeprl `dreamer_v3.yaml:134` is `max: 1.0`. Cross-check:
  sheeprl XL overrides default `max_=1e8` to `1.0` in the YAML. So the plan's `1.0` is
  correct *for matching dreamer_v3.yaml*, but the meaning is subtle: `1 / self._max` becomes
  `1.0` so the invscale-floor is `1.0` rather than `1e-8`. This means **the actor advantage
  is never scaled by a denominator smaller than 1**, regardless of how tight the lambda-value
  distribution is. The walkthrough's annotation calls it a "ceiling" but it's actually a
  floor on the invscale (= a ceiling on the scale-down factor). The plan should name this
  precisely so the developer does not silently set it back to `1e8`. Also, the docstring
  should call out that `Moments` is **per-rank in our single-device setup** — sheeprl's
  `fabric.all_gather` is dropped (plan says so in §File Changes line 142, good), but the
  semantic consequence is the percentiles are computed over the local batch only, which is
  identical for `world_size=1` (matches sheeprl XS at single-rank). Just flag this.
- **Proposed correction**: In `agent_xs.yaml`, add a comment:
  ```yaml
    moments_max: 1.0              # sheeprl dreamer_v3.yaml:134 — invscale floor is 1/1.0 = 1.0
                                  # (NOT the upstream default 1e8 — XL deliberately overrides).
  ```
  And in `utils.py:Moments` docstring, write:
  $$\text{invscale}_t \;=\; \max\bigl(1/\text{max}_*\,,\; \text{high}_t - \text{low}_t\bigr)$$
  with $\text{max}_* = 1.0$ for dreamer_v3, so $\text{invscale}_t \geq 1.0$ — meaning when
  the lambda-value spread is tiny the advantage is *not* re-scaled larger than the spread
  itself. Document this since it's a non-obvious effect of the `1.0` value choice.

---

## §2. Items that look like deviations but are correct on closer reading

- **`Independent(OneHotCategoricalStraightThrough, 1)` in the KL term.** The plan's
  description of `categorical_kl(post_logits, prior_logits)` in `loss.py` (line 218) reads
  *"Apply over the `discrete` axis, reshape `[T, B, 32, 32]`, sum the inner two axes after
  KL on the last."* This matches sheeprl `loss.md` lines 97–106:
  `Independent(OneHotCategoricalStraightThrough(logits=...), 1)` — the `1` reinterprets
  the *32-slot stochastic axis* (not the 32-class discrete axis) as the event dim. So KL is
  computed per-`(T, B, slot)` element, then summed over the 32 slots. The plan's "sum the
  inner two axes after KL on the last" is correct: KL operates on the 32 classes (last
  axis), and the 32-slot reduction is the `Independent`-wrap sum. Just verify the
  implementer reads the plan's prose carefully — the math is right.

- **`OneHotCategorical` (no straight-through) for entropy logging.** `dreamer_v3.md` lines
  372/376 use `OneHotCategorical` (the non-STE class) when computing prior/posterior entropy
  for logging only. This is correct — entropy needs no gradient, so the STE wrap is
  optional. The plan does not name the logging step but the metric-name set in §1 of the
  plan covers `State/post_entropy` and `State/prior_entropy`. Good.

- **`prefill_steps = learning_starts - int(learning_starts > 0)`.** `dreamer_v3.md` line 557.
  This off-by-one is to handle the fact that `learning_starts=1024` means "start training at
  step 1024," not "start training at step 1023." The plan does not surface this, but it is
  bookkeeping that the developer translating `main()` will naturally write. Flag only
  if the implementer asks.

- **Use of `1 - terminated` as continue target.** `dreamer_v3.md` line 201:
  `continues_targets = 1 - data["terminated"]`. The plan's `loss.py` docstring (line 217)
  says *"`continue_targets` are typically `(1 - dones) * γ`"* per the sheeprl docstring, but
  the actual call site at line 201 of `dreamer_v3.py` uses just `1 - terminated` (no `* γ`).
  Sheeprl's docstring is misleading and the code is authoritative — the plan inherits the
  ambiguity but the implementer should follow the code (`1 - terminated`, no γ). Document
  this in the plan to head off the confusion. *(Borderline item — could be Item 12 if the
  audit is strict; leaving here as a "look but don't add as a deviation" because the plan
  does reference the docstring rather than the code.)*

- **No CNN encoder/decoder.** Plan §"CREATE `src/algorithms/dreamer_srl/agent.py`" correctly
  flags that `MultiEncoder` / `MultiDecoder` are kept dict-aware with the CNN branch always
  `None`. This matches sheeprl's contract — `agent.md` line 1346 conditions
  `cnn_encoder = CNNEncoder(...) if cnn_keys.encoder else None`. Good.

- **Polyak `tau=1` first call vs. `tau=0.02` subsequent.** Plan line 236 matches `dreamer_v3.md`
  lines 720–726 exactly:
  $$\theta'_{\text{critic}} \leftarrow \tau \theta_{\text{critic}} + (1-\tau) \theta'_{\text{critic}},\qquad \tau = \begin{cases} 1 & \text{first update}\\ 0.02 & \text{subsequent} \end{cases}$$

- **The `Independent(BernoulliSafeMode, 1).mode` call to replace first imagined continue with
  real `1 - terminated`.** `dreamer_v3.md` line 280–281. Plan §File Changes describes this
  in the critic-phase row of `one_train_step` (line 234) but the wording *"first continue
  replaced by the real `1 - terminated` of the seeding step"* is correct.

---

## §3. Hyperparameter coverage table

Walking every key in `tmp/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml` (XL defaults)
and `dreamer_v3_XS.yaml` (XS overrides) against `configs/models/dreamer_srl/agent_xs.yaml` in
the plan. ✓ = present in plan, ✗ = missing, ≈ = present under a different name.

| sheeprl YAML key | plan YAML key | status | notes |
|---|---|---|---|
| `gamma: 0.996996996996997` | `agent.gamma` | ✓ | |
| `lmbda: 0.95` | `agent.lmbda` | ✓ | |
| `horizon: 15` | `agent.horizon` | ✓ | |
| `replay_ratio: 1` | `agent.replay_ratio` | ✓ | |
| `learning_starts: 1024` | `agent.learning_starts` | ✓ in YAML, but not wired (Item 3) | |
| `per_rank_pretrain_steps: 0` | `agent.per_rank_pretrain_steps` | ✓ | |
| `per_rank_sequence_length: ???` | `agent.per_rank_sequence_length: 64` | ✓ | sheeprl marks `???` (must override); plan picks `64` per `jzgkcep4` exp config — good |
| `per_rank_batch_size: ???` | `agent.per_rank_batch_size: 16` | ✓ | same — plan picks `16` per exp config |
| `cnn_keys: encoder, decoder` | `agent.cnn_keys: []` | ✓ | |
| `mlp_keys: encoder, decoder` | `agent.mlp_keys` | ✓ | |
| `cnn_layer_norm: cls, kw` | (CNN unused) | n/a | |
| `mlp_layer_norm: { cls, kw{ eps: 1e-3 } }` | `agent.layer_norm_eps: 1.0e-3` | ≈ | sheeprl uses class+kwarg dispatch; plan flattens to a single eps value. OK because we only have one layer-norm class — but if the developer ever adds a second, this collapses. Document the choice. |
| `dense_units: 1024 (XL) → 256 (XS)` | `agent.dense_units: 256` | ✓ | |
| `mlp_layers: 5 (XL) → 1 (XS)` | `agent.mlp_layers: 1` | ✓ | |
| `dense_act: torch.nn.SiLU` | `agent.dense_act: "silu"` | ✓ | string-vs-class dispatch — translation hazard, but documented in plan §File Changes line 277 |
| `cnn_act: torch.nn.SiLU` | (CNN unused) | n/a | |
| `unimix: 0.01` | `agent.unimix` | ✓ | |
| `hafner_initialization: True` | `agent.hafner_initialization` | ✓ | |
| `world_model.discrete_size: 32` | `agent.world_model.discrete_size` | ✓ | |
| `world_model.stochastic_size: 32` | `agent.world_model.stochastic_size` | ✓ | |
| `world_model.kl_dynamic: 0.5` | `agent.world_model.kl_dynamic` | ✓ | |
| `world_model.kl_representation: 0.1` | `agent.world_model.kl_representation` | ✓ | |
| `world_model.kl_free_nats: 1.0` | `agent.world_model.kl_free_nats` | ✓ | |
| `world_model.kl_regularizer: 1.0` | `agent.world_model.kl_regularizer` | ✓ | |
| `world_model.continue_scale_factor: 1.0` | `agent.world_model.continue_scale_factor` | ✓ | |
| `world_model.clip_gradients: 1000.0` | `agent.world_model.clip_gradients` | ✓ | |
| `world_model.decoupled_rssm: False` | (implicit — not in YAML) | ✗ | Risk #8 of plan flags decoupled_rssm as out-of-scope. Add an explicit `decoupled_rssm: false` line to YAML for transparency. |
| `world_model.learnable_initial_recurrent_state: True` | `agent.world_model.learnable_initial_recurrent_state` | ✓ | |
| `world_model.encoder.{cnn_channels_multiplier, cnn_act, dense_act, mlp_layers, cnn_layer_norm, mlp_layer_norm, dense_units}` | (interpolated from top-level) | ≈ | See Item 8 — needs an explicit "where does each net get its mlp_layers/dense_units" table. |
| `world_model.recurrent_model.recurrent_state_size: 4096 (XL) → 256 (XS)` | `agent.world_model.recurrent_state_size: 256` | ✓ | |
| `world_model.recurrent_model.layer_norm: ${mlp_layer_norm}` | (implicit) | ✗ | Recurrent model's LayerNorm presence is load-bearing — plan §"CREATE `agent.py`" line 184 calls this out ("the cell carries LayerNorm wrapping"). Add to YAML for explicitness. |
| `world_model.recurrent_model.dense_units: ${dense_units}` | (interpolated from top-level) | ≈ | Same as Item 8. |
| `world_model.transition_model.hidden_size: 1024 → 256` | `agent.world_model.transition_hidden_size` | ✓ | |
| `world_model.representation_model.hidden_size: 1024 → 256` | `agent.world_model.representation_hidden_size` | ✓ | |
| `world_model.observation_model.{mlp_layers, dense_units, ...}` | (interpolated) | ≈ | Item 8 |
| `world_model.reward_model.{dense_act, mlp_layers, layer_norm, dense_units, bins: 255}` | `agent.world_model.reward_bins: 255` (only `bins`) | ✗ partial | Item 7 — needs `dense_act/mlp_layers/dense_units` |
| `world_model.discount_model.{learnable, dense_act, mlp_layers, layer_norm, dense_units}` | ✗ | ✗ | Item 7 |
| `world_model.optimizer.{lr: 1e-4, eps: 1e-8, weight_decay: 0}` | `world_model_lr: 1.0e-4`, `world_model_eps: 1.0e-8` | ≈ | Missing `weight_decay: 0` (default for AdamW). Trivial — add. |
| `actor.cls: sheeprl.algos.dreamer_v3.agent.Actor` | (implicit) | ✗ | OK, only one actor class — drop explicitly in plan note "MinedojoActor skipped" |
| `actor.ent_coef: 3e-4` | `agent.actor.ent_coef` | ✓ | |
| `actor.min_std: 0.1` | ✗ | ✗ | Item 6 |
| `actor.max_std: 1.0` | ✗ | ✗ | Item 6 |
| `actor.init_std: 2.0` | ✗ | ✗ | Item 6 |
| `actor.dense_act/mlp_layers/dense_units/layer_norm` | (interpolated) | ≈ | Item 8 |
| `actor.clip_gradients: 100.0` | `agent.actor.clip_gradients` | ✓ | |
| `actor.unimix: ${unimix}` | (interpolated; plan uses top-level) | ✓ | |
| `actor.action_clip: 1.0` | `agent.actor.action_clip` | ✓ | |
| `actor.moments.decay: 0.99` | `agent.actor.moments_decay` | ✓ | |
| `actor.moments.max: 1.0` | `agent.actor.moments_max` | ✓ | But see Item 11 — annotate the effect |
| `actor.moments.percentile.low: 0.05` | `agent.actor.moments_percentile_low` | ✓ | |
| `actor.moments.percentile.high: 0.95` | `agent.actor.moments_percentile_high` | ✓ | |
| `actor.optimizer.{lr: 8e-5, eps: 1e-5, weight_decay: 0}` | `actor_lr: 8.0e-5`, `actor_eps: 1.0e-5` | ≈ | Missing `weight_decay: 0`. Add. |
| `critic.dense_act/mlp_layers/dense_units/layer_norm` | (interpolated) | ≈ | Item 8 |
| `critic.per_rank_target_network_update_freq: 1` | `agent.critic.per_rank_target_network_update_freq` | ✓ | |
| `critic.tau: 0.02` | `agent.critic.tau` | ✓ | |
| `critic.bins: 255` | `agent.critic.bins` | ✓ | |
| `critic.clip_gradients: 100.0` | `agent.critic.clip_gradients` | ✓ | |
| `critic.optimizer.{lr: 8e-5, eps: 1e-5, weight_decay: 0}` | `critic_lr`, `critic_eps` | ≈ | Missing `weight_decay: 0`. Add. |
| `player.discrete_size: ${world_model.discrete_size}` | (implicit) | ✗ | Player needs `discrete_size`. Trivial — but mention. |
| `cfg.distribution.{type, validate_args}` (top-level) | ✗ | ✗ | Item 5 |
| `cfg.buffer.size` | `agent.buffer_size: 1_000_000` | ✓ | |
| `cfg.buffer.memmap`, `cfg.buffer.checkpoint`, `cfg.buffer.from_numpy`, `cfg.buffer.validate_args` | (out-of-scope per non-goals) | ✓ | |

The critical takeaway from this table: **~10 sheeprl YAML keys are either missing from the
plan's YAML or only implicit through interpolation**. Items 5, 6, 7, 8, 11 collectively name
all of them. Folding them in is YAML-bookkeeping work, not algorithm work — but they need to
land before the developer starts implementation.

---

## §4. The five cascade items — confirmation pass

| # | Item | Plan correctness | Notes |
|---|---|---|---|
| #2 | Two-hot bins as `symexp(linspace(-20, +20, 255))` in symlog space | ✓ correct | Plan §File Changes line 214 + Checkpoint 5 + YAML lines 296–297 + 321–322 all consistent. |
| #27 | Zero-init reward + critic terminal Linear via `uniform_init_weights(0.0)` | ✓ correct | Plan §File Changes line 197 enumerates every head's init scale exactly. |
| #28 | `cand = tanh(reset * cand_proj)` (reset gate gates candidate, not recurrence sum) | ✓ correct | Plan §File Changes line 184 + Checkpoint 2. The math also nails the update-gate `sigmoid(update - 1.0)` keep-old-state bias. |
| #29 | Critic value loss = `-qv.log_prob(λ.detach()) - qv.log_prob(target_value.detach())` | ✓ correct | Plan §File Changes line 234 (critic phase) + Checkpoint 6. |
| #30 | RSSM prior + posterior heads have one hidden layer (`MLP(hidden_sizes=[hidden_size])`) | ✓ correct | Plan §File Changes line 197 (transition/representation final-layer init) + Checkpoint 4 + YAML lines 293–294. |

All five cascade items are named, placed, and gated by a numbered Checkpoint. This is the
strongest part of the plan.

---

## §5. Non-goal consistency check

| Non-goal | Verified absent | Notes |
|---|---|---|
| No NMN / FiLM / precision-modulation hooks | ✓ | No injection points in any file. §Implementation order Step 1 reads NNX conventions from `src/models/dreamer_v3_nnx.py` for *style* only — does not import or wrap. Plan line 70 explicit: *"dreamer-srl does NOT reuse anything from `src/models/dreamer_v3_*.py`"*. |
| No continuous-action support | ✓ | Plan §File Changes line 193 explicit: *"continuous-action paths — we skip continuous entirely"*. The `TruncatedNormal` / `TruncatedStandardNormal` block of `distribution.py` is correctly named as out-of-scope. ⚠ But see Item 6 — sheeprl's `Actor.__init__` reads `init_std`, `min_std`, `max_std` from config even on the discrete path. Either keep the keys in YAML (recommended for signature parity) or prune them from the translated `Actor` and document. |
| No MLflow | ✓ | `log_models_from_checkpoint` correctly out-of-scope. |
| No memmap | ✓ | §"CREATE `buffers.py`" line 163 explicit. |
| No Hydra | ✓ | Plan §"CREATE `train.py`" line 240 uses `optax.adam` directly. |
| No `gym.vector.SyncVectorEnv` | ✓ | Uses our `ParallelEnv` via `jax_step`. |
| No `RestartOnException` | ✓ | Plan §"CREATE `train.py`" line 235 explicit: *"our JAX env doesn't raise mid-step"*. |
| No `fabric.all_gather` | ✓ | Plan §File Changes line 142 explicit: drop `all_gather` from `Moments`. |
| No `EpisodeBuffer` | ✓ | Plan §"CREATE `buffers.py`" line 163 explicit. |

The decoupled_rssm branch is correctly skipped per plan Risk #8.

---

## §6. Arbitrary additions (in plan but not in walkthrough)

Walking the plan looking for items that the walkthrough does **not** name.

| Plan item | Walkthrough has this? | Verdict |
|---|---|---|
| `agent.buffer_device: cpu` YAML key | No — sheeprl always uses CPU numpy rings | Defensible: plan explicitly justifies as a single new key to head off a developer choice that would diverge (line 165). Not algorithm-shaping. Allow. |
| `scripts/dreamer_srl_offline_check.py` cross-framework parity script | No | Allow — pure diagnostic, not part of the training loop. Plan flags as "optional but recommended" (line 437). |
| Checkpoint 8 cross-framework forward-pass parity check | No (this is a project-specific diagnostic) | Allow — this is exactly the right way to verify bit-identity. The tolerance `1e-4` is reasonable. |
| `apply_gru_reset_gate` debug flag in Checkpoint 2 | No (sheeprl has no such flag) | Acceptable as a temporary checkpoint debug flag *if removed before commit* — plan line 476 says "After the check, delete the debug flag." |
| Second-term-off debug flag for critic self-EMA in Checkpoint 6 | No | Same as above — acceptable if removed (plan line 480 says so). |
| `agent.algorithm: "dreamer-srl"` YAML field | No — sheeprl uses Hydra `name: dreamer_v3` | Allow — needed for our `train.py` dispatch. Not algorithm-shaping. |
| `defaults_from:` YAML merge key | No (Hydra-specific) | Plan §"CREATE `configs/models/dreamer_srl/01_food_only.yaml`" line 367 already flags this as a "developer confirms by reading get_default_config()". Mark as **resolve before implementation**, not arbitrary. |

**No arbitrary algorithmic additions found.** Every algorithm-shaping line traces back to a
sheeprl source line. The plan is disciplined on this dimension.

---

## §7. Math summary — the algorithm dreamer-srl is replicating

For completeness, the canonical math the plan must implement (all symbols match Hafner et al.
2023, [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)).

**World-model loss** (Eq. 5 of the paper, sheeprl `loss.py:80`):
$$
\mathcal L_{\text{wm}}(\phi)
= \mathbb E_{q_\phi}\Bigl[
  \beta_{\text{KL}} \bigl( \beta_{\text{dyn}} \max(1, \text{KL}[\, \mathrm{sg}(q) \,\|\, p\,]) + \beta_{\text{rep}} \max(1, \text{KL}[\, q \,\|\, \mathrm{sg}(p)\,]) \bigr)
  + \sum_k \mathcal L^k_{\text{recon}} + \mathcal L_{\text{reward}} + \mathcal L_{\text{cont}}
\Bigr]
$$
with $\beta_{\text{dyn}} = 0.5$, $\beta_{\text{rep}} = 0.1$, $\beta_{\text{KL}} = 1.0$,
free_nats $= 1.0$. The KL is analytic between two
`Independent(OneHotCategoricalStraightThrough(logits), 1)` over 32 slots of 32 classes.

**Critic loss** (Eq. 10 of the paper, sheeprl `dreamer_v3.py:345-349`):
$$
\mathcal L_{\text{critic}}(\psi)
= \mathbb E_t\bigl[\, w_t \bigl(\, -\log q_\psi(G^{\lambda}_t \mid s_t)  - \log q_\psi(v_{\bar\psi}(s_t) \mid s_t) \,\bigr) \,\bigr]
$$
where $q_\psi$ is the `TwoHotEncodingDistribution`, $v_{\bar\psi}$ is the EMA-target critic
mean, and $w_t$ is the cumulative discount $\prod_{t' < t}(\gamma c_{t'})$.

**Actor loss** (Eq. 11 of the paper, sheeprl `dreamer_v3.py:305-330`), discrete branch only:
$$
\mathcal L_{\text{actor}}(\theta)
= -\mathbb E_t \Bigl[\, w_t \bigl( \log \pi_\theta(a_t \mid s_t) \cdot \mathrm{sg}\, \hat A_t  +  \eta \, \mathcal H[\pi_\theta(\cdot \mid s_t)] \bigr) \,\Bigr]
$$
with normalised advantage
$$
\hat A_t = \frac{G^\lambda_t - \mu^{0.95}_{\text{EMA}}}{\max(1, \mu^{0.95}_{\text{EMA}} - \mu^{0.05}_{\text{EMA}})} - \frac{v_t - \mu^{0.95}_{\text{EMA}}}{\max(1, \mu^{0.95}_{\text{EMA}} - \mu^{0.05}_{\text{EMA}})}
$$
where the percentile EMAs come from `Moments` and $\eta = 3\!\times\!10^{-4}$.

**λ-return** (sheeprl `utils.py:66-77`, backward recursion):
$$
G^{\lambda}_t = r_t + c_t \bigl[(1-\lambda) v_{t+1} + \lambda G^{\lambda}_{t+1}\bigr],\qquad G^\lambda_T = v_T
$$
with $c_t = \gamma \cdot \mathrm{mask}_t$ pre-multiplied by the caller.

**Polyak EMA target critic** (sheeprl `dreamer_v3.py:720-726`):
$$
\theta'_{\text{critic}} \leftarrow \tau \theta_{\text{critic}} + (1-\tau) \theta'_{\text{critic}},\qquad \tau = 1 \text{ on first call, else } 0.02
$$

**RSSM dynamic** (sheeprl `agent.py:396-444`):
$$
\begin{aligned}
h_t &= \mathrm{LayerNormGRUCell}\bigl([z_{t-1}; a_{t-1}],\, h_{t-1}\bigr) \\
p_\phi(z_t \mid h_t) &= \mathrm{unimix}(\mathrm{softmax}(\mathrm{MLP}_{\text{trans}}(h_t)))) \\
q_\phi(z_t \mid h_t, x_t) &= \mathrm{unimix}(\mathrm{softmax}(\mathrm{MLP}_{\text{repr}}([h_t; e(x_t)])))) \\
z_t &= \mathrm{STE}(\text{OneHot-Categorical}(q_\phi))
\end{aligned}
$$
with the $\texttt{is\_first}$ reset:
$$
h_t \leftarrow (1-\texttt{is\_first}) \cdot h_t + \texttt{is\_first} \cdot \tanh(\bar h_0)
$$
and the analogous reset for the previous posterior, where $\bar h_0$ is the learnable initial
recurrent state.

**LayerNormGRUCell** (sheeprl `models.py:370-651`):
$$
\begin{aligned}
[r,\, c,\, u] &= \mathrm{chunk}_3\bigl(\mathrm{LayerNorm}(W [\, h_{t-1};\, x_t\,] + b)\bigr) \\
r &= \sigma(r),\quad c = \tanh(r \cdot c),\quad u = \sigma(u - 1) \\
h_t &= u \cdot c + (1 - u) \cdot h_{t-1}
\end{aligned}
$$
The $\sigma(u - 1)$ bias keeps the prior of $u$ near $\sigma(-1) \approx 0.27$ so the cell
defaults to retaining old state. The candidate gating happens **inside** the tanh:
$\tanh(r \cdot c)$, not $r \cdot \tanh(c)$. This is cascade fix #28.

**Two-hot encoding** (sheeprl `distribution.py:225-260`):
$$
\text{bins} = \mathrm{symexp}\bigl(\mathrm{linspace}(-20,\, 20,\, 255)\bigr),\qquad \mathrm{symlog}(x) = \mathrm{sign}(x) \log(|x|+1),\quad \mathrm{symexp}(x) = \mathrm{sign}(x) (e^{|x|} - 1)
$$
The target $y$ is `symlog`-mapped, then placed on the two adjacent bins with weights
summing to one. Log-prob is the cross-entropy of softmax-logits against this soft target.

**Hafner init** (sheeprl `utils.py:143-166`):
$$
\sigma = \frac{1}{0.87962566103423978} \cdot \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}},\qquad W \sim \mathcal{TN}(0,\, \sigma^2,\, [-2\sigma, +2\sigma])
$$
and `uniform_init_weights(s)`:
$$
W \sim \mathcal U(-L, L),\qquad L = \sqrt{\frac{3 s}{(n_{\text{in}} + n_{\text{out}})/2}}
$$
with $s = 0$ → $L = 0$ → zero-init (cascade fix #27 for reward + critic terminal Linears).

---

## §8. Recommendations for handoff

### Before code is written

1. **Fold Items 1–11 above into the IMPLEMENTATION_PLAN.md** — eleven small edits, mostly
   YAML additions, one missing training-step line (`is_first[0]=1`), one missing
   training-step line (action shift), one missing collection-step branch (random-action
   prefill before `learning_starts`). Estimated edit cost: ~30 minutes.
2. **Add a hyperparameter coverage table** to §File Changes that enumerates *which network
   reads which `mlp_layers` / `dense_units` value* (Item 8). Without it, the developer will
   guess.
3. **Resolve the `defaults_from:` YAML merge mechanism** (plan line 367) before
   Checkpoint 9. This is config-loader plumbing, not algorithm.

### Recommended additions to the Checkpoint list

4. **Add Checkpoint 4b (RSSM `is_first` reset)** — feed a sequence with
   `is_first = [1, 0, 0, ..., 0]` and a different sequence with `is_first = [0, 0, ..., 0]`;
   assert that on step 0 the RSSM recurrent state is exactly $\tanh(\bar h_0)$ in the first
   case and whatever the carry was in the second. This directly tests Item 1.
5. **Add Checkpoint 2b (action shift)** — assert that the actions fed to the RSSM are
   `concat(zeros_like(actions[:1]), actions[:-1])`, not raw `actions`. This directly tests
   Item 2.
6. **Add Checkpoint 9b (random-action prefill)** — assert that on policy steps `0..1023` the
   actions in the replay buffer have entropy `log(action_dim)` (uniform), and from step
   1024 onward the entropy starts to decrease. This directly tests Item 3.

### Next steps by agent

- **senior-developer**: incorporate Items 1–11 into IMPLEMENTATION_PLAN.md as plan edits.
  None are algorithm-altering; they are bit-identity-restoring. After the three reviewer
  memos return, the senior-developer is the natural appender of a Cross-Review Summary
  section at the top of the plan that links to all three reviewer files.
- **experiment-designer**: not engaged until the parity gate. Note that the plan's
  three-seed gate (mean survival ≥ 480 over 40-200k window, ≤ 25 h wall-clock per seed) is
  algorithmically defensible — it directly mirrors the `jzgkcep4` sheeprl drop-in benchmark.
- **developer**: implementation starts only after Items 1–11 are folded in. Otherwise the
  Checkpoint 8 forward-pass parity check will fail in a way the implementer has no map for.
- **code-reviewer** and **math-reviewer**: their reviews are running in parallel — read them
  alongside this one. This memo intentionally does **not** comment on JAX/Flax/NNX
  correctness (their lane) or equation-vs-paper faithfulness at the line-of-derivation level
  (also their lane).

---

## §9. What this review does NOT cover (out of scope per agent definition)

- JAX/NNX state-handling correctness (PRNG threading, `nnx.split`/`merge` at JIT
  boundaries, EMA target update mechanics in NNX, replay-buffer pytree semantics). →
  `code-reviewer`.
- Equation-vs-paper faithfulness for individual derivations (e.g., does the JAX symlog
  match the paper definition exactly, does the two-hot decoder match Eq. 9 to the last
  index). → `math-reviewer`.
- Per-paper backbone review of *why* DreamerV3 makes the choices it makes (this is the
  walkthrough's job, not this memo's). → `literature-reviewer` if a paper-level review is
  ever needed.
- Whether the parity gate (3 seeds, mean survival ≥ 480, ≤ 25 h wall-clock) is the
  scientifically right gate. The parity gate is by definition algorithm-bit-identity, not
  scientific gain — so it is the right framing for *this* plan, even though it is not
  itself a research result.

---

## §Verdict

**DEVIATIONS FOUND — 11 items above.**

None of the 11 items are algorithm-breaking on their own — the cascade items #2/27/28/29/30 are all
correctly placed, the RSSM math is right, the KL-balancing is right, the imagination loop is right,
the Polyak EMA is right, and the Hafner init is right. The five non-goals (no NMN, no continuous,
no MLflow, no memmap, no Hydra) are walled off cleanly with no algorithmic leakage. **No
arbitrary algorithmic additions found.**

But three of the items are **silent omissions** that the developer will not catch by reading the
plan alone:

1. **Item 1**: `is_first[0] = 1` force-set is missing from the training step. Sheeprl always
   resets the RSSM carry at the start of every sampled chunk. Without this, the dynamic-learning
   rollout starts from whatever the buffer happened to store at index 0 of the chunk, breaking
   bit-identity with sheeprl.
2. **Item 2**: action-shift `concat(zeros_like(actions[:1]), actions[:-1])` is missing from
   the training step. Without it the RSSM is fed actions one timestep ahead of the paper
   formulation.
3. **Item 3**: the `learning_starts=1024` random-action prefill phase is loaded from YAML but
   never wired into either the collection branch or the gradient-step gate. Without this the
   replay buffer starts with stale-policy trajectories, not uniform-random ones, biasing the
   world model's curriculum off the sheeprl reference.

The remaining items (4–11) are mostly hyperparameter / YAML omissions — they are bookkeeping
fixes that take an hour total to fold in. Items 5, 6, 7, 8 in particular ensure the developer
will not have to guess about MLP sizing, std parameters, distribution config keys, or the
discount head's `learnable` flag. Item 9 sharpens the λ-return signature contract; Item 10
adds the `Independent(...)` wrap; Item 11 documents the `Moments` `max_=1.0` floor behaviour.

After Items 1–11 are folded into the plan, **this is a clean replication blueprint at the
algorithm level**. The five-cascade naming, the math, the network architecture, the loss
structure, and the training loop are all correctly described — the gaps are around omissions
the implementer would otherwise default-fill in a way that breaks bit-identity. None of the
gaps require new research; all eleven are mechanical fixes traceable to specific sheeprl source
lines.

— Feedback from professor-rl-bayesian-dl — 2026-05-12
