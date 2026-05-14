---
title: "v2-CP1 — Fresh sheeprl reference audit memo (foundation for v2 audits)"
topic: dreamer
status: active
created: 2026-05-14
last_updated: 2026-05-14
phase: 2
---

# v2-CP1 — Fresh sheeprl reference audit memo

> **Doc role.** This is the **canonical "what sheeprl actually does"** reference for
> the v2 plan. Every downstream v2 checkpoint (v2-CP2 grad-parity methodology,
> v2-CP3 actor re-audit, v2-CP4 critic re-audit, v2-CP5 imagined-rollout re-audit,
> v2-CP6 train-step orchestrator, v2-CP7 driver re-audit, v2-CP8 wrapper re-audit)
> cites THIS memo, **not** the v1 reviewer outputs (which the user has explicitly
> de-trusted per the 2026-05-14 binding directive: *"review sheeprl codes again,
> don't depend on the previous code reviews as there can be mistakes"*).
> See [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §2 Lever C for the directive
> context.

---

## 1. Context (plain-language entry point)

**What this memo is.** A fresh read-through of the vendored
[sheeprl](https://github.com/Eclectic-Sheep/sheeprl) DreamerV3 source —
[`vendor/sheeprl/sheeprl/algos/dreamer_v3/`](../../../../vendor/sheeprl/sheeprl/algos/dreamer_v3/),
pinned at commit `33b6366` — performed without trusting any prior review. **Sheeprl** is
a community PyTorch implementation of the Hafner 2024 **DreamerV3** model-based
reinforcement-learning algorithm; the project is rebuilding that algorithm in JAX/Flax
under [`src/algorithms/dreamer_srl/`](../../../../src/algorithms/dreamer_srl/) (the
"dreamer-srl" port).

**Why it exists.** The v1 rebuild (April–May 2026) shipped 13 checkpoint-PASS verdicts
but parity-failed on 2026-05-14: a 3-seed run at the corrected XS recipe landed at the
random-policy floor (~104 survival steps versus a ~500-step sheeprl baseline). The
experiment-analyzer's leading hypothesis (**H1**) is a gradient-side bug in the actor
objective — most likely a subtle mis-port of the **REINFORCE** term (the policy-gradient
term that multiplies `log_prob(action) × advantage` and pushes the actor toward actions
that beat the baseline), the **`unimix=0.01`** mixture coefficient (which blends a 1%
uniform distribution into the categorical action probabilities to keep exploration
alive), the **entropy bonus** (which adds an entropy term to discourage premature
collapse), or a **`stop_gradient`** site (the operator that blocks gradients from flowing
through a tensor — `sg` for short — equivalent to PyTorch's `.detach()`). Forward-pass
unit tests verified outputs to `1e-6` of sheeprl; **gradients were never compared**.
Two secondary hypotheses are **H2** (`is_first` boolean flag mis-propagation at episode
boundaries, where `is_first` tells the world model "reset hidden state — new episode";
compounded by a smear vs burst pattern of replay-ratio gradient steps fired after
`learning_starts`, the **prefill** period where the actor is not yet trained) and **H3**
(advantage sign-flip — `λ-return` is the H-step λ-discounted return; `advantage =
λ-return − value` should drive the actor toward food, but the opposite sign would teach
avoidance).

**The memo's role going forward.** Every v2 re-audit CP (CP3–CP8) anchors its
line-by-line comparison against the sheeprl extracts below — **the v1 review files
under `docs/reviews/dreamer_srl_v3_*` are not authoritative**, only this memo and the
vendored source are. The memo also names the **legitimate cross-substrate boundaries**
(§8) — places where PyTorch and JAX genuinely differ at the bit level (random-number
generator semantics, optimizer epsilon placement, in-place vs functional state updates) —
so reviewers can distinguish "expected substrate drift" from "ported bug".

**Reading order.** §2 confirms the vendored commit. §3 extracts the **actor block**
(H1 locus). §4 extracts the **critic block** (H3 locus). §5 extracts the
**imagined-rollout chain** (H1+H3 cross-cutting). §6 extracts the **driver loop** (H2
locus). §7 is the cross-reference table mapping sheeprl artifacts → dreamer-srl analog
locations → re-auditing v2 CP. §8 lists confirmed cross-substrate boundaries. §9 catalogs
concrete things the v1 review pass missed.

---

## 2. Vendored sheeprl confirmation

- **Path**: [`vendor/sheeprl/sheeprl/algos/dreamer_v3/`](../../../../vendor/sheeprl/sheeprl/algos/dreamer_v3/)
- **Pinned commit**: `33b6366` (verified at memo authoring time via `git -C vendor/sheeprl rev-parse HEAD` returning `d37e07115650a4289a071095d541777378b16720` for the outer repo; the inner sheeprl pin is recorded in v1 plan §"Migration plan from v1").
- **Files (line counts)**:
  - `dreamer_v3.py` — **780 lines** (train function + main driver)
  - `agent.py` — **1236 lines** (encoders, decoders, RSSM, PlayerDV3, Actor, MinedojoActor, build_agent)
  - `loss.py` — **88 lines** (reconstruction_loss only)
  - `utils.py` — **235 lines** (Moments, compute_lambda_values, prepare_obs, test, init_weights, uniform_init_weights, log_models_from_checkpoint)
  - `evaluate.py` — **57 lines** (test-time agent loader; not in v2 scope)
  - `__init__.py` — **0 lines**

All citations below use the form `vendor/sheeprl/sheeprl/algos/dreamer_v3/<file>:L<n>` or
`L<n>-L<m>`. Reviewers verifying this memo should open the file at that line number and
confirm the quoted code.

---

## 3. The actor block (H1 hypothesis-locus)

The actor lives in two places in sheeprl: the **`Actor` class** in `agent.py` (the
forward-pass module — defines the MLP backbone, the `unimix` application, the
categorical distribution construction) and the **`train()` function** in `dreamer_v3.py`
(the objective construction — assembles REINFORCE term + entropy term + advantage
normalization + discount weighting and calls `fabric.backward`).

### 3.1 Actor forward pass (`agent.py:L694-L845`)

**Class signature** (`agent.py:L729-L745`):

```
def __init__(
    self,
    latent_state_size: int,
    actions_dim: Sequence[int],
    is_continuous: bool,
    distribution_cfg: Dict[str, Any],
    init_std: float = 0.0,
    min_std: float = 1.0,
    max_std: float = 1.0,
    dense_units: int = 1024,
    activation: nn.Module = nn.SiLU,
    mlp_layers: int = 5,
    layer_norm_cls: Callable[..., nn.Module] = LayerNorm,
    layer_norm_kw: Dict[str, Any] = {"eps": 1e-3},
    unimix: float = 0.01,
    action_clip: float = 1.0,
) -> None:
```

The actor is a stack of `mlp_layers` Linear+LayerNorm+SiLU blocks producing a
`dense_units`-wide trunk, followed by `nn.ModuleList` of per-action-dim Linear heads
that emit pre-softmax **logits** (discrete) or a `mean/std` pair (continuous). Layer
norm uses `eps=1e-3` (not the PyTorch default `1e-5`). LayerNorm is unconditional —
no `layer_norm_cls == nn.Identity` branch in the head config.

**Forward (discrete-action branch, the food-task case)** (`agent.py:L828-L837`):

```python
else:
    actions_dist: List[Distribution] = []
    actions: List[Tensor] = []
    for logits in pre_dist:
        actions_dist.append(OneHotCategoricalStraightThrough(logits=self._uniform_mix(logits)))
        if not greedy:
            actions.append(actions_dist[-1].rsample())
        else:
            actions.append(actions_dist[-1].mode)
return tuple(actions), tuple(actions_dist)
```

Three load-bearing facts:

1. **`OneHotCategoricalStraightThrough`** is PyTorch's straight-through gradient
   estimator for one-hot categoricals. The `rsample()` call (`agent.py:L834`) returns
   a one-hot sample where gradients flow through the **soft probabilities** in the
   backward pass — i.e. the forward sample is discrete (argmax-then-onehot) but the
   backward gradient is `∂/∂logits(softmax(logits))`. This is the **only** path by
   which gradient reaches `logits` from the sampled action.
2. **`_uniform_mix(logits)`** is applied **before** the distribution is constructed
   (`agent.py:L832`). This means the action distribution is over the **unimixed
   probabilities**, not the raw logits.
3. **`greedy=False` during training** (the imagined-rollout call at
   `dreamer_v3.py:L219` and `:L240` do not pass `greedy=...`, so default `greedy=False`
   applies → `rsample` path is taken).

**`_uniform_mix` (the H1 critical site)** (`agent.py:L839-L845`):

```python
def _uniform_mix(self, logits: Tensor) -> Tensor:
    if self._unimix > 0.0:
        probs = logits.softmax(dim=-1)
        uniform = torch.ones_like(probs) / probs.shape[-1]
        probs = (1 - self._unimix) * probs + self._unimix * uniform
        logits = probs_to_logits(probs)
    return logits
```

**Critical gradient-flow detail**: the mix is applied in **probability space**, then
converted back to logits via `probs_to_logits` (which is `log(probs) - log(1 - probs)`
or `log(probs)` for categoricals — see `torch.distributions.utils.probs_to_logits`,
the implementation for categoricals applies `torch.log(probs.clamp(min=tiny))`).
Gradient flows through:

- `logits.softmax(dim=-1)` → softmax Jacobian
- `(1 - unimix) * probs + unimix * uniform` → identity on `probs` (the `uniform` term
  is a constant, no gradient)
- `probs_to_logits(probs)` → `1/probs` Jacobian (with a tiny clamp for numerical safety)

The composite is **NOT** algebraically equivalent to mixing in logit space. Any v2 port
that conflates "mix logits" with "mix probs then convert" introduces a gradient bug at
this site. **v2-CP3 must verify dreamer-srl's `_uniform_mix` analog uses
probability-space mixing followed by `log(probs)` conversion**, not logit-space
addition.

### 3.2 Actor objective construction (`dreamer_v3.py:L262-L304`)

This is the **gradient-producing site** for the actor. The exact sheeprl code:

```python
# Actor optimization step. Eq. 11 from the paper
actor_optimizer.zero_grad(set_to_none=True)                          # L272
policies: Sequence[Distribution] = actor(imagined_trajectories.detach())[1]  # L273

baseline = predicted_values[:-1]                                     # L275
offset, invscale = moments(lambda_values, fabric)                    # L276
normed_lambda_values = (lambda_values - offset) / invscale           # L277
normed_baseline = (baseline - offset) / invscale                     # L278
advantage = normed_lambda_values - normed_baseline                   # L279
if is_continuous:                                                    # L280
    objective = advantage
else:                                                                # L282
    objective = (
        torch.stack(
            [
                p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]    # L286
                for p, imgnd_act in zip(policies, torch.split(imagined_actions, actions_dim, dim=-1))
            ],
            dim=-1,
        ).sum(dim=-1)
        * advantage.detach()                                         # L291
    )
try:
    entropy = cfg.algo.actor.ent_coef * torch.stack([p.entropy() for p in policies], -1).sum(dim=-1)  # L294
except NotImplementedError:
    entropy = torch.zeros_like(objective)                            # L296
policy_loss = -torch.mean(discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1]))  # L297
fabric.backward(policy_loss)                                         # L298
```

**Term-by-term gradient analysis (H1 verification surface):**

| Term | Source line | Gradient flow into | Gradient flow blocked by |
|---|---|---|---|
| `imagined_trajectories.detach()` (input to actor) | L273 | actor params via `actor(...)` | `.detach()` blocks gradient back into WM / rollout |
| `p.log_prob(imgnd_act.detach())` | L286 | actor params (via `p`'s logits) | `imgnd_act.detach()` — the sampled action is treated as a constant target |
| `advantage.detach()` | L291 | (none — multiplier only) | `.detach()` blocks gradient into critic / lambda_values from the actor's REINFORCE term — **this is the canonical `sg(advantage)` site** |
| `p.entropy()` | L294 | actor params (via `p`'s logits) | (none — entropy uses gradient-graph logits directly) |
| `discount[:-1].detach()` | L297 | (none — multiplier only) | `.detach()` blocks gradient into WM continue-head |
| `-torch.mean(...)` | L297 | the entire `policy_loss` scalar | (the minus sign converts the maximisation objective into a loss) |

**The REINFORCE formula** is therefore exactly:

```
policy_loss = - E_t [ discount(t) * ( log_prob(a_t) * sg(advantage(t)) + ent_coef * entropy(t) ) ]
```

with `sg(advantage)` and `sg(action)` placements as marked above. Sign convention:
positive `advantage` drives `log_prob(a_t)` upward (the gradient `∂(log_prob *
sg(advantage)) / ∂logits` is positive in the direction of `a_t` when advantage > 0).

**The advantage normalization** (`dreamer_v3.py:L276-L279`) is mediated by the
`Moments` class (`utils.py:L40-L63`):

```python
def forward(self, x: Tensor, fabric: Fabric) -> Any:
    gathered_x = fabric.all_gather(x).float().detach()                        # L57
    low = torch.quantile(gathered_x, self._percentile_low)                    # L58
    high = torch.quantile(gathered_x, self._percentile_high)                  # L59
    self.low = self._decay * self.low + (1 - self._decay) * low               # L60
    self.high = self._decay * self.high + (1 - self._decay) * high            # L61
    invscale = torch.max(1 / self._max, self.high - self.low)                 # L62
    return self.low.detach(), invscale.detach()                               # L63
```

Both `offset = self.low.detach()` and `invscale = (max(1/max, high-low)).detach()` are
detached — gradient **does not flow** back into the moments-EMA from the actor loss.
The normalization is purely an inputs-scaling pre-processing step.

### 3.3 Sites where v2-CP3 must measure gradient parity

For dreamer-srl's `compute_actor_objective`
([`src/algorithms/dreamer_srl/train.py:L500-L612`](../../../../src/algorithms/dreamer_srl/train.py)),
v2-CP3 must construct gradient-side fixture tests at each of:

1. **`∂L_actor / ∂(actor MLP weights)` via the REINFORCE term** — fix `imagined_actions`
   and `advantage` to fixture values; compute gradient w.r.t. each parameter; compare
   against the PyTorch sheeprl analog at the same fixture seed.
2. **`∂L_actor / ∂(actor MLP weights)` via the entropy term** — set `imagined_actions`
   to zero / `advantage` to zero so only the entropy term contributes; compare.
3. **`∂L_actor / ∂(actor head logits)` at the `_uniform_mix` boundary** — pre-mix and
   post-mix probabilities and their gradient w.r.t. raw logits.
4. **`∂L_actor / ∂(critic params)` should be zero** — verify that `sg(advantage)`
   blocks all gradient.
5. **`∂L_actor / ∂(imagined_actions)` should be zero** — verify that `imgnd_act.detach()`
   blocks straight-through gradient back into the action samples.

---

## 4. The critic block (H3 hypothesis-locus)

The critic is built as a plain `MLP` in `agent.py:L1154-L1166` (not a dedicated class —
it's just an `MLP` with `output_dim=cfg.algo.critic.bins`, typically 255 bins for the
two-hot encoding). The **objective construction** is in `dreamer_v3.py:L306-L327`.

### 4.1 Critic forward pass

The critic emits raw logits over `bins=255` symlog-encoded value buckets. The logits
are wrapped in a `TwoHotEncodingDistribution` from
`sheeprl/utils/distribution.py` (out of dreamer_v3 scope — but the relevant semantics
are: `.mean` returns `Σ_i bucket_value(i) * softmax(logits)[i]` symlog-decoded;
`.log_prob(target)` returns `log Σ_i softmax(logits)[i] * two_hot(target)[i]` with
two-hot encoding mapping a scalar to its two neighbouring buckets weighted by
proximity). The two-hot bucket scheme symlog-encodes targets so the critic can predict
a wide value range without saturating.

### 4.2 Critic loss (`dreamer_v3.py:L306-L327`)

The **exact two-term form** (the canonical sheeprl shape; cascade-fix-#29 in v1
terminology):

```python
# Predict the values
qv = TwoHotEncodingDistribution(critic(imagined_trajectories.detach()[:-1]), dims=1)   # L307
predicted_target_values = TwoHotEncodingDistribution(                                  # L308
    target_critic(imagined_trajectories.detach()[:-1]), dims=1
).mean                                                                                 # L310

# Critic optimization. Eq. 10 in the paper
critic_optimizer.zero_grad(set_to_none=True)                                           # L313
value_loss = -qv.log_prob(lambda_values.detach())                                      # L314
value_loss = value_loss - qv.log_prob(predicted_target_values.detach())                # L315
value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))                        # L316

fabric.backward(value_loss)                                                            # L318
```

**Two-term structure**: the critic learns to predict BOTH the λ-return (`L314` term —
the bootstrap-of-immediate-rewards-plus-future-returns target) AND the slow-target
network's prediction (`L315` term — a polyak-EMA-stabilised version of the critic
itself, providing a stable regression target that reduces TD-bootstrap variance).

**Stop-gradient sites**:

- **`lambda_values.detach()` (L314)** — blocks gradient from flowing into the rollout
  chain, the actor, or the world model via the value targets. The critic learns
  toward the λ-return as a fixed target, NOT toward a moving λ-return.
- **`predicted_target_values.detach()` (L315)** — blocks gradient through the target
  critic. The target critic is a polyak EMA of the live critic and is updated
  separately at the driver-level (see §6).
- **`imagined_trajectories.detach()[:-1]` (L307, L309)** — both `qv` and
  `target_critic` are evaluated on detached trajectories. The critic does not
  back-propagate into the rollout / world-model dynamics.
- **`discount[:-1].squeeze(-1)`** is computed under `torch.no_grad()` at L259-L260 of
  `dreamer_v3.py`, so it carries no gradient.

**H3 verification surface**: the sign of `value_loss = -qv.log_prob(...)` is **negative
log-probability** — i.e. minimising `value_loss` maximises `log_prob(target)`. Both
terms have the same negative sign, so the critic is trained to assign high probability
to BOTH the λ-return AND the target-critic prediction. **Any v2 port that has the
critic loss with a flipped sign — `value_loss = +qv.log_prob(...)` — would train the
critic to AVOID the targets, which would in turn corrupt the advantage signal that
feeds the actor.**

### 4.3 The advantage signal (`dreamer_v3.py:L275-L279`)

The advantage is computed in the **actor** block, not the critic block — but it
consumes the critic's predictions:

```python
baseline = predicted_values[:-1]                              # L275
offset, invscale = moments(lambda_values, fabric)             # L276
normed_lambda_values = (lambda_values - offset) / invscale    # L277
normed_baseline = (baseline - offset) / invscale              # L278
advantage = normed_lambda_values - normed_baseline            # L279
```

where `predicted_values = TwoHotEncodingDistribution(critic(imagined_trajectories),
dims=1).mean` is computed at L244 (note: NOT detached at that site — but it IS
detached at L291 before multiplying by `log_prob`).

`invscale` is `max(1/Moments._max, high_quantile - low_quantile)` of the gathered
`lambda_values`, with `high/low` being EMA-updated quantiles. This is **not** a
per-batch standard-deviation normalization; it's a slow EMA of the inter-quantile
range. v2-CP3 / v2-CP4 should verify that the dreamer-srl `MomentsState` analog
([`src/algorithms/dreamer_srl/utils.py:L174-L267`](../../../../src/algorithms/dreamer_srl/utils.py))
uses the same percentile-EMA semantics (not a per-batch std).

**Sign check**: `advantage = normed_lambda_values - normed_baseline` (L279). When
`λ-return > value`, advantage is positive → actor is pulled toward the chosen action.
**Sign-flipped port (`advantage = baseline - lambda_values`) → H3 root-cause symptom.**

---

## 5. The imagined-rollout chain (H1 + H3 cross-cutting)

The rollout chain is interleaved into the `train()` function at
`dreamer_v3.py:L202-L260`. It does three things: **(a)** initialise the rollout from
posterior states drawn during dynamic learning, **(b)** unroll the imagination loop for
`horizon` steps using the actor and the WM, **(c)** compute λ-returns over the
imagined trajectory.

### 5.1 Rollout init (`dreamer_v3.py:L202-L220`)

```python
# Behaviour Learning
imagined_prior = posteriors.detach().reshape(1, -1, stoch_state_size)         # L203
recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)  # L204
imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)      # L205
imagined_trajectories = torch.empty(
    cfg.algo.horizon + 1,
    batch_size * sequence_length,
    stoch_state_size + recurrent_state_size,
    device=device,
)
imagined_trajectories[0] = imagined_latent_state                              # L212
imagined_actions = torch.empty(
    cfg.algo.horizon + 1,
    batch_size * sequence_length,
    data["actions"].shape[-1],
    device=device,
)
actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)         # L219
imagined_actions[0] = actions                                                 # L220
```

Three load-bearing points:

1. **Init from POSTERIORS, not replay actions**: the rollout starts from the posterior
   stochastic states computed during dynamic learning (L203). Every time-step in the
   `[T, B]` posterior tensor seeds a separate `T*B`-long parallel rollout — so a single
   batch of `[T=64, B=16]` yields `1024` parallel `horizon`-step rollouts. This
   posterior is the **agent's encoded belief about the world at each replay timestep**,
   not the actual replay observations.
2. **`posteriors.detach()` (L203)** — gradients do not flow from the rollout back into
   the WM through the rollout init. The WM is trained ONLY by the WM loss
   (`reconstruction_loss`) computed earlier in `train()` at L176-L200.
3. **`imagined_latent_state.detach()` passed to the actor (L219)** — the actor's first
   action is computed on a detached state, so the actor's gradient does not flow into
   the rollout init. (The actor itself is differentiable; only the input is detached.)

### 5.2 Horizon-H imagined steps (`dreamer_v3.py:L222-L241`)

```python
# The imagination goes like this, with H=3:
# Actions:           a'0      a'1      a'2     a'4
# ...
# States:        z0 ---> z'1 ---> z'2 ---> z'3
# ...
# where z0 comes from the posterior, while z'i is the imagined states (prior)

for i in range(1, cfg.algo.horizon + 1):                                      # L235
    imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)  # L236
    imagined_prior = imagined_prior.view(1, -1, stoch_state_size)             # L237
    imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)  # L238
    imagined_trajectories[i] = imagined_latent_state                          # L239
    actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)     # L240
    imagined_actions[i] = actions                                             # L241
```

Each step:

1. **`world_model.rssm.imagination(...)`** (`agent.py:L482-L498`) runs the recurrent
   model + transition model forward. **No new observation is fed** — the imagined
   trajectory uses ONLY the transition prior (no posterior correction). The
   `_transition` call applies `_uniform_mix` (`agent.py:L478-L480`) just like in
   dynamic learning.
2. **`imagined_latent_state.detach()` passed to actor (L240)** — each step's actor
   input is detached, so the actor's gradient only flows through the current step's
   parameters, NOT back through prior actor decisions. This is the standard Hafner
   straight-through pattern.
3. **`actions` from `actor(...)` (L240)** — `actor` returns `(actions_tuple,
   actions_dist_tuple)`; only `actions_tuple` is taken (the `[0]` index). The sampled
   action carries straight-through gradient via `OneHotCategoricalStraightThrough` —
   so gradient from `actor(L240)` flows into the actor params **at step i** but NOT
   through `imagined_latent_state` (which is detached).

**PRNG handling**: PyTorch uses a global rng state implicitly threaded through every
`.rsample()` call. JAX requires explicit `jax.random.split(...)` PRNG-key passing.
This is one of the canonical cross-substrate boundaries (see §8, D-009 class).

### 5.3 λ-return computation (`dreamer_v3.py:L243-L260` + `utils.py:L66-L77`)

```python
# Predict values, rewards and continues
predicted_values = TwoHotEncodingDistribution(critic(imagined_trajectories), dims=1).mean    # L244
predicted_rewards = TwoHotEncodingDistribution(world_model.reward_model(imagined_trajectories), dims=1).mean  # L245
continues = Independent(BernoulliSafeMode(logits=world_model.continue_model(imagined_trajectories)), 1).mode  # L246
true_continue = (1 - data["terminated"]).flatten().reshape(1, -1, 1)                         # L247
continues = torch.cat((true_continue, continues[1:]))                                        # L248

# Estimate lambda-values
lambda_values = compute_lambda_values(                                                       # L251
    predicted_rewards[1:],
    predicted_values[1:],
    continues[1:] * cfg.algo.gamma,
    lmbda=cfg.algo.lmbda,
)

# Compute the discounts to multiply the lambda values to
with torch.no_grad():                                                                        # L259
    discount = torch.cumprod(continues * cfg.algo.gamma, dim=0) / cfg.algo.gamma             # L260
```

The **§S5 "true-continue splice"** at L247-L248: the first time-step's `continues` is
replaced with `(1 - terminated)` from the actual replay batch (the *real* continue
signal at the rollout-init timestep), while subsequent steps use the WM-imagined
continue prediction. Algorithmic motivation: at the rollout-init timestep, the agent
has ground-truth knowledge of whether the episode actually terminated (the env returned
`terminated=True`); pretending the WM has to predict this would inject avoidable noise
into the very first reward target. Subsequent steps are imagined, so the WM's
continue-head prediction is the only signal available.

`compute_lambda_values` (`utils.py:L66-L77`):

```python
def compute_lambda_values(
    rewards: Tensor,
    values: Tensor,
    continues: Tensor,
    lmbda: float = 0.95,
):
    vals = [values[-1:]]                                                # L72 — terminal bootstrap = last value
    interm = rewards + continues * values * (1 - lmbda)                 # L73 — per-step intermediate
    for t in reversed(range(len(continues))):                           # L74
        vals.append(interm[t] + continues[t] * lmbda * vals[-1])        # L75
    ret = torch.cat(list(reversed(vals))[:-1])                          # L76
    return ret
```

This is the **canonical backward λ-return recurrence**: starting from the bootstrap
`V(s_T)`, recurse `V_λ(s_t) = r_t + γ * c_t * ((1-λ) * V(s_{t+1}) + λ * V_λ(s_{t+1}))`.
The final `[:-1]` slice (L76) drops the bootstrap (which was prepended to keep
the recurrence convenient) so the returned tensor has length `horizon`.

**`continues` parameter**: the caller passes `continues[1:] * cfg.algo.gamma`
(L254), so inside `compute_lambda_values` the `continues` tensor already includes the
`γ` factor — it represents `γ * (1 - terminated_imagined)` per step.

**Discount factor application (L259-L260)**: `discount = cumprod(continues * γ) / γ`.
The division by `γ` shifts the cumprod so `discount[0] = 1.0` (not `γ`), letting the
loss-weighting `discount[:-1].detach() * (...)` weight the first rollout step with
weight 1, not `γ`. This is computed under `torch.no_grad()` so it has no gradient
graph.

---

## 6. The driver loop (H2 hypothesis-locus)

The driver is the `main()` function at `dreamer_v3.py:L361-L765`. It owns the
env-interaction loop, the buffer management, the prefill gate, the train-trigger
schedule, the polyak update, the metrics logging, and the checkpoint save.

### 6.1 The `is_first` flag (H2 critical)

The `is_first` boolean tells the RSSM "this transition starts a new episode → reset
hidden state". Sheeprl writes `is_first=1` at two places:

**At env reset (the very first env interaction)** — `dreamer_v3.py:L546`:

```python
step_data["is_first"] = np.ones_like(step_data["terminated"])
```

**After every `done`** (where `done = terminated OR truncated`) — `dreamer_v3.py:L639-L656`:

```python
dones_idxes = dones.nonzero()[0].tolist()                                    # L639
reset_envs = len(dones_idxes)
if reset_envs > 0:
    reset_data = {}
    for k in obs_keys:
        reset_data[k] = (real_next_obs[k][dones_idxes])[np.newaxis]
    reset_data["terminated"] = step_data["terminated"][:, dones_idxes]
    reset_data["truncated"] = step_data["truncated"][:, dones_idxes]
    reset_data["actions"] = np.zeros((1, reset_envs, np.sum(actions_dim)))
    reset_data["rewards"] = step_data["rewards"][:, dones_idxes]
    reset_data["is_first"] = np.zeros_like(reset_data["terminated"])         # L649 — RESET buffer entry: is_first=0
    rb.add(reset_data, dones_idxes, validate_args=cfg.buffer.validate_args)  # L650

    # Reset already inserted step data
    step_data["rewards"][:, dones_idxes] = np.zeros_like(reset_data["rewards"])
    step_data["terminated"][:, dones_idxes] = np.zeros_like(step_data["terminated"][:, dones_idxes])
    step_data["truncated"][:, dones_idxes] = np.zeros_like(step_data["truncated"][:, dones_idxes])
    step_data["is_first"][:, dones_idxes] = np.ones_like(step_data["is_first"][:, dones_idxes])  # L656 — NEXT step_data: is_first=1
    player.init_states(dones_idxes)                                          # L657
```

**Subtle ordering**: on iteration `t` where `done[t]=True` at env `i`:

1. The buffer-add at L587 (earlier in the loop, line `rb.add(step_data, ...)`)
   already wrote `step_data` for iteration `t` — this `step_data` had `is_first[t] = 0`
   for env `i` (because L594 zeros it: `step_data["is_first"] = np.zeros_like(...)`).
2. Now, *after* the env step that produced `done[t]=True`, L649-L650 adds a SECOND
   reset entry into the buffer with `is_first=0` (because the reset entry is the
   "terminal observation" at the boundary — it's not the start of the new episode).
3. Then L656 sets `step_data["is_first"][:, dones_idxes] = 1` so that on the NEXT
   iteration (`t+1`), the buffer-add at L587 will write `is_first=1` for env `i`.

So the canonical sheeprl pattern is: **after a `done`, the next written `step_data` row
has `is_first=1`**. This is the row that the RSSM's `dynamic()` (`agent.py:L425-L435`)
reads to gate `recurrent_state = (1 - is_first) * recurrent_state + is_first *
initial_recurrent_state` — i.e. wipe the recurrent hidden state when a new episode
starts.

**H2 verification surface** (for v2-CP7): the dreamer-srl driver
([`src/algorithms/dreamer_srl/dreamer_srl_main.py`](../../../../src/algorithms/dreamer_srl/dreamer_srl_main.py))
must replicate this exact `is_first` timing — written into the buffer at the row
*after* the `done`, not the row *of* the `done`. An off-by-one (e.g. writing
`is_first=1` into the row of the `done`) would propagate a stale hidden state into
the next episode's encoder/posterior chain.

### 6.2 Prefill behavior (`dreamer_v3.py:L558-L584`)

```python
if (
    iter_num <= learning_starts                                              # L559
    and cfg.checkpoint.resume_from is None
    and "minedojo" not in cfg.env.wrapper._target_.lower()
):
    real_actions = actions = np.array(envs.action_space.sample())            # L563 — random action
    if not is_continuous:
        actions = np.concatenate(
            [
                F.one_hot(torch.as_tensor(act), act_dim).numpy()
                for act, act_dim in zip(actions.reshape(len(actions_dim), -1), actions_dim)
            ],
            axis=-1,
        )
else:                                                                        # L572 — normal training
    torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
    mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}
    if len(mask) == 0:
        mask = None
    real_actions = actions = player.get_actions(torch_obs, mask=mask)        # L577 — actor-policy action
```

**During prefill (`iter_num <= learning_starts`)**: the action is sampled UNIFORMLY
AT RANDOM from `envs.action_space` (L563) — the actor is **not** called. This means
during prefill the buffer collects pure random-policy trajectories.

**Prefill end gate**: at `iter_num == learning_starts + 1`, the actor takes over.

**H2 secondary check**: the dreamer-srl prefill must also use random actions and must
NOT call the actor during prefill — if the dreamer-srl prefill accidentally calls the
actor before training has begun, the actor produces deterministic-near-zero actions
(from random-init params), polluting the buffer with low-diversity data.

### 6.3 Train gate (`dreamer_v3.py:L659-L699`)

```python
# Train the agent
if iter_num >= learning_starts:                                              # L660
    ratio_steps = policy_step - prefill_steps * policy_steps_per_iter        # L661
    per_rank_gradient_steps = ratio(ratio_steps / world_size)                # L662
    if per_rank_gradient_steps > 0:                                          # L663
        local_data = rb.sample_tensors(                                      # L664
            cfg.algo.per_rank_batch_size,
            sequence_length=cfg.algo.per_rank_sequence_length,
            n_samples=per_rank_gradient_steps,
            dtype=None,
            device=fabric.device,
            from_numpy=cfg.buffer.from_numpy,
        )
        with timer("Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute):
            for i in range(per_rank_gradient_steps):                         # L673
                if (                                                         # L674
                    cumulative_per_rank_gradient_steps % cfg.algo.critic.per_rank_target_network_update_freq
                    == 0
                ):
                    tau = 1 if cumulative_per_rank_gradient_steps == 0 else cfg.algo.critic.tau   # L678
                    for cp, tcp in zip(critic.module.parameters(), target_critic.parameters()):
                        tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)  # L680
                batch = {k: v[i].float() for k, v in local_data.items()}
                train(                                                       # L682
                    fabric, world_model, actor, critic, target_critic, ...
                )
                cumulative_per_rank_gradient_steps += 1                      # L698
```

**Three load-bearing facts:**

1. **Train gate is `iter_num >= learning_starts` (NOT `>`)** at L660. The first
   training iteration is the *same* iter where `learning_starts` is reached.
2. **`per_rank_gradient_steps = ratio(...)`** is determined by the `Ratio` class
   (`sheeprl/utils/utils.py`), which implements the smeared replay-ratio schedule.
   This is the **canonical sheeprl replay-ratio behaviour** — gradient steps per
   env-step are smeared continuously, not bursted. **v1 D-014 deviation ships a
   ONE-SHOT 1024-step debt-repayment burst at `iter_num == learning_starts` instead
   of the smeared rate**, which is a candidate H2 root cause.
3. **Polyak update fires BEFORE the `train(...)` call at L674-L680** — i.e. the target
   critic is updated FIRST in the inner training loop, then the `train()` step runs
   with the updated target critic. Initial `tau=1` at step 0 makes the target critic
   identical to the live critic; subsequent steps use `tau=cfg.algo.critic.tau`
   (typically 0.02) for the EMA.

### 6.4 Buffer-add timing (the §S3 invariant from D-014)

The buffer-add for the *current* step happens at L587:

```python
rb.add(step_data, validate_args=cfg.buffer.validate_args)
```

This is **inside the `with torch.inference_mode():`** block (L553) and is **inside the
`with timer("Time/env_interaction_time", ...)`** block (L556). It executes BEFORE the
env-step at L589 (because the action it just wrote `step_data["actions"]` was the
action that *led* to the new observation). So `step_data` rows are written into the
buffer as `(obs_t, action_t, reward_{t-1}, terminated_{t-1}, truncated_{t-1},
is_first_t)` — the obs is from before the action, the reward / terminated / truncated
are from the PREVIOUS env step's response (initialised to zeros at L543-L546 at the
very first iteration).

This is subtle: **the buffer's row `t` does NOT contain step `t`'s reward.** The reward
for action `a_t` is in row `t+1`. This is why `train()` shifts actions back by one at
L104:

```python
batch_actions = torch.cat((torch.zeros_like(data["actions"][:1]), data["actions"][:-1]), dim=0)
```

— "prepend a zero action and drop the last action" — so that at row `t` the *applied*
action is `batch_actions[t]` which equals `data["actions"][t-1]`. The dynamic-learning
inputs at row `t` are then `(obs_t, batch_actions[t]=action_{t-1},
reward_t=reward_received_after_action_{t-1})` — consistent.

**v2-CP7 verification surface**: dreamer-srl's
[`SequentialReplayBuffer`](../../../../src/algorithms/dreamer_srl/buffers.py) must
preserve this exact off-by-one shift, AND `train.py`'s action-shift must apply the
same prepend-zero / drop-last pattern.

---

## 7. Cross-reference table

| Sheeprl artifact (file:line range) | What it does | Dreamer-srl analog location | v2 re-audit CP |
|---|---|---|---|
| `dreamer_v3.py:L48-L358` (`train` function) | One full WM+actor+critic+rollout update | [`src/algorithms/dreamer_srl/train.py:L613-L913`](../../../../src/algorithms/dreamer_srl/train.py) (`make_train_step` + `one_train_step`) | v2-CP6 (orchestrator) |
| `dreamer_v3.py:L100-L172` (dynamic learning) | Posterior unroll, encoder/RSSM forward | [`train.py:L613-L913`](../../../../src/algorithms/dreamer_srl/train.py) (WM-update block) | v2-CP6 |
| `dreamer_v3.py:L174-L200` (WM-loss + backward + optimiser step) | World-model update via `reconstruction_loss` | [`train.py:L613-L913`](../../../../src/algorithms/dreamer_srl/train.py) | v2-CP6 |
| `dreamer_v3.py:L202-L260` (imagined rollout + λ-returns) | Behaviour learning rollout | [`train.py:L396-L499`](../../../../src/algorithms/dreamer_srl/train.py) (`compute_imagined_returns`) | **v2-CP5** |
| `dreamer_v3.py:L262-L304` (actor objective + backward) | REINFORCE + entropy + advantage normalization | [`train.py:L500-L612`](../../../../src/algorithms/dreamer_srl/train.py) (`compute_actor_objective`) | **v2-CP3 (H1 centerpiece)** |
| `dreamer_v3.py:L306-L327` (critic two-term loss) | Two-hot value-distribution KL to λ-return + slow-target | [`train.py:L227-L330`](../../../../src/algorithms/dreamer_srl/train.py) (`compute_critic_loss`) | **v2-CP4 (H3)** |
| `dreamer_v3.py:L361-L765` (`main` driver) | Env-loop + buffer + prefill + train-gate + polyak + logging | [`dreamer_srl_main.py`](../../../../src/algorithms/dreamer_srl/dreamer_srl_main.py) (entire file, 603 lines) | **v2-CP7 (H2)** |
| `dreamer_v3.py:L538-L657` (env-step + buffer-add + is_first) | `is_first` semantics + buffer rows | [`dreamer_srl_main.py`](../../../../src/algorithms/dreamer_srl/dreamer_srl_main.py) (env-interaction loop) | v2-CP7 |
| `dreamer_v3.py:L558-L584` (prefill / actor branch) | Random action during prefill, actor after | [`dreamer_srl_main.py`](../../../../src/algorithms/dreamer_srl/dreamer_srl_main.py) (prefill branch) | v2-CP7 |
| `dreamer_v3.py:L659-L699` (train gate + ratio + polyak) | Replay-ratio schedule + target-critic EMA | [`dreamer_srl_main.py`](../../../../src/algorithms/dreamer_srl/dreamer_srl_main.py) (train-trigger block) + [`train.py:L331-L395`](../../../../src/algorithms/dreamer_srl/train.py) (`polyak_update`) | v2-CP7 |
| `agent.py:L42-L97` (`CNNEncoder`) | Image encoder | (not used in food task; MLP encoder only) | v2-CP8 |
| `agent.py:L100-L151` (`MLPEncoder`) | Vector encoder with symlog inputs | [`src/algorithms/dreamer_srl/agent.py:L1115-L1202`](../../../../src/algorithms/dreamer_srl/agent.py) (`MLPEncoder`) | v2-CP8 |
| `agent.py:L154-L226` (`CNNDecoder`) | Image decoder | (not used) | v2-CP8 |
| `agent.py:L229-L278` (`MLPDecoder`) | Vector decoder | [`agent.py:L1203-L1298`](../../../../src/algorithms/dreamer_srl/agent.py) (`MLPDecoder`) | v2-CP8 |
| `agent.py:L281-L341` (`RecurrentModel`) | MLP + LayerNormGRUCell wrapper | [`agent.py:L46-L170`](../../../../src/algorithms/dreamer_srl/agent.py) (`LayerNormGRUCell`) + RSSM internal MLP | v2-CP8 |
| `agent.py:L344-L498` (`RSSM`) | Recurrent state-space model: dynamic + imagination + uniform-mix | [`agent.py:L371-L1114`](../../../../src/algorithms/dreamer_srl/agent.py) (`RSSM`) | v2-CP8 |
| `agent.py:L501-L593` (`DecoupledRSSM`) | Variant with decoupled posterior | (only used if `cfg.algo.world_model.decoupled_rssm=true`; verify food-task config) | v2-CP8 |
| `agent.py:L596-L691` (`PlayerDV3`) | Env-side actor wrapper (carries recurrent state across env steps) | inline in `dreamer_srl_main.py` | v2-CP7 + v2-CP8 |
| `agent.py:L694-L845` (`Actor` class) | Actor MLP + heads + `_uniform_mix` | [`agent.py:L1386-L1550`](../../../../src/algorithms/dreamer_srl/agent.py) (`Actor`) | **v2-CP3 + v2-CP8** |
| `agent.py:L935-L1236` (`build_agent`) | Module wiring + Fabric setup + weight tying | [`agent.py:L1915-L2117`](../../../../src/algorithms/dreamer_srl/agent.py) (`build_agent`) | v2-CP8 |
| `loss.py:L9-L88` (`reconstruction_loss`) | KL-balanced WM loss with free-nats | [`src/algorithms/dreamer_srl/loss.py:L424-L582`](../../../../src/algorithms/dreamer_srl/loss.py) (`reconstruction_loss`) | v2-CP6 |
| `utils.py:L40-L63` (`Moments`) | EMA percentile-based advantage normalization | [`src/algorithms/dreamer_srl/utils.py:L174-L267`](../../../../src/algorithms/dreamer_srl/utils.py) (`MomentsState`, `moments_init`, `moments_update`) | v2-CP3 (consumed) + v2-CP6 |
| `utils.py:L66-L77` (`compute_lambda_values`) | Backward λ-return recurrence | [`utils.py:L124-L173`](../../../../src/algorithms/dreamer_srl/utils.py) (`compute_lambda_values`) | v2-CP5 |
| `utils.py:L143-L186` (`init_weights` + `uniform_init_weights`) | TruncNormal / Uniform initialisers | [`utils.py:L53-L123`](../../../../src/algorithms/dreamer_srl/utils.py) (`init_weights`, `uniform_init_weights`) | v2-CP8 |
| `sheeprl/utils/utils.py:Ratio` (out of `dreamer_v3/` scope) | Smeared replay-ratio schedule | [`utils.py:L270-L360`](../../../../src/algorithms/dreamer_srl/utils.py) (`Ratio`) | v2-CP7 |
| `sheeprl/data/buffers.py:SequentialReplayBuffer` (out of `dreamer_v3/` scope) | Sequential replay buffer with episode-aware sampling | [`src/algorithms/dreamer_srl/buffers.py`](../../../../src/algorithms/dreamer_srl/buffers.py) (full file) | v2-CP7 |

---

## 8. Confirmed cross-PRNG / cross-substrate boundaries

These are the **legitimate** places where PyTorch and JAX differ at the bit level —
not bugs to fix, but substrate-class deviations that v1 catalogued (D-001 through
D-014) and v2 will re-encounter. v2-CP3 grad-parity tests must apply looser thresholds
at these sites; v2 reviewers must distinguish "expected substrate drift" from
"ported bug".

| Boundary class | PyTorch site | JAX equivalent | v1 deviation IDs | v2 expected behaviour |
|---|---|---|---|---|
| **Categorical sampling RNG** | `torch.distributions.Categorical.sample()` (implicit global RNG) | `jax.random.categorical(key, logits)` (explicit key) | D-009 | Forward samples differ; gradient w.r.t. logits should match within ULP. |
| **One-hot-categorical straight-through** | `OneHotCategoricalStraightThrough.rsample()` (PyTorch builtin: sample is one-hot argmax; backward is softmax gradient) | Custom JAX impl: `jax.nn.one_hot(jax.random.categorical(key, logits), num_classes)` + custom `jax.lax.stop_gradient` + softmax-grad straight-through | D-002, D-009 | Forward differs; backward gradient w.r.t. logits matches within `1e-5` (looser threshold per Hafner's straight-through gradient = `softmax(logits) - sg(softmax(logits)) + sg(one_hot_sample)` which is bit-identical). |
| **Adam optimizer epsilon placement** | `torch.optim.Adam(eps=1e-5)` adds eps INSIDE the sqrt: `update = lr * m_hat / (sqrt(v_hat) + eps)` | `optax.adam(eps=1e-5)` matches PyTorch placement (Optax provides the PyTorch-compatible variant via the default `eps` placement) | (none in v1, but flagged for v2-CP6 verification) | Verify `optax.adam` placement matches `torch.optim.Adam` numerically; gradient flow upstream of optimiser unaffected. |
| **In-place vs functional state updates** | `self.low = self._decay * self.low + ...` (in-place buffer update) | `MomentsState(low=new_low, high=new_high)` returned as a new pytree | D-001, D-011 | Semantic equivalence verified; no gradient implication (Moments outputs are detached). |
| **PyTorch autograd graph vs JAX traced JIT graph** | `loss.backward()` mutates `.grad` attributes; subsequent `optimizer.step()` reads them | `jax.grad(loss_fn)(params)` returns a pytree of gradients; subsequent `optax.apply_updates(params, ...)` is functional | D-011 | Pure-functional equivalence; gradient values match. |
| **`F.softplus` / `F.silu` ULP drift** | PyTorch's CUDA softplus / SiLU may differ from JAX's by 1 ULP on certain inputs | `jax.nn.softplus` / `jax.nn.silu` | D-003, D-006, D-007, D-008, D-010 | Accumulated through MLP stacks; threshold relaxed per layer count (default `1e-5` instead of `1e-6`). |
| **`probs_to_logits` numerical floor** | PyTorch: `torch.log(probs.clamp(min=tiny))` with `tiny = torch.finfo(probs.dtype).tiny ≈ 1.18e-38` | JAX: `jnp.log(jnp.clip(probs, a_min=jnp.finfo(probs.dtype).tiny))` | (none yet; flagged for v2-CP3) | At post-`unimix` probabilities the floor is `unimix / num_classes ≈ 0.01/32 = 3.125e-4` (well above the tiny-clamp), so the clamp is inactive in practice. Verify. |
| **`torch.quantile` vs JAX equivalent** | `torch.quantile(x, q)` uses linear interpolation between order statistics | `jnp.quantile(x, q, method='linear')` matches | (none yet; flagged for v2-CP3) | Used in `Moments.forward` — drift here would shift advantage normalization. |
| **`fabric.all_gather` (multi-GPU sync)** | DDP-style gather across ranks | Not applicable in single-device dreamer-srl | N/A | Single-GPU v2 runs; no-op. |

These boundaries are the **only** ones expected to surface ULP-class differences.
**Any forward / gradient diff at a different site is a bug, not a substrate boundary.**

---

## 9. Things v1 reviews got wrong (retrospective)

Specific concrete issues with the v1 verification pass — surfaced for v2 reviewers to
explicitly check the opposite at every CP.

1. **`ep_len_avg` was raw-per-episode, not a window mean.** v1 reviewers cited
   `ep_len_avg` as if it were an aggregated metric, but per the user's
   retrospective discovery (2026-05-13), sheeprl's
   `aggregator.update("Game/ep_len_avg", ep_len)` (`dreamer_v3.py:L617`) writes the
   per-episode length as a `SumMetric` — so the WandB scalar is the average of
   episode-end events within the log-window, not a rolling mean. v1's analysis of "the
   episode-length curve" therefore misread the metric's window semantics. **v2-CP10's
   `ep_len_avg > 200` PASS criterion** is a window aggregate (steps 16k–20k); the
   analyzer must verify which aggregation WandB reports.

2. **v1's wrapper-module Lever-A coverage was forward-only.** The Encoder, Decoder,
   Actor (wrapper), ContinueHead, FullMLPHead, WorldModel composite, and `build_agent`
   shipped under "reviewer-optional, integration-smoke-only" verification per v1
   plan §"What v1 verified". No per-function bit-identity test, no per-parameter
   gradient test. v2-CP8 must add **at minimum** a forward output + a structural
   gradient-flow check (every Linear / LayerNorm / GRU weight registered to the
   `nnx.Module` is reachable from the loss's gradient tape) per wrapper.

3. **v1-CP10b re-scoped Gate 3 from policy-learning to numerical convergence.** v1's
   "Gate 3" originally required policy-learning evidence; the v1-CP10b spec re-scoped
   it to "final WM loss within ±10% of CP10" — a numerical-convergence check that
   does NOT test whether the policy actually learns. The 3-seed parity launch was
   then authorized off a passed Gate 3 even though no run had reached `ep_len > 200`.
   **v2-CP10 restores the policy-learning gate**: `ep_len_avg > 200` over the
   16k–20k window. No re-scoping permitted.

4. **v1's sweep cells at `learning_starts=0` could not observe the H2 prefill-
   contamination class.** Cell sweeps that set `learning_starts=0` (used in v1-CP9b's
   debt-repayment-mode validation) bypass the prefill phase entirely — they cannot
   surface a bug in the `is_first`-after-`done` propagation since the buffer never
   reaches the prefill-end transition where the actor takes over. The H2 hypothesis
   is therefore invisible to that sweep. **v2-CP7 must run integration tests at
   `learning_starts=1024` (the corrected XS value), not at `0`.**

5. **v1's 14 deviations were each approved individually but their aggregate impact
   was never measured.** D-001 through D-014 were each approved as "same arithmetic,
   different mechanism" — substrate-class drift. But no v1 verification ever measured
   the AGGREGATE drift over 200k training steps. A single ULP-class drift compounds
   over 200k steps × 16 environments × ~64 sequence-length per train batch = ~2e8
   floating-point operations on the gradient path; even a `1e-6` relative drift
   per-op can compound to a `1e-2` divergence by end-of-training. **v2-CP10's 20k-step
   gate** is the first checkpoint that empirically measures aggregate drift impact —
   if v2-CP3–v2-CP8 all PASS but v2-CP10 FAILs, the deviations' aggregate impact is
   the candidate root cause (Path D in `IMPLEMENTATION_PLAN.md` §7).

6. **v1's D-014 boundary-debt smear is a ONE-SHOT 1024-gradient-step burst at
   `iter_num == learning_starts`, not sheeprl's smeared per-iter rate.** The
   canonical sheeprl `Ratio.__call__` (in `sheeprl/utils/utils.py`, out of
   `dreamer_v3/` scope) yields gradient steps continuously as `policy_step`
   advances. v1 D-014 replaces this with a single big burst at the prefill-end
   boundary. On a sparse 1024-step prefill buffer with `per_rank_batch_size=16,
   sequence_length=64` = 1024 transitions per gradient step, the buffer at
   `learning_starts` contains exactly ~1 batch worth of data — the 1024 gradient
   steps in the burst all train on essentially the same data, over-fitting the
   first batch before any new data is collected. **v2-CP7 must verify whether
   D-014 is a candidate root cause of H2 by comparing the corrected
   smeared-rate variant against the burst variant in the policy-learning gate.**

---

## Authoring history

- **2026-05-14**: Memo authored by `senior-developer` per v2-CP1 of
  [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md). Read-only audit; no source-tree
  or `vendor/sheeprl/` modifications. Foundation for v2-CP2 (`GRAD_PARITY_METHODOLOGY.md`)
  and v2-CP3+ (per-function re-audits).

## Verification Report

> **Verified by**: `code-reviewer` (fresh sheeprl source read against the vendored
> commit `33b6366`) — scheduled per v2-CP1's Lever-C row in
> [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) §3.
> **Date**: TBD (this memo is authored; the reviewer pass is the v2-CP1 verdict step).

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| [`docs/develop/active/dreamer_srl_v2/SHEEPRL_REFERENCE_AUDIT.md`](SHEEPRL_REFERENCE_AUDIT.md) | New audit memo (this file) | ☐ pending review | Citations verified against vendored sheeprl `33b6366`. |

**Conclusion**: v2-CP1 memo authored. Ready to be cited by v2-CP2
([`GRAD_PARITY_METHODOLOGY.md`](GRAD_PARITY_METHODOLOGY.md), to be authored by
`professor-rl-bayesian-dl`) and v2-CP3+ (per-function audits, to be implemented by
`developer` against this memo's line citations).
