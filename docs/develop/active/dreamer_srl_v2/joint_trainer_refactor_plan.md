---
title: "dreamer-srl v2 — JointTrainer refactor (collapse 7 modules into 1 composite to unlock lax.scan perf)"
topic: dreamer
status: active
created: 2026-05-20
last_updated: 2026-05-20
phase: 3

---

# dreamer-srl v2 — JointTrainer refactor (collapse 7 modules into 1 composite to unlock lax.scan perf)

> **Status**: PLANNED
> **Opened**: 2026-05-20
> **Related**:
> - Triggering finding: [`docs/develop/active/dreamer_srl_v2/buffer_perf_fix_plan_option_L.md`](buffer_perf_fix_plan_option_L.md) §Step 3 — the current 7-module decomposition makes the scanned grad-step loop **~70× SLOWER** than the Python for-loop, with a CUDA-graph memory leak.
> - Perf-regression diagnosis: [`docs/memory/memories/dreamer_diagnosis/20260519_1507_dreamer_srl_v2_cpu_buffer_regression.md`](../../../memory/memories/dreamer_diagnosis/20260519_1507_dreamer_srl_v2_cpu_buffer_regression.md)
> - The 4-phase retrofit that fixed the same class of bug on the original JAX Dreamer: [`docs/memory/memories/dreamer_diagnosis/20260519_1508_dreamer_jax_perf_retrofit_4_phases.md`](../../../memory/memories/dreamer_diagnosis/20260519_1508_dreamer_jax_perf_retrofit_4_phases.md)
> - The `nnx.split / nnx.merge` rule-set this plan inherits: [`docs/memory/memories/dreamer_diagnosis/20260519_1509_nnx_lax_scan_split_merge_pattern.md`](../../../memory/memories/dreamer_diagnosis/20260519_1509_nnx_lax_scan_split_merge_pattern.md)
> - The parity-pass discipline this plan extends: [`docs/memory/memories/dreamer_diagnosis/20260518_1511_dreamer_srl_v2_parity_pass_outperform.md`](../../../memory/memories/dreamer_diagnosis/20260518_1511_dreamer_srl_v2_parity_pass_outperform.md)
> - The Strong A+B+C+D audit-chain discipline: [`docs/memory/memories/dreamer_diagnosis/20260513_2308_strong_strategy_validates_on_cp1.md`](../../../memory/memories/dreamer_diagnosis/20260513_2308_strong_strategy_validates_on_cp1.md)
> - Working reference (the proven-correct composite-trainer pattern): `src/models/dreamer_v3_trainer.py:685-833` (the `_scan_train_gpu` body + `train_multiple_gpu` driver).

---

## Context

The new "dreamer-srl v2" trainer carries the world model, actor, critic, target-critic, and three optimizers as **seven separate `nnx.Module` / `nnx.Optimizer` instances** in the training driver. When the grad-step loop was wrapped in `jax.lax.scan` — a JAX construct that fuses a fixed-length Python `for` loop into a single GPU kernel — performance got **~70× WORSE** and the GPU started leaking memory. The root cause is that with seven separate modules, the scan body must execute seven `nnx.split` calls (outside) and seven `nnx.merge` calls (inside) every iteration, and JAX has to track seven graph-definitions through every step; the resulting CUDA-graph bookkeeping dominates. The original JAX Dreamer (the unrelated `JAX_DreamerV3` trainer at `src/models/dreamer_v3_trainer.py`) does NOT have this problem because it carries everything in **one composite `nnx.Module` called `DreamerV3Trainer`** that holds the world model, actor, critic, and optimizers as attributes — one split, one merge, one update. This plan collapses dreamer-srl v2 into the same one-module shape. Concretely we introduce a `JointTrainer(nnx.Module)` whose attributes are exactly the seven things the current driver tracks separately, then rewire the scan path to do one `nnx.split(joint)` outside and one `nnx.merge(graphdef, current_state)` inside — matching the working template at `dreamer_v3_trainer.py:791-833`. Success looks like: every test still passes byte-for-byte (the 11 grad-parity tests vs sheeprl) or to math-equivalence (the 5 lax-scan equivalence tests), and the scan path's measured steps-per-second beats the Python for-loop on the same node + GPU + seed + budget.

## Analysis

### Why we hit a wall in Step 3 of the Option-L plan

Step 3 of [`buffer_perf_fix_plan_option_L.md`](buffer_perf_fix_plan_option_L.md) ported the `lax.scan` pattern from the original Dreamer (which sustains ~32 SPS on `yxij4lrc`, the production XS/16/4M reference) directly onto dreamer-srl v2. The math equivalence tests (`tests/algorithms/dreamer_srl/test_lax_scan_train.py`, 5 tests at `atol=1e-5`) **pass** — so the math port is correct. But the SPS bench reports a regression of roughly two orders of magnitude on the scanned grad-step path, and `nvidia-smi` shows GPU memory creeping up between iterations. The structural difference is shape, not math:

| Property | Original Dreamer (works fast) | dreamer-srl v2 today (slow when scanned) |
|---|---|---|
| Top-level `nnx.Module` | 1 (`DreamerV3Trainer`) | n/a — driver holds 7 refs |
| `nnx.Optimizer` count | 1 (composite, wrapped inside the trainer) | 3 (`wm_opt`, `actor_opt`, `critic_opt`) standalone |
| `nnx.split` calls outside scan | 1 | 7 |
| `nnx.merge` calls inside scan body | 1 | 7 |
| `nnx.state` calls inside scan body | 1 | up to 4 (one per module to extract carry) |
| `nnx.update` calls after scan | 1 | 7 |
| Per-iteration graphdef objects in closure | 1 | 7 |

Every extra graphdef is one more static-meta object the XLA cache has to key on; every extra `merge` inside the body is one more pytree-walk per step. The remedy is structural: collapse the seven instances into one composite owner.

### Why this is safe to do

`nnx.Module` composition is the canonical pattern in Flax NNX — a module can hold other modules as attributes, and `nnx.split` / `nnx.merge` / `nnx.state` / `nnx.update` walk the composite recursively. The `JointTrainer(nnx.Module)` simply re-homes the attribute references; the underlying `WorldModel`, `Actor`, `Critic`, and `nnx.Optimizer` objects (and their parameter arrays) are unchanged. **No math changes. No optimizer-hyperparameter changes. No buffer changes.** The only externally visible surface that moves is the **driver code** at `dreamer_srl_main.py` — and the rewire is mechanical (replace `world_model` with `joint.world_model` etc.).

The working template at `src/models/dreamer_v3_trainer.py:791-833` (`train_multiple_gpu`) is the goal shape. Compare its body to what dreamer-srl v2's scan path should look like post-refactor:

```python
# Goal shape (from dreamer_v3_trainer.py:797-832, paraphrased to dreamer-srl v2 names)
graphdef, _ = nnx.split(joint)                         # ONE split outside

def scan_body(carry, batch_i):
    current_state, rng = carry
    trainer = nnx.merge(graphdef, current_state)       # ONE merge inside
    new_moments, losses = trainer.train_step(batch_i, rng)
    new_state = nnx.state(trainer)                     # ONE state extract
    return (new_state, rng), losses

final_carry, losses = jax.lax.scan(scan_body, init_carry, scan_xs)

nnx.update(joint, final_carry[0])                      # ONE update after
```

vs. the seven-everything mess presently at `dreamer_srl_main.py:958-1079`.

### Why the rollback path is cheap

The `--legacy-grad-loop` flag already exists (`dreamer_srl_main.py:195-200`, default `False`). It selects the Python for-loop path that is bit-identity-equivalent to sheeprl per the L1 grad-parity tests. If anything goes wrong mid-transition, flipping the flag back to its pre-Step-5 default of `True` (or even just passing `--legacy-grad-loop` at the CLI for any compromised run) gives an immediate safe-mode fallback. **Do not remove this flag**; it is the rollback.

## Implementation Plan

### Design

#### The `JointTrainer` class

Introduce a new file `src/algorithms/dreamer_srl/joint_trainer.py` with a single class:

```python
# src/algorithms/dreamer_srl/joint_trainer.py
from __future__ import annotations
from flax import nnx
from src.algorithms.dreamer_srl.agent import WorldModel, Actor, Critic

class JointTrainer(nnx.Module):
    """Composite nnx.Module that owns every mutable training artefact.

    Mirrors the proven-correct shape of src/models/dreamer_v3_trainer.py:DreamerV3Trainer.
    Collapsing world model + actor + critic + target_critic + 3 optimizers into one
    composite enables a single nnx.split outside lax.scan and a single nnx.merge inside
    the scan body, matching the perf pattern at dreamer_v3_trainer.py:791-833.
    """

    def __init__(
        self,
        world_model: WorldModel,
        actor: Actor,
        critic: Critic,
        target_critic: Critic,
        wm_opt: nnx.Optimizer,
        actor_opt: nnx.Optimizer,
        critic_opt: nnx.Optimizer,
    ) -> None:
        self.world_model   = world_model
        self.actor         = actor
        self.critic        = critic
        self.target_critic = target_critic
        self.wm_opt        = wm_opt
        self.actor_opt     = actor_opt
        self.critic_opt    = critic_opt
```

That's the whole class. No methods, no logic. It is a **pure attribute container** so that `nnx.split(joint)` walks all seven children in one pass.

#### Optimizer-state verification (please read before C1)

The current driver constructs:

```python
# src/algorithms/dreamer_srl/dreamer_srl_main.py:321-323
wm_opt    = nnx.Optimizer(world_model, optax.adam(wm_lr,     eps=wm_eps),     wrt=nnx.Param)
actor_opt = nnx.Optimizer(actor,       optax.adam(actor_lr,  eps=actor_eps),  wrt=nnx.Param)
critic_opt = nnx.Optimizer(critic,     optax.adam(critic_lr, eps=critic_eps), wrt=nnx.Param)
```

These are **`flax.nnx.Optimizer` wrappers** (not bare `optax.OptState`). The `nnx.Optimizer` is itself an `nnx.Module` — it composes cleanly into `JointTrainer` as an attribute, and `nnx.split` / `nnx.merge` / `nnx.update` handle it like any other child. The L2 test fixture at `tests/algorithms/dreamer_srl/test_lax_scan_train.py:123-130` already demonstrates this works under `nnx.split` today (it just splits seven things instead of one). The `wrt=nnx.Param` argument is preserved verbatim in the JointTrainer construction site — it is a property of the optimizer, not of how the optimizer is held.

#### Carry-and-scan shape post-refactor

Driver carry today (10 entries; `dreamer_srl_main.py:969-974`):
```
(state_wm, state_ac, state_cr, state_tg,
 state_wm_opt, state_ac_opt, state_cr_opt,
 moments, key, step_idx)
```

Driver carry post-refactor (4 entries):
```
(state_joint, moments, key, step_idx)
```

The `moments`, `key`, and `step_idx` items stay in the carry because they are NOT `nnx.Module` attributes — they are scalar / dict / PRNG state that the train-step closure consumes alongside the trainer. They do not need to move into `JointTrainer`. (We could, but it would force a `MomentsState` rewrite for no measured gain. **Non-goal**.)

#### Why `step_idx` stays out of `JointTrainer`

`step_idx` is the Polyak-update scheduler counter (`dreamer_srl_main.py:973`). It is a Python int promoted to `jnp.int32` per scan iteration. Holding it in `JointTrainer` would force every `nnx.split` to traverse it as a non-parameter leaf — fine in principle but a fixture-fragility risk for the L1 grad-parity tests (changing what `nnx.state(trainer, nnx.Param)` returns is unsafe). Keep it in the scan carry instead.

### Goals & Non-goals

**Goals**:

- (G1) Collapse the seven `nnx.Module` / `nnx.Optimizer` instances tracked by the driver into one composite `JointTrainer(nnx.Module)`.
- (G2) Preserve byte-for-byte equivalence on the L1 grad-parity tests (`test_grad_parity.py`, 11 tests) under the Python for-loop path.
- (G3) Preserve math-equivalence at `atol=1e-5` on the L2 scan-equivalence tests (`test_lax_scan_train.py`, 5 tests) under the scan path.
- (G4) Unlock the scan path's perf: measured SPS on the scan path under `JointTrainer` must equal-or-exceed the SPS of the legacy Python for-loop on the same node + GPU + config + seed + step budget.
- (G5) Preserve the L3 food-only smoke launch: 50k env-steps must yield `ep_len_avg ≥ 100` after the refactor lands.

**Non-goals** (explicit — flag any drift toward these in your Implementation Report):

- NO math change of any kind — loss formulas, KL terms, advantage normalization, two-hot encoding, Polyak schedule, optimizer choice, gradient clipping. The composite is a pure structural refactor.
- NO buffer change — `SequentialReplayBuffer` and the CPU/GPU device flag are untouched.
- NO optimizer hyperparameter change — `wm_lr`, `actor_lr`, `critic_lr`, `*_eps`, `wrt=nnx.Param` all unchanged.
- NO API surface change outside the trainer module — `Player.__init__(world_model, actor, num_envs)` and `dreamer_srl_eval_rollout(world_model=…, actor=…)` keep their current signatures; the driver passes `joint.world_model` and `joint.actor` to them.
- NO removal of the `--legacy-grad-loop` CLI flag. It is the rollback. The default may flip (see C5) but the flag itself stays.
- NO change to the build-agent factory (`agent.py:1934 build_agent`) — it still returns `(world_model, actor, critic, target_critic)`; `JointTrainer` is constructed in the driver one level up.
- NO change to the `make_train_step` signature in `train.py:613-660`. It continues to take `(world_model, actor, critic, target_critic, wm_opt, actor_opt, critic_opt, moments, batch, key)` and is called as `train_step(joint.world_model, joint.actor, …)` from the for-loop and as `trainer.world_model, trainer.actor, …` from the scan body where `trainer = nnx.merge(graphdef, current_state)`.

### Call-site enumeration (every reference to the seven module/optimizer instances in `dreamer_srl_main.py`)

Walked top-to-bottom of `src/algorithms/dreamer_srl/dreamer_srl_main.py`. Every line below currently uses one of the seven bare references. Each row names the commit that rewires it (C1–C6 per the [Commit topology](#commit-topology-c1-c6) section).

| # | Line(s) | What it does | Rewires in commit |
|---|---|---|---|
| 1 | `64-66, 80, 83, 90-91, 129, 133, 148` | `Player.__init__` and `Player.get_actions`/`init_states` — bind `world_model`, `actor` as attributes; use them in `vmap(encoder)`, `rssm.get_initial_states`, `rssm.dynamic`, `self.actor(latent, k_act)` | none — Player keeps existing signature; driver passes `joint.world_model, joint.actor` at construction (line 336) |
| 2 | `310` | `world_model, actor, critic, target_critic = build_agent(...)` | C1 (no rewire; assignment stays; `JointTrainer` constructed immediately after) |
| 3 | `321-323` | `wm_opt`, `actor_opt`, `critic_opt` constructed as `nnx.Optimizer` wrappers | C1 (assignment stays; passed into `JointTrainer` immediately) |
| 4 | `336` | `player = Player(world_model, actor, num_envs)` | C2 — rewire to `Player(joint.world_model, joint.actor, num_envs)` |
| 5 | `757-760` | `_save_checkpoint(..., world_model=world_model, actor=actor, critic=critic, target_critic=target_critic, ...)` | C2 — rewire to `world_model=joint.world_model, actor=joint.actor, critic=joint.critic, target_critic=joint.target_critic` |
| 6 | `784-785` | `dreamer_srl_eval_rollout(world_model=world_model, actor=actor, ...)` — checkpoint-video pass | C2 — rewire to `world_model=joint.world_model, actor=joint.actor` |
| 7 | `824-825` | `dreamer_srl_eval_rollout(world_model=world_model, actor=actor, ...)` — checkpoint-stats pass | C2 — rewire to `world_model=joint.world_model, actor=joint.actor` |
| 8 | `923-924, 929` | Polyak update in for-loop: `nnx.state(critic, nnx.Param)`, `nnx.state(target_critic, nnx.Param)`, `nnx.update(target_critic, ...)` | C2 — rewire to `joint.critic`, `joint.target_critic` |
| 9 | `937-938` | `train_step(world_model, actor, critic, target_critic, wm_opt, actor_opt, critic_opt, ...)` — for-loop body | C2 — rewire to `train_step(joint.world_model, joint.actor, joint.critic, joint.target_critic, joint.wm_opt, joint.actor_opt, joint.critic_opt, ...)` |
| 10 | `958-964` | Seven `nnx.split(…)` calls preparing scan inputs | C3 — collapse to one `graphdef_joint, state_joint = nnx.split(joint)` |
| 11 | `969-974` | 10-entry `init_carry` tuple | C3 — collapse to 4-entry `(state_joint, moments, key, jnp.array(cumulative_grad_steps))` |
| 12 | `984-998` | 7-entry scan-body carry unpack | C3 — collapse to 4-entry unpack |
| 13 | `1014-1029` | Polyak update inside scan body: `nnx.merge(graphdef_cr, s_cr)` + `nnx.merge(graphdef_tg, s_tg)` + `nnx.state(_cr_for_polyak, nnx.Param)` + `nnx.update(_tg_for_polyak, …)` + `nnx.state(_tg_for_polyak)` | C3 — collapse: one `trainer = nnx.merge(graphdef_joint, state_joint)`; access `trainer.critic`, `trainer.target_critic`; apply Polyak via `trainer.target_critic = …` then extract `nnx.state(trainer.target_critic)` (or use `nnx.update(trainer.target_critic, new_params)` then return `nnx.state(trainer)`). See the worked BEFORE/AFTER below. |
| 14 | `1032-1038` | Seven `nnx.merge(...)` calls (one per module/optimizer) | C3 — fold into the single `trainer = nnx.merge(graphdef_joint, state_joint)` above |
| 15 | `1042-1046` | `train_step(_wm, _ac, _cr, _tg, _wm_o, _ac_o, _cr_o, ...)` | C3 — rewire to `train_step(trainer.world_model, trainer.actor, trainer.critic, trainer.target_critic, trainer.wm_opt, trainer.actor_opt, trainer.critic_opt, ...)` |
| 16 | `1049-1060` | 7-entry `new_carry` with `nnx.state(_wm)` ... `nnx.state(_cr_o)` | C3 — collapse to 4-entry `(nnx.state(trainer), new_moments, carry_key, step_idx + 1)` |
| 17 | `1064-1066` | `jax.lax.scan(_scan_body, init_carry, scan_xs)` | C3 — no change (signature stays; carry shape changes) |
| 18 | `1069-1071` | 10-entry `final_carry` unpack | C3 — collapse to 4-entry unpack |
| 19 | `1073-1079` | Seven `nnx.update(...)` calls after the scan | C3 — collapse to one `nnx.update(joint, state_joint_f)` |

Additional file the developer must touch:

- **`src/algorithms/dreamer_srl/joint_trainer.py`** — new file, ~30 LoC, the `JointTrainer` class above (C1).

Files that do **NOT** change:

- `src/algorithms/dreamer_srl/agent.py` (the `build_agent` factory keeps its current return tuple — the driver constructs `JointTrainer` one level up).
- `src/algorithms/dreamer_srl/train.py` (the `make_train_step` / `one_train_step` signature is unchanged; callers pass `joint.world_model`, `joint.actor`, … as positional args).
- `src/algorithms/dreamer_srl/checkpoint.py` (`save_checkpoint` / `restore_checkpoint` keep their keyword signatures; driver passes `joint.world_model`, `joint.actor`, … as the values).
- `src/algorithms/dreamer_srl/eval.py` (`dreamer_srl_eval_rollout` keeps `world_model=…, actor=…` kwargs).
- `src/algorithms/dreamer_srl/buffers.py`, `loss.py`, `utils.py`.
- All `tests/algorithms/dreamer_srl/*.py` — the test fixtures already build the 7 modules separately (e.g. `test_lax_scan_train.py:102-130`). Leaving the test fixtures untouched means the L1 + L2 tests stay an **independent** validator of correctness (they don't share the new code path). **Symmetry note**: this is intentional — the test fixtures should NOT be rewritten to use `JointTrainer`. If they were, a bug in `JointTrainer` could pass the test by being symmetric with itself. The asymmetry is the value.

### Scan-body change — worked BEFORE/AFTER

The biggest behavioural rewrite is inside the scan body. Show the developer what success looks like.

#### BEFORE (current — `dreamer_srl_main.py:954-1080`, abbreviated)

```python
# Outside scan (7 splits)
graphdef_wm,      state_wm      = nnx.split(world_model)
graphdef_ac,      state_ac      = nnx.split(actor)
graphdef_cr,      state_cr      = nnx.split(critic)
graphdef_tg,      state_tg      = nnx.split(target_critic)
graphdef_wm_opt,  state_wm_opt  = nnx.split(wm_opt)
graphdef_ac_opt,  state_ac_opt  = nnx.split(actor_opt)
graphdef_cr_opt,  state_cr_opt  = nnx.split(critic_opt)

init_carry = (
    state_wm, state_ac, state_cr, state_tg,
    state_wm_opt, state_ac_opt, state_cr_opt,
    moments, key,
    jnp.array(cumulative_grad_steps, dtype=jnp.int32),
)

def _scan_body(carry, batch_i):
    (s_wm, s_ac, s_cr, s_tg,
     s_wm_opt, s_ac_opt, s_cr_opt,
     carry_moments, carry_key, step_idx) = carry

    # Polyak — 2 merges, 2 state extracts, 1 update, 1 state extract
    _cr_for_polyak = nnx.merge(graphdef_cr, s_cr)
    _tg_for_polyak = nnx.merge(graphdef_tg, s_tg)
    online_params = nnx.state(_cr_for_polyak, nnx.Param)
    target_params = nnx.state(_tg_for_polyak, nnx.Param)
    new_target_params = jax.tree.map(
        lambda c, t: jnp.where(do_update, (1.0 - tau_val) * t + tau_val * c, t),
        online_params, target_params,
    )
    nnx.update(_tg_for_polyak, new_target_params)
    s_tg_new = nnx.state(_tg_for_polyak)

    # train_step — 7 merges
    _wm   = nnx.merge(graphdef_wm,     s_wm)
    _ac   = nnx.merge(graphdef_ac,     s_ac)
    _cr   = nnx.merge(graphdef_cr,     s_cr)
    _tg   = nnx.merge(graphdef_tg,     s_tg_new)
    _wm_o = nnx.merge(graphdef_wm_opt, s_wm_opt)
    _ac_o = nnx.merge(graphdef_ac_opt, s_ac_opt)
    _cr_o = nnx.merge(graphdef_cr_opt, s_cr_opt)
    new_moments, losses = train_step(
        _wm, _ac, _cr, _tg, _wm_o, _ac_o, _cr_o,
        carry_moments, batch_i, k_train,
    )

    # Carry out — 7 state extracts
    new_carry = (
        nnx.state(_wm), nnx.state(_ac), nnx.state(_cr), s_tg_new,
        nnx.state(_wm_o), nnx.state(_ac_o), nnx.state(_cr_o),
        new_moments, carry_key, step_idx + 1,
    )
    return new_carry, losses

final_carry, losses_stack = jax.lax.scan(_scan_body, init_carry, scan_xs)

(s_wm_f, s_ac_f, s_cr_f, s_tg_f,
 s_wm_opt_f, s_ac_opt_f, s_cr_opt_f,
 moments, key, _step_idx_f) = final_carry

nnx.update(world_model,   s_wm_f)   # 7 updates
nnx.update(actor,         s_ac_f)
nnx.update(critic,        s_cr_f)
nnx.update(target_critic, s_tg_f)
nnx.update(wm_opt,        s_wm_opt_f)
nnx.update(actor_opt,     s_ac_opt_f)
nnx.update(critic_opt,    s_cr_opt_f)
```

#### AFTER (target shape — mirrors `dreamer_v3_trainer.py:791-832`)

```python
# Outside scan (1 split)
graphdef_joint, state_joint = nnx.split(joint)

init_carry = (
    state_joint,
    moments,
    key,
    jnp.array(cumulative_grad_steps, dtype=jnp.int32),
)

def _scan_body(carry, batch_i):
    state_joint, carry_moments, carry_key, step_idx = carry

    # ONE merge — reconstruct full composite trainer
    trainer = nnx.merge(graphdef_joint, state_joint)

    # Polyak — direct attribute access on the reconstructed trainer
    do_update = (step_idx % target_update_freq) == 0
    tau_val = jnp.where(step_idx == 0, jnp.float32(1.0), jnp.float32(critic_tau))
    online_params = nnx.state(trainer.critic, nnx.Param)
    target_params = nnx.state(trainer.target_critic, nnx.Param)
    new_target_params = jax.tree.map(
        lambda c, t: jnp.where(do_update, (1.0 - tau_val) * t + tau_val * c, t),
        online_params, target_params,
    )
    nnx.update(trainer.target_critic, new_target_params)

    # train_step — direct attribute access
    carry_key, k_train = jax.random.split(carry_key)
    new_moments, losses = train_step(
        trainer.world_model, trainer.actor, trainer.critic, trainer.target_critic,
        trainer.wm_opt, trainer.actor_opt, trainer.critic_opt,
        carry_moments, batch_i, k_train,
    )

    # ONE state extract — the train_step / Polyak mutations on `trainer`
    # already propagated to `trainer.world_model`, `trainer.actor`, …, in place
    # via NNX's reference semantics, so nnx.state(trainer) walks the whole
    # composite and returns the updated pytree.
    new_state = nnx.state(trainer)

    new_carry = (new_state, new_moments, carry_key, step_idx + 1)
    return new_carry, losses

final_carry, losses_stack = jax.lax.scan(_scan_body, init_carry, scan_xs)
state_joint_f, moments, key, _step_idx_f = final_carry

# ONE update — propagates back to the live joint.world_model, joint.actor, …
nnx.update(joint, state_joint_f)
```

Counts: split 7→1, merge 7→1 (plus the 2-merge Polyak shortcut goes away), state 7→1, update 7→1. This is structurally identical to `_scan_train_gpu` at `dreamer_v3_trainer.py:685-789`.

### Commit topology (C1–C6)

Each commit names exactly one gating test. Do not advance until the gate is green.

#### **C1 — Add `JointTrainer` class skeleton; thread its construction through the driver**

- New file: `src/algorithms/dreamer_srl/joint_trainer.py` (~30 LoC, the class above).
- Edit `src/algorithms/dreamer_srl/dreamer_srl_main.py` around line 323 (immediately after the three `nnx.Optimizer` constructions): insert `joint = JointTrainer(world_model, actor, critic, target_critic, wm_opt, actor_opt, critic_opt)`.
- **Do not yet rewire any downstream reference.** `joint` is constructed but unused at the end of C1. The existing 7 bare references continue to drive both the for-loop and the scan paths.
- **Gate**: full test suite stays green. Run `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest tests/algorithms/dreamer_srl/ -q -x` and confirm no new failures vs. the pre-C1 baseline. **Expected wall time**: ~12-15 min.
- **Why this is its own commit**: lets `code-reviewer` audit the new class in isolation before any behavioural change.

#### **C2 — Rewire the Python for-loop path to use `joint.*` attributes**

- Edit `dreamer_srl_main.py`:
  - Line 336 (Player construction): `joint.world_model`, `joint.actor`.
  - Lines 757-760 (checkpoint save): `joint.world_model`, `joint.actor`, `joint.critic`, `joint.target_critic`.
  - Lines 784-785 + 824-825 (eval rollout calls): `joint.world_model`, `joint.actor`.
  - Lines 923-929 (legacy Polyak): `joint.critic`, `joint.target_critic`.
  - Lines 937-938 (`train_step` call in for-loop): all seven references go through `joint.*`.
- Do NOT touch the scan path (lines 945-1079) in this commit — it still uses the 7 bare refs.
- **Gate**: L1 grad-parity, bit-identity (11 tests). Run `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest tests/algorithms/dreamer_srl/test_grad_parity.py -q -x`. Expected wall time: ~7 min.
- **Why L1 not L2**: the for-loop path is what L1 exercises. L2 still hits the unchanged (7-everything) scan path here. Both should still pass, but L1 is the **load-bearing** gate at C2.
- **What it proves**: the JointTrainer's attribute structure is bit-identity-equivalent under reference semantics — accessing `joint.world_model` yields the same module object the driver previously held as `world_model`, so the for-loop's compute is unchanged at the byte level.

#### **C3 — Rewire the scan path to single split/merge**

- Edit `dreamer_srl_main.py:954-1079` to the BEFORE→AFTER shape shown above. The 7 splits collapse to 1; the scan body uses `trainer = nnx.merge(graphdef_joint, state_joint)` and then direct attribute access; the 7 post-scan updates collapse to 1.
- **Gate**: L2 math-equivalence, `atol=1e-5` (5 tests). Run `/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest tests/algorithms/dreamer_srl/test_lax_scan_train.py -q`. Expected wall time: ~5 min.
- **Plus regression-check**: also re-run L1 (`test_grad_parity.py`) — it should still pass since C2 already wired the for-loop path through `joint.*` and C3 doesn't touch the for-loop. If L1 regresses at C3, something in the scan path is leaking state back into the for-loop (e.g., a stale-reference bug — see Risks §R1). **Both L1 + L2 must be green** before advancing to C4.
- **What it proves**: the scan path with one split/merge produces the same numbers (to `atol=1e-5`) as the for-loop path. Math is preserved by construction.

#### **C4 — Perf bench on the scan path under `JointTrainer`**

- No code change. Run the bench harness already in the repo: `tests/algorithms/dreamer_srl/bench_sps.py` (created in Step 0.1 of the Option-L plan).
- Launch via `run_command.py` on **Node 114 cuda:1** (cuda:0 is occupied by the Option-M bench; cuda:2 is reserved for C6; cuda:3 is unused — keep it free).
- Use `XLA_FLAGS='--xla_gpu_enable_command_buffer='` for parity with the Step-0 baseline measurement (per Option-L plan §0.1 baseline note).
- Bench both paths back-to-back, same node + GPU + config + seed, for a fair head-to-head:
  1. `--legacy-grad-loop` (Python for-loop path, post-refactor) — record SPS_legacy.
  2. Default (scan path under `JointTrainer`) — record SPS_scan.
- Use `configs/dreamer_srl/01_food_only_smoke.yaml` + `configs/experiment/dreamer_curriculum/01_food_only.yaml`, `--num-envs 16`, `--total-steps 50000`, `--seed 0`, `--no-wandb`.
- **Gate**: SPS_scan ≥ SPS_legacy on the same hardware/config/seed/budget. Record both numbers + the ratio in the Implementation Report.
- **Soft success bar**: SPS_scan ≥ 32 (the original Dreamer's `yxij4lrc` reference). Hard success bar (this gate): SPS_scan ≥ SPS_legacy. If SPS_scan < SPS_legacy, stop and triage — the structural collapse failed to deliver the expected XLA fusion, and we need a code-reviewer audit before continuing.
- **Memory check**: `nvidia-smi --query-gpu=memory.used --format=csv,noheader -l 5` for the duration of the scan-path bench. Memory must be stable (no monotonic growth between iterations). The pre-refactor scan path leaked; the post-refactor scan path must not.

#### **C5 — Flip `--legacy-grad-loop` default behavior**

This is a one-line change. The flag's `default=False` setting in `dreamer_srl_main.py:195` already means the scan path is the default at the CLI level. After C3 + C4 prove the scan path is fast and correct, this commit:

- Updates the help string at `dreamer_srl_main.py:196-200` to reflect that the scan path is now the validated default (the existing help text already says "Default: False (scan path)" — fact-check this is accurate after C3 and tweak if needed).
- Updates the Option-L plan doc to mark Step 3 as DONE and Step 4 / Step 5 as superseded by this plan.
- **Gate**: L1 + L2 both green with the new default. Run both test files. Expected wall time: ~12 min.

If the developer determines C5 is purely documentation/comment cleanup with no behaviour change (the default flip already happened mechanically in C3), this commit can be merged into C3 — see planner's call below.

#### **C6 — Food-only parity launch (L3 end-to-end smoke)**

- Launch the 50k-env-step food-only smoke via `run_command.py` on **Node 114 cuda:2** (`cuda:1` is busy with the C4 bench from the same session; cuda:0 is the Option-M bench).
- Configs: `configs/dreamer_srl/01_food_only_smoke.yaml` + `configs/experiment/dreamer_curriculum/01_food_only.yaml`.
- WandB logging ON (this is the end-to-end smoke; we want the full metric trace). Record the run ID in the Implementation Report.
- **Gate**: `ep_len_avg ≥ 100` at the end of the 50k-step run.
- Expected wall time: ~30 min on a free node.
- **What it proves**: the composite trainer end-to-end produces a training trajectory consistent with the production reference. If `ep_len_avg < 100`, we have a silent integration bug that bypassed L1 + L2 + C4 — most likely a stale-reference (R1) or a `Player` capturing the pre-refactor `world_model` (see R6).

### Risks

#### R1 — Stale-reference bug (HIGHEST risk)

**The pattern to avoid:** at line 336 today, `player = Player(world_model, actor, num_envs)` captures the live `world_model` reference. Post-refactor, the driver constructs `joint`, then needs to keep `player.world_model` pointing at `joint.world_model`. Because `nnx.Module` attribute access returns the **same underlying object** (reference semantics, not a copy), `joint.world_model is world_model` holds — so `Player(joint.world_model, joint.actor, num_envs)` is bit-identical to `Player(world_model, actor, num_envs)` at construction. But if the developer later writes `world_model = joint.world_model` as a convenience binding and then uses the bare `world_model` after a `nnx.update(joint, …)` call, the bare name **still points at the original object** — which `nnx.update` mutated in place via reference semantics, so this is actually safe.

**Where it goes wrong:** if the developer instead writes `world_model = nnx.merge(graphdef, state)` inside a scan body and then accidentally references the **outer** `world_model` name in a later iteration — the merge created a NEW object, and the outer name still points at the original. Mitigation: use **only** `trainer.world_model`, `trainer.actor`, etc. inside the scan body. **Never** bind a short name. The worked AFTER block above models the correct pattern.

#### R2 — Optimizer-state pytree shape change

If `nnx.split(joint)` produces an `Optimizer` substate with a different pytree structure than `nnx.split(wm_opt)` alone (e.g., extra wrapping), then `train_step(joint.wm_opt, …)` inside the scan body might see a subtly different state shape than the for-loop path. The access pattern for substates of an `nnx.State` is not guaranteed stable across flax-nnx versions (the project pins `flax==0.12.4` per `dreamer_srl_main.py:321` comment) — dict-subscript access (`joint_state['wm_opt']`) and attribute access (`joint_state.wm_opt`) are both implementation-detail-dependent, so the plan does NOT prescribe a runtime assertion against a guessed layout. Instead, the developer captures the actual pytree-path layout at runtime via `jax.tree_util.tree_paths` and pastes it into the Implementation Report:

```python
# C1 one-shot diagnostic (delete after C3 lands)
_, joint_state = nnx.split(joint)
print("[C1 diag] joint_state tree_paths:", jax.tree_util.tree_paths(joint_state))
```

Paste the resulting tree-path list verbatim into the Implementation Report. The `code-reviewer` / `senior-developer` will verify in PR review that the `wm_opt` substate paths (and the `actor_opt`, `critic_opt` paths) match the standalone `nnx.split(wm_opt)` / `nnx.split(actor_opt)` / `nnx.split(critic_opt)` tree paths byte-for-byte. If the paths diverge in a way that affects `train_step` (e.g. an extra wrapping level introduced by the composite container), flag before C3 starts — do NOT proceed to the scan rewire until the divergence is understood. This print-and-paste pattern replaces the older "assert against a guessed dict key" pseudocode because the latter is non-runnable as written under flax 0.12.4.

#### R3 — `nnx.split` accidentally placed inside scan body

Per memory insight [`20260519_1509_nnx_lax_scan_split_merge_pattern.md`](../../../memory/memories/dreamer_diagnosis/20260519_1509_nnx_lax_scan_split_merge_pattern.md), the rule is: **split outside, merge inside, state inside, update outside**. The worked AFTER block above conforms. The developer must NOT write `nnx.split(trainer)` inside `_scan_body` — that would re-traverse the graph on every iteration and recreate the original 70× regression on a smaller scale.

#### R4 — Nested merges

Do not `nnx.merge(graphdef_joint, state_joint)` to get `trainer`, then `nnx.merge(trainer.wm_opt_graphdef, trainer.wm_opt_state)` to get the optimizer. The composite handles this for you — `trainer.wm_opt` is already a full live `nnx.Optimizer`. Mitigation: the AFTER block models the no-nested-merge pattern.

#### R5 — Optimizer wrapper type difference

If a future refactor moves any optimizer to bare `optax.OptState` (no `nnx.Optimizer` wrapper), composing it as a `JointTrainer` attribute would fail because `nnx.split` does not walk arbitrary pytrees as module children. **Today this is not a problem** — all three optimizers are `nnx.Optimizer` (verified at `dreamer_srl_main.py:321-323`). The plan flags this as a forward-compatibility note: if anyone later switches to bare `optax`, the JointTrainer carry will need a custom pytree-flatten or a `nnx.Optimizer` re-wrap step.

#### R6 — `Player` holds a stale `world_model` after `nnx.update(joint, …)`

The driver passes `joint.world_model` and `joint.actor` to `Player.__init__` at line 336. Between iterations, `nnx.update(joint, state_joint_f)` mutates `joint.world_model` **in place** via NNX reference semantics. So `player.world_model is joint.world_model` continues to hold, and inference uses the updated weights. **This is correct** — no mitigation needed beyond confirming it works at C2 (the for-loop gate). Flag for the developer: if for any reason `nnx.update` produced a fresh object (it does not, but a future NNX version might), `Player` would need re-binding each iteration. Document this expectation in the Implementation Report.

#### R7 — Removing the `--legacy-grad-loop` flag

**DO NOT remove this flag.** It is the rollback. The plan explicitly keeps it through C5 + C6. Future cleanup (after a multi-week perf-validation window) might remove it, but that is OUT OF SCOPE for this plan.

#### R8 — JIT cache pressure between C2 and C3

C2 leaves the scan path on the old 7-module shape. If a contributor accidentally runs the scan path between C2 and C3 lands, they will see the same 70× regression Step 3 of Option-L flagged. Mitigation: communicate explicitly in the C2 commit message that the scan path is still slow and `--legacy-grad-loop` should be used until C3.

### Rollback

Single line: `git revert <C1-commit>..<C6-commit>` (or any prefix range — each commit is individually revertible since C1 is purely additive and C2–C6 are mechanical rewires plus one new file).

During the transition (between commits), the `--legacy-grad-loop=true` path remains a safe-mode fallback — it bypasses the scan code entirely. A user hitting a problem on, say, the C5 default-flip commit can simply pass `--legacy-grad-loop` at the CLI to switch back to the validated for-loop path until the regression is fixed.

### File Changes (grouped by commit)

#### C1 — additive
- `src/algorithms/dreamer_srl/joint_trainer.py` *(NEW, ~30 LoC)* — the `JointTrainer(nnx.Module)` composite class.
- `src/algorithms/dreamer_srl/dreamer_srl_main.py` *(~5 LoC added around line 324)* — import `JointTrainer` and construct `joint = JointTrainer(world_model, actor, critic, target_critic, wm_opt, actor_opt, critic_opt)` immediately after the three optimizer constructions. The 7 bare names continue to drive both code paths.
- *(optional, removed in C3)* `src/algorithms/dreamer_srl/dreamer_srl_main.py` — temporary debug assertion per R2 to confirm optimizer-state pytree-shape parity.

#### C2 — Python for-loop rewire
- `src/algorithms/dreamer_srl/dreamer_srl_main.py` *(lines 336, 757-760, 784-785, 824-825, 923-929, 937-938; ~10 substitutions)* — rewire every bare-name reference in the non-scan paths to use `joint.world_model`, `joint.actor`, `joint.critic`, `joint.target_critic`, `joint.wm_opt`, `joint.actor_opt`, `joint.critic_opt`.

#### C3 — scan path rewire
- `src/algorithms/dreamer_srl/dreamer_srl_main.py` *(lines 945-1080; ~125 LoC rewritten to ~50 LoC)* — collapse the 7-everything scan path to the AFTER shape above. Drops the 7 `graphdef_*` / `state_*` locals, the 7-tuple carry, the 7 inner merges, the 7-tuple output unpack, and the 7 post-scan updates. Replaces with `graphdef_joint, state_joint = nnx.split(joint)` + 4-entry carry + `trainer = nnx.merge(graphdef_joint, state_joint)` + `nnx.state(trainer)` + `nnx.update(joint, state_joint_f)`.
- *(remove)* the R2 debug assertion added in C1.

#### C4 — no source change; bench measurement
- No `src/` changes. May add a CSV-output row format to `tests/algorithms/dreamer_srl/bench_sps.py` if needed — defer until the developer hits a real need.
- **Output**: `tmp/sps_bench_joint_trainer_C4_<timestamp>.csv` with SPS_legacy and SPS_scan rows.

#### C5 — comment / help-text cleanup
- `src/algorithms/dreamer_srl/dreamer_srl_main.py` *(lines 195-200, ~5 LoC)* — fact-check + tighten the `--legacy-grad-loop` help string.
- `docs/develop/active/dreamer_srl_v2/buffer_perf_fix_plan_option_L.md` — mark Step 3 as DONE (cross-link this plan); mark Step 4 / Step 5 as superseded by this plan.
- *(this commit may merge into C3 — see planner's call below)*

#### C6 — no source change; food-only L3 launch
- No `src/` changes. May add a stat-summary line to the Implementation Report.
- **Output**: a WandB run ID + `ep_len_avg` final value.

### Testing Strategy

Surface-level (one-paragraph for `code-reviewer` and `math-reviewer`):

The refactor is gated by **three independent test layers** that already exist in the repo:

- **L1 — `test_grad_parity.py`** (11 tests, bit-identity vs sheeprl PyTorch reference). Gates C2 (for-loop rewire) and is re-run at C3 + C5 as a regression check. ~7 min.
- **L2 — `test_lax_scan_train.py`** (5 tests, math equivalence at `atol=1e-5` between for-loop and scan paths). Gates C3 (scan rewire) and is re-run at C5. ~5 min.
- **L3 — food-only smoke launch** (50k env-steps, `ep_len_avg ≥ 100`). Gates C6. ~30 min on a free node.

Plus **C4 — perf bench** as a non-correctness gate: SPS_scan ≥ SPS_legacy on the same hardware. ~15 min for two runs.

**Load-bearing role of the R2 `tree_paths` diagnostic (do NOT skip).** The L2 test fixtures at `test_lax_scan_train.py:102-130` intentionally stay on the seven-everything carry shape — the design symmetry argument at Plan line 209 (tests should NOT share the new code path) explicitly forbids rewriting them to use `JointTrainer`. The benefit is that L2 remains an independent validator of correctness. The cost is a coverage gap: L2 never compares the JointTrainer carry's pytree structure against the seven-everything carry's pytree structure. If `nnx.split(joint)` produces a substate whose `wm_opt` (or `actor_opt` / `critic_opt`) substructure has a subtly different pytree-path layout than the standalone `nnx.split(wm_opt)` substate — extra wrapping, a renamed key, a swapped traversal order — the math will still match (so L2 stays green) but downstream code that addresses the substate by path (e.g. checkpoint save/restore, or any future code that introspects optimizer state) will silently break. The R2 diagnostic (the `jax.tree_util.tree_paths(joint_state)` print introduced in C1 and pasted into the Implementation Report) is the ONLY check that catches this class of drift before C3 lands. The L2 test does NOT cover it. Treat the diagnostic output as a load-bearing C1 deliverable: if the `developer` agent forgets to paste it into the report, the `senior-developer` verification must block C2 advance until it appears.

Run order per commit:
- C1 → full pytest (no behaviour change).
- C2 → L1 (load-bearing) + full pytest (regression).
- C3 → L2 (load-bearing) + L1 (regression).
- C4 → bench, head-to-head.
- C5 → L1 + L2 (regression).
- C6 → L3 (end-to-end).

If any gate fails, **stop**. Do not advance. Fix or revert before continuing.

### Hand-off

- **Implementer**: `developer` agent. The plan's File Changes section, Scan-body BEFORE/AFTER, and Commit topology give enough detail to execute without further design ambiguity.
- **Pre-implementation audit**: `code-reviewer` agent (separately spawned by the parent to audit this plan document before C1). Focus: verify the `JointTrainer` attribute list is complete; verify the BEFORE/AFTER scan-body shape matches the `dreamer_v3_trainer.py:791-833` template; verify R1 (stale-reference) and R2 (optimizer-state pytree shape) are real risks worth their mitigation steps.
- **Math-equivalence audit**: `math-reviewer` agent (separately spawned). Focus: confirm that wrapping the seven modules in a composite container introduces NO computational change — i.e., that `trainer.world_model.observe(…)` produces the same forward pass as `world_model.observe(…)`, that `nnx.update(joint, …)` propagates correctly to the children, and that Polyak via `trainer.target_critic` matches the original `target_critic` update. The math-reviewer should also confirm the L2 test (`test_lax_scan_train.py`) is a sufficient gate for "no math change."
- **Post-implementation verification**: `senior-developer` (me) runs the standard Verification Protocol after `developer` reports back: diff stats check, git diff against the File Changes section, flag unexpected changes, fill the Verification Report table.

## Checkpoints

What the implementing developer should verify during implementation:

- [ ] **C1.a** — `from src.algorithms.dreamer_srl.joint_trainer import JointTrainer` succeeds (no circular import).
- [ ] **C1.b** — `joint = JointTrainer(world_model, actor, critic, target_critic, wm_opt, actor_opt, critic_opt)` succeeds at runtime.
- [ ] **C1.c** — `nnx.split(joint)` returns a `(graphdef, state)` pair where `state` walks all 7 children. Print `jax.tree.structure(state)` once at C1 and paste into the Implementation Report for the `code-reviewer` to audit.
- [ ] **C1.d** — Full pytest suite green pre-vs-post C1 (zero new failures).
- [ ] **C2.a** — `joint.world_model is world_model` returns `True` (proves reference semantics).
- [ ] **C2.b** — After one gradient step via the for-loop path, the parameter arrays accessed via `joint.world_model` and via the bare `world_model` are still byte-identical (proves no copy was made).
- [ ] **C2.c** — L1 grad-parity green (`test_grad_parity.py`, 11 tests pass).
- [ ] **C3.a** — Five-constraint grep checklist on the post-refactor grad-step block of `dreamer_srl_main.py` (the block from the one `nnx.split(joint)` line through the one post-scan `nnx.update(joint, …)` line). All five must hold simultaneously; if any one fails, the scan rewire is wrong and C3 is not green:
  - `nnx.split` **outside** scan body (in the grad-step block, between the start of the block and the `def _scan_body(...)` line): count `== 1` (the single `graphdef_joint, state_joint = nnx.split(joint)`).
  - `nnx.split` **inside** scan body (between `def _scan_body(...)` and its `return`): count `== 0` (no inner splits — this is the R3 anti-pattern that would recreate the 70× regression).
  - `nnx.merge` **inside** scan body: count `== 1` (the single `trainer = nnx.merge(graphdef_joint, state_joint)`).
  - `nnx.update` **outside** scan body (post-`lax.scan`, in the grad-step block): count `== 1` (the single `nnx.update(joint, state_joint_f)` that propagates back to the live `joint.world_model`, `joint.actor`, … via NNX reference semantics).
  - `nnx.update` **inside** scan body (the Polyak target-update path): count `== 1` (the `nnx.update(trainer.target_critic, new_target_params)` line — this one is expected and is NOT the R3 anti-pattern; it is a write on a sub-module, not a fresh split of the composite).
- [ ] **C3.b** — L2 math-equivalence green (`test_lax_scan_train.py`, 5 tests pass at `atol=1e-5`).
- [ ] **C3.c** — L1 grad-parity STILL green at C3 (regression check; the C3 rewire must not leak back into the for-loop path).
- [ ] **C4.a** — `bench_sps.py` produces a CSV with both SPS_legacy and SPS_scan rows from back-to-back runs on the same node + GPU + config + seed + budget.
- [ ] **C4.b** — `SPS_scan ≥ SPS_legacy`. Document both numbers and the ratio.
- [ ] **C4.c** — `nvidia-smi` memory is **stable** (no monotonic growth) for the duration of the scan-path bench. Sample every 5 s; record max-min in the Implementation Report.
- [ ] **C5.a** — L1 + L2 both green with the new default behaviour.
- [ ] **C6.a** — Food-only 50k-step launch completes (no crash, no OOM).
- [ ] **C6.b** — `ep_len_avg ≥ 100` at end of run. Paste the WandB run ID into the Implementation Report.

## Implementation Report

> **Implemented by**: [to be filled by `developer` agent]
> **Date**: [to be filled]

[Sections for each of C1-C6: what was done, deviations from plan + rationale, gate-test outputs, perf numbers, WandB run IDs, any new follow-up issues discovered.]

## Verification Report

> **Verified by**: [to be filled by `senior-developer`]
> **Date**: [to be filled]

| Commit | File(s) | Change | Status | Notes |
|---|---|---|:---:|---|
| C1 | `joint_trainer.py` (new) + `dreamer_srl_main.py:~324` | Add JointTrainer class + construct in driver | | |
| C2 | `dreamer_srl_main.py` (lines 336, 757-760, 784-785, 824-825, 923-929, 937-938) | Rewire for-loop path to `joint.*` | | |
| C3 | `dreamer_srl_main.py:945-1080` | Collapse scan path to 1 split / 1 merge / 1 update | | |
| C4 | (no src change) `tmp/sps_bench_joint_trainer_C4_*.csv` | Perf bench, scan path beats for-loop | | |
| C5 | `dreamer_srl_main.py:195-200` + Option-L plan cross-link | Doc / help-text cleanup; mark Option-L Step 3 DONE | | |
| C6 | (no src change) WandB run ID | Food-only 50k-step launch, `ep_len_avg ≥ 100` | | |

**Conclusion**: [one-line summary]

---

### Revision history

- 2026-05-20: applied 3 revisions from code-reviewer (R2 pseudocode, L2 coverage callout, C3.a grep checklist). Math-reviewer accepted unchanged.

---

<!-- New-issue follow-up policy: if implementation surfaces a separate bug or design
     issue (e.g., the optimizer-state pytree shape really IS different, or the
     bench shows a different bottleneck), append as "## Issue #2: ..." here with
     the same template sections, OR open a separate doc and cross-link both ways.
     Do not silently expand the scope of this plan. -->
