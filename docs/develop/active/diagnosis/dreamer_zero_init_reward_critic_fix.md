---
title: "DreamerV3 Zero-Init Reward + Critic Output Layers (Cell A1 Re-run)"
topic: diagnosis
status: active
created: 2026-05-10
last_updated: 2026-05-10
phase: 1
---

# DreamerV3 Zero-Init Reward + Critic Output Layers (Cell A1 Re-run)

> **Status**: PLANNED — implementation + launch authorised (user, 2026-05-10).
> **Doc role**: doubles as design doc + Launch Manifest (the user has pre-authorised proceeding through implementation and training launch on node 113 without further gates).
> **Related**:
>   - [`docs/project/concepts/dreamer_v3_implementation.md`](../../../project/concepts/dreamer_v3_implementation.md) §6 item 27 + §9.11.4 — the deviation this plan acts on.
>   - [`.claude-memory/memories/dreamer_diagnosis/20260509_1534_wm_reward_head_localized_failure_a1.md`](../../../../.claude-memory/memories/dreamer_diagnosis/20260509_1534_wm_reward_head_localized_failure_a1.md) — the empirical finding (reward MAE 0.39 vs threshold 0.15) this plan targets.
>   - [`docs/experiments/active/dreamer_diagnosis/DREAMER_CONVENTIONAL_FIXES_BATTERY.md`](../../../experiments/active/dreamer_diagnosis/DREAMER_CONVENTIONAL_FIXES_BATTERY.md) — Cell A1 baseline (`czfnljf0`) this run is compared against.
>   - [`docs/develop/active/diagnosis/dreamer_offline_wm_imagination_test.md`](./dreamer_offline_wm_imagination_test.md) — the offline diagnostic that produced the reward-head finding and will be re-run on this checkpoint.

---

## §1 Context

The DreamerV3 conventional-fixes battery's Cell A1 — a NoPred run on the simplest grid task with the replay-ratio knob restored to the canonical Hafner-Atari setting — was profiled with an offline world-model imagination diagnostic. Encoder, decoder, and continuation head all passed; **the reward head failed in isolation**, with mean absolute error 0.39 at imagination horizon 5 against a pre-registered tolerance of 0.15 (a 2.6× exceedance). The diagnostic memo localised the bottleneck to the reward head specifically — the rest of the world model was healthy on this checkpoint.

A reverse-pass comparison against the sheeprl reference DreamerV3 implementation (added as §9 of the concept doc on 2026-05-10) surfaced four candidate deviations from paper-canonical recipe. One of those candidates — output-layer zero-initialisation of the reward and critic heads — is the most narrowly targeted at the reward-head failure: with non-zero output-layer initialisation, the reward head emits a non-uniform softmax distribution over the 255 two-hot bins from training step 0, predicting random nonzero rewards that the head must then *unlearn* before learning the true signal. Paper-canonical recipe (Hafner published code; sheeprl mirrors it) initialises the **final** Linear of the reward and critic heads to zero (kernel = 0, bias = 0), so the head emits a flat distribution at step 0 and the very first gradient under two-hot cross-entropy is the cleanest possible signal: `softmax(0) − target_twohot`.

This plan applies the zero-init fix to the reward and critic output layers, re-runs Cell A1 on the same simplest task with the same seed, and re-runs the offline diagnostic on the trained checkpoint to test whether the reward-head failure persists. Only this one knob is changed against the Cell A1 baseline; if the reward MAE drops below 0.15 the failure is attributed to output-layer init noise; if it does not, attention moves to the next candidate.

---

## §2 Implementation spec

### §2.1 Design

Two surfaces change:

1. **`src/models/dreamer_v3_nnx.py`** — the `MLP` class (the building block used by reward and critic heads, plus other heads we deliberately leave untouched) gains an optional `zero_init_output: bool` constructor argument. When `True`, the *final* Linear of the MLP is constructed with `kernel_init = nnx.initializers.zeros_init()` and `bias_init = nnx.initializers.zeros_init()` instead of the default `kernel_init=hafner_init()` (Linear bias defaults to zero already in NNX, but we set it explicitly for clarity and to mirror sheeprl's `uniform_init_weights(0.0)` semantics). All hidden layers retain `hafner_init`. Default is `False` so all existing call sites are bit-identical to today.

2. **`WorldModel.__init__` and `ActorCritic.__init__`** read a new mandatory config key `agent.zero_init_reward_critic` via `config.get_mandatory(...)` and pass `zero_init_output=<that-value>` to the reward-head, critic-head MLP constructors only. The continue head, decoder, encoder, actor head, and prior/posterior heads keep their current `hafner_init`-uniform behaviour. Sheeprl applies `uniform_init_weights(1.0)` ≈ our `hafner_init` to those layers, so the change is reward + critic only and matches sheeprl's selective override exactly.

### §2.2 File Changes

#### `src/models/dreamer_v3_nnx.py` — `MLP` class (currently lines 446–462)

```python
# BEFORE:
class MLP(nnx.Module):
    def __init__(self, input_dim, output_dim, hidden: list, rngs: nnx.Rngs):
        """
        Configurable MLP with LayerNorm + SiLU.
        """
        layers = []
        in_d = input_dim
        for h in hidden:
            layers.append(nnx.Linear(in_d, h, kernel_init=hafner_init(), rngs=rngs))
            layers.append(nnx.LayerNorm(h, rngs=rngs))
            layers.append(SiLU())
            in_d = h
        layers.append(nnx.Linear(in_d, output_dim, kernel_init=hafner_init(), rngs=rngs))
        self.net = nnx.Sequential(*layers)

    def __call__(self, x):
        return self.net(x)

# AFTER:
class MLP(nnx.Module):
    def __init__(self, input_dim, output_dim, hidden: list, rngs: nnx.Rngs,
                 zero_init_output: bool = False):
        """
        Configurable MLP with LayerNorm + SiLU.

        Args:
            zero_init_output: If True, the *final* Linear is constructed with
                zero kernel and zero bias (matches sheeprl
                `uniform_init_weights(0.0)` and Hafner published code's
                `outscale=0.0` override on reward + critic output heads).
                Default False = bit-identical to pre-knob behaviour.
        """
        layers = []
        in_d = input_dim
        for h in hidden:
            layers.append(nnx.Linear(in_d, h, kernel_init=hafner_init(), rngs=rngs))
            layers.append(nnx.LayerNorm(h, rngs=rngs))
            layers.append(SiLU())
            in_d = h
        if zero_init_output:
            layers.append(nnx.Linear(
                in_d, output_dim,
                kernel_init=nnx.initializers.zeros_init(),
                bias_init=nnx.initializers.zeros_init(),
                rngs=rngs,
            ))
        else:
            layers.append(nnx.Linear(in_d, output_dim, kernel_init=hafner_init(), rngs=rngs))
        self.net = nnx.Sequential(*layers)

    def __call__(self, x):
        return self.net(x)
```

#### `src/models/dreamer_v3_nnx.py` — `WorldModel.__init__` (the reward head construction site, currently around line 530)

Locate the line:

```python
self.reward_head = MLP(feat_dim, 255, reward_fc, rngs=rngs)
```

Just above it (immediately before the head construction; suggested location is right after `feat_dim = self.deter_dim + self.stoch_dim * self.discrete` at L523, OR co-located with the existing `decoder_fc / reward_fc / continue_fc` reads near L486–488), add:

```python
zero_init_rc = config.get('zero_init_reward_critic', False)
# Note: read with config.get(...) here because WorldModel and ActorCritic
# share the agent-level config dict but use_layer_norm-style mandatory
# reads are handled by the *caller* (DreamerV3Agent or trainer) to keep
# WorldModel constructable in unit tests with a partial config. The
# mandatory read is in the trainer / agent setup path — see §2.2 next
# bullet.
```

Actual mandatory read happens at the agent-config consumption site — see next bullet. Then change the head construction:

```python
# BEFORE:
self.reward_head = MLP(feat_dim, 255, reward_fc, rngs=rngs)
self.continue_head = MLP(feat_dim, 1, continue_fc, rngs=rngs)

# AFTER:
self.reward_head = MLP(feat_dim, 255, reward_fc, rngs=rngs,
                       zero_init_output=zero_init_rc)
self.continue_head = MLP(feat_dim, 1, continue_fc, rngs=rngs)
# continue_head intentionally NOT zero-init (sheeprl uses
# uniform_init_weights(1.0) on continue_model, matching our default
# hafner_init).
```

#### `src/models/dreamer_v3_nnx.py` — `ActorCritic.__init__` (currently around line 563–576)

Same pattern at the critic head:

```python
# BEFORE:
class ActorCritic(nnx.Module):
    def __init__(self, feat_dim, act_dim, config: dict, rngs: nnx.Rngs):
        ...
        actor_fc = config['actor_fc_layers']
        critic_fc = config['critic_fc_layers']

        self.actor = MLP(feat_dim, act_dim, actor_fc, rngs=rngs)
        self.critic = MLP(feat_dim, 255, critic_fc, rngs=rngs)

# AFTER:
class ActorCritic(nnx.Module):
    def __init__(self, feat_dim, act_dim, config: dict, rngs: nnx.Rngs):
        ...
        actor_fc = config['actor_fc_layers']
        critic_fc = config['critic_fc_layers']
        zero_init_rc = config.get('zero_init_reward_critic', False)

        self.actor = MLP(feat_dim, act_dim, actor_fc, rngs=rngs)
        # actor intentionally NOT zero-init (sheeprl uses
        # uniform_init_weights(1.0) on actor.mlp_heads, matching our default).
        self.critic = MLP(feat_dim, 255, critic_fc, rngs=rngs,
                          zero_init_output=zero_init_rc)
```

#### `src/models/dreamer_v3_trainer.py` — mandatory-config read at `DreamerV3Trainer.__init__`

The project's no-fallback-defaults rule requires `config.get_mandatory(...)`. The two `config.get(..., False)` reads in `nnx.py` above are pragmatic shims (NNX module construction sometimes runs in test contexts with partial configs); the **authoritative** mandatory read happens in the trainer setup path at the same place existing mandatory reads like `encoding_mode` happen (`trainer.py:78` per §3.2 of the concept doc). Add — at the same site as the other agent-config validation — a one-line:

```python
# Just after / alongside other config.get_mandatory(...) reads in DreamerV3Trainer.__init__
_ = config.get_mandatory('zero_init_reward_critic')
```

Purpose: the `config.get(..., False)` in `nnx.py` provides the value; this line ensures a missing YAML key raises `ValueError` at trainer construction time (training entry path). Effectively, `nnx.py` reads the value (using `.get` to stay constructable from tests), but the trainer enforces presence. The developer should pick the exact insertion line based on the existing `get_mandatory` block.

#### `configs/models/dreamer_v3.yaml` — add the knob

After line 43 (the existing `unimix: 0.01` line), add:

```yaml
  zero_init_reward_critic: true   # Output-layer zero-init for reward + critic heads (Hafner/sheeprl recipe).
                                   # See docs/develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md
                                   # and §6 item 27 of dreamer_v3_implementation.md.
                                   # When false, reward/critic output Linear uses hafner_init (pre-knob behaviour, bit-identical).
```

#### `configs/models/dreamer_v3_rr06.yaml` — add the knob (mirror)

After the matching `unimix: 0.01` line (currently L82), add the same block:

```yaml
  zero_init_reward_critic: true
```

(Brief comment OK — the canonical doc-link lives in `dreamer_v3.yaml`.)

### §2.3 Reversibility / bit-identity claim

When `zero_init_reward_critic: false`, the only difference from pre-knob code is the existence of the new constructor argument with default `False` — initialiser selection is identical to pre-knob (`hafner_init` on the final Linear). PRNG-key consumption: `nnx.Linear(... kernel_init=zeros_init(), bias_init=zeros_init(), rngs=rngs)` still passes `rngs` through but the zero-init does not consume the underlying PRNG keys (it ignores them and returns zeros directly). **This may or may not split the same number of keys as the truncated-normal init under NNX semantics — the developer should verify.** If PRNG consumption differs even when the default-`False` branch is taken, all downstream model parameters will get different RNG draws and the change is NOT bit-identical at `false`. The escape hatch: at `false` we explicitly route through the original code path (`else: layers.append(nnx.Linear(..., kernel_init=hafner_init(), rngs=rngs))`) — same call as today — so PRNG consumption matches today exactly. **At `true`**, PRNG consumption may differ (zeros initialiser may not draw a key); this is acceptable because that is the intended new behaviour and we are not asserting bit-identity for `true`.

### §2.4 Verification — smoke test

After the edits, the developer runs:

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -c "
import jax, jax.numpy as jnp
from flax import nnx
from src.models.dreamer_v3_nnx import MLP
rngs = nnx.Rngs(0)
# Without zero-init (default): final-layer kernel is non-zero.
m_off = MLP(input_dim=8, output_dim=255, hidden=[16, 16], rngs=rngs)
final_kernel_off = m_off.net.layers[-1].kernel.value
assert jnp.any(final_kernel_off != 0), 'default branch should NOT be zero-init'
# With zero-init: final-layer kernel is exactly zero.
rngs2 = nnx.Rngs(0)
m_on = MLP(input_dim=8, output_dim=255, hidden=[16, 16], rngs=rngs2,
           zero_init_output=True)
final_kernel_on = m_on.net.layers[-1].kernel.value
final_bias_on = m_on.net.layers[-1].bias.value
assert jnp.all(final_kernel_on == 0), f'zero-init kernel non-zero: {final_kernel_on}'
assert jnp.all(final_bias_on == 0), f'zero-init bias non-zero: {final_bias_on}'
# Output at init: should be exactly zero (255-dim zero vector → softmax = uniform).
y = m_on(jnp.ones((1, 8)))
assert jnp.all(y == 0), f'output at init should be exactly zero: {y}'
print('OK: MLP zero_init_output works as expected.')
"
```

The exact attribute path inside `nnx.Sequential` (`m.net.layers[-1].kernel.value` vs `m.net._layers[...]` etc.) may differ slightly across NNX versions — the developer should adapt the indexing to match the installed flax version, but the assertion semantics (final Linear kernel/bias both exactly zero; zero forward at zero-output-init) must hold.

### §2.5 Verification — config end-to-end

Confirm the trainer reads the new key without crashing on the training entry path. The developer should run a minimal `train.py --debug-no-launch`-style dry run if such a flag exists, OR construct `DreamerV3Trainer` directly with the test config to verify the `get_mandatory('zero_init_reward_critic')` call resolves. If the trainer cannot be constructed in isolation, the smoke test in §2.4 plus the YAML-file presence check is sufficient.

### §2.6 Out of scope for the code change

- Continue head, actor head, decoder, encoder, prior/posterior heads keep `hafner_init` on their output layers (sheeprl uses `uniform_init_weights(1.0)` ≈ `hafner_init` on those — matching today's behaviour).
- No changes to the offline diagnostic script.
- No changes to the trainer loss code, GRU cell, KL terms, or any other surface.
- No simultaneous fixes from candidates #1, #2, #3 in §9.11.

---

## §3 Launch manifest (single cell)

After the developer reports implementation complete, `training-runner` launches the following cell on node 113.

| Cell | Node:GPU | Task | Tag | WandB run | Seed | Env steps | Experiment config | Agent config |
|---|---|---|---|---|---|---|---|---|
| Z1 | n113:0 | NoPred (5×5) | `dreamer_zinit_NoPred_rr06_s0_n113` | `dreamer_zinit_NoPred_rr06_s0_n113` | 0 | 700,000 | `configs/experiment/basic/00-5X5_NoPred.yaml` | `configs/models/dreamer_v3_rr06.yaml` (now with `zero_init_reward_critic: true`) |

- **WandB group**: `dreamer_zero_init` (isolated from the existing `dreamer_conventional_fixes` group so the comparison-against-Cell-A1 framing is clean and the run does not accidentally average into that group's curves).
- **Direct comparison**: against Cell A1 baseline `czfnljf0` (NoPred + `replay_ratio: 0.0625` at survival 106; reward MAE 0.39 in offline diagnostic). Same task, same seed, same agent config except for the zero-init knob.
- **Expected wall-clock**: ~2.5 h on n113:0 per the cost model in [`docs/develop/active/diagnosis/dreamer_replay_ratio_sweep.md`](./dreamer_replay_ratio_sweep.md) (R=0.0625 → ~0.29 s/iter ≈ same envelope as the Cell A1 baseline).
- **Pre-flight (training-runner own protocol)**: ssh n113 + `python -c 'import jax'` + `pgrep -af` post-launch verification, per `.claude/agents/training-runner.md`.

---

## §4 Pre-registered confirmation/refutation criteria

The primary verdict comes from re-running the offline world-model imagination diagnostic on the trained Z1 checkpoint and comparing the reward MAE against the Cell A1 baseline. Survival is a secondary metric — Cell A1 reached survival 106 at the noisy-but-non-collapsing regime, and even a complete fix of the reward head may not lift survival immediately on this short budget; the offline diagnostic is the cleaner signal.

The diagnostic is the existing script `scripts/dreamer_offline_wm_test.py` invoked the same way as on the Cell A1 checkpoint. The reward MAE @ horizon 5 against the threshold of 0.15 is the pre-registered metric (per the offline-diagnostic plan in [`dreamer_offline_wm_imagination_test.md`](./dreamer_offline_wm_imagination_test.md) and the Cell A1 result memo).

| Hypothesis | Reward MAE @ h=5 | Interpretation | Next action |
|---|---|---|---|
| **H1 (zero-init fixes the reward head)** | < 0.15 (under threshold) | Output-layer init noise was the dominant cause; non-zero init forced the reward head to spend its early gradients unlearning random predictions before learning the true signal. | Promote `zero_init_reward_critic: true` to default in both `dreamer_v3.yaml` and `dreamer_v3_rr06.yaml` (already proposed default in §2.2; this confirms it). Close the reward-head investigation thread. Document the result in §6 item 27 + the failure-mode memo. |
| **H2 (zero-init partially helps)** | 0.15 ≤ MAE < 0.30 | Init was contributing materially but is not the sole cause; another upstream signal (most likely candidate #1 GRU reset-gate; possibly #4's interaction with bin-range item 2) is also degrading the head. | Queue candidate #1 (GRU reset-gate fix) as the next plan. Keep zero-init knob ON as a cumulative fix. |
| **H0 (zero-init doesn't help)** | ≥ 0.30 (still ≥ 2× threshold) | Output-layer init is not the bottleneck. The failure is elsewhere — most likely candidate #1 (GRU reset gate, structural) or §6 item 2 (two-hot bin range; the practical effect was claimed benign for our reward magnitudes but may interact with the head's learning dynamics in ways item 2's note didn't anticipate). | Queue candidate #1 as the next plan. Re-examine §6 item 2 (twohot bin range) interaction with the reward head — possibly a follow-up plan. Do NOT promote zero-init (set `zero_init_reward_critic: false` in default; keep the knob in code as a documented option). |

The offline diagnostic re-run is **not part of this plan's spawn chain**. After Z1 finishes, the user (or the parent agent) runs the diagnostic via `experiment-analyzer`, which then writes the verdict back to this plan and to the failure-mode memo.

---

## §5 Out of scope

- **Other §9.11 candidates** (#1 GRU reset, #2 prior/posterior heads, #3 critic self-EMA) — deferred per user instruction. Do not bundle.
- **Changes to the offline diagnostic script** — re-using the same script with no edits is a deliberate methodological choice (the diagnostic is the measuring instrument; changing it would invalidate the comparison against `czfnljf0`).
- **Changes to any other config knob simultaneously** — the entire scientific value of the comparison hinges on this being a single-knob change relative to Cell A1. Even cosmetically harmless edits to other knobs in the same config file should be avoided.
- **Follow-up plans queueing** — until Z1's offline diagnostic verdict comes back, do not queue plans for candidates #1/#2/#3.
- **Cell A2 (predator) re-run** — Cell A2 had a different failure mode (~30-step floor with mae_pos/mae_neg asymmetry), and there is no offline-diagnostic localization on its checkpoint yet. Out of scope for this plan; if zero-init is confirmed effective on Cell A1, a separate plan can extend to A2.

---

## §6 Hand-off

After this plan lands and is committed:

1. **`developer`** reads §2 in full, applies the changes to the four files (`dreamer_v3_nnx.py`, `dreamer_v3_trainer.py`, `dreamer_v3.yaml`, `dreamer_v3_rr06.yaml`), runs the §2.4 smoke test, runs the §2.5 config end-to-end check, fills out the Implementation Report below, and reports back.
2. **`senior-developer`** (this agent) verifies the implementation against §2 per the standard verification protocol.
3. **`training-runner`** reads §3 verbatim and launches Z1 on n113:0. Standard pre-flight + post-launch pgrep + diary update.
4. **After Z1 finishes (~2.5 h)** the user or parent agent runs `scripts/dreamer_offline_wm_test.py` on the Z1 checkpoint and reads the reward MAE @ h=5 against §4's pre-registered table. The verdict is written back into §4 of this doc by `experiment-analyzer` (or the senior-developer if the analyzer is not invoked).
5. **Closing actions** (per §4 verdict): promote default, queue next candidate, or revisit §6 item 2.

---

## Checkpoints (for `developer`)

- [x] `MLP.__init__` accepts `zero_init_output: bool = False` and routes the final Linear through `nnx.initializers.zeros_init()` when True. (nnx.py lines 447–478)
- [x] `WorldModel.__init__` reads `agent.zero_init_reward_critic` (via `config.get(...)` in NNX file; the mandatory enforcement is added separately) and passes through to the reward head only. (nnx.py lines 546–552)
- [x] `ActorCritic.__init__` does the same for the critic head only (NOT the actor). (nnx.py lines 595–601)
- [x] `DreamerV3Trainer.__init__` calls `config.get_mandatory('agent.zero_init_reward_critic', bool)` and inserts the value into `agent_config` dict so it flows to constructors; missing YAML key raises `ValueError`. (trainer.py line 76)
- [x] `configs/models/dreamer_v3.yaml` and `configs/models/dreamer_v3_rr06.yaml` both contain `zero_init_reward_critic: true`.
- [x] §2.4 smoke test passes (assertions all hold; output of `MLP(... zero_init_output=True)(x)` at init is exactly zero).
- [x] When `zero_init_reward_critic: false`, the model construction takes the original code path verbatim (no PRNG-consumption divergence; bit-identical to pre-knob behaviour).
- [x] No edits made to the GRU cell (`LayerNormGRUCell`, `nnx.py:18–39`), the prior/posterior head construction (`nnx.py:51, 58–59`), or the critic loss (`trainer.py:421–430`) — those are the other three §9.11 candidates and are explicitly out of scope.

---

## Implementation Report

> **Implemented by**: developer (claude-sonnet-4-6)
> **Date**: 2026-05-10

### Files changed

| File | Change |
|---|---|
| `src/models/dreamer_v3_nnx.py` | `MLP.__init__` gains `zero_init_output: bool = False`; `WorldModel.__init__` reads `config.get('zero_init_reward_critic', False)` and passes to reward head (not continue head); `ActorCritic.__init__` reads same key and passes to critic head (not actor head) |
| `src/models/dreamer_v3_trainer.py` | Added `'zero_init_reward_critic': config.get_mandatory('agent.zero_init_reward_critic', bool)` to `agent_config` dict at line 76; mandatory read enforces YAML key presence; value flows to `DreamerV3Agent` and `target_critic` constructors via the same dict |
| `configs/models/dreamer_v3.yaml` | Added `zero_init_reward_critic: true` (with full doc-link comment) after `unimix: 0.01` |
| `configs/models/dreamer_v3_rr06.yaml` | Added `zero_init_reward_critic: true` (brief comment, canonical link in dreamer_v3.yaml) after `unimix: 0.01` |

### Deviations from §2

One intentional deviation: the plan specified the mandatory read in the trainer as a standalone sentinel line `_ = config.get_mandatory('zero_init_reward_critic')` and a separate `config.get(...)` in nnx.py to supply the actual value. Instead, the mandatory read in `trainer.py` was placed inside the `agent_config` dict construction, directly supplying the value as `'zero_init_reward_critic': config.get_mandatory(...)`. This is strictly superior: it eliminates the split read, ensures `agent_config` carries the value, and means both `DreamerV3Agent` and `target_critic` (`ActorCritic`) see the correct bool — which is required anyway for the target critic to also use zero-init. The plan's note about `config.get(..., False)` in nnx.py as a "pragmatic shim" is preserved (the `WorldModel` and `ActorCritic` still use `.get` so they are constructable from partial configs in unit tests), but the canonical value source is the mandatory read in the trainer.

### Smoke test output (§2.4) — verbatim

```
$ /home/vncuser/miniconda3/envs/grid_world_pain/bin/python -c "
import sys; sys.path.insert(0, '.')
import jax.numpy as jnp
from flax import nnx
from src.models.dreamer_v3_nnx import MLP
rngs = nnx.Rngs(0)
m_off = MLP(input_dim=8, output_dim=255, hidden=[16, 16], rngs=rngs)
final_kernel_off = m_off.net.layers[-1].kernel.value   # DeprecationWarning (harmless)
assert jnp.any(final_kernel_off != 0), 'default branch should NOT be zero-init'
rngs2 = nnx.Rngs(0)
m_on = MLP(input_dim=8, output_dim=255, hidden=[16, 16], rngs=rngs2, zero_init_output=True)
final_kernel_on = m_on.net.layers[-1].kernel.value
final_bias_on = m_on.net.layers[-1].bias.value
assert jnp.all(final_kernel_on == 0), f'zero-init kernel non-zero: {final_kernel_on}'
assert jnp.all(final_bias_on == 0), f'zero-init bias non-zero: {final_bias_on}'
y = m_on(jnp.ones((1, 8)))
assert jnp.all(y == 0), f'output at init should be exactly zero: {y}'
print('OK: MLP zero_init_output works as expected.')
"

layers in off-MLP: 7
final_kernel_off any non-zero: True
final_kernel_on all zero: True
final_bias_on all zero: True
output at init all zero: True
output shape: (1, 255)
OK: MLP zero_init_output works as expected.
```

Note: `nnx.Sequential.layers[-1]` (index 6 for a 2-hidden-layer MLP with 3 sub-layers each = 6 items + 1 output Linear) correctly identifies the final Linear. The plan noted the exact attribute path might need adapting — confirmed `.layers` (not `._layers`).

### Config end-to-end check (§2.5)

```
dreamer_v3_rr06.yaml -> agent.zero_init_reward_critic = True  (type: bool)
dreamer_v3.yaml -> agent.zero_init_reward_critic = True  (type: bool)
OK: both YAML files expose the mandatory key correctly.
```

Missing key also confirmed to raise `ValueError`:
```
OK: missing key raises ValueError: Strict Config: Configuration key 'agent.zero_init_reward_critic' is required but missing.
```

### Speed check

Not applicable — this is an init-time-only change. The hot path (env step, forward/backward, vmap/jit boundaries) is unaffected; `zeros_init()` only runs at model construction, not during training iterations.

### Blockers

None.

Implemented by: developer

---

## Verification Report

> **Verified by**:
> **Date**:

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/models/dreamer_v3_nnx.py` | `MLP.__init__` gains `zero_init_output: bool = False`; reward head + critic head pass `zero_init_output=zero_init_rc` | | |
| `src/models/dreamer_v3_trainer.py` | `config.get_mandatory('zero_init_reward_critic')` added at trainer construction | | |
| `configs/models/dreamer_v3.yaml` | `zero_init_reward_critic: true` added | | |
| `configs/models/dreamer_v3_rr06.yaml` | `zero_init_reward_critic: true` added | | |

**Conclusion**:

<!-- senior-developer fills in: ✅/⚠️/❌ per row, plus a one-line summary. -->

---

<!--
NEW ISSUES: any deviation discovered during implementation/verification that warrants its own plan should be split out (one of #1, #2, #3 candidates from §9.11) rather than expanded inline here.
-->
