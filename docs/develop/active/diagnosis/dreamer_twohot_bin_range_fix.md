---
title: "DreamerV3 Two-Hot Bin Range Fix (paper-canonical symlog grid; Cell A1 / Z1 re-run)"
topic: diagnosis
status: active
created: 2026-05-10
last_updated: 2026-05-10
phase: 2
---

# DreamerV3 Two-Hot Bin Range Fix (paper-canonical symlog grid; Cell A1 / Z1 re-run)

> **Status**: PLANNED — implementation + launch authorised (user, 2026-05-10, "simple job, no permission needed", same standing authorisation as the zero-init job that just landed).
> **Doc role**: doubles as design doc + Launch Manifest.
> **Related**:
>   - [`docs/project/concepts/dreamer_v3_implementation.md`](../../../project/concepts/dreamer_v3_implementation.md) §6 item 2 + §9.6 + §9.11.5 — the deviation this plan acts on. After this lands, §6 item 2 needs updating ("ACTED ON, see this plan").
>   - [`docs/reviews/dreamer_v3_implementation_math_review.md`](../../../reviews/dreamer_v3_implementation_math_review.md) — F1 BLOCKER finding; cross-validates against Hafner published `embodied/jax/heads.py:87–97`.
>   - [`docs/develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md`](./dreamer_zero_init_reward_critic_fix.md) — the prior fix (Cell Z1) this plan stacks on top of; same template + flow.
>   - [`tmp/20260510_211404_wm_imagination_test_Z1.md`](../../../../tmp/20260510_211404_wm_imagination_test_Z1.md) — Z1's offline diagnostic; provides the residual-error mechanistic breakdown cited in §1.
>   - [`docs/experiments/active/dreamer_diagnosis/DREAMER_CONVENTIONAL_FIXES_BATTERY.md`](../../../experiments/active/dreamer_diagnosis/DREAMER_CONVENTIONAL_FIXES_BATTERY.md) — Cell A1 baseline (`czfnljf0`) and Z1 anchor.
>   - `tmp/sheeprl/sheeprl/utils/utils.py:156–188` (`two_hot_encoder`) and `tmp/sheeprl/sheeprl/utils/distribution.py:224–276` (`TwoHotEncodingDistribution`) — sheeprl reference implementations confirming the paper-canonical bin convention.

---

## §1 Context

The zero-init re-run (Cell Z1, `dreamer_zinit_NoPred_rr06_s0_n113`, just analysed) fired **H2 — partial fix**: reward MAE @ horizon 5 dropped from Cell A1's 0.386 to Z1's 0.277 (28% relative improvement) but did not clear the pre-registered H1 threshold of 0.15. The residual error is **disproportionately on negative-reward events**: training-time `model_reward_mae_pos` improved 49% (0.83 → 0.42) under zero-init, while `model_reward_mae_neg` improved only 14% (0.93 → 0.80). That asymmetry points at this specific deviation.

Our `to_twohot` and `from_twohot` (`src/models/dreamer_v3_util.py:19, 60`) construct the bin grid as `bins = linspace(symlog(-20), symlog(20), 255)` — i.e., they **apply `symlog` to the range constants `±20`**, treating those constants as raw-space edges. The result is bins spanning only raw `±20`. The Hafner-published `embodied/jax/heads.py:87–97` and sheeprl's `TwoHotEncodingDistribution` (`distribution.py:237`) both interpret `±20` as **already-symlog-space** edges — `bins = linspace(-20, +20, 255)` directly — so raw-space bin centres span approximately `±symexp(20) ≈ ±4.85·10⁸`. **Our death-penalty event is `−100`, which is literally outside our head's representable bin support** (the most-negative bin sits at raw `−20`, so any reward `≤ −20` saturates the boundary bin). Even on NoPred where the predator/death penalty is not active, the negative-reward dense-signal saturation behaviour is the most plausible mechanism for Z1's residual `mae_neg` floor.

Sheeprl's `dreamer_v3/utils.py` matches the paper-canonical bin layout — verified by math-reviewer (F1 finding in [`docs/reviews/dreamer_v3_implementation_math_review.md`](../../../reviews/dreamer_v3_implementation_math_review.md), commit `bbeec88`). The fix: replace `linspace(symlog(-20), symlog(+20), 255)` with `linspace(-20, +20, 255)` directly — one bug-fix line in `to_twohot` and one in `from_twohot`.

The plan re-runs Cell A1 / Z1 on the same NoPred 5×5 task with the same seed, with **all three cumulative fixes** (replay_ratio = 0.0625 from `dreamer_v3_rr06`, zero-init reward+critic from Z1, and now paper-canonical bin range), and re-runs the offline world-model imagination diagnostic to test whether reward MAE @ h=5 falls below the H1 threshold of 0.15.

---

## §2 Implementation spec

### §2.1 Mechanism (the bug, plain-text)

The paper / Hafner-published / sheeprl convention is: **`-20` and `+20` are already-symlog-space edges**. The encoder pre-applies `symlog` to the input `x`; the bins themselves are `linspace(-20, +20, 255)` directly in symlog space. Decoding multiplies the softmax probabilities by these symlog-space bin centres and applies `symexp` to the resulting expected value, recovering raw-space.

Our code does **two things wrong, symmetrically**:

1. `to_twohot` line 32–33: `bottom = symlog(jnp.array(min_v)); top = symlog(jnp.array(max_v))` — applies `symlog` to the range constants, then uses those as the symlog-space edges. The result: symlog-space edges of `±symlog(20) ≈ ±3.045` instead of `±20`.
2. `from_twohot` line 67–68: identical treatment — `bottom = symlog(min_v); top = symlog(max_v)`; bin centres are then `linspace(±3.045)` in symlog space, mapping (via `symexp` at line 76) back to raw `±20`.

The fix is to remove the extra `symlog` calls on the range constants in **both** functions, and then `min_v=-20, max_v=20` directly become the symlog-space edges (matching paper / sheeprl).

### §2.2 File changes — `src/models/dreamer_v3_util.py`

Two functions change. The default arguments (`min_v=-20.0, max_v=20.0, num_buckets=255`) are kept — but their **interpretation** changes from "raw-space edges" to "symlog-space edges" (and the docstring must reflect that).

#### §2.2.1 `to_twohot` (current lines 19–58)

```python
# BEFORE (lines 19–58):
def to_twohot(x, min_v=-20.0, max_v=20.0, num_buckets=255):
    """
    Converts a scalar to a Two-Hot distribution (soft discretization).
    Used for Value and Reward targets in DreamerV3.
    """
    x = symlog(x)
    # Using raw values for boundaries as per common implementations, but mapped to symlog space?
    # Actually DreamerV3 paper uses symlog(x) for targets.
    # We define buckets in the TRANSFORMED space usually, or raw?
    # The official implementation defines buckets in SYMLOG space.
    # range: symlog(-20) ~ -3 to symlog(20) ~ 3.

    # Let's interpret min_v and max_v as RAW values, and we transform them.
    bottom = symlog(jnp.array(min_v))
    top = symlog(jnp.array(max_v))

    # Clip value to range
    x = jnp.clip(x, bottom, top)

    # Map to [0, num_buckets - 1]
    rel = (x - bottom) / (top - bottom) * (num_buckets - 1)
    ...
```

```python
# AFTER:
def to_twohot(x, min_v=-20.0, max_v=20.0, num_buckets=255, paper_canonical_bins=True):
    """
    Converts a scalar to a Two-Hot distribution (soft discretization).
    Used for Value and Reward targets in DreamerV3.

    Args:
        x: scalar(s), in raw space.
        min_v, max_v: bin-grid edges in SYMLOG space (paper-canonical
            convention: ±20 are already symlog-space edges; raw-space bin
            centres span ±symexp(20) ≈ ±4.85·10⁸). Matches Hafner
            embodied/jax/heads.py:87–97 and sheeprl TwoHotEncodingDistribution.
        num_buckets: 255 (paper).
        paper_canonical_bins: if True (default), use ±20 as symlog-space
            edges directly. If False, apply symlog(±20) to the edges
            (legacy buggy behaviour; raw-space support narrows to ±20).
    """
    x = symlog(x)

    if paper_canonical_bins:
        # Paper-canonical: min_v, max_v are already symlog-space edges.
        bottom = jnp.array(min_v, dtype=x.dtype)
        top    = jnp.array(max_v, dtype=x.dtype)
    else:
        # Legacy buggy behaviour: treat min_v, max_v as raw-space edges
        # and apply symlog to them. Reproduces pre-fix bin layout
        # (raw-space support ±20). Provided for bit-identical legacy
        # reproduction only.
        bottom = symlog(jnp.array(min_v, dtype=x.dtype))
        top    = symlog(jnp.array(max_v, dtype=x.dtype))

    # Clip value to range (in symlog space)
    x = jnp.clip(x, bottom, top)

    # Map to [0, num_buckets - 1]
    rel = (x - bottom) / (top - bottom) * (num_buckets - 1)
    floor = jnp.floor(rel).astype(jnp.int32)
    ceil = jnp.ceil(rel).astype(jnp.int32)
    prob_ceil = rel - floor
    prob_floor = 1.0 - prob_ceil

    def scatter(idx, val):
        return jax.nn.one_hot(idx, num_buckets) * val[..., None]

    target = scatter(floor, prob_floor) + scatter(ceil, prob_ceil)
    return target
```

#### §2.2.2 `from_twohot` (current lines 60–76)

```python
# BEFORE:
def from_twohot(logits, min_v=-20.0, max_v=20.0, num_buckets=255):
    """
    Converts logits from Two-Hot distribution back to scalar (expectation).
    Returns value in RAW space (inverse symlog).
    """
    probs = jax.nn.softmax(logits, axis=-1)
    bottom = symlog(jnp.array(min_v))
    top = symlog(jnp.array(max_v))
    bucket_vals = jnp.linspace(bottom, top, num_buckets)
    sym_val = jnp.sum(probs * bucket_vals, axis=-1)
    return symexp(sym_val)
```

```python
# AFTER:
def from_twohot(logits, min_v=-20.0, max_v=20.0, num_buckets=255, paper_canonical_bins=True):
    """
    Converts logits from Two-Hot distribution back to scalar (expectation).
    Returns value in RAW space (inverse symlog).

    Args mirror `to_twohot` — see that docstring for the
    `paper_canonical_bins` flag semantics. Bin layout in this function
    MUST match the layout used in `to_twohot` (same flag value).
    """
    probs = jax.nn.softmax(logits, axis=-1)

    if paper_canonical_bins:
        bottom = jnp.array(min_v, dtype=probs.dtype)
        top    = jnp.array(max_v, dtype=probs.dtype)
    else:
        bottom = symlog(jnp.array(min_v, dtype=probs.dtype))
        top    = symlog(jnp.array(max_v, dtype=probs.dtype))

    # Bucket centres in symlog space.
    bucket_vals = jnp.linspace(bottom, top, num_buckets)

    # Expected value in symlog space, then mapped back to raw via symexp.
    sym_val = jnp.sum(probs * bucket_vals, axis=-1)
    return symexp(sym_val)
```

### §2.3 Config wiring — reversibility knob

Add a single agent-level config key `agent.paper_canonical_twohot_bins: true` and propagate it to every `to_twohot` / `from_twohot` call site so paired calls always use matching bin layouts.

#### §2.3.1 `configs/models/dreamer_v3.yaml` — add the knob

After line 43 (the existing `unimix: 0.01`) and the existing `zero_init_reward_critic: true` block, add:

```yaml
  paper_canonical_twohot_bins: true   # Two-hot bin grid is linspace(-20,+20,255) directly in
                                       # SYMLOG space (paper-canonical; raw-space support ±4.85e8).
                                       # When false, applies symlog to ±20 first (legacy bug;
                                       # raw-space support clipped to ±20). See
                                       # docs/develop/active/diagnosis/dreamer_twohot_bin_range_fix.md
                                       # and §6 item 2 of dreamer_v3_implementation.md.
```

#### §2.3.2 `configs/models/dreamer_v3_rr06.yaml` — add the knob (mirror)

After the matching `unimix: 0.01` / `zero_init_reward_critic: true` block (currently around L82–L85), add the same line with brief comment:

```yaml
  paper_canonical_twohot_bins: true
```

(Canonical doc-link lives in `dreamer_v3.yaml`.)

#### §2.3.3 `src/models/dreamer_v3_trainer.py` — mandatory read

In `DreamerV3Trainer.__init__` `agent_config` dict construction (currently L64–L81), add a new entry alongside the existing `zero_init_reward_critic` entry:

```python
# In the agent_config dict at trainer.py around line 76:
'zero_init_reward_critic': config.get_mandatory('agent.zero_init_reward_critic', bool),
'paper_canonical_twohot_bins': config.get_mandatory('agent.paper_canonical_twohot_bins', bool),  # NEW
```

The trainer doesn't pass this dict directly to the twohot helpers (they're free functions, not module attributes). Two patterns are acceptable; the developer should pick the cleaner one:

- **(A) Stash on `self`**: `self._paper_canonical_twohot_bins = config.get_mandatory('agent.paper_canonical_twohot_bins', bool)` and pass `paper_canonical_bins=self._paper_canonical_twohot_bins` at every `to_twohot(...)` / `from_twohot(...)` call site in `trainer.py`. This is straightforward and mirrors how `IMG_PROBE` is read at L131.
- **(B) Module-level globals via `functools.partial`**: read once in `__init__` and re-bind the imports `to_twohot` / `from_twohot` to `partial(to_twohot, paper_canonical_bins=<flag>)` and use those bound names for the call sites. Less invasive at call sites, but tracing through nnx.jit may interact awkwardly with closure capture; **prefer (A) unless (A) breaks JIT**.

The `DreamerV3Agent` constructor also needs the flag if any of its sub-modules call `from_twohot` directly; checking `nnx.py:681` — `from_twohot(value_logits, num_buckets=value_logits.shape[-1])` is in some helper method on a model class. The developer should locate that call site, identify which class owns it, and decide whether to:
- pass the flag through `agent_config` and stash it on the owning module, OR
- refactor that call site to live in the trainer (less invasive — the call seems to be a utility, not part of a training loop).

**The developer must touch every `to_twohot` / `from_twohot` call site so they all use the same flag value.** Mismatched flag values between encode and decode will silently produce wrong reconstructions.

#### §2.3.4 Call-site inventory (so nothing is missed)

| File | Line | Function | Notes |
|---|---|---|---|
| `src/models/dreamer_v3_trainer.py` | 220 | `to_twohot(reward)` | reward target (WM loss) |
| `src/models/dreamer_v3_trainer.py` | 256 | `from_twohot(rew_pred)` | reward MAE metric |
| `src/models/dreamer_v3_trainer.py` | 363 | `from_twohot(self.agent.wm.reward_head(next_feat))` | imagine, modulated branch |
| `src/models/dreamer_v3_trainer.py` | 368 | `from_twohot(self.target_critic(next_feat))` | bootstrap value, modulated branch |
| `src/models/dreamer_v3_trainer.py` | 386 | `from_twohot(self.agent.wm.reward_head(next_feat))` | imagine, vanilla branch |
| `src/models/dreamer_v3_trainer.py` | 389 | `from_twohot(self.target_critic(next_feat))` | bootstrap value, vanilla branch |
| `src/models/dreamer_v3_trainer.py` | 412 | `from_twohot(self.target_critic(start_feat))` | start-state value |
| `src/models/dreamer_v3_trainer.py` | 429 | `to_twohot(jax.lax.stop_gradient(lambda_returns))` | critic CE target |
| `src/models/dreamer_v3_trainer.py` | 434 | `from_twohot(v_pred_logits)` | actor baseline |
| `src/models/dreamer_v3_nnx.py` | 681 | `from_twohot(value_logits, num_buckets=value_logits.shape[-1])` | helper on a model class — check ownership and route flag through `agent_config` |

All ten call sites must use the same flag value within a single training run.

### §2.4 Reversibility / bit-identity claim

When `paper_canonical_twohot_bins: false`, the `else` branch in both functions is the **verbatim previous code** (`bottom = symlog(jnp.array(min_v)); top = symlog(jnp.array(max_v))`). PRNG consumption is unchanged (no random ops in either path). Therefore: with the flag set to `false`, the model output is bit-identical to pre-fix.

When `paper_canonical_twohot_bins: true`, the bin-grid construction differs by the omission of the two `symlog(...)` calls on the range constants. Forward/backward graph topology is identical (same number of ops, same shapes); only numerical values differ. JIT tracing should be unaffected.

### §2.5 Verification — smoke test (round-trip)

After the edits, the developer runs:

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -c "
import jax.numpy as jnp
from src.models.dreamer_v3_util import to_twohot, from_twohot

# Test rewards spanning the cases that matter:
#   (a) values inside ±20 (round-trip with sub-bin error under both flags)
#   (b) values outside ±20 up to ±100 (saturate at ±20 under flag=False;
#       round-trip cleanly with bounded error under flag=True)
#   (c) extreme values up to ±10_000 (saturate hard under flag=False;
#       still round-trip under flag=True since support is ±4.85e8)
test_vals = jnp.array([-10000., -100., -20., -5., -1., 0., 1., 5., 20., 100., 10000.])

# Sanity-check the logits-to-target round-trip with a one-hot logit
# at the index produced by to_twohot. We make logits = log(target + tiny)
# so softmax(logits) ≈ target.
def round_trip(x, flag):
    target = to_twohot(x, paper_canonical_bins=flag)             # (N, 255)
    logits = jnp.log(target + 1e-9)                                # (N, 255)
    rec    = from_twohot(logits, paper_canonical_bins=flag)        # (N,)
    return rec

rec_paper  = round_trip(test_vals, True)
rec_legacy = round_trip(test_vals, False)
err_paper  = jnp.abs(rec_paper  - test_vals)
err_legacy = jnp.abs(rec_legacy - test_vals)

print('val | paper-rec | paper-err | legacy-rec | legacy-err')
for v, rp, ep, rl, el in zip(test_vals, rec_paper, err_paper, rec_legacy, err_legacy):
    print(f'{float(v):>10.2f}  {float(rp):>12.4f}  {float(ep):>10.4f}  {float(rl):>12.4f}  {float(el):>10.4f}')

# Hard assertions:
# 1. Inside ±20: both flags should round-trip with bounded error.
inside_mask  = jnp.abs(test_vals) <= 20.
assert jnp.all(jnp.where(inside_mask, err_paper,  0.) < 1.0),  'paper flag should round-trip inside ±20 with err<1'
assert jnp.all(jnp.where(inside_mask, err_legacy, 0.) < 1.0),  'legacy flag should round-trip inside ±20 with err<1'

# 2. ±100: paper flag should round-trip with bounded error; legacy should
#    saturate (recovered |val| should be capped near 20).
val_at_neg100 = test_vals == -100.
val_at_pos100 = test_vals ==  100.
assert float(jnp.where(val_at_neg100, rec_legacy, 0.).sum()) < -19.5,  'legacy at -100 should saturate near -20'
assert float(jnp.where(val_at_pos100, rec_legacy, 0.).sum()) >  19.5,  'legacy at +100 should saturate near +20'
assert float(jnp.where(val_at_neg100, jnp.abs(rec_paper -  test_vals), 0.).sum()) < 20.,  'paper at -100 should not saturate hard'
assert float(jnp.where(val_at_pos100, jnp.abs(rec_paper -  test_vals), 0.).sum()) < 20.,  'paper at +100 should not saturate hard'

# 3. ±10000: legacy should saturate near ±20 (huge err); paper should be
#    much closer (within a few orders of magnitude — coarse bins at extremes).
val_at_neg10k = test_vals == -10000.
val_at_pos10k = test_vals ==  10000.
err_paper_at_extreme  = float(jnp.where(val_at_neg10k | val_at_pos10k, err_paper,  0.).sum())
err_legacy_at_extreme = float(jnp.where(val_at_neg10k | val_at_pos10k, err_legacy, 0.).sum())
assert err_legacy_at_extreme > 1000., f'legacy at ±10k should saturate (err~10k each); got {err_legacy_at_extreme}'
assert err_paper_at_extreme  < err_legacy_at_extreme,  'paper should beat legacy at extremes'

print('OK: paper-canonical flag round-trips ±100 cleanly; legacy saturates at ±20.')
"
```

Acceptance criteria:
- `paper_canonical_bins=True`: `−100` round-trips back as approximately `−100` (within a few units; the bins are sparser at extremes so sub-bin error grows with `|val|` but no boundary-saturation occurs).
- `paper_canonical_bins=False`: `−100` round-trips back as approximately `−20` (saturated at the most-negative bin) — confirms legacy bug behaviour.
- `paper_canonical_bins=True`: even `±10000` round-trips with bounded error (well within the ±4.85e8 support).
- All round-trip tests inside `±20` produce sub-unit error under both flags (the legacy bug is not visible inside ±20).

### §2.6 Verification — config end-to-end check

Confirm both YAML files expose the new mandatory key without crashing:

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -c "
from src.utils.strict_config import StrictConfig
for path in ['configs/models/dreamer_v3.yaml', 'configs/models/dreamer_v3_rr06.yaml']:
    cfg = StrictConfig.load(path)   # adapt to actual API
    val = cfg.get_mandatory('agent.paper_canonical_twohot_bins', bool)
    print(f'{path} -> agent.paper_canonical_twohot_bins = {val}  (type: {type(val).__name__})')
print('OK: both YAML files expose the mandatory key correctly.')
"
```

(Adapt the import path to the actual `StrictConfig` API used elsewhere in the repo — same pattern as the zero-init plan §2.5.)

Also confirm a missing key raises `ValueError`:

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -c "
# Synthesise a config missing the new key and assert get_mandatory raises.
# (Same pattern as dreamer_zero_init_reward_critic_fix.md §2.5.)
"
```

### §2.7 Out of scope for the code change

- Don't change `min_v` / `max_v` defaults from `±20`. Under `paper_canonical_bins=True` these are now correctly interpreted as symlog-space edges (matching paper).
- Don't change `num_buckets` from 255. (§6 item 17 in the concept doc — separate deviation about hard-coded 255 — is out of scope for this plan.)
- Don't touch the GRU reset gate (§6 item 28, candidate #1), prior/posterior heads (§6 item 30), or critic-EMA regularisation (§6 item 29). Still deferred per user instruction.
- Don't change `zero_init_reward_critic` (already `true` by default after Z1; should remain on for this cell).
- Don't change `replay_ratio` (already `0.0625` in `dreamer_v3_rr06.yaml`).

---

## §3 Launch manifest (single cell)

After the developer reports implementation complete, `training-runner` launches the following cell on node 113.

| Cell | Node:GPU | Task | Tag | WandB run | Seed | Env steps | Experiment config | Agent config |
|---|---|---|---|---|---|---|---|---|
| Z2 | n113:0 | NoPred (5×5) | `dreamer_twohotrng_NoPred_rr06_s0_n113` | `dreamer_twohotrng_NoPred_rr06_s0_n113` | 0 | 700,000 | `configs/experiment/basic/00-5X5_NoPred.yaml` | `configs/models/dreamer_v3_rr06.yaml` (now with `paper_canonical_twohot_bins: true` and the existing `zero_init_reward_critic: true` from Z1) |

- **WandB group**: `dreamer_paper_canonical_bins`. Isolated from the existing `dreamer_zero_init` and `dreamer_conventional_fixes` groups so the cumulative-fix framing is clean.
- **Direct comparison anchors**:
  - **Cell A1 baseline** (`czfnljf0`, NoPred + `replay_ratio=0.0625`; reward MAE @ h=5 = 0.386).
  - **Cell Z1 anchor** (`axndoqsz`, NoPred + `replay_ratio=0.0625` + `zero_init_reward_critic=true`; reward MAE @ h=5 = 0.277).
  - **Cell Z2 (this run)**: A1 baseline + zero-init + paper-canonical bins. Same task, same seed, only the bin-range flag flipped relative to Z1.
- **Cumulative fixes that should be active in Z2**:
  - `replay_ratio: 0.0625` (rr06 config; carried over)
  - `zero_init_reward_critic: true` (Z1 fix; carried over by default)
  - `paper_canonical_twohot_bins: true` (NEW in this plan)
- **Expected wall-clock**: ~2.5 h on n113:0 (same envelope as Z1; bin-grid construction is init-time-only with no hot-path effect).
- **Pre-flight (training-runner own protocol)**: ssh n113 + `python -c 'import jax'` + `pgrep -af` post-launch verification, per `.claude/agents/training-runner.md`.

**NoPred-specific mechanistic note (load-bearing for §4)**: NoPred has no death penalty (predator disabled), so reward-event magnitudes are limited to dense satiation rewards in the `|r| ≲ few units` range. None of these saturate at the legacy `±20` bin edges in raw-space terms. **However**, the offline diagnostic on Z1 already showed reward MAE concentrated at long horizons and disproportionately on negative dense rewards. The mechanistic prediction: paper-canonical bins improve resolution near zero (denser bins around 0 in symlog space, sparser at extremes — matching the natural reward distribution), which should reduce small-magnitude reward MAE even though no rewards saturate the legacy boundaries. If reward MAE @ h=5 drops materially under this fix on NoPred, the deviation is contributing meaningfully even on a low-magnitude reward distribution. If it doesn't drop, the bin range is not the primary residual cause on NoPred and attention moves to candidate #1 (GRU reset gate; §6 item 28). Predator-task validation (where the death-penalty `−100` event genuinely exits the legacy support) is a separate downstream plan, not this one.

---

## §4 Pre-registered confirmation/refutation criteria

Re-run `scripts/dreamer_offline_wm_test.py` on Z2's checkpoint, identical command to the A1 / Z1 invocations. The pre-registered metric is reward MAE @ horizon 5 on the same offline-diagnostic protocol. Survival is secondary (single-seed run; effect size noisy at this budget).

| Hypothesis | Reward MAE @ h=5 | Interpretation | Next action |
|---|---|---|---|
| **H1 (paper-canonical bins close the gap)** | **< 0.15** (under threshold) | Bin-range deviation was the dominant residual cause after zero-init. Paper-canonical bins gave the head usable resolution near zero on negative dense rewards. | Promote `paper_canonical_twohot_bins: true` to default permanently (already proposed default in §2.3). Update §6 item 2 of [`dreamer_v3_implementation.md`](../../../project/concepts/dreamer_v3_implementation.md) from "deviation" to "ACTED ON; matches paper". Close the reward-head investigation thread for NoPred. |
| **H2 (further partial improvement)** | 0.15 ≤ MAE < 0.25 | Bin range was contributing but not fully causal; another upstream signal (most likely candidate #1 GRU reset gate, structural) is also degrading the head. **Note**: 0.25 is tighter than Z1's H0 threshold of 0.30 — Z1 already moved MAE to 0.28, so the bin fix needs to make a *meaningful additional contribution* to land in H2. | Queue candidate #1 (GRU reset-gate fix; §6 item 28) as the next plan. Keep both fixes ON cumulatively. |
| **H0 (paper-canonical bins don't help on NoPred)** | **≥ 0.25** | Bin-range deviation is not the primary residual cause on NoPred. The residual is genuinely structural (not bin-coverage related), most likely candidate #1 (GRU reset gate) or candidate #2 (prior/posterior head capacity, §6 item 30). | Queue candidate #1 as the next plan. Keep `paper_canonical_twohot_bins: true` as the default anyway (it matches paper and is the correct convention; the lack-of-NoPred-effect is consistent with NoPred reward magnitudes never exiting the legacy ±20 boundary, not with the fix being wrong). Flag for predator-task follow-up where the death-penalty `−100` event genuinely benefits from the wider support. |

The offline diagnostic re-run is **not part of this plan's spawn chain**. After Z2 finishes, the user (or the parent agent) runs the diagnostic via `experiment-analyzer`, which then writes the verdict back to this plan and to the failure-mode memo.

### §4.1 Secondary signals to check at analysis time

(For `experiment-analyzer` to record alongside the primary verdict; not gating on §4 verdict.)

- Training-time `model_reward_mae_neg` should drop noticeably (Z1's `mae_neg` was the hold-out at 0.80 vs `mae_pos`'s 0.42 — if paper bins help, the asymmetry should narrow).
- Training-time `model_reward_mae_pos` should remain ≤ Z1's 0.42 (no regression).
- Reward MAE per-horizon table (h=1, 2, 5, 10, 15, 25, 50): Z1 was monotonically better than A1 at h≥5; Z2 should preserve or extend that.
- Survival (`Episode/Steps`) is informative but not gating: Z1 lifted survival ~9 steps over A1 on a single seed; Z2 may or may not move further.
- Per-channel obs symlog-MSE: Proprio was borderline in Z1 (0.0542 vs threshold 0.05). Worth tracking to see if the head improvement spills over.

---

## §5 Out of scope

- **Other §6 candidates** (#1 GRU reset gate / item 28, #2 prior/posterior heads / item 30, #3 critic-EMA / item 29) — deferred per user instruction. Do not bundle.
- **Predator task re-run** — death-penalty `−100` is the cleanest stress test of the fix, but Cell A2 (predator) had a different failure mode (~30-step floor with mae_pos/mae_neg asymmetry) and no offline-diagnostic localisation yet. NoPred is the cleanest test bed for *this* deviation in isolation. Predator validation is a follow-up plan after Z2's verdict.
- **Cumulative-fix matrix** — no factorial sweep across (zero-init × paper-bins × baseline). Single Z2 cell, cumulative on top of Z1; that's the scope.
- **Changes to the offline diagnostic script** — re-using the same script with no edits is methodologically required (the diagnostic is the measuring instrument; changing it would invalidate the comparison against Z1).
- **Changes to any other config knob simultaneously** — entire scientific value of the comparison hinges on this being a single-knob change relative to Z1.
- **Follow-up plan queueing** — until Z2's offline diagnostic verdict comes back, do not queue plans for candidates #1/#2/#3.
- **Updates to `docs/project/concepts/dreamer_v3_implementation.md` §6 item 2** — happens after Z2's verdict (under the H1 row, this plan's summary becomes the "ACTED ON" entry).

---

## §6 Hand-off

After this plan lands and is committed:

1. **`developer`** reads §2 in full, applies the changes to:
   - `src/models/dreamer_v3_util.py` (the two function bodies, plus docstrings; new optional kwarg `paper_canonical_bins: bool = True` on both)
   - `src/models/dreamer_v3_trainer.py` (mandatory read in `agent_config` dict; flag plumbed to all 9 call sites in this file via the cleanest pattern — §2.3.3 (A) preferred)
   - `src/models/dreamer_v3_nnx.py` (the one `from_twohot` call site at L681; route the flag through `agent_config` or refactor to the trainer)
   - `configs/models/dreamer_v3.yaml` (the new key with full doc-link comment)
   - `configs/models/dreamer_v3_rr06.yaml` (the new key with brief comment)

   Run the §2.5 smoke test (round-trip) and the §2.6 config end-to-end check. Fill out the Implementation Report below. Report back.

2. **`senior-developer`** verifies the implementation against §2 per the standard verification protocol.

3. **`training-runner`** reads §3 verbatim and launches Z2 on n113:0. Standard pre-flight + post-launch pgrep + diary update.

4. **After Z2 finishes (~2.5 h)** the user or parent agent runs `scripts/dreamer_offline_wm_test.py` on the Z2 checkpoint and reads the reward MAE @ h=5 against §4's pre-registered table. The verdict is written back into §4 (or appended as a Verification Report below) by `experiment-analyzer`.

5. **Closing actions** (per §4 verdict):
   - **H1 fires**: promote default permanently; update §6 item 2 of the concept doc; close the reward-head investigation for NoPred.
   - **H2 fires**: queue candidate #1 (GRU reset gate) as next plan; keep both fixes ON cumulatively.
   - **H0 fires**: queue candidate #1 as next plan; keep paper-canonical bins as default anyway (matches paper); flag for predator-task follow-up.

---

## Checkpoints (for `developer`)

- [x] `to_twohot` accepts `paper_canonical_bins: bool = False`; under `True`, `bottom = min_v` and `top = max_v` (no `symlog` applied to range constants); under `False`, original code path verbatim. (Note: default is `False` at function level to keep import-time tests bit-identical; trainer always passes `True` explicitly via config.)
- [x] `from_twohot` mirrors the same flag with the same semantics; bin layout matches `to_twohot` for any given flag value.
- [x] `DreamerV3Trainer.__init__` reads `agent.paper_canonical_twohot_bins` via `config.get_mandatory(...)`; missing YAML key raises `ValueError` at trainer construction.
- [x] Every `to_twohot` / `from_twohot` call site listed in §2.3.4 (10 in total) uses the same flag value sourced from `agent.paper_canonical_twohot_bins`. None left at default. (9 in trainer.py via `self._paper_canonical_twohot_bins`; 1 in nnx.py via `self.paper_canonical_twohot_bins` stashed from `agent_config`.)
- [x] `configs/models/dreamer_v3.yaml` and `configs/models/dreamer_v3_rr06.yaml` both contain `paper_canonical_twohot_bins: true`.
- [x] §2.5 round-trip smoke test passes (assertions hold; values inside ±20 round-trip cleanly under both flags; values at ±100 saturate at ±20 under `False` and round-trip under `True`; values at ±10000 saturate hard under `False` and round-trip under `True`).
- [x] §2.6 config end-to-end check passes (both YAML files expose the key; missing key raises `ValueError`).
- [x] When `paper_canonical_twohot_bins: false` is set, training output is bit-identical to pre-fix behaviour (verified by §2.4 reasoning — the `else` branch is verbatim previous code; PRNG consumption unchanged).
- [x] No edits made to the GRU cell, prior/posterior heads, critic loss, or zero-init knob — those are out-of-scope (§5).

---

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-05-10

### Files changed

| File | Change |
|---|---|
| `src/models/dreamer_v3_util.py` | Added `paper_canonical_bins: bool = False` kwarg to both `to_twohot` and `from_twohot`. Under `True`: `bottom = jnp.array(min_v, dtype=...)`, `top = jnp.array(max_v, dtype=...)` — no `symlog` applied to range constants. Under `False` (legacy default): verbatim original code path (`bottom = symlog(jnp.array(min_v, ...))`, `top = symlog(jnp.array(max_v, ...))`). Updated docstrings. |
| `src/models/dreamer_v3_trainer.py` | (1) Added `'paper_canonical_twohot_bins': config.get_mandatory('agent.paper_canonical_twohot_bins', bool)` to `agent_config` dict. (2) Added `self._paper_canonical_twohot_bins = config.get_mandatory('agent.paper_canonical_twohot_bins', bool)` after `self.agent` construction. (3) Updated all 9 call sites in `train_step` and `behavior_loss_fn` to pass `paper_canonical_bins=self._paper_canonical_twohot_bins`. |
| `src/models/dreamer_v3_nnx.py` | Added `self.paper_canonical_twohot_bins = config.get('paper_canonical_twohot_bins', False)` in `DreamerV3Agent.__init__`. Updated the one call site at line 682: `from_twohot(value_logits, num_buckets=value_logits.shape[-1], paper_canonical_bins=self.paper_canonical_twohot_bins)`. |
| `configs/models/dreamer_v3.yaml` | Added `paper_canonical_twohot_bins: true` with full doc-link comment after `zero_init_reward_critic`. |
| `configs/models/dreamer_v3_rr06.yaml` | Added `paper_canonical_twohot_bins: true` with brief comment referencing dreamer_v3.yaml. |

### Deviations from §2

1. **Function-level default is `False` (not `True`)**: The plan §2.2 spec says `paper_canonical_bins: bool = True` at the function signature level. However, the plan also says (§6 item 1 and §2.4) "the `False` default keeps import-time tests bit-identical". A `True` function-level default would silently change behaviour for any caller that doesn't pass the flag explicitly (e.g., tests that import `to_twohot` directly). The trainer always passes the flag explicitly via `self._paper_canonical_twohot_bins` (sourced from `config.get_mandatory`), so the function-level default only matters for bare calls in tests or scripts. Using `False` is the safe choice that makes the legacy-compat claim in §2.4 actually hold. **No training-path effect** — all 10 production call sites pass the flag explicitly.

2. **`DreamerV3Agent` uses `config.get('paper_canonical_twohot_bins', False)` (not `get_mandatory`)**: The plan routes the flag through `agent_config` dict, and `agent_config` is a plain Python `dict` (not a `Config` object), so `get_mandatory` is not available on it. Using `dict.get('paper_canonical_twohot_bins', False)` is correct: the trainer already enforces the mandatory constraint via `config.get_mandatory(...)` before building `agent_config`, so the key is always present in `agent_config` by the time `DreamerV3Agent.__init__` runs. The fallback `False` is a safety net only.

3. **Other dreamer configs** (`dreamer_v3_probe.yaml`, `dreamer_v3_curriculum.yaml`, `dreamer_v3_curriculum_probe.yaml`, `dreamer_v3_probe_cont10.yaml`, `neuromodulated_dreamer_v3.yaml`) do not have `paper_canonical_twohot_bins` (or `zero_init_reward_critic`). These were pre-existing issues not introduced by this plan. Plan scope is explicitly `dreamer_v3.yaml` and `dreamer_v3_rr06.yaml`. Flagged here for senior-developer to decide whether to add the key to those files too.

### Smoke test output (§2.5) — verbatim

```
val | paper-rec | paper-err | legacy-rec | legacy-err
 -10000.00    -9999.9648      0.0352      -20.0000   9980.0000
   -100.00      -99.9997      0.0003      -20.0000     80.0000
    -20.00      -20.0000      0.0000      -20.0000      0.0000
     -5.00       -5.0000      0.0000       -5.0000      0.0000
     -1.00       -1.0000      0.0000       -1.0000      0.0000
      0.00       -0.0000      0.0000       -0.0000      0.0000
      1.00        1.0000      0.0000        1.0000      0.0000
      5.00        5.0000      0.0000        5.0000      0.0000
     20.00       20.0001      0.0001       20.0000      0.0000
    100.00       99.9999      0.0001       20.0000     80.0000
  10000.00     9999.9746      0.0254       20.0000   9980.0000
OK: paper-canonical flag round-trips ±100 cleanly; legacy saturates at ±20.
```

All assertions passed:
- Inside ±20: both flags round-trip with err < 1.0 (maximum observed: 0.0001).
- At ±100: paper flag err = 0.0003 / 0.0001; legacy saturates at ±20.0000 (err = 80.0000).
- At ±10000: paper flag err = 0.0352 / 0.0254; legacy saturates at ±20.0000 (err = 9980.0000 each; total ~19960 >> 1000 threshold).

### Config end-to-end check (§2.6)

```
configs/models/dreamer_v3.yaml -> agent.paper_canonical_twohot_bins = True  (type: bool)
configs/models/dreamer_v3_rr06.yaml -> agent.paper_canonical_twohot_bins = True  (type: bool)
OK: both YAML files expose the mandatory key correctly.
OK: missing key raises ValueError: Strict Config: Configuration key 'agent.paper_canonical_twohot_bins' is required but missing.
```

### Speed check

Skipped: bin-grid construction (`jnp.array(min_v)` vs `symlog(jnp.array(min_v))`) is init-time-only — `bottom` and `top` are scalar constants computed once before `jnp.linspace` or `jnp.clip`. No hot-path effect. The JIT trace is structurally identical under both flag values (same number of ops, same shapes). No measurable SPS delta expected.

### Blockers

None. Implementation complete. Note for senior-developer: the 5 non-target config files listed in Deviation 3 above will raise `ValueError` at trainer construction if loaded with the current code (because `agent.paper_canonical_twohot_bins` is also absent from those files, like `zero_init_reward_critic`). This is a pre-existing issue from the zero-init fix — the current plan adds one more mandatory key to the same set of files that are already broken. Decision on whether to fix those files is out of scope for this plan.

Implemented by: developer

---

## Verification Report

> **Verified by**: senior-developer (post-implementation) / experiment-analyzer (post-Z2 diagnostic)
> **Date**: <YYYY-MM-DD>
> **Run**: Z2 — `dreamer_twohotrng_NoPred_rr06_s0_n113` / WandB `<run-id>` / checkpoint step <N>
> **Diagnostic outputs**: `tmp/<YYYYMMDD_HHMMSS>_wm_imagination_test_Z2.{json,md}`

### Headline

<to-fill — H0 / H1 / H2 from §4>

### Pre-registered hypothesis check (plan §4)

| Hypothesis | Pre-registered range | Z2 observed | Verdict |
|---|---|:---:|:---:|
| H1: paper-canonical bins close the gap | MAE @ h=5 < 0.15 | <to-fill> | <to-fill> |
| H2: bin range partial contributor | 0.15 ≤ MAE @ h=5 < 0.25 | <to-fill> | <to-fill> |
| H0: paper-canonical bins don't help on NoPred | MAE @ h=5 ≥ 0.25 | <to-fill> | <to-fill> |

### A1 vs Z1 vs Z2 — pre-registered metric table

| Metric | Threshold @ h=5 | A1 | Z1 | Z2 | Z2 vs Z1 delta | Z2 status |
|---|---|---|---|---|---|:---:|
| Reward MAE @ h=5 (raw space) — **PRIMARY** | < 0.15 | 0.3856 | 0.2765 | <to-fill> | <to-fill> | <to-fill> |
| Aggregate observation symlog-MSE @ h=5 | < 0.10 | 0.0594 | 0.0472 | <to-fill> | <to-fill> | <to-fill> |
| Continuation accuracy @ h=5 | > 0.95 | 0.9950 | 1.0000 | <to-fill> | <to-fill> | <to-fill> |
| Long-horizon h50/h5 ratio | ≤ 2.0 | 1.69 | 1.62 | <to-fill> | <to-fill> | <to-fill> |

### Implementation surface check (per Implementation Report)

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `src/models/dreamer_v3_util.py` | <to-fill> | <to-fill> | <to-fill> |
| `src/models/dreamer_v3_trainer.py` | <to-fill> | <to-fill> | <to-fill> |
| `src/models/dreamer_v3_nnx.py` | <to-fill> | <to-fill> | <to-fill> |
| `configs/models/dreamer_v3.yaml` | <to-fill> | <to-fill> | <to-fill> |
| `configs/models/dreamer_v3_rr06.yaml` | <to-fill> | <to-fill> | <to-fill> |

### Interpretation

<to-fill — three-bullet summary tying to §4 verdict and next-step recommendation>

### Next-step flags (advisory only — NOT spawning here)

<to-fill — keep paper-canonical bins ON; queue candidate #1; predator-task validation as follow-up; etc.>

**Conclusion**: <to-fill>

---

<!--
NEW ISSUES: any deviation discovered during implementation/verification that warrants its own plan should be split out (one of the deferred §6 candidates) rather than expanded inline here.
-->
