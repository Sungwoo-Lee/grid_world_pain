---
title: "DreamerV3 GRU Reset Gate Fix (apply reset to candidate; Cell Z3 re-run after A1 / Z1 / Z2)"
topic: diagnosis
status: active
created: 2026-05-11
last_updated: 2026-05-11
phase: 2
---

# DreamerV3 GRU Reset Gate Fix (apply reset to candidate; Cell Z3 re-run after A1 / Z1 / Z2)

> **Status**: PLANNED — implementation + launch authorised (user, 2026-05-11, "Plan + Implement + Launch this turn", same standing authorisation as Z1 and Z2).
> **Doc role**: doubles as design doc + Launch Manifest.
> **Related**:
>   - [`docs/project/concepts/dreamer_v3_implementation.md`](../../../project/concepts/dreamer_v3_implementation.md) §6 item 28 (currently flagged **"NOW NEXT IN THE FIX CASCADE"**) + §9.11.1 + §9.3.1 — the deviation this plan acts on. After this lands, §6 item 28 needs updating ("ACTED ON, see this plan").
>   - [`docs/reviews/dreamer_v3_implementation_math_review.md`](../../../reviews/dreamer_v3_implementation_math_review.md) — cross-validates against Hafner published `dreamerv3/nets.py` and sheeprl `models/models.py:399–403`.
>   - [`docs/develop/active/diagnosis/dreamer_twohot_bin_range_fix.md`](./dreamer_twohot_bin_range_fix.md) — the prior fix (Cell Z2) this plan stacks on top of; same template + flow.
>   - [`docs/develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md`](./dreamer_zero_init_reward_critic_fix.md) — Cell Z1, two rungs back.
>   - [`tmp/20260511_044500_wm_imagination_test_Z2_papercanonical.md`](../../../../tmp/20260511_044500_wm_imagination_test_Z2_papercanonical.md) — Z2's authoritative offline diagnostic; provides the long-horizon residual-error pattern cited in §1.
>   - [`docs/experiments/active/dreamer_diagnosis/DREAMER_CONVENTIONAL_FIXES_BATTERY.md`](../../../experiments/active/dreamer_diagnosis/DREAMER_CONVENTIONAL_FIXES_BATTERY.md) — Cell A1 baseline (`czfnljf0`), Z1 anchor (`axndoqsz`), Z2 anchor (`q66macky`).
>   - `tmp/sheeprl/sheeprl/models/models.py:399–403` (`LayerNormGRUCell.__call__`) — paper-canonical reference; the load-bearing line is `cand = torch.tanh(reset * cand)`.

---

## §1 Context

Two rungs of the DreamerV3 fix cascade have landed and partially repaired the world model's reward head, but the residual signature has shifted in a way that points at a different deviation. Rung 1 (Cell Z1, zero-init of the reward + critic output layers) dropped reward MAE @ horizon 5 from the A1 baseline's 0.386 to 0.277 — a 28% improvement, partial. Rung 2 (Cell Z2, paper-canonical two-hot symlog bin grid) dropped it further to 0.177 — a cumulative 54% reduction across A1 → Z1 → Z2, and the load-bearing mechanistic prediction was validated (training-time negative-reward MAE dropped 45% from Z1 to Z2, exactly the asymmetry repair the bin-coverage mechanism predicted). But Z2 narrowly missed the pre-registered H1 threshold of MAE < 0.15, landing at 0.177 — verdict **H2 partial improvement**, not full repair. The pre-registered action under H2 is to queue the next candidate.

The new residual signature is **long-horizon compounding**. Z2's reward MAE per horizon reads:

| Horizon | h=1 | h=2 | h=5 | h=10 | h=15 | h=25 | h=50 |
|---|---|---|---|---|---|---|---|
| Z2 reward MAE | 0.094 | 0.007 | **0.177** | 0.088 | 0.262 | 1.175 | **3.051** |
| Z1 reward MAE | 0.445 | 0.369 | 0.277 | 0.367 | 0.304 | 0.143 | 0.233 |
| A1 reward MAE | 0.323 | 0.438 | 0.386 | 0.679 | 0.546 | 0.330 | 0.678 |

Z2's h50/h5 ratio is **17.2** (3.05 / 0.177), where Z1's was 0.84 and A1's was 1.76. The paper-canonical bin grid is materially wider in raw space (raw support ±4.85·10⁸ vs the legacy ±20), which means the reward head's symexp readout is now more sensitive to small symlog-space logit perturbations on imagined off-manifold features. The h=5 number improved, the h=50 number got worse by an order of magnitude — those two patterns together say the per-step decoder is fine but the **per-step latent drift inside imagination is accumulating across the recurrent rollout**. The only component that touches every imagined step uniformly and could produce exactly that compounding pattern is the GRU's recurrent dynamics. Z2's plan §V.4 named this prediction explicitly: "the residual is consistent with imagination-trajectory stability rather than bin coverage, pointing at GRU reset gate (candidate #1) as the next rung."

Our `LayerNormGRUCell` in `src/models/dreamer_v3_nnx.py:18–39` and its modulated sibling `ModulatedLayerNormGRUCell` in `src/models/modulated_layer_norm_gru_cell.py:24–78` both **compute the reset gate via sigmoid but never multiply it into the candidate update**. The paper-canonical GRU (Cho et al. 2014; Hafner published `dreamerv3/nets.py`; sheeprl `models/models.py:399–401`) is `cand = tanh(reset * cand)` — the reset gate's purpose is exactly to gate how much of the previous hidden state leaks into the candidate update on each step. Ours computes `reset = sigmoid(reset)` and then never uses the variable; `cand = tanh(cand)` proceeds without it. Dead-computation pattern — likely a refactor that stranded the gate. Severity is broad because the cell is called once per env step and `T_imag = 15` times per imagined rollout per replay state.

This plan applies the one-line fix in both cells, gates it behind a single new config knob `agent.apply_gru_reset_gate` (default `true`, off for bit-identical legacy reproduction), and re-runs the same NoPred 5×5 cell with all three cumulative fixes (replay_ratio 0.0625 + zero-init reward+critic + paper-canonical two-hot bins + GRU reset gate). The offline world-model imagination diagnostic is then re-run against the Z3 checkpoint to test (a) whether reward MAE @ h=5 falls below H1's 0.15 floor and (b) whether the long-horizon h50/h5 ratio collapses back below the historical PASS band of ≤ 2.0.

---

## §2 Implementation spec

### §2.1 Mechanism (the bug, plain-text)

The Gated Recurrent Unit (Cho et al. 2014) has three gates per step: a **reset gate** `r`, an **update gate** `u`, and a **candidate state** `c̃`. The canonical update is

```
r  = σ(W_r · [h_{t-1}, x_t])
u  = σ(W_u · [h_{t-1}, x_t])
c̃  = tanh(W_c · [r ⊙ h_{t-1}, x_t])         ← reset gate applied here
h_t = (1 - u) ⊙ h_{t-1} + u ⊙ c̃
```

In LayerNormGRU variants the LN is applied per-gate after the linear projection, but the reset-gate-times-candidate multiplication still happens. Sheeprl's faithful port at `tmp/sheeprl/sheeprl/models/models.py:399–403`:

```python
reset, cand, update = torch.chunk(x, 3, -1)
reset  = torch.sigmoid(reset)
cand   = torch.tanh(reset * cand)             # ← reset gate APPLIED
update = torch.sigmoid(update - 1)             # (Hafner -1 bias, separate issue)
hx     = update * cand + (1 - update) * hx
```

Our `LayerNormGRUCell.__call__` in `src/models/dreamer_v3_nnx.py:27–39` (verbatim current code):

```python
def __call__(self, x, h):
    gates_ih = self.ln_ih(self.dense_ih(x))
    gates_hh = self.ln_hh(self.dense_hh(h))
    gates = gates_ih + gates_hh

    reset, update, cand = jnp.split(gates, 3, axis=-1)

    reset = nnx.sigmoid(reset)
    update = nnx.sigmoid(update)
    cand = jnp.tanh(cand)                      # ← reset NOT applied

    h_new = (1 - update) * h + update * cand
    return h_new
```

`reset` is computed (sigmoid applied) and then the local variable is never used. `cand = jnp.tanh(cand)` ignores it. Same bug in the modulated sibling at `src/models/modulated_layer_norm_gru_cell.py:62–77` (the modulation branch adds a `gate_bias` to the update gate, which is orthogonal to the reset-gate bug).

The fix is one line in each cell: `cand = jnp.tanh(reset * cand)`. Plus the reversibility knob to keep legacy traces bit-identical.

Note on chunk order: sheeprl splits `(reset, cand, update)`; ours splits `(reset, update, cand)`. The order within the linear-projection output is internal to each cell and not load-bearing — both cells already initialise their own weights from scratch. The fix preserves our current `(reset, update, cand)` ordering and modifies only the candidate computation.

### §2.2 File changes

Two source files. Quoted line numbers below are verified from current HEAD.

#### §2.2.1 `src/models/dreamer_v3_nnx.py:18–39` (`LayerNormGRUCell`)

```python
# BEFORE (lines 18–39):
class LayerNormGRUCell(nnx.Module):
    def __init__(self, hidden_size: int, rngs: nnx.Rngs):
        self.hidden_size = hidden_size
        self.dense_ih = nnx.Linear(hidden_size, 3 * hidden_size, use_bias=False, kernel_init=hafner_init(), rngs=rngs)
        self.dense_hh = nnx.Linear(hidden_size, 3 * hidden_size, use_bias=False, kernel_init=hafner_init(), rngs=rngs)

        self.ln_ih = nnx.LayerNorm(3 * hidden_size, rngs=rngs)
        self.ln_hh = nnx.LayerNorm(3 * hidden_size, rngs=rngs)

    def __call__(self, x, h):
        gates_ih = self.ln_ih(self.dense_ih(x))
        gates_hh = self.ln_hh(self.dense_hh(h))
        gates = gates_ih + gates_hh

        reset, update, cand = jnp.split(gates, 3, axis=-1)

        reset = nnx.sigmoid(reset)
        update = nnx.sigmoid(update)
        cand = jnp.tanh(cand)

        h_new = (1 - update) * h + update * cand
        return h_new
```

```python
# AFTER:
class LayerNormGRUCell(nnx.Module):
    def __init__(self, hidden_size: int, rngs: nnx.Rngs,
                 apply_reset_gate: bool = False):
        """
        Args:
            hidden_size: Hidden dimension (= deter_dim in RSSM).
            rngs:        Flax NNX random number generators.
            apply_reset_gate:
                If True (paper-canonical), candidate state is computed as
                `tanh(reset * cand)` — the reset gate gates how much of the
                previous hidden state leaks into the candidate update each
                step. Matches Cho et al. 2014 GRU, Hafner published
                `dreamerv3/nets.py`, and sheeprl
                `models/models.py:399–401` (`cand = tanh(reset * cand)`).
                If False (legacy buggy behaviour), candidate state is
                `tanh(cand)` — `reset` is computed but its result is
                discarded. Provided for bit-identical pre-fix reproduction
                only. See
                docs/develop/active/diagnosis/dreamer_gru_reset_gate_fix.md
                and §6 item 28 of dreamer_v3_implementation.md.
        """
        self.hidden_size = hidden_size
        self.apply_reset_gate = apply_reset_gate
        self.dense_ih = nnx.Linear(hidden_size, 3 * hidden_size, use_bias=False, kernel_init=hafner_init(), rngs=rngs)
        self.dense_hh = nnx.Linear(hidden_size, 3 * hidden_size, use_bias=False, kernel_init=hafner_init(), rngs=rngs)

        self.ln_ih = nnx.LayerNorm(3 * hidden_size, rngs=rngs)
        self.ln_hh = nnx.LayerNorm(3 * hidden_size, rngs=rngs)

    def __call__(self, x, h):
        gates_ih = self.ln_ih(self.dense_ih(x))
        gates_hh = self.ln_hh(self.dense_hh(h))
        gates = gates_ih + gates_hh

        reset, update, cand = jnp.split(gates, 3, axis=-1)

        reset = nnx.sigmoid(reset)
        update = nnx.sigmoid(update)

        if self.apply_reset_gate:
            # Paper-canonical: reset gate multiplies into the candidate.
            cand = jnp.tanh(reset * cand)
        else:
            # Legacy buggy behaviour: reset is computed (sigmoid applied
            # above) but never used. Kept verbatim for bit-identical
            # pre-fix reproduction. PRNG consumption and op count are
            # unchanged relative to the legacy path (the sigmoid on
            # `reset` still executes; only the multiplication on the
            # next line is gated).
            cand = jnp.tanh(cand)

        h_new = (1 - update) * h + update * cand
        return h_new
```

#### §2.2.2 `src/models/modulated_layer_norm_gru_cell.py:24–78` (`ModulatedLayerNormGRUCell`)

```python
# BEFORE (lines 35–78):
def __init__(self, hidden_size: int, rngs: nnx.Rngs):
    self.hidden_size = hidden_size

    self.dense_ih = nnx.Linear(hidden_size, 3 * hidden_size, use_bias=False, rngs=rngs)
    self.dense_hh = nnx.Linear(hidden_size, 3 * hidden_size, use_bias=False, rngs=rngs)

    self.ln_ih = nnx.LayerNorm(3 * hidden_size, rngs=rngs)
    self.ln_hh = nnx.LayerNorm(3 * hidden_size, rngs=rngs)

def __call__(
    self,
    x: jnp.ndarray,
    h: jnp.ndarray,
    gate_bias: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    gates_ih = self.ln_ih(self.dense_ih(x))
    gates_hh = self.ln_hh(self.dense_hh(h))
    gates = gates_ih + gates_hh

    reset, update, cand = jnp.split(gates, 3, axis=-1)

    reset = nnx.sigmoid(reset)

    if gate_bias is not None:
        update = nnx.sigmoid(update + gate_bias)
    else:
        update = nnx.sigmoid(update)

    cand = jnp.tanh(cand)

    h_new = (1.0 - update) * h + update * cand
    return h_new
```

```python
# AFTER:
def __init__(self, hidden_size: int, rngs: nnx.Rngs,
             apply_reset_gate: bool = False):
    """
    Args:
        hidden_size: Hidden dimension (= deter_dim in RSSM).
        rngs:        Flax NNX random number generators.
        apply_reset_gate:
            Mirrors LayerNormGRUCell.apply_reset_gate. See that
            docstring + the plan at
            docs/develop/active/diagnosis/dreamer_gru_reset_gate_fix.md
            for semantics. Default False keeps direct-import / test
            behaviour bit-identical pre-fix; the RSSM constructor passes
            the configured value explicitly via agent_config.
    """
    self.hidden_size = hidden_size
    self.apply_reset_gate = apply_reset_gate

    self.dense_ih = nnx.Linear(hidden_size, 3 * hidden_size, use_bias=False, rngs=rngs)
    self.dense_hh = nnx.Linear(hidden_size, 3 * hidden_size, use_bias=False, rngs=rngs)

    self.ln_ih = nnx.LayerNorm(3 * hidden_size, rngs=rngs)
    self.ln_hh = nnx.LayerNorm(3 * hidden_size, rngs=rngs)

def __call__(
    self,
    x: jnp.ndarray,
    h: jnp.ndarray,
    gate_bias: Optional[jnp.ndarray] = None,
) -> jnp.ndarray:
    gates_ih = self.ln_ih(self.dense_ih(x))
    gates_hh = self.ln_hh(self.dense_hh(h))
    gates = gates_ih + gates_hh

    reset, update, cand = jnp.split(gates, 3, axis=-1)

    reset = nnx.sigmoid(reset)

    if gate_bias is not None:
        update = nnx.sigmoid(update + gate_bias)
    else:
        update = nnx.sigmoid(update)

    if self.apply_reset_gate:
        cand = jnp.tanh(reset * cand)
    else:
        cand = jnp.tanh(cand)

    h_new = (1.0 - update) * h + update * cand
    return h_new
```

#### §2.2.3 `src/models/dreamer_v3_nnx.py:41–60` (`RSSM`) — thread the flag to the cell

The `RSSM.__init__` currently builds the cell at lines 53–56 with no flag. Add an `apply_gru_reset_gate: bool = False` kwarg to `RSSM.__init__` and pass it through to both cell constructors:

```python
# BEFORE (lines 41–60):
class RSSM(nnx.Module):
    def __init__(self, action_dim: int, deter_dim: int, stoch_dim: int,
                 discrete: int, embed_dim: int,
                 modulation_enabled: bool, rngs: nnx.Rngs):
        self.deter_dim = deter_dim
        ...
        if modulation_enabled:
            self.cell = ModulatedLayerNormGRUCell(deter_dim, rngs=rngs)
        else:
            self.cell = LayerNormGRUCell(deter_dim, rngs=rngs)
```

```python
# AFTER:
class RSSM(nnx.Module):
    def __init__(self, action_dim: int, deter_dim: int, stoch_dim: int,
                 discrete: int, embed_dim: int,
                 modulation_enabled: bool, rngs: nnx.Rngs,
                 apply_gru_reset_gate: bool = False):
        self.deter_dim = deter_dim
        ...
        if modulation_enabled:
            self.cell = ModulatedLayerNormGRUCell(
                deter_dim, rngs=rngs, apply_reset_gate=apply_gru_reset_gate
            )
        else:
            self.cell = LayerNormGRUCell(
                deter_dim, rngs=rngs, apply_reset_gate=apply_gru_reset_gate
            )
```

#### §2.2.4 `src/models/dreamer_v3_nnx.py:480–527` (`WorldModel`) — forward the flag from agent_config

`WorldModel.__init__` reads the agent_config dict already. Read the new key with a safe fallback (`agent_config['apply_gru_reset_gate']` is guaranteed present because the trainer enforces `get_mandatory` — same pattern Z2 used for `paper_canonical_twohot_bins`):

```python
# In WorldModel.__init__, just before self.rssm = RSSM(...) at line 522:
apply_gru_reset_gate = config.get('apply_gru_reset_gate', False)

self.rssm = RSSM(
    act_dim, self.deter_dim, self.stoch_dim, self.discrete,
    embed_dim=encoder_dim,
    modulation_enabled=self.modulation_enabled,
    rngs=rngs,
    apply_gru_reset_gate=apply_gru_reset_gate,   # NEW
)
```

The `config.get(...)` fallback is `False` (matches Z2's pattern at `DreamerV3Agent.__init__:621`). The trainer enforces presence via `get_mandatory` before the agent_config dict is built; the fallback is a safety net only.

#### §2.2.5 `src/models/dreamer_v3_trainer.py:64–82` — mandatory read into agent_config

Add the new entry alongside the existing `paper_canonical_twohot_bins` entry at line 77:

```python
# BEFORE (current line 77, after the Z2 plan landed):
'paper_canonical_twohot_bins': config.get_mandatory('agent.paper_canonical_twohot_bins', bool),  # NEW (Z2)
```

```python
# AFTER (add new line right below it):
'paper_canonical_twohot_bins': config.get_mandatory('agent.paper_canonical_twohot_bins', bool),
'apply_gru_reset_gate':        config.get_mandatory('agent.apply_gru_reset_gate', bool),  # NEW (Z3)
```

No further plumbing in `trainer.py` needed — the flag is consumed inside `WorldModel.__init__` (§2.2.4) where the RSSM is built; the trainer doesn't call the GRU cell directly.

### §2.3 Config wiring

#### §2.3.1 `configs/models/dreamer_v3.yaml` — add the knob (canonical, full comment)

After the existing `paper_canonical_twohot_bins: true` block at line 48 (and its multi-line comment), add:

```yaml
  apply_gru_reset_gate: true   # Apply the GRU reset gate to the candidate
                                # update: cand = tanh(reset * cand). Matches
                                # paper / Hafner published / sheeprl. When
                                # false, computes reset but never uses it
                                # (legacy bug; bit-identical pre-fix). See
                                # docs/develop/active/diagnosis/dreamer_gru_reset_gate_fix.md
                                # and §6 item 28 of dreamer_v3_implementation.md.
```

#### §2.3.2 `configs/models/dreamer_v3_rr06.yaml` — add the knob (mirror, brief)

After the existing `paper_canonical_twohot_bins: true` at line 86, add:

```yaml
  apply_gru_reset_gate: true   # See configs/models/dreamer_v3.yaml for full comment.
```

#### §2.3.3 Pre-existing config files (per Z2's Deviation 3 / Implementation Report)

`dreamer_v3_probe.yaml`, `dreamer_v3_curriculum.yaml`, `dreamer_v3_curriculum_probe.yaml`, `dreamer_v3_probe_cont10.yaml`, `neuromodulated_dreamer_v3.yaml` are already broken under the current `get_mandatory` set (missing `zero_init_reward_critic` and `paper_canonical_twohot_bins`). This plan **adds one more mandatory key to the same set**. Out of scope to fix here per Z2's deferred decision; flagged for senior-developer to decide whether to bundle the cleanup into a separate plan. The Z3 launch only loads `dreamer_v3_rr06.yaml`, which gets the key.

### §2.4 Reversibility / bit-identity claim

When `apply_gru_reset_gate: false`:

- `LayerNormGRUCell.__call__`: the `else` branch is the **verbatim previous code** (`cand = jnp.tanh(cand)`). The `reset = nnx.sigmoid(reset)` line still runs (unchanged from legacy), preserving op count and tracing behaviour.
- `ModulatedLayerNormGRUCell.__call__`: same — the `else` branch is verbatim previous code; `gate_bias` modulation path is untouched.
- PRNG consumption: no random ops in either cell, so unchanged.
- Op count: identical (one extra `jnp.multiply` in the `True` branch only).

Therefore: with `apply_gru_reset_gate: false`, model output is **bit-identical** to pre-fix.

When `apply_gru_reset_gate: true`, the forward graph gains a single `(reset * cand)` multiplication before the existing `jnp.tanh`. JIT tracing should be unaffected (same shapes, one extra elementwise op per GRU step); no measurable SPS regression expected (the cell is far from the dominant cost — encoder + decoder MLP heads dominate WM forward time).

### §2.5 Verification — smoke test (round-trip + behavioural divergence)

The fix changes a hidden-state update rule, not a discretisation grid, so the round-trip framing from Z2 doesn't translate. The right smoke test is: **with the flag flipped, the cell's output diverges from the legacy output on the same input** — concretely, the new `(reset * cand)` term in the candidate is *not* a no-op for typical sigmoid outputs (mean ≈ 0.5).

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -c "
import jax
import jax.numpy as jnp
from flax import nnx
from src.models.dreamer_v3_nnx import LayerNormGRUCell
from src.models.modulated_layer_norm_gru_cell import ModulatedLayerNormGRUCell

hidden = 64
B = 4

# Two cells with identical RNG seed and identical weights, only the flag differs.
def build(cell_cls, flag, seed):
    rngs = nnx.Rngs(params=seed)
    return cell_cls(hidden, rngs=rngs, apply_reset_gate=flag)

# We CAN'T trivially weight-share between two flax modules built with independent
# Rngs, but we CAN build one cell and call it twice via parameter snapshotting.
# Simpler approach: build one cell with apply_reset_gate=True, manually run the
# forward both ways using the cell's parameters, and compare.

cell_paper  = build(LayerNormGRUCell, True,  seed=0)
cell_legacy = build(LayerNormGRUCell, False, seed=0)

# Snapshot params from cell_paper, copy into cell_legacy so weights match.
# (nnx pattern: nnx.state(...) + nnx.update(...).)
state_paper = nnx.state(cell_paper, nnx.Param)
nnx.update(cell_legacy, state_paper)

key = jax.random.PRNGKey(7)
x = jax.random.normal(key, (B, hidden))
h = jax.random.normal(jax.random.split(key)[0], (B, hidden))

h_paper  = cell_paper(x, h)
h_legacy = cell_legacy(x, h)

print('LayerNormGRUCell — same x, h, same weights, different flag:')
print(f'  max |h_paper - h_legacy| = {float(jnp.max(jnp.abs(h_paper - h_legacy))):.6f}')
print(f'  mean|h_paper - h_legacy| = {float(jnp.mean(jnp.abs(h_paper - h_legacy))):.6f}')

# Assertion: outputs MUST diverge — the reset gate's mean over the
# 0.5-sigmoid prior gives ~0.5x scaling on candidate, so legacy and paper
# outputs should differ by at least an order of 1e-3 with random inputs.
assert float(jnp.max(jnp.abs(h_paper - h_legacy))) > 1e-3, \
    'flag had no effect on output — multiplication not wired in correctly'

# Also assert the legacy-flag output reproduces a hand-computed legacy reference:
gates_ih = cell_paper.ln_ih(cell_paper.dense_ih(x))
gates_hh = cell_paper.ln_hh(cell_paper.dense_hh(h))
gates = gates_ih + gates_hh
reset, update, cand = jnp.split(gates, 3, axis=-1)
reset_s  = jax.nn.sigmoid(reset)
update_s = jax.nn.sigmoid(update)
cand_legacy = jnp.tanh(cand)
cand_paper  = jnp.tanh(reset_s * cand)
h_ref_legacy = (1 - update_s) * h + update_s * cand_legacy
h_ref_paper  = (1 - update_s) * h + update_s * cand_paper
assert float(jnp.max(jnp.abs(h_legacy - h_ref_legacy))) < 1e-6, \
    'legacy-flag output deviates from hand-computed legacy reference'
assert float(jnp.max(jnp.abs(h_paper - h_ref_paper))) < 1e-6, \
    'paper-flag output deviates from hand-computed paper reference'

# Repeat for the modulated cell with gate_bias=None (should match LayerNormGRUCell).
cell_mod_paper  = build(ModulatedLayerNormGRUCell, True,  seed=1)
cell_mod_legacy = build(ModulatedLayerNormGRUCell, False, seed=1)
state_mod_paper = nnx.state(cell_mod_paper, nnx.Param)
nnx.update(cell_mod_legacy, state_mod_paper)
h_mod_paper  = cell_mod_paper(x, h)
h_mod_legacy = cell_mod_legacy(x, h)
assert float(jnp.max(jnp.abs(h_mod_paper - h_mod_legacy))) > 1e-3, \
    'modulated cell: flag had no effect on output'

print('OK: paper-flag and legacy-flag outputs differ as expected; '
      'each matches its hand-computed reference; both cells.')
"
```

Acceptance criteria:
- `LayerNormGRUCell` output with `apply_reset_gate=True` differs from `apply_reset_gate=False` on the same input/state/weights by max |Δ| > 1e-3 (typically O(0.1)).
- `LayerNormGRUCell` output with `apply_reset_gate=False` matches the hand-computed legacy reference (`cand = tanh(cand)`) to within numerical noise (< 1e-6).
- `LayerNormGRUCell` output with `apply_reset_gate=True` matches the hand-computed paper reference (`cand = tanh(reset * cand)`) to within numerical noise (< 1e-6).
- `ModulatedLayerNormGRUCell` (with `gate_bias=None`) exhibits the same divergence pattern.

### §2.6 Verification — config end-to-end check

Confirm both YAML files expose the new mandatory key and a missing key raises:

```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python -c "
from src.utils.strict_config import StrictConfig   # adapt to actual API
for path in ['configs/models/dreamer_v3.yaml', 'configs/models/dreamer_v3_rr06.yaml']:
    cfg = StrictConfig.load(path)
    val = cfg.get_mandatory('agent.apply_gru_reset_gate', bool)
    print(f'{path} -> agent.apply_gru_reset_gate = {val}  (type: {type(val).__name__})')
print('OK: both YAML files expose the mandatory key correctly.')
"
```

(Adapt the import to the actual `StrictConfig` API — same pattern as Z2's plan §2.6.)

Also confirm a missing key raises `ValueError` — same idiom as Z1 / Z2.

### §2.7 Out of scope for the code change

- Don't change the GRU's `update = sigmoid(update)` to `sigmoid(update - 1)` (the Hafner `-1` update-gate bias). That's a separate deviation flagged in §9.11.1 of the concept doc (we identified it but did not promote to a §6 candidate). The reset-gate fix is the strictly more load-bearing one (every cell step uses the reset gate; the `-1` bias only affects init).
- Don't touch the prior/posterior head MLP undersizing (§6 item 30) — that's the candidate after this one.
- Don't touch the critic self-EMA regularisation term (§6 item 29) — deferred per user instruction.
- Don't touch zero-init (already on; Z1 fix) or paper-canonical bins (already on; Z2 fix).
- Don't change `replay_ratio` (already 0.0625 in `dreamer_v3_rr06.yaml`).
- Don't change the GRU chunk order `(reset, update, cand)` to sheeprl's `(reset, cand, update)` — internal to each cell, weights are learned from scratch, no transferable artefact.
- Don't touch the diagnostic script `scripts/dreamer_offline_wm_test.py` unless the §6 hand-off grep reveals a direct GRU-cell instantiation (it doesn't — see §6 below; the script loads checkpoints through `DreamerTrainer` and never instantiates the cell directly).

---

## §3 Launch manifest (single cell)

After the developer reports implementation complete, `training-runner` launches the following cell on node 113.

| Cell | Node:GPU | Task | Tag | WandB run name | WandB group | Seed | Env steps | Experiment config | Agent config |
|---|---|---|---|---|---|---|---|---|---|
| Z3 | n113:0 | NoPred (5×5) | `dreamer_gru_NoPred_rr06_s0_n113` | `dreamer_gru_NoPred_rr06_s0_n113` | `dreamer_gru_reset_gate` | 0 | 700,000 | `configs/experiment/basic/00-5X5_NoPred.yaml` | `configs/models/dreamer_v3_rr06.yaml` (now with `apply_gru_reset_gate: true`, `paper_canonical_twohot_bins: true`, `zero_init_reward_critic: true`) |

- **WandB group**: `dreamer_gru_reset_gate` — isolated from `dreamer_paper_canonical_bins` (Z2), `dreamer_zero_init` (Z1), and `dreamer_conventional_fixes` (A-cells) so the cumulative-fix framing is clean.
- **Tag == WandB run name** (per the standing Launch Manifest rule; `feedback_launch_manifest.md`).
- **Direct comparison anchors**:
  - **Cell A1 baseline** (`czfnljf0`, NoPred + `replay_ratio=0.0625`; reward MAE @ h=5 = 0.386).
  - **Cell Z1 anchor** (`axndoqsz`, A1 + `zero_init_reward_critic=true`; reward MAE @ h=5 = 0.277).
  - **Cell Z2 anchor** (`q66macky`, Z1 + `paper_canonical_twohot_bins=true`; reward MAE @ h=5 = 0.177; long-horizon h50/h5 = 17.2).
  - **Cell Z3 (this run)**: Z2 + GRU reset gate. Same task, same seed, only the GRU flag flipped relative to Z2.
- **Cumulative fixes active in Z3**:
  - `replay_ratio: 0.0625` (rr06 config; carried over)
  - `zero_init_reward_critic: true` (Z1 fix)
  - `paper_canonical_twohot_bins: true` (Z2 fix)
  - `apply_gru_reset_gate: true` (NEW in this plan)
- **Expected wall-clock**: ~2.5 h on n113:0 (same envelope as Z1 / Z2; the cell change is a single elementwise multiply per step — well below noise on the WM hot path).
- **Pre-flight (training-runner own protocol)**: ssh n113 + `python -c 'import jax'` env check + `pgrep -af '<TAG>'` post-launch verification with single-PID assertion, per `.claude/agents/training-runner.md` and `feedback_runner_post_launch_pgrep.md`.

**NoPred-specific mechanistic note (load-bearing for §4)**: NoPred has no death penalty, so reward magnitudes are small and the long-horizon compounding seen in Z2 isn't dominated by extreme rewards. The mechanism the fix targets — `(reset * cand)` gating in the GRU — is structural and affects every imagined-step latent update uniformly, regardless of reward distribution. If the fix is doing its job, the **h50/h5 ratio should collapse back below the historical PASS band of ≤ 2.0** (A1 was 1.76, Z1 was 0.84, Z2 spiked to 17.2). This is the secondary signal that most directly tests the §1 hypothesis. The primary metric (reward MAE @ h=5 < 0.15) is the H1 gate, but the h50/h5 ratio is the mechanism check. Both will be recorded.

---

## §4 Pre-registered confirmation/refutation criteria

Re-run `scripts/dreamer_offline_wm_test.py` on Z3's checkpoint, identical command to A1 / Z1 / Z2 invocations. The pre-registered primary metric is reward MAE @ horizon 5; the load-bearing secondary metric is the long-horizon h50/h5 reward-MAE ratio (which directly tests the §1 mechanism). Survival is informative but not gating on a single seed.

| Hypothesis | Reward MAE @ h=5 | h50/h5 ratio | Interpretation | Next action |
|---|---|---|---|---|
| **H1 (GRU reset gate closes the gap)** | **< 0.15** | ≤ 2.0 (back inside historical PASS band) | The reset-gate deviation was the dominant residual cause after zero-init + bin range. Reset gating restored imagination-trajectory stability, which both lowered h=5 error and collapsed the long-horizon compounding. | Promote `apply_gru_reset_gate: true` to permanent default (already proposed default in §2.3). Update §6 item 28 of [`dreamer_v3_implementation.md`](../../../project/concepts/dreamer_v3_implementation.md) from "NOW NEXT IN THE FIX CASCADE" to "ACTED ON; matches paper". Close the reward-head investigation thread for NoPred. Queue predator-task validation as a separate downstream plan (the −100 death-penalty event is the cleanest stress test of the full cumulative stack). |
| **H2 (further partial improvement)** | 0.15 ≤ MAE < 0.18 | < 17.2 (some collapse) but > 2.0 | Reset-gate fix is contributing but not fully causal; head capacity (§6 item 30, prior/posterior heads no hidden layer) or critic self-EMA (§6 item 29) is also degrading the WM. Note the H2 band is tighter than Z2's [0.15, 0.25) — Z2 already moved MAE to 0.177, so to claim H2 here the fix must make a *meaningful additional contribution*. | Queue candidate #30 (prior/posterior head capacity) as the next plan. Keep all four fixes ON cumulatively. |
| **H0 (GRU reset gate doesn't move the primary metric)** | **≥ 0.18** | secondary check (see below) | The reset-gate fix doesn't move reward MAE @ h=5 *as a single-knob delta*. **Secondary signal still load-bearing**: if h50/h5 collapses dramatically (≤ 2.0) even though MAE @ h=5 doesn't move, the mechanism is partly validated — the long-horizon stability *is* what the fix addresses, and the h=5 floor is bottlenecked elsewhere. Treat as H0a (reset gate validated on the long-horizon mechanism but not on the h=5 gate) vs H0b (no mechanistic effect at all). | H0a: keep `apply_gru_reset_gate: true` as default anyway (matches paper); queue candidate #30. H0b: keep flag as default (matches paper) but the mechanism hypothesis is invalidated; investigate prior/posterior heads (§6 item 30) and critic-EMA (§6 item 29) together. |

**Important**: the H2 / H0 thresholds are tighter than Z2's because Z2 already sits at 0.177 (just 0.027 above H1). Each rung in the cascade needs to make *meaningful* additional headroom. The h50/h5 ratio is a separately load-bearing secondary signal because it's the metric the §1 mechanism most directly predicts.

The offline diagnostic re-run is **not part of this plan's spawn chain**. After Z3 finishes, the user (or parent agent) runs `scripts/dreamer_offline_wm_test.py` via `experiment-analyzer`, which then writes the verdict back to this plan and to the failure-mode memo. The diagnostic script was patched during Z2 verification (commit `1703a4c`) to read `agent.paper_canonical_twohot_bins` from the saved config and route it through to `from_twohot`; the same pattern does **not** need extending here because the script does not instantiate the GRU cell directly (verified — see §6 below), it loads the cell weights via the full `DreamerTrainer` checkpoint path, which already reads the new `agent.apply_gru_reset_gate` key through `get_mandatory`.

### §4.1 Secondary signals to record at analysis time

(For `experiment-analyzer` to log alongside the primary verdict; not gating on §4 verdict.)

- **Long-horizon reward MAE per-horizon table** (h=1, 2, 5, 10, 15, 25, 50). This is the §1 load-bearing pattern; Z2's row is 0.094 / 0.007 / 0.177 / 0.088 / 0.262 / 1.175 / 3.051. Z3 should show meaningful collapse in the right tail.
- **h50/h5 reward-MAE ratio**. Pre-registered: ≤ 2.0 under H1, < 17.2 (collapse but not pass) under H2, otherwise H0. Historical: A1 1.76, Z1 0.84, Z2 17.2.
- **Per-channel obs symlog-MSE @ h=5**. Z2's Proprioception was the only FAIL (0.094 vs threshold 0.05); track for spillover.
- **Per-channel obs symlog-MSE @ long horizons** (h=25, 50). The GRU fix should also stabilise the latent → observation decoder over long horizons; secondary signal of mechanism.
- **Training-time `model_reward_mae_pos` and `model_reward_mae_neg`**. Z2 had `mae_neg` = 0.359 (the win) and `mae_pos` regressed to 0.805. If the GRU fix is genuinely a stability fix, both arms should move closer together, not further apart.
- **Continuation accuracy @ h=5, h=50**. Z2 was 1.000 / 0.990 — a good baseline; should stay at or above.
- **Survival (`Episode/Steps` steady-state)**. Z2 was 114.8, Z1 was 120.1, A1 was 112.0 — within noise on a single seed. Single-seed effect size is small; track for direction only.
- **WM training-time KL terms** (`L_dyn`, `L_rep`). If the reset gate genuinely changes the prior/posterior coupling, these should shift.

---

## §5 Out of scope

- **Other §6 candidates** — #30 (prior/posterior heads no hidden layer) and #29 (critic self-EMA regularisation) deferred per user instruction. Do not bundle.
- **GRU `update - 1` bias fix** (concept doc §9.11.1, fourth sub-item). Real deviation but lower-severity than the reset-gate fix; queue separately if needed after Z3 verdict.
- **Predator task re-run** — the cumulative stack (Z1 + Z2 + Z3) is the cleanest test bed for the full deviation cascade on predator's −100 death-penalty regime, but A2 (predator) had a different failure mode (~30-step floor) and a separate offline-diagnostic localisation. NoPred remains the cleanest single-knob test bed for *this* deviation in isolation. Predator validation is the natural follow-up plan after Z3's verdict.
- **Cumulative-fix matrix** — no factorial sweep across (zero-init × paper-bins × reset-gate × baseline). Single Z3 cell, cumulative on top of Z2; that's the scope.
- **Changes to the offline diagnostic script** — none required (verified via grep in §6); re-using the same script with no edits is methodologically required (the diagnostic is the measuring instrument).
- **Changes to any other config knob simultaneously** — entire scientific value of the comparison hinges on this being a single-knob change relative to Z2.
- **Follow-up plan queueing** — until Z3's offline diagnostic verdict comes back, do not queue plans for #29 / #30 / predator.
- **Updates to `docs/project/concepts/dreamer_v3_implementation.md` §6 item 28** — happens after Z3's verdict (this plan's summary becomes the "ACTED ON" entry under whichever hypothesis row fires).
- **Pre-existing config files missing the new mandatory key** (`dreamer_v3_probe.yaml` etc., per §2.3.3). Pre-existing breakage from the Z1 + Z2 cascade; bundle separately if needed.

---

## §6 Hand-off

Pre-hand-off integration check — direct GRU-cell instantiation outside the trainer:

```bash
grep -rn "LayerNormGRUCell\|ModulatedLayerNormGRUCell" \
    /media/nas01/projects/Interoceptive-AI/grid_world_pain/scripts/ \
    /media/nas01/projects/Interoceptive-AI/grid_world_pain/src/
```

Result (verified at plan-time): the cell classes are only constructed inside `src/models/dreamer_v3_nnx.py:54` and `src/models/dreamer_v3_nnx.py:56` (the RSSM constructor). The diagnostic script `scripts/dreamer_offline_wm_test.py` and any other script loads cells via the full `DreamerTrainer` checkpoint path — never via direct instantiation. So unlike Z2 (where the diagnostic decoded reward logits via a direct `from_twohot` call site that needed the new flag), **Z3 needs no diagnostic-script changes**. The flag flows: YAML → `config.get_mandatory(...)` in `DreamerV3Trainer.__init__` → `agent_config` dict → `WorldModel.__init__` → `RSSM.__init__` → cell `__init__`. One linear path.

After this plan lands and is committed:

1. **`developer`** reads §2 in full, applies the changes to:
   - `src/models/dreamer_v3_nnx.py` (LayerNormGRUCell class — new `apply_reset_gate` kwarg + gated branch; RSSM constructor — new `apply_gru_reset_gate` kwarg + pass-through to cell; WorldModel constructor — read `agent_config['apply_gru_reset_gate']` and forward to RSSM)
   - `src/models/modulated_layer_norm_gru_cell.py` (ModulatedLayerNormGRUCell — new `apply_reset_gate` kwarg + gated branch, mirrors the LN variant)
   - `src/models/dreamer_v3_trainer.py` (new entry in `agent_config` dict reading `config.get_mandatory('agent.apply_gru_reset_gate', bool)`)
   - `configs/models/dreamer_v3.yaml` (new key with full doc-link comment)
   - `configs/models/dreamer_v3_rr06.yaml` (new key with brief comment)

   Run the §2.5 smoke test (cell divergence + hand-computed reference match) and the §2.6 config end-to-end check. Fill out the Implementation Report below. Report back.

2. **`senior-developer`** verifies the implementation against §2 per the standard verification protocol (diff stat check, file-by-file diff, speed-change review at smoke-test budget, fill Verification Report below).

3. **`training-runner`** reads §3 verbatim and launches Z3 on n113:0 with the Launch Manifest bound at design time per `feedback_launch_manifest.md`. Standard pre-flight (env check), post-launch single-PID `pgrep` verification per `feedback_runner_post_launch_pgrep.md`, and diary `training-start` row.

4. **After Z3 finishes (~2.5 h)** the user or parent agent runs `scripts/dreamer_offline_wm_test.py` on the Z3 checkpoint and reads:
   - Reward MAE @ h=5 (primary) against §4's pre-registered table.
   - h50/h5 reward-MAE ratio (secondary mechanism check).
   The verdict is written back into §4 (or appended as a Verification Report below) by `experiment-analyzer`.

5. **Closing actions** (per §4 verdict):
   - **H1 fires**: promote default permanently; update §6 item 28 of the concept doc; close the reward-head investigation for NoPred; queue predator-task validation of the full cumulative stack.
   - **H2 fires**: queue candidate #30 (prior/posterior head capacity) as next plan; keep all four fixes ON cumulatively.
   - **H0 fires (split a/b based on h50/h5 ratio)**: keep the flag as default anyway (matches paper); investigate #30 + #29 next.

---

## Checkpoints (for `developer`)

- [x] `LayerNormGRUCell.__init__` accepts `apply_reset_gate: bool = False`; under `True`, `__call__` computes `cand = jnp.tanh(reset * cand)`; under `False`, `cand = jnp.tanh(cand)` (verbatim previous code). The `reset = nnx.sigmoid(reset)` line runs unchanged under both flags.
- [x] `ModulatedLayerNormGRUCell.__init__` mirrors the same flag with the same semantics; the `gate_bias` modulation branch on the update gate is untouched.
- [x] `RSSM.__init__` accepts `apply_gru_reset_gate: bool = False` and passes it to both cell constructors (modulated and non-modulated branches both updated).
- [x] `WorldModel.__init__` reads `config.get('apply_gru_reset_gate', False)` from `agent_config` and forwards it to `RSSM(...)`.
- [x] `DreamerV3Trainer.__init__` adds `'apply_gru_reset_gate': config.get_mandatory('agent.apply_gru_reset_gate', bool)` to the `agent_config` dict alongside the existing `paper_canonical_twohot_bins` entry; missing YAML key raises `ValueError` at trainer construction.
- [x] `configs/models/dreamer_v3.yaml` and `configs/models/dreamer_v3_rr06.yaml` both contain `apply_gru_reset_gate: true`.
- [x] §2.5 smoke test passes: (a) paper-flag and legacy-flag cell outputs diverge by max |Δ| > 1e-3 on the same input/state/weights (observed: 0.641342); (b) legacy-flag output matches hand-computed legacy reference within 1e-6; (c) paper-flag output matches hand-computed paper reference within 1e-6; (d) both assertions hold for `ModulatedLayerNormGRUCell` (with `gate_bias=None`).
- [x] §2.6 config end-to-end check passes (both YAML files expose the key; missing key raises `ValueError`).
- [x] When `apply_gru_reset_gate: false` is set, training output is bit-identical to pre-fix behaviour (verified by §2.4 reasoning — the `else` branch is verbatim previous code; PRNG consumption unchanged; op count identical except for one extra `jnp.multiply` only under `True`).
- [x] No edits made to the prior/posterior heads, critic loss, zero-init, two-hot bin grid, or the diagnostic script — those are out-of-scope (§5, §6 pre-hand-off grep confirmed only two instantiation sites, both in `RSSM.__init__`).
- [x] Speed sanity check at smoke-test scale: 500-call JIT-warmed forward pass shows paper-flag 107.2 µs/call vs legacy-flag 112.2 µs/call (−4.4%); well within ±5% noise floor; no regression.

---

## Implementation Report

> **Implemented by**: developer
> **Date**: 2026-05-11

### Files changed

| File | Change |
|---|---|
| `src/models/dreamer_v3_nnx.py` (`LayerNormGRUCell`) | Added `apply_reset_gate: bool = False` kwarg to `__init__`; stored as `self.apply_reset_gate`; added docstring. In `__call__`, added `if self.apply_reset_gate: cand = jnp.tanh(reset * cand)` branch; `else: cand = jnp.tanh(cand)` (verbatim previous code). |
| `src/models/dreamer_v3_nnx.py` (`RSSM`) | Added `apply_gru_reset_gate: bool = False` kwarg to `__init__`; passes `apply_reset_gate=apply_gru_reset_gate` to both `ModulatedLayerNormGRUCell(...)` and `LayerNormGRUCell(...)` constructors. |
| `src/models/dreamer_v3_nnx.py` (`WorldModel`) | Added `apply_gru_reset_gate = config.get('apply_gru_reset_gate', False)` read from agent_config before `self.rssm = RSSM(...)`, forwarded as `apply_gru_reset_gate=apply_gru_reset_gate`. |
| `src/models/modulated_layer_norm_gru_cell.py` | Added `apply_reset_gate: bool = False` kwarg to `__init__`; stored as `self.apply_reset_gate`; updated class docstring. In `__call__`, added `if self.apply_reset_gate: cand = jnp.tanh(reset * cand)` branch; `else: cand = jnp.tanh(cand)` (verbatim previous code). Gate_bias modulation branch on update gate is untouched. |
| `src/models/dreamer_v3_trainer.py` | Added `'apply_gru_reset_gate': config.get_mandatory('agent.apply_gru_reset_gate', bool)` to `agent_config` dict, right after the `paper_canonical_twohot_bins` entry. |
| `configs/models/dreamer_v3.yaml` | Added `apply_gru_reset_gate: true` with full 7-line doc-link comment, inserted after the `paper_canonical_twohot_bins` block and before `use_layer_norm`. |
| `configs/models/dreamer_v3_rr06.yaml` | Added `apply_gru_reset_gate: true` with brief back-reference comment, inserted after `paper_canonical_twohot_bins` and before `use_layer_norm`. |

### Deviations from §2

1. **Function-level default is `False` (not `True`)**: Matching Z2's Deviation 1. The plan's spec describes the semantics but the function-level default `False` is the safe choice: the trainer always passes the flag explicitly via `config.get_mandatory`; the `False` default keeps any direct-import tests or scripts bit-identical to pre-fix behaviour. No training-path effect — the trainer enforces `get_mandatory` and always passes the resolved value.

2. **`WorldModel` uses `config.get('apply_gru_reset_gate', False)` (not `get_mandatory`)**: Matching Z2's Deviation 2. The `agent_config` dict is a plain Python dict (not a Config object), so `get_mandatory` is not available on it. The trainer already enforces the mandatory constraint via `config.get_mandatory(...)` before building the dict, so the key is always present when `WorldModel.__init__` runs. The `False` fallback is a safety net only.

3. **Pre-existing config files** (`dreamer_v3_probe.yaml`, `dreamer_v3_curriculum.yaml`, etc.) were already broken under the Z1+Z2 mandatory-key set. This plan adds one more mandatory key to the same set. Out of scope per §2.3.3; flagged for senior-developer. The Z3 launch only needs `dreamer_v3_rr06.yaml`, which has the key.

### Smoke test output (§2.5) — verbatim

```
LayerNormGRUCell — same x, h, same weights, different flag:
  max |h_paper - h_legacy| = 0.641342
  mean|h_paper - h_legacy| = 0.110876
OK: paper-flag and legacy-flag outputs differ as expected; each matches its hand-computed reference; both cells.
```

All four assertions passed:
- `max |h_paper - h_legacy| = 0.641342` > 1e-3 (flag is wired in; typical O(0.6) on random hidden = 64 inputs — well above the O(0.1) expected per §2.5).
- Legacy-flag output matches hand-computed legacy reference within 1e-6 (`cand = tanh(cand)` path).
- Paper-flag output matches hand-computed paper reference within 1e-6 (`cand = tanh(reset * cand)` path).
- `ModulatedLayerNormGRUCell` (gate_bias=None) exhibits the same divergence pattern (same assertion, different cell class).

### Config end-to-end check (§2.6)

```
configs/models/dreamer_v3.yaml -> agent.apply_gru_reset_gate = True  (type: bool)
configs/models/dreamer_v3_rr06.yaml -> agent.apply_gru_reset_gate = True  (type: bool)
OK: both YAML files expose the mandatory key correctly.
OK: missing key raises ValueError: Strict Config: Configuration key 'agent.apply_gru_reset_gate' is required but missing.
```

Trainer dry-construction paranoia check:
```
Trainer construction OK — apply_gru_reset_gate read without error.
  RSSM cell type: LayerNormGRUCell
  apply_reset_gate on cell: True
```
The flag flows correctly from YAML → `get_mandatory` → `agent_config` → `WorldModel` → `RSSM` → `LayerNormGRUCell.apply_reset_gate = True`.

### Speed check

**Hardware**: CPU (JAX platform=cpu; NAS node, Intel). **Cell**: `LayerNormGRUCell` with `hidden=512` (production deter_dim), `B=32`, JIT-compiled. **Method**: 500 forward calls timed after 3 warm-up calls, averaged.

| Flag | us/call |
|---|---|
| `apply_reset_gate=True` (paper-canonical) | 107.2 |
| `apply_reset_gate=False` (legacy) | 112.2 |
| Delta | −4.4% (paper-flag is faster, within noise) |

**Verdict**: no regression. The single `jnp.multiply(reset, cand)` added in the True branch is dominated by the existing LayerNorm + Linear ops. Delta is well within the ±5% noise floor on CPU timing; the change is hot-path neutral as predicted by §2.4.

### Blockers

None. Implementation complete.

Pre-existing config breakage (Deviation 3): 5 other dreamer YAML files will raise `ValueError` at trainer construction if loaded (missing `apply_gru_reset_gate`, same as for `zero_init_reward_critic` and `paper_canonical_twohot_bins` from Z1/Z2). Already broken pre-Z3; flagged for senior-developer cleanup plan, out of Z3 scope.

Implemented by: developer

---

## Verification Report

> **Verified by**: <senior-developer post-impl OR experiment-analyzer post-Z3-diagnostic>
> **Date**: <to fill>
> **Run**: Z3 — `dreamer_gru_NoPred_rr06_s0_n113` / WandB `<id>` / checkpoint step <step>
> **Diagnostic outputs**: <tmp paths>

### V.1 Headline

<one-paragraph H1/H2/H0 verdict; mirror Z2's V.1>

### V.2 Pre-registered hypothesis check (plan §4)

<table mirroring Z2's V.2>

### V.3 A1 vs Z1 vs Z2 vs Z3 — pre-registered metric table

<table mirroring Z2's V.3; include per-horizon reward MAE row and h50/h5 ratio>

### V.4 Mechanism check — long-horizon error compounding (load-bearing for the §1 hypothesis)

<table comparing per-horizon reward MAE A1 / Z1 / Z2 / Z3 and h50/h5 ratio; interpret per §4.1>

### V.5 Implementation surface check (per Implementation Report)

<one-row-per-file table mirroring Z2's V.5>

### V.6 Metrics Requested

<list any follow-up metric or diagnostic that needs senior-developer to pick up; mirror Z2's V.6 (which surfaced the diagnostic-script bin-flag bug)>

### V.7 Interpretation

<plain-language 3–4 paragraph interpretation; mirror Z2's V.7>

### V.8 Next-step flags (advisory only — NOT spawning here)

<list candidate next rungs per §6 step 5 closing actions>

---

<!--
NEW ISSUES: any deviation discovered during implementation/verification that warrants its own plan should be split out (one of the deferred §6 candidates) rather than expanded inline here.
-->
