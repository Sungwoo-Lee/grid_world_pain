# tests/algorithms/dreamer_srl/ — Lever-A bit-identity test suite

This directory holds the per-function bit-identity tests for the dreamer-srl v3
JAX rebuild. Each test asserts that the JAX implementation agrees with the
vendored sheeprl@33b6366 reference to within `1e-6` max-absolute-difference
(Lever A in [IMPLEMENTATION_PLAN.md](../../../docs/develop/active/dreamer_srl_v1/IMPLEMENTATION_PLAN.md)).

## Test naming convention

```
test_<file>.py::test_<function>_matches_sheeprl()
```

For example:
- `test_loss.py::test_twohot_encode_matches_sheeprl`
- `test_utils.py::test_compute_lambda_values_matches_sheeprl`
- `test_agent.py::test_layernorm_gru_cell_matches_sheeprl`

Each test file is paired with one source file under
`src/algorithms/dreamer_srl/`.

## Lever-A gate rule

A function is **not marked done** in the checkpoint table until its paired
test passes at the `1e-6` threshold. Tests that pass before and after a
change are not testing what you think — verify that a pre-fix run fails.

## Test structure (template)

```python
def test_<function>_matches_sheeprl():
    import numpy as np
    import torch
    import jax.numpy as jnp

    # sheeprl (vendor) side
    from vendor.sheeprl.sheeprl.<path> import <SheeprlClass>

    # JAX (dreamer-srl) side
    from src.algorithms.dreamer_srl.<file> import <JaxFunction>

    fixture = np.load("tests/fixtures/dreamer_srl/<function>_input.npz")
    # ... load inputs from fixture ...

    # Run torch side
    torch_out = <SheeprlClass>(...).detach().numpy()

    # Run JAX side
    jax_out = <JaxFunction>(...)

    # Assert bit-identity
    max_abs_diff = float(jnp.max(jnp.abs(jnp.asarray(torch_out) - jax_out)))
    assert max_abs_diff < 1e-6, f"<function> deviates by {max_abs_diff:.3e}"
```

## File layout (grows per CP)

```
tests/algorithms/dreamer_srl/
├── __init__.py
├── README.md                 # this file
├── test_utils.py             # CP1: symlog, symexp, init_weights, compute_lambda_values, Moments, Ratio, prepare_obs
├── test_buffers.py           # CP3b: buffer state-evolution parity + cadence trace (state-evolution bit-identity, not pure-function — see CP3B_SPEC.md)
├── test_agent.py             # CP2 (LayerNormGRUCell), CP3 (build_agent zero-init heads), CP4 (RSSM + get_initial_states), CP4b (is_first reset)
├── test_loss.py              # CP5: TwoHotEncoding, Symlog, MSE, BernoulliSafeMode, reconstruction_loss
└── test_train.py             # CP2b (action_shift), CP6 (critic loss), CP7 (Polyak update)
```

## Paired diff tool

Each test is mirrored by the CLI diff tool at `scripts/sheeprl_jax_diff.py`:
```bash
python scripts/sheeprl_jax_diff.py \
    --function twohot_encode \
    --fixture tests/fixtures/dreamer_srl/twohot_encode_input.npz \
    --threshold 1e-6
```

The diff tool is used by reviewers at each CP gate to produce the PASS/FAIL
table in the review doc (`docs/develop/active/dreamer_srl_v1/review_code_CP<N>.md`).

## Deviation log

If a test cannot pass at `1e-6` (e.g. JAX-platform float32/float64 mismatch),
the developer logs the deviation in
[DEVIATION_LOG.md](../../../docs/develop/active/dreamer_srl_v1/DEVIATION_LOG.md)
and waits for PI sign-off before marking the function done.
