# tests/fixtures/dreamer_srl/ — Lever-A fixture files

This directory holds the `.npz` fixture files used by the bit-identity tests in
`tests/algorithms/dreamer_srl/`. All fixtures are committed to the repo so any
developer or reviewer can reproduce the test without re-running the generation script.

## Fixed PRNG seed

**Seed: `0xD3EAF` (decimal 868591)**

Every fixture is generated with this seed — both for the NumPy random state
(fixture construction) and the PyTorch `torch.manual_seed()` call that
produces the sheeprl reference outputs. Use this seed consistently so any
developer can regenerate a fixture and get the same bytes.

## Fixture naming convention

```
<function-name>_input.npz
```

For example:
- `twohot_encode_input.npz`
- `layernorm_gru_input.npz`
- `compute_lambda_values_input.npz`

Where `<function-name>` matches the key in `scripts/sheeprl_jax_diff.py::FUNCTION_REGISTRY`
and the test function name in `tests/algorithms/dreamer_srl/test_<file>.py`.

## Storage format

All fixtures use `np.savez_compressed` with named keys for each tensor argument:

```python
np.savez_compressed(
    "tests/fixtures/dreamer_srl/twohot_encode_input.npz",
    logits=logits_np,   # [T=16, B=4, 255] float32
    target=target_np,   # [T=16, B=4, 1]   float32
    seed=np.array(0xD3EAF),
)
```

Load in tests with `np.load(path)["logits"]` etc.

## Fixture generation scripts

Each CP gets a deterministic generation script:

```
scripts/fixtures/
├── gen_cp1_fixtures.py   # CP1: symlog, symexp, compute_lambda_values, Moments, Ratio, prepare_obs
├── gen_cp2_fixtures.py   # CP2: LayerNormGRUCell, action_shift
├── gen_cp3_fixtures.py   # CP3: zero-init heads
├── gen_cp4_fixtures.py   # CP4/CP4b: RSSM, is_first reset
├── gen_cp5_fixtures.py   # CP5: TwoHotEncoding
├── gen_cp6_fixtures.py   # CP6: critic loss
├── gen_cp7_fixtures.py   # CP7: Polyak update
└── gen_cp8_fixtures.py   # CP8: end-to-end forward parity
```

A reviewer can regenerate all fixtures for a CP and verify the `.npz` bytes
match what is committed:
```bash
python scripts/fixtures/gen_cp5_fixtures.py
# Re-creates tests/fixtures/dreamer_srl/twohot_encode_input.npz etc.
# Should produce identical bytes (deterministic via seed 0xD3EAF).
```

## Representative shapes

The fixture shapes match what the function sees in a standard training run
with `batch_size=4`, `seq_len=16`, `stoch_dim=32`, `discrete=32`,
`deter_dim=512`, `action_dim=4`, `obs_dim=76`:

| Function | Key tensors | Shape |
|---|---|---|
| `twohot_encode` | `logits`, `target` | `[16, 4, 255]`, `[16, 4, 1]` |
| `compute_lambda_values` | `rewards`, `values`, `continues` | `[16, 4]` each |
| `layernorm_gru` | `x`, `h` | `[4, 512]`, `[4, 512]` |
| `moments_update` | `x`, `low`, `high` | `[16, 4]`, scalar, scalar |
| `rssm_transition` | `stoch`, `action`, `h_prev` | `[4, 1024]`, `[4, 4]`, `[4, 512]` |
