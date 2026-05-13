---
title: "dreamer-srl v3 — CP2 + CP2b code review"
topic: dreamer
status: active
reviewer: code-reviewer
created: 2026-05-14
last_updated: 2026-05-14
---

# dreamer-srl v3 — CP2 + CP2b code review

## Verdict at a glance

This review checks the JAX port of two pieces of the sheeprl Dreamer-v3 recipe:
(a) the LayerNorm GRU recurrent cell — a known historical trap where the reset
gate must multiply the candidate state **inside** `tanh`, not outside; and
(b) a one-line action-shift function that wires actions[t-1] (the action that
produced obs_t) into the world-model rollout at time t. Both are landed in a
single file `src/algorithms/dreamer_srl/agent.py`, both pass their bit-identity
tests against the vendored sheeprl@33b6366 reference, and the full 21-test
checkpoint suite stays green.

The reset-before-tanh trap (the v2 plan's "cascade fix #28" — the recurring
silent-pattern-match bug class that this whole rebuild was launched to prevent)
is implemented correctly and is genuinely exercised by the fixture (reset gate
takes values across 0.11 to 0.91, mean 0.55, 98.4% of values in the active
[0.1, 0.9] band). The action-shift function is a pure 2-line concatenate.
The new deviation D-007 (5e-4 threshold relaxation) is documented but has not
yet been signed off by the PI.

**Verdict:** ⚠ **PASS WITH FIX** — the CP2 code is correct as written and all
gates pass, but a **latent semantic deviation** in the LayerNorm epsilon will
silently affect CP4 wire-up if not fixed now. The fix is small (one
constructor argument); the issue is **not blocking** for the CP2 gate review
chain (math-reviewer can fire next), but it must be resolved before CP4
implementation. See finding 🟡-1 below.

## Scope of changes

```
df9c328 feat(dreamer-srl): ✨ CP2 + CP2b — port LayerNormGRUCell + action_shift
 .../dreamer_srl_v3/DEVIATION_LOG.md      |   3 +-
 .../dreamer_srl_v3/IMPLEMENTATION_PLAN.md|  74 +++++++-
 scripts/fixtures/gen_cp2_fixtures.py     | 193 +++++++++++++++++ (new)
 scripts/sheeprl_jax_diff.py              |  90 +++++++++
 src/algorithms/dreamer_srl/agent.py      | 182 +++++++++++++++++ (new)
 tests/algorithms/dreamer_srl/test_agent.py| 196 +++++++++++++++++ (new)
 tests/fixtures/dreamer_srl/*.npz         | 2 binary fixtures (new)
```

The change is well-scoped: one new module file, one new test file, one new
fixture-generation script, two new fixtures, and three minor edits (plan
report appended, deviation row added, diff-tool registry extended).

## Findings

| ID | Severity | File:line | Issue | Suggested fix |
|---|---|---|---|---|
| 🟡-1 | concern (latent) | `src/algorithms/dreamer_srl/agent.py:88-91` | `nnx.LayerNorm` default `epsilon=1e-6`, but sheeprl's production call site uses `eps=1e-3` (`vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L305-L306`, `layer_norm_kw = {"eps": 1e-3}`). The cell hardcodes the JAX default with no `epsilon` argument exposed. Measured forward-pass difference between eps=1e-6 and eps=1e-3 on the CP2 fixture is **1.72e-3** — 3.4× the D-007 relaxed threshold and ~6× the observed D-007 drift. The CP2 test happens to pass because its fixture uses `layer_norm_kw={}` (PyTorch's `nn.LayerNorm` default eps=1e-5, which is close to JAX's 1e-6), but the production wire-up at CP4 will silently lose parity. | Add `eps: float = 1e-3` (or accept it via constructor / a layer_norm_kw-style dict) to `LayerNormGRUCell.__init__` and pass it to `nnx.LayerNorm(epsilon=eps)`. Default should be **1e-3** to match the sheeprl production setting, not the JAX `nnx.LayerNorm` default. Optionally re-generate the CP2 fixture with the same `eps=1e-3` so the test exercises the production configuration. |
| 🟢-2 | nit | `src/algorithms/dreamer_srl/agent.py:115-122` | Chunk-split uses `z[..., :H]`, `z[..., H:2*H]`, `z[..., 2*H:]` — semantically identical to `torch.chunk(x, 3, -1)` and chunk-order `(reset, cand, update)` matches sheeprl L399. No bug. Inline comment is excellent (line 119 calls out the chunk order). | None — informational. The hand-coded split is arguably clearer than `jnp.split` and avoids the array-vs-tuple ambiguity. Keep as-is. |
| 🟢-3 | nit | `scripts/fixtures/gen_cp2_fixtures.py:79-80` | Fixture uses `layer_norm_cls=nn.LayerNorm, layer_norm_kw={}`, leaving PyTorch eps at its default 1e-5 — not the sheeprl production value 1e-3. This is consistent with the fixture's purpose (exercising the trap, not the production config), but it would be more defensible if the fixture pinned `layer_norm_kw={"eps": 1e-3}` to mirror real wire-up. | After fixing 🟡-1, optionally re-generate the fixture with `layer_norm_kw={"eps": 1e-3}` so the bit-identity test covers the production setting. Re-running `gen_cp2_fixtures.py` is deterministic (seed 0xD3EAF). |
| 🟢-4 | nit | `tests/algorithms/dreamer_srl/test_agent.py:179-186` | Structural assertions for `action_shift` test only `T=5`. T=1 (no prior action) and T=2 (minimum non-trivial shift) edge cases are not covered. I manually verified T=1 → output[0]=0 of shape (1,...) and T=2 → output[0]=0, output[1]=input[0], both correct via `jnp.concatenate` arithmetic. | Optional — add a `pytest.mark.parametrize` over T ∈ {1, 2, 5} for the structural assertion block, or document in the test docstring that T=1/T=2 are out of scope. Not blocking. |
| 🟢-5 | nit | `src/algorithms/dreamer_srl/agent.py:108` | The local `H = self.hidden_size` is good style for readability. No issue; flagging to commend the explicit aliasing — it matches the v2 plan's "make critical chunk offsets visible" guidance. | None — informational. |

No 🔴 blockers.

## The 8 verification points — point-by-point

### 1. Reset-before-tanh discipline (cascade fix #28) — ✅ confirmed

`src/algorithms/dreamer_srl/agent.py:126` reads:
```python
cand   = jnp.tanh(reset * cand_proj)
```
The reset gate (post-sigmoid) multiplies the candidate **projection** before
`tanh` is applied — matching sheeprl `models.py:L401`:
```python
cand = torch.tanh(reset * cand)
```
The structural test catches the inverted form via the fixture: reset stats
across the 4×16 = 64 output positions are mean=0.5457, min=0.1100, max=0.9099,
std=0.1939, with 98.4% of values in the (0.1, 0.9) "trap-active" band. If a
future commit inverts the order to `reset * tanh(cand_proj)`, the deviation
would be O(0.1) per element — 336× the relaxed D-007 threshold, well above
both the 1e-6 default and the 5e-4 D-007 ceiling. Trap is genuinely loud.

The docstring at lines 38-49 explicitly calls out the GOTCHA with both
correct and incorrect forms side-by-side, and the inline comment at line 125
re-asserts it at the line of action. Defense-in-depth on the discipline is
strong.

### 2. Fixture genuineness — ✅ confirmed

`scripts/fixtures/gen_cp2_fixtures.py:51` imports the actual sheeprl class:
```python
from sheeprl.models.models import LayerNormGRUCell
```
Then instantiates it (line 73-81), runs the forward pass under
`torch.no_grad()` (line 113), and stores the output to `torch_out_np` (line
114). The fixture is a real PyTorch→numpy snapshot, not a JAX self-comparison.

I re-ran the diagnostic at the bottom of the file (lines 100-109) inside the
grid_world_pain env (using a numpy-only LayerNorm reconstruction) and the
reset stats reported above match the script's claim of "reset ≈ 0.55". The
sanity check line that prints `reset mean (post-sigmoid)` in the generator
ensures any future fixture regeneration will catch a degenerate (all-near-0
or all-near-1) reset gate before the file is written.

Action-shift fixture (T=5, B=4, A=3) uses
`torch.cat((torch.zeros_like(actions[:1]), actions[:-1]), dim=0)` directly —
the literal sheeprl line 104. Asserts structural properties (line 170-172)
before writing. Pure-arithmetic fixture; no float drift possible.

### 3. `action_shift` exactness — ✅ confirmed

`src/algorithms/dreamer_srl/agent.py:181-182`:
```python
zeros = jnp.zeros_like(actions[:1])   # [1, B, A]
return jnp.concatenate([zeros, actions[:-1]], axis=0)   # [T, B, A]
```
Matches sheeprl `dreamer_v3.py:L102-L104` line-for-line in semantics. Test
runs with T=5 and confirms max_abs_diff = 0.0 exact-equality. Structural
checks (line 179-187) verify shape, output[0]==0, output[1:]==input[:-1]
independently of the byte-level comparison.

Edge cases verified manually (T=1 → (1,B,A) of zeros; T=2 → output[0]=0,
output[1]=input[0]) — both correct via the arithmetic. T=0 is degenerate
(empty array in, empty array out) and out of scope. Recommend adding a
T-parametrized check (🟢-4) if there's appetite for it, but not blocking.

### 4. D-007 cascade plausibility — ✅ confirmed; threshold policy consistent

Observed: 2.97e-4 on the LayerNormGRUCell forward pass. Float64 reference
gives 1.85e-7 vs PyTorch (per the deviation log), confirming the issue is
purely float32 accumulation order and not semantic.

Threshold-policy comparison:

| Deviation | Observed | Threshold | Margin |
|---|---|---|---|
| D-003 (`symexp`) | 1.526e-5 | 2e-5 | 1.3× |
| D-006 (`twohot_log_prob`) | 1.812e-5 | 3e-5 | 1.7× |
| **D-007 (this CP)** | **2.97e-4** | **5e-4** | **1.7×** |

D-007's 1.7× margin is the same as D-006, both wider than D-003's 1.3×.
This is principled — D-007 has the largest absolute magnitude of the three,
and the cascade through LayerNorm + sigmoid + tanh adds amplification that
1e-6-precision-class drifts don't see. A 5e-4 ceiling still leaves 336×
headroom to the reset-trap O(0.1) detection. Policy is internally consistent.

I separately verified by enabling a side-by-side `eps=1e-5` (PyTorch's
default — matching the fixture's `layer_norm_kw={}` setting) and the
deviation dropped from 2.975e-4 to 2.953e-4 — only a 7% drop. The dominant
contribution (~99%) is genuinely matmul accumulation order, not LayerNorm
epsilon. D-007's diagnosis is correct.

**However:** see finding 🟡-1. The LayerNorm epsilon difference becomes a
real concern when the CP2 cell is wired into the production network at CP4
(sheeprl uses eps=1e-3 there). That's not a D-007 issue per se; it's a
forward-looking gap in the cell's API.

### 5. NNX_CONVENTIONS adherence — ✅ confirmed

- `LayerNormGRUCell` is correctly an `nnx.Module` (line 32). It holds
  learnable params (`nnx.Linear`, `nnx.LayerNorm`) — both standard nnx
  network layers, not a pytree-state struct. ✅
- `__init__` takes `rngs: nnx.Rngs` as the last positional argument
  (line 70). ✅
- `rngs` is used only during `__init__` (passed to the two sub-layers) and
  **not stored** on the module. ✅
- Forward pass `__call__(x, hx)` is pure-functional, no randomness, no
  hidden state — matches the convention (no `rngs` in `__call__`). ✅
- No `@nnx.jit` or `nnx.split/merge` usage in this module — correct, this
  is a building-block cell, not a training method. JIT will be wired at
  CP4 when the cell becomes part of the RSSM scan. ✅
- Hafner truncated-normal init: the cell does **not** pre-apply `init_weights`
  to its `nnx.Linear` and `nnx.LayerNorm`. This is **correct** — sheeprl's
  `LayerNormGRUCell` itself relies on PyTorch's default `nn.Linear` /
  `nn.LayerNorm` init, and the Hafner init is applied externally via
  `.apply(init_weights)` at the world-model construction site
  (`sheeprl/algos/dreamer_v3/agent.py:L1058`, `recurrent_model.apply(init_weights)`).
  The CP4 wire-up is the right place to apply the Hafner init via
  `init_weights` from `src/algorithms/dreamer_srl/utils.py`. Verified — no
  premature init at CP2. ✅

### 6. Isolation rule (v2 Risks §13) — ✅ confirmed

```
$ grep -rn "from src.models" src/algorithms/dreamer_srl/
src/algorithms/dreamer_srl/agent.py:7:    This module does NOT import from src.models.dreamer_v3_* or any other
```
The only match is the docstring note **declaring** the isolation rule, not
violating it. Zero imports from `src/models/`. ✅

### 7. Lever-B citation headers — ✅ confirmed

- `LayerNormGRUCell` class docstring (line 35-36): `sheeprl@33b6366:sheeprl/models/models.py:L331-L410` — matches the actual class span (L331 `class LayerNormGRUCell(nn.Module):` to L410 `return hx`). ✅
- `LayerNormGRUCell.__call__` docstring (line 103-104): `L370-L410` (the
  `forward` method body). Matches actual `def forward` at L370 to `return hx`
  at L410. ✅
- `action_shift` docstring (line 143): `sheeprl/algos/dreamer_v3/dreamer_v3.py:L102-L104`. Matches the `batch_actions = torch.cat(...)` statement
  (the assignment statement spans those three lines). ✅
- `vendor/sheeprl/sheeprl/models/models.py` total file length is 525 lines;
  `dreamer_v3.py` is 780 lines — both citations are valid in-range. ✅

Inline comments at the line of action (`Sheeprl L396`, `Sheeprl L397-L398`,
`Sheeprl L399-L403`) cross-reference the specific source lines being mirrored.
This is exemplary Lever-B discipline — better than just the function-level
header citation.

### 8. Diff-tool registry — ✅ confirmed

`scripts/sheeprl_jax_diff.py`:
- `FUNCTION_REGISTRY` has `"layernorm_gru_cell"` → `_run_layernorm_gru_cell`
  (line 952) and `"action_shift"` → `_run_action_shift` (line 953). ✅
- `FUNCTION_THRESHOLDS` has `"layernorm_gru_cell": 5e-4` (line 984) with
  citation to D-007 in the comment. `action_shift` is not in
  `FUNCTION_THRESHOLDS` — correct, it uses the default 1e-6 (exact
  equality). ✅
- `CHECKPOINT_REGISTRY` already had `"CP2": ["layernorm_gru_cell"]` and
  `"CP2b": ["action_shift"]` from earlier scaffolding (line 1004-1005).
  No change needed here. ✅

End-to-end diff-tool run confirms both PASS:
```
$ python scripts/sheeprl_jax_diff.py --checkpoint CP2
  layernorm_gru_cell  PASS (max_abs_diff=2.975e-04 < 5.0e-04)
$ python scripts/sheeprl_jax_diff.py --checkpoint CP2b
  action_shift        PASS (max_abs_diff=0.000e+00 < 1.0e-06)
```

The pytest suite also re-confirms:
```
tests/algorithms/dreamer_srl/test_agent.py::test_layernorm_gru_cell_matches_sheeprl PASSED
tests/algorithms/dreamer_srl/test_agent.py::test_action_shift_matches_sheeprl       PASSED
```

## Conventions audit checklist

This is a JAX/NNX bit-identity port, not a `src/environment/` change — the
classic environment-conventions checklist (pytree, vmap, PRNG, sensor sync,
config protocol) is **not in scope** for this review. The dreamer-srl-specific
adapted checklist:

- [x] **Isolation (v2 Risks §13):** no imports from `src/models/dreamer_v3_*`. ✅
- [x] **NNX convention 1 (Rngs in `__init__`, last positional):** ✅
- [x] **NNX convention 1 (Rngs not stored, not in `__call__`):** ✅
- [x] **NNX convention 2 (forward pass takes explicit `key`, or is pure):**
      ✅ (forward is pure — no PRNG needed)
- [x] **NNX convention 3 (no `nnx.split/merge` inside `nnx.Module`):** ✅
- [x] **NNX convention 4 (Hafner init applied at wire-up, not inside cell):**
      ✅ — deferred to CP4 correctly
- [x] **Lever-A bit-identity test paired (1+1 = 2 tests):** ✅
- [x] **Lever-B citation header + line range + GOTCHA paragraph:** ✅
- [x] **Lever-B inline comments at line of action:** ✅
- [x] **Lever-D diff-tool registry + threshold:** ✅
- [x] **Lever-E deviation logged (D-007):** ✅
- [ ] **D-007 PI sign-off:** ☐ pending (intentional — gate task)
- [⚠] **LayerNorm epsilon configurable & defaulted to sheeprl production
      value (1e-3):** ❌ currently hardcoded to JAX default 1e-6. See 🟡-1.
- [x] **No regressions: full 21-test suite passes:** ✅

## Recommendations to the chain

1. **Math-reviewer:** verify the GRU gate formulas independently against the
   GRU literature and confirm sheeprl-vs-dreamerv2 chunk-order conventions
   align (some GRU implementations use `(update, reset, cand)` order; sheeprl
   uses `(reset, cand, update)` which the JAX port matches). Confirm the
   `update_proj - 1` bias shift (line 128) is mathematically what the
   dreamer-v2 paper / sheeprl prescribes, not a typo.

2. **Math-reviewer / professor:** sanity-check 🟡-1. If the cell will be
   used with `eps=1e-3` at CP4 wire-up, the 1.7e-3 forward-pass shift from
   the eps mismatch matters more than D-007's 2.97e-4. It's not a CP2-blocker
   (the test passes with the fixture as-shipped), but professor-rl-bayesian-dl
   should call out that the CP4 wire-up should regenerate the CP2 fixture at
   `eps=1e-3` and re-run the diff so the test exercises the production config.

3. **PI:** D-007 sign-off is queued for the CP2 gate. The deviation rationale
   in DEVIATION_LOG.md is sound (float64 validation at 1.85e-7 confirms pure
   accumulation-order drift). Recommend approve with the standard pattern
   already used for D-003 (CP1, `symexp`) and D-006 (CP5, `twohot_*`).

## Conclusion

CP2 + CP2b implementation is correct. The reset-before-tanh discipline
(cascade fix #28 — the historical-trap class this whole rebuild was launched
to prevent) is matched line-for-line, the fixture genuinely exercises it at
~98% trap-active reset coverage with 336× margin to the trap signature, and
the diff tool + pytest suite + 21-test full-CP suite all pass. D-007 is
documented and queued for PI sign-off with a threshold policy consistent with
D-003 and D-006. The one latent concern (LayerNorm epsilon mismatch with
sheeprl production setting) does not block CP2 closure but must be addressed
before CP4 wire-up.

**Verdict: ⚠ PASS WITH FIX** — math-reviewer may proceed; the eps fix can be
applied as a follow-up commit on v1.4 before CP4 begins.

Reviewed by: code-reviewer
