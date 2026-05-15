---
title: "dreamer-srl v3 CP5 — code-reviewer audit (historical-scar checkpoint)"
topic: dreamer
status: active
reviewer: code-reviewer
created: 2026-05-14
last_updated: 2026-05-14
---

# dreamer-srl v3 CP5 — code-reviewer audit

## Plain-language verdict

This review covers the third algorithmic checkpoint (CP5) of the dreamer-srl v3
rebuild: the port of sheeprl's `TwoHotEncodingDistribution` (the 255-bin grid
that the world-model's reward and critic heads use to predict scalar values as
a soft histogram) to `src/algorithms/dreamer_srl/loss.py` as the `TwoHotEncoding`
class. CP5 is the **historical-scar checkpoint** — the v1 of the in-house
DreamerV3 stored the bin grid in *real reward space* (i.e., the linspace was
wrapped with `symexp`) while sheeprl stores it in *symlog space*; the two
implementations train to different basins, losses drop in both, and three
reviewers reading a 1056-line static plan never caught the divergence. The
five-lever guardrail stack (per-function bit-identity tests, source citations,
3-reviewer gate, vendored sheeprl diff tool, deviation log) was built to catch
exactly this bug class.

I audited the new `loss.py` module (one class, four methods), its 5 paired
tests (3 Lever-A + 2 structural), the `gen_cp5_fixtures.py` generator, the
three new `_run_twohot_*` runners in `sheeprl_jax_diff.py`, and the new
deviation-log entry **D-006** (JAX `jnp.linspace` vs PyTorch `torch.linspace`
producing a 1-ULP difference at `bins[127]`, the midpoint).

**The historical bug is structurally prevented.** `grep "symexp(self.bins)"`
on `loss.py` returns empty; `self.bins = jnp.linspace(low, high, n_bins)` at
line 111 is the bare linspace with no symexp wrap; the only `symexp` calls in
the module are at lines 134 and 146 — inside the `mean`/`mode` properties at
consumption time, exactly as sheeprl does. The triple-consistency contract
(v2 plan cascade row, v3 plan CP5 row, this file's class docstring) is
preserved. The `test_bins_not_symexp_at_storage` test would catch a future
regression: if a developer wrapped the linspace in `symexp`, `bins[0]` would
become `-4.85e8` and the test's `abs(bins[0] - (-20.0)) < 1e-6` assertion
would fire loudly. **Verdict: PASS.** One nit on a stale citation in a
docstring; no blockers. Math-reviewer can begin.

## Critical assertion verification — the load-bearing CP5 audit

### Symlog-space discipline triple-consistency

| Site | File | Line | What it says | Verdict |
|---|---|---|---|---|
| Storage | `src/algorithms/dreamer_srl/loss.py` | 111 | `self.bins = jnp.linspace(low, high, logits.shape[-1])` — **no symexp wrap** | ✅ |
| Consumption (mean) | `src/algorithms/dreamer_srl/loss.py` | 134 | `return symexp(jnp.sum(self.probs * self.bins, axis=self.dims, keepdims=True))` | ✅ |
| Consumption (mode) | `src/algorithms/dreamer_srl/loss.py` | 146 | `return symexp(jnp.sum(self.probs * self.bins, axis=self.dims, keepdims=True))` | ✅ |
| log_prob (target) | `src/algorithms/dreamer_srl/loss.py` | 192 | `x = symlog(x)` — target symlog-encoded BEFORE bin lookup | ✅ |
| v3 plan CP5 row | `docs/develop/active/dreamer_srl_v1/IMPLEMENTATION_PLAN.md` | L522 | "`bins[0]=-20, bins[127]≈0, bins[254]=+20`, in symlog space" | ✅ |
| v2 archived plan cascade row #2 | `docs/develop/archive/dreamer_srl/IMPLEMENTATION_PLAN.md` | L82 | "stored in *symlog space*" with explicit warning DO NOT `symexp(linspace)` | ✅ |
| Sheeprl source (vendored) | `vendor/sheeprl/sheeprl/utils/distribution.py` | 237 | `self.bins = torch.linspace(low, high, logits.shape[-1], device=logits.device)` — bare linspace, no transbwd | ✅ |

All seven sites agree. The grep verification:

```
$ grep -n "symexp(self.bins)" src/algorithms/dreamer_srl/loss.py
(no output)

$ grep -n "self\.bins =" src/algorithms/dreamer_srl/loss.py
111:        self.bins = jnp.linspace(low, high, logits.shape[-1])
```

The only `symexp` occurrences in `loss.py` are at lines 134 and 146 (mean/mode
properties, consumption side). There are **no** `symexp(self.bins)` or
`self.bins = symexp(...)` patterns anywhere in the file. The historical
bug is structurally prevented.

### `test_bins_not_symexp_at_storage` genuineness check

The test (test_loss.py:301-324) asserts:

```python
assert abs(float(jax_dist.bins[0]) - (-20.0)) < 1e-6
assert abs(float(jax_dist.bins[254]) - 20.0) < 1e-6
```

**Mental mutation test.** If a future developer changes line 111 of `loss.py`
from `self.bins = jnp.linspace(low, high, ...)` to
`self.bins = symexp(jnp.linspace(low, high, ...))`:

- `symexp(-20) = sign(-20) * (exp(20) - 1) ≈ -4.85e8`
- The test's `abs(-4.85e8 - (-20.0)) ≈ 4.85e8` is many orders of magnitude
  above the `1e-6` threshold.
- The test FAILS LOUDLY with the documented error message that explicitly
  flags "If this is ~-485165195, symexp was incorrectly applied at storage
  — that is the historical bug."

**The test is genuine — not a false-PASS.** The mutation a developer would
plausibly make (wrapping the linspace in `symexp`) is caught at a 14-order-
of-magnitude margin. The redundancy with `test_twohot_bins_endpoints_match_sheeprl`
is deliberate (the docstring at test_loss.py:308-309 explicitly calls this out:
"This test is structurally redundant with test_twohot_bins_endpoints_match_sheeprl
but makes the historical-bug-prevention intent explicit"). For the historical-
scar checkpoint, the redundancy is the point.

## Per-test audit table

| # | Test | Bit-identity real? | Source citation OK? | JAX correctness | Issues |
|---|------|--------------------|---------------------|-----------------|--------|
| 1 | `test_twohot_bins_endpoints_match_sheeprl` | YES — fixture-gen calls `TwoHotEncodingDistribution(dummy_logits, dims=0)` on the vendored sheeprl side (gen_cp5_fixtures.py:73-75) and stores `torch_bins` (sheeprl's actual 255-element float32 array) in the `.npz`. Test compares JAX `jax_dist.bins` against stored `torch_bins` at threshold `THRESHOLD_D006=3e-5`. | YES — sheeprl L237 cited at test_loss.py:84 and in `_run_twohot_bins_endpoints` runner at scripts/sheeprl_jax_diff.py:755. Verified by direct file inspection: sheeprl L237 is `self.bins = torch.linspace(low, high, logits.shape[-1], device=logits.device)` — exact match. | Correct — JAX uses `jnp.linspace(low, high, logits.shape[-1])` (loss.py:111). Endpoint asserts `bins[0] == -20.0` and `bins[254] == 20.0` are exact (float32 linspace endpoints are exact); midpoint asserts `|bins[127]| < 1e-6` correctly captures both JAX (`0.0`) and PyTorch (`7.45e-8`). | None |
| 2 | `test_twohot_encode_matches_sheeprl` | YES — fixture-gen runs the actual sheeprl `TwoHotEncodingDistribution` and replicates the internal two-hot-target computation byte-for-byte from sheeprl L253-L274 (gen_cp5_fixtures.py:128-156), storing the resulting `[T=4,B=16,255]` `torch_out_twohot` array in the `.npz`. The test rebuilds the two-hot target via the same arithmetic in JAX (test_loss.py:186-200) and compares against the stored sheeprl bytes. **THIS IS THE TEST THAT CATCHES THE HISTORICAL BUG.** | YES — sheeprl L253-L274 cited at test_loss.py:137 and at scripts/sheeprl_jax_diff.py:790. Verified: sheeprl L253-L274 is the `log_prob` body up through the `target = ... .squeeze(-2)` line (the two-hot encoding part of log_prob, before the `log_pred` multiplication at L275). Cited range is accurate. | Correct — the JAX bin-lookup arithmetic (test_loss.py:186-200) is byte-faithful to sheeprl L256-L274: `(self.bins <= x).astype(int32).sum(axis=-1, keepdims=True) - 1` mirrors sheeprl's `(self.bins <= x).type(torch.int32).sum(dim=-1, keepdim=True) - 1`; `jnp.minimum/maximum` clamps match `torch.minimum/maximum`; the cross-weight assignment (`weight_below = dist_to_above / total`) is correctly the linear-interpolation form. **The historical bug would produce max_abs_diff >> 0.1 here** (bins[127] in real space ≠ 0 → wrong bin indices for any target near 0 → wrong weights). Observed `6.080e-6` is the D-006 ULP cascade scale, not the historical-bug scale. | None |
| 3 | `test_twohot_log_prob_matches_sheeprl` | YES — fixture-gen calls `sheeprl_dist.log_prob(targets_torch)` directly (gen_cp5_fixtures.py:185) and stores the resulting `[T=4,B=16]` array in `torch_out_log_prob`. Test compares JAX `jax_dist.log_prob(targets_jax)` against stored sheeprl bytes. Exercises both encode path (Test 2's machinery) AND the logsumexp normalization + cross-entropy reduction. | YES — sheeprl L253-L276 cited at test_loss.py:222 and at scripts/sheeprl_jax_diff.py:842. Verified: this is the complete `log_prob` body. | Correct — `jax.scipy.special.logsumexp` is the JAX equivalent of `torch.logsumexp`; the keepdim convention (`keepdims=True` in JAX, `keepdims=True` in PyTorch, since recent versions of PyTorch accept this spelling) is preserved. `(target * log_pred).sum(axis=self.dims)` matches sheeprl L276. Shape check at test_loss.py:244-246 verifies the event-dim reduction. | None |
| 4 | `test_loss_module_does_not_import_from_src_models` (structural) | n/a (AST-based isolation check, not numerical) | YES — references the isolation rule from v2 Risks §13; verifies `loss.py` has no `from src.models.dreamer_v3*` imports. | Correct — uses `ast.parse` + `ast.walk` to enumerate all `Import`/`ImportFrom` nodes and check for the forbidden substring. False-positive-resistant (the docstring at loss.py:9-10 mentions `src.models.dreamer_v3` in *text*, not as an import — the AST walk only inspects `node.module` / `alias.name`, so the docstring text is correctly ignored). | None |
| 5 | `test_bins_not_symexp_at_storage` (structural) | n/a (assertion-based regression guard, not numerical bit-identity) | n/a — historical-bug regression check, no sheeprl side needed. | Correct — see "genuineness check" section above. Catches the symexp-at-storage mutation at 14-order-of-magnitude margin. | None |

## Per-function audit of `src/algorithms/dreamer_srl/loss.py`

| Function (line range) | Sheeprl source range cited | Citation accurate? | JAX/Flax patterns | Notes |
|---|---|---|---|---|
| `TwoHotEncoding.__init__` (L85-L115) | sheeprl L224-L243 (in commit msg); L225-L243 in inner docstring | YES — sheeprl L224 is `class TwoHotEncodingDistribution:`, L225-L243 is the `__init__` body. Inline citations: L234 → `self.logits = logits` (matches loss.py:103); L235 → `self.probs = F.softmax(logits, dim=-1)` (matches loss.py:105 with `jax.nn.softmax`); L236 → `self.dims = tuple([-x for x in range(1, dims + 1)])` (matches loss.py:107 exactly); L237 → `self.bins = torch.linspace(...)` (matches loss.py:111 with `jnp.linspace`). All four inline citations verified against the vendored file. | **Plain Python class, not `@flax.struct.dataclass`.** Same precedent as CP3b's `SequentialReplayBuffer`. **Acceptable choice** for `TwoHotEncoding` because: (1) the class is constructed fresh per loss computation (`TwoHotEncoding(logits, dims=1)` inside a forward pass), so there's no persistent state to pytree-register; (2) the class never crosses JIT (it's invoked inside JIT-traced loss functions, but the class instance itself is local to the trace — `self.logits`, `self.probs`, `self.bins` are all JAX arrays that trace cleanly); (3) following sheeprl's structure exactly preserves Lever-B parity. The user's brief notes "TwoHotEncoding is closer to a math primitive (like CP1's Moments pattern) than to a stateful container" — I'd note `Moments` is a `@struct.dataclass` precisely because its state *is* carried across update calls (the EMA buffers persist), whereas `TwoHotEncoding` is one-shot. The plain-class choice is correct for the use case. | None |
| `mean` property (L117-L134) | sheeprl L245-L247 (TwoHotEncodingDistribution.mean) | YES — sheeprl L247 verbatim: `return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))`. JAX version (loss.py:134): `return symexp(jnp.sum(self.probs * self.bins, axis=self.dims, keepdims=True))` — semantically identical (`self.transbwd = symexp` per sheeprl L232 default; `keepdim → keepdims`; `dim → axis`). | Pure-functional; no side effects; deterministic given `self.probs`/`self.bins`. No PRNG needed. JIT-safe (only `jnp` ops on traced arrays). | None |
| `mode` property (L136-L146) | sheeprl L249-L251 | YES — sheeprl L251 verbatim: `return self.transbwd((self.probs * self.bins).sum(dim=self.dims, keepdim=True))`. Identical to `mean` (correct — for a two-hot distribution mean equals mode by sheeprl's convention). | Same as mean. | None |
| `log_prob` (L148-L236) | sheeprl L253-L276 | YES — every inline citation verified against vendored file. The 12 line-tagged sub-citations (`sheeprl L254` → `x = self.transfwd(x)`, `sheeprl L256` → `below = ...`, `sheeprl L258` → `above = below + 1`, `sheeprl L261` → `above = torch.minimum(...)`, `sheeprl L263` → `below = torch.maximum(...)`, `sheeprl L265` → `equal = ...`, `sheeprl L266-L267` → distances, `sheeprl L268` → `total = ...`, `sheeprl L269-L270` → cross-weights, `sheeprl L271-L274` → one_hot scatter, `sheeprl L275` → log_pred, `sheeprl L276` → return) all match the vendored source line numbers exactly. | Pure-functional; no PRNG; deterministic given `(self.logits, self.bins, x)`. JIT-safe — no Python branches on traced values; `jnp.where` is traced cleanly; `jax.nn.one_hot` is the JAX-idiomatic equivalent of `F.one_hot`. The cross-weight assignment (`weight_below = dist_to_above / total`) is correctly the linear-interpolation form (closer to above → more weight to below) and matches sheeprl exactly. The `jnp.squeeze(target, axis=-2)` mirrors sheeprl's `.squeeze(-2)`. | None |

### Pure-functional discipline

`TwoHotEncoding` is a plain Python class with three instance attributes set at
`__init__` (`logits`, `probs`, `bins`, `dims`, `low`, `high`) and no mutation
after construction. All methods (`mean`, `mode`, `log_prob`) are read-only over
`self`. The class is JIT-safe: instances are created inside trace contexts,
the attributes are JAX arrays that trace cleanly, and there is no Python
branching on traced values. No PRNG key is needed (deterministic given
inputs).

### Isolation rule

```
$ grep -r "from src.models.dreamer_v3" src/algorithms/dreamer_srl/
(no output)

$ grep -r "import src.models.dreamer_v3" src/algorithms/dreamer_srl/
(no output)

$ grep -r "src\.models" src/algorithms/dreamer_srl/
src/algorithms/dreamer_srl/utils.py:9:src.models.dreamer_v3_nnx, src.models.dreamer_v3_trainer, or any other file
src/algorithms/dreamer_srl/utils.py:10:in src.models. The only allowed shared import is src.utils.config.Config.
src/algorithms/dreamer_srl/utils.py:74:    src/models/dreamer_v3_util.py — that truncation is a known bug (v2
src/algorithms/dreamer_srl/loss.py:9:from the legacy dreamer-v3 models in src/models/ (dreamer_v3_nnx, dreamer_v3_trainer,
src/algorithms/dreamer_srl/loss.py:10:or any other file in src.models). The only allowed shared import is
src/algorithms/dreamer_srl/buffers.py:11:src.models.dreamer_v3_nnx, src.models.dreamer_v3_trainer, or any other file
src/algorithms/dreamer_srl/buffers.py:12:in src.models.
```

All `src.models` hits are **docstring text** describing the isolation rule
itself, not actual imports. AST-based verification (test 4) confirms zero
import statements reference `src.models.dreamer_v3*`. ✅

### Hafner full-precision constants check

CP5 has no truncated-literal risk: the only numerical constants in `loss.py`
are `low=-20`, `high=20` (verbatim from sheeprl L229-L230 defaults), and the
implicit `n_bins=255` from `logits.shape[-1]` (matches sheeprl XS default at
`vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml`). Spot-checked
against vendored sheeprl: identical. ✅

## Deviation review (D-006)

**Sheeprl source line cited:** `vendor/sheeprl/sheeprl/utils/distribution.py:L237`.
Verified by direct file read — line 237 is exactly
`self.bins = torch.linspace(low, high, logits.shape[-1], device=logits.device)`.
Citation accurate. ✅

**What the deviation is.** JAX's `jnp.linspace(-20, 20, 255)` produces
`bins[127] = 0.0` exactly. PyTorch's `torch.linspace(-20, 20, 255)` produces
`bins[127] = 7.45e-8` (1 float32 ULP from zero at the midpoint of step
`40/254 * 127`). This is a linspace-*implementation* ULP difference, not a
mathematical-formula difference — both bin grids are valid IEEE 754 float32
evaluations of the same mathematical linspace.

**Cascade plausibility check (code-correctness lens).**

- **bins level**: `max_abs_diff = 1.907e-6` — this is the raw 1-ULP linspace
  diff propagated through the 255-element array. Plausible.
- **encode level**: `max_abs_diff = 6.080e-6` — the bin diff cascades through
  `|bins[below] - x|`, `|bins[above] - x|`, divisions to form `weight_below`/
  `weight_above`. A 3.2× amplification from bin level to encode level is
  consistent with the arithmetic depth (4-5 float32 operations involving
  subtractions and divisions where rounding can compound).
- **log_prob level**: `max_abs_diff = 1.812e-5` — the encode diff cascades
  through the `(target * log_pred).sum(axis=self.dims)` reduction over 255
  bins (since `target` is the 255-bin two-hot vector), plus the float32
  `logsumexp` normalization. A 3× amplification from encode to log_prob is
  consistent with summing 255 noisy products. The reported relative diff of
  **2.4e-6** at `|log_prob| ~ 7.5` is exactly the float32 platform-drift
  signature (verified independently: `2.4e-6` relative is well below 1 ULP
  relative).

The 10× total amplification (bins → log_prob) is on the boundary but is
consistent with the arithmetic depth of the operations involved (linspace ULP
→ 5 ops per bin pair → 255-element reduction → logsumexp normalization).
**I do not see evidence of a code bug masquerading as hardware drift.** Math-
reviewer should re-verify the per-step amplification math is what they expect
analytically.

**Threshold rationale.** D-006 is the same *class* of deviation as D-003
(symexp float32 ULP drift, threshold `2e-5`, PI-approved 2026-05-13).
D-006's threshold `3e-5` is 1.5× the observed `1.812e-5` worst case — modest
headroom for hardware variability across CUDA versions / driver versions /
RNG state initialization, but tight enough that any semantic error (e.g., a
sign flip in a weight, an off-by-one in the bin clamp, a wrong reduction
axis) would still produce a diff orders of magnitude above `3e-5`. The
threshold is *principled, not padded*. I forward to PI my concurrence:
**recommend ✅ APPROVE** at CP5 gate, on the same JAX-platform-drift basis
as D-003.

**Code-correctness verdict on D-006:** ULP cascade plausible from
implementation; not masquerading as a code bug. Forward to math-reviewer for
analytical confirmation of the 10× amplification, and to PI for
portfolio-level acceptance.

## Diff-tool registry check

| Entry | Location | Verified |
|---|---|---|
| `FUNCTION_REGISTRY["twohot_bins_endpoints"]` | scripts/sheeprl_jax_diff.py:886 | ✅ maps to `_run_twohot_bins_endpoints` (L747-L778) |
| `FUNCTION_REGISTRY["twohot_encode"]` | scripts/sheeprl_jax_diff.py:887 | ✅ maps to `_run_twohot_encode` (L781-L834) |
| `FUNCTION_REGISTRY["twohot_log_prob"]` | scripts/sheeprl_jax_diff.py:888 | ✅ maps to `_run_twohot_log_prob` (L837-L865) |
| `FUNCTION_THRESHOLDS["twohot_bins_endpoints"]` | scripts/sheeprl_jax_diff.py:905 | ✅ `3e-5` (D-006) |
| `FUNCTION_THRESHOLDS["twohot_encode"]` | scripts/sheeprl_jax_diff.py:906 | ✅ `3e-5` (D-006) |
| `FUNCTION_THRESHOLDS["twohot_log_prob"]` | scripts/sheeprl_jax_diff.py:907 | ✅ `3e-5` (D-006) |
| `CHECKPOINT_REGISTRY["CP5"]` | scripts/sheeprl_jax_diff.py:925 | ✅ lists all 3 functions (was pre-existing skeleton row, now populated) |

All three runners follow the established CP1/CP3b runner contract:

1. **Load the right fixture** — runner names match fixture names
   (`twohot_bins_endpoints` → `twohot_bins_endpoints_input.npz`), so
   `run_checkpoint("CP5")` correctly infers the fixture path.
2. **Return `(jax_out, torch_out, metadata)`** — all three runners return
   the right tuple shape with a multi-line metadata string carrying sheeprl
   source citation + JAX file:method + fixture shape + D-006 note.
3. **Use the right threshold** — `_effective_threshold` is called by both
   `run_single` and `run_checkpoint`; all three CP5 functions are in
   `FUNCTION_THRESHOLDS` with `3e-5`, so the diff tool applies the relaxed
   threshold automatically.

**Nit — stale example output citation.** The top-of-file docstring example
output at `scripts/sheeprl_jax_diff.py:21` says
`sheeprl: vendor/sheeprl/sheeprl/utils/distribution.py:L185-L260` —
this is the **v2-archived-plan citation range** from when sheeprl lived at
`tmp/sheeprl/`. In the actual vendored file at commit `33b6366`, the
`TwoHotEncodingDistribution` class spans L224-L276, and the runners
correctly cite L237 / L253-L274 / L253-L276. The stale `L185-L260` only
appears in the example output of the module docstring, not in any active
code path. **🟢 nit — easy fix when next touching the file**: update
`scripts/sheeprl_jax_diff.py:21` to `L224-L276` to match the runners.

## Test fixtures generated against sheeprl side (not self-comparison)

Verified: `scripts/fixtures/gen_cp5_fixtures.py:46` imports
`from sheeprl.utils.distribution import TwoHotEncodingDistribution` (after
adding `vendor/sheeprl` to sys.path at L41). The three fixtures all store
sheeprl-side reference outputs alongside inputs:

| Fixture | Input keys | Sheeprl reference key |
|---|---|---|
| `twohot_bins_endpoints_input.npz` | `n_bins, low, high, expected_*` | `torch_bins` (sheeprl's 255-element float32 bin grid) |
| `twohot_encode_input.npz` | `logits, targets, n_bins, low, high` | `torch_out_twohot` (sheeprl's `[T,B,255]` two-hot target) |
| `twohot_log_prob_input.npz` | `logits, targets, n_bins, low, high` | `torch_out_log_prob` (sheeprl's `[T,B]` log_prob output) |

Files persist on disk (verified `ls -la tests/fixtures/dreamer_srl/twohot_*.npz`
→ 2382 / 62908 / 62335 bytes, May 14 00:56).
**No self-comparison.** ✅

## Conventions audit checklist

| Check | Result |
|---|---|
| Pytree / immutability — no in-place mutation in JIT-traced code | ✅ |
| JIT recompilation triggers — no Python branching on traced values | ✅ |
| vmap conventions — n/a (CP5 has no vmap) | n/a |
| PRNG key threading — n/a (TwoHotEncoding is deterministic given inputs) | ✅ |
| Sensor / observation sync — n/a (not env code) | n/a |
| Configuration protocol — n/a (no new YAML keys) | n/a |
| Source-citation accuracy (Lever B) | ✅ all inline citations verified line-by-line |
| Isolation rule — no imports from `src.models.dreamer_v3*` | ✅ |
| Fixtures generated against sheeprl side, not self-comparison | ✅ |
| Diff-tool registry — FUNCTION_REGISTRY + THRESHOLDS + CHECKPOINT_REGISTRY entries | ✅ |
| **Symlog-space discipline triple-consistency** (load-bearing CP5 check) | ✅ |
| **`test_bins_not_symexp_at_storage` genuineness** (load-bearing CP5 check) | ✅ |

## Verdict

✅ **PASS** — no blockers, one 🟢 nit on a stale docstring example output.

### Findings table

| Severity | File:line | Issue | Suggested fix |
|---|---|---|---|
| 🟢 nit | `scripts/sheeprl_jax_diff.py:21` | Top-of-file docstring example output cites `L185-L260` (stale v2-plan range from when sheeprl lived at `tmp/sheeprl/`). All active runners correctly cite the vendored ranges (`L237` / `L253-L274` / `L253-L276`). | Update the example output line to `L224-L276` to match the actual `TwoHotEncodingDistribution` class span in `vendor/sheeprl/sheeprl/utils/distribution.py`. Cosmetic — does not affect any test result. |

### Hand-off

- **Math-reviewer** can begin. Focus areas: the `log_prob` formula, the
  two-hot weight linear-interpolation derivation, and analytical confirmation
  that the 10× ULP-cascade amplification from `bins` (1.9e-6) → `log_prob`
  (1.8e-5) is what the encoding/cross-entropy math predicts.
- **D-006** forwarded to PI with code-reviewer concurrence to ✅ approve on
  the same JAX-platform-drift basis as D-003. Threshold `3e-5` is principled
  (1.5× observed worst-case; same class as D-003's `2e-5`).
- **Nit** can be picked up by `developer` at any future touch of
  `sheeprl_jax_diff.py` — does not block CP5 closure.

Reviewed by: code-reviewer
