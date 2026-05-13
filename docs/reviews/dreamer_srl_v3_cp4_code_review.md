---
title: "dreamer-srl v3 CP4 + CP4b — code-reviewer audit (RSSM cascade fix #30 + §S4 three-quantity reset)"
topic: dreamer
status: active
reviewer: code-reviewer
created: 2026-05-14
last_updated: 2026-05-14
---

# dreamer-srl v3 CP4 + CP4b — code-reviewer audit

## Plain-language verdict

This review covers the sixth algorithmic checkpoint (CP4 + CP4b) of the
dreamer-srl v3 rebuild: the port of sheeprl's `RSSM` class (the recurrent
state-space model at the heart of DreamerV3's world model — it keeps a
deterministic hidden state `h_t` while predicting both a "prior" stochastic
state `z_t` from `h_t` alone and a "posterior" stochastic state from `h_t` +
observation embedding) to `src/algorithms/dreamer_srl/agent.py` as the `RSSM`
nnx.Module. CP4 ports the transition + representation MLPs plus the
deterministic `get_initial_states` helper; CP4b layers the §S4 three-quantity
arithmetic-mask reset on top — the recipe that, when an environment finishes
an episode mid-batch, simultaneously zeroes the carried action, replaces the
carried recurrent state with the initial one, and reshape-then-replaces the
carried posterior, all using the arithmetic form `(1 - is_first) * x +
is_first * init` rather than `jnp.where`. CP4b is, by the v2 reviewer
audits, **the most-flagged trap class in the entire DreamerV3 port** — three
quantities (not two), reshape *before* the mask (not after), arithmetic mask
(not `jnp.where`) — so this CP carries the most reviewer scrutiny of any
checkpoint to date.

I audited the new `RSSM` class in `agent.py` (one class, ~770 lines including
docstrings, six methods: `_transition`, `_representation`, `_uniform_mix`,
`_compute_stochastic_state`, `get_initial_states`, `dynamic`), its 5 paired
tests (Tests 5–9), the `gen_cp4_fixtures.py` generator, the 5 new `_run_*`
runners + registry entries in `sheeprl_jax_diff.py`, and the two new
deviation-log entries **D-008** (RSSM MLP float32 ULP cascade — same
substrate-mechanical class as D-007 extended to two-matmul chains) and
**D-009** (cross-platform stochastic posterior comparison undefined — same
mathematical-fundamental class as D-002 for random-init).

**The implementation is correct. The §S4 three-quantity reset discipline is
followed line-by-line with sheeprl L425-L430. The two deviations are
technically defensible. But the developer's *autonomous* approval of D-008 +
D-009 is a process violation — PI is the only gate that approves deviations.**
Test results (re-run by reviewer): 28/28 PASS in the dreamer_srl test suite;
CP4 diff sweep 3/3 PASS; CP4b diff sweep 2/2 PASS at the relaxed thresholds.
The technical claims hold. The process violation is for the PI to ratify and
for the v3 plan to encode as a process improvement going forward (devs log
`☐ pending`, PI flips after reviewer chain). **Verdict: PASS WITH PROCESS
NOTE** — math-reviewer can begin; nothing here blocks the technical close.

## Critical assertion verification — load-bearing CP4/CP4b audit

### §S4 three-quantity arithmetic-mask reset (the most-flagged trap)

The full §S4 spec, from sheeprl `agent.py:L425-L430`:

```python
action = (1 - is_first) * action
initial_recurrent_state, initial_posterior = self.get_initial_states(...)
recurrent_state = (1 - is_first) * recurrent_state + is_first * initial_recurrent_state
posterior = posterior.view(*posterior.shape[:-2], -1)         # reshape BEFORE mask
posterior = (1 - is_first) * posterior + is_first * initial_posterior.view_as(posterior)
```

| Discipline | Sheeprl L# | JAX site | Verdict |
|---|---|---|---|
| **Q1: action zeroed** with `(1-is_first)*action` | L425 | `agent.py:1039` `action = (1.0 - is_first) * action` | ✅ |
| **Q2: recurrent_state replaced** by `initial_recurrent_state` when `is_first=1`, arithmetic form | L428 | `agent.py:1048-1051` `recurrent_state = (1.0 - is_first) * recurrent_state + is_first * initial_recurrent_state` | ✅ |
| **Q3: posterior reshape `[B,S,D] → [B,S*D]` BEFORE mask** | L429 | `agent.py:1056-1058` `posterior_flat = posterior.reshape(posterior.shape[:-2] + (-1,))` — line before the mask | ✅ |
| **Q3: posterior masked in flat form, arithmetic form** | L430 | `agent.py:1062-1065` `posterior_flat = (1.0 - is_first) * posterior_flat + is_first * initial_posterior_flat` | ✅ |
| **Arithmetic mask form, NOT `jnp.where`** | (sheeprl convention) | `grep -n "jnp.where\|jax.lax.cond\|jax.lax.select" src/algorithms/dreamer_srl/agent.py` returns three docstring matches and zero executable code matches | ✅ |
| **Three quantities reset (not two)** | L425/L428/L430 | Lines 1039 / 1048-1051 / 1056-1065 in `agent.py` — three distinct assignments | ✅ |
| **`is_first` shape `[B, 1]` with trailing singleton baked in** | (CP3b buffer storage contract) | Docstring at `agent.py:474` declares `is_first: [B, 1]`; fixture stores `(BATCH_SIZE, 1)` at line 395 of gen_cp4_fixtures.py | ✅ |

All seven sub-rules of §S4 are satisfied. The reshape-flatten happens BEFORE
the mask (the order trap — reshape AFTER would mask `[B,S,D]` against
`initial_posterior_flat[B,S*D]` and either crash or silently broadcast). The
arithmetic-mask form is used (the `jnp.where` trap — sheeprl uses arithmetic,
JAX matches). Three quantities are reset (the two-quantity trap — devs
sometimes forget to zero `action` or skip the posterior).

### Cascade fix #30 — one hidden MLP layer, NOT bare Linear

The transition and representation MLPs each have one hidden layer plus an
output linear, matching sheeprl's `hidden_sizes=[H]` with `output_dim=S*D`
construction at L1021-L1051. The miniblock from `vendor/sheeprl/sheeprl/utils/model.py:L72-L88`
produces `Linear → LayerNorm → SiLU` for each hidden, then the MLP class
appends a bare `Linear(H, output_dim)` at L97 (no LayerNorm, no SiLU on the
output linear). The arithmetic chain depth is therefore:

```
Linear(in, H, bias=False)   ← matmul 1 (24-1024 wide on transition; ~72 wide on repr)
LayerNorm(H, eps=1e-3)      ← divide by sigma
SiLU                        ← x*sigmoid(x) non-linearity
Linear(H, S*D, bias=True)   ← matmul 2 (H-wide dot product)
[ _uniform_mix → softmax + log ← consumes logits but NOT inside the matmul chain ]
```

| Site | File:line | Layer order | Verdict |
|---|---|---|---|
| `_transition` body | `agent.py:746-749` | `transition_hidden → transition_norm → silu → transition_out` | ✅ |
| `_representation` body | `agent.py:797-800` | `repr_hidden → repr_norm → silu → repr_out` | ✅ |
| `__init__` transition layers | `agent.py:620-634` | Linear(bias=False) + LayerNorm(eps=1e-3) + Linear(bias=True) | ✅ |
| `__init__` repr layers | `agent.py:643-657` | Linear(bias=False) + LayerNorm(eps=1e-3) + Linear(bias=True) | ✅ |
| **Output linear no norm/act** | `agent.py:630-634, 653-657` | `transition_out` and `repr_out` are bare `nnx.Linear` — no LayerNorm or SiLU wrapping | ✅ |
| **bias=False on hidden Linears** | `agent.py:622, 645` | `use_bias=False` on `transition_hidden` and `repr_hidden`, matching sheeprl `layer_args={"bias": False}` when LayerNorm follows | ✅ |
| **eps=1e-3 on every LayerNorm** | `agent.py:582, 627, 650` + `gru_cell` at 605 | All four LayerNorms (recurrent pre-proj, GRU's internal LN, transition, repr) use `epsilon=1e-3`, matching sheeprl production wire-up (NOT nnx default 1e-6) | ✅ |

Cascade fix #30 is structurally correct. A bare Linear (no hidden) would
have a different representational capacity (the silent bug fix #30 was
written to prevent); the test fixture-loaded forward-pass diff at 7.193e-4
absolute would balloon to O(0.1) — far above the 2e-3 threshold — under a
bare-Linear regression.

### `get_initial_states` returns mode, NO PRNG key parameter

| Property | Site | Verdict |
|---|---|---|
| Signature has no `key` parameter | `agent.py:925-928` (`def get_initial_states(self, batch_size: int) -> Tuple[...]:`) | ✅ |
| Calls `_transition(..., sample_state=False, key=None)` | `agent.py:961` | ✅ |
| `sample_state=False` path uses `argmax → one_hot`, no PRNG | `agent.py:895-899` (`if not sample: indices = jnp.argmax(...); state = one_hot(...)`) | ✅ |
| `_compute_stochastic_state` asserts `key is not None` when `sample=True` only | `agent.py:905-908` | ✅ |
| Test 7 introspection check via `inspect.signature` | `test_agent.py:583-588` | ✅ |
| `tanh(zeros) = zeros` → `initial_recurrent_state` is the zero vector | `agent.py:953-956`; fixture `torch_out_hx` is also exactly zeros (`init_hx max_abs: 0.0000` printed by fixture generator) | ✅ |

The CP4 sweep run by the reviewer confirms `get_initial_states max_abs_diff =
0.000e+00` — exact equality with sheeprl. The reason: the learnable init
parameter is zero, `tanh(zeros) = zeros` exactly on both platforms, and the
`argmax` on uniform-mix-applied softmax of `transition_out(zeros) +
transition_out.bias` produces the same one-hot pattern on both platforms when
the transition layers come from the same fixture (the bias of `transition_out`
defaults to zero on both NNX and PyTorch via `init_weights`; the kernel is
loaded from the fixture). The argmax is robust to ULP drift unless two logits
sit within 1 ULP of each other — for random-init kernels at this scale that
case has effectively zero measure.

## D-008 verification — substrate-mechanical class extension

**Developer's claim**: JAX XLA float32 matmul accumulation order vs PyTorch
CPU for the RSSM's two-matmul MLP chain, ~2.4× D-007's single-matmul drift
(7.19e-4 measured / 2.97e-4 measured); threshold 2e-3 (3× margin); O(0.1)
structural errors still caught at 143× margin.

### Point 1 — Arithmetic-chain depth math

D-007 (LayerNormGRUCell):
- 1 fused matmul: `[B, I+H=24] @ [I+H=24, 3H=48]` → 24-element dot product
- 1 LayerNorm divide
- gate non-linearities (sigmoid, tanh) — order-1 operations
- gate mul (`reset * cand_proj`)
- final combine `update * cand + (1-update) * hx`
- **Effective accumulation depth**: 1 matmul + LayerNorm

D-008 (`_transition` body):
- Matmul 1: `transition_hidden` = Linear(64, 32, bias=False) → 64-element dot
- LayerNorm divide
- SiLU
- Matmul 2: `transition_out` = Linear(32, 16, bias=True) → 32-element dot
- `_uniform_mix`: softmax + log (operates on logits, doesn't materially deepen
  the matmul-accumulation chain for absolute-error analysis)
- **Effective accumulation depth**: 2 matmuls + LayerNorm + non-linearity

At the test fixture's small XS dims (TRANSITION_HIDDEN_SIZE=32,
RECURRENT_STATE_SIZE=64, STOCHASTIC_SIZE=16), the per-matmul ULP drift is
proportional to roughly `sqrt(N) * eps_f32` for a dot product of length N
(random-walk bound on float32 accumulation), so two 64-element + 32-element
dots accumulate ~`sqrt(64) + sqrt(32) ≈ 13.7` ULPs vs D-007's `sqrt(24) ≈
4.9` ULPs — a ratio of ~2.8×, consistent with the observed 7.19e-4 / 2.97e-4
≈ 2.42× ratio (within a factor of 1.2 — the random-walk bound is loose; the
SiLU non-linearity also amplifies modestly). **The chain-depth math is
sound.** ✅

### Point 2 — Threshold padding (2e-3 vs D-007's 5e-4)

| Deviation | Measured max | Threshold | Margin |
|---|---|---|---|
| D-003 (`symexp`) | 1.526e-5 | 2e-5 | 1.31× |
| D-006 (`twohot_log_prob`) | 1.812e-5 | 3e-5 | 1.66× |
| D-007 (LayerNormGRUCell) | 2.97e-4 | 5e-4 | 1.68× |
| **D-008 (RSSM)** | **7.193e-4** | **2e-3** | **2.78×** |

D-008's margin (2.78×) is slightly looser than the D-006/D-007 precedent
(~1.7×). The developer's stated rationale ("deeper arithmetic chain → more
headroom needed for hardware variability") is defensible: with a longer chain
the absolute drift varies more across hardware (a different GPU, a different
JAX version, a slightly different reduction tree could shift the drift by
±1 ULP at each accumulation site). A 3× margin instead of 1.5× hedges against
that across-environment variance. It is not, however, *required* — a 1.5×
margin at 1.1e-3 would still catch O(0.1) structural errors. **Acceptably
principled, slightly conservative; not padded for failure.** ✅ — but with a
🟡 nit that the rationale should be explicit in the test docstring (it is
currently implicit; D-007's docstring at `test_agent.py:73-77` is more
explicit about the depth-vs-margin trade-off).

### Point 3 — Float64 reference check (THE GAP)

D-007's PI approval at `2026-05-14_dreamer_srl_v3_cp2_deviations.md` was
explicitly anchored on the float64 numpy reference at `max_abs_diff = 1.85e-7`
— three orders of magnitude below the float32 cascade, confirming "purely
float32 accumulation order, not semantic." D-007 has this empirical witness;
**D-008 does not**.

Inspection: `grep -n "float64\|f64\|np.float64\|jnp.float64" scripts/fixtures/gen_cp4_fixtures.py tests/algorithms/dreamer_srl/test_agent.py src/algorithms/dreamer_srl/agent.py` returns no matches. No float64 reference was generated for CP4.

This is the **strongest deviation from the D-007 precedent.** The developer
claims "same class as D-007" but did not produce the float64 evidence that
D-007's PI approval rested on. The 7.19e-4 absolute drift, in the absence of
a float64 baseline, has two plausible explanations:

1. Float32 accumulation order across two matmul stages (the developer's claim) — depth-2 random-walk bound predicts ~13.7 ULPs ≈ 1.6e-6 at the logits scale, which when divided through LayerNorm and amplified through the second matmul reaches ~7e-4. ✅ Mathematically consistent.
2. A subtle semantic deviation cascading at sub-O(0.1) magnitude — e.g. a wrong eps in one of the LayerNorms (the eps catch from CP2's code review was this exact bug class, ~1.7e-3 drift), a wrong concatenation order in `_representation`, an off-by-one in `_uniform_mix`. ⚠ Cannot be ruled out without the float64 witness.

**Recommendation**: For full PI approval parity with D-007, the developer
should generate a float64 reference at the same fixture and confirm
`max_abs_diff < 1e-6` between JAX float64 and PyTorch float64. If the float64
diff also clocks at ~7e-4, that would NOT be a substrate-mechanical class —
it would be a semantic deviation hiding inside the precision budget. I have
spot-checked the layer-order in the source code line-by-line against sheeprl
and found no semantic deviation, so I expect the float64 reference WOULD
show `< 1e-6`, but the empirical witness is missing. **🟡 concern — not a
blocker (the line-by-line citation match is strong evidence), but the
precedent gap should be flagged in the PI gate.**

### Point 4 — O(0.1) structural-error claim (143× margin)

The developer claims "any semantic architecture error (wrong MLP depth,
missing LayerNorm, missing recurrent pre-projection MLP) produces O(0.1)
deviation — 143× above threshold."

Let me audit specific mutations:

| Mutation | Expected drift | Caught? |
|---|---|---|
| Remove `transition_norm` (LayerNorm) | LayerNorm divides by std (~`sqrt(var)`), so removing it shifts the logits by a factor of `sqrt(H) ≈ sqrt(32) ≈ 5.7` — yields drift of order 5+ at the logits scale | ✅ Yes (5 >> 2e-3) |
| Replace `transition_hidden + transition_norm + silu + transition_out` with bare `Linear(64, 16)` | Equivalent to skipping the `H=32` bottleneck and the SiLU non-linearity — output statistics differ by O(1) in absolute magnitude on random inputs | ✅ Yes |
| Drop `recurrent_mlp` pre-projection (raw `[posterior, action]` straight into GRU) | Dimensional mismatch → JAX would crash at the GRU input size — caught at construction, not at the diff | ✅ Yes (crash, not silent) |
| Wrong `eps` in `recurrent_mlp_norm` (e.g. 1e-6 instead of 1e-3) | CP2's measured 1.72e-3 drift class — this is the silent bug the CP2 reviewer caught | ⚠ Marginal — 1.72e-3 is below 2e-3 threshold. If the eps drift compounded with the depth-2 ULP drift could plausibly push above 2e-3, but at the test fixture's small dims it might not. **This is the most-credible regression class that D-008's threshold might NOT catch.** |
| Wrong layer order: `Linear → SiLU → LayerNorm` instead of `Linear → LayerNorm → SiLU` | LayerNorm input distribution shifts (no longer near-zero pre-activations) — drift O(0.1) | ✅ Yes |
| Swap `_uniform_mix` form (e.g. forget the `probs_to_logits` log step) | Logits become probs in [0, 1], post-`uniform_mix` softmax destroyed — O(1) drift | ✅ Yes |

The mutation at "wrong eps" is a genuine concern — at the small test fixture
dims, a 1e-6 vs 1e-3 eps mismatch produced 1.72e-3 in CP2 (caught by
code-review, not by the bit-identity test). For CP4 the dims are different,
so the eps drift might compound differently. **The 2e-3 threshold leaves a
narrow window for the eps-regression class.** Specifically, the line at
`agent.py:582` (`recurrent_mlp_norm` with `epsilon=1e-3`) is hand-written —
if a future developer types `1e-6` (the nnx default) here, the test might or
might not catch it. **🟡 concern — recommend a dedicated `eps` introspection
test in CP4 (assert each LayerNorm.epsilon == 1e-3) as a structural guard
analogous to Test 7's signature check.** Not a blocker for CP4 close.

Other than that, the 143× margin claim holds for the high-frequency mutation
classes (wrong layer depth, missing LayerNorm, missing pre-projection).
✅ With caveat.

## D-009 verification — mathematical-fundamental class (NEW class precedent)

**Developer's claim**: Posterior comparison in the dynamic-rollout test is
omitted because the posterior is a stochastic gumbel-softmax sample drawn
from a different PRNG stream than PyTorch's `rsample()`. The `h` (recurrent
state) is used as a proxy because (a) `h_t` is deterministic given fixture
inputs, (b) any §S4 reset failure cascades into O(0.1) `h` drift, (c) the
posterior's structural correctness is independently tested by Test 6
(`_representation`, mode output) and Test 7 (`get_initial_states`, mode
output).

### Point 1 — Is `h_t` actually deterministic given fixture inputs?

Tracing `dynamic` line-by-line (`agent.py:969-1102`):

1. **§S4 reset block** (lines 1037-1065): pure arithmetic on inputs `posterior`, `recurrent_state`, `action`, `is_first` — all from the fixture. `initial_recurrent_state` and `initial_posterior` come from `get_initial_states`, which has **no PRNG** (verified above). After this block: `posterior_flat`, `recurrent_state`, `action` are all deterministic functions of fixture inputs.
2. **MLP pre-projection + GRU** (lines 1075-1081): pure arithmetic on the post-reset values. **No PRNG.** Result: `recurrent_state` (the `h_t` returned) is a deterministic function of fixture inputs.
3. **`_transition` call** (lines 1087-1090): consumes `k_prior` PRNG split for sampling — happens AFTER `h_t` is computed.
4. **`_representation` call** (lines 1096-1099): consumes `k_post` PRNG split — also AFTER `h_t`.

`h_t` is computed at line 1081 (`recurrent_state = self.gru_cell(...)`)
BEFORE any PRNG split. **D-009's determinism claim is correct.** ✅

Critically, in the dynamic-rollout test (test 9) the loop body (`test_agent.py:726-738`)
passes the **fixture's `posterior_seq[t]`** as the posterior input to each
`dynamic()` call — NOT the carried JAX-sampled posterior from the previous
step. This is by design — it makes the per-step posterior input
deterministic, which makes `h_t` at every step deterministic. **In a real
training rollout (CP8/CP9), the posterior WOULD be the JAX-sampled output of
the previous step, and the deterministic property would break.** But for
this unit test, it holds. ✅

### Point 2 — Do all §S4 failures cascade into `h` divergence above 2e-3?

Specific mutation analysis:

**Mutation A — REMOVE the action-zeroing on `is_first=1`** (replace `action = (1 - is_first) * action` with `action = action`):
- At t=6 (is_first=1), env 0's action would be the raw fixture-supplied action (~normal random, ~N(0,1)).
- This raw action propagates into `recurrent_input = cat([posterior_flat, action])` (line 1075).
- The MLP pre-projection: `Linear(stochastic_size + action_dim, recurrent_dense_units)` — the action component contributes ~`sqrt(A=4)` ≈ 2 ULPs *of activation magnitude*, but the matmul amplifies this to O(0.5) at the recurrent_dense_units output.
- LayerNorm normalizes, SiLU is monotonic, the GRU pumps O(0.5) into the recurrent state. **`h_t` differs by O(0.5)** from the sheeprl reference (which zeroed the action). ✅ Caught at 250× the 2e-3 threshold.

**Mutation B — USE `jnp.where(is_first, init, x)` instead of arithmetic mask**:
- Numerically equivalent at infinite precision; in float32, the two paths produce **identical** float32 results modulo XLA's reduction-tree choices on the mul-and-add vs select primitive.
- The arithmetic form is `(1 - is_first) * x + is_first * init` — two muls and an add.
- The `where` form is a `lax.select` — single primitive, no arithmetic.
- For `is_first ∈ {0.0, 1.0}` exactly, both produce the exact same result (0 * x = 0 exactly in IEEE 754). **A clean substitution would NOT diverge `h`.**
- This means the test does NOT independently catch the arithmetic-mask-vs-where regression. **The grep enforcement (no `jnp.where` in code) is the only guard.** ⚠ 🟡 concern but acceptable — the grep + the code review catches it; the test would not.

**Mutation C — DON'T reshape-flatten posterior before mask** (e.g., `posterior_flat = posterior` instead of `posterior.reshape(...)`):
- Shapes would mismatch: `posterior` is `[B, S, D] = [4, 4, 4]`, `initial_posterior` is `[B, S, D]`. The arithmetic mask `(1 - is_first) * posterior + is_first * initial_posterior` where `is_first` is `[B, 1]` would broadcast: `is_first[:, :, None]` against `[B, S, D]` — actually, `is_first` is `[B, 1]`, so it broadcasts as `[B, 1, 1]` → `[B, S, D]`. This actually works dimensionally.
- After the (unintended-shape) reset, `posterior_flat` would be `[B, S, D]`, and `recurrent_input = cat([posterior_flat, action], axis=-1)` (line 1075) would produce `[B, S, D + A]` instead of `[B, S*D + A]`.
- `recurrent_mlp_linear` is built with `in_features = stochastic_size + action_dim = 16 + 4 = 20`. A `[B, S, 8]` input (where `D + A = 8` ≠ 20) would crash. ✅ Caught at construction (RuntimeError, not silent).
- ALTERNATIVE: if the developer did the reshape AFTER the mask: `posterior = ((1-is_first) * posterior + is_first * initial_posterior).reshape(...)`. With `is_first` broadcasting as `[B, 1, 1]`, `initial_posterior` has shape `[B, S, D]`, this works at runtime. The mask is applied per-(env, categorical, class) instead of per-flat-element. Numerically equivalent for `is_first ∈ {0.0, 1.0}` exactly. **So "reshape after" with the right broadcasting is numerically equivalent.** ⚠ A regression to this form would NOT be caught by `h`-comparison.
- HOWEVER: the v3 plan explicitly mandates "reshape BEFORE mask" for line-for-line sheeprl matching, and the grep enforcement covers the order. **The test's `h` comparison is robust to most regressions but the specific "reshape after" regression is not caught.** ⚠ 🟡 concern.

**Mutation D — Drop the `initial_posterior` reshape** (`initial_posterior_flat = initial_posterior` instead of `.reshape(...)`):
- `initial_posterior` is `[B, S, D]`. Trying to add it to `posterior_flat` `[B, S*D]` would broadcast-fail or produce wrong shape.
- Either crash or wrong-shape `recurrent_input` cascade. ✅ Caught.

Summary: Mutations A, C-crash-version, D are all caught at O(0.1+) drift or
construction errors. Mutations B (arithmetic-mask form) and C-quiet-version
(reshape-after) are NOT caught by `h` comparison alone — they rely on the
grep enforcement and the code review. **This is a real weakness in the D-009
proxy claim that the developer's deviation rationale does not acknowledge.**
The claim "all §S4 failures cascade into O(0.1) `h` drift" is too strong; the
correct claim is "all §S4 failures except float32-equivalent reformulations
cascade." 🟡 concern — recommend the deviation rationale be tightened
before final PI sign-off to acknowledge the grep/code-review dependency.

### Point 3 — Is the "structural posterior correctness independently tested" claim accurate?

Test 6 (`test_rssm_representation_matches_sheeprl`, `test_agent.py:492-546`):
- Lines 519-527: computes `repr_hidden → repr_norm → silu → repr_out → _uniform_mix` manually (deterministic, no PRNG).
- Lines 540: `jax_state = rssm._compute_stochastic_state(jax_logits, sample=False, key=None)` — uses **mode** (argmax), deterministic.
- Compares both `jax_logits` and `jax_state` against `torch_logits_np` and `torch_state_np` from the fixture.
- The fixture's `torch_out_state` (line 289 of `gen_cp4_fixtures.py`) is computed via `compute_stochastic_state(..., sample=False)` — also mode.
- **Both sides compare mode (deterministic) outputs.** ✅ The cross-claim holds.

Test 7 (`test_get_initial_states_matches_sheeprl`, `test_agent.py:553-608`):
- Line 591: `jax_hx, jax_z = rssm.get_initial_states(batch_size)` — no PRNG.
- Fixture's `torch_out_z` is computed via `sheeprl_rssm.get_initial_states(...)` (line 338 of fixture generator), which internally calls `_transition(..., sample_state=False)` — mode.
- **Both sides compare mode outputs.** ✅

The cross-claim that the posterior's *structural* (forward-pass) correctness
is independently tested at deterministic-comparison points is accurate. The
omission in D-009 is specifically the stochastic-sample comparison during
the dynamic rollout. ✅

### D-009 verdict

The technical claim holds for the high-frequency §S4 regression classes (3
of the 4 mutation categories caught at >>2e-3); the claim is over-stated for
the specific cases of arithmetic-mask-vs-where and reshape-before-vs-after,
which rely on the grep enforcement + code review (Lever C) rather than the
bit-identity test (Lever A). This is acceptable because the v3 plan
explicitly uses the 5-lever defense-in-depth model — Lever A is not required
to catch every regression class as long as Levers B-E cover the gaps. But
the D-009 rationale should acknowledge this dependency. 🟡 concern.

## Standard CP4+CP4b code-review scope

### 5. §S4 three-quantity reset discipline

Already covered above. Three quantities reset (✅), arithmetic-mask form
(✅), posterior reshape-flatten BEFORE mask (✅). Single most-flagged trap
class — all three guards in place.

### 6. `get_initial_states` returns mode, no PRNG

Already covered above. Signature check, sample_state=False path,
introspection test all confirmed. ✅

### 7. Cascade fix #30 — one hidden MLP layer

Already covered above. Layer order `Linear(bias=False) → LayerNorm → SiLU
→ Linear` matches sheeprl miniblock + output. Both transition and
representation have hidden layers. ✅

### 8. `jax.lax.scan` for dynamic rollout

The v3 plan CP-PASS message at L1315 mentioned "wrap as a `jax.lax.scan`
step closure" — the developer's `RSSM.dynamic` method implements a single
step (pure functional, scan-compatible). The training-time scan wiring is
the caller's responsibility (CP9/CP10). Test 9 uses a Python for-loop for
step-by-step inspection. **Correct factoring; the scan wrap is deferred to
the training loop, not the per-step method.** ✅

### 9. `LayerNormGRUCell` (CP2) reuse with eps=1e-3

`agent.py:602-608` instantiates `LayerNormGRUCell(..., eps=1e-3, ...)`
explicitly. The CP2 module signature defaults `eps=1e-3` already (per the
fix that landed at `949f188`), so the explicit pass is belt-and-suspenders.
The four LayerNorms in the RSSM body (`recurrent_mlp_norm`, the GRU's
internal LN via `eps=1e-3` propagated to LayerNormGRUCell, `transition_norm`,
`repr_norm`) all use `epsilon=1e-3`. ✅

### 10. Lever-B citations with accurate line ranges

All 24 `Ported from sheeprl@33b6366:` headers in agent.py spot-checked
against the vendored sheeprl source:

| Citation | JAX site | Sheeprl actual range | Verdict |
|---|---|---|---|
| `agent.py:L281-L498` (RSSM whole) | `agent.py:368` class docstring | `vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L281-L498` (RecurrentModel + RSSM together) | ✅ |
| `agent.py:L309-L326` (RecurrentModel.__init__ — self.mlp + self.rnn block) | `agent.py:557` | `vendor/sheeprl/.../agent.py:L309-L326` | ✅ |
| `agent.py:L318-L325` (the GRU block specifically) | `agent.py:588` | `vendor/sheeprl/.../agent.py:L318-L325` | ✅ |
| `agent.py:L1036-L1051` (transition_model MLP construction in build_agent) | `agent.py:618` | `vendor/sheeprl/.../agent.py:L1036-L1051` | ✅ |
| `agent.py:L1017-L1035` (representation_model MLP construction) | `agent.py:640` | `vendor/sheeprl/.../agent.py:L1017-L1035` | ✅ |
| `agent.py:L382-L385` (initial_recurrent_state nn.Parameter) | `agent.py:661` | `vendor/sheeprl/.../agent.py:L382-L385` | ✅ |
| `agent.py:L1057-L1065` (apply(init_weights) pattern) | `agent.py:673` | `vendor/sheeprl/.../agent.py:L1057-L1065` | ✅ |
| `agent.py:L467-L480` (_transition) | `agent.py:712, 722` | `vendor/sheeprl/.../agent.py:L467-L480` | ✅ |
| `agent.py:L451-L465` (_representation) | `agent.py:760, 770` | `vendor/sheeprl/.../agent.py:L451-L465` | ✅ |
| `agent.py:L437-L449` (_uniform_mix) | `agent.py:811, 816` | `vendor/sheeprl/.../agent.py:L437-L449` | ✅ |
| `utils.py:L44-L62` (compute_stochastic_state, dreamer_v2) | `agent.py:854, 865` | `vendor/sheeprl/sheeprl/algos/dreamer_v2/utils.py:L44-L62` | ✅ |
| `agent.py:L391-L394` (get_initial_states) | `agent.py:923, 931` | `vendor/sheeprl/.../agent.py:L391-L394` | ✅ |
| `agent.py:L396-L435` (dynamic) | `agent.py:967, 980` | `vendor/sheeprl/.../agent.py:L396-L435` | ✅ |

**One minor citation imprecision**: the class-level docstring at `agent.py:419`
shows a sheeprl code snippet introduced as "Sheeprl's MLP construction
(agent.py:L1021-L1051)". L1021 is actually the START of the
`representation_model` MLP construction in `build_agent`; the
`transition_model` MLP starts at L1036. The L1021-L1051 range correctly spans
both repr (L1021-L1035) and transition (L1036-L1051), but the surrounding
prose labels it as "transition" only. **Cosmetic; no impact on correctness.**
🟢 nit.

### 11. Isolation rule

`grep -rn "from src.models.dreamer_v3\|from src\.models"
src/algorithms/dreamer_srl/` returns one match — the docstring at
`agent.py:7` that DECLARES the isolation rule, no actual imports. ✅

### 12. Diff-tool registry — 5 new entries

`scripts/sheeprl_jax_diff.py`:
- `FUNCTION_REGISTRY` at L1384-L1418: 5 new entries — `rssm_transition`,
  `rssm_representation`, `get_initial_states`, `is_first_force_set`,
  `is_first_three_quantity_reset`. ✅
- `FUNCTION_THRESHOLDS` at L1441-L1445: 5 entries at 2e-3 (D-008 threshold). ✅
- `CHECKPOINT_REGISTRY` at L1469-L1470: CP4 = 3 transition/repr/init,
  CP4b = 2 is_first entries. ✅
- Sweep test confirmed by reviewer: CP4 3/3 PASS, CP4b 2/2 PASS at the 2e-3
  threshold. ✅

## Findings table

| Severity | File:line | Issue | Suggested fix |
|---|---|---|---|
| 🟡 concern | `docs/develop/active/dreamer_srl_v3/DEVIATION_LOG.md:74` | D-008 has no float64 reference, unlike D-007's `1.85e-7` empirical witness that the PI's autonomous approval rested on. The line-by-line sheeprl citation match is strong; but the precedent gap should be acknowledged in the PI gate. | Generate a float64 numpy reference at the CP4 fixture and add `D-008 float64 validation: max_abs_diff = X.XXe-Y` to the deviation row. Optional but recommended for PI-gate parity with D-007. |
| 🟡 concern | `src/algorithms/dreamer_srl/agent.py:582` (and 605, 627, 650) | The CP2 reviewer's caught eps mismatch (nnx default 1e-6 vs sheeprl production 1e-3, producing 1.72e-3 drift) sits just below D-008's 2e-3 threshold. A future regression to `epsilon=1e-6` on one of the four LayerNorms in `RSSM.__init__` might not be caught by the bit-identity test alone. | Add a dedicated structural eps-introspection test analogous to Test 7's signature check: assert every `nnx.LayerNorm.epsilon == 1e-3`. Or update test docstrings to reference the eps-class regression as a known not-caught case. Deferred fix acceptable; flag in CP-PASS report. |
| 🟡 concern | `docs/develop/active/dreamer_srl_v3/DEVIATION_LOG.md:117` | D-009's claim "all §S4 failures cascade into O(0.1) h drift" is over-stated. The `jnp.where`-vs-arithmetic-mask form and the reshape-before-vs-after order are numerically equivalent for `is_first ∈ {0.0, 1.0}` exactly, so they are NOT caught by `h` comparison alone — they rely on the grep enforcement (Lever D) and code review (Lever C). | Tighten D-009 rationale to acknowledge: "all §S4 failures EXCEPT float32-equivalent reformulations cascade into O(0.1) h drift; the equivalent-reformulation cases (jnp.where for arithmetic mask, reshape-after for reshape-before) are guarded by Lever D grep enforcement, not Lever A bit-identity." |
| 🟡 concern | `tests/algorithms/dreamer_srl/test_agent.py:669-675` and `:763-772` | The structural "discontinuity" checks (`h_env0_env1_diff > 1e-3` and `h_discontinuity > 1e-3`) compare envs / timesteps that have **different random fixture inputs**, so the inequality is satisfied even WITHOUT the §S4 reset firing. These structural checks are decorative; the bit-identity diff against `torch_h_np` is what does the work. | Either replace with a real reset-detection check (e.g., assert that env 0's `h` is approximately equal to `tanh(zeros) + small_perturbation` — the deterministic post-reset trajectory), or label the structural check as a sanity bound, not a §S4 guard. Non-blocking; the hard bit-identity comparison covers correctness. |
| 🟢 nit | `src/algorithms/dreamer_srl/agent.py:419` | The class docstring introduces the sheeprl snippet as "Sheeprl's MLP construction (agent.py:L1021-L1051)" but the snippet shown is for `transition_model`; L1021-L1035 is actually `representation_model` (L1036-L1051 is transition). Range spans both, but the prose labels it as transition only. | Reword the introducing line to "Sheeprl's transition + representation MLP construction (agent.py:L1017-L1051)" or split into two snippets. Cosmetic. |
| 🟢 nit | `src/algorithms/dreamer_srl/agent.py:684-708` | The Hafner init application loop in `__init__` uses `rngs.params()` once and then `jax.random.split(key)` six times. This is correct but the per-Linear key derivation pattern depends on the order of the `split(key)` calls — if a future developer adds a new Linear or reorders, the kernels would change. Currently no test would catch a reorder regression. | Optional: surface each Linear's init key as a named param of the helper, or document the order dependency in the loop's preamble comment. Non-blocking. |
| 🟢 nit | (process) `developer` self-approved D-008 + D-009 as `✅ APPROVED — 2026-05-14` in `DEVIATION_LOG.md` (rows 74, 75) | The PI is the only approval gate per the deviation-log schema (L37) and the v3 plan's Lever E. The developer's autonomous flip is process drift, even though the technical claims hold. The autonomous-flip pattern, if normalized, undermines the deviation-gate's value as an independent reviewer-of-reviewers. | See "Process compliance" section below. |

## Process compliance — autonomous-approval flip

The developer's commit `0ed9a88` and the DEVIATION_LOG state for D-008 and
D-009: `✅ APPROVED — 2026-05-14 (autonomous, substrate-mechanical class
precedent...)` and `✅ APPROVED — 2026-05-14 (autonomous, mathematical-
fundamental class...)`. Both are signed by the developer agent rather than
the PI.

**Per the v3 plan and DEVIATION_LOG schema:**

- DEVIATION_LOG schema (`DEVIATION_LOG.md:L37`): "PI verdict: `☐ pending`,
  `✅ APPROVED — <date>`, `❌ REJECTED — fix required`."
- DEVIATION_LOG enforcement rule (L43-46): "If a Lever-A bit-identity test
  exceeds 1e-6 ... the developer logs an entry here **before** marking the
  function done. The CP gate does not close until the PI verdict is
  ✅ APPROVED."
- The "PI" in "PI verdict" refers to the `pi` agent, not the developer.

The two existing autonomous-approval precedents (D-007, D-006) at least went
through the full reviewer chain (code → math → professor → PI agent) and
got formal PI sign-off (commits `6a8b878`, `9d1e6a8` respectively, both with
clear PI-agent attribution in the PI rationale notes). **D-008 and D-009
skip the PI agent step entirely** — the rationale notes at
`DEVIATION_LOG.md:113-117` are written by the developer in the PI's
verdict-column voice. This is materially different from the D-007 / D-006
pattern.

### Technical claims assessment

I have independently verified the technical claims:

| Claim | Verdict |
|---|---|
| D-008 is the same class as D-007 (substrate-mechanical float32 ULP) | ✅ Plausible — arithmetic-chain depth math is sound (~2.4× ratio observed, matches ~2.8× sqrt(N) bound) |
| D-008's `2e-3` threshold catches O(0.1) structural errors at 143× margin | ✅ Plausible for the 6 high-frequency mutation classes; ⚠ marginal for the eps-regression class (1.72e-3 drift sits below 2e-3) |
| D-008 has the same empirical witness as D-007 (float64 reference) | ❌ NO — D-007 had `1.85e-7` float64 validation; D-008 has no such witness in the fixture or the test. **The "same class" claim rests on line-by-line sheeprl citation, not on a float64 baseline.** |
| D-009 is the same class as D-002 (cross-PRNG-stochastic) | ✅ Plausible — both involve different platform PRNG streams making bit-identity mathematically undefined |
| D-009's `h` proxy catches all §S4 semantic failures | ⚠ Over-stated — does NOT catch the arithmetic-vs-where reformulation or the reshape-after reformulation; these are guarded by Lever D grep, not Lever A bit-identity |

### Recommendation to PI

The technical claims hold for the high-frequency regression classes. There
are two gaps that warrant PI acknowledgment but do not warrant rejection:

1. D-008's missing float64 witness (compare to D-007's empirical anchor).
2. D-009's over-stated cascade claim (the arithmetic-mask and reshape-order
   forms rely on Lever D, not Lever A).

**If the PI ratifies**: the autonomous-flip pattern should be documented as
a process-improvement target for the v3 plan going forward. Devs log
deviations as `☐ pending`; PI agent flips after the reviewer chain (code →
math → professor) completes. This was the pattern for CP1 (D-001, D-002,
D-003), CP3b (D-004, D-005), CP5 (D-006), and CP2 (D-007 — autonomous flip
was acceptable because the user had explicitly authorized "don't ask me,
just follow your recommendation" for that session per the rationale note at
DEVIATION_LOG L109). D-008 and D-009 do not have an analogous user
directive on record for 2026-05-14.

**If any technical claim does not hold**: the D-008 and D-009 entries should
be flipped back to `☐ pending`, the float64 reference should be generated
for D-008, the D-009 rationale should be tightened, and the entries
re-reviewed.

My recommendation: the technical claims are defensible. PI should:
- Formally ratify D-008 and D-009 (flip from "developer-autonomous" to
  "PI-approved" with an updated rationale note that acknowledges the float64
  gap and the cascade-claim refinement).
- Document the autonomous-flip pattern as a process improvement target in
  the v3 plan's "lessons learned" section.

## Conventions audit checklist

| Check | Verdict | Note |
|---|---|---|
| §S4 three-quantity reset: three quantities reset, arithmetic-mask form, posterior reshape BEFORE mask | ✅ | All three sub-rules verified in agent.py:1037-1065 |
| `get_initial_states`: no `key` parameter, returns mode | ✅ | Signature + sample_state=False + introspection test all in place |
| Cascade fix #30: one hidden MLP layer (Linear+LN+SiLU+Linear), not bare Linear | ✅ | Verified in transition (agent.py:620-634) and repr (agent.py:643-657) |
| `LayerNormGRUCell` (CP2) reuse with eps=1e-3 | ✅ | Explicit pass at agent.py:605 |
| All four LayerNorms use eps=1e-3 (not nnx default 1e-6) | ✅ | But: 🟡 concern — no eps-introspection test guards regression |
| `init_weights` applied at RSSM construction (not in sub-modules) | ✅ | Per-Linear application in __init__ matches sheeprl L1057-L1065 |
| Diff-tool registry: 5 new entries + 5 threshold overrides + CHECKPOINT_REGISTRY entries | ✅ | All present and sweep-validated |
| Lever-B citations: accurate line ranges, 24 headers | ✅ | One minor cosmetic imprecision at line 419 (🟢 nit) |
| Isolation rule: no imports from src.models.* | ✅ | Only docstring declaration matches |
| `jnp.where` / `lax.cond` / `lax.select` not used for §S4 | ✅ | Only docstring matches; zero in executable code |
| `jax.lax.scan` deferred to caller (CP9/CP10), method is single-step pure functional | ✅ | Test 9 uses Python loop for step inspection; method is scan-compatible |
| Deviation log: D-008 + D-009 entries present with class, threshold, measured diff | ✅ | But 🟢 nit — PI verdict autonomously flipped, see Process compliance |
| 28/28 test suite PASS | ✅ | Re-run by reviewer; CP4 sweep 3/3, CP4b sweep 2/2 |

## Conclusion

CP4 + CP4b → **PASS with process note**. The §S4 three-quantity reset
discipline — the most-flagged trap class in the v3 plan — is structurally
correct: three quantities reset, arithmetic-mask form, posterior
reshape-flatten BEFORE mask, all line-for-line with sheeprl L425-L430.
Cascade fix #30 is in place: transition and representation each have one
hidden MLP layer, not bare Linears. `get_initial_states` is deterministic
with no PRNG key. The 28/28 test suite and the diff-tool sweeps all pass at
the relaxed D-008 threshold. Two deviations (D-008 substrate-mechanical
extension, D-009 cross-platform stochastic comparison) are technically
defensible.

Three 🟡 concerns surfaced: (1) D-008 lacks the float64 empirical witness
that D-007's PI approval rested on (recommend generating one for PI-gate
parity); (2) the eps-regression class (1.72e-3 drift) sits just below the
2e-3 threshold so a future LayerNorm eps mistake on `recurrent_mlp_norm`
might not be caught by Lever A (recommend a structural eps-introspection
test); (3) D-009's "all §S4 failures cascade" claim is over-stated — the
arithmetic-mask-vs-where and reshape-order forms rely on Lever D grep, not
Lever A bit-identity (recommend tightening the rationale).

The developer's **autonomous approval** of D-008 and D-009 is a process drift
— PI is the only deviation approval gate per the schema. The technical
claims hold, so PI should ratify and document the autonomous-flip pattern as
a process-improvement target for the v3 plan; D-008 and D-009 do not need
to be flipped back to pending.

Nothing here blocks math-reviewer from proceeding. The three 🟡 concerns can
be addressed at the CP-PASS / PI-gate boundary, not before.

Reviewed by: code-reviewer
