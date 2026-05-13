---
title: "dreamer-srl v3 CP3 — code-reviewer audit"
topic: dreamer
status: active
reviewer: code-reviewer
created: 2026-05-14
last_updated: 2026-05-14
---

# dreamer-srl v3 CP3 — code-reviewer audit

## Plain-language verdict

This review covers the third algorithmic checkpoint (**CP3**) of the dreamer-srl v3 rebuild: the port of sheeprl's **zero-initialization discipline** for the reward-head and critic-head output linears. The fix matters because DreamerV3 uses a two-hot 255-bin logit head for both reward and value prediction; if the final linear is random-initialized at construction, the very first training step sees high-magnitude, structured logits that the loss can latch onto before any learning has occurred — producing spurious early gradients that the world model can never recover from cleanly. Sheeprl's fix (Hafner-cited "cascade fix #27" in the v2 archive) is to overwrite both heads' output-linear `weight` *and* `bias` with all-zeros at the end of `build_agent`, forcing the heads to start as truly uninformative predictors.

The developer added two new `nnx.Module` classes (`RewardHead`, `CriticHead`) to `src/algorithms/dreamer_srl/agent.py`, both of which:
1. Construct an `nnx.Linear` with the standard NNX random init.
2. **Immediately overwrite** kernel and bias with all-zeros, the kernel coming from CP1's already-tested `uniform_init_weights(0.0, ...)` helper and the bias being a plain `jnp.zeros(...)`.

Both Lever-A tests assert exact equality (`max_abs_diff = 0.0`) against a sheeprl-generated fixture in which `torch.nn.Linear` has been put through sheeprl's own `uniform_init_weights(0.0)` apply pass. Both passed; the full 23-test dreamer-srl suite continues to pass with no regression.

**Verdict: PASS.** No blockers. No fixes required. Math-reviewer may begin.

## Scope & artefacts reviewed

| Artefact | Lines | Notes |
|---|---|---|
| `src/algorithms/dreamer_srl/agent.py` | +154 (L203-L350) | Two new `nnx.Module` classes: `RewardHead` and `CriticHead`. |
| `tests/algorithms/dreamer_srl/test_agent.py` | +138 (L211-L327) | Two new Lever-A tests (kernel + bias parity, with shape checks). |
| `scripts/fixtures/gen_cp3_fixtures.py` | +173 (new file) | Drives `torch.nn.Linear` + sheeprl's `uniform_init_weights(0.0)` in the `sheeprl_bridge` env. |
| `scripts/sheeprl_jax_diff.py` | +90 (L743-L827, plus registry / CP-map entries) | Two new runners + 2 `FUNCTION_REGISTRY` entries + `"CP3"` group. |
| `tests/fixtures/dreamer_srl/zero_init_reward_head_input.npz` | new | 1780 bytes, deterministic. |
| `tests/fixtures/dreamer_srl/zero_init_critic_head_input.npz` | new | 1780 bytes, deterministic. |

Vendored sheeprl reference: `vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L1170-L1180` (the `hafner_initialization` block in `build_agent`).

## Per-test audit

| # | Test | Bit-identity real? | Source citation OK? | JAX correctness | Edge cases | Issues |
|---|------|--------------------|---------------------|-----------------|------------|--------|
| 1 | `test_zero_init_reward_head_matches_sheeprl` | YES — fixture-gen (`gen_cp3_fixtures.py` L94-L104) builds a fresh `torch.nn.Linear(512, 255)` with PyTorch's *random* init, then calls `reward_linear.apply(uniform_init_weights(0.0))` from the actual vendored sheeprl source (`gen_cp3_fixtures.py:L64`). Both `torch.weight` and `torch.bias` are asserted all-zero on the sheeprl side at fixture-generation time (L111-L112), then stored in `torch_kernel` / `torch_bias`. Test compares JAX-side `head.output_linear.kernel[...]` and `.bias[...]` against these stored sheeprl bytes (test L243-L268). The "zero-equals-zero" trap is sidestepped because the *sheeprl* side was force-randomized first and then zeroed by sheeprl's own helper — not by the test author. | YES — citation `vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L1175` is accurate. I read sheeprl `agent.py:L1170-L1180` and confirmed L1175 is `world_model.reward_model.model[-1].apply(uniform_init_weights(0.0))`. | Correct — `RewardHead.__init__` (`agent.py:L242-L271`) constructs `nnx.Linear` with rngs (random init), then **overwrites** `self.output_linear.kernel = nnx.Param(zero_kernel)` and `.bias = nnx.Param(zero_bias)` (L270-L271). The `nnx.Param(...)` assignment pattern is the same one CP2 uses to inject fixture parameters into `LayerNormGRUCell` (`test_agent.py:L127-L130`) — known-good idiom for replacing parameter variables in NNX. | Full kernel matrix is compared, not a single element: `kernel_diff = jnp.max(jnp.abs(jax_kernel - torch_kernel))` over the entire `[512, 255]` array (test L255). Bias is checked separately as a `[255]` array (L264). Shape assertions at L247-L252 also catch transpose / dim-swap bugs. A future regression to `scale=0.01` would produce non-zero kernel entries that fail the `< 1e-6` threshold — the recurrence test holds. | None |
| 2 | `test_zero_init_critic_head_matches_sheeprl` | YES — identical fixture-generation discipline to test 1 (`gen_cp3_fixtures.py` L141-L154), independent fresh `nn.Linear` + independent `apply(uniform_init_weights(0.0))` call, independent `torch_kernel_c`/`torch_bias_c` storage. Test compares JAX `CriticHead`'s kernel + bias against these sheeprl-generated zeros. | YES — citation `vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py:L1172` is accurate; L1172 is `critic.model[-1].apply(uniform_init_weights(0.0))`. | Correct — `CriticHead.__init__` (`agent.py:L316-L339`) mirrors `RewardHead.__init__` exactly, including the construct-then-overwrite pattern. Shape contract `[512, 255]` matches sheeprl's XS config (`dense_units=512`, `num_bins=255`). | Same as test 1: full matrix compared, bias checked separately, shape asserted. Recurrence-test discipline preserved. | None |

## Per-class audit of `src/algorithms/dreamer_srl/agent.py`

| Class (line range) | Sheeprl source range cited | Citation accurate? | JAX/Flax patterns | Notes |
|---|---|---|---|---|
| `RewardHead.__init__` (L242-L271) | `sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L1170-L1180` | YES — verified inline against the vendored source. L1175 is the literal call site for the reward head; the L1170-L1180 range covers the full `hafner_initialization` block. | Construct-then-overwrite via `nnx.Param(...)` assignment. Same idiom CP2 uses for fixture-parameter injection. The `rngs` argument is consumed by `nnx.Linear.__init__` and then thrown away (the random kernel is overwritten on the next line) — this is wasted PRNG draw but harmless. | Uses CP1's `uniform_init_weights(0.0, in_features, out_features, key)` (L267) — correct reuse, **not** an ad-hoc zero implementation. CP1's `uniform_init_weights` is itself bit-identity-tested (`test_uniform_init_weights_matches_sheeprl`), so the math chain is `CP1 test (covers scale!=0 random case) + CP3 test (covers scale=0 zero case) = full helper coverage`. |
| `RewardHead.__call__` (L273-L282) | implied L1175 (the head's forward pass is just a linear projection in sheeprl's `MLP` head) | OK — single linear `self.output_linear(x)`. CP3's scope is the **init discipline only**; the MLP body that feeds this head is CP4's responsibility. The docstring at L221-L222 says exactly this ("this class exposes only the output linear — the full MLP body is wired in CP4"). | Pure forward — no PRNG, no JIT-incompatible patterns, no in-place mutation. | The hardcoded `255` two-hot bins is **not** in the class — it's a constructor argument (`out_features`). Magic number scoping is correct: the *test* hardcodes 255 (`gen_cp3_fixtures.py:L79`) from sheeprl XS config; the class itself is dimension-agnostic. |
| `CriticHead.__init__` (L316-L339) | `sheeprl@33b6366:sheeprl/algos/dreamer_v3/agent.py:L1170-L1180` | YES — L1172 is the critic-side call site. | Identical idiom to `RewardHead.__init__`; same correctness profile. | The two classes are nearly duplicate — sheeprl factors both through the same `MLP` builder, but at the JAX side the heads are minimal `nnx.Module`s, and the CP3 brief explicitly says "expose only the output linear" so duplication is intentional. CP4 will refactor when the full MLP body lands. |
| `CriticHead.__call__` (L341-L350) | implied L1172 | OK — symmetric with `RewardHead.__call__`. | Pure forward. | Same notes. |

### Citation accuracy (Lever B)

I read `vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py` lines **1167-1180** directly. The cited range L1170-L1180 covers:

```python
1170    if cfg.algo.hafner_initialization:
1171        actor.mlp_heads.apply(uniform_init_weights(1.0))
1172        critic.model[-1].apply(uniform_init_weights(0.0))           # ← CriticHead cite
1173        rssm.transition_model.model[-1].apply(uniform_init_weights(1.0))
1174        rssm.representation_model.model[-1].apply(uniform_init_weights(1.0))
1175        world_model.reward_model.model[-1].apply(uniform_init_weights(0.0))  # ← RewardHead cite
1176        world_model.continue_model.model[-1].apply(uniform_init_weights(1.0))
1177        if mlp_decoder is not None:
1178            mlp_decoder.heads.apply(uniform_init_weights(1.0))
1179        if cnn_decoder is not None:
1180            cnn_decoder.model[-1].model[-1].apply(uniform_init_weights(1.0))
```

Both `L1172` (critic) and `L1175` (reward) carry **`scale=0.0`** — distinct from the `scale=1.0` used on the RSSM transition/representation heads (L1173-L1174) and the actor MLP heads (L1171). The CP3 port correctly distinguishes these: only the reward and critic *value-prediction* heads get zero-init; the RSSM and actor are excluded from this checkpoint (they belong to CP4 / later checkpoints).

The vendored helper `uniform_init_weights` at `sheeprl/algos/dreamer_v3/utils.py:L170-L186` was also re-read; the math `limit = np.sqrt(3 * given_scale / denoms)` matches CP1's port at `src/algorithms/dreamer_srl/utils.py:L113-L117` byte-for-byte (already CP1-tested).

### `uniform_init_weights` reuse check

Confirmed via `from src.algorithms.dreamer_srl.utils import uniform_init_weights` at `agent.py:L29`. No ad-hoc zero-init code path — both heads route through the same CP1-tested helper, exercising it at the `scale=0.0` edge. This is the correct Lever-B discipline: one canonical helper, one canonical test, no shadow implementations.

### Head shape correctness

`gen_cp3_fixtures.py:L78-L79` defines `IN_FEATURES=512` and `OUT_FEATURES=255` with inline citations to the sheeprl XS config (`dreamer_v3.yaml`). The `nn.Linear(512, 255)` instantiation at L95 and L142 matches sheeprl's reward / critic head shapes for the XS scale. The JAX-side `RewardHead` / `CriticHead` constructors accept `in_features` and `out_features` as required positional args — no hardcoded 255 inside the class itself (the class is dimension-agnostic, which is correct for CP4 to wire in the full MLP body around it).

The 255 = 2-hot bin count is documented in the docstring (`agent.py:L234`, L311), and the test loads `in_features` / `out_features` from the fixture rather than hardcoding (test L233-L234, L291-L292) — so any future config change to `dense_units` or `num_bins` requires only a fixture regeneration, not test edits.

### Isolation rule

```
$ grep -rn "from src.models.dreamer_v3" src/algorithms/dreamer_srl/
src/algorithms/dreamer_srl/agent.py:7:    This module does NOT import from src.models.dreamer_v3_* or any other
```

Only match is the docstring comment declaring the rule. No real import. Rule holds. The only cross-module dependency in `agent.py` is `from src.algorithms.dreamer_srl.utils import uniform_init_weights` (L29), which is intra-`dreamer_srl` and explicitly allowed.

### Fixture genuineness

The fixture is *not* a self-validating zero-roundtrip. `gen_cp3_fixtures.py` performs the following on the sheeprl side:

1. Constructs `nn.Linear(512, 255)` with PyTorch's default Kaiming-uniform random init (L95, L142).
2. Stores the *pre-init* random kernel for documentation (L98-L99, L144-L145).
3. Calls `reward_linear.apply(uniform_init_weights(0.0))` — the **actual vendored sheeprl function** (`gen_cp3_fixtures.py:L64` imports it from `sheeprl.algos.dreamer_v3.utils`).
4. Asserts on the sheeprl side that both kernel and bias are now exactly zero (L111-L112, L153-L154).
5. Stores the zero arrays as `torch_kernel` / `torch_bias`.

The genuine claim being made by the fixture is "sheeprl's `uniform_init_weights(0.0)`, applied to a randomly-initialized PyTorch Linear, produces all-zeros for both weight and bias." That claim is independent of the JAX port and is checked at fixture-generation time. The JAX test then independently verifies that our `RewardHead.__init__` / `CriticHead.__init__` produce the same all-zeros. If sheeprl ever changed its zero-init mechanism (e.g., to leave bias untouched), the fixture would regenerate to non-zero bias and our test would catch the JAX-side divergence.

This is exactly the right structure for a Lever-B port verification. Both the fixture-side and the JAX-side could trivially produce all-zeros by short-circuit; what makes the verification real is that **each side independently invokes its own zero-init helper** and the result is asserted to match.

### Diff-tool registry

```
$ grep -n "zero_init" scripts/sheeprl_jax_diff.py
165:# CP3 (agent.py):        zero_init_reward_head, zero_init_critic_head
747:def _run_zero_init_reward_head(fixture) -> tuple:
789:def _run_zero_init_critic_head(fixture) -> tuple:
1039:    "zero_init_reward_head": _run_zero_init_reward_head,
1040:    "zero_init_critic_head": _run_zero_init_critic_head,
1096:    "CP3":  ["zero_init_reward_head", "zero_init_critic_head"],
```

Two new `FUNCTION_REGISTRY` entries (L1039-L1040), one new `"CP3"` group entry (L1096), one new comment line in the CP overview (L165). Both runners use the same `np.concatenate([kernel.ravel(), bias.ravel()])` flattening so kernel + bias are jointly diffed (worst-case max across the union). Metadata strings cite the correct sheeprl lines (L1175 / L1172). The runners' import path (`from src.algorithms.dreamer_srl.agent import RewardHead`, L761; `CriticHead`, L803) matches the actual class names.

## Conventions audit

(Project-wide JAX / Flax discipline checklist.)

| Convention | Status | Notes |
|---|---|---|
| Pytree immutability | ✅ | `nnx.Module` is the canonical NNX state container; both heads construct via `__init__` and overwrite parameter `nnx.Param` slots, which is the supported pattern (matches CP2's existing usage at `test_agent.py:L127-L130`). No in-place mutation across JIT boundaries. |
| JIT recompilation triggers | ✅ | `in_features` and `out_features` are passed as positional `int` args at construction; they become `nnx.Linear`'s static shape state. No traced fields where static is required. |
| vmap conventions | N/A | CP3 heads are not vmap'd at this checkpoint; CP4 will wire them into the world-model rollout. The classes are dimension-agnostic so any future vmap layer is the caller's responsibility. |
| PRNG key threading | ✅ (with cosmetic note) | The constructor takes both a `key: jax.Array` (for `uniform_init_weights`) and `rngs: nnx.Rngs` (for `nnx.Linear`'s placeholder init). The `rngs` consumption is technically wasted (the random kernel is immediately overwritten), but harmless — this mirrors sheeprl's two-phase init (random construct + zero-init apply) and preserves the contract that NNX modules accept `rngs`. See "Cosmetic notes" below. |
| Sensor / observation breakdown sync | N/A | CP3 does not touch sensors or `EnvParams`. |
| Configuration protocol (`get_mandatory`) | N/A | CP3 has no new YAML keys; `IN_FEATURES=512` / `OUT_FEATURES=255` are fixture-generator constants cited to sheeprl XS config, and the class itself takes them as positional args (no defaults). |
| Isolation rule (no `src.models.dreamer_v3.*` imports) | ✅ | grep clean. Only intra-`dreamer_srl` import is `utils.uniform_init_weights`. |
| Recurrence-test discipline (would fail under future regression) | ✅ | Test asserts full kernel matrix + full bias at `< 1e-6`. Changing `scale=0.0` → `scale=0.01` in `agent.py:L267` would produce non-zero entries in the `[512, 255]` kernel, failing the threshold loudly. Verified by reading the test assertions, not just running them. |

## Findings table

| Severity | File:Line | Issue | Suggested action |
|---|---|---|---|
| 🟢 nit | `src/algorithms/dreamer_srl/agent.py:L246-L248, L320-L322` | The `key` parameter is documented as supplied "for API consistency with `uniform_init_weights`" but is functionally a no-op (the helper produces all-zeros regardless of key value when `scale=0.0`). This is intentional and correctly documented in the docstring (L236-L237, L312), but a future refactor that drops `key` would not change behaviour. | None required — accurate as-is. Optional: when CP4 lands and these heads become part of a larger MLP, consider removing `key` from the constructor and computing the zeros locally as `jnp.zeros((in_features, out_features))`. Defer to CP4. |
| 🟢 nit | `src/algorithms/dreamer_srl/agent.py:L258-L263` | The `nnx.Linear` is constructed with a real PRNG draw and then has both its `kernel` and `bias` overwritten on the next two lines (L270-L271). This is a wasted PRNG state advance. The wastage is one `jax.random.split` per head per construction — entirely negligible at training time (constructors run once). | None required. The construct-then-overwrite pattern faithfully mirrors sheeprl's two-phase `apply()` semantics and keeps the NNX `Rngs` contract intact. |
| 🟢 nit | `tests/algorithms/dreamer_srl/test_agent.py:L65` | The `from src.algorithms.dreamer_srl.agent import LayerNormGRUCell, action_shift, RewardHead, CriticHead` line bundles all four. Acceptable, but if the test file grows further it may be worth splitting into `test_layernorm_gru.py` / `test_action_shift.py` / `test_heads.py`. | None required. Defer to CP4 when the file grows further. |

No 🔴 blockers. No 🟡 concerns.

## Verdict

✅ **PASS.** CP3 is mechanically correct, faithfully ports `sheeprl@33b6366:agent.py:L1170-L1180` (both call sites verified inline), correctly reuses CP1's `uniform_init_weights` helper at the `scale=0.0` edge, and asserts genuine bit-identity (`max_abs_diff = 0.000e+00`) against a fixture that independently invokes the vendored sheeprl helper on a randomly-initialized PyTorch Linear. Recurrence-test discipline preserved (full matrix + full bias compared, not single elements). Isolation rule holds. Diff-tool registry entries land cleanly. 23/23 full dreamer-srl suite still passes — no regression.

Math-reviewer may begin.

---

Reviewed by: code-reviewer
