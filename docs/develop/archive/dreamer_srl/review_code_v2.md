---
title: 'dreamer-srl plan v2 — JAX/NNX re-audit'
topic: dreamer
status: superseded
created: 2026-05-12
last_updated: 2026-05-12
---

> **Superseded by**: [`docs/pi/calls/2026-05-12_dreamer_backend.md`](../../../pi/calls/2026-05-12_dreamer_backend.md) — PI pivot to sheeprl-direct (Option 1) shelves the dreamer-srl v2 plan this re-audit cleared.

# dreamer-srl plan v2 — JAX/NNX re-audit

## Plain-language entry point

The senior-developer produced v2 of the dreamer-srl implementation plan, claiming all 11 JAX/Flax-NNX deviations the code-reviewer flagged in v1 have been folded in. v2 is 1033 lines. This re-audit confirms two things: (a) every one of the 11 v1 deviations is resolved at a location a developer reading the file-by-file change table would actually see, and (b) v2 did not introduce any new JAX/NNX errors while folding in 29 deviations from three reviewers.

The headline: **v2 passes.** All 8 blockers and 3 concerns from v1 are resolved at locations the developer will encounter while implementing the named file. The four highest-risk silent-pattern-match traps — fused-gate GRU cell, arithmetic-mask `is_first` reset with three-quantity flatten, mode-not-sample initial states with no PRNG, and pure-functional `Moments` pytree — each now carry an explicit ⚠️ warning against pattern-matching the existing in-house Dreamer scaffolding. The new "Training-loop semantics" section (§S1–§S10) holds the cross-cutting silent omissions globally so the developer reads them once before walking the file-by-file table. No new pytree-mutation language, no parameter aliasing, no PRNG-in-state, no contradictory NNX patterns crept in. The bit-identity gate at Checkpoint 8 is achievable.

One minor underspecification worth naming: `collect_step` uses a Python-style `if use_random:` branch on `iter_num <= learning_starts` without explicitly stating whether `collect_step` is JIT-compiled. This is benign as long as the developer follows the Python-side loop pattern already used by the existing Dreamer for the `Ratio` scheduler — but the plan should say so explicitly.

## Per-deviation table

| # | v1 deviation | Severity | v2 status | v2 location | Notes |
|---|---|---|---|---|---|
| 1 | `lax.scan` carry signature for RSSM dynamic learning | 🔴 blocker | ✅ RESOLVED | `agent.py` JAX/Flax discipline, lines 310-327 | Carry `(recurrent_state, posterior)` named; scan input/output named; full code sketch present with `nnx.scan` form. |
| 2 | `LayerNormGRUCell` 1+1 fused-gate NOT 2+2; chunk order `(reset, cand, update)`; warning against existing class | 🔴 blocker | ✅ RESOLVED | `agent.py` `LayerNormGRUCell` row, line 291 | ⚠️ CRITICAL block. Explicit "ONE Linear + ONE LayerNorm" + "DO NOT pattern-match on `src/models/dreamer_v3_nnx.py:18-70:LayerNormGRUCell`". Full sheeprl-canonical code sketch with `(reset, cand, update)` ordering. |
| 3 | `is_first` arithmetic mask not `jnp.where`; three quantities reset (action+rec+post); posterior pre-flattened | 🔴 blocker | ✅ RESOLVED | Training-loop semantics §S4 lines 120-132 + `agent.py` RSSM row line 297 | Both locations have the arithmetic-mask form. Three quantities explicitly named. Posterior reshape-flatten BEFORE masking explicit. Stated 1:1 with sheeprl `agent.py:423-429`. |
| 4 | `get_initial_states` uses transition mode not sample; no PRNG consumed | 🔴 blocker | ✅ RESOLVED | `agent.py` RSSM row line 297 (second code sketch) | Explicit "MODE, NOT a sample. NO key consumed." Full code sketch with `jax.nn.softmax` of uniform-mixed logits. "DO NOT pass a PRNG key" callout. |
| 5 | PRNG sub-key split count per `one_train_step` (~94 at XS) | 🟡 concern | ✅ RESOLVED | `train.py` JAX/Flax discipline lines 366-378 | Full split-count math (T + 2(H+1) = ~94). Code sketch shows `rssm_keys`, `img_prior_keys`, `img_actor_keys`. "Do NOT reuse key inside scan body (Z3-class bug)" + "Do NOT Python-side counter (retrace)" warnings. "Never carry a key in `state`" rule. |
| 6 | `Player` interpretation (A) vs (B) ambiguity | 🔴 blocker | ✅ RESOLVED | `agent.py` `Player` row line 302 | "Decision locked 2026-05-12: interpretation (B)". `DreamerSrlState` pytree explicitly shows `player_recurrent_state`, `player_stochastic_state`, `player_action`. "Reject interpretation (A)" + "No sync step needed; no aliasing; no deep-copy." |
| 7 | Polyak cadence (BEFORE `train`; `train_step`=cumulative gradient steps) | 🟡 concern | ✅ RESOLVED | `train.py` `polyak_update` row line 361 | Ordering explicit ("Polyak fires BEFORE `one_train_step`, not after"). Code sketch shows polyak block inside `for _ in range(per_rank_gradient_steps)` before `one_train_step` call. `train_step` semantics named: "cumulative gradient steps applied since training began". |
| 8 | `Moments` as `flax.struct.dataclass` not `nnx.Variable` mutation; warn against existing class | 🔴 blocker | ✅ RESOLVED | `utils.py` `Moments` row line 249 | "Decision locked 2026-05-12 (code-reviewer 🔴 #8)". Full pure-functional code sketch with `MomentsState` dataclass + `moments_update` returning `(new_state, offset, invscale)`. Explicit "DO NOT pattern-match on `src/models/dreamer_v3_util.py:Moments`". |
| 9 | `defaults_from:` is not a real Config mechanism | 🟡 concern | ✅ RESOLVED | `01_food_only.yaml` note line 567-573 | "Selected option (c)" with 5-line loader helper code. `defaults_from:` line is to be dropped from the implementation YAML (kept in example for context). Uses `config.get_mandatory('agent_config')`. |
| 10 | `per_rank_*` config-key naming convention | 🟡 concern | ✅ RESOLVED | Risks §14 line 809 + YAML header comment lines 396-399 | "Decision locked 2026-05-12: keep verbatim sheeprl names". YAML header documents the `per_rank_*` meaning ("sheeprl's distributed-training prefix; we run single-device so per_rank=global"). |
| 11 | `lax.stop_gradient` for `discount` cumprod | 🟢 nit | ✅ RESOLVED | `train.py` `one_train_step` row line 359 + Training-loop semantics §S6 line 148 | Explicit `jax.lax.stop_gradient(jnp.cumprod(continues * gamma, axis=0) / gamma)` code. "use `jax.lax.stop_gradient`, NOT `with jax.disable_jit()` or any other pattern". |

**Score**: 11/11 ✅ RESOLVED. Zero ⚠️ partials. Zero ❌ unresolved.

## High-risk silent-pattern-match traps — focused re-check

| Trap | Resolution location | Verdict |
|---|---|---|
| `LayerNormGRUCell` 1+1 vs 2+2 (would fail Checkpoint 8) | `agent.py` row line 291 — ⚠️ CRITICAL block with full warning + code sketch + chunk-order callout | ✅ The warning is at exactly the location a developer pattern-matching the existing class would land. |
| `is_first` arithmetic mask + 3-quantity reset | §S4 (line 120-132, globally) AND `agent.py` RSSM row line 297 (file-specific) | ✅ Dual-located. The developer reads §S4 first then re-encounters at the RSSM row. Posterior pre-flatten step explicit. |
| `get_initial_states` mode-not-sample | `agent.py` RSSM row line 297, dedicated code sketch | ✅ "MODE, NOT a sample. NO key consumed." All-caps callout. |
| `Moments` as `flax.struct.dataclass` | `utils.py` Moments row line 249, dedicated decision-lock block | ✅ Full pure-functional sketch + explicit "DO NOT pattern-match on `src/models/dreamer_v3_util.py:Moments`". Return signature explicit. |

## New JAX/NNX errors introduced by v2

Scanned for: pytree mutation language, parameter aliasing, vmap/scan axis conflicts, hidden recompilation triggers, PRNG-in-state, contradictory NNX patterns, Config Protocol violations.

**Errors found: none.**

**Minor underspecifications (not blockers; flag for developer awareness):**

- **Concern N-1 (🟢 nit) — `collect_step` JIT boundary unstated.** v2 line 360 shows `use_random = (iter_num <= learning_starts)` with a Python-style `if use_random: ... else: ...` branch. This works only if `collect_step` is Python-side (not JIT-compiled) or if `iter_num` is a static-argname. The plan does not explicitly say which. The existing Dreamer's `Ratio` scheduler uses the same Python-side pattern (`train.py:789`), so the precedent is set, but the plan should add a one-line discipline statement: "`collect_step` is Python-side (not JIT-compiled); `iter_num` enters as a Python int. The inner env-step / sample-action body can be JIT'd as a separate function called from inside `collect_step`." Not a blocker; the developer can resolve by mirroring the existing Dreamer's pattern.

- **Concern N-2 (🟢 nit) — `batch_size` undefined in `get_initial_states` code sketch.** v2 line 297 `get_initial_states` sketch passes `(batch_size,)` to itself in the dynamic call but uses `batch_shape` as the formal parameter. Pseudocode-level inconsistency; the actual signature has `batch_shape` and the caller passes `recurrent_state.shape[:1]` or equivalent. Trivial for the developer to resolve.

- **Concern N-3 (🟢 nit) — `WorldModel` container is named but not specified.** v2 line 303 says `WorldModel` is "a container `nnx.Module` with sub-modules as attributes" — fine. But the relationship between `WorldModel` (`encoder + rssm + observation_model + reward_model + continue_model`) and `DreamerSrlState`'s `world_model: nnx.GraphState` is not spelled out. Specifically: does `nnx.split(world_model)` produce the graph state stored on `DreamerSrlState`, or is `world_model` carried as an NNX-module reference? Existing `dreamer_v3_trainer.py` uses the `nnx.split` / `nnx.merge` pattern; the plan should explicitly say `DreamerSrlState.world_model` is the `nnx.GraphState` half of `nnx.split(world_model)`. Trivial — the developer follows the existing pattern.

None of the three concerns above is a blocker. None changes the algorithm. All are addressable inline by the developer.

## Conventions audit checklist

| Convention | Status | Notes |
|---|---|---|
| Pytree immutability (no in-place updates) | ✅ | `agent.py` JAX/Flax discipline §6: "No mutation of pytrees in-place. Every `update_*` returns a new pytree." `Moments` is `flax.struct.dataclass` (resolved deviation #8). |
| JIT recompilation triggers (static-vs-traced) | ✅ | `agent.py` JAX/Flax discipline §7: all static shapes named. PRNG key threading rules prevent retrace (deviation #5). |
| vmap / scan axis discipline | ✅ | `lax.scan` carry signature for RSSM dynamic learning fully specified (deviation #1). |
| PRNG threading | ✅ | Per-step key arrays as scan inputs not carries; ~94 sub-keys per `one_train_step` named (deviation #5). Never carry a key in `state`. |
| Target-critic aliasing | ✅ | Second `Critic` instance; `nnx.update(target, jax.tree.map(...))` form; "do NOT alias parameters across the two instances". |
| Sensor / observation breakdown sync | N/A | Fresh algorithm with `obs_keys=["state"]` only; no new sensor introduced. |
| Configuration Protocol (no-fallback-defaults) | ✅ | Risks §10 lists every key as mandatory; loader helper uses `config.get_mandatory`; `defaults_from:` invented key dropped (deviation #9). |
| `is_first` reset broadcast pattern | ✅ | Arithmetic mask form `(1 - is_first) * x + is_first * init`; three-quantity reset; posterior pre-flatten (deviation #3). |
| Existing-code reuse strictness | ✅ | Risks §13 "No-import-from-existing-Dreamer rule" with named classes that must be re-implemented (`LayerNormGRUCell`, `Moments`, `hafner_init`, `compute_lambda_values`). |
| `Ratio` stays Python-side | ✅ | `utils.py` `Ratio` row + Risks §12 enforce this. |
| Replay sequence semantics (straddle episode boundaries) | ✅ | `buffers.py` row preserves "windows that DO NOT respect episode boundaries; trainer relies on stored `is_first` flag". |

## Final verdict

**✅ PASS — zero residual JAX/NNX deviations; plan is implementation-ready.**

All 11 v1 deviations are resolved at developer-visible locations. The four highest-risk silent-pattern-match traps each carry explicit ⚠️ warnings against the existing in-house Dreamer scaffolding. Three minor underspecifications were noted (`collect_step` JIT boundary, `get_initial_states` pseudocode parameter name, `WorldModel` ↔ `DreamerSrlState` split/merge convention); none is a blocker; all are resolvable inline by the developer using the existing Dreamer's patterns. No new JAX/NNX errors introduced by v2's mechanical edit pass. Senior-developer's "v2 Changes Applied" traceability section (lines 974-1029) accurately maps each reviewer deviation to its applied location — spot-checked five entries and all five matched. The plan is ready for `developer` to begin Step 1 of the Implementation order.

Reviewed by: code-reviewer (re-audit pass on v2)
