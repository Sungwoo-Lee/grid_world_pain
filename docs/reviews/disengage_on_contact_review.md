---
title: "Code Review — disengage_on_contact (strike-and-retreat flag)"
topic: env_entities
status: complete
created: 2026-06-09
last_updated: 2026-06-09
---

# Code Review — `disengage_on_contact` (strike-and-retreat flag)

> **Verdict**: APPROVE
> **Reviewed commit**: `edab52d` feat(env): add disengage_on_contact flag
> **Reviewed by**: code-reviewer

## Plain-language summary

This change adds a per-animal on/off switch called **`disengage_on_contact`**. When
it is on for a hunting animal, the moment that animal lands on the agent's cell its
"energy bar" (stamina) is forced to zero. The environment already knows what to do
with a zero-stamina hunter: it gives up the chase, walks back toward the middle of
its home patch, rests until recharged, and then starts chasing again. So this one
flag turns a "sits on the agent forever" chaser into a "strike, back off, recover,
re-engage" chaser, with no new behaviour code — it just pulls an existing trigger.

The switch is **off by default**, so every one of the 86 existing experiment configs
behaves exactly as before. The review confirms that claim two ways: the change adds
**no random-number draws** (so trajectories stay bit-identical), and the existing
byte-parity test suite still passes unchanged (38 passed, 0 failures). The feature's
own three-test module passes. **No blockers, no concerns, no nits — approved as-is.**

## What I verified

The implementation threads one new boolean array (`animal_disengage_on_contact`,
shape `[N]`) from YAML through the loader into `EnvParams`, plus one `jnp.where`
override line in `jax_step`. The high-risk part flagged by the plan was the
`_load_animals` return-tuple slot — a one-position slip would silently swap two
per-animal arrays. I walked every producer and consumer of that tuple by line.

## Findings

| Severity | File:line | Issue | Status |
|---|---|---|---|
| (none) | — | No blockers, concerns, or nits found | — |

## Detailed audit

### 1. Tuple-arity / threading correctness (the main footgun) — PASS

The new array is inserted in **exactly one slot** — immediately after
`animal_is_damaging`, immediately before `animal_visual_channel` — in all three
places that must agree:

- Zero-N return tuple: `config_loader.py:450`
- Non-zero return tuple: `config_loader.py:613`
- Caller unpacking in `load_env_params`: `config_loader.py:691`

`_load_animals` has a **single** call site (`config_loader.py:696`, unpacked at 681–696);
no other consumer exists (grep confirmed). Everything below the inserted name in the
unpack — `animal_classes`, `animal_behaviours`, `animal_tags`, `hunt_idx`, `wander_idx`,
`static_idx`, `predator_indices`, `neutral_indices`, the two placement spawn-area arrays —
is **unshifted**: producer order (613) and consumer order (691) are byte-identical.
The `EnvParams(...)` constructor passes it by keyword (`config_loader.py:869`), so even
if positions did drift the kwarg binding would protect the field — but they do not drift.
`predator_indices` / `neutral_indices` / `hunt_idx` are NOT shifted.

### 2. Pytree discipline — PASS

`animal_disengage_on_contact: jnp.ndarray` is declared at `state.py:124`, **above** the
`pytree_node=False` block that starts at `state.py:126`. It is therefore a plain Flax
struct leaf (a traced pytree node), correctly so — it is read inside the JIT'd `jax_step`.
It is built as `jnp.array(..., dtype=jnp.bool_)` at `config_loader.py:579`, shape `[N]`.
No host-side Python bool, no `struct.field(pytree_node=False)`, no recompile-per-value.
Test 3 asserts `shape == (3,)` and per-entry order `[True, False, False]`.

### 3. The drain-line placement — PASS

At `core.py:486–488`:
- `at_animal` (`core.py:471`) and `new_animal_stamina` (from `update_animals`, `core.py:403`)
  are both in scope and POST-update at the insertion point. Confirmed by line walk.
- `at_animal & params.animal_disengage_on_contact` is `[N] & [N]` → `[N]` bool, broadcasts
  correctly over all N animals; the masked `jnp.where(..., 0.0, new_animal_stamina)` is `[N]`.
- The result flows unchanged into `state._replace(animal_stamina=new_animal_stamina, ...)`
  at `core.py:646`. No intervening reassignment of `new_animal_stamina`.
- A flagged-but-non-hunting animal: its stamina is zeroed on contact but never read for
  state transitions (`_wander_step` / static path ignore stamina), so the override is a
  harmless no-op — matches Checkpoint 4. Correct.
- Placement relative to the attack-delay block (`core.py:480`) is independent: attack-delay
  touches `new_animal_at`, this override touches `new_animal_stamina`. No interaction.

### 4. PRNG byte-parity — PASS

The override is a pure `jnp.where` over already-materialised arrays — **zero**
`jax.random` calls (grep-confirmed; the only draws nearby are the pre-existing
`damage_key` uniforms at `core.py:474` and `:494`, untouched). The key-split order is
unchanged, so the v2.0 per-subset draw-shape contract is intact. For all 86 existing
configs the flag is all-False, making the override `where(False, 0.0, x) == x` — an
algebraic leaf-by-leaf no-op. **Empirically confirmed**: `test_unified_parity.py` +
`test_entities_schema.py` → 38 passed, 71 skipped, 0 failures.

### 5. vmap over envs — PASS

`animal_disengage_on_contact` is a leading-`[N]` leaf, so it composes with the
`[num_envs]` vmap exactly like every other per-animal array (`EnvParams` is broadcast,
not batched — the array lives on the shared params, correct). The override is applied
over the **full** unified animal array, not the `hunt_idx` static subset, so it does not
interact with `update_animals`' subset slicing. Test 3 runs `jax.vmap(jax_step)` over a
batch of 4 envs with a mixed `[hunt+flag, wander, hunt+no-flag]` config; output
stamina/state shapes are `(4, 3)`, no NaN, `hunt_idx == (0, 2)`, `wander_idx == (1,)`.

### 6. Default-False backward-compat — PASS

All three YAML paths read the flag optional-with-default-False:
- `entities:` path — `config_loader.py:328` `bool(ent.get('disengage_on_contact', False))`
- legacy `predators:` — `config_loader.py:361` `bool(p.get('disengage_on_contact', False))`
- legacy `neutral_animals:` — `config_loader.py:395` `bool(n.get('disengage_on_contact', False))`

This is the documented "off" state of an opt-in switch, NOT a fallback default for a
critical key, so `.get(..., False)` is the correct discipline here (mandatory would
`ValueError` all 86 pre-existing configs). The zero-animal branch builds the right-shaped
empty array: `config_loader.py:428` `jnp.zeros(0, dtype=jnp.bool_)` → shape `(0,)`.
Test 2 asserts the no-flag config yields an all-False `[1]` array.

## Conventions audit checklist

- Pytree / immutability: PASS — leaf array, state assembled via `_replace`, no in-place mutation.
- JIT recompilation: PASS — traced bool array, not static; no Python branch on a traced value.
- vmap & batch: PASS — `[N]` leaf on broadcast `EnvParams`; composes with env vmap.
- PRNG threading: PASS — zero draws added; key-split order unchanged; byte-parity proven.
- Sensor / obs-breakdown sync: N/A — no sensor, observation, or noise-modality change.
- Config protocol: PASS — opt-in `.get(..., False)` is correct (not a mandatory key); new key
  matches the plan's File Changes section exactly (path + type + default).

## Tests run by reviewer (conda env)

```
tests/env/test_disengage_on_contact.py          3 passed in 22.31s
tests/env/test_unified_parity.py + schema       38 passed, 71 skipped in 224.86s, 0 failures
```

## Conclusion

APPROVE — surgical, byte-safe, correctly threaded. The main footgun (tuple slot) is
clean across all producers and consumers; pytree, vmap, and PRNG-parity discipline all
hold; backward-compat is empirically proven. No fixes required.

Reviewed by: code-reviewer
