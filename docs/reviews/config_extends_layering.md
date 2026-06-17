# Code Review — `extends:` config layering (v3.0, commit c13a3ac)

## Verdict (plain language)

**APPROVE-WITH-NITS.** The new opt-in config-layering feature works correctly for
its documented contract: a YAML file may carry a top-level `extends: environment/default`
(a string, or a list of strings) and the loader will read those base files, deep-merge
them underneath, and let the child file's keys win. Configs that do **not** carry
`extends:` load exactly as before — I verified that the no-`extends:` path returns a
byte-identical dict to the old loader, so nothing existing changes behaviour. The six
correctness questions in scope (cycle safety, path resolution, merge precedence, key
stripping, post-merge validation, list-replace semantics) all check out. The remaining
findings are robustness nits — clearer error messages for a missing top-level file and
for malformed `extends:` values, plus a low-severity note about Python object aliasing
that cannot bite given how configs are consumed today. None block merge.

No `extends:` configs exist in the repo yet (I grepped `configs/`), so this is pure
opt-in infrastructure exercised only by the new test suite — the blast radius today is
zero.

## Scope

Reviewed only the new loading path:
- `src/environment/config_loader.py:33-98` — `_resolve_extends()` + `load_env_config()`.
- `src/utils/config.py:56-74` — `Config.merge()` / `deep_update()` (the merge primitive invoked).
- All 13 call sites of `load_env_config` across `src/` and `scripts/`.
- `tests/env/test_extends_layering.py` (to see what the tests already cover).

## Findings

| Severity | Location | Issue | Suggested fix |
|---|---|---|---|
| 🟢 nit | `config_loader.py:57` (via `config.py:9-14`) | **Silent-miss on missing top-level config.** If `config_path` itself does not exist, `Config.load_yaml` prints a `Warning:` to stdout and returns an empty `Config`. `_resolve_extends` then sees `extends=None` and returns `Config({})`. Downstream `get_mandatory` will eventually raise, but on a confusing "key … is required" message rather than "file not found". Note the asymmetry: a missing **base** (`extends:` target) raises a clear `ValueError` at line 68-72, but a missing **top-level** file does not. This is **pre-existing** behaviour inherited from `Config.load_yaml`, not a regression from this diff. | Optionally add an explicit `os.path.exists(config_path)` check at the top of `load_env_config` that raises `ValueError` with the path, to match the clarity of the base-not-found error. |
| 🟢 nit | `config_loader.py:64,67` | **No type guard on `extends:` values.** A malformed `extends:` that is a dict iterates its keys as if they were path strings (silently "works" if a key happens to name a real base, else raises the generic "not found"). A list element that is not a string raises a bare `TypeError` on `base_rel + ".yaml"`. The schema documents str-or-list-of-str, so this is only a clarity issue. | Add `if not isinstance(base_rel, str): raise ValueError(...)` inside the loop, or validate `extends` is str/list-of-str right after the `pop`. |
| 🟢 nit | `config.py:66-71` (`deep_update`) | **Reference aliasing in merge.** `deep_update` copies dict leaves by recursion (safe) but assigns **lists and any base-only nested dict by reference** into the merged result. I confirmed `merged['environment']['entities'] is base['environment']['entities']` after a merge where the child omits `entities:`. This is harmless **today** because (a) the aliased base is a fresh `load_yaml` re-read on every `_resolve_extends` call and is GC'd, so there is no cross-config leak, and (b) config consumers are read-only (`get_mandatory`). It would only bite if a future consumer mutated a merged-config list/dict in place. | No action needed now. If in-place config mutation ever appears downstream, switch `deep_update` to `copy.deepcopy` the assigned value, or document "merged Config is read-only". |

## Per-question audit

1. **Recursion / cycle safety — ✅.** `_seen` is a `frozenset` of `os.path.abspath`-normalised paths, extended (`_seen | {abs_path}`) before recursing, and checked at entry (line 51). A→B→A and A→A both hit the check and raise `ValueError("…cycle detected at …")` with the offending path. Because the set is threaded by value (frozenset, immutable), sibling bases in a list don't pollute each other's `_seen` — correct. Cycle test C3 exercises the self-cycle path.
2. **Path resolution — ✅.** `_CONFIGS_ROOT` is computed from `__file__` (line 28-30), so resolution is independent of CWD — correct for this repo's "run via explicit interpreter, many entry points" rule. Targets are `configs/<rel>.yaml`. A missing base raises a clear `ValueError` naming both the logical target and the resolved absolute path (line 68-72). One observation: there is no guard against `..` traversal in `extends:` (e.g. `extends: ../../etc/foo`), but configs are author-controlled and the file must exist *and* parse as YAML, so this is not a practical security concern — noted, not flagged.
3. **Merge order & precedence — ✅ and documented.** For `extends: [a, b]`, bases are merged in declared order (line 66-73), so **later bases override earlier** ones, and then `merged.merge(Config(raw))` at line 74 makes **the child win over all bases**. The docstring at line 88-93 documents the list-replace rule. Child-wins is verified by C2 (height/width override) and C9 (full-rollout parity).
4. **`extends` key stripping — ✅.** `raw.pop("extends", None)` at line 58 removes the key from the dict before it is wrapped in `Config(raw)` and merged, so `extends` never reaches `load_env_params` / `get_mandatory`. The popped value drives resolution and is discarded. Confirmed by inspection and by C9's dict-equality assertion (no `extends` key in the merged result).
5. **No-fallback rule — ✅.** `load_env_config` returns the fully-merged `Config`; `get_mandatory` runs inside `load_env_params` on that merged object (every `config.get_mandatory(...)` in `load_env_params` operates on the merged dict). Validation is post-merge, so a sparse child that relies on the base for mandatory keys passes — exactly what C2 asserts. The no-fallback contract is preserved.
6. **List-replace footgun — ✅ correct as implemented and documented.** `deep_update` replaces lists wholesale (a non-dict value, including a list, is assigned directly at `config.py:70-71`). So omitting `entities:` inherits the base's list (C4 `test_c4_omitting_entities_leaks_base_animals`), and only `entities: []` suppresses it (C4 `test_c4_explicit_empty_entities_suppresses_base`). The docstring at `config_loader.py:88-93` calls this out explicitly for sparse authors. **Byte-parity for standalone configs holds**: when `extends` is absent the code returns `Config(raw)` directly (line 60-62) — it never constructs or seeds a base, so a no-`extends:` config cannot accidentally inherit anything. C1 asserts dict-identity against `Config.load_yaml`, and the 31-pass parity suite is the regression guard.
7. **JAX static-field concerns — N/A.** This is eager Python load-time code that produces a plain dict; it does not run under `jit`/`vmap`. It does feed static `EnvParams` fields (`height`, `placement_mode`, `interoceptive_kernel_length`, etc.) downstream, but merge precedence is deterministic and resolved before `load_env_params`, so there is no new recompilation-trigger surface beyond what already existed.

## Conventions audit

| Convention | Status | Note |
|---|---|---|
| Pytree / immutability | ✅ N/A | Load-time dict code, no `EnvState` mutation. |
| JIT recompilation triggers | ✅ N/A | Eager Python; resolved before tracing. |
| vmap / batch | ✅ N/A | No vmap surface. |
| PRNG threading | ✅ N/A | No PRNG in this path. |
| Sensor / obs-breakdown sync | ✅ N/A | Loader does not touch the breakdown; merge is upstream of `_parse_noise_config`, which is unchanged. |
| Config protocol (no-fallback) | ✅ pass | `get_mandatory` runs post-merge; `extends` is meta and correctly stripped. |
| `property` vs `properties` | ✅ N/A | Untouched by this diff. |

## Conclusion

The `extends:` layering is logically correct, the standalone path is provably unchanged,
and all six in-scope correctness questions pass. Ship it; consider the two clarity nits
(missing-top-level-file error, `extends:` type guard) as a follow-up polish rather than a
blocker.

Reviewed by: code-reviewer
