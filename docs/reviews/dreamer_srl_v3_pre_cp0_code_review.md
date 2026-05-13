---
title: "dreamer-srl v3 — Pre-CP0 setup code review"
topic: dreamer
status: active
reviewer: code-reviewer
created: 2026-05-13
last_updated: 2026-05-13
phase: 2
---

# dreamer-srl v3 — Pre-CP0 setup code review

## Verdict

**🔴 FAIL — gating bug in vendor tracking; one minor fix and a few nits in the diff tool / docs.** CP1 cannot begin until the vendored sheeprl source tree is actually committed to git. Right now the entire Python package `vendor/sheeprl/sheeprl/` is silently gitignored — 232 files on disk, 0 in the commit — including the source-of-truth `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py` that the whole rebuild references. This breaks Lever D's premise (the pinned source-of-truth must travel with the repo) and breaks Lever A (every per-function test imports `from vendor.sheeprl.sheeprl.<...>` — on a fresh clone those imports fail). One small `.gitignore` fix plus a `git add -f` is enough to resolve.

Everything else (diff tool skeleton, NNX_CONVENTIONS, DEVIATION_LOG schema, test/fixture READMEs) is structurally sound. The NNX_CONVENTIONS doc faithfully describes the patterns actually used in `dreamer_v3_nnx.py` / `dreamer_v3_trainer.py` (spot-checked Rngs construction, `nnx.split`/`merge`/`state`/`update`, and the EMA target-critic update). After the gitignore fix and the diff-tool nits below, CP1 is unblocked.

## Per-component status

| Component | Verdict | Note |
|---|---|---|
| 1. `vendor/sheeprl/` tracking | 🔴 **BLOCKER** | Inner `sheeprl/` package is gitignored — 0 tracked algorithm files (Finding 1). |
| 2. `scripts/sheeprl_jax_diff.py` | ⚠ pass-with-fix | Empty-registry returns exit 0 (Finding 2); shape-mismatch error opaque (Finding 3); two style nits (Findings 4, 5). |
| 3. `tests/algorithms/dreamer_srl/README.md` | ⚠ minor-nit | CP↔file-layout taxonomy drift vs diff tool's `CHECKPOINT_REGISTRY` (Finding 6). |
| 4. `tests/fixtures/dreamer_srl/README.md` | ✅ PASS | Seed 0xD3EAF, naming, storage format all unambiguous. |
| 5. `docs/.../NNX_CONVENTIONS.md` | ✅ PASS | Spot-checked against actual sources — patterns match. |
| 6. `docs/.../DEVIATION_LOG.md` | ✅ PASS | Schema clear; PI-verdict workflow concrete; cross-links resolve. |

---

## Findings

### 🔴 Finding 1 — Vendored sheeprl algorithm package is gitignored

**File:** `.gitignore:15` and `.gitignore:16`
**Severity:** blocker

The current rules are:

```
sheeprl/        # line 15 — generic project-wide ignore
!vendor/sheeprl/  # line 16 — un-ignores the vendor directory itself
```

`!vendor/sheeprl/` un-ignores the directory entry, but per git's documented gitignore semantics, files inside still match the more-general `sheeprl/` rule because that rule matches **any directory named `sheeprl/` anywhere in the tree** — including the nested Python-package directory `vendor/sheeprl/sheeprl/`. Verified:

```
$ git check-ignore -v vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py
.gitignore:15:sheeprl/  vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py
$ git ls-tree -r 292dd3a -- vendor/sheeprl/ | grep "vendor/sheeprl/sheeprl/" | wc -l
0
$ find vendor/sheeprl/sheeprl -type f | wc -l
232
```

**232 files on disk, 0 tracked in commit `292dd3a`.** A fresh clone of the repo will not contain `dreamer_v3.py`, `agent.py`, `loss.py`, `utils.py`, or anything else under `vendor/sheeprl/sheeprl/`. This silently breaks:

1. **Lever D (vendored source-of-truth)** — the pinned reference is supposed to travel with the repo so reviewers can grep / read line-by-line. It doesn't.
2. **Lever A (bit-identity tests)** — the README template at `tests/algorithms/dreamer_srl/README.md:38` shows `from vendor.sheeprl.sheeprl.<path> import <SheeprlClass>`. On any machine other than the developer's current workstation, that import will fail.
3. **CP1 onwards** — every per-function port references sheeprl line numbers; without the tracked source the citations cannot be verified.

**Suggested fix.** Two edits, then a forced re-add:

1. Edit `.gitignore` line 15 — replace `sheeprl/` with `/sheeprl/` (root-anchored, so only a top-level `sheeprl/` directory is ignored — which is the historical intent of the rule). This is the least-surprise fix.
   - Alternative: keep `sheeprl/` and add an explicit recursive negation `!vendor/sheeprl/**` right after the existing `!vendor/sheeprl/`. Verify with `git check-ignore -v vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py` returning exit 1 (not ignored).
2. After the rule is fixed, run `git add vendor/sheeprl/` (no `-f` should be needed; if it is, double-check the gitignore fix). Verify `git ls-files vendor/sheeprl/ | wc -l` jumps from 75 to ~300+.
3. Amend or follow-up commit (recommend a follow-up: `fix(dreamer-srl): track vendored sheeprl algorithm source`).
4. After commit, re-run the verification: `git ls-tree -r HEAD -- vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py` should show one blob.

---

### 🟡 Finding 2 — `--checkpoint` returns exit 0 when no functions are registered

**File:** `scripts/sheeprl_jax_diff.py:228-254`
**Severity:** concern

`run_checkpoint("CP1", ...)` currently iterates the CP1 function list, prints "SKIP: not yet in FUNCTION_REGISTRY" for each, and returns `True`. Reproduced:

```
$ python scripts/sheeprl_jax_diff.py --checkpoint CP1
... 8x SKIP (not ported)
CP1 summary
  symlog                                             SKIP (not ported)
  ...
$ echo $?
0
```

The audit criterion is **exit non-zero on FAIL so CI can gate on it**. A CP gate that returns PASS when zero functions ran is a silent green light — if CP5 ships and the developer forgets to register the function in `FUNCTION_REGISTRY`, the gate still says PASS. This is exactly the kind of "no test ran but test command exited 0" failure mode CI is supposed to catch.

**Suggested fix.** In `run_checkpoint` (line 244+), distinguish three states:

- All registered + all pass → exit 0.
- Any FAIL → exit 1.
- Any SKIP (i.e. expected functions not yet ported) → exit 2 (or print a `WARN` and return False).

The "no Lever-A tests" short-circuit at line 217-218 (CP8/CP9/CP10) is fine because those checkpoints are explicitly integration / speed gates with zero registered functions by design — that path can stay returning True. The skip-because-unregistered path is the one to harden.

---

### 🟡 Finding 3 — `compare()` raises an opaque numpy broadcast error on shape mismatch

**File:** `scripts/sheeprl_jax_diff.py:128`
**Severity:** concern

Reproduced:

```python
>>> M.compare(np.zeros(3), np.zeros(4), 1e-6)
ValueError: operands could not be broadcast together with shapes (3,) (4,)
```

The audit criterion explicitly requires "Handles shape mismatch with a clear error". The current behavior throws numpy's generic broadcast error, which doesn't tell the reviewer *which side* (JAX or PyTorch) produced the wrong shape — a common bug class when a sheeprl shape is `[T, B, ...]` and the JAX side accidentally outputs `[B, T, ...]`.

**Suggested fix.** Add a shape check before line 128:

```python
if jax_out_np.shape != torch_out_np.shape:
    raise ValueError(
        f"Shape mismatch: jax={jax_out_np.shape} torch={torch_out_np.shape}. "
        f"This is a structural deviation, not a numerical one — "
        f"log it in DEVIATION_LOG.md with the sheeprl source line."
    )
```

Same idea for dtype: if the dtypes differ in *kind* (float vs int) raise; if they differ in *width* (float32 vs float64), warn but proceed (numpy will upcast, max-abs-diff is meaningful).

---

### 🟢 Finding 4 — Brittle JAX-array isinstance check

**File:** `scripts/sheeprl_jax_diff.py:119`
**Severity:** nit

```python
if isinstance(jax_out, type(jnp.zeros(0))):  # jnp.ndarray check
```

This works today but constructs an empty array just to read its `type()`. Idiomatic alternative:

```python
if hasattr(jax_out, "__jax_array__") or "jaxlib" in type(jax_out).__module__:
```

or simpler — just always do `np.asarray(jax_out)`; numpy handles both JAX and numpy inputs.

The current code happens to do `np.asarray` on both branches anyway (lines 120 and 122), so the entire `if/else` block at 118-124 collapses to `jax_out_np = np.asarray(jax_out)`. Recommended simplification.

---

### 🟢 Finding 5 — Output format docstring promises more than `compare()` prints

**File:** `scripts/sheeprl_jax_diff.py:20-29` (docstring) vs `90-139` (implementation)
**Severity:** nit

The module-level docstring shows the intended output as:

```
sheeprl: vendor/sheeprl/sheeprl/utils/distribution.py:L185-L260
jax:     src/algorithms/dreamer_srl/loss.py:TwoHotEncoding
fixture: shape=(16, 4, 255) logits + (16, 4, 1) target, seed=0xD3EAF
max_abs_diff = 1.2e-7
PASS  (< 1.0e-6 threshold)
```

But `compare()` only prints two lines (`fixture: ...` and `max_abs_diff = ...`). The richer header (`sheeprl: ...`, `jax: ...`) must come from the runner's `metadata` string — meaning the runner is responsible for formatting those lines into a multi-line metadata. This isn't broken, but the docstring sets an expectation that isn't encoded in `compare()`'s contract.

**Suggested fix.** Either:

- Document explicitly that the runner returns a multi-line metadata string (and update the FUNCTION_REGISTRY comment at line 144-167 to say so), OR
- Extend `compare()` to take explicit `sheeprl_src: str` and `jax_src: str` kwargs and format them in the header. The latter makes the runner contract clearer.

Recommend the former — keeps `compare()` simple, and the metadata string can be assembled by the runner that already knows the sheeprl line range and JAX location.

---

### 🟡 Finding 6 — Test README's file layout drifts from the diff tool's CHECKPOINT_REGISTRY

**File:** `tests/algorithms/dreamer_srl/README.md:60-66`
**Severity:** concern

The test README's "File layout" claims:

```
├── test_buffers.py      # CP3: SequentialReplayBuffer
├── test_agent.py        # CP2/CP3/CP4/CP4b: LayerNormGRUCell, MLP, encoder/decoder, RSSM, Actor, Critic, build_agent
```

But `sheeprl_jax_diff.py:175` says CP3 = `["zero_init_reward_head", "zero_init_critic_head"]` (agent heads, not buffers). The test README's "CP3 = SequentialReplayBuffer" claim does not appear in any of the CHECKPOINT_REGISTRY entries. This is documentation drift that will confuse the developer at CP3 — which file should the SequentialReplayBuffer test live in, and is it actually a CP3 deliverable?

**Suggested fix.** Reconcile against the v3 plan ([IMPLEMENTATION_PLAN.md](../develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md) §CP table) — whichever is the source of truth wins. Either edit the test README to put the buffer test on the correct CP, or extend `CHECKPOINT_REGISTRY` to include a CP3 buffer entry.

---

## Conventions audit checklist

For pre-CP0 setup the relevant checks are different from a typical environment review — most project-wide JAX conventions (pytree, vmap, PRNG, sensor sync) don't apply because no algorithmic code shipped. The relevant checks:

| Check | Status |
|---|---|
| Vendor tracking discipline (Lever D) | ❌ — Finding 1 |
| Diff tool CLI correctness | ⚠ — Findings 2, 3, 4, 5 |
| Diff tool extensibility per CP | ✅ — FUNCTION_REGISTRY and CHECKPOINT_REGISTRY shapes are clean dict-of-callables; new entries are one-line additions |
| Test naming + Lever-A gate | ✅ — README states "function not marked done until test passes" |
| Fixture determinism | ✅ — Seed `0xD3EAF` fixed; naming convention unambiguous |
| NNX_CONVENTIONS fidelity to actual sources | ✅ — Spot-checked `nnx.Rngs`/`nnx.split`/`nnx.merge`/`nnx.state`/`nnx.update` against `dreamer_v3_nnx.py` and `dreamer_v3_trainer.py:507-510` (EMA) |
| Isolation rule (no `src.models.dreamer_v3_*` imports into `src/algorithms/dreamer_srl/`) | ✅ — Stated explicitly in NNX_CONVENTIONS §5 |
| DEVIATION_LOG schema clarity | ✅ — 9 columns with clear definitions; PI-verdict workflow concrete (per-CP review at gate close) |
| Cross-document link integrity | ✅ — All paths resolved: `IMPLEMENTATION_PLAN.md`, `DEVIATION_LOG.md`, `.claude/agents/pi.md`, Lever A-E anchors |
| `.pinned_commit` matches reality | ✅ — Contents = `33b6366`; consistent with commit-message claim |
| `linguist-vendored` attribute set | ✅ — `.gitattributes` line 1: `vendor/sheeprl/** linguist-vendored` |

---

## Recommendation

1. **Block CP1 launch** until Finding 1 is resolved (the gitignore fix + `git add vendor/sheeprl/` + a follow-up commit). This is non-negotiable — without the vendored package in git, every Lever-A test will fail to import.
2. Route Findings 2, 3 to `developer` for fix before CP1 lands (small, ~20-line diff in `sheeprl_jax_diff.py`).
3. Route Finding 6 to `senior-developer` to reconcile the CP↔file taxonomy between the test README and the v3 plan (~5-line edit, but needs a source-of-truth decision).
4. Findings 4 and 5 are nits — can be left or batched into the same Finding 2/3 fix commit.

After those fixes, the pre-CP0 setup meets the bar.

---

Reviewed by: code-reviewer

---

## Second-pass audit (2026-05-13 — post-fix verification)

**Verdict: ✅ PASS — all 6 findings resolved; pre-CP0 gate opens; CP1 may begin.**

In plain English: the gating bug from the first pass (the vendored sheeprl algorithm tree was silently gitignored despite living on disk) is fixed — all 232 files are now tracked, so a fresh clone will actually contain `dreamer_v3.py` and its siblings that every per-function test imports. The diff tool's silent-green failure mode (a checkpoint with zero ported functions returning exit 0) is also fixed — CP1 now exits 1 with an explicit "0 functions ran" warning, while checkpoints that are *designed* to be empty (CP8/CP9/CP10 integration/speed gates) still legitimately exit 0 and announce themselves as such. The remaining concerns and nits — opaque shape errors, brittle JAX-array detection, docstring↔output mismatch, CP-id taxonomy drift between the test README and the diff registry — are all cleaned up. CP1 can proceed.

### Per-finding verification

| Finding | Status | Verification |
|---|---|---|
| 1 (gitignore blocker) | ✅ RESOLVED | `git ls-tree -r HEAD -- vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py` returns blob `babcebf8`. `git check-ignore -v <same>` exits 1 (not ignored). `.gitignore:15` reads `/sheeprl/` (root-anchored). `git ls-files vendor/sheeprl/sheeprl/ \| wc -l` = 232, matches on-disk count. |
| 2 (skip-vs-pass) | ✅ RESOLVED | `python scripts/sheeprl_jax_diff.py --checkpoint CP1` prints `WARN: 0 functions ran for CP1 — all were SKIP` and exits 1. `--checkpoint CP8` (designed-empty) prints `CP8: no Lever-A functions registered (integration / speed CP).` and exits 0. The two paths are now distinguishable: the empty-by-design short-circuit at `sheeprl_jax_diff.py:237-240` returns True; the all-SKIP path at `:279-282` forces False. Docstring at `:222-229` documents the contract. |
| 3 (opaque shape error) | ✅ RESOLVED | `scripts/sheeprl_jax_diff.py:132-137` adds an explicit `if jax_out_np.shape != torch_out_np.shape` check that raises `ValueError("Shape mismatch: jax=(...) torch=(...). …")` naming both sides. Reproduced: `compare(np.zeros(3), np.zeros(4), 1e-6)` raises the new message, not numpy's generic broadcast error. |
| 4 (brittle isinstance) | ✅ RESOLVED | `grep "jnp.zeros(0)" scripts/sheeprl_jax_diff.py` returns empty; `grep "isinstance.*jnp"` returns empty. The `if/else` block collapsed to a single `np.asarray(jax_out)` at `scripts/sheeprl_jax_diff.py:127`, exactly the recommended simplification. |
| 5 (docstring↔output) | ✅ RESOLVED | Module docstring at `scripts/sheeprl_jax_diff.py:37-48` now states explicitly that the runner returns `(jax_out, torch_out, metadata)` where `metadata` is a multi-line string and `compare()` prints `fixture: <metadata>` plus diff + PASS/FAIL — matching the actual implementation at `:142-148`. The "former" suggested fix (tighten docstring) was chosen. |
| 6 (CP-id drift) | ✅ RESOLVED | (a) `docs/develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md:502-511` adds a "CP-id convention (canonical — read before editing CP references)" subsection stating Option A: "CP-ids in this table are stable numerical labels, NOT implementation-order slots." (b) `tests/algorithms/dreamer_srl/README.md:63` shows `test_buffers.py             # (no CP — buffers.py is an inter-CP sanity round-trip; see plan §"Implementation order" step 3)` — no `CP3:` label. (c) `:64` shows `test_agent.py             # CP2 (LayerNormGRUCell), CP3 (build_agent zero-init heads), CP4 (RSSM + get_initial_states), CP4b (is_first reset)`. (d) `:66` shows `test_train.py              # CP2b (action_shift), CP6 (critic loss), CP7 (Polyak update)`. (e) `grep -c '"action_shift"' scripts/sheeprl_jax_diff.py` = 1, located at `:186` under `"CP2b"` (absent from CP2). (f) `grep -c '"is_first_force_set"' scripts/sheeprl_jax_diff.py` = 1, located at `:189` under `"CP4b"` (absent from CP4). |

### No regressions

Verified the fix commits did not introduce new issues: `compare()` still handles torch tensors via `.detach().numpy()` at `:121-124`, JAX→numpy conversion at `:127` is uniform, and the `CHECKPOINT_REGISTRY` entries for CP1/CP2/CP3/CP4/CP5/CP6/CP7/CP9b still match the plan's table. The empty-by-design CP8/CP9/CP10 entries are intentional and documented.

### Conclusion

Pre-CP0 gate opens. CP1 may begin.

Reviewed by: code-reviewer
