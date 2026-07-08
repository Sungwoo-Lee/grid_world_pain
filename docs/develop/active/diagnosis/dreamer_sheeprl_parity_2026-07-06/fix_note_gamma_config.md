---
title: "Fix note — discount-factor (gamma) typo in all live dreamer_srl configs (WP-GAMMA / P4 / D-02)"
topic: diagnosis
status: active
created: 2026-07-08
last_updated: 2026-07-08
---

# Fix note: gamma typo in the dreamer_srl configs

## Purpose

Every live config for our JAX Dreamer reimplementation ("dreamer_srl") was training with a
slightly-too-small discount factor: `0.996840347` instead of the `0.996996996996997` used by
the sheeprl reference implementation we claim parity with. The discount factor controls how far
into the future the agent's value estimates look — the reference value gives an effective
credit-assignment horizon of 333 steps (because 1/(1−γ) = 333), while our typo'd value gives
only ~316.5 steps, about 5% shorter. Worse, the comment next to the wrong value cited the
sheeprl reference file as its source, which was false: the value was hand-inlined during
planning (traced to `CP9_PLAN.md:441`) and never matched the reference. This silently broke the
"same recipe as the benchmark" premise of every strict-parity comparison against sheeprl
baselines.

This note records the one-line-per-file config fix: 18 YAML files corrected, the false citation
comment replaced with an accurate one, and a grep confirming no stale value remains. The bug was
found by the 2026-07-06/08 Dreamer-vs-sheeprl parity audit (master-comparison finding **P4**;
area-report finding **D-02** in both the actor-critic-returns and training-loop-replay reports).

## Correct value and derivation

- **Reference:** `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml`, line 11:
  `gamma: 0.996996996996997`
- **Derivation:** γ = 1 − 1/333 = 332/333 = 0.996996996996997 (DreamerV3's canonical
  "horizon 333" discount).
- **Old (wrong) value:** `0.996840347` → 1/(1−γ) ≈ 316.5 steps — no principled derivation;
  provenance traced by the audit to a hand-inlined constant in `CP9_PLAN.md:441`.
- **Comment fix:** old comment read `# sheeprl dreamer_v3.yaml:L22` (wrong line, and the value
  didn't match that file anywhere). New comment: `# = 1 - 1/333; sheeprl dreamer_v3.yaml:L11`.

## Files changed (18)

All under `configs/models/dreamer_srl/`, each a single-line change to `algo.gamma`:

| File | Line |
|---|---|
| `01_food_only.yaml` | 41 |
| `01_food_only_L.yaml` | 40 |
| `01_food_only_M.yaml` | 40 |
| `01_food_only_M_seqlen128.yaml` | 53 |
| `01_food_only_M_seqlen32.yaml` | 53 |
| `01_food_only_S.yaml` | 40 |
| `01_food_only_S_seqlen128.yaml` | 53 |
| `01_food_only_S_seqlen32.yaml` | 53 |
| `01_food_only_XL.yaml` | 48 |
| `01_food_only_buf100k.yaml` | 41 |
| `01_food_only_buf256k.yaml` | 41 |
| `01_food_only_buf256k_log50k.yaml` | 41 |
| `01_food_only_buf256k_log5k.yaml` | 41 |
| `01_food_only_buf500k.yaml` | 41 |
| `01_food_only_buf50k.yaml` | 41 |
| `01_food_only_seqlen128.yaml` | 54 |
| `01_food_only_seqlen32.yaml` | 54 |
| `01_food_only_smoke.yaml` | 52 |

Notes on the set:

- The parity audit said "all 19 live configs"; the directory holds 19 YAMLs but only **18**
  carry a `gamma` key. The 19th, `agent_xs.yaml`, is the CP3b cadence-parity key overlay
  (learning_starts / replay_ratio / batch-and-sequence shape / total_steps only) and defines no
  gamma — nothing to fix there. **No file was skipped for carrying a deliberate nonstandard
  gamma**: all 18 held the identical typo'd value, none is a horizon ablation.
- `01_food_only_smoke.yaml` was the only file whose gamma line had no citation comment; it now
  carries the same corrected value + comment as the rest.

## Training effect

- Effective credit-assignment horizon on the value/λ-return targets lengthens from ~316.5 to
  333 steps (~5%). Small but systematic; present in **every** dreamer_srl run to date at every
  env count, including the `num_envs=1` parity launches.
- Restores the discount side of the "same recipe as the benchmark" claim for strict-parity
  comparisons against sheeprl baselines.

## Comparability caveat

This gamma fix ships together with the parallel **WP-SRL code fixes** (loss-clipping /
observation-loss / metric-bleed corrections from the same parity audit) as **one dreamer_srl
comparability epoch**. Runs launched after this epoch are **not comparable** to any earlier
dreamer_srl run — do not mix pre-fix and post-fix runs in the same analysis, learning-curve
overlay, or baseline table. This mirrors the earlier H4–H7 comparability break. The parent
session commits this config change and WP-SRL as one unit so the epoch boundary is a single
commit.

## Verification

Grep over the whole `configs/models/dreamer_srl/` tree after the edit — no file still carries
the old value (grep exits 1 = zero matches):

```
$ grep -rn "0.996840347" configs/models/dreamer_srl/ ; echo "exit=$?"
exit=1
```

Diff audit: `git diff` over the directory shows exactly 18 files, 1 line each — 17× the
commented form and 1× the bare form (smoke) removed, 18× the corrected line added. Nothing else
changed.

## Observed but out of scope

The neighboring citation comments on `lmbda` / `horizon` / `unimix` in the same files cite
`sheeprl dreamer_v3.yaml:L23/L24/L25`, but those keys live at lines 12–14 of the vendored file.
Their **values** match the reference (0.95 / 15 / 0.01 — audit rows all PARITY), so these are
stale line-number comments only, not value bugs. Left untouched per the WP-GAMMA scope fence;
flagging here for a future comment-hygiene pass.

## Links

- Audit master doc: [[00_master_comparison]] §3 row P4
- Area reports: [[03_actor_critic_returns]] (headline finding), [[04_training_loop_replay]]
  (D-02, incl. provenance of the wrong constant)
- Reference config: `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml:11`
