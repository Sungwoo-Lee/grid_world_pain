---
title: "NNX DreamerV3 recipe-deviation register — kept / fixed / open dispositions"
topic: diagnosis
status: active
created: 2026-07-08
last_updated: 2026-07-10
---

# NNX DreamerV3 recipe-deviation register

> **Status note (2026-07-10)**: the DreamerV3-NNX stack this register describes was
> **archived** to `src/models/archive/dreamer_v3_nnx/` per
> [[archive_plan_dreamer_v3_nnx]]. This register is **frozen**; the open rows
> **U6 and R1 are moot unless the stack is revived**. File paths below refer to
> the pre-move locations (now under the archive).

## What this register is (plain-language entry point)

The in-house DreamerV3 agent (the JAX/Flax-NNX world-model agent under
`src/models/`, trained by `train.py`) was never a port of a reference
implementation, so — unlike the `dreamer_srl` package, which has a
DEVIATION_LOG — it never had a single place recording where it consciously
departs from the canonical DreamerV3 recipe (the vendored sheeprl reference +
Hafner 2023). This register is that place: **every deviation found by the
2026-07-06→08 audit ([[05_dreamer_v3_nnx_conventions]]) that we consciously
KEEP, plus the disposition of the rows that work package WP-NNX
([[fix_plan_nnx_parity]]) fixed**. If a future audit or reviewer asks "does
the NNX stack know it differs from the recipe here?", the answer should be a
row in this table.

Landing WP-NNX opened a **new comparability epoch** for the NNX stack
("NNX parity epoch, 2026-07 / WP-NNX"): post-fix runs are not comparable with
pre-fix runs (actor gradient composition, world-model loss balance,
value-learning style, replay accounting, and discount weighting all changed).

## Register

| Row | Item (from [[05_dreamer_v3_nnx_conventions]]) | Status after WP-NNX |
|---|---|---|
| U1, U3, K1 | stop-gradients / obs-loss sum / is_first collection reset | FIXED — WP-NNX F2/F3/F1, 2026-07-08 (single WP-NNX package commit; ref filled at commit time) |
| U2, U4, U5 | online bootstrap + slow-critic regularizer / replay-ratio per-env-step semantics + random prefill / true-continue start weights | FIXED — WP-NNX F4/F7/F5, 2026-07-08 (same package commit; U4 note: all live `configs/models/dreamer_v3/*` `replay_ratio` values were rescaled ÷128 by `experiment-designer` in the same change window, and a mandatory `agent.learning_starts: 1024` key was added — cost-neutral by construction, the knob is now honest) |
| U6 | decoder trailing LayerNorm (recipe: bare Linear) | **OPEN — fix STOPPED at the WP-NNX gate.** The plan's shared-module guard found `DreamerGroupedMLP` is instantiated by BOTH the hierarchical encoder (`dreamer_v3_nnx.py:259`) and the hierarchical decoder (`:434`); per the parent ruling on this contingency the fix was halted and flagged instead of taking the constructor-flag route. Any future fix must gate the trailing LN per-instantiation (encoder keeps it; recipe encoders DO end in LN) and accepts a decoder param-tree change (old checkpoints unrestorable) |
| C1 | LayerNorm eps 1e-6 vs recipe 1e-3 | KEPT — cosmetic; flax default |
| C2 | `hafner_init` effective std ≈ 0.77× recipe; trunc-normal on heads where sheeprl gives uniform | KEPT — init-time only |
| C3 | zeros initial recurrent state vs learnable `tanh(param)` | KEPT — interacts with E2 equivalence; revisit only if learnable init is ever added |
| C4 | no pre-GRU projection LayerNorm | KEPT — cosmetic |
| C5 | dead `agent.unimix` YAML knob (hardcoded 0.01) | KEPT — flag to experiment-designer for eventual config cleanup |
| C6 | actor/critic lr 3e-5 vs 8e-5; seq_len 128 vs 64 | KEPT — declared knobs, comparative-alignment choice |
| C7 | block-aligned sequence sampling + mixture pools | KEPT — declared DreamerV4-inspired extension |
| C8 | HORIZON/GAMMA/LAMBDA/FREE_NATS hardcoded constants | KEPT — recipe values; config-exposure is a separate refactor |
| K2–K7 | eval path, unimix placement, checkpoint omissions, collect_interval validation, target-critic init, PRNG hygiene | OPEN — owned by [[KNOWN_BUGS]]; referenced, not duplicated (K6 note: U2's fix makes the regularizer the target critic's ONLY role, so its fresh-random init matters slightly more early in training — benign while `zero_init_reward_critic: true`) |
| R1 | imagined continues enter the λ-return and the cumulative discount weights as **sigmoid PROBABILITIES** (`dreamer_v3_trainer.py:473-474,498-499,533,546`); the recipe takes the **Bernoulli MODE** — hard 0/1 at p=0.5 (`Independent(BernoulliSafeMode(...)).mode`, vendor `dreamer_v3.py:246`), so ours applies a soft geometric decay where sheeprl applies a hard cutoff | **OPEN — declared 2026-07-08 (post-WP-NNX review, [[review_nnx_parity_fixes]] finding 2), severity Low-Med.** Pre-existing — NOT introduced by WP-NNX, but it rides the exact lines F4/F5 modified. Fix shape if taken: `cont = (sigmoid(...) > 0.5).astype(f32)` at the two `scan_imag` sites; declare-or-fix decision belongs to senior-developer. Note a fix opens another comparability epoch for high-uncertainty continue predictions |

## Maintenance

- New audits of the NNX stack add rows here (or flip dispositions) in the
  same change that lands the audit/fix.
- Rows K2–K7 are tracked in the Known Bugs registry; this table only points
  at them so the recipe picture stays complete in one glance.
