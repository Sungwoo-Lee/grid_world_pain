---
title: "dreamer-srl v3 — config-correction plan (XL→XS) for the parity launch (D-013 disposition)"
topic: dreamer
status: active
created: 2026-05-14
last_updated: 2026-05-14
phase: 2
---

# dreamer-srl v3 — config-correction plan (XL → XS)

> **Status**: PLANNED
> **Opened**: 2026-05-14
> **Related**:
> - PI disposition that triggers this plan: [`docs/pi/calls/2026-05-14_d013_parity_launch_disposition.md`](../../../pi/calls/2026-05-14_d013_parity_launch_disposition.md) (commit `3c8b9f8`)
> - Deviation closed by this plan: [`DEVIATION_LOG.md` D-013](DEVIATION_LOG.md#deviation-table)
> - Wall-clock measurement that runs after this plan lands: [`CP10B_SPEC.md`](CP10B_SPEC.md)
> - Independent confirmation that sheeprl's 12.5 h baseline is at the real XS: [`docs/experiments/active/sheeprl_bridge/JAX_SHEEPRL_MATCHED_SPS_DESIGN.md`](../../../experiments/active/sheeprl_bridge/JAX_SHEEPRL_MATCHED_SPS_DESIGN.md)
> - Sheeprl XS overlay (source of truth for the corrected values): `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3_XS.yaml`
> - Sheeprl base config (the file that was mis-ported into the dreamer-srl YAML): `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml`

---

## Context

The project is rebuilding a published RL agent called **DreamerV3** — a world-model learner that imagines future observations in a learned latent space and trains its policy on those imagined rollouts — inside our JAX codebase. To call the rebuild "faithful" we have to clear a **parity gate**: the JAX version of DreamerV3 has to reach the same survival performance as the upstream PyTorch reference (the `sheeprl` library) on a small grid-world environment, within a wall-clock budget of about 25 hours per seed (twice the sheeprl baseline's measured 12.5 hours).

Sheeprl ships five named **size presets** — XS / S / M / L / XL — that scale the model's hidden width, recurrent-state width, MLP depth, and CNN channel multiplier. **XS** is the smallest preset (a 256-wide network with a single hidden layer per MLP miniblock) and is what the sheeprl baseline ran. **XL** is the largest (a 1024-wide network with five hidden layers and a 4096-wide recurrent state) and is roughly **16× larger on the dominant recurrent-state axis** than XS.

When the dreamer-srl team ported sheeprl's config into our JAX YAML at `configs/dreamer_srl/01_food_only.yaml`, they copied from sheeprl's **base** config (which carries XL-equivalent numbers) **without applying** the small XS-override file that brings those numbers down. Every dreamer-srl plan since then has called that YAML "the XS default", but it is actually carrying XL-equivalent values — so the 14.4 GB out-of-memory error we saw last week on a single 24 GB GPU was a 16× over-sized XL model being asked to fit an XS budget. The fix is a YAML edit, not an engineering change. After the fix, the model fits trivially on a single GPU, the like-for-like parity claim is preserved (we run what sheeprl runs), and the wall-clock budget drops by roughly an order of magnitude. This plan specifies the edit, the documentation correction-note sweep across the v3 plan docs that mis-named the file as "XS-default", and the updated scope of the wall-clock-measurement checkpoint (CP10b) that follows the fix.

The terms used in the manifest below — `dense_units` (the hidden width of each fully-connected MLP layer), `mlp_layers` (how many of those layers are stacked per miniblock), `recurrent_state_size` (the width of the RNN's hidden vector that carries the agent's running memory of an episode), `cnn_channels_multiplier` (the per-block channel scaling for the convolutional encoder; unused in our pure-MLP grid-world setup but still a sheeprl XS-overlay knob), "**parity target**" (the configuration the parity-launch sweep will actually run at), and "**the 12.5 h baseline**" (the sheeprl reference run at WandB `i4ulpn95`, ep_len_avg ≈ 399.4 steps on this same food-only NoPred env) — are all introduced here so the manifest below can cite them by short name.

---

## Analysis

### Root cause

The dreamer-srl developers ported `configs/dreamer_srl/01_food_only.yaml` from `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml`. That sheeprl file is the **base config** in a Hydra-style override tree — it carries XL-equivalent values, and the smaller sizes are applied as **overrides** on top of it (e.g. `dreamer_v3_XS.yaml` inherits from the base and then overwrites only the dimensions that shrink). The dreamer-srl port did not apply any overlay; it inlined the base values into a single flat YAML file and labeled the result "XS default".

Empirical evidence the mis-port happened (cross-checked against `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml` and `dreamer_v3_XS.yaml`):

| Knob | sheeprl base (XL-equivalent) | sheeprl XS overlay | dreamer-srl `01_food_only.yaml` (current) | What this means |
|---|---|---|---|---|
| `dense_units` | `1024` | `256` | `1024` (six sites) | Hidden width of every MLP layer. The dreamer-srl YAML matches base, not XS. |
| `mlp_layers` | `5` | `1` | `5` (six sites, explicit) | Stacked-layer depth per MLP miniblock. The dreamer-srl YAML matches base, not XS. |
| `world_model.encoder.cnn_channels_multiplier` | `96` | `24` | unset (CNN unused — MLP-only grid-world) | Sheeprl XS knob; in our setup it has no runtime effect, but for documentary completeness the corrected YAML should NOT carry the XL value either. |
| `world_model.recurrent_model.recurrent_state_size` | `4096` | `256` | `4096` | Width of the RNN hidden state. **The dominant memory axis**: XS is `16×` smaller than XL here. |
| `world_model.transition_model.hidden_size` | `1024` | `256` | `1024` | Width of the prior MLP that predicts next-step latent. |
| `world_model.representation_model.hidden_size` | `1024` | `256` | `1024` | Width of the posterior MLP that absorbs the new observation. |
| `world_model.encoder.dense_units` (effective) | `1024` (`${algo.dense_units}`) | `256` (inherited) | `1024` | Confirmed via the per-site comments in `01_food_only.yaml` lines 47–96. |

The mis-named "XS default" wording propagated into the v3 plan docs because every doc consistently referenced the same YAML — the mis-name was internally self-consistent, but every doc was wrong about what the YAML represented.

### Why the disposition is "fix the config", not "change the substrate"

The PI consultation surfaced four options that all assumed the OOM was real on a true-XS model (multi-GPU launch, single-GPU + gradient checkpointing, reduced-dim parity, intermediate-dim compromise) and all therefore required either a substrate change or a parity-claim weakening. The user's investigative push-back exposed the mis-name; with that, the simplest disposition — **edit the YAML to match real XS** — strictly dominates all four:

- **No multi-GPU plumbing** — the JAX driver stays single-GPU.
- **No gradient-checkpointing code path** — `train.py` stays as-is.
- **No parity-claim erosion** — we run what sheeprl runs (256/1/24/256), so the "JAX rebuild matches sheeprl on the same env at the same config" claim is preserved.
- **Faster wall-clock** — at the corrected XS scale the projected per-seed wall-clock drops by roughly an order of magnitude relative to the XL projection, comfortably inside the 25 h budget gate.

### Adjacent-config audit

Checked alongside `01_food_only.yaml`:

- **`configs/dreamer_srl/01_food_only_smoke.yaml`** — already at reduced-dim values (`dense_units=256`, `recurrent_state_size=512`, `mlp_layers=3`, `stochastic_size=8`, `discrete_size=8`, `horizon=7`, `per_rank_batch_size=4`, `per_rank_sequence_length=16`). The header comment block on lines 1–24 calls itself a "REDUCED size for OOM-safe dry-run" and describes its values as smoke-only reductions **from `01_food_only.yaml` (XS sizes)** — i.e. it inherits the same "XS default" mis-name framing. The runtime values are NOT claimed to be sheeprl XS (they are explicitly smaller than what the file calls "XS"), so the file does not need numeric changes; it does need the header comment to be corrected so the framing is no longer "XL-mis-called-XS minus a bit" but "sheeprl-XS minus a bit, for OOM-safe integration smoke". A specific correction-note pattern is in the File Changes section below.

- **`configs/dreamer_srl/agent_xs.yaml`** — cadence-only file (no dimension knobs; only `learning_starts`, `replay_ratio`, `per_rank_*`, `total_steps`, `env.num_envs`, `buffer.size`). Already correct against sheeprl XS; no leakage. **No changes needed.**

**Conclusion of adjacent-config audit:** XL-equivalent leakage is confined to `01_food_only.yaml` (and the mis-name framing in `01_food_only_smoke.yaml`'s header). No other dreamer-srl config carries XL-equivalent values.

---

## Implementation Plan

### Design

Two surfaces of change:

1. **Code-side (one YAML)**: edit `configs/dreamer_srl/01_food_only.yaml` to apply the sheeprl XS overlay to every site that currently carries XL-equivalent values. The full override-set from `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3_XS.yaml` is:

   ```yaml
   dense_units: 256                       # was 1024
   mlp_layers: 1                          # was 5
   world_model:
     encoder:
       cnn_channels_multiplier: 24        # was 96 (no runtime effect — CNN unused)
     recurrent_model:
       recurrent_state_size: 256          # was 4096
     transition_model:
       hidden_size: 256                   # was 1024
     representation_model:
       hidden_size: 256                   # was 1024
   ```

   Every per-site repetition of `dense_units: 1024` (six sites) and `mlp_layers: 5` (six sites) in `01_food_only.yaml` becomes `256` / `1`. The `cnn_channels_multiplier` is added (not currently present in the dreamer-srl YAML); since the CNN path is unused (`cnn_keys.encoder: []`), this is for documentary completeness and parity with the sheeprl XS overlay — it has no runtime effect.

   `stochastic_size: 32` and `discrete_size: 32` (the categorical-state shape that is `S × D = 32 × 32`) are NOT changed — they are NOT in the XS overlay; they live in the sheeprl base and are common across all five sheeprl size presets.

   `horizon: 15` (imagination rollout length) and `bins: 255` (two-hot reward/critic bin count) are likewise NOT changed — neither is in the XS overlay.

   `learning_starts: 1024` is already correct (matches sheeprl XS; restored at CP9b per D-012).

2. **Docs-side (six v3-plan docs)**: add an inline **correction note** at the top of each plan doc that historically mis-named `01_food_only.yaml` as "XS-default". The originals stay verbatim — the corrections are additive, per the PI call's explicit "no silent rewrite" rule. Each correction note links back to this plan + the PI call.

   Plus a **scope update** on `CP10B_SPEC.md`: now that D-013 is dispositioned via config-correction, CP10b runs the same protocol but on the corrected XS config on a single GPU, with the expected wall-clock dropping roughly an order of magnitude relative to the obsolete XL projections.

### File Changes

#### `configs/dreamer_srl/01_food_only.yaml`

**Header comment block (lines 1–11)** — update the "Full XS hyperparameter set" claim so it reflects the corrected values, and add a correction-note paragraph pointing at this plan.

```yaml
# BEFORE (header comment block, current):
# dreamer-srl parity-track config — food-only NoPred 5x5
#
# Full XS hyperparameter set for parity-launch training runs.
# Extends agent_xs.yaml cadence keys with algo + model + optimizer keys.
#
# CP9b: learning_starts restored to 1024 (sheeprl XS default), reverting
#   the D-012 deviation used during CP9 smoke. §S3 random-action prefill
#   is now active; see CP9B_PLAN.md for implementation details.
#
# Sheeprl reference: vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml
#   + vendor/sheeprl/sheeprl/configs/exp/dreamer_v3.yaml

# AFTER (header comment block, corrected):
# dreamer-srl parity-track config — food-only NoPred 5x5
#
# Full sheeprl-XS hyperparameter set for parity-launch training runs.
# Extends agent_xs.yaml cadence keys with algo + model + optimizer keys.
#
# CP9b: learning_starts restored to 1024 (sheeprl XS default), reverting
#   the D-012 deviation used during CP9 smoke. §S3 random-action prefill
#   is now active; see CP9B_PLAN.md for implementation details.
#
# CORRECTION (2026-05-14, PI call docs/pi/calls/2026-05-14_d013_parity_launch_disposition.md):
#   Prior to this date, this file carried sheeprl-XL-equivalent values
#   (dense_units=1024, mlp_layers=5, recurrent_state_size=4096,
#   transition/representation hidden_size=1024) inlined from sheeprl's
#   BASE config dreamer_v3.yaml WITHOUT applying the dreamer_v3_XS.yaml
#   overlay. The mis-port produced the 14.38 GB JIT-compile OOM that
#   surfaced as D-013 at CP9. Per the PI disposition (user verbatim:
#   "Go with XS"), this file now mirrors the real sheeprl XS preset
#   (256 / 1 / 24 / 256 / 256). See CONFIG_CORRECTION_PLAN.md.
#
# Sheeprl references:
#   - Base config (XL-equivalent): vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml
#   - XS overlay (the values applied here): vendor/sheeprl/sheeprl/configs/algo/dreamer_v3_XS.yaml
#   - Experiment defaults: vendor/sheeprl/sheeprl/configs/exp/dreamer_v3.yaml
```

**World-model encoder (lines 46–48)** — apply sheeprl-XS dense width + MLP depth.

```yaml
# BEFORE:
    encoder:
      dense_units: 1024        # sheeprl exp/dreamer_v3.yaml:L31 (encoder.dense_units)
      mlp_layers: 5            # sheeprl exp/dreamer_v3.yaml:L32 (encoder.mlp_layers)

# AFTER:
    encoder:
      dense_units: 256         # sheeprl algo/dreamer_v3_XS.yaml:L5 (XS overlay overrides base 1024 → 256)
      mlp_layers: 1            # sheeprl algo/dreamer_v3_XS.yaml:L6 (XS overlay overrides base 5 → 1)
      cnn_channels_multiplier: 24  # sheeprl algo/dreamer_v3_XS.yaml:L9 (XS overlay; no runtime effect — cnn_keys empty)
```

**World-model decoder (lines 49–51)** — match encoder.

```yaml
# BEFORE:
    decoder:
      dense_units: 1024        # symmetric with encoder
      mlp_layers: 5

# AFTER:
    decoder:
      dense_units: 256         # symmetric with encoder (sheeprl XS overlay)
      mlp_layers: 1
```

**World-model recurrent model (lines 52–54)** — apply the XS recurrent-state width.

```yaml
# BEFORE:
    recurrent_model:
      recurrent_state_size: 4096   # sheeprl exp/dreamer_v3.yaml:L25 (recurrent_model.recurrent_state_size)
      dense_units: 1024            # sheeprl exp/dreamer_v3.yaml:L26 (recurrent_model.dense_units)

# AFTER:
    recurrent_model:
      recurrent_state_size: 256    # sheeprl algo/dreamer_v3_XS.yaml:L11 (XS overlay overrides base 4096 → 256)
      dense_units: 256             # sheeprl algo/dreamer_v3_XS.yaml:L5 (XS overlay overrides base 1024 → 256)
```

**World-model transition + representation (lines 55–58)** — apply XS hidden width.

```yaml
# BEFORE:
    transition_model:
      hidden_size: 1024        # sheeprl exp/dreamer_v3.yaml:L29 (transition_model.hidden_size)
    representation_model:
      hidden_size: 1024        # sheeprl exp/dreamer_v3.yaml:L28 (representation_model.hidden_size)

# AFTER:
    transition_model:
      hidden_size: 256         # sheeprl algo/dreamer_v3_XS.yaml:L13 (XS overlay overrides base 1024 → 256)
    representation_model:
      hidden_size: 256         # sheeprl algo/dreamer_v3_XS.yaml:L15 (XS overlay overrides base 1024 → 256)
```

**World-model reward head (lines 59–62)** — apply XS dense width + depth.

```yaml
# BEFORE:
    reward_model:
      dense_units: 1024        # sheeprl exp/dreamer_v3.yaml:L38 (reward_model.dense_units)
      mlp_layers: 5            # sheeprl exp/dreamer_v3.yaml:L39 (reward_model.mlp_layers)
      bins: 255                # sheeprl dreamer_v3.yaml:L36 (reward_model.bins)

# AFTER:
    reward_model:
      dense_units: 256         # sheeprl algo/dreamer_v3_XS.yaml:L5 (XS overlay; inherits algo.dense_units)
      mlp_layers: 1            # sheeprl algo/dreamer_v3_XS.yaml:L6 (XS overlay; inherits algo.mlp_layers)
      bins: 255                # sheeprl dreamer_v3.yaml:L36 (reward_model.bins; NOT overridden by XS)
```

**World-model continue head (lines 63–65)** — apply XS dense width + depth.

```yaml
# BEFORE:
    continue_model:
      dense_units: 1024        # sheeprl exp/dreamer_v3.yaml:L44 (continue_model.dense_units)
      mlp_layers: 5            # sheeprl exp/dreamer_v3.yaml:L45 (continue_model.mlp_layers)

# AFTER:
    continue_model:
      dense_units: 256         # sheeprl algo/dreamer_v3_XS.yaml:L5 (XS overlay; inherits algo.dense_units)
      mlp_layers: 1            # sheeprl algo/dreamer_v3_XS.yaml:L6 (XS overlay; inherits algo.mlp_layers)
```

**Actor (lines 73–75)** — apply XS dense width + depth. The `dense_units` and `mlp_layers` here are the only two fields that need to change in the actor block; `ent_coef`, `moments`, and `optimizer` remain.

```yaml
# BEFORE:
  actor:
    dense_units: 1024          # sheeprl exp/dreamer_v3.yaml:L49
    mlp_layers: 5              # sheeprl exp/dreamer_v3.yaml:L50

# AFTER:
  actor:
    dense_units: 256           # sheeprl algo/dreamer_v3_XS.yaml:L5 (XS overlay; inherits algo.dense_units)
    mlp_layers: 1              # sheeprl algo/dreamer_v3_XS.yaml:L6 (XS overlay; inherits algo.mlp_layers)
```

**Critic (lines 88–90)** — apply XS dense width + depth. Other critic fields stay.

```yaml
# BEFORE:
  critic:
    dense_units: 1024          # sheeprl exp/dreamer_v3.yaml:L53
    mlp_layers: 5              # sheeprl exp/dreamer_v3.yaml:L54

# AFTER:
  critic:
    dense_units: 256           # sheeprl algo/dreamer_v3_XS.yaml:L5 (XS overlay; inherits algo.dense_units)
    mlp_layers: 1              # sheeprl algo/dreamer_v3_XS.yaml:L6 (XS overlay; inherits algo.mlp_layers)
```

**Not changed in this YAML** (called out explicitly so the `developer` agent does not over-edit):

| Knob | Current | Why preserved |
|---|---|---|
| `algo.learning_starts: 1024` | `1024` | Already correct; sheeprl XS default; restored at CP9b. |
| `algo.replay_ratio: 1` | `1` | sheeprl XS default. |
| `algo.per_rank_sequence_length: 64` | `64` | sheeprl exp/dreamer_v3.yaml — common across all sizes. |
| `algo.per_rank_batch_size: 16` | `16` | sheeprl exp/dreamer_v3.yaml. |
| `algo.total_steps: 5000` | `5000` | Smoke budget (XS default is 5,000,000; parity launch sets explicitly). |
| `algo.horizon: 15` | `15` | sheeprl base — NOT in the XS overlay. |
| `world_model.stochastic_size: 32` / `discrete_size: 32` | `32 / 32` | sheeprl base (32×32 categorical state) — NOT in the XS overlay; all five sheeprl size presets share this. |
| `world_model.reward_model.bins: 255` / `critic.bins: 255` | `255` | sheeprl base — NOT in the XS overlay. |
| `actor.ent_coef`, `actor.moments.*`, `actor.optimizer.*` | unchanged | sheeprl base — NOT in the XS overlay. |
| `critic.tau`, `critic.per_rank_target_network_update_freq`, `critic.optimizer.*` | unchanged | sheeprl base — NOT in the XS overlay. |
| `buffer.*`, `env.num_envs` | unchanged | already XS-correct (verified against `agent_xs.yaml` cadence base). |

#### `configs/dreamer_srl/01_food_only_smoke.yaml`

**Header comment block (lines 1–24)** — correct the framing so the smoke is no longer described as a reduction "from XS sizes" (which were actually XL); after the parent config is corrected to real XS, the reductions in the smoke are now reductions **from real XS, for OOM-safe integration testing**. Numeric values stay; this is documentation only.

```yaml
# BEFORE (lines 1–24):
# dreamer-srl CP9 smoke config — REDUCED size for OOM-safe dry-run
#
# Architecture is structurally identical to 01_food_only.yaml but with
# smaller dimensions that fit on a single RTX 4090 (24GB GPU).
# The full XS dimensions (1024 units, 4096 recurrent, 32x32 stochastic)
# cause OOM at training time; these reduced dimensions are for the
# integration smoke only.
#
# Differences from 01_food_only.yaml (XS sizes):
#   dense_units:           1024 → 256
#   recurrent_state_size:  4096 → 512
#   stochastic_size:       32   → 8
#   discrete_size:         32   → 8
#   mlp_layers:            5    → 3 (encoder/decoder/reward/continue/actor/critic)
#   per_rank_batch_size:   16   → 4
#   per_rank_sequence_length: 64 → 16
#   horizon:               15   → 7
#
# Smoke-only deviation: learning_starts: 0 — keeps the 5,000-step smoke budget
# focused on integration debugging. The parity-track config (01_food_only.yaml)
# uses learning_starts: 1024 (sheeprl XS default) after CP9b restored §S3
# prefill. The zero-init actor (cascade fix #27) acts effectively uniform-random
# for the first ~100 steps, so the smoke's integration-surface coverage is
# unchanged by the local deviation.

# AFTER (lines 1–32, corrected framing — numeric values UNCHANGED):
# dreamer-srl CP9 smoke config — REDUCED size for OOM-safe dry-run
#
# Architecture is structurally identical to 01_food_only.yaml but with
# smaller dimensions that fit on a single RTX 4090 (24 GB GPU) with
# generous JIT headroom. Tuned for integration-smoke speed, not parity.
#
# CORRECTION (2026-05-14, PI call docs/pi/calls/2026-05-14_d013_parity_launch_disposition.md):
#   Prior to 2026-05-14 this comment block claimed the reductions below
#   were "from XS sizes (1024 / 4096 / 32×32)" — but those source values
#   were actually sheeprl XL-equivalent, mis-named as XS in the
#   dreamer-srl plan (see CONFIG_CORRECTION_PLAN.md). After 01_food_only.yaml
#   was corrected to real sheeprl XS (256 / 256 / 32×32 / mlp_layers=1),
#   this smoke config's reductions are now best understood as
#   "OOM-safe-but-still-smaller-than-real-XS" — chiefly to keep horizon,
#   per_rank_batch_size, and per_rank_sequence_length small for fast
#   integration iteration. Numeric values in this file have NOT been
#   changed by the correction; only this comment block has.
#
# Reductions vs. 01_food_only.yaml (corrected real-XS parity target):
#   dense_units:              256 (matches parity target — no reduction)
#   recurrent_state_size:     256 → 512  (slight INCREASE — pre-correction artifact)
#   stochastic_size:          32  → 8    (reduction for integration smoke)
#   discrete_size:            32  → 8    (reduction for integration smoke)
#   mlp_layers:               1   → 3    (slight INCREASE — pre-correction artifact)
#   per_rank_batch_size:      16  → 4    (reduction for fast iteration)
#   per_rank_sequence_length: 64  → 16   (reduction for fast iteration)
#   horizon:                  15  → 7    (reduction for fast iteration)
#
# NOTE: the smoke config is OOM-safe but not minimally-sized; it pre-dates
# the discovery that the parity target itself was mis-sized. A future
# refactor may revisit whether the smoke should match parity-XS on
# dense_units / recurrent_state_size / mlp_layers and shrink ONLY the
# fast-iteration knobs (horizon, batch, seq). Out of scope for this plan.
#
# Smoke-only deviation: learning_starts: 0 — keeps the 5,000-step smoke
# budget focused on integration debugging. The parity-track config
# (01_food_only.yaml) uses learning_starts: 1024 (sheeprl XS default)
# after CP9b restored §S3 prefill. The zero-init actor (cascade fix #27)
# acts effectively uniform-random for the first ~100 steps, so the
# smoke's integration-surface coverage is unchanged by the local
# deviation.
```

No numeric edits in `01_food_only_smoke.yaml`. **The body of the YAML stays unchanged.**

#### Correction-note pattern for the five v3 plan docs

Each of the five docs gets a fenced block inserted **immediately after the frontmatter and immediately before the first body section**, preserving every word of the existing body. The text below is the canonical pattern; the `developer` agent copies it verbatim into each doc with the per-doc anchor adjustment noted below.

```markdown
> **CORRECTION NOTE (2026-05-14, PI call [`3c8b9f8`](../../../pi/calls/2026-05-14_d013_parity_launch_disposition.md))**
>
> References below to "XS-default" / "XS default" / "the full XS configuration" / "the XS config"
> as the content of `configs/dreamer_srl/01_food_only.yaml` **pre-date the discovery** that this
> file was mis-ported from the sheeprl base config (`vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml`)
> rather than the sheeprl XS overlay (`vendor/sheeprl/sheeprl/configs/algo/dreamer_v3_XS.yaml`).
> The base config carries **sheeprl-XL-equivalent** values (`dense_units=1024`, `mlp_layers=5`,
> `recurrent_state_size=4096`, `transition/representation hidden_size=1024`, `cnn_channels_multiplier=96`);
> the real sheeprl XS preset is **`256 / 1 / 256 / 256 / 24`** — i.e. roughly 16× smaller on the
> dominant recurrent-state axis. The 14.38 GB JIT-compile OOM that surfaced as D-013 at CP9 was
> measured at the XL-equivalent values, not at real XS.
>
> **User disposition (verbatim):** *"Go with XS"* — fix `01_food_only.yaml` to mirror the real
> sheeprl XS preset; single-GPU is the natural substrate; no multi-GPU plumbing and no gradient
> checkpointing. See [`CONFIG_CORRECTION_PLAN.md`](CONFIG_CORRECTION_PLAN.md) for the corrected
> values and the file-by-file diff.
>
> **The historical wording below is preserved unchanged** — the correction is additive, per the
> PI call's explicit "no silent rewrite" rule. Read every subsequent "XS-default" / "XS config"
> mention as "the XL-equivalent values then mis-named XS"; the corrected parity target lives in
> [`CONFIG_CORRECTION_PLAN.md`](CONFIG_CORRECTION_PLAN.md) and its post-correction wall-clock
> measurement lives in [`CP10B_SPEC.md`](CP10B_SPEC.md).
```

The five doc paths and per-doc placement:

| # | Doc | Insertion point | Notes |
|---|---|---|---|
| 1 | `docs/develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md` | After the closing `---` of the YAML frontmatter (line 8, just before the H1) | The doc's `last_updated:` frontmatter comment already references this PI call; the inline note is the body-level surfacing. |
| 2 | `docs/develop/active/dreamer_srl_v3/CP9_PLAN.md` | After the closing `---` of the YAML frontmatter (line 8, just before the H1 at line 10) | The doc has 20+ "XS" mentions in the body (lines 46, 77, 99, 301, 351, 416, 443, 448, 643, 671, 784, 870, 920, 922, 924) that the correction note retroactively covers. |
| 3 | `docs/develop/active/dreamer_srl_v3/CP9B_PLAN.md` | After the closing `---` of the YAML frontmatter, before the H1 | The doc references "sheeprl XS default" at lines 26, 275, 279, 302, 622, 695 — most of these are about `learning_starts` (which IS sheeprl XS) rather than dimensions (which were the mis-named XL-equivalents), so the correction note should clarify that the `learning_starts: 1024` references remain correct; only the dimension-bearing references retroactively flip from "XS" to "XL-mis-named-as-XS". An extra sentence is appended to the note for CP9B_PLAN.md specifically — see addendum below. |
| 4 | `docs/develop/active/dreamer_srl_v3/CP10B_SPEC.md` | After the closing `---` of the YAML frontmatter (line 8, before the H1 at line 10) | This doc also gets a scope update — see the "CP10B_SPEC.md scope update" section below. |
| 5 | `docs/develop/active/dreamer_srl_v3/DEVIATION_LOG.md` | The D-013 row's verdict cell already carries an inline correction sub-clause as part of the resolution narrative; the body-level note here goes after the closing `---` of frontmatter (line 8, before the H1 at line 10) and primarily serves rows D-012 and D-013. | D-013 row needs no further textual change; D-012 row's "sheeprl XS default" reference (line 78) is about `learning_starts` and is correct. |

**Addendum line for CP9B_PLAN.md only** (append to the canonical note above):

```markdown
> **CP9B_PLAN.md-specific addendum:** Most "sheeprl XS default" mentions in this doc refer to
> `learning_starts: 1024`, which **IS correctly the real sheeprl XS default** (it is a cadence
> knob, not a dimension knob, and was always correct in `01_food_only.yaml`). Only the
> dimension-bearing mentions (e.g. line 622's framing of `01_food_only.yaml` as "the sheeprl
> XS default" file in dimension terms) retroactively re-read as "XL-mis-named-XS". The §S3
> prefill / `learning_starts` restoration that CP9b implemented is unaffected by this
> correction.
```

#### `CP10B_SPEC.md` scope update

Beyond the correction note, `CP10B_SPEC.md` needs a substantive scope edit to reflect that D-013 is now dispositioned and CP10b's target changes accordingly. The edits are scoped to the **Purpose**, **When CP10b runs**, **What CP10b measures**, and **Acceptance criteria** sections.

**Edit 1 — Purpose section (lines 12–16)**:

```markdown
# BEFORE (lines 12–16):
## Purpose

CP10 closed at the **reduced-dim wall-clock baseline** (9.50 SPS steady-state on a single RTX 4090 at 256 dense units / 8×8 stochastic state / horizon=7) because the **full XS configuration** that sheeprl's reference 12.5-hour baseline used (1024 dense units / 32×32 stochastic / horizon=15) OOMs on a single RTX 4090 during JIT compilation of the training step (peak VRAM demand 14.38 GB, exceeds 24 GB headroom after JAX's 90% pre-allocation policy). The OOM is documented as deviation **D-013** in [DEVIATION_LOG.md](DEVIATION_LOG.md); its disposition (multi-GPU launch vs gradient checkpointing vs intermediate config) is a portfolio-level "what config does the parity launch run at?" question that the **parity-launch PI consultation** is chartered to resolve via [pi.md](../../../../.claude/agents/pi.md)'s "pre-launch of a multi-run experiment" trigger.

CP10b is the structured spec for **converting the CP10 proxy projection into a real like-for-like measurement** of the dreamer-srl JAX rebuild against sheeprl's XS reference, *after* D-013 is dispositioned. It does NOT execute now; it waits on the PI consultation. Once the parity-launch PI consultation chooses a substrate disposition for D-013, CP10b runs the same protocol CP10 ran (a wall-clock-budget measurement) but on the disposed configuration, and reports a like-for-like 12.5h-baseline comparison.

# AFTER (lines 12–16, post-D-013 disposition):
## Purpose

CP10 closed at the **reduced-dim wall-clock baseline** (9.50 SPS steady-state on a single RTX 4090 at 256 dense units / 8×8 stochastic state / horizon=7). The 14.38 GB JIT-compile OOM that motivated the reduced-dim measurement was originally framed as "the full XS configuration cannot fit on a single 24 GB GPU"; the **2026-05-14 PI consultation** ([`2026-05-14_d013_parity_launch_disposition.md`](../../../pi/calls/2026-05-14_d013_parity_launch_disposition.md)) disposed D-013 as a **config-correction** rather than a substrate change — the file mis-named "the full XS configuration" was actually carrying sheeprl-XL-equivalent values (`dense_units=1024`, `mlp_layers=5`, `recurrent_state_size=4096`). After [`CONFIG_CORRECTION_PLAN.md`](CONFIG_CORRECTION_PLAN.md) corrects `configs/dreamer_srl/01_food_only.yaml` to mirror real sheeprl XS (256/1/256/256), the OOM disappears and single-GPU is the natural substrate.

CP10b is now scoped as **the like-for-like wall-clock measurement at the corrected XS config on a single GPU** — confirming both (a) that the corrected config compiles without OOM (the headline empirical closure for D-013) and (b) that the projected per-seed wall-clock lands well inside the ≤ 25 h budget gate (the 41–58 h single-GPU XL projection that CP10's reduced-dim proxy produced is obsolete; real XS projects roughly an order of magnitude faster). If CP10b instead shows the corrected XS config STILL OOMs, escalate back to PI per the disposition's stop rule.
```

**Edit 2 — When CP10b runs (lines 26–32)**: remove the multi-option disposition pathway; the disposition has been made.

```markdown
# BEFORE:
## When CP10b runs

CP10b is **blocked** on these gates, in order:

1. **D-013 disposition** at the parity-launch PI consultation. The PI surfaces 2–4 candidate paths via `AskUserQuestion`; the user decides; the PI logs the call under `docs/pi/calls/`.
2. **Implementation** (if any) of the disposed config. If the disposition is "multi-GPU launch", an implementation plan + developer agent runs first to wire data-parallel scaling into the JAX driver. If "gradient checkpointing", an implementation plan + developer agent runs first to wire `jax.checkpoint` into the world-model and imagination rollouts. If "intermediate config" (e.g. 512 dense / 16×16 stoch / horizon=10 that fits single-GPU but is closer to XS than the CP10 reduced-dim baseline), no implementation is needed — just a config edit.
3. **Senior-developer authorization** to run CP10b. Authorization is granted after gates 1+2 close.

CP10b does NOT block the parity launch. The parity launch can run alongside CP10b; CP10b's purpose is the **published wall-clock measurement**, not a gating decision.

# AFTER:
## When CP10b runs

CP10b is **blocked** on these gates, in order:

1. **D-013 disposition** at the parity-launch PI consultation → ✅ DISPOSED 2026-05-14 ([`2026-05-14_d013_parity_launch_disposition.md`](../../../pi/calls/2026-05-14_d013_parity_launch_disposition.md)). Disposition: config-correction (fix `01_food_only.yaml` to mirror real sheeprl XS).
2. **Implementation of the corrected config** → owned by `developer` per [`CONFIG_CORRECTION_PLAN.md`](CONFIG_CORRECTION_PLAN.md); YAML-only edit, no `src/` or `scripts/` changes; no test regression expected.
3. **Senior-developer authorization** to run CP10b — granted after gates 1+2 close.

CP10b does NOT block the parity launch. The parity launch can run alongside CP10b; CP10b's purpose is the **published wall-clock measurement** and the **empirical confirmation that the corrected XS config compiles without OOM**, not a gating decision.
```

**Edit 3 — What CP10b measures (lines 34–41)**: replace the "disposed configuration" wording with the concrete corrected-XS configuration, and add the "no OOM" criterion to the protocol.

```markdown
# BEFORE (first paragraph):
A 20,000-step (or, by user election, 200,000-step at-target) dreamer-srl smoke on the **disposed configuration**, on the same hardware substrate the parity launch will use, with the same `learning_starts: 1024` setting that the parity-track config carries. The protocol is identical to CP10's:

# AFTER (first paragraph):
A 20,000-step dreamer-srl smoke on the **corrected `configs/dreamer_srl/01_food_only.yaml`** (real sheeprl XS: `dense_units=256`, `mlp_layers=1`, `recurrent_state_size=256`, `transition/representation hidden_size=256`, `cnn_channels_multiplier=24`), on a single GPU (likely a free RTX 6000 Ada on node 114, but any single-GPU node will work since the corrected XS fits comfortably in 24 GB), with the same `learning_starts: 1024` setting that the parity-track config carries. The protocol is identical to CP10's:
```

**Edit 4 — Acceptance criteria (lines 58–65)**: tighten the pass conditions to match the corrected-config scope.

```markdown
# BEFORE (criteria 1):
1. **Run completes**: exit code 0, no crash, no OOM at the disposed config on the disposed hardware.

# AFTER (criterion 1):
1. **Run completes**: exit code 0, no crash, **no OOM at the corrected XS config on a single 24 GB GPU** (this is the empirical closure for D-013 — confirms that the JIT compile fits comfortably on single-GPU once the XL-mis-named-as-XS values are corrected).
```

Criteria 2–6 remain unchanged. The "speed verdict" thresholds (≤ 25 h ✅ / 25–30 h ⚠ / > 30 h ❌) carry through; the only adjustment in spirit is that the **expected** wall-clock is now well inside the green band (the obsolete XL projection was 41–58 h on single-GPU, while real-XS is projected at roughly an order of magnitude faster, so a green ✅ is the strongly-expected outcome).

### Checkpoints

The `developer` agent should verify, during implementation:

- [ ] **CP-A — corrected YAML loads without ConfigError.** Run `from omegaconf import OmegaConf; OmegaConf.load('configs/dreamer_srl/01_food_only.yaml')` and confirm no parse error.
- [ ] **CP-B — all dimensions are at the corrected XS values.** Programmatically verify every site listed in the File Changes table now holds `256` (for `dense_units`, `hidden_size`, `recurrent_state_size`) or `1` (for `mlp_layers`), and `cnn_channels_multiplier: 24` was added under `world_model.encoder`. No site still holds `1024` or `4096` or `5` (mlp_layers).
- [ ] **CP-C — preserved keys are still at their pre-edit values.** Confirm `algo.learning_starts == 1024`, `algo.replay_ratio == 1`, `algo.per_rank_sequence_length == 64`, `algo.per_rank_batch_size == 16`, `algo.horizon == 15`, `world_model.stochastic_size == 32`, `world_model.discrete_size == 32`, `world_model.reward_model.bins == 255`, `algo.critic.bins == 255` — unchanged by this plan.
- [ ] **CP-D — pytest suite is green.** Run the dreamer-srl Lever-A bit-identity suite + offline-check fixture suite:
  ```bash
  /home/vncuser/miniconda3/envs/grid_world_pain/bin/python -m pytest tests/algorithms/dreamer_srl/ -x -q
  /home/vncuser/miniconda3/envs/grid_world_pain/bin/python scripts/dreamer_srl_offline_check.py
  ```
  Expectation: **no regression** — every test passes that passed at the CP9b → CP10 boundary (commit `51822cc` onward). Lever-A and offline-check are config-independent of `01_food_only.yaml` (they consume test fixtures, not the parity-track YAML), so no test outcome should change from this edit.
- [ ] **CP-E — correction-note block lands in all five docs.** Grep for the literal marker string `CORRECTION NOTE (2026-05-14, PI call` in each of `IMPLEMENTATION_PLAN.md`, `CP9_PLAN.md`, `CP9B_PLAN.md`, `CP10B_SPEC.md`, `DEVIATION_LOG.md` — exactly one occurrence per doc.
- [ ] **CP-F — `CP10B_SPEC.md` scope-update edits land.** Grep `CP10B_SPEC.md` for the string `"corrected sheeprl XS"` / `"DISPOSED 2026-05-14"` — confirm Edit 1 / Edit 2 wording landed.
- [ ] **CP-G — `01_food_only_smoke.yaml` body is unchanged.** Diff the file — the only change should be the header comment block (lines 1–24 → 1–32). All keys under `algo:` / `buffer:` / `env:` are byte-identical to the pre-edit version.
- [ ] **CP-H — INDEX regenerated.** Run `python scripts/regen_dev_index.py` so `docs/develop/INDEX.md` updates with the new `CONFIG_CORRECTION_PLAN.md` doc.

---

## Scope of CP10b after the correction

After this plan lands and `developer` commits the edit, CP10b runs with the following concrete scope (operationalising the CP10B_SPEC.md scope-update edits above):

- **Hardware**: single GPU on a free lab node. Node 114's 4× RTX 6000 Ada (49 GB each) are confirmed idle; any single-GPU node (101–114) works since 24 GB is now ample headroom for true XS. Pick at training-runner launch time per the `training-runner` agent's input contract (user confirms node + GPU upfront).
- **Config**: `configs/dreamer_srl/01_food_only.yaml` (corrected; this plan's File Changes section).
- **Step budget**: 20,000 steps (same as CP10) so the wall-clock measurement is directly comparable to CP10's reduced-dim 2347.2 s.
- **Same protocol as CP10**: WandB logging on, name = `dreamer_srl_cp10b_corrected_xs_20k_s<seed>`, capture aggregate + steady-state env-SPS, peak VRAM, NaN count across the 7 Loss/* keys, WM-loss drop step-200 → final, `Diagnostic/moments_invscale` min/max/final, `Params/replay_ratio` convergence to sheeprl-spec, and the iter-1024 debt-repayment burst (per CP9b D-014 F3).
- **New pass condition (vs. CP10)**: JIT compile succeeds without OOM — the headline empirical observation that closes D-013.
- **Speed verdict**: `wall_clock_per_200k = 200000 / steady_state_SPS`. Pass thresholds: ✅ ≤ 25 h (parity launch can proceed); ⚠ 25–30 h (user discusses with PI); ❌ > 30 h (escalate per stop rule).
- **Expected outcome**: ✅ comfortably inside 25 h — at the corrected XS scale the per-step compute drops by roughly an order of magnitude relative to the obsolete XL projection. A wall-clock of less than 5 h for 200k steps is plausible (back-of-envelope from the reduced-dim 9.5 SPS × the partial scale-up to real XS), though the actual number is what CP10b measures.

---

## Execution chain (hand-off sequence)

After this plan is approved and committed:

1. **`developer`** — apply the YAML edits per the File Changes section above, apply the five correction-note insertions, apply the `CP10B_SPEC.md` scope-update edits, run pytest + offline-check, commit. Implementation Report goes at the bottom of this doc.
2. **`senior-developer`** (a follow-up invocation) — verify the implementation against the File Changes manifest + CP-A–CP-H checkpoints; fill the Verification Report below; sign off.
3. **`training-runner`** — launch CP10b on a single GPU per the "Scope of CP10b" section above; collect wall-clock + sanity-pass conditions; report back.
4. **`senior-developer`** (a follow-up invocation) — verify CP10b → CP-PASS via the standard speed-verdict protocol; if PASS, flip the CP10b row in `IMPLEMENTATION_PLAN.md` to ✅ and update `CP10B_SPEC.md`'s verification subsection.
5. **`experiment-designer`** — author the parity-launch sweep configs (3 seeds × the corrected XS `01_food_only.yaml` × different env seeds), in `docs/experiments/active/sheeprl_bridge/`.
6. **`training-runner`** — launch the parity-gate sweep on the corrected XS config (single-GPU per seed).
7. **`experiment-analyzer`** — after the parity launch finishes, analyse the 3-seed result and write the parity verdict in `docs/experiments/active/sheeprl_bridge/`.

Each hand-off step gates on the previous; CP10b and the parity-launch sweep run sequentially (CP10b's empirical "compiles without OOM at corrected XS" + speed verdict gate the parity launch).

---

## Risks and rollback

- **Risk 1 — the corrected XS config STILL OOMs on single GPU.** *Probability:* low. The dominant memory axis (`recurrent_state_size`) drops 16× from 4096 → 256; the per-MLP-layer width drops 4× from 1024 → 256; the MLP depth drops 5× from 5 → 1. The 14.38 GB measurement was at the XL-equivalent values; at true XS the projected compile cost is well under 5 GB. *Mitigation:* the PI call doc's "Stop rule for next-PI escalation" already pre-declares that if this happens, the option set in the disposition re-opens and option A (multi-GPU launch) becomes the natural fallback — escalate to PI before deferring.
- **Risk 2 — the corrected XS config trains too slowly to hit the survival-step target.** This is a **different concern from D-013** — a learning-dynamics question that surfaces at the parity-analysis stage (step #7 of the execution chain), not at the wall-clock-budget stage. The corrected XS values exactly mirror what sheeprl ran at the 12.5 h baseline (ep_len_avg ≈ 399.4 steps on this same env), so a learning-dynamics failure here would be a JAX-port-faithfulness failure, not a config-size failure. Out of scope for this plan; covered by the parity-gate analysis downstream.
- **Risk 3 — adjacent dreamer-srl configs carry XL-equivalent leakage too.** *Probability:* low, addressed by the adjacent-config audit in the Analysis section above (verified `agent_xs.yaml` cadence-only and clean; `01_food_only_smoke.yaml` already at reduced dims with no parity claim). *Mitigation:* if the post-correction CP10b run still shows unexpectedly high memory or wall-clock, file a follow-up correction PR and surface to PI before launching the parity sweep.
- **Risk 4 — the docs-side correction-note sweep silently rewrites historical content.** *Probability:* low if the `developer` agent follows the canonical pattern in the File Changes section. *Mitigation:* CP-E + CP-F checkpoint greps confirm the additive note landed AND the original body text is unchanged (verifiable by `git diff` line-count: the correction note adds ~25 lines per doc; no lines below the note should change).
- **Rollback procedure** if the corrected XS config has any unforeseen failure mode: revert `configs/dreamer_srl/01_food_only.yaml` to the pre-correction commit via `git revert <commit_hash>`; the correction-note blocks in the v3 plan docs can stay (they remain accurate as a historical record); CP10b's spec edits can stay or revert depending on the failure mode. File a new PI call to dispose D-013 from option A / B / C / D.

---

## Estimate

Half a day of `developer` time for: the YAML edit (one file, ~12 sites), the correction-note insertions (five docs), the `CP10B_SPEC.md` scope-update edits, the `01_food_only_smoke.yaml` header rewrite, pytest + offline-check (~60 s combined), commit. Plus `senior-developer` verification (~15 min) + CP10b launch via `training-runner` (~5 min to start, then wall-clock measured by CP10b itself).

**The parity-launch sweep itself** is the long wall-clock step (~25 h × 3 seeds), but that's **not part of this plan** — it's step #6 of the execution chain, gated on CP10b PASS.

---

## Implementation Report

> **Implemented by**: [TBD — `developer` agent]
> **Date**: [TBD]

<!-- Filled by the `developer` agent after the YAML edit + correction-note sweep + CP10B_SPEC.md scope-update edits land.
     Describe what was done, any deviations from the plan, and why. Cite commit hash(es). -->

---

## Verification Report

> **Verified by**: [TBD — `senior-developer`]
> **Date**: [TBD]

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `configs/dreamer_srl/01_food_only.yaml` | XL-equivalent values → real sheeprl XS at every site listed in File Changes | | |
| `configs/dreamer_srl/01_food_only_smoke.yaml` | Header comment block corrected; body unchanged | | |
| `docs/develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md` | Inline correction note added after frontmatter | | |
| `docs/develop/active/dreamer_srl_v3/CP9_PLAN.md` | Inline correction note added after frontmatter | | |
| `docs/develop/active/dreamer_srl_v3/CP9B_PLAN.md` | Inline correction note + CP9B-specific addendum added after frontmatter | | |
| `docs/develop/active/dreamer_srl_v3/CP10B_SPEC.md` | Inline correction note + scope-update edits (Edits 1–4) | | |
| `docs/develop/active/dreamer_srl_v3/DEVIATION_LOG.md` | Inline correction note added after frontmatter (D-013 row already carries inline resolution) | | |
| `docs/develop/INDEX.md` | Regenerated by `scripts/regen_dev_index.py` to pick up `CONFIG_CORRECTION_PLAN.md` | | |

**Speed-change review**: not applicable at the senior-developer verification step — the YAML edit is config-only and does not change `src/` code. The runtime-speed verification happens at **CP10b** (single-GPU corrected-XS wall-clock measurement, separate sign-off).

**Conclusion**: [one-line summary at sign-off]

---

<!--
NEW ISSUES: If a new issue is discovered during implementation/verification:
- If closely related: append as "## Issue #2: [title]" below with the same template sections.
- If independent: create a separate document and cross-reference.
-->
