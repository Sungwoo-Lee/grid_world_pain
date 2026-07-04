# Archive — basic ladder re-level (2026-07-04)

Snapshot of configs retired when the `basic/` curriculum was cut from an 8-level to a
clean 6-level ladder. These files are historical and are no longer loaded by any live
config — do not resurrect them; author against the new ladder instead.

## New basic/ ladder (live)

| Level | Config | Notes |
|---|---|---|
| 00 | `00-static_predator_5x5` | static predator |
| 01 | `01-slow_predator_5x5` | slow predator |
| 02 | `02-predator_and_rabbit_10x10` | fast predator + wandering rabbit (was old-03) |
| 03 | `03-random_init_10x10` | all-combined random pressure + random body init (was old-05) |
| 04 | `04-jump_attack_10x10` | jump/pounce, attack_range {2,3}; extends 03, NO noise (was old-07) |
| 05 | `05-sensory_noise_10x10` | hardest: jump + injury-gated noise; extends 04 (was old-06) |

Chain: `03 → 04 (jump) → 05 (noise)`. Jump now precedes noise.

## What was retired here and why

- **`02-fast_predator_8x8.yaml`** — folded into the fast-predator+rabbit level (new-02).
- **`04-far_sight_predator_10x10.yaml`** — its fixed far detection is subsumed by new
  level-03's RANDOM `detection_range [1,7]`.
- **`basic05_variants/` (6 yaml + README)** — predator-pressure variants of the old
  basic-05. Now redundant against the new ladder:
  - `01-more_hiding_predators` and `04-all_combined` ≡ new level-03 (hiding 2–12 etc.
    already folded into the all-combined random-init scene).
  - `05-all_combined_noise` ≡ the old noise level (now new level-05).
  - `06-jump_range_2to3` — its wider jump reach is now the **basic default**:
    new level-04 ships `attack_range [2,3]` ({2,3} under inclusive-integer sampling).
  - `02-relentless_stamina` and `03-fast_move_interval` — their single factors
    (`max_stamina [30,150]`, `move_interval [1,1]`) are already randomized inside new
    level-03.

## Dangling `extends:` note

The archived `basic05_variants/*` still carry `extends:` pointing at the pre-rename live
paths (`basic/05-random_init`, `basic05_variants/04-all_combined`, `basic/07-jump_attack`).
These are intentionally left as historical snapshots and are NOT repointed — the variants
are retired and never loaded. No LIVE config references any moved path.
