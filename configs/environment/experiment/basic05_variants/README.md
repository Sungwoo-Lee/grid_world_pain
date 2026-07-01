# basic05_variants — predator-pressure variants of basic level 05

Variants of [`basic/05-random_init_10x10.yaml`](../basic/05-random_init_10x10.yaml), kept
separate so the `basic/` folder stays the canonical collection. Each variant `extends:`
basic-05 (which itself `extends: environment/default`, so default + basic-05 + variant are
all merged), changing **one factor** — except `04` which combines all three.

**Motivation:** the basic-05 randomized-predator agent learned to *run around the world*
rather than hide in bushes (unlike the fixed-predator agent, which hid). These variants test
whether tougher predator strategies push the agent back toward hiding.

| Config | Factor changed | vs basic-05 |
|---|---|---|
| `01-more_hiding_predators` | hiding_predator count | `2–4` → **`2–12`** (more ambush spots → running is risky) |
| `02-relentless_stamina` | predator `max_stamina` | `[30,30]` → **`[30,150]`** (high draws chase ~150 steps → toward starvation) |
| `03-fast_move_interval` | predator `move_interval` | `[1,3]` → **`[1,1]`** (always full-speed, no slow episodes) |
| `04-all_combined` | all three above | — |
| `05-all_combined_noise` | variant-04 + level-06 sensory noise | all-combined predators **plus** injury-gated olfactory noise (hardest hypervigilance probe) |

Everything else (counts, random start nutrition/injury, detection_range `[1,7]`,
attack_delay `[1,3]`, damage, smell, obstacles) is inherited from basic-05 unchanged.

Verified on creation: each variant loads, changes only its intended factor, inherits the
basic-05 scene + body random-init, and (factor 2) `max_stamina` linearly controls chase
duration (30→~30, 150→~150 hunt steps).
