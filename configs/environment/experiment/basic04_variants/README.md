# basic04 variants — softened versions of the jump level

## What these are

`configs/environment/experiment/basic/04-jump_attack_10x10.yaml` is the "jump" level of the
basic curriculum: a chasing predator that can **pounce** (teleport onto the agent) from a few
cells away. In its current tuning the level is very harsh — the predator always moves at the
agent's own speed (so it cannot be outrun), it can pounce from up to 3 cells away (so keeping
your distance barely helps), and a single hit can do up to 120 injury when the body dies at 100
(so one landed pounce can kill outright from full health).

The four configs here each relax **one** of those pressures (plus one that relaxes all three),
so we can find out which knob was actually making the level unlearnable — rather than guessing.

**`basic/04` itself is unchanged.** It stays the harsh baseline; these are additive siblings.

## The files

| File | What it changes vs `basic/04` | Intent |
|---|---|---|
| `01-slow_move_interval.yaml` | predator step cadence `move_interval` `[1,1]` → `[1,3]` | the predator is sometimes slower than the agent, so fleeing can work again |
| `02-short_attack_range.yaml` | pounce reach `attack_range` `[2,3]` → `[1,2]` | a 2-3 cell standoff is genuinely safe again, so distance-keeping is a learnable defence |
| `03-reduced_damage.yaml` | hit damage `damage` `[15,120]` → `[15,80]` | removes the one-shot kill (80 < `body.max_injury` = 100); every hit is survivable from full health |
| `04-all_combined.yaml` | all three of the above together | the fully-softened variant — "can the agent learn this level at all?" |

Everything else is inherited from `basic/04` and verified unchanged: 50% pounce hit rate,
detection range `[1,7]`, stamina `[30,150]`, attack delay `[1,3]`, 0-2 chasing predators, the
wandering rabbit, 2-12 ambush predators, 4-10 bushes, random start injury/nutrition, and no
sensory noise.

## The list-replace footgun (why each file restates the whole entity list)

`environment.entities` is a YAML **list**. A child config that redeclares it **replaces the
entire list** — it does not merge entry-by-entry. So each variant restates the complete list
(the predator **and** the rabbit), copied verbatim from `basic/04` except for the changed
line(s). Nothing else is redeclared: `resources`, `obstacles`, `body`, and `perceptual_noise`
stay inherited so they cannot silently drift. (An earlier `basic05_variants` attempt was broken
by exactly this footgun.)

## Verification performed

Each variant was resolved with `load_env_config` + `load_env_params` and deep-diffed against a
resolved `basic/04`. The only resolved differences are the intended fields:

- `01`: `move_interval` only → `animal_move_int_high` `[1,1,1,1]` → `[3,3,1,1]`
- `02`: `attack_range` only → `animal_attack_range_low/high` `[2,2,0,0]/[3,3,0,0]` → `[1,1,0,0]/[2,2,0,0]`
- `03`: `damage` only → `animal_damage` predator rows `[15,120]` → `[15,80]`
- `04`: all three of the above, and nothing else
