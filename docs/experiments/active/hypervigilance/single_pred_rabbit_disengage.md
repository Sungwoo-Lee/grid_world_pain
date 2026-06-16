---
title: "Single-predator vs single-rabbit, strike-and-retreat, differ only in damage"
topic: hypervigilance
status: active
created: 2026-06-09
last_updated: 2026-06-09
phase: hypervigilance
wandb_tag: hypervigilance
develop_link: "[[06-matchedAggression_chasingRabbit_eval]]"
---

# Single predator vs single rabbit — cleanest danger-discrimination contrast (cell 08)

## Question / Purpose

Does an agent treat a harmful animal differently from a harmless one when the two are
**behaviourally indistinguishable from the outside** — same chase, same smell, same
strike-and-retreat — and the *only* thing separating them is whether contact actually hurts?
And, now that a single contact with the harmful animal can be **instantly lethal**, does the
agent's avoidance become more **anticipatory** (steering clear before contact) rather than
reactive (only avoiding after being hit)?

This is the cleanest predator-vs-rabbit comparison in the hypervigilance line. Earlier cells
in this lineage matched the predator's chase parameters to harmless "chasing rabbits" to remove
an approach-speed confound, but still ran a 2-rabbit corner layout plus four extra hidden
damage sources. This config strips all of that to a **single predator + single rabbit**, both
roaming the whole grid, byte-identical on every observable and chase-driving field. If the agent
still behaves differently toward the two, that difference can only come from inferring danger.

A "predator" here is an animal whose touch injures the agent; a "rabbit" is the same kind of
chasing animal that does no harm. "Strike-and-retreat" means that on contact the animal's
energy (stamina) drains to zero, so it must back off and recover rather than pin the agent —
both animals now do this, so the retreat pattern is matched too.

## Two environment changes vs the prior lineage

1. **Disengage-on-contact, on both animals.** A feature shipped today: when the animal touches
   the agent, its stamina is forced to zero, producing a hit-and-back-off rhythm instead of a
   continuous pin. Both the predator and the rabbit get it, so the strike-and-retreat dynamic is
   identical between them — it cannot be a tell for which one is dangerous.

2. **Wide predator damage range, single contact can be lethal.** The predator's per-contact
   injury is drawn uniformly from a wide band (low 5, high 120 on a 0–100 injury scale where 100
   = death). Roughly one contact in six lands at or above the lethal threshold, so a single
   mistake can end the episode. This raises the stakes of contact and is what lets us test whether
   avoidance shifts from reactive to anticipatory. The rabbit remains completely harmless.

## What is matched vs what differs (validated through the loader)

Loaded through `config_loader.load_env_params`; per-entity printout pasted in the manifest
section below. Matched between predator and rabbit: smell signature, behaviour (both hunt),
detection range, stamina, stamina recovery, hunt-threshold, lose-interest persistence, move
interval, attack delay, disengage-on-contact, and full-grid spawn + patrol. They differ in
exactly three coupled things: the animal class (which sets whether it can damage, and a separate
by-class visual channel the agent sees only on contact), the damage range, and the nociceptive
signature. Nothing the agent can sense *at a distance* tells the two apart.

## Confounds removed

- The four hidden-predator damage sources from the cell-06 layout are **removed** — they were a
  second, uncontrolled source of injury that would muddy a clean one-vs-one. (Cell 05 made the
  same removal.)
- The bush obstacles are **kept** — the agent needs cover to express a hide/bush-dive response,
  and one of the behaviour measures requires bushes that conceal the agent.
- Food is a **simple symmetric forage task** (four quadrants, two food sources each) so survival
  is a real forage-vs-avoid trade-off, not a starvation cap, and the layout is not obviously
  asymmetric the way the old decoupled-corner food was.

## Design (locked)

- **Type:** training config, trains from scratch (no checkpoint-load constraint).
- **Headline dependent variable:** survival steps (never cumulative reward).
- **Behaviour measures:** kept enabled so the trained model can later be analysed for
  approach/avoid, bush-diving, and motif structure toward the predator vs the rabbit.
- **Independent variable under test (within the trained agent's behaviour):** harmful vs harmless
  animal — everything else about the two is held identical.

### Configs to produce

| Run | Env config | Agent config |
|---|---|---|
| single-pred-vs-rabbit, disengage, lethal predator | `configs/experiment/hypervigilance/08-singlePredRabbit_disengage.yaml` | recurrent_ppo (project default) |

Multi-seed by project convention (>= 3 seeds) when promoted to a production sweep; the initial
launch in the handoff is a single named seed (42) to confirm the environment trains cleanly
before fanning out.

## Loader validation (pasted)

```
animal_classes               = ('predator', 'neutral')
animal_is_damaging           = [True, False]
hunt_idx                     = (0, 1)
animal_disengage_on_contact  = [True, True]
animal_tags                  = ('full', 'rab')

PER-ENTITY TABLE (idx 0 = predator/full, idx 1 = rabbit/rab):
field                       ent0             ent1
detection_range             (10.0, 10.0)     (10.0, 10.0)
max_stamina                 (60.0, 60.0)     (60.0, 60.0)
stamina_recovery_rate       (1.0, 1.0)       (1.0, 1.0)
hunt_stamina_threshold      (0.3, 0.3)       (0.3, 0.3)
lose_interest_multiplier    (3.0, 3.0)       (3.0, 3.0)
move_interval               1                1
attack_delay                3                3
disengage_on_contact        True             True
-- DIFFERING FIELDS --
damage [lo,hi]              [5.0, 120.0]     [0.0, 0.0]
nociception_intensity       0.9              0.0
visual_channel              5                7
smell properties            [0,1,0,0,0]      [0,1,0,0,0]

behavior_measures.enabled    = True
bush count (hides_agent=true)= 12
hiding_predator resources    = 0
total food sources           = 8
```

All core assertions passed: predator and rabbit are identical on chase + smell + disengage, and
differ only in damage / nociception / visual channel. `disengage_on_contact: true` parsed for
both animals. Behaviour measures enabled; hiding bushes survive; hidden-predator resources gone.

## Results / Conclusions

_To be filled after training + analysis._
