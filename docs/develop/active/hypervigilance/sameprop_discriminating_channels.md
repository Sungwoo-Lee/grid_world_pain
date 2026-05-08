---
title: "SameProp obs-space discriminators (rabbit vs predator)"
topic: hypervigilance
status: active
created: 2026-05-07
last_updated: 2026-05-07
---

# SameProp Discriminating Channels — Diagnostic Memo

**Question.** Under [`configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml`](../../../../configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml), rabbit and predator share `properties [0,1,0,0,0]`. Yet the RPPO agent still avoids predators (and not rabbits, per user observation). Which obs-space channels actually differ?

**Method.** Read-only audit of [`src/environment/sensor.py`](../../../../src/environment/sensor.py) and [`src/environment/core.py`](../../../../src/environment/core.py). Citations are `file:line`.

---

## Per-channel verdicts

### 1. Olfactory aggregation — INDISTINGUISHABLE *up to position layout*

`get_observation` calls `sense_resource` separately for resources, predators, obstacles, and **neutrals**, then **sums** all four into a single 5-dim olfactory vector (`sensor.py:290–295`). Each call uses additive inverse-square decay (`sensor.py:14`, `decay = 1/dist^2`) with the entity's own `*_property_sampled` weights.

When two entities share the same `[0,1,0,0,0]` vector, the per-cell olfactory signal is `Σᵢ wᵢ · 1/dᵢ²` for all entities of that type. **The class label is not preserved** — only the spatial intensity field. So instantaneous olfactory cannot tell rabbit from predator. *However*, the **shape** of that field over time differs (see §4). With `properties_std = 0`, there is no per-step olfactory noise that could leak identity.

### 2. Extero nociception — DOES NOT BROADCAST (contact-only)

`sense_extero_nociception` (`sensor.py:59–87`) checks `dist < 0.1` for `res_pos`, `pred_pos`, and `obs_pos`. It is a phasic contact sensor, not a distance gradient. **`neutral_pos` (rabbits) is not even in the list** — rabbits' `nociception_intensity: 0.1` never reaches the agent through this channel (`neutral_nociception` is parsed in `config_loader.py:204` but never read by the sensor). So extero-noc fires `0.9` only on the step the agent collides with a predator (or hiding_predator/rock).

This means extero-noc is a **post-hoc teaching signal at contact**, not a pre-contact discriminator. It cannot drive avoidance from a distance. (It does, of course, drive avoidance in the next episode via the recurrent state — see §4.)

### 3. Visual sensor at range 0 — STRONG DISCRIMINATOR, BUT ONLY ON COLOCATION

`get_visual_offsets(0)` returns `[[0,0]]` only (`sensor.py:112–114`). `sense_visual` one-hots dynamic entities into 8 channels: rabbit → channel **7** (Neutral), predator → channel **5** (Predator), hiding_predator → channel 4 (`sensor.py:139–195`). With matmul over exact-position matches (`sensor.py:202`), the visual vector is non-zero **only when an entity is on the agent's exact cell**. Adjacent cells produce no visual response.

So visual gives a perfect class label, but only AFTER the agent has stepped onto the entity. Combined with extero-noc (§2), this means the **first-encounter pair (visual ch.5 + noc 0.9) is unambiguous**. Subsequent steps, mediated by the RNN, exploit this.

### 4. Movement / temporal signature — STRONG IMPLICIT DISCRIMINATOR

Predator chase logic in `update_predators` (`core.py:132–251`):
- `pred_detect=5` (`detection_range`), `pred_hunt_thresh=0.7`, `pred_lose_interest_mult=1.5`, `attack_delay=3`.
- When `dist ≤ 5` and stamina `≥ 0.7·max`, predator transitions to HUNT (`core.py:156–160`) and steps toward `agent_pos` each tick (`core.py:201–202`).
- Rabbit (`update_neutral_animals`, `core.py:253+`) does pure random jitter, independent of `agent_pos`.

The olfactory gradient summed across both classes therefore has a **directional time-derivative** that correlates with the agent's recent positions only for the predator component. A GRU/recurrent policy can detect this without ever reading a per-step class label. Patrol-area also differs: predator covers `[[1,1],[10,10]]` (full grid), rabbits are confined to TL/BR quadrants — meaning the spatial distribution of "smell coming from the BR quadrant" is monomodal-rabbit while "smell tracking my position" is monomodal-predator.

### 5. Other channels

- **Collision** (`sensor.py:24–50`): boolean OOB / blocking-rock only; rabbit and predator are non-blocking and do not appear here.
- **Proprioception** / **interoceptive_nociception** / **satiation**: not class-conditional; no entity-identity leakage.
- **Locale / location sensor**: `location_sensor: false` in this config; not in obs.
- **`extero_nociception` damage broadcast**: none. Damage only applies on contact via `pred_nociception` array (`config_loader.py:107`), gated by `dist_pred < 0.1`.
- **Special predator-presence flags**: none surfaced to the policy.

---

## Ranked discriminators (most → least likely to drive sameProp avoidance)

1. **Movement / temporal signature in olfaction** (§4). Strongest *pre-contact* discriminator. Predator's HUNT mode mechanically tracks `agent_pos`; rabbit jitters. RPPO's GRU can pick this up across timesteps even with identical instantaneous olfactory components. *This is almost certainly the dominant channel driving avoidance.*
2. **Visual ch.5 vs ch.7 at colocation + Extero-noc 0.9 vs 0** (§§2–3). Provides a definitive teaching signal at the moment of contact. Drives avoidance learning episode-over-episode and within-episode via the RNN, but cannot drive same-step avoidance.
3. **Patrol-area asymmetry** (§4 secondary). Predator roams full grid; rabbits are quadrant-locked. The marginal distribution of "where olfactory peaks appear over time" carries class info without any per-step label.
4. **Olfactory instantaneous shape** (§1). Indistinguishable in expectation when properties match; only carries info via the temporal patterns in (1) and (3).
5. **Extero-noc as a distance broadcaster** — *does not exist* (§2). Often hypothesized; the code does not implement it.

## Implication for next experiments

If the goal is to ablate the agent's last remaining discriminators under sameProp, the design must target (1) and (3): randomize predator patrol to match rabbit confinement, or freeze predator out of HUNT mode (e.g. `hunt_stamina_threshold: 1.1`, `detection_range: 0`). Channel (2) cannot be ablated without disabling the visual sensor or changing the rabbit→ch7 / predator→ch5 mapping.

No code changes recommended in this memo.
