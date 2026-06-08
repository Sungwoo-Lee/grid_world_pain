# Observation-pipeline class-leak audit — sameProp / chasing-rabbit experiments

## Purpose (plain language)

The "chasing-rabbit" experiments are designed so that a dangerous **predator** and two
harmless **rabbits** are supposed to look, smell, and behave the *same* to the agent: all
three give off the identical smell `[0,1,0,0,0]` and all three chase the agent (`behaviour:
hunt`). The only thing that truly differs is whether contact hurts (predator deals 15–45
damage; rabbits deal 0). Yet a trained agent clearly tells them apart — it keeps the
predator about 4.3 cells away but lets a rabbit close to ~1.8 cells. That should be
*impossible* unless some observation channel secretly encodes "this one is a predator."

This review walks every one of the agent's **27 observation numbers**, channel by channel,
and asks: does predator vs rabbit produce a different value here, and is that *intended* or
a *bug*? The verdict drives whether the experiment is measuring what it thinks it is.

**Headline verdict:** The discrimination is **fully explained by two intended-by-code but
premise-breaking contact channels** — the *visual* sensor's per-class one-hot (predator =
channel 5, rabbit = channel 7) and the *exteroceptive-nociception* sensor (only the
predator's contact emits pain). **No additional unintended leak or implementation bug was
found.** Smell, collision, proprioception, satiation, and interoceptive nociception are all
class-blind, confirmed empirically. The eval path feeds the policy the exact same
observation as training. **Crucially**, because the experiment runs the visual sensor at
**range 0** (`visual_sensor_range: 0` — the agent only "sees" its own cell), the visual
class label is available **only on overlap, not at a distance** — so the visual channel
*cannot by itself* explain distance-keeping, and the contact-nociception channel is
contact-only too. The agent must therefore be **learning** the class from the *consequences*
of past contacts (a hit logs into injury → interoceptive pain history) and generalizing —
which is exactly why it still discriminates in the **no-predator transfer world** where no
class signal exists at all. That is a real, defensible scientific result, not an artifact.

The premise "predator and rabbit are indistinguishable" is **technically false at the
observation level on contact** (visual channel + contact pain both differ). If the intent is
a *truly* indistinguishable test, the minimal fix is to render predator and neutral to the
**same visual channel** (and optionally gate contact-nociception class-blind). Recommended
owners named at the end.

---

## Confirmed starting point

The user's reading of the code is **correct**:

- `config_loader.py:33` — `ANIMAL_CLASS_TO_VIS_CHANNEL = {"predator": 5, "neutral": 7}`.
- `config_loader.py:533` — `visual_channel_list.append(ANIMAL_CLASS_TO_VIS_CHANNEL[cls])`,
  building the per-entity `animal_visual_channel` array purely from the static `class` tag.
- `sensor.py:197-205` — `sense_visual` renders each animal via
  `jax.nn.one_hot(params.animal_visual_channel, 8)`. The channel is chosen **only** by
  class; `behaviour` (hunt/wander) does **not** change the channel. A neutral with
  `behaviour: hunt` still renders to channel 7. Verified empirically below.

Empirical confirmation (config `04-sameProp_R4_chasingRabbit.yaml`, loaded with the conda
interpreter):

```
classes        : ('predator', 'neutral', 'neutral')
behaviours     : ('hunt', 'hunt', 'hunt')          # all three hunt
visual_channel : [5, 7, 7]                          # predator=5, rabbits=7
is_damaging    : [True, False, False]               # only predator hurts
property mean  : predator=[0,1,0,0,0], rabbits=[0,1,0,0,0]   # identical smell
properties_std : [0,0,0,0,0]                        # deterministic; no per-episode jitter
```

---

## Channel-by-channel table (27 dims, config 04)

| Dim range | Sensor | Encodes | Class-discriminating? | Intended-design vs bug |
|---|---|---|---|---|
| 0 | Satiation | `state.satiation / max_satiation` | **No** | intended — class-blind |
| 1 | Interoceptive Nociception | alpha-kernel convolution of injury history (`sensor.py:97-113`) | **Indirect only** — nonzero *after* a predator hit; never references class | intended — *consequence* correlate, not a direct label |
| 2 | Extero Nociception | max contact pain across res/animal/rock (`sensor.py:59-95`) | **Yes, on contact only** — animal term masked by `params.animal_is_damaging`, so only the predator emits pain on overlap (`<0.1`); fires at distance for *no* animal | intended-by-code; **premise-breaking** — a contact-only class signal |
| 3–7 | Olfaction (5) | `res_chem + animal_chem + obs_chem` (`sensor.py:300-303`) | **No** — both classes map to channel 1, value `[0,1,0,0,0]`; isolated equidistant test gives byte-identical `[0,0.1111,0,0,0]` for predator and rabbit. Magnitude differences are purely *positional* (different cells), not class | intended — class-blind |
| 8–12 | Collision (5) | OOB + blocking rocks only (`sensor.py:24-50`) | **No** — no animal reference at all | intended — class-blind |
| 13–18 | Proprioception (6) | one-hot of `state.last_action` (`sensor.py:310`) | **No** | intended — class-blind |
| 19–26 | Visual (8) | single own-cell one-hot over 8 channels; `visual_sensor_range: 0` (`sensor.py:145-220`) | **Yes, on overlap only** — predator renders channel 5, rabbit channel 7; but range 0 means it only fires when the agent stands **on** the animal, never at a distance | intended-by-code; **premise-breaking** — a contact/overlap-only class label |

(Injury dim 0 and Nutrition are **not** present: `injury_observable: false`,
`nutrition_observable: false` in the config. Location is disabled. The 27 dims above are the
full vector; verified `get_observation_breakdown` total = 27.)

---

## Empirical leak demonstration

Same state, agent moved to four positions, `apply_noise=False`:

**Agent 3 cells from predator vs 3 cells from rabbit (DISTANCE):**
- Visual: `[1,0,0,0,0,0,0,0]` for **both** (only the agent's own grass cell; range 0 hides
  the animal's channel). **No class signal at distance.**
- Extero Nociception: `0.0` for **both**. **No class signal at distance.**
- Olfaction magnitudes differ, but only because the two animals sit at different positions
  with different neighbouring resources — not a class effect (isolated equidistant test
  proves olfaction is identical per class).

**Agent ON predator cell vs ON rabbit cell (OVERLAP):**
- Visual predator `[...,1,0,0]` (ch 5) vs rabbit `[...,0,0,1]` (ch 7) → **class leak on contact.**
- Extero Nociception predator `0.9` vs rabbit `0.0` → **class leak on contact.**

So both class-discriminating channels are **contact/overlap-only** in this config. The agent
cannot read "predator" off a distant animal from the observation directly; it must infer
class from the *history* of contact consequences (a predator overlap → injury → tonic
interoceptive pain over the next ~12 steps) and learn a policy that keeps the
predator-smelling-but-painful-on-contact entity farther away. This is learning, not a
free observational label — and it explains why discrimination **persists in the no-predator
transfer world** (`05-noPredator_chasingRabbit_eval.yaml`), where every animal is
`class: neutral` (visual channel 7, `is_damaging=False`) and there is literally no class
signal in the observation at all.

---

## Bug hunt (errors, not just design)

- **Animal ordering / indexing / one-hot off-by-one:** none found. `animal_visual_channel`
  is built in entity order and consumed positionally in the 3-way concat
  (`res, animal, obs`) in `sense_visual`; channels are correct (5/7) and verified.
- **Per-entity property bleed:** none. `animal_property_sampled` is `[0,1,0,0,0]` for all
  three; `_sample_property` with `std=0` is deterministic; no per-class scaling in the
  olfaction summation.
- **Eval-path parity (user's explicit worry):** **PASS.** The policy in `eval_rollout.py`
  receives `get_observation(state, params)` (`scripts/eval_rollout.py:642`) — the identical
  function and code path as training, with default `apply_noise=True`. The recorder-aware
  variant computes extra `obs0`/`true_obs0` for *recording only* (`:193-194`, `:220-221`);
  these are side-channel logs and do **not** alter the obs handed to `policy_fn`. The
  `--record` flag adds recording metadata and the recorder loop but does not touch obs
  construction. No alternate obs builder injects or drops information at eval.
- **Perceptual-noise modality↔breakdown sync:** **PASS.** Every breakdown key
  (`Satiation, Interoceptive Nociception, Extero Nociception, Olfaction, Collision,
  Proprioception, Visual`) is present in `noise_modality_order`; the missing-key set is
  empty, so `apply_perceptual_noise` would not `KeyError`. Noise is disabled
  (`perceptual_noise_enabled: False`) in these configs regardless.
- **vmap / PRNG (obs-relevant only):** `get_observation` folds `state.key` with a fixed salt
  (999) for noise; pure-functional; no obs-construction PRNG issue. (Noise off here anyway.)

---

## Conventions audit checklist

- Pytree / immutability: ✅ (no in-place mutation in obs path; `_replace` used in test)
- JIT recompilation triggers: ✅ (`visual_sensor_range`, `nociception_enabled` etc. are static; no traced-value branches added)
- vmap conventions: ✅ (obs path not implicated; renderer not called on batched state here)
- PRNG threading: ✅ (obs noise uses folded state key; deterministic)
- Sensor / observation-breakdown sync: ✅ (breakdown ↔ modality_order aligned, no KeyError)
- Config protocol: ✅ (config loads via `get_mandatory`; no fallback defaults observed in obs path)

---

## Verdict

The agent's predator/rabbit discrimination is **entirely explained by two intended-by-code,
premise-breaking channels** — the visual per-class one-hot (channel 5 vs 7) and the
exteroceptive-nociception class mask (`animal_is_damaging`) — **both of which fire only on
contact/overlap** under `visual_sensor_range: 0`. **No additional unintended leak or
implementation bug exists.** Smell, collision, proprioception, satiation, interoceptive
nociception, and the eval observation path are all class-blind / parity-correct, verified
empirically. The persistence of discrimination in the no-predator transfer world is
therefore a genuine learned-generalization result, not an observation artifact.

The literal claim "predator and rabbit are indistinguishable in the observation" is **false
on contact** (visual channel + contact pain differ). If the experiment intends a *strictly*
indistinguishable test, the minimal change is to **render predator and neutral to the same
visual channel** (e.g. add a config/schema flag that maps both classes to one shared animal
channel, collapsing `ANIMAL_CLASS_TO_VIS_CHANNEL`), and decide whether contact-nociception
should also be class-blind (it currently is the *only* channel that can teach "this one
hurts" — removing it would remove the learnable signal entirely, which may not be desired).

**Recommended owners (do not implement here — review only):**
- **experiment-designer** — owns the config/schema decision: whether to add an
  `animal_shared_visual_channel` (or obs-ablation) flag and how it interacts with the
  contact-pain signal; must define the intended "indistinguishable" semantics first.
- **developer** — owns the code change once the schema decision is made: collapse the
  visual-channel mapping behind the flag in `config_loader.py` / `sense_visual`, keeping the
  static-field / JIT and breakdown↔noise-sync conventions intact.

One-line conclusion: discrimination is fully accounted for by the visual class-channel and
contact-nociception (both contact-only); no hidden bug — the result is real, but the
"indistinguishable" premise needs the visual channel collapsed to be literally true.

Reviewed by: code-reviewer
