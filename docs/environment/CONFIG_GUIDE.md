# Config Guide — how to author and extend environment configs (v3.0)

> **Source**: `src/environment/config_loader.py` (the source of truth for keys) | **Deep reference**: [02_config_schema.md](02_config_schema.md) | **Back to hub**: [ENVIRONMENT_SUMMARY](ENVIRONMENT_SUMMARY.md)

---

## What this document is about

Every experiment in this project is just a **variation of one environment**. The environment is configured entirely through YAML files under `configs/environment/`. This guide is the collaborator-facing introduction to *how those config files work* and *how to write or change one* — it is a workflow guide, not an exhaustive key list (that lives in [02_config_schema.md](02_config_schema.md)).

The one thing to understand first: there is a single canonical "standard setup" file, `configs/environment/default.yaml`, called the **base config**. As of v3.0, a new experiment config no longer copies the whole world spec. Instead it writes one line — `extends: environment/default` — and then lists **only the handful of values that differ** from the base (a smaller grid, a different scene, a noise tweak). At load time the system **deep-merges** the base underneath those overrides, so the agent still receives a complete config. ("Deep-merge" = walk both files key-by-key; where both have a nested block, merge recursively; where a value is present in the experiment, it wins.) This is called a **sparse, layered** config.

A config that does **not** carry an `extends:` line loads **standalone** — exactly as before v3.0, byte-for-byte. This is how the ~91 older "full" configs (now archived) keep working untouched. So the rule is simple: **new work is sparse and layered; old work is standalone and frozen.**

The rest of this guide explains the merge model and its one footgun, where configs live, the five v3.0 features you will meet, how to author a new config and add a new config key, and the parity test that guards all of it. It ends with a **Maintenance Contract** that binds the config-caring agents to keep this guide and the schema doc in step with the code.

---

## 1. The model — base + sparse overrides

### Standalone vs. layered

| Config style | Has `extends:`? | How it loads | Who uses it |
|---|---|---|---|
| **Layered (sparse)** | yes | base deep-merged underneath; this file's keys win | all new experiment configs |
| **Standalone (full)** | no | loaded exactly as written, byte-for-byte (pre-v3.0 behaviour) | the base itself + the ~91 archived configs |

The single chokepoint that implements this is `load_env_config(path)` in `config_loader.py`. It resolves any `extends:` chain (a config may extend a config that itself extends another), strips the `extends:` key (it is metadata, not an env param), deep-merges, and hands the merged result to the existing `load_env_params(...)`. Mandatory-key validation runs **after** the merge — so a sparse config satisfies a required key *through the base*, and the no-fallback contract still holds.

**Missing config file = hard error.** As of the 2026-07-04 strict-load fix (H3, [[fix_plan_h1h2h3_resume_config]]), the low-level loader `Config.load_yaml` raises `FileNotFoundError` when the given path does not exist — it no longer prints a warning and returns an empty config. A typo'd `--config` path used to train silently on `configs/environment/default.yaml`; now it dies immediately. Intentional *optional* loads (train/eval/logger/visualization defaults, etc.) must guard with `os.path.exists(path)` at the call site — every existing optional call site already does.

### Deep-merge semantics and the list-replace footgun

Deep-merge treats **dicts** and **lists** differently:

- **Dicts merge recursively.** If the base has `body: { max_nutrition: 100, death_penalty: 5 }` and your config sets only `body: { death_penalty: 0 }`, the result keeps `max_nutrition: 100` and overrides `death_penalty`.
- **Lists replace wholesale.** If your config declares `resources:` (a list), it **completely replaces** the base's `resources:` list — elements are *not* merged in.

The list rule is the **footgun**. The base config carries scene lists — `entities:` (animals), `resources:`, `obstacles:`. To get a *clean* scene (e.g. a world with no animals), you must declare the list **explicitly empty**:

```yaml
environment:
  entities: []     # SUPPRESSES the base's rabbits + predator
  obstacles: []    # SUPPRESSES the base's rocks/bushes
```

If you **omit** `entities:` instead of setting it to `[]`, the base's animals survive the merge and silently appear in your world. **Omitting a list does not remove it — only an explicit empty list does.** This is the single most common authoring mistake; keep it in mind whenever you want fewer entities than the base.

---

## 2. Where configs live

```
configs/environment/
├── default.yaml                    ← the canonical BASE (full, standalone)
└── experiment/
    ├── basic/                      ← live curriculum: sparse `extends:` configs
    │   ├── 00-forage_5x5.yaml
    │   ├── 01-slowPred_5x5.yaml
    │   └── ...
    ├── <your-topic>/               ← new sparse configs go here, grouped by topic
    └── archive/                    ← the ~91 pre-v3.0 full configs (frozen)
```

- **New work** → `configs/environment/experiment/<topic>/` as a sparse `extends: environment/default` file.
- **`archive/`** holds the pre-v3.0 configs, relocated by `git mv` (history preserved). They are **frozen but still loadable** — they have no `extends:` key, so they load standalone exactly as they always did. Do not edit them to "modernise"; if you need a variant, author a fresh sparse config.

---

## 3. v3.0 feature quick-reference

Five capabilities landed in the v3.0 config overhaul. Each is opt-in and defaults to today's behaviour.

### 3.1 `extends:` layering (covered above)

```yaml
extends: environment/default       # or a list: [environment/default, environment/foo]
```
Absent → standalone load. Present → base(s) deep-merged underneath.

### 3.2 Configurable visual properties

Each entity (resource, animal, obstacle) may carry a `visual_properties` appearance vector — the vision analogue of the olfactory `properties` vector. Its length must equal `sensory.visual_vector_size` (default **8**). Omit it and the entity falls back to its historical one-hot channel.

Default channels at V=8: predator→5, neutral→7, food→3, hiding_predator→4, obstacle→6, background grass/sand/plain→0/1/2.

```yaml
- type: "food"
  visual_properties: [0,0,0,1,0,0,0,0]   # one-hot channel 3 = today's default
```

At `visual_vector_size ≠ 8` the class→channel defaults are undefined, so **every** entity must declare an explicit `visual_properties`, and `sensory.visual_background_properties` (a 3×V table, rows = grass/sand/plain) becomes required.

### 3.3 Per-episode visual sampling

An optional `visual_properties_std` (same length V) makes appearance jitter per episode via Gaussian sampling. Default is **zeros → deterministic** (sampled value equals the mean exactly, byte-identical to no sampling). The visual sampler uses an **independent PRNG stream**, so turning it on does not perturb olfactory sampling.

```yaml
  visual_properties:     [0,0,0,1,0,0,0,0]
  visual_properties_std: [0,0,0,0,0,0,0,0]   # zeros = deterministic
```

### 3.4 Initial-state randomization ranges

When you randomize the agent's starting nutrition or injury, you can now set the exact band. The four range keys are **conditional-mandatory** — required only when the matching flag is on:

```yaml
body:
  random_start_nutrition: true
  start_nutrition_low: 0
  start_nutrition_high: 100     # full range; the old code was locked to the upper half
  random_start_injury: true
  start_injury_low: 0
  start_injury_high: 100
```

With the flag `false` the range keys are not read (and need not be present). `default.yaml` carries them explicitly with the flags off (inert).

### 3.5 `eval_seeds` generator spec

The behaviour-measure block's `behavior_measures.eval_seeds` accepts either an explicit list **or** a compact generator spec that derives `eval_n_episodes` seeds deterministically:

```yaml
behavior_measures:
  eval_n_episodes: 64
  eval_seeds: { rng: 12345, sort: true }   # 64 seeds from RNG 12345, sorted
```

`rng` is mandatory inside the dict; `sort: true` (the default) keeps the seed order fixed so episode *i* is comparable across runs.

### 3.6 Per-episode entity count ranges (v3.0 PER\_EPISODE\_ENV\_VARIANCE)

Instead of a fixed `count: N`, each entity entry may declare a range. The engine draws an actual count K uniformly from `[count_low, count_high]` at each episode reset:

```yaml
environment:
  resources:
    - name: "food"
      type: "food"
      count_low: 2          # per-episode lower bound (inclusive)
      count_high: 6         # per-episode upper bound (inclusive); allocation size
      spawn_area: [[1, 1], [10, 10]]
      # ... other fields unchanged
```

**How it works.** The loader allocates `count_high` entity slots — the array shape is fixed (JAX requires it). At each episode reset, K is redrawn and the surplus `count_high − K` slots are **marked inactive**: their positions are parked off-grid so they cannot collide, cause damage, or be sensed.

**Backward compatibility.** `count: N` (no range keys) behaves exactly as before — it is treated as `count_low = count_high = N`, so all K slots are always active (all-True mask). The K-draw is **skipped entirely** when every entry in a class uses a degenerate range, so existing configs produce byte-identical output.

**Constraint.** Do not specify both `count` and `count_low`/`count_high` on the same entry — the loader raises `ValueError`. Use one style or the other.

**Applies to all three entity classes:**
- `resources:` — food, hiding\_predator
- `entities:` — predator, rabbit (any `class`/`behaviour` animal)
- `obstacles:` — rock, bush, tree

### 3.7 Predator jump / pounce (`attack_range` + `attack_success_rate`)

A hunting animal may occasionally lunge several cells in one step instead of its normal 1-cell chase move, either landing on the agent (hit) or beside it (miss) — see [PREDATOR_JUMP_MECHANISM.md](../develop/active/env_entities/PREDATOR_JUMP_MECHANISM.md) for the full design. Both keys are **optional**; omitting either leaves the jump disabled (byte-identical to today):

```yaml
    attack_range: [3, 5]          # or a scalar, e.g. 4. Absent -> [0,0] = disabled.
    attack_success_rate: 0.6      # float in [0,1]. Absent -> 0.0.
```

A bushed (`hides_agent`) agent can never be jumped onto — the jump reuses the existing `agent_hidden` gate.

---

## 4. How to author a new config (worked example)

Goal: a 5×5 foraging world with food only — no animals, no obstacles — inheriting everything else (body, sensors, noise) from the base.

```yaml
# configs/environment/experiment/basic/00-forage_5x5.yaml
extends: environment/default

environment:
  height: 5
  width: 5
  start_pos: [3, 3]

  resources:                       # REPLACES the base resource list wholesale
    - name: "food"
      type: "food"
      count: 2
      spawn_area: [[1, 1], [5, 5]]
      properties: [1.0, 0.0, 0.0, 0.0, 0.0]
      properties_std: [0.0, 0.0, 0.0, 0.0, 0.0]
      visual_properties: [0,0,0,1,0,0,0,0]
      visual_properties_std: [0,0,0,0,0,0,0,0]
      max_consumption: 12
      regeneration_delay: 0
      damage: [0.0, 0.0]
      nociception_intensity: 0.0

  entities: []                     # explicit empty → SUPPRESS base rabbits + predator
  location_areas:
    - { type: "grass", area: [[1, 1], [5, 5]] }
```

Everything not mentioned — the whole `body:`, `sensory:`, `perceptual_noise:`, `behavior_measures:` blocks — comes from `default.yaml` unchanged. This is the real `00-forage_5x5.yaml`; read the live `configs/environment/experiment/basic/` files for more patterns.

---

## 5. How to add a new config key (the no-fallback workflow)

The project rule is **no fallback defaults**: critical keys are read with `config.get_mandatory('key')` and a missing key raises `ValueError`. Adding a new key is therefore a multi-step change that must land **atomically**:

1. **Add the key to `configs/environment/default.yaml`** with an explicit value and an inline comment explaining it. The base must always carry every key it reads, so layered configs inherit a valid default-of-record.
2. **Read it in `config_loader.py`** via `config.get_mandatory(...)` (or, for a conditional key, gate the `get_mandatory` behind its enabling flag — see the initial-state range keys for the pattern). If it is shape-determining, store it on `EnvParams` as a static field (`struct.field(pytree_node=False)`); otherwise as a traced leaf.
3. **Document it here** (in the quick-reference if it is a feature surface) **and in [02_config_schema.md](02_config_schema.md)** (the deep key list). Both move in the same change.
4. **Add or extend a test** that proves the key is read and that a missing/invalid value raises. For a regression-class change, the test must fail before the code change and pass after.
5. **For sensory / noise keys, keep observation↔noise width in sync.** Observation width is computed in one place, `get_observation_breakdown`; the per-modality noise block auto-resizes from it. A new sensor or a width change must keep the noise modality list aligned — route through `env-config-auditor`.

If the key is experiment-facing, the schema/loader work is `senior-developer` + `developer`'s job first; only then does `experiment-designer` author configs that use it.

---

## 6. The parity gate

Two regression tests are the safety net for the entire config system:

- **`tests/env/test_unified_parity.py`** — for each config, instantiates `EnvParams` and checks a reset+rollout against a committed fixture. It protects the byte-for-byte behaviour of every config, including all archived ones.
- **`tests/env/test_visual_parity.py`** — the same discipline for the visual observation slice, across the base plus the basic curriculum.

These gates must stay **green** on every config-system or schema change. A red parity test means a change perturbed a config that was supposed to be byte-identical — **stop and diff**, do not regenerate fixtures to make it pass (regenerating hides the very regression the gate exists to catch). Fixtures are regenerated only deliberately, on a pre-change commit, when the change *intends* to alter observations.

---

## 7. Training-config layering (`configs/train/`)

This guide is mostly about `configs/environment/` (the world spec), but the same base-plus-override pattern also governs the **training-run** config — checkpoint cadence, logging cadence, and similar knobs that are not part of the environment itself. Two trainers read this layer: **rPPO** (`train.py`) and **Dreamer** (`src/algorithms/dreamer_srl/dreamer_srl_main.py`).

**The rule: `configs/train/default.yaml` is the reference/documentation file** — it declares every training/logging knob with a generic default value, and doubles as the fallback for algorithms that load neither per-algo layer (DQN, DRQN, PPO). Anything rPPO or Dreamer wants different from that generic default lives in that trainer's own override file, merged **above** `default.yaml` (see the "Self-contained per-algo files" note below — the two layers now intentionally duplicate keys rather than one being authoritative-only):

- `configs/train/recurrent_ppo.yaml` — rPPO's per-algo layer. `train.py` merges it **only** when the agent config declares `agent.algorithm == "RecurrentPPO"` — a real gate, because `train.py` is shared across algorithms and must not apply rPPO's values to a non-rPPO run.
- `configs/train/dreamer_srl.yaml` — Dreamer's per-algo layer. `dreamer_srl_main.py` merges it **unconditionally**, at both its merge sites (single-config and the `--configs-dir` curriculum path). There is no algorithm gate here, and this is a deliberate asymmetry with rPPO, not an oversight: `dreamer_srl_main.py` is a Dreamer-only entry point, so the gate would have no job to do — and it would actively break two real agent configs (`configs/models/dreamer_srl/agent_xs.yaml`, `configs/models/dreamer_srl/01_food_only_smoke.yaml`) that do not declare `agent.algorithm` at all.

Full merge order (each stage's keys win over the ones before it):

```
get_default_config() (built-in seed)
  → configs/train/default.yaml            (algorithm-neutral)
  → configs/train/<algo>.yaml             (recurrent_ppo.yaml or dreamer_srl.yaml)
  → configs/evaluation/default.yaml
  → configs/visualization/default.yaml
  → env --env-config  (or, for Dreamer curriculum, the per-stage env YAML)
  → --agent-config
  → CLI flags (e.g. --log-interval, --checkpoint-frequency)
```

**Self-contained per-algo files (v3.0 convention — reverses an earlier structural-enforcement design).** Each of `recurrent_ppo.yaml` and `dreamer_srl.yaml` declares its **own complete set of training/logging values** — `training.num_envs`, `training.checkpoint_frequency`, `training.max_checkpoints_to_keep`, and all four `logging.*` knobs (`episode.smoothing_episodes`, `episode.interval_episodes`, `step.smoothing_iters`, `step.interval_iters`) — so either file can be read top-to-bottom without cross-referencing `default.yaml` or the other algo's file. `configs/train/default.yaml` **also** declares every one of these keys, with generic default values: it is the reference/documentation layer (read it to see every knob that exists) **and** the fallback for algorithms that load neither per-algo file (DQN, DRQN, PPO — see the merge order above). The per-algo file wins for its own algorithm because it is merged **above** `default.yaml`.

An earlier revision of this split enforced `logging.episode.smoothing_episodes` (how many episodes are averaged into each dashboard point) **structurally identical** across algorithms by declaring it *only* in `default.yaml` and omitting it from both per-algo files, so they inherited the same value by construction. **That structural enforcement has been intentionally reversed** — the user may legitimately want different smoothing per algorithm, so it must not be locked. `smoothing_episodes` is now declared explicitly in `default.yaml`, `recurrent_ppo.yaml`, and `dreamer_srl.yaml` (currently all `5000`, by convention, not by code). **Keeping the value equal across the two per-algo files is a documented convention, not a structural constraint** — the comparability guarantee (a Dreamer curve and an rPPO curve being equally noisy) now depends on a human keeping the two copies in sync, not on the merge order making drift impossible. See [[DREAMER_TRAIN_CONFIG_SPLIT]] for the full design history, including the superseded rationale.

---

## 8. Pointers

- **[02_config_schema.md](02_config_schema.md)** — the deep, key-by-key reference (YAML → `EnvParams`, mandatory keys, expansion rules).
- **`src/environment/config_loader.py`** — the source of truth. When the doc and the code disagree, the code wins and the doc is wrong; fix the doc.
- **[ENVIRONMENT_SUMMARY.md](ENVIRONMENT_SUMMARY.md)** — the environment hub (observation table, latent-bug FAQ, reading order).
- Design rationale (not required reading): [`docs/develop/active/refactors/CONFIG_LAYERING_AND_EXPERIMENT_REORG.md`](../develop/active/refactors/CONFIG_LAYERING_AND_EXPERIMENT_REORG.md), [`docs/develop/active/refactors/CONFIGURABLE_INITIAL_STATE_RANGES.md`](../develop/active/refactors/CONFIGURABLE_INITIAL_STATE_RANGES.md), [`docs/develop/active/sensors/CONFIGURABLE_VISUAL_PROPERTIES_PLAN.md`](../develop/active/sensors/CONFIGURABLE_VISUAL_PROPERTIES_PLAN.md).

---

## Maintenance Contract

**Any change to the config schema or the config system MUST, in the same change:**

1. **Update `configs/environment/default.yaml`** — add/rename the key with its explicit value and update its inline comment.
2. **Update this guide AND [02_config_schema.md](02_config_schema.md)** — keep the workflow guide and the deep key reference in step with the code.
3. **Add or extend a test** that exercises the new/changed behaviour (and proves a missing/invalid value raises, for mandatory keys).
4. **Keep the parity gate green** (`tests/env/test_unified_parity.py`, `tests/env/test_visual_parity.py`) — or, if the change deliberately alters observations, regenerate fixtures on a pre-change commit and say so explicitly.

**The agents below are bound to READ this guide before any config work and to UPDATE it (and `02_config_schema.md`) in the same change whenever the schema or system changes:**

- `env-config-auditor` and `experiment-designer` — primary config owners.
- `developer`, `senior-developer`, `code-reviewer` — secondary, whenever their work touches `config_loader.py`, `state.py` (`EnvParams`), or `configs/`.
