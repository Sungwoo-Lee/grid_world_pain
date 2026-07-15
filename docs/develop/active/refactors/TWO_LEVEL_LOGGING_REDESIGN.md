---
title: "Two-level logging redesign: separate smoothing from interval, in explicit units"
topic: refactors
status: active
created: 2026-07-15
last_updated: 2026-07-15
---

# Two-level logging redesign (episode-level + step-level)

> **Status**: PLANNED
> **Opened**: 2026-07-15
> **Applies to**: `train.py` (recurrent PPO) and `src/algorithms/dreamer_srl/dreamer_srl_main.py` (Dreamer)
> **Related**: [[FIX_PPO_MODULATION_LOGGING]], [[PER_EPISODE_ENV_VARIANCE]]

---

## Context

Both of our trainers decide how often to write a row to the training dashboard using a single
setting called `log_interval`, counted in **training iterations**. The problem is that an
"iteration" is not the same amount of experience in the two trainers. In recurrent PPO one
iteration collects a whole rollout — 128 steps across 16 parallel worlds, so **2,048 steps of
the agent actually living in the environment**. In Dreamer one iteration is a single step across
16 parallel worlds — **16 environment steps**, which is **128 times smaller**. The same number
in the same-named setting therefore means two wildly different things, and nobody reading the
config can see that.

That single ambiguity has now produced three separate failures:

1. **A phantom "Dreamer is unstable" conclusion.** Dreamer's learning curve looked roughly 640×
   noisier than PPO's and was read as an algorithmic instability. It was not — each Dreamer point
   was averaging far fewer episodes than each PPO point. The difference was entirely an artifact
   of how the curve was recorded.
2. **An operator error that silently degraded a run.** Someone passed `--log-interval 100` on the
   command line, which silently overrode the value of 2000 that had been carefully tuned in the
   config file (command-line arguments win over config files in our read-priority order). The
   resulting curves were about 4× noisier than the design intended, with no warning anywhere.
3. **A config that reads backwards.** PPO is set to 50 and Dreamer to 2000, which looks like
   Dreamer logs far *less* often. Measured in environment steps, Dreamer actually logs about
   **3.2× more often** than PPO. The config actively misleads its reader.

The root cause is not any one of those numbers. It is that **one knob is controlling two
independent concerns**: how many episodes get averaged into a single point (which sets how
**noisy** the curve is) and how often a row gets written at all (which sets how much **data
volume** the run produces). Those are different questions with different right answers, and they
need different knobs. Additionally, every knob must carry its unit in its own name, so that a
reader can never again mistake iterations for episodes.

This plan splits the one knob into four explicitly-named, unit-bearing knobs, applies the same
scheme to both trainers, and adds a spread (standard deviation, min, max) alongside every mean so
that a curve can no longer hide a bimodal mixture behind a healthy-looking average.

**Safety note, read this first:** a parallel session is running PPO training via `train.py` right
now (the "basic04" size sweep on nodes 107/108/110) and may relaunch at any moment. Every new key
in this plan is **optional**, and when the new keys are absent the trainers must behave exactly as
they do today. `log_interval` is **deprecated, not removed**.

---

## Analysis

### The unit mismatch, measured

| | recurrent PPO (`train.py`) | Dreamer (`dreamer_srl_main.py`) |
|---|---|---|
| what one "iteration" is | one rollout | one env-step batch |
| env-steps per iteration | `num_steps(128) × num_envs(16)` = **2,048** | `1 × num_envs(16)` = **16** |
| ratio | — | **128× smaller** |
| today's configured `log_interval` | 50 (via CLI) / 500 ([`configs/train/recurrent_ppo.yaml:17`](../../../../configs/train/recurrent_ppo.yaml)) | 2000 ([`configs/models/dreamer_srl/01_food_only_buf256k.yaml:140`](../../../../configs/models/dreamer_srl/01_food_only_buf256k.yaml)) |
| env-steps per logged row | 50 × 2,048 = **102,400** | 2000 × 16 = **32,000** |

Dreamer logs a row every 32k env-steps; PPO every 102k. Dreamer logs **3.2× more often**, while
its config number (2000) is 40× *larger* than PPO's (50). The config reads backwards.

### One knob, two concerns

`log_interval` currently sets both:

- **NOISE** — the window of episodes averaged into one point. Today the window *is* the interval
  (`iteration_episodes` is cleared after each emission), so the two are welded together.
- **VOLUME** — how many rows a session writes.

You cannot ask for "smooth curves but few rows", or "many rows but each still comparable to the
other algorithm's rows". Both trainers are stuck on the diagonal.

Worse, **NOISE is the axis that must be shared across algorithms** (that is the only thing that
makes a Dreamer curve visually comparable to a PPO curve), while **VOLUME is the axis that must
differ per algorithm** (their iteration rates differ by 128×). Welding them together makes the
two requirements mutually unsatisfiable.

### Real measured numbers (verified 2026-07-15)

Measured from the live `dsrl_basic04_size_xs_rr0p0625` Dreamer run (started 2026-07-14 15:14:45):

| observation | value |
|---|---|
| checkpoint at episode 1,000 | **+5m 20s** into the run |
| checkpoint at episode 10,000 | **+30m 06s** into the run |
| ⇒ first 5,000 episodes | **≈ 16 minutes** |
| run position at ~28.5 h | episode **155,742** |
| average episode rate | **~131k episodes / 24 h** |
| throughput | **147 SPS** (steps per second) |

The episode rate is **not constant** — it falls as the agent gets better and episodes get longer:
~23k episodes/hour early (mean survival ~25 steps) → ~3.6k episodes/hour now (mean survival ~147
steps). This is the justification for `smoothing_episodes = 5000`:

- **5,000 episodes ≈ 3–4% of a full run** — a window wide enough to smooth, narrow enough to still
  track real learning progress.
- **The first point appears ~16 minutes in** — acceptable startup latency for a multi-day run.
- At the *late*, slow rate (~3.6k eps/h) a 5,000-episode window spans ~1.4 h of wall-clock, which
  is still a small fraction of a ~30 h run.

> **⚠️ Stale memory to correct.** The memory insight
> `20260529_1825_log_interval_anchored_rows_per_session` quotes "~38 SPS, 20–40k eps/24h". That
> was measured at `replay_ratio=1`. Our current runs use `replay_ratio=0.0625`, which is ~4×
> faster (147 SPS, ~131k eps/24h). **Follow-up:** ask `/memorize` (or `bug-curator`-style
> curation of the memory layer) to append a correction to that insight. Any reasoning that
> derives a log cadence from the old figures will be wrong by ~4×.

### Why spread, not just mean

A mean survival of 133 steps looks perfectly healthy. It can also be the average of a **bimodal
mixture**: ~400 steps on episodes where no predator spawned near the agent, and ~20 steps on
episodes where one did. The mean hides that completely; the standard deviation exposes it
instantly. Every mean we emit from a window should be accompanied by `std`, `min`, and `max`
computed from that same window.

---

## Implementation Plan

### Design

Four explicit, unit-bearing knobs, in a new `logging:` config block:

```yaml
logging:
  episode:
    smoothing_episodes: 5000   # rolling window: mean over the last N EPISODES
                               # UNIVERSAL — identical in every algo → equal noise → curves comparable
    interval_episodes:  200    # write one row per N EPISODES — PER-ALGO → rows/session
  step:
    smoothing_iters:    50     # rolling window: mean over the last N ITERATIONS
                               # per-algo (losses are never cross-compared between algos)
    interval_iters:     100    # write one row per N ITERATIONS — PER-ALGO → rows/session
```

Rules:

- **Episode metrics** (survival / `Episode/Steps`, reward, behavior measures) use the **EPISODE**
  knobs. **Step metrics** (losses, steps-per-second) use the **ITERATION** knobs. Each level's
  smoothing is a count of samples **at its own level** — episodes for episode metrics, iterations
  for step metrics.
- `smoothing_episodes` **MUST be universal** — the same value in both trainers. That is precisely
  what makes a Dreamer curve and a PPO curve comparable: equal window ⇒ equal noise. `interval_*`
  is per-algorithm, because it controls row volume and the two trainers' iteration rates differ
  by 128×.
- **smoothing > interval** for both levels → overlapping (rolling) windows.
- **Wait for a full window before the first emission.** The first row is emitted at the first
  multiple of `interval_episodes` that is ≥ `smoothing_episodes`. This prevents the first few
  points being computed from 2 episodes and looking like a wild transient.
- **Emit spread from the same window** alongside the mean: `std`, `min`, `max`.

### Buffer structure

- **Buffer A** — episodes that finished on **this** step. Maximum size = `num_envs`. The order
  within Buffer A is irrelevant: episodes that finish simultaneously in different parallel worlds
  are exchangeable samples.
- **Buffer B** — a rolling `deque(maxlen=smoothing_episodes)` over the episode stream. Each
  finished episode from Buffer A is pushed into Buffer B; after each push, increment an episode
  counter; when `ep_count % interval_episodes == 0` **AND** the window is full
  (`len(B) == smoothing_episodes`) → emit `mean` / `std` / `min` / `max` of Buffer B.
- **Step level** — the same mechanic with a `deque(maxlen=smoothing_iters)` over per-iteration
  loss scalars, emitting every `interval_iters` once the window is full.

Note that Buffer B is **never cleared** — it evicts. This is the key behavioral change from today,
where `iteration_episodes` is emptied after each emission (that clearing is what welds the window
to the interval).

### Worked example — episode level

`smoothing_episodes=4`, `interval_episodes=2`, `num_envs=4`:

| when | finishes | Buffer A | push → Buffer B (maxlen 4) | ep# | log? |
|---|---|---|---|---|---|
| iter 10 | env1, env3 | [20, 35] | push 20 → `[20]` | 1 | no |
| | | | push 35 → `[20,35]` | 2 | **LOG mean=27.5** |
| iter 14 | env0 | [50] | push 50 → `[20,35,50]` | 3 | no |
| iter 15 | env2 | [12] | push 12 → `[20,35,50,12]` full | 4 | **LOG mean=29.25** |
| iter 22 | env1 | [60] | evict 20 → `[35,50,12,60]` | 5 | no |
| iter 23 | env3, env0 | [18,44] | evict 35 → `[50,12,60,18]` | 6 | **LOG mean=35.0** |
| | | | evict 50 → `[12,60,18,44]` | 7 | no |

→ 3 rows for 7 episodes (one per 2 ✓); each = mean of the last ≤4 episodes ✓. Windows at ep4 and
ep6 share episodes 50 and 12 → genuinely overlapping (rolling).

### Worked example — step level

`smoothing_iters=3`, `interval_iters=2`:

| iter | loss | Buffer S (maxlen 3) | log? |
|---|---|---|---|
| 1 | 8.0 | `[8.0]` | no |
| 2 | 6.0 | `[8.0,6.0]` | **LOG mean=7.0** |
| 3 | 5.0 | `[8.0,6.0,5.0]` full | no |
| 4 | 4.0 | evict 8.0 → `[6.0,5.0,4.0]` | **LOG mean=5.0** |
| 5 | 4.5 | evict 6.0 → `[5.0,4.0,4.5]` | no |
| 6 | 3.5 | evict 5.0 → `[4.0,4.5,3.5]` | **LOG mean=4.0** |

> **Note on the two worked examples vs. the "wait for a full window" rule.** Both tables show
> emissions *before* the window is full (ep2 at mean=27.5; iter2 at mean=7.0) because they are
> illustrating the buffer mechanic, not the warm-up gate. In the shipped code the warm-up gate
> applies: the first episode row lands at the first multiple of `interval_episodes` that is
> ≥ `smoothing_episodes` (in the episode example: ep4), and the first step row at the first
> multiple of `interval_iters` that is ≥ `smoothing_iters` (in the step example: iter4). The
> eviction/overlap behavior shown from ep4 / iter4 onward is exactly what ships.

### Rule table

| relationship | behavior |
|---|---|
| smoothing > interval | overlapping → true rolling smoothing ← **what we want** |
| smoothing == interval | disjoint blocks → today's behavior |
| smoothing < interval | gaps → some episodes never logged |

### ⚠️ Open question for the user — the step-level defaults

The design block above illustrates the step level with `smoothing_iters: 50`, `interval_iters: 100`
— but that is `smoothing < interval`, which the rule table classifies as **gaps: some iterations
never logged**. It contradicts the stated rule "**smoothing > interval** for both".

Two readings, both defensible:

- **(a)** The illustrative 50/100 is a typo for 100/50 and the rule holds at both levels.
- **(b)** Gaps at the *step* level are acceptable on purpose — a loss curve is a diagnostic, not a
  cross-algorithm comparison, and subsampling it is harmless.

**This plan assumes (a)** and ships defaults that satisfy `smoothing > interval` at both levels
(see the config table below), plus a **startup WARNING** whenever a config sets
`smoothing_* < interval_*` at either level, so reading (b) remains reachable by explicit choice
without being reachable by accident. **If the user prefers (b), only the default numbers change —
no code changes.**

### Config keys

New keys, all **optional**. The `logging:` block is read from the merged config exactly like
`training.*` is today.

| YAML path | Dreamer value | rPPO value | unit | rationale |
|---|---:|---:|---|---|
| `logging.episode.smoothing_episodes` | **5000** | **5000** | episodes | **UNIVERSAL — must match.** ~3–4% of a run; first point ~16 min in (measured above). |
| `logging.episode.interval_episodes` | **200** | **4000** | episodes | Per-algo row volume. Both < 5000 → overlap ✓. Dreamer at ~131k eps/24h → ~655 rows/24h. rPPO's much lower episode throughput → 4000 keeps rows/session in the same ballpark. |
| `logging.step.smoothing_iters` | **200** | **100** | iterations | > interval ✓. Dreamer: 200 iters = 3,200 env-steps of loss averaging. rPPO: 100 iters = 204k env-steps. |
| `logging.step.interval_iters` | **100** | **50** | iterations | Dreamer: preserves the "100" from the design sketch. rPPO: **50 exactly preserves today's step-row cadence**, so the parallel session's dashboards do not change density. |

Files:

- **`configs/train/default.yaml`** — add the `logging:` block with the **Dreamer** values (this
  file is the shared base; Dreamer keeps `default.yaml`'s values, per its own header).
- **`configs/train/recurrent_ppo.yaml`** — override `logging.episode.interval_episodes: 4000`,
  `logging.step.smoothing_iters: 100`, `logging.step.interval_iters: 50`. Do **not** re-declare
  `smoothing_episodes` here — inheriting it from `default.yaml` is what mechanically enforces
  "universal". Add a comment saying exactly that.

### Backward compatibility & deprecation

**Hard requirement: a config with no `logging:` block must produce byte-identical logging
behavior to today.** The parallel basic04 sweep relaunches against unmodified configs.

Resolution logic, per trainer:

```
if config has any `logging.*` key:
    → NEW two-level path (rolling deques, warm-up gate, spread metrics)
    → if --log-interval was ALSO passed on the CLI: print a loud WARNING that it is
      ignored under the new path, and name the four keys that replaced it.
else:
    → LEGACY path, unchanged: `log_interval` resolution exactly as today
      (train.py:492, dreamer_srl_main.py:1169-1172), clear-after-emit semantics.
    → print a one-line DEPRECATION notice naming this doc.
```

- **Do NOT remove** `training.log_interval` / `training.log_accumulate` from
  `configs/train/default.yaml` or `configs/train/recurrent_ppo.yaml`. They stay as the legacy
  fallback and are what the deprecation notice points away from.
- **Do NOT remove** the `--log-interval` / `--log-accumulate` CLI flags. Under the new path
  `--log-interval` is ignored-with-a-warning — this is the direct fix for **failure #2** (a CLI
  flag silently beating a tuned config).
- **Do NOT add** CLI flags for the four new knobs. They are config-owned by design; adding CLI
  overrides would re-open exactly the failure mode we are closing. (`smoothing_episodes` in
  particular must not be per-launch overridable — its whole value is being identical everywhere.)
- **The `DQN` / `DRQN` / `PPO` branches of `train.py` stay on the legacy path entirely.** They are
  not part of the active research program and touching them is unbudgeted risk. Their
  `iteration % log_interval` sites are explicitly **out of scope**.
- **Config dump:** both trainers persist resolved values into the dumped `config.yaml` (e.g.
  `train.py:495`). The resolved `logging.*` values must be `config.set(...)` the same way, so a
  post-hoc reader can tell which path a run took.

Migration for the user (document in the doc, do not automate): add the `logging:` block to a
config → that config flips to the new path on its next launch. Nothing else to change.

---

### File Changes

#### NEW `src/utils/rolling_logging.py`

New shared module holding the buffer mechanic used by both trainers. (New file under `src/`, not
`scripts/` — [SCRIPTS_DEPENDENCY_MAP.md](../../../environment/SCRIPTS_DEPENDENCY_MAP.md) does
**not** need an update.) Sits alongside the existing shared
[`src/utils/episode_logging.py`](../../../../src/utils/episode_logging.py) (110 lines), which it
complements: `episode_logging.py` owns the *fan-out of metric keys*, this module owns *when and
over what window a row is emitted*.

```python
"""Two-level rolling logging: separate smoothing (noise) from interval (volume).

See docs/develop/active/refactors/TWO_LEVEL_LOGGING_REDESIGN.md.
"""
from collections import deque
from typing import Any, Dict, List, Optional
import numpy as np


class RollingWindow:
    """Rolling window over a sample stream with an independent emission interval.

    smoothing: deque maxlen — how many samples are averaged (NOISE).
    interval:  emit one row per `interval` samples (VOLUME).
    Emission requires a FULL window, so the first emission lands at the first
    multiple of `interval` that is >= `smoothing`.
    """

    def __init__(self, smoothing: int, interval: int, name: str = ""):
        if smoothing < 1 or interval < 1:
            raise ValueError(f"{name}: smoothing/interval must be >= 1 "
                             f"(got {smoothing}/{interval})")
        self.smoothing = smoothing
        self.interval = interval
        self.name = name
        self.buf: deque = deque(maxlen=smoothing)
        self.count = 0

    def push(self, sample: Any) -> bool:
        """Append one sample. Returns True iff a row should be emitted now."""
        self.buf.append(sample)
        self.count += 1
        return (self.count % self.interval == 0) and (len(self.buf) == self.smoothing)

    def full(self) -> bool:
        return len(self.buf) == self.smoothing


def spread(values: List[float], key: str, out: Dict[str, float]) -> None:
    """Write mean/std/min/max of `values` into `out` under `key`{,_Std,_Min,_Max}."""
    if not values:
        return
    a = np.asarray(values, dtype=np.float64)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return
    out[key]            = float(a.mean())
    out[f"{key}_Std"]   = float(a.std())
    out[f"{key}_Min"]   = float(a.min())
    out[f"{key}_Max"]   = float(a.max())


def resolve_logging_cfg(cfg_get, defaults: Dict[str, int]) -> Optional[Dict[str, int]]:
    """Return the four resolved knobs, or None if no `logging.*` key is present
    (→ caller must take the LEGACY log_interval path).

    `cfg_get` is a callable(key, default) -> value (e.g. config.get).
    Emits a WARNING when smoothing <= interval at either level.
    """
    keys = {
        'smoothing_episodes': 'logging.episode.smoothing_episodes',
        'interval_episodes':  'logging.episode.interval_episodes',
        'smoothing_iters':    'logging.step.smoothing_iters',
        'interval_iters':     'logging.step.interval_iters',
    }
    raw = {k: cfg_get(path, None) for k, path in keys.items()}
    if all(v is None for v in raw.values()):
        return None
    out = {k: (raw[k] if raw[k] is not None else defaults[k]) for k in keys}
    for lvl, s, i in (("episode", 'smoothing_episodes', 'interval_episodes'),
                      ("step",    'smoothing_iters',    'interval_iters')):
        if out[s] <= out[i]:
            print(f"[WARN] logging.{lvl}: smoothing ({out[s]}) <= interval ({out[i]}) — "
                  f"windows will NOT overlap; some samples are never logged. "
                  f"See docs/develop/active/refactors/TWO_LEVEL_LOGGING_REDESIGN.md",
                  flush=True)
    return out
```

---

#### `train.py` — knob resolution (lines 492–496)

```python
# BEFORE (492-496):
    log_interval = args.log_interval or config.get('training.log_interval', 1)
    log_accumulate = args.log_accumulate if args.log_accumulate is not None else config.get('training.log_accumulate', True)
    config.set('training.num_envs', num_envs)
    config.set('training.log_interval', log_interval)
    config.set('training.log_accumulate', log_accumulate)

# AFTER:
    from src.utils.rolling_logging import resolve_logging_cfg
    # Two-level logging (docs/develop/active/refactors/TWO_LEVEL_LOGGING_REDESIGN.md).
    # `logging_cfg is None` -> LEGACY log_interval path (unchanged behavior for configs
    # that predate this block; the basic04 sweep depends on this).
    logging_cfg = resolve_logging_cfg(
        config.get,
        defaults={'smoothing_episodes': 5000, 'interval_episodes': 4000,
                  'smoothing_iters': 100, 'interval_iters': 50},
    )
    if logging_cfg is not None and args.log_interval is not None:
        print("[WARN] --log-interval is IGNORED: this config uses the two-level `logging:` "
              "block. Set logging.episode.interval_episodes / logging.step.interval_iters "
              "in the config instead.", flush=True)
    log_interval = args.log_interval or config.get('training.log_interval', 1)
    log_accumulate = args.log_accumulate if args.log_accumulate is not None else config.get('training.log_accumulate', True)
    if logging_cfg is None:
        print("[DEPRECATION] training.log_interval is deprecated — it conflates smoothing "
              "with interval and its unit (iterations) differs 128x between rPPO and Dreamer. "
              "Migrate to the `logging:` block: "
              "docs/develop/active/refactors/TWO_LEVEL_LOGGING_REDESIGN.md", flush=True)
    config.set('training.num_envs', num_envs)
    config.set('training.log_interval', log_interval)
    config.set('training.log_accumulate', log_accumulate)
    if logging_cfg is not None:
        # Persist resolved values so the dumped models/config.yaml records what actually ran.
        config.set('logging.episode.smoothing_episodes', logging_cfg['smoothing_episodes'])
        config.set('logging.episode.interval_episodes',  logging_cfg['interval_episodes'])
        config.set('logging.step.smoothing_iters',       logging_cfg['smoothing_iters'])
        config.set('logging.step.interval_iters',        logging_cfg['interval_iters'])
```

#### `train.py` — buffer allocation (line 1054)

```python
# BEFORE (1053-1054):
    # Buffer for episodes that finish across iterations (Stage 3)
    iteration_episodes = []

# AFTER:
    # Buffer for episodes that finish across iterations (Stage 3) — LEGACY path only.
    iteration_episodes = []
    # Two-level logging: Buffer B (episode stream) + Buffer S (iteration stream).
    ep_window = step_window = None
    if logging_cfg is not None:
        from src.utils.rolling_logging import RollingWindow
        ep_window = RollingWindow(logging_cfg['smoothing_episodes'],
                                  logging_cfg['interval_episodes'], name="episode")
        step_window = RollingWindow(logging_cfg['smoothing_iters'],
                                    logging_cfg['interval_iters'], name="step")
```

#### `train.py` — do not clear Buffer B (line 1198)

```python
# BEFORE (1197-1199):
                # Reset behavior depends on accumulation mode
                if not log_accumulate or (iteration - 1) % log_interval == 0:
                    iteration_episodes = []

# AFTER:
                # Reset behavior depends on accumulation mode.
                # Two-level path: Buffer B EVICTS (deque maxlen), never clears — the
                # clear-after-emit is exactly what welds window to interval today.
                if logging_cfg is None:
                    if not log_accumulate or (iteration - 1) % log_interval == 0:
                        iteration_episodes = []
```

#### `train.py` — episode push + emission (lines 1289–1352)

This is the core change. At the per-episode finalisation site (**line 1292**,
`iteration_episodes.append(ep_data)` — the Buffer A drain), push into Buffer B and emit when the
window says so. The existing emission block at **1309–1352** becomes a function so it can be
called from the push site.

```python
# BEFORE (1289-1292):
                                # Store for moving average (tqdm)
                                ep_info_buffer.append(ep_data)
                                # Store for iteration-level logging (Stage 3)
                                iteration_episodes.append(ep_data)

# AFTER:
                                # Store for moving average (tqdm)
                                ep_info_buffer.append(ep_data)
                                if logging_cfg is None:
                                    # LEGACY: Buffer A drains into a cleared-per-window list.
                                    iteration_episodes.append(ep_data)
                                else:
                                    # Two-level: Buffer A (this step's finishes, order
                                    # irrelevant — simultaneous finishes are exchangeable)
                                    # drains into Buffer B (rolling, maxlen=smoothing_episodes).
                                    if ep_window.push(ep_data) and wandb_enabled:
                                        _emit_episode_row(list(ep_window.buf),
                                                          total_episodes_completed)
```

Refactor **1309–1352** into `_emit_episode_row(eps, total_eps)`, defined next to
`_bm_log_wandb` (near **line 1000**). It keeps every existing key and adds spread:

```python
    def _emit_episode_row(eps, total_eps):
        """Emit one WandB row aggregated over `eps` (a window of episode dicts).
        Shared by the two-level path (rolling window) and the legacy path
        (cleared-per-interval list) so key coverage can never diverge."""
        from src.utils.rolling_logging import spread
        ep_log = {"Episode/Number": total_eps, **_stage_tag()}
        # Spread on the two headline metrics: a mean survival of 133 that is secretly
        # bimodal (~400 no-predator vs ~20 with-predator) looks fine while hiding a
        # mixture — the std exposes it. (Reward_Min/Max already existed; keep names.)
        spread([ep['r'] for ep in eps], "Episode/Reward", ep_log)
        spread([ep['l'] for ep in eps], "Episode/Steps",  ep_log)
        if 'ate_food' in eps[0]:
            ep_log.update({ ...unchanged block from lines 1322-1339, with
                            `iteration_episodes` -> `eps`... })
            term_reasons = [ep['termination_reason'] for ep in eps]
            for code, name in [(1, 'MaxSteps'), (2, 'Starvation'), (3, 'Overeating'), (4, 'Injury')]:
                ep_log[f"Episode/Term_{name}"] = np.mean([1.0 if r == code else 0.0 for r in term_reasons])
            _append_per_tag_means(ep_log, eps, neutral_tags,  'mean_dist_rabbit',   'Episode/MeanDistRabbit')
            _append_per_tag_means(ep_log, eps, predator_tags, 'mean_dist_predator', 'Episode/MeanDistPredator')
            if bm_enabled:
                _bm_log_wandb(ep_log, eps)
        wandb.log(ep_log)
```

> **Key-compatibility requirement.** `spread(..., "Episode/Reward", ...)` writes
> `Episode/Reward`, `Episode/Reward_Std`, `Episode/Reward_Min`, `Episode/Reward_Max`. Today's code
> (1313-1315) writes `Episode/Reward`, `Episode/Reward_Min`, `Episode/Reward_Max`. The three
> existing names are preserved exactly; `_Std` is new. `Episode/Steps` gains `_Std`/`_Min`/`_Max`,
> all new. **No existing WandB key is renamed or dropped** — dashboard history stays intact. The
> `wandb.define_metric("Episode/*", step_metric="Episode/Number")` pattern at **train.py:676**
> already covers the new keys; no define_metric change needed.

Then the legacy block at 1309–1352 collapses to:

```python
# BEFORE (1308-1309):
                    # Log AGGREGATED stats for the iteration (Stage 3)
                    if wandb_enabled and iteration_episodes and iteration % log_interval == 0:
                        ...50 lines...

# AFTER:
                    # LEGACY path only — two-level path emits at the push site above.
                    if (logging_cfg is None and wandb_enabled and iteration_episodes
                            and iteration % log_interval == 0):
                        _emit_episode_row(iteration_episodes, total_episodes_completed)
```

#### `train.py` — step-level window (lines 1358–1403)

```python
# BEFORE (1358-1365):
                    avg_policy_loss = jnp.mean(jnp.array([l[1][0] for l in losses]))
                    ... (avg_value_loss, avg_ent_loss, avg_grad_norm, avg_mod_grad_norm, total_loss)
                    if wandb_enabled and iteration % log_interval == 0:
                        wandb_logs = {
                            "loss/total": total_loss,
                            ...
                        }

# AFTER: keep the avg_* computation unchanged, then:
                    _step_sample = {
                        "loss/total":     float(total_loss),
                        "loss/policy":    float(avg_policy_loss),
                        "loss/value":     float(avg_value_loss),
                        "loss/entropy":   float(avg_ent_loss),
                        "loss/grad_norm": float(avg_grad_norm),
                    }
                    if mod_info is not None:
                        _step_sample.update({ ...the modulator/* scalars from 1377-1394,
                                              float()-ed, unchanged keys... })
                    if logging_cfg is None:
                        _do_step_log = wandb_enabled and iteration % log_interval == 0
                        _step_vals = [_step_sample]
                    else:
                        _emit = step_window.push(_step_sample)
                        _do_step_log = wandb_enabled and _emit
                        _step_vals = list(step_window.buf)
                    if _do_step_log:
                        from src.utils.rolling_logging import spread
                        wandb_logs = {}
                        for k in _step_vals[0]:
                            spread([s[k] for s in _step_vals if k in s], k, wandb_logs)
                        wandb_logs.update({
                            "timesteps": global_step,
                            "iteration": iteration,
                            "Time/sps_env": global_step / max((datetime.now() - start_time).total_seconds(), 1e-9),
                            **_stage_tag(),
                        })
                        wandb.log(wandb_logs)
```

> **Note on the `modulator/*` keys.** Lines 1378-1394 currently log `_mean`/`_std`/`_min`/`_max`
> of the *within-iteration* distribution (e.g. `modulator/gamma_uni_std` = std across the rollout
> batch). Feeding those through `spread()` would produce `modulator/gamma_uni_std_Std`, which is
> the std-across-iterations of the within-iteration std — meaningful, but the names get confusing.
> **Decision: put only the five `loss/*` scalars through `spread()`. The `modulator/*` keys keep
> their current single-iteration semantics** and are taken from the *most recent* sample
> (`_step_vals[-1]`), not spread. Rationale: they already carry their own spread, and renaming
> them would break dashboard history. The developer must keep `modulator/*` names byte-identical.

#### `train.py` — CLI help text (line 248)

```python
# BEFORE:
    parser.add_argument("--log-interval", type=int, help="WandB logging interval in iterations (default: 1)")

# AFTER:
    parser.add_argument("--log-interval", type=int,
                        help="DEPRECATED (use the config `logging:` block; ignored when that "
                             "block is present). Legacy WandB logging interval in ITERATIONS "
                             "— note 1 rPPO iteration = num_steps*num_envs env-steps, 128x a "
                             "Dreamer iteration. See "
                             "docs/develop/active/refactors/TWO_LEVEL_LOGGING_REDESIGN.md")
```

#### `train.py` — OUT OF SCOPE (do not touch)

Lines **1538** (DQN), **1744** (DRQN), **1897** (PPO): `if wandb_enabled and iteration % log_interval == 0`.
These branches stay on the legacy path. Since `logging_cfg` only diverts the `RecurrentPPO`
branch, and `log_interval` remains resolved for all branches, they are unaffected. **Do not
"helpfully" migrate them.**

---

#### `src/algorithms/dreamer_srl/dreamer_srl_main.py` — knob resolution (lines 1160–1173)

```python
# BEFORE (1160-1173):
    # log_every: iterations between WandB metric logs.
    # ...comment block...
    log_every = (
        args.log_interval
        or agent_cfg.get('training.log_interval')
        or env_cfg.get('training.log_interval', 50)
    )
    last_log_step = 0

# AFTER:
    from src.utils.rolling_logging import resolve_logging_cfg, RollingWindow
    # Two-level logging (docs/develop/active/refactors/TWO_LEVEL_LOGGING_REDESIGN.md).
    # Checked agent_cfg first, then env_cfg — mirrors the log_interval precedence below.
    def _log_cfg_get(path, default=None):
        v = agent_cfg.get(path, None)
        return v if v is not None else env_cfg.get(path, default)
    logging_cfg = resolve_logging_cfg(
        _log_cfg_get,
        defaults={'smoothing_episodes': 5000, 'interval_episodes': 200,
                  'smoothing_iters': 200, 'interval_iters': 100},
    )
    if logging_cfg is not None and args.log_interval is not None:
        print("[WARN] --log-interval is IGNORED: this config uses the two-level `logging:` "
              "block.", flush=True)
    ep_window = step_window = None
    if logging_cfg is not None:
        ep_window   = RollingWindow(logging_cfg['smoothing_episodes'],
                                    logging_cfg['interval_episodes'], name="episode")
        step_window = RollingWindow(logging_cfg['smoothing_iters'],
                                    logging_cfg['interval_iters'], name="step")

    # LEGACY log_every: iterations between WandB metric logs.
    # ...keep the existing comment block verbatim...
    log_every = (
        args.log_interval
        or agent_cfg.get('training.log_interval')
        or env_cfg.get('training.log_interval', 50)
    )
    if logging_cfg is None:
        print("[DEPRECATION] training.log_interval is deprecated — see "
              "docs/develop/active/refactors/TWO_LEVEL_LOGGING_REDESIGN.md", flush=True)
    last_log_step = 0
```

#### `dreamer_srl_main.py` — episode push + emission (line 1370)

**This is the decoupling change and the most important one in the Dreamer file.** Today the
episode row is emitted **inside** the step-log gate (`if last_losses and (iter_num - last_log_step
>= log_every ...)` at **line 1841**). Under the new design the two levels have independent
cadences, so **episode emission must move out of that gate** and fire at the push site.

```python
# BEFORE (1370):
                iteration_episodes.append(ep_data)

# AFTER:
                if logging_cfg is None:
                    iteration_episodes.append(ep_data)
                else:
                    # Buffer A (this step's finishes) drains into Buffer B (rolling).
                    if ep_window.push(ep_data) and use_wandb:
                        _emit_episode_row(list(ep_window.buf), total_episodes_completed,
                                          policy_step)
```

Refactor the block at **1848–1898** into `_emit_episode_row(eps, total_eps, step)` defined before
the training loop (after the `neutral_tags`/`predator_tags`/`bm_enabled` bindings are available —
**note the closure hazard below**). Identical shape to the `train.py` version: use `spread()` for
`Episode/Reward` and `Episode/Steps`, keep every other key and its name unchanged, and finish with
`wandb.log(ep_log, step=step)`.

> **⚠️ Closure hazard — `neutral_tags` / `predator_tags` / `bm_enabled` are REBOUND at the
> curriculum stage swap** (lines **1512–1530**: `neutral_tags = tuple(env_params.neutral_tags)`,
> `_bm_state = make_bm_state(...)`). A nested function that closes over these names reads the
> *current* binding at call time, which is what we want — but only if the emitter is defined with
> `def` in the same scope and the swap uses plain assignment (it does; they are locals of `main()`).
> The developer must **not** capture these as default arguments (`def _emit(..., tags=neutral_tags)`)
> — that would freeze the pre-swap roster and re-open the exact bug that
> [`dreamer_srl_main.py:1498-1510`](../../../../src/algorithms/dreamer_srl/dreamer_srl_main.py)
> already comments on ("Risk 3 mitigation").

#### `dreamer_srl_main.py` — stage-swap must clear Buffer B (line 1510)

```python
# BEFORE (1497-1510):
                    # 4. Wipe in-flight episode accumulators (all envs — partial
                    #    episodes dropped). Risk 3 mitigation: clear iteration_episodes
                    #    so pre-swap per-tag keys don't reach WandB fan-out.
                    ...
                    iteration_episodes = []

# AFTER:
                    ...
                    iteration_episodes = []
                    # Two-level: Buffer B holds pre-swap episodes carrying the OLD tag
                    # roster. Same Risk 3 mitigation — a rolling window would otherwise
                    # keep feeding pre-swap per-tag keys into the fan-out for up to
                    # smoothing_episodes episodes after the swap. Reset the counter too,
                    # so the warm-up gate re-arms and the first post-swap row is again a
                    # full window of post-swap episodes.
                    if ep_window is not None:
                        ep_window.buf.clear()
                        ep_window.count = 0
```

> This is a **behavioral decision worth flagging**: after a curriculum stage swap, the first
> episode row is delayed by `smoothing_episodes` episodes (~16 min for Dreamer). The alternative —
> letting the window bleed across the swap — mixes two different environments into one mean and
> emits per-tag keys for entities that no longer exist. The delay is the correct trade. The
> step-level window is **not** cleared (losses are continuous across a swap and the tag roster
> does not enter them).

#### `dreamer_srl_main.py` — step-level window (lines 1841–1943)

```python
# BEFORE (1841-1848):
        if last_losses and (iter_num - last_log_step >= log_every or will_be_last):
            last_log_step = iter_num
            sps_env = policy_step / max(time.time() - t_start, 1e-9)
            # Commit 2: per-iteration episode aggregation block.
            if use_wandb and iteration_episodes:
                ...episode block 1849-1898 — MOVES OUT to _emit_episode_row...

# AFTER:
        # Step-level gate. Episode-level emission has moved to the push site (L1370)
        # under the two-level path — the two cadences are now independent.
        if logging_cfg is None:
            _do_step_log = bool(last_losses) and (iter_num - last_log_step >= log_every or will_be_last)
            _step_vals = [last_losses] if last_losses else []
        else:
            _emit = bool(last_losses) and step_window.push(dict(last_losses))
            _do_step_log = _emit or (will_be_last and bool(last_losses))
            _step_vals = list(step_window.buf)
        if _do_step_log:
            last_log_step = iter_num
            sps_env = policy_step / max(time.time() - t_start, 1e-9)
            if logging_cfg is None and use_wandb and iteration_episodes:
                _emit_episode_row(iteration_episodes, total_episodes_completed, policy_step)
                iteration_episodes = []  # LEGACY clear-after-emit
            ...
```

Inside the `log_dict` construction (**1901–1943**), the per-key `float(mv)` loop must average over
the window instead of taking the last value. Keep the prefix-sort routing (`Behavior/*`,
`WorldModel/*`, aliases) **exactly as-is** — only the value changes from "last iteration" to "mean
over the window", plus `_Std`/`_Min`/`_Max` siblings:

```python
            # Mean/std/min/max each loss key over the step window (single sample on the
            # legacy path => std=0, min=max=mean, i.e. numerically identical curve).
            from src.utils.rolling_logging import spread
            _agg = {}
            for mk in _step_vals[-1]:
                spread([float(s[mk]) for s in _step_vals if mk in s], mk, _agg)
            # ...then the EXISTING prefix-sort loop, iterating `_agg` instead of
            # `last_losses.items()`, so `loss_actor` -> `Behavior/loss_actor` and
            # `loss_actor_Std` -> `Behavior/loss_actor_Std` fall out of the same
            # startswith() rules. Verify: `moments_invscale_Std` must NOT collide with
            # the explicit "Diagnostic/moments_invscale" key set at the top of log_dict.
```

> **`will_be_last` interaction.** The final-iteration flush must survive: on the last iteration we
> log even if the window is not full or the interval has not elapsed. Under the two-level path
> that means `_do_step_log` ORs in `will_be_last`, and `spread()` over a partial window is fine
> (it just averages fewer samples). Correspondingly, **add a final episode flush** after the loop:
> if `ep_window` is non-empty at exit, emit one last row so a short run is not silently
> row-less. Guard it with `if ep_window is not None and len(ep_window.buf) > 0`.

#### `dreamer_srl_main.py` — CLI help text (lines 438–441)

Same deprecation wording as `train.py:248`.

---

#### `configs/train/default.yaml` (after line 25, `log_accumulate: true`)

```yaml
# BEFORE (tail of the `training:` block):
  log_interval: 10
  log_accumulate: true

# AFTER:
  # DEPRECATED — kept as the backward-compatible fallback. A config that declares a
  # top-level `logging:` block (below) ignores these two entirely.
  # See docs/develop/active/refactors/TWO_LEVEL_LOGGING_REDESIGN.md
  log_interval: 10
  log_accumulate: true

# Two-level logging. These are the DREAMER values: dreamer_srl_main.py reads this file
# and does NOT load configs/train/recurrent_ppo.yaml, which overrides the per-algo knobs.
# Every knob names its own unit. smoothing = NOISE (how many samples averaged per point);
# interval = VOLUME (how often a row is written). They are independent on purpose.
logging:
  episode:
    # UNIVERSAL — must be IDENTICAL in every algorithm. This is the ONLY thing that makes
    # a Dreamer curve and an rPPO curve equally noisy and therefore comparable.
    # recurrent_ppo.yaml deliberately does NOT override it. Do not add a CLI flag for it.
    # 5000 eps ~= 3-4% of a run; first point lands ~16 min in (measured, see the doc).
    smoothing_episodes: 5000
    # PER-ALGO (row volume). Dreamer runs ~131k eps/24h -> ~655 rows/24h. < smoothing -> overlap.
    interval_episodes: 200
  step:
    # PER-ALGO. Losses are never cross-compared between algorithms, so no universality rule.
    # 200 iters = 3200 env-steps of loss averaging at num_envs=16.
    smoothing_iters: 200
    # < smoothing -> overlapping rolling windows.
    interval_iters: 100
```

#### `configs/train/recurrent_ppo.yaml` (after line 19)

```yaml
# BEFORE (tail of file):
training:
  log_interval: 500             # one WandB row every 500 training iterations
  checkpoint_frequency: 200000
  max_checkpoints_to_keep: null

# AFTER:
training:
  # DEPRECATED — legacy fallback only; ignored when the `logging:` block is active.
  log_interval: 500             # one WandB row every 500 training iterations
  checkpoint_frequency: 200000
  max_checkpoints_to_keep: null

# Two-level logging: rPPO's PER-ALGO overrides on top of configs/train/default.yaml.
# NOTE the deliberate omission: logging.episode.smoothing_episodes is NOT set here.
# It is inherited from default.yaml so that rPPO and Dreamer necessarily share it —
# that inheritance is the mechanism that enforces the universality rule. Do not add it.
logging:
  episode:
    # rPPO's episode throughput is far below Dreamer's, so a larger interval keeps
    # rows/session in the same ballpark. 4000 < 5000 -> windows still overlap.
    interval_episodes: 4000
  step:
    # 1 rPPO iteration = num_steps(128) * num_envs(16) = 2048 env-steps, i.e. 128x a
    # Dreamer iteration. 100 iters = ~204k env-steps of loss averaging.
    smoothing_iters: 100
    # 50 exactly preserves today's step-row cadence, so existing dashboards keep density.
    interval_iters: 50
```

---

## Checkpoints

- [ ] **CP1 — Backward compat is airtight (do this FIRST).** With **no** `logging:` block in any
      config, run a short rPPO smoke and confirm the emitted WandB keys and row count are
      **identical** to a pre-change run at the same seed. This is the checkpoint that protects
      the live basic04 sweep. If it fails, stop.
- [ ] **CP2 — `RollingWindow` unit test.** Add `tests/test_rolling_logging.py` asserting the two
      worked examples in this doc reproduce exactly: (a) episode level `smoothing=4, interval=2`
      over the stream `[20,35,50,12,60,18,44]` emits at ep4 (mean 29.25) and ep6 (mean 35.0) —
      **not** at ep2, because the warm-up gate requires a full window; (b) step level
      `smoothing=3, interval=2` over `[8,6,5,4,4.5,3.5]` emits at iter4 (mean 5.0) and iter6
      (mean 4.0). Also assert `smoothing < interval` prints the warning and still runs.
- [ ] **CP3 — Warm-up gate.** No episode row is emitted before `smoothing_episodes` episodes have
      completed; the first row lands at the first multiple of `interval_episodes` that is
      ≥ `smoothing_episodes`.
- [ ] **CP4 — Overlap is real.** With `smoothing_episodes=5000, interval_episodes=200`, log
      `len(ep_window.buf)` at each emission and confirm it stays pinned at 5000 (evicting, not
      clearing). If it ever drops to 0 after an emission, the clear-after-emit path is still live.
- [ ] **CP5 — No key regressions.** Diff the WandB key set of a new-path run against a legacy run.
      The only difference must be **added** `*_Std` / `*_Min` / `*_Max` keys. Any **removed** or
      **renamed** key is a bug — `Episode/Reward_Min` / `Episode/Reward_Max` / every `Episode/*`
      behavior key / every `modulator/*` key must survive byte-identical.
- [ ] **CP6 — Dreamer stage-swap.** In a curriculum run, confirm `ep_window` clears at the swap and
      that no post-swap row carries a per-tag key for a pre-swap entity roster.
- [ ] **CP7 — Speed.** Record steps-per-second before/after on the same node/config/seed, over a
      long-enough window that warm-up does not dominate. The change is pure Python bookkeeping on
      the host, but Buffer B now holds 5,000 episode dicts (each with tens of keys) and each
      emission means-over-5000 across every key. **Watch for the emission cost**: at
      `interval_episodes=200`, a 5000-episode aggregate runs every 200 episodes. If that shows up
      as a >5% slowdown, report it — the fix is to keep running sums rather than re-reducing the
      window, but do not pre-optimize.

## Verification plan

Two short smoke runs, one per trainer, on any free GPU (see the `gpu-status` skill — this is a
smoke, so take a low-tier card):

1. **Dreamer** — a temporary config with `smoothing_episodes: 20, interval_episodes: 5,
   smoothing_iters: 6, interval_iters: 3` (scaled down so the smoke terminates in seconds while
   still exercising the overlap and warm-up logic), `--episodes 200`. Verify: first `Episode/*`
   row at episode 20; rows every 5 episodes thereafter; `Episode/Steps_Std` present and non-zero;
   `WorldModel/loss_model_Std` present; loss rows on their own independent cadence.
2. **rPPO** — same scaled-down knobs, short `--episodes`. Same checks, plus confirm
   `--log-interval 100` on the CLI prints the ignored-warning and does **not** change the cadence.
3. **Legacy regression** — the same two smokes with the `logging:` block deleted. Row cadence and
   key set must match the pre-change baseline exactly (this is CP1 re-run as a final gate).
4. **Deprecation notice** — confirm it prints once, not once per iteration.

## Follow-ups (not part of this change)

- **Correct the stale memory.** `20260529_1825_log_interval_anchored_rows_per_session` quotes
  ~38 SPS / 20–40k eps/24h from a `replay_ratio=1` run. Current runs at `replay_ratio=0.0625` do
  147 SPS / ~131k eps/24h — ~4× faster. Anyone deriving a cadence from the old numbers will be off
  by 4×. Append a correction via `/memorize`.
- **Migrate the existing Dreamer configs.** `configs/models/dreamer_srl/01_food_only_buf256k.yaml`
  and its `_log5k` / `_log50k` siblings exist *only* to sweep `log_interval`. Once the `logging:`
  block is live and the noise/volume axes are separate, that sweep is obsolete and those variant
  configs should be archived. Route to `experiment-designer` (config owner) — **not** in scope here.
- **`train_command-agent.sh`** carries ~15 `# log_interval=50` comments that will be misleading
  once configs migrate. Cosmetic; fold into whichever change next touches that file.

## Implementation Report

> **Implemented by**: _(developer)_
> **Date**: _

## Verification Report

> **Verified by**: _(senior-developer)_
> **Date**: _

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| | | | |

**Conclusion**: _
