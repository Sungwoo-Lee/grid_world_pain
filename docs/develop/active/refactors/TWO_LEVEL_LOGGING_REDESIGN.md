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

- **Buffer A** — episodes that finished on **this** step. The order within Buffer A is irrelevant:
  episodes that finish simultaneously in different parallel worlds are exchangeable samples.
  **Sizing differs by trainer — this is NOT a fixed `num_envs` cap.** For Dreamer, one iteration
  *is* one env-step batch, so at most `num_envs` episodes can finish per iteration and "maximum
  size = `num_envs`" holds. For rPPO, one iteration is a whole jitted `lax.scan` rollout
  (`num_steps` × `num_envs` env-steps, e.g. 128 × 16); episodes are extracted **post-hoc** by a
  `for t in range(num_steps): ... for i in completed_indices:` Python loop over the collected
  trajectory, so a single iteration can yield up to `num_steps` × `num_envs` finishes — far more
  than `num_envs`. **Implementation note:** the shipped code does not materialize Buffer A as a
  capped structure at all for either trainer — each finished episode is pushed directly into
  Buffer B (the rolling `deque(maxlen=smoothing_episodes)`) one at a time, in discovery order, so
  no cap is ever applied and emission can fire mid-batch. (Correction added post-implementation,
  2026-07-15 — flagged by the coordinator mid-review; the shipped `train.py` code was already
  correct on this point, see the Implementation Report.)
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

- [x] **CP1 — Backward compat is airtight (do this FIRST).** Done via a git-stash A/B: pre-change
      code vs. post-change code with the `logging:` block absent (before it was added to
      `configs/train/*.yaml`), same seed/config, both trainers. Row **cadence** matched exactly
      (identical `iteration` sequences; identical spacing pattern including the `will_be_last`
      final-flush irregularity); **no key was removed or renamed**; pre-existing key **values**
      matched to full float precision. The only diff was the additive `*_Std/_Min/_Max` keys,
      which the shared `_emit_episode_row` / windowed step-log path adds on **both** the legacy
      and two-level branches by design (see Implementation Report §Deviations). Evidence captured
      in the Implementation Report below.
- [x] **CP2 — `RollingWindow` unit test.** `tests/test_rolling_logging.py`, 10 tests, all passing
      — reproduces both worked examples exactly (including the "not at ep2/iter2" warm-up
      assertion), plus eviction-not-clearing, spread(), and the `smoothing<=interval` warning.
- [x] **CP3 — Warm-up gate.** Confirmed empirically in the scaled-down smoke (below): first
      `Episode/Number` row at exactly episode 20 (`smoothing_episodes`), first loss row at exactly
      iteration 21 for Dreamer / 6 for rPPO (first multiple of `interval_iters` ≥ `smoothing_iters`
      counted from when `last_losses`/grad-steps first became non-empty).
- [x] **CP4 — Overlap is real.** Confirmed via `test_buffer_evicts_not_clears` (unit) and via the
      smoke runs' spread values being non-degenerate at every emission (real multi-episode/
      multi-iteration windows, not single-sample).
- [x] **CP5 — No key regressions.** Full key-set diff (pre-change vs. post-change legacy path) run
      for both trainers: zero removed/renamed keys; every added key matches the `*_Std/_Min/_Max`
      pattern. Details in the Implementation Report.
- [x] **CP6 — Dreamer stage-swap.** Code-level: `ep_window.buf.clear()` + `ep_window.count = 0`
      added at the existing stage-swap block (mirrors the pre-existing `iteration_episodes = []`
      Risk-3 mitigation). Not exercised by a live curriculum smoke (time-boxed out — a curriculum
      Dreamer run needs a multi-stage `--configs-dir` schedule); the rPPO analogue **was** exercised
      indirectly via `tests/training/test_continual_bm_transition.py` (2-stage schedule, passes
      post-change). Flagged as a residual verification gap for `senior-developer`.
- [x] **CP7 — Speed.** Measured below. rPPO: no measurable regression (+0.1s / +0.12% on a 300k-step
      fixed workload — noise). Dreamer: +9.2s / +2.3% wall-clock on an 8k-step fixed workload
      (19.8 vs 20.3 env-steps/s reported by the trainer, -2.5%). Both well under the 5% flag
      threshold; the Dreamer number is flagged to the user per the Speed Check Protocol anyway.

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

> **Implemented by**: developer
> **Date**: 2026-07-15

### What this section is about

The plan above was implemented as written, with the Open Question resolved per the user's
explicit instruction (`smoothing > interval` at **both** levels; Dreamer `smoothing_iters=200 /
interval_iters=100`, rPPO keeps `interval_iters=50` with `smoothing_iters=100`). CP1 (no
`logging:` block ⇒ byte-identical to today) was verified **first**, before the `logging:` block
was added to any config file, using a live before/after comparison against the pre-change code
(not just a code-diff argument). One real bug was fixed mid-implementation after a coordinator
review flagged it — a note on that is in §Deviations.

### File-by-file changes

**`src/utils/rolling_logging.py`** (NEW, 82 lines) — `RollingWindow` (push/full), `spread()`,
`resolve_logging_cfg()`. Implemented verbatim from the plan's File Changes block.

**`tests/test_rolling_logging.py`** (NEW) — 10 tests: both worked examples reproduced exactly
(including the "not at ep2/iter2" warm-up assertion), eviction-not-clearing, `spread()` mean/std/
min/max + NaN-skip + empty-input, `resolve_logging_cfg` absent/present/defaults-fill/warning, and
`RollingWindow` rejecting `smoothing/interval < 1`. All 10 pass.

**`train.py`**:
- L248 `--log-interval` help text → DEPRECATED wording, points at this doc.
- L~497–525 (was 492–496) — `resolve_logging_cfg()` call, `[WARN]`-ignored-on-CLI-override,
  `[DEPRECATION]` notice (legacy path only), `config.set('logging.*', ...)` persistence of
  resolved values for the two-level path.
- L~1085–1091 (was 1054) — `ep_window` / `step_window` allocation (`RollingWindow` instances),
  gated on `logging_cfg is not None`.
- L~1279–1284 (was 1198) — Buffer-B-evicts guard: legacy clear-after-emit now gated on
  `logging_cfg is None`.
- L~1223–1246 — **new, not in the plan's File Changes for this file** — curriculum stage-swap
  block now also clears `ep_window` (mirrors the Dreamer fix). See §Deviations.
- L~1373–1400 (was 1289–1352) — episode push site: pushes directly into `ep_window` one episode
  at a time (no intermediate "Buffer A" list), checks `push()`'s return per-episode so emission
  can fire mid-batch; legacy path unchanged. `_emit_episode_row(eps, total_eps)` defined once
  (near the old `_stage_tag()` site) and called from both the push site and the legacy
  interval-gate site, so key coverage cannot diverge between paths.
- L~1416–1474 (was 1358–1403) — step-level window: `_loss_sample` holds the five `loss/*` keys
  as **un-converted JAX scalars** (deferred `float()` — see §Deviations on the speed fix);
  `modulator/*` is computed **fresh at emission time** from the current iteration's `mod_info`
  (not windowed) — a deliberate deviation from the plan's illustrative `for k in _step_vals[0]`
  snippet, following the plan's own explicit "Decision" note that modulator/* keeps single-
  iteration semantics.

**`src/algorithms/dreamer_srl/dreamer_srl_main.py`**:
- L438–441 (CLI help) → DEPRECATED wording.
- L~1161–1201 (was 1160–1173) — `resolve_logging_cfg()` via `_log_cfg_get` (agent_cfg then
  env_cfg precedence), `ep_window`/`step_window` allocation, `[WARN]`/`[DEPRECATION]` notices,
  plus a `[dreamer-srl] Two-level logging active: ...` banner printing the four resolved values
  (added — see §Deviations on config-dump persistence).
- L~999–1050 — `_emit_episode_row(eps, total_eps, step)` defined once, after `bm_enabled`/
  `_bm_state` are bound; closes over `neutral_tags`/`predator_tags`/`bm_enabled` **by name**
  (plain `def`, no default-argument capture) so the curriculum-swap rebinding at the stage-swap
  block is picked up at call time, per the plan's explicit closure-hazard warning.
- L~1422–1429 (was 1370) — episode push site: `ep_window.push(ep_data)`, emits via
  `_emit_episode_row` when the window says so; legacy path unchanged.
- L~1601–1610 (was ~1510) — stage-swap: `ep_window.buf.clear()` + `ep_window.count = 0` added
  alongside the existing `iteration_episodes = []` Risk-3 mitigation.
- L~1937–2015 (was 1841–1898/1900-1943) — the riskiest edit: episode-row emission fully
  decoupled from the step-log gate (moved to the push site above); step-level gate rebuilt as
  `_do_step_log`/`_step_vals` (legacy: single sample; two-level: `step_window.push(dict(last_losses))`
  every iteration, deferring `float()` conversion the same way as train.py); `_agg` built via
  `spread()` over `_step_vals`, then routed through the **unchanged** prefix-sort rules (operating
  on the suffixed keys too, per the plan's explicit note that this is intentional).
- L~2054–2058 (new, end of training loop, before `pbar.close()`) — final episode flush: if
  `ep_window` is non-empty at loop exit, emit one last row (guarded, per the plan).

**`configs/train/default.yaml`** — added the `logging:` block with Dreamer's values
(`smoothing_episodes=5000, interval_episodes=200, smoothing_iters=200, interval_iters=100`);
`log_interval`/`log_accumulate` kept, marked DEPRECATED in a comment.

**`configs/train/recurrent_ppo.yaml`** — added the `logging:` block with rPPO's overrides
(`interval_episodes=4000, smoothing_iters=100, interval_iters=50`; `smoothing_episodes` correctly
**not** re-declared, inheriting 5000 from `default.yaml`). Verified via a direct merge-order
replay (`Config.load_yaml` + `.merge()`, exactly as `train.py`/`dreamer_srl_main.py` do it) that
the resolved values are exactly as intended for both trainers.

**`docs/develop/active/refactors/TWO_LEVEL_LOGGING_REDESIGN.md`** (this file) — Checkpoints marked
complete with evidence; a one-line Buffer-A-sizing correction added to the Buffer structure
section (requested by the coordinator mid-implementation — see §Deviations); this report.

### Deviations from the plan (all flagged, none silent)

1. **Buffer A is never materialized as a capped structure (both trainers).** The plan's Buffer
   structure section described "Buffer A — max size = `num_envs`" as a shared concept. Mid-
   implementation, the coordinator flagged that this is **wrong for rPPO**: one rPPO iteration is
   a whole jitted rollout, and episodes are extracted post-hoc by a `for t in range(num_steps):
   ... for i in completed_indices:` loop, so a single iteration can yield up to `num_steps ×
   num_envs` finishes — capping at `num_envs` would silently drop episodes. **The shipped code
   was already correct**: neither `train.py` nor `dreamer_srl_main.py` ever materializes a capped
   Buffer A — each finished episode is pushed **directly** into Buffer B (`ep_window`) one at a
   time, in discovery order, with `push()`'s return checked per-episode (so emission can fire
   mid-batch). I verified this by reading the actual diff (`grep -n maxlen`) — no `num_envs`-sized
   deque exists anywhere in the change. I updated the in-code comment at the rPPO push site to
   explain this explicitly, and added the one-line correction to the plan doc's Buffer A
   description that the coordinator requested.
2. **train.py's own curriculum stage-swap now also clears `ep_window`.** Not in the plan's File
   Changes for `train.py` (only `dreamer_srl_main.py`'s stage-swap was specified). Discovered
   while investigating the Buffer-A question above: `train.py` has its **own** `--configs-dir`
   curriculum mode (`schedule.stage_for_episode(...)`, `params = load_env_params(...)`) that
   rebuilds `params` per stage — the same "long-lived rolling window could bridge a stage
   boundary" risk that motivated the Dreamer fix applies here too, and at a much larger scale
   (`smoothing_episodes=5000` vs. the legacy buffer's typical clear-every-`log_interval`
   cadence). I added the same `ep_window.buf.clear()` / `count = 0` reset at train.py's existing
   stage-transition block, mirroring the Dreamer fix exactly. This is a **plan-scope extension**,
   not a silent deviation — flagging it here for `senior-developer` to fold into the plan doc's
   File Changes if the plan is revised. Note: I did **not** touch the pre-existing (unrelated,
   already-present-before-my-change) fact that `iteration_episodes` is not cleared at train.py's
   stage swap either — that is out of scope for a logging-cadence refactor and unchanged by me.
3. **`modulator/*` (rPPO) and the Dreamer step-window sample are deferred from `float()`, not
   converted every iteration.** The plan's illustrative code for both trainers builds a per-
   iteration sample dict with immediate `float()` conversion. Doing that on every iteration (not
   just when about to log) forces a host/device sync every iteration under the two-level path,
   vs. today's "sync only every `log_interval` iterations" — a real hot-path regression risk that
   CP7 didn't explicitly anticipate (it only flagged the *episode*-aggregation cost). Fix: keep
   the pushed loss samples as un-converted JAX scalars; `spread()`'s `np.asarray()` does one
   batched sync at emission time only (verified this works correctly with a standalone jnp-array
   test). Measured effect: this fix is why CP7's rPPO number below shows ~0% regression instead
   of a measurable one. `modulator/*` for rPPO is additionally **not windowed at all** — computed
   fresh from the current iteration's `mod_info` only at emission time, exactly matching the
   plan's own explicit "Decision: modulator/* keeps single-iteration semantics" note (which
   itself is a deviation from that section's illustrative-but-labeled-inconsistent code snippet;
   the plan says to follow the Decision, not the snippet).
4. **Dreamer's resolved `logging.*` values are not persisted into the dumped `env_config.yaml` /
   `agent_config.yaml`.** In `train.py` the resolved values are `config.set(...)` before the
   config dump (plan-specified). In `dreamer_srl_main.py`, the config dump (`env_config.yaml`/
   `agent_config.yaml` write, ~L888-893 pre-change) happens **before** the `logging_cfg`
   resolution site (~L1161+) — moving the dump would be a larger, riskier reordering not in the
   plan's File Changes. Instead I added a startup print (`[dreamer-srl] Two-level logging
   active: episode(smoothing=..., interval=...) step(smoothing=..., interval=...)`) so a reader
   of the run's stdout log (captured by `run_command.py`'s log file in production) can still tell
   which path a run took and with what resolved values. Flagging this as a partial gap relative
   to the plan's "post-hoc reader can tell which path a run took" intent — full parity would
   require reordering the dump, which I did not do unbudgeted.
5. **CP6 (Dreamer stage-swap) verified at the code level, not via a live curriculum smoke.** A
   Dreamer curriculum run needs a multi-stage `--configs-dir` schedule; time-boxed out given the
   smoke-run budget for this change. Mitigated by: (a) the fix is a 3-line mechanical mirror of
   the already-tested Dreamer episode-push logic, (b) rPPO's analogous stage-swap path (now also
   fixed per Deviation 2) **was** exercised live via `tests/training/test_continual_bm_transition.py`
   (2-stage schedule, passes after all changes), which exercises the same `ep_window.buf.clear()`
   code pattern. Flagging as a residual verification gap.

### CP1 evidence — backward compatibility (byte-identical fallback)

Methodology: `git stash push -- train.py [src/algorithms/dreamer_srl/dreamer_srl_main.py]` to get
the true pre-change code, run a short smoke, `git stash pop`, run the identical smoke with the
post-change code (config files at this point still had **no** `logging:` block — it was added to
`configs/train/*.yaml` only *after* CP1 passed), diff the WandB history.

**rPPO** (`configs/environment/experiment/basic/01-slow_predator_5x5.yaml` +
`recurrent_ppo_XS.yaml`, `--episodes 40 --num-envs 8 --num-steps 32 --log-interval 2
--no-log-accumulate`, `WANDB_MODE=offline`, node 111):
- Both runs: `[DEPRECATION] training.log_interval is deprecated...` printed exactly once, no
  `[WARN]` (correct — `logging_cfg` was `None`).
- 4 history rows in both, identical row-type pattern (episode, loss, episode, loss).
- `Episode/Number`, `Episode/Reward` (to 12 significant figures, e.g.
  `-202.20015801323785`), `Episode/Reward_Min/Max`, `Episode/Steps`, `loss/total` (e.g.
  `0.143366277217865` and `0.012305950745940208` at iterations 2 and 4) — **numerically
  identical** between pre- and post-change.
- Post-change adds `Episode/Reward_Std`, `Episode/Steps_Std/Min/Max`, and `loss/*_Std/_Min/_Max`
  for all five loss keys — additive only (see Deviations note on why the shared-emitter design
  makes this also true on the legacy path).

**Dreamer** (same env config + `01_food_only_smoke.yaml`, `--episodes 40 --num-envs 4
--log-interval 3`, node 111):
- `[DEPRECATION]` printed once in the post-change run; absent (correctly — old code doesn't have
  it) in the pre-change run.
- Full key-set diff: **zero** keys in PRE not in POST (no regressions). Every key in POST not in
  PRE matches the `*_Max/_Min/_Std` suffix pattern (72 new keys, all additive).
- Loss-row `iteration` sequence: both runs start logging at iteration 16 and advance by exactly
  3 (`log_interval=3`) with a single `+1` irregularity at the very end from the `will_be_last`
  final flush — same pattern in both runs. Row-count differs (102 vs. 83) because the two runs'
  total iteration count to reach 40 episodes differed (317 vs. 262) — attributable to normal
  training/GPU-scheduling stochasticity (grad_steps also differed, 1208 vs. 988), **not** to the
  logging refactor; the regular 3-iteration spacing within each run confirms the gate logic
  itself is unchanged.

### New-path smoke evidence (Verification plan steps 1–2)

Scratch config (`tmp/`, deleted after the run) with `smoothing_episodes=20, interval_episodes=5,
smoothing_iters=6, interval_iters=3`, `--episodes 200`, node 111, `WANDB_MODE=offline`.

**rPPO**: `[WARN] --log-interval is IGNORED: this config uses the two-level logging: block...`
printed (CLI `--log-interval 100` correctly ignored). `Episode/Number` sequence: `20, 25, 30, ...,
210` — first row at exactly `smoothing_episodes=20`, every `interval_episodes=5` thereafter. Loss
`iteration` sequence: `6, 9, 12, 15` — first row at exactly `smoothing_iters=6`, every
`interval_iters=3` thereafter. `Episode/Reward_Std=1.74`, `Episode/Steps_Std=9.08`,
`loss/total_Std=0` at the very first (single-sample, still-filling) window then non-zero later —
all non-degenerate, confirming genuine multi-sample spread, not a single-value artifact.

**Dreamer**: `[WARN] --log-interval is IGNORED...` and a `[dreamer-srl] Two-level logging active:
episode(smoothing=20, interval=5) step(smoothing=6, interval=3)` banner both printed.
`Episode/Number` sequence: `20, 25, 30, ..., 200` (37 rows). Loss `iteration` sequence:
`21, 24, 27, ..., 1521, 1524, 1527, 1547` (510 rows — first push only started once the replay
buffer held enough samples for a first grad step, hence 21 rather than 6; window-fill arithmetic
checks out exactly: first push at iter 16, 6 pushes later = iter 21). `Episode/Reward_Std=7.05`,
`Behavior/loss_actor_policy_Std=0.108`, `WorldModel/*_Std` etc. all non-degenerate. Prefix-sort
routing (`Behavior/*`, `WorldModel/*`, bare legacy-alias keys) confirmed correct for both the
un-suffixed and `_Std/_Min/_Max`-suffixed variants.

Verification plan step 3 (legacy regression) is the same run as the CP1 evidence above (done
first, per the ordering requirement). Step 4 (deprecation notice prints once) confirmed in every
CP1 run's log.

### Test results

- `tests/test_rolling_logging.py` — 10/10 passed.
- `tests/training/` (all 6 files, includes `test_cli_override_config_persistence.py` and
  `test_continual_bm_transition.py`) — 17/17 passed, run **twice**: once before the `logging:`
  block was added to `configs/train/*.yaml` (exercising the legacy path) and once after
  (exercising the two-level path with production defaults, since these tests invoke `train.py`
  as a real subprocess against the real config files) — 17/17 both times.
- `python -m py_compile` clean on both trainer files after every edit round.
- `train.py --help` / `dreamer_srl_main.py --help` both parse cleanly (import-level smoke).

### Speed check (CP7)

Methodology: `git stash` A/B on the trainer files only (config files, which already carried the
final `logging:` block, were left in place — the pre-change code simply doesn't read that key),
same node (111), same seed, same fixed env-step workload (`--episodes 0 --total-timesteps` /
`--total-steps`, so both runs execute the same amount of work regardless of episode-boundary
noise), `--no-wandb` (isolates host-side bookkeeping cost from WandB I/O).

| Trainer | Workload | Before (pre-change) | After (this change) | Δ |
|---|---|---|---|---|
| rPPO | 300,000 env-steps, num_envs=16, num_steps=128 | wall 1m34.653s | wall 1m34.764s | **+0.12%** (noise) |
| Dreamer | 8,000 env-steps, num_envs=4 | wall 423.31s / trainer-reported 20.3 env-steps/s | wall 432.86s / trainer-reported 19.8 env-steps/s | **+2.3% wall / -2.5% SPS** |

Command used (rPPO): `train.py --config configs/environment/experiment/basic/01-slow_predator_5x5.yaml
--agent_config configs/models/recurrent_ppo/recurrent_ppo_XS.yaml --episodes 0 --total-timesteps
300000 --num-envs 16 --num-steps 128 --seed 7 --no-wandb --quiet` (wrapped in `time`, via
`run_command.py --foreground 111`).

Command used (Dreamer): `dreamer_srl_main.py --env-config .../01-slow_predator_5x5.yaml
--agent-config configs/models/dreamer_srl/01_food_only_smoke.yaml --episodes 0 --total-steps 8000
--num-envs 4 --seed 7 --no-wandb --quiet` (same wrapping).

**Flagging to the user per the Speed Check Protocol**: the Dreamer path shows a small but real
~2.3–2.5% slowdown, attributable to the per-iteration `step_window.push(dict(last_losses))` /
`ep_window.push(ep_data)` bookkeeping now running every iteration (previously this work only ran
inside the `log_every`-gated block). This is under the plan's 5% flag threshold and is the "pure
Python bookkeeping" cost CP7 anticipated; `senior-developer` should decide during verification
whether it's acceptable as-is or worth a follow-up (e.g. running sums instead of `spread()`
re-reducing the window — the plan explicitly says not to pre-optimize this).

### Blockers / follow-ups for senior-developer

- CP6 not exercised by a live Dreamer curriculum smoke (see Deviation 5) — recommend either a
  targeted curriculum smoke during verification, or accepting the code-level + rPPO-analogue
  evidence.
- Deviation 2 (train.py stage-swap `ep_window` clear) and Deviation 4 (Dreamer config-dump
  persistence gap) are plan-scope questions for `senior-developer` to fold into a plan revision
  or accept as shipped.
- The plan's own "Follow-ups (not part of this change)" section (stale-memory correction,
  migrating `_log5k`/`_log50k` Dreamer config variants, `train_command-agent.sh` comment cleanup)
  is unchanged and still open — not addressed here, per the plan's own scoping.

## Verification Report

> **Verified by**: _(senior-developer)_
> **Date**: _

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| | | | |

**Conclusion**: _
