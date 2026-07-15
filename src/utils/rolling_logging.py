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
