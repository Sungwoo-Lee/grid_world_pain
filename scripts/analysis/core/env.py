#!/usr/bin/env python3
"""env.py - everything derived from a run's own saved config, in one place.

These five helpers existed in three copies each across the analysis layer, and every copy had
drifted textually. Before consolidating them they were tested for BEHAVIOURAL equivalence on real
configs, because textual drift and behavioural drift are different questions and only the second one
decides whether merging is safe:

    smell_channels        3 copies, 3 distinct texts, ONE distinct result   -> merged as-is
    nociception_kernel    2 copies, 2 distinct texts, ONE distinct result   -> merged as-is
    listcol               3 copies, 3 distinct texts, ONE distinct result   -> merged as-is
    slot_layout           3 copies, 3 distinct texts, THREE distinct results

`slot_layout` genuinely differed, and the difference is worth stating precisely because it is the
kind that looks alarming and is not: all three agree on every slot list any caller reads - `pred`,
`neutral`, `bush`, `rock`, `food`, `ambush` are identical - and differ only in whether they also
return the convenience counts `n_animal`, `n_obs`, `n_res`. This module returns the superset. That is
safe only because no caller iterates the dict or takes its length; every one of the four callers
subscripts it by name, which was checked rather than assumed.

The configs these read are the ones the TRAINER saved beside each run's checkpoints, not the ones in
the repo - so the layout always describes the world the agent was actually in.
"""
from __future__ import annotations
import numpy as np


def slot_layout(cfg: dict) -> dict:
    """Entity / obstacle / resource slot indices, derived from the run's own config.

    Slots are laid out by declaration order within each family, so the index of an animal in the
    step table's `animal_row` column is its position here. `count_high` rather than the sampled
    count: the columns are allocated for the maximum, and unused slots are masked by `animal_active`.
    """
    env = cfg["environment"]
    pred, neu, s = [], [], 0
    for e in env["entities"]:
        n = e["count_high"]
        (pred if e["class"] == "predator" else neu).extend(range(s, s + n)); s += n
    bush, rock, s = [], [], 0
    for o in env["obstacles"]:
        n = o["count_high"]
        (bush if o.get("hides_agent") else rock).extend(range(s, s + n)); s += n
    food, amb, s = [], [], 0
    for r in env["resources"]:
        n = r["count_high"]
        (amb if max(r.get("damage", [0, 0])) > 0 else food).extend(range(s, s + n)); s += n
    return dict(pred=pred, neutral=neu, bush=bush, rock=rock, food=food, ambush=amb,
                n_animal=len(pred) + len(neu), n_obs=len(bush) + len(rock),
                n_res=len(food) + len(amb))


def smell_channels(cfg: dict) -> tuple[int, int]:
    """The two olfactory channels that separate predators from neutrals, derived not assumed."""
    ent = cfg["environment"]["entities"]
    pm = np.mean([e["properties"] for e in ent if e["class"] == "predator"], axis=0)
    nm = np.mean([e["properties"] for e in ent if e["class"] != "predator"], axis=0)
    d = np.asarray(pm) - np.asarray(nm)
    a, b = int(np.argmax(d)), int(np.argmin(d))
    if a == b or d[a] <= 0 or d[b] >= 0:
        raise SystemExit("this run's config does not separate predator and neutral odour")
    return a, b


def nociception_kernel(cfg: dict) -> np.ndarray:
    """The alpha kernel the environment convolves the injury buffer with (config_loader.py:1424)."""
    n = int(cfg["sensory"]["interoceptive_kernel_length"])
    tau = float(cfg["sensory"]["interoceptive_kernel_tau"])
    k = np.arange(n, dtype=np.float64)
    raw = (k / tau) * np.exp(1.0 - k / tau)
    return raw / raw.sum()


def perceived_nociception(inj, t, estart, kernel):
    """Rebuild the scalar the nociceptor actually hands the agent, row by row.

    Mirrors core.py:115 (roll the buffer, write the new injury at slot 0) and core.py:1112 (the
    buffer is ZEROED at reset, so the reset row's injury is never in it). The `src > estart` guard
    is strict for exactly that reason - including the reset row was a real bug once (Known Bugs,
    2026-08-25) and it leaks the randomised starting injury into the first steps as if the agent had
    felt it immediately, which is precisely the claim the sensor-ladder study tests.
    """
    idx = np.arange(len(t))
    out = np.zeros(len(t))
    for j in range(1, len(kernel)):
        src = idx - j
        ok = src > estart
        out[ok] += kernel[j] * inj[src[ok]]
    return out


def listcol(col, width):
    """Fixed-width parquet list column -> (n, width) array without the slow to_pylist path."""
    ch = col.chunks if hasattr(col, "chunks") else [col]
    return np.concatenate([c.flatten().to_numpy(zero_copy_only=False)
                           for c in ch]).reshape(-1, width)
