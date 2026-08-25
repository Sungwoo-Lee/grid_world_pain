"""Shared helpers for the per-figure analysis scripts.

Every figure in the internal analysis has exactly ONE script in this folder. Each script is
self-contained and reviewable on its own: it states its question, reads a trajectory store, writes
a single JSON, and prints a table. Nothing here computes a result — this module only holds the
plumbing they would otherwise duplicate.

Design rules for scripts in this folder:
  * one script produces one figure's data, and nothing else
  * the slot layout is DERIVED from the run's saved config, never hardcoded, so a script works on
    any collected run
  * output goes to results/analysis/figures/<name>.json and is also printed
  * anything the script conditions on, excludes, or knows to be biased is stated in its docstring
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import pyarrow.parquet as pq
import yaml

OUT_ROOT = "results/analysis/figures"
DEFAULT_RUN = "results/JAX_RecurrentPPO/20260810-185749_rppo_restprem_a01_n106"


def slot_layout(cfg: dict) -> dict:
    """Entity / obstacle / resource slot indices, derived from the run's own config."""
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
    """The alpha kernel the environment convolves the injury buffer with (sensor.py)."""
    n = int(cfg["sensory"]["interoceptive_kernel_length"])
    tau = float(cfg["sensory"]["interoceptive_kernel_tau"])
    k = np.arange(n, dtype=np.float64)
    raw = (k / tau) * np.exp(1.0 - k / tau)
    return raw / raw.sum()


def find_store(run: str, checkpoint=None, root="results/trajectories") -> str:
    tag = os.path.basename(run.rstrip("/"))
    hits = sorted(glob.glob(f"{root}/{tag}/{checkpoint or '*'}/*/"))
    if not hits:
        raise SystemExit(f"no store under {root}/{tag}/{checkpoint or '*'}/")
    if len(hits) > 1 and checkpoint is None:
        raise SystemExit("several checkpoints collected; pass --checkpoint:\n  " + "\n  ".join(hits))
    return hits[0]


def listcol(col, width):
    """Fixed-width parquet list column -> (n, width) array without the slow to_pylist path."""
    ch = col.chunks if hasattr(col, "chunks") else [col]
    return np.concatenate([c.flatten().to_numpy(zero_copy_only=False) for c in ch]).reshape(-1, width)


def episode_table(store, columns):
    """Episode-level columns, sorted by seed."""
    tb = pq.read_table(sorted(glob.glob(store + "episodes_*.parquet")), columns=columns)
    o = np.argsort(tb.column("episode_seed").to_numpy())
    return tb, o


def step_shards(store):
    return sorted(glob.glob(store + "steps_*.parquet"))


def episode_starts(t):
    """Row indices where each episode begins, and a per-row array of its episode's start index.

    Relies on three store properties, asserted by the caller's first shard: rows sorted by seed,
    one t==0 row per episode, and episodes never split across shards.
    """
    st = np.flatnonzero(t == 0)
    ends = np.append(st[1:], len(t))
    return st, ends, np.repeat(st, ends - st)


def reconstruct_nociception(inj, t, kernel):
    """The interoceptive signal the agent actually receives, rebuilt from recorded injury.

    Mirrors src/environment/core.py:115 (buffer of injury LEVELS, rolled each step) and :1112
    (buffer zeroed at reset, so the reset row is NEVER in it) and sensor.py's convolution.
    The reset row must be excluded — including it was a real bug that fabricated a table column.
    """
    n = len(t)
    st, ends, estart = episode_starts(t)
    idx = np.arange(n)
    sig = np.zeros(n)
    for j in range(1, len(kernel)):
        src = idx - j
        ok = src > estart                      # strictly greater: the reset row is not in the buffer
        sig[ok] += kernel[j] * inj[src[ok]]
    return sig


def previous_row(x, t):
    """Value at t-1 within the same episode, zero at the episode's first row.

    The action that produced row t was chosen on the row t-1 observation, so anything the agent
    conditioned on must be taken from the previous row.
    """
    st, ends, estart = episode_starts(t)
    out = np.zeros_like(x, dtype=np.float64)
    out[1:] = x[:-1]
    out[np.arange(len(t)) == estart] = 0.0
    return out


def load_run(run):
    cfg = yaml.safe_load(open(f"{run}/models/config.yaml"))
    return cfg, slot_layout(cfg)


def save(name, payload, quiet=False):
    os.makedirs(OUT_ROOT, exist_ok=True)
    p = f"{OUT_ROOT}/{name}.json"
    json.dump(payload, open(p, "w"), indent=1, default=float)
    if not quiet:
        print(f"\nwritten: {p}")
    return p


def base_args(desc):
    import argparse
    ap = argparse.ArgumentParser(description=desc,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=DEFAULT_RUN)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--store-root", default="results/trajectories")
    ap.add_argument("--quiet", action="store_true")
    return ap
