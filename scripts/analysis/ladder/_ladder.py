"""Shared plumbing for the sensor-ladder analysis (the 14 `lad_*` runs).

The ladder is fourteen agents trained on ONE environment with ONE seed, differing only in what
the agent can sense. Every arm replayed the same 1,000,000 evaluation worlds (`--seed-base
1000000`), so any difference between two arms is attributable to the sensory change and not to
the worlds they happened to meet.

Nothing here computes a result. Each figure script owns exactly one analysis; this module only
holds what they would otherwise duplicate: which runs are in the ladder, what sensory settings
each one carries, and where its trajectory store lives.

Bin edges are FIXED (not per-arm quantiles) so that a bar in arm A means the same thing as the
bar beside it in arm B. Start injury and start nutrition are both drawn uniform on [0, 100]
(`body.random_start_injury: true`), which is what makes the injury analysis causal rather than
merely associational: the wound the agent wakes up with was assigned by the environment, not
earned by its own earlier behaviour.
"""
from __future__ import annotations
import glob, json, os
import numpy as np
import yaml

RUN_ROOT   = "results/JAX_RecurrentPPO"
# The evaluation population was collected in TWO passes and lives in two store roots. This is not
# untidiness: `n_episodes` is a guarded field of a store's manifest, so a store collected for
# a store collected for 300,000 episodes cannot be reopened and extended to 1,000,000 - the collector hard-fails, by
# design, because a past incident contaminated a store that way. The second pass therefore used a
# fresh seed base. Together they cover seeds 1,000,000 .. 1,999,999 with no gap and no overlap,
# identically for all fourteen arms, so the paired design is exactly preserved.
STORE_ROOTS = ["results/trajectories_lad", "results/trajectories_lad2"]
OUT_ROOT   = "results/analysis/ladder"
FIG_ROOT   = "docs/experiments/active/sensor_ladder/figures"

# Ladder order: sensory capability, poorest first. This is the reading order for every figure.
ARM_ORDER = ["A_baseline", "B_olf_only", "R1_range1",
             "V1_blur40", "V2_blur20", "V3_blur10", "V4_blur05", "V5_sharp",
             "P1_blur05_iso", "Q1_presence_sum", "Q2_presence_binary",
             "O1_occl_rock", "O2_occl_veg", "O3_occl_all"]

# Plain-English name for each arm, and the single setting that separates it from its reference.
#
# READ THE ZEROS CAREFULLY. `range: 0` does NOT switch a sense off.
#   olfactory_grid_range 0 -> ONE olfactory sample, taken at the agent's own cell. That sample is
#     still a distance-decayed sum over every animal within sensor_radius (20 cells, i.e. the whole
#     10x10 grid), so the agent smells everything - it just cannot tell which way the smell is
#     coming from. Range 1 adds a 5-cell diamond around the agent, and comparing those five cells
#     is what turns a whiff into a DIRECTION.
#   visual_sensor_range 0 -> the agent sees a 1-cell diamond: the square it is standing on, and
#     nothing else. Range 1 is 5 cells, range 2 is 13 cells.
# (sensor.py: sense_olfaction_cells takes a static single-point branch at range 0; sense_visual
#  uses get_visual_offsets, which returns [[0,0]] at range 0.)
ARM_LABEL = {
    "A_baseline":         ("smell without direction",  "olfactory grid range 0 - one omnidirectional whiff"),
    "B_olf_only":         ("smell gains direction",     "olfactory grid range 1 - a 5-cell smell diamond"),
    "R1_range1":          ("short-range sight",         "visual range 1 (5 cells) instead of 2 (13 cells)"),
    "V1_blur40":          ("very blurred sight",        "visual blur radial scale 4.0"),
    "V2_blur20":          ("blurred sight",             "visual blur radial scale 2.0"),
    "V3_blur10":          ("mildly blurred sight",      "visual blur radial scale 1.0"),
    "V4_blur05":          ("reference agent",           "visual blur radial scale 0.5"),
    "V5_sharp":           ("sharp sight",               "visual blur disabled"),
    "P1_blur05_iso":      ("blur equal in all directions", "visual blur anisotropy 1.0 instead of 3.0"),
    "Q1_presence_sum":    ("sight without identity",    "visual vector size 1 - sees THAT, not WHAT"),
    "Q2_presence_binary": ("sight without identity or count", "visual value mode clamp instead of sum"),
    "O1_occl_rock":       ("rocks block the view",      "visual occlusion on, rocks only"),
    "O2_occl_veg":        ("rocks and bushes block",    "visual occlusion also by bushes"),
    "O3_occl_all":        ("everything blocks",         "visual occlusion also by animals and ambush predators"),
}

# What each arm is a single-variable step away from. None = it is a root of the ladder.
ARM_REFERENCE = {
    "B_olf_only": "A_baseline", "R1_range1": "V5_sharp",
    "V1_blur40": "V4_blur05", "V2_blur20": "V4_blur05", "V3_blur10": "V4_blur05",
    "V5_sharp": "V4_blur05", "P1_blur05_iso": "V4_blur05",
    "Q1_presence_sum": "V4_blur05", "Q2_presence_binary": "Q1_presence_sum",
    "O1_occl_rock": "V4_blur05", "O2_occl_veg": "O1_occl_rock", "O3_occl_all": "O2_occl_veg",
}

INJ_EDGES = np.array([25.0, 50.0, 75.0])          # start injury 0-100, four equal quarters
INJ_NAMES = ["0-25", "25-50", "50-75", "75-100"]
DIST_MAX  = 8                                      # chebyshev distance bins 1..7, then "8+"
DIST_NAMES = ["1", "2", "3", "4", "5", "6", "7", "8+"]
TERM_NAMES = {1: "survived to time limit", 2: "starved", 4: "killed by predator"}


def arm_runs() -> dict[str, str]:
    """Arm name -> run directory, for every `lad_*` run that has a collected store."""
    out = {}
    for d in sorted(glob.glob(f"{RUN_ROOT}/*_lad_*/")):
        arm = os.path.basename(d.rstrip("/")).split("_lad_")[1].rsplit("_n", 1)[0]
        out[arm] = d.rstrip("/")
    missing = [a for a in ARM_ORDER if a not in out]
    if missing:
        raise SystemExit(f"ladder runs not found on disk: {missing}")
    return out


def arm_stores(arm: str) -> list[str]:
    """Every collected store for this arm, in seed order.

    Returns one path per collection pass. Callers must treat them as ONE episode population:
    concatenate, sort by seed, and assert contiguity over the union rather than per store.
    """
    out = []
    for root in STORE_ROOTS:
        hits = sorted(glob.glob(f"{root}/*_lad_{arm}_*/*/*/"))
        if len(hits) > 1:
            raise SystemExit(f"{arm}: {root} holds {len(hits)} stores, expected at most one")
        out.extend(hits)
    if not out:
        raise SystemExit(f"no collected store for {arm} under {STORE_ROOTS}")
    return out


def store_files(stores: list[str], kind: str) -> list[str]:
    """All episode or step shards across the passes. `kind` is 'episodes' or 'steps'.

    Sorted per store and then concatenated, NOT sorted globally: shard names repeat across stores
    (both begin at 00000), so a global sort would interleave the two passes and break the
    one-t=0-row-per-episode ordering every sweep relies on.
    """
    out = []
    for st in stores:
        out.extend(sorted(glob.glob(st + f"{kind}_*.parquet")))
    return out


def arm_store(arm: str) -> str:
    raise SystemExit("arm_store() is gone - the population spans several stores; use arm_stores()")


def arm_config(run: str) -> dict:
    return yaml.safe_load(open(f"{run}/models/config.yaml"))


def sensory_summary(cfg: dict) -> dict:
    """The seven sensory knobs the ladder varies, pulled straight from the run's saved config."""
    s = cfg["sensory"]
    return {
        "olfactory_grid_range":   s["olfactory_grid_range"],
        "visual_sensor_range":    s["visual_sensor_range"],
        "visual_blur_enabled":    s["visual_blur_enabled"],
        "visual_blur_radial_scale": s["visual_blur_radial_scale"] if s["visual_blur_enabled"] else None,
        "visual_blur_anisotropy":   s["visual_blur_anisotropy"] if s["visual_blur_enabled"] else None,
        "visual_vector_size":     s["visual_vector_size"],
        "visual_value_mode":      s["visual_value_mode"],
        "visual_occlusion_enabled": s["visual_occlusion_enabled"],
    }


def slot_layout(cfg: dict) -> dict:
    """Which entity / obstacle / resource slots are what, derived from the run's own config."""
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
    return dict(pred=pred, neutral=neu, bush=bush, rock=rock, food=food, ambush=amb)


def load_arm(arm: str) -> dict:
    p = f"{OUT_ROOT}/{arm}.json"
    if not os.path.exists(p):
        raise SystemExit(f"{p} missing - run scripts/analysis/ladder/build_arm_data.py first")
    return json.load(open(p))


def load_all() -> dict[str, dict]:
    return {a: load_arm(a) for a in ARM_ORDER}


def rate(bush, tot):
    """Percentage, NaN where the cell has too little support to mean anything."""
    b, t = np.asarray(bush, float), np.asarray(tot, float)
    return np.where(t >= 1000, 100.0 * b / np.maximum(t, 1), np.nan)


def save_json(name, payload):
    os.makedirs(OUT_ROOT, exist_ok=True)
    p = f"{OUT_ROOT}/{name}.json"
    json.dump(payload, open(p, "w"), indent=1, default=float)
    return p


NEAR_BINS = (0, 1)          # nearest animal 1-2 cells away
FAR_BINS  = (5, 6, 7)       # nearest animal 6 or more cells away


def proximity_effect(bush, tot, inj_bins=None):
    """How much more often the agent is in a bush when an animal is close than when it is far.

    Returned in percentage points. `bush` / `tot` are the (distance x injury) grids written by
    build_arm_data.py. `inj_bins` restricts to some starting-wound quarters; None pools all four.
    A pooled contrast is NOT the same as an average of the four - the quarters carry different
    step counts - so this pools counts, not rates.
    """
    b, t = np.asarray(bush, float), np.asarray(tot, float)
    cols = slice(None) if inj_bins is None else list(inj_bins)
    nb, nt = b[list(NEAR_BINS)][:, cols].sum(), t[list(NEAR_BINS)][:, cols].sum()
    fb, ft = b[list(FAR_BINS)][:, cols].sum(), t[list(FAR_BINS)][:, cols].sum()
    if nt < 1000 or ft < 1000:
        return np.nan
    return 100.0 * (nb / nt - fb / ft)


def dist_curve(bush, tot, inj_bins=None):
    """Bush occupancy (%) at each nearest-animal distance, pooled over the chosen wound quarters."""
    b, t = np.asarray(bush, float), np.asarray(tot, float)
    cols = slice(None) if inj_bins is None else list(inj_bins)
    bb, tt = b[:, cols].sum(1), t[:, cols].sum(1)
    return np.where(tt >= 1000, 100.0 * bb / np.maximum(tt, 1), np.nan)


def settings(cfg) -> dict:
    """The sensory knobs, collapsed so that each key is one thing a person could decide to change.

    Written because a bug hid here. `ARM_REFERENCE` claimed every pair differs in exactly one
    setting, and `R1_range1` was paired against `V4_blur05` when it differs in TWO - it drops the
    visual range from 2 to 1 AND has blur switched off - so the survival difference attributed to
    "visual range" was actually range plus blur. Comparing raw config keys does not catch that
    cleanly either, because turning blur off leaves `visual_blur_radial_scale` in the file as an
    inert value that reads as a third difference. So the two conditional groups are collapsed:

      blur      -> "off", or the (radial scale, anisotropy) pair that is live when it is on
      occlusion -> "off", or the set of things configured to block sight when it is on

    With that, `setting_diff` returns exactly the changes a reader would name out loud, and
    `check_single_variable_pairs` can assert the claim the figures make.
    """
    v = cfg["sensory"]
    return {
        "olfactory_grid_range": v["olfactory_grid_range"],
        "visual_sensor_range": v["visual_sensor_range"],
        "visual_vector_size": v["visual_vector_size"],
        "visual_value_mode": v["visual_value_mode"],
        "blur": ((v["visual_blur_radial_scale"], v["visual_blur_anisotropy"])
                 if v["visual_blur_enabled"] else "off"),
        "occlusion": (tuple(blocks_sight(cfg)) if v["visual_occlusion_enabled"] else "off"),
    }


def setting_diff(cfg_a, cfg_b) -> list[str]:
    a, b = settings(cfg_a), settings(cfg_b)
    return [k for k in a if a[k] != b[k]]


def check_single_variable_pairs(configs: dict):
    """Assert that every (arm, reference) pair really does differ in exactly one setting.

    Called by the figure that plots those differences. A pair that drifts to two settings makes
    that figure attribute a survival change to the wrong cause, silently.
    """
    bad = {a: setting_diff(configs[a], configs[r])
           for a, r in ARM_REFERENCE.items()
           if len(setting_diff(configs[a], configs[r])) != 1}
    if bad:
        raise SystemExit("ARM_REFERENCE pairs that are not single-variable:\n  " +
                         "\n  ".join(f"{a} vs {ARM_REFERENCE[a]}: differs in {d}"
                                     for a, d in bad.items()))


def blocks_sight(cfg) -> list[str]:
    """Which kinds of object were configured to block the agent's line of sight.

    The three occlusion arms carry IDENTICAL `sensory` blocks - cone 15 degrees, strength 1.0 in
    all three. What separates them is a per-object `blocks_sight` flag on the entities, obstacles
    and resources, so it has to be read from those lists rather than from the sensory block.
    """
    env = cfg["environment"]
    out = []
    for section in ("obstacles", "entities", "resources"):
        for o in env.get(section, []):
            if o.get("blocks_sight"):
                out.append(o.get("name") or o.get("class") or f"<unnamed {section[:-1]}>")
    return out


def resolves_identity(sensory: dict) -> bool:
    """Can this arm's sight tell a predator from a rabbit at a distance?

    Two conditions: a visual field bigger than the agent's own cell and its immediate ring
    (range >= 2, i.e. thirteen cells), AND more than one appearance channel, so the field carries
    WHAT is in a cell rather than only THAT something is. Stated here once because three figures
    group the arms by it and they must group them identically.

    This rule is POST-HOC - it was written after seeing which arms separated, not before - so it is
    a description of the split rather than a prediction of it. The ladder also contains no arm that
    varies range and channel count independently at range 1, so the two conditions are not fully
    disentangled by this data.
    """
    return sensory["visual_sensor_range"] >= 2 and sensory["visual_vector_size"] > 1


GROUP_LABEL = {True: "sight resolves WHAT it sees  (range 2, 8 appearance channels)",
               False: "sight cannot resolve WHAT it sees  (range < 2, or 1 channel)"}


def load_time_course() -> dict:
    """The per-arm step-by-step sweeps written by build_time_course.py.

    Asserts that every arm's file was produced by the SAME episode population as the aggregates -
    a stale file left over from a smaller collection is exactly the failure this layout replaced.
    """
    import json
    out = {}
    for arm in ARM_ORDER:
        p = f"{OUT_ROOT}/time_course_{arm}.json"
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing - run scripts/analysis/ladder/build_time_course.py")
        out[arm] = json.load(open(p))
    return out
