#!/usr/bin/env python
"""Does the agent's behaviour depend on its INTERNAL STATE, and does a neuromodulator
strengthen that dependence?

Why this exists
---------------
A FiLM neuromodulator reads the agent's internal state (injury, nutrition, ...) and rescales its
perceptual features. The sharp prediction is therefore NOT "the modulated agent behaves
differently" — it is that the modulated agent's **response to external cues should vary more with its
internal state**. That is an interaction, and it is what this tool measures.

B0 — bush-entry rate (the headline behavioural measure)
-------------------------------------------------------
`B0 = P(in a bush at t+1 | not in a bush at t, and a MOVEMENT action was chosen at t)`.

Every measure built on *where the agent is* inherits a known artefact: a wounded agent mostly
freezes to heal, and whether that registers as "hides more" depends only on whether it happened to
be standing on a bush when it stopped. B0 escapes that by scoring a **movement decision** instead of
a location — an agent that freezes leaves the DENOMINATOR rather than loading the numerator, because
both `Rest` and `Eat` are excluded.

Row alignment is the footgun. Row `t` carries the environment state AT `t` together with the action
that ARRIVED at `t` (see docs/environment/TRAJECTORY_STORE_SCHEMA.md §1), so an entry event is

    agent_in_bush[t]   == False
    action[t+1]        in {0,1,2,3}
    agent_in_bush[t+1] == True

i.e. the deciding row is `t` but the action and the outcome are both read off row `t+1`. Shifting
that by one silently measures something else. `tests/scripts/test_context_dependence_b0.py` pins it.

Binned by the episode's RANDOMISED STARTING injury (the value at `t = 0`, drawn before the agent
acted — the only causally identified internal-state variable available) using FIXED edges, never
quantiles: sample-defined bins would sit at different injury values in each run and destroy the
cross-run comparability the measure exists for.

`Δ_B0` = B0(top populated injury bin) − B0(bottom populated injury bin), in percentage points.
Its predicted sign is **positive** (more cover-seeking when wounded). A large NEGATIVE Δ_B0 is not a
confirmation; it is state dependence in the direction opposite to the project's target behaviour.

Reported alongside: B1 (rest rate per injury bin — what B0's denominator removes), B3
(`B0(predator near) − B0(predator far)` within each bin, and its range across bins), and a
robustness variant of B0 whose denominator is a REALISED DISPLACEMENT (the agent's cell actually
changed) rather than a chosen move, which drops moves a rock blocked.

The three older occupancy measures, weakest evidence to strongest
-----------------------------------------------------------------
1. `state_span`   — how much hiding changes across internal-state cells, with no predator nearby.
                    Pure internal-state dependence of baseline behaviour.
2. `proximity_effect_range`    — the PROX EFFECT in each internal-state cell, defined as
                    P(hide | predator near) - P(hide | no predator near) within that cell, and how
                    far that effect ranges across cells. This is the direct analogue of what a
                    modulator is supposed to do: change the response to a percept according to state.
3. `proximity_effect_range` (randomised block)  — the same effect, but computed early in the episode and binned by the RANDOMISED
                    starting injury / nutrition the environment assigns before the agent acts.
                    Only this one is causally identified; 1 and 2 condition on state the agent
                    itself produced.

Internal state can be read either as the raw body variable (`--state injury`, `--state nutrition`)
or as the signal the agent actually receives (`--state felt_pain`), which for injury is a lagged,
smoothed convolution rather than the level itself. `felt_pain` is usually the right choice for
injury: the agent has no direct readout of its wound.

Usage
-----
  P=/home/vncuser/miniconda3/envs/grid_world_pain/bin/python
  # one run
  $P scripts/analysis/context_dependence.py --run results/JAX_RecurrentPPO/<RUN> \
      --store-root results/trajectories_nmn --checkpoint <STEP>
  # compare two (e.g. modulated vs not) — pass --label to name them
  $P scripts/analysis/context_dependence.py --compare A.json B.json

Run from the repo root. Re-run this against new neuromodulator checkpoints as they land; the
metrics are defined so they are comparable across runs and across worlds.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np, pyarrow.parquet as pq, yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hiding_drivers import slot_layout, find_stores, shard_files, listcol   # noqa: E402

MOVE_ACTIONS = (0, 1, 2, 3)   # up/down/left/right; `Rest` is 4 and `Eat` 5 (core.py:580,661)

INJ_EDGES  = [1e-9, 25.0, 50.0]
NUT_EDGES  = [25.0, 50.0, 75.0]
PAIN_EDGES = [1e-9, 8.0, 18.0, 32.0]
NEAR_D     = 2
EARLY      = 25          # steps counted as "early" for the randomised-state measure
LABELS = {"injury":    ["injury 0", "0-25", "25-50", ">=50"],
          "nutrition": ["nutr <25", "25-50", "50-75", ">=75"],
          "felt_pain": ["felt 0", "0-8", "8-18", "18-32", "32+"]}


def _episode_side(stores, lay):
    """Shared episode-table read: sorted seeds + the per-episode animal-active mask."""
    ep = pq.read_table(shard_files(stores, "episodes"),
                       columns=["episode_seed", "animal_active"])
    seeds = ep.column("episode_seed").to_numpy()
    o = np.argsort(seeds)
    seed = seeds[o]
    if seed.max() - seed.min() + 1 != len(seed):
        raise SystemExit("episode seeds are not contiguous; this reader assumes they are")
    return int(seed[0]), np.array(ep.column("animal_active").to_pylist(), bool)[o]


def sweep(stores, lay, state, kernel, exclude_rest=False, verbose=True):
    """One pass. Returns counts[state_bin, predator_near, in_bush] and the early/randomised
    version binned by the STARTING value of the same body variable.

    `exclude_rest` drops every row whose ARRIVING action was `Rest`, which turns the occupancy
    measure into B2 ("bush occupancy conditioned on acting"). It is a descriptive companion, not
    an escape from the freeze-to-heal artefact: conditioning on "not resting" does not remove it,
    because injury and position both plausibly cause whether the agent rests. B0 is the measure
    that escapes it."""
    na = len(lay["pred"]) + len(lay["neutral"])
    seed0, act = _episode_side(stores, lay)
    P = lay["pred"]
    nb = len(LABELS[state])
    C = np.zeros((nb, 2, 2)); E = np.zeros((nb, 2, 2))
    body_col = "nutrition" if state == "nutrition" else "injury_level"
    edges = {"injury": INJ_EDGES, "nutrition": NUT_EDGES, "felt_pain": PAIN_EDGES}[state]
    files = shard_files(stores, "steps"); t0 = time.time()
    cols = ["episode_seed", "t", "agent_in_bush", body_col,
            "agent_row", "agent_col", "animal_row", "animal_col"]
    if exclude_rest:
        cols.append("rested")
    for fi, f in enumerate(files):
        tb = pq.read_table(f, columns=cols)
        sd = tb.column("episode_seed").to_numpy(); t = tb.column("t").to_numpy(); N = len(t)
        body = tb.column(body_col).to_numpy(zero_copy_only=False).astype(np.float64)
        bu = tb.column("agent_in_bush").to_numpy(zero_copy_only=False).astype(np.int64)
        ar = tb.column("agent_row").to_numpy(); ac = tb.column("agent_col").to_numpy()
        st = np.flatnonzero(t == 0); ends = np.append(st[1:], N)
        estart = np.repeat(st, ends - st); idx = np.arange(N)

        if state == "felt_pain":
            # reconstruct what the agent receives: buffer of injury LEVELS, zeroed at reset, so
            # the reset row is never in it (core.py:1112 / :115), convolved with the alpha kernel
            sig = np.zeros(N)
            for j in range(1, len(kernel)):
                src = idx - j; ok = src > estart
                sig[ok] += kernel[j] * body[src[ok]]
        else:
            sig = body
        # the action producing row t was chosen on the row t-1 observation
        prev = np.zeros(N); prev[1:] = sig[:-1]; prev[idx == estart] = sig[estart[0]] * 0
        start_val = np.repeat(body[st], ends - st)

        AR = listcol(tb.column("animal_row"), na); AC = listcol(tb.column("animal_col"), na)
        near = (np.maximum(np.abs(AR - ar[:, None]), np.abs(AC - ac[:, None])) <= NEAR_D) \
               & act[sd - seed0]
        pn = near[:, P].any(1).astype(np.int64)

        m = t >= 2
        if exclude_rest:
            m = m & ~tb.column("rested").to_numpy(zero_copy_only=False)
        b = np.digitize(prev[m], edges)
        k = (b * 2 + pn[m]) * 2 + bu[m]
        C += np.bincount(k, minlength=nb * 4).reshape(nb, 2, 2)
        me = m & (t <= EARLY)
        be = np.digitize(start_val[me], edges)
        ke = (be * 2 + pn[me]) * 2 + bu[me]
        E += np.bincount(ke, minlength=nb * 4).reshape(nb, 2, 2)
        if verbose and fi % 50 == 0:
            print(f"  shard {fi}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)
    return C, E


ENTRY_LABELS = LABELS["injury"]          # B0 bins by the RANDOMISED STARTING injury
ENTRY_MIN_N  = 2000                      # min qualifying denominator steps for a bin to be used
WINDOWS      = (f"first_{EARLY}", "whole_episode")
VARIANTS     = ("action", "displacement")


def sweep_entry(stores, lay, verbose=True):
    """B0/B1 pass — bush ENTRY out of the open, per starting-injury bin and predator condition.

    Returns
    -------
    ENT  : (2 windows, 2 variants, 4 injury bins, 2 predator-near, 2 entered) counts.
           Axis -1 is the outcome `agent_in_bush[t+1]`; ENT[..., 1] is the numerator and
           ENT[...].sum(-1) the denominator.
    REST : (2 windows, 4 injury bins, 2) counts of chosen actions, [..., 1] = `Rest` (B1).
    diag : index-convention checks, so the hard-coded action numbering is verified per run.

    THE ROW ALIGNMENT (docs/environment/TRAJECTORY_STORE_SCHEMA.md §1): row `t` holds the state AT
    `t` and the action that ARRIVED at `t`. The deciding row is therefore `t`, while BOTH the
    action and the outcome are read from row `t+1`. Successor validity is checked with
    `t[i+1] == t[i] + 1`, which is false at the last row of an episode and at a shard boundary, so
    an episode's final row can never pair with the next episode's spawn row."""
    na = len(lay["pred"]) + len(lay["neutral"])
    seed0, act = _episode_side(stores, lay)
    P = lay["pred"]
    nb = len(ENTRY_LABELS)
    ENT = np.zeros((2, 2, nb, 2, 2)); REST = np.zeros((2, nb, 2))
    diag = {"rest_action_values": set(), "action_values": set(),
            "displaced_without_move_action": 0, "n_rows": 0, "n_episodes": 0}
    files = shard_files(stores, "steps"); t0 = time.time()
    cols = ["episode_seed", "t", "action", "rested", "agent_in_bush", "injury_level",
            "agent_row", "agent_col", "animal_row", "animal_col"]
    for fi, f in enumerate(files):
        tb = pq.read_table(f, columns=cols)
        sd = tb.column("episode_seed").to_numpy(); t = tb.column("t").to_numpy(); N = len(t)
        a  = tb.column("action").to_numpy(zero_copy_only=False).astype(np.int64)
        rested = tb.column("rested").to_numpy(zero_copy_only=False)
        bu = tb.column("agent_in_bush").to_numpy(zero_copy_only=False)
        inj = tb.column("injury_level").to_numpy(zero_copy_only=False).astype(np.float64)
        ar = tb.column("agent_row").to_numpy(); ac = tb.column("agent_col").to_numpy()
        st = np.flatnonzero(t == 0); ends = np.append(st[1:], N)
        start_inj = np.repeat(inj[st], ends - st)
        diag["n_rows"] += N; diag["n_episodes"] += len(st)
        diag["action_values"] |= set(np.unique(a).tolist())
        diag["rest_action_values"] |= set(np.unique(a[rested]).tolist())

        # --- row t+1, only where it exists and belongs to the same episode -------------
        nxt = np.zeros(N, bool); nxt[:-1] = t[1:] == t[:-1] + 1
        sh = lambda v: np.concatenate([v[1:], v[:1]])      # value at t+1 (garbage where ~nxt)
        move_next = np.isin(sh(a), MOVE_ACTIONS)
        bush_next = sh(bu)
        disp_next = (sh(ar) != ar) | (sh(ac) != ac)

        AR = listcol(tb.column("animal_row"), na); AC = listcol(tb.column("animal_col"), na)
        near = (np.maximum(np.abs(AR - ar[:, None]), np.abs(AC - ac[:, None])) <= NEAR_D) \
               & act[sd - seed0]
        pn = near[:, P].any(1).astype(np.int64)
        b = np.digitize(start_inj, INJ_EDGES)
        ent = bush_next.astype(np.int64)
        diag["displaced_without_move_action"] += int((nxt & disp_next & ~move_next).sum())

        open_now = nxt & ~bu
        den = {"action": open_now & move_next, "displacement": open_now & disp_next}
        for wi, wm in enumerate((t <= EARLY, np.ones(N, bool))):
            for vi, vn in enumerate(VARIANTS):
                m = den[vn] & wm
                if m.any():
                    k = (b[m] * 2 + pn[m]) * 2 + ent[m]
                    ENT[wi, vi] += np.bincount(k, minlength=nb * 4).reshape(nb, 2, 2)
            # B1 — rest rate over rows carrying a chosen action (t >= 1, i.e. action >= 0)
            mr = wm & (a >= 0)
            if mr.any():
                k = b[mr] * 2 + rested[mr].astype(np.int64)
                REST[wi] += np.bincount(k, minlength=nb * 2).reshape(nb, 2)
        if verbose and fi % 50 == 0:
            print(f"  [B0] shard {fi}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)

    bad = diag["rest_action_values"] & set(MOVE_ACTIONS)
    if bad:
        raise SystemExit(f"action-index convention violated: `rested` is True on action(s) {bad}, "
                         f"which this script counts as movement. Re-derive MOVE_ACTIONS.")
    for k in ("rest_action_values", "action_values"):
        diag[k] = sorted(diag[k])
    return ENT, REST, diag


def _prop_ci(x, n):
    """Rate in percentage points and its 95% Wald half-width, or NaN if the cell is empty."""
    if n <= 0:
        return np.nan, np.nan
    p = x / n
    return 100 * p, 100 * 1.96 * np.sqrt(max(p * (1 - p), 0.0) / n)


def entry_metrics(ENT, min_n=ENTRY_MIN_N):
    """B0 per (window, variant, predator condition, injury bin), plus Δ_B0 and B3.

    `cond` is 'near' (a predator within Chebyshev NEAR_D), 'far' (none), or 'any' (pooled).
    Δ_B0 takes the TOP minus the BOTTOM *populated* bin, where populated means at least `min_n`
    qualifying denominator steps; when a bin is too small the next bin inward is used and the
    substitution is recorded in `bins_used`."""
    out = {}
    for wi, w in enumerate(WINDOWS):
        for vi, v in enumerate(VARIANTS):
            A = ENT[wi, vi]                                   # (bin, near, entered)
            slices = {"far": A[:, 0, :], "near": A[:, 1, :], "any": A.sum(1)}
            for cond, M in slices.items():
                rows = []
                for i, lab in enumerate(ENTRY_LABELS):
                    n = float(M[i].sum()); x = float(M[i, 1])
                    rate, hw = _prop_ci(x, n) if n >= min_n else (np.nan, np.nan)
                    rows.append(dict(bin=lab, n=n, entries=x, b0=rate, b0_ci95=hw))
                ok = [i for i, r in enumerate(rows) if np.isfinite(r["b0"])]
                if len(ok) >= 2:
                    lo, hi = ok[0], ok[-1]
                    p1, n1 = rows[hi]["b0"] / 100, rows[hi]["n"]
                    p0, n0 = rows[lo]["b0"] / 100, rows[lo]["n"]
                    d = 100 * (p1 - p0)
                    ci = 100 * 1.96 * np.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
                    used = [rows[lo]["bin"], rows[hi]["bin"]]
                else:
                    d = ci = np.nan; used = [r["bin"] for r in rows if np.isfinite(r["b0"])]
                out[f"{w}|{v}|{cond}"] = dict(
                    window=w, variant=v, predator=cond, rows=rows,
                    delta_b0=d, delta_b0_ci95=ci, bins_used=used,
                    substituted=bool(len(ok) >= 2 and (ok[0] != 0 or ok[-1] != len(rows) - 1)))
            # B3 — threat response within each bin, and its range across bins
            b3 = []
            for i, lab in enumerate(ENTRY_LABELS):
                nn, nf = float(A[i, 1].sum()), float(A[i, 0].sum())
                if nn >= min_n and nf >= min_n:
                    b3.append(dict(bin=lab, threat_response=100 * (A[i, 1, 1] / nn - A[i, 0, 1] / nf)))
            rng = (max(r["threat_response"] for r in b3) - min(r["threat_response"] for r in b3)) \
                  if len(b3) > 1 else np.nan
            out[f"{w}|{v}|B3"] = dict(window=w, variant=v, rows=b3, threat_response_range=rng)
    return out


def entry_rest_metrics(REST, min_n=ENTRY_MIN_N):
    """B1 — fraction of acting steps on which the agent chose `Rest`, per starting-injury bin.

    Δ_rest takes the TOP minus the BOTTOM *populated* bin, where populated means at least `min_n`
    rows — the same guard B0 uses, and for the same reason. Without it this took the first bin that
    happened to be finite, which in this environment is `injury 0`: the starting wound is drawn from
    a continuous range, so exactly-zero is a measure-zero event and that bin held **25 rows out of
    five million**. Every Δ_rest computed before this guard was a difference against a 25-sample
    estimate, and the resulting spread across runs (8.6 to 54.5 pp) was almost entirely that noise.
    With the guard the same runs span 17.2 to 28.6.
    """
    out = {}
    for wi, w in enumerate(WINDOWS):
        rows = []
        for i, lab in enumerate(ENTRY_LABELS):
            n = float(REST[wi, i].sum()); x = float(REST[wi, i, 1])
            rate, _ = _prop_ci(x, n) if n >= min_n else (np.nan, np.nan)
            rows.append(dict(bin=lab, n=n, rest_rate=rate))
        fin = [r for r in rows if np.isfinite(r["rest_rate"])]
        out[w] = dict(rows=rows, min_n=min_n,
                      bins_used=[fin[0]["bin"], fin[-1]["bin"]] if len(fin) > 1 else [],
                      delta_rest=(fin[-1]["rest_rate"] - fin[0]["rest_rate"]) if len(fin) > 1
                                 else np.nan)
    return out


def show_entry(E, R, diag, title):
    print(f"\n=== {title} — B0 bush-entry rate ===")
    print(f"  action values seen {diag['action_values']}   `rested` on action(s) "
          f"{diag['rest_action_values']}  (must be disjoint from {list(MOVE_ACTIONS)})")
    print(f"  {diag['n_episodes']:,} episodes / {diag['n_rows']:,} rows;  displaced without a "
          f"movement action: {diag['displaced_without_move_action']:,}")
    for w in WINDOWS:
        for v in VARIANTS:
            print(f"\n-- window {w}   denominator: {v} --")
            print(f"{'start injury':14}{'B0 far':>12}{'B0 near':>12}{'B3 near-far':>13}"
                  f"{'n far':>12}{'n near':>12}")
            far = E[f"{w}|{v}|far"]["rows"]; nr = E[f"{w}|{v}|near"]["rows"]
            for a, b in zip(far, nr):
                f_ = f"{a['b0']:.2f}%" if np.isfinite(a["b0"]) else "—"
                n_ = f"{b['b0']:.2f}%" if np.isfinite(b["b0"]) else "—"
                d_ = f"{b['b0']-a['b0']:+.2f}" if np.isfinite(a["b0"]) and np.isfinite(b["b0"]) else "—"
                print(f"{a['bin']:14}{f_:>12}{n_:>12}{d_:>13}{a['n']:>12,.0f}{b['n']:>12,.0f}")
            for cond in ("any", "far", "near"):
                m = E[f"{w}|{v}|{cond}"]
                sub = "  (bins substituted)" if m["substituted"] else ""
                print(f"  Δ_B0 [{cond:>4}] = {m['delta_b0']:+.2f} pp  ±{m['delta_b0_ci95']:.2f} "
                      f"(within-run 95% CI)   {m['bins_used']}{sub}")
            print(f"  B3 range across bins: {E[f'{w}|{v}|B3']['threat_response_range']:.2f} pp")
    print("\n-- B1 rest rate (descriptive; this is what B0's denominator removes) --")
    for w in WINDOWS:
        cells = "  ".join(f"{r['bin']}: {r['rest_rate']:.1f}%" for r in R[w]["rows"]
                          if np.isfinite(r["rest_rate"]))
        print(f"  {w:16} {cells}   Δ = {R[w]['delta_rest']:+.1f} pp")
    print("\nReading it: Δ_B0's PRE-REGISTERED direction is POSITIVE (more cover-seeking when")
    print("wounded). A large negative Δ_B0 is state dependence in the direction OPPOSITE to the")
    print("project's target behaviour, not a confirmation.")


def metrics(C, labels, min_n=2e4):
    """P(hide) with and without a predator near, per internal-state cell, plus the spans."""
    rows = []
    for i, lab in enumerate(labels):
        n0, n1 = C[i, 0].sum(), C[i, 1].sum()
        p0 = C[i, 0, 1] / n0 if n0 >= min_n else np.nan
        p1 = C[i, 1, 1] / n1 if n1 >= min_n else np.nan
        rows.append(dict(state=lab, n_calm=float(n0), n_threat=float(n1),
                         hide_calm=100 * p0, hide_threat=100 * p1,
                         proximity_effect=100 * (p1 - p0)))
    ok = [r for r in rows if np.isfinite(r["proximity_effect"])]
    span = lambda key, rs: (max(r[key] for r in rs) - min(r[key] for r in rs)) if len(rs) > 1 else np.nan
    # The zero bin ("no signal at all") is qualitatively unlike the graded bins and dominates the
    # step count, so it distorts a max-minus-min span. Report the span over the SIGNAL-carrying
    # bins separately, and a trend, which is the cleanest single number: does the predator-proximity effect rise
    # or fall as internal state increases, and how steeply per bin. Weighted by cell size.
    sig = [r for r in ok if not r["state"].endswith(" 0")]
    if len(sig) > 1:
        x = np.arange(len(sig), dtype=float)
        y = np.array([r["proximity_effect"] for r in sig])
        w = np.array([r["n_calm"] + r["n_threat"] for r in sig])
        xm = np.average(x, weights=w); ym = np.average(y, weights=w)
        trend = float(np.sum(w * (x - xm) * (y - ym)) / max(np.sum(w * (x - xm) ** 2), 1e-9))
    else:
        trend = np.nan
    return rows, {"state_span": span("hide_calm", ok), "proximity_effect_range": span("proximity_effect", ok),
                  "proximity_effect_range_signal": span("proximity_effect", sig),
                  "proximity_effect_trend_per_bin": trend,
                  "pe_min": min((r["proximity_effect"] for r in ok), default=np.nan),
                  "pe_max": max((r["proximity_effect"] for r in ok), default=np.nan),
                  "cells_used": len(ok)}


def show(rows, m, title):
    print(f"\n=== {title} ===")
    print(f"{'internal state':16}{'no predator':>13}{'predator near':>15}{'PROX EFFECT':>14}"
          f"{'n (M)':>12}")
    for r in rows:
        if not np.isfinite(r["proximity_effect"]):
            print(f"{r['state']:16}{'— too few steps —':>54}"); continue
        print(f"{r['state']:16}{r['hide_calm']:>12.1f}%{r['hide_threat']:>14.1f}%"
              f"{r['proximity_effect']:>+13.1f}{(r['n_calm']+r['n_threat'])/1e6:>12.1f}")
    print(f"  state span (baseline hiding across states) : {m['state_span']:.1f} pp")
    print(f"  RANGE  (all states)                    : {m['proximity_effect_range']:.1f} pp"
          f"   [{m['pe_min']:.1f} .. {m['pe_max']:.1f}]")
    print(f"  RANGE  (signal-carrying states only)   : {m['proximity_effect_range_signal']:.1f} pp")
    print(f"  TREND (pp per state bin, weighted)    : {m['proximity_effect_trend_per_bin']:+.2f}"
          f"   <- positive = more internal signal, stronger threat response")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run"); ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--store-root", nargs="+", default=["results/trajectories"],
                    help="one or more store roots. A run's evaluation population may be "
                         "split across several, because `n_episodes` is a guarded manifest "
                         "field: a store collected for N episodes cannot be reopened and "
                         "extended, so a top-up goes to a fresh root with a continuing seed "
                         "base. find_stores() already treats the union as ONE population; "
                         "only this flag was single-valued.")
    ap.add_argument("--state", default="felt_pain", choices=["injury", "nutrition", "felt_pain"])
    ap.add_argument("--measures", default="all", choices=["all", "occupancy", "entry"],
                    help="'entry' = B0/B1/B3 only (one pass); 'occupancy' = the older "
                         "P(in a bush) measures only; 'all' = both (two passes)")
    ap.add_argument("--exclude-rest", action="store_true",
                    help="B2: drop rows whose arriving action was `Rest` from the OCCUPANCY "
                         "measures. Descriptive only — it does not remove the freeze-to-heal "
                         "artefact, which is what B0 exists for.")
    ap.add_argument("--label", default=None, help="name for this run in the output")
    ap.add_argument("--out", default=None, help="write metrics JSON here")
    ap.add_argument("--compare", nargs=2, metavar=("A.json", "B.json"),
                    help="compare two previously written metric files")
    a = ap.parse_args()

    if a.compare:
        A = json.load(open(a.compare[0])); B = json.load(open(a.compare[1]))
        assert A["state"] == B["state"], "the two runs used different --state"
        print(f"### context dependence on {A['state']}: {A['label']}  vs  {B['label']}\n")
        for tag in ("observed", "randomised_early"):
            if tag not in A or tag not in B:
                print(f"--- {tag}: absent (run with --measures all/occupancy) ---"); continue
            ra, rb = A[tag]["rows"], B[tag]["rows"]
            ma, mb = A[tag]["metrics"], B[tag]["metrics"]
            print(f"--- {tag} ---")
            print(f"{'internal state':16}{A['label'][:11]:>13}{B['label'][:11]:>13}"
                  f"{'effect A':>10}{'effect B':>10}{'Δ':>9}")
            for x, y in zip(ra, rb):
                if not (np.isfinite(x["proximity_effect"]) and np.isfinite(y["proximity_effect"])): continue
                print(f"{x['state']:16}{x['hide_calm']:>12.1f}%{y['hide_calm']:>12.1f}%"
                      f"{x['proximity_effect']:>+10.1f}{y['proximity_effect']:>+10.1f}"
                      f"{y['proximity_effect']-x['proximity_effect']:>+9.1f}")
            for key, nm in (("proximity_effect_range", "RANGE (all)"),
                            ("proximity_effect_range_signal", "RANGE (signal)"),
                            ("proximity_effect_trend_per_bin", "TREND /bin")):
                print(f"  {nm:22} {A['label']}: {ma[key]:+.2f}    "
                      f"{B['label']}: {mb[key]:+.2f}    diff {mb[key]-ma[key]:+.2f}")
            print(f"  state span  {A['label']}: {ma['state_span']:.1f} pp    "
                  f"{B['label']}: {mb['state_span']:.1f} pp    "
                  f"difference {mb['state_span']-ma['state_span']:+.1f} pp\n")
        print("Reading it: a modulator that gates perception by internal state should show a")
        print("LARGER range — its response to a nearby predator should depend more on how")
        print("hurt or hungry it is. The randomised_early block is the only causally identified")
        print("half; the observed block conditions on state the agent produced itself.")
        return

    if not a.run:
        ap.error("--run is required unless --compare is used")
    cfg = yaml.safe_load(open(f"{a.run}/models/config.yaml"))
    lay = slot_layout(cfg)
    KL = int(cfg["sensory"]["interoceptive_kernel_length"])
    TAU = float(cfg["sensory"]["interoceptive_kernel_tau"])
    k = np.arange(KL, dtype=np.float64); raw = (k / TAU) * np.exp(1.0 - k / TAU)
    kernel = raw / raw.sum()
    stores = find_stores(a.run, a.checkpoint, a.store_root)
    label = a.label or os.path.basename(a.run.rstrip("/"))[16:]
    print(f"run   {a.run}\nstore {' '.join(stores)}\nstate {a.state}   label {label}")

    out = {"label": label, "run": a.run, "state": a.state, "store": stores,
           "checkpoint": a.checkpoint, "exclude_rest": bool(a.exclude_rest)}

    if a.measures in ("all", "entry"):
        ENT, REST, diag = sweep_entry(stores, lay)
        Em, Rm = entry_metrics(ENT), entry_rest_metrics(REST)
        show_entry(Em, Rm, diag, label)
        out["entry"] = {"b0": Em, "b1": Rm, "diagnostics": diag,
                        "min_n": ENTRY_MIN_N, "early_window": EARLY,
                        "injury_edges": INJ_EDGES, "move_actions": list(MOVE_ACTIONS)}

    if a.measures in ("all", "occupancy"):
        if a.state == "felt_pain":
            print(f"kernel tau={TAU} len={KL}  (weights lag1..lag{KL-1}, current step excluded)")
        C, E = sweep(stores, lay, a.state, kernel, exclude_rest=a.exclude_rest)
        labs = LABELS[a.state]
        rows_o, m_o = metrics(C, labs)
        rows_e, m_e = metrics(E, labs)
        tag = " [Rest steps excluded — B2]" if a.exclude_rest else ""
        show(rows_o, m_o, f"{label} — observed internal state (associational){tag}")
        show(rows_e, m_e, f"{label} — RANDOMISED starting state, first {EARLY} steps (causal){tag}")
        out["observed"] = {"rows": rows_o, "metrics": m_o}
        out["randomised_early"] = {"rows": rows_e, "metrics": m_e}

    suffix = "_norest" if a.exclude_rest else ""
    dest = a.out or f"results/analysis/context_dependence/{label}_{a.state}{suffix}.json"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    json.dump(out, open(dest, "w"), indent=1, default=float)
    print(f"\nwritten: {dest}")


if __name__ == "__main__":
    main()
