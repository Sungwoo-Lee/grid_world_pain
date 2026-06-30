#!/usr/bin/env python3
"""trajectory_story.py — step-level, story-level qualitative analysis of eval-rollout recordings.

Reads the .rec.gz recordings written by `scripts/eval_rollout.py --record`
(full per-step state: agent_pos, animal_pos, obs, actions, injury, nutrition) and
provides trajectory-level reads that aggregate statistics tend to hide.

WHY THIS EXISTS (read before trusting any aggregate):
  - Aggregates (episode-mean distance, fresh-vs-fresh flee rates, M1/M2/M5) can average
    away CONDITIONAL behaviour. Inspect individual trajectories + the RAW observation
    vector EARLY, before concluding.
  - When the hypothesis is "the agent does X", bin by the state X might be gated on.
  - Design a clean CONTROL before trusting a clever mechanism (e.g. remove an entity).
  - A careful observer's repeated, specific contradicting observation is evidence the
    MODEL is wrong, not noise to explain away.

Subcommands:
  summary  — per-episode survival, first-contact step, nearest-animal distance distribution
  dump     — step-by-step table for chosen episodes (agent, action, per-animal pos+dist,
             injury, nutrition, in-bush, contact markers)  <-- the "story" view
  flee     — flee decomposition: does the agent's OWN move increase distance to each animal
             (animal position held), distance-binned, pre-contact, optional fresh-vs-fresh
  obs      — decode the observation vector by sensor block for one episode (e.g. visual-
             channel COUNTS of on-cell animals)

Usage (run from project root with the conda interpreter):
  python scripts/trajectory_story.py summary <recordings_dir>
  python scripts/trajectory_story.py dump    <recordings_dir> [--episodes 0,1,2] [--longest N] [--every K] [--max-steps M]
  python scripts/trajectory_story.py flee    <recordings_dir> [--fresh] [--dist-max 6]
  python scripts/trajectory_story.py obs     <recordings_dir> --episode 0 [--steps 0-45] [--every 2]

<recordings_dir> = results/eval/<run>/models/<ckpt>/recordings/<pct>/  (has run_meta.pkl + episode_*.rec.gz)

SHORT WATCHABLE VIDEO (separate, not in this script):
  1) python scripts/eval_rollout.py --config <cfg> --agent_config <acfg> --checkpoint <ckpt> \
       --output-root <out> --eval-n-episodes N --record --record-n-episodes N --device gpu
  2) python scripts/render_recordings.py <out>/.../recordings/<pct>/ --workers 8 --fps 5 --concat
  3) If the concat is too long to open, make a first-5 clip (stream-copy, fast):
       printf "file '%s'\n" $PWD/<videos>/episode_00000{0,1,2,3,4}.mp4 > /tmp/c.txt
       ffmpeg -y -f concat -safe 0 -i /tmp/c.txt -c copy <videos>/eval_first5.mp4
"""
import os, sys, argparse
os.environ.setdefault("JAX_PLATFORMS", "cpu")          # avoid GPU init / OOM — pure analysis
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.utils.eval_recording import load_episode, load_run_meta

def man(a, b): return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))

def _load(recdir):
    recdir = Path(recdir)
    meta = load_run_meta(recdir)
    p = meta["params"]; AM = meta["action_map"]
    classes = list(p.animal_classes); tags = list(p.animal_tags)
    pred = [i for i, c in enumerate(classes) if c == "predator"]
    neut = [i for i, c in enumerate(classes) if c == "neutral"]
    hides = np.array(p.obs_hides_agent).astype(bool)
    files = sorted(recdir.glob("episode_*.rec.gz"))
    return meta, p, AM, classes, tags, pred, neut, hides, files

def _episode(f):
    ep = load_episode(f); sn = ep["snapshots"]; T = len(sn)
    A = [np.array(s["agent_pos"]) for s in sn]
    AN = [np.array(s["animal_pos"]) for s in sn]
    OB = [np.array(s["obs_pos"]) for s in sn]
    inj = [float(s.get("injury_level", 0.0)) for s in sn]
    nut = [float(s.get("nutrition", 0.0)) for s in sn]
    return ep, sn, T, A, AN, OB, inj, nut

def _first_contact(A, AN, idxs, T):
    """first step the agent shares a cell with any animal in idxs (-> contact)."""
    for t in range(T):
        if any(man(A[t], AN[t][i]) == 0 for i in idxs):
            return t
    return T

def cmd_summary(recdir):
    meta, p, AM, classes, tags, pred, neut, hides, files = _load(recdir)
    print(f"{len(files)} episodes | classes={classes} tags={tags}")
    lens, fcs = [], []
    distbins = {"pred": np.zeros(4), "neut": np.zeros(4)}; nsteps = 0
    deaths = 0
    for f in files:
        ep, sn, T, A, AN, OB, inj, nut = _episode(f)
        lens.append(T)
        if pred:
            fc = _first_contact(A, AN, pred, T)
            if fc < T: fcs.append(fc)
        term = sn[-1] if False else None
        if inj[-1] >= 99: deaths += 1
        for t in range(T):
            nsteps += 1
            for key, idxs in (("pred", pred), ("neut", neut)):
                if not idxs: continue
                d = min(man(A[t], AN[t][i]) for i in idxs)
                b = 0 if d == 0 else (1 if d <= 1.5 else (2 if d <= 3 else 3))
                distbins[key][b] += 1
    lens = np.array(lens)
    print(f"survival: mean={lens.mean():.0f} median={np.median(lens):.0f} %reach-max={(lens>=lens.max()).mean():.0%} | injury-death eps={deaths}")
    if fcs: print(f"first predator contact: mean step={np.mean(fcs):.0f} (n contacted={len(fcs)}/{len(files)})")
    for key, lbl in (("pred", "PREDATOR"), ("neut", "RABBIT/NEUTRAL")):
        b = distbins[key]
        if b.sum() == 0: continue
        b = b / b.sum()
        print(f"nearest {lbl:>14} dist:  on-cell={b[0]:.0%}  adjacent={b[1]:.0%}  near(<=3)={b[2]:.0%}  far={b[3]:.0%}")

def cmd_dump(recdir, episodes, longest, every, max_steps):
    meta, p, AM, classes, tags, pred, neut, hides, files = _load(recdir)
    def inbush(OB_t, a):
        return any((OB_t[i, 0] == a[0] and OB_t[i, 1] == a[1] and hides[i]) for i in range(OB_t.shape[0]))
    chosen = episodes
    if chosen is None:
        scored = []
        for i, f in enumerate(files):
            ep, sn, T, A, AN, OB, inj, nut = _episode(f)
            fc = _first_contact(A, AN, pred, T) if pred else T
            scored.append((i, fc))
        scored.sort(key=lambda x: -x[1])
        chosen = [i for i, _ in scored[:(longest or 3)]]
    for epi in chosen:
        ep, sn, T, A, AN, OB, inj, nut = _episode(files[epi])
        acts = ep["actions"]
        fc = _first_contact(A, AN, pred, T) if pred else T
        # per-animal first contacts
        labels = [f"{classes[i][0].upper()}:{tags[i]}" for i in range(len(classes))]
        print(f"\n#### EP {epi}: len={T}  predator-1st-contact={fc}  final_injury={inj[-1]:.0f}")
        hdr = "  {:>3} {:>7} {:>4} |".format("t", "agent", "act")
        for lab in labels: hdr += " {:>10}{:>2} |".format(lab[:10], "d")
        hdr += " {:>4} {:>4} {:>4}".format("inj", "nut", "bush")
        print(hdr)
        last = T if max_steps is None else min(fc + 5, T)
        for t in range(0, last, every):
            a = A[t]
            row = "  {:>3} {:>7} {:>4} |".format(t, str((int(a[0]), int(a[1]))), AM[int(acts[t])][:4])
            for i in range(len(classes)):
                pos = AN[t][i]; row += " {:>10}{:>2} |".format(str((int(pos[0]), int(pos[1]))), man(a, pos))
            mark = ""
            if t == fc: mark = " <<PRED CONTACT"
            row += " {:>4.0f} {:>4.0f} {:>4}{}".format(inj[t], nut[t], "YES" if inbush(OB[t], a) else "", mark)
            print(row)

def cmd_flee(recdir, fresh, dist_max):
    meta, p, AM, classes, tags, pred, neut, hides, files = _load(recdir)
    from collections import defaultdict
    bins = defaultdict(lambda: [0.0, 0, 0, 0])  # (class, dist) -> [sum_flee, n, away, toward]
    for f in files:
        ep, sn, T, A, AN, OB, inj, nut = _episode(f)
        fc_pred = _first_contact(A, AN, pred, T) if pred else T
        for kind, idxs in (("pred", pred), ("neut", neut)):
            for i in idxs:
                # window: pre this-animal's own first contact (fresh) OR pre any-predator contact
                fc_own = _first_contact(A, AN, [i], T)
                end = fc_own if fresh else fc_pred
                for t in range(end - 1):
                    d = man(A[t], AN[t][i])
                    if 1 <= d <= dist_max:
                        nxt = AN[t + 1][i]
                        fl = man(A[t + 1], nxt) - man(A[t], nxt)   # >0 = agent's move increased dist (flee)
                        s = bins[(kind, min(d, 5))]
                        s[0] += fl; s[1] += 1; s[2] += fl > 0; s[3] += fl < 0
    mode = "FRESH (pre-own-contact)" if fresh else "pre-predator-contact"
    print(f"Flee decomposition [{mode}] — agent's OWN move effect on distance (>0 = moved away), by distance:")
    print(f"{'dist':>5} | {'PRED flee':>9} {'%away':>6} {'n':>6} | {'NEUT flee':>9} {'%away':>6} {'n':>6}")
    for d in range(1, 6):
        sp = bins.get(("pred", d), [0, 0, 0, 0]); sr = bins.get(("neut", d), [0, 0, 0, 0])
        pm = sp[0] / sp[1] if sp[1] else float("nan"); pa = sp[2] / (sp[2] + sp[3]) if (sp[2] + sp[3]) else float("nan")
        rm = sr[0] / sr[1] if sr[1] else float("nan"); ra = sr[2] / (sr[2] + sr[3]) if (sr[2] + sr[3]) else float("nan")
        print(f"{d:>5} | {pm:>+9.3f} {pa:>6.0%} {sp[1]:>6} | {rm:>+9.3f} {ra:>6.0%} {sr[1]:>6}")
    print("NOTE: a `Rest` action gives flee=0 (excluded from %away) — high %away over a tiny n can still mean 'mostly sitting'.")

def cmd_obs(recdir, episode, steps, every):
    meta, p, AM, classes, tags, pred, neut, hides, files = _load(recdir)
    from src.environment.sensor import get_observation_breakdown
    bd = get_observation_breakdown(p)
    print(f"obs breakdown (in order): {dict(bd)}  | total {sum(bd.values())}")
    # offsets
    off = {}; c = 0
    for k, v in bd.items(): off[k] = (c, c + v); c += v
    vis0 = off["Visual"][0] if "Visual" in off else None
    ep, sn, T, A, AN, OB, inj, nut = _episode(files[episode])
    obsv = ep["obs"]; acts = ep["actions"]
    s0, s1 = (0, min(T, 46))
    if steps:
        a, b = steps.split("-"); s0, s1 = int(a), min(T, int(b) + 1)
    print(f"\nEP {episode}: observation visual-channel decode (ch5=predator, ch7=neutral; range-0 = on-cell only)")
    print(f"{'t':>3} {'agent':>7} {'act':>4} | nearest-pred-d nearest-neut-d | vis_ch5(pred) vis_ch7(neut) | full-visual-block")
    for t in range(s0, s1, every):
        a = A[t]
        dP = min((man(a, AN[t][i]) for i in pred), default=-1)
        dN = min((man(a, AN[t][i]) for i in neut), default=-1)
        v5 = obsv[t][vis0 + 5] if vis0 is not None else float("nan")
        v7 = obsv[t][vis0 + 7] if vis0 is not None else float("nan")
        visblk = np.round(obsv[t][vis0:off["Visual"][1]], 1).tolist() if vis0 is not None else []
        print(f"{t:>3} {str((int(a[0]),int(a[1]))):>7} {AM[int(acts[t])][:4]:>4} |  {dP:>11} {dN:>11} |   {v5:>10.1f} {v7:>11.1f} | {visblk}")

def main():
    ap = argparse.ArgumentParser(description="Trajectory-level qualitative analysis of eval recordings.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("summary", "dump", "flee", "obs"):
        s = sub.add_parser(name); s.add_argument("recordings_dir")
        if name == "dump":
            s.add_argument("--episodes", default=None, help="comma list e.g. 0,1,2 (default: longest pre-contact)")
            s.add_argument("--longest", type=int, default=3, help="if --episodes omitted, dump the N longest-pre-contact episodes")
            s.add_argument("--every", type=int, default=1)
            s.add_argument("--max-steps", action="store_true", help="stop a few steps past first contact (default: full episode)")
        if name == "flee":
            s.add_argument("--fresh", action="store_true", help="condition on each animal's OWN pre-contact window (fresh vs fresh)")
            s.add_argument("--dist-max", type=int, default=6)
        if name == "obs":
            s.add_argument("--episode", type=int, default=0)
            s.add_argument("--steps", default=None, help="range e.g. 0-45")
            s.add_argument("--every", type=int, default=2)
    a = ap.parse_args()
    if a.cmd == "summary": cmd_summary(a.recordings_dir)
    elif a.cmd == "dump":
        eps = [int(x) for x in a.episodes.split(",")] if a.episodes else None
        cmd_dump(a.recordings_dir, eps, a.longest, a.every, (True if a.max_steps else None))
    elif a.cmd == "flee": cmd_flee(a.recordings_dir, a.fresh, a.dist_max)
    elif a.cmd == "obs": cmd_obs(a.recordings_dir, a.episode, a.steps, a.every)

if __name__ == "__main__":
    main()
