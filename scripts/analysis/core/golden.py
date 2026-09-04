#!/usr/bin/env python3
"""golden.py - decide whether a ported sweep reproduced the pipeline it replaced.

This is the gate the analysis-pipeline refactor is built on. Every ported sweep writes to a scratch
path and is compared here against the pre-refactor snapshot in `results/_golden_prerefactor_*`.

WHY THIS EXISTS RATHER THAN `diff`. The three sweeps being unified use three different NumPy
summation primitives for the same logical sums. On this project's data the float columns happen to
be bit-exact across all of them - they are float32-origin values summed in float64, so every partial
sum is exactly representable and order cannot matter - but that is a property of the DATA, not of
the primitives, and a gate that depends on today's columns being conveniently exact would fail a
correct port the first time an awkward column appeared.

So the criterion is tiered, and the tiers were fixed BEFORE any porting started:

    tier 1  integer-valued fields (0/1 sums, counts)   bit-exact
    tier 2  float fields                               rtol = 1e-12

The tiers are not a judgement call at comparison time. A field is tier 1 if every one of its values
satisfies `x == round(x)` in BOTH the golden and the candidate; anything else is tier 2. That rule
is mechanical, so the gate cannot be relaxed by reclassifying a field that failed - and a field that
is integer-valued in the golden but not in the candidate is reported as a TIER BREAK, which is
itself a defect: a count that has acquired a fractional part is wrong however small the difference.

The tolerance is pre-registered here, in the source, rather than chosen after seeing a diff.
"""
from __future__ import annotations
import argparse, csv, json, math, os, sys

RTOL = 1e-12                      # pre-registered; do not widen to make a comparison pass


def _is_intlike(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) \
        and math.isfinite(v) and v == round(v)


def _walk(node, path=""):
    """Yield (dotted_path, value) for every scalar leaf, lists included."""
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _walk(v, f"{path}.{k}" if path else str(k))
    elif isinstance(node, (list, tuple)):
        for i, v in enumerate(node):
            yield from _walk(v, f"{path}[{i}]")
    else:
        yield path, node


class Result:
    def __init__(self):
        self.tier1 = self.tier2 = 0
        self.fails: list[str] = []
        self.breaks: list[str] = []
        self.missing: list[str] = []

    @property
    def ok(self) -> bool:
        return not (self.fails or self.breaks or self.missing)


def compare_values(gold, cand, r: Result, label: str) -> None:
    g = dict(_walk(gold))
    c = dict(_walk(cand))
    for k in g:
        if k not in c:
            r.missing.append(f"{label}:{k} present in golden, absent in candidate")
    for k in c:
        if k not in g:
            r.missing.append(f"{label}:{k} present in candidate, absent in golden")
    for k, gv in g.items():
        if k not in c:
            continue
        cv = c[k]
        if isinstance(gv, str) or gv is None or isinstance(gv, bool):
            if gv != cv:
                r.fails.append(f"{label}:{k} non-numeric differs: {gv!r} -> {cv!r}")
            continue
        if not isinstance(cv, (int, float)):
            r.fails.append(f"{label}:{k} type changed: {type(gv).__name__} -> {type(cv).__name__}")
            continue
        # NaN is a legitimate value here: rate() writes NaN where a cell has too little support.
        if isinstance(gv, float) and math.isnan(gv):
            if not (isinstance(cv, float) and math.isnan(cv)):
                r.fails.append(f"{label}:{k} golden is NaN, candidate is {cv!r}")
            else:
                r.tier2 += 1
            continue
        if isinstance(cv, float) and math.isnan(cv):
            r.fails.append(f"{label}:{k} candidate is NaN, golden is {gv!r}")
            continue
        if _is_intlike(gv):
            if not _is_intlike(cv):
                r.breaks.append(f"{label}:{k} TIER BREAK - integer-valued in golden ({gv}), "
                                f"fractional in candidate ({cv!r})")
            elif gv != cv:
                r.fails.append(f"{label}:{k} tier-1 must be bit-exact: {gv} != {cv}")
            else:
                r.tier1 += 1
        else:
            if gv == cv:
                r.tier2 += 1
            elif math.isclose(gv, cv, rel_tol=RTOL, abs_tol=0.0):
                r.tier2 += 1
            else:
                rel = abs(cv - gv) / abs(gv) if gv else float("inf")
                r.fails.append(f"{label}:{k} tier-2 exceeds rtol={RTOL:g}: "
                               f"{gv!r} != {cv!r} (rel {rel:.3e})")


def load(path: str):
    if path.endswith(".json"):
        return json.load(open(path))
    if path.endswith(".csv"):
        rows = list(csv.DictReader(open(path)))
        out = []
        for row in rows:
            conv = {}
            for k, v in row.items():
                try:
                    conv[k] = float(v)
                except (TypeError, ValueError):
                    conv[k] = v
            out.append(conv)
        return out
    raise SystemExit(f"unsupported product type: {path}")


def compare_file(gold_path: str, cand_path: str, r: Result) -> None:
    label = os.path.basename(gold_path)
    if not os.path.exists(cand_path):
        r.missing.append(f"{label}: candidate not produced at {cand_path}")
        return
    compare_values(load(gold_path), load(cand_path), r, label)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("golden", help="a golden .json/.csv file, or a directory of them")
    ap.add_argument("candidate", help="the matching file, or the directory the port wrote to")
    ap.add_argument("--quiet", action="store_true", help="only print the verdict line")
    a = ap.parse_args()

    r = Result()
    if os.path.isdir(a.golden):
        n = 0
        for root, _, files in os.walk(a.golden):
            for f in sorted(files):
                if not (f.endswith(".json") or f.endswith(".csv")) or f.startswith("_"):
                    continue
                gp = os.path.join(root, f)
                cp = os.path.join(a.candidate, os.path.relpath(gp, a.golden))
                compare_file(gp, cp, r); n += 1
        scope = f"{n} product(s)"
    else:
        compare_file(a.golden, a.candidate, r)
        scope = os.path.basename(a.golden)

    if not a.quiet:
        for m in r.missing[:20]: print(f"  MISSING   {m}")
        for m in r.breaks[:20]:  print(f"  TIERBREAK {m}")
        for m in r.fails[:20]:   print(f"  FAIL      {m}")
        extra = len(r.missing) + len(r.breaks) + len(r.fails) - 60
        if extra > 0: print(f"  ... and {extra} more")
    print(f"{'REPRODUCED' if r.ok else 'DIVERGED'}: {scope} | "
          f"tier-1 bit-exact {r.tier1} | tier-2 within rtol {r.tier2} | "
          f"fail {len(r.fails)} | tier-break {len(r.breaks)} | missing {len(r.missing)}")
    sys.exit(0 if r.ok else 1)


if __name__ == "__main__":
    main()
