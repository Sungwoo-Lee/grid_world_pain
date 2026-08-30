"""Rebuild every sensor-ladder figure from the per-arm data, in order.

Assumes build_arm_data.py has already written results/analysis/ladder/<arm>.json for all fourteen
arms. Each figure script is independent and can be run on its own; this only saves typing.
"""
import glob, os, runpy, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
os.chdir(os.path.abspath(os.path.join(HERE, "..", "..", "..")))
fail = []
for f in sorted(glob.glob(f"{HERE}/lad*.py")):
    n = os.path.basename(f)
    print(f"\n{'='*78}\n{n}\n{'='*78}", flush=True)
    t = time.time()
    try:
        runpy.run_path(f, run_name="__main__")
        print(f"[ok] {n}  {time.time()-t:.1f}s", flush=True)
    except Exception as e:
        fail.append((n, repr(e)))
        print(f"[FAIL] {n}: {e!r}", flush=True)
print("\n" + "=" * 78)
if fail:
    print("FAILED:"); [print(f"  {n}: {e}") for n, e in fail]; sys.exit(1)
print("all ladder figures rebuilt")
