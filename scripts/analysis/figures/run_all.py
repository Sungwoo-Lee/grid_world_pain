#!/usr/bin/env python
"""Run every per-figure script in order and report which produced output.

Exists so that a missing or broken figure is a loud failure rather than a blank panel discovered
later by a reader.
"""
import glob, os, subprocess, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))

def main():
    scripts = sorted(glob.glob(os.path.join(HERE, "fig*.py")))
    extra = [os.path.join(HERE, "verify_environment_parity.py")]
    scripts = [s for s in extra if os.path.exists(s)] + scripts
    args = sys.argv[1:]
    ok, failed = [], []
    for s in scripts:
        name = os.path.basename(s)
        t0 = time.time()
        print(f"\n{'='*70}\n  {name}\n{'='*70}", flush=True)
        r = subprocess.run([sys.executable, s, *args], cwd=ROOT)
        (ok if r.returncode == 0 else failed).append((name, time.time()-t0))
    print(f"\n{'='*70}\n  SUMMARY\n{'='*70}")
    for n, t in ok: print(f"  ok      {n:44} {t:6.0f}s")
    for n, t in failed: print(f"  FAILED  {n:44} {t:6.0f}s")
    print(f"\n{len(ok)} succeeded, {len(failed)} failed")
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
