#!/usr/bin/env python3
"""Wrapper around `graphify` to produce graphify-out/GRAPH_REPORT.md.

Run:
    python scripts/regen_code_graph.py [--scope src|all]

What it does:
    - If `graphify` is on PATH (installed in the active conda env), runs a
      graphify scan over the target scope and writes the output artefacts to
      graphify-out/ at the repo root.  The key file that code-review agents
      read is graphify-out/GRAPH_REPORT.md.
    - If `graphify` is NOT on PATH, prints install instructions and exits 0
      (treat as a no-op so this script can be wired into hooks or pre-commit
      without breaking CI).

Prerequisites (when you want the real output):
    pip install graphify        # see https://github.com/safishamsi/graphify
    python scripts/regen_code_graph.py

The output directory graphify-out/ is gitignored (large, per-machine
artefacts).  Only GRAPH_REPORT.md is the "contract" file that agents read;
the rest of graphify-out/ (call graphs, symbol tables, etc.) is supplementary.

Graphify CLI surface (from https://github.com/safishamsi/graphify README):
    graphify [--output-dir DIR] [--language LANG] [PATH]
    Default invocation: graphify . (scans current directory)
    Output: writes to ./graphify-out/ by default.

Usage:
    python scripts/regen_code_graph.py              # scan src/ (default)
    python scripts/regen_code_graph.py --scope all  # scan entire repo

Code-side graph is optional — agents fall back to grep / Read when
graphify-out/ is absent.

TODO (to enable):
    pip install graphify
    python scripts/regen_code_graph.py
"""

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _stub_exit() -> None:
    """Print install instructions and exit 0 (no-op stub)."""
    print("graphify not installed. To enable the code-side wiki:")
    print()
    print("    # Install graphify into the project conda env:")
    print("    /home/vncuser/miniconda3/envs/grid_world_pain/bin/pip install graphify")
    print()
    print("    # Then regenerate the code graph:")
    print("    python scripts/regen_code_graph.py")
    print()
    print("graphify reference: https://github.com/safishamsi/graphify")
    print()
    print("Code-side graph is optional — agents fall back to grep / Read when")
    print("graphify-out/ is absent or stale.")
    sys.exit(0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate graphify-out/GRAPH_REPORT.md from src/ (or full repo).",
    )
    p.add_argument(
        "--scope",
        choices=["src", "all"],
        default="src",
        help="Scope of the graphify scan: 'src' (default) scans src/ only; "
             "'all' scans the entire repo root.",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # --- Gate: check if graphify is installed ----------------------------------
    graphify_bin = shutil.which("graphify")
    if graphify_bin is None:
        _stub_exit()
        return  # unreachable; _stub_exit() calls sys.exit(0)

    # --- Determine scan target -------------------------------------------------
    repo_root = Path(__file__).resolve().parent.parent  # scripts/ -> repo root
    if args.scope == "src":
        scan_target = repo_root / "src"
        if not scan_target.is_dir():
            print(f"ERROR: src/ directory not found at {scan_target}", file=sys.stderr)
            sys.exit(1)
    else:
        scan_target = repo_root

    output_dir = repo_root / "graphify-out"
    output_dir.mkdir(exist_ok=True)

    print(f"Running graphify over {scan_target} → {output_dir}/")

    # --- Invoke graphify -------------------------------------------------------
    # CLI surface per https://github.com/safishamsi/graphify README:
    #   graphify [PATH]
    # Output directory is graphify-out/ by default (created in cwd).
    # We run from repo root so the default output path matches our gitignore.
    cmd = [graphify_bin, str(scan_target)]
    result = subprocess.run(cmd, cwd=str(repo_root))

    if result.returncode != 0:
        print(f"graphify exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    # Confirm the key output file exists.
    report = output_dir / "GRAPH_REPORT.md"
    if report.exists():
        print(f"OK — graphify-out/GRAPH_REPORT.md written ({report.stat().st_size} bytes).")
    else:
        print(
            "WARNING: graphify ran successfully but graphify-out/GRAPH_REPORT.md was not found.",
            file=sys.stderr,
        )
        print(
            "Check graphify's output directory flag (--output-dir?) against its installed version.",
            file=sys.stderr,
        )
        # Exit 0: graphify itself succeeded; the report location may differ by version.

    sys.exit(0)


if __name__ == "__main__":
    main()
