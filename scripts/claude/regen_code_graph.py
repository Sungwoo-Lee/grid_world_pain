#!/usr/bin/env python3
"""Wrapper around `graphify` to produce <scope>/graphify-out/GRAPH_REPORT.md.

Run:
    python scripts/regen_code_graph.py [--scope src|all]

What it does:
    - If `graphify` (the binary installed by graphifyy) is on PATH, runs
      `graphify update <scope>` to extract a code knowledge graph via tree-sitter
      (no LLM, no API key needed). Writes graph.json + graph.html +
      GRAPH_REPORT.md inside <scope>/graphify-out/.
    - If `graphify` is NOT on PATH, prints install instructions and exits 0
      (treat as a no-op so this script can be wired into hooks or pre-commit
      without breaking CI).

Install (when you want the real output):
    /home/vncuser/miniconda3/envs/grid_world_pain/bin/pip install graphifyy
    # OR (if available): uv tool install graphifyy / pipx install graphifyy

The PyPI package is `graphifyy` (double-y); the installed binary is `graphify`
(single y). Other `graphify*` packages on PyPI are unaffiliated. See
https://github.com/safishamsi/graphify for the upstream README.

Output location:
    graphify writes to <scanned_path>/graphify-out/, NOT the cwd. With the
    default --scope src this means the output lives at src/graphify-out/.
    The wrapper does not move it; instead .gitignore matches **/graphify-out/
    so any scan target's output is ignored.

Output files (per graphify upstream):
    graph.json        - queryable knowledge graph
    graph.html        - interactive HTML visualisation
    GRAPH_REPORT.md   - reader-facing summary (god-nodes, surprising connections,
                        community hubs); the file code-review agents read

Usage:
    python scripts/regen_code_graph.py              # scan src/ (default)
    python scripts/regen_code_graph.py --scope all  # scan entire repo

Code-side graph is optional — agents fall back to grep / Read when
<scope>/graphify-out/ is absent.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


CONDA_PIP = "/home/vncuser/miniconda3/envs/grid_world_pain/bin/pip"


def _stub_exit() -> None:
    """Print install instructions and exit 0 (no-op stub)."""
    print("graphify not installed. To enable the code-side wiki:")
    print()
    print("    # Install graphifyy (double-y) into the project conda env:")
    print(f"    {CONDA_PIP} install graphifyy")
    print()
    print("    # Or, if uv / pipx are available:")
    print("    #   uv tool install graphifyy")
    print("    #   pipx install graphifyy")
    print()
    print("    # Then regenerate the code graph:")
    print("    python scripts/regen_code_graph.py")
    print()
    print("graphify reference: https://github.com/safishamsi/graphify")
    print()
    print("Code-side graph is optional — agents fall back to grep / Read when")
    print("<scope>/graphify-out/ is absent or stale.")
    sys.exit(0)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Regenerate <scope>/graphify-out/GRAPH_REPORT.md from src/ (or full repo).",
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

    # graphify might live in the same env as this Python interpreter even
    # when that env's bin/ is not on the shell PATH (e.g. conda env not activated).
    graphify_bin = shutil.which("graphify")
    if graphify_bin is None:
        sibling = Path(sys.executable).parent / "graphify"
        if sibling.is_file():
            graphify_bin = str(sibling)
    if graphify_bin is None:
        _stub_exit()
        return  # unreachable; _stub_exit() calls sys.exit(0)

    repo_root = Path(__file__).resolve().parent.parent  # scripts/ -> repo root
    if args.scope == "src":
        scan_target = repo_root / "src"
        if not scan_target.is_dir():
            print(f"ERROR: src/ directory not found at {scan_target}", file=sys.stderr)
            sys.exit(1)
    else:
        scan_target = repo_root

    expected_out = scan_target / "graphify-out"
    print(f"Running `graphify update {scan_target}` → {expected_out}/")

    cmd = [graphify_bin, "update", str(scan_target)]
    result = subprocess.run(cmd, cwd=str(repo_root))

    if result.returncode != 0:
        print(f"graphify exited with code {result.returncode}", file=sys.stderr)
        sys.exit(result.returncode)

    report = expected_out / "GRAPH_REPORT.md"
    if report.exists():
        size_kb = report.stat().st_size // 1024
        print(f"OK — {report.relative_to(repo_root)} written ({size_kb} KB).")
    else:
        print(
            f"WARNING: graphify exited 0 but {report} not found. "
            "Check graphify's installed version vs its README.",
            file=sys.stderr,
        )

    sys.exit(0)


if __name__ == "__main__":
    main()
