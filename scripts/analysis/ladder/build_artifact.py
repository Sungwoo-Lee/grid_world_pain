"""Assemble the sensor-ladder artifact: template + real figures + generated tables.

The page's prose lives in `docs/experiments/active/sensor_ladder/artifact_template.html` and is
written by hand. Its numbers and images are not. This script substitutes two kinds of token so
that nothing in the published page is transcribed:

    {{FIG:lad07_hypervigilance_proximity}}   -> the real PNG, inlined as a base64 data URI
    {{TABLE:4}}                              -> table 4 from make_report_tables.py, as HTML

Re-run after any change to the data, the figures, or the template. A token that names a figure or
table that does not exist is a hard error rather than a silently empty slot - a blank panel that
nobody noticed is the single most common way one of these pages ships broken.
"""
from __future__ import annotations
import base64, io, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
DOC = os.path.join(ROOT, "docs/experiments/active/sensor_ladder")
FIGS = os.path.join(DOC, "figures")
TEMPLATE = os.path.join(DOC, "artifact_template.html")
OUT = os.path.join(DOC, "sensor_ladder.html")


def figure_uri(name: str) -> str:
    p = os.path.join(FIGS, name + ".png")
    if not os.path.exists(p):
        raise SystemExit(f"template references a figure that does not exist: {name}")
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()


def md_tables() -> dict[str, str]:
    """Run the table generator and convert each markdown table to HTML, keyed by its number."""
    r = subprocess.run([sys.executable, os.path.join(HERE, "make_report_tables.py")],
                       cwd=ROOT, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit("make_report_tables.py failed:\n" + r.stderr)
    out, num, rows, caption = {}, None, [], ""
    def flush():
        if num and rows:
            head, body = rows[0], rows[2:]
            th = "".join(f"<th>{c}</th>" for c in head)
            tb = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body)
            out[num] = (f'<figure class="tbl"><div class="scroll"><table>'
                        f'<caption>Table {num}. {caption}</caption>'
                        f'<thead><tr>{th}</tr></thead><tbody>{tb}</tbody></table></div></figure>')
    for line in r.stdout.splitlines():
        m = re.match(r"### TABLE (\d+) - (.+)", line)
        if m:
            flush(); num, caption, rows = m.group(1), m.group(2), []
            continue
        if line.startswith("|"):
            rows.append([c.strip() for c in line.strip().strip("|").split("|")])
    flush()
    return out


def inline_code(cells: str) -> str:
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", cells)


def main():
    if not os.path.exists(TEMPLATE):
        raise SystemExit(f"missing template: {TEMPLATE}")
    html = open(TEMPLATE).read()
    tables = md_tables()

    used_f, used_t = set(), set()
    def fig(m):
        used_f.add(m.group(1)); return figure_uri(m.group(1))
    def tab(m):
        n = m.group(1)
        if n not in tables:
            raise SystemExit(f"template references TABLE {n}, which the generator did not emit")
        used_t.add(n); return inline_code(tables[n])

    html = re.sub(r"\{\{FIG:([a-z0-9_]+)\}\}", fig, html)
    html = re.sub(r"\{\{TABLE:(\d+)\}\}", tab, html)

    left = re.findall(r"\{\{[^}]+\}\}", html)
    if left:
        raise SystemExit(f"unsubstituted tokens remain: {sorted(set(left))}")

    on_disk = {f[:-4] for f in os.listdir(FIGS) if f.endswith(".png")}
    if on_disk - used_f:
        raise SystemExit(f"figures exist but the page never shows them: {sorted(on_disk - used_f)}")
    if set(tables) - used_t:
        print(f"note: generated tables not shown on the page: {sorted(set(tables) - used_t)}")

    open(OUT, "w").write(html)
    print(f"written: {OUT}  ({len(html)/1e6:.2f} MB)")
    print(f"figures inlined: {len(used_f)}/{len(on_disk)}   tables inlined: {len(used_t)}")


if __name__ == "__main__":
    main()
