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
import base64, io, json, os, re, subprocess, sys

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

    def is_number(cell: str) -> bool:
        t = cell.strip().strip("`").replace(",", "").rstrip("%x")
        if t in ("", "-", "\u2014"):
            return True
        try:
            float(t); return True
        except ValueError:
            return False

    def flush():
        if num and rows:
            head, body = rows[0], rows[2:]
            # Right-aligning a prose column next to the numbers produced ragged-left text that
            # read as a number column. Decide per column from its own contents: a column whose
            # body cells are mostly not numbers is left-aligned and allowed to wrap.
            ncol = len(head)
            prose = [sum(is_number(r[c]) for r in body if c < len(r)) < 0.6 * len(body)
                     for c in range(ncol)]
            cls = lambda c: ' class="txt"' if c and prose[c] else ""   # col 0 is already left
            th = "".join(f"<th{cls(i)}>{c}</th>" for i, c in enumerate(head))
            tb = "".join("<tr>" + "".join(f"<td{cls(i)}>{c}</td>" for i, c in enumerate(r))
                         + "</tr>" for r in body)
            # The caption must sit OUTSIDE the overflow container: inside it, scrolling a wide
            # table sideways carried its own title off the left edge.
            out[num] = (f'<figure class="tbl">'
                        f'<figcaption>Table {num}. {caption}</figcaption>'
                        f'<div class="scroll"><table>'
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


def samples_block(name: str) -> str:
    """The data accounting for one figure, rendered from what the figure script recorded.

    Several figures filter - a third of episodes contain no predator, the odour regression needs
    exactly one predator and one rabbit - and a reader cannot judge a number without knowing the
    denominator it came from. The figure script emits these counts; nothing here is typed by hand,
    and a figure that recorded none is a hard error rather than a silent omission.
    """
    p = os.path.join(ROOT, "results/analysis/ladder", f"samples_{name}.json")
    if not os.path.exists(p):
        raise SystemExit(f"figure {name} recorded no data accounting. Its script must call "
                         f"L.record_samples(...) - see scripts/analysis/ladder/_ladder.py.")
    rows = json.load(open(p))
    if not rows:
        raise SystemExit(f"figure {name} recorded an empty data accounting")
    body = "".join(
        f'<tr><td class="txt">{r["what"]}</td>'
        f'<td>{r["used"]:,}</td><td>{r["total"]:,}</td><td>{r["pct"]:.1f}%</td>'
        f'<td class="txt">{r["note"] or "&mdash;"}</td></tr>' for r in rows)
    return ('<details class="samples"><summary>Data behind this figure &mdash; '
            f'{rows[0]["used"]:,} of {rows[0]["total"]:,} ({rows[0]["pct"]:.1f}%)</summary>'
            '<div class="scroll"><table><thead><tr>'
            '<th class="txt">what</th><th>used</th><th>available</th><th>share</th>'
            '<th class="txt">why this subset</th></tr></thead>'
            f'<tbody>{body}</tbody></table></div></details>')


def attach_provenance(html: str) -> tuple[str, dict[str, str]]:
    """Name the generating script under every figure, and prove there is exactly one.

    The rule this enforces is the one the user asked for: every figure on the page is reproducible
    from a single named script. Writing that by hand in the template would let it drift silently the
    first time a figure was renamed, so it is derived here from the figure token itself and checked
    against what is actually on disk. A figure whose script is missing, or a script that the page
    never shows, is a hard error rather than a quiet inconsistency.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    seen: dict[str, str] = {}

    def one_figure(m):
        block = m.group(0)
        names = re.findall(r"\{\{FIG:([a-z0-9_]+)\}\}", block)
        if not names:
            return block                                    # a table-only <figure>, nothing to do
        if len(names) > 1:
            raise SystemExit(f"one <figure> shows several images: {names}. "
                             "Each figure must map to exactly one script.")
        name = names[0]
        script = f"{name}.py"
        if not os.path.exists(os.path.join(here, script)):
            raise SystemExit(f"figure {name} has no generating script "
                             f"(expected scripts/analysis/ladder/{script})")
        if name in seen:
            raise SystemExit(f"figure {name} appears on the page more than once")
        seen[name] = script
        prov = (f'<p class="prov">Reproduce this figure: '
                f'<code>python scripts/analysis/ladder/{script}</code></p>'
                + samples_block(name))
        if "</figcaption>" not in block:
            raise SystemExit(f"figure {name} has no <figcaption> to attach provenance to")
        # Every figure states its axes and how it was computed. Both were requested explicitly, and
        # both have gone missing before - once because a regex written to insert a method block
        # reached across a figure boundary and overwrote a neighbour's instead.
        if "<b>Axes.</b>" not in block:
            raise SystemExit(f"figure {name}'s caption does not state its axes "
                             "(the caption must contain an '<b>Axes.</b>' sentence)")
        if "How it is computed" not in block:
            raise SystemExit(f"figure {name} has no 'How it is computed' block")
        return block.replace("</figcaption>", "</figcaption>\n  " + prov, 1)

    out = re.sub(r"<figure\b.*?</figure>", one_figure, html, flags=re.S)
    return out, seen


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

    html, scripts = attach_provenance(html)
    html = re.sub(r"\{\{FIG:([a-z0-9_]+)\}\}", fig, html)
    html = re.sub(r"\{\{TABLE:(\d+)\}\}", tab, html)

    left = re.findall(r"\{\{[^}]+\}\}", html)
    if left:
        raise SystemExit(f"unsubstituted tokens remain: {sorted(set(left))}")

    on_disk = {f[:-4] for f in os.listdir(FIGS) if f.endswith(".png")}
    if on_disk - used_f:
        raise SystemExit(f"figures exist but the page never shows them: {sorted(on_disk - used_f)}")
    # and the reverse: every one-figure script must have produced a figure the page shows
    all_scripts = {f[:-3] for f in os.listdir(os.path.dirname(os.path.abspath(__file__)))
                   if re.fullmatch(r"lad\d\d_[a-z_]+\.py", f)}
    if all_scripts != set(scripts):
        raise SystemExit("figure scripts and shown figures disagree:\n"
                         f"  scripts with no figure on the page: {sorted(all_scripts - set(scripts))}\n"
                         f"  figures with no script: {sorted(set(scripts) - all_scripts)}")
    if set(tables) - used_t:
        print(f"note: generated tables not shown on the page: {sorted(set(tables) - used_t)}")

    open(OUT, "w").write(html)
    print(f"written: {OUT}  ({len(html)/1e6:.2f} MB)")
    print(f"figures inlined: {len(used_f)}/{len(on_disk)}   tables inlined: {len(used_t)}")
    print(f"every figure names its script: {len(scripts)}/{len(all_scripts)} scripts accounted for")


if __name__ == "__main__":
    main()
