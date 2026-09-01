"""Render an artifact page in a real browser and report what is visually broken.

WHY THIS EXISTS. Two rounds of careful static review of `sensor_ladder.html` - one by me, one by a
dedicated reviewing agent - both missed a defect that was obvious within one second of looking at
the rendered page: three numbered lists were rendering one word per line, because `display:grid`
turns every contiguous run of BARE TEXT among an element's children into its own anonymous grid
item. Reading the CSS cannot find that. You have to compute the box tree, which means you have to
render.

Chrome is available in this container, so there is no excuse for shipping a page nobody looked at.
This script renders the page at several viewport widths and reports, per width:

  * horizontal page overflow, and which element causes it
  * any element whose box sticks out past the viewport
  * text blocks squeezed into an absurdly narrow column (the anonymous-grid-item signature)
  * elements that collapsed to zero height while still holding text
  * text that visually overlaps other text
  * images with no alt text, and images rendered below their natural size by less than 1x
  * a full-page screenshot per width, for a human or an agent to actually look at

Usage
-----
    python scripts/claude/check_artifact_layout.py <page.html> [--out DIR] [--widths 390 834 1440]

The page is wrapped in the same skeleton the Artifact host injects at publish time (doctype, a
charset and viewport meta, `body{margin:0}`, `img{max-width:100%}`, `[hidden]{display:none}`), so
what is measured is what a reader gets - NOT a raw file:// load of the fragment, which differs.
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, tempfile

HOST_SKELETON_HEAD = (
    '<!doctype html><html><head><meta charset="utf-8">'
    '<meta name="viewport" content="width=device-width,initial-scale=1">'
    '<style>:root{color-scheme:light}body{margin:0;font:14px system-ui,-apple-system,sans-serif;'
    'background:#fafafa}img{max-width:100%}[hidden]{display:none!important}</style>'
    '</head><body>')

# Collected in the page and printed as one JSON line the driver greps out of the DOM dump.
PROBE = r"""
<script>
// Content inside a CLOSED <details> is laid out and reports a real bounding box, but is never
// painted. Without this the probe reports every collapsed panel as overlapping whatever sits above
// it - 24 false positives on the first page that used one.
// OPEN_DETAILS is set by the driver's --open-details pass: collapsed panels are never
// geometry-checked otherwise, so a defect can hide inside one indefinitely.
if (window.__OPEN_DETAILS__) {
  document.addEventListener('DOMContentLoaded', function () {
    var dd = document.querySelectorAll('details');
    for (var i = 0; i < dd.length; i++) dd[i].setAttribute('open', '');
  });
}

function inClosedDetails(el) {
  // the <summary> of a closed <details> IS painted - only its siblings are hidden
  if (el.tagName === 'SUMMARY') return false;
  for (var n = el.parentElement; n && n !== document.body; n = n.parentElement) {
    if (n.tagName === 'DETAILS' && !n.hasAttribute('open')) return true;
  }
  return false;
}

// A box that overflows its own scroll container is not a defect - that is what the container is
// for. Only report a box that escapes to the PAGE.
function clipped(el) {
  for (var n = el.parentElement; n && n !== document.body; n = n.parentElement) {
    var c = getComputedStyle(n);
    if (/auto|scroll|hidden|clip/.test(c.overflowX + c.overflowY)) return true;
  }
  return false;
}
window.__probe = function () {
  var out = {overflow:null, stickout:[], narrow:[], zero:[], overlap:[], img:[]};
  var vw = document.documentElement.clientWidth;
  var de = document.documentElement;
  out.pageHeight = Math.max(de.scrollHeight, document.body.scrollHeight);
  if (de.scrollWidth > de.clientWidth + 1)
    out.overflow = {scrollWidth: de.scrollWidth, clientWidth: de.clientWidth};

  var all = document.querySelectorAll('body *');
  var texts = [];
  for (var i = 0; i < all.length; i++) {
    var el = all[i], r = el.getBoundingClientRect(), cs = getComputedStyle(el);
    if (cs.display === 'none' || cs.visibility === 'hidden' || el.closest('[hidden]')) continue;
    if (inClosedDetails(el)) continue;      // laid out, but never painted
    var tag = el.tagName.toLowerCase();
    var id = tag + (el.className && typeof el.className === 'string'
                    ? '.' + el.className.trim().split(/\s+/).join('.') : '');

    if (r.width > 0 && (r.right > vw + 1 || r.left < -1) && !clipped(el))
      out.stickout.push({el:id, left:Math.round(r.left), right:Math.round(r.right), vw:vw});

    // direct text of this element only (not descendants)
    var own = '';
    for (var n = el.firstChild; n; n = n.nextSibling)
      if (n.nodeType === 3) own += n.nodeValue;
    own = own.replace(/\s+/g, ' ').trim();

    if (own.length > 40) {
      if (r.width > 0 && r.width < 150)
        out.narrow.push({el:id, width:Math.round(r.width), chars:own.length,
                         text:own.slice(0, 60)});
      if (r.height === 0)
        out.zero.push({el:id, chars:own.length, text:own.slice(0, 60)});
      texts.push({id:id, el:el, r:{t:r.top, b:r.bottom, l:r.left, ri:r.right},
                  s:own.slice(0,40)});
    }
    if (tag === 'img') {
      var ph = (el.getAttribute('src') || '').indexOf('image/svg+xml') !== -1;
      if (el.getAttribute('alt') === null) out.img.push({el:id, why:'no alt'});
      // Only meaningful against the REAL raster: the lean-mode stand-in is a vector with no
      // intrinsic width, which would report every figure as upscaled.
      if (!ph && el.naturalWidth && r.width > el.naturalWidth * 1.02)
        out.img.push({el:id, why:'upscaled', shown:Math.round(r.width), natural:el.naturalWidth});
    }
  }
  // text-on-text overlap: same-ish vertical band AND horizontal overlap, different elements
  for (var i = 0; i < texts.length; i++)
    for (var j = i + 1; j < texts.length; j++) {
      var A = texts[i].r, B = texts[j].r;
      var vo = Math.min(A.b, B.b) - Math.max(A.t, B.t);
      var ho = Math.min(A.ri, B.ri) - Math.max(A.l, B.l);
      // <strong> inside <p> is a descendant, not a collision
      if (texts[i].el.contains(texts[j].el) || texts[j].el.contains(texts[i].el)) continue;
      if (vo > 6 && ho > 6) {
        out.overlap.push({a:texts[i].id, b:texts[j].id, vpx:Math.round(vo), hpx:Math.round(ho),
                          atext:texts[i].s, btext:texts[j].s});
        if (out.overlap.length > 25) return out;
      }
    }
  return out;
};
// DOMContentLoaded, NOT load+setTimeout: under --virtual-time-budget the deferred path never
// fired and the probe silently produced nothing. Images are placeholders in lean mode, so layout
// is already final here.
document.addEventListener('DOMContentLoaded', function () {
  var d = document.createElement('div');
  d.id = '__probe_result';
  d.textContent = JSON.stringify(window.__probe());
  d.style.display = 'none';
  document.body.appendChild(d);
});
</script>
"""


def chrome() -> str:
    for c in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser",
              "/opt/google/chrome/chrome"):
        p = subprocess.run(["which", c], capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout.strip()
        if os.path.exists(c):
            return c
    raise SystemExit("no Chrome found - this check requires a real browser")


def wrap(page_html: str, lean: bool, open_details: bool = False) -> str:
    body = page_html
    if lean:
        # swap inlined images for same-aspect placeholders: 60x smaller, identical text layout
        body = re.sub(r'src="data:image/png;base64,[^"]+"',
                      'src="data:image/svg+xml;utf8,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22'
                      ' viewBox=%220 0 1000 430%22%3E%3Crect width=%221000%22 height=%22430%22'
                      ' fill=%22%23e8e6e1%22/%3E%3C/svg%3E"', body)
    flag = "<script>window.__OPEN_DETAILS__=true;</script>" if open_details else ""
    return HOST_SKELETON_HEAD + flag + body + PROBE + "</body></html>"


def measure(binary: str, path: str, width: int, height: int) -> dict:
    out = subprocess.run(
        [binary, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
         f"--window-size={width},{height}", "--virtual-time-budget=10000", "--dump-dom",
         f"file://{path}"], capture_output=True, text=True, timeout=180).stdout
    # the serialized element carries a style attribute too, so allow any attributes here
    m = re.search(r'id="__probe_result"[^>]*>(.*?)</div>', out, re.S)
    if not m:
        raise SystemExit(f"probe did not run at width {width} (page failed to load?)")
    return json.loads(m.group(1).replace("&quot;", '"').replace("&amp;", "&")
                      .replace("&lt;", "<").replace("&gt;", ">"))


def shoot(binary: str, path: str, width: int, height: int, dest: str):
    subprocess.run([binary, "--headless=new", "--disable-gpu", "--no-sandbox", "--hide-scrollbars",
                    f"--window-size={width},{height}", "--virtual-time-budget=10000",
                    f"--screenshot={dest}", f"file://{path}"],
                   capture_output=True, timeout=240)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("page")
    ap.add_argument("--out", default="tmp/artifact_layout")
    ap.add_argument("--widths", nargs="+", type=int, default=[500, 834, 1440])
    ap.add_argument("--shot-height", type=int, default=24000)
    ap.add_argument("--open-details", action="store_true",
                    help="also render with every <details> expanded, so collapsed panels are "
                         "geometry-checked; they are invisible to the default pass")
    a = ap.parse_args()
    binary = chrome()
    # Chrome headless refuses to make the viewport narrower than 500px: --window-size=390 silently
    # reports clientWidth 500. Asking for less would test a width that was never rendered.
    too_narrow = [w for w in a.widths if w < 500]
    if too_narrow:
        print(f"note: Chrome headless floors the viewport at 500px; {too_narrow} raised to 500")
        a.widths = sorted({max(w, 500) for w in a.widths})
    os.makedirs(a.out, exist_ok=True)
    src = open(a.page).read()

    problems = 0
    for w in a.widths:
        with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False) as f:
            f.write(wrap(src, lean=True, open_details=a.open_details)); lean_path = f.name
        r = measure(binary, lean_path, w, 1400)
        shot = os.path.join(a.out, f"page_{w}.png")
        shoot(binary, lean_path, w, a.shot_height, shot)
        os.unlink(lean_path)

        print(f"\n{'='*74}\nviewport {w}px\n{'='*74}")
        if r["overflow"]:
            problems += 1
            print(f"  HORIZONTAL PAGE SCROLL: document is {r['overflow']['scrollWidth']}px wide "
                  f"in a {r['overflow']['clientWidth']}px viewport "
                  f"({r['overflow']['scrollWidth']-r['overflow']['clientWidth']}px of sideways drag)")
        for k, label in (("stickout", "element sticks out past the viewport"),
                         ("narrow", "text squeezed into a very narrow column"),
                         ("zero", "element has text but zero height"),
                         ("overlap", "text overlaps other text"),
                         ("img", "image problem")):
            for item in r[k][:8]:
                problems += 1
                print(f"  {label}: {json.dumps(item)}")
            if len(r[k]) > 8:
                print(f"  ... and {len(r[k])-8} more '{label}'")
        if not r["overflow"] and not any(r[k] for k in ("stickout","narrow","zero","overlap","img")):
            print("  clean")
        ph = int(r.get("pageHeight") or 0)
        if ph > a.shot_height:
            problems += 1
            print(f"  SCREENSHOT TRUNCATED: the page is {ph:,}px tall but the capture is only "
                  f"{a.shot_height:,}px. The bottom {ph - a.shot_height:,}px was never rendered to "
                  f"an image and CANNOT have been reviewed. Re-run with "
                  f"--shot-height {int(ph * 1.05 // 1000 + 1) * 1000}.")
        print(f"  screenshot: {shot}   (page is {ph:,}px tall)")

    print(f"\n{'='*74}")
    print(f"{problems} problem(s) found across {len(a.widths)} viewport(s)")
    print("Screenshots are written for a human or an agent to LOOK at - the numbers above catch")
    print("geometry, not ugliness. Open them.")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
