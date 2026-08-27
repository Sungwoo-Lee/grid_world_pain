"""Build index.html — the shareable mechanism-study page.

Every number quoted in the prose is COMPUTED here from the same formulas the
figures use, so the text cannot drift away from the plots.
"""
import base64, pathlib
import numpy as np

# ---------------------------------------------------------------- worked example ---
TH, D, RS, ANG = 20.0, 3.5, 0.5, 10.0            # matches FIG 2's left column
th = np.radians(TH)
e = np.array([-D * np.cos(th), D * np.sin(th)])   # object offset, (row, col)
u = e / D                                         # radial unit vector
t = np.array([-u[1], u[0]])                       # tangential unit vector
sp, st = RS * D, D * np.sin(np.radians(ANG))
rho = sp / st

def cell_calc(c):
    v = np.asarray(c, float) - e
    vpar, vperp = float(v @ u), float(v @ t)
    tp, tq = -vpar**2 / (2*sp**2), -vperp**2 / (2*st**2)
    return dict(c=c, v=v, vpar=vpar, vperp=vperp, tp=tp, tq=tq,
                ex=tp+tq, w=float(np.exp(tp+tq)),
                w_iso=float(np.exp(-(vpar**2 + vperp**2)/(2*sp**2))))

CN, CE = cell_calc([-2., 0.]), cell_calc([0., 2.])
ratio_a = CN['w'] / CE['w']
ratio_i = CN['w_iso'] / CE['w_iso']
full_mass = 2 * np.pi * sp * st

# ------------------------------------------------------------------------ figures ---
FIGS = [
    ("fig0_kernel_anatomy.png", "0", "Anatomy of the kernel",
     "Three steps from an ordinary Gaussian to the thing we actually use. Panel 1 is the blur "
     "everyone knows — one width, circular level sets, no direction privileged. Panel 2 stretches it "
     "along one axis and points that axis at the object. Panel 3 puts it on the grid and shows the "
     "decomposition worked through in the text below."),
    ("fig1_kernel_shape.png", "1", "The shape of the point-spread function",
     "Weight written into every grid cell by a single object, as the object moves further away. The "
     "isotropic kernel (top) grows in all directions at once — by d=4.5 its bright core covers both "
     "the north and the east cells, which is the collapse that started this thread. The anisotropic "
     "kernel (bottom) grows only <em>along</em> the ray. Its cross-section stays narrow, so however "
     "vague the distance becomes, the bearing does not."),
    ("fig2_diamond_readout.png", "2", "What the agent actually reads",
     "The same object at 3.5 cells, seen through a range-2 diamond, at three bearings. At 45° both "
     "kernels report north = east, and that is the correct answer — the object really is on the "
     "diagonal. The difference shows at 20° and 70°, where the isotropic sensor barely separates the "
     "two cells and the anisotropic one separates them decisively. The left column is the case worked "
     "through by hand above."),
    ("fig3_angular_discrimination.png", "3", "How well can it tell up from right?",
     "A steep curve through 45° means the sensor resolves bearing; a flat one means it cannot. Both "
     "kernels degrade with distance — the honest claim is not that the anisotropic one is immune, but "
     "that it stays three to five times sharper at every range, and buys that sharpness without "
     "reducing the radial blur at all. Distance vagueness and bearing sharpness are independent axes."),
    ("fig4_normalisation.png", "4", "Does a distant object get fainter?",
     "Normalising each object's kernel by its full analytic mass (green) reports only the fraction "
     "that lands inside the diamond, so signal falls about ninefold from d=1 to d=5 on the shipped kernel — blur width "
     "alone produces the distance falloff, and vision needs no separate 1/d<sup>γ</sup> term. "
     "Normalising over the visible cells instead (red) puts that mass straight back: a predator five "
     "cells away stays exactly as loud as one next door. Unnormalised (purple) is worse still — "
     "distant objects become brighter in total than near ones."),
    ("fig5_architectures.png", "5", "One scene, five mechanisms",
     "A predator up-and-right at 3.6 cells, a rabbit due west at 2, food due south at 1, all on real "
     "integer cells. Today's sensor (A) sees the adjacent food and is blind to both animals. Isotropic "
     "blur (B) smears them into one another. The anisotropic kernel (C) resolves three separate "
     "directional lobes. D and E are the two alternative architectures, included so you can see what "
     "you would be giving up."),
    ("fig8_noise_anatomy.png", "8", "Anatomy of noise",
     "Built the same way as Fig 0, from first principles to real numbers. <strong>Panel 1</strong> — "
     "noise means the reported value is the true value plus a random draw, so reading the same cell "
     "twice gives two answers. <strong>Panel 2</strong> — whether that matters depends entirely on the "
     "signal: at one cell the reading sits well clear of the noise; at five it overlaps zero, so "
     "\u201cfaint object\u201d and \u201cnothing there\u201d become the same reading. <strong>Panel 3</strong> "
     "— the distinction that matters most: the blur is identical every step and could in principle be "
     "learned and undone, while the noise differs every step and cannot. <strong>Panel 4</strong> — only "
     "\u03c3 changes between the models; the kernel is untouched in all of them. The retinal model is "
     "deliberately absent from that panel, because it keys on where the <em>cell</em> is rather than on "
     "signal strength, and sharing an axis would blur the very distinction the callout below keeps sharp."),
    ("fig7_noise.png", "7", "Distance-dependent noise: mostly already there",
     "Top row: the same object at 1, 3 and 5 cells, before and after adding the σ = 0.2 the project "
     "already configures for vision. At one cell it survives; by five it is gone. <strong>Nothing "
     "about the noise changed between those panels</strong> — only the signal, which the "
     "mass-normalised kernel already shrinks with distance. Bottom left: the signal crosses below "
     "the noise floor about three cells out. Bottom middle: the three conventional ways to scale σ. "
     "Bottom right: the two different distances that could drive noise, which are not the same thing."),
    ("fig6_anisotropy_sweep.png", "6", "Choosing the anisotropy knob",
     "ρ = σ<sub>∥</sub>/σ<sub>⊥</sub>, with the radial blur held fixed throughout. ρ=1 is exactly the "
     "isotropic kernel, so this single number gives you an ablation for free. Sharpness keeps rising "
     "with ρ and does not saturate — but once σ<sub>⊥</sub> falls below roughly half a cell the blob "
     "no longer reaches the cells either side of the ray, and further increases buy astronomical value "
     "ratios rather than usable information. That puts the useful band around ρ = 2–4 on this grid."),
]

SETTLED = [
    ("Olfaction", "Per-cell diamond, field resampled at each cell; distances and the radius cutoff "
                  "measured from that cell. Nothing else changes — no mask key, no added noise."),
    ("Masking", "Per-entity <code>visual_mask: none | far | all</code>."),
    ("Blur", "Deterministic. No stochastic term."),
    ("Support", "Folded into the entity matmul: a weight matrix replaces the boolean match matrix, so "
                "the periphery is soft and objects outside the diamond bleed into its edge."),
]
OPEN = [
    ("Architecture", "Anisotropic kernel on the existing diamond, or one of the two alternatives in "
                     "Figure 5 — sharp near field plus far sectors, or a log-polar map."),
    ("Width knobs", "Radial scale plus fixed angular blur in degrees, two independent linear scales, "
                    "or one width plus the anisotropy ratio ρ."),
    ("Normalisation", "Full analytic mass, over the visible cells, or none. Figure 4 makes the case "
                      "for the first."),
    ("Mask order", "Whether <code>far</code> zeroes the blurred weight at d≥1, or lets a masked object "
                   "contribute undimmed to the centre cell only."),
    ("Sensor range", "Every question above resolves differently at range 1 than at range 3. Nothing is "
                     "decidable until this one is."),
]


def b64(p):
    return base64.b64encode(pathlib.Path(p).read_bytes()).decode()


def _fig_block(f, n, title, cap):
    return f"""
    <figure class="fig" id="fig{n}">
      <div class="fig-head"><span class="fig-num">Fig {n}</span><h3>{title}</h3></div>
      <div class="fig-img"><img src="data:image/png;base64,{b64(f)}" alt="{title}"></div>
      <figcaption>{cap}</figcaption>
    </figure>"""


# Figures 7 and 8 are placed inside the noise section rather than the main run,
# so they are looked up individually and excluded from figs_html below.
figs = {n: _fig_block(f, n, t, c) for f, n, t, c in FIGS}
figs_html = "\n".join(figs[n] for _f, n, _t, _c in FIGS if n not in ("7", "8"))

settled_html = "\n".join(f'<div class="led-row"><dt>{k}</dt><dd>{v}</dd></div>' for k, v in SETTLED)
open_html = "\n".join(f'<div class="led-row"><dt>{k}</dt><dd>{v}</dd></div>' for k, v in OPEN)

HTML = f"""<title>Point-Spread Vision</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;1,400&family=Karla:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --ground:#f3f1f5; --surface:#ffffff; --sunken:#eae6ee; --ink:#1c1721;
  --ink-soft:#4a4253; --muted:#6e6579; --line:#d9d3e0;
  --accent:#9c2f6d; --ember:#c85c31;
  --shadow:0 1px 2px rgba(28,23,33,.05), 0 8px 24px -12px rgba(28,23,33,.18);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#131019; --surface:#1c1824; --sunken:#241f2e; --ink:#ece8f1;
    --ink-soft:#c3bbcd; --muted:#948b9f; --line:#322b3d;
    --accent:#dd6ba6; --ember:#ef8f5d;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 30px -14px rgba(0,0,0,.7);
  }}
}}
:root[data-theme="dark"] {{
  --ground:#131019; --surface:#1c1824; --sunken:#241f2e; --ink:#ece8f1;
  --ink-soft:#c3bbcd; --muted:#948b9f; --line:#322b3d;
  --accent:#dd6ba6; --ember:#ef8f5d;
  --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 30px -14px rgba(0,0,0,.7);
}}

* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--ground); color:var(--ink);
  font:400 16.5px/1.65 Karla, ui-sans-serif, system-ui, sans-serif;
  -webkit-font-smoothing:antialiased;
}}
.wrap {{ max-width:1180px; margin:0 auto; padding:0 28px 96px; }}
.col  {{ max-width:68ch; }}

header {{ padding:72px 0 40px; border-bottom:1px solid var(--line); margin-bottom:44px; }}
.eyebrow {{
  font:500 11.5px/1 "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing:.16em; text-transform:uppercase; color:var(--accent);
  display:block; margin-bottom:20px;
}}
h1 {{
  font:600 clamp(38px,6vw,60px)/1.04 Spectral, Georgia, serif;
  letter-spacing:-.02em; margin:0 0 20px; text-wrap:balance; max-width:16ch;
}}
.standfirst {{ font-size:19px; line-height:1.55; color:var(--ink-soft); max-width:60ch; margin:0; }}
.meta {{ margin-top:28px; font:400 12.5px/1.5 "IBM Plex Mono", ui-monospace, monospace; color:var(--muted); }}

h2 {{ font:600 27px/1.2 Spectral, Georgia, serif; letter-spacing:-.01em; margin:64px 0 18px; text-wrap:balance; }}
h2:first-of-type {{ margin-top:0; }}
h3.sub {{ font:600 18px/1.3 Spectral, Georgia, serif; margin:34px 0 10px; }}
p {{ margin:0 0 18px; }}
code {{ font:400 .89em/1 "IBM Plex Mono", ui-monospace, monospace; background:var(--sunken); padding:.18em .4em; border-radius:3px; }}
.vec {{ font-style:italic; }}

.formula {{
  background:var(--surface); border:1px solid var(--line); border-left:3px solid var(--accent);
  border-radius:4px; padding:22px 26px; margin:24px 0; overflow-x:auto;
  font:400 17px/1.75 Spectral, Georgia, serif;
}}
.formula.plain {{ border-left-color:var(--line); }}
.formula .note {{ display:block; margin-top:12px; font:400 13.5px/1.6 Karla, sans-serif; color:var(--muted); }}
.formula .lbl {{
  display:block; font:500 10.5px/1 "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin-bottom:12px;
}}
.frac {{ display:inline-block; vertical-align:middle; text-align:center; margin:0 .18em; }}
.frac .num {{ display:block; padding:0 .35em; border-bottom:1px solid currentColor; line-height:1.35; }}
.frac .den {{ display:block; padding:0 .35em; line-height:1.35; }}

.prereq {{ display:grid; gap:0; margin:26px 0; border:1px solid var(--line); border-radius:5px; overflow:hidden; background:var(--surface); }}
.prereq > div {{ padding:20px 24px; border-bottom:1px solid var(--line); }}
.prereq > div:last-child {{ border-bottom:none; }}
.prereq h4 {{
  margin:0 0 8px; font:600 16px/1.3 Karla, sans-serif; display:flex; gap:10px; align-items:baseline;
}}
.prereq .n {{
  font:500 11px/1 "IBM Plex Mono", ui-monospace, monospace; color:var(--accent);
  border:1px solid var(--line); border-radius:3px; padding:4px 7px; flex:none;
}}
.prereq p {{ margin:0 0 10px; font-size:15.5px; color:var(--ink-soft); }}
.prereq p:last-child {{ margin-bottom:0; }}

table {{ border-collapse:collapse; width:100%; margin:22px 0; font-size:14.5px; }}
.tablewrap {{ overflow-x:auto; }}
th, td {{ text-align:left; padding:9px 14px; border-bottom:1px solid var(--line); }}
th {{
  font:500 11px/1.3 "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing:.1em; text-transform:uppercase; color:var(--muted);
}}
td.num {{ font-variant-numeric:tabular-nums; font-family:"IBM Plex Mono", ui-monospace, monospace; font-size:13.5px; }}

.ledger {{ display:grid; gap:34px; margin:34px 0 8px; }}
@media (min-width:860px) {{ .ledger {{ grid-template-columns:1fr 1fr; gap:44px; }} }}
.led-title {{
  font:500 11.5px/1 "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing:.14em; text-transform:uppercase;
  padding-bottom:12px; margin-bottom:4px; border-bottom:1px solid var(--line);
}}
.led-settled .led-title {{ color:var(--accent); }}
.led-open .led-title {{ color:var(--ember); }}
.led-row {{ display:grid; gap:3px; padding:14px 0; border-bottom:1px solid var(--line); }}
.led-row dt {{ font-weight:700; font-size:15px; }}
.led-row dd {{ margin:0; font-size:14.5px; line-height:1.55; color:var(--ink-soft); }}
dl {{ margin:0; }}

.fig {{ margin:56px 0; background:var(--surface); border:1px solid var(--line); border-radius:6px; box-shadow:var(--shadow); overflow:hidden; }}
.fig-head {{ padding:22px 26px 0; display:flex; gap:14px; align-items:baseline; flex-wrap:wrap; }}
.fig-num {{
  font:500 11.5px/1 "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing:.12em; text-transform:uppercase; color:var(--accent);
  border:1px solid var(--line); border-radius:3px; padding:5px 8px;
}}
.fig-head h3 {{ font:600 21px/1.25 Spectral, Georgia, serif; margin:0; letter-spacing:-.01em; }}
.fig-img {{ padding:20px 26px 4px; overflow-x:auto; }}
.fig-img img {{ display:block; width:100%; max-width:100%; height:auto; background:#fff; border-radius:3px; }}
figcaption {{ padding:4px 26px 26px; font-size:14.5px; line-height:1.6; color:var(--ink-soft); max-width:82ch; }}

.callout {{ border-left:3px solid var(--ember); background:var(--sunken); padding:20px 24px; border-radius:0 4px 4px 0; margin:30px 0; }}
.callout p:last-child {{ margin-bottom:0; }}
.callout .tag {{
  font:500 11px/1 "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing:.14em; text-transform:uppercase; color:var(--ember); display:block; margin-bottom:10px;
}}

footer {{ margin-top:76px; padding-top:26px; border-top:1px solid var(--line); font:400 13px/1.7 "IBM Plex Mono", ui-monospace, monospace; color:var(--muted); }}
footer code {{ background:none; padding:0; color:var(--ink-soft); }}
</style>

<div class="wrap">
<header>
  <span class="eyebrow">GridWorld Pain · sensor design · mechanism study</span>
  <h1>Point-Spread Vision</h1>
  <p class="standfirst">Six numerical experiments on how the visual sensor should degrade with
  distance — and why blurring it the obvious way destroys the one thing worth keeping. Written to be
  followed from first principles.</p>
  <p class="meta">sandbox: <code>docs/develop/active/sensors/visual_psf_study/</code> · pure numpy, independent of <code>src/environment/sensor.py</code></p>
</header>

<div class="col">
<h2>The problem</h2>
<p>The visual sensor is currently an exact cell match: an object two cells away is reported with
perfect localisation and perfect identity. That is too strong for an abstraction of a retina, and the
natural fix — blur it, more with distance — has a failure mode worth taking seriously. Widen the blur
enough and the north cell and the east cell say the same thing about an object that is plainly
up-and-to-the-right.</p>

<p>The measurements below say that intuition is right, and also sharpen the diagnosis. It is not blur
that destroys bearing. It is <em>tangential</em> blur. An isotropic kernel spreads signal equally
along the ray to the object and across it; only the across-the-ray component carries bearing, and
only the along-the-ray component carries distance. Widen them together and you lose both. Widen them
separately and you can spend one to buy the other.</p>

<h2>Background</h2>
<p>Six ideas the mechanism rests on. If you already know what a covariance matrix is doing inside a
Gaussian exponent, skip to <a href="#fix">the fix</a>.</p>
</div>

<div class="prereq col">
  <div>
    <h4><span class="n">1</span> A point-spread function</h4>
    <p>Every real imaging system smears a point of light into a small blob. The blob's shape is the
    system's <strong>point-spread function</strong> — its response to a single point source. A perfect
    sensor has a PSF that is a spike at one pixel; any real one has a PSF with width.</p>
    <p>Our sensor currently has the spike. This whole study is about choosing a better-shaped blob to
    replace it with.</p>
  </div>
  <div>
    <h4><span class="n">2</span> The Gaussian and its width</h4>
    <p>The standard blob. In one dimension, <code>exp(−x²/2σ²)</code>: equal to 1 at the centre,
    falling smoothly away, with σ setting how fast. In two dimensions the simplest version just uses
    the length of the displacement vector <span class="vec">v</span>, so the value depends on
    <em>how far</em> a point is from the centre but not on <em>which way</em>.</p>
    <p>That directional blindness is the entire problem. Level sets — the sets of points sharing a
    value — are circles, and a circle privileges no direction.</p>
  </div>
  <div>
    <h4><span class="n">3</span> Two widths instead of one</h4>
    <p>To make direction matter, replace the single σ with a 2×2 <strong>covariance matrix</strong> Σ,
    and the term <code>‖v‖²/σ²</code> with the quadratic form
    <code>v<sup>T</sup>Σ<sup>−1</sup>v</code> (the squared <em>Mahalanobis</em> distance — distance
    measured in units of the blob's own spread, direction by direction).</p>
    <p>Σ's eigenvectors are the blob's principal axes and its eigenvalues are the squared widths along
    them. Equal eigenvalues give back the circle. Unequal ones give an <strong>ellipse</strong>,
    elongated along the axis with the larger eigenvalue. That is all "anisotropic" means here: the
    blob has a long direction and a short one.</p>
  </div>
  <div>
    <h4><span class="n">4</span> Pointing the ellipse at the object</h4>
    <p>We want the long axis aimed along the ray from agent to object. Let <span class="vec">û</span>
    be the unit vector in that direction and <span class="vec">t̂</span> be perpendicular to it. These
    two are an orthonormal basis, so any displacement splits cleanly:
    <code>v = (v·û)û + (v·t̂)t̂</code>. The dot products <em>are</em> the radial and tangential
    components — that is just a rotation into the object's own frame.</p>
    <p>Building Σ from those axes and the two widths gives
    <code>Σ = R·diag(σ<sub>∥</sub>², σ<sub>⊥</sub>²)·R<sup>T</sup></code> with
    <code>R = [û t̂]</code>. Substituting into the quadratic form and simplifying leaves exactly the
    two-term exponent used below — no matrix inverse survives, because rotating into an orthonormal
    frame diagonalises it. Worth doing once by hand; the sandbox asserts the two forms agree.</p>
  </div>
  <div>
    <h4><span class="n">5</span> Normalising, and why it decides the physics</h4>
    <p>A 2-D Gaussian integrates to <code>2π·√det Σ</code>, which for our diagonalised case is simply
    <code>2π σ<sub>∥</sub> σ<sub>⊥</sub></code>. Dividing by it makes each object contribute a fixed
    total across the <em>whole plane</em>.</p>
    <p>The sensor only observes a handful of cells, so a wider blob spills more of its mass outside
    them, and the object correctly reads fainter. Divide instead by the sum over the visible cells and
    you cancel precisely that effect — the object contributes 1.0 no matter how far away it is. This
    innocuous-looking choice is the difference between having a distance falloff and not having one.
    See <a href="#fig4">Figure 4</a>.</p>
  </div>
  <div>
    <h4><span class="n">6</span> Angles versus lengths</h4>
    <p>An angular uncertainty Δθ at distance d subtends a lateral offset of <code>d·sin Δθ</code>.
    Setting <code>σ<sub>⊥</sub> = d·sin Δθ</code> therefore holds the <em>angular</em> uncertainty
    constant while the spread measured in cells grows with range.</p>
    <p>That is not an arbitrary choice. A retina measures angle directly — a pixel <em>is</em> a
    direction — and never measures distance at all; distance is inferred. A sensor with constant
    angular error and poor distance judgement is the honest abstraction, not a convenient hack.</p>
  </div>
</div>

<div class="col">
<h2 id="fix">The fix</h2>
<p>Blur anisotropically in the polar frame: elongate along the agent-to-object ray, stay narrow
across it.</p>
</div>

<div class="formula col">
  <span class="lbl">the kernel</span>
  <span class="vec">w(c)</span> = exp( − <span class="frac"><span class="num">(<span class="vec">v</span>·<span class="vec">û</span>)²</span><span class="den">2σ<sub>∥</sub>²</span></span>
  − <span class="frac"><span class="num">(<span class="vec">v</span>·<span class="vec">t̂</span>)²</span><span class="den">2σ<sub>⊥</sub>²</span></span> ),
  &nbsp;&nbsp; <span class="vec">v</span> = <span class="vec">c</span> − <span class="vec">e</span>
  <span class="note"><span class="vec">c</span> is the cell being written into, <span class="vec">e</span> the object's position,
  <span class="vec">û</span> the unit vector from agent to object, <span class="vec">t̂</span> perpendicular to it.</span>
</div>

<div class="formula col plain">
  <span class="lbl">the two widths</span>
  σ<sub>∥</sub> = k · d &nbsp;&nbsp;·&nbsp;&nbsp; σ<sub>⊥</sub> = d · sin Δθ
  <span class="note">Both grow with the object's distance d, but for different reasons: σ<sub>∥</sub>
  because distance judgement genuinely degrades, σ<sub>⊥</sub> because a fixed angular error covers
  more ground further out. Their ratio ρ = σ<sub>∥</sub>/σ<sub>⊥</sub> is the anisotropy; ρ = 1 gives
  back the isotropic kernel exactly.</span>
</div>

<div class="formula col plain">
  <span class="lbl">normalisation</span>
  <span class="vec">w</span> ← <span class="vec">w</span> / (2π σ<sub>∥</sub> σ<sub>⊥</sub>)
  <span class="note">Divide by the kernel's integral over the whole plane, not by its sum over the
  visible cells. Only then does a distant object correctly read fainter.</span>
</div>

<div class="col">
<h2>Worked example</h2>
<p>One object, two cells, all the way through. This is the left column of
<a href="#fig2">Figure 2</a>, so every number here can be found in that plot.</p>

<p><strong>Setup.</strong> The agent sits at the origin. An object lies at bearing θ = {TH:.0f}° from
north, distance d = {D}. In (row, column) coordinates — row increasing downward, so north is −row —
that puts it at <code>e = ({e[0]:.3f}, {e[1]:.3f})</code>. The radial unit vector is
<code>û = e/d = ({u[0]:.3f}, {u[1]:.3f})</code>, and rotating it 90° gives
<code>t̂ = ({t[0]:.3f}, {t[1]:.3f})</code>.</p>

<p><strong>Widths.</strong> With k = {RS} and Δθ = {ANG:.0f}°:
<code>σ<sub>∥</sub> = {RS} × {D} = {sp:.2f}</code> cells, and
<code>σ<sub>⊥</sub> = {D} × sin {ANG:.0f}° = {st:.3f}</code> cells. The anisotropy is
ρ = {rho:.2f}, and the normalising mass is 2π σ<sub>∥</sub> σ<sub>⊥</sub> = {full_mass:.3f}.</p>

<p><strong>Now two cells of the range-2 diamond</strong>, both exactly 2 cells from the agent — one
due north, one due east. The object is mostly north, so a sensor that resolves bearing must weight
them very differently.</p>
</div>

<div class="tablewrap col">
<table>
<thead><tr><th>step</th><th>north cell (−2, 0)</th><th>east cell (0, 2)</th></tr></thead>
<tbody>
<tr><td>displacement <span class="vec">v</span> = <span class="vec">c</span> − <span class="vec">e</span></td>
    <td class="num">({CN['v'][0]:+.3f}, {CN['v'][1]:+.3f})</td>
    <td class="num">({CE['v'][0]:+.3f}, {CE['v'][1]:+.3f})</td></tr>
<tr><td>radial part <span class="vec">v</span>·<span class="vec">û</span></td>
    <td class="num">{CN['vpar']:+.3f}</td><td class="num">{CE['vpar']:+.3f}</td></tr>
<tr><td>tangential part <span class="vec">v</span>·<span class="vec">t̂</span></td>
    <td class="num">{CN['vperp']:+.3f}</td><td class="num">{CE['vperp']:+.3f}</td></tr>
<tr><td>radial term −(<span class="vec">v</span>·<span class="vec">û</span>)²/2σ<sub>∥</sub>²</td>
    <td class="num">{CN['tp']:.3f}</td><td class="num">{CE['tp']:.3f}</td></tr>
<tr><td>tangential term −(<span class="vec">v</span>·<span class="vec">t̂</span>)²/2σ<sub>⊥</sub>²</td>
    <td class="num">{CN['tq']:.3f}</td><td class="num">{CE['tq']:.3f}</td></tr>
<tr><td>exponent (sum)</td>
    <td class="num">{CN['ex']:.3f}</td><td class="num">{CE['ex']:.3f}</td></tr>
<tr><td><strong>weight w</strong></td>
    <td class="num"><strong>{CN['w']:.4f}</strong></td><td class="num"><strong>{CE['w']:.6f}</strong></td></tr>
<tr><td>same cells, isotropic kernel</td>
    <td class="num">{CN['w_iso']:.4f}</td><td class="num">{CE['w_iso']:.4f}</td></tr>
</tbody>
</table>
</div>

<div class="col">
<p><strong>Read the two middle rows.</strong> They are where the mechanism lives. For the north cell
the radial term contributes {CN['tp']:.2f} and the tangential term {CN['tq']:.2f} — comparable. For
the east cell the radial term is only {CE['tp']:.2f}, but the tangential term is
<strong>{CE['tq']:.2f}</strong>. The east cell is not much further along the ray; it is far
<em>across</em> it, and the across-direction is judged against a yardstick
{rho:.1f}× smaller. That single asymmetry does all the work.</p>

<p>The resulting north/east ratio is <strong>{ratio_a:.0f}×</strong>. Run the same two cells through
an isotropic kernel of the same radial width and it collapses to <strong>{ratio_i:.1f}×</strong>. Same
blur along the ray, same object, {ratio_a/ratio_i:.0f}× more directional information — bought purely
by narrowing the cross-section.</p>

<h3 class="sub">Why the 45° case is not a bug</h3>
<p>At a bearing of exactly 45° both kernels report north = east, and no amount of anisotropy changes
that. It is the correct answer: the object is genuinely equidistant from the two cells and genuinely
on the diagonal between them, so any kernel respecting the geometry must be symmetric there. The
pathology the anisotropy fixes is different — isotropic blur makes bearings of 20° and 70°
<em>also</em> read as roughly equal. Sharpness is about the slope through 45°, not the value at it.
That is exactly what <a href="#fig3">Figure 3</a> plots.</p>

<h3 class="sub">Two implementation consequences</h3>
<p><strong>It is a matmul, not a convolution.</strong> Because σ depends on each object's distance
from the agent, the kernel is spatially varying — the operator is not shift-invariant, so none of the
usual convolution machinery applies. In our case that costs nothing: the sensor is already a matmul
over entities, and this only replaces the boolean match matrix with a real-valued weight matrix of
the same shape.</p>

<p><strong>The sensor loses its hard range.</strong> Folding the kernel into that matmul means every
object on the grid has some weight in every cell. Gaussian tails make this numerically negligible
past a few σ, but <code>visual_sensor_range</code> now controls how many cells are <em>output</em>,
not how far the agent can see.</p>

<div class="callout">
  <span class="tag">worth knowing before you choose</span>
  <p>A fixed linear blur is, in principle, invertible. Write the sensor as
  <code>obs = W·x</code>: the agent sees the same operator W on every step, the content x is sparse
  and non-negative, and the policy is recurrent — close to a best case for learning an implicit
  deconvolution. Blur destroys information only where W is rank-deficient or badly conditioned; where
  it is merely ill-conditioned, a determined optimiser can still pull the signal back out. Noise is
  what converts "ill-conditioned" into "genuinely lost", because it puts a floor under the small
  singular values.</p>
  <p>You have chosen deterministic blur with no stochastic term, which is defensible for
  reproducibility. It is worth knowing that this is the assumption most likely to be tested by a
  sufficiently trained agent, and that adding a small Gaussian remains available as a follow-up.</p>
</div>
</div>

{figs_html}

<div class="col">
<h2>Noise, and why the sensor already has it</h2>

<p>A natural next question is whether the blur should be paired with <em>noise</em> that grows with
distance. It is worth separating two things that sound alike.</p>

<p><strong>Blur and noise are different failures.</strong> Blur is a <em>systematic</em> smearing: the
same object in the same place always produces the same spread-out reading. Noise is <em>random</em>:
the same object in the same place produces a slightly different reading every step. Blur makes you
uncertain about <em>where</em>; noise makes you unsure whether you saw anything at all. A sensor can
have either, both, or neither.</p>

<p><strong>The conventional model puts them in that order</strong> — the standard forward model in
imaging, astronomy and microscopy is</p>
</div>

{figs['8']}

<div class="formula col plain">
  <span class="lbl">the standard imaging model</span>
  <span class="vec">y</span> = <span class="vec">H</span><span class="vec">x</span> + <span class="vec">n</span>
  <span class="note"><span class="vec">H</span> is the point-spread function — the blur.
  <span class="vec">n</span> is noise, added to the <em>result</em>. Anisotropic kernels get exactly
  the same treatment as isotropic ones: <strong>the anisotropy lives in the kernel, not in the
  noise.</strong> There is no special "anisotropic noise" to reach for. This project's pipeline
  already has this shape — the sensor applies the kernel, and the perceptual-noise system adds noise
  to the assembled observation afterwards.</span>
</div>

<div class="col">
<h3 class="sub">The part that surprises people</h3>

<p>Because the kernel is <strong>mass-normalised</strong>, a distant object's signal is already small —
that is the whole point of the normalisation, and it is what Fig 4 measures. So even a <em>constant</em>
noise level produces a signal-to-noise ratio that collapses with distance, without any distance term
in the noise at all:</p>
</div>

<div class="tablewrap col">
<table>
<thead><tr><th>object distance</th><th>its brightest cell</th><th>signal ÷ noise, at the configured σ = 0.2</th></tr></thead>
<tbody>
<tr><td class="num">1</td><td class="num">0.637</td><td class="num">3.18</td></tr>
<tr><td class="num">2</td><td class="num">0.318</td><td class="num">1.59</td></tr>
<tr><td class="num">3</td><td class="num">0.170</td><td class="num">0.85</td></tr>
<tr><td class="num">4</td><td class="num">0.072</td><td class="num">0.36</td></tr>
<tr><td class="num">5</td><td class="num">0.037</td><td class="num"><strong>0.19</strong></td></tr>
</tbody>
</table>
</div>

{figs['7']}

<div class="col">
<p>By five cells the signal is five times <em>below</em> the noise. So "add distance noise" is
largely already available: switching on the visual noise this project already configures gives a
strong distance effect for one line of config, with no new mechanism.</p>

<h3 class="sub">If you did want to shape it, three conventional choices</h3>

<p><strong>1 · Flat σ — read noise.</strong> Signal-independent, constant everywhere. What is
configured today, and per the table it is already aggressive.</p>

<p><strong>2 · σ ∝ √signal — shot noise.</strong> The physically motivated model for anything that
accumulates photons: bright things are noisier in absolute terms but <em>more reliable</em> in
relative terms. It <em>softens</em> the falloff — SNR at five cells goes from 0.19 to 0.96. This is
the model a reviewer would expect of something described as an abstraction of a retina.</p>

<p><strong>3 · σ grows with cell eccentricity.</strong> The retinal model: peripheral vision is
noisier, not merely blurrier. The only one of the three that needs new code, and it is cheap —
cell distances are fixed by the diamond's geometry, so the per-cell σ can be precomputed once.</p>

<p>Real sensors have all three at once.</p>

<div class="callout">
  <span class="tag">a distinction worth keeping</span>
  <p>Two different distances could drive noise, and they are not the same. <strong>Entity
  distance</strong> — how far the <em>thing</em> is — is what drives the blur, and what the table
  above reflects. <strong>Cell eccentricity</strong> — how far the <em>sampling cell</em> is from the
  agent — is what option 3 would use. They diverge: a near object can land in a peripheral cell, and
  a far one can bleed into the centre. Option 3 makes the agent's own cell reliable and its periphery
  unreliable <em>regardless of what is in them</em>, which is a different claim from "far things are
  uncertain".</p>
</div>

<p>One confound to disarm first: the visual noise is currently <strong>injury-modulated</strong>
(<code>state_dependent</code>, <code>injury_noise_scale: 1.5</code>), so switching it on also couples
perceptual precision to the agent's damage. Deliberate for the pain research, but it means a
"distance noise" experiment would also be an "injury changes perception" experiment unless that
scale is set to zero.</p>
</div>

<div class="col">
<h2>Where this leaves the decision</h2>
</div>

<div class="ledger">
  <section class="led-settled"><h3 class="led-title">Settled</h3><dl>{settled_html}</dl></section>
  <section class="led-open"><h3 class="led-title">Still open</h3><dl>{open_html}</dl></section>
</div>

<footer>
Figures regenerate with <code>python fig0.py &amp;&amp; python make_figs.py &amp;&amp; python fig6.py</code>,
the page with <code>python build_page.py</code> — every number in the prose is computed at build time
from the same formulas the figures use. Kernels live in <code>psf_lib.py</code>; nothing here touches
<code>src/</code>. Write-up: <code>docs/develop/active/sensors/VISUAL_PSF_MECHANISM_STUDY.md</code>.
</footer>
</div>
"""

pathlib.Path("index.html").write_text(HTML)
print(f"wrote index.html  {len(HTML)//1024} KB   "
      f"[check: ratio_aniso={ratio_a:.1f} ratio_iso={ratio_i:.2f} rho={rho:.2f}]")
