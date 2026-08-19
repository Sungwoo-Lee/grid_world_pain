import base64, pathlib

FIGS = [
    ("fig1_kernel_shape.png", "1", "The shape of the point-spread function",
     "Weight written into every grid cell by a single object, as the object moves further away. "
     "The isotropic kernel (top) grows in all directions at once — by d=4.5 its bright core covers "
     "both the north and the east cells, which is the collapse that started this thread. The "
     "anisotropic kernel (bottom) grows only <em>along</em> the ray to the object. Its cross-section "
     "stays narrow, so however vague the distance becomes, the bearing does not."),
    ("fig2_diamond_readout.png", "2", "What the agent actually reads",
     "The same object at 3.5 cells, seen through a range-2 diamond, at three bearings. At 45° both "
     "kernels report north = east, and that is the correct answer — the object really is on the "
     "diagonal. The difference shows at 20° and 70°, where the isotropic sensor barely separates the "
     "two cells and the anisotropic one separates them decisively."),
    ("fig3_angular_discrimination.png", "3", "How well can it tell up from right?",
     "A steep curve through 45° means the sensor resolves bearing; a flat one means it cannot. Both "
     "kernels degrade with distance — the honest claim is not that the anisotropic one is immune, but "
     "that it stays three to five times sharper at every range, and buys that sharpness without "
     "reducing the radial blur at all. Distance vagueness and bearing sharpness are genuinely "
     "independent axes."),
    ("fig4_normalisation.png", "4", "Does a distant object get fainter?",
     "This one surprised me and is worth a look. Normalising each object's kernel by its full analytic "
     "mass (green) reports only the fraction that lands inside the diamond, so signal falls about "
     "thirtyfold from d=1 to d=5 — blur width alone produces the distance falloff, and vision needs no "
     "separate 1/d<sup>γ</sup> term. Normalising over the visible cells instead (red) puts that mass "
     "straight back: a predator five cells away stays exactly as loud as one next door. Unnormalised "
     "(purple) is worse still — distant objects become brighter in total than near ones."),
    ("fig5_architectures.png", "5", "One scene, five mechanisms",
     "A predator up-and-right at 3.6 cells, a rabbit due west at 2, food due south at 1, all on real "
     "integer cells. Today's sensor (A) sees the adjacent food and is blind to both animals. Isotropic "
     "blur (B) smears them into one another. The anisotropic kernel (C) resolves three separate "
     "directional lobes. D and E are the two alternative architectures, included so you can see what "
     "you would be giving up."),
    ("fig6_anisotropy_sweep.png", "6", "Choosing the anisotropy knob",
     "ρ = σ_radial / σ_tangential, with the radial blur held fixed throughout. ρ=1 is exactly the "
     "isotropic kernel, so this single number gives you an ablation for free. Sharpness keeps rising "
     "with ρ and does not saturate — but once σ⊥ falls below roughly half a cell the blob no longer "
     "reaches the cells either side of the ray, and further increases buy astronomical value ratios "
     "rather than usable information. That puts the useful band around ρ = 2–4 on this grid."),
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


figs_html = "\n".join(f"""
    <figure class="fig">
      <div class="fig-head">
        <span class="fig-num">Fig {n}</span>
        <h3>{title}</h3>
      </div>
      <div class="fig-img"><img src="data:image/png;base64,{b64(f)}" alt="{title}"></div>
      <figcaption>{cap}</figcaption>
    </figure>""" for f, n, title, cap in FIGS)

settled_html = "\n".join(
    f'<div class="led-row"><dt>{k}</dt><dd>{v}</dd></div>' for k, v in SETTLED)
open_html = "\n".join(
    f'<div class="led-row"><dt>{k}</dt><dd>{v}</dd></div>' for k, v in OPEN)

HTML = f"""<title>Point-Spread Vision</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;1,400&family=Karla:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root {{
  --ground:   #f3f1f5;
  --surface:  #ffffff;
  --sunken:   #eae6ee;
  --ink:      #1c1721;
  --ink-soft: #4a4253;
  --muted:    #6e6579;
  --line:     #d9d3e0;
  --accent:   #9c2f6d;
  --ember:    #c85c31;
  --shadow:   0 1px 2px rgba(28,23,33,.05), 0 8px 24px -12px rgba(28,23,33,.18);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:   #131019;
    --surface:  #1c1824;
    --sunken:   #241f2e;
    --ink:      #ece8f1;
    --ink-soft: #c3bbcd;
    --muted:    #948b9f;
    --line:     #322b3d;
    --accent:   #dd6ba6;
    --ember:    #ef8f5d;
    --shadow:   0 1px 2px rgba(0,0,0,.4), 0 10px 30px -14px rgba(0,0,0,.7);
  }}
}}
:root[data-theme="dark"] {{
  --ground:   #131019;
  --surface:  #1c1824;
  --sunken:   #241f2e;
  --ink:      #ece8f1;
  --ink-soft: #c3bbcd;
  --muted:    #948b9f;
  --line:     #322b3d;
  --accent:   #dd6ba6;
  --ember:    #ef8f5d;
  --shadow:   0 1px 2px rgba(0,0,0,.4), 0 10px 30px -14px rgba(0,0,0,.7);
}}

* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  background: var(--ground);
  color: var(--ink);
  font: 400 16.5px/1.65 Karla, ui-sans-serif, system-ui, sans-serif;
  -webkit-font-smoothing: antialiased;
}}
.wrap {{ max-width: 1180px; margin: 0 auto; padding: 0 28px 96px; }}
.col  {{ max-width: 68ch; }}

header {{ padding: 72px 0 40px; border-bottom: 1px solid var(--line); margin-bottom: 44px; }}
.eyebrow {{
  font: 500 11.5px/1 "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing: .16em; text-transform: uppercase; color: var(--accent);
  display: block; margin-bottom: 20px;
}}
h1 {{
  font: 600 clamp(38px, 6vw, 60px)/1.04 Spectral, Georgia, serif;
  letter-spacing: -.02em; margin: 0 0 20px; text-wrap: balance; max-width: 16ch;
}}
.standfirst {{ font-size: 19px; line-height: 1.55; color: var(--ink-soft); max-width: 60ch; margin: 0; }}
.meta {{
  margin-top: 28px; font: 400 12.5px/1.5 "IBM Plex Mono", ui-monospace, monospace;
  color: var(--muted);
}}

h2 {{
  font: 600 27px/1.2 Spectral, Georgia, serif; letter-spacing: -.01em;
  margin: 64px 0 18px; text-wrap: balance;
}}
h2:first-of-type {{ margin-top: 0; }}
p {{ margin: 0 0 18px; }}
em {{ font-style: italic; }}
strong {{ font-weight: 700; }}
code {{
  font: 400 .89em/1 "IBM Plex Mono", ui-monospace, monospace;
  background: var(--sunken); padding: .18em .4em; border-radius: 3px;
}}

.formula {{
  background: var(--surface); border: 1px solid var(--line); border-left: 3px solid var(--accent);
  border-radius: 4px; padding: 22px 26px; margin: 26px 0; overflow-x: auto;
  font: 400 17px/1.7 Spectral, Georgia, serif;
}}
.formula .fx {{ font-style: italic; }}
.formula .note {{
  display: block; margin-top: 12px; font: 400 13.5px/1.6 Karla, sans-serif;
  font-style: normal; color: var(--muted);
}}

.ledger {{ display: grid; gap: 34px; margin: 34px 0 8px; }}
@media (min-width: 860px) {{ .ledger {{ grid-template-columns: 1fr 1fr; gap: 44px; }} }}
.led-title {{
  font: 500 11.5px/1 "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing: .14em; text-transform: uppercase;
  padding-bottom: 12px; margin-bottom: 4px; border-bottom: 1px solid var(--line);
}}
.led-settled .led-title {{ color: var(--accent); }}
.led-open .led-title {{ color: var(--ember); }}
.led-row {{ display: grid; gap: 3px; padding: 14px 0; border-bottom: 1px solid var(--line); }}
.led-row dt {{ font-weight: 700; font-size: 15px; }}
.led-row dd {{ margin: 0; font-size: 14.5px; line-height: 1.55; color: var(--ink-soft); }}
dl {{ margin: 0; }}

.fig {{
  margin: 56px 0; background: var(--surface); border: 1px solid var(--line);
  border-radius: 6px; box-shadow: var(--shadow); overflow: hidden;
}}
.fig-head {{ padding: 22px 26px 0; display: flex; gap: 14px; align-items: baseline; flex-wrap: wrap; }}
.fig-num {{
  font: 500 11.5px/1 "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing: .12em; text-transform: uppercase; color: var(--accent);
  border: 1px solid var(--line); border-radius: 3px; padding: 5px 8px;
}}
.fig-head h3 {{ font: 600 21px/1.25 Spectral, Georgia, serif; margin: 0; letter-spacing: -.01em; }}
.fig-img {{ padding: 20px 26px 4px; overflow-x: auto; background: var(--surface); }}
.fig-img img {{
  display: block; width: 100%; max-width: 100%; height: auto;
  background: #fff; border-radius: 3px;
}}
figcaption {{
  padding: 4px 26px 26px; font-size: 14.5px; line-height: 1.6; color: var(--ink-soft);
  max-width: 82ch;
}}

.callout {{
  border-left: 3px solid var(--ember); background: var(--sunken);
  padding: 20px 24px; border-radius: 0 4px 4px 0; margin: 30px 0;
}}
.callout p:last-child {{ margin-bottom: 0; }}
.callout .tag {{
  font: 500 11px/1 "IBM Plex Mono", ui-monospace, monospace;
  letter-spacing: .14em; text-transform: uppercase; color: var(--ember);
  display: block; margin-bottom: 10px;
}}

footer {{
  margin-top: 76px; padding-top: 26px; border-top: 1px solid var(--line);
  font: 400 13px/1.7 "IBM Plex Mono", ui-monospace, monospace; color: var(--muted);
}}
footer code {{ background: none; padding: 0; color: var(--ink-soft); }}
</style>

<div class="wrap">
<header>
  <span class="eyebrow">GridWorld Pain · sensor design · mechanism study</span>
  <h1>Point-Spread Vision</h1>
  <p class="standfirst">Six numerical experiments on how the visual sensor should degrade with
  distance — and why blurring it the obvious way destroys the one thing worth keeping.</p>
  <p class="meta">sandbox: <code>docs/develop/active/sensors/visual_psf_study/</code> · pure numpy, independent of <code>src/environment/sensor.py</code></p>
</header>

<div class="col">
<h2>The problem</h2>
<p>The visual sensor is currently an exact cell match, so an object two cells away is reported with
perfect localisation and perfect identity. That is too strong for an abstraction of a retina, and the
natural fix — blur it, more with distance — has a failure mode you spotted before we built anything:
widen the blur enough and the north cell and the east cell say the same thing about an object that is
plainly up-and-to-the-right.</p>

<p>The measurements below say that intuition is exactly right, and also that the diagnosis can be
sharpened. It is not blur that destroys bearing. It is <em>tangential</em> blur. An isotropic kernel
spreads signal equally along the ray to the object and across it; only the across-the-ray component
carries bearing, and only the along-the-ray component carries distance. Widen them together and you
lose both. Widen them separately and you can spend one to buy the other.</p>

<h2>The fix</h2>
<p>Blur anisotropically in a polar frame: elongate along the agent-to-object ray, stay narrow across
it.</p>
</div>

<div class="formula col">
<span class="fx">w(c) = exp( −(v·û)² / 2σ<sub>∥</sub>² − (v·t̂)² / 2σ<sub>⊥</sub>² ),&nbsp;&nbsp; v = c − e</span>
<span class="note">û is the unit vector from agent to object, t̂ is perpendicular to it. σ<sub>∥</sub>
grows quickly with distance — where it is becomes vague. σ<sub>⊥</sub> stays tight — which way it is
stays sharp.</span>
</div>

<div class="col">
<p>There is a principled reason this is the right abstraction rather than a patch. A real retina
measures <strong>angle</strong> directly — a pixel <em>is</em> a direction — and never measures distance
at all; distance is inferred. So the honest abstraction of a camera is precisely one whose angular
precision is roughly constant and whose radial precision is poor. That falls out for free if the
tangential width is defined by a fixed angular blur, σ<sub>⊥</sub> = d · sin Δθ: the linear spread
grows with range while the angular uncertainty does not. Which is your night-vision example — you know
something is on your left, you do not know how far.</p>

<div class="callout">
  <span class="tag">worth knowing before you choose</span>
  <p>A fixed linear blur is, in principle, invertible. The network sees the same operator on every step,
  the content is sparse and non-negative, and the policy is recurrent — close to a best case for
  learning an implicit deconvolution. Blur destroys information only where the operator is
  rank-deficient. You have chosen deterministic blur with no stochastic term, which is a defensible
  call for reproducibility; it is worth knowing that it is the assumption most likely to be tested by
  a sufficiently trained agent.</p>
</div>
</div>

{figs_html}

<div class="col">
<h2>Where this leaves the decision</h2>
</div>

<div class="ledger">
  <section class="led-settled">
    <h3 class="led-title">Settled</h3>
    <dl>{settled_html}</dl>
  </section>
  <section class="led-open">
    <h3 class="led-title">Still open</h3>
    <dl>{open_html}</dl>
  </section>
</div>

<footer>
Figures regenerate with <code>python make_figs.py &amp;&amp; python fig6.py</code>, the page with
<code>python build_page.py</code>. Kernels live in <code>psf_lib.py</code>; nothing here touches
<code>src/</code>. Write-up: <code>docs/develop/active/sensors/VISUAL_PSF_MECHANISM_STUDY.md</code>.
</footer>
</div>
"""

pathlib.Path("index.html").write_text(HTML)
print("wrote index.html", len(HTML) // 1024, "KB")
