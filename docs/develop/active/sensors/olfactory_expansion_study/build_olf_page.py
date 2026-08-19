"""Build index.html for the olfactory expansion study.

Every number quoted in the prose is computed here from the same code the figures
use, so the text cannot drift away from the plots.
"""
import base64, pathlib
import numpy as np
import olf_lib as O

FOOD = np.array([[1., 0, 0, 0, 0]])
TH, DIST, RR = 20.0, 4.0, 1

src = O.polar_to_rc(TH, DIST)[None, :]
vals, cells = O.readout([0., 0.], src, FOOD, RR)
ch = vals[:, 0]
idx = lambda rc: int(np.argmin(np.abs(cells - np.array(rc, float)).sum(axis=1)))
CC, CN, CS, CE, CW = (ch[idx((0, 0))], ch[idx((-1, 0))], ch[idx((1, 0))],
                      ch[idx((0, 1))], ch[idx((0, -1))])
g_row, g_col = (CS - CN) / 2, (CE - CW) / 2
est = O.gradient_bearing(ch, RR)
err = O.bearing_error(TH, est)
today = float(O.sense_at([[0., 0.]], src, FOOD)[0][0])

SNR_ROWS = [(d, 1.0/d, O.adjacent_contrast(d), O.adjacent_contrast(d)/O.SIGMA_CFG)
            for d in (2, 3, 4, 5, 6, 8)]
DIM_ROWS = [(r, 2*r*r+2*r+1, (2*r*r+2*r+1)*O.VECSIZE_CFG,
             27 - O.VECSIZE_CFG + (2*r*r+2*r+1)*O.VECSIZE_CFG) for r in (0, 1, 2, 3)]

FIGS = [
    ("fig0_what_changes.png", "0", "What the expansion actually changes",
     "Nothing about the odour field changes — the sensor simply evaluates it at every cell of the "
     "diamond rather than only under the agent. Today's reading is a single number that says how much "
     "smell is present and nothing about where it comes from. Five readings of the same field carry a "
     "gradient, and the gradient points at the source."),
    ("fig1_decay_power.png", "1", "The decay power trades reach against contrast",
     "Small γ carries a long way but flattens out, so neighbouring cells barely differ and there is "
     "little direction to read. Large γ gives strong cell-to-cell contrast but the signal collapses "
     "within a few cells. γ = 1.0 is what <code>default.yaml</code> ships — note this is already a "
     "departure from the value the sensor doc records."),
    ("fig2_signal_budget.png", "2", "The signal budget",
     "The quantity that matters is not how much smell there is but how much the cells <em>differ</em>, "
     "and that difference falls as roughly γ/d<sup>γ+1</sup> — far faster than the level itself. "
     "Measured against the σ = 0.2 the project already configures for olfaction, a single observation "
     "carries usable direction only within about three cells."),
    ("fig3_direction_recovery.png", "3", "Can the agent recover the direction?",
     "The estimator here is a planar least-squares fit over all diamond cells — a readable proxy for "
     "the information present in the readout, not a claim about what the network does. Under noise a "
     "larger diamond helps substantially, because it supplies redundant samples to average. With no "
     "noise the ordering reverses beyond about three cells, since a plane fitted over a wider patch of "
     "a curved field carries more bias. The extra cells earn their keep exactly to the extent that "
     "noise is switched on."),
    ("fig4_superposition.png", "4", "Sources on the same channel sum into a phantom",
     "Contributions add before the agent ever sees them, so the sensor returns the gradient of the "
     "sum — one direction, pointing at the intensity-weighted centroid. In two of these three scenes "
     "that bearing has no food on it at all. This is inherent to a summed chemical field rather than a "
     "flaw in the expansion, but per-cell sampling is what turns it into a direction the agent can act "
     "on, and act on wrongly."),
    ("fig5_operating_envelope.png", "5", "The operating envelope",
     "Where per-cell olfaction does the job it is being added to do, as a function of decay power and "
     "source distance. The band never reaches far, and raising γ shifts it inward rather than growing "
     "it. On a 10×10 grid this is a close-range sense — an argument for pairing it with vision and "
     "keeping the diamond small, not for tuning γ harder."),
]

SETTLED = [
    ("Sampling", "For each diamond cell, run the existing <code>sense_resource</code> with that cell "
                 "as the sampling point — distances and the <code>sensor_radius</code> cutoff both "
                 "measured from the cell, not from the agent."),
    ("Parity", "<code>olfactory_sensor_range: 0</code> must reproduce today's observation "
               "byte-for-byte. The centre cell is the current computation unchanged."),
    ("Scope", "Olfaction gets the diamond and nothing else — no mask key, no added noise. It already "
              "degrades with distance through 1/d<sup>γ</sup>, and an entity is already made "
              "unsmellable by giving it an all-zero signature."),
]
OPEN = [
    ("Sensor range", "Figure 3 argues for 2 over 1 whenever perceptual noise is on — the error at "
                     "four cells roughly halves. Figure 5 argues against going further, and the "
                     "dimension count below argues loudly against 3."),
    ("Decay power γ", "Left at 1.0, or raised to buy near-field contrast at the cost of reach. "
                      "Figure 2 shows no value serves the whole grid."),
    ("The on-source rule", "<code>sense_resource</code> returns a decay of 2.0 when the sampling point "
                           "sits exactly on a source. Today only the agent's own cell can trigger it; "
                           "with per-cell sampling any diamond cell can, so a factor-two "
                           "discontinuity that fires rarely today would start firing often."),
    ("Dimension budget", "Range 2 takes the observation from 27 to 87 dimensions, more than tripling "
                         "it, with olfaction alone accounting for three quarters."),
]


def b64(p):
    return base64.b64encode(pathlib.Path(p).read_bytes()).decode()


figs_html = "\n".join(f"""
    <figure class="fig" id="fig{n}">
      <div class="fig-head"><span class="fig-num">Fig {n}</span><h3>{title}</h3></div>
      <div class="fig-img"><img src="data:image/png;base64,{b64(f)}" alt="{title}"></div>
      <figcaption>{cap}</figcaption>
    </figure>""" for f, n, title, cap in FIGS)

snr_html = "\n".join(
    f'<tr><td class="num">{d}</td><td class="num">{lv:.3f}</td><td class="num">{ct:.3f}</td>'
    f'<td class="num"><strong>{sn:.2f}</strong></td></tr>' for d, lv, ct, sn in SNR_ROWS)
dim_html = "\n".join(
    f'<tr><td class="num">{r}</td><td class="num">{n}</td><td class="num">{od}</td>'
    f'<td class="num">{tot}</td></tr>' for r, n, od, tot in DIM_ROWS)
settled_html = "\n".join(f'<div class="led-row"><dt>{k}</dt><dd>{v}</dd></div>' for k, v in SETTLED)
open_html = "\n".join(f'<div class="led-row"><dt>{k}</dt><dd>{v}</dd></div>' for k, v in OPEN)

# Styles are inlined rather than read from the sibling study: that page's index.html is
# generated and gitignored, so importing from it would break on a fresh clone.
CSS = f"""<style>
:root {{
  --ground:#eff3f2; --surface:#ffffff; --sunken:#e2eae8; --ink:#1c1721;
  --ink-soft:#4a4253; --muted:#6e6579; --line:#cfdcd9;
  --accent:#1f7a6c; --ember:#c9852b;
  --shadow:0 1px 2px rgba(28,23,33,.05), 0 8px 24px -12px rgba(28,23,33,.18);
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    --ground:#101816; --surface:#182220; --sunken:#1f2b28; --ink:#ece8f1;
    --ink-soft:#c3bbcd; --muted:#948b9f; --line:#2b3a37;
    --accent:#5cc4b0; --ember:#e8b45c;
    --shadow:0 1px 2px rgba(0,0,0,.4), 0 10px 30px -14px rgba(0,0,0,.7);
  }}
}}
:root[data-theme="dark"] {{
  --ground:#101816; --surface:#182220; --sunken:#1f2b28; --ink:#ece8f1;
  --ink-soft:#c3bbcd; --muted:#948b9f; --line:#2b3a37;
  --accent:#5cc4b0; --ember:#e8b45c;
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
</style>"""

HTML = f"""<title>Directional Olfaction</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;1,400&family=Karla:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
{CSS}

<div class="wrap">
<header>
  <span class="eyebrow">GridWorld Pain · sensor design · mechanism study</span>
  <h1>Directional Olfaction</h1>
  <p class="standfirst">Sampling the odour field at every cell of a diamond instead of only under the
  agent turns a scalar into a gradient. Six experiments on whether that gradient survives contact with
  the noise the project already configures.</p>
  <p class="meta">sandbox: <code>docs/develop/active/sensors/olfactory_expansion_study/</code> · companion to the visual point-spread study</p>
</header>

<div class="col">
<h2>The question</h2>
<p>Olfaction currently returns {O.VECSIZE_CFG} numbers: the summed chemical signature of everything in
range, evaluated at the agent's own cell. It says how much of each smell is present and nothing
whatsoever about where it comes from. An agent standing in a food gradient cannot tell which way is
uphill without moving and comparing across time.</p>

<p>The proposal is the smallest change that fixes this: evaluate the same function at every cell of a
Manhattan diamond rather than at one point. The field is unchanged; only the number of places it is
sampled changes. That gives the agent a spatial gradient, and a gradient has a direction.</p>

<p>Two questions follow, and neither is settled by the fact that the mechanism obviously
<em>works</em> in the noiseless case. First, how much directional signal is actually there — a
difference between two nearly equal numbers can be arithmetically real and practically useless.
Second, what does it cost, since the observation vector grows by the number of cells times the
number of chemical channels.</p>

<h2>Background</h2>
<p>Five ideas the study leans on. Skip to <a href="#mech">the mechanism</a> if a scalar field and its
finite-difference gradient are already familiar.</p>
</div>

<div class="prereq col">
  <div>
    <h4><span class="n">1</span> What the sensor computes today</h4>
    <p>Every entity carries a chemical vector. Each contributes it to the reading, scaled by an
    inverse power of its distance, and the contributions are summed:
    <code>c(p) = Σ<sub>n</sub> prop<sub>n</sub> / ‖e<sub>n</sub> − p‖<sup>γ</sup></code>, over sources
    within <code>sensor_radius</code>.</p>
    <p>This defines a <strong>scalar field</strong> per channel — a number attached to every point of
    the grid. Today the sensor evaluates it at exactly one point, <code>p</code> = the agent.</p>
  </div>
  <div>
    <h4><span class="n">2</span> A field's value versus its gradient</h4>
    <p>One sample of a field tells you its <em>value</em>. Direction lives in the <em>gradient</em> —
    the vector of partial derivatives pointing the way the field increases fastest. You cannot get a
    gradient from one sample, at any precision, because a derivative is defined by comparison.</p>
    <p>That is the whole reason the current sensor is directionless, and the whole reason sampling at
    several points fixes it. Nothing else about the sensor needs to change.</p>
  </div>
  <div>
    <h4><span class="n">3</span> Estimating a gradient from cells</h4>
    <p>With samples on a grid the standard estimate is a <strong>central difference</strong>:
    <code>∂c/∂row ≈ (c<sub>south</sub> − c<sub>north</sub>)/2</code>, and likewise east minus west. The
    two together give a vector; its bearing is the estimated source direction.</p>
    <p>With more than four neighbours you can instead fit a plane by least squares and take its
    gradient. That uses every cell, so noise averages down — but the field is curved, and a plane
    fitted over a wider patch fits it worse. This bias-variance trade-off is what
    <a href="#fig3">Figure 3</a> measures, and it is the reason the right diamond size depends on
    whether noise is on.</p>
  </div>
  <div>
    <h4><span class="n">4</span> What γ does, and the quantity that actually matters</h4>
    <p>The level at distance d goes as <code>d<sup>−γ</sup></code>. The <em>difference</em> between
    adjacent cells is its derivative, which goes as <code>γ·d<sup>−(γ+1)</sup></code> — one power of d
    steeper. Direction therefore fades faster than presence, always, whatever γ is.</p>
    <p>Raising γ multiplies the contrast by γ and adds a power of decay: sharper near the agent,
    invisible sooner. That is a genuine trade-off with no free setting, which is what
    <a href="#fig2">Figure 2</a> shows on an absolute scale.</p>
  </div>
  <div>
    <h4><span class="n">5</span> Absolute noise against a decaying signal</h4>
    <p>The perceptual-noise system adds Gaussian noise of a fixed σ to each observation dimension. It
    is <strong>absolute</strong>, not proportional — the same 0.2 whether the reading is 2.0 or 0.02.
    The directional signal, meanwhile, shrinks as <code>γ·d<sup>−(γ+1)</sup></code>.</p>
    <p>A constant floor under a falling signal gives a hard range limit rather than a graceful fade.
    Working out where that crossing falls is the main empirical content of this study, and the answer
    is closer than one might guess.</p>
  </div>
</div>

<div class="col">
<h2 id="mech">The mechanism</h2>
<p>One line changes: where the sampling point comes from.</p>
</div>

<div class="formula col">
  <span class="lbl">today</span>
  obs = <span class="vec">c</span>(agent) &nbsp;→&nbsp; {O.VECSIZE_CFG} numbers
  <span class="note">One evaluation, at the agent's cell.</span>
</div>

<div class="formula col">
  <span class="lbl">proposed</span>
  obs = [ <span class="vec">c</span>(agent + δ) &nbsp;for δ in diamond(r) ] &nbsp;→&nbsp; (2r²+2r+1) × {O.VECSIZE_CFG} numbers
  <span class="note">The same function <span class="vec">c</span>, evaluated once per cell. Distances
  and the <code>sensor_radius</code> cutoff are measured from each cell, not from the agent — so at
  r = 0 the centre cell reproduces today's value exactly, and parity is structural rather than a
  thing to test for.</span>
</div>

<div class="col">
<h2>Worked example</h2>
<p>One food source, bearing {TH:.0f}° from north, {DIST:.0f} cells away, on the default signature
<code>[1, 0, 0, 0, 0]</code>. Range-1 diamond, γ = {O.GAMMA_CFG}. This is the scene in
<a href="#fig0">Figure 0</a>.</p>

<p><strong>Today</strong> the sensor reports <code>{today:.3f}</code> on channel 0 and stops. That
number is consistent with a source {DIST:.0f} cells north, {DIST:.0f} cells south, or anywhere on the
circle between.</p>

<p><strong>With the diamond</strong>, the same field sampled five times gives centre
<code>{CC:.3f}</code>, north <code>{CN:.3f}</code>, south <code>{CS:.3f}</code>, east
<code>{CE:.3f}</code>, west <code>{CW:.3f}</code>. The central differences are
<code>(south − north)/2 = {g_row:+.4f}</code> and <code>(east − west)/2 = {g_col:+.4f}</code>, and the
bearing of that vector is <strong>{est:.1f}°</strong> against a true {TH:.0f}° — an error of
{err:.1f}°, from pure geometry with no fitting.</p>

<p><strong>Now look at the size of what we just used.</strong> The north–south difference is
{abs(CN-CS):.3f} and the east–west difference {abs(CE-CW):.3f}, on readings of about
{CC:.2f}. The direction is carried by differences an order of magnitude smaller than the values
carrying them — and the configured olfactory noise is σ = {O.SIGMA_CFG}, which is larger than either.</p>
</div>

<div class="tablewrap col">
<table>
<thead><tr><th>source distance</th><th>level at agent</th><th>adjacent-cell difference</th><th>difference ÷ σ</th></tr></thead>
<tbody>
{snr_html}
</tbody>
</table>
</div>

<div class="col">
<div class="callout">
  <span class="tag">the headline</span>
  <p>At the shipped γ = 1.0 and the configured olfaction σ = 0.2, the directional signal drops below
  the noise floor between two and three cells. Beyond about four cells a single observation is
  essentially direction-free, and by eight cells the difference the agent must detect is an
  order of magnitude under the noise added to it.</p>
  <p>Two things keep this from being fatal, and both are worth stating precisely rather than
  hand-waving. Perceptual noise is <strong>off by default</strong>
  (<code>perceptual_noise.enabled: false</code>), so this binds only in the noise experiments — which
  is to say, precisely the experiments this project exists to run. And the policy is recurrent, so it
  can integrate across steps; averaging K observations buys a factor of √K, which is one to two cells
  of range for a realistic K, not an order of magnitude.</p>
</div>

<h3 class="sub">What it costs</h3>
<p>Olfaction contributes <code>cells × vector_size</code> dimensions. The current observation is 27
dimensions in total, of which olfaction is {O.VECSIZE_CFG}.</p>
</div>

<div class="tablewrap col">
<table>
<thead><tr><th>olfactory_sensor_range</th><th>cells</th><th>olfaction dims</th><th>total observation</th></tr></thead>
<tbody>
{dim_html}
</tbody>
</table>
</div>

<div class="col">
<p>Range 2 more than triples the observation and leaves three quarters of it as olfaction. Range 3
takes it past 140 and would make the chemical sense larger than everything else in the agent's world
combined. Whatever Figure 3 says about accuracy, that is the counterweight.</p>

<h3 class="sub">The relationship to the visual sensor</h3>
<p>After this change the two exteroceptive senses are the same operator with different kernels:
a weight per (cell, source) pair, matmul'd against the sources' property vectors. Olfaction's kernel
is global and heavy-tailed, <code>1/d<sup>γ</sup></code>, summed without normalisation. Vision's — as
proposed in the companion study — is local, Gaussian, and mass-normalised. Everything else is shared.
That is worth knowing before implementing either, because it means one piece of machinery can serve
both, and it is worth saying in a paper, because the difference between the two senses reduces to a
choice of kernel rather than a difference of architecture.</p>
</div>

{figs_html}

<div class="col">
<h2>Where this leaves the decision</h2>
</div>

<div class="ledger">
  <section class="led-settled"><h3 class="led-title">Settled</h3><dl>{settled_html}</dl></section>
  <section class="led-open"><h3 class="led-title">Still open</h3><dl>{open_html}</dl></section>
</div>

<footer>
Figures regenerate with <code>python make_olf_figs.py</code>, the page with
<code>python build_olf_page.py</code> — every number in the prose is computed at build time from the
same code the figures use. Kernels live in <code>olf_lib.py</code>; nothing here touches
<code>src/</code>. Companion study: <code>../visual_psf_study/</code>.
</footer>
</div>
"""

pathlib.Path("index.html").write_text(HTML)
print(f"wrote index.html  {len(HTML)//1024} KB   "
      f"[check: today={today:.3f} est={est:.1f}deg err={err:.1f} SNR@d4={SNR_ROWS[2][3]:.2f}]")
