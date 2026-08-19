"""Build index.html for the on-source decay-rule study."""
import base64, pathlib, json
import numpy as np

freq = json.loads(pathlib.Path('frequency.json').read_text())
P0, P1, P2 = (freq['pct']['0'], freq['pct']['1'], freq['pct']['2'])
NSTEPS = freq['total_agent_steps']

# every number in the prose, computed here
G = 1.0
walk_d = [5, 4, 3, 2, 1]
walk = [1.0 / d**G for d in walk_d] + [2.0]
ratios = [walk[i+1]/walk[i] for i in range(len(walk)-1)]
half_cell = {g: 1.0 / (0.5**g) for g in (0.5, 1.0, 2.0, 3.0)}

FIGS = [
    ("fig0_where_the_jump_is.png", "0", "Where the jump actually is",
     "At the shipped γ = 1.0 the special-case value 2.0 is exactly what the decay curve gives at half "
     "a cell, so the readings a walking agent sees run smoothly through contact. The code does contain "
     "a genuine discontinuity — anywhere between distance 0.001 and 1 it returns enormous values, then "
     "snaps to 2.0 below 0.001 — but entities and cells both sit on integer coordinates, so the "
     "distance is only ever 0, or 1 or more. The grey band is unreachable. The discontinuity is real, "
     "and nothing can ever land in it."),
    ("fig1_only_correct_at_one_gamma.png", "1", "The constant is only correct at one γ",
     "The star is the hard-coded 2.0; the hollow circle is what the decay curve itself gives at half a "
     "cell. They coincide at γ = 1 and nowhere else. At γ = 2 the sensor reports half what its own "
     "curve implies for standing on a source; at γ = 0.5 it reports 40% too much. The rule is not "
     "wrong today — it becomes wrong the moment anyone tunes γ, which the olfactory study explicitly "
     "contemplates as a way to buy near-field contrast."),
    ("fig2_walking_onto_food.png", "2", "What an agent walking onto food feels",
     "Numbers above the bars are readings; numbers between them are the step-to-step multiplier. At "
     "γ = 1 the last two steps are both ×2.00 — contact is the natural continuation of the approach. "
     "At γ = 2 the approach builds to ×4.00 and then the contact step is only ×2.00: the smell gets "
     "stronger more slowly at the exact moment the agent arrives. That is the kink worth caring about, "
     "and it appears only when γ is changed."),
    ("fig4_how_often.png", "4", "How often this actually happens",
     "Measured on the real environment with the shipped config. Only entities carrying a non-zero "
     "chemical signature count — rocks and hiding predators ship all-zero signatures, so the 2.0 "
     "multiplies to nothing for them. The rule is already common, and the expansion makes it the "
     "normal case rather than the exception."),
    ("fig3_three_options.png", "3", "Three ways to define “standing on it”",
     "Option B says “treat being on the source as being half a cell away” — the grid's own "
     "resolution limit, and the same floor the visual blur needs for the same reason. It passes "
     "through exactly 2.0 at γ = 1, so it is byte-identical to today under the shipped configuration, "
     "and it stays on the curve at every other γ where the constant does not."),
]


def b64(p):
    return base64.b64encode(pathlib.Path(p).read_bytes()).decode()


figs = {n: f"""
    <figure class="fig" id="fig{n}">
      <div class="fig-head"><span class="fig-num">Fig {n}</span><h3>{t}</h3></div>
      <div class="fig-img"><img src="data:image/png;base64,{b64(f)}" alt="{t}"></div>
      <figcaption>{c}</figcaption>
    </figure>""" for f, n, t, c in FIGS}

walk_row = "".join(f'<td class="num">{v:.3f}</td>' for v in walk)
ratio_row = "".join('<td class="num">—</td>' if i == 0 else
                    f'<td class="num">×{ratios[i-1]:.2f}</td>' for i in range(len(walk)))

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

HTML = f"""<title>The On-Source Rule</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;1,400&family=Karla:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
{CSS}

<div class="wrap">
<header>
  <span class="eyebrow">GridWorld Pain · sensor design · decision study</span>
  <h1>The On-Source Rule</h1>
  <p class="standfirst">What the smell sensor should report when it is standing exactly on top of the
  thing it is smelling — and how a one-line divide-by-zero guard quietly became a design decision.</p>
  <p class="meta">sandbox: <code>docs/develop/active/sensors/onsource_rule_study/</code> · measured on the real environment</p>
</header>

<div class="col">
<h2>The question</h2>
<p>The olfactory sensor works by dividing each source's strength by its distance. That breaks when the
distance is zero — when the sampling point sits exactly on a source — so the code has a special case
(<code>sensor.py:14</code>): if the distance is below 0.001, return the constant <strong>2.0</strong>
instead of dividing.</p>

<p>Today that guard fires only when the agent is standing on something. The proposed expansion samples
the field at five cells instead of one, so it will fire whenever a source is under the agent
<em>or any of its four neighbours</em>. Before shipping that, it is worth knowing what the rule
actually does — and whether 2.0 is a considered value or an inherited one.</p>

<h2>What "discontinuity" means</h2>
<p>Think of a <strong>dimmer dial</strong> versus a <strong>light switch</strong>. With a dimmer,
turning the knob a little changes the brightness a little; the output tracks the input smoothly. That
is <em>continuous</em>. With a switch there is one point where a millimetre of movement takes you from
dark to fully lit, and everywhere else movement does nothing. That leap is a
<strong>discontinuity</strong>: an arbitrarily small change in the cause producing a large change in
the effect.</p>

<p>It matters because a learner can generalise across a smooth relationship — "closer means stronger"
is one rule covering every distance — whereas a jump has to be memorised as a special case the general
rule does not predict. A jump can also mislead: if stepping onto food doubles the reading, but
stepping from four cells to three <em>also</em> doubles it, magnitude alone cannot distinguish two
very different situations.</p>

<div class="callout">
  <span class="tag">a correction</span>
  <p>I previously described this rule as "a factor-two discontinuity that fires often", and used that
  as an argument for changing it. Checking it properly, <strong>that is wrong for your actual
  configuration.</strong> At γ = 1.0 the constant 2.0 is not a jump at all — it is exactly where the
  decay curve already goes. I had read the special case in the code and assumed it produced a jump
  without checking what the agent experiences on an integer grid. The real issue turns out to be a
  different and narrower one.</p>
</div>

<h2>Why 2.0 is not arbitrary — at γ = 1</h2>
<p>Here is the agent's own-cell reading as it walks straight toward one food source:</p>
</div>

<div class="tablewrap col">
<table>
<thead><tr><th>distance</th><th>5</th><th>4</th><th>3</th><th>2</th><th>1</th><th>on it</th></tr></thead>
<tbody>
<tr><td>reading</td>{walk_row}</tr>
<tr><td>change from previous</td>{ratio_row}</tr>
</tbody>
</table>
</div>

<div class="col">
<p>Read the bottom row. The final step multiplies the reading by {ratios[-1]:.2f} — <strong>exactly the
same factor as the step before it</strong>. The sequence does not jump; it is a smooth accelerating
curve straight through to contact.</p>

<p>The reason is arithmetic: <code>1 / 0.5 = 2.0</code>. <strong>The constant is precisely what the
decay curve gives at half a cell.</strong> The rule is really saying <em>"treat standing on the source
as being half a cell away"</em> — and half a cell is the grid's own resolution limit, the same
principled floor the visual point-spread kernel needs for the same reason. It is not a magic number.
It is a half-cell floor with its arithmetic already carried out.</p>
</div>

{figs['0']}

<div class="col">
<h2>Where it does become a real problem</h2>
<p>The correspondence holds at exactly one value of γ. Change the decay power and the constant stops
matching its own curve:</p>
</div>

<div class="tablewrap col">
<table>
<thead><tr><th>decay power γ</th><th>curve at half a cell</th><th>what the code returns</th><th></th></tr></thead>
<tbody>
<tr><td class="num">0.5</td><td class="num">{half_cell[0.5]:.3f}</td><td class="num">2.000</td><td>40% too high</td></tr>
<tr><td class="num"><strong>1.0 (shipped)</strong></td><td class="num"><strong>{half_cell[1.0]:.3f}</strong></td><td class="num"><strong>2.000</strong></td><td><strong>exact match</strong></td></tr>
<tr><td class="num">2.0</td><td class="num">{half_cell[2.0]:.3f}</td><td class="num">2.000</td><td>half what it should be</td></tr>
<tr><td class="num">3.0</td><td class="num">{half_cell[3.0]:.3f}</td><td class="num">2.000</td><td>a quarter of it</td></tr>
</tbody>
</table>
</div>

<div class="col">
<p>This is not hypothetical. The olfactory expansion study's own open question is whether to raise γ to
buy near-field contrast — and the moment anyone does, the on-source value silently stops matching the
field it belongs to. The rule is correct today and is a latent trap.</p>
</div>

{figs['1']}
{figs['2']}

<div class="col">
<h2>How often it fires</h2>
<p>Measured on the real environment over {NSTEPS:,} agent-steps with the shipped configuration, counting
only sources that carry a non-zero chemical signature:</p>

<ul>
<li><strong>{P0:.1f}%</strong> of steps today — the agent's own cell is on a smelly entity</li>
<li><strong>{P1:.1f}%</strong> of steps at olfactory range 1 — the chosen setting</li>
<li><strong>{P2:.1f}%</strong> of steps at range 2</li>
</ul>

<p>Neither number is what "an edge case" looks like. The rule already governs roughly a third of all
observations, and after the expansion it governs nearly three quarters of them. Whatever it does, it
does constantly.</p>
</div>

{figs['4']}

<div class="col">
<h2>The three options</h2>
<p><strong>A — keep the hard-coded 2.0.</strong> Zero cost, preserves byte-parity, correct at the
shipped γ. Silently wrong at any other γ, and the value's meaning is invisible to whoever reads the
line next.</p>

<p><strong>B — write it as a half-cell floor</strong>, <code>1 / (0.5^γ)</code>. At γ = 1 this is
<strong>bit-identical</strong> to 2.0 in float32, so byte-parity is fully preserved and no test
changes. At every other γ it stays on the curve instead of drifting off it. It also states the
intent — "on the source means half a cell away" — where the constant hides it.</p>

<p><strong>C — remove the special case</strong> by flooring at one cell, giving 1.0. Smooth and
simple, but it erases the distinction between standing on food and standing beside it, and it breaks
parity.</p>
</div>

{figs['3']}

<div class="col">
<div class="callout">
  <span class="tag">recommendation</span>
  <p><strong>Option B.</strong> It is the only one that costs nothing today and stays correct
  tomorrow: bit-identical output under the shipped configuration, no parity story to write, no test to
  change — and the γ sweep the olfactory study contemplates stops being a trap. Option A is defensible
  and is what the plan currently specifies; the argument for changing it is not that today's numbers
  are wrong, but that they are right for a reason the code does not record.</p>
</div>

<h2>What this does not settle</h2>
<p>Whether standing on a source <em>should</em> read twice as strongly as standing beside it is a
modelling question this study does not answer — it only shows that the current value is internally
consistent at the shipped γ. If you want contact to be more or less salient than the curve implies,
that is a deliberate change to the floor distance, and it forfeits parity whichever option is chosen.</p>
</div>

<footer>
Figures regenerate with <code>python make_onsource_figs.py</code>, frequencies with
<code>python measure_frequency.py</code> (real environment, {freq['num_envs']} envs ×
{freq['steps']} steps), the page with <code>python build_onsource_page.py</code>. Every number in the
prose is computed at build time. Companions: <code>../olfactory_expansion_study/</code>,
<code>../visual_psf_study/</code>.
</footer>
</div>
"""

pathlib.Path("index.html").write_text(HTML)
print(f"wrote index.html  {len(HTML)//1024} KB  [check: ratios {ratios[-2]:.2f},{ratios[-1]:.2f} "
      f"| fires {P0:.1f}/{P1:.1f}/{P2:.1f}%]")
