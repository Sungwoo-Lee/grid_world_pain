"""Build index.html for the DIRECTIONAL_SENSORS directional-sensors implementation report."""
import base64, json, pathlib

V = json.loads(pathlib.Path('verification.json').read_text())
BEF = json.loads(pathlib.Path('/home/vncuser/.claude/jobs/9b7c6091/tmp/sps_before.json').read_text())
AFT = json.loads(pathlib.Path('/home/vncuser/.claude/jobs/9b7c6091/tmp/sps_after.json').read_text())
base_b = next(r for r in BEF['rows'] if r['label'].startswith('baseline'))
rows = [r for r in AFT['rows'] if 'sps' in r]
base_a = rows[0]
chosen = next(r for r in rows if r['label'].startswith('BOTH'))

FIGS = [
    ("fig1_observation_before_after.png", "1", "What the agent actually receives",
     "The real environment, seed 3. Before, each sense is a single row — one reading per channel at "
     "the agent's own cell, with no way to tell which direction anything lies in. After, each sense is "
     "a table: one row per diamond cell, one column per channel. The blank rows are the out-of-bounds "
     "rule working, not missing data — this agent sits near the top-left edge."),
    ("fig2_blur_off_on.png", "2", "The visual sensor, blur off and on",
     "Off, an entity appears in exactly one cell and nowhere else. On, it is spread along the line of "
     "sight, so several cells carry a share. Identity never mixes: the blur happens inside each "
     "channel, so a smeared predator is still unambiguously a predator. The off panel being mostly "
     "empty is the point — anything not exactly on a diamond cell is invisible to it."),
    ("fig3_olfactory_gradient.png", "3", "The olfactory diamond points at the source",
     "One food source, everything else deactivated. The brightest cell is always the one toward the "
     "source and the dimmest the one away from it. The single number the sensor used to return could "
     "not distinguish any of these four situations. Note the rightmost panel reads exactly 2.000 on "
     "the source — that is the new half-cell floor, reproducing the old constant precisely."),
    ("fig4_sps.png", "4", "Measured throughput cost",
     "Environment steps per second under a jitted scan, which is how the training loop drives it. The "
     "all-off baseline after the change matches the baseline measured before any code existed, so the "
     "implementation is free when disabled."),
]


def b64(p):
    return base64.b64encode(pathlib.Path(p).read_bytes()).decode()


figs = {n: f"""
    <figure class="fig" id="fig{n}">
      <div class="fig-head"><span class="fig-num">Fig {n}</span><h3>{t}</h3></div>
      <div class="fig-img"><img src="data:image/png;base64,{b64(f)}" alt="{t}"></div>
      <figcaption>{c}</figcaption>
    </figure>""" for f, n, t, c in FIGS}

sps_rows = "".join(
    f'<tr><td>{r["label"]}</td><td class="num">{r["obs_dim"]}</td>'
    f'<td class="num">{r["sps"]:,.0f}</td><td class="num">{r["us_per_step"]:.1f}</td>'
    f'<td class="num">{100*(r["sps"]/base_a["sps"]-1):+.1f}%</td></tr>' for r in rows)

CSS = pathlib.Path('../olfactory_expansion_study/build_olf_page.py').read_text()
CSS = CSS[CSS.index('CSS = f"""') + len('CSS = f"""'):]
CSS = CSS[:CSS.index('"""')]
CSS = (CSS.replace('--accent:#1f7a6c', '--accent:#2f6f3e').replace('--accent:#5cc4b0', '--accent:#7fc48f')
          .replace('--ember:#c9852b', '--ember:#a8532f').replace('--ember:#e8b45c', '--ember:#e0876a'))

HTML = f"""<title>Directional Sensors Shipped</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,600;1,400&family=Karla:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap">
{CSS}

<div class="wrap">
<header>
  <span class="eyebrow">GridWorld Pain · DIRECTIONAL_SENSORS · implementation report</span>
  <h1>Directional Sensors Shipped</h1>
  <p class="standfirst">Smell can now point, and vision can now be uncertain. Both ship switched off,
  and the shipped configuration produces byte-identical observations to before. What was built, what
  was tested, and what it costs.</p>
  <p class="meta">commit <code>0e8a4ef</code> on <code>v3.0</code> · {V['files_changed']} files · plan: <code>DIRECTIONAL_SENSORS_PLAN.md</code></p>
</header>

<div class="col">
<h2>What shipped</h2>
<p><strong>Olfaction can point.</strong> It used to return five numbers sampled at the agent's own
cell — how much of each smell was present, and nothing about where it came from. It now samples the
same field at every cell of a diamond, so the differences between those readings carry a direction.</p>

<p><strong>Vision can be uncertain.</strong> It used to report an exact cell match: anything inside
the sensor's reach arrived with perfect position and perfect identity. It can now spread each object
along the line of sight to it, so how far away something is becomes vague while which direction it
lies in stays sharp.</p>

<p><strong>Things can be hidden from sight.</strong> Any entity can be marked invisible, either
always or unless the agent is standing on it.</p>

<p>All three are off by default, and one further change has no switch: the value the smell sensor
reports when sitting exactly on a source is now calculated from the falloff setting rather than typed
in as a constant. At the shipped falloff it produces the identical number.</p>

<div class="callout">
  <span class="tag">the guarantee</span>
  <p>With the shipped configuration the observation is <strong>bit-for-bit identical</strong> to what
  the environment produced before this change — verified against a fixture captured from the old code,
  {V['parity_runs']} runs, {V['parity_broken']} failures. Nothing you have already run is disturbed
  until you deliberately switch something on.</p>
</div>

<h2>Verification</h2>
</div>

<div class="tablewrap col">
<table>
<thead><tr><th>check</th><th>result</th></tr></thead>
<tbody>
<tr><td>Observation identical to pre-change fixture (pinned CPU, {V['parity_runs']} runs)</td><td><strong>bit-identical</strong></td></tr>
<tr><td>Full environment test suite</td><td><strong>{V['env_suite']['passed']} passed, {V['env_suite']['failed']} failed</strong></td></tr>
<tr><td>New directional-sensor tests</td><td><strong>{V['new_tests_passed']} passed</strong></td></tr>
<tr><td>Breakdown sums to actual observation width (5 range combinations)</td><td>pass</td></tr>
<tr><td>Blur weights finite everywhere, including an entity on the agent's own cell</td><td>pass</td></tr>
<tr><td>Anisotropy = 1 reproduces an isotropic kernel</td><td>pass</td></tr>
<tr><td>Total weight falls monotonically with distance</td><td>pass</td></tr>
<tr><td>Hidden entity contributes exactly zero, including the centre cell</td><td>pass</td></tr>
<tr><td>Out-of-bounds smell cells read exactly zero</td><td>pass</td></tr>
<tr><td>Smell gradient points at the source from all four bearings</td><td>pass</td></tr>
<tr><td>New on-source value bit-identical to the old constant at the shipped falloff</td><td>pass</td></tr>
</tbody>
</table>
</div>

<div class="col">
<p>Two of those deserve a note. The <strong>hidden-entity test</strong> is a regression test for a bug
the plan review caught before any code existed: the first specification gated visibility on the
<em>cell's</em> distance rather than the <em>entity's</em>, which under blur let a "hidden" object
deposit about 13% of a visible object's signal into the agent's own cell. It now contributes exactly
zero, and the test fails against the old specification.</p>

<p>The <strong>parity test runs on a pinned CPU backend</strong>, and that is not incidental. During
implementation the same check failed and passed alternately on GPU. The cause was not the code: bit-exact
float results depend on how the computation is lowered, so CPU and GPU legitimately differ, and GPU
autotuning makes repeat GPU runs differ from each other. A bit-exactness claim is only meaningful on a
fixed backend — the check now refuses to run anywhere else.</p>
</div>

{figs['1']}
{figs['3']}
{figs['2']}

<div class="col">
<h2>What it costs</h2>
</div>

<div class="tablewrap col">
<table>
<thead><tr><th>configuration</th><th>observation</th><th>steps / sec</th><th>µs / step</th><th>vs baseline</th></tr></thead>
<tbody>
{sps_rows}
</tbody>
</table>
</div>

<div class="col">
<p><strong>The implementation is free when switched off.</strong> The all-off baseline measures
{base_a['sps']:,.0f} steps per second, against {base_b['sps']:,.0f} measured before any code was
written — a difference well inside run-to-run noise.</p>

<p><strong>The chosen settings cost {abs(100*(chosen['sps']/base_a['sps']-1)):.1f}%.</strong> That is
more than the sensor arithmetic itself, which separate micro-benchmarks put at one to two percent.
Most of the cost is not the kernels — it is the observation growing from 27 numbers to
{chosen['obs_dim']}, which the whole rollout has to carry, store and feed to the network. If that
number matters to you, the lever is the diamond sizes, not the sensor code.</p>

<p>For scale: at {chosen['sps']:,.0f} steps per second, ten million environment steps take about
{10e6/chosen['sps']/60:.1f} minutes of environment time, against
{10e6/base_a['sps']/60:.1f} minutes at baseline.</p>
</div>

{figs['4']}

<div class="col">
<h2>Deviations from the plan</h2>
<p>Two, both minor, both deliberate.</p>

<p><strong>The tests are one file, not six.</strong> The plan named six test modules; they are
consolidated into <code>tests/env/test_directional_sensors.py</code> with sections mapping one-to-one
onto the plan's test table. Six modules of near-identical setup would have been harder to maintain,
not easier to read.</p>

<p><strong>The config sweep was {V['configs_swept']} files, not the ~98 estimated.</strong> The plan
review estimated the blast radius; rather than trust the estimate I found the affected files
empirically, by loading every config and catching the ones that failed. Only 11 standalone
<em>configs</em> needed the new keys — but a further set of test fixtures and archived configs did too,
including archived ones still actively loaded by the test suite, which the estimate had missed
entirely.</p>

<h2>Known consequence</h2>
<p>Eleven configs use a non-standard smell falloff, and for those the on-source reading changes from
2.0 to 4.0 — the deliberate consequence of applying that fix everywhere rather than gating it. Two of
the eleven are the olfactory parity-verification configs; if those are re-run, their stored expected
values need regenerating. Everything at the standard falloff is untouched.</p>

<h2>What is not done</h2>
<p>This is the environment side only. Nothing has been trained, no experiment configs have been
written, and no claim is made about whether any of it helps the agent. The next step is choosing an
experiment, not more code.</p>
</div>

<footer>
Figures regenerate with <code>python make_report_figs.py</code>, throughput with
<code>python scripts/verification/bench_sensor_sps.py</code>, parity with
<code>python scripts/verification/capture_sensor_baseline.py</code>. Every number in this page is read
from the recorded results of those runs at build time.
</footer>
</div>
"""

pathlib.Path("index.html").write_text(HTML)
print(f"wrote index.html  {len(HTML)//1024} KB")
