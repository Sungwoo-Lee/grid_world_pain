---
title: "Neuromodulated FiLM vs plain RecurrentPPO on the long stage-4 curriculum — did the modulator help?"
topic: basic_curriculum
status: active
created: 2026-06-27
last_updated: 2026-06-27
phase: 1
wandb_tag: "rppo_basic_curriculum_longL4 / rppo_nmn_film_curric_longL4_n114"
---

# Neuromodulated FiLM vs plain RecurrentPPO on the long stage-4 curriculum

## Headline finding (plain language)

We trained two otherwise-identical agents straight through the same five-world
difficulty ladder (easy 5×5 room → hard 10×10 world with a fast, far-seeing
predator), with the last and hardest world ("stage 4", the far-sight predator)
stretched out so the agent spends almost all of training there. Performance is
measured in **survival steps** — how many steps the agent lives before it starves
or is killed — never reward.

- **Agent A — "plain":** a vanilla recurrent PPO agent, no neuromodulation.
- **Agent B — "NMN FiLM":** the same agent plus a *neuromodulator* — a small extra
  network that watches the agent's internal state and rescales/shifts every neuron
  of the policy network on the fly (a per-neuron "FiLM" gain γ and bias β), with a
  learned "temperature" knob that controls how strongly it pushes. This is the
  `grouping_size: 1` variant — one γ/β pair **per neuron**, the most flexible (and
  least constrained) setting.

The user expected the neuromodulated agent to **beat** the plain agent on the hard
far-sight world. **It did not.** At the point where the modulated run was stopped
(50 million episodes) the plain agent was surviving ~290 steps and the modulated
agent ~276 — the plain agent was **~13 steps (~5%) ahead**. The plain run was then
left to train 16 million episodes longer and pulled further ahead to ~297 steps.

But the honest story is more interesting than "the modulator is worse." For most of
training the modulated agent **led or matched** the plain agent — it was clearly
better on the middle stages and on the first ~28M episodes of the hard stage. Then,
around 34M episodes, it suffered a **sudden destabilisation** (survival crashed from
~268 to ~180 steps over a couple of million episodes) that coincided with its
temperature knob pinning at its maximum allowed value and a burst of exploding
gradients. It partially recovered but never caught back up, while the plain agent
just kept slowly grinding upward. So the modulator is **not inert** — it is very
active, helps early, and then **over-drives itself into an instability** in the long
asymptotic grind.

**Verdict: the "NMN did not beat plain" claim is CONFIRMED** (at matched episodes
and at endpoints), and the most likely mechanism is an **over-aggressive,
under-regularised per-neuron modulator** that saturates its own temperature ceiling
and blows up its gain variance late in training — a plasticity-vs-asymptote
trade-off, not a dead modulator. This directly motivates the planned `grouping_size`
search: coarser grouping (fewer, larger modulation groups) is the obvious lever to
regularise the modulator and test whether the early-training benefit can be kept
without the late-stage crash.

> **Post-hoc framing (Mode B).** There was no pre-registered hypothesis with numeric
> pass/fail thresholds for this comparison. The research question below was posed
> *after* the runs existed. Both arms are **single-seed**, so the *magnitude* of the
> endpoint gap is uncertain; the within-run trajectory shape (lead → crash →
> partial recovery) is the more trustworthy signal. Conclusions are directional.

---

## 1. Research question

Does adding the per-neuron FiLM neuromodulator to a recurrent PPO agent **improve
survival on the hardest curriculum stage** (far-sight predator, 10×10) relative to
the identical un-modulated agent, when both are trained through the same long
stage-4 schedule?

Sub-questions:
1. Final and best survival for each arm, and how far each progressed (episode
   budget is a confound — they were stopped manually at different points).
2. Survival trajectory over training — did the modulator track, lead, or lag?
3. Did the modulator actually underperform, by how much, and is the gap inside or
   outside noise?
4. Diagnostic signatures — entropy collapse? temperature-head behaviour? value-loss
   health? FiLM γ/β magnitudes? Was the modulator inert, harmful, or neutral?

## 2. Design (as run)

| | Plain | NMN FiLM |
|---|---|---|
| WandB name | `rppo_basic_curriculum_longL4` | `rppo_nmn_film_curric_longL4_n114` |
| WandB run id | `oq2vvh8g` | `b81gyq0y` |
| Local result dir | `results/JAX_RecurrentPPO/20260623-220259_rppo_basic_curriculum_longL4` | `results/JAX_RecurrentPPO/20260624-043011_rppo_nmn_film_curric_longL4_n114` |
| Agent config | `configs/models/recurrent_ppo/recurrent_ppo.yaml` (`modulation.type: null`) | `configs/models/recurrent_ppo/recurrent_ppo_nmn_film_g1_tempceil5.yaml` |
| Modulator | none | FiLM, per-neuron (`grouping_size: 1`), temperature clip [0.5, 5.0] |
| Curriculum schedule | `configs/continual/basic_curriculum_schedule_longL4.yaml` | identical |
| Env config | identical (byte-identical per user) | identical |
| PPO hyperparameters | identical | identical |
| WandB group / job_type | `basic_curriculum` / `prod` | same |
| Node | 114 | 114 |
| Seed | single | single |

The **only** intended difference is the FiLM neuromodulator.

**Stage schedule (identical for both, confirmed from logs):** stage 0 static
predator 5×5 (ep 0–1.0M), stage 1 slow predator 5×5 (1.0–2.0M), stage 2 fast
predator 8×8 (2.0–4.0M), stage 3 predator+rabbit 10×10 (4.0–6.0M), **stage 4
far-sight predator 10×10 (6.0M → end)**. Both agents enter stage 4 at exactly 6.0M
episodes, so all stage-4 comparisons below are at matched curriculum position.

**Reference baseline (from user):** from-scratch stage-4 (no curriculum)
`rppo_basic04_farsight_n114` (run `c8cd77ft`) ≈ **261 survival steps**. Both
curriculum arms exceed this.

> **Budget confound (important).** The runs were stopped manually at different
> points: plain reached **66.0M episodes** (18.1B env steps); NMN FiLM reached only
> **50.0M episodes** (13.2B env steps) — 16M episodes / 5B steps fewer. Endpoint
> comparison is therefore unfair to NMN; the matched-episode comparison (§3) is the
> one to trust.

## 3. Results

### 3.1 Survival headline (survival steps)

| Statistic | Plain | NMN FiLM |
|---|---|---|
| Final stopped at | 66.0M ep / 18.1B steps | 50.0M ep / 13.2B steps |
| Final survival (last-1M-ep mean) | **296.8** | **276.3** |
| Best 50-pt rolling mean (stage 4) | 297.0 (@67.4M) | 275.8 (@50.0M) |
| Survival at **matched 50.0M ep** | **289.6** | 276.3 |
| Survival at plain's own end (67.6M) | 296.7 | — (not run that far) |

**Matched-episode gap at NMN's stop (50.0M): plain 289.6 vs NMN 276.3 → plain ahead
by ~13 steps (~4.6%).** Unmatched endpoint gap: 296.8 vs 276.3 → ~20 steps, but ~7
of that is the extra 16M-episode budget.

### 3.2 Survival trajectory (matched-episode windows, survival steps)

| Episode window | Stage | Plain | NMN FiLM | NMN − Plain |
|---|---|---|---|---|
| 0–1M | 0 static | 463.9 | 462.3 | −1.5 |
| 1–2M | 1 slow | **100.6** | **261.9** | **+161** |
| 2–4M | 2 fast | 195.1 | 362.0 | **+167** |
| 4–6M | 3 rabbit | 368.5 | 404.2 | +36 |
| 6–10M | 4 far-sight | 191.5 | 236.4 | +45 |
| 10–15M | 4 | 228.0 | 235.2 | +7 |
| 15–20M | 4 | 246.3 | 262.9 | +17 |
| 20–25M | 4 | 254.9 | 266.0 | +11 |
| 25–30M | 4 | 264.4 | 268.1 | +4 |
| 30–35M | 4 | 271.5 | 256.2 | −15 |
| 35–40M | 4 | 278.6 | **210.1** | **−68** |
| 40–45M | 4 | 283.7 | 256.0 | −28 |
| 45–50M | 4 | 288.4 | 274.3 | −14 |
| 50–55M | 4 | 289.3 | 272.2 | −17 |
| 55–60M | 4 | 291.5 | — | — |
| 60–68M | 4 | 295.4 | — | — |

Three regimes:
- **Stages 1–3 + early stage 4 (1M–30M): NMN leads.** Strikingly, the plain agent
  *collapsed* on stage 1 (slow predator) to the ~100-step starve floor, while NMN
  held 262; NMN also doubled plain on stage 2. Through the first ~24M episodes of
  stage 4 NMN led by 4–45 steps.
- **The crash (~34–40M): NMN destabilises.** Fine resolution: 32–34M plain 271.9 /
  NMN 269.4 (still even); **34–36M plain 275.7 / NMN 180.8 (−95)**; 36–38M NMN 199;
  38–40M NMN 226. A discrete ~95-step survival crash, not gradual drift.
- **Partial recovery + plain pulls away (40–50M):** NMN climbs back to ~276 but
  never regains the trajectory; plain monotonically grinds 288 → 297.

### 3.3 Diagnostic signatures

**Temperature head — saturates at the ceiling.** The modulator's learned temperature
(`modulator/temperature_mean`) rises monotonically through training: 0.8 → 1.9 → 3.3
→ 4.0 → 4.7 → 4.99 → **5.00 and pinned** for the entire late run. The last 10 logged
values are exactly 5.0; **34% of all stage-4 points sit at the [0.5, 5.0] clip
ceiling**, and 100% of the final phase. The controller wants *more* modulation
strength than the clip allows — the temperature head is **railed**.

**FiLM γ/β magnitudes — large and ballooning.** Per-neuron gains/shifts are far from
the identity transform (γ=1, β=0) and grow over training, with variance exceeding the
mean:

| Window | γ_uni mean | γ_uni **std** | β_uni mean | β_uni std |
|---|---|---|---|---|
| 6–10M | 1.28 | 3.21 | −1.06 | 3.06 |
| 25–30M | 1.51 | 3.40 | −1.17 | 2.85 |
| 35–40M | **2.99** | **5.53** | −1.86 | 4.05 |
| 45–50M | 2.66 | 4.72 | −2.48 | 3.97 |

By late training the modulator is scaling features by ~2.5× on average with a
per-neuron std of ~5 and shifting them by ~−2 — an aggressive, high-variance
transform. The γ-std blow-up coincides in time with the temperature pinning and the
survival crash.

**Gradient spikes.** Both runs show occasional pre-clip grad-norm spikes, but NMN's
are larger and time-aligned with the crash: windowed-mean `loss/grad_norm` (and
`modulator/grad_norm`) spike to ~3.1M around 10–15M and ~8,160 around 35–40M — the
latter exactly overlapping the survival crash window. (These are single-point
outliers inflating window means; gradient clipping caps the actual update, so they
flag instability rather than prove a single fatal step.)

**Policy entropy — no collapse, NMN marginally tighter.** `loss/entropy` sits ~−0.54
(plain) vs ~−0.48 to −0.51 (NMN) in the steady state — comparable, with NMN slightly
lower-magnitude (marginally more deterministic) late. Neither run shows an
entropy-collapse signature.

**Value/critic loss — healthy in both.** `loss/value` ≈ 0.22–0.25 steady-state for
both, no critic divergence (the one 1.44 plain spike at 6–10M is the stage-3→4
transition). Reward and food-eaten move consistently with survival.

## 4. Analysis

**Did NMN underperform? Yes, asymptotically — but it is the opposite of inert.** The
modulator is highly active (temperature railed at its ceiling, γ scaled ~2.5× with
std ~5). It *helped* early: it prevented the stage-1 collapse that sank the plain
agent and led through the first ~28M episodes of stage 4. The underperformance is
confined to the **late asymptotic regime** and is precipitated by a **discrete
destabilisation event around 34M episodes** — a ~95-step survival crash that
coincides with (a) the temperature head pinning at its clip ceiling, (b) γ per-neuron
variance ballooning to ~5, and (c) an ~8,000 gradient-norm spike. After the crash NMN
recovers only to ~276 while the simpler plain network keeps climbing to ~297.

**Is the gap inside or outside noise?** Both arms are single-seed, so the ~13-step
matched-episode endpoint gap **cannot be cleanly attributed to architecture vs. seed/
run noise** — we have no seed dispersion to bound it. However, the within-run crash
(~95 steps, time-locked to the temperature-pin + gradient-spike) is far too large and
too structured to be sampling noise; it is a real dynamical event in the NMN run. So
the trustworthy claim is not "NMN is 13 steps worse" but "NMN under-regularisation
caused a late instability that cost it the lead it had earned."

**Most likely mechanism.** The `grouping_size: 1` (per-neuron) FiLM modulator is
*over-parameterised / under-regularised* for the long stage-4 grind. Its capacity
lets it adapt fast early (good for curriculum transfer / plasticity) but, with the
temperature able to climb to its ceiling and per-neuron γ/β unconstrained, it
over-drives the policy network, blows up its gain variance, and tips into a gradient
instability from which it only partially recovers. The plain network has no such
high-gain feedback path and so trades slower early adaptation for a stable,
monotone asymptote. This is a **plasticity-vs-asymptotic-stability trade-off**, and
it rhymes with the project's existing "plasticity-not-budget" curriculum finding.

**Confounds and caveats.**
- Single seed per arm; budget mismatch (16M episodes) handled by matched-episode
  comparison but seed variance is unbounded.
- Window means in §3.2 are over millions of episodes (tight SEM within-run) but that
  does not translate to seed-level confidence.
- The temperature clip ceiling (5.0) is itself a railed hyperparameter — the
  saturation could partly reflect a too-low ceiling rather than (or in addition to)
  too much capacity. The `grouping_size` search should not change capacity in
  isolation without watching the temperature.

## 5. Conclusions

1. **Claim confirmed.** The neuromodulated FiLM agent did **not** beat the plain
   agent on the hard far-sight stage — plain led by ~13 steps at matched episodes
   (50.0M) and extended to ~20 steps by its longer endpoint.
2. **Not because the modulator was inert** — it was the most active component in the
   system, and it *helped* on stages 1–3 and early stage 4 (it prevented the stage-1
   collapse that sank the plain agent).
3. **The loss came from a late, structured instability**: temperature pinned at its
   clip ceiling, per-neuron γ variance ballooned, a gradient spike hit, and survival
   crashed ~95 steps around 34M episodes, only partially recovering.
4. **Implication for the planned `grouping_size` search:** the per-neuron (group=1)
   setting is the maximum-capacity / least-regularised corner and it is exactly where
   the instability appeared. Coarser grouping (fewer, larger groups) is the
   first-order lever to test whether the early-transfer benefit survives without the
   late crash. **Recommend also logging/monitoring the temperature ceiling** — it is
   currently railed, so the search confounds "group size" with "temperature
   headroom" unless the ceiling is held or swept deliberately.

## 6. Metrics Requested

| Subfield | Content |
|---|---|
| **Metric** | `modulator/temperature_frac_at_ceiling` (fraction of the batch whose temperature sits within ε of the clip bound), unitless [0,1]. |
| **Why now** | The temperature head is railed at 5.0; mean alone hides whether it is uniformly pinned or bimodal. A ceiling-fraction scalar makes saturation a first-class, sweepable diagnostic for the `grouping_size` search instead of something inferred post-hoc from min/mean/max coincidence. |
| **Where it'd live** | The FiLM modulator forward in `src/` (the module that emits `modulator/temperature_{min,mean,max}` — same logging site). |
| **Cost** | Cheap (one scalar reduction per logging step). |

| Subfield | Content |
|---|---|
| **Metric** | `modulator/grad_norm_preclip` already logged, but add `modulator/update_clipped_frac` (fraction of updates where the global grad norm exceeded the clip threshold). |
| **Why now** | The crash is time-locked to grad spikes; window-mean grad-norm is dominated by single outliers and cannot tell a one-off spike from sustained clipping. A clipped-fraction scalar distinguishes "rare transient" from "chronic instability" — central to diagnosing whether group size fixes the stability problem. |
| **Where it'd live** | The PPO update / optimizer step in `src/` where `max_grad_norm` clipping is applied. |
| **Cost** | Cheap (one scalar per update). |

## 7. Related Issues / Follow-ups

- **TODO (re-run as pre-registered):** this comparison should be re-run as a
  pre-registered design with **≥3 seeds per arm** and a **matched episode budget**
  before the ~13-step endpoint gap is treated as a real effect size. Surfaced to the
  user.
- **Feeds:** the planned hyperparameter search over FiLM `grouping_size` — use the
  temperature-saturation + γ-variance-blow-up signatures here as the things the
  sweep must move. If accepted, route the two requested metrics via `feature-workflow`
  (`senior-developer` → `developer`).
- **Related docs:** [[basic_curriculum_continual_result]] (the single continual run
  vs from-scratch baselines), [[basic_curriculum_convergence]] (from-scratch
  per-stage convergence), [[basic_curriculum]] (study overview).

## 8. Manifest / provenance

- Local history parsed offline from the `.wandb` transaction logs (no web API):
  `wandb/run-20260623_220300-oq2vvh8g/run-oq2vvh8g.wandb` (plain),
  `wandb/run-20260624_043012-b81gyq0y/run-b81gyq0y.wandb` (NMN FiLM).
- Working extractions: `tmp/20260627_nmnfilm_longL4_{plain,nmn}_hist.jsonl`,
  parser `tmp/_wb_read.py`, analyses `tmp/_wb_cmp.py` + `tmp/_wb_fine.py`.
- Summary snapshots: each run's `files/wandb-summary.json` and `files/output.log`.
