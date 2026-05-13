---
title: Nature MI Paper Framing — Cross-Professor Synthesis
topic: paper-framing / cross-professor synthesis
status: draft
mode: cross-professor synthesis
author: research-postdoc
created: 2026-05-07
last_updated: 2026-05-07
synthesises:
  - docs/project/ideas/nature_mi_paper_framing.md
  - docs/project/concepts/pain_vs_nociception_construct.md
  - docs/project/concepts/active_inference_hypervigilance.md
  - docs/project/directions/architecture_for_pain_computation.md
  - docs/project/critiques/biological_plausibility_of_three_site_modulation.md
---

# Nature MI Paper Framing — Cross-Professor Synthesis

> **Thesis of this memo.** Four professor memos converge on a single picture
> that is *narrower and more defensible* than the original framing memo
> proposed: the project ships a paper about an **interoceptive ascending
> modulatory analog producing channel-selective precision-weighting under
> bodily threat** — earning the *substrate-for-pain-like-behaviour*
> register through a three-control dissociation (C0/C1/C2), an architecture
> that is *strictly smaller* than the current Phase-3 plan, and a tightened,
> theory-derived G2'/G1 fingerprint. The strong "pain computation" headline
> and the H5 chronic-pain claim require an *expensive* upgrade
> (two-modulator T/P split + opioid descending head) that no professor
> rules out but no professor calls cheap, and that the paper does *not*
> require for a defensible submission. The user's authorize/decline
> decision turns on three points: (i) the headline register, (ii) whether
> H5 takes the upgrade or the downgrade, and (iii) whether to commit now
> to the simpler precision-head architecture (RL-BDL) or run the current
> blended architecture in parallel.
>
> This memo is the document the user reads to authorize a rewrite. It does
> not modify `project_plan.md`; it scopes three options for that rewrite.

---

## 1. Convergences — the load-bearing claims and changes

These are the points where two or more professors say the same thing and the
project should treat the convergence as load-bearing. Each convergence is
named by *who agrees, on what, and what change to the plan it implies*.

### 1.1 Sober register for the headline

**Who agrees.** Pain-modeling §4.4. Bayesian-brain §5. Neuromodulation §5.

**On what.** The phrase "pain computation" is not licensed *yet* — it is a
construct claim that requires (a) C0/C1/C2 dissociation on the channel-
selective fingerprint, *and* (b) cross-rung transfer of the modulator-state
representation. Until both hold, the title and abstract should use the
sober register.

**Reconciled register choice.** The three professors actually offer *three*
sober registers, not one. The synthesis is to use them in different parts of
the manuscript:

| Surface | Register | Source |
|---|---|---|
| Title / abstract / cover letter | **"Interoceptive neuromodulation as a substrate for pain-like behaviour"** | Pain-modeling §4.4; Bayesian-brain §5 |
| Body, mechanism sections | **"An interoceptive ascending modulatory analog producing coordinated perceptual–mnemonic–policy reweighting under bodily threat"** | Neuromodulation §5.2 (the "R-bio" register) |
| Discussion | "Pain computation" foreshadowed *only* if Option B (§4 below) is taken and all three controls + transfer pass | Pain-modeling §4.4; Bayesian-brain §5 |

**What changes in the plan.** Drop "full pain syndrome" everywhere
(neuromodulation §7.1; pain-modeling §4.3). Rewrite project_plan.md §1 and
§3 to use the body-section register; rewrite §2's "hypervigilance" framing
to match the four-property fingerprint of §1.4 below.

### 1.2 Three controls — C0, C1, C2 — and why each is load-bearing

**Who agrees.** Pain-modeling §3 (proposes the trio). Bayesian-brain §3.2(c)
(certifies C1 as the *theoretical spine* of Claim 2(b) — a 1-D EMA cannot
in principle reach the channel-selective fixed point). RL-BDL §1.6 and §4
(specifies the C1 architecture concretely; folds C0/C1/C2 into the run
budget as 3 cells).

**Reconciled spec.**

| Control | What it is | What it rules out |
|---|---|---|
| **C0** — pure nociception | Damage signal in obs, no modulator architecture, standard recurrent PPO/Dreamer | Sets survival-benefit floor; failure = Claim 2(a) drops, fall back to 2(b) |
| **C1** — slow-input mimic | Trainable scalar EMA on $o_{\text{noc}}$, broadcast through per-injection learned read-outs $w \cdot m_t + b$; same auxiliary loss; same seed budget (RL-BDL §4) | FP-2 (smoothing-as-pain). Theoretically **cannot** reach channel-selective $\gamma$ because $m_t$ is one number and per-channel post-injury *rate-of-rise* differences are forced to zero (Bayesian-brain §3.2(c)) |
| **C2** — interoception-masked | Full modulator architecture, modulator's *input* has interoceptive channels (injury, satiation, nutrition) zeroed | FP — "architecture only, no interoceptive read"; the AI memo's pre-registrable cross-domain $r$ contrast is computed *between* modulated agent and C2 |

**The load-bearing dissociation metric (the spine of Claim 2(b)).** The
modulated agent must dissociate from C1 on **cross-channel rate-of-rise
difference** $\Delta_t \gamma_{\mathcal{T}}^{(i)} - \Delta_t
\gamma_{\mathcal{T}}^{(j)}$ for two threat channels with different
injury-scaled noise profiles (RL-BDL §4.4). Not on temporal envelope, not
on within-trial $r$ (C1 fakes both). This single metric is what
operationalises "pain is computation, not smoothing".

**What changes in the plan.** Add §2bis "Construct-validity controls
(C0/C1/C2)" specifying all three as standing comparison conditions for
every phase from Phase 2 onward. Phase 4 is gated on dissociation, not on
G1+G2 alone (pain-modeling §4.2).

### 1.3 Minimum rung set {1, 3, 5}

**Who agrees.** Pain-modeling §5.1 (rungs 1, 3, 5 are construct-mandatory;
2, 4 are methodology-flavour). RL-BDL §3.2 (concurs; recommends the
canonical-rung 63-run factorial with cross-rung transfer to {1, 5}).

**Reconciled spec.**

| Rung | Description | Necessary for | Cell coverage |
|---|---|---|---|
| **1** | Empty grid + injury, no predators, no food | C0/reflex floor; FP-3 rule-out (no drive conflict) | Headline + C0 + C1 only |
| **3** | + predators with heterogeneous threat profile + canonical heterogeneous noise preset | Channel-selective fingerprint test; the 11-cell factorial + C0/C1/C2; AI vs. heteroscedastic-BDL regional contrast (Bayesian-brain §2.3) | Full factorial (the 63-run cell of RL-BDL §2.3) |
| **5** | + slow injury dynamics (long recovery window) | H5 / chronicity test or its downgraded version (modulator-inertia analog of post-injury caution persistence); timescale-ordering pre-registration | Headline + C0 + C1 only |

**What changes in the plan.** Replace the current Phase 1 "tune the noise
landscape" with a Phase-1' that sweeps the noise preset *on rung 3* and
verifies it on rungs 1 and 5. Add §4bis "Environment ladder" naming the
three rungs explicitly. The methodology-paper figure (Claim 5) is the
rung × signature matrix; rungs 2 and 4 are listed as planned extensions
post-submission, not as load-bearing for the ladder figure.

### 1.4 Tightened G2 fingerprint — single pre-registrable G2'

**Who agrees.** Pain-modeling §2.2 (proposes operational thresholds:
$d \geq 0.5$, lag $\leq 5$ steps, $r \geq 0.3$, $\Delta\gamma \geq 0.2$,
policy-shift $\geq 3 \times$ recovery half-life). Bayesian-brain §4
(derives a sharper four-property AI fingerprint with theory-licensed
direction and timescale ordering, separating *load-bearing* from
*calibration* numbers).

**Reconciliation.** Bayesian-brain refines pain-modeling's thresholds:
where pain-modeling proposed round numbers, Bayesian-brain identifies
which are theory-derived (sign, direction, ordering) and which are
calibration-licensed (the magnitudes 0.2 and 0.3). The synthesis
G2' to pre-register, in order of decreasing theoretical force, is:

1. **Sign** of all three injection-site responses post-injury:
   $\gamma_{\mathcal{T}} \uparrow$, $z_{\text{memory}} \downarrow$
   (retention), $T_\pi \downarrow$. *Load-bearing; falsification target.*
2. **Channel selectivity in $\gamma$**: $\gamma_{\mathcal{T}} -
   \gamma_{\mathcal{N}} > 0$ post-injury, magnitude derived from the
   canonical preset's $\Sigma_i(s_t)$ via $\Delta\gamma \propto
   \log(\Sigma_{\mathcal{N}}/\Sigma_{\mathcal{T}})$ — *not* the round
   0.2 number.
3. **Cross-domain zero-lag correlation contrast** between modulated
   agent and C2 (interoception-masked): the modulated agent's $r$
   exceeds C2's $r$ by $\geq 0.2$. (Pain-modeling's $r \geq 0.3$ is
   replaced because C1 will fake high *absolute* $r$.)
4. **Timescale ordering** $\tau_A < \tau_B < \tau_C$ with each ratio
   $\geq 2$. *Architecturally a predicate* (neuromodulation §4.2) —
   requires the per-injection-site read-out filter upgrade of §1.6
   below, otherwise this property is not testable on the current
   single-GRU.
5. **C1 dissociation**: cross-channel rate-of-rise difference
   $\Delta_t \gamma_{\mathcal{T}}^{(i)} - \Delta_t
   \gamma_{\mathcal{T}}^{(j)} > 0$ (modulated) and $\approx 0$ (C1).
   *The construct-spine metric.*

Lag is bounded: peak cross-correlation at lag $\leq \min(5, 2\tau_{\text{mod}})$.

**What changes in the plan.** Rewrite G2 (project_plan.md §4) to G2' as
the five quantities above, with items 1, 2, 4, 5 as
*qualitative theoretical predictions* (falsification meaning, no
multiple-comparisons inflation) and item 3 as the only quantitative
contrast.

### 1.5 G1 restated as channel-rank non-degeneracy — cheap diagnostic

**Who agrees.** Bayesian-brain §3.2(a) proposes G1 holds iff the per-
channel residual variance ratio on the LayerNorm baseline is non-
degenerate (max/min ratio $> \tau$, $\tau$ derivable from the canonical
preset). RL-BDL §5 specifies the cheap implementation: a *frozen-head*
reconstruction MLP attached to the LayerNorm baseline, per-modality MSE
conditioned on `injury > 0.5` vs. `injury < 0.1`, $\sim$1 hour wallclock
per noise preset, no architectural training.

**Reconciliation.** RL-BDL is the right operationalisation of Bayesian-
brain's theoretical statement. The two refinements RL-BDL adds (state-
conditional, on a reconstruction head not on raw $o$) are practical and
should be adopted.

**What changes in the plan.** Replace G1 (project_plan.md §4) with G1':
"per-modality residual MSE max/min ratio under high-injury > 2 on the
LayerNorm baseline with frozen reconstruction head, on the canonical
noise preset". Add the diagnostic as a Phase-1' deliverable; if no noise
preset clears the threshold, retune the preset (this is the single
loudest *go/no-go* check in the project).

### 1.6 Per-injection-site read-out filters — the cheap shared upgrade

**Who agrees.** Neuromodulation §4.2 option (1): give each injection
site's read-out an explicit low-pass filter on $h_{\text{mod}}$ with a
learnable time constant. Bayesian-brain §4.2 timescale-ordering
prediction is unlocked by this change. Neuromodulation §7.2 lists this
as recommendation 3 ("do this *regardless* of whether the bigger
upgrades are taken; it is small and unlocks the timescale-ordering
pre-registration"). RL-BDL §1.5 does not require this but does not
oppose it (the FiLM heads are already independent injection paths;
adding a learnable temporal filter on each is a minor add).

**Reconciliation.** This is the cheapest architectural upgrade that any
professor recommends; it is biologically defensible (post-synaptic
integration time of the modulator's target population differs across
A/B/C — neuromodulation §4.2); it converts the Bayesian-brain timescale
ordering from "trivially false on a single GRU" to a measurable property
of the trained model; and it does not affect the AI fingerprint's
fixed-point structure (neuromodulation §7.2 confirms).

**What changes in the plan.** Add to project_plan.md §3 (architecture):
"Per-injection-site low-pass filter on $h_{\text{mod}}$ with learnable
time constant per site." This is the cheap upgrade that goes into
*both* Option A and Option B (§4 below).

---

## 2. Tensions — points where the user must decide

These are the points where two or more professors disagree, or where their
recommendations cannot all be taken. Each tension is named, the disagreement
is spelled out, and a recommended resolution is offered.

### 2.1 Architecture direction — the multiplicative blend

**The disagreement.** RL-BDL §1.4 recommends *removing* the precision-gating
blend $\pi_{\text{gate}} \cdot x_{\text{mod}} + (1 - \pi_{\text{gate}})
\cdot x_{\text{bypass}}$ from Phase 3. The argument: the blend creates a
non-identifiability hazard between $\gamma_i$ and $\hat{\pi}_i$, so channel
selectivity can hide in either, and worse, the **C1 dissociation argument
is destroyed** because $\pi_{\text{gate}}^{(i)}$ is itself per-channel —
C1 can fake channel-selective *effective* gain through the gate even
though it cannot fake selective $\gamma$ (RL-BDL §1.4 point 2).

**The cross-check.** Does the Bayesian-brain free-energy formulation
*require* the blend? Reading Bayesian-brain §1.2 carefully:
$\partial F_t / \partial s_i = \frac{1}{2}(\pi_i \varepsilon_{i,t}^2 -1)$
is the gradient that drives $\hat{\pi}_i$ toward
$1/\mathbb{E}[\varepsilon_{i,t}^2]$. **This gradient flows via the
Kendall–Gal heteroscedastic loss, not via the multiplicative gate.** The
blend is a separate architectural choice — it implements precision-
weighted gain control as a soft mixture-of-experts, but the *gradient*
the AI fixed point needs is provided by the auxiliary loss on the
precision head, regardless of whether that head is also wired into a
gate.

**Verdict.** The blend is theoretically dispensable. RL-BDL's
recommendation is unambiguous on construct-validity grounds: the blend
makes the spine of Claim 2(b) (the C1 channel-selectivity dissociation)
non-falsifiable. **Recommendation: drop the blend in the first Phase-3
run.** Add it back as a follow-up cell only if the simpler architecture
clears G2'.

This is an *unforced error* in the current plan: the multiplicative
gate is not load-bearing for any professor's account, and it actively
breaks one professor's central argument. The RL-BDL §7.5 risk register
lists "user rejects recommendation" as low severity — the test is to
include a single "with-blend" cell at the headline factorial (5 extra
runs, +5%) so the recommendation is itself testable.

### 2.2 H5 chronic-pain claim — upgrade or downgrade

**The disagreement (more accurately, an asymmetric finding).**
Neuromodulation §3 *kills* H5 under the current single-GRU
mod_hidden_size sweep: "recurrent inertia in a single GRU is a single
$\tau_{\text{mod}}$; clinical chronic pain is a qualitative regime
change in which the controller's recovery dynamics fail and the system
enters a new attractor — sweeping `mod_hidden_size` parameterises the
speed of return to a *single* attractor, not the existence of *two*."
Pain-modeling §5.2 had already raised this concern as a recommendation
("downgrade unless transfer probe is added"). Bayesian-brain §4.2
predicts the timescale ordering but does not directly address chronic
pain. RL-BDL did not address H5.

**Two options the user has.**

**Option H5-up (the upgrade).** Take the neuromodulation §3.2 expensive
upgrade: two-modulator T/P split (§2.3) plus opioid-analog descending
head. This earns a defensible H5 claim with a pre-registrable
acute/recovery/chronic three-regime structure. **Cost (engineering):**
new modulation type, new descending-modulation head, asymmetric T/P
coupling with hysteresis, per-modulator teaching signals
(neuromodulation §6 — without separate teaching signals, the multi-
modulator analog is rhetorical even if the architecture is multi-
modulator). **Cost (runs):** the factorial expands; the H5 cell on
rung 5 is no longer 5 runs but a sub-factorial across (T-only / P-only /
T+P / T+P+opioid-head). Estimated marginal: +30–60 runs.

**Option H5-down (the downgrade).** Restate H5 as "modulator-state
inertia analog of post-injury caution persistence" (pain-modeling §5.2
verbatim). Drop "chronic pain" from the paper. The single-GRU
timescale sweep then tests what it actually tests — recurrent inertia,
not chronicity — and the paper is honest. **Cost: zero.**

**Recommendation.** Take H5-down. Three reasons. (1) The headline
register is already sober (§1.1 above) — the paper is not making a
strong "pain computation" claim, so a strong chronic-pain sub-claim is
inconsistent with the rest of the framing. (2) Neuromodulation §7.2
specifically advises "if RL-BDL judges the upgrade out of budget, the
paper should take the downgrade rather than the upgrade — it is more
defensible to ship the smaller claim than to ship the bigger
architecture incomplete." RL-BDL §7 has not flagged the upgrade as in
budget. (3) The H5-up upgrade is itself the right next paper, not a
sub-claim of this one. Shipping it incomplete costs more than deferring
it.

This is the user's biggest decision in this synthesis. If the user
disagrees and wants Option H5-up, see Option B in §4.

### 2.3 Architecture envelope — RL-BDL's smaller, neuromodulation's larger

**The disagreement (apparent, mostly resolvable).** RL-BDL §1.5 says
the smallest architecture is *strictly less* than current Phase 3 (drop
the blend; precision head as parallel auxiliary, measurement only).
Neuromodulation §3.2 says a *cheap upgrade* (per-injection-site
learnable-$\tau$ filters) is needed and an *expensive upgrade* (T/P
split + opioid head) may be needed. These are not directly contradictory
— the cheap neuromodulation upgrade adds three small filters to the
output of a system that, on RL-BDL's recommendation, is already smaller
than the current plan. The union of all four professors' constraints
is therefore:

**Minimum architecture (the union spec).**

| Component | Status | Provenance |
|---|---|---|
| FiLMNoNorm at Injection A (no LayerNorm in modulated branch) | Required | RL-BDL §1.3 |
| Decoupled per-modality precision head, Kendall–Gal NLL on obs reconstruction | Required | RL-BDL §1.5 |
| **No multiplicative gate of $\hat{\pi}$ on $\gamma$** in the first Phase-3 run | Required (C1 dissociation depends on it) | RL-BDL §1.4; §2.1 above |
| Existing single-GRU $h_{\text{mod}}$ at hidden size 16 | Required | RL-BDL §1.5 |
| **Per-injection-site low-pass filter on $h_{\text{mod}}$ with learnable $\tau$** | Required (unlocks timescale-ordering pre-registration; biologically defensible) | Neuromodulation §4.2 option 1; §1.6 above |
| C1 architecture: scalar EMA on $o_{\text{noc}}$, broadcast through per-injection learned $w \cdot m_t + b$ | Required (the construct-spine control) | RL-BDL §4 |
| Frozen-head reconstruction logger on LayerNorm baseline (G1' diagnostic) | Required | RL-BDL §5.2 |

This minimum is *less* than the current plan in one place (no blend) and
*more* than the current plan in two places (per-site $\tau$ filters; C1
control). Net engineering cost vs. current plan: roughly equal (the
filters are tiny; C1 is one new modulation type; the blend deletion is
a removal). Net run-budget cost vs. current plan: **the C0/C1/C2
controls plus the rung ladder add ~108 runs of new headline experiment
budget** (RL-BDL §2.3) compared to the current plan's looser "Phase 4
analyses on whichever runs we have."

**Verdict.** The tension here is more apparent than real. Both the
"smaller than current Phase 3" and the "cheap upgrade" recommendations
fit cleanly into the union spec above. The real cost is in run budget
(the controls + ladder), not architecture. **Recommendation: lock the
union spec as Option A's architecture.** Option B (if H5-up is taken)
adds the T/P split and opioid head on top.

---

## 3. Updated claim/gap matrix

The six minimum claims of the original framing memo §2, mapped to (a) which
professor memos certify or contest each, (b) what is now required to defend
each, and (c) where the run budget lives.

| Claim | Certifications / contests | What defends it now | Run-budget cell | Status |
|---|---|---|---|---|
| **C1. Nociception ≠ Pain** | Pain-modeling §1, §3 (defines the trio); Bayesian-brain §3.2(c) (theoretical spine via C1); RL-BDL §1.6, §4 (architecture for the dissociation) | C0 vs. modulated on survival + four-property fingerprint; **C1 dissociation on cross-channel rate-of-rise difference** (§1.2 metric); C2 dissociation on cross-domain $r$ contrast | Headline 8 seeds + C0 5 seeds + C1 8 seeds + C2 5 seeds = 26 runs (canonical rung) | **Funded** by RL-BDL §2.3 budget |
| **C2. Pain computation buys something** (a survival, b clinical signature) | Pain-modeling §1, §2.2 (the four-property fingerprint); Bayesian-brain §2.3 (regional-contrast experiment for AI vs. heteroscedastic-BDL); RL-BDL §2.5 | (a) ΔSurvival modulated vs. C0 on rungs 1, 3, 5; (b) the four-property fingerprint as G2' (§1.4); the C1 dissociation is the *theoretical spine* | Same factorial (no new cell); regional contrast is a *post-hoc analysis* of canonical-rung runs, not new runs | **Funded for (a) and (b);** the regional contrast is *unfunded* — it requires regions of the grid with stereotyped-vs-novel predator threat, which is environment design (deferred to experiment-designer) |
| **C3. Whole-network modulation, not single site** | Pain-modeling (does not directly address); Bayesian-brain §1.4 (the A/B/C site mapping); RL-BDL §2.1 (specifies the 8-cell factorial + shared-core lesion); neuromodulation §2.1, §2.3 (H4 weak form — defensible; strong form — overclaimed) | 8-cell factorial + shared-core lesion on canonical rung 3, with the **weak-form H4 reading** ("co-vary because they share a slow latent driven by interoception", neuromodulation §2.1) | 8 lesion cells × 5 seeds + shared-core × 5 seeds = 45 runs at canonical rung | **Funded** by RL-BDL §2.3 budget |
| **C4. Time-locked, persistent, replicates on richer rung** | Pain-modeling §2.2(d) (policy-shift duration $\geq 3 \times$ recovery half-life); Bayesian-brain §4.2 (timescale ordering as the H5 theoretically-meaningful version); neuromodulation §3 (kills the strong H5 / certifies the downgrade) | Time-locked sign + lag from G2' (§1.4 items 1–2); **persistence via the timescale ordering**, requires the per-site $\tau$-filter upgrade (§1.6); replication via cross-rung transfer to rungs 1 and 5 | Headline + C0 + C1 on rungs 1 and 5 = 30 runs | **Funded** for the H5-down version. **Unfunded** for H5-up (Option B): would add ~30–60 runs for the T/P-split sub-factorial on rung 5 |
| **C5. Methodology contribution (the ladder)** | Pain-modeling §5.1 (rungs 1, 3, 5 are construct-mandatory); RL-BDL §3.2 (concurs; same architecture across rungs feasible) | Rungs 1, 3, 5 with the same architecture; the necessity/sufficiency matrix figure is the rung × signature matrix | Folded into C4's 30 cross-rung runs, plus the canonical-rung factorial | **Funded** in the {1, 3, 5} skeleton form. Rungs 2 and 4 remain unfunded — listed as planned extensions |
| **C6. Platform release** | Not directly addressed by any memo (this is documentation/reproducibility, not science) | Code release, single-GPU baseline, documented extension path; the §1.5 union architecture *is* rung-stable per RL-BDL §3.3, so the platform claim has architectural support | No dedicated runs; the headline runs themselves are the released baselines | **Implicit;** requires `senior-developer` to produce a release plan (§5 below) |

**Total funded headline budget:** ~108 runs (RL-BDL §2.3), covering Claims
1, 2(a), 2(b), 3, 4 (down-version), and 5. **Unfunded cells:** the AI-vs-
heteroscedastic-BDL regional-contrast experiment (Bayesian-brain §2.3) and
any Option B upgrade. The user's authorization should be read as approving
the funded set; unfunded cells become Phase-5 / post-submission work.

---

## 4. Recommended plan-rewrite scope — three options

Each option specifies (a) what changes in `project_plan.md` at the section
level, (b) total run budget, (c) what the paper can claim under the option,
and (d) what the paper cannot claim.

### Option A — Sober / minimum (recommended)

> "Ship a defensible paper at the *substrate-for-pain-like-behaviour*
> register, with the §1.5-union architecture, the {1, 3, 5} ladder, and
> H5 downgraded. This is the smallest set of changes that makes every
> professor's load-bearing recommendation land."

**Plan changes (project_plan.md):**

- §1 — rewrite robotics motivation in Register 2 (sober). Add a §1.1
  "Nociception vs. pain — the operational distinction" using the
  pain-modeling §1 form.
- §2 — rewrite "hypervigilance" from "agent becomes more cautious after
  injury" to the four-property joint fingerprint of §1.4 above.
  Replace §3's "full pain syndrome" phrasing per pain-modeling §4.3 and
  neuromodulation §7.1. Use the neuromodulation R-bio register
  ("ascending modulatory analog") in mechanism descriptions.
- §3 — architectural section: lock the §1.5 union spec. Drop the
  multiplicative blend; add per-injection-site $\tau$ filters; add C1
  modulation type spec.
- New §2bis — **Construct-validity controls (C0/C1/C2).** Pain-modeling
  §3 verbatim with RL-BDL §4 architectural specifications.
- New §4bis — **Environment ladder (rungs 1, 3, 5).** Define each rung
  with its modality structure and obs vector.
- §4 — rewrite G1 → G1' (channel-rank non-degeneracy diagnostic, RL-BDL
  §5.2). Rewrite G2 → G2' (the five pre-registrable quantities of
  §1.4).
- New §6 — **Robotics motivation.** Closes G-D. One page.
- New §7 — **Platform-release plan.** Closes G-F.
- §3 H5 — rewrite to the down-version language: "modulator-state
  inertia analog of post-injury caution persistence; we do not claim
  this models the chronic-pain transition." Per pain-modeling §5.2 and
  neuromodulation §3.3.

**Run budget.** ~108 runs (RL-BDL §2.3). Wallclock estimate:
depends on per-run cost; on the lab's standard node, ~one to two
weeks for the canonical-rung factorial + λ-sweep, plus another week
for the cross-rung headlines. Seeds: 5 per cell, 8 for the headline
cell.

**The paper can claim:**
- Claim 1 (nociception ≠ pain) on cross-channel rate-of-rise
  dissociation between modulated and C1.
- Claim 2(a) (survival benefit) modulated vs. C0 on rungs 1 and 3.
- Claim 2(b) (clinical signature) on the four-property fingerprint.
- Claim 3 (whole-network) on the 8-cell factorial + shared-core lesion,
  in the **weak H4** reading (shared-latent coordination, not "one
  modulator").
- Claim 4 in the **down-version**: time-locked, persistent on the
  modulator-inertia timescale, replicates on rungs 1 and 5; *not*
  chronic-pain transition.
- Claim 5 (methodology) in the {1, 3, 5} skeleton.
- Claim 6 (platform) supported by the rung-stable architecture.

**The paper cannot claim:**
- "Pain computation" as the headline term (only as a discussion-section
  foreshadowing).
- A chronic-pain transition. Single-GRU cannot exhibit one.
- "One modulator implements all three sites" (strong H4). The
  architecture's three outputs span at least three Doya slots, which
  the biology partitions across opponent pairs.
- A placebo-analgesia analog (no opioid-descending-head architecture).

**Justification.** This is the *defensible* paper. It uses every load-
bearing convergence the four professors agree on (§1.1–1.6) and takes
no risk on the disagreements (§2.1–2.3) where the construct guardian
or biology guardian has flagged a problem. It is also the cheapest path
that all four professors will sign.

### Option B — Full upgrade (the strong term)

> "Ship the paper at the *pain computation* register, with the H5-up
> upgrade (T/P split + opioid descending head) earning the strong
> headline."

**Plan changes (project_plan.md):** All of Option A, *plus*:

- §3 — replace the single-GRU $h_{\text{mod}}$ with a two-modulator
  $(h_{\text{mod-T}}, h_{\text{mod-P}})$ split. Add the opioid-analog
  descending-modulation head (a learned multiplicative gate on the
  nociception channel, conditioned on the tonic modulator state,
  neuromodulation §3.2). Add asymmetric T/P coupling with hysteresis.
- §4 H5 — rewrite to the up-version: pre-registrable
  acute/recovery/chronic three-regime structure with
  $T \to$ absorbing-state criterion (neuromodulation §3.2).
- New §6 of the architecture section — **Per-modulator teaching
  signals.** Each output of each modulator has a documented gradient
  source (neuromodulation §6).
- §1, abstract, title — Register 3 ("pain computation") *if* C0/C1/C2
  pass + transfer + the chronic-regime test passes.

**Run budget.** ~108 runs (Option A baseline) **+ ~30–60 runs** for
the T/P-split sub-factorial on rung 5 + the opioid-head ablation cells
+ additional seeds for the chronic-regime detection. Total: **~140–170
runs**, plus ~2–3 weeks of additional engineering time for the new
modulators and head.

**The paper can claim:** All of Option A *if* the upgraded controls
land. *Plus*:
- "Pain computation" headline (Register 3).
- Strong H4 — the *learned coordination* between T and P is the
  finding (neuromodulation §2.3).
- A defensible H5 chronic-pain claim with the three-regime
  pre-registration.
- A sotto-voce placebo-analgesia analog discussion (the opioid head is
  the architectural pre-condition; pain-modeling §1 property 3).

**The paper cannot claim:** Affective vs. sensory dimension of pain
(this is out of scope under any architecture in this project; declared
limitation per pain-modeling §5.3). A multi-modulator full Doya analog
(only T and P, not the four-way decomposition).

**Justification.** This is the *strong* paper. It earns the headline
term and the chronic-pain claim. It is also significantly more
expensive in both runs and engineering, and adds the risk of a deeper
architecture not landing in the time budget. Neuromodulation §7.2
explicitly recommends *not* taking this option if the upgrade is in
doubt: "more defensible to ship the smaller claim than to ship the
bigger architecture incomplete."

### Option C — No rewrite (status quo)

> "Stick with the current plan and Phase-3 architecture (with the
> multiplicative blend). Run the current factorial without the C0/C1/C2
> controls."

**Plan changes:** None.

**Run budget:** Whatever the current plan implies. Lower than Option A,
but the runs that *do* happen are not legible to the construct claim.

**The paper can claim:** "The neuromodulator architecture, when paired
with the heteroscedastic precision loss in a heterogeneous-noise
environment, escapes the v8 null and produces post-injury joint
modulation across perception, memory, and policy." This is publishable
as a workshop or domain-conference paper. **It is not a Nature MI
paper**, because:

**The paper cannot claim:** Any of Claims 1, 2(b), 3 (in the form a
reviewer expects), 4, 5, or 6 in defensible form. Specifically:

- **Claim 1 fails** because no nociception-only control (C0) is run, and
  the multiplicative blend destroys the C1 dissociation argument
  (RL-BDL §1.4). A reviewer will require these controls.
- **Claim 2(b) fails** as a construct claim: G2 is too loose and the
  channel-selectivity metric is non-identifiable under the blend.
- **Claim 3 fails** in the H4-weak form too, because the $r \geq 0.3$
  threshold inflates with multiple-comparisons (postdoc framing memo
  §5.3) and there is no C2 contrast.
- **Claim 5 fails** because no rung ladder.
- **The H4 strong form**, **the H5 chronic-pain claim**, and the
  headline term "pain computation" all fail.

**What survives review:** Option C will not survive a competitive
Nature MI submission. It survives a domain-conference (e.g., NeurIPS
RL workshop, COSYNE) submission as a proof-of-concept.

---

## 5. Hand-offs (named)

The following downstream agents pick up specific pieces of the rewrite,
each tagged with the gap(s) it closes and the professor memo(s) it
derives from. These are recommendations to the user, not invocations.

### 5.1 `senior-developer` — author the rewritten `project_plan.md`

**Scope.** Author the project_plan.md rewrite per the chosen Option (A or
B). Authorize and own the doc-edit plan.

**Specific deliverables:**

- §1 robotics motivation (Register 2). [G-D]
- §1.1 nociception vs. pain operational distinction. [G-A, G-E; pain-modeling §1]
- §2 four-property fingerprint replacing "hypervigilance" loose form.
  [G-A; pain-modeling §2.2; Bayesian-brain §4.1]
- §2bis C0/C1/C2 controls. [G-A; pain-modeling §3; RL-BDL §4]
- §3 architecture: §1.5 union spec; drop blend; add per-site $\tau$
  filters; spec C1 modulation type. [RL-BDL §1.5, §1.4; neuromodulation
  §4.2]
- §4 G1' (channel-rank non-degeneracy) and G2' (five pre-registrable
  quantities). [Bayesian-brain §3.2(a), §4; RL-BDL §5.2]
- §4bis environment ladder rungs 1, 3, 5. [G-B; pain-modeling §5.1;
  RL-BDL §3.2]
- §6 robotics-motivation page. [G-D]
- §7 platform-release plan. [G-F; C6]
- H5 language change to down-version (Option A) or up-version (Option B).
  [pain-modeling §5.2; neuromodulation §3]
- Drop "full pain syndrome." [pain-modeling §4.3; neuromodulation §7.1]

If Option B is taken, also: T/P-split architecture spec, opioid-head
spec, per-modulator teaching-signal documentation. [neuromodulation §3.2,
§6]

### 5.2 `experiment-designer` — controls' configs, factorial
pre-registration, fingerprint metrics

**Scope.** Author the configs, the factorial pre-registration (RL-BDL
§2.1–2.3), and the tightened G1'/G2' metrics (RL-BDL §5.2,
Bayesian-brain §4.3). Also pre-registers the cross-correlation lag and
null distribution per Bayesian-brain §4.

**Specific deliverables:**

- C0, C1, C2 configs. [pain-modeling §3; RL-BDL §4]
- The 12-cell factorial config family (8 lesion subsets + shared-core +
  C0 + C1 + C2) on canonical rung 3. [RL-BDL §2.1]
- The $\lambda_{\text{prec}}$ sweep config (3 cells × 5 seeds on full
  modulator). [RL-BDL §2.2]
- Cross-rung transfer config (headline + C0 + C1 on rungs 1 and 5).
  [RL-BDL §2.3]
- Pre-registration document for G2' five quantities, with theory-derived
  $\Delta\gamma$ from the canonical preset's $\Sigma_i(s_t)$ (replacing
  the placeholder 0.2). [Bayesian-brain §4.1, §4.3]
- The per-injection-site rate-of-rise dissociation metric
  $\Delta_t \gamma_{\mathcal{T}}^{(i)} - \Delta_t
  \gamma_{\mathcal{T}}^{(j)}$ for two threat channels with different
  injury-scaled noise ($\alpha = 2$ vs. $\alpha = 3$). [RL-BDL §4.4]
- The G1' frozen-head reconstruction logger, run on each candidate
  Phase-1' noise preset. [RL-BDL §5.2]
- The Bayesian-brain regional-contrast experiment design (deferred /
  unfunded; flag for Phase 5). [Bayesian-brain §2.3]

If Option B is taken, also: T/P-split factorial cells on rung 5;
opioid-head ablation cells; chronic-regime detection metric per the
three-regime acute/recovery/chronic structure. [neuromodulation §3.2]

### 5.3 `developer` — implementation

**Scope.** Implement the architectural changes derived from RL-BDL §1.5
and neuromodulation §4.2.

**Specific deliverables:**

- C1 EMA modulator: new `modulation.type = "EMAControl"` per RL-BDL §4.
  Trainable scalar $\alpha$ on $o_{\text{noc}}$, broadcast through
  per-injection learned linear read-outs. [pain-modeling §3, RL-BDL §4]
- Per-injection-site low-pass filter on $h_{\text{mod}}$ with learnable
  time constant per site. [neuromodulation §4.2 option 1]
- Remove the multiplicative blend $\pi_{\text{gate}} \cdot x_{\text{mod}}
  + (1 - \pi_{\text{gate}}) \cdot x_{\text{bypass}}$ from the Phase-3
  architecture. [RL-BDL §1.4]
- Frozen-head reconstruction logger on the LayerNorm baseline (G1'
  measurement). [RL-BDL §5.2]
- Per-modality MSE logger conditioned on `injury > 0.5` vs.
  `injury < 0.1` for the G1' check. [RL-BDL §5.2]

If Option B is taken, also: T/P-split modulator (two GRU cores with
different timescales, learned phasic→tonic coupling), opioid-analog
descending-modulation head (multiplicative gate on the nociception
channel itself, conditioned on tonic modulator state).
[neuromodulation §3.2]

### 5.4 `agent-manager` — orchestrate the rewrite

**Scope.** Coordinate the multi-agent rewrite flow.

**Recommended sequence:**

1. `senior-developer` produces the project_plan.md rewrite plan. (Plan
   only, not the rewrite itself.) Audit by the user.
2. *In parallel:* `experiment-designer` drafts configs and the
   pre-registration; `developer` implements C1, the per-site $\tau$
   filters, the blend deletion, and the G1' logger.
3. After (1) is approved, `senior-developer` writes the rewrite.
4. `env-config-auditor` audits the new config family.
5. `code-reviewer` reviews developer's implementation.
6. `agent-manager` launches the canonical-rung factorial via
   `training-runner`.
7. `experiment-analyzer` post-hoc analysis on the factorial results,
   with the G2' metrics applied.

This is a standard `agent-manager` flow; the value is mainly in
parallelising (1) and (2), and in catching the audit hand-offs.

---

## 6. Open questions and recommended user decisions

The synthesis surfaces three decisions the user has to make to authorize a
rewrite, plus a small number of subordinate decisions.

### 6.1 The three load-bearing decisions

1. **Headline register.** Register 2 (sober — "interoceptive neuromodulation
   as a substrate for pain-like behaviour") or Register 3 ("pain
   computation")? **Three of four professors recommend Register 2 unless
   Option B is taken and all controls dissociate + transfer + chronic
   regime.** I concur.

2. **H5 chronic-pain claim.** Upgrade (Option B, +30–60 runs and the T/P
   split + opioid head) or downgrade (Option A, "modulator-state inertia
   analog of post-injury caution persistence")? **Two professors
   recommend the downgrade if the upgrade is in doubt.** I recommend the
   downgrade, on the principle "ship the smaller claim defensibly."

3. **Multiplicative blend in Phase 3.** Drop now (RL-BDL recommendation)
   or keep and run a "with-blend" cell in parallel (5 extra runs,
   testable)? **Drop** — the blend destroys the C1 dissociation argument
   and is theoretically dispensable; running the with-blend cell as
   diagnostic is fine but should not be the default.

### 6.2 Subordinate decisions (lower stakes, mentioned for completeness)

4. **8 vs. 5 seeds on the headline cell.** RL-BDL §2.3 recommends 8 for
   the headline cell (clean bootstrap CIs on $r$) and 5 elsewhere. If the
   user is budget-constrained, drop to 5 across the board (saves ~3 runs;
   widens CIs on the cross-domain $r$ contrast). Recommend 8.

5. **Rungs 2 and 4.** Construct-mandatory only at {1, 3, 5} (pain-modeling
   §5.1; RL-BDL §3.2). Rungs 2 and 4 enrich the methodology figure but
   are not load-bearing. Defer to post-submission.

6. **The Bayesian-brain regional-contrast experiment** (AI vs.
   heteroscedastic-BDL, Bayesian-brain §2.3). Unfunded under Option A.
   The paper can commit to AI as the framing without running this
   experiment, declaring the choice as theory-led; the regional contrast
   is the *resolution* of the framing-vs-implementation question and is
   appropriate for a follow-up. Recommend deferral.

### 6.3 What still needs investigation before authorising a rewrite

- The G1' diagnostic should be run on the *current* canonical noise
  preset before authorising Phase 2/3, as a go/no-go on the noise
  landscape. RL-BDL §5.2 specifies this as ~1 hour wallclock per preset.
  This is the single highest-priority check; if no preset clears the
  threshold, the entire program is gated on first retuning the noise.

---

## 7. Recommended next step for the user

1. **Read this memo.** Make decisions §6.1.1, §6.1.2, §6.1.3.
2. **Authorize Option A** (recommended) or Option B. If Option C,
   acknowledge that this is not a Nature MI submission and revise scope
   accordingly.
3. **Spawn `agent-manager`** with the authorised Option as the brief.
   The manager orchestrates the rewrite per §5 above.
4. **In parallel with the rewrite, run the G1' diagnostic** (§6.3) so
   that any noise-preset retuning happens before Phase 2 launches.

The full chain — synthesis → user decision → manager flow → rewrite +
implementation → factorial launch — fits in roughly two weeks of
calendar time before the Phase-3 factorial begins, if all four agent
families work in parallel.

---

## 8. Cross-references

- [nature_mi_paper_framing.md](nature_mi_paper_framing.md) — the original framing scaffold.
- [pain_vs_nociception_construct.md](../concepts/pain_vs_nociception_construct.md) — pain-modeling memo.
- [active_inference_hypervigilance.md](../concepts/active_inference_hypervigilance.md) — Bayesian-brain memo.
- [architecture_for_pain_computation.md](../directions/architecture_for_pain_computation.md) — RL-BDL memo.
- [biological_plausibility_of_three_site_modulation.md](../critiques/biological_plausibility_of_three_site_modulation.md) — neuromodulation memo.
- [project_plan.md](../project_plan.md) — the document to be rewritten under Option A or B.
- [NEUROMODULATION_ALGORITHM.md §1.4 H1–H5](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md) — hypotheses certified/contested by the synthesis.
- [NMN_PERFORMANCE_DIAGNOSIS_v8.md](../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md) — the v8 null result, now read as the correct AI fixed point in homogeneous noise (Bayesian-brain §3.1).
- [PRECISION_MODULATION_ARCHITECTURE.md](../../develop/active/precision/PRECISION_MODULATION_ARCHITECTURE.md), [FILM_MODULATION_PLAN.md](../../develop/active/filim/FILM_MODULATION_PLAN.md), [FiLM_ENSEMBLE_SENSORY_PRECISION.md](../../develop/active/filim/FiLM_ENSEMBLE_SENSORY_PRECISION.md) — Phase 3 architectural anchors.
