---
title: Pain vs. Nociception — Construct-Validity Concept Memo
topic: construct-validity / pain-modeling
status: draft
author: professor-pain-modeling
created: 2026-05-07
last_updated: 2026-05-07
closes_gaps: G-A, G-E, G-B (construct side)
---

# Pain vs. Nociception — Construct-Validity Concept Memo

> One-line summary: **Nociception is a thresholded peripheral damage signal
> that drives reflexive avoidance; pain is a posterior over bodily threat
> inferred from nociception together with priors, context, and competing
> drives, computed on a slower timescale and *jointly* reweighting perception,
> memory, and policy.** The project's nociception-only control is the *right
> direction* but, as currently sketched, is not yet a sufficient control —
> two stronger comparison conditions are required for the headline claim. The
> phrase "pain computation" is defensible if and only if the modulator is
> shown to be selective, transferable, and dissociable from a richer-but-still-
> peripheral nociception baseline; otherwise the sober alternative
> "**interoceptive neuromodulation as a substrate for pain-like behaviour**"
> should be preferred.

> Construct-validity verdict: **partial**. The architectural intent is
> aligned with the pain-computation literature; the empirical operationali-
> sation in `project_plan.md` is missing two of the three controls a Nature MI
> pain-modelling reviewer will demand, and the chronic-pain claim (H5) is at
> risk of being a story about modulator inertia rather than about chronicity.

> Anchors: [project_plan.md §§1–2, §4 G2](../project_plan.md);
> [NEUROMODULATION_ALGORITHM.md §1.4 H1, H2, H4, H5](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md);
> [nature_mi_paper_framing.md §§1–3, §5.1](../ideas/nature_mi_paper_framing.md);
> [computational_pain_hypervigilance_tutorial.md](../tutorials/computational_pain_hypervigilance_tutorial.md).

---

## 1. The Operational Distinction (Reviewer-Acceptable Form)

A computational-pain reviewer at Nature MI will not accept "pain ≠ nociception"
as a slogan. They will want it stated as a *computational* claim about *what is
being inferred from what, on which timescale, with which downstream effects*.
The claim the paper has to make is the following.

**Nociception** is a likelihood channel: a transduction $o_{\text{noc}, t}$
of tissue-damaging stimuli into a noisy signal that the agent receives as
input. It is *peripheral* (in the agent's input layer), *fast* (per-step),
and *driver-of-reflex* (it can be wired directly into the policy via reward
shaping or a withdrawal heuristic without any hidden state). Loss of
nociception (CIP patients; the project's `noc=0` ablation) abolishes the
reflex but does not in itself touch perceptual or mnemonic processing of
threat (Brand & Yancey 1997; Cox et al. 2006). This is the textbook
"alarm bell".

**Pain** is the posterior over a hidden bodily-threat variable $s_t$
(injury state, recovery progress, danger of the current context) given the
full sensory and interoceptive history:

$$
p(s_t \mid o_{1:t}, \text{prior}_t)
\propto
\underbrace{p(o_t \mid s_t)}_{\text{likelihood, incl.\ nociception}}
\,
\underbrace{p(s_t \mid s_{t-1})}_{\text{slow dynamics}}
\,
\underbrace{p(\text{prior}_t)}_{\text{expectation, context}} .
$$

This is the predictive-coding view of pain (Büchel et al. 2014;
Wiech 2016; Tabor & Burr 2019; Geuter et al. 2017 for the Bayesian
placebo decomposition; Seth & Friston 2016 and Barrett & Simmons 2015 for
the interoceptive-inference frame). Three properties of $p(s_t \mid \cdot)$
are decisive for the construct, and all three appear *together*:

1. **Slow dynamics.** $s_t$ evolves on the timescale of injury and
   recovery (seconds–minutes in this grid-world; hours–weeks
   clinically), not the timescale of nociceptive onset. This is what
   Seymour, Crook & Chen (2023) call *post-injury pain* — pain as a
   feedback controller for recuperation, distinct from acute
   nociception. Chronic pain is the failure mode of this controller
   (Vlaeyen & Linton 2000 fear-avoidance; Apkarian et al. 2009).

2. **Joint downstream effect.** The posterior reweights *perception*
   (precision-gain on threat-relevant channels — Eccleston & Crombez
   1999 attentional bias; Van Damme et al. 2010), *memory* (retention
   of threat-context associations — Crombez et al. 2005), and *policy*
   (avoidance, withdrawal, freezing — Vlaeyen & Linton 2000) through a
   *common* modulatory pathway. This is the empirical core of the
   Vlaeyen fear-avoidance model and the predictive-coding account: the
   three are not independently regulated.

3. **Dissociability from the input.** The posterior can stay elevated
   when nociception falls (chronic pain without ongoing tissue damage —
   Apkarian; placebo analgesia in which $o_{\text{noc}}$ is unchanged
   but $p(s_t)$ moves — Wiech 2016, Büchel et al. 2014). This
   dissociation is the strongest construct-validity evidence that pain
   is a *computation*, not a relabelling of the input.

The project's modulator architecture (shared GRU driving Injections A/B/C
on perception, memory, policy) is a defensible operationalisation of
properties (1) and (2). Property (3) — dissociability from $o_{\text{noc}}$
— is the one the empirical design has to *demonstrate*. As written,
`project_plan.md` does not.

---

## 2. The Hypervigilance Fingerprint — Quantitative Definition

The clinical signature the paper would be claiming the modulated agent
reproduces is a tightly-defined pattern. The paper has to commit to it in
advance.

### 2.1 Empirical signature in animals/humans

Crombez, Van Damme & Eccleston (2005) define hypervigilance as
"an unintentional and efficient process that emerges when the threat
value of pain is high … and the individual's current concern is to escape
and avoid pain." Operationally, in human and animal pain studies, four
properties co-occur after a noxious event:

- **(a) Attentional bias toward threat cues.** Reaction-time advantages,
  dot-probe / Stroop-analog effects on threat-relevant stimuli (Eccleston
  & Crombez 1999; Asmundson et al. 1997).
- **(b) Difficulty disengaging.** Once attention is captured, redirection
  is slow and effortful (Van Damme et al. 2004).
- **(c) Memory bias.** Threat-paired contexts are recalled and avoided
  more strongly than neutral ones, even after the noxious stimulus is
  removed (Vlaeyen & Linton 2000).
- **(d) Behavioural avoidance / freezing.** Risk-aversive policy shift,
  disproportionate to the current nociceptive input (Vlaeyen, Crombez &
  Linton 2016).

All four must be *time-locked to the noxious event*, *correlated across
domains within an individual* (Crombez et al. 2005, "concern-dependent"
modulation), and *outlast the input itself* (the chronic-pain transition;
Apkarian et al. 2009).

### 2.2 Simulation analog (what would count, in this project)

I propose the paper commit to the following four observables, each with a
quantitative threshold the modulated agent must clear and the
nociception-only control must not:

| Clinical property | Project observable | Operational threshold |
|---|---|---|
| (a) Attentional bias | Increase in encoder gain $\gamma$ on threat channels (olfaction, visual when those carry threat info) within $\Delta t_{\text{lock}}$ of injury, relative to the pre-injury 50-step window | Effect-size $d \ge 0.5$ across $\ge 5$ seeds, time-locked to event onset within $\pm 5$ steps, *channel-selective* (gain on non-threat channels does not move by more than $0.2 d$) |
| (b) Disengagement difficulty | Auto-correlation length of $\gamma_{\text{threat}}$ post-injury vs. pre-injury baseline | $\tau_{\text{post}} \ge 2 \tau_{\text{pre}}$ on the same agent, same seed |
| (c) Memory bias | Post-injury negative shift of $z_{\text{memory}}$ (retention bias on the GRU update gate) | $z_{\text{memory}}$ shift effect-size $d \ge 0.5$, time-locked, *correlated within-trial* with the gain shift in (a) at lag $0$–$5$ steps, $r \ge 0.3$ |
| (d) Avoidance / freezing | PPO temperature drop or DreamerV3 reward-scale drop in dangerous regions, *conditional on having returned to baseline injury level*, i.e. the policy shift outlasts the input | Policy-shift duration $\ge$ 3× injury-recovery half-life; $d \ge 0.5$ |

The **joint** signature — all four observables co-modulated, time-locked,
within-agent, channel-selective — is the hypervigilance fingerprint. Any
one of them alone is consistent with much weaker constructs (a reflex,
a slow input, a stuck recurrent state). The paper's central empirical
claim should be the joint signature, not the marginals.

This sharpens G2 in the project plan. Currently G2 says "time-locked,
cross-domain". The construct-relevant version says "time-locked, within
$\pm 5$ steps; cross-domain with within-trial correlation $r \ge 0.3$ at
lag $\le 5$; channel-selective; outlasting the input". I recommend
the user adopt this in any plan rewrite, because the looser version is
vulnerable to the multiple-comparisons / post-hoc-fitting pushback raised
in the postdoc memo §5.3.

### 2.3 False-positive hypervigilance — what to rule out

Three failure modes look like H1+H2+H5 from the marginals but are not
the construct. The paper has to rule each out before publishing the
fingerprint.

**FP-1: Degenerate modulator collapse.** The modulator outputs a
near-constant amplification factor across all channels post-injury; on
any single channel this looks like H1, but it is just a scalar gain
applied to everything. *Rule-out:* channel-selectivity check — the
post-injury gain on threat-relevant channels must exceed the gain on
threat-irrelevant channels (e.g. proprioception) by a fixed margin
($\Delta\gamma \ge 0.2$, say). If not, this is volume-control, not
hypervigilance. This is the formal version of the v8 null result's
"flat $\gamma$" pathology dressed up as a positive finding.

**FP-2: Slow-input mimicry.** The agent's "modulator state" is just a
slow filter of $o_{\text{noc}}$ — i.e. it carries the same information
as a low-pass filter of nociception. There is no posterior computation,
just smoothing. *Rule-out:* a control that *replaces* the modulator with
an exponential-moving-average of $o_{\text{noc}}$ (a single scalar with
a trainable timescale), wired into the same three injection sites. If
this control reproduces the fingerprint, the paper cannot claim it is
showing pain *computation* — it is showing pain *smoothing*. This is a
strong test and the project plan does not include it.

**FP-3: Reward-shaping equivalent.** The avoidance signature (d) can be
reproduced by a non-modulated agent whose reward function is
$r' = r - \lambda \cdot o_{\text{noc}}$ with appropriate $\lambda$, on a
sufficiently long episode. *Rule-out:* the avoidance must be
*context-modulated* — drop only in dangerous regions, not uniformly —
and must be tied to memory-trace activation, not to instantaneous
nociception. The project plan's "conflicting-needs" framing already
gestures at this; the paper needs the explicit comparison.

If FP-1, FP-2, and FP-3 cannot be ruled out, the headline claim
"pain computation, not nociception" is not yet defensible.

---

## 3. The Nociception-Only Control — Adjudication

The postdoc memo (G-A) proposes a nociception-only control: damage signal
preserved in observation, modulator architecture present but
`modulation.type = null`, optionally with interoceptive channels masked
in the modulator input.

This is a step forward but is **not sufficient**. Construct validity for
the pain-vs-nociception claim requires a *ladder of three controls*, not
one:

### Control C0 — Pure nociception (reflex baseline)
- Damage signal $o_{\text{noc}}$ in observation.
- No modulator architecture present.
- No shared interoceptive state.
- Policy is a standard recurrent PPO / DreamerV3 with the full obs
  vector.
- **What it tests:** can the task be solved by reflexive avoidance + standard
  RL? If yes, it sets the survival-benefit floor. If the modulated agent
  does not beat this, Claim 2(a) fails and the paper has to fall back on
  Claim 2(b) (the clinical-signature match).

### Control C1 — Slow-input mimic (rules out FP-2)
- Modulator architecture replaced with a 1-D exponential-moving-average
  of $o_{\text{noc}}$, trainable timescale, injected at A/B/C in the
  same way the modulator's GRU output would have been.
- **What it tests:** is the GRU's posterior computation doing anything
  beyond temporal smoothing? This is the cleanest test that pain is a
  *computation* over interoceptive evidence, not a relabelling of
  nociception's slow envelope. **Without this control, the construct
  claim is rhetorical.**

### Control C2 — Modulator with masked interoception (the postdoc's variant)
- Full modulator architecture present, but the modulator's *input* has
  the interoceptive channels (injury, satiation, nutrition) masked or
  zeroed.
- **What it tests:** is the modulator using the interoceptive read, or
  is it functioning as a generic recurrent context module that would
  help any RL agent? This is the architecture-vs-information control.

The modulated agent must beat **all three**. C0 establishes the floor,
C1 rules out smoothing-as-pain, C2 rules out architecture-as-pain. Only
when the modulated agent dissociates from C1 and C2 — i.e. is doing
something with interoception that an EMA and a generic recurrent module
cannot — does the "pain computation" framing earn its keep.

The plan as written contemplates only an analog of C2 (and that only
optionally). I recommend the user route a recommendation to
`experiment-designer` to add C0 and C1 as standing controls in any
phase-2-or-later configuration.

---

## 4. The "Pain-Like Behaviour" Phrase — Where It Strains

The postdoc's pushback flag §5.1 is correct. There are three places the
phrase "pain-like behaviour" is doing rhetorical work that the construct
will not yet support.

**4.1 In §2 of `project_plan.md`** ("hypervigilance" and
"conflicting-needs modulation" framed as the success target). This is
fine *if* the §2.2 fingerprint above is adopted. As currently written,
"hypervigilance" in the plan is closer to "the agent becomes more
cautious after injury" — which is consistent with reward-shaping
(FP-3), with slow-input mimicry (FP-2), and with degenerate modulator
collapse (FP-1). The phrase will not bear the weight a Nature MI
reviewer puts on it. *Recommendation:* rewrite §2.1 of the plan to
adopt the four-property joint definition above.

**4.2 In Phase 4 ("hypervigilance as the primary scientific
readout").** This phrasing implies the analyses *are* the readout. They
are not — they are the readout *if* C0/C1/C2 are dissociated from the
modulated agent on the joint fingerprint. Without that dissociation,
Phase-4 traces are descriptive, not constitutive. *Recommendation:*
gate Phase 4 on C0/C1/C2 dissociation, not just on G1 + G2.

**4.3 In §3 ("Whole-Network Neuromodulation") describing the modulator
output as producing the "full pain syndrome".** This is the strongest
overstatement. The pain syndrome includes affective dimensions
(anterior-cingulate-mediated unpleasantness, distinct from
somatosensory intensity — Rainville et al. 1997; Price 2000) which the
project's single-modulator design is not equipped to dissociate. A
sober rewording is "the joint perceptual–mnemonic–policy reweighting
component of the post-injury behavioural syndrome" — long, but
defensible. *Recommendation:* drop "full pain syndrome"; the paper
should be explicit that the affective/sensory dissociation is *out of
scope* and cite that as a limitation rather than glide past it.

### 4.4 The headline term: "pain computation" or something more sober?

The postdoc raised this in §5.1. My recommendation, as construct-validity
guardian, is:

- **If** C0, C1, C2 are all dissociated from the modulated agent on the
  joint fingerprint, **and** the modulator-state representation transfers
  across environment rungs (gap G-G), **then** "pain computation" is
  defensible — the modulator is computing something on interoceptive
  evidence that yields a clinical-style signature that nociception alone,
  smoothed nociception, and an interoception-blind modulator do not. This
  is a real construct claim.

- **If** any one of C0/C1/C2 is *not* dissociated, or transfer fails,
  **then** the headline term "pain computation" is rhetorical and the
  paper should retreat to **"interoceptive neuromodulation as a substrate
  for pain-like behaviour"**. This phrasing is honest about what is being
  shown — that adding shared, interoceptive, slowly-evolving modulation
  produces behaviour that *resembles* pain phenotypes — without
  pre-committing to the stronger claim that the agent is *computing pain*.

The user does not have to choose now; the choice should follow the
empirics. But the paper should be drafted in two registers, and the
robotics-angle abstract should be in the sober register *until*
C0/C1/C2 results are in.

---

## 5. Implications for the Plan (Construct-Validity Side)

Three concrete implications for the construct-validity side of gap **G-B**
(the environment ladder), in addition to G-A and G-E which this memo
closes.

**5.1 Minimum rungs for the construct.** The pain-computation construct
*requires* (i) an injury event with non-trivial dynamics, (ii) at least
one threat channel separable from non-threat channels (for the
selectivity check, FP-1), (iii) at least one drive that competes with
avoidance (for the context-modulation check, FP-3), and (iv) a recovery
window long enough to test outlasting (for H5). This implies **rungs 1,
3, 5** of the postdoc's proposed ladder are construct-mandatory; rungs
2 and 4 are valuable for the methodology contribution (Claim 5) but are
not on the critical path for the pain-vs-nociception headline. If the
user is cost-constrained, the headline can survive with three rungs.

**5.2 The H5 chronic-pain claim is the highest-risk sub-claim.** The
project's current operationalisation of chronicity is "modulator-GRU
timescale sweep — does the cautious mode outlast the injury?" This is a
test of *recurrent inertia*, not of *chronic pain*. Clinical chronic
pain is the failure of a learned controller (Vlaeyen-Linton; Apkarian);
the analog in this project would require the modulator to acquire,
through training, a representation that *generalises* to novel
post-recovery contexts and continues to produce avoidance there — not
just to decay slowly. *Recommendation:* the chronic-pain claim should
be downgraded to "modulator inertia analog of post-injury caution
persistence" *unless* the transfer probe of gap G-G is added and
passes. This is the single most important construct-validity caveat in
the paper.

**5.3 The affective/sensory dissociation is out of scope and should be
declared so.** Single-modulator architectures are not equipped to
dissociate sensory-intensity from affective-unpleasantness components
of pain (Price 2000; Rainville 1997). The paper should declare this as
a scope limitation in §1 of the manuscript and not let reviewers
discover it.

---

## 6. Construct-Validity Verdict

**Partial.**

- *Validated:* the architectural intent (shared interoceptive
  modulator driving perception, memory, and policy via a common slow
  state) is a defensible operationalisation of the predictive-coding /
  interoceptive-inference account of post-injury pain-like behaviour.
  The three-injection-site design corresponds well to the
  precision-gain / prior-dominance / policy-shift triad in the
  literature.
- *Mis-aligned (currently):* the empirical design as written in
  `project_plan.md` does not include the two controls (C0, C1) that
  would dissociate "pain computation" from "richer nociception". G2 is
  too loose to detect the joint fingerprint with the false-positive
  rule-outs the construct demands. The chronic-pain claim (H5) is at
  risk of being a recurrent-inertia claim mis-labelled.
- *Blocking:* the rhetorical use of "full pain syndrome" in §3 of the
  plan, and the implicit equivalence of "more cautious after injury"
  with "hypervigilance" in §2.

The path to a fully validated construct is well-defined and within
reach: adopt the four-property joint fingerprint of §2.2, add controls
C0 and C1, gate Phase 4 on dissociation, add the cross-rung transfer
probe for H5, and declare the affective/sensory limitation up front. If
those changes are made, the paper has a defensible construct claim that
will survive a Nature MI pain-modelling reviewer.

---

## 7. Next Steps

1. **`professor-bayesian-brain`** (next in the postdoc's chain) — should
   formalise the precision-matrix story for the four-property joint
   fingerprint above, and identify the specific algorithmic
   architecture under which selective (vs. flat) precision elevation is
   the free-energy-minimising solution. Their FP-1 rule-out is the
   theoretical mirror of mine.
2. **`professor-rl-bayesian-dl`** — should adjudicate whether the
   factorial ablation can deliver the joint-fingerprint test within
   budget, and how many seeds the within-trial $r \ge 0.3$ threshold
   requires.
3. **`professor-neuromodulation`** — should rule on whether the H5
   chronic-pain claim survives if the modulator is restricted to a
   single GRU timescale (likely no, in light of the multi-timescale
   tonic/phasic literature; their memo will close that loop).
4. **`experiment-designer`** (downstream, after the four memos) —
   should add controls C0 and C1 to the canonical configuration set
   and pre-register the four-property fingerprint thresholds.
5. **`senior-developer`** (downstream) — minimal architecture work to
   support C1 (the EMA-of-nociception modulator) and the cross-rung
   modulator-state-freeze probe for the transfer test in §5.2.

The user should hold any plan rewrite (`project_plan.md` §§1–4) until
all four professor memos are in. The Bayesian-brain memo will tighten
§2.2's fingerprint definitions; the neuromodulation memo will resolve
the H5 chronicity claim; the RL/BDL memo will tell the user what the
ablation table costs.

---

## 8. References (papers the project should add to `docs/project/references/`)

Papers cited above that anchor the construct claims. Where they are not
already in the project's references directory, they should be added so
that the paper's pain-modelling lineage (gap G-E) is documented.

- Apkarian, A.V., Baliki, M.N., Geha, P.Y. (2009). "Towards a theory
  of chronic pain." *Progress in Neurobiology*. — chronic-pain-as-
  controller-failure framing.
- Asmundson, G.J., Kuperos, J.L., Norton, G.R. (1997). "Do patients
  with chronic pain selectively attend to pain-related information?"
  *Pain.* — attentional bias.
- Barrett, L.F., Simmons, W.K. (2015). "Interoceptive predictions in
  the brain." *Nature Reviews Neuroscience.* — interoceptive-inference
  framing.
- Brand, P., Yancey, P. (1997). *The Gift of Pain.* — CIP / pain-as-
  signal essential for life.
- Büchel, C., Geuter, S., Sprenger, C., Eippert, F. (2014). "Placebo
  analgesia: a predictive coding perspective." *Neuron.* — Bayesian
  placebo.
- Cox, J.J. et al. (2006). "An SCN9A channelopathy causes congenital
  inability to experience pain." *Nature.* — CIP molecular basis.
- Crombez, G., Van Damme, S., Eccleston, C. (2005). "Hypervigilance
  to pain: an experimental and clinical analysis." *Pain.* — the
  operational definition adopted in §2.1.
- Eccleston, C., Crombez, G. (1999). "Pain demands attention: a
  cognitive-affective model of the interruptive function of pain."
  *Psychological Bulletin.* — interruption hypothesis.
- Geuter, S., Koban, L., Wager, T.D. (2017). "The cognitive
  neuroscience of placebo effects." *Annual Review of Neuroscience.*
  — Bayesian decomposition of placebo.
- Price, D.D. (2000). "Psychological and neural mechanisms of the
  affective dimension of pain." *Science.* — sensory/affective
  dissociation.
- Rainville, P., Duncan, G.H., Price, D.D., Carrier, B., Bushnell,
  M.C. (1997). "Pain affect encoded in human anterior cingulate but
  not somatosensory cortex." *Science.* — same.
- Seth, A.K., Friston, K.J. (2016). "Active interoceptive inference
  and the emotional brain." *Phil. Trans. B.* — interoceptive active
  inference.
- Seymour, B., Crook, R.J., Chen, Z.S. (2023). "Post-injury pain and
  behaviour: a control theory perspective." *Nature Reviews
  Neuroscience.* — three-tier nociception/post-injury/chronic
  taxonomy adopted in §1.
- Tabor, A., Burr, C. (2019). "Bayesian learning models of pain: a
  call to action." *Current Opinion in Behavioral Sciences.*
- Van Damme, S., Crombez, G., Eccleston, C. (2004). "Disengagement
  from pain: the role of catastrophic thinking about pain." *Pain.*
- Vlaeyen, J.W.S., Linton, S.J. (2000). "Fear-avoidance and its
  consequences in chronic musculoskeletal pain: a state of the art."
  *Pain.* — fear-avoidance model.
- Vlaeyen, J.W.S., Crombez, G., Linton, S.J. (2016). "The fear-
  avoidance model of pain." *Pain.* — updated.
- Wiech, K. (2016). "Deconstructing the sensation of pain: the
  influence of cognitive processes on pain perception." *Science.* —
  pain as constructed perception, the headline review for the
  argument.
