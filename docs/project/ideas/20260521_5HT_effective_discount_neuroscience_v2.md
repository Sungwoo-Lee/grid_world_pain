---
title: "5-HT and the effective discount — affective + computational neuroscience perspective (v2 stage)"
status: draft
author: top-level-claude (synthesis of user-Claude discussion, 2026-05-21)
audience: user + pi + research-postdoc
date: 2026-05-21
v_stage: v2
companions:
  - "docs/project/ideas/20260519_doya_modulation_extended_creative_ideas_v2.md"
  - "docs/project/ideas/20260521_doya_modulation_summary_v2.md"
  - "docs/project/critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md"
integration_status: "Standalone perspective memo, NOT yet integrated into the v2 extended ideas memo or the audit. The user will integrate later."
one_line_summary: "Reframes 5-HT's role from 'modulating the discount parameter γ' to 'inducing an effective horizon as an emergent property of timescale-mixture gating' — better-grounded in affective and computational neuroscience and matching the FiLM-only architectural route."
---

## 1. What this doc is

A standalone perspective memo on the **5-HT → time-discount** knob of the project's neuromodulator-network framing. It is **not** an integration into the v2 extended ideas memo or the technical audit — both of those will be revised later. This doc captures a single reframing that came out of a 2026-05-21 user-Claude discussion: the *effective-discount* reading of 5-HT is more biologically realistic than the *implementation-discount* reading the project has been using, and the architectural consequences are favourable for the FiLM-only route.

A reader with basic RL only: when the textbook says $V^\pi(s) = \mathbb{E}\!\left[\sum_t \gamma^t r_t\right]$, the discount factor $\gamma$ is a *parameter the experimenter sets*. The Bellman recursion then "applies" $\gamma$ at each backward step. Doya (2002) mapped this $\gamma$ onto the serotonin (5-HT) system in the brain. **This memo argues that the mapping is closer to the truth if we drop the word "parameter" and replace it with "measured effective horizon."** The brain doesn't have a $\gamma$; it has multiple representations of the future at different timescales, and 5-HT gates which timescale dominates behaviour. The discount you can measure (by perturbing a reward at delay $k$ and reading the slope of $\log|\Delta V|$ vs. $k$) is the *consequence* of that gating, not its cause.

## 2. The shift in framing — implementation γ versus effective horizon

The project's v2 extended memo handles the 5-HT route via a **γ-conditional UVFA** — a separate output head (Head 2) emits a learned discount $\gamma_B(c)$ which is plugged into the TD target during training. This is algorithmically clean. It is also algorithmically *artificial* — there is no neural operation in the brain that multiplies the next-state value by a scalar before adding it to the current reward. The Bellman recursion is a useful abstraction; it is not a biological mechanism.

The alternative framing is to keep the training discount $\gamma_\text{train}$ **fixed** (its standard role in the implementation), and let the FiLM modulator change which value function the critic emits at each interoceptive state $c$. Different $c$ → different emitted value function → different *effective* horizon, measured ex post via reward-perturbation. The modulator never "sets" the discount; it *induces* an effective horizon as an emergent property of how it gates the critic's representations.

**Architecturally this is the FiLM-only route from the audit:** Head 1 (FiLM scales and shifts on critic activations) is sufficient on its own; no Head 2 emission of a discount-into-target is required for the discount-like behavioural signature to appear. **Biologically this is the right framing:** the brain implements effective horizons, not discount parameters.

The cost is that the project can *measure* the effective horizon (via the calibrated-effective-horizon test the audit specified for 5-HT) but cannot *directly control* it — there is no knob labelled "set $\gamma_\text{eff}$ to 0.85." The neuroscience perspective makes the cost into a feature: brains can't directly set their effective horizons either; they shift as a downstream consequence of affective and interoceptive modulation.

## 3. Affective neuroscience grounding

The strongest evidence that effective horizon is *gated by affective and interoceptive state* — not by a parameter — comes from four well-established lines of work in affective neuroscience.

**Delay-discounting tasks across clinical populations.** Depression, anxiety, addiction, chronic pain, ADHD, and acute stress all reliably shift effective discount toward myopia (Ainslie 1992; Mazur 1987; the post-2000 clinical literature is extensive). The shift is dose-dependent on affective intensity and reversible with treatment. Crucially, the *parameter* the brain would need to "change" if Doya 2002 were taken literally is not the kind of object that varies on the timescales these clinical effects show. What does vary on those timescales is the gain of affective and interoceptive systems on valuation — which then *induces* the observed discount shift.

**Damasio's somatic marker hypothesis.** Bodily and visceral feedback (interoceptive signals from the body to vmPFC) weight future scenarios during deliberation. A future outcome that elicits a strong visceral signal carries more *felt* weight in current decision-making. This is exactly the gating mechanism the effective-discount reading needs: the modulator (driven by interoceptive state) changes which futures are "felt" loudly, and the effective horizon shifts as a consequence. The somatic marker hypothesis is the canonical pre-existing instance of "interoceptive state modulates effective valuation across time."

**Craig's interoceptive theory and insula timescales.** The anterior insula integrates interoceptive afferents into a unified affective state and feeds vmPFC and ACC for valuation. Insular lesions and lesion-overlap studies in chronic pain show shortened effective horizons that correlate with insular dysfunction (not with any γ-like parameter). The project's interoceptive modulator c is structurally analogous to the insula's role here — a single integrated representation of bodily state that gates downstream valuation.

**Berridge & Robinson's wanting-vs-liking dissociation.** Dopamine modulates motivational salience independently of hedonic value, and this dissociation produces *another* route to effective-discount shifts — when wanting is amplified without liking, near-term outcomes dominate. This is important for the project because it shows the brain has *multiple parallel routes* to producing effective-discount shifts, not just a single γ-like channel. A neural-network modulator that produces effective-horizon shifts via any of these routes is biologically defensible; one that claims to modulate a literal γ is not.

**Summary.** Across these four lines of work, the consensus from affective neuroscience is that effective horizon is a *downstream emergent property* of how affective and interoceptive systems gate representation, attention, and policy. It is not an upstream parameter the brain reads off and applies. The project's effective-discount framing matches this consensus.

## 4. Computational neuroscience — three implementations the brain actually has

Three concrete mechanisms by which the brain produces effective-horizon shifts without ever performing a γ-multiplication step:

**(1) Cortical timescale hierarchy.** Hasson et al. (2008, 2015), Murray et al. (2014), Chaudhuri et al. (2015): different cortical regions have *intrinsic* time constants. Primary sensory cortex integrates over hundreds of milliseconds; lateral and posterior temporal regions over a few seconds; medial prefrontal and hippocampus over many seconds to minutes. The brain implements "operating at different horizons" by *which region's representation drives behaviour* at a given moment. Neuromodulators including 5-HT bias the gain across this hierarchy via region-specific receptor distributions (5-HT1A predominantly in cortex with one effect, 5-HT2A in another, 5-HT3 elsewhere). When 5-HT levels rise, the long-timescale (prefrontal, hippocampal) contributions to behaviour amplify; when they fall, short-timescale (limbic, striatal) contributions dominate.

This is *exactly* the FiLM-modulator-on-critic-activations story. The critic's intermediate representations span a range of intrinsic timescales (perhaps explicitly hierarchical, perhaps implicitly so via the network's depth). The FiLM modulator gates per-feature gain across these representations. The effective horizon of the emitted value function shifts as a consequence. There is no backward multiplication anywhere.

**(2) Successor-representation modulation.** Dayan (1993); Momennejad et al. (2017); Stachenfeld et al. (2017) on hippocampal place cells as successor features: the brain appears to represent expected future state visitations directly, and *different ψ representations correspond to different effective horizons*. A representation that emphasises near-term transitions implements an effective short horizon; one that emphasises longer-range transitions implements a longer effective horizon. Switching between (or interpolating across) ψ representations gives effective-discount shifts without any γ-multiplication.

For the project, this connects directly to the v2 extended memo's 5-HT Candidate 3 (γ-Models) and to the DA-route's successor-features candidate. The brain's actual implementation looks closer to "modulate which successor representation drives behaviour" than to "set γ."

**(3) Active inference / predictive coding.** Friston, Pezzulo: value-of-policy is computed as expected free energy over a generative model of the future. The effective horizon is set by *the precision of the prior over future states* — high precision on near-term futures makes the agent effectively myopic, regardless of how the formal sum is written. 5-HT in this framework modulates precision; precision modulation is a forward-pass operation; the effective horizon shifts as a downstream consequence. The project's interoceptive modulator naturally maps onto a precision-modulator in this reading.

These three are not exhaustive but they cover the dominant computational-neuroscience interpretations. Across all three, the discount factor is **never a parameter the brain manipulates directly**. It is always a property of the emitted value function, induced by some upstream modulation of representation or precision.

## 5. The empirical anchor — Tanaka et al. 2007

If the project wants a single bridging paper from affective + computational neuroscience to Doya 2002's mapping, the cleanest candidate is **Tanaka, Schweighofer, Asahi et al. (2007), *Serotonin Differentially Regulates Short- and Long-Term Prediction of Rewards*** (Nature Neuroscience; with the 2004 PNAS predecessor). Acute tryptophan depletion (lowering brain 5-HT) shifted fMRI activity from regions correlated with long-timescale reward predictions to regions correlated with short-timescale reward predictions. The interpretation the authors gave is precisely the cortical-timescale-hierarchy story above: **5-HT does not "change γ"; it shifts the brain's reliance from far-future-predicting regions to near-future-predicting regions, and the behavioural effective discount shifts as a consequence.**

This paper is the most direct empirical support for the effective-discount reading. It also explicitly inspired Doya 2002's mapping — Doya was working from the *measured* behavioural effective discount, not from any neural γ-multiplication operation. So the effective-discount reading is in some sense closer to what Doya was actually pointing at than the parameter-mapping reading the project has been using.

## 6. Implications for the project's framing

Three concrete implications of taking the effective-discount reading seriously, written here for the integration the user will do later.

**(a) FiLM-only suffices for 5-HT.** The v2 extended memo's 5-HT section recommends the γ-conditional UVFA (Head 2 emits $\gamma_B(c)$ into the TD target) as the cleanest implementation. Under the effective-discount reading, **Head 1 (FiLM on critic activations) is sufficient on its own** — the modulator changes the critic's representation, the emitted value function has a different effective horizon, and the calibrated-effective-horizon test can verify the shift ex post. This is architecturally simpler, biologically more realistic, and matches the audit's "FiLM-only cannot reach 5-HT" objection only under the implementation-discount reading.

**(b) The "we induce, we don't emit" framing is a feature, not a bug.** The project's inability to directly set $\gamma_B(c)$ under the FiLM-only route is a *match* to the biological reality: brains cannot set their effective horizons directly either. They shift effective horizons by modulating representation, attention, and precision, all of which are forward-pass operations. The control-vs-emergence trade-off the audit flagged for the effective-discount reading is the same trade-off the brain itself makes.

**(c) The construct-validity story strengthens substantially.** Under the implementation-discount reading, the project has to argue that the FiLM modulator "is" the 5-HT system because both modulate a discount-like quantity. Under the effective-discount reading, the project can argue that the FiLM modulator *implements the same mechanism* the brain does — gain modulation across a representation hierarchy with intrinsic timescale diversity — and the effective horizon shifts as the same kind of emergent consequence. The mapping is no longer "we built the 5-HT analogue;" it is "we instantiated the algorithmic principle 5-HT implements." Stronger, more honest, more publishable.

**(d) The v2 memo's 5-HT section probably needs reordering.** Currently the v2 memo orders the candidates with γ-conditional UVFA as Candidate 1 (simplest) and γ-Models as Candidate 3 (richest). Under the effective-discount reading, the natural ordering is to introduce a "Candidate 0" (FiLM-only, effective-discount-induction) ahead of all three. This is the *cheapest* candidate and the one that requires the *least* architectural extension beyond what NA and ACh already use. The user has explicitly said this integration is for later.

## 7. Pointers and integration status

This is a **standalone perspective memo**. It does not modify the v2 extended ideas memo, the audit, the audit's story doc, the v1 compact memo, or the v2 summary. The user will integrate as they see fit.

- **V2 extended creative ideas memo (the doc this reframing is most directly relevant to)**: [`docs/project/ideas/20260519_doya_modulation_extended_creative_ideas_v2.md`](20260519_doya_modulation_extended_creative_ideas_v2.md) — Section 6 (5-HT) is the section that would be reordered under integration.
- **V2 summary memo**: [`docs/project/ideas/20260521_doya_modulation_summary_v2.md`](20260521_doya_modulation_summary_v2.md) — the 5-HT row of the per-knob candidate table is the part that would shift.
- **Theoretical audit (with the 5-HT γ-conditional UVFA addendum)**: [`docs/project/critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md`](../critiques/20260518_film_as_hyperparameter_modulator_theoretical_audit.md) — the audit's "FiLM-only cannot reach 5-HT" verdict is the verdict this reframing partially reverses, under the effective-discount reading.
- **Gamma lit-review folder**: `docs/project/references/Gamma/` — the four lit-review files there (universal VF, multi-horizon, generative γ, non-exponential) are the references that ground the architectural side of the effective-discount route. The neuroscience anchors in this memo (Tanaka et al. 2007, Hasson, Murray, Damasio, Craig, Berridge & Robinson, Friston & Pezzulo) are *not* in those lit reviews and would need to be added if the project wants neuroscience-side references integrated.

**For the integration the user will do later:** the cleanest update would be to add a "Candidate 0 — FiLM-only effective-discount induction" as a fourth candidate to the v2 extended memo's Section 6, with a brief neuroscience-anchor paragraph drawn from this perspective doc, and to update the v2 summary's 5-HT row in the per-knob candidate table to acknowledge the new candidate. The neuroscience-anchor references would need to be added to the Gamma lit-review folder as a fifth review file or to a new neuroscience-anchors folder.
