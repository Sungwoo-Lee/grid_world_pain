---
title: "Independent audit — can LC-NA alone defend a five-effect policy-layer basket?"
status: draft
author: professor-neuromodulation
audience: user + pi + research-postdoc
date: 2026-05-18
scope: independent audit of LC-NA-alone basket framing (no project docs read)
---

## Question

Can a single biological exemplar — noradrenergic projection from locus coeruleus (LC-NA) — defensibly be claimed to instantiate all five of the following policy-layer effects simultaneously, via one upstream signal and one target-specific readout: (1) softmax temperature, (2) action-prior shift, (3) attentional bias at the decision stage, (4) policy precision in the active-inference sense, (5) exploration-bonus / explore-exploit arbitration?

## Headline

Conditional yes — but only with framing concessions. LC-NA is the *strongest single* candidate for effects (1), (4), and (5); it is a *plausible but not canonical* contributor to (3); and it is *not the canonical owner* of (2). A reviewer who is a neuromodulation specialist will let "LC-NA-analog" stand for (1), (4), (5) and probably (3), but will push back on (2) unless the framing is softened.

## Q1 — Per-effect canonical attribution

**(1) Softmax temperature / action-selection gain.** Canonically NA. Aston-Jones & Cohen (2005, *Annu. Rev. Neurosci.*) is the load-bearing reference: tonic LC firing flattens the action-selection function (high tonic → exploratory, low gain); phasic LC sharpens it. Servan-Schreiber, Printz & Cohen (1990, *Science*) gave the original "gain → temperature" formalisation. Eldar, Cohen & Niv (2013, *Nat. Neurosci.*) showed pupil-linked (NA-proxy) gain modulation reshaping representations consistent with temperature change. DA also affects choice stochasticity (Humphries, Khamassi & Gurney 2012; Beeler et al.), but the canonical *temperature* construct is NA-owned in the gain-theory tradition.

**(2) Action-prior shift.** Not canonically NA. The literature that talks about biasing *which* action is selected (independent of value) is dominated by **dopamine** in basal ganglia direct/indirect pathway models (Frank 2005; Collins & Frank 2014 OpAL): tonic DA shifts the Go/NoGo balance, which is an action-prior shift in everything but name. Serotonin also enters here via behavioural-inhibition / Pavlovian-instrumental priors (Dayan & Huys 2008, 2009; Crockett et al.; Boureau & Dayan 2011 opponency review). NA is largely absent from the canonical action-prior story. The closest NA-flavoured prior is the "network reset" of Bouret & Sara (2005, *TINS*) — but that resets the prior, it does not *shift* it in a content-specific way.

**(3) Attentional bias at the decision stage.** Mixed; both NA and ACh, with ACh historically dominant for *selective* attention and NA for *arousal-gated* attention. Sarter, Hasselmo, Bruno & Givens (2005) and Sarter & Lustig reviews place ACh as the canonical owner of cue-detection and signal-vs-noise filtering. Yu & Dayan (2005, *Neuron*) is the load-bearing computational account: ACh codes expected uncertainty (likelihood precision, top-down attentional weight), NA codes unexpected uncertainty (reset / re-weight). Corbetta & Shulman (2002) dorsal/ventral attention networks place NA in the *ventral* (reorienting / salience) stream, ACh in the *dorsal* (top-down) stream. So NA owns *some* attentional phenomena (reorienting, oddball, P3a) but not the full attentional-bias construct.

**(4) Policy precision (active-inference sense).** NA is the leading candidate in the Friston framework. Friston, FitzGerald, Rigoli, Schwartenbeck & Pezzulo (2017, *Neural Comput.*) and Parr & Friston (2017, 2019) explicitly identify NA (and to a lesser extent DA, depending on the paper) with precision on policy/state beliefs. The mapping "precision = postsynaptic gain on superficial pyramidal cells modulated by NA" is the canonical active-inference position. Note the active-inference literature is *not* unanimous (some papers put DA on policy precision, NA on state precision); but if you must name one modulator for policy precision, NA is defensible.

**(5) Exploration bonus / explore-exploit arbitration.** Canonically NA. Aston-Jones & Cohen (2005) tonic-vs-phasic adaptive gain *is* the explore-exploit theory in the NA literature. Pupillometry work (Jepma & Nieuwenhuis 2011; Gilzenrat et al. 2010) operationalises this. Note: there is a competing DA-as-exploration tradition (Frank et al. 2009 on COMT / striatal DA and directed exploration; Beeler on tonic DA and vigor-vs-exploration), and a 5-HT story (Cohen et al. on dorsal-raphe patience that *opposes* exploration). But the cleanest single-modulator owner of the explore-exploit arbitration construct is NA.

**Doya 2002 cross-check.** Doya's mapping puts NA = noise / inverse temperature, ACh = learning rate, DA = TD error, 5-HT = discount factor. Under Doya, your basket effects (1) and (5) are explicitly NA; (4) is not in Doya's frame (it's a later active-inference addition); (3) is ACh-territory; (2) is unmapped (Doya doesn't have an explicit action-prior term).

## Q2 — Defensibility of "one LC-NA-analog signal, one readout, five effects"

**Where canonical attribution is to ACh / DA / 5-HT, can receptor-density / co-release / interaction rescue the LC-NA framing?**

Partially, and only for some effects. Three concrete defences exist in the real literature:

1. **NA receptor subtype heterogeneity.** α1, α2, β1, β2 receptors have different affinities, different cellular targets, and different functional effects. Berridge & Waterhouse (2003, *Brain Res. Rev.*) is the standard reference for how the *same* NA signal produces qualitatively different effects in different targets through receptor-density differences. This is a genuine biological mechanism and lets you defend "one signal, target-specific readout → multiple effect-flavours" for *NA-adjacent* effects (gain, arousal, reorienting, precision).
2. **LC co-release of dopamine.** Recent work (Kempadoo et al. 2016; Takeuchi et al. 2016) shows LC terminals in hippocampus and elsewhere can co-release DA. This blurs the NA/DA boundary at the target and gives some cover for "LC-analog" producing DA-flavoured effects — but it is a *narrow* defence (hippocampus, novelty) and reviewers will not accept it as a general rescue for action-prior shifts in basal ganglia, which is where canonical action-prior lives.
3. **LC modulation of downstream DA / ACh systems.** LC projects to VTA, SNc, and basal forebrain, so LC activity *indirectly* modulates DA and ACh tone. This lets you say "LC-NA is upstream of the broader modulatory state". But this is a far weaker claim than "LC-NA instantiates these effects directly".

**Does the explicit disclaimer of one-to-one mapping rescue the framing?**

It helps but doesn't fully rescue. The disclaimer correctly anticipates that biology doesn't honour clean channel boundaries, and a reviewer will appreciate the epistemic honesty. But it pushes the problem one level down: if the readout produces effect-flavours that are *canonically owned* by ACh or DA, the reviewer will ask "then why call this LC-NA?" The disclaimer works for effects where NA is *one of* the canonical owners (the basket's effects 1, 3, 4, 5). It does *not* work for effect (2) action-prior shift, where NA is essentially absent from the canonical literature.

**What a neuromodulation-expert reviewer will push back on:**

- "Effect 2 (action-prior shift) is a basal-ganglia DA story (Frank 2005; Collins & Frank OpAL). You either need to drop this from the basket, rename it as 'reset of action prior' (Bouret & Sara network-reset, which is NA-canonical), or acknowledge it as a DA-flavoured leak."
- "Effect 3 (attentional bias) is ACh-dominant in the canonical selective-attention literature (Sarter; Yu & Dayan). NA owns the reorienting/salience attentional flavour but not top-down attentional bias. Specify which flavour you mean."
- "Effects 1, 4, 5 are NA-canonical and the framing is fine."
- "Receptor-density heterogeneity (Berridge & Waterhouse) is a real mechanism but does not extend across modulators — α1 vs α2 differentiation cannot manufacture DA-receptor or M1-receptor effects."
- "The Doya 2002 mapping is the obvious cross-check the field uses; explicitly say where you agree and where you diverge."

## Q3 — Recommended framing

I recommend **(a) with a partial (d) move on effect (2)**.

Concretely: keep "LC-NA-analog" as the single biological exemplar, *retain* effects (1), (3), (4), (5) in the basket, and **rename or drop effect (2)**. Two acceptable moves on (2):

- **Rename (preferred).** Replace "action-prior shift" with "action-prior *reset* / re-weighting", which maps cleanly onto Bouret & Sara (2005) LC-as-network-reset. A reset is content-neutral (you don't claim NA specifies *which* action becomes more likely; you claim it perturbs the prior so that re-learning happens). This is canonical NA territory.
- **Drop.** If the experimental design genuinely requires content-specific action-prior shifts (e.g., "the modulator should make action *up* more likely"), then this is DA / basal-ganglia work and shouldn't be in an LC-NA-framed basket. Drop it or move it explicitly to a DA-analog channel.

For effect (3), I'd add a single qualifying clause: "attentional bias of the reorienting / salience flavour (Corbetta & Shulman ventral attention; Yu & Dayan unexpected uncertainty)", not "selective attention" in the Sarter/ACh sense. This costs one sentence and pre-empts the obvious reviewer objection.

**Why not (b) "ascending neuromodulation"?** Because it gives up the predictive specificity. LC-NA makes specific predictions (pupil correlates, tonic-vs-phasic dynamics on the seconds-to-minutes timescale, β-adrenergic pharmacology) that "ascending neuromodulation" does not. Trading specificity for safety is a worse paper.

**Why not (c) "cocktail"?** Because if the architecture is genuinely one signal, one readout, calling it a four-modulator cocktail is overclaiming in the *opposite* direction — claiming biological richness the architecture doesn't have. The cocktail framing is only honest if the architecture instantiates multiple modulators (e.g., parallel signal streams with different timescales). If there's one signal, call it after the closest canonical owner and acknowledge bleed.

**Why not pure (d)?** Pure narrowing throws away effects (3), (4) that NA *can* legitimately own with one clause of framing care. That's an unnecessary sacrifice.

**Trade-off acknowledged.** (a)+(partial d) keeps breadth (4 of 5 effects retained) at the cost of one renaming and one qualifying clause. This is the framing-cheapest reviewer-defensible position.

## One-line verdict

**Conditional yes**: LC-NA can defend effects (1) softmax temperature, (3) reorienting-flavour attentional bias, (4) policy precision, and (5) exploration-exploitation arbitration as a single-exemplar basket, *if* effect (2) is renamed as "action-prior reset" (Bouret & Sara) or dropped — as raw "action-prior shift" it lives in DA / basal-ganglia territory and is not LC-NA's to claim.

## Canonical references named (selective)

- Aston-Jones & Cohen 2005, *Annu. Rev. Neurosci.* — adaptive gain theory, tonic vs phasic LC.
- Servan-Schreiber, Printz & Cohen 1990, *Science* — gain → temperature formalisation.
- Berridge & Waterhouse 2003, *Brain Res. Rev.* — NA receptor heterogeneity, target-specific effects.
- Bouret & Sara 2005, *TINS* — LC as network-reset.
- Sara 2009, *Nat. Rev. Neurosci.* — LC review.
- Yu & Dayan 2005, *Neuron* — ACh expected, NA unexpected uncertainty.
- Sarter, Hasselmo, Bruno & Givens 2005; Sarter & Lustig reviews — ACh and attention.
- Doya 2002, *Neural Networks* — modulator-to-RL meta-parameter mapping.
- Friston, FitzGerald, Rigoli, Schwartenbeck & Pezzulo 2017, *Neural Comput.* — active-inference precision.
- Parr & Friston 2017, 2019 — NA-as-policy-precision in active inference.
- Eldar, Cohen & Niv 2013, *Nat. Neurosci.* — pupil-linked gain reshapes representations.
- Frank 2005; Collins & Frank 2014 OpAL — DA Go/NoGo action-prior story.
- Boureau & Dayan 2011 — DA/5-HT opponency review.
- Dayan & Huys 2008, 2009 — 5-HT Pavlovian behavioural inhibition.
- Corbetta & Shulman 2002 — dorsal/ventral attention networks.
- Kempadoo et al. 2016; Takeuchi et al. 2016 — LC co-release of DA in hippocampus.
- Jepma & Nieuwenhuis 2011; Gilzenrat et al. 2010 — pupil-LC explore-exploit operationalisation.
