---
title: Biological Plausibility of Three-Site Shared-Core Modulation — Critique Memo
topic: neuromodulator-systems / biological-plausibility / H4 / H5
status: draft
author: professor-neuromodulation
created: 2026-05-07
last_updated: 2026-05-07
closes_gaps: biological-plausibility side of Claim 3 (H4 coordinated multi-domain), Claim 4 (H5 chronic-pain analog); plus AI-timescale-ordering check (Bayesian-brain memo §4.2) and headline-term check
---

# Biological Plausibility of Three-Site Shared-Core Modulation — Critique Memo

> One-line summary: **The single-GRU, three-injection-site architecture is in
> a biological trench-coat. Its three outputs map onto at least three
> distinct ascending modulatory systems (ACh-like at A, NE-like at C
> phasic-arousal, DA/5-HT-like at C value/patience and at B retention) that
> the literature explicitly does *not* treat as a single signal. As an
> engineering operationalisation of "coordinated affective state" this is
> fine; as a literal claim that one biological modulator does this work it is
> overclaimed. The H4 "coordinated multi-domain" hypothesis survives only in
> the weak form (the three computations co-vary because they share a slow
> latent driven by interoception); the strong form (one modulator analog
> performs all three) does not, and the paper should not be drafted as if it
> did. The H5 chronic-pain claim cannot survive a single-timescale modulator;
> it requires either an explicit tonic/phasic split or a multi-timescale
> (preferably opioid-analog vs. NE/ACh-analog) decomposition. The Bayesian-
> brain memo's predicted timescale ordering $\tau_A < \tau_B < \tau_C$ is
> directionally consistent with modulator kinetics (cholinergic / noradren-
> ergic phasic transients precede serotonergic / opioidergic tonic shifts),
> but a single GRU cannot satisfy it without parameter surgery — the
> ordering is an architecture predicate, not an emergent property.**

> Biological-plausibility verdict: **Partial → overclaimed in places.**
> Architectural intent (whole-network shared modulation tied to interoception)
> is biologically defensible at the level of "ascending modulatory analog".
> Specific claims attached to that architecture — that the same signal
> implements ACh-like precision, NE-like reset, DA-like vigor, 5-HT-like
> patience, and opioidergic descending control — are not. The minimum
> upgrade for H4 to be biologically defensible to a comp-neuro reviewer is a
> **two-modulator** architecture (one ACh/NE-analog phasic / fast, one
> 5-HT/opioid-analog tonic / slow); the minimum upgrade for H5 is the same
> upgrade.

> Anchors: [project_plan.md §3](../project_plan.md);
> [NEUROMODULATION_ALGORITHM.md §1.4 H1–H5](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md);
> [NMN_PERFORMANCE_DIAGNOSIS_v8.md](../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md);
> [pain_vs_nociception_construct.md §§5.2, 5.3, 7](pain_vs_nociception_construct.md → ../concepts/);
> [active_inference_hypervigilance.md §§3.2c, 4.2, 5](../concepts/active_inference_hypervigilance.md).

---

## 1. Audit: Which Modulator(s) Is the Project's Shared GRU Output Most Analogous To?

The project's modulator emits, from a single GRU hidden state $h_{\text{mod}}$, three (in PPO) or four (in DreamerV3) outputs:

- **A.** Encoder FiLM $\gamma$ (and $\beta$) on the perceptual encoder.
- **B.** Memory-gate bias $z_{\text{memory}}$ on the task GRU's update gate.
- **C-PPO.** Policy temperature $T_\pi$.
- **C-Dreamer.** Imagined reward scale $z_{\text{reward}}$.

I ask the literature: under the cleanest mapping (Doya 2002, *Neural Networks*, "Metalearning and neuromodulation"), which modulator does each output correspond to?

### 1.1 Doya 2002 baseline mapping

Doya proposes the canonical four-way assignment:

| Modulator | Algorithmic role | Project output it most resembles |
|---|---|---|
| **Acetylcholine (ACh)** | Learning rate; precision-on-likelihood; expected uncertainty | **Injection A** ($\gamma$ as channel-wise gain on sensory likelihood) |
| **Noradrenaline (NE)** | Noise / exploration / network reset under unexpected uncertainty | **Injection C-PPO** ($T_\pi$, action entropy) — partially |
| **Dopamine (DA)** | TD error, reward prediction, vigor | **Injection C-Dreamer** ($z_{\text{reward}}$, scaling imagined value) |
| **Serotonin (5-HT)** | Discount factor, patience, aversive/inhibitory control | **Injection B** ($z_{\text{memory}}$ retention) and **C-PPO** (caution) — split |

The architecture's three outputs therefore span at least three of Doya's four modulator slots, with the policy-temperature output sitting at the seam of NE (exploration) and 5-HT (aversive inhibition) and the memory output sitting at the seam of ACh (precision-on-prior) and 5-HT (slow integration).

### 1.2 Refinements that sharpen the mismatch

Doya's clean mapping has been refined and complicated in three directions, each of which deepens the mismatch with a single-GRU implementation.

**(a) ACh as expected uncertainty / precision on likelihood (Yu & Dayan 2005, *Neuron*).** ACh-mediated precision tracks the *likelihood* term in a Bayesian update — it scales how much sensory channels are trusted relative to prior. This is *exactly* the role Injection A plays under the active-inference reading (Bayesian-brain memo §1.2). So far the mapping holds: A is ACh-like.

**(b) NE as unexpected uncertainty and network reset (Yu & Dayan 2005; Aston-Jones & Cohen 2005, *Annu. Rev. Neurosci.*).** NE under the adaptive-gain theory has *two* distinct modes — tonic (sustained arousal, exploration) and phasic (event-locked gain burst, network reset). Phasic LC bursts are precisely time-locked to surprising / threatening events (Bouret & Sara 2005, *TINS*) and *transiently broaden the gain function on cortical pyramidal cells*. The project's policy-temperature drop after injury is a defensible NE-phasic analog *if* it is event-locked at lag $\sim 0$ steps. But the *direction* of NE's effect is to *broaden* exploration in the tonic mode and to *narrow / reset* in the phasic mode (Aston-Jones & Cohen 2005). The project conflates the two.

**(c) DA as RPE *and* policy-precision (Schultz 1998, *J. Neurophysiol.*; Niv et al. 2007, *Psychopharmacology*; Friston et al. 2014, *Brain*).** Phasic DA encodes RPE (Schultz/Montague/Dayan); tonic DA encodes vigor (Niv); the active-inference re-reading casts DA as precision on policy beliefs (Friston). The project's $z_{\text{reward}}$ output in Dreamer is *closest* to the tonic-vigor / policy-precision reading (it scales imagined returns, broadcast across the imagination rollout) — *not* the phasic RPE reading. This is a DA-tonic-analog.

**(d) 5-HT as aversive control and patience (Daw, Kakade & Dayan 2002, *Neural Networks*; Dayan & Huys 2008, *PLoS Comp. Bio.*; Cohen et al. 2015, *eLife*).** Daw 2002 proposes 5-HT as the *opponent* of DA on the aversive axis: where DA reports appetitive RPE, 5-HT reports aversive prediction error and reflective behavioural inhibition. Dorsal-raphe 5-HT also encodes patience (Cohen et al. 2015), i.e. discount-factor analog. The project's Injection B (memory retention bias) is most cleanly read as 5-HT-like in the patience / discount sense; Injection C-PPO's temperature drop in dangerous regions is most cleanly read as 5-HT-like in the aversive-inhibition sense.

**(e) Opioid descending modulation (Eippert et al. 2009, *Neuron*; Wager & Atlas 2015, *Nature Reviews Neuroscience*; Wiech 2016, *Science*).** Endogenous opioids in the periaqueductal gray gate nociceptive transmission *descendingly* — they implement an *expectation-driven* down-modulation of the pain signal. The project has **no analog of this system at all**. Yet the construct claim being made (pain *computation*, with placebo-style dissociation between $o_{\text{noc}}$ and $p(s_t)$, pain-modeling memo §1 property 3) requires exactly an opioid-analog descending modulator, because that is the empirical substrate of placebo analgesia in humans. If the paper goes for the strong "pain computation" headline (which both pain-modeling §4.4 and Bayesian-brain §5 advise against without C1 dissociation), the absence of an opioid-analog is the single largest unaddressed substrate gap.

### 1.3 Verdict on the mapping

The cleanest fair statement is that the shared-GRU output is a **composite "ascending modulatory analog with interoceptive read"** — defensible as a *functional* analog at the broadest level, not as a 1-to-1 substrate analog. Specifically:

- **A is ACh-like.** Strongest mapping, holds under both Doya 2002 and Yu & Dayan 2005. The "$\gamma$ as precision-weighted gain on sensory likelihood" reading is exact.
- **B is 5-HT-like (patience axis) with a weaker NE-tonic component.** Memory retention bias is most cleanly a discount/patience signal. Under H2 ("hold onto threat information longer"), the bias direction matches dorsal-raphe 5-HT's role in extending integration windows (Cohen et al. 2015).
- **C-PPO is at the NE/5-HT seam.** Phasic temperature drop = NE phasic-reset analog. Persistent caution in dangerous regions = 5-HT aversive-inhibition analog. *These should not be the same signal in biology* (Daw 2002; Cools et al. 2008, *Nat. Rev. Neurosci.*).
- **C-Dreamer is DA-tonic-like.** Reward-scale broadcast = vigor / policy-precision analog (Niv 2007; Friston 2014). It is *not* a phasic-RPE analog.

Where the mapping is loose or wrong:

- The shared GRU treats ACh-like, NE-like, 5-HT-like, and DA-like signals as outputs of *one* slow latent. In the biology, **ACh and NE are partially opponent on the expected/unexpected-uncertainty axis** (Yu & Dayan 2005); **DA and 5-HT are explicitly opponent on the appetitive/aversive axis** (Daw, Kakade & Dayan 2002); **NE phasic and NE tonic are opponent within a single neuromodulator** (Aston-Jones & Cohen 2005). A single slow latent driving all four cannot reproduce these opponencies; it can only reproduce their (signed) sum.
- The opioid descending-modulation system is absent, despite being the substrate the headline term "pain computation" most directly invokes.

---

## 2. What Can the Paper Claim About Biological Plausibility?

This is the load-bearing question — what is the strongest H4 framing the architecture survives, and what would the minimum upgrade be to support a stronger framing?

### 2.1 H4 in the weak form ("shared affective tone") — defensible

The weakest H4 reading is:

> The agent's perceptual gain, memory retention, and policy caution co-vary
> across time because they are all read out of a *common* slowly-evolving
> interoceptive latent. The latent itself is not committed to being any one
> biological modulator; it is an *abstract affective tone* whose biological
> substrate is presumably distributed across multiple ascending systems.

This is biologically defensible. There is independent evidence that under threat / pain / arousal, the major ascending systems do *co-activate* (Mather et al. 2016, *Behav. Brain Sci.* — the GANE framework; Sara & Bouret 2012, *Neuron* — LC coupling to amygdala, basal forebrain, and raphe under salience). The shared-latent read of H4 is a *coarse* but fair operationalisation of this co-activation. **Under this reading, the project's H4 is defensible.**

### 2.2 H4 in the strong form ("one modulator does it all") — overclaimed

The strong reading would be:

> One ascending neuromodulatory system (e.g. NE, or ACh, or "the
> interoceptive modulator") simultaneously implements precision-weighted
> perception, memory retention, and policy caution.

This is **not biologically defensible**. The Yu–Dayan, Daw, and Aston-Jones literatures all argue that the biology *partitions* these computations across at least two opponent pairs (ACh vs. NE on uncertainty; DA vs. 5-HT on valence). A single-modulator strong-H4 claim collapses opponencies that the biology does not collapse.

The project plan §3 reads as:

> "neuromodulators (acetylcholine, noradrenaline, dopamine, serotonin,
> opioidergic systems) do not act on a single cognitive domain. They
> simultaneously adjust sensory gain, memory persistence, exploration drive,
> and reward sensitivity, producing the coordinated shifts in state we
> recognize as affective tone."

This sentence is correct *as a description of the union over modulators* but is rhetorically used to motivate a *single shared core*. A comp-neuro reviewer will read this as a category error — the union over *several* modulators is not the same as the output of *one* modulator, and the architectural choice of one shared GRU silently picks the latter. **Recommendation:** rewrite project_plan.md §3 to make explicit that the shared GRU is a *coarse* analog of the *ensemble* of modulators co-activated under interoceptive arousal, not of any single system.

### 2.3 Minimum upgrade to defend strong H4

If the paper wants to keep the strong-H4 framing, the minimum architecture change is **two modulators with explicit tonic/phasic and uncertainty-axis decomposition**:

- **Modulator T (tonic-slow, 5-HT/opioid analog).** Long timescale ($\tau_T \sim 50$–$200$ steps in this gridworld). Drives Injection B (memory retention) and the persistent component of Injection C (sustained risk-aversion). Substrate analog: dorsal raphe 5-HT + descending opioid.
- **Modulator P (phasic-fast, ACh/NE analog).** Short timescale ($\tau_P \sim 1$–$5$ steps), event-locked. Drives Injection A ($\gamma$ on threat channels) and the phasic component of Injection C (acute reset / temperature drop at injury onset). Substrate analog: basal forebrain ACh + LC NE.
- **Coordination mechanism.** A learned coupling $T \leftarrow P$ (phasic events drive tonic state changes — the standard biological readout, Sara 2009, *Nat. Rev. Neurosci.*) and a separate weak $P \leftarrow T$ feedback (tonic state biases phasic responsivity — adaptive-gain framing). The *learned coordination* is what an H4 strong claim would need to demonstrate, and is what makes the paper's H4 a finding rather than an architectural assumption.

This is a non-trivial change but is in the same complexity class as the Phase-3 precision head. A paper that wants to make the strong-H4 claim has to commit to it.

### 2.4 What I do *not* recommend

- A four- or five-modulator full Doya implementation. The combinatorial cost (per-modulator ablation tables, opponent-pair tests) is incommensurate with the paper's scope. Two-modulator is the pragmatic bound.
- A learned modulator-routing mechanism that "discovers" how many modulators are needed. This is a paper of its own, not a sub-claim.

The **two-modulator (T, P) split** is the single highest-leverage upgrade for H4 biological-plausibility. If the user's calendar permits it, I recommend Phase 3.5 between current Phase 3 and Phase 4.

---

## 3. The H5 Chronic-Pain Claim — Adjudication

The pain-modeling memo §5.2 explicitly hands me this verdict. I take it.

### 3.1 The current operationalisation is the wrong tool

H5 ([NEUROMODULATION_ALGORITHM.md §1.4](../../develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md)) operationalises chronic-pain dynamics as:

> "The duration of this persistence is controlled by the GRU's effective
> timescale relative to the environment's injury/healing dynamics."

and the Phase 4 readout is the modulator-GRU timescale sweep (`mod_hidden_size` varied). The pain-modeling memo flagged this as "a recurrent-inertia story mis-labelled as chronic pain". From the modulator-systems side, I concur, and I can sharpen the verdict.

**A single-GRU modulator cannot, in principle, exhibit a clinically-meaningful chronic-vs-acute dissociation,** for three reasons.

**(a) Single timescale is not chronicity.** Recurrent inertia in a single GRU is parameterised by one effective timescale $\tau_{\text{mod}}$ (set by the spectral radius of the recurrent weight matrix; for `mod_hidden_size` $\to$ large, $\tau_{\text{mod}}$ rises but remains a *single* number per training run). Clinical chronic pain is not "a longer acute pain" — it is a *qualitative regime change* in which the controller's recovery dynamics fail, the pain posterior decouples from peripheral input, and the system enters a *new attractor* (Apkarian et al. 2009, *Prog. Neurobiol.*; Vlaeyen & Linton 2000, *Pain*). Sweeping `mod_hidden_size` parameterises the speed of return to a *single* attractor, not the existence of *two*.

**(b) The biology of acute-vs-chronic is multi-system and explicitly multi-timescale.** Acute pain is dominated by phasic ACh / NE / glutamatergic signalling on a seconds timescale. Chronic pain involves sustained reorganisation in 5-HT, DA, and opioid systems on a hours-to-weeks timescale, with descending opioid tone *failing* in chronic pain (the "endogenous analgesia failure" hypothesis; Bannister & Dickenson 2017, *Anaesthesia*). The two are not the same system at different speeds — they are *different systems*. A one-GRU architecture cannot encode this distinction.

**(c) Chronicity in the construct sense requires *learned* representations that generalise.** Pain-modeling §5.2 makes the deeper point: chronic pain is the failure of a *learned controller* in the Vlaeyen-Linton sense. The H5 claim therefore needs the modulator to acquire, through training, a representation that *generalises* to novel post-recovery contexts — not just to decay slowly. A timescale sweep cannot test this; only a transfer probe (gap G-G) can.

### 3.2 What a biologically-defensible H5 would need

The minimum upgrade is the same two-modulator (T, P) split of §2.3, with two additions specific to H5:

- **Asymmetric coupling.** $T$ is driven slowly by integrated $P$ activity, *but $T$ does not decay back to baseline at the same rate it rose*. In the biology, chronic-pain transitions involve hysteresis in 5-HT and opioid systems (Apkarian; Bannister). Architectural analog: $T$'s decay is gated by a learned variable, so under sufficient $P$ excursion, $T$ enters a *high-tonic absorbing state* that does not relax even after $P$ returns to baseline. This is the substrate-level analog of a chronic-pain transition.
- **Descending-opioid analog.** A *predicted-pain → pain-signal-suppression* pathway: a head that, conditional on the modulator's predicted-injury level, *down-modulates* the nociceptive input itself (placebo analgesia analog). Without this, the placebo-style dissociability that pain-modeling §1 property 3 demands has no architectural substrate; chronic pain as "controller failure of opioid descending tone" cannot be modelled.

If these two additions land, H5 can be operationalised as:

- *Acute regime:* high $P$, transient $T$ excursion, descending-opioid-analog active.
- *Recovery regime:* $P$ returns to baseline, $T$ relaxes, descending-opioid-analog gates persistent caution.
- *Chronic regime:* repeated/prolonged $P$ excursions push $T$ into the absorbing state; descending-opioid-analog fails to suppress nociception under expected-injury prior; behavioural caution persists *and the agent's posterior over $s_t$ remains elevated even under low $o_{\text{noc}}$*.

This is a pre-registrable, falsifiable, biologically-grounded chronic-pain claim. The current single-GRU timescale sweep is not.

### 3.3 If the upgrade is out of budget

If the user cannot fit the two-modulator + descending-opioid-analog upgrade into the timeline, **H5 should be downgraded** and the term "chronic-pain analog" should not appear in the paper. The fallback claim is:

> "Modulator-state inertia controls the persistence of post-injury caution
> on the order of the GRU's effective timescale; we do not claim this
> models the chronic-pain transition."

This is honest. The pain-modeling memo's §5.2 already recommended this fallback wording. From the modulator-systems side I confirm: there is no biologically defensible path from single-GRU-timescale-inertia to chronic pain. **The claim either takes the upgrade or takes the downgrade.**

---

## 4. The Bayesian-Brain Timescale Ordering — Adjudication

The Bayesian-brain memo (§4.2) predicts:

$$
\tau_{\text{precision update}} \;\lesssim\; \tau_{\text{modulator GRU}} \;\lesssim\; \tau_{\text{policy persistence}}
$$

i.e. $\tau_A < \tau_B < \tau_C$, with each ratio $\geq 2$. The biology question is whether this ordering is consistent with known modulator kinetics, and whether a single GRU can satisfy it.

### 4.1 Biological consistency of the ordering — yes

The ordering is directionally consistent with the substrate-analog mapping of §1.3:

- **A (ACh-like, fastest).** Cortical ACh transients are sub-second to seconds (Sarter & Lustig 2019, *Curr. Opin. Behav. Sci.*; Parikh et al. 2007, *Neuron*). ACh-mediated precision updates are the fastest of the four canonical modulators.
- **B (5-HT-like / NE-tonic-like, intermediate).** Dorsal-raphe 5-HT integration windows for patience-relevant contexts are seconds to tens of seconds (Cohen et al. 2015); NE-tonic shifts are on tens-of-seconds (Aston-Jones & Cohen 2005). Both are slower than ACh-phasic.
- **C (5-HT-aversive / DA-tonic / opioid-tonic, slowest).** Sustained risk-aversion under injury-elevated prior is mediated by the slow components of 5-HT (Dayan & Huys 2008) and by tonic descending opioid tone (Bannister & Dickenson 2017). These are tens-of-seconds to minutes.

So the *biological* ordering is broadly $\tau_{\text{ACh-phasic}} < \tau_{\text{5-HT/NE-tonic}} < \tau_{\text{5-HT-aversive/opioid-tonic}}$, which lines up with $\tau_A < \tau_B < \tau_C$ on the project's substrate-analog mapping. The Bayesian-brain memo's prediction is biologically compatible.

### 4.2 Can a single GRU satisfy it? No — by construction

A single GRU has *one* effective timescale per hidden unit, set by the spectral radius of the update gate. The three injection-site outputs are linear (FiLM) or low-capacity (precision/temperature head) read-outs of the same hidden state. Their effective timescales are bounded above by $\tau_{\text{mod}}$ and bounded below by the read-out filter (typically the output head's decay, which is essentially zero for instantaneous heads).

In the limit of an instantaneous read-out head, *all three injection-site signals share the same effective timescale*, $\tau_{\text{mod}}$. The ordering $\tau_A < \tau_B < \tau_C$ with each ratio $\geq 2$ cannot emerge — there is one number, not three.

The ordering can be approximated *only* if:

1. The read-out heads themselves have differentiable temporal filters (e.g., $z_{\text{memory}}$ is a low-pass of $h_{\text{mod}}$ with a learned coefficient, while $\gamma$ is the instantaneous read-out). This is a small architectural change and is the *cheapest* way to satisfy the prediction.
2. Or, the modulator is decomposed into multiple hidden subspaces with explicitly-different recurrent timescales (the multi-timescale RNN literature: Tallec & Ollivier 2018; the project's own [NEUROMODULATION_ALGORITHM.md Block I §5 Durstewitz et al. 2025] is in this lineage).
3. Or, the two-modulator (T, P) split of §2.3 — which automatically produces $\tau_P < \tau_T$.

**Verdict.** The timescale ordering is biologically consistent but is *architecturally a predicate, not an emergent property*, of the single-GRU design. If the paper wants to pre-register this as a falsifiable AI prediction (Bayesian-brain memo §4.3 item 4), it must first commit to one of (1)/(2)/(3). Without that, the ordering claim is either trivially true (if the read-outs all share $\tau_{\text{mod}}$, the "ordering" is 1:1:1, which fails the $\geq 2$ ratio) or trivially false. **It cannot survive the single-GRU substrate as written.**

The lowest-cost architectural commitment that makes the ordering testable is option (1): give Injection B's $z_{\text{memory}}$ an explicit low-pass filter on $h_{\text{mod}}$ with a learnable time constant, and give Injection C's policy-temperature an even slower learnable filter. This is small, biologically defensible (the read-out filter is the post-synaptic integration time of the modulator's target population, which differs across A, B, C target sites), and turns the ordering into a measurable property of the trained model.

---

## 5. Headline-Term Verdict (From the Neuromodulation Side)

Pain-modeling §4.4 and Bayesian-brain §5 both recommend the sober register, "interoceptive neuromodulation as a substrate for pain-like behaviour", until the C1 dissociation is in hand. From the modulator-systems side I add a third register specific to this literature:

### 5.1 The three registers

- **Register 1 — "Interoceptive precision-weighted perception"** (Bayesian-brain level 1). Architectural claim only. Defensible from Phase 3 alone.
- **Register 2 — "Interoceptive neuromodulation as a substrate for pain-like behaviour"** (sober, recommended by both prior memos). Defensible if C0/C1/C2 dissociate.
- **Register 3 — "Pain computation"** (strong, recommended-against by both prior memos). Requires Register 2 + transfer.

### 5.2 The neuromodulation-specific register I add

Between Register 1 and Register 2, there is a meaningful intermediate that the modulator-systems literature uses naturally:

> **"An interoceptive ascending modulatory analog producing coordinated
> perceptual–mnemonic–policy reweighting under bodily threat"**

This is the *biology-side* register. It explicitly:

- Names the architecture as an *analog*, not a literal modulator.
- Names the *coordination* (the H4 weak form of §2.1) as the load-bearing claim.
- Does not commit to "pain" as a construct — defers that to the pain-modeling memo's controls.
- Is honest about what the shared-GRU design actually is.

This register survives even the *single-GRU* architecture (no Phase 3.5 upgrade required), because it does not claim that the shared signal is one biological modulator — only that it is an analog of the *ensemble* read-out (Mather et al. 2016 GANE; Sara & Bouret 2012). It is the strongest register the *current* architecture can carry.

### 5.3 Recommendation

- **Title and abstract:** Register 2 (sober). Concur with pain-modeling §4.4 and Bayesian-brain §5.
- **Body of paper, mechanism sections:** Register R-bio (the §5.2 register above) when describing what the architecture *is*. This is more accurate than calling it "the neuromodulator" and is publication-credible to a comp-neuro reviewer.
- **Discussion:** Register 3 may be foreshadowed *only* conditional on (i) all three controls dissociating, (ii) cross-rung transfer, *and* (iii) the H5 chronic-pain claim taking the upgrade of §3.2. Without (iii), the paper should not foreshadow Register 3 at all — the chronic-pain failure mode is the strongest evidence against a "pain computation" claim, and a careful pain-modelling reviewer will read the absence of a chronic regime as evidence that the system is doing *acute* affective modulation only.

The phrase "pain syndrome" in project_plan.md §3 is, from the modulator-systems side, the most overclaimed phrase in the document. I concur with pain-modeling §4.3 to drop it. A defensible substitute is "**the joint perceptual–mnemonic–policy reweighting component of post-injury behavioural change**".

---

## 6. The v8 Null Result Read in Modulator-System Terms

A short note that reinforces the diagnosis. The v8 null ([NMN_PERFORMANCE_DIAGNOSIS_v8.md §4](../../develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md), [project_plan.md §4 cause 1](../project_plan.md)) is "γ collapses to near-identity, modulator effectively bypassed". The Bayesian-brain memo (§3.1) reads this as the *correct* free-energy solution to a homogeneous-noise environment. I add the modulator-systems reading:

**Doya 2002 requires teaching signals.** Each of ACh / NE / DA / 5-HT in Doya's framework has a *distinct teaching signal* — an environmental / homeostatic readout that drives its update. ACh tracks expected likelihood-precision; NE tracks unexpected uncertainty; DA tracks RPE; 5-HT tracks long-horizon discounted return. The project's modulator has *one* teaching signal: the RL loss. **A single teaching signal cannot select among the four Doya roles.** In the homogeneous-noise environment, the RL loss does not even disambiguate "be a precision modulator" from "be silent"; the modulator's optimal behaviour given that loss is exactly the v8 collapse.

This is an independent argument for Phase 3's heteroscedastic precision loss (Bayesian-brain §3.2b): it provides the *missing teaching signal* for the ACh-precision component of the modulator, and only that component. The other components (NE-reset, 5-HT-patience, DA-vigor, opioid-descent) remain without teaching signals under the current design. Even after Phase 3, the modulator is still under-determined as a multi-modulator analog — only its A-injection (ACh-like) is being told what to compute. **This is another argument for the two-modulator T/P upgrade**: each modulator should have its own teaching signal (e.g., $P$ trained on heteroscedastic precision; $T$ trained on a sustained-survival auxiliary or on the policy's expected-free-energy pragmatic-value gradient). Without per-modulator teaching, the multi-modulator analog is rhetorical even if the architecture is multi-modulator.

---

## 7. Biological-Plausibility Verdict and Recommendations

### 7.1 Verdict

| Item | Verdict |
|---|---|
| Shared-GRU architecture as "ascending modulatory analog" (broad sense) | **Defensible.** |
| Shared-GRU architecture as "one modulator" (literal sense) | **Overclaimed.** |
| H4 weak form (shared-latent coordination) | **Defensible.** |
| H4 strong form (one biological modulator) | **Overclaimed.** Minimum upgrade: two-modulator T/P split (§2.3). |
| H5 chronic-pain claim under single-GRU timescale sweep | **Not defensible.** Minimum upgrade: T/P split + descending-opioid-analog (§3.2). |
| H5 fallback ("modulator inertia analog") under single-GRU | **Defensible** if the term "chronic pain" is removed. |
| Bayesian-brain timescale ordering $\tau_A < \tau_B < \tau_C$ under single GRU | **Architecturally a predicate, not emergent.** Lowest-cost fix: per-injection-site read-out filters with learnable time constants (§4.2 option 1). |
| Headline term "pain computation" | **Concur with pain-modeling §4.4 and Bayesian-brain §5: not licensed yet.** Use Register 2 (sober) for title/abstract; Register R-bio (§5.2) in body. |
| Headline phrase "full pain syndrome" in §3 of plan | **Drop.** |

### 7.2 Concrete recommendations (ordered by priority and decoupled by professor agent that should pick them up)

**For `senior-developer` (architecture).** *If* the user wants to defend the strong H4 and the H5 chronic-pain claim:
1. Implement the two-modulator T/P split (§2.3): two GRU cores with explicitly different timescales, learned phasic→tonic coupling.
2. Add an opioid-analog descending-modulation head (§3.2): a learned multiplicative gate on the nociception channel itself, conditioned on the tonic modulator state.
3. Per-injection-site read-out low-pass filters with learnable time constants (§4.2 option 1) — do this *regardless* of whether (1) and (2) are taken; it is small and unlocks the timescale-ordering pre-registration.

If the user takes only the third item, the architectural change is minimal but only Register 2 + sober-H5 + testable timescale-ordering are defensible.

**For `experiment-designer`.** Per-modulator teaching-signal audit (§6): document, for each output of the modulator (γ, $z_{\text{memory}}$, $T_\pi$, $z_{\text{reward}}$), what gradient signal is teaching it. Where the answer is "only the RL loss", flag it. The output of this audit drives whether additional auxiliary losses are needed beyond Phase 3's heteroscedastic precision loss.

**For `professor-rl-bayesian-dl`** (running in parallel; will see this memo at synthesis time). The architectural reconciliation question: is the two-modulator T/P split implementable without breaking the Phase-3 precision-head plumbing? The shared-GRU choice is in part a complexity-budget choice. RL-BDL is the right agent to rule on whether the upgrade is in budget. *If RL-BDL judges the upgrade out of budget, my recommendation is that the paper take the downgrade* (Register 2 + sober H5) rather than the upgrade — it is more defensible to ship the smaller claim than to ship the bigger architecture incomplete.

**For `professor-pain-modeling`**: the descending-opioid analog of §3.2 is a substrate-level analog of placebo analgesia. If the user wants to make even a sotto voce placebo claim in the discussion, this is the architectural pre-condition. Cross-link.

**For `professor-bayesian-brain`**: the timescale-ordering pre-registration of §4.2 is unlocked by per-injection-site read-out filters. Confirm that the AI fingerprint of Bayesian-brain §4.1 is robust to this small architectural change (it should be; the change does not affect the fixed-point structure, only the per-output integration window).

---

## 8. References (papers anchoring §§1–4)

Papers cited above. Where not already in `docs/project/references/`, they should be added so the modulator-systems lineage is documented.

- Apkarian, A.V., Baliki, M.N., Geha, P.Y. (2009). "Towards a theory of chronic pain." *Progress in Neurobiology* 87, 81–97. — chronic pain as controller failure.
- Aston-Jones, G., Cohen, J.D. (2005). "An integrative theory of locus coeruleus-norepinephrine function: adaptive gain and optimal performance." *Annual Review of Neuroscience* 28, 403–450. — phasic vs. tonic NE.
- Bannister, K., Dickenson, A.H. (2017). "The plasticity of descending controls in pain: translational probing." *J. Physiol.* 595, 4159–4166. — descending opioid failure in chronic pain.
- Bouret, S., Sara, S.J. (2005). "Network reset: a simplified overarching theory of locus coeruleus noradrenaline function." *TINS* 28, 574–582. — LC reset.
- Cohen, J.Y., Amoroso, M.W., Uchida, N. (2015). "Serotonergic neurons signal reward and punishment on multiple timescales." *eLife* 4, e06346. — 5-HT and patience.
- Cools, R., Roberts, A.C., Robbins, T.W. (2008). "Serotoninergic regulation of emotional and behavioural control processes." *TINS* 31, 31–40. — DA/5-HT opponency.
- Daw, N.D., Kakade, S., Dayan, P. (2002). "Opponent interactions between serotonin and dopamine." *Neural Networks* 15, 603–616. — opponent DA/5-HT.
- Dayan, P., Huys, Q.J.M. (2008). "Serotonin, inhibition, and negative mood." *PLoS Comp. Biol.* 4, e4. — reflective behavioural inhibition.
- Doya, K. (2002). "Metalearning and neuromodulation." *Neural Networks* 15, 495–506. — the canonical four-way mapping.
- Durstewitz, D. (2025). [as in NEUROMODULATION_ALGORITHM.md Block I §5] — multiscale plasticity.
- Eippert, F., Bingel, U., Schoell, E.D., Yacubian, J., Klinger, R., Lorenz, J., Büchel, C. (2009). "Activation of the opioidergic descending pain control system underlies placebo analgesia." *Neuron* 63, 533–543.
- Friston, K., Schwartenbeck, P., FitzGerald, T., Moutoussis, M., Behrens, T., Dolan, R.J. (2014). "The anatomy of choice: dopamine and decision-making." *Phil. Trans. B* 369, 20130481. — DA as policy precision.
- Mather, M., Clewett, D., Sakaki, M., Harley, C.W. (2016). "Norepinephrine ignites local hotspots of neuronal excitation: how arousal amplifies selectivity in perception and memory." *Behav. Brain Sci.* 39, e200. — GANE; multi-modulator co-activation under arousal.
- Niv, Y., Daw, N.D., Joel, D., Dayan, P. (2007). "Tonic dopamine: opportunity costs and the control of response vigor." *Psychopharmacology* 191, 507–520. — DA-tonic vigor.
- Parikh, V., Kozak, R., Martinez, V., Sarter, M. (2007). "Prefrontal acetylcholine release controls cue detection on multiple timescales." *Neuron* 56, 141–154. — ACh phasic transients.
- Sara, S.J. (2009). "The locus coeruleus and noradrenergic modulation of cognition." *Nat. Rev. Neurosci.* 10, 211–223. — LC overview; phasic-drives-tonic.
- Sara, S.J., Bouret, S. (2012). "Orienting and reorienting: the locus coeruleus mediates cognition through arousal." *Neuron* 76, 130–141. — LC coupling.
- Sarter, M., Lustig, C. (2019). "Cholinergic double duty: cue detection and attentional control." *Curr. Opin. Behav. Sci.* 26, 102–109. — ACh and cue detection.
- Schultz, W. (1998). "Predictive reward signal of dopamine neurons." *J. Neurophysiol.* 80, 1–27. — DA RPE.
- Tallec, C., Ollivier, Y. (2018). "Can recurrent neural networks warp time?" *ICLR.* — multi-timescale RNN architectures.
- Vlaeyen, J.W.S., Linton, S.J. (2000). "Fear-avoidance and its consequences in chronic musculoskeletal pain." *Pain* 85, 317–332. — fear-avoidance / controller failure.
- Wager, T.D., Atlas, L.Y. (2015). "The neuroscience of placebo effects." *Nat. Rev. Neurosci.* 16, 403–418. — placebo / opioid descending.
- Wiech, K. (2016). "Deconstructing the sensation of pain." *Science* 354, 584–587. — pain as constructed perception.
- Yu, A.J., Dayan, P. (2005). "Uncertainty, neuromodulation, and attention." *Neuron* 46, 681–692. — ACh expected vs. NE unexpected uncertainty.
