---
title: "Predictions for v3 direction memo §5 — biological-substrate side"
parent: docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md
reviewer: professor-neuromodulation
date: 2026-05-16
inputs:
  - docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v2.md
  - docs/project/references/neuromodulatory_algorithms/neuromodulatory_algorithms_predictions_synthesis.md
---

# Predictions for v3 direction memo §5 — biological-substrate side

## Plain-English entry point

The project's neuromodulator network (NMN, a small side-circuit watching state and emitting scaling/shifting signals into the main policy) claims to be two things at once: a cellular-level account of how brain neuromodulators work (they change the slope and operating point of target neurons' input-output curves) and a behaviour-level account of how reinforcement-learning hyperparameters (how fast you learn, how much you explore, how much you trust noisy evidence) get retuned moment-by-moment. The predictions synthesis flagged that no published paper in the corpus tests both readings on the same agent on the same trials — that is the gap v3 is meant to fill. This memo proposes eight predictions from the biological-substrate side: which neuromodulator system the NMN's single recurrent state most likely implements; which behavioural signatures separate brief (phasic) from sustained (tonic) modulator action; which cellular mathematical form (slope-rescaling vs operating-point-shift vs divisive-normalisation) the FiLM substrate instantiates at each of the project's three injection sites; what correlated movements across sites would signal a single channel doing several jobs versus genuinely multi-channel modulation; and whether different injection sites should be read as targeting different cell classes the way real neuromodulators do. Each prediction names a canonical paper, names the project-testbed measure that maps to it, and names the null result that would refute it.

## Predictions

### P1 — `mod_h` is a noradrenaline / inverse-temperature signal until proven otherwise

**Claim.** Under the v2 architecture (single shared `mod_h` recurrent state feeding three FiLM injection sites), the `mod_h` trajectory should look like a Locus-Coeruleus (LC) noradrenaline (NA) signal: a phasic burst at regime changes, followed by relaxation toward a tonic baseline that is itself sensitive to slow drift in task uncertainty. This is the default channel attribution; competing readings (acetylcholine-as-learning-rate, dopamine-as-RPE, 5-HT-as-discount) make sharper predictions only after the Phase 0 T/P split lands.

**Biological grounding.** Aston-Jones & Cohen 2005 adaptive-gain theory (canonical LC phasic-burst-on-task-change + tonic shift between contexts), Wainstein et al. 2025 Fig. 1C (pupil-locked perceptual switches as the empirical anchor for LC-gain-driven regime transitions), Doya 2002 §3.3 (NA as inverse-temperature β), Lee et al. 2024 §3.4 Eq. 4 (β⁻¹(s) = 1/⟨E(s, â)⟩ — NA temperature as inverse of mean aleatoric uncertainty over actions). The reasoning: of the three injection sites, only site C (policy temperature) has a *direct, readable* RL-hyperparameter consequence — scaling actor logits literally is an inverse-temperature change. Sites A and B are then most parsimoniously "passenger heads" until shown otherwise.

**Project-testbed measure.** Regress `mod_h` (or, post-T/P, the phasic recurrent state `mod_h_P`) on (i) actor-entropy peri-regime-change and (ii) |δ| (absolute TD error / critic residual). Window: 50–200 episodes pre- vs post-stage-switch. Logged WandB scalars: `train/entropy_loss`, `train/value_loss`, plus a new `train/mod_h_mean`, `train/mod_h_corr_delta`. Co-time-lock peaks within ±5 episodes of each regime change.

**Falsification signature.** Null cross-correlation between `mod_h` and both actor-entropy *and* |δ|, combined with a strong correlation between `mod_h` and a one-hot regime label, refutes the NA reading. In that case `mod_h` is a context detector (Vecoven 2020 family, predictions synthesis §2 variant (f)) — neither a Doya channel nor a gain-modulator per se. The unification claim then collapses to "the substrate could in principle be a neuromodulator, but the trained instance is not behaving as one".

---

### P2 — Phasic and tonic readouts dissociate; only the phasic state tracks |δ|, only the tonic state tracks cumulative cost

**Claim.** After the Phase 0 T/P split (two recurrent states with separable phasic and tonic timescales, v2 §3 prerequisite), the phasic state P should track per-episode |δ| and lock to regime-change events at the episode timescale; the tonic state T should track a slow integral of cost / aversive-signal exposure across episodes and remain elevated long after a regime change has passed. The two states should de-correlate at lag 0 but co-vary at large positive lags (P-spike now → T-elevation later).

**Biological grounding.** Aston-Jones & Cohen 2005 §"Tonic-phasic modes" — the canonical LC dissociation: phasic bursts time-locked to salient events on a ~100 ms scale, tonic baseline shifts on a seconds-to-minutes scale, with different downstream effects (phasic → exploitation/focus, tonic → exploration/disengagement). Rodriguez-Garcia et al. 2026 Algorithm 1 (g(t+1) = γg(t) + (1-γ)g₀ + ηH(y)) is the closest in-corpus analogue but uses a *single* time-constant — the project's contribution is the explicit T/P split. Mei et al. 2022 Box 1 cites the same Aston-Jones tonic-phasic decomposition. The predictions synthesis §5.5 calls out the T/P split as Gap 5 — no published forward-pass FiLM-style machine has it.

**Project-testbed measure.** Two new WandB scalars: `train/mod_h_P_corr_abs_delta` (phasic-vs-|δ| Pearson over a 50-episode window) and `train/mod_h_T_corr_cumcost` (tonic-vs-cumulative-aversive-signal Pearson over a 500-episode window). Cross-lag analysis: P at episode *t* vs T at episode *t+k* for k ∈ {1, 5, 20, 100}.

**Falsification signature.** Two failure modes refute. (i) P and T converge to the same trajectory (effective rank 1 across the two-state representation): the T/P split is architecturally present but functionally collapsed — the network found no use for two timescales, undermining the v2 claim that the unification *requires* two timescales. (ii) P tracks cumulative cost and T tracks |δ| (the inverted assignment): refutes the Aston-Jones-derived channel-attribution prediction and forces a re-derivation. This prediction is the cheapest test that the T/P split is doing work biology *and* the unification claim *both* require.

---

### P3 — Site C implements multiplicative slope rescaling (Ferguson & Cardin Box 1b); site B implements additive operating-point shift (Box 1c)

**Claim.** At injection site C (policy temperature), the FiLM γ should be the dominant arm — the cellular analogue is Ferguson & Cardin 2020 Box 1 panel b (multiplicative gain that rescales the slope of f(I) without moving the firing threshold). At site B (GRU update gate), the FiLM β should be the dominant arm — the cellular analogue is Box 1 panel c (additive/subtractive bias that shifts the input required to fire, leaving slope intact). At site A (encoder pre-fusion) both arms should be active, because perceptual gain-on-evidence is a known multiplicative + additive mixture (Shine 2021 Fig. 1).

**Biological grounding.** Ferguson & Cardin 2020 Box 1 (the canonical multiplicative-vs-additive cellular distinction, p. 81): multiplicative changes sensitivity, additive changes selectivity. Shine 2021 Box 1 + Fig. 1 (neural gain defined formally as dQ/dI, slope of the input-output curve). The mapping to the project's substrate: rescaling actor logits *is* a slope change on the softmax — multiplicative is the only operation that produces a temperature movement, so γ has to dominate at C. The GRU update gate z = σ(W_z · [h, x] + b_z + β) — additive β shifts the "stay vs update" decision boundary without changing how sensitively the gate responds to input — exactly the rheobase analogue. v2 prediction (b′) is essentially a γ-vs-β arm-dissociation test at site C; this prediction extends it to all three sites with biologically-grounded per-site expectations.

**Project-testbed measure.** Per-site freeze ablations across three conditions (γ=1, β=0, both free): C-site freeze drops should be γ-dominant; B-site freeze drops should be β-dominant; A-site freeze drops should be mixed. WandB readout: `eval/survival_steps_per_stage` × {freeze_C_gamma, freeze_C_beta, freeze_B_gamma, freeze_B_beta, freeze_A_gamma, freeze_A_beta, baseline} = 7 conditions. Statistic: per-site (γ-freeze-drop − β-freeze-drop), expected sign: positive at C, negative at B, near-zero at A.

**Falsification signature.** If C's γ-freeze and β-freeze are equally damaging, multiplicative-arm-as-Doya-NA-temperature is refuted at C and the unification at site C walks back to "joint gain control with no clean RL-hyperparameter interpretation". If B's γ-freeze is more damaging than β-freeze, the GRU's modulation is operating as multiplicative-on-weights (Tsuda-style, predictions synthesis §2 variant (c)) — interesting but unanchored to the Ferguson & Cardin rheobase reading. The headline contribution then shrinks to "we built a multi-arm gain controller, but the cellular-form interpretation is not clean per site".

---

### P4 — `mod_h` activity time-locks to |δ| at regime change, not to a one-hot regime label

**Claim.** When the agent crosses a stage boundary, the `mod_h` (or, post-T/P, `mod_h_P`) trajectory's burst should correlate with the *magnitude* of the value-prediction error at that boundary, not merely with the boundary's existence. Within a regime-change window, episodes with high |δ| should produce larger `mod_h` bursts than episodes with low |δ|; the within-window |δ|–`mod_h` correlation should be non-zero with the same sign across stage boundaries.

**Biological grounding.** Schultz / Montague / Dayan DA-as-RPE story (canonical TD-error signal in midbrain DA); Aston-Jones & Cohen 2005 (phasic LC bursts time-locked to salient/surprising events — surprise is operationalised as RPE magnitude); Yu & Dayan 2005 (NA as unexpected uncertainty, the second-moment of RPE distribution). The reasoning: a true Doya-channel signal — whether DA-flavoured (δ itself) or NA-flavoured (the surprise-magnitude derivative of δ) — must scale with prediction-error magnitude. A signal that fires identically on every regime change regardless of |δ| is a *context detector*, not a neuromodulator analogue. This is the v1 §8 prediction (d) that `professor-neuromodulation` staked the headline on; it is preserved here as a hard test of channel-attribution.

**Project-testbed measure.** Within-window correlation `corr(|δ|_t, mod_h_t)` computed over 50-episode windows centred on each stage boundary. WandB readout: `train/mod_h_corr_delta_window`. Three boundaries × N seeds gives a per-boundary distribution; the prediction is that the median correlation is significantly > 0 and the sign is consistent across boundaries.

**Falsification signature.** Null within-window correlation, *combined with* a strong correlation between `mod_h` and a one-hot regime label, refutes the Doya-channel reading. In combination with the P1 entropy-tracking test, the joint outcome "tracks regime label but not |δ| and not actor-entropy" demotes `mod_h` to a context-conditioning signal — Vecoven-2020 territory, no unification claim possible.

---

### P5 — Entanglement signature: γ-rank-collapse to 1 at site C means single-channel; high-rank means passenger-head architecture

**Claim.** If `mod_h` is genuinely a single Doya-NA-like channel acting at site C with two passenger heads at A and B (the rl-bayesian-dl warning), then the effective rank of the γ matrix at site C across regime changes should collapse to 1 — the network has discovered that only one scalar's worth of variation is needed to do the temperature work. Conversely, if the multi-site architecture has trained to use γ_C, γ_A, γ_B as *dissociable* channels (the optimistic multi-channel reading), the effective rank at C should be > 1 *and* the principal directions at A, B, C should be orthogonal (or at least linearly independent) across the per-feature γ vectors.

**Biological grounding.** Vecoven et al. 2020 Fig. 7 (the canonical "how many effective scalars does the learned modulator actually use" measurement — many z-dimensions go unrecruited on simple tasks). Costacurta et al. 2024 Fig. 3F (dissociation-by-ablation only emerges when the network *learned* to dedicate different latent dimensions to different sub-computations). Doya 2002 §3 (the four-system anatomical partitioning that the project's three injection sites are an analogue of). The reasoning: biology partitions across four nuclei because one tonic signal cannot simultaneously do learning-rate + temperature + discount + precision work. The project's single `mod_h` is the analogue of having one nucleus; the per-site readouts are the analogue of receptor-density-driven downstream effects. If the architecture is doing multi-channel work, the per-site γ statistics must carry the dissociation in their rank structure.

**Project-testbed measure.** SVD on the per-site γ vector collected across a regime-change window (≥ 1000 samples). Compute (i) effective rank of γ_C, γ_A, γ_B individually (participation ratio of the SVD spectrum); (ii) principal-angle between subspaces spanned by γ at different sites. WandB readout: `eval/gamma_C_effective_rank`, `eval/gamma_AB_principal_angle`. Tracked post-training on the final checkpoint per seed.

**Falsification signature.** If γ_C effective rank is > 5 (high-rank), the per-feature FiLM at C is doing something Doya doesn't describe — the unification claim at C is overparameterised and the project's contribution is not "we made a learned Doya-NA temperature modulator" but "we made an over-parameterised activation modulator that may or may not have a clean RL interpretation". If γ_C is rank 1 *but* γ_A and γ_B are also rank 1 *and* aligned to γ_C (small principal angles), the multi-site architecture is single-channel-with-passengers — the design is heterogeneous in disguise, and the architectural claim "multi-site coordination" should be walked back.

---

### P6 — Cross-site correlation structure dissociates single-channel-with-passengers from genuine multi-channel

**Claim.** Under a single-`mod_h` design (v2 architecture, pre-T/P), the per-site FiLM outputs are read out by learned linear maps from one recurrent state. If the network has converged on a passenger-head solution, the time series γ_C(t), γ_A(t), γ_B(t) should be perfectly cross-correlated at zero lag (they are all scalar multiples of the same `mod_h` projected through different readouts). If the network has somehow learned to use cross-site interactions to break the single-state degeneracy, the cross-correlations should be < 1 at zero lag and exhibit non-trivial lag structure. After the T/P split, the cross-correlation between sites should re-organise to reflect biology's partitioning: γ_C should covary with `mod_h_P` (phasic, NA-temperature timescale); γ_B should covary with `mod_h_T` (tonic, longer-memory-stability timescale); γ_A should be a mixture.

**Biological grounding.** Aston-Jones & Cohen 2005 (tonic vs phasic LC firing dissociates exploration/exploitation modes — different downstream targets respond on different timescales). Doya 2002 §3 (four anatomically distinct systems with independent timescales). Costacurta et al. 2024 Prop. 1 (LSTM-equivalence of low-rank weight scaling — different latent dimensions emerge to carry different temporal motifs). The predictions synthesis §5.3 (Gap 3 — multi-site dissociation of named hyperparameter channels has not been published). The reasoning: if the architecture's multi-site coordination is real, cross-site lag structure is the substrate-level fingerprint of it; if it is illusory, perfect zero-lag cross-correlation is the fingerprint of that.

**Project-testbed measure.** Pearson cross-correlation between γ_C(t), γ_A(t), γ_B(t) at lags ∈ {-5, -1, 0, 1, 5} episodes. WandB readout: `eval/gamma_C_A_xcorr_lag{−5,−1,0,1,5}`, etc. Pre-T/P prediction: all zero-lag cross-correlations are very high (> 0.9) — confirms single-channel-with-passengers. Post-T/P prediction: zero-lag cross-correlations drop substantially (< 0.7) and lag-structure emerges with γ_B trailing γ_C by several episodes.

**Falsification signature.** Post-T/P zero-lag cross-correlations remain > 0.9: the T/P split did not architecturally disentangle the channels — either the split was insufficient (deeper architectural changes needed) or the task does not demand multi-channel work (the unification is non-identifiable on this testbed and the conclusion should be limited).

---

### P7 — Site A's encoder gain is divisive-normalisation-shaped, not pure multiplicative — and this matters for the precision reading

**Claim.** At injection site A (encoder pre-fusion), if the encoder gain γ_A is to be read as an attentional / precision-on-evidence signal (the Yu-Dayan ACh-as-precision branch, v2 §3 explicitly chose the Doya-α branch but precision-on-features is still latent in the encoder), then γ_A should show divisive-normalisation behaviour: high values of γ_A on one feature should suppress the relative weighting of competing features in the downstream policy decision. This is distinct from pure multiplicative gain (which would scale all features simultaneously) and from pure additive shift (which would change selection thresholds without changing relative weighting).

**Biological grounding.** Ferguson & Cardin 2020 Box 1 panel b/d (multiplicative vs divisive normalisation — divisive normalisation is a canonical cortical computation, often produced by interneuron-mediated lateral inhibition under cholinergic control). Shine 2021 §"Cellular mechanisms" (gain modulation through divisive normalisation is the canonical V1 / attention computation). Yu & Dayan 2005 (ACh as expected uncertainty, with feature-specific precision-weighting as the cortical readout — out of corpus but cited by Lee 2024 §3.2). The reasoning: a precision-on-features signal that scales all features uniformly does not implement attention; attention requires that gain *on one feature* suppresses *relative* contribution of others. This is divisive normalisation, not pure multiplicative gain. Whether the project's FiLM at A spontaneously develops this property — without any architectural divisive-normalisation primitive — is a substantive empirical question.

**Project-testbed measure.** Logit-attribution analysis at site A: compute, for each input feature i and each action a, the partial-derivative of the policy logit on a w.r.t. feature i, under high-γ_A,i and low-γ_A,i conditions. If divisive-normalisation behaviour has emerged, high γ_A,i should *reduce* the |partial-derivative| on other features j ≠ i, not just increase it on i. WandB readout: `eval/gamma_A_divisive_index` defined as the median |∂π/∂x_j| at high γ_A,i divided by the same at low γ_A,i, averaged over j ≠ i. Index < 1 means divisive; index = 1 means pure multiplicative; index > 1 means anti-divisive (cross-feature facilitation).

**Falsification signature.** Index near 1 across the regime-change window: γ_A is pure-multiplicative, not divisive — site A's modulation is not implementing precision-on-features in the cortical-attention sense. The ACh-as-precision reading is unavailable at A, and any v3 §5 prediction that depends on it has to walk back to "site A gain is a generic feature scaler with no clean attention interpretation".

---

### P8 — Target specificity: the three injection sites should learn dissociable cell-class analogues, testable by per-class freeze

**Claim.** Ferguson & Cardin 2020 emphasises that real neuromodulators act with target specificity — different cell classes (excitatory vs inhibitory, distal vs proximal dendrites, projection-class-specific) receive different modulator effects. The project's three injection sites are an architectural analogue of three target classes: A (sensory encoder = early sensory cortex / thalamic input layer), B (memory GRU = hippocampal / association-cortex working-memory), C (policy logits = motor output / striatal action selection). If the analogy is meaningful, the three sites should produce *qualitatively dissociable* behavioural deficits when frozen: A-freeze should hurt under noisy / partially-observed states; B-freeze should hurt across regime memories (long-dormancy tests); C-freeze should hurt at regime-change moments specifically.

**Biological grounding.** Ferguson & Cardin 2020 §"Cellular mechanisms" (cell-class specificity of gain modulation: E vs I cells receive different effects; layer-specific cholinergic and noradrenergic innervation patterns). Doya 2002 §3.1 (the four-system mapping is *partly* derived from target-specificity: DA's striatal target gives it RPE function; ACh's basal-forebrain projection gives it precision-on-cortex function; etc.). v2 prediction (c) is the project's per-site clamp test; this prediction sharpens it by naming the *qualitative profile* of deficit expected at each site, not just the existence of dissociation.

**Project-testbed measure.** Three behavioural-readout subtests run under each per-site clamp condition: (i) high-noise / partial-observability subtest (expected to load on A-freeze); (ii) long-dormancy stage-return subtest with R2/R3 returns (expected to load on B-freeze; the project's empirical anchor lives here); (iii) regime-change recovery subtest measured by post-switch survival-step curve (expected to load on C-freeze). WandB readout: 3 × 3 deficit matrix `eval/per_site_freeze_per_subtest_drop`. Diagonal dominance (each site's freeze worst on its predicted subtest) confirms target specificity.

**Falsification signature.** No diagonal dominance — the deficit matrix is flat across sites and subtests, or worse, *all* freezes hurt subtest (iii) most (which is the "single-channel-NA-with-passengers" prediction). In that case the three injection sites are not implementing target-class-specific modulation; the architectural claim that the NMN is doing biology-like multi-site coordination is unsupported, and the contribution shrinks to a single-channel result.

## Methodological notes

Operational prerequisites for the predictions above:

1. **Phase 0 T/P split is required for P2, P5, P6, P8.** Without it, P1 reduces to a coarse channel-attribution check, and P3–P4 still run but cannot distinguish single-channel-with-passengers from multi-channel readings. The senior-developer should treat T/P as on the critical path for v3.

2. **Logging hooks needed.** New per-episode WandB scalars: `train/mod_h_mean`, `train/mod_h_P_mean`, `train/mod_h_T_mean`, `train/mod_h_corr_delta`, `train/mod_h_P_corr_abs_delta`, `train/mod_h_T_corr_cumcost`. New eval-time scalars: `eval/gamma_C_effective_rank`, `eval/gamma_AB_principal_angle`, `eval/gamma_C_A_xcorr_lag{k}`, `eval/gamma_A_divisive_index`, `eval/per_site_freeze_per_subtest_drop` (a 3×3 deficit matrix).

3. **Ablation harness.** A freeze-clamp utility that pins γ=1 or β=0 at one site at a time, callable at eval time with the trained checkpoint. v2 prediction (b′) already calls for the γ-freeze and β-freeze conditions at site C; P3 extends to A and B (6 conditions total); P8 needs the per-subtest split (×3 subtests = 21 evals per seed). Cost-bounded — the conditions reuse the same checkpoint.

4. **softplus constraint on γ.** The v1 §8 Q7 flag carries: if the headline word is "gain", γ should be constrained ≥ 0 (softplus). P3 and P5 assume this; without it, γ-rank analysis is contaminated by sign-flipping degeneracies. A senior-developer item.

5. **Behavioural subtests for P8.** The three subtests need to be reasonably orthogonal — see if the existing testbed library covers (i)-(iii); if not, experiment-designer should construct them. The long-dormancy R2/R3 return subtest is the project's existing empirical anchor and already exists.

## Cross-references to other professors' likely contributions

**Overlap with professor-rl-bayesian-dl.** P3 (the per-site γ vs β arm-dissociation) extends rl-bayesian-dl's v1 §8 prediction (b′) — they will likely re-stake on that test and probably add finer-grained predictions about the effective-rank-of-γ measurement (P5) and the gradient-vs-forward-pass differentiation from Rodriguez-Garcia 2026. Where we diverge: rl-bayesian-dl will probably propose more architectural variants (hypernet alternatives to FiLM, etc.); my predictions hold the architecture fixed at v2's spec and ask whether *that* architecture is doing biology-consistent work. The two views should compose cleanly — they ask "is this the right architecture", I ask "given this architecture, what biology is it instantiating".

**Overlap with professor-bayesian-brain.** P7 (divisive-normalisation at site A as a precision-on-features signature) is the load-bearing handoff. They will likely propose stronger predictions about precision-weighted prediction-error tracking and a Bayesian-decision-theory framing of the modulator's outputs. Where we diverge: I have anchored P7 in Ferguson & Cardin's cortical divisive-normalisation mechanism; they will probably anchor in active-inference / free-energy formulations. The two readings are not in conflict — divisive normalisation *is* precision in the cortical implementation — but the postdoc should reconcile the citations in v3.

**Overlap with professor-pain-modeling.** P2 (tonic state tracks cumulative cost) and P8 (target-specificity by subtest) are where their contributions will likely focus. Pain-relevant: a tonic NA elevation that does not relax after the aversive regime passes is the substrate-level signature of hypervigilance / chronic-pain analogues, and the project's testbed has a predator-avoidance / nociception story that loads on this. I expect pain-modeling to push prediction P2 toward an explicit hysteresis claim (T enters a high-tonic near-absorbing state after sustained aversive exposure); I am open to that and have intentionally left P2's tonic claim under-specified so they can sharpen it.

**Non-overlap (mine alone).** P5 (γ-rank-collapse as the entanglement signature), P6 (cross-site cross-correlation lag structure), and P8 (target-class-specificity profile across subtests) are substrate-side predictions that the other professors are less likely to propose, because they require explicit commitment to a Ferguson & Cardin cell-class-targeting reading and to the Aston-Jones tonic-phasic-timescale reading. These three should be the v3 §5 contributions where the biological-substrate professor (me) is the sole owner.

---

*Predictions by `professor-neuromodulation`, 2026-05-16.*
