# Directions v2 — NMN architecture vitality check via 2-D context-conditioned meta-RL

**Author**: professor-rl-bayesian-dl
**Date**: 2026-05-09
**Type**: Direction memo (architecture-driven, revision)
**Verdict**: **Run a 2×3 = 6-context cross-product mix** (predator behaviour ∈ {active, passive} × olfactory assignment ∈ {matched, distinct-canonical, swapped}), still **gated on the v1 ΔCKA measurement-only probe**, with the gating logic *generalised to multi-pair* and a new **per-cue clamp ablation** added. Hand-off named at the end.
**Supersedes**: This memo extends — does NOT supersede — [`nmn_meta_context_conditioning.md`](nmn_meta_context_conditioning.md). The v1 stays as a record. Only the §5 recommendation and §3 falsifying-measurement design are revised. The v1 P1/P2 derivations remain valid as written.
**Anchors**:
- v1 memo: `docs/project/directions/nmn_meta_context_conditioning.md` (P1, P2, three-way decision table).
- Triage: `docs/project/triage/20260509_1517_nmn_meta_continual_pivot.md` §4 (Candidate A), §5 (env-capability triage).
- Postdoc synthesis: `docs/project/ideas/nmn_meta_continual_synthesis.md` (cross-professor reconciliation; this memo will obsolete the synthesis's §5 menu and require a re-synthesis).
- The two existing sameProp configs: `configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml`, `configs/experiment/hypervigilance/02-sameProp_R2_passivePredator.yaml`.
- The non-sameProp default: `configs/experiment/hypervigilance/01-interoNocicept.yaml`.
- Verification harness: `configs/verification/olfaction_parity_predator.yaml`, `configs/verification/olfaction_parity_neutral.yaml`.
- Round 2 sameProp partial verdict: `.claude-memory/memories/hypervigilance/20260509_1532_sameprop_round2_truncated_verdict.md`.

---

## 1. What of v1 still holds, what does not

### 1.1 Still holds
- **P1 (context-distinguishability of $h^{\text{mod}}$)** as a *necessary* condition for any (b)-frame win. The derivation in v1 §2.1 is independent of how many contexts the mixture has; with $K$ contexts it generalises trivially to a $K \times K$ CKA Gram matrix whose off-diagonal blocks must be measurably smaller than the within-context diagonal.
- **P2 (alignment of branched-head injection direction with the encoder's context-discriminative subspace)** is unchanged in form. The 128-dim modality-uniform spatial-gate broadcast (review §2.4.1) still has to align with whichever encoder neurons carry the discriminative features — the question is just which features carry the discrimination per context-pair.
- **The three-way decision table (CKA × clamp × survival).** Survival alone is still uninterpretable; the clamp ablation is still the load-bearing causal arm; (CKA-pos, clamp-pos, survival-null) is still *redundancy* not failure, and still motivates Phase 0.
- **The $\Delta_{\text{CKA}}$ on existing v8 trajectories is the cheapest gate available.** The measurement-only probe in v1 §6 still runs first, even though now it must produce a multi-context Gram matrix rather than a single scalar.

### 1.2 Needs revising
- **The v1 cleanness argument** ("A1's discriminator is *only* movement signature → if the modulator does anything, it should show up here"). This argument was *load-bearing for the cell selection in v1 §5*, and **the user's question breaks it**: once olfactory assignment varies across contexts, contexts become discriminable by olfactory alone, and the agent could solve the mix without ever using the recurrent movement signature. §3 of this memo restores cleanness through *per-cue clamp ablations*, not through cue restriction.
- **The v1 single-pair clamp ablation** ($\Delta_{\text{clamp}} = \mathrm{Survival}_B^{\text{free}} - \mathrm{Survival}_B^{\text{clamp-to-A}}$). With $K \geq 3$ contexts this becomes a $K \times K$ matrix of pairwise clamp drops; the analyzer needs the off-diagonal mass, not a single scalar. §4 specifies.
- **The v1 "12 cells" budget**. A 2×3 cross-product with 2 seeds + 2 unmod controls + 1 ΔCKA-only baseline lands at 8 modulated cells × 2 seeds + 4 unmod control cells × 2 seeds = ~24 cell-slots. §5 reconciles with the project's ≤ 10–12 cell ceiling by tiering into rounds.

### 1.3 The single most-load-bearing additional claim of v2
**The 2-D cross-product (behaviour mode × olfactory assignment) is not a richer test of the *same* property — it is a test of a *strictly stronger* property**: that the modulator builds a context representation that is **invariant to which cue carries the context information**. P1 says the modulator hidden state is context-distinguishable; v2's stronger claim, call it **P3**, says the modulator hidden state factorises:

$$
h^{\text{mod}}_t \approx \phi_{\text{mode}}(s_t) \;\oplus\; \phi_{\text{prop}}(s_t) \;\oplus\; \phi_{\text{shared}}(s_t),
$$

with two distinct subspaces for the two manipulation axes (or, equivalently, the across-context CKA structure is **block-rank-2**, not block-rank-1). P3 is a *capacity* claim about the modulator's 16-dim GRU; P1 is only an *existence* claim. NPN (Ben-Iwhiwhu 2022a, review §B.6) demonstrates P1; P3 is, to my knowledge, **not directly demonstrated in any of our reference papers**, which puts a v2-positive into novel-finding territory but also raises the prior probability of a v2-null.

## 2. Re-derivation of the meta-task design

### 2.1 The 2-D manipulation space

Let the env expose a behaviour mode $m \in \mathcal{M} = \{\text{active}, \text{passive}\}$ and an olfactory assignment $p \in \mathcal{P}$. The user's enumeration of $\mathcal{P}$:

- $p_{\text{matched}} = (\text{predator}=[0,1,0,0,0],\; \text{rabbit}=[0,1,0,0,0])$ — sameProp regime.
- $p_{\text{distinct}} = (\text{predator}=[0,1,0,0,0],\; \text{rabbit}=[0,0,1,0,0])$ — non-sameProp default (close to `01-interoNocicept.yaml`'s setup, modulo `properties_std`; see §2.4 below).
- $p_{\text{swapped}} = (\text{predator}=[0,0,1,0,0],\; \text{rabbit}=[0,1,0,0,0])$ — the user's "rabbit smells like predator" trick.
- (optional) $p_{\text{distinct-2}}, p_{\text{distinct-3}}, \ldots$ — N additional distinct-canonical assignments rotating the canonical basis vectors.

A context is the cross-product $c = (m, p) \in \mathcal{M} \times \mathcal{P}$. The full mixture is the empirical distribution over $\mathcal{C} = \mathcal{M} \times \mathcal{P}$ at episode reset.

### 2.2 Why $|\mathcal{P}| \geq 3$ is strictly more diagnostic than $|\mathcal{P}| = 2$

A 2×2 mixture (active/passive × matched/swapped) can be solved by the modulator memorising *one* binary context-discriminative subspace plus the (m, p) parity bit — i.e., the modulator only has to build a 2-bit context code, and the 16-dim modulator GRU has more than enough capacity for that without committing to any factorisation. A 2×2-positive does **not** demonstrate P3.

A 2×3 mixture forces a richer encoding: with 6 contexts, the modulator must distinguish $\binom{6}{2} = 15$ pairs, and the structure of which pairs are confusable directly diagnoses *which axis* it is encoding. Concretely, if the modulator factorises into mode and prop subspaces (P3), the off-diagonal CKA gaps between (active, $p_i$) and (active, $p_j$) should match those between (passive, $p_i$) and (passive, $p_j$) — i.e., the prop axis is encoded the same way regardless of mode. If the modulator does not factorise, the gaps are heterogeneous and P3 fails.

**Recommended $|\mathcal{P}|$**: **$|\mathcal{P}| = 3$** ({matched, distinct-canonical, swapped}), giving a 2×3 = 6-context mix. Beyond 3 the diagnostic returns diminish (the within-axis CKA structure is already estimable from 3 levels); the marginal cost is not worth it for an "easy first" experiment.

### 2.3 Olfactory swap as a strictly stronger test than distinct-canonical

The user is right that swap probes a *different* mechanism than distinct-canonical:

- **Distinct-canonical** ($p_{\text{distinct}}$) — the cues are non-misleading. The agent can in principle use olfactory alone; movement signature is redundant. A successful agent on distinct-canonical demonstrates only that olfactory features are linearly separable, which the encoder achieves trivially.
- **Swapped** ($p_{\text{swapped}}$) — the cues are *anti-correlated* with their distinct-canonical signature. An agent that learned (during distinct-canonical training) "olfactory channel 1 = predator" will, in swapped, treat the rabbit as if it were a predator. Recovering correct behaviour on swap requires either (a) re-learning the olfactory→identity mapping per context, or (b) using a non-olfactory cue (movement signature) to override the misleading olfactory signal. Mechanism (b) is exactly the (b)-frame meta-RL we want to test.

Swap is a strict-superset test of distinct-canonical: positive on swap $\Rightarrow$ positive on distinct-canonical *modulo* the agent having seen distinct-canonical first; null on swap with positive on distinct-canonical $\Rightarrow$ the agent solved distinct-canonical via olfactory shortcut, not via context inference.

### 2.4 An env-side caveat the user should hear before locking the design

The non-sameProp `01-interoNocicept.yaml` does **not** use clean canonical basis vectors — its rabbit is `[0, 0.5, 0.7, 0, 0]` with `properties_std=[0, 0.4, 0.4, 0, 0]` (lines 111–112) and predator is `[0, 0.7, 0.5, 0, 0]` with the same std (lines 132–133). This is a *partially-overlapping, stochastic* olfactory pair, not a clean distinct-canonical assignment. For the 2×3 design here we need a **new YAML** (~1 hour of env-config work, no code changes) that uses:

- $p_{\text{matched}}$: pred=`[0,1,0,0,0]`, rab=`[0,1,0,0,0]`, std=zero (already in `01-interoNocicept_sameProp.yaml`).
- $p_{\text{distinct}}$: pred=`[0,1,0,0,0]`, rab=`[0,0,1,0,0]`, std=zero.
- $p_{\text{swapped}}$: pred=`[0,0,1,0,0]`, rab=`[0,1,0,0,0]`, std=zero.

Std=zero matters: with stochastic properties, the olfactory channel itself becomes noisy and the swap manipulation is partially confounded with within-context noise. Hand off to `experiment-designer`.

## 3. Restoring diagnostic cleanness: per-cue clamp ablations

The v1 cleanness argument was "the only available cue is movement, so any context behaviour must use the modulator." With 2-D variation, both cues are available in some contexts, so the modulator can be bypassed via olfactory shortcut. We restore cleanness not by restricting cues but by **measuring per-cue causal contribution at eval**, on the *same* trained agent:

For a trained 2×3-mixture agent, run four eval modes per context $c = (m, p)$:

1. **Free.** All cues intact. $\mathrm{Survival}^{\text{free}}_c$.
2. **Olfactory-clamped.** Replace the 5-dim olfactory observation with its sample-average across all training contexts (i.e., zero out the cue while preserving the obs distribution). $\mathrm{Survival}^{\text{olf-clamp}}_c$. Tests: how much survival depends on olfactory.
3. **Movement-clamped (modulator-state-clamped, v1 §3.2).** Freeze $h^{\text{mod}}$ to its sample-average across all training contexts. $\mathrm{Survival}^{\text{mod-clamp}}_c$. Tests: how much survival depends on the modulator's recurrent inference (which, on swapped contexts, is the only way to override the misleading olfactory signal).
4. **Both-clamped.** Both the above. $\mathrm{Survival}^{\text{both-clamp}}_c$ is the cue-free floor.

Define per-context cue-attribution scores:

$$
\rho^{\text{olf}}_c \;\triangleq\; \mathrm{Survival}^{\text{free}}_c - \mathrm{Survival}^{\text{olf-clamp}}_c, \qquad
\rho^{\text{mod}}_c \;\triangleq\; \mathrm{Survival}^{\text{free}}_c - \mathrm{Survival}^{\text{mod-clamp}}_c.
$$

The v2 falsifying signature is then **structural**, not scalar: the modulator is doing genuine context-conditioning iff $\rho^{\text{mod}}_c$ is **largest on swapped contexts** (where olfactory misleads and the modulator is the only override) and *smaller* on matched contexts (where olfactory is uninformative and movement is the only cue) and *smallest* on distinct-canonical contexts (where olfactory is sufficient).

Predicted $\rho^{\text{mod}}_c$ ranking under P3:

$$
\rho^{\text{mod}}_{(m,\,\text{swapped})} \;>\; \rho^{\text{mod}}_{(m,\,\text{matched})} \;\gtrsim\; \rho^{\text{mod}}_{(m,\,\text{distinct})} \quad\forall m.
$$

Predicted $\rho^{\text{olf}}_c$ ranking under P3:

$$
\rho^{\text{olf}}_{(m,\,\text{distinct})} \;>\; \rho^{\text{olf}}_{(m,\,\text{swapped})} \;\gtrsim\; \rho^{\text{olf}}_{(m,\,\text{matched})} \quad\forall m,
$$

with the extra prediction that on *swapped* contexts $\rho^{\text{olf}}_{(m,\,\text{swapped})}$ should be **negative** if the modulator successfully overrides the misleading cue (clamping olfactory to its training-mean *helps* survival on swap).

This is the v2's payoff: a *signed* prediction on the per-cue ablation matrix that distinguishes "modulator is doing context-conditioning" from "modulator is using olfactory shortcut" — exactly the ambiguity the user's question creates.

## 4. The falsifying measurement, scaled to multi-context

### 4.1 Measurement-only ΔCKA gate (preserved from v1, scaled)

Replace v1 §3.1's scalar $\Delta_{\text{CKA}}$ with a $6 \times 6$ Gram matrix on existing v8 modulator hidden states, evaluated at **matched timesteps in matched grid states across all six (m, p) contexts**:

$$
G_{ij} \;\triangleq\; \mathrm{CKA}\!\left(H^{\text{mod}}_{c_i},\; H^{\text{mod}}_{c_j}\right), \qquad c_i, c_j \in \mathcal{M} \times \mathcal{P}.
$$

Two scalar summary statistics:

$$
\bar\Delta_{\text{CKA}} \;\triangleq\; \mathbb{E}_i\!\left[G_{ii}\right] - \mathbb{E}_{i \neq j}\!\left[G_{ij}\right], \qquad
\bar\Delta_{\text{CKA}}^{\text{factor}} \;\triangleq\; \mathbb{E}\!\left[G_{(m_1, p),\,(m_2, p)}\right] - \mathbb{E}\!\left[G_{(m, p_1),\,(m, p_2)}\right].
$$

The first is the v1 P1 statistic generalised. The second tests **factorisation (P3 above)**: if the modulator encodes mode and prop in disjoint subspaces, off-diagonal CKA between contexts that share $p$ but differ on $m$ should be roughly equal to off-diagonal CKA between contexts that share $m$ but differ on $p$ (i.e., $\bar\Delta_{\text{CKA}}^{\text{factor}} \approx 0$). Strong asymmetry signals *which axis* the modulator is encoding preferentially.

**Falsification thresholds** (still soft-eyeball; report distributions over seeds):
- $\bar\Delta_{\text{CKA}} < 0.05$: P1 fails. Predict null on the experiment. Escalate to Phase 0.
- $\bar\Delta_{\text{CKA}} \geq 0.05$ but $|\bar\Delta_{\text{CKA}}^{\text{factor}}| > 0.15$: P1 holds but P3 fails. The modulator builds a context code but does not factorise. Run the experiment but predict *partial* clamp-ablation cleanness.

**Caveat**: v8 modulators were trained on a *single context* (or on noise-heterogeneity variants of one), so $\bar\Delta_{\text{CKA}}$ on v8 logs measures whether modulators built distinguishable state from *training-distribution variation*, not from the proposed 2×3 mix. This is what the v1 already assumed; the v2 generalises but inherits the same caveat. If $\bar\Delta_{\text{CKA}}$ fails on v8, it fails *a fortiori* on a richer task; if it passes, the experiment is greenlit but a positive result on the 2×3 mix is still required to demonstrate P3.

### 4.2 Per-context survival × clamp matrix (new)

After training, for each seed and each (m, p) ∈ $\mathcal{C}$, log the four-tuple $(\mathrm{Survival}^{\text{free}}_c, \mathrm{Survival}^{\text{olf-clamp}}_c, \mathrm{Survival}^{\text{mod-clamp}}_c, \mathrm{Survival}^{\text{both-clamp}}_c)$. The analyzer then computes the $\rho$ matrices in §3 and tests the predicted rankings. The full read-out is a $6 \times 4$ table (6 contexts × 4 eval modes) per seed.

## 5. Concrete recommended experiment

### 5.1 Pre-registered design

| Property | Setting |
|---|---|
| Behaviour modes | $\mathcal{M} = \{\text{active}, \text{passive}\}$. Active = `01-interoNocicept_sameProp.yaml` predator settings; passive = `02-sameProp_R2_passivePredator.yaml`'s `hunt_stamina_threshold: 1.1`, `detection_range: 0`. |
| Olfactory assignments | $\mathcal{P} = \{\text{matched}, \text{distinct}, \text{swapped}\}$, all with `properties_std=[0,0,0,0,0]`. New YAMLs needed (~1 hr config-only work). |
| Mixture sampling | **Uniform** over the 6 (m, p) contexts at episode reset. *Not* curriculum: curriculum confounds context-conditioning with progressive specialisation. *Not* episode-stratified: stratification gives the agent a deterministic context-detection signal from episode index, which would let it bypass the modulator. |
| Cell count (modulated) | 6 contexts × 1 architecture (current branched FiLM modulator) × **2 seeds** = **2 cells** (each cell trains one modulator on the full 6-context mix). |
| Cell count (unmodulated control) | Same 6-context mix × unmodulated GRU baseline × **2 seeds** = **2 cells**. |
| Cell count (per-context specialists, ceiling reference) | 6 contexts × unmodulated × **1 seed each** = **6 cells**. |
| Cell count (ΔCKA-only baseline) | None — runs on existing v8 logs. |
| **Total** | **10 cells**, fits within the project's ≤ 10–12 ceiling for "easy first" experiments. |
| Cost per cell | Match v1 §7 estimate of ~13h/cell for joint-mixture training, plus ~1h eval pass per cell for the 4-mode clamp ablation. Total ~140 cell-hours, ≈ 2 nights on 4–6 nodes (matches v1's 1-night A1 estimate × 2 because the mix is 6× richer). |

### 5.2 If the user wants a tier-2 ablation

Add a third behaviour mode (e.g., `move_interval=2` predator → "slow-active") to make the mix 3×3 = 9 contexts; or add a fourth olfactory assignment ($p_{\text{distinct-2}}$ rotating to a different basis vector). Each adds ~3 modulated + 3 unmod-control cells for ~6 extra cells. **Defer to round 2**; not part of the launch design.

### 5.3 Why this fits within the project's "easy first" ceiling

10 cells × ~14h ≈ 140 cell-hours is comparable to the v1 12-cell estimate (12 × 13 = 156 cell-hours). The richer task does not blow the budget *if* the modulated arm uses 2 seeds (v1 used 3); the cost saving comes from spending the third seed only on the load-bearing arm in round 2 if round 1 lands borderline. Round 2 sameProp's ±4–5 step seed noise (insight `20260509_1532_sameprop_round2_truncated_verdict`) suggests 2 seeds is *thin* — the analyzer must report seed dispersion before drawing v2 conclusions.

## 6. Phase 0 implications — does the 2-D task change the answer?

**Yes, it tightens the case for the precision head and weakly raises the case for the T/P split.** Three reasons:

1. **The 16-dim modulator GRU is now likely under-capacitised.** P3 demands the modulator factorise across two axes. Empirically (Ben-Iwhiwhu 2022a, review §B.6), NPN's intra-layer modulators have one set of parameters per layer, not a single shared 16-dim recurrent state. A single 16-dim GRU encoding two 1-bit-plus-shape factors is feasible but tight. If $\bar\Delta_{\text{CKA}}^{\text{factor}}$ in §4.1 fails on v8 logs, that argues for either a wider modulator (≥ 32 dim) or the **T/P split** (project_plan §3.2), since splitting "tonic" from "phasic" gives the modulator two recurrent streams whose timescale-separated factorisation could naturally absorb the (m, p) factorisation.
2. **The aleatoric teaching signal becomes more important on swap.** Under olfactory swap, olfactory features actively mislead the policy, and the agent benefits from down-weighting olfactory in the precision-weighted prediction error. The current architecture has no aleatoric teaching signal (postdoc synthesis §1, prof-Neuromod identification of Lee 2024 substrate gap); adding the **precision head** (project_plan §3.4) gives the modulator a target for "this cue is unreliable here, gate it down" via FiLM γ at site A. On a swap-heavy mix, this is the cleanest reading of what the precision head buys.
3. **The opioid head remains out of scope** — it is for the chronic-pain analog (H5) and the four-property fingerprint, not for context-conditioning. Don't pull it in.

**Net effect on the v1 verdict ("run, don't block")**: I still say **run**, because the measurement-only gate is free and the data from the 2×3 mix is informative regardless of which Phase 0 lever ends up binding. But if the v2 mix nulls *and* $\bar\Delta_{\text{CKA}}^{\text{factor}}$ fails, the case for blocking on Phase 0 strengthens noticeably from where v1 left it.

## 7. The Round 2 sameProp finding enriches but does not complicate v2

The Round 2 sameProp partial verdict (insight `20260509_1532_sameprop_round2_truncated_verdict`) reports MeanDistRabbit ≈ 3.84, MeanDistPredator ≈ 7.70 on Cell A1 (passive predator, matched olfactory) — i.e., the agent under matched olfactory **does not avoid the rabbit at all**, and operationally lives in the BR rabbit-and-food corner for 482/500 steps. This is exactly the §5 row-2 failure mode (quadrant-camping) that makes Δ uninterpretable without per-tag distance.

For v2 this is **enriching, not complicating**, in three ways:

1. **It calibrates the prior on $p_{\text{matched}}$ context-conditioning.** If the agent under matched olfactory simply parks itself in a safe corner regardless of which animal is which, then on the 2×3 mix the matched contexts will look *easy* by survival but *empty* by clamp ablation — $\rho^{\text{mod}}$ on matched contexts will be small not because the modulator failed but because the corner-camping policy doesn't need the modulator. The analyzer must compute per-quadrant occupancy and report whether matched-context survival is from corner-camping or from active discrimination.
2. **It strengthens the case for swap as the load-bearing context.** Round 2 showed that under matched olfactory the agent *could* learn to discriminate but didn't have to — corner-camping was sufficient. Under swap, corner-camping does not solve the task: if the rabbit smells like a predator, the agent that camps with the rabbit is now camping with the apparent-predator and will either flee (and lose food access) or freeze (and die from real predator exposure on other episodes). Swap forces the agent to either ignore olfactory or correctly invert it; both routes reveal the modulator's contribution.
3. **It tells us 2 seeds is the floor, not the target.** Round 1 had ±0.03 cells across seeds 42/43; Round 2 had a sign flip on Cell C between R1 and R2 (which the insight notes is plausibly an early-learning artifact, not a true sign flip). The analyzer must check seed convergence on the v2 mix before drawing conclusions; if 2 seeds disagree by more than ~5 survival steps, escalate to a third seed.

## 8. Hand-off

### 8.1 Named next steps

1. **`experiment-analyzer`** — first, *unchanged from v1 §6* in spirit but generalised. Run the 6-context $\bar\Delta_{\text{CKA}}$ + $\bar\Delta_{\text{CKA}}^{\text{factor}}$ gate on existing v8 logged trajectories. Output at `docs/experiments/active/nmn_meta_pivot/measurement_only_probe_v2.md`. If `mod_h` is not logged on v8, re-evaluate from checkpoints (~1 cell-hour).
2. **`experiment-designer`** — *gated on (1)*. If $\bar\Delta_{\text{CKA}} \geq 0.05$, lock the 10-cell grid in §5.1, generate the three new YAMLs ($p_{\text{matched}}$ exists; $p_{\text{distinct}}$, $p_{\text{swapped}}$ are new ~1-hour configs), and the design doc at `docs/experiments/active/nmn_meta_pivot/v2_design.md`. The Launch Manifest must lock all 6 (m, p) tag/wandb names at design time per `feedback_launch_manifest`.
3. **`developer`** — independent of (1)'s outcome. The two flags from v1 §8 (`--mixture-mode` on `train.py`; `--mod-clamp` on `evaluate.py`) are still needed. v2 adds a third: **`--obs-clamp <modality-name>`** on `evaluate.py` for the per-cue olfactory clamp in §3. Three flags total, all contained, ~1.5 days. No invasive changes to encoder/GRU/actor/critic.
4. **`senior-developer`** — *gated on (1) negative*. Phase 0 plan as in v1 §8, with the additional v2 case (§6 above) for the precision head being the load-bearing fix.
5. **`research-postdoc`** — re-synthesise. The synthesis memo `nmn_meta_continual_synthesis.md` §5 lock-in menu was written under v1's design; v2 changes Option B/C cell counts and adds the per-cue ablation. Synthesis re-write is needed before the user is asked to choose.

### 8.2 What this memo does not do

- Does not propose env code changes. The 2×3 design is fully config-only modulo the three new YAMLs.
- Does not propose a new architecture. The modulator and FiLM heads are unchanged. Phase 0 levers remain as-named in project_plan; v2 only changes the *prior* on which lever binds.
- Does not specify mixture-sampling implementation details (e.g., per-episode RNG keying). That is `developer`'s call once `--mixture-mode` is implemented.

## 9. Risk register (v2-specific additions)

- **Risk A — The 6-context mix is too hard for any architecture.** A null on both modulated and unmodulated arms tells us the env is over-engineered; no architecture-vitality conclusion is drawable. *Mitigation*: the 6 per-context specialists (§5.1, 6 unmod cells × 1 seed) are the ceiling reference. If the specialists themselves don't separate, the contexts are not actually task-conflicting and the 2×3 design is degenerate. Pre-flight: 1 specialist per context for 0.5M episodes before launching the joint grid.
- **Risk B — Swap is *too* hard.** The swap context might be effectively-unsolvable (the agent has to use only movement under misleading olfactory), pushing all the survival mass onto matched + distinct contexts and degenerating to a 2×2 problem. *Mitigation*: report per-context survival distribution; if swap survival floors at ~30 steps regardless of architecture, swap is a black-hole context and the analyzer should drop it and re-interpret as a 2×2 matched/distinct mix. This would *weaken* the v2 claim back toward the v1 claim, but cleanly.
- **Risk C — The $p_{\text{distinct}}$ assignment leaks information about $p_{\text{swapped}}$.** If the agent trains on all three concurrently, it might learn a "context detector" that predicts $p$ from olfactory alone (since matched, distinct, swapped have distinguishable olfactory marginals), then uses $p$ to gate its policy. This is *exactly the (b)-frame meta-RL win we want* — but it's solved without the modulator if the encoder + task GRU can hold the olfactory mapping. *Mitigation*: the per-cue clamp ablation in §3 catches this; if $\rho^{\text{mod}}$ is small across all contexts, the win is in the encoder + task GRU (the v1 §4 redundancy worry, generalised), and the conclusion is the same as v1 — Phase 0 first.
- **Risk D — `properties_std` differences across the three new YAMLs leak context.** If the std vectors differ across (matched, distinct, swapped), the agent could classify $p$ from observation noise alone. *Mitigation*: enforce std=zero across all three (§2.4); flag to `env-config-auditor` for the pre-flight check.
- **Risks 1–5 from v1 §9 still apply** with the obvious generalisation from 2 contexts to 6.

---

**Document end.** The single most-load-bearing claim of this memo is in §1.3 (P3, factorisation across the 2-D manipulation space) and §3 (per-cue clamp matrix); together they convert the user's "perturb the olfactory properties as well" intuition into a strictly stronger *and* more diagnostic experiment than v1, without exceeding the project's cell-count ceiling.
