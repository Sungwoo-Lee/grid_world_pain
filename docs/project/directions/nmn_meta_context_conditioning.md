# Directions — NMN architecture vitality check via context-conditioned meta-RL

**Author**: professor-rl-bayesian-dl
**Date**: 2026-05-09
**Type**: Direction memo (architecture-driven)
**Verdict**: **Run Candidate A1** (active vs. passive predator mix), **gated on a one-week measurement-only probe** on existing v8 runs first. Hand-off named at the end.
**Anchors**:
- Triage memo `docs/project/triage/20260509_1517_nmn_meta_continual_pivot.md` §4 (Candidate A).
- `docs/develop/active/neuromodulation/NMN_ARCHITECTURE_REVIEW.md` §2.1–2.4, §3.1–3.3 (architecture as built).
- `docs/develop/active/neuromodulation/NEUROMODULATION_ALGORITHM.md` §B.1–B.6 (NPN; the canonical (b)-frame result), §K (Tsuda hypertube; mechanistic ground for (b)), §1 (architectural-bottleneck argument), Hypotheses H1–H5.
- v8 null `docs/develop/active/diagnosis/NMN_PERFORMANCE_DIAGNOSIS_v8.md` (sensory-modulation null this memo is a vitality check for).

---

## 1. Question and target venue

**Question (falsifiable):** *On a two-context mixture in our env, where contexts share the obs space but require opposite policies, does the project's branched recurrent FiLM modulator deliver an NPN-style gain over an unmodulated GRU baseline in joint-mixture survival, and is that gain causally attributable to context-conditioning of the modulator's recurrent state?*

**Why this question now.** The v8 null (§4 of `project_plan.md`, four causes) does not yet distinguish "this modulator can never produce a benefit" from "this modulator can produce the benefit the literature predicts but not the harder benefit project_plan asks for." Frame (b) (NPN-style; review §B.6: ML45 ~2× SPN success) is the easiest published win for an architecture of our form; failing it would localise the v8 null to **causes 1 and 4** (architecture and identifiability), not to **cause 2** (sensory-modulation as a hard problem).

**Venues.** A positive (b)-frame replication on a survival-RL env is a workshop-paper-grade finding (NeurIPS Intrinsic-Motivation / Goal-RL / SSL-RL workshops; CCN proceedings). A negative result with a clean falsifying measurement is a blog/tech-report finding internally; only worth a venue if combined with a sensory-modulation positive later.

## 2. Theoretical contribution — the mechanistic property the NMN must have

### 2.1 Sketch derivation

Let two contexts $c \in \{A, B\}$ share an obs space $\mathcal{X}$ but have task-conflicting optimal policies $\pi^*_A, \pi^*_B$. Denote the modulator's GRU hidden state by $h^{\text{mod}}_t \in \mathbb{R}^{16}$ (per `NMN_ARCHITECTURE_REVIEW.md` §2.1) and the task GRU's hidden state by $h^{\text{task}}_t \in \mathbb{R}^{128}$. The branched heads emit, for each step $t$, a vector

$$
\mathbf{z}_t \;=\; \big( z^{\gamma_1}_t,\, z^{\beta_1}_t,\, z^{\gamma_2}_t,\, z^{\beta_2}_t,\, z^{\text{mem}}_t,\, \tau_t \big), \qquad \mathbf{z}_t = \Phi(h^{\text{mod}}_t;\,\theta_{\text{heads}}, \theta_{\text{baseline}})
$$

where $\Phi$ is the repeat-and-slice + per-neuron-baseline map (review §2.3–2.4), and the heads inject at three sites — A perceptual FiLM, B GRU update-gate bias, C policy temperature (review §3).

Define the **context-conditional injection signature**

$$
\mathbf{Z}^{(c)}(s) \;\triangleq\; \mathbb{E}_{\pi}\!\left[\, \mathbf{z}_t \,\middle|\, s_t = s,\, c\, \right]
$$

i.e. the average modulation pattern the agent emits in state $s$ when in context $c$.

**Property P1 (necessary)** — *Context distinguishability of the modulator state.*  There must exist a measurable $\delta > 0$ such that

$$
\mathrm{CKA}\!\left(H^{\text{mod}}_A,\, H^{\text{mod}}_B\right) \;<\; \mathrm{CKA}\!\left(H^{\text{mod}}_A,\, H^{\text{mod}}_A\right) - \delta,
$$

with $H^{\text{mod}}_c$ a sample of modulator hidden states drawn under context $c$ at matched timesteps. Without P1, the modulator emits the same $\mathbf{z}$ in both contexts — the branched heads are deterministic functions of $h^{\text{mod}}$ — and the architecture cannot context-condition. (This is the project-side analogue of Ben-Iwhiwhu et al.'s CKA argument: review §B.6, *"NPNs generate dissimilar representations for different tasks"*.)

**Property P2 (necessary, additional, branched-head-specific)** — *Behavioural-relevance of the modulation difference.* P1 is **not sufficient**. The branched-head structure (review §2.4) imposes that the difference $\mathbf{Z}^{(A)}(s) - \mathbf{Z}^{(B)}(s)$ live in a subspace of $\mathbf{z}$ that the **task pathway is gradient-sensitive to in a context-conflicting direction** at $s$. Concretely, write the task-loss gradient w.r.t. the FiLM gain at site A as $\partial \mathcal{L} / \partial \gamma_t$. Then the per-context injection signatures must satisfy

$$
\Big\langle \mathbf{Z}^{(A)}(s) - \mathbf{Z}^{(B)}(s),\;\; g^{(A)}(s) - g^{(B)}(s) \Big\rangle \;>\; 0,
$$

where $g^{(c)}(s) = \mathbb{E}_{\pi^*_c}[\partial \mathcal{L}/\partial \mathbf{z}\,|\,s,c]$ is the optimal context-conditional injection-gradient. In words: it is not enough that the modulator's hidden state differ; the branched heads must **route that difference into modulation directions that resolve the policy conflict**.

This is a stronger condition than NPN's, because NPN's modulator is **intra-layer** (review §B.4): every layer's modulator sees the same input as that layer's standard neurons, so context inferred from $x$ trivially aligns with the gradient at that layer. In our architecture the modulator is **separate**, and `head_unimodal` projects through (a) repeat-and-slice grouping (G=4 → 32 groups broadcast to 128 dims) and (b) **the same spatial gate pattern broadcast over all 9 modalities** (review §2.4.1, "all modalities receive the same spatial gate pattern"). For P2 to hold, the **128-dim spatial subspace** the modulator gates must align with the encoder neurons whose features carry the active/passive-predator discrimination. There is no a-priori reason that the encoder learns to put context-discriminative features in a *modality-uniform* spatial pattern; it could just as easily put them in modality-specific subspaces (e.g., olfactory neurons 0–15 carry one signature, visual neurons 32–63 carry the other). **If the encoder does not so align, the branched heads can satisfy P1 yet fail P2 — the modulator distinguishes contexts in $h^{\text{mod}}$ but cannot project the distinction onto a useful gating pattern.**

### 2.2 Why this differs from the closest precedent (NPN; review §B.1–B.6)

NPN (Ben-Iwhiwhu 2022a) is feedforward, intra-layer, no spatial grouping (review §B.4). Our architecture is recurrent, separate, with grouped + per-modality-broadcast gates. The closest precedent for the *separate-recurrent-modulator* form is **Vecoven 2020** (review §A) — but Vecoven modulates activation-function parameters (slope + bias of sReLU) globally rather than via FiLM gain, and Vecoven does **not** do the branched-head decomposition. So the closest published architecture to ours is NPN-with-an-RNN — *which has not been published with the branched + grouped + temperature form we have*. P2 is the price of that combination.

## 3. Falsifying measurement

### 3.1 Primary falsifier (cheap; runs on existing v8 logged trajectories)

For an existing trained modulator, sample $N \approx 500$ states $s$ that occur in **both** sub-environments (A1 and A2 of triage §4 — predator-mode mix). Log $h^{\text{mod}}_t(s,c)$ for $c \in \{A, B\}$ and compute

$$
\Delta_{\text{CKA}} \;\triangleq\; \mathrm{CKA}\!\left(H^{\text{mod}}_A, H^{\text{mod}}_A\right) \;-\; \mathrm{CKA}\!\left(H^{\text{mod}}_A, H^{\text{mod}}_B\right).
$$

**Falsification threshold**: $\Delta_{\text{CKA}} < 0.05$ on a trained modulator that has *seen both contexts in training* rules out P1 and falsifies the (b)-frame claim regardless of survival outcome. The 0.05 threshold is a soft-eyeball value taken from the NPN paper's CKA gap reading (review §B.6) — it is *not* a calibrated test statistic; treat it as the sign of the effect, not its $p$-value.

### 3.2 Secondary falsifier (requires the experiment to actually run)

After training Candidate A1, run the **modulator-state-clamp ablation** (triage §4 DV (ii)): freeze $h^{\text{mod}}$ to its sample-average across context A and re-evaluate on context-B episodes. Define

$$
\Delta_{\text{clamp}} \;\triangleq\; \mathrm{Survival}_B^{\,\text{free}} - \mathrm{Survival}_B^{\,\text{clamp-to-A}}.
$$

**Falsification threshold**: $\Delta_{\text{clamp}} < 1\sigma_{\text{seed}}$ rules out P2: the modulator's recurrent state, even if context-distinguishable (P1), is not *causally* responsible for the policy difference between contexts. This is the load-bearing falsifier — it directly answers the question "does the architecture's modulator do work it would not do if you replaced it with a fixed average vector".

## 4. Alternative interpretation: GRU-already-suffices

The contrast the prompt asks for: maybe the **unmodulated** task GRU's 128-dim hidden state is itself enough for context-conditioning. Frame (b) papers report gains of NPN over MLP backbones; our baseline is **already a recurrent backbone** (review §1.A: 128-dim GRU). This is the reduction-to-redundancy worry.

### 4.1 When is the modulator additive over the GRU rather than redundant?

The architectural-bottleneck argument in `NEUROMODULATION_ALGORITHM.md` §1 is the load-bearing answer: the modulator's value over the task GRU is *not* "more recurrent capacity" but **timescale-separated, gain-form, multi-site coordination**.

Concretely, the modulator's contribution is non-redundant with the task GRU when one or more of the following holds:

1. **Timescale separation (H5 in `NEUROMODULATION_ALGORITHM.md` §1.4).** The modulator GRU's effective time constant exceeds the task GRU's. For two contexts that switch at episode boundaries (Candidate A's regime), the task GRU's hidden state is **reset** at every reset; its context inference has to re-converge each episode from observation alone. The modulator's GRU is also reset at episode boundaries in the current code path, so this argument is weaker than it sounds — but the modulator can still build a *faster* within-episode context inference if the heads' init-bias keeps the early-episode signal stronger than the task GRU's slower-to-warm-up hidden state.
2. **Gain-form intervention vs. additive recurrent state.** The task GRU adds context information *additively* into $h^{\text{task}}$, which still has to be linearly read out by the actor head. The FiLM modulator multiplicatively gates *the encoder features*, restructuring the **basis** the actor head reads. This is the Tsuda hypertube argument (review §K): a multiplicative gain $f$ shifts the policy into a non-overlapping manifold without requiring additive separability in $h^{\text{task}}$. Where the two contexts' optimal policies live in *non-linearly-separable* regions of the encoder feature space, the modulator can route them to disjoint hypertubes while the task GRU alone cannot.
3. **Multi-site coordination (H4).** The task GRU cannot coordinate Injection A (perception), B (memory), and C (temperature) from a *shared, low-dim* affective latent. Three independent linear readouts from $h^{\text{task}}$ would do this only if the contexts' optimal $(\gamma, z_{\text{mem}}, \tau)$ triples lie on a 3-d affine subspace identifiable by gradient descent — possible but not architecturally enforced.

### 4.2 The implication for falsification

If $\Delta_{\text{CKA}} > 0.05$ **and** $\Delta_{\text{clamp}} > 1\sigma_{\text{seed}}$ **but** modulated joint-mixture survival ≈ unmodulated joint-mixture survival, the conclusion is **not** "the architecture is broken" — it is "the modulator builds context-distinguishable state and uses it, but the task GRU's 128-dim hidden state independently captures the same information; the architecture is redundant rather than non-functional." That is a softer, but still publishable, finding — and it directly argues the project should pursue the **Phase 0** T/P split + opioid head before re-running Candidate A.

This three-way table (CKA × clamp × survival) is the actual decision tool from the experiment, not the survival number alone.

## 5. Recommendation: which 2-context mixture

Triage §4 offers A1 (active vs. passive predator) and A2 (resource-layout swap). The cleanest test of §4's redundancy-vs-additivity question is:

**Pick A1, not A2.** Reasons:

1. A1's discriminator is **predator movement signature** — a *temporal* feature available only to a recurrent encoder. Both the task GRU and the modulator GRU have to integrate over time to detect it. This is the regime where claim (2) of §4.1 (multiplicative-gain hypertube routing) gets the hardest test: if the modulator wins, it is *not* because it has more state than the GRU — it is because gain-form intervention is doing the work.
2. A2's discriminator is **spatial layout** — purely a function of the *current* observation. The task GRU's 128-dim hidden is overkill for that; the redundancy worry of §4 is at its strongest. A null on A2 is barely informative; a positive on A2 is barely surprising. Avoid.
3. A1 inherits matched olfactory properties (`01-interoNocicept_sameProp.yaml` ∪ `02-sameProp_R2_passivePredator.yaml`) — the only available context cue is movement. This is the exact regime the project's chronic-pain analog (H5) has been written to address (slow-affective-state integration of predator behaviour over an episode).

A2 should be held as a follow-up if A1 is **null**, to disambiguate "the task GRU was already enough" from "the modulator is broken".

## 6. The blocker: measurement-only first

The most-load-bearing claim of this memo is that **$\Delta_{\text{CKA}}$ on existing v8-trained modulators is computable now, on existing logged trajectories, with no new training**. If $\Delta_{\text{CKA}} \approx 0$ across the v8 sweep, P1 already fails on the *current* modulator family — and Candidate A would be expected to fail for the same reason. That would change the recommendation from "run A1" to "fix the modulator (Phase 0) before any further compute".

So the **gating** measurement-only probe is: pull the v8 modulated-agent rollouts (heterogeneity sweep + temp-clip rerun, summary `docs/experiments/summaries/20260509_1421_nmn_comparison_study.md`), compute per-state $h^{\text{mod}}$ traces, and produce $\Delta_{\text{CKA}}$ between matched states across noise regimes (the v8 mix already presents a context-like manipulation in the heterogeneity-axis runs). Cost: hours, not days.

- If $\Delta_{\text{CKA}}$ is healthy on v8 (≥ 0.05), proceed to Candidate A1.
- If $\Delta_{\text{CKA}} < 0.05$ on v8, escalate to `senior-developer`: the modulator does not build context-distinguishable state even on training-distribution variation. Phase 0 (T/P split, opioid head, EMA) must run first.

## 7. Required experiments (sketch)

If A1 is greenlit:

| Cell | Mix | Architecture | Seeds | Notes |
|---|---|---|---|---|
| A1-mod | active ∪ passive predator | branched FiLM modulator (current) | 3 | Joint-mixture training |
| A1-unmod | active ∪ passive predator | unmodulated GRU baseline | 3 | Same hidden size, same obs |
| A1-spec-A | active only | unmodulated | 3 | Per-context specialist (ceiling reference) |
| A1-spec-B | passive only | unmodulated | 3 | Per-context specialist (ceiling reference) |

**Read-out**: joint-mixture survival (mod vs. unmod), per-context survival from the joint runs (mod vs. unmod), $\Delta_{\text{CKA}}$ on rollouts, $\Delta_{\text{clamp}}$ at eval. Compute footprint within triage §4 budget (12 cells).

## 8. Required architecture changes (sketch)

- **Mixture-of-envs at episode reset**: cleanly via a `--mixture-mode` flag on `train.py` (~half-day `developer` work per triage §5). Workaround via alternating-episode continual schedule exists but is uglier.
- **Modulator-state clamp at eval**: a small `evaluate.py` patch exposing a `--mod-clamp ref-state.npy` option (~1 day). Required for the §3.2 falsifier; not required for the §3.1 falsifier or the joint-survival number.

Both changes are **contained** — single new flag in `train.py`, single new flag in `evaluate.py`. No invasive cross-cutting changes to encoder, GRU, actor, or critic. This is well below the cost of Phase 0.

## 9. Risk register

- **Risk 1 — A1's predator-movement cue is too weak.** Even a recurrent expert may not learn an active/passive discrimination from movement signature alone in our env. Mitigation: include a baseline "obs feature: predator-velocity-magnitude" diagnostic to confirm the cue is learnable at all.
- **Risk 2 — Per-context specialists do not separate.** If A1-spec-A and A1-spec-B reach the same survival, the contexts are not actually task-conflicting and the (b)-frame test is degenerate. Mitigation: pre-flight 1-seed runs per specialist before launching the full cell grid.
- **Risk 3 — $\Delta_{\text{CKA}}$ threshold is wrong.** The 0.05 number is eyeballed from review §B.6, not calibrated. Mitigation: report the full $\Delta_{\text{CKA}}$ distribution across seeds and let the unmodulated-baseline's task-GRU CKA serve as a lower bound.
- **Risk 4 — Reduction-to-redundancy.** As §4 flags, a positive joint-survival null could be redundancy rather than failure. The three-way table (CKA × clamp × survival) is the only way to disambiguate; do not interpret survival in isolation.
- **Risk 5 — v8 logs do not contain modulator hidden-state traces.** If `mod_h` is not logged in the v8 rollouts, the §6 measurement-only probe is blocked and we have to run a small re-evaluation pass on existing checkpoints. Mitigation: `experiment-analyzer` to confirm log completeness before committing to the gating probe.

---

## Next steps (named hand-offs)

1. **`experiment-analyzer`** — first. Run the §6 measurement-only probe on the v8 NMN comparison study (`docs/experiments/summaries/20260509_1421_nmn_comparison_study.md`). Compute $\Delta_{\text{CKA}}$ between matched states across the heterogeneity-axis variants. Confirm whether `mod_h` is logged; if not, re-evaluate from checkpoints. Report at `docs/experiments/active/nmn_meta_pivot/measurement_only_probe.md`.
2. **`experiment-designer`** — *gated on (1)*. If $\Delta_{\text{CKA}} \geq 0.05$ on v8, lock the Candidate A1 cell grid (§7), produce configs under `configs/active/nmn_meta_a1/`, and the design doc at `docs/experiments/active/nmn_meta_pivot/a1_design.md`.
3. **`senior-developer`** — *gated on (1) negative*. If $\Delta_{\text{CKA}} < 0.05$ on v8, write a Phase 0 engineering plan (T/P split + opioid head + EMA) under `docs/develop/active/neuromodulation/`.
4. **`developer`** — independent of (1)'s outcome. Implement the `--mixture-mode` flag on `train.py` and the `--mod-clamp` flag on `evaluate.py` (§8). These are needed regardless.
