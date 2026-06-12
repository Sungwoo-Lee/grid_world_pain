# Visualizing the Dreamer agent's "dream" — RL / world-model semantics memo

**One-line summary.** A trained `dreamer_srl` agent can roll its internal world model forward without looking at the world ("imagine"); this memo establishes the RL semantics of that imagined rollout, confirms the decoder makes per-modality dream-vs-real comparison feasible, names the four divergence observables that quantify dream fidelity, and lays out three visualization approaches against the user's locked decision slate.

---

## LOCKED DECISIONS (design discussion, 2026-06-11)

These were fixed by the user during the design discussion and bind any downstream build. They are recorded here verbatim so the figure spec is unambiguous; the rationale for each lives in the body sections below.

1. **Primary view** = a **per-modality dream-strip** (one horizontal ribbon per sensor channel, imagined steps along x) **with a latent-KL drift footer** (a single drift curve under the strip showing how far the dream has drifted from reality at each imagined step).
2. **Comparison rows** = three rows stacked per modality: **open-loop** (the agent imagines from its own prior, never re-seeing the world), **teacher-forced** (the agent is re-fed the real observations so only one-step dynamics error shows), and **ground truth** (what actually happened).
3. **Start-points** = **event-anchored** — dreams are launched from the rising edge of meaningful events: just before a predator enters sensing range, just before a contact/damage event, and just before death.
4. **Scope** = **single-episode microscope first** — one vivid recorded episode, hand-picked onsets, before any aggregate-across-episodes statistic.
5. **Defaults:**
   - **Horizon** = the agent's training imagination horizon, plus one longer multiple (e.g. if trained at H=15, also render H≈30–45) to expose drift the training horizon hides.
   - **Categorical channels** (visual object-class, proprioception direction) rendered as **softmax-probability heatmaps** over their class slots — *not* argmax icons — so the dream's *uncertainty* over object class is visible (the hypervigilance-relevant signal).
   - **Headline drift scalar** = the **categorical KL** between the dream's prior latent and the posterior latent that the real next observation would have induced, written $\mathcal{D}_k$ below.
   - **Imagination = mode** (greedy actor / argmax categorical) for the first figure, so the dream is deterministic and reproducible; sampled rollouts are a later variant.

**Companion memo.** The practical visualization-design + tooling-integration analysis (renderer reuse, `build_sensory_viz` seam, the `eval_rollout.py` Dreamer-harness gap, candidate layouts A–D) lives in the research-postdoc memo at [`docs/project/ideas/dreamer_dream_visualization_design_space.md`](../ideas/dreamer_dream_visualization_design_space.md). This memo owns the **RL/world-model semantics**; that one owns the **build**. Where they touch (decoder trust, which divergence metric is "correct"), this memo is the authority and that memo defers to it.

---

## 1. Concept in one paragraph

A DreamerV3-style agent maintains a **recurrent state-space model (RSSM)**: a deterministic recurrent carry $h_t$ plus a stochastic categorical latent $z_t$ that together summarise the agent's belief about the world. At any real timestep the agent has a **posterior** latent $z_t \sim q(z_t \mid h_t, x_t)$ formed by *looking at* the real observation $x_t$, and the model can also produce a **prior** latent $\hat z_t \sim p(z_t \mid h_t)$ that *predicts* the latent *without* the observation. "Dreaming" (imagination) means: take a real latent as the seed, then repeatedly (i) let the actor pick an action from the current latent, (ii) advance the deterministic carry, (iii) draw the *prior* latent — never re-feeding a real observation. Because the model also carries a **decoder** $D$ that maps a latent back to a reconstructed 27-dim observation, we can render *what the dream thinks it senses* at every imagined step and lay it beside what actually happened. The scientific payload: the gap between dream and reality, resolved per sensor channel, is direct evidence about *what the agent's network represents about damage and threat* — and whether, after injury, the dream **over-predicts threat** (a hypervigilance signature).

---

## 2. Mathematical statement

### 2.1 RSSM and the two latents

Let the model state at time $t$ be $s_t = (h_t, z_t)$ with deterministic carry $h_t$ and stochastic categorical latent $z_t$ (in this codebase $z_t$ is a $S \times C$ array of $S = $ `num_categoricals` independent $C = $ `num_classes` categoricals; `src/algorithms/dreamer_srl/train.py:726–727`). The RSSM components:

$$
\begin{aligned}
h_t &= f_\theta(h_{t-1}, z_{t-1}, a_{t-1}) && \text{(recurrent / sequence model)}\\
\hat z_t &\sim p_\theta(z_t \mid h_t) = \mathrm{Cat}(\mathrm{softmax}(\ell^{\text{prior}}_t)) && \text{(prior — no observation)}\\
z_t &\sim q_\theta(z_t \mid h_t, x_t) = \mathrm{Cat}(\mathrm{softmax}(\ell^{\text{post}}_t)) && \text{(posterior — uses } x_t)\\
\hat x_t &= D_\theta(h_t, z_t) && \text{(decoder → 27-dim reconstructed obs)}\\
\hat r_t &= R_\theta(h_t, z_t), \quad \hat c_t = C_\theta(h_t, z_t) && \text{(reward, continue heads)}
\end{aligned}
$$

The prior/posterior logits $\ell^{\text{prior}}_t, \ell^{\text{post}}_t$ are exactly the `prior_logits` / `posterior_logits` reshaped to $[T,B,S,C]$ at `train.py:728–732`.

### 2.2 Imagined (open-loop) rollout — the "dream"

Seed with a real model state $s_{t_0} = (h_{t_0}, z_{t_0})$ where $z_{t_0}$ is the **posterior** at the launch step (the last moment the dream is grounded in reality). Then for $k = 1, \dots, H$:

$$
a_{t_0+k-1} = \pi_\phi(\,\cdot \mid s_{t_0+k-1}), \quad
h_{t_0+k} = f_\theta(h_{t_0+k-1}, z_{t_0+k-1}, a_{t_0+k-1}), \quad
\hat z_{t_0+k} \sim p_\theta(\cdot \mid h_{t_0+k}).
$$

This is **open-loop**: only **prior** latents are used after the seed; no real $x$ ever re-enters. The decoded dream observation at imagined step $k$ is $\hat x_{t_0+k} = D_\theta(h_{t_0+k}, \hat z_{t_0+k})$. (Code: `WorldModel.imagine`, `agent.py:1727`; under LOCKED default 5, the actor and the categorical draw both take the **mode**, making the rollout deterministic.)

### 2.3 Teacher-forced rollout — the dynamics-only control

The teacher-forced comparison row re-feeds the *real* observations, so the latent at each step is the **posterior** $z_{t_0+k} \sim q_\theta(\cdot \mid h_{t_0+k}, x_{t_0+k})$ and the only thing being tested is the *one-step* prior-vs-posterior gap, not compounding policy/model drift. This isolates **world-model dynamics error** from **policy + compounding error**. (Code: `WorldModel.observe`, `agent.py:1622`, which already produces posterior latents and `reconstructed_obs` at `agent.py:1712`.)

> **Why both rows matter (the identifiability point).** Open-loop divergence confounds three error sources: (a) one-step model error, (b) compounding of that error over the horizon, and (c) *policy* drift — the dream's imagined action sequence diverging from what the agent really did. Teacher-forcing strips (b) and (c), leaving only (a). A dream that looks faithful teacher-forced but drifts wildly open-loop is telling you the **dynamics are fine but the rollout/policy compounds** — a very different diagnosis from a model that is wrong at one step. The LOCKED three-row layout (open-loop / teacher-forced / ground-truth) makes this decomposition readable directly off the figure.

### 2.4 CASE-YES: the decoder exists — per-modality dream rendering is feasible

The feasibility question ("can we show *what* the dream sensed, not just *that* it diverged?") resolves **YES** for `dreamer_srl`. Confirmed in code:

- `class MLPDecoder` — latent → reconstructed 27-dim obs (`src/algorithms/dreamer_srl/agent.py:1203`; forward at `:1276`, returns `reconstructed_obs: [..., obs_dim]` at `:1283`).
- `WorldModel.decoder` is assigned (`agent.py:1618`) and invoked inside `observe` to produce `reconstructed_obs = jax.vmap(self.decoder)(latent_flat)` (`agent.py:1712`), surfaced in the output dict (`agent.py:1722`).
- `WorldModel.imagine` (`agent.py:1727`) produces the imagined prior latents that we decode the same way.

**Load-bearing caveat for the categorical channels.** The decoder is trained under a single **symlog-Gaussian** reconstruction NLL over the *whole* 27-vector — `obs_log_prob = -½‖symlog(x̂) − symlog(x)‖²` at `train.py:704–712` — **not** with per-modality categorical heads. So the decoder emits continuous symlog-space values for *all* 27 dims, including the visual object-class and proprioception-direction slots that are *semantically* categorical. Consequently:

- The "softmax-probability heatmap" rendering of the categorical channels (LOCKED default 5) is a **visualization-side reinterpretation** — take the decoded raw values over a categorical slot group, optionally invert symlog, and softmax them — **not** a native network output. This is sound (it shows where the decoder's continuous reconstruction concentrates across classes) but it must be labelled as a *visualization choice*, because the reconstruction loss never enforced a proper categorical distribution there. A dream that "sees PRD with probability 0.6" is reporting *the softmax of the decoder's symlog reconstruction over the object-class slots*, not a calibrated belief. Flag for `math-reviewer` if the figure caption ever claims calibrated class probabilities.

### 2.5 The four diagnostic observables (the fidelity signals)

| # | Symbol | Definition | Needs decoder? | What it isolates |
|---|--------|-----------|----------------|------------------|
| O1 | $\mathcal{D}_k$ | **Categorical latent KL** between dream prior and real-obs posterior at imagined step $k$ | No (latent only) | Pure model drift, modality-agnostic — **the LOCKED headline drift scalar** |
| O2 | $\Delta x_k^{(m)}$ | Per-modality decoded-obs error, channel $m$ | Yes | *Which sense* the dream gets wrong (the hypervigilance-relevant one) |
| O3 | $\delta r_k$ | Reward-prediction error $\hat r_{t_0+k} - r_{t_0+k}$ | No (reward head) | Does the dream foresee value/danger of the path? |
| O4 | $\delta c_k$ | Continuation-prediction error $\hat c_{t_0+k} - c_{t_0+k}$ | No (continue head) | Does the dream foresee death/termination? |

The headline drift scalar $\mathcal{D}_k$ is the **categorical KL** already computed in the world-model loss, summed over the $S$ categoricals and $C$ classes (`train.py:735–739`):

$$
\mathcal{D}_k \;=\; \sum_{s=1}^{S} \mathrm{KL}\!\Big(\, q_\theta(z^{(s)}_{t_0+k}\mid h, x) \;\big\|\; p_\theta(\hat z^{(s)}_{t_0+k}\mid h) \,\Big)
\;=\; \sum_{s=1}^{S}\sum_{c=1}^{C} q^{(s)}_c \big(\log q^{(s)}_c - \log p^{(s)}_c\big),
$$

with $q$ the posterior (real-obs-induced) and $p$ the prior (dream) categorical, $\log q = \texttt{log\_softmax}(\ell^{\text{post}})$, $\log p = \texttt{log\_softmax}(\ell^{\text{prior}})$. This is **the same quantity the world model is trained to minimise** (the KL-balanced regulariser), which is why it is the principled drift scalar: $\mathcal{D}_k$ rising over the horizon is precisely the model leaving the region where its own training objective held it accountable.

> **Direction convention.** Note $\mathcal{D}_k = \mathrm{KL}(q \| p)$ here uses the posterior as the reference (forward KL from the dream's perspective: "how surprised would the dream be by reality"). The DreamerV3 KL-balancing trick splits the gradient between $\mathrm{KL}(q\|p)$ (dynamics loss) and $\mathrm{KL}(\mathrm{sg}(q)\|p)$ / $\mathrm{KL}(q\|\mathrm{sg}(p))$ for *training*; for *visualization* we want the un-stopped, un-balanced $\mathrm{KL}(q\|p)$ as a scalar readout. Keep the visualization KL distinct from the training KL terms — they share a formula but not a purpose.

---

## 3. Project mapping

| Concept here | Codebase correspondent |
|---|---|
| RSSM carry + categorical latent | `WorldModel` / `rssm` (`agent.py:1570`), `num_categoricals`/`num_classes` (`train.py:726`) |
| Posterior latent (teacher-forced row) | `WorldModel.observe` → `posterior_logits`, `reconstructed_obs` (`agent.py:1622`, `:1712`) |
| Prior latent (open-loop row) | `WorldModel.imagine` (`agent.py:1727`), `prior_logits` (`train.py:731`) |
| Decoder $D_\theta$ (O2) | `MLPDecoder` (`agent.py:1203`), assigned `agent.py:1618` |
| Reward/continue heads (O3/O4) | reward/continue models (`train.py:715`, `:719`) |
| Headline drift $\mathcal{D}_k$ (O1) | categorical KL (`train.py:735–739`) |
| Per-modality split for rendering | `build_sensory_viz(obs, state, params, true_obs)` (`src/environment/sensor.py`) — companion memo F3 |
| Dream-solid / real-ghost overlay | `render_jax_state` two-layer draw (`src/environment/renderer.py`) — companion memo F2 |
| **The build gap** | `scripts/eval_rollout.py` raises `NotImplementedError` for the Dreamer branch — companion memo F4 |

**Hypothesis anchor.** The hypervigilance reading of O2 (decoded object-class channel) maps to **H5 / the chronic-pain analog** and to **Phase 4 (hypervigilance readout)** in [`project_plan.md`](../project_plan.md): the falsifiable claim is that *after injury, the dream's object-class softmax over threat classes (PRD/DNG) is elevated relative to ground truth*. This is a **representational-level** readout — it is direct evidence about what the network encodes about threat, the level Paper 1 needs at least one category to carry. It does **not** change how G1/G2 are tested; it is mechanism-inspection infrastructure, usable as soon as a Dreamer checkpoint exists.

---

## 4. Three visualization approaches (against the locked slate)

The companion memo enumerates candidate layouts A–D in full; here is the RL-semantics view of the three that survive, mapped onto the LOCKED primary view.

**Approach 1 — Per-modality dream-strip + latent-KL footer (LOCKED PRIMARY).** One ribbon per sensor channel, imagined step on x, three rows per modality (open-loop / teacher-forced / ground-truth per LOCKED decision 2), categorical channels as softmax heatmaps (LOCKED default 5), and a single $\mathcal{D}_k$ drift curve as the footer (LOCKED headline scalar). This is the richest and most diagnostic: it shows *that* (footer $\mathcal{D}_k$) and *what* (per-modality O2) and *whether the policy or the dynamics is to blame* (open-loop vs. teacher-forced rows). Build rides on the existing `build_sensory_viz` + renderer overlay; the cost is the Dreamer rollout harness. **This is what the first figure builds.**

**Approach 2 — Dream-scene reconstruction (figure-for-readers, later).** Collapse the decoded visual + collision channels into a reconstructed egocentric local grid played as a clip beside reality. Most intuitive for a non-technical reader / paper figure, but lossy (discards scalar interoceptive channels to side-gauges) and requires new inverse-rendering glue (decoded class softmax → icon). Deferred; revisit once Approach 1 exists and a reader-facing figure is wanted.

**Approach 3 — Divergence dashboard (decoder-free fallback + aggregate).** $\mathcal{D}_k$, $\delta r_k$, $\delta c_k$ curves over the horizon plus scalar-channel line overlays — no decoder needed. This is the **decoder-trust fallback** (if a checkpoint's reconstruction is poor, the categorical-heatmap claim in Approach 1 weakens, but $\mathcal{D}_k$/reward/continue remain valid) **and** the natural **aggregate** view once the single-episode microscope graduates to mean-divergence-across-onsets (the quantitative follow-up to LOCKED decision 4's microscope-first scope).

---

## 5. Closest published precedents

- **DreamerV3 (Hafner et al., 2023).** The canonical "imagined rollout vs. real" film-strip and the symlog/two-hot decoder + categorical RSSM this codebase ports. *The closest thing to our primary view is DreamerV3's reconstruction film-strip; we differ in that (i) our observation is a 27-dim heterogeneous interoceptive vector with no image channel, so we render per-modality ribbons instead of pixel frames, and (ii) we anchor dreams at threat/damage/death events to surface a hypervigilance signature rather than sampling uniformly.*
- **IRIS (Micheli et al., 2023) / world-model imagination visualizations.** Discrete-code transformer world model; same open-loop-vs-real diagnostic logic. We differ by using the **categorical-latent KL** as the headline drift scalar rather than reconstruction error alone.
- **TD-MPC2 (Hansen et al., 2024).** Value-equivalent latent model; relevant as a *contrast* — it deliberately has **no decoder**, so it could only ever produce our Approach 3 (divergence dashboard). That our `dreamer_srl` *has* a decoder is exactly why Approaches 1–2 are open to us.
- **Plan2Explore / latent-drift analyses.** Precedent for treating prior-vs-posterior KL as a model-disagreement / drift signal — the lineage of our $\mathcal{D}_k$.

Missing-reference note for `docs/project/references/`: the DreamerV3 and IRIS papers should be the primary anchors; neither is currently a curated source under `references/Dreamer/` as a *visualization-method* citation.

---

## 6. Failure modes

1. **Symlog decoder ≠ categorical belief (most important).** As in §2.4: rendering the visual/proprio channels as softmax heatmaps reads a categorical structure *out of* a continuous symlog reconstruction the network was never asked to calibrate. The heatmap is a faithful picture of the decoder's reconstruction, **not** a calibrated posterior over object class. A caption claiming "the dream believes there is a predator with p=0.6" overclaims. Safe framing: "the decoded object-class channel concentrates on PRD."
2. **Open-loop drift mis-attributed to dynamics.** Without the teacher-forced row, a faithful one-step model that simply *compounds* over the horizon looks identical to a broken model. LOCKED decision 2 (three-row layout) is what prevents this; do not drop the teacher-forced row to save space.
3. **Mode-collapse vs. sampled rollout.** LOCKED default 5 (imagination = mode) makes the dream deterministic — but a *mode* rollout can look artificially confident/faithful because it never explores the tails the stochastic latent would. The first figure is mode; a sampled-rollout variant (with a $\mathcal{D}_k$ band over samples) is the honest follow-up before any strong fidelity claim.
4. **Horizon beyond training horizon is extrapolation.** The "one longer multiple" (LOCKED default 5) deliberately runs the model past where it was trained to roll out in imagination; rising $\mathcal{D}_k$ there may reflect *untrained horizon* rather than a *meaningful* drift. Mark the training-horizon boundary on the x-axis.
5. **Event-anchoring selection bias.** Dreams launched only at threat/damage/death onsets (LOCKED decision 3) sample the *hardest* states for the model; the resulting fidelity is a lower bound, not a representative average. State this when comparing to any baseline.

---

## 7. Empirical signatures

- **Healthy world model:** $\mathcal{D}_k$ (O1) rises gently and sub-linearly over the horizon, teacher-forced rows track ground truth tightly while open-loop rows drift only after several steps, reward/continue errors (O3/O4) stay small until genuinely unpredictable events.
- **Hypervigilance signature (the target):** after an injury/damage onset, the decoded threat-class channel (O2, visual PRD/DNG slots) shows **elevated softmax mass relative to ground truth** in the open-loop dream — the dream *hallucinates threat that reality does not contain* — while the teacher-forced row does not, localising the effect to imagination/policy rather than one-step dynamics.
- **Refutation:** if O2's threat-class mass tracks ground truth post-injury and $\mathcal{D}_k$ is indistinguishable pre/post-injury, there is no representational hypervigilance to report from this readout (a clean null).

---

## 8. Open questions

1. **Is $\mathcal{D}_k$ or reward-error $\delta r_k$ the *right* fidelity metric for the hypervigilance claim?** $\mathcal{D}_k$ is modality-agnostic drift; the hypervigilance story is specifically about the *threat channel*. We may want a **channel-restricted** divergence (decoded-obs error on PRD/DNG slots only) as a fifth observable alongside the global $\mathcal{D}_k$.
2. **Mode vs. sampled for the headline claim.** §6.3 — at what point does the fidelity claim require the sampled-rollout band rather than the mode rollout?
3. **Does the symlog decoder reconstruct the categorical channels well enough** on a real checkpoint to support Approach 1's heatmaps, or does the decoder-trust gate force Approach 3? Empirical, checkpoint-dependent — resolve at first build.
4. **Aggregate statistic.** Once the microscope works, what is the right population quantity — mean $\mathcal{D}_k(k)$ ± band across onsets, or a hallucination-rate (fraction of onsets where threat-channel mass exceeds ground truth by a threshold)?

---

## Next steps

- **`experiment-designer` / `senior-developer`** — own the build. The load-bearing prerequisite (from companion memo F4): **the Dreamer branch of `scripts/eval_rollout.py` is unimplemented** (`NotImplementedError`). A `dreamer_srl` checkpoint loader + `WorldModel.imagine()`-and-decode rollout (open-loop) and `WorldModel.observe()` rollout (teacher-forced) is the shared prerequisite for the LOCKED primary view. The visualization layer is a thin composition over `build_sensory_viz` and `render_jax_state`.
- **`math-reviewer`** — gate the categorical-channel framing (§2.4): confirm any figure caption distinguishes "decoder reconstruction softmax" from "calibrated class posterior" before publication.
- **`professor-dl-theory`** — only if the decoder's symlog-vs-categorical mismatch motivates an *architectural* change (per-modality categorical decoder heads). That architecture call is theirs; the RL coupling is mine.
- **Companion build/tooling detail** — [`docs/project/ideas/dreamer_dream_visualization_design_space.md`](../ideas/dreamer_dream_visualization_design_space.md).
