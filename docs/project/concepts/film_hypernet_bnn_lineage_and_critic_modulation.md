---
title: "FiLM ⊂ Hypernet ⊂ Bayesian Hypernet vs BNN — and where γ_Bellman modulation has to live"
status: draft
audience: research-postdoc, math-reviewer, experiment-designer, senior-developer
last_updated: 2026-05-16
classification: concept-memo (investigation-style follow-up)
type: targeted Q1/Q2 follow-up to math-reviewer audit of film_neuromod_integration.md §10
inputs:
  - docs/project/concepts/film_neuromod_integration.md
  - docs/project/references/FiLM/film_neuromod_integration_synthesis.md
  - docs/project/references/FiLM/reviews/perez_2018_film.md
  - docs/project/references/FiLM/reviews/ha_2016_hypernetworks.md
  - docs/project/references/FiLM/reviews/krueger_2017_bayesian_hypernets.md
  - docs/project/references/FiLM/reviews/kendall_gal_2017_uncertainties.md
  - docs/project/references/FiLM/reviews/galanti_wolf_2020_hypernet_modularity.md
  - docs/project/references/FiLM/reviews/turkoglu_2022_film_ensemble.md
  - docs/project/references/FiLM/reviews/abdollahzadeh_2021_multimodal_meta.md
  - docs/project/references/neuromodulatory_algorithms/reviews/doya_2002_metalearning_neuromodulation.md
  - docs/project/references/neuromodulatory_algorithms/reviews/lee_2024_lifelong_rl.md
  - docs/project/references/neuromodulatory_algorithms/reviews/xing_2022_neuromodulation_rl_environment_changes.md
  - docs/project/references/neuromodulatory_algorithms/reviews/wang_2024_neuromod_meta.md
  - docs/project/references/neuromodulatory_algorithms/reviews/rodriguezgarcia_2026_ne_stability_gap.md
  - docs/project/directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md
related:
  - docs/project/concepts/film_neuromod_integration.md
---

# FiLM ⊂ Hypernet ⊂ Bayesian Hypernet vs BNN — and where γ_Bellman modulation has to live

## §1 — Plain-English entry point

The project's main concept memo on FiLM and neuromodulators ([film_neuromod_integration.md](film_neuromod_integration.md)) got a math-reviewer audit. Two findings need follow-up investigation before the next direction-memo revision (v4).

**Question 1 (the family tree).** The memo lists four families of context-conditional networks — **FiLM** (a small side-network produces scale $\gamma$ and shift $\beta$ that multiply and add to the main network's activations); **Hypernetworks** (a small network *outputs the weights* of a larger one); **Bayesian Hypernetworks** (the small network is *stochastic*, sampling weight posteriors); and ordinary **Bayesian Neural Networks** (a posterior is placed directly over the main network's weights). The memo treats these as a cluster but does not state, formally, which family is a special case of which. **Headline answer**: FiLM is a strict special case of Hypernetwork; Hypernetwork is the point-mass limit of Bayesian Hypernetwork; Bayesian Hypernetwork is *orthogonal* to Bayesian NN — they put uncertainty at different architectural levels. The project's three-site FiLM substrate is therefore a deterministic Hypernetwork, and the FiLM-Ensemble + heteroscedastic-precision combination it proposes is a finite-ensemble approximation of a Bayesian Hypernetwork with a heteroscedastic likelihood.

**Question 2 (where to modulate the discount factor).** The math-reviewer ruled out the memo's claim that an additive shift at the GRU update-gate ($\beta^{(B)}$) is equivalent to a state-dependent **Bellman discount factor** (γ_Bellman, the parameter controlling how far ahead the value function looks). The GRU update gate is per-unit state-retention, not value-function horizon — different unit, different quantity. So if we still want a γ_Bellman channel (the project's hypothesised 5-HT analogue), where does it go? Should we modulate the **critic network** that outputs $V(s)$? **Headline answer**: A clean γ_Bellman channel cannot live in the forward pass of a FiLM substrate. Its cleanest mathematical home is the TD-target computation $y_t = r_t + \gamma(c) V(s_{t+1})$, in the *learning update*, not the forward pass. Modulating critic output is **value-scale** modulation, not horizon modulation — a γ_Bellman *proxy* at best. We recommend **Option A** below: drop the 5-HT/γ_Bellman channel from the project's substrate-level claims for v4 and acknowledge that the FiLM substrate cleanly carries ACh/α and NA/β but not 5-HT/γ — exactly the position Lee 2024 already arrived at when they declined to instantiate the 5-HT branch in the Doya-DaYu agent.

---

## §2 — Q1: The mathematical lineage of FiLM, Hypernet, Bayesian Hypernet, BNN

### §2.1 — Setup and notation

Let the **main network** be a function $y = f_W(x)$ with weights $W \in \Theta$. Let $c$ denote a *context signal* (a task ID, a question embedding, a neuromodulator state, anything the architecture wants to condition on).

The four families differ in two binary choices: (a) are the weights of the main network **deterministic or stochastic**, and (b) are they **context-conditional or unconditional**?

- **Standard NN** — $W$ is a single deterministic point estimate $\hat W$. Forward pass $f_{\hat W}(x)$. No conditioning; no posterior.
- **FiLM (Perez et al. 2018).** A side-network $(\gamma(c), \beta(c)) = g_\phi(c)$ produces a per-channel scale and shift; the main network's activations at layer $\ell$ are modulated as $h_\ell \mapsto \gamma_\ell(c) \odot h_\ell + \beta_\ell(c)$. The main network's *trained* weights $W$ are deterministic and unconditional; only the FiLM scalars are context-dependent.
- **Hypernetwork (Ha et al. 2016).** A side-network $W = g_\phi(c)$ produces *the entire weight tensor* of the main network as a function of $c$. The main network's effective weights are context-conditional but deterministic given $c$.
- **Bayesian Neural Network / BNN (Kendall & Gal 2017 context; broader BDL literature).** A *posterior distribution* $p(W \mid \mathcal{D})$ is placed directly over the main network's weights given training data $\mathcal{D}$. Forward predictions marginalise: $p(y \mid x, \mathcal{D}) = \int p(y \mid x, W) p(W \mid \mathcal{D}) dW$, approximated by MC dropout, deep ensembles, or variational inference. No context-conditioning; uncertainty is placed on $W$ itself.
- **Bayesian Hypernetwork / BHN (Krueger et al. 2017).** A side-network $W = h_\phi(\epsilon, c)$ with $\epsilon \sim \mathcal{N}(0, I)$ produces a *stochastic* main-network weight tensor. If $h$ is an invertible normalising flow, the implicit distribution $q_\phi(W \mid c)$ can be evaluated and trained variationally. This combines hypernetwork's context-conditioning with BNN's posterior-over-weights.

### §2.2 — The 2×2 taxonomy

The four families occupy the corners of a 2×2 over {deterministic, stochastic} × {unconditional, context-conditional} main-network weights:

| | **Weights unconditional** ($W$ does not depend on $c$) | **Weights context-conditional** ($W = W(c)$) |
|---|---|---|
| **$W$ deterministic** (point estimate) | Standard NN | **Hypernetwork** (Ha et al. 2016) — and **FiLM** as a restricted hypernet (Perez et al. 2018) |
| **$W$ stochastic** (posterior) | **BNN** (Kendall & Gal 2017; MacKay 1992; Neal 1995) | **Bayesian Hypernetwork** (Krueger et al. 2017) |

Three observations the project memo's §3 taxonomy did not make explicit:

1. **FiLM is *inside* the deterministic-hypernet cell, not adjacent to it.** The Krueger 2017 review section 2.2 says this in one sentence: "FiLM, CBN, CIN are special cases of hypernetworks that output only per-channel $\gamma, \beta$ instead of full weights." This is the inclusion we make rigorous in §2.3.
2. **The 5-paper FiLM-corpus enumeration in [film_neuromod_integration.md §3](film_neuromod_integration.md) lives entirely in the right column of the top row** — all of vanilla FiLM, CIN, AdaIN, Temporal FiLM, KML, Sparse MoE, and BatchEnsemble are deterministic hypernetworks at different restriction levels. FiLM-Ensemble (Turkoglu 2022) is the lone member that crosses into the *stochastic* row, because it produces an *ensemble* of weight settings rather than one. We position it precisely in §2.4.
3. **The BNN and BHN cells are not unified by the FiLM corpus.** The project's `references/FiLM/` directory contains Kendall & Gal 2017 (BNN, via MC dropout) and Krueger 2017 (BHN) as separate papers; the memo §3 taxonomy lists BHN (§3.6) but not BNN as a peer. Q1's job is to make the relationship explicit.

### §2.3 — Formal inclusion relations

We name where each relation is rigorous and where it is heuristic.

**Claim 1 (FiLM ⊂ Hypernetwork — *rigorous*).** FiLM is a hypernetwork whose generated "weights" are restricted to per-channel diagonal scaling plus a constant translation.

*Formal statement.* The Ha et al. 2016 HyperRNN scaling-vector trick (Ha §3.2, Eqs. 7–8) is $h_t = \phi(d_h(z_h) \odot (W_h h_{t-1}) + b(z_b))$ — element-wise multiplication of a fixed matrix-multiply by a context-generated vector, plus a context-generated bias. This is the exact structural form of FiLM's $\gamma \odot h + \beta$ at per-row granularity; the [ha_2016_hypernetworks.md §"Connection to normalization and to FiLM"](../references/FiLM/reviews/ha_2016_hypernetworks.md) section makes the collision explicit. Equivalently, the "generated weight matrix" the hypernetwork emits is constrained to $\mathrm{diag}(\gamma_\ell(c))$ — one degree of freedom per output channel — when the underlying hypernet could in principle emit a full $n \times n$ matrix.

*Independent route via Abdollahzadeh 2021.* The reinterpretation lemma ([abdollahzadeh_2021_multimodal_meta.md Eq. 5](../references/FiLM/reviews/abdollahzadeh_2021_multimodal_meta.md)) gives the same conclusion at the conv-layer level: vanilla FiLM is equivalent to scaling every entry of the corresponding conv-kernel slice by one scalar — degenerate (per-channel-uniform) per-weight kernel modulation, with KML the general per-weight case. Both are hypernets in the Ha 2016 sense; FiLM is the rank-1-per-channel restriction.

**Claim 2 (Hypernet ⊆ Bayesian-Hypernet-with-point-mass — *rigorous as a limit*).** A deterministic hypernetwork is the special case of a Bayesian Hypernetwork whose implicit posterior over $W$ is a point mass.

*Formal statement.* The Krueger 2017 BHN defines $W = h_\phi(\epsilon)$ with $\epsilon \sim \mathcal{N}(0, I_D)$ and induces a distribution $q_\phi(W) = q_\epsilon(\epsilon) |\det \partial h / \partial \epsilon|^{-1}$ ([krueger_2017_bayesian_hypernets.md §"Bayesian hypernetwork formulation"](../references/FiLM/reviews/krueger_2017_bayesian_hypernets.md)). A deterministic Ha 2016 hypernet $W = g_\phi(c)$ corresponds to the BHN limit $h_\phi(\epsilon, c) \equiv g_\phi(c)$ (constant in $\epsilon$), giving $q_\phi(W \mid c) = \delta(W - g_\phi(c))$.

*Caveat.* This is a *limiting* inclusion, not a sub-architecture obtained by restricting an output space. Calling Ha 2016 "a BHN with $q = \delta(\cdot)$" is technically correct but should not be paraphrased as "Ha 2016 is a BHN" — the operational character (no posterior uncertainty) is what distinguishes the cells of the 2×2.

**Claim 3 (BNN and BHN are orthogonal — *rigorous*).** A BNN places a posterior on the main network's weights $p(W \mid \mathcal{D})$ without context-conditioning. A BHN places a posterior on the *generator*, inducing a context-conditional $q_\phi(W \mid c)$ over main-network weights. The two place uncertainty at different levels — BNN at the leaf weights, BHN at the generator — and are not nested. They can be combined (a hierarchical architecture with both posteriors active), but no paper in the project's corpus does this.

**Claim 4 (FiLM-Ensemble ≈ finite-mixture BHN — *holds with caveat*).** Turkoglu 2022's FiLM-Ensemble produces $M$ deterministic settings of $(\gamma^m, \beta^m)$ and averages their predictions. If we treat the member index as a categorical latent with $q(\gamma, \beta) = \frac{1}{M} \sum_m \delta(\gamma - \gamma^m) \delta(\beta - \beta^m)$, the ensemble's predictive distribution is a Monte Carlo estimate of the marginal $\mathbb{E}_{q}[\sigma(f_{\theta, \gamma, \beta}(x))]$. The predictive variance decomposes via the law of total variance into a within-member (aleatoric) and between-member (epistemic) term — same structure as Kendall & Gal 2017 Eq. 7.

*Caveat.* FiLM-Ensemble is *not* a clean BHN — the members are trained jointly under one objective rather than sampled from a learned posterior $q_\phi$; there is no KL-to-prior term and the $\rho$-gain init controls spread heuristically rather than via an ELBO. Honest framing: "finite-mixture approximation of a posterior over FiLM scalars", not "a BHN".

**Claim 5 (Modularity carries through the family — *rigorous*).** Galanti & Wolf 2020 Thm. 4 proves a hypernetwork's primary network $g$ can be $O(\epsilon^{-m_1/r})$ to approximate any target in $W^{r,m}$ to error $\epsilon$, vs $\Omega(\epsilon^{-(m_1+m_2)/r})$ for embedding-concatenation — exponentially more parameters in the conditioning dimension. The bound applies to every member of the family: FiLM, BHN, FiLM-Ensemble inherit the same parameter-efficient capacity allocation. The [galanti_wolf_2020 review](../references/FiLM/reviews/galanti_wolf_2020_hypernet_modularity.md) §"Why the asymmetry?" notes that "the embedding method is a special case of a hypernetwork" — hypernetworks *strictly contain* embedding methods.

### §2.4 — Where the project's substrate sits in the 2×2

The project's current substrate ([directions v3 §3](../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md)) places FiLM at three sites (encoder A, GRU update-gate B, policy logits C) of a recurrent PPO policy, with $(\gamma, \beta)$ generated by a self-recurrent modulator that watches the agent's internal state. In the 2×2:

- **Cell occupied today.** Top-right — deterministic, context-conditional. By Claim 1 the project's FiLM substrate *is* a hypernetwork, with the modulator as generator $\phi$ and the policy/critic as primary $g$.
- **Moving down a row (toward BHN).** A posterior $q_\phi(\gamma, \beta \mid c)$ over the FiLM scalars — calibrated uncertainty in the modulator's output, which translates to a learnable "how confident is the neuromodulator in this $(\gamma, \beta)$?" signal. The math-reviewer-flagged compound in [film_neuromod_integration.md §5.4 Row EE-6](film_neuromod_integration.md) is the first step toward this cell: by Claim 4, FiLM-Ensemble + Kendall-Gal heteroscedastic precision is a finite-ensemble approximation of a BHN with a heteroscedastic likelihood. The headline framing for EE-6 is therefore **"first instantiation of an approximate BHN with a heteroscedastic likelihood for an RL policy"** — positioned cleanly against Krueger 2017 (full BHN, no heteroscedastic head, no RL) and Kendall & Gal 2017 (BNN, no context-conditioning, no RL).
- **Moving across to BNN-only.** Posterior on the *base* policy/value weights — what MC dropout in the policy network gives. This erases context-conditioning; there is no neuromodulator at all. Not a useful move for the project.
- **An ambitious diagonal step (BNN + BHN jointly).** Base-weight posterior *and* modulator posterior, trained jointly. No paper in either corpus does this. Architecturally heavy; flag as long-range, not a v4 commitment.

**Load-bearing position for v4.** The project's substrate is a deterministic Hypernetwork with FiLM-restricted output, and EE-6 upgrades it toward a finite-ensemble Bayesian Hypernetwork with a heteroscedastic likelihood. The compound's novelty survives (no paper combines FiLM-Ensemble with Kendall-Gal on an RL policy), but its framing should be "finite-ensemble approximation of a BHN with heteroscedastic likelihood" — Q1's lineage shows every component has clear ancestors.

**Gap Q1 exposes in [film_neuromod_integration.md §3](film_neuromod_integration.md).** The taxonomy lists BHN (§3.6) but not BNN as a peer; the FiLM ⊂ Hypernet ⊂ point-mass-BHN chain is not stated. Recommendation: add a §3.0 callout summarising Claim 1–Claim 5 in one paragraph each so the §5 mapping inherits a clean lineage.

---

## §3 — Q2: Where γ_Bellman modulation has to live

### §3.1 — Why GRU update-gate doesn't work (math-reviewer flag (d) recap)

The math-reviewer audit in [film_neuromod_integration.md §10 flag (d)](film_neuromod_integration.md) ruled the v8 / Row PG-2 sketch-box claim "additive β at the GRU update-gate ⇔ effective Bellman γ change" *dimensionally incoherent*. The GRU update gate $z_t = \sigma(W_z[h_{t-1}, x_t] + b_z) \in (0,1)^H$ is a per-hidden-unit retention scalar that produces the recurrent update $h_t = (1-z_t) \odot h_{t-1} + z_t \odot \tilde h_t$. Its dimensional content is *per-unit recurrent-state-retention probability per inner-network-timestep*. The Bellman discount factor in the value-function definition $V^\pi(s) = \mathbb{E}[\sum_{t=0}^\infty \gamma_{\text{Bellman}}^t r_t]$ is a *global scalar* that governs the value-function horizon, with dimensional content *temporal weighting per environment-timestep*. Modulating $z_t$ via an additive $\beta^{(B)}$ shifts the policy's *memory persistence*; it does not shift the *value-function horizon*. The two effects can correlate empirically (longer hidden-state memory tends to support longer-horizon credit assignment) but they are not the same quantity. `professor-rl-bayesian-dl` flagged this exact conflation in v1 §8 Q3, and the v3 §5.2 site-B description now correctly downgrades the claim to "effective discount on past memory ... related to $\gamma$ in the Bellman sense, distinct from the FiLM $\gamma$ symbol" — but the concept memo's §5.3 sketch-box still asserted the equivalence and must be revised.

So if the project still wants a 5-HT / γ_Bellman modulation channel (the Doya 2002 fourth knob), where does it actually go?

### §3.2 — The four candidate modulation points

For each, the formal operation and the *type* of γ-effect:

**Candidate (a) — Critic-head output FiLM.** $V_{\text{mod}}(s) = \gamma_V(c) \cdot V(s) + \beta_V(c)$ applied to the critic's scalar output (or pre-readout layer).

*Type.* **Value-scale modulation, not horizon modulation.** The Bellman discount lives inside $V^\pi(s) = \mathbb{E}[\sum_t \gamma_{\text{Bellman}}^t r_t]$; multiplying $V(s)$ after the network has computed it changes *magnitude*, not *which future rewards are integrated*. Cellular analogy: gain on a value-read-out neuron — Ferguson & Cardin 2020 multiplicative-arm-on-readout, not a horizon controller. **Cleanly forward-pass-only; fits the FiLM substrate; not γ_Bellman.**

**Candidate (b) — TD-target γ modulation.** $y_t = r_t + \gamma(c) V(s_{t+1})$, with $\gamma(c)$ a learned function of context evaluated at each transition, consumed in the critic's update.

*Type.* **Literal γ_Bellman modulation** — the Doya §3.2 5-HT prediction made operational. Precedent: White 1995 (general-value-functions allow per-state γ); Schaul et al. 2015 Universal Value Function Approximators allow $V(s, g, \gamma)$. **Architectural cost: lives in the loss / learning update, not the forward pass.** The forward pass still produces $V(s_{t+1})$; the critic's loss now consumes a learned γ. Not a FiLM operator — no per-channel affine on activations. **Leaves the forward-pass FiLM substrate.**

**Candidate (c) — GAE λ modulation.** $\hat A_t = \sum_{k=0}^\infty (\gamma \lambda)^k \delta_{t+k}$ with $\lambda(c)$ a learned eligibility-trace decay (Schulman et al. 2016).

*Type.* **Effective-n-step-horizon at advantage-computation.** $\lambda \to 0$ is TD(0) (short effective horizon); $\lambda \to 1$ is Monte Carlo (full horizon). Modulating $\lambda$ is "how many real future steps to credit"; modulating γ_Bellman is "how far into the predicted future to value rewards". Related but distinct. Lives at the loss / advantage layer, not the forward pass.

**Candidate (d) — Learned discount network $\hat\gamma(c)$.** A small side-network on the modulator produces $\hat\gamma(c) \in [0, 1)$ consumed wherever the algorithm uses a discount (TD target, GAE, value definition).

*Type.* Same consumer as (b), but explicit about *how* γ is produced — γ_Bellman as a side-signal-conditioned scalar with its own generator, exactly the "neuromodulator as side-signal" template. Gradient: $\partial \mathcal{L}_V / \partial \hat\gamma(c) = 2 \delta_t \cdot V(s_{t+1}) \cdot \partial \hat\gamma / \partial c$, differentiable end-to-end. **The cleanest architectural expression of γ_Bellman modulation; still leaves the FiLM substrate.**

### §3.3 — What the literature has actually done

**Lee 2024 explicitly declines to instantiate γ_Bellman.** [Lee 2024 §6 limitations](../references/neuromodulatory_algorithms/reviews/lee_2024_lifelong_rl.md): "the serotonin mapping is dropped because no convergent normative theory exists. So the 'framework' actually only instantiates 2 of Doya's 4 mappings." Lee 2024's Doya-DaYu agent instantiates ACh/α and NA/β with closed-form $\alpha = E/(E+A)$ and $\beta^{-1} = 1/\langle E\rangle$; γ stays fixed at 0.95 in the bandit experiments. The acknowledged reason is exactly Q2's: there is no agreed agent-environment statistic whose reciprocal gives γ in the way $1/\langle E\rangle$ gives β.

**Doya 2002 only predicts the 5-HT/γ mapping; never implements it.** The mechanistic substrate Doya proposes (5-HT modulates direct vs indirect basal-ganglia pathway balance, setting effective γ via the $(1-\gamma)V(s)$ coefficient in his alternative TD-error form, Eq. 13) is a *circuit-level* prediction, not an algorithm. No deep-RL paper in the corpus implements it.

**Xing 2022, Wang 2024, Rodriguez-Garcia 2026 all leave γ fixed.** Xing's ACh+NE for context-change detection is forward-pass-only (per-task confidence vector, novelty threshold) — no γ modulation. Wang's NeuronML structure mask gates which weights are active, not the value-function horizon. Rodriguez-Garcia is the canonical example of leaving the forward-pass substrate for a hyperparameter channel — but the channel they leave for is the *learning rate*, via gain-amplified gradient $\partial \mathcal{L}/\partial w = g \cdot \partial \mathcal{L}/\partial W_{\text{eff}}$, not γ_Bellman. The structural analogy "γ_Bellman modulation will require something Rodriguez-Garcia-shaped" holds — both put the modulator's output in the update equation, not the forward pass.

**Summary.** No paper in either corpus implements γ_Bellman modulation. The literature stops at γ's neighbours — α (Xing, Wang, Lee), β (Lee, Doya, Vecoven). The user's intuition that "this is hard" is the literature's verdict, not an idiosyncratic gap. A consequence: a deep-RL implementation of γ_Bellman modulation would itself be a contribution — but it must use the right architectural choice (b or d, not the GRU-as-γ_Bellman conflation flagged in v8).

### §3.4 — Forward-pass-only proxies for γ_Bellman

Two proxies — both *value-scale*, not horizon:

**Proxy P1: Critic-feature gain.** FiLM at the critic's pre-readout layer, $h_V \mapsto \gamma_V(c) \odot h_V + \beta_V(c)$. If $\gamma_V(c)$ amplifies long-horizon-relevant features when the modulator predicts a high-γ regime, the bootstrap target $V(s_{t+1})$ scales differently across regimes — but the actual discount in $y_t = r_t + \gamma V(s_{t+1})$ is unchanged. Credit-assignment depth unaffected. Closest analogy to Doya 2002 Eq. 13's "5-HT shifts the $(1-\gamma)V(s)$ coefficient" — but the project does not use Doya's alternative TD-error form, so the analogy is loose.

**Proxy P2: Modulator-conditioned re-weighting of multi-step returns in the critic loss.** A FiLM-like gate $\alpha_n(c)$ on $n$-step-return contributions. This *does* shift effective bootstrapping depth, but it lives in the loss — i.e., Candidate (c) in disguise. Not pure forward-pass.

**Honest naming.** If the project insists on forward-pass-only, P1 is the best one can do; its mapping to 5-HT must be called *suggestive*, not formal. v4 should not claim the substrate "instantiates" Doya §3.2's 5-HT/γ mapping.

### §3.5 — The architectural decision

Three options for v4:

**Option A — Drop the 5-HT/γ_Bellman channel from substrate claims.** v4 restricts its Doya-channel mapping to **two cleanly-instantiable channels**: ACh/α at site A (encoder) and NA/β at site C (policy logits). γ_Bellman is acknowledged as out-of-substrate — the same position Lee 2024 explicitly takes, and consistent with [v3 §5.6 limitations](../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md) ("the 5-HT / discount-factor branch is not testable: the grid-world does not natively dissociate immediate from delayed reward. This is *fine* — the corpus already calls 5-HT the weakest branch"). Pros: mathematical honesty; substrate uniformity; aligned with Lee 2024 and v3's existing stance; math-reviewer flag (d) revision becomes surgical. Cons: the framing shifts from "four-channel Doya mapping" to "two-channel Doya mapping with two acknowledged out-of-substrate boundaries" — actually *closer* to the math-reviewer's spirit, since the no-map cases in [film_neuromod_integration.md §5.5](film_neuromod_integration.md) draw substrate boundaries honestly.

**Option B — Keep 5-HT/γ_Bellman as a hybrid extension.** v4 commits to FiLM at A/B/C plus a learned-discount head (Candidate d) feeding the TD target. The modulator gets four arms: $(\gamma_A, \beta_A)$, $(\gamma_C, \beta_C)$, an additive $\beta^{(B)}$ at the GRU update gate (re-named *plasticity gating* per flag (d), not γ_Bellman), and a scalar $\hat\gamma(c)$ feeding the loss. Pros: four-channel Doya mapping survives; first deep-RL γ_Bellman implementation. Cons: breaks substrate uniformity (one knob isn't FiLM); doubles v4 implementation effort; structurally weak gradient signal in the current grid-world testbed (exactly v3 §5.6's caveat); needs experiment-designer to construct an immediate-vs-delayed-reward variant.

**Option C — Forward-pass-only γ_proxy.** Keep everything FiLM; add critic-feature gain at site D as Proxy P1, named *γ_proxy* (not γ_Bellman), mapping to 5-HT called *suggestive*. Pros: substrate uniformity preserved; modest extension. Cons: a careful reviewer will (correctly) point out that γ_proxy is value-scale, not horizon — the 5-HT analogy is structurally weaker than α and β; weakens v3 §3's unification claim.

**Recommendation: Option A.** The math is clean, the precedent (Lee 2024) supports it, the architecture stays uniform, and v3 already concedes 5-HT is the weakest branch. v4 ships with **two cleanly-instantiated Doya channels** (α at A, β at C), **B re-named as plasticity gating** rather than effective discount (per flag (d)), and **5-HT/γ_Bellman acknowledged as a fourth no-map row** in [film_neuromod_integration.md §5.5](film_neuromod_integration.md) — joining Rodriguez-Garcia 2026 (gradient-level), Wainstein 2025 (internal-RNN-gain), Osman 2024 (Hopfield-attractor). Option B is the natural v5 follow-up if the project later wants the γ channel. Option C is not recommended: the value-scale-as-horizon sleight-of-hand is exactly the kind of conflation math-reviewer flagged as (d) in v8.

---

## §4 — Connection between Q1 and Q2

Q1's lineage tells us *which family the project's substrate sits in*: a deterministic Hypernetwork with FiLM-restricted output, upgraded toward a finite-ensemble approximate Bayesian Hypernetwork with a heteroscedastic likelihood by the EE-6 compound formula. Q2's architectural choice tells us *whether the family can carry all four Doya channels uniformly*. The answer is no — and Q1's framework makes the no precise.

**The load-bearing finding for v4.** The project's substrate is a (FiLM ⊂ Hypernet) ⊂ deterministic-BHN-limit architecture that elegantly carries α (forward-pass FiLM on encoder activations) and β (forward-pass FiLM on policy logits) but cannot carry a clean γ_Bellman channel without architectural extension into either the loss-computation layer (Candidate b/d) or a gradient-level architecture parallel to Rodriguez-Garcia 2026. **A clean γ_Bellman modulation requires leaving the forward-pass FiLM substrate.** The honest options are: (i) drop the γ channel from the substrate's claims (Option A — recommended); (ii) keep it as a named hybrid extension and absorb the additional architectural complexity (Option B — defer to v5); (iii) name the closest forward-pass proxy as a value-scale knob rather than γ_Bellman (Option C — not recommended).

This is not a failure of the FiLM substrate. It is a *clarification of its boundary*. The substrate carries the Doya channels whose mathematical content is *per-feature affine on activations*; it does not carry the channels whose mathematical content is *scalar inside a temporal sum*. ACh/α and NA/β are the former; 5-HT/γ is the latter. Lee 2024 hit this boundary first and dropped the 5-HT branch for exactly this reason. v4 should adopt Lee 2024's stance, name the boundary explicitly in §5.5 of the concept memo (extending it from three to four no-map rows), and re-anchor the unification claim on the two channels the substrate genuinely supports.

---

## §5 — Recommendations for v4 of the direction memo

Concrete updates v4 should absorb:

**v3 §2 (lineages) and §3 (unification claim).** Update the unification to: "FiLM on the policy's activations carries two of Doya's four channels — α (encoder, ACh) and β (policy logits, NA). A third (plasticity gating at the GRU update gate) is re-named from v1/v3's mis-cited 'effective discount' framing per math-reviewer flag (d). The fourth (5-HT/γ_Bellman) lies outside the forward-pass FiLM substrate and is treated as a substrate boundary, joining the no-map rows in [concept memo §5.5](../concepts/film_neuromod_integration.md)." Cite Q1's lineage (Claims 1–5) to position the substrate as a deterministic Hypernetwork with FiLM-restricted output.

**v3 §5.3 (plasticity gating prediction).** Rewrite per math-reviewer flag (d): "Additive β at the GRU update gate shifts per-unit hidden-state retention $\tau_h$, measurable as recurrent-state autocorrelation. This is *not* a Bellman discount modulation; it is per-unit memory-persistence control. γ_Bellman lives at the TD-target layer and would require Option B's hybrid extension, which v4 does not commit to."

**v3 §5.4 (4-arm clamp falsifier).** No change. The test was always organised around α and β; the 5-HT/γ channel was never on the experimental critical path.

**v3 §7 (gaps).** Replace gap §7.6 with: "No paper has put an approximate Bayesian Hypernetwork (FiLM-Ensemble) with a heteroscedastic likelihood (Kendall-Gal) on the modulator output of an RL policy. Q1's lineage shows each component has clear ancestors; their combination is new."

**Concept memo [film_neuromod_integration.md §3](film_neuromod_integration.md).** Add a §3.0 callout summarising Claims 1–5 (FiLM ⊂ Hypernet; Hypernet ⊆ point-mass BHN; BNN ⊥ BHN; FiLM-Ensemble ≈ finite-mixture BHN; modularity carries through) — closes the §3 taxonomy gap Q1 exposed without rewriting §3.1–§3.12.

**Concept memo [§5.5 no-map boundary](film_neuromod_integration.md).** Add a fourth no-map row — *Doya 2002 §3.2 5-HT/γ_Bellman* — with the explanation: "γ_Bellman is a scalar inside a temporal sum, not a per-feature affine on activations. The forward-pass FiLM substrate does not produce this quantity. Lee 2024 hit the same boundary and dropped the 5-HT branch for the same reason. The clean architectural home for γ_Bellman is the TD-target computation (Q2 §3.2 Candidates b/d), at the loss layer, not the FiLM substrate."

---

## §6 — References

### FiLM / hypernet / BNN corpus (used in §2)

- **Perez et al. 2018** ([perez_2018_film.md](../references/FiLM/reviews/perez_2018_film.md)) — vanilla FiLM, the per-channel affine on activations.
- **Ha, Dai & Le 2016** ([ha_2016_hypernetworks.md](../references/FiLM/reviews/ha_2016_hypernetworks.md)) — static and dynamic hypernetworks; §3.2 scaling-vector trick is "FiLM at per-row granularity, two years before FiLM was named". Used for Claim 1.
- **Krueger et al. 2017** ([krueger_2017_bayesian_hypernets.md](../references/FiLM/reviews/krueger_2017_bayesian_hypernets.md)) — Bayesian hypernetwork via normalising flow; §2.2 states FiLM/CBN/CIN as special cases of hypernets. Used for Claim 2 and Claim 3.
- **Kendall & Gal 2017** ([kendall_gal_2017_uncertainties.md](../references/FiLM/reviews/kendall_gal_2017_uncertainties.md)) — heteroscedastic regression loss + MC-dropout BNN; Eq. 6 is the project's anchor heteroscedastic loss. Used for Claim 3 (BNN definition) and Claim 4 (Row EE-6 compound).
- **Galanti & Wolf 2020** ([galanti_wolf_2020_hypernet_modularity.md](../references/FiLM/reviews/galanti_wolf_2020_hypernet_modularity.md)) — modularity bound (Thm. 4: $N_g = O(\epsilon^{-m_1/r})$ for hypernets vs $\Omega(\epsilon^{-(m_1+m_2)})$ for embeddings); §"Why the asymmetry?" states "embedding method is a special case of a hypernetwork". Used for Claim 1 and Claim 5.
- **Turkoglu et al. 2022** ([turkoglu_2022_film_ensemble.md](../references/FiLM/reviews/turkoglu_2022_film_ensemble.md)) — FiLM-Ensemble; Eq. 8 predictive mean / variance decomposition. Used for Claim 4.
- **Abdollahzadeh et al. 2021** ([abdollahzadeh_2021_multimodal_meta.md](../references/FiLM/reviews/abdollahzadeh_2021_multimodal_meta.md)) — reinterpretation lemma (FiLM ≡ per-output-channel-uniform kernel modulation); KML as the generalisation. Used as independent confirmation of Claim 1.

### Neuromodulation corpus (used in §3)

- **Doya 2002** ([doya_2002_metalearning_neuromodulation.md](../references/neuromodulatory_algorithms/reviews/doya_2002_metalearning_neuromodulation.md)) — §3.2 5-HT/γ mapping; §3.3 NA/β; §3.4 ACh/α; Eq. 13 alternative TD-error form with $(1-\gamma)V(s)$ term; Fig. 9 modulator-interaction algebra. The foundational mapping; never implemented at the algorithmic level by Doya himself.
- **Lee et al. 2024** ([lee_2024_lifelong_rl.md](../references/neuromodulatory_algorithms/reviews/lee_2024_lifelong_rl.md)) — Doya-DaYu agent; §4 Eq. 4 hand-coded $\alpha, \beta^{-1}$ functional forms; §6 limitations explicitly drops the 5-HT/γ branch because "no convergent normative theory exists". Closest published precedent that names γ_Bellman and refuses to instantiate it.
- **Xing et al. 2022** ([xing_2022_neuromodulation_rl_environment_changes.md](../references/neuromodulatory_algorithms/reviews/xing_2022_neuromodulation_rl_environment_changes.md)) — ACh/NE for environment-change detection; fixed γ, no γ_Bellman modulation.
- **Wang et al. 2024 NeuronML** ([wang_2024_neuromod_meta.md](../references/neuromodulatory_algorithms/reviews/wang_2024_neuromod_meta.md)) — per-task structure mask; fixed γ; bi-level MAML.
- **Rodriguez-Garcia et al. 2026** ([rodriguezgarcia_2026_ne_stability_gap.md](../references/neuromodulatory_algorithms/reviews/rodriguezgarcia_2026_ne_stability_gap.md)) — gradient-level gain modulation; canonical example of "leaving the forward-pass substrate for a hyperparameter channel" (the channel they leave for is α, not γ). Architectural template for what Option B's hybrid extension would look like.

### Project anchors

- **Concept memo (under audit)**: [film_neuromod_integration.md](film_neuromod_integration.md) — §3 taxonomy gap addressed by Claim 1–Claim 5; §5.3 plasticity sketch-box revised per Q2 §3.1; §5.5 no-map boundary extended to a fourth row per Q2 §3.5.
- **Math-reviewer audit**: [film_neuromod_integration.md §10](film_neuromod_integration.md) flag (d) (GRU-as-Bellman-γ conflation), flag (c) (Hessian-direction reversal — out of Q1/Q2 scope), flag (a) (Abdollahzadeh lemma — addressed by Claim 1's independent confirmation), flag (b) (EE-6 compound formula — addressed by Claim 4's BHN-finite-ensemble framing).
- **Direction memo v3** (superseded target): [20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md](../directions/20260516_nmn_scaling_shifting_as_hyperparameter_modulation_v3.md) — §2 lineages, §3 unification, §5.2 per-site predictions (site B re-named per math-reviewer flag (d)), §5.6 limitations (5-HT already named as weakest branch in v3 — Option A's position is continuous with v3's).
- **Prior `professor-rl-bayesian-dl` warning**: v1 §8 Q3 (the same GRU/γ conflation, raised at v1 and re-flagged by the math-reviewer at v8) — referenced in the user's brief.

### Next steps by agent

- **`research-postdoc`**: revise [film_neuromod_integration.md §3](film_neuromod_integration.md) to add Claim 1–Claim 5 as §3.0 (or a callout box at §3 top); revise §5.3 sketch-box per math-reviewer flag (d); extend §5.5 no-map boundary to four rows per Q2 §3.5; cite this memo as the audit follow-up.
- **v4 direction memo author** (postdoc, with `professor-rl-bayesian-dl` and `professor-neuromodulation` review): adopt Option A; rewrite §2/§3 unification claim as "two of four Doya channels cleanly carried, two acknowledged as out-of-substrate boundaries"; preserve the four-arm clamp falsifier (§5.4); add a §5.5-style substrate-decision paragraph naming the 5-HT/γ_Bellman channel as out-of-substrate.
- **`experiment-designer`**: no action required for Option A. If the user / PI overrides toward Option B (v5), the experiment program needs an immediate-vs-delayed-reward gridworld variant that exercises γ-modulation — out of scope for v4.
- **`senior-developer`**: no action required for Option A. The math-reviewer's required revisions are surgical (rewrite §5.3 sketch-box, fix §6.3 Hessian direction); none require code changes. Option B (deferred to v5) would require a new learned-discount head on the modulator; not committed.
- **`math-reviewer`**: should re-audit [film_neuromod_integration.md](film_neuromod_integration.md) after the postdoc's revisions land. Specifically, check (a) Claim 1's "FiLM ⊂ Hypernet" derivation in the new §3.0 against Ha 2016 §3.2 and Galanti-Wolf 2020 §"Why the asymmetry?"; (b) Claim 4's BHN-finite-ensemble framing of Row EE-6 against the audit's sub-check 4 on gradient flow; (c) the §5.5 extension to a fourth no-map row.

---

*Author: `professor-rl-bayesian-dl`, 2026-05-16.*
