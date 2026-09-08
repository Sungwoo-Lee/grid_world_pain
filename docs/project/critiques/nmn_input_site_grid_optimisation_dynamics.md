# Critique: what the optimisation dynamics of the modulation-site grid do and do not tell us

> One-line summary: the gain-below-one / offset-below-zero pattern is the canonical FiLM outcome and its *time-averaged* part is a pure re-parameterisation the modulated layer can absorb — so "did the offsets converge" is the wrong question; the right question is whether units are dormant and whether the modulator's output still varies with context, and both are answerable today from the saved checkpoints without a new run. Of the five observations, one is a genuine open concern (possible unit dormancy), two are measurement gaps, and two are normal.

**Author**: professor-rl · 2026-09-08
**Reviews**: the training-health audit of the two sixteen-run modulation-site grids ([[TRAINING_HEALTH_AUDIT]] under `docs/experiments/active/nmn_input_site_grid/`), read against the trainer and network code the runs actually used. This memo does not touch the audit, the two design documents, `src/`, `configs/` or `scripts/`.

---

## Verdict (plain language)

Thirty-two training runs were audited for whether the learning machinery worked. They did. This memo answers the follow-on question the audit could not: what do the numbers say about *how* the agent learned, and which of the odd-looking numbers should worry a reader.

The setup, in one paragraph: a small side network — the "neuromodulator" — reads the agent's senses every step and emits, for up to four places in the main network, a per-unit multiplier (a gain) and a per-unit shift (an offset). Every gain starts at 1 and every offset at 0, so at the start the side network changes nothing. By the end of ten million episodes, the gains had fallen to roughly 0.2–0.9 and the offsets had gone negative, to as far as −1.8; two of the offsets were still sliding when training stopped; the side network received *more* gradient late in training under one learning-signal recipe but not the other; and the runs whose side network read only the two body signals had markedly quieter gradients overall.

**The reading.** Gains well below 1 with negative offsets is exactly what the original FiLM paper reports its modulators learning, and most of it is a bookkeeping convention rather than a behaviour: the *average* gain and offset at a site can be absorbed into the modulated layer's own weights without changing what the network computes, so their drift is not something that has to converge. What *cannot* be absorbed — and what nothing currently logged measures — is (a) whether the shift has pushed a fraction of units permanently below their rectifier threshold (a known deep-RL pathology called dormancy), and (b) whether the side network's output still *varies with context* or has settled into a fixed re-tuning. Both can be measured from the checkpoints already on disk. The estimator difference in the side network's gradient is real, is not explained by the critic's error size (which is flat in every run), and is currently unexplained; two log additions would settle it. The gradient clip did its job and is not interacting with the modulator in any known harmful way.

---

## 1. What is actually being optimised — the equations the audit's numbers refer to

Everything below was read from the code the runs used (`src/models/recurrent_ppo_network.py`, `src/models/neuromodulator.py`, `src/models/recurrent_ppo_trainer.py`, `train.py`) and from one run's trainer-written config. Symbols: `t` indexes (timestep, environment), `i` indexes hidden units (128 per site), ⊙ is elementwise product.

### 1.1 The modulator's output

For every enabled site the modulator's gain and offset are

$$
\gamma_{it} = g_i + c_i + w_i^{\top} h^{m}_t , \qquad
\beta_{it} = g'_i + c'_i + w'^{\top}_i h^{m}_t ,
$$

where `h^m_t` is the 16-unit GRU state of the modulator (driven by the observation slice), `w_i, w'_i` are rows of the linear heads, `c_i, c'_i` the head biases (initialised to 1 and 0), and `g_i, g'_i` are **learned per-unit baselines** (`z_<site>_baseline`, initialised to 0). The head weight matrices are *not* zero-initialised, which is why the audit's §8.1 finds a per-unit spread of ±0.3 at step 0 — the corpus's "zero-init prophylactic" (Beck et al. 2023 Bias-HyperInit; AdaLN-Zero) is half-applied here, exactly as the recent-variants survey suspected ([film_rl_recent_variants_survey.md §5.5](../references/FiLM/film_rl_recent_variants_survey.md)).

Note the decomposition this gives for free: the **constant** part of the modulation is `g_i + c_i + w_i^⊤ E[h^m]`, the **contextual** part is `w_i^⊤ (h^m_t − E[h^m])`. The logged `gamma_<site>_mean` is the sum of both averaged over units; the logged `_std` pools unit-to-unit and time-to-time spread. No logged quantity isolates the contextual part.

### 1.2 Where the modulation is applied

Encoder (both stages), with `use_layer_norm: true`, so LayerNorm with its own learnable scale `s` and bias `c` sits *before* FiLM, and a rectifier after:

$$
h_t = \mathrm{ReLU}\!\big(\gamma_t \odot (s \odot \hat z_t + c) + \beta_t\big).
$$

Actor and critic hidden layers (`activation: relu` in these configs), FiLM on the pre-activation:

$$
a_t = \mathrm{ReLU}\!\big(\gamma_t \odot (W_1 x_t + b_1) + \beta_t\big), \qquad \text{logits or } V = W_2 a_t + b_2 .
$$

Recurrent site: the GRU's *emitted* output is transformed with no nonlinearity, `x_h ← γ_t ⊙ x_h + β_t`, and then read by the actor's and critic's first linear layers.

### 1.3 The loss and the two return modes

$$
\mathcal L = -\,\mathbb E\!\left[\min\!\big(r_t \hat A_t,\ \mathrm{clip}(r_t, 1\pm\epsilon)\hat A_t\big)\right]
 + \tfrac{1}{2} v_f\, \mathbb E\!\left[(V_t - y_t)^2\right] - c_H\, \mathbb E[H_t],
$$

with `ε = 0.1`, `v_f = 0.5`, `c_H = 0.01`, four epochs per rollout, and the optimiser `clip_by_global_norm(0.5) → Adam(5e-4)` with no weight decay. The two grids differ only in how `y_t` and `Â_t` are built:

- **MC**: `y_t = z(G_t)` — the Monte-Carlo return within the 128-step window, z-scored over the whole batch; `Â_t = y_t − V_t`.
- **GAE_NORM**: `y_t = z(A^{GAE}_t + V_t)` with `λ = 0.95`; `Â_t = y_t − V_t`.

In **both** modes the advantage is the critic's residual against a z-scored target. This is not the mainstream convention (the project's own survey adjudicates it: [ppo_return_normalization_survey.md](../references/modulation_in_rl/ppo_return_normalization_survey.md)), and the attempt to replace it lost badly at 1M episodes ([[mc_return_units_bug_severity_and_repair]] §10). For this memo the consequence that matters is simple: **the scalar that weights every gradient — policy term and value term alike — is the same residual `y_t − V_t`**, whose root-mean-square is `√(2 × value loss)`.

---

## 2. Observation 1 — gains below 1, offsets below 0, and two offsets still sliding

### 2.1 Is this a recognised failure mode? No — it is the canonical FiLM outcome

The FiLM paper itself reports, for its learned modulators, "a sharp peak at zero" in the gain histogram (whole channels turned off), 36% of gains negative, and offsets "mostly negative" (Perez et al. 2018 §4.2, as extracted in the project's own review, [perez_2018_film.md](../references/FiLM/reviews/perez_2018_film.md)). A FiLM layer in front of a rectifier learns to *gate*: a small gain plus a negative offset is the mechanism by which it silences a feature in some contexts and admits it in others. So the direction of drift is not evidence of pathology. It is what the operator is for.

### 2.2 Why the time-averaged gain and offset are not identifiable — the gauge identity

Take the actor site and write the modulation as its time-average plus a deviation, `γ_t = γ̄ + δγ_t`, `β_t = β̄ + δβ_t`. For any `γ̄` with no zero entries define

$$
\tilde W_1 = \mathrm{diag}(\bar\gamma)\, W_1, \qquad
\tilde b_1 = \bar\gamma \odot b_1 + \bar\beta, \qquad
\tilde\gamma_t = \gamma_t \oslash \bar\gamma, \qquad
\tilde\beta_t = \beta_t - (\gamma_t \oslash \bar\gamma)\odot \bar\beta .
$$

Then, identically in `x`,

$$
\gamma_t \odot (W_1 x + b_1) + \beta_t \;=\; \tilde\gamma_t \odot (\tilde W_1 x + \tilde b_1) + \tilde\beta_t ,
$$

and the transformed modulator has time-average gain 1 and time-average offset 0. The same identity holds at the encoder with `(s, c)` of the LayerNorm playing the role of `(W_1, b_1)`, and at the recurrent site with the two downstream linear layers absorbing the constants. **The logged means `gamma_<site>_mean` and `beta_<site>_mean` are therefore gauge quantities**: the network's function is invariant to moving them, provided the modulated layer's own affine parameters move in compensation. Only the contextual deviations `(δγ_t, δβ_t)` — and, separately, the *sign* structure and any exact zeros of `γ̄` — are identifiable from behaviour.

Three consequences follow, and they answer the user's three sub-questions.

**(a) "Is a gain settling well below 1 with a negative offset a failure mode?"** Not by itself. Under the gauge above, `(γ̄, β̄) = (0.5, −1.8)` with `(W_1, b_1)` is the same network as `(1, 0)` with `(0.5 W_1, 0.5 b_1 − 1.8)`. What the pair *does* determine, jointly with `b_1`, is the effective threshold `γ̄_i b_{1,i} + β̄_i` of unit `i`, which the logs cannot see because `b_1` is not logged. This is the precise form of the audit's §8.2 caveat.

**(b) "What does it mean that two offsets had not converged after ten million episodes?"** That the loss is flat along the direction they are moving in. Under gradient flow, scale symmetries of this kind come with a conserved quantity (for a ReLU layer, the difference of squared norms of the two sides of the cut — Du et al. 2018; the general statement is Kunin et al. 2021, "Neural Mechanics"). **Adam breaks the conservation law**: its per-parameter normalisation moves every parameter by roughly the learning rate per update regardless of the gradient's size, so a parameter with a persistent-sign but tiny gradient — which is what a near-flat direction produces — drifts at a rate set by the learning rate, not by the loss. There is no weight decay in this optimiser chain to pin it. So the offsets sliding at 10M episodes is a *budget-independent* property of this parameterisation under this optimiser: they do not need to converge for the policy to have converged, and giving them another 10M episodes would not make them stop. **This is reasoning from theory; the checkable prediction is in §2.4.**

**(c) "Is it an ill-conditioned parameterisation?"** Partly, and the ill-conditioning is in the *effective learning rate*, not the loss. Adam's step on a parameter of magnitude `m` is a relative step of order `lr / m`. As `γ̄` falls toward 0.2 while `W_2` (or the LN scale) grows to compensate, the gain becomes the highest-relative-learning-rate parameter in the layer and the compensating downstream weights the lowest. That reallocation is harmless in itself, but it means the modulator's *contextual* channel is also being trained on an inflated relative step late in training. The literature on this reallocation is van Laarhoven 2017 and Kunin et al. 2021; neither is in the project's reference corpus (§10).

### 2.3 The "exactly 11%" coincidence

The action-head offset moved by 11% of its final magnitude over the last tenth of training in both grids. The absolute rates differ (−0.197 vs −0.118) in the same ratio as the final magnitudes (−1.83 vs −1.11). A *proportional* rate is what a multiplicative feedback produces, whereas Adam's scale-free step along a persistent-sign direction predicts a roughly *constant absolute* rate. So this is mildly informative: it points away from free drift and toward a coupled process — the most economical candidate being the gauge coupling itself, in which the modulator's offset and the layer's own bias `b_1` move together and the rate scales with the current magnitudes. It could also be shared-seed coincidence; two grids at one seed cannot tell these apart. It is not a health concern under either reading.

### 2.4 The U-shape at the encoder — what early versus late modulation might be

The encoder gains fall, bottom out at 25–62% of the run, then climb back; the recurrent and critic gains plateau; the actor gain keeps creeping down. Nothing logged distinguishes the readings below, and I list them as candidates, not findings.

- **Early phase, all sites.** The gradient on a gain is `∂L/∂γ_i = Σ_t δ_{it} a_{it}` (with `δ` the backpropagated error and `a` the pre-FiLM activation), while the gradient on an offset is `Σ_t δ_{it}`. At the encoder, LayerNorm pins `|a| = O(1)`, so the multiplicative channel receives a gradient of the same order as the additive one from step 1, and Adam moves both at learning-rate speed. Early in training the policy is bad and the residual is large, so this is the phase in which the modulator can most cheaply reduce the loss by *attenuating* channels — the classic early "suppress what is not yet useful" FiLM behaviour.
- **Late phase, encoder only.** The climb-back coincides with the LayerNorm affine and the downstream hub having had time to absorb the constant part (gauge motion), after which the net pressure on `γ̄_uni` reverses. Under this reading the U is a gauge artefact, not a change in what the modulator does. *Test:* across the 50 saved checkpoints per run, the LayerNorm scale `s` and the multimodal-hub input weights should grow while `γ̄_uni` recovers, and the post-FiLM active fraction (§3) should stay flat.
- **Alternative.** The recovery is a genuine re-admission of channels as the policy improves and previously uninformative sensory channels become informative. *Test:* the active fraction rises with `γ̄_uni`, and the per-unit time-average activity redistributes.
- **Why the actor gain does not turn around.** The actor's hidden layer is the only modulated layer that feeds an entropy-regularised output. Shrinking that layer's activations shrinks the logits toward the output bias, which raises entropy; the entropy bonus is a persistent pressure that never vanishes at convergence. The critic's gain plateaus and the actor's does not, which is consistent with this — but the effect is confounded with the gauge drift and I would not put weight on it without the parameter-norm series.

---

## 3. Observation 2 — the ambiguity, and what measurement resolves it

### 3.1 Is the post-FiLM active-unit fraction the right diagnostic? Yes, with three refinements

The concern is that `γ̄ = 0.2, β̄ = −1.05` in front of a rectifier silences a large fraction of units. The corpus already names this as an untested confound: "a shrinking γ silences units the same way a dying ReLU does — and no paper has looked" ([film_in_rl_survey.md §7, item 2](../references/FiLM/film_in_rl_survey.md)), with the deep-RL dormancy literature behind it (Sokar et al. 2023; Abbas et al. 2023; Lyle et al. 2022–24, all held under `docs/project/references/continual_learning/`). The active fraction is the right first number. Three refinements make it decisive rather than suggestive:

1. **Measure per unit over time, not per layer per iteration.** The pooled active fraction cannot distinguish "every unit is on 30% of the time" (contextual gating — the intended operation) from "30% of units are on all the time and 70% never" (dormancy). Compute, for each unit, `p_i = P_t(pre-activation_{it} > 0)` over a rollout and report the *distribution* of `p_i`: the fraction with `p_i < 0.01` is the dead population; the mass in `(0.05, 0.95)` is the contextually gated population.
2. **Use Sokar's dormancy score alongside it**, because a unit can be positive yet negligible:

$$
s_i = \frac{\mathbb E_t\,|a_{it}|}{\tfrac{1}{H}\sum_k \mathbb E_t\,|a_{kt}|}, \qquad \text{unit } i \text{ is } \tau\text{-dormant if } s_i \le \tau
$$

   (Sokar et al. 2023 use `τ = 0.025`). This is the field's standard object and lets the result be compared with published dormancy rates.
3. **Compare against the unmodulated control's own layer at the same checkpoint.** A ReLU actor hidden layer in a converged PPO agent already carries dead units; the question is whether modulation *adds* dormancy, so the control's `p_i` distribution is the null.

The ratchet argument for why this is the one real concern: a unit that is off in every context receives zero gradient from its layer, so its own `γ_i, β_i` rows and its downstream weights freeze. Under FiLM the unit can only be revived by drift of the shared modulator state `h^m_t`, not by its own gradient. Dormancy is therefore an absorbing state, and a *still-sliding* mean offset is compatible with a still-growing dead population. The audit's decision to rank this metric first is right; the refinement is that the mean active fraction alone would not have settled it.

### 3.2 Separating contextual modulation from a degenerate constant re-parameterisation

The user's statement of the gap is exact: a perfectly constant modulator and a wildly context-varying one produce identical pooled means and standard deviations. The decomposition that separates them is the standard one-way variance split over the (unit × time) table of gains from a rollout:

$$
\underbrace{\mathrm{Var}_{i,t}[\gamma_{it}]}_{\text{logged } \_\text{std}^2}
= \underbrace{\mathrm{Var}_i\!\big(\mathbb E_t[\gamma_{it}]\big)}_{\text{static per-unit re-tuning}}
+ \underbrace{\mathbb E_i\!\big(\mathrm{Var}_t[\gamma_{it}]\big)}_{\text{contextual modulation}} ,
\qquad
\rho_\gamma = \frac{\mathbb E_i \mathrm{Var}_t[\gamma_{it}]}{\mathrm{Var}_{i,t}[\gamma_{it}]} .
$$

`ρ_γ` (and `ρ_β`) is the **contextual fraction**. A modulator that has degenerated into a fixed re-parameterisation has `ρ → 0` regardless of how far its means have travelled. This is the audit's §9 item 2 in one number, and it is what the design's own "engagement" criterion (`gamma_<site>_std` leaving zero, [[NMN_INPUT_SITE_GRID]] §4.3) should have been — the pooled standard deviation leaving zero is satisfied by the static term alone.

Two further measurements turn "varies with context" into "varies with the *right* context":

- **Explained variance of `γ_it` by the body state.** Regress the unit-averaged `γ̄_t` (and each unit's `γ_it`) on satiation and interoceptive nociception over a rollout; the `R²` is the direct test of the design's H5 ("the modulator must read the body") at the representational level, and it is what the design's §6.1 request for per-step modulator columns in the trajectory store is *for*.
- **The freeze-at-mean counterfactual.** Run the saved policy with `γ_t, β_t` replaced by their rollout time-averages `(γ̄, β̄)` (which, by §2.2, is the same as running it with the modulator removed and the constants folded into the layer) and measure survival and the pre-registered behavioural measures. If nothing changes, the modulator is functionally static *whatever `ρ` says*, because a contextual variation the policy does not use is not modulation. This is the definitive test and it needs no training.

### 3.3 All of this is computable now, from disk, with no new runs

Every run saved a checkpoint every 200,000 episodes — 50 per run, 1,600 across the two grids. The network's forward pass already returns `mod_info`; the evaluation script discards it (`scripts/eval/eval_rollout.py:317` binds it to `_mod_info`), and the trajectory collector drops it too (the design's §6.1 says so). A short replay script that loads a checkpoint, rolls out a few hundred episodes, and records `γ_it, β_it` and the post-FiLM pre-activations gives, per checkpoint: the `p_i` distribution, the dormancy score, `ρ_γ, ρ_β`, the body-state `R²`, the freeze-at-mean counterfactual, and — from the parameters alone — the norms of `W_1, b_1, W_2`, the LayerNorm affine and the modulator heads, which is the test of the gauge reading in §2.2–§2.4. **This is a measurement gap, not a training problem, and it does not require the next wave to answer it.** Routed to `senior-developer` in the Next steps.

---

## 4. Observation 3 — the modulator's gradient under the two estimators

### 4.1 What the logs already exclude

The modulator's gradient norm is `‖∂L/∂θ_m‖`, and by §1.3 every term in it is weighted by the residual `y_t − V_t`. So the first candidate for "why is it 0.70–0.94× under GAE_NORM, and why does it grow under MC" is the residual's scale. I checked this against the extracted histories (`tmp/nmnsite2/`, `tmp/nmnsite3/`):

| Quantity, per decile of training | MC grid | GAE_NORM grid |
|---|---|---|
| value loss `½·MSE`, first → last decile, every arm | 0.227–0.239 → 0.226–0.240 (flat) | 0.196–0.207 → 0.203–0.209 (flat) |
| implied residual RMS `√(2·vloss)` | ≈ 0.68 | ≈ 0.64 |
| ratio of residual RMS, GAE_NORM / MC | | ≈ 0.94 |

The residual scale explains at most a 6% gap (the audit's median is 17%) and **none of the growth**: the MC residual is flat to three decimals in every arm while the modulator gradient rises by 1.2–1.8×. A second candidate — that the gain's shrinkage inflates `∂L/∂γ` through downstream compensation (`W_2 ∝ 1/γ̄`) — predicts the growth should track `1/γ̄`; across the 30 modulated arms the correlation between log-growth of the modulator gradient and log-shrinkage of the site's gain is −0.15, and the within-run correlations are of both signs. Not supported.

What the logs do show, and nobody has remarked on: the **total** gradient norm falls over training in every run of both batches (e.g. 0.42 → 0.29, 0.39 → 0.24) while the modulator's does not fall (MC) or stays flat (GAE_NORM). The modulator's share of the squared gradient norm therefore rises from roughly 15–45% early to 40–75% late in the encoder and four-site arms. At 10M episodes the modulator is the part of the network the loss is steepest in. Whether that is the contextual channel still learning or the gauge direction still drifting is exactly the §3 question.

### 4.2 The principled λ-dependence, stated as a hypothesis

There is a reason a `λ = 0.95` advantage should drive an *observation-conditioned* modulator differently from a Monte-Carlo one, and it is about correlation structure, not scale. For an estimator with effective horizon `H`, the advantage at time `t` is a function of rewards and values out to roughly `t + H`. Here

$$
H_{\text{GAE}} \approx \frac{1}{1-\gamma\lambda} \approx 10 \text{ steps}, \qquad H_{\text{MC}} \approx \frac{1}{1-\gamma} \approx 20 \text{ steps (window-limited at 128)}.
$$

The modulator reads the observation — in the narrow-input arms, only satiation and interoceptive nociception, which are slow variables that predict the *long-horizon* future (starvation, wound trajectory) far better than the next ten steps. The modulator's gradient is a covariance between the advantage and the modulator's Jacobian; the longer the estimator's horizon, the larger the component of the advantage that slow body state can explain, and the more coherent (less self-cancelling) the gradient the modulator receives. Under this hypothesis MC drives the modulator harder *because* its advantage carries more body-state-predictable variance, and the growth over training under MC would reflect episodes lengthening (from 18 to 165 steps) so that more of each window's return is long-horizon. The decile data are consistent but not decisive: the jump between the first and second deciles — when survival crosses the 128-step window — is present in both batches, and the *continued* growth in deciles 2–10 is MC-specific. The estimator's variance term, `‖ĝ‖² = ‖E ĝ‖² + tr Σ/N`, pulls the same way (MC's advantage variance is larger), so the two are confounded in the norm.

**Discriminating measurements**, neither logged today: (i) the modulator's gradient norm split by loss term (policy / value / entropy) — cheap, three extra reductions on the existing `grads['modulator']` pytree; (ii) the `R²` of the batch advantage on the two body channels, under each return mode — computable offline from stored rollouts. If (ii) is higher under MC and (i) shows the growth in the policy term, the hypothesis stands; if the growth is in the value term, look instead at the critic's dependence on the modulator.

### 4.3 What the growth-versus-flat difference does and does not tell us

It does **not** tell us the modulator is learning "more" under MC: the same total variance, larger norm, is what a noisier estimator produces at a fixed parameter. It does tell us that whatever the modulator is fitting under MC, it had not stopped fitting at 10M episodes, while under GAE_NORM it had reached a stationary gradient regime by the second decile. The action-head site is the one place where the modulator's gradient *declines* in both batches (0.87×, and 0.73–0.75×), and it is also the only site whose gain never turns around (§2.4); those two facts are consistent with an actor-side modulator that is progressively being folded into the layer — the gauge motion — rather than one that is still shaping context-dependent action. Again: §3's measurements decide it.

---

## 5. Observation 4 — why the narrow-input arms have quieter total gradients

Three mechanisms could produce a quieter *whole-network* gradient when the modulator's input is restricted to two slow channels. The data rule one of them out at one site and cannot separate the other two.

1. **Input bandwidth → modulation smoothness → policy sensitivity.** With nineteen fast exteroceptive channels, `γ_t, β_t` change from step to step, so the effective layer `diag(γ_t) W_1` is a fast function of the observation. The modulator then constitutes a second, shallow observation-to-action pathway (27 inputs → 16-unit GRU → per-unit gains on the actor layer → logits) that bypasses the recurrent trunk, has few parameters, and — under Adam — a high relative learning rate. The PPO ratio `r_t` becomes more sensitive to that pathway, which raises the variance of the advantage-weighted gradient and its tail. With two slow inputs the pathway carries almost no high-frequency content. This is the audit's own mechanism, made precise. *Prediction:* `ρ_γ` (§3.2) and the temporal standard deviation of `γ_t` are larger in the ALL and X arms than in the I arms.
2. **Backward-pass attenuation by the gain.** The gradient reaching `W_1` and everything upstream of a FiLM site is multiplied by `γ_t` — the audit's May theoretical memo calls this the "chain-rule gradient-gating channel" ([[20260518_film_as_hyperparameter_modulator_theoretical_audit]] Q2). Lower gains mean quieter upstream gradients. **This fails at the action head**: the I arm's actor gain (0.667) is *higher* than the ALL arm's (0.508) while its total gradient is far quieter (4.5% vs 83% of windows at the ceiling). It may still contribute at the encoder and value head, where the I arms' gains are the lowest in the batch.
3. **Dormancy.** Units that are off carry no gradient; if the I arms have driven more units off (their offsets at the encoder are the most negative), fewer parameters carry gradient. *Prediction:* the I arms' `p_i` distributions (§3.1) have more mass near zero.

Mechanism 1 is the one I would bet on, mechanism 2 is excluded where the effect is largest, mechanism 3 is a live worry rather than an explanation. The same measurements as §3 separate them. One further point for the outcome analysis, since the two are entangled: a quieter gradient is *not* a quieter policy — a modulator whose output is slow can still produce a strongly context-dependent policy, provided the context is slow, which is precisely the interoceptive case. The narrow-input arms being quiet is therefore neutral for H5 until `ρ` and the body-state `R²` are known.

---

## 6. Observation 5 — gradient clipping, the 1030 spike, and FiLM

### 6.1 What the clip does under this optimiser

The chain is `clip_by_global_norm(0.5)` **then** Adam. Under Adam, a *constant* rescaling of the gradient cancels — `m̂ / √v̂` is invariant to multiplying `g` by a constant — so a run whose gradient sat permanently above the ceiling would be barely affected (this is the point [[mc_return_units_bug_severity_and_repair]] §6b makes and §10 confirms). What Adam cannot cancel is *intermittent* clipping, because it rescales some updates and not others, which is equivalent to an update-dependent learning rate. **The relevant statistic is therefore the one the audit computed as the "softer" measure — the fraction of windows whose peak touches the ceiling — not the fraction whose mean does.** Read that way, "83–87% of windows" (wide-input actor arm) versus "4.5%" (narrow-input) is a real difference in optimiser regime: the wide-input arms are trained with an effective learning rate that fluctuates by a factor of ≈2 inside most windows, the narrow-input arms with one that almost never does. This is a consequence of mechanism 1 in §5, and it is a second reason the narrow-input arms should not be compared to the wide-input arms as if they had been optimised identically. It is not a defect; Andrychowicz et al. 2021 find global clipping a "small boost, threshold-insensitive" (choice C68, [ppo_implementation_details_lit_review.md §4.7](../references/ppo_implementation_details/ppo_implementation_details_lit_review.md)), and the intermittent regime is the normal one.

### 6.2 The 1030 spike

Its pre-clip norm was ≈ 4,000× typical; the clip scaled the update by `0.5/1030 ≈ 5 × 10⁻⁴`. The part of this that matters and was not said: had it *not* been clipped, Adam's second-moment estimate for every affected parameter would have absorbed a `g²` term ≈ 10⁷× normal, inflating `v` by roughly `(1 − β₂) × 10⁷ ≈ 10⁴` relative to steady state, suppressing those parameters' steps by ≈ 100× and decaying back over ≈ `1/(1 − β₂) = 1000` updates — about 250 iterations at four epochs. Clipping before Adam is what prevented a quarter-thousand-iteration stall, and it is the reason the run's loss curves show nothing at all. That the sixteen-run twin under GAE_NORM produced nothing above 8.4 is consistent with the audit's reading that this is a tail event of the MC target. Two candidate mechanisms specific to that target, both unresolvable from what is logged: the per-batch z-score divides by `std(G) + 10⁻⁷`, which explodes if one rollout's returns are nearly degenerate; and the PPO ratio `exp(log π_new − log π_old)` has no ceiling on the log-probability difference. The metrics that would localise a recurrence are the standard ones from the "37 implementation details" corpus that this trainer does not log: approximate KL, clip fraction, and the per-update (not per-window) gradient norm, plus the batch return standard deviation under MC.

### 6.3 Does the clip interact with FiLM in a way to worry about?

No known pathology, and the corpus has none (gain blow-up is "unattested" in both the FiLM and modulation-in-RL surveys). Two mild couplings, for completeness: the clip is global, so when the main network spikes the modulator's step is throttled with it and vice versa — symmetric, and the modulator is 10–60% of the norm, so neither side dominates the clip decision; and the clip-then-Adam chain makes the step along the gauge direction of §2.2 learning-rate-sized on every update regardless of how flat the loss is, which is why the offsets drift at a rate unrelated to any loss quantity. Neither is a reason to change the ceiling mid-experiment.

---

## 7. Classification — normal, concern, or measurement gap

| Observation | Classification | Basis |
|---|---|---|
| Gains 0.2–0.9, offsets negative, at every site | **Normal** | Canonical FiLM behaviour (Perez 2018 §4.2); the mean part is gauge (§2.2) |
| Action- and value-head offsets still sliding at 10M | **Normal for the parameterisation; not a budget problem** | Flat direction under Adam without weight decay (§2.2b); would not converge with more budget |
| Encoder gain U-shape | **Undetermined; probably gauge** | Two readings, separable on checkpoints (§2.4) |
| Possible dormancy from `γ̄ ≈ 0.2, β̄ ≈ −1` before a rectifier | **The one real concern** | Absorbing state, invisible in logs; corpus names it as the untested confound (§3.1) |
| Whether modulation is contextual or static | **Measurement gap** | Pooled std cannot separate the terms; `ρ` and the freeze counterfactual can (§3.2) |
| Modulator gradient 0.70–0.94× under GAE_NORM; grows under MC | **Real, unexplained; not a health problem** | Residual scale and gain compensation excluded by logs (§4.1); horizon hypothesis testable (§4.2) |
| Modulator's share of squared gradient norm rising to 40–75% late | **Worth a reader's attention; nobody has looked** | Follows from total falling while modulator does not (§4.1) |
| Narrow-input arms quieter | **Real; probably input bandwidth** | Gain attenuation excluded at the actor (§5) |
| Intermittent clipping in 83–87% vs 4.5% of windows | **Real difference in optimiser regime, not a defect** | Adam cancels constant but not intermittent scaling (§6.1) |
| The 1030 spike | **Tail event, correctly absorbed** | Clip-before-Adam protected the second moment (§6.2) |
| Per-unit spread ±0.3 at step 0 | **Minor; document, do not fix mid-experiment** | Half-applied zero-init (§1.1); audit §8.1 |

---

## 8. What to add or change before the next wave

**Before any new run (offline, on existing checkpoints):**

1. A checkpoint replay script that records `γ_it, β_it` and post-FiLM pre-activations over a rollout and emits, per checkpoint: the `p_i` distribution and dead fraction; Sokar's `τ`-dormancy at `τ = 0.025`; `ρ_γ, ρ_β`; the body-state `R²`; and the freeze-at-mean survival and behaviour. Run on the control at matched checkpoints for the null. This answers §2.4, §3, §5 for all thirty modulated runs and is the single highest-value item.
2. From the same checkpoints, parameter-norm series of `W_1, b_1, W_2`, the LayerNorm affine and the modulator heads, to confirm or refute the gauge reading of the drift.

**Logging additions (cheap, for the next wave):**

3. `modulator/grad_norm_{policy,value,entropy}` — the modulator's gradient norm per loss term (§4.2).
4. `network/<site>_active_fraction` as requested by the audit, **plus** the per-unit `p_i` histogram at a low cadence (e.g. every checkpoint), since the mean alone cannot separate gating from dormancy (§3.1).
5. `modulator/<site>_temporal_std` and `_unit_std` (the two terms of §3.2), as the audit requests.
6. The standard PPO health trio the trainer does not log — approximate KL, clip fraction, per-update gradient norm — and, under MC, the batch return standard deviation before z-scoring (§6.2).

**Design changes to consider, not to make mid-experiment:**

7. Zero-initialise the FiLM head weight matrices (Beck et al. 2023 Bias-HyperInit; the recent-variants survey's "universal prophylactic"), making the step-0 identity exact per unit. One-run control, as the audit proposes.
8. If item 1 finds modulation-induced dormancy, the two remedies with precedent are a damped gain parameterisation, `γ = 1 + 0.1·tanh(·)`, which confines gains near identity throughout training (Marquis, in [modulation_in_rl_lit_review.md Q3](../references/modulation_in_rl/modulation_in_rl_lit_review.md)), and a small weight decay on the modulator heads and baselines, which restores a restoring force along the gauge direction. Both change the science; neither should be adopted before item 1 says it is needed.
9. Seeds. Everything above is per-arm at one shared seed; the gauge and dormancy claims are about the parameterisation and should replicate, but the arm-level attributions (which arm is quietest, which offset is still moving) should not be leaned on, as the audit already says.

---

## 9. Relation to the project's hypotheses and prior diagnoses

- **H1 (perceptual gain rises after injury)** and **H5 (the modulator must read the body)** in [[NEUROMODULATION_ALGORITHM]] §1.4 are stated in terms of `mod_gamma_mean` moving. By §2.2 the *level* of the mean gain is not a behavioural quantity; only the event-locked *deviation* `δγ_t` is. Any H1 analysis should therefore be written as a within-run contrast of `γ_t` against its own time-average, never as an absolute level or a cross-run comparison of means.
- **The grid's engagement criterion** ([[NMN_INPUT_SITE_GRID]] §4.3, `gamma_<site>_std` leaving zero) is satisfied by static per-unit re-tuning alone and does not establish that a site is *contextually* engaged. `ρ` (§3.2) is the criterion that does.
- **The May theoretical audit** ([[20260518_film_as_hyperparameter_modulator_theoretical_audit]]) identified a scalar gauge between a learning-rate-like gain and the learning rate; this memo's gauge is the per-unit affine one between the modulator's constant output and the modulated layer's own affine parameters. They are different symmetries with the same consequence: a logged FiLM statistic is behaviourally meaningful only in its non-gauge part.
- **The v8 null-result series.** The agent profile's anchor to "four causes in `project_plan.md` §4" no longer resolves — the plan has been rewritten around the two papers and that section is gone. The relevant prior is [[NMN_PERFORMANCE_DIAGNOSIS_v8]] together with [[mc_return_units_bug_severity_and_repair]] §10, whose finding that the shared trunk is trained by the critic when the value gradient dominates is the same *kind* of observation as §4.1 here (the modulator becoming the steepest part of the network late in training): both are statements about where the gradient norm lives, and both are only interpretable once gradients are logged per term and per parameter group.

---

## 10. Precedents and missing references

Held in the corpus and used: Perez et al. 2018 (FiLM; gain/offset histograms); Beck et al. 2023 (Bias-HyperInit) and the AdaLN-Zero line via the FiLM surveys; Sokar et al. 2023 (dormant neurons / ReDo), Abbas et al. 2023 (activation collapse), Lyle et al. (plasticity loss) under `continual_learning/`; Andrychowicz et al. 2021, Engstrom et al. 2020, Huang et al. 2022 under `ppo_implementation_details/`; the Marquis damped-gain and Lipschitz diagnostic under `modulation_in_rl/`.

Not held, and needed to cite §2.2–§2.3 properly — for `literature-reviewer`:

- Kunin, Sagastuy-Brena, Ganguli, Yamins & Tanaka 2021, *Neural Mechanics: Symmetry and Broken Conservation Laws in Deep Learning Dynamics* (ICLR) — the conserved quantity under scale symmetry and its breaking by Adam and weight decay.
- Du, Hu & Lee 2018, *Algorithmic Regularization in Learning Deep Homogeneous Models* (NeurIPS) — balancedness conservation for ReLU layers under gradient flow.
- Neyshabur, Salakhutdinov & Srebro 2015, *Path-SGD* (NeurIPS) — rescaling invariance of ReLU networks.
- van Laarhoven 2017, *L2 Regularization versus Batch and Weight Normalization* (arXiv:1706.05350) — effective learning rate under scale invariance.
- Lu, Shin, Su & Karniadakis 2019, *Dying ReLU and Initialization* (arXiv:1903.06733) — the absorbing-state argument for rectifier death.
- Zhang, He, Sra & Jadbabaie 2020, *Why Gradient Clipping Accelerates Training* (ICLR) — clipping under heavy-tailed gradient noise, for §6.

The closest published design to "an observation-conditioned FiLM generator modulating an on-policy actor and critic, trained end-to-end through the policy gradient" remains PAPL and Marquis (both in `modulation_in_rl/`); neither reports gain/offset trajectories over training, per-unit dormancy, or a contextual-fraction statistic. If §3's measurements are run and reported, that is a small but genuinely unreported result.

---

## Next steps

- **`senior-developer`** — plan the offline checkpoint-replay diagnostic (§8 items 1–2): a script under `scripts/analysis/` that loads an rPPO checkpoint, rolls out with `mod_info` and post-FiLM pre-activations captured, and emits the §3 quantities per checkpoint; plus the six logging additions (§8 items 3–6) for the next wave. Items 1–2 need no code change in `src/` beyond reading what the model already returns.
- **`experiment-designer`** — fold `ρ_γ, ρ_β`, the body-state `R²`, and the freeze-at-mean counterfactual into the grid's §4.3 diagnostic outcome as the *contextual* engagement criterion; restate the design's step-0 identity claim per §8.1 of the audit.
- **`experiment-analyzer`** — when running the outcome analysis, treat all `gamma_/beta_<site>_mean` levels as gauge quantities: event-locked deviations only, no cross-arm comparison of means.
- **`literature-reviewer`** — fetch the six papers in §10 into a `references/optimisation_symmetry/` topic (or under `FiLM/`), for the gauge and dormancy citations.
- **`professor-dl-theory`** — the per-unit affine gauge in §2.2 is an architectural statement about FiLM-before-ReLU and is theirs to formalise if it is to appear in Paper 2's "modulation as a hypernetwork-class object" framing; this memo uses it only for what it implies about training dynamics.
- **`bug-curator`** — no new bug; the half-applied zero-init (§1.1) is a documented design property, already recorded by the audit's §8.1.
