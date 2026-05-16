---
title: "Anti-Exploration by Random Network Distillation"
authors: ["Alexander Nikulin", "Vladislav Kurenkov", "Denis Tarasov", "Sergey Kolesnikov"]
year: 2023
venue: "ICML 2023 (PMLR 202)"
slug: nikulin_2023_anti_exploration_rnd
source_pdf: docs/project/references/FiLM/sources/Nikulin et al. 2023 - Anti-exploration by Random Network Distillation.pdf
topic: FiLM
---

# Anti-Exploration by Random Network Distillation

## Plain-English entry point

This paper sits in **offline reinforcement learning** — the setting where an agent must learn a good policy from a *fixed* dataset of past experience, without any new interaction with the environment. The central failure mode of offline RL is that, during policy improvement, the actor proposes actions outside the dataset (out-of-distribution, OOD), and the Q-network — which has never seen those actions — wildly overestimates their value. The actor then chases this hallucinated value, and training collapses.

The dominant fix has been **deep Q-ensembles**: train many Q-networks (sometimes 100s) and use their *disagreement* as a per-(state, action) uncertainty bonus that is *subtracted* from the reward — penalising actions the ensemble can't agree on. The authors call this **anti-exploration**: it inverts the usual "intrinsic motivation" idea (an explorer is *rewarded* for novelty; an offline learner is *penalised* for novelty). Ensembles work, but they are slow and memory-heavy.

**Random Network Distillation (RND)** — a fixed randomly-initialised "prior" network and a trainable "predictor" that tries to copy the prior on the dataset — is a cheap *ensemble-free* alternative to estimating epistemic uncertainty. But a prior paper (Rezaeifar et al. 2022) reported that RND fails as an anti-exploration bonus in continuous action spaces, and concluded RND just isn't discriminative enough for offline RL.

Nikulin et al. show this is wrong. The real problem is **how the action is fused into the RND prior**: standard "concatenate state and action" creates a bonus landscape with noisy gradients that *the actor cannot follow*. Replacing concatenation with a **FiLM (Feature-wise Linear Modulation)** layer — where the state produces channel-wise gain and bias parameters that modulate the action features — yields a smooth, globally-coherent gradient field. With this single change, their algorithm **SAC-RND** matches state-of-the-art Q-ensemble methods on D4RL (Gym + AntMaze) using no ensembles. **FiLM is the load-bearing architectural choice.**

## Section-ordered backbone

**1. Introduction.** Offline RL needs uncertainty penalties on OOD actions. Ensembles dominate but are expensive. RND is a cheap alternative, but prior work reported it doesn't work for continuous-action offline RL. The paper revisits this and identifies *prior conditioning on the action* as the actual bottleneck. Their fix (FiLM prior) yields SAC-RND, an ensemble-free method matching ensemble-based SOTA on D4RL.

**2. Background.** Reviews offline RL as constrained MDP optimisation; anti-exploration framing — instead of policy-constraint regularisation, subtract a state-action uncertainty bonus $b(s, a)$ from the TD target:

$$y_t = r_t + \gamma\,\mathbb{E}_{a' \sim \pi(\cdot \mid s')}\!\bigl[Q(s', a') - b(s', a')\bigr].$$

Reviews **RND**: two networks $f_\psi, \bar f_{\bar\psi}$ mapping states (or here, state-action pairs) to $\mathbb{R}^K$ embeddings; predictor trains to minimise $\|f_\psi(s) - \bar f_{\bar\psi}(s)\|_2^2$ on the dataset; the prior is frozen. The bonus is the prediction error. Reviews multiplicative-interaction operators used in the paper: **gating** $f(a,s) = \tanh(W_1 a + b_1) \odot \sigma(W_2 s + b_2)$; **bilinear** $f(a, s) = s^\top W a + s^\top U + V a + b$; and **FiLM** $f(h, s) = \gamma(s) \odot h + \beta(s)$.

**3. RND is discriminative enough.** The authors replicate Rezaeifar et al.'s "RND can't distinguish ID from OOD actions" experiment and find the *opposite* result: with a properly-sized predictor (same as or larger than prior, not smaller), RND on plain state-action concatenation does separate ID from OOD action distributions — comparable to a trained Q-ensemble. Discriminativity is not the issue. (The discrepancy with prior work is traced to Rezaeifar et al. using a predictor *smaller* than the prior, contradicting Ciosek et al. 2019's recommendation.)

**4. Concatenation prior hinders bonus minimisation.** A well-behaved bonus should be (a) discriminative (already shown in §3) and (b) *minimisable by the actor*. To probe (b), the authors strip the critic and run a SAC variant where the actor *only* minimises the RND bonus + entropy term. With concatenation in the prior, the actor cannot drive the bonus down to its ID minimum, and the distance to dataset actions does not decrease — the actor cannot clone the behavioural policy. With **FiLM** in the prior, the actor successfully minimises the bonus and clones the behavioural policy. Bilinear works as well; gating works less well; concatenation fails. **Table 1** confirms this on actual SAC training: FiLM and bilinear priors give 95–100 average D4RL score; concatenation gives 89.6, gating 67.3.

**5. Anti-Exploration by Random Network Distillation (SAC-RND).** Method assembly. RND is pretrained with MSE between prior and predictor on the offline dataset (both networks are 4-layer MLPs, comparable in size to actor/critic). The **prior** uses FiLM conditioning on its penultimate layer (state generates $\gamma, \beta$, action is the modulated stream). The **predictor** uses bilinear conditioning in its first layer (selected by Table 1 sweep). The anti-exploration bonus is

$$b(s, a) = \frac{\|f_\psi(s, a) - \bar f_{\bar\psi}(s, a)\|_2^2}{\sigma_{\text{RND}}},$$

i.e. the RND prediction error divided by a running standard deviation of the prediction error tracked during pretraining, so the bonus has a comparable scale across environments. The bonus is subtracted from the TD target as in Eq. (1) and an extra BC-like penalty on the policy uses the same bonus.

**6. Experiments.** D4RL Gym domain (HalfCheetah, Walker2d, Hopper, 18 datasets) and D4RL AntMaze domain. SAC-RND averages 85.2 on Gym — matching EDAC (85.2) and just behind RORL (85.7), both ensemble-based — while using *no ensemble*. On AntMaze it averages 76.6, beating ensemble-free baselines (IQL 63.0, CQL 50.6) and matching ensemble-based RORL (65.2), but trailing MSG (83.6). Section 6.3 visualises the *anti-gradient field* of the bonus on a toy 4-state, 2D-action environment: concatenation yields noisy local gradients pointing to nearby minima; **FiLM yields smooth gradients pointing to the correct global minimum across the entire action space**. Section 6.4 sweeps predictor/prior conditioning types (gated, concat, bilinear, full bilinear, FiLM) at three depths (first / last / all layers): for the *prior*, conditioning at every layer (FiLM-full, bilinear-full) wins; for the *predictor*, conditioning only at the last layer is best.

**7. Related Work.** Surveys offline RL constraint methods (BCQ, BEAR, CQL, IQL, TD3+BC, advantage-weighted regression), ensemble methods (SAC-N, EDAC, MSG, RORL), and ensemble alternatives (MC-dropout, BatchEnsemble). Positions SAC-RND as the *first ensemble-free* RND-based method to match ensemble SOTA in continuous-action offline RL.

**8. Discussion.** FiLM is one of several conditioning operators that work; what matters is that the conditioner can produce a smooth, globally-coherent gradient field for the actor. Open question: why FiLM/bilinear succeed where concatenation fails — likely the multiplicative inductive bias of FiLM that lets the state genuinely *re-weight* action features rather than just appending to them.

## Phase 1 — Undergraduate-level synthesis

**What the paper is about.** In offline RL, the actor proposes actions never seen in the dataset and the Q-network overestimates their value. To stop this, prior work subtracts a "novelty bonus" $b(s, a)$ from the Q-target — penalising actions the system isn't sure about. The cheap way to compute that bonus is **RND**: train a small network to imitate a fixed random network on the dataset, then use the imitation error as novelty. The expensive way is a giant Q-ensemble.

**The puzzle.** Prior work (Rezaeifar et al. 2022) tried RND for continuous-action offline RL and reported it didn't work. Nikulin et al. show RND *can* tell ID actions from OOD actions just fine. So why does plain RND still fail in SAC training? Because the **actor can't follow the gradient of the bonus**. The bonus surface is too bumpy when the action is just *concatenated* with the state inside the RND prior — local gradients don't point toward the global minimum.

**The fix.** Inject the state into the RND prior via a **FiLM layer**: the state computes channel-wise scales $\gamma(s)$ and shifts $\beta(s)$, and these multiply and shift the *action's* features inside the prior. This makes the bonus surface smooth across the whole action space, so the actor can find dataset-like actions by gradient descent.

**Why this matters for FiLM as a concept.** FiLM is normally treated as a tool for *what features should I compute given this conditioning input* (e.g. given this question, attend differently to this image). Nikulin et al. show that FiLM also has a *gradient-shaping* property — it produces a smoother and more globally informative loss landscape for a downstream optimiser than plain concatenation does. This is a new use of FiLM and the paper's contribution beyond "use FiLM, get better numbers".

**Anti-exploration vs. exploration.** The conceptual inversion is worth pausing on. In online RL, RND prediction error is *added* as an intrinsic reward, pushing the agent toward novel states. In offline RL, the same prediction error is *subtracted*, pulling the agent toward familiar (in-distribution) actions. Same operator, opposite sign — hence "anti-exploration".

**Result.** SAC-RND with the FiLM prior matches state-of-the-art ensemble methods on the D4RL benchmark while using no ensemble, dropping memory and compute by an order of magnitude.

## Phase 2 — Graduate-level deep dive

### 2.1 Anti-exploration framing of offline RL

The actor-critic update with an anti-exploration bonus $b: \mathcal{S} \times \mathcal{A} \to \mathbb{R}_{\ge 0}$ subtracted from the next-state value:

$$
y_t \;=\; r_t \;+\; \gamma\,\mathbb{E}_{a' \sim \pi(\cdot \mid s')}\!\bigl[Q(s', a') - b(s', a')\bigr],
$$

with critic update minimising $\mathbb{E}\bigl[(Q_\theta(s_t, a_t) - y_t)^2\bigr]$ and actor update maximising $\mathbb{E}_{a \sim \pi_\phi}\bigl[Q_\theta(s, a) - \alpha \log \pi_\phi(a \mid s)\bigr]$ in the SAC entropy-regularised form. The contrast with online RL is that there the intrinsic reward $+b$ is *added*:

$$
y_t^{\text{online}} \;=\; r_t \;+\; b(s_t, a_t) \;+\; \gamma\, \mathbb{E}_{a'}\bigl[Q(s', a')\bigr].
$$

Same novelty signal, opposite sign — anti-exploration is exploration's mirror.

### 2.2 Random Network Distillation for state-action novelty

Two networks map $\mathcal{S} \times \mathcal{A} \to \mathbb{R}^K$: a frozen prior $\bar f_{\bar\psi}$ with random init, and a trained predictor $f_\psi$ trained to imitate the prior on the offline dataset $\mathcal{D}$:

$$
\mathcal{L}_{\text{RND}}(\psi) \;=\; \mathbb{E}_{(s,a) \sim \mathcal{D}}\!\bigl[\|f_\psi(s, a) - \bar f_{\bar\psi}(s, a)\|_2^2\bigr],
$$

with the gradient through the prior disabled (the prior is the random target). After pretraining, the bonus is the (rescaled) prediction error:

$$
b(s, a) \;=\; \frac{\|f_\psi(s, a) - \bar f_{\bar\psi}(s, a)\|_2^2}{\sigma_{\text{RND}}},
$$

where $\sigma_{\text{RND}}$ is the running standard deviation of $\mathcal{L}_{\text{RND}}$ tracked during pretraining, so the bonus is scaled comparably across environments. The Ciosek et al. (2019) interpretation: predictor capacity should be **at least equal** to prior capacity, so that the predictor can drive $\mathcal{L}_{\text{RND}}$ to (near-)zero on the support of $\mathcal{D}$. Then $b(s, a) \approx 0$ for ID state-action pairs and $b(s, a) > 0$ for OOD pairs — exactly the inductive bias offline RL needs.

### 2.3 FiLM-based prior conditioning — the load-bearing equation

For the **prior network**, conditioning is applied on the penultimate hidden layer. Let $h \in \mathbb{R}^d$ be the action-stream hidden representation (i.e. the action is processed through a small MLP); the state $s$ produces channel-wise scale and shift parameters via a learnt linear projection:

$$
\boldsymbol{\gamma}(s) \;=\; W_\gamma s + b_\gamma \in \mathbb{R}^d, \qquad
\boldsymbol{\beta}(s) \;=\; W_\beta s + b_\beta \in \mathbb{R}^d.
$$

The FiLM modulation:

$$
\widetilde h \;=\; \boldsymbol{\gamma}(s) \odot h \;+\; \boldsymbol{\beta}(s),
$$

where $\odot$ is element-wise multiplication. The crucial structural difference from concatenation is that $s$ enters the action computation **multiplicatively** through $\gamma(s)$, allowing the state to *gate* (re-weight) the action features rather than merely sitting alongside them. This is FiLM in the Perez et al. 2018 form — the same operator as `perez_2018_film.md` (B1) and `jang_2022_bcz.md` (this batch), but here applied to *condition the OOD-detection prior* rather than to condition a visuomotor policy.

For the **predictor**, the authors find that **bilinear** conditioning in the first layer works slightly better than FiLM (Table 1: bilinear 99.9 vs. FiLM 95.0 average). The bilinear form (PyTorch's simplified version):

$$
f(a, s) \;=\; s^\top W a + b,
$$

with $W \in \mathbb{R}^{d_s \times d_a}$ (or with explicit hidden dim, $W \in \mathbb{R}^{d \times d_s \times d_a}$), so action and state interact through a learned outer-product-like form. Note that **FiLM is a special low-rank case of bilinear** — Perez et al. 2018 derived FiLM as a low-rank-constrained bilinear layer — so the FiLM/bilinear split here is a continuum, not a categorical difference.

### 2.4 Why FiLM yields a minimisable bonus surface

The empirical evidence (Fig. 4 of the paper, toy 4-state 2-D action environment) is striking: under concatenation prior, the anti-gradient field of the bonus has noisy local minima and the directions only point to the *correct* minimum in a small neighbourhood; under FiLM prior, the anti-gradient field is smooth and globally points to the correct ID action minimum for each state. The hypothesised mechanism (paper's Section 6.3 + 8):

- With concatenation, the state vector enters as a constant additive perturbation to a network whose subsequent layers must learn to *distinguish* in-distribution from out-of-distribution action regions purely by varying their *additive* response to the action. The resulting bonus surface as a function of $a$ for fixed $s$ has a complex non-convex structure determined by random initialisation in the prior.
- With FiLM, the state vector enters as a *multiplicative gain* on the action features, which is a strictly more expressive way to encode "this state cares about this dimension of action and not that one". Random initialisation of the prior with FiLM still produces a non-trivial surface, but the multiplicative structure means the partial derivative $\partial b / \partial a$ is shaped by the state-conditioned gain $\gamma(s)$, smoothing it across the action space.

This is the cleanest "FiLM as gradient-landscape shaper" claim in the corpus, and is **why this paper deserves inclusion in the FiLM corpus** despite its primary venue being offline RL.

### 2.5 Anti-exploration policy regularisation

The actor in SAC-RND maximises (entropy-regularised SAC objective with anti-exploration bonus):

$$
\mathcal{L}_\pi(\phi) \;=\; \mathbb{E}_{s \sim \mathcal{D},\, a \sim \pi_\phi(\cdot \mid s)}\!\Bigl[Q_\theta(s, a) - \alpha \log \pi_\phi(a \mid s) - \eta\,b(s, a)\Bigr],
$$

where $\eta > 0$ is a conservatism coefficient. The bonus serves a *dual* role: it penalises Q-target overestimation through the critic update (Eq. 1) **and** it directly penalises the actor for proposing OOD actions through the term $\eta\,b(s, a)$. The latter is a soft behavioural-cloning constraint — minimising $b$ pulls the policy toward dataset-like actions.

### 2.6 SAC-RND training algorithm (schematic)

```
Pretrain phase:
  for k = 1 .. K_RND:
    sample (s, a) ~ D
    update predictor ψ to minimize ||f_ψ(s,a) - f̄(s,a)||²
  freeze ψ and f̄; record σ_RND

Online training phase:
  for t = 1 .. T:
    sample (s, a, r, s') ~ D
    compute b(s', a') = ||f_ψ(s', a') - f̄(s', a')||² / σ_RND   for a' ~ π_φ(·|s')
    critic update: minimize (Q_θ(s,a) - [r + γ(Q(s',a') - b(s',a'))])²
    actor update:  maximize  Q_θ(s, a) - α log π_φ(a|s) - η b(s, a)   for a ~ π_φ(·|s)
```

The pretrain phase is offline-data-only; the online phase is the standard SAC update with the bonus inserted as in §2.5.

### 2.7 Empirical signature

Table 2 (Gym): SAC-RND 85.2 ≈ EDAC 85.2 ≈ RORL 85.7, vs. ensemble-free baselines (CQL 73.6, IQL 68.9, TD3+BC 67.5). Table 3 (AntMaze): SAC-RND 76.6 vs. RORL 65.2 vs. MSG 83.6 vs. ensemble-free best (IQL 63.0). The clean message: **a single FiLM swap in the RND prior closes the gap between ensemble-free and ensemble-based offline RL.**

## Connections

- **`perez_2018_film.md` (B1, canonical FiLM)** — Nikulin et al. cite Perez et al. directly in §2 (their FiLM definition) and use FiLM with the standard $\gamma(s) \odot h + \beta(s)$ form. The novel contribution is *applying FiLM to an OOD-detection prior* and discovering its **gradient-landscape-shaping property** — a new theoretical / empirical role for FiLM beyond visual-question-answering conditioning.
- **`jang_2022_bcz.md` (this batch)** — BC-Z uses FiLM as a *control-policy* conditioner (task embedding → policy); Nikulin et al. use FiLM as a *novelty-prior* conditioner (state → action features in OOD detector). Same operator, very different role: one steers a policy, the other shapes a loss landscape so an optimiser can find dataset-like actions.
- **`moon_2023_hierarchical_achievements.md` (this batch)** — Both papers use FiLM as a fusion operator in an auxiliary module rather than the policy itself. Moon uses FiLM to fuse action into state embedding for a contrastive head; Nikulin uses FiLM to fuse state into action features in an RND prior. Together with BC-Z they bracket the "FiLM at large scale" (BC-Z) vs. "FiLM as a small fusion gadget" (Moon, Nikulin) range.
- **`turkoglu_2022_film_ensemble.md` (B3, FiLM-Ensemble)** — Indirectly relevant: Nikulin et al.'s motivation is to *eliminate* deep Q-ensembles for offline RL with a single FiLM-prior RND. FiLM-Ensemble's motivation is to *replicate* ensembles using FiLM channels in a single network. Both leverage FiLM's expressivity-per-parameter, but for opposite ensemble-elimination goals.
- **Project-side relevance** — For NMN-as-modulator: Nikulin's "FiLM smooths the gradient field that the actor sees" is the most useful claim in this paper for our project. If the interoceptive modulator's role is to *shape the policy's loss landscape* (e.g., by injecting state-conditioned gain/bias into the value head), Nikulin gives empirical evidence that FiLM does this better than concat. Worth flagging to `professor-rl-bayesian-dl` if the NMN-as-modulator direction goes deeper.
- **Hand-off** — No code change recommended here; this is a corpus-reference review.
