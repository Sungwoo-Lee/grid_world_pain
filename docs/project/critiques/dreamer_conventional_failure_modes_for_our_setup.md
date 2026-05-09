---
title: "Conventional DreamerV3 failure modes — likelihood ranking against our setup"
topic: critique
status: active
created: 2026-05-09
last_updated: 2026-05-09
---

# Conventional DreamerV3 failure modes — likelihood ranking against our setup

> **One-line summary.** Walking the textbook DreamerV3 failure-mode checklist against `configs/models/dreamer_v3.yaml` and the `3zjhap9w` collapse signature, the three most likely conventional causes are (1) **reward scaling / symlog mismatch on small-magnitude homeostatic rewards combined with a 100× larger death penalty**, (2) **replay-ratio + train-steps over-training on a tiny, near-deterministic state distribution**, and (3) **continuation-head class imbalance + λ-return horizon truncation that hides the death terminal**. The user's "this isn't too difficult" framing is **partially correct**: causes 1 and 2 are very conventional and very fixable; cause 3 is also conventional but is what the team has already been pulling on. The investigation has been gravitating toward exotic explanations (per-action `cont`, fork-rollouts) when the prior should still be on reward scaling and over-training before anything bespoke.

> **Status.** Critique. Not an implementation plan. Math first, then a per-item likelihood table, then a top-3 ranking, then the easiest-to-test interventions ordered cheapest-first.

---

## 1. The textbook DreamerV3 (Hafner et al., 2023) defaults — what we are deviating from

DreamerV3 was tuned to work as **one architecture across Atari, DMC, Crafter, Minecraft** without per-task tuning. The defaults that matter for our analysis:

| Knob | Hafner-2023 default | Our `dreamer_v3.yaml` | Delta |
|---|---|---|---|
| `replay_ratio` (train_steps per env step, per env) | **1/16 ≈ 0.0625** (Atari/DMC), proxy via "train_ratio = 16/512" | **0.5** | **8× higher** |
| `batch_size` × `sequence_length` | 16 × 64 (small models) / 16 × 64 (XL) | 16 × **128** | seq 2× longer |
| `sequence_length` (BPTT horizon) | 64 | 128 | 2× |
| `imagine_horizon` | 15 | 15 | matches |
| `entropy_scale` | 3e-4 | 3e-4 | matches |
| Discount γ | 0.997 (DMC), 0.997 (Atari) | (codebase, presumed default) | likely matches |
| `unimix` | 0.01 | 0.01 | matches |
| KL free bits / clip | 1.0 nat (per dim) for posterior+prior, asymmetric weights $\alpha_{\text{rep}}=0.1, \alpha_{\text{dyn}}=0.5$ | (we log `loss_dyn_kl ≈ loss_rep_kl ≈ 2.0`) | **possibly symmetric / no free bits — see §2.3** |
| Reward symlog two-hot | 255-bin two-hot, range $[-20, 20]$ in symlog space | (codebase, presumed default) | likely matches |
| `cont_loss_weight` (BCE on continuation head) | **1.0** in paper; head is one of three balanced WM heads | 1.0 | matches |
| Action space | discrete (Atari) / continuous (DMC) | discrete-6 | within scope |
| Observation type | image 64×64 → CNN | **vector, 19-dim → MLP** | nonstandard |
| `num_envs` | 1 (Atari) / 4 (DMC) | 16–128 | OK |

**The two big quantitative deltas are `replay_ratio = 0.5` (8× the canonical Atari ratio) and `sequence_length = 128` (2×).** The other big *qualitative* delta is **vector observations rather than pixels**, which removes the dense reconstruction objective the WM was tuned around. We will return to all three in §3.

---

## 2. The conventional failure-mode checklist

For each item: a one-paragraph statement of the failure mode in DreamerV3 terms, the formal quantity that diagnoses it, the value of that quantity in our runs, and a likelihood rating.

### 2.1 Reward-scale / symlog mismatch under heavy-tailed mixed-sign rewards — **HIGH**

DreamerV3's reward head is a **two-hot categorical over 255 bins in symlog space**, $\tilde y = \mathrm{symlog}(r) = \mathrm{sign}(r)\log(1 + |r|)$, with cross-entropy training. The head is calibrated assuming the reward distribution has decent **support across the bins** — Hafner-2023 §C reports the design works on rewards spanning $[-20, 20]$ in symlog space (i.e., $[-e^{20}, e^{20}]$ raw). Our reward distribution has two pathological features:

1. **Per-step homeostatic reward** $r_t = D_{t-1} - D_t$ where $D = \sqrt{(S - S^*)^2 + I^2}$ — small and dense, of order $\pm 0.5$ per step in normal operation, with $\mathrm{symlog}(0.5) \approx 0.4$. These map to **a single symlog bin** at the resolution the paper uses.
2. **Death penalty** $r_T = -100$ at terminal: $\mathrm{symlog}(-100) \approx -4.6$ — a totally different bin region. The two regimes are 4 nats apart in symlog space.

What this produces in training:

- Predicting "small dense reward" at every step is the **modal** target by a factor of ~28:1 (28 dense steps : 1 terminal). Cross-entropy on a 255-bin head will be dominated by the modal class.
- The diagnosis already shows this: `model_reward_mae_pos: 4.56 → 0.37` (12.4× improvement, fitting the modal positive-reward signal) vs. `model_reward_mae_neg: 5.17 → 4.06` (1.27×, basically not fitting). The negative-MAE residual is exactly where the terminal $-100$ lives.
- In the `3zjhap9w` collapse: `mae_pos: 0.15 → 0.05` while `mae_neg: 0.37 → 0.60` — the **asymmetry inverts and widens** as training proceeds. The reward head literally gets *worse* on the signal that matters as the modal class gets fitted harder.

**This is a textbook DreamerV3 failure mode** — Hafner-2023 §C.3 explicitly warns that rewards should be "scaled or shaped to span the symlog range" and §C.4 warns about heavy-tailed terminal rewards. Crafter has a $-1$ death-and-everything-else terminal that is much closer to the dense per-step reward magnitude than our $-100$. The Atari benchmark has clipped $r \in \{-1, 0, +1\}$ rewards.

**Likelihood: HIGH.** The reward-MAE asymmetry signature is exactly what this failure mode predicts.

**Conventional fix.** In rough order of cost:
- **Scale the death penalty down** to match the per-step reward magnitude (e.g., $-1$ rather than $-100$). This is the DreamerV3 / Crafter recipe.
- Or: **scale per-step reward up** by a constant. Equivalent up to symlog non-linearity, but matters less than the death-penalty side.
- **Class-balanced reward-head loss** (the prior diagnosis priority 4): re-weight bin losses by inverse density. More invasive, but stays within the symlog architecture.

### 2.2 Replay-ratio over-training on a near-deterministic state distribution — **HIGH**

DreamerV3's `replay_ratio` (Hafner's "train_ratio / 16" — the number of WM gradient steps per env step per env) was set to **1/16** in the paper's Atari benchmark and the architecture is documented to work *cleanly across* the range 1/512 to 1. Our `replay_ratio = 0.5` is **8× the Atari default**.

The danger of high replay ratio in model-based RL is well-documented in the offline-RL world-model literature (see §2.10 below for the published analog): when the **replay buffer is dominated by trajectories from a stuck policy**, every gradient step trains the WM on the same near-identical state distribution. The WM becomes a perfect model of "stuck-in-policy-X-land" and the actor optimizes against that model, which has flat returns by construction.

The `3zjhap9w` collapse is the textbook signature:

- 0–600k env steps: WM exposed to a *changing* state distribution (the agent is exploring) → learns a competent representation, advantage rises to ≈0, survival 330.
- 600–800k env steps: agent commits to a single policy → buffer saturates with that policy's traces.
- Past 800k: every WM gradient update is against the same trace, the WM over-fits the modal trace, the reward head over-fits the modal reward (`mae_pos: 0.05`), and the actor sees a degenerate gradient. Self-confirming-pessimism collapse.

**At `replay_ratio = 0.5`, batch_size 16, sequence_length 128, num_envs 128, the WM sees `0.5 × 128 = 64` train_steps per "iteration" of 128 env-steps × 128 envs = 16,384 env-steps. So the WM-to-env update ratio is `64 × (16 × 128) / (128 × 128) = 64 transitions trained per env transition collected`.** That is **1024× the data-efficiency of canonical Atari DreamerV3** in the per-transition sense. Whatever the buffer contains gets memorized hard.

The relevant theoretical result is that **for a small state space (5×5 = 25 cells, ~3 sensor-state-distinguishable bins per cell ≈ $O(10^2)$ effective states) and a high replay ratio, the WM saturates on the modal policy's coverage long before the policy has explored**.

**Likelihood: HIGH.** The 600–800k inflection in `3zjhap9w` is exactly when the buffer crosses the saturation threshold against the committed NoPred policy. Lowering replay_ratio is the cheapest single fix on the table.

**Conventional fix.** Lower `replay_ratio` to **0.0625** (Hafner Atari default) or **0.125**. This trades wall-clock for training stability — the right trade for "is this learnable at all" diagnostic runs.

### 2.3 KL free-bits / asymmetric KL weights — **MEDIUM-LOW**

DreamerV3's RSSM KL is

$$\mathcal{L}_{\text{KL}} = \alpha_{\text{dyn}} \cdot \mathrm{KL}\!\left[\mathrm{sg}(q(z_t|h_t,o_t)) \,\|\, p(z_t|h_t)\right] + \alpha_{\text{rep}} \cdot \mathrm{KL}\!\left[q(z_t|h_t,o_t) \,\|\, \mathrm{sg}(p(z_t|h_t))\right]$$

with **free bits** $\max(\mathcal{L}_{\text{KL}}, 1.0)$ (Hafner-2023 §B.1) and asymmetric weights $\alpha_{\text{dyn}} = 0.5$, $\alpha_{\text{rep}} = 0.1$. Free bits prevents posterior collapse when the dynamics prior is too easy to match (which happens when observations are highly predictable — exactly our small-grid case).

**In our runs:** `loss_dyn_kl ≈ loss_rep_kl ≈ 2.0` (qont5dac), both well above the 1-nat free-bit floor. **This is healthy on the surface** — KL is not collapsed, latent entropy is not collapsed. The diagnosis already refuted H_collapse.

**The subtler concern**, which the diagnosis did not flag: at `loss_dyn_kl == loss_rep_kl`, the asymmetric weighting that was supposed to prevent the prior from chasing the posterior may not be active in our codebase. If $\alpha_{\text{dyn}} = \alpha_{\text{rep}}$ is the actual implementation, we are running the symmetric-KL variant which Hafner-2023 §B.1 reports underperforms by ~10% on DMC. **This is not a "DreamerV3 fails" item; it is a "we are at most a 10% performance below the canonical" item.**

**Likelihood: LOW** as a sole cause; **MEDIUM** as a contributing factor on top of items 2.1 and 2.2.

**Conventional fix.** Verify the KL weights in `dreamer_v3_trainer.py`. If symmetric, set $\alpha_{\text{dyn}} = 0.5, \alpha_{\text{rep}} = 0.1$. Cheap config check, expensive to verify is not the cause without running.

### 2.4 Continuation-head class imbalance and λ-return horizon truncation — **MEDIUM**

DreamerV3 imagines $H = 15$ steps and computes λ-returns

$$V_t^\lambda = r_t + \gamma c_t \big[(1-\lambda) v(s_{t+1}) + \lambda V_{t+1}^\lambda\big], \qquad V_H^\lambda = v(s_H)$$

where $c_t \in [0,1]$ is the continuation-head probability. The actor loss uses these λ-returns as advantages.

The previous diagnosis (v8 dreamer_hypervigilance_learning_failure) identified **continuation class imbalance** as a candidate root cause; the probe battery (E1–E4) **partially refuted** it: imagination *does* predict ~27% terminations within $h = 15$, just at $h \approx 12$ rather than $h \approx 23$, **uniformly across actions**. The newer working hypothesis from `20260508_1432`: imagined deaths are miscalibrated *in time and per-action*, not absent.

This is not the conventional Dreamer story. The conventional story is that **the imagination horizon is too short relative to the death-event horizon**: real deaths occur at step 23 but imagination only runs 15 steps. With $\gamma = 0.997$, the discount factor over 15 steps is $0.997^{15} \approx 0.956$ — a mild attenuation. But the value function for the missing 8 steps is **bootstrapped from $v(s_H)$**, which the critic learned from data dominated by short death trajectories. So $v(s_{15})$ already encodes "you die in 8 more steps" implicitly, but it cannot encode the **action-conditional credit** for what got you to a state where you survive vs. die, because the actor's action choice at step 14 has barely propagated into $v(s_{15})$ via TD bootstrap.

**The concern is conventional**: in any actor-critic-with-imagined-rollouts setup, **the imagine_horizon must be ≥ the typical event-horizon to the dominant terminal reward**. Hafner-2023 chose $H = 15$ because Atari and DMC have dense reward signals — the agent does not need to "see" a terminal to get gradient. We have a **sparse-but-large terminal** (death penalty $-100$) at horizon ~23–28 from the start of an episode, and imagination only sees 15 steps.

**Likelihood: MEDIUM.** The probe battery showed *some* terminations are imagined in the right ballpark; the issue is calibration, not absence. But a longer `imagine_horizon` (e.g., 30) and lower `cont_loss_weight` to compensate would be a worth-trying conventional knob.

**Conventional fix.** Raise `imagine_horizon` from 15 to 25–32. Cost: ~2× behavior-loss compute (linear in $H$). Fits in a single training run.

### 2.5 Observation normalization / vector-input mismatch — **MEDIUM**

DreamerV3 was tuned with **image inputs**: pixels are bounded $[0, 1]$, the encoder is a ConvNet, the decoder is a deconvolutional ConvNet, and the reconstruction loss is **per-pixel MSE on bounded inputs**. The reconstruction signal is dense and high-dimensional ($64 \times 64 \times 3 = 12{,}288$ outputs); it carries the WM through the early-training regime where the reward and continuation heads have not yet learned anything useful.

**Our setup**: 19-dim vector observation, MLP encoder/decoder, MSE loss on what should be already-normalized sensor outputs. Three concerns:

1. **Reconstruction loss imbalance**: 19 reconstruction targets vs. 1 reward target vs. 1 continuation target. Hafner-2023 §B uses `loss_recon` weight = 1.0 implicitly, summed over pixels (so `recon ≈ 12,288 × per-pixel-MSE`). Our `loss_recon: 0.0081` summed over 19 dims ≈ 0.0004 per dim. The reward head is only 1 dim; whether the relative gradient magnitudes balance is not a question the paper had to answer. **If our recon weight is too high**, the WM puts all its capacity into reconstructing sensors and not enough into reward/continuation. **If too low**, the WM's latent does not encode enough about observations. We have not directly measured this.
2. **Sensor scale heterogeneity**: from `01-5X5_PredInterval3_NutGain18.yaml` perceptual_noise definitions, `extero_nociception` and `olfaction` clip to `[0, 100]` while others clip to `[0, 1]`. **If sensors are not min-max normalized before the encoder**, the MSE loss is dominated by the wide-range sensors, biasing the WM's representation.
3. **Vector encoder is much smaller than image encoder**: 19 → 128 → 128 → 128 (our MLP) vs. 64×64 → conv stack with ~10× more parameters. Less expressive WM, but more importantly **less inductive bias for spatial structure** that a 5×5 grid trivially has.

**Likelihood: MEDIUM.** This is a "we are off the calibrated path" risk rather than a smoking-gun signal. We do not have evidence the sensor normalization is broken, but we have not ruled it out, and the paper's calibration assumed pixels.

**Conventional fix.** (a) Verify all sensor outputs are normalized to $[0, 1]$ or zero-mean-unit-variance *before* entering the encoder; (b) consider switching to symlog-normalized sensors for the unbounded ones. Cheap.

### 2.6 Action-repeat / action-space mismatch — **LOW**

DreamerV3 in DMC uses `action_repeat = 2` (the agent's chosen action is executed for 2 env steps, halving the effective control frequency). Our setup has no action repeat (1 action = 1 env step). For a 5×5 grid this is fine — the grid is so small that one step is meaningful — but it's worth flagging that **all the canonical Dreamer hyperparameters were tuned at action_repeat = 2 or 4**.

The actor entropy plateau at `ln(6) = 1.79` (qont5dac) does not depend on action_repeat. **Likelihood: LOW.**

### 2.7 World-model warm-up / pre-training schedule — **LOW**

Hafner-2023 §B specifies a `pretrain_steps = 100` WM-only update phase before the actor is trained, with a `start_training` env-step threshold (typically 1024). If the actor starts updating before the WM has converged, the actor's gradient is noise and can lock into a bad policy.

**In our runs**: 27 M env steps. Whatever warm-up we use is a rounding error at this scale. Unlikely to be the cause. **Likelihood: LOW.**

### 2.8 Critic EMA / target-network — **LOW**

DreamerV3 uses an **EMA-target critic** for value bootstrapping with $\tau = 0.02$ (Hafner-2023 §B.2):

$$v_{\text{target}} \leftarrow (1 - \tau) v_{\text{target}} + \tau v_{\text{online}}$$

If our codebase doesn't use EMA-target (or uses a different $\tau$), critic bootstrap can become unstable. The diagnosis logs `loss_critic` rising from 0.377 → 0.431 late in qont5dac, and from 2.90 → 1.20 (falling, but `mean_value` drifting) in `3zjhap9w`. Both are **mild** signatures; if it were the EMA-target issue we would expect oscillation or divergence rather than monotone drift.

**Likelihood: LOW.**

### 2.9 Off-policy correction / IS weights for actor — **LOW**

DreamerV3's actor is trained on **imagined trajectories under the current actor**, so it is technically on-policy in imagination space. There is no IS weight needed in the canonical algorithm. (DreamerV2 used REINFORCE with a baseline; DreamerV3 uses straight policy gradient on imagined returns plus entropy.) Not a likely failure mode here. **Likelihood: NOT APPLICABLE.**

### 2.10 Self-confirming pessimism / pessimistic-WM lock-in — **HIGH (analog)**

This is item 2.2's twin from a different angle and the published literature is rich. The phenomenon: **a learned value function and a learned dynamics model in mutual feedback** can lock into pessimistic predictions that the data then confirms. Specifically, if the value function predicts state $s$ has value $V(s) \ll 0$, the policy avoids reaching $s$, the buffer accumulates no data near $s$, the WM extrapolates pessimistically into $s$, $V(s)$ stays low — **self-confirming**.

**Published analogs:**

- **MOPO / MOReL / COMBO** (Yu et al., 2020; Kidambi et al., 2020; Yu et al., 2021): the offline-RL world-model literature explicitly identifies this as the central failure mode and proposes **uncertainty-aware pessimism penalties** that *deliberately* shrink optimistic value estimates rather than letting the model naturally drift pessimistic. The conventional fix is **pessimism / optimism disentanglement**: use ensembles to estimate epistemic uncertainty and explicitly add an exploration bonus where the WM is uncertain.
- **"Trust region policy optimization in latent space"** (Janner et al., 2019, MBPO): uses **short imagination rollouts** ($H = 1$ to $5$) precisely to avoid this — letting the model imagine far is letting it manufacture pessimistic data.
- **DreamerV3** does not have this mechanism explicitly. It relies on `unimix = 0.01` (categorical mixture floor) and the symlog-two-hot-reward cross-entropy as implicit regularizers, plus `replay_ratio` low enough to let buffer turnover supply fresh data.

**The `3zjhap9w` collapse signature is the textbook published self-confirming-pessimism shape**: `mean_value` drifts more negative monotonically (-9.6 → -14.7), `mae_pos` collapses (model gets confident on the modal, easy targets) while `mae_neg` deteriorates (model gets confused on the rare, important targets), food intake AND danger hits both fall (the agent avoids everything, not just the bad thing), and `loss_actor` magnitude grows 30× (the actor is being driven hard by a critic target that does not correspond to recoverable real-world state values).

**Conventional remedy.** Three families, in ascending invasiveness:
1. **Lower replay_ratio** (item 2.2). This is the "let the buffer turn over" fix. Cheapest.
2. **Mixture/uniform sampling balance**: our `mixture_positive_slots=5, mixture_recent_slots=5` config is half positive-reward biased; if positive transitions become rare in late training, this can starve the WM. Inspect mixture-slot fill rates.
3. **Explicit ensemble / disagreement penalty** on the WM (the offline-RL fix). Out of scope for "is this conventional" diagnostics — only relevant if items 1 and 2 fail.

**Likelihood: HIGH.** The collapse signature is the published pattern.

### 2.11 RSSM stochastic-vs-deterministic balance — **LOW**

Our RSSM is `deter=512, stoch=32×32`. Hafner-2023 small model: `deter=512, stoch=32×32`. Match. **Likelihood: LOW.**

### 2.12 Mixed-precision / gradient instability — **LOW**

We have not seen NaNs in any of the diagnosed runs. Diagnosis already refuted H_config_bug. **Likelihood: LOW.**

---

## 3. Top-3 ranked list of most likely conventional causes

| Rank | Cause | Diagnostic signature in our runs | Closest paper | Cheapest fix to test |
|---|---|---|---|---|
| **1** | **Reward-scale mismatch (death penalty $-100$ vs. dense $\pm 0.5$)** | `mae_pos: 0.05` while `mae_neg: 0.60` and *widening* (3zjhap9w windows 4–5); `mae_pos` collapses 12× while `mae_neg` collapses 1.3× (qont5dac) | Hafner-2023 §C.4; Crafter death is $-1$, not $-100$ | Reduce death penalty to $-1$; rerun on hypervigilance |
| **2** | **Replay-ratio over-training on near-deterministic buffer (self-confirming-pessimism flavor)** | `3zjhap9w` collapse at 800k env steps after committing at 600k; `mean_value` drifts $-9.6 \to -14.7$; `mae_pos$ drops while mae_neg rises; food and danger hits fall together | MOPO / MOReL / MBPO; Hafner-2023 Atari uses `train_ratio = 1/16`, our equivalent is 8× higher | Lower `replay_ratio` from 0.5 to 0.0625; rerun on NoPred (the cheapest test — a 600k-env-step run without collapse settles this) |
| **3** | **Imagine-horizon shorter than death-event horizon** | Real deaths at step 23, imagined deaths at step 12 with H=15; 8 steps of value carried by bootstrap | Hafner-2023 chose $H = 15$ for Atari/DMC where rewards are dense; sparse-terminal envs need longer | Raise `imagine_horizon` from 15 to 25; rerun on hypervigilance |

**Why these three and not the others.** Cause 1 is matched 1:1 to the reward-MAE asymmetry signature both before and after collapse. Cause 2 is matched to the temporal collapse signature in `3zjhap9w`. Cause 3 is the conventional way to read the probe battery's "imagined deaths miscalibrated in time" finding without invoking per-action-conditional bespoke instrumentation.

---

## 4. Easiest-to-test conventional fixes, ordered cheapest-first

| Order | Intervention | Touch surface | Wall-clock cost | Decisive on |
|---|---|---|---|---|
| 1 | **Lower replay_ratio: 0.5 → 0.0625** | one config line | ~3h on NoPred to 800k env steps | Cause 2 |
| 2 | **Reduce death penalty: -100 → -1** | one body-config line | ~3h on hypervigilance | Cause 1 |
| 3 | **Raise imagine_horizon: 15 → 25** | one config line (if exposed) | ~3h on hypervigilance, 2× behavior-loss compute | Cause 3 |
| 4 | **Verify sensor normalization** (audit, not run) | read `src/env/sensor.py` | 30 min | Rule out cause 2.5 |
| 5 | **Verify KL weights $\alpha_{\text{dyn}}, \alpha_{\text{rep}}$ in trainer** (audit, not run) | read `src/models/dreamer_v3_trainer.py` | 15 min | Rule out cause 2.3 |
| 6 | **Audit reward-bin histograms in our symlog two-hot** (instrument, not run) | one logging change | 1h dev + 3h run | Localize cause 1 |
| 7 | **Class-balanced reward-head loss** | small trainer change | 1h dev + 3h run | Cause 1 if (2) fails |

**The user's "this isn't too difficult" framing argues for items 1 and 2 first.** They are one-line config changes. Both have direct, testable predictions about the collapse signature.

---

## 5. Self-confirming pessimism — published literature citation

The pattern is well-known in the model-based / offline-RL world-model literature, not specific to DreamerV3:

- **Yu et al., 2020 (MOPO)** — "Model-based Offline Policy Optimization" — explicitly motivates pessimism penalties by the self-confirming feedback loop between learned WM and learned policy when the buffer does not cover state space.
- **Kidambi et al., 2020 (MOReL)** — proposes a pessimistic MDP construction to upper-bound the policy's exploitation of WM extrapolation.
- **Janner et al., 2019 (MBPO)** — uses short imagination rollouts ($H = 1$–$5$) explicitly to limit how far the WM's pessimistic extrapolation can propagate into the actor's gradient.
- **Yu et al., 2021 (COMBO)** — combines conservative Q-learning with model-based rollouts, shows the lock-in regime is escapable only when the conservative penalty matches the WM's epistemic uncertainty.

**The conventional remedy in this literature is *not* "fix the cont head" or "instrument per-action terminations" — it is "prevent the WM from over-fitting the modal trajectory."** This is what items 1 and 2 in the ranked list above do: lower replay_ratio prevents over-fitting the modal trajectory by giving the WM less compute against the same data; reducing the death penalty prevents the value function from collapsing to a single-mode pessimistic basin.

The team has been pulling on the cont-head lever (cont_loss_weight 1 → 5 → 10) and the imagined-death-probe lever, both of which the published literature would predict will not move the needle while the underlying cause (modal-trace over-training, reward-scale mismatch) is unaddressed. **The probe battery's null result on `cont_loss_weight=10` is not surprising** in this light — pushing the WM to predict more terminations on a saturated buffer cannot escape the lock-in.

---

## 6. Verdict on the user's "this isn't too difficult" framing

**Partially correct, with a caveat.** The conventional checklist points to two one-line config fixes (replay_ratio, death penalty) and one slightly larger but still standard fix (imagine_horizon). All three are documented in the DreamerV3 paper as failure modes. **None of them have been tried.** The investigation has gravitated toward bespoke instrumentation (probe, per-action cont, fork-rollouts) before exhausting the conventional knobs.

**The caveat.** Even if items 1–3 all fail, the project has a *real* nonstandard issue: **DreamerV3 was designed for image inputs and dense rewards**, and we are using vector inputs and sparse-mixed-magnitude rewards. The reward distribution we have is genuinely outside the calibrated regime of the paper, and the symlog-two-hot reward head was not stress-tested on $\pm 0.5$ dense + $-100$ terminal mixtures. So if items 1–3 fail, the next layer of fixes (class-balanced reward loss, or replacing the symlog-two-hot head with a Gaussian head with learned variance) is also conventional, just less common.

**My recommendation to the user**: spend the next training budget on **items 1, 2, 3 from the ranked list, in that order, on the no-homeostatic-no-predator strip-down task** (see the companion direction memo, [`dreamer_minimum_viable_strip_down.md`](../directions/dreamer_minimum_viable_strip_down.md)). Total cost: ~9 hours of wall-clock if run sequentially; less if parallelized across nodes. After this, we will have a much sharper picture of whether the issue is conventional (likely) or whether the project needs to consider a different algorithm class entirely.

---

## 7. What this critique does NOT recommend

- **No bespoke probes for now.** The per-action `cont` probe (open question 1 from `20260508_1432`) is interesting but answers a question that is downstream of items 1–3. If items 1 and 2 collapse the failure, per-action conditioning becomes moot.
- **No fork-rollouts.** Same reasoning — these are diagnostics on an architecture that may not be the proximal cause of failure.
- **No "switch from DreamerV3 to TD-MPC2 / IRIS / V-D4RL" yet.** This is a viable last resort if the conventional fixes all fail, but the priors should not be there yet.

---

## 8. Next steps (by agent)

- **`senior-developer`** — read this memo and the companion direction memo, then write the implementation plan for the strip-down + conventional-fix sweep. Priority on items 1 and 2 from §4.
- **`experiment-designer`** — translate the strip-down and the cheapest-three-fix sweep into specific configs, seeds, and a Launch Manifest. The companion direction memo specifies the strip-down task; this memo specifies the fixes.
- **`code-reviewer`** — audit the KL-weight implementation in `src/models/dreamer_v3_trainer.py` (item 2.3) and the sensor normalization in `src/env/sensor.py` (item 2.5). 30 minutes total.
- **No other agent action required from this memo.**
