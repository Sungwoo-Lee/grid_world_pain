---
title: "What mainstream PPO implementations do to returns and advantages — a source-code survey"
topic: rl_conventions
status: active
created: 2026-09-01
last_updated: 2026-09-01
---

# What mainstream PPO implementations do to returns and advantages

## Purpose (plain language)

Every actor-critic algorithm produces two numbers per timestep before it can take a gradient step.
The **return** (sometimes called the value target) is "how much reward actually followed this state",
and it is what the critic network is trained to predict. The **advantage** is "how much better than
expected this action turned out", and it is what multiplies the policy gradient. Both quantities
usually come out of the environment in awkward units — an environment that pays 1 point per surviving
step gives returns in the hundreds, one that pays a sparse 0/1 bonus gives returns near zero — so
implementations rescale them before use. **Which** of the two gets rescaled, and **how**, turns out to
be near-universal across the field, and this document establishes what that convention is by reading
the actual source code of ten implementations rather than by citing blog posts.

The short answer, verified in every library checked: **the advantage is standardised (shifted to mean
zero, divided by its standard deviation), and the critic's regression target is left in raw reward
units.** Where a library does rescale the critic's target — two of the ten do — it uses a slow-moving
running statistic and undoes that rescaling before the advantage is formed, so the advantage stays in
raw units. Nobody in a production library recomputes a mean and standard deviation from the current
batch and regresses the critic onto the result. (Separately, and perhaps surprisingly, the largest
ablation study ever run on on-policy RL found that advantage standardisation itself makes little
difference to final score — it is near-universally *implemented* but only weakly *evidenced*. See
[§ What the ablation papers actually measured](#what-the-ablation-papers-actually-measured).)

This matters here because this project's recurrent-PPO trainer has a Monte-Carlo branch
([`src/models/recurrent_ppo_trainer.py`](../../../../src/models/recurrent_ppo_trainer.py), lines
370–380) that does exactly the reverse: it z-scores the returns per batch, hands the z-scored returns
to the critic as its regression target, and then leaves the advantage un-normalised. The GAE branch
in the *same function*, ten lines below, does the conventional thing. This survey adjudicates whether
the Monte-Carlo branch has any precedent. It does — but only in two toy teaching scripts, and the
specific reason those scripts get away with it does not hold here.

## What was surveyed

Ten implementations, fetched from their public repositories on 2026-09-01 and read directly. Every
claim below carries a file and a line number or a quoted snippet. Nothing in the per-implementation
table is recalled from memory; everything was verified from source. Repository HEAD commits at the
time of fetch are recorded in [§ Provenance](#provenance).

## The three questions

For each implementation:

1. **Critic target** — is the value function regressed onto a raw return, or a normalised one?
2. **Advantage normalisation** — is the advantage standardised, and is that the default?
3. **Upstream reward/return normalisation** — is there a running estimate of the return scale applied
   to the rewards before anything else?

## Results table

| Implementation | Critic's regression target | Advantages normalised? | Upstream reward/return normalisation |
|---|---|---|---|
| **Stable-Baselines3** (PPO) | **Raw.** `value_loss = F.mse_loss(rollout_data.returns, values_pred)` — `ppo/ppo.py:244`; `returns` built as `self.returns = self.advantages + self.values` in `common/buffers.py:438`, never rescaled | **Yes, default on.** `normalize_advantage: bool = True` (`ppo/ppo.py:92`); `advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)` (`ppo/ppo.py:219`), per minibatch | **Optional wrapper.** `VecNormalize(norm_reward=True)` divides reward by `sqrt(ret_rms.var)` using a running estimate of the *discounted-return* std (`common/vec_env/vec_normalize.py:211–212, 256`). Scale-only — no mean subtraction |
| **CleanRL** (`ppo.py`, `ppo_continuous_action.py`) | **Raw.** `v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()` — `cleanrl/ppo.py:282`; `returns = advantages + values` at `:231` | **Yes, default on.** `norm_adv: bool = True` (`:57`); `mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)` (`:262`), per minibatch | **Yes in the continuous script.** `env = gym.wrappers.NormalizeReward(env, gamma=gamma)` — `ppo_continuous_action.py:99`. Absent from the discrete `ppo.py` |
| **OpenAI Baselines** (`ppo2`) | **Raw.** `vf_losses1 = tf.square(vpred - R)` where `R` is fed `returns` verbatim — `ppo2/model.py:71, 145`; `mb_returns = mb_advs + mb_values` in `ppo2/runner.py`, unrescaled | **Yes, unconditionally.** `advs = returns - values` then `advs = (advs - advs.mean()) / (advs.std() + 1e-8)` — `ppo2/model.py:135–139`. No flag to turn it off | **Yes, the canonical one.** `VecNormalize(ret=True)`: maintains `self.ret = self.ret * gamma + rews`, updates `ret_rms` on it, then `rews = np.clip(rews / np.sqrt(ret_rms.var + eps), -10, 10)` — `common/vec_env/vec_normalize.py`. This is the original "reward scaling" that Engstrom et al. later ablated |
| **OpenAI Spinning Up** (PPO, VPG) | **Raw — and this is the closest analogue to our MC branch.** `self.ret_buf[path_slice] = core.discount_cumsum(rews, self.gamma)[:-1]` where `rews = np.append(self.rew_buf[path_slice], last_val)` — i.e. a Monte-Carlo discounted sum *seeded with a bootstrap value*, exactly our construction (`ppo/ppo.py:59, 67`). It is handed to the critic untouched: `return ((ac.v(obs) - ret)**2).mean()` (`:248`) | **Yes, unconditionally,** and the code calls it out by name: "the next two lines implement the advantage normalization trick", `adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf); self.adv_buf = (self.adv_buf - adv_mean) / adv_std` — `ppo/ppo.py:79–81`. Statistics are pooled across MPI workers | **No** |
| **RLlib (Ray)** | **Raw.** `Postprocessing.VALUE_TARGETS` set to `advantages + vf_preds` (GAE branch) or to the raw discounted return (non-GAE branch) — `rllib/evaluation/postprocessing.py:128–139` — and consumed as-is: `vf_loss = torch.pow(value_fn_out - batch[Postprocessing.VALUE_TARGETS], 2.0)` (`algorithms/ppo/ppo_learner.py:99`) | **Yes.** New stack: `module_advantages = (module_advantages - module_advantages.mean()) / max(1e-4, module_advantages.std())`, with the comment "Standardize advantages (used for more stable and better weighted policy gradient computations)" — `connectors/learner/general_advantage_estimation.py:146–150`. Old stack: `train_batch = standardize_fields(train_batch, ["advantages"])` — `algorithms/ppo/ppo.py:495`. Note it standardises `advantages` **only** | **Not built into PPO** (left to env wrappers) |
| **Tianshou** (A2C / PPO base) | **Raw by default; optionally divided by a *running* return-std, never per-batch z-scored.** `batch.returns = unnormalized_returns` unless `return_scaling` is on, in which case `batch.returns = unnormalized_returns / np.sqrt(self.ret_rms.var + self._eps)` — `tianshou/algorithm/modelfree/a2c.py:147–150`. Crucially the value predictions are **de-normalised back to raw units before the advantage is computed** (`v_s = v_s * np.sqrt(ret_rms.var + eps)`, `:135–136`), and the code explicitly refuses to subtract the mean: *"when normalizing values, we do not minus self.ret_rms.mean to be numerically consistent with OPENAI baselines' value normalization pipeline. Empirical study also shows that 'minus mean' will harm performances"* (`:130–133`) | **Yes, default on.** `advantage_normalization: bool = True` (`modelfree/ppo.py:28`), applied per minibatch at `:184–185` | **`return_scaling`** is exactly that — an EMA/running return-std applied to returns and values, off by default |
| **torchrl** (`ClipPPOLoss`, `GAE`) | **Raw.** `target_return = tensordict.get(self.tensor_keys.value_target)` and regressed directly — `objectives/ppo.py:850`, and `GAE` writes `value_target` unrescaled (`objectives/value/advantages.py:2247`) | **Yes, opt-in on the loss (`normalize_advantage: bool = False`, `objectives/ppo.py:492`) or on the estimator (`average_gae`, `advantages.py:2005`).** torchrl's own reference scripts turn it on: `normalize_advantage=True` in both `sota-implementations/ppo/ppo_mujoco.py:187` and `ppo_atari.py:109` | **No reward wrapper in the loss.** But torchrl offers `advantage_norm` — a *stateful* `ValueNorm` (e.g. `PercentileValueNorm`, "for DreamerV3-style return normalization") whose statistics are updated from the **value targets** and which then divides **the advantage**: `return advantage / self.advantage_norm.scale()`, "without re-centering (the advantage is already centred by the value baseline)" — `objectives/ppo.py:178–192, 991–1010`. The critic target itself is still raw |
| **sheeprl** (Eclectic-Sheep, `algos/ppo`) | **Raw.** `gae()` returns `returns = advantages + values` (`utils/utils.py:99`) and `value_loss(...)` does `F.mse_loss(values_pred, returns)` — `algos/ppo/loss.py:55` | **Opt-in, default OFF.** `if cfg.algo.normalize_advantages: batch["advantages"] = normalize_tensor(batch["advantages"])` — `algos/ppo/ppo.py:64–65`; `normalize_advantages: False` in `configs/algo/ppo.yaml`. `normalize_tensor` is the standard `(x - mean) / (std + eps)` (`utils/utils.py:126`) | **No** |
| **MAPPO** (`marlbenchmark/on-policy`) — *the one that does normalise the target* | **Normalised — but by a slow running statistic, and with de-normalisation before the advantage.** `error_original = self.value_normalizer.normalize(return_batch) - values` (`r_mappo.py:67`) with `ValueNorm` an EMA (`beta=0.99999`) running mean/var, debiased (`utils/valuenorm.py`). The advantage is then computed in **raw** units: `advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])` (`r_mappo.py:180`) | **Yes, separately and afterwards.** `advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)` — `r_mappo.py:187` | The `ValueNorm`/`PopArt` normaliser *is* the return normalisation, held as a persistent running statistic rather than recomputed per batch |
| **pytorch/examples** `reinforce.py` and `actor_critic.py` — *the toy scripts* | `reinforce.py`: **no critic at all.** `actor_critic.py`: **z-scored return used as the critic target** — see [§ The REINFORCE origin](#the-reinforce-origin) | `actor_critic.py`: **no** — `advantage = R - value.item()` is used as-is (`:115`) | **No** |

**The pattern is unanimous.** Grepping all ten fetched sources for a per-batch z-score applied to
returns returns exactly two hits, and both are in `pytorch/examples`:

```
pytorch_ac.py:112:       returns = (returns - returns.mean()) / (returns.std() + eps)
pytorch_reinforce.py:71: returns = (returns - returns.mean()) / (returns.std() + eps)
```

No library — Stable-Baselines3, CleanRL, Baselines, Spinning Up, RLlib, Tianshou, torchrl, sheeprl —
per-batch z-scores the critic's regression target. Every one of them standardises the advantage
(unconditionally in Baselines, Spinning Up and RLlib; default-on in SB3, CleanRL and Tianshou;
opt-in in torchrl and sheeprl).

### A telling side-note from Tianshou

Tianshou is the one library that ships an explicit "standardise the returns" switch — and it exists
**only in REINFORCE, the algorithm with no critic**, is **off by default**, uses a **running** statistic
rather than a per-batch one, and carries this warning in its own docstring
(`tianshou/algorithm/modelfree/reinforce.py:263–265`):

> `:param return_standardization: whether to standardize episode returns by subtracting the running
> mean and dividing by the running standard deviation. **Note that this is known to be detrimental to
> performance in many cases!**`

The same library's actor-critic base class (`a2c.py`) offers no such option — there, the return
rescaling that exists (`return_scaling`) is scale-only and is symmetrically undone on the value
predictions before the advantage is formed. The library's authors evidently considered standardising
returns, restricted it to the critic-free case, and warned against it even there.

## The REINFORCE origin

The hypothesis that this project's Monte-Carlo branch was copied from
[`pytorch/examples/reinforcement_learning/reinforce.py`](https://github.com/pytorch/examples/blob/main/reinforcement_learning/reinforce.py)
is **confirmed as to the line, and refined as to the file**. The z-score line does appear there, but
the *closer* match — the one that also uses the z-scored return as a critic target and leaves the
advantage un-normalised — is the sibling script `actor_critic.py` in the same directory.

`reinforce.py`, lines 63–76 (verified):

```python
def finish_episode():
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
```

`actor_critic.py`, lines 99–121 (verified) — this is the project's pattern almost verbatim:

```python
    R = 0
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)

    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
```

Same three steps, same order: z-score the returns, subtract the critic's prediction to get an
un-normalised advantage, regress the critic onto the z-scored return.

### Why it is harmless in `reinforce.py`

In `reinforce.py` **there is no critic**. The z-scored return is used for one thing only: as the
scalar coefficient multiplying `-log_prob`. Standardising it does two legitimate things at once.
Subtracting the mean is a **constant baseline** — the classic variance-reduction trick, and it is
provably unbiased for the policy gradient because `E[∇log π] = 0`, so subtracting any quantity that
does not depend on the action leaves the gradient's expectation unchanged. Dividing by the standard
deviation is a **per-episode learning-rate rescale** — it changes the step size, not the direction.
Nothing downstream is asked to *predict* the z-scored quantity, so no prediction problem is
corrupted. It is a policy-gradient coefficient, and coefficients may be rescaled freely.

### Why it is not harmless in an actor-critic with a bootstrapped value target

The moment a critic exists and is regressed onto the same z-scored number, three things break, and
the third is specific to this project.

**1. The regression problem becomes non-stationary for reasons unrelated to learning.** The critic is
asked to predict `(G − μ_batch) / σ_batch`, where `μ_batch` and `σ_batch` are recomputed from scratch
on every rollout. Two states with identical true returns get different targets in different batches;
a batch of uniformly excellent episodes and a batch of uniformly terrible episodes both get mapped to
mean zero. **All absolute value information is destroyed** — the critic can only ever learn the
*within-batch ranking* of states, never that this batch was better than the last one. In a survival
task where the whole signal of interest is "the agent survives longer now than it did 200k episodes
ago", that is precisely the information being deleted from the critic's target.

**2. The advantage's scale drifts with critic quality.** `advantages = returns − values` is left
un-normalised. Early in training the critic is bad, so this quantity is large; as the critic fits,
it shrinks toward zero. The effective policy-gradient step size therefore decays over training as an
uncontrolled side effect of critic convergence — the exact failure mode advantage standardisation
exists to prevent, and which every library in the table above prevents.

**3. A unit mismatch at the bootstrap seed — the concrete, checkable defect.** In this project's
`compute_mc_returns` (`src/models/recurrent_ppo_trainer.py:95–120`) the reverse scan is *seeded with
the critic's own output*, `bootstrap_value = V(s')`. But the critic has been trained to output
z-scored returns. So a z-scored-unit quantity is used as the initial carry of a discounted sum of
**raw** environment rewards, and the resulting mixed-unit quantity is then z-scored again and fed back
to the critic as its target. The two units are never reconciled anywhere. This is a genuine
scale-consistency bug, not a stylistic disagreement, and it is invisible in `actor_critic.py`
precisely because that script has no bootstrap at all: `R = 0` at line 99, full episodes only, so
every number entering the z-score is in raw reward units.

Note also that `actor_critic.py` is a ~180-line teaching script whose success criterion is solving
CartPole-v1, with one gradient step per episode and no minibatching, no clipping, and no
bootstrapping. It gets away with the pattern because CartPole is trivially easy, not because the
pattern is sound. It is a demonstration of `nn.Module` and `log_prob`, not a reference PPO.

## Adjudication of this project's pattern

The code in question, `src/models/recurrent_ppo_trainer.py:370–380` (MC branch):

```python
returns = jax.vmap(compute_mc_returns, ...)(reward, done, terminateds, bootstrap_value, gamma)
returns = (returns - jnp.mean(returns)) / (jnp.std(returns) + 1e-7)   # z-scored per batch
targets = returns                                                     # critic regression target
advantages = returns - trajectories.value                             # NOT re-normalised
```

versus the GAE branch, thirteen lines below in the *same function* (`:391–396`):

```python
advantages = jax.vmap(compute_gae, ...)(...)
targets = advantages + trajectories.value                             # raw
advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1e-8)   # normalised
```

**Verdict: the Monte-Carlo branch has no precedent in any mainstream implementation.** Its two
precedents are `pytorch/examples/actor_critic.py` (exact match, toy CartPole script, no bootstrap)
and `pytorch/examples/reinforce.py` (the z-score line, but no critic, where it is legitimate). The
GAE branch, by contrast, matches the universal convention exactly. The two branches of the same
function disagree with each other, and the MC one is the outlier.

The closest *legitimate* analogue to the MC branch's construction is **Spinning Up's PPO**, which
computes the identical object — a Monte-Carlo discounted sum seeded with a bootstrap value
(`ppo.py:59–67`) — and then does the opposite of what this project does with it: the return goes to
the critic **raw**, and the **advantage** is standardised. If a single reference is wanted for what
this branch should look like, that is it.

### Does anyone normalise the critic's target?

Yes — but never this way, and the differences are the whole point.

| Approach | Statistic | Applied to | Advantage computed in | Output preservation |
|---|---|---|---|---|
| **This project's MC branch** | per-batch mean and std, recomputed every rollout | critic target | z-scored units | none |
| **PopArt** (van Hasselt et al. 2016) | running/EMA over all targets seen | critic target | raw units | **yes** — output-layer weights and bias are rescaled so the un-normalised prediction is unchanged |
| **MAPPO `ValueNorm`** | EMA, `beta = 0.99999`, debiased | critic target | **raw** (`denormalize(value_preds)` before subtraction, `r_mappo.py:180`) | not needed — de-normalisation at the read site |
| **Tianshou `return_scaling`** | running `ret_rms`, **scale only, no mean subtraction** | returns *and* values, symmetrically | raw (values un-normalised first, `a2c.py:135–136`) | symmetric un-normalisation |
| **torchrl `advantage_norm`** | stateful `ValueNorm` / `PercentileValueNorm`, updated from value targets | **the advantage, not the target**; scale only, no re-centering | — | n/a |
| **DreamerV3 / `symlog` + two-hot; Farebrother et al. 2024 HL-Gauss** | fixed, *learning-independent* transform of the target (or a change of loss from regression to classification) | critic target | consistent throughout | the transform is invertible and fixed, so nothing drifts |

Every principled scheme has at least one of: a **slow-moving or fixed** statistic (so the target does
not jitter batch to batch), **explicit de-normalisation** at every site that mixes normalised with raw
quantities, or **output preservation** so the network's function is unchanged when the statistic
moves. The MC branch has none of the three. Per-batch z-scoring is the maximally non-stationary
choice — the statistic changes completely every single rollout.

## What the ablation papers actually measured

Three mechanisms get lumped together as "normalisation" and the literature separates them cleanly.
All quotes below were verified from the arXiv PDFs.

### Engstrom, Ilyas, Santurkar, Tsipras, Janoos, Rudolph & Madry (2020), *Implementation Matters in Deep Policy Gradients* — [arXiv:2005.12729](https://arxiv.org/abs/2005.12729)

They catalogue **nine** code-level optimisations in PPO's reference implementation: value-function
clipping, **reward scaling**, orthogonal initialisation with per-layer scaling, Adam learning-rate
annealing, reward clipping, observation normalisation, observation clipping, tanh activations, and
global gradient clipping.

Their **reward scaling** is the OpenAI Baselines `VecNormalize` mechanism, described verbatim as:
*"the rewards are divided through by the standard deviation of a rolling discounted sum of the rewards
(**without subtracting and re-adding the mean**)"*. Their Appendix A.2 pseudocode makes the divisor a
running std of a discount accumulator, applied to the per-step reward.

**Findings, with the honest caveat.** Compute limits meant they ablated only the *first four*
optimisations (value clipping, reward scaling, initialisation, LR annealing), and the result is
reported as a distributional shift, not a per-optimisation number. Figure 1 caption, verbatim:
*"Our results show that **reward normalization**, Adam annealing, and network initialization each
significantly impact the rewards landscape with respect to hyperparameters, and were necessary for
attaining the highest PPO reward within the tested hyperparameter grid."* So reward scaling **did**
matter — and value-function clipping, conspicuously, did not make that list.

The one numeric table bundles all the optimisations. `PPO-M` is defined as PPO with *"the standard
value network loss, **no reward scaling**, the default network initialization, and Adam with a fixed
learning rate"*, and `TRPO+` is TRPO with PPO's code-level optimisations (≥80 agents per cell,
95% bootstrap CIs):

| | Walker2d-v2 | Hopper-v2 | Humanoid-v2 |
|---|---|---|---|
| PPO | 3292 [3157, 3426] | 2513 [2391, 2632] | 806 [785, 827] |
| PPO-M (no reward scaling, plain init/LR) | 2735 [2602, 2866] | 2142 [2008, 2279] | 674 [656, 695] |
| TRPO | 2791 [2709, 2873] | 2043 [1948, 2136] | 586 [576, 596] |
| TRPO+ (with code-level opts) | 3050 [2976, 3126] | 2466 [2381, 2549] | 1030 [979, 1083] |
| **AAI** — effect attributable to the *algorithm* | 242 | 99 | 224 |
| **ACLI** — effect attributable to *code-level* choices | 557 | 421 | 444 |

Code-level effects exceed algorithmic effects by roughly 2–4× on all three tasks. Headline conclusion,
verbatim from the abstract: code-level optimisations *"are responsible for most of PPO's gain in
cumulative reward over TRPO"*, and *"the clipping mechanism is not necessary to achieve high
performance."*

**On advantage normalisation: they never discuss it.** The word "advantage" appears three times in the
paper, none of them about standardisation. Do not cite this paper on advantage normalisation.

### Andrychowicz et al. (2021), *What Matters In On-Policy Reinforcement Learning?* — [arXiv:2006.05990](https://arxiv.org/abs/2006.05990)

Scale, verbatim: *">50 such 'choices' in a unified on-policy RL framework … We **train over 250,000
agents** in five continuous control environments"* (Hopper, Walker2d, HalfCheetah, Ant, Humanoid; 3
seeds per config, 1–2M steps, scored by area under the learning curve, analysed at the conditional
95th percentile).

**Advantage normalisation (their choice C67) — the direct answer.** Defined as *"per minibatch
advantage normalization … subtracting their mean and dividing by their standard deviation for the
policy loss."* Finding, verbatim: ***"per-minibatch advantage normalization (C67) seems not to affect
the performance too much."*** It is **absent from their §3.3 recommendation**, and their own baseline
default table lists `C67 Per minibatch advantage normalization: **False**`. So the largest ablation
ever run on on-policy RL found advantage normalisation to be **roughly cosmetic** — neither
recommended nor warned against.

**Value-function-target normalisation (their choice C66) — and this is the closest thing in the
literature to what our MC branch attempts.** Defined verbatim: *"we also maintain the empirical mean
`v_µ` and standard deviation `v_ρ` of **value function targets** … The value function network
**predicts normalized targets** `(V̂ − v_µ)/max(v_ρ, 10⁻⁶)` and its outputs are **denormalized
accordingly** to obtain predicted values."* Note the two differences from our pattern: the statistic is
**running** (over all targets seen), and the network's output is **de-normalised at read-out** so that
predicted values are always available in raw units.

Even so, they find it wildly unstable across environments, verbatim: *"Quite surprisingly, value
function normalization (C66) **also influences the performance very strongly** — it is crucial for good
performance on HalfCheetah and Humanoid, helps slightly on Hopper and Ant and **significantly hurts the
performance on Walker2d**. We are not sure why the value function scale matters that much but suspect
that it affects the performance by changing the speed of the value function fitting."*

**§3.3 recommendation, verbatim, in full:** *"Always use observation normalization and check if value
function normalization improves performance. Gradient clipping might slightly help but is of secondary
importance."* Note what that says: value-target normalisation is a **check-empirically** item, not a
default — and it is the *disciplined* version being described.

**Reward scaling is not studied in this paper at all** (zero occurrences of "reward scaling" /
"reward normalization"). Do not cite it either way on that question.

Adjacent and relevant: *"PPO-style value loss clipping (C13) hurts the performance regardless of the
clipping threshold … Recommendation. Use GAE with λ = 0.9 but neither Huber loss nor PPO-style value
loss clipping."*

### van Hasselt, Guez, Hessel, Mnih & Silver (2016), *Learning values across many orders of magnitude* (PopArt) — [arXiv:1602.07714](https://arxiv.org/abs/1602.07714)

**Yes — PopArt is framed as the solution to exactly the moving-target problem that per-batch return
normalisation creates.** The abstract sets up the problem as *"adaptively normalize the targets used
in learning … where the magnitude of appropriate value approximations can change over time when we
update the policy of behavior."*

The name decomposes the fix, verbatim: *"**(ART)** to update scale Σ and shift µ such that `Σ⁻¹(Y − µ)`
is appropriately normalized, and **(POP)** to preserve the outputs of the unnormalized function when we
change the scale and shift … an acronym for '**Preserving Outputs Precisely, while Adaptively Rescaling
Targets**'."* Proposition 1 gives the output-preserving weight surgery: when the statistics move from
`(σ, µ)` to `(σ_new, µ_new)`, the output layer is rewritten as

$$W_{\text{new}} = \frac{\sigma}{\sigma_{\text{new}}} W, \qquad b_{\text{new}} = \frac{\sigma b + \mu - \mu_{\text{new}}}{\sigma_{\text{new}}}$$

so that the un-normalised function is **unchanged for every input**.

The paper's §2.1 states the danger of doing ART without POP — i.e. of naive target rescaling — in
terms that describe our MC branch precisely:

> *"**Unless care is taken, repeated updates to the normalization might make learning harder rather
> than easier because the normalized targets become non-stationary. More importantly, whenever we adapt
> the normalization based on a certain target, this would simultaneously change the output of the
> unnormalized function of all inputs. If there is little reason to believe that other unnormalized
> outputs were incorrect, this is undesirable and may hurt performance in practice.**"*

Their toy experiment demonstrates it: the ART-only variant *"has high error after the spikes because
**the changing normalization also changes the outputs of the smaller inputs, increasing the errors on
these**"*, and its tuned statistic-decay rate is forced two orders of magnitude slower (β = 10⁻⁴ vs
10⁻⁰·⁵ for full PopArt) — i.e. without output preservation you are pushed toward an almost-frozen
normaliser. A per-batch z-score is the opposite extreme: the fastest-moving normaliser possible, with
no output preservation at all.

### DreamerV3 (Hafner, Pasukonis, Ba & Lillicrap, 2023) — [arXiv:2301.04104](https://arxiv.org/abs/2301.04104)

Two separate mechanisms, and the split answers the question directly.

**The critic's target is not rescaled by any running statistic.** The critic regresses the raw
λ-return `R^λ` under a categorical **symexp two-hot** loss with fixed exponentially-spaced bins,
verbatim: *"we parameterize the critic as categorical distribution with exponentially spaced bins,
**decoupling the scale of gradients from the prediction targets**"*, and *"the loss only depends on the
probabilities assigned to the bins but not on the continuous values associated with the bin locations."*
The `symlog` transform `sign(x)·ln(|x|+1)` is a **fixed, invertible, learning-independent** function —
it does not drift with the data.

**The return normalisation is applied to the actor, not the critic**, and the paper is explicit about
the statistic being a running one. Actor loss (Eq. 6) divides the advantage `R^λ − v_ψ(s)` by
`max(1, S)`, where (Eq. 7)

$$S \doteq \mathrm{EMA}\big(\mathrm{Per}(R^\lambda, 95) - \mathrm{Per}(R^\lambda, 5),\ 0.99\big)$$

— percentiles computed within the batch, then **smoothed by an exponential moving average with decay
0.99**, and used **scale-only** (*"subtracting an offset from the returns does not change the actor
gradient and thus dividing by the range S is sufficient"*), with a one-sided floor so small returns are
never amplified. Their hyperparameter table names every one of these rows "Actor RetNorm …", and the
corresponding ablation is literally labelled "No retnorm (**advnorm**)" — i.e. their alternative to
return normalisation *is* conventional advantage normalisation.

Two of their stated arguments bear directly on our case, verbatim:

> *"**Normalizing rewards or returns by standard deviation can fail under sparse rewards where their
> standard deviation is near zero, drastically amplifying rewards regardless of their size.**"*

> *"On the other hand, **normalizing targets based on running statistics introduces non-stationarity
> into the optimization.** We suggest the symlog squared error as a simple solution to this dilemma."*
> … *"symlog transformations avoid truncating large targets, **introducing non-stationarity from
> normalization**, or adjusting network weights when new extreme values are detected."*

That last clause is an explicit rejection of PopArt — and note that the thing DreamerV3 rejects as
too non-stationary is *running-statistic* target normalisation. Per-batch z-scoring is strictly worse
on that axis.

### Farebrother et al. (2024), *Stop Regressing* — [arXiv:2403.03950](https://arxiv.org/abs/2403.03950)

Replaces MSE regression on a scalar value target with **categorical cross-entropy** (HL-Gauss: spread
a Gaussian around the target and integrate over bins; or two-hot). Reported gains: *"consistently 30%
better performance when scaling parameters with Mixture-of-Experts in single-task RL on Atari; 1.8–2.1×
… in multi-task setups"* and so on.

The mechanism they identify is the one at issue here, verbatim: *"the benefits of categorical
cross-entropy primarily stem from its ability to mitigate issues inherent to value-based RL, such as
**noisy targets and non-stationarity**"*, and *"when learning a value function with TD, the prediction
targets are non-stationary and often increase in magnitude over time as the policy improves … **classification losses retain higher plasticity under non-stationary targets compared to regression**."*
Their sharpest control: in an *offline SARSA* setting, where policy-driven target drift is removed,
*"most of the benefit from HL-Gauss compared to the MSE loss vanishes"* — so the gain really is about
the moving target.

**Important limitation for citation purposes:** this paper does **not** compare against PopArt, symlog,
or any running-statistics normalisation (zero mentions of either). It cannot be cited as
"classification beats normalisation". Its ablation axis is loss family only. Note also that two-hot
*underperformed* plain MSE in their study; HL-Gauss was the winner.

### Summary of the evidence on load-bearing-ness

| Mechanism | Best evidence | Verdict |
|---|---|---|
| **Reward scaling** (running std of a discounted-reward accumulator, applied to rewards, scale-only) | Engstrom Fig. 1: one of three (of four) ablated optimisations that "significantly impact the rewards landscape … and were necessary for attaining the highest PPO reward" | **Load-bearing** in MuJoCo PPO. Not tested by Andrychowicz. Rejected by DreamerV3 for sparse-reward domains |
| **Advantage normalisation** (per-minibatch z-score) | Andrychowicz C67: "seems not to affect the performance too much"; omitted from their recommendations; their own default is `False`. Engstrom does not mention it | **Closer to cosmetic than load-bearing.** Universally *implemented*, weakly *evidenced*. Its value is mainly step-size insensitivity, not final score |
| **Value-target normalisation** (running mean/std, de-normalised read-out) | Andrychowicz C66: "influences the performance very strongly — crucial on HalfCheetah and Humanoid … **significantly hurts** on Walker2d" | **High-variance, environment-dependent.** Their recommendation is "check if it improves performance", not "use it" |
| **Value-target normalisation with output preservation** (PopArt) | van Hasselt et al. 2016, Prop. 1 | The principled version; exists *because* naive rescaling breaks the regression |
| **Fixed target transform** (symlog + two-hot; HL-Gauss) | DreamerV3 ablation ranks retnorm and symexp-twohot second only to the world-model KL; Farebrother 2024 | The current preferred answer — sidesteps normalisation entirely by making the transform fixed and the loss scale-free |

The pattern across all five papers: **the field's disagreement is about the target-side, and every
proposal that touches the target either keeps the statistic slow-moving, de-normalises at read-out,
preserves the network's outputs when the statistic moves, or replaces the statistic with a fixed
transform.** Nobody proposes recomputing a mean and standard deviation from the current batch and
regressing onto the result.

## Corroboration from the community reference

The most-cited catalogue of PPO implementation choices — Huang, Dossa, Raffin, Kanervisto & Wang,
*"The 37 Implementation Details of Proximal Policy Optimization"* (ICLR Blog Track, 2022,
<https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/>) — lists the two relevant
items as **separate, differently-targeted** details:

- **Detail #7, "Normalization of Advantages"**: *"PPO normalizes the advantages by subtracting their
  mean and dividing by their standard deviation. In particular, this normalization happens at the
  minibatch level instead of the whole batch level!"*
- **Detail #8, "Reward Scaling"**: *"The `VecNormalize` applies a discount-based scaling scheme, where
  rewards are divided by the standard deviation of a rolling discounted sum of the rewards (without
  subtracting and re-adding the mean)."*

Normalisation of the **value-function target / returns** appears nowhere in the list of 37 details.
Two secondary points follow for this project. First, the convention is *per-minibatch* advantage
standardisation; this project's GAE branch standardises over the whole batch — a minor deviation, and
much less consequential than the MC branch's. Second, the accepted form of return-scale correction
(detail #8) is **scale-only and rolling** — no mean subtraction, no per-batch statistic — which is the
opposite of a per-batch z-score on both counts.

## Provenance

All sources fetched 2026-09-01 from the default branch of each repository. HEAD commit at fetch time
and date of that commit:

| Repository | Branch | HEAD at fetch | HEAD commit date |
|---|---|---|---|
| `DLR-RM/stable-baselines3` | `master` | `3246f506` | 2026-08-17 |
| `vwxyzjn/cleanrl` | `master` | `fe8d8a03` | 2026-04-20 |
| `openai/baselines` | `master` | `ea25b9e8` | 2020-01-31 (archived; unchanged since) |
| `openai/spinningup` | `master` | `038665d6` | 2020-02-07 (unmaintained; unchanged since) |
| `pytorch/examples` | `main` | `acc295dc` | 2025-09-01 |
| `Eclectic-Sheep/sheeprl` | `main` | `33b63668` | 2024-07-12 |
| `thu-ml/tianshou` | `master` | `f2402056` | 2026-04-03 |
| `ray-project/ray` | `master` | `167681c2` | 2026-09-01 |
| `pytorch/rl` (torchrl) | `main` | `5b1ae9cc` | 2026-08-31 |
| `marlbenchmark/on-policy` (MAPPO) | `main` | `de66d7a4` | 2024-07-18 |

**Version caveats.** Tianshou reorganised its package layout: the PPO/A2C code now lives under
`tianshou/algorithm/modelfree/` (formerly `tianshou/policy/modelfree/`), and the flag formerly called
`rew_norm` is now `return_scaling`. RLlib carries two API stacks; both were checked and both
standardise advantages only. torchrl's `advantage_norm` / `ValueNorm` hook is recent — older releases
expose only `normalize_advantage` and `average_gae`. Line numbers are valid for the commits above.

## Related project docs

This survey was commissioned as an independent, primary-source check and it **corroborates work the
project had already done** — it did not discover the bug. The relationship:

- [[mc_return_units_bug_severity_and_repair]] (`docs/project/critiques/`, `professor-rl`, 2026-09-01)
  is the severity analysis and repair critique of this exact defect. Its §1.3 asserts that
  "raw value targets + normalised advantages is what reference PPO implementations do (Schulman et
  al. 2017; Stable-Baselines3; CleanRL)". **This document is the source-level verification of that
  assertion**, extended from three implementations to ten, plus the ablation-paper evidence on how
  load-bearing each mechanism actually is. Its independently-reached archaeological note — that the
  `"PyTorch parity"` comment points at the `pytorch/examples` REINFORCE idiom, harmless without a
  critic — is confirmed here from source, and refined: the exact-match ancestor is the sibling
  `actor_critic.py`, not `reinforce.py`.
- [[FIX_MC_RETURN_UNITS_AND_LEARNING_RATES]] (`docs/develop/active/issues/`) — the repair plan.
- [[findings_rppo_trainer]] (`docs/reviews/diagnosis_20260723/`) — the 2026-07-23 diagnosis that first
  flagged the pattern (Findings 1 and 3), rated P2 at the time.
- [[fix_plan_h4_mc_window_bootstrap]] — the July change that introduced the bootstrap seed and thereby
  turned a tolerable idiom into a units bug. Worth noting for the bug registry as a reusable lesson:
  *a correctness invariant can be destroyed by a different, correct change elsewhere.*
- Trainer under discussion: `src/models/recurrent_ppo_trainer.py` (MC branch at lines 370–380, GAE
  branch at 391–396) and the sibling non-recurrent `src/models/ppo_trainer.py:219`, which carries the
  same MC-branch pattern.

### Two points from this survey that bear on the open fix plan

1. **The critique's §2(b) observation is corroborated by the convention.** The trainer standardises
   over the whole batch (~16k values pooled across 128 envs × 128 steps). The community reference is
   explicit that PPO's advantage standardisation happens *"at the minibatch level instead of the whole
   batch level"* (detail #7). Worth folding into the repair rather than fixing only the target side.
2. **On how much the advantage-normalisation half of the repair will buy**: Andrychowicz et al.
   (>250,000 agents) found per-minibatch advantage normalisation *"seems not to affect the performance
   too much"* and left it `False` by default. The expected win from the repair should therefore be
   attributed almost entirely to **removing the non-stationary critic target and the units mismatch**,
   not to adding advantage normalisation. A pre-registered gate that expects a large effect from the
   advantage side specifically is likely to be disappointed.

- Bug registry: [[KNOWN_BUGS]] — if this is escalated, record it there.
