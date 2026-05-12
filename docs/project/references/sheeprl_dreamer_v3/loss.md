---
title: "Sheeprl Reference: loss.py"
source: tmp/sheeprl/sheeprl/algos/dreamer_v3/loss.py
generated: 2026-05-12
status: reference
---

# Sheeprl Reference — `loss.py`

> **Source**: `tmp/sheeprl/sheeprl/algos/dreamer_v3/loss.py` — 88 lines.
> **Purpose** (one-line): The combined world-model loss: reconstruction + reward NLL + continue BCE + KL dynamic + KL representation, with free-nats clipping.
> **Imports from elsewhere in this index**: [`distribution.md`](distribution.md) (TwoHot + Symlog + Bernoulli distributions used here for `log_prob`); [`agent.md`](agent.md) (the `PlayerDV3` / world model emits the `po`, `pr`, `pc` distributions and `priors_logits` / `posteriors_logits` consumed below).

---

## Table of Contents

- [Lines 1–6 — Imports](#lines-16--imports)
- [Line 9 — `reconstruction_loss`](#line-9--reconstruction_loss)

---

## Lines 1–6 — Imports

```python
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.distributions import Distribution, Independent, OneHotCategoricalStraightThrough
from torch.distributions.kl import kl_divergence
```

**What this is**: standard typing primitives plus the torch tensor / distribution machinery the loss depends on. `Distribution` is the abstract base used by sheeprl's `TwoHotEncodingDistribution`, `SymlogDistribution`, `MSEDistribution`, and `BernoulliSafeMode` (see [`distribution.md`](distribution.md)) — the loss only ever touches them through `log_prob`, so any sheeprl distribution that defines `log_prob` plugs in here. `Independent` wraps the categorical over each of the 32 stochastic-state slots into a single joint event so `kl_divergence` returns one number per (batch, time) cell; `OneHotCategoricalStraightThrough` is the categorical-with-straight-through-estimator used for the discrete latents `z` in DreamerV3 (Hafner et al. 2023, §2). `kl_divergence` is torch's distribution registry that dispatches `KL(p ‖ q)` analytically for two `OneHotCategoricalStraightThrough` arguments.

---

## Line 9 — `reconstruction_loss`

```python
def reconstruction_loss(
    po: Dict[str, Distribution],
    observations: Tensor,
    pr: Distribution,
    rewards: Tensor,
    priors_logits: Tensor,
    posteriors_logits: Tensor,
    kl_dynamic: float = 0.5,
    kl_representation: float = 0.1,
    kl_free_nats: float = 1.0,
    kl_regularizer: float = 1.0,
    pc: Optional[Distribution] = None,
    continue_targets: Optional[Tensor] = None,
    continue_scale_factor: float = 1.0,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """
    Compute the reconstruction loss as described in Eq. 5 in
    [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104).

    Args:
        po (Dict[str, Distribution]): the distribution returned by the observation_model (decoder).
        observations (Tensor): the observations provided by the environment.
        pr (Distribution): the reward distribution returned by the reward_model.
        rewards (Tensor): the rewards obtained by the agent during the "Environment interaction" phase.
        priors_logits (Tensor): the logits of the prior.
        posteriors_logits (Tensor): the logits of the posterior.
        kl_dynamic (float): the kl-balancing dynamic loss regularizer.
            Defaults to 0.5.
        kl_balancing_alpha (float): the kl-balancing representation loss regularizer.
            Defaults to 0.1.
        kl_free_nats (float): lower bound of the KL divergence.
            Default to 1.0.
        kl_regularizer (float): scale factor of the KL divergence.
            Default to 1.0.
        pc (Bernoulli, optional): the predicted Bernoulli distribution of the terminal steps.
            0s for the entries that are relative to a terminal step, 1s otherwise.
            Default to None.
        continue_targets (Tensor, optional): the targets for the discount predictor. Those are normally computed
            as `(1 - data["dones"]) * args.gamma`.
            Default to None.
        continue_scale_factor (float): the scale factor for the continue loss.
            Default to 10.

    Returns:
        observation_loss (Tensor): the value of the observation loss.
        KL divergence (Tensor): the KL divergence between the posterior and the prior.
        reward_loss (Tensor): the value of the reward loss.
        state_loss (Tensor): the value of the state loss.
        continue_loss (Tensor): the value of the continue loss (0 if it is not computed).
        reconstruction_loss (Tensor): the value of the overall reconstruction loss.
    """
    rewards.device
    observation_loss = -sum([po[k].log_prob(observations[k]) for k in po.keys()])
    reward_loss = -pr.log_prob(rewards)
    # KL balancing
    dyn_loss = kl = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits.detach()), 1),
        Independent(OneHotCategoricalStraightThrough(logits=priors_logits), 1),
    )
    free_nats = torch.full_like(dyn_loss, kl_free_nats)
    dyn_loss = kl_dynamic * torch.maximum(dyn_loss, free_nats)
    repr_loss = kl_divergence(
        Independent(OneHotCategoricalStraightThrough(logits=posteriors_logits), 1),
        Independent(OneHotCategoricalStraightThrough(logits=priors_logits.detach()), 1),
    )
    repr_loss = kl_representation * torch.maximum(repr_loss, free_nats)
    kl_loss = dyn_loss + repr_loss
    if pc is not None and continue_targets is not None:
        continue_loss = continue_scale_factor * -pc.log_prob(continue_targets)
    else:
        continue_loss = torch.zeros_like(reward_loss)
    reconstruction_loss = (kl_regularizer * kl_loss + observation_loss + reward_loss + continue_loss).mean()
    return (
        reconstruction_loss,
        kl.mean(),
        kl_loss.mean(),
        reward_loss.mean(),
        observation_loss.mean(),
        continue_loss.mean(),
    )
```

**What it does**: implements the full DreamerV3 world-model loss of Hafner et al. 2023 (Eq. 5 of [arxiv 2301.04104](https://arxiv.org/abs/2301.04104)),
ℒ(φ) = 𝔼_{q_φ}[ Σ_k −ln p_φ(o^k | z, h) − ln p_φ(r | z, h) − ln p_φ(c | z, h) + β_dyn · max(1, KL[ sg(q_φ(z|h,o)) ‖ p_φ(z|h) ]) + β_rep · max(1, KL[ q_φ(z|h,o) ‖ sg(p_φ(z|h)) ]) ],
broken down term-by-term as follows.

1. **`observation_loss = -Σ_k po[k].log_prob(observations[k])`** (line 61) — the decoder NLL summed over every observation key in the multimodal dict (pixels use `MSEDistribution`, vectors use `SymlogDistribution`, see [`distribution.md`](distribution.md)). With Gaussian-unit-variance decoders this reduces to MSE plus a constant; with `SymlogDistribution` it is MSE in symlog-warped space. The minus sign turns log-likelihood into a loss.
2. **`reward_loss = -pr.log_prob(rewards)`** (line 62) — the reward predictor's NLL. `pr` is a `TwoHotEncodingDistribution` (see [`distribution.md`](distribution.md)): the reward target is symlog-transformed and represented as a soft two-hot vector over 255 exponentially-spaced bins, and `log_prob` is the cross-entropy of the predicted soft-bin distribution against that target. This is the discrete-regression head from §3 ("Symlog two-hot loss") of the paper.
3. **KL balancing** (lines 64–75) — splits the latent KL into two asymmetric pieces, exactly as in Hafner et al. 2023 Eq. 5. **Dynamic loss** `dyn_loss = β_dyn · max(KL[ sg(q) ‖ p ], free_nats)` pulls the prior `p_φ(z|h)` toward the (stop-gradient) posterior `sg(q_φ(z|h,o))`, teaching the dynamics. **Representation loss** `repr_loss = β_rep · max(KL[ q ‖ sg(p) ], free_nats)` pulls the posterior toward the (stop-gradient) prior, regularising the encoder. The `.detach()` on alternating sides is the kl-balancing trick (Hafner et al. 2021 DreamerV2 §3.3) that lets the two trade off at different learning rates (`kl_dynamic=0.5`, `kl_representation=0.1`); the variable `kl` (line 64) is kept un-clipped, un-scaled, for logging only. **Free-nats clipping** `torch.maximum(·, 1.0)` (lines 69, 74) zeros the gradient whenever the KL is already below 1 nat per token, preventing posterior collapse — a per-element floor, not a global one. The KL itself is computed in closed form by torch's registered KL of two `Independent(OneHotCategoricalStraightThrough, 1)` distributions over the 32-slot discrete latent, where `Independent(..., 1)` re-interprets the slot axis as the event dim so one scalar KL is returned per (batch, time) cell.
4. **`continue_loss = continue_scale_factor · −pc.log_prob(continue_targets)`** (lines 76–79) — the discount-predictor BCE. `pc` is a `BernoulliSafeMode` (see [`distribution.md`](distribution.md)), `continue_targets` are typically `(1 − dones) · γ` per the docstring, and `log_prob` is the Bernoulli log-likelihood = BCE up to sign. Falls back to zeros if no continue head is configured.
5. **Aggregation** (line 80) — `reconstruction_loss = mean( kl_regularizer · kl_loss + obs_loss + reward_loss + continue_loss )` averages across batch+time so the optimiser sees one scalar; `kl_regularizer` is the outer β on top of the inner `kl_dynamic` / `kl_representation` weights. The 6-tuple return packs the aggregate loss plus per-term means for WandB logging.

This single function is the entire world-model training objective in sheeprl DreamerV3 — it is called once per gradient step inside the training loop (see [`dreamer_v3.md`](dreamer_v3.md)) to back-prop through the RSSM, encoder, decoder, reward head, and continue head jointly. The actor and critic have their own separate losses elsewhere.
