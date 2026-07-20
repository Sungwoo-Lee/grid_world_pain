# Dreamer Q&A Log

## Purpose

This file is a running transcript of questions asked about the Dreamer algorithms
and this project's implementations of them, together with the answers given. It
exists so that understanding built up in conversation is not lost when the
session ends.

It is deliberately **not** organised by theme. Entries appear in the order the
questions were asked, numbered `Q1`, `Q2`, … New conversations append to the
bottom; nothing above is rewritten. If a later answer corrects an earlier one,
the correction is noted in the later entry and the earlier entry is left as it
was — the log records what was actually said, including the wrong turns.

Companion documents:

- [[dreamer_lit_review]] — the paper-by-paper review (World Models → Dreamer 4)
  that this discussion draws on.
- `src/algorithms/dreamer_srl/` — this project's JAX/Flax-NNX port of
  sheeprl@`33b6366`, which is what the code references below point at.
- `vendor/sheeprl/sheeprl/algos/dreamer_v3/` — the PyTorch reference the port
  is line-for-line matched against.

## Append protocol

- Add new entries at the bottom, continuing the number sequence.
- Append **as the conversation goes**, not batched at the end — the log is expected
  to stay current with the live discussion.
- Keep the `**Q:**` text close to what was actually asked.
- Keep the answer substantive — a reader should get the answer here, not a
  pointer to go and re-derive it.
- Line references (`agent.py:975`) are pinned to the code as of the entry's
  date; re-verify before relying on them.

---

## Session 2026-07-15 — world-model loss, RSSM, KL terms

### Q1 — Which observation does the world-model loss reconstruct?

**Q:** When we train the dreamer, how do we calculate the loss for the world
model? We have input observation at time `t`, `O_t`, which generates `z_t`. Then
will it generate `Ôhat_t` to calculate loss? Or `ẑ_{t+1}` → `Ô_{t+1}` and compare
with `O_{t+1}`?

**A:** `Ô_t` vs `O_t` — **the same timestep**. The decoder reconstructs the
*current* observation from the *current* posterior latent. There is no `Ô_{t+1}`
comparison anywhere in the DreamerV3 world-model loss.

The temporal prediction happens **in latent space via the KL**, not in
observation space. At step `t`:

```
a_{t-1} ─┐
z_{t-1} ─┴─→ h_t ─┬─→ ẑ_t  (prior,     from h_t alone)  ──┐
                  │                                        ├─ KL
                  └─→ z_t  (posterior, from h_t + φ(o_t)) ─┘
                          │
                          └─→ [z_t ; h_t] ─→ ô_t ≈ o_t      (decoder)
                                           ─→ r̂_t ≈ r_t      (reward head)
                                           ─→ ĉ_t ≈ 1-term_t (continue head)
```

`h_t` carries information only up to `t-1`, so the prior `p(ẑ_t | h_t)` is a
genuine one-step-ahead prediction made before seeing `o_t`. The three loss terms
split the labour: reconstruction forces `z_t` to *encode* `o_t`; the KL forces
the prior to *anticipate* `z_t`.

Compose them and next-step observation prediction comes free (roll prior →
decode), but that composition is never trained end-to-end against `O_{t+1}`.

Code: `train.py:705` runs `wm.observe(...)`; `train.py:720` wraps the decoder
output; `train.py:744` passes it to `reconstruction_loss` against
`batch["obs"]` — same, unshifted index. `agent.py:1711-1717` shows the decoder
reads the concatenated **posterior** latent `[z_t ; h_t]`.

Why: the decoder is a representation-shaping device, not a predictor you ever
call. During imagination (`WorldModel.imagine`, agent.py:1732) it is never
invoked — only reward, continue, and critic heads read the latent. Observations
are never materialised in the dream.

Footnote: sheeprl (and therefore our port) pins `kl_dynamic=0.5`, the DreamerV3
**preprint** value. The Nature paper raised `β_dyn` to 1.0. It's a config key,
not a code change.

### Q2 — Explain the RSSM

**Q:** It's confusing, explain me the RSSM first.

**A:** The agent sees `o_t`, which is not the state of the world. It needs a
belief that has two properties pulling in opposite directions: **memory**
(remember what you can't see) and **honest uncertainty** (represent "the predator
either is or isn't there" as two possibilities, not one blurry average).

Two obvious designs each get one and fail the other:

- **Plain RNN** `h_t = f(h_{t-1}, a_{t-1})`: great memory, but always 100%
  confident. An actor trained inside it will exploit any confident-but-wrong
  prediction.
- **Stochastic SSM** `z_t ~ p(z_t | z_{t-1}, a_{t-1})`: can hedge, but re-samples
  every step so information leaks away — it forgets.

PlaNet's answer: carry **both**.

| | what | XS config | sampled? | job |
|---|---|---|---|---|
| `h_t` | GRU hidden state | 256-dim | never | **memory** |
| `z_t` | 32 categoricals × 32 classes | 1024-dim, 32 bits on | every step | **hedging** |

The model state is the **concatenation** `[z_t ; h_t]` = 1280 numbers. Everything
downstream (decoder, reward, continue, actor, critic) reads only that. Nothing
sees `o_t` directly — that bottleneck is what makes the scheme work.

**One tick**, three steps:

1. **Carry memory forward.** `h_t = GRU(MLP([z_{t-1} ; a_{t-1}]), h_{t-1})`.
   No observation. `h_t` is the state of mind *before looking*. (This is also why
   the action-shift exists: the action that produced `o_t` is `a_{t-1}`.)
2. **Guess what you're about to see — the PRIOR.** `ẑ_t ~ p(· | h_t)`.
3. **Look — the POSTERIOR.** `z_t ~ q(· | h_t, φ(o_t))`.

The prior/posterior pair is the crux. Identical shape, near-identical networks;
the only difference is the information they're allowed:

- prior sees `h_t` only → info to `t-1` → a **prediction**
- posterior sees `h_t` + `o_t` → info to `t` → a **measurement**

When observations exist (replay), use the posterior; the prior is computed anyway
so it can be trained to match it (that's the KL). When they don't (imagination),
roll on the prior alone. **The KL is the bridge that makes imagination
legitimate.**

Evidence the split is load-bearing: PlaNet's ablation — a purely deterministic
model fails on nearly every task; a purely stochastic one fails too ("the agent
does not learn without it").

Code: `RSSM.dynamic` (agent.py:975) is exactly one tick. `WorldModel.observe`
(agent.py:1625) loops it over the 64-step sequence. `is_first` handling at the
top of `dynamic` is the episode-boundary reset.

Punchline tying back to Q1: the decoder reads the **posterior** — the belief about
*now* — so of course it reconstructs `o_t`. And in imagination (agent.py:1813)
`new_latent = concat([prior_flat_new, recurrent_state])` — the prior sits in the
same 1280-wide slot the posterior occupied. Actor and critic can't tell the
difference, which is the point.

### Q3 — In a VAE the prior is fixed; here both sides are learned. How does the posterior network get updated?

**Q:** In the VAE, no RSSM, the latent `z` is compared with the theoretical `z`,
so we can consider that `z` will be closer to the theoretical `z`. But here KL
compares `z_prior` and `z_posterior`, generated by different inputs (`h` only, vs
`h` with obs). Then both are models. How can we update the `(h, obs) → z` network?

**A:** The worry is exactly right: in a VAE the prior `N(0,I)` is a fixed fence
post. Here both sides are learned networks — so what stops them agreeing on
garbage? Both could output the same constant, KL → 0, nothing learned. The KL
alone cannot prevent that.

**The answer: the posterior is not anchored by the KL. It's anchored by
reconstruction.** `L_pred` gradient flows decoder → through `z_t` → into the
representation network. And `o_t` reaches the decoder *only* through the 32×32
bottleneck. So the posterior has no choice but to pack `o_t` into those slots.
That's the fence post — not a fixed distribution, but a *job*.

Who gets which gradient:

| loss term | weight | flows into | job |
|---|---|---|---|
| `L_pred` | 1.0 | posterior net, GRU, decoder, encoder | make `z_t` **informative** |
| `L_dyn` = `KL(sg(q) ‖ p)` | 0.5 | prior net (`q` detached) | teach prior to **predict** |
| `L_rep` = `KL(q ‖ sg(p))` | 0.1 | posterior net (`p` detached) | stop encoding the **unpredictable** |

The posterior gets a strong informative signal and a weak (0.1) tug. The prior's
*only* signal is `L_dyn` at 0.5 — it chases 5× harder than the posterior is
dragged back. Deliberate: *let the posterior do what reconstruction demands; make
the prior catch up.*

**The real VAE contrast:** in a VAE the KL has **one** job (regularise `q` —
nobody wants to learn `N(0,I)`). In the RSSM it has **two**, which is exactly why
it's split into halves with stop-gradients: the dominant job is *training the
prior* (`L_dyn`), the minor one is the VAE-ish *regularise the posterior*
(`L_rep`). A VAE never needs `L_dyn` because there's nothing to train.

**The collapse is real.** DreamerV2's ablation removing image-reconstruction
gradients collapses the agent to a **0.04 gamer median** — two orders of magnitude
worse than the next-worst ablation. Conversely, stopping *reward* gradients into
the encoder slightly *helps*. The representation is essentially entirely carried
by reconstruction.

**Free bits** (`loss.py:588`): `jnp.maximum(dyn_loss, kl_free_nats)` with
`free_nats = 1.0`. Below 1 nat, `max(KL, 1)` is constant → gradient exactly zero.
The KL pressure switches off. This prevents posterior collapse.

Punchline: during world-model training `WorldModel.observe` **throws away** the
prior's sample (agent.py:1688 — `prior_state` unpacked, never stored). Only
`prior_logits` survives, and its only destination is the KL. The prior is a pure
student during training; it becomes load-bearing only in `imagine()`.

### Q4 — Are `L_pred` / `L_dyn` / `L_rep` the names in our scripts?

**Q:** You have jargon like `L_dyn`, `L_rep`, `L_pred` — you can use them as they
are in the scripts (are they?) but you need to explain first.

**A:** **No, they are not in the scripts.** They are Hafner's DreamerV3 paper
names, lifted from the lit review and used as if they were ours. Grep confirms
they appear nowhere in `src/algorithms/dreamer_srl/`.

Defined:

- **`L_pred`** — the *prediction loss*, better read as **reconstruction**. Three
  NLLs off the posterior latent at time `t`: rebuild `o_t`, `r_t`, `c_t`. Forces
  `z_t` to encode something. β_pred = 1.
- **`L_dyn`** — the *dynamics loss*, `KL(sg(q) ‖ p)`. Posterior detached → pulls
  the prior. β_dyn = 0.5.
- **`L_rep`** — the *representation loss*, `KL(q ‖ sg(p))`. Prior detached →
  pulls the posterior. β_rep = 0.1.

Mapping:

| Paper | `loss.py` local | config key | WandB key |
|---|---|---|---|
| `L_pred` | **no single name** | — (β_pred=1 implicit) | `observation_loss`, `reward_loss`, `continue_loss` |
| ↳ obs | `observation_loss` | — | `observation_loss` |
| ↳ reward | `reward_loss` | — | `reward_loss` |
| ↳ continue | `continue_loss` | `continue_scale_factor: 1.0` | `continue_loss` |
| `L_dyn` | `dyn_loss` | `kl_dynamic: 0.5` | `loss_dyn_kl` |
| `L_rep` | `repr_loss` | `kl_representation: 0.1` | `loss_rep_kl` |
| `L_dyn + L_rep` | `kl_loss` | — | **`state_loss`** |
| raw KL (pre-weight/floor) | `kl` | — | (sheeprl: `State/kl`) |
| free bits | — | `kl_free_nats: 1.0` | — |
| overall KL scale | — | `kl_regularizer: 1.0` | — |
| whole thing | `total` | — | `world_model_loss` |

Assembly, `loss.py:613`:

```python
total = (kl_regularizer * kl_loss + observation_loss + reward_loss + continue_loss).mean()
```

β_pred = 1 is *implicit* — the reconstruction terms carry no coefficient.

**Four naming traps:**

1. **`state_loss` is the KL** (`dyn_loss + repr_loss`), not "loss on the state".
   Inherited from sheeprl's return signature (`dreamer_v3.py:176`). Most likely
   to mislead on a dashboard.
2. `repr_loss` (local) vs `loss_rep_kl` (WandB) — same thing, inconsistent name.
3. `kl` ≠ `kl_loss`. `kl` is raw, pre-weight, pre-floor.
4. There is no `L_pred` metric — sum three WandB keys yourself.

**Falls out of this:** `train.py:758-764` computes both KL metrics from the *same*
raw tensor `_kl_tb` — correct, because `KL(sg(q)‖p)` and `KL(q‖sg(p))` have the
same forward value (stop-gradient changes only gradient routing, never the
number). Consequence: **`loss_dyn_kl` and `loss_rep_kl` are the same curve scaled
by `0.5/0.1 = 5`, always in exact 5:1 ratio.** They carry no independent
information.

### Q5 — Explain `L_rep`; and introduce the networks first so the losses can be connected to them

**Q:** I don't understand the `L_rep`. And also it will be first to introduce the
network structures and name them to connect the loss.

**A:** `build_agent` (agent.py:1939) constructs eight modules. XS config:
`recurrent_state_size=256`, `dense_units=256`, `z` is 32×32=1024 flat, latent =
1280.

| # | code name | maps | what it is |
|---|---|---|---|
| 1 | `world_model.encoder` | `o_t` → 256 | symlog + embed → `φ(o_t)` |
| 2 | `rssm.recurrent_mlp_*` + `rssm.gru_cell` | `[z_{t-1} ; a_{t-1}]`, `h_{t-1}` → `h_t` | **memory** (Hafner: *sequence model*) |
| 3 | `rssm.transition_*` (`_transition`) | `h_t` → 1024 logits | **prior net** |
| 4 | `rssm.repr_*` (`_representation`) | `[h_t ; φ(o_t)]` → 1024 logits | **posterior net** |
| 5 | `world_model.decoder` | 1280 → `o_t` | sheeprl calls it `observation_model` |
| 6 | `world_model.reward_model` | 1280 → 255 | two-hot reward head |
| 7 | `world_model.continue_model` | 1280 → 1 | Bernoulli continue head |
| 8 | `actor` / `critic` / `target_critic` | 1280 → A / 255 / 255 | **not** part of the world model |

Modules 1–7 are the `WorldModel` (agent.py:1573): **one** loss scalar, **one**
backward pass, **one** optimizer (`wm_opt`, train.py:810-814).

Which term moves which network:

| loss term | pushes on |
|---|---|
| `observation_loss` | decoder — and back through `z_t` → posterior net, `h_t` → GRU, `φ(o_t)` → encoder |
| `reward_loss` | reward_model + same path back |
| `continue_loss` | continue_model + same path back |
| `dyn_loss` ×0.5 | prior net + **back through `h_t` into the GRU** |
| `repr_loss` ×0.1 | posterior net + back through `h_t` and `φ(o_t)` |

> **Correction to Q3's table.** I said `L_dyn` reaches "the prior net only." Too
> strong. The `sg` blocks the `q` *side* of the KL — it decides which distribution
> is pulled toward which — but gradient on the surviving side still flows back
> through its inputs. Since `p = p(z_t|h_t)`, `L_dyn` also trains the GRU. The
> `sg` controls **direction of pull**, not subnetwork isolation.

**Now `L_rep`.** The puzzle: the posterior saw the observation, the prior didn't,
so the posterior is strictly better informed — why drag it *toward* the prior?
That's making it worse.

Yes. On purpose. **Because not everything in `o_t` is predictable, and encoding
the unpredictable part is a bad trade.**

Take perceptual noise on an interoceptive channel — pure noise, nothing predicts
it. Then:

- `observation_loss` *wants* `z_t` to encode it (the decoder is graded on
  reproducing `o_t`, noise included).
- The prior can **never** match it → `dyn_loss` stays permanently elevated.
- The latent budget is **32 slots**. Every slot on unpredictable flicker is a slot
  not spent on the predator.
- In imagination the prior will *hallucinate* those slots, and the reward head —
  trained on posteriors where they meant something — reads noise.

**One-line version: `L_rep` is the VAE's KL term.**

| | VAE | RSSM |
|---|---|---|
| "encode everything" | reconstruction | `observation_loss` |
| "don't encode too much" | `KL(q ‖ N(0,I))` | **`L_rep`** |
| "learn a better anchor" | *doesn't exist* | **`L_dyn`** |

Difference: a VAE's anchor is a dumb fixed Gaussian, so "stay near it" = "stay
compressed". The RSSM's anchor is a **learned predictor**, so "stay near it" =
"**stay predictable**". That's the information-bottleneck term from DreamerV1
Appendix B.

**Why 0.5 vs 0.1:** both close the *same* gap `KL(q‖p)` but propose different
repairs. `L_dyn` — make the **predictor smarter** (costs nothing, keep your
information). `L_rep` — make the **target easier** (costs information, you're
deleting things). So weight the first 5× harder: the default verdict is "the gap
is the prior's fault, it should study more." Only when the prior genuinely
*cannot* catch up does the weak 0.1 pressure win.

**And it switches off** below 1 nat (free bits) — it cannot nibble at an
already-well-behaved latent.

### Q6 — `L_dyn` vs `L_rep` in more depth; how do they converge; network dims; the categorical

**Q:** [Paraphrased] `L_dyn` relates to the prior, which considers `h_{t-1}` not
`φ_t`, so it's only the prior prediction. `L_rep` uses `φ_t` with `h_{t-1}` so
it's more posterior. Reducing `L_dyn` makes the prior more similar to the
observation-informed posterior, which can be noisy. `L_rep` makes the posterior
closer to the prior — less noise but maybe biased. How can they converge? And
explain the network structures, input/output dims, and the categorical in detail.

**A:** **Correction to the premise.** (a) It's `h_t`, not `h_{t-1}` — the GRU has
already stepped. (b) `h_t` is **not** observation-free: it was built from
`z_{t-1}`, a posterior that saw `o_{t-1}`, recursively back. **`h_t` carries every
observation up to `t-1`.** So the prior isn't "the blind branch" — it's
"everything except the newest observation", and the KL measures exactly *how
surprising `o_t` is given everything already known*.

**`L_dyn` and `L_rep` are not opponents.** `L_dyn` moves `p`; `L_rep` moves `q`.
Different parameters. `L_dyn` never asks `q` to do anything — it's a pure "prior,
catch up", a follower with no competitor. The real tug-of-war is:

```
observation_loss (1.0) ──→ "encode MORE of o_t"  ─┐
                                                  ├─ same network (q)
L_rep            (0.1) ──→ "encode LESS of o_t"  ─┘

L_dyn            (0.5) ──→ "p, predict whatever q settles on"  ← acts on p; no opponent
```

**Where `L_dyn` converges — and why it isn't "noisy".** Minimising
`E[KL(q ‖ p)]` over `p` has a closed-form optimum: `p*` = the **aggregate
posterior**. That's not noisy — it's **correctly uncertain**. Where the future is
predictable it goes sharp; where it isn't, it spreads mass over the 32 classes.
The prior learns the *distribution* of the noise, not its realisations. What it
*can't* do is drive the KL to zero, because a residual remains: the genuinely
unpredictable information `I(z_t ; o_t | h_t)`.

**The equilibrium:**

| force | weight | wants |
|---|---|---|
| `observation_loss` | 1.0 | `I(z_t ; o_t | h_t)` **large** |
| `L_rep` | 0.1 | `I(z_t ; o_t | h_t)` **small** |
| `L_dyn` | 0.5 | chases whatever those settle on |
| free bits | — | switches `L_rep` off below 1 nat |

A **rate–distortion trade-off**, and the "noise vs bias" framing names both sides
correctly: encode everything → low bias, high noise (unpredictable bits, prior
hedges, imagination degrades, slots wasted); drag toward the prior → low noise,
higher bias (you deleted real information). It converges because the two forces
have opposite signs in the same quantity at a fixed 10:1 ratio, plus free bits
zeroes `L_rep` below 1 nat.

**Network structures, exactly.** `D_obs` = env obs width, `A` = n actions. Every
`Linear` has `use_bias=False` when a LayerNorm follows; every LayerNorm uses
`eps=1e-3`.

```
encoder (agent.py:1115), mlp_layers=1:
  o_t [B, D_obs] → symlog → Linear(D_obs→256) → LayerNorm(256) → SiLU → φ(o_t) [B,256]

sequence model (agent.py:580-614):
  [z_{t-1} ; a_{t-1}] [B, 1024+A]
    → Linear(1024+A→256) → LayerNorm(256) → SiLU                 [B,256]
    → LayerNormGRUCell(in=256, hidden=256) with h_{t-1}          → h_t [B,256]
         └─ internally: Linear(256+256→768) → LayerNorm(768) → chunk 3×256

transition = prior net (agent.py:626-640):
  h_t [B,256] → Linear(256→256) → LN → SiLU → Linear(256→1024)
              → logits → unimix → reshape [B,32,32] → sample → ẑ_t

repr = posterior net (agent.py:648-663):
  [h_t ; φ(o_t)] [B,512] → Linear(512→256) → LN → SiLU → Linear(256→1024)
                         → logits → unimix → reshape [B,32,32] → sample → z_t

  ^ identical to the prior net EXCEPT the input width: 512 vs 256.
    That one difference is the entire prior/posterior distinction.

downstream, all reading latent = [z_t ; h_t] = [B,1280]:
  decoder:        1280 → Linear→256 → LN → SiLU → Linear(256→D_obs)  (output in SYMLOG space)
  reward_model:   1280 → Linear→256 → LN → SiLU → Linear(256→255)    (output ZERO-init)
  continue_model: 1280 → Linear→256 → LN → SiLU → Linear(256→1)
  actor:          1280 → Linear→256 → LN → SiLU → Linear(256→A) → unimix
  critic:         1280 → Linear→256 → LN → SiLU → Linear(256→255)    (output ZERO-init)
```

**The categorical.** `z_t` is **not** 1024 floats. It's **32 independent
categorical variables, each one-hot over 32 classes**:

```
logits           [B, 32, 32]   ← 32 groups × 32 classes
softmax(axis=-1)               ← normalised WITHIN each group
sample per group               ← each group picks 1 of its 32
z_t              [B, 32, 32]   ← each row: exactly one 1.0, thirty-one 0.0
flatten          [B, 1024]     ← exactly 32 ones, 992 zeros
```

32 discrete questions, each with 32 answers. Capacity 32³² = **160 bits/step** —
finite and scarce, which is why spending slots on noise is a real cost and why
`L_rep` has a job.

Straight-through sampling (agent.py:915-923):

```python
gumbel_noise = jax.random.gumbel(key, shape=logits_2d.shape)
hard = jax.nn.one_hot(jnp.argmax(logits_2d + gumbel_noise, axis=-1), self.num_classes)
soft = jax.nn.softmax(logits_2d, axis=-1)
state = hard - jax.lax.stop_gradient(soft) + soft
```

Forward: last two cancel → `state == hard`, a real discrete one-hot. Backward:
`∂state/∂logits = ∂soft/∂logits` — the gradient pretends you passed the softmax.
Biased, far lower variance than score-function.

Unimix (agent.py:819) — 1% uniform floor:

```python
probs  = (1 - 0.01) * softmax(logits) + 0.01 * (1/32)
logits = log(probs)
```

Every class keeps ≥ `0.01/32 ≈ 3e-4` mass → **bounds the log-probs, therefore
bounds the KL**. Without it a collapsed posterior sends the KL to infinity.

The KL (loss.py:560-571):

```python
p = jnp.exp(log_p)
kl_per_cat = (p * (log_p - log_q)).sum(axis=-1)   # over 32 CLASSES → [T,B,32]
return       kl_per_cat.sum(axis=-1)              # over 32 GROUPS  → [T,B]
```

Groups are independent → joint KL is the **sum** of 32 per-group KLs
(`Independent(OneHotCategorical, 1)`). Consequence: **the 1-nat free-bits floor is
1 nat across all 32 groups combined** — ~0.03 nats/group. Tight.

**Why the categorical closes the convergence question:** `L_dyn`'s optimum is the
aggregate posterior, which is a *mixture* of the individual posteriors. **A
mixture of categoricals is itself a categorical** → the optimum is inside the
prior's hypothesis class → `L_dyn` can actually reach it. **A mixture of
Gaussians is not a Gaussian** → a Gaussian prior (DreamerV1/PlaNet) can never
match its own aggregate posterior, leaving an irreducible KL floor that is pure
model misspecification. That's a large part of why Gaussian → categorical took
Dreamer from "works on DM-Control" to "beats humans on Atari".

### Q7 — Why can the KL never be zero?

**Q:** You told me the KL never can be zero from `L_dyn` and `L_rep`. Why?
(compact)

**A:** **KL = 0 is achievable — it's just a failure mode.** It's called posterior
collapse.

At the prior's optimum (`p` = aggregate posterior), the residual is exactly:

```
E[ KL(q(z_t|h_t,o_t) ‖ p(z_t|h_t)) ]  =  I(z_t ; o_t | h_t)
```

So **KL = 0 ⟺ `z_t` carries nothing about `o_t` that `h_t` didn't already know ⟺
the posterior ignores the observation.**

Why forbidden: `observation_loss` requires rebuilding `o_t` from `[z_t ; h_t]`.
But `h_t` was formed *before* `o_t` arrived — it cannot contain `o_t`'s novelty.
So the novelty **must** be in `z_t`, forcing `I(z_t ; o_t | h_t) > 0`, forcing
KL > 0. And `p` can never absorb it: `p` is a function of `h_t` alone —
**structurally blind to `o_t` by construction**, not merely under-trained.

```
KL → 0              ⟹ posterior ignores obs ⟹ reconstruction dies
reconstruction works ⟹ z_t encodes novelty  ⟹ KL > 0
```

Exception: KL → 0 *honestly* only in a fully deterministic, fully observable world
where `h_t` already implies `o_t` — no novelty exists. Degenerate.

So the KL floor is a **measurement**: the world's per-step surprise, in nats.
That's why free bits exists — `max(KL, 1.0)` declares *"1 nat of novelty per step
is the expected cost of living, stop optimising it away."*

### Q8 — What is the aggregate posterior?

**Q:** What is the aggregate posterior?

**A:** The **average posterior over all observations that could have followed** —
"before I look, what distribution over `z_t` should I expect?"

```
q̄(z_t | h_t)  :=  E                [ q(z_t | h_t, o_t) ]
                   o_t ~ p(o_t|h_t)
```

Concrete: `h_t` = *"in the corridor, no predator seen for 5 steps."*

| `o_t` shows | how often | `q(z_t|h_t,o_t)` mass on |
|---|---|---|
| empty corridor | 70% | "empty" codes |
| predator left | 20% | "predator-left" codes |
| predator right | 10% | "predator-right" codes |

Each individual posterior is **sharp** (once you've looked, you know). The
aggregate is **broad**: `{70% empty, 20% left, 10% right}` — and that broad
distribution is exactly what a good prior *should* say. Not vague because the
network is weak; vague because the world is.

Why it's the target — the decomposition:

```
E [ KL(q ‖ p) ]  =  I(z_t ; o_t | h_t)   +   KL( q̄ ‖ p )
                    └── irreducible ──┘       └── what L_dyn can fix ──┘
                     (world's novelty)          (zero when p = q̄)
```

`L_dyn` has one job: drive the second term to zero. The first it cannot touch —
that's the KL floor from Q7.

### Q9 — Checking the mechanics of prior vs posterior inputs

**Q:** At time `t`, the prior network gets `h_t` from the recurrent network, which
used `z_{t-1}`, `h_{t-1}`, `a_{t-1}`. The posterior network uses the same `h_t`
but also gets `φ_t` from the encoder. Is this correct? Do I need to consider the
aggregate posterior here?

**A:** **Both claims correct.** `RSSM.dynamic` (agent.py:1083-1104) in full:

```python
recurrent_feat  = silu(norm(self.recurrent_mlp_linear(concat([posterior_flat, action]))))
recurrent_state = self.gru_cell(recurrent_feat, recurrent_state)          # h_t ← z_{t-1}, a_{t-1}, h_{t-1}

prior_logits, prior         = self._transition(recurrent_state)                    # sees h_t
posterior_logits, posterior = self._representation(recurrent_state, embedded_obs)  # sees h_t + φ(o_t)
```

It's literally the **same `recurrent_state` variable** passed to both — `h_t` is
computed once; prior and posterior nets are two readers of one tensor. The only
asymmetry is the extra `embedded_obs` argument.

Refinement: `z_{t-1}` is the **posterior** from `t-1` during training (`observe`
feeds `posterior_state` back round the loop). In imagination the prior sample
takes that slot.

**Aggregate posterior here? No.** It is never computed, never a tensor, appears
nowhere in the code. It's an **analytical device** — the answer to "where does
`L_dyn`'s optimum lie?" You need it to *reason* about convergence, not to *run*
anything.

You couldn't compute it anyway: it means averaging over all `o_t` that could
follow `h_t`, requiring replays of alternative futures from the same `h_t`. Each
minibatch hands you **one** `(h_t, o_t)` pair and `L_dyn` nudges `p` toward that
one posterior. Across thousands of batches, cross-entropy under SGD drifts `p`
toward their average — same reason MSE regression on noisy targets converges to
the conditional mean without ever computing a mean.

> **The aggregate posterior is where the optimiser walks. It is not something the
> code builds.**

### Q10 — Is the aggregate posterior connected with amortized inference?

**Q:** I think the aggregate posterior is connected with the amortized inference.

**A:** Half right — and the half that's off matters.

**Orthogonal by definition.** *Amortized inference* is about **how you obtain**
`q(z|x)`: one shared network mapping `x → posterior params` instead of
per-datapoint optimisation. A statement about mechanism. *The aggregate posterior*
is a **derived quantity** from whatever `q(z|x)` you have: `q̄(z) = E_x[q(z|x)]`.
You could do classical per-datapoint VI with no amortization and still form it.
Neither implies the other.

**The real connection: the prior net is an amortized estimator of the conditional
aggregate posterior.** Exact, not analogy. `q̄(z_t|h_t)` is defined by an
expectation over all `o_t` that could follow `h_t` — you'd have to enumerate
futures. Instead you train a network to output it in one forward pass. That *is*
amortization, in the same sense the encoder is.

So the RSSM has **two** amortized networks doing two different expensive things:

| network | amortizes | replaces |
|---|---|---|
| `repr_*` (posterior) | **inference** | per-datapoint variational optimisation |
| `transition_*` (prior) | **marginalization** | averaging `q` over all possible `o_t` given `h_t` |

`L_dyn` is the training signal that does the amortizing — it teaches `p` to output
a marginal it never computes.

**Which makes the RSSM a learned-prior VAE**, squarely in the VampPrior line
(Tomczak & Welling): a fixed `N(0,I)` mismatches `q̄`, leaving a **prior hole** —
regions where `p` puts mass but `q̄` doesn't, so `p`'s samples decode to garbage
the decoder never trained on. Their fix: learn `p` to approximate `q̄`. That is
`L_dyn`, conditioned on `h_t`.

**And that closes the loop.** The prior hole **is** the imagination-validity
problem. In `imagine()` the prior's samples occupy the latent slot the posterior
used during training. If `p ≠ q̄`, actor and critic read latents drawn from a
distribution the reward head and decoder never saw — the policy optimises against
a fiction. That is Ha & Schmidhuber's *"cheat the world model"* failure (Paper 1),
which they patched with a temperature knob `τ` on the MDN-RNN. DreamerV2/V3's
answer is structural: **make the prior match the aggregate posterior** (`L_dyn`,
5× over `L_rep`) and **make that match reachable** (categoricals).

### Q11 — Why is the recurrent step split into pre-projection + GRU?

**Q:** Why do you calculate `recurrent_feat` and `recurrent_state` in two steps?
Why not use a single network to infer `recurrent_state` by getting `z_{t-1}`,
`a_{t-1}` and `h_{t-1}` at once?

**A:** The GRU **already does that internally** — `LayerNormGRUCell.__call__`
(agent.py:143-147):

```python
cat = jnp.concatenate([hx, x], axis=-1)   # ALREADY concatenates hidden with input
z   = self.linear(cat)                    # ...and fuses them in ONE linear
z   = self.layer_norm(z)
reset_proj, cand_proj, update_proj = z[..., :H], z[..., H:2*H], z[..., 2*H:]
```

So "one network taking `z_{t-1}`, `a_{t-1}`, `h_{t-1}` at once" is what you get by
**deleting the pre-projection** and passing `[z_{t-1} ; a_{t-1}]` straight in as
`x`. The real question is why the extra layer:

**1. It's an embedding table (main reason).** `z_{t-1}` is **32-hot over 1024
slots**. A `Linear` on a one-hot *is* an embedding lookup. So
`Linear(1024+A → 256)` = sum the 32 selected rows of a 1024-row embedding table
(row `i*32+k` = "slot `i` took class `k`"), plus the action's embedding. Same move
as embedding word tokens before an LSTM. Without it, one weight matrix must serve
a **sparse binary** input (`z`, 3% dense) and a **dense continuous** one (`h`)
simultaneously.

**2. The GRU pays 3× per input dim.** Compress once at 1×, then let the GRU pay
3× on the smaller thing. At XS (`A=5`):

| | params |
|---|---|
| pre-projection `Linear(1029 → 256)` | 263k |
| GRU `Linear(512 → 768)` | 393k |
| **total (ours)** | **657k** |
| **direct** `Linear(1285 → 768)` | **987k** |

50% larger without it. Caveat, honestly: this depends on `dense_units < 1024`. At
the base/XL config `dense_units=1024` there's no compression and it's a wash —
the argument is real at XS, not universal.

**3. One extra nonlinearity.** With: `reset = σ(W_r·[SiLU(LN(W_p·[z,a])) ; h])` —
**nonlinear** in `z`. Direct: `reset = σ(W_r·[z,a,h])` — the gate argument is
linear in `z`.

**4. Decouples two config knobs.** `recurrent_model.dense_units` and
`recurrent_model.recurrent_state_size` are separate keys (both 256 at XS; 1024 and
4096 in the base config). Without a pre-projection there's one width and you
can't size memory independently of the input pathway.

**Honest answer:** we didn't choose any of this. `dreamer_srl` is a
line-referenced port; this is Hafner's architecture as sheeprl implements it
(`RecurrentModel`, sheeprl agent.py:309-326). It's called out loudly in our RSSM
docstring because an earlier rebuild **omitted it** — "cascade fix #30".
agent.py:409: *"without this MLP, the GRU receives raw `[posterior_flat, action]`
… a dimensional mismatch **and** a semantic difference."* It silently trains
either way, which is what made it dangerous.

### Q12 — Discretized latents in general; is VQ-VAE connected?

**Q:** I think you need to introduce me the 32-hot or any discretized latent
approach. You can start from VQ-VAE if you think it is connected.

**A:** **Connected — but family resemblance, not shared mechanism.** Both are
"discrete latents with straight-through gradients"; they solve overlapping but
different problems, and the difference is exactly why Dreamer doesn't use VQ.

**VQ-VAE (van den Oord et al., 2017):**

```
encoder → z_e(x) ∈ R^D            a continuous vector
codebook  e_1 … e_K ∈ R^D         K learned embedding vectors
quantize: z_q(x) = e_k,  k = argmin_j ||z_e(x) − e_j||₂     ← nearest neighbour
decoder ← z_q
```

- **Straight-through gradient**: `argmin` has no derivative → *copy* the gradient
  from `z_q` back to `z_e` (pretend the quantiser was the identity).
- **Codebook loss** `||sg(z_e) − e||²` pulls codes to the encoder; **commitment
  loss** `β||z_e − sg(e)||²` pulls the encoder to its code.
- **No KL at all** — the prior is uniform over K codes, so KL = const `log K` and
  drops out. The prior is learned in a **second stage** (a PixelCNN over the
  frozen discrete codes).

Side by side:

| | VQ-VAE | Dreamer V2/V3 |
|---|---|---|
| encoder emits | a **vector** `z_e` | **logits** |
| discretisation | nearest neighbour in embedding space | **sample** from a categorical |
| deterministic? | **yes** (`argmin`) | **no** — stochastic |
| codebook | explicit, learned vectors | none, explicitly |
| gradient trick | straight-through *copy* | straight-through *Gumbel-softmax* |
| prior | uniform → PixelCNN, **stage 2** | learned **jointly**, conditioned on `h_t` |
| KL | absent (constant) | **the central term** |
| pathology | **codebook collapse** (codes go unused) | KL spikes → patched by unimix |

**The difference that decides it: VQ is deterministic.** VQ-VAE hands you
*"code 47."* Dreamer hands you *"70% empty, 20% predator-left, 10% right."*
Everything from Q6–Q8 collapses under a point estimate: `q(z|h,o)` becomes a
delta, there is no aggregate posterior to train a prior against, you cannot take
a `KL` between two `argmin` operations (no probability semantics), and there is
no hedging in imagination — which was the entire reason PlaNet added the
stochastic path.

> **VQ-VAE discretises for compression. Dreamer discretises for multimodal
> belief.** Same tool, different job.

**But Dreamer *does* have a codebook — see Q11.** `recurrent_mlp_linear` is
`Linear(1024 → 256)` on a 32-hot = the sum of 32 selected rows of a 1024-row
table. That table **is** a codebook (1024 entries × 256 dims); Dreamer just never
calls it one. The real distinction is **ordering**:

- **VQ-VAE — embed first, then choose.** The code is picked by *proximity in
  embedding space*; code identity is defined by where it sits. This is what makes
  codebook collapse possible: codes far from any encoder output get no gradient
  and die.
- **Dreamer — choose first, then embed.** The code is picked by *sampling from
  logits*; the embedding is learned freely downstream. There is no "distance", so
  nothing to collapse. Unimix's 1% floor guarantees every class keeps gradient.

Dreamer's is strictly simpler and sidesteps VQ's central pathology.

**Wider landscape:**

- **Straight-through estimator** (Bengio et al., 2013) — the umbrella trick:
  forward hard, backward soft. Both approaches use it.
- **Gumbel-Softmax / Concrete** (Jang et al.; Maddison et al., 2017) — a *relaxed*
  categorical with temperature `τ`; `τ → 0` approaches one-hot. Dreamer uses the
  **hard straight-through** variant at fixed temperature, not annealed.
- **Factorisation is the scaling trick, and both use it.** VQ-VAE quantises each of
  `H'×W'` spatial positions independently (e.g. 1024 positions × 512 codes ≈ 9216
  bits). Dreamer: 32 abstract slots × 32 classes = **160 bits**. Same structural
  move; Dreamer's slots aren't spatial and its budget is tiny, because it encodes
  a *state*, not an image.
- **VQ world models exist**: IRIS tokenises with a VQ-VAE + transformer over
  tokens. DreamerV3 outperforms it on Atari 100k.

**The arc — where multimodality lives:**

| | latent | who handles multimodality |
|---|---|---|
| PlaNet / DreamerV1 | Gaussian | **nobody** — the weakness V2 fixed |
| DreamerV2 / V3 | 32×32 categorical | **the latent** |
| **Dreamer 4** | **continuous, tanh-projected bottleneck** | **the dynamics** (shortcut/diffusion forcing) |

Dreamer 4 went *back to continuous* at the tokenizer. Not a reversal of V2's
lesson — a **relocation**. Multimodality has to live somewhere; a diffusion model
is natively multimodal, so once the dynamics is a diffusion-forcing transformer,
the latent no longer carries that burden and can return to being a smooth,
easy-to-predict continuous code.

> Discrete latents were never the goal. **Representing "either A or B" was the
> goal**, and 32×32 one-hots were the cheapest way to get it inside a GRU. When a
> better multimodal dynamics model arrived, the discreteness was dropped.

### Q13 — The categorical approach explained from scratch

**Q:** Explain me the dreamer's categorical approach again. Think I'm new for this
categorical approach.

**A:** From the ground up.

**1. A network cannot output "a category."** It outputs numbers. To represent a
choice among 32 options it emits **32 numbers** read as preferences:

```
raw output (logits):   [ 2.0,  1.0,  0.1 ]
softmax:               [ 0.66, 0.24, 0.10 ]    sum = 1
```

`softmax(x)_k = exp(x_k) / Σⱼ exp(x_j)`. The raw numbers are **logits** —
unnormalised log-probabilities.

**2. One-hot, and why not an integer.** Having chosen option 2, write it
`[0, 1, 0]`. Why not the integer `2`? Because a number carries arithmetic that
categories don't have — `2` implies it sits *between* 1 and 3 and is *twice* 1.
For options `{empty, predator, food}` that's nonsense. **One-hot strips out all
accidental ordering and magnitude**; every option is equidistant from every other.

**3. Why a categorical latent at all.** Truth: *"the predator is at position 3 or
position 8."*

```
truth:        ▁ ▁ █ ▁ ▁ ▁ ▁ █ ▁ ▁      either here or there
Gaussian:     ▁ ▂ ▄ ▆ █ ▆ ▄ ▂ ▁ ▁      "around 5.5 ± 2"  ← WRONG. It's never at 5.5.
Categorical:  ▁ ▁ █ ▁ ▁ ▁ ▁ █ ▁ ▁      "50% at 3, 50% at 8"  ← exactly right
```

A Gaussian is **unimodal**; asked for two possibilities it puts mass in the *gap*,
on a state that never occurs. That is DreamerV2's whole case for the swap.

**4. Capacity, and the factorisation fix.** One question with 32 answers = only
`log₂32 = 5 bits`. Two ways to get more: one huge question (2¹⁶⁰ answers → 2¹⁶⁰
logits, impossible), or **many small questions** ✅.

```
32 slots × 32 choices = 32³² = 2¹⁶⁰ ≈ 1.5e48 states   ...from only 32×32 = 1024 logits
```

Like a 32-character password over a 32-letter alphabet. **Combinatorial capacity
at linear parameter cost.** That's what makes discrete latents practical.

**5. The shapes.**

```
transition_out / repr_out  →  1024 numbers
reshape                    →  [32, 32]      32 groups × 32 classes
softmax(axis=-1)           →  per-group probabilities  ← normalised WITHIN each row only
sample each group          →  each row picks one class
z_t                        →  [32, 32]      each row: one 1.0, thirty-one 0.0
flatten                    →  [1024]        exactly 32 ones, 992 zeros
```

Softmax runs **only along the last axis** → the 32 groups are independent → the KL
is a *sum* of 32 per-group KLs.

**6. You can't differentiate a sample.** `one_hot(argmax(logits))` is a
**staircase** in the logits — flat everywhere, jumping at boundaries. Derivative
zero almost everywhere. Three tricks fix it:

- **Gumbel-max** — adding Gumbel noise then taking argmax gives an *exact*
  categorical sample: `k = argmax_j(logit_j + g_j)`, `g_j ~ Gumbel(0,1)`. It
  relocates randomness into an **additive noise term**, leaving the logits on a
  deterministic path — the same move as the VAE's `z = μ + σ·ε`.
- **Straight-through** (agent.py:915-923):
  ```python
  hard  = one_hot(argmax(logits + gumbel))
  soft  = softmax(logits)
  state = hard - stop_gradient(soft) + soft
  ```
  Forward: `hard − soft + soft = hard`, a genuine discrete one-hot. Backward:
  `∂state/∂logits = ∂softmax/∂logits`. **The forward pass is a staircase; the
  backward pretends it was a smooth ramp.** A lie → **biased**, but far lower
  variance than REINFORCE.
- **Unimix** (agent.py:819) — if softmax gives ~0 to a class, `log 0 = −∞` and the
  KL explodes. So `probs = 0.99·softmax(logits) + 0.01·(1/32)`; every class keeps
  ≥ `0.0003` mass → log-probs bounded → **KL bounded**. (`softmax(log p) = p`, so
  calling the result "logits" is consistent.)

**7. One group end to end.**

```
raw logits (group 7):  [ 3.2,  0.1, -1.5, …,  0.4 ]
softmax:               [0.710, 0.031, 0.006, …, 0.043]
unimix:                0.99 × 0.710 + 0.0003 = 0.7033  →  [0.7033, 0.0306, …]
log:                   [-0.352, -3.487, …]             →  passed on as "logits"
+ gumbel:              [ 0.2, 1.8, -0.3, … ]
argmax(logits+gumbel): → 1     ← noise favoured class 1 despite lower probability
hard:                  [0, 1, 0, …, 0]
state = hard − sg(soft) + soft  =  [0, 1, 0, …, 0]     ← value hard; gradient via soft
```

Group 7 answered **"class 1."** Repeat ×32 → `z_t`.

**8. Two things newcomers assume wrongly.**

- **The slots have no assigned meaning.** Nobody designs slot 7 to be "predator
  direction." The network decides; the code may be distributed and
  uninterpretable. Learned, not engineered.
- **The factorisation has a real cost.** Given a fixed input `(h_t, o_t)` the 32
  groups are independent — the posterior **cannot represent correlations between
  slots at the same timestep**. If "slot 3 = A" should imply "slot 9 = B", a
  factorised distribution can't express that within one step. Correlations *across
  the data* are captured (different inputs → different logit patterns);
  correlations *given the input* are not. That's the price of the combinatorial
  capacity, and it's rarely mentioned.

## Session 2026-07-15/16 — the non-differentiable-sample problem (step-by-step)

> Asked as a slow walkthrough rather than one answer. Steps 1–2 plus two detours
> are below as Q14–Q18; Q19 is the provenance finding that reframed the whole
> thread and triggered a correction to [[dreamer_lit_review]].

### Q14 — Step 1: where the gradient needs to go

**Q:** I understood until §5. From §6 "The hard part: you can't differentiate a
sample", explain step-by-step.

**A:** The path from the posterior net's weights to the loss:

```
repr_out weights ─► logits ─► z_t ─► [z_t ; h_t] ─► decoder ─► ô_t ─► observation_loss
```

Chain rule:

```
∂loss/∂weights = (∂loss/∂ô_t) × (∂ô_t/∂z_t) × (∂z_t/∂logits) × (∂logits/∂weights)
                                                    ↑ the problem
```

Every other factor is smooth (squared distance; Linear/LayerNorm/SiLU; Linear).
**The whole problem reduces to `∂z_t/∂logits`** — and that arrow is not a formula,
it's a *sampling operation*. If it's zero, the entire product is zero: not "small",
**exactly zero**. Encoder, posterior net, GRU all receive nothing. Pass/fail for the
world model.

### Q15 — Step 2: what that derivative is, and why it's zero

**A:** One group, 3 classes, randomness frozen. Slide `ℓ₁` upward:

| `ℓ` | argmax | `z` |
|---|---|---|
| `[0.00, 1.0, 0.5]` | 2 | `[0,1,0]` |
| `[0.90, 1.0, 0.5]` | 2 | `[0,1,0]` |
| `[0.99, 1.0, 0.5]` | 2 | `[0,1,0]` |
| `[1.01, 1.0, 0.5]` | **1** | **`[1,0,0]`** ← jump |

```
z₁
 1 |                     ┌───────────
 0 |─────────────────────┘
   └─────────────────────┴──────────► ℓ₁
                        1.0
```

Derivative: **exactly 0** on the flats (almost everywhere), **undefined** at the
jump. Never useful.

**Why zero is worse than small:** it's a *confident wrong statement* — "changing
`ℓ₁` has no effect." And locally that's **true**. All the information lives in a
jump the derivative cannot see. The optimiser concludes there's nothing to do.

**This is discrete-specific.** Gaussian: `z = μ + σ·ε` → `∂z/∂μ = 1`. Nonzero,
constant, perfect. That's why VAEs "just work" and nobody discusses this.

### Q16 — Why does the staircase happen at all? Is it connected to 32×32?

**A:** **The staircase IS the discreteness.** Topologically forced:

> A continuous map from a connected space to a finite set **must be constant**.

Logits live in `R³²` (connected). Output lives in `{32 one-hots}` (isolated
points). Any map is constant (useless) or discontinuous (staircase). No third
option. **Deciding `z` is discrete = accepting the staircase.**

**Why not just use the softmax as `z`?** Geometry — the simplex:

```
              [1,0,0]  ← vertex: a REAL category
                 ▲
                ╱ ╲
               ╱  ● ╲     ← interior: [0.7,0.2,0.1] — a BLEND, no such state exists
              ╱_____╲
        [0,1,0]     [0,0,1]
```

One-hot = **vertex**. Softmax output = **interior**. Handing the decoder an
interior point hands it something that corresponds to nothing in the world —
which is the Gaussian's "mass at 5.5" failure wearing new clothes.

**And it destroys imagination.** Predator goes left or right 50/50. Softmax-as-`z`:
every rollout identical, following the blend — *"the predator goes straight"*. The
agent learns to dodge a predator that never exists. Sampled one-hot: this rollout
left, next rollout right — each a **concrete, plausible** future.

**Which decision causes what:**

| decision | why | causes staircase? |
|---|---|---|
| **discrete** | multimodality — "either A or B" | ✅ **the whole cause** |
| **factorised** (32 groups) | capacity: 32³² states from 1024 logits | ❌ no — just 32 staircases |

The bargain: `want multimodality → must go discrete → must accept a staircase →
must fake the gradient`.

### Q17 — Are you comparing sampling vs one-hot?

**A:** No — that table was badly built. They are **two independent axes**, each
with its own alternative:

| | **blend** (interior) | **one-hot** (vertex) |
|---|---|---|
| **deterministic** | `z = softmax(ℓ)` — mean-field failure, a non-state | `z = one_hot(argmax(ℓ))` — the **mode**. Real state, zero diversity |
| **stochastic** | `z = softmax((ℓ+g)/τ)` — soft Gumbel-Softmax. Random, still a blend | `z = one_hot(argmax(ℓ+g))` ← **Dreamer** |

Proof they're separable — **our code uses two of these cells** (agent.py:901-919):
`get_initial_states` takes the `sample=False` branch (deterministic + vertex, the
mode); every other RSSM call takes `sample=True` (stochastic + vertex).

The staircase comes from the **support axis alone** — plain `argmax`, no noise,
still has it. Randomness isn't the culprit; landing on vertices is.

### Q18 — What is the Gumbel? (and: the dice example was wrong)

**Q:** How does the Gumbel become the distribution of the maximum? Rolling 1000
dice and taking the max sounds like it should skew to 6, not have mean 0.577 with
a long right tail.

**A:** **The dice example was wrong.** Dice are **bounded** — the max of 1000 rolls
is 6, essentially always. Fisher–Tippett–Gnedenko says the max of many iid samples
converges to one of **three** families, depending on the tail:

| tail | limiting max distribution |
|---|---|
| **bounded** (dice, uniform) | **Weibull** ← piles up at the ceiling |
| **heavy / power-law** (Cauchy, Pareto) | **Fréchet** |
| **unbounded, exponentially decaying** (Gaussian, exponential) | **Gumbel** ✅ |

Dice are in the **Weibull** branch. Wrong example entirely.

**Correct example:** `X₁…Xₙ ~ Exponential(1)`. `P(max ≤ x) = (1−e^{−x})ⁿ`. Sub
`x = log n + y`: `(1 − e^{−y}/n)ⁿ → exp(−e^{−y})` — the Gumbel CDF, exactly. Two
things: the max **grows** (it sits near `log n`), and we **subtracted that growth**.
The Gumbel describes `max − log n`, the *fluctuation around the typical max*. That's
why it's centred at 0.577 rather than pinned at a ceiling — there is no ceiling.

**Why right-skewed:** to land far **below** typical, *every* sample must be small —
an **AND**, so probabilities multiply → **double-exponential** collapse
(`exp(−e^{|x|})`). To land far **above**, only **one** sample must be big — an
**OR** → merely **exponential** decay (`e^{−x}`). Left cliff, right ramp. The skew
is the difference between AND and OR.

**How it makes softmax exact** — not via the max story, via a race:

```
Gumbel = −log( Exponential(1) )        check: P(−log T ≤ x) = P(T ≥ e^{−x}) = exp(−e^{−x}) ✓
```

Give every class an **alarm clock** ringing at rate `e^{ℓ_k}`. Higher logit → faster
clock. Whoever rings first wins. The classic exponential-race fact:

```
P(clock k rings first) = λ_k / Σ_j λ_j = e^{ℓ_k} / Σ_j e^{ℓ_j} = softmax(ℓ)_k
```

**Winning probability is your share of total rate.** And that race *is* the trick:

```
T_k ~ Exp(1);   T_k/e^{ℓ_k} ~ Exp(e^{ℓ_k});   argmin_k(T_k/e^{ℓ_k})  → softmax ✓
−log(T_k/e^{ℓ_k}) = ℓ_k + (−log T_k) = ℓ_k + g_k,   g_k ~ Gumbel(0,1)
−log is decreasing  ⟹  argmin over clocks ≡ argmax over (ℓ_k + g_k) ✓
```

Same race, two coordinate systems. The hinge: `g ~ Gumbel(0,1)` has CDF
`exp(−e^{−x})`, and `g + ℓ` has CDF `exp(−e^{ℓ}·e^{−x})` — **adding `ℓ` multiplies
the rate by `e^{ℓ}`.** Softmax needs to exponentiate the logits; shifting a Gumbel
exponentiates them for free. No other distribution does this.

Bonus: `∏_k exp(−e^{ℓ_k}e^{−x}) = Gumbel(logsumexp(ℓ), 1)` — the max of Gumbels is
a Gumbel centred at the softmax's own normaliser, and it's independent of the
argmax.

### Q19 — Did the Gumbel come from the Dreamer papers, or from our JAX port?

**Q:** Is this Gumbel trick from the original Dreamer papers? Or from the JAX
implementation?

**A:** **VERIFIED — it is not in the papers.** Searched the PDFs directly
(`fitz`, since `pdftotext` isn't installed — the first attempt returned empty
output that was *tool failure*, not evidence):

```
'gumbel' hits:   DreamerV2 (2021):        0
                 DreamerV3 preprint:      0
                 DreamerV3 Nature (2025): 0
```

Nor in sheeprl (`grep -rin gumbel vendor/sheeprl/` → nothing).

**DreamerV2 Algorithm 1, verbatim from the PDF:**

```
Algorithm 1: Straight-Through Gradients with Automatic Differentiation
    sample = one_hot(draw(logits))          # sample has no gradient
    probs  = softmax(logits)                # want gradient of this
    sample = sample + probs - stop_grad(probs)
```

`draw(logits)` — **the paper does not specify how you draw.** It cites
**Bengio et al. 2013** for straight-through. **Zero** hits for Jang or Maddison
(the one "Maddison" hit in the V3 preprint is Chris Maddison as an *AlphaGo
co-author*, unrelated to the Concrete distribution).

**Provenance chain:**

| where | fills `draw()` with |
|---|---|
| Hafner's papers | **unspecified** |
| sheeprl | PyTorch multinomial (`OneHotCategoricalStraightThrough`) |
| our JAX port | **Gumbel-max** (`agent.py:915`, `:1542`) |

The Gumbel came from the **JAX port**, filling a slot Hafner left open. It's a
standard, exact sampler (and what `jax.random.categorical` uses internally),
chosen because JAX wants an explicit PRNG key. This is precisely why **D-009** is
approvable as substrate-class: `draw()` is a free slot, so any correct sampler
fills it without changing the algorithm.

**Two self-corrections this forced:**

1. **The reparameterisation framing (Q18 and before) was imported, not the
   lineage's.** Hafner never frames the categorical path as reparameterisation.
   The papers' story is flat: *draw a sample somehow → attach the softmax gradient
   → cite Bengio*. Several messages of scaffolding were built on a concept this
   lineage does not use.
2. **[[dreamer_lit_review]] had a real error** — it rendered Algorithm 1 as
   `onehot(argmax(ℓ))` (the deterministic *mode*) instead of `onehot(draw(ℓ))` (a
   *sample*), and called ST "unbiased" while the paper says *"This results in a
   biased gradient estimate with low variance."* **Fixed 2026-07-16**, with an
   additive correction note.

**Root cause of the confusion:** our own code comments call the technique
"straight-through Gumbel-softmax" (agent.py:882, 907, 1413, 1531, 1534, 1541).
That is a **misnomer** — see Q20.

### Q20 — Step 3(b): how does `sample + probs − sg(probs)` recover the gradient?

**Q:** Let's follow the paper's version, not our script. How can
`sample + probs - sg(probs)` recover the gradient?

**A:**

```
sample = one_hot(draw(logits))          # a vertex. No gradient.
probs  = softmax(logits)                # smooth. Has a gradient.
output = sample + probs - sg(probs)
```

**Forward.** `sg(probs)` is numerically identical to `probs` — stop-gradient
changes the tape, never the value. So `output = sample + probs − probs = sample`,
**bit-exact**. With `ℓ = [2.0, 1.0, 0.1]` → `probs = [0.659, 0.242, 0.099]`, and
`draw` returning class 1:

```
output = [0,1,0] + [0.659,0.242,0.099] − [0.659,0.242,0.099] = [0,1,0]
```

The decoder gets a clean vertex. No blend, no softening.

**Backward.**

| term | `∂/∂logits` | why |
|---|---|---|
| `sample` | **0** | the staircase — `draw` returns an integer index |
| `probs` | **J** (softmax Jacobian) | smooth |
| `− sg(probs)` | **0** | definition of stop-gradient |

`∂output/∂logits = 0 + J − 0 = J`.

**The insight — `probs − sg(probs)` is a zero with a gradient attached:**

```
value:       probs − probs = 0        ← identically zero, always
derivative:    J   −   0   = J        ← the full softmax Jacobian
```

You add **zero** to the sample. The value cannot change. But the derivative of
that nothing is `J`, and autodiff picks it up. Two channels through one
expression: the **value** channel carries `sample` (the `probs` cancel); the
**gradient** channel carries `J` (the `sample` and `sg` vanish).

**The lie, precisely:**

```
forward computes:   z = one_hot(draw(ℓ))    ← staircase, derivative 0
backward reports:   z = softmax(ℓ)          ← smooth ramp, derivative J
```

Different functions → **biased**. The paper says so outright: *"This results in a
biased gradient estimate with low variance."*

**Why it's a good lie:** `E[sample] = probs`. The softmax **is** the expectation of
the one-hot. ST says *"I can't differentiate the sample, so I'll differentiate its
mean."* And the split is cleaner than it looks:

| piece | honest? |
|---|---|
| `∂loss/∂z` evaluated **at the vertex actually drawn** | **honest** — that's where the decoder ran |
| `∂z/∂ℓ ≈ J`, routing it back to the logits | **fake** — this is the lie |

The loss signal is real; only the routing is invented. The sample still matters —
it decides **where** the loss is evaluated, even though `J` is the same regardless.

**In words:** with class 1 drawn, `∂p₁/∂ℓ₁ = 0.242(1−0.242) = +0.183` and
`∂p₁/∂ℓ₀ = 0.242(0−0.659) = −0.159`. If class 1 lowered the loss, the update
raises `ℓ₁` and lowers its main rival `ℓ₀`. Direction right; magnitude fiction.

**The trade:**

| estimator | bias | variance |
|---|---|---|
| REINFORCE (score function) | **unbiased** ✅ | **enormous** ❌ |
| straight-through | **biased** ❌ | **small** ✅ |

Dreamer takes ST for the *world model*. Note it makes the **opposite** call for the
**actor** on Atari (`ρ=1`, pure REINFORCE) — same menu, different dish, because the
actor's gradient traverses a long imagined rollout where ST's bias compounds.

### Q21 — Why not `sample − probs + sg(probs)`? Isn't the arrangement arbitrary?

**A:** Not arbitrary — that version **flips the sign of the gradient**.

```
output = sample − probs + sg(probs)
forward:   sample − probs + probs  =  sample     ← identical! ✓
backward:  0      −  J   +  0      =  −J         ← the exact negative ✗
```

Gradient descent becomes gradient **ascent**. If class 1 lowered the loss, the
correct update raises `ℓ₁`; this one lowers it. The model learns *"that choice
helped → make it less likely."* It anti-learns and diverges.

**The general family:** `output = sample + α·(probs − sg(probs))`. The bracket is
always zero in value, so the forward is `sample` for **any** `α`. Only the backward
changes:

| expression | forward | backward | verdict |
|---|---|---|---|
| `sample + probs − sg(probs)` | `sample` ✓ | `+J` | ✅ the paper |
| `sample − probs + sg(probs)` | `sample` ✓ | `−J` | ❌ anti-learning |
| `sample + 2(probs − sg(probs))` | `sample` ✓ | `2J` | ⚠️ right direction, wrong scale |
| `sample` alone | `sample` ✓ | `0` | ❌ no learning |
| `probs + sg(sample − probs)` | `sample` ✓ | `+J` | ✅ same thing, different spelling |

**Two constraints pin it:** (1) forward = `sample` → any `α`; (2) backward =
`∂probs/∂ℓ` → **forces `α = 1`**. Constraint 2 *is* the design intent — "pretend the
output was `probs`." Not `−probs`, not `2·probs`.

**The clearer spelling:** `output = probs + sg(sample − probs)` — *"start from
`probs`, which has the gradient I want; snap the value onto `sample` with a
stop-gradient'd offset."* Identical maths; the ordering shows the intent.

**⚠️ The part that should worry you:** every row's **forward** column says `sample`.
**The forward pass cannot tell you which arrangement is correct.** A sign typo
passes every value-based test — shapes match, one-hots are perfect, reconstruction
looks fine — and the model simply never learns, with nothing in the output saying
why. Same class as the `soft = softmax(logits)` trap: **sign and stop-gradient
placement live entirely in the backward pass, where no forward assertion can see
them.** This is exactly why the project has `tests/algorithms/dreamer_srl/test_grad_parity.py`
comparing `jax.grad` vs `torch.autograd` head-by-head. Forward-only tests are
structurally blind to this bug.

### Q22 — Is `sample + probs − sg(probs)` Hafner's invention?

**A:** **No, and he says so.** Verbatim from the DreamerV2 PDF:

> *"In DreamerV2, we instead use a vector of several categorical variables and
> optimize them using **straight-through gradients (Bengio et al., 2013)**, which
> are easy to implement using automatic differentiation **as shown in Algorithm 1**."*

The grammar is explicit: the **technique** is Bengio's; Algorithm 1 is a
convenience *"easy to implement… as shown in"*, not a claimed contribution.

| | contribution |
|---|---|
| **Hinton (2012)**, lecture 15b | the original "pretend the derivative is 1" idea — *believed*, unverified (no PDF locally) |
| **Bengio, Léonard & Courville (2013)** | *Estimating or Propagating Gradients Through Stochastic Neurons…* — formalises it, names it, compares vs REINFORCE |
| **autodiff folklore** (~2015+) | the `a + b − sg(b)` spelling — a custom gradient without a custom op |
| **Hafner (2021)** | **none of the above** — *applies* borrowed machinery to categorical latents in an RSSM |

**The idiom predates Dreamer.** VQ-VAE (2017) uses it, spelled the other way:
`z_q = z_e + sg(quantize(z_e) − z_e)` — *"copy gradients from decoder input to
encoder output."* Four years earlier.

**One general recipe; both are special cases.** To build a node whose **value** is
`f(x)` but whose **gradient** is `g′(x)`:

```
out = g(x) + sg( f(x) − g(x) )
value:     g + (f − g) = f     ✓
gradient:  g′ +   0    = g′    ✓
```

| | `f` (value you want) | `g` (gradient you want) |
|---|---|---|
| **Dreamer** | `sample` (a vertex) | `probs` (softmax) |
| **VQ-VAE** | `quantize(z_e)` (a codebook entry) | `z_e` (identity → gradient copied straight through) |

That makes Q12's "family resemblance" exact: they are the **same construction with
different `f` and `g`**. The whole choose-then-embed vs embed-then-choose
distinction lives entirely in what you plug into `f`.

**So what IS DreamerV2's contribution here?** The **architectural choice**
(Gaussian → 32×32 categorical in an RSSM), **KL balancing**, and the empirical
demonstration. Straight-through is enabling machinery, borrowed and credited.
Dreamer-specific in this area: categorical latents ✅, KL balancing ✅;
straight-through ❌ (Bengio's); Gumbel-max ❌ (nobody's in this lineage — our JAX
port's choice, filling `draw()`).

## Session 2026-07-16 — world-model forward pass, module-by-module

### Q23 — Summary of world-model training

**Q:** Give me the summary of your explanation.

**A:** Seven modules, **one** loss scalar, **one** backward pass, **one** optimizer
(`wm_opt`). XS: `z` = 32×32 = 1024 flat, `h` = 256, latent = **1280**.

```
batch from replay:   obs [64,16,D_obs], actions, rewards, terminated, is_first

§S1   is_first[0] := 1                                   train.py:690
§S2   actions := action_shift(actions)                   train.py:696   (a_t → a_{t-1})

for t in 0..63:                                          agent.py:1687
    h_t  = GRU( MLP([z_{t-1} ; a_{t-1}]), h_{t-1} )      ← memory.  NO observation.
    ẑ_t  = sample( prior_net(h_t) )                      ← PREDICTION   (sample DISCARDED)
    z_t  = sample( posterior_net([h_t ; φ(o_t)]) )       ← MEASUREMENT  (kept)

latent = [z_t ; h_t]                                     [64,16,1280]
ô_t, r̂_t, ĉ_t = decoder / reward_model / continue_model (latent)

loss  =  1.0·(obs + reward + continue NLL)  +  0.5·max(KL(sg q‖p),1)  +  0.1·max(KL(q‖sg p),1)
one backward → one Adam step (clip 1000) → all seven modules
```

The six load-bearing points: (1) reconstruction is same-`t`, prediction lives in the
KL; (2) the posterior is anchored by reconstruction, not the KL (DreamerV2: remove
image gradients → 0.04 gamer median); (3) `L_dyn` and `L_rep` are not opponents —
`L_dyn` is a follower chasing the aggregate posterior; the real tug-of-war is
`L_pred` vs `L_rep` at 10:1; (4) the KL floor `I(z_t;o_t|h_t)` is the world's
per-step surprise, not an error; (5) the prior's *sample* is discarded during
training — only logits reach the KL; (6) discreteness → staircase → straight-through.

### Q24 — Is `q` the posterior and `p` the prior?

**A:** Yes. `q` = posterior `q(z_t|h_t,φ(o_t))` (sees the obs); `p` = prior
`p(z_t|h_t)` (doesn't).

```
L_dyn = KL( sg(q) ‖ p )     posterior frozen  →  the PRIOR moves
L_rep = KL(  q ‖ sg(p) )    prior frozen      →  the POSTERIOR moves
```

Mnemonic: **whichever isn't stop-gradient'd is the one being trained.**

⚠️ **DreamerV1 uses the opposite convention** (`p` = posterior, `q` = prior). It
flipped at V2. Our code follows V2/V3.

### Q25 — Does the GRU depend on `L_pred` only? And explain `z_t`

**A:** **All three losses reach the GRU** — it's the most heavily-supervised module.
Four paths per timestep:

| # | path | from |
|---|---|---|
| 1 | `decoder(latent)` → `latent[1024:]` = **`h_t`** | `L_pred` — **direct** |
| 2 | `decoder(latent)` → `latent[:1024]` = `z_t` → `posterior_net` → `h_t` | `L_pred` — via `z` |
| 3 | `prior_net(h_t)` → `h_t` | `L_dyn` |
| 4 | `posterior_net([h_t ; φ(o_t)])` → `h_t` | `L_rep` |

Path 1 is the one people miss: **`latent = [z_t ; h_t]`, so `h_t` goes straight into
the decoder.** It isn't internal plumbing — it's half the decoder's input. (The
bottleneck claim is narrower than "all observations": `h_t` carries `o_1…o_{t-1}`;
it's **`o_t`'s novelty** that has only one route, via `z_t`.) Then multiply by BPTT
over all 64 steps.

**The hidden premise in "which loss updates `z_t`'s weights":**

> **`z_t` has no weights. It isn't a parameter.**

| | what it is | trained? |
|---|---|---|
| `h_t` | **activation** — GRU output | ❌ the **GRU's weights** are |
| `z_t` | **activation** — a sample from the posterior net's logits | ❌ the **posterior net's weights** are |
| `initial_recurrent_state` | a genuine **parameter** (agent.py:673) | ✅ `tanh`'d → `h_0` |

`z_t` has **two destinations**: `[z_t ; h_t]` → heads at time `t`, **and** → GRU input
at `t+1`. That second path is how a measurement becomes memory — and why a gradient
arriving at step 40 can reach the encoder's weights at step 3.

### Q26 — The forward pass, with the models

**A:**

```
Stage 1 — encode everything ONCE, outside the loop        agent.py:1671
    embedded_obs = encoder(obs)      [64,16,D_obs] ──► [64,16,256]

Stage 2 — initialise
    h, z = rssm.get_initial_states(16)
        h [16,256] = tanh(initial_recurrent_state)   ← the one learned "state" param
        z [16,32,32] = the prior's MODE at h₀        ← deterministic, no PRNG

Stage 3 — the loop over time (sequential: h_t needs h_{t-1})

  ╔══ INPUTS to tick t ════════════════════════════════════════════╗
  ║  z_{t-1} [16,32,32]   h_{t-1} [16,256]   a_{t-1} [16,A]        ║
  ║  φ(o_t)  [16,256]     is_first [16,1]                          ║
  ╚════════════════════════════════════════════════════════════════╝

  (a) RESET (§S4, only if is_first=1)
        a_{t-1} := (1−is_first)·a_{t-1}
        h_{t-1} := (1−is_first)·h_{t-1} + is_first·h₀
        z_{t-1} := (1−is_first)·z_{t-1} + is_first·z₀

  (b) MEMORY — sequence model                        ◄── NO observation
        concat([z_{t-1}, a_{t-1}])       [16,1024+A]
          → recurrent_mlp_linear → norm → SiLU       [16,256]
          → gru_cell(feat, h_{t-1})      ═══► h_t    [16,256]

  (c) PRIOR — transition_*                           ◄── reads h_t ONLY
        h_t → transition_hidden → norm → SiLU → transition_out  [16,1024]
          → unimix → reshape → sample  ═══► ẑ_t

  (d) POSTERIOR — repr_*                             ◄── reads h_t AND φ(o_t)
        concat([h_t, φ(o_t)])            [16,512]
          → repr_hidden → norm → SiLU → repr_out     [16,1024]
          → unimix → reshape → sample  ═══► z_t

  ╔══ OUTPUTS ═════════════════════════════════════════════════════╗
  ║  h_t, z_t → tick t+1     z_logits, ẑ_logits → saved for the KL ║
  ║  ẑ_t → DISCARDED  (agent.py:1688)                              ║
  ╚════════════════════════════════════════════════════════════════╝

Stage 4 — stack & assemble
    latent = concat([z_all_flat, h_all], -1)         [64,16,1280]

Stage 5 — heads (parallel, no time dependence)
    ô = decoder(latent)                [64,16,D_obs]  ← SYMLOG space
    r̂_logits = reward_model(latent)     [64,16,255]
    ĉ_logits = continue_model(latent)   [64,16,1]
```

Five tensors out. Then the losses are just *"compare these against the batch"* —
the first three look **outward** at the data; the fourth (`z_logits` vs `ẑ_logits`)
looks **inward**, the model grading itself on how well it saw its own measurement
coming.

### Q27 — What are the inputs to generate `h_t`? Both `z_{t-1}` and `h_{t-1}`?

**A:** **Three inputs** — yes, both, plus the action. But they enter **differently**:

```
z_{t-1} ──┐
          ├─► concat ─► recurrent_mlp_linear ─► norm ─► SiLU ─► feat [256] ──┐
a_{t-1} ──┘            (1024+A → 256)                                        │
                                                                             ├─► gru_cell
h_{t-1} ─────────────────────────────────────────────────────────────────────┘
   (raw, no preprocessing)
```

`z_{t-1}`, `a_{t-1}` are the GRU's **input**; `h_{t-1}` is its **hidden state**
(agent.py:1081-1087 — `recurrent_input` does *not* contain `h`). They finally meet
inside the cell: `cat = concat([hx, x])` → `Linear(512→768)` → 3 gates.

The asymmetry is deliberate: `z_{t-1}` is **32-hot sparse binary** → needs an
embedding table (`recurrent_mlp_linear` = sum the 32 selected rows). `h_{t-1}` is
already dense at the right width → nothing to embed.

During training `z_{t-1}` is the **posterior**; in imagination the **prior** takes
that slot.

### Q28 — Why `Linear → norm → SiLU` for `z` and `a`?

**A:** **It isn't special to `z`/`a`** — it's DreamerV3's universal MLP block. Every
module uses `Linear(bias=False) → LayerNorm(eps=1e-3) → SiLU`: encoder, decoder,
recurrent_mlp, transition, repr, reward, continue, actor, critic.

What each piece does **here**:

- **Linear** — the embedding (for a one-hot, a Linear *is* a lookup: sum 32 rows).
- **LayerNorm** — scale control, and load-bearing here: `feat` gets **concatenated
  with `h_{t-1}` and fed through one linear** inside the GRU. `feat` is a sum of 32
  embedding vectors whose magnitude drifts as the table trains; `h` is a
  gate-bounded GRU output. LN pins them to comparable scale.
- **SiLU** — the real nonlinearity. Without it you'd have two linears with a
  normalisation sandwiched. It's also what makes the gate arguments *nonlinear* in
  `z`.

Two details: **`bias=False`** because LayerNorm subtracts the mean and supplies its
own shift `β`. **`eps=1e-3`, not the 1e-6 default** — Hafner's choice; our docstring
(agent.py:80) records that the nnx default produces 1.72e-3 forward drift vs the
reference.

### Q29 — Brief intro: LayerNorm and SiLU

**LayerNorm** (Ba, Kiros & Hinton 2016) — rescale each sample's feature vector to
zero mean / unit variance, **across features, within one sample**:

```
μ = (1/D) Σᵢ uᵢ ;   σ² = (1/D) Σᵢ (uᵢ−μ)²
LayerNorm(u)ᵢ = γᵢ·(uᵢ−μ)/√(σ²+ε) + βᵢ          γ, β learnable
```

For `[16,256]`: each of the 16 rows independently gets mean 0, var 1 across its own
256 features.

| | normalises over | depends on other samples? | train/eval modes? |
|---|---|---|---|
| **BatchNorm** | the **batch**, per feature | ✅ | ✅ running stats |
| **LayerNorm** | the **features**, per sample | ❌ | ❌ |

BatchNorm is unusable in an RNN (per-timestep stats, breaks at batch 1, needs
running averages). Hence LN in RNNs and transformers. Dreamer is an RNN.

**SiLU / Swish** — `SiLU(x) = x·sigmoid(x) = x/(1+e⁻ˣ)`.

| `x` | −3 | −2 | **−1.278** | −1 | 0 | 1 | 2 | 3 |
|---|---|---|---|---|---|---|---|---|
| `SiLU(x)` | −0.14 | −0.24 | **−0.278** ← min | −0.27 | 0 | 0.73 | 1.76 | 2.86 |

Large positive `x` → `≈ x` (like ReLU); large negative → `→ 0` (like ReLU); **in
between it dips below zero** → non-monotonic; smooth everywhere. Beats ReLU because:
smooth gradients; **no dying-ReLU** (ReLU's gradient is exactly 0 for all `x<0`, so a
unit pushed negative never recovers); and it's **self-gated** — `x·sigmoid(x)` reads
as *the unit gates the identity by its own value*.

DreamerV1 used **ELU**; **SiLU + LayerNorm arrived with DreamerV3's architecture
refresh**, among the changes that let one hyperparameter set span 150+ tasks.

### Q30 — Does the reward model's gradient affect `h` and `z`?

**A:** **Mechanically yes** — both paths, no `stop_gradient` anywhere in the chain
(`agent.py:1718` → `train.py:721` → `loss.py:551`):

```
reward_loss
   └─► reward_model weights
   └─► latent [64,16,1280]
          ├─ latent[...,1024:] = h_t  ──► gru_cell ──► BPTT into t−1, t−2, …
          └─ latent[...,:1024] = z_t  ──STE──► repr_* ──► h_t, and ──► φ(o_t) ──► encoder
```

**But it does far less work than the 1:1 weights suggest**, for two reasons:

**1. Dimensionality — an asymmetry hidden from every β.**

```
observation_loss = Σ over D_obs dims of squared error    ← D_obs terms
reward_loss      = one 255-bin cross-entropy              ← 1 term
continue_loss    = one Bernoulli NLL                      ← 1 term
```

All carry coefficient 1.0, so the effective ratio is **`D_obs : 1 : 1`**, not
`1:1:1`. Reconstruction outweighs reward by roughly the observation width, purely
from dimensionality. `β_pred = 1` hides it.

**2. The ablations say to remove it.**

> **DreamerV2:** stopping reward gradients into the encoder **helps slightly** —
> better on 15 tasks. **DreamerV3 Nature:** stopping **task-specific** (reward +
> value) gradients hurts **less** than stopping **task-agnostic** (reconstruction)
> gradients.

Against: removing *image* gradients → **0.04 gamer median**. Stark ordering. Their
reading: representations specialised to *past* rewards generalise worse; the world
model should model the **world**, not the task.

**Project relevance** (flagged in [[dreamer_lit_review]] Paper 4): pain reward has
exactly the punished profile — **sparse, high-magnitude, possibly mis-scaled**. A
safer recipe keeps the encoder reconstruction-driven and lets pain enter only via
the reward/value heads. **`dreamer_srl` does not stop reward gradients** (it follows
sheeprl's default); the ablation would be a `stop_gradient` on `latent` in the
reward head's call — **not a config knob today**.

## Session 2026-07-16 (cont.) — behaviour-learning setup, batch shape, replay, framework fit

### Q31 — Behaviour learning: the setup (partial — thread detoured, actor/critic still open)

**Q:** Let's move on to the actor and critic.

**A:** *(Setup only — the conversation detoured into Q32-Q35 before λ-returns / the
actor objective / the two-term critic were covered. **These remain unexplained.**)*

**What changes:** behaviour learning runs on **imagined** data, and the world model
is now **frozen** (`train.py:820-821` stop-gradient the posteriors and recurrent
states — it has already taken its gradient step).

**The leverage:** every posterior becomes a starting point.

```
posteriors [64,16,...] = 1024 real states  →  flattened to 1024 independent starts
                                           →  1024 parallel dreams × 15 steps
```

Not "imagine from the last state" — **every one of the 64×16 latents is branched from,
in parallel.** DreamerV2 reports **468B imagined states per 200M real steps** (10⁴×).
That ratio is the economic argument for world models.

**Three modules**, all reading the same 1280-wide latent the decoder read (they cannot
tell a dreamed latent from a real one — which is what `L_dyn` bought):

| module | maps | note |
|---|---|---|
| `actor` | 1280 → A logits | + unimix |
| `critic` | 1280 → 255 | two-hot. **Output linear zero-init** |
| `target_critic` | 1280 → 255 | same arch, separate weights, Polyak EMA |

**The rollout** reuses the world model's own `gru_cell` + `transition_*`, identical
weights. What's **absent** is the point:

| module | in imagination? | why |
|---|---|---|
| `encoder` | ❌ | no `o_t` to encode |
| `repr_*` (posterior) | ❌ | nothing to condition on — **where the prior earns its keep** |
| `decoder` | ❌ | observations never materialised |
| `gru_cell`, `transition_*` | ✅ | identical weights and code path |
| `reward_model`, `continue_model`, `critic` | ✅ | read off imagined latents |

Output: `r̂`, `v̂`, `ĉ` over `[16, 1024, 1]`. Everything downstream (λ-returns, the
two-term critic, Moments, the actor objective) is *what you do with those three arrays*.

**One closed loop:** our actor uses **pure REINFORCE**, no dynamics backprop
(`train.py:905` stop-gradients the imagined latents) — DreamerV2's `ρ=1` for discrete
actions. So the two halves make **opposite** bias/variance choices:

| | estimator | bias | variance |
|---|---|---|---|
| **world model** (through the latent sample) | straight-through | biased ❌ | low ✅ |
| **actor** (through the action sample) | REINFORCE | unbiased ✅ | high ❌ |

Because the actor's gradient would traverse 15 steps of imagined dynamics, where ST's
bias compounds.

### Q32 — Why 64 × 16? Where do the numbers come from?

**A:** Literally from `configs/models/dreamer_srl/01_food_only.yaml`
(`per_rank_sequence_length: 64`, `per_rank_batch_size: 16`), copied from sheeprl,
copied from Hafner's *"batch shape B=16, L=64."* **Tuned constants, no derivation.**
But the two are different *kinds* of number.

**`B = 16` — the ordinary one.** How many independent replayed sequences at once.
Standard batch trade. **Parallel** — vectorised, nearly free on GPU.

**`T = 64` — the one that carries meaning.** It is the **BPTT horizon**: *the longest
dependency the world model can possibly learn.* Every chunk **starts from scratch** —
`get_initial_states` (agent.py:1675) plus §S1 forcing `is_first[0]=1` (train.py:690)
trips the §S4 reset on the first tick, so `h` is **not** carried in from wherever the
chunk was sampled. A 100-step dependency is physically unreachable by the gradient.

*(At **inference** the `Player` carries `h` across the whole episode unbroken — so
this is a **learning** limit, not a runtime one. The GRU must generalise its
64-step-learned dynamics to arbitrarily long rollouts.)*

**They multiply:** `T × B = 1024` = the number of imagination starting points.

**They cost differently:** the `T` loop is **sequential** (`h_t` needs `h_{t-1}` — 64
GRU steps that cannot overlap); the `B` axis is **parallel**. So `T` is the wall-clock
bottleneck and `B` the memory bottleneck. Doubling `B` costs memory; doubling `T` costs
**time**, unrecoverable with a bigger GPU.

**It shrank at V3:**

| | B | T | states/batch |
|---|---|---|---|
| DreamerV1 | 50 | 50 | 2500 |
| DreamerV2 | 50 | 50 | 2500 |
| **DreamerV3** | **16** | **64** | **1024** |

More than halved the batch while **lengthening** the sequence — trading gradient-noise
reduction for a longer learnable horizon. A deliberate reallocation.

**This project already sweeps it**: `01_food_only_seqlen{32,128}.yaml` (+ M-size
variants). Given what `T` is, that sweep asks *"how far back does this grid world's
structure reach?"* — not "what's a good batch size."

### Q33 — Confirming: 64 = GRU horizon, 16 = batch, 15 = imagination horizon?

**A:** Correct on all three, with three refinements.

**1. "64 real trajectories" → 64 consecutive *timesteps*, possibly spanning several
episodes.** V3 uses subsequence replay: a 64-step window dropped **anywhere** in the
buffer, which can **straddle an episode boundary**. That is why `is_first` is a full
`[64,16,1]` tensor — a flag at *every* step — and why §S4 lives **inside** the tick
loop (agent.py:1043), not at initialisation.

```
chunk:      t=0 ..................... t=37 .................... t=63
is_first:    1   0  0  0  ...  0   0    1    0  0  0  ...  0   0
             ▲                          ▲
        §S1 forces this           a real episode boundary
```

**2. 64 is the MAXIMUM horizon, not the actual one.** When `is_first=1`,
`h_{t-1} := (1−1)·h_{t-1} + 1·h₀` — multiplied by **zero**, so the gradient through
that path is zero. **The reset severs BPTT at the boundary** (correctly — memory must
not leak across episodes). So:

```
effective horizon = min( 64 , steps since the last is_first = 1 )
```

Mostly benign here (episodes ~500 steps at parity), but in a short-episode regime the
effective horizon collapses regardless of the config.

**3. 15 imagined steps, but 16 latents.** `for i in range(1, horizon+1)` → i = 1…15.
Index **0 is the starting latent — the POSTERIOR from real data, not dreamed**;
indices 1–15 are imagined from the prior. Hence `H+1` throughout `train.py` and the
`[:-1]` / `[1:]` slicing asymmetry (λ-returns have 15 entries; latents have 16).

**Corrected arithmetic** (an earlier "1024 × 15 = 16,384" was wrong):

```
16 × 1024 = 16,384   latents total
15 × 1024 = 15,360   actually imagined
     1024            real posteriors (the starts)
```

**The full number map:**

| number | symbol | what |
|---|---|---|
| 64 | `T` | consecutive timesteps/chunk; **max** BPTT horizon |
| 16 | `B` | independent chunks/batch |
| **1024** | `T×B` | **imagination starting points** |
| 15 | `H` | imagined steps per start |
| 16 | `H+1` | latents per dream (index 0 = real posterior) |
| 15,360 | `H×T×B` | imagined states per gradient step |

The two horizons are **independent**: the world model learns dynamics over **64** steps
of real data; the actor plans over **15** steps of dreamed data. Nothing forces a match.

### Q34 — Why does Dreamer allow subsequence replay instead of sampling within an episode?

**A:** It's not "allowed" — it's a **change V3 made**. V1/V2 waited for episode
completion; V3 dropped that, stated reason: **shortening the feedback loop**.

**Why waiting is bad.** With `replay_ratio=1` and ~500-step episodes, the freshest
trainable data is **500 steps stale** — every gradient step chases a policy the agent
has already left. And at scale it becomes impossible: Minecraft Diamond is a 60-minute
episode (~72,000 steps); you cannot build a buffer out of units that large.

**Why it's SAFE — the reset is complete.** Exactly **three** things cross a tick
boundary (`z_{t-1}`, `a_{t-1}`, `h_{t-1}`) and **all three** are reset
(agent.py:1043-1071). That's 100% of the cross-tick channel; each is multiplied by
zero when `is_first=1`.

> **A 64-step chunk containing a boundary is semantically identical to two independent
> chunks.** They merely share a tensor.

So §S4's completeness is what **licenses** subsequence replay. Reset only `h` and
forget `z`, and the model learns that the last frame of one life predicts the first of
the next. Not hypothetical here: **"missing reset_data write" and "terminated/truncated
conflation" were two of the five P-blockers** behind the v1 failure at ep_len 103.8.

**A bonus that matters for this project:**

| scheme | uniform over | consequence |
|---|---|---|
| sample an **episode**, then a window | **episodes** | a 20-step episode's transitions sampled as often as a 500-step episode's |
| sample a **start index** over the buffer | **transitions** | every transition equally likely |

Subsequence replay is **transition-uniform**. For a survival task — where episode
length **is** the metric — episode-uniform sampling would over-weight the early,
short, death-heavy episodes precisely *because* there are more of them, biasing the
world model toward the failure regime it is trying to escape.

**And post-boundary steps aren't wasted:** at inference, an agent at step 4 genuinely
*has* only 4 steps of context. Those short-context segments are the only thing teaching
the **early-episode regime**. Strict within-episode-full-length sampling would never
train it.

### Q35 — Does subsequence replay break our experiment framework? (basic04, random init, probes)

**Q:** Basic motivation: put the agent in a **simpler** eval situation than the complex
training env, to see behaviour isolated, starting from different init conditions
(nutrition, injury). Level 04 was made for random init and rPPO shows proper behaviour.
Does this work with DreamerV3?

**A:** **No structural incompatibility — but the Dreamer eval path is a separate,
new driver, and there is no Dreamer checkpoint worth probing yet.**

**Naming, first — there are two basic04s.** The `basic/` ladder was re-levelled
2026-07-04 (8 → 6 levels; `archive/basic_releveled_20260704/README.md`):

| | old | current |
|---|---|---|
| basic04 | far-sight predator, **NO** random init | **jump attack**, HAS random init |

Current `04-jump_attack_10x10.yaml` has `extends: environment/experiment/basic/03-random_init_10x10`,
so it **inherits** random init. Random init is now level **03**, whose header notes
*"was random-init only; variant-04 merged 2026-07-02."* The keys:

```yaml
body:
  random_start_nutrition: true
  start_nutrition_low: 0      # can start STARVING
  start_nutrition_high: 100
  random_start_injury: true
  start_injury_low: 0
  start_injury_high: 100      # can start NEARLY DEAD
```

`basic04_farsight` still appears in older docs; live sweeps (`rppo_basic04_jump_n109`,
`dsrl_basic04_size_*`) are on the jump level.

**The probe framework** (`configs/environment/experiment/behavior_probes/`, `core/`
validated vs `explore/` one-off): `core/forage_nutrition/` (departure-delay satiety,
`forage_nutr{000..100}`), `core/avoidance/` (12 configs: animal {pred,rabbit,none} ×
injury {0,70} × smell ablations), `core/forage_direction/`. They `extends:
environment/default` — **not** the training env, exactly as described. Init is pinned
per-config via a **degenerate range** through the random path:

```yaml
random_start_injury: true
start_injury_low: 0
start_injury_high: 0        # low == high pins the exact value
```

(There is no `start_satiation_low/high` — satiation is derived from nutrition
`S = MaxS·(N/MaxN)^scaling`, so probes vary nutrition.)

**⚠️ The gap: `scripts/eval/eval_rollout.py` HARD-FAILS on Dreamer** —
`raise NotImplementedError(f"Agent type '{agent_type}' not yet supported…")`.
**`scripts/eval/dreamer_srl_probe_eval.py`** fills it, and the two **converge**: same
probe YAMLs, same `.rec.gz` format, same `avoidance_stats_heatmap.py`. Two drivers, one
pipeline.

**Three things favour Dreamer here:**

1. **Subsequence replay rehearses the probe condition.** §S1 forces `is_first=1` on
   every chunk start → `h:=h₀`, `z:=z₀`, `a:=0` — *"you know nothing; here's an
   observation; figure out where you are."* **That is a probe**, run 1024× per gradient
   step. rPPO has no such mechanism; Dreamer's replay scheme is *better* matched to the
   probe design than rPPO's.
2. **Random init is coverage.** A world model wants the whole (nutrition × injury)
   space, not one corner — more valuable to Dreamer than to a policy.
3. **The pin-by-degenerate-range idiom already works for both.**

**Three caveats, descending:**

1. **The Dreamer probe driver has two commits ever** (`6d1d895` create, `69e0399` fix:
   topology-agnostic CPU restore). Documents a live Orbax *"Topology mismatch"* trap
   restoring GPU-saved checkpoints on CPU, and GPU OOM at ~18 concurrent processes
   (hence `--device cpu` default).
2. **`true_obs` is NOT recorded for dreamer_srl** (`eval.py:122` — "no noise API").
   Harmless for `core/avoidance` / `core/forage_nutrition` (noise-free; the "smell
   ablations" remove a channel, they don't add noise) — **but it forecloses noise
   probes**, and `basic/05-sensory_noise_10x10` plus the whole `olfactory_ambiguity`
   family exist.
3. **Horizon × episode-length.** Dreamer's episodes are currently **5–68 steps**;
   `T = 64`. Every chunk is boundary-dense, so the 64 isn't being used — paying 64
   *sequential* GRU steps to learn ~30-step dependencies. Self-correcting as survival
   improves; `seqlen32` (config exists) would be strictly cheaper meanwhile.

**The empirical state (diary 2026-07-15, 16:43):**

| | trained rPPO | Dreamer |
|---|---|---|
| checkpoint | `rppo_basic04_jump_128env_100M` @ **100M** | size-S @ **66k** |
| behaviour | flees to bush & heals; cover 9/10 eps; injury 50→0; 4× full-500 survival | **dies in 5–68 steps**; injury→100; ~2% bush |

**1,500× training gap — not a comparison.** The diary explicitly verified the near-zero
Dreamer bush-dwell is *"genuine not-learned-yet, not an eval bug."* So the framework
isn't the blocker; the absence of a trained Dreamer checkpoint is.

## Session 2026-07-20 — Dreamer 4

> Logging was paused after Q35 at the user's request, then resumed here.

### Q36 — How does Dreamer 4 use a transformer, and how does it differ from DreamerV3?

**A:** **The headline swap:**

```
DreamerV3:   RSSM              = GRU (256) + 32×32 categorical latent    ~12M–400M params
Dreamer 4:   tokenizer (400M) +  dynamics transformer (1.6B)             = 2B params
```

The RSSM had been stable since PlaNet (2019) — four papers, one backbone. Dreamer 4
discards it.

**The two new pieces.**

1. **Causal tokenizer** (replaces encoder/decoder): block-causal transformer
   encoder/decoder with a **tanh-projected continuous latent bottleneck**, trained as a
   *masked autoencoder* — MSE + 0.2·LPIPS, patch dropout `p ~ U(0, 0.9)`.
2. **Dynamics transformer** (replaces GRU + prior net): **2D block-causal** transformer
   over an interleaved sequence of `[latents, actions, shortcut-noise control tokens]`.
   Three speed tricks: **factored attention** (separate space-only and time-only layers
   instead of full 3D), **sparse temporal attention** (temporal only every 4th layer),
   **GQA** (shared KV heads → smaller cache). Plus QK-Norm, SwiGLU, RoPE, pre-RMSNorm,
   attention-logit soft capping. → **21 FPS on a single H100** with 9.6 s context, fast
   enough for a human to play inside.

**Shortcut forcing — the real algorithmic change.** V3 predicted the next latent in
**one GRU step**; Dreamer 4 **denoises** it over `K = 4` forward passes per frame. It
fuses *diffusion forcing* (per-timestep noise levels in a sequence) with *shortcut
models* (network conditioned on step size `d`, so inference takes few large steps).
Corrupt as `z̃_t = (1−τ_t)z⁰_t + τ_t z¹_t`; at `d_min` the loss is squared error to the
clean latent (flow matching), at larger `d` it is a **bootstrap** against the average of
two half-step outputs computed in v-space and rescaled to x-space.

**x-prediction, not v-prediction**: velocity is high-frequency, so iterating it
autoregressively compounds into drift; the clean latent is low-frequency and stays
stable over long rollouts. Equivalent at a given `τ`
(`‖x̂₁−x₁‖² = (1−τ)²‖v̂_τ−v_τ‖²`) but the gradients differ — x-prediction doesn't blow
up as `τ → 0`.

**What DISAPPEARS — i.e. everything in Q3–Q10 of this log:**

| DreamerV3 | Dreamer 4 |
|---|---|
| prior `p(z_t\|h_t)` vs posterior `q(z_t\|h_t,o_t)` | **gone** — no such split |
| `L_dyn`, `L_rep`, KL balancing, free bits | **gone** — no KL anywhere |
| aggregate posterior / prior hole | **gone** — dissolved by the objective |
| straight-through gradients | **gone** — the latent is continuous |
| 32×32 categorical | **continuous, tanh-projected** |

The tokenizer is a *masked autoencoder*, not a VAE — no KL. The dynamics is trained by a
denoising objective, already a generative model over latent space, so there is no "make
the prior match the aggregate posterior" problem to solve.

**But multimodality still has to live somewhere** — it moved from the **latent** to the
**dynamics** (cf. Q12). A diffusion model is natively multimodal; once the dynamics can
represent "either A or B" itself, the latent can return to being a smooth,
easy-to-predict continuous code.

**What SURVIVES from V3:** symexp twohot reward head; distributional critic with symexp
twohot on λ-returns, `γ = 0.997`; and the core paradigm — train the agent inside the
world model.

**Training: online → offline, one phase → three.**

| phase | trains | how |
|---|---|---|
| **1. Pretrain** | tokenizer + dynamics | 2.5K hours VPT contractor video |
| **2. Finetune** | task tokens inserted into the *same* transformer | BC + reward via multi-token-prediction heads (L=8) |
| **3. Imagine** | policy + value heads only, **transformer frozen** | RL purely on imagined rollouts |

Phase 2's **token insertion**: agent tokens attend to everything, but *nothing attends to
agent tokens*. That one-way mask prevents task conditioning from contaminating the
dynamics — frames must still be predicted from prior frames and actions alone. A 50/50
batch mix of BC-loss and dynamics-loss sequences keeps the world model honest.

**The actor changes too — PMPO.** V3 scales returns by their 5–95th percentile range and
does REINFORCE. Dreamer 4 **discards advantage magnitudes entirely**: positive-advantage
states pushed up, negative pushed down, equally, plus a reverse-KL to the BC prior
(`α=0.5`, `β=0.3`). Reasoning: V3's percentile scaling assumes imagined return
*magnitudes* are roughly trustworthy; inside a 2B-param video model they aren't, but the
*signs* still correlate with real reward. PMPO assumes only the sign. The behavioural-prior
KL stops the policy drifting where the world model hallucinates — the "cheat the world
model" problem from Ha & Schmidhuber, patched with a KL instead of a temperature knob.

**Cost and payoff.** 2B params, 256–1024 TPU-v5p to train (inference is single-GPU).
Bought: **first agent to obtain Minecraft diamonds with zero environment interaction** —
0.7% of 60-minute episodes, purely offline. VPT-finetuned, plain BC, and a Gemma-3 VLA
baseline all score **0%**; iron pickaxe 29% vs VLA's 11%. Honest limits: 0.7% is far
below VPT-*online*'s ~20% (the price of pure offline), and inventory items go incoherent
over long rollouts because the 9.6 s context bounds consistency.

**⚠️ The one regression:** `h_t` in an RSSM is a recurrent state summarising — in
principle — *unbounded* history. A transformer attends over a **fixed window** (192
frames, 9.6 s). Dreamer 4 traded a formally-unbounded (though practically 64-step-trained,
cf. Q32) memory for a large-but-hard-capped one. Big win for 9.6 s of Minecraft; a
ceiling the RSSM did not have for genuine long-horizon recall.

### Q37 — In the first World Models paper, why an evolutionary algorithm for the controller?

**A:** *(Clarification first: the EA does not select actions. CMA-ES optimises the
controller's **weights**; the controller then selects actions by a plain linear map
`a_t = W_c[z_t ; h_t] + b_c`. What is evolved is 867 numbers, offline — not per-step
decisions.)*

**Because the controller is deliberately tiny — that's the paper's thesis.**

| component | job | params | trained by |
|---|---|---|---|
| **V** (VAE) | compress frames | large | backprop |
| **M** (MDN-RNN) | predict next latent | large | backprop |
| **C** (linear) | pick actions | **867** | **CMA-ES** |

Put the parameter mass where the learning signal is *dense and cheap*; keep the part
with the *hard* signal small enough that a crude optimiser suffices. V and M have dense,
immediate, supervised targets (reconstruct this frame, predict the next latent) — every
timestep gives full gradient information. C has **one scalar per episode**, arriving
hundreds of steps after the actions that earned it. The move is to **shrink the search
space** until the hard problem is tractable, rather than invent better credit assignment.

Per the review: *backpropagation handles the high-dimensional but easy problem; evolution
handles the low-dimensional but hard problem.*

**Why CMA-ES specifically** — four reasons, all of which fail if C were large:

1. **Only needs a scalar.** `F(θ) = E[Σ r_t]` — no gradient, no differentiability, no
   per-step reward decomposition. Sparse/noisy/delayed rewards all fine.
2. **The environment is a black box.** CarRacing fitness is the **true game engine's**
   return; you cannot backprop through a game engine.
3. **Covariance estimation is only tractable at small `n`.** CMA-ES maintains a full
   multivariate Gaussian `θ ~ N(m_g, Σ_g)` — an `n×n` covariance. Fine at `n≈10³`,
   hopeless at `n≈10⁶`. The 867-param controller is *chosen* to sit in that regime.
4. **Embarrassingly parallel.** Population λ=64 × 16 rollouts each = 1024 independent
   episodes per generation, one per CPU core, no gradient sync. CarRacing took ~1800
   generations.

**Why NOT backprop through M?** M *is* differentiable in principle. They declined
because: M is an **MDN-RNN** that samples a mixture component — a discrete
non-differentiable choice (the staircase of Q15, before straight-through was standard);
gradients through hundreds of stochastic rollout steps were known to be unstable
("chaotic gradients"); and dream-training deliberately **injects noise** via temperature
`τ` to stop C exploiting M's flaws — backpropagating through noise you added on purpose
to make the model less exploitable works against yourself.

**This question sits on the lineage's fault line:**

| | how behaviour is optimised | gradients through the world model? |
|---|---|---|
| **World Models (2018)** | CMA-ES on 867 params | ❌ |
| **PlaNet (2019)** | CEM planning, re-run every step | ❌ |
| **DreamerV1 (2020)** | actor-critic, **reparameterised gradients through learned dynamics** | ✅ |

DreamerV1's headline contribution is exactly the rebuttal — *if you've built a
differentiable simulator inside your network, why use derivative-free search?* It paid
off (8× faster wall-clock than PlaNet, and it unlocked long-credit-assignment tasks CEM
could not solve).

**But DreamerV2 partly walked it back:** on Atari it uses **pure REINFORCE** (`ρ=1`), a
score-function estimator — closer in spirit to evolution than to backprop-through-dynamics.
That is the setting `dreamer_srl` runs. So the arc is not "evolution wrong, gradients
right"; it is that **the right estimator depends on how much you trust the path from
action to return** — and with discrete actions through 15 steps of imagined dynamics,
Hafner also concluded you should not differentiate through it. (cf. Q31's bias/variance
table.)
