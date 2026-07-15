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
