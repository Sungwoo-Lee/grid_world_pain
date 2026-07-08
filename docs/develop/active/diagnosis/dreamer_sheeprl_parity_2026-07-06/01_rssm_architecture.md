---
title: "Dreamer↔sheeprl parity audit (area 1/5) — RSSM + world-model architecture"
topic: diagnosis
status: active
created: 2026-07-06
last_updated: 2026-07-06
---

# Parity audit 1/5 — RSSM + world-model architecture

## Purpose (plain-language entry point)

The project rebuilt the DreamerV3 reinforcement-learning agent in JAX (the
`dreamer_srl` package) by porting a community PyTorch implementation called
**sheeprl** (vendored under `vendor/sheeprl/`, commit `33b6366`). Because the
two frameworks differ, the code cannot be textually identical — the question
is whether it is **numerically and algorithmically the same model**. Any
unnoticed difference (a wrong layer, a different weight initialization, a
mis-placed normalization) can silently distort training.

This document is a from-scratch, line-by-line re-comparison of the **network
architecture**: the recurrent world model (RSSM), the observation encoder and
decoder, the reward and episode-continuation heads, weight initialization, and
the sampling machinery for the discrete latent state. It deliberately does
NOT trust the previous audit — every claim was re-derived from both sources,
and cheap empirical checks were run (the existing bit-identity test suite:
9/9 pass). Verdict: the architecture is a faithful port, with **two
undeclared deviations found** — a wrong discount factor in every training
config, and a different initialization distribution on two RSSM output
layers — plus a handful of cosmetic differences. Details below.

**Sources compared** (all line numbers pinned to these files):

- OURS: [`src/algorithms/dreamer_srl/agent.py`](../../../../../src/algorithms/dreamer_srl/agent.py) (2137 lines), with cross-checks into `train.py`, `utils.py`, and `configs/models/dreamer_srl/*.yaml`
- REF: [`vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py`](../../../../../vendor/sheeprl/sheeprl/algos/dreamer_v3/agent.py) (1236 lines), `sheeprl/models/models.py`, `sheeprl/algos/dreamer_v3/utils.py`, `sheeprl/algos/dreamer_v2/utils.py`, `sheeprl/configs/algo/dreamer_v3{,_XS,_M}.yaml`

**Classification legend**

| Tag | Meaning |
|---|---|
| PARITY | Same math, same structure, line-for-line equivalent |
| EQUIV | Equivalent-by-design — JAX-necessitated difference, provably same math (proof given) |
| DECLARED | Already logged in [DEVIATION_LOG.md](../../dreamer_srl_v1/DEVIATION_LOG.md) (D-NNN) or the Known Bugs registry |
| UNDECL-I | Undeclared, with training impact (severity given) |
| UNDECL-C | Undeclared, cosmetic (no numerical/gradient effect) |

---

## Complete comparison table

Every checked item, including PARITY rows — this is a reference record.

| # | Item | Ours (file:line) | Sheeprl (file:line) | Verdict |
|---|---|---|---|---|
| 1 | GRU fused projection `[hx;x] → 3H`, single Linear, `bias=False` in RSSM wire-up | agent.py:105-110, 612 | models.py:365, agent.py:321 | PARITY |
| 2 | GRU LayerNorm after projection, `eps=1e-3` | agent.py:118-122, 611 | models.py:368, agent.py:305-306 (`layer_norm_kw={"eps":1e-3}`) | PARITY |
| 3 | GRU concat order: `hx` first, then input | agent.py:143 | models.py:396 | PARITY |
| 4 | GRU gate chunk order `(reset, cand, update)` | agent.py:151-153 | models.py:399 | PARITY |
| 5 | GRU reset applied **inside** tanh: `cand = tanh(reset * cand_proj)` | agent.py:157 | models.py:401 | PARITY |
| 6 | GRU update-gate bias shift: `sigmoid(update_proj − 1)` | agent.py:159 | models.py:402 | PARITY |
| 7 | GRU state blend `update*cand + (1−update)*hx` | agent.py:162 | models.py:403 | PARITY |
| 8 | Recurrent pre-projection MLP: `Linear(S·D+A → dense_units, bias=False) + LN(1e-3) + SiLU` before the GRU | agent.py:580-590, 1081-1087 | agent.py:309-317, 339-341 | PARITY |
| 9 | Transition model (prior): one hidden layer `Linear(bias=False)+LN+SiLU` then output `Linear(hidden → S·D, bias=True)` | agent.py:626-640, 751-755 | agent.py:1036-1051, models.py miniblock | PARITY |
| 10 | Representation model (posterior): same shape, input `concat(h, embedded_obs)` | agent.py:648-663, 799-806 | agent.py:1017-1035, 463 | PARITY |
| 11 | Unimix 1% on **both** prior and posterior logits, applied in probability space then converted back via `log(probs)` | agent.py:819-856 (RSSM), 1472-1482 (Actor) | agent.py:437-449, 839-845 | PARITY |
| 12 | `probs_to_logits` clamp: torch clamps probs to `[eps≈1.19e-7, 1−eps]`; ours plain `jnp.log(probs)` | agent.py:852, 1481 | torch.distributions.utils (verified source) | EQUIV (proof §D1) |
| 13 | Stochastic state 32 categoricals × 32 classes, `Independent(OneHotCategoricalStraightThrough).rsample()` vs Gumbel-max + straight-through | agent.py:863-925 | dreamer_v2/utils.py:44-61 | EQUIV / DECLARED D-009 (proof §D2) |
| 14 | Mode path (`sample=False`): `one_hot(argmax(logits))`, no PRNG, no straight-through | agent.py:901-905 | torch `dist.mode` | PARITY |
| 15 | Learnable initial recurrent state: zeros-init `nn.Parameter`, used as `tanh(param)`, initial posterior = transition **mode** (deterministic) | agent.py:673-675, 931-969 | agent.py:382-394 | PARITY |
| 16 | `dynamic()` §S4 three-quantity reset: action zeroed, recurrent + posterior arithmetic-masked; posterior reshaped **before** mask | agent.py:1037-1071 | agent.py:425-430 | PARITY |
| 17 | `dynamic()` order: reset → recurrent model → transition → representation; returns 5-tuple | agent.py:1073-1107 | agent.py:432-435 | PARITY |
| 18 | Sequence-scan carry-in: ours seeds with `get_initial_states`; sheeprl seeds with **zeros** | agent.py:1671 | dreamer_v3.py:107-131 | EQUIV (proof §D3) |
| 19 | `is_first[0]` forced to 1 before the world-model unroll (§S1) | train.py:686-687 | dreamer_v3.py:100 | PARITY |
| 20 | Encoder: symlog on inputs, `mlp_layers` × (`Linear(bias=False)+LN(1e-3)+SiLU`), output dim = `dense_units` | agent.py:1143-1195 | agent.py:100-151 | PARITY |
| 21 | Decoder: same MLP body + plain `Linear(dense_units → obs_dim)` head | agent.py:1234-1290 | agent.py:229-278 | PARITY |
| 22 | Decoder sizes: `build_agent` reuses **encoder** `dense_units`/`mlp_layers`; the config's `world_model.decoder.*` keys are ignored | agent.py:2052-2058 vs configs `decoder:` block | agent.py:1083-1093 (`observation_model.*`) | UNDECL-C (§D4) |
| 23 | Reward head: MLP + `Linear(→255 bins)`, output kernel **and** bias zero-init (cascade fix #27) | agent.py:2064-2071, 1899-1906 | agent.py:1100-1112, 1175 | PARITY |
| 24 | Continue head: MLP + `Linear(→1)`, output uniform-init scale 1.0, bias 0 | agent.py:2074-2081, 1907-1910 | agent.py:1114-1127, 1176 | PARITY |
| 25 | Critic: MLP + `Linear(→255 bins)`, zero-init output | agent.py:2104-2111 | agent.py:1154-1166, 1172 | PARITY |
| 26 | Target critic at t=0: sheeprl `deepcopy(critic)`; ours a **fresh random** network, relying on first Polyak call with `tau=1.0` | agent.py:2114-2121; train.py:675 | agent.py:1217 | EQUIV-conditional (§D5) |
| 27 | Actor body/head: `mlp_layers`×(`Linear+LN+SiLU`) + `Linear(→action_dim)`; unimix before the distribution | agent.py:1444-1509 | agent.py:761-845 | PARITY |
| 28 | Actor log-prob: `Σ onehot · log_softmax(logits)` | agent.py:1552-1555; train.py:905-910 | torch `OneHotCategorical.log_prob` | PARITY |
| 29 | Actor entropy: ours `−Σ p·log(p + 1e-8)`; torch `−Σ p·log_softmax(logits)` (with `min_real` clamp) | agent.py:1557-1559; train.py:912-913 | torch Categorical.entropy (verified source) | UNDECL-C (§D6) |
| 30 | Body weight init: truncated normal, fan-avg scale, Hafner constant `0.87962566103423978`, ±2σ truncation, bias 0 | utils.py:53-88 | dreamer_v3/utils.py:143-166 | PARITY (RNG stream: DECLARED D-002) |
| 31 | Output-head uniform init `±sqrt(3·scale/den)`, bias 0 | utils.py:91-117 | dreamer_v3/utils.py:170-186 | PARITY (D-002) |
| 32 | RSSM transition/representation **output-layer** init: sheeprl overwrites with `uniform_init_weights(1.0)`; ours keeps truncated-normal | agent.py:703-706, 711-714 | agent.py:1173-1174 | **UNDECL-I Low** (§D7) |
| 33 | LayerNorm affine init (scale 1, bias 0) everywhere | flax nnx defaults | dreamer_v3/utils.py:164-166 + torch defaults | PARITY |
| 34 | Sheeprl custom `LayerNorm` subclass only re-casts output dtype (mixed-precision aid) | n/a | models.py:521-525 | PARITY at float32 policy |
| 35 | Activations: SiLU everywhere (encoder, decoder, RSSM MLPs, heads, actor) | agent.py throughout (`jax.nn.silu`) | dreamer_v3.yaml `dense_act: torch.nn.SiLU` | PARITY |
| 36 | dtype policy: float32 end-to-end, no mixed precision | JAX default | fabric precision 32 default | PARITY |
| 37 | Imagination rollout: prior-only transition per step, H+1 latents incl. init, actor action per step | agent.py:1727-1827 | dreamer_v3.py:202-241, agent.py:482-498 | PARITY |
| 38 | Imagination actor input not per-step `detach()`ed (sheeprl detaches at L219/L240) | agent.py:1774, 1812 | dreamer_v3.py:219, 240 | EQUIV (proof §D8) |
| 39 | `imagine()` returns per-step log-probs/entropies that the training step **discards** (loss recomputes both from stop-gradient'd latents) | agent.py:1752-1826; train.py:826, 891-917 | n/a (sheeprl computes only at loss time) | UNDECL-C (§D9) |
| 40 | `imagine()` re-implements the recurrent forward inline instead of sharing one method with `dynamic()` | agent.py:1794-1799 vs 1081-1087 | agent.py:328-341 single method | UNDECL-C (drift risk) |
| 41 | PRNG threading: fresh split per sequence step; separate keys for prior sample, posterior sample, actor sample; no key reuse found | agent.py:1093, 1102, 1675, 1786 | torch global RNG | EQUIV (DECLARED D-009 class) |
| 42 | Float32 accumulation-order drift through GRU / RSSM MLP chains | measured 2.97e-4 / 7.19e-4 | — | DECLARED D-007, D-008 |
| 43 | Size presets: XS = 256 units / 1 layer / 256 recurrent / 256 hidden; M = 640/3/1024; stochastic 32×32; bins 255 | configs/models/dreamer_srl/01_food_only{,_M}.yaml | dreamer_v3_XS.yaml, dreamer_v3_M.yaml | PARITY |
| 44 | **Discount factor γ**: ours `0.996840347` in every dreamer_srl config; sheeprl `0.996996996996997` | all `configs/models/dreamer_srl/01_food_only*.yaml` (e.g. 01_food_only.yaml:41) | dreamer_v3.yaml (`gamma`) | **UNDECL-I Med** (§D10) |
| 45 | Gradient clipping: sheeprl clips WM/actor/critic at 1000/100/100; ours has none | configs (no `clip_gradients` key); train.py | dreamer_v3.yaml `clip_gradients`, dreamer_v3.py:196-200 | DECLARED-KNOWN (open Known-Bugs row; §D11) |
| 46 | `learnable_initial_recurrent_state` flag: ours hardcodes learnable (config default True) | agent.py:673 | agent.py:382-389, yaml | PARITY (flag not surfaced — cosmetic) |
| 47 | DecoupledRSSM variant: not ported; sheeprl default `decoupled_rssm: False` | n/a | agent.py:501-593 | PARITY (N/A) |
| 48 | Greedy/mode action path: sheeprl `Actor.forward(greedy=True)` used at eval; our `Actor` has no mode path in agent.py | agent.py:1511-1561 | agent.py:833-836 | NOT VERIFIED here (eval-path area; see §Verdict) |
| 49 | `PlayerDV3` (env-side stateful player): no analog class; inlined in the driver | dreamer_srl_main.py | agent.py:596-691 | out of scope (area 5 — driver) |
| 50 | Vestigial classes: `RewardHead`/`CriticHead` (single-linear) unused by `build_agent`; return type hints say `CriticHead` but `FullMLPHead` is returned | agent.py:220-363, 1939 | n/a | UNDECL-C |
| 51 | Lazy `symlog` import inside `MLPEncoder.__call__` | agent.py:1189 | n/a | UNDECL-C (nit; prior "lazy import" incident class) |
| 52 | Modulation hooks (project extension): **none present** in agent.py — baseline network path is clean | grep verified | n/a | PARITY (clean) |

Empirical spot-check: `tests/algorithms/dreamer_srl/test_agent.py` — **9/9 pass**
(2026-07-08, main conda env), covering GRU cell (D-007 threshold 5e-4), transition /
representation / initial-states / is_first resets (D-008 threshold 2e-3), action-shift,
and zero-init head discipline against pregenerated PyTorch fixtures.

---

## Detail sections (non-PARITY rows)

### D1 — `probs_to_logits` clamp is provably inactive post-unimix (row 12, EQUIV)

Torch's `probs_to_logits(probs)` = `log(clamp(probs, eps, 1−eps))` with float32
`eps = 1.19e-7` (verified against installed torch source in the `sheeprl_bridge`
env). After the 1% unimix blend, every probability is at least
`unimix / num_classes = 0.01/32 ≈ 3.13e-4` (RSSM) or `0.01/n_actions ≈ 2e-3`
(actor) and at most `0.99·1 + 0.01/32 ≈ 0.9903` — both strictly inside the clamp
band, so the clamp never fires and our plain `jnp.log(probs)`
(agent.py:852, 1481) is exactly equal. Matches the boundary already flagged in
the v2 audit (SHEEPRL_REFERENCE_AUDIT.md §8).

### D2 — Straight-through sampler equivalence (row 13, EQUIV / D-009)

Torch: `rsample() = sample + (probs − probs.detach())` — forward value is the
one-hot sample; backward gradient w.r.t. logits is the softmax Jacobian
(verified against installed torch source). Ours (agent.py:917-923):
`hard − stop_gradient(soft) + soft` with `soft = softmax(logits)` — forward
value is `hard` exactly, backward gradient is the same softmax Jacobian.
Sampling distributions are identical (Gumbel-max ≡ categorical sampling); only
the PRNG streams differ — the declared D-009 cross-platform class.

### D3 — Sequence carry-in: learned initial state vs zeros (row 18, EQUIV)

Sheeprl seeds the training unroll with `recurrent_state = zeros`,
`posterior = zeros` (dreamer_v3.py:107-131); ours seeds with
`rssm.get_initial_states(B)` (agent.py:1671). This is provably equivalent
because both drivers force `is_first[0] = 1` for every sampled sequence
(sheeprl dreamer_v3.py:100; ours train.py:686-687), and the §S4 reset inside
`dynamic()` is `(1 − is_first)·carry + is_first·initial` — with `is_first = 1`
the carry coefficient is exactly 0.0, so the seed values (zeros or otherwise)
are annihilated at step 0 in both implementations, in value **and** in
gradient. Both then use the same `tanh(learnable)` initial state via
`get_initial_states` inside the reset. The only residual difference is one
wasted transition-model forward per `observe()` call (negligible).

### D4 — Decoder config keys are dead (row 22, UNDECL-C)

`build_agent` constructs the decoder with the **encoder's**
`dense_units`/`mlp_layers` (agent.py:2052-2058, comment "symmetric"), while the
config files carry a separate `world_model.decoder:` block
(01_food_only.yaml:67-69) that nothing reads. Sheeprl sizes its decoder from
`world_model.observation_model.*`, which in every stock preset resolves to the
same `algo.dense_units`/`algo.mlp_layers` as the encoder — so today the values
coincide at every size preset and there is **no numerical effect**. The trap:
editing the config's decoder block silently does nothing. Suggested fix
(developer): either read the keys or delete the block.

### D5 — Target-critic initialization is conditional on Polyak ordering (row 26, EQUIV-conditional)

Sheeprl's target critic starts as `deepcopy(critic)` (agent.py:1217) — byte
identical. Ours is a freshly initialized network with different random body
weights (agent.py:2114-2121). Equivalence relies on the driver firing the first
Polyak update with `tau = 1.0` **before** the first read of the target critic —
sheeprl guarantees that ordering inside its train gate (dreamer_v3.py:674-680),
and our train step's comment asserts the same ("Polyak fires in the main loop
BEFORE one_train_step", train.py:675). I did not re-verify the driver ordering
here — **flagged for the area-5 (driver) audit**. If the ordering ever regresses,
the first gradient step's critic regression target is garbage (self-limiting —
one step — but a real transient).

### D6 — Entropy epsilon fudge (row 29, UNDECL-C)

Torch computes categorical entropy as `−Σ p · log_softmax(logits)`; ours uses
`−Σ p · log(p + 1e-8)` (agent.py:1557-1559 and, on the actual loss path,
train.py:912-913). Since post-unimix probabilities are ≥ 2e-3 for the actor,
the absolute entropy error is bounded by `n_actions · 1e-8 / p·p ≈ 1e-7`-class
and the gradient perturbation is similarly negligible; there is also no
numerical-stability hazard because probs are bounded away from zero. Cosmetic,
but it *is* in the entropy-bonus gradient path — using
`jax.nn.log_softmax(logits)` directly would be both closer to the reference and
cheaper. Recurs at both sites.

### D7 — RSSM transition/representation output-layer init distribution (row 32, UNDECL-I, severity **Low**)

Under `hafner_initialization: True` (the default), sheeprl **overwrites** the
final linear of the transition and representation MLPs with
`uniform_init_weights(1.0)` — uniform on `±sqrt(3/den)` (agent.py:1173-1174) —
after the truncated-normal `init_weights` pass. Our RSSM applies the
truncated-normal `init_weights` to `transition_out` and `repr_out` and never
overwrites (agent.py:703-706, 711-714). Both are zero-mean with **identical
standard deviation** `sqrt(1/den)` (truncated normal at ±2σ with the Hafner
constant has std `σ_param·0.8796 = sqrt(1/den)`; uniform `±sqrt(3/den)` has std
`sqrt(3/den)/sqrt(3) = sqrt(1/den)`), so second moments match exactly; only the
distribution shape/support differs (bounded uniform vs bell-shaped with tails
to ±2σ·1.137). Training effect: initial prior/posterior logit statistics differ
slightly in shape at step 0 only; no effect on converged behavior is expected,
which is why severity is Low — but it is a real, undeclared divergence from the
Hafner initialization recipe on the two layers that emit the latent-state
logits, and it is invisible to the Lever-A fixture tests (they load fixed
weights). Note our Actor head, decoder head, continue head, reward head and
critic head all DO follow the sheeprl overwrite pattern — only these two RSSM
output layers were missed.

### D8 — Imagination-time actor input detach (row 38, EQUIV)

Sheeprl calls `actor(imagined_latent_state.detach())` at every rollout step
(dreamer_v3.py:219, 240). Our `imagine()` passes the latent **without**
`stop_gradient` (agent.py:1774, 1812). This is inert in our port because no
`value_and_grad` closes over `imagine()`: the rollout runs outside all three
loss functions (train.py:816), and its outputs enter the losses only as
`stop_gradient(imagined_latents)` / `stop_gradient(imagined_actions)`
(train.py:888-889, 960) or via fresh recomputation (`forward_logits` on
stop-gradient'd latents, train.py:905). Therefore no gradient can flow through
the undetached path — same effective graph as sheeprl. Fragility note: if a
future change ever differentiates through `imagine()` outputs directly, the
missing per-step detach becomes load-bearing; an explicit
`stop_gradient(new_latent)` at the actor call would make the invariant local.

### D9 — Dead rollout outputs (row 39, UNDECL-C)

`Actor.__call__` computes and `imagine()` accumulates per-step `log_probs` and
`entropies` (agent.py:1752-1826), but `one_train_step` uses only
`imagined_latents` and `imagined_actions` (train.py:818-826) and recomputes
log-prob/entropy inside the actor loss via `forward_logits` — the correct,
sheeprl-matching path (and the fix for the v1 "REINFORCE resamples actions"
root cause). The rollout-time computations are dead work: one `log_softmax` +
entropy per step per rollout (~16 extra elementwise passes over `[BT,
n_actions]` per train step — negligible at gridworld sizes, but misleading to
readers who may assume those outputs feed the loss).

### D10 — Discount factor γ mismatch (row 44, UNDECL-I, severity **Med**)

Every dreamer_srl training config sets `algo.gamma: 0.996840347` with the
comment "`sheeprl dreamer_v3.yaml:L22`" — but sheeprl's actual value at that
line is `gamma: 0.996996996996997` (= 1 − 1/333, the DreamerV3 discount). The
two imply effective credit-assignment horizons `1/(1−γ)` of **316.5 vs 333.0
steps** (~5% shorter). γ enters `compute_imagined_returns` (λ-returns) and the
cumulative discount weights of both actor and critic losses, so every
dreamer_srl run to date has trained against slightly more myopic value targets
than the sheeprl baseline it is compared to. The value `0.996840347` appears
nowhere else in the repo (not in the sheeprl tree, not in the DreamerV3-NNX
configs) — it looks like a transcription introduced at CP9 plan time
(CP9_PLAN.md:441) and propagated to all 19 config variants with the same
mis-citation. Not covered by any D-NNN row. Impact on a food-survival task
with short episodes is probably modest (episode lengths ≪ either horizon), but
it breaks strict parity claims and is trivially fixable. **Recommend:
developer sets `gamma: 0.996996996996997` (or documents the choice as a
deliberate D-row) across `configs/models/dreamer_srl/*.yaml`.**

### D11 — No gradient clipping (row 45, DECLARED-KNOWN)

Sheeprl clips gradient norms (world model 1000, actor 100, critic 100 —
dreamer_v3.yaml + dreamer_v3.py:196-200 and the analogous actor/critic sites);
our train step has no clipping. Already recorded as an **open** Known-Bugs row
("Dreamer-srl world-model loss can spike to astronomical values (no gradient
clipping)", raised to Med priority after empirical ~1e29 loss spikes; see
[[KNOWN_BUGS]] / `[[04_dreamer_srl]]` Finding 3). Listed here only because a
completeness table without it would be misleading; the fix belongs to the loss
/ optimizer audit area, not to architecture.

---

## Verdict

The RSSM and world-model **network architecture is a faithful port**: layer
structure, sizes-per-preset, activations, LayerNorm placement and epsilon,
GRU-cell internals (including the reset-inside-tanh and −1 update-gate bias
traps), unimix placement and form, straight-through sampling gradients,
initial-state parameterization, the §S4 three-quantity reset, and the
zero-init / uniform-init head discipline all match sheeprl line-for-line or
with proven equivalence, and the 9/9 bit-identity tests re-confirm it
empirically. The two undeclared findings are (1) a **wrong discount factor
(γ = 0.996840347 vs sheeprl's 0.996997) in all dreamer_srl configs** —
medium severity, affects every run's value targets, one-line fix — and
(2) a **truncated-normal instead of uniform initialization on the transition
and representation output layers** — low severity, init-time only, identical
variance. Cosmetic items: dead decoder config keys, dead rollout
log-prob/entropy outputs, an entropy `+1e-8` fudge, a duplicated recurrent
forward, vestigial head classes, and a lazy import.

### Items NOT verified here (and why)

1. **Driver-side Polyak-before-first-use ordering** (D5 condition) — lives in
   `dreamer_srl_main.py`; area 5 (driver) scope.
2. **Eval/greedy action path** (row 48) — sheeprl evaluates with `greedy=True`
   (argmax mode); whether our eval path samples or argmaxes is decided in
   `eval.py` / the driver, not agent.py; area covering eval should check.
3. **Optimizer parity** (Adam betas/eps placement, lr wiring) — train/optimizer
   area; config eps values (1e-8 / 1e-5 / 1e-5) do match sheeprl's yaml.
4. **Loss-side distribution details** (two-hot bins, SymlogDistribution,
   KL balancing, free nats) — deliberately out of area-1 scope (losses area);
   spot-reads during cross-checks showed no red flags.
5. **Standalone torch-vs-flax LayerNorm bit comparison** — not re-run in
   isolation; covered indirectly by the passing composite GRU/RSSM fixture
   tests whose tolerances (5e-4 / 2e-3) sit 100×+ below any structural-error
   signature (per D-007/D-008 analysis).
6. **Sheeprl baseline run's actual γ** — assumed to be the stock yaml value
   0.996997; if the historical sheeprl baseline (WandB `i4ulpn95`) overrode γ,
   the D10 severity call should be revisited.

Reviewed by: code-reviewer (parity audit area 1/5), 2026-07-08.
