---
title: "Parity audit area 2 — world-model losses (dreamer_srl JAX port vs vendored sheeprl)"
topic: diagnosis
status: active
created: 2026-07-06
last_updated: 2026-07-08
---

# Parity Audit Area 2 — World-Model Losses

## Verdict (plain-language entry point)

This document is one of five area reports in a from-scratch, line-by-line numerical
audit comparing our JAX re-implementation of the DreamerV3 world-model training loss
(the `dreamer_srl` package) against the PyTorch reference implementation vendored from
the sheeprl library. Because the port is a rewrite in a different framework, the code
cannot be identical — the question is whether the *mathematics* is.

**Headline: the world-model loss is at numerical parity everywhere except the
observation-reconstruction term.** The KL-balancing terms (the two regularizers that
keep the model's predicted latent state close to its inferred latent state), the
free-nats floor (the rule that stops the KL penalty from shrinking below 1.0 per
sample), the reward loss (a 255-bin discretized regression), and the continue loss
(a binary "did the episode keep going" classifier) all match the reference to within
float32 rounding noise — confirmed here with fresh empirical probes that ran the real
PyTorch reference and our real JAX code on identical inputs.

The observation loss deviates from the reference in **three** ways, none recorded in
the port's deviation ledger: (1) our decoder is trained as a real-space predictor
squashed at loss time, while the reference trains its decoder directly in compressed
(symlog) space; (2) an extra ½ factor gives the observation term **exactly half** the
weight the reference gives it relative to the KL, reward, and continue terms (measured
ratio 0.500000 against the actual reference code); (3) the reference's small-error
cutoff (squared errors below 1e-8 are zeroed) is missing from our port. The first two
were already diagnosed on 2026-07-04 (the Fable 5 diagnosis of this algorithm, Finding 2);
this audit **confirms both and quantifies them precisely**. The third is new but
negligible in magnitude. One additional latent hazard was found in how the "episode
really ended" flag is derived from the environment (dormant under all canonical
configs), and the faithful ported loss function in `loss.py` turns out to be dead
code — the training path uses a divergent inline copy, which is exactly where the
observation-loss deviations crept in.

Details, file:line citations for both sides, and probe numbers follow.

## Scope and method

- **Ours:** `src/algorithms/dreamer_srl/loss.py` (TwoHotEncoding, BernoulliSafeMode,
  IndependentBernoulli, `reconstruction_loss`), the WM-loss assembly in
  `src/algorithms/dreamer_srl/train.py:700-797` (`wm_loss_fn` inside
  `make_train_step`), symlog/symexp in `src/algorithms/dreamer_srl/utils.py:27-46`,
  and the loss-target derivations in `src/algorithms/dreamer_srl/dreamer_srl_main.py`
  (terminated flag, reward storage).
- **Reference:** `vendor/sheeprl/sheeprl/algos/dreamer_v3/loss.py` (all 88 lines),
  the loss call region in `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:146-200`,
  the distribution classes in `vendor/sheeprl/sheeprl/utils/distribution.py:152-276, 409-416`,
  and defaults in `vendor/sheeprl/sheeprl/configs/algo/dreamer_v3.yaml`.
- **Method:** every term re-derived independently from the vendored source; empirical
  probes ran the actual reference classes (torch, `sheeprl_bridge` env) and our actual
  formulas (JAX, main env) on a shared numpy fixture (`T=8, B=4, D=24`, seed 42;
  rewards ~ N(0,2); 255-bin logits ~ N(0,1); 8×8-categorical KL logits ~ N(0,1)).
  Prior docs (`DEVIATION_LOG.md` D-001…D-014, `SHEEPRL_REFERENCE_AUDIT.md`,
  `KNOWN_BUGS.md`, `diag_fable5_20260704/04_dreamer_srl.md`) were used only to
  classify declared-vs-undeclared, not as evidence of correctness.

## Comparison table (complete)

| # | Item | Ours | Reference | Classification | Evidence |
|---|------|------|-----------|----------------|----------|
| 1 | Dynamic KL: `KL(sg(post) ‖ prior)`, coefficient 0.5 | `train.py:743,747` | `loss.py:64-69` | **PARITY** | stop-gradient on the posterior side both; probe max_abs_diff 1.43e-6 |
| 2 | Representation KL: `KL(post ‖ sg(prior))`, coefficient 0.1 | `train.py:744,748` | `loss.py:70-74` | **PARITY** | stop-gradient on the prior side both |
| 3 | Free nats = 1.0, per-element `max(KL, 1.0)` over [T,B], applied **before** the 0.5/0.1 coefficients and **before** the mean | `train.py:747-748` | `loss.py:68-74` | **PARITY** | identical order: floor → coefficient → sum → mean |
| 4 | `kl_regularizer` multiplies the balanced KL sum only | `train.py:751` | `loss.py:80` | **PARITY** | value 1.0 in all canonical configs, matching `dreamer_v3.yaml` |
| 5 | KL computation form: factored-categorical sum vs torch `kl_divergence(Independent(OneHotCategoricalST,1), …)` | `train.py:735-744` | `loss.py:64-73` + `dreamer_v3.py:171-172` reshape | **EQUIVALENT-BY-DESIGN** | log-softmax form `Σ p·(log p − log q)` over D classes then S categoricals is the same math; probe 1.43e-6 on mean-KL 6.64 (≤ float32 noise) |
| 6 | Observation loss: decoder-output interpretation | `symlog()` applied to decoder output — decoder trained as real-space predictor (`train.py:709-710`) | decoder output **is** the symlog-space prediction, no transform (`distribution.py:180`) | **UNDECLARED-with-impact** (confirms 2026-07-04 Finding 2a) | see detail A |
| 7 | Observation loss: scale | extra `0.5` factor (`train.py:709`) | no ½ (`distribution.py:179-193`) | **UNDECLARED-with-impact** (confirms 2026-07-04 Finding 2b) | measured effective-weight ratio **0.500000** (range 0.499999732–0.500000179) against real torch `SymlogDistribution` |
| 8 | Observation loss: small-error tolerance | none | squared distances `< tol=1e-8` zeroed (`distribution.py:159,181`) | **UNDECLARED-cosmetic** (new) | see detail B |
| 9 | Observation loss reduction: sum over obs dim, per-key sum over decoder dict | single flat `"obs"` key, `sum(axis=-1)` (`train.py:709-711`) | `-sum([po[k].log_prob(...) for k])`, `dims=1` per mlp key (`loss.py:61`, `dreamer_v3.py:156-161`) | **EQUIVALENT-BY-DESIGN** | one flat vector key ≡ sum of per-key event-dim sums |
| 10 | Reward loss: two-hot bin grid `linspace(-20, 20, 255)` stored in symlog space | `loss.py:120` | `distribution.py:237` | **PARITY** (bin-midpoint 1-ULP drift is **DECLARED** D-006) | bins 255 in all canonical configs |
| 11 | Reward loss: encode/decode symmetry — `symlog` on target inside `log_prob`, `symexp` only at `mean`/`mode` | `loss.py:143,155,201` | `distribution.py:247,251,254` | **PARITY** | cross-weight interpolation (`weight_below = dist_to_above/total`) line-for-line |
| 12 | Reward loss: cross-entropy form `(target · (logits − logsumexp)) . sum(dims)` on raw replay rewards | `train.py:715-716`, `loss.py:242-245` | `loss.py:62`, `distribution.py:275-276` | **PARITY** | probe max_abs_diff 7.15e-6 at mean |loss| 6.12 — inside the declared D-006/D-010 substrate band (≤ 5e-5) |
| 13 | Reward target: raw env reward, no tanh clip | `dreamer_srl_main.py:1200` | `dreamer_v3.py:409,637` with `clip_rewards: False` default | **EQUIVALENT-BY-DESIGN** | omitted code path is disabled in the reference recipe |
| 14 | Continue loss: Bernoulli log-prob (BCE-with-logits) wrapped `Independent(·, 1)` | `loss.py:318-354, 401-417`, `train.py:719-723` | `distribution.py:409-416` + torch `Bernoulli.log_prob`, `dreamer_v3.py:167` | **PARITY** | probe max_abs_diff 1.19e-7; log-sigmoid form ≡ torch BCE-with-logits |
| 15 | Continue target: `1 − terminated`, **no** gamma multiplier | `train.py:722` | `dreamer_v3.py:168` (code, not the stale docstring) | **PARITY** | |
| 16 | `terminated` derivation from env: `termination_reason >= 2` (deaths), `== 1` (truncation) | `dreamer_srl_main.py:1163-1165` | native gym `terminated`/`truncated` booleans | **DECLARED** (CP7-P2, `docs/reviews/dreamer_srl_v2_cp7_driver_review.md` §P2) — with a **config-gated latent hazard, see detail C** | sound under canonical configs (`with_injury: true`, `overeating_death: false`) |
| 17 | `continue_scale_factor` = 1.0 multiplies the continue NLL | `train.py:723` | `loss.py:77`, `dreamer_v3.yaml:52` | **PARITY** | |
| 18 | Total: `(kl_reg·kl_loss + obs + rew + cont).mean()` over [T,B]; single scalar, gradients w.r.t. world-model params only | `train.py:751, 793-797` | `loss.py:80`, `dreamer_v3.py:174-200` | **PARITY** | reduction order identical |
| 19 | First-step masking | none; `is_first[0]` force-set to 1 before the WM forward | none; same force-set (`dreamer_v3.py:100`) | **PARITY** | neither implementation masks any loss element |
| 20 | Gradient clipping before the WM optimizer step | absent (`dreamer_srl_main.py:658`, plain `optax.adam`) | global-norm clip at 1000 (`dreamer_v3.py:193-199`, `dreamer_v3.yaml:52`) | **UNDECLARED-with-impact** (confirms 2026-07-04 Finding 3; now an OPEN, priority-raised row in `KNOWN_BUGS.md` with empirical 1e29–1e31 WM-loss spikes) | boundary of this area; re-confirmed absent in live code |
| 21 | Extra WM-quality probes computed inside the loss fn (reward MAE, latent entropy, continue accuracy, with `1e-8` denominators) | `train.py:753-790` | absent | **UNDECLARED-cosmetic** | aux-only; not part of `total`, no gradient contribution |
| 22 | Faithful `reconstruction_loss` port in `loss.py:424-582` | imported at `train.py:158` but **never called** — training path uses the divergent inline copy | n/a | **UNDECLARED-cosmetic** (structurally load-bearing, see detail D) | `grep` — sole references are the import and a docstring |
| 23 | KL logging split (`kl` = pre-floor dynamic KL; per-term means) | `train.py:745, 775-783` | `loss.py:81-88` | **PARITY** on shared keys (+ cosmetic extra keys) | |

## Detail A — Observation loss trains the decoder in the wrong space (row 6)

- Ours: `src/algorithms/dreamer_srl/train.py:709-711`
  `obs_log_prob = -0.5 * jnp.sum((_symlog(reconstructed_obs) - _symlog(obs_target))**2, axis=-1)`
- Reference: `vendor/sheeprl/sheeprl/utils/distribution.py:177-193` (`SymlogDistribution.log_prob`,
  instantiated at `dreamer_v3.py:156-161` with the raw decoder output as `mode`):
  `distance = (self._mode - symlog(value))**2; return -distance.sum(dims)`

In the reference, the decoder's raw output *is* the symlog-space prediction; the target
is symlog-encoded and the residual lives entirely in symlog space (real-space
reconstructions are produced by `symexp` at `mode`/`mean` consumption,
`distribution.py:170-175`). Our port instead symlog-squashes the decoder output at loss
time, so the decoder learns a real-space mapping and every gradient that reaches it is
scaled by symlog's slope `1/(1+|pred|)`.

**Quantification** (probe, decoder outputs near the env's ~[0,1]-normalized obs):
per-unit-symlog-residual gradient scale on the decoder output is `symlog'(pred)/2`,
measured **0.226–0.499** of the reference's (this folds in row 7's ½). The objective's
optimum is unchanged (perfect reconstruction minimizes both), so this is a
gradient-geometry and representation-semantics deviation, not a wrong-basin bug —
consistent with the 2026-07-04 diagnosis calling departure (a) minor for this
environment's obs range. It becomes material if obs channels leave [0,1] (the slope
then decays like 1/|pred|).

## Detail B — Extra ½ and missing tolerance (rows 7–8)

Feeding the actual torch `SymlogDistribution` and our actual formula the *same
symlog-space residuals* (shared fixture) gives ratio ours/reference =
**0.500000** (min 0.499999732, max 0.500000179 over 32 elements — pure float32
noise around exactly ½). So relative to the KL, reward, and continue terms — which
are all at parity — our world model weights observation reconstruction at **exactly
half** the reference recipe. The in-file comment at `train.py:704` ("Normal(symlog(pred), 1)")
models the reference as a unit-Gaussian log-density (which would carry ½ and constants);
the reference is an MSE distribution with neither. This mis-model is the origin of both
rows 6 and 7, as the 2026-07-04 diagnosis already concluded.

The missing `tol=1e-8` clamp (`distribution.py:159, 181`: squared distances below
1e-8 are zeroed) means our port keeps gradients for per-element symlog residuals
below 1e-4 where the reference emits exactly zero. Maximum possible per-element loss
difference is 1e-8; classified cosmetic.

## Detail C — `termination_reason >= 2` latent holes (row 16)

The mapping itself (reason 1 = max-steps truncation; reason ≥ 2 = death) is a declared
adaptation to our env's single-`done` API. Two config-gated holes, both **dormant under
every canonical config** (`configs/environment/default.yaml:147,183` —
`with_injury: true`, `overeating_death: false`), verified in env source:

1. **Overeating (known env latent bug interaction).** With `overeating_death: true`,
   `src/environment/core.py:707-708` sets `reason=3` whenever satiation ≥ max, but
   `update_body` (`core.py:117-127`) never sets `done` for overeating. The buffer would
   then mark *live mid-episode steps* `terminated=1` (possibly many consecutive steps),
   corrupting continue targets and the true-continue splice. This is the documented
   "overeating_death sets reason=3 but not done" latent bug propagating into WM training.
2. **Instant-death configs.** With `with_injury: false` and any damage source,
   `core.py:126` sets `done=True` on `damage > 0`, but no reason code fires
   (`new_injury` never reaches `max_injury`), so `reason` stays 0 → the death step is
   recorded as **neither** terminated **nor** truncated: continue target 1 on a death
   step and no bootstrap zeroing.

Recommendation: an assertion or triage note tying `terminated`-derivation validity to
`with_injury=true ∧ overeating_death=false`, or deriving `terminated` as
`done ∧ reason != 1` instead of `reason >= 2` (which closes hole 2 and half of hole 1).
Fixes belong to `developer`; no code changed by this audit.

## Detail D — The faithful port is dead code (row 22)

`src/algorithms/dreamer_srl/loss.py:424-582` is a line-for-line port of sheeprl's
`reconstruction_loss` whose observation term correctly delegates to
`po[k].log_prob(...)` (`loss.py:506`) — it would have been at parity had it been used
with a `SymlogDistribution`-equivalent. The training path never calls it: `train.py`
imports it (line 158) and then inlines a re-derived copy (`wm_loss_fn`,
`train.py:700-791`) whose hand-written observation term introduced rows 6–8.
`SHEEPRL_REFERENCE_AUDIT.md` (mapping row "loss.py:L9-L88 → our loss.py L424-582")
records the faithful port as the counterpart, which masks the inline divergence.
Flag for cleanup: either route `wm_loss_fn` through `reconstruction_loss` with a
proper Symlog distribution object, or delete the dead function and fix the inline term.

## Empirical probe summary

Shared fixture, torch side `/home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python`
(real `sheeprl.utils.distribution` classes), JAX side
`/home/vncuser/miniconda3/envs/grid_world_pain/bin/python` (real `dreamer_srl` code):

| Term | Comparison | Result |
|---|---|---|
| Observation | ours / torch `SymlogDistribution` NLL, identical symlog-space residuals | ratio 0.500000 ± 2.7e-7 → **exact ½ down-weight** |
| Observation | same raw array fed as decoder output to both | ratio 0.044–0.202 (mean 0.090) — shows rows 6+7 compound off-optimum |
| Reward (two-hot) | max abs diff | 7.15e-6 (mean |loss| 6.12) — within declared D-006/D-010 band |
| Continue (Bernoulli) | max abs diff | 1.19e-7 |
| KL (8×8 categoricals) | max abs diff vs torch `kl_divergence(Independent(...))` | 1.43e-6 (mean KL 6.64) |
| Decoder gradient scale | ours vs reference per unit symlog residual, obs ∈ [0,1] | 0.226–0.499 |

## Verdict

World-model loss parity holds for KL balance, free nats, reward, continue, targets,
weights, and reduction — every deviation found concentrates in the single inline
observation-loss expression at `train.py:709-712` (three departures: representation
space, exact ½ weight, missing tolerance) plus the already-registered absence of
gradient clipping at the optimizer boundary. The two previously diagnosed
observation-loss departures are hereby confirmed and quantified; they remain unfixed
and absent from `DEVIATION_LOG.md`. Runs trained with this code optimize a recipe
whose reconstruction term is exactly half-weighted relative to the sheeprl baseline
being parity-tracked — any WM-quality comparison against sheeprl runs must carry that
caveat until fixed.

### Conventions audit (per code-reviewer checklist)

pytree ✅ (functional updates throughout `wm_loss_fn`) · JIT ✅ (static hyperparams
closed over in `make_train_step`; no traced-value Python branching in the loss) ·
vmap ✅ (heads vmapped over flattened [T·B]) · PRNG ✅ (single `k_wm` split for the WM
forward, main key advanced) · sensor sync n/a (no obs-layout change) · config
protocol ✅ (all loss coefficients via `get_mandatory`, `dreamer_srl_main.py:513-521`)

## Unverifiable in this audit

- Full end-to-end WM **gradient** parity (would require weight-identical models across
  frameworks; only loss values and analytical gradient scales were probed).
- Two-hot behavior for |reward| beyond the symlog ±20 grid (≈ |r| > 4.8e8) — untested,
  unreachable in this env.
- Loss behavior under mixed/lower precision (both sides probed in float32 only).
- Runtime activation of the Detail C holes (requires a non-canonical env config; none
  of the active dreamer_srl configs enable one).
- `loss.py` module docstring claims bit-identity tests "pair every function"; its own
  `reconstruction_loss` docstring concedes no standalone test exists (`loss.py:497-501`)
  — the claim is unverified for that function and moot while it is dead code.

Cross-references: [[04_dreamer_srl]] (Findings 2, 3) · [[SHEEPRL_REFERENCE_AUDIT]] ·
`docs/develop/active/dreamer_srl_v1/DEVIATION_LOG.md` (D-001…D-014; none cover rows 6–8) ·
`docs/develop/active/issues/KNOWN_BUGS.md` (gradient-clipping OPEN row).

Reviewed by: code-reviewer
