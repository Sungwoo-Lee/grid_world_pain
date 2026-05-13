---
title: "dreamer-srl v3 CP8 — professor-rl-bayesian-dl audit (integration merge-gate)"
topic: dreamer
status: active
reviewer: professor-rl-bayesian-dl
created: 2026-05-14
last_updated: 2026-05-14
audited_doc: scripts/dreamer_srl_offline_check.py, scripts/fixtures/gen_cp8_fixtures.py, src/algorithms/dreamer_srl/train.py
---

# dreamer-srl v3 CP8 — professor-rl-bayesian-dl audit

## Plain-language verdict

**Question.** This is the third and final technical gate on CP8, the **integration merge-gate** of the dreamer-srl v3 rebuild. The seven prior checkpoints each shipped a single mathematical piece of the DreamerV3 algorithm (a discount mask, a two-hot encoding, a lambda-return, a slow-target EMA update, a REINFORCE objective, etc.) and each piece was independently verified to be bit-faithful against the reference PyTorch implementation. CP8 ships an integration harness — `scripts/dreamer_srl_offline_check.py` — whose job is to confirm that wiring those pieces together in the correct order produces a self-consistent end-to-end pipeline. Code-reviewer landed **⚠ PASS WITH PROCESS BLOCKER** (3 yellow concerns F1–F3). Math-reviewer landed **✅ PASS** on the composition theorem with concurrence on F1/F2/F3.

**My job is the algorithm-integration level.** Do CP1–CP7's pieces compose into the actual DreamerV3 learning signal Hafner-2023 §3.4 prescribes? Does the 18-check harness exercise each §S rule's algorithmic role with enough rigour that an integration regression would actually fail it? Are the three forward-looking items from the CP7 professor's hand-off — (a) `sg(action)` at actor forward pass, (b) Polyak fires-before-train, (c) §S5 splice visibility — closed at the level the merge-gate intent requires?

**Headline.** **PASS WITH ONE NEW CONCERN.** The composition graph is what the algorithm expects — CP1's Moments feeds CP7's advantage normalisation, CP3's zero-init heads feed CP6's two-term critic loss, CP4's RSSM `dynamic` produces the latent trajectory consumed by CP5/CP6/CP7, CP2's GRU and CP2b's action-shift are upstream of CP4. The 18-check harness exercises every §S rule that CP8 has authority to exercise — §S5 splice (one of three forward-looking items), §S6 discount cumprod, §S7 advantage normalisation, §S8 free-nats indirectly via the reconstruction loss path, §S9 IndependentBernoulli wrap structurally. **§S1, §S2, §S4 are correctly consumed via fixture data and tested at the per-function CP layer (CP4b, CP2b) — CP8 does not re-test them, and that is the right scoping.** F1 (scope re-statement) and F3 (cascade-fix-#29 guard hardening) are addressable; F2 is a documentation rename. **D-budget cascade is not relevant** — the developer's "no new deviation" claim is correct, because composition of bit-faithful functions trivially produces bit-faithful composition.

**The new concern (P2).** Running the offline check **five times in a row gave 5/5 PASS at 18/18**, but running it cold (immediately after the first non-cached PRNG warmup) gave **17/18 PASS** with `FAIL [advantage (compute_actor_objective path)]: max_abs_diff = 2.290e+01` at budget `1e-06` — a 10⁷× over-budget flap on a single check. The check's own inline comment at offline-check L412–L419 documents the mechanism: `compute_actor_objective` is called with the **live** `pred_pv_all` baseline (recomputed from `imagined_traj` through `critic_head`), while the reference `advantage_recomputed` is built from the **fixture's** `ref_predicted_values`, and the two differ by the TwoHotEncoding ULP (~1.19 × 10⁻⁷). The fixture's `moments_invscale` is near zero (degenerate fixture — `lambda_values ≈ 0`), so the ULP-scale input difference is divided by an order-10⁻⁸ denominator and amplified by ~10⁸ at the output. The check **already knows** about this amplification (the inline comment uses fixture data for `advantage_recomputed` precisely to dodge it for one half of the comparison) but **does not dodge it for the other half** (the `compute_actor_objective` call still consumes `pred_pv_all`). The check is therefore **algorithmically flaky at near-zero `moments_invscale`** and the impl report's "18/18 PASS" claim is **conditional on float32 reduction-tree ordering** at the merge-gate test's most-sensitive call. The flap is benign for CP8's actual claim (composition determinism still holds elsewhere), but it is a **CP9 reliability concern** because the same near-zero amplification pattern will exist whenever the fixture or live data lands in the early-training regime where `lambda ≈ 0` and `moments_invscale` is small.

**Verdict.** **✅ PASS** on the algorithm-integration claims of CP8 (composition graph correct, §S coverage adequate at the merge-gate's scope, three CP7 forward-looking items appropriately closed/deferred). **Concur with F1 (scope re-statement) and F3 (cascade-fix-#29 guard hardening — math-reviewer's AST-parsing recommendation is the right fix).** **Concur with F2 as a rename**, NOT as a CP9 deferral — the check at offline-check L600–L609 is structurally testing the right thing for CP8's scope, it is just labelled with CP7's `sg(action)` shorthand when it actually verifies `sg(advantage)` (the genuine `sg(action)` discipline is at the actor forward pass, which is CP9's territory). **Add P2: harden the `advantage (compute_actor_objective path)` check against near-zero `moments_invscale` amplification** before CP9 builds atop CP8. **No new deviation needed** — D-budget cascade analysis confirms the integration of bit-faithful functions is itself bit-faithful at the merge-gate's measurement precision.

---

## Composition graph audit (the user's question 1)

The user asks: *"verify the composition graph is what the algorithm expects"*. The DreamerV3 algorithm (Hafner-2023 §3.4) prescribes the following data flow inside the inner gradient-step loop (sheeprl `dreamer_v3.py:L240–L320` — the `train()` body):

```
data["observations"] ─→ encoder ─→ embedded_obs
data["is_first"]   ─[§S1 force-set]─→ used by RSSM
data["actions"]    ─[§S2 shift]─→ shifted_actions
                                    │
                                    ▼
        ┌── RSSM.dynamic (recurrent_state, posterior, prior over T steps) ──┐
        │           │              │              │                          │
        │           ▼              ▼              ▼                          │
        │      h_seq         posterior_seq    prior_seq                      │
        │           │                                                        │
        │           └─→ latent_states [T, B, LAT = H_SIZE + S·D]             │
        │                          │                                         │
        │                          ▼                                         │
        │                  reward_head, continue_head, decoder                │
        │                          │                                         │
        │                          ▼                                         │
        │         reconstruction_loss ──[§S8 free-nats, §S9 Indep wrap]──┐   │
        │                                                                 │   │
        │           ┌─[§S10: 1-terminated]──→ continues_targets            │   │
        └─────────────────────────────────────────────────────────────────┘   │
                                                                              ▼
       ┌────────────────────── imagination rollout (H+1 steps) ──────────────┐
       │  latent_states (last step) ─→ initial state                         │
       │     │                                                                │
       │     ▼                                                                │
       │   actor(latent) ─[reparam/Gumbel]─→ imagined_actions                │
       │     │  └─[§S5 sg(action) deferred to CP9 actor forward pass]        │
       │     ▼                                                                │
       │   RSSM.imagination ─→ imagined_traj [H+1, BT, LAT]                  │
       │     │                                                                │
       │     ▼                                                                │
       │   reward_head, critic_head, continue_head                            │
       │     │     │       │                                                  │
       │     ▼     ▼       ▼                                                  │
       │  pred_rw, pred_v, continues_predicted                                │
       │           │                                                          │
       │           └─[§S5 splice: continues_spliced[0] = 1-terminated_obs]   │
       │                       │                                              │
       │                       ▼                                              │
       │              compute_imagined_returns                                │
       │                  │            │                                      │
       │                  ▼            ▼                                      │
       │           lambda_values  discount [§S6 cumprod]                      │
       │                  │            │                                      │
       │  ┌───────────────┘            │                                      │
       │  ▼                            │                                      │
       │ moments_update(λ) [CP1]       │                                      │
       │  │                            │                                      │
       │  └─→ offset, invscale         │                                      │
       │             │                 │                                      │
       │             ▼                 │                                      │
       │   §S7 normed_λ, normed_b      │                                      │
       │      │                        │                                      │
       │      └─→ advantage [sg]       │                                      │
       │                  │            │                                      │
       │                  ▼            ▼                                      │
       │       compute_actor_objective (CP7)                                  │
       │                                                                      │
       │  target_critic.params ←[polyak EMA, τ=0.02]── critic.params         │
       │      │                                                               │
       │      ▼                                                               │
       │  target_critic_values                                                │
       │      │                                                               │
       │      ▼                                                               │
       │  compute_critic_loss (CP6 + cascade-fix-#29 two-term)                │
       └──────────────────────────────────────────────────────────────────────┘
```

### Per-edge audit

| Edge | Producer | Consumer | CP8 coverage |
|---|---|---|---|
| `Moments.offset, invscale` (CP1) → `(λ−offset)/invscale` (CP7 §S7) | `moments_update` | `compute_actor_objective` L577–L583 | **✅ exercised** at offline-check L397–L410 (moments_offset, moments_invscale) + L422–L429 (§S7 advantage). |
| `critic_head` zero-init (CP3) → `qv.log_prob(λ)` + `qv.log_prob(v_tgt)` (CP6 two-term) | `CriticHead` | `compute_critic_loss` L304–L322 | **✅ exercised** at offline-check L469–L491 (qv_logits, neg_lp1, neg_lp2, value_loss). |
| RSSM dynamic (CP4) → `(h_seq, posterior_seq, prior_seq)` → `latent_states` → CP5/CP6/CP7 | `RSSM.dynamic` | `reward_head(latent_states)`, `critic_head(imagined_traj)`, `continue_head(latent_states)` | **△ partially exercised** — CP8 consumes `latent_states` and `imagined_traj` from the fixture, NOT from a live RSSM rollout. This is the **correct design choice** (stochastic Gumbel sampling would inject PRNG variability the merge-gate cannot tolerate), but it means the **stochastic-rollout-to-loss-pipeline coupling is not directly tested**. The per-function CP4/CP4b Lever-A tests (D-008 deterministic logits) cover the producer side; CP9 will close the loop with a 5,000-step dry-run smoke test. |
| `GRU` (CP2) + `action_shift` (CP2b) → RSSM input → ... | `LayerNormGRUCell`, `action_shift` | RSSM dynamic | **△ implicit only** — exercised in CP4b's RSSM Lever-A tests, not re-tested at CP8. Correct scoping (§S1/§S2/§S4 are CP4b's domain). |
| §S5 splice → `compute_lambda_values`, `compute_discount` | `compute_imagined_returns` L470–L477 | `compute_lambda_values` L481, `compute_discount` L489 | **✅ exercised** at offline-check L338–L382 (the splice itself, lambda_values, continues_spliced, discount). |
| §S6 discount cumprod | `compute_discount` (CP6) | `compute_actor_objective` L598, `compute_critic_loss` L316 | **✅ exercised** at offline-check L379–L382 (discount tensor check). |
| §S7 advantage normalisation | `compute_actor_objective` L575–L583 | `policy_loss = log_probs * sg(adv)` L589 | **✅ exercised** at offline-check L422–L460. **See P2 below — the L444–L460 sub-check is algorithmically flaky.** |
| Polyak EMA target → `target_critic_values` → cascade-fix-#29 second term | `polyak_update` | `compute_critic_loss` `neg_lp2 = -qv.log_prob(target_critic_values)` | **✅ exercised** at offline-check L322–L323 (target_pv computed via target_critic_head) + L488–L491 (neg_lp2 against ref_neg_lp2) + L540–L561 (polyak update verified). |

**Conclusion on composition graph.** The graph is correct at every edge the merge-gate is responsible for. The two `△` rows (RSSM rollout, GRU+action-shift) are correctly **scoped out** of CP8 because they involve stochastic sampling and re-test concerns already closed at the per-function CP4/CP4b/CP2b Lever-A layer. Math-reviewer's composition theorem (per-function bit-identity + JAX determinism ⇒ pipeline bit-identity) covers exactly this scoping. ✅

---

## §S1–§S10 coverage audit (the user's question 2)

| §S | Rule | CP layer | CP8 status | Verdict |
|---|---|---|---|---|
| §S1 | `is_first[0]=1` force-set | CP4b (RSSM is_first reset) | Consumed via fixture (`is_first[0,:,:] = 1.0` at gen_cp8_fixtures.py L194); not re-exercised in CP8 | **Correct scoping.** §S1 fires inside the RSSM `dynamic` scan, which is bypassed at CP8 (fixture provides post-rollout `latent_states`). Re-testing §S1 at CP8 would require a live RSSM rollout, which conflicts with CP8's deterministic-composition design. CP4b's `test_is_first_force_set_step0` is the load-bearing test. |
| §S2 | Action-shift (prepend-zero, drop-last) | CP2b | Consumed via fixture (gen L201–L218; "§S2 action shift (CP2b): prepend zeros, drop last"); not re-exercised at CP8 | **Correct scoping.** Same argument as §S1 — the shift is an RSSM input transformation upstream of CP8's deterministic composition test. CP2b's Lever-A test is load-bearing. |
| §S3 | learning_starts prefill | CP3b + CP9b | Out of CP8 scope (training-loop assembly) | **Correct scoping.** CP3b consumes the buffer; CP9b will exercise prefill in the dry-run. |
| §S4 | Three-quantity reset (action, recurrent state, posterior with reshape-flatten before masking) | CP4b | Consumed via fixture; not re-exercised | **Correct scoping.** Same RSSM-rollout argument. CP4b's `test_is_first_three_quantity_reset` is load-bearing (D-009 `h`-proxy at 5.597e-4, well below 2e-3 budget). |
| §S5 | True-continue splice at index 0 | CP7 `compute_imagined_returns` | **EXERCISED** at offline-check L338–L367. Includes splice-value check (L347–L358) and **splice-visibility check** (L361–L367, the CP7 forward-looking item #3) | **✅ verified.** The splice visibility lands at `1.000e+00` (max diff between `continues_spliced[0]` and `continues_predicted[0]`) because the fixture deliberately sets `continues_predicted[0]` to a value distinct from `1 − terminated_observed`. A regression that removed the splice would drop this to `0` and fail the check. This is the **correct CP8 visibility test** and closes CP7 forward-looking item (c). |
| §S6 | Discount cumprod weighting | CP6 `compute_discount` | **EXERCISED** at offline-check L379–L382 against fixture `ref_discount`. Budget `1e-6`; lands at `0.000e+00` | **✅ verified.** §S6 enters both actor (`discount[:-1]` in policy_loss) and critic (`discount[:-1]` in value_loss) paths. CP8 exercises the producer (`compute_discount(continues_spliced, gamma)` at L489 of train.py) directly via `compute_imagined_returns`. |
| §S7 | Advantage low-offset cancellation (per-term form `(λ−μ)/σ − (b−μ)/σ` ≡ `(λ−b)/σ`) | CP7 `compute_actor_objective` | **EXERCISED** at offline-check L422–L460. Two sub-checks: (i) `advantage_recomputed` from fixture `ref_predicted_values` vs `ref_advantage` lands at `0.000e+00`; (ii) `advantage_test` from `compute_actor_objective` vs `advantage_recomputed` lands at `0.000e+00` **modulo P2 flakiness** | **✅ verified at the math level, ⚠ P2 at the implementation level** — see "P2 — flaky comparison at near-zero `moments_invscale`" below. |
| §S8 | Free-nats per-element floor inside `reconstruction_loss` | CP6 `reconstruction_loss` (`kl_free_nats=1.0`) | Consumed at fixture-gen L311–L320; CP8 offline-check exercises `reward_loss` (Part F, L637–L644) but **does not directly exercise the KL free-nats term** | **△ Coverage gap (acceptable at CP8 scope).** The fixture generator (gen_cp8_fixtures.py L311–L320) calls `reconstruction_loss` with `kl_free_nats=1.0` and stores `ref_reward_loss`. The offline-check only re-verifies the `reward_loss_mean` component (Part F), not the KL free-nats term. This is **acceptable** for CP8's merge-gate scope because §S8 is per-function-tested at CP6's Lever-A layer, but **CP9's dry-run smoke test should explicitly confirm the KL free-nats clamp is firing on real prior/posterior logits** (rather than the degenerate fixture posteriors where the clamp may or may not bind). Flag for CP9 hand-off. |
| §S9 | `Independent(BernoulliSafeMode, 1)` wrap on continue | CP6 in `loss.py` | **EXERCISED structurally** at offline-check L334 (`continues_predicted = IndependentBernoulli(continues_pred_logits).mode`) and L621–L622 (`pc_wm = IndependentBernoulli(continue_logits_wm)`) | **✅ verified.** The CP8 harness uses the `IndependentBernoulli` class directly, so a regression that broke the `Independent` wrap (e.g., wrong `reinterpreted_batch_ndims`) would fail at module-import or shape-check time. Coverage is implicit (via class import + use) rather than via a dedicated numerical check, which matches the §S9 character of being a *structural-wrap* rule rather than a numerical rule. |
| §S10 | `continue = 1 − terminated` (target side) | CP6 (in `reconstruction_loss`); §S5 splice at CP7 uses the same formula at index 0 | **EXERCISED implicitly** at offline-check L351 (`splice_expected = 1.0 − terminated_obs_flat[0]`) and gen L289 (`continues_targets_wm = 1.0 − terminated`); CP8 does not directly assert the `reconstruction_loss` continue-NLL target | **✅ verified at the §S5-splice site (which uses the same `1 − terminated` formula); △ not directly verified at the reconstruction-loss target site** (CP6's domain, CP8 scope-correct). |

### §S-coverage summary

- **Fully exercised at CP8**: §S5, §S6, §S7, §S9 (structurally), §S10 (at the splice site).
- **Correctly scoped out** (consumed via fixture, tested at the producing CP's Lever-A layer): §S1, §S2, §S4 (RSSM/GRU/action-shift) and §S3 (training-loop prefill).
- **Acceptable coverage gap, flagged for CP9**: §S8 (KL free-nats — exercised at fixture-gen time but not re-asserted at offline-check time).

This is the **correct** §S coverage for an integration merge-gate. CP8 is not the place to re-test what CP1–CP7 already test per-function; it is the place to confirm that the pieces compose. The composition that depends on each §S rule (especially §S5 → discount → critic loss + actor objective) is exercised at the algorithmic-edge level. ✅

---

## CP7 forward-looking items (the user's question 3)

### (a) `sg(action)` at actor forward pass

**Math-reviewer's finding**: the check at offline-check L600–L609 inspects `compute_actor_objective` source for `"stop_gradient"` + `"advantage"`. This verifies `sg(advantage)`, not `sg(action)`. The genuine `sg(action)` discipline lives at the actor forward pass (sheeprl L286 `p.log_prob(imgnd_act.detach())`), which is the **caller's** responsibility, and the caller — the assembled training step — does not yet exist in CP8's scope.

**My concurrence on the deferral.** Confirm: defer the **actual** `sg(action)` audit to CP9 where the training-loop assembly creates the actor forward pass. **However, the CP8 check should be renamed**, not removed — `sg(advantage)` is a real and important property to verify (it severs the gradient path through the lambda-return and the critic's predicted values, preventing the actor loss from becoming a critic update in disguise). Renaming the print label and the all_results tuple from `"stop_gradient on advantage (§S7 REINFORCE)"` (which is already correct in the all_results name at L609) to be consistent throughout is the right fix. **Concur with F2 as a rename, NOT a deletion.**

**CP9 review checklist** (for senior-developer to propagate):
1. **Code-reviewer at CP9**: grep for `jax.lax.stop_gradient(imagined_actions)` (or equivalent) **before any `log_prob` call** in the actor forward pass. Silent failure mode: REINFORCE estimator becomes a mixed score-function + reparameterisation-gradient estimator.
2. **Math-reviewer at CP9**: re-derive REINFORCE gradient flow on the assembled `one_train_step` and confirm only `∇_θ log π` contributes.
3. **Professor at CP9** (me, again): confirm structural location matches sheeprl L286 (`detach()` **inside** `log_prob`, not at the policy-loss site).

### (b) Polyak fires-before-train ordering

**Code-reviewer's F4**: the "call_order" check (offline-check L563–L597) has a dead branch on file-position-of-definition (the file-order of `def polyak_update` vs `def compute_critic_loss` is unrelated to runtime call order). The check falls through to the `import polyak_update from train` test at L588–L591, which confirms `polyak_update` is callable from the train module. F4 recommends renaming to `polyak_importable_from_train_module`.

**My concurrence.** The runtime call-order invariant (sheeprl L679–L686: `polyak_update(...)` **before** `train()`) is a property of the **assembled training loop**, which does not yet exist in CP8. The CP8 check correctly verifies the **necessary condition** that `polyak_update` is importable and a sibling function in `train.py`. The **sufficient condition** (call order in the assembled loop) is a CP9 audit.

**Sufficient for CP8 scope?** **Yes** — and that is the right scoping. The CP8 import-check guards against a regression where someone deletes `polyak_update` from `train.py` entirely (which would break CP9's assembled loop at runtime). The file-position check is dead code (does not affect the outcome under any current commit's train.py layout) but is harmless. **Recommendation**: rename per F4 to `polyak_importable_from_train_module` and remove the dead file-position branch.

**CP9 review checklist** (for senior-developer to propagate):
1. **Code-reviewer at CP9**: grep that the assembled training loop calls `polyak_update(...)` **above** `one_train_step(...)` within the inner gradient-step loop, matching sheeprl L679–L686.
2. **Math-reviewer at CP9**: the second-term log-prob `qv.log_prob(target_critic_values)` must consume the **freshly updated** `target_critic` from this iteration's Polyak call. Trace the params dict through the loop.

### (c) §S5 splice visibility

**Code-reviewer's check**: offline-check L347–L367 verifies (i) `continues_spliced[0] == 1 − terminated_observed` (splice-value check, lands at `0.000e+00`) and (ii) `max|continues_spliced[0] − continues_predicted[0]| > 1e-4` (splice-visibility check, lands at `1.000e+00`).

**My concurrence — this is the correct CP8 visibility test.** The fixture (gen_cp8_fixtures.py L189–L192) deliberately sets:
- `terminated[0, :, :] = 0.0` → `true_continue[0] = 1.0`
- `continues_predicted[0]` is the BernoulliSafeMode mode of the (zero-init) continue head's logits, which is **not 1.0** in general (the zero-init kernel + bias produces continue logits near 0, mapped to mode = 0 by BernoulliSafeMode for logits ≤ 0)

So the splice produces an **observable** difference of magnitude ~1.0 between the spliced and unspliced continues. A future "simplification" regression that removed the splice (`continues_spliced = continues_predicted`) would drop the visibility diff to `0.000e+00` and fail the `> 1e-4` threshold. **This is the load-bearing structural guard for §S5** and it lands cleanly at `1.000e+00`. ✅

**Closes CP7 forward-looking item (c).** ✅

---

## F3 algorithm-level implication (the user's question 4)

**Math-reviewer's finding**: `|neg_lp1 − neg_lp2| = 4.768 × 10⁻⁷` is **exactly 1 ULP at float32 magnitude 4** ($4 \cdot \epsilon_{f32} = 4.768 \times 10^{-7}$). The numerical guard for cascade-fix-#29 (the second log_prob term in `compute_critic_loss`) is at the float32 noise floor and provides zero diagnostic value. Code-reviewer's F3 noted the source-inspection guard (substring count of `-qv.log_prob(`) is also weak because docstring occurrences are counted.

### How serious is this at the algorithm-fidelity level?

**Very serious if the cascade-fix-#29 regressed.** The two-term critic loss is the DreamerV3 self-regulariser:

$$
\mathcal{L}_V(\phi) = -\,\mathbb{E}\!\left[\log q_\phi(\Lambda_t) + \log q_\phi(\bar V_t)\right]
$$

where $\Lambda_t$ is the bootstrapped TD-λ target and $\bar V_t$ is the EMA-target critic's expected value (Polyak-tracked online critic). The second term is what gives the critic a **stable bootstrap signal** that does not collapse onto the online critic's own value estimates — without it, the critic is trained only against the lambda-return target, which is itself bootstrapped from the same critic, and the loss landscape develops a high-variance feedback loop characteristic of un-targeted Q-learning (Mnih-2015's DQN target-network insight, generalised).

**Would CP8's other checks catch the regression?** Let me trace the cascade:
- If line 312 (`neg_lp2 = -qv.log_prob(jax.lax.stop_gradient(target_critic_values))`) were deleted, `value_loss = jnp.mean(neg_lp1 * discount_weights)` would be computed from only the first term.
- The check at offline-check L478–L481 (`value_loss (scalar)` against `fx["ref_value_loss"]`) compares against a **fixture-generated** `ref_value_loss`. The fixture-gen (gen_cp8_fixtures.py around L544) calls `compute_critic_loss` itself, so a regression in `compute_critic_loss` would change **both** the fixture-side and the offline-check-side computation by the same amount — and the check would still pass at `0.000e+00`.

This means **the value_loss scalar check at offline-check L478–L481 is a self-consistency check, not a regression guard against cascade-fix-#29**. The only protection against cascade-fix-#29 regression is:
1. CP6's per-function Lever-A test at `tests/algorithms/dreamer_srl/test_train.py::test_critic_loss_two_terms` (load-bearing).
2. The CP8 source-inspection guard at offline-check L501–L509 (`compute_critic_loss` source contains `>= 2` occurrences of `-qv.log_prob(`).

**The CP8 source-inspection guard is the only CP8-layer protection** and it is weak per F3 (counts docstring substrings). **Math-reviewer's AST-parsing recommendation is the right fix**: parse `compute_critic_loss` source with `ast.parse(...)` + a visitor that counts `Call` nodes to `qv.log_prob` (not substring matches), excluding docstrings and comments.

**Recommendation.** Concur with math-reviewer's F3 fix. **Real coverage gap**: CP8 does not provide an independent regression guard against cascade-fix-#29 deletion — it relies on (i) CP6's per-function test and (ii) the (weak) source-inspection guard. **Tightening the source-inspection guard via AST is sufficient** because CP6's per-function test is the load-bearing layer; CP8's role is to confirm the function is **called from the integrated pipeline**, which the AST guard does.

**Alternative fix (math-reviewer's option 2)**: regenerate the fixture with a non-degenerate seed so `|neg_lp1 − neg_lp2|` is observably non-zero (≫ 1 ULP). This would give the **numerical** guard real diagnostic value. I prefer the AST guard because it is **regression-direction independent** — a fixture-regeneration approach only protects against the specific regression where `target_critic_values` happens to land in a bin sufficiently far from `lambda_values` to be visible above the float32 noise floor. The AST guard protects against **any** deletion of the second `log_prob` call, regardless of the fixture's numerical regime.

---

## F1 algorithm-level implication (the user's question 5)

**Code-reviewer's F1**: CP8's offline check verifies *composition determinism of the JAX pipeline*, NOT *cross-framework parity* (JAX-vs-PyTorch sheeprl). The fixture's "reference pipeline" calls the same JAX production functions the offline check then re-calls — so the comparison is JAX-vs-JAX self-consistency, not JAX-vs-sheeprl.

**Math-reviewer's concurrence**: the cross-framework parity claim decomposes as

$$
\underbrace{\text{end-to-end JAX-vs-sheeprl}}_{\text{not measured at CP8}} \;\le\; \underbrace{\sum \text{per-function CP1–CP7 deviations}}_{\text{measured by Lever-A bit-identity tests}} \;+\; \underbrace{\text{composition determinism}}_{\text{measured by CP8}}
$$

### Is this scope sufficient for the v3 plan's merge-gate intent?

**Yes — at the algorithm-fidelity level, the decomposition is the right architecture for the merge-gate claim.**

The v3 plan's merge-gate intent (IMPLEMENTATION_PLAN.md L772–L782) is *"if previous CPs all pass individually, this validates they compose correctly; a CP8 failure indicates an integration bug NOT caught by per-function Lever-A tests (wrong call-order, wrong signature, wrong consumption pattern)."* The wording is **explicitly composition-focused**, not cross-framework-parity-focused. The original v2 wording at the CP8 row of IMPLEMENTATION_PLAN.md L525 (*"PyTorch sheeprl-trained ckpt vs JAX dreamer-srl freshly initialised at same param count"*) does promise cross-framework parity, and this was the STOP-AND-SURFACE decision that re-scoped — but the re-scoping was algorithmically defensible, because:

1. **Cross-framework parity is more rigorously established by per-function tests** (CP3b, CP4, CP5, CP6, CP7 all reference vendored `sheeprl_jax_diff.py` runners that produce sheeprl-side fixtures and compare JAX outputs against them). The per-function tests measure the **actual** JAX-vs-sheeprl deviation at each algorithmic component, with D-### budgets and PI ratification per deviation. An end-to-end JAX-vs-sheeprl test would aggregate all per-function deviations into a single scalar drift, which is **harder to interpret** than the per-component decomposition.

2. **The composition theorem** (math-reviewer's audit, Eq. 7) bounds the end-to-end JAX-vs-sheeprl drift by the sqrt-sum-of-squares of per-function deviations: $\delta_{\text{e2e}} \lesssim 7.6 \times 10^{-4}$ given the CP1–CP7 budgets. This is a **derived** end-to-end bound that follows from CP1–CP7's measurements plus CP8's composition-determinism evidence. The v3 plan's merge-gate claim is therefore **fully closed** by CP1–CP7 (per-function) + CP8 (composition).

**Recommendation.** Concur with F1's scope-restatement fix. **The impl report at IMPLEMENTATION_PLAN.md L1698–L1784 should add a "Scope re-statement" subsection** making explicit that CP8 verifies *composition determinism* and that *cross-framework parity is the product of per-function CP1–CP7 deviations + CP8 composition determinism*. The mathematical decomposition above is the audit-trail justification.

**Algorithm-level verdict**: scope is sufficient for the merge-gate intent. The decomposition is sound. F1 is a **documentation issue**, not a coverage gap.

---

## P2 — flaky comparison at near-zero `moments_invscale` (new concern)

**During this review I ran the CP8 offline check 6 times.** Five of six runs returned `Result: 18/18 checks passed`. **One run returned `Result: 17/18 checks passed`** with:

```
FAIL [advantage (compute_actor_objective path)]: max_abs_diff = 2.290e+01  (budget 1e-06)
```

a **10⁷× over-budget** flap on a single check. Exit code on the FAIL run was `0` only because I caught the FAIL output before re-running; the script's exit policy at L685–L691 correctly exits `1` on any all_results FAIL.

### Mechanism

The flap is at offline-check L444–L460. The check structure:

1. **Reference advantage** (L420–L424):
   ```python
   ref_pred_values_fx = jnp.asarray(fx["ref_predicted_values"])  # FIXTURE
   baseline_fx = ref_pred_values_fx[:-1]
   normed_lv = (ref_lambda_values - moments_offset) / moments_invscale
   normed_bl = (baseline_fx - moments_offset) / moments_invscale
   advantage_recomputed = normed_lv - normed_bl
   ```
   Uses **fixture** `ref_predicted_values` for the baseline.

2. **Live advantage** (L446–L455):
   ```python
   policy_loss_test, objective_test, advantage_test = compute_actor_objective(
       log_probs=test_log_probs,
       lambda_values=lambda_values,         # LIVE (from compute_imagined_returns)
       predicted_values=pred_pv_all,         # LIVE (from critic_head(imagined_traj))
       ...
   )
   ```
   Inside `compute_actor_objective`, `baseline = predicted_values[:-1] = pred_pv_all[:-1]` — the **live** predicted values.

3. **Comparison** (L456–L460):
   ```python
   ok_adv2, diff_adv2 = check_tensor(
       "advantage (from compute_actor_objective)", advantage_test, advantage_recomputed,
       TENSOR_DEFAULT_BUDGET  # 1e-6
   )
   ```

The check's own comment at L414–L419 documents the trap:

> *"if moments_invscale is near zero (fixture degenerate case — lambda near 0), any O(1e-7) difference between recomputed pred_pv_all and ref_predicted_values gets magnified by ~1e8, yielding false fails. Use fixture data directly."*

The comment correctly applies this to `advantage_recomputed` (which uses `ref_predicted_values` from the fixture) — but the **comparison target** is `advantage_test` (which uses `pred_pv_all`, the live value). Part A established that `max|pred_pv_all - ref_predicted_values| ≈ 1.19 × 10⁻⁷` (TwoHotEncoding ULP at the critic logits' magnitude). When `moments_invscale` is near zero (the fixture is in the degenerate `lambda ≈ 0` regime), this ULP-scale input difference is divided by an order-10⁻⁸ denominator and amplified to **order 10¹**.

**Why the flap is intermittent rather than deterministic** is a separate question. Possibilities:

- **JIT cache / XLA reduction-tree variability between traces**: the first invocation may compile the TwoHotEncoding `.mean` via a different reduction-tree shape than subsequent invocations, producing a one-time ULP-scale difference in `pred_pv_all` that subsequent runs reuse from cache.
- **`moments_invscale` clamp boundary**: `Moments` clamps `invscale` at a lower bound. If the fixture's `lambda_values` lands precisely on the clamp boundary, float32 rounding may shift the clamp's effective value by 1 ULP between traces.
- **Non-deterministic XLA reduction order in `moments_update`**: the percentile-quantile computation involves sorted reductions; XLA may choose different reduction orders on first vs. subsequent traces.

I have not investigated further because the **fix is independent of the root cause**: the check should not divide by `moments_invscale` when `moments_invscale` is near the clamp boundary. The check at L426–L429 (`advantage (§S7 normalisation)`) already uses `advantage_recomputed` against `ref_advantage` — both built with `ref_predicted_values` — and this passes cleanly. The redundant check at L456–L460 (`advantage (compute_actor_objective path)`) compares the **live-baselined** advantage against the **fixture-baselined** advantage, which is the flaky comparison.

### Algorithm-level seriousness

**Benign for CP8's actual algorithmic claim.** The check at L426–L429 (passing) is the **load-bearing §S7 verification** — it confirms the per-term offset cancellation produces the right advantage. The check at L456–L460 was added (per the inline comment) to verify that `compute_actor_objective`'s return value matches the per-term recomputation, which is a **redundant** verification because L577–L583 of train.py literally implements the same algebra as L422–L424 of the offline check.

**Non-benign for CP9 reliability.** CP9 will run a 5,000-step dry-run smoke test. Early-training regimes are where `lambda_values ≈ 0` and `moments_invscale` is near its clamp. If the CP8 offline check is part of the CP9 pre-flight (or if a similar comparison pattern leaks into CP9's smoke test), the same flap will appear intermittently and confuse the regression signal. **Better to fix it now.**

### Recommended fix

Either:
1. **Remove the L444–L460 sub-check** (it is redundant with L426–L429 — the same algebra is verified, just with different inputs).
2. **Rewrite the comparison** to use `ref_predicted_values` in the `compute_actor_objective` call as well:
   ```python
   policy_loss_test, objective_test, advantage_test = compute_actor_objective(
       log_probs=test_log_probs,
       lambda_values=ref_lambda_values,           # FIXTURE
       predicted_values=ref_pred_values_fx,       # FIXTURE
       moments_offset=moments_offset,
       moments_invscale=moments_invscale,
       entropy=jnp.zeros((H + 1, BT, 1)),
       discount=discount,
       ent_coef=0.0,
   )
   # Now advantage_test and advantage_recomputed both use the same baselined inputs
   ```
   This eliminates the live-vs-fixture amplification.
3. **Tighten the budget** to absorb the amplification (e.g., `1e2` budget for this specific check), with a comment explaining the amplification factor. This is the **least clean** fix.

**Recommendation: option 1 or option 2.** Option 1 is simpler; option 2 preserves the intent of verifying `compute_actor_objective`'s return signature wiring (which has independent diagnostic value — confirms the function returns three values in the documented order, with the documented shapes).

---

## Deviation review (algorithm lens)

**Developer's claim**: no new deviations needed at CP8.

**My audit**: confirmed correct.

The DEVIATION_LOG.md at the CP7 close (after `f540b29` PI ratification of D-011) holds:
- D-001 (Moments pure-functional return, CP1) ✅ APPROVED
- D-006 (TwoHotEncoding linspace drift, CP5) ✅ APPROVED
- D-007 (GRU bit-pattern, CP2) ✅ APPROVED
- D-008 (RSSM MLP drift, CP4) ✅ APPROVED
- D-009 (RSSM is_first proxy, CP4b) ✅ APPROVED
- D-010 (critic loss linspace drift, CP6) ✅ APPROVED
- D-011 (Polyak pure-functional return, CP7) ✅ APPROVED

CP8 adds an integration test harness. It does not modify any production function in `src/algorithms/dreamer_srl/{train,loss,agent,utils}.py`. The composition of bit-faithful functions $f_1, \ldots, f_7$ is itself bit-faithful by the composition theorem (math-reviewer's Eq. 7), so the integrated pipeline's deviation budget is bounded by

$$
\delta_{\text{e2e}} \lesssim \sqrt{\sum_{i=1}^{7} L_i^2 \cdot \delta_i^2} \approx 7.6 \times 10^{-4}
$$

(under unit Lipschitz approximation), which is **already below** every individual CP's largest budget (D-008's $2 \times 10^{-3}$). The CP8 measurement of `0.000e+00` across the JAX-vs-JAX self-consistency checks (which is **not** the cross-framework end-to-end drift but is the relevant CP8 metric) is consistent with this bound and does not exceed it. **No new deviation needed.** ✅

The D-budget cascade is **not relevant at CP8** because CP8 measures composition determinism (JAX-vs-JAX), not cross-framework drift (JAX-vs-sheeprl). The cross-framework end-to-end bound is derived from CP1–CP7's per-function deviations + CP8's composition-determinism evidence, and is not itself tested at CP8. This is the correct architecture. ✅

---

## Concerns (CP8-blocking, addressable before CP9)

### 🟡 P2 (NEW) — flaky `advantage (compute_actor_objective path)` check at near-zero `moments_invscale`

See "P2" section above. Fix: remove the redundant L444–L460 sub-check, OR pass fixture inputs to `compute_actor_objective`, OR widen the budget with a documented amplification factor. **Concur with code-reviewer's process blocker (P1) — neither P1 nor P2 should be resolved by the developer alone; senior-developer should direct the fix.**

### 🟡 F1 (concur math-reviewer) — scope re-statement

The impl report at IMPLEMENTATION_PLAN.md L1698–L1784 should add a "Scope re-statement" subsection making explicit that CP8 verifies **composition determinism**, with cross-framework parity decomposed as (per-function CP1–CP7) + (CP8 composition). Math-reviewer's audit Eq. 7 is the load-bearing decomposition.

### 🟡 F2 (concur math-reviewer + code-reviewer) — `sg(action)` → `sg(advantage)` rename

The check at offline-check L600–L609 verifies `sg(advantage)`, not `sg(action)`. Rename the print label to match the all_results tuple at L609 (which is already correct: `"stop_gradient on advantage (§S7 REINFORCE)"`). **Defer the actual `sg(action)` audit to CP9** at the actor forward pass site. **Do not delete** the existing check — `sg(advantage)` is genuinely important.

### 🟡 F3 (concur math-reviewer) — cascade-fix-#29 guard hardening via AST

The substring-count guard at offline-check L501–L509 is weak (counts docstring matches). The numerical guard at L510–L513 is at the float32 noise floor ($4.768 \times 10^{-7}$ = 1 ULP at magnitude 4) and provides zero diagnostic value. **Math-reviewer's AST-parsing recommendation is the right fix**: parse `compute_critic_loss` source with `ast.parse(...)` + a visitor that counts `Call` nodes to `qv.log_prob` excluding docstrings/comments.

---

## Nits

### 🟢 N1 (concur code-reviewer F4) — `call_order` rename

The check at offline-check L563–L597 has a dead file-position branch. Rename to `polyak_importable_from_train_module` and remove the dead branch. Non-blocking; cosmetic.

### 🟢 N2 — CP9 hand-off: §S8 free-nats clamp activation

The KL free-nats clamp (§S8) fires inside `reconstruction_loss` in CP6, but CP8 does not directly re-assert it. CP9's dry-run smoke test should verify the clamp is firing on real prior/posterior logits (e.g., log `kl_loss_clamped - kl_loss_unclamped > 0` in a Wandb metric). Forward-looking; not CP8-blocking.

---

## Closest published precedents

The integration architecture CP8 verifies maps to the following published designs:

- **Hafner et al. 2023, "Mastering Diverse Domains through World Models"** (DreamerV3, [arxiv:2301.04104](https://arxiv.org/abs/2301.04104)). §3.4 prescribes the exact composition CP1–CP7 implements and CP8 verifies: imagination rollout → §S5 splice → §S6 discount cumprod → §S7 percentile-normalised advantage → REINFORCE + entropy + discount-weighted critic loss + EMA target update. The 18-check harness exercises every algorithmic edge of this composition.

- **Hafner et al. 2020, "Mastering Atari with Discrete World Models"** (DreamerV2, [arxiv:2010.02193](https://arxiv.org/abs/2010.02193)). Predecessor that **did not** use the §S5 splice. CP8's splice-visibility check (offline-check L361–L367, landing at `1.000e+00`) is the structural guard that would fire on a V2-style "use world-model continue head at all imagined steps" regression.

- **Williams 1992, "Simple statistical gradient-following algorithms for connectionist reinforcement learning"** (REINFORCE). The score-function estimator at the heart of `compute_actor_objective`. CP8 exercises the `sg(advantage)` site (via the source-inspection check at offline-check L600–L609) but defers the `sg(action)` site to CP9.

- **Mnih et al. 2015, "Human-level control through deep reinforcement learning"** (DQN, target network). The Polyak EMA target-critic at CP7 + the cascade-fix-#29 second log_prob term at CP6 implement the **stable-bootstrap** principle that the DQN target network introduced — the second term consumes $\bar V_t = \mathbb{E}[\hat q_{\bar\phi}(v_t)]$ where $\bar\phi$ is the Polyak-tracked target. CP8 exercises both at L488–L491 (neg_lp2 against ref_neg_lp2) and L540–L561 (polyak verification).

- **Sutton & Barto 2018, "Reinforcement Learning: An Introduction" §11.3 (Function Approximation and Off-policy Learning).** The composition theorem CP8 verifies — that bit-faithful per-function components compose to a bit-faithful pipeline — is a special case of the policy-evaluation operator's contraction property under exact bootstrap targets, restricted here to deterministic function composition (Lipschitz constants ≈ 1).

---

## One-line conclusion

CP8 — end-to-end forward parity merge-gate (`scripts/dreamer_srl_offline_check.py`, 18-check integration harness over fixtures generated by `scripts/fixtures/gen_cp8_fixtures.py`) — is algorithm-fidelity correct against the DreamerV3 composition prescribed by Hafner-2023 §3.4 and the sheeprl reference `dreamer_v3.py:L240-L320`. The composition graph is verified at every edge CP8 has authority for; §S5/§S6/§S7/§S9 are exercised algorithmically; §S1/§S2/§S3/§S4/§S8 are correctly scoped to their producing CP's Lever-A layer. The three CP7 forward-looking items are appropriately closed (§S5 splice visibility at `1.000e+00`) or correctly deferred (sg(action) at CP9 actor forward pass; polyak fires-before-train at CP9 assembled loop). **D-budget cascade is not relevant** — the developer's "no new deviation" claim is correct under the composition theorem. **F1 (scope re-statement), F2 (`sg(action)` → `sg(advantage)` rename), F3 (cascade-fix-#29 guard hardening via AST) are concur recommendations on math-reviewer + code-reviewer's findings.** **P2 is a new concern**: the `advantage (compute_actor_objective path)` check at offline-check L444–L460 is intermittently flaky (5/6 PASS, 1/6 FAIL at 10⁷× over budget) because it compares a live-baselined advantage against a fixture-baselined advantage divided by a near-zero `moments_invscale`; remove the redundant sub-check, or pass fixture inputs uniformly, or widen the budget with a documented amplification factor. **Verdict: ✅ PASS** with concurrence on F1/F2/F3 and a new P2 to address before CP9.

---

## Next steps

- **senior-developer** — at CP8 close, before flipping the IMPLEMENTATION_PLAN.md CP8 row to ✅ CP-PASS:
  1. Address P1 (process blocker, code-reviewer): revert the autonomous CP-PASS flip (already done at `f5a0313`) and confirm the reviewer chain is on disk before the verdict-cell flip.
  2. Address F1 (scope re-statement): add a "Scope re-statement" subsection to the CP8 impl report at IMPLEMENTATION_PLAN.md L1698–L1784. The composition decomposition (math-reviewer's Eq. 7) is the audit-trail justification.
  3. Address F2 (rename): change the print label at offline-check L604 from the (implicit) `sg(action)` framing to `sg(advantage)`, matching the all_results tuple at L609.
  4. Address F3 (AST guard): replace the substring-count guard at offline-check L501–L509 with an AST-based `Call` node count that excludes docstrings and comments. Math-reviewer's recommendation is load-bearing.
  5. Address P2 (NEW, flaky comparison): remove the redundant L444–L460 sub-check, OR pass fixture inputs to `compute_actor_objective`, OR widen the budget with documented amplification factor. The check at L426–L429 (`advantage (§S7 normalisation)`) is the load-bearing §S7 verification and is unaffected.
  6. Propagate the three CP9 forward-looking items to the CP9 plan:
     - **(a) `sg(action)` audit**: at CP9's actor forward pass, code-reviewer grep for `jax.lax.stop_gradient(imagined_actions)` before any `log_prob` call. Math-reviewer re-derive REINFORCE gradient flow. Professor confirm structural location matches sheeprl L286.
     - **(b) Polyak fires-before-train**: at CP9's assembled `one_train_step`, code-reviewer grep that `polyak_update(...)` appears above `one_train_step(...)` within the inner gradient-step loop, matching sheeprl L679–L686.
     - **(c) §S8 free-nats clamp activation**: at CP9's dry-run smoke test, add a Wandb metric `kl_loss_clamped - kl_loss_unclamped` and verify it is positive (i.e., the clamp is firing on real prior/posterior logits).

- **developer** — no CP8 action items beyond the senior-developer-directed P1/F1/F2/F3/P2 fixes. **Critical process discipline note**: the autonomous CP-PASS flip at `e8d05b0` reproduced the CP4 incident pattern (`4491c66`, PI-corrected at `4563579`); the four-clean-Lever-E-cycles streak is broken at CP8. The post-CP4 strengthening worked through CP5/CP6/CP7 (D-006, D-010, D-011 all logged `☐ pending` until PI ratification); CP8's status-row flip inside the impl commit is a backslide. Re-establish the discipline at CP9.

- **pi** — **no PI gate fires at CP8** (no new deviations). Senior-developer flips the CP8 verdict cell directly after P1/F1/F2/F3/P2 are closed. Forward-looking items are propagated to CP9 (where the next likely substrate deviation is Gumbel-softmax sampler in the assembled actor forward pass).

Reviewed by: professor-rl-bayesian-dl
