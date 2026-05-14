---
title: "dreamer-srl v2 — full re-audit + gradient-parity + policy-learning gate"
topic: dreamer
status: active
created: 2026-05-14
last_updated: 2026-05-14
phase: 2
---

# dreamer-srl v2 — full re-audit + gradient-parity + policy-learning gate

> **Naming convention.** "v2" here means the **second iteration of the implementation-plan series**,
> NOT a second iteration of the rebuild itself. The codebase under
> [`src/algorithms/dreamer_srl/`](../../../../src/algorithms/dreamer_srl/) was authored under the
> "v3" implementation plan at [`docs/develop/active/dreamer_srl_v3/`](../dreamer_srl_v3/) (the
> historical numbering reflects v1/v2/v3 of the **rebuild attempts**; this v2 plan is the
> **second master implementation-plan document**, supplanting the v3 plan as the authoritative
> spec going forward). Cross-link below in §"Migration plan from v1".

---

## 1. Context (plain-language entry point)

**What v1 was.** Over April–May 2026 we rebuilt the sheeprl DreamerV3 algorithm
([sheeprl@33b6366](../../../../vendor/sheeprl/) — a community PyTorch implementation of
the Hafner 2024 DreamerV3 model-based RL agent) in JAX/Flax, file-by-file, with five
guardrail levers: per-function bit-identity unit tests (Lever A), mandatory
sheeprl-source line citations in every docstring (Lever B), a three-reviewer chain at
every checkpoint (Lever C — `code-reviewer` + `math-reviewer` + `professor-rl-bayesian-dl`),
a vendored pinned copy of sheeprl plus a diff tool at
[`scripts/sheeprl_jax_diff.py`](../../../../scripts/sheeprl_jax_diff.py) (Lever D), and a
deviation log + PI sign-off (Lever E). The deliverable was a parity gate: 3 seeds at
the corrected XS recipe (`256 / 1 / 256 / 256 / 24` — i.e. roughly 16× smaller than the
sheeprl base-config values, on the dominant recurrent-state axis), survival-step mean
≥ 480 over the final 20 % of a 200 000-step training run. The v3 plan closed CP1–CP10b
with 13 checkpoint-PASS verdicts, 14 substrate-class deviations approved (D-001 through
D-014, mostly cross-platform PRNG and pure-functional vs in-place mutation patterns),
and a 16 env-SPS steady-state on a single RTX 6000 Ada — projecting 3.47 hours per
parity-launch seed.

**Why v1 parity-failed.** On 2026-05-14 the 3-seed parity launch landed at
`ep_len_avg = 103.8` averaged across seeds — i.e. **the agent did not learn**
(the random-policy floor for this 100-step-timeout NoPred food task is ~100 steps; the
sheeprl baseline at the same recipe is ~500 steps; reaching 480 was the pre-registered
PASS bar). The experiment-analyzer ranked three failure hypotheses
(see [v1 parity launch](../../../experiments/active/dreamer_srl_v3/PARITY_LAUNCH.md)):

- **H1 (HIGH confidence) — gradient-side bug in the actor REINFORCE block.** The most
  likely culprit is a subtle mis-port at the gradient-flow site in
  `compute_actor_objective`: temperature, the `unimix=0.01` mixture coefficient (which
  blends a small uniform noise into the categorical action distribution to keep
  exploration alive), the entropy bonus term, or a `stop_gradient` placement that lets
  gradient leak where it should not. Forward outputs at v1 CP7 matched sheeprl to
  `1e-6`; gradients were never measured.
- **H2 (MED) — prefill / `is_first` contamination.** The `is_first` boolean
  flag tells the RSSM (Recurrent State-Space Model) "this transition starts a new
  episode, reset the hidden state"; an off-by-one or wrong-broadcast at the
  episode-boundary site would poison early training. Compounding risk: deviation D-014
  ships a one-shot 1024-gradient-step debt-repayment burst at iter
  `learning_starts=1024` instead of sheeprl's smeared per-iter rate, which on a sparse
  buffer can over-fit the first batch.
- **H3 (MED-LOW) — advantage sign bug.** A flipped sign in
  `advantage = lambda_values - predicted_values` would teach the actor to AVOID food.

**What v1 verified — and what it didn't.** v1's bit-identity tests (Lever A) verified
**forward outputs** of `compute_actor_objective`, `compute_critic_loss`, and
`compute_imagined_returns` to within `1e-6` of the PyTorch sheeprl analog on fixed-seed
fixtures. They did NOT verify that `jax.grad(loss)(params)` returns the same gradient
as `torch.autograd.grad(loss, params)` on those same inputs. The driver
(`dreamer_srl_main.py`, 603 lines), the train-step orchestrator (`make_train_step` /
`one_train_step`, ~313 lines inside `train.py`), and the wrapper modules (Encoder,
Decoder, Actor, ContinueHead, FullMLPHead, WorldModel composite, `build_agent`) shipped
under **reviewer-optional, integration-smoke-only** verification — no bit-identity
tests at all. v1's "policy learns at all" check was absent: CP9 / CP10 / CP10b
checked no-NaN, world-model-loss convergence, and the existence of metric channels;
nothing measured whether episode length rose above 100 random-policy steps.

**What v2 changes.** v2 is a **full re-audit** of every file under
[`src/algorithms/dreamer_srl/`](../../../../src/algorithms/dreamer_srl/) against a
fresh sheeprl re-reading — **prior v1 reviewer outputs are explicitly untrusted** per
user directive ("review sheeprl codes again, don't depend on the previous code reviews
as there can be mistakes"). v2 extends Lever A from forward-only to **forward +
gradient bit-identity** (`jax.grad(loss)(params)` vs `torch.autograd.grad(loss, params)`
on the same fixture inputs, ~10–15 new tests). v2 adds two **new gates** between CP9
(implementation fixes) and the re-parity launch: a Gradient-Parity Methodology spec
written by `professor-rl-bayesian-dl`, and a 20 000-step **Policy-Learning Gate** that
requires `ep_len_avg > 200` over the last-20 % window (well above the 100 random floor)
before authorizing the 3-seed re-parity launch. Total estimate ~2–3 weeks. Hypothesis
H1 is the centerpiece — if v2-CP3's gradient diff on `compute_actor_objective` surfaces
a temperature / `unimix` / entropy / `stop_gradient` mis-port, v2 closes by fixing it,
re-running v2-CP10 (policy-learning gate), and re-running v2-CP11 (parity).

**Reading order.** §2 defines the A+B+C+D verification spine (E is reserved for major
PI decision points only). §3 lists the 12 v2 CPs in execution order with their
reviewer chains and expected deviations. §4 names the implementing + reviewing agent
per CP. §5 traces the hand-off chain. §6 estimates wall-clock. §7 enumerates failure
modes. §8 maps the migration from v1 artifacts. §9 lists every NEW file v2 will
produce plus every v1 file it will modify (with inline correction notes per v1's
no-silent-rewrite convention).

---

## 2. A+B+C+D discipline for v2 (the verification spine)

v1 used five levers (A bit-identity tests / B source citations / C three-reviewer
chain / D vendored sheeprl + diff tool / E PI sign-off). v2 retains all five but
re-scopes A–D as the operational spine and reserves E for portfolio-level decision
points (pre-launch of the re-parity gate, post-analysis if v2 also fails). The
"A+B+C+D" framing below is the user's binding scoping language from the
2026-05-14 AskUserQuestion session.

### Lever A (extended) — Forward AND gradient bit-identity tests

**v1 scope (kept).** Every public function in `src/algorithms/dreamer_srl/*.py`
has a paired test under `tests/algorithms/dreamer_srl/test_<file>.py` that imports the
JAX function and the PyTorch sheeprl analog, runs both on a fixed-seed fixture, and
asserts `max_abs_diff < threshold` (default `1e-6`; relaxed per deviation row in
[v1 `DEVIATION_LOG.md`](../dreamer_srl_v3/DEVIATION_LOG.md) for substrate-class drift).

**v2 extension (new).** Every gradient-producing function gets an **additional
gradient-side test** that compares the JAX gradient to the PyTorch sheeprl gradient on
the same fixture inputs and the same parameter dict at fixed PRNG seed. Coverage
estimate: ~10–15 new tests across the actor objective, critic loss, imagined-returns
chain, polyak update (no-op — pure functional), the `sg(advantage)` and `sg(action)`
stop-gradient sites, and the optimizer step (Optax adam vs `torch.optim.Adam` epsilon
placement). Each test specifies:

- **Threshold.** Default `1e-5` (looser than forward's `1e-6` because gradient
  computation involves one additional float32 backward pass; the threshold per
  function is set by the methodology doc, see v2-CP2).
- **Cited sheeprl line range.** Path under `vendor/sheeprl/` + line range, e.g.
  `vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L307-L341`.
- **Fixture seed.** New v2 fixtures use the convention `0xD3EAF + 0x100*<cp_id>` to
  avoid collision with v1 forward fixtures (which used `0xD3EAF` and `0xD3EAF + 1`).
- **Per-test gradient site list.** A test docstring header naming which gradient
  flows it covers, e.g. "covers ∂L_actor/∂π_logits via the
  `−log_prob × sg(advantage)` term".

**Diff-tool extension.** [`scripts/sheeprl_jax_diff.py`](../../../../scripts/sheeprl_jax_diff.py)
is extended with a `register_grad_diff(...)` helper that takes a forward-pass callable,
a loss callable, and a parameter dict; it runs `jax.grad` on the JAX side and
`torch.autograd.grad` on the sheeprl side and emits a JSON file
`tmp/diff_runs/<fn>_grad.json` with `max_abs_diff`, `max_rel_diff`, the threshold,
and the margin. The forward and gradient diffs run side-by-side in the same script
invocation so a developer can compare them in one pass.

**v2 forward-pass tests inherit v1's PASS history but get re-read.** Per the user
directive, the v1 forward-pass test results stay (they are run by `pytest` in CI), but
v2-CP1 re-reads the sheeprl source they cite, and any test whose citation has drifted
gets a fresh review.

### Lever B — Source-citation discipline (extended to gradient sites)

**v1 scope (kept).** Every function in `src/algorithms/dreamer_srl/*.py` carries a
docstring header `# Ported from sheeprl@33b6366:<file>:L<start>-L<end>`. The audit at
v2-CP1 re-validates these citations against the vendored sheeprl.

**v2 extension.** Each **gradient site** — every place where `stop_gradient` is
applied, every place where the `unimix=0.01` mixture coefficient is multiplied in,
every place where the entropy bonus is added or temperature scaled, every place where
advantage normalization is computed — gets an **extra citation** noting the
gradient-flow semantics. Example:

```python
# Ported from sheeprl@33b6366:dreamer_v3.py:L307-L341
# Gradient semantics: advantage is sg-stopped before multiplying by log_prob;
# unimix=0.01 is applied to the categorical mixture INSIDE the log_prob computation
# so gradient flows through the (1 - unimix) * π component only.
# Sheeprl source line: dreamer_v3.py:L321 (the `(advantage).detach()` site).
```

The v2-CP1 fresh audit will validate every existing citation against the vendored
`vendor/sheeprl/sheeprl/algos/dreamer_v3/` source and produce a per-function
citation-status matrix in
[`SHEEPRL_REFERENCE_AUDIT.md`](SHEEPRL_REFERENCE_AUDIT.md) (created at v2-CP1).

### Lever C — Three-reviewer chain on the SHEEPRL ORIGINAL (fresh, do not trust v1 reviews)

**Binding user directive (2026-05-14):** *"review sheeprl codes again, don't depend on
the previous code reviews as there can be mistakes."*

**v2 scope.** Every v2 CP runs the full three-reviewer chain — `code-reviewer` →
`math-reviewer` → `professor-rl-bayesian-dl` — but the **target of review is the
sheeprl block under `vendor/sheeprl/`**, not the v1 JAX port and not the v1 reviewer
outputs. Each reviewer reads:

1. The sheeprl PyTorch source at the cited line range.
2. The JAX port at the corresponding `src/algorithms/dreamer_srl/<file>.py` range.
3. The paired Lever-A test (forward + new gradient test).
4. The corresponding v1 reviewer output **only as a non-authoritative reference** —
   the reviewer's job is to either confirm the v1 verdict on the sheeprl block or flag
   a divergence.

v2 review outputs land under `docs/reviews/dreamer_srl_v2_cp<N>_<reviewer>.md`. v1
reviews under `docs/reviews/dreamer_srl_v3_cp<N>_<reviewer>.md` stay in place for
historical reference but are EXPLICITLY UNTRUSTED for v2 verdict purposes.

**Substrate-class precedent.** Deviations matching the substrate classes already
approved in v1 (D-001 / D-011 pure-functional return; D-002 / D-009 cross-PRNG
stochastic divergence; D-003 / D-006 / D-007 / D-008 / D-010 float32 ULP drift;
D-014 boundary-debt smear) are approved by `senior-developer` inline without
PI consultation, matching the v1 reviewer-optional precedent for substrate-class.
Lever E fires only at major decision points (see below).

### Lever D — Vendored sheeprl + diff tool (extended to gradient diffs)

**v1 scope (kept).** [`vendor/sheeprl/`](../../../../vendor/sheeprl/) is pinned to
commit `33b6366`. [`scripts/sheeprl_jax_diff.py`](../../../../scripts/sheeprl_jax_diff.py)
runs forward-pass diffs registered per function.

**v2 extension.** The diff tool gains a gradient-diff registry (see Lever A above).
Output is `tmp/diff_runs/<fn>_grad.json` per registered grad-fixture; the per-CP
gradient-status matrix lives in
[`GRAD_PARITY_METHODOLOGY.md`](GRAD_PARITY_METHODOLOGY.md) (created at v2-CP2).

### Lever E (reserved) — PI sign-off at major decision points only

**Not in the v2 user-binding A+B+C+D framing.** Lever E is preserved from v1 but
fires only at major decision points:

- **Pre-launch of v2-CP10 / v2-CP11** (the policy-learning gate + the re-parity
  launch) — the v2 plan triggers a PI consult per [pi.md](../../../../.claude/agents/pi.md)
  "pre-launch of a multi-run experiment".
- **Post-analysis of v2-CP11** if it also FAILS — the PI is consulted on whether to
  pivot to the sheeprl-bridge option (option D from the original 2026-05-14 scoping
  question), restart the rebuild, or escalate to a deeper diagnostic.
- **NOT consulted** on routine substrate-class deviation approvals (those flip
  inline by `senior-developer`).

---

## 3. Checkpoint table (v2)

The 12 v2 CPs execute in the order listed. Each row's "Expected deviations" column
flags substrate-class precedents from v1 that are expected to re-surface and be
re-approved by `senior-developer` inline; novel deviations escalate to Lever E.

| CP id | Scope | New Lever-A tests | Reviewer chain (fresh, see Lever C) | Expected deviations | Authoring agent | Verdict |
|---|---|---|---|---|---|---|
| **v2-CP1** | **Fresh sheeprl re-read — foundation memo.** Read-only audit of the sheeprl actor / critic / imagined-rollout / driver source at `vendor/sheeprl/sheeprl/algos/dreamer_v3/`. Output: [`SHEEPRL_REFERENCE_AUDIT.md`](SHEEPRL_REFERENCE_AUDIT.md) — a per-function table of (a) canonical sheeprl behaviour, (b) gradient-flow site map, (c) line-citation validation of v1's existing Lever-B citations, (d) discrepancy flags where v1 citations have drifted from current vendored source. ALL downstream v2 CPs reference THIS memo, not v1 reviewer outputs. | none (read-only) | `code-reviewer` (fresh) reads the sheeprl source from scratch and signs off the audit memo as faithful to the vendored copy. | none — read-only | senior-developer | ☐ pending |
| **v2-CP2** | **Gradient-parity test methodology spec.** Output: [`GRAD_PARITY_METHODOLOGY.md`](GRAD_PARITY_METHODOLOGY.md) — how to construct `jax.grad`-vs-`torch.autograd.grad` fixture pairs at fixed seed, per-function threshold table, treatment of cross-PRNG-stochastic gradient cases (where forward sampling differs but gradient w.r.t. logits should still match), gradient-diff registry schema for the extended diff tool. | none (spec only) | `professor-rl-bayesian-dl` writes the spec; `math-reviewer` validates the threshold table against the forward thresholds in v1's `DEVIATION_LOG.md`. | none — spec only | senior-developer + professor-rl-bayesian-dl | ☐ pending |
| **v2-CP3** | **`compute_actor_objective` forward + gradient re-audit (CENTERPIECE — surfaces H1).** Re-read sheeprl `dreamer_v3.py:L307-L341` (the actor REINFORCE block including `unimix=0.01`, the categorical mixture, the entropy bonus, the temperature, and the `sg(advantage)` stop-gradient site). Re-verify v1's forward Lever-A test; add 3–4 new gradient tests covering ∂L_actor/∂π_logits at each of the (a) clean REINFORCE term, (b) entropy bonus term, (c) unimix mixture coefficient, (d) stop-gradient site. | 3–4 new grad tests | `code-reviewer` → `math-reviewer` → `professor-rl-bayesian-dl` chain on the sheeprl original. | likely 1–2 substrate-class (float32 ULP drift in cross-platform `log_softmax` path; cross-PRNG stochastic divergence if a sampled term is in scope). NEW deviations (e.g. semantic mis-port at `unimix` or `stop_gradient`) escalate to fix at v2-CP9. | senior-developer + reviewer chain | ☐ pending |
| **v2-CP4** | **`compute_critic_loss` forward + gradient re-audit (surfaces H3).** Re-read sheeprl `dreamer_v3.py:L300-L341` (the critic loss including `λ-return - predicted_values` advantage sign convention, the symlog-space two-hot encoding of value targets, the polyak-update target network). Re-verify v1's forward Lever-A test (D-010 substrate-class threshold-raised precedent applies); add 2–3 new gradient tests covering ∂L_critic/∂value_logits, ∂L_critic/∂lambda_target (should be zero — sg site), advantage sign correctness. | 2–3 new grad tests | `code-reviewer` → `math-reviewer` → `professor-rl-bayesian-dl` chain on the sheeprl original. | likely 1 substrate-class (D-010-class — `linspace` 1-ULP drift cascading through TwoHotEncoding). NEW deviation: sign-flipped advantage triggers an immediate fix at v2-CP9. | senior-developer + reviewer chain | ☐ pending |
| **v2-CP5** | **`compute_imagined_returns` + λ-return chain forward + gradient re-audit.** Re-read sheeprl `dreamer_v3.py:L165-L215` (the imagined-rollout horizon-H loop, the λ-return backward recurrence, the §S5 splice point where the WM-rollout-tail meets the actor-rollout-head). Re-verify v1's forward Lever-A test; add 2–3 new gradient tests covering ∂λ_return/∂value_estimates, ∂λ_return/∂rewards, the splice gradient continuity. | 2–3 new grad tests | `code-reviewer` → `math-reviewer` → `professor-rl-bayesian-dl` chain on the sheeprl original. | likely 1 substrate-class (float32 ULP drift over the H-step recurrence). | senior-developer + reviewer chain | ☐ pending |
| **v2-CP6** | **`make_train_step` / `one_train_step` orchestrator re-audit (the composition surface).** The ~313-line `train.py` block that composes WM-update + actor-update + critic-update + polyak + imagined-rollout into one JIT-compiled step. v1 verified this only via integration smoke. Add a **full-step gradient check**: a single deterministic step on a fixture batch produces JAX gradients on all three loss heads that match the PyTorch sheeprl analog within Lever-A thresholds. Verify Optax adam vs `torch.optim.Adam` epsilon-placement equivalence (sheeprl `dreamer_v3.py:L656-L680`). | 1 full-step grad test + 2 optimizer-step tests | `code-reviewer` → `math-reviewer` chain on the sheeprl original. | likely 1–2 substrate-class (D-001/D-011-class pure-functional state). NEW deviation if Optax adam epsilon differs semantically from `torch.optim.Adam` ε at the gradient site. | senior-developer + reviewer chain | ☐ pending |
| **v2-CP7** | **`dreamer_srl_main.py` driver re-audit (surfaces H2).** The 603-line driver: env-step collection loop, buffer management (`SequentialReplayBuffer.add` / `.sample`), the prefill gate (`learning_starts=1024`), the train-gate ratio (D-014 boundary-debt site), the `is_first` propagation from env-reset to RSSM-state-reset, WandB logging, episode-boundary handling, evaluation cadence. v1 verified this only via integration smoke. Add **driver-level invariant tests** (not bit-identity but rather invariant assertions): (a) `is_first[t]` is True iff the env reset at step t; (b) zero gradient steps fire for `iter < learning_starts`; (c) the §S3 hard invariant from D-014 holds. | 3–4 new invariant tests | `code-reviewer` → `math-reviewer` chain on the sheeprl driver original (`dreamer_v3.py:main` train loop). | likely 1 substrate-class (D-014 boundary-debt re-confirmed) and possibly one new H2-related deviation if `is_first` mis-propagation surfaces. | senior-developer + reviewer chain | ☐ pending |
| **v2-CP8** | **Wrapper modules re-audit.** For each of Encoder, Decoder, Actor, ContinueHead, FullMLPHead, WorldModel composite, `build_agent`: verify forward output matches sheeprl analog on a fixture batch within Lever-A threshold, and verify gradient flow is structurally clean (no spurious `stop_gradient`, no missing nn.Module / `nnx.Module` registration that loses a parameter from the gradient tape). | 1 per wrapper (~7 tests) | `code-reviewer` chain on the sheeprl wrapper originals. | likely 2–3 substrate-class (D-007/D-008-class accumulation drift through MLP stacks). | senior-developer + reviewer chain | ☐ pending |
| **v2-CP9** | **Apply fixes for divergences found at v2-CP3 through v2-CP8.** Each fix lands as a single named PR-equivalent commit with a paired Lever-A test (forward + grad) that fails on the pre-fix code and passes on the post-fix code. Re-run the v1-CP8 composition-determinism offline check on the corrected XS config to confirm no regression. | regression tests per fix | `code-reviewer` per fix commit. | substrate-class only at this point (semantic bugs are fixed, not deviated). | developer (implementer) + senior-developer (verifier) | ☐ pending |
| **v2-CP10** | **NEW — Policy-learning gate.** A 20 000-step training run at the corrected XS config (`configs/dreamer_srl/01_food_only.yaml` post-config-correction: `dense_units=256`, `mlp_layers=1`, `recurrent_state_size=256`, `transition/representation hidden_size=256`, `cnn_channels_multiplier=24`, `learning_starts=1024`), single RTX 6000 Ada (node 114 GPU 0). **PASS criterion**: `ep_len_avg > 200` averaged over the last-20 % window (training steps 16 000–20 000). Rationale: 100 is the random-policy floor at the 100-step timeout cap; 500 is the saturated cap (sheeprl baseline); 200 sits well above the floor and well below saturation — sufficient to demonstrate the policy is learning without requiring a full parity-launch budget. **The 3-seed parity launch (v2-CP11) is ONLY authorized to spawn if v2-CP10 PASSes.** Catches the v1 silent-fail mode (random-floor parity launch) up-front at 5× lower wall-clock cost than running 3 full 200k-step seeds. | none — empirical gate | analysis: `experiment-analyzer`. Launch via `training-runner`. | none — empirical gate | training-runner (launch) + experiment-analyzer (verdict) | ☐ pending |
| **v2-CP11** | **Re-parity launch.** 3 seeds at corrected XS, 200 000 training steps each, same pre-registered PASS criterion as v1 (3-seed mean ≥ 480 over the final-20 % window AND all 3 seeds ≥ 400). Comparison to sheeprl baseline at WandB [`i4ulpn95`](https://wandb.ai/sungwoolee/grid_world_pain_dreamer_srl_smoke/runs/i4ulpn95) on the same recipe. **PI consult (Lever E) at this gate.** | none — empirical gate | analysis: `experiment-analyzer`. Launch via `training-runner`. PI consult via `pi`. | none — empirical gate | pi (consult) + training-runner (launch) + experiment-analyzer (verdict) | ☐ pending |
| **v2-CP12** | **Verdict + portfolio call.** `experiment-analyzer` writes the verdict; `pi` consults if any portfolio-level question lands (e.g., v2 also FAILs → pivot to sheeprl-bridge?). | none | analysis: `experiment-analyzer`. PI consult if needed. | none | experiment-analyzer + pi | ☐ pending |

---

## 4. Agent assignment per CP

| CP id | Implementing agent | Reviewer chain (fresh, on sheeprl original) | Launching agent (if training run) |
|---|---|---|---|
| v2-CP1 | senior-developer (writes the audit memo) | code-reviewer (validates against vendored sheeprl) | — |
| v2-CP2 | professor-rl-bayesian-dl (writes the methodology spec) | math-reviewer (validates threshold table) | — |
| v2-CP3 | developer (ports / re-audits / writes new tests) | code-reviewer → math-reviewer → professor-rl-bayesian-dl | — |
| v2-CP4 | developer | code-reviewer → math-reviewer → professor-rl-bayesian-dl | — |
| v2-CP5 | developer | code-reviewer → math-reviewer → professor-rl-bayesian-dl | — |
| v2-CP6 | developer | code-reviewer → math-reviewer | — |
| v2-CP7 | developer | code-reviewer → math-reviewer | — |
| v2-CP8 | developer | code-reviewer | — |
| v2-CP9 | developer (per-fix commits) | code-reviewer per fix; senior-developer verifies the bundle | — |
| v2-CP10 | training-runner (launches; configs read-only) | experiment-analyzer (writes verdict) | training-runner |
| v2-CP11 | training-runner (launches 3 seeds; configs read-only) | experiment-analyzer (writes 3-seed verdict); pi (consult before launch and on failure) | training-runner |
| v2-CP12 | experiment-analyzer | pi (consult if v2-CP11 FAILs) | — |

---

## 5. Hand-off chain

Each CP authorizes the next. Pre-conditions are explicit.

1. **v2-CP1** (SHEEPRL_REFERENCE_AUDIT.md PASS) authorizes spawning of **v2-CP2** (the methodology spec needs the foundation memo).
2. **v2-CP2** (GRAD_PARITY_METHODOLOGY.md PASS by math-reviewer) authorizes spawning of **v2-CP3** (the per-function grad tests need the threshold table and fixture conventions).
3. **v2-CP3** (compute_actor_objective re-audit PASS — all new grad tests pass OR all surfaced divergences are queued for v2-CP9 fix) authorizes spawning of **v2-CP4** (parallel CP would be possible but sequential ordering enforces reviewer-load smoothing).
4. **v2-CP4** PASS authorizes spawning of **v2-CP5**.
5. **v2-CP5** PASS authorizes spawning of **v2-CP6** (the composition surface needs all three loss heads to be cleared first).
6. **v2-CP6** PASS authorizes spawning of **v2-CP7** (the driver re-audit depends on a correct train-step orchestrator).
7. **v2-CP7** PASS authorizes spawning of **v2-CP8** (wrapper re-audit can run in parallel with v2-CP3–v2-CP7 if reviewer bandwidth allows; sequential default ordering shown).
8. **v2-CP8** PASS authorizes spawning of **v2-CP9** (the fix bundle implements all queued divergences from v2-CP3–v2-CP8).
9. **v2-CP9** PASS (all queued divergences fixed; regression tests pass; v1-CP8 composition-determinism check re-confirms no regression) authorizes spawning of **v2-CP10** (the policy-learning gate).
10. **v2-CP10** PASS (`ep_len_avg > 200` over the 16k–20k window on a single seed) authorizes spawning of **v2-CP11** (the re-parity launch). **If v2-CP10 FAILs**, return to v2-CP3 for a deeper re-audit pass (see §7 failure modes).
11. **v2-CP11** (3-seed mean ≥ 480 AND all 3 seeds ≥ 400) authorizes spawning of **v2-CP12** (verdict + portfolio call).
12. **v2-CP12** writes the final verdict.

**PI consult points (Lever E):**

- Pre-v2-CP10 launch: the senior-developer pings `pi` with the per-CP deviation delta from v2-CP3–v2-CP9 for a portfolio-level "is this ready to launch?" call.
- Pre-v2-CP11 launch: the senior-developer pings `pi` for the 3-seed launch authorization.
- Post-v2-CP11 if FAIL: `pi` consults on the pivot question (sheeprl-bridge / restart / deeper diagnostic).

---

## 6. Estimate

Target total wall-clock: **2–3 weeks** end-to-end from v2-CP1 kickoff to v2-CP12
verdict.

| CP | Estimate | Critical-path notes |
|---|---|---|
| v2-CP1 | 1–2 days | Read-only senior-developer audit; one code-reviewer pass. |
| v2-CP2 | 1 day | Single professor + math-reviewer pass; small spec. |
| v2-CP3 | 2–3 days | The H1 centerpiece. Full 3-reviewer chain; 3–4 new tests; likely surfaces 1+ semantic divergence to queue for v2-CP9. |
| v2-CP4 | 1–2 days | 3-reviewer chain; 2–3 new tests; H3 check. |
| v2-CP5 | 1–2 days | 3-reviewer chain; 2–3 new tests; splice scrutiny. |
| v2-CP6 | 1–2 days | 2-reviewer chain (code + math); 1 full-step grad + 2 optimizer tests. |
| v2-CP7 | 1–2 days | 2-reviewer chain (code + math); 3–4 new invariant tests; H2 check. |
| v2-CP8 | 1–2 days | code-reviewer only; ~7 wrapper tests in parallel. |
| v2-CP9 | 1–2 days | Implementation of queued fixes from v2-CP3–v2-CP8; per-fix Lever-A regression test. |
| v2-CP10 | 0.5 day wall (20 000 steps at 16 SPS = 21 minutes compute + launch + verdict cycle) | Policy-learning gate; single seed. |
| v2-CP11 | 0.75 day wall (3 seeds × 200 000 steps at 16 SPS = 3.47 h/seed; parallel-able if 3 GPUs free → 4 h, sequential ≈ 11 h) | Parity launch; PI consult before. |
| v2-CP12 | 0.5 day | Verdict write-up; optional pi pivot consult. |

Subtotals: re-audit phase (v2-CP1–v2-CP8) ≈ 9–14 days; fix + smoke + parity
(v2-CP9–v2-CP11) ≈ 2–3 days; verdict ≈ 0.5 day. **Re-parity launch start ≈ day 12;
verdict ≈ day 14.**

---

## 7. Failure-mode catalog

What if v2 also fails parity? The branch points below pre-register the next move so
re-decision overhead is minimal.

### Path A — v2-CP3 surfaces H1 (HIGH-confidence actor REINFORCE gradient bug)

This is the most likely outcome per the experiment-analyzer's ranking. v2-CP3's
gradient-side test on `compute_actor_objective` returns `max_abs_diff > 1e-4` at the
`unimix` site OR the `stop_gradient(advantage)` site OR the entropy bonus site. The
divergence is queued for v2-CP9; v2-CP9 implements the fix as a single-line or
small-block correction; the new Lever-A grad test passes; v2-CP10 PASSes; v2-CP11
parity-launches and (most likely) PASSes.

### Path B — v2-CP4 surfaces H3 (advantage sign bug)

`compute_critic_loss` or the surrounding advantage computation has
`advantage = predicted_values - lambda_values` (sign flipped) somewhere. v2-CP9
flips the sign back; the fix is mechanical. v2-CP10 PASSes; v2-CP11 PASSes.

### Path C — v2-CP7 surfaces H2 (prefill / `is_first` contamination)

The driver mis-broadcasts `is_first[t]` to the RSSM, or D-014's debt-burst is too
aggressive on the sparse early buffer. v2-CP9 patches the propagation; if D-014 is
the root cause we either tighten the burst or port sheeprl's smeared rate (overturning
the v1 D-014 approval). v2-CP10 PASSes; v2-CP11 PASSes.

### Path D — ALL audits clean, v2-CP10 PASSes, v2-CP11 FAILs (the worst case)

If every gradient-parity test passes and the 20k-step gate PASSes but the 200k-step
3-seed parity still lands at the random floor, the issue is deeper than a per-function
mis-port. Candidates:

- **Hyperparameter mismatch** in the corrected XS config vs sheeprl's real XS overlay
  (e.g., `replay_ratio`, `learning_starts`, `actor_lr`, KL-balance coefficient, free
  bits, slow-target update tau, or the discount factor). Diagnostic: dump both
  configs side-by-side and re-grep against
  [`vendor/sheeprl/sheeprl/configs/algo/dreamer_v3_XS.yaml`](../../../../vendor/sheeprl/sheeprl/configs/algo/dreamer_v3_XS.yaml).
- **Env semantics divergence** between the JAX env and what sheeprl was trained on.
  Diagnostic: run the sheeprl baseline against the JAX env (not the sheeprl env) to
  isolate the env layer.
- **JIT-compile-vs-step semantics** — a JAX functional purity violation that does not
  surface in per-step tests but accumulates over 200k steps. Diagnostic: pure-Python
  un-JIT-ed 1000-step micro-run and compare per-step gradients to JIT version.

**At Path D the PI is consulted** for a portfolio call. The original 2026-05-14
scoping question included option **D — pivot to sheeprl-bridge** (use sheeprl directly
with the JAX env as a bridge), which becomes a candidate path. The decision is the
user's; `pi` surfaces the trade-off.

### Path E — v2-CP10 FAILs (policy-learning gate)

The 20k-step gate fails: `ep_len_avg < 200` over the 16k–20k window. v2-CP9's fixes
weren't enough. Loop back to v2-CP3 with the v2-CP10 trace as a new diagnostic:
which loss head's gradient is still wrong? Likely a second-order gradient issue
(e.g., the gradient was bit-identical at a single fixture seed but the wider sample
of training-step seeds exposes a divergent path the fixture missed). Add a
broader-fixture grad test and re-run.

---

## 8. Migration plan from v1

| v1 artifact | v2 disposition |
|---|---|
| [`src/algorithms/dreamer_srl/`](../../../../src/algorithms/dreamer_srl/) (5 source files, ~5021 lines total) | **Stays in place.** v2 fixes land inline as named commits; no parallel rebuild. |
| [`tests/algorithms/dreamer_srl/`](../../../../tests/algorithms/dreamer_srl/) (test_agent.py, test_buffers.py, test_end_to_end_parity.py, test_loss.py, test_prefill.py, test_train.py, test_utils.py) | **Stays in place.** v2 adds new grad tests as `test_<file>_grad.py` siblings; existing forward tests are not modified. |
| [`tests/fixtures/dreamer_srl/`](../../../../tests/fixtures/dreamer_srl/) (`*.npz` fixtures, seed `0xD3EAF` and `0xD3EAF + 1`) | **Stays in place.** v2 adds new grad fixtures with seed convention `0xD3EAF + 0x100*<cp_id>` to avoid collision. |
| [`vendor/sheeprl/`](../../../../vendor/sheeprl/) (pinned commit `33b6366`) | **Stays in place.** v2 re-reads from the same pinned commit. |
| [`scripts/sheeprl_jax_diff.py`](../../../../scripts/sheeprl_jax_diff.py) | **Extended in place at v2-CP2.** New `register_grad_diff(...)` helper added; existing forward-diff registry not modified. |
| [`docs/develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md`](../dreamer_srl_v3/IMPLEMENTATION_PLAN.md) (v1 master plan, "v3" by historical naming) | **Stays active.** v2 references it as historical context; does NOT set `superseded_by`. The v1 plan documented the CP1–CP10b PASS chain — that history is preserved unchanged. v2 is a follow-on audit, not a replacement. |
| [`docs/develop/active/dreamer_srl_v3/DEVIATION_LOG.md`](../dreamer_srl_v3/DEVIATION_LOG.md) (D-001 through D-014) | **Stays active.** v2's new deviations append to a new log at [`DEVIATION_LOG.md`](DEVIATION_LOG.md) in this directory, starting at D-015. Cross-link both ways. |
| [`docs/develop/active/dreamer_srl_v3/CP9_PLAN.md`](../dreamer_srl_v3/CP9_PLAN.md), [`CP9B_PLAN.md`](../dreamer_srl_v3/CP9B_PLAN.md), [`CP10B_SPEC.md`](../dreamer_srl_v3/CP10B_SPEC.md), [`CP3B_SPEC.md`](../dreamer_srl_v3/CP3B_SPEC.md), [`CONFIG_CORRECTION_PLAN.md`](../dreamer_srl_v3/CONFIG_CORRECTION_PLAN.md), [`NNX_CONVENTIONS.md`](../dreamer_srl_v3/NNX_CONVENTIONS.md) | **Stay active in the v1 directory.** v2 cross-references them as needed; does not modify them silently. |
| `docs/reviews/dreamer_srl_v3_cp<N>_<reviewer>.md` (existing v1 reviews) | **Stay in place.** v1 review files are EXPLICITLY UNTRUSTED as authoritative for v2 verdicts (per user directive). v2 reviews land as `docs/reviews/dreamer_srl_v2_cp<N>_<reviewer>.md` (new sibling files). |
| [`docs/experiments/active/dreamer_srl_v3/PARITY_LAUNCH.md`](../../../experiments/active/dreamer_srl_v3/PARITY_LAUNCH.md) (v1 parity FAIL artifact) | **Stays in place.** v2 cross-references it as the failure-mode evidence; v2's re-parity launch artifact lands as a new file under `docs/experiments/active/dreamer_srl_v2/PARITY_LAUNCH.md` (created at v2-CP11 by `experiment-designer` + `experiment-analyzer`, not in this plan's scope). |
| [`configs/dreamer_srl/01_food_only.yaml`](../../../../configs/dreamer_srl/01_food_only.yaml) (corrected XS post-config-correction) | **Stays in place.** v2-CP9 may modify if v2 fixes require a new mandatory config key; if so it is listed in §9 below. |

**No silent rewrites.** Any inline modification to a v1 doc carries an inline
correction note matching the v1 convention from
[`docs/develop/active/dreamer_srl_v3/IMPLEMENTATION_PLAN.md`](../dreamer_srl_v3/IMPLEMENTATION_PLAN.md)
(see the "CORRECTION NOTE (2026-05-14, PI call …)" block at the top of the file).
Originals stay; corrections are additive.

---

## 9. Concrete deliverables

### NEW files v2 will produce

Under [`docs/develop/active/dreamer_srl_v2/`](.):

- **[`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)** — this file.
- **[`SHEEPRL_REFERENCE_AUDIT.md`](SHEEPRL_REFERENCE_AUDIT.md)** — v2-CP1 deliverable; the canonical sheeprl-flow memo that all downstream CPs cite.
- **[`GRAD_PARITY_METHODOLOGY.md`](GRAD_PARITY_METHODOLOGY.md)** — v2-CP2 deliverable; the per-function grad-fixture + threshold spec.
- **[`DEVIATION_LOG.md`](DEVIATION_LOG.md)** — v2's deviation log, starting at D-015, schema identical to v1's. Cross-link to v1 [`DEVIATION_LOG.md`](../dreamer_srl_v3/DEVIATION_LOG.md) for historical context.
- Per-CP spec docs as needed: **`CP3_SPEC.md`**, **`CP4_SPEC.md`**, **`CP5_SPEC.md`**, **`CP6_SPEC.md`**, **`CP7_SPEC.md`**, **`CP8_SPEC.md`**, **`CP9_PLAN.md`**, **`CP10_GATE.md`** — each owned by `senior-developer` ahead of `developer` hand-off, structured per [`docs/TEMPLATES/issue_plan.md`](../../../TEMPLATES/issue_plan.md).

Under [`docs/reviews/`](../../../reviews/):

- For each of v2-CP1 through v2-CP9: per-reviewer files
  - `dreamer_srl_v2_cp1_code_review.md`
  - `dreamer_srl_v2_cp2_math_review.md`, `dreamer_srl_v2_cp2_professor_rl_bayesian_dl_review.md`
  - `dreamer_srl_v2_cp3_code_review.md`, `_math_review.md`, `_professor_rl_bayesian_dl_review.md`
  - `dreamer_srl_v2_cp4_*`, `dreamer_srl_v2_cp5_*` (3 reviews each)
  - `dreamer_srl_v2_cp6_code_review.md`, `_math_review.md`
  - `dreamer_srl_v2_cp7_code_review.md`, `_math_review.md`
  - `dreamer_srl_v2_cp8_code_review.md`
  - `dreamer_srl_v2_cp9_code_review.md`

Under [`tests/algorithms/dreamer_srl/`](../../../../tests/algorithms/dreamer_srl/) (~10–15 new grad tests, owned by `developer` at each CP):

- **`test_loss_grad.py`** — grad-side tests for `compute_actor_objective` (4 tests at v2-CP3), `compute_critic_loss` (3 tests at v2-CP4).
- **`test_train_grad.py`** — grad-side tests for `compute_imagined_returns` (3 tests at v2-CP5), `make_train_step` full-step grad (1 test at v2-CP6), optimizer-step equivalence (2 tests at v2-CP6).
- **`test_dreamer_srl_main_invariants.py`** — driver-level invariant tests (3–4 at v2-CP7).
- **`test_agent_grad.py`** — wrapper-module grad-flow structural tests (~7 at v2-CP8).

Under [`tests/fixtures/dreamer_srl/`](../../../../tests/fixtures/dreamer_srl/):

- New grad-fixture `.npz` files per new test, seed convention `0xD3EAF + 0x100*<cp_id>`.
  Generation script extended at
  [`scripts/fixtures/gen_dreamer_srl_grad_fixtures.py`](../../../../scripts/fixtures/gen_dreamer_srl_grad_fixtures.py)
  (new file at v2-CP2).

Under [`docs/pi/calls/`](../../../pi/calls/):

- **`2026-MM-DD_dreamer_srl_v2_cp10_launch_authorization.md`** (pre-v2-CP10 PI consult, dated at consult time).
- **`2026-MM-DD_dreamer_srl_v2_cp11_launch_authorization.md`** (pre-v2-CP11 PI consult, dated at consult time).
- **`2026-MM-DD_dreamer_srl_v2_cp12_verdict.md`** (post-v2-CP11 PI consult if v2-CP11 FAILs; dated at consult time).

### v1 files v2 may modify

- **[`scripts/sheeprl_jax_diff.py`](../../../../scripts/sheeprl_jax_diff.py)** — extended at v2-CP2 with the `register_grad_diff(...)` helper. **Inline correction note required** at the function-table comment block matching the v1 "originals stay, corrections are additive" convention.
- **[`src/algorithms/dreamer_srl/*.py`](../../../../src/algorithms/dreamer_srl/)** — modified at v2-CP9 only, per-fix named commits, one fix per commit; no batched touches. **Each modified line carries an inline citation note** pointing at the v2 review file that authorized the fix.
- **[`configs/dreamer_srl/01_food_only.yaml`](../../../../configs/dreamer_srl/01_food_only.yaml)** — may be modified at v2-CP9 only if a v2 fix requires a new mandatory config key. **All new keys are listed inline here as a v2-CP9 update.** No fallback defaults: every new key must use `config.get_mandatory('<key>')` per the project-wide rule.

### v1 files v2 will NOT modify

- v1 reviews under `docs/reviews/dreamer_srl_v3_*` — untrusted but preserved.
- v1 plan docs under `docs/develop/active/dreamer_srl_v3/*` — preserved unchanged.
- v1 deviation log — preserved; v2 deviations append to v2's own log.

---

## Checkpoints (this plan's authoring + diary handoff)

- [x] Plan structure complete (Sections 1–9 authored).
- [ ] Diary `planned` event logged via `scripts/diary_append.py` linking to this plan.
- [ ] `docs/develop/INDEX.md` regenerated via `scripts/regen_dev_index.py`.
- [ ] Plan + diary committed as one named commit.

## Implementation Report

> **Implemented by**: (this plan is the implementation spec; v2-CP execution is the developer / training-runner / experiment-analyzer hand-off chain in §5)
> **Date**: 2026-05-14 (plan authored)

## Verification Report

> **Verified by**: senior-developer (this plan's authoring); per-CP verification chains
> see §4.
> **Date**: 2026-05-14 (plan published)

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| [`docs/develop/active/dreamer_srl_v2/IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) | New master plan | ✅ | This file. |

**Conclusion**: v2 master plan published. v2-CP1 (fresh sheeprl re-read) is the next
authorized agent action.
