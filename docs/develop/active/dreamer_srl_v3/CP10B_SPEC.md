---
title: "dreamer-srl v3 — CP10b spec: XS like-for-like wall-clock comparison vs sheeprl 12.5 h baseline (post-D-013 disposition)"
topic: dreamer
status: active
created: 2026-05-14
last_updated: 2026-05-14
phase: 2
---

# CP10b — XS like-for-like wall-clock comparison (post-D-013 disposition)

## Purpose

CP10 closed at the **reduced-dim wall-clock baseline** (9.50 SPS steady-state on a single RTX 4090 at 256 dense units / 8×8 stochastic state / horizon=7) because the **full XS configuration** that sheeprl's reference 12.5-hour baseline used (1024 dense units / 32×32 stochastic / horizon=15) OOMs on a single RTX 4090 during JIT compilation of the training step (peak VRAM demand 14.38 GB, exceeds 24 GB headroom after JAX's 90% pre-allocation policy). The OOM is documented as deviation **D-013** in [DEVIATION_LOG.md](DEVIATION_LOG.md); its disposition (multi-GPU launch vs gradient checkpointing vs intermediate config) is a portfolio-level "what config does the parity launch run at?" question that the **parity-launch PI consultation** is chartered to resolve via [pi.md](../../../../.claude/agents/pi.md)'s "pre-launch of a multi-run experiment" trigger.

CP10b is the structured spec for **converting the CP10 proxy projection into a real like-for-like measurement** of the dreamer-srl JAX rebuild against sheeprl's XS reference, *after* D-013 is dispositioned. It does NOT execute now; it waits on the PI consultation. Once the parity-launch PI consultation chooses a substrate disposition for D-013, CP10b runs the same protocol CP10 ran (a wall-clock-budget measurement) but on the disposed configuration, and reports a like-for-like 12.5h-baseline comparison.

## Question (plain language)

**Does the JAX dreamer-srl rebuild fit inside the 2× wall-clock budget (≤ 25 h per seed) on the configuration the parity launch will actually run at?**

Sheeprl's reference 12.5-hour 200,000-step run (WandB [`jzgkcep4`](https://wandb.ai/sungwoolee/grid_world_pain_dreamer_srl_smoke/runs/jzgkcep4), 4.43 env-SPS, full XS recipe — see [SPS_COMPARISON_JAX_VS_SHEEPRL.md §3.1](../../../experiments/active/sheeprl_bridge/SPS_COMPARISON_JAX_VS_SHEEPRL.md#31-end-to-end-environment-steps-per-second)) sets the comparison anchor. The v3-plan's CP10 row promised a like-for-like measurement at the same configuration; D-013's OOM prevented that at CP10 on a single GPU. CP10b runs the like-for-like measurement on whatever configuration the PI consultation chooses for the parity launch.

## When CP10b runs

CP10b is **blocked** on these gates, in order:

1. **D-013 disposition** at the parity-launch PI consultation. The PI surfaces 2–4 candidate paths via `AskUserQuestion`; the user decides; the PI logs the call under `docs/pi/calls/`.
2. **Implementation** (if any) of the disposed config. If the disposition is "multi-GPU launch", an implementation plan + developer agent runs first to wire data-parallel scaling into the JAX driver. If "gradient checkpointing", an implementation plan + developer agent runs first to wire `jax.checkpoint` into the world-model and imagination rollouts. If "intermediate config" (e.g. 512 dense / 16×16 stoch / horizon=10 that fits single-GPU but is closer to XS than the CP10 reduced-dim baseline), no implementation is needed — just a config edit.
3. **Senior-developer authorization** to run CP10b. Authorization is granted after gates 1+2 close.

CP10b does NOT block the parity launch. The parity launch can run alongside CP10b; CP10b's purpose is the **published wall-clock measurement**, not a gating decision.

## What CP10b measures

A 20,000-step (or, by user election, 200,000-step at-target) dreamer-srl smoke on the **disposed configuration**, on the same hardware substrate the parity launch will use, with the same `learning_starts: 1024` setting that the parity-track config carries. The protocol is identical to CP10's:

1. Launch the run with WandB logging enabled, name = `dreamer_srl_cp10b_<disposition>_<step_budget>_s<seed>`.
2. Capture: aggregate env-SPS (`Time/sps_env` summary), steady-state env-SPS (mean of inter-log Δstep/Δruntime across post-JIT windows), peak VRAM, NaN count across the 7 Loss/* keys, WM-loss drop step-200 → final, `Diagnostic/moments_invscale` min/max/final, `Params/replay_ratio` convergence to sheeprl-spec, and **the iter-1024 debt-repayment burst** (per CP9b D-014 + professor's F3 forward-looking note: SPS trace at iters `[1, 1024, 1025, 1026, 2048]` to confirm the debt is paid in one burst and steady-state SPS resumes at iter 1025+).
3. Compute the projected per-seed parity-launch wall-clock: `200,000 / steady_state_SPS`.
4. Compare to sheeprl's 12.5h baseline at 4.43 env-SPS; verdict at the 2× budget gate (≤ 25 h).

## Forward-looking diagnostics inherited from CP9b

Per [CP9b verification](IMPLEMENTATION_PLAN.md#cp9b---s3-random-action-prefill), two CP9b professor's forward-looking findings carry into CP10b's scope:

- **F1** — Add a one-line WandB log key recording the sampled action during the prefill window (one int per iter for iter `<= learning_starts`), so the parity-launch run produces a visible cross-check that the §S3 branch is wired. The post-cascade-fix-#27 zero-init actor produces approximately uniform actions on its own for the first ~10 gradient steps, which is a bounded but real failure-mode overlap with CP9b's Test 1 empirical-entropy criterion.
- **F3** — The iter-`learning_starts` debt-repayment burst should be explicitly characterised so it is not mistaken for a hang during the parity launch. With `learning_starts=1024` and `replay_ratio=1` on the full XS config, the burst at iter 1024 is approximately 1024 invocations of the JIT-compiled `train_step` in a tight Python loop, estimated at 700-1024 seconds of wall-clock on the full XS configuration. This is not a stability concern (each individual `train_step` is identical to a steady-state one; only cumulative GPU memory pressure could be an issue, and the JIT-compiled trace runs at constant memory). The CP10b run **should observe and log the burst explicitly**.

Both F1 and F3 are non-blockers for CP10b's CP-PASS — they are diagnostic additions for clarity at the parity launch.

## What CP10b does NOT do

- CP10b is **NOT** a parity-result check. The parity launch (task #11 in implementation order) is what measures "does the JAX dreamer-srl actually learn the task to the threshold survival of ≥ 480 steps on food-only NoPred?". CP10b only measures wall-clock; the parity launch measures both wall-clock and survival.
- CP10b is **NOT** a substitute for D-013's resolution. D-013 closes when the PI consultation disposes it; CP10b just runs whatever config that disposition produces.
- CP10b is **NOT** a code change. No `src/`, `configs/`, or `scripts/` modifications happen at CP10b (with the exception of any config-only edit that the PI's disposition mandates, which would be authored by `developer` under a separate implementation plan).

## Acceptance criteria for CP10b CP-PASS

1. **Run completes**: exit code 0, no crash, no OOM at the disposed config on the disposed hardware.
2. **Steady-state SPS measured** at the disposed config: WandB run published, inter-log Δstep/Δruntime mean computed across post-JIT windows.
3. **Like-for-like comparison**: projected per-seed parity-launch wall-clock (`200,000 / steady_state_SPS`) recorded; compared against sheeprl's 12.5h baseline.
4. **Speed verdict**: ✅ inside 25 h budget (= 2× sheeprl's 12.5h) ⇒ parity launch can proceed at this configuration; ⚠ outside 25 h but inside 30 h ⇒ user discusses with PI whether budget extension is acceptable; ❌ outside 30 h ⇒ disposition revisit (re-fire PI consultation).
5. **Training-loop health**: zero NaN, WM-loss drop ≥ 20% step-200 → final, `moments_invscale` ≥ 1.0 throughout, `replay_ratio` converges to within 1% of sheeprl-spec target.
6. **CP9b forward-looking diagnostics surfaced** (F1 action-log + F3 debt-repayment-burst trace), both informationally.

## Out of scope for CP10b

- Substrate-faithful Lever-A bit-identity testing — these closed at CP1 through CP9b on the per-function basis and are not re-tested at runtime.
- Algorithmic-correctness verification — the offline-check fixture suite (CP8) is the gate for that, and CP9 + CP9b already verified the integration surface on a live run.
- Hyperparameter tuning — the disposed config is what the parity launch runs at, not a tuning target.

## CP10b checkpoint-table entry (when CP10b actually runs)

When CP10b runs, the IMPLEMENTATION_PLAN.md checkpoint table gets a new row inserted between CP10 (line 528) and the parity-launch entry:

```markdown
| **CP10b** | XS like-for-like wall-clock comparison (post-D-013 disposition) — see [CP10B_SPEC.md](CP10B_SPEC.md) | No new tests; measured under speed-check protocol on the D-013-disposed configuration | (no reviewer chain; senior-developer judges per the standard ≤ 5%/≤ 15% rule) | (no new deviations at CP10b unless the disposition introduces one) | NOT STARTED — gated on D-013 disposition + any disposition-implementation plan |
```

And the verification table row (line 1236 area) gets a similar entry.

## Notes for the senior-developer who runs CP10b

- The PI consultation will likely take the form of an `AskUserQuestion` from `pi` listing 2–4 candidate dispositions. The user picks one. The PI logs the call.
- After the PI consultation closes, the senior-developer:
  1. Reads the PI call doc under `docs/pi/calls/<date>_dreamer_srl_v3_parity_launch_disposition.md`.
  2. If the disposition needs code work (multi-GPU wiring, gradient checkpointing wiring), writes an implementation plan under `docs/develop/active/dreamer_srl_v3/<plan_name>.md`, hands off to `developer`, verifies the implementation.
  3. Authorizes CP10b under the disposed configuration.
  4. Runs the CP10b protocol exactly as CP10 ran (single-node, single-or-multi-GPU per the disposition, with WandB logging).
  5. Writes a CP10b verification report subsection in IMPLEMENTATION_PLAN.md mirroring CP10's structure.
  6. Flips CP10b's row in the checkpoint table + verification table to ✅ CP-PASS (or ❌ if the run lands outside the 30 h band).

## Links

- CP10 closure (the reduced-dim baseline measurement): [IMPLEMENTATION_PLAN.md §CP10 — wall-clock budget](IMPLEMENTATION_PLAN.md#cp10--wall-clock-budget-reduced-dim-baseline-xs-comparison-deferred-to-cp10b)
- D-013 (XS-on-single-GPU OOM): [DEVIATION_LOG.md D-013](DEVIATION_LOG.md#deviation-table)
- Sheeprl baseline: [SPS_COMPARISON_JAX_VS_SHEEPRL.md §3.1](../../../experiments/active/sheeprl_bridge/SPS_COMPARISON_JAX_VS_SHEEPRL.md#31-end-to-end-environment-steps-per-second), WandB [`jzgkcep4`](https://wandb.ai/sungwoolee/grid_world_pain_dreamer_srl_smoke/runs/jzgkcep4)
- PI consultation trigger spec: [pi.md](../../../../.claude/agents/pi.md)
- CP9b D-014 forward-looking notes for F1+F3 diagnostics: [IMPLEMENTATION_PLAN.md §CP9b](IMPLEMENTATION_PLAN.md#cp9b---s3-random-action-prefill)
