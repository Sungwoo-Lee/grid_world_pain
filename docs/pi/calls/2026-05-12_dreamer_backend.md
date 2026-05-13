---
title: "PI Call — Dreamer backend: in-house JAX rebuild vs. sheeprl-direct"
date: 2026-05-12
trigger: Pre-launch / roadmap-level — dreamer-srl v2 plan was 3-reviewer-approved and ready to spawn `developer`, but the user surfaced a research-velocity concern before authorizing the build.
status: decided
---

# PI Call — Dreamer backend: in-house JAX rebuild vs. sheeprl-direct

## Question

**Do we spend the next 2–6 weeks rebuilding a paper-fidelity DreamerV3 in JAX/Flax inside our own repo, or do we drop that plan and use the existing community PyTorch implementation (`sheeprl`) as the Dreamer backend for the upcoming neuromodulation paper?**

## Headline

**The user picked Option 1: sheeprl-direct (minimal bridge).** The 1,033-line `dreamer-srl` v2 plan — a JAX re-implementation of sheeprl's DreamerV3 with three reviewer companion files all marked PASS — is shelved. Instead, we build the smallest bridge needed (env adapter, WandB hook, training-runner sheeprl-mode) to run sheeprl on our 5×5 food-only grid environment, and port the project's neuromodulation hooks (the modulator that adjusts the world-model's confidence, the FiLM conditioning blocks, the precision-modulation knobs) into sheeprl's PyTorch agent when the next experiment goes. This trades single-stack JAX uniformity for research velocity. The user did not preserve the v2 plan as a deferred path (Option 4) — they archived it outright, which is itself a signal: confidence in the sheeprl-direct call, not a hedge.

## Context for a fresh reader

The project's in-house DreamerV3 (a JAX/Flax world-model RL agent we built) has been failing to learn a very simple survival task — a 5×5 grid where the only thing the agent has to do is find food before starving. It survives ~106 environment steps. The community PyTorch implementation `sheeprl`, run unmodified on the *same* environment with the *same* observation and reward function, survives the full 500-step time limit. That's a 5× gap on a task that should be trivial. The diagnosis run that established the gap is documented in [`docs/develop/active/diagnosis/sheeprl_drop_in_test.md`](../../develop/active/diagnosis/sheeprl_drop_in_test.md).

The project spent the last week debugging that gap one fix at a time — a five-step cascade comparing our code line-by-line against sheeprl. Two fixes landed (paper-canonical two-hot reward bins, zero-init reward and critic output heads); three remained pending (GRU reset gate, critic slow-target self-EMA loss term, RSSM hidden-layer count). The cumulative risk after the cascade-walk was high: even with all five fixes shipped, latent divergences elsewhere could still leave us short of sheeprl's regime.

That risk produced the `dreamer-srl` plan — stop walking the cascade, re-implement sheeprl's DreamerV3 in JAX/Flax verbatim as a new module ([`docs/develop/active/dreamer_srl/IMPLEMENTATION_PLAN.md`](../../develop/active/dreamer_srl/IMPLEMENTATION_PLAN.md)). The v2 plan resolved 29 deviations and earned PASS reviews from three reviewers (the RL/Bayesian-DL professor, math reviewer, code reviewer). It was ready to hand off to `developer` for a 2–6 week build.

The user paused at the handoff and asked the strategic question: *is rebuilding the right move at all, given that sheeprl-the-thing-we-are-copying already works on our env?*

## Options considered

1. **Sheeprl-direct (minimal bridge).** Drop the `dreamer-srl` rebuild. Use sheeprl (PyTorch + Lightning Fabric) as the Dreamer backend. Build only what's needed for the next experiment — env adapter, WandB hook, sheeprl-mode for the training-runner. Port the neuromodulation hooks into sheeprl's PyTorch agent when the experiment goes. *Cost:* mixed-stack codebase (JAX env + PyTorch agent). *Buys:* fastest time-to-first-experiment, leverages an implementation that already passed parity on our env.
2. **Sheeprl + paved JAX-env bridge.** Same as Option 1 but invest one-time engineering on a clean numpy↔JAX env adapter, a full `run_command.py` sheeprl-mode, WandB conventions matched, configs/ mirror. ~1 week before first experiment. Cleaner long-term hybrid.
3. **Execute the `dreamer-srl` plan as written.** Build the JAX rebuild over 2–6 weeks behind the parity gate (mean survival ≥ 500 steps on food-only NoPred, 3 seeds, within 2× sheeprl wall-clock). Preserves single-stack JAX. *Cost:* multi-week build before any neuromodulation experiment runs; reviewer-passed but the same review process did not catch the integration-layer issues in the original Dreamer either.
4. **Hybrid / deferred-execution.** Pick Option 1 or 2 for the immediate paper. Archive the v2 plan as `superseded-pending-revisit` with cross-link; revisit only on a concrete trigger (TPU access, throughput bottleneck, second paper). Keeps the JAX rebuild as a future asset.

## Earlier PI's framing (carried forward, not overridden)

The earlier PI session noted: the multi-month cascade-debugging history is strong evidence that the JAX Dreamer is failing at an integration layer that static review can't surface. The same three reviewers who signed off PASS on the v2 plan are the same people who reviewed the cascade fixes one at a time and did not catch the integration-layer issue then either. That favored Options 1 or 4 over 3.

The earlier PI also flagged the load-bearing fact: **the paper does not care which language the Dreamer is in.** The neuromodulation story is the publication; the Dreamer backend is instrumentation.

## User decision

**Option 1 — Sheeprl-direct (minimal bridge).**

Verbatim rationale:

> "I have spent more time to debug and solve the implementation issue. So, now I more prefer to use sheeprl version of dreamer to my project."

The user is a single PhD researcher targeting a publishable paper, not a lab maintaining a long-lived platform. Research velocity is the binding constraint. The user did not pick Option 4 (deferred-execution / preserve plan as future asset) — they picked the more decisive Option 1 outright. That is itself worth noting: the user is willing to drop the 1,033-line v2 plan + 6 reviewer companion files (all PASS) without preserving them as a deferred path. Confidence in the sheeprl-direct call, not a hedge.

## Rationale captured

- **Velocity over codebase uniformity.** A mixed-stack codebase (JAX env + PyTorch agent) is acceptable cost for cutting weeks off the time-to-first-experiment for the neuromodulation paper.
- **Working implementation beats reviewed-but-unbuilt plan.** Sheeprl already passed the parity gate on our environment (the diagnosis run survives the full 500-step time limit). The `dreamer-srl` plan would have to *reach* that same gate over 2–6 weeks of build time.
- **Reviewer-PASS is not a parity guarantee.** The three reviewers who signed off on the v2 plan are the same reviewers who signed off on the cascade fixes one at a time, and the cascade did not close the 5× gap. The user implicitly weighted this evidence.
- **The paper is the deliverable, not the platform.** The neuromodulation/pain-modeling story is what gets published; the Dreamer backend is instrumentation.

## What this implies operationally

1. **The `dreamer-srl` plan and all 6 reviewer companion files are shelved.** The folder `docs/develop/active/dreamer_srl/` (7 files: `IMPLEMENTATION_PLAN.md`, `review_code.md`, `review_code_v2.md`, `review_math.md`, `review_math_v2.md`, `review_professor_rl_bayesian_dl.md`, `review_professor_rl_bayesian_dl_v2.md`) needs to be `git mv`'d to `docs/develop/archive/dreamer_srl/`, with each file's frontmatter updated from `status: active` / `status: draft` → `status: superseded` and a `superseded_by:` field pointing back to this call log. `senior-developer` owns this — they own `docs/develop/`.
2. **A new minimum-bridge plan needs to be drafted.** Scope per Option 1 — env adapter (gymnasium bridge wrapping our `jax_reset` / `jax_step` / `get_observation`), WandB hook (a `wandb.yaml` for sheeprl's Hydra logger selector), training-runner sheeprl-mode (the runner needs to know how to launch sheeprl jobs on node 114), and a plan for porting NMN / FiLM / precision-modulation hooks into sheeprl's PyTorch agent when the next experiment authorizes them. `senior-developer` writes this plan.
3. **The cascade-fix work is paused.** The three remaining fixes (GRU reset gate, critic self-EMA, RSSM hidden layers) are not being shipped to our in-house JAX Dreamer. The in-house JAX Dreamer stays as-is — neither extended nor deleted — until a future call decides its fate.
4. **The portfolio is still unratified.** [`docs/pi/PORTFOLIO.md`](../PORTFOLIO.md) has TBD placeholders for the active publication tracks. A separate PI call to name the tracks (the neuromodulation/pain-modeling paper is the obvious candidate for Track A) should land soon — this dreamer-backend call is structurally illegible until the portfolio anchors it.

## Hand-off

- **Next agent:** `senior-developer`
- **Concrete next step (two parallelizable tasks):**
  - **(a) Archive task.** `git mv docs/develop/active/dreamer_srl/ docs/develop/archive/dreamer_srl/`; update each of the 7 files' frontmatter (`status: superseded`, add `superseded_by: docs/pi/calls/2026-05-12_dreamer_backend.md`, bump `last_updated: 2026-05-12`); regenerate the develop INDEX via `scripts/regen_dev_index.py`.
  - **(b) New plan task.** Draft `docs/develop/active/sheeprl_bridge/IMPLEMENTATION_PLAN.md` for the minimum-bridge integration (env adapter, WandB hook, training-runner sheeprl-mode, deferred NMN/FiLM port). Use [`docs/develop/active/diagnosis/sheeprl_drop_in_test.md`](../../develop/active/diagnosis/sheeprl_drop_in_test.md) as the technical starting point — it already documents the env interface, sheeprl's contract, the `sheeprl_bridge` conda env design, and the node 114 caveats.
- **Stop rule:** If `senior-developer` finds during plan-drafting that the NMN/FiLM port into PyTorch is materially harder than expected (e.g., the modulator design has JAX-specific assumptions that don't translate, or the FiLM hook lives somewhere sheeprl's agent doesn't expose), escalate back to PI before sinking implementation time. The call to switch backends rested on "porting the hooks is straightforward"; if that premise breaks, the call should be revisited.

## Links

- Earlier PI session that produced the four candidate paths: (this session's parent — no separate log; this file IS the log)
- Shelved plan: [`docs/develop/active/dreamer_srl/IMPLEMENTATION_PLAN.md`](../../develop/active/dreamer_srl/IMPLEMENTATION_PLAN.md) (will move to `archive/` per hand-off task (a))
- 5×-gap evidence: [`docs/develop/active/diagnosis/sheeprl_drop_in_test.md`](../../develop/active/diagnosis/sheeprl_drop_in_test.md)
- Portfolio (unratified): [`docs/pi/PORTFOLIO.md`](../PORTFOLIO.md)
