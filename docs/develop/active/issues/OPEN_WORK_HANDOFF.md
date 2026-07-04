---
title: "Open-Work Handoff — remaining v3.0-audit work"
topic: issues
status: active
created: 2026-07-04
last_updated: 2026-07-04
---

# Open-Work Handoff — remaining v3.0-audit work

## What this is

A **handoff checklist** of the work left over from the 2026-07-04 v3.0-audit
session, written so another Claude session can pick it up cold. Each item is a
one-line, actionable task with the file refs and links you need — **not**
re-analysis. For the full bug record and root-cause detail, see [[KNOWN_BUGS]]
(owned by `bug-curator`) and the [[v3_pipeline_correctness_diagnosis]].

Items are grouped by readiness:
- **A** — ready to implement, no user decision needed.
- **B** — needs a user decision before any code is written.
- **C** — latent / unconfirmed; triage before touching.
- **D** — experiment-ops / user-driven, not code.

Do the routing the normal way: write a senior-developer fix plan per item, hand
to `developer`, verify. Ask `bug-curator` to update the registry after a fix
lands — do not hand-edit [[KNOWN_BUGS]].

---

## A. Ready to implement (no user decision needed)

- [ ] **A1 — Regenerate 4 stale `observability_gates` parity fixtures.** `tests/env/test_unified_parity.py::observability_gates_S1-S4` are RED because the G2 fix (commit `84014e4`) changed those configs' start position, but the golden `.npz` fixtures were captured with the old one. The parity test is already migrated to `animal_*` keys (F fix, `0bebe06`), so regenerating these 4 fixtures against the new fixed start is safe. Goal: suite fully green. **Small.**
- [ ] **A2 — dreamer-srl checkpoint omits optimizer state.** `src/algorithms/dreamer_srl/checkpoint.py:85-88` saves only `world_model/actor/critic/target_critic` params via `nnx.state(.., nnx.Param)` — **no** `opt_state`. Effect: resuming/continuing a run silently restarts Adam momentum from zero (matters for continual learning). Needs a small fix plan → `developer`. Ref: memory `docs/memory/memories/env_entities/20260609_1726_doc_audit_surfaces_latent_bugs.md`. **Med.**
- [ ] **A3 — "chasing rabbit" rides the agent's cell after contact.** `src/environment/core.py:629` applies the post-contact pause only `where(at_damaging, ...)`, so a non-damaging animal gets no pause and stays at distance 0 — skews the chasing-rabbit / hypervigilance behaviour read. Needs a fix plan → `developer`. Ref: memory `docs/memory/memories/env_entities/20260609_1722_renderer_no_neutral_icon_and_attack_delay_ride.md`. **Med.**

## B. Needs a user decision first (do NOT implement blind)

- [ ] **B1 — M1/M2 behaviour-metric "episode-end rule".** Three conflicting documented semantics for events still in progress at episode end (finish them / drop them / exclude from denominator). One rule must be adjudicated before coding. Refs: [[v3_pipeline_correctness_diagnosis]] Finding M1/M2; `docs/reviews/diag_v3_pipeline_math.md`; `docs/experiments/active/behavior_measures/behavior_measure_toolkit_v1_design.md` §7.2; `docs/develop/active/behavior/behavior_measure_toolkit_v1_plan.md` line 382. *(Measures are OFF by default now → low urgency.)*
- [ ] **B2 — CLI model-size overrides not saved to config (L4).** `--hidden_size` / `--num_steps` / `--lr` aren't written into the saved config → re-eval rebuilds the model at the wrong size. The `--no-satiation` sibling was already fixed (`75976e2`). **Low** — decide whether it's worth doing.

## C. Latent / needs verification (triage before any fix)

- [ ] **C1 — Checkpoint Orbax↔NNX restore skew.** Recorded risk, never confirmed. Triage whether it's real. Ref: memory `docs/memory/memories/cluster_ops/20260509_1536_train_py_checkpoint_restore_nnx_skew.md`.
- [ ] **C2 — Env doc-audit: ~11 remaining sub-findings un-triaged** (beyond the 2 named in A2/A3 above — e.g. overeating-death never ends the episode; unreliable `termination_reason` when a body system is off). Triage which are real vs cosmetic. Ref: memory `docs/memory/memories/env_entities/20260609_1726_doc_audit_surfaces_latent_bugs.md`.

## D. Experiment ops (user-driven, not code)

- [ ] **D1 — Relaunch jump-reach (node 113)** with `attack_range: [2,3]` on corrected code (reward fix `ef0fd25` + range fix `7ff8d1f`) for clean {2,3} semantics.
- [ ] **D2 — Relaunch the 6-run basic ladder** on the corrected reward + range semantics when desired (current runs carry the old semantics).
- [ ] **D3 — 266 commits are unpushed** — decide whether to push.

---

## Reference block

**Key commits this session:** reward `ef0fd25`, GAE `3c60f6f`, dreamer continue-head
`5b093bf`, eval E `a3ab4cc` / A+L3 `2ad9104` / L2 `863052f`, parity F `0bebe06`,
measures-off `9eacf82`, true-obs recover `80d3b70`, range fix `7ff8d1f`.

**Full record:** [[KNOWN_BUGS]] · **Diagnosis:** [[v3_pipeline_correctness_diagnosis]]
