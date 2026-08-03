---
title: "Plan review (round 3) — dreamer_srl → train.py integration, rev-4 reconciliation delta"
topic: dreamer
status: active
created: 2026-08-03
last_updated: 2026-08-03
---

# Round-3 review — dreamer_srl → train.py integration plan, rev 4

## Verdict

**NOT READY** — one Critical finding in the newly added resume-parity gate; everything else in the rev-4 delta is Moderate or below.

Plain-language summary: rev 4 of the plan added a third proof gate ("Gate 1c") that is supposed to show that resuming a Dreamer training run from a saved checkpoint behaves identically whether launched through the old entry point or the new unified one. As written, that gate can pass **without running any training at all**: the training loop treats the episode budget as an *absolute total*, not an increment, so "resume at episode 20 with a further 20-episode budget" makes the loop condition (`total_episodes_completed < episodes`, i.e. `20 < 20`) false on entry and the run exits immediately. Both entry points would produce identical *empty* telemetry, every pass criterion (array equality, counter equality) would be satisfied vacuously, and the gate would go green while proving nothing about the resume path — the exact feature the user just confirmed as in-scope because curriculum/continual learning is a priority. The fix is one line of spec: pin the resumed budget to an absolute value past the checkpoint (e.g. `--episodes 40`) and add a pass criterion that the resumed segment executed at least one gradient-taking iteration (non-empty loss arrays).

Severity legend: 🔴 Critical = fix before going further · 🟡 Moderate = likely costs a re-run · 🟢 Low = cosmetic · ❓ Open = an assumption nobody has verified yet.

## Findings

| # | Sev | Location | Issue | Suggested fix |
|---|---|---|---|---|
| 1 | 🔴 | Plan §"How captured — Gate 1c" | Episode budget is **absolute**: the loop runs `while total_episodes_completed < episodes` (dreamer_srl_main.py:1606) and §12b restores `total_episodes_completed` (:1388). "`--load-episode 20` … for a further 20-episode budget" naturally implements as `--episodes 20` → `20 < 20` → instant exit, zero iterations, empty telemetry on both sides. Empty == empty satisfies pass criteria 1–2 → **vacuous green gate** on the resume leg. The claimed coverage ("exercises the restore path … the `train_start_iter` buffer-refill gate") is then false — the refill gate never fires because the loop never runs. | State the absolute-budget semantics in the spec; pin `--episodes 40`; add to the 1c pass criteria: telemetry arrays non-empty AND ≥1 iteration with `last_losses` updated after `train_start_iter` (this also makes a missing-Adam-restore divergence observable — Adam state is never directly checksummed and only surfaces through post-resume parameter updates). |
| 2 | 🟡 | Same section, closing parenthetical | Curriculum resume is never end-to-end gated through train.py. Gate 1c uses the single-config 1a fixture, so §12c stage-rebuild-on-resume (`if args.load_checkpoint and schedule is not None`, :1437) never fires in **any** gate; the retargeted `test_continual_resume_rebuild.py` drives the seam directly, bypassing train.py's schedule construction + resume-flag plumbing *in combination*. The 1b×1c composition argument leaves that interaction untested — and curriculum crash-relaunch is the priority use case that put resume in scope. | Add a cheap 1c variant on the 1b fixture: resume into stage 2 (e.g. `--load-episode 40`, `--episodes 60`, reusing the completed 1b run dir) through both entry points. |
| 3 | 🟡 | §"Live-run constraint" vs Phase-1 authorization | Internal tension: the constraint says behavior-affecting changes to files the live run launched from are "held conservatively", yet Phase 1 rewrites `dreamer_srl_main.py:main()` and extracts the body — the exact file a crash-relaunch (with `--load-checkpoint`, through §12b/12c) would execute. No gate covers refactored-legacy vs **pre-refactor** legacy (Gate 1 compares two refactored sides); equivalence at relaunch rests only on Checkpoint 1 (test suite + smoke) and the byte-diff review. | Either defer landing Phase 1's `dreamer_srl_main.py` edit until the live run completes, or state explicitly that crash-relaunch-on-refactored-code is accepted with Checkpoint 1 as the sole guard. |
| 4 | 🟢 | Risk R7; Verification checklist bullet 2 | Stale anchors survived the rev-4 renumber: R7 cites the wandb label edits at "lines 829/865" (rev-2 numbering; File Change 1 correctly says :944/:980), and the checklist says "line 577 retained" for the budget deletion (correct: :593). | Update the two citations. |
| 5 | 🟢 | Rev-4 header; Binding-decision status; Analysis 8 | Commit misattribution: the resume flags landed in `166f261`, not `e834ec1` (which added only §12c + the RollingWindow change); the `testing.seed` eval-seed switch is `b228117`, not "ad8929a's commit region". Two further commits in the window (`10000af` async checkpoint-video render, `0030c04` hierarchical encoder) touched the file and go unmentioned — verified: neither adds CLI flags or new below-seam `args.*` reads, so no spec impact, but "reconciled with four commits" under-describes the window. | Cosmetic history fix; no plan-substance change needed. |

## What was verified clean (rev-4 delta)

- **Anchors** — every renumbered anchor spot-checked against HEAD and correct: dreamer_srl_main.py resume flags :497-506, pre-seam locals :514-515, waist :574, num_envs :583, `derive_prefill` :588, `env_max_steps` :593 (stays), deletion range :594-618 (WARNING :605-613, `total_steps` alias :617-618), retention guard :667-705, `eval_seed`/`testing.seed` :712, seed sites exactly 4 (:733, :734, :956, :1585 — :707 is a comment), wandb labels :944/:980, §12b :1369+, §12c :1437+; train.py signals :449-450, peek :508, propagation loop :556-561, stage-0 alias :564, agent merge :589, algorithm read :650, probe gate :669.
- **Gate 1c comparison set is sufficient once finding 1 is fixed**: a one-sided missing `moments` restore is caught directly (moments checksum in the dump); a Ratio/grad-accounting divergence surfaces via per-iteration `cumulative_grad_steps`/`n_grad_steps`; replay-buffer contents after refill are not dumped but are covered *indirectly* — identical restored policy + identical PRNG stream on the CPU backend makes refill deterministic, and any content divergence perturbs sampled batches → loss arrays diverge. All of this **requires the resumed segment to actually train** — which is finding 1. Note Gate 1c is a *parity* gate: a restore bug introduced by the seam extraction itself hits both sides identically and passes; correctness there rests on the byte-diff checklist plus the retargeted resume regression tests, which the risk register (R12) scopes correctly.
- **Config-freeze list is complete for the steps in the plan**: no Phase 1/Gate 1 step touches `configs/train/default.yaml`, `evaluation/default.yaml`, the `fb54bc0` schedule/stage configs, or the stage configs' `extends:` parents; fixtures live under `tests/`; the 19-file rename is deferred; the harness uses a temp agent-config copy (Checkpoint 6). Gate 2 node avoidance is concrete enough (gpu-status at validation time; live node identifiable from the diary training-start row).
- **Gate 1b fixture** correctly mirrors the real schedule shape ([2000, 5000, 5000] verified in `configs/continual/basic_01_02_03_dreamer.yaml`; cadence 5000 / keep-ALL 1000000 verified in `configs/train/dreamer_srl.yaml`). The `ad8929a` retention guard *executes* in 1b (schedule non-None) but its warn branch is likely dormant and its console output is not parity-compared — acceptable: the guard is warn-only and pinned by its own unit test (`test_curriculum_retention_guard.py`).
- **Compat table / File Changes / gate specs are mutually consistent** on resume (C13 ↔ C20 ↔ spec fields ↔ shim forwarding); no other accretion contradictions found beyond finding 4.

## Cost of being wrong

If finding 1 ships as written, the resume leg of the integration is certified by a gate that proved nothing; the first real curriculum crash-relaunch through train.py that hits a resume-plumbing divergence would restore or budget wrongly and burn days of GPU time on a run whose continual-learning comparison is invalid. No data-loss hazard found in the plan.

## Exit condition

Verdict flips to SOUND WITH CONCERNS once Gate 1c's spec pins an absolute post-checkpoint budget and a non-vacuity pass criterion (finding 1); findings 2–3 are strongly recommended but user-acceptable risks.

Reviewed by: plan-reviewer (round 3, rev-4 delta), 2026-08-03
