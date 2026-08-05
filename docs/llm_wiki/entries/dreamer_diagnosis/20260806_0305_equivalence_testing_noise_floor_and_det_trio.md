---
id: 20260806_0305_equivalence_testing_noise_floor_and_det_trio
date: 2026-08-06
time: "03:05"
folder: dreamer_diagnosis
tags: [dreamer, design, learned_lesson, meta]
summary: "Reusable recipe for proving a refactor did not change a working training system: audit and seed every entropy source first, measure the noise floor by running the same side twice, and settle the residual with a deterministic trio (A, A-repeat, B) that upgrades the claim from 'indistinguishable' to 'the same computation'."
related: ["20260806_0304_dreamer_sampler_seeding_full_determinism_recipe"]
relations: ["depends_on:20260806_0304_dreamer_sampler_seeding_full_determinism_recipe"]
session_origin: claude_code
session_label: "dreamer-integration Gate 2 — determinism audit + fix + deterministic leg"
importance: high
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/87d7d010-fa7d-466e-9113-1a7618f94d95.jsonl
raw_completeness: full
---

# How to prove a refactor did not change a working training system: seed everything, measure the noise floor, then close it with a deterministic trio

## Key conclusion

When a refactor moves a training system to a new entry point (or any change that is supposed to be behaviour-neutral), "the two runs look similar" is not a verdict. The pattern that worked here has four moves: (1) audit and seed every source of randomness before any same-seed comparison is attempted, because an unseeded sampler makes same-seed comparisons meaningless; (2) expect an episode-driven budget to amplify tiny GPU floating-point differences into different control flow, so an exact GPU comparison needs the deterministic-kernel flags *and* a self-replication control that proves the pinned environment is actually deterministic; (3) for the real production configuration, where determinism is off, judge the old-vs-new gap against a *measured* noise floor obtained by running the same side twice — not against intuition; (4) finish with a deterministic trio (old, old-repeated, new) which upgrades the claim from "statistically indistinguishable" to "the same computation".

## Evidence, measurements, facts

- **(a) Seed first.** Before the audit, both entry points shared an unseeded replay sampler, so two launches of the *same* entry point already diverged. Any same-seed A/B run before that fix would have been measuring sampler noise. The fix-set was applied identically to both trees so neither side got a code advantage. Details in [[20260806_0304_dreamer_sampler_seeding_full_determinism_recipe]].
- **(b) Episode budgets amplify float noise into control-flow divergence.** The budget is "20,000 episodes", so a last-place-digit difference in a GPU reduction changes an episode length, which changes how many gradient iterations fit — the runs stop being comparable step-for-step. The earlier A-vs-A triage confirmed this: two byte-identical launches of the same code diverged by *iteration count* without the determinism flags. That is why the deterministic leg pre-registered a control (`detA` vs `detA2`, an identical repeat on a different GPU) — without it, `detA ≡ detB` would not distinguish "the entry points are equivalent" from "the environment happened to be quiet".
- **(c) Measure the noise floor; do not assume it.** A dedicated pair `N1`/`N2` — the *legacy* entry point launched twice with identical commands on the same two GPU slots — gave the yardstick. Its divergence envelope: pointwise |diff| mean 1.77 survival steps, peak 5.23, end-of-run delta −3.88, and (the counter-intuitive part) **81/99 points one-directional** — pure noise reproduced the very sign-consistency that had made the old-vs-new offset look systematic. The actual A/B profile (mean 2.86, peak 5.96, end-of-run +2.56) sat inside that envelope on every statistic except run-mean, where it was ~1.6× a single noise draw.
- The same yardstick resolved two other apparent findings: the world-model loss gap between old and new (0.0094) was **7× smaller** than the noise pair's own spread (0.067), and a −4.6% throughput reading was reproduced almost exactly by legacy-vs-legacy on the same two GPU slots (−4.9%) — so it was a property of the slower GPU slot, not of the code.
- **(d) The deterministic trio closes it.** Three 20,000-episode runs on node 114: `detA` (legacy entry point on a pre-refactor throwaway tree + fix-set), `detA2` (byte-identical repeat, different GPU), `detB` (new entry point + same fix-set). `detA ≡ detA2` validated the environment; `detA ≡ detB` then proved exact equivalence — 20,000/20,000 episode records, 781,424 gradient steps on all three, 39 final-loss entries to 6 decimals, all 14 final-checkpoint modules SHA-256-identical including the PRNG key.
- **Comparison channels must be shown non-vacuous.** The comparator script was re-run and its record counts reproduced (20,000 / 489 / 482 / 39) and its first-divergence logic checked — a comparator that silently compares nothing passes every test. Provenance was checked too: the three logs are distinct files with distinct md5sums and correct per-tree launch banners, and the WandB IDs match the launch manifest.
- **Log-diffing gotcha:** the wrapper logs are tqdm-heavy, so carriage returns must be converted to newlines (`tr '\r' '\n'`) before any line-wise comparison, or every progress bar reads as one giant line.
- **The verdict itself got an adversarial review.** `plan-reviewer` re-checked the analysis against raw artifacts rather than the doc's claims, independently recomputing the checkpoint checksums at finer granularity, and returned CONCLUSION SUPPORTED WITH CAVEATS. The material caveat was an undisclosed operating point: the deterministic trio ran at 16 parallel environments with the CPU replay buffer, so "production scale" honestly means the 20,000-episode budget, not the 128-environment production configuration — coverage of which rests on the statistical leg plus identical resolved-config dumps.

## Decisions and actions

- Gate 2 of the Dreamer integration was declared PASS on both legs, unblocking the deprecation-shim phase.
- The pre-registered decision rule was applied as written even though the run-mean statistic sat in its undefined middle ("within the envelope" passes, "clearly exceeds" fails); the residual ambiguity was deliberately routed to the deterministic leg — the stronger instrument — rather than to more stochastic replication pairs.
- The plan-reviewer caveats were absorbed into the analysis doc as explicit disclosure sentences rather than argued away.
- Adopted as the house pattern for future behaviour-neutrality claims: audit → seed → noise floor → deterministic trio → adversarial verdict review.

## Open questions and follow-ups

- A two-run noise estimate is a single draw, so the statistical leg is a plausibility check, not a significance test. It was acceptable here only because the deterministic leg carried the equivalence burden; a comparison with no deterministic path available would need more pairs.
- Bit-identity was proven at the parity operating point (16 environments, CPU buffer), not at the 128-environment / GPU-buffer production point. A configuration-specific difference is bounded by the noise envelope, not excluded exactly.
- The stack's run-to-run spread is large (final world-model loss 1.753 vs 1.819 for byte-identical launches) — a standing warning against any future single-run Dreamer comparison.

## References

- Full arc: `docs/experiments/active/dreamer_integration/gate2_ab_analysis.md` (§4.5 noise floor, §4.6 deterministic leg, plan-reviewer feedback section).
- Gate specifications and the signed amendments: `docs/develop/active/dreamer/DREAMER_SRL_TRAIN_PY_INTEGRATION_PLAN.md`.
- Plan review: `docs/reviews/plan_dreamer_integration_rev4.md`.
- Comparison artifacts: `tmp/gate2_analysis_20260805/cmp_det_logs.py`, `det_ckpt_checksums.json`.
- Prerequisite finding: [[20260806_0304_dreamer_sampler_seeding_full_determinism_recipe|depends_on]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 87d7d010-fa7d-466e-9113-1a7618f94d95` (re-enter the session) or `python scripts/claude/claude_jsonl_to_md.py <jsonl> /tmp/20260806_0305_equivalence_testing_noise_floor_and_det_trio.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260806_0304_dreamer_sampler_seeding_full_determinism_recipe]] (dreamer_diagnosis, 2026-08-06) — Dreamer runs were never same-seed reproducible because the replay sampler drew b
- [[20260806_0306_gpu_claim_and_nas_git_lock_protocol]] (cluster_ops, 2026-08-06) — Three cluster-ops rules from the Dreamer Gate-2 arc: a free-looking GPU can alre
<!-- END BACKLINKS -->
