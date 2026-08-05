---
id: 20260806_0304_dreamer_sampler_seeding_full_determinism_recipe
date: 2026-08-06
time: "03:04"
folder: dreamer_diagnosis
tags: [dreamer, learned_lesson, decision, meta]
summary: "Dreamer runs were never same-seed reproducible because the replay sampler drew batches from an unseeded random generator; seeding it from the run seed fixes that, and adding two GPU determinism environment flags makes whole 20,000-episode runs bit-identical."
related: ["20260806_0305_equivalence_testing_noise_floor_and_det_trio"]
relations: ["extends:20260806_0305_equivalence_testing_noise_floor_and_det_trio"]
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

# Dreamer same-seed reproducibility: the unseeded replay sampler was the only cause, and the full-determinism recipe needs two GPU flags on top of the fix

## Key conclusion

Dreamer (`dreamer_srl`) training was never reproducible from a seed: the replay buffer built its own random generator with no seed, so which sequences formed each gradient batch was decided by OS entropy and two identically-seeded runs diverged from the very first gradient step. A fresh grep-and-trace audit of the whole stack found this to be the *only* training-affecting source of unseeded randomness — everything else traces back to the run seed — and confirmed the standing claim that recurrent-PPO is already fully same-seed clean. Seeding the sampler from the run seed fixes reproducibility on CPU; making a GPU run *bit*-identical additionally requires pinning XLA's deterministic-kernel flag, because GPU kernels are nondeterministic by default.

## Evidence, measurements, facts

- Root cause, two sites: `src/algorithms/dreamer_srl/buffers.py:76` (`SequentialReplayBuffer.__init__`) and `:782` (`EnvIndependentSequentialReplayBuffer.__init__`) each called `np.random.default_rng()` with no argument. Consumed by the CPU replay `sample()` path for start-index / env-index draws, i.e. it decides the contents of every gradient batch.
- Measured symptom: a 40-episode CPU smoke test with an identical seed produced **404 vs 440 gradient steps** on two launches (comment block at `dreamer_srl_main.py:1107-1115`).
- A partial pin already existed but was dead for real runs: `dreamer_srl_main.py:1116-1120` pinned both generators, but only inside `if spec.parity_dump is not None` — i.e. only under the parity harness. Production runs kept OS-entropy sampling.
- The GPU-buffer opt-in path (`--buffer-device gpu`) was never affected — it threads a JAX key split from the run key (`buffers.py:393-406`).
- Fix: seed the wrapper/single buffer with `default_rng(seed)` and sub-buffer *i* with `default_rng(seed + 1 + i)` — distinct streams, and resume-safe because `buffer.reset()` keeps the Generator objects and only zeroes `_pos`/`_full` (`buffers.py:598-604, 901-905`). Landed as commit `2016f05` on the integration worktree branch `agent-a18eeb59fa7ffffeb`; it reaches `v3.0` when that branch merges.
- Recurrent-PPO audit verdict: **clean**, traced end to end — a single `jax.random.PRNGKey(seed)` master thread, no `np.random` / `random` / `uuid` / `tempfile` anywhere in its path, no dropout, and no minibatch shuffling at all (the update runs full-batch epochs with no permutation, so there is nothing to seed). DQN/DRQN/PPO co-audited seed-clean.
- Full bit-determinism recipe (GPU): the sampler fix **plus** `XLA_FLAGS=--xla_gpu_deterministic_ops=true` and a matched `XLA_PYTHON_CLIENT_PREALLOCATE` setting across the runs being compared. GPU kernels are nondeterministic by default; the deterministic-ops flag costs roughly 15% throughput. CPU runs are deterministic without any flags.
- Proven at scale: three 20,000-episode runs (~19 h each) on node 114 came out bit-identical in every logged number — 20,000/20,000 per-episode records, identical checkpoint boundaries, 781,424 gradient steps on all three, 39-entry final-loss dicts equal to 6 decimals — and all 14 modules of the final checkpoint matched by SHA-256, including the optimizer states and the PRNG key itself.
- Only-cosmetic randomness left in the stack (deliberately not fixed): which checkpoints get a *live* evaluation or a rendered video depends on subprocess wall-clock (skip-if-busy dispatch), and run-directory names are timestamped. Coverage varies; the numbers those evaluations produce do not.
- Audit report: `tmp/20260805_nondeterminism_audit.md`. Scale proof: `docs/experiments/active/dreamer_integration/gate2_ab_analysis.md` §4.6.

## Decisions and actions

- Decided (user-endorsed) to seed the production sampler rather than leave the pin parity-only — the reproducibility gain was judged worth more than preserving OS-entropy sampling, which had no upside.
- The known-bugs registry row "Dreamer training is not repeatable — the same seed gives a different run every time" is to be closed by `bug-curator` when the integration branch lands.
- The two GPU determinism environment variables were adopted as the standard recipe for any run pair that must be compared exactly, not as a default for production runs (the ~15% speed cost is not worth paying when nobody is comparing).
- Recurrent-PPO's "fully same-seed reproducible" claim is now audit-backed rather than assumed, so rPPO comparisons need no special handling.

## Open questions and follow-ups

- **Sampler state is still not checkpointed.** A run resumed from a checkpoint restores weights, optimizer states and the PRNG key, but the replay sampler's generator restarts from its seed rather than from where it left off. So a resumed run is not bit-identical to an uninterrupted one. Tracked as an open refinement in the bug registry, not blocking.
- The determinism environment flags are not recorded in the run logs themselves, so a deterministic run is currently self-certified by whoever launched it. Worth emitting into the log banner if this gets used often.

## References

- Audit: `tmp/20260805_nondeterminism_audit.md` (per-site evidence, rPPO trace, cosmetic-findings table).
- Scale proof + verdict: `docs/experiments/active/dreamer_integration/gate2_ab_analysis.md` §4.6; checksum table `tmp/gate2_analysis_20260805/det_ckpt_checksums.json`.
- Bug registry row: `docs/develop/active/issues/KNOWN_BUGS.md` ("Dreamer training is not repeatable…").
- Plan: `docs/develop/active/dreamer/DREAMER_SRL_TRAIN_PY_INTEGRATION_PLAN.md` (deterministic Gate-2 amendment).
- Fix commits on worktree branch `agent-a18eeb59fa7ffffeb`: `2016f05` (sampler seeding), `6eddaa5` (determinism environment + bench config paths), `a04ffc6` (sign-off).
- The methodology this enabled is captured in [[20260806_0305_equivalence_testing_noise_floor_and_det_trio|extends]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 87d7d010-fa7d-466e-9113-1a7618f94d95` (re-enter the session) or `python scripts/claude/claude_jsonl_to_md.py <jsonl> /tmp/20260806_0304_dreamer_sampler_seeding_full_determinism_recipe.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- [[20260806_0305_equivalence_testing_noise_floor_and_det_trio]] (dreamer_diagnosis, 2026-08-06) — Reusable recipe for proving a refactor did not change a working training system:
<!-- END BACKLINKS -->
