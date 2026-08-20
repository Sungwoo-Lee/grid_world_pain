---
id: 20260820_1606_reset_ulp_divergence_is_compiler_fusion
date: 2026-08-20
time: "16:06"
folder: env_entities
tags: [learned_lesson, refutation, meta]
summary: "Two runs of the same environment-reset code can disagree by one float32 last-bit (about 6e-08) on the sampled animal property, because the compiler is free to fuse 'mean + std x noise' into a single instruction in one place and not the other; this was first blamed on batching (vmap) and that was wrong."
related: ["20260806_0304_dreamer_sampler_seeding_full_determinism_recipe", "20260806_0305_equivalence_testing_noise_floor_and_det_trio"]
relations: ["extends:20260806_0304_dreamer_sampler_seeding_full_determinism_recipe", "see_also:20260806_0305_equivalence_testing_noise_floor_and_det_trio"]
session_origin: claude_code
session_label: "trajectory-store ULP divergence diagnosis"
importance: medium
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4efbe660-28c2-4643-b231-d3c6d2635b5a.jsonl
raw_completeness: full
---

# The one-last-bit disagreement in environment reset is compiler arithmetic reordering, not batching — and the wrong first answer is the lesson

## Key conclusion

When the environment is reset twice with the same random seed, one recorded column — `animal_property_sampled`, the per-episode visual/chemical property drawn for each animal — can come out different in its very last bit: a gap of at most 5.96e-08, which is the smallest representable step (one "unit in the last place", or ULP) for a 32-bit float near 1.0. Every integer and boolean field is always exactly equal, and the other five property columns drawn by the same helper are exactly equal too. The cause is not randomness and not a logic bug: the compiler (XLA) is allowed to compute `mean + std * noise` either as two separate instructions or as one fused multiply-add that keeps extra internal precision, and it makes that choice differently depending on the surrounding code. The first explanation offered — that batching (`vmap`) was responsible — was plausible and **wrong**, and the way it was disproved is the reusable part of this entry.

## Evidence, measurements, facts

- **Symptom.** `animal_property_sampled` differs between two runs by at most 5.96e-08 absolute, on roughly half the sampled episodes: 242 of 5,120 elements over 256 episodes in one measurement, and 116 of 200 episodes in a later check. All integer and boolean recorded fields are always bit-exact. `res_property_sampled` and `obs_property_sampled` — drawn by the *same* `_sample_property` helper — are also bit-exact.
- **Mechanism.** `mean + std * noise` may be emitted as a multiply followed by an add, or fused into a single fused-multiply-add (FMA) instruction. FMA carries more intermediate precision, so the two forms differ in the final bit. Which one XLA picks depends on how the surrounding code lowers. `animal_property_sampled` is the only one of the three property columns assembled next to a `jnp.zeros_like(...).at[idx].set(...)` scatter (`src/environment/core.py:1128-1140`, the per-class predator/neutral split), and that neighbouring scatter changes the fusion decision — which is exactly why its two siblings are unaffected.
- **The misattribution, and how it was settled.** The divergence was first attributed to batching: batched `vmap(jax_reset)` versus unbatched `jax_reset`. It was disproved by reproducing the *same* divergence — same column, same magnitude, at most 5.867e-08 — using the project's shipped parity-fixture generator (`scripts/fixtures/generate_parity_fixtures.py`), which calls **unbatched `jax_reset` only and contains no `vmap` anywhere**. Two unbatched runs disagreeing by precisely the effect that had been attributed to batching eliminates batching as the cause.
- **Two supporting arguments that were already available before that experiment** (they narrowed the search but could not settle it alone):
  - The PRNG key-parity guard passed, so both paths fed *identical bits* into the random-number generator. Whatever differs is therefore downstream, in the floating-point arithmetic, not in the randomness.
  - A genuine logic error — wrong key, wrong index set, a scatter writing the same index twice — would produce order-one differences across essentially every element of a Gaussian draw. Getting a difference of one last bit on about 5% of elements is not what a logic bug looks like.
- **Scale context.** One ULP on a property valued in [0, 1] is about 6e-08. The environment deliberately injects perceptual noise with a standard deviation between 0.01 and 0.20. The divergence is therefore six to seven orders of magnitude below the signal's own noise — scientifically irrelevant, but worth knowing because it looks alarming when a bit-exactness check trips on it.
- **Where this is now encoded in code.** `device` sits in `MANIFEST_GUARDED_FIELDS` in `src/utils/trajectory_store.py:442-464`, so a single trajectory store cannot mix CPU-produced and GPU-produced blocks (bit-level results are lowering-dependent, and a mixed store would hide that with nothing in the data to reveal it). The bound itself is pinned by `test_v1_animal_property_divergence_is_one_float32_ulp` in `tests/test_trajectory_collection.py:667-720`, which asserts both that the gap stays within one ULP and that **no other column joins it** — a widening or a spread is a different phenomenon and must be re-diagnosed rather than absorbed into a looser tolerance.

## Decisions and actions

- Accepted the divergence as pre-existing environment/compiler behaviour rather than a bug to fix: it is bounded, pinned by a regression test, and far below the environment's own injected noise.
- `device` was made a guarded manifest field so trajectory stores stay homogeneous; seed-pairing between two runs is treated as exact **only** per-device and per-lowering, never across them.
- **Recorded as a reusable diagnostic move**: to test whether X causes an effect, find an execution path that avoids X entirely and check whether the effect persists. This is stronger than reasoning about X, and it is cheap when the codebase already ships a second path (here, a fixture generator written for an unrelated purpose). It also beats the two "supporting arguments" above, which were sound but only narrowed the space.
- Second lesson, on plausibility: `vmap` was a *plausible* culprit — it is the visible difference between the two call sites, and batching genuinely does change lowering. Plausibility is what makes a wrong cause dangerous; it survives review because nobody objects to it. The refutation had to come from an experiment, not from more argument.

## Open questions and follow-ups

- The in-code documentation still names `vmap` as the cause. The comment at `src/utils/trajectory_store.py:452-455` and the docstrings of `test_v1_realised_draws_match_an_independent_unbatched_replay` / `test_v1_animal_property_divergence_is_one_float32_ulp` all frame it as "batched vs unbatched". Those descriptions predate this diagnosis and should be reworded to "lowering-dependent fusion" so a future reader is not re-taught the refuted cause. The guard and the bound themselves are correct as written and need no change.

## References

- Environment reset / property sampling: `src/environment/core.py:1128-1140` (the per-class scatter next to the divergent column).
- Manifest guard + rationale comment: `src/utils/trajectory_store.py:442-464`.
- Pinned bound + no-spread assertion: `tests/test_trajectory_collection.py:667-720`.
- The no-`vmap` reproduction path: `scripts/fixtures/generate_parity_fixtures.py`.
- The project's broader determinism picture — what is seedable, and what bit-identity costs on GPU: [[20260806_0304_dreamer_sampler_seeding_full_determinism_recipe|extends]].
- Kindred methodology (prove equivalence with a control that could have failed, not with an argument): [[20260806_0305_equivalence_testing_noise_floor_and_det_trio|see_also]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 4efbe660-28c2-4643-b231-d3c6d2635b5a` (re-enter the session) or `python scripts/claude/claude_jsonl_to_md.py <jsonl> /tmp/20260820_1606_reset_ulp_divergence_is_compiler_fusion.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
