# Dreamer Investigation — Synthesis (2026-07-27)

## Headline finding

We asked why our in-house DreamerV3 — an algorithm famous for solving Minecraft — looks
underwhelming on a 10x10 grid world. Five parallel investigations (speed, implementation
faithfulness, training curves, hyperparameter regime, task difficulty) converged on one
answer: **the implementation is clean and the agent is learning well per unit of
experience — but we run it ~100x slower than the model-free baseline in wall-clock, on
an oversized network, and the one thing still limiting learning is the value pathway
around rare lethal events** (the death signal a predator delivers roughly once per
hundred steps). Meanwhile the "Dreamer fails at predators" impression was manufactured
by measurement artifacts (a mixture of predator-free episodes in the survival metric,
an eval video seed that never drew a predator, and a model-free baseline that — with
its short discount horizon — is barely optimizing survival at all). Concretely: at
matched experience Dreamer beats the recurrent-PPO baseline 2.4x on survival; the paper's
own tables say our network M is over- not under-sized; and ~2x wall-clock is recoverable
at low risk. The open question is no longer "is Dreamer broken" but "is the remaining
gap the critic's resolution of rare catastrophes, or the task's irreducible randomness"
— and both have cheap, decisive probes.

## Verdicts per hypothesis

| # | Hypothesis | Verdict | Doc |
|---|---|---|---|
| H1 | Speed can be optimized further | **CONFIRMED — ~2x low-risk headroom.** 34–44% of wall-clock is periodic stalls (checkpoint + blocking video render every 1000 eps); the acting path is un-jitted (436 eager dispatches/iter); per-done-env Python autoreset costs ~22 ms/iter. Steady-state is 320 SPS (M) / 582 (XS), not the 209/314 cumulative figures. Buffer exonerated (~9 ms/iter). | [[dreamer_srl_h1_speed_investigation]] |
| H2a | Residual mis-implementation | **REFUTED — 0 🔴 findings.** Line-faithful to sheeprl@33b6366; all historical fixes verified present in live code; D-015 quantization benign at our operating points; free-nats clamp and truncated-vs-terminated handling both correct (checked on request of H2c). | [[dreamer_srl_faithfulness_review]] |
| H2b | Curves show pathological learning | **REFUTED — healthy-but-slow, throughput-bound.** World model plateaued (only negative-reward MAE still falling, −19%/8M steps); behavior still improving monotonically (48→123 survival steps, no plateau); value MAE still falling −20%. XS matches M at matched episodes → not capacity- or gradient-starved. | [[DREAMER_SRL_INVESTIGATION]] |
| H2c | Settings put us in the wrong regime | **PARTLY CONFIRMED, direction inverted.** Replay ratio is NOT too low (train_ratio 64 ≈ the paper's data-rich 32; M sits at 62 replayed-transitions/param vs the paper's 4–16 — over-updated if anything). The real mismatches: **M is over-sized** (paper's only vector benchmark uses its smallest model at its highest update rate) and **the critic's two-hot bins are too coarse** — the entire survival improvement to date spans ~1.2 bins. Prescribes XS @ rr 0.15–0.25 (wall-clock-POSITIVE per the paper's data-efficiency exponent) + bins narrowed ±20→±6. | [[dreamer_srl_settings_regime_critique]] |
| H3 | Task simpler / harder than the DreamerV3 suite | **DIFFERENT IN KIND.** Simpler on every axis the suite discriminates on (~90% conf). Harder in kind on axes the suite barely tests (~85%): contact-only vision, globally-superposed smell, unobservable injury, zero-smell hiding predators, per-episode hidden dynamics (HiP-MDP), some episodes unwinnable at t=0. The dense homeostatic reward is near-potential-based → the only controllable signal is the sparse death catastrophe. The achievable ceiling has never been measured. | [[gridworld_vs_dreamerv3_benchmarks_difficulty]] |

## Where the five reports converge (unplanned, from independent directions)

1. **The value pathway around rare lethal events is the binding constraint.**
   H2b: the only still-improving WM error is reward-MAE on negative rewards, and value
   MAE lags. H2c: the critic can resolve the whole task in ~1.2 bins. H3: the reward
   algebra says death is the only policy-sensitive signal. Three independent methods,
   one location.
2. **XS is the right size.** H2b empirically (XS ≈ M at matched episodes, 1.5x faster);
   H2c from the paper's own scaling tables (M over-updated per parameter).
3. **The historical "Dreamer fails" impression was measurement, not learning.**
   Predator-count mixture in the survival metric (established earlier this week), the
   seed-0 predator-free eval videos (fixed, commit b228117), decay_power 2.0 on all
   pre-07-26 runs (reverted, d6240f5), and — new from H3 — the rPPO baseline's γ=0.95
   makes death at 40 steps worth 0.13, i.e. the baseline we compared against is
   optimizing a ~20-step myopic surrogate. (`configs/models/recurrent_ppo/recurrent_ppo_M.yaml:6`;
   note basic-config rPPO runs inherit γ from their own config — verify per run before reuse.)

## Conflicts between workers (surfaced, not smoothed over)

- **Raise replay ratio?** H2b reads the curves as "not gradient-starved" (XS fits worse
  yet behaves the same → more updates shouldn't help). H2c's frontier math predicts
  rr 0.25 is wall-clock-positive (α ≈ 0.66–0.78 from the paper's Fig. 6a). These are
  reconcilable — H2b measures the WM, H2c's α is about the full agent including the
  critic — but they make opposite bets on the same knob. **Resolution: the pre-registered
  rr sweep (0.0625 / 0.25 / 1.0, XS, ≥3 seeds) H2b already specified.**
- **Is the task hard?** H3 refuses to say "harder overall" (~35%) until the oracle
  ceiling is measured; H2b's healthy curves are compatible with either a low ceiling
  (aleatoric-limited) or a large learnable gap. **Resolution: H3's P1 oracle-observation
  probe (~4 rPPO runs, ~1.2 h each).**

## Decision options implied (for the PI call)

A. **Retune + relaunch** the Dreamer arm at the prescribed point (XS, rr 0.25, bins ±6,
   with the H1 stall fixes) — cheapest path to a defensible Dreamer baseline.
B. **Probe first**: run P4 (critic-bin histogram — one checkpoint, no retraining) and
   P1 (oracle ceiling) before touching any training knob.
C. **Speed engineering first** (async render, jit the actor, masked reset, then 256 envs)
   — multiplies everything downstream.
D. **Accept + document**: keep the current 2x2 grid running as-is for the size study;
   fold changes into the NEXT study.

## Manifest

Investigators: senior-developer (H1), code-reviewer (H2a + follow-up), experiment-analyzer
(H2b, Mode B snapshot), professor-rl ×2 (H2c, H3), bug-curator (baseline). All read-only;
the four node-114 runs were not disturbed. Live runs analyzed: dsrl_b03_M_dp1, dsrl_b03_XS_dp1,
dsrl_b04_M_dp1, dsrl_b04_XS_dp1 (saved configs verified: decay_power 1.0, rr 0.0625,
num_envs 128, seed 0); rPPO references rppo_b03/b04_mc_dp1_n110.
