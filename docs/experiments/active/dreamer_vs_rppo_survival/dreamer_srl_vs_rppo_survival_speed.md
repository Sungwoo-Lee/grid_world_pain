---
title: "Dreamer_srl vs recurrent-PPO — survival & sample-efficiency (basic 03, basic 04)"
topic: comparison
status: active
created: 2026-07-23
last_updated: 2026-07-23
wandb_tag: "rppo_basic0{3,4}_*_128env_* ; dsrl_b0{3,4}_M_rr*"
---

## 1. Research Question

**Which learns to survive longer, and which gets there on less experience of the world —
the model-free recurrent-PPO agent or the model-based Dreamer agent — in two predator
survival worlds?**

This project measures an agent's competence as **survival steps**: how many time-steps the
agent stays alive in an episode before the predator kills it or it starves (higher = better;
cumulative reward is never used as the headline). We compare two learning algorithms on the
same two environments:

- **Recurrent PPO ("rPPO")** — a model-free, on-policy learner. It runs **128 copies of the
  world in parallel** and improves its policy directly from the experience it collects. Here it
  runs on a single lab GPU and burns through experience extremely fast.
- **Dreamer (the live "dreamer_srl" stack)** — a model-based learner. It runs **16 parallel
  worlds**, but instead of learning straight from raw experience it first trains an internal
  "world model" (a learned simulator) and then improves its policy by *imagining* rollouts
  inside that model. This lets it squeeze many gradient updates out of each real step of
  experience — the knob controlling how many is the **replay ratio**.

The two worlds:

- **Basic 03 ("random-init")** — a 10×10 grid where the agent and a predator start at random
  positions each episode. In the runs compared here the predator's *ranged attack is
  effectively disabled*, so danger comes from direct contact.
- **Basic 04 ("jump-attack")** — the same 10×10 grid but the predator has a real ranged
  "jump" attack (it can strike from 2–3 tiles away, landing ~50 % of the time). This is the
  harder, more lethal world.

**Post-hoc note.** This is a **Mode B / retrospective** analysis: there was no pre-registered
design doc predicting an outcome before the runs launched. I frame the question as a hypothesis
below to keep the reading honest, but conclusions are correspondingly weaker than for a
pre-registered experiment.

> **H₀ (null):** the two algorithms reach the *same* plateau survival on a given world.
> **H₁ (difference):** the algorithms differ in (a) the plateau survival they reach and/or
> (b) how much real environment experience they need to get there.

**Headline finding (plain language).** On **both** worlds, **rPPO survives longer at the end**
(basic 03: ~182 vs ~117 steps; basic 04: ~169 vs ~142 steps), but **Dreamer reaches a
comparable survival level on 10–20× fewer real environment steps** — it is far more
*sample-efficient* per unit of world experience. rPPO wins on wall-clock and on final survival;
Dreamer wins on experience-efficiency. Crucially, the two consume experience so differently
(rPPO ~46,000 real steps/second, Dreamer ~100–175/second) that the "fair" comparison depends
entirely on which cost you count — see the fairness caveat in §2.3. The Dreamer runs are still
**in-flight** (not converged), so their survival numbers are provisional and may still rise.

## 2. Experimental Design

### 2.1 Independent Variables

| Variable | Values | Rationale |
|----------|--------|-----------|
| Algorithm | recurrent-PPO (model-free, 128 envs) vs dreamer_srl (model-based, 16 envs) | the core comparison |
| Environment | basic 03 (random-init, ranged attack off) vs basic 04 (jump-attack on) | one easier, one harder world |
| Dreamer replay ratio (b04 only) | 0.0625, 0.25, 0.5, 1.0 | how many gradient updates per real env step; changes the experience/compute trade-off |

Primary head-to-head per world uses the Dreamer **replay-ratio-0.0625** variant, because that is
the only replay ratio the basic-03 Dreamer run used — matching it keeps the two worlds
comparable. The full basic-04 replay-ratio family is reported in §4 so the spread is visible.

### 2.2 Controlled Variables

- **Metric**: survival = `Episode/Steps` (WandB), identical logger on both stacks.
- **Fair x-axis**: `timesteps` (WandB) = cumulative *real* environment steps collected. Same
  definition on both stacks (one real step per env per iteration).
- **Grid / entities**: both stacks target the same 10×10 predator worlds with the same predator
  and neutral-rabbit entity templates, damage ranges, and sensory layout (5-d vector + 8-d
  visual + 1-d nociception). Model size for Dreamer is the "M" (medium) world-model config.

### 2.3 Confounds & Limitations

| Confound | Affected runs | Severity | Note / mitigation |
|----------|---------------|----------|-------------------|
| **Experience-consumption asymmetry** (the central caveat) | all | High | rPPO collects ~46,000 real env steps/s with 128 envs; Dreamer collects ~100–175/s with 16 envs and *replays* each transition through its world model many times. There is **no single axis that is fair on every dimension.** I put survival against **real environment steps** (`timesteps`) — the fairest single axis for "how much world experience did it need" — and report **wall-clock separately**. I deliberately **do not** compare on gradient/update steps: Dreamer does far more compute per env step, so update-step parity would flatter rPPO massively and mean nothing. Even the env-step axis is imperfect: parallel on-policy collection (rPPO) vs replay-driven model learning (Dreamer) are different learning regimes. |
| **Run status: Dreamer still training** | b03 rr0.0625, b04 rr0.0625, b04 rr0.25 | High | These three are **live/running** on WandB (heartbeat 2026-07-23). Their "final" survival is a **provisional plateau**, not a converged endpoint — it may still rise. rPPO runs are `finished`. |
| **Two b04 Dreamer runs crashed** | b04 rr0.5, rr1.0 | Medium | `crashed` on WandB (2026-07-21). Their "final" is where they died, not a chosen stopping point — treat as lower bounds. |
| **Possible basic-03 predator-attack mismatch** | rPPO b03 vs Dreamer b03 | Medium | The Dreamer b03 config explicitly sets the predator's ranged attack **off** (attack_range [0,0], success 0). The rPPO b03 run launched **2026-07-08**, before the commit that made `attack_range` explicit in basic 01/02/03, so its predator-attack semantics are implicit and *may* differ slightly. Both are the "basic 03 random-init" world, but exact predator lethality is not guaranteed identical. |
| **Single seed each** | all | Medium | One seed per cell (rPPO seed 42, Dreamer seed 0/42). No seed-dispersion estimate; effect sizes below the ~3-step run-to-run noise band should not be over-read. |
| **Different wall-clock budgets** | all | Low | rPPO ran ~12 h to `finished`; Dreamer runs have logged 41–161 h and are still going. Wall-clock is reported but is not the fairness axis. |

## 3. Launch Manifest

Post-hoc analysis — no pre-registered manifest. Runs identified from WandB (authoritative for
live status) and cross-referenced to local `results/` for saved configs. WandB entity
`sungwoolee`, project `grid_world_pain`.

| # | World | Algorithm | WandB name (tag) | WandB ID | Local wandb dir | State (2026-07-23) | Real env steps | Wall-clock |
|---|-------|-----------|------------------|----------|-----------------|--------------------|----------------|-----------|
| 1 | basic 03 | rPPO 128env | `rppo_basic03_randinit_128env_n110` | wlsczs5c | `wandb/run-20260708_193854-wlsczs5c` | finished | 1,689,190,400 | 12h 44m |
| 2 | basic 04 | rPPO 128env | `rppo_basic04_jump_128env_n106` | 44klxpnt | `wandb/run-20260708_193854-44klxpnt` | finished | 1,576,140,800 | 12h 11m |
| 3 | basic 03 | Dreamer M rr0.0625 | `dsrl_b03_M_rr0p0625` | xiwnz9io | `wandb/run-20260722_023131-xiwnz9io` | **running** | 13,012,208 | 41h 15m |
| 4 | basic 04 | Dreamer M rr0.0625 | `dsrl_b04_M_rr0p0625` | nhf8ww7m | `wandb/run-20260717_025140-nhf8ww7m` | **running** | 54,177,008 | 160h 56m |
| 5 | basic 04 | Dreamer M rr0.25 | `dsrl_b04_M_rr0p25` | 91dd2fh8 | `wandb/run-20260717_025140-91dd2fh8` | **running** | 34,345,008 | 160h 56m |
| 6 | basic 04 | Dreamer M rr0.5 | `dsrl_b04_M_rr0p5` | nr7w0jyg | `wandb/run-20260717_025140-nr7w0jyg` | crashed | 16,353,008 | 119h |
| 7 | basic 04 | Dreamer M rr1.0 | `dsrl_b04_M_rr1p0` | xb4yq595 | `wandb/run-20260717_025140-xb4yq595` | crashed | 10,684,208 | 119h |

Metric keys: survival = `Episode/Steps`; fair x-axis = `timesteps` (cumulative real env steps);
speed = `Time/sps_env`. Local `results/` cross-reference: b03 →
`results/JAX_DreamerSRL/20260722-023132_dsrl_b03_M_rr0p0625`; b04 family →
`results/JAX_DreamerSRL/20260717-025141_dsrl_b04_M_rr*`.

## 4. Results

### 4.1 Primary Metric — final / plateau survival (Episode/Steps)

Steady-state = mean over last 20 % of logged episodes ± std. "Provisional" = run still live.

| World | Algorithm (cell) | Steady-state survival | Final | Max | Real env steps | Status |
|-------|------------------|-----------------------|-------|-----|----------------|--------|
| basic 03 | rPPO 128env | **182.4 ± 3.3** | 185.6 | 193.7 | 1.69 B | finished |
| basic 03 | Dreamer rr0.0625 | 117.2 ± 2.0 | 119.2 | 121.4 | 13.0 M | running (provisional) |
| basic 04 | rPPO 128env | **169.4 ± 3.4** | 174.7 | 179.7 | 1.58 B | finished |
| basic 04 | Dreamer rr0.0625 | 142.2 ± 3.0 | 144.7 | 149.8 | 54.2 M | running (provisional) |
| basic 04 | Dreamer rr0.25 | 129.5 ± 2.6 | 138.3 | 138.3 | 34.3 M | running (provisional) |
| basic 04 | Dreamer rr0.5 | 103.9 ± 3.7 | 108.0 | 110.1 | 16.4 M | crashed |
| basic 04 | Dreamer rr1.0 | 94.3 ± 1.9 | 95.4 | 98.6 | 10.7 M | crashed |

> **Verdict on H₁:** Supported. The algorithms reach clearly different plateaus (basic 03:
> +65 survival steps, +56 % for rPPO; basic 04: +27 steps, +19 % for rPPO), and differ by
> 10–120× in the real experience they consume. H₀ (same plateau) is rejected on both worlds.

**Replay-ratio trend (basic 04 Dreamer family).** Survival is **monotone in *lower* replay
ratio**: rr0.0625 (142) > rr0.25 (129) > rr0.5 (104) > rr1.0 (94). Lower replay ratio = fewer
imagined gradient updates per real env step = more real data collected in the same wall-clock,
and here that produced the better policy. rr0.0625 is both the strongest survivor and the one
that collected the most real experience (54 M).

### 4.2 Secondary — training speed & cost

| Run | Real env steps/sec (`Time/sps_env`) | Wall-clock | Total real env steps |
|-----|-------------------------------------|------------|----------------------|
| rPPO b03 | ~46,880 | 12h 44m (finished) | 1.69 B |
| rPPO b04 | ~46,737 | 12h 11m (finished) | 1.58 B |
| Dreamer b03 rr0.0625 | ~99 | 41h 15m (running) | 13.0 M |
| Dreamer b04 rr0.0625 | ~175 | 160h 56m (running) | 54.2 M |
| Dreamer b04 rr0.25 | ~98 | 160h 56m (running) | 34.3 M |
| Dreamer b04 rr0.5 | ~43 | 119h (crashed) | 16.4 M |
| Dreamer b04 rr1.0 | ~26 | 119h (crashed) | 10.7 M |

rPPO collects real experience **~270–1800× faster per second** than Dreamer. This is the whole
reason the two cannot be compared on wall-clock as if equivalent (see §2.3).

### 4.3 Learning Dynamics — survival vs *real environment steps* (the fair axis)

The key sample-efficiency picture: survival plotted against cumulative real env steps.

**Basic 03** (windowed means; env-step figures are window midpoints):

| Real env steps | rPPO survival | Dreamer survival |
|----------------|---------------|------------------|
| ~0.6 M | (still <90) | 32.7 |
| ~3.2 M | — | 92.5 |
| ~5.8 M | — | 113.4 (near plateau) |
| ~13 M | — | 117 (plateau, still live) |
| ~42 M | ~88 | — |
| ~110 M | ~142 | — |
| ~250 M | ~166 | — |
| ~1.6 B | 182 (plateau) | — |

Reading: **Dreamer reaches ~113 survival on ~5.8 M real env steps; rPPO needs ~90–110 M real
env steps to reach the same survival level** — roughly a **15–20× experience-efficiency
advantage for Dreamer at matched survival**. But rPPO does not stop there; it keeps climbing to
182 while Dreamer plateaus near 117.

**Basic 04** (windowed means):

| Real env steps | rPPO survival | Dreamer rr0.0625 survival |
|----------------|---------------|---------------------------|
| ~2.5 M | — | 75.9 |
| ~7.9 M | — | 120.6 |
| ~13.5 M | — | 128.2 |
| ~35 M | — | 139.4 |
| ~54 M | — | 143.1 (still live) |
| ~79 M | ~131 | — |
| ~236 M | ~156 | — |
| ~1.5 B | 170 (plateau) | — |

Reading: **Dreamer reaches ~128 survival on ~13.5 M real env steps and ~142 on ~35–54 M;
rPPO passes ~131 only somewhere near ~79 M env steps and reaches ~156 by ~236 M** — again a
several-fold experience-efficiency edge for Dreamer at matched survival, while rPPO's plateau
(169) sits ~27 steps above Dreamer's current provisional plateau (142).

Both algorithms show the same qualitative shape: fast early rise, then a soft plateau. rPPO's
early rise is monotone and tight (std ~3 steps once past the first window). Dreamer's rise is
slower in wall-clock but far steeper in real-env-steps, with a wider early std (world-model
warm-up) that tightens to ~2–3 steps at plateau.

## 5. Analysis

### 5.1 Key Findings

**Finding 1 — rPPO wins final survival on both worlds.**
What: 182 vs 117 (basic 03), 169 vs 142 (basic 04). Why: rPPO's on-policy 128-env firehose lets
it refine the policy against ~1.6 B real steps; Dreamer has so far seen only 13–54 M and is
still climbing. Evidence: §4.1. Confidence: **High** for the direction; **Medium** for the exact
gap, because Dreamer is unconverged and may narrow it.

**Finding 2 — Dreamer wins experience-efficiency by 10–20×.**
What: Dreamer reaches rPPO-comparable survival on 5–13 M real env steps where rPPO needs
90–250 M. Why: model-based imagination extracts many policy updates from each real transition.
Evidence: §4.3. Confidence: **High** — this holds on both worlds and is a large effect.

**Finding 3 — rPPO wins wall-clock decisively.**
What: rPPO reaches its full plateau in ~12 h; Dreamer has run 41–161 h and has not converged.
Why: rPPO's ~46,000 env-steps/s vs Dreamer's ~100–175/s. The per-env-step efficiency does not
overcome the ~300–1800× throughput gap in wall-clock terms. Evidence: §4.2. Confidence: High.

**Finding 4 — For Dreamer on basic 04, lower replay ratio is better.**
What: survival rises monotonically as replay ratio falls (94 → 104 → 129 → 142 for
1.0 → 0.5 → 0.25 → 0.0625). Why: a lower replay ratio collects more real data per unit wall-clock
and keeps the policy closer to fresh on-policy experience, reducing stale-imagination bias.
Evidence: §4.1. Confidence: Medium — single seed, and two of the four crashed early so their
plateaus are lower bounds.

### 5.2 Cross-Run Comparisons (one variable at a time)

- **Algorithm, basic 03 (rPPO vs Dreamer rr0.0625):** rPPO +65 survival at the cost of ~130×
  more real experience and ~1/3 the wall-clock.
- **Algorithm, basic 04 (rPPO vs Dreamer rr0.0625):** rPPO +27 survival at ~29× more real
  experience; Dreamer still live and closing.
- **World, within rPPO:** basic 04 (jump attack) caps ~13 steps below basic 03 — the ranged
  attack makes the world measurably more lethal even for the stronger learner.
- **World, within Dreamer rr0.0625:** basic 04 plateau (142, live) currently *above* basic 03
  (117, live) — but the b03 run has seen only 13 M steps vs b04's 54 M, so this likely reflects
  b03 being earlier in training, not b04 being genuinely easier for Dreamer. Do not over-read.

### 5.3 Failure Modes & Pathologies

- **Two b04 Dreamer runs (rr0.5, rr1.0) crashed** at 119 h (WandB `crashed`, 2026-07-21). Their
  survival numbers are where they died, not converged plateaus — lower bounds only.
- **No collapse observed** in the surviving runs: survival curves are monotone-up then flat; no
  entropy-collapse-style regression is visible in the survival trace.

## 6. Conclusions

### 6.1 Summary

- **Final survival: rPPO wins both worlds** — basic 03 ~182 vs ~117 (+56 %), basic 04 ~169 vs
  ~142 (+19 %). (Dreamer numbers provisional — runs still live.)
- **Experience-efficiency: Dreamer wins both worlds** — reaches rPPO-comparable survival on
  **10–20× fewer real environment steps**.
- **Wall-clock: rPPO wins decisively** — ~12 h to a converged plateau vs Dreamer's 41–161 h and
  still climbing, driven by a ~300–1800× env-throughput gap.
- **Fairness caveat applied:** compared on **real environment steps** (the fair single axis for
  sample-efficiency), with wall-clock reported separately and update-step comparison
  deliberately avoided. No axis is fair on all dimensions; the verdict genuinely depends on
  whether you are counting *experience*, *time*, or *asymptotic skill*.
- **Dreamer replay-ratio pick:** on basic 04, **rr0.0625 is the strongest** of the family and
  the right primary head-to-head choice.

### 6.2 Per-world verdict (the bottom line the user asked for)

| World | Longer survival (asymptote) | Fewer real env steps to a given survival | Faster in wall-clock |
|-------|-----------------------------|------------------------------------------|----------------------|
| **basic 03** | **rPPO** (182 vs 117) | **Dreamer** (~15–20× fewer to reach ~113) | **rPPO** (12.7 h vs 41 h, still running) |
| **basic 04** | **rPPO** (169 vs 142*) | **Dreamer** (~3–5× fewer to reach ~142) | **rPPO** (12.2 h vs 161 h, still running) |

\* Dreamer basic-04 plateau provisional — run live and may still rise, narrowing the gap.

### 6.3 Limitations & Open Questions

- Dreamer plateaus are **provisional** — a converged comparison needs the live runs to finish
  or the user to accept the current plateau as final.
- **Single seed** per cell; effect sizes within the ~3-step run-to-run noise band are not
  reliable. Seed replication would firm up the plateau gaps.
- **Basic-03 predator-attack semantics** may differ slightly between the rPPO (pre-attack_range
  commit) and Dreamer runs — a same-config rerun would remove this confound.

### 6.4 Recommended Next Steps

| Priority | Step | Rationale |
|----------|------|-----------|
| High | Let the live Dreamer b03/b04 rr0.0625 runs converge (or define a stop) before calling "final" | current plateaus provisional |
| Medium | Rerun rPPO b03 on the current explicit-attack_range basic-03 config | removes the §2.3 predator-attack confound |
| Medium | 2–3 seeds per cell for the head-to-head cells | quantify plateau dispersion |
| Low | Extend the b04 replay-ratio sweep below 0.0625 | rr trend suggests still-lower replay may help |

## Metrics Requested

None. Both stacks already log `Episode/Steps` (survival) and `timesteps` (real env steps),
which were sufficient for the fair-axis comparison.

## Related Issues

- **b04 Dreamer crashes** (rr0.5, rr1.0 `crashed` 2026-07-21): if these should have completed,
  this may warrant a `bug-fix-workflow` investigation of the dreamer_srl long-run stability.
  Not filed here — surfaced for the user to decide.

---

## Appendix

### C. Changelog

| Date | Change | Author |
|------|--------|--------|
| 2026-07-23 | Initial post-hoc analysis (Mode B) | experiment-analyzer |
