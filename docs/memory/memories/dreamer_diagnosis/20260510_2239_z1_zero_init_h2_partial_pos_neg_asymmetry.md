---
id: 20260510_2239_z1_zero_init_h2_partial_pos_neg_asymmetry
date: 2026-05-10
time: "22:39"
folder: dreamer_diagnosis
tags: [dreamer, learned_lesson, decision, refutation]
summary: "Cell Z1 (zero-init reward+critic output layers, sheeprl candidate #4) re-run on the simplest food-only task fired H2 — partial fix. Reward MAE @ h=5 dropped 0.386 → 0.277 (28%). Mechanistic headline: training-time positive-reward MAE improved 49% but negative-reward MAE only 14%. The pos/neg asymmetry directed the next fix candidate (twohot bin range, §6 item 2) since negative rewards of magnitude > 20 sit outside our head's representable bin support."
related: ["20260509_1534_wm_reward_head_localized_failure_a1", "20260509_1535_conventional_fixes_battery_verdict_predator_refute"]
session_origin: claude_code
session_label: "dreamer_sheeprl_compare_and_zero_init_2026-05-10"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/4ae5401e-443a-406e-93e1-5431c67b5f59.jsonl
raw_completeness: full
---

# Z1 zero-init H2 partial fix; pos/neg reward-MAE asymmetry directs next candidate

## Key conclusion
The first targeted code-side fix to the localized DreamerV3 reward-head failure — sheeprl-style zero-initialization of the reward and critic output Linear layers — produced a **partial fix (H2-band)**. Reward MAE @ h=5 fell from 0.386 (Cell A1 baseline) to 0.277 (Cell Z1) — a 28% reduction, but still 1.85× over the pre-registered H1 threshold of 0.15. The mechanistic finding that emerged from the offline WM-imagination diagnostic is the load-bearing one: the residual reward-head error after zero-init is **disproportionately on negative-reward events**. Training-time `model_reward_mae_pos` improved 49%; `model_reward_mae_neg` improved only 14%. This pos/neg asymmetry — not just the H2 band — directs the next candidate fix toward the two-hot bin-range deviation (§6 item 2 in the concept doc), because our bins span only raw ±20 while the death-penalty event of −100 lives outside the head's representable support entirely.

## Evidence, measurements, facts
- **Cell Z1 launch**: `dreamer_zinit_NoPred_rr06_s0_n113`, n113:0, WandB `axndoqsz`, group `dreamer_zero_init`, 700k env-steps, ~2.5h wall-clock. Single seed (0). Same NoPred config and rr06 agent config as Cell A1; the only delta is the new mandatory key `agent.zero_init_reward_critic: true` (default true after this fix).
- **Code change**: `src/models/dreamer_v3_nnx.py` `MLP` class gains `zero_init_output: bool = False`; `WorldModel` and `ActorCritic` route the flag to reward + critic heads only (continue / actor / decoder / encoder / prior+posterior keep `hafner_init`). Commits `a88002f` (code) + `dad09e1` (diary). Smoke test: kernel = 0, bias = 0, forward-at-init = 0 confirmed.
- **Offline diagnostic on Z1 checkpoint** (`scripts/dreamer_offline_wm_test.py`, M=200 starts, horizons {1, 2, 5, 10, 15, 25, 50}, deterministic mode):
  - Reward MAE @ h=5: **0.277** (vs A1 0.386). Δ = −0.109, Δ% = −28%. Pre-registered: H1 < 0.15 (FAIL); H2 ∈ [0.15, 0.30) (FIRE); H0 ≥ 0.30 (refuted).
  - Aggregate obs symlog-MSE @ h=5: 0.047 (was 0.059) — PASS both. −20%.
  - Continuation accuracy @ h=5: 1.000 (was 0.995) — PASS both.
  - Long-horizon h50/h5 ratio: 1.62 (was 1.69) — PASS both. Autoregressive drift unchanged.
  - Per-channel obs MSE @ h=5 — proprioception (= prev-action one-hot): 0.054 (was 0.107). Improved 50%, still hairline over 0.05 threshold.
- **Training-time WandB metrics on Z1** (the load-bearing mechanism):
  - `model_reward_mae_pos`: 0.42 (vs A1 0.83) — **−49%**.
  - `model_reward_mae_neg`: improved only **14%** vs A1 — the residual reward-head error is concentrated on negative rewards.
- **Survival**: Z1 steady-state ~115 vs A1 ~106 — soft +9-step lift on a single seed, not a primary signal.
- **Comparison anchor**: Cell A1 baseline (`czfnljf0`, conventional-fixes battery, `dreamer_conv_NoPred_rr06_s0_n113`); offline diagnostic at `tmp/20260509_wm_imagination_test_A1.{json,md}`; design doc `docs/experiments/active/dreamer_diagnosis/DREAMER_CONVENTIONAL_FIXES_BATTERY.md`.
- **Mechanistic interpretation**: zero-init removes startup noise on the reward head's output, so positive rewards (small dense food signals) get learned faster and more accurately. But the residual error on negative rewards is structural — our bins span raw ±20 (`linspace(symlog(-20), symlog(20), 255)`), so values larger than 20 in magnitude saturate at the boundary bin. The death-penalty event of −100 is unrepresentable by construction. Sheeprl + Hafner-published code use `symexp(linspace(-20, 0, ...))` + mirror, giving raw-space bins ±4.85×10⁸.

## Decisions and actions
- **Promote `zero_init_reward_critic: true` to default**. Both `configs/models/dreamer_v3.yaml` and `configs/models/dreamer_v3_rr06.yaml` ship with the key set to `true`. The `false` branch is preserved for legacy reproduction.
- **Queue twohot bin-range fix as the next candidate** — selected via residual-error mechanistic match, not via the plan §4 H2 prescription (which named candidate #1 GRU reset gate). The asymmetry pos 49% vs neg 14% pointed specifically at the bin-range deviation. (See sibling insight `20260510_2241_residual_error_pattern_directs_next_fix` for the methodology lesson.)
- **Cell Z2 launched** (`dreamer_twohotrng_NoPred_rr06_s0_n113`, n113:0, WandB `q66macky`, group `dreamer_paper_canonical_bins`, ~2.5h wall-clock). Cumulative test: rr=0.0625 + zero-init + paper-canonical bins. Verdict pending.
- **Concept doc updated**: §6 item 2 (twohot bin range) reframing already in place from Phase 4 reviewer findings; §6 items 27-30 added from sheeprl comparison. Z1 verdict folded into `docs/develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md` Verification Report.

## Open questions and follow-ups
- Will Z2 (paper-canonical bins) close the gap below the H1 threshold of 0.15, or further partial-fix? If partial, candidates #1 (GRU reset gate), #3 (critic-EMA regularization), or some combination remain on the queue.
- Is the survival lift (106 → 115) a real signal or single-seed noise? A 3-seed sweep at the cumulative-fix configuration would tighten the estimate but is lower priority than the next candidate test.
- Does the same fix sequence transfer to the predator task (Cell A2 / hypervigilance)? Cell A2 reached survival 27 with mean_advantage = −0.19; if Z2 closes the NoPred gap, the natural next test is Z2-equivalent on predator. NoPred has no death-penalty event, so the bin-range fix's largest impact will only manifest on the predator task.
- 5 other dreamer configs (`dreamer_v3_probe.yaml`, `dreamer_v3_curriculum*.yaml`, `dreamer_v3_probe_cont10.yaml`, `neuromodulated_dreamer_v3.yaml`) lack the new mandatory keys (`zero_init_reward_critic`, `paper_canonical_twohot_bins`). They will fail with `ValueError` if used. Out-of-scope cleanup; flagged for senior-developer.

## References
- Cell Z1 plan + Verification Report: `docs/develop/active/diagnosis/dreamer_zero_init_reward_critic_fix.md`. Commits: `cff1faf` (plan), `a88002f` (code), `dad09e1` (diary), `c4b7633` (verification report).
- Offline diagnostic outputs: `tmp/20260510_211404_wm_imagination_test_Z1.{json,md}`; A1 baseline at `tmp/20260509_wm_imagination_test_A1.{json,md}`.
- Concept doc: `docs/project/concepts/dreamer_v3_implementation.md` — §6 has the deviation list (item 2 = twohot bin range, items 27-30 = sheeprl-comparison candidates including #4 zero-init).
- Sibling insights from this session: `20260510_2240_reference_impl_compare_only_act_intersections` (the rule for using sheeprl as comparator), `20260510_2241_residual_error_pattern_directs_next_fix` (the methodology that selected bin-range over GRU-reset-gate).
- Parent insights: `20260509_1534_wm_reward_head_localized_failure_a1` (the offline diagnostic that originally localized the failure to the reward head) and `20260509_1535_conventional_fixes_battery_verdict_predator_refute` (the conventional-fixes battery that motivated the offline diagnostic).
- Sheeprl reference: `tmp/sheeprl/sheeprl/algos/dreamer_v3/agent.py:1170-1180` for the `uniform_init_weights(0.0)` pattern.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume 4ae5401e-443a-406e-93e1-5431c67b5f59` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
