---
id: 20260508_1444_sameprop_round1_finding_and_confound
date: 2026-05-08
time: "14:44"
folder: hypervigilance
tags: [hypervigilance, learned_lesson, decision]
summary: "Round 1 of hypervigilance/sameProp confirmed RPPO discriminates rabbit (3.77) from predator (4.40) under matched olfactory `properties [0,1,0,0,0]` and the asymmetry grows with training, but the result is hostage to a food/rabbit spawn-area overlap in 01-interoNocicept_sameProp.yaml until Round 2 (Cell C) decouples them."
related: ["20260508_1445_sameprop_discriminating_channels"]
session_origin: claude_code
session_label: "hypervigilance_sameprop_2026-05-08"
importance: high
status: settled
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/b12b1fea-94dc-491e-998b-735c656560cb.jsonl
raw_completeness: full
---

# RPPO discriminates rabbit vs predator under sameProp — hostage to food/rabbit-quadrant confound

## Key conclusion

The user's intuition ("agent is not avoiding rabbits despite identical olfactory properties") is empirically supported by Round 1 of the hypervigilance/sameProp investigation: at ~7.2M episodes the RPPO agent settles at MeanDistRabbit ≈ 3.77 vs MeanDistPredator ≈ 4.40 (uniform-random baseline ≈ 4.68), with the gap *growing* across training (rabbit distance trends from 3.94 → 3.76 while predator distance edges from 4.35 → 4.41), and RabbitHits per episode roughly doubling from 3.7 → 6.6. However, in `01-interoNocicept_sameProp.yaml` the active food spawn rectangles (TL `[[1,1],[5,5]]` and BR `[[6,6],[10,10]]`) coincide exactly with the rabbit spawn rectangles, while the predator patrols full grid — so MeanDistRabbit < MeanDistPredator could be **food-seeking spillover**, not failed olfactory discrimination. Until Round 2 Cell C (`02-sameProp_R2_decoupleFood.yaml`) decouples food and rabbit quadrants, the Round 1 result isolates "co-location-with-food + movement-signature olfactory discriminator", **not olfaction alone**. Future hypervigilance configs must avoid co-locating food with neutral animals.

## Evidence, measurements, facts

- Runs: `hypervigilance-sameprop-relog-seed42_n112_gpu0` (WandB id `rg5nl1ov`, dir `wandb/run-20260507_223118-rg5nl1ov`, 7.21M ep, 15h12m, ~55k SPS) and `hypervigilance-sameprop-relog-seed43_n112_gpu1` (WandB id `6ks4bjbq`, dir `wandb/run-20260507_223541-6ks4bjbq`, 7.28M ep, 15h16m, ~51k SPS). Both stopped early via SIGTERM at user's authorization; verdict was already clear from data.
- Steady-state (last 20%, both seeds agree to ±0.03):
  - MeanDistFood ≈ 2.41
  - MeanDistRabbit ≈ 3.77 (sameProp `[0,1,0,0,0]`)
  - MeanDistPredator ≈ 4.40 (sameProp `[0,1,0,0,0]`)
  - MeanDistHidingPredator ≈ 2.64 (different vector `[0,0,0,0,0]` — out of scope, plus structural corner geometry)
  - RabbitHits/ep ≈ 6.5; PredatorHits/ep ≈ 3.4; HidingPredatorHits/ep ≈ 3.0 (≡ DangerHits, alias preserved).
  - Term_Injury 0.97 → 0.30 (drops); Term_Starvation 0.03 → 0.39 (rises); Term_MaxSteps 0.00 → 0.30 (rises). Steps 27 → 334.
- Trajectory shape (8-window, both seeds): MeanDistRabbit monotonically decreases (3.94 → 3.76) — agent moves *closer* to rabbits over training. RabbitHits monotonically increases (3.7 → 6.6 — near-doubled). PredatorHits is U-shaped: 3.16 → 3.47 (peak) → 3.40 (mild retreat). MeanDistPredator slowly grows (+1.4%).
- Pre-existing logging gap that motivated the rerun: prior 10M-episode run on this config (`wandb/run-20260506_150949-rmw6m8zg`) logged `MeanDistPredator` but not `MeanDistRabbit` or any rabbit-contact counter — the user's claim could not be directly tested without per-entity logging. Logging fix landed in commit `4b55fc6` (Episode/MeanDistRabbit, Episode/RabbitHits, Episode/MeanDistHidingPredator, Episode/HidingPredatorHits added across `core.py`, both trainers, and 5 mirrored aggregation sites in `train.py`; DangerHits preserved as alias for HidingPredatorHits).
- The food/rabbit-quadrant confound is structural in `configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml`:
  - Food TL `[[1,1],[5,5]]` (count=2) ↔ Rabbit TL `[[1,1],[5,5]]` (count=1)
  - Food BR `[[6,6],[10,10]]` (count=2) ↔ Rabbit BR `[[6,6],[10,10]]` (count=1)
  - Predator: full-grid `[[1,1],[10,10]]`
- Predator's MeanDistPredator ≈ 4.40 ≈ random-baseline 4.68 means the agent may not actually be *avoiding* the predator either — it is just on average a fixed distance from a full-grid roamer. So the asymmetry "agent avoids predator more than rabbit" might dissolve into "agent seeks food (which co-locates with rabbits) and the predator is just average-distance".
- Phase-1 channel ranking (from `docs/develop/active/hypervigilance/sameprop_discriminating_channels.md`) is observationally entangled with this confound: ranked-channel #3 "patrol-area asymmetry" (predator full-grid vs rabbit quadrant-locked) is observationally **identical** to the food-quadrant confound — Round 2 must decouple these two channels, not just decouple food/rabbit spawn.

## Decisions and actions

- Round 1 stopped early at user authorization (~7.2M ep / 10M target). GPUs freed for Round 2.
- Formal Mode-B analysis written: `docs/experiments/active/hypervigilance/round1_relog_baseline_analysis.md` (cross-links Phase 1 channel memo, logging plan, and existing-run survey).
- Round 2 designed as a 2x2 over {confound resolved? × movement signature ablated?}:
  - **Cell C (mandatory)** — `02-sameProp_R2_decoupleFood.yaml`: food → TR `[[1,6],[5,10]]` + BL `[[6,1],[10,5]]`; rabbits stay TL+BR; predator stays full-grid. Tests whether MeanDistRabbit < MeanDistPredator persists when rabbit quadrants no longer contain food.
  - **Cell A1** — `02-sameProp_R2_passivePredator.yaml`: predator HUNT structurally disabled (`hunt_stamina_threshold: 1.1` exceeds the `max_stamina: 30` clip → 33 > 30 → unreachable; plus `detection_range: 0` belt-and-braces); patrol+spawn shrunk to TL `[[1,1],[5,5]]` matching rabbit TL.
- Pre-registered thresholds locked in `docs/experiments/active/hypervigilance/sameprop_round2_design.md` §4.2/§4.3:
  - Cell C confirms H₀ (no genuine discrimination, asymmetry was food-confound) if |Δ| ≤ 0.1 cells AND |ΔH| ≤ 1.0/ep at last 10% window. Confirms H₁ (real discrimination) if Δ ≥ 0.3 cells AND ΔH ≥ 1.5/ep stable across last 3 windows.
  - Cell A1 confirms H₀ (movement signature was dominant) if PredatorHits ≈ RabbitHits within 1.0/ep AND |MeanDistΔ| ≤ 0.3. Confirms H₁ (post-contact teaching alone suffices) if PredatorHits < RabbitHits − 1.5/ep stably.
- Round 2 launched on node 112: Cell C (seed 42, cuda:0, WandB `0u266oj5`) and Cell A1 (seed 43, cuda:1, WandB `27svrmhv`). Seed pairing matches Round 1 (42→`rg5nl1ov`, 43→`6ks4bjbq`) for clean same-seed cross-round delta.
- Single seed per cell forced by 2-GPU budget; n=2 reserved for Round 3 only if any Round 2 verdict lands borderline. Round 1's seed42/seed43 agreement to ±0.03 cells justifies starting Round 2 with n=1.
- **Project-rule recommendation for future hypervigilance configs**: do not co-locate food with neutral animals. If the experiment requires neutral animals, place food in predator-only or empty quadrants. Document this in any new sameProp variants.

## Open questions and follow-ups

- Round 2 verdict will close the confound question. If Cell C confirms H₀, the Round 1 finding dissolves to food-seeking spillover and the user's original "rabbit avoidance failure" framing was correct in symptom but wrong in mechanism. If Cell C confirms H₁, the finding survives the confound and Round 2 Cell A1 then attributes the cause (movement signature vs post-contact teaching).
- A future round may want to add finer-grained logging: `MeanDistRabbit_TL` vs `MeanDistRabbit_BR` (per-quadrant) and `QuadrantOccupancy_*` to disambiguate quadrant-vs-class effects. Deferred to Round 3 unless borderline.
- The full-grid predator vs quadrant-locked rabbit asymmetry is not just about distance — it also affects how often each entity is in the agent's olfactory radius. Round 2 Cell A1 partially controls this by shrinking predator's quadrant.

## References

- Doc: `docs/experiments/active/hypervigilance/round1_relog_baseline_analysis.md` (formal Mode-B Round 1 analysis).
- Doc: `docs/experiments/active/hypervigilance/sameprop_existing_run_survey.md` (pre-Round-1 survey + Launch Manifest with actuals).
- Doc: `docs/experiments/active/hypervigilance/sameprop_round2_design.md` (Round 2 cells C + A1 with pre-registered thresholds).
- Doc: `docs/develop/active/hypervigilance/sameprop_discriminating_channels.md` (Phase 1 code investigation, channel ranking).
- Doc: `docs/develop/active/hypervigilance/per_entity_avoidance_logging.md` (logging plan).
- Configs: `configs/experiment/hypervigilance/01-interoNocicept_sameProp.yaml` (Round 1 baseline, food/rabbit confound), `02-sameProp_R2_decoupleFood.yaml` (Round 2 Cell C), `02-sameProp_R2_passivePredator.yaml` (Round 2 Cell A1).
- Commit: `4b55fc6` (per-entity avoidance logging — Episode/MeanDistRabbit, Episode/RabbitHits, Episode/MeanDistHidingPredator, Episode/HidingPredatorHits + DangerHits alias preservation).
- Sibling insight: `20260508_1445_sameprop_discriminating_channels` (the code-derived ranking of discriminating channels referenced in this insight).
- WandB runs: `rg5nl1ov` (R1 seed42), `6ks4bjbq` (R1 seed43), `0u266oj5` (R2 Cell C seed42), `27svrmhv` (R2 Cell A1 seed43).
- **Why a new folder**: closest existing folder is `nmn_diagnosis` ("NMN performance diagnosis findings"), but this Round 1 finding is about plain RPPO (not NMN) under sameProp olfactory matching, and future hypervigilance work — including NMN agents, FiLM gating, multi-modal noise, and Dreamer-vs-RPPO contrasts — will all share the `hypervigilance` topic. Putting this insight under `nmn_diagnosis` would mis-route both this insight and any future RPPO-specific hypervigilance findings.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then either `claude --resume b12b1fea-94dc-491e-998b-735c656560cb` (re-enter the session) or `python scripts/claude_jsonl_to_md.py claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/b12b1fea-94dc-491e-998b-735c656560cb.jsonl /tmp/20260508_1444.md` (one-shot markdown view).

<!-- BACKLINKS — auto-generated by scripts/regen_wiki_graph.py; do not edit -->
## Backlinks
- _no inbound links yet_
<!-- END BACKLINKS -->
