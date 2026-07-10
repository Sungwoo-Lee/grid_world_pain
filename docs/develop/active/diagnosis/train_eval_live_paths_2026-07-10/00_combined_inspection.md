---
title: "Live-Path Inspection — train.py / evaluation.py / dreamer_srl (rPPO, rPPO-NMN, srl, srl-NMN)"
topic: diagnosis
status: active
created: 2026-07-10
last_updated: 2026-07-10
---

# Live-Path Inspection — combined report

## What this is (plain-language entry point)

The user asked for an inspection of the whole training and evaluation entry points across
all four agent variants. The first discovery reshaped the job: the "Dreamer" reachable from
`train.py` turned out to be the **in-house DreamerV3-NNX stack abandoned by the 2026-05-12
PI pivot** — the live Dreamer is the sheeprl port `dreamer_srl`, which has its own entry
point and, notably, **no neuromodulation support at all** (the NMN port promised at the
pivot never happened). Two decisions followed (user-approved): the legacy NNX stack was
**archived** out of the live tree (`fd7af84` — `train.py` shrank by ~500 lines behind a
fail-fast "use dreamer_srl" stub), and the inspection was re-scoped to the LIVE paths:
the rPPO core, the rPPO-NMN wiring (never before dedicated-audited), dreamer_srl's
remaining un-audited surface, and a dreamer_srl-NMN **readiness assessment** in place of
auditing code that does not exist. Three sequential single-agent audits followed
(reports `01`–`03` beside this file).

**Headline:** the live paths are fundamentally sound — the rPPO train→save→eval contract
round-trips faithfully, and the modulated (NMN) forward pass is empirically identical
between training and evaluation. The real debt is **not in the math but at the edges**:
config keys that lie (a dead `lr_critic` means the critic *and modulator* train at 5×
the advertised rate), continual-stage corners (accumulators frozen at stage 0), an eval
metric polluted by small-sample video episodes in dreamer_srl, and the fact that **the
live Dreamer has no offline behavioral evaluation at all**.

---

## 1. What was done (sequence)

| Step | Outcome | Record |
|---|---|---|
| 0 | Ground truth: NNX abandoned 05-12, srl live, srl-NMN nonexistent (git history + pivot memory + grep) | this doc §0 context, auto-memory row |
| 1 | **NNX stack archived** — tracked `src/models/archive/dreamer_v3_nnx/` (never "legacy": gitignored), train.py/evaluation.py stubs, suite at the 8 known-red baseline, rPPO smoke proven untouched | [[archive_plan_dreamer_v3_nnx]] · commits `fd7af84` (+renames in `2a7c6f9`) · registry `84664e4` |
| 2 | rPPO core-path audit | [[01_rppo_core_path]] |
| 3 | rPPO-NMN wiring audit | [[02_rppo_nmn_wiring]] |
| 4 | dreamer_srl gaps + NMN readiness | [[03_dreamer_srl_gaps_and_nmn_readiness]] |

## 2. Findings that should drive action (cross-report roll-up)

### Interpretation hazards for PAST/CURRENT runs
| # | Finding | Where | Why it matters |
|---|---|---|---|
| A1 | **`agent.lr_critic` is dead in every rPPO config** — critic AND modulator train at `lr_actor` (0.0005), not the advertised 0.0001 | `train.py:509,796-803` | Anyone reasoning about NMN learning dynamics from the YAML is misled; affects interpretation of all rPPO/NMN sweeps |
| A2 | **Stage-transition accumulators frozen at stage 0** — per-tag names/widths + BM state never rebuilt | `train.py:945-950 vs 1129-1174` | Live today: `nmn_double_return_stages` stage-2 predator curves silently mislabeled; roster-size changes drop metrics or crash |
| A3 | **dreamer_srl `Eval/Mean*` polluted** — the small-N video pass logs into the same metric as the stats pass | `dreamer_srl_main.py:1573-1605` | The headline survival-steps eval curve mixes two estimators; rPPO does this correctly |
| A4 | **No heteroscedastic loss exists** — `*_het_*` config names mean the perceptual-noise heterogeneity sweep; the modulator trains solely through the shared PPO loss | `recurrent_ppo_trainer.py:184` | Corrects a long-standing naming-driven misreading (this session's own earlier messages included) |

### Silent-failure traps (latent, config-boundary)
| # | Finding | Where |
|---|---|---|
| B1 | Unknown `modulation.type` string silently builds a Multiplicative modulator (no whitelist, no error; probe-verified) | `recurrent_ppo_network.py:211-218` |
| B2 | Missing/typo'd `agent.modulation` block silently trains baseline | `train.py:767-771` |
| B3 | `percept_bias_init: 3.0` dead in all 11 live FiLM configs (code forces 1.0); + `unimodal_overrides`, `fc_layers` siblings unread | `neuromodulator.py:92-95` |
| B4 | Unknown `agent.algorithm` string spins the training loop forever with no error | `train.py:759-922` |
| B5 | Single-config resume pairs checkpoint `h_state` with freshly-reset envs (continual path re-inits; single-config doesn't) | `train.py:1069,735` |

### Capability gaps
| # | Finding |
|---|---|
| C1 | **The live Dreamer has no offline behavioral eval** — `eval_rollout.py` raises `NotImplementedError` for dreamer_srl; only in-driver eval (which never restores) + dream-viz consume its checkpoints. Together with the known no-restore-path row: a finished dreamer_srl run can never be re-evaluated. |
| C2 | **dreamer_srl-NMN does not exist** — readiness memo in [[03_dreamer_srl_gaps_and_nmn_readiness]] Part B: ~7 hook sites, ~600–900 lines (the 187-line `DreamerNeuromodulatorRNN` is already live reusable code); 3 design decisions gate the port — (1) which of the three disjoint optimizers trains the modulator (WM-only is the defensible default; z_reward/temperature heads would be gradient-dead), (2) modulate encoder-only vs +RSSM (comparability with the rPPO-NMN sweep), (3) `mod_h` reset in all three habitats (the observe-scan is-first gating is the likeliest silent bug). Preconditions: the B-row config hardening + DEVIATION_LOG pre-declaration per hook. |

### Lows / nits
Full lists in the per-area reports: orphaned `bm_drive_batch` import, post-resume SPS distortion, `--seed 0` swallowed + cwd-relative default configs + dead wandb plumbing in evaluation.py, no end-of-training checkpoint save (platform-wide), dreamer_srl dead keys (`eval_stats_num_envs`), recordings missing the true-obs noise panel (rPPO got this in `80d3b70`; srl never did), stale/dead metric definitions.

### Explicitly verified sound
rPPO save↔restore↔eval contract (both files derive input_dim/action_dim/breakdown/modulation identically); H1/H2 fixes hold; episode accounting + Site-1 BM ordering; NMN forward parity train↔eval (max |Δlogits| 5e-4 f32); modulator state rides the audited h_state + checkpoint contracts; dreamer_srl eval env fidelity, PRNG hygiene, survival counting; archival left no dangling symbols.

## 3. Recommended next actions (priority order)

1. **Config-hardening package** (B1-B4 + A1's lr_critic wire-or-remove + dead-key cleanup) — small, kills the whole silent-failure class; precondition for the next NMN sweep design.
2. **A2 stage-accumulator rebuild** — before the next continual rPPO run (one config is affected today).
3. **A3 eval-metric split** in dreamer_srl (small) + a decision on **C1** (does the project need offline dreamer eval? if yes, it's a feature package).
4. **C2 NMN-on-srl port** — research-direction decision (PI-worthy: it's the "Dreamer NMN" the paper roadmap assumed exists); the readiness memo is the input.
5. bug-curator records A/B/C rows (done in the same close-out as this doc).

## 4. Reports

[[01_rppo_core_path]] · [[02_rppo_nmn_wiring]] · [[03_dreamer_srl_gaps_and_nmn_readiness]] ·
Archive: [[archive_plan_dreamer_v3_nnx]] · Registry: [[KNOWN_BUGS]] · Prior audits: [[00_combined_diagnosis]] (07-04), [[00_master_comparison]] (parity)
