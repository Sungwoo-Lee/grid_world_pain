---
id: 20260723_1915_two_level_logging_and_config_layering
date: 2026-07-23
time: "19:15"
folder: config_system
tags: [config, wandb, design, decision, learned_lesson]
summary: "Two-level logging redesign: split SMOOTHING (rolling deque window = noise) from INTERVAL (emission cadence = volume) as independent unit-bearing knobs at episode and step level. Buffer of finished episodes must be UNCAPPED for rPPO (128-step rollout yields up to num_steps*num_envs finishes) though <=num_envs for Dreamer. Added configs/train/dreamer_srl.yaml (unconditional merge, no algorithm gate). Fixed Dreamer num_envs (was CLI-only, silently 1). Regression I caused: moving checkpoint keys out of shared default.yaml broke DQN/DRQN/PPO. Principle: default.yaml documents EVERY field."
related: ["20260710_1632_num_envs_cli_override_config_owns_values", "20260723_1914_dreamer_noise_is_logging_granularity_artifact"]
session_origin: claude_code
session_label: "logging-redesign + predator-mixture + replay-ratio session"
importance: medium
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/3c11010b-dbfb-4d17-af58-5b86e54d6815.jsonl
raw_completeness: full
---

# Two-level logging redesign + train-config layering (dreamer_srl.yaml, num_envs fix, default-documents-all)

## Key conclusion
Redesigned logging so one knob no longer does two jobs. Each level (episode, step) gets two independent, **unit-bearing** knobs: `smoothing_*` (a rolling `deque` window = how many samples averaged per point = NOISE) and `interval_*` (emission cadence = how often a row is written = VOLUME). Keeping the **episode smoothing window equal across algorithms** makes Dreamer and rPPO curves comparable (same #episodes/point); `interval` is per-algo for rows-per-session. In the same arc: created `configs/train/dreamer_srl.yaml` (symmetric with `recurrent_ppo.yaml`), fixed a Dreamer `num_envs` footgun, and hit + fixed a config-layering regression I introduced.

## Evidence, measurements, facts
- **Buffer shapes differ by trainer**: the "finished-this-step" list is <= `num_envs` for Dreamer (1 step/iter) but must be **UNCAPPED for rPPO** — its rollout is a jitted `lax.scan` over `num_steps=128`, episodes are extracted post-hoc, so one iteration yields up to `num_steps x num_envs` finishes. A `num_envs` cap would silently drop rPPO episodes. Only the rolling window (`deque(maxlen=smoothing)`) is capped.
- **dreamer_srl.yaml merged UNCONDITIONALLY** (no `agent.algorithm == "DreamerV3"` gate): two Dreamer agent configs (`agent_xs.yaml`, `01_food_only_smoke.yaml`) declare no `algorithm` key, so a gate would skip their merge and kill them at a `get_mandatory`. rPPO needs its gate only because `train.py` is shared across rPPO/DQN/DRQN/PPO; `dreamer_srl_main.py` isn't.
- **num_envs fix**: Dreamer read `num_envs` CLI-only (`argparse default=1`, then `args.num_envs or config` always short-circuited on the truthy 1) => forgetting `--num-envs` silently trained on 1 env. Changed default to `None` and now reads `training.num_envs`. Proven live: no flag => `num_envs=16`.
- **Regression I caused (4a419eb)**: moving `checkpoint_frequency` / `max_checkpoints_to_keep` OUT of shared `default.yaml` broke DQN/DRQN/PPO — `train.py:2071/628` `get_mandatory`s them on a path shared by all algorithms, and those three never load `recurrent_ppo.yaml`. Verified it raised `ValueError: ... required but missing`; fixed by restoring them to `default.yaml` (e191426).
- Commits: `82e8d96` (two-level logging + new `src/utils/rolling_logging.py`), `4a419eb` (train-config split), `cb5b4da` (self-contained configs + num_envs), `e191426` (regression fix). CP1 (backward-compat) verified byte-identical across four merge chains.

## Decisions and actions
- Universal-smoothing enforcement (single value only in `default.yaml`) was later **relaxed** per user: each algo file declares its own full logging block, so they are self-contained and independently tunable; comparability is now a documented convention, not structural enforcement.
- **Principle (the user's, now in CONFIG_GUIDE.md §7)**: `configs/train/default.yaml` is the canonical REFERENCE — it must contain EVERY field (documentation + fallback for algorithms without an override file). Algo files override; duplication is intentional. This rule directly prevents the checkpoint regression class.
- Shipped values: rPPO `num_envs 128`, Dreamer `num_envs 16`; episode smoothing 5000 both; interval 4000 (rPPO) / 200 (Dreamer); step 100/50 (rPPO), 200/100 (Dreamer).

## Open questions and follow-ups
- **Curriculum warm-up blackout** (unfixed): a curriculum stage-swap clears the rolling window AND resets its count, re-arming the "wait for a full 5000-episode window" gate — so any stage shorter than the smoothing window gets ZERO episode rows (2 of 3 project schedules affected). Fix = allow partial-window emission (mean + sample-count n) after a stage swap.
- 19 `configs/models/dreamer_srl/*.yaml` still declare `env.num_envs: 1` that Dreamer never reads — now actively misleading; route to experiment-designer.

## References
- Plans: `docs/develop/active/refactors/TWO_LEVEL_LOGGING_REDESIGN.md`, `DREAMER_TRAIN_CONFIG_SPLIT.md`. Guide: `docs/environment/CONFIG_GUIDE.md` §7.
- Motivated by [[20260723_1914_dreamer_noise_is_logging_granularity_artifact]]; num_envs echoes [[20260710_1632_num_envs_cli_override_config_owns_values]].
- Raw conversation: synced via `./sync-agent-data.sh claude push`; `claude --resume 3c11010b-dbfb-4d17-af58-5b86e54d6815`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260723_1914_dreamer_noise_is_logging_granularity_artifact]] (cluster_ops, 2026-07-23) — Dreamer's WandB survival curve looking far noisier than rPPO's is a LOGGING-GRAN
<!-- END BACKLINKS -->
