---
id: 20260622_1747_dreamer_srl_single_config_budget_source
date: 2026-06-22
time: "17:47"
folder: cluster_ops
tags: [dreamer, training_runner, learned_lesson, decision]
summary: "In dreamer_srl single-config (--env-config) mode the training budget is read from env_cfg.training.* (seeded from configs/train/default.yaml: episodes=100, checkpoint_frequency=10000, log_interval=10), NOT from the agent config — so a budget-free scene config silently runs 100 episodes and exits in ~20s. Pass --episodes/--log-interval on the CLI; checkpoint_frequency has no CLI flag (env-config only). WandB showing 200000 is a display artifact (agent config logged over env config)."
related: ["20260529_1825_log_interval_anchored_rows_per_session", "20260622_1746_dreamer_srl_recompile_storm_done_count"]
session_origin: claude_code
session_label: "dreamer_srl basic-curriculum: 5-run crash → recompile-storm diagnosis + fix"
importance: medium
status: settled
valid_until: null
confidence: high
supersedes: []
raw_source: claude_data/.claude/projects/-media-nas01-projects-Interoceptive-AI-grid-world-pain/3c11010b-dbfb-4d17-af58-5b86e54d6815.jsonl
raw_completeness: full
---

# dreamer_srl single-config budget comes from env_cfg, not the agent config

## Key conclusion
When launching `dreamer_srl_main.py` in **single-config mode** (`--env-config <scene>`), the episode budget + checkpoint cadence are read from `env_cfg.training.*`, which is seeded from `configs/train/default.yaml` (`episodes=100`, `checkpoint_frequency=10000`, `log_interval=10`) and then deep-merged with the scene config. The **agent config's `training:` block is NOT merged into `env_cfg`** — only its `algo.*` keys are read. So a scene config with no `training:` block silently runs the **100-episode default and exits in ~20 s with 0 gradient steps**. The agent preset (`01_food_only_buf256k.yaml`) carries `training: {episodes: 10M, checkpoint_frequency: 200000, log_interval: 2000}`, which *looks* like it sets the budget but does nothing in single-config mode. WandB's config panel shows the agent's 200000/2000 because the run spreads `agent_config_dict` over `env_config_dict` when logging — a **display artifact**; the live `checkpoint_frequency` (read from `env_cfg`, `dreamer_srl_main.py:487`) stays at the env value.

## Evidence, measurements, facts
- First launch of the 5 basic-curriculum runs: all exited at exactly 100 episodes, 0 checkpoints, 0 gradient steps, ~15-23 s each.
- Budget resolution is `dreamer_srl_main.py:405-441`: `env_cfg = get_default_config()` → merge `configs/train/default.yaml` (+eval/viz) → merge `load_env_config(--env-config)`; `agent_cfg = Config.load_yaml(--agent-config)` is loaded separately and only `algo.*` is read.
- CLI overrides: `--episodes` (priority over `training.episodes`) and `--log-interval` (CLI > agent > env) WORK. There is **no `--checkpoint-frequency` flag** — checkpoint cadence is whatever `env_cfg` resolves to (default 10000 episodes, confirmed in the saved `models/env_config.yaml`).
- `checkpoint_frequency` is in EPISODES (`dreamer_srl_main.py:854`). For terminate-on-convergence runs (which end far below the from-scratch 10×10 reference of ~47k episodes), the default **10000 is correct** — it yields ~5 checkpoints; the agent preset's 200000 would yield 0–1 (model-loss risk). So do NOT copy the agent preset's 200000 into a scene config.
- The basic scene configs are intentionally budget-free shared files (rPPO consumes them with its own CLI flags; parity tests guard them) — the fix was CLI flags `--episodes 10000000 --log-interval 2000`, NO config edits.
- Curriculum mode (`--configs-dir` + `--continual-schedule`) is unaffected: the schedule YAML defines the budget directly, which is why the T1–T4 runs never hit this.

## Decisions and actions
- Standing launch recipe for dreamer_srl single-config production runs: `--episodes 10000000 --log-interval 2000`, leave checkpoint_frequency at the env default (10000), do not add a `training:` block to shared scene configs.
- Also captured as an operational rule in the built-in auto-memory (`feedback_dreamer_srl_single_config_budget.md`) so the training-runner obeys it on every invocation.
- Complements (does not contradict) the log_interval cadence value insight [[20260529_1825_log_interval_anchored_rows_per_session]] — that one fixes the VALUE (2000), this one fixes WHERE the budget is read from and how to override it.
- Same-session companion: the recompile-storm root cause [[20260622_1746_dreamer_srl_recompile_storm_done_count]].

## Open questions and follow-ups
None.

## References
- `src/algorithms/dreamer_srl/dreamer_srl_main.py:405-441` (budget resolution), `:487` + `:854` (checkpoint_frequency read, episodes unit).
- Built-in auto-memory: `~/.claude/.../memory/feedback_dreamer_srl_single_config_budget.md`.
- Raw conversation: synced via `./sync-agent-data.sh claude push`. To read on another node: `./sync-agent-data.sh claude pull`, then `claude --resume 3c11010b-dbfb-4d17-af58-5b86e54d6815` or `python scripts/claude_jsonl_to_md.py <jsonl> /tmp/<id>.md`.

<!-- BACKLINKS — auto-generated by scripts/regen_memory_graph.py; do not edit -->
## Backlinks
- [[20260622_1746_dreamer_srl_recompile_storm_done_count]] (dreamer_diagnosis, 2026-06-22) — dreamer_srl crashed all 5 basic-curriculum runs because the per-step env-reset p
<!-- END BACKLINKS -->
