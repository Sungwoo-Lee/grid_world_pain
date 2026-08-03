---
title: "Integrate dreamer_srl into train.py as the single training entry point (thin dispatch, 3-gate proof)"
topic: dreamer
status: active
created: 2026-07-24
last_updated: 2026-08-03
---

# Integrate dreamer_srl into train.py — single entry point, thin dispatch, three proof gates

> **Status**: PLANNED, rev 4 (reconciled with the 2026-07-28/29 dreamer-stack commits — **dreamer now has resume**, so one scope premise changed and Gate 1 gains a resume leg; one decision-delta flagged for user confirmation at approval — see "Binding-decision status" below)
> **Opened**: 2026-07-24
> **Related**: [[DREAMER_SRL_TRAIN_PY_DIVERGENCE_MATRIX]] (full investigation notes — read it first) · `docs/reviews/review_dreamer_integration_plan_20260724.md` (adversarial review, verdict SOUND-WITH-CORRECTIONS — all 12 findings folded into this revision) · [[DREAMER_SRL_EVAL_TELEMETRY_FIX]] (Track C semantics that must not regress) · `docs/reviews/diagnosis_20260723/findings_train_entry.md` · `docs/reviews/diagnosis_20260723/findings_dreamer_main.md`

**Revision history**
- rev 1 (2026-07-24): initial plan.
- rev 2 (2026-07-25): adversarial-review corrections — Gate 1 gains a curriculum leg (**Gate 1b**) + stage-0 mutation fix (finding 1); WandB algorithm label made spec-driven, two seam-line edits enumerated (finding 2); Gate 1 harness A-side flags fixed (finding 3); reconciled with landed `fff6055` — `--eval-config` enumerated, `--experiment-eval` references deleted, gate key named (finding 4); shim handles `--episodes M --total-steps N` (finding 5); minors 6–12 (mask seed row, pre-seam locals + delete-range precision, log_code spec field, Phase 4 test/bench drivers + preserved imports, shim env-var timing + GPU shim smoke, C9 path form, Gate 3 flip-target correction).
- rev 3 (2026-07-25): round-2 review nits (`docs/reviews/review_dreamer_integration_plan_rev2_20260724.md`, verdict APPROVED-WITH-NITS, 12/12 round-1 closure confirmed) — N1 stage-0 anti-pollution mechanism named (dreamer-conditional deepcopy at train.py:563); N2 propagation-loop extension peek-gated so rPPO curriculum dumps are untouched; N3 per-key conditional Gate 1 mask + `last_ckpt_episode` in the parity dump; N4 signal-handler cite corrected to train.py:448-449; N5 `args.seed` substitution restated as an exhaustive 6-site sweep. Two review observations adopted: drop the stale `total_steps` "final-log print" comment with its lines; loud-fail `--total-timesteps` + `--configs-dir` on the dreamer dispatch.
- rev 4 (2026-08-03): reconciled with four dreamer-stack commits landed 2026-07-28/29 (`e834ec1`, `ad8929a`, `90cd4c0`, `fb54bc0`) — **dreamer gained `--load-checkpoint`/`--load-episode` resume** (with Adam optimizer state in checkpoints per D-017, and `e834ec1`'s stage-rebuild-on-resume fix), so decision 5's "resume out of scope (dreamer has none today)" premise is obsolete: resume becomes a pass-through feature with a new **Gate 1c** resume-parity leg (decision-delta flagged for user confirmation); `ad8929a`'s curriculum retention guard rides inside the seam (no new keys/spec fields); `90cd4c0` re-baselines dreamer cadence (5,000 episodes, keep-ALL = `1000000`) — Gate 2 sizing + compat/matrix rows updated; `fb54bc0`'s real 3-stage curriculum configs adopted as the Gate 1b fixture shape (per-stage differing frequencies). Eval seed now comes from `testing.seed`, so the `args.seed` sweep is 4 sites, not 6. All seam/label/anchor line numbers renumbered (dreamer_srl_main.py 2,112 → 2,425 lines; train.py 2,472 → 2,523). New **live-run constraint** section (a 3-stage Dreamer curriculum run is live on the legacy entry point). Gate structure: still three gates; Gate 1 = 1a + 1b + **1c**.

---

## Context

Today the project trains its two live agents through **two different front doors**: the recurrent-PPO agent through `train.py` at the repo root, and the Dreamer world-model agent through its own script, `src/algorithms/dreamer_srl/dreamer_srl_main.py`. The two scripts parse different flag names, layer the YAML config files in slightly different orders, name their result folders differently, and have accumulated separate gotchas (for example, a Dreamer run that forgets to pass `--episodes` silently exits after 100 smoke-test episodes, and Dreamer has no `--checkpoint-frequency` flag at all). Every launch, every launch-script edit, and every "why does the Dreamer flag not exist?" moment pays a tax for this split.

This plan merges them: `train.py` becomes the **single entry point**. It gains a branch that recognizes `agent.algorithm: dreamer_srl`, resolves flags and configs by the one shared convention, and then **delegates to Dreamer's existing training loop unchanged** — no rewrite of Dreamer internals, because the just-landed eval-telemetry semantics (videos on the right clock, the two eval estimators kept apart — commit `39f851b`) live inside that loop and must not be re-implemented. The old Dreamer script keeps working as a deprecation shim.

Because a wrong config merge could **silently change what a Dreamer run trains on**, the integration is gated three times before anyone's launch script changes: Gate 1, a parity harness proving both entry points produce bit-identical training telemetry from the same seed and config — in **two legs**, one single-config (Gate 1a) and one exercising a 2-stage curriculum (Gate 1b, added after review because curriculum is where the two entry points resolve configs most differently); Gate 2, one real same-seed GPU A/B run compared on WandB; Gate 3, only then the shim + launch-script flip.

Six user decisions are **binding requirements** for this plan (thin dispatch; unified CLI/config; deprecated shim; the three gates; feature scope = match today exactly; A/B node picked at validation time via gpu-status).

### Binding-decision status (rev 4 — one flagged delta)

Five of the six decisions are unchanged. **Decision 5 has an obsolete premise**: it excluded resume "(dreamer has none today)" — but as of commit `e834ec1` (2026-07-28) dreamer **does** have resume (`--load-checkpoint` / `--load-episode`: networks + counters + moments + Adam optimizer state restored; replay buffer refilled with the restored policy; curriculum resume rebuilds the correct stage's env, mirroring the rPPO H2 fix). Decision 5's *headline* — "feature scope = match today exactly" — now **requires** resume to pass through the integrated path, and decision 3 (shim keeps working) requires the shim to forward it. This plan therefore treats resume as **in scope: pass-through + a Gate 1c resume-parity leg** (no new resume machinery is built — the existing loop code is delegated like everything else). **Flagged for user confirmation at approval.** Fallback if the user prefers the literal exclusion: the dispatch rejects `--load-checkpoint` for dreamer until a follow-up plan — accepting that legacy resume invocations through the Gate 3 shim would then break, i.e., decisions 3 and 5 cannot both be honored literally. NMN hooks and `eval_stats_num_envs` batching remain excluded as before.

### Live-run constraint (rev 4)

A 3-stage Dreamer curriculum training run is **currently live on the legacy entry point** (diary training-start, commit `338a13e`), which makes the legacy entry point the **de-facto reference implementation until that run completes**. Binding constraints on every phase:
- **No phase may disturb the live run.** Edits to `dreamer_srl_main.py` / `src/` do not affect the already-running process (Python loads code at process start), and the run dumped its own config copies at startup — but a crash-relaunch must be able to reproduce the original invocation, so behavior-affecting changes to the files it launched from are held conservatively.
- **Config edits wait**: the 19-agent-config `agent.algorithm` rename (File Change 3) and any edit to `configs/train/dreamer_srl.yaml`, the `configs/continual/basic_01_02_03_dreamer.yaml` schedule, or its three stage env configs **land only after the live run completes** (or after the user explicitly clears them). Phase 1's code changes may proceed; its config commit is the deferred part.
- **Gate 3 waits**: the launch-template flip, module-docstring update, and shim conversion happen only after the live run completes.
- **Gate 2 node selection avoids the live run's node** (checked via `gpu-status` at validation time, per decision 6).
- The live run's results dir is never read-written by any harness or smoke (parity dumps go to tmp/test dirs).

## Analysis

The full divergence inventory — every place the two entry points do the same job differently, with a verdict per row — is in the companion doc [[DREAMER_SRL_TRAIN_PY_DIVERGENCE_MATRIX]]. Summary of what the investigation established:

1. **The extraction seam is clean after one mechanical refactor.** `dreamer_srl_main.main()` has a natural waist right after config resolution (line ~557): everything below consumes only `env_cfg`, `agent_cfg`, `schedule`, and ~15 scalar CLI fields. Extracting lines ~557–2108 into `run_dreamer_training(spec, env_cfg, agent_cfg, schedule)` **in the same module** preserves every existing test import and all loop behavior. The only entangled pieces are `wandb.init`/`define_metric` and results-dir creation, which stay inside the extracted body (their *mechanics* are Track-C-load-bearing) while their *values* enter via the spec.
2. **Dispatch must fail loudly.** train.py's known open bug "unknown `agent.algorithm` spins the training loop forever" (KNOWN_BUGS B1–B4, OPEN) is fixed as part of this change with an explicit whitelist.
3. **The algorithm identity string must change in configs.** All 19 `configs/models/dreamer_srl/*.yaml` currently declare `agent.algorithm: "DreamerV3"` (17 files) or nothing (2 files) — but train.py hard-rejects `DreamerV3` as the archived NNX stack. Dispatching on `dreamer_srl` requires a one-line edit in all 19 configs.
4. **train.py's open P1 (lifetime-average episode metrics on the DQN/DRQN/PPO branches) does not reach the Dreamer branch** — dispatch returns before train.py's loop, and the Dreamer loop's own episode path always takes the two-level logging route under the unified layering. Documented, not inherited.
5. **Offline eval constrains the run-dir layout**: `scripts/eval/eval_rollout.py` requires `models/agent_config.yaml` and `checkpoints/<episode>` in Dreamer run dirs. Both are preserved.
6. **The experiment-eval refactor has now landed** (`fff6055`): the opt-in lives in the evaluation config layer, selectable via the new `--eval-config` flag (train.py:415, loaded through `load_env_config`, missing path → loud ValueError), and the config gate is `experiment.during_training.enabled` (train.py:668), which already raises for any non-rPPO algorithm when enabled. There is **no `--experiment-eval` CLI flag** — earlier drafts of this plan referenced one; all such references are removed in rev 2. Line numbers in this plan reference the rev-4 baseline (train.py 2,523 lines, dreamer_srl_main.py 2,425 lines — see Analysis point 8; train.py anchors shifted +1 vs `fff6055`, so this gate now reads at :669).
7. **Curriculum is the highest-divergence surface and train.py's rPPO path mutates the schedule in place** (review finding 1): dreamer rebuilds each stage config from scratch (defaults + 4 fixed files + stage YAML, `dreamer_srl_main.py:102-134`) while train.py deep-copies the fully-merged base (train.py:193-201, which includes the logger layer and the selectable `--eval-config` layer). Worse, train.py then **aliases** `config = schedule.stage_configs[0]` (train.py:564) and merges the agent config + CLI overrides (`wandb.*`, `tag`, `seed`) into that same object — polluting stage 0 relative to stages 1..N. The Dreamer dispatch must work from **copies** (File Change 2) and Gate 1 gains a curriculum leg (**Gate 1b**) so a curriculum-side resolution divergence can no longer pass all gates silently.
8. **Rev-4 stack reconciliation** (four commits landed 2026-07-28/29; baseline now dreamer_srl_main.py 2,425 lines / train.py 2,523 lines, train.py anchors +1: peek :508, stage-0 alias :564, propagation loop :556-561, algorithm read :650, probe gate :669, signal registration :449-450):
   - `e834ec1` — **dreamer resume exists** (see Binding-decision status): flags at dreamer_srl_main.py:497-506, restore block §12b at :1369+ (inside the extracted body → two new spec fields), curriculum stage-rebuild-on-resume fix at :1437+; also `RollingWindow` now emits **partial windows** on the interval and logs `Episode/_window_n` — a shared `src/utils/rolling_logging.py` change hitting both entry points identically, so parity is unaffected.
   - `ad8929a` — startup **retention guard** (warn-only) for curriculum cadence × `training.max_checkpoints_to_keep`; sits inside the extracted body, uses existing keys, no spec impact. Same commit region also switched checkpoint-eval seeding to the **`testing.seed`** evaluation key (`eval_seed`, :712) — the eval passes no longer consume `args.seed`.
   - `90cd4c0` — dreamer cadence re-baselined: `training.checkpoint_frequency: 5000`, `max_checkpoints_to_keep: 1000000` (keep-ALL; literal large int because dreamer's `get_mandatory` rejects `null`, unlike rPPO's `null` convention — an enumerated asymmetry).
   - `fb54bc0` — **real 3-stage curriculum configs** exist (`configs/continual/basic_01_02_03_dreamer.yaml`, boundaries [20k, 80k, 100M], per-stage frequencies [2000, 5000, 5000], + three stage env configs); Gate 1b's fixture mirrors this shape, and these files join the live-run config-freeze list.

## Compatibility table — every behavior delta vs today's `dreamer_srl_main.py` launch

Applies to a Dreamer run launched **via train.py** (and, after Gate 3, via the shim). "Preserved by shim" = the shim injects the old default so historical invocations behave identically.

| # | Behavior | Today (direct dreamer_srl_main.py) | Via train.py | Class |
|---|---|---|---|---|
| C1 | Env-config flag | `--env-config` | `--config` | rename; shim translates |
| C2 | Agent-config flag | `--agent-config` | `--agent_config` (+ register `--agent-config` alias) | rename; shim translates |
| C3 | Env-step budget flag | `--total-steps` or `--total-timesteps`; env-step mode when `--episodes` absent | `--total-timesteps` only; env-step mode via `--episodes 0 --total-timesteps N` | shim translates `--total-steps N` (no `--episodes`) → `--episodes 0 --total-timesteps N` |
| C4 | Seed when `--seed` omitted | 0 (argparse default) | 42 (config `seed`) | **delta**; preserved by shim (injects `--seed 0`) |
| C5 | Episode-budget fallback key | `training.episodes` | top-level `episodes` | same value (100 smoke placeholder) in both defaults; real runs pass `--episodes` regardless |
| C6 | `--checkpoint-frequency` | flag does not exist | works (sets `training.checkpoint_frequency`) | gotcha fixed, additive |
| C7 | `agent.algorithm` in configs + wandb.config | `"DreamerV3"` in configs (or absent); wandb.config **hardcodes** `algorithm`/`agent.algorithm` to `"DreamerV3"` inside the loop's payload (dreamer_srl_main.py:829, :865 — the `setdefault` at :865 unconditionally overwrites the agent dict's value) | `"dreamer_srl"` in configs AND in wandb.config — the two payload lines become **spec-driven** (`spec.wandb_algorithm_label`; enumerated seam edits, File Change 1). Legacy direct launches keep the `"DreamerV3"` label until Gate 3 (their spec passes the old value) | **delta** for train.py-launched runs only: WandB dashboard filters on `DreamerV3` won't match them; old runs + pre-flip direct launches unchanged |
| C8 | WandB kwargs | project/name from CLI; entity hardcoded; no group/job_type; ambient auth; no code upload | config-driven (logger/wandb.yaml) + CLI; `name = --wandb-name or tag`; group/job_type set (job_type wiring from commit `6e82fc3`); `wandb_login` helper; `wandb.disabled` honored; `log_code` uploads .py files via a new `spec.wandb_log_code` field + call site inside the seam (File Change 1 — legacy spec passes False) | same entity/project values today → no data moves; new metadata fields |
| C9 | Results dir | `results/JAX_DreamerSRL/<ts>_<wandb-run-name>` built as an **absolute** path under the hardcoded project root (:913); `tmp/JAX_DreamerSRL_<ts>` when `--no-wandb` | `results/JAX_DreamerSRL/<ts>_<tag>` always (parent name pinned, NOT `JAX_dreamer_srl`), built **relative** like rPPO — identical on disk when CWD is the repo root, which the `run_command.py` launch convention guarantees | **delta** in name component + path form + no tmp/ diversion; internal layout (`models/env_config.yaml`, `models/agent_config.yaml`, `checkpoints/<ep>`) unchanged |
| C10 | GPU memory env | caller-exported vars only; JAX preallocates by default | `XLA_PYTHON_CLIENT_PREALLOCATE=false` + `--device` supported | **delta**: lower VRAM footprint; Gate 2 watches for allocator-related SPS change |
| C11 | Dumped `models/env_config.yaml` content | no `wandb:` block | contains `wandb:` block (logger layer merged) + CLI overrides written back (seed, checkpoint freq, body flags) | cosmetic + improved self-description |
| C12 | New flags usable for Dreamer | — | `--tag`, `--quiet`/`--debug` (already existed), `--no-satiation`, `--no-overeating-death`, `--device`, `--checkpoint-frequency`, **`--eval-config`** (selects the evaluation config layer — a capability direct launches never had; its keys, e.g. `testing.auto_render_after_eval`, feed the env_cfg the loop already reads, so pass-through is the natural unified behavior) | additive |
| C13 | rPPO-only flags with a Dreamer config | n/a | `--wandb-resume-id`, `--num-steps`, `--hidden-size`, `--lr`, `--log-accumulate`, `--profile` → **ValueError** (`--wandb-resume-id` stays rejected: dreamer resume starts a fresh WandB run, matching today). The behavior-probe battery is gated by config (`experiment.during_training.enabled`, train.py:669), which already raises for any non-rPPO algorithm when true — no CLI flag exists for it since `fff6055` | loud-fail. `--load-checkpoint` moved OUT of this list in rev 4 — see C20 |
| C14 | Dreamer-only flags | `--buffer-device`, `--legacy-grad-loop` | same flags on train.py, passed through; ValueError with non-dreamer algorithms | parity |
| C15 | Ctrl-C / SIGTERM | raw KeyboardInterrupt kill | unchanged (train.py's graceful-stop handlers registered only on the non-dreamer path) | no delta — deliberately preserved so SIGTERM is never swallowed |
| C16 | Everything inside the loop | training math, PRNG stream, replay/Ratio machinery, prefill, curriculum swap, checkpoint payload/cadence, eval video+stats passes, Track C step clock, episode logging | **delegated, unchanged except three enumerated one-line substitutions** (two wandb algorithm-label lines → spec-driven, C7; one log_code call added, C8) | Gate 1 proves the numerics |
| C17 | Behavior-probe eval battery (12 avoidance conditions) | not available | still not available; loud error if enabled via `--eval-config` preset | unchanged per decision 5 |
| C18 | `training.eval_stats_num_envs` | read-but-dead | still read-but-dead | unchanged per decision 5 (OPEN reminder row stays) |
| C19 | Curriculum stage-config construction (`--configs-dir`) | each stage rebuilt from scratch: defaults + 4 fixed files (train/default, train/dreamer_srl, evaluation/default, visualization/default) + stage YAML (dreamer_srl_main.py:102-134); stage 0 never mutated after build | each stage = deep-copy of the fully-merged unified base (incl. logger layer + selected `--eval-config` layer) + stage overlay; env-level CLI overrides (seed, body flags, checkpoint frequency) propagate to **every** stage; agent config merged into **no** stage. The rPPO path's in-place stage-0 aliasing/pollution (train.py:564 + :589 + CLI block) is explicitly bypassed on the dreamer branch (File Change 2) | **delta** in stage-config content (extra `wandb:` block, written-back overrides — same class as C11) + dumped `stage_XX_*.yaml` content; **Gate 1b** proves stages resolve equal after masking and stage 0 is not polluted |
| C20 | Resume (rev 4 — new) | `--load-checkpoint <run>/checkpoints [--load-episode N]` since `e834ec1`: restores networks + counters + moments + Adam optimizer state (D-017; checkpoints now 44-50 MB), refills the replay buffer with the restored policy, rebuilds the correct curriculum stage's env | same flags via train.py, passed through the spec into the unchanged §12b restore block; `--load-episode` is a **new train.py flag** (dreamer-only, ValueError otherwise); train.py's existing `--load-checkpoint` keeps its rPPO meaning on the rPPO path | pass-through (decision-delta flagged in "Binding-decision status"); proven by **Gate 1c**. The rev-1 claim "dreamer checkpoints drop optimizer momentum (OPEN bug)" is obsolete — fixed by D-017 |

---

## Implementation Plan

### Design

**Thin dispatch, values-in / mechanics-stay.** train.py owns everything *up to and including* deciding what the run is (flags, config layers, budget, seed, tag, WandB kwargs, results path); the Dreamer module owns everything that *executes* the run (env/agent build, loop, logging clocks, checkpoints, evals). The boundary object is a frozen dataclass:

```python
# src/algorithms/dreamer_srl/dreamer_srl_main.py
@dataclass(frozen=True)
class DreamerRunSpec:
    seed: int
    num_envs: int
    episodes: int                 # 0 = env-step mode
    total_timesteps: int
    results_dir: Optional[str]    # explicit path, or None -> legacy naming convention
    wandb_enabled: bool
    wandb_kwargs: dict            # project/entity required; name/group/job_type optional (None = omit)
    wandb_algorithm_label: str    # value for the payload's algorithm identity fields (finding 2):
                                  #   train.py passes "dreamer_srl"; legacy main() passes "DreamerV3"
    wandb_log_code: bool          # train.py passes True; legacy main() passes False (finding 8)
    quiet: bool
    debug: bool
    buffer_device: str            # "cpu" | "gpu"
    legacy_grad_loop: bool
    load_checkpoint: Optional[str]  # rev 4: dreamer resume (e834ec1) — path to <run>/checkpoints
    load_episode: Optional[int]     # rev 4: checkpoint episode to resume from (None = latest)
    log_interval: Optional[int]   # legacy --log-interval passthrough
    env_config_path: Optional[str]     # provenance (wandb payload)
    agent_config_path: str
    configs_dir: Optional[str]
    continual_schedule_path: Optional[str]
    parity_dump: Optional[str] = None  # Phase 2 harness hook; None in production
```

`run_dreamer_training(spec, env_cfg, agent_cfg, schedule)` is the extracted body of today's `main()` from the "mandatory agent config reads" (line ~574 @ rev-4 baseline) to the end (line ~2425), with `args.X` → `spec.X` and with num_envs/budget resolution **removed** (the callers supply `episodes`/`total_timesteps`/`num_envs`; exact line range in File Change 1). The resume block (§12b, :1369+) and the retention guard (`ad8929a`, :667-705) sit inside the extracted body and are delegated unchanged. Until Gate 3, `dreamer_srl_main.main()` keeps its config resolution unchanged and its argparse unchanged **except one additive `--parity-dump` flag** (needed so the Gate 1 harness can drive the legacy A-side — review finding 3; "byte-for-byte argparse" was rev-1 wording and is retired). It merely packages a spec, so its behavior is otherwise today's, and it serves as the A-side of both gates. train.py's dispatch is the B-side. After Gate 3, `main()` becomes a flag-translating deprecation shim onto train.py.

The extracted body is *byte-untouched except an enumerated substitution list* (File Change 1): spec-field renames, the num_envs/budget deletion, two spec-driven wandb algorithm-label lines, one log_code call, two pre-seam locals re-derived, and the dormant parity hooks. Nothing else.

Both `ContinualSchedule` dataclasses are field-identical; train.py passes its own schedule object into the seam (duck-typed — add a one-line comment in both files pinning the field contract).

### File Changes — Phase 1 (dispatch + unified CLI + seam)

#### 1. `src/algorithms/dreamer_srl/dreamer_srl_main.py`

- Add `DreamerRunSpec` (above) near the top of the module.
- Extract `run_dreamer_training(spec, env_cfg, agent_cfg, schedule)` — the body of `main()` from line ~574 (`learning_starts_cfg = agent_cfg.get_mandatory(...)`) through the end of the loop/final-log (line ~2425), **in this same module** so the existing test imports (`_advance_episode_counters`, `_eval_scalar_prefix`, `_reset_terminal_step_data`, `ContinualSchedule`, `_build_continual_schedule`, `_load_stage_env_cfg`, `Player`, `main`) stay valid. Mechanical substitutions only (line numbers @ rev-4 baseline, dreamer_srl_main.py 2,425 lines):
  - `args.seed` → `spec.seed` — **4 sites, exhaustive** (was 6 in rev 3; the two eval passes' `seed=` now read `eval_seed` from the `testing.seed` evaluation key at :712 — a config-driven read inside the seam, no spec field needed): `np.random.seed` (:733), `PRNGKey` (:734), the WandB payload `"seed": args.seed` (:956), the console banner (:1585). Blanket rule for the whole extraction: **every remaining `args.*` reference below the seam becomes `spec.*` — developer verifies completeness with `grep -n "args\." ` over the extracted body (expect zero hits)**,
  - `args.quiet/debug/buffer_device/legacy_grad_loop/log_interval` → spec fields; **rev 4: `args.load_checkpoint`/`args.load_episode` → `spec.load_checkpoint`/`spec.load_episode`** (resume block §12b at :1369+ and the curriculum stage-rebuild-on-resume block at :1437+ are inside the extracted body and delegated unchanged),
  - `num_envs`/`episodes`/`total_timesteps` → spec fields. **Precise deletion range** (renumbered): remove line 583 (`num_envs = args.num_envs or ...`) and lines 594–618 (`env_step_override` + the budget if/elif chain + the env-step-mode WARNING print + `training.episodes` fallback + `total_steps` alias); **line 593 (`env_max_steps = env_cfg.get_mandatory(...)`) stays** — the banner uses it. The env-step-mode WARNING (:605-613) moves with the budget logic to **both callers** (train.py `_run_dreamer_srl` step 2 and legacy `main()`), so the announcement survives on both paths. The `total_steps` alias (:617-618) with its stale "final-log print" comment is dropped (zero consumers below the seam). Keep `learning_starts, prefill_steps = derive_prefill(...)` (:588) inside, fed by `spec.num_envs`,
  - **pre-seam locals re-derived inside the function**: `import os as _os` (:514) and `_project_root` (:515) are consumed below the seam (results dir + dumps) — re-import/re-define them at the top of `run_dreamer_training`,
  - WandB init (§9): `use_wandb = spec.wandb_enabled`; `wandb.init(**{k: v for k, v in spec.wandb_kwargs.items() if v is not None}, config=wandb_config)`; payload's `args.env_config/agent_config/configs_dir/continual_schedule` → spec provenance fields; **two enumerated label edits (finding 2, renumbered)**: line **944** `"algorithm": "DreamerV3"` → `"algorithm": spec.wandb_algorithm_label`, and line **980** `wandb_config.setdefault("agent", {})["algorithm"] = "DreamerV3"` → `... = spec.wandb_algorithm_label` (this line unconditionally overwrites the agent dict's value — the spec keeps that overwrite but makes the value honest); **one added call (finding 8)**: after a successful `wandb.init`, `if spec.wandb_log_code: run.log_code(".", include_fn=lambda p: p.endswith(".py"))`; **define_metric block unchanged**,
  - Results dir (§9b, ~:1019+): `if spec.results_dir: results_dir = spec.results_dir` else keep today's convention verbatim,
  - Everything else (config dumps, checkpoint manager, retention guard (:667-705), resume restore, loop, curriculum swap, eval passes incl. the `testing.seed` eval-seed read, Track C logging) **untouched**.
- `main()`: config resolution stays exactly as today; argparse gains **exactly one additive flag, `--parity-dump PATH`** (finding 3 — the Gate 1 harness must be able to drive this side; no `--checkpoint-frequency` is added here, the harness sets the cadence in its env YAML fixture instead). Replace the fall-through body with spec construction + `run_dreamer_training(spec, env_cfg, agent_cfg, schedule)`. Spec values reproduce today's behavior: `wandb_kwargs = {"project": args.wandb_project, "entity": "sungwoolee", "name": args.wandb_name, "group": None, "job_type": None}`, `wandb_algorithm_label="DreamerV3"`, `wandb_log_code=False`, `load_checkpoint=args.load_checkpoint`, `load_episode=args.load_episode`, `results_dir = args.results_dir` (None → legacy naming), `episodes`/`total_timesteps` from the existing resolution block (which stays in `main()`, together with the env-step-mode WARNING).
- Parity hook: when `spec.parity_dump` is set (never in production), (a) at startup write `<dump>/resolved.json` = `{spec fields (minus paths that embed timestamps), env_cfg.to_dict(), agent_cfg.to_dict(), schedule boundaries/names or null}`; (b) per iteration append to in-memory lists: `iter_num, policy_step, total_episodes_completed, cumulative_grad_steps, n_grad_steps(_scan), buffer.filled_size`, and — only on iterations where `last_losses` was updated — each loss value as float64; (c) at loop exit write `<dump>/telemetry.npz` plus `<dump>/final.json` = final counters **including `last_ckpt_episode` and the list of episode counts at which a checkpoint fired** (round-2 nit N3 — gives the telemetry a checkpoint-cadence signal, so a `training.checkpoint_frequency` resolution bug fails numeric parity too, not only the config diff), PRNG `key` as a list, per-module parameter checksums (`float64` sum + L2 norm over `jax.tree.leaves(nnx.state(m, nnx.Param))` for world_model/actor/critic/target_critic) and the same for `moments`. All of it inside `if spec.parity_dump is not None:` guards — zero work otherwise. (The per-iteration float() of losses forces a host sync; acceptable because parity runs are ≤ a few hundred iterations.)

#### 2. `train.py`

- **Algorithm whitelist (bug fix for the OPEN "unknown algorithm hangs forever" row).** Immediately after `algorithm = config.get_mandatory('agent.algorithm')` (line 650 @ rev-4 baseline):
  ```python
  KNOWN_ALGORITHMS = ("RecurrentPPO", "DQN", "DRQN", "PPO", "dreamer_srl")
  # existing DreamerV3 archived-stack error stays, with an added hint:
  #   "...If you meant the live world-model agent, set agent.algorithm: dreamer_srl."
  if algorithm not in KNOWN_ALGORITHMS:
      raise ValueError(f"Unknown agent.algorithm {algorithm!r}. Valid: {KNOWN_ALGORITHMS}. "
                       "(Previously this fell through to an infinite no-op training loop.)")
  ```
- **Dreamer train-defaults peek-merge**, parallel to the rPPO peek (line 507 @ `fff6055`): if `Config.load_yaml(args.agent_config).get("agent.algorithm") == "dreamer_srl"`, merge `configs/train/dreamer_srl.yaml` right after `train/default.yaml`. (Do the YAML peek once, reuse for both gates.)
- **env_cfg snapshot for the dreamer branch**: when the peeked algorithm is `dreamer_srl`, deep-copy the merged config right **before** the agent-config merge (line ~589) into `dreamer_env_cfg` (the Dreamer loop requires separate env/agent configs — divergence matrix §2.1). Apply the env-level CLI overrides to it after the main CLI-override block (seed, `--no-satiation`/`--no-overeating-death` body keys, `training.checkpoint_frequency`) so the dumped `env_config.yaml` self-describes. The `--eval-config`-selected evaluation layer (train.py:515-525) is already merged at this point and flows through unchanged (compat C12).
- **Curriculum stage isolation (finding 1 / compat C19; mechanism per round-2 nit N1)**: the pollution happens in **shared pre-dispatch code** — `config = schedule.stage_configs[0]` (train.py:564) *aliases* stage 0, and the agent merge (:589) + CLI-override block (:592-605) then mutate it in place before the dispatch point is ever reached. Protecting `dreamer_env_cfg` alone would NOT stop the aliased `stage_configs[0]` from being polluted and later dumped by the seam as a divergent `stage_00_*.yaml`. The concrete edits:
  - **line 564 itself becomes a deep copy when the peeked algorithm is `dreamer_srl`**: `config = copy.deepcopy(schedule.stage_configs[0])` (dreamer-conditional — the rPPO path keeps today's aliasing, which stays declared-out-of-scope);
  - `dreamer_env_cfg` is then the pre-agent-merge deep copy as before (belt and braces — two copies, one per concern);
  - env-level CLI overrides (seed, body flags, `training.checkpoint_frequency`) are propagated to **every** `schedule.stage_configs[i]` by extending the existing `--no-satiation`/`--no-overeating-death` propagation loop (train.py:556-561) — **gated on the dreamer peek** (round-2 nit N2): extended unconditionally it would add written-back keys to rPPO curriculum `stage_XX_*.yaml` dumps, an rPPO-side delta this plan does not authorize (and one Checkpoint 2's single-config rPPO smoke would not catch);
  - the agent config is merged into **no** stage config (stage YAMLs stay env-only, matching dreamer's convention and the dumped `stage_XX_*.yaml` contract).
- **Continual guard** (line ~694 area): allow `{"RecurrentPPO", "dreamer_srl"}`; message updated.
- **Dispatch point**: after the behavior-probe config gate (`experiment.during_training.enabled`, read at train.py:669 — it already raises for any non-rPPO algorithm when enabled, so a dreamer run with an `--eval-config` preset that turns it on fails loudly with no extra code):
  ```python
  if algorithm == "dreamer_srl":
      _run_dreamer_srl(args, config, dreamer_env_cfg, schedule)
      return
  ```
- **`_run_dreamer_srl(args, config, env_cfg, schedule)`** (new function, ~80 lines):
  1. Reject rPPO-only flags loudly (compat row C13): any of `--wandb-resume-id`, `--num-steps`, `--hidden-size`, `--lr`, `--profile`, non-None `--log-accumulate` → `ValueError("--X is not supported for agent.algorithm=dreamer_srl")`. (No `--experiment-eval` flag exists post-`fff6055`; the probe battery's config gate at :669 covers that path. `--eval-config` is pass-through, C12. **Rev 4: `--load-checkpoint` is NO LONGER rejected** — it passes through as dreamer resume, compat C20; the new `--load-episode` flag is dreamer-only and rejected for every other algorithm.)
  2. Resolve, using the same expressions as the rPPO path. First a guard (round-2 review observation adopted): `--total-timesteps` together with `--configs-dir` → `ValueError` — the legacy entry point rejects that combination (dreamer_srl_main.py:517-522) and silently honoring it as an env-step cap would be a behavior delta. Then: `episodes = schedule.episode_boundaries[-1] if schedule else (args.episodes if args.episodes is not None else config.get_mandatory('episodes'))` (persist via `config.set('episodes', ...)` for the dump, mirroring line ~693); `num_envs = args.num_envs or env_cfg.get_mandatory('training.num_envs')`; `total_timesteps = args.total_timesteps or (episodes * env_cfg.get_mandatory('environment.max_steps') * num_envs)`; `seed = args.seed if args.seed is not None else config.get_mandatory('seed')`; `tag = args.tag or config.get_mandatory('tag')`.
  3. WandB kwargs exactly as the rPPO block resolves them (project/entity/group from config + CLI, `job_type` mandatory from config per commit `6e82fc3`, `name = args.wandb_name or tag`); `wandb_enabled = WANDB_AVAILABLE and not args.no_wandb and not config.get_mandatory('wandb.disabled')`.
  4. `results_dir = args.results_dir or os.path.join("results", "JAX_DreamerSRL", f"{ts}_{tag}")` — parent name **pinned** to the existing `JAX_DreamerSRL`; relative form deliberate (compat row C9 — identical on disk under the repo-root-CWD launch convention).
  5. Build `DreamerRunSpec` (incl. `buffer_device`, `legacy_grad_loop` from the two new train.py flags; `load_checkpoint=args.load_checkpoint`, `load_episode=args.load_episode` — rev 4 resume pass-through; `log_interval=args.log_interval`; `wandb_algorithm_label="dreamer_srl"`; `wandb_log_code=True`; provenance paths) and call `run_dreamer_training(spec, env_cfg, Config.load_yaml(args.agent_config), schedule)`.
- **New CLI flags**: `--buffer-device {cpu,gpu}` (default "cpu"), `--legacy-grad-loop` (default False), `--load-episode N` (default None; rev 4), `--parity-dump PATH` (hidden/undocumented in help beyond one line; Phase 2). Add `--agent-config` as an alias of `--agent_config`. All four dreamer-only flags → `ValueError` if the resolved algorithm is not `dreamer_srl` (no silent ignore).
- **Signal handlers**: move the `signal.signal(SIGINT/SIGTERM, ...)` registration (train.py:449-450 @ rev-4 baseline) to after the dispatch point (just before "2. Setup Results Directory") so the Dreamer branch keeps today's raw-KeyboardInterrupt semantics and SIGTERM is never swallowed by a handler the Dreamer loop can't see (compat row C15).

#### 3. `configs/models/dreamer_srl/*.yaml` — 19 files

- 17 files: `agent.algorithm: "DreamerV3"` → `"dreamer_srl"` (update the adjacent "WandB filter" comment).
- 2 files (`agent_xs.yaml`, `01_food_only_smoke.yaml`): **add** the `agent:` block with `algorithm: "dreamer_srl"`.
- No new config *keys* are introduced anywhere in this plan (Configuration Protocol: nothing to add to the schema; value-only edits).

#### 4. `tests/algorithms/dreamer_srl/test_train_dispatch.py` (new)

- Unknown `agent.algorithm` → ValueError (regression test for the OPEN hang row; must FAIL on pre-fix train.py — construct by driving `train.main()` with a temp agent config declaring `algorithm: "Bogus"` and asserting the raise happens within seconds, not a hang: the pre-fix behavior loops forever, so implement as "raises ValueError before entering the loop" using a monkeypatched guard or a subprocess with timeout).
- `DreamerV3` → archived-stack ValueError containing the `dreamer_srl` hint.
- dreamer_srl + each rPPO-only flag → ValueError; non-dreamer + `--buffer-device gpu` → ValueError.
- dreamer_srl dispatch smoke (CPU, `--no-wandb`, tiny budget): asserts run dir layout `models/env_config.yaml`, `models/agent_config.yaml`, `checkpoints/` exists — the offline-eval layout contract (risk R4).

### File Changes — Phase 2 (parity harness → **Gate 1**)

#### 5. `tests/algorithms/dreamer_srl/test_entry_parity.py` (new) + fixtures

Fixtures under `tests/algorithms/dreamer_srl/fixtures/` (new): one single-config env YAML carrying `training.checkpoint_frequency: 20` (Gate 1a), two tiny stage env YAMLs + one schedule YAML (Gate 1b). No files under `scripts/` are touched (no SCRIPTS_DEPENDENCY_MAP update needed).

**What is compared** (decision 4's list, mapped to concrete fields):
| Requirement | Concrete artifact |
|---|---|
| losses | per-iteration `last_losses` values (all keys: world-model/actor/critic components) from `telemetry.npz` |
| policy_step | per-iteration `policy_step` array |
| episode counters | `total_episodes_completed` array + final value |
| replay-ratio state | `cumulative_grad_steps`, `n_grad_steps` per iteration, `buffer.filled_size` per iteration |
| PRNG-sensitive values | final PRNG `key`, per-module parameter checksums (sum + L2), `moments` checksum |
| checkpoint cadence | `last_ckpt_episode` + episode counts of every checkpoint save (from `final.json`; nit N3) |
| config resolution | `resolved.json` deep-diff of env_cfg/agent_cfg/spec |

**How captured — Gate 1a (single-config leg)**: both entry points run as subprocesses with `--no-wandb --parity-dump <tmpdir>` and identical `{env config, agent config, --episodes 40, --seed 7, --num-envs 4}`. **The legacy A-side is invoked only with flags its argparse actually has** (finding 3): `--env-config/--agent-config/--episodes/--num-envs/--seed/--no-wandb/--quiet/--parity-dump` — nothing else. The checkpoint cadence comes from the harness's **env YAML fixture** (`training.checkpoint_frequency: 20` in a small test-fixtures env config), not from a CLI flag, so neither side needs `--checkpoint-frequency`. Agent config = `01_food_only_smoke.yaml` (small `learning_starts` so the run crosses prefill → train); the 40-episode budget with cadence 20 crosses ≥1 checkpoint+eval (video pass on, stats off — keeps runtime ~minutes). Exercises budget resolution, prefill, the grad scan, episode accounting, checkpointing, and the eval path.

**How captured — Gate 1b (curriculum leg — review finding 1)**: same harness, curriculum mode. Fixtures: a 2-stage toy schedule (two tiny env YAMLs differing in a non-architectural knob, e.g. food count; `episode_boundaries: [30, 60]`, `checkpoint_frequencies: [10, 20]` — **per-stage frequencies deliberately differ**, mirroring the shape of the real 3-stage schedule `configs/continual/basic_01_02_03_dreamer.yaml` ([2000, 5000, 5000] per `fb54bc0`/`90cd4c0`) so a per-stage-cadence resolution divergence is also caught). A-side: `dreamer_srl_main.py --configs-dir <fixtures> --continual-schedule <sched> --agent-config ... --seed 7 --num-envs 4 --no-wandb --parity-dump ...` (all existing legacy flags); B-side: the train.py equivalents. The parity dump's `resolved.json` is extended to include **every** `schedule.stage_configs[i].to_dict()` plus boundaries/frequencies/names. The 60-episode run crosses the stage boundary, so the telemetry covers the swap iteration (buffer clear, player reset, counter wipe, `checkpoint_frequency_active` switch) and post-swap training.

**How captured — Gate 1c (resume leg — rev 4, forced by `e834ec1`)**: reuses Gate 1a's fixture and one completed 1a run's dir as the shared source checkpoint. Both entry points resume from **the same checkpoint** (`--load-checkpoint <1a-run>/checkpoints --load-episode 20`, legacy side with legacy flag spellings) for a further 20-episode budget with `--parity-dump`, same seed. This exercises the restore path (networks + counters + moments + Adam state), the `train_start_iter` buffer-refill gate, and — because the dump records `resolved.json` + telemetry as usual — proves train.py's resume flag plumbing feeds the unchanged §12b block identically. (A curriculum-resume variant is deliberately NOT a separate leg: the stage-rebuild-on-resume code is exercised by `e834ec1`'s own regression test `tests/algorithms/dreamer_srl/test_continual_resume_rebuild.py`, which is entry-point-agnostic once retargeted at the seam; Gate 1b + 1c jointly cover the resolution surfaces.)

**Pass criteria (Gate 1 = 1a AND 1b AND 1c)**:
1. *Config parity*: `agent_cfg` dicts byte-identical. `env_cfg` — and in 1b **every per-stage config dict, stage 0 included** — identical after masking the **enumerated accepted-delta keys only**, where every written-back key is masked **conditionally: only when the corresponding CLI flag was actually passed in that harness invocation** (round-2 nit N3 — same rule the `body.*` entry already had): `wandb.*` block and top-level `tag` (unconditional — structural deltas), top-level `seed` only because the harness passes `--seed 7` (finding 6), top-level `episodes` only because it passes `--episodes`, `training.checkpoint_frequency` **not masked** in the planned invocations (no `--checkpoint-frequency` flag is passed — cadence comes from the fixture YAML, so both sides must agree on this key and a resolution bug in it now fails the config criterion), `body.*` only if a body flag was passed (it is not). The mask applies **uniformly across stages** — a diff present in stage 0 but not stage 1 (the aliasing-pollution signature) is a FAIL even for masked keys' *siblings*; concretely, after masking, `stage_0` and `stage_i` must diff from their A-side counterparts by the *same* key set. Any other diff = FAIL and must either be fixed or promoted to a new compat-table row before the gate can pass.
2. *Numeric parity, CPU (`JAX_PLATFORMS=cpu`)*: **bit-identical** — `np.array_equal` on every telemetry array, exact equality on all integer counters, PRNG key, and checksums; in 1b this includes the swap-iteration index and post-swap counters; in 1c the restored counters (`policy_step`, episodes, grad-steps), the post-refill buffer fill, and the final checksums after the resumed segment. Bit-identity is the target and is achievable because both processes execute the *same* `run_dreamer_training` function with identical inputs on a deterministic backend.
3. *Numeric parity, GPU (manual, once, on the Gate 2 node before the long run; 1a leg only)*: integer counters + PRNG key **exact**; losses/checksums within relative tolerance `1e-6`. Documented justification: across two *processes*, cuDNN/XLA autotuning may select different kernel algorithms for the same HLO, which can perturb float reductions at the ULP level; it cannot perturb control flow, counters, or the PRNG stream — so those stay exact-match. If GPU is also bit-identical (likely), record that and keep the tolerance clause as fallback.

Both legs are pytest cases in `test_entry_parity.py` (marked `slow`); the GPU variant is the 1a case parametrized, skipped unless `PARITY_GPU=1`. Note: the legacy A-side curriculum resolution (`_build_continual_schedule`/`_load_stage_env_cfg` in dreamer_srl_main.py) stays in place through Gate 3 precisely because Gate 1b needs it as the reference.

#### 6. Phase 2 also verifies the seam refactor cost

`tests/algorithms/dreamer_srl/bench_sps.py` before/after Phase 1 on one GPU, same config/seed/step budget: the seam extraction + dormant parity hooks must be ≤ noise. >5% SPS drop = discuss; >15% = blocker (project speed policy).

### File Changes — Phase 3 (GPU A/B → **Gate 2**)

No code changes. Gate 2 is deliberately single-config: curriculum-mode config resolution is proven at Gate 1b (CPU bit-identity), and a multi-day GPU curriculum A/B would add cost without adding coverage the gates don't already have. Procedure (executed with `training-runner`, analyzed with `experiment-analyzer`):

1. **Node/GPU selection at validation time via the `gpu-status` skill** (decision 6 — named step, not pre-assigned here). One node, two GPUs, pack-node-first policy; **must avoid the node running the live 3-stage curriculum run** (Live-run constraint).
2. Config family: basic04 M — env `configs/environment/experiment/basic/04-jump_attack_10x10.yaml` (or the family's current canonical file; `experiment-designer` confirms), agent `configs/models/dreamer_srl/01_food_only_M.yaml`, same seed, `--episodes` sized for ≥3 checkpoints at the 5,000-episode cadence (`90cd4c0` re-baseline; ≈15–20k episodes).
3. Run A = legacy invocation (`dreamer_srl_main.py --env-config ... --agent-config ... --seed 42 ...` with `agent.algorithm` already reading `dreamer_srl` — value is inert on the A path). Run B = `train.py --config ... --agent_config ... --seed 42 --tag ...`.
4. **Gate 2 pass criteria** on WandB: survival curves (`Episode/Steps`) and `Eval/MeanLength` overlap within same-seed GPU run-to-run noise (no systematic offset after the first checkpoint); `WorldModel/loss_model` trajectories overlap; `Time/sps_env` within ±5% (watches compat row C10's preallocation change); Track C intact on run B — eval videos visible in the WandB media panel, `Eval/*` vs `Eval/video/*` split present, no "dropped row" warnings in the log; run-B dir passes an `eval_rollout.py` smoke restore of one checkpoint.

### File Changes — Phase 4 (shim + flip → **Gate 3**)

#### 7. `src/algorithms/dreamer_srl/dreamer_srl_main.py` — `main()` becomes the shim

- Print a `DeprecationWarning` banner ("launch via train.py; this shim translates flags and will be removed").
- Flag translation (compat rows C1–C4; finding 5): `--env-config`→`--config`; `--agent-config`→`--agent_config`; **`--total-steps N` is always rewritten `--total-timesteps N`** (train.py has no `--total-steps`), and `--episodes 0` is injected **only when `--episodes` is absent** — so the legacy-supported combo `--episodes M --total-steps N` (episode mode with an explicit env-step cap, dreamer_srl_main.py:584-586) survives as `--episodes M --total-timesteps N`, which train.py resolves with identical semantics (`args.total_timesteps or derived product`). If both legacy aliases appear, `--total-timesteps` wins, matching the legacy `args.total_timesteps or args.total_steps` precedence. Inject `--seed 0` when `--seed` absent (preserves the historical default); `--load-checkpoint`/`--load-episode` forward verbatim (rev 4 — same spellings on both sides); forward everything else verbatim; unknown leftover flags → error (argparse handles).
- Invoke the integrated path in-process: **rewrite `sys.argv` BEFORE `import train`** (finding 10 — train.py sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` and the `--device` platform vars at module import time; the dreamer module chain has already imported jax, which is safe only while nothing initialized the backend — verified true today, no module-level array creation in the chain, but the ordering must be preserved and asserted). Then call `train.main()`. The old config-resolution body in `main()` is deleted; `run_dreamer_training`, `DreamerRunSpec`, `ContinualSchedule`, `_build_continual_schedule`, `_load_stage_env_cfg`, all helpers and the module import surface **stay** (finding 9 — `test_continual_schedule.py:38` imports the builder; removal of the then-dead legacy builders is deferred to a named post-integration cleanup, out of this plan's scope).
- Guard: the shim asserts the peeked agent config declares `agent.algorithm: dreamer_srl` and otherwise errors with the config-edit instruction (protects anyone launching with a stale private config still saying `DreamerV3`).
- Phase 4 smoke (CPU): legacy-form invocation through the shim reaches the training loop AND asserts `os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"` took effect post-`import train`. Additionally **one short GPU shim smoke** (~2 min, any free low-tier GPU) after the flip: confirms preallocation is off (nvidia-smi memory footprint) and the device pick honors `CUDA_VISIBLE_DEVICES` — this is the named shim-on-GPU coverage decision (finding 10): a full GPU A/B of the shim is NOT run (Gate 2 already proved the integrated path; the shim adds only flag translation + import ordering, which the smoke covers).

#### 8. Legacy-driver test/bench consumers (finding 9 — three, not one)

- `tests/algorithms/dreamer_srl/test_eval_telemetry_wandb.py` — currently drives `dreamer_srl_main.main()`; retarget at `run_dreamer_training` (build the spec directly) so it keeps testing the telemetry mechanics rather than flag translation.
- `tests/algorithms/dreamer_srl/test_eval_video_smoke.py` (:26, :79-86) — subprocess-drives `dreamer_srl_main.py` with legacy flags (`--env-config`, `--total-steps`, no `--episodes`). **Named decision: keep it as-is post-flip** — it becomes a shim-integration test exercising translation + the full train.py path end-to-end, which is coverage we want; the developer re-checks its 10-minute timeout budget against the shim path's startup cost and bumps it if needed.
- `tests/algorithms/dreamer_srl/bench_sps.py` (:57) — `TRAINER_SCRIPT` pins `dreamer_srl_main.py`; post-flip that would benchmark shim+train.py, not the loop. Update it to drive the train.py form directly (so SPS numbers stay comparable across the flip); the shim's own overhead is startup-only and needs no benchmark.
- Add one small shim-translation test (`test_shim_translation.py` or a case in the dispatch test): legacy argv → the exact translated argv, including the `--episodes M --total-steps N` combo and the both-aliases precedence case.

#### 9. Launch-command flip target (**only after Gate 3**; corrected per finding 12)

Grep confirms **no `.sh` file in the repo currently invokes `dreamer_srl_main.py`** — `train_command-agent.sh`'s dreamer-named blocks are archived NNX-era train.py invocations, and real dreamer launches are ad-hoc `run_command.py` command lines. The flip is therefore even lower-risk than rev 1 assumed; the real deliverables are (a) the canonical new command template below (for `training-runner` and future `train_command-agent.sh` blocks), and (b) updating the two places that teach the legacy CLI: the `dreamer_srl_main.py` module docstring (lines 10-18) and any howto/README doc found by `grep -rl "dreamer_srl_main" docs/` at implementation time. `train_command-agent.sh` itself needs no retroactive edit (it is user-edited and currently modified by a parallel session — add fresh blocks only, never rewrite).
```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/<...>.yaml \
  --agent_config configs/models/dreamer_srl/<...>.yaml \
  --episodes <N> --seed <S> --tag "<run-tag>" \
  --wandb-name "<run-name>"
```
(no more `--env-config`/`--agent-config`/`--total-steps`; `CUDA_VISIBLE_DEVICES` still fine, or `--device gpu:<i>`.)

#### 10. Post-flip bookkeeping (same change set)

- Ask `bug-curator` to: (a) close/update the "unknown agent.algorithm hangs forever" portion of the B1–B4 row (fixed by the whitelist, with the regression test path), (b) add a row-note that dreamer_srl now launches via train.py.
- Update the user auto-memory gotcha note `feedback_dreamer_srl_single_config_budget.md` (the `--checkpoint-frequency` gap and `env_cfg.training.*` budget quirk are superseded) — flag to the user; auto-memory is user-owned.
- `docs/environment/SCRIPTS_DEPENDENCY_MAP.md`: no `scripts/` file is added/moved/renamed/deleted by this plan, so no map update is required; developer double-checks nothing under `scripts/` gained/lost a caller (eval_rollout.py's inputs are unchanged by design).
- CONFIG_GUIDE / `02_config_schema.md`: no schema change (no new keys), no update required.

## Checkpoints (for `developer`, during implementation)

1. After the seam extraction (before any train.py change): full Dreamer test suite green (`test_episode_metrics`, `test_continual_schedule`, `test_prefill`, `test_terminal_step_data_reset`, `test_eval_telemetry`, `test_eval_telemetry_wandb`, `test_lax_scan_train`, `test_eval_video_smoke`) + a direct `dreamer_srl_main.py` CPU smoke run completes with unchanged console banner and run-dir layout.
2. After the train.py dispatch: dispatch smoke (File Change 4) green; an rPPO smoke via train.py unchanged (dispatch must be invisible to rPPO).
3. Phase 2: parity tests green on CPU for **all three legs** (1a single-config, 1b curriculum, 1c resume — bit-identical); config-diff reports attached to the Implementation Report with every masked key listed, per stage for 1b.
4. bench_sps before/after numbers recorded in the Implementation Report (same GPU, config, seed, ≥ 2k iterations post-warmup).
5. Phase 4: shim translation test green (incl. the `--total-steps` combo cases); full suite green; legacy-form CPU smoke through the shim reaches the training loop and the `XLA_PYTHON_CLIENT_PREALLOCATE` assertion passes; GPU shim smoke done.
6. Commits: one per phase minimum; Phase 1 config edits (19 YAMLs) in their own commit — **deferred until the live curriculum run completes** (Live-run constraint); Phase 1's code can land and be Gate-1-tested before that commit because the dispatch peeks the algorithm from the agent config passed on the harness CLI (the harness can use a temp copy of one agent config carrying `dreamer_srl` until the rename lands).

## Risk register — what could silently change Dreamer training, and which gate catches it

| # | Risk | Mechanism | Caught by |
|---|---|---|---|
| R1 | Config-resolution drift (layer order, logger block, CLI override landing on the wrong key) silently changes what a run trains on | train.py builds env_cfg differently than dreamer_srl_main did | **Gate 1** config-parity deep-diff with an explicit accepted-mask; any unlisted diff fails the gate |
| R2 | Seam extraction perturbs the PRNG stream or seeding (e.g., `np.random.seed` moved/dropped, key split order changed) | training becomes non-reproducible vs today | **Gate 1** bit-identity on losses + final PRNG key + param checksums |
| R3 | Track C telemetry regression (step clock, `Eval/` vs `Eval/video/` split, video upload) from touching the WandB init/logging | dashboards silently lose rows/videos again | keep-dreamer mechanics (init + define_metric + eval blocks byte-untouched inside the seam); existing `test_eval_telemetry*`; **Gate 2** dashboard checklist |
| R4 | Run-dir layout drift breaks offline eval (`eval_rollout.py` needs `models/agent_config.yaml` + `checkpoints/<ep>`) | finished runs can't be re-evaluated | layout assertion in the dispatch smoke test (Phase 1); **Gate 2** eval_rollout restore smoke |
| R5 | Silent flag swallowing (an rPPO flag ignored on the Dreamer branch, or vice versa) | user believes an option applied when it didn't | loud-fail whitelists both directions + dispatch unit tests (Phase 1) |
| R6 | Perf regression from the seam / dormant parity hooks / preallocation change | slower training on every future Dreamer run | bench_sps A/B (Phase 2, ±5% policy) + **Gate 2** `Time/sps_env` ±5% |
| R7 | `agent.algorithm` rename breaks WandB dashboard filters and any tooling matching `DreamerV3` | new runs invisible in old views — **and without the seam label edits (File Change 1, lines 829/865) the wandb.config would keep saying `DreamerV3` while the dumped configs say `dreamer_srl`, a worse mixed-identity state** | spec-driven label (finding 2 fix); enumerated (compat C7); Gate 2 analyst confirms new runs filterable; grep for `DreamerV3` consumers in `scripts/`/`analysis/` during Phase 1 |
| R8 | train.py's SIGTERM handler swallows kill signals for a loop that never checks `stop_requested` | remote kills hang; zombie GPU jobs | handler registration moved below dispatch (compat C15) + code-review item |
| R9 | Shim translation subtly changes an old launch (seed default, env-step mode, `--episodes M --total-steps N` combo) | historical commands stop reproducing or die on argparse | shim injects legacy defaults + always-rewrite of `--total-steps` (C3/C4, finding 5); shim translation unit test incl. the combo case (Phase 4) |
| R10 | ~~In-flight experiment-eval refactor collides~~ **RESOLVED in rev 2**: the refactor landed (`fff6055`); plan reconciled — `--eval-config` enumerated (C12), gate key named (`experiment.during_training.enabled`), stale `--experiment-eval` references removed | — | rev 2 reconciliation (finding 4) |
| R11 | Curriculum stage-config divergence — from-scratch vs clone-of-base construction, plus train.py's in-place stage-0 pollution — silently changes what each stage trains on and what `stage_XX_*.yaml` records | a curriculum run's stage 0 carries agent/wandb/tag keys its siblings lack; stage envs differ from today's direct launches | dispatch works from copies + uniform override propagation (File Change 2); **Gate 1b** per-stage config parity with the uniform-mask rule + swap-crossing telemetry bit-identity |
| R12 | (rev 4) Integration work disturbs the **live** 3-stage curriculum run on the legacy entry point, or resume-path plumbing through train.py silently diverges from the legacy resume | a crash-relaunch of the live run fails to reproduce its invocation; or a train.py-resumed run restores differently than a legacy-resumed one | Live-run constraint section (config-edit freeze incl. the `fb54bc0` curriculum configs, Gate 3 deferred, Gate 2 node avoidance, results-dir isolation); **Gate 1c** same-checkpoint resume parity; `test_continual_resume_rebuild.py` retargeted at the seam stays green |

## Test plan / regression tests per phase

- **Phase 1**: new `test_train_dispatch.py` (incl. the must-fail-pre-fix unknown-algorithm regression test); full existing Dreamer suite; one rPPO CPU smoke (no behavior change on the rPPO path); layout assertion.
- **Phase 2**: `test_entry_parity.py` CPU, all three legs — Gate 1a single-config, Gate 1b 2-stage curriculum, Gate 1c same-checkpoint resume (CI, slow-marked); GPU parametrization manual (1a only); bench_sps A/B; `test_continual_resume_rebuild.py` stays green through the seam refactor.
- **Phase 3**: no code — Gate 2 WandB checklist above, result recorded in this doc by `experiment-analyzer`.
- **Phase 4**: shim translation test (incl. `--episodes M --total-steps N` and both-aliases precedence); retargeted `test_eval_telemetry_wandb.py`; `bench_sps.py` retargeted at the train.py form; `test_eval_video_smoke.py` kept as the shim-integration test (timeout re-checked); full suite; legacy-form CPU smoke through the shim with the `XLA_PYTHON_CLIENT_PREALLOCATE` assertion + one short GPU shim smoke.

## Verification checklist (senior-developer, per phase)

- [ ] Phase 1 diff-stat proportionate: dreamer_srl_main.py ≈ net-zero logic (extraction), train.py + ~150, 19 one-line config edits, no unplanned files.
- [ ] `run_dreamer_training` byte-diff vs the old main() body reviewed: only the enumerated mechanical substitutions (spec fields, num_envs/budget deletion with line 577 retained, the two wandb label lines, the log_code call, the two re-derived pre-seam locals, parity guards).
- [ ] Whitelist raises before any env/model construction; DreamerV3 hint present.
- [ ] Signal registration verified below the dispatch return.
- [ ] Curriculum isolation verified: line 564 is a dreamer-conditional `copy.deepcopy` (N1); the :556-561 propagation-loop extension is peek-gated (N2) — an rPPO curriculum dump is byte-identical to pre-change; agent config in no stage config; `grep -n "DreamerV3" src/algorithms/dreamer_srl/dreamer_srl_main.py` returns only comments/spec defaults (no hardcoded payload values).
- [ ] Seam sweep exhaustive: `grep -n "args\." ` over the extracted `run_dreamer_training` body returns zero hits (N5).
- [ ] Gate 1 artifacts reviewed for ALL THREE legs (1a/1b/1c): config-diff masked keys ⊆ compat table, uniform across stages in 1b; CPU bit-identity reports; 1c resumed-counter equality.
- [ ] Live-run constraint honored: no config commit (File Change 3, `configs/train/dreamer_srl.yaml`, `fb54bc0` curriculum files) and no Gate 3 step lands while the 3-stage curriculum run is live; Gate 2 node ≠ live-run node.
- [ ] bench_sps delta verdict recorded (✅/⚠️/❌ per speed policy).
- [ ] Gate 2 checklist all green before Phase 4 is authorized.
- [ ] Phase 4: no launch-script edit before Gate 3 sign-off; sys.argv-before-import ordering in the shim confirmed in the diff; bug-curator updates requested.

## Implementation Report

*(to be filled by `developer`)*

## Verification Report

*(to be filled by `senior-developer` after each phase)*

## Feedback from plan-reviewer (round 3 — rev-4 delta, 2026-08-03)

Verdict: **NOT READY** — one 🔴 Critical in the new Gate 1c spec; full report in `docs/reviews/plan_dreamer_integration_rev4.md`.

1. 🔴 **Gate 1c can pass vacuously.** The episode budget is *absolute* (`while total_episodes_completed < episodes`, dreamer_srl_main.py:1606) and §12b restores `total_episodes_completed = 20`; "resume at `--load-episode 20` for a further 20-episode budget" implemented as `--episodes 20` exits the loop instantly — empty telemetry on both sides satisfies every pass criterion, so the resume gate goes green having exercised neither the `train_start_iter` refill gate nor any resumed training. Fix: state the absolute-budget semantics, pin `--episodes 40`, and add a non-vacuity pass criterion (≥1 post-`train_start_iter` iteration with `last_losses` updated; non-empty arrays). This also makes an Adam-restore divergence observable (Adam state is never directly checksummed; it surfaces only through post-resume parameter updates).
2. 🟡 Curriculum resume (§12c, :1437 — `schedule is not None`) never fires in any gate: Gate 1c is single-config, and the retargeted `test_continual_resume_rebuild.py` bypasses train.py's schedule+resume plumbing in combination. Recommend a cheap 1c variant on the 1b fixture resuming into stage 2 (`--load-episode 40 --episodes 60`, reusing the 1b run dir).
3. 🟡 Live-run tension: Phase 1 rewrites `main()` in the very file a crash-relaunch of the live run would execute; no gate compares refactored-legacy vs pre-refactor legacy (Gate 1 compares two refactored sides). Either defer landing that file's edit, or state that crash-relaunch-on-refactored-code is accepted with Checkpoint 1 as the sole guard.
4. 🟢 Stale anchors from accretion: R7 cites the wandb label lines as "829/865" (correct: :944/:980); Verification-checklist bullet 2 says "line 577 retained" (correct: :593).
5. 🟢 Commit attribution: resume flags landed in `166f261` (e834ec1 added only §12c + RollingWindow); the `testing.seed` switch is `b228117`, not ad8929a. `10000af`/`0030c04` also touched the file in the window — verified no CLI-flag or below-seam `args.*` impact, so no spec change needed.

Verified clean: every rev-4 renumbered anchor spot-checked against HEAD (all correct); the 4-site `args.seed` sweep confirmed exhaustive; config-freeze list complete for the plan's steps (fixtures under `tests/`, `extends:` parents untouched); Gate 1b fixture shape matches `fb54bc0`/`90cd4c0` on disk; C13↔C20↔spec↔shim mutually consistent. Exit condition: fix finding 1 → SOUND WITH CONCERNS.

Reviewed by: plan-reviewer
