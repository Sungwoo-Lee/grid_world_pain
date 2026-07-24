---
title: "Integrate dreamer_srl into train.py as the single training entry point (thin dispatch, 3-gate proof)"
topic: dreamer
status: active
created: 2026-07-24
last_updated: 2026-07-24
---

# Integrate dreamer_srl into train.py — single entry point, thin dispatch, three proof gates

> **Status**: PLANNED (awaiting user approval → `developer`)
> **Opened**: 2026-07-24
> **Related**: [[DREAMER_SRL_TRAIN_PY_DIVERGENCE_MATRIX]] (full investigation notes — read it first) · [[DREAMER_SRL_EVAL_TELEMETRY_FIX]] (Track C semantics that must not regress) · `docs/reviews/diagnosis_20260723/findings_train_entry.md` · `docs/reviews/diagnosis_20260723/findings_dreamer_main.md`

---

## Context

Today the project trains its two live agents through **two different front doors**: the recurrent-PPO agent through `train.py` at the repo root, and the Dreamer world-model agent through its own script, `src/algorithms/dreamer_srl/dreamer_srl_main.py`. The two scripts parse different flag names, layer the YAML config files in slightly different orders, name their result folders differently, and have accumulated separate gotchas (for example, a Dreamer run that forgets to pass `--episodes` silently exits after 100 smoke-test episodes, and Dreamer has no `--checkpoint-frequency` flag at all). Every launch, every launch-script edit, and every "why does the Dreamer flag not exist?" moment pays a tax for this split.

This plan merges them: `train.py` becomes the **single entry point**. It gains a branch that recognizes `agent.algorithm: dreamer_srl`, resolves flags and configs by the one shared convention, and then **delegates to Dreamer's existing training loop unchanged** — no rewrite of Dreamer internals, because the just-landed eval-telemetry semantics (videos on the right clock, the two eval estimators kept apart — commit `39f851b`) live inside that loop and must not be re-implemented. The old Dreamer script keeps working as a deprecation shim.

Because a wrong config merge could **silently change what a Dreamer run trains on**, the integration is gated three times before anyone's launch script changes: Gate 1, a parity harness proving both entry points produce bit-identical training telemetry from the same seed and config; Gate 2, one real same-seed GPU A/B run compared on WandB; Gate 3, only then the shim + launch-script flip.

Six user decisions are **binding requirements** for this plan (thin dispatch; unified CLI/config; deprecated shim; the three gates; feature scope = match today exactly, resume/NMN/eval-batching excluded; A/B node picked at validation time via gpu-status).

## Analysis

The full divergence inventory — every place the two entry points do the same job differently, with a verdict per row — is in the companion doc [[DREAMER_SRL_TRAIN_PY_DIVERGENCE_MATRIX]]. Summary of what the investigation established:

1. **The extraction seam is clean after one mechanical refactor.** `dreamer_srl_main.main()` has a natural waist right after config resolution (line ~557): everything below consumes only `env_cfg`, `agent_cfg`, `schedule`, and ~15 scalar CLI fields. Extracting lines ~557–2108 into `run_dreamer_training(spec, env_cfg, agent_cfg, schedule)` **in the same module** preserves every existing test import and all loop behavior. The only entangled pieces are `wandb.init`/`define_metric` and results-dir creation, which stay inside the extracted body (their *mechanics* are Track-C-load-bearing) while their *values* enter via the spec.
2. **Dispatch must fail loudly.** train.py's known open bug "unknown `agent.algorithm` spins the training loop forever" (KNOWN_BUGS B1–B4, OPEN) is fixed as part of this change with an explicit whitelist.
3. **The algorithm identity string must change in configs.** All 19 `configs/models/dreamer_srl/*.yaml` currently declare `agent.algorithm: "DreamerV3"` (17 files) or nothing (2 files) — but train.py hard-rejects `DreamerV3` as the archived NNX stack. Dispatching on `dreamer_srl` requires a one-line edit in all 19 configs.
4. **train.py's open P1 (lifetime-average episode metrics on the DQN/DRQN/PPO branches) does not reach the Dreamer branch** — dispatch returns before train.py's loop, and the Dreamer loop's own episode path always takes the two-level logging route under the unified layering. Documented, not inherited.
5. **Offline eval constrains the run-dir layout**: `scripts/eval/eval_rollout.py` requires `models/agent_config.yaml` and `checkpoints/<episode>` in Dreamer run dirs. Both are preserved.
6. **In-flight collision**: an uncommitted parallel-session refactor is moving experiment-eval keys (`training.experiment_eval_*` → `evaluation.experiment.*`) in train.py + configs. The `developer` must start Phase 1 **after that lands**, and re-locate the experiment-eval gate accordingly. Line numbers in this plan reference v3.0 HEAD `385b92f`..`8afb961` and may shift by a few lines.

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
| C7 | `agent.algorithm` in configs + wandb.config | `"DreamerV3"` (or absent) | `"dreamer_srl"` | **delta**: WandB dashboard filters on `DreamerV3` won't match NEW runs; old runs unchanged |
| C8 | WandB kwargs | project/name from CLI; entity hardcoded; no group/job_type; ambient auth | config-driven (logger/wandb.yaml) + CLI; `name = --wandb-name or tag`; group/job_type set (job_type wiring from commit `6e82fc3`); `wandb_login` helper; `wandb.disabled` honored; `log_code` uploads .py files | same entity/project values today → no data moves; new metadata fields |
| C9 | Results dir | `results/JAX_DreamerSRL/<ts>_<wandb-run-name>`; `tmp/JAX_DreamerSRL_<ts>` when `--no-wandb` | `results/JAX_DreamerSRL/<ts>_<tag>` always (parent name pinned, NOT `JAX_dreamer_srl`) | **delta** in name component + no tmp/ diversion; internal layout (`models/env_config.yaml`, `models/agent_config.yaml`, `checkpoints/<ep>`) unchanged |
| C10 | GPU memory env | caller-exported vars only; JAX preallocates by default | `XLA_PYTHON_CLIENT_PREALLOCATE=false` + `--device` supported | **delta**: lower VRAM footprint; Gate 2 watches for allocator-related SPS change |
| C11 | Dumped `models/env_config.yaml` content | no `wandb:` block | contains `wandb:` block (logger layer merged) + CLI overrides written back (seed, checkpoint freq, body flags) | cosmetic + improved self-description |
| C12 | New flags usable for Dreamer | — | `--tag`, `--quiet`/`--debug` (already existed), `--no-satiation`, `--no-overeating-death`, `--device`, `--checkpoint-frequency` | additive |
| C13 | rPPO-only flags with a Dreamer config | n/a | `--load-checkpoint`, `--wandb-resume-id`, `--num-steps`, `--hidden-size`, `--lr`, `--log-accumulate`, `--profile`, `--experiment-eval` → **ValueError** | loud-fail (resume stays out of scope per decision 5) |
| C14 | Dreamer-only flags | `--buffer-device`, `--legacy-grad-loop` | same flags on train.py, passed through; ValueError with non-dreamer algorithms | parity |
| C15 | Ctrl-C / SIGTERM | raw KeyboardInterrupt kill | unchanged (train.py's graceful-stop handlers registered only on the non-dreamer path) | no delta — deliberately preserved so SIGTERM is never swallowed |
| C16 | Everything inside the loop | training math, PRNG stream, replay/Ratio machinery, prefill, curriculum swap, checkpoint payload/cadence, eval video+stats passes, Track C step clock, episode logging | **byte-identical (delegated, unchanged)** | Gate 1 proves it |
| C17 | Behavior-probe eval battery (12 avoidance conditions) | not available | still not available; loud error if enabled | unchanged per decision 5 |
| C18 | `training.eval_stats_num_envs` | read-but-dead | still read-but-dead | unchanged per decision 5 (OPEN reminder row stays) |

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
    quiet: bool
    debug: bool
    buffer_device: str            # "cpu" | "gpu"
    legacy_grad_loop: bool
    log_interval: Optional[int]   # legacy --log-interval passthrough
    env_config_path: Optional[str]     # provenance (wandb payload)
    agent_config_path: str
    configs_dir: Optional[str]
    continual_schedule_path: Optional[str]
    parity_dump: Optional[str] = None  # Phase 2 harness hook; None in production
```

`run_dreamer_training(spec, env_cfg, agent_cfg, schedule)` is the extracted body of today's `main()` from the "mandatory agent config reads" (line ~557) to the end (line ~2108), with `args.X` → `spec.X` and with budget/num_envs resolution (lines 567, 574–594) **removed** (the callers supply `episodes`/`total_timesteps`/`num_envs`). Until Gate 3, `dreamer_srl_main.main()` keeps its argparse + config resolution byte-for-byte and merely packages a spec — so its behavior is unchanged and it serves as the A-side of both gates. train.py's dispatch is the B-side. After Gate 3, `main()` becomes a flag-translating deprecation shim onto train.py.

Both `ContinualSchedule` dataclasses are field-identical; train.py passes its own schedule object into the seam (duck-typed — add a one-line comment in both files pinning the field contract).

### File Changes — Phase 1 (dispatch + unified CLI + seam)

#### 1. `src/algorithms/dreamer_srl/dreamer_srl_main.py`

- Add `DreamerRunSpec` (above) near the top of the module.
- Extract `run_dreamer_training(spec, env_cfg, agent_cfg, schedule)` — the body of `main()` from line ~557 (`learning_starts_cfg = agent_cfg.get_mandatory(...)`) through the end of the loop/final-log (line ~2108), **in this same module** so the existing test imports (`_advance_episode_counters`, `_eval_scalar_prefix`, `_reset_terminal_step_data`, `ContinualSchedule`, `main`) stay valid. Mechanical substitutions only:
  - `args.seed` → `spec.seed` (3 sites: `np.random.seed`, `PRNGKey`, both eval passes' `seed=`),
  - `args.quiet/debug/buffer_device/legacy_grad_loop/log_interval` → spec fields,
  - `num_envs`/`episodes`/`total_timesteps` → spec fields (delete lines 567, 574–594 from the extracted body; keep `learning_starts, prefill_steps = derive_prefill(...)` inside, fed by `spec.num_envs`),
  - `env_max_steps` read stays (banner uses it),
  - WandB init (§9, lines 815–901): `use_wandb = spec.wandb_enabled`; `wandb.init(**{k: v for k, v in spec.wandb_kwargs.items() if v is not None}, config=wandb_config)`; payload's `args.env_config/agent_config/configs_dir/continual_schedule` → spec provenance fields; **define_metric block unchanged**,
  - Results dir (§9b, lines 903–917): `if spec.results_dir: results_dir = spec.results_dir` else keep today's convention verbatim,
  - Everything else (config dumps, checkpoint manager, loop, curriculum swap, eval passes, Track C logging) **untouched**.
- `main()`: keep argparse + config resolution exactly as today; replace the fall-through body with spec construction + `run_dreamer_training(spec, env_cfg, agent_cfg, schedule)`. Spec values reproduce today's behavior bit-for-bit: `wandb_kwargs = {"project": args.wandb_project, "entity": "sungwoolee", "name": args.wandb_name, "group": None, "job_type": None}`, `results_dir = args.results_dir` (None → legacy naming), `episodes`/`total_timesteps` from the existing resolution block (which stays in `main()`).
- Parity hook: when `spec.parity_dump` is set (never in production), (a) at startup write `<dump>/resolved.json` = `{spec fields (minus paths that embed timestamps), env_cfg.to_dict(), agent_cfg.to_dict(), schedule boundaries/names or null}`; (b) per iteration append to in-memory lists: `iter_num, policy_step, total_episodes_completed, cumulative_grad_steps, n_grad_steps(_scan), buffer.filled_size`, and — only on iterations where `last_losses` was updated — each loss value as float64; (c) at loop exit write `<dump>/telemetry.npz` plus `<dump>/final.json` = final counters, PRNG `key` as a list, per-module parameter checksums (`float64` sum + L2 norm over `jax.tree.leaves(nnx.state(m, nnx.Param))` for world_model/actor/critic/target_critic) and the same for `moments`. All of it inside `if spec.parity_dump is not None:` guards — zero work otherwise. (The per-iteration float() of losses forces a host sync; acceptable because parity runs are ≤ a few hundred iterations.)

#### 2. `train.py`

- **Algorithm whitelist (bug fix for the OPEN "unknown algorithm hangs forever" row).** Immediately after `algorithm = config.get_mandatory('agent.algorithm')` (line ~633):
  ```python
  KNOWN_ALGORITHMS = ("RecurrentPPO", "DQN", "DRQN", "PPO", "dreamer_srl")
  # existing DreamerV3 archived-stack error stays, with an added hint:
  #   "...If you meant the live world-model agent, set agent.algorithm: dreamer_srl."
  if algorithm not in KNOWN_ALGORITHMS:
      raise ValueError(f"Unknown agent.algorithm {algorithm!r}. Valid: {KNOWN_ALGORITHMS}. "
                       "(Previously this fell through to an infinite no-op training loop.)")
  ```
- **Dreamer train-defaults peek-merge**, parallel to the rPPO peek (line ~496): if `Config.load_yaml(args.agent_config).get("agent.algorithm") == "dreamer_srl"`, merge `configs/train/dreamer_srl.yaml` right after `train/default.yaml`. (Do the YAML peek once, reuse for both gates.)
- **env_cfg snapshot for the dreamer branch**: when the peeked algorithm is `dreamer_srl`, deep-copy the merged config right **before** the agent-config merge (line ~565) into `dreamer_env_cfg` (the Dreamer loop requires separate env/agent configs — divergence matrix §2.1). Apply the env-level CLI overrides to it after the main CLI-override block (seed, `--no-satiation`/`--no-overeating-death` body keys, `training.checkpoint_frequency`) so the dumped `env_config.yaml` self-describes.
- **Continual guard** (line ~678): allow `{"RecurrentPPO", "dreamer_srl"}`; message updated.
- **Dispatch point**: after the experiment-eval gate (line ~650-667 today; the parallel refactor is moving this gate's config key — developer re-locates it, the gate itself already raises for non-rPPO):
  ```python
  if algorithm == "dreamer_srl":
      _run_dreamer_srl(args, config, dreamer_env_cfg, schedule)
      return
  ```
- **`_run_dreamer_srl(args, config, env_cfg, schedule)`** (new function, ~80 lines):
  1. Reject rPPO-only flags loudly (compat row C13): any of `--load-checkpoint`, `--wandb-resume-id`, `--num-steps`, `--hidden-size`, `--lr`, `--profile`, non-None `--log-accumulate` → `ValueError("--X is not supported for agent.algorithm=dreamer_srl")`. (`--experiment-eval` is already rejected by the existing gate.)
  2. Resolve, using the same expressions as the rPPO path: `episodes = schedule.episode_boundaries[-1] if schedule else (args.episodes if args.episodes is not None else config.get_mandatory('episodes'))` (persist via `config.set('episodes', ...)` for the dump, mirroring line ~693); `num_envs = args.num_envs or env_cfg.get_mandatory('training.num_envs')`; `total_timesteps = args.total_timesteps or (episodes * env_cfg.get_mandatory('environment.max_steps') * num_envs)`; `seed = args.seed if args.seed is not None else config.get_mandatory('seed')`; `tag = args.tag or config.get_mandatory('tag')`.
  3. WandB kwargs exactly as the rPPO block resolves them (project/entity/group from config + CLI, `job_type` mandatory from config per commit `6e82fc3`, `name = args.wandb_name or tag`); `wandb_enabled = WANDB_AVAILABLE and not args.no_wandb and not config.get_mandatory('wandb.disabled')`.
  4. `results_dir = args.results_dir or os.path.join("results", "JAX_DreamerSRL", f"{ts}_{tag}")` — parent name **pinned** to the existing `JAX_DreamerSRL` (compat row C9).
  5. Build `DreamerRunSpec` (incl. `buffer_device`, `legacy_grad_loop` from the two new train.py flags; `log_interval=args.log_interval`; provenance paths) and call `run_dreamer_training(spec, env_cfg, Config.load_yaml(args.agent_config), schedule)`.
- **New CLI flags**: `--buffer-device {cpu,gpu}` (default "cpu"), `--legacy-grad-loop` (default False), `--parity-dump PATH` (hidden/undocumented in help beyond one line; Phase 2). Add `--agent-config` as an alias of `--agent_config`. All three dreamer-only flags → `ValueError` if the resolved algorithm is not `dreamer_srl` (no silent ignore).
- **Signal handlers**: move the `signal.signal(SIGINT/SIGTERM, ...)` registration (lines 437–438) to after the dispatch point (just before "2. Setup Results Directory") so the Dreamer branch keeps today's raw-KeyboardInterrupt semantics and SIGTERM is never swallowed by a handler the Dreamer loop can't see (compat row C15).

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

#### 5. `tests/algorithms/dreamer_srl/test_entry_parity.py` (new)

**What is compared** (decision 4's list, mapped to concrete fields):
| Requirement | Concrete artifact |
|---|---|
| losses | per-iteration `last_losses` values (all keys: world-model/actor/critic components) from `telemetry.npz` |
| policy_step | per-iteration `policy_step` array |
| episode counters | `total_episodes_completed` array + final value |
| replay-ratio state | `cumulative_grad_steps`, `n_grad_steps` per iteration, `buffer.filled_size` per iteration |
| PRNG-sensitive values | final PRNG `key`, per-module parameter checksums (sum + L2), `moments` checksum |
| config resolution | `resolved.json` deep-diff of env_cfg/agent_cfg/spec |

**How captured**: both entry points run as subprocesses with `--no-wandb --parity-dump <tmpdir>` and identical `{env config, agent config, --episodes 40, --seed 7, --num-envs 4}`; agent config = `01_food_only_smoke.yaml` (small `learning_starts` so the run crosses prefill → train), env config = the small food-only env already used by the Dreamer smoke tests; `--checkpoint-frequency 20` so the run crosses ≥1 checkpoint+eval (video pass on, stats off — keeps runtime ~minutes) — this exercises budget resolution, prefill, the grad scan, episode accounting, checkpointing, and the eval path.

**Pass criteria (Gate 1)**:
1. *Config parity*: `agent_cfg` dicts byte-identical. `env_cfg` dicts identical after masking the **enumerated accepted-delta keys only** (`wandb.*` block, top-level `tag`, top-level `episodes` written-back value, `training.checkpoint_frequency` written-back value, `body.*` only if a CLI flag was passed). Any other diff = FAIL and must either be fixed or promoted to a new compat-table row before the gate can pass.
2. *Numeric parity, CPU (`JAX_PLATFORMS=cpu`)*: **bit-identical** — `np.array_equal` on every telemetry array, exact equality on all integer counters, PRNG key, and checksums. Bit-identity is the target and is achievable because both processes execute the *same* `run_dreamer_training` function with identical inputs on a deterministic backend.
3. *Numeric parity, GPU (manual, once, on the Gate 2 node before the long run)*: integer counters + PRNG key **exact**; losses/checksums within relative tolerance `1e-6`. Documented justification: across two *processes*, cuDNN/XLA autotuning may select different kernel algorithms for the same HLO, which can perturb float reductions at the ULP level; it cannot perturb control flow, counters, or the PRNG stream — so those stay exact-match. If GPU is also bit-identical (likely), record that and keep the tolerance clause as fallback.

The CPU test is a pytest (marked `slow`); the GPU variant is the same test parametrized, skipped unless `PARITY_GPU=1`.

#### 6. Phase 2 also verifies the seam refactor cost

`tests/algorithms/dreamer_srl/bench_sps.py` before/after Phase 1 on one GPU, same config/seed/step budget: the seam extraction + dormant parity hooks must be ≤ noise. >5% SPS drop = discuss; >15% = blocker (project speed policy).

### File Changes — Phase 3 (GPU A/B → **Gate 2**)

No code changes. Procedure (executed with `training-runner`, analyzed with `experiment-analyzer`):

1. **Node/GPU selection at validation time via the `gpu-status` skill** (decision 6 — named step, not pre-assigned here). One node, two GPUs, pack-node-first policy.
2. Config family: basic04 M — env `configs/environment/experiment/basic/04-jump_attack_10x10.yaml` (or the family's current canonical file; `experiment-designer` confirms), agent `configs/models/dreamer_srl/01_food_only_M.yaml`, same seed, `--episodes` sized for ≥3 checkpoints at the 10,000-episode cadence (≈30–40k episodes).
3. Run A = legacy invocation (`dreamer_srl_main.py --env-config ... --agent-config ... --seed 42 ...` with `agent.algorithm` already reading `dreamer_srl` — value is inert on the A path). Run B = `train.py --config ... --agent_config ... --seed 42 --tag ...`.
4. **Gate 2 pass criteria** on WandB: survival curves (`Episode/Steps`) and `Eval/MeanLength` overlap within same-seed GPU run-to-run noise (no systematic offset after the first checkpoint); `WorldModel/loss_model` trajectories overlap; `Time/sps_env` within ±5% (watches compat row C10's preallocation change); Track C intact on run B — eval videos visible in the WandB media panel, `Eval/*` vs `Eval/video/*` split present, no "dropped row" warnings in the log; run-B dir passes an `eval_rollout.py` smoke restore of one checkpoint.

### File Changes — Phase 4 (shim + flip → **Gate 3**)

#### 7. `src/algorithms/dreamer_srl/dreamer_srl_main.py` — `main()` becomes the shim

- Print a `DeprecationWarning` banner ("launch via train.py; this shim translates flags and will be removed").
- Flag translation (compat rows C1–C4): `--env-config`→`--config`; `--agent-config`→`--agent_config`; bare `--total-steps N` (no `--episodes`) → `--episodes 0 --total-timesteps N`; inject `--seed 0` when `--seed` absent (preserves the historical default); forward everything else verbatim; unknown leftover flags → error (argparse handles).
- Invoke the integrated path in-process: rewrite `sys.argv` and call `train_main()` (import `main` from the repo-root `train.py`; the module already hard-codes the repo root on `sys.path`). The old config-resolution body in `main()` is deleted; `run_dreamer_training`, `DreamerRunSpec`, `ContinualSchedule`, all helpers and the module import surface stay.
- Guard: the shim asserts the peeked agent config declares `agent.algorithm: dreamer_srl` and otherwise errors with the config-edit instruction (protects anyone launching with a stale private config still saying `DreamerV3`).

#### 8. `tests/algorithms/dreamer_srl/test_eval_telemetry_wandb.py`

Currently drives `dreamer_srl_main.main()`; after the shim conversion, retarget it at `run_dreamer_training` (build the spec directly) so it keeps testing the telemetry mechanics rather than flag translation. Add one small shim test (`test_shim_translation.py` or a case in the dispatch test): legacy argv → the exact translated argv.

#### 9. `train_command-agent.sh` (read-only for this plan; **flip happens only after Gate 3**)

Needed change, for the `developer`/`training-runner` at flip time — future Dreamer blocks use:
```bash
/home/vncuser/miniconda3/envs/grid_world_pain/bin/python train.py \
  --config configs/environment/experiment/<...>.yaml \
  --agent_config configs/models/dreamer_srl/<...>.yaml \
  --episodes <N> --seed <S> --tag "<run-tag>" \
  --wandb-name "<run-name>"
```
(no more `--env-config`/`--agent-config`/`--total-steps`; `CUDA_VISIBLE_DEVICES` still fine, or `--device gpu:<i>`). **Coordination note**: this file is user-edited and currently modified by a parallel session — flip via a fresh block, don't rewrite history.

#### 10. Post-flip bookkeeping (same change set)

- Ask `bug-curator` to: (a) close/update the "unknown agent.algorithm hangs forever" portion of the B1–B4 row (fixed by the whitelist, with the regression test path), (b) add a row-note that dreamer_srl now launches via train.py.
- Update the user auto-memory gotcha note `feedback_dreamer_srl_single_config_budget.md` (the `--checkpoint-frequency` gap and `env_cfg.training.*` budget quirk are superseded) — flag to the user; auto-memory is user-owned.
- `docs/environment/SCRIPTS_DEPENDENCY_MAP.md`: no `scripts/` file is added/moved/renamed/deleted by this plan, so no map update is required; developer double-checks nothing under `scripts/` gained/lost a caller (eval_rollout.py's inputs are unchanged by design).
- CONFIG_GUIDE / `02_config_schema.md`: no schema change (no new keys), no update required.

## Checkpoints (for `developer`, during implementation)

1. After the seam extraction (before any train.py change): full Dreamer test suite green (`test_episode_metrics`, `test_continual_schedule`, `test_prefill`, `test_terminal_step_data_reset`, `test_eval_telemetry`, `test_eval_telemetry_wandb`, `test_lax_scan_train`, `test_eval_video_smoke`) + a direct `dreamer_srl_main.py` CPU smoke run completes with unchanged console banner and run-dir layout.
2. After the train.py dispatch: dispatch smoke (File Change 4) green; an rPPO smoke via train.py unchanged (dispatch must be invisible to rPPO).
3. Phase 2: parity test green on CPU (bit-identical); config-diff report attached to the Implementation Report with every masked key listed.
4. bench_sps before/after numbers recorded in the Implementation Report (same GPU, config, seed, ≥ 2k iterations post-warmup).
5. Phase 4: shim translation test green; full suite green; one legacy-form CPU smoke through the shim reaches the training loop.
6. Commits: one per phase minimum; Phase 1 config edits (19 YAMLs) in their own commit.

## Risk register — what could silently change Dreamer training, and which gate catches it

| # | Risk | Mechanism | Caught by |
|---|---|---|---|
| R1 | Config-resolution drift (layer order, logger block, CLI override landing on the wrong key) silently changes what a run trains on | train.py builds env_cfg differently than dreamer_srl_main did | **Gate 1** config-parity deep-diff with an explicit accepted-mask; any unlisted diff fails the gate |
| R2 | Seam extraction perturbs the PRNG stream or seeding (e.g., `np.random.seed` moved/dropped, key split order changed) | training becomes non-reproducible vs today | **Gate 1** bit-identity on losses + final PRNG key + param checksums |
| R3 | Track C telemetry regression (step clock, `Eval/` vs `Eval/video/` split, video upload) from touching the WandB init/logging | dashboards silently lose rows/videos again | keep-dreamer mechanics (init + define_metric + eval blocks byte-untouched inside the seam); existing `test_eval_telemetry*`; **Gate 2** dashboard checklist |
| R4 | Run-dir layout drift breaks offline eval (`eval_rollout.py` needs `models/agent_config.yaml` + `checkpoints/<ep>`) | finished runs can't be re-evaluated | layout assertion in the dispatch smoke test (Phase 1); **Gate 2** eval_rollout restore smoke |
| R5 | Silent flag swallowing (an rPPO flag ignored on the Dreamer branch, or vice versa) | user believes an option applied when it didn't | loud-fail whitelists both directions + dispatch unit tests (Phase 1) |
| R6 | Perf regression from the seam / dormant parity hooks / preallocation change | slower training on every future Dreamer run | bench_sps A/B (Phase 2, ±5% policy) + **Gate 2** `Time/sps_env` ±5% |
| R7 | `agent.algorithm` rename breaks WandB dashboard filters and any tooling matching `DreamerV3` | new runs invisible in old views | enumerated (compat C7); Gate 2 analyst confirms new runs filterable; grep for `DreamerV3` consumers in `scripts/`/`analysis/` during Phase 1 |
| R8 | train.py's SIGTERM handler swallows kill signals for a loop that never checks `stop_requested` | remote kills hang; zombie GPU jobs | handler registration moved below dispatch (compat C15) + code-review item |
| R9 | Shim translation subtly changes an old launch (seed default, env-step mode) | historical commands stop reproducing | shim injects legacy defaults (C3/C4); shim translation unit test (Phase 4) |
| R10 | In-flight experiment-eval refactor collides with this plan's train.py edits | merge conflicts / gate reading a moved key | sequencing precondition: Phase 1 starts after that change lands; developer re-locates the gate |

## Test plan / regression tests per phase

- **Phase 1**: new `test_train_dispatch.py` (incl. the must-fail-pre-fix unknown-algorithm regression test); full existing Dreamer suite; one rPPO CPU smoke (no behavior change on the rPPO path); layout assertion.
- **Phase 2**: `test_entry_parity.py` CPU (CI, slow-marked); GPU parametrization manual; bench_sps A/B.
- **Phase 3**: no code — Gate 2 WandB checklist above, result recorded in this doc by `experiment-analyzer`.
- **Phase 4**: shim translation test; retargeted `test_eval_telemetry_wandb.py`; full suite; legacy-form smoke through the shim.

## Verification checklist (senior-developer, per phase)

- [ ] Phase 1 diff-stat proportionate: dreamer_srl_main.py ≈ net-zero logic (extraction), train.py + ~150, 19 one-line config edits, no unplanned files.
- [ ] `run_dreamer_training` byte-diff vs the old main() body reviewed: only the enumerated mechanical substitutions (spec fields, budget-resolution removal, parity guards).
- [ ] Whitelist raises before any env/model construction; DreamerV3 hint present.
- [ ] Signal registration verified below the dispatch return.
- [ ] Gate 1 artifacts reviewed: config-diff masked keys ⊆ compat table; CPU bit-identity report.
- [ ] bench_sps delta verdict recorded (✅/⚠️/❌ per speed policy).
- [ ] Gate 2 checklist all green before Phase 4 is authorized.
- [ ] Phase 4: no launch-script edit before Gate 3 sign-off; bug-curator updates requested.

## Implementation Report

*(to be filled by `developer`)*

## Verification Report

*(to be filled by `senior-developer` after each phase)*
