---
title: "Diagnosis: training entry point + config system (train.py, config_loader.py, utils/config.py)"
topic: issues
status: active
created: 2026-07-04
last_updated: 2026-07-04
---

# Diagnosis — training entry point + config system

## What this document is (plain-language entry point)

This is an independent bug hunt over the code that **starts a training run**: the script that
parses command-line flags, assembles the final configuration from several YAML layers, saves
that configuration next to the trained model, saves/restores checkpoints, and hands off to the
learning algorithm. A prior audit already fixed the two headline bugs here (config inheritance
being silently dropped, and command-line switches not being written into the saved config);
this pass **verifies those fixes and hunts for what they missed**.

Headline findings, in plain English:

1. **Resuming a run from a saved checkpoint does not actually work for the main algorithm
   (RecurrentPPO).** The restore code compares the checkpoint against the live model using two
   textually different key formats, concludes every single weight is "missing", raises an
   error — and then a blanket error-swallower prints one line (or, in quiet mode, nothing) and
   **training continues from freshly-random weights** while the operator believes it resumed.
   This was empirically demonstrated, not just read from the code.
2. **Resuming a multi-stage (continual) run puts the agent back in the first stage's world**
   while the logs claim it is in the later stage — because restoring the saved stage number
   suppresses the very check that would have rebuilt the environment.
3. Several smaller cases of **"the saved config says one thing, the run did another"** —
   the exact drift family the prior audit was hunting — remain: DreamerV3 records CLI
   learning-rate/size overrides it never applied; the `--total-timesteps` flag is accepted but
   ignored; a typo in a sensory-noise config silently disables that noise; a typo'd `--config`
   path silently trains on the default environment.

The previously landed fixes themselves (`extends:` resolution, commit `22c73ba`; CLI-flag
persistence, commit `75976e2`) were **verified correct on their intended paths** — details and
one incompleteness below. Everything here is a report only; no code was changed.

> **Fix plan (2026-07-04):** Findings 1, 2, and 5 (= combined-report rows H1–H3) are covered by
> [[fix_plan_h1h2h3_resume_config]] — approved work package WP-A, handed to `developer`.

---

## Finding 1 — `--load-checkpoint` never restores RecurrentPPO; failure is silently swallowed

**Severity: High** (silent training-from-scratch when the operator believes they resumed; also
blocks transfer/continual workflows) — **confirms + upgrades the LATENT KNOWN_BUGS row**
"Checkpoint restore may not map onto model" (memory `20260509_1536_train_py_checkpoint_restore_nnx_skew`).
The mechanism and the empirical proof below are new.

**Where:** `train.py:1141-1200` (strict-match restore), `train.py:2450-2460` (save side).

**What happens.** The checkpoint saves `nnx.state(model, nnx.Param)` via Orbax `StandardSave`
(`train.py:2452`). Restore reads it back with `ocp.args.PyTreeRestore()` (`train.py:1127`),
which returns a **raw nested dict**. The strict-match code then string-compares tree paths of
the restored dict against the live `nnx.state(model)` (`train.py:1147-1171`). Under the
installed flax/orbax versions the leaf path of the restored dict ends in
`DictKey(key='value')` while the live NNX state's ends in `GetAttrKey(name='value')` — so
`str(path)` differs for **100 % of leaves**, every parameter is reported "Missing in
Checkpoint" *and* "Missing in Current", the code raises the architecture-mismatch
`ValueError` (`train.py:1173-1177`)… which is caught by the blanket handler:

```python
except Exception as e:
    if not args.quiet:
        print(f"Error restoring checkpoint: {e}")
```
(`train.py:1198-1200`) — and **training proceeds from random initialization**. With `--quiet`
(the standard flag for launched lab runs) there is *zero* output.

**Empirical proof (2026-07-04, tiny ActorCriticRNN, exact save/restore code copied verbatim
from train.py; scripts `tmp/20260704_diag_ckpt_restore_test.py` / `..._test2.py`):**

- full-state vs Param-only state: identical (22 leaves each) — the Param-only save is *not*
  the problem;
- restored key: `(DictKey('actor_fc1'), DictKey('bias'), DictKey('value'))`
  vs live key: `(DictKey('actor_fc1'), DictKey('bias'), GetAttrKey('value'))`
  → **44/44 path mismatches** → `ValueError` raised, exactly as the live code would;
- independent second defect on the same path: even if the match succeeded,
  `nnx.update(optimizer, restored['optimizer'])` (`train.py:1184`) fails with
  `ValueError: Cannot set key '1' on immutable node of type tuple` (the optax-chain state
  tuple), so the restore would abort *after* the model weights were updated but *before*
  counters/stage were restored — a half-restored state, also swallowed;
- a **direct** `nnx.update(model, restored['model'])` with the raw dict works and restores
  weights bit-exactly — i.e. the DreamerV3 restore style (`train.py:1131-1133`) is
  structurally sound, and the ~200 lines of strict-match code are both broken and unnecessary
  for the "same architecture" case.

**Concrete failure scenario.** Any `python train.py --load-checkpoint results/.../models
--quiet ...` rPPO invocation: run starts, prints nothing, trains a fresh network from step 0
counters. WandB shows a plausible learning curve; nobody notices the "resume" was a restart.

**Suggested direction (for `developer`):** replace the strict-match block with restore-to-target
(`ocp.args.StandardRestore(item=<abstract state>)`) or direct `nnx.update` + an explicit leaf-count
assertion; make restore failure **fatal** (`raise`), never print-and-continue; restore optimizer
via a target tree of the live `nnx.state(optimizer)`.

---

## Finding 2 — Continual resume trains in the wrong stage's world (stage restore suppresses the env rebuild)

**Severity: High** (silently trains on the wrong environment; live for DreamerV3 resume today,
and for rPPO the moment Finding 1 is fixed) — **NEW** (sibling of fixed Finding E, which was
the same defect on the *eval* path).

**Where:** `train.py:1138-1139` and `1193-1194` (restore `current_stage`), `train.py:1224-1238`
(stage-transition check), `train.py:523,729-734` (env/params built from stage 0 before restore).

**What happens.** The environment is constructed **once**, from stage 0's config
(`params = load_env_params(config)` at `train.py:523`, `env = ParallelEnv(params)` at 729),
*before* checkpoint restoration. Restore then sets `current_stage = restored.get('stage', 0)`.
The per-iteration stage-transition check only rebuilds the environment when
`schedule.stage_for_episode(total_episodes_completed) != current_stage` — and after a resume
these are already **equal**, so the rebuild never fires. Result: a checkpoint saved in stage 3
resumes with stage-3 episode counters, stage-3 WandB tags (`_stage_tag()`,
`train.py:1205-1210`), and a **stage-0 world** — until the episode count crosses into stage 4,
if it ever does.

Ironically, *not* restoring `stage` would have produced correct behaviour (mismatch → rebuild
on the first iteration). The restored value is only used by the checkpoint payload and the tag.

**Concrete failure scenario.** A 5-stage predator curriculum crashes at episode 70 000
(stage 3). Operator relaunches with `--configs-dir ... --load-checkpoint ...`. The run resumes,
logs `stage/index = 3`, and trains against the stage-0 (no-predator) world for the remaining
episodes of stage 3. All post-resume metrics for that stage are garbage; the wrongness is
invisible in WandB because the stage tag says 3.

**Suggested direction:** after a successful restore in continual mode, force the rebuild:
load `schedule.stage_configs[current_stage]`, rebuild `params`/`env`, reset `env_state` — or
simply set `current_stage = -1` sentinel post-restore so the existing transition block fires.

---

## Finding 3 — DreamerV3: CLI `--lr` / `--hidden-size` are written into the saved config but never applied to the live trainer

**Severity: Med** (saved-config vs live-run divergence; corrupts later eval/rebuild) —
**NEW**; this is the *inverse* of the open L4 row, introduced by the same commit (`75976e2`)
that fixed L4 for the other algorithms.

**Where:** `train.py:500-510` (CLI resolution + `config.set('agent.actor_lr', lr)` /
`config.set('agent.rssm_deter_dim', hidden_size)`), `train.py:838`
(`DreamerTrainer(..., agent_config, ...)` — the **raw** `Config.load_yaml(args.agent_config)`
object from `train.py:389`), `src/models/dreamer_v3_trainer.py:83,117,125,133`
(trainer reads `agent.rssm_deter_dim`, `agent.model_lr`, `agent.actor_lr`, `agent.value_lr`
from that raw object).

**What happens.** `Config.merge` deep-copies dict structure, so `config.set('agent.*', …)`
after the merge does **not** touch the separate `agent_config` object. For DreamerV3 the
trainer is built from `agent_config`, so `--lr` and `--hidden-size` have **no effect on the
live run** — but they *are* recorded in `models/config.yaml` (`train.py:608-610`). The live
eval path rebuilds the agent from that saved config when `--agent_config` is not given
(`scripts/eval/eval_rollout.py:625-631`), so an eval of such a run reconstructs a
**different-sized RSSM** than the one that trained (restore shape-mismatch at best, silently
wrong at worst). `--num-steps` is unaffected (the `num_steps` variable is passed directly to
`collect_sequence`, `train.py:1545-1546`).

**Also — registry correction:** the open KNOWN_BUGS row **L4** ("model-size CLI overrides not
written to saved config") is **stale for RecurrentPPO / PPO / DQN / DRQN**: commit `75976e2`
added `config.set('agent.sequence_length'/'agent.num_steps'/'agent.hidden_size'/'agent.lr*')`
at `train.py:491-517`, and those branches build the model from the same resolved variables, so
CLI values are now both applied *and* persisted there. L4's residue is exactly (a) this Dreamer
inversion and (b) `--total-timesteps` (Finding 4).

---

## Finding 4 — `--total-timesteps` is accepted, echoed to WandB, and ignored as a budget

**Severity: Med** (flag looks applied but isn't; help text is wrong) — **NEW**.

**Where:** `train.py:238` (help: "overrides episodes if provided"), `train.py:487`
(`total_timesteps = args.total_timesteps or (episodes * env_max_steps * num_envs)`),
`train.py:1215` (loop condition
`(total_episodes_completed < episodes) if episodes > 0 else (global_step < total_timesteps)`).

**What happens.** The loop is episode-budgeted whenever `episodes > 0`, and `episodes` always
resolves > 0 in practice (`configs/train/default.yaml` sets `episodes: 100`;
`config.get_mandatory('episodes')` at `train.py:472`; continual mode uses the schedule's last
boundary). So `--total-timesteps` changes only the WandB `total_timesteps` field and the
console banner — the run still stops on the episode budget. It is also never persisted to the
saved config. A user launching "a 5 M-step run" via this flag gets whatever
`episodes × max_steps` implies, and the WandB config lies about the budget.

---

## Finding 5 — Typo'd `--config` path silently trains on the default environment

**Severity: High** (silently trains on the wrong environment; needs an operator typo to
trigger, but the failure mode is the project's worst class) — **NEW**.

**Where:** `src/utils/config.py:30-33` (`Config.load_yaml`: missing file → *print a warning,
return empty config*), reached from `train.py:381` via
`load_env_config(args.config)` → `_resolve_extends` → `Config.load_yaml`
(`src/environment/config_loader.py:57`). No existence check on `args.config` anywhere in
`train.py`.

**What happens.** A misspelled `--config` path produces one stdout line
(`Warning: Config file … not found. Using defaults.`) buried in the launch banner, merges
nothing, and the run proceeds on `configs/environment/default.yaml` (+ train/eval/logger/vis
defaults + agent config). Every mandatory key exists in the defaults, so **no `ValueError`
fires** — the no-fallback contract is structurally bypassed because train.py always merges the
default env config underneath (`train.py:298`). Contrast: `extends:` *targets* are
existence-checked and raise (`config_loader.py:68-72`), and a missing `--agent_config` fails
loudly at `agent.algorithm`. Only the top-level `--config` path fails soft.

**Suggested direction:** `raise FileNotFoundError` in `Config.load_yaml` (or at minimum an
`os.path.exists` guard on `args.config` in train.py). The warning-and-empty behaviour dates
from an era when defaults were acceptable; it now contradicts the project's strict-config rule.

---

## Finding 6 — Perceptual-noise config: unknown `mode` strings and misspelled modality keys silently mean "no noise"

**Severity: Med** (silent config drift in the project's central experimental manipulation) —
**NEW**.

**Where:** `src/environment/config_loader.py:1506-1527` (`_parse_noise_config`).

**What happens.**
- `_parse_mode(s)` (`config_loader.py:1509-1510`) maps `'state_dependent'`→2, `'constant'`→1,
  and **any other string** — including typos like `state-dependent`, `state_dependant`,
  `gaussian` — to 0 (`none`), with no error.
- `mode` itself is read with a soft default: `modalities_cfg[k].get('mode', 'none')`
  (`config_loader.py:1520`), and `sigma` with `.get('sigma', 0.0)` — contra the project's
  `get_mandatory` rule for critical keys.
- Modality keys not in `_YAML_KEY_TO_SENSOR_NAME` are **silently filtered out**
  (`if k in _YAML_KEY_TO_SENSOR_NAME`, lines 1513-1516) — a misspelled modality block (e.g.
  `visal:` for `visual:`) simply vanishes.

**Concrete failure scenario.** A noise-robustness study config declares
`perceptual_noise: {enabled: true, modalities: {visual: {mode: state-dependent, sigma: 0.4}}}`
(hyphen instead of underscore). The run loads cleanly, `perceptual_noise_enabled=True`, the
saved config looks right — and the visual channel receives **zero noise** for the whole
training. Every downstream comparison against the noiseless baseline is a null result by
construction. Suggested: whitelist-validate `mode` strings and modality key names; make
`mode` and (when mode≠none) `sigma` mandatory.

---

## Finding 7 — DreamerV3 resume: optimizer states and target critic are never saved/restored

**Severity: Med** (resume correctness; bounded but real post-resume distortion) — **NEW for
this file** (train.py's DreamerV3 path); sibling of the OPEN KNOWN_BUGS row
"Dreamer-srl checkpoints drop the optimizer's momentum" which covers the *separate*
`src/algorithms/dreamer_srl/checkpoint.py` entry point.

**Where:** `train.py:2461-2471` (checkpoint payload: `wm`/`actor`/`critic` Params + counters
only), `train.py:1130-1137` (restore side), `src/models/dreamer_v3_trainer.py:108`
(`self.target_critic` exists) and `:531-533` (EMA soft-update).

**What happens.** On resume: the three Adam optimizers restart with zero momentum, and
`target_critic` stays at its **fresh random init**, then EMA-converges toward the restored
critic over many updates (`dreamer_v3_trainer.py:531-533`). Until convergence, λ-return value
targets (`:395,416,439`) are computed against a random network — a transient but genuine
corruption of critic/actor updates right after every resume. (The replay buffer is also empty
post-resume, which is at least a visible/known property.)

---

## Finding 8 — WandB kwargs drift: `wandb.job_type` never sent; `wandb.name` config key ignored

**Severity: Low** (logging metadata only) — **NEW**.

**Where:** `train.py:638-644`. `wandb_kwargs` carries project/entity/group/name/reinit only:
`--wandb-job-type` (stored at `train.py:395`) and `configs/logger/wandb.yaml`'s
`job_type:` are never passed to `wandb.init`; the run name is `args.wandb_name or tag`,
ignoring the `wandb.name` config key (which exists in `configs/logger/wandb.yaml`). Also
`wandb.mode` is loaded but unused.

---

## Finding 9 — Assorted Low / hygiene items

- **Graceful shutdown discards progress** (`train.py:1216-1217`, `2555-2562`): SIGINT/SIGTERM
  breaks the loop and exits **without a final checkpoint save**; everything since the last
  periodic save (default `checkpoint_frequency: 10000` episodes) is lost. Low–Med robustness. NEW.
- **DQN/DRQN never checkpoint** (`train.py:2449-2473`): `ckpt_data` stays empty, so no save and
  no during-training eval ever fires for these algorithms — silently. Low (unused algos). NEW.
- **DRQN saved `agent.hidden_size` can misrepresent the architecture** (`train.py:913` derives
  `hidden_size = recurrent_layers[0]` *after* `train.py:516` persisted the CLI/YAML
  `agent.hidden_size`). Low. NEW.
- **rPPO branch logs lowercase `loss/*`, `modulator/*`** (`train.py:1493-1514`) while the
  `define_metric` "uppercase fix" (`train.py:679-682`) registers `Loss/*`, `Modulator/*` — the
  fix's own comment ("actual logged keys use uppercase prefixes") is wrong for this branch, so
  rPPO loss/modulator curves fall through to the `*` catch-all (timesteps axis, not iteration).
  Cosmetic/analysis-axis only. NEW.
- **Continual stage-0 dump asymmetry / aliasing** (`train.py:365`, `390`, `616-619`):
  `config` *is* `schedule.stage_configs[0]`, so the agent config and CLI overrides are merged
  into stage 0's object only; `stage_00_*.yaml` is dumped with agent+CLI content while
  `stage_01+` are env-only. The modality-fingerprint check (`train.py:551-576`) compares the
  agent-merged stage 0 against raw stage i — an agent config that (illegitimately) carried
  `environment.*` keys would skew stage 0 silently. Low. NEW.
- **`Config.merge` shares list references** between source and destination
  (`src/utils/config.py:87-95`: non-dict values, including lists, are assigned by reference)
  and **crashes with a bare `TypeError`** when a dict overrides a scalar. No live mutation
  path today; latent aliasing hazard. Low. NEW.
- **`--agent_config` is loaded without `extends:` resolution** (`train.py:389`,
  plain `Config.load_yaml`). No agent config uses `extends:` today (checked:
  only `configs/environment/**` do), but the failure mode if one ever does is the exact
  `22c73ba` bug reborn, plus the literal `extends:` key leaking into the saved config. Latent,
  Low. NEW.
- **Dead/misleading code**: `base_config_path` (`train.py:296`) computed and never used;
  `placement_mode` validated with a strippable `assert` (`config_loader.py:1261`);
  `main.last_checkpoint_save` as a function attribute (`train.py:2441-2442`) — works, but
  state leaks across hypothetical repeated `main()` calls. Nits. NEW.

---

## Verification of the previously-landed fixes in this area

- **`extends:` resolution (commit `22c73ba`)** — **correct and complete** on the training
  paths: single `--config` (`train.py:381` → `load_env_config`), continual stage files
  (`train.py:195`), and the live eval path (`scripts/eval/eval_rollout.py:535`) all resolve
  `extends:`; targets are existence-checked and cycle-guarded (`config_loader.py:50-75`);
  standalone files load byte-identically. No regression found. (The soft-fail on a missing
  *top-level* path — Finding 5 — predates and is orthogonal to this fix.)
- **CLI persistence (commit `75976e2`)** — correct for `--no-satiation` /
  `--no-overeating-death` (written to the same `body.*` keys `load_env_params` reads,
  `train.py:404-405`, plus continual propagation at `357-362`), `--episodes`, `--num-envs`,
  `--seed`, `--tag`, and the model-size scalars on the rPPO/PPO/DQN/DRQN branches
  (`train.py:491-517`) — which means **KNOWN_BUGS row L4 is stale for those algorithms**.
  Incomplete for: DreamerV3 persist-but-not-apply (Finding 3) and `--total-timesteps`
  (Finding 4).
- **Inclusive-integer range sampling (`7ff8d1f`)** — spot-checked in
  `config_loader.py:684-754`: `attack_range`/`detection_range` reject fractional bounds and
  carry lo/hi as ints; consistent with the plan. No issue found.

## Conventions audit

| Check | Status |
|---|---|
| Pytree immutability (train-side `params.replace`) | ✅ |
| JIT / static-field handling (stage rebuild recompiles by design) | ✅ |
| vmap conventions (DQN/DRQN manual vmap paths, `EnvParams` broadcast) | ✅ |
| PRNG threading (key split/advance per phase; restore carries `key`) | ✅ (unreachable on rPPO resume — Finding 1) |
| Sensor/obs↔noise sync | ⚠️ Finding 6 (silent-`none` on typo) |
| Config protocol (`get_mandatory` discipline) | ⚠️ Findings 5, 6 (soft-fail file load; soft-default noise keys) |

## Verdict — is this area sound?

**The happy path is sound; the resume path is not.** A fresh, correctly-spelled, single-config
or continual launch now assembles, applies, and saves the right configuration — the two prior
headline fixes hold. But **every checkpoint-resume mechanism in train.py is defective in some
way**: RecurrentPPO restore never succeeds and fails silently (Finding 1), continual resume
rebuilds the wrong world (Finding 2), and DreamerV3 resume drops optimizer/target-critic state
(Finding 7). Until these are fixed, treat `--load-checkpoint` as **non-functional for rPPO and
hazardous for continual runs**. Secondary but worth scheduling: the Dreamer CLI-override
inversion (Finding 3) and the two silent-typo traps (Findings 5, 6), which are precisely the
"silent config drift" class this project keeps being bitten by.

Reviewed by: code-reviewer (independent diagnosis pass, Fable 5, 2026-07-04)
