---
title: "train.py does not resolve `extends:` — training configs lose inherited layers"
topic: diagnosis
status: archive
created: 2026-07-03
last_updated: 2026-07-03
---

# train.py does not resolve `extends:` → training runs silently drop inherited config layers

## Headline (plain language)
The v3.0 config system lets a config say `extends: <parent>` and inherit that parent's
settings (deep-merge). BUT `train.py` loads the `--config` file with a **plain YAML load**
that does **not** resolve `extends:`. So a training run gets only (a) `default.yaml` as the
base, plus (b) the config file's *own* explicitly-written keys — **every layer inherited from
a non-default `extends` parent is silently dropped and replaced by `default.yaml`'s value.**
Tooling that uses `load_env_config` (eval_rollout, tests, ad-hoc verification) DOES resolve
extends, so configs *look* correct there — which masks the training bug.

Concretely: **basic/07 trained with NO perceptual noise, NO random-init, and default hiding
count** — because those come from its `extends` parents (basic/06, basic/05), not from
basic/07's own file. This is the real reason "the video shows no noise": the run never had noise.

## The bug
`train.py:369-373`:
```python
elif args.config:
    user_config = Config.load_yaml(args.config)   # <-- plain YAML, NO extends resolution
    config.merge(user_config)
```
`Config.load_yaml` = `yaml.safe_load`. The `extends:` key is left as an inert string; the
parent chain (`load_env_config` / `_resolve_extends`) is never invoked. `train.py` never calls
`load_env_config` anywhere in its `--config` path.

## Evidence (confirmed)
- basic/07 saved `results/.../20260703-025113_rppo_basic07_jump_n113/models/config.yaml`:
  `perceptual_noise.enabled: false`, `random_start_injury: false`, hiding_predator `count_high: 4`.
- Reproducing train.py's load (`get_default_config()` + `Config.load_yaml(basic/07)` + merge)
  gives byte-for-byte the same: noise **false**, random_injury **false**, hiding **4**.
- Whereas `load_env_config('.../basic/07')` (extends resolved) gives noise **true**, random_injury
  **true**, hiding **2-12** — what we *intended* and what all our verification scripts saw.
- Rule confirmed: a config's OWN keys survive; anything inherited from a non-default parent is lost.

## Scope — this session's live runs (what they ACTUALLY trained with)
| Run (WandB) | Config | Own keys | Actually trained with |
|---|---|---|---|
| eylrft3q | basic/05 | full env+body in own file | CORRECT (self-contained) |
| v1–v4 (m3u4lqcd, cz9bz2so, wvey6fye, i176lo0s) | variants 01–04 (extend basic/05) | only the redeclared list | lose random-init + non-redeclared layers |
| v5 w3o8izox | 05-all_combined_noise (own noise key) | perceptual_noise | noise ON, but DEFAULT scene (no random-init, no all-combined) |
| basic/07 u1tyn8xk | basic/07 (extends 06) | environment.entities | **NO noise, NO random-init, default hiding** |
| jump-reach fkbk0st7 | variant (extends 07) | environment.entities | **NO noise, NO random-init** |

Broadly: any v3.0-era config that relies on `extends` to inherit layers it does not itself
redeclare has trained wrong since the v3.0 extends system landed (2026-06-19).

## Fix landed (2026-07-03)
Fixed in `train.py`: both the single `--config` path (main(), ~line 372) and the continual
`--configs-dir` per-stage path (`_build_continual_schedule`, ~line 192) now load through
`load_env_config()` (resolves `extends:`) instead of `Config.load_yaml` (plain YAML, no
resolution). The continual path had the identical bug — confirmed independently before the
fix (basic/06, basic/07 stage configs loaded with `random_start_injury=False` instead of the
inherited `True`) and confirmed fixed after.

Merge-order analysis: the resolved env config's top-level namespaces
(`environment`, `body`, `sensory`, `perceptual_noise`, `behavior_measures`,
`visualization.local_view_size`) do not collide with the train/eval/wandb/CLI keys merged
earlier in `train.py` (`training`, `testing`, `wandb`, `episodes`/`seed`/`tag`) — the one
shared top-level key, `visualization`, deep-merges without clobbering because
`configs/environment/default.yaml` only sets the `local_view_size` leaf, disjoint from
`configs/visualization/default.yaml`'s leaves, and re-asserting `local_view_size` is
idempotent (same source file, same value). A legacy no-`extends:` config was confirmed to
load byte-identical to before (backward-compat preserved). `pytest tests/env/ -q`: 192
passed, 492 skipped, 0 failed (pre-existing skip count, unaffected by this change).

Acceptance-test before/after (5 configs, `load_env_params` on train.py's exact merge
sequence vs. `load_env_params(load_env_config(path))` ground truth):

| Config | Before fix | After fix |
|---|---|---|
| basic/05 | MATCH (self-contained) | MATCH |
| basic/06 | MISMATCH (random_start_injury, hiding count range, max_stamina) | MATCH |
| basic/07 | MISMATCH (noise, random_start_injury, hiding count range) | MATCH |
| basic05_variants/04-all_combined | MISMATCH (random_start_injury) | MATCH |
| basic05_variants/06-jump_range_2to3 | MISMATCH (noise, random_start_injury, hiding count range) | MATCH |

Verification script (scratch, not committed): `tmp/20260703_extends_fix_verify.py`.

Portfolio decision on relaunching the affected runs (basic/07, jump-reach, the variants, v5)
is still open — left to the user / senior-developer.

## Recommended fix (NOT yet applied — needs user decision + senior-developer)
1. `train.py`: load `--config` via `load_env_config(args.config)` (resolves extends) instead of
   `Config.load_yaml`. Verify interaction with: get_default_config base, train/agent/eval-default
   merges, CLI overrides, and the continual `--configs-dir` path (which already uses stage configs).
2. Re-verify a few configs load-identically to `load_env_config` after the fix; run the env suite.
3. Relaunch the affected runs (basic/07, jump-reach, the variants, v5) — a portfolio decision.

## NOT done autonomously (deliberately)
- Did NOT change train.py's core config loading (affects EVERY training launch).
- Did NOT kill/relaunch any run.
- The separate `evaluation_core.py` true_obs-recording fix (video sensory panel) is written but
  left UNCOMMITTED; it is valid for correctly-configured noise runs but is secondary to this bug.
- The basic/07 "noise video" I generated used `eval_rollout` (extends resolved → noise ON) on a
  checkpoint that trained WITHOUT noise, so it is NOT representative of the run.
