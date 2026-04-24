# Checkpoint Retention Hardcoded at 5

> **Status**: PLANNED
> **Opened**: 2026-04-24
> **Related**: [labmeeting task list #4](../../CLAUDE.md) · [docs/environment/13_checkpoint_scheduling.md](../environment/13_checkpoint_scheduling.md) · sibling plans: [ISSUE_01](ISSUE_01_PREDATOR_COUNT.md), [ISSUE_02](ISSUE_02_PROPERTY_KEY_UNIFY.md), [ISSUE_03](ISSUE_03_DANGER_TO_HIDING_PREDATOR.md)

---

## Context

Only the five most recent checkpoints survive on disk after a long run. Early checkpoints get pruned silently, which breaks mid-training evaluation and makes it impossible to reconstruct long-horizon training-curve diagnostics (e.g. compare episode 200 vs. 20000 behavior after the run is complete).

## Analysis

**Root cause.** [train.py:349-352](../../train.py#L349-L352) instantiates Orbax's `CheckpointManager` with a hardcoded `max_to_keep=5`. There is no YAML override.

```python
# train.py:349-352
checkpointer = ocp.CheckpointManager(
    os.path.abspath(models_dir),
    options=ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
)
```

The literal `5` appears only once in the codebase (per the Explore agent's grep). Orbax's default pruning policy deletes the oldest checkpoint once the sixth is written, so a run that saves 40 checkpoints ends with only the last 5 on disk — the pruning is intentional on Orbax's part, just configured too aggressively for this project's needs.

**Supporting evidence in existing docs.** [docs/environment/13_checkpoint_scheduling.md:44](../environment/13_checkpoint_scheduling.md#L44) and §"Regime 3" already document this behavior — so the fix is well-understood, just never threaded into the config.

## Implementation Plan

### Design

Make `max_to_keep` a YAML-configurable parameter with a sensible default. No config → keep current behavior (`5`) to avoid surprising anyone running existing scripts; explicit config → new value. This matches the "no silent fallback for critical params" rule in `CLAUDE.md` while still being backwards-compatible for the non-critical case.

### File Changes

#### `train.py` (lines 349-352)

```python
# BEFORE:
checkpointer = ocp.CheckpointManager(
    os.path.abspath(models_dir),
    options=ocp.CheckpointManagerOptions(max_to_keep=5, create=True)
)

# AFTER:
max_checkpoints = config.get_mandatory('training.max_checkpoints_to_keep')
checkpointer = ocp.CheckpointManager(
    os.path.abspath(models_dir),
    options=ocp.CheckpointManagerOptions(max_to_keep=max_checkpoints, create=True)
)
```

Per CLAUDE.md this must be a **mandatory** config key (no silent default). Missing YAML key → `ValueError`. The training configs must be updated in the same change.

Allow `max_checkpoints_to_keep: null` / `None` in YAML and pass it through to Orbax (Orbax treats `None` as "keep everything"). Document this so users who want full history can opt in explicitly.

#### `configs/training/default.yaml` (and every other training config)

Add a new key under `training:`:

```yaml
# AFTER:
training:
  checkpoint_frequency: 1000
  max_checkpoints_to_keep: 5   # integer; set to null to keep every checkpoint
```

Every training YAML in `configs/training/` and any experiment YAML that overrides training defaults must declare this key. The implementing agent must enumerate them with `git grep -l "training:" configs/`.

#### `docs/environment/13_checkpoint_scheduling.md`

- Update the §"Regime 3" narrative to point at the new config key.
- Update the FAQ entry that says "change the value at line 352" — it should now read "change `training.max_checkpoints_to_keep` in your YAML."

### Alternatives Considered

- **Just bump the hardcoded value to 20 or 50.** Tempting one-line fix, but still leaves the retention count opaque to anyone reading YAML — and any user who wants unlimited history still has to patch source.
- **Make the key optional with a default of 5.** Rejected per CLAUDE.md's no-silent-default rule; a surprising retention policy is exactly the kind of thing that should not be silently defaulted.

## Checkpoints

- [ ] **C1** — Remove the key from a training config; confirm training fails fast with a clear `ValueError: Strict Config: training.max_checkpoints_to_keep is required.` (or equivalent).
- [ ] **C2** — Set `max_checkpoints_to_keep: 20`; run a short training loop that saves ~25 checkpoints (use a tiny `checkpoint_frequency`); confirm exactly 20 survive on disk.
- [ ] **C3** — Set `max_checkpoints_to_keep: null`; run same loop; confirm every checkpoint is retained.
- [ ] **C4** — Resume training from an older (not-most-recent) checkpoint directory to confirm the `CheckpointManager` still addresses old steps by their step key after a restart.
- [ ] **C5** — `git grep -n "max_to_keep"` — only the single `train.py` assignment should remain; no stray hardcoded integers.

## Implementation Report

> **Implemented by**: _(pending)_
> **Date**: _(pending)_

## Verification Report

> **Verified by**: _(pending)_
> **Date**: _(pending)_

| File | Change | Status | Notes |
|------|--------|:------:|-------|
| `train.py` | Read `training.max_checkpoints_to_keep` from config | | |
| `configs/training/*.yaml` | Add mandatory `max_checkpoints_to_keep` key | | |
| Experiment configs that override training | Add key where missing | | |
| `docs/environment/13_checkpoint_scheduling.md` | Update Regime 3 prose + FAQ | | |

**Conclusion**: _(pending)_
