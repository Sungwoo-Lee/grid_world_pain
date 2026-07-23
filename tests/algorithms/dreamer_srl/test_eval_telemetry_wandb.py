"""Track C telemetry-fix regression test (T4) — WandB step monotonicity.

Drives a real `main()` training loop (tiny single-env config, WandB in local
`offline` mode so no network call is made) and records every `wandb.log(...)`
call's keys + `step` via a thin wrapper around the real offline `wandb.log`.
Verifies the fix's monotone, explicit-step invariant end to end.

Stated limitation (see plan §T4): offline mode still exercises WandB's real
client-side step bookkeeping, but this test asserts the *inputs* the fix
guarantees (explicit, monotone steps) rather than independently re-deriving
WandB's drop behavior from the offline run file.

See docs/develop/active/dreamer/DREAMER_SRL_EVAL_TELEMETRY_FIX.md.
"""
import os
import sys
import tempfile

import pytest

sys.path.insert(0, '/media/nas01/projects/Interoceptive-AI/grid_world_pain')

_ROOT = '/media/nas01/projects/Interoceptive-AI/grid_world_pain'


def _write_smoke_config(path: str):
    import yaml
    cfg = {
        'environment': {
            'height': 5, 'width': 5, 'start_pos': [2, 2],
            'max_steps': 20, 'random_start_pos': True,
            'rest_action_enabled': True, 'eat_action_enabled': True,
            'placement': {'mode': 'per_entity'},
            'resources': [{
                'name': 'food', 'type': 'food', 'count': 2,
                'spawn_area': [[1, 1], [4, 4]],
                'properties': [1.0, 0.0, 0.0, 0.0, 0.0],
                'properties_std': [0.0, 0.0, 0.0, 0.0, 0.0],
                'max_consumption': 12, 'regeneration_delay': 0,
                'damage': [0.0, 0.0], 'nociception_intensity': 0.0,
            }],
        },
        'training': {
            'checkpoint_frequency': 3,
            'video_during_training': True,
            'stats_during_training': False,
            'eval_video_episodes': 1,
        },
    }
    with open(path, 'w') as f:
        yaml.dump(cfg, f)


@pytest.mark.slow
def test_eval_logs_monotone_explicit_steps(tmp_path, monkeypatch):
    """Every wandb.log() call in a real (tiny) training run carries an
    explicit, monotone-non-decreasing `step=`, and the eval-video /
    Eval/Mean* rows share the checkpoint's `policy_step`.
    """
    wandb_dir = tmp_path / 'wandb_offline'
    wandb_dir.mkdir()
    monkeypatch.setenv('WANDB_MODE', 'offline')
    monkeypatch.setenv('WANDB_DIR', str(wandb_dir))
    monkeypatch.setenv('WANDB_SILENT', 'true')

    results_dir = tmp_path / 'results'
    results_dir.mkdir()
    cfg_path = tmp_path / 'smoke_config.yaml'
    _write_smoke_config(str(cfg_path))

    argv = [
        'dreamer_srl_main.py',
        '--env-config', str(cfg_path),
        '--agent-config', os.path.join(_ROOT, 'configs/models/dreamer_srl/01_food_only.yaml'),
        '--total-steps', '300',
        '--num-envs', '1',
        '--seed', '42',
        '--quiet',
        '--results-dir', str(results_dir),
        '--wandb-project', 'grid_world_pain_test_offline',
    ]
    monkeypatch.setattr(sys, 'argv', argv)

    import wandb
    from wandb.sdk.wandb_run import Run

    # NOTE: monkeypatching the `wandb.log` module attribute directly does not
    # work here -- wandb.init() reassigns `wandb.log` to a fresh bound method
    # of the new Run right after init, silently overwriting any pre-init
    # patch. Patching the Run.log *class* method instead survives that
    # reassignment because the bound method resolves against the (already
    # patched) class at bind time.
    calls = []  # list of (sorted(keys), step)
    _real_run_log = Run.log

    def _capturing_run_log(self, data, step=None, **kwargs):
        calls.append((sorted(data.keys()), step))
        return _real_run_log(self, data, step=step, **kwargs)

    monkeypatch.setattr(Run, 'log', _capturing_run_log)

    from src.algorithms.dreamer_srl.dreamer_srl_main import main
    try:
        main()
    finally:
        if wandb.run is not None:
            wandb.run.finish()

    assert calls, "no wandb.log(...) calls captured — smoke run produced no telemetry"

    # (a) every recorded step is an int (no step=None among eval/stage/video/training rows)
    none_step_calls = [c for c in calls if c[1] is None]
    assert not none_step_calls, (
        f"found wandb.log calls with step=None: {none_step_calls[:5]} "
        f"(all sites must pass an explicit step= per the Track C fix)"
    )

    # (b) step sequence is monotone non-decreasing across the whole run
    steps = [c[1] for c in calls]
    for i in range(1, len(steps)):
        assert steps[i] >= steps[i - 1], (
            f"non-monotone step sequence at index {i}: {steps[i-1]} -> {steps[i]} "
            f"(full sequence: {steps})"
        )

    # (c) the eval/video row and the Eval/Mean* row carry the same step as
    #     the checkpoint's policy_step (i.e. some eval/video keys and some
    #     Eval/Mean* keys share an identical step value).
    video_steps = {c[1] for c in calls if any(k.startswith('eval/') for k in c[0])}
    eval_mean_steps = {c[1] for c in calls if any(k.startswith('Eval/Mean') for k in c[0])}
    assert video_steps, "no eval/* (video upload) wandb.log rows captured"
    assert eval_mean_steps, "no Eval/Mean* wandb.log rows captured"
    assert video_steps & eval_mean_steps, (
        f"eval/video step(s) {video_steps} and Eval/Mean* step(s) {eval_mean_steps} "
        f"never coincide — expected the same policy_step at a given checkpoint"
    )
