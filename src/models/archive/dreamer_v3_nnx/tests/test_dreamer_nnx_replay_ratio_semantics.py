"""Regression tests for WP-NNX F7 (U4) — `replay_ratio` semantics and the
missing random-action prefill.

Plain-language context: sheeprl's `replay_ratio` means "gradient steps per
ENV step" (per policy step). Our train.py called the same `Ratio` class with
`global_step // num_steps` — counting SEQUENCES — so at collect_interval=128
one unit of "replay ratio" here performed 1/128 of the gradient work the same
number means in sheeprl, silently. The fix makes the call site pass the raw
per-env-step `global_step` (config values were rescaled /128 in the same
change window to keep effective intensity unchanged — cost-neutral by
construction). Secondly, sheeprl prefills the buffer with 1024 RANDOM-policy
env steps before any training (`learning_starts`); ours started training on
data from the untrained network. The fix adds a mandatory
`agent.learning_starts` key and a static `random_actions` flag on
`collect_sequence`.

See docs/develop/active/diagnosis/dreamer_sheeprl_parity_2026-07-06/
fix_plan_nnx_parity.md (F7) and 05_dreamer_v3_nnx_conventions.md (U4).
"""

import os as _os_guard
import pytest as _pytest_guard
if _os_guard.environ.get("GWP_RUN_ARCHIVED_NNX_TESTS") != "1":
    _pytest_guard.skip(
        "archived DreamerV3-NNX stack -- set GWP_RUN_ARCHIVED_NNX_TESTS=1 to run",
        allow_module_level=True)
import os
import re
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from dreamer_nnx_fixtures import build_trainer_and_env, _REPO

import jax
import jax.numpy as jnp

from src.models.archive.dreamer_v3_nnx.dreamer_v3_util import Ratio

TRAIN_PY = os.path.join(_REPO, "train.py")

# Archival note (2026-07-10): the three tests below were SOURCE-LEVEL PINS on the
# live train.py DreamerV3 call sites (Ratio arithmetic, learning_starts gating).
# Those call sites were deleted when the DreamerV3-NNX stack was archived
# (archive_plan_dreamer_v3_nnx.md, train.py regions T4/T8), so the pins can never
# hold again — they are skipped explicitly rather than left red. The remaining
# tests in this file pin the Ratio CLASS itself and still run.
_TRAIN_PY_PIN_SKIP = pytest.mark.skip(
    reason="source-level pin on train.py's DreamerV3 call sites — deleted when the "
           "NNX stack was archived 2026-07-10 (see archive_plan_dreamer_v3_nnx.md)")


# ---------------------------------------------------------------------------
# T7(a) — Ratio semantics at the call site
# ---------------------------------------------------------------------------

@_TRAIN_PY_PIN_SKIP
def test_call_site_counts_env_steps_not_sequences():
    """Pin the actual train.py call-site arithmetic (source-level pin, as the
    expression is one line inside the training loop and train.py is not
    importable in tests). Pre-fix: `ratio_scaled_updates(global_step //
    num_steps)` — the per-SEQUENCE counting that made replay_ratio mean 1/128
    of sheeprl's value. Post-fix: `ratio_scaled_updates(global_step)`."""
    src = open(TRAIN_PY).read()
    assert "ratio_scaled_updates(global_step // num_steps)" not in src, (
        "train.py still counts replay_ratio per SEQUENCE (global_step // "
        "num_steps) — 1/128 of sheeprl's per-env-step semantics (U4)"
    )
    assert re.search(r"ratio_scaled_updates\(global_step\s*-\s*prefill_env_steps\)", src), (
        "expected the per-env-step, prefill-subtracted call "
        "`ratio_scaled_updates(global_step - prefill_env_steps)` in train.py "
        "(WP-NNX F7 + review_nnx_parity_fixes.md finding 1)"
    )


@_TRAIN_PY_PIN_SKIP
def test_learning_starts_mandatory_and_gating():
    """The random-prefill budget must be read via get_mandatory (no fallback
    default, per the Configuration Protocol) and must gate the train step."""
    src = open(TRAIN_PY).read()
    assert re.search(r"get_mandatory\(\s*['\"]agent\.learning_starts['\"]", src), (
        "agent.learning_starts must be read via config.get_mandatory (WP-NNX F7)"
    )
    assert re.search(r"global_step\s*>=\s*learning_starts", src), (
        "the Dreamer train gate must respect learning_starts (no training "
        "before the random prefill completes)"
    )


def test_ratio_per_env_step_accumulation():
    """Simulate the (post-fix) call-site loop: Ratio(r) fed the raw
    global_step, which increments by num_envs*num_steps per iteration.
    Cumulative gradient steps after N iters must track r * global_step within
    +-1 (the Ratio class carries fractional remainders exactly). Covers both
    the live rescaled value (0.00390625 with 16 envs) and a value that forces
    multi-iteration fractional accumulation (1 env)."""
    for r, num_envs, num_steps, iters in [
        (0.00390625, 16, 128, 50),   # live dreamer_v3.yaml geometry
        (0.00390625, 1, 128, 50),    # fractional: 0.5 grad steps per iter
        (0.5, 1, 128, 20),           # sheeprl-matched-style intensity
    ]:
        ratio = Ratio(r)
        global_step = 0
        cumulative = 0
        for _ in range(iters):
            global_step += num_envs * num_steps
            cumulative += ratio(global_step)
        expected = r * global_step
        assert abs(cumulative - expected) <= 1.0, (
            f"Ratio({r}) drifted: {cumulative} grad steps after "
            f"{global_step} env steps (expected ~{expected}) — per-env-step "
            f"semantics broken"
        )


@_TRAIN_PY_PIN_SKIP
def test_ratio_first_call_after_prefill_no_backlog_burst():
    """Review finding 1 (review_nnx_parity_fixes.md): sheeprl subtracts the
    random-prefill env steps from the Ratio argument before the first call
    (vendor dreamer_v3.py:661, with the one-iteration-back convention
    `prefill_steps = learning_starts_iters - 1`, vendor :511), so its first
    post-prefill call performs ~one iteration's steady-state work. Pre-fix,
    train.py passed the raw global_step, and Ratio's first-call branch
    returns `int(step * ratio)` — a one-time backlog burst of
    ≈ replay_ratio × learning_starts EXTRA gradient steps (~1,024 extra on
    dreamer_v3_sheeprl_matched.yaml geometry: ratio 1.0, learning_starts
    1024). This test simulates the ACTUAL call-site arithmetic (expression
    extracted from train.py source, since train.py is not importable) and
    asserts the first post-prefill call does no more than one iteration's
    steady-state work plus a few steps' tolerance (sheeprl's own
    one-iteration-back convention), and that steady-state accounting is
    undistorted afterwards."""
    src = open(TRAIN_PY).read()
    m = re.search(r"ratio_scaled_updates\((.+)\)", src)
    assert m, "could not find the ratio_scaled_updates(...) call site in train.py"
    call_expr = m.group(1)

    for r, num_envs, num_steps, learning_starts in [
        (1.0, 4, 128, 1024),         # dreamer_v3_sheeprl_matched intensity (the discriminative case)
        (0.00390625, 4, 128, 1024),  # live rescaled intensity (pre-fix burst was already ~4 — negligible)
    ]:
        env_steps_per_iter = num_envs * num_steps
        # Mirror of the train.py prefill accounting (vendor :508-511):
        # prefill_steps = learning_starts_iters - 1 (floored, clamped at 0),
        # expressed in env steps.
        prefill_env_steps = max(learning_starts // env_steps_per_iter - 1, 0) \
            * env_steps_per_iter

        ratio = Ratio(r)
        global_step = 0
        bursts = []
        for _ in range(200):
            global_step += env_steps_per_iter
            if global_step >= learning_starts:
                bursts.append(ratio(eval(call_expr, {}, {
                    "global_step": global_step,
                    "prefill_env_steps": prefill_env_steps,
                    "learning_starts": learning_starts,
                    "num_envs": num_envs,
                    "num_steps": num_steps,
                })))
                if len(bursts) == 4:
                    break
        steady_state = r * env_steps_per_iter
        assert bursts and bursts[0] <= steady_state + 4, (
            f"post-prefill FIRST Ratio call burst {bursts[0]} gradient steps "
            f"(ratio={r}, learning_starts={learning_starts}) — expected at "
            f"most one iteration's steady-state work "
            f"(~{steady_state:.0f} + a few steps; sheeprl subtracts the "
            f"prefill, vendor dreamer_v3.py:661)"
        )
        # Steady state after the first call must be unaffected by the
        # constant subtraction (deltas are what Ratio consumes).
        for b in bursts[1:]:
            assert abs(b - steady_state) <= 1.0, (
                f"steady-state accounting distorted after prefill "
                f"subtraction: got {b} grad steps/iter, expected "
                f"~{steady_state:.2f}"
            )


# ---------------------------------------------------------------------------
# T7(b) — random-action prefill in collect_sequence
# ---------------------------------------------------------------------------

def _degenerate_actor(trainer):
    """Force the actor head to a huge logit on action 0 (softmax -> ~1)."""
    head = trainer.agent.ac.actor.net.layers[-1]
    head.kernel.value = jnp.zeros_like(head.kernel.value)
    head.bias.value = jnp.zeros_like(head.bias.value).at[0].set(1000.0)


def test_collect_sequence_random_actions_flag():
    """`collect_sequence(..., random_actions=True)` must execute uniform
    random actions (ignoring the degenerate policy) and keep the carried
    prev_action consistent with the executed action; with the flag off the
    degenerate policy dominates. Pre-fix: TypeError (flag does not exist)."""
    NUM_STEPS = 128
    trainer, params, env_state, _ = build_trainer_and_env(max_steps=1000)
    _degenerate_actor(trainer)
    key = jax.random.PRNGKey(42)

    # Policy collection: action 0 dominates (unimix leaves ~1% leakage).
    _, fd_pol, _, tr_pol = trainer.collect_sequence(
        env_state, params, NUM_STEPS, key, None, False)
    pol_idx = jnp.argmax(tr_pol['action'][:, 0], axis=-1)
    frac0_pol = float(jnp.mean((pol_idx == 0).astype(jnp.float32)))
    assert frac0_pol >= 0.9, (
        f"degenerate-policy control broken: only {frac0_pol:.2f} of policy "
        f"actions are action 0"
    )

    # Random prefill: uniform actions, prev_action tracks the executed action.
    _, fd_rand, _, tr_rand = trainer.collect_sequence(
        env_state, params, NUM_STEPS, key, None, True)
    rand_idx = jnp.argmax(tr_rand['action'][:, 0], axis=-1)
    distinct = int(jnp.unique(rand_idx).shape[0])
    frac0_rand = float(jnp.mean((rand_idx == 0).astype(jnp.float32)))
    assert distinct > 1, (
        "random_actions=True still produces a single action — prefill is not "
        "random (U4)"
    )
    assert distinct >= 3 and frac0_rand <= 0.5, (
        f"random_actions=True actions not uniform-ish: {distinct} distinct, "
        f"frac(action 0) = {frac0_rand:.2f}"
    )

    # Carry consistency: the final carry's prev_action must be the LAST
    # executed (random) action, so the RSSM belief update stays paired with
    # what actually happened (sheeprl's player pairs likewise).
    assert jnp.array_equal(fd_rand['prev_action'][0], tr_rand['action'][-1, 0]), (
        "carried prev_action does not match the executed random action"
    )
    assert jnp.array_equal(fd_pol['prev_action'][0], tr_pol['action'][-1, 0])
