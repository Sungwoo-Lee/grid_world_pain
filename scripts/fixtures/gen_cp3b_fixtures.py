"""gen_cp3b_fixtures.py — deterministic fixture generator for CP3b (buffers.py) tests.

Generates .npz fixture files for all 6 CP3b tests:
  1. buffer_storage_state_after_deterministic_adds
  2. buffer_sample_at_indices_matches_sheeprl
  3. buffer_is_first_marker_placement_in_straddling_window
  4. buffer_parallel_env_lane_non_interference
  5. cadence_yaml_key_parity_with_sheeprl_xs
  6. cadence_env_grad_step_trace_5000_iters

All fixtures use PRNG seed 0xD3EAF (decimal 868591) per the fixture convention
in tests/fixtures/dreamer_srl/README.md.

The state-evolution fixtures store:
  - The raw inputs (deterministic add() sequence)
  - The pre-computed sheeprl buffer state (for comparison by JAX tests without
    needing the sheeprl env at test time)

Run from the repo root in the sheeprl_bridge env (needs both PyTorch + JAX):
    /home/vncuser/miniconda3/envs/sheeprl_bridge/bin/python \\
        scripts/fixtures/gen_cp3b_fixtures.py

This script is deterministic — re-running produces byte-identical .npz files.
"""
import os
import sys
import numpy as np

SEED = 0xD3EAF  # 868591
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "dreamer_srl")
os.makedirs(FIXTURE_DIR, exist_ok=True)

sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "vendor", "sheeprl"))

# We need sheeprl's SequentialReplayBuffer for state-evolution comparison
from sheeprl.data.buffers import SequentialReplayBuffer as SheeprlSRB  # noqa: E402

rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# Shared buffer configuration (used across fixtures 1–4)
# ---------------------------------------------------------------------------
BUFFER_SIZE = 200
N_ENVS_1 = 1     # single-env for fixtures 1–3
N_ENVS_4 = 4     # 4-env for fixture 4
OBS_DIM = 8
N_STEPS_ADD = 100  # steps to add for fixtures 1–2
SEQUENCE_LENGTH = 10
BATCH_SIZE = 4


def _make_step_data(rng_gen: np.random.Generator, n_envs: int, obs_dim: int) -> dict:
    """Generate one step of synthetic transition data (seq_len=1)."""
    obs = rng_gen.uniform(-1.0, 1.0, size=(1, n_envs, obs_dim)).astype(np.float32)
    actions = rng_gen.uniform(-1.0, 1.0, size=(1, n_envs, 2)).astype(np.float32)
    rewards = rng_gen.uniform(-1.0, 1.0, size=(1, n_envs, 1)).astype(np.float32)
    terminated = np.zeros((1, n_envs, 1), dtype=np.float32)
    truncated = np.zeros((1, n_envs, 1), dtype=np.float32)
    is_first = np.zeros((1, n_envs, 1), dtype=np.float32)
    return dict(
        observations=obs,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        is_first=is_first,
    )


# ---------------------------------------------------------------------------
# Fixture 1: buffer_storage_state_after_deterministic_adds_input.npz
#
# Drive sheeprl's SRB with a fixed 100-step add() sequence.
# Store the resulting buffer state (_buf, _pos, _full, _n_envs, _buffer_size)
# as numpy arrays for comparison.
# ---------------------------------------------------------------------------
print("Generating fixture 1: buffer_storage_state_after_deterministic_adds_input.npz")

# Reproducible data sequence: generate all steps upfront from fixed rng
fixture1_rng = np.random.default_rng(SEED)
steps_data = [_make_step_data(fixture1_rng, N_ENVS_1, OBS_DIM) for _ in range(N_STEPS_ADD)]

# Drive sheeprl's buffer
sheeprl_rb1 = SheeprlSRB(buffer_size=BUFFER_SIZE, n_envs=N_ENVS_1)
for step in steps_data:
    sheeprl_rb1.add(step)

# Serialize all step data
all_step_keys = list(steps_data[0].keys())
steps_serialized = {}
for k in all_step_keys:
    steps_serialized[f"step_{k}"] = np.stack([s[k] for s in steps_data], axis=0)
    # shape: [N_STEPS_ADD, 1, n_envs, ...]

# Serialize sheeprl buffer state
sheeprl_state = {}
for k, v in sheeprl_rb1.buffer.items():
    sheeprl_state[f"buf_{k}"] = np.asarray(v)

fixture1_data = dict(
    buffer_size=np.int32(BUFFER_SIZE),
    n_envs=np.int32(N_ENVS_1),
    n_steps=np.int32(N_STEPS_ADD),
    obs_dim=np.int32(OBS_DIM),
    sheeprl_pos=np.int32(sheeprl_rb1._pos),
    sheeprl_full=np.bool_(sheeprl_rb1._full),
    seed=np.array(SEED),
    **steps_serialized,
    **sheeprl_state,
)
np.savez_compressed(
    os.path.join(FIXTURE_DIR, "buffer_storage_state_after_deterministic_adds_input.npz"),
    **fixture1_data,
)
print(f"  buffer after {N_STEPS_ADD} adds: pos={sheeprl_rb1._pos}, full={sheeprl_rb1._full}")
for k, v in sheeprl_rb1.buffer.items():
    print(f"    buf['{k}'].shape={v.shape}")


# ---------------------------------------------------------------------------
# Fixture 2: buffer_sample_at_indices_matches_sheeprl_input.npz
#
# Use the same buffer state from fixture 1.
# Pre-compute sheeprl's sample-index sequence using sheeprl's RNG.
# Store: precomputed start_idxes + env_idxes + the sheeprl _get_samples output.
# ---------------------------------------------------------------------------
print("\nGenerating fixture 2: buffer_sample_at_indices_matches_sheeprl_input.npz")

# Clone the sheeprl buffer state by rebuilding it
sheeprl_rb2 = SheeprlSRB(buffer_size=BUFFER_SIZE, n_envs=N_ENVS_1)
fixture2_rng = np.random.default_rng(SEED)
for step in [_make_step_data(fixture2_rng, N_ENVS_1, OBS_DIM) for _ in range(N_STEPS_ADD)]:
    sheeprl_rb2.add(step)

# Fix sheeprl's RNG seed for reproducible index sampling
sheeprl_rb2._rng = np.random.default_rng(SEED + 1)

# Pre-compute sheeprl's sample() index sequence manually
# (replicate the logic of sheeprl SequentialReplayBuffer.sample)
n_samples_f2 = 1
batch_dim_f2 = BATCH_SIZE * n_samples_f2

# valid start indices (buffer not full: _pos=100, seq_len=10 → valid: [0..90])
if sheeprl_rb2._full:
    first_range_end = sheeprl_rb2._pos - SEQUENCE_LENGTH + 1
    second_range_end = (
        BUFFER_SIZE if first_range_end >= 0 else BUFFER_SIZE + first_range_end
    )
    valid_idxes = np.array(
        list(range(0, first_range_end)) + list(range(sheeprl_rb2._pos, second_range_end)),
        dtype=np.intp,
    )
    raw_start_idxes = valid_idxes[
        sheeprl_rb2._rng.integers(0, len(valid_idxes), size=(batch_dim_f2,), dtype=np.intp)
    ]
else:
    raw_start_idxes = sheeprl_rb2._rng.integers(
        0, sheeprl_rb2._pos - SEQUENCE_LENGTH + 1, size=(batch_dim_f2,), dtype=np.intp
    )

# env_idxes (n_envs=1 → all zeros)
if N_ENVS_1 == 1:
    raw_env_idxes = np.zeros((batch_dim_f2,), dtype=np.intp)
else:
    raw_env_idxes = sheeprl_rb2._rng.integers(0, N_ENVS_1, size=(batch_dim_f2,), dtype=np.intp)

# Now call sheeprl's _get_samples with pre-computed idxes
chunk_length = np.arange(SEQUENCE_LENGTH, dtype=np.intp).reshape(1, -1)
precomputed_idxes = (raw_start_idxes.reshape(-1, 1) + chunk_length) % BUFFER_SIZE

# Manually call _get_samples on sheeprl's buffer using our precomputed indices
# We patch sheeprl's _rng to return our pre-computed env_idxes
class _FixedRNG:
    """Mock RNG that returns pre-computed env_idxes on next integers() call."""
    def __init__(self, env_idxes):
        self._env_idxes = env_idxes

    def integers(self, low, high, size, dtype=np.intp):
        return self._env_idxes.copy()

sheeprl_rb2._rng = _FixedRNG(raw_env_idxes)
sheeprl_samples = sheeprl_rb2._get_samples(
    precomputed_idxes, BATCH_SIZE, n_samples_f2, SEQUENCE_LENGTH
)

# Store fixture 2
fixture2_data = dict(
    buffer_size=np.int32(BUFFER_SIZE),
    n_envs=np.int32(N_ENVS_1),
    n_steps=np.int32(N_STEPS_ADD),
    obs_dim=np.int32(OBS_DIM),
    sequence_length=np.int32(SEQUENCE_LENGTH),
    batch_size=np.int32(BATCH_SIZE),
    n_samples=np.int32(n_samples_f2),
    precomputed_start_idxes=raw_start_idxes.astype(np.int64),
    precomputed_env_idxes=raw_env_idxes.astype(np.int64),
    seed=np.array(SEED),
)
# Add step data
fixture3_rng_f2 = np.random.default_rng(SEED)
for k in all_step_keys:
    fixture2_data[f"step_{k}"] = np.stack(
        [_make_step_data(fixture3_rng_f2, N_ENVS_1, OBS_DIM)[k] for _ in range(N_STEPS_ADD)],
        axis=0,
    )
# Regenerate clean — use same seed pattern as fixture 1
fixture2_steps_rng = np.random.default_rng(SEED)
fixture2_steps = [_make_step_data(fixture2_steps_rng, N_ENVS_1, OBS_DIM) for _ in range(N_STEPS_ADD)]
for k in all_step_keys:
    fixture2_data[f"step_{k}"] = np.stack([s[k] for s in fixture2_steps], axis=0)
# Add sheeprl reference output
for k, v in sheeprl_samples.items():
    fixture2_data[f"sheeprl_sample_{k}"] = np.asarray(v)

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "buffer_sample_at_indices_matches_sheeprl_input.npz"),
    **fixture2_data,
)
print(f"  precomputed {BATCH_SIZE} start_idxes={raw_start_idxes[:4]}")
for k, v in sheeprl_samples.items():
    print(f"    sheeprl_sample['{k}'].shape={v.shape}")


# ---------------------------------------------------------------------------
# Fixture 3: buffer_is_first_marker_placement_in_straddling_window_input.npz
#
# Build a buffer where is_first=1 at step 500 (done at step 499).
# Force-sample a window with start_idx=495, sequence_length=10.
# Store the expected sheeprl _get_samples output.
# ---------------------------------------------------------------------------
print("\nGenerating fixture 3: buffer_is_first_marker_placement_in_straddling_window_input.npz")

BUFFER_SIZE_3 = 1000
N_STEPS_3 = 600
DONE_AT = 499         # done at step 499 → is_first=1 at step 500
SEQ_LEN_3 = 10
START_IDX_3 = 495     # straddles the done (steps 495..504, done at 499, is_first at 500)

fixture3_rng = np.random.default_rng(SEED)

# Build transitions with a done at step DONE_AT
steps3 = []
for i in range(N_STEPS_3):
    s = _make_step_data(fixture3_rng, 1, OBS_DIM)
    if i == DONE_AT:
        s["terminated"] = np.ones((1, 1, 1), dtype=np.float32)
        s["is_first"] = np.zeros((1, 1, 1), dtype=np.float32)
    elif i == DONE_AT + 1:
        # is_first=1 marks the first step of the new episode
        s["is_first"] = np.ones((1, 1, 1), dtype=np.float32)
    steps3.append(s)

sheeprl_rb3 = SheeprlSRB(buffer_size=BUFFER_SIZE_3, n_envs=1)
for step in steps3:
    sheeprl_rb3.add(step)

# Force-sample the straddling window using pre-computed indices
start_idx_arr = np.array([START_IDX_3], dtype=np.intp)
env_idxes_arr = np.array([0], dtype=np.intp)

chunk_length_3 = np.arange(SEQ_LEN_3, dtype=np.intp).reshape(1, -1)
precomputed_idxes_3 = (start_idx_arr.reshape(-1, 1) + chunk_length_3) % BUFFER_SIZE_3

class _FixedRNG3:
    def integers(self, low, high, size, dtype=np.intp):
        return env_idxes_arr.copy()

sheeprl_rb3._rng = _FixedRNG3()
sheeprl_samples3 = sheeprl_rb3._get_samples(precomputed_idxes_3, 1, 1, SEQ_LEN_3)

# Verify is_first placement
is_first_window = sheeprl_samples3["is_first"]  # shape [1, 10, 1, 1]
is_first_flat = is_first_window.ravel()
done_offset = (DONE_AT + 1) - START_IDX_3  # = 5
print(f"  is_first window (start={START_IDX_3}, len={SEQ_LEN_3}): {is_first_flat}")
print(f"  Expected is_first=1 at window offset {done_offset}")
assert is_first_flat[done_offset] == 1.0, f"is_first not at offset {done_offset}: {is_first_flat}"

# Build step arrays for storage in fixture
fixture3_data = dict(
    buffer_size=np.int32(BUFFER_SIZE_3),
    n_envs=np.int32(1),
    n_steps=np.int32(N_STEPS_3),
    obs_dim=np.int32(OBS_DIM),
    done_at=np.int32(DONE_AT),
    is_first_at=np.int32(DONE_AT + 1),
    start_idx=np.int32(START_IDX_3),
    sequence_length=np.int32(SEQ_LEN_3),
    is_first_expected_offset=np.int32(done_offset),
    seed=np.array(SEED),
)
for k in all_step_keys:
    fixture3_data[f"step_{k}"] = np.stack([s[k] for s in steps3], axis=0)
for k, v in sheeprl_samples3.items():
    fixture3_data[f"sheeprl_sample_{k}"] = np.asarray(v)
for k, v in sheeprl_rb3.buffer.items():
    fixture3_data[f"buf_{k}"] = np.asarray(v)
fixture3_data["sheeprl_pos"] = np.int32(sheeprl_rb3._pos)
fixture3_data["sheeprl_full"] = np.bool_(sheeprl_rb3._full)

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "buffer_is_first_marker_placement_in_straddling_window_input.npz"),
    **fixture3_data,
)
print(f"  Fixture 3 saved. is_first[{done_offset}]=1 confirmed.")


# ---------------------------------------------------------------------------
# Fixture 4: buffer_parallel_env_lane_non_interference_input.npz
#
# Add 200 transitions across 4 envs (50 per env) with distinct sentinel
# observation values per env column.
# Verify no cross-lane leakage.
# ---------------------------------------------------------------------------
print("\nGenerating fixture 4: buffer_parallel_env_lane_non_interference_input.npz")

N_ENVS_4 = 4
N_STEPS_4 = 50         # 50 time steps (50 * 4 = 200 total transitions)
BUFFER_SIZE_4 = 100
SEQ_LEN_4 = 8
N_WINDOWS_4 = 10       # windows per lane

fixture4_rng = np.random.default_rng(SEED)

# Build transitions: env column i has sentinel value (i+1).0 in observations
steps4 = []
for t in range(N_STEPS_4):
    obs = np.zeros((1, N_ENVS_4, OBS_DIM), dtype=np.float32)
    for e in range(N_ENVS_4):
        obs[0, e, :] = float(e + 1)  # sentinel: env 0 → 1.0, env 1 → 2.0, ...
    actions = fixture4_rng.uniform(-1.0, 1.0, size=(1, N_ENVS_4, 2)).astype(np.float32)
    rewards = fixture4_rng.uniform(-1.0, 1.0, size=(1, N_ENVS_4, 1)).astype(np.float32)
    terminated = np.zeros((1, N_ENVS_4, 1), dtype=np.float32)
    truncated = np.zeros((1, N_ENVS_4, 1), dtype=np.float32)
    is_first = np.zeros((1, N_ENVS_4, 1), dtype=np.float32)
    steps4.append(dict(
        observations=obs,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        is_first=is_first,
    ))

sheeprl_rb4 = SheeprlSRB(buffer_size=BUFFER_SIZE_4, n_envs=N_ENVS_4)
for step in steps4:
    sheeprl_rb4.add(step)

# Verify sheeprl buffer has distinct sentinel values per env lane
for e in range(N_ENVS_4):
    sentinel = float(e + 1)
    obs_lane = sheeprl_rb4.buffer["observations"][:, e, :]  # shape [BUFFER_SIZE_4, OBS_DIM]
    filled = obs_lane[:sheeprl_rb4._pos]
    assert np.allclose(filled, sentinel), f"Lane {e} has contaminated obs"
print(f"  Sentinel values confirmed for {N_ENVS_4} env lanes")

fixture4_data = dict(
    buffer_size=np.int32(BUFFER_SIZE_4),
    n_envs=np.int32(N_ENVS_4),
    n_steps=np.int32(N_STEPS_4),
    obs_dim=np.int32(OBS_DIM),
    sequence_length=np.int32(SEQ_LEN_4),
    n_windows=np.int32(N_WINDOWS_4),
    seed=np.array(SEED),
    sheeprl_pos=np.int32(sheeprl_rb4._pos),
    sheeprl_full=np.bool_(sheeprl_rb4._full),
)
for k in steps4[0].keys():
    fixture4_data[f"step_{k}"] = np.stack([s[k] for s in steps4], axis=0)
for k, v in sheeprl_rb4.buffer.items():
    fixture4_data[f"buf_{k}"] = np.asarray(v)

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "buffer_parallel_env_lane_non_interference_input.npz"),
    **fixture4_data,
)
print(f"  Fixture 4 saved. {N_ENVS_4} lanes, {N_STEPS_4} steps each.")


# ---------------------------------------------------------------------------
# Fixture 5: cadence_yaml_key_parity_with_sheeprl_xs_input.npz
#
# Store the expected XS cadence values (hard-coded from sheeprl config).
# The test loads this fixture and compares against agent_xs.yaml.
# ---------------------------------------------------------------------------
print("\nGenerating fixture 5: cadence_yaml_key_parity_with_sheeprl_xs_input.npz")

EXPECTED_CADENCE = {
    "learning_starts": 1024,
    "replay_ratio": 1,
    "per_rank_gradient_steps": 1,
    "per_rank_sequence_length": 64,
    "per_rank_batch_size": 16,
    "per_rank_pretrain_steps": 0,
    "per_rank_target_network_update_freq": 1,
    "total_steps": 5000000,
    "num_envs": 1,
}

fixture5_data = dict(seed=np.array(SEED))
for k, v in EXPECTED_CADENCE.items():
    fixture5_data[f"expected_{k}"] = np.int64(v)

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "cadence_yaml_key_parity_with_sheeprl_xs_input.npz"),
    **fixture5_data,
)
print(f"  Expected cadence keys: {EXPECTED_CADENCE}")


# ---------------------------------------------------------------------------
# Fixture 6: cadence_env_grad_step_trace_5000_iters_input.npz
#
# Simulate the training-cadence loop for 5000 iterations on both sheeprl and
# our JAX, using the XS config. Mock out envs.step() and train().
# Store the expected (env_step, grad_step, per_rank_gradient_steps) trace.
# ---------------------------------------------------------------------------
print("\nGenerating fixture 6: cadence_env_grad_step_trace_5000_iters_input.npz")

from sheeprl.utils.utils import Ratio as SheeprlRatio  # noqa: E402

# XS config values
NUM_ENVS = 1
WORLD_SIZE = 1
TOTAL_ITERS = 5000
REPLAY_RATIO = 1.0
LEARNING_STARTS_STEPS = 1024
PER_RANK_PRETRAIN_STEPS = 0

# Sheeprl cadence trace
# Ref: vendor/sheeprl/sheeprl/algos/dreamer_v3/dreamer_v3.py:L505-L515, L550-L551, L661-L662
policy_steps_per_iter = int(NUM_ENVS * WORLD_SIZE)  # = 1
learning_starts = LEARNING_STARTS_STEPS // policy_steps_per_iter  # = 1024
prefill_steps = learning_starts - int(learning_starts > 0)  # = 1023

sheeprl_ratio = SheeprlRatio(REPLAY_RATIO, pretrain_steps=PER_RANK_PRETRAIN_STEPS)

policy_step = 0
cumulative_grad_steps = 0
sheeprl_trace = []  # list of (env_step, grad_step, per_rank_gradient_steps)

for iter_num in range(1, TOTAL_ITERS + 1):
    policy_step += policy_steps_per_iter  # += 1

    per_rank_gradient_steps = 0
    if iter_num >= learning_starts:
        ratio_steps = policy_step - prefill_steps * policy_steps_per_iter
        per_rank_gradient_steps = sheeprl_ratio(ratio_steps / WORLD_SIZE)
        cumulative_grad_steps += per_rank_gradient_steps

    sheeprl_trace.append((policy_step, cumulative_grad_steps, per_rank_gradient_steps))

sheeprl_trace_arr = np.array(sheeprl_trace, dtype=np.int64)
print(f"  Sheeprl trace: first 5={sheeprl_trace[:5]}")
print(f"  Sheeprl trace: last 5={sheeprl_trace[-5:]}")
print(f"  Total grad steps after {TOTAL_ITERS} iters: {cumulative_grad_steps}")

fixture6_data = dict(
    num_envs=np.int64(NUM_ENVS),
    world_size=np.int64(WORLD_SIZE),
    total_iters=np.int64(TOTAL_ITERS),
    replay_ratio=np.float64(REPLAY_RATIO),
    learning_starts_steps=np.int64(LEARNING_STARTS_STEPS),
    per_rank_pretrain_steps=np.int64(PER_RANK_PRETRAIN_STEPS),
    expected_trace=sheeprl_trace_arr,
    seed=np.array(SEED),
)

np.savez_compressed(
    os.path.join(FIXTURE_DIR, "cadence_env_grad_step_trace_5000_iters_input.npz"),
    **fixture6_data,
)
print(f"  Fixture 6 saved. trace shape={sheeprl_trace_arr.shape}")


print(f"\nAll CP3b fixtures generated successfully in {FIXTURE_DIR}")
