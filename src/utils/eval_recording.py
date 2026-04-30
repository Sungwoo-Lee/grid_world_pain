"""Episode recording format for offline (post-hoc) video rendering.

Each episode is serialized to ONE file per episode:

    <results_dir>/recordings/<checkpoint_pct>/episode_<NNNNNN>.rec

Plus ONE shared metadata file per eval run:

    <results_dir>/recordings/<checkpoint_pct>/run_meta.pkl
    (contains: params pytree, icon_config, action_map, config path, git sha)

Format: pickle-gzip (level 5) chosen based on Phase 0 benchmark.
"""
import gzip
import pickle
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

RECORDING_FORMAT_VERSION = 1
_DEFAULT_COMPRESSLEVEL = 5


def _snapshot_state(state) -> Dict[str, Any]:
    """Exactly the fields render_jax_state reads. Keep in lockstep with renderer.py."""
    return {
        'agent_pos': np.asarray(state.agent_pos),
        'satiation': float(state.satiation),
        'nutrition': float(state.nutrition),
        'injury_level': float(state.injury_level),
        'rest_streak': int(state.rest_streak),
        'res_pos': np.asarray(state.res_pos),
        'res_active': np.asarray(state.res_active),
        'pred_pos': np.asarray(state.pred_pos),
        'neutral_pos': np.asarray(state.neutral_pos),
        'obs_pos': np.asarray(state.obs_pos),
    }


class EpisodeRecorder:
    """Accumulates per-step data for one episode, then writes a single file."""

    def __init__(self, episode_index: int, train_episode: int, seed: int):
        self.episode_index = int(episode_index)
        self.train_episode = int(train_episode)
        self.seed = int(seed)
        self.snapshots: List[Dict[str, Any]] = []
        self.obs: List[np.ndarray] = []
        self.true_obs: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []

    def append(self, state, obs, true_obs, action_idx: int, reward: float):
        self.snapshots.append(_snapshot_state(state))
        self.obs.append(np.asarray(obs))
        self.true_obs.append(np.asarray(true_obs) if true_obs is not None else None)
        self.actions.append(int(action_idx))
        self.rewards.append(float(reward))

    def write(self, out_path: Path):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            'version': RECORDING_FORMAT_VERSION,
            'episode_index': self.episode_index,
            'train_episode': self.train_episode,
            'seed': self.seed,
            'snapshots': self.snapshots,
            'obs': np.stack(self.obs),
            'true_obs': (np.stack([t for t in self.true_obs]) if self.true_obs[0] is not None else None),
            'actions': np.asarray(self.actions, dtype=np.int32),
            'rewards': np.asarray(self.rewards, dtype=np.float32),
        }
        with gzip.open(out_path, 'wb', compresslevel=_DEFAULT_COMPRESSLEVEL) as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def write_run_meta(out_dir: Path, params, icon_config, action_map, config_path: str, extras: Dict = None):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        'version': RECORDING_FORMAT_VERSION,
        'params': params,           # EnvParams is a Flax struct.dataclass — picklable
        'icon_config': icon_config,
        'action_map': list(action_map),
        'config_path': str(config_path),
        'extras': dict(extras or {}),
    }
    with open(out_dir / 'run_meta.pkl', 'wb') as fh:
        pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)


def load_run_meta(recording_dir: Path) -> Dict[str, Any]:
    with open(recording_dir / 'run_meta.pkl', 'rb') as fh:
        return pickle.load(fh)


def load_episode(path: Path) -> Dict[str, Any]:
    with gzip.open(path, 'rb') as fh:
        return pickle.load(fh)
