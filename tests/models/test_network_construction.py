"""Construction-time guard tests for ActorCriticRNN.

Plain-language context: the neuromodulator injects its signal into the task
network at three points — perceptual gain (Injection A), a memory gate bias fed
into the recurrent cell (Injection B), and an action temperature (Injection C).
Injection B requires the custom ModulatedGRUCell; the plain nnx.LSTMCell has no
`gate_bias` input, so building a modulated network with `rnn_type: "LSTM"` used
to silently drop Injection B — the modulator's z_memory head (and its params)
became dead weight with no error or warning. The constructor must refuse this
combination loudly instead.

See docs/reviews/review_nmn_trainer_parity_20260722.md (findings row 4).
"""
import pytest
from flax import nnx

from src.models.recurrent_ppo_network import ActorCriticRNN

MODULATION_CONFIG = {
    'type': 'FiLM',
    'mod_hidden_size': 8,
    'grouping_size': 1,
    'percept_bias_init': 1.0,
    'memory_bias_init': 0.0,
    'memory_clip': [-1.0, 1.0],
    # Modulation-site refactor: WHERE the modulator writes is now explicit, and the
    # action temperature is opt-in. These values reproduce the pre-refactor network
    # (encoder + task-GRU gate bias + temperature on), which is what this file's
    # LSTM guard is about.
    'sites': {'encoder': True, 'rnn': True, 'actor': False, 'critic': False},
    'rnn_mechanism': 'gate_bias',
    'temperature': {'enabled': True, 'clip': [0.5, 2.0]},
}

ENCODING_CONFIG = {
    'encoding_mode': 'flat',
    'use_layer_norm': False,
}


def _build(rnn_type, modulation_config):
    return ActorCriticRNN(
        input_dim=8,
        action_dim=4,
        hidden_size=16,
        rngs=nnx.Rngs(0),
        rnn_type=rnn_type,
        modulation_config=modulation_config,
        encoding_config=ENCODING_CONFIG,
    )


def test_modulation_plus_lstm_raises():
    """LSTM + modulation must raise at construction: Injection B (memory gate
    bias) requires the GRU cell, and silently dropping it makes the modulator's
    z_memory head dead weight."""
    with pytest.raises(ValueError, match="GRU"):
        _build("LSTM", dict(MODULATION_CONFIG))


def test_modulation_plus_gru_constructs():
    """Control: the supported GRU + modulation combination must still build."""
    model = _build("GRU", dict(MODULATION_CONFIG))
    assert model.modulation_enabled


def test_unmodulated_lstm_still_constructs():
    """Control: LSTM without modulation is unaffected by the guard."""
    model = _build("LSTM", None)
    assert not model.modulation_enabled
