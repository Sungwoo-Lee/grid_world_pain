"""Generate (and verify) the NMN input x site grids -- sixteen agent configs per grid.

PLAIN LANGUAGE
--------------
The agent in this project can carry a small second network -- the *neuromodulator* --
that reads the agent's senses and continuously re-tunes the main policy network. This
study crosses two things that have never been varied independently: WHAT the modulator
reads (all 27 sensory numbers / the two body channels only / the nineteen outside-world
channels only) and WHICH parts of the main network it re-tunes (the sensory front-end,
the memory cell, the action head, the value head, or all four at once). One extra run
carries no modulator at all and is the control.

  1 control + (5 write targets x 3 input slices) = 16 agent configs, per grid.

TWO GRIDS, A MATCHED PAIR
-------------------------
There are now TWO such grids, identical in all sixteen cells except for one setting --
`return_mode`, the recipe the trainer uses to turn a rollout into the two numbers PPO
learns from (the critic's target, and the advantage):

  grid "mc"       return_mode "MC"        Monte-Carlo estimator, matched scale.
                  configs/models/recurrent_ppo/nmn_input_site_grid/
  grid "gaenorm"  return_mode "GAE_NORM"  GAE(lambda) estimator, matched scale.
                  configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm/

`GAE_NORM` rather than plain `GAE` is the deliberate choice: `MC` and `GAE` differ in TWO
things at once -- the estimator AND the scale convention -- whereas `GAE_NORM` is the GAE
estimator carrying MC's matched-scale convention. Swapping MC -> GAE_NORM therefore changes
only the estimator, so a difference between the two grids is attributable to that alone.
`--verify` enforces this claim mechanically: every gaenorm config is diffed key-by-key
against its mc twin and must differ in exactly `agent.return_mode`.

Design docs:
  docs/experiments/active/nmn_input_site_grid/NMN_INPUT_SITE_GRID.md          (grid "mc")
  docs/experiments/active/nmn_input_site_grid/NMN_INPUT_SITE_GRID_GAENORM.md  (grid "gaenorm")
Code this depends on: modulation-site refactor Part A (83b8140b) and Part B (e1aab726).

WHY A GENERATOR RATHER THAN THIRTY-TWO HAND-WRITTEN FILES
---------------------------------------------------------
The files are ~60 lines each and differ in exactly two blocks. Hand-copying near-identical
files is where a stray edit drifts unnoticed -- and a drifted hyperparameter in a single-seed
screen is indistinguishable from a result. Design-doc §3.1 requirement 1 therefore pins the
files as GENERATED. The same argument applies one level up, which is why the second grid
EXTENDS this generator instead of forking it: two generators are how two grids that are
supposed to differ in one key quietly come to differ in two. This script lives under
configs/ rather than scripts/ deliberately: scripts/ carries a dependency-map
maintenance contract (docs/environment/SCRIPTS_DEPENDENCY_MAP.md) that a config
generator has no business triggering. Precedent:
configs/environment/experiment/sensory_directional/generate_weakened_vision_arms.py

HOW THE COMMON PART IS GUARANTEED IDENTICAL
-------------------------------------------
The shared agent body is not retyped here. It is READ from that grid's base agent config --
one of the unmodulated arms of the 25-run return-mode study -- so the control run of each grid
is a lineage-exact continuation of an existing five-seed record. Exactly ONE key is changed
relative to that file, in all sixteen outputs including the control: `lr_critic`
0.0001 -> 0.0005 (design-doc confound C5 / requirement 5). `lr_critic` is read by no code path
today, so this changes nothing about how these runs train; it exists so that a later rerun from
any of these saved configs, after the pending learning-rate fix lands, does not silently train
one arm's critic at a fifth of the rate the rest of the grid used.

USAGE (from anywhere; the script chdir's to the repo root itself)
----------------------------------------------------------------
  python configs/models/recurrent_ppo/nmn_input_site_grid/generate_site_grid_arms.py
      -> (re)writes all sixteen YAML files of BOTH grids.

  ... generate_site_grid_arms.py --grid gaenorm
      -> restrict every action to one grid ("mc", "gaenorm", or "all"; default "all").

  ... generate_site_grid_arms.py --check
      -> writes nothing; fails if any file on disk differs from what this script
         would generate today. Cheap drift detector; safe to run any time.

  ... generate_site_grid_arms.py --verify
      -> --check, PLUS: loads every config through the real training config path and
         CONSTRUCTS the model, asserting the resolved modulator input width (27 / 2 / 19),
         the four site booleans, the RNN mechanism, temperature-off, a finite forward pass,
         and -- when both grids are in scope -- the one-key twin diff between them.
         Requires JAX; runs on CPU (JAX_PLATFORMS=cpu is set internally) so it never
         competes with training for a GPU.
"""
import argparse
import copy
import difflib
import io
import os
import sys

import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, *(['..'] * 4)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

from src.utils.config import dump_config_yaml, _FlowListDumper   # noqa: E402


class _QuotedStr(str):
    """A string that dumps with explicit double quotes.

    Used for the sensor NAMES in `input_sensors`. They contain spaces, and their
    exact spelling is load-bearing -- an unknown name is a hard error at model
    construction, by design -- so they are quoted rather than left as plain
    scalars, for the benefit of the human reading the file.
    """


_FlowListDumper.add_representer(
    _QuotedStr,
    lambda dumper, data: dumper.represent_scalar('tag:yaml.org,2002:str', str(data), style='"'))

ENV_CONFIG = 'configs/environment/experiment/basic/04-jump_attack_10x10.yaml'

# The ONE permitted deviation from a grid's base agent config, applied to all sixteen files.
# See the module docstring and design-doc C5 / §3.1 requirement 5.
LR_CRITIC = 0.0005

# --- The two grids ---------------------------------------------------------
# Everything that distinguishes one grid from the other lives here and nowhere else.
# `base_agent` is the ONLY source of the shared agent body, so the two grids inherit the
# same 25-run-study lineage and differ in exactly the key their base configs differ in.
GRIDS = {
    'mc': dict(
        base_agent='configs/models/recurrent_ppo/recurrent_ppo_cmp_mc.yaml',
        out_dir='configs/models/recurrent_ppo/nmn_input_site_grid',
        file_prefix='nmnsite',
        return_mode='MC',
        grid_label='NMN input x site grid',
        design_doc='docs/experiments/active/nmn_input_site_grid/NMN_INPUT_SITE_GRID.md',
        ancestor_desc='the unmodulated Monte-Carlo arm of the 25-run return-mode study',
        twin_lines=[],
    ),
    'gaenorm': dict(
        base_agent='configs/models/recurrent_ppo/recurrent_ppo_cmp_gaenorm.yaml',
        out_dir='configs/models/recurrent_ppo/nmn_input_site_grid_gaenorm',
        file_prefix='nmngaenorm',
        return_mode='GAE_NORM',
        grid_label='NMN input x site grid, GAE_NORM twin',
        design_doc='docs/experiments/active/nmn_input_site_grid/NMN_INPUT_SITE_GRID_GAENORM.md',
        ancestor_desc='the unmodulated GAE_NORM arm of the 25-run return-mode study',
        twin_lines=[
            "# THE ESTIMATOR-SWAPPED TWIN. This file has a one-for-one counterpart in",
            "# configs/models/recurrent_ppo/nmn_input_site_grid/ that is identical to it in every",
            "# key except one: `return_mode`, which is \"MC\" there and \"GAE_NORM\" here. That is",
            "# the recipe the trainer uses to turn a rollout into the two numbers PPO learns from",
            "# -- the critic's target, and the advantage. GAE_NORM rather than plain GAE, because",
            "# GAE would ALSO change the scale convention and the two grids would then differ in",
            "# two things at once. The one-key claim is checked mechanically, not asserted:",
            "#   ... generate_site_grid_arms.py --verify   (step [5], the twin diff)",
            "#",
        ],
    ),
}

# --- Factor 1: WHAT the modulator reads ------------------------------------
# Sensor names are the keys of get_observation_breakdown(params) under ENV_CONFIG.
# Naming a sensor that this environment config does not produce is a hard error in
# _resolve_modulator_input_indices, which is why these spellings are checked by
# --verify against a live construction rather than trusted from prose.
# Widths under basic/04: Satiation 1, Interoceptive Nociception 1,
# Extero Nociception 1, Olfaction 5, Collision 5, Proprioception 6, Visual 8 = 27.
#
# PROPRIOCEPTION IS DELIBERATELY IN NEITHER RESTRICTED SLICE (user decision,
# 2026-09-07): it is classically neither interoceptive nor exteroceptive, so it
# appears only under ALL. Consequence, stated so nobody rediscovers it the hard way:
# ALL and X differ by EIGHT dimensions (both body channels plus six proprioceptive
# ones), so an ALL-vs-X difference may never be attributed to interoception. The
# clean interoceptive contrast in this grid is I vs X, and only that one (confound C2).
SLICES = {
    'ALL': ("all", 27, "everything the agent senses"),
    'I':   (["Satiation", "Interoceptive Nociception"], 2,
            "the body's own signals only: fullness, and the smoothed internal ache "
            "that is the agent's only trace of its own wound"),
    'X':   (["Extero Nociception", "Olfaction", "Collision", "Visual"], 19,
            "the outside world only: contact pain, smell, collision, vision"),
}

# --- Factor 2: WHERE the modulator writes ----------------------------------
# (stem, sites, plain-language description). Cell numbers are the user's own.
TARGETS = [
    ('t2enc',   dict(encoder=True,  rnn=False, actor=False, critic=False),
     "cell 2 -- the sensory front-end (per-sensor stage and fusion hub, one atomic site)"),
    ('t3rnn',   dict(encoder=False, rnn=True,  actor=False, critic=False),
     "cell 3 -- the memory cell's emitted output (never its carried state)"),
    ('t4act',   dict(encoder=False, rnn=False, actor=True,  critic=False),
     "cell 4 -- the hidden layer of the action head, pre-activation"),
    ('t5crt',   dict(encoder=False, rnn=False, actor=False, critic=True),
     "cell 5 -- the hidden layer of the value head, pre-activation"),
    ('t16quad', dict(encoder=True,  rnn=True,  actor=True,  critic=True),
     "cell 16 -- all four sites at once"),
]

# Modulator hyperparameters, identical across all fifteen modulated arms (design §2.2).
MOD_COMMON = dict(
    mod_hidden_size=16,
    grouping_size=1,
    percept_bias_init=3.0,       # INERT under type FiLM (the gain head's output bias is
                                 # forced to 1.0); carried forward from the config family
                                 # and load-bearing for nothing here. See design C-note.
    percept_add_bias_init=0.0,   # stated explicitly so no run depends on the fallback
                                 # default at recurrent_ppo_network.py (confound C12)
    memory_bias_init=0.0,        # unused under rnn_mechanism "activation", still mandatory
    memory_clip=[-2.0, 2.0],     # ditto
)


def _header(grid, stem, target_desc, slice_code):
    """The comment block prepended to a generated file."""
    lines = [
        "# GENERATED FILE -- DO NOT EDIT BY HAND.",
        "# Regenerate with:",
        "#   python configs/models/recurrent_ppo/nmn_input_site_grid/generate_site_grid_arms.py",
        "# Drift check (writes nothing):  ... generate_site_grid_arms.py --check",
        "#",
        "# %s -- %s" % (grid['grid_label'], stem),
        "# Study: %s" % grid['design_doc'],
        "#",
    ]
    lines += list(grid['twin_lines'])
    if slice_code is None:
        lines += [
            "# THE UNMODULATED CONTROL (cell 1). No neuromodulator is constructed at all:",
            "# `modulation.type: null` collapses to the plain agent, so none of the",
            "# modulation keys the other fifteen carry is read here.",
            "#",
            "# This file is IDENTICAL to %s" % grid['base_agent'],
            "# -- %s -- except for" % grid['ancestor_desc'],
            "# `lr_critic` (0.0001 there, 0.0005 here, as in all sixteen files of this grid).",
            "# `lr_critic` is read by no code path today, so this run trains exactly as that",
            "# arm's five existing seeds did; the key is set for the benefit of a later rerun",
            "# from this saved config. That lineage is what makes this run a REPLICATION check",
            "# (the design's C7 gate): if it does not reproduce the known result, the training",
            "# path changed underneath the grid and every comparison in it is suspect.",
        ]
    else:
        sensors, width, slice_desc = SLICES[slice_code]
        lines += [
            "# WHERE the modulator writes: %s." % target_desc,
            "# WHAT the modulator reads:   slice %s -- %s (%d of the 27 observation numbers)."
            % (slice_code, slice_desc, width),
            "#",
            "# Everything else is identical across all sixteen arms of the grid and is inherited",
            "# from %s (see the generator)." % grid['base_agent'],
            "# The modulator uses ONE mechanism at every site -- FiLM, i.e. multiply each neuron",
            "# by a learned gain and add a learned offset -- so that WHERE it acts is varied",
            "# without also varying HOW it acts. `rnn_mechanism: activation` keeps the task GRU",
            "# the same plain cell the unmodulated control uses; the legacy gate-bias operator",
            "# would swap the cell and reintroduce a known initialisation confound.",
            "# Temperature modulation is OFF in every arm: it is a fifth, differently-shaped",
            "# channel (it divides the action scores) and leaving it on would confound all five",
            "# write targets with it.",
        ]
    lines += [
        "#",
        "# Seed is config-owned (configs/train/default.yaml: 42); no --seed is passed at launch.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _build(grid, stem, sites, slice_code):
    """Return (filename, file text) for one arm of one grid."""
    base = yaml.safe_load(open(grid['base_agent']))
    agent = copy.deepcopy(base['agent'])
    agent['lr_critic'] = LR_CRITIC

    # Guard the one key the two grids are supposed to differ in. If a base config is ever
    # edited, this fails here rather than producing a grid that is quietly the wrong twin.
    if agent.get('return_mode') != grid['return_mode']:
        raise SystemExit(
            "%s has return_mode %r but grid expects %r -- refusing to generate."
            % (grid['base_agent'], agent.get('return_mode'), grid['return_mode']))

    if slice_code is None:
        # Requirement 5: the control carries `modulation: {type: null}` and nothing else.
        agent['modulation'] = {'type': None}
        fname = '%s_%s.yaml' % (grid['file_prefix'], stem)
        target_desc = None
    else:
        sensors = SLICES[slice_code][0]
        mod = {'type': 'FiLM'}
        mod.update(copy.deepcopy(MOD_COMMON))
        mod['input_sensors'] = (_QuotedStr(sensors) if isinstance(sensors, str)
                                else [_QuotedStr(n) for n in sensors])
        mod['sites'] = {'encoder': sites['encoder'], 'rnn': sites['rnn'],
                        'actor': sites['actor'], 'critic': sites['critic']}
        mod['rnn_mechanism'] = 'activation'
        # `clip` is deliberately ABSENT: it is only read when enabled is true, and the
        # project forbids writing `key: null` to mean "optional".
        mod['temperature'] = {'enabled': False}
        agent['modulation'] = mod
        fname = '%s_%s_%s.yaml' % (grid['file_prefix'], stem, slice_code)
        target_desc = dict((t[0], t[2]) for t in TARGETS)[stem]

    buf = io.StringIO()
    dump_config_yaml({'agent': agent}, buf)
    return fname, _header(grid, stem, target_desc, slice_code) + buf.getvalue()


def arm_specs():
    """(stem, sites, slice_code) for all sixteen arms, in manifest order."""
    out = [('t1none', None, None)]
    for stem, sites, _desc in TARGETS:
        for slice_code in ('ALL', 'I', 'X'):
            out.append((stem, sites, slice_code))
    return out


def build_all(grid):
    """Return an ordered list of (filename, text) for all sixteen arms of one grid."""
    return [_build(grid, stem, sites, sc) for stem, sites, sc in arm_specs()]


def arm_filename(grid, stem, slice_code):
    if slice_code is None:
        return '%s_%s.yaml' % (grid['file_prefix'], stem)
    return '%s_%s_%s.yaml' % (grid['file_prefix'], stem, slice_code)


def write_all(grid_keys, check_only=False):
    stale = []
    total = 0
    for gk in grid_keys:
        grid = GRIDS[gk]
        files = build_all(grid)
        total += len(files)
        if not check_only and not os.path.isdir(grid['out_dir']):
            os.makedirs(grid['out_dir'])
        for fname, text in files:
            path = os.path.join(grid['out_dir'], fname)
            if check_only:
                if not os.path.exists(path):
                    stale.append("%s: MISSING on disk" % path)
                    continue
                on_disk = open(path).read()
                if on_disk != text:
                    diff = "\n".join(difflib.unified_diff(
                        on_disk.splitlines(), text.splitlines(),
                        fromfile=path + ' (on disk)', tofile=path + ' (generated)',
                        lineterm=''))
                    stale.append("%s: DIFFERS from generated output\n%s" % (path, diff))
            else:
                with open(path, 'w') as f:
                    f.write(text)
                print("wrote %s" % path)
    if check_only:
        if stale:
            print("\n".join(stale))
            raise SystemExit("--check FAILED: %d of %d files drifted from the generator."
                             % (len(stale), total))
        print("--check OK: all %d files (grids: %s) match the generator byte-for-byte."
              % (total, ", ".join(grid_keys)))


# --------------------------------------------------------------------------
# Verification: load each generated config through the REAL training path and
# construct the model. A YAML that merely parses proves nothing here -- the whole
# point of the refactor's mandatory keys is that a wrong config fails loudly, and
# the place to collect that failure is now, not at launch.
# --------------------------------------------------------------------------

def _flatten(d, prefix=''):
    flat = {}
    for k, v in d.items():
        key = "%s.%s" % (prefix, k) if prefix else str(k)
        if isinstance(v, dict):
            flat.update(_flatten(v, key))
        else:
            flat[key] = v
    return flat


def _diff_keys(a, b):
    """Keys that are missing on one side or hold different values."""
    return sorted(set(a) ^ set(b)) + sorted(k for k in set(a) & set(b) if a[k] != b[k])


def verify(grid_keys):
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from src.utils.config import Config, get_default_config
    from src.environment.config_loader import load_env_config, load_env_params
    from src.environment.sensor import get_observation_breakdown
    from src.environment.wrapper import ParallelEnv
    from src.models.recurrent_ppo_network import ActorCriticRNN
    from src.models.modulated_gru_cell import ModulatedGRUCell

    failures = []

    def check(cond, msg):
        if cond:
            print("    ok   %s" % msg)
        else:
            print("    FAIL %s" % msg)
            failures.append(msg)

    env_layers = None
    breakdown_seen = None

    for gk in grid_keys:
        grid = GRIDS[gk]
        out_dir = grid['out_dir']
        print("")
        print("=" * 78)
        print("GRID %r  (return_mode %s)  ->  %s" % (gk, grid['return_mode'], out_dir))
        print("=" * 78)

        # --- 1. The control against its ancestor: the C7 replication gate ---
        print("[1] control vs %s -- flattened diff must be exactly {agent.lr_critic}"
              % grid['base_agent'])
        ctrl_name = arm_filename(grid, 't1none', None)
        ctrl = _flatten(yaml.safe_load(open(os.path.join(out_dir, ctrl_name))))
        anc = _flatten(yaml.safe_load(open(grid['base_agent'])))
        dk = _diff_keys(ctrl, anc)
        check(dk == ['agent.lr_critic'],
              "control differs from ancestor only in agent.lr_critic (got %s)" % dk)
        check(ctrl.get('agent.lr_critic') == LR_CRITIC and anc.get('agent.lr_critic') == 0.0001,
              "lr_critic 0.0001 (ancestor) -> %s (grid)" % LR_CRITIC)
        check(ctrl.get('agent.return_mode') == grid['return_mode'],
              "control return_mode == %s" % grid['return_mode'])

        # --- 2. Every modulated arm against one reference modulated arm -----
        ref_name = arm_filename(grid, 't2enc', 'ALL')
        print("[2] each modulated arm vs %s -- diff only in sites/input_sensors" % ref_name)
        ref = _flatten(yaml.safe_load(open(os.path.join(out_dir, ref_name))))
        allowed = {'agent.modulation.sites.encoder', 'agent.modulation.sites.rnn',
                   'agent.modulation.sites.actor', 'agent.modulation.sites.critic',
                   'agent.modulation.input_sensors'}
        for stem, _sites, _d in TARGETS:
            for sc in ('ALL', 'I', 'X'):
                fname = arm_filename(grid, stem, sc)
                arm = _flatten(yaml.safe_load(open(os.path.join(out_dir, fname))))
                sym = set(arm) ^ set(ref)
                changed = set(k for k in set(arm) & set(ref) if arm[k] != ref[k])
                check(not sym and changed <= allowed,
                      "%s differs from reference only in %s"
                      % (fname, sorted(changed) if not sym
                         else "EXTRA/MISSING keys %s" % sorted(sym)))
                check(arm.get('agent.return_mode') == grid['return_mode'],
                      "%s return_mode == %s" % (fname, grid['return_mode']))

        # --- 3. Construct every model through the real path -----------------
        print("[3] construct all sixteen models (env %s)" % ENV_CONFIG)
        for stem, sites, sc in arm_specs():
            fname = arm_filename(grid, stem, sc)
            print("  %s" % fname)
            # Mirror train.py's merge order for everything that reaches the model.
            cfg = get_default_config()
            cfg.merge(Config.load_yaml('configs/train/default.yaml'))
            cfg.merge(Config.load_yaml('configs/train/recurrent_ppo.yaml'))
            cfg.merge(load_env_config(ENV_CONFIG))
            cfg.merge(Config.load_yaml(os.path.join(out_dir, fname)))

            params = load_env_params(cfg)
            obs_breakdown = get_observation_breakdown(params)
            if breakdown_seen is None:
                breakdown_seen = obs_breakdown
                print("    observation breakdown: %s = %d"
                      % (obs_breakdown, sum(obs_breakdown.values())))
            else:
                check(obs_breakdown == breakdown_seen,
                      "observation breakdown identical to the first arm's")

            if env_layers is None:
                env = ParallelEnv(params)
                _st, _obs = env.reset(jax.random.PRNGKey(0), 2)
                env_layers = int(_obs.shape[-1])
            input_dim = env_layers
            action_dim = 4 + int(params.rest_action_enabled) + int(params.eat_action_enabled)

            check(cfg.get_mandatory('agent.return_mode') == grid['return_mode'],
                  "merged config resolves return_mode to %s" % grid['return_mode'])

            modulation_config = cfg.get('agent.modulation')
            if modulation_config is not None and modulation_config.get('type') is None:
                modulation_config = None

            model = ActorCriticRNN(
                input_dim=input_dim,
                action_dim=action_dim,
                hidden_size=cfg.get_mandatory('agent.hidden_size'),
                rngs=nnx.Rngs(jax.random.PRNGKey(42)),
                rnn_type=cfg.get_mandatory('agent.rnn_type'),
                activation=cfg.get_mandatory('agent.activation'),
                modulation_config=modulation_config,
                observation_breakdown=obs_breakdown,
                encoding_config=cfg.to_dict().get('agent', {}),
            )

            if sc is None:
                check(model.modulation_enabled is False, "modulation disabled (the control)")
                check(not hasattr(model, 'modulator'), "no modulator module constructed")
                check((model.site_encoder, model.site_rnn, model.site_actor, model.site_critic)
                      == (False, False, False, False), "all four site flags False")
            else:
                want = (sites['encoder'], sites['rnn'], sites['actor'], sites['critic'])
                got = (model.site_encoder, model.site_rnn, model.site_actor, model.site_critic)
                check(got == want, "site flags on the model %s == intended %s" % (got, want))
                got_mod = (model.modulator.site_encoder, model.modulator.site_rnn,
                           model.modulator.site_actor, model.modulator.site_critic)
                check(got_mod == want, "site flags inside the modulator %s == intended %s"
                      % (got_mod, want))

                want_w = SLICES[sc][1]
                check(len(model.mod_input_idx) == want_w,
                      "resolved modulator input width %d == %d for slice %s"
                      % (len(model.mod_input_idx), want_w, sc))
                # Read the width off the CONSTRUCTED weights too, not just the index tuple.
                gru_in = int(model.modulator.gru.in_features)
                check(gru_in == want_w,
                      "modulator GRU in_features %d == %d" % (gru_in, want_w))
                kernel_rows = set()
                for leaf_path, leaf in jax.tree_util.tree_flatten_with_path(
                        nnx.state(model.modulator.gru, nnx.Param))[0]:
                    if getattr(leaf, 'ndim', 0) == 2:
                        kernel_rows.add(int(leaf.shape[0]))
                check(want_w in kernel_rows,
                      "a modulator GRU input kernel really has %d rows (rows seen: %s)"
                      % (want_w, sorted(kernel_rows)))
                check(model._mod_input_is_all == (sc == 'ALL'),
                      "gather is %s for slice %s"
                      % ("skipped" if sc == 'ALL' else "applied", sc))
                check(model.rnn_mechanism == 'activation', "rnn_mechanism == activation")
                check(model.temperature_enabled is False, "temperature disabled")
                check(not isinstance(model.rnn_cell, ModulatedGRUCell),
                      "task RNN is the plain GRU cell, as in the control")
                check(model.modulator.grouping_size == 1, "grouping_size == 1")
                check(model.modulator.mod_hidden_size == 16, "mod_hidden_size == 16")
                check(model.modulation_type == 'FiLM', "modulation type == FiLM")

            # A construction that cannot take a step is not a construction.
            h = model.initial_state(batch_size=2)
            x = jnp.zeros((2, input_dim))
            logits, value, _h_new, _mod = model(x, h)
            check(logits.shape == (2, action_dim) and bool(jnp.all(jnp.isfinite(logits))),
                  "forward pass returns finite logits of shape (2, %d)" % action_dim)
            check(bool(jnp.all(jnp.isfinite(value))), "forward pass returns a finite value")

        # --- 4. Negative controls: the mandatory keys really do fail loudly --
        # A passing config proves the happy path. These three prove the property the
        # whole design leans on -- that a WRONG config is refused at construction time
        # rather than training quietly with the wrong sensors or the wrong sites.
        print("[4] negative controls -- malformed modulation blocks must raise")
        good_path = os.path.join(out_dir, arm_filename(grid, 't2enc', 'I'))
        good = yaml.safe_load(open(good_path))['agent']['modulation']

        def _expect_error(label, mutate):
            bad = copy.deepcopy(good)
            mutate(bad)
            try:
                ActorCriticRNN(
                    input_dim=input_dim, action_dim=action_dim, hidden_size=128,
                    rngs=nnx.Rngs(jax.random.PRNGKey(0)), rnn_type='GRU', activation='relu',
                    modulation_config=bad, observation_breakdown=breakdown_seen,
                    encoding_config=cfg.to_dict().get('agent', {}))
            except ValueError:
                check(True, "%s raises ValueError" % label)
            else:
                check(False,
                      "%s raises ValueError (it did NOT -- constructed silently)" % label)

        _expect_error("a misspelled sensor name",
                      lambda m: m.__setitem__('input_sensors', ['Interoceptive nociception']))
        _expect_error("a missing sites key", lambda m: m['sites'].pop('critic'))
        _expect_error("every site off with temperature off",
                      lambda m: m['sites'].update(encoder=False, rnn=False, actor=False,
                                                  critic=False))

    # --- 5. The twin diff: the whole point of the second grid ---------------
    # Each gaenorm config against its mc counterpart, flattened key-by-key. If this
    # passes, a difference between the two grids is attributable to the return estimator
    # and to nothing else. If it fails, the two grids are two loosely similar studies.
    if 'mc' in grid_keys and 'gaenorm' in grid_keys:
        print("")
        print("=" * 78)
        print("[5] twin diff -- every gaenorm config vs its mc counterpart must differ")
        print("    in exactly {agent.return_mode}")
        print("=" * 78)
        gmc, ggn = GRIDS['mc'], GRIDS['gaenorm']
        for stem, _sites, sc in arm_specs():
            mc_name = arm_filename(gmc, stem, sc)
            gn_name = arm_filename(ggn, stem, sc)
            a = _flatten(yaml.safe_load(open(os.path.join(gmc['out_dir'], mc_name))))
            b = _flatten(yaml.safe_load(open(os.path.join(ggn['out_dir'], gn_name))))
            dk = _diff_keys(a, b)
            check(dk == ['agent.return_mode'],
                  "%s vs %s: diff == %s" % (mc_name, gn_name, dk))
            check(a.get('agent.return_mode') == 'MC'
                  and b.get('agent.return_mode') == 'GAE_NORM',
                  "%s: MC -> GAE_NORM in the right direction" % gn_name)
    else:
        print("")
        print("[5] twin diff SKIPPED -- needs both grids in scope (--grid all).")

    print("")
    if failures:
        for f in failures:
            print("FAILED: %s" % f)
        raise SystemExit("verification FAILED: %d checks" % len(failures))
    print("verification OK: every config in %s loads, constructs and steps."
          % ", ".join(grid_keys))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--grid', default='all', choices=['all'] + sorted(GRIDS),
                    help="which grid to act on (default: all)")
    ap.add_argument('--check', action='store_true',
                    help="write nothing; fail if any file on disk differs from the generator")
    ap.add_argument('--verify', action='store_true',
                    help="--check, plus construct every model through the real training path")
    args = ap.parse_args()
    keys = sorted(GRIDS) if args.grid == 'all' else [args.grid]
    write_all(keys, check_only=args.check or args.verify)
    if args.verify:
        verify(keys)
