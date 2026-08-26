"""Generate the weakened-vision arms of the directional-sensors sweep.

These are written as FULL standalone configs rather than `extends:` overlays.
Reason: five of the six change per-entity keys (`visual_properties` for V=1,
`blocks_sight` for occlusion), and deep-merge replaces list values wholesale
(CONFIG_GUIDE.md §1 "list-replace footgun") -- an overlay would have to restate
every entity anyway, with the added risk of silently dropping one.

Run from the repo root:  python configs/environment/experiment/sensory_directional/generate_weakened_vision_arms.py
"""
import copy, os, sys
import yaml

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 4)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
from src.utils.config import dump_config_yaml
from src.environment.config_loader import load_env_config

OUT = 'configs/environment/experiment/sensory_directional'
# Which entities occlude. Named per arm so the blocker SET is itself a variable:
# a bush that conceals the agent from predators is physically opaque and arguably
# ought to block the agent's own view too, so "does vegetation block sight?" is a
# question worth measuring rather than deciding.
SCENERY = {'rock', 'tree'}                       # solid scenery only
SCENERY_VEG = SCENERY | {'bush'}                 # + vegetation
EVERYTHING = SCENERY_VEG | {'pred', 'rabbit', 'hiding_predator'}   # + living things

ARMS = [
    ('E_clamp',        dict(visual_value_mode='clamp'), None, False,
     'Presence instead of a count. Identity KEPT (V=8). vs C: removes "how many".'),
    ('F_presence_v1',  dict(), 1, False,
     'Identity REMOVED (V=1, terrain zeroed); count kept. vs C: removes "what".'),
    ('G_presence_binary', dict(visual_value_mode='clamp'), 1, False,
     'Identity and count both removed — the RF-like sensor: "something is there".'),
    ('H_occlusion05',  dict(visual_occlusion_enabled=True, visual_occlusion_cone_deg=5.0,
                            visual_occlusion_strength=1.0), None, SCENERY,
     'Occlusion, 5 deg cone, ROCKS only (~19% of live entities hidden).'),
    ('I_occlusion15',  dict(visual_occlusion_enabled=True, visual_occlusion_cone_deg=15.0,
                            visual_occlusion_strength=1.0), None, SCENERY,
     'Occlusion, 15 deg cone, ROCKS only (~44% hidden).'),
    ('J_all_weakened', dict(visual_value_mode='clamp', visual_occlusion_enabled=True,
                            visual_occlusion_cone_deg=5.0, visual_occlusion_strength=1.0), 1, SCENERY,
     'Every weakening stacked: no identity, no count, occluded. The floor.'),
    # --- blocker-set sweep: H and I hold the cone fixed and vary WHAT blocks ---
    ('K_occl05_veg',   dict(visual_occlusion_enabled=True, visual_occlusion_cone_deg=5.0,
                            visual_occlusion_strength=1.0), None, SCENERY_VEG,
     'Occlusion, 5 deg, rocks AND BUSHES block (~31% hidden). vs H: does vegetation block?'),
    ('L_occl15_veg',   dict(visual_occlusion_enabled=True, visual_occlusion_cone_deg=15.0,
                            visual_occlusion_strength=1.0), None, SCENERY_VEG,
     'Occlusion, 15 deg, rocks AND BUSHES block (~58% hidden). vs I: same question, wider cone.'),
    ('M_occl05_all',   dict(visual_occlusion_enabled=True, visual_occlusion_cone_deg=5.0,
                            visual_occlusion_strength=1.0), None, EVERYTHING,
     'Occlusion, 5 deg, EVERYTHING blocks incl. creatures (~38% hidden). The physically complete version.'),
    ('N_occl15_all',   dict(visual_occlusion_enabled=True, visual_occlusion_cone_deg=15.0,
                            visual_occlusion_strength=1.0), None, EVERYTHING,
     'Occlusion, 15 deg, EVERYTHING blocks (~65% hidden). The darkest arm.'),
]

HEADER = """# ============================================================================
# DIRECTIONAL SENSORS sweep — weakened-vision arm {name}
# {desc}
#
# CONTROL is arm C (C_vision_sharp.yaml): visual_sensor_range 2, sharp, V=8,
# value_mode sum. Every arm here removes exactly one thing from C, except J
# which stacks them. The question is how much can be taken away from strong
# vision before it stops dominating the baseline.
#
# obs width: {obs}  (C is 123, baseline A is 27)
# {matched}
#
# GENERATED — do not hand-edit. Regenerate with:
#   python {gen}
# Written standalone rather than as an `extends:` overlay because per-entity
# keys change and deep-merge replaces lists wholesale.
#
# Single seed (42, config-owned). Plan: docs/develop/active/sensors/DIRECTIONAL_SENSORS_PLAN.md
# ============================================================================
"""


def build(sensory_ov, vec, blockers):
    d = copy.deepcopy(load_env_config('configs/environment/default.yaml').to_dict())
    d.pop('extends', None)
    d['sensory']['visual_sensor_range'] = 2
    d['sensory'].update(sensory_ov)
    if vec is not None:
        d['sensory']['visual_vector_size'] = vec
        d['sensory']['visual_background_properties'] = [[0.0] * vec for _ in range(3)]
    for group in ('resources', 'entities', 'obstacles'):
        for e in d['environment'].get(group) or []:
            if vec is not None:
                e['visual_properties'] = [1.0] * vec
                e['visual_properties_std'] = [0.0] * vec
            if blockers:
                nm = e.get('name') or e.get('tag')
                e['blocks_sight'] = bool(nm in blockers)
    return d


def main():
    import jax
    from src.utils.config import Config
    from src.environment.config_loader import load_env_params
    from src.environment.wrapper import ParallelEnv
    for name, ov, vec, blockers, desc in ARMS:
        d = build(ov, vec, blockers)
        p = load_env_params(Config(copy.deepcopy(d)))
        env = ParallelEnv(p)
        _, o = env.reset(jax.random.PRNGKey(0), 2)
        obs = int(o.shape[-1])
        path = os.path.join(OUT, f'{name}.yaml')
        with open(path, 'w') as fh:
            fh.write(HEADER.format(
                name=name, desc=desc, obs=obs,
                matched=('DIMENSION-MATCHED to C — any difference is the mechanism, not width.'
                         if obs == 123 else
                         'NOT dimension-matched to C: removing identity removes dimensions. '
                         'Read results with that confound in mind.'),
                gen='configs/environment/experiment/sensory_directional/generate_weakened_vision_arms.py'))
            dump_config_yaml(d, fh)
        print(f"  {name:<22} obs={obs:>4}  blockers={','.join(sorted(blockers)) if blockers else '-':<28} "
              f"V={p.visual_vector_size} mode={p.visual_value_mode}")


if __name__ == '__main__':
    main()
