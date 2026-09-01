"""Vision-degradation ladder on basic/04. DIRECTIONAL_SENSORS.

Design rationale (all numbers measured, see docs):
  * Olfaction is held at grid_range=1 for EVERY vision arm so the B->ceiling
    ladder is a one-factor comparison. A_baseline (olf 0) is the only arm
    without it, kept as the no-olfaction reference.
  * visual_blur_radial_scale is the gradient lever: it is monotone and spans
    the full decodability range (probe: 78.6% at 4.0 -> 99.4% sharp).
  * The occlusion cone is a two-valued switch on a range-2 diamond (only
    exact collinearity fires below 45 deg), so it is FIXED at 15 deg and the
    blocker SET is varied instead: 2.61% / 4.62% / 7.30% in-diamond hidden.
"""
import copy, os, sys, yaml
sys.path.insert(0, '/media/nas01/projects/Interoceptive-AI/grid_world_pain')
os.chdir('/media/nas01/projects/Interoceptive-AI/grid_world_pain')
from src.environment.config_loader import load_env_config

BASE_TASK = 'configs/environment/experiment/basic/04-jump_attack_10x10.yaml'
OUT = 'configs/environment/experiment/sensory_ladder/'
SCEN = {'rock'}; VEG = SCEN | {'bush'}
ALL  = VEG | {'pred', 'rabbit', 'hiding_predator'}
CONE = 15.0            # any value < 45 deg is geometrically equivalent here

#     name                olf vrange vec  rho  scale  mode    occl-blockers
ARMS = [
    ('A_baseline',          0, 0, 8, None, None, 'sum',   None),
    ('B_olf_only',          1, 0, 8, None, None, 'sum',   None),
    ('V1_blur40',           1, 2, 8, 3.0,  4.0,  'sum',   None),
    ('V2_blur20',           1, 2, 8, 3.0,  2.0,  'sum',   None),
    ('V3_blur10',           1, 2, 8, 3.0,  1.0,  'sum',   None),
    ('V4_blur05',           1, 2, 8, 3.0,  0.5,  'sum',   None),
    ('V5_sharp',            1, 2, 8, None, None, 'sum',   None),
    ('R1_range1',           1, 1, 8, None, None, 'sum',   None),
    ('P1_blur05_iso',       1, 2, 8, 1.0,  0.5,  'sum',   None),
    ('Q1_presence_sum',     1, 2, 1, 3.0,  0.5,  'sum',   None),
    ('Q2_presence_binary',  1, 2, 1, 3.0,  0.5,  'clamp', None),
    ('O1_occl_rock',        1, 2, 8, 3.0,  0.5,  'sum',   SCEN),
    ('O2_occl_veg',         1, 2, 8, 3.0,  0.5,  'sum',   VEG),
    ('O3_occl_all',         1, 2, 8, 3.0,  0.5,  'sum',   ALL),
]

def build(olf, vr, vec, rho, scale, mode, blockers):
    d = copy.deepcopy(load_env_config(BASE_TASK).to_dict()); d.pop('extends', None)
    s = d['sensory']
    s['olfactory_grid_range'] = olf
    s['visual_sensor_range'] = vr
    s['visual_vector_size'] = vec
    s['visual_value_mode'] = mode
    s['visual_blur_enabled'] = rho is not None
    if rho is not None:
        s['visual_blur_anisotropy'] = rho
        s['visual_blur_radial_scale'] = scale
        s['visual_blur_sigma_floor'] = 0.5
    s['visual_occlusion_enabled'] = blockers is not None
    if blockers is not None:
        s['visual_occlusion_cone_deg'] = CONE
        s['visual_occlusion_strength'] = 1.0
    if vec != 8:
        s['visual_background_properties'] = [[0.0] * vec for _ in range(3)]
    for grp in ('resources', 'entities', 'obstacles'):
        for e in d['environment'].get(grp) or []:
            if vec != 8:
                e['visual_properties'] = [1.0] * vec
                e['visual_properties_std'] = [0.0] * vec
            if blockers is not None:
                nm = e.get('name') or e.get('tag')
                if nm not in {'food','hiding_predator','pred','rabbit','bush','rock'}:
                    raise KeyError(f'unknown entity {nm!r}')
                e['blocks_sight'] = bool(nm in blockers)
    if blockers:
        known = {'food','hiding_predator','pred','rabbit','bush','rock'}
        missing = set(blockers) - known
        if missing:
            raise KeyError(f'blocker names match no entity: {missing}')
    return d

if __name__ == '__main__':
    for nm, olf, vr, vec, rho, sc, mode, bl in ARMS:
        yaml.safe_dump(build(olf, vr, vec, rho, sc, mode, bl),
                       open(OUT + nm + '.yaml', 'w'), sort_keys=False)
        print('wrote', nm)
