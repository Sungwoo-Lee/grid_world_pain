"""Figures for the v3.1 directional-sensors implementation report.

Everything here comes from the REAL environment via load_env_params +
ParallelEnv -- not from the mechanism-study sandboxes.
"""
import os, sys, json
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 5)))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)          # config paths in this script are repo-relative
_OUT = os.path.dirname(os.path.abspath(__file__))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np, jax, jax.numpy as jnp
from src.utils.config import Config
from src.environment.config_loader import load_env_params
from src.environment.sensor import get_observation, get_observation_breakdown, get_visual_offsets
from src.environment.wrapper import ParallelEnv

plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.titlesize': 9.5,
                     'savefig.bbox': 'tight', 'savefig.facecolor': 'white'})
TEAL, AMBER, PLUM, GREY = '#1f7a6c', '#c9852b', '#7b4b91', '#6e6579'
VIS_CH = ['GRS', 'SND', 'PLN', 'FOD', 'DNG', 'PRD', 'RCK', 'NEU']
OLF_CH = ['FOOD', 'AN-A', 'AN-B', 'BUSH', 'TREE']
SEED = 3


def P(**ov):
    c = Config.load_yaml('configs/environment/default.yaml')
    for k, v in ov.items():
        c.set(k, v)
    return load_env_params(c)


def scene(p, seed=SEED):
    env = ParallelEnv(p)
    st, obs = env.reset(jax.random.PRNGKey(seed), 1)
    return jax.tree.map(lambda x: x[0], st), np.asarray(obs[0])


def slice_of(p, obs, name):
    bd = get_observation_breakdown(p); i = 0
    for k, d in bd.items():
        if k == name:
            return obs[i:i+d]
        i += d
    raise KeyError(name)


def draw_grid(ax, st, p, title):
    H, W = p.height, p.width
    lt = np.asarray(p.grid_location_type)
    ax.imshow(np.where(lt == 1, .92, .84), cmap='Greys_r', vmin=0, vmax=1,
              extent=[-.5, W-.5, H-.5, -.5])
    for pos, act, col, mk, lab in [
            (st.res_pos, st.res_active, '#4ad97a', 'o', 'food/danger'),
            (st.animal_pos, st.animal_active, '#d94a4a', '^', 'animal'),
            (st.obs_pos, st.obs_active, '#8a8f98', 's', 'obstacle')]:
        pp, aa = np.asarray(pos), np.asarray(act)
        for (r, c), a in zip(pp, aa):
            if a and 0 <= r < H and 0 <= c < W:
                ax.plot(c, r, mk, ms=6, mfc=col, mec='black', mew=.5)
    ar, ac = np.asarray(st.agent_pos)
    ax.plot(ac, ar, marker='X', ms=13, mfc='#111', mec='white', mew=1.3)
    for rr, col in [(p.olfactory_grid_range, TEAL), (p.visual_sensor_range, PLUM)]:
        if rr > 0:
            ax.plot([ac, ac+rr, ac, ac-rr, ac], [ar-rr, ar, ar+rr, ar, ar-rr],
                    color=col, lw=1.6, ls='--')
    ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=9)


def heat(ax, M, rows, cols, title, fmt='{:.2f}', cmap='viridis'):
    M = np.atleast_2d(M)
    ax.imshow(M, cmap=cmap, aspect='auto', vmin=0, vmax=max(M.max(), 1e-9))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, fontsize=7, rotation=0)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=7)
    mx = max(M.max(), 1e-9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if M[i, j] > 1e-4:
                ax.text(j, i, fmt.format(M[i, j]), ha='center', va='center', fontsize=6.2,
                        color='white' if M[i, j] / mx < .55 else 'black')
    ax.set_title(title, fontsize=9)


def cell_names(r):
    out = []
    for dr, dc in np.asarray(get_visual_offsets(r)):
        if dr == 0 and dc == 0: out.append('centre')
        else: out.append(f'({int(dr):+d},{int(dc):+d})')
    return out


# ------------------------------------------------------------------ FIG 1 ---
def fig1():
    p_b = P()
    p_a = P(**{'sensory.olfactory_grid_range': 1, 'sensory.visual_sensor_range': 2,
               'sensory.visual_blur_enabled': True})
    st_b, ob = scene(p_b); st_a, oa = scene(p_a)
    fig = plt.figure(figsize=(13.6, 6.6))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 1.5], hspace=.42, wspace=.30)

    draw_grid(fig.add_subplot(gs[:, 0]), st_a, p_a,
              'the scene\nteal = olfactory diamond, purple = visual')

    heat(fig.add_subplot(gs[0, 1]), slice_of(p_b, ob, 'Olfaction')[None, :],
         ['agent cell'], OLF_CH, f'BEFORE — olfaction, {len(slice_of(p_b,ob,"Olfaction"))} numbers')
    heat(fig.add_subplot(gs[1, 1]), slice_of(p_b, ob, 'Visual')[None, :],
         ['agent cell'], VIS_CH, f'BEFORE — vision, {len(slice_of(p_b,ob,"Visual"))} numbers')

    olf_a = slice_of(p_a, oa, 'Olfaction').reshape(-1, len(OLF_CH))
    vis_a = slice_of(p_a, oa, 'Visual').reshape(-1, len(VIS_CH))
    heat(fig.add_subplot(gs[0, 2]), olf_a, cell_names(1), OLF_CH,
         f'AFTER — olfaction, {olf_a.size} numbers  ({olf_a.shape[0]} cells x {olf_a.shape[1]} channels)')
    heat(fig.add_subplot(gs[1, 2]), vis_a, cell_names(2), VIS_CH,
         f'AFTER — vision, {vis_a.size} numbers  ({vis_a.shape[0]} cells x {vis_a.shape[1]} channels)')

    fig.suptitle('FIG 1 — What the agent actually receives, before and after (real environment, seed 3)',
                 fontsize=11.5, y=1.0)
    fig.text(.5, -.045,
             'Before, each sense is a single row: one reading per channel at the agent\'s own cell, and no way to tell which direction anything lies in. After, each sense is a\n'
             'TABLE — one row per diamond cell, one column per channel. Reading down a column shows one thing spread across space; reading across a row shows everything\n'
             'present in one cell. The observation goes from 27 numbers to 143.\n'
             'The blank rows are not missing data: this agent sits near the top-left edge, so cells like (-1,+0) and (-2,+0) fall outside the grid and read exactly zero, '
             'which is the\nout-of-bounds rule working. Note also that vision reports GRS=1.00 in every in-bounds cell — terrain is always present — while the entity channels '
             'carry the blurred content.',
             ha='center', fontsize=8.5)
    fig.savefig(os.path.join(_OUT, 'fig1_observation_before_after.png')); plt.close(fig); print('ok fig1')


# ------------------------------------------------------------------ FIG 2 ---
def fig2():
    p_off = P(**{'sensory.visual_sensor_range': 2})
    p_on = P(**{'sensory.visual_sensor_range': 2, 'sensory.visual_blur_enabled': True})
    st, o_off = scene(p_off); _, o_on = scene(p_on)
    v_off = slice_of(p_off, o_off, 'Visual').reshape(-1, 8)
    v_on = slice_of(p_on, o_on, 'Visual').reshape(-1, 8)
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4),
                             gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
    draw_grid(axes[0], st, p_on, 'the scene')
    heat(axes[1], v_off, cell_names(2), VIS_CH, 'blur OFF — exact cell match')
    heat(axes[2], v_on, cell_names(2), VIS_CH, 'blur ON — anisotropic point-spread')
    fig.suptitle('FIG 2 — The visual sensor with the blur off and on, same scene', fontsize=11, y=1.02)
    fig.text(.5, -.10,
             'Off, an entity appears in exactly one cell and nowhere else — crisp and, for an abstraction of a retina, implausibly certain. On, each entity is spread along the line of\n'
             'sight to it, so several cells carry a share. Identity never mixes: the blur happens INSIDE each channel, so a smeared predator is still unambiguously a predator.\n'
             'Note the off panel is mostly empty — anything not exactly on a diamond cell is invisible, which is the information the blur recovers as a soft, directional hint.',
             ha='center', fontsize=8.5)
    fig.savefig(os.path.join(_OUT, 'fig2_blur_off_on.png')); plt.close(fig); print('ok fig2')


# ------------------------------------------------------------------ FIG 3 ---
def fig3():
    """Olfactory gradient: put one food source at a known bearing and read the diamond."""
    p = P(**{'sensory.olfactory_grid_range': 1})
    st, _ = scene(p)
    agent = jnp.array([5, 5])
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6))
    for ax, (dr, dc, lab) in zip(axes, [(-3, 0, 'food 3 north'), (0, 3, 'food 3 east'),
                                        (2, 2, 'food SE diagonal'), (-1, 0, 'food adjacent north')]):
        st2 = st.replace(
            agent_pos=agent,
            res_pos=jnp.full_like(st.res_pos, 99).at[0].set(jnp.array([5+dr, 5+dc])),
            res_active=jnp.zeros_like(st.res_active).at[0].set(True),
            animal_active=jnp.zeros_like(st.animal_active),
            obs_active=jnp.zeros_like(st.obs_active))
        obs = np.asarray(get_observation(st2, p, False))
        olf = slice_of(p, obs, 'Olfaction').reshape(-1, len(OLF_CH))[:, 0]
        offs = np.asarray(get_visual_offsets(1))
        cm = plt.get_cmap('viridis'); mx = max(olf.max(), 1e-9)
        for (r, c), v in zip(offs, olf):
            ax.add_patch(Rectangle((c-.46, -r-.46), .92, .92, facecolor=cm(v/mx),
                                   edgecolor='0.75', lw=.6))
            ax.text(c, -r, f'{v:.3f}', ha='center', va='center', fontsize=7,
                    color='white' if v/mx < .55 else 'black')
        ax.annotate('', xy=(np.sign(dc)*1.75, -np.sign(dr)*1.75), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color='#4ad97a', lw=2))
        ax.set_xlim(-2.1, 2.1); ax.set_ylim(-2.1, 2.1)
        ax.set_aspect('equal'); ax.axis('off'); ax.set_title(lab, fontsize=9)
    fig.suptitle('FIG 3 — The olfactory diamond points at the source (food channel, real sensor)',
                 fontsize=11, y=1.04)
    fig.text(.5, -.12,
             'One food source, everything else deactivated, agent at the centre of a 10x10 grid. The green arrow is the true bearing. In each case the brightest cell is the one\n'
             'toward the source and the dimmest is the one away from it — the single number the sensor used to return could not distinguish any of these four situations.',
             ha='center', fontsize=8.5)
    fig.savefig(os.path.join(_OUT, 'fig3_olfactory_gradient.png')); plt.close(fig); print('ok fig3')


# ------------------------------------------------------------------ FIG 4 ---
def fig4():
    before = json.load(open('/home/vncuser/.claude/jobs/9b7c6091/tmp/sps_before.json'))
    after = json.load(open('/home/vncuser/.claude/jobs/9b7c6091/tmp/sps_after.json'))
    rows = [r for r in after['rows'] if 'sps' in r]
    base_b = next(r for r in before['rows'] if r['label'].startswith('baseline'))
    labels = [r['label'].replace(' (all new features off)', '').replace(' (chosen settings)', '\n(chosen)')
              for r in rows]
    sps = [r['sps'] for r in rows]
    dims = [r['obs_dim'] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.3))
    ax = axes[0]
    cols = [GREY] + [TEAL]*3 + [PLUM]
    b = ax.bar(range(len(sps)), sps, color=cols[:len(sps)], edgecolor='white', width=.62)
    ax.axhline(base_b['sps'], color=AMBER, lw=1.8, ls='--')
    ax.text(len(sps)-.4, base_b['sps']*1.005, 'pre-change baseline', color='#8a5a12',
            fontsize=8, ha='right')
    for r_, v, d in zip(b, sps, dims):
        ax.text(r_.get_x()+r_.get_width()/2, v+max(sps)*.012, f'{v/1e6:.2f}M\n{d} dims',
                ha='center', fontsize=7.8)
    ax.set_xticks(range(len(sps))); ax.set_xticklabels(labels, fontsize=7.4)
    ax.set_ylabel('environment steps per second'); ax.set_ylim(0, max(sps)*1.18)
    ax.set_title(f'throughput, {after["num_envs"]} envs, jitted scan of {after["scan"]} steps')
    ax.grid(alpha=.2, axis='y')

    ax = axes[1]
    rel = [100*(s/sps[0]-1) for s in sps]
    b2 = ax.barh(range(len(rel)), rel, color=cols[:len(rel)], edgecolor='white', height=.55)
    for r_, v in zip(b2, rel):
        ax.text(v - .35, r_.get_y()+r_.get_height()/2, f'{v:+.1f}%', va='center',
                ha='right', fontsize=8.5, color='white' if v < -2 else 'black')
    ax.set_yticks(range(len(rel))); ax.set_yticklabels(labels, fontsize=7.4)
    ax.invert_yaxis(); ax.set_xlabel('change in throughput vs the all-off baseline')
    ax.set_title('cost of each setting'); ax.grid(alpha=.2, axis='x')
    fig.suptitle('FIG 4 — Measured throughput cost', fontsize=11, y=1.03)
    fig.text(.5, -.10,
             f'The all-off baseline after the change ({sps[0]/1e6:.2f}M SPS) matches the baseline measured before any code was written '
             f'({base_b["sps"]/1e6:.2f}M) — the implementation costs nothing when disabled.\n'
             'The chosen settings cost 13.5%. Note that is more than the sensor kernels themselves, which micro-benchmarks put near 1-2%: '
             'most of it is the observation growing from 27 to 143 numbers,\nwhich the whole rollout has to carry.',
             ha='center', fontsize=8.5)
    fig.savefig(os.path.join(_OUT, 'fig4_sps.png')); plt.close(fig); print('ok fig4')


for f in (fig1, fig2, fig3, fig4):
    f()
