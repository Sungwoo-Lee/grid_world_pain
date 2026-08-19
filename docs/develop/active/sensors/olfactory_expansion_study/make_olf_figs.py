"""Figures for the per-cell olfactory expansion study."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import olf_lib as O

plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.titlesize': 9.5,
                     'savefig.bbox': 'tight', 'savefig.facecolor': 'white'})
CMAP = 'viridis'
TEAL, AMBER, PLUM = '#1f7a6c', '#c9852b', '#7b4b91'
FOOD = np.array([[1., 0, 0, 0, 0]])          # default.yaml food signature


def draw_diamond(ax, cells, vals, title, annot=True, vmax=None, cell=0.94, fmt='{:.3f}'):
    vmax = vmax if vmax is not None else max(float(np.max(vals)), 1e-12)
    cm = plt.get_cmap(CMAP)
    for (r, c), v in zip(cells, vals):
        col = cm(float(np.clip(v / vmax, 0, 1)))
        ax.add_patch(Rectangle((c - cell/2, -r - cell/2), cell, cell,
                               facecolor=col, edgecolor='0.75', lw=.5))
        if annot:
            ax.text(c, -r, fmt.format(v), ha='center', va='center', fontsize=6.8,
                    color='white' if v/vmax < .55 else 'black')
    lim = np.abs(cells).max() + .8
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title)


def field_grid(half=5):
    return np.array([(r, c) for r in range(-half, half+1) for c in range(-half, half+1)], float)


# ------------------------------------------------------------------- FIG 0 ---
def fig0():
    src = O.polar_to_rc(20, 4.0)[None, :]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))

    # today
    now = O.sense_at([[0., 0.]], src, FOOD)[0]
    ax = axes[0]
    ax.plot(src[0][1], -src[0][0], 'o', ms=9, mfc='#4ad97a', mec='black', mew=.7)
    ax.text(src[0][1], -src[0][0]+.62, 'food source', ha='center', fontsize=8)
    for ang in (0, 90, 180, 270):
        a = np.radians(ang)
        ax.annotate('', xy=(1.55*np.sin(a), 1.55*np.cos(a)), xytext=(.55*np.sin(a), .55*np.cos(a)),
                    arrowprops=dict(arrowstyle='->', color='0.75', lw=1.4, ls=(0, (2, 2))))
        ax.text(2.0*np.sin(a), 2.0*np.cos(a), '?', ha='center', va='center',
                fontsize=13, color='0.6')
    ax.add_patch(Rectangle((-.47, -.47), .94, .94,
                           facecolor=plt.get_cmap(CMAP)(.35), edgecolor='0.6'))
    ax.text(0, 0, f'{now[0]:.3f}', ha='center', va='center', color='white', fontsize=10.5)
    ax.set_xlim(-2.6, 2.9); ax.set_ylim(-2.6, 4.6)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'1 · today — one number\n"{now[0]:.2f} of smell-0 here", and nothing about where')

    # the field
    g = field_grid(5)
    f = O.sense_at(g, src, FOOD)[:, 0].reshape(11, 11)
    ax = axes[1]
    ax.imshow(f, cmap=CMAP, extent=[-5.5, 5.5, 5.5, -5.5], interpolation='nearest')
    for r in (1, 2):
        ax.plot([0, r, 0, -r, 0], [-r, 0, r, 0, -r], color='white', lw=.9, alpha=.75)
    ax.plot(0, 0, marker='X', ms=10, mfc='#111', mec='white', mew=.9)
    ax.plot(src[0][1], src[0][0], 'o', ms=8, mfc='#4ad97a', mec='black', mew=.7)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('2 · the field it is sampling\n(sampled at one point, direction is thrown away)')

    # per-cell readout + recovered bearing
    vals, cells = O.readout([0., 0.], src, FOOD, 1)
    est = O.gradient_bearing(vals[:, 0])
    ax = axes[2]
    draw_diamond(ax, cells, vals[:, 0], '')
    th_t, th_e = np.radians(20.0), np.radians(est)
    ax.annotate('', xy=(1.55*np.sin(th_t), 1.55*np.cos(th_t)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#4ad97a', lw=2.2))
    ax.annotate('', xy=(1.15*np.sin(th_e), 1.15*np.cos(th_e)), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color=AMBER, lw=2.2, ls=(0, (3, 2))))
    ax.set_title(f'3 · proposed — one number per cell\ntrue bearing 20°, recovered {est:.1f}° '
                 f'(error {O.bearing_error(20, est):.1f}°)')
    ax.set_xlim(-2.1, 2.4); ax.set_ylim(-2.1, 2.6)

    fig.suptitle('FIG 0 — What the expansion actually changes: sample the same field at five points instead of one',
                 fontsize=11, y=1.02)
    fig.text(.5, -.06,
             'Nothing about the odour field changes. The sensor simply evaluates it at every cell of the diamond rather than only under the agent, and the '
             'differences between\nthose readings encode the source direction. A central difference on the four neighbours recovers the bearing to within a couple of degrees — with no noise.',
             ha='center', fontsize=8.5)
    fig.savefig('fig0_what_changes.png'); plt.close(fig); print('ok fig0')


# ------------------------------------------------------------------- FIG 1 ---
def fig1():
    gammas = [0.5, 1.0, 2.0, 3.0]
    src = O.polar_to_rc(20, 4.0)[None, :]
    g = field_grid(5)
    fig, axes = plt.subplots(2, len(gammas), figsize=(12.8, 6.4))
    for j, gm in enumerate(gammas):
        f = O.sense_at(g, src, FOOD, gamma=gm)[:, 0].reshape(11, 11)
        ax = axes[0, j]
        ax.imshow(np.log10(f + 1e-6), cmap=CMAP, extent=[-5.5, 5.5, 5.5, -5.5],
                  interpolation='nearest')
        ax.plot(0, 0, marker='X', ms=8, mfc='#111', mec='white', mew=.8)
        ax.plot(src[0][1], src[0][0], 'o', ms=7, mfc='#4ad97a', mec='black', mew=.6)
        ax.set_xticks([]); ax.set_yticks([])
        lab = 'γ = 1.0  ← current config' if gm == 1.0 else f'γ = {gm:g}'
        ax.set_title(f'{lab}\nfield (log colour)', fontsize=8.8)
        vals, cells = O.readout([0., 0.], src, FOOD, 1, gamma=gm)
        v = vals[:, 0]
        rel = (v.max() - v.min()) / v.mean() * 100
        draw_diamond(axes[1, j], cells, v, f'spread = {rel:.0f}% of mean', fmt='{:.3f}')
    fig.suptitle('FIG 1 — The decay power γ trades reach against contrast   (source at 20°, distance 4)',
                 fontsize=11, y=1.0)
    fig.text(.5, -.04,
             'Small γ carries a long way but flattens: at γ=0.5 the five cells barely differ, so there is little direction to read. Large γ gives strong cell-to-cell '
             'contrast\nbut the signal collapses to nothing a few cells out. γ=1.0 is what the config ships. The next figure puts this trade-off on an absolute scale.',
             ha='center', fontsize=8.5)
    fig.savefig('fig1_decay_power.png'); plt.close(fig); print('ok fig1')


# ------------------------------------------------------------------- FIG 2 ---
def fig2():
    ds = np.linspace(1.5, 9, 200)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))
    ax = axes[0]
    ax.plot(ds, 1/ds, color=TEAL, lw=2, label='level at the agent  (1/d)')
    ax.plot(ds, O.adjacent_contrast(ds), color=PLUM, lw=2,
            label='directional signal  (difference between adjacent cells)')
    ax.axhline(O.SIGMA_CFG, color=AMBER, lw=1.8, ls='--',
               label=f'configured noise σ = {O.SIGMA_CFG}')
    xc = ds[np.argmin(np.abs(O.adjacent_contrast(ds) - O.SIGMA_CFG))]
    ax.axvspan(xc, 9, color=AMBER, alpha=.10)
    ax.annotate(f'direction below the\nnoise floor beyond d ≈ {xc:.1f}',
                xy=(xc+.15, .3), fontsize=8, color='#8a5a12')
    ax.set_yscale('log'); ax.set_xlabel('source distance from agent (cells)')
    ax.set_ylabel('signal magnitude'); ax.set_title('γ = 1.0, the shipped configuration')
    ax.legend(fontsize=7.6); ax.grid(alpha=.25)

    ax = axes[1]
    for gm, col in [(0.5, '#5ab5a6'), (1.0, TEAL), (2.0, PLUM), (3.0, '#b0457a')]:
        ax.plot(ds, O.adjacent_contrast(ds, gm) / O.SIGMA_CFG, color=col, label=f'γ = {gm:g}')
    ax.axhline(1.0, color=AMBER, lw=1.8, ls='--')
    ax.text(8.6, 1.15, 'SNR = 1', color='#8a5a12', fontsize=8, ha='right')
    ax.set_yscale('log'); ax.set_ylim(1e-3, 1e2); ax.grid(alpha=.25)
    ax.set_xlabel('source distance from agent (cells)')
    ax.set_ylabel('directional signal ÷ noise σ')
    ax.set_title('how far direction survives, per decay power')
    ax.legend(fontsize=7.6)
    fig.suptitle('FIG 2 — The signal budget: is there enough contrast between cells to read a direction from?',
                 fontsize=11, y=1.04)
    fig.text(.5, -.1,
             'The purple curve is the quantity that matters — not how much smell there is, but how much the cells DIFFER. It falls as roughly γ/d^(γ+1), far faster than the level '
             'itself.\nAgainst the olfaction σ = 0.2 the project already configures, a single observation carries usable direction only within about three cells. Raising γ buys contrast '
             'near the agent\nand loses reach; no single γ makes direction readable across a 10×10 grid in one glance. (Noise is off by default — this binds whenever it is switched on.)',
             ha='center', fontsize=8.5)
    fig.savefig('fig2_signal_budget.png'); plt.close(fig); print('ok fig2')


# ------------------------------------------------------------------- FIG 3 ---
def fig3():
    """Direction recovery. The estimator is a planar least-squares fit over ALL
    diamond cells — a readable proxy for the information present in the readout,
    not a claim about what the network does."""
    rng = np.random.default_rng(0)
    ds = np.arange(2, 8)
    trials = 3000
    RANGES = [(1, TEAL), (2, PLUM), (3, AMBER)]
    noisy, clean = {}, {}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.3))

    for rr, col in RANGES:
        errs = []
        for d in ds:
            th = rng.uniform(0, 360, trials)
            acc = []
            for k in range(trials):
                e = O.polar_to_rc(th[k], d)[None, :]
                v, _ = O.readout([0., 0.], e, FOOD, rr)
                vn = np.clip(v[:, 0] + rng.normal(0, O.SIGMA_CFG, v.shape[0]), 0, None)
                acc.append(O.bearing_error(th[k], O.gradient_bearing_lsq(vn, rr)))
            errs.append(np.median(acc))
        n = 2*rr*rr + 2*rr + 1
        noisy[rr] = dict(zip(ds, errs))
        axes[0].plot(ds, errs, color=col, marker='o', ms=3.5,
                     label=f'range {rr}  ({n} cells, {n*O.VECSIZE_CFG} dims)')

    axes[0].axhline(90, color='0.6', lw=1, ls=':')
    axes[0].text(7.05, 93, 'chance', fontsize=7.5, color='0.4', ha='right')
    axes[0].set_ylim(0, 110); axes[0].grid(alpha=.25)
    axes[0].set_xlabel('source distance from agent (cells)')
    axes[0].set_ylabel('median bearing error (degrees)')
    axes[0].set_title(f'with the configured noise σ = {O.SIGMA_CFG}\nmore cells = more samples to average')
    axes[0].legend(fontsize=7.6)

    for rr, col in RANGES:
        bias = []
        for d in ds:
            acc = []
            for th in np.linspace(0, 360, 145)[:-1]:
                e = O.polar_to_rc(th, d)[None, :]
                v, _ = O.readout([0., 0.], e, FOOD, rr)
                acc.append(O.bearing_error(th, O.gradient_bearing_lsq(v[:, 0], rr)))
            bias.append(np.median(acc))
        clean[rr] = dict(zip(ds, bias))
        axes[1].plot(ds, bias, color=col, marker='o', ms=3.5, label=f'range {rr}')
    axes[1].grid(alpha=.25); axes[1].legend(fontsize=7.6)
    axes[1].set_xlabel('source distance from agent (cells)')
    axes[1].set_ylabel('median bearing error (degrees)')
    axes[1].set_title('noiseless — the cost of a bigger diamond\nthe field is curved, a plane fitted over more cells fits it worse')

    fig.suptitle('FIG 3 — Can the agent recover the direction? (median over 3000 random bearings)',
                 fontsize=11, y=1.04)
    fig.text(.5, -.13,
             f'Left: under noise a bigger diamond helps a great deal — at four cells the median error falls from {noisy[1][4]:.0f}° (range 1) to {noisy[2][4]:.0f}° (range 2) to '
             f'{noisy[3][4]:.0f}° (range 3).\nRange 1 offers only four usable neighbours and no redundancy, so every reading is believed exactly as noisy as it is. Right: with no noise the '
             f'ordering flips beyond about three cells\n(range 1 reaching {clean[1][6]:.1f}° at d=6 against {clean[3][6]:.1f}° for range 3), because a plane fitted across a wider patch of a '
             f'curved 1/d field carries more bias — though at d=2 the larger\ndiamond is the more accurate one. Textbook bias–variance: the extra cells earn their keep exactly to the extent '
             'that noise is switched on.',
             ha='center', fontsize=8.5)
    fig.savefig('fig3_direction_recovery.png'); plt.close(fig); print('ok fig3')


# ------------------------------------------------------------------- FIG 4 ---
def fig4():
    g = field_grid(5)
    SCENES = [("two sources, ±60°", [(-60, 3.5), (60, 3.5)]),
              ("north and east", [(0, 4.0), (90, 4.0)]),
              ("near SE, far NW", [(135, 2.0), (-45, 5.0)])]
    fig, axes = plt.subplots(2, len(SCENES), figsize=(11.4, 7.2),
                             gridspec_kw=dict(height_ratios=[1.0, 1.0], hspace=.30))
    for j, (name, spec) in enumerate(SCENES):
        srcs = np.stack([O.polar_to_rc(b, d) for b, d in spec])
        props = np.repeat(FOOD, len(spec), axis=0)
        f = O.sense_at(g, srcs, props)[:, 0].reshape(11, 11)
        ax = axes[0, j]
        ax.imshow(f, cmap=CMAP, extent=[-5.5, 5.5, 5.5, -5.5], interpolation='nearest')
        for sc in srcs:
            ax.plot(sc[1], sc[0], 'o', ms=8, mfc='#4ad97a', mec='black', mew=.6)
        ax.plot(0, 0, marker='X', ms=9, mfc='#111', mec='white', mew=.9)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(name, fontsize=9)

        vals, cells = O.readout([0., 0.], srcs, props, 1)
        est = O.gradient_bearing_lsq(vals[:, 0], 1)
        ax = axes[1, j]
        draw_diamond(ax, cells, vals[:, 0], '')
        for b, _d in spec:
            a = np.radians(b)
            ax.annotate('', xy=(2.0*np.sin(a), 2.0*np.cos(a)), xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='#4ad97a', lw=1.5, alpha=.8))
        a = np.radians(est)
        ax.annotate('', xy=(1.5*np.sin(a), 1.5*np.cos(a)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle='->', color=AMBER, lw=2.4))
        truth = ', '.join(f'{b:+.0f}°' for b, _d in spec)
        ax.set_title(f'sources at {truth}\nreadout says {est:+.0f}°', fontsize=8.8)
        ax.set_xlim(-2.4, 2.4); ax.set_ylim(-2.4, 2.7)
    fig.suptitle('FIG 4 — Sources on the same channel sum into a single phantom direction',
                 fontsize=11, y=0.99)
    fig.text(.5, -.03,
             'Green arrows are the true source bearings; amber is what the readout reports. Contributions add before the agent ever sees them, so the sensor returns the gradient of '
             'the SUM — one\ndirection, pointing at the intensity-weighted centroid. In the first two scenes that bearing has no food on it at all. The third shows the weighting: a source '
             'at two cells dominates one at\nfive, so the phantom sits close to the near source rather than between them. This is inherent to a summed chemical field, not a flaw in the '
             'expansion — but per-cell sampling is what turns\nit into a direction the agent can act on, and act on wrongly.',
             ha='center', fontsize=8.5)
    fig.savefig('fig4_superposition.png'); plt.close(fig); print('ok fig4')


# ------------------------------------------------------------------- FIG 5 ---
def fig5():
    gammas = np.linspace(0.3, 3.5, 90)
    ds = np.linspace(1.5, 9, 90)
    G, Dm = np.meshgrid(gammas, ds)
    level = 1.0 / Dm**G
    contrast = 1.0/(Dm-1)**G - 1.0/Dm**G
    detect = level > O.SIGMA_CFG
    direct = contrast > O.SIGMA_CFG
    code = np.zeros_like(level)
    code[detect] = 1.0
    code[detect & direct] = 2.0
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    im = ax.contourf(G, Dm, code, levels=[-.5, .5, 1.5, 2.5],
                     colors=['#efe9f2', '#9fd0c6', TEAL])
    ax.axvline(1.0, color=AMBER, lw=1.8, ls='--')
    ax.text(1.06, 7.6, 'γ = 1.0\nshipped', color='#8a5a12', fontsize=8)
    reach = Dm[code == 2.0].max()
    reach_at_1 = Dm[(code == 2.0) & (np.abs(G - 1.0) < .04)].max()
    ax.set_xlabel('decay power γ'); ax.set_ylabel('source distance (cells)')
    ax.set_title('Operating envelope at the configured noise σ = 0.2')
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=TEAL, label='smell detected AND direction readable'),
                       Patch(color='#9fd0c6', label='smell detected, direction lost in noise'),
                       Patch(color='#efe9f2', label='below the noise floor entirely')],
              fontsize=7.6, loc='upper right')
    fig.text(.5, -.16,
             f'The dark band is where per-cell olfaction does the job it is being added to do. Across every γ tested it never reaches past {reach:.1f} cells, and at the shipped γ=1.0 it stops '
             f'at {reach_at_1:.1f}.\nRaising γ sharpens contrast close in while shrinking detection range at the same time, so the band shifts rather than grows. On a 10×10 grid this is a '
             'close-range sense.\nThat is an argument for pairing it with vision and for keeping the diamond small, not for tuning γ harder. (Both boundaries move if σ changes; noise is off '
             'by default.)',
             ha='center', fontsize=8.5)
    fig.savefig('fig5_operating_envelope.png'); plt.close(fig); print('ok fig5')


for f in (fig0, fig1, fig2, fig3, fig4, fig5):
    f()
