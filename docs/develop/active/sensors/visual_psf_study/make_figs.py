"""Generate the mechanism-comparison figures."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Wedge, Circle
import numpy as np
import psf_lib as P

plt.rcParams.update({'figure.dpi': 200, 'font.size': 9,
                     'axes.titlesize': 9.5, 'savefig.bbox': 'tight',
                     'savefig.facecolor': 'white'})
CMAP = 'magma'
R_SCALE, ANG_DEG = 0.5, 10.0


def full_grid(half=6):
    return np.array([(r, c) for r in range(-half, half + 1)
                            for c in range(-half, half + 1)], dtype=float)


def draw_cells(ax, cells, vals, title, vmax=None, annot=True, cell=0.94):
    """Draw a set of (row, col) cells as squares coloured by value."""
    vmax = vmax if vmax is not None else max(vals.max(), 1e-9)
    cm = plt.get_cmap(CMAP)
    for (r, c), v in zip(cells, vals):
        col = cm(float(np.clip(v / vmax, 0, 1)))
        ax.add_patch(Rectangle((c - cell / 2, -r - cell / 2), cell, cell,
                               facecolor=col, edgecolor='0.75', lw=0.5))
        if annot:
            ax.text(c, -r, f'{v:.2f}', ha='center', va='center', fontsize=6.5,
                    color='white' if v / vmax < 0.55 else 'black')
    lim = np.abs(cells).max() + 0.8
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal'); ax.axis('off'); ax.set_title(title)


# ------------------------------------------------------------------ FIG 1 ---
def fig1():
    grid = full_grid(6)
    dists = [1.0, 2.0, 3.0, 4.5]
    fig, axes = plt.subplots(2, len(dists), figsize=(4 * len(dists) * 0.78, 6.4))
    for j, d in enumerate(dists):
        e = P.polar_to_rc(30, d)
        for i, (kern, name) in enumerate([('iso', 'isotropic'), ('aniso', 'anisotropic')]):
            w = P.readout(grid, e, kern, 'mass',
                          scale=R_SCALE, radial_scale=R_SCALE, angular_deg=ANG_DEG)
            ax = axes[i, j]
            img = w.reshape(13, 13)
            ax.imshow(img, cmap=CMAP, extent=[-6.5, 6.5, 6.5, -6.5],
                      interpolation='nearest')
            ax.plot(0, 0, marker='X', ms=9, mfc='#4dd0ff', mec='black', mew=0.8)
            ax.plot(e[1], e[0], marker='o', ms=7, mfc='none', mec='#4dff9e', mew=1.6)
            for r in (1, 2, 3):
                ax.plot([0, r, 0, -r, 0], [-r, 0, r, 0, -r], color='white',
                        lw=0.5, alpha=0.35)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f'{name}  ·  object at d={d:g}', fontsize=8.5)
    fig.suptitle('FIG 1 — Shape of the point-spread function on the grid\n'
                 'blue X = agent, green circle = object at 30° (up-and-right); '
                 'white diamonds = sensor range 1/2/3',
                 fontsize=10.5, y=1.015)
    fig.text(0.5, -0.02, 'Top row: an isotropic blur spreads equally in all directions — at d=4.5 the '
             'bright region straddles both the "up" and the "right" cells.\n'
             'Bottom row: the anisotropic kernel elongates ALONG the ray to the object, so it stays narrow '
             'across the ray. Distance gets vague; direction does not.',
             ha='center', fontsize=8.5)
    fig.savefig('fig1_kernel_shape.png'); plt.close(fig)


# ------------------------------------------------------------------ FIG 2 ---
def fig2():
    cells = P.diamond_offsets(2)
    thetas = [20, 45, 70]
    ratios = {}
    fig, axes = plt.subplots(2, len(thetas), figsize=(9.6, 6.6))
    for j, th in enumerate(thetas):
        e = P.polar_to_rc(th, 3.5)
        for i, (kern, name) in enumerate([('iso', 'isotropic'), ('aniso', 'anisotropic')]):
            w = P.readout(cells, e, kern, 'mass',
                          scale=R_SCALE, radial_scale=R_SCALE, angular_deg=ANG_DEG)
            n_idx = np.argmin(np.abs(cells - np.array([-2., 0.])).sum(axis=1))
            e_idx = np.argmin(np.abs(cells - np.array([0., 2.])).sum(axis=1))
            ratio = w[n_idx] / max(w[e_idx], 1e-12)
            ratios[(kern, th)] = ratio
            draw_cells(axes[i, j], cells, w / w.max(),
                       f'{name}  ·  object at {th}°\nN/E signal ratio = {ratio:.2f}')
    fig.suptitle('FIG 2 — What the agent actually reads (range-2 diamond, object at distance 3.5)\n'
                 'values normalised to the brightest cell in each panel',
                 fontsize=10.5, y=1.02)
    fig.text(0.5, -0.015,
             'At 45° both kernels report north = east — that is correct, the object really is on the diagonal.\n'
             'The difference is at 20° and 70°: the isotropic sensor separates them by only '
             f'{ratios[("iso",20)]:.1f}× vs {ratios[("iso",70)]:.2f}×, '
             f'the anisotropic one by {ratios[("aniso",20)]:.0f}× vs {ratios[("aniso",70)]:.2f}×.\n'
             'The anisotropic separation here is arguably too aggressive — see FIG 6 for the knob that sets it.',
             ha='center', fontsize=8.5)
    fig.savefig('fig2_diamond_readout.png'); plt.close(fig)


# ------------------------------------------------------------------ FIG 3 ---
def fig3():
    cells = P.diamond_offsets(2)
    n_idx = np.argmin(np.abs(cells - np.array([-2., 0.])).sum(axis=1))
    e_idx = np.argmin(np.abs(cells - np.array([0., 2.])).sum(axis=1))
    thetas = np.linspace(0, 90, 181)
    dists = [2.0, 3.5, 5.0]
    fig, axes = plt.subplots(2, len(dists), figsize=(11, 6.2))
    for j, d in enumerate(dists):
        curves = {}
        for kern in ('iso', 'aniso'):
            N, E = [], []
            for th in thetas:
                e = P.polar_to_rc(th, d)
                w = P.readout(cells, e, kern, 'mass',
                              scale=R_SCALE, radial_scale=R_SCALE, angular_deg=ANG_DEG)
                N.append(w[n_idx]); E.append(w[e_idx])
            curves[kern] = (np.array(N), np.array(E))
        ax = axes[0, j]
        for kern, col, lab in [('iso', '#e07b39', 'isotropic'), ('aniso', '#3b82c4', 'anisotropic')]:
            N, E = curves[kern]
            ax.plot(thetas, N / N.max(), color=col, label=f'{lab} — north cell')
            ax.plot(thetas, E / E.max(), color=col, ls='--', label=f'{lab} — east cell')
        ax.set_title(f'cell response vs object angle  (d={d:g})')
        ax.set_xlabel('object angle: 0° = north, 90° = east')
        ax.set_ylabel('cell value (per-curve max = 1)')
        ax.axvline(45, color='0.7', lw=0.8, ls=':')
        if j == 0:
            ax.legend(fontsize=6.6, loc='center left')
        ax = axes[1, j]
        for kern, col, lab in [('iso', '#e07b39', 'isotropic'), ('aniso', '#3b82c4', 'anisotropic')]:
            N, E = curves[kern]
            ax.plot(thetas, np.log10(np.maximum(N, 1e-30) / np.maximum(E, 1e-30)),
                    color=col, label=lab)
        ax.axhline(0, color='0.7', lw=0.8); ax.axvline(45, color='0.7', lw=0.8, ls=':')
        ax.set_ylim(-6, 6)
        slopes = {}
        for kern in ('iso', 'aniso'):
            N, E = curves[kern]
            lr = np.log10(np.maximum(N, 1e-30) / np.maximum(E, 1e-30))
            i45 = np.argmin(np.abs(thetas - 45))
            slopes[kern] = abs(np.gradient(lr, thetas)[i45])
        ax.set_title(f'angular discriminability  (d={d:g})\n'
                     f'slope at 45°:  iso {slopes["iso"]:.3f}   aniso {slopes["aniso"]:.3f}'
                     f'   ({slopes["aniso"]/slopes["iso"]:.1f}× sharper)', fontsize=8.5)
        ax.set_xlabel('object angle'); ax.set_ylabel('log₁₀( north / east )')
        if j == 0:
            ax.legend(fontsize=7)
    fig.suptitle('FIG 3 — How well can the agent tell "up" from "right"?', fontsize=11, y=1.015)
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    fig.text(0.5, 0.005,
             'A steep curve through 45° means the sensor resolves direction; a flat one means it cannot. '
             'Both kernels degrade with distance —\nthe honest claim is not that the anisotropic one is immune, '
             'but that it stays several times sharper at every range, and it buys that\n'
             'sharpness without reducing the radial blur at all. The two axes are genuinely independent.',
             ha='center', fontsize=8.5)
    fig.savefig('fig3_angular_discrimination.png'); plt.close(fig)


# ------------------------------------------------------------------ FIG 4 ---
def fig4():
    cells = P.diamond_offsets(2)
    ds = np.linspace(0.6, 6.0, 60)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.9))
    styles = {'mass': ('#2a9d5c', 'normalise by full kernel mass'),
              'diamond': ('#c94f4f', 'normalise over visible cells'),
              'none': ('#7b61b8', 'no normalisation')}
    for mode, (col, lab) in styles.items():
        tot, peak = [], []
        for d in ds:
            e = P.polar_to_rc(30, d)
            w = P.readout(cells, e, 'aniso', mode,
                          radial_scale=R_SCALE, angular_deg=ANG_DEG)
            tot.append(w.sum()); peak.append(w.max())
        axes[0].plot(ds, tot, color=col, label=lab)
        axes[1].plot(ds, peak, color=col, label=lab)
    for ax, t in zip(axes, ['total signal the object contributes to the diamond',
                            'brightest single cell']):
        ax.set_xlabel('object distance from agent (cells)'); ax.set_title(t)
        ax.set_yscale('log'); ax.grid(alpha=0.25)
    axes[0].set_ylabel('summed value'); axes[1].set_ylabel('peak cell value')
    axes[0].legend(fontsize=7.5)
    fig.suptitle('FIG 4 — Does a distant object get fainter? (anisotropic kernel, range-2 diamond)',
                 fontsize=10.5, y=1.06)
    fig.text(0.5, -0.09,
             'Green: normalising by the kernel\'s full mass means only the fraction landing inside the diamond is '
             'reported — signal falls ~30× from d=1 to d=5. Blur width alone produces the distance falloff.\n'
             'Red: normalising over visible cells puts that mass straight back — a distant predator stays exactly '
             'as loud as an adjacent one. Purple: unnormalised — a distant object is BRIGHTER in total than a near one.',
             ha='center', fontsize=8.5)
    fig.tight_layout(); fig.savefig('fig4_normalisation.png'); plt.close(fig)


# ------------------------------------------------------------------ FIG 5 ---
# entities sit on integer grid cells, as they do in the real environment
SCENE = [('predator', np.array([-3.,  2.]), '#d94a4a'),   # up 3, right 2  (d=3.6)
         ('rabbit',   np.array([ 0., -2.]), '#4a9ad9'),   # due west       (d=2.0)
         ('food',     np.array([ 1.,  0.]), '#4ad97a')]   # due south      (d=1.0)


def _scene_panel(ax):
    for name, e, col in SCENE:
        ax.plot(e[1], -e[0], 'o', ms=9, mfc=col, mec='black', mew=0.7)
        ax.annotate(name, (e[1], -e[0]), textcoords='offset points',
                    xytext=(0, 10), ha='center', fontsize=7.5)
    ax.plot(0, 0, marker='X', ms=11, mfc='#222', mec='white', mew=1.0)
    ax.annotate('agent', (0, 0), textcoords='offset points', xytext=(0, -14),
                ha='center', fontsize=7.5)
    for r in (1, 2):
        ax.plot([0, r, 0, -r, 0], [-r, 0, r, 0, -r], color='0.6', lw=0.7, ls='--')
    ax.set_xlim(-4.2, 4.2); ax.set_ylim(-4.2, 4.2)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('the scene\n(dashed = sensor range 1 and 2)')


def fig5():
    cells1, cells2 = P.diamond_offsets(1), P.diamond_offsets(2)
    fig = plt.figure(figsize=(13.5, 6.4))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.15)
    _scene_panel(fig.add_subplot(gs[0, 0]))

    def stack(cells, kern, norm):
        tot = np.zeros(len(cells))
        for _, e, _c in SCENE:
            tot += P.readout(cells, e, kern, norm, scale=R_SCALE,
                             radial_scale=R_SCALE, angular_deg=ANG_DEG)
        return tot

    draw_cells(fig.add_subplot(gs[0, 1]), cells1, stack(cells1, 'exact', 'none'),
               'A · today: exact match, range 1\nsees the adjacent food, blind to the rest')
    draw_cells(fig.add_subplot(gs[0, 2]), cells2, stack(cells2, 'iso', 'mass'),
               'B · isotropic blur, range 2\neverything smeared together')
    draw_cells(fig.add_subplot(gs[1, 0]), cells2, stack(cells2, 'aniso', 'mass'),
               'C · anisotropic blur, range 2\nthree distinct directional lobes')

    ax = fig.add_subplot(gs[1, 1])
    sec = P.sector_readout([e for _, e, _c in SCENE], n_sectors=8, near_radius=1.5)
    cm = plt.get_cmap(CMAP); mx = max(sec.max(), 1e-9)
    for i, v in enumerate(sec):
        centre = 90 - i * 45          # compass bearing -> matplotlib angle
        ax.add_patch(Wedge((0, 0), 1.0, centre - 22.5, centre + 22.5,
                           facecolor=cm(v / mx), edgecolor='0.75', lw=0.6))
        ang = np.radians(centre)
        ax.text(0.65 * np.cos(ang), 0.65 * np.sin(ang), f'{v:.2f}',
                ha='center', va='center', fontsize=6.5,
                color='white' if v / mx < 0.55 else 'black')
    ax.add_patch(Circle((0, 0), 0.28, facecolor='white', edgecolor='0.6'))
    ax.text(0, 0, 'near\n(sharp)', ha='center', va='center', fontsize=6)
    for i, lab in enumerate(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']):
        a = np.radians(90 - i * 45)
        ax.text(1.06 * np.cos(a), 1.06 * np.sin(a), lab, ha='center',
                va='center', fontsize=6.5, color='0.35')
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title('D · sharp near field + 8 far sectors\ndirection only, no distance')

    ax = fig.add_subplot(gs[1, 2])
    lp = P.logpolar_readout([e for _, e, _c in SCENE], n_ang=8, ring_edges=(1.0, 3.0, 9.0))
    ax.imshow(lp, cmap=CMAP, aspect='auto', interpolation='nearest')
    ax.set_xticks(range(8)); ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'], fontsize=7)
    ax.set_yticks(range(3)); ax.set_yticklabels(['d<1', '1–3', '3–9'], fontsize=7)
    for i in range(3):
        for j in range(8):
            if lp[i, j] > 0:
                ax.text(j, i, f'{lp[i,j]:.0f}', ha='center', va='center',
                        fontsize=7, color='black')
    ax.set_title('E · log-polar map\nangle exact, distance in coarse rings')
    fig.suptitle('FIG 5 — The same scene under each candidate mechanism', fontsize=11.5, y=0.99)
    fig.savefig('fig5_architectures.png'); plt.close(fig)


for f in (fig1, fig2, fig3, fig4, fig5):
    f(); print('ok', f.__name__)
