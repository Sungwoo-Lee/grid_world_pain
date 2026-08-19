"""FIG 6 — sweep of the anisotropy knob rho = sigma_radial / sigma_tangential.

Ratios are computed in LOG space: at high rho the east-cell weight underflows
float64, so exponentiating first gives a spurious collapse to zero.
"""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, psf_lib as P
from make_figs import draw_cells, R_SCALE

N_OFF, E_OFF = np.array([-2., 0.]), np.array([0., 2.])


def log10_ratio(theta, d, rho, radial_scale=R_SCALE):
    """log10(north / east) computed analytically — no underflow."""
    e = P.polar_to_rc(theta, d)
    u = e / np.linalg.norm(e)
    t = np.array([-u[1], u[0]])
    sp = radial_scale * d
    st = sp / rho
    def logw(cell):
        v = cell - e
        return -(v @ u) ** 2 / (2 * sp ** 2) - (v @ t) ** 2 / (2 * st ** 2)
    return (logw(N_OFF) - logw(E_OFF)) / np.log(10)


def sharpness(d, rho, h=0.5):
    """|d log10(N/E) / d theta| at 45 deg, by central difference."""
    return abs(log10_ratio(45 + h, d, rho) - log10_ratio(45 - h, d, rho)) / (2 * h)


RHOS = [1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
cells = P.diamond_offsets(2)
fig = plt.figure(figsize=(13.6, 7.8))
gs = fig.add_gridspec(2, len(RHOS), height_ratios=[1.05, 1.0], hspace=0.28, wspace=0.06)

for j, rho in enumerate(RHOS):
    w = P.readout(cells, P.polar_to_rc(20, 3.5), 'aniso', 'mass',
                  radial_scale=R_SCALE, ratio=rho)
    lr = log10_ratio(20, 3.5, rho)
    st_cells = R_SCALE * 3.5 / rho
    lab = 'ρ = 1.0  (isotropic)' if rho == 1.0 else f'ρ = {rho:g}'
    flag = '' if st_cells >= 0.5 else '  (sub-cell)'
    draw_cells(fig.add_subplot(gs[0, j]), cells, w / w.max(),
               f'{lab}\nσ⊥ = {st_cells:.2f} cells{flag}\nlog₁₀(N/E) = {lr:.1f}', annot=False)

rho_fine = np.linspace(1.0, 10.0, 60)
ax = fig.add_subplot(gs[1, :3])
for d, col in [(2.0, '#2a9d5c'), (3.5, '#3b82c4'), (5.0, '#b05ac2')]:
    ax.plot(rho_fine, [sharpness(d, r) for r in rho_fine], color=col,
            label=f'object at d={d:g}')
    r_half = R_SCALE * d / 0.5                      # rho at which sigma_perp = 0.5 cells
    if r_half <= 10:
        ax.axvline(r_half, color=col, lw=0.8, ls=':', alpha=0.7)
ax.set_xlabel('anisotropy ρ  =  σ_radial / σ_tangential      (ρ = 1 is the isotropic kernel)')
ax.set_ylabel('angular sharpness\n(|slope of log₁₀(N/E)| at 45°)')
ax.set_title('direction resolved, vs the anisotropy knob\n'
             'dotted line per curve: where σ⊥ falls below half a cell', fontsize=9)
ax.grid(alpha=0.25); ax.legend(fontsize=8)

ax = fig.add_subplot(gs[1, 3:])
ds = np.linspace(0.8, 6.0, 50)
for rho, col in [(1.0, '#e07b39'), (3.0, '#3b82c4'), (10.0, '#b05ac2')]:
    ax.plot(ds, [P.readout(cells, P.polar_to_rc(20, d), 'aniso', 'mass',
                           radial_scale=R_SCALE, ratio=rho).sum() for d in ds],
            color=col, label=f'ρ = {rho:g}')
ax.set_yscale('log'); ax.grid(alpha=0.25); ax.legend(fontsize=8)
ax.set_xlabel('object distance from agent (cells)')
ax.set_ylabel('total signal in the diamond')
ax.set_title('anisotropy also changes the distance falloff:\nhigher ρ = dimmer up close, brighter far away',
             fontsize=9)

fig.suptitle('FIG 6 — Choosing the anisotropy knob   (range-2 diamond, σ_radial = 0.5·d held fixed throughout)',
             fontsize=11.5, y=0.98)
fig.text(0.5, 0.005,
         'ρ = 1 is exactly the isotropic kernel, so this single number gives you a free ablation. Angular sharpness keeps '
         'rising with ρ and does NOT saturate —\nbut once σ⊥ drops below about half a cell (dotted lines) the blob no longer '
         'reaches the cells either side of the ray, so further increases buy\nastronomical value ratios rather than more usable '
         'information. That puts the useful band at roughly ρ = 2–4 for objects at d = 2–5 on this grid.',
         ha='center', fontsize=8.5)
fig.savefig('fig6_anisotropy_sweep.png'); print('ok fig6')
