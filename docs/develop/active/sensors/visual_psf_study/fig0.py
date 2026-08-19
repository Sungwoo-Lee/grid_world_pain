"""FIG 0 — anatomy of the kernel: from a plain Gaussian to the polar decomposition."""
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle
import numpy as np

plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'savefig.bbox': 'tight',
                     'savefig.facecolor': 'white'})
ACC, EMB, GREY = '#9c2f6d', '#c85c31', '#6e6579'

TH, D, RS, ANG = 20.0, 3.5, 0.5, 10.0
th = np.radians(TH)
e_rc = np.array([-D*np.cos(th), D*np.sin(th)])            # (row, col)
u_rc = e_rc / D
t_rc = np.array([-u_rc[1], u_rc[0]])
sp, st = RS*D, D*np.sin(np.radians(ANG))
rc2xy = lambda p: np.array([p[1], -p[0]])                 # (row,col) -> plot (x,y)
E, U, T = rc2xy(e_rc), rc2xy(u_rc), rc2xy(t_rc)
ang_deg = np.degrees(np.arctan2(U[1], U[0]))

fig, axes = plt.subplots(1, 3, figsize=(13.6, 5.4))

# ---- panel 1: the isotropic Gaussian ----------------------------------------
ax = axes[0]
for k, a in [(1, .55), (2, .3)]:
    ax.add_patch(Circle((0, 0), k*1.0, fill=False, ec=GREY, lw=1.3, alpha=a))
ax.annotate('', xy=(1.0, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='<->', color=ACC, lw=1.5))
ax.text(.5, .16, 'σ', color=ACC, ha='center', fontsize=12, style='italic')
ax.plot(0, 0, 'o', ms=6, color=ACC)
ax.text(0, -3.25, r'$w(v)=\exp\!\left(-\dfrac{\|v\|^2}{2\sigma^2}\right)$',
        ha='center', fontsize=11)
ax.text(0, 2.65, 'level sets are CIRCLES\nno direction is special',
        ha='center', fontsize=8.5, color=GREY)
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.9, 3.6)
ax.set_title('1 · isotropic: one width', fontsize=10)

# ---- panel 2: stretch + rotate ----------------------------------------------
ax = axes[1]
for k, a in [(1, .6), (1.45, .3)]:
    ax.add_patch(Ellipse((0, 0), 2*k*sp, 2*k*st, angle=ang_deg,
                         fill=False, ec=GREY, lw=1.3, alpha=a))
ax.annotate('', xy=tuple(U*sp), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=ACC, lw=2))
ax.annotate('', xy=tuple(T*st), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color=EMB, lw=2))
ax.text(*(U*sp*0.62 + np.array([.28, -.1])), 'σ∥ û', color=ACC, fontsize=11)
ax.text(*(T*st*1.5 + np.array([-.15, .3])), 'σ⊥ t̂', color=EMB, fontsize=11)
ax.plot(0, 0, 'o', ms=6, color='0.2')
ax.text(0, -3.25,
        r'$w(v)=\exp\!\left(-\dfrac{(v\cdot\hat{u})^2}{2\sigma_\parallel^2}'
        r'-\dfrac{(v\cdot\hat{t})^2}{2\sigma_\perp^2}\right)$',
        ha='center', fontsize=10.5)
ax.text(0, 2.65, 'level sets are ELLIPSES\nlong along û, narrow across it',
        ha='center', fontsize=8.5, color=GREY)
ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.9, 3.6)
ax.set_title(f'2 · anisotropic: two widths  (ρ = σ∥/σ⊥ = {sp/st:.1f})', fontsize=10)

# ---- panel 3: the real geometry, with the worked example --------------------
ax = axes[2]
for r in (1, 2):
    ax.plot([0, r, 0, -r, 0], [-r, 0, r, 0, -r], color='0.75', lw=.8, ls='--')
for k, a in [(1, .5)]:
    ax.add_patch(Ellipse(tuple(E), 2*k*sp, 2*k*st, angle=ang_deg,
                         fill=False, ec=GREY, lw=1.2, alpha=a))
ax.plot([0, E[0]], [0, E[1]], color=ACC, lw=1.1, ls=':')
ax.plot(0, 0, marker='X', ms=10, mfc='#222', mec='white', mew=.9)
ax.text(.16, -.42, 'agent', fontsize=8)
ax.plot(*E, 'o', ms=8, mfc='#4ad97a', mec='black', mew=.7)
ax.text(E[0]+.28, E[1]-.30, 'object\nd = 3.5, θ = 20°', fontsize=8)
ax.annotate('', xy=tuple(E + U*0.95), xytext=tuple(E),
            arrowprops=dict(arrowstyle='->', color=ACC, lw=1.8))
ax.text(*(E + U*1.15 + np.array([.12, -.05])), 'û', color=ACC, fontsize=12)
ax.annotate('', xy=tuple(E + T*0.95), xytext=tuple(E),
            arrowprops=dict(arrowstyle='->', color=EMB, lw=1.8))
ax.text(*(E + T*1.2 + np.array([-.12, .12])), 't̂', color=EMB, fontsize=12)

N = rc2xy(np.array([-2., 0.]))
v = N - E
vpar, vperp = float(np.dot(v, U)), float(np.dot(v, T))
ax.plot(*N, 's', ms=9, mfc='none', mec='#3b82c4', mew=1.8)
ax.text(N[0]-.05, N[1]-.5, 'cell N', fontsize=8, color='#3b82c4', ha='center')
ax.annotate('', xy=tuple(N), xytext=tuple(E),
            arrowprops=dict(arrowstyle='->', color='#3b82c4', lw=1.6))
ax.text(*((E+N)/2 + np.array([.16, .1])), 'v', color='#3b82c4', fontsize=11)
foot = E + U*vpar
ax.plot([E[0]+U[0]*vpar, N[0]], [E[1]+U[1]*vpar, N[1]], color=EMB, lw=1.1, ls='--')
ax.plot([E[0], foot[0]], [E[1], foot[1]], color=ACC, lw=1.6, ls='--')
ax.text(0.6, -2.45,
        f'v·û  = {vpar:+.2f}   judged against σ∥ = {sp:.2f}\n'
        f'v·t̂  = {vperp:+.2f}   judged against σ⊥ = {st:.2f}',
        ha='center', fontsize=9.5)
ax.set_xlim(-3.0, 4.4); ax.set_ylim(-2.9, 5.4)
ax.set_title('3 · the same thing on the grid', fontsize=10)

for ax in axes:
    ax.set_aspect('equal'); ax.axis('off')
fig.suptitle('FIG 0 — Anatomy of the kernel: an ordinary Gaussian, stretched along one axis and pointed at the object',
             fontsize=11, y=1.0)
fig.text(0.5, -0.10,
         'Panel 3 is the worked example in the text. The displacement v from object to cell N is split into a component ALONG the ray (compared against the wide σ∥)\n'
         'and a component ACROSS it (compared against the narrow σ⊥). Because the across-component is judged by a much smaller yardstick, moving a cell sideways\n'
         'off the ray costs far more weight than moving it further along the ray — which is exactly "distance is vague, bearing is sharp".',
         ha='center', fontsize=8.5)
fig.savefig('fig0_kernel_anatomy.png'); print('ok fig0')
