"""FIG 8 — Anatomy of noise: what it is, what SNR means, how it differs from
blur, and the three conventional ways to scale it. Built up from first
principles the way FIG 0 does for the kernel, landing on real sensor numbers."""
import os, sys
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 5)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
_OUT = os.path.dirname(os.path.abspath(__file__))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np, jax.numpy as jnp
from src.utils.config import Config
from src.environment.config_loader import load_env_config, load_env_params
from src.environment.sensor import _psf_weights, get_visual_offsets

plt.rcParams.update({'figure.dpi': 200, 'font.size': 9, 'savefig.bbox': 'tight',
                     'savefig.facecolor': 'white'})
PLUM, AMBER, TEAL, GREY = '#7b4b91', '#c9852b', '#1f7a6c', '#6e6579'
SIG = 0.2

d = load_env_config('configs/environment/default.yaml').to_dict()
d['sensory'].update(visual_sensor_range=2, visual_blur_enabled=True)
p = load_env_params(Config(d))
agent = jnp.array([5, 5]); cells = agent + get_visual_offsets(2)
peak = lambda dist: float(np.asarray(
    _psf_weights(cells, jnp.array([[5 - dist, 5]], dtype=jnp.float32), agent, p)).max())

NEAR, FAR = peak(1), peak(5)
fig, axes = plt.subplots(1, 4, figsize=(15.2, 4.3))

# ---- 1. what noise does to a single number ---------------------------------
ax = axes[0]
x = np.linspace(-0.4, 1.3, 500)
ax.plot(x, np.exp(-((x - NEAR) ** 2) / (2 * SIG ** 2)), color=PLUM, lw=2.2)
ax.fill_between(x, np.exp(-((x - NEAR) ** 2) / (2 * SIG ** 2)), color=PLUM, alpha=.15)
ax.axvline(NEAR, color='black', lw=1.6)
ax.annotate(f'the TRUE value\n{NEAR:.3f}', xy=(NEAR, 1.02), xytext=(NEAR - 0.62, 1.12),
            fontsize=8.4, arrowprops=dict(arrowstyle='->', lw=1.2))
rng = np.random.default_rng(1)
for v in NEAR + rng.normal(0, SIG, 7):
    ax.plot(v, 0.06, 'v', ms=7, mfc=AMBER, mec='black', mew=.5)
ax.text(NEAR, -0.13, 'seven actual readings', ha='center', fontsize=8, color='#8a5a12')
ax.annotate('', xy=(NEAR - SIG, 0.62), xytext=(NEAR + SIG, 0.62),
            arrowprops=dict(arrowstyle='<->', color=GREY, lw=1.4))
ax.text(NEAR, 0.68, 'σ = 0.2', ha='center', fontsize=9, color=GREY)
ax.set_ylim(-0.22, 1.35); ax.set_yticks([])
ax.set_xlabel('value the sensor reports\n'
              r'reported $=$ true $+\ \varepsilon$,  $\varepsilon \sim N(0,\sigma)$')
ax.set_title('1 · noise turns one number\ninto a spread of possible readings', fontsize=9.5)


# ---- 2. same noise, two signals -> SNR -------------------------------------
ax = axes[1]
for val, col, lab in ((NEAR, PLUM, f'object 1 cell away\ntrue {NEAR:.3f}'),
                      (FAR, TEAL, f'object 5 cells away\ntrue {FAR:.3f}')):
    y = np.exp(-((x - val) ** 2) / (2 * SIG ** 2))
    ax.plot(x, y, color=col, lw=2.2, label=f'{lab}\nSNR = {val/SIG:.2f}')
    ax.fill_between(x, y, color=col, alpha=.13)
ax.axvline(0, color='black', lw=1.4, ls='--')
ax.text(0.03, 0.40, '"nothing\nthere"', fontsize=8.0, color='0.35')
ax.set_ylim(0, 1.35); ax.set_yticks([]); ax.legend(fontsize=7.4, loc='upper right')
ax.set_xlabel('value the sensor reports')
ax.set_title('2 · same noise, two signals\nSNR = signal ÷ σ', fontsize=9.5)


# ---- 3. blur is systematic, noise is random --------------------------------
ax = axes[2]
offs = np.asarray(get_visual_offsets(2))
row = np.array([i for i, (r, c) in enumerate(offs) if r == 0])
order = row[np.argsort(offs[row][:, 1])]
clean = np.asarray(_psf_weights(cells, jnp.array([[5, 7]], dtype=jnp.float32), agent, p))[:, 0]
xs = offs[order][:, 1]
for k in range(6):
    ax.plot(xs, np.clip(clean[order] + rng.normal(0, SIG, len(order)), 0, None),
            color=AMBER, lw=1, alpha=.55)
ax.plot(xs, clean[order], color=PLUM, lw=3, marker='o', ms=6, label='the blur (same every step)')
ax.plot([], [], color=AMBER, lw=1, label='six noisy readings of it')
ax.set_xticks(xs); ax.set_xlabel('cells along one row of the diamond')
ax.set_ylabel('value'); ax.legend(fontsize=7.6); ax.grid(alpha=.22)
ax.set_title('3 · blur is SYSTEMATIC,\nnoise is RANDOM', fontsize=9.5)

# ---- 4. sigma models, each against the variable it actually uses ----------
ax = axes[3]
ds = np.arange(1, 8); sig = np.array([peak(x) for x in ds])
ax.plot(sig, np.full_like(sig, SIG), color=PLUM, lw=2.4, marker='o', ms=5,
        label=r'1 · flat:  $\sigma=\sigma_0$')
ax.plot(sig, SIG * np.sqrt(np.maximum(sig, 1e-9)), color=TEAL, lw=2.4, marker='s', ms=5,
        label=r'2 · shot:  $\sigma=\sigma_0\sqrt{s}$')
for xv, dv in zip(sig, ds):
    if dv in (1, 3, 5, 7):
        ax.annotate(f'd={dv}', xy=(xv, SIG), xytext=(xv, SIG + 0.035),
                    fontsize=7.2, ha='center', color=GREY)
ax.set_xscale('log')
ax.set_xlabel('signal strength $s$   (falls with distance)')
ax.set_ylabel('σ actually applied'); ax.set_ylim(0, 0.30)
ax.legend(fontsize=7.8, loc='upper left'); ax.grid(alpha=.22)
ax.text(0.97, 0.42, r'3 · retinal:  $\sigma=\sigma_0(1+k\,e)$' '\n'
        'keys on CELL eccentricity,\nnot on signal — a different\naxis, so not plotted here',
        transform=ax.transAxes, fontsize=7.6, color=AMBER, va='top', ha='right')
ax.set_title('4 · two conventional σ models,\nagainst the signal they depend on', fontsize=9.5)

fig.suptitle('FIG 8 — Anatomy of noise: what it is, and how it differs from blur', fontsize=11.5, y=1.04)
fig.text(.5, -.16,
         'Panel 1 — noise means the reported value is the true value plus a random draw, so reading the same cell twice gives two answers. Panel 2 — whether that matters depends '
         'entirely on\nthe signal: a near object (0.637) sits well clear of the noise, while a far one (0.037) overlaps zero, so "faint object" and "nothing there" become the same reading. '
         'Panel 3 — the difference\nthat matters most: the blur is IDENTICAL every step and could in principle be learned and undone, while the noise differs every step and cannot. '
         'Panel 4 — only σ changes between the\nmodels; the kernel is untouched in all of them. The third model is deliberately absent from that panel because it depends on where the CELL '
         'is rather than how strong the signal is,\nand sharing an axis would blur exactly the distinction this study is trying to keep sharp.',
         ha='center', fontsize=8.5)
fig.savefig(os.path.join(_OUT, 'fig8_noise_anatomy.png')); print('ok fig8')
