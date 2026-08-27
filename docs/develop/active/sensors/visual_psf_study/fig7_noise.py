"""FIG 7 — why the sensor already has distance-dependent noise, and what the
three conventional sigma models do about it. Uses the REAL _psf_weights."""
import os, sys
os.environ.setdefault('JAX_PLATFORMS', 'cpu')
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *(['..'] * 5)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
_OUT = os.path.dirname(os.path.abspath(__file__))
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np, jax.numpy as jnp
from src.utils.config import Config
from src.environment.config_loader import load_env_config, load_env_params
from src.environment.sensor import _psf_weights, get_visual_offsets

plt.rcParams.update({'figure.dpi': 130, 'font.size': 9, 'axes.titlesize': 9.5,
                     'savefig.bbox': 'tight', 'savefig.facecolor': 'white'})
PLUM, AMBER, TEAL, GREY = '#7b4b91', '#c9852b', '#1f7a6c', '#6e6579'
SIG = 0.2

d = load_env_config('configs/environment/default.yaml').to_dict()
d['sensory'].update(visual_sensor_range=2, visual_blur_enabled=True)
p = load_env_params(Config(d))
agent = jnp.array([5, 5]); cells = agent + get_visual_offsets(2)
offs = np.asarray(get_visual_offsets(2))


def readout(dist):
    e = jnp.array([[5 - dist, 5]], dtype=jnp.float32)
    return np.asarray(_psf_weights(cells, e, agent, p))[:, 0]


fig = plt.figure(figsize=(13.4, 7.2))
gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0], hspace=.42, wspace=.28)

# --- row 1: the diamond, clean vs noisy, at three distances ------------------
rng = np.random.default_rng(0)
for j, dist in enumerate((1, 3, 5)):
    clean = readout(dist)
    noisy = np.clip(clean + rng.normal(0, SIG, clean.shape), 0, None)
    ax = fig.add_subplot(gs[0, j])
    vmax = max(clean.max(), noisy.max(), 1e-9)
    cm = plt.get_cmap('magma')
    for k, (rr, cc) in enumerate(offs):
        ax.add_patch(Rectangle((cc - .46, -rr - .46), .92, .92,
                               facecolor=cm(clean[k] / vmax), edgecolor='0.8', lw=.5))
        ax.add_patch(Rectangle((cc - .46 + 5.6, -rr - .46), .92, .92,
                               facecolor=cm(noisy[k] / vmax), edgecolor='0.8', lw=.5))
    ax.plot(0, 0, marker='x', ms=6, color='#4dd0ff')
    ax.plot(5.6, 0, marker='x', ms=6, color='#4dd0ff')
    ax.text(0, 2.9, 'no noise', ha='center', fontsize=8.5)
    ax.text(5.6, 2.9, 'with noise', ha='center', fontsize=8.5)
    ax.set_xlim(-2.4, 8.0); ax.set_ylim(-2.8, 3.5)
    ax.set_aspect('equal'); ax.axis('off')
    ax.set_title(f'object {dist} cell{"s" if dist>1 else ""} away\n'
                 f'brightest cell = {clean.max():.3f}  ·  noise σ = {SIG}', fontsize=9)

# --- row 2 left: signal vs the noise floor ----------------------------------
ax = fig.add_subplot(gs[1, 0])
ds = np.arange(1, 8)
sig = np.array([readout(x).max() for x in ds])
ax.plot(ds, sig, color=PLUM, lw=2.4, marker='o', ms=4, label='signal (brightest cell)')
ax.axhline(SIG, color=AMBER, lw=2, ls='--', label=f'noise σ = {SIG}')
cross = ds[np.argmax(sig < SIG)]
ax.axvspan(cross - .5, 7.5, color=AMBER, alpha=.10)
ax.annotate(f'signal below the\nnoise from {cross} cells out',
            xy=(cross + .1, SIG * 1.6), fontsize=8.4, color='#8a5a12')
ax.set_yscale('log'); ax.grid(alpha=.25); ax.legend(fontsize=7.8)
ax.set_xlabel('object distance (cells)'); ax.set_ylabel('value')
ax.set_title('the signal already falls — noise need not vary at all')

# --- row 2 middle: SNR under the three conventional sigma models -------------
ax = fig.add_subplot(gs[1, 1])
ecc = np.array([np.abs(offs[np.argmax(readout(x))]).sum() for x in ds])
ax.plot(ds, sig / SIG, color=PLUM, lw=2.2, marker='o', ms=4, label='1 · flat σ  (read noise)')
ax.plot(ds, sig / (SIG * np.sqrt(np.maximum(sig, 1e-9))), color=TEAL, lw=2.2, marker='s', ms=4,
        label='2 · σ ∝ √signal  (shot noise)')
ax.plot(ds, sig / (SIG * (1 + 0.5 * ecc)), color=AMBER, lw=2.2, marker='^', ms=4,
        label='3 · σ grows with cell eccentricity')
ax.axhline(1.0, color=GREY, lw=1.2, ls=':')
ax.text(7.4, 1.15, 'SNR = 1', fontsize=7.6, color=GREY, ha='right')
ax.set_yscale('log'); ax.grid(alpha=.25); ax.legend(fontsize=7.4, loc='lower left', framealpha=.95)
ax.set_xlabel('object distance (cells)'); ax.set_ylabel('signal ÷ noise')
ax.set_title('three conventional ways to scale σ')

# --- row 2 right: the two different "distances" -----------------------------
ax = fig.add_subplot(gs[1, 2])
for (rr, cc) in offs:
    ax.add_patch(Rectangle((cc - .46, -rr - .46), .92, .92,
                           facecolor='#e8e2ee', edgecolor='0.75', lw=.6))
    ax.text(cc, -rr, f'{int(abs(rr)+abs(cc))}', ha='center', va='center', fontsize=7.5, color=GREY)
ax.plot(0, 0, marker='X', ms=11, mfc='#111', mec='white', mew=1)
ax.plot(1.0, 3.2, 'o', ms=9, mfc='#4ad97a', mec='black', mew=.7)
ax.annotate('', xy=(0.9, 2.9), xytext=(0.1, 0.35),
            arrowprops=dict(arrowstyle='->', color=PLUM, lw=1.8))
ax.text(1.5, 1.7, 'ENTITY distance\nsets the BLUR', fontsize=8.2, color=PLUM)
ax.text(-2.9, -3.5, 'numbers in cells = CELL eccentricity\n(what option 3 would use)',
        fontsize=8.2, color=AMBER)
ax.set_xlim(-3.2, 4.2); ax.set_ylim(-4.4, 4.2)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title('two different "distances"')

fig.suptitle('FIG 7 — Distance-dependent noise: mostly already there', fontsize=11.5, y=0.99)
fig.text(.5, -.03,
         'Top: the same object at 1, 3 and 5 cells, before and after adding the σ = 0.2 the project already configures for vision. At one cell it survives; by five it is gone. '
         'Nothing about\nthe noise changed between those panels — only the signal, which the mass-normalised kernel already shrinks with distance. That is why a CONSTANT σ '
         'is already distance-dependent\nin the only sense that matters: signal-to-noise.',
         ha='center', fontsize=8.5)
fig.savefig(os.path.join(_OUT, 'fig7_noise.png')); print('ok fig7')
