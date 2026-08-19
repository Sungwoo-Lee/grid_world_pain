"""Mechanism sandbox for the distance-degraded visual sensor.

Pure numpy — deliberately independent of src/environment/sensor.py so the
mechanisms can be compared before any of them is implemented.

Coordinates are (row, col) as in the environment. Row increases DOWNWARD, so
"top"/north is row-1 and "right"/east is col+1. Angles theta are measured from
straight up (north), increasing toward the right (east): theta=0 is due north,
theta=90 is due east, theta=45 is exactly top-right.
"""
import numpy as np

# ---------------------------------------------------------------- geometry ---

def diamond_offsets(r):
    """Manhattan diamond offsets, same set as get_visual_offsets(r)."""
    return np.array([(dr, dc)
                     for dr in range(-r, r + 1)
                     for dc in range(-r, r + 1)
                     if abs(dr) + abs(dc) <= r], dtype=float)


def polar_to_rc(theta_deg, dist):
    """Entity offset from the agent, from (angle-from-north, euclidean distance)."""
    th = np.radians(theta_deg)
    return np.array([-dist * np.cos(th), dist * np.sin(th)])


# ----------------------------------------------------------------- kernels ---
# Every kernel returns w[num_cells]: the weight with which one entity's
# appearance vector is written into each observed cell.

def k_exact(cells, e, **kw):
    """Today's sensor: exact cell match, no spreading."""
    return (np.abs(cells - e).sum(axis=1) < 1e-6).astype(float)


def k_isotropic(cells, e, scale=0.5, **kw):
    """Isotropic gaussian PSF, width growing with the entity's distance."""
    d = np.linalg.norm(e)
    sig = max(scale * d, 1e-6)
    v = cells - e
    w = np.exp(-(v ** 2).sum(axis=1) / (2 * sig ** 2))
    return w, 2 * np.pi * sig * sig


def k_anisotropic(cells, e, radial_scale=0.5, angular_deg=10.0, ratio=None, **kw):
    """Gaussian PSF elongated ALONG the agent->entity ray.

    sigma_par  = radial_scale * d          (distance judgement degrades)
    sigma_perp = d * sin(angular_deg)      (constant angular resolution)
    """
    d = np.linalg.norm(e)
    if d < 1e-9:                                   # entity on the agent's cell
        return k_isotropic(cells, e, scale=radial_scale)
    u = e / d                                      # radial unit vector
    t = np.array([-u[1], u[0]])                    # tangential unit vector
    sp = max(radial_scale * d, 1e-6)
    # ratio overrides angular_deg: sigma_perp = sigma_par / rho  (rho = 1 -> isotropic)
    st = (sp / ratio) if ratio is not None else d * np.sin(np.radians(angular_deg))
    st = max(st, 1e-6)
    v = cells - e
    w = np.exp(-(v @ u) ** 2 / (2 * sp ** 2) - (v @ t) ** 2 / (2 * st ** 2))
    return w, 2 * np.pi * sp * st


# ----------------------------------------------------------- normalisation ---

def normalise(w, full_mass, mode):
    """mode: 'mass' (full analytic integral) | 'diamond' (visible cells) | 'none'."""
    if mode == 'mass':
        return w / full_mass
    if mode == 'diamond':
        s = w.sum()
        return w / s if s > 1e-12 else w
    return w


def readout(cells, e, kernel='aniso', norm='mass', **kw):
    """One entity's contribution to every observed cell."""
    if kernel == 'exact':
        return k_exact(cells, e)
    w, mass = (k_isotropic if kernel == 'iso' else k_anisotropic)(cells, e, **kw)
    return normalise(w, mass, norm)


# ------------------------------------------------- alternative architectures ---

def sector_readout(entities, n_sectors=8, near_radius=1.5, compress=1.0):
    """Far-field wedge channels: everything beyond the sharp diamond collapsed
    into n_sectors directional bins, with distance compressed away."""
    out = np.zeros(n_sectors)
    for e in entities:
        d = np.linalg.norm(e)
        if d <= near_radius:
            continue
        ang = np.degrees(np.arctan2(e[1], -e[0])) % 360.0   # 0 = north, cw
        idx = int(((ang + 180.0 / n_sectors) % 360.0) / (360.0 / n_sectors))
        out[idx] += 1.0 / (d ** compress)
    return out


def logpolar_readout(entities, n_ang=8, ring_edges=(1.0, 3.0, 9.0)):
    """Retinotopic map: constant angular bins, log-spaced radial rings."""
    out = np.zeros((len(ring_edges), n_ang))
    for e in entities:
        d = np.linalg.norm(e)
        ring = next((i for i, hi in enumerate(ring_edges) if d < hi), None)
        if ring is None:
            continue
        ang = np.degrees(np.arctan2(e[1], -e[0])) % 360.0
        idx = int(((ang + 180.0 / n_ang) % 360.0) / (360.0 / n_ang))
        out[ring, idx] += 1.0
    return out
