"""Mechanism sandbox for the per-cell (directional) olfactory sensor.

Pure numpy, deliberately independent of src/environment/sensor.py, so the
proposal can be measured before any of it is implemented.

Mirrors sense_resource() exactly: for a sampling point p and sources at e_n
carrying chemical vectors prop_n,

    d_n     = ||e_n - p||
    decay_n = 2.0                 if d_n < 0.001      (sampling point ON the source)
            = 1 / (d_n^gamma)     otherwise
    mask_n  = active_n AND (d_n <= radius)
    c(p)    = sum_n prop_n * decay_n * mask_n

Today the sensor evaluates c() once, at the agent's cell. The proposal evaluates
it at every cell of a Manhattan diamond, which is what makes it directional.

Coordinates are (row, col); row increases DOWNWARD, so north is -row. Bearings
theta are measured from north, increasing toward east.
"""
import numpy as np

# --- values taken from configs/environment/default.yaml -----------------------
GAMMA_CFG   = 1.0    # sensory.decay_power
RADIUS_CFG  = 20.0   # sensory.sensor_radius  (exceeds the 10x10 grid diagonal)
SIGMA_CFG   = 0.2    # perceptual_noise.modalities.olfaction.sigma
VECSIZE_CFG = 5      # sensory.vector_size
GRID        = 10


def diamond_offsets(r):
    return np.array([(dr, dc)
                     for dr in range(-r, r + 1)
                     for dc in range(-r, r + 1)
                     if abs(dr) + abs(dc) <= r], dtype=float)


def polar_to_rc(theta_deg, dist):
    th = np.radians(theta_deg)
    return np.array([-dist * np.cos(th), dist * np.sin(th)])


def sense_at(points, sources, props, gamma=GAMMA_CFG, radius=RADIUS_CFG):
    """Chemical vector sensed at each sampling point. points [P,2] -> [P, V]."""
    points = np.atleast_2d(np.asarray(points, float))
    sources = np.atleast_2d(np.asarray(sources, float))
    props = np.atleast_2d(np.asarray(props, float))
    diff = sources[None, :, :] - points[:, None, :]         # [P, N, 2]
    d = np.linalg.norm(diff, axis=-1)                       # [P, N]
    decay = np.where(d < 0.001, 2.0, 1.0 / (np.power(d, gamma) + 1e-10))
    decay = decay * (d <= radius)
    return decay @ props                                    # [P, V]


def readout(agent, sources, props, rng_range=1, **kw):
    """The proposed per-cell olfactory observation: [num_cells, V], flattened in use."""
    cells = np.asarray(agent, float) + diamond_offsets(rng_range)
    return sense_at(cells, sources, props, **kw), cells


# --- direction recovery -------------------------------------------------------

def gradient_bearing(vals, rng_range=1):
    """Estimate the source bearing from a diamond readout of ONE channel.

    Central difference on the 4 axis-neighbours at distance 1. Returns degrees
    from north, increasing toward east.
    """
    cells = diamond_offsets(rng_range)
    idx = lambda rc: int(np.argmin(np.abs(cells - np.array(rc, float)).sum(axis=1)))
    N, S = vals[idx((-1, 0))], vals[idx((1, 0))]
    E, W = vals[idx((0, 1))], vals[idx((0, -1))]
    g_row, g_col = (S - N) / 2.0, (E - W) / 2.0      # gradient points UP the field
    return np.degrees(np.arctan2(g_col, -g_row))


def bearing_error(true_deg, est_deg):
    """Smallest absolute angular difference, in degrees."""
    return np.abs((est_deg - true_deg + 180.0) % 360.0 - 180.0)


def adjacent_contrast(d, gamma=GAMMA_CFG):
    """Absolute difference between the cell one step toward a source and the
    centre cell, for a unit-strength source at distance d."""
    return 1.0 / (d - 1.0) ** gamma - 1.0 / d ** gamma


def gradient_bearing_lsq(vals, rng_range=1):
    """Bearing from a least-squares planar fit over ALL diamond cells.

    Unlike the 4-neighbour central difference, this uses every cell, so a larger
    diamond genuinely averages down noise — at the cost of more curvature bias,
    since the field is 1/d^gamma and not a plane.
    """
    cells = diamond_offsets(rng_range)
    A = np.column_stack([cells[:, 0], cells[:, 1], np.ones(len(cells))])
    coef, *_ = np.linalg.lstsq(A, np.asarray(vals, float), rcond=None)
    g_row, g_col = coef[0], coef[1]          # gradient of the fitted plane
    return np.degrees(np.arctan2(g_col, -g_row))
