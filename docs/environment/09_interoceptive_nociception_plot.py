"""Conceptual figure for docs/environment/09_sensors_and_observation.md.

Two-panel comparison of how the alpha-kernel interoceptive nociception sensor
tracks the (hidden) injury body state across kernel time-constants:

  (a) Transient injury — single damage event followed by rest.
  (b) Sustained injury — repeated hits, no rest, demonstrating that the
      perceived signal converges to the hidden injury level (DC gain = 1).

Three τ values (2, 4, 8) show the speed/smoothing trade-off. An inset in panel
(a) overlays the three alpha kernels for direct shape comparison.

Run with the `grid_world_pain` conda env; saves the figure next to this
script as `09_interoceptive_nociception_dynamics.png`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


C_INJURY = "#1B4F72"   # dark blue: hidden body state
C_DAMAGE = "#B9770E"
C_TEXT = "#2C3E50"
C_MUTED = "#7F8C8D"
C_DAMAGE_BAND = "#FDEBD0"

# τ → curve color (light → dark = fast → slow kernel).
TAU_COLORS = {
    1.0: "#F39C12",
    2.0: "#E67E22",
    4.0: "#C0392B",
    8.0: "#641E16",
}


def simulate_injury(rest_pattern, damage_steps, damage, smoothing_duration,
                    recovery_base=0.1, recovery_accel=0.5, max_injury=100.0):
    """Replicates the body update loop from core.py for `len(rest_pattern)` steps.

    Supports multiple damage events via `damage_steps` (list of step indices).
    """
    n_steps = len(rest_pattern)
    buffer = np.zeros(smoothing_duration)
    injury = 0.0
    streak = 0
    injuries = np.zeros(n_steps)
    damage_set = {int(s) for s in damage_steps}

    for t in range(n_steps):
        inc_from_new = damage / smoothing_duration if t in damage_set else 0.0
        temp_buffer = buffer + inc_from_new
        applied_inc = temp_buffer[0]
        injury = injury + applied_inc
        buffer = np.concatenate([temp_buffer[1:], [0.0]])

        rested = bool(rest_pattern[t])
        streak = streak + 1 if rested else 0
        can_recover = rested and (applied_inc <= 0)
        mult = (1.0 + recovery_accel) ** (max(streak, 1) - 1)
        rec = recovery_base * mult if can_recover else 0.0

        injury = float(np.clip(injury - rec, 0.0, max_injury))
        injuries[t] = injury
    return injuries


def alpha_kernel(length, tau):
    """Discrete alpha kernel; k[0]=0, peaks at i=tau, sum(k)=1.

    k_raw[i] = (i/tau) * exp(1 - i/tau)
    k[i]     = k_raw[i] / sum(k_raw)
    """
    i = np.arange(length, dtype=np.float64)
    k_raw = (i / tau) * np.exp(1.0 - i / tau)
    return k_raw / k_raw.sum()


def causal_fir(signal, kernel):
    """Causal FIR: y[t] = Σ_i k[i] * x[t - i] (zero-padded for t < i)."""
    return np.convolve(signal, kernel, mode="full")[: len(signal)]


def first_zero_after(inj, after, eps=1e-9):
    tail = inj[after:]
    mask = tail <= eps
    if not mask.any():
        return None
    return after + int(np.argmax(mask))


def main():
    # ---- Shared kernels ----------------------------------------------
    taus = [1.0, 2.0, 4.0, 8.0]
    kernel_length = 30  # large enough that the slowest kernel's tail is captured
    kernels = {tau: alpha_kernel(kernel_length, tau) for tau in taus}

    smoothing_duration = 10
    max_injury = 100.0
    n_steps = 100

    # ---- Case (a): Transient — single hit + rest ----------------------
    damage_a_step = 5
    damage_a = 25.0
    rest_from_a = damage_a_step + smoothing_duration
    rest_a = np.zeros(n_steps, dtype=bool)
    rest_a[rest_from_a:] = True
    injury_a = simulate_injury(rest_a, [damage_a_step], damage_a,
                               smoothing_duration, max_injury=max_injury)
    healed_a = first_zero_after(injury_a, rest_from_a) or (n_steps - 1)
    rest_a[healed_a + 1:] = False
    injury_a = simulate_injury(rest_a, [damage_a_step], damage_a,
                               smoothing_duration, max_injury=max_injury)

    # ---- Case (b): Sustained — repeated hits, no rest -----------------
    damage_b_steps = [5, 12, 19, 26, 33]
    damage_b = 15.0
    rest_b = np.zeros(n_steps, dtype=bool)  # never rest
    injury_b = simulate_injury(rest_b, damage_b_steps, damage_b,
                               smoothing_duration, max_injury=max_injury)

    inj_a = injury_a / max_injury
    inj_b = injury_b / max_injury
    intero_a = {tau: causal_fir(inj_a, kernels[tau]) for tau in taus}
    intero_b = {tau: causal_fir(inj_b, kernels[tau]) for tau in taus}

    # ---- Plot ----------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": C_MUTED,
    })

    fig, (axL, axR, axK) = plt.subplots(
        1, 3, figsize=(15, 5.4),
        gridspec_kw={"width_ratios": [4, 4, 2]},
    )
    steps = np.arange(n_steps)

    # ── Panel (a): Transient ───────────────────────────────────────────
    axL.axvspan(damage_a_step, damage_a_step + smoothing_duration,
                color=C_DAMAGE_BAND, alpha=0.6, zorder=0, label="Damage absorbing")
    axL.plot(steps, inj_a, color=C_INJURY, lw=2.6, solid_capstyle="round",
             label="Hidden injury", zorder=3)
    for tau in taus:
        axL.plot(steps, intero_a[tau], color=TAU_COLORS[tau], lw=1.9, linestyle="--",
                 dash_capstyle="round", label=f"Intero  τ={int(tau)}", zorder=3)

    axL.annotate(
        "Damage",
        xy=(damage_a_step, 0.010), xytext=(damage_a_step - 6, 0.06),
        fontsize=11, color=C_DAMAGE, fontweight="bold", ha="right",
        arrowprops=dict(arrowstyle="->", color=C_DAMAGE, lw=1.0),
    )
    axL.text(
        damage_a_step + smoothing_duration + 1.5, 0.005,
        "k[0] = 0  →  no instantaneous leak",
        fontsize=9.5, color=C_MUTED, style="italic", ha="left", va="bottom",
    )
    axL.set_title("(a) Transient injury — single hit + rest",
                  fontsize=12, color=C_TEXT, pad=10, loc="left", fontweight="bold")

    # ── Panel (b): Sustained ───────────────────────────────────────────
    # Draw a damage band per event; first one labelled for the legend.
    for k, ds in enumerate(damage_b_steps):
        axR.axvspan(ds, ds + smoothing_duration, color=C_DAMAGE_BAND,
                    alpha=0.45, zorder=0,
                    label="Damage absorbing" if k == 0 else None)
    axR.plot(steps, inj_b, color=C_INJURY, lw=2.6, solid_capstyle="round",
             label="Hidden injury", zorder=3)
    for tau in taus:
        axR.plot(steps, intero_b[tau], color=TAU_COLORS[tau], lw=1.9, linestyle="--",
                 dash_capstyle="round", label=f"Intero  τ={int(tau)}", zorder=3)

    plateau = inj_b[-1]
    axR.axhline(plateau, color=C_INJURY, lw=0.9, linestyle=":", alpha=0.55, zorder=2)
    axR.annotate(
        "Steady state:  intero → injury  (DC gain = 1)",
        xy=(n_steps - 4, plateau), xytext=(n_steps - 4, plateau + 0.10),
        ha="right", va="bottom", fontsize=10, color=C_TEXT,
        arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=0.9),
    )

    axR.annotate(
        "Repeated hits",
        xy=(damage_b_steps[2], 0.012), xytext=(damage_b_steps[2] + 4, 0.07),
        fontsize=11, color=C_DAMAGE, fontweight="bold", ha="left",
        arrowprops=dict(arrowstyle="->", color=C_DAMAGE, lw=1.0),
    )
    axR.set_title("(b) Sustained injury — repeated hits, no rest",
                  fontsize=12, color=C_TEXT, pad=10, loc="left", fontweight="bold")

    # ── Panel (c): Alpha kernels ───────────────────────────────────────
    k_idx = np.arange(kernel_length)
    for tau in taus:
        axK.plot(k_idx, kernels[tau], color=TAU_COLORS[tau], lw=2.0, label=f"τ = {int(tau)}")
    axK.set_title("(c) Alpha kernels", fontsize=12, color=C_TEXT, pad=10,
                  loc="left", fontweight="bold")
    axK.set_xticks([0, kernel_length - 1])
    axK.set_xticklabels(["0", f"{kernel_length - 1}"], fontsize=9, color=C_MUTED)
    axK.set_xlabel("Lag (steps)", fontsize=10, color=C_MUTED, labelpad=3)
    axK.set_yticks([])
    axK.tick_params(axis="x", length=3, pad=2, colors=C_MUTED)
    axK.spines["right"].set_visible(False)
    axK.spines["top"].set_visible(False)
    axK.spines["left"].set_color(C_MUTED)
    axK.spines["bottom"].set_color(C_MUTED)
    axK.legend(fontsize=10, frameon=False, loc="upper right",
               handlelength=1.6, labelspacing=0.4)

    # ── Shared axis cosmetics ──────────────────────────────────────────
    top_a = max(inj_a.max(), max(intero_a[t].max() for t in taus))
    top_b = max(inj_b.max(), max(intero_b[t].max() for t in taus))
    for ax, top in [(axL, top_a), (axR, top_b)]:
        ax.set_xlabel("Time", fontsize=11, color=C_TEXT)
        ax.set_xlim(0, n_steps)
        ax.set_ylim(-0.02, top * 1.30)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    axL.set_ylabel("Normalized intensity", fontsize=11, color=C_TEXT)

    axL.legend(loc="upper right", fontsize=10, frameon=False, handlelength=2.2)
    axR.legend(loc="lower right", fontsize=10, frameon=False, handlelength=2.2)

    fig.suptitle(
        "Hidden injury vs. perceived nociception — slower kernels lag and smooth more",
        fontsize=13.5, fontweight="bold", color=C_TEXT, y=0.995,
    )

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = Path(__file__).resolve().parent / "09_interoceptive_nociception_dynamics.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
