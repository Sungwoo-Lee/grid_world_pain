"""Conceptual figure for docs/environment/10_perceptual_noise.md.

Single-panel schematic showing how state-dependent perceptual noise grows
when the agent is in pain: healthy → observation hugs the true signal;
injured → σ_eff = σ_base × (1 + α·injury_norm) inflates, so the same
sensor returns a visibly noisier trace.

Run with the `grid_world_pain` conda env; saves the figure next to this
script as `10_perceptual_noise_dynamics.png`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


C_HEALTHY = "#1B4F72"
C_INJURED = "#C0392B"
C_TRUTH = "#7F8C8D"
C_TEXT = "#2C3E50"
C_MUTED = "#7F8C8D"
C_INJURED_BAND = "#FDEBD0"


def main():
    # Values slightly exaggerated vs doc defaults (σ=0.10, α=1.5) so the
    # healthy trace reads "clean" and the injured trace reads "noisy"
    # at a glance. Same mechanism: σ_eff = σ_base · (1 + α · injury_norm).
    n = 140
    transition = n // 2
    sigma_base = 0.025
    alpha = 6.0  # → σ_healthy = 0.025, σ_injured = 0.175 (7×)

    rng = np.random.default_rng(3)

    injury = np.zeros(n)
    injury[transition:] = 1.0
    sigma = sigma_base * (1.0 + alpha * injury)
    noise = rng.standard_normal(n) * sigma

    # Smooth underlying "true" signal the sensor is trying to report.
    t = np.linspace(0, 3 * np.pi, n)
    truth = 0.5 + 0.28 * np.sin(t)
    obs = np.clip(truth + noise, 0.0, 1.0)  # matches the sensor clip

    # ---- Plot ----------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": C_MUTED,
    })

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Shaded "injured" phase, matching the style of the recovery figure.
    ax.axvspan(transition, n - 1, color=C_INJURED_BAND, alpha=0.6, zorder=0,
               label="Injured (in pain)")

    # True value — thin dotted reference line.
    ax.plot(np.arange(n), truth, color=C_TRUTH, lw=1.4, linestyle=":",
            label="True signal", zorder=2)

    # Observation trace, split at pain onset.
    ax.plot(np.arange(transition + 1), obs[:transition + 1],
            color=C_HEALTHY, lw=2.2, solid_capstyle="round",
            label="Healthy perception (small σ)", zorder=3)
    ax.plot(np.arange(transition, n), obs[transition:],
            color=C_INJURED, lw=1.8, linestyle="--", dash_capstyle="round",
            label="Injured perception (large σ)", zorder=3)

    # Transition guideline.
    ax.axvline(transition, color=C_INJURED, linestyle=":", alpha=0.6, lw=1.2,
               zorder=1)

    # Three minimal labels — parallel to the recovery figure.
    ax.annotate(
        "Pain onset",
        xy=(transition, 0.92), xytext=(transition + 8, 1.06),
        fontsize=11, color=C_INJURED, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=C_INJURED, lw=1.1),
    )
    ax.text(transition * 0.5, 0.03, "Clear",
            ha="center", va="center",
            fontsize=11, color=C_HEALTHY, fontweight="bold")
    ax.text((transition + n) * 0.5, 0.03, "Noisy",
            ha="center", va="center",
            fontsize=11, color=C_INJURED, fontweight="bold")

    ax.set_xlabel("Time", fontsize=12, color=C_TEXT)
    ax.set_ylabel("Observed signal", fontsize=12, color=C_TEXT)
    ax.set_xlim(0, n - 1)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    ax.legend(loc="upper right", fontsize=11, frameon=False)

    ax.set_title(
        "Injury degrades perception — the same sensor "
        "returns noisier observations when in pain",
        fontsize=13, fontweight="bold", color=C_TEXT, pad=14,
    )

    fig.tight_layout()
    out_path = Path(__file__).resolve().parent / "10_perceptual_noise_dynamics.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
