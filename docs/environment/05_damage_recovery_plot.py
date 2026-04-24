"""Conceptual figure for docs/environment/05_body_homeostasis.md.

Single-panel schematic of how a damage event ramps injury up and how rest
recovers it — compared for continuing vs intermittent rest. Kept intentionally
spare so the idea reads in five seconds.

Run with the `grid_world_pain` conda env; saves the figure next to this
script as `05_damage_recovery_dynamics.png`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


C_CONT = "#1B4F72"
C_INTER = "#C0392B"
C_DAMAGE = "#B9770E"
C_TEXT = "#2C3E50"
C_MUTED = "#7F8C8D"
C_DAMAGE_BAND = "#FDEBD0"


def simulate(rest_pattern, damage_step, damage, smoothing_duration,
             recovery_base=0.1, recovery_accel=0.5, max_injury=100.0):
    """Run the injury update loop from core.py for `len(rest_pattern)` steps."""
    n_steps = len(rest_pattern)
    buffer = np.zeros(smoothing_duration)
    injury = 0.0
    streak = 0

    injuries = np.zeros(n_steps)
    recoveries = np.zeros(n_steps)

    for t in range(n_steps):
        inc_from_new = damage / smoothing_duration if t == damage_step else 0.0
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
        recoveries[t] = rec

    return injuries, recoveries


def first_zero_after(inj, after):
    tail = inj[after:]
    mask = tail <= 1e-9
    if not mask.any():
        return None
    return after + int(np.argmax(mask))


def main():
    n_steps = 150
    damage_step = 5
    damage = 20.0
    smoothing_duration = 10
    rest_from = damage_step + smoothing_duration
    absorption_end = damage_step + smoothing_duration - 1

    continuing = np.zeros(n_steps, dtype=bool)
    continuing[rest_from:] = True

    intermittent = np.zeros(n_steps, dtype=bool)
    for t in range(rest_from, n_steps):
        intermittent[t] = ((t - rest_from) % 5) != 4

    cont_inj, _ = simulate(continuing, damage_step, damage, smoothing_duration)
    inter_inj, _ = simulate(intermittent, damage_step, damage, smoothing_duration)

    cont_end = first_zero_after(cont_inj, rest_from)
    inter_end = first_zero_after(inter_inj, rest_from)

    # Stop resting once healed, so the curve stays flat at 0 afterwards.
    ct = continuing.copy(); ct[cont_end + 1:] = False
    it = intermittent.copy(); it[inter_end + 1:] = False
    cont_inj, _ = simulate(ct, damage_step, damage, smoothing_duration)
    inter_inj, _ = simulate(it, damage_step, damage, smoothing_duration)

    # ---- Plot ----------------------------------------------------------
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": C_MUTED,
    })

    x_max = inter_end + 8
    steps = np.arange(n_steps)

    fig, ax = plt.subplots(figsize=(11, 5.5))

    # Light shaded damage window (neutral amber, not red).
    ax.axvspan(damage_step, absorption_end + 1, color=C_DAMAGE_BAND,
               alpha=0.6, zorder=0, label="Damage absorbing")

    # Curves.
    ax.plot(steps, cont_inj, color=C_CONT, lw=2.8, solid_capstyle="round",
            label="Continuous rest", zorder=3)
    ax.plot(steps, inter_inj, color=C_INTER, lw=1.8, linestyle="--",
            dash_capstyle="round", label="Intermittent rest", zorder=3)

    # Three minimal event labels.
    ax.annotate(
        "Damage",
        xy=(damage_step, 1.5), xytext=(damage_step - 8, 8),
        fontsize=11, color=C_DAMAGE, fontweight="bold", ha="right",
        arrowprops=dict(arrowstyle="->", color=C_DAMAGE, lw=1.1),
    )
    ax.annotate(
        "Rest begins",
        xy=(rest_from, cont_inj[rest_from - 1]),
        xytext=(rest_from + 4, 22),
        fontsize=11, color=C_TEXT,
        arrowprops=dict(arrowstyle="->", color=C_MUTED, lw=1.0),
    )

    # Endpoint dots + simple labels.
    ax.scatter([cont_end], [0], color=C_CONT, s=110, zorder=5,
               edgecolor="white", linewidths=1.6)
    ax.scatter([inter_end], [0], color=C_INTER, s=110, zorder=5,
               edgecolor="white", linewidths=1.6)
    ax.text(cont_end, -2.2, "Healed", ha="center", va="top",
            fontsize=11, color=C_CONT, fontweight="bold")
    ax.text(inter_end, -2.2, "Healed", ha="center", va="top",
            fontsize=11, color=C_INTER, fontweight="bold")

    ax.set_xlabel("Time", fontsize=12, color=C_TEXT)
    ax.set_ylabel("Injury", fontsize=12, color=C_TEXT)
    ax.set_xlim(0, x_max)
    ax.set_ylim(-4, 26)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    ax.legend(loc="upper right", fontsize=11, frameon=False)

    ax.set_title(
        "Injury rises with damage, falls with rest — "
        "continuous rest recovers faster",
        fontsize=13, fontweight="bold", color=C_TEXT, pad=14,
    )

    fig.tight_layout()
    out_path = Path(__file__).resolve().parent / "05_damage_recovery_dynamics.png"
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
