"""Plot 20 simulated force curves across billet temperature and ram speed (Task 1.1 deliverable).

    python -m skinflow_discard_optimizer.sim.plot_force_curves
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from skinflow_discard_optimizer.paths import REPORTS_DIR  # noqa: E402
from skinflow_discard_optimizer.sim.force_model import Press, nominal_stroke, simulate_stroke  # noqa: E402


def main(out: Path | None = None, alloy: str = "AA6063") -> Path:
    press = Press.load()
    temps = np.linspace(440, 500, 5)
    speeds = np.linspace(6, 18, 4)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    cmap = plt.get_cmap("coolwarm")
    rng = np.random.default_rng(0)
    for i, T in enumerate(temps):
        for v in speeds:
            spec = nominal_stroke(alloy, press, T_front_C=float(T), ram_speed_mm_s=float(v))
            d = simulate_stroke(spec, press, rng)
            F = d.force_measured_N(press) / 1e6
            step = 20  # thin 1 kHz data for plotting
            color = cmap(i / (len(temps) - 1))
            lw = 0.6 + 0.4 * (v - speeds[0]) / (speeds[-1] - speeds[0])
            axes[0].plot(d.x_mm[::step], F[::step], color=color, lw=lw, alpha=0.85)
            tail = d.h_true_mm < 80
            axes[1].plot(d.h_true_mm[tail][::5], F[tail][::5], color=color, lw=lw, alpha=0.85)
    axes[0].set(xlabel="Ram stroke x (mm)", ylabel="Ram force (MN)",
                title=f"{alloy}: 20 strokes, T = 440-500 °C (blue→red), v = 6-18 mm/s (thin→thick)")
    axes[1].set(xlabel="Remaining thickness h (mm)", title="End of stroke (upturn)")
    axes[1].invert_xaxis()
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("Simulated ram force from measured cap/rod pressure (placeholder constants)", fontsize=10)
    fig.tight_layout()
    out = out or REPORTS_DIR / "figures" / "task_1_1_force_curves.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--alloy", default="AA6063")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    print(main(a.out, a.alloy))
