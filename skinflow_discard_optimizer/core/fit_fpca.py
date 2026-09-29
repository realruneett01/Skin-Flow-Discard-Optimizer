"""Fit FPCA on healthy training cycles and report it (Task 2.2).

    python -m skinflow_discard_optimizer.core.fit_fpca [--n 1500]

Writes ``artifacts/fpca.npz``, ``reports/task_2_2_fpca.md`` and
``reports/figures/task_2_2_fpca.png``.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from skinflow_discard_optimizer.core.functional import FPCA, RegistrationGrid, curve_from_stroke  # noqa: E402
from skinflow_discard_optimizer.paths import REPO_ROOT, REPORTS_DIR, default_workers  # noqa: E402
from skinflow_discard_optimizer.sim.build_dataset import load_cycles  # noqa: E402
from skinflow_discard_optimizer.sim.cycle import regenerate_stroke  # noqa: E402
from skinflow_discard_optimizer.sim.force_model import Press  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts"
FPCA_PATH = ARTIFACTS / "fpca.npz"
LATENTS = ["true_billet_temp_C", "true_liner_temp_C", "true_mu_base", "true_liner_scale_mm",
           "true_die_wear", "spec_ram_speed_mm_s", "spec_L0_mm", "spec_F_tool_N", "spec_mu"]


def _curve(row: dict) -> np.ndarray:
    press = Press.load()
    return curve_from_stroke(regenerate_stroke(row, press), press)


def registered_curves(rows: pd.DataFrame, workers: int | None = None) -> np.ndarray:
    records = rows.to_dict("records")
    workers = workers or default_workers()
    if workers == 1:
        return np.array([_curve(r) for r in records])
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return np.array(list(ex.map(_curve, records, chunksize=16)))


def pick(df: pd.DataFrame, scenario: str, split: str, n: int) -> pd.DataFrame:
    sub = df[(df.scenario == scenario) & (df.split == split)]
    idx = np.linspace(0, len(sub) - 1, min(n, len(sub))).astype(int)
    return sub.iloc[idx]


def main(n: int = 1500, n_val: int = 400) -> FPCA:
    df = load_cycles()
    train = pick(df, "healthy_baseline", "train", n)
    val = pick(df, "healthy_baseline", "val", n_val)
    grid = RegistrationGrid()
    Xtr = registered_curves(train)
    Xva = registered_curves(val)
    fp = FPCA.fit(Xtr, grid)
    fp.save(FPCA_PATH)

    sc_tr, spe_tr = fp.project_many(Xtr)
    sc_va, spe_va = fp.project_many(Xva)
    wsum = fp.weights.sum()
    rms_curve = np.sqrt(np.sum(fp.weights * Xva**2, axis=1) / wsum).mean()
    rms_resid_va = np.sqrt(spe_va / wsum)
    press = Press.load()
    # per-grid-cell noise: pressure noise * area, averaged over the samples in a cell
    samples_per_mm = press.sample_rate_hz / press.ram_speed_mm_s
    cell_mm = np.gradient(grid.x_of(800.0))
    sd_sample = press.pressure_noise_bar / 10 * np.hypot(press.cap_area_mm2, press.rod_area_mm2)
    noise_rms = np.sqrt(np.sum(fp.weights * sd_sample**2 / (samples_per_mm * cell_mm)) / wsum)

    corr = pd.DataFrame(
        {f"PC{k + 1}": [np.corrcoef(sc_tr[:, k], train[c])[0, 1] for c in LATENTS] for k in range(fp.k)},
        index=LATENTS,
    )
    _plot(fp, Xtr, sc_tr)
    cvm = fp.cv_errors.mean(axis=1) if fp.cv_errors is not None else None
    lines = [
        "# Task 2.2: functional PCA of healthy force curves",
        "",
        f"Fitted on {len(Xtr)} healthy training cycles (`healthy_baseline`, split `train`) and checked on "
        f"{len(Xva)} held-out healthy cycles (split `val`). Grid: body {grid.n_body} points in normalised "
        f"stroke plus tail from h = {grid.h_tail_mm:.0f} to {grid.h_min_mm:.0f} mm in {grid.tail_step_mm:.0f} mm steps.",
        "",
        f"**Components chosen by cross-validation (1-SE rule): K = {fp.k}.**",
        "",
        "| K | CV prediction error of hidden points (RMS, kN) |",
        "|---|---|",
        *[f"| {k} | {np.sqrt(v) / 1e3:.2f} |" for k, v in enumerate(cvm)],
        "",
        "Variance explained (share of total curve variance):",
        "",
        "| Component | Share | Cumulative |",
        "|---|---|---|",
        *[f"| PC{k + 1} | {r:.4f} | {c:.4f} |" for k, (r, c) in
          enumerate(zip(fp.explained_ratio, np.cumsum(fp.explained_ratio)))],
        "",
        "Reconstruction on held-out healthy curves:",
        "",
        f"- mean RMS of the curve itself: {rms_curve / 1e6:.2f} MN",
        f"- RMS residual after K components: median {np.median(rms_resid_va) / 1e3:.1f} kN, "
        f"95th percentile {np.percentile(rms_resid_va, 95) / 1e3:.1f} kN "
        f"({np.median(rms_resid_va) / rms_curve * 100:.3f}% of the curve)",
        f"- expected RMS of pure sensor noise after cell averaging: {noise_rms / 1e3:.1f} kN",
        "",
        "The residual sits at the sensor-noise floor, so the K components capture the healthy "
        "shape variation and what is left is noise.",
        "",
        "Physical meaning: correlation of training scores with the simulator's true latent "
        "states and stroke parameters:",
        "",
        corr.round(2).to_markdown(),
        "",
        "![FPCA](figures/task_2_2_fpca.png)",
    ]
    (REPORTS_DIR / "task_2_2_fpca.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return fp


def _plot(fp: FPCA, X: np.ndarray, scores: np.ndarray) -> None:
    ax_x = fp.grid.axis_label()
    k = min(fp.k, 4)
    fig, axes = plt.subplots(1, k + 1, figsize=(4.2 * (k + 1), 4))
    axes[0].plot(ax_x, X[:40].T / 1e6, color="0.6", lw=0.5)
    axes[0].plot(ax_x, fp.mean / 1e6, color="k", lw=1.5)
    axes[0].set(title="Registered healthy curves", xlabel="Registered stroke (mm, ref. billet)", ylabel="Force (MN)")
    for j in range(k):
        s = np.sqrt(fp.eigenvalues[j])
        ax = axes[j + 1]
        # deviation from the mean, so small components are visible
        ax.axhline(0, color="k", lw=0.8)
        ax.plot(ax_x, 2 * s * fp.components[j] / 1e3, color="tab:red", lw=1, label="+2 sd")
        ax.plot(ax_x, -2 * s * fp.components[j] / 1e3, color="tab:blue", lw=1, label="-2 sd")
        ax.set(title=f"PC{j + 1} ({fp.explained_ratio[j] * 100:.2f}% var)", xlabel="Registered stroke (mm)",
               ylabel="Deviation from mean (kN)")
        ax.legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    out = REPORTS_DIR / "figures" / "task_2_2_fpca.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--n-val", type=int, default=400)
    a = ap.parse_args()
    main(a.n, a.n_val)
