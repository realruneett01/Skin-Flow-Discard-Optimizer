"""Evaluate the onset detector against the naive second-derivative baseline (Task 3.2).

    python -m skinflow_discard_optimizer.core.observer.eval_onset [--n 150]

Held-out strokes: split ``val`` of the healthy scenario and split ``test`` of the
physical-fault scenarios. The naive baseline's offset is calibrated on healthy
*training* strokes. Writes ``reports/task_3_2_onset.md``,
``reports/figures/task_3_2_onset.png`` and ``artifacts/onset_eval.parquet``.

Truth is the observable onset (``observable_onset``). Errors are reported at two
causal horizons:

* ``at_onset``: data only down to the moment the ram reaches the true onset;
* ``full``: data down to ``h_stop`` = 12 mm (the minimum allowed cut).

Detection delay is in mm of ram travel: ``h_true_onset - h_at_alarm``. Negative
means the alarm came *before* the ram reached the onset.
"""
from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from skinflow_discard_optimizer.core.observer.onset import (  # noqa: E402
    detect_onset, naive_second_derivative, observable_onset,
)
from skinflow_discard_optimizer.core.observer.tune_ukf import context_for  # noqa: E402
from skinflow_discard_optimizer.core.observer.ukf import UKFParams, run_stroke  # noqa: E402
from skinflow_discard_optimizer.paths import REPO_ROOT, REPORTS_DIR, default_workers  # noqa: E402
from skinflow_discard_optimizer.sim.build_dataset import load_cycles  # noqa: E402
from skinflow_discard_optimizer.sim.cycle import regenerate_stroke  # noqa: E402
from skinflow_discard_optimizer.sim.force_model import Press, pressures_to_force  # noqa: E402

GATE_FRAC = 0.15          # gate: x >= 0.85 * L0, i.e. h <= 0.15 * L0
H_STOP = 12.0
PHYSICAL = ["die_wear", "liner_scale", "temperature_drift", "lubricant_loss", "supply_pressure_sag",
            "combined_wear_and_scale", "cold_start_days", "alloy_change", "die_change"]


def _one(args) -> dict:
    row, offset = args
    press = Press.load()
    s = regenerate_stroke(row, press)
    ctx = context_for(row, press)
    hg = GATE_FRAC * ctx.L0_mm
    tr = run_stroke(s.t_s, s.x_mm, s.p_cap_bar, s.p_rod_bar, ctx, UKFParams.for_press(press), block=20,
                    h_stop_mm=H_STOP, theta_every=10**9, freeze_h_mm=hg)
    g = tr.h_mm < hg
    true_on = observable_onset(row["spec_h_onset_mm"], row["spec_lam_mm"], row["spec_upturn_amp_N"],
                               press.amplitude_at_onset_N)
    marg = dict(baseline_cov=tr.final_phi_cov, noise_var=tr.r_var)
    full = detect_onset(tr.h_mm[g], tr.innovation[g], tr.S[g], press.amplitude_at_onset_N, H=tr.H[g], **marg)
    before = g & (tr.h_mm >= true_on)
    at_on = detect_onset(tr.h_mm[before], tr.innovation[before], tr.S[before], press.amplitude_at_onset_N,
                         H=tr.H[before], **marg)
    F = pressures_to_force(s.p_cap_bar, s.p_rod_bar, press)
    n_det, n_est = naive_second_derivative(s.x_mm, F, ctx.L0_mm, hg, H_STOP, offset_mm=offset)
    return {
        "scenario": row["scenario"], "cycle": row["cycle"], "shape": row["shape"],
        "true_onset": true_on, "sim_h_onset": row["spec_h_onset_mm"], "h_crit": row["h_crit_mm"],
        "glr_full_mean": full.h_onset_mean, "glr_full_sd": full.h_onset_sd,
        "glr_full_q05": full.h_onset_q05, "glr_full_q95": full.h_onset_q95,
        "glr_at_onset_mean": at_on.h_onset_mean, "glr_at_onset_sd": at_on.h_onset_sd,
        "glr_at_onset_q05": at_on.h_onset_q05, "glr_at_onset_q95": at_on.h_onset_q95,
        "glr_fired_h": full.glr_fired_at_h, "bocpd_fired_h": full.bocpd_fired_at_h, "confidence": full.confidence,
        "naive_detect_h": n_det, "naive_onset": n_est,
    }


def run(rows: list[dict], offset: float, workers: int | None = None) -> pd.DataFrame:
    workers = workers or default_workers()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return pd.DataFrame(list(ex.map(_one, [(r, offset) for r in rows], chunksize=4)))


def _sample(df, scenarios, split, n, seed=0):
    sub = df[df.scenario.isin(scenarios) & (df.split == split)]
    return sub.sample(n=min(n, len(sub)), random_state=seed).to_dict("records")


def main(n: int = 150, n_cal: int = 60) -> pd.DataFrame:
    df = load_cycles()
    cal = run(_sample(df, ["healthy_baseline"], "train", n_cal), offset=0.0)
    offset = float((cal.naive_detect_h - cal.true_onset).median())
    ev = pd.concat([run(_sample(df, ["healthy_baseline"], "val", n), offset),
                    run(_sample(df, PHYSICAL, "test", n, seed=1), offset)], ignore_index=True)
    ev["group"] = np.where(ev.scenario == "healthy_baseline", "healthy (val)", "physical faults (test)")
    out = REPO_ROOT / "artifacts" / "onset_eval.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    ev.to_parquet(out, index=False)

    def stats(d: pd.DataFrame) -> dict:
        e_glr_on = d.glr_at_onset_mean - d.true_onset
        e_glr_full = d.glr_full_mean - d.true_onset
        e_naive = d.naive_onset - d.true_onset
        return {
            "n": len(d),
            "GLR onset error at onset, RMS (mm)": np.sqrt(np.nanmean(e_glr_on**2)),
            "GLR onset error, full data, RMS (mm)": np.sqrt(np.nanmean(e_glr_full**2)),
            "Naive onset error, full data, RMS (mm)": np.sqrt(np.nanmean(e_naive**2)),
            "Naive: no detection (%)": 100 * d.naive_onset.isna().mean(),
            "GLR 90% interval coverage at onset (%)":
                100 * np.mean((d.true_onset >= d.glr_at_onset_q05) & (d.true_onset <= d.glr_at_onset_q95)),
            "GLR 90% interval coverage, full (%)":
                100 * np.mean((d.true_onset >= d.glr_full_q05) & (d.true_onset <= d.glr_full_q95)),
            "GLR alarm delay, median (mm)": np.nanmedian(d.true_onset - d.glr_fired_h),
            "GLR alarm delay, 5-95% (mm)": f"{np.nanpercentile(d.true_onset - d.glr_fired_h, 5):.1f} to "
                                           f"{np.nanpercentile(d.true_onset - d.glr_fired_h, 95):.1f}",
            "Naive alarm delay, median (mm)": np.nanmedian(d.true_onset - d.naive_detect_h),
            "GLR no alarm (%)": 100 * d.glr_fired_h.isna().mean(),
            "Confidence high (%)": 100 * (d.confidence == "high").mean(),
        }

    table = pd.DataFrame({g: stats(d) for g, d in ev.groupby("group")})
    beats = all(table.loc["GLR onset error, full data, RMS (mm)"] < table.loc["Naive onset error, full data, RMS (mm)"])
    lines = [
        "# Task 3.2: skin-flow onset detection",
        "",
        f"Evaluated on {len(ev)} held-out strokes. The naive baseline's offset "
        f"({offset:.1f} mm, detection minus true onset) was calibrated on {len(cal)} healthy training strokes.",
        "",
        "Truth is the *observable* onset: the thickness where the end-of-stroke force change reaches the "
        "reference amplitude (docs/assumptions.md A-19). Delay is in mm of ram travel, "
        "`true onset - h at alarm`; negative means the alarm came before the ram reached the onset.",
        "",
        table.to_markdown(floatfmt=".2f"),
        "",
        f"**Done-when: GLR beats the naive baseline on onset-position error in every group: {'yes' if beats else 'NO'}.**",
        "",
        "**Posterior calibration, honestly:**",
        "",
        "- At the decision horizon (data until the ram reaches the onset) the 90% interval covers close to "
        "90% on healthy strokes and somewhat less under physical faults, where the frozen baseline is "
        "slightly off (e.g. a colder, more tapered billet than the nominal taper assumed).",
        "- With data down to 12 mm the posterior is badly overconfident. Once the upturn is several MN, "
        "tiny model errors (0.01 mm of position, 0.1% of flow stress) outweigh the noise model. That "
        "posterior is diagnostic only: by then the cut has already been made, so no decision uses it. "
        "The conformal step in Task 3.3 recalibrates what the decision does use.",
        "",
        "Why the naive rule struggles: the second derivative of force only clears its noise band once "
        "the upturn is steep, which is after the onset and near the cost-optimal cut. A decision made then "
        "is already too late. The GLR works on the frozen-baseline residuals and models the upturn's shape, "
        "so it sees the rise while it is still tens of kN and can extrapolate to the onset before the ram "
        "gets there.",
        "",
        "![onset](figures/task_3_2_onset.png)",
    ]
    (REPORTS_DIR / "task_3_2_onset.md").write_text("\n".join(lines), encoding="utf-8")
    _plot(ev)
    print("\n".join(lines))
    return ev


def _plot(ev: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].scatter(ev.true_onset, ev.glr_at_onset_mean, s=8, label="GLR (data until onset)")
    ax[0].scatter(ev.true_onset, ev.naive_onset, s=8, marker="x", label="naive 2nd derivative (all data)")
    lim = [ev.true_onset.min() - 3, ev.true_onset.max() + 3]
    ax[0].plot(lim, lim, "k--", lw=0.8)
    ax[0].set(xlabel="True onset h (mm)", ylabel="Estimated onset h (mm)", title="Onset position")
    ax[0].legend(fontsize=8)
    ax[1].hist(ev.true_onset - ev.glr_fired_h, bins=30, alpha=0.7, label="GLR alarm")
    ax[1].hist(ev.true_onset - ev.naive_detect_h, bins=30, alpha=0.7, label="naive alarm")
    ax[1].axvline(0, color="k", lw=0.8)
    ax[1].set(xlabel="Delay (mm of ram travel; <0 = before onset)", title="Alarm timing")
    ax[1].legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "figures" / "task_3_2_onset.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    main(a.n)
