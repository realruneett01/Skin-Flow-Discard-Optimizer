"""Batched decision trajectories on the GPU: the same algorithm as ``CutEngine.trajectory``, for many strokes at once.

The CPU engine replays one stroke at a time in Python (~250 ms each). Here all
strokes of a batch advance together:

* **Filter.** In phi the UKF measurement is linear, so the unscented update equals
  the Kalman update exactly. One loop over update index k runs every stroke as
  (N, 3) / (N, 3, 3) tensor operations. Strokes of different length are padded
  and masked.
* **Onset GLR (marginalised).** The running sums ``B, C, c, M, b`` are cumulative
  sums over the gated updates, so they are computed for every update at once with
  ``cumsum``. The effective ``B, C`` need one batched 3x3 solve per (stroke, update).
* **BOCPD.** One loop over gated updates, with run-length arrays for all strokes.
* **Posterior summaries.** Evaluated only at checkpoints after the alarm, in
  chunks, including the same local refinement as ``OnsetGLR.summary``.

Stroke regeneration stays on the CPU (NumPy seeds; GPU random streams could not
reproduce the dataset bit for bit), and so does ``resolve_commit`` (the calibrator
links cycles in time order). Everything runs in float64. ``tests/test_gpu_engine.py``
checks the result against the CPU engine.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from skinflow_discard_optimizer.core.estimator.pipeline import (
    GATE_FRAC,
    GROUPS,
    ONSET_FEATURES,
    PRIOR_FEATURES,
    CutEngine,
    DecisionTrajectory,
    StrokeInputs,
    state_features_from_phi,
)
from skinflow_discard_optimizer.core.observer.onset import BOCPD, H_GRID, LAM_GRID
from skinflow_discard_optimizer.core.observer.ukf import PHI_SCALE, UKF, StrokeContext, default_prior, stroke_blocks


@dataclass
class _Prepared:
    si: StrokeInputs
    L0: float
    xb: np.ndarray
    yb: np.ndarray
    H: np.ndarray        # scaled regressors (n, 3)
    r_var: float
    x0: np.ndarray       # initial scaled phi
    P0: np.ndarray       # initial scaled covariance
    q: np.ndarray        # scaled process noise per mm


def _prepare(engine: CutEngine, si: StrokeInputs) -> _Prepared:
    L0 = engine.press.upset_length(si.billet_length_mm)
    ctx = StrokeContext(L0, si.extrusion_ratio, si.billet_temp_C, engine._alloy(si.alloy_id), engine.press)
    sb = stroke_blocks(si.t_s, si.x_mm, si.p_cap_bar, si.p_rod_bar, ctx, engine.ukf_params, engine.block,
                       None, engine.st.min_cut_mm - 2.0)
    f = UKF(*default_prior(ctx.alloy, engine.press, engine.ukf_params), engine.ukf_params)
    return _Prepared(si, L0, sb.xb, sb.yb, sb.H * PHI_SCALE, sb.r_var, f.x.copy(), f.P.copy(), f.q.copy())


def _pad(arrs: list[np.ndarray], T: int, fill: float = 0.0) -> np.ndarray:
    out = np.full((len(arrs), T) + arrs[0].shape[1:], fill, dtype=float)
    for i, a in enumerate(arrs):
        out[i, : len(a)] = a
    return out


def batch_trajectories(engine: CutEngine, inputs: list[StrokeInputs] | None = None, device=None,
                       **kwargs) -> list[DecisionTrajectory]:
    """``engine.trajectory(si)`` for every input, computed in GPU batches.

    Pass ``prepared`` (from ``prepare_rows``) to skip the CPU preparation here.
    """
    from skinflow_discard_optimizer.accel import get_device

    chunk: int = kwargs.get("chunk", 192)
    prepared: list[_Prepared] | None = kwargs.get("prepared", None)
    dev = device or get_device()
    preps_all = prepared if prepared is not None else [_prepare(engine, si) for si in (inputs or [])]
    out: list[DecisionTrajectory] = []
    for a in range(0, len(preps_all), chunk):
        out.extend(_run_chunk(engine, preps_all[a:a + chunk], dev))
    return out


def batch_features_at(engine: CutEngine, prepared: list[_Prepared], horizons: np.ndarray,
                      device=None, **kwargs) -> list[dict]:
    """``engine.features_at(si, horizon)`` for every stroke: state features plus the onset
    summary using gated data down to that stroke's horizon (training-set builder)."""
    from skinflow_discard_optimizer.accel import get_device

    chunk: int = kwargs.get("chunk", 192)
    dev = device or get_device()
    out: list[dict] = []
    for a in range(0, len(prepared), chunk):
        out.extend(_run_chunk(engine, prepared[a:a + chunk], dev,
                              horizons=np.asarray(horizons[a:a + chunk], float)))
    return out


# --------------------------------------------------------------------------- CPU preparation in parallel

_worker_engine: CutEngine | None = None


def _init_worker() -> None:
    global _worker_engine
    _worker_engine = CutEngine()


def _prepare_row(row: dict) -> _Prepared:
    from skinflow_discard_optimizer.core.estimator.pipeline import inputs_from_row
    from skinflow_discard_optimizer.sim.cycle import regenerate_stroke

    assert _worker_engine is not None
    si = inputs_from_row(row, regenerate_stroke(row, _worker_engine.press))
    p = _prepare(_worker_engine, si)
    # the raw 1 kHz arrays are not needed after preparation; drop them before pickling back
    p.si = StrokeInputs(np.zeros(0), np.zeros(0), np.zeros(0), np.zeros(0), si.billet_length_mm, si.billet_temp_C,
                        si.liner_temp_mean_C, si.extrusion_ratio, si.alloy_id, si.die_id)
    return p


def prepare_rows(rows: list[dict], workers: int | None = None) -> list[_Prepared]:
    """Regenerate and block-prepare dataset rows in parallel CPU workers (order preserved)."""
    from concurrent.futures import ProcessPoolExecutor

    from skinflow_discard_optimizer.paths import default_workers

    with ProcessPoolExecutor(max_workers=workers or default_workers(), initializer=_init_worker) as ex:
        return list(ex.map(_prepare_row, rows, chunksize=16))


# --------------------------------------------------------------------------- one chunk

@dataclass
class _KalmanOutput:
    nu: Any
    S: Any
    x: Any
    P: Any
    h: Any
    frozen: Any
    valid: Any
    xb: Any
    yb: Any
    H: Any
    lens: Any
    r_var: Any


def _run_kalman_batch(preps: list[_Prepared], dev, torch) -> _KalmanOutput:
    f64 = torch.float64
    N = len(preps)
    T = max(len(p.xb) for p in preps)
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=f64, device=dev)  # noqa: E731
    xb = t(_pad([p.xb for p in preps], T))
    yb = t(_pad([p.yb for p in preps], T))
    H = t(_pad([p.H for p in preps], T))
    lens = torch.as_tensor([len(p.xb) for p in preps], device=dev)
    kidx = torch.arange(T, device=dev)
    valid = kidx[None, :] < lens[:, None]
    L0 = t([p.L0 for p in preps])
    h = L0[:, None] - xb
    frozen = h < GATE_FRAC * L0[:, None]
    r_var = t([p.r_var for p in preps])
    q = t(np.stack([p.q for p in preps]))
    x = t(np.stack([p.x0 for p in preps]))
    P = t(np.stack([p.P0 for p in preps]))

    nu = torch.zeros(N, T, dtype=f64, device=dev)
    S = torch.zeros(N, T, dtype=f64, device=dev)
    xprev = xb[:, 0].clone()
    for k in range(T):
        vk = valid[:, k]
        dx = torch.clamp(xb[:, k] - xprev, min=0.0) * vk
        xprev = torch.where(vk, xb[:, k], xprev)
        P = P + torch.diag_embed(q * dx[:, None])
        Hk = H[:, k]
        yp = (Hk * x).sum(-1)
        PH = (P @ Hk[:, :, None]).squeeze(-1)
        Sk = (Hk * PH).sum(-1) + r_var
        nuk = yb[:, k] - yp
        upd = (vk & ~frozen[:, k]).to(f64)
        K = PH / Sk[:, None]
        x = x + upd[:, None] * K * nuk[:, None]
        P = P - upd[:, None, None] * (K[:, :, None] * K[:, None, :]) * Sk[:, None, None]
        P = 0.5 * (P + P.transpose(1, 2))
        nu[:, k], S[:, k] = nuk, Sk

    return _KalmanOutput(nu, S, x, P, h, frozen, valid, xb, yb, H, lens, r_var)


def _compute_glr_sums(g_data: tuple, torch, dev):
    """Compute GLR sums B, C and first firing indices for gated updates."""
    hg, rg, Hg, gval, P, r_var = g_data
    f64 = torch.float64
    lam = torch.as_tensor(LAM_GRID, dtype=f64, device=dev)
    R = r_var[:, None]
    gv = gval.to(f64)
    g = torch.exp(-hg[:, :, None] / lam) * gv[:, :, None]
    rR = (rg * gv) / R
    HR = Hg * gv[:, :, None]
    Bg = torch.cumsum(g * rR[:, :, None], 1)
    Cgg = torch.cumsum(g * g / R[:, :, None], 1)
    cgH = torch.cumsum(g[..., None] * HR[:, :, None, :] / R[:, :, None, None], 1)
    MHH = torch.cumsum(HR[..., :, None] * HR[..., None, :] / R[:, :, None, None], 1)
    bH = torch.cumsum(HR * rR[:, :, None], 1)
    Pinv = torch.linalg.inv(P + 1e-15 * torch.eye(3, dtype=f64, device=dev))
    A = Pinv[:, None] + MHH
    Ainv_b = torch.linalg.solve(A, bH[..., None]).squeeze(-1)
    Ainv_c = torch.linalg.solve(A, cgH.transpose(-1, -2))
    B = Bg - (cgH * Ainv_b[:, :, None, :]).sum(-1)
    C = Cgg - torch.einsum("nglj,ngjl->ngl", cgH, Ainv_c)
    del cgH, Ainv_c
    llr = torch.where(B > 0, B * B / (2 * torch.clamp(C, min=1e-300)), torch.zeros_like(B))
    stat = llr.max(-1).values
    stat = torch.where(gval, stat, torch.zeros_like(stat))
    fired = stat > 12.0
    any_f = fired.any(1)
    first_f = torch.where(any_f, fired.to(torch.int64).argmax(1), torch.full_like(any_f.to(torch.int64), -1)).cpu().numpy()
    return B, C, first_f


def _compute_checkpoints(hg_row: np.ndarray, update_every_mm: float) -> np.ndarray:
    ck, nxt = [], np.inf
    for j, val in enumerate(hg_row):
        if val <= nxt:
            nxt = val - update_every_mm
            ck.append(j)
    return np.array(ck, int)


@dataclass
class _TrajEvalContext:
    first_f: int
    bo_first: int
    summ: dict
    m0s0: tuple[float, float]
    gap: tuple[float, float]
    h_last: float
    feat: dict
    st: Any
    summ_idx: int = 0


def _update_onset_alarm(j: int, ctx: _TrajEvalContext, onset_state: list) -> tuple[tuple[float, float] | None, str]:
    glr_on = ctx.first_f >= 0 and j >= ctx.first_f
    bo_on = ctx.bo_first >= 0 and j >= ctx.bo_first
    onset = onset_state[0]
    if glr_on:
        mean, sd = ctx.summ[(ctx.summ_idx, int(j))]
        agree = bo_on and abs(onset_state[1] - onset_state[2]) <= 12.0
        conf = "high" if agree else "low"
        if sd <= ctx.st.onset_sd_max_mm:
            onset = (mean, sd)
            onset_state[0] = onset
        return onset, conf
    if bo_on:
        return onset, "low"
    return onset, "none"


def _compute_prediction_at_step(onset, m0s0, gap):
    if onset is None:
        return m0s0[0], m0s0[1], np.nan, np.nan
    gm, gvar = gap
    return onset[0] + gm, math.sqrt(gvar + onset[1] ** 2), onset[0], onset[1]


def _eval_checkpoint_step(j: int, ctx: _TrajEvalContext, onset_state: list) -> tuple[float, float, str, float, float]:
    onset, conf = _update_onset_alarm(j, ctx, onset_state)
    m, s, om, osd = _compute_prediction_at_step(onset, ctx.m0s0, ctx.gap)
    return m, s, conf, om, osd


def _build_single_traj(h_row: np.ndarray, ck_list: np.ndarray, ctx: _TrajEvalContext) -> DecisionTrajectory:
    gf = float(h_row[ctx.first_f]) if ctx.first_f >= 0 else np.nan
    bf = float(h_row[ctx.bo_first]) if ctx.bo_first >= 0 else np.nan
    ms, ss, confs, om, osd = [], [], [], [], []
    onset_state = [None, bf, gf]
    for j in ck_list:
        m, s, conf, om_val, osd_val = _eval_checkpoint_step(j, ctx, onset_state)
        ms.append(m)
        ss.append(s)
        confs.append(conf)
        om.append(om_val)
        osd.append(osd_val)

    return DecisionTrajectory(
        h=h_row.astype(float), ck_idx=ck_list, m=np.array(ms), s=np.array(ss), conf=confs,
        onset_mean=np.array(om), onset_sd=np.array(osd), glr_fired_h=gf, bocpd_fired_h=bf,
        m0=ctx.m0s0[0], s0=ctx.m0s0[1], h_last=ctx.h_last, features=ctx.feat,
    )


@dataclass
class _HorizonQuery:
    hg: Any
    g_lists: list
    horizons: np.ndarray
    first_f: np.ndarray


def _eval_horizons_chunk(q: _HorizonQuery, feats: list, BC: tuple, a_ref: float):
    hg_np = q.hg.cpu().numpy()
    N = len(feats)
    k_h = [int(np.sum(hg_np[i, :len(q.g_lists[i])] >= q.horizons[i])) - 1 for i in range(N)]
    pairs = [(i, k) for i, k in enumerate(k_h) if k >= 0]
    summ = _summaries(BC[0], BC[1], pairs, a_ref)
    return [_features_row(feats[i], summ.get((i, k_h[i])), bool(q.first_f[i] >= 0 and q.first_f[i] <= k_h[i]))
            for i in range(N)]


def _build_trajectories_chunk(g_data: tuple, first_f, bo_first, summ, aux: tuple) -> list[DecisionTrajectory]:
    hg_np, g_lists, ck_lists = g_data
    m0s0, gap, h_last, feats, st = aux
    trajs = []
    for i in range(len(feats)):
        n_i = len(g_lists[i])
        ctx = _TrajEvalContext(
            first_f=int(first_f[i]), bo_first=int(bo_first[i]), summ=summ,
            m0s0=m0s0[i], gap=gap[i], h_last=h_last[i], feat=feats[i], st=st,
            summ_idx=i,
        )
        trajs.append(_build_single_traj(hg_np[i, :n_i], ck_lists[i], ctx))
    return trajs


# --------------------------------------------------------------------------- one chunk

def _run_chunk(engine: CutEngine, preps: list[_Prepared], dev=None, horizons: np.ndarray | None = None):
    """Trajectories for a chunk of strokes, or (with ``horizons``) training feature rows."""
    import torch
    dev = dev or preps[0].xb.device if hasattr(preps[0].xb, 'device') else "cpu"
    N = len(preps)
    k_out = _run_kalman_batch(preps, dev, torch)
    gate_mask = (k_out.frozen & k_out.valid).cpu().numpy()
    g_lists = [np.flatnonzero(gate_mask[i]) for i in range(N)]
    G = max((len(g) for g in g_lists), default=0)

    f64 = torch.float64
    t = lambda x: torch.as_tensor(np.asarray(x), dtype=f64, device=dev)  # noqa: E731
    phi = (k_out.x * t(PHI_SCALE)).cpu().numpy()
    P_phys = (k_out.P * t(np.outer(PHI_SCALE, PHI_SCALE))).cpu().numpy()
    feats = [state_features_from_phi(phi[i], P_phys[i], preps[i].si) for i in range(N)]
    m0s0 = _predict_prior(engine, feats)
    gap = _predict_gap(engine, feats)
    h_np = k_out.h.cpu().numpy()
    h_last = [float(h_np[i, len(preps[i].xb) - 1]) for i in range(N)]

    if G == 0:
        if horizons is not None:
            return [_features_row(feats[i], None, False) for i in range(N)]
        return [_empty_traj(m0s0[i], h_last[i], feats[i]) for i in range(N)]

    gidx = t(_pad([g.astype(float) for g in g_lists], G, fill=0.0)).long()
    gval = torch.arange(G, device=dev)[None, :] < torch.as_tensor([len(g) for g in g_lists], device=dev)[:, None]
    take = lambda a: torch.gather(a, 1, gidx)  # noqa: E731
    hg, rg, Sg = take(k_out.h), take(k_out.nu), take(k_out.S)
    Hg = torch.gather(k_out.H, 1, gidx[:, :, None].expand(-1, -1, 3))

    B, C, first_f = _compute_glr_sums((hg, rg, Hg, gval, k_out.P, k_out.r_var), torch, dev)

    if horizons is not None:
        q = _HorizonQuery(hg, g_lists, horizons, first_f)
        return _eval_horizons_chunk(q, feats, (B, C), engine.press.amplitude_at_onset_N)

    bo_first = _bocpd_batch(rg / torch.sqrt(Sg), gval, torch, dev)
    hg_np = hg.cpu().numpy()
    st = engine.st
    ck_lists = [_compute_checkpoints(hg_np[i, :len(g_lists[i])], st.update_every_mm) for i in range(N)]
    pairs = [(i, j) for i in range(N) for j in ck_lists[i] if first_f[i] >= 0 and j >= first_f[i]]
    summ = _summaries(B, C, pairs, engine.press.amplitude_at_onset_N)
    return _build_trajectories_chunk((hg_np, g_lists, ck_lists), first_f, bo_first, summ, (m0s0, gap, h_last, feats, st))


def _features_row(f: dict, summary: tuple[float, float] | None, fired: bool) -> dict:
    """Same keys as ``CutEngine.features_at``. With no gated data the posterior is the flat prior."""
    out = dict(f)
    if summary is None:
        mean = float(H_GRID.mean())
        summary = (mean, float(np.sqrt(np.mean((H_GRID - mean) ** 2))))
    out.update({"onset_mm": summary[0], "onset_sd_mm": summary[1], "glr_fired": fired})
    return out


def _empty_traj(m0s0, h_last, f) -> DecisionTrajectory:
    e = np.zeros(0)
    return DecisionTrajectory(e, np.zeros(0, int), e, e, [], e, e, np.nan, np.nan, m0s0[0], m0s0[1], h_last, f)


def _predict_prior(engine: CutEngine, feats: list[dict]) -> list[tuple[float, float]]:
    if engine.prior_model is None:
        return [(np.nan, np.nan)] * len(feats)
    X = np.array([[f[c] for c in PRIOR_FEATURES] for f in feats])
    groups = {g: np.array([f[g] for f in feats]) for g in GROUPS}
    m, v = engine.prior_model.predict(X, groups)
    return [(float(a), float(np.sqrt(b))) for a, b in zip(m, v)]


def _predict_gap(engine: CutEngine, feats: list[dict]) -> list[tuple[float, float]]:
    if engine.onset_model is None:
        return [(np.nan, np.nan)] * len(feats)
    X = np.array([[f[c] for c in ONSET_FEATURES] for f in feats])
    groups = {g: np.array([f[g] for f in feats]) for g in GROUPS}
    m, v = engine.onset_model.predict(X, groups)
    return [(float(a), float(b)) for a, b in zip(m, v)]


def _bocpd_batch(z, gval, torch, dev) -> np.ndarray:
    """First alarm index of ``BOCPD`` for every stroke (same recursion, vectorised over strokes)."""
    ref = BOCPD()
    f64 = torch.float64
    N, G = z.shape
    W = G + 2
    neg = -math.inf
    log_r = torch.full((N, W), neg, dtype=f64, device=dev)
    log_r[:, 0] = 0.0
    mu = torch.full((N, W), ref.mu0, dtype=f64, device=dev)
    var = torch.full((N, W), ref.var0, dtype=f64, device=dev)
    first = torch.full((N,), -1, dtype=torch.int64, device=dev)
    lh, l1h = math.log(ref.h), math.log1p(-ref.h)
    tcount = torch.zeros(N, dtype=torch.int64, device=dev)
    for k in range(G):
        vk = gval[:, k]
        zk = z[:, k:k + 1]
        pv = var + 1.0
        log_pred = -0.5 * (torch.log(2 * math.pi * pv) + (zk - mu) ** 2 / pv)
        lp = log_r + log_pred
        log_cp = torch.logsumexp(lp + lh, 1, keepdim=True)
        grown = lp + l1h
        new = torch.cat([log_cp, grown[:, :-1]], 1)
        new = new - torch.logsumexp(new, 1, keepdim=True)
        post_var = 1.0 / (1.0 / var + 1.0)
        post_mu = post_var * (mu / var + zk)
        new_mu = torch.cat([torch.full((N, 1), ref.mu0, dtype=f64, device=dev), post_mu[:, :-1]], 1)
        new_var = torch.cat([torch.full((N, 1), ref.var0, dtype=f64, device=dev), post_var[:, :-1]], 1)
        m = vk[:, None]
        log_r = torch.where(m, new, log_r)
        mu = torch.where(m, new_mu, mu)
        var = torch.where(m, new_var, var)
        tcount = tcount + vk.to(torch.int64)
        p_recent = torch.exp(torch.logsumexp(log_r[:, : ref.recent + 1], 1))
        hit = vk & (first < 0) & (tcount > ref.recent * 2) & (p_recent > ref.p_fire)
        first = torch.where(hit, torch.full_like(first, k), first)
    return first.cpu().numpy()


def _refine_narrow_posterior(p, sd, b, c, a_ref: float):
    import torch
    dev = b.device
    f64 = b.dtype
    lam = torch.as_tensor(LAM_GRID, dtype=f64, device=dev)
    hgrid = torch.as_tensor(H_GRID, dtype=f64, device=dev)
    step = float(H_GRID[1] - H_GRID[0])
    mode = hgrid[p.argmax(1)]
    half = torch.clamp(10 * torch.clamp(sd, min=step), min=1.0)
    lo = torch.clamp(mode - half, min=float(H_GRID[0]))
    hi = torch.clamp(mode + half, max=float(H_GRID[-1]))
    u = torch.linspace(0, 1, 801, dtype=f64, device=dev)
    hh = lo[:, None] + (hi - lo)[:, None] * u[None, :]
    a = a_ref * torch.exp(torch.clamp(hh[:, :, None] / lam, max=60.0))
    l2 = torch.logsumexp(a * b[:, None, :] - 0.5 * a * a * c[:, None, :], 2)
    l2 = l2 - torch.logsumexp(l2, 1, keepdim=True)
    p2 = torch.exp(l2)
    m2 = (p2 * hh).sum(1)
    sd2 = torch.sqrt((p2 * (hh - m2[:, None]) ** 2).sum(1))
    return m2, sd2


def _summaries(B, C, pairs: list, a_ref: float) -> dict:
    """Posterior mean and sd of the onset at (stroke, update) pairs, as ``OnsetGLR.summary`` computes them."""
    out: dict = {}
    if not pairs:
        return out
    import torch
    dev = B.device
    f64 = torch.float64
    lam = torch.as_tensor(LAM_GRID, dtype=f64, device=dev)
    hgrid = torch.as_tensor(H_GRID, dtype=f64, device=dev)
    step = float(H_GRID[1] - H_GRID[0])
    a_tab = a_ref * torch.exp(torch.clamp(hgrid[:, None] / lam[None, :], max=60.0))
    sub = 256
    for s0 in range(0, len(pairs), sub):
        pp = pairs[s0:s0 + sub]
        ii = torch.as_tensor([p[0] for p in pp], device=dev)
        jj = torch.as_tensor([p[1] for p in pp], device=dev)
        b = B[ii, jj]
        c = C[ii, jj]
        ll = a_tab[None] * b[:, None, :] - 0.5 * a_tab[None] ** 2 * c[:, None, :]
        lp = torch.logsumexp(ll, 2)
        lp = lp - torch.logsumexp(lp, 1, keepdim=True)
        p = torch.exp(lp)
        mean = (p * hgrid).sum(1)
        sd = torch.sqrt((p * (hgrid - mean[:, None]) ** 2).sum(1))
        need = sd < 5 * step
        if need.any():
            m2, sd2 = _refine_narrow_posterior(p, sd, b, c, a_ref)
            mean = torch.where(need, m2, mean)
            sd = torch.where(need, sd2, sd)
        for (i, j), mv, sv in zip(pp, mean.cpu().numpy(), sd.cpu().numpy()):
            out[(i, int(j))] = (float(mv), float(sv))
    return out
