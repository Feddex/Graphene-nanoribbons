# src/spindft_tb_fit_core_nnn.py
import numpy as np
from zgnr_mf_nnn import solve_zgnr_mf_nnn

def load_dft_central_bands(bands_pi_npz: str):
    data = np.load(bands_pi_npz, allow_pickle=True)
    k_plot = np.asarray(data["k_plot"], float).ravel()
    E_rel = np.asarray(data["E_rel"], float)
    bands_up = np.asarray(data["bands_up"], int)

    # choose central val/cond in the pi window for spin up
    E = E_rel[0]  # (Nk, nb)
    E_mean = E[:, bands_up].mean(axis=0)
    mask_val = (E_mean < 0.0)
    if np.any(mask_val):
        val_band = bands_up[mask_val][np.argmin(np.abs(E_mean[mask_val]))]
    else:
        val_band = bands_up[np.argmin(np.abs(E_mean))]
    mask_con = (E_mean > 0.0)
    if np.any(mask_con):
        con_band = bands_up[mask_con][np.argmin(np.abs(E_mean[mask_con]))]
    else:
        order = np.argsort(np.abs(E_mean))
        con_band = bands_up[order[0]] if len(order) == 1 else bands_up[order[1]]

    return dict(
        k_plot=k_plot,
        E_val=E[:, val_band],
        E_con=E[:, con_band],
    )

def compute_tb_central_bands_nnn(
    *,
    Ny, M,
    C_C, VACUUM, SATURATED,
    t1, t2, U,
    Nk_tb,
    k_plot_target,
    filling, max_iter, mix, tol,
):
    k_plot_target = np.asarray(k_plot_target, float).ravel()

    res = solve_zgnr_mf_nnn(
        Ny=Ny, U=U, t1=t1, t2=t2,
        C_C=C_C, vacuum=VACUUM, saturated=SATURATED, M=M,
        Nk=Nk_tb, filling=filling,
        max_iter=max_iter, mix=mix, tol=tol,
        m0=0.05, verbose=False,
    )

    k = np.asarray(res["k_grid"], float)
    mu = float(res["mu"])
    Eavg = 0.5 * (np.asarray(res["E"][0]) + np.asarray(res["E"][1]))  # (Nk, nb)

    # take k>=0 half (Gamma->Z)
    mask = (k >= 0.0)
    k_pos = k[mask]
    # normalize to [0,1]
    k_plot_tb = (k_pos - k_pos.min()) / (k_pos.max() - k_pos.min() + 1e-14)

    Erel = Eavg - mu
    # central bands by closest to 0 below/above
    val = np.full(mask.sum(), np.nan)
    con = np.full(mask.sum(), np.nan)
    for i, Ek in enumerate(Erel[mask]):
        below = np.where(Ek <= 0.0)[0]
        above = np.where(Ek >  0.0)[0]
        if below.size:
            val[i] = Ek[below[np.argmax(Ek[below])]]
        if above.size:
            con[i] = Ek[above[np.argmin(Ek[above])]]

    val_i = np.interp(k_plot_target, k_plot_tb, val)
    con_i = np.interp(k_plot_target, k_plot_tb, con)

    return dict(val=val_i, con=con_i, result=res)

def band_cost(params, dft, *, Ny, M, C_C, VACUUM, SATURATED, Nk_tb, filling, max_iter, mix, tol):
    t1, t2, U = float(params[0]), float(params[1]), float(params[2])
    if t1 <= 0.0 or U <= 0.0:
        return 1e9

    tb = compute_tb_central_bands_nnn(
        Ny=Ny, M=M,
        C_C=C_C, VACUUM=VACUUM, SATURATED=SATURATED,
        t1=t1, t2=t2, U=U,
        Nk_tb=Nk_tb, k_plot_target=dft["k_plot"],
        filling=filling, max_iter=max_iter, mix=mix, tol=tol,
    )
    return float(np.mean((tb["val"] - dft["E_val"])**2) + np.mean((tb["con"] - dft["E_con"])**2))

def save_tbfit_npz_nnn(out_npz, *, Ny, M, t1, t2, U, cost, k_plot, dft_val, dft_con, tb_val, tb_con, m_strand_dft, m_strand_tb):
    np.savez(
        out_npz,
        Ny=int(Ny), M=int(M),
        t_fit=float(t1),
        t2_fit=float(t2),
        U_fit=float(U),
        cost=float(cost),
        k_plot=np.asarray(k_plot, float),
        E_val_dft=np.asarray(dft_val, float),
        E_con_dft=np.asarray(dft_con, float),
        E_val_tb=np.asarray(tb_val, float),
        E_con_tb=np.asarray(tb_con, float),
        m_strand_dft=np.asarray(m_strand_dft, float),
        m_strand_tb=np.asarray(m_strand_tb, float),
    )
