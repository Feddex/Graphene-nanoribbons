# run_spindft_tbfit_nnn.py
import os
import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
INPUTS_DIR = os.path.join(PROJECT_ROOT, "inputs")
for p in (SRC_DIR, INPUTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from inputs_spindft import MODE, NY_SINGLE, NY_LIST, M, C_C, VACUUM, SATURATED, BASE_OUTDIR
from inputs_tbfit_nnn import (
    NK_TB_FACTOR, FILLING, MAX_ITER, MIX, TOL,
    USE_SCIPY_IF_AVAILABLE, T1_0, T2_0, U0,
    T1_GRID, T2_GRID, U_GRID
)
from spindft_paths import case_dir, ensure_dirs, relabel_cfg_files
from spindft_tb_fit_core_nnn import (
    load_dft_central_bands,
    compute_tb_central_bands_nnn,
    band_cost,
    save_tbfit_npz_nnn,
)

def _try_minimize():
    try:
        from scipy.optimize import minimize
        return minimize
    except Exception:
        return None

def fit_one_case(mode: str, Ny: int):
    case = case_dir(BASE_OUTDIR, mode, Ny)
    dirs = ensure_dirs(case)
    files = relabel_cfg_files(dirs, Ny=Ny, M=M)

    if not os.path.isfile(files["bands_pi_npz"]):
        print("[SKIP] missing bands_pi:", files["bands_pi_npz"])
        return
    if not os.path.isfile(files["mag_AB_npz"]):
        print("[SKIP] missing mag_AB:", files["mag_AB_npz"])
        return

    dft = load_dft_central_bands(files["bands_pi_npz"])
    Nk_dft = len(dft["k_plot"])
    Nk_tb = int(NK_TB_FACTOR) * Nk_dft

    x0 = np.array([T1_0, T2_0, U0], float)
    minimize = _try_minimize() if USE_SCIPY_IF_AVAILABLE else None

    if minimize is not None:
        res = minimize(
            lambda x: band_cost(x, dft, Ny=Ny, M=M, C_C=C_C, VACUUM=VACUUM, SATURATED=SATURATED,
                                Nk_tb=Nk_tb, filling=FILLING, max_iter=MAX_ITER, mix=MIX, tol=TOL),
            x0,
            method="Nelder-Mead",
            options={"maxiter": 120, "disp": True},
        )
        t1_opt, t2_opt, U_opt = res.x
        cost = float(res.fun)
    else:
        best = 1e99
        t1_opt, t2_opt, U_opt = x0
        for t1 in T1_GRID:
            for t2 in T2_GRID:
                for U in U_GRID:
                    c = band_cost((t1, t2, U), dft, Ny=Ny, M=M, C_C=C_C, VACUUM=VACUUM, SATURATED=SATURATED,
                                  Nk_tb=Nk_tb, filling=FILLING, max_iter=MAX_ITER, mix=MIX, tol=TOL)
                    if c < best:
                        best = c
                        t1_opt, t2_opt, U_opt = float(t1), float(t2), float(U)
        cost = float(best)

    tb = compute_tb_central_bands_nnn(
        Ny=Ny, M=M,
        C_C=C_C, VACUUM=VACUUM, SATURATED=SATURATED,
        t1=t1_opt, t2=t2_opt, U=U_opt,
        Nk_tb=Nk_tb, k_plot_target=dft["k_plot"],
        filling=FILLING, max_iter=MAX_ITER, mix=MIX, tol=TOL,
    )

    # Use DFT strand magnetization proxy from mag_AB_npz
    mag = np.load(files["mag_AB_npz"], allow_pickle=True)
    mA = np.asarray(mag["mA"], float).ravel()
    mB = np.asarray(mag["mB"], float).ravel()
    m_strand_dft = 0.5 * (mA + mB)

    m_strand_tb = np.asarray(tb["result"]["m_strand"], float).ravel()

    out_npz = os.path.join(dirs["tbfit"], f"zgnr_Ny{Ny}_M{M}_tb_fit_nnn.npz")
    save_tbfit_npz_nnn(
        out_npz,
        Ny=Ny, M=M,
        t1=t1_opt, t2=t2_opt, U=U_opt,
        cost=cost,
        k_plot=dft["k_plot"],
        dft_val=dft["E_val"], dft_con=dft["E_con"],
        tb_val=tb["val"], tb_con=tb["con"],
        m_strand_dft=m_strand_dft,
        m_strand_tb=m_strand_tb,
    )

    print(f"[OK] TB+NNN fit Ny={Ny}: t1={t1_opt:.4f} t2={t2_opt:.4f} U={U_opt:.4f} cost={cost:.3e}")

def main():
    if MODE == "single":
        fit_one_case("single", NY_SINGLE)
    elif MODE == "sweep_N":
        for Ny in NY_LIST:
            fit_one_case("sweep_N", Ny)
    else:
        raise ValueError("MODE must be 'single' or 'sweep_N'.")

if __name__ == "__main__":
    main()
