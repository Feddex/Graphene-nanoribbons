# run_mf_nnn.py
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
INPUT_DIR = os.path.join(PROJECT_ROOT, "inputs")
for p in (SRC_DIR, INPUT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from inputs_mf_nnn import (
    MODE,
    N_SINGLE, U_SINGLE,
    N_MIN, N_MAX, U_FIXED_FOR_SWEEP_N,
    N_FIXED_FOR_SWEEP_U, Umin_over_t1, Umax_over_t1, dU_over_t1,
    C_C, VACUUM, SATURATED, M,
    t1, t2,
    Nk, filling, max_iter, mix, tol, m0, verbose,
    dos_nE, dos_eta,
    base_dir,
)
from zgnr_mf_nnn import run_single_case, run_sweep_N, run_sweep_U

def main():
    base_dir_abs = os.path.join(PROJECT_ROOT, base_dir)

    if MODE == "single":
        out = run_single_case(
            N=N_SINGLE, U=U_SINGLE,
            C_C=C_C, VACUUM=VACUUM, SATURATED=SATURATED, M=M,
            t1=t1, t2=t2,
            Nk=Nk, filling=filling,
            max_iter=max_iter, mix=mix, tol=tol, m0=m0,
            dos_nE=dos_nE, dos_eta=dos_eta,
            base_dir=base_dir_abs, mode_tag="single",
            copy_to_all=True, verbose=verbose,
        )
        print("\nDONE single.")
        print("Saved:", out)

    elif MODE == "sweep_N":
        run_sweep_N(
            N_min=N_MIN, N_max=N_MAX,
            U=U_FIXED_FOR_SWEEP_N,
            C_C=C_C, VACUUM=VACUUM, SATURATED=SATURATED, M=M,
            t1=t1, t2=t2,
            a_dummy=1.0,
            Nk=Nk, filling=filling,
            max_iter=max_iter, mix=mix, tol=tol, m0=m0,
            dos_nE=dos_nE, dos_eta=dos_eta,
            base_dir=base_dir_abs, mode_tag="sweep_N",
            verbose=verbose,
        )

    elif MODE == "sweep_U":
        run_sweep_U(
            N_fixed=N_FIXED_FOR_SWEEP_U,
            Umin_over_t1=Umin_over_t1, Umax_over_t1=Umax_over_t1, dU_over_t1=dU_over_t1,
            C_C=C_C, VACUUM=VACUUM, SATURATED=SATURATED, M=M,
            t1=t1, t2=t2,
            Nk=Nk, filling=filling,
            max_iter=max_iter, mix=mix, tol=tol, m0=m0,
            dos_nE=dos_nE, dos_eta=dos_eta,
            base_dir=base_dir_abs, mode_tag="sweep_U",
            verbose=verbose,
        )
    else:
        raise ValueError("MODE must be 'single', 'sweep_N', or 'sweep_U'.")

if __name__ == "__main__":
    main()
