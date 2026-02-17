# run_non_interacting_nnn.py
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
INPUT_DIR = os.path.join(PROJECT_ROOT, "inputs")
for p in (SRC_DIR, INPUT_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from inputs_non_interacting_nnn import (
    MODE, N_SINGLE, N_MIN, N_MAX,
    C_C, VACUUM, SATURATED, M,
    t1, t2,
    nk, eta, n_E, mu,
    base_dir,
)

from zgnr_noninteracting_nnn import run_single_case, run_sweep_N

def main():
    base_dir_abs = os.path.join(PROJECT_ROOT, base_dir)

    if MODE == "single":
        out = run_single_case(
            N=N_SINGLE,
            C_C=C_C, VACUUM=VACUUM, SATURATED=SATURATED, M=M,
            t1=t1, t2=t2,
            nk=nk, eta=eta, n_E=n_E, mu=mu,
            base_dir=base_dir_abs,
            copy_to_all=True,
        )
        print("\nDONE single.")
        print("Saved:", out)

    elif MODE == "sweep_N":
        run_sweep_N(
            N_min=N_MIN, N_max=N_MAX,
            C_C=C_C, VACUUM=VACUUM, SATURATED=SATURATED, M=M,
            t1=t1, t2=t2,
            nk=nk, eta=eta, n_E=n_E, mu=mu,
            base_dir=base_dir_abs,
        )
        print("\nDONE sweep_N.")
        print("Saved under:", base_dir_abs)
    else:
        raise ValueError("MODE must be 'single' or 'sweep_N'.")

if __name__ == "__main__":
    main()
