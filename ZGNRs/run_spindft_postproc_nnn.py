# run_spindft_postproc_nnn.py
import os
import sys
import glob

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
INPUTS_DIR = os.path.join(PROJECT_ROOT, "inputs")
for p in (SRC_DIR, INPUTS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from inputs_spindft import MODE, NY_SINGLE, NY_LIST, BASE_OUTDIR
from spindft_paths import case_dir, ensure_dirs
from spindft_pretty_plots_core import pretty_bands_from_npz  # keep your existing DFT pretty bands
from spindft_pretty_plots_core_nnn import pretty_tbfit_nnn_bands, pretty_tbfit_nnn_magnetization

def case_list():
    return [NY_SINGLE] if MODE == "single" else list(NY_LIST)

def _find_one(patterns):
    for pat in patterns:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None

def main():
    for Ny in case_list():
        case = case_dir(BASE_OUTDIR, MODE, Ny)
        dirs = ensure_dirs(case)

        bands_npz = _find_one([os.path.join(dirs["data"], "*_bands_pi.npz")])
        tbfit_nnn = _find_one([os.path.join(dirs["tbfit"], "*_tb_fit_nnn.npz"),
                               os.path.join(dirs["tbfit"], "*_tbfit_nnn.npz")])

        pretty_dir = os.path.join(dirs["plots"], "pretty")
        os.makedirs(pretty_dir, exist_ok=True)

        if bands_npz:
            out_col = os.path.join(pretty_dir, "bands_colored.png")
            out_gra = os.path.join(pretty_dir, "bands_gray.png")
            pretty_bands_from_npz(bands_npz, out_col, out_gra, Ny_label=Ny)
            print("[OK] DFT bands ->", out_col, out_gra)

        if tbfit_nnn:
            out_b = os.path.join(pretty_dir, "tbfit_nnn_bands.png")
            out_m = os.path.join(pretty_dir, "tbfit_nnn_magnetization.png")
            pretty_tbfit_nnn_bands(tbfit_nnn, out_b, Ny_label=Ny)
            pretty_tbfit_nnn_magnetization(tbfit_nnn, out_m, Ny_label=Ny)
            print("[OK] NNN fit plots ->", out_b, out_m)
        else:
            print("[SKIP] missing tbfit NNN npz in:", dirs["tbfit"])

if __name__ == "__main__":
    main()
