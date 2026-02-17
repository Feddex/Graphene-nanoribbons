# src/zgnr_noninteracting_nnn.py
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from ase.build import graphene_nanoribbon

# ----------------------------
# Geometry + neighbor graph
# ----------------------------
def _build_atoms(Ny, M, C_C, vacuum, saturated):
    atoms = graphene_nanoribbon(
        Ny, M, type="zigzag",
        saturated=saturated,
        C_C=C_C, vacuum=vacuum
    )
    atoms.set_pbc((False, False, True))
    return atoms

def _pair_list_with_shifts(atoms, r0, tol, max_shift=1):
    """
    Return list of (i, j, m) meaning coupling between i in cell and j in cell shifted by m*Lz.
    """
    pos = atoms.get_positions()
    cell = atoms.get_cell()
    Lz = float(cell[2, 2])
    pairs = []
    for m in range(-max_shift, max_shift + 1):
        shift = np.array([0.0, 0.0, m * Lz])
        for i in range(len(atoms)):
            for j in range(len(atoms)):
                if m == 0 and j == i:
                    continue
                d = np.linalg.norm((pos[j] + shift) - pos[i])
                if abs(d - r0) < tol:
                    pairs.append((i, j, m))
    return pairs, Lz

def _bloch_hk(atoms, k, pairs_nn, pairs_nnn, t1, t2, Lz):
    dim = len(atoms)
    H = np.zeros((dim, dim), dtype=complex)
    for (i, j, m) in pairs_nn:
        phase = np.exp(1j * k * (m * Lz))
        H[i, j] += t1 * phase
    for (i, j, m) in pairs_nnn:
        phase = np.exp(1j * k * (m * Lz))
        H[i, j] += t2 * phase
    # Hermitize (numerical safety)
    H = 0.5 * (H + H.conj().T)
    return H

# ----------------------------
# Bands + DOS
# ----------------------------
def bands_zgnr_nnn(Ny, *, M, C_C, vacuum, saturated, t1, t2, nk):
    atoms = _build_atoms(Ny, M, C_C, vacuum, saturated)

    # distances: NN = C_C, NNN = sqrt(3)*C_C (graphene)
    pairs_nn, Lz = _pair_list_with_shifts(atoms, r0=C_C, tol=0.08, max_shift=1)
    pairs_nnn, _  = _pair_list_with_shifts(atoms, r0=np.sqrt(3)*C_C, tol=0.10, max_shift=1)

    ks = np.linspace(-np.pi / Lz, np.pi / Lz, int(nk), endpoint=False)
    nb = len(atoms)
    E = np.zeros((len(ks), nb), float)

    for ik, k in enumerate(ks):
        Hk = _bloch_hk(atoms, k, pairs_nn, pairs_nnn, t1, t2, Lz)
        w = np.linalg.eigvalsh(Hk)
        E[ik] = np.sort(w.real)

    return ks, E, Lz

def dos_from_bands(E, n_E=1000, eta=0.05):
    eps = E.ravel()
    Emin, Emax = float(eps.min()), float(eps.max())
    pad = 0.2 * (Emax - Emin) if Emax > Emin else 1.0
    Emin -= pad
    Emax += pad
    Egrid = np.linspace(Emin, Emax, int(n_E))
    x = Egrid[:, None] - eps[None, :]
    gauss = np.exp(-0.5 * (x / eta) ** 2) / (np.sqrt(2 * np.pi) * eta)
    DOS = gauss.sum(axis=1) / E.shape[0]
    return Egrid, DOS

def extract_central_bands(E, Ny):
    # central around half filling: middle two bands (sorted)
    nb = E.shape[1]
    i0 = nb // 2 - 1
    i1 = nb // 2
    return E[:, i0], E[:, i1]

# ----------------------------
# Plot + outputs (mirrors your style)
# ----------------------------
def _add_corner_label(ax, text):
    ax.text(
        0.97, 0.97, text,
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85),
    )

def _make_output_dirs(base_dir, N, t1, t2):
    run_dir = os.path.join(base_dir, f"N{N}", f"t1{t1:.4f}", f"t2{t2:.4f}")
    dirs = {
        "run": run_dir,
        "bands": os.path.join(run_dir, "bands"),
        "dos": os.path.join(run_dir, "dos"),
        "central": os.path.join(run_dir, "central_bands"),
        "data": os.path.join(run_dir, "data"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def _make_all_dirs(base_dir):
    all_dirs = {
        "bands": os.path.join(base_dir, "_ALL", "bands"),
        "dos": os.path.join(base_dir, "_ALL", "dos"),
        "central": os.path.join(base_dir, "_ALL", "central_bands"),
    }
    for d in all_dirs.values():
        os.makedirs(d, exist_ok=True)
    return all_dirs

def _plot_bands(ks, E, mu, label, out_png):
    plt.figure(figsize=(7.2, 4.6))
    for b in range(E.shape[1]):
        plt.plot(ks, E[:, b], lw=0.9, alpha=0.85)
    plt.axhline(mu, ls="--", lw=1.2, alpha=0.7)
    plt.xlabel("k")
    plt.ylabel("Energy (eV)")
    plt.grid(True, alpha=0.25)
    _add_corner_label(plt.gca(), label)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def _plot_central(ks, val, con, mu, label, out_png):
    plt.figure(figsize=(7.2, 4.6))
    plt.plot(ks, val, lw=2.0)
    plt.plot(ks, con, lw=2.0)
    plt.axhline(mu, ls="--", lw=1.2, alpha=0.7)
    plt.xlabel("k")
    plt.ylabel("Energy (eV)")
    plt.grid(True, alpha=0.25)
    _add_corner_label(plt.gca(), label)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def _plot_dos(Egrid, DOS, mu, label, out_png):
    plt.figure(figsize=(6.2, 4.6))
    plt.plot(Egrid, DOS, lw=1.6)
    plt.axvline(mu, ls="--", lw=1.2, alpha=0.7)
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS (arb.)")
    plt.grid(True, alpha=0.25)
    _add_corner_label(plt.gca(), label)
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def run_single_case(
    *,
    N,
    C_C, VACUUM, SATURATED, M,
    t1, t2,
    nk, eta, n_E, mu,
    base_dir,
    copy_to_all=True,
):
    ks, E, Lz = bands_zgnr_nnn(N, M=M, C_C=C_C, vacuum=VACUUM, saturated=SATURATED, t1=t1, t2=t2, nk=nk)
    Egrid, DOS = dos_from_bands(E, n_E=n_E, eta=eta)
    val, con = extract_central_bands(E, N)

    dirs = _make_output_dirs(base_dir, N, t1, t2)
    all_dirs = _make_all_dirs(base_dir) if copy_to_all else None

    label = f"$N={N}$\n$t_1={t1:.2f}$ eV\n$t_2={t2:.2f}$ eV"
    bands_png = os.path.join(dirs["bands"], "bands.png")
    central_png = os.path.join(dirs["central"], "central_bands.png")
    dos_png = os.path.join(dirs["dos"], "dos.png")

    _plot_bands(ks, E, mu, label, bands_png)
    _plot_central(ks, val, con, mu, label, central_png)
    _plot_dos(Egrid, DOS, mu, label, dos_png)

    np.savez(
        os.path.join(dirs["data"], "run_summary.npz"),
        N=int(N), t1=float(t1), t2=float(t2),
        C_C=float(C_C), VACUUM=float(VACUUM), SATURATED=bool(SATURATED), M=int(M),
        Lz=float(Lz),
        nk=int(nk), eta=float(eta), n_E=int(n_E), mu=float(mu),
        k_grid=ks, E=E,
        Egrid=Egrid, dos=DOS,
        valence_central=val,
        conduction_central=con,
    )

    if copy_to_all:
        shutil.copy2(bands_png, os.path.join(all_dirs["bands"], f"bands_N{N:02d}.png"))
        shutil.copy2(dos_png, os.path.join(all_dirs["dos"], f"dos_N{N:02d}.png"))
        shutil.copy2(central_png, os.path.join(all_dirs["central"], f"central_N{N:02d}.png"))

    return dirs["run"]

def run_sweep_N(
    *,
    N_min, N_max,
    C_C, VACUUM, SATURATED, M,
    t1, t2,
    nk, eta, n_E, mu,
    base_dir,
):
    _make_all_dirs(base_dir)
    for N in range(int(N_min), int(N_max) + 1):
        print(f"\nRunning TB+NNN: N={N}")
        out = run_single_case(
            N=N,
            C_C=C_C, VACUUM=VACUUM, SATURATED=SATURATED, M=M,
            t1=t1, t2=t2,
            nk=nk, eta=eta, n_E=n_E, mu=mu,
            base_dir=base_dir,
            copy_to_all=True,
        )
        print("Saved:", out)
