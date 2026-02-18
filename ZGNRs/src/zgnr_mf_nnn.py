# src/zgnr_mf_nnn.py
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from ase.build import graphene_nanoribbon

# ----------------------------
# Geometry & neighbor graph
# ----------------------------
def _build_atoms(Ny, M, C_C, vacuum, saturated):
    atoms = graphene_nanoribbon(
        Ny, M, type="zigzag",
        saturated=saturated,
        C_C=C_C, vacuum=vacuum,
    )
    atoms.set_pbc((False, False, True))
    return atoms

def _pair_list_with_shifts(atoms, r0, tol, max_shift=1):
    pos = atoms.get_positions()
    cell = atoms.get_cell()
    Lz = float(cell[2, 2])
    pairs = []
    for m in range(-max_shift, max_shift + 1):
        shift = np.array([0.0, 0.0, m * Lz])
        for i in range(len(atoms)):
            for j in range(len(atoms)):
                if m == 0 and i == j:
                    continue
                d = np.linalg.norm((pos[j] + shift) - pos[i])
                if abs(d - r0) < tol:
                    pairs.append((i, j, m))
    return pairs, Lz

def _bloch_h0(atoms, k, pairs_nn, pairs_nnn, t1, t2, Lz):
    dim = len(atoms)
    H = np.zeros((dim, dim), dtype=complex)
    for (i, j, m) in pairs_nn:
        H[i, j] += t1 * np.exp(1j * k * (m * Lz))
    for (i, j, m) in pairs_nnn:
        H[i, j] += t2 * np.exp(1j * k * (m * Lz))
    return 0.5 * (H + H.conj().T)

# ----------------------------
# MF onsite potential
# ----------------------------
def _build_V_sigma(dim, sigma, U, m):
    # m is per-site (dim,)
    return np.diag(-U * sigma * m.astype(float))

# ----------------------------
# Solver with spin resolved
# ----------------------------
def solve_zgnr_mf_nnn(
    *,
    Ny,
    U,
    t1,
    t2,
    C_C,
    vacuum,
    saturated,
    M,
    Nk,
    filling,
    max_iter,
    mix,
    tol,
    m0,
    verbose=False,
):
    atoms = _build_atoms(Ny, M, C_C, vacuum, saturated)
    dim = len(atoms)

    pairs_nn, Lz = _pair_list_with_shifts(atoms, r0=C_C, tol=0.08, max_shift=1)
    pairs_nnn, _  = _pair_list_with_shifts(atoms, r0=np.sqrt(3)*C_C, tol=0.10, max_shift=1)

    k_grid = np.linspace(-np.pi / Lz, np.pi / Lz, int(Nk), endpoint=False)

    nelec_per_cell = filling * dim
    nelec_total = int(round(nelec_per_cell * Nk))

    # initial staggered seed on edges
    m = np.zeros(dim, float)
    m[0] = +m0
    m[-1] = -m0

    E = np.zeros((2, Nk, dim), float)
    Uvec = np.zeros((2, Nk, dim, dim), complex)
    mu = 0.0

    for it in range(int(max_iter)):
        # diag
        for s_idx, sigma in enumerate([+1, -1]):
            V = _build_V_sigma(dim, sigma, U, m)
            for ik, k in enumerate(k_grid):
                H0 = _bloch_h0(atoms, k, pairs_nn, pairs_nnn, t1, t2, Lz)
                H = H0 + V
                vals, vecs = np.linalg.eigh(H)
                E[s_idx, ik] = vals.real
                Uvec[s_idx, ik] = vecs

        # chemical potential by filling
        E_flat = E.reshape(-1)
        idx = np.argsort(E_flat)
        if nelec_total >= len(E_flat):
            mu = float(E_flat[idx[-1]] + 1e-6)
        else:
            eN1 = float(E_flat[idx[nelec_total - 1]])
            eN  = float(E_flat[idx[nelec_total]])
            mu = 0.5 * (eN1 + eN)

        # densities -> magnetization per site
        n_up = np.zeros(dim, float)
        n_dn = np.zeros(dim, float)
        w_k = 1.0 / Nk

        for s_idx, sigma in enumerate([+1, -1]):
            for ik in range(Nk):
                vals = E[s_idx, ik]
                vecs = Uvec[s_idx, ik]
                occ = np.where(vals <= mu)[0]
                # accumulate occupations
                for b in occ:
                    v = vecs[:, b]
                    prob = np.abs(v) ** 2
                    if sigma == +1:
                        n_up += w_k * prob
                    else:
                        n_dn += w_k * prob

        m_new = 0.5 * (n_up - n_dn)
        m_mixed = (1.0 - mix) * m + mix * m_new
        delta = float(np.max(np.abs(m_mixed - m)))
        m = m_mixed

        if verbose:
            print(f"Iter {it+1:3d}: mu={mu:+.6f} max|Δm|={delta:.3e}")

        if delta < tol:
            break

    # For compatibility with your plotting expectations, we expose mA/mB-like arrays:
    # Here we don’t have explicit A/B indexing from ASE; but the MF lattice postproc expects mA/mB per strand.
    # Here is provided a strand-averaged profile by grouping atoms by transverse coordinate.
    pos = atoms.get_positions()
    x = pos[:, 0]
    # group into Ny strands by sorting x and binning
    order = np.argsort(x)
    groups = np.array_split(order, Ny)
    m_strand = np.array([m[g].mean() for g in groups], float)

    return dict(
        k_grid=k_grid,
        E=E,
        Uvec=Uvec,
        mu=float(mu),
        m_site=m,
        m_strand=m_strand,
        Ny=int(Ny),
        Nk=int(Nk),
        t1=float(t1),
        t2=float(t2),
        U=float(U),
        Lz=float(Lz),
        C_C=float(C_C),
        saturated=bool(saturated),
        vacuum=float(vacuum),
        filling=float(filling),
        mix=float(mix),
        tol=float(tol),
        max_iter=int(max_iter),
        m0=float(m0),
    )

# ----------------------------
# DOS + plots + outputs
# ----------------------------
def compute_dos(result, nE=600, eta=0.05):
    E_flat = np.asarray(result["E"], float).reshape(-1)
    Emin = float(E_flat.min() - 3 * eta)
    Emax = float(E_flat.max() + 3 * eta)
    Egrid = np.linspace(Emin, Emax, int(nE))
    dos = np.zeros_like(Egrid)
    pref = 1.0 / (np.sqrt(2*np.pi) * eta)
    Nk = len(result["k_grid"])
    for E0 in E_flat:
        dos += (1.0 / Nk) * pref * np.exp(-0.5 * ((Egrid - E0)/eta)**2)
    return Egrid, dos

def _add_corner_label(ax, text, where="tr"):
    x, ha = (0.97, "right") if where == "tr" else (0.03, "left")
    ax.text(
        x, 0.97, text, transform=ax.transAxes,
        ha=ha, va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.7", alpha=0.85),
    )

def _make_output_dirs(base_dir, mode_tag, N, t1, t2, U):
    run_dir = os.path.join(base_dir, mode_tag, f"N{N}", f"t1{t1:.4f}", f"t2{t2:.4f}", f"U{U:.4f}")
    dirs = {
        "run": run_dir,
        "bands": os.path.join(run_dir, "bands"),
        "dos": os.path.join(run_dir, "dos"),
        "mag": os.path.join(run_dir, "magnetization"),
        "data": os.path.join(run_dir, "data"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs

def _make_all_dirs(base_dir, mode_tag):
    all_dirs = {
        "bands": os.path.join(base_dir, mode_tag, "_ALL", "bands"),
        "dos": os.path.join(base_dir, mode_tag, "_ALL", "dos"),
        "mag": os.path.join(base_dir, mode_tag, "_ALL", "magnetization"),
    }
    for d in all_dirs.values():
        os.makedirs(d, exist_ok=True)
    return all_dirs

def _plot_bands(result, out_png, annotate):
    k = result["k_grid"]
    Eavg = 0.5 * (result["E"][0] + result["E"][1])
    mu = float(result["mu"])

    plt.figure(figsize=(7.2, 4.6))
    for b in range(Eavg.shape[1]):
        plt.plot(k, Eavg[:, b], lw=0.9, alpha=0.85)
    plt.axhline(mu, ls="--", lw=1.2, alpha=0.7)
    plt.xlabel("k")
    plt.ylabel("Energy (eV)")
    plt.grid(True, alpha=0.25)
    _add_corner_label(plt.gca(), annotate, "tr")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def _plot_dos(Egrid, dos, mu, out_png, annotate):
    plt.figure(figsize=(6.2, 4.6))
    plt.plot(Egrid, dos, lw=1.6)
    plt.axvline(mu, ls="--", lw=1.2, alpha=0.7)
    plt.xlabel("Energy (eV)")
    plt.ylabel("DOS (arb.)")
    plt.grid(True, alpha=0.25)
    _add_corner_label(plt.gca(), annotate, "tr")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def _plot_mstrand(m_strand, out_png, annotate):
    x = np.arange(len(m_strand))
    plt.figure(figsize=(6.2, 4.6))
    plt.plot(x, m_strand, marker="o", lw=1.8)
    plt.axhline(0.0, ls="--", lw=1.2, alpha=0.8)
    plt.xlabel("strand index")
    plt.ylabel("magnetization (μB)")
    plt.grid(True, alpha=0.25)
    _add_corner_label(plt.gca(), annotate, "tl")
    plt.tight_layout()
    plt.savefig(out_png, dpi=300)
    plt.close()

def run_single_case(
    *,
    N, U,
    C_C, VACUUM, SATURATED, M,
    t1, t2,
    Nk, filling, max_iter, mix, tol, m0,
    dos_nE, dos_eta,
    base_dir, mode_tag="single",
    copy_to_all=True,
    verbose=False,
):
    result = solve_zgnr_mf_nnn(
        Ny=N, U=U, t1=t1, t2=t2,
        C_C=C_C, vacuum=VACUUM, saturated=SATURATED, M=M,
        Nk=Nk, filling=filling,
        max_iter=max_iter, mix=mix, tol=tol, m0=m0, verbose=verbose,
    )

    dirs = _make_output_dirs(base_dir, mode_tag, N, t1, t2, U)
    all_dirs = _make_all_dirs(base_dir, mode_tag) if copy_to_all else None

    annotate = f"$N={N}$\n$t_1={t1:.2f}$\n$t_2={t2:.2f}$\n$U/t_1={U/t1:.2f}$"

    bands_png = os.path.join(dirs["bands"], "bands.png")
    _plot_bands(result, bands_png, annotate)

    Egrid, dos = compute_dos(result, nE=dos_nE, eta=dos_eta)
    dos_png = os.path.join(dirs["dos"], "dos.png")
    _plot_dos(Egrid, dos, mu=result["mu"], out_png=dos_png, annotate=annotate)

    mag_png = os.path.join(dirs["mag"], "mag_profile.png")
    _plot_mstrand(result["m_strand"], mag_png, annotate)

    np.savez(os.path.join(dirs["data"], "run_summary.npz"), **result, dos_nE=int(dos_nE), dos_eta=float(dos_eta),
             Egrid=Egrid, dos=dos)

    if copy_to_all:
        shutil.copy2(bands_png, os.path.join(all_dirs["bands"], f"bands_N{N:02d}.png"))
        shutil.copy2(dos_png, os.path.join(all_dirs["dos"], f"dos_N{N:02d}.png"))
        shutil.copy2(mag_png, os.path.join(all_dirs["mag"], f"mag_N{N:02d}.png"))

    return dirs["run"]

def run_sweep_N(*, N_min, N_max, U, C_C, VACUUM, SATURATED, M, t1, t2,
                a_dummy, Nk, filling, max_iter, mix, tol, m0, dos_nE, dos_eta,
                base_dir, mode_tag="sweep_N", verbose=False):
    _make_all_dirs(base_dir, mode_tag)
    for N in range(int(N_min), int(N_max) + 1):
        print(f"\nRunning MF+NNN: N={N} U={U:.4f}")
        out = run_single_case(
            N=N, U=U,
            C_C=C_C, VACUUM=VACUUM, SATURATED=SATURATED, M=M,
            t1=t1, t2=t2,
            Nk=Nk, filling=filling,
            max_iter=max_iter, mix=mix, tol=tol, m0=m0,
            dos_nE=dos_nE, dos_eta=dos_eta,
            base_dir=base_dir, mode_tag=mode_tag,
            copy_to_all=True, verbose=verbose,
        )
        print("Saved:", out)

def run_sweep_U(*, N_fixed, Umin_over_t1, Umax_over_t1, dU_over_t1,
                C_C, VACUUM, SATURATED, M, t1, t2,
                Nk, filling, max_iter, mix, tol, m0, dos_nE, dos_eta,
                base_dir, mode_tag="sweep_U", verbose=False):
    _make_all_dirs(base_dir, mode_tag)
    Ut_list = np.arange(Umin_over_t1, Umax_over_t1 + 1e-12, dU_over_t1)
    for Ut in Ut_list:
        U = float(t1) * float(Ut)
        print(f"\nRunning MF+NNN: N={N_fixed} U/t1={Ut:.2f}")
        out = run_single_case(
            N=N_fixed, U=U,
            C_C=C_C, VACUUM=VACUUM, SATURATED=SATURATED, M=M,
            t1=t1, t2=t2,
            Nk=Nk, filling=filling,
            max_iter=max_iter, mix=mix, tol=tol, m0=m0,
            dos_nE=dos_nE, dos_eta=dos_eta,
            base_dir=base_dir, mode_tag=mode_tag,
            copy_to_all=True, verbose=verbose,
        )
        print("Saved:", out)
