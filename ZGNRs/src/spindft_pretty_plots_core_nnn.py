# src/spindft_pretty_plots_core_nnn.py
import os
import numpy as np
import matplotlib.pyplot as plt

def pretty_tbfit_nnn_bands(tbfit_npz, out_png, Ny_label=None, dpi=320):
    d = np.load(tbfit_npz, allow_pickle=True)
    k = np.asarray(d["k_plot"], float).ravel()
    dft_val = np.asarray(d["E_val_dft"], float).ravel()
    dft_con = np.asarray(d["E_con_dft"], float).ravel()
    tb_val  = np.asarray(d["E_val_tb"], float).ravel()
    tb_con  = np.asarray(d["E_con_tb"], float).ravel()

    Ny = int(np.array(d["Ny"]).item())
    if Ny_label is None:
        Ny_label = Ny

    t1 = float(np.array(d["t_fit"]).item())
    t2 = float(np.array(d["t2_fit"]).item())
    U  = float(np.array(d["U_fit"]).item())

    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axhline(0.0, lw=1.2, ls="--", color="0.55")
    ax.plot(k, dft_val, lw=3.2, color="tab:red", label="DFT (central π)")
    ax.plot(k, dft_con, lw=3.2, color="tab:red")
    ax.plot(k, tb_val,  lw=2.8, ls="--", color="tab:blue", label="TB+NNN fit")
    ax.plot(k, tb_con,  lw=2.8, ls="--", color="tab:blue")

    ax.set_xlabel(r"$ka/\pi$")
    ax.set_ylabel(r"$E - E_F$ (eV)")
    ax.grid(True, alpha=0.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=12)

    info = rf"$N={Ny_label}$" + "\n" + rf"$t_1={t1:.4f}$ eV" + "\n" + rf"$t_2={t2:.4f}$ eV" + "\n" + rf"$U={U:.4f}$ eV"
    ax.text(0.82, 0.24, info, transform=ax.transAxes,
            ha="center", va="center", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.95))

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)

def pretty_tbfit_nnn_magnetization(tbfit_npz, out_png, Ny_label=None, dpi=320):
    d = np.load(tbfit_npz, allow_pickle=True)
    m_dft = np.asarray(d["m_strand_dft"], float).ravel()
    m_tb  = np.asarray(d["m_strand_tb"], float).ravel()
    Ny = int(np.array(d["Ny"]).item())
    if Ny_label is None:
        Ny_label = Ny

    t1 = float(np.array(d["t_fit"]).item())
    t2 = float(np.array(d["t2_fit"]).item())
    U  = float(np.array(d["U_fit"]).item())

    x = np.arange(len(m_tb))
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.axhline(0.0, lw=1.2, ls="--", color="0.55")
    ax.plot(x, m_tb,  lw=2.6, marker="o", color="tab:blue", label="TB+NNN")
    ax.plot(x, m_dft, lw=2.6, ls="--", marker="s", color="#c44e52", alpha=0.85, label="DFT (strand avg)")
    ax.set_xlabel("strand index")
    ax.set_ylabel("magnetization (μB)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper right", frameon=True, framealpha=0.95, fontsize=12)

    info = rf"$N={Ny_label}$" + "\n" + rf"$t_1={t1:.4f}$" + "\n" + rf"$t_2={t2:.4f}$" + "\n" + rf"$U={U:.4f}$"
    ax.text(0.52, 0.30, info, transform=ax.transAxes,
            ha="left", va="top", fontsize=12,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", alpha=0.95))

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
