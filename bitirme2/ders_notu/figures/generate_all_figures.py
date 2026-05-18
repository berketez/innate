#!/usr/bin/env python3
"""
INNATE Bitirme-2 Ders Notu - Tum Gorselleri Ureten Script
==========================================================
Berke Tezgoçen - ITU Fizik Muhendisligi
Tarih: 2026-02-19

12 adet akademik kalitede gorsel uretir.
Cikti: /Users/apple/Desktop/nsneuron/bitirme2/ders_notu/figures/
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.gridspec as gridspec
from matplotlib import cm

# --- Genel Ayarlar ---
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "text.usetex": False,
    "mathtext.fontset": "cm",
})

SAVE_KW = {"dpi": 300, "bbox_inches": "tight"}

OUTDIR = "/Users/apple/Desktop/nsneuron/bitirme2/ders_notu/figures/"


# =====================================================================
# 1. problem_geometry.png - 3D kutu sematik
# =====================================================================
def fig_problem_geometry():
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    Lx, Ly, Lz = 6, 10, 4

    # Kutu koselerini tanimla
    # Alt yuz (y=0): kirmizi (sicak)
    # Ust yuz (y=Ly): mavi (soguk)
    def face(corners, color, alpha=0.15):
        poly = Poly3DCollection([corners], alpha=alpha, facecolor=color,
                                edgecolor="k", linewidth=1.2)
        ax.add_collection3d(poly)

    # Alt yuz (y=0) - SICAK
    face([(0,0,0),(Lx,0,0),(Lx,0,Lz),(0,0,Lz)], "red", 0.3)
    # Ust yuz (y=Ly) - SOGUK
    face([(0,Ly,0),(Lx,Ly,0),(Lx,Ly,Lz),(0,Ly,Lz)], "blue", 0.3)
    # On yuz (z=Lz)
    face([(0,0,Lz),(Lx,0,Lz),(Lx,Ly,Lz),(0,Ly,Lz)], "gray", 0.08)
    # Arka yuz (z=0)
    face([(0,0,0),(Lx,0,0),(Lx,Ly,0),(0,Ly,0)], "gray", 0.08)
    # Sol yuz (x=0)
    face([(0,0,0),(0,Ly,0),(0,Ly,Lz),(0,0,Lz)], "gray", 0.08)
    # Sag yuz (x=Lx)
    face([(Lx,0,0),(Lx,Ly,0),(Lx,Ly,Lz),(Lx,0,Lz)], "gray", 0.08)

    # Boyut etiketleri
    ax.text(Lx/2, -1.5, 0, r"$L_x = 6$", fontsize=13, ha="center",
            fontweight="bold")
    ax.text(-1.5, Ly/2, 0, r"$L_y = 10$", fontsize=13, ha="center",
            fontweight="bold")
    ax.text(0, -1.5, Lz/2, r"$L_z = 4$", fontsize=13, ha="center",
            fontweight="bold")

    # Sicaklik etiketleri
    ax.text(Lx/2, -0.8, Lz/2, r"$T_{\mathrm{hot}} = 20\,°\mathrm{C}$",
            fontsize=14, ha="center", color="darkred", fontweight="bold")
    ax.text(Lx/2, Ly+0.8, Lz/2, r"$T_{\mathrm{cold}} = 0\,°\mathrm{C}$",
            fontsize=14, ha="center", color="darkblue", fontweight="bold")

    # Yatay oklar: ruzgar/forcing yonu (x-yonu)
    for yy in [3, 7]:
        for zz in [1, 3]:
            ax.quiver(-1.5, yy, zz, 2, 0, 0, arrow_length_ratio=0.3,
                      color="green", linewidth=2, alpha=0.8)
    ax.text(-2.5, 5, 2, r"$\vec{F}_x$" + "\n(Forcing)", fontsize=11,
            color="darkgreen", ha="center", fontweight="bold")

    # Yercekimi oku
    ax.quiver(Lx+1.5, Ly/2, Lz/2, 0, -3, 0, arrow_length_ratio=0.15,
              color="purple", linewidth=3)
    ax.text(Lx+2.2, Ly/2-1.5, Lz/2, r"$\vec{g}$", fontsize=16,
            color="purple", fontweight="bold")

    # Eksen ayarlari
    ax.set_xlim(-3, Lx+3)
    ax.set_ylim(-3, Ly+3)
    ax.set_zlim(-1, Lz+1)
    ax.set_xlabel("x (ruzgar)", fontsize=12, labelpad=10)
    ax.set_ylabel("y (dikey)", fontsize=12, labelpad=10)
    ax.set_zlabel("z (derinlik)", fontsize=12, labelpad=10)
    ax.set_title("Problem Geometrisi: Mixed Convection in a 3D Box", fontsize=15,
                 fontweight="bold", pad=20)
    ax.view_init(elev=22, azim=-55)
    ax.set_box_aspect([Lx, Ly, Lz])

    fig.savefig(OUTDIR + "problem_geometry.png", **SAVE_KW)
    plt.close(fig)
    print("  [1/12] problem_geometry.png")


# =====================================================================
# 2. convection_types.png - 3 panel konveksiyon turleri
# =====================================================================
def fig_convection_types():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))

    nx, ny = 20, 25
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)

    # Panel 1: Dogal konveksiyon (Ri >> 1) - dikey plume'lar
    ax = axes[0]
    u1 = 0.3 * np.sin(3*np.pi*X) * np.cos(np.pi*Y)
    v1 = np.sin(np.pi*Y) * (1 + 0.5*np.sin(3*np.pi*X))
    speed1 = np.sqrt(u1**2 + v1**2)
    ax.streamplot(X, Y, u1, v1, color=speed1, cmap="Reds", density=1.8,
                  linewidth=1.2, arrowsize=1.2)
    ax.set_title(r"Dogal Konveksiyon" + "\n" + r"$\mathrm{Ri} \gg 1$",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # Panel 2: Zorlanmis konveksiyon (Ri << 1) - yatay akis
    ax = axes[1]
    u2 = np.ones_like(X) + 0.1*np.sin(2*np.pi*Y)
    v2 = 0.08 * np.sin(4*np.pi*X) * np.sin(2*np.pi*Y)
    speed2 = np.sqrt(u2**2 + v2**2)
    ax.streamplot(X, Y, u2, v2, color=speed2, cmap="Blues", density=1.8,
                  linewidth=1.2, arrowsize=1.2)
    ax.set_title(r"Zorlanmis Konveksiyon" + "\n" + r"$\mathrm{Ri} \ll 1$",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("x")

    # Panel 3: Mixed convection (Ri ~ 1) - egik plume'lar
    ax = axes[2]
    u3 = 0.6 + 0.4*np.sin(2*np.pi*X)*np.sin(np.pi*Y)
    v3 = 0.5*np.sin(np.pi*Y)*(1 + 0.5*np.cos(3*np.pi*X))
    speed3 = np.sqrt(u3**2 + v3**2)
    ax.streamplot(X, Y, u3, v3, color=speed3, cmap="Greens", density=1.8,
                  linewidth=1.2, arrowsize=1.2)
    ax.set_title(r"Mixed Konveksiyon" + "\n" + r"$\mathrm{Ri} \sim 1$",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("x")

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # Alt kirmizi, ust mavi cizgi
        ax.axhline(0, color="red", linewidth=3, alpha=0.6)
        ax.axhline(1, color="blue", linewidth=3, alpha=0.6)

    fig.suptitle("Konveksiyon Rejimleri: Richardson Sayisina Gore",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTDIR + "convection_types.png", **SAVE_KW)
    plt.close(fig)
    print("  [2/12] convection_types.png")


# =====================================================================
# 3. regime_map.png - Re-Ra rejim haritasi
# =====================================================================
def fig_regime_map():
    fig, ax = plt.subplots(figsize=(9, 7))

    Re = np.logspace(2, 5, 500)
    Ra = np.logspace(3, 9, 500)
    RE, RA = np.meshgrid(Re, Ra)

    # Ri = Ra / (Re^2 * Pr),  Pr ~ 1 icin Ri ~ Ra / Re^2
    # Gercekte Ri = Gr/Re^2 = (Ra/Pr)/Re^2
    # Basitlik icin Ri = Ra / Re^2
    Ri = RA / RE**2

    # Renkli bolgeler
    from matplotlib.colors import ListedColormap, BoundaryNorm
    levels = [0, 0.1, 1, 10, 1e10]
    colors_map = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]
    cmap = ListedColormap(colors_map)
    norm = BoundaryNorm(levels, cmap.N)

    pcm = ax.pcolormesh(RE, RA, Ri, cmap=cmap, norm=norm, alpha=0.4, shading="auto")

    # Ri cizgileri
    for ri_val, label, ls in [(0.1, "Ri = 0.1", "--"), (1, "Ri = 1", "-"),
                               (10, "Ri = 10", "--")]:
        # Ra = Ri * Re^2
        ra_line = ri_val * Re**2
        mask = (ra_line >= 1e3) & (ra_line <= 1e9)
        ax.plot(Re[mask], ra_line[mask], ls, color="k", linewidth=2, alpha=0.8)
        # Etiket
        idx = np.searchsorted(ra_line[mask], 1e6)
        if idx < len(Re[mask]):
            ax.text(Re[mask][min(idx, len(Re[mask])-1)],
                    ra_line[mask][min(idx, len(ra_line[mask])-1)] * 1.5,
                    label, fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                              edgecolor="k", alpha=0.8))

    # Bizim parametre araligi
    rect = plt.Rectangle((5000, 1e5), 5000, 1e7-1e5, linewidth=3,
                          edgecolor="black", facecolor="gold", alpha=0.5,
                          linestyle="-", zorder=5)
    ax.add_patch(rect)
    ax.text(7000, 3e6, "Bizim\nParametre\nAraligi", fontsize=11,
            ha="center", va="center", fontweight="bold", zorder=6,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    # Bolge etiketleri
    ax.text(5e4, 5e3, "Zorlanmis\nKonveksiyon", fontsize=13,
            ha="center", color="#2c3e50", fontweight="bold", alpha=0.8)
    ax.text(300, 5e8, "Dogal\nKonveksiyon", fontsize=13,
            ha="center", color="#c0392b", fontweight="bold", alpha=0.8)
    ax.text(1e4, 5e8, "Mixed\nConvection", fontsize=13,
            ha="center", color="#27ae60", fontweight="bold", alpha=0.8)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$\mathrm{Re}$", fontsize=14)
    ax.set_ylabel(r"$\mathrm{Ra}$", fontsize=14)
    ax.set_xlim(1e2, 1e5)
    ax.set_ylim(1e3, 1e9)
    ax.set_title("Konveksiyon Rejim Haritasi", fontsize=15, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3, linestyle=":")

    # Legend
    legend_patches = [
        mpatches.Patch(color="#3498db", alpha=0.5, label=r"Zorlanmis ($\mathrm{Ri}<0.1$)"),
        mpatches.Patch(color="#2ecc71", alpha=0.5, label=r"Mixed ($0.1<\mathrm{Ri}<10$)"),
        mpatches.Patch(color="#e74c3c", alpha=0.5, label=r"Dogal ($\mathrm{Ri}>10$)"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=11)

    fig.tight_layout()
    fig.savefig(OUTDIR + "regime_map.png", **SAVE_KW)
    plt.close(fig)
    print("  [3/12] regime_map.png")


# =====================================================================
# 4. kolmogorov_forcing.png - Forcing profili
# =====================================================================
def fig_kolmogorov_forcing():
    fig, ax = plt.subplots(figsize=(6, 8))

    Ly = 10
    A = 1.0
    y = np.linspace(0, Ly, 500)
    Fx = A * np.sin(2 * np.pi * y / Ly)

    ax.plot(Fx, y, color="#2c3e50", linewidth=2.5)
    ax.fill_betweenx(y, 0, Fx, where=(Fx > 0), alpha=0.2, color="green",
                     label=r"$F_x > 0$ (sag)")
    ax.fill_betweenx(y, 0, Fx, where=(Fx < 0), alpha=0.2, color="red",
                     label=r"$F_x < 0$ (sol)")

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel(r"$F_x = A \sin(2\pi y / L_y)$", fontsize=14)
    ax.set_ylabel(r"$y$", fontsize=14)
    ax.set_title("Kolmogorov Forcing Profili", fontsize=15, fontweight="bold")
    ax.set_ylim(0, Ly)
    ax.set_xlim(-1.5, 1.5)
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, alpha=0.3, linestyle=":")

    # Onemli noktalari isaretleme
    ax.annotate(r"Maks: $F_x = +A$", xy=(1, Ly/4), xytext=(0.5, Ly/4 + 1),
                fontsize=11, arrowprops=dict(arrowstyle="->", color="green"),
                color="green", fontweight="bold")
    ax.annotate(r"Min: $F_x = -A$", xy=(-1, 3*Ly/4), xytext=(-0.5, 3*Ly/4 + 1),
                fontsize=11, arrowprops=dict(arrowstyle="->", color="red"),
                color="red", fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUTDIR + "kolmogorov_forcing.png", **SAVE_KW)
    plt.close(fig)
    print("  [4/12] kolmogorov_forcing.png")


# =====================================================================
# 5. energy_spectrum.png - Enerji spektrumu
# =====================================================================
def fig_energy_spectrum():
    fig, ax = plt.subplots(figsize=(10, 7))

    k = np.logspace(0, 3, 1000)
    Ck = 1.5
    eps = 1.0

    # Teorik -5/3 spektrum (tam halini model edelim)
    k_peak = 5
    # Enjeksiyon bolgesi: yukselen
    E_inject = Ck * eps**(2/3) * k**2 * np.exp(-(k/k_peak)**2)
    # Inertial: -5/3
    E_inertial = Ck * eps**(2/3) * k**(-5/3)
    # Dissipasyon: ustel azalma
    k_eta = 300  # Kolmogorov
    E_dissip = Ck * eps**(2/3) * k**(-5/3) * np.exp(-2*(k/k_eta)**2)

    # Birlestir: inject + inertial + dissipasyon
    E = np.minimum(E_inject + 0.01, E_inertial) * np.exp(-2*(k/k_eta)**2)
    E = np.where(k < k_peak, E_inject, E_dissip)

    # Daha temiz model: piecewise
    E_model = np.zeros_like(k)
    for i, ki in enumerate(k):
        if ki < k_peak:
            E_model[i] = 0.1 * (ki/k_peak)**3
        else:
            E_model[i] = 0.1 * (ki/k_peak)**(-5/3) * np.exp(-1.5*(ki/k_eta)**1.5)

    ax.loglog(k, E_model, color="#2c3e50", linewidth=2.5, label=r"$E(k)$ (model)")

    # -5/3 referans cizgisi
    k_ref = np.logspace(0.8, 2.2, 100)
    E_ref = 0.08 * (k_ref/k_peak)**(-5/3)
    ax.loglog(k_ref, E_ref, "--", color="red", linewidth=2,
              label=r"$k^{-5/3}$ (Kolmogorov)")

    # Bolge etiketleri
    ax.axvspan(1, k_peak, alpha=0.1, color="blue")
    ax.text(2.5, 0.003, "Enerji\nEnjeksiyonu", fontsize=12, ha="center",
            color="#2980b9", fontweight="bold")

    ax.axvspan(k_peak, 100, alpha=0.1, color="green")
    ax.text(30, 0.003, "Inertial\nSubrange\n" + r"$E \propto k^{-5/3}$",
            fontsize=12, ha="center", color="#27ae60", fontweight="bold")

    ax.axvspan(100, 1000, alpha=0.1, color="red")
    ax.text(400, 0.003, "Dissipasyon", fontsize=12, ha="center",
            color="#c0392b", fontweight="bold")

    # DNS/LES cozunurluk
    k_dns = 200
    k_les = 50
    ax.axvline(k_dns, color="purple", linestyle="-.", linewidth=1.5, alpha=0.7)
    ax.text(k_dns*1.2, 5e-2, r"$k_{\mathrm{DNS}}$", fontsize=12, color="purple",
            fontweight="bold")
    ax.axvline(k_les, color="orange", linestyle="-.", linewidth=1.5, alpha=0.7)
    ax.text(k_les*1.3, 5e-2, r"$k_{\mathrm{LES}}$", fontsize=12, color="orange",
            fontweight="bold")

    ax.set_xlabel(r"Dalga sayisi $k$", fontsize=14)
    ax.set_ylabel(r"Enerji yoğunlugu $E(k)$", fontsize=14)
    ax.set_title("Turbulans Enerji Spektrumu", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12, loc="upper right")
    ax.grid(True, which="both", alpha=0.2, linestyle=":")
    ax.set_xlim(1, 1000)
    ax.set_ylim(1e-10, 1)

    fig.tight_layout()
    fig.savefig(OUTDIR + "energy_spectrum.png", **SAVE_KW)
    plt.close(fig)
    print("  [5/12] energy_spectrum.png")


# =====================================================================
# 6. temperature_decomposition.png - T = Tbase + T'
# =====================================================================
def fig_temperature_decomposition():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    Ly = 10
    nx, ny = 64, 80
    x = np.linspace(0, 6, nx)
    y = np.linspace(0, Ly, ny)
    X, Y = np.meshgrid(x, y)

    # Panel 1: T_base(y) = lineer profil (20 -> 0)
    ax = axes[0]
    T_base_1d = 20 * (1 - y/Ly)
    T_base = 20 * (1 - Y/Ly)
    ax.plot(T_base_1d, y, color="#e74c3c", linewidth=3)
    ax.fill_betweenx(y, 0, T_base_1d, alpha=0.15, color="red")
    ax.set_xlabel(r"$T_{\mathrm{base}}$ [°C]", fontsize=13)
    ax.set_ylabel(r"$y$", fontsize=13)
    ax.set_title(r"$T_{\mathrm{base}}(y)$: Lineer Profil", fontsize=13,
                 fontweight="bold")
    ax.set_xlim(-1, 22)
    ax.set_ylim(0, Ly)
    ax.axhline(0, color="red", linewidth=2, alpha=0.5)
    ax.axhline(Ly, color="blue", linewidth=2, alpha=0.5)
    ax.text(18, 0.3, r"$20\,°C$", fontsize=11, color="red")
    ax.text(2, Ly-0.5, r"$0\,°C$", fontsize=11, color="blue")
    ax.grid(True, alpha=0.3)

    # Panel 2: T'(x,y) = pertürbasyon
    ax = axes[1]
    np.random.seed(42)
    # Periyodik pertürbasyon: dusuk frekanslarin toplami
    T_prime = np.zeros_like(X)
    for kx in range(1, 5):
        for ky in range(1, 5):
            amp = 2.0 / (kx + ky)
            phase = np.random.uniform(0, 2*np.pi)
            T_prime += amp * np.sin(2*np.pi*kx*X/6 + phase) * np.sin(np.pi*ky*Y/Ly)

    im = ax.pcolormesh(X, Y, T_prime, cmap="RdBu_r", shading="auto",
                       vmin=-3, vmax=3)
    fig.colorbar(im, ax=ax, label=r"$T'$ [°C]", shrink=0.8)
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$y$", fontsize=13)
    ax.set_title(r"$T'(x,y)$: Perturbasyon", fontsize=13, fontweight="bold")
    ax.set_aspect("auto")

    # Panel 3: T_total
    ax = axes[2]
    T_total = T_base + T_prime
    im2 = ax.pcolormesh(X, Y, T_total, cmap="hot", shading="auto")
    fig.colorbar(im2, ax=ax, label=r"$T$ [°C]", shrink=0.8)
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel(r"$y$", fontsize=13)
    ax.set_title(r"$T = T_{\mathrm{base}} + T'$", fontsize=13, fontweight="bold")
    ax.set_aspect("auto")

    # Arti/esittir isaretleri paneller arasi
    fig.text(0.34, 0.5, "+", fontsize=36, ha="center", va="center",
             fontweight="bold", color="#2c3e50")
    fig.text(0.65, 0.5, "=", fontsize=36, ha="center", va="center",
             fontweight="bold", color="#2c3e50")

    fig.suptitle(r"Sicaklik Ayristirmasi: $T = T_{\mathrm{base}}(y) + T'(\mathbf{x},t)$",
                 fontsize=15, fontweight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUTDIR + "temperature_decomposition.png", **SAVE_KW)
    plt.close(fig)
    print("  [6/12] temperature_decomposition.png")


# =====================================================================
# 7. spectral_derivative.png - Spectral turev
# =====================================================================
def fig_spectral_derivative():
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))

    N = 64
    x = np.linspace(0, 1, N, endpoint=False)
    f = np.sin(2*np.pi*x)
    fp = 2*np.pi * np.cos(2*np.pi*x)

    # Ust panel: Fiziksel uzay
    ax = axes[0]
    ax.plot(x, f, color="#2c3e50", linewidth=2.5, label=r"$f(x) = \sin(2\pi x)$")
    ax.plot(x, fp / (2*np.pi), color="#e74c3c", linewidth=2.5, linestyle="--",
            label=r"$f'(x)/(2\pi) = \cos(2\pi x)$")
    ax.set_xlabel(r"$x$", fontsize=13)
    ax.set_ylabel("Deger", fontsize=13)
    ax.set_title("Fiziksel Uzay", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)

    # Alt panel: Fourier uzayi (stem plot)
    ax = axes[1]
    fhat = np.fft.fft(f) / N
    fphat = np.fft.fft(fp) / N
    k = np.arange(N)

    # Sadece ilk N/2 modlari goster
    Nh = N // 2
    markerline, stemlines, baseline = ax.stem(
        k[:Nh], np.abs(fhat[:Nh]), linefmt="C0-", markerfmt="C0o",
        basefmt="gray", label=r"$|\hat{f}(k)|$")
    plt.setp(stemlines, linewidth=2)
    plt.setp(markerline, markersize=6)

    markerline2, stemlines2, baseline2 = ax.stem(
        k[:Nh]+0.3, np.abs(fphat[:Nh])/(2*np.pi), linefmt="C3-", markerfmt="C3s",
        basefmt="gray", label=r"$|ik \cdot \hat{f}(k)|/(2\pi)$")
    plt.setp(stemlines2, linewidth=2)
    plt.setp(markerline2, markersize=6)

    ax.set_xlabel(r"Mod indeksi $k$", fontsize=13)
    ax.set_ylabel("Genlik", fontsize=13)
    ax.set_title("Fourier Uzayi", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11, loc="upper right")
    ax.set_xlim(-1, 10)
    ax.grid(True, alpha=0.3)

    # FFT ok
    fig.text(0.5, 0.48, r"$\mathrm{FFT} \;\rightarrow\; ik\cdot\hat{f}(k) \;\rightarrow\; \mathrm{IFFT}$",
             fontsize=14, ha="center", fontweight="bold",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#ecf0f1",
                       edgecolor="#2c3e50"))

    fig.tight_layout(h_pad=3)
    fig.savefig(OUTDIR + "spectral_derivative.png", **SAVE_KW)
    plt.close(fig)
    print("  [7/12] spectral_derivative.png")


# =====================================================================
# 8. dealiasing.png - 2/3 kurali
# =====================================================================
def fig_dealiasing():
    fig, axes = plt.subplots(2, 1, figsize=(10, 7))

    N = 32
    k = np.arange(N)
    k_cut = N // 3  # 2/3 kurali

    # Sinyal spektrumu (ornek)
    np.random.seed(7)
    fhat = np.zeros(N)
    fhat[1] = 1.0
    fhat[2] = 0.7
    fhat[3] = 0.3
    fhat[4] = 0.15

    ghat = np.zeros(N)
    ghat[1] = 0.8
    ghat[3] = 0.5
    ghat[5] = 0.2

    # Ust panel: orijinal spektrum
    ax = axes[0]
    markerline, stemlines, baseline = ax.stem(k, fhat, linefmt="C0-",
                                               markerfmt="C0o", basefmt="gray")
    plt.setp(stemlines, linewidth=2)
    markerline2, stemlines2, baseline2 = ax.stem(k+0.3, ghat, linefmt="C1-",
                                                   markerfmt="C1s", basefmt="gray")
    plt.setp(stemlines2, linewidth=2)

    ax.axvline(k_cut, color="red", linewidth=2.5, linestyle="--", alpha=0.8)
    ax.axvspan(k_cut, N, alpha=0.15, color="red")
    ax.axvspan(0, k_cut, alpha=0.08, color="green")
    ax.text(k_cut/2, 0.9, "Korunan\nModlar", fontsize=12, ha="center",
            color="green", fontweight="bold")
    ax.text((k_cut+N)/2, 0.9, "Sifirlanan\nModlar", fontsize=12, ha="center",
            color="red", fontweight="bold")
    ax.text(k_cut+0.5, 1.05, r"$k_c = N/3$", fontsize=13, color="red",
            fontweight="bold")
    ax.set_xlabel(r"Mod indeksi $k$", fontsize=13)
    ax.set_ylabel("Genlik", fontsize=13)
    ax.set_title(r"2/3 Dealiasing Kurali: $\hat{f}(k)$ ve $\hat{g}(k)$",
                 fontsize=14, fontweight="bold")
    ax.legend([r"$\hat{f}(k)$", r"$\hat{g}(k)$"], fontsize=11)
    ax.set_xlim(-1, N)
    ax.grid(True, alpha=0.3)

    # Alt panel: carpim sonrasi (aliased vs dealiased)
    ax = axes[1]

    # Konvolusyon (carpim)
    fg_conv = np.convolve(fhat, ghat, mode="full")[:N]
    fg_dealiased = fg_conv.copy()
    fg_dealiased[k_cut:] = 0

    markerline, stemlines, baseline = ax.stem(k, fg_conv, linefmt="C3-",
                                               markerfmt="C3^", basefmt="gray")
    plt.setp(stemlines, linewidth=1.5, alpha=0.5)
    markerline2, stemlines2, baseline2 = ax.stem(k+0.3, fg_dealiased, linefmt="C2-",
                                                   markerfmt="C2o", basefmt="gray")
    plt.setp(stemlines2, linewidth=2)

    ax.axvline(k_cut, color="red", linewidth=2.5, linestyle="--", alpha=0.8)
    ax.axvspan(k_cut, N, alpha=0.15, color="red")
    ax.set_xlabel(r"Mod indeksi $k$", fontsize=13)
    ax.set_ylabel("Genlik", fontsize=13)
    ax.set_title(r"Carpim Sonrasi: $\widehat{f \cdot g}(k)$",
                 fontsize=14, fontweight="bold")
    ax.legend(["Aliased (ham)", "Dealiased (2/3 kurali)"], fontsize=11)
    ax.set_xlim(-1, N)
    ax.grid(True, alpha=0.3)

    # Aliasing oku
    ax.annotate("Aliased\nmodlar!", xy=(k_cut+3, fg_conv[k_cut+3]),
                xytext=(k_cut+8, fg_conv[k_cut+3]+0.3),
                fontsize=11, color="red", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="red"))

    fig.tight_layout(h_pad=2)
    fig.savefig(OUTDIR + "dealiasing.png", **SAVE_KW)
    plt.close(fig)
    print("  [8/12] dealiasing.png")


# =====================================================================
# 9. innate_architecture.png - INNATE noron ag diyagrami
# =====================================================================
def fig_innate_architecture():
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(-1, 17)
    ax.set_ylim(-2, 10)
    ax.axis("off")

    # Kutu bilgileri: (isim, ogrenilen parametre, renk)
    boxes = [
        ("FluidState\n(u,v,w,T,p)", "", "#ecf0f1", 0),
        ("Advection3D\n(NL term)", "3 params", "#3498db", 1),
        ("Diffusion3D\n" + r"($\nu_{\mathrm{eff}}\nabla^2$)", "2 params", "#2ecc71", 2),
        ("EddyVisc3D\n(Smagorinsky)", "1 param\n" + r"($C_s$)", "#e67e22", 3),
        ("Forcing3D\n(Kolmogorov)", "2 params\n(A, k)", "#9b59b6", 4),
        ("Buoyancy3D\n" + r"($\alpha g T$)", "1 param\n" + r"($\alpha$)", "#e74c3c", 5),
    ]

    boxes2 = [
        ("Projection3D\n(div-free)", "0 params", "#1abc9c", 0),
        ("ThermalAdv\n(NL)", "1 param", "#3498db", 1),
        ("ThermalDiff\n" + r"($\kappa\nabla^2 T$)", "1 param", "#2ecc71", 2),
        ("TimeMarcher3D\n(RK integrator)", "0 params", "#34495e", 3),
        ("FluidState_new\n" + r"($\mathbf{u}^{n+1}, T^{n+1}$)", "", "#ecf0f1", 4),
    ]

    y_top = 7.5
    y_bot = 2.0
    box_w = 2.2
    box_h = 1.6

    # Ust sira
    for name, params, color, i in boxes:
        x = 0.5 + i * 2.6
        rect = FancyBboxPatch((x, y_top), box_w, box_h,
                              boxstyle="round,pad=0.15", facecolor=color,
                              edgecolor="#2c3e50", linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + box_w/2, y_top + box_h/2, name, fontsize=9,
                ha="center", va="center", fontweight="bold", color="white")
        if params:
            ax.text(x + box_w/2, y_top - 0.3, params, fontsize=8,
                    ha="center", va="top", color="#7f8c8d", style="italic")
        # Ok (sonraki kutuya)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(x + box_w + 0.35, y_top + box_h/2),
                        xytext=(x + box_w + 0.05, y_top + box_h/2),
                        arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2))

    # Baglanti oku: ust sira sonu -> alt sira basi
    ax.annotate("", xy=(1.6, y_bot + box_h + 0.1),
                xytext=(boxes[-1][3] * 2.6 + 0.5 + box_w/2, y_top - 0.05),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2.5,
                                connectionstyle="arc3,rad=-0.3"))

    # Alt sira
    for name, params, color, i in boxes2:
        x = 0.5 + i * 3.2
        rect = FancyBboxPatch((x, y_bot), box_w, box_h,
                              boxstyle="round,pad=0.15", facecolor=color,
                              edgecolor="#2c3e50", linewidth=1.5, alpha=0.8)
        ax.add_patch(rect)
        fc = "white" if color != "#ecf0f1" else "#2c3e50"
        ax.text(x + box_w/2, y_bot + box_h/2, name, fontsize=9,
                ha="center", va="center", fontweight="bold", color=fc)
        if params:
            ax.text(x + box_w/2, y_bot - 0.3, params, fontsize=8,
                    ha="center", va="top", color="#7f8c8d", style="italic")
        if i < len(boxes2) - 1:
            ax.annotate("", xy=(x + box_w + 0.95, y_bot + box_h/2),
                        xytext=(x + box_w + 0.05, y_bot + box_h/2),
                        arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2))

    # Baslik
    ax.text(8, 9.5, "INNATE Architecture: Physics-Embedded Neural Operator",
            fontsize=16, ha="center", fontweight="bold", color="#2c3e50")
    ax.text(8, -1.2, r"Toplam ogrenilebilir parametre: $\sim 11$  (vs. milyon parametre in vanilla NN)",
            fontsize=12, ha="center", color="#7f8c8d", style="italic")

    # Geri besleme oku (output -> input)
    ax.annotate("", xy=(0.5, y_top + box_h),
                xytext=(boxes2[-1][3] * 3.2 + 0.5 + box_w, y_bot + box_h/2),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=2,
                                linestyle="--",
                                connectionstyle="arc3,rad=0.4"))
    ax.text(15.5, 6, "Zaman\nadimi\ndongüsü", fontsize=10, color="#c0392b",
            fontweight="bold", ha="center")

    fig.tight_layout()
    fig.savefig(OUTDIR + "innate_architecture.png", **SAVE_KW)
    plt.close(fig)
    print("  [9/12] innate_architecture.png")


# =====================================================================
# 10. curriculum_phases.png - Training fazlari
# =====================================================================
def fig_curriculum_phases():
    fig, ax = plt.subplots(figsize=(14, 6))

    phases = [
        {"name": "Faz A:\nTemel Fizik", "start": 0, "end": 3000,
         "Re": "1000-3000",
         "losses": [("Physics Loss", "#3498db"), ("Continuity", "#2ecc71")]},
        {"name": "Faz B:\nSpectral", "start": 3000, "end": 8000,
         "Re": "3000-5000",
         "losses": [("Physics", "#3498db"), ("Continuity", "#2ecc71"),
                    ("Spectral", "#e67e22")]},
        {"name": "Faz C:\nTurbulans", "start": 8000, "end": 15000,
         "Re": "5000-8000",
         "losses": [("Physics", "#3498db"), ("Continuity", "#2ecc71"),
                    ("Spectral", "#e67e22"), ("Energy Cascade", "#9b59b6")]},
        {"name": "Faz D:\nFull Regime", "start": 15000, "end": 22000,
         "Re": "5000-10000",
         "losses": [("Physics", "#3498db"), ("Continuity", "#2ecc71"),
                    ("Spectral", "#e67e22"), ("Energy", "#9b59b6"),
                    ("Nusselt", "#e74c3c")]},
    ]

    y_base = 0
    bar_h = 0.5
    colors_bg = ["#d5e8f0", "#d4efdf", "#fdebd0", "#f5d5d5"]

    for i, phase in enumerate(phases):
        s = phase["start"]
        e = phase["end"]

        # Arka plan
        ax.axvspan(s, e, alpha=0.2, color=colors_bg[i])

        # Faz ismi (ust)
        ax.text((s+e)/2, len(phases[3]["losses"])*0.7 + 1.2, phase["name"],
                fontsize=12, ha="center", va="center", fontweight="bold")

        # Re araligi
        ax.text((s+e)/2, len(phases[3]["losses"])*0.7 + 0.3,
                f"Re: {phase['Re']}", fontsize=10, ha="center", color="#7f8c8d")

        # Loss cubuklar
        for j, (lname, lcolor) in enumerate(phase["losses"]):
            ax.barh(j*0.7, e-s, left=s, height=bar_h, color=lcolor, alpha=0.7,
                    edgecolor="white", linewidth=1)
            # Sadece en genis fazda yaz
            if i == 3:
                ax.text(e + 200, j*0.7, lname, fontsize=10, va="center",
                        fontweight="bold", color=lcolor)

    # Faz sinirlari
    for x_val in [3000, 8000, 15000]:
        ax.axvline(x_val, color="#2c3e50", linewidth=1.5, linestyle="--", alpha=0.5)

    ax.set_xlabel("Training Adimi (iteration)", fontsize=13)
    ax.set_xlim(-500, 25000)
    ax.set_ylim(-0.5, 5.5)
    ax.set_yticks([])
    ax.set_title("Curriculum Training Fazlari", fontsize=15, fontweight="bold")
    ax.grid(True, axis="x", alpha=0.3)

    # Timeline oklari
    ax.annotate("", xy=(22000, -0.3), xytext=(0, -0.3),
                arrowprops=dict(arrowstyle="->", color="#2c3e50", lw=2))

    fig.tight_layout()
    fig.savefig(OUTDIR + "curriculum_phases.png", **SAVE_KW)
    plt.close(fig)
    print("  [10/12] curriculum_phases.png")


# =====================================================================
# 11. nusselt_vs_ra.png - Nu(Ra) teorik egrisi
# =====================================================================
def fig_nusselt_vs_ra():
    fig, ax = plt.subplots(figsize=(9, 7))

    Ra = np.logspace(2, 10, 2000)
    Ra_cr = 1708

    # Fiziksel olarak dogru Nu(Ra) egrisi
    # Ra < Ra_cr: saf iletim, Nu = 1
    # Ra_cr < Ra < ~10^6: laminar konveksiyon, Nu ~ 0.069*Ra^(1/3) (Pr~0.7)
    # Ra > 10^6: turbulant, ayni korelasyon devam eder (klasik 1/3 scaling)
    # Yumusak gecis icin sigmoid kullan
    Nu_conv = 0.069 * Ra**(1/3)  # Konvektif dalga (tum Ra icin)
    # Gecis: Ra_cr civarinda sigmoid ile 1'den Nu_conv'a
    transition = 1 / (1 + np.exp(-3 * (np.log10(Ra) - np.log10(Ra_cr))))
    Nu = 1 * (1 - transition) + Nu_conv * transition
    # Nu asla 1'in altina dusmemeli
    Nu = np.maximum(Nu, 1.0)

    ax.loglog(Ra, Nu, color="#2c3e50", linewidth=2.5, label=r"$\mathrm{Nu} \approx 0.069\,\mathrm{Ra}^{1/3}$")

    # Globe-Dropkin korelasyonu
    Ra2 = np.logspace(3.5, 10, 500)
    Nu_globe = 0.1 * Ra2**(2/7)
    ax.loglog(Ra2, Nu_globe, "--", color="#e74c3c", linewidth=1.5, alpha=0.7,
              label=r"Globe-Dropkin: $0.1 \cdot \mathrm{Ra}^{2/7}$")

    # Ra_cr cizgisi
    ax.axvline(Ra_cr, color="green", linewidth=2, linestyle=":", alpha=0.8)
    ax.text(Ra_cr*0.5, 120, r"$\mathrm{Ra}_{cr} = 1708$", fontsize=12, color="green",
            fontweight="bold", rotation=0, va="top")

    # Bizim Ra araligi (Nu degerlerini goster)
    ax.axvspan(1e5, 1e7, alpha=0.2, color="gold")
    # Ra=10^5 -> Nu~3.2, Ra=10^7 -> Nu~14.8
    ax.text(1e6, 25, "Bizim\nAralik", fontsize=12, ha="center",
            fontweight="bold", color="#8B6914",
            bbox=dict(boxstyle="round", facecolor="lightyellow", edgecolor="#8B6914", alpha=0.9))

    # Bolge etiketleri
    ax.text(400, 1.5, "Iletim\n(Nu = 1)", fontsize=11, ha="center",
            color="#7f8c8d", style="italic")
    ax.text(1e4, 2, "Laminar\nKonveksiyon", fontsize=11, ha="center",
            color="#2980b9", style="italic")
    ax.text(3e8, 100, "Turbulant\nKonveksiyon", fontsize=11, ha="center",
            color="#c0392b", style="italic")

    ax.set_xlabel(r"Rayleigh Sayisi $\mathrm{Ra}$", fontsize=14)
    ax.set_ylabel(r"Nusselt Sayisi $\mathrm{Nu}$", fontsize=14)
    ax.set_title("Nu-Ra Iliskisi (Rayleigh-Benard Konveksiyon)", fontsize=15,
                 fontweight="bold")
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(True, which="both", alpha=0.2, linestyle=":")
    ax.set_xlim(1e2, 1e10)
    ax.set_ylim(0.8, 300)

    fig.tight_layout()
    fig.savefig(OUTDIR + "nusselt_vs_ra.png", **SAVE_KW)
    plt.close(fig)
    print("  [11/12] nusselt_vs_ra.png")


# =====================================================================
# 12. energy_cascade.png - Kolmogorov kaskad semasi
# =====================================================================
def fig_energy_cascade():
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(-1, 15)
    ax.set_ylim(-1, 8)
    ax.axis("off")

    # Baslik
    ax.text(7, 7.5, "Kolmogorov Enerji Kaskadi", fontsize=18,
            ha="center", fontweight="bold", color="#2c3e50")

    # --- Buyuk girdap (sol) ---
    theta = np.linspace(0, 2*np.pi, 100)
    # Ana girdap
    r_big = 1.8
    cx_big, cy_big = 2, 4
    # Spiral girdap
    t = np.linspace(0, 4*np.pi, 300)
    r_spiral = r_big * (1 - t/(4*np.pi) * 0.3)
    x_spiral = cx_big + r_spiral * np.cos(t)
    y_spiral = cy_big + r_spiral * np.sin(t)
    ax.plot(x_spiral, y_spiral, color="#3498db", linewidth=3, alpha=0.7)
    # Oklar spiral ustune
    for idx in [50, 150, 250]:
        dx = x_spiral[idx+1] - x_spiral[idx]
        dy = y_spiral[idx+1] - y_spiral[idx]
        ax.annotate("", xy=(x_spiral[idx+1], y_spiral[idx+1]),
                    xytext=(x_spiral[idx], y_spiral[idx]),
                    arrowprops=dict(arrowstyle="->", color="#3498db", lw=2))
    ax.text(cx_big, cy_big - 2.5, "Buyuk Olcek\n" + r"$\ell \sim L$",
            fontsize=13, ha="center", fontweight="bold", color="#2c3e50")
    ax.text(cx_big, cy_big + 2.5, "URETIM", fontsize=12, ha="center",
            fontweight="bold", color="#3498db",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#d5e8f0"))

    # --- Orta girdaplar (orta) ---
    centers_mid = [(6.5, 4.8), (7.5, 3.2), (6.8, 3.0), (7.2, 5.0)]
    for cx, cy in centers_mid:
        r_mid = 0.7
        t2 = np.linspace(0, 3*np.pi, 150)
        r2 = r_mid * (1 - t2/(3*np.pi) * 0.2)
        ax.plot(cx + r2*np.cos(t2), cy + r2*np.sin(t2),
                color="#e67e22", linewidth=2, alpha=0.6)
    ax.text(7, 1.5, "Orta Olcek\n" + r"$L > \ell > \eta$",
            fontsize=13, ha="center", fontweight="bold", color="#2c3e50")
    ax.text(7, 6.3, "TRANSFER", fontsize=12, ha="center",
            fontweight="bold", color="#e67e22",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdebd0"))

    # --- Kucuk girdaplar (sag) ---
    centers_small = [(11, 4.5), (11.8, 3.5), (12.3, 5.0), (11.5, 5.5),
                     (12, 4.0), (11.3, 3.0), (12.5, 3.3), (11.8, 5.8)]
    for cx, cy in centers_small:
        r_sm = 0.3
        t3 = np.linspace(0, 2.5*np.pi, 80)
        r3 = r_sm * (1 - t3/(2.5*np.pi) * 0.15)
        ax.plot(cx + r3*np.cos(t3), cy + r3*np.sin(t3),
                color="#e74c3c", linewidth=1.5, alpha=0.5)
    ax.text(12, 1.5, "Kucuk Olcek\n" + r"$\ell \sim \eta$",
            fontsize=13, ha="center", fontweight="bold", color="#2c3e50")
    ax.text(12, 6.3, "DISSIPASYON", fontsize=12, ha="center",
            fontweight="bold", color="#e74c3c",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5d5d5"))

    # Transfer oklari
    ax.annotate("", xy=(5.5, 4.3), xytext=(3.8, 4.3),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=3,
                                mutation_scale=20))
    ax.text(4.65, 4.8, r"$\varepsilon$", fontsize=16, ha="center",
            color="#2c3e50", fontweight="bold")

    ax.annotate("", xy=(10.2, 4.3), xytext=(8.5, 4.3),
                arrowprops=dict(arrowstyle="-|>", color="#2c3e50", lw=3,
                                mutation_scale=20))
    ax.text(9.35, 4.8, r"$\varepsilon$", fontsize=16, ha="center",
            color="#2c3e50", fontweight="bold")

    # Olcek cubugu
    ax.annotate("", xy=(13.5, 0.3), xytext=(0.5, 0.3),
                arrowprops=dict(arrowstyle="<->", color="#7f8c8d", lw=1.5))
    ax.text(0.5, -0.2, r"$L$ (integral)", fontsize=11, ha="left",
            color="#7f8c8d", fontweight="bold")
    ax.text(13.5, -0.2, r"$\eta$ (Kolmogorov)", fontsize=11, ha="right",
            color="#7f8c8d", fontweight="bold")
    ax.text(7, -0.2, r"$\lambda$ (Taylor)", fontsize=11, ha="center",
            color="#7f8c8d", fontweight="bold")
    # Tick isaretleri
    for xp in [0.5, 7, 13.5]:
        ax.plot([xp, xp], [0.1, 0.5], color="#7f8c8d", linewidth=1.5)

    fig.tight_layout()
    fig.savefig(OUTDIR + "energy_cascade.png", **SAVE_KW)
    plt.close(fig)
    print("  [12/12] energy_cascade.png")


# =====================================================================
# MAIN: Tum gorstalleri uret
# =====================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("INNATE Bitirme-2 Ders Notu - Gorsel Uretici")
    print("=" * 60)
    print(f"Cikti dizini: {OUTDIR}")
    print()

    fig_problem_geometry()
    fig_convection_types()
    fig_regime_map()
    fig_kolmogorov_forcing()
    fig_energy_spectrum()
    fig_temperature_decomposition()
    fig_spectral_derivative()
    fig_dealiasing()
    fig_innate_architecture()
    fig_curriculum_phases()
    fig_nusselt_vs_ra()
    fig_energy_cascade()

    print()
    print("=" * 60)
    print("TAMAMLANDI: 12/12 gorsel basariyla uretildi.")
    print(f"Dizin: {OUTDIR}")
    print("=" * 60)
