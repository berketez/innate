"""
plots.py - Statik grafikler (PNG/PDF)

Icerik:
  - E(t), Z(t), Nu(t) zaman serileri
  - E(k) enerji spektrumu (-5/3 referans cizgisi ile)
  - T(y) ortalama sicaklik profili
  - Loss curriculum grafigi
  - Re-Ra rejim haritasi
  - Ogrenilmis parametre dagilimlari

Kullanim:
  python -m visualize.plots --checkpoint results/checkpoints/checkpoint_epoch015000.pt
  python -m visualize.plots --eval-json results/evaluation_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch

_this_dir = Path(__file__).resolve().parent
_project_dir = _this_dir.parent
sys.path.insert(0, str(_project_dir.parent))
sys.path.insert(0, str(_project_dir))

from config import Config
from model import INNATE3D_MixedConvection, ThermalFluidState


# =====================================================================
# Renk paleti ve stil
# =====================================================================

COLORS = {
    "energy": "#2196F3",
    "enstrophy": "#F44336",
    "nusselt": "#4CAF50",
    "divergence": "#FF9800",
    "dns": "#333333",
    "innate": "#2196F3",
    "spectrum": "#9C27B0",
    "k53": "#999999",
}

def _setup_style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 200,
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
    })


# =====================================================================
# 1. Zaman serisi grafikleri
# =====================================================================


def plot_time_series(
    diagnostics: Dict[str, List[float]],
    dt: float = 0.005,
    save_path: Optional[str] = None,
    title_prefix: str = "",
) -> plt.Figure:
    """
    E(t), Z(t), Nu(t), div(t) zaman serileri -- 2x2 panel.

    Args:
        diagnostics: {"E": [...], "Z": [...], "Nu": [...], "div": [...]}
        dt: zaman adimi (x-ekseni icin)
    """
    _setup_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    n = len(diagnostics.get("E", []))
    t = np.arange(n) * dt

    panels = [
        ("E", "Kinetic Energy E(t)", COLORS["energy"], axes[0, 0]),
        ("Z", "Enstrophy Z(t)", COLORS["enstrophy"], axes[0, 1]),
        ("Nu", "Nusselt Number Nu(t)", COLORS["nusselt"], axes[1, 0]),
        ("div", "Max |div(u)|", COLORS["divergence"], axes[1, 1]),
    ]

    for key, label, color, ax in panels:
        data = diagnostics.get(key, [])
        if not data:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(label)
            continue

        ax.plot(t[: len(data)], data, color=color, linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("Time")
        ax.set_ylabel(key)

        if key == "div":
            ax.set_yscale("log")
            ax.axhline(1e-5, color="gray", linestyle="--", alpha=0.5, label="Target 1e-5")
            ax.legend()
        if key == "Nu":
            ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Nu=1")
            ax.legend()

    fig.suptitle(f"{title_prefix}Diagnostics Time Series", fontsize=14, y=1.02)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


# =====================================================================
# 2. Enerji spektrumu
# =====================================================================


def plot_energy_spectrum(
    spectrum: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Energy Spectrum E(k)",
) -> plt.Figure:
    """
    Shell-averaged enerji spektrumu, -5/3 referans cizgisi ile.

    Args:
        spectrum: 1D numpy array, E(k) degerleri
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    k = np.arange(1, len(spectrum) + 1)
    valid = spectrum > 1e-30

    ax.loglog(k[valid], spectrum[valid], "o-", color=COLORS["spectrum"],
              markersize=3, linewidth=1.0, label="E(k)")

    # -5/3 referans
    if valid.sum() > 2:
        k_ref = k[valid]
        E_ref = spectrum[valid][0] * (k_ref / k_ref[0]) ** (-5.0 / 3.0)
        ax.loglog(k_ref, E_ref, "--", color=COLORS["k53"],
                  linewidth=1.5, label=r"$k^{-5/3}$")

    ax.set_xlabel("Wavenumber k")
    ax.set_ylabel("E(k)")
    ax.set_title(title)
    ax.legend()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


# =====================================================================
# 3. Sicaklik profili T(y)
# =====================================================================


def plot_temperature_profile(
    state: ThermalFluidState,
    config: Config,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Yatay ortalama sicaklik profili T(y) = T_base(y) + <T'>(y).
    Lineer baz profil ile karsilastirir.
    """
    _setup_style()
    dom = config.domain
    phys = config.physics

    y = np.linspace(0, dom.Ly, dom.Ny, endpoint=False)

    # T_base(y)
    T_base = phys.T_hot - (phys.dT / dom.Ly) * y

    # <T'>(y) -- x, z uzerinden ortalama, batch ortalama
    theta_np = state.theta.detach().cpu().numpy()
    theta_mean_y = theta_np.mean(axis=(0, 1, 3))  # [Ny]

    T_total = T_base + theta_mean_y

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(T_total, y, color=COLORS["energy"], linewidth=2, label="T_total(y)")
    ax.plot(T_base, y, "--", color=COLORS["k53"], linewidth=1.5, label="T_base(y) (linear)")
    ax.set_xlabel("Temperature")
    ax.set_ylabel("y (height)")
    ax.set_title("Mean Temperature Profile")
    ax.legend()
    ax.invert_yaxis()  # Ust = soguk, alt = sicak

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


# =====================================================================
# 4. Loss curriculum grafigi
# =====================================================================


def plot_loss_history(
    loss_log: Dict[str, List[float]],
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Training loss bilesenlerinin zaman ilerlemesi.

    Args:
        loss_log: {"L_divergence": [epoch_vals...], ...}
                  veya {"epochs": [...], "L_divergence": [...], ...}
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 6))

    epochs = loss_log.get("epochs", None)
    loss_colors = [
        "#2196F3", "#F44336", "#4CAF50", "#FF9800",
        "#9C27B0", "#795548", "#607D8B", "#E91E63",
        "#00BCD4", "#CDDC39",
    ]

    i = 0
    for key, vals in sorted(loss_log.items()):
        if key == "epochs" or not vals:
            continue
        x = epochs if epochs else list(range(len(vals)))
        color = loss_colors[i % len(loss_colors)]
        ax.semilogy(x[: len(vals)], vals, linewidth=1.0, label=key, color=color, alpha=0.8)
        i += 1

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Training Loss Components")
    ax.legend(ncol=2, fontsize=8)

    # Curriculum faz sinirlari
    for boundary, label in [(3000, "A|B"), (8000, "B|C"), (15000, "C|D")]:
        ax.axvline(boundary, color="gray", linestyle=":", alpha=0.4)
        ax.text(boundary, ax.get_ylim()[1], label, ha="center", va="bottom", fontsize=8)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


# =====================================================================
# 5. 2D kesit (slice) grafigi
# =====================================================================


def plot_field_slices(
    state: ThermalFluidState,
    config: Config,
    field: str = "theta",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    3D alandan ortadaki 3 kesit: xy (z=Nz/2), xz (y=Ny/2), yz (x=Nx/2).

    Args:
        field: "theta", "u", "v", "w", "p", "speed", "vorticity_mag"
    """
    _setup_style()
    dom = config.domain

    # Alan sec
    if field == "theta":
        data = state.theta[0].detach().cpu().numpy()
        cmap, label = "coolwarm", "T' (perturbation)"
    elif field == "u":
        data = state.u[0].detach().cpu().numpy()
        cmap, label = "RdBu_r", "u (x-velocity)"
    elif field == "v":
        data = state.v[0].detach().cpu().numpy()
        cmap, label = "RdBu_r", "v (y-velocity)"
    elif field == "w":
        data = state.w[0].detach().cpu().numpy()
        cmap, label = "RdBu_r", "w (z-velocity)"
    elif field == "p":
        data = state.p[0].detach().cpu().numpy()
        cmap, label = "viridis", "pressure"
    elif field == "speed":
        s = state
        data = np.sqrt(
            s.u[0].detach().cpu().numpy() ** 2
            + s.v[0].detach().cpu().numpy() ** 2
            + s.w[0].detach().cpu().numpy() ** 2
        )
        cmap, label = "inferno", "|u| (speed)"
    elif field == "vorticity_mag":
        s = state
        data = np.sqrt(
            s.omega_x[0].detach().cpu().numpy() ** 2
            + s.omega_y[0].detach().cpu().numpy() ** 2
            + s.omega_z[0].detach().cpu().numpy() ** 2
        )
        cmap, label = "magma", "|omega| (vorticity)"
    else:
        raise ValueError(f"Unknown field: {field}")

    Nx, Ny, Nz = data.shape
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # XY slice (z = Nz//2)
    im0 = axes[0].imshow(
        data[:, :, Nz // 2].T, origin="lower", cmap=cmap, aspect="auto",
        extent=[0, dom.Lx, 0, dom.Ly],
    )
    axes[0].set_title(f"XY slice (z={dom.Lz/2:.1f})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    # XZ slice (y = Ny//2)
    im1 = axes[1].imshow(
        data[:, Ny // 2, :].T, origin="lower", cmap=cmap, aspect="auto",
        extent=[0, dom.Lx, 0, dom.Lz],
    )
    axes[1].set_title(f"XZ slice (y={dom.Ly/2:.1f})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    # YZ slice (x = Nx//2)
    im2 = axes[2].imshow(
        data[Nx // 2, :, :].T, origin="lower", cmap=cmap, aspect="auto",
        extent=[0, dom.Ly, 0, dom.Lz],
    )
    axes[2].set_title(f"YZ slice (x={dom.Lx/2:.1f})")
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("z")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.suptitle(f"{label} - 3 Orthogonal Slices (t={state.t[0,0].item():.3f})", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


# =====================================================================
# 6. Ogrenilmis parametre dagilimlari
# =====================================================================


def plot_learned_parameters(
    model: INNATE3D_MixedConvection,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Modelin ogrenilmis parametrelerini gorsellestirir:
    - layer_scale, nu_scale (per layer)
    - forcing amplitude, buoyancy strength, kappa_scale
    """
    _setup_style()

    params = {}
    for name, p in model.named_parameters():
        if p.numel() <= 10:  # sadece skaler/kucuk parametreler
            params[name] = p.detach().cpu().numpy().flatten()

    if not params:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No scalar parameters found", ha="center", va="center")
        return fig

    # Bar chart
    names = list(params.keys())
    values = [p[0] if len(p) == 1 else p.mean() for p in params.values()]

    # Kisa isimler
    short_names = []
    for n in names:
        parts = n.split(".")
        short = ".".join(parts[-2:]) if len(parts) > 2 else n
        short_names.append(short)

    fig, ax = plt.subplots(figsize=(max(10, len(names) * 0.6), 6))
    bars = ax.bar(range(len(values)), values, color=COLORS["energy"], alpha=0.7)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=7)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.5, label="Init value (1.0)")
    ax.set_ylabel("Parameter Value")
    ax.set_title("Learned Physics Parameters")
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


# =====================================================================
# 7. Utility: checkpoint'tan calistir
# =====================================================================


def _load_model_and_run(
    checkpoint_path: str,
    n_steps: int = 50,
    device: str = "cpu",
) -> Tuple[INNATE3D_MixedConvection, List[ThermalFluidState], Config, Dict]:
    """
    Checkpoint yukle, n_steps ilerlet, diagnostik topla.
    Returns: (model, states, config, diagnostics)
    """
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = Config.from_dict(ckpt["config"]) if "config" in ckpt else Config()
    config._device_override = device

    model = INNATE3D_MixedConvection(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    diagnostics = {"E": [], "Z": [], "Nu": [], "div": []}
    states = []

    with torch.no_grad():
        state = model.create_initial_condition(batch_size=1, device=device)
        states.append(state)

        for step in range(n_steps):
            state = model(state)
            states.append(state)

            diagnostics["E"].append(state.kinetic_energy().mean().item())
            diagnostics["Z"].append(state.enstrophy().mean().item())
            diagnostics["Nu"].append(
                (1.0 + (state.v * state.theta).mean(dim=(-3, -2, -1))).mean().item()
            )
            diagnostics["div"].append(
                model.ops.divergence(state.u, state.v, state.w).abs().max().item()
            )

    return model, states, config, diagnostics


def generate_all_plots(
    checkpoint_path: str,
    output_dir: str = "results/plots",
    n_steps: int = 50,
    device: str = "cpu",
):
    """Tum statik grafikleri uret."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    model, states, config, diag = _load_model_and_run(
        checkpoint_path, n_steps=n_steps, device=device,
    )
    epoch = torch.load(checkpoint_path, weights_only=False, map_location="cpu").get("epoch", "?")
    prefix = f"[Epoch {epoch}] "

    print("Generating plots...")

    # 1. Time series
    plot_time_series(diag, dt=config.physics.dt,
                     save_path=str(out / "time_series.png"), title_prefix=prefix)

    # 2. Energy spectrum
    from train import compute_energy_spectrum
    spec = compute_energy_spectrum(
        states[-1].u, states[-1].v, states[-1].w, model.ops
    ).detach().cpu().numpy()
    plot_energy_spectrum(spec, save_path=str(out / "energy_spectrum.png"),
                         title=f"{prefix}Energy Spectrum E(k)")

    # 3. Temperature profile
    plot_temperature_profile(states[-1], config,
                             save_path=str(out / "temperature_profile.png"))

    # 4. Field slices
    for field in ("theta", "speed", "vorticity_mag"):
        plot_field_slices(states[-1], config, field=field,
                          save_path=str(out / f"slice_{field}.png"))

    # 5. Learned parameters
    plot_learned_parameters(model, save_path=str(out / "learned_params.png"))

    print(f"\nAll plots saved to {out}/")
    plt.close("all")


# =====================================================================
# Entry point
# =====================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate static plots")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/plots")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    generate_all_plots(
        args.checkpoint, args.output_dir,
        n_steps=args.steps, device=args.device,
    )
