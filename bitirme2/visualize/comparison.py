"""
comparison.py - DNS vs INNATE karsilastirma panelleri

Icerik:
  - Yan yana 2D kesit karsilastirma (DNS sol, INNATE sag, fark alt)
  - E(t) zaman serisi karsilastirma
  - Spektrum karsilastirma
  - Nusselt karsilastirma
  - Re/Ra sweep karsilastirma tablosu

Kullanim:
  python -m visualize.comparison --checkpoint results/checkpoints/checkpoint_epoch015000.pt
  python -m visualize.comparison --eval-json results/evaluation_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_this_dir = Path(__file__).resolve().parent
_project_dir = _this_dir.parent
sys.path.insert(0, str(_project_dir.parent))
sys.path.insert(0, str(_project_dir))

from config import Config
from model import INNATE3D_MixedConvection, ThermalFluidState


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
# 1. E(t) karsilastirma (evaluation report'tan)
# =====================================================================


def plot_energy_comparison(
    dns_E: List[float],
    innate_E: List[float],
    dt: float = 0.005,
    save_path: Optional[str] = None,
    title: str = "DNS vs INNATE Energy Comparison",
) -> plt.Figure:
    """
    Kinetik enerji zaman serisi karsilastirmasi.
    """
    _setup_style()
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[3, 1])

    n = min(len(dns_E), len(innate_E))
    t = np.arange(n) * dt

    # Ust panel: E(t)
    ax = axes[0]
    ax.plot(t, dns_E[:n], color="#333333", linewidth=2, label="DNS (reference)")
    ax.plot(t, innate_E[:n], color="#2196F3", linewidth=1.5, linestyle="--",
            label="INNATE")
    ax.set_xlabel("Time")
    ax.set_ylabel("Kinetic Energy E")
    ax.set_title(title)
    ax.legend()

    # Alt panel: relative error
    ax2 = axes[1]
    dns_arr = np.array(dns_E[:n])
    innate_arr = np.array(innate_E[:n])
    rel_error = np.abs(dns_arr - innate_arr) / (np.abs(dns_arr) + 1e-20)
    ax2.semilogy(t, rel_error, color="#F44336", linewidth=1.0)
    ax2.axhline(0.15, color="gray", linestyle="--", alpha=0.5, label="15% threshold")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Relative Error")
    ax2.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


# =====================================================================
# 2. Yan yana alan karsilastirma
# =====================================================================


def plot_field_comparison(
    dns_field: np.ndarray,
    innate_field: np.ndarray,
    field_name: str = "T'",
    extent: Optional[list] = None,
    save_path: Optional[str] = None,
    cmap: str = "coolwarm",
) -> plt.Figure:
    """
    3-panel karsilastirma: DNS | INNATE | Fark.

    Args:
        dns_field: 2D numpy array (bir kesit)
        innate_field: 2D numpy array (ayni kesit)
    """
    _setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    vmin = min(dns_field.min(), innate_field.min())
    vmax = max(dns_field.max(), innate_field.max())

    im0 = axes[0].imshow(dns_field.T, origin="lower", cmap=cmap, aspect="auto",
                          extent=extent, vmin=vmin, vmax=vmax)
    axes[0].set_title(f"DNS - {field_name}")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)

    im1 = axes[1].imshow(innate_field.T, origin="lower", cmap=cmap, aspect="auto",
                          extent=extent, vmin=vmin, vmax=vmax)
    axes[1].set_title(f"INNATE - {field_name}")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    diff = innate_field - dns_field
    abs_max = max(abs(diff.min()), abs(diff.max()), 1e-10)
    im2 = axes[2].imshow(diff.T, origin="lower", cmap="RdBu_r", aspect="auto",
                          extent=extent, vmin=-abs_max, vmax=abs_max)
    axes[2].set_title(f"Difference (INNATE - DNS)")
    plt.colorbar(im2, ax=axes[2], shrink=0.8)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    return fig


# =====================================================================
# 3. DNS vs INNATE rollout karsilastirma
# =====================================================================


def compare_dns_vs_innate(
    checkpoint_path: str,
    n_steps: int = 100,
    Re: float = 1000.0,
    Ra: float = 1e5,
    save_dir: str = "results/plots",
    device: str = "cpu",
) -> Dict:
    """
    DNS ve INNATE'i ayni IC'den calistir, karsilastirma grafikleri uret.

    Returns: karsilastirma metrikleri dict
    """
    from dns_reference import PseudoSpectralDNS3D_MixedConvection
    from train import compute_energy_spectrum

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = Config.from_dict(ckpt["config"]) if "config" in ckpt else Config()
    config._device_override = device

    dom = config.domain
    phys = config.physics

    # DNS (kucuk grid -- karsilastirma icin)
    print(f"Running DNS: Re={Re}, Ra={Ra}, {n_steps} steps...")
    dns = PseudoSpectralDNS3D_MixedConvection(
        Nx=dom.Nx, Ny=dom.Ny, Nz=dom.Nz,
        Lx=dom.Lx, Ly=dom.Ly, Lz=dom.Lz,
        Re=Re, Ra=Ra, Pr=phys.Pr,
        dt=phys.dt, device=device,
    )
    u_dns, v_dns, w_dns, theta_dns, p_dns = dns.create_initial_condition()
    dns_E, dns_Z = [], []

    def _dns_enstrophy(u, v, w, dns_solver):
        """Spectral curl ile enstrophy hesapla."""
        u_hat = dns_solver._fftn(u)
        v_hat = dns_solver._fftn(v)
        w_hat = dns_solver._fftn(w)
        ox_hat = 1j * (dns_solver.ky * w_hat - dns_solver.kz * v_hat)
        oy_hat = 1j * (dns_solver.kz * u_hat - dns_solver.kx * w_hat)
        oz_hat = 1j * (dns_solver.kx * v_hat - dns_solver.ky * u_hat)
        ox = dns_solver._ifftn(ox_hat).real
        oy = dns_solver._ifftn(oy_hat).real
        oz = dns_solver._ifftn(oz_hat).real
        return 0.5 * (ox ** 2 + oy ** 2 + oz ** 2).mean()

    with torch.no_grad():
        for step in range(n_steps):
            u_dns, v_dns, w_dns, theta_dns, p_dns = dns._rk4_step(
                u_dns, v_dns, w_dns, theta_dns
            )
            dns_E.append(0.5 * (u_dns ** 2 + v_dns ** 2 + w_dns ** 2).mean().item())
            dns_Z.append(_dns_enstrophy(u_dns, v_dns, w_dns, dns).item())

    # INNATE
    print(f"Running INNATE: {n_steps} steps...")
    model = INNATE3D_MixedConvection(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model.set_physics(Re, Ra)

    innate_E, innate_Z = [], []
    with torch.no_grad():
        state = model.create_initial_condition(batch_size=1, device=device)
        for step in range(n_steps):
            state = model(state)
            innate_E.append(state.kinetic_energy().mean().item())
            innate_Z.append(state.enstrophy().mean().item())

    # --- Grafikler ---

    # 1. E(t) karsilastirma
    plot_energy_comparison(
        dns_E, innate_E, dt=phys.dt,
        save_path=str(save_dir / f"compare_E_Re{Re:.0f}.png"),
        title=f"DNS vs INNATE: E(t) (Re={Re:.0f}, Ra={Ra:.0e})",
    )

    # 2. Z(t) karsilastirma
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    t = np.arange(len(dns_Z)) * phys.dt
    ax.plot(t, dns_Z, color="#333333", linewidth=2, label="DNS")
    ax.plot(t, innate_Z, color="#F44336", linewidth=1.5, linestyle="--", label="INNATE")
    ax.set_xlabel("Time")
    ax.set_ylabel("Enstrophy Z")
    ax.set_title(f"DNS vs INNATE: Z(t) (Re={Re:.0f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(str(save_dir / f"compare_Z_Re{Re:.0f}.png"), bbox_inches="tight")
    plt.close(fig)

    # 3. Son state slice karsilastirma (theta)
    # DNS shape: [1, Nx, Ny, Nz], INNATE shape: [B, Nx, Ny, Nz]
    dns_theta_slice = theta_dns[0].detach().cpu().numpy()[:, :, dom.Nz // 2]
    innate_theta_slice = state.theta[0].detach().cpu().numpy()[:, :, dom.Nz // 2]
    plot_field_comparison(
        dns_theta_slice, innate_theta_slice,
        field_name=f"T' (z-midplane, Re={Re:.0f})",
        extent=[0, dom.Lx, 0, dom.Ly],
        save_path=str(save_dir / f"compare_theta_slice_Re{Re:.0f}.png"),
    )

    # Metrikler
    dns_arr = np.array(dns_E)
    innate_arr = np.array(innate_E[:len(dns_E)])
    rel_error = np.abs(dns_arr - innate_arr) / (np.abs(dns_arr) + 1e-20)

    metrics = {
        "Re": Re,
        "Ra": Ra,
        "n_steps": n_steps,
        "mean_E_rel_error": float(rel_error.mean()),
        "max_E_rel_error": float(rel_error.max()),
        "final_E_rel_error": float(rel_error[-1]),
    }

    print(f"  Mean E relative error: {metrics['mean_E_rel_error']:.4f}")
    print(f"  Final E relative error: {metrics['final_E_rel_error']:.4f}")

    plt.close("all")
    return metrics


# =====================================================================
# 4. Evaluation report'tan grafikler
# =====================================================================


def plot_from_eval_report(
    report_path: str,
    save_dir: str = "results/plots",
) -> None:
    """
    evaluate.py ciktisi (JSON) uzerinden karsilastirma grafikleri uret.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    report = json.loads(Path(report_path).read_text())

    # 1. DNS vs INNATE E(t)
    ts = report.get("dns_time_series", {})
    dns_E = ts.get("dns_E", [])
    innate_E = ts.get("innate_E", [])
    if dns_E and innate_E:
        dt = report.get("config", {}).get("physics", {}).get("dt", 0.005)
        plot_energy_comparison(
            dns_E, innate_E, dt=dt,
            save_path=str(save_dir / "eval_dns_comparison.png"),
            title="Evaluation: DNS vs INNATE Energy",
        )

    # 2. Stability diagnostics
    diag = report.get("stability_diagnostics", {})
    if diag:
        from visualize.plots import plot_time_series
        dt = report.get("config", {}).get("physics", {}).get("dt", 0.005)
        plot_time_series(
            diag, dt=dt,
            save_path=str(save_dir / "eval_stability_timeseries.png"),
            title_prefix="Evaluation: ",
        )

    # 3. Summary box
    _setup_style()
    summary = report.get("summary", {})
    physics = report.get("physics_metrics", {})
    stability = report.get("stability", {})

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")

    def _fmt(val, fmt_str):
        """Guvenli format: float ise formatla, degilse 'N/A' dondur."""
        if isinstance(val, (int, float)):
            return f"{val:{fmt_str}}"
        return "N/A"

    text_lines = [
        f"Criteria passed: {summary.get('criteria_passed', '?')}/{summary.get('criteria_total', '?')}",
        f"Critical pass: {summary.get('all_critical_pass', '?')}",
        "",
        f"Divergence max: {_fmt(physics.get('divergence_max'), '.2e')}  "
        f"{'PASS' if physics.get('divergence_pass') else 'FAIL'}",
        f"Energy balance: {_fmt(physics.get('energy_balance_ratio'), '.4f')}  "
        f"{'PASS' if physics.get('energy_balance_pass') else 'FAIL'}",
        f"Spectrum slope: {_fmt(physics.get('spectrum_slope'), '.3f')}  "
        f"{'PASS' if physics.get('spectrum_slope_pass') else 'FAIL'}"
        if isinstance(physics.get('spectrum_slope'), (int, float)) else
        "Spectrum slope: N/A",
        f"Nusselt: {_fmt(physics.get('nusselt'), '.4f')}  "
        f"{'PASS' if physics.get('nusselt_pass') else 'FAIL'}",
        "",
        f"Stability: {'PASS' if stability.get('nan_free') else 'FAIL at step ' + str(stability.get('nan_step'))}",
        f"DNS error: {report.get('dns_comparison', {}).get('final_E_rel_error', '?')}",
    ]

    ax.text(0.1, 0.9, "\n".join(text_lines), transform=ax.transAxes,
            fontsize=11, verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))
    ax.set_title("Evaluation Summary", fontsize=14)

    fig.tight_layout()
    fig.savefig(str(save_dir / "eval_summary.png"), bbox_inches="tight")
    print(f"  Saved: {save_dir / 'eval_summary.png'}")
    plt.close("all")


# =====================================================================
# Entry point
# =====================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DNS vs INNATE comparison")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--eval-json", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/plots")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--Re", type=float, default=1000.0)
    parser.add_argument("--Ra", type=float, default=1e5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    if args.eval_json:
        plot_from_eval_report(args.eval_json, args.output_dir)
    elif args.checkpoint:
        compare_dns_vs_innate(
            args.checkpoint, n_steps=args.steps,
            Re=args.Re, Ra=args.Ra,
            save_dir=args.output_dir, device=args.device,
        )
    else:
        print("ERROR: --checkpoint veya --eval-json gerekli.")
