"""
animation.py - MP4/GIF animasyon uretici

Icerik:
  - 2D slice animasyonu (matplotlib)
  - 3D volume animasyonu (PyVista, opsiyonel)
  - Diagnostik overlay (E, Z, Nu, t)

Kullanim:
  python -m visualize.animation --checkpoint results/checkpoints/checkpoint_epoch015000.pt
  python -m visualize.animation --checkpoint ... --field theta --fps 15 --steps 200

Gereksinim: pip install matplotlib (ffmpeg MP4 icin: brew install ffmpeg)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
import numpy as np
import torch

_this_dir = Path(__file__).resolve().parent
_project_dir = _this_dir.parent
sys.path.insert(0, str(_project_dir.parent))
sys.path.insert(0, str(_project_dir))

from config import Config
from model import INNATE3D_MixedConvection, ThermalFluidState


# =====================================================================
# 1. 2D slice animasyonu (matplotlib)
# =====================================================================


def animate_slice(
    checkpoint_path: str,
    field: str = "theta",
    slice_axis: str = "z",
    n_steps: int = 200,
    save_path: str = "results/animations/slice_animation.mp4",
    fps: int = 15,
    device: str = "cpu",
) -> None:
    """
    2D kesit animasyonu. Her frame bir forward step.

    Args:
        field: "theta", "u", "v", "w", "speed", "vorticity_mag"
        slice_axis: "x", "y", "z" -- hangi eksende kesit alinacak
        n_steps: toplam kare sayisi
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = Config.from_dict(ckpt["config"]) if "config" in ckpt else Config()
    config._device_override = device

    model = INNATE3D_MixedConvection(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dom = config.domain

    # Extent ve label'lar
    slice_config = {
        "z": {
            "idx": dom.Nz // 2, "dim": 3,
            "extent": [0, dom.Lx, 0, dom.Ly],
            "xlabel": "x", "ylabel": "y",
        },
        "y": {
            "idx": dom.Ny // 2, "dim": 2,
            "extent": [0, dom.Lx, 0, dom.Lz],
            "xlabel": "x", "ylabel": "z",
        },
        "x": {
            "idx": dom.Nx // 2, "dim": 1,
            "extent": [0, dom.Ly, 0, dom.Lz],
            "xlabel": "y", "ylabel": "z",
        },
    }
    sc = slice_config[slice_axis]

    # Field config
    field_config = {
        "theta": ("coolwarm", "T' (perturbation)"),
        "u": ("RdBu_r", "u (x-velocity)"),
        "v": ("RdBu_r", "v (y-velocity)"),
        "w": ("RdBu_r", "w (z-velocity)"),
        "speed": ("inferno", "|u| (speed)"),
        "vorticity_mag": ("magma", "|omega| (vorticity)"),
    }
    cmap, field_label = field_config.get(field, ("viridis", field))

    def _extract_field(state):
        """State'ten numpy field extract et. [Nx, Ny, Nz]"""
        if field == "speed":
            return np.sqrt(
                state.u[0].detach().cpu().numpy() ** 2
                + state.v[0].detach().cpu().numpy() ** 2
                + state.w[0].detach().cpu().numpy() ** 2
            )
        elif field == "vorticity_mag":
            return np.sqrt(
                state.omega_x[0].detach().cpu().numpy() ** 2
                + state.omega_y[0].detach().cpu().numpy() ** 2
                + state.omega_z[0].detach().cpu().numpy() ** 2
            )
        else:
            return getattr(state, field)[0].detach().cpu().numpy()

    def _take_slice(data_3d):
        """3D alandan 2D kesit al."""
        if slice_axis == "z":
            return data_3d[:, :, sc["idx"]].T
        elif slice_axis == "y":
            return data_3d[:, sc["idx"], :].T
        else:  # x
            return data_3d[sc["idx"], :, :].T

    # Seed sabitle -- pre-scan ve animasyon ayni IC'den baslasin
    seed = torch.seed()

    # Ilk frame icin vmin/vmax belirle -- kisa on-run
    print(f"Pre-scanning {min(20, n_steps)} steps for color range...")
    torch.manual_seed(seed)
    with torch.no_grad():
        state = model.create_initial_condition(batch_size=1, device=device)
        vmin, vmax = float("inf"), float("-inf")
        first_slice = None
        for si in range(min(20, n_steps)):
            state = model(state)
            d = _extract_field(state)
            sl = _take_slice(d)
            vmin = min(vmin, sl.min())
            vmax = max(vmax, sl.max())
            if si == 0:
                first_slice = sl
    margin = (vmax - vmin) * 0.1
    vmin -= margin
    vmax += margin

    # Animasyon -- ilk frame gercek veri ile
    print(f"Generating animation: {n_steps} frames @ {fps} fps...")
    fig, ax = plt.subplots(figsize=(10, 7))
    init_data = first_slice if first_slice is not None else np.zeros((10, 10))
    im = ax.imshow(
        init_data, origin="lower", cmap=cmap,
        extent=sc["extent"], vmin=vmin, vmax=vmax, aspect="auto",
    )
    ax.set_xlabel(sc["xlabel"])
    ax.set_ylabel(sc["ylabel"])
    cb = plt.colorbar(im, ax=ax, shrink=0.8)
    title = ax.set_title("")
    diag_text = ax.text(
        0.02, 0.98, "", transform=ax.transAxes, va="top", fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    # Writer secimi
    if manimation.writers.is_available('ffmpeg'):
        writer = manimation.FFMpegWriter(fps=fps, metadata={"title": field_label})
    else:
        print("  ffmpeg bulunamadi, GIF olarak kaydedilecek.")
        save_path = save_path.replace(".mp4", ".gif")
        writer = manimation.PillowWriter(fps=fps)

    torch.manual_seed(seed)
    with writer.saving(fig, save_path, dpi=120):
        with torch.no_grad():
            state = model.create_initial_condition(batch_size=1, device=device)
            for step in range(n_steps):
                state = model(state)

                data = _extract_field(state)
                sl = _take_slice(data)

                im.set_data(sl)
                t_val = state.t[0, 0].item()
                E = state.kinetic_energy().mean().item()
                Z = state.enstrophy().mean().item()

                title.set_text(
                    f"{field_label} | {slice_axis}={sc['idx']} slice | t={t_val:.3f}"
                )
                diag_text.set_text(f"E={E:.4f}  Z={Z:.4f}\nstep={step+1}/{n_steps}")

                writer.grab_frame()

                if (step + 1) % 50 == 0:
                    print(f"  Frame {step+1}/{n_steps}")

    plt.close(fig)
    print(f"  Saved: {save_path}")


# =====================================================================
# 2. Diagnostik zaman serisi animasyonu
# =====================================================================


def animate_diagnostics(
    checkpoint_path: str,
    n_steps: int = 200,
    save_path: str = "results/animations/diagnostics_animation.mp4",
    fps: int = 15,
    device: str = "cpu",
) -> None:
    """
    E(t), Z(t), Nu(t), div(t) canli gelisen grafik animasyonu.
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = Config.from_dict(ckpt["config"]) if "config" in ckpt else Config()
    config._device_override = device

    model = INNATE3D_MixedConvection(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    dt = config.physics.dt

    # Oncelikle tum veriyi topla
    print(f"Running {n_steps} steps...")
    E_list, Z_list, Nu_list, div_list = [], [], [], []
    with torch.no_grad():
        state = model.create_initial_condition(batch_size=1, device=device)
        for step in range(n_steps):
            state = model(state)
            E_list.append(state.kinetic_energy().mean().item())
            Z_list.append(state.enstrophy().mean().item())
            Nu_list.append(
                (1.0 + (state.v * state.theta).mean(dim=(-3, -2, -1))).mean().item()
            )
            div_list.append(
                model.ops.divergence(state.u, state.v, state.w).abs().max().item()
            )

    t_arr = np.arange(1, n_steps + 1) * dt

    # Animasyon
    print(f"Generating diagnostics animation...")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    panels = [
        (axes[0, 0], E_list, "E(t)", "#2196F3"),
        (axes[0, 1], Z_list, "Z(t)", "#F44336"),
        (axes[1, 0], Nu_list, "Nu(t)", "#4CAF50"),
        (axes[1, 1], div_list, "max|div|", "#FF9800"),
    ]

    lines = []
    for ax, data, label, color in panels:
        line, = ax.plot([], [], color=color, linewidth=1.5)
        ax.set_title(label)
        ax.set_xlabel("Time")
        ax.set_xlim(0, t_arr[-1])
        y_arr = np.array(data)
        margin = (y_arr.max() - y_arr.min()) * 0.1 + 1e-10
        if label == "max|div|":
            ax.set_yscale("log")
            ax.set_ylim(max(y_arr.min() * 0.1, 1e-15), y_arr.max() * 10)
        else:
            ax.set_ylim(y_arr.min() - margin, y_arr.max() + margin)
        ax.grid(True, alpha=0.3)
        lines.append(line)

    fig.tight_layout()

    if manimation.writers.is_available('ffmpeg'):
        writer = manimation.FFMpegWriter(fps=fps)
    else:
        save_path = save_path.replace(".mp4", ".gif")
        writer = manimation.PillowWriter(fps=fps)

    with writer.saving(fig, save_path, dpi=120):
        for i in range(1, n_steps + 1):
            for line, (_, data, _, _) in zip(lines, panels):
                line.set_data(t_arr[:i], data[:i])
            writer.grab_frame()

    plt.close(fig)
    print(f"  Saved: {save_path}")


# =====================================================================
# Entry point
# =====================================================================


def generate_all_animations(
    checkpoint_path: str,
    output_dir: str = "results/animations",
    n_steps: int = 200,
    fps: int = 15,
    device: str = "cpu",
):
    """Tum animasyonlari uret."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Theta slice animasyonu
    animate_slice(
        checkpoint_path, field="theta", slice_axis="z",
        n_steps=n_steps, fps=fps, device=device,
        save_path=str(out / "theta_z_slice.mp4"),
    )

    # 2. Speed slice animasyonu
    animate_slice(
        checkpoint_path, field="speed", slice_axis="z",
        n_steps=n_steps, fps=fps, device=device,
        save_path=str(out / "speed_z_slice.mp4"),
    )

    # 3. Vorticity slice
    animate_slice(
        checkpoint_path, field="vorticity_mag", slice_axis="z",
        n_steps=n_steps, fps=fps, device=device,
        save_path=str(out / "vorticity_z_slice.mp4"),
    )

    # 4. Diagnostik animasyon
    animate_diagnostics(
        checkpoint_path, n_steps=n_steps, fps=fps, device=device,
        save_path=str(out / "diagnostics.mp4"),
    )

    print(f"\nAll animations saved to {out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate animations")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/animations")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--field", type=str, default=None,
                        help="Tek alan: theta, speed, vorticity_mag, u, v, w")
    parser.add_argument("--slice-axis", type=str, default="z", choices=["x", "y", "z"])
    args = parser.parse_args()

    if args.field:
        animate_slice(
            args.checkpoint, field=args.field, slice_axis=args.slice_axis,
            n_steps=args.steps, fps=args.fps, device=args.device,
            save_path=str(Path(args.output_dir) / f"{args.field}_{args.slice_axis}_slice.mp4"),
        )
    else:
        generate_all_animations(
            args.checkpoint, args.output_dir,
            n_steps=args.steps, fps=args.fps, device=args.device,
        )
