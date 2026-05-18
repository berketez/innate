"""
renderer3d.py - PyVista 3D gorselleştirme

Icerik:
  - Volume rendering (sicaklik, hiz buyuklugu)
  - Isosurface (Q-criterion, vorticity)
  - Streamlines (hiz alani)
  - Multi-panel gorunumler

Kullanim:
  python -m visualize.renderer3d --checkpoint results/checkpoints/checkpoint_epoch015000.pt
  python -m visualize.renderer3d --checkpoint ... --field theta --mode volume

Gereksinim: pip install pyvista
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_this_dir = Path(__file__).resolve().parent
_project_dir = _this_dir.parent
sys.path.insert(0, str(_project_dir.parent))
sys.path.insert(0, str(_project_dir))

from config import Config
from model import INNATE3D_MixedConvection, ThermalFluidState


def _check_pyvista():
    try:
        import pyvista as pv
        return pv
    except ImportError:
        print("ERROR: pyvista gerekli. Kur: pip install pyvista")
        sys.exit(1)


def state_to_grid(
    state: ThermalFluidState,
    config: Config,
) -> "pyvista.RectilinearGrid":
    """ThermalFluidState'i PyVista RectilinearGrid'e cevir."""
    pv = _check_pyvista()
    dom = config.domain

    x = np.linspace(0, dom.Lx, dom.Nx + 1)
    y = np.linspace(0, dom.Ly, dom.Ny + 1)
    z = np.linspace(0, dom.Lz, dom.Nz + 1)

    grid = pv.RectilinearGrid(x, y, z)

    # Alanlari cell data olarak ekle
    def _to_cell(tensor):
        return tensor[0].detach().cpu().numpy().flatten(order="F")

    grid.cell_data["theta"] = _to_cell(state.theta)
    grid.cell_data["u"] = _to_cell(state.u)
    grid.cell_data["v"] = _to_cell(state.v)
    grid.cell_data["w"] = _to_cell(state.w)
    grid.cell_data["p"] = _to_cell(state.p)

    # Hiz buyuklugu
    speed = np.sqrt(
        state.u[0].detach().cpu().numpy() ** 2
        + state.v[0].detach().cpu().numpy() ** 2
        + state.w[0].detach().cpu().numpy() ** 2
    )
    grid.cell_data["speed"] = speed.flatten(order="F")

    # Vorticity buyuklugu
    vort_mag = np.sqrt(
        state.omega_x[0].detach().cpu().numpy() ** 2
        + state.omega_y[0].detach().cpu().numpy() ** 2
        + state.omega_z[0].detach().cpu().numpy() ** 2
    )
    grid.cell_data["vorticity_mag"] = vort_mag.flatten(order="F")

    # Hiz vektoru
    vel = np.column_stack([
        _to_cell(state.u),
        _to_cell(state.v),
        _to_cell(state.w),
    ])
    grid.cell_data["velocity"] = vel

    return grid


def render_volume(
    state: ThermalFluidState,
    config: Config,
    field: str = "theta",
    save_path: Optional[str] = None,
    show: bool = False,
    cmap: str = "coolwarm",
    opacity: str = "sigmoid_5",
    window_size: tuple = (1200, 800),
) -> None:
    """
    3D volume rendering.

    Args:
        field: "theta", "speed", "vorticity_mag", "p"
        cmap: matplotlib colormap adi
        opacity: PyVista opacity transfer function
    """
    pv = _check_pyvista()
    pv.set_plot_theme("document")

    grid = state_to_grid(state, config)
    grid_point = grid.cell_data_to_point_data()

    pl = pv.Plotter(off_screen=not show, window_size=window_size)
    pl.add_volume(
        grid_point,
        scalars=field,
        cmap=cmap,
        opacity=opacity,
        shade=True,
    )
    pl.add_axes()
    t_val = state.t[0, 0].item()
    pl.add_text(f"{field} (t={t_val:.3f})", position="upper_left", font_size=12)

    if save_path:
        pl.screenshot(save_path)
        print(f"  Saved: {save_path}")
    if show:
        pl.show()
    pl.close()


def render_isosurface(
    state: ThermalFluidState,
    config: Config,
    field: str = "vorticity_mag",
    n_contours: int = 5,
    save_path: Optional[str] = None,
    show: bool = False,
    cmap: str = "magma",
    window_size: tuple = (1200, 800),
) -> None:
    """
    Isosurface (contour) rendering.

    Q-criterion veya vorticity buyuklugu icin ideal.
    """
    pv = _check_pyvista()
    pv.set_plot_theme("document")

    grid = state_to_grid(state, config)
    grid_point = grid.cell_data_to_point_data()

    pl = pv.Plotter(off_screen=not show, window_size=window_size)

    data_range = grid_point.get_data_range(field)
    if data_range[1] - data_range[0] < 1e-10:
        print(f"  WARNING: {field} alani neredeyse sabit, isosurface anlamsiz.")
        pl.close()
        return

    contours = grid_point.contour(
        n_contours,
        scalars=field,
    )
    if contours.n_points > 0:
        pl.add_mesh(contours, cmap=cmap, opacity=0.6)
    else:
        pl.add_text("No contour surfaces", position="upper_left")

    pl.add_axes()
    t_val = state.t[0, 0].item()
    pl.add_text(f"{field} isosurfaces (t={t_val:.3f})", position="upper_left", font_size=12)

    if save_path:
        pl.screenshot(save_path)
        print(f"  Saved: {save_path}")
    if show:
        pl.show()
    pl.close()


def render_streamlines(
    state: ThermalFluidState,
    config: Config,
    save_path: Optional[str] = None,
    show: bool = False,
    n_points: int = 200,
    window_size: tuple = (1200, 800),
) -> None:
    """
    Hiz alani streamline gorunumu.
    """
    pv = _check_pyvista()
    pv.set_plot_theme("document")

    grid = state_to_grid(state, config)
    grid_point = grid.cell_data_to_point_data()
    grid_point.set_active_vectors("velocity")

    pl = pv.Plotter(off_screen=not show, window_size=window_size)

    dom = config.domain
    seed = pv.Plane(
        center=(dom.Lx / 2, dom.Ly / 2, dom.Lz / 2),
        direction=(1, 0, 0),
        i_size=dom.Ly * 0.8,
        j_size=dom.Lz * 0.8,
        i_resolution=10,
        j_resolution=10,
    )

    try:
        streams = grid_point.streamlines_from_source(
            seed,
            vectors="velocity",
            max_time=dom.Lx * 2,
            max_steps=500,
            integration_direction="both",
        )
        if streams.n_points > 0:
            tube = streams.tube(radius=0.02)
            if "speed" not in tube.point_data and "velocity" in tube.point_data:
                vel = tube.point_data["velocity"]
                tube.point_data["speed"] = np.linalg.norm(vel, axis=1)
            scalars = "speed" if "speed" in tube.point_data else None
            pl.add_mesh(tube, cmap="viridis", scalars=scalars, opacity=0.8)
    except Exception as e:
        print(f"  Streamline hatasi: {e}")
        pl.add_text("Streamline generation failed", position="upper_left")

    pl.add_axes()
    t_val = state.t[0, 0].item()
    pl.add_text(f"Streamlines (t={t_val:.3f})", position="upper_left", font_size=12)

    if save_path:
        pl.screenshot(save_path)
        print(f"  Saved: {save_path}")
    if show:
        pl.show()
    pl.close()


def render_multi_panel(
    state: ThermalFluidState,
    config: Config,
    save_path: Optional[str] = None,
    show: bool = False,
    window_size: tuple = (1600, 1000),
) -> None:
    """
    2x2 panel: theta volume, speed volume, vorticity isosurface, streamlines.
    """
    pv = _check_pyvista()
    pv.set_plot_theme("document")

    grid = state_to_grid(state, config)
    grid_point = grid.cell_data_to_point_data()

    pl = pv.Plotter(shape=(2, 2), off_screen=not show, window_size=window_size)

    t_val = state.t[0, 0].item()

    # (0,0) Theta volume
    pl.subplot(0, 0)
    pl.add_volume(grid_point, scalars="theta", cmap="coolwarm",
                  opacity="sigmoid_5", shade=True)
    pl.add_text(f"Temperature T' (t={t_val:.3f})", font_size=10)
    pl.add_axes()

    # (0,1) Speed volume
    pl.subplot(0, 1)
    pl.add_volume(grid_point, scalars="speed", cmap="inferno",
                  opacity="sigmoid_5", shade=True)
    pl.add_text("Speed |u|", font_size=10)
    pl.add_axes()

    # (1,0) Vorticity isosurface
    pl.subplot(1, 0)
    try:
        contours = grid_point.contour(5, scalars="vorticity_mag")
        if contours.n_points > 0:
            pl.add_mesh(contours, cmap="magma", opacity=0.6)
    except Exception:
        pass
    pl.add_text("Vorticity |omega|", font_size=10)
    pl.add_axes()

    # (1,1) Pressure slice
    pl.subplot(1, 1)
    dom = config.domain
    sliced = grid_point.slice(normal="z", origin=(0, 0, dom.Lz / 2))
    if sliced.n_points > 0:
        pl.add_mesh(sliced, scalars="p", cmap="viridis")
    pl.add_text("Pressure (z-midplane)", font_size=10)
    pl.add_axes()

    if save_path:
        pl.screenshot(save_path)
        print(f"  Saved: {save_path}")
    if show:
        pl.show()
    pl.close()


# =====================================================================
# Entry point
# =====================================================================


def generate_3d_renders(
    checkpoint_path: str,
    output_dir: str = "results/plots",
    n_steps: int = 20,
    device: str = "cpu",
    show: bool = False,
):
    """Tum 3D render'lari uret."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = Config.from_dict(ckpt["config"]) if "config" in ckpt else Config()
    config._device_override = device

    model = INNATE3D_MixedConvection(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Running {n_steps} forward steps...")
    with torch.no_grad():
        state = model.create_initial_condition(batch_size=1, device=device)
        for _ in range(n_steps):
            state = model(state)

    print("Generating 3D renders...")
    render_volume(state, config, field="theta",
                  save_path=str(out / "3d_theta_volume.png"), show=show)
    render_volume(state, config, field="speed", cmap="inferno",
                  save_path=str(out / "3d_speed_volume.png"), show=show)
    render_isosurface(state, config, field="vorticity_mag",
                      save_path=str(out / "3d_vorticity_iso.png"), show=show)
    render_streamlines(state, config,
                       save_path=str(out / "3d_streamlines.png"), show=show)
    render_multi_panel(state, config,
                       save_path=str(out / "3d_multi_panel.png"), show=show)

    print(f"\nAll 3D renders saved to {out}/")


def _load_state(checkpoint_path, n_steps, device):
    """Checkpoint yukle ve n_steps ilerlet."""
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = Config.from_dict(ckpt["config"]) if "config" in ckpt else Config()
    config._device_override = device
    model = INNATE3D_MixedConvection(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    with torch.no_grad():
        state = model.create_initial_condition(batch_size=1, device=device)
        for _ in range(n_steps):
            state = model(state)
    return state, config


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="3D PyVista rendering")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/plots")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--field", type=str, default=None,
                        help="Tek alan render et: theta, speed, vorticity_mag")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "volume", "iso", "stream", "multi"])
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "all" and args.field is None:
        generate_3d_renders(
            args.checkpoint, args.output_dir,
            n_steps=args.steps, device=args.device, show=args.show,
        )
    else:
        state, config = _load_state(args.checkpoint, args.steps, args.device)
        field = args.field or "theta"
        mode = args.mode if args.mode != "all" else "volume"

        if mode == "volume":
            cmap = "inferno" if field == "speed" else ("magma" if "vort" in field else "coolwarm")
            render_volume(state, config, field=field, cmap=cmap,
                          save_path=str(out / f"3d_{field}_volume.png"), show=args.show)
        elif mode == "iso":
            render_isosurface(state, config, field=field,
                              save_path=str(out / f"3d_{field}_iso.png"), show=args.show)
        elif mode == "stream":
            render_streamlines(state, config,
                               save_path=str(out / "3d_streamlines.png"), show=args.show)
        elif mode == "multi":
            render_multi_panel(state, config,
                               save_path=str(out / "3d_multi_panel.png"), show=args.show)
