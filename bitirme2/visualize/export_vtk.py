"""
export_vtk.py - ParaView VTK export

Icerik:
  - Tek state'i VTK dosyasina export et
  - Rollout'u VTK serisi olarak export et (ParaView animasyon icin)
  - Tum alanlar dahil: u, v, w, p, theta, speed, vorticity_mag, rho

Kullanim:
  python -m visualize.export_vtk --checkpoint results/checkpoints/checkpoint_epoch015000.pt
  python -m visualize.export_vtk --checkpoint ... --steps 100 --interval 5

Gereksinim: pip install pyvista (veya vtk)
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


def _check_vtk():
    try:
        import pyvista as pv
        return pv
    except ImportError:
        try:
            import vtk
            from vtk.util.numpy_support import numpy_to_vtk
            return None  # vtk mevcut ama pyvista yok
        except ImportError:
            print("ERROR: pyvista veya vtk gerekli. Kur: pip install pyvista")
            sys.exit(1)


def state_to_vtk(
    state: ThermalFluidState,
    config: Config,
    save_path: str,
    time_value: Optional[float] = None,
) -> None:
    """
    ThermalFluidState'i VTK RectilinearGrid olarak kaydet.

    ParaView'da acilabilir. Tum skaler ve vektor alanlarini icerir.
    """
    pv = _check_vtk()
    if pv is None:
        _export_vtk_raw(state, config, save_path)
        return

    dom = config.domain

    x = np.linspace(0, dom.Lx, dom.Nx + 1)
    y = np.linspace(0, dom.Ly, dom.Ny + 1)
    z = np.linspace(0, dom.Lz, dom.Nz + 1)

    grid = pv.RectilinearGrid(x, y, z)

    def _add_field(name, tensor):
        grid.cell_data[name] = tensor[0].detach().cpu().numpy().flatten(order="F")

    # Skaler alanlar
    _add_field("theta", state.theta)
    _add_field("u", state.u)
    _add_field("v", state.v)
    _add_field("w", state.w)
    _add_field("p", state.p)

    # Turetilmis alanlar
    speed = np.sqrt(
        state.u[0].detach().cpu().numpy() ** 2
        + state.v[0].detach().cpu().numpy() ** 2
        + state.w[0].detach().cpu().numpy() ** 2
    )
    grid.cell_data["speed"] = speed.flatten(order="F")

    vort_mag = np.sqrt(
        state.omega_x[0].detach().cpu().numpy() ** 2
        + state.omega_y[0].detach().cpu().numpy() ** 2
        + state.omega_z[0].detach().cpu().numpy() ** 2
    )
    grid.cell_data["vorticity_mag"] = vort_mag.flatten(order="F")

    # Vektor alanlari
    vel = np.column_stack([
        state.u[0].detach().cpu().numpy().flatten(order="F"),
        state.v[0].detach().cpu().numpy().flatten(order="F"),
        state.w[0].detach().cpu().numpy().flatten(order="F"),
    ])
    grid.cell_data["velocity"] = vel

    vort = np.column_stack([
        state.omega_x[0].detach().cpu().numpy().flatten(order="F"),
        state.omega_y[0].detach().cpu().numpy().flatten(order="F"),
        state.omega_z[0].detach().cpu().numpy().flatten(order="F"),
    ])
    grid.cell_data["vorticity"] = vort

    # Yogunluk (Non-Boussinesq)
    if state.rho is not None:
        _add_field("rho", state.rho)

    # T_total (diagnostic)
    phys = config.physics
    y_grid = np.linspace(0, dom.Ly, dom.Ny, endpoint=False)
    T_base = phys.T_hot - (phys.dT / dom.Ly) * y_grid
    T_base_3d = np.broadcast_to(T_base[None, :, None], (dom.Nx, dom.Ny, dom.Nz))
    T_total = T_base_3d + state.theta[0].detach().cpu().numpy()
    grid.cell_data["T_total"] = T_total.flatten(order="F")

    # Zaman meta verisi
    if time_value is not None:
        grid.field_data["TimeValue"] = np.array([time_value])

    grid.save(save_path)
    print(f"  Saved VTK: {save_path}")


def _export_vtk_raw(state, config, save_path):
    """vtk paketi ile (pyvista olmadan) export -- fallback."""
    import vtk
    from vtk.util.numpy_support import numpy_to_vtk

    dom = config.domain
    grid = vtk.vtkRectilinearGrid()
    grid.SetDimensions(dom.Nx + 1, dom.Ny + 1, dom.Nz + 1)

    x = np.linspace(0, dom.Lx, dom.Nx + 1)
    y = np.linspace(0, dom.Ly, dom.Ny + 1)
    z = np.linspace(0, dom.Lz, dom.Nz + 1)

    grid.SetXCoordinates(numpy_to_vtk(x))
    grid.SetYCoordinates(numpy_to_vtk(y))
    grid.SetZCoordinates(numpy_to_vtk(z))

    for name, tensor in [
        ("theta", state.theta),
        ("u", state.u),
        ("v", state.v),
        ("w", state.w),
        ("p", state.p),
    ]:
        arr = numpy_to_vtk(tensor[0].detach().cpu().numpy().flatten(order="F"))
        arr.SetName(name)
        grid.GetCellData().AddArray(arr)

    writer = vtk.vtkXMLRectilinearGridWriter()
    writer.SetFileName(save_path)
    writer.SetInputData(grid)
    writer.Write()
    print(f"  Saved VTK (raw): {save_path}")


# =====================================================================
# Rollout export (VTK serisi)
# =====================================================================


def export_rollout(
    checkpoint_path: str,
    output_dir: str = "results/vtk",
    n_steps: int = 100,
    save_interval: int = 5,
    device: str = "cpu",
) -> None:
    """
    Model rollout'unu VTK serisi olarak export et.
    ParaView'da zaman animasyonu icin ideal.

    Dosya isimlendirme: state_000000.vtr, state_000005.vtr, ...
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
    config = Config.from_dict(ckpt["config"]) if "config" in ckpt else Config()
    config._device_override = device

    model = INNATE3D_MixedConvection(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Exporting {n_steps} steps (interval={save_interval})...")

    pvd_entries = []

    with torch.no_grad():
        state = model.create_initial_condition(batch_size=1, device=device)
        state_to_vtk(state, config,
                      str(out / f"state_{0:06d}.vtr"),
                      time_value=0.0)
        pvd_entries.append((0.0, f"state_{0:06d}.vtr"))

        for step in range(1, n_steps + 1):
            state = model(state)

            if step % save_interval == 0:
                t_val = state.t[0, 0].item()
                fname = f"state_{step:06d}.vtr"
                state_to_vtk(state, config, str(out / fname), time_value=t_val)
                pvd_entries.append((t_val, fname))

    # PVD dosyasi (ParaView zaman serisi icin)
    _write_pvd(out / "time_series.pvd", pvd_entries)
    print(f"\nVTK series saved to {out}/ ({len(pvd_entries)} files)")
    print(f"  ParaView: Open {out / 'time_series.pvd'}")


def _write_pvd(path: Path, entries: list) -> None:
    """ParaView PVD (collection) dosyasi yaz."""
    lines = [
        '<?xml version="1.0"?>',
        '<VTKFile type="Collection" version="0.1">',
        '  <Collection>',
    ]
    for t_val, fname in entries:
        lines.append(f'    <DataSet timestep="{t_val:.6f}" file="{fname}"/>')
    lines.append('  </Collection>')
    lines.append('</VTKFile>')
    path.write_text("\n".join(lines))
    print(f"  Saved PVD: {path}")


# =====================================================================
# Entry point
# =====================================================================


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export VTK files")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/vtk")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--interval", type=int, default=5,
                        help="Her kac step'te VTK kaydet")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--single", action="store_true",
                        help="Tek frame export et (rollout yerine)")
    parser.add_argument("--single-steps", type=int, default=20,
                        help="--single modunda kac step ilerlet")
    args = parser.parse_args()

    if args.single:
        ckpt = torch.load(args.checkpoint, weights_only=False, map_location=args.device)
        config = Config.from_dict(ckpt["config"]) if "config" in ckpt else Config()
        config._device_override = args.device

        model = INNATE3D_MixedConvection(config).to(args.device)
        model.load_state_dict(ckpt["model"])
        model.eval()

        with torch.no_grad():
            state = model.create_initial_condition(batch_size=1, device=args.device)
            for _ in range(args.single_steps):
                state = model(state)

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        state_to_vtk(
            state, config,
            str(Path(args.output_dir) / "single_state.vtr"),
            time_value=state.t[0, 0].item(),
        )
    else:
        export_rollout(
            args.checkpoint, args.output_dir,
            n_steps=args.steps, save_interval=args.interval,
            device=args.device,
        )
