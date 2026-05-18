"""
04_render_3d.py — ANSYS Fluent / SimScale tarzı 4K@60fps 3D CFD görselleştirme

KRİTİK FİZİK:
- LES solver: theta = T - T_bar(y) PERTURBATION (dimensional °C, range ±0.04°C)
- Y AXIS VERTICAL (Ri*theta*e_y, dT/Ly)
- Hot wall: y=0, T_hot=20°C
- Cold wall: y=Ly=10, T_cold=0°C
- T_total(x,y,z) = T_hot - dT*(y/Ly) + theta(x,y,z)  → range [0°C, 20°C]

KOMPOZİSYON (tam ekran 3D + minimal üst metadata):
- Üst 200px metadata bar (Re, Ra, Cs, t, TKE, Nu)
- 3840×1960 tam ekran 3D scene:
  * Volume rendering T_total (dimensional °C, ANSYS thermal cmap)
  * Q-criterion isosurface (multi-level p98 + p90, theta-colored)
  * Streamlines tube (velocity-magnitude colored, viridis)
  * Domain wireframe box (gri saydam)
  * Hot wall slab (parlak kırmızı, y=0, label "Sıcak Duvar T=20°C")
  * Cold wall slab (parlak mavi, y=Ly, label "Soğuk Duvar T=0°C")
  * Dimensional colorbar 0-20°C (sağ taraf)
  * 3-light Phong + background gradient
- Y-axis up rotation (XZ plane azimuth orbit)

Test modu: --single-frame IDX (PNG kaydet)
"""
from __future__ import annotations
import sys, os, gc, argparse, time
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

os.environ.setdefault("PYVISTA_USE_PANEL", "false")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import imageio.v2 as imageio
import pyvista as pv

pv.OFF_SCREEN = True


# Boussinesq fiziksel sabitler
T_HOT = 20.0      # °C (alt duvar y=0)
T_COLD = 0.0      # °C (üst duvar y=Ly)
DT = T_HOT - T_COLD


# ============================================================
# ANSYS thermal colormap (coolwarm benzeri ama daha kontrastlı)
# ============================================================
def ansys_thermal_cmap():
    return LinearSegmentedColormap.from_list(
        "ansys_thermal",
        [
            (0.00, "#08306b"),
            (0.18, "#2171b5"),
            (0.35, "#00c8ff"),
            (0.50, "#2ca25f"),
            (0.68, "#ffff33"),
            (0.84, "#f03b20"),
            (1.00, "#67000d"),
        ],
        N=256,
    )


# ============================================================
# Q-criterion (gerçek tensor invariant)
# ============================================================
def q_criterion(u, v, w, dx, dy, dz):
    du_dx = np.gradient(u, dx, axis=0, edge_order=2)
    du_dy = np.gradient(u, dy, axis=1, edge_order=2)
    du_dz = np.gradient(u, dz, axis=2, edge_order=2)
    dv_dx = np.gradient(v, dx, axis=0, edge_order=2)
    dv_dy = np.gradient(v, dy, axis=1, edge_order=2)
    dv_dz = np.gradient(v, dz, axis=2, edge_order=2)
    dw_dx = np.gradient(w, dx, axis=0, edge_order=2)
    dw_dy = np.gradient(w, dy, axis=1, edge_order=2)
    dw_dz = np.gradient(w, dz, axis=2, edge_order=2)

    Sxx, Syy, Szz = du_dx, dv_dy, dw_dz
    Sxy = 0.5 * (du_dy + dv_dx)
    Sxz = 0.5 * (du_dz + dw_dx)
    Syz = 0.5 * (dv_dz + dw_dy)

    Oxy = 0.5 * (du_dy - dv_dx)
    Oxz = 0.5 * (du_dz - dw_dx)
    Oyz = 0.5 * (dv_dz - dw_dy)

    S2 = Sxx**2 + Syy**2 + Szz**2 + 2.0 * (Sxy**2 + Sxz**2 + Syz**2)
    O2 = 2.0 * (Oxy**2 + Oxz**2 + Oyz**2)
    return 0.5 * (O2 - S2)


# ============================================================
# Total temperature reconstruction
# Boussinesq: T(x,y,z) = T_hot - dT*(y/Ly) + theta(x,y,z)
# ============================================================
def reconstruct_T_total(theta_field, Ly):
    """theta (dimensionless perturbation) -> dimensional total T [°C].

    LES solver boyutsuz formülasyon (les_solver.py line 144):
        dT_over_Ly = 1.0 / Ly  (dT kullanılmıyor source'ta)
    Yani theta boyutsuz, range [-0.5, +0.5] (clamp).
    Reconstruction: T = T_bar(y) + dT * theta_dimensionless
    """
    Nx, Ny, Nz = theta_field.shape
    y_centers = (np.arange(Ny) + 0.5) * (Ly / Ny)
    T_bar = T_HOT - DT * (y_centers / Ly)  # 20 -> 0 °C
    T_bar_3d = T_bar[None, :, None]
    return T_bar_3d + DT * theta_field


# ============================================================
# Grid builder (cell -> point conversion ŞART, Y vertical)
# ============================================================
def build_full_grid(u, v, w, theta, Q, Lx, Ly, Lz):
    """Tek grid: T_total + theta + velocity + Q (cell_data → point_data)."""
    nx, ny, nz = theta.shape
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    grid = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=(dx, dy, dz),
        origin=(0.0, 0.0, 0.0),
    )

    T_total = reconstruct_T_total(theta, Ly)

    grid.cell_data["T_C"] = T_total.ravel(order="F")
    grid.cell_data["theta"] = theta.ravel(order="F")
    grid.cell_data["Q"] = Q.ravel(order="F")
    grid.cell_data["velocity"] = np.column_stack([
        u.ravel(order="F"),
        v.ravel(order="F"),
        w.ravel(order="F"),
    ])

    grid = grid.cell_data_to_point_data(pass_cell_data=False)
    grid.point_data["speed"] = np.linalg.norm(grid.point_data["velocity"], axis=1)
    return grid


# ============================================================
# Streamline seeds (Y vertical, hot wall yakını + Q-peak hedefli)
# ============================================================
def make_streamline_seeds(Q, Lx, Ly, Lz, n_uniform=(6, 10, 4), n_targeted=40):
    nu_x, nu_y, nu_z = n_uniform
    xs = np.linspace(0.08 * Lx, 0.92 * Lx, nu_x)
    ys = np.linspace(0.08 * Ly, 0.92 * Ly, nu_y)
    zs = np.linspace(0.08 * Lz, 0.92 * Lz, nu_z)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    uniform = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    # Hot wall yakınından (y küçük) ekstra seed — plume tracking
    nx, ny, nz = Q.shape
    n_hot_seed = 30
    hot_x = np.random.uniform(0.1 * Lx, 0.9 * Lx, n_hot_seed)
    hot_y = np.random.uniform(0.05 * Ly, 0.20 * Ly, n_hot_seed)
    hot_z = np.random.uniform(0.1 * Lz, 0.9 * Lz, n_hot_seed)
    hot_seeds = np.column_stack([hot_x, hot_y, hot_z])

    # Q-peak targeted
    Q_pos = np.where(Q > 0, Q, 0)
    flat_ids = np.argpartition(Q_pos.ravel(), -n_targeted)[-n_targeted:]
    ix, iy, iz = np.unravel_index(flat_ids, Q.shape)
    targeted = np.column_stack([
        (ix + 0.5) * Lx / nx,
        (iy + 0.5) * Ly / ny,
        (iz + 0.5) * Lz / nz,
    ])

    return pv.PolyData(np.vstack([uniform, hot_seeds, targeted]))


# ============================================================
# Camera (Y vertical, up=(0,1,0), XZ orbit)
# ============================================================
def set_camera_orbit(plotter, focal, az_deg, el_deg=20.0, radius=None,
                     domain_size=None):
    """Y-up rotation: kamera XZ planında dönsün, Y biraz yukarıda."""
    if radius is None:
        radius = max(domain_size) * 1.6 if domain_size else 4.2
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    pos = focal + radius * np.array([
        np.cos(az) * np.cos(el),
        np.sin(el),
        np.sin(az) * np.cos(el),
    ])
    plotter.camera.position = tuple(pos)
    plotter.camera.focal_point = tuple(focal)
    plotter.camera.up = (0.0, 1.0, 0.0)
    plotter.camera.view_angle = 28
    plotter.camera.clipping_range = (0.01, 100.0)


# ============================================================
# Wall surfaces (parlak, opak, Y horizontal duvarlar)
# ============================================================
def add_wall_surfaces(plotter, Lx, Ly, Lz):
    """Hot wall y=0 (colormap T=20 rengi), Cold wall y=Ly (colormap T=0 rengi).
    Renkler ANSYS thermal cmap uçlarıyla TAM EŞLEŞİR — colorbar tutarlı."""
    eps = 0.04 * Ly

    hot_wall = pv.Box(bounds=(0, Lx, -eps, 0.0, 0, Lz))
    cold_wall = pv.Box(bounds=(0, Lx, Ly, Ly + eps, 0, Lz))

    # Colormap node renkleri (ansys_thermal_cmap):
    # 0.00 → #08306b (cold dark blue, 0°C)
    # 1.00 → #67000d (hot dark red, 20°C)
    plotter.add_mesh(
        hot_wall, color="#67000d", opacity=1.0,
        smooth_shading=True,
        ambient=0.55, diffuse=0.50,
        specular=0.55, specular_power=30,
    )
    plotter.add_mesh(
        cold_wall, color="#08306b", opacity=1.0,
        smooth_shading=True,
        ambient=0.55, diffuse=0.50,
        specular=0.55, specular_power=30,
    )


def add_domain_box(plotter, Lx, Ly, Lz):
    box = pv.Box(bounds=(0, Lx, 0, Ly, 0, Lz))
    plotter.add_mesh(
        box, style="wireframe", color="#22D3EE",
        opacity=0.55, line_width=2.5,
    )


# ============================================================
# Wall labels (3D dünya koordinatları, Türkçe)
# ============================================================
def add_wall_labels(plotter, Lx, Ly, Lz):
    """Wall labels — wall slab'ın dışına yeterince offset, always_visible."""
    label_offset = 0.10 * Ly

    plotter.add_point_labels(
        points=np.array([[Lx/2, -label_offset, Lz/2]]),
        labels=["Sicak Duvar T = 20 C"],
        text_color="#FFFFFF", font_size=32, bold=True,
        show_points=False, always_visible=True, shadow=True,
        shape="rect", fill_shape=True, shape_color="#B00010",
        shape_opacity=0.92, margin=14, font_family="arial",
    )
    plotter.add_point_labels(
        points=np.array([[Lx/2, Ly + label_offset, Lz/2]]),
        labels=["Soguk Duvar T = 0 C"],
        text_color="#FFFFFF", font_size=32, bold=True,
        show_points=False, always_visible=True, shadow=True,
        shape="rect", fill_shape=True, shape_color="#0A4990",
        shape_opacity=0.92, margin=14, font_family="arial",
    )


# ============================================================
# 3-light Phong setup
# ============================================================
def add_lighting(plotter, focal):
    plotter.remove_all_lights()

    def add_light(pos, intensity, color="white"):
        light = pv.Light(position=tuple(pos), focal_point=tuple(focal),
                         color=color, light_type="scene light")
        light.intensity = intensity
        light.positional = False
        plotter.add_light(light)

    fx, fy, fz = focal
    add_light((fx,        fy + 12.0, fz - 8.0), 0.85)  # key (top-front)
    add_light((fx - 8.0,  fy + 4.0,  fz),       0.40)  # fill (side)
    add_light((fx + 6.0,  fy + 8.0,  fz + 8.0), 0.55)  # rim (back)


# ============================================================
# Multi-process precompute (Tier 1 optimizasyon)
# ============================================================
def _precompute_worker(args):
    """Worker: tek snapshot için Q-criterion + Q-iso + streamlines hesabı.
    Geri dönüş numpy array (pickle güvenli). Ana process mesh inşa eder."""
    idx, snap, Lx, Ly, Lz, q_strong_p, q_medium_p = args

    u, v, w, theta = snap[0], snap[1], snap[2], snap[3]
    nx, ny, nz = theta.shape
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    Q = q_criterion(u, v, w, dx, dy, dz)
    grid = build_full_grid(u, v, w, theta, Q, Lx, Ly, Lz)

    # Q-iso multi-level (numpy point + face arrays)
    Qp = grid.point_data["Q"]
    Qpos = Qp[(Qp > 0) & np.isfinite(Qp)]

    iso_strong_data = None
    iso_medium_data = None
    if Qpos.size > 100:
        q_s = float(np.percentile(Qpos, q_strong_p))
        q_m = float(np.percentile(Qpos, q_medium_p))
        try:
            iso_s = grid.contour(isosurfaces=[q_s], scalars="Q")
            if iso_s.n_points > 100:
                iso_strong_data = {
                    "points": np.asarray(iso_s.points, dtype=np.float32),
                    "faces": np.asarray(iso_s.faces, dtype=np.int64),
                    "theta": np.asarray(iso_s.point_data["theta"], dtype=np.float32),
                }
        except Exception:
            pass
        try:
            iso_m = grid.contour(isosurfaces=[q_m], scalars="Q")
            if iso_m.n_points > 100:
                iso_medium_data = {
                    "points": np.asarray(iso_m.points, dtype=np.float32),
                    "faces": np.asarray(iso_m.faces, dtype=np.int64),
                    "theta": np.asarray(iso_m.point_data["theta"], dtype=np.float32),
                }
        except Exception:
            pass

    # Streamlines tubes
    tubes_data = None
    try:
        seeds = make_streamline_seeds(Q, Lx, Ly, Lz)
        stream = grid.streamlines_from_source(
            seeds, vectors="velocity",
            integration_direction="both",
            max_length=2.5, terminal_speed=1e-5,
            max_step_length=0.020, initial_step_length=0.005,
        )
        if stream.n_points > 0:
            tubes = stream.tube(radius=0.012, n_sides=8)
            if tubes.n_points > 0:
                tubes_data = {
                    "points": np.asarray(tubes.points, dtype=np.float32),
                    "faces": np.asarray(tubes.faces, dtype=np.int64),
                    "speed": np.asarray(tubes.point_data["speed"], dtype=np.float32),
                }
    except Exception:
        pass

    # Volume için T_C array (cell-centered)
    T_C_cell = reconstruct_T_total(theta, Ly).astype(np.float32)

    # Cell→point conversion (precompute'da bir kez, render loop'ta tekrar yapılmasın)
    tmp = pv.ImageData(
        dimensions=(nx + 1, ny + 1, nz + 1),
        spacing=(dx, dy, dz),
        origin=(0.0, 0.0, 0.0),
    )
    tmp.cell_data["T_C"] = T_C_cell.ravel(order="F")
    tmp_p = tmp.cell_data_to_point_data(pass_cell_data=False)
    T_C_point = np.asarray(tmp_p.point_data["T_C"], dtype=np.float32).copy()

    return idx, {
        "T_C_point": T_C_point,
        "iso_strong": iso_strong_data,
        "iso_medium": iso_medium_data,
        "tubes": tubes_data,
    }


def precompute_all_geom(full_snaps, Lx, Ly, Lz, q_strong_p, q_medium_p,
                        n_workers=None):
    """ProcessPool ile tüm snapshot'ları paralel precompute."""
    from concurrent.futures import ProcessPoolExecutor

    n_snaps = full_snaps.shape[0]
    if n_workers is None:
        n_workers = min(8, os.cpu_count() or 4)

    print(f"\n[PRECOMPUTE] {n_snaps} snapshot, {n_workers} worker")
    args_list = [
        (i, full_snaps[i], Lx, Ly, Lz, q_strong_p, q_medium_p)
        for i in range(n_snaps)
    ]

    cache = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for done_count, (idx, data) in enumerate(ex.map(_precompute_worker, args_list), 1):
            cache[idx] = data
            if done_count % 10 == 0 or done_count == n_snaps:
                elapsed = time.time() - t0
                rate = done_count / elapsed
                eta = (n_snaps - done_count) / max(rate, 1e-3)
                print(f"  precomputed {done_count}/{n_snaps}  "
                      f"({rate:.1f}/s, eta {eta:.0f}s)")

    print(f"[PRECOMPUTE] {time.time() - t0:.1f}s ({n_snaps / (time.time() - t0):.1f}/s)")
    return cache


def cached_to_meshes(cached):
    """Cached numpy array'leri → PyVista PolyData mesh'leri (ana process)."""
    iso_strong = None
    if cached["iso_strong"] is not None:
        d = cached["iso_strong"]
        m = pv.PolyData(d["points"], faces=d["faces"])
        m.point_data["theta"] = d["theta"]
        iso_strong = m

    iso_medium = None
    if cached["iso_medium"] is not None:
        d = cached["iso_medium"]
        m = pv.PolyData(d["points"], faces=d["faces"])
        m.point_data["theta"] = d["theta"]
        iso_medium = m

    tubes = None
    if cached["tubes"] is not None:
        d = cached["tubes"]
        m = pv.PolyData(d["points"], faces=d["faces"])
        m.point_data["speed"] = d["speed"]
        tubes = m

    return iso_strong, iso_medium, tubes


# ============================================================
# Tek snapshot pipeline (legacy single-frame test için)
# ============================================================
def render_scene(plotter, snap, Lx, Ly, Lz, q_percentiles=(95, 88),
                 stream_seeds=(6, 10, 4), stream_target=40,
                 stream_max_time=2.5, stream_radius=0.012):
    """
    snap: (4, Nx, Ny, Nz) — u, v, w, theta
    Returns: (actor_list, T_max, T_min, q_iso_levels)
    """
    u, v, w, theta = snap[0], snap[1], snap[2], snap[3]
    nx, ny, nz = theta.shape
    dx, dy, dz = Lx / nx, Ly / ny, Lz / nz

    Q = q_criterion(u, v, w, dx, dy, dz)
    grid = build_full_grid(u, v, w, theta, Q, Lx, Ly, Lz)
    cmap = ansys_thermal_cmap()

    actors = []

    # ============ Volume rendering T_total (0-20°C) ============
    # ImageData korunur (clip_box UnstructuredGrid yapar → CPU rendering)
    # GPU mapper zorla → RTX 4090 hızlı ray-casting
    # Opacity orta band çok şeffaf (0.018) → Q-iso ve walls görünür
    sigma_otf = [0.32, 0.18, 0.055, 0.018, 0.055, 0.18, 0.36]
    vol_actor = plotter.add_volume(
        grid, scalars="T_C", cmap=cmap, clim=(0.0, 20.0),
        opacity=sigma_otf, shade=False,
        ambient=0.30, diffuse=0.65, specular=0.30,
        mapper="gpu", show_scalar_bar=False,
    )
    actors.append(vol_actor)

    # ============ Dimensional colorbar (sağ taraf) ============
    try:
        sb = plotter.add_scalar_bar(
            title="Sicaklik [C]",
            mapper=vol_actor.mapper,
            n_labels=5, fmt="%.0f",
            title_font_size=26, label_font_size=22,
            position_x=0.91, position_y=0.16,
            width=0.060, height=0.70, vertical=True,
            color="white", italic=False, bold=True,
            font_family="arial", outline=True, fill=True,
        )
    except Exception as e:
        print(f"  [WARN] scalar bar: {e}")

    # ============ Multi-level Q-criterion isosurface ============
    Qp = grid.point_data["Q"]
    Qpos = Qp[(Qp > 0) & np.isfinite(Qp)]
    q_levels_used = []
    if Qpos.size > 100:
        q_p_strong = float(np.percentile(Qpos, q_percentiles[0]))
        q_p_medium = float(np.percentile(Qpos, q_percentiles[1]))

        for q_val, color, op in [
            (q_p_strong, "#8B0000", 0.85),
            (q_p_medium, "#FF8C00", 0.50),
        ]:
            try:
                q_iso = grid.contour(isosurfaces=[q_val], scalars="Q")
                if q_iso.n_points > 100:
                    actors.append(plotter.add_mesh(
                        q_iso, color=color, opacity=op,
                        smooth_shading=True,
                        ambient=0.20, diffuse=0.55,
                        specular=0.55, specular_power=25,
                        show_scalar_bar=False,
                    ))
                    q_levels_used.append(q_val)
            except Exception as e:
                print(f"  [WARN] Q-iso skip ({q_val}): {e}")

    # ============ Streamlines ============
    try:
        seeds = make_streamline_seeds(Q, Lx, Ly, Lz,
                                      n_uniform=stream_seeds,
                                      n_targeted=stream_target)
        stream = grid.streamlines_from_source(
            seeds, vectors="velocity",
            integration_direction="both",
            max_length=stream_max_time,
            terminal_speed=1e-5,
            max_step_length=0.020,
            initial_step_length=0.005,
        )
        if stream.n_points > 0:
            tubes = stream.tube(radius=stream_radius, n_sides=8)
            actors.append(plotter.add_mesh(
                tubes, scalars="speed", cmap="viridis",
                opacity=0.90, smooth_shading=True,
                show_scalar_bar=False,
                ambient=0.20, diffuse=0.55,
                specular=0.45, specular_power=20,
            ))
    except Exception as e:
        print(f"  [WARN] Streamline skip: {e}")

    T_field = grid.point_data["T_C"]
    return actors, float(T_field.max()), float(T_field.min()), q_levels_used


# ============================================================
# Üst metadata bar (200px) — PIL ile hızlı text composit (matplotlib YOK)
# ============================================================
from PIL import Image, ImageDraw, ImageFont

# Font cache (her frame'de yeniden yüklenmesin)
_FONT_CACHE = {}

def _get_font(size, bold=False):
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidates = [
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    font = None
    for path in candidates:
        try:
            if Path(path).exists():
                font = ImageFont.truetype(path, size)
                break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def compose_with_header(pv_img, t_now, TKE, Nu, Re, Ra, Cs,
                        label="LES Referansi"):
    """Hızlı header composit — PIL Draw, matplotlib YOK.
    Frame başına ~3 sn → ~0.05 sn."""
    pv_img = np.asarray(pv_img)[..., :3].astype(np.uint8)
    H_render, W = pv_img.shape[:2]
    header_px = 200

    # Header bandı (#141420 koyu)
    header = np.full((header_px, W, 3), 20, dtype=np.uint8)
    header[..., 0] = 20; header[..., 1] = 20; header[..., 2] = 32

    img = Image.fromarray(header)
    draw = ImageDraw.Draw(img)

    title_font = _get_font(60, bold=True)
    meta_font = _get_font(40, bold=False)

    # Sol başlık
    draw.text((40, header_px // 2 - 30), label,
              fill=(255, 255, 255), font=title_font)

    # Sağ metadata
    meta = (f"Re={Re:.0f}   Ra={Ra:.0e}   Cs={Cs:.2f}   "
            f"t={t_now:.1f}   TKE={TKE:.4f}   Nu={Nu:.2f}")
    bbox = draw.textbbox((0, 0), meta, font=meta_font)
    meta_w = bbox[2] - bbox[0]
    draw.text((W - meta_w - 40, header_px // 2 - 20), meta,
              fill=(232, 232, 238), font=meta_font)

    header_arr = np.asarray(img)
    return np.vstack([header_arr, pv_img])


# ============================================================
# Plotter setup (statik elemanlar)
# ============================================================
def setup_plotter(window_size, Lx, Ly, Lz, focal):
    plotter = pv.Plotter(off_screen=True, window_size=window_size, lighting="none")
    plotter.set_background(color="#000000", top="#071A2D", all_renderers=True)

    add_domain_box(plotter, Lx, Ly, Lz)
    add_wall_surfaces(plotter, Lx, Ly, Lz)
    add_wall_labels(plotter, Lx, Ly, Lz)
    add_lighting(plotter, focal)

    return plotter


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="ANSYS-tarzi 3D CFD render (4K@60fps)")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--label", type=str, default="LES Referansi (Smagorinsky SGS)")
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--bitrate", type=str, default="40M")
    parser.add_argument("--no-rotate", action="store_true")
    parser.add_argument("--q-strong", type=float, default=95.0)
    parser.add_argument("--q-medium", type=float, default=88.0)
    parser.add_argument("--single-frame", type=int, default=-1)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--workers", type=int, default=None,
                        help="ProcessPool worker sayısı (default: min(8, cpu))")
    parser.add_argument("--no-nvenc", action="store_true",
                        help="GPU encoding (h264_nvenc) kapat, libx264 kullan")
    args = parser.parse_args()

    rotate = (not args.no_rotate)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("3D ANSYS-tarzi render basliyor (Y-vertical, T_total reconstruction)")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {output}")

    d = np.load(args.input, allow_pickle=True)
    full_snaps = d["full_snaps"]
    full_times = d["full_times"]
    TKE_arr = d["metric_TKE"]
    Nu_arr = d["metric_Nu"]
    Lx = float(d["Lx"]); Ly = float(d["Ly"]); Lz = float(d["Lz"])

    def _meta(key, default):
        try:
            if hasattr(d, "files") and key in d.files:
                return float(d[key])
        except Exception:
            pass
        return default

    Re_val = _meta("Re", 10000.0)
    Ra_val = _meta("Ra", 1e5)
    Cs_val = _meta("Cs", 0.17)

    n_snaps = full_snaps.shape[0]
    Nx, Ny, Nz = full_snaps.shape[2], full_snaps.shape[3], full_snaps.shape[4]
    print(f"  Snapshots: {n_snaps}, grid: ({Nx},{Ny},{Nz})")
    print(f"  Domain: {Lx} x {Ly} x {Lz} (Y vertical)")
    print(f"  T_hot={T_HOT}C (y=0), T_cold={T_COLD}C (y={Ly})")
    print(f"  Re={Re_val:.0f}  Ra={Ra_val:.1e}  Cs={Cs_val:.3f}")
    print(f"  Time range: {full_times[0]:.2f} -> {full_times[-1]:.2f}")

    focal = np.array([Lx / 2, Ly / 2, Lz / 2])

    # ------------------------------------------------------------
    # SINGLE FRAME TEST
    # ------------------------------------------------------------
    if args.single_frame >= 0:
        idx = min(args.single_frame, n_snaps - 1)
        print(f"\n[TEST] Single frame: snapshot {idx}, t={full_times[idx]:.2f}")

        plotter = setup_plotter((1920, 1080), Lx, Ly, Lz, focal)
        actors, T_max, T_min, q_levels = render_scene(
            plotter, full_snaps[idx], Lx, Ly, Lz,
            q_percentiles=(args.q_strong, args.q_medium),
        )
        set_camera_orbit(plotter, focal, az_deg=125.0, el_deg=22.0,
                         domain_size=(Lx, Ly, Lz))

        print(f"  T range: [{T_min:.2f}, {T_max:.2f}] C")
        print(f"  Q levels: {q_levels}")

        img = plotter.screenshot(return_img=True)
        composed = compose_with_header(
            img, full_times[idx], TKE_arr[idx], Nu_arr[idx],
            Re_val, Ra_val, Cs_val, label=args.label,
        )
        out_png = output.with_suffix(".png")
        imageio.imwrite(str(out_png), composed)
        plotter.close()
        size_mb = out_png.stat().st_size / 1e6
        print(f"\n[OK] Test PNG: {out_png} ({size_mb:.1f} MB)")
        return

    # ------------------------------------------------------------
    # FULL RENDER LOOP (multi-process precompute + nvenc encoding)
    # ------------------------------------------------------------
    if args.preview:
        win_size = (2560, 1440)
        n_frames = 60
        print(f"  [PREVIEW] 1440p x {n_frames} frame")
    else:
        win_size = (3840, 2160)
        n_frames = int(args.duration * args.fps)
        print(f"  4K x {n_frames} frame ({args.duration:.0f}s @ {args.fps}fps)")

    # ============ PREFETCH precompute (ProcessPool + sliding window) ============
    # Paralel 4 worker, render bir snap işlerken sonraki N snap'i önceden hazırla.
    # Cache her zaman aktif i0 etrafında sliding window (RAM güvenli).
    from concurrent.futures import ProcessPoolExecutor

    n_workers = args.workers if args.workers else 4
    PREFETCH = max(n_workers * 2, 8)   # i0 önünde 8 snap pipeline'da
    WINDOW_BACK = 2                     # i0 arkasında 2 snap tut

    pool = ProcessPoolExecutor(max_workers=n_workers)
    geom_cache = {}
    pending_futures = {}   # idx -> future

    def submit_precompute(idx):
        if idx in geom_cache or idx in pending_futures or idx < 0 or idx >= n_snaps:
            return
        args_tuple = (idx, full_snaps[idx], Lx, Ly, Lz,
                      args.q_strong, args.q_medium)
        pending_futures[idx] = pool.submit(_precompute_worker, args_tuple)

    def get_or_wait(idx):
        """Cache'te varsa döndür, yoksa future'dan bekleyip al."""
        if idx in geom_cache:
            return geom_cache[idx]
        if idx not in pending_futures:
            submit_precompute(idx)
        _, data = pending_futures[idx].result()  # block
        geom_cache[idx] = data
        del pending_futures[idx]
        return data

    def prefetch_ahead(i0):
        """i0+1..i0+PREFETCH için future submit (block etmez)."""
        for k in range(i0 + 1, min(i0 + PREFETCH + 1, n_snaps)):
            submit_precompute(k)

    def trim_cache(active_idx):
        for k in list(geom_cache.keys()):
            if k < active_idx - WINDOW_BACK or k > active_idx + PREFETCH:
                del geom_cache[k]

    # İlk snap için tüm prefetch buffer'ı doldur, sonra ilk snap'i al
    print(f"\n[PREFETCH] workers={n_workers}, prefetch={PREFETCH}, window_back={WINDOW_BACK}")
    print(f"[PREFETCH] ilk {PREFETCH+1} snap pipeline'a alınıyor...")
    for k in range(0, min(PREFETCH + 1, n_snaps)):
        submit_precompute(k)
    # İlk snap'i bekle (base_grid init için)
    _ = get_or_wait(0)
    print(f"[PREFETCH] hazır, render başlıyor (toplam {n_snaps} snap)")

    # ============ Plotter setup ============
    plotter = setup_plotter(win_size, Lx, Ly, Lz, focal)

    # Base ImageData (point-centered, scalar her frame interpolate edilecek)
    base_grid = pv.ImageData(
        dimensions=(Nx + 1, Ny + 1, Nz + 1),
        spacing=(Lx / Nx, Ly / Ny, Lz / Nz),
        origin=(0.0, 0.0, 0.0),
    )
    base_grid.point_data["T_C"] = geom_cache[0]["T_C_point"].copy()
    cmap = ansys_thermal_cmap()

    # ============ Tier 2: ffmpeg subprocess pipe + h264_nvenc ============
    import subprocess
    try:
        import imageio_ffmpeg
        ff_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ff_exe = "ffmpeg"

    # Test header'lı frame boyutu için bir dummy compose yap, height al
    dummy_pv_img = np.zeros((win_size[1], win_size[0], 3), dtype=np.uint8)
    dummy_frame = compose_with_header(
        dummy_pv_img, 0.0, 0.0, 0.0, Re_val, Ra_val, Cs_val, label=args.label
    )
    H_total, W_total = dummy_frame.shape[:2]
    print(f"\n  Frame size: {W_total}x{H_total}")

    use_nvenc = not args.no_nvenc
    codec_str = "h264_nvenc (GPU)" if use_nvenc else "libx264 (CPU)"
    print(f"  Codec: {codec_str}, bitrate {args.bitrate}")

    if use_nvenc:
        ff_cmd = [
            ff_exe, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{W_total}x{H_total}",
            "-pix_fmt", "rgb24",
            "-r", str(args.fps),
            "-i", "-",
            "-c:v", "h264_nvenc",
            "-preset", "p6", "-tune", "hq",
            "-rc", "cbr",
            "-b:v", args.bitrate, "-maxrate", args.bitrate, "-bufsize", args.bitrate,
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output),
        ]
    else:
        ff_cmd = [
            ff_exe, "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{W_total}x{H_total}",
            "-pix_fmt", "rgb24",
            "-r", str(args.fps),
            "-i", "-",
            "-c:v", "libx264", "-preset", "fast",
            "-b:v", args.bitrate, "-pix_fmt", "yuv420p",
            str(output),
        ]

    proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # ============ Tier 3: Volume actor bir kez ekle, scalar update ile reuse ============
    sigma_otf = [0.32, 0.18, 0.055, 0.018, 0.055, 0.18, 0.36]
    vol_actor = plotter.add_volume(
        base_grid, scalars="T_C", cmap=cmap, clim=(0.0, 20.0),
        opacity=sigma_otf, shade=False,
        ambient=0.30, diffuse=0.65, specular=0.30,
        mapper="gpu", show_scalar_bar=False,
    )
    try:
        plotter.add_scalar_bar(
            title="Sicaklik [C]",
            mapper=vol_actor.mapper,
            n_labels=5, fmt="%.0f",
            title_font_size=26, label_font_size=22,
            position_x=0.91, position_y=0.16,
            width=0.060, height=0.70, vertical=True,
            color="white", italic=False, bold=True,
            font_family="arial", outline=True, fill=True,
        )
    except Exception:
        pass

    # MSAA off — 4K'da 30% hızlanma
    try:
        plotter.render_window.SetMultiSamples(0)
    except Exception:
        pass

    t_start = time.time()
    last_i0 = -1
    dyn_actors = []  # iso_strong, iso_medium, tubes (snapshot başına swap)

    try:
        for fi in range(n_frames):
            # Float snapshot index — interpolation için
            f_idx = fi / max(n_frames - 1, 1) * (n_snaps - 1)
            i0 = int(f_idx)
            i1 = min(i0 + 1, n_snaps - 1)
            alpha = f_idx - i0
            t_idx = i0  # legacy reference

            # === PREFETCH precompute: future'ları submit + i0/i1'i bekle ===
            prefetch_ahead(i0)
            data0 = get_or_wait(i0)
            data1 = get_or_wait(i1)
            trim_cache(i0)

            # === Volume scalar HER FRAME interpolate (akış smooth) ===
            T_C_blend = ((1.0 - alpha) * data0["T_C_point"]
                         + alpha * data1["T_C_point"])
            base_grid.point_data["T_C"] = T_C_blend
            base_grid.Modified()

            # === Q-iso & tubes sadece i0 değişince güncellenir ===
            if i0 != last_i0:
                for a in dyn_actors:
                    plotter.remove_actor(a)
                dyn_actors = []

                cached = geom_cache[i0]

                # Q-iso & tubes mesh inşa et, actor ekle
                iso_s, iso_m, tubes = cached_to_meshes(cached)

                if iso_s is not None:
                    dyn_actors.append(plotter.add_mesh(
                        iso_s, color="#8B0000", opacity=0.85,
                        smooth_shading=True,
                        ambient=0.20, diffuse=0.55,
                        specular=0.55, specular_power=25,
                        show_scalar_bar=False,
                    ))
                if iso_m is not None:
                    dyn_actors.append(plotter.add_mesh(
                        iso_m, color="#FF8C00", opacity=0.50,
                        smooth_shading=True,
                        ambient=0.20, diffuse=0.55,
                        specular=0.55, specular_power=25,
                        show_scalar_bar=False,
                    ))
                if tubes is not None:
                    dyn_actors.append(plotter.add_mesh(
                        tubes, scalars="speed", cmap="viridis",
                        opacity=0.90, smooth_shading=True,
                        show_scalar_bar=False,
                        ambient=0.20, diffuse=0.55,
                        specular=0.45, specular_power=20,
                    ))
                last_i0 = i0

            # Camera
            if rotate:
                az = 360.0 * fi / max(n_frames - 1, 1)
                set_camera_orbit(plotter, focal, az_deg=az, el_deg=22.0,
                                 domain_size=(Lx, Ly, Lz))
            else:
                set_camera_orbit(plotter, focal, az_deg=125.0, el_deg=22.0,
                                 domain_size=(Lx, Ly, Lz))

            t_now = full_times[t_idx]
            tke_now = TKE_arr[t_idx] if t_idx < len(TKE_arr) else TKE_arr[-1]
            nu_now = Nu_arr[t_idx] if t_idx < len(Nu_arr) else Nu_arr[-1]

            img = plotter.screenshot(return_img=True)
            frame = compose_with_header(
                img, t_now, tke_now, nu_now,
                Re_val, Ra_val, Cs_val, label=args.label,
            )
            proc.stdin.write(frame.tobytes())

            if (fi + 1) % 60 == 0 or fi == 0:
                elapsed = time.time() - t_start
                fps_render = (fi + 1) / max(elapsed, 1e-6)
                eta = (n_frames - fi - 1) / max(fps_render, 1e-6)
                eta_str = f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"
                print(f"  frame {fi+1}/{n_frames}  t_idx={t_idx}  "
                      f"{fps_render:.2f}fps  eta {eta_str}")
    finally:
        try:
            proc.stdin.close()
            proc.wait(timeout=60)
        except Exception:
            proc.kill()
        plotter.close()
        try:
            pool.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    elapsed = time.time() - t_start
    size_mb = output.stat().st_size / 1e6 if output.exists() else 0
    print(f"\n[OK] Video: {output} ({size_mb:.1f} MB)")
    print(f"  Total render loop: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
