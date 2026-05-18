#!/opt/anaconda3/bin/python
"""
3D Mixed Convection Room Simulation - Bitirme Tezi Demo
========================================================
Rayleigh-Benard convection (hot floor, cold ceiling) + Kolmogorov-like
horizontal wind forcing in a 3D room geometry.

Physics (synthetic but plausible):
  - Temperature: Linear gradient + Gaussian thermal plumes rising from floor
  - Velocity u: Kolmogorov-like A*sin(2*pi*z/Lz) + plume entrainment
  - Velocity w: Buoyancy-driven updrafts in plume cores
  - Recirculation cells formed by interaction of wind and buoyancy

Author: Berke Tezgozcen (ITU Physics Engineering)
Visualization: PyVista 3D + ffmpeg MP4 export
"""

import numpy as np
import pyvista as pv
from PIL import Image
import subprocess
import os
import sys
import tempfile
import shutil

# ============================================================
#  PHYSICAL PARAMETERS (non-dimensionalized for visualization)
# ============================================================
Lx, Ly, Lz = 4.0, 2.0, 2.0        # Room dimensions (x=length, y=depth, z=height)
Nx, Ny, Nz = 80, 48, 48            # Grid resolution (higher for smoother visuals)
T_hot, T_cold = 1.0, 0.0           # Floor / ceiling temperatures
A_wind = 0.7                        # Kolmogorov wind amplitude
N_plumes = 6                        # Number of thermal plumes

# ============================================================
#  SYNTHETIC FLOW FIELD GENERATION
# ============================================================

def make_grid():
    """Create 3D structured grid."""
    x = np.linspace(0, Lx, Nx)
    y = np.linspace(0, Ly, Ny)
    z = np.linspace(0, Lz, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    return X, Y, Z, x, y, z


def generate_plume_positions(t, seed=42):
    """Generate plume positions that drift slowly with time."""
    rng = np.random.RandomState(seed)
    px = rng.uniform(0.5, Lx - 0.5, N_plumes)
    py = rng.uniform(0.3, Ly - 0.3, N_plumes)
    drift_x = 0.2 * np.sin(2 * np.pi * t / 8.0 + rng.uniform(0, 2*np.pi, N_plumes))
    drift_y = 0.1 * np.cos(2 * np.pi * t / 12.0 + rng.uniform(0, 2*np.pi, N_plumes))
    return px + drift_x, py + drift_y


def generate_fields(t):
    """
    Generate synthetic velocity and temperature fields at time t.

    Temperature:
      T(x,y,z,t) = T_hot - dT*z/Lz + SUM_plumes[Gaussian * envelope]

    Velocity:
      u = A*sin(2*pi*z/Lz) Kolmogorov wind + plume entrainment
      v = secondary circulation
      w = buoyancy plume updrafts
    """
    X, Y, Z, x, y, z = make_grid()
    phase_t = 2 * np.pi * t / 5.0

    # --- Temperature field ---
    # Base: linear stratification
    T = T_hot - (T_hot - T_cold) * Z / Lz

    # Thermal plumes (Gaussians rising from floor)
    px, py = generate_plume_positions(t)
    plume_strengths = np.array([0.55, 0.40, 0.60, 0.35, 0.50, 0.45])

    for i in range(N_plumes):
        # Plume widens as it rises (mushroom cap effect)
        sigma_h = 0.15 + 0.18 * (Z / Lz)**0.7
        sigma_z_factor = 0.7 + 0.3 * np.sin(2 * np.pi * t / 6.0 + i * 1.2)

        r2 = ((X - px[i])**2 + (Y - py[i])**2) / sigma_h**2

        # Vertical envelope: mushroom shape
        # Strong near floor, has a cap at ~60% height, weak above
        z_norm = Z / Lz
        z_env = (
            0.8 * np.exp(-(z_norm - 0.08)**2 / (2 * 0.05**2))   # Hot root
            + 1.0 * np.exp(-(z_norm - 0.30 * sigma_z_factor)**2 / (2 * 0.12**2))  # Rising stem
            + 0.5 * np.exp(-(z_norm - 0.55 * sigma_z_factor)**2 / (2 * 0.10**2))  # Cap
        )

        plume_T = plume_strengths[i] * np.exp(-r2 / 2) * z_env
        T += plume_T

    # Small-scale turbulent texture
    T += 0.04 * np.sin(5*np.pi*X/Lx + phase_t) * np.sin(4*np.pi*Y/Ly - 0.3*phase_t) * np.sin(3*np.pi*Z/Lz)
    T += 0.02 * np.cos(7*np.pi*X/Lx - phase_t*0.5) * np.cos(5*np.pi*Z/Lz + phase_t*0.3)

    T = np.clip(T, T_cold - 0.05, T_hot + 0.55)

    # --- Velocity field ---
    # u-component: Kolmogorov-like horizontal wind
    u = A_wind * np.sin(2 * np.pi * Z / Lz)
    u *= (1.0 + 0.2 * np.sin(2 * np.pi * Y / Ly + 0.5 * phase_t))
    u *= (1.0 + 0.12 * np.sin(phase_t * 0.7))

    # Wind deflection by plumes
    for i in range(N_plumes):
        r2 = ((X - px[i])**2 + (Y - py[i])**2)
        r = np.sqrt(r2 + 0.01)
        deflect = 0.35 * plume_strengths[i] * np.exp(-r2 / (2 * 0.25**2))
        u -= deflect * (X - px[i]) / (r + 0.1) * (Z / Lz)

    # v-component: secondary circulation
    v = 0.12 * np.sin(2 * np.pi * X / Lx + phase_t * 0.5) * np.cos(np.pi * Z / Lz)
    v += 0.08 * np.cos(3 * np.pi * Y / Ly) * np.sin(np.pi * Z / Lz)
    # Entrainment into plumes
    for i in range(N_plumes):
        r2 = ((X - px[i])**2 + (Y - py[i])**2)
        r = np.sqrt(r2 + 0.01)
        entrain = 0.15 * plume_strengths[i] * np.exp(-r2 / (2 * 0.3**2))
        v -= entrain * (Y - py[i]) / (r + 0.1) * np.sin(np.pi * Z / Lz)

    # w-component: buoyancy-driven updrafts
    w = np.zeros_like(X)
    for i in range(N_plumes):
        sigma_h = 0.18
        r2 = ((X - px[i])**2 + (Y - py[i])**2) / sigma_h**2
        # Updraft in plume core, weak downdraft in annulus
        plume_w = plume_strengths[i] * 1.5 * (1 - r2/5) * np.exp(-r2 / 2)
        z_prof = np.sin(np.pi * Z / Lz)  # Zero at floor and ceiling
        w += plume_w * z_prof

    # Background subsidence (mass conservation)
    w -= 0.06 * np.sin(np.pi * Z / Lz)

    # Turbulent fluctuations
    turb = 0.035
    w += turb * np.sin(5*np.pi*X/Lx + phase_t) * np.cos(4*np.pi*Y/Ly + 0.3*phase_t)
    u += turb * 0.5 * np.cos(3*np.pi*Z/Lz + phase_t) * np.sin(2*np.pi*X/Lx)
    v += turb * 0.5 * np.sin(2*np.pi*Z/Lz) * np.cos(3*np.pi*X/Lx - phase_t)

    return X, Y, Z, u, v, w, T


def fields_to_pyvista(X, Y, Z, u, v, w, T):
    """Convert numpy arrays to PyVista StructuredGrid."""
    grid = pv.StructuredGrid(X, Y, Z)
    grid['Temperature'] = T.flatten(order='F')
    vectors = np.column_stack([u.flatten(order='F'),
                               v.flatten(order='F'),
                               w.flatten(order='F')])
    grid['Velocity'] = vectors
    grid['Speed'] = np.linalg.norm(vectors, axis=1)
    return grid


# ============================================================
#  VISUALIZATION COMPONENTS
# ============================================================

def setup_plotter(off_screen=False):
    """Create and configure the PyVista plotter with cinematic settings."""
    p = pv.Plotter(off_screen=off_screen, window_size=[1920, 1080])
    p.set_background('#0a0a1a', top='#1a1a3a')
    p.enable_anti_aliasing('ssaa')
    return p


def add_lighting(p):
    """Setup cinematic 3-point lighting."""
    p.remove_all_lights()

    # Key light: warm, from upper-right-front
    key = pv.Light(position=(Lx + 4, -3, Lz + 5),
                   focal_point=(Lx/2, Ly/2, Lz/3),
                   color='#FFF5E6', intensity=0.9)
    key.positional = True
    key.cone_angle = 60
    p.add_light(key)

    # Fill light: cool, from left
    fill = pv.Light(position=(-4, Ly + 3, Lz * 0.8),
                    focal_point=(Lx/2, Ly/2, Lz/2),
                    color='#C0D8F0', intensity=0.35)
    p.add_light(fill)

    # Rim/back light: warm gold accent
    rim = pv.Light(position=(Lx/2, -4, -1),
                   focal_point=(Lx/2, Ly/2, Lz/2),
                   color='#FFD080', intensity=0.2)
    p.add_light(rim)

    # Subtle ambient fill
    amb = pv.Light(light_type='headlight', intensity=0.12)
    p.add_light(amb)


def add_room_geometry(p):
    """Add room box with stylized walls."""
    # Wireframe box (main structural element)
    box = pv.Box(bounds=(0, Lx, 0, Ly, 0, Lz))
    p.add_mesh(box, style='wireframe', color='#E0E0FF', line_width=2.5,
               opacity=0.7)

    # Floor: warm heated surface with grid texture
    floor = pv.Plane(center=(Lx/2, Ly/2, 0), direction=(0, 0, 1),
                     i_size=Lx, j_size=Ly, i_resolution=16, j_resolution=8)
    p.add_mesh(floor, color='#A0522D', opacity=0.7, show_edges=True,
               edge_color='#C07040', edge_opacity=0.15, lighting=True)

    # Ceiling: cool surface
    ceiling = pv.Plane(center=(Lx/2, Ly/2, Lz), direction=(0, 0, -1),
                       i_size=Lx, j_size=Ly, i_resolution=16, j_resolution=8)
    p.add_mesh(ceiling, color='#4A6A8A', opacity=0.5, show_edges=True,
               edge_color='#5A7A9A', edge_opacity=0.1, lighting=True)

    # Back wall: subtle fill
    back = pv.Plane(center=(Lx/2, 0, Lz/2), direction=(0, 1, 0),
                    i_size=Lx, j_size=Lz, i_resolution=1, j_resolution=1)
    p.add_mesh(back, color='#404060', opacity=0.12, lighting=True)

    # Left wall (wind inlet side): slightly highlighted
    left = pv.Plane(center=(0, Ly/2, Lz/2), direction=(1, 0, 0),
                    i_size=Ly, j_size=Lz, i_resolution=1, j_resolution=1)
    p.add_mesh(left, color='#405070', opacity=0.10, lighting=True)

    # Add "HOT" / "COLD" labels on floor/ceiling edges
    p.add_point_labels(
        np.array([[Lx/2, -0.15, 0.0]]),
        ['T = T_hot (heated floor)'],
        font_size=11, text_color='#FF8060', shape=None,
        show_points=False, always_visible=True, font_family='times'
    )
    p.add_point_labels(
        np.array([[Lx/2, -0.15, Lz]]),
        ['T = T_cold (cooled ceiling)'],
        font_size=11, text_color='#60A0FF', shape=None,
        show_points=False, always_visible=True, font_family='times'
    )


def add_temperature_volume(p, grid):
    """
    Show temperature as:
    1. A prominent vertical slice (mid-y) showing full thermal structure
    2. Transparent horizontal slices for depth
    3. Glowing isosurfaces for thermal plumes
    """
    # --- Main vertical cross-section (the money shot) ---
    slice_y = grid.slice(normal='y', origin=(Lx/2, Ly * 0.48, Lz/2))
    p.add_mesh(slice_y, scalars='Temperature', cmap='coolwarm',
               clim=[T_cold - 0.02, T_hot + 0.4], opacity=0.92,
               show_scalar_bar=False, lighting=True, smooth_shading=True)

    # --- Second vertical slice offset for depth ---
    slice_y2 = grid.slice(normal='y', origin=(Lx/2, Ly * 0.2, Lz/2))
    p.add_mesh(slice_y2, scalars='Temperature', cmap='coolwarm',
               clim=[T_cold - 0.02, T_hot + 0.4], opacity=0.3,
               show_scalar_bar=False, lighting=True, smooth_shading=True)

    # --- Vertical x-slice for cross-view ---
    slice_x = grid.slice(normal='x', origin=(Lx * 0.65, Ly/2, Lz/2))
    p.add_mesh(slice_x, scalars='Temperature', cmap='coolwarm',
               clim=[T_cold - 0.02, T_hot + 0.4], opacity=0.25,
               show_scalar_bar=False, lighting=True, smooth_shading=True)

    # --- Floor temperature heatmap (replaces flat floor) ---
    floor_sl = grid.slice(normal='z', origin=(Lx/2, Ly/2, 0.03))
    p.add_mesh(floor_sl, scalars='Temperature', cmap='inferno',
               clim=[T_hot - 0.05, T_hot + 0.5], opacity=0.85,
               show_scalar_bar=False, lighting=True, smooth_shading=True)

    # --- Isosurfaces: thermal plume cores (the impressive part) ---
    for iso_val, color, opa in [(T_hot + 0.20, '#FF4500', 0.45),
                                 (T_hot + 0.30, '#FF6600', 0.55),
                                 (T_hot + 0.40, '#FFAA00', 0.65)]:
        try:
            iso = grid.contour(isosurfaces=[iso_val], scalars='Temperature')
            if iso.n_points > 20:
                p.add_mesh(iso, color=color, opacity=opa,
                           smooth_shading=True, show_scalar_bar=False,
                           specular=0.5, specular_power=30)
        except Exception:
            pass

    # --- Scalar bar ---
    sbar = grid.slice(normal='x', origin=(Lx * 0.3, Ly/2, Lz/2))
    p.add_mesh(sbar, scalars='Temperature', cmap='coolwarm',
               clim=[T_cold - 0.02, T_hot + 0.4], opacity=0.0,
               scalar_bar_args={
                   'title': 'T / T_ref',
                   'title_font_size': 14,
                   'label_font_size': 11,
                   'shadow': True,
                   'color': 'white',
                   'position_x': 0.88,
                   'position_y': 0.2,
                   'width': 0.06,
                   'height': 0.6,
                   'fmt': '%.2f',
               })


def add_velocity_glyphs(p, grid):
    """Add sparse, elegant velocity arrows that do not clutter the scene."""
    # Very sparse subsampling: ~10x8x6 arrows total
    subset = grid.extract_subset(
        voi=(0, Nx-1, 0, Ny-1, 0, Nz-1),
        rate=(8, 6, 8)
    )
    subset.set_active_vectors('Velocity')

    # Use cone glyphs (slimmer and more elegant than arrows)
    arrows = subset.glyph(orient='Velocity', scale='Speed',
                          factor=0.45, geom=pv.Arrow(
                              tip_length=0.35, tip_radius=0.07,
                              shaft_radius=0.025, shaft_resolution=12))
    p.add_mesh(arrows, scalars='Speed', cmap='hot',
               clim=[0.05, 1.0], opacity=0.75,
               show_scalar_bar=False, lighting=True,
               smooth_shading=True)


def add_streamlines(p, grid, dense=False):
    """Add thick, glowing streamlines from wind inlet."""
    grid.set_active_vectors('Velocity')

    # Seed points on left wall (inlet) + some near plumes
    seeds = []
    n_y = 5 if dense else 3
    n_z = 5 if dense else 4
    for yy in np.linspace(0.2, Ly - 0.2, n_y):
        for zz in np.linspace(0.2, Lz - 0.2, n_z):
            seeds.append([0.05, yy, zz])

    # Additional seeds near floor (to catch plume updrafts)
    for xx in np.linspace(0.5, Lx - 0.5, 4):
        for yy in np.linspace(0.3, Ly - 0.3, 3):
            seeds.append([xx, yy, 0.15])

    seed = pv.PointSet(np.array(seeds))

    try:
        # Use max_time for integration length
        streamlines = grid.streamlines_from_source(
            seed, vectors='Velocity',
            max_steps=800,
            terminal_speed=0.005,
            integration_direction='both',
            max_time=15.0,
        )
        if streamlines.n_points > 50:
            tubes = streamlines.tube(radius=0.018, n_sides=12)
            p.add_mesh(tubes, scalars='Speed', cmap='plasma',
                       clim=[0.0, 1.0], opacity=0.65,
                       show_scalar_bar=False, lighting=True,
                       smooth_shading=True, specular=0.3)
    except Exception as e:
        print(f"  [Warning] Streamlines: {e}")


def add_wind_indicator(p):
    """Wind direction indicator outside the room."""
    # Large directional arrow
    arrow = pv.Arrow(start=(-1.0, Ly/2, Lz * 0.75),
                     direction=(1, 0, 0), scale=1.2,
                     tip_length=0.25, tip_radius=0.10, shaft_radius=0.04)
    p.add_mesh(arrow, color='#00D4FF', opacity=0.85, smooth_shading=True,
               specular=0.5)

    # Sinusoidal profile indicator (showing Kolmogorov pattern)
    zz = np.linspace(0, Lz, 60)
    u_prof = 0.5 * np.sin(2 * np.pi * zz / Lz)
    sine_pts = np.column_stack([
        -0.5 + u_prof,
        np.full_like(zz, Ly/2),
        zz
    ])
    sine_line = pv.Spline(sine_pts, 200)
    tube = sine_line.tube(radius=0.012)
    p.add_mesh(tube, color='#00D4FF', opacity=0.6, smooth_shading=True)

    # Label
    p.add_point_labels(
        np.array([[-0.7, Ly/2, Lz + 0.25]]),
        ['u = A sin(2 pi z/L)'],
        font_size=12, text_color='#00D4FF', shape=None,
        show_points=False, always_visible=True, font_family='times'
    )


def add_annotations(p, t):
    """Title and info text."""
    p.add_text(
        "3D Mixed Convection in a Heated Room",
        position='upper_left', font_size=16, color='white',
        shadow=True, font='times'
    )

    info = (f"Ra ~ 10^4  |  Kolmogorov wind + Rayleigh-Benard  |  t = {t:.2f}")
    p.add_text(info, position='lower_left', font_size=10,
               color='#B0B0C0', shadow=True, font='times')

    p.add_text(
        "ITU Physics Engineering  -  B. Tezgozcen",
        position='lower_right', font_size=9,
        color='#808090', shadow=True, font='times'
    )


def set_camera(p, angle_deg):
    """Cinematic camera orbit."""
    cx, cy, cz = Lx / 2, Ly / 2, Lz / 2
    radius = 7.0
    elev = 28  # degrees above horizontal

    theta = np.radians(angle_deg)
    phi = np.radians(elev)

    cam_x = cx + radius * np.cos(theta) * np.cos(phi)
    cam_y = cy + radius * np.sin(theta) * np.cos(phi)
    cam_z = cz + radius * np.sin(phi) + 0.3

    p.camera_position = [(cam_x, cam_y, cam_z),
                         (cx, cy, cz * 0.85),
                         (0, 0, 1)]
    p.camera.view_angle = 32


# ============================================================
#  RENDER A SINGLE FRAME
# ============================================================

def render_frame(t, angle_deg, off_screen=True, dense_streamlines=False):
    """Render one complete frame."""
    print(f"  Generating fields at t={t:.2f} ...")
    X, Y, Z, u, v, w, T = generate_fields(t)
    grid = fields_to_pyvista(X, Y, Z, u, v, w, T)

    p = setup_plotter(off_screen=off_screen)
    add_lighting(p)
    add_room_geometry(p)
    add_temperature_volume(p, grid)
    add_velocity_glyphs(p, grid)
    add_streamlines(p, grid, dense=dense_streamlines)
    add_wind_indicator(p)
    add_annotations(p, t)
    set_camera(p, angle_deg)

    return p, grid


# ============================================================
#  ANIMATION EXPORT
# ============================================================

def render_animation(n_frames=120, fps=24, output_dir=None):
    """
    Render animation: camera orbits while flow evolves.
    Outputs MP4 (via ffmpeg) and GIF (via Pillow).
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    mp4_path = os.path.join(output_dir, 'room_simulation.mp4')
    gif_path = os.path.join(output_dir, 'room_simulation.gif')
    poster_path = os.path.join(output_dir, 'room_simulation_poster.png')

    tmpdir = tempfile.mkdtemp(prefix='room_sim_')
    print(f"[INFO] Temp frames: {tmpdir}")
    print(f"[INFO] Rendering {n_frames} frames at 1920x1080 ...")

    gif_frames = []

    for i in range(n_frames):
        t = i * 10.0 / n_frames
        angle = 220 + i * (100.0 / n_frames)  # Slow orbit

        print(f"\n--- Frame {i+1}/{n_frames} ---")
        p, grid = render_frame(t, angle, off_screen=True)

        frame_path = os.path.join(tmpdir, f'frame_{i:04d}.png')
        p.screenshot(frame_path, transparent_background=False)
        p.close()

        # Poster frame (nicest view)
        if i == int(n_frames * 0.4):
            shutil.copy2(frame_path, poster_path)
            print(f"  -> Poster: {poster_path}")

        # GIF: every 3rd frame, half-res
        if i % 3 == 0:
            img = Image.open(frame_path)
            img = img.resize((960, 540), Image.LANCZOS)
            gif_frames.append(img)

    # --- MP4 via ffmpeg ---
    print(f"\n[INFO] Encoding MP4 ...")
    ffmpeg_cmd = [
        '/opt/homebrew/bin/ffmpeg', '-y',
        '-framerate', str(fps),
        '-i', os.path.join(tmpdir, 'frame_%04d.png'),
        '-c:v', 'libx264', '-preset', 'slow', '-crf', '18',
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
        mp4_path
    ]
    res = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if res.returncode == 0:
        mb = os.path.getsize(mp4_path) / (1024*1024)
        print(f"  -> MP4: {mp4_path} ({mb:.1f} MB, {n_frames/fps:.1f}s)")
    else:
        print(f"  [ERROR] ffmpeg:\n{res.stderr[:500]}")

    # --- GIF ---
    print(f"[INFO] Saving GIF ({len(gif_frames)} frames) ...")
    try:
        gif_frames[0].save(
            gif_path, save_all=True, append_images=gif_frames[1:],
            duration=int(1000 / (fps / 3)), loop=0, optimize=True
        )
        mb = os.path.getsize(gif_path) / (1024*1024)
        print(f"  -> GIF: {gif_path} ({mb:.1f} MB)")
    except Exception as e:
        print(f"  [ERROR] GIF: {e}")

    shutil.rmtree(tmpdir, ignore_errors=True)
    print("[DONE] Animation complete.")
    return mp4_path, gif_path, poster_path


# ============================================================
#  INTERACTIVE MODE
# ============================================================

def show_interactive(t=3.0):
    """Open interactive PyVista window."""
    print("[INFO] Interactive viewer | Left-drag=rotate, Scroll=zoom, q=quit")
    p, grid = render_frame(t, angle_deg=235, off_screen=False, dense_streamlines=True)
    p.show(title="3D Mixed Convection - Interactive Viewer")


# ============================================================
#  POSTER (single high-res frame)
# ============================================================

def render_poster(t=3.5, angle=230):
    """Render a single showcase frame."""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    poster_path = os.path.join(output_dir, 'room_simulation_poster.png')

    p, grid = render_frame(t, angle, off_screen=True, dense_streamlines=True)
    p.screenshot(poster_path, transparent_background=False)
    p.close()

    print(f"[DONE] Poster: {poster_path}")
    return poster_path


# ============================================================
#  MULTI-VIEW PANEL (4 views)
# ============================================================

def render_panel(t=3.5):
    """Render a 2x2 panel with different views for thesis figure."""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    panel_path = os.path.join(output_dir, 'room_simulation_panel.png')

    print("[INFO] Generating 2x2 panel ...")
    X, Y, Z, u, v, w, T = generate_fields(t)
    grid = fields_to_pyvista(X, Y, Z, u, v, w, T)

    p = pv.Plotter(off_screen=True, shape=(2, 2), window_size=[1920, 1080])

    # --- Panel 1: Temperature cross-section ---
    p.subplot(0, 0)
    p.set_background('#0a0a1a', top='#1a1a3a')
    sl = grid.slice(normal='y', origin=(Lx/2, Ly*0.48, Lz/2))
    p.add_mesh(sl, scalars='Temperature', cmap='coolwarm',
               clim=[T_cold, T_hot + 0.4], lighting=True, smooth_shading=True)
    box = pv.Box(bounds=(0, Lx, 0, Ly, 0, Lz))
    p.add_mesh(box, style='wireframe', color='white', opacity=0.3, line_width=1.5)
    p.add_text("Temperature Field", font_size=10, color='white', position='upper_left')
    p.camera_position = [(Lx/2, -5, Lz*0.7), (Lx/2, Ly/2, Lz/2), (0, 0, 1)]

    # --- Panel 2: Velocity magnitude ---
    p.subplot(0, 1)
    p.set_background('#0a0a1a', top='#1a1a3a')
    sl2 = grid.slice(normal='y', origin=(Lx/2, Ly*0.48, Lz/2))
    p.add_mesh(sl2, scalars='Speed', cmap='viridis',
               clim=[0, 1.0], lighting=True, smooth_shading=True)
    p.add_mesh(box, style='wireframe', color='white', opacity=0.3, line_width=1.5)
    p.add_text("Velocity Magnitude", font_size=10, color='white', position='upper_left')
    p.camera_position = [(Lx/2, -5, Lz*0.7), (Lx/2, Ly/2, Lz/2), (0, 0, 1)]

    # --- Panel 3: Isosurfaces ---
    p.subplot(1, 0)
    p.set_background('#0a0a1a', top='#1a1a3a')
    for iv, c, o in [(T_hot+0.15, '#FF4500', 0.5), (T_hot+0.30, '#FFAA00', 0.6)]:
        try:
            iso = grid.contour(isosurfaces=[iv], scalars='Temperature')
            if iso.n_points > 10:
                p.add_mesh(iso, color=c, opacity=o, smooth_shading=True)
        except:
            pass
    p.add_mesh(box, style='wireframe', color='white', opacity=0.3, line_width=1.5)
    p.add_text("Thermal Plume Isosurfaces", font_size=10, color='white', position='upper_left')
    p.camera_position = [(Lx*1.5, -3, Lz*1.5), (Lx/2, Ly/2, Lz/2), (0, 0, 1)]

    # --- Panel 4: Streamlines ---
    p.subplot(1, 1)
    p.set_background('#0a0a1a', top='#1a1a3a')
    grid.set_active_vectors('Velocity')
    seeds = []
    for yy in np.linspace(0.2, Ly-0.2, 4):
        for zz in np.linspace(0.2, Lz-0.2, 4):
            seeds.append([0.05, yy, zz])
    seed = pv.PointSet(np.array(seeds))
    try:
        sl3 = grid.streamlines_from_source(seed, vectors='Velocity',
                                            max_steps=600, max_time=12.0,
                                            integration_direction='forward')
        if sl3.n_points > 10:
            tubes = sl3.tube(radius=0.015, n_sides=10)
            p.add_mesh(tubes, scalars='Speed', cmap='plasma', clim=[0, 1.0],
                       opacity=0.7, smooth_shading=True, show_scalar_bar=False)
    except Exception as e:
        print(f"  Panel streamlines: {e}")
    p.add_mesh(box, style='wireframe', color='white', opacity=0.3, line_width=1.5)
    p.add_text("Wind Streamlines", font_size=10, color='white', position='upper_left')
    p.camera_position = [(Lx*1.5, -3, Lz*1.5), (Lx/2, Ly/2, Lz/2), (0, 0, 1)]

    p.screenshot(panel_path, transparent_background=False)
    p.close()
    print(f"[DONE] Panel: {panel_path}")
    return panel_path


# ============================================================
#  MAIN
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description='3D Mixed Convection Room Simulation - Thesis Demo',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --mode poster            # Single showcase frame (fast, ~10s)
  %(prog)s --mode panel             # 2x2 multi-view figure
  %(prog)s --mode interactive       # Interactive 3D viewer
  %(prog)s --mode animate           # Full animation MP4+GIF (~5-10 min)
  %(prog)s --mode animate -n 60     # Shorter animation
  %(prog)s --mode all               # Everything
        """
    )
    parser.add_argument('--mode', choices=['poster', 'panel', 'interactive', 'animate', 'all'],
                        default='poster',
                        help='Visualization mode (default: poster)')
    parser.add_argument('-n', '--n-frames', type=int, default=120,
                        help='Animation frames (default: 120)')
    parser.add_argument('--fps', type=int, default=24,
                        help='Animation FPS (default: 24)')
    parser.add_argument('-t', '--time', type=float, default=3.5,
                        help='Time for poster/interactive (default: 3.5)')

    args = parser.parse_args()

    print("=" * 60)
    print("  3D MIXED CONVECTION - ROOM SIMULATION")
    print("  Rayleigh-Benard + Kolmogorov Wind")
    print("  ITU Physics Engineering - B. Tezgozcen")
    print("=" * 60)
    print()

    if args.mode == 'poster':
        render_poster(t=args.time)

    elif args.mode == 'panel':
        render_panel(t=args.time)

    elif args.mode == 'interactive':
        show_interactive(t=args.time)

    elif args.mode == 'animate':
        render_animation(n_frames=args.n_frames, fps=args.fps)

    elif args.mode == 'all':
        render_poster(t=args.time)
        render_panel(t=args.time)
        render_animation(n_frames=args.n_frames, fps=args.fps)
        print("\n[INFO] For interactive mode, run: --mode interactive")


if __name__ == '__main__':
    main()
