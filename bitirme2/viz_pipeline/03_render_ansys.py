"""
03_render_ansys.py — ANSYS-tarzı 4K@60fps karşılaştırma videosu

Layout (her bölüm tek tam ekran):
  0-28 sn:   LES Referansı (Smagorinsky SGS, Cs=0.17, gerçek referans)
  28-30 sn:  Geçiş kartı
  30-58 sn:  INNATE Modeli (saf-fizik PINO, eğitilmiş)

Her bölümde:
  - Üst (büyük): θ field 3D-tarzı volume rendering (tilted slice + perspective)
  - Sol alt: TKE(t)
  - Orta alt: Nu(t)
  - Sağ alt: E(k) spektrum

Renk: RdBu_r (kırmızı=sıcak, mavi=soğuk)
Ölçek: 4K (3840×2160), 60 fps, MP4 (h264)
"""
from __future__ import annotations
import sys, argparse, time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import imageio.v2 as imageio


# ============================================================
# Renk haritaları (ANSYS Fluent tarzı)
# ============================================================
ANSYS_THERMAL = LinearSegmentedColormap.from_list(
    "ansys_thermal",
    [
        (0.00, "#1a3a8a"),   # mavi (soğuk)
        (0.20, "#4a8ed4"),
        (0.40, "#a8d4e8"),
        (0.50, "#f5f5f0"),   # beyaz/krem (orta)
        (0.60, "#fcd34d"),
        (0.80, "#f97316"),
        (1.00, "#9b1c1c"),   # koyu kırmızı (sıcak)
    ],
)


def load_npz(path):
    print(f"Yükleniyor: {path}")
    d = np.load(path, allow_pickle=True)
    return d


def setup_4k_figure():
    """4K (3840×2160) figure. dpi=160, figsize=(24, 13.5) → 3840×2160 piksel."""
    fig = plt.figure(figsize=(24, 13.5), dpi=160, facecolor="#0a0a0a")
    return fig


def make_layout(fig, title_text=""):
    """Tek bölüm layout: üst büyük 3D-tarzı viz, alt 3 küçük plot."""
    gs = GridSpec(
        4, 3,
        figure=fig,
        height_ratios=[0.5, 4.5, 4.5, 1.5],
        width_ratios=[1, 1, 1],
        hspace=0.25,
        wspace=0.20,
        left=0.04, right=0.97, top=0.95, bottom=0.05,
    )

    # Üst başlık
    ax_title = fig.add_subplot(gs[0, :])
    ax_title.axis("off")

    # Üst büyük: 3D-tarzı θ field viewport (tilted XY-Z perspektif)
    ax_main = fig.add_subplot(gs[1:3, :])  # 2 satır, 3 kolon
    ax_main.set_facecolor("#0a0a0a")

    # Alt panel: TKE, Nu, E(k)
    ax_tke = fig.add_subplot(gs[3, 0])
    ax_nu = fig.add_subplot(gs[3, 1])
    ax_spec = fig.add_subplot(gs[3, 2])

    return ax_title, ax_main, ax_tke, ax_nu, ax_spec


def draw_thermal_field(ax, slice_y_data, slice_z_data, Lx, Ly, Lz, vmin, vmax,
                       cmap=ANSYS_THERMAL):
    """ANSYS-tarzı θ alanı: y-mid plane (xz) + z-mid plane (xy) yan yana,
    perspektif görünüm hissi için kompozisyon."""
    ax.clear()
    ax.set_facecolor("#0a0a0a")

    # Iki slice yan yana koyalım — y-mid (xz) sol, z-mid (xy) sağ
    # slice_y: [4, Nx, Nz] → theta = slice_y[3]
    # slice_z: [4, Nx, Ny] → theta = slice_z[3]
    th_xz = slice_y_data[3]   # [Nx, Nz]
    th_xy = slice_z_data[3]   # [Nx, Ny]

    # |u| büyüklüğü streamline overlay için
    u_xz = slice_y_data[0]
    v_xz = slice_y_data[1]
    w_xz = slice_y_data[2]
    umag_xz = np.sqrt(u_xz**2 + v_xz**2 + w_xz**2)

    Nx, Nz = th_xz.shape
    _, Ny = th_xy.shape

    # Subplot içinde iki imshow yerine, tek figure'da iki görüntü
    # Daha temiz: ax içinde inset_axes kullan
    ax.set_xlim(0, 2.4)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    # Sol panel: xz plane (y=Ly/2)
    extent_xz = (0, 1.1, 0, 1.0)
    im_xz = ax.imshow(
        th_xz.T, origin="lower",
        extent=extent_xz, aspect="auto",
        cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="bicubic",
    )
    ax.text(0.55, 1.02, f"y = Ly/2 düzlemi (xz)", transform=ax.transData,
            ha="center", color="white", fontsize=14, weight="bold")

    # Sağ panel: xy plane (z=Lz/2)
    extent_xy = (1.30, 2.40, 0, 1.0)
    im_xy = ax.imshow(
        th_xy.T, origin="lower",
        extent=extent_xy, aspect="auto",
        cmap=cmap, vmin=vmin, vmax=vmax,
        interpolation="bicubic",
    )
    ax.text(1.85, 1.02, f"z = Lz/2 düzlemi (xy)", transform=ax.transData,
            ha="center", color="white", fontsize=14, weight="bold")

    # Streamline overlay (sol panel)
    X, Z = np.meshgrid(
        np.linspace(0, 1.1, Nx),
        np.linspace(0, 1.0, Nz),
        indexing="ij",
    )
    try:
        ax.streamplot(
            X.T, Z.T, u_xz.T, w_xz.T,
            density=1.5, color="white", linewidth=0.4, arrowsize=0.8,
        )
    except Exception:
        pass

    return im_xz


def draw_timeseries(ax, t, y, ylabel, color="#fcd34d", current_t=None):
    ax.clear()
    ax.set_facecolor("#0a0a0a")
    ax.plot(t, y, color=color, linewidth=1.2, alpha=0.9)
    if current_t is not None:
        idx = np.argmin(np.abs(t - current_t))
        ax.axvline(t[idx], color="white", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.scatter([t[idx]], [y[idx]], color="white", s=30, zorder=5,
                   edgecolor=color, linewidth=2)
    ax.set_xlabel("t", color="white", fontsize=10)
    ax.set_ylabel(ylabel, color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.grid(True, alpha=0.2, color="white")


def draw_spectrum(ax, slice_data, current_t=None, label="LES"):
    """Slice'tan basit 1D spektrum hesapla (FFT |u|²)."""
    ax.clear()
    ax.set_facecolor("#0a0a0a")
    u = slice_data[0]  # [Nx, Nz] veya [Nx, Ny]
    Nx = u.shape[0]
    # 1D radial spectrum ham
    U = np.fft.rfft(u, axis=0)
    E = (np.abs(U) ** 2).mean(axis=1)
    k = np.arange(len(E))
    ax.loglog(k[1:], E[1:], color="#f97316", linewidth=1.5, label=label)
    # -5/3 referans
    if len(k) > 5:
        k_ref = np.arange(2, len(k))
        E_ref = E[3] * (k_ref / k_ref[0]) ** (-5 / 3)
        ax.loglog(k_ref, E_ref, "--", color="cyan", linewidth=0.8,
                  label="-5/3 (Kolmogorov)")
    ax.set_xlabel("k", color="white", fontsize=10)
    ax.set_ylabel("E(k)", color="white", fontsize=10)
    ax.tick_params(colors="white", labelsize=9)
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.legend(facecolor="#0a0a0a", edgecolor="white", labelcolor="white", fontsize=8)
    ax.grid(True, alpha=0.2, color="white")


def render_section(writer, npz_data, label, n_frames, fps, color_palette,
                   global_vmin=None, global_vmax=None,
                   start_progress=0.0, total_progress=1.0):
    """Bir bölümü (LES veya INNATE) render et."""
    slice_y = npz_data["slice_y"]  # [N, 4, Nx, Nz]
    slice_z = npz_data["slice_z"]  # [N, 4, Nx, Ny]
    t_arr = npz_data["metric_t"]
    TKE_arr = npz_data["metric_TKE"]
    Nu_arr = npz_data["metric_Nu"]
    Lx = float(npz_data["Lx"])
    Ly = float(npz_data["Ly"])
    Lz = float(npz_data["Lz"])

    N_data = slice_y.shape[0]
    if N_data == 0:
        print(f"⚠ {label}: data yok!")
        return

    # vmin/vmax (theta için)
    if global_vmin is None:
        global_vmin = float(np.percentile(slice_y[:, 3], 1))
    if global_vmax is None:
        global_vmax = float(np.percentile(slice_y[:, 3], 99))
    abs_max = max(abs(global_vmin), abs(global_vmax))
    vmin_t, vmax_t = -abs_max, abs_max  # symmetric for RdBu_r

    # Metadata'dan Cs ve Re oku (eski hardcoded "Cs=0.05" hatasını düzelt)
    def _meta(d, key, default):
        try:
            if hasattr(d, "files"):
                return float(d[key]) if key in d.files else default
            else:
                return float(d[key]) if key in d else default
        except Exception:
            return default
    Re_val = _meta(npz_data, "Re", 10000.0)
    Ra_val = _meta(npz_data, "Ra", 1e5)
    Pr_val = _meta(npz_data, "Pr", 0.71)
    Cs_val = _meta(npz_data, "Cs", 0.17)

    print(f"\n{label}: {n_frames} frame render @ {fps}fps")
    print(f"  Data frame: {N_data}, video frame: {n_frames}")
    print(f"  θ range: [{vmin_t:.4e}, {vmax_t:.4e}]")

    fig = setup_4k_figure()
    ax_title, ax_main, ax_tke, ax_nu, ax_spec = make_layout(fig)

    t0 = time.time()
    for i in range(n_frames):
        # Data idx
        data_idx = int(i * N_data / n_frames)
        data_idx = min(data_idx, N_data - 1)

        s_y = slice_y[data_idx]  # [4, Nx, Nz]
        s_z = slice_z[data_idx]  # [4, Nx, Ny]
        t_now = float(t_arr[data_idx]) if data_idx < len(t_arr) else 0.0

        # Title
        ax_title.clear()
        ax_title.axis("off")
        ax_title.text(
            0.5, 0.5,
            f"{label}    |    Re={Re_val:.0f}  Ra={Ra_val:.0e}  Pr={Pr_val:.2f}  Cs={Cs_val:.2f}  |  "
            f"t = {t_now:6.2f} / {float(t_arr[-1]):.2f}",
            transform=ax_title.transAxes,
            ha="center", va="center",
            color="white", fontsize=22, weight="bold",
        )

        # Ana θ alanı
        draw_thermal_field(ax_main, s_y, s_z, Lx, Ly, Lz, vmin_t, vmax_t,
                           cmap=color_palette)

        # Alt panel
        draw_timeseries(ax_tke, t_arr, TKE_arr, "TKE", color="#fcd34d", current_t=t_now)
        draw_timeseries(ax_nu, t_arr, Nu_arr, "Nu", color="#f97316", current_t=t_now)
        draw_spectrum(ax_spec, s_y, current_t=t_now, label=label)

        # Render frame (matplotlib → numpy → ffmpeg)
        fig.canvas.draw()
        # Modern matplotlib (3.8+) kullanır buffer_rgba
        try:
            frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]  # RGBA → RGB
        except Exception:
            frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
            w, h = fig.canvas.get_width_height()
            frame = frame.reshape(h, w, 3)
        writer.append_data(frame)

        if (i + 1) % 30 == 0 or i == 0:
            elapsed = time.time() - t0
            fps_render = (i + 1) / max(elapsed, 1e-6)
            eta = (n_frames - i - 1) / max(fps_render, 1e-6)
            eta_str = f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"
            print(f"  frame {i+1}/{n_frames}  t={t_now:.2f}  "
                  f"render {fps_render:.1f} fps  eta {eta_str}")

    plt.close(fig)


def render_transition(writer, n_frames=120, fps=60):
    """Geçiş kartı: 'Şimdi INNATE Modelinin Tahmini'"""
    print(f"\nGeçiş kartı: {n_frames} frame")
    fig = setup_4k_figure()
    fig.patch.set_facecolor("#0a0a0a")
    ax = fig.add_subplot(111)
    ax.axis("off")
    ax.set_facecolor("#0a0a0a")

    ax.text(0.5, 0.6, "Şimdi INNATE Modelinin Tahmini",
            transform=ax.transAxes, ha="center", va="center",
            color="white", fontsize=64, weight="bold")
    ax.text(0.5, 0.45, "(Saf-fizik PINO, 503 parametre, eğitilmiş)",
            transform=ax.transAxes, ha="center", va="center",
            color="#fcd34d", fontsize=32, style="italic")
    ax.text(0.5, 0.32, "Aynı başlangıç koşulu, aynı zaman, aynı fizik denklemi",
            transform=ax.transAxes, ha="center", va="center",
            color="#a8d4e8", fontsize=24)

    fig.canvas.draw()
    try:
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    except Exception:
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        w, h = fig.canvas.get_width_height()
        frame = frame.reshape(h, w, 3)
    for _ in range(n_frames):
        writer.append_data(frame)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="ANSYS-tarzı 4K karşılaştırma videosu")
    parser.add_argument("--les", type=str, default="data/sim_states/les_aggressive_15k.npz")
    parser.add_argument("--innate", type=str, default="data/sim_states/innate_rollout.npz")
    parser.add_argument("--output", type=str, default="data/sim_states/comparison_4k_60fps.mp4")
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--les-duration", type=float, default=28.0,
                        help="LES bölüm süresi (saniye)")
    parser.add_argument("--transition-duration", type=float, default=2.0)
    parser.add_argument("--innate-duration", type=float, default=28.0)
    parser.add_argument("--bitrate", type=str, default="20M")
    parser.add_argument("--les-tail", type=int, default=None,
                        help="LES'in son N frame'ini al (steady-state segment, INNATE ile zaman senkron)")
    args = parser.parse_args()

    print("=" * 80)
    print("ANSYS-TARZI 4K KARŞILAŞTIRMA VİDEOSU")
    print("=" * 80)
    print(f"LES:    {args.les}")
    print(f"INNATE: {args.innate}")
    print(f"Output: {args.output}")
    print(f"FPS={args.fps}, total≈{args.les_duration + args.transition_duration + args.innate_duration:.1f}s")

    les_d = load_npz(args.les)
    print(f"  LES: {les_d['slice_y'].shape[0]} frame, "
          f"t_max={float(les_d['metric_t'][-1]):.2f}")

    # Zaman senkron için LES son N frame'i al (INNATE ile aynı zaman aralığı)
    if args.les_tail is not None and args.les_tail > 0:
        n_keep = min(args.les_tail, les_d["slice_y"].shape[0])
        # NpzFile read-only; copy view'i dict'e dönüştür
        les_d_subset = {
            "slice_y": les_d["slice_y"][-n_keep:],
            "slice_z": les_d["slice_z"][-n_keep:],
            "metric_t": les_d["metric_t"][-n_keep:],
            "metric_TKE": les_d["metric_TKE"][-n_keep:],
            "metric_Nu": les_d["metric_Nu"][-n_keep:],
            "Lx": float(les_d["Lx"]),
            "Ly": float(les_d["Ly"]),
            "Lz": float(les_d["Lz"]),
            "Re": float(les_d["Re"]) if "Re" in les_d.files else 10000.0,
            "Ra": float(les_d["Ra"]) if "Ra" in les_d.files else 1e5,
            "Pr": float(les_d["Pr"]) if "Pr" in les_d.files else 0.71,
            "Cs": float(les_d["Cs"]) if "Cs" in les_d.files else 0.17,
        }
        les_d = les_d_subset
        print(f"  → LES son {n_keep} frame seçildi (steady-state, "
              f"t={les_d['metric_t'][0]:.2f}-{les_d['metric_t'][-1]:.2f})")

    innate_path = Path(args.innate)
    if innate_path.exists():
        innate_d = load_npz(args.innate)
        print(f"  INNATE: {innate_d['slice_y'].shape[0]} frame, "
              f"t_max={float(innate_d['metric_t'][-1]):.2f}")
    else:
        innate_d = None
        print(f"  ⚠ INNATE çıktısı yok ({args.innate}), sadece LES bölümü render edilecek")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Frame sayıları
    n_les = int(args.les_duration * args.fps)
    n_trans = int(args.transition_duration * args.fps)
    n_innate = int(args.innate_duration * args.fps) if innate_d is not None else 0

    # Ortak vmin/vmax (LES + INNATE uyumlu)
    global_min = float(np.percentile(les_d["slice_y"][:, 3], 1))
    global_max = float(np.percentile(les_d["slice_y"][:, 3], 99))
    if innate_d is not None and innate_d["slice_y"].shape[0] > 0:
        global_min = min(global_min, float(np.percentile(innate_d["slice_y"][:, 3], 1)))
        global_max = max(global_max, float(np.percentile(innate_d["slice_y"][:, 3], 99)))

    # Writer (4K @ fps, h264)
    print(f"\nMP4 writer: {output}, fps={args.fps}, bitrate={args.bitrate}")
    writer = imageio.get_writer(
        str(output), fps=args.fps, codec="h264", quality=8,
        bitrate=args.bitrate, macro_block_size=8,
    )

    try:
        # 1. LES bölümü
        render_section(writer, les_d, "LES Referansı (Smagorinsky SGS)",
                       n_les, args.fps, ANSYS_THERMAL,
                       global_vmin=global_min, global_vmax=global_max)

        # 2. Geçiş + INNATE bölümü (varsa)
        if innate_d is not None and n_innate > 0:
            render_transition(writer, n_frames=n_trans, fps=args.fps)
            render_section(writer, innate_d, "INNATE Modeli (Saf-Fizik PINO)",
                           n_innate, args.fps, ANSYS_THERMAL,
                           global_vmin=global_min, global_vmax=global_max)
    finally:
        writer.close()

    size_mb = output.stat().st_size / 1e6
    print(f"\n✓ Video kaydedildi: {output} ({size_mb:.1f} MB)")
    print(f"  Süre: {(n_les + n_trans + n_innate) / args.fps:.1f}s @ {args.fps}fps")


if __name__ == "__main__":
    main()
