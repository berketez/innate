"""
01_les_aggressive.py — LES rollout (gerçek referans Cs=0.17, Nu=70 hedef)

Re=10K, Ra=1e5, 96×160×64 grid, dt=0.02 SABİT (zaman senkron için fixed dt)
60000 step × dt=0.02 = 1200 zaman birim → steady-state turbulent regime
Cs=0.17 (Smagorinsky default), buoyancy_damping=True, damping_safety=3.5
LES referansındaki Re=10K_Ra1e5_v2/metrics.json ile birebir aynı setup.

Çıktı:
  data/sim_states/les_real_60k.npz       — tüm slice + 60 full snap
  data/sim_states/shared_ic_seed42.npz   — INNATE rollout için ortak IC
"""
from __future__ import annotations
import sys, os, time, gc
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import torch

from les_solver import LESSolver as LESSolver3D


def main():
    parser = argparse.ArgumentParser(description="Aggressive-Cs LES rollout for viz")
    parser.add_argument("--Re", type=float, default=10000.0)
    parser.add_argument("--Ra", type=float, default=1e5)
    parser.add_argument("--Pr", type=float, default=0.71)
    parser.add_argument("--Cs", type=float, default=0.17,
                        help="Smagorinsky katsayisi (gercek LES referans default 0.17)")
    parser.add_argument("--n-steps", type=int, default=60000,
                        help="Toplam step sayisi (60K = steady-state turbulent regime)")
    parser.add_argument("--snapshot-every", type=int, default=1,
                        help="Her N step'te state kaydet")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="data/sim_states/les_aggressive.npz")
    parser.add_argument("--device", type=str, default=None,
                        help="cpu|mps|cuda (default: auto)")
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("LES AGRESIF (ANSYS-tarzi detay icin)")
    print("=" * 80)
    print(f"Re={args.Re}, Ra={args.Ra:.0e}, Pr={args.Pr}, Cs={args.Cs}")
    print(f"Grid: 96×160×64, n_steps={args.n_steps}")
    print(f"Output: {output}")

    # LES solver
    solver = LESSolver3D(
        Nx=96, Ny=160, Nz=64,
        Lx=6.0, Ly=10.0, Lz=4.0,
        Re=args.Re, Ra=args.Ra, Pr=args.Pr,
        forcing_amplitude=0.005, k_f=4,
        Cs=args.Cs, Pr_t=0.85,
        cfl_target=0.5, dt_max=0.02, dt_min=1e-5,
        buoyancy_damping=True, damping_safety=3.5,
        device=args.device,
    )

    # IC
    u, v, w, theta, p = solver.create_initial_condition(
        noise_scale=0.01, seed=args.seed
    )
    device = u.device
    print(f"Device: {device}")

    # === KRİTİK: IC'yi diske kaydet (INNATE rollout aynısını yükleyecek) ===
    ic_path = output.parent / f"shared_ic_seed{args.seed}.npz"
    np.savez(
        ic_path,
        u=u[0].cpu().numpy().astype(np.float32),
        v=v[0].cpu().numpy().astype(np.float32),
        w=w[0].cpu().numpy().astype(np.float32),
        theta=theta[0].cpu().numpy().astype(np.float32),
        p=p[0].cpu().numpy().astype(np.float32),
        seed=args.seed, noise_scale=0.01,
    )
    print(f"Shared IC kaydedildi: {ic_path}")
    print(f"  IC istatistik: u_mean={u.mean().item():.4e}, theta_mean={theta.mean().item():.4e}")

    # State buffer'lari (RAM'de tut, sona toplu kaydet)
    n_save = args.n_steps // args.snapshot_every + 1
    Nx, Ny, Nz = 96, 160, 64

    # Disk maliyeti: 1800 step × (4 field × 96×160×64 × float32) = 1800 × 15.7 MB = 28 GB
    # Cozum: float16 sıkıştırma → 14 GB. Hâlâ büyük.
    # Daha iyi: Sadece SLICE'lar + her 30. step'te tam state.
    # Slice: y-orta plane (xz grid 96×64) + z-orta plane (xy 96×160)
    # Tam state: 60 snapshot × 15.7 MB = 943 MB

    full_snap_every = max(1, args.n_steps // 1800)  # ~1800 tam snapshot (akıcı 3D video için)
    print(f"Slice her step ({args.n_steps} frame), tam state her {full_snap_every} step (~1800 snap)")

    # Buffer
    slice_y_mid = []   # [n_steps, 4, 96, 64]  — y=Ly/2 ortasında xz slice (theta + |u|)
    slice_z_mid = []   # [n_steps, 4, 96, 160] — z=Lz/2 ortasında xy slice
    full_snaps = []    # [n_full, 4, 96, 160, 64]  — 3D volume render için
    metrics_t = []     # [n_steps] — TKE, Nu, theta_rms zaman serisi

    # Slice indeksleri
    j_mid = Ny // 2  # y orta
    k_mid = Nz // 2  # z orta

    t_start = time.time()
    t_sim = 0.0

    print(f"\n{'step':>6} {'t_sim':>8} {'TKE':>10} {'Nu':>8} {'theta_rms':>10} {'|u|max':>8} {'eta':>8}")
    print("=" * 80)

    # === FIXED dt — INNATE rollout ile zaman senkronu için ===
    # (Codex consultancy 2026-04-30: adaptive dt LES vs fixed dt_step INNATE arasında
    #  zaman drift'i 1-2 birim olabilirdi → fixed dt=0.02 ile zaman birebir eşleşir)
    solver.dt = 0.02
    print(f"[LES] dt FIXED = {solver.dt} (zaman drift'i önlemek için)")

    for step in range(1, args.n_steps + 1):
        # solver._update_dt(u, v, w)  # KAPATILDI — fixed dt
        u, v, w, theta, p = solver._rk4_step(u, v, w, theta)
        t_sim += solver.dt

        if torch.isnan(u).any() or torch.isnan(theta).any():
            print(f"\n⚠ NaN @ step {step}, dt={solver.dt:.6e}")
            break

        if step % args.snapshot_every == 0:
            with torch.no_grad():
                # Slices
                u_y = u[0, :, j_mid, :].cpu().numpy().astype(np.float32)
                v_y = v[0, :, j_mid, :].cpu().numpy().astype(np.float32)
                w_y = w[0, :, j_mid, :].cpu().numpy().astype(np.float32)
                th_y = theta[0, :, j_mid, :].cpu().numpy().astype(np.float32)
                slice_y_mid.append(np.stack([u_y, v_y, w_y, th_y], axis=0))

                u_z = u[0, :, :, k_mid].cpu().numpy().astype(np.float32)
                v_z = v[0, :, :, k_mid].cpu().numpy().astype(np.float32)
                w_z = w[0, :, :, k_mid].cpu().numpy().astype(np.float32)
                th_z = theta[0, :, :, k_mid].cpu().numpy().astype(np.float32)
                slice_z_mid.append(np.stack([u_z, v_z, w_z, th_z], axis=0))

                # Metrikler (her step)
                TKE = 0.5 * (u.pow(2) + v.pow(2) + w.pow(2)).mean().item()
                vT = (v * theta).mean().item()
                theta_rms = theta.pow(2).mean().sqrt().item()
                Nu = 1.0 + vT / (solver.kappa * solver.dT_over_Ly + 1e-10)
                u_max = max(u.abs().max().item(), v.abs().max().item(), w.abs().max().item())
                metrics_t.append({
                    "step": step, "t": t_sim,
                    "TKE": TKE, "Nu": Nu, "theta_rms": theta_rms,
                    "vT": vT, "u_max": u_max, "dt": solver.dt,
                })

        # Tam 3D volume snapshot her N step
        if step % full_snap_every == 0 or step == args.n_steps:
            with torch.no_grad():
                full = np.stack([
                    u[0].cpu().numpy().astype(np.float32),
                    v[0].cpu().numpy().astype(np.float32),
                    w[0].cpu().numpy().astype(np.float32),
                    theta[0].cpu().numpy().astype(np.float32),
                ], axis=0)
                full_snaps.append((step, t_sim, full))

        # Log
        if step % 50 == 0 or step == 1:
            elapsed = time.time() - t_start
            steps_per_sec = step / max(elapsed, 1e-6)
            eta = (args.n_steps - step) / max(steps_per_sec, 1e-6)
            eta_str = f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"
            m = metrics_t[-1] if metrics_t else dict(TKE=0, Nu=0, theta_rms=0, u_max=0)
            print(f"{step:>6} {t_sim:>8.3f} {m['TKE']:>10.4e} {m['Nu']:>8.2f} "
                  f"{m['theta_rms']:>10.4e} {m['u_max']:>8.4f} {eta_str:>8}")

        if step % 100 == 0:
            gc.collect()
            if torch.cuda.is_available() and device.type == "cuda":
                torch.cuda.empty_cache()
            elif (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and device.type == "mps"
            ):
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()

        # === Ara flush her 10K step (NaN/crash güvencesi, Gemini önerisi) ===
        if step % 10000 == 0 and step > 0:
            flush_path = output.parent / f"_flush_step{step:06d}.npz"
            slice_y_tmp = np.stack(slice_y_mid, axis=0) if slice_y_mid else np.zeros((0, 4, Nx, Nz))
            slice_z_tmp = np.stack(slice_z_mid, axis=0) if slice_z_mid else np.zeros((0, 4, Nx, Ny))
            try:
                np.savez_compressed(
                    flush_path,
                    slice_y=slice_y_tmp, slice_z=slice_z_tmp,
                    last_step=step, last_t=t_sim,
                )
                size_mb = flush_path.stat().st_size / 1e6
                print(f"  [ARA FLUSH @ step {step}] {flush_path.name} ({size_mb:.0f} MB)")
            except Exception as e:
                print(f"  [ARA FLUSH HATA] {e}")

    # === Kaydet ===
    print("\nKaydediliyor...")
    slice_y_arr = np.stack(slice_y_mid, axis=0) if slice_y_mid else np.zeros((0, 4, Nx, Nz))
    slice_z_arr = np.stack(slice_z_mid, axis=0) if slice_z_mid else np.zeros((0, 4, Nx, Ny))

    full_snaps_arr = np.stack([s[2] for s in full_snaps], axis=0) if full_snaps else np.zeros((0, 4, Nx, Ny, Nz))
    full_steps = np.array([s[0] for s in full_snaps])
    full_times = np.array([s[1] for s in full_snaps])

    metrics_arr = {
        k: np.array([m[k] for m in metrics_t])
        for k in ("step", "t", "TKE", "Nu", "theta_rms", "vT", "u_max", "dt")
    }

    np.savez_compressed(
        output,
        slice_y=slice_y_arr,           # [N, 4=(u,v,w,th), 96, 64]
        slice_z=slice_z_arr,           # [N, 4, 96, 160]
        full_snaps=full_snaps_arr,     # [60, 4, 96, 160, 64]
        full_steps=full_steps,
        full_times=full_times,
        Re=args.Re, Ra=args.Ra, Pr=args.Pr, Cs=args.Cs,
        Lx=6.0, Ly=10.0, Lz=4.0,
        nu=solver.nu, kappa=solver.kappa, Ri=solver.Ri,
        **{f"metric_{k}": v for k, v in metrics_arr.items()},
    )

    elapsed = time.time() - t_start
    size_mb = output.stat().st_size / 1e6
    print(f"\n✓ Kaydedildi: {output} ({size_mb:.1f} MB)")
    print(f"  Slice frames: {len(slice_y_mid)}, Full snapshots: {len(full_snaps)}")
    print(f"  Toplam süre: {elapsed/60:.1f} dakika")


if __name__ == "__main__":
    main()
