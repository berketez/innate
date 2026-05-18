"""
02_innate_rollout.py — INNATE eğitilmiş checkpoint ile rollout

Aynı IC'den (LES ile aynı seed → aynı initial state),
Aynı süre (2000 step × dt=0.02 = 40 zaman birim),
Eğitilmiş Tier 1+2+3 modeli ile rollout.

Çıktı: data/sim_states/innate_states.npz (LES ile aynı format)
"""
from __future__ import annotations
import sys, os, time, gc, argparse
from pathlib import Path

# Windows cp1254 unicode workaround
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from config import Config
from model import INNATE3D_MixedConvection, ThermalFluidState
from les_solver import LESSolver as LESSolver3D


def load_shared_ic(ic_path, device):
    """LES tarafından kaydedilen ortak IC'yi yükle.
    Bu garanti eder: LES ve INNATE rollout BIREBIR aynı başlangıç state'ten başlar.
    """
    print(f"Shared IC yükleniyor: {ic_path}")
    d = np.load(ic_path)
    u = torch.from_numpy(d["u"]).unsqueeze(0).to(device)
    v = torch.from_numpy(d["v"]).unsqueeze(0).to(device)
    w = torch.from_numpy(d["w"]).unsqueeze(0).to(device)
    theta = torch.from_numpy(d["theta"]).unsqueeze(0).to(device)
    p = torch.from_numpy(d["p"]).unsqueeze(0).to(device)
    print(f"  IC istatistik: u_mean={u.mean().item():.4e}, theta_mean={theta.mean().item():.4e}")
    return u, v, w, p, theta


def make_initial_condition_seeded(cfg, device, seed=42, noise_scale=0.01):
    """FALLBACK: shared IC yoksa kendi üret. LES'le tutarlı OLMAYABİLİR — uyarı verir."""
    print("⚠ FALLBACK IC üretimi — LES ile birebir aynı OLMAYABİLİR.")
    print("  Önerilen: önce LES çalıştır, shared_ic_seed42.npz oluşsun.")
    Nx, Ny, Nz = cfg.domain.Nx, cfg.domain.Ny, cfg.domain.Nz
    torch.manual_seed(seed)  # LES ile aynı global manual_seed
    shape = (1, Nx, Ny, Nz)
    u = noise_scale * torch.randn(shape, dtype=torch.float32, device=device)
    v = noise_scale * torch.randn(shape, dtype=torch.float32, device=device)
    w = noise_scale * torch.randn(shape, dtype=torch.float32, device=device)
    theta = noise_scale * torch.randn(shape, dtype=torch.float32, device=device)
    theta = theta - theta.mean(dim=(-3, -2, -1), keepdim=True)
    p = torch.zeros(shape, dtype=torch.float32, device=device)
    return u, v, w, p, theta


def main():
    parser = argparse.ArgumentParser(description="INNATE rollout for viz")
    parser.add_argument("--checkpoint", type=str,
                        default="results_v2/checkpoints/checkpoint_epoch000004.pt",
                        help="Tier 1+2+3 sonrası eğitilmiş ckpt")
    parser.add_argument("--Re", type=float, default=10000.0)
    parser.add_argument("--Ra", type=float, default=1e5)
    parser.add_argument("--Pr", type=float, default=0.71)
    parser.add_argument("--n-steps", type=int, default=2000,
                        help="Toplam fractional-step sayısı (LES ile aynı)")
    parser.add_argument("--snapshot-every", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="data/sim_states/innate_rollout.npz")
    parser.add_argument("--shared-ic", type=str,
                        default="data/sim_states/shared_ic_seed42.npz",
                        help="LES tarafından kaydedilen IC dosyası (önerilen)")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dt", type=float, default=None,
                        help="dt_layer (eğitimle uyumlu için 0.005; default config 0.02)")
    parser.add_argument("--use-spectral-cs", action="store_true",
                        help="Spectral-Cs modu (9905 param spectral ckpt için ZORUNLU)")
    parser.add_argument("--spectral-kx", type=int, default=5)
    parser.add_argument("--spectral-ky", type=int, default=8)
    parser.add_argument("--spectral-kz", type=int, default=6)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("INNATE EĞİTİLMİŞ ROLLOUT (LES ile aynı IC ve süre)")
    print("=" * 80)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Re={args.Re}, Ra={args.Ra:.0e}, Pr={args.Pr}, n_steps={args.n_steps}")

    cfg = Config()
    if args.device:
        cfg._device_override = args.device

    # dt override (eğitimle uyumlu olması için)
    if args.dt is not None:
        cfg.physics.dt = args.dt
        print(f"[CONFIG] dt_layer override: {args.dt}")

    # Spectral-Cs mode (9905 param spectral ckpt için zorunlu)
    if args.use_spectral_cs:
        cfg.model.use_spectral_cs = True
        cfg.model.spectral_cs_kx_max = args.spectral_kx
        cfg.model.spectral_cs_ky_max = args.spectral_ky
        cfg.model.spectral_cs_kz_max = args.spectral_kz
        cfg.model.use_mlp_sgs = False  # spectral aktifken MLP bypass
        print(f"[CONFIG] use_spectral_cs=True (kx,ky,kz)=({args.spectral_kx},{args.spectral_ky},{args.spectral_kz})")

    device = cfg.device
    print(f"Device: {device}")

    # Model
    model = INNATE3D_MixedConvection(cfg).to(device)
    model.set_physics(Re=args.Re, Ra=args.Ra, Pr=args.Pr)

    # Tier 1 freeze (kanonik param'lar) — checkpoint öncesi uygula
    if hasattr(model, 'buoyancies'):
        for b in model.buoyancies:
            if hasattr(b, 'buoyancy_strength'):
                with torch.no_grad():
                    b.buoyancy_strength.fill_(1.0)
                b.buoyancy_strength.requires_grad = False
    if hasattr(model, 'advections'):
        for a in model.advections:
            if hasattr(a, 'advection_modulator'):
                with torch.no_grad():
                    a.advection_modulator.fill_(1.0)
                a.advection_modulator.requires_grad = False
    if hasattr(model, 'thermal_advections'):
        for t in model.thermal_advections:
            if hasattr(t, 'thermal_adv_modulator'):
                with torch.no_grad():
                    t.thermal_adv_modulator.fill_(1.0)
                t.thermal_adv_modulator.requires_grad = False
    if hasattr(model, 'thermal_diffusions'):
        for d in model.thermal_diffusions:
            if hasattr(d, 'kappa_scale'):
                with torch.no_grad():
                    d.kappa_scale.fill_(1.0)
                d.kappa_scale.requires_grad = False
            for axis in ('kappa_scale_x', 'kappa_scale_y', 'kappa_scale_z'):
                if hasattr(d, axis):
                    p = getattr(d, axis)
                    with torch.no_grad():
                        p.fill_(1.0)
                    p.requires_grad = False

    # Checkpoint yükle (Tier 1 freeze sonrası, eğitilen SGS Cs/cs_thermal'lar ckpt'den gelir)
    if Path(args.checkpoint).exists():
        ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
        sd = ckpt.get("model", ckpt)
        skip = {k for k in sd if any(s in k for s in
                ("diff_ops.k", "ops.kx", "ops.ky", "ops.kz", "ops.k_sq",
                 "ops.dealias", "_elevator_mask"))}
        for k in skip:
            del sd[k]
        missing, unexpected = model.load_state_dict(sd, strict=False)
        print(f"Checkpoint yüklendi: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        print(f"⚠ Checkpoint bulunamadı: {args.checkpoint}")
        print(f"  Rastgele init ile devam (eğitim öncesi model)")

    # Forcing freeze (eğitimle aynı koşul)
    if hasattr(model, 'forcing') and hasattr(model.forcing, 'amplitude'):
        with torch.no_grad():
            model.forcing.amplitude.fill_(0.005)
        model.forcing.amplitude.requires_grad = False

    # IC: LES tarafından kaydedilen shared IC (önerilen) veya fallback
    used_shared_ic = False
    if Path(args.shared_ic).exists():
        u, v, w, p, theta = load_shared_ic(args.shared_ic, device)
        used_shared_ic = True
        print("  -> Shared IC zaten div-free + projection sonrasi, ek projection ATLA")
    else:
        print(f"⚠ Shared IC bulunamadı: {args.shared_ic}")
        u, v, w, p, theta = make_initial_condition_seeded(cfg, device, seed=args.seed)
        # Fallback IC için projection uygula
        u, v, w, p = model.projections[0](u, v, w)
    state = ThermalFluidState(u=u, v=v, w=w, p=p, theta=theta,
                              t=torch.zeros(1, device=device), rho=None)

    Nx, Ny, Nz = cfg.domain.Nx, cfg.domain.Ny, cfg.domain.Nz
    j_mid = Ny // 2
    k_mid = Nz // 2

    # Buffer
    slice_y_mid = []
    slice_z_mid = []
    full_snaps = []
    metrics_t = []

    # full snapshot her N step (~60 snapshot total)
    full_snap_every = max(1, args.n_steps // 1800)  # ~1800 tam snapshot (akıcı 3D video için)
    print(f"Slice her step ({args.n_steps} frame), tam state her {full_snap_every} step (~1800 snap)")

    model.eval()
    n_layers = cfg.model.n_layers
    dt_layer = model._dt_base
    dt_step = n_layers * dt_layer

    # INNATE'te bir model() çağrısı = n_layers adım = bir "step"
    # Yani n_steps INNATE step = n_steps fractional-step çağrısı
    # Toplam zaman = n_steps × n_layers × dt_layer = n_steps × dt_step

    # AMA LES'te bir step = bir RK4 = bir dt (≈0.02). LES dt_step = 0.02
    # INNATE dt_step = 20 × 0.02 = 0.4 — 20× daha uzun!
    # Aynı zaman için: INNATE 100 step ≈ LES 2000 step
    # Bu yüzden INNATE'i 100 step çalıştırıyoruz (LES 2000 step ile aynı zaman = 40 birim)

    n_innate_steps = args.n_steps // n_layers
    print(f"Not: INNATE 1 step = {n_layers} layer = {dt_step:.4f} zaman birim")
    print(f"     LES {args.n_steps} step ≈ INNATE {n_innate_steps} step (aynı zaman)")
    print(f"     Frame eşitlemek için INNATE her step = {n_layers} LES frame")

    t_sim = 0.0
    t_start = time.time()

    print(f"\n{'innate_step':>11} {'frame':>6} {'t_sim':>8} {'TKE':>10} {'Nu':>8} {'theta_rms':>10} {'eta':>8}")
    print("=" * 80)

    frame_idx = 0
    with torch.no_grad():
        for ist in range(1, n_innate_steps + 1):
            # Önceki state'i 20 ara state'e dönüştürmek için return_intermediates kullan
            intermediates = model(state, return_intermediates=True)
            # 20 ara state → 20 frame
            for i, st in enumerate(intermediates):
                frame_idx += 1
                if frame_idx > args.n_steps:
                    break

                with torch.no_grad():
                    u_y = st.u[0, :, j_mid, :].cpu().numpy().astype(np.float32)
                    v_y = st.v[0, :, j_mid, :].cpu().numpy().astype(np.float32)
                    w_y = st.w[0, :, j_mid, :].cpu().numpy().astype(np.float32)
                    th_y = st.theta[0, :, j_mid, :].cpu().numpy().astype(np.float32)
                    slice_y_mid.append(np.stack([u_y, v_y, w_y, th_y], axis=0))

                    u_z = st.u[0, :, :, k_mid].cpu().numpy().astype(np.float32)
                    v_z = st.v[0, :, :, k_mid].cpu().numpy().astype(np.float32)
                    w_z = st.w[0, :, :, k_mid].cpu().numpy().astype(np.float32)
                    th_z = st.theta[0, :, :, k_mid].cpu().numpy().astype(np.float32)
                    slice_z_mid.append(np.stack([u_z, v_z, w_z, th_z], axis=0))

                    TKE = 0.5 * (st.u.pow(2) + st.v.pow(2) + st.w.pow(2)).mean().item()
                    vT = (st.v * st.theta).mean().item()
                    theta_rms = st.theta.pow(2).mean().sqrt().item()
                    Nu = 1.0 + vT / (cfg.physics.kappa * (1.0 / cfg.domain.Ly) + 1e-10)
                    metrics_t.append(dict(
                        step=frame_idx, t=t_sim + (i + 1) * dt_layer,
                        TKE=TKE, Nu=Nu, theta_rms=theta_rms, vT=vT,
                    ))

                    # Tam 3D snapshot her N frame
                    if frame_idx % full_snap_every == 0 or frame_idx == args.n_steps:
                        full = np.stack([
                            st.u[0].cpu().numpy().astype(np.float32),
                            st.v[0].cpu().numpy().astype(np.float32),
                            st.w[0].cpu().numpy().astype(np.float32),
                            st.theta[0].cpu().numpy().astype(np.float32),
                        ], axis=0)
                        full_snaps.append((frame_idx, t_sim + (i + 1) * dt_layer, full))

            # Sonraki step için state güncelle
            _last = intermediates[-1]
            state = ThermalFluidState(
                u=_last.u.detach(), v=_last.v.detach(), w=_last.w.detach(),
                p=_last.p.detach(), theta=_last.theta.detach(),
                t=_last.t, rho=_last.rho.detach() if _last.rho is not None else None,
            )
            t_sim += dt_step

            # Memory hijyen
            del intermediates
            if ist % 10 == 0:
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

            # Log
            if ist % 5 == 0 or ist == 1:
                elapsed = time.time() - t_start
                steps_per_sec = ist / max(elapsed, 1e-6)
                eta = (n_innate_steps - ist) / max(steps_per_sec, 1e-6)
                eta_str = f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"
                m = metrics_t[-1] if metrics_t else dict(TKE=0, Nu=0, theta_rms=0)
                print(f"{ist:>11d} {frame_idx:>6d} {t_sim:>8.3f} "
                      f"{m['TKE']:>10.4e} {m['Nu']:>8.2f} {m['theta_rms']:>10.4e} {eta_str:>8}")

            if frame_idx >= args.n_steps:
                break

    # Kaydet
    print("\nKaydediliyor...")
    slice_y_arr = np.stack(slice_y_mid, axis=0) if slice_y_mid else np.zeros((0, 4, Nx, Nz))
    slice_z_arr = np.stack(slice_z_mid, axis=0) if slice_z_mid else np.zeros((0, 4, Nx, Ny))
    full_arr = np.stack([s[2] for s in full_snaps], axis=0) if full_snaps else np.zeros((0, 4, Nx, Ny, Nz))
    full_steps = np.array([s[0] for s in full_snaps])
    full_times = np.array([s[1] for s in full_snaps])
    metrics_arr = {
        k: np.array([m[k] for m in metrics_t])
        for k in ("step", "t", "TKE", "Nu", "theta_rms", "vT")
    }

    np.savez_compressed(
        output,
        slice_y=slice_y_arr,
        slice_z=slice_z_arr,
        full_snaps=full_arr,
        full_steps=full_steps,
        full_times=full_times,
        Re=args.Re, Ra=args.Ra, Pr=args.Pr,
        Lx=cfg.domain.Lx, Ly=cfg.domain.Ly, Lz=cfg.domain.Lz,
        nu=cfg.physics.nu, kappa=cfg.physics.kappa, Ri=cfg.physics.Ri,
        **{f"metric_{k}": v for k, v in metrics_arr.items()},
    )

    elapsed = time.time() - t_start
    size_mb = output.stat().st_size / 1e6
    print(f"\n✓ Kaydedildi: {output} ({size_mb:.1f} MB)")
    print(f"  Slice frames: {len(slice_y_mid)}, Full snapshots: {len(full_snaps)}")
    print(f"  Toplam süre: {elapsed/60:.1f} dakika")


if __name__ == "__main__":
    main()
