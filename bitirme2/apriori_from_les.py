"""A priori test: LES referans parametrelerini (Cs=0.17, Pr_t=0.85) INNATE'e
elle enjekte et, egitimsiz 1000 step rollout yap, LES metrikleriyle karsilastir.

Mantik: LES referansi zaten bu sabitlerle calismis (metrics.json'da Cs=0.17,
Pr_t=0.85 kayitli). INNATE architecture'i bu sabitleri alinca LES sonucuna
ulasabilmeli. Ulasamazsa sorun MLP'de degil mimaride (IMEX, forcing, elevator
mask, vs) veya IC farkinda.

Kullanim: python apriori_from_les.py --re 7000
"""
from __future__ import annotations
import argparse
import json
import sys
import os
from pathlib import Path
from typing import Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from config import Config
from model import INNATE3D_MixedConvection, ThermalFluidState


class FrozenMLPSGS(torch.nn.Module):
    """Drop-in replacement for MLPSGS with constant (Cs, Pr_t) outputs."""

    def __init__(self, Cs_value: float, Pr_t_value: float, n_layers: int = 20):
        super().__init__()
        self.Cs_value = Cs_value
        self.Pr_t_value = Pr_t_value
        self.n_layers = n_layers

    def forward(self, strain_mag, omega_mag, Ri_g, Re_normalized, layer_idx, delta):
        Cs = torch.full_like(strain_mag, self.Cs_value)
        Pr_t = torch.tensor(self.Pr_t_value, device=strain_mag.device, dtype=strain_mag.dtype)
        return Cs, Pr_t


def freeze_mlp_sgs(model: INNATE3D_MixedConvection, Cs_value: float, Pr_t_value: float) -> None:
    """Replace MLPSGS with FrozenMLPSGS in model AND all EddyViscosity3D layers."""
    if model.mlp_sgs is None:
        raise RuntimeError("model.mlp_sgs is None — freeze etmek icin MLP olmali")

    frozen = FrozenMLPSGS(Cs_value, Pr_t_value, n_layers=model.n_layers)
    model.mlp_sgs = frozen
    for ev in model.eddy_viscosities:
        ev.mlp_sgs = frozen
    print(f"  [freeze_mlp_sgs] Cs={Cs_value}, Pr_t={Pr_t_value} — MLP bypass aktif")
    print(f"    -> {len(model.eddy_viscosities)} EddyViscosity3D layer'a enjekte edildi")


def apply_ablations(model: INNATE3D_MixedConvection, ablations: list) -> None:
    """Ablation flag'lerini uygula. Bilesenleri tek tek kapatir."""
    if "no_forcing" in ablations:
        with torch.no_grad():
            model.forcing.amplitude.copy_(torch.tensor(0.0))
            if hasattr(model.forcing, 'amplitude_k2'):
                model.forcing.amplitude_k2.copy_(torch.tensor(0.0))
                model.forcing.amplitude_k3.copy_(torch.tensor(0.0))
        # Clamp [0.001, 0.02] yuzunden 0'a duselmez — clamp'i bypass et
        orig_fwd = model.forcing.forward
        def zero_fwd():
            zero = torch.zeros_like(model.forcing.y_grid)
            return zero, zero, zero
        model.forcing.forward = zero_fwd
        print(f"  [ablation] forcing OFF (Fx=Fy=Fz=0)")

    if "no_scale_sim" in ablations:
        for ev in model.eddy_viscosities:
            ev.use_scale_similarity = False
        print(f"  [ablation] scale_similarity OFF")

    if "no_backscatter" in ablations:
        for ev in model.eddy_viscosities:
            ev.use_backscatter = False
        print(f"  [ablation] backscatter OFF")

    if "no_elevator" in ablations:
        # Tum elevator mask'i 1'lerle doldur (u, v, w, theta hepsine uygulanan mask)
        if hasattr(model, '_elevator_mask'):
            with torch.no_grad():
                model._elevator_mask.fill_(1.0)
            print(f"  [ablation] elevator_mask OFF (tum alanlar icin)")
        else:
            print(f"  [ablation] elevator_mask buffer bulunamadi")

    if "no_elevator_theta" in ablations:
        # Sadece theta icin mask'i bypass et — model.py:599 satirini monkey-patch
        import model as _model_module
        original_layer_step = model._layer_step
        def _layer_step_no_theta_mask(self, *args, **kwargs):
            # Elevator mask'ten gecmeden once theta'nin kopyasini sakla
            return original_layer_step(*args, **kwargs)
        # Bu monkey-patch basit degil, direkt mask'i sadece theta'dan once bypass
        # Cozum: 2 parcali mask yarat, theta icin 1'ler, diger icin orijinal
        if not hasattr(model, '_elevator_mask_theta'):
            with torch.no_grad():
                model.register_buffer('_elevator_mask_theta',
                                      torch.ones_like(model._elevator_mask))
        print(f"  [ablation] no_elevator_theta — NOT YET WIRED, need model.py change")

    if "no_buoyancy" in ablations:
        # Buoyancy'yi devre disi birak
        for b in getattr(model, 'buoyancies', []):
            with torch.no_grad():
                if hasattr(b, 'strength'):
                    b.strength.copy_(torch.tensor(0.0))
        print(f"  [ablation] buoyancy OFF (strength=0)")


def compute_metrics(state: ThermalFluidState, model: INNATE3D_MixedConvection) -> dict:
    """TKE, enstrofi, max vel, divergence, nusselt, spectrum slope, vb."""
    u, v, w, theta = state.u, state.v, state.w, state.theta
    p = state.p if hasattr(state, 'p') else torch.zeros_like(u)

    tke = 0.5 * (u ** 2 + v ** 2 + w ** 2).mean().item()
    max_vel = torch.sqrt(u ** 2 + v ** 2 + w ** 2).max().item()

    # Divergence (spectral)
    ops = model.ops if hasattr(model, 'ops') else None
    if ops is not None:
        try:
            u_hat = ops.to_hat(u) if hasattr(ops, 'to_hat') else None
            v_hat = ops.to_hat(v) if hasattr(ops, 'to_hat') else None
            w_hat = ops.to_hat(w) if hasattr(ops, 'to_hat') else None
            if u_hat is not None and v_hat is not None and w_hat is not None:
                du_dx = ops.gradient_from_hat(u_hat)[0]
                dv_dy = ops.gradient_from_hat(v_hat)[1]
                dw_dz = ops.gradient_from_hat(w_hat)[2]
                div = du_dx + dv_dy + dw_dz
                div_rms = torch.sqrt((div ** 2).mean()).item()
            else:
                div_rms = float('nan')
        except Exception:
            div_rms = float('nan')
    else:
        div_rms = float('nan')

    # v*theta flux (Nusselt surrogate)
    v_theta = (v * theta).mean().item()

    # Enstrophy via curl (approx)
    if ops is not None:
        try:
            u_hat = ops.to_hat(u)
            v_hat = ops.to_hat(v)
            w_hat = ops.to_hat(w)
            du_dx, du_dy, du_dz = ops.gradient_from_hat(u_hat)
            dv_dx, dv_dy, dv_dz = ops.gradient_from_hat(v_hat)
            dw_dx, dw_dy, dw_dz = ops.gradient_from_hat(w_hat)
            ox = dw_dy - dv_dz
            oy = du_dz - dw_dx
            oz = dv_dx - du_dy
            enstrophy = 0.5 * (ox ** 2 + oy ** 2 + oz ** 2).mean().item()
        except Exception:
            enstrophy = float('nan')
    else:
        enstrophy = float('nan')

    theta_rms = torch.sqrt((theta ** 2).mean()).item()

    return {
        "TKE": tke,
        "enstrophy": enstrophy,
        "max_velocity": max_vel,
        "div_rms": div_rms,
        "v_theta_flux": v_theta,
        "theta_rms": theta_rms,
    }


def compute_spectrum_slope(state: ThermalFluidState, model: INNATE3D_MixedConvection) -> float:
    """k-bazli enerji spektrumu fit: E(k) ~ k^slope."""
    u, v, w = state.u, state.v, state.w
    N_total = u.numel()

    u_hat = torch.fft.fftn(u, dim=(-3, -2, -1)) / N_total
    v_hat = torch.fft.fftn(v, dim=(-3, -2, -1)) / N_total
    w_hat = torch.fft.fftn(w, dim=(-3, -2, -1)) / N_total

    E_3d = 0.5 * (u_hat.abs() ** 2 + v_hat.abs() ** 2 + w_hat.abs() ** 2)

    _, Nx, Ny, Nz = u.shape
    Lx = model.cfg.domain.Lx if hasattr(model, 'cfg') else 6.0
    Ly = model.cfg.domain.Ly if hasattr(model, 'cfg') else 10.0
    Lz = model.cfg.domain.Lz if hasattr(model, 'cfg') else 4.0
    kx = torch.fft.fftfreq(Nx, Lx / Nx) * 2 * torch.pi
    ky = torch.fft.fftfreq(Ny, Ly / Ny) * 2 * torch.pi
    kz = torch.fft.fftfreq(Nz, Lz / Nz) * 2 * torch.pi
    KX, KY, KZ = torch.meshgrid(kx, ky, kz, indexing='ij')
    k_mag = torch.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)

    k_max = int(min(Nx, Ny, Nz) / 2)
    # kk and E_k aligned: indices 0..k_max-1 correspond to k=0..k_max-1
    kk = torch.arange(k_max, dtype=torch.float32, device=u.device)
    E_k = torch.zeros(k_max, device=u.device)
    for ki in range(1, k_max):
        mask = (k_mag >= ki - 0.5) & (k_mag < ki + 0.5)
        E_k[ki] = E_3d[0][mask].sum() if mask.any() else 0.0

    # Log-log fit in inertial range [k=3, k=12]
    valid = (kk >= 3) & (kk <= 12) & (E_k > 1e-20)
    if valid.sum() < 3:
        return float('nan')
    lk = torch.log(kk[valid])
    lE = torch.log(E_k[valid])
    # Linear regression
    n = valid.sum().float()
    slope = (n * (lk * lE).sum() - lk.sum() * lE.sum()) / (n * (lk ** 2).sum() - lk.sum() ** 2)
    return slope.item()


def run_rollout(
    model: INNATE3D_MixedConvection,
    n_steps: int,
    device: torch.device,
    seed: int = 42,
    spinup_steps: int = 0,
    sample_every: int = 5,
) -> Tuple[ThermalFluidState, dict, list]:
    """Random IC uret, rollout yap, her sample_every adimda metric kaydet."""
    torch.manual_seed(seed)
    state = model.create_initial_condition(batch_size=1, device=device)

    # IC metric'lerini de kaydet (step 0)
    m0 = compute_metrics(state, model)
    m0["step"] = 0
    metrics_history = [m0]

    for step in range(1, n_steps + 1):
        with torch.no_grad():
            state = model(state)

        if torch.isnan(state.u).any() or torch.isinf(state.u).any():
            print(f"  [step {step}] NaN/Inf — rollout patladi!")
            break

        if step % sample_every == 0 or step == n_steps:
            m = compute_metrics(state, model)
            m["step"] = step
            metrics_history.append(m)
            # Erken patlama sinyali bas
            if m["TKE"] > 1.0:
                print(f"  [step {step}] TKE={m['TKE']:.3f} max_vel={m['max_velocity']:.2f} — PATLAMA ISARETI")

    final_metrics = compute_metrics(state, model)
    final_metrics["spectrum_slope"] = compute_spectrum_slope(state, model)
    return state, final_metrics, metrics_history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--re", type=int, default=7000, choices=[7000, 10000])
    parser.add_argument("--n_steps", type=int, default=1000)
    parser.add_argument("--spinup", type=int, default=0)
    parser.add_argument("--sample_every", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--ablation", type=str, default="",
                        help="Comma-separated: no_forcing,no_scale_sim,no_backscatter,no_elevator,no_buoyancy")
    parser.add_argument("--forcing-amp", type=float, default=None,
                        help="Override forcing amplitude (default: LES value 0.005, clamped to [0.001, 0.02])")
    parser.add_argument("--buoyancy-scale", type=float, default=None,
                        help="Override buoyancy strength per layer")
    parser.add_argument("--prt", type=float, default=None,
                        help="Override turbulent Prandtl number (default 0.85)")
    args = parser.parse_args()

    device = torch.device(args.device)

    # LES reference yukle
    les_dir = Path(__file__).parent / "les_reference"
    les_key = f"Re{args.re}_Ra1e5" if args.re == 7000 else f"Re{args.re}_Ra1e5_v2"
    les_path = les_dir / les_key / "metrics.json"
    if not les_path.exists():
        print(f"LES referans yok: {les_path}")
        sys.exit(1)
    with open(les_path) as f:
        les_ref = json.load(f)
    les_final = les_ref["final_metrics"]
    les_params = les_ref["params"]
    print(f"\n=== LES Referans: Re={args.re} ===")
    print(f"  Cs = {les_params['Cs']}, Pr_t = {les_params['Pr_t']}")
    print(f"  TKE = {les_final['TKE']:.4e}")
    print(f"  slope = {les_final['spectrum_slope']:.3f}")
    print(f"  Nu = {les_final['nusselt']:.2f}")
    print(f"  max_vel = {les_final['max_velocity']:.3f}")
    print(f"  div_error = {les_final['div_error']:.3e}")
    print(f"  v_theta_flux = {les_final['v_theta_flux']:.3e}")
    print(f"  mean_nu_t = {les_final.get('mean_nu_t', float('nan')):.3e}")

    # INNATE olustur
    cfg = Config()
    model = INNATE3D_MixedConvection(cfg).to(device)
    model.eval()
    model.set_physics(Re=float(args.re), Ra=1e5, Pr=0.71)

    # MLP'yi LES parametreleriyle freeze et (Pr_t override varsa kullan)
    prt_val = args.prt if args.prt is not None else les_params['Pr_t']
    freeze_mlp_sgs(model, Cs_value=les_params['Cs'], Pr_t_value=prt_val)

    # Ablation'lari uygula
    ablations = [a.strip() for a in args.ablation.split(",") if a.strip()]
    if ablations:
        apply_ablations(model, ablations)

    # Forcing amp override
    if args.forcing_amp is not None:
        with torch.no_grad():
            model.forcing.amplitude.copy_(torch.tensor(args.forcing_amp))
        print(f"  [override] forcing amplitude = {args.forcing_amp}")

    # Buoyancy scale override (her katman veya global)
    if args.buoyancy_scale is not None:
        count = 0
        if hasattr(model, 'buoyancies'):
            for b in model.buoyancies:
                with torch.no_grad():
                    b.buoyancy_strength.copy_(torch.tensor(args.buoyancy_scale))
                count += 1
        elif hasattr(model, 'buoyancy'):
            with torch.no_grad():
                model.buoyancy.buoyancy_strength.copy_(torch.tensor(args.buoyancy_scale))
            count = 1
        print(f"  [override] buoyancy_strength = {args.buoyancy_scale} ({count} layer)")

    # Rollout
    print(f"\n=== INNATE + Frozen(Cs={les_params['Cs']}, Pr_t={les_params['Pr_t']}) Rollout ===")
    print(f"  {args.n_steps} step, sample every {args.sample_every}, seed {args.seed}")
    _, final, history = run_rollout(
        model, n_steps=args.n_steps, device=device,
        seed=args.seed, spinup_steps=args.spinup,
        sample_every=args.sample_every,
    )

    # TKE/max_vel zaman evrimi + LES history karsilastirma
    print(f"\n=== TKE EVOLUTION (INNATE vs LES) ===")
    print(f"{'step':>5s} | {'t':>6s} | {'TKE_us':>10s} | {'max_vel':>8s} | {'theta_rms':>9s} | {'TKE_LES(step)':>14s}")
    print("-" * 72)
    les_history = les_ref.get("metrics_history", [])
    dt = les_params.get("dt", 0.02)
    for m in history:
        t = m["step"] * dt
        les_idx = min(m["step"], len(les_history) - 1) if les_history else -1
        les_tke_str = f"{les_history[les_idx].get('TKE', 0):.3e}" if les_idx >= 0 else "n/a"
        print(f"{m['step']:>5d} | {t:>6.2f} | {m['TKE']:>10.3e} | {m['max_velocity']:>8.3f} | {m['theta_rms']:>9.3e} | {les_tke_str:>14s}")

    # Save history to JSON
    out_path = Path(__file__).parent / f"apriori_history_Re{args.re}.json"
    with open(out_path, "w") as f:
        json.dump({"history": history, "final": final,
                   "les_params": les_params, "les_final": les_final,
                   "les_history": les_history}, f, indent=2)
    print(f"\n  History saved: {out_path.name}")

    print(f"\n  Final metrics:")
    for k, v in final.items():
        if isinstance(v, float):
            print(f"    {k:20s} = {v:.4e}" if abs(v) < 1e-2 or abs(v) > 1e3 else f"    {k:20s} = {v:.4f}")

    # Karsilastirma tablosu
    print(f"\n=== KARSILASTIRMA: INNATE(frozen) vs LES(Re={args.re}) ===")
    print(f"{'metric':<20s} | {'LES':>12s} | {'INNATE':>12s} | {'oran':>8s}")
    print("-" * 62)
    cmp_keys = [
        ("TKE", "TKE"),
        ("max_velocity", "max_velocity"),
        ("enstrophy", "enstrophy"),
        ("div_error", "div_rms"),
        ("v_theta_flux", "v_theta_flux"),
        ("theta_rms", "theta_rms"),
        ("spectrum_slope", "spectrum_slope"),
    ]
    for les_k, our_k in cmp_keys:
        les_v = les_final.get(les_k, float('nan'))
        our_v = final.get(our_k, float('nan'))
        ratio = our_v / les_v if (les_v != 0 and les_v == les_v) else float('nan')
        print(f"{les_k:<20s} | {les_v:>12.4e} | {our_v:>12.4e} | {ratio:>8.3f}")

    # Ozet
    print(f"\n=== OZET ===")
    tke_ratio = final["TKE"] / les_final["TKE"] if les_final["TKE"] else float('nan')
    slope_ok = abs(final["spectrum_slope"] - les_final["spectrum_slope"]) < 0.3 \
        if final["spectrum_slope"] == final["spectrum_slope"] else False
    stable = not (torch.isnan(torch.tensor(final["TKE"])) or final["TKE"] > 10 * les_final["TKE"])

    print(f"  TKE oran LES'e: {tke_ratio:.2f}x  ({'OK' if 0.3 < tke_ratio < 3.0 else 'UZAK'})")
    print(f"  Slope eslesme: {'OK' if slope_ok else 'UZAK'} (fark {abs(final['spectrum_slope'] - les_final['spectrum_slope']):.2f})")
    print(f"  Rollout stable: {'OK' if stable else 'PATLAMA'}")
    print(f"  Div_rms: {final['div_rms']:.2e} (LES: {les_final['div_error']:.2e})")

    # Sonuc: A mi B mi?
    print(f"\n=== KARAR ===")
    if stable and 0.3 < tke_ratio < 3.0 and slope_ok:
        print("  -> INNATE arch sabit closure ile LES'e YAKIN sonuc veriyor.")
        print("     Demek ki MLP'nin isi zaten bu sabit degerleri bulmakti.")
        print("     MLP ogreniminde sorun var ama arch saglam.")
        print("     TAVSIYE: A yolu (FiLM+pushforward) cogerli ama B (dinamik Smag) da uygulanabilir.")
    elif stable and not (0.3 < tke_ratio < 3.0):
        print("  -> INNATE sabit closure ile stable ama TKE uyusmuyor.")
        print("     Arch'da sistematik bir problem var (forcing, elevator mask, IMEX, IC).")
        print("     TAVSIYE: Arch-level debug + A yolu. B tek basina cozmez.")
    else:
        print("  -> INNATE sabit closure ile patladi veya divergence sorunlu.")
        print("     Arch'da yapisal sorun var. MLP ile maskelenmis olabilir.")
        print("     TAVSIYE: Arch-level audit + B gerekir (MLP'siz saf INNATE test).")


if __name__ == "__main__":
    main()
