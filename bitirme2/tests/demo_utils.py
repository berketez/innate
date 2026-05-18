"""
demo_utils.py - Generalization Demo icin paylasilan altyapi modulu.

Her demo scripti bu moduldeki fonksiyonlari kullanir:
  - load_trained_model: Checkpoint yukle, fizik parametrelerini ayarla
  - run_simulation: N adim forward pass, metrik kaydi
  - compute_physics_metrics: Tek state uzerinde fiziksel buyuklukler
  - compute_energy_spectrum: Shell-averaged 1D enerji spektrumu E(k)
  - compute_spectrum_slope: Inertial range egerim tahmini
  - save_results: JSON formatinda sonuc kaydi
  - print_comparison_table: Formatlı karsilastirma tablosu

Formul referanslari:
  E_kin  = 0.5 * <u^2 + v^2 + w^2>           (domain ortalama kinetik enerji)
  Z      = <omega_x^2 + omega_y^2 + omega_z^2> (enstrophy, ops.curl ile)
  Nu     = 1 + <v*theta> / kappa               (Nusselt, theta=T'/dT nondim)
  E(k)   = shell-averaged spektral enerji       (k^{-5/3} beklenir)
  CFL    = max(|u|) * dt_eff / dx_min          (Courant-Friedrichs-Lewy)
  div    = max |nabla . u|                      (incompressibility olcusu)
  TKE    = E_kin (= 0.5 * <u_i * u_i>)

Yazar: Berke Tezgocen (tasarim), Claude (implementasyon)
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch

# -- path setup --
_this_dir = Path(__file__).resolve().parent
_project_dir = _this_dir.parent
sys.path.insert(0, str(_project_dir.parent))
sys.path.insert(0, str(_project_dir))

from config import Config, DomainConfig, PhysicsConfig
from model import INNATE3D_MixedConvection, ThermalFluidState


# =====================================================================
# 1. Model Yukleme
# =====================================================================


def load_trained_model(
    checkpoint_path: str,
    Re: float = None,
    Ra: float = None,
    Pr: float = 0.71,
    Nx: int = None,
    Ny: int = None,
    Nz: int = None,
    Lx: float = None,
    Ly: float = None,
    Lz: float = None,
    forcing_mode: str = None,
    non_boussinesq: bool = False,
    device: str = None,
) -> Tuple[INNATE3D_MixedConvection, Config, torch.device]:
    """
    Egitilmis INNATE3D modelini checkpoint'tan yukle ve yeni parametrelerle konfigure et.

    Akis:
      1. Config olustur (varsa custom domain/grid)
      2. Model olustur
      3. Checkpoint yukle (strict=False -- grid degisebilir)
      4. Re/Ra/Pr set et (model.set_physics)
      5. Forcing mode degistir (varsa)

    Args:
        checkpoint_path: Egitilmis model checkpoint dosyasi
        Re, Ra, Pr: Fizik parametreleri (None ise checkpoint'taki deger kalir)
        Nx, Ny, Nz: Grid boyutlari (None ise default 96x160x64)
        Lx, Ly, Lz: Domain boyutlari (None ise default 6x10x4)
        forcing_mode: "kolmogorov" | "uniform" | "stochastic" (None ise degismez)
        non_boussinesq: True ise Phase D noronlarini aktive et
        device: "cuda" | "mps" | "cpu" (None ise otomatik)

    Returns:
        (model, config, device)
    """
    cfg = Config()

    # Domain override
    if Nx is not None:
        cfg.domain.Nx = Nx
    if Ny is not None:
        cfg.domain.Ny = Ny
    if Nz is not None:
        cfg.domain.Nz = Nz
    if Lx is not None:
        cfg.domain.Lx = Lx
    if Ly is not None:
        cfg.domain.Ly = Ly
    if Lz is not None:
        cfg.domain.Lz = Lz

    # Forcing mode
    if forcing_mode is not None:
        cfg.physics.forcing_mode = forcing_mode

    # Non-Boussinesq
    cfg.physics.non_boussinesq = non_boussinesq
    cfg.model.use_non_boussinesq = non_boussinesq

    # Device
    if device is not None:
        cfg.device = device
    dev = cfg.device

    # Model olustur
    model = INNATE3D_MixedConvection(cfg).to(dev)

    # Checkpoint yukle (sadece learnable parametreler, buffer'lar atlanir)
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location=dev)
    model_buffers = {name for name, _ in model.named_buffers()}
    filtered_sd = {k: v for k, v in ckpt["model"].items() if k not in model_buffers}
    missing, unexpected = model.load_state_dict(filtered_sd, strict=False)
    epoch = ckpt.get("epoch", "?")
    print(f"  Checkpoint loaded: epoch {epoch}, missing={len(missing)}, unexpected={len(unexpected)}")

    # Re/Ra/Pr override
    if Re is not None or Ra is not None:
        _Re = Re if Re is not None else cfg.physics.Re
        _Ra = Ra if Ra is not None else cfg.physics.Ra
        _Pr = Pr
        model.set_physics(_Re, _Ra, _Pr)
        print(f"  Physics set: Re={_Re:.0f}, Ra={_Ra:.1e}, Pr={_Pr}")

    model.eval()
    return model, cfg, dev


# =====================================================================
# 2. Fiziksel Buyukluk Hesaplari
# =====================================================================


def compute_enstrophy(state: ThermalFluidState, ops) -> torch.Tensor:
    """
    Enstrophy: Z = <omega_x^2 + omega_y^2 + omega_z^2>

    omega = curl(u) = (dw/dy - dv/dz, du/dz - dw/dx, dv/dx - du/dy)
    Spectral curl: omega_i = i * epsilon_{ijk} * k_j * u_hat_k

    Returns: [B] tensor
    """
    ox, oy, oz = ops.curl(state.u, state.v, state.w)
    return (ox**2 + oy**2 + oz**2).mean(dim=(-3, -2, -1))


def compute_energy_spectrum(
    u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, ops
) -> torch.Tensor:
    """
    Shell-averaged 1D energy spectrum E(k).

    E(k) = sum_{|k| in [k-0.5, k+0.5)} 0.5 * |u_hat(k)|^2 / N^2

    Kolmogorov -5/3 yasasina uyum beklenir:
      E(k) = C_K * epsilon^{2/3} * k^{-5/3}

    Returns: spectrum [n_bins]
    """
    from innate import safe_fftn

    u_hat = safe_fftn(u)
    v_hat = safe_fftn(v)
    w_hat = safe_fftn(w)

    N_total = u.shape[-3] * u.shape[-2] * u.shape[-1]
    E_hat = 0.5 * (u_hat.abs() ** 2 + v_hat.abs() ** 2 + w_hat.abs() ** 2)
    E_hat = E_hat / N_total**2

    if E_hat.dim() == 4:
        E_hat = E_hat.mean(dim=0)

    k_mag = torch.sqrt(ops.k_squared)
    k_max = k_mag.max().item()
    n_bins = min(64, int(k_max) + 1)
    if n_bins < 2:
        return torch.ones(2, device=u.device) * 1e-10

    k_idx = (k_mag * n_bins / (k_max + 1e-10)).long().clamp(0, n_bins - 1)
    spectrum = torch.zeros(n_bins, device=u.device)
    counts = torch.zeros(n_bins, device=u.device)

    spectrum.scatter_add_(0, k_idx.flatten(), E_hat.flatten())
    counts.scatter_add_(
        0, k_idx.flatten(), torch.ones_like(k_idx.flatten(), dtype=torch.float32)
    )
    return spectrum / (counts + 1e-10)


def compute_spectrum_slope(spectrum: torch.Tensor, k_range: Tuple[int, int] = (6, 15)) -> float:
    """
    Inertial range icerisinde E(k) ~ k^alpha uyumu.

    log(E(k)) = alpha * log(k) + C
    En kucuk kareler ile alpha tahmin edilir.
    Hedef: alpha ≈ -5/3 ≈ -1.667

    Returns: slope (float), NaN eger yeterli veri yoksa
    """
    k = torch.arange(len(spectrum), device=spectrum.device, dtype=spectrum.dtype)
    k_min, k_max = k_range
    k_max = min(k_max, len(spectrum) - 1)

    mask = (k >= k_min) & (k <= k_max) & (spectrum > 1e-20)
    if mask.sum() < 3:
        return float("nan")

    log_k = torch.log(k[mask])
    log_E = torch.log(spectrum[mask])

    n = log_k.shape[0]
    sum_xy = (log_k * log_E).sum()
    sum_x = log_k.sum()
    sum_y = log_E.sum()
    sum_x2 = (log_k ** 2).sum()
    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2 + 1e-10)
    return slope.item()


def compute_physics_metrics(
    model: INNATE3D_MixedConvection,
    state: ThermalFluidState,
    config: Config,
) -> Dict[str, float]:
    """
    Tek state uzerinde tum fiziksel buyuklukleri hesapla.

    Formul ozeti:
      E_kin   = 0.5 * <u_i * u_i>              (kinetik enerji)
      Z       = <omega_i * omega_i>             (enstrophy)
      TKE     = E_kin                            (turbulant kinetik enerji)
      div_max = max |nabla . u|                  (incompressibility)
      Nu      = 1 + <v*theta> / kappa            (Nusselt, theta=T'/dT nondim)
      CFL     = max(|u|) * dt_eff / dx_min      (CFL sayisi)
      E(k)    = shell-averaged spectrum          (enerji spektrumu)
      slope   = d(log E)/d(log k) in inertial   (spektral egim)
      eps     = nu*Z + eps_sgs                  (toplam dissipasyon, ε=νZ)
      TI      = sqrt(2/3 * TKE) / U_mean       (turbulent intensity)
      eta     = (nu^3/eps)^{1/4}                (Kolmogorov scale)
      Re_stress = <u'_i * u'_j>                 (Reynolds stress tensoru)

    Returns: dict of metric_name -> value
    """
    d = config.domain
    p = config.physics

    # Kinetik enerji
    E_kin = state.kinetic_energy().mean().item()

    # Enstrophy
    Z = compute_enstrophy(state, model.ops).mean().item()

    # Divergence
    div_field = model.ops.divergence(state.u, state.v, state.w)
    div_max = div_field.abs().max().item()
    div_mean = div_field.abs().mean().item()

    # Nusselt: Nu = 1 + <v*theta> / kappa
    # theta nondimensional (T'/dT), kappa = 1/(Re*Pr)
    # Training ile AYNI formul (model.py:nusselt_number)
    vT_mean = (state.v * state.theta).mean(dim=(-3, -2, -1)).mean().item()
    Nu = 1.0 + vT_mean / (p.kappa + 1e-10)

    # CFL
    dt_eff = model._dt_base * torch.clamp(model.dt_scale, 0.5, 2.0).item()
    u_max = max(
        state.u.abs().max().item(),
        state.v.abs().max().item(),
        state.w.abs().max().item(),
    )
    CFL = u_max * dt_eff / model.dx_min

    # SGS parametrelerini ONCE oku (eps_sgs hesabi icin gerekli)
    cs_values = []
    prt_values = []
    if hasattr(model, 'eddy_viscosities'):
        for ev in model.eddy_viscosities:
            if hasattr(ev, 'cs_mid'):
                cs_values.append(ev.cs_mid.item())
            elif hasattr(ev, 'smagorinsky_coeff'):
                cs_values.append(ev.smagorinsky_coeff.item())
            if hasattr(ev, 'cs_thermal'):
                prt_values.append(ev.cs_thermal.item())
            elif hasattr(ev, 'pr_t'):  # eski checkpoint uyumu
                prt_values.append(ev.pr_t.item())

    # Enerji dissipasyon orani: eps = nu * Z
    # Periodic BC + div(u)=0 => 2<S_ij S_ij> = <omega^2> = Z
    # Dolayisiyla eps = 2*nu*<S_ij S_ij> = nu*Z
    eps_molecular = p.nu * Z

    # SGS dissipasyon tahmini
    # |S| = sqrt(2*S_ij*S_ij) = sqrt(Z) (vorticity-strain identity)
    # nu_sgs = (Cs*Delta)^2 * |S|
    # eps_sgs = nu_sgs * Z (= 2*nu_sgs*<S_ij S_ij>)
    dx_avg = (d.dx * d.dy * d.dz) ** (1.0 / 3.0)
    Cs_eff = sum(cs_values) / len(cs_values) if cs_values else 0.1
    S_mag = max(Z, 1e-20) ** 0.5
    nu_sgs = (Cs_eff * dx_avg) ** 2 * S_mag
    eps_sgs = nu_sgs * Z
    eps_total = eps_molecular + eps_sgs

    # Forcing gucunu hesapla: P_f = <F . u>
    Fx, Fy_f, Fz = model.forcing()
    P_forcing = (Fx * state.u + Fy_f * state.v + Fz * state.w).mean().item()

    # Buoyancy gucu: P_b = Ri * <v * T'>
    P_buoyancy = model.Ri * (state.v * state.theta).mean().item()

    # Spectrum slope
    spectrum = compute_energy_spectrum(state.u, state.v, state.w, model.ops)
    slope = compute_spectrum_slope(spectrum)

    # Sicaklik istatistikleri (boyutlu T)
    # theta nondimensional (T'/dT), boyutlu T = T_base + theta * dT
    y = torch.linspace(0, d.Ly, d.Ny + 1, device=state.theta.device)[:-1]
    y_grid = y.view(1, 1, d.Ny, 1)
    T_base = p.T_hot - (p.dT / d.Ly) * y_grid
    T_total = T_base + state.theta * p.dT
    T_min = T_total.min().item()
    T_max = T_total.max().item()
    theta_rms = state.theta.pow(2).mean().sqrt().item()

    # -- Turbulent Intensity: TI = sqrt(2/3 * TKE) / U_mean --
    # U_mean = domain-averaged velocity magnitude
    U_mean = (state.u.pow(2) + state.v.pow(2) + state.w.pow(2)).sqrt().mean().item()
    TI = ((2.0 / 3.0 * E_kin) ** 0.5) / (U_mean + 1e-10)

    # -- Reynolds Stress: <u'v'>, <u'w'>, <v'w'> --
    # u' = u - <u>, domain-ortalamadan sapma
    u_prime = state.u - state.u.mean(dim=(-3, -2, -1), keepdim=True)
    v_prime = state.v - state.v.mean(dim=(-3, -2, -1), keepdim=True)
    w_prime = state.w - state.w.mean(dim=(-3, -2, -1), keepdim=True)
    Re_uv = (u_prime * v_prime).mean().item()
    Re_uw = (u_prime * w_prime).mean().item()
    Re_vw = (v_prime * w_prime).mean().item()

    # -- Kolmogorov Scale: eta = (nu^3 / eps)^{1/4} --
    # LES grid kalitesinin temel olcusu: dx/eta < 20 → iyi LES
    eta = (p.nu ** 3 / (eps_total + 1e-20)) ** 0.25
    dx_over_eta = d.dx_min / (eta + 1e-20)

    # -- LES Quality Index (Pope kriteri) --
    # LES_IQ = resolved_TKE / total_TKE ~ 1 - (Cs*Delta/L_int)^2
    # Basit tahmin: resolved = E_kin, SGS ~ nu_sgs * S_mag
    k_sgs = nu_sgs * S_mag  # SGS kinetik enerji tahmini
    LES_IQ = E_kin / (E_kin + k_sgs + 1e-20)

    metrics = {
        "E_kin": E_kin,
        "TKE": E_kin,
        "Z_enstrophy": Z,
        "div_max": div_max,
        "div_mean": div_mean,
        "Nu": Nu,
        "CFL": CFL,
        "u_max": u_max,
        "dt_eff": dt_eff,
        "eps_molecular": eps_molecular,
        "eps_sgs": eps_sgs,
        "eps_total": eps_total,
        "P_forcing": P_forcing,
        "P_buoyancy": P_buoyancy,
        "spectrum_slope": slope,
        "T_min": T_min,
        "T_max": T_max,
        "theta_rms": theta_rms,
        # Yeni metrikler
        "TI": TI,
        "U_mean": U_mean,
        "Re_uv": Re_uv,
        "Re_uw": Re_uw,
        "Re_vw": Re_vw,
        "eta_kolmogorov": eta,
        "dx_over_eta": dx_over_eta,
        "LES_IQ": LES_IQ,
        "nu_sgs": nu_sgs,
    }

    if cs_values:
        metrics["Cs_mean"] = sum(cs_values) / len(cs_values)
        metrics["Cs_min"] = min(cs_values)
        metrics["Cs_max"] = max(cs_values)
    if prt_values:
        metrics["Pr_t_mean"] = sum(prt_values) / len(prt_values)

    # Non-Boussinesq ek metrikleri
    if state.rho is not None:
        metrics["rho_min"] = state.rho.min().item()
        metrics["rho_max"] = state.rho.max().item()
        metrics["rho_mean"] = state.rho.mean().item()

    return metrics


# =====================================================================
# 3. Simulasyon Calistirma
# =====================================================================


@dataclass
class SimulationResult:
    """Tek simulasyon calistirmasinin sonuclari."""
    name: str
    config_summary: Dict[str, float]
    n_steps: int
    wall_time: float
    stable: bool
    nan_step: int  # -1 ise NaN yok
    metrics_history: List[Dict[str, float]]  # her log_interval'da metrikler
    final_metrics: Dict[str, float]


def run_simulation(
    model: INNATE3D_MixedConvection,
    config: Config,
    device: torch.device,
    n_steps: int = 1000,
    log_interval: int = 50,
    name: str = "simulation",
    ic_state: ThermalFluidState = None,
    post_step_fn: Callable = None,
) -> SimulationResult:
    """
    Egitilmis model ile forward simulasyon calistir.

    Her adim = model(state) = 20-layer fractional-step NS solver.
    Toplam simulasyon suresi = n_steps * 20 * dt_eff zaman birimi.

    Args:
        model: Egitilmis INNATE3D modeli
        config: Config objesi (domain/physics parametreleri)
        device: torch device
        n_steps: Forward adim sayisi (her biri 20 katman iceriyor)
        log_interval: Her kac adimda metrik kaydedilecek
        name: Simulasyon adi (raporlama icin)
        ic_state: Ozel ilk kosul (None ise model.create_initial_condition)
        post_step_fn: Her adimdan sonra cagrilan fonksiyon
                      Imza: fn(state, step, model) -> state
                      Ornek: custom forcing eklemek icin

    Returns:
        SimulationResult
    """
    model.eval()

    # Forcing phase reset (training ile tutarli)
    if hasattr(model, 'forcing') and hasattr(model.forcing, 'reset_phase'):
        model.forcing.reset_phase()

    # IC
    if ic_state is not None:
        state = ic_state
    else:
        state = model.create_initial_condition(batch_size=1, device=device)

    dt_eff = model._dt_base * torch.clamp(model.dt_scale, 0.5, 2.0).item()
    total_sim_time = n_steps * model.n_layers * dt_eff

    print(f"\n{'='*70}")
    print(f"SIMULATION: {name}")
    print(f"{'='*70}")
    print(f"  Re={config.physics.Re:.0f}  Ra={config.physics.Ra:.1e}  "
          f"Pr={config.physics.Pr}  Ri={config.physics.Ri:.4f}")
    print(f"  Grid: {config.domain.Nx}x{config.domain.Ny}x{config.domain.Nz}")
    print(f"  Domain: {config.domain.Lx}x{config.domain.Ly}x{config.domain.Lz}")
    print(f"  Steps: {n_steps} x 20 layers = {n_steps * 20} total time steps")
    print(f"  dt_eff: {dt_eff:.4f}")
    print(f"  Simulation time: {total_sim_time:.1f} time units")
    print(f"  Forcing: {config.physics.forcing_mode}")
    print(f"  Device: {device}")
    print()

    # Header
    header = (f"{'Step':>6s}  {'E_kin':>10s}  {'Z':>10s}  {'div':>10s}  "
              f"{'Nu':>8s}  {'CFL':>8s}  {'slope':>8s}  {'TI':>8s}  {'dx/eta':>8s}")
    print(header)
    print("-" * len(header))

    metrics_history = []
    nan_step = -1
    t0 = time.time()

    with torch.no_grad():
        for step in range(n_steps):
            state = model(state)

            # Post-step hook (custom forcing vb.)
            if post_step_fn is not None:
                state = post_step_fn(state, step, model)

            # NaN check
            has_nan = (
                torch.isnan(state.u).any()
                or torch.isnan(state.v).any()
                or torch.isnan(state.w).any()
                or torch.isnan(state.theta).any()
                or (state.rho is not None and torch.isnan(state.rho).any())
            )
            if has_nan:
                nan_step = step
                print(f"\n*** NaN at step {step}! ***")
                break

            # Log
            if step % log_interval == 0 or step == n_steps - 1:
                m = compute_physics_metrics(model, state, config)
                metrics_history.append({"step": step, **m})

                slope_str = f"{m['spectrum_slope']:.3f}" if m['spectrum_slope'] == m['spectrum_slope'] else "N/A"
                ti_str = f"{m.get('TI', 0):.4f}"
                dx_eta_str = f"{m.get('dx_over_eta', 0):.2f}"
                print(f"{step:6d}  {m['E_kin']:10.6f}  {m['Z_enstrophy']:10.4f}  "
                      f"{m['div_max']:10.2e}  {m['Nu']:8.4f}  {m['CFL']:8.4f}  "
                      f"{slope_str:>8s}  {ti_str:>8s}  {dx_eta_str:>8s}")

    wall_time = time.time() - t0
    stable = (nan_step == -1)

    # Final metrics
    final_metrics = metrics_history[-1] if metrics_history else {}

    print()
    print(f"Completed: {step + 1 if stable else nan_step} steps in {wall_time:.1f}s "
          f"({wall_time / max(step + 1, 1):.3f}s/step)")
    if stable:
        print(f"STATUS: STABLE")
    else:
        print(f"STATUS: UNSTABLE (NaN at step {nan_step})")

    config_summary = {
        "Re": config.physics.Re,
        "Ra": config.physics.Ra,
        "Pr": config.physics.Pr,
        "Ri": config.physics.Ri,
        "nu": config.physics.nu,
        "kappa": config.physics.kappa,
        "Nx": config.domain.Nx,
        "Ny": config.domain.Ny,
        "Nz": config.domain.Nz,
        "Lx": config.domain.Lx,
        "Ly": config.domain.Ly,
        "Lz": config.domain.Lz,
        "forcing_mode": config.physics.forcing_mode,
    }

    return SimulationResult(
        name=name,
        config_summary=config_summary,
        n_steps=n_steps,
        wall_time=wall_time,
        stable=stable,
        nan_step=nan_step,
        metrics_history=metrics_history,
        final_metrics=final_metrics,
    )


# =====================================================================
# 4. Sonuc Kaydi ve Raporlama
# =====================================================================


def save_results(results: List[SimulationResult], output_path: str):
    """
    Simulasyon sonuclarini JSON formatinda kaydet.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for r in results:
        entry = {
            "name": r.name,
            "config": r.config_summary,
            "n_steps": r.n_steps,
            "wall_time": r.wall_time,
            "stable": r.stable,
            "nan_step": r.nan_step,
            "final_metrics": _sanitize(r.final_metrics),
            "metrics_history": [_sanitize(m) for m in r.metrics_history],
        }
        data.append(entry)

    path.write_text(json.dumps(data, indent=2, default=str))
    print(f"\nResults saved: {path}")


def _sanitize(d: dict) -> dict:
    """NaN ve Tensor'lari JSON-uyumlu yap."""
    out = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            v = v.item() if v.numel() == 1 else v.tolist()
        if isinstance(v, float) and v != v:
            v = "NaN"
        out[k] = v
    return out


def print_comparison_table(results: List[SimulationResult], title: str = ""):
    """
    Simulasyon sonuclarini karsilastirma tablosu olarak yazdir.

    Sutunlar: Name | Re | Ra | Ri | Stable | E_kin | Z | Nu | slope | CFL | Time
    """
    print()
    if title:
        print(f"{'='*90}")
        print(f"  {title}")
        print(f"{'='*90}")

    header = (f"{'Name':<30s} {'Re':>8s} {'Ra':>10s} {'Ri':>8s} {'OK':>4s} "
              f"{'E_kin':>10s} {'Nu':>8s} {'slope':>8s} {'CFL':>8s} {'Time':>8s}")
    print(header)
    print("-" * len(header))

    for r in results:
        c = r.config_summary
        m = r.final_metrics

        Re_str = f"{c.get('Re', 0):.0f}"
        Ra_str = f"{c.get('Ra', 0):.1e}"
        Ri_str = f"{c.get('Ri', 0):.4f}"
        ok_str = "OK" if r.stable else "FAIL"

        E_str = f"{m.get('E_kin', 0):.6f}" if m else "N/A"
        Nu_str = f"{m.get('Nu', 0):.4f}" if m else "N/A"
        slope_val = m.get('spectrum_slope', float('nan')) if m else float('nan')
        slope_str = f"{slope_val:.3f}" if slope_val == slope_val else "N/A"
        CFL_str = f"{m.get('CFL', 0):.4f}" if m else "N/A"
        time_str = f"{r.wall_time:.1f}s"

        print(f"{r.name:<30s} {Re_str:>8s} {Ra_str:>10s} {Ri_str:>8s} {ok_str:>4s} "
              f"{E_str:>10s} {Nu_str:>8s} {slope_str:>8s} {CFL_str:>8s} {time_str:>8s}")

    print()


def print_final_report(all_results: Dict[str, List[SimulationResult]]):
    """
    Tum demolarin sonuclarini tek bir rapor olarak yazdir.

    Args:
        all_results: {"Demo 1": [results], "Demo 2": [results], ...}
    """
    print()
    print("=" * 90)
    print("  INNATE 3D MIXED CONVECTION — GENERALIZATION DEMO REPORT")
    print("=" * 90)
    print()

    total_tests = 0
    total_stable = 0
    total_time = 0.0

    for demo_name, results in all_results.items():
        n = len(results)
        n_stable = sum(1 for r in results if r.stable)
        t = sum(r.wall_time for r in results)

        total_tests += n
        total_stable += n_stable
        total_time += t

        status = "ALL PASS" if n_stable == n else f"{n_stable}/{n} PASS"
        print(f"  {demo_name:<40s}  {status:<15s}  ({t:.1f}s)")

    print()
    print(f"  {'TOTAL':<40s}  {total_stable}/{total_tests} PASS  ({total_time:.1f}s)")
    print()

    # Basarisiz testleri listele
    failures = []
    for demo_name, results in all_results.items():
        for r in results:
            if not r.stable:
                failures.append(f"  - {demo_name}/{r.name}: NaN at step {r.nan_step}")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f)
        print()
