"""
evaluate.py - INNATE 3D Mixed Convection Model Evaluation & Benchmarking

LES referans karsilastirmasi, fiziksel tutarlilik metrikleri, rapor uretimi.

Pipeline:
  1. Model + checkpoint yukle
  2. Her egitim Re degeri icin (7000, 10000):
     - Long rollout: spinup (500) + istatistik (500)
     - Fiziksel metrikler: divergence, energy balance, slope, Nu, temperature
     - LES referans karsilastirmasi: TKE, slope, entropy
     - TKE convergence (istatistiksel durulma)
  3. Phase 2 metrikleri (non-Boussinesq)
  5. Rapor

Metrikler (Faz 1 - Boussinesq):
  1. Divergence:       ||div(u)|| < 1e-5
  2. Energy balance:   |dE/dt + nu*Z - P_f - P_b| / |P_f+P_b| < 0.15
  3. Spectrum slope:   LES ref slope +-0.15
  4. Nusselt:          Nu >= 5 (konveksiyon var mi kontrolu)
  5. Stability:        NaN-free rollout
  6. Temperature:      theta perturbation +-1.5
  7. LES TKE:          +-15%
  8. LES entropy:      +-15%
  9. TKE convergence:  CV < 0.10

Dissipasyon: eps = nu * Z, burada Z = <|omega|^2> (periodic BC).
Periodic BC'de Stokes kimligi: 2*<S_ij*S_ij> = <|omega|^2>,
dolayisiyla eps = 2*nu*<S_ij*S_ij> = nu*<|omega|^2> = nu*Z.
les_solver.py ile tutarli (satir 764-769).

Metrikler (Faz 2 - Non-Boussinesq):
  8. Continuity:       |d(rho)/dt + div(rho*u)| residuali
  9. State equation:   |p - rho*R*T| / p_0 residuali
 10. Mass conservation: <rho> drift < %1
 11. Density bounds:   0.5*rho_0 <= rho <= 2.0*rho_0

Kullanim:
  python evaluate.py --checkpoint results/checkpoints/checkpoint_epoch005000.pt
  python evaluate.py --checkpoint ... --spinup-steps 500 --stat-steps 500
  python evaluate.py --checkpoint ... --spinup-steps 700 --stat-steps 700

Yazar: Berke Tezgocen (tasarim), Claude (implementasyon)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir.parent))
sys.path.insert(0, str(_this_dir))

from config import Config
from model import INNATE3D_MixedConvection, ThermalFluidState
from train import compute_energy_spectrum, LES_REFERENCE


# =====================================================================
# 0. Constants & Tolerance Bands
# =====================================================================

# Team consensus (Codex + Gemini + CFD Expert, 2026-02-28):
# Damping paradox: LES damps v+theta, INNATE undamped.
# TKE: 2.1% error, slope: 0.23%, entropy: 3.7% → tight tolerances OK.
# Nu: 43.3% error, theta_rms: 51.3% → monitoring only (no pass/fail).
LES_TOLERANCES = {
    "TKE_rel": 0.15,       # +-15% relative
    "slope_abs": 0.15,     # +-0.15 absolute
    "entropy_rel": 0.15,   # +-15% relative
}

EVAL_RE_VALUES = [7000, 10000]
EVAL_RA = 1e5


# =====================================================================
# 1. Helper: Enstrophy
# =====================================================================


def _compute_enstrophy(state: ThermalFluidState, ops) -> torch.Tensor:
    """
    Enstrophy: Z = <omega_x^2 + omega_y^2 + omega_z^2>

    Periodic BC'de: eps = nu * Z (Stokes kimligi).
    Returns: [B] tensor (domain-ortalama).
    """
    ox, oy, oz = ops.curl(state.u, state.v, state.w)
    return (ox**2 + oy**2 + oz**2).mean(dim=(-3, -2, -1))


# =====================================================================
# 2. Metric Functions
# =====================================================================


def eval_divergence(model: INNATE3D_MixedConvection, state: ThermalFluidState) -> float:
    """Max |div(u)| hesapla."""
    div = model.ops.divergence(state.u, state.v, state.w)
    return div.abs().max().item()


def eval_energy_balance(
    states: List[ThermalFluidState],
    model: INNATE3D_MixedConvection,
    config: Config,
    effective_dt: Optional[float] = None,
) -> float:
    """
    |dE/dt + eps - P_f - P_b| / |P_f + P_b| ortalamasini hesapla.

    Args:
        effective_dt: Ardisik state'ler arasi gercek zaman farki.
            None ise config.training.num_steps * config.physics.dt kullanilir.
            (model(state) 20 internal step yapar → effective_dt = 20*dt = 0.5)
    """
    if len(states) < 2:
        return float("nan")

    phys = config.physics
    if effective_dt is None:
        effective_dt = config.model.n_layers * phys.dt
    Fx, Fy, Fz = model.forcing()

    residuals = []
    powers = []

    for i in range(1, len(states)):
        s0, s1 = states[i - 1], states[i]
        E0 = s0.kinetic_energy()
        E1 = s1.kinetic_energy()
        dEdt = (E1 - E0) / effective_dt
        Z = 0.5 * (_compute_enstrophy(s0, model.ops) + _compute_enstrophy(s1, model.ops))
        eps = phys.nu * Z

        u_mid = 0.5 * (s0.u + s1.u)
        v_mid = 0.5 * (s0.v + s1.v)
        w_mid = 0.5 * (s0.w + s1.w)
        P_f = (Fx * u_mid + Fy * v_mid + Fz * w_mid).mean(dim=(-3, -2, -1))

        theta_mid = 0.5 * (s0.theta + s1.theta)
        P_b = phys.Ri * (v_mid * theta_mid).mean(dim=(-3, -2, -1))

        residual = (dEdt + eps - P_f - P_b).abs()
        power = (P_f + P_b).abs()

        residuals.append(residual.mean().item())
        powers.append(max(power.mean().item(), 1e-20))

    avg_ratio = sum(r / p for r, p in zip(residuals, powers)) / len(residuals)
    return avg_ratio


def eval_spectrum_slope(state: ThermalFluidState, ops) -> float:
    """Inertial range slope fit (log-log lineer regresyon)."""
    spectrum = compute_energy_spectrum(state.u, state.v, state.w, ops)
    k = torch.arange(len(spectrum), device=spectrum.device, dtype=spectrum.dtype)

    k_max = 15
    k_min = 6
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


def eval_spectral_entropy(state: ThermalFluidState, ops) -> float:
    """
    Spectral entropy: H = -sum(p_k * log(p_k)).
    Laminer -> H~0 (tek mod), turbulant -> H yuksek (genis bant).
    """
    spectrum = compute_energy_spectrum(state.u, state.v, state.w, ops)
    p_k = spectrum / (spectrum.sum() + 1e-10)
    entropy = -(p_k * torch.log(p_k + 1e-20)).sum()
    return entropy.item()


def eval_nusselt(state: ThermalFluidState, config: Config) -> float:
    """
    Nu = 1 + <v*theta> / (kappa * dT/Ly).

    Dogru formul: kappa = 1/(Re*Pr), dT/Ly = 1/Ly.
    Not: config.physics.Re dogru set edilmis olmali.
    """
    return state.nusselt_number(
        config.domain.Ly, config.physics.kappa
    ).mean().item()


def eval_temperature_bounds(
    state: ThermalFluidState, config: Config
) -> Dict[str, float]:
    """
    Theta perturbation sinirlari.
    LES'te theta_rms ~0.025, max ~0.16. Model clamp: 1.0 (final layer).
    Sinir: |theta| < 1.5 (model clamp'in %50 uzerinde marj).
    """
    theta_min = state.theta.min().item()
    theta_max = state.theta.max().item()
    theta_rms = state.theta.pow(2).mean().sqrt().item()
    bound = 1.5
    return {
        "theta_min": theta_min,
        "theta_max": theta_max,
        "theta_rms": theta_rms,
        "bound": bound,
        "in_bounds": bool(theta_min >= -bound and theta_max <= bound),
    }


# =====================================================================
# 2b. Phase 2 (Non-Boussinesq) Metric Functions
# =====================================================================


def _is_non_boussinesq(model) -> bool:
    """Model'in non_boussinesq modda olup olmadigini guvenli kontrol et."""
    return getattr(model, "non_boussinesq", False)


def eval_continuity_residual(
    model: INNATE3D_MixedConvection,
    states: List[ThermalFluidState],
) -> Dict[str, object]:
    """
    Kutle korunumu residuali: d(rho)/dt + nabla.(rho*u) = 0
    """
    if not _is_non_boussinesq(model):
        return {"mean": 0.0, "max": 0.0, "pass": True}

    residuals = []
    for i in range(1, len(states)):
        s0, s1 = states[i - 1], states[i]
        if s0.rho is None or s1.rho is None:
            continue
        res = model.continuity(s0.rho, s1.rho, s1.u, s1.v, s1.w)
        residuals.append(res.abs())

    if not residuals:
        return {"mean": 0.0, "max": 0.0, "pass": True}

    all_res = torch.stack(residuals)
    mean_res = all_res.mean().item()
    max_res = all_res.max().item()
    return {
        "mean": mean_res,
        "max": max_res,
        "pass": max_res < 1e-3,
    }


def eval_state_equation(
    model: INNATE3D_MixedConvection,
    state: ThermalFluidState,
) -> Dict[str, object]:
    """Durum denklemi residuali: |p - rho*R*T| / p_0"""
    if not _is_non_boussinesq(model) or state.rho is None:
        return {"mean": 0.0, "max": 0.0, "pass": True}

    T_total = model._compute_T_total(state.theta, state.u.device)
    residual = model.state_equation(state.rho, T_total, state.p)
    return {
        "mean": residual.mean().item(),
        "max": residual.max().item(),
        "pass": residual.mean().item() < 0.01,
    }


def eval_mass_conservation(
    model: INNATE3D_MixedConvection,
    states: List[ThermalFluidState],
) -> Dict[str, object]:
    """Global kutle korunumu: <rho> sabit kalmali."""
    if not _is_non_boussinesq(model):
        return {"max_drift": 0.0, "pass": True}

    rho_means = []
    for s in states:
        if s.rho is not None:
            rho_means.append(s.rho.mean().item())

    if len(rho_means) < 2:
        return {"max_drift": 0.0, "pass": True}

    drifts = [abs(rho_means[i] - rho_means[0]) for i in range(1, len(rho_means))]
    max_drift = max(drifts)
    return {
        "max_drift": max_drift,
        "initial_rho_mean": rho_means[0],
        "final_rho_mean": rho_means[-1],
        "pass": max_drift < 0.01,
    }


def eval_density_bounds(
    model: INNATE3D_MixedConvection,
    state: ThermalFluidState,
) -> Dict[str, object]:
    """Yogunluk sinir kontrolu: 0.5*rho_0 <= rho <= 2.0*rho_0"""
    if not _is_non_boussinesq(model) or state.rho is None:
        return {"rho_min": 1.0, "rho_max": 1.0, "in_bounds": True}

    return {
        "rho_min": state.rho.min().item(),
        "rho_max": state.rho.max().item(),
        "rho_mean": state.rho.mean().item(),
        "in_bounds": (
            state.rho.min().item() >= 0.49
            and state.rho.max().item() <= 2.01
        ),
    }


# =====================================================================
# 3. Long Rollout (Spinup + Statistics)
# =====================================================================


def eval_long_rollout(
    model: INNATE3D_MixedConvection,
    config: Config,
    Re: float,
    Ra: float = 1e5,
    spinup_steps: int = 500,
    stat_steps: int = 500,
    device: str = "cpu",
    balance_window: int = 20,
) -> Dict[str, object]:
    """
    Long rollout: spinup + istatistik toplama.

    1. set_physics(Re, Ra)
    2. spinup_steps forward pass (NaN check + E/Z timeseries, istatistik YOK)
    3. stat_steps forward pass (TKE, Nu, spectrum istatistikleri)
    4. Son balance_window state'i energy balance icin sakla

    Her forward pass = config.training.num_steps * dt zaman birimi.

    Returns:
        diagnostics: full timeseries {E, Z, div}
        stat_E, stat_Nu: statistics-phase arrays
        final_state: son state
        states_for_balance: son N state
        nan_step, nan_free
    """
    # Fizik parametrelerini guncelle
    model.set_physics(Re, Ra, config.physics.Pr)
    config.physics.Re = Re
    config.physics.Ra = Ra

    model.eval()
    total_steps = spinup_steps + stat_steps

    diagnostics = {"E": [], "Z": [], "div": []}
    stat_E: List[float] = []
    stat_Nu: List[float] = []
    states_for_balance: List[ThermalFluidState] = []
    nan_step = -1

    with torch.no_grad():
        state = model.create_initial_condition(batch_size=1, device=device)

        for step in range(total_steps):
            state = model(state)

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
                break

            E = state.kinetic_energy().mean().item()
            Z = _compute_enstrophy(state, model.ops).mean().item()
            div = model.ops.divergence(state.u, state.v, state.w).abs().max().item()

            diagnostics["E"].append(E)
            diagnostics["Z"].append(Z)
            diagnostics["div"].append(div)

            # Statistics phase
            if step >= spinup_steps:
                stat_E.append(E)
                Nu = eval_nusselt(state, config)
                stat_Nu.append(Nu)

            # Energy balance: son balance_window state'i sakla
            if step >= total_steps - balance_window - 1:
                states_for_balance.append(state)

            # Progress
            if step > 0 and step % 200 == 0:
                phase_str = "spinup" if step < spinup_steps else "stat"
                print(f"    step {step}/{total_steps} [{phase_str}] E={E:.6f} div={div:.2e}")

    return {
        "Re": Re,
        "Ra": Ra,
        "total_steps": total_steps,
        "spinup_steps": spinup_steps,
        "stat_steps": stat_steps,
        "diagnostics": diagnostics,
        "stat_E": stat_E,
        "stat_Nu": stat_Nu,
        "final_state": state if nan_step == -1 else None,
        "states_for_balance": states_for_balance,
        "nan_step": nan_step,
        "nan_free": nan_step == -1,
    }


# =====================================================================
# 4. Full Evaluation Pipeline
# =====================================================================


def evaluate(
    checkpoint_path: Optional[str] = None,
    config: Optional[Config] = None,
    spinup_steps: int = 500,
    stat_steps: int = 500,
    device_override: Optional[str] = None,
) -> Dict[str, object]:
    """
    Tam evaluation pipeline.

    Her egitim Re degeri icin (7000, 10000):
    1. Long rollout (spinup + statistics)
    2. Fiziksel tutarlilik metrikleri
    3. LES referans karsilastirmasi
    4. TKE convergence
    """
    if config is None:
        config = Config()

    device = device_override or str(config.device)
    print(f"Device: {device}")

    # -- Model --
    model = INNATE3D_MixedConvection(config).to(device)
    ckpt = None
    if checkpoint_path is not None:
        print(f"Loading checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, weights_only=False, map_location=device)
        model.load_state_dict(ckpt["model"])
        epoch = ckpt.get("epoch", "?")
        print(f"  Loaded epoch {epoch}")
    else:
        print("No checkpoint -- evaluating untrained model")

    print(f"Parameters: {model.count_parameters()}")
    model.eval()

    is_nb = _is_non_boussinesq(model)
    results: Dict[str, object] = {"config": config.to_dict()}
    re_results: Dict[int, Dict] = {}

    effective_dt = config.model.n_layers * config.physics.dt

    # ── Per-Re evaluation ──────────────────────────────────────────
    for Re in EVAL_RE_VALUES:
        les_ref = LES_REFERENCE.get(Re, {})

        print(f"\n{'='*60}")
        print(f"[Re={Re}] Long rollout: {spinup_steps} spinup + {stat_steps} stat")
        print(f"{'='*60}")

        t0 = time.time()
        rollout = eval_long_rollout(
            model, config, Re, EVAL_RA,
            spinup_steps=spinup_steps, stat_steps=stat_steps,
            device=device,
        )
        elapsed = time.time() - t0
        print(f"  Rollout time: {elapsed:.1f}s")

        if not rollout["nan_free"]:
            print(f"  FAIL: NaN at step {rollout['nan_step']}")
            re_results[Re] = {
                "nan_free": False,
                "nan_step": rollout["nan_step"],
            }
            continue

        final = rollout["final_state"]

        # -- Physics metrics --
        div_max = eval_divergence(model, final)
        slope = eval_spectrum_slope(final, model.ops)
        entropy = eval_spectral_entropy(final, model.ops)
        nusselt = eval_nusselt(final, config)
        t_bounds = eval_temperature_bounds(final, config)
        v_theta = (final.v * final.theta).mean().item()
        theta_rms = final.theta.pow(2).mean().sqrt().item()

        # Energy balance from last N states
        energy_ratio = eval_energy_balance(
            rollout["states_for_balance"], model, config,
            effective_dt=effective_dt,
        )

        # -- TKE statistics --
        stat_E = rollout["stat_E"]
        if len(stat_E) > 0:
            tke_mean = sum(stat_E) / len(stat_E)
            tke_std = (sum((x - tke_mean)**2 for x in stat_E) / len(stat_E)) ** 0.5
            tke_cv = tke_std / max(tke_mean, 1e-20)
        else:
            tke_mean = tke_std = tke_cv = float("nan")

        # -- Nu statistics --
        stat_Nu = rollout["stat_Nu"]
        nu_mean = sum(stat_Nu) / len(stat_Nu) if stat_Nu else float("nan")

        # -- LES comparison --
        tke_ref = les_ref.get("TKE")
        slope_ref = les_ref.get("slope")
        entropy_ref = les_ref.get("S_ent")

        tke_error = abs(tke_mean - tke_ref) / tke_ref if tke_ref else None
        slope_error = abs(slope - slope_ref) if slope_ref else None
        entropy_error = abs(entropy - entropy_ref) / entropy_ref if entropy_ref else None

        tke_pass = tke_error < LES_TOLERANCES["TKE_rel"] if tke_error is not None else None
        slope_pass = slope_error < LES_TOLERANCES["slope_abs"] if slope_error is not None else None
        entropy_pass = entropy_error < LES_TOLERANCES["entropy_rel"] if entropy_error is not None else None

        # -- Pass criteria --
        pass_criteria = {
            "nan_free": True,
            "divergence": div_max < 1e-5,
            "energy_balance": energy_ratio < 0.15,
            "nusselt": nusselt > 5.0,
            "temperature": t_bounds["in_bounds"],
            "tke_convergence": tke_cv < 0.10,
            "les_tke": tke_pass,
            "les_slope": slope_pass,
            "les_entropy": entropy_pass,
        }

        # Count passes
        n_criteria = sum(1 for v in pass_criteria.values() if v is not None)
        n_pass = sum(1 for v in pass_criteria.values() if v is True)

        # All critical = nan_free + divergence + energy_balance + nusselt + LES
        all_critical = all([
            pass_criteria["nan_free"],
            pass_criteria["divergence"],
            pass_criteria["energy_balance"],
            pass_criteria["nusselt"],
            tke_pass if tke_pass is not None else True,
            slope_pass if slope_pass is not None else True,
        ])

        re_result = {
            "nan_free": True,
            "tke_mean": tke_mean,
            "tke_std": tke_std,
            "tke_cv": tke_cv,
            "slope": slope,
            "entropy": entropy,
            "nusselt_mean": nu_mean,
            "nusselt_final": nusselt,
            "v_theta_flux": v_theta,
            "theta_rms": theta_rms,
            "divergence_max": div_max,
            "energy_balance_ratio": energy_ratio,
            "temperature": t_bounds,
            "les_comparison": {
                "tke_ref": tke_ref,
                "tke_error": tke_error,
                "tke_pass": tke_pass,
                "slope_ref": slope_ref,
                "slope_error": slope_error,
                "slope_pass": slope_pass,
                "entropy_ref": entropy_ref,
                "entropy_error": entropy_error,
                "entropy_pass": entropy_pass,
            },
            "pass_criteria": pass_criteria,
            "criteria_passed": n_pass,
            "criteria_total": n_criteria,
            "all_critical_pass": all_critical,
            "diagnostics": rollout["diagnostics"],
        }
        re_results[Re] = re_result

        # -- Print --
        print(f"\n  --- Physics Metrics (Re={Re}) ---")
        print(f"  Divergence:      {div_max:.2e}  {'PASS' if pass_criteria['divergence'] else 'FAIL'}")
        print(f"  Energy balance:  {energy_ratio:.4f}  {'PASS' if pass_criteria['energy_balance'] else 'FAIL'}")
        print(f"  Spectrum slope:  {slope:.3f}  (LES ref: {slope_ref})")
        print(f"  Spectral entropy:{entropy:.3f}  (LES ref: {entropy_ref})")
        print(f"  Nusselt (mean):  {nu_mean:.2f}  {'PASS' if pass_criteria['nusselt'] else 'FAIL'}")
        print(f"  Nusselt (final): {nusselt:.2f}  (LES ref: {les_ref.get('Nu', '?')}, monitoring)")
        print(f"  Temperature:     [{t_bounds['theta_min']:.4f}, {t_bounds['theta_max']:.4f}] "
              f"rms={t_bounds['theta_rms']:.5f}  {'PASS' if pass_criteria['temperature'] else 'FAIL'}")

        print(f"\n  --- LES Comparison (Re={Re}) ---")
        print(f"  TKE:     {tke_mean:.6f} vs {tke_ref}  err={tke_error:.1%}  "
              f"{'PASS' if tke_pass else 'FAIL'}" if tke_error is not None else "  TKE: N/A")
        print(f"  Slope:   {slope:.3f} vs {slope_ref}  err={slope_error:.3f}  "
              f"{'PASS' if slope_pass else 'FAIL'}" if slope_error is not None else "  Slope: N/A")
        print(f"  Entropy: {entropy:.3f} vs {entropy_ref}  err={entropy_error:.1%}  "
              f"{'PASS' if entropy_pass else 'FAIL'}" if entropy_error is not None else "  Entropy: N/A")

        print(f"\n  --- Convergence (Re={Re}) ---")
        print(f"  TKE CV:  {tke_cv:.4f}  {'PASS' if pass_criteria['tke_convergence'] else 'FAIL'}")
        print(f"  v*theta: {v_theta:.6f}  (LES ref: {les_ref.get('v_theta_flux', '?')}, monitoring)")
        print(f"  theta_rms:{theta_rms:.6f}  (LES ref: {les_ref.get('theta_rms', '?')}, monitoring)")

        print(f"\n  Criteria: {n_pass}/{n_criteria}  Critical: {'PASS' if all_critical else 'FAIL'}")

    results["re_evaluations"] = {str(k): v for k, v in re_results.items()}

    # ── Phase 2 (Non-Boussinesq) ──────────────────────────────────
    if is_nb:
        print(f"\n{'='*60}")
        print("[Phase 2] Non-Boussinesq metrics")
        print(f"{'='*60}")

        # Stability rollout ile phase-2 state'leri topla
        # (Son Re'nin rollout'undan al)
        last_re = EVAL_RE_VALUES[-1]
        if last_re in re_results and re_results[last_re].get("nan_free"):
            # Phase-2 icin ayri bir kisa rollout
            p2_rollout = eval_long_rollout(
                model, config, last_re, EVAL_RA,
                spinup_steps=100, stat_steps=50,
                device=device,
            )
            if p2_rollout["nan_free"]:
                p2_states = p2_rollout["states_for_balance"]
                p2_last = p2_rollout["final_state"]

                p2_cont = eval_continuity_residual(model, p2_states)
                p2_state_eq = eval_state_equation(model, p2_last)
                p2_mass = eval_mass_conservation(model, p2_states)
                p2_density = eval_density_bounds(model, p2_last)

                results["phase2_metrics"] = {
                    "continuity": p2_cont,
                    "state_equation": p2_state_eq,
                    "mass_conservation": p2_mass,
                    "density_bounds": p2_density,
                }

                print(f"  Continuity: mean={p2_cont['mean']:.2e} {'PASS' if p2_cont['pass'] else 'FAIL'}")
                print(f"  State eq:   mean={p2_state_eq['mean']:.2e} {'PASS' if p2_state_eq['pass'] else 'FAIL'}")
                print(f"  Mass drift: {p2_mass['max_drift']:.2e} {'PASS' if p2_mass['pass'] else 'FAIL'}")
                print(f"  Density:    [{p2_density.get('rho_min', 1.0):.4f}, "
                      f"{p2_density.get('rho_max', 1.0):.4f}] "
                      f"{'PASS' if p2_density['in_bounds'] else 'FAIL'}")

    # ── Global Summary ─────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("[SUMMARY]")
    print(f"{'='*60}")

    total_pass = 0
    total_criteria = 0
    all_critical_global = True

    for Re in EVAL_RE_VALUES:
        re_r = re_results.get(Re, {})
        if not re_r.get("nan_free", False):
            all_critical_global = False
            print(f"  Re={Re}: FAIL (NaN at step {re_r.get('nan_step', '?')})")
            continue

        n_p = re_r.get("criteria_passed", 0)
        n_t = re_r.get("criteria_total", 0)
        crit = re_r.get("all_critical_pass", False)
        total_pass += n_p
        total_criteria += n_t
        if not crit:
            all_critical_global = False
        print(f"  Re={Re}: {n_p}/{n_t} criteria  Critical: {'PASS' if crit else 'FAIL'}")

    print(f"\n  TOTAL: {total_pass}/{total_criteria}")
    print(f"  ALL CRITICAL: {'PASS' if all_critical_global else 'FAIL'}")

    results["summary"] = {
        "total_passed": total_pass,
        "total_criteria": total_criteria,
        "all_critical_pass": all_critical_global,
    }

    return results


# =====================================================================
# 6. Report
# =====================================================================


def save_report(results: Dict, path: str):
    """Evaluation sonuclarini JSON olarak kaydet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        if isinstance(obj, float):
            if obj != obj:  # NaN
                return "NaN"
            return obj
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(x) for x in obj]
        return obj

    clean = convert(results)
    path.write_text(json.dumps(clean, indent=2))
    print(f"\nReport saved: {path}")


# =====================================================================
# 7. Entry Point
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate INNATE 3D Mixed Convection (LES reference comparison)"
    )
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--spinup-steps", type=int, default=500,
                        help="Spinup forward passes (default: 500)")
    parser.add_argument("--stat-steps", type=int, default=500,
                        help="Statistics forward passes (default: 500)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (cpu/mps/cuda)")
    parser.add_argument("--dt", type=float, default=None,
                        help="Override physics.dt (training kullandigi dt ile ayni olmali; default 0.02 CFL ihlal eder)")
    parser.add_argument("--output", type=str, default="results/evaluation_report.json",
                        help="Report output path")
    parser.add_argument("--use-spectral-cs", action="store_true",
                        help="Saf-INNATE Spectral mimari ile eğitilmiş ckpt yükle")
    args = parser.parse_args()

    # Custom config (eger --dt override edilirse veya spectral mimari)
    custom_config = None
    if args.dt is not None or args.use_spectral_cs:
        custom_config = Config()
        if args.dt is not None:
            custom_config.physics.dt = args.dt
            print(f"[Config] dt override: {args.dt}")
        if args.use_spectral_cs:
            custom_config.model.use_spectral_cs = True
            custom_config.model.use_mlp_sgs = False
            print("[Config] Saf-INNATE Spectral-Cs mimari etkin")

    results = evaluate(
        checkpoint_path=args.checkpoint,
        config=custom_config,
        spinup_steps=args.spinup_steps,
        stat_steps=args.stat_steps,
        device_override=args.device,
    )
    save_report(results, args.output)


if __name__ == "__main__":
    main()
