"""
train.py - INNATE v2 3D Mixed Convection Training Pipeline

Physics-only training (NO DNS data in training loop).
Saf INNATE mimari: 20 katman, MLP YOK. IMEX time stepping.

Model internally 20-layer fractional-step yapiyor (return_intermediates=True).
Disarida unrolling YOK -- tek forward pass = 20 zaman adimi.

Loss terimleri:
  Guard rails (fizik constraintleri):
  1. L_divergence:      ||div(u)||^2 + 0.1*||div(u)||_1
  2. L_energy_balance:  |dE/dt + nu*Z - P_forcing - P_buoyancy|
  3. L_spectrum:        (slope - (-5/3))^2 inertial range guard rail
  4. L_dissipation:     |eps_spectral - nu*Z| enstrophy-dissipation tutarliligi
  5. L_thermal_var:     relu(var(T') - var_max)^2 (sicaklik blowup onleme)
  6. L_stability:       relu(Z_ratio - 10000)^2 (enstrophy blowup onleme)
  7. L_spectral_entropy: relu(target - entropy)^2 (laminer collapse onleme)
  8. L_theta_min:        relu(0.005 - theta_rms)^2 (isothermal collapse onleme)

  LES referans loss'lar (ground truth hedefleri):
  8. L_tke_ref:         (log(TKE) - log(TKE_ref))^2 log-space MSE
  9. L_nu_ref:          DEVRE DISI (damping paradoksu: %43 hata, 2026-02-28)
  10. L_slope_ref:      (slope - slope_ref)^2 Re-spesifik hedef
  11. L_entropy_ref:    (S - S_ref)^2 kesin hedef
  12. L_spectrum_shape: MSE(log(E(k)), log(E_ref(k))) tam sekil eslestirmesi

  INNATE v2 yeni loss'lar:
  13. L_anti_laminar:   relu(0.3*TKE_ref - TKE)^2 + relu(-2.5-slope)^2 (A fazindan)
  14. L_nu_phys:        Nusselt hedef loss (C fazindan, buoyancy damping kaldirildi)
  15. L_germano:        Germano identity ||L_ij - C*M_ij||^2 (D fazindan)

Curriculum (INNATE v2):
  Phase A (0-300):    Re=5K,7K — Guard rails + basic refs, warmup
  Phase B (300-600):  Re=5K,7K,10K — Ramp weights A→C
  Phase C (600-1000): Re=5K,7K,10K,15K — Full weights + Nu loss
  Phase D (1000-1500): Re=5K,7K,10K,15K,20K — + Germano loss, fine-tune

Re secimi: stratified round-robin (epoch % len(Re_list)), deterministic.

Yazar: Berke Tezgocen (tasarim), Claude (implementasyon)
"""

from __future__ import annotations

import random
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# MPS backend'inde rfftn/irfftn internal tensor resize warning'i (PyTorch 2.10 bug)
warnings.filterwarnings("ignore", message=".*resized since it had shape.*")

import numpy as np
import torch
import torch.nn as nn

# -- path setup --
_this_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(_this_dir.parent))
sys.path.insert(0, str(_this_dir))

from config import Config
from model import INNATE3D_MixedConvection, ThermalFluidState

import json
import math
from pathlib import Path as _Path

# -- Loss scale normalizasyonu (kritik!) ---------------------------------
# 2026-04-25 boyut analizi raporu: PhysicsLoss terimleri farklı boyut
# sınıflarında (1e-6 ile 1e3 arası) residual üretip ham toplanıyordu.
# Bu, optimizer'ın hangi terimi öncelikleyemediği için 63 saatlik
# eğitimin Nu=1.0'a sıkışmasının kök sebebiydi. Şimdi her loss return
# etmeden önce karakteristik scale'e bölünür → boyutsuz O(1) residual.
from loss_scales import (
    DIV_SCALE,
    ENERGY_BALANCE_SCALE,
    DISSIPATION_SCALE,
    ENSTROPHY_SCALE,
    SLOPE_SCALE,
    NU_LOG_SCALE,
    FLUX_LOG_SCALE,
    TKE_LOG_SCALE,
    THETA_RMS_LOG_SCALE,
    CORR_SCALE,
    ENTROPY_SCALE,
    SPECTRUM_SHAPE_SCALE,
    CFL_SCALE,
    THERMAL_VAR_SCALE,
    THETA_MIN_SCALE,
    CONTINUITY_RHO_SCALE,
    STATE_EQ_SCALE,
    MASS_SCALE,
    GERMANO_SCALE,
    NS_RES_SCALE,
    TH_RES_SCALE,
    ANTI_LAMINAR_SCALE,
    scale_loss,
)


# =====================================================================
# 0. LES Reference Metrics (Ground Truth)
# =====================================================================

# 60K step LES validation solver runs (Ra=1e5, Pr=0.71)
# Damping: anisotropic sigma-profile, sadece v+theta, safety=2.0
# Kaynak: bitirme2/les_reference/Re{N}_Ra1e5/spectrum.pt
# k_range: (6,15) — LES solver ile AYNI. Grid: 96x160x64, 32 bin spectrum.
# NOT: Re=7000 ve Re=10000 degerleri eski 96x160x64 grid'den. Yeni grid ile guncellenecek.
#      Re=5000, 15000, 20000 henuz LES referansi uretilmedi — placeholder.
LES_REFERENCE = {
    # TODO: Re=5000 referans verisi henuz uretilmedi (96x160x64 grid ile)
    # Placeholder: power-law interpolasyon (TKE ~ Re^-0.415, Nu ~ Re^0.8, vb.)
    # Uyari: Re=5000 periodic BC'de buoyancy-unstable (Ri=0.0056) -- dikkatli kullan
    5000: {
        "TKE": 0.0095,       # TODO: placeholder, LES ile dogrula
        "Nu": 24.0,           # TODO: placeholder
        "slope": -1.68,       # TODO: placeholder, k=(6,15)
        "S_ent": 2.05,        # TODO: placeholder
        "epsilon": 2.7e-04,   # TODO: placeholder
        "v_theta_flux": 6.0e-04,  # TODO: placeholder
        "theta_rms": 0.022,   # TODO: placeholder
    },
    7000: {
        "TKE": 0.008479,
        "Nu": 39.3194,
        "slope": -1.666,  # k=(6,15), LES solver ile ayni k_range
        "S_ent": 2.175,
        "epsilon": 2.4707e-04,
        "v_theta_flux": 7.7101e-04,
        "theta_rms": 0.02404,
    },
    10000: {
        "TKE": 0.007311,
        "Nu": 70.4587,
        "slope": -1.647,  # k=(6,15), LES solver ile ayni k_range
        "S_ent": 2.411,
        "epsilon": 2.2415e-04,
        "v_theta_flux": 9.7829e-04,
        "theta_rms": 0.02681,
    },
    # TODO: Re=15000 referans verisi henuz uretilmedi (96x160x64 grid ile)
    # Placeholder: power-law ekstrapolasyon
    15000: {
        "TKE": 0.0060,       # TODO: placeholder
        "Nu": 115.0,          # TODO: placeholder
        "slope": -1.63,       # TODO: placeholder, k=(6,15)
        "S_ent": 2.55,        # TODO: placeholder
        "epsilon": 2.0e-04,   # TODO: placeholder
        "v_theta_flux": 1.2e-03,  # TODO: placeholder
        "theta_rms": 0.030,   # TODO: placeholder
    },
    # TODO: Re=20000 referans verisi henuz uretilmedi (96x160x64 grid ile)
    # Placeholder: power-law ekstrapolasyon, grid resolution sinirinda
    20000: {
        "TKE": 0.0052,       # TODO: placeholder
        "Nu": 160.0,          # TODO: placeholder
        "slope": -1.60,       # TODO: placeholder, k=(6,15)
        "S_ent": 2.65,        # TODO: placeholder
        "epsilon": 1.8e-04,   # TODO: placeholder
        "v_theta_flux": 1.5e-03,  # TODO: placeholder
        "theta_rms": 0.033,   # TODO: placeholder
    },
}

# LES referans spektrumlari (lazy load, caching)
_LES_SPECTRA_CACHE: Dict[int, torch.Tensor] = {}


def get_les_spectrum(Re: int, device: torch.device) -> Optional[torch.Tensor]:
    """LES referans E(k) spektrumunu yukle (32 bin, 96x160x64 grid). Cache'lenir."""
    if Re not in _LES_SPECTRA_CACHE:
        spec_path = _Path(__file__).parent / f"les_reference/Re{Re}_Ra1e5/spectrum.pt"
        if spec_path.exists():
            data = torch.load(spec_path, weights_only=True)
            _LES_SPECTRA_CACHE[Re] = data["E_k"]
        else:
            return None
    spec = _LES_SPECTRA_CACHE[Re]
    return spec.to(device) if spec.device != device else spec


# =====================================================================
# 1. Energy Spectrum Utilities
# =====================================================================


def compute_energy_spectrum(
    u: torch.Tensor,
    v: torch.Tensor,
    w: torch.Tensor,
    ops,
) -> torch.Tensor:
    """
    Shell-averaged 1D energy spectrum E(k).

    Integer mode-index shells: ki <= |k_mode| < ki+1
    LES solver ile ayni binning (k_max_shell = min(Nx,Ny,Nz)//2).
    96x160x64 grid -> 32 bin (LES solver ile ayni).
    Returns: spectrum [n_bins] (batch ortalamalanmis)
    """
    from innate import safe_fftn

    u_hat = safe_fftn(u)
    v_hat = safe_fftn(v)
    w_hat = safe_fftn(w)

    # Spektral enerji yogunlugu
    Nx, Ny, Nz = u.shape[-3], u.shape[-2], u.shape[-1]
    N_total = Nx * Ny * Nz
    E_hat = 0.5 * (u_hat.abs() ** 2 + v_hat.abs() ** 2 + w_hat.abs() ** 2)
    E_hat = E_hat / N_total**2  # normalize

    # Batch boyutunu duzlestir (ortalama al)
    if E_hat.dim() == 4:
        E_hat = E_hat.mean(dim=0)  # [Nx, Ny, Nz]

    # Integer mode indices (LES solver ile ayni)
    mx = torch.fft.fftfreq(Nx, d=1.0, device=u.device) * Nx
    my = torch.fft.fftfreq(Ny, d=1.0, device=u.device) * Ny
    mz = torch.fft.fftfreq(Nz, d=1.0, device=u.device) * Nz
    Mx, My, Mz = torch.meshgrid(mx, my, mz, indexing='ij')
    k_index = torch.sqrt(Mx**2 + My**2 + Mz**2)

    # Shell: ki <= |k| < ki+1, LES ile ayni n_bins
    k_max_shell = min(Nx, Ny, Nz) // 2  # = 32
    if k_max_shell < 2:
        return torch.ones(2, device=u.device) * 1e-10

    # Scatter-add ile bin toplami (differentiable)
    k_idx = k_index.long().clamp(0, k_max_shell - 1)
    spectrum = torch.zeros(k_max_shell, device=u.device)
    spectrum.scatter_add_(0, k_idx.flatten(), E_hat.flatten())

    return spectrum


def spectral_entropy_loss(
    u: torch.Tensor, v: torch.Tensor, w: torch.Tensor, ops
) -> torch.Tensor:
    """
    Spectral entropy: enerji dagiliminin genisligini olcer.
    Laminer -> H~0 (tek mod), turbulant -> H yuksek (genis bant).
    Laminer'de bile requires_grad=True -- her zaman gradient uretir.
    """
    spectrum = compute_energy_spectrum(u, v, w, ops)
    # Normalize -> olasilik dagilimi
    p_k = spectrum / (spectrum.sum() + 1e-10)
    # Shannon entropy
    entropy = -(p_k * torch.log(p_k + 1e-20)).sum()
    max_entropy = torch.log(torch.tensor(float(len(spectrum)), device=u.device))
    # Hedef: entropi en az max'in %30'u olsun (turbulant dagilim)
    target = 0.3 * max_entropy
    return torch.relu(target - entropy).pow(2)


def spectrum_slope_loss(
    spectrum: torch.Tensor, k_range: Tuple[int, int] = (6, 15)
) -> torch.Tensor:
    """
    Inertial range'de log-log slope'u -5/3'e zorla.
    Linear regression ile slope fit edip hedeften sapmayi cezalandirir.
    k_range: (6, 15) — 96x160x64 grid, LES solver ile ayni.
    """
    k = torch.arange(len(spectrum), device=spectrum.device, dtype=spectrum.dtype)

    # Inertial range secimi
    mask = (k >= k_range[0]) & (k <= k_range[1]) & (spectrum > 1e-20)
    if mask.sum() < 3:
        return torch.tensor(0.0, device=spectrum.device)

    log_k = torch.log(k[mask])
    log_E = torch.log(spectrum[mask])

    # log-log uzayinda linear regression
    n = log_k.shape[0]
    sum_xy = (log_k * log_E).sum()
    sum_x = log_k.sum()
    sum_y = log_E.sum()
    sum_x2 = (log_k**2).sum()

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x**2 + 1e-10)

    target_slope = -5.0 / 3.0
    return (slope - target_slope).pow(2)


# =====================================================================
# 2. PhysicsLoss
# =====================================================================


class PhysicsLoss:
    """
    Tum fizik-bazli loss fonksiyonlarini hesaplar.
    DNS data kullanmaz -- tamamen physics-informed.
    """

    # Energy balance'da kullanilan ardisik cift sayisi.
    # 20 yerine 3: son 3 cift yeterli (erken state'ler IC'ye yakin).
    # compute_all ve energy_balance_loss'ta ortak kullanilir.
    N_BALANCE_PAIRS = 3

    def __init__(self, config: Config, model: INNATE3D_MixedConvection):
        self.config = config
        self.model = model
        self.ops = model.ops
        self._Z_ref: Optional[torch.Tensor] = None  # stability loss referansi

    def _enstrophy(self, state: ThermalFluidState) -> torch.Tensor:
        """Dogru enstrophy: <omega^2> = curl(u) uzerinden.  Returns [B]."""
        ox, oy, oz = self.ops.curl(state.u, state.v, state.w)
        return (ox**2 + oy**2 + oz**2).mean(dim=(-3, -2, -1))

    # ---------------------------------------------------------------- #
    # Yardimci: non_boussinesq mod kontrolu                             #
    # ---------------------------------------------------------------- #

    def _is_non_boussinesq(self) -> bool:
        """Model'in non_boussinesq modda olup olmadigini kontrol et."""
        return getattr(self.model, "non_boussinesq", False)

    # ---------------------------------------------------------------- #
    # Bireysel loss terimleri (Phase A-C: Boussinesq)                   #
    # ---------------------------------------------------------------- #

    def divergence_loss(self, state: ThermalFluidState) -> torch.Tensor:
        """
        L_div = mean(div^2) / DIV_SCALE  (boyutsuz, ~O(1))

        Incompressibility zorlamasi. En temel fizik constrainti.

        NOT (2026-04-25 fix): Eski form `mean(div^2) + 0.1*mean(|div|)`
        idi → L1+L2 toplamı boyutsal olarak tutarsız (ayrı boyut sınıfları).
        Saf L2'ye çekildi ve loss_scales.DIV_SCALE ile boyutsuzlaştırıldı.
        """
        div = self.ops.divergence(state.u, state.v, state.w)
        raw = div.pow(2).mean()
        return scale_loss(raw, DIV_SCALE)

    def energy_balance_loss(
        self, states: List[ThermalFluidState],
        enstrophy_cache: Optional[Dict[int, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        L_energy = |dE/dt + eps_mol - P_forcing - P_buoyancy|

        Enerji denkleminin saglanamasini cezalandirir.
        Son 3 ardisik cift uzerinden hesaplanir (states[-4:]).
        Erken state'ler IC'ye yakin ve fractional-step gecis asamasi --
        enerji dengesi son ciktilarda saglansa yeterli.

        dE/dt: ardisik state'ler arasi kinetik enerji farki / dt
        eps_mol = nu * Z (molecular dissipation ONLY)
        P_forcing = <Fx*u + Fy*v + Fz*w> (forcing power input)
        P_buoyancy = Ri * <v*T'> (buoyancy power)

        NOT: SGS dissipasyon (eps_SGS = <2*nu_t*S_ij*S_ij>) dahil DEGIL.
        Eddy viscosity aktifken gercek denge: dE/dt + eps_mol + eps_SGS = P_f + P_b.
        Bu loss sadece molekuler dengeyi olcer. Residual ~eps_SGS mertebesinde
        kalacaktir, bu nedenle weight dusuk tutulmali.
        """
        if len(states) < 2:
            return torch.tensor(0.0, device=states[0].u.device)

        phys = self.config.physics
        # Modelin ogrendigi efektif dt'yi kullan (dt_base * dt_scale)
        dt = self.model._dt_base * torch.clamp(self.model.dt_scale, 0.5, 2.0)
        loss = torch.tensor(0.0, device=states[0].u.device)

        # Forcing terimleri (model'den al)
        Fx, Fy, Fz = self.model.forcing()

        # 2026-04-25 fix: buoyancy_strength loss'a dahil edilmeli.
        # Modelin gerçekten uyguladığı buoyancy: Ri * s * theta (s=strength).
        # Önceden P_b = Ri*<vθ> yazılıyordu → s≠1 öğrenildiğinde tutarsız.
        # Layer-bağımlı strength'lerin ortalamasını alıyoruz.
        if hasattr(self.model, "buoyancies"):
            with torch.no_grad():
                strengths = [
                    b.buoyancy_strength.clamp(0.0, 50.0)
                    for b in self.model.buoyancies
                ]
                s_mean = torch.stack(strengths).mean()
        elif hasattr(self.model, "buoyancy"):
            with torch.no_grad():
                s_mean = self.model.buoyancy.buoyancy_strength.clamp(0.0, 50.0)
        else:
            s_mean = torch.tensor(1.0, device=states[0].u.device)

        # Son N cift: states[-(N+1):] (N+1 state = N cift)
        # 20 cift yerine 3 cift = %85 FFT tasarrufu (240 -> 36 FFT/step)
        start_idx = max(1, len(states) - self.N_BALANCE_PAIRS)

        for i in range(start_idx, len(states)):
            s0, s1 = states[i - 1], states[i]

            E0 = s0.kinetic_energy()  # [B]
            E1 = s1.kinetic_energy()
            dEdt = (E1 - E0) / dt

            # Enstrophy: cache varsa kullan, yoksa hesapla
            if enstrophy_cache is not None and (i - 1) in enstrophy_cache:
                Z0 = enstrophy_cache[i - 1]
            else:
                Z0 = self._enstrophy(s0)
            if enstrophy_cache is not None and i in enstrophy_cache:
                Z1 = enstrophy_cache[i]
            else:
                Z1 = self._enstrophy(s1)

            Z = 0.5 * (Z0 + Z1)  # ortalama enstrophy
            eps = phys.nu * Z

            # Forcing power: <Fx*u + Fy*v + Fz*w>
            u_mid = 0.5 * (s0.u + s1.u)
            v_mid = 0.5 * (s0.v + s1.v)
            w_mid = 0.5 * (s0.w + s1.w)
            P_f = (Fx * u_mid + Fy * v_mid + Fz * w_mid).mean(dim=(-3, -2, -1))

            # Buoyancy power: Ri * s * <v*T'>  (strength dahil!)
            theta_mid = 0.5 * (s0.theta + s1.theta)
            P_b = phys.Ri * s_mean * (v_mid * theta_mid).mean(dim=(-3, -2, -1))

            # Energy balance: dE/dt = -eps + P_f + P_b
            # 2026-04-27 TUNING v3 (Codex F + CFD d sentezi):
            #   1) Sabit LES-olcek normalizasyon (eps0 = eps_LES_ref)
            #   2) Production floor (asimetrik): P_f >= 0.6*eps0, P_b >= 0.4*eps0
            #      Goodhart trivial null space (hepsi=0) bu floor ile cezalandirilir
            #   3) scale_loss UYGULANMAZ (formul zaten ~O(1) boyutsuz)
            #   v2'deki value-level normalize (denom~1e-4 patlamasi) kaldirildi.
            eps0 = 1.4e-3                           # eps_LES_ref Re=7K
            P_f_star = 0.6 * eps0
            P_b_star = 0.4 * eps0
            lam = 2.0
            residual = (dEdt + eps - P_f - P_b)
            l_residual = (residual / eps0).pow(2).mean()
            l_pf_floor = (torch.relu(P_f_star - P_f.abs()) / eps0).pow(2).mean()
            l_pb_floor = (torch.relu(P_b_star - P_b.abs()) / eps0).pow(2).mean()
            loss = loss + l_residual + lam * (l_pf_floor + l_pb_floor)

        n_pairs = len(states) - start_idx
        raw = loss / max(n_pairs, 1)
        # NO scale_loss: formul zaten eps0 ile boyutsuzlastirilmis
        return raw

    def spectrum_loss(self, state: ThermalFluidState, spectrum: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        L_spectrum = (slope_fit - (-5/3))^2 / SLOPE_SCALE  (boyutsuz)
        Inertial range'de Kolmogorov -5/3 yasasini zorlar.
        """
        if spectrum is None:
            spectrum = compute_energy_spectrum(state.u, state.v, state.w, self.ops)
        # Sabit k_range — tum dosyalarda (6, 15) standardize edildi (96x160x64 grid)
        raw = spectrum_slope_loss(spectrum, k_range=(6, 15))
        return scale_loss(raw, SLOPE_SCALE)

    def spectral_entropy_loss(self, state: ThermalFluidState, spectrum: Optional[torch.Tensor] = None) -> torch.Tensor:
        """L_spectral_entropy / ENTROPY_SCALE: laminer collapse'i cezalandirir."""
        if spectrum is not None:
            # spectrum zaten hesaplanmis, dogrudan entropy hesapla
            p_k = spectrum / (spectrum.sum() + 1e-10)
            entropy = -(p_k * torch.log(p_k + 1e-20)).sum()
            max_entropy = torch.log(torch.tensor(float(len(spectrum)), device=state.u.device))
            target = 0.3 * max_entropy
            raw = torch.relu(target - entropy).pow(2)
        else:
            raw = spectral_entropy_loss(state.u, state.v, state.w, self.ops)
        return scale_loss(raw, ENTROPY_SCALE)

    def dissipation_loss(self, state: ThermalFluidState,
                         cached_Z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        L_dissipation = |eps_spectral - nu*Z| / DISSIPATION_SCALE
        Spectral ve physical space dissipation tutarliligi (boyutsuz).
        """
        phys = self.config.physics
        Z = cached_Z if cached_Z is not None else self._enstrophy(state)  # [B]
        eps = phys.nu * Z

        # Spectral dissipation: 2*nu * sum(k^2 * E_hat(k))
        from innate import safe_fftn

        u_hat = safe_fftn(state.u)
        v_hat = safe_fftn(state.v)
        w_hat = safe_fftn(state.w)
        N_total = state.u.shape[-3] * state.u.shape[-2] * state.u.shape[-1]
        E_hat = 0.5 * (u_hat.abs() ** 2 + v_hat.abs() ** 2 + w_hat.abs() ** 2)
        E_hat = E_hat / N_total**2
        eps_spectral = 2.0 * phys.nu * (self.ops.k_squared_full * E_hat).sum(
            dim=(-3, -2, -1)
        )

        raw = (eps - eps_spectral).abs().mean()
        return scale_loss(raw, DISSIPATION_SCALE)

    def nusselt_reference_loss(self, state: ThermalFluidState, ref_nu: float) -> torch.Tensor:
        """
        L_nu = simetrik log-MSE + asimetrik anti-fizik ust-sinir cezasi

        2026-04-27 TUNING v3 (Berke direktifi: anti-fizige musaade etmeyiz):
        - Simetrik log-MSE: model Nu_ref'e yakinsasin
        - Asimetrik overshoot: Nu > Nu_max_phys icin patlayan ceza

        2026-04-27 TUNING v4 (Codex onerisi anti-fizik fren guclendir):
        - overshoot weight 5.0 -> 20.0 (v3 ep1'de Nu=244 patladi, fren yetmedi)
        - Nu_max_phys 60 -> 50 (Re=5-7K icin daha siki tavan)
        """
        _eps = 0.1
        Nu_max_phys = 50.0  # v4: 60 -> 50
        Nu = state.nusselt_number(
            self.config.domain.Ly, self.config.physics.kappa
        )  # [B]
        # Simetrik log-MSE
        L_log = (torch.log(Nu.clamp(min=0.1) + _eps) - math.log(ref_nu + _eps)).pow(2).mean()
        # Asimetrik anti-fizik fren: Nu > Nu_max_phys icin sert ceza
        L_overshoot = (torch.relu(Nu - Nu_max_phys) / Nu_max_phys).pow(2).mean()
        raw = L_log + 20.0 * L_overshoot   # v4: 5.0 -> 20.0
        return scale_loss(raw, NU_LOG_SCALE)

    def thermal_variance_loss(self, state: ThermalFluidState) -> torch.Tensor:
        """
        L_thermal_var = relu(var(T') - var_max)^2 / THERMAL_VAR_SCALE
        Sicaklik perturbasyonunun patlamasini onler (boyutsuz).
        """
        theta_var = state.theta.var(dim=(-3, -2, -1))  # [B]
        var_max = 100.0
        raw = torch.relu(theta_var - var_max).pow(2).mean()
        return scale_loss(raw, THERMAL_VAR_SCALE)

    def theta_min_loss(self, state: ThermalFluidState) -> torch.Tensor:
        """
        L_theta_min = relu(threshold - theta_rms)^2 / THETA_MIN_SCALE
        Isothermal collapse onleme: theta_rms alt sinir guard rail.
        LES'te theta_rms ~0.024-0.027 (Re'ye bagli). Esik 0.005 = ~5x marj.
        Normal egitimde aktive OLMAZ, sadece theta->0 durumunda frene basar.
        """
        theta_rms = state.theta.pow(2).mean(dim=(-3, -2, -1)).sqrt()  # [B]
        threshold = 0.005
        raw = torch.relu(threshold - theta_rms).pow(2).mean()
        return scale_loss(raw, THETA_MIN_SCALE)

    def cfl_guard_loss(self, state: ThermalFluidState) -> torch.Tensor:
        """
        L_cfl_guard = (0.1*relu(CFL-0.08)^2 + relu(CFL-0.3)^2) / CFL_SCALE

        Iki katmanli CFL ceza (boyutsuz):
          Soft zone (CFL 0.08-0.3): hafif erken uyari, dead zone yok
          Hard zone (CFL > 0.3): agresif fren
        """
        u_max = torch.stack([
            state.u.abs().amax(dim=(-3, -2, -1)),
            state.v.abs().amax(dim=(-3, -2, -1)),
            state.w.abs().amax(dim=(-3, -2, -1)),
        ]).amax(dim=0)  # [B]

        dt_eff = self.model._dt_base * torch.clamp(self.model.dt_scale, 0.5, 2.0)
        CFL = u_max * dt_eff / self.model.dx_min
        soft = torch.relu(CFL - 0.08).pow(2)  # CFL=0.15: 0.005, erken uyari
        hard = torch.relu(CFL - 0.3).pow(2)   # CFL=0.5: 0.04, agresif fren
        raw = (0.1 * soft + hard).mean()
        return scale_loss(raw, CFL_SCALE)

    def tke_overshoot_loss(self, state: ThermalFluidState, ref_tke: float) -> torch.Tensor:
        """
        L_tke_overshoot = softplus(TKE/TKE_ref - 1.0)^2 / TKE_LOG_SCALE

        DEVRE DISI (weight=0): L_tke_ref bidirectional yeterli.
        """
        TKE = state.kinetic_energy()  # [B]
        ratio = TKE / max(ref_tke, 1e-8)
        raw = torch.nn.functional.softplus(ratio - 1.0, beta=5.0).pow(2).mean()
        return scale_loss(raw, TKE_LOG_SCALE)

    def stability_loss(self, state: ThermalFluidState,
                       cached_Z: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        L_stability = relu(Z_ratio - 10000)^2 / ENSTROPHY_SCALE
        Sadece gercek patlama durumunda devreye girer.
        """
        Z = cached_Z if cached_Z is not None else self._enstrophy(state)  # [B]
        if self._Z_ref is None or self._Z_ref.item() < 1e-12:
            self._Z_ref = Z.detach().mean().clamp(min=1.0)
        Z_ratio = Z / self._Z_ref
        raw = torch.relu(Z_ratio - 10000.0).pow(2).mean()
        return scale_loss(raw, ENSTROPHY_SCALE)

    def v_theta_flux_loss(self, state: ThermalFluidState, ref_flux: float,
                          ref_theta_rms: float = 0.024) -> torch.Tensor:
        """
        L_v_theta = (log(<v*theta>/theta_rms) - log(ref/theta_rms_ref))^2

        2026-04-28 TUNING v5 (Codex'in ASIL onerisi - amplitude-pumping cozumu):
        v4'te L_v_theta tamamen kaldirildi -> Nu=211 patladi cunku model
        <v*theta>'yi sadece L_corr ile kontrol edemiyor. Asil cozum: payda'ya
        theta_rms koy. Bu sayede:
        - Model theta_rms'i buyuterek <v*theta>'yi sismeyemez (oran sabit kalir)
        - Sadece **gercek korelasyon** ve **dogru ratio** ile loss dusurulur
        - Coklu null space tek noktaya daralir (CFD-expert cozumunun saglikli versiyonu)

        Onceki versiyonlar:
        v3: log(<v*theta>_capped) - log(ref) -> amplitude pumping mumkun
        v4: KALDIR -> kontrolsuz <v*theta>
        v5: log(<v*theta>/theta_rms) - log(ref/theta_rms_ref) -> normalize

        Mantik: ratio = <v*theta>/theta_rms = v_rms * corr (dogasi geregi).
        Model bu ratio'yu hedefe getirir -> v_rms ve corr birlikte ayarlanir.
        """
        _eps = 1e-4
        v_theta = (state.v * state.theta).mean(dim=(-3, -2, -1))  # [B]
        theta_rms = state.theta.pow(2).mean(dim=(-3, -2, -1)).sqrt()  # [B]
        # Normalize: ratio = <v*theta> / theta_rms
        ratio = v_theta / (theta_rms + _eps)
        ref_ratio = ref_flux / (ref_theta_rms + _eps)
        raw = (torch.log(ratio.abs() + _eps) - math.log(abs(ref_ratio) + _eps)).pow(2).mean()
        return scale_loss(raw, FLUX_LOG_SCALE)

    def v_theta_correlation_loss(self, state: ThermalFluidState,
                                  ref_flux: float, ref_tke: float,
                                  ref_theta_rms: float) -> torch.Tensor:
        """
        L_corr = (corr - ref_corr)^2 / CORR_SCALE  (TKE-bağımsız korelasyon)

        Deadlock kırıcı: sadece v-θ korelasyonunu düzelt, TKE'ye dokunma.
        ref_corr = 7.71e-4 / (sqrt(0.0085) * 0.024) = 0.349 (Re=7000)
        """
        _eps = 1e-6
        v_theta = (state.v * state.theta).mean(dim=(-3, -2, -1))  # [B]
        TKE = state.kinetic_energy()  # [B]
        theta_rms = state.theta.pow(2).mean(dim=(-3, -2, -1)).sqrt()  # [B]
        corr = v_theta / (TKE.sqrt() * theta_rms + _eps)
        ref_corr = ref_flux / (math.sqrt(ref_tke) * ref_theta_rms + _eps)
        raw = (corr - ref_corr).pow(2).mean()
        return scale_loss(raw, CORR_SCALE)

    def tke_reference_loss(self, state: ThermalFluidState, ref_tke: float) -> torch.Tensor:
        """
        L_tke_ref = (log(TKE + eps) - log(TKE_ref + eps))^2 / TKE_LOG_SCALE
        Log-space MSE; bidirectional gradient. Zaten boyutsuz O(1).
        """
        _eps = 1e-3  # gradient cap: O(1/eps) = O(10^3)
        TKE = state.kinetic_energy()  # [B], 0.5 * <u^2+v^2+w^2>
        raw = (torch.log(TKE + _eps) - math.log(ref_tke + _eps)).pow(2).mean()
        return scale_loss(raw, TKE_LOG_SCALE)

    def theta_rms_reference_loss(self, state: ThermalFluidState, ref_theta_rms: float) -> torch.Tensor:
        """
        L_theta_rms_ref = (log(theta_rms + eps) - log(ref + eps))^2 / THETA_RMS_LOG_SCALE
        theta_rms referansa log-space MSE; bidirectional, boyutsuz.
        """
        _eps = 1e-4
        theta_rms = state.theta.pow(2).mean(dim=(-3, -2, -1)).sqrt()  # [B]
        raw = (torch.log(theta_rms + _eps) - math.log(ref_theta_rms + _eps)).pow(2).mean()
        return scale_loss(raw, THETA_RMS_LOG_SCALE)

    def theta_max_loss(self, state: ThermalFluidState, ref_theta_rms: float) -> torch.Tensor:
        """
        L_theta_max = relu(theta_rms - cap)^2 / ref^2  (quadratic, sert fren)

        2026-04-28 TUNING v5: log-form yumusakti (v4'te theta_rms=0.097'de bile
        loss=1.35, sinyal zayif). Quadratic ile sert fren: theta_rms cap'i
        astigi anda gradient lineer artiyor (ceza karesel). v5'te L_v_theta
        paydada theta_rms ile geri donduğu icin bu loss artik **safety net**
        rolunde - normalde tetiklenmemeli.

        Hesap: theta_rms=0.097 vs cap=0.036 -> raw=((0.097-0.036)/0.024)^2=6.45
        weight=10 -> 64.5 (v4 log-form: 13.5, v5 quadratic: 64.5, 5x daha sert)
        """
        alpha = 1.5  # cap = 1.5 x LES_ref
        theta_rms = state.theta.pow(2).mean(dim=(-3, -2, -1)).sqrt()  # [B]
        cap = alpha * ref_theta_rms
        excess = torch.relu(theta_rms - cap) / max(ref_theta_rms, 1e-6)
        raw = excess.pow(2).mean()  # quadratic: (excess)^2
        return raw  # boyutsuz

    # ---------------------------------------------------------------- #
    # LES Referans Loss Terimleri (Yeni)                                #
    # ---------------------------------------------------------------- #

    def slope_reference_loss(self, state: ThermalFluidState, ref_slope: float,
                             spectrum: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        L_slope_ref = (slope - slope_ref)^2

        Re-spesifik spectrum slope hedefi. Mevcut L_spectrum -5/3 hedefini
        tamamlar (o guard rail kalir, bu ince ayar yapar).
        """
        if spectrum is None:
            spectrum = compute_energy_spectrum(state.u, state.v, state.w, self.ops)
        k = torch.arange(len(spectrum), device=state.u.device, dtype=spectrum.dtype)
        mask = (k >= 6) & (k <= 15) & (spectrum > 1e-20)  # 96x160x64 grid: (6,15)
        if mask.sum() < 3:
            return spectrum.sum() * 0.0  # gradient akisi icin graph-connected sifir

        log_k = torch.log(k[mask])
        log_E = torch.log(spectrum[mask])
        n = log_k.shape[0]
        slope = (n * (log_k * log_E).sum() - log_k.sum() * log_E.sum()) / \
                (n * (log_k**2).sum() - log_k.sum()**2 + 1e-10)

        raw = (slope - ref_slope).pow(2)
        return scale_loss(raw, SLOPE_SCALE)

    def entropy_reference_loss(self, state: ThermalFluidState, ref_entropy: float,
                               spectrum: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        L_entropy_ref = (S - S_ref)^2 / ENTROPY_SCALE
        LES referans spectral entropy hedefi (boyutsuz).
        """
        if spectrum is None:
            spectrum = compute_energy_spectrum(state.u, state.v, state.w, self.ops)
        p_k = spectrum / (spectrum.sum() + 1e-10)
        entropy = -(p_k * torch.log(p_k + 1e-20)).sum()
        raw = (entropy - ref_entropy).pow(2)
        return scale_loss(raw, ENTROPY_SCALE)

    def spectrum_shape_loss(self, state: ThermalFluidState, ref_E_k: torch.Tensor,
                            spectrum: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        L_spectrum_shape = MSE(log(E_model(k)), log(E_ref(k)))

        Tam spektrum sekli eslestirmesi. Sadece slope degil, tum enerji
        dagilimini zorlar. Metric hacking'i engeller.
        Log-space: E(k) mertebeleri arasinda degisiyor (1e-20 ile 1e-3).
        Binning: compute_energy_spectrum ve LES solver ayni integer-shell (32 bin).
        """
        if spectrum is None:
            spectrum = compute_energy_spectrum(state.u, state.v, state.w, self.ops)
        n = min(len(spectrum), len(ref_E_k))
        model_spec = spectrum[:n]
        ref_spec = ref_E_k[:n]

        # NaN guard (2026-05-08 Codex): upstream u,v,w NaN'i FFT uzerinden spectrum'a
        # propage olur, clamp NaN'i duzeltmez. nan_to_num + isfinite mask ekle.
        model_spec = torch.nan_to_num(model_spec, nan=1e-20, posinf=1e-20, neginf=1e-20)

        # Sadece anlamli binleri kullan (ref_spec finite + model_spec finite)
        mask = (ref_spec > 1e-15) & torch.isfinite(model_spec)
        if mask.sum() < 3:
            return model_spec.sum() * 0.0  # gradient akisi icin graph-connected sifir

        # log(E + eps) form: clamp yerine offset ekleme, gradient daha smooth
        EPS = 1e-12
        model_log = torch.log(model_spec[mask].clamp(min=0.0) + EPS)
        ref_log = torch.log(ref_spec[mask].clamp(min=0.0) + EPS)

        raw = (model_log - ref_log).pow(2).mean()
        # raw NaN olursa loss zinciri kirilmasin
        if not torch.isfinite(raw):
            return model_spec.sum() * 0.0
        return scale_loss(raw, SPECTRUM_SHAPE_SCALE)

    # ---------------------------------------------------------------- #
    # INNATE v2: Germano Consistency Loss                               #
    # ---------------------------------------------------------------- #

    def germano_consistency_loss(self, state: ThermalFluidState) -> torch.Tensor:
        """
        Germano identity: L_ij - C_dynamic * M_ij = 0

        Germano (1991) identitesi SGS modelin self-consistency'sini test eder.
        0 ekstra parametre gerektirir — MLP'nin verdigi Cs ile tutarlilik olcer.

        L_ij = test_filter(u_i * u_j) - test_filter(u_i) * test_filter(u_j)
        M_ij = test_filter(Delta^2 * |S| * S_ij) - (2*Delta)^2 * |S_bar| * S_bar_ij

        C_dynamic = model'in MLP'sinden gelen Cs^2

        Loss: mean(||L_ij - C_dynamic * M_ij||^2)

        Test filter: sharp spectral cutoff at k_mag < k_max/2
        Pahali hesaplama (ekstra FFT'ler) — sadece loss_every step'lerinde cagir.
        """
        from innate import safe_rfftn, safe_irfftn

        ops = self.ops
        u, v, w = state.u, state.v, state.w
        device = u.device

        # --- Test filter mask: sharp cutoff at k_max / 2 ---
        # k_squared: [Nx, Ny, Nz//2+1] (rfftn domain)
        k_mag = torch.sqrt(ops.k_squared)
        k_max_val = k_mag.max().item()
        test_cutoff = k_max_val / 2.0
        test_mask = (k_mag < test_cutoff).float()  # [Nx, Ny, Nz//2+1]

        def test_filter(f: torch.Tensor) -> torch.Tensor:
            """Sharp spectral test filter: zero modes above k_max/2."""
            f_hat = safe_rfftn(f)
            return safe_irfftn(f_hat * test_mask, s=ops.spatial_shape)

        # --- L_ij = test_filter(u_i * u_j) - test_filter(u_i) * test_filter(u_j) ---
        # Sadece trace-free diagonal + off-diagonal (6 bilesik): ij = 11,22,33,12,13,23
        u_f, v_f, w_f = test_filter(u), test_filter(v), test_filter(w)

        # Leonard stress L_ij (resolved turbulent stress)
        L_11 = test_filter(u * u) - u_f * u_f
        L_22 = test_filter(v * v) - v_f * v_f
        L_33 = test_filter(w * w) - w_f * w_f
        L_12 = test_filter(u * v) - u_f * v_f
        L_13 = test_filter(u * w) - u_f * w_f
        L_23 = test_filter(v * w) - v_f * w_f

        # --- Strain rate S_ij at grid scale ---
        du_dx, du_dy, du_dz = ops.gradient(u)
        dv_dx, dv_dy, dv_dz = ops.gradient(v)
        dw_dx, dw_dy, dw_dz = ops.gradient(w)

        S_11 = du_dx
        S_22 = dv_dy
        S_33 = dw_dz
        S_12 = 0.5 * (du_dy + dv_dx)
        S_13 = 0.5 * (du_dz + dw_dx)
        S_23 = 0.5 * (dv_dz + dw_dy)

        S_mag = torch.sqrt(2.0 * (S_11**2 + S_22**2 + S_33**2 +
                                   2.0 * (S_12**2 + S_13**2 + S_23**2)) + 1e-12)

        # --- Strain rate S_bar_ij at test-filter scale ---
        du_f_dx, du_f_dy, du_f_dz = ops.gradient(u_f)
        dv_f_dx, dv_f_dy, dv_f_dz = ops.gradient(v_f)
        dw_f_dx, dw_f_dy, dw_f_dz = ops.gradient(w_f)

        Sb_11 = du_f_dx
        Sb_22 = dv_f_dy
        Sb_33 = dw_f_dz
        Sb_12 = 0.5 * (du_f_dy + dv_f_dx)
        Sb_13 = 0.5 * (du_f_dz + dw_f_dx)
        Sb_23 = 0.5 * (dv_f_dz + dw_f_dy)

        Sb_mag = torch.sqrt(2.0 * (Sb_11**2 + Sb_22**2 + Sb_33**2 +
                                    2.0 * (Sb_12**2 + Sb_13**2 + Sb_23**2)) + 1e-12)

        # --- M_ij = test_filter(Delta^2 * |S| * S_ij) - (2*Delta)^2 * |S_bar| * S_bar_ij ---
        # Delta^2: grid-scale filter width squared
        dx = self.config.domain.Lx / self.config.domain.Nx
        dy = self.config.domain.Ly / self.config.domain.Ny
        dz = self.config.domain.Lz / self.config.domain.Nz
        Delta_sq = (dx * dy * dz) ** (2.0 / 3.0)  # geometric mean filter width squared

        M_11 = test_filter(Delta_sq * S_mag * S_11) - 4.0 * Delta_sq * Sb_mag * Sb_11
        M_22 = test_filter(Delta_sq * S_mag * S_22) - 4.0 * Delta_sq * Sb_mag * Sb_22
        M_33 = test_filter(Delta_sq * S_mag * S_33) - 4.0 * Delta_sq * Sb_mag * Sb_33
        M_12 = test_filter(Delta_sq * S_mag * S_12) - 4.0 * Delta_sq * Sb_mag * Sb_12
        M_13 = test_filter(Delta_sq * S_mag * S_13) - 4.0 * Delta_sq * Sb_mag * Sb_13
        M_23 = test_filter(Delta_sq * S_mag * S_23) - 4.0 * Delta_sq * Sb_mag * Sb_23

        # --- C_dynamic: MLP'den gelen Cs^2 (son katmanin tahmini) ---
        # Model'in eddy viscosity'sinden son katmanin Cs degerini al
        C_dyn = torch.tensor(0.04, device=device)  # default: Cs=0.2 -> Cs^2=0.04
        if hasattr(self.model, 'eddy_viscosities') and len(self.model.eddy_viscosities) > 0:
            last_ev = self.model.eddy_viscosities[-1]
            if hasattr(last_ev, '_last_cs') and last_ev._last_cs is not None:
                # MLP'nin son forward'daki Cs tahmini (spatial average)
                C_dyn = last_ev._last_cs.pow(2).mean(dim=(-3, -2, -1)).mean()
            elif hasattr(last_ev, 'cs_mid'):
                C_dyn = last_ev.cs_mid.pow(2)
            elif hasattr(last_ev, 'smagorinsky_coeff'):
                C_dyn = last_ev.smagorinsky_coeff.pow(2)

        # --- Residual: ||L_ij - C_dyn * M_ij||^2 ---
        res_11 = (L_11 - C_dyn * M_11).pow(2)
        res_22 = (L_22 - C_dyn * M_22).pow(2)
        res_33 = (L_33 - C_dyn * M_33).pow(2)
        res_12 = (L_12 - C_dyn * M_12).pow(2)
        res_13 = (L_13 - C_dyn * M_13).pow(2)
        res_23 = (L_23 - C_dyn * M_23).pow(2)

        # Sadece off-diagonal agirlikli (trace-free diagonal noise'lu)
        residual = (res_11 + res_22 + res_33 +
                    2.0 * (res_12 + res_13 + res_23))

        raw = residual.mean()
        return scale_loss(raw, GERMANO_SCALE)

    # ---------------------------------------------------------------- #
    # INNATE v2: Anti-Laminarizasyon Loss                               #
    # ---------------------------------------------------------------- #

    def anti_laminarization_loss(self, state: ThermalFluidState,
                                  ref_tke: float,
                                  spectrum: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Anti-laminarizasyon loss: iki parcali koruma mekanizmasi.

        1. TKE minimum guard:
           relu(0.3 * TKE_ref - TKE)^2
           TKE referansin %30'unun altina duserse aktive olur.
           Normal egitimde aktive OLMAZ (dead zone), sadece laminarizasyon
           basladiginda gradient verir. %30 esigi LES referans varyansini
           (~%10-15) genis marjla karsilar.

        2. Spectrum slope guard:
           relu(-2.5 - slope)^2
           Slope -2.5'ten daha dik ise (ornegin -3.0) asiri dissipasyon
           var demektir = laminarizasyon baslamis. Kolmogorov -5/3 = -1.667,
           -2.5 limiti %50 sapma toleransi verir.

        Args:
            state: Son zaman adiminin state'i
            ref_tke: LES referans TKE degeri (power-law veya dict'ten)
            spectrum: Onceden hesaplanmis E(k) (cache, None ise yeniden hesaplanir)

        Returns:
            Skaler loss (iki parcali toplam)
        """
        device = state.u.device

        # Part 1: TKE minimum guard
        TKE = state.kinetic_energy()  # [B]
        tke_floor = 0.3 * ref_tke
        L_tke_min = torch.relu(tke_floor - TKE).pow(2).mean()

        # Part 2: Spectrum slope guard
        if spectrum is None:
            spectrum = compute_energy_spectrum(state.u, state.v, state.w, self.ops)

        k = torch.arange(len(spectrum), device=device, dtype=spectrum.dtype)
        mask = (k >= 6) & (k <= 15) & (spectrum > 1e-20)  # 96x160x64 grid: (6,15)

        if mask.sum() >= 3:
            log_k = torch.log(k[mask])
            log_E = torch.log(spectrum[mask])
            n = log_k.shape[0]
            slope = (n * (log_k * log_E).sum() - log_k.sum() * log_E.sum()) / \
                    (n * (log_k**2).sum() - log_k.sum()**2 + 1e-10)
            # Slope -2.5'ten daha dik (orn: -3.0) = asiri dissipasyon = laminarizasyon
            # slope=-3.0: -2.5 - (-3.0) = 0.5 > 0 → aktive olur ✓
            # slope=-2.0: -2.5 - (-2.0) = -0.5 < 0 → relu=0, aktive olmaz ✓
            L_slope_guard = torch.relu(-2.5 - slope).pow(2)
        else:
            L_slope_guard = spectrum.sum() * 0.0  # gradient akisi icin graph-connected sifir

        raw = L_tke_min + L_slope_guard
        return scale_loss(raw, ANTI_LAMINAR_SCALE)

    # ---------------------------------------------------------------- #
    # Phase D loss terimleri (Non-Boussinesq)                           #
    # ---------------------------------------------------------------- #

    def continuity_loss(self, states: List[ThermalFluidState]) -> torch.Tensor:
        """
        L_continuity_rho = mean(|d(rho)/dt + nabla.(rho*u)|^2)

        Ardisik state ciftleri uzerinden sikistirilamaz kutle korunumu.
        Sadece non_boussinesq modda anlamli -- Boussinesq'te rho=None
        oldugu icin otomatik 0 doner.

        Fizik: Sureklilik denklemi d(rho)/dt + div(rho*u) = 0
        FD yaklasim: (rho1 - rho0)/dt + div(rho1*u1) = 0
        """
        if not self._is_non_boussinesq() or len(states) < 2:
            return torch.tensor(0.0, device=states[0].u.device)

        dt = self.config.physics.dt
        loss = torch.tensor(0.0, device=states[0].u.device)
        count = 0

        for i in range(1, len(states)):
            s0, s1 = states[i - 1], states[i]
            if s0.rho is None or s1.rho is None:
                continue

            # d(rho)/dt yaklasim (forward Euler)
            drho_dt = (s1.rho - s0.rho) / dt

            # div(rho*u): spectral divergence of momentum density
            rho_u = s1.rho * s1.u
            rho_v = s1.rho * s1.v
            rho_w = s1.rho * s1.w
            div_rho_u = self.ops.divergence(rho_u, rho_v, rho_w)

            # Residual: d(rho)/dt + div(rho*u) = 0
            residual = drho_dt + div_rho_u
            loss = loss + residual.pow(2).mean()
            count += 1

        raw = loss / max(count, 1)
        return scale_loss(raw, CONTINUITY_RHO_SCALE)

    # ---------------------------------------------------------------- #
    # Tier 2 (2026-04-29) — PDE Residual Loss                           #
    # NS + thermal denkleminin BIREBIR sağlanmasını cezalandırır.       #
    # Soft loss Goodhart problemini çözer: optimizer artık denklem-     #
    # değiştirici param'ları bozmaya çalışırken doğrudan ceza alır.    #
    # ---------------------------------------------------------------- #

    def ns_residual_loss(self, states: List[ThermalFluidState]) -> torch.Tensor:
        """
        L_NS_residual = mean( |(u_{n+1}-u_n)/dt - RHS_canonical|² )

        RHS_canonical (Boussinesq, kanonik DNS form):
          ∂_t u = -(u·∇)u + ν∇²u + Ri·θ·ê_y + F - ∇p

        ν_t (SGS) YOK SAY — kanonik DNS form, SGS hata payı tolerans olarak kalır.
        Tier 1 freeze ile zaten Boussinesq=1 · advection_mod=1 — RHS
        gerçekten kanonik. Eğer optimizer SGS Cs aracılığıyla ν_t'yi
        manipüle ederse residual büyür.
        """
        if len(states) < 2:
            return torch.tensor(0.0, device=states[0].u.device)

        ops = self.ops
        cfg = self.config
        nu = cfg.physics.nu
        Ri = cfg.physics.Ri
        Ly = cfg.domain.Ly
        dt = self.model._dt_base  # tek layer dt

        # Forcing (sabit, kanonik)
        Fx, Fy, Fz = self.model.forcing()

        loss = torch.tensor(0.0, device=states[0].u.device)
        n_pairs = 0

        # Son N_BALANCE_PAIRS+1 state üzerinden ardışık çiftler (yeterli sample)
        n_keep = min(self.N_BALANCE_PAIRS + 1, len(states))
        states_used = states[-n_keep:]

        for i in range(1, len(states_used)):
            s0, s1 = states_used[i - 1], states_used[i]

            # Time derivative: forward Euler yaklaşımı
            du_dt = (s1.u - s0.u) / dt
            dv_dt = (s1.v - s0.v) / dt
            dw_dt = (s1.w - s0.w) / dt

            # Convective form advection: (u·∇)u — kanonik form, pressure ham basınç olabilir
            # (Lamb form ω×u kullanılırsa pressure dynamic head içermeli; bu ambiguity'den
            # kaçınmak için convective form tercih edildi — Codex consultancy 2026-04-30)
            du_dx, du_dy, du_dz = ops.gradient(s0.u)
            dv_dx, dv_dy, dv_dz = ops.gradient(s0.v)
            dw_dx, dw_dy, dw_dz = ops.gradient(s0.w)
            adv_u = s0.u * du_dx + s0.v * du_dy + s0.w * du_dz
            adv_v = s0.u * dv_dx + s0.v * dv_dy + s0.w * dv_dz
            adv_w = s0.u * dw_dx + s0.v * dw_dy + s0.w * dw_dz
            adv_u = ops.dealias(adv_u)
            adv_v = ops.dealias(adv_v)
            adv_w = ops.dealias(adv_w)

            # Diffusion: ν∇²u (kanonik, ν_t YOK)
            lap_u = ops.laplacian(s0.u)
            lap_v = ops.laplacian(s0.v)
            lap_w = ops.laplacian(s0.w)
            diff_u = nu * lap_u
            diff_v = nu * lap_v
            diff_w = nu * lap_w

            # Buoyancy: Ri·θ·ê_y (kanonik, s=1)
            buoy_v = Ri * s0.theta

            # Pressure gradient: -∇p
            dp_dx, dp_dy, dp_dz = ops.gradient(s0.p)

            # RHS_canonical
            rhs_u = -adv_u + diff_u - dp_dx + Fx
            rhs_v = -adv_v + diff_v + buoy_v - dp_dy + Fy
            rhs_w = -adv_w + diff_w - dp_dz + Fz

            # Bug 6 fix (2026-05-01): gamma_damp residual'a ekle
            # model.py:563-564 v-component damping uyguluyor (Calzavarini 2005)
            # Residual eşit olmalı yoksa kalıcı bias → optimizer Cs/forcing'i bozar
            if (hasattr(self.model, '_gamma_damp')
                    and self.model.Ri > 0
                    and float(self.model._gamma_damp.max().item()) > 0.0):
                gamma = self.model._gamma_damp
                v_hat = ops.to_hat(s0.v)
                rhs_v = rhs_v - ops.from_hat(gamma * v_hat)

            # Residual
            R_u = du_dt - rhs_u
            R_v = dv_dt - rhs_v
            R_w = dw_dt - rhs_w

            loss = loss + (R_u.pow(2) + R_v.pow(2) + R_w.pow(2)).mean()
            n_pairs += 1

        raw = loss / max(n_pairs, 1)
        return scale_loss(raw, NS_RES_SCALE)

    def thermal_residual_loss(self, states: List[ThermalFluidState]) -> torch.Tensor:
        """
        L_thermal_residual = mean( |(θ_{n+1}-θ_n)/dt - RHS_θ|² )

        RHS_θ (kanonik, Boussinesq):
          ∂_t θ = -(u·∇)θ + κ∇²θ + v/Ly

        +v/Ly = mean-gradient kaynağı (T_base(y)=T_hot - dT·y/Ly,
        ∂_y T_base = -dT/Ly = -1/Ly nondim. Energy denkleminde
        -v·∂_y T_base = +v/Ly).

        κ_t (SGS) YOK SAY. Tier 1 freeze ile kappa_scale=1 kanonik.
        """
        if len(states) < 2:
            return torch.tensor(0.0, device=states[0].u.device)

        ops = self.ops
        cfg = self.config
        kappa = cfg.physics.kappa
        Ly = cfg.domain.Ly
        dt = self.model._dt_base

        loss = torch.tensor(0.0, device=states[0].u.device)
        n_pairs = 0

        n_keep = min(self.N_BALANCE_PAIRS + 1, len(states))
        states_used = states[-n_keep:]

        for i in range(1, len(states_used)):
            s0, s1 = states_used[i - 1], states_used[i]

            # Time derivative
            dtheta_dt = (s1.theta - s0.theta) / dt

            # Convective form: u·∇θ (dealiased)
            dT_dx, dT_dy, dT_dz = ops.gradient(s0.theta)
            adv_theta = s0.u * dT_dx + s0.v * dT_dy + s0.w * dT_dz
            adv_theta = ops.dealias(adv_theta)

            # Diffusion: κ∇²θ
            lap_theta = ops.laplacian(s0.theta)
            diff_theta = kappa * lap_theta

            # Mean-gradient source: +v/Ly
            src_theta = s0.v / Ly

            # RHS
            rhs_theta = -adv_theta + diff_theta + src_theta

            # Bug 6 fix (2026-05-01): gamma_damp thermal'da da var (model.py:639-640)
            if (hasattr(self.model, '_gamma_damp')
                    and self.model.Ri > 0
                    and float(self.model._gamma_damp.max().item()) > 0.0):
                gamma = self.model._gamma_damp
                theta_hat = ops.to_hat(s0.theta)
                rhs_theta = rhs_theta - ops.from_hat(gamma * theta_hat)

            # Residual
            R_theta = dtheta_dt - rhs_theta
            loss = loss + R_theta.pow(2).mean()
            n_pairs += 1

        raw = loss / max(n_pairs, 1)
        return scale_loss(raw, TH_RES_SCALE)

    def state_equation_loss(self, state: ThermalFluidState) -> torch.Tensor:
        """
        L_state = mean(|p - rho * R_specific * T_total|^2 / p_0^2)

        Durum denklemi (EOS) tutarliligi. Non-Boussinesq modda
        basinc alani ideal gaz denklemine uygunluk gosteren yogunluk
        ve sicaklik alanlariyla tutarli olmali.

        Normalizasyon: p_0 = rho_0 * R * T_0 ile boyutsuzlestirilir.
        Self-contained: model'deki state_equation noronuna bagli degil.
        """
        if not self._is_non_boussinesq() or state.rho is None:
            return torch.tensor(0.0, device=state.u.device)

        # model._compute_T_total ile AYNI hesap (tutarlilik icin)
        T_total = self.model._compute_T_total(state.theta, state.u.device)

        # Referans yogunluk ve basinc (ideal gaz normalizasyonu)
        # rho_0 = 1.0 (boyutsuz), R_specific = 1.0 (boyutsuz)
        # p_ideal = rho * T_total (boyutsuz ideal gaz)
        rho_0 = 1.0
        p_0 = 1.0  # nondim referans basinc

        p_ideal = state.rho * T_total
        residual = (state.p - p_ideal) / p_0
        raw = residual.pow(2).mean()
        return scale_loss(raw, STATE_EQ_SCALE)

    def mass_conservation_loss(
        self, states: List[ThermalFluidState]
    ) -> torch.Tensor:
        """
        L_mass = mean(|<rho_new> - <rho_old>|^2)

        Global (domain-ortalama) yogunluk korunumu. Toplam kutle
        degismemeli -- kapalı domain'de muhafaza edici bir constraint.

        Continuity_loss'dan farki: bu INTEGRAL (ortalama) korunum,
        continuity_loss LOCAL (noktasal) PDE residuali.
        """
        if not self._is_non_boussinesq() or len(states) < 2:
            return torch.tensor(0.0, device=states[0].u.device)

        loss = torch.tensor(0.0, device=states[0].u.device)
        count = 0

        for i in range(1, len(states)):
            s0, s1 = states[i - 1], states[i]
            if s0.rho is None or s1.rho is None:
                continue

            # Domain-ortalama yogunluk (spatial dims uzerinden)
            rho_mean_old = s0.rho.mean(dim=(-3, -2, -1))  # [B]
            rho_mean_new = s1.rho.mean(dim=(-3, -2, -1))  # [B]

            loss = loss + (rho_mean_new - rho_mean_old).pow(2).mean()
            count += 1

        raw = loss / max(count, 1)
        return scale_loss(raw, MASS_SCALE)

    # ---------------------------------------------------------------- #
    # Toplam loss (curriculum agirlikli)                                #
    # ---------------------------------------------------------------- #

    def compute_all(
        self, states: List[ThermalFluidState], weights: Dict[str, float],
        Re: Optional[float] = None, Ra: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Tum loss terimlerini hesapla ve curriculum agirliklari ile agirliklandir.

        Args:
            states: Unrolled zaman adimlarindan ThermalFluidState listesi
            weights: CurriculumScheduler'dan gelen agirlik dict'i
            Re: Aktif Reynolds sayisi (LES referans secimi icin)
            Ra: Aktif Rayleigh sayisi (LES referans sadece Ra=1e5 icin gecerli)
        Returns:
            Loss dict: {"L_divergence": weighted_val, ...}
        """
        device = states[0].u.device
        last_state = states[-1]

        # -- Enstrophy cache: ihtiyac duyulan state'ler icin 1 kez hesapla --
        # energy_balance (son 3 cift = states[-4:]), dissipation (son), stability (son),
        # Z_ref (ilk) hepsi _enstrophy kullaniyor. Cache ile 3x tekrar hesap onlenir.
        # Onceki: ~252 FFT/step enstrophy icin. Simdi: ~30 FFT/step.
        _enstrophy_cache: Dict[int, torch.Tensor] = {}

        # Z_ref: ilk state (index 0)
        _enstrophy_cache[0] = self._enstrophy(states[0])
        self._Z_ref = _enstrophy_cache[0].detach().mean().clamp(min=1.0)

        # energy_balance icin son N+1 state (N cift), dissipation+stability icin son state
        eb_start = max(0, len(states) - self.N_BALANCE_PAIRS - 1)
        for idx in range(eb_start, len(states)):
            if idx not in _enstrophy_cache:
                _enstrophy_cache[idx] = self._enstrophy(states[idx])

        # Son state kesinlikle cache'de (dissipation + stability icin)
        last_idx = len(states) - 1
        if last_idx not in _enstrophy_cache:
            _enstrophy_cache[last_idx] = self._enstrophy(last_state)

        # LES referans verisi (Re'ye gore, sadece Ra=1e5 icin gecerli)
        _ra_match = Ra is not None and abs(Ra - 1e5) < 1.0
        ref = LES_REFERENCE.get(int(Re), None) if Re is not None and _ra_match else None

        loss_dict: Dict[str, torch.Tensor] = {}

        # Spectrum cache: spectral loss'lar ayni E(k)'yi paylarir (6x FFT tasarrufu)
        _needs_spectrum = any(
            weights.get(k, 0) > 0
            for k in ("L_spectrum", "L_spectral_entropy", "L_slope_ref",
                       "L_entropy_ref", "L_spectrum_shape", "L_anti_laminar")
        )
        _spectrum = (
            compute_energy_spectrum(last_state.u, last_state.v, last_state.w, self.ops)
            if _needs_spectrum else None
        )

        # 1. Divergence -- sadece son state (Leray projector div-free garanti)
        # Onceki: tum 20 state = 80 FFT/step. Simdi: 1 state = 4 FFT/step.
        if weights.get("L_divergence", 0) > 0:
            div_loss = self.divergence_loss(last_state)
            loss_dict["L_divergence"] = weights["L_divergence"] * div_loss

        # 2. Energy balance -- son 3 ardisik cift (enstrophy cache ile)
        if weights.get("L_energy_balance", 0) > 0:
            eb_loss = self.energy_balance_loss(states, enstrophy_cache=_enstrophy_cache)
            loss_dict["L_energy_balance"] = weights["L_energy_balance"] * eb_loss

        # 3. Spectrum -- son state (guard rail: -5/3)
        if weights.get("L_spectrum", 0) > 0:
            sp_loss = self.spectrum_loss(last_state, spectrum=_spectrum)
            loss_dict["L_spectrum"] = weights["L_spectrum"] * sp_loss

        # 4. Dissipation -- son state (cached enstrophy ile)
        if weights.get("L_dissipation", 0) > 0:
            diss_loss = self.dissipation_loss(last_state, cached_Z=_enstrophy_cache.get(last_idx))
            loss_dict["L_dissipation"] = weights["L_dissipation"] * diss_loss

        # 5. Thermal variance -- tum state'ler
        if weights.get("L_thermal_var", 0) > 0:
            tv_loss = torch.stack(
                [self.thermal_variance_loss(s) for s in states[1:]]
            ).mean()
            loss_dict["L_thermal_var"] = weights["L_thermal_var"] * tv_loss

        # 6. Stability -- son state (cached enstrophy ile)
        if weights.get("L_stability", 0) > 0:
            stab_loss = self.stability_loss(last_state, cached_Z=_enstrophy_cache.get(last_idx))
            loss_dict["L_stability"] = weights["L_stability"] * stab_loss

        # 7. Spectral entropy -- guard rail (alt sinir)
        if weights.get("L_spectral_entropy", 0) > 0:
            se_loss = self.spectral_entropy_loss(last_state, spectrum=_spectrum)
            loss_dict["L_spectral_entropy"] = weights["L_spectral_entropy"] * se_loss

        # 8. Theta min -- isothermal collapse onleme (alt sinir)
        if weights.get("L_theta_min", 0) > 0:
            tm_loss = self.theta_min_loss(last_state)
            loss_dict["L_theta_min"] = weights["L_theta_min"] * tm_loss

        # 9. CFL guard -- patlama onleme, TUM state'ler uzerinden max
        if weights.get("L_cfl_guard", 0) > 0:
            cfl_losses = [self.cfl_guard_loss(s) for s in states]
            cfl_loss = torch.stack(cfl_losses).max()
            loss_dict["L_cfl_guard"] = weights["L_cfl_guard"] * cfl_loss

        # ---- LES Referans Loss'lari ----
        # Sadece referans verisi olan Re degerleri + Ra=1e5 icin aktif.
        if ref is not None:
            # 10. TKE referans (log-space, eski L_tke_min'in yerini alir)
            if weights.get("L_tke_ref", 0) > 0:
                tke_loss = self.tke_reference_loss(last_state, ref["TKE"])
                loss_dict["L_tke_ref"] = weights["L_tke_ref"] * tke_loss

            # 10b. Intermediate TKE penalty -- her 5 adimda TKE kontrolu
            if weights.get("L_tke_intermediate", 0) > 0:
                inter_tke_losses = []
                for si in range(4, len(states), 5):  # adim 5, 10, 15, 20
                    inter_tke = self.tke_reference_loss(states[si], ref["TKE"])
                    inter_tke_losses.append(inter_tke)
                if inter_tke_losses:
                    loss_dict["L_tke_intermediate"] = (
                        weights["L_tke_intermediate"] * torch.stack(inter_tke_losses).mean()
                    )

            # 11. TKE overshoot -- patlama onleme (TKE > 2x ref ceza)
            if weights.get("L_tke_overshoot", 0) > 0:
                tke_os_loss = self.tke_overshoot_loss(last_state, ref["TKE"])
                loss_dict["L_tke_overshoot"] = weights["L_tke_overshoot"] * tke_os_loss

            # 12. v*theta flux -- termal coupling (Nu icin kritik)
            if weights.get("L_v_theta", 0) > 0:
                # v5: theta_rms paydaya kondu, ref_theta_rms argumani gerekli
                vt_loss = self.v_theta_flux_loss(last_state, ref["v_theta_flux"],
                                                  ref_theta_rms=ref["theta_rms"])
                loss_dict["L_v_theta"] = weights["L_v_theta"] * vt_loss

            # 12b. v*theta korelasyon -- TKE-bagimsiz (deadlock kirici)
            if weights.get("L_corr", 0) > 0:
                corr_loss = self.v_theta_correlation_loss(
                    last_state, ref["v_theta_flux"], ref["TKE"], ref["theta_rms"]
                )
                loss_dict["L_corr"] = weights["L_corr"] * corr_loss

            # 13. theta_rms referans -- termal varyans kontrolu
            if weights.get("L_theta_rms_ref", 0) > 0:
                tr_loss = self.theta_rms_reference_loss(last_state, ref["theta_rms"])
                loss_dict["L_theta_rms_ref"] = weights["L_theta_rms_ref"] * tr_loss

            # 13b. v4: theta_rms ust sinir guard (anti-amplitude-pumping)
            if weights.get("L_theta_max", 0) > 0:
                tmax_loss = self.theta_max_loss(last_state, ref["theta_rms"])
                loss_dict["L_theta_max"] = weights["L_theta_max"] * tmax_loss

            # 9. Nu referans (log-space MSE, eski L_nusselt'in yerini alir)
            if weights.get("L_nu_ref", 0) > 0:
                nu_loss = self.nusselt_reference_loss(last_state, ref["Nu"])
                loss_dict["L_nu_ref"] = weights["L_nu_ref"] * nu_loss

            # 10. Slope referans (Re-spesifik hedef)
            if weights.get("L_slope_ref", 0) > 0:
                slope_loss = self.slope_reference_loss(last_state, ref["slope"], spectrum=_spectrum)
                loss_dict["L_slope_ref"] = weights["L_slope_ref"] * slope_loss

            # 11. Entropy referans (kesin hedef)
            if weights.get("L_entropy_ref", 0) > 0:
                ent_loss = self.entropy_reference_loss(last_state, ref["S_ent"], spectrum=_spectrum)
                loss_dict["L_entropy_ref"] = weights["L_entropy_ref"] * ent_loss

            # 12. Spectrum shape (tam E(k) eslestirmesi — simdilik DEVRE DISI)
            if weights.get("L_spectrum_shape", 0) > 0:
                ref_Ek = get_les_spectrum(int(Re), device)
                if ref_Ek is not None:
                    shape_loss = self.spectrum_shape_loss(last_state, ref_Ek, spectrum=_spectrum)
                    loss_dict["L_spectrum_shape"] = weights["L_spectrum_shape"] * shape_loss

        # ---- Tier 2 (2026-04-29): PDE Residual Losses ----
        # Continuum NS + thermal denkleminin BIREBIR sağlanmasını cezalandırır.
        # Goodhart problemine doğrudan çözüm: optimizer artık denklem'i bozamaz.
        if weights.get("L_NS_residual", 0) > 0:
            ns_res = self.ns_residual_loss(states)
            loss_dict["L_NS_residual"] = weights["L_NS_residual"] * ns_res

        if weights.get("L_thermal_residual", 0) > 0:
            th_res = self.thermal_residual_loss(states)
            loss_dict["L_thermal_residual"] = weights["L_thermal_residual"] * th_res

        # ---- Phase D: Non-Boussinesq losses ----
        if weights.get("L_continuity_rho", 0) > 0:
            cont_loss = self.continuity_loss(states)
            loss_dict["L_continuity_rho"] = weights["L_continuity_rho"] * cont_loss

        if weights.get("L_state", 0) > 0:
            state_loss = self.state_equation_loss(last_state)
            loss_dict["L_state"] = weights["L_state"] * state_loss

        if weights.get("L_mass", 0) > 0:
            mass_loss = self.mass_conservation_loss(states)
            loss_dict["L_mass"] = weights["L_mass"] * mass_loss

        # ---- INNATE v2: Germano Consistency Loss ----
        # Pahali hesaplama (ekstra FFT'ler), sadece weight>0 oldugunda calisir.
        if weights.get("L_germano", 0) > 0:
            germano_loss = self.germano_consistency_loss(last_state)
            loss_dict["L_germano"] = weights["L_germano"] * germano_loss

        # ---- INNATE v2: Anti-Laminarizasyon Loss ----
        # ref_tke: LES referanstan veya power-law interpolasyondan
        if weights.get("L_anti_laminar", 0) > 0:
            # TKE referans degeri: LES ref varsa ondan, yoksa power-law tahmin
            _ref_tke_al = ref["TKE"] if ref is not None else max(
                0.008479 * (Re / 7000.0) ** (-0.415), 1e-4
            ) if Re is not None else 0.008
            al_loss = self.anti_laminarization_loss(
                last_state, ref_tke=_ref_tke_al, spectrum=_spectrum
            )
            loss_dict["L_anti_laminar"] = weights["L_anti_laminar"] * al_loss

        # ---- INNATE v2: Nu Loss (L_nu_phys) ----
        # Buoyancy damping KALDIRILDI, artik Nu loss dogru gradient verebilir.
        # L_nu_ref (eski) hala weight=0 ile mevcut, L_nu_phys yeni key.
        if weights.get("L_nu_phys", 0) > 0 and ref is not None:
            nu_phys_loss = self.nusselt_reference_loss(last_state, ref["Nu"])
            loss_dict["L_nu_phys"] = weights["L_nu_phys"] * nu_phys_loss

        return loss_dict


# =====================================================================
# 3. CurriculumScheduler
# =====================================================================


class CurriculumScheduler:
    """
    INNATE v2 Curriculum Scheduler.

    5 Re noktasi (5K, 7K, 10K, 15K, 20K), 1500 epoch, 4 faz.
    Re secimi: stratified round-robin (deterministic, reproducible).
    Weight gecisleri: linear interpolation ile smooth ramp.

    Phase A (0-300):    Re=5K,7K — Guard rails + basic refs, warmup
    Phase B (300-600):  Re=5K,7K,10K — Ramp weights A->C
    Phase C (600-1000): Re=5K,7K,10K,15K — Full weights + Nu loss
    Phase D (1000-1500): Re=5K,7K,10K,15K,20K — + Germano loss, fine-tune
    """

    PHASE_BOUNDARIES = {
        "A": (0, 300),
        "B": (300, 600),
        "C": (600, 1000),
        "D": (1000, 1500),
    }

    @classmethod
    def set_phase_boundaries(cls, total_epochs: int):
        """Total epoch'a gore proportional curriculum kisalt.
        30 ep -> A=5, B=10, C=20, D=30 (literatur curriculum + tuning v2).
        """
        a = max(1, int(total_epochs * 0.17))
        b = max(a + 1, int(total_epochs * 0.34))
        c = max(b + 1, int(total_epochs * 0.67))
        d = max(c + 1, total_epochs)
        cls.PHASE_BOUNDARIES = {
            "A": (0, a), "B": (a, b), "C": (b, c), "D": (c, d),
        }

    # Re noktalari faza gore (stratified round-robin ile secilir)
    # 2026-05-01: Re=5K/15K/20K placeholder LES referansları — kaldırıldı.
    # Sadece validated 7K + 10K (les_reference/Re7000_Ra1e5/, Re10000_Ra1e5_v2/).
    RE_TABLE = {
        "A": [7000],
        "B": [7000, 10000],
        "C": [7000, 10000],
        "D": [7000, 10000],
    }

    # Ra: tum fazlarda sabit 1e5
    RA = 1e5

    # Phase A: 2026-04-27 TUNING v4 (CFD-expert yapisal cozum + Codex anti-fizik).
    # v3 ep1 Nu=244, theta_rms=0.097 (LES 0.024'un 4x'i) anti-fizik patladi.
    # CFD-expert teshisi: L_v_theta skaler hedef icin <v.theta> = alpha*theta_rms*v_rms
    # 3 boyutta serbest -> COKLU NULL SPACE. Model amplitude pumping yapiyor.
    # COZUM: L_v_theta=0 KALDIR. L_corr (alpha) + L_theta_rms_ref (theta_rms) +
    # L_tke_ref (v_rms) UC bagimsiz constraint -> null space tek noktaya daralir.
    # Codex eki: Nu overshoot weight 5->20 (anti-fizik fren), L_theta_max safety.
    # === Tier 3 (2026-04-29): MINIMAL 5-loss seti (27 → 5) ===
    # Önceki 27 loss çelişiyordu (anti_laminar pumping'i ödüllendiriyor, Nu/vT
    # duplicate gradient, theta_max/rms_ref residual'la redundant). Tier 1+2
    # sayesinde denklem-değiştirici param'lar fixed ve denklem doğrudan
    # cezalandırılıyor — istatistik loss'lar gereksiz.
    #
    # TUTULAN (5):
    #   L_NS_residual       — momentum denklemi (Tier 2)
    #   L_thermal_residual  — termal denklem (Tier 2)
    #   L_divergence        — ∇·u=0 (Leray ek garantisi)
    #   L_corr              — vθ/(√TKE·θ_rms) amplitude-invariant (Goodhart-immün)
    #   L_spectrum_shape    — LES E(k) referansı (yumuşak regularizer)
    WEIGHTS_A = {
        "L_NS_residual":      10.0,
        "L_thermal_residual": 10.0,
        "L_divergence":       10.0,
        "L_corr":             15.0,
        "L_spectrum_shape":    5.0,
        # === Atılan 22 loss (Tier 3 minimalleştirme) ===
        "L_energy_balance":    0.0,
        "L_spectrum":          0.0,
        "L_dissipation":       0.0,
        "L_thermal_var":       0.0,
        "L_stability":         0.0,
        "L_spectral_entropy":  0.0,
        "L_theta_min":         0.0,
        "L_cfl_guard":         0.0,
        "L_anti_laminar":      0.0,
        "L_tke_ref":           0.0,
        "L_tke_overshoot":     0.0,
        "L_tke_intermediate":  0.0,
        "L_theta_rms_ref":     0.0,
        "L_v_theta":           0.0,
        "L_nu_ref":            0.0,
        "L_slope_ref":         0.0,
        "L_entropy_ref":       0.0,
        "L_nu_phys":           0.0,
        "L_theta_max":         0.0,
        "L_germano":           0.0,
    }

    # Phase C: Tier 3 minimal — 2026-05-08 Codex consultation sonrası WEIGHTS_A
    # ile EŞIT yapildi (jump=0). Onceden 15/15/10/20/8 (sum=68, %36 jump) Phase C
    # divergence'a sebep oluyordu. Codex onerisi: "Ilk koşuda hiç jump koyma,
    # stable kalirsa sonraki iterasyonda sum=57 softened C'ye 20 epoch ramp et."
    # Re=7K -> Re=10K geçişinin yarattigi yuk zaten yeterli stress.
    WEIGHTS_C = {
        "L_NS_residual":      10.0,
        "L_thermal_residual": 10.0,
        "L_divergence":       10.0,
        "L_corr":             15.0,
        "L_spectrum_shape":    5.0,
        # Atılan 22 loss
        "L_energy_balance":    0.0,
        "L_spectrum":          0.0,
        "L_dissipation":       0.0,
        "L_thermal_var":       0.0,
        "L_stability":         0.0,
        "L_spectral_entropy":  0.0,
        "L_theta_min":         0.0,
        "L_cfl_guard":         0.0,
        "L_anti_laminar":      0.0,
        "L_tke_ref":           0.0,
        "L_tke_overshoot":     0.0,
        "L_tke_intermediate":  0.0,
        "L_theta_rms_ref":     0.0,
        "L_v_theta":           0.0,
        "L_nu_ref":            0.0,
        "L_slope_ref":         0.0,
        "L_entropy_ref":       0.0,
        "L_nu_phys":           0.0,
        "L_theta_max":         0.0,
        "L_germano":           0.0,
    }

    # Phase D: Tier 3 minimal — Phase C ile aynı + L_germano (SGS consistency)
    WEIGHTS_D = {
        "L_NS_residual":      15.0,
        "L_thermal_residual": 15.0,
        "L_divergence":       10.0,
        "L_corr":             20.0,
        "L_spectrum_shape":    8.0,
        # Phase D ek: Germano (SGS dynamic kontrol, opsiyonel)
        "L_germano":           5.0,
        # Atılan 21 loss
        "L_energy_balance":    0.0,
        "L_spectrum":          0.0,
        "L_dissipation":       0.0,
        "L_thermal_var":       0.0,
        "L_stability":         0.0,
        "L_spectral_entropy":  0.0,
        "L_theta_min":         0.0,
        "L_cfl_guard":         0.0,
        "L_anti_laminar":      0.0,
        "L_tke_ref":           0.0,
        "L_tke_overshoot":     0.0,
        "L_tke_intermediate":  0.0,
        "L_theta_rms_ref":     0.0,
        "L_v_theta":           0.0,
        "L_nu_ref":            0.0,
        "L_slope_ref":         0.0,
        "L_entropy_ref":       0.0,
        "L_nu_phys":           0.0,
        "L_theta_max":         0.0,
    }

    def __init__(self, config: Config):
        self.config = config

    def get_phase(self, epoch: int) -> str:
        """Epoch'a gore aktif fazi dondur."""
        for phase, (start, end) in self.PHASE_BOUNDARIES.items():
            if start <= epoch < end:
                return phase
        # Epoch son fazin otesindeyse son faz
        return "D"

    def get_weights(self, epoch: int) -> Dict[str, float]:
        """
        Epoch'a gore interpolated loss agirliklarini dondur.

        Faz gecislerinde linear interpolation ile smooth ramp:
        - A: sabit WEIGHTS_A
        - B: WEIGHTS_A -> WEIGHTS_C linear ramp
        - C: sabit WEIGHTS_C
        - D: WEIGHTS_C -> WEIGHTS_D linear ramp (Germano dahil)
        """
        phase = self.get_phase(epoch)

        if phase == "A":
            return dict(self.WEIGHTS_A)

        elif phase == "B":
            # Phase B: A -> C linear ramp-up
            b_start, b_end = self.PHASE_BOUNDARIES["B"]
            t = (epoch - b_start) / max(b_end - b_start, 1)
            t = max(0.0, min(1.0, t))
            weights = {}
            all_keys = set(self.WEIGHTS_A) | set(self.WEIGHTS_C)
            for key in all_keys:
                w_a = self.WEIGHTS_A.get(key, 0.0)
                w_c = self.WEIGHTS_C.get(key, 0.0)
                weights[key] = w_a + t * (w_c - w_a)
            return weights

        elif phase == "C":
            return dict(self.WEIGHTS_C)

        else:
            # Phase D: C -> D linear ramp-up (Germano dahil)
            d_start, d_end = self.PHASE_BOUNDARIES["D"]
            t = (epoch - d_start) / max(d_end - d_start, 1)
            t = max(0.0, min(1.0, t))
            weights = {}
            all_keys = set(self.WEIGHTS_C) | set(self.WEIGHTS_D)
            for key in all_keys:
                w_c = self.WEIGHTS_C.get(key, 0.0)
                w_d = self.WEIGHTS_D.get(key, 0.0)
                weights[key] = w_c + t * (w_d - w_c)
            return weights

    def get_physics_params(self, epoch: int) -> Tuple[float, float]:
        """
        Epoch'a gore Re, Ra sec.

        Re secimi: stratified round-robin (deterministic).
        Re = Re_list[epoch % len(Re_list)]
        Bu sayede her Re noktasi esit pay alir ve seed-bagimsiz reproducible.
        """
        phase = self.get_phase(epoch)
        re_list = self.RE_TABLE[phase]
        Re = re_list[epoch % len(re_list)]
        return float(Re), float(self.RA)


# =====================================================================
# 4. LR Scheduler (Warmup + Cosine Annealing)
# =====================================================================


def create_lr_scheduler(
    optimizer: torch.optim.Optimizer, config: Config
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Warmup + cosine annealing with warm restarts LR schedule.

    0 -> 100 epochs: linear warmup 1e-5 -> 3e-4
    100 -> 1500 epochs: cosine with warm restarts (T_0=500)
    T_0=500 ile 1400 post-warmup epoch'ta ~3 cycle.
    Her restart'ta LR tekrar peak'e cikar, plato kirilir.
    """
    # Bug 3 fix (2026-05-01): warmup ve T_0 max_epochs orantılı
    # Eski: warmup=100 + max_epochs=100 → tüm eğitim warmup, cosine hiç çalışmaz
    max_epochs = max(int(config.training.max_epochs), 1)
    warmup = max(1, min(int(config.training.warmup_epochs), max_epochs // 5))  # max %20 warmup
    base_lr = config.training.lr
    min_lr = 1e-5
    # T_0 post-warmup süresinin ~%50-70'i (1-2 cosine cycle)
    T_0 = max(10, int((max_epochs - warmup) * 0.6))

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            alpha = epoch / max(warmup, 1)
            return (min_lr + alpha * (base_lr - min_lr)) / base_lr
        else:
            epoch_since_warmup = epoch - warmup
            progress = (epoch_since_warmup % T_0) / T_0
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (min_lr + cosine_decay * (base_lr - min_lr)) / base_lr

    print(f"  [LR] warmup={warmup} ep, T_0={T_0} ep (max_epochs={max_epochs})")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =====================================================================
# 5. Checkpoint
# =====================================================================


def save_checkpoint(
    model: INNATE3D_MixedConvection,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    config: Config,
    loss_dict: Optional[Dict[str, float]] = None,
    scheduler=None,
):
    """
    Training checkpoint kaydet.
    model state, optimizer state, scheduler state, epoch, config, ve son loss'lar dahil.
    """
    path = Path(config.training.checkpoint_dir) / f"checkpoint_epoch{epoch:06d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    # torch.compile sarmaliysa _orig_mod'dan al (temiz key'ler icin)
    raw_model = getattr(model, "_orig_mod", model)
    payload = {
        "epoch": epoch,
        "model": raw_model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "config": config.to_dict(),
    }
    if loss_dict is not None:
        payload["losses"] = loss_dict
    torch.save(payload, path)
    print(f"  [Checkpoint] {path}")


# =====================================================================
# 5b. Transfer Learning: Phase D Optimizer Setup
# =====================================================================


# =====================================================================
# 5b. Gradient Routing (3-Group Parameter Partitioning)
# =====================================================================

# Momentum losses: TKE, spectrum, dissipation — Cs, forcing, advection parametrelerini yonlendirir
MOMENTUM_LOSSES = frozenset({
    "L_tke_ref", "L_tke_intermediate", "L_spectrum_shape",
    "L_slope_ref", "L_spectral_entropy", "L_entropy_ref", "L_spectrum",
    "L_dissipation",  # Codex: velocity/spectrum constraint, momentum'a ait
    "L_germano",      # INNATE v2: SGS model consistency — momentum parametrelerini yonlendirir
    "L_anti_laminar",  # INNATE v2: anti-laminarizasyon — TKE + spectrum slope guard
})

# Thermal losses: v*theta, theta — termal parametreleri yonlendirir
THERMAL_LOSSES = frozenset({
    "L_v_theta", "L_theta_min", "L_thermal_var",
    "L_corr",           # Coordinator: momentum'a gradient gondermesin
    "L_theta_rms_ref",  # Coordinator: termal parametrelere yogunlassin
    "L_theta_max",      # 2026-04-27 v4: saf termal amplitude freni, thermal grubuna ait
    "L_thermal_residual",  # 2026-04-30 reviewer: thermal residual SADECE termal denkleme
                           # ait, momentum SGS Cs'ye gradient gondermesin
})

# Diger her sey SHARED: divergence, energy_balance, stability, cfl_guard, L_corr
# L_corr (Codex): v, TKE, theta_rms iceriyor — amplitude-normalized, pure thermal degil
# Shared losses tum parametrelere gradient verir.

# Parametre siniflandirma keyword'leri
_MOMENTUM_KEYWORDS = ("advection", "eddy", "smagorinsky", "forcing", "backscatter", "aniso",
                      "mlp_sgs", "C_ss")  # INNATE v2: MLP SGS + scale-similarity coeff
_THERMAL_KEYWORDS = ("thermal", "kappa", "cs_thermal", "buoyancy")  # pr_t -> cs_thermal (bagimsiz termal SGS)
_DT_KEYWORDS = ("dt_scale", "dt_mults")
# cs_re_a, cs_re_b KALDIRILDI (INNATE v2'de yok — MLP SGS bunlarin yerini aldi)
# Bridge: buoyancy — her iki gruptan gradient alir
# DT: dt_scale, dt_mults — sadece SHARED losses gorur (Codex: integrator loophole onleme)


def _classify_parameters(model: INNATE3D_MixedConvection):
    """
    Model parametrelerini 4 gruba ayir: momentum, thermal, bridge, dt.

    Codex onerisi: dt parametreleri ayri grup, sadece shared/guard losses gorur.
    Buoyancy ise bridge olarak kalir (hem momentum hem thermal gradient alir).

    Returns:
        (momentum_ids, thermal_ids, bridge_ids, dt_ids) — her biri set of id(param)
    """
    momentum_ids = set()
    thermal_ids = set()
    bridge_ids = set()
    dt_ids = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pid = id(param)
        # DT parametreleri en spesifik — once kontrol et
        if any(kw in name for kw in _DT_KEYWORDS):
            dt_ids.add(pid)
        # Thermal: thermal_advections, thermal_diffusions, kappa
        elif any(kw in name for kw in _THERMAL_KEYWORDS):
            thermal_ids.add(pid)
        # Momentum: advection, eddy, forcing, etc.
        elif any(kw in name for kw in _MOMENTUM_KEYWORDS):
            momentum_ids.add(pid)
        # Bridge: buoyancy
        else:
            bridge_ids.add(pid)

    return momentum_ids, thermal_ids, bridge_ids, dt_ids


def _create_phase_d_optimizer(
    model: INNATE3D_MixedConvection, config: Config
) -> torch.optim.AdamW:
    """
    Phase D transfer learning icin per-parameter-group optimizer.

    Parametre gruplari:
      1. Frozen (cok dusuk lr): Momentum layers + Projection
         Bunlar Phase A-C'de ogrenilmis, korunmali.
      2. Fine-tune (orta lr): Forcing, Buoyancy, Thermal noronlar
         Mevcut bilgiyi koruyarak ince ayar.
      3. New (yuksek lr): Density-related noronlar (Phase D'de yeni)
         Sifirdan ogrenmesi gereken parametreler.

    Returns:
        Konfigured AdamW optimizer
    """
    # Parametre isimlerine gore gruplama
    frozen_params = []     # layers.*, projection.*
    finetune_params = []   # forcing.*, buoyancy.*, thermal.*
    new_params = []        # density.*, var_density.*, continuity.*, state_equation.*

    frozen_keywords = ("layers", "projection")
    finetune_keywords = ("forcing", "buoyancy", "thermal")
    new_keywords = ("density", "var_density", "continuity", "state_equation")

    assigned = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if any(kw in name for kw in new_keywords):
            new_params.append(param)
            assigned.add(name)
        elif any(kw in name for kw in finetune_keywords):
            finetune_params.append(param)
            assigned.add(name)
        elif any(kw in name for kw in frozen_keywords):
            frozen_params.append(param)
            assigned.add(name)
        else:
            # Bilinmeyen parametreler fine-tune grubuna
            finetune_params.append(param)
            assigned.add(name)

    param_groups = []
    if frozen_params:
        param_groups.append({
            "params": frozen_params,
            "lr": 1e-5,
            "name": "frozen",
        })
    if finetune_params:
        param_groups.append({
            "params": finetune_params,
            "lr": 5e-5,
            "name": "finetune",
        })
    if new_params:
        param_groups.append({
            "params": new_params,
            "lr": 3e-4,
            "name": "new",
        })

    # Fallback: hic param_group yoksa (olamaz ama guvenlik)
    if not param_groups:
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.training.lr,
            weight_decay=0,
        )

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=0,  # Fizik parametreleri weight decay'den zarar gorur
        betas=(0.9, 0.999),
    )

    # Log
    print("  [Transfer Learning] Phase D optimizer param groups:")
    for pg in param_groups:
        n_params = sum(p.numel() for p in pg["params"])
        print(f"    {pg.get('name', '?'):12s}: {n_params:6d} params, lr={pg['lr']:.1e}")

    return optimizer


# =====================================================================
# 6. Training Loop
# =====================================================================


def train(
    config: Optional[Config] = None, resume_from: Optional[str] = None
) -> INNATE3D_MixedConvection:
    """
    INNATE v2 training fonksiyonu.

    Physics-only training loop (1500 epoch, 5 Re noktasi):
    1. Curriculum'a gore Re sec (stratified round-robin)
    2. model.set_physics(Re, Ra) ile fizik parametrelerini ayarla
    3. IC olustur
    4. num_steps adim unroll et
    5. Physics loss hesapla (curriculum agirliklari ile)
    6. Backprop + gradient clipping + optimizer step
    7. Log + checkpoint

    Curriculum:
    - Phase A (0-300):    Re=5K,7K — warmup, basic guard rails
    - Phase B (300-600):  Re=5K,7K,10K — weight ramp A->C
    - Phase C (600-1000): Re=5K,7K,10K,15K — full weights + Nu loss
    - Phase D (1000-1500): Re=5K,7K,10K,15K,20K — + Germano, fine-tune

    Args:
        config: Config instance (None ise default Config() kullanilir)
        resume_from: Checkpoint dosyasi yolu (opsiyonel)
    Returns:
        Egitilmis model
    """
    if config is None:
        config = Config()

    # 2026-05-01: Reproducibility seed (ML-Expert #6)
    _seed = int(getattr(config.training, 'seed', 42))
    torch.manual_seed(_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_seed)
    import random as _r
    _r.seed(_seed); np.random.seed(_seed)
    print(f"  [SEED] manual_seed={_seed}")

    device = config.device
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # CUDA optimizasyonlari
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    elif device.type == "mps":
        print(f"  Apple Silicon MPS backend")
    print(config)

    # -- Model --
    model = INNATE3D_MixedConvection(config).to(device)
    print(f"Parameters: {model.count_parameters()}")

    # 2026-04-26 TUNING v2: forcing_A freeze (cfd-expert + Eswaran-Pope DNS standart)
    # Goodhart sebebi: forcing_A ogrenilebilir oldugunda model "vanayi kisarak"
    # L_energy_balance trivial null space'e yakinsiyor. Sabit forcing ile model
    # "TKE'yi soldur" hilesi yapamaz. Phase A-B boyunca frozen, Phase C'den sonra
    # serbest birakilir (varsa termal kuplaj zaten kurulmus olur).
    if getattr(config.training, 'freeze_forcing', False):
        if hasattr(model, 'forcing') and hasattr(model.forcing, 'amplitude'):
            model.forcing.amplitude.requires_grad = False
            with torch.no_grad():
                model.forcing.amplitude.fill_(0.005)  # LES referans degeri
            print(f"  [forcing FREEZE] amplitude=0.005 sabitlendi (Phase A-B)")
            for attr in ('amplitude_k2', 'amplitude_k3'):
                if hasattr(model.forcing, attr):
                    p = getattr(model.forcing, attr)
                    p.requires_grad = False
                    print(f"  [forcing FREEZE] {attr} frozen")

    # === Tier 1 — Param hijyeni (2026-04-29) ===
    # Denklem-değiştirici parametreleri sabitle (Boussinesq/Newton/Fick kanonik).
    # Optimizer denkleme dokunamasın — sadece SGS Cs/cs_thermal trainable kalsın.
    if getattr(config.training, 'freeze_canonical_params', False):
        n_frozen = 0

        # Buoyancy3D.buoyancy_strength = 1.0 (Boussinesq sabit)
        if hasattr(model, 'buoyancies'):
            for b in model.buoyancies:
                if hasattr(b, 'buoyancy_strength'):
                    with torch.no_grad():
                        b.buoyancy_strength.fill_(1.0)
                    b.buoyancy_strength.requires_grad = False
                    n_frozen += 1
        elif hasattr(model, 'buoyancy') and hasattr(model.buoyancy, 'buoyancy_strength'):
            with torch.no_grad():
                model.buoyancy.buoyancy_strength.fill_(1.0)
            model.buoyancy.buoyancy_strength.requires_grad = False
            n_frozen += 1

        # Advection3D.advection_modulator = 1.0 (Newton 2. yasası sabit)
        if hasattr(model, 'advections'):
            for a in model.advections:
                if hasattr(a, 'advection_modulator'):
                    with torch.no_grad():
                        a.advection_modulator.fill_(1.0)
                    a.advection_modulator.requires_grad = False
                    n_frozen += 1

        # ThermalAdvection3D.thermal_adv_modulator = 1.0
        if hasattr(model, 'thermal_advections'):
            for t in model.thermal_advections:
                if hasattr(t, 'thermal_adv_modulator'):
                    with torch.no_grad():
                        t.thermal_adv_modulator.fill_(1.0)
                    t.thermal_adv_modulator.requires_grad = False
                    n_frozen += 1

        # ThermalDiffusion3D.kappa_scale = 1.0 (Fick yasası sabit)
        if hasattr(model, 'thermal_diffusions'):
            for d in model.thermal_diffusions:
                if hasattr(d, 'kappa_scale'):
                    with torch.no_grad():
                        d.kappa_scale.fill_(1.0)
                    d.kappa_scale.requires_grad = False
                    n_frozen += 1
                # Anisotropic varsa
                for axis in ('kappa_scale_x', 'kappa_scale_y', 'kappa_scale_z'):
                    if hasattr(d, axis):
                        p = getattr(d, axis)
                        with torch.no_grad():
                            p.fill_(1.0)
                        p.requires_grad = False
                        n_frozen += 1

        # Bug 4 fix (2026-05-01) — dt_scale, dt_mults, backscatter_coeff freeze
        # Codex+Gemini: integrator loophole — optimizer dt'yi büyütüp residual'ı sahte azaltabilir
        if hasattr(model, 'dt_scale'):
            with torch.no_grad():
                model.dt_scale.fill_(1.0)
            model.dt_scale.requires_grad = False
            n_frozen += 1
        if hasattr(model, 'dt_mults'):
            for dm in model.dt_mults:
                with torch.no_grad():
                    dm.fill_(1.0)
                dm.requires_grad = False
                n_frozen += 1
        # backscatter_coeff: anti-diffusion riski (Gemini #3) — freeze 0.0 (kanonik LES sönümleme)
        for mod in model.modules():
            if hasattr(mod, 'backscatter_coeff'):
                p = mod.backscatter_coeff
                if hasattr(p, 'fill_'):
                    with torch.no_grad():
                        p.fill_(0.0)
                    p.requires_grad = False
                    n_frozen += 1

        n_train = sum(1 for p in model.parameters() if p.requires_grad)
        n_total = sum(1 for _ in model.parameters())
        print(f"  [Tier 1 FREEZE] {n_frozen} kanonik param fixed (Boussinesq/Newton/Fick + dt + backscatter)")
        print(f"  [Tier 1 FREEZE] Toplam {n_train}/{n_total} trainable (SGS Cs/cs_thermal kalan)")

    # -- Optimizer --
    # v2: standart AdamW (Non-Boussinesq transfer learning kaldirildi)
    # 2026-05-01: SADECE trainable param'lar — frozen Tier 1 paramları için
    # AdamW state'i (exp_avg/exp_avg_sq) boş üretmesin (state_dict şişmesin)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    n_train_p = sum(p.numel() for p in trainable_params)
    n_total_p = sum(p.numel() for p in model.parameters())
    print(f"  [OPTIM] AdamW {len(trainable_params)} param tensors "
          f"({n_train_p:,}/{n_total_p:,} elements trainable)")
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.999),
    )

    # -- LR Scheduler --
    scheduler = create_lr_scheduler(optimizer, config)

    # -- Curriculum + Physics Loss --
    curriculum = CurriculumScheduler(config)
    physics_loss = PhysicsLoss(config, model)

    # -- Resume --
    start_epoch = 0
    if resume_from is not None:
        print(f"Resuming from {resume_from}")
        checkpoint = torch.load(resume_from, weights_only=False, map_location=device)
        # strict=False: model degisikligi durumunda esneklik
        # PyTorch 2.5+: expanded buffer'lar (kx/ky/kz) copy hatasi verir, skip et
        _ckpt_state = checkpoint["model"]
        _skip = {k for k in _ckpt_state if any(
            s in k for s in ("diff_ops.k", "ops.kx", "ops.ky", "ops.kz",
                              "ops.k_sq", "ops.dealias", "_elevator_mask")
        )}
        for k in _skip:
            del _ckpt_state[k]
        missing, unexpected = model.load_state_dict(
            _ckpt_state, strict=False
        )
        if missing:
            print(f"  [Resume] Missing keys: {len(missing)}")
            for k in missing[:5]:
                print(f"    - {k}")
            if len(missing) > 5:
                print(f"    ... and {len(missing) - 5} more")
        if unexpected:
            print(f"  [Resume] Unexpected keys: {len(unexpected)}")

        # Optimizer state — RESET (2026-05-08 Codex consultation)
        # Eski koşuda Phase A→B ramp şokunda grad spike (8282) yaşandı.
        # Adam momentum buffer'larında bu kötü update direction kalıntısı kalmış
        # olabilir. Phase C divergence'ın katkı sebebi olabilir.
        # Resume'da optimizer state YÜKLEMİYORUZ → momentum=0, exp_avg=0 sıfır.
        # Sadece model weights yüklenir, optimizer fresh start.
        if checkpoint.get("optimizer") is not None:
            print(f"  [Resume] Optimizer state RESET (momentum kalıntısı temizlendi)")
        else:
            print(f"  [Resume] Optimizer state yok, fresh start")

        start_epoch = checkpoint["epoch"] + 1
        # LR scheduler state'i yukle (varsa), yoksa fallback loop
        if "scheduler" in checkpoint and checkpoint["scheduler"] is not None:
            try:
                scheduler.load_state_dict(checkpoint["scheduler"])
                print(f"  [Resume] Scheduler state yuklendi")
            except Exception as e:
                print(f"  [Resume] Scheduler state uyumsuz, loop fallback: {e}")
                for _ in range(start_epoch):
                    scheduler.step()
        else:
            for _ in range(start_epoch):
                scheduler.step()
        print(f"  Resumed at epoch {start_epoch}")
        # 2026-05-08 Codex: optimizer reset sonrası ilk 3 epoch LR×0.25 warmup
        # Adam momentum buffer'ı sıfır olunca ilk gradient agresif olabilir,
        # yumuşak başlangıç için LR'i geçici düşür.
        _resume_lr_warmup_remaining = 3
        _resume_lr_warmup_scale = 0.25
    else:
        _resume_lr_warmup_remaining = 0
        _resume_lr_warmup_scale = 1.0

    # -- torch.compile DISABLED (2026-05-07) --
    # 100ep koşusunda dynamo recompile döngüsü 5-6 saat hang yaptı (sorun.txt analiz):
    # "Malformed guard:" mesajları → reduce-overhead CUDA graph stride değişikliklerine
    # duyarlı + fullgraph=False Python conditional (model.py:494 _tke_ref) → cache invalid
    # → sonsuz recompile. Eager mode 4× yavaş ama GUARANTEED stable.
    # Hız kaybı: 5 dk/ep → 20 dk/ep. 100 ep = 33 saat (kabul edilebilir).
    raw_model = model
    _compiled_model = None
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    print("  [torch.compile] DISABLED — eager mode (dynamo hang fix 2026-05-07)")

    # -- Training Loop --
    print(f"\nTraining: epochs {start_epoch} -> {config.training.max_epochs}")
    print(f"  internal layers={config.model.n_layers}, dt={config.physics.dt}")
    print(f"  batch_size={config.training.batch_size}")
    print(f"  num_steps={config.training.num_steps}, loss_every={getattr(config.training, 'loss_every', 20)}")
    print(f"  gradient_checkpointing={config.model.gradient_checkpointing}")
    print(f"  curriculum phases: A(0-300) B(300-600) C(600-1000) D(1000-1500)")
    print()

    loss_dict = {}  # default for final checkpoint edge case
    epoch_times = []  # ETA hesabi icin

    # -- Gradient Routing Setup --
    # Bug 5 fix (2026-05-01): Tier 1 freeze sonrası THERMAL/BRIDGE param grupları BOŞ
    # routing aktifken L_corr ve L_thermal_residual sıfır gradient verir → kapat
    use_routing = config.training.use_gradient_routing and not getattr(
        config.training, 'freeze_canonical_params', False
    )
    if config.training.use_gradient_routing and not use_routing:
        print("  [ROUTING] Tier 1 freeze aktif → routing OTOMATİK KAPATILDI "
              "(thermal/bridge grupları boş, anlamsız)")
    if use_routing:
        momentum_ids, thermal_ids, bridge_ids, dt_ids = _classify_parameters(model)
        all_trainable = [p for p in model.parameters() if p.requires_grad]
        # Thermal sızıntı cıkarılacak parametreler (momentum + dt)
        mom_dt_params = [p for p in all_trainable if id(p) in momentum_ids or id(p) in dt_ids]
        # Routing artik loss step'lerine gore hesaplaniyor (training loop icinde)
        n_mom = sum(p.numel() for p in all_trainable if id(p) in momentum_ids)
        n_therm = sum(p.numel() for p in all_trainable if id(p) in thermal_ids)
        n_bridge = sum(p.numel() for p in all_trainable if id(p) in bridge_ids)
        n_dt = sum(p.numel() for p in all_trainable if id(p) in dt_ids)
        _le = getattr(config.training, 'loss_every', 20)
        _nls = max(1, config.training.num_steps // _le)
        _rls = max(1, _nls // 4)
        print(f"  [Gradient Routing] ACTIVE — 2-pass, last {_rls}/{_nls} loss steps")
        print(f"    momentum: {n_mom}  thermal: {n_therm}  bridge: {n_bridge}  dt: {n_dt} (shared-only)")

    # -- Early Stopping: Hedef metrikler karsilaninca dur --
    # Kosullar: TKE, slope, entropy hepsi +-15% band icinde,
    # son CONVERGENCE_WINDOW epoch boyunca stabil.
    CONVERGENCE_WINDOW = 10  # ardisik log_interval kontrol (10 x log_interval=10 = 100 epoch)
    CONVERGENCE_TOL = 0.20   # +-20% tolerans (oscillasyona toleransli)
    convergence_counter: Dict[int, int] = {}  # Re bazli ardisik basarili epoch sayaci

    _forcing_unfrozen = False  # Phase C girisinde unfreeze

    for epoch in range(start_epoch, config.training.max_epochs):
        model.train()
        t0 = time.time()

        # Resume LR warmup (2026-05-08): ilk 3 epoch LR×0.25, sonra normale dön
        if _resume_lr_warmup_remaining > 0:
            for pg in optimizer.param_groups:
                pg['lr'] = pg['lr'] * _resume_lr_warmup_scale
            print(f"  [Resume warmup] epoch={epoch} LR×{_resume_lr_warmup_scale} "
                  f"({_resume_lr_warmup_remaining} epoch kaldı)")
            _resume_lr_warmup_remaining -= 1

        # 1. Curriculum: Re/Ra ve loss agirliklari
        Re, Ra = curriculum.get_physics_params(epoch)
        weights = curriculum.get_weights(epoch)
        raw_model.set_physics(Re, Ra)

        # 1a. TUNING v2 forcing UNFREEZE — KALDIRILDI (2026-05-08)
        # Eski mantik: Phase C girisinde forcing.amplitude trainable yapiyordu.
        # Sorun: --freeze-forcing flag'in niyetini bozuyor, Phase C divergence'a
        # katki saglayan 3 faktorden biri (weight jump + Re=10K + UNFREEZE).
        # Codex consultation 2026-05-08: "forcing UNFREEZE'i kaldir, kullanici
        # niyeti kalici olsun." Phase C boyunca amplitude=0.005 sabit kalir.

        # 1b. TKE-coupled forcing damping: referans TKE set et
        _ra_ok = abs(Ra - 1e5) < 1.0
        _les_ref = LES_REFERENCE.get(int(Re), None) if _ra_ok else None
        if _les_ref is not None:
            raw_model._tke_ref.fill_(_les_ref["TKE"])
        else:
            # Bilinmeyen Re: iki-noktali power-law interpolasyon
            # TKE ~ Re^(-0.415), Re=7K -> 0.00848, Re=10K -> 0.00731
            _tke_est = 0.008479 * (Re / 7000.0) ** (-0.415)
            raw_model._tke_ref.fill_(max(_tke_est, 1e-4))

        # 2. Forcing phase reset (generalization icin)
        raw_model.forcing.reset_phase()

        # 3. Initial condition
        state = raw_model.create_initial_condition(
            batch_size=config.training.batch_size, device=device
        )

        # 4. Forward pass: autoregressive unrolling (TBPTT)
        # Her step = 20-layer fractional-step. num_steps kez tekrarla.
        # Sparse loss: her loss_every step'te loss hesapla + backward.
        # Arada sadece forward + detach (hizli).
        # Gradient routing sadece loss step'lerinde aktif.
        optimizer.zero_grad()
        total_loss = torch.tensor(0.0, device=device)
        loss_dict = {}
        grad_norm = torch.tensor(0.0)

        current = state
        num_steps = config.training.num_steps
        loss_every = getattr(config.training, 'loss_every', 20)
        n_loss_steps = max(1, num_steps // loss_every)
        scale = 1.0 / n_loss_steps  # loss step sayisina gore normalize

        for _step in range(num_steps):
            # Loss sadece her loss_every step'te hesaplanir
            is_loss_step = (_step + 1) % loss_every == 0 or _step == num_steps - 1

            if is_loss_step:
                # Loss step: intermediates lazim (gradient graph korunur)
                intermediates = model(current, return_intermediates=True)
                step_states = [current] + intermediates
                step_loss_dict = physics_loss.compute_all(step_states, weights, Re=Re, Ra=Ra)

                if len(step_loss_dict) > 0:
                    # Routing: son %25 loss step'lerinde aktif
                    loss_step_idx = (_step + 1) // loss_every
                    is_routing_step = (
                        use_routing
                        and loss_step_idx > n_loss_steps - max(1, n_loss_steps // 4)
                    )

                    # Bug 1 fix (2026-05-01): NaN/Inf loss guard — sessiz corruption önle
                    step_total_raw = sum(step_loss_dict.values())
                    if not torch.isfinite(step_total_raw):
                        print(f"  [NaN-GUARD] epoch={epoch} step={_step} "
                              f"loss={step_total_raw.item()} — backward atlandı")
                        optimizer.zero_grad(set_to_none=True)
                        continue

                    if is_routing_step:
                        # -- 2-PASS GRADIENT ROUTING --
                        step_total = step_total_raw * scale
                        therm_terms = [v for k, v in step_loss_dict.items() if k in THERMAL_LOSSES]
                        need_retain = len(therm_terms) > 0
                        step_total.backward(retain_graph=need_retain)

                        if therm_terms:
                            therm_loss = sum(therm_terms) * scale
                            leaked = torch.autograd.grad(
                                therm_loss, mom_dt_params,
                                retain_graph=False, allow_unused=True,
                            )
                            for p, g in zip(mom_dt_params, leaked):
                                if g is not None and p.grad is not None:
                                    p.grad.data.sub_(g)
                    else:
                        # -- STANDART: tek backward --
                        (step_total_raw * scale).backward()

                    # Loss tracking
                    for k, v_val in step_loss_dict.items():
                        if k in loss_dict:
                            loss_dict[k] = loss_dict[k] + v_val.detach() / n_loss_steps
                        else:
                            loss_dict[k] = v_val.detach() / n_loss_steps
                    total_loss = total_loss + sum(v.detach() for v in step_loss_dict.values()) / n_loss_steps

            else:
                # Non-loss step: sadece final state lazim (bellek tasarrufu)
                # Compiled model varsa kullan (kernel fusion + CUDA graphs)
                _fwd = _compiled_model if _compiled_model is not None else model
                with torch.no_grad():
                    intermediates = _fwd(current, return_intermediates=False)

            # TBPTT: sonraki step icin current'i detach et
            _last = intermediates[-1] if isinstance(intermediates, list) else intermediates
            current = ThermalFluidState(
                u=_last.u.detach(),
                v=_last.v.detach(),
                w=_last.w.detach(),
                p=_last.p.detach(),
                theta=_last.theta.detach(),
                t=_last.t,
                rho=_last.rho.detach() if _last.rho is not None else None,
            )

        # 6. Gradient clip + optimizer step
        # Bug 1 fix (2026-05-01): grad NaN/Inf guard — clip_grad_norm NaN'ı yutmaz
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), config.training.grad_clip
        )
        if not torch.isfinite(grad_norm):
            print(f"  [NaN-GUARD] epoch={epoch} grad_norm={grad_norm} — step atlandı")
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            continue
        optimizer.step()
        scheduler.step()

        # 7. Log
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        if epoch % config.training.log_interval == 0:
            phase = curriculum.get_phase(epoch)
            lr = optimizer.param_groups[0]["lr"]

            # ETA hesabi (son 50 epoch ortalamasi)
            recent = epoch_times[-50:]
            avg_time = sum(recent) / len(recent)
            remaining = config.training.max_epochs - epoch - 1
            eta_seconds = remaining * avg_time
            if eta_seconds >= 3600:
                eta_str = f"{eta_seconds/3600:.1f}h"
            elif eta_seconds >= 60:
                eta_str = f"{eta_seconds/60:.1f}m"
            else:
                eta_str = f"{eta_seconds:.0f}s"

            # Memory monitoring
            mem_str = ""
            if device.type == "cuda":
                peak_mb = torch.cuda.max_memory_allocated() / 1e6
                mem_str = f" mem={peak_mb:.0f}MB"
                torch.cuda.reset_peak_memory_stats()

            print(
                f"Epoch {epoch:5d} [{phase}] Re={Re:.0f} Ra={Ra:.0e} "
                f"loss={total_loss.item():.6f} lr={lr:.2e} "
                f"grad={grad_norm:.4f} [{epoch_time:.1f}s/ep ETA={eta_str}]{mem_str}"
            )
            for k, v in sorted(loss_dict.items()):
                print(f"  {k}: {v.item():.6f}")

            # Parametre izleme (her log_interval'da)
            # Laminarizasyon ve CFL kontrolu icin kritik
            with torch.no_grad():
                TKE = current.kinetic_energy().mean().item()
                dt_eff = raw_model._dt_base * torch.clamp(raw_model.dt_scale, 0.5, 2.0).item()
                u_max = max(current.u.abs().max().item(),
                            current.v.abs().max().item(),
                            current.w.abs().max().item())
                CFL = u_max * dt_eff / raw_model.dx_min

                # SGS parametreleri: Cs ve Pr_t ortalamasi
                cs_vals, prt_vals = [], []
                if raw_model.use_eddy:
                    for ev in raw_model.eddy_viscosities:
                        if hasattr(ev, 'cs_mid'):
                            cs_vals.append(ev.cs_mid.item())
                        elif hasattr(ev, 'smagorinsky_coeff'):
                            cs_vals.append(ev.smagorinsky_coeff.item())
                        if hasattr(ev, 'cs_thermal'):
                            prt_vals.append(ev.cs_thermal.item())
                        elif hasattr(ev, 'pr_t'):  # eski checkpoint uyumu
                            prt_vals.append(ev.pr_t.item())

                cs_str = f"Cs={sum(cs_vals)/len(cs_vals):.4f}" if cs_vals else "Cs=N/A"
                prt_str = f"CsT={sum(prt_vals)/len(prt_vals):.4f}" if prt_vals else "CsT=N/A"
                # Nu monitoring (loss degil, sadece izleme — damping paradoksu nedeniyle)
                # NOT: undamped PDE ciktisi. Damped LES ref (Re=10k: ~70) ile kiyaslanamaz.
                Nu_val = current.nusselt_number(
                    config.domain.Ly, config.physics.kappa
                ).mean().item()

                # v*theta flux ve theta_rms monitoring
                vT_val = (current.v * current.theta).mean().item()
                theta_rms_val = current.theta.pow(2).mean().sqrt().item()

                print(
                    f"  [PHYS] TKE={TKE:.4f} CFL={CFL:.3f} dt_eff={dt_eff:.5f} "
                    f"|u|_max={u_max:.3f} {cs_str} {prt_str} Nu={Nu_val:.2f}"
                )
                # Forcing amplitude izleme
                forcing_amp = raw_model.forcing.amplitude.item()
                print(
                    f"  [THERM] v*theta={vT_val:.6f} theta_rms={theta_rms_val:.6f}"
                    f" forcing_A={forcing_amp:.6f}"
                )

                # Alarmlar
                if TKE < 0.005:
                    print("  ⚠ WARNING: TKE < 0.005 -- laminarizasyon riski!")
                if CFL > 0.8:
                    print("  ⚠ WARNING: CFL > 0.8 -- numerik kararlilik riski!")
                if theta_rms_val < 1e-4:
                    print("  ⚠ WARNING: theta_rms < 1e-4 -- isothermal collapse!")
                if vT_val < 1e-6:
                    print("  ⚠ WARNING: v*theta < 1e-6 -- termal coupling yok!")

                # -- Early Stopping Check --
                # Sadece referans verisi olan Re icin kontrol et
                _ref_es = LES_REFERENCE.get(int(Re), None)
                if _ref_es is not None and epoch >= 100:
                    # Spectrum slope hesapla (loss'taki ile ayni)
                    _spec_es = compute_energy_spectrum(
                        current.u, current.v, current.w, raw_model.ops
                    )
                    _k_es = torch.arange(len(_spec_es), device=device, dtype=_spec_es.dtype)
                    _k_max_es = 15   # 96x160x64 grid icin (6,15)
                    _k_min_es = 6
                    _mask_es = (_k_es >= _k_min_es) & (_k_es <= _k_max_es) & (_spec_es > 1e-20)
                    if _mask_es.sum() >= 3:
                        _lk = torch.log(_k_es[_mask_es])
                        _lE = torch.log(_spec_es[_mask_es])
                        _n = _lk.shape[0]
                        slope_val = ((_n * (_lk * _lE).sum() - _lk.sum() * _lE.sum()) /
                                     (_n * (_lk**2).sum() - _lk.sum()**2 + 1e-10)).item()
                    else:
                        slope_val = -999.0  # hesaplanamadi

                    # Entropy hesapla
                    _p_es = _spec_es / (_spec_es.sum() + 1e-10)
                    entropy_val = -(_p_es * torch.log(_p_es + 1e-20)).sum().item()

                    # Convergence kontrolu
                    tke_ok = abs(TKE - _ref_es["TKE"]) / _ref_es["TKE"] < CONVERGENCE_TOL
                    slope_ok = abs(slope_val - _ref_es["slope"]) / abs(_ref_es["slope"]) < CONVERGENCE_TOL
                    entropy_ok = abs(entropy_val - _ref_es["S_ent"]) / _ref_es["S_ent"] < CONVERGENCE_TOL

                    _re_int = int(Re)
                    if tke_ok and slope_ok and entropy_ok:
                        convergence_counter[_re_int] = convergence_counter.get(_re_int, 0) + 1
                        # Tum aktif Re'ler converge etmeli (placeholder olmayanlar)
                        _phase = curriculum.get_phase(epoch)
                        _active_res = [r for r in curriculum.RE_TABLE[_phase] if r in (7000, 10000)]
                        _all_conv = all(
                            convergence_counter.get(int(r), 0) >= CONVERGENCE_WINDOW
                            for r in _active_res
                        ) if _active_res else False
                        if _all_conv:
                            print(f"\n  === EARLY STOPPING at epoch {epoch} ===")
                            for _r in _active_res:
                                _rr = LES_REFERENCE[_r]
                                print(f"  Re={_r}: streak={convergence_counter.get(_r, 0)}/{CONVERGENCE_WINDOW}")
                            loss_snapshot = {k: v.item() for k, v in loss_dict.items()}
                            save_checkpoint(model, optimizer, epoch, config, loss_snapshot, scheduler=scheduler)
                            print("\nTraining completed (early stopping).")
                            return model
                    else:
                        convergence_counter[_re_int] = 0

                    if epoch % config.training.log_interval == 0:
                        _re_int = int(Re)
                        print(
                            f"  [CONV] Re={_re_int} TKE={'OK' if tke_ok else 'X'} "
                            f"slope={'OK' if slope_ok else 'X'} "
                            f"entropy={'OK' if entropy_ok else 'X'} "
                            f"streak={convergence_counter.get(_re_int, 0)}/{CONVERGENCE_WINDOW}"
                        )

        # 8. Checkpoint
        # Bug 2 fix (2026-05-01): max_epochs orantılı kayıt + latest copy (resume için)
        ckpt_every = max(1, config.training.max_epochs // 10)  # 100 ep → her 10 ep
        legacy_milestones = (300, 600, 1000, 1250, 1500)
        if epoch > 0 and (epoch % ckpt_every == 0 or epoch in legacy_milestones):
            loss_snapshot = {k: float(v.item()) for k, v in loss_dict.items()
                             if torch.isfinite(v)}
            save_checkpoint(model, optimizer, epoch, config, loss_snapshot, scheduler=scheduler)
            # latest copy — crash sonrası resume için
            try:
                import shutil
                src = Path(config.training.checkpoint_dir) / f"checkpoint_epoch{epoch:06d}.pt"
                dst = Path(config.training.checkpoint_dir) / "checkpoint_latest.pt"
                if src.exists():
                    shutil.copy2(src, dst)
            except Exception as _e:
                print(f"  [WARN] latest copy fail: {_e}")

    # Son checkpoint (max_epochs - 1 = son epoch numarasi)
    if config.training.max_epochs > 0:
        loss_snapshot = {k: v.item() for k, v in loss_dict.items()}
        save_checkpoint(
            model, optimizer, config.training.max_epochs - 1, config, loss_snapshot, scheduler=scheduler
        )

    print("\nTraining completed.")
    return model


# =====================================================================
# 7. Entry Point
# =====================================================================


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--max-epochs", type=int, default=None, help="Override max epochs")
    parser.add_argument("--log-interval", type=int, default=None, help="Override log interval")
    parser.add_argument("--dt", type=float, default=None, help="Override physics.dt (numerical step)")
    parser.add_argument("--num-steps", type=int, default=None, help="Override training.num_steps (rollout length)")
    parser.add_argument("--proportional-curriculum", action="store_true",
                        help="Curriculum boundaries'i max_epochs'a proportional scale et "
                             "(Phase A=17%%, B=34%%, C=67%%, D=100%%)")
    parser.add_argument("--freeze-forcing", action="store_true",
                        help="forcing.amplitude'i 0.005 sabit tut (Goodhart fix, Eswaran-Pope DNS)")
    parser.add_argument("--use-spectral-cs", action="store_true",
                        help="Saf-INNATE: MLP-SGS yerine SpectralCsField (Fourier mod katsayilari learnable)")
    args = parser.parse_args()

    config = Config()
    if args.max_epochs is not None:
        config.training.max_epochs = args.max_epochs
    if args.log_interval is not None:
        config.training.log_interval = args.log_interval
    if args.dt is not None:
        config.physics.dt = args.dt
    if args.num_steps is not None:
        config.training.num_steps = args.num_steps
    if args.freeze_forcing:
        config.training.freeze_forcing = True
    if args.use_spectral_cs:
        config.model.use_spectral_cs = True
        config.model.use_mlp_sgs = False
        print("[CONFIG] Saf-INNATE Spectral-Cs ENABLED (MLP-SGS bypass)")
    if args.proportional_curriculum:
        CurriculumScheduler.set_phase_boundaries(config.training.max_epochs)
        print(f"[curriculum] Proportional boundaries -> {CurriculumScheduler.PHASE_BOUNDARIES}")

    train(config=config, resume_from=args.resume)
