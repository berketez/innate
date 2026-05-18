"""
loss_scales.py — Karakteristik fizik ölçekleri (boyutsuzlaştırma)
=================================================================

INNATE Mixed Convection projesinin **en kritik dosyası**.

Sorun (cross-check raporu, 2026-04-25):
  PhysicsLoss içindeki 7+ farklı loss terimi farklı **boyut sınıflarında**
  residual üretiyor ve ham toplanıyordu. Örnek:
      L_divergence:      [1/T]²
      L_energy_balance:  [U²/T]
      L_dissipation:     [1/T]
      L_v_theta:         [1] (log-MSE)
      L_cfl_guard:       [1]
  Bu durumda `total_loss = Σ wᵢ·Lᵢ` toplamında bazı terimlerin doğal
  scale'i ~1e-6, bazılarının ~1e3. Optimizer pratik olarak hangi
  fizik daha "zorlu" (büyük gradient) onu önceleyemediği için takıldı —
  63 saatlik eğitimin Nu=1.0'a sıkışmasının kök sebebi.

Çözüm:
  Her loss return etmeden ÖNCE bu modülün karakteristik scale'ine
  bölünür → **boyutsuz residual**. Toplam loss artık tüm bileşenleri
  eşit ağırlıklı görür; weight'ler gerçekten "fizik önceliği" demektir.

Konvansiyon
-----------
Tüm ölçekler **boyutsuz iç-ölçek** birimindedir (U_ref = L_ref = 1).
Yani değerler O(1) civarındadır; küçük epsilon ile bölme güvenli.

Kullanım
--------
    from loss_scales import (
        DIV_SCALE, ENERGY_BALANCE_SCALE, DISSIPATION_SCALE,
        ENSTROPHY_SCALE, NU_LOG_SCALE, FLUX_LOG_SCALE,
        CFL_SCALE, scale_loss
    )

    # Ham residual hesapla
    raw = (div ** 2).mean()           # [1/T]²

    # Sonra ölçekle:
    return scale_loss(raw, DIV_SCALE)  # → boyutsuz O(1) residual
"""
from __future__ import annotations

import math
from typing import Optional

import torch


# =============================================================================
# 1. KARAKTERİSTİK ÖLÇEKLER (boyutsuz iç-konvansiyon)
# =============================================================================
#
# U_REF = 1.0  (referans hız — k_f mod forcing ile eşlenmiş O(1) hız)
# L_REF = 1.0  (Ly/Ly = 1; tüm uzunluklar Ly ile boyutsuzlaştırılmış)
# T_REF = L_REF / U_REF = 1.0
#
# Türev ölçekler:
#   E_REF      ~ U²       = 1.0    (kinetik enerji yoğunluğu)
#   EPS_REF    ~ U³/L     = 1.0    (dissipation rate)
#   ENSTR_REF  ~ U²/L²    = 1.0    (enstrophy)
#   DIVU_REF   ~ U/L      = 1.0    (∇·u)
#
# Nondim sistemde hepsi 1.0; ama fiziksel değişkenlerin ortalama
# büyüklükleri (LES referansından kalibre):
#   <u²> ~ 0.008-0.017       (Re=7K-10K LES)
#   <ω²> ~ 50-100            (LES enstrophy)
#   <(∇·u)²> ~ 1e-6 (Leray)
#   ν·<ω²> ~ 0.007-0.014     (dissipation rate)
#
# Bu sayede her loss'un "doğal mertebesi"ni kullanırız.

# Velocity / length / time references (boyutsuz, sabit)
U_REF: float = 1.0
L_REF: float = 1.0
T_REF: float = L_REF / U_REF

# Energy / dissipation
E_REF: float = U_REF ** 2                         # kinetik enerji
EPS_REF: float = U_REF ** 3 / L_REF               # dissipation rate

# =============================================================================
# 2. LOSS-SPECIFIC SCALES
# =============================================================================
#
# Her loss'un **doğal mertebesi** (LES referans + analitik tahmin).
# Loss'un kendisi residual² olduğu için scale = (doğal_residual)².
# Bu sayede bölmeden sonra ~O(1).
# Eğer loss zaten log-space MSE ise scale=1.0 (zaten boyutsuz O(1)).

# Divergence: (∇·u)² ~ (1e-3)² = 1e-6 (Leray sonrası tipik)
# Açıklama: spectral projection sonrası div ~1e-3 mertebesinde,
#           kareli ortalama ~1e-6.
DIV_SCALE: float = 1.0e-6

# Energy balance: |dE/dt + eps - P_f - P_b| ~ eps_SGS mertebesinde,
# tipik 1e-3 nondim (LES kapatma artığı).
ENERGY_BALANCE_SCALE: float = 1.0e-3

# Dissipation: |eps_spectral - nu*Z| ~ 1e-3
DISSIPATION_SCALE: float = 1.0e-3

# Enstrophy / stability — Z_ratio kareli, scale = 1.0 (relu→0 normalde)
ENSTROPHY_SCALE: float = 1.0

# Spectrum slope: (slope - (-5/3))² ~ O(1) (slope ±0.5 sapma → 0.25)
SLOPE_SCALE: float = 1.0

# Log-space loss'lar zaten boyutsuz, scale = 1.0
NU_LOG_SCALE: float = 1.0
FLUX_LOG_SCALE: float = 1.0
TKE_LOG_SCALE: float = 1.0
THETA_RMS_LOG_SCALE: float = 1.0
CORR_SCALE: float = 1.0
ENTROPY_SCALE: float = 1.0
SPECTRUM_SHAPE_SCALE: float = 50.0  # 2026-05-01: 1.0 → 50, smoke L_spec=223 fazla; hedef O(5)

# CFL guard: relu(CFL - thresh)² ~ O(1e-3) tipik training
CFL_SCALE: float = 1.0e-2

# Thermal variance: relu(var - 100)², zaten guard rail
THERMAL_VAR_SCALE: float = 1.0

# theta_min: relu(0.005 - rms)², ~O(1e-5)
THETA_MIN_SCALE: float = 1.0e-5

# Phase D (non-Boussinesq): yoğunluk korunum residual² ~ 1e-3
CONTINUITY_RHO_SCALE: float = 1.0e-3
STATE_EQ_SCALE: float = 1.0e-3
MASS_SCALE: float = 1.0e-3

# Germano: residual² mertebesi LES'te ~1e-2
GERMANO_SCALE: float = 1.0e-2

# Anti-laminarization: relu² guard rail, ~1.0
ANTI_LAMINAR_SCALE: float = 1.0

# === Tier 2 (2026-04-29): PDE Residual loss scales ===
# NS residual: |(u_{n+1}-u_n)/dt - RHS_canonical|²
# RHS = -(u·∇)u + ν∇²u + Ri·θ·ê_y + F - ∇p (kanonik DNS form, ν_t YOK)
# Mertebe: ν_t YOK varsayımı LES'te SGS-typical ν_t·∇²u ~ 1e-2 yaratır;
# residual² ~ 1e-4. Smoke test (Tier 2): raw residual² ~ 1.5e-3 (Re=10K).
# Scale 1e-4 → loss değeri O(10) bandı, gradient pathology yok.
NS_RES_SCALE: float = 1.0e-3  # 2026-05-01: 1e-4 → 1e-3, smoke L_NS=137 fazla; hedef O(15)

# Thermal residual: |(θ_{n+1}-θ_n)/dt - RHS_θ|²
# RHS_θ = -(u·∇)θ + κ∇²θ + v/Ly (mean-grad source)
# Smoke test: ham residual² ~ 3e-3 (Re=10K). Scale 1e-4 → loss O(30).
TH_RES_SCALE: float = 1.0e-4


# =============================================================================
# 3. UTILITY: scale_loss
# =============================================================================

def scale_loss(
    raw_loss: torch.Tensor,
    scale: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    Ham loss'u karakteristik scale'e bölerek boyutsuzlaştır.

    Parametreler
    -----------
    raw_loss : torch.Tensor
        Ham residual loss (kareli veya mutlak; boyutlu).
    scale : float
        Bu loss tipinin doğal mertebesi (genellikle ~residual²).
    eps : float
        Sıfıra bölmeyi önler.

    Dönüş
    -----
    torch.Tensor: ~O(1) mertebesinde boyutsuz residual.

    Örnek
    -----
    >>> raw = (div ** 2).mean()             # ~1e-6
    >>> scaled = scale_loss(raw, DIV_SCALE) # ~O(1)
    """
    return raw_loss / (scale + eps)


# =============================================================================
# 4. INSPECTION: ağırlık dengesini kontrol etme
# =============================================================================

def report_scales() -> str:
    """Tüm tanımlı scale'leri tablolu döndür (debug için)."""
    rows = [
        ("U_REF", U_REF, "referans hız (boyutsuz)"),
        ("L_REF", L_REF, "referans uzunluk = Ly/Ly"),
        ("T_REF", T_REF, "referans zaman"),
        ("E_REF", E_REF, "kinetik enerji ölçeği U²"),
        ("EPS_REF", EPS_REF, "dissipation U³/L"),
        ("DIV_SCALE", DIV_SCALE, "(∇·u)² doğal mertebe"),
        ("ENERGY_BALANCE_SCALE", ENERGY_BALANCE_SCALE, "TKE bütçe residualı"),
        ("DISSIPATION_SCALE", DISSIPATION_SCALE, "spektral-fiziksel dissipation farkı"),
        ("ENSTROPHY_SCALE", ENSTROPHY_SCALE, "stability guard"),
        ("SLOPE_SCALE", SLOPE_SCALE, "spectrum slope MSE"),
        ("NU_LOG_SCALE", NU_LOG_SCALE, "log-space MSE (zaten ~O(1))"),
        ("CFL_SCALE", CFL_SCALE, "CFL relu² guard"),
        ("THETA_MIN_SCALE", THETA_MIN_SCALE, "isothermal collapse guard"),
        ("THERMAL_VAR_SCALE", THERMAL_VAR_SCALE, "theta varyans guard"),
        ("GERMANO_SCALE", GERMANO_SCALE, "Germano identity residual²"),
    ]
    out = "Loss Scales (loss_scales.py):\n"
    out += f"{'NAME':<28} {'VALUE':>12}    DESCRIPTION\n"
    out += "-" * 80 + "\n"
    for name, val, desc in rows:
        out += f"{name:<28} {val:>12.3e}    {desc}\n"
    return out


if __name__ == "__main__":
    print(report_scales())
