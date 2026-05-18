#!/usr/bin/env python3
"""
demo4_limit_problems.py - INNATE 3D Limit Problemleri Testi

Egitilmis modeli tamamen farkli fizik problemlerinde test eder.
Mixed convection icin egitilmis SGS parametreleri ne kadar evrensel?

Test vakalari:

A) Saf Rayleigh-Benard Konveksiyonu
   - Forcing YOK, sadece buoyancy
   - Altta sicak, ustte soguk plaka → konveksiyon hucreleri
   - Formul:
       du/dt = -(u.nabla)u - nabla_p + nu * nabla^2 u + Ri * theta * e_y
       dtheta/dt = -(u.nabla)theta + kappa * nabla^2 theta
       nabla . u = 0
       F_external = 0
   - Beklenti: Nu > 1 (konveksiyon olmasi gerektigi icin)

B) Saf Kolmogorov Akisi
   - Buoyancy YOK, sadece sinuzoidal forcing
   - Klasik homogeneous turbulence testi
   - Formul:
       du/dt = -(u.nabla)u - nabla_p + nu * nabla^2 u + F_kolmogorov
       dtheta/dt = -(u.nabla)theta + kappa * nabla^2 theta
       nabla . u = 0
       F_kolmogorov = A * sin(k_f * 2*pi*y/Ly) * e_x
       Ri = 0 (buoyancy deaktif)
   - Beklenti: E(k) ~ k^{-5/3} enerji spektrumu, enstrophy cascade

C) Taylor-Green Vortex 3D Decay
   - Forcing YOK, buoyancy YOK, sadece serbest curume
   - Analitik IC'den baslayan klasik benchmark
   - Formul:
       du/dt = -(u.nabla)u - nabla_p + nu * nabla^2 u
       w = 0, theta = 0
       nabla . u = 0
   - IC (anisotropik domain'e uyumlu TGV):
       kx = 2*pi/Lx, ky = 2*pi/Ly, kz = 2*pi/Lz
       u =  A * sin(kx*x) * cos(ky*y) * cos(kz*z)
       v = -A * (kx/ky) * cos(kx*x) * sin(ky*y) * cos(kz*z)
       w = 0, theta = 0
       p = projection'dan (Poisson cozumu)
     Not: v amplitudu (kx/ky) = (Ly/Lx) ile olceklenir → div-free
   - Beklenti: E(t) monoton azalir, max enstrophy t ~ 9 (Re=1600)

Kullanim:
  python tests/demo4_limit_problems.py --checkpoint results/checkpoints/best.pt
  python tests/demo4_limit_problems.py --checkpoint best.pt --steps 300 --device cpu

Yazar: Berke Tezgocen (tasarim), Claude (implementasyon)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# -- path setup --
_this_dir = Path(__file__).resolve().parent
_project_dir = _this_dir.parent
sys.path.insert(0, str(_project_dir.parent))
sys.path.insert(0, str(_project_dir))

from config import Config
from model import INNATE3D_MixedConvection, ThermalFluidState

from demo_utils import (
    SimulationResult,
    load_trained_model,
    run_simulation,
    compute_physics_metrics,
    compute_energy_spectrum,
    compute_spectrum_slope,
    save_results,
    print_comparison_table,
    print_final_report,
)


# =====================================================================
# IC Olusturma Fonksiyonlari
# =====================================================================


def create_rayleigh_benard_ic(
    model: INNATE3D_MixedConvection,
    config: Config,
    device: torch.device,
) -> ThermalFluidState:
    """
    Rayleigh-Benard konveksiyonu icin ilk kosul.

    Sicaklik: kucuk random perturbasyonlar (konveksiyonu tetiklemek icin).
    Hiz: sifira yakin (baslangicta durgun akiskan).

    Fizik: Rayleigh-Benard'da konveksiyon, kritik Ra sayisi asildiginda
    baslar. Ra > Ra_c ~ 1708 (2D sonsuz plakalar arasi) icin
    konveksiyon hucreleri olusur. 3D ve periyodik BC'de Ra_c farkli
    olabilir ama Ra=1e7 ile kesinlikle turbulant konveksiyon beklenir.

    Perturbasyonlar:
      - Hiz: cok kucuk noise (1e-3 olcek) → doganin perturbasyonlarini taklit
      - Sicaklik: daha buyuk noise (0.5 olcek) → konveksiyonu hizli tetikler
      - Sicaklik perturbasyonu mean=0 (Boussinesq tutarliligi)
    """
    d = config.domain
    shape = (1, d.Nx, d.Ny, d.Nz)

    # Hiz: cok kucuk perturbasyonlar
    # (tamamen sifir baslatirsan konveksiyon baslamaz — simetri kirilmaz)
    noise_scale = 1e-3
    u = noise_scale * torch.randn(shape, device=device)
    v = noise_scale * torch.randn(shape, device=device)
    w = noise_scale * torch.randn(shape, device=device)
    p = torch.zeros(shape, device=device)

    # Sicaklik perturbasyonu: orta katmanda daha guclu (konveksiyonu hizla tetikler)
    # y-yonunde sin profili + random noise
    y = torch.linspace(0, d.Ly, d.Ny, device=device).view(1, 1, d.Ny, 1)
    # Orta bolgede buyuk, kenarlarda kucuk perturbasyonlar
    y_envelope = torch.sin(math.pi * y / d.Ly)  # [1, 1, Ny, 1]
    theta = 0.5 * y_envelope * torch.randn(shape, device=device)
    # Mean removal (Boussinesq)
    theta = theta - theta.mean(dim=(-3, -2, -1), keepdim=True)

    # Divergence-free projection
    u, v, w, p = model.projections[0](u, v, w)

    return ThermalFluidState(
        u=u, v=v, w=w, p=p, theta=theta,
        t=torch.zeros(1, device=device),
    )


def create_kolmogorov_ic(
    model: INNATE3D_MixedConvection,
    config: Config,
    device: torch.device,
) -> ThermalFluidState:
    """
    Kolmogorov akisi icin ilk kosul.

    Hiz: Kolmogorov forcing'in analitik steady-state cozumune yakin bir
    baslangic profili + kucuk perturbasyonlar (turbulansi tetiklemek icin).

    Kolmogorov steady-state (laminar):
      u_ss = (A / (nu * k_f^2)) * sin(k_f * 2*pi*y/Ly)
      v_ss = 0, w_ss = 0
    Ama Re >> Re_c ise bu cozum INSTABIL → turbulans olusur.

    Sicaklik: sifir (buoyancy kapali, theta pasif skaler olarak advekte olur).
    """
    d = config.domain
    p_cfg = config.physics
    shape = (1, d.Nx, d.Ny, d.Nz)

    # y koordinati
    y = torch.linspace(0, d.Ly, d.Ny, device=device).view(1, 1, d.Ny, 1)

    # Kolmogorov base profili
    # A ~ model.forcing.amplitude (checkpoint'tan gelen deger)
    # Ama forcing amplitude'u bilmiyoruz, basit bir sinuzoidal profil kullanalim
    k_f = p_cfg.k_f
    u_base = 0.1 * torch.sin(k_f * 2 * math.pi * y / d.Ly)
    u = u_base.expand(shape).clone()

    # Perturbasyonlar (turbulans tetikleyici)
    u = u + 0.01 * torch.randn(shape, device=device)
    v = 0.01 * torch.randn(shape, device=device)
    w = 0.01 * torch.randn(shape, device=device)
    p = torch.zeros(shape, device=device)

    # Sicaklik: sifir (buoyancy kapali, pasif skaler)
    theta = torch.zeros(shape, device=device)

    # Divergence-free projection
    u, v, w, p = model.projections[0](u, v, w)

    return ThermalFluidState(
        u=u, v=v, w=w, p=p, theta=theta,
        t=torch.zeros(1, device=device),
    )


def create_tgv_ic(
    model: INNATE3D_MixedConvection,
    config: Config,
    device: torch.device,
) -> ThermalFluidState:
    """
    Anisotropik domain icin Taylor-Green Vortex 3D ilk kosul.

    Klasik TGV [0, 2*pi]^3 izotropik domain'de tanimlanir:
      u =  sin(x) * cos(y) * cos(z)
      v = -cos(x) * sin(y) * cos(z)

    PROBLEM: SpectralOps3DAniso wavenumber'lari domain boyutlarina gore
    hesaplar: kx = 2*pi/Lx, ky = 2*pi/Ly, kz = 2*pi/Lz.
    IC koordinatlari [0, 2*pi) uzerinde sin(x) seklinde verilirse,
    fiziksel wavenumber 1.0 olur; ama SpectralOps k=1 modunu 2*pi/Lx
    olarak gorur. Bu uyumsuzluk spectral turevlerde ciddi hata yaratir.

    COZUM: IC'yi domain periyoduna uyumlu yapalim.
    Koordinatlar [0, Lx) x [0, Ly) x [0, Lz), wavenumber'lar:
      kx = 2*pi/Lx,  ky = 2*pi/Ly,  kz = 2*pi/Lz

    Divergence-free kosulu (w=0 icin):
      div = kx*A*cos(kx*x)*cos(ky*y)*cos(kz*z)
          + ky*B*cos(kx*x)*cos(ky*y)*cos(kz*z) = 0
      => B = -A * kx/ky = -A * Ly/Lx

    Bu nedenle:
      u =  A * sin(kx*x) * cos(ky*y) * cos(kz*z)
      v = -A * (kx/ky) * cos(kx*x) * sin(ky*y) * cos(kz*z)
      w = 0

    Analitik div-free: du/dx + dv/dy
      = A*kx*cos(kx*x)*cos(ky*y)*cos(kz*z)
        - A*(kx/ky)*ky*cos(kx*x)*cos(ky*y)*cos(kz*z) = 0  [tamam]

    Kinetik enerji: E_kin(0) = 0.5 * <u^2 + v^2>
      = 0.5 * A^2/4 * (1 + (kx/ky)^2) = A^2/8 * (1 + (Ly/Lx)^2)
    Default domain (Lx=6, Ly=10): E_kin(0) = 1/8 * (1 + 25/9) = 1/8 * 34/9 ~ 0.4722

    Basinc: p = 0 ile baslayip projection'dan hesaplatiyoruz.
    (Anisotropik domain'de analitik TGV basinci farkli form alir,
     projection zaten dogru Poisson cozumu verir.)

    theta = 0 (termal etkiler yok).
    """
    d = config.domain
    shape = (1, d.Nx, d.Ny, d.Nz)

    # Domain periyoduna uyumlu wavenumber'lar
    kx = 2 * math.pi / d.Lx
    ky = 2 * math.pi / d.Ly
    kz = 2 * math.pi / d.Lz

    # Koordinatlar: [0, L) periyodik (endpoint=False)
    x = torch.linspace(0, d.Lx, d.Nx + 1, device=device)[:-1]
    y = torch.linspace(0, d.Ly, d.Ny + 1, device=device)[:-1]
    z = torch.linspace(0, d.Lz, d.Nz + 1, device=device)[:-1]

    # 3D meshgrid: [Nx, Ny, Nz] -> [1, Nx, Ny, Nz]
    x = x.view(1, d.Nx, 1, 1)
    y = y.view(1, 1, d.Ny, 1)
    z = z.view(1, 1, 1, d.Nz)

    # Anisotropik TGV IC (div-free: B = -A * kx/ky)
    A = 1.0
    u = A * torch.sin(kx * x) * torch.cos(ky * y) * torch.cos(kz * z)
    v = -A * (kx / ky) * torch.cos(kx * x) * torch.sin(ky * y) * torch.cos(kz * z)
    w = torch.zeros(shape, device=device)

    # Basinc: projection'dan hesaplanacak (p=0 baslangic)
    p = torch.zeros(shape, device=device)

    # Sicaklik: sifir (TGV'de termal efekt yok)
    theta = torch.zeros(shape, device=device)

    # Divergence-free dogrulama (sayisal garanti icin projection)
    # Analitik olarak zaten div-free ama sayisal yuvarlama hatalari
    # icin projection ile temizliyoruz.
    u, v, w, p_proj = model.projections[0](u, v, w)

    # Analitik E_kin hesabi (dogrulama icin)
    E_kin_analytic = A**2 / 8.0 * (1.0 + (kx / ky) ** 2)
    E_kin_numeric = 0.5 * (u**2 + v**2 + w**2).mean().item()
    print(f"  TGV IC: A={A}, kx={kx:.4f}, ky={ky:.4f}, kz={kz:.4f}")
    print(f"  TGV IC: kx/ky = {kx/ky:.4f} (= Ly/Lx = {d.Ly/d.Lx:.4f})")
    print(f"  TGV IC: E_kin analitik = {E_kin_analytic:.6f}, sayisal = {E_kin_numeric:.6f}")

    return ThermalFluidState(
        u=u, v=v, w=w, p=p_proj, theta=theta,
        t=torch.zeros(1, device=device),
    )


# =====================================================================
# Forcing / Buoyancy Sifirlama
# =====================================================================


def disable_forcing(model: INNATE3D_MixedConvection):
    """
    Forcing terimini deaktif et.

    Forcing3D.forward() icinde amplitude.clamp(1e-5, 0.01) var,
    yani amplitude=0 yapsan bile 1e-5'e clamp olur.
    Bu ihmal edilebilir kucuklukte (~1e-5 vs tipik hiz ~0.1-1.0),
    ama tam sifir degil.

    Ek olarak harmonik amplitude'leri de sifirlayalim.
    """
    with torch.no_grad():
        # Ana amplitude'u sifirla
        model.forcing.amplitude.fill_(0.0)
        # Harmonikleri sifirla (varsa)
        if hasattr(model.forcing, "amplitude_k2"):
            model.forcing.amplitude_k2.fill_(0.0)
        if hasattr(model.forcing, "amplitude_k3"):
            model.forcing.amplitude_k3.fill_(0.0)

    print("  Forcing deaktif edildi (amplitude → 0, harmonikler → 0)")
    print(f"  NOT: Forward'da clamp(1e-5, 0.01) nedeniyle residual ~1e-5 olacak.")


def disable_buoyancy(model: INNATE3D_MixedConvection):
    """
    Buoyancy terimini deaktif et.

    Ra=0 → Ri=0 → Buoyancy3D.forward() icinde:
      Fy = Ri * strength * theta = 0 * strength * theta = 0
    Otomatik kapanir, ekstra islem gerekmez.
    Sadece dogrulayalim.
    """
    # set_physics ile Ri=0 yapildiysa buoyancy otomatik kapali
    print(f"  Buoyancy durumu: Ri = {model.Ri:.6f}")
    if abs(model.Ri) < 1e-10:
        print("  Buoyancy etkin sekilde KAPALI (Ri ~ 0)")
    else:
        print(f"  UYARI: Ri = {model.Ri:.6f} > 0, buoyancy hala aktif!")


# =====================================================================
# Post-Step Fonksiyonlari
# =====================================================================


def post_step_zero_forcing(
    state: ThermalFluidState, step: int, model: INNATE3D_MixedConvection
) -> ThermalFluidState:
    """
    Her adimdan sonra forcing etkisini cikar.

    Forcing3D'nin clamp(1e-5, ...) nedeniyle tam sifir olamamasi sorununu
    gidermek icin, model icindeki forcing ciktisini hesaplayip state'den
    cikarabiliriz.

    ANCAK bu karmasik ve model'in internal dt'sine bagimli.
    Pratikte 1e-5'lik forcing ihmal edilebilir (hiz ~0.1-1.0 oldugunda
    etki ~0.001% seviyesinde).

    Bu fonksiyon sadece theta'nin mean=0 kalmasini saglar (gauge fix).
    """
    # theta gauge fix (mean removal — model zaten yapiyor ama garanti olsun)
    state.theta = state.theta - state.theta.mean(dim=(-3, -2, -1), keepdim=True)
    return state


def post_step_tgv_no_thermal(
    state: ThermalFluidState, step: int, model: INNATE3D_MixedConvection
) -> ThermalFluidState:
    """
    TGV3D decay icin: theta'yi sifir tut.

    TGV'de termal efekt yok (Ra=0, forcing=0). theta advection denklemi
    hala cozuluyor ama theta=0'dan basladigi icin adv(0)=0, diff(0)=0
    → theta=0 kalmali. Sayisal hata birikimini onlemek icin her adimda
    sifirlayalim.
    """
    state.theta = torch.zeros_like(state.theta)
    return state


# =====================================================================
# Test Calistirma Fonksiyonlari
# =====================================================================


def run_rayleigh_benard(
    checkpoint_path: str,
    n_steps: int,
    device: str,
    log_interval: int,
) -> SimulationResult:
    """
    Test A: Saf Rayleigh-Benard Konveksiyonu.

    Fizik:
      - Forcing YOK → sadece buoyancy surukler
      - Ra = 1e7 → guclu konveksiyon (turbulant rejim)
      - Re = 5000 → nu = 2e-4 (viscosity)
      - Ri = Ra / (Re^2 * Pr) = 1e7 / (2.5e7 * 0.71) = 0.563

    Beklentiler:
      - Konveksiyon hucreleri olusur (yukselen sicak, alcan soguk parcaciklar)
      - Nu > 1 (konvektif isi transferi, saf iletimden fazla)
      - E_kin > 0 (akis var, durgun degil)
      - Ra >> Ra_c ~ 1708 oldugundan turbulant konveksiyon beklenir
    """
    print("\n" + "=" * 80)
    print("  TEST A: SAF RAYLEIGH-BENARD KONVEKSIYONU")
    print("  Forcing = 0, Ra = 1e7, sadece buoyancy")
    print("=" * 80)

    # Model yukle: Ra=1e7 (guclu konveksiyon), Re=5000
    model, cfg, dev = load_trained_model(
        checkpoint_path=checkpoint_path,
        Re=5000.0,
        Ra=1e7,  # Yuksek Ra → turbulant konveksiyon
        device=device,
    )

    # Forcing'i deaktif et
    disable_forcing(model)
    # Buoyancy aktif (Ri > 0)
    disable_buoyancy(model)  # sadece durumu yazdirir

    # IC: rayleigh-benard'a ozgu
    ic = create_rayleigh_benard_ic(model, cfg, dev)

    # E_kin(0) kontrol
    E0 = ic.kinetic_energy().item()
    print(f"  IC E_kin = {E0:.6f} (kucuk perturbasyonlardan)")
    print(f"  Ri = {model.Ri:.4f} (buoyancy gucu)")

    # Simulasyon
    result = run_simulation(
        model=model,
        config=cfg,
        device=dev,
        n_steps=n_steps,
        log_interval=log_interval,
        name="Rayleigh-Benard (Ra=1e7)",
        ic_state=ic,
        post_step_fn=post_step_zero_forcing,
    )

    # Fiziksel dogrulama
    m = result.final_metrics
    if m:
        print("\n  FIZIKSEL DOGRULAMA:")
        Nu = m.get("Nu", 0)
        E = m.get("E_kin", 0)
        print(f"    Nu = {Nu:.4f} {'(PASS: konveksiyon var)' if Nu > 1.0 else '(FAIL: Nu <= 1)'}")
        print(f"    E_kin = {E:.6f} {'(PASS: akis var)' if E > 1e-6 else '(FAIL: durgun)'}")

    # Memory temizle
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return result


def run_kolmogorov(
    checkpoint_path: str,
    n_steps: int,
    device: str,
    log_interval: int,
) -> SimulationResult:
    """
    Test B: Saf Kolmogorov Akisi.

    Fizik:
      - Buoyancy YOK → Ra = 0, Ri = 0
      - Forcing ACIK → Kolmogorov sinuzoidal forcing
      - Re = 5000 → turbulant rejim (Re_c ~ 40 Kolmogorov akisi icin)

    Beklentiler:
      - Kolmogorov -5/3 enerji spektrumu (inertial range)
      - E_kin doygunluga ulasir (forcing-dissipation dengesi)
      - Enstrophy cascade (3D'de)
      - theta pasif skaler olarak advekte olur (buoyancy feedback yok)
    """
    print("\n" + "=" * 80)
    print("  TEST B: SAF KOLMOGOROV AKISI")
    print("  Buoyancy = 0, Ra = 0, sadece Kolmogorov forcing")
    print("=" * 80)

    # Model yukle: Ra=0 → Ri=0 → buoyancy kapali
    model, cfg, dev = load_trained_model(
        checkpoint_path=checkpoint_path,
        Re=5000.0,
        Ra=0.0,  # Buoyancy KAPALI
        device=device,
    )

    # Buoyancy durumu dogrula (Ri=0 olmali)
    disable_buoyancy(model)  # sadece durumu yazdirir

    # Forcing ACIK (checkpoint'tan gelen degerler kullanilir)
    print(f"  Forcing modu: {model.forcing.mode}")
    print(f"  Forcing amplitude: {model.forcing.amplitude.item():.6f}")

    # IC: kolmogorov'a ozgu
    ic = create_kolmogorov_ic(model, cfg, dev)

    # E_kin(0) kontrol
    E0 = ic.kinetic_energy().item()
    print(f"  IC E_kin = {E0:.6f}")

    # Simulasyon
    result = run_simulation(
        model=model,
        config=cfg,
        device=dev,
        n_steps=n_steps,
        log_interval=log_interval,
        name="Kolmogorov (Ra=0, forcing ON)",
        ic_state=ic,
    )

    # Fiziksel dogrulama
    m = result.final_metrics
    if m:
        print("\n  FIZIKSEL DOGRULAMA:")
        slope = m.get("spectrum_slope", float("nan"))
        E = m.get("E_kin", 0)
        Z = m.get("Z_enstrophy", 0)
        print(f"    E_kin = {E:.6f} {'(PASS: enerji var)' if E > 1e-6 else '(FAIL: sondurulmus)'}")
        if slope == slope:  # NaN check
            # Kolmogorov -5/3 = -1.667
            slope_ok = -2.5 < slope < -1.0
            print(f"    Spektral egim = {slope:.3f} (hedef: -1.667) "
                  f"{'(KABUL EDILEBILIR)' if slope_ok else '(BEKLENENDEN FARKLI)'}")
        else:
            print(f"    Spektral egim = N/A")
        print(f"    Enstrophy = {Z:.4f}")

    # Memory temizle
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return result


def run_tgv3d(
    checkpoint_path: str,
    n_steps: int,
    device: str,
    log_interval: int,
) -> SimulationResult:
    """
    Test C: Taylor-Green Vortex 3D Decay.

    Fizik:
      - Forcing YOK, buoyancy YOK
      - Analitik IC'den serbest curume (decay)
      - Re = 1600 (klasik TGV benchmark)

    Beklentiler:
      - E_kin(t) monoton azalir
      - Enstrophy onceleri artar (vortex stretching), sonra azalir
      - Max enstrophy t ~ 9 civarinda (Re=1600 icin)
      - Turbulanstan sonra viscous decay hakim
      - theta = 0 kalmali (termal efekt yok)

    Anisotropik domain uyumu:
      IC wavenumber'lari domain periyoduna uyumlu:
        kx = 2*pi/Lx, ky = 2*pi/Ly, kz = 2*pi/Lz
      Divergence-free kosulu icin v amplitudu olceklenir:
        v = -A * (kx/ky) * cos(kx*x) * sin(ky*y) * cos(kz*z)
      Bu sayede SpectralOps3DAniso turevleri dogru hesaplar.
      E_kin(0) = A^2/8 * (1 + (Ly/Lx)^2), domain geometrisine bagli.
    """
    print("\n" + "=" * 80)
    print("  TEST C: TAYLOR-GREEN VORTEX 3D DECAY")
    print("  Forcing = 0, Buoyancy = 0, Re = 1600")
    print("  Analitik IC → serbest curume")
    print("=" * 80)

    # Anisotropik domain (Lx x Ly x Lz) uzerinde TGV.
    # IC wavenumber'lari domain periyoduna uyumlu: kx=2*pi/Lx, ky=2*pi/Ly, kz=2*pi/Lz
    # Divergence-free: v = -A*(kx/ky)*cos(kx*x)*sin(ky*y)*cos(kz*z)
    # Izotropik TGV ile ayni fizik, sadece domain olcekleme farki.

    # Model yukle: Re=1600 (klasik TGV benchmark), Ra=0 (buoyancy yok)
    model, cfg, dev = load_trained_model(
        checkpoint_path=checkpoint_path,
        Re=1600.0,     # Klasik TGV benchmark Re
        Ra=0.0,        # Buoyancy KAPALI
        device=device,
    )

    # Forcing'i deaktif et
    disable_forcing(model)
    # Buoyancy durumu dogrula
    disable_buoyancy(model)

    # IC: TGV analitik
    ic = create_tgv_ic(model, cfg, dev)

    # IC dogrulamasi
    E0 = ic.kinetic_energy().item()
    div0 = model.ops.divergence(ic.u, ic.v, ic.w).abs().max().item()
    theta_max = ic.theta.abs().max().item()
    # Analitik E_kin domain'e bagli: A^2/8 * (1 + (Ly/Lx)^2)
    d = cfg.domain
    kx_ky = d.Ly / d.Lx  # = kx/ky cunku kx=2*pi/Lx, ky=2*pi/Ly
    E_analytic = 1.0 / 8.0 * (1.0 + kx_ky**2)
    print(f"  IC E_kin = {E0:.6f} (analitik: {E_analytic:.6f})")
    print(f"  IC div_max = {div0:.2e} (hedef: ~0)")
    print(f"  IC theta_max = {theta_max:.2e} (hedef: 0)")

    # Simulasyon
    result = run_simulation(
        model=model,
        config=cfg,
        device=dev,
        n_steps=n_steps,
        log_interval=log_interval,
        name="TGV3D Decay (Re=1600)",
        ic_state=ic,
        post_step_fn=post_step_tgv_no_thermal,
    )

    # Fiziksel dogrulama
    m = result.final_metrics
    hist = result.metrics_history
    if m and hist:
        print("\n  FIZIKSEL DOGRULAMA:")
        E_final = m.get("E_kin", 0)
        print(f"    E_kin(0) = {E0:.6f}")
        print(f"    E_kin(final) = {E_final:.6f}")
        if E0 > 0:
            decay_ratio = E_final / E0
            print(f"    E_decay_ratio = {decay_ratio:.4f} (final/initial)")
            if decay_ratio < 1.0:
                print(f"    (PASS: enerji azalmis)")
            else:
                print(f"    (FAIL: enerji ARTMIS — korunum ihlali)")

        # Enstrophy evrimi
        Z_values = [h.get("Z_enstrophy", 0) for h in hist]
        Z_max = max(Z_values) if Z_values else 0
        Z_max_step = Z_values.index(Z_max) if Z_values else -1
        print(f"    Z_max = {Z_max:.4f} at log_step={Z_max_step}")
        print(f"    Z_final = {m.get('Z_enstrophy', 0):.4f}")

        # theta kontrolu
        theta_rms = m.get("theta_rms", 0)
        print(f"    theta_rms = {theta_rms:.2e} (hedef: ~0)")

    # Memory temizle
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return result


# =====================================================================
# Limit Analizi
# =====================================================================


def print_limit_analysis(results: List[SimulationResult]):
    """
    Limit problem sonuclarinin fiziksel analizini yazdir.

    Her limit problem farkli bir fizik mekanizmasini izole eder:
      - Rayleigh-Benard: buoyancy-driven konveksiyon
      - Kolmogorov: forcing-driven turbulans
      - TGV: serbest decay (dissipation)

    INNATE'in mixed convection icin egitilmis SGS parametreleri
    bu limit durumlarda ne kadar basarili?
    """
    if len(results) < 3:
        print("  Yeterli sonuc yok, analiz atlanacak.")
        return

    rb, kolm, tgv = results[0], results[1], results[2]

    print()
    print("=" * 80)
    print("  LIMIT PROBLEM ANALIZI")
    print("=" * 80)

    # Stabilite ozeti
    print("\n  STABILITE:")
    for r in results:
        status = "STABIL" if r.stable else f"INSTABIL (NaN at step {r.nan_step})"
        print(f"    {r.name:<40s} {status}")

    # Fizik dogrulamasi
    print("\n  FIZIK DOGRULAMASI:")

    # A) Rayleigh-Benard
    if rb.final_metrics:
        Nu = rb.final_metrics.get("Nu", 0)
        E = rb.final_metrics.get("E_kin", 0)
        rb_pass = Nu > 1.0 and E > 1e-6
        print(f"\n  A) Rayleigh-Benard:")
        print(f"     Nu = {Nu:.4f} (beklenti: > 1.0) {'PASS' if Nu > 1.0 else 'FAIL'}")
        print(f"     E_kin = {E:.6f} (beklenti: > 0) {'PASS' if E > 1e-6 else 'FAIL'}")
        # Buoyancy-driven konveksiyonda termal isi transferi Nu ile olculur.
        # Nu > 1: konvektif transfer > saf iletim. Basarili.
    else:
        print(f"\n  A) Rayleigh-Benard: SONUC YOK (NaN)")

    # B) Kolmogorov
    if kolm.final_metrics:
        slope = kolm.final_metrics.get("spectrum_slope", float("nan"))
        E = kolm.final_metrics.get("E_kin", 0)
        print(f"\n  B) Kolmogorov:")
        print(f"     E_kin = {E:.6f} (beklenti: > 0, doygunluk)")
        if slope == slope:
            slope_ok = -2.5 < slope < -1.0
            print(f"     Spektral egim = {slope:.3f} (hedef: -1.667) "
                  f"{'KABUL' if slope_ok else 'FARKLI'}")
        else:
            print(f"     Spektral egim = N/A")
    else:
        print(f"\n  B) Kolmogorov: SONUC YOK (NaN)")

    # C) TGV
    if tgv.final_metrics and tgv.metrics_history:
        E_first = tgv.metrics_history[0].get("E_kin", 0)
        E_final = tgv.final_metrics.get("E_kin", 0)
        Z_values = [h.get("Z_enstrophy", 0) for h in tgv.metrics_history]
        Z_max_val = max(Z_values) if Z_values else 0

        decay_ok = E_final < E_first if E_first > 0 else False
        print(f"\n  C) Taylor-Green Vortex:")
        print(f"     E_kin(0) = {E_first:.6f}")
        print(f"     E_kin(final) = {E_final:.6f}")
        print(f"     Enerji korunumu: {'PASS (azalis)' if decay_ok else 'FAIL (artis!)'}")
        print(f"     Z_max = {Z_max_val:.4f}")
        print(f"     theta_rms = {tgv.final_metrics.get('theta_rms', 0):.2e} (hedef: ~0)")
    else:
        print(f"\n  C) TGV: SONUC YOK (NaN)")

    # Genel degerlendirme
    print("\n  GENEL DEGERLENDIRME:")
    n_stable = sum(1 for r in results if r.stable)
    print(f"    Stabilite: {n_stable}/3 limit problem stabil")

    if n_stable == 3:
        print("    INNATE SGS parametreleri farkli fizik rejimlerinde STABIL.")
        print("    Bu, ogrenilen parametrelerin fiziksel anlamliligi icin olumlu.")
    elif n_stable >= 2:
        print("    Cogunluk stabil. Basarisiz vaka incelenmeli.")
        failed = [r.name for r in results if not r.stable]
        print(f"    Basarisiz: {', '.join(failed)}")
    else:
        print("    Cogu limit problem stabil degil.")
        print("    SGS parametreleri egitim rejiminine cok spesifik olabilir.")
    print()


# =====================================================================
# Ana Fonksiyon
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="INNATE 3D Limit Problemleri Testi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test vakalari:
  A) Saf Rayleigh-Benard konveksiyonu (forcing=0, Ra=1e7)
  B) Saf Kolmogorov akisi (buoyancy=0, Ra=0)
  C) Taylor-Green Vortex 3D decay (forcing=0, buoyancy=0, Re=1600)

Ornekler:
  # Tam test (3 vaka x 500 adim)
  python tests/demo4_limit_problems.py --checkpoint best.pt

  # Hizli test
  python tests/demo4_limit_problems.py --checkpoint best.pt --steps 200

  # Sadece TGV testi
  python tests/demo4_limit_problems.py --checkpoint best.pt --only tgv
        """,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Egitilmis model checkpoint dosyasi (zorunlu)",
    )
    parser.add_argument(
        "--steps", type=int, default=500,
        help="Her vaka icin forward adim sayisi (default: 500)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda | mps | cpu (default: otomatik)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/demo4_limit_problems",
        help="Sonuc dizini (default: results/demo4_limit_problems)",
    )
    parser.add_argument(
        "--log-interval", type=int, default=50,
        help="Her kac adimda metrik kaydedilecek (default: 50)",
    )
    parser.add_argument(
        "--only", type=str, default=None, choices=["rb", "kolmogorov", "tgv"],
        help="Sadece tek bir vaka calistir: rb | kolmogorov | tgv",
    )

    args = parser.parse_args()

    # Checkpoint var mi kontrol
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"HATA: Checkpoint bulunamadi: {ckpt_path}")
        sys.exit(1)

    # Output dizini
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ================================================================
    # Header
    # ================================================================

    print()
    print("=" * 80)
    print("  INNATE 3D — LIMIT PROBLEM TESTI")
    print("  Soru: Ogrenilen SGS parametreleri ne kadar evrensel?")
    print("=" * 80)
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  Steps/vaka: {args.steps}")
    if args.only:
        print(f"  Sadece: {args.only}")
    print()

    all_results: List[SimulationResult] = []
    total_t0 = time.time()

    # ================================================================
    # Test A: Rayleigh-Benard
    # ================================================================
    if args.only is None or args.only == "rb":
        rb_result = run_rayleigh_benard(
            checkpoint_path=str(ckpt_path),
            n_steps=args.steps,
            device=args.device,
            log_interval=args.log_interval,
        )
        all_results.append(rb_result)

    # ================================================================
    # Test B: Kolmogorov
    # ================================================================
    if args.only is None or args.only == "kolmogorov":
        kolm_result = run_kolmogorov(
            checkpoint_path=str(ckpt_path),
            n_steps=args.steps,
            device=args.device,
            log_interval=args.log_interval,
        )
        all_results.append(kolm_result)

    # ================================================================
    # Test C: TGV3D
    # ================================================================
    if args.only is None or args.only == "tgv":
        tgv_result = run_tgv3d(
            checkpoint_path=str(ckpt_path),
            n_steps=args.steps,
            device=args.device,
            log_interval=args.log_interval,
        )
        all_results.append(tgv_result)

    total_time = time.time() - total_t0

    # ================================================================
    # Sonuclari raporla
    # ================================================================

    # Karsilastirma tablosu
    print_comparison_table(
        all_results,
        title="LIMIT PROBLEMLERI — KARSILASTIRMA TABLOSU",
    )

    # Detayli analiz (sadece 3 vaka varsa)
    if len(all_results) == 3:
        print_limit_analysis(all_results)

    # JSON kaydi
    json_path = str(out_dir / "demo4_results.json")
    save_results(all_results, json_path)

    # Ozet
    print(f"\nToplam sure: {total_time:.1f}s ({total_time/60:.1f} dakika)")
    n_stable = sum(1 for r in all_results if r.stable)
    print(f"Stabil vakalar: {n_stable}/{len(all_results)}")

    # Fiziksel limit sonuclari ozet tablosu
    print("\n  LIMIT SONUCLARI:")
    print(f"  {'Vaka':<40s} {'Fizik':>10s} {'Stabilite':>12s}")
    print(f"  {'-'*65}")
    for r in all_results:
        m = r.final_metrics
        if "Rayleigh" in r.name:
            fizik = f"Nu={m.get('Nu', 0):.2f}" if m else "N/A"
        elif "Kolmogorov" in r.name:
            s = m.get("spectrum_slope", float("nan")) if m else float("nan")
            fizik = f"slope={s:.2f}" if s == s else "N/A"
        elif "TGV" in r.name:
            fizik = f"E={m.get('E_kin', 0):.4f}" if m else "N/A"
        else:
            fizik = "?"
        stab = "STABIL" if r.stable else f"NaN@{r.nan_step}"
        print(f"  {r.name:<40s} {fizik:>10s} {stab:>12s}")
    print()


if __name__ == "__main__":
    main()
