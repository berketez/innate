#!/usr/bin/env python3
"""
demo5_engineering.py - Gercek Dunya Muhendislik Problemleri

INNATE 3D Mixed Convection modelinin gercek muhendislik senaryolarina
uygulanmasi. Spectral periodic domain'de "oz fizik" test ediliyor:
gercek geometri yok, ama turbulans, isi transferi, SGS mekanizmalari ayni.

5 farkli muhendislik senaryosu:
  5a) Ruzgar turbini wake: Actuator disk + uniform forcing
  5b) Bina etrafinda ruzgar: Brinkman penalizasyon + mixed convection
  5c) Dere/nehir akisi: Stratifiye kanal + buoyancy mixing
  5d) Veri merkezi sogutma: Fan forcing + guclu buoyancy (Ri >> 1)
  5e) Atmosferik sinir tabakasi: Extreme Re/Ra (Re=500K, Ra=1e10)

Her problem INNATE'in spectral periodic yapisina uyarlanmistir.
Custom forcing/obstacle'lar post_step_fn callback ile eklenir.

Fizik Notlari:
  - INNATE egitim rejimi: Re=5000-10000, Ra=1e5-1e7, Kolmogorov forcing
  - Bu test: Re=5K-500K, Ra=0-1e10, uniform forcing + custom post-step
  - Periyodik domain 6x10x4: gercek geometri icin uygun degil,
    ama fiziksel mekanizmalari (wake deficit, penalizasyon, stratifikasyon)
    spectral cozucu uzerinde test edebiliyoruz.

Yazar: Berke Tezgocen (tasarim), Claude (implementasyon)
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch

# -- path setup --
_this_dir = Path(__file__).resolve().parent
_project_dir = _this_dir.parent
sys.path.insert(0, str(_this_dir))
sys.path.insert(0, str(_project_dir))

from demo_utils import (
    SimulationResult,
    load_trained_model,
    run_simulation,
    save_results,
    print_comparison_table,
    print_final_report,
)
from model import ThermalFluidState


# =====================================================================
# 5a. Ruzgar Turbini Iz Bolgesi (Wind Turbine Wake)
# =====================================================================


def _make_drag_disk_post_step(model) -> Callable:
    """
    Actuator disk modeli: lokalize drag kuvveti post-step callback.

    Fizik:
      Ruzgar turbininin arkasindaki turbulant iz bolgesi, yuksek Re,
      guclu shear, anizotropik turbulansi. Periyodik domain'de
      "actuator disk" yaklasimi ile turbin etkisi modelleniyor.

    Formul:
      F_drag = -C_T * 0.5 * U_inf^2 * G(x - x_disk, sigma) * e_x

      G(x, sigma) = exp(-(x - x_disk)^2 / (2*sigma^2)) / (sigma * sqrt(2*pi))
        -> Gaussian disk profili (x-yonunde lokalize)

      C_T = 0.75  (tipik thrust coefficient, modern turbinler icin)
      sigma = 0.3  (disk kalinligi ~ 3 grid cell)
      x_disk = Lx / 4  (domain'in ilk ceyreginde)
      U_inf = 1.0  (boyutsuz freestream hizi)

    Uygulama:
      u_new = u + dt_eff * F_drag
      Sadece x-yonunde (akisa karsi) etki eder.
      v, w, p, theta degismez.

    Neden post-step:
      Model icindeki Forcing3D "uniform" modda baseline ruzgari saglar.
      Drag disk ek bir kuvvet olarak model step'inden SONRA uygulanir.
      Bu two-step yaklasim (Forcing + Post-Step Drag) actuator disk
      literaturundeki standart uygulamadir.
    """
    device = next(model.parameters()).device
    d = model.config.domain

    # x-koordinat grid'i [1, Nx, 1, 1] -- broadcast icin
    x = torch.linspace(0, d.Lx, d.Nx, device=device).view(1, d.Nx, 1, 1)

    # Disk parametreleri
    x_disk = d.Lx / 4.0   # turbin konumu: domain'in 1/4'u
    sigma = 0.3            # disk kalinligi (Gaussian std dev)
    C_T = 0.75             # thrust coefficient
    U_inf = 1.0            # freestream hizi (boyutsuz)

    # Gaussian disk profili (onceden hesapla -- her adimda ayni)
    # G(x) = exp(-(x - x_disk)^2 / (2*sigma^2)) / (sigma * sqrt(2*pi))
    G = torch.exp(-(x - x_disk) ** 2 / (2 * sigma ** 2)) / (
        sigma * math.sqrt(2 * math.pi)
    )

    # Drag kuvveti (sabit, onceden hesapla)
    # F_drag = -C_T * 0.5 * U_inf^2 * G  (negatif: akisa karsi)
    F_drag = -C_T * 0.5 * U_inf ** 2 * G  # [1, Nx, 1, 1]

    def drag_disk_post_step(state: ThermalFluidState, step: int, model) -> ThermalFluidState:
        """Her adimda actuator disk drag kuvvetini uygula."""
        # Efektif zaman adimi (model'den al)
        dt_eff = model._dt_base * torch.clamp(model.dt_scale, 0.5, 2.0).item()

        # u-momentum guncelle: u_new = u + dt * F_drag
        u_new = state.u + dt_eff * F_drag

        # Re-projection: drag disk u'yu degistirdi, divergence-free garanti et
        u_new, v_new, w_new, p_new = model.projections[0](u_new, state.v, state.w)

        return ThermalFluidState(
            u=u_new,
            v=v_new,
            w=w_new,
            p=p_new,
            theta=state.theta,
            t=state.t,
            rho=state.rho,
        )

    return drag_disk_post_step


def run_wind_turbine_wake(args) -> SimulationResult:
    """
    5a: Ruzgar Turbini Iz Bolgesi (Wake Turbulence)

    Senaryo:
      - Atmosferik ruzgar (uniform forcing) + actuator disk (post-step drag)
      - Buoyancy KAPALI (Ra=0 -> saf momentum problemi)
      - Re = 100,000 (egitimden ~20x yuksek!)

    Beklentiler:
      - Disk arkasinda velocity deficit olusacak (u azalma)
      - Wake bolgesi turbulant olacak (yuksek enstrophy)
      - Periyodik BC nedeniyle wake domain'i dolasip tekrar disk'e gelecek
        (gercekci degil ama fiziksel mekanizma dogru)
      - Bu Re'de model stabil kalirsa basarili
      - CFL < 1 olmali (yoksa NaN riski)

    Fiziksel olcekler:
      - Wake recovery distance: ~10D (D = turbin cap)
      - Turbulence intensity: ~15-25% near-wake, ~5-10% far-wake
      - Periyodik domain'de tam recovery beklenmez
    """
    print("\n" + "#" * 70)
    print("# 5a: RUZGAR TURBINI IZ BOLGESI (Wind Turbine Wake)")
    print("#" * 70)
    print("# Actuator disk modeli, Re=100K, buoyancy KAPALI")
    print("# F_drag = -C_T * 0.5 * U_inf^2 * G(x - x_disk)")
    print("# C_T=0.75, sigma=0.3, x_disk=Lx/4")
    print("#" * 70)

    # Ra=0: buoyancy kapali. Saf momentum problemi.
    # Forcing: uniform (freestream ruzgar)
    model, cfg, dev = load_trained_model(
        checkpoint_path=args.checkpoint,
        Re=100_000,
        Ra=1e-10,  # 0 yerine cok kucuk deger (log(0) sorununu onle)
        Pr=0.71,
        forcing_mode="uniform",
        device=args.device,
    )

    # Actuator disk post-step callback olustur
    post_step = _make_drag_disk_post_step(model)

    result = run_simulation(
        model=model,
        config=cfg,
        device=dev,
        n_steps=args.steps,
        log_interval=max(args.steps // 10, 1),
        name="5a: Wind Turbine Wake",
        post_step_fn=post_step,
    )

    # Ek analiz: wake deficit
    if result.stable and result.final_metrics:
        m = result.final_metrics
        print(f"\n  [Wake Analiz]")
        print(f"    E_kin = {m.get('E_kin', 0):.6f}")
        print(f"    Z_enstrophy = {m.get('Z_enstrophy', 0):.4f}")
        print(f"    CFL = {m.get('CFL', 0):.4f}")
        print(f"    Buoyancy (Ra~0): P_buoyancy = {m.get('P_buoyancy', 0):.2e}")
        print(f"    Spectrum slope = {m.get('spectrum_slope', float('nan')):.3f}")
        print(f"    (Hedef: slope ~ -5/3 = -1.667)")

        # Wake deficit: U(x)/U_inf profili (x-yonunde, y ve z ortalamalanmis)
        # Delta_U(x) = 1 - <U(x)>_{y,z} / U_inf
        with torch.no_grad():
            state = result.final_state if hasattr(result, 'final_state') and result.final_state is not None else None
            if state is not None:
                u_profile = state.u.mean(dim=(-2, -1)).squeeze()  # [Nx]
                U_inf = u_profile.max().item()
                deficit = 1.0 - u_profile / (U_inf + 1e-10)
                max_deficit = deficit.max().item()
                print(f"  Wake: max deficit = {max_deficit:.3f}, U_inf = {U_inf:.4f}")
            else:
                print(f"  Wake deficit: final_state mevcut degil, hesaplanamadi")

    return result


# =====================================================================
# 5b. Bina Etrafinda Ruzgar Akisi (Urban Wind)
# =====================================================================


def _make_building_post_step(model) -> Callable:
    """
    Brinkman penalizasyon: bina bolgesinde hizi sonumle.

    Fizik:
      Sehir ortaminda binalar arasi ruzgar + termal konveksiyon.
      "Bina" gercek geometri degil, lokalize drag bolgesi.
      Brinkman penalizasyon ile akiskana giren bolgelerde
      hiz sifira yaklastirilir (kati cisim etkisi).

    Formul (implicit penalizasyon):
      du/dt = ... - (1/eta) * M(x,y,z) * u

      M(x,y,z) = 1 binanin icinde, 0 disinda
        -> Rectangular mask: |x-x_c| < w_x  AND  |y-y_c| < w_y  AND  |z-z_c| < w_z

      Bina boyutlari: 1.0 x 2.0 x 1.0 (genislik x yukseklik x derinlik)
      Bina konumu: domain merkezinde, y-yonunde alttan 1/3 noktasinda
        x_c = Lx/2, y_c = Ly/3, z_c = Lz/2

      Implicit form (stabil):
        u_new = u / (1 + alpha * dt * M)
        alpha = 10000.0  (penalizasyon katsayisi, yuksek -> sert duvar)

      Neden implicit:
        Explicit form: u_new = u * (1 - alpha*dt*M)
        alpha*dt >> 1 oldugunda explicit form negatif hiz verir (instabil).
        Implicit form: u_new = u / (1 + alpha*dt*M) her zaman [0, u] araliginda.

    Not:
      Brinkman penalizasyonu momentum'a (u,v,w) uygulanir.
      Basinc (p) ve sicaklik (theta) DEGISMEZ.
      (Gercekte bina icerisinde sicaklik da sabitlenebilir,
       ama bu basit test icin sadece momentum damping yeterli.)
    """
    device = next(model.parameters()).device
    d = model.config.domain

    # 3D koordinat grid'leri
    x = torch.linspace(0, d.Lx, d.Nx, device=device).view(1, d.Nx, 1, 1)
    y = torch.linspace(0, d.Ly, d.Ny, device=device).view(1, 1, d.Ny, 1)
    z = torch.linspace(0, d.Lz, d.Nz, device=device).view(1, 1, 1, d.Nz)

    # Bina merkezi ve boyutlari
    x_c = d.Lx / 2.0   # domain'in ortasi
    y_c = d.Ly / 3.0    # alttan 1/3 (bina tabaninin ustu)
    z_c = d.Lz / 2.0    # domain'in ortasi

    w_x = 0.5   # yari-genislik (toplam 1.0)
    w_y = 1.0   # yari-yukseklik (toplam 2.0)
    w_z = 0.5   # yari-derinlik (toplam 1.0)

    # Boolean mask -> float mask (broadcast icin onceden hesapla)
    # mask[i,j,k] = 1.0 eger nokta binanin icindeyse, 0.0 disindaysa
    mask = (
        ((x - x_c).abs() < w_x)
        & ((y - y_c).abs() < w_y)
        & ((z - z_c).abs() < w_z)
    ).float()  # [1, Nx, Ny, Nz]

    # Penalizasyon katsayisi
    # alpha=100 yetersiz: 1/(1+100*0.025)=1/3.5 -> bina icinde hiz %29 kaliyor
    # alpha=10000: 1/(1+10000*0.025)=1/251 -> bina icinde hiz %0.4 (gercekci)
    alpha = 10000.0  # buyuk alpha -> sert duvar (eta = 1/alpha -> kucuk)

    # Bina mask istatistikleri
    n_bldg = mask.sum().item()
    n_total = d.Nx * d.Ny * d.Nz
    print(f"  [Building mask] {n_bldg:.0f}/{n_total} cells ({100*n_bldg/n_total:.1f}%)")
    print(f"  [Building] center=({x_c:.1f}, {y_c:.1f}, {z_c:.1f}), "
          f"size=({2*w_x:.1f}, {2*w_y:.1f}, {2*w_z:.1f})")
    print(f"  [Penalization] alpha={alpha:.0f}, implicit form: u/(1+alpha*dt*M)")

    def building_post_step(state: ThermalFluidState, step: int, model) -> ThermalFluidState:
        """Her adimda bina bolgesindeki hizi implicit Brinkman ile sonumle."""
        dt_eff = model._dt_base * torch.clamp(model.dt_scale, 0.5, 2.0).item()

        # Implicit penalizasyon: damp = 1 / (1 + alpha * dt * mask)
        # Bina icinde: damp ~ 1/(1+10000*0.025*1) = 1/251 ~ 0.004
        # Bina disinda: damp = 1/(1+0) = 1.0 (degismez)
        damp = 1.0 / (1.0 + alpha * dt_eff * mask)

        u_d = state.u * damp
        v_d = state.v * damp
        w_d = state.w * damp

        # Re-projection: Brinkman damping divergence bozar, divergence-free garanti et
        u_d, v_d, w_d, p_new = model.projections[0](u_d, v_d, w_d)

        return ThermalFluidState(
            u=u_d,
            v=v_d,
            w=w_d,
            p=p_new,
            theta=state.theta,
            t=state.t,
            rho=state.rho,
        )

    return building_post_step


def run_urban_wind(args) -> SimulationResult:
    """
    5b: Bina Etrafinda Ruzgar Akisi (Urban Wind Flow)

    Senaryo:
      - Uniform forcing (ruzgar) + buoyancy (gunes isinmasi)
      - Bina = lokalize Brinkman penalizasyon bolgesi
      - Mixed convection: hem ruzgar hem termal etki

    Parametreler:
      Re = 50,000 (sehir olcegi, egitimden 10x)
      Ra = 1e8 (asfalt isinmasi -> guclu buoyancy)
      Ri = Ra/(Re^2*Pr) = 1e8/(2.5e9*0.71) = 0.056
        -> Forced convection baskIn (Ri < 1)
        -> Ruzgar etkisi termalden guclu

    Beklentiler:
      - Bina arkasinda recirculation bolgesi
      - Termal plume bina etrafinda yukari cikacak
      - Wake bolgesi turbulant
      - Ri < 1 oldugu icin ruzgar etkisi baskIn
      - Spectral method periyodik BC nedeniyle "bina tekrari" olusturacak
        (bu urban canopy yaklasimina benzer -- bina dizisi!)
    """
    print("\n" + "#" * 70)
    print("# 5b: BINA ETRAFINDA RUZGAR AKISI (Urban Wind)")
    print("#" * 70)
    print("# Brinkman penalizasyon + mixed convection")
    print("# Re=50K, Ra=1e8, Ri=0.056 (forced convection baskin)")
    print("# Bina: 1x2x1, merkez=(Lx/2, Ly/3, Lz/2)")
    print("#" * 70)

    model, cfg, dev = load_trained_model(
        checkpoint_path=args.checkpoint,
        Re=50_000,
        Ra=1e8,
        Pr=0.71,
        forcing_mode="uniform",
        device=args.device,
    )

    # Building post-step callback
    post_step = _make_building_post_step(model)

    result = run_simulation(
        model=model,
        config=cfg,
        device=dev,
        n_steps=args.steps,
        log_interval=max(args.steps // 10, 1),
        name="5b: Urban Wind",
        post_step_fn=post_step,
    )

    # Ek analiz
    if result.stable and result.final_metrics:
        m = result.final_metrics
        Ri = 1e8 / (50_000 ** 2 * 0.71)
        print(f"\n  [Urban Wind Analiz]")
        print(f"    Ri = {Ri:.4f} ({'forced conv. baskin' if Ri < 1 else 'buoyancy baskin'})")
        print(f"    Nu = {m.get('Nu', 0):.4f} (termal etki olcusu)")
        print(f"    T_min = {m.get('T_min', 0):.2f}, T_max = {m.get('T_max', 0):.2f}")
        print(f"    Spectrum slope = {m.get('spectrum_slope', float('nan')):.3f}")

    return result


# =====================================================================
# 5c. Dere / Nehir Akisi (Environmental Flow)
# =====================================================================


def run_river_flow(args) -> SimulationResult:
    """
    5c: Stratifiye Acik Kanal Akisi (River / Environmental Flow)

    Fizik:
      Stratifiye (sicaklik katmanli) acik kanal akisi.
      Alt sicak, ust soguk -> KARARSIZ stratifikasyon (unstable).
      Buoyancy-driven mixing: sicak su yukari cikar, soguk iner.
      Bu aslinda INNATE'in egitildigi problemin yakin benzeri!

    Neden onemli:
      Nehir/gol stratifikasyonu, okyanus karisimi, atmosferik konveksiyon
      hep ayni fizik: sicaklik gradyani -> buoyancy -> karisim.
      INNATE bunu ogrendiyse, farkli Re'de de yapabilmeli.

    Parametreler:
      Re = 20,000 (egitimden 4x yuksek)
      Ra = 1e7 (egitim araliginin ustu)
      Ri = Ra/(Re^2*Pr) = 1e7/(4e8*0.71) = 0.035
        -> Forced convection baskin

    Default IC:
      Alt sicak (T_hot=20), ust soguk (T_cold=0)
      Bu INNATE'in standart IC'si -- unstable stratifikasyon

    Beklentiler:
      - Termal konveksiyon hucreleri olusacak
      - Enstrophy artacak (turbulansin gelismesi)
      - Nu > 1 (konvektif isi transferi)
      - Bu demodan en iyi sonuclari bekliyoruz (egitim rejimine yakin)
      - Spectrum slope ~ -5/3 mumkun

    Custom forcing yok: uniform forcing (basinc gradyani) + buoyancy yeterli.
    """
    print("\n" + "#" * 70)
    print("# 5c: DERE/NEHIR AKISI (Environmental Flow)")
    print("#" * 70)
    print("# Stratifiye kanal, Re=20K, Ra=1e7")
    print("# Kararsiz stratifikasyon: alt sicak, ust soguk -> buoyancy mixing")
    print("# Custom forcing YOK -- INNATE'in egitim rejimine en yakin test")
    print("#" * 70)

    model, cfg, dev = load_trained_model(
        checkpoint_path=args.checkpoint,
        Re=20_000,
        Ra=1e7,
        Pr=0.71,
        forcing_mode="uniform",
        device=args.device,
    )

    result = run_simulation(
        model=model,
        config=cfg,
        device=dev,
        n_steps=args.steps,
        log_interval=max(args.steps // 10, 1),
        name="5c: River Flow",
        post_step_fn=None,  # custom forcing yok
    )

    # Ek analiz
    if result.stable and result.final_metrics:
        m = result.final_metrics
        Ri = 1e7 / (20_000 ** 2 * 0.71)
        print(f"\n  [River Flow Analiz]")
        print(f"    Ri = {Ri:.4f}")
        print(f"    Nu = {m.get('Nu', 0):.4f} (Nu>1 ise konvektif isi transferi var)")
        print(f"    theta_rms = {m.get('theta_rms', 0):.4f} (sicaklik fluktuasyonu)")
        print(f"    P_buoyancy = {m.get('P_buoyancy', 0):.2e}")
        print(f"    Spectrum slope = {m.get('spectrum_slope', float('nan')):.3f}")

    return result


# =====================================================================
# 5d. Veri Merkezi Sogutma (Data Center Cooling)
# =====================================================================


def run_data_center_cooling(args) -> SimulationResult:
    """
    5d: Veri Merkezi Sogutma (Data Center Cooling)

    Fizik:
      Server rack'leri arasi sicak/soguk koridor akisi.
      Dusuk hizli fan (forced convection) + guclu isi kaynaklari (buoyancy).
      Mixed convection'in EXTREME hali: buoyancy tamamen baskin.

    Parametreler:
      Re = 5,000 (dusuk hiz fan -- egitim rejiminde!)
      Ra = 1e9 (cok yuksek isi akisi -- egitimden 1000x)
      Pr = 0.71
      Ri = Ra/(Re^2*Pr) = 1e9/(2.5e7*0.71) = 56.3

    Richardson number analizi:
      Ri = 56.3 >> 1 -> Buoyancy TAMAMEN baskin
      Egitimde tipik Ri ~ 0.03-5.6 arasi
      Bu test 10x-1000x daha buyuk Ri

    Beklentiler:
      - Fan etkisi neredeyse hissedilmeyecek (Ri >> 1)
      - Guclu termal plume'lar olusacak
      - Sicaklik fluktuasyonlari cok buyuk olacak
      - Nu >> 1 (cok guclu konvektif isi transferi)
      - NaN riski YUKSEK (Ra=1e9 ciddi)
      - Stabil kalirsa buyuk basari

    Custom forcing yok: uniform forcing + extreme buoyancy.
    Ri >> 1 oldugu icin buoyancy zaten dominate edecek.
    """
    print("\n" + "#" * 70)
    print("# 5d: VERI MERKEZI SOGUTMA (Data Center Cooling)")
    print("#" * 70)
    print("# Fan (uniform) + EXTREME buoyancy")
    print("# Re=5K, Ra=1e9, Ri=56.3 >> 1 (buoyancy tamamen baskin)")
    print("# Egitimden 1000x buyuk Ra -- EXTREME test")
    print("#" * 70)

    model, cfg, dev = load_trained_model(
        checkpoint_path=args.checkpoint,
        Re=5_000,
        Ra=1e9,
        Pr=0.71,
        forcing_mode="uniform",
        device=args.device,
    )

    Ri = 1e9 / (5_000 ** 2 * 0.71)
    print(f"  [Data Center] Ri = {Ri:.1f} (egitimde tipik ~0.03-5.6)")
    print(f"  [Data Center] Ra/Ra_train = {1e9/1e7:.0f}x (100x egitim ustu)")

    result = run_simulation(
        model=model,
        config=cfg,
        device=dev,
        n_steps=args.steps,
        log_interval=max(args.steps // 10, 1),
        name="5d: Data Center Cooling",
        post_step_fn=None,
    )

    # Ek analiz
    if result.stable and result.final_metrics:
        m = result.final_metrics
        print(f"\n  [Data Center Analiz]")
        print(f"    Ri = {Ri:.1f} >> 1: buoyancy baskin")
        print(f"    Nu = {m.get('Nu', 0):.4f} (cok yuksek beklenir)")
        print(f"    theta_rms = {m.get('theta_rms', 0):.4f}")
        print(f"    P_buoyancy = {m.get('P_buoyancy', 0):.2e}")
        print(f"    P_forcing = {m.get('P_forcing', 0):.2e}")
        P_b = abs(m.get('P_buoyancy', 0))
        P_f = abs(m.get('P_forcing', 1e-10))
        if P_f > 0:
            print(f"    P_buoyancy/P_forcing = {P_b/P_f:.1f}x")
    elif not result.stable:
        print(f"\n  [Data Center] NaN at step {result.nan_step}")
        print(f"  Ra=1e9 cok agresif. Beklenen davranis.")

    return result


# =====================================================================
# 5e. Atmosferik Sinir Tabakasi (ABL)
# =====================================================================


def run_atmospheric_boundary_layer(args) -> SimulationResult:
    """
    5e: Atmosferik Sinir Tabakasi (Atmospheric Boundary Layer)

    Fizik:
      Yer yuzeyindeki ilk 1-2 km atmosfer katmani.
      Gunduz: konvektif (Ra baskin), gece: stabil.
      Geostrophic ruzgar (uniform forcing) + termal konveksiyon.

    Parametreler:
      Re = 500,000 (atmosferik -- egitimden 100x yuksek!)
      Ra = 1e10 (egitimden 10,000x yuksek!)
      Pr = 0.71
      Ri = Ra/(Re^2*Pr) = 1e10/(2.5e11*0.71) = 0.056
        -> Ilginc: Ri ~ 0.056, forced convection baskin!
        -> Bunun sebebi: Re cok buyuk

    EXTREME test -- amac:
      1. Model patliyor mu? (NaN check)
      2. Kac adim hayatta kaliyor?
      3. CFL makul mu?
      4. Enerji patliyor mu yoksa dissipate mi oluyor?

    Nu = 1/Re = 2e-6 -> COKK kucuk viskozite!
    Grid cozunurlugu yetersiz olacak (LES skoru << 50).
    Ama modelin SGS (EddyViscosity3D) noronlari bunu
    kompanse etmeye calisacak -- INNATE'in gercek testi bu.

    Basari kriterleri (esik dusuk!):
      - 50 adim stabil: KOTU ama bir seyler yapiyor
      - 200 adim stabil: MAKUL
      - 500 adim stabil: IYI (beklenmiyor)

    Custom forcing yok: sadece uniform + buoyancy.
    """
    print("\n" + "#" * 70)
    print("# 5e: ATMOSFERIK SINIR TABAKASI (ABL)")
    print("#" * 70)
    print("# EXTREME TEST: Re=500K (100x egitim), Ra=1e10 (10000x egitim)")
    print("# nu = 2e-6, kappa = 2.8e-6 -- viskozite neredeyse sifir")
    print("# Modelin SGS noronlari bu Re'de yeterli mi?")
    print("#" * 70)

    model, cfg, dev = load_trained_model(
        checkpoint_path=args.checkpoint,
        Re=500_000,
        Ra=1e10,
        Pr=0.71,
        forcing_mode="uniform",
        device=args.device,
    )

    nu = 1.0 / 500_000
    Ri = 1e10 / (500_000 ** 2 * 0.71)
    print(f"  [ABL] nu = {nu:.2e} (cok kucuk viskozite)")
    print(f"  [ABL] Ri = {Ri:.4f} (forced conv baskin: Re cok buyuk)")
    print(f"  [ABL] Re/Re_train = {500_000/10_000:.0f}x")
    print(f"  [ABL] Ra/Ra_train = {1e10/1e7:.0f}x")

    result = run_simulation(
        model=model,
        config=cfg,
        device=dev,
        n_steps=args.steps,
        log_interval=max(args.steps // 10, 1),
        name="5e: ABL (Extreme)",
        post_step_fn=None,
    )

    # Ek analiz
    if result.stable:
        m = result.final_metrics
        print(f"\n  [ABL Analiz]")
        print(f"    {args.steps} adim STABIL! Bu etkileyici.")
        print(f"    Re=500K, Ra=1e10'da model patlamamis.")
        print(f"    CFL = {m.get('CFL', 0):.4f}")
        print(f"    Spectrum slope = {m.get('spectrum_slope', float('nan')):.3f}")
    else:
        survived = result.nan_step
        total = args.steps
        ratio = survived / total * 100 if total > 0 else 0
        print(f"\n  [ABL Analiz]")
        print(f"    NaN at step {survived}/{total} ({ratio:.1f}% hayatta)")
        if survived >= 200:
            print(f"    200+ adim -- MAKUL performans (Re=500K icin)")
        elif survived >= 50:
            print(f"    50+ adim -- BIR SEYLER yapiyor ama yetersiz")
        else:
            print(f"    <50 adim -- BASARISIZ (model bu Re'de calisamiyor)")

    return result


# =====================================================================
# OZET RAPOR
# =====================================================================


def print_engineering_summary(results: List[SimulationResult]):
    """
    Muhendislik problem sonuclarini ozetleyen detayli rapor.

    Her problem icin:
      - Fiziksel anlamlilik degerlendirmesi
      - Egitim rejimi ile karsilastirma
      - Basari/basarisizlik analizi
    """
    print("\n" + "=" * 70)
    print("  DEMO 5: MUHENDISLIK PROBLEMLERI OZET RAPOR")
    print("=" * 70)

    # Re/Ra bazinda siralama
    problem_info = {
        "5a: Wind Turbine Wake": {
            "re_factor": "20x", "ra_factor": "~0", "key_test": "High-Re wake",
            "custom": "Actuator disk (post-step)",
        },
        "5b: Urban Wind": {
            "re_factor": "10x", "ra_factor": "100x", "key_test": "Brinkman obstacle",
            "custom": "Building penalization",
        },
        "5c: River Flow": {
            "re_factor": "4x", "ra_factor": "1x", "key_test": "Stratified flow",
            "custom": "None (yakin rejim)",
        },
        "5d: Data Center Cooling": {
            "re_factor": "1x", "ra_factor": "1000x", "key_test": "Extreme Ri",
            "custom": "None (Ri>>1)",
        },
        "5e: ABL (Extreme)": {
            "re_factor": "100x", "ra_factor": "10000x", "key_test": "Extreme everything",
            "custom": "None (stres testi)",
        },
    }

    n_stable = sum(1 for r in results if r.stable)
    n_total = len(results)

    print(f"\n  Sonuc: {n_stable}/{n_total} problem STABIL\n")
    print(f"  {'Problem':<30s} {'Re(x)':<8s} {'Ra(x)':<8s} {'Stabil':>7s} {'Adim':>8s} {'Notlar'}")
    print(f"  {'-'*90}")

    for r in results:
        info = problem_info.get(r.name, {})
        ok = "OK" if r.stable else "FAIL"
        steps = f"{r.n_steps}" if r.stable else f"{r.nan_step}/{r.n_steps}"
        re_x = info.get("re_factor", "?")
        ra_x = info.get("ra_factor", "?")
        custom = info.get("custom", "")
        print(f"  {r.name:<30s} {re_x:<8s} {ra_x:<8s} {ok:>7s} {steps:>8s} {custom}")

    # Kategori bazli degerlendirme
    print(f"\n  Kategori Degerlendirmesi:")
    for r in results:
        if r.stable and r.final_metrics:
            m = r.final_metrics
            slope = m.get("spectrum_slope", float("nan"))
            nu_val = m.get("Nu", 0)
            cfl = m.get("CFL", 0)

            flags = []
            if slope == slope and abs(slope + 1.667) < 0.5:
                flags.append("spektrum OK")
            if nu_val > 1.0:
                flags.append(f"Nu={nu_val:.2f}")
            if cfl < 1.0:
                flags.append(f"CFL={cfl:.3f}")
            elif cfl < 2.0:
                flags.append(f"CFL={cfl:.3f} (dikkat)")
            else:
                flags.append(f"CFL={cfl:.3f} (YUKSEK!)")

            flags_str = ", ".join(flags) if flags else "metrik yok"
            print(f"    {r.name}: {flags_str}")
        elif not r.stable:
            print(f"    {r.name}: NaN @ step {r.nan_step}")

    print()


# =====================================================================
# MAIN
# =====================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Demo 5: Gercek Dunya Muhendislik Problemleri",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Ornekler:
  # Tum problemleri calistir (500 adim)
  python demo5_engineering.py --checkpoint model.pt --steps 500

  # Sadece ruzgar turbini ve ABL
  python demo5_engineering.py --checkpoint model.pt --problems 5a 5e

  # Hizli test (100 adim)
  python demo5_engineering.py --checkpoint model.pt --steps 100 --problems 5c

  # CPU'da calistir
  python demo5_engineering.py --checkpoint model.pt --device cpu --steps 200
""",
    )
    parser.add_argument(
        "--checkpoint", required=True, type=str,
        help="Egitilmis model checkpoint dosyasi (.pt)",
    )
    parser.add_argument(
        "--steps", type=int, default=500,
        help="Her problem icin forward adim sayisi (default: 500)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: cuda, mps, cpu (default: otomatik)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/demos",
        help="Sonuc JSON dosyasinin kaydedilecegi dizin",
    )
    parser.add_argument(
        "--problems", nargs="+", default=["5a", "5b", "5c", "5d", "5e"],
        choices=["5a", "5b", "5c", "5d", "5e"],
        help="Calistirilacak problemler (default: hepsi)",
    )
    args = parser.parse_args()

    # Checkpoint dosyasi kontrolu
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"HATA: Checkpoint bulunamadi: {ckpt_path}")
        sys.exit(1)

    print("=" * 70)
    print("  DEMO 5: GERCEK DUNYA MUHENDISLIK PROBLEMLERI")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Steps: {args.steps}")
    print(f"  Device: {args.device or 'auto'}")
    print(f"  Problems: {', '.join(args.problems)}")
    print(f"  Output: {args.output_dir}/demo5_engineering.json")
    print("=" * 70)

    # Problem dispatch tablosu
    problem_runners = {
        "5a": ("Wind Turbine Wake", run_wind_turbine_wake),
        "5b": ("Urban Wind", run_urban_wind),
        "5c": ("River Flow", run_river_flow),
        "5d": ("Data Center Cooling", run_data_center_cooling),
        "5e": ("ABL (Extreme)", run_atmospheric_boundary_layer),
    }

    results: List[SimulationResult] = []
    t0 = time.time()

    for prob_id in args.problems:
        if prob_id not in problem_runners:
            print(f"UYARI: Bilinmeyen problem: {prob_id}, atlaniyor.")
            continue

        name, runner = problem_runners[prob_id]
        print(f"\n>>> Problem {prob_id}: {name}")

        try:
            result = runner(args)
            results.append(result)
        except Exception as e:
            print(f"\n*** HATA: Problem {prob_id} calistirilirken hata: {e}")
            import traceback
            traceback.print_exc()
            # Basarisiz sonuc olustur
            results.append(SimulationResult(
                name=f"{prob_id}: {name} (ERROR)",
                config_summary={"error": str(e)},
                n_steps=args.steps,
                wall_time=0.0,
                stable=False,
                nan_step=0,
                metrics_history=[],
                final_metrics={},
            ))

    total_time = time.time() - t0

    # Sonuclari yazdir
    print_comparison_table(results, "Demo 5: Engineering Problems")
    print_engineering_summary(results)

    print(f"  Toplam sure: {total_time:.1f}s")

    # JSON kaydet
    save_results(results, f"{args.output_dir}/demo5_engineering.json")


if __name__ == "__main__":
    main()
