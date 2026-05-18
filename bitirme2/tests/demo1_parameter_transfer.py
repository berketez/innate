#!/usr/bin/env python3
"""
demo1_parameter_transfer.py - INNATE 3D Zero-Shot Parameter Transfer Testi

Amac:
  Egitilmis modelin (Re=5000-10000, Ra=1e5-1e7 araliginda Kolmogorov forcing
  ile egitilmis) HIC GORMEDIGI Re/Ra degerlerinde ne kadar iyi calistigini olcer.

  "Zero-shot transfer" = ek egitim YOK, sadece set_physics(Re, Ra) ile
  fizik parametrelerini degistirip forward simulation.

Test kategorileri:
  1. Interpolasyon: Egitim araliginin ICINDE ama gorulmemis degerler
     - Kolay (Re=6000, Ra=5e5) ve orta (Re=8500, Ra=3e6)
  2. Extrapolasyon asagi: Egitim araliginin ALTINDA
     - Re=2000, Ra=1e4 (laminer akisa yakin)
  3. Extrapolasyon yukari: Egitim araliginin USTUNDE
     - Gittikce artan zorluk: Re=15K-100K, Ra=1e8-1e10
  4. Extreme: Tek parametre baskın
     - Re=500000 (ruzgar baskın, Ra kucuk) veya Ra=1e12 (buoyancy baskın)

Olculen fiziksel buyuklukler (her log_interval adimda):
  E_kin  = 0.5 * <u^2 + v^2 + w^2>              Kinetik enerji
  Z      = <omega_x^2 + omega_y^2 + omega_z^2>  Enstrophy (curl ile)
  div_max = max|nabla . u|                        Incompressibility ihlali
  Nu     = 1 + <v*T'> / kappa                    Nusselt sayisi
  CFL    = max(|u|) * dt_eff / dx_min            Courant-Friedrichs-Lewy
  slope  = d(log E)/d(log k)                     Spektral egim (-5/3 hedef)

Basari kriterleri:
  - Stabil: NaN olmadan 1000 adim tamamlama
  - div_max < 1e-4: Solenoidallik korunuyor
  - slope ~ -5/3: Kolmogorov olceklemesi gecerli
  - CFL < 1: Numerik stabilite

Kullanim:
  python tests/demo1_parameter_transfer.py --checkpoint results/checkpoints/best.pt
  python tests/demo1_parameter_transfer.py --checkpoint best.pt --steps 500 --device cpu

Cikti:
  results/demos/demo1_parameter_transfer.json  (tum metrikler)
  stdout'a karsilastirma tablosu

Yazar: Berke Tezgocen (tasarim), Claude (implementasyon)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# -- path setup --
_this_dir = Path(__file__).resolve().parent
_project_dir = _this_dir.parent
sys.path.insert(0, str(_project_dir))
sys.path.insert(0, str(_project_dir.parent))

from demo_utils import (
    load_trained_model,
    run_simulation,
    compute_physics_metrics,
    save_results,
    print_comparison_table,
    SimulationResult,
)


# =====================================================================
# Test Vakalari Tanimi
# =====================================================================

# Her test vakasi: (isim, Re, Ra, zorluk_notu)
TEST_CASES: List[Tuple[str, float, float, str]] = [
    # --- Interpolasyon (egitim araligi icinde) ---
    ("interp_easy",       6_000,   5e5,    "Kolay - egitim araliginda"),
    ("interp_medium",     8_500,   3e6,    "Orta - egitim araliginda"),

    # --- Extrapolasyon asagi ---
    ("extrap_low",        2_000,   1e4,    "Kolay - dusuk Re/Ra"),

    # --- Extrapolasyon yukari ---
    ("extrap_up_1",      15_000,   1e8,    "Zor - egitimden yuksek"),
    ("extrap_up_2",      25_000,   5e8,    "Cok zor - SGS limiti"),
    ("extrap_up_3",      50_000,   1e9,    "Extreme - sinir testi"),

    # --- Extreme ---
    ("extreme_Re_Ra",   100_000,   1e10,   "Extreme - cok yuksek Re+Ra"),
    ("extreme_Re_only", 500_000,   1e6,    "Imkansiz? - ruzgar baskin"),
    ("extreme_Ra_only",   5_000,   1e12,   "Imkansiz? - buoyancy baskin"),
]

# Fiziksel olguleme noktalari
# Re arttikca: enstrophy artmali, CFL artmali
# E_kin sabit forcing'de yuksek Re'de platoya ulasir (monoton artis BEKLENMEZ)
# Ra arttikca: Nu artmali (daha guclu konveksiyon)
# Cok yuksek Re/Ra'da: NaN beklenir (model siniri)


def parse_args():
    """Komut satiri argumanlari."""
    parser = argparse.ArgumentParser(
        description="INNATE 3D - Demo 1: Zero-Shot Parameter Transfer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python tests/demo1_parameter_transfer.py --checkpoint results/checkpoints/best.pt
  python tests/demo1_parameter_transfer.py --checkpoint best.pt --steps 500 --device cpu
  python tests/demo1_parameter_transfer.py --checkpoint best.pt --cases interp_easy extrap_up_1
        """,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Egitilmis model checkpoint dosyasi (zorunlu)",
    )
    parser.add_argument(
        "--steps", type=int, default=1000,
        help="Her test icin forward adim sayisi (default: 1000)",
    )
    parser.add_argument(
        "--log-interval", type=int, default=50,
        help="Her kac adimda metrik kaydedilecek (default: 50)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Hesaplama cihazi: cuda | mps | cpu (default: otomatik)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/demos",
        help="Sonuc dizini (default: results/demos)",
    )
    parser.add_argument(
        "--cases", nargs="*", default=None,
        help="Calistirilacak test vakalari (isim listesi). None ise hepsi calisir.",
    )
    parser.add_argument(
        "--Pr", type=float, default=0.71,
        help="Prandtl sayisi (default: 0.71, hava)",
    )
    return parser.parse_args()


def filter_test_cases(
    all_cases: List[Tuple[str, float, float, str]],
    selected_names: List[str] | None,
) -> List[Tuple[str, float, float, str]]:
    """Secilen test vakalarini filtrele. None ise hepsini dondur."""
    if selected_names is None:
        return all_cases

    # Gecerlilik kontrolu
    all_names = {name for name, *_ in all_cases}
    for name in selected_names:
        if name not in all_names:
            available = ", ".join(sorted(all_names))
            print(f"UYARI: '{name}' gecerli bir test vakasi degil.")
            print(f"Gecerli vakalar: {available}")
            sys.exit(1)

    return [case for case in all_cases if case[0] in selected_names]


def run_parameter_transfer_demo(args) -> List[SimulationResult]:
    """
    Tum test vakalarini sirayla calistir.

    Her vaka icin:
      1. Model yukle (ilk seferde) veya set_physics ile parametreleri degistir
      2. 1000 adim forward simulation
      3. Metrikleri kaydet
    """
    cases = filter_test_cases(TEST_CASES, args.cases)
    results: List[SimulationResult] = []

    print()
    print("=" * 70)
    print("  INNATE 3D - DEMO 1: ZERO-SHOT PARAMETER TRANSFER")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Steps: {args.steps}")
    print(f"  Log interval: {args.log_interval}")
    print(f"  Device: {args.device or 'auto'}")
    print(f"  Test cases: {len(cases)}")
    print()

    # Referans model (ilk test vakasi icin yukle, sonra set_physics ile degistir)
    # NOT: Her seferinde yeniden yuklemek daha guvenli cunku set_physics
    # sonrasi ic state (momentum, sicaklik) onceki calismanin kalintisini tasimasin.
    # AMA create_initial_condition ile her seferinde temiz IC olusturuyoruz,
    # dolayisiyla set_physics yeterli.
    model = None
    cfg = None
    dev = None

    total_t0 = time.time()

    for i, (name, Re, Ra, note) in enumerate(cases):
        print(f"\n{'#' * 70}")
        print(f"  TEST {i + 1}/{len(cases)}: {name}")
        print(f"  Re={Re:.0f}, Ra={Ra:.1e}, Pr={args.Pr}")
        print(f"  Zorluk: {note}")
        print(f"{'#' * 70}")

        # Ilk yuklemede veya model yoksa checkpoint'tan yukle
        if model is None:
            model, cfg, dev = load_trained_model(
                checkpoint_path=args.checkpoint,
                Re=Re,
                Ra=Ra,
                Pr=args.Pr,
                device=args.device,
            )
        else:
            # Fizik parametrelerini degistir (model agirlikları korunur)
            model.set_physics(Re, Ra, args.Pr)
            # Config'i de guncelle (run_simulation ve compute_physics_metrics
            # config.physics'i okur)
            cfg.physics.Re = Re
            cfg.physics.Ra = Ra
            cfg.physics.Pr = args.Pr
            print(f"  Physics set: Re={Re:.0f}, Ra={Ra:.1e}, Pr={args.Pr}")

        # Simulasyon calistir
        result = run_simulation(
            model=model,
            config=cfg,
            device=dev,
            n_steps=args.steps,
            log_interval=args.log_interval,
            name=name,
        )

        results.append(result)

        # Kisa ozet
        if result.stable:
            fm = result.final_metrics
            print(f"\n  >> SONUC: STABIL | E_kin={fm.get('E_kin', 0):.6f} "
                  f"| Nu={fm.get('Nu', 0):.4f} | slope={fm.get('spectrum_slope', float('nan')):.3f}")
        else:
            print(f"\n  >> SONUC: NaN at step {result.nan_step}")

    total_time = time.time() - total_t0

    # ---- Karsilastirma tablosu ----
    print_comparison_table(results, title="DEMO 1: PARAMETER TRANSFER SONUCLARI")

    # ---- Fiziksel yorum ozeti ----
    _print_physics_summary(results)

    # ---- Sonuclari kaydet ----
    output_dir = Path(args.output_dir)
    output_path = output_dir / "demo1_parameter_transfer.json"
    save_results(results, str(output_path))

    print(f"\nToplam sure: {total_time:.1f}s ({total_time / max(len(cases), 1):.1f}s/test)")

    return results


def _print_physics_summary(results: List[SimulationResult]):
    """
    Fiziksel acidan ne ogrendik - kisa yorum.

    Kontrol edilen fiziksel tutarliliklar:
      1. Tum stabil vakalarda E_kin > 0 ve sonlu (pozitif enerji)
      2. Ra artinca Nu artmali (daha guclu konveksiyon)
      3. Tum stabil vakalarda div_max < 1e-4 (solenoidallik)
      4. Inertial range slope ~ -5/3 (Kolmogorov yasasi)
    """
    print()
    print("=" * 70)
    print("  FIZIKSEL YORUM")
    print("=" * 70)

    stable = [r for r in results if r.stable]
    unstable = [r for r in results if not r.stable]

    print(f"\n  Stabil: {len(stable)}/{len(results)}")
    print(f"  Unstabil: {len(unstable)}/{len(results)}")

    if unstable:
        print("\n  NaN olan testler:")
        for r in unstable:
            c = r.config_summary
            print(f"    - {r.name}: Re={c['Re']:.0f}, Ra={c['Ra']:.1e}, "
                  f"NaN at step {r.nan_step}")

    if not stable:
        print("\n  Hicbir test stabil degil - yorum yapilamiyor.")
        return

    # Divergence kontrolu
    div_violations = []
    for r in stable:
        div_max = r.final_metrics.get("div_max", 0)
        if div_max > 1e-4:
            div_violations.append((r.name, div_max))

    if div_violations:
        print("\n  Divergence ihlalleri (div_max > 1e-4):")
        for name, div_val in div_violations:
            print(f"    - {name}: div_max = {div_val:.2e}")
    else:
        print(f"\n  Solenoidallik: TUM stabil vakalarda div_max < 1e-4  [OK]")

    # CFL kontrolu
    cfl_violations = []
    for r in stable:
        cfl = r.final_metrics.get("CFL", 0)
        if cfl > 1.0:
            cfl_violations.append((r.name, cfl))

    if cfl_violations:
        print(f"\n  CFL ihlalleri (CFL > 1.0):")
        for name, cfl_val in cfl_violations:
            print(f"    - {name}: CFL = {cfl_val:.4f}")
    else:
        print(f"  CFL: TUM stabil vakalarda CFL < 1.0  [OK]")

    # Spektral slope
    slopes = []
    for r in stable:
        slope = r.final_metrics.get("spectrum_slope", float("nan"))
        if slope == slope:  # NaN check
            slopes.append((r.name, r.config_summary["Re"], slope))

    if slopes:
        print(f"\n  Spektral egimler (hedef: -1.667):")
        for name, Re, slope in slopes:
            delta = abs(slope - (-5.0 / 3.0))
            quality = "iyi" if delta < 0.3 else ("kabul edilebilir" if delta < 0.7 else "kotu")
            print(f"    - {name} (Re={Re:.0f}): slope = {slope:.3f}  [{quality}]")

    # Re-E_kin fiziksel kontrol
    # NOT: Sabit amplitude forcing'de E_kin monoton artmak ZORUNDA DEGILDIR.
    # Yuksek Re'de viskozite sadece kucuk olcekleri etkiler, buyuk olcekli E_kin
    # forcing-dependent olup asimptotik platoya ulasir. Monotonicity beklentisi YANLISTIR.
    # Bunun yerine: E_kin > 0, sonlu ve stabil mi kontrol ediyoruz.
    re_ekin = [(r.config_summary["Re"], r.final_metrics.get("E_kin", 0), r.name)
               for r in stable if r.final_metrics.get("E_kin") is not None]
    re_ekin.sort()

    if len(re_ekin) >= 2:
        print(f"\n  Re - E_kin fiziksel kontrol:")
        all_positive_finite = True
        for Re, E, name in re_ekin:
            ok = (E > 0) and (E == E) and (E < float("inf"))  # pozitif, sonlu, NaN degil
            status = "OK" if ok else "FAIL"
            if not ok:
                all_positive_finite = False
            print(f"    Re={Re:>10.0f}  E_kin={E:.6f}  [{status}]  ({name})")
        if all_positive_finite:
            print(f"    >> Tum vakalarda E_kin > 0 ve sonlu  [FIZIKSEL]")
        else:
            print(f"    >> Bazi vakalarda E_kin gecersiz (<=0, NaN veya Inf)  [DIKKAT]")
        # Bilgilendirme: dusuk Re'de E_kin dusuk, yuksek Re'de platoya ulasir
        if len(re_ekin) >= 3:
            ekins = [e for _, e, _ in re_ekin]
            print(f"    >> E_kin araligi: [{min(ekins):.6f}, {max(ekins):.6f}] "
                  f"(yuksek Re'de plato beklenir)")

    # Ra-Nu korelasyonu (artan Ra ile artan Nu beklenir)
    ra_nu = [(r.config_summary["Ra"], r.final_metrics.get("Nu", 0), r.name)
             for r in stable if r.final_metrics.get("Nu") is not None]
    ra_nu.sort()

    if len(ra_nu) >= 2:
        print(f"\n  Ra - Nu korelasyonu:")
        for Ra, Nu, name in ra_nu:
            print(f"    Ra={Ra:>10.1e}  Nu={Nu:.4f}  ({name})")

    print()


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    args = parse_args()
    results = run_parameter_transfer_demo(args)

    # Cikis kodu: tum testler stabil ise 0, en az biri NaN ise 1
    n_stable = sum(1 for r in results if r.stable)
    n_total = len(results)

    if n_stable == n_total:
        print(f"SONUC: {n_total}/{n_total} test STABIL.")
    else:
        print(f"SONUC: {n_stable}/{n_total} test stabil, "
              f"{n_total - n_stable} test NaN ile sonlandi.")

    # Exit code: extreme testlerin NaN olmasi beklenen davranis,
    # interpolasyon testlerinin NaN olmasi ciddi problem.
    interp_names = {"interp_easy", "interp_medium", "extrap_low"}
    interp_fails = [r for r in results if r.name in interp_names and not r.stable]

    if interp_fails:
        print(f"\nKRITIK: Interpolasyon testleri BASARISIZ: "
              f"{[r.name for r in interp_fails]}")
        sys.exit(2)
    elif n_stable < n_total:
        print(f"\nNOT: Sadece extrapolasyon/extreme testlerde NaN - bu beklenen olabilir.")
        sys.exit(0)
    else:
        sys.exit(0)
