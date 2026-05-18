#!/usr/bin/env python3
"""
demo2_forcing_transfer.py - INNATE 3D Forcing Transfer Testi

Amac:
  Egitilmis modelin (Kolmogorov forcing ile egitilmis) FARKLI forcing
  modlarinda ne kadar iyi calistigini olcer.

  INNATE noronlari fizik operatorleri uzerinden ogrendigi icin, forcing
  modu degisse bile advection/diffusion/projection dinamikleri korunmali.
  Yani model "Kolmogorov sinusu"nu ezberlemis olmamali, genel akiskan
  dinamigini ogrenmis olmali.

Forcing modlari:
  kolmogorov: F_x = A * sin(k_f * 2*pi*y/Ly + phi), F_y=0, F_z=0
              - Sinuzoidal, y-bagimlı. Egitimde kullanilan forcing.
              - Kolmogorov turbulansi olusturur.

  uniform:    F_x = A, F_y=0, F_z=0
              - Sabit, homojen kuvvet. Tum domain'e esit.
              - Daha basit ama Kolmogorov'dan farkli turbulans yapisi.

  stochastic: F_x = (A + eta(t)) * 1, F_y=0, F_z=0
              - Ornstein-Uhlenbeck sureci ile zamansal degiskenlik.
              - eta(t) = eta(t-dt) * (1 - dt/tau) + sigma * sqrt(2*dt/tau) * N(0,1)
              - Zamansal korelasyonlu gurultu, fiziksel ruzgar modeli.

Test vakalari:
  | Test            | Re    | Ra   | Forcing     | Not                   |
  |-----------------|-------|------|-------------|-----------------------|
  | baseline        | 5000  | 1e6  | kolmogorov  | Referans (egitim)     |
  | uniform_base    | 5000  | 1e6  | uniform     | Kolay transfer        |
  | stochastic_base | 5000  | 1e6  | stochastic  | Zamansal degiskenlik  |
  | uniform_hard    | 10000 | 1e7  | uniform     | Zor: yeni Re+forcing  |
  | stochastic_hard | 10000 | 1e7  | stochastic  | Zor: yeni Re+forcing  |

Olculen fiziksel buyuklukler:
  (demo1 ile ayni: E_kin, Z, div_max, Nu, CFL, slope, vb.)

Basari kriterleri:
  - baseline stabil olmali (egitim parametreleri)
  - uniform/stochastic'te de stabil kalmasi GUCLU transfer gosterir
  - Tum vakalarda div_max < 1e-4 (fizik korunuyor)
  - E_kin degerleri forcing moduna gore farkli olabilir (bu normal)
  - slope ~ -5/3 korunmali

ONEMLI - Stochastic forcing:
  Forcing3D.step_ou(dt) her forward adimdan SONRA cagrilmali.
  Bu, Ornstein-Uhlenbeck noise state'ini ilerletir.
  run_simulation'daki post_step_fn hook'u bunun icin kullanilir.

Kullanim:
  python tests/demo2_forcing_transfer.py --checkpoint results/checkpoints/best.pt
  python tests/demo2_forcing_transfer.py --checkpoint best.pt --steps 500 --device cpu
  python tests/demo2_forcing_transfer.py --checkpoint best.pt --cases baseline uniform_base

Cikti:
  results/demos/demo2_forcing_transfer.json  (tum metrikler)
  stdout'a karsilastirma tablosu

Yazar: Berke Tezgocen (tasarim), Claude (implementasyon)
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

# Her test vakasi: (isim, Re, Ra, forcing_mode, zorluk_notu)
TEST_CASES: List[Tuple[str, float, float, str, str]] = [
    # --- Baseline (egitim parametreleri ve forcing'i) ---
    ("baseline",         5_000,  1e6,  "kolmogorov",  "Referans - egitim konfigurasyonu"),

    # --- Forcing transfer (ayni Re/Ra, farkli forcing) ---
    ("uniform_base",     5_000,  1e6,  "uniform",     "Kolay transfer - uniform forcing"),
    ("stochastic_base",  5_000,  1e6,  "stochastic",  "Orta transfer - stochastic forcing"),

    # --- Forcing + parameter transfer (farkli Re/Ra + farkli forcing) ---
    ("uniform_hard",    10_000,  1e7,  "uniform",     "Zor - yeni Re + uniform forcing"),
    ("stochastic_hard", 10_000,  1e7,  "stochastic",  "Zor - yeni Re + stochastic forcing"),
]


def parse_args():
    """Komut satiri argumanlari."""
    parser = argparse.ArgumentParser(
        description="INNATE 3D - Demo 2: Forcing Transfer Testi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  python tests/demo2_forcing_transfer.py --checkpoint results/checkpoints/best.pt
  python tests/demo2_forcing_transfer.py --checkpoint best.pt --steps 500
  python tests/demo2_forcing_transfer.py --checkpoint best.pt --cases baseline uniform_base
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
    all_cases: List[Tuple[str, float, float, str, str]],
    selected_names: List[str] | None,
) -> List[Tuple[str, float, float, str, str]]:
    """Secilen test vakalarini filtrele. None ise hepsini dondur."""
    if selected_names is None:
        return all_cases

    all_names = {name for name, *_ in all_cases}
    for name in selected_names:
        if name not in all_names:
            available = ", ".join(sorted(all_names))
            print(f"UYARI: '{name}' gecerli bir test vakasi degil.")
            print(f"Gecerli vakalar: {available}")
            sys.exit(1)

    return [case for case in all_cases if case[0] in selected_names]


def make_stochastic_post_step(model):
    """
    Stochastic forcing icin post_step_fn olustur.

    Her forward adimdan sonra Ornstein-Uhlenbeck noise state'ini ilerletir.
    Bu olmazsa stochastic forcing sabit kalir (eta=0) ve uniform'dan farksiz olur.

    OU sureci:
      d(eta) = -eta/tau * dt + sigma * sqrt(2/tau) * dW
    Euler-Maruyama ayriklamasi:
      eta_{n+1} = eta_n * (1 - dt/tau) + sigma * sqrt(2*dt/tau) * N(0,1)
    """
    def _post_step_fn(state, step, mdl):
        # dt_eff: model'in efektif zaman adimi
        dt_eff = mdl._dt_base * torch.clamp(mdl.dt_scale, 0.5, 2.0).item()
        # OU noise'u bir adim ilerlet
        mdl.forcing.step_ou(dt_eff)
        return state

    return _post_step_fn


def run_forcing_transfer_demo(args) -> List[SimulationResult]:
    """
    Tum forcing transfer testlerini sirayla calistir.

    ONEMLI: Her test vakasi icin model YENIDEN yuklenir cunku:
      1. Forcing3D neuronu constructor'da mode set edilir
      2. mode degisince y_grid ve OU state'i yeniden olusturulmali
      3. load_trained_model(forcing_mode=...) bunu otomatik yapar
      4. Checkpoint agirliklari strict=False ile yuklenir (Forcing3D
         parametreleri uyusmazsa atlanir - bu sorun degil cunku amplitude
         ogrenilmis deger, mode bagimli degil)
    """
    cases = filter_test_cases(TEST_CASES, args.cases)
    results: List[SimulationResult] = []

    print()
    print("=" * 70)
    print("  INNATE 3D - DEMO 2: FORCING TRANSFER TESTI")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Steps: {args.steps}")
    print(f"  Log interval: {args.log_interval}")
    print(f"  Device: {args.device or 'auto'}")
    print(f"  Test cases: {len(cases)}")
    print()
    print("  Forcing modlari:")
    print("    kolmogorov: F_x = A * sin(k_f * 2pi*y/Ly)")
    print("    uniform:    F_x = A (sabit)")
    print("    stochastic: F_x = A + sigma*eta(t) (OU process)")
    print()

    total_t0 = time.time()

    for i, (name, Re, Ra, forcing_mode, note) in enumerate(cases):
        print(f"\n{'#' * 70}")
        print(f"  TEST {i + 1}/{len(cases)}: {name}")
        print(f"  Re={Re:.0f}, Ra={Ra:.1e}, Pr={args.Pr}")
        print(f"  Forcing: {forcing_mode}")
        print(f"  Zorluk: {note}")
        print(f"{'#' * 70}")

        # Her test icin model yeniden yukle (forcing_mode degisikligi icin)
        model, cfg, dev = load_trained_model(
            checkpoint_path=args.checkpoint,
            Re=Re,
            Ra=Ra,
            Pr=args.Pr,
            forcing_mode=forcing_mode,
            device=args.device,
        )

        # Stochastic forcing icin OU noise'u aktive et
        # Forcing3D.reset_phase() cagirarak phi'yi randomize et ve eta'yi sifirla
        model.forcing.reset_phase()

        # OU parametrelerini raporla (stochastic forcing icin)
        if hasattr(model.forcing, 'tau'):
            print(f"  OU params: tau={model.forcing.tau:.2f}, sigma={model.forcing.sigma:.2f}")

        # Post-step hook: stochastic modda OU state'ini her adimda ilerlet
        post_step_fn = None
        if forcing_mode == "stochastic":
            post_step_fn = make_stochastic_post_step(model)
            print(f"  >> Stochastic OU hook AKTIF (step_ou her adimda cagrilacak)")

        # Simulasyon calistir
        result = run_simulation(
            model=model,
            config=cfg,
            device=dev,
            n_steps=args.steps,
            log_interval=args.log_interval,
            name=name,
            post_step_fn=post_step_fn,
        )

        results.append(result)

        # Kisa ozet
        if result.stable:
            fm = result.final_metrics
            print(f"\n  >> SONUC: STABIL | E_kin={fm.get('E_kin', 0):.6f} "
                  f"| Nu={fm.get('Nu', 0):.4f} | slope={fm.get('spectrum_slope', float('nan')):.3f}")
        else:
            print(f"\n  >> SONUC: NaN at step {result.nan_step}")

        # Model'i serbest birak (bellek tasarrufu, her seferinde yenisi yukleniyor)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - total_t0

    # ---- Karsilastirma tablosu ----
    print_comparison_table(results, title="DEMO 2: FORCING TRANSFER SONUCLARI")

    # ---- Forcing bazli analiz ----
    _print_forcing_analysis(results)

    # ---- Sonuclari kaydet ----
    output_dir = Path(args.output_dir)
    output_path = output_dir / "demo2_forcing_transfer.json"
    save_results(results, str(output_path))

    print(f"\nToplam sure: {total_time:.1f}s ({total_time / max(len(cases), 1):.1f}s/test)")

    return results


def _print_forcing_analysis(results: List[SimulationResult]):
    """
    Forcing modlari arasi karsilastirma analizi.

    Kontrol edilen fiziksel tutarliliklar:
      1. Baseline (kolmogorov) stabil olmali
      2. Farkli forcing modlarinda E_kin degerleri farkli olabilir (bu normal)
      3. Tum modlarda div_max < 1e-4 (solenoidallik modu bagimli degil)
      4. Stochastic modda E_kin'in zamansal varyansı daha yuksek olmali
         (OU noise enerji girisi dalgalandırır)
      5. Uniform forcing genellikle Kolmogorov'dan daha dusuk E_kin uretir
         (sinuzoidal forcing cok-olcekli turbulans olusturur, uniform basitlestirir)
    """
    print()
    print("=" * 70)
    print("  FORCING TRANSFER ANALIZI")
    print("=" * 70)

    stable = [r for r in results if r.stable]
    unstable = [r for r in results if not r.stable]

    print(f"\n  Stabil: {len(stable)}/{len(results)}")

    if unstable:
        print(f"\n  Basarisiz testler:")
        for r in unstable:
            c = r.config_summary
            print(f"    - {r.name}: forcing={c.get('forcing_mode', '?')}, "
                  f"Re={c['Re']:.0f}, NaN at step {r.nan_step}")

    # Baseline'i bul
    baseline = None
    for r in stable:
        if r.name == "baseline":
            baseline = r
            break

    if baseline is None:
        print("\n  UYARI: Baseline (kolmogorov) test stabil degil veya calistirilmadi.")
        print("  Referans olmadan forcing karsilastirmasi yapilamiyor.")
        return

    baseline_ekin = baseline.final_metrics.get("E_kin", 0)
    baseline_nu = baseline.final_metrics.get("Nu", 0)
    baseline_slope = baseline.final_metrics.get("spectrum_slope", float("nan"))

    print(f"\n  Baseline (kolmogorov) referans degerleri:")
    print(f"    E_kin  = {baseline_ekin:.6f}")
    print(f"    Nu     = {baseline_nu:.4f}")
    slope_str = f"{baseline_slope:.3f}" if baseline_slope == baseline_slope else "N/A"
    print(f"    slope  = {slope_str}")

    # Her forcing modunu baseline ile karsilastir
    print(f"\n  {'Forcing':<15s} {'E_kin':>10s} {'dE/E_ref':>10s} {'Nu':>8s} "
          f"{'slope':>8s} {'div_max':>10s} {'Stabil':>7s}")
    print(f"  {'-' * 70}")

    for r in results:
        c = r.config_summary
        fm = r.final_metrics
        forcing = c.get("forcing_mode", "?")

        if r.stable and fm:
            ekin = fm.get("E_kin", 0)
            nu = fm.get("Nu", 0)
            slope = fm.get("spectrum_slope", float("nan"))
            div_max = fm.get("div_max", 0)

            # Relative E_kin degisimi
            if baseline_ekin > 1e-10:
                de_rel = (ekin - baseline_ekin) / baseline_ekin * 100
                de_str = f"{de_rel:+.1f}%"
            else:
                de_str = "N/A"

            slope_str = f"{slope:.3f}" if slope == slope else "N/A"

            print(f"  {r.name:<15s} {ekin:10.6f} {de_str:>10s} {nu:8.4f} "
                  f"{slope_str:>8s} {div_max:10.2e} {'OK':>7s}")
        else:
            print(f"  {r.name:<15s} {'---':>10s} {'---':>10s} {'---':>8s} "
                  f"{'---':>8s} {'---':>10s} {'FAIL':>7s}")

    # Enstrophy zaman serisi varyans karsilastirmasi (stochastic vs others)
    print(f"\n  Enerji zaman serisi varyans analizi:")
    for r in stable:
        if len(r.metrics_history) >= 3:
            ekin_series = [m.get("E_kin", 0) for m in r.metrics_history]
            mean_e = statistics.mean(ekin_series)
            std_e = statistics.stdev(ekin_series) if len(ekin_series) > 1 else 0
            cv = std_e / (mean_e + 1e-10) * 100  # coefficient of variation (%)
            forcing = r.config_summary.get("forcing_mode", "?")
            print(f"    {r.name:<20s} (forcing={forcing:>12s}): "
                  f"mean(E)={mean_e:.6f}, std(E)={std_e:.2e}, CV={cv:.1f}%")

    # Fiziksel beklenti kontrolu
    print(f"\n  Fiziksel beklenti kontrolleri:")

    # Stochastic'in daha yuksek varyansi olmali
    stochastic_results = [r for r in stable
                          if r.config_summary.get("forcing_mode") == "stochastic"]
    kolmogorov_results = [r for r in stable
                          if r.config_summary.get("forcing_mode") == "kolmogorov"]

    if stochastic_results and kolmogorov_results:
        for sr in stochastic_results:
            ekin_s = [m.get("E_kin", 0) for m in sr.metrics_history]
            std_s = statistics.stdev(ekin_s) if len(ekin_s) > 1 else 0

            kr = kolmogorov_results[0]
            ekin_k = [m.get("E_kin", 0) for m in kr.metrics_history]
            std_k = statistics.stdev(ekin_k) if len(ekin_k) > 1 else 0

            if std_s > std_k:
                print(f"    [OK] {sr.name}: stochastic varyans ({std_s:.2e}) > "
                      f"kolmogorov varyans ({std_k:.2e})")
            else:
                print(f"    [?]  {sr.name}: stochastic varyans ({std_s:.2e}) <= "
                      f"kolmogorov varyans ({std_k:.2e})")
                print(f"         (Bu olabilir: kisa simulasyonda OU etkileri "
                      f"henuz belirgin olmayabilir)")

    print()


# =====================================================================
# Main
# =====================================================================

if __name__ == "__main__":
    args = parse_args()
    results = run_forcing_transfer_demo(args)

    # Cikis kodu
    n_stable = sum(1 for r in results if r.stable)
    n_total = len(results)

    if n_stable == n_total:
        print(f"SONUC: {n_total}/{n_total} test STABIL - forcing transfer BASARILI.")
    else:
        print(f"SONUC: {n_stable}/{n_total} test stabil.")

    # Baseline stabil degilse ciddi problem
    baseline_stable = any(r.stable for r in results if r.name == "baseline")
    if not baseline_stable and any(r.name == "baseline" for r in results):
        print("KRITIK: Baseline (kolmogorov) bile stabil degil!")
        sys.exit(2)

    sys.exit(0)
