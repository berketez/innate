#!/usr/bin/env python3
"""
demo3_domain_transfer.py - INNATE 3D Domain ve Grid Transfer Testi

INNATE'in spectral parametreleri (Cs, Pr_t, dt_scale vb.) fiziksel anlamli,
grid'e bagli degil. Bu script farkli domain boyutlari ve grid cozunurlukleri
ile egitilmis modeli test eder.

Fiziksel arka plan:
  - SpectralOps3DAniso her (Nx, Ny, Nz, Lx, Ly, Lz) kombinasyonu icin
    farkli dalga sayilari (kx, ky, kz) hesaplar.
  - Learnable parametreler (advection_modulator, cs_mid, pr_t vb.)
    fiziksel buyuklukler -- belirli bir grid'e bagimli degil.
  - strict=False ile checkpoint yukleme: SpectralOps buffer'lari
    eslesmeyebilir (boyut farki), ama bunlar register_buffer oldugundan
    yeni grid icin otomatik olusturulur.

Test vakalari:
  | # | Domain     | Grid        | Not                              |
  |---|------------|-------------|----------------------------------|
  | 1 | 6x10x4     | 96x160x64   | Egitim konfigurasyonu (referans) |
  | 2 | 12x20x8    | 96x160x64   | 2x domain, ayni grid → kaba LES |
  | 3 | 3x5x2      | 96x160x64   | 0.5x domain, ayni grid → ince LES|
  | 4 | 6x10x4     | 64x96x48    | Ayni domain, kaba grid           |
  | 5 | 6x10x4     | 128x192x96  | Ayni domain, ince grid           |

Kullanim:
  python tests/demo3_domain_transfer.py --checkpoint results/checkpoints/best.pt
  python tests/demo3_domain_transfer.py --checkpoint best.pt --steps 200 --device cpu

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

from demo_utils import (
    SimulationResult,
    load_trained_model,
    run_simulation,
    compute_physics_metrics,
    save_results,
    print_comparison_table,
    print_final_report,
)


# =====================================================================
# Test Vakasi Tanimlari
# =====================================================================

# Her test vakasi icin parametreler
# (isim, domain_boyutlari, grid_boyutlari, aciklama)
TEST_CASES = [
    {
        "name": "Default (referans)",
        "Lx": 6.0,  "Ly": 10.0, "Lz": 4.0,
        "Nx": 96,   "Ny": 160,  "Nz": 64,
        "description": "Egitim konfigurasyonu. Tum sonuclar buna gore karsilastirilir.",
    },
    {
        "name": "Buyuk domain (2x)",
        "Lx": 12.0, "Ly": 20.0, "Lz": 8.0,
        "Nx": 96,   "Ny": 160,  "Nz": 64,
        "description": (
            "2x domain, ayni grid noktasi sayisi → dx 2 kat buyuk."
            " Kaba LES simulasyonu. Daha buyuk yapilar cozumlenir,"
            " kucuk olcekler kaybedilir. SGS modeli daha cok is yapmali."
        ),
    },
    {
        "name": "Kucuk domain (0.5x)",
        "Lx": 3.0,  "Ly": 5.0,  "Lz": 2.0,
        "Nx": 96,   "Ny": 160,  "Nz": 64,
        "description": (
            "0.5x domain, ayni grid → dx 2 kat kucuk. Daha ince LES."
            " Daha fazla turbulans yapisi cozumlenir (DNS'e yakin)."
            " SGS modeli daha az etkili olmali."
        ),
    },
    {
        "name": "Kaba grid (64x96x48)",
        "Lx": 6.0,  "Ly": 10.0, "Lz": 4.0,
        "Nx": 64,   "Ny": 96,   "Nz": 48,
        "description": (
            "Ayni domain, daha az grid noktasi → daha kaba cozunurluk."
            " SGS modelinin daha agresif filtrelemesi gerekir."
            " Stabilite bozulabilir (CFL yukselebilir)."
        ),
    },
    {
        "name": "Ince grid (128x192x96)",
        "Lx": 6.0,  "Ly": 10.0, "Lz": 4.0,
        "Nx": 128,  "Ny": 192,  "Nz": 96,
        "description": (
            "Ayni domain, daha fazla grid noktasi → daha ince cozunurluk."
            " DNS'e yaklasir. SGS'in etkisi azalmali."
            " DIKKAT: 2.36M nokta, memory-yogun (~6GB+ unified RAM)."
        ),
    },
]


def estimate_memory_mb(Nx: int, Ny: int, Nz: int, n_fields: int = 6) -> float:
    """
    Tahmini memory kullanimi (MB).

    Bir state = 5 alan (u,v,w,p,theta), her biri [1, Nx, Ny, Nz] float32.
    Model hesaplamalarinda ~3x-5x ekstra buffer gerekir.
    SpectralOps buffer'lari (kx, ky, kz, k_squared vb.) ~6 alan.

    Args:
        Nx, Ny, Nz: Grid boyutlari
        n_fields: Toplam alan sayisi (state + ara hesaplar)

    Returns:
        Tahmini MB
    """
    # Bir float32 tensor [1, Nx, Ny, Nz]
    single_field = Nx * Ny * Nz * 4 / (1024 * 1024)  # bytes -> MB
    # State (5 alan) + SpectralOps (6 buffer) + ara hesaplar (~10 alan)
    total_fields = 5 + 6 + 10  # state + ops + intermediates
    return single_field * total_fields


def print_transfer_analysis(results: List[SimulationResult]):
    """
    Domain transfer sonuclarinin fiziksel analizini yazdir.

    Grid spacing degistiginde:
      - dx buyurse → LES filtre genisligi artar → SGS daha onemli
      - dx kuculdukce → DNS'e yaklasir → SGS ihmal edilebilir
      - CFL = u_max * dt / dx → dx kuculdukce CFL artar (stabilite riski)

    Domain degistiginde:
      - Buyuk domain → daha buyuk girdap yapilari → enerji dagilimi degisir
      - Kucuk domain → confinement etkisi → yapay bloklama olabilir
    """
    if not results:
        return

    print()
    print("=" * 80)
    print("  DOMAIN TRANSFER ANALIZI")
    print("=" * 80)

    # Referans (ilk vaka = default)
    ref = results[0]
    ref_m = ref.final_metrics
    if not ref_m:
        print("  Referans vaka basarisiz, analiz yapilamadi.")
        return

    ref_Ekin = ref_m.get("E_kin", 0)
    ref_Nu = ref_m.get("Nu", 0)
    ref_CFL = ref_m.get("CFL", 0)
    ref_slope = ref_m.get("spectrum_slope", float("nan"))

    print(f"\n  Referans: {ref.name}")
    print(f"    E_kin={ref_Ekin:.6f}, Nu={ref_Nu:.4f}, CFL={ref_CFL:.4f}")
    print()

    for r in results[1:]:
        m = r.final_metrics
        if not m:
            print(f"  {r.name}: BASARISIZ (NaN at step {r.nan_step})")
            continue

        # Goreceli farklar (referansa gore)
        E_ratio = m.get("E_kin", 0) / (ref_Ekin + 1e-20)
        Nu_ratio = m.get("Nu", 0) / (ref_Nu + 1e-20) if ref_Nu != 0 else float("inf")
        CFL_ratio = m.get("CFL", 0) / (ref_CFL + 1e-20) if ref_CFL != 0 else float("inf")

        # Grid spacing degisimi
        c = r.config_summary
        dx_test = c["Lx"] / c["Nx"]
        dx_ref = ref.config_summary["Lx"] / ref.config_summary["Nx"]
        dx_ratio = dx_test / dx_ref

        n_points_test = c["Nx"] * c["Ny"] * c["Nz"]
        n_points_ref = (ref.config_summary["Nx"] * ref.config_summary["Ny"]
                        * ref.config_summary["Nz"])

        print(f"  {r.name}:")
        print(f"    Grid: {c['Nx']}x{c['Ny']}x{c['Nz']} ({n_points_test:,} nokta)")
        print(f"    Domain: {c['Lx']}x{c['Ly']}x{c['Lz']}")
        print(f"    dx degisimi: {dx_ratio:.2f}x (referansa gore)")
        print(f"    Stabil: {'EVET' if r.stable else 'HAYIR'}")
        print(f"    E_kin: {m.get('E_kin', 0):.6f} ({E_ratio:.2f}x referans)")
        print(f"    Nu:    {m.get('Nu', 0):.4f} ({Nu_ratio:.2f}x referans)")
        print(f"    CFL:   {m.get('CFL', 0):.4f} ({CFL_ratio:.2f}x referans)")
        slope = m.get("spectrum_slope", float("nan"))
        if slope == slope:  # NaN check
            print(f"    Spektral egim: {slope:.3f} (hedef: -1.667)")
        print(f"    Sure: {r.wall_time:.1f}s ({r.wall_time / max(r.n_steps, 1):.3f}s/step)")
        print()

    # Ozet tablo
    print("  TRANSFER BASARISI OZETI:")
    n_stable = sum(1 for r in results if r.stable)
    print(f"    Stabil: {n_stable}/{len(results)}")

    # Eger tum vakalar stabil ise → iyi transfer
    if n_stable == len(results):
        print("    SONUC: Model tum domain/grid konfigurasyonlarinda stabil.")
        print("    Fiziksel parametreler grid'den bagimsiz gorunuyor.")
    else:
        failed = [r.name for r in results if not r.stable]
        print(f"    BASARISIZ VAKALAR: {', '.join(failed)}")
        print("    Transfer sinirli. Bazi konfigurasyonlarda stabilite bozuluyor.")
    print()


# =====================================================================
# Ana Fonksiyon
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="INNATE 3D Domain ve Grid Transfer Testi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ornekler:
  # Tam test (5 vaka x 500 adim)
  python tests/demo3_domain_transfer.py --checkpoint best.pt

  # Hizli test (200 adim)
  python tests/demo3_domain_transfer.py --checkpoint best.pt --steps 200

  # Sadece belirli vakalar
  python tests/demo3_domain_transfer.py --checkpoint best.pt --cases 0 1 3

  # Ince grid'i atla (memory yetersizse)
  python tests/demo3_domain_transfer.py --checkpoint best.pt --skip-fine
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
        "--output-dir", type=str, default="results/demo3_domain_transfer",
        help="Sonuc dizini (default: results/demo3_domain_transfer)",
    )
    parser.add_argument(
        "--log-interval", type=int, default=50,
        help="Her kac adimda metrik kaydedilecek (default: 50)",
    )
    parser.add_argument(
        "--cases", type=int, nargs="+", default=None,
        help="Sadece belirli test vakalarini calistir (0-indexed). Ornek: --cases 0 1 3",
    )
    parser.add_argument(
        "--skip-fine", action="store_true",
        help="Ince grid vakasini atla (memory yetersizse)",
    )
    parser.add_argument(
        "--Re", type=float, default=5000.0,
        help="Reynolds number (default: 5000)",
    )
    parser.add_argument(
        "--Ra", type=float, default=1e6,
        help="Rayleigh number (default: 1e6)",
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

    # Test vakalarini sec
    cases = TEST_CASES.copy()
    if args.skip_fine:
        # Ince grid vakasini cikar (son eleman)
        cases = [c for c in cases if "128" not in str(c.get("Nx", ""))]
        print("NOT: Ince grid vakasi atlanacak (--skip-fine).")

    if args.cases is not None:
        selected = []
        for idx in args.cases:
            if 0 <= idx < len(TEST_CASES):
                selected.append(TEST_CASES[idx])
            else:
                print(f"UYARI: Gecersiz vaka indeksi: {idx} (0-{len(TEST_CASES)-1} arasi)")
        cases = selected

    if not cases:
        print("HATA: Hic test vakasi secilmedi.")
        sys.exit(1)

    # ================================================================
    # Tum vakalari calistir
    # ================================================================

    print()
    print("=" * 80)
    print("  INNATE 3D — DOMAIN & GRID TRANSFER TESTI")
    print("=" * 80)
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Re={args.Re:.0f}, Ra={args.Ra:.1e}")
    print(f"  Steps/vaka : {args.steps}")
    print(f"  Vaka sayisi: {len(cases)}")
    print(f"  Output     : {out_dir}")
    print()

    all_results: List[SimulationResult] = []
    total_t0 = time.time()

    for i, case in enumerate(cases):
        print(f"\n{'#'*80}")
        print(f"# VAKA {i+1}/{len(cases)}: {case['name']}")
        print(f"# {case['description']}")
        print(f"{'#'*80}")

        # Memory tahmini
        mem_mb = estimate_memory_mb(case["Nx"], case["Ny"], case["Nz"])
        n_points = case["Nx"] * case["Ny"] * case["Nz"]
        dx = case["Lx"] / case["Nx"]
        dy = case["Ly"] / case["Ny"]
        dz = case["Lz"] / case["Nz"]

        print(f"\n  Grid: {case['Nx']}x{case['Ny']}x{case['Nz']} = {n_points:,} nokta")
        print(f"  Domain: {case['Lx']}x{case['Ly']}x{case['Lz']}")
        print(f"  dx={dx:.4f}, dy={dy:.4f}, dz={dz:.4f}")
        print(f"  Tahmini memory: ~{mem_mb:.0f} MB")

        # Model yukle — her vaka icin yeni SpectralOps olusur
        try:
            model, cfg, dev = load_trained_model(
                checkpoint_path=str(ckpt_path),
                Re=args.Re,
                Ra=args.Ra,
                Nx=case["Nx"],
                Ny=case["Ny"],
                Nz=case["Nz"],
                Lx=case["Lx"],
                Ly=case["Ly"],
                Lz=case["Lz"],
                device=args.device,
            )
        except RuntimeError as e:
            print(f"\n  MODEL YUKLEME HATASI: {e}")
            print(f"  Bu vaka atlanacak.")
            # Bos sonuc ekle
            all_results.append(SimulationResult(
                name=case["name"],
                config_summary={
                    "Re": args.Re, "Ra": args.Ra, "Pr": 0.71,
                    "Ri": args.Ra / (args.Re**2 * 0.71),
                    "nu": 1.0 / args.Re,
                    "Nx": case["Nx"], "Ny": case["Ny"], "Nz": case["Nz"],
                    "Lx": case["Lx"], "Ly": case["Ly"], "Lz": case["Lz"],
                    "forcing_mode": "kolmogorov",
                },
                n_steps=0,
                wall_time=0.0,
                stable=False,
                nan_step=0,
                metrics_history=[],
                final_metrics={},
            ))
            continue

        # Parametre sayisi kontrolu
        n_params = model.count_parameters()
        print(f"  Model parametreleri: {n_params}")
        print(f"  SpectralOps grid: {model.ops.Nx}x{model.ops.Ny}x{model.ops.Nz}")

        # Simulasyonu calistir
        result = run_simulation(
            model=model,
            config=cfg,
            device=dev,
            n_steps=args.steps,
            log_interval=args.log_interval,
            name=case["name"],
        )
        all_results.append(result)

        # Memory temizle (sonraki vaka icin yer ac)
        del model
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_time = time.time() - total_t0

    # ================================================================
    # Sonuclari raporla
    # ================================================================

    # Karsilastirma tablosu
    print_comparison_table(
        all_results,
        title="DOMAIN & GRID TRANSFER — KARSILASTIRMA TABLOSU",
    )

    # Detayli transfer analizi
    print_transfer_analysis(all_results)

    # JSON kaydi
    json_path = str(out_dir / "demo3_results.json")
    save_results(all_results, json_path)

    # Toplam sure
    print(f"\nToplam sure: {total_time:.1f}s ({total_time/60:.1f} dakika)")
    print(f"Stabil vakalar: {sum(1 for r in all_results if r.stable)}/{len(all_results)}")

    # Cs x Delta grid-bagimliligi uyarisi
    print()
    print("ONEMLI NOT: Smagorinsky Cs x Delta grid-bagimliligi")
    print("=" * 60)
    print("  nu_sgs = (Cs * Delta)^2 * |S|")
    print("  Cs ogrenilmis sabit, Delta = dx grid spacing.")
    print("  Domain 2x buyutulunce dx 2x artar -> nu_sgs 4x artar!")
    print("  Bu over-dissipation etkisi Demo 3 sonuclarinda gorulebilir.")
    print("  Dynamic Smagorinsky (Germano, 1991) bu sorunu cozer")
    print("  ama mevcut model static Smagorinsky kullaniyor.")

    # Grid spacing tablosu
    print("\n  GRID SPACING KARSILASTIRMASI:")
    print(f"  {'Vaka':<30s} {'dx':>8s} {'dy':>8s} {'dz':>8s} {'Toplam':>12s}")
    print(f"  {'-'*72}")
    for i, r in enumerate(all_results):
        c = r.config_summary
        dx_val = c["Lx"] / c["Nx"]
        dy_val = c["Ly"] / c["Ny"]
        dz_val = c["Lz"] / c["Nz"]
        n_pts = c["Nx"] * c["Ny"] * c["Nz"]
        print(f"  {r.name:<30s} {dx_val:8.4f} {dy_val:8.4f} {dz_val:8.4f} {n_pts:>12,}")
    print()


if __name__ == "__main__":
    main()
