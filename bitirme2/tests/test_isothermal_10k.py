#!/usr/bin/env python3
"""
INNATE Isothermal 10K Step Doğrulama Testi
==========================================
3 gerçekçi mühendislik senaryosu, termal nöronlar dondurulmuş (Ra=0).

Senaryolar:
  1. Rüzgar Türbini Wake (Re=100K) — actuator disk drag
  2. Veri Merkezi Koridor (Re=50K) — sadece momentum, hava akışı
  3. Helikopter Pervane Downwash (Re=200K) — yüksek TKE bölgesi

Her senaryoda ölçülen:
  - Enerji spektrumu slope (beklenen: -1.5 ~ -1.8)
  - TKE (turbulent kinetic energy)
  - Türbülans yoğunluğu TI = u_rms / U_mean
  - max|div(u)| (incompressibility)
  - CFL sayısı
  - LES kalite indeksi (LES_IQ)
  - Kolmogorov ölçeği eta ve dx/eta oranı

Kullanım:
  python tests/test_isothermal_10k.py [--steps 10000] [--scenario all|wake|datacenter|helicopter]
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import json
from pathlib import Path
from typing import Callable, Dict, List

import torch
import numpy as np

# -- path setup --
_this_dir = Path(__file__).resolve().parent
_project_dir = _this_dir.parent
sys.path.insert(0, str(_this_dir))
sys.path.insert(0, str(_project_dir))

from demo_utils import (
    load_trained_model,
    run_simulation,
    compute_energy_spectrum,
    compute_spectrum_slope,
    compute_physics_metrics,
    save_results,
)
from model import ThermalFluidState

# Default checkpoint
DEFAULT_CKPT = str(_project_dir / "results" / "checkpoints" / "checkpoint_epoch001250.pt")


# =====================================================================
# Senaryo 1: Rüzgar Türbini Wake (Wind Turbine Wake)
# =====================================================================

def setup_wind_turbine_wake(steps: int, ckpt: str):
    """
    Re=100K, Ra=0 (isothermal), uniform forcing + actuator disk drag.
    Türbin arkasında hız açığı ve TKE artışı beklenir.
    """
    model, cfg, dev = load_trained_model(
        ckpt, Re=100000, Ra=1e-10, forcing_mode="uniform"
    )

    # Actuator disk post-step callback
    d = cfg.domain
    x = torch.linspace(0, d.Lx, d.Nx, device=dev).view(1, d.Nx, 1, 1)
    x_disk = d.Lx / 4.0
    C_T = 0.75
    sigma = 0.3
    G = torch.exp(-0.5 * ((x - x_disk) / sigma) ** 2) / (sigma * math.sqrt(2 * math.pi))
    U_inf = 1.0

    def post_step_fn(state, model, step):
        dt_eff = cfg.physics.dt * 1.0
        drag = -C_T * 0.5 * U_inf**2 * G * dt_eff
        new_u = state.u + drag
        return ThermalFluidState(
            u=new_u, v=state.v, w=state.w, p=state.p, theta=state.theta
        )

    return {
        "name": "1. Rüzgar Türbini Wake (Re=100K, isothermal)",
        "model": model,
        "config": cfg,
        "device": dev,
        "steps": steps,
        "post_step_fn": post_step_fn,
        "expected": {
            "spectrum_slope": (-2.0, -1.2),
            "TI": (0.05, 0.60),
            "div_max": (0, 1e-4),
            "CFL": (0, 0.5),
        },
    }


# =====================================================================
# Senaryo 2: Veri Merkezi Koridor Akışı (Data Center Corridor)
# =====================================================================

def setup_data_center(steps: int, ckpt: str):
    """
    Re=50K, Ra=0 (isothermal — sadece momentum/hava akışı).
    Sunucu koridorundan geçen zorlanmış hava.
    """
    model, cfg, dev = load_trained_model(
        ckpt, Re=50000, Ra=1e-10, forcing_mode="uniform"
    )

    return {
        "name": "2. Veri Merkezi Koridor (Re=50K, isothermal)",
        "model": model,
        "config": cfg,
        "device": dev,
        "steps": steps,
        "post_step_fn": None,
        "expected": {
            "spectrum_slope": (-2.0, -1.2),
            "TI": (0.03, 0.50),
            "div_max": (0, 1e-4),
            "CFL": (0, 0.5),
        },
    }


# =====================================================================
# Senaryo 3: Helikopter Pervane Downwash
# =====================================================================

def setup_helicopter_downwash(steps: int, ckpt: str):
    """
    Re=200K, Ra=0 (isothermal), Kolmogorov forcing.
    Yüksek Re, güçlü türbülans, tip vortex benzeri yapılar.
    """
    model, cfg, dev = load_trained_model(
        ckpt, Re=200000, Ra=1e-10, forcing_mode="kolmogorov"
    )

    return {
        "name": "3. Helikopter Downwash (Re=200K, isothermal)",
        "model": model,
        "config": cfg,
        "device": dev,
        "steps": steps,
        "post_step_fn": None,
        "expected": {
            "spectrum_slope": (-2.0, -1.2),
            "TI": (0.05, 0.70),
            "div_max": (0, 1e-4),
            "CFL": (0, 0.5),
        },
    }


# =====================================================================
# Test Runner
# =====================================================================

def run_scenario(scenario: dict) -> dict:
    """Tek senaryo çalıştır, sonuçları döndür."""
    name = scenario["name"]
    model = scenario["model"]
    cfg = scenario["config"]
    dev = scenario["device"]
    steps = scenario["steps"]
    post_step_fn = scenario.get("post_step_fn")
    expected = scenario.get("expected", {})

    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"  Re={cfg.physics.Re:.0f}, Ra={cfg.physics.Ra:.1e}, dt={cfg.physics.dt}")
    print(f"  Grid: {cfg.domain.Nx}x{cfg.domain.Ny}x{cfg.domain.Nz}")
    print(f"  Steps: {steps}")
    print(f"{'='*70}")

    # Initial condition
    state = model.create_initial_condition(batch_size=1, device=dev)

    # Run
    t0 = time.time()
    metrics_log = []
    log_interval = max(1, steps // 20)  # ~20 log noktası

    with torch.no_grad():
        for i in range(1, steps + 1):
            state = model(state)

            # Post-step callback (actuator disk vb.)
            if post_step_fn is not None:
                state = post_step_fn(state, model, i)

            # Metric logging
            if i % log_interval == 0 or i == steps:
                m = compute_physics_metrics(model, state, cfg)
                metrics_log.append({"step": i, **m})

                # NaN check
                if math.isnan(m.get("E_kin", 0)):
                    print(f"  !! NaN at step {i} — ABORTING")
                    break

                elapsed = time.time() - t0
                eta = elapsed / i * (steps - i)
                print(f"  Step {i:>6d}/{steps} | E_kin={m['E_kin']:.6f} | "
                      f"slope={m.get('spectrum_slope', 0):.3f} | "
                      f"TKE={m.get('TKE', 0):.6f} | "
                      f"div={m.get('div_max', 0):.2e} | "
                      f"CFL={m.get('CFL', 0):.3f} | "
                      f"ETA={eta:.0f}s")

    wall_time = time.time() - t0
    final = metrics_log[-1] if metrics_log else {}

    # Değerlendirme
    print(f"\n  --- SONUÇLAR ({name}) ---")
    print(f"  Toplam süre:      {wall_time:.1f}s ({wall_time/60:.1f} dk)")

    results = {
        "name": name,
        "Re": cfg.physics.Re,
        "Ra": cfg.physics.Ra,
        "steps": steps,
        "wall_time": wall_time,
        "stable": not math.isnan(final.get("E_kin", float("nan"))),
        "final_metrics": final,
        "metrics_history": metrics_log,
        "pass_fail": {},
    }

    # Beklenti karşılaştırması
    for key, (lo, hi) in expected.items():
        val = final.get(key, None)
        if val is not None:
            passed = lo <= val <= hi
            results["pass_fail"][key] = "PASS" if passed else "FAIL"
            status = "✓" if passed else "✗"
            print(f"  {key:>20s}: {val:.6f}  [{lo}, {hi}]  {status}")
        else:
            results["pass_fail"][key] = "N/A"

    return results


def main():
    parser = argparse.ArgumentParser(description="INNATE Isothermal 10K Doğrulama")
    parser.add_argument("--steps", type=int, default=10000, help="Simülasyon adımı")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--scenario", type=str, default="all",
                        choices=["all", "wake", "datacenter", "helicopter"])
    args = parser.parse_args()

    print("=" * 70)
    print("  INNATE Isothermal 10K Step Doğrulama")
    print("  Termal nöronlar dondurulmuş (Ra ≈ 0)")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Steps: {args.steps}")

    scenarios = []
    if args.scenario in ("all", "wake"):
        scenarios.append(setup_wind_turbine_wake(args.steps, args.checkpoint))
    if args.scenario in ("all", "datacenter"):
        scenarios.append(setup_data_center(args.steps, args.checkpoint))
    if args.scenario in ("all", "helicopter"):
        scenarios.append(setup_helicopter_downwash(args.steps, args.checkpoint))

    all_results = []
    for sc in scenarios:
        result = run_scenario(sc)
        all_results.append(result)

    # Final özet tablo
    print("\n" + "=" * 70)
    print("  GENEL ÖZET TABLO")
    print("=" * 70)
    print(f"  {'Senaryo':<45s} {'Slope':>8s} {'TKE':>10s} {'TI':>8s} {'div':>10s} {'CFL':>6s} {'Durum':>6s}")
    print("  " + "-" * 95)
    for r in all_results:
        m = r["final_metrics"]
        stable = "OK" if r["stable"] else "FAIL"
        print(f"  {r['name']:<45s} "
              f"{m.get('spectrum_slope', 0):>8.3f} "
              f"{m.get('TKE', 0):>10.6f} "
              f"{m.get('TI', 0):>8.4f} "
              f"{m.get('div_max', 0):>10.2e} "
              f"{m.get('CFL', 0):>6.3f} "
              f"{stable:>6s}")

    # JSON kaydet
    out_path = _project_dir / "results" / "isothermal_10k_validation.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Sonuçlar kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
