#!/usr/bin/env python3
"""
run_all_demos.py - Tum Generalization Demo'larini Sirali Calistir

INNATE 3D Mixed Convection modelinin generalization yeteneklerini
test eden 5 demo scriptini tek komutla calistirip toplu rapor uretir.

Demolar:
  Demo 1: Zero-shot parameter transfer (farkli Re/Ra)
  Demo 2: Forcing transfer (uniform, stochastic)
  Demo 3: Domain ve grid transfer (farkli boyutlar)
  Demo 4: Limit problemler (saf RB, Kolmogorov, TGV3D)
  Demo 5: Muhendislik problemleri (wake, urban, river, datacenter, ABL)

Kullanim:
  # Tum demolar
  python tests/run_all_demos.py --checkpoint results/checkpoints/best.pt

  # Secici calistirma
  python tests/run_all_demos.py --checkpoint best.pt --demos 1 2 3

  # Kisa test (az adim)
  python tests/run_all_demos.py --checkpoint best.pt --steps 100

Yazar: Berke Tezgocen (tasarim), Claude (implementasyon)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


_this_dir = Path(__file__).resolve().parent


def run_demo(demo_num: int, checkpoint: str, steps: int, device: str,
             output_dir: str) -> dict:
    """
    Tek bir demo scriptini subprocess olarak calistir.

    Returns:
        {"demo": int, "success": bool, "returncode": int,
         "wall_time": float, "error": str | None}
    """
    script_map = {
        1: "demo1_parameter_transfer.py",
        2: "demo2_forcing_transfer.py",
        3: "demo3_domain_transfer.py",
        4: "demo4_limit_problems.py",
        5: "demo5_engineering.py",
    }

    script = _this_dir / script_map[demo_num]
    if not script.exists():
        return {
            "demo": demo_num,
            "success": False,
            "returncode": -1,
            "wall_time": 0.0,
            "error": f"Script not found: {script}",
        }

    cmd = [
        sys.executable, str(script),
        "--checkpoint", checkpoint,
        "--steps", str(steps),
        "--output-dir", output_dir,
    ]
    if device:
        cmd.extend(["--device", device])

    print(f"\n{'#' * 70}")
    print(f"#  DEMO {demo_num}: {script.stem}")
    print(f"{'#' * 70}")

    t0 = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_this_dir.parent),
            timeout=3600,  # 1 saat max per demo
        )
        returncode = result.returncode
        error = None
    except subprocess.TimeoutExpired:
        returncode = -2
        error = "Timeout (1 hour)"
    except Exception as e:
        returncode = -3
        error = str(e)

    wall_time = time.time() - t0

    success = returncode == 0
    status = "OK" if success else f"FAIL (rc={returncode})"
    print(f"\nDemo {demo_num} {status} [{wall_time:.1f}s]")

    return {
        "demo": demo_num,
        "success": success,
        "returncode": returncode,
        "wall_time": wall_time,
        "error": error,
    }


def main():
    parser = argparse.ArgumentParser(
        description="INNATE 3D Generalization Demo Runner"
    )
    parser.add_argument(
        "--checkpoint", required=True,
        help="Egitilmis model checkpoint dosyasi"
    )
    parser.add_argument(
        "--demos", nargs="+", type=int, default=[1, 2, 3, 4, 5],
        choices=[1, 2, 3, 4, 5],
        help="Calistirilacak demo numaralari (default: hepsi)"
    )
    parser.add_argument(
        "--steps", type=int, default=500,
        help="Her demo icin adim sayisi (default: 500)"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device override (cuda/mps/cpu)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results/demos",
        help="Sonuc dizini"
    )
    args = parser.parse_args()

    # Checkpoint kontrol
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print("=" * 70)
    print("  INNATE 3D MIXED CONVECTION -- GENERALIZATION DEMO SUITE")
    print("=" * 70)
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Demos:      {args.demos}")
    print(f"  Steps:      {args.steps}")
    print(f"  Device:     {args.device or 'auto'}")
    print(f"  Output:     {args.output_dir}")
    print()

    # Demolari sirala ve calistir
    results = []
    total_t0 = time.time()

    for demo_num in sorted(args.demos):
        r = run_demo(demo_num, args.checkpoint, args.steps,
                     args.device, args.output_dir)
        results.append(r)

    total_time = time.time() - total_t0

    # Final rapor
    print()
    print("=" * 70)
    print("  FINAL REPORT")
    print("=" * 70)
    print()
    print(f"  {'Demo':<8s} {'Status':<10s} {'Time':>10s} {'Note':<30s}")
    print(f"  {'-'*8} {'-'*10} {'-'*10} {'-'*30}")

    n_ok = 0
    for r in results:
        status = "OK" if r["success"] else "FAIL"
        time_str = f"{r['wall_time']:.1f}s"
        note = r.get("error", "") or ""
        if r["returncode"] == 2:
            note = "interpolation tests failed"
        print(f"  Demo {r['demo']:<4d} {status:<10s} {time_str:>10s} {note:<30s}")
        if r["success"]:
            n_ok += 1

    print()
    print(f"  TOTAL: {n_ok}/{len(results)} demos passed  [{total_time:.1f}s]")
    print()

    # Exit code
    if n_ok == len(results):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
