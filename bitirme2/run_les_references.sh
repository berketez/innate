#!/bin/bash
# LES referans verisi uretimi — INNATE v2 curriculum icin
# 3 eksik Re noktasi: 15000, 20000, 5000 (riskli)
# + Re=7000, 10000 slope'larini k=(8,24) ile yeniden hesaplamak icin
# Ra=1e5, Pr=0.71, 60K step, Cs=0.17
# Grid: 96x160x64 (LES-converged, metrikler grid-bagimsiz)
# spectrum.pt 32 bin — train.py min(model,ref) ile handle eder (satir 616)
#
# Kullanim: 4090'da calistir
#   cd C:\Users\berke\nsneuron\bitirme2
#   nohup bash run_les_references.sh > les_log.txt 2>&1 &
#
# Tahmini sure: her biri ~2 saat, toplam ~6-10 saat

set -e

COMMON="--Ra 1e5 --n_steps 60000 --log_interval 500 --snapshot_interval 5000 --Cs 0.17 --device cuda --seed 42"

echo "======================================"
echo "INNATE v2 LES Referans Verisi Uretimi"
echo "Grid: 96x160x64 (default, LES-converged)"
echo "$(date)"
echo "======================================"

# --- 1. Re=15000 (yeni, dusuk Ri) ---
echo ""
echo "[1/3] Re=15000 basliyor... $(date)"
python les_solver.py --Re 15000 $COMMON \
    --cfl 0.5 --dt_max 0.02 --damping_safety 2.0 \
    --save_dir les_reference/Re15000_Ra1e5
echo "[1/3] Re=15000 TAMAMLANDI $(date)"

# --- 2. Re=20000 (yeni, en dusuk Ri) ---
echo ""
echo "[2/3] Re=20000 basliyor... $(date)"
python les_solver.py --Re 20000 $COMMON \
    --cfl 0.5 --dt_max 0.02 --damping_safety 2.0 \
    --save_dir les_reference/Re20000_Ra1e5
echo "[2/3] Re=20000 TAMAMLANDI $(date)"

# --- 3. Re=5000 (RISKLI — Ri=0.0056, yuksek damping) ---
echo ""
echo "[3/3] Re=5000 basliyor (yuksek damping)... $(date)"
python les_solver.py --Re 5000 $COMMON \
    --cfl 0.4 --dt_max 0.015 --damping_safety 5.0 \
    --save_dir les_reference/Re5000_Ra1e5
echo "[3/3] Re=5000 TAMAMLANDI $(date)"

echo ""
echo "======================================"
echo "TAMAMLANDI — $(date)"
echo "======================================"
echo "Sonuc dizinleri:"
echo "  les_reference/Re15000_Ra1e5/"
echo "  les_reference/Re20000_Ra1e5/"
echo "  les_reference/Re5000_Ra1e5/"
echo ""
echo "NOT: Mevcut Re=7000 ve Re=10000 verileri (32 bin) uyumlu."
echo "train.py spectrum_shape_loss min(model,ref) ile handle eder."
echo "Slope degerleri spectrum.pt'den k=(8,24) ile yeniden hesaplanmali."
