#!/bin/bash
# Re=10K survival map: Cs x dt sweep (frozen mode)
# Soru: Re=10K'da stabil bir INNATE konfigürasyonu var mi?

set -u
cd "$(dirname "$0")"
mkdir -p results/diagnostics/survival_map

LOG=/tmp/survival_map.log
: > "$LOG"

echo "=== Re=10K Survival Map: Cs x dt sweep (frozen) ===" | tee -a "$LOG"
date | tee -a "$LOG"

declare -a CS_LIST=("0.17" "0.30" "0.50")
declare -a DT_LIST=("0.02" "0.01")

for CS in "${CS_LIST[@]}"; do
  for DT in "${DT_LIST[@]}"; do
    TAG="re10k_frozen_cs${CS//./p}_dt${DT//./p}"
    # Step sayisi: dt=0.02 -> 1000 step (~6.7dk), dt=0.01 -> 2000 step (~13dk)
    if [ "$DT" = "0.02" ]; then
      STEPS=1000
    else
      STEPS=2000
    fi
    echo "" | tee -a "$LOG"
    echo "----- Cs=$CS dt=$DT steps=$STEPS  tag=$TAG -----" | tee -a "$LOG"
    date +"start: %H:%M:%S" | tee -a "$LOG"

    python3 -u diagnose_blowup.py \
      --frozen --frozen-cs "$CS" --frozen-prt 0.85 \
      --Re 10000 --dt "$DT" --steps "$STEPS" \
      --tag "$TAG" --device mps \
      --log-interval 200 2>&1 | tee /tmp/diag_${TAG}.log | \
      grep -E "MODE|Parameters|UNSTABLE|Stable: no|Elapsed|NaN at step|^[ ]+[0-9]+ +[0-9]" | tee -a "$LOG"

    date +"end:   %H:%M:%S" | tee -a "$LOG"
  done
done

echo "" | tee -a "$LOG"
echo "=== SURVIVAL MAP OZETI ===" | tee -a "$LOG"
for CS in "${CS_LIST[@]}"; do
  for DT in "${DT_LIST[@]}"; do
    TAG="re10k_frozen_cs${CS//./p}_dt${DT//./p}"
    LINE=$(grep -E "UNSTABLE|Stable: no NaN" /tmp/diag_${TAG}.log | tail -1)
    printf "  Cs=%s dt=%s : %s\n" "$CS" "$DT" "$LINE" | tee -a "$LOG"
  done
done
echo "Cikti: $LOG"
