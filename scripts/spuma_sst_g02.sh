#!/usr/bin/env bash
# 泛用平台路線（HiSA 建在 SPUMA 上，D-026 P1）在原本的 basicFinner SST 案例上的每步時間（g02 GPU3）：
# 以 D-026 的 p1_runs.sh 跑 Atlas v2606 hisa（CPU 單核）與 SPUMA 上的 hisa（GPU3，
# -pool fixedSizeMemoryPool），3 步。同 spuma_compare_g02.sh。
# 在 g02 上執行：bash scripts/spuma_sst_g02.sh > ... 2>&1
set -u
C=/home/sclin/hisa-gpu-research-cases
OUT=/ssd/sclin/gpu-native-cfd/runs/${SPUMA_PREFIX:-SPUMA-g02-sst}
P1=/ssd/sclin/gpu-platform/scripts/p1_runs.sh
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
[ -e "$OUT" ] && { echo "ERROR: $OUT exists"; exit 2; }
mkdir -p "$OUT"
{
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uptime=$(uptime)"
    sha256sum "$P1"
} > "$OUT/RUN_INFO"
# 案例（預設 basicFinner SST）：SPUMA_NAME 為紀錄中的案例名稱，SPUMA_SRC 為案例
run bash "$P1" "$OUT/${SPUMA_NAME:-basicFinnerSST}" "${SPUMA_SRC:-$C/basicFinner-seq5/simulation}" 3 16
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/RUN_INFO"
echo "=== done $(date -u +%FT%TZ)"
