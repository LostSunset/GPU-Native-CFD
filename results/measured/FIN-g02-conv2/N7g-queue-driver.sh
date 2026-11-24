#!/usr/bin/env bash
# 依序：N7 CUDA Graph 的量測（n7g_g02.sh，含建置）；泛用平台路線在 basicFinner SST 的每步時間
# （spuma_sst_g02.sh）；basicFinner SST 的 1000 步重跑（能量限界實作之後；第一次在第 492 步
# 因限界未實作而停止，FIN-g02-conv/），gncHisa 參考版與 -fast，HiSA 32 核沿用 FIN-g02-hisa32/。
# 在 g02 上執行：bash scripts/n7g_queue_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
FAST="GNC_LUSGS=color GNC_SGS=color GNC_REDUCE=tree GNC_JAC=auto"
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
run bash scripts/n7g_g02.sh
run bash scripts/spuma_sst_g02.sh
run bash scripts/variant_compare.sh --host-env $E --label FIN-g02-conv2 --end 1000 \
    --variant ref= --variant "fast=$FAST" --case basicFinnerSST=$C/basicFinner-seq5/simulation \
    --forces basicFinnerSST=forces
echo "=== all done $(date -u +%FT%TZ)"
