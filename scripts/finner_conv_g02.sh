#!/usr/bin/env bash
# basicFinner（3D、kOmegaSST、218 萬格）的力與到 1000 步的比較（g02）：力的 L3 逐位元
# （meshWave 變體，10 步，case 的 forces function）；原本的案例（Poisson 牆距離）依案例的
# endTime 1000 步：gncHisa 參考版與 -fast（variant_compare.sh），HiSA 32 核 MPI（計時與力，
# perf_compare.sh --forces）。只用 GPU3。
# 在 g02 上執行：bash scripts/finner_conv_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
P=${FINNER_PREFIX:-FIN-g02}
FAST="GNC_LUSGS=color GNC_SGS=color GNC_REDUCE=tree GNC_JAC=auto"
FINNER=$C/basicFinner-seq5/simulation
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-L3-forces \
    --end-default 10 --time-indices 1,2,10 --case basicFinnerSST=$FINNER --meshwave basicFinnerSST \
    --forces basicFinnerSST=forces
run bash scripts/variant_compare.sh --host-env $E --label $P-conv --end 1000 \
    --variant ref= --variant "fast=$FAST" --case basicFinnerSST=$FINNER --forces basicFinnerSST=forces
run bash scripts/perf_compare.sh --host-env $E --label $P-hisa32 --end 1000 --np-list 32 \
    --no-serial --no-gnc --case basicFinnerSST=$FINNER --forces basicFinnerSST=forces
echo "=== done $(date -u +%FT%TZ)"
