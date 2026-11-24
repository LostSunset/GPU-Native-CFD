#!/usr/bin/env bash
# kOmegaSST（N4b）的效能（g02，主機忙碌時的參考值）：原本的案例（Poisson 牆距離）上
# HiSA 序列與 MPI、gncHisa 參考版（perf_compare.sh）、gncHisa -fast（variant_compare.sh）、
# 混合式路線（hybrid_onera_g02.sh，HiSA-GPU-Research 的 GMRESGPU MCBSGS-jgpu）。只用 GPU3。
# 在 g02 上執行：bash scripts/sst_perf_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
P=${SST_PREFIX:-SSTP-g02}
FAST="GNC_LUSGS=color GNC_SGS=color GNC_REDUCE=tree GNC_JAC=auto"
NP=${CPU_NP_LIST:-8,32}
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
run bash scripts/perf_compare.sh --host-env $E --label $P-naca --end 200 --np-list "$NP" \
    --case nacaAirfoilSST=$C/nacaAirfoil-prepared/simulation
run bash scripts/perf_compare.sh --host-env $E --label $P-finner --end 5 --np-list "$NP" \
    --case basicFinnerSST=$C/basicFinner-seq5/simulation
run bash scripts/variant_compare.sh --host-env $E --label $P-gncfast-naca --end 200 --variant "fast=$FAST" \
    --case nacaAirfoilSST=$C/nacaAirfoil-prepared/simulation
run bash scripts/variant_compare.sh --host-env $E --label $P-gncfast-finner --end 5 --variant "fast=$FAST" \
    --case basicFinnerSST=$C/basicFinner-seq5/simulation
run env HYBRID_PREFIX=$P-hyb HYBRID_SRC=$C/basicFinner-seq5/simulation HYBRID_NAME=basicFinnerSSTHybrid \
    HYBRID_TAG=finner HYBRID_END=5 bash scripts/hybrid_onera_g02.sh
echo "=== done $(date -u +%FT%TZ)"
