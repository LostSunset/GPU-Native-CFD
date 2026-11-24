#!/usr/bin/env bash
# HiSA 的 CPU 計時（論文的分母，g02 的 AMD EPYC 9684X）：perf_compare.sh 以同一個穩態變體
# 跑 HiSA 序列、MPI（metis 分解）與 gncHisa（參考版），再以 variant_compare.sh 跑 gncHisa -fast
# 的同樣步數。主機不閒置時的時間只作參考（RUN_INFO 記下量測前後的 load average）。
# 在 g02 上執行：bash scripts/cpu_baseline_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
P=${CPU_PREFIX:-CPU-g02}
FAST="GNC_LUSGS=color GNC_SGS=color GNC_REDUCE=tree GNC_JAC=auto"
NP=${CPU_NP_LIST:-8,32}
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
run bash scripts/perf_compare.sh --host-env $E --label $P-small --end 200 --np-list "$NP" \
    --case flatPlateSA=$C/flatPlate-g2000/simulation --case rae2822SA=$C/rae2822-prepared/simulation
run bash scripts/perf_compare.sh --host-env $E --label $P-onera --end 20 --np-list "$NP" \
    --case oneraM6SA=$C/oneraM6-prepared/simulation
run bash scripts/perf_compare.sh --host-env $E --label $P-finner --end 5 --np-list "$NP" \
    --case basicFinnerLaminar=$C/basicFinner-seq5/simulation --laminar basicFinnerLaminar
run bash scripts/variant_compare.sh --host-env $E --label $P-gncfast-small --end 200 --variant "fast=$FAST" \
    --case flatPlateSA=$C/flatPlate-g2000/simulation --case rae2822SA=$C/rae2822-prepared/simulation
run bash scripts/variant_compare.sh --host-env $E --label $P-gncfast-onera --end 20 --variant "fast=$FAST" \
    --case oneraM6SA=$C/oneraM6-prepared/simulation
run bash scripts/variant_compare.sh --host-env $E --label $P-gncfast-finner --end 5 --variant "fast=$FAST" \
    --case basicFinnerLaminar=$C/basicFinner-seq5/simulation --laminar basicFinnerLaminar
echo "=== done $(date -u +%FT%TZ)"
