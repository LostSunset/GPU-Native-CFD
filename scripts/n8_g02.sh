#!/usr/bin/env bash
# N8：HiSA 範例中最大的網格 modifiedFinner（3,940,188 格，Hexpress；runs 表第 8 列：馬赫 2.5、攻角 1°，
# kOmegaSST、Poisson 牆距離）的紀錄（g02，只用 GPU3），比照 basicFinner（218 萬格）：
#   MF-g02-L3          逐位元：SST（meshWave 變體）10 步、含 forces；層流 10 步
#   MF-g02-sst／-lam   每步時間：HiSA 序列、8、32 核 MPI 與 gncHisa 參考版（perf_compare.sh）；
#   MF-g02-gncfast-*   gncHisa -fast（variant_compare.sh），5 步
#   MF-g02-hyb-sst     混合式路線（hybrid_onera_g02.sh，HiSA-GPU-Research 的 GMRESGPU MCBSGS-jgpu）
#   MF-g02-spuma-sst   泛用平台路線（HiSA 建在 SPUMA 上，D-026 的 p1_runs.sh），3 步
#   MF-g02-conv        原本的案例到 endTime 1000：gncHisa 參考版與 -fast、HiSA 32 核（力係數）
# 案例由 scripts/cases/fetch_hisa_example.sh 與 prepare_basicFinner.sh --seq 8 準備
# （/home/sclin/hisa-gpu-research-cases/modifiedFinner-seq8）。
# 在 g02 上執行：bash scripts/n8_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
P=${N8_PREFIX:-MF-g02}
FAST="GNC_LUSGS=color GNC_SGS=color GNC_REDUCE=tree GNC_JAC=auto"
MF=$C/modifiedFinner-seq8/simulation
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
run bash scripts/build.sh --host-env $E
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-L3 \
    --end-default 10 --time-indices 1,2,10 \
    --case modifiedFinnerSST=$MF --meshwave modifiedFinnerSST --forces modifiedFinnerSST=forces \
    --case modifiedFinnerLaminar=$MF --laminar modifiedFinnerLaminar
run bash scripts/perf_compare.sh --host-env $E --label $P-sst --end 5 --np-list 8,32 \
    --case modifiedFinnerSST=$MF
run bash scripts/variant_compare.sh --host-env $E --label $P-gncfast-sst --end 5 --variant "fast=$FAST" \
    --case modifiedFinnerSST=$MF
run bash scripts/perf_compare.sh --host-env $E --label $P-lam --end 5 --np-list 8,32 \
    --case modifiedFinnerLaminar=$MF --laminar modifiedFinnerLaminar
run bash scripts/variant_compare.sh --host-env $E --label $P-gncfast-lam --end 5 --variant "fast=$FAST" \
    --case modifiedFinnerLaminar=$MF --laminar modifiedFinnerLaminar
run env HYBRID_PREFIX=$P-hyb HYBRID_SRC=$MF HYBRID_NAME=modifiedFinnerSSTHybrid HYBRID_TAG=sst HYBRID_END=5 \
    bash scripts/hybrid_onera_g02.sh
run env SPUMA_PREFIX=$P-spuma-sst SPUMA_NAME=modifiedFinnerSST SPUMA_SRC=$MF bash scripts/spuma_sst_g02.sh
run bash scripts/variant_compare.sh --host-env $E --label $P-conv --end 1000 \
    --variant ref= --variant "fast=$FAST" --case modifiedFinnerSST=$MF --forces modifiedFinnerSST=forces
run bash scripts/perf_compare.sh --host-env $E --label $P-hisa32 --end 1000 --np-list 32 \
    --no-serial --no-gnc --case modifiedFinnerSST=$MF --forces modifiedFinnerSST=forces
echo "=== done $(date -u +%FT%TZ)"
