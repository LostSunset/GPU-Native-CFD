#!/usr/bin/env bash
# ONERA M6（3D、SA）的到收斂比較與力係數（g02）：加上翼面 forceCoeffs 的衍生案例
# （make_onera_forces_case.sh）；力係數的 L3 逐位元（10 步）；gncHisa 參考版與 -fast 依案例的
# endTime 2000 步（殘差歷程、Cl、Cd、CmPitch、每步時間）；HiSA 32 核 MPI 的同樣 2000 步（計時）。
# 在 g02 上執行：bash scripts/onera_conv_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
P=${ONERA_PREFIX:-M6-g02}
FAST="GNC_LUSGS=color GNC_SGS=color GNC_REDUCE=tree GNC_JAC=auto"
M6=/ssd/sclin/gpu-native-cfd/cases/oneraM6-forces/simulation
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
mkdir -p "$(dirname "$(dirname "$M6")")"
[ -d "$M6" ] || run bash scripts/make_onera_forces_case.sh --host-env $E \
    --src $C/oneraM6-prepared/simulation --dest "$M6"
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-L3-forces \
    --end-default 10 --time-indices 1,2,10 --case oneraM6SA="$M6" --forces oneraM6SA=forceCoeffsWing
run bash scripts/variant_compare.sh --host-env $E --label $P-conv --end 2000 \
    --variant ref= --variant "fast=$FAST" --case oneraM6SA="$M6" --forces oneraM6SA=forceCoeffsWing
run bash scripts/perf_compare.sh --host-env $E --label $P-hisa32 --end 2000 --np-list 32 \
    --no-serial --no-gnc --case oneraM6SA="$M6"
echo "=== done $(date -u +%FT%TZ)"
