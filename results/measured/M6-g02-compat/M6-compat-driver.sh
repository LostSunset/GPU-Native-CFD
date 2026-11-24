#!/usr/bin/env bash
# ONERA M6：HiSA v2606 的 3D 黏性應力轉置問題（N-013 的 transposeAliasing）對力係數的影響（g02）。
# 建置；-fast 以 HiSA 的實際行為（預設）與原始碼的正確結果（GNC_FOAM_COMPAT=off）各跑 2000 步。
# 衍生案例由 onera_conv_g02.sh 建立（make_onera_forces_case.sh）。
# 在 g02 上執行：bash scripts/onera_compat_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
P=${ONERA_PREFIX:-M6-g02}
FAST="GNC_LUSGS=color GNC_SGS=color GNC_REDUCE=tree GNC_JAC=auto"
M6=/ssd/sclin/gpu-native-cfd/cases/oneraM6-forces/simulation
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
[ -d "$M6" ] || { echo "ERROR: $M6 missing (run onera_conv_g02.sh)"; exit 2; }
run bash scripts/build.sh --host-env $E
run bash scripts/variant_compare.sh --host-env $E --label $P-compat --end 2000 \
    --variant "fast=$FAST" --variant "fastFixed=$FAST GNC_FOAM_COMPAT=off" \
    --case oneraM6SA="$M6" --forces oneraM6SA=forceCoeffsWing
echo "=== done $(date -u +%FT%TZ)"
