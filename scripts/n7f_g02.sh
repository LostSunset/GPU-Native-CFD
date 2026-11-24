#!/usr/bin/env bash
# N7（FP32 Jacobian，GNC_JAC=float）的 g02 量測（GPU3）：建置；預設執行的 L3 逐位元；
# fast、fast32（fast 加 GNC_JAC=float）、fastauto（GNC_JAC=auto，即 gncHisa -fast）的每步時間；
# 到收斂與力係數的比較。
# 在 g02 上執行：bash scripts/n7f_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
P=${N7F_PREFIX:-N7f-g02}
FAST="GNC_LUSGS=color GNC_SGS=color GNC_REDUCE=tree"
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
run bash scripts/build.sh --host-env $E
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-verify \
    --end-default 50 --time-indices 1,2,10,50 --end oneraM6SA=10 --time-indices-for oneraM6SA=1,2,10 \
    --case forwardStep=$C/forwardStep-prepared/simulation \
    --case flatPlateSA=$C/flatPlate-g2000/simulation \
    --case rae2822SA=$C/rae2822-prepared/simulation \
    --case oneraM6SA=$C/oneraM6-prepared/simulation
run bash scripts/variant_compare.sh --host-env $E --label $P-perf-onera --end 30 \
    --variant ref= --variant "fast=$FAST" --variant "fast32=$FAST GNC_JAC=float" \
    --variant "fastauto=$FAST GNC_JAC=auto" --case oneraM6SA=$C/oneraM6-prepared/simulation
run bash scripts/variant_compare.sh --host-env $E --label $P-perf200 --end 200 \
    --variant ref= --variant "fast=$FAST" --variant "fast32=$FAST GNC_JAC=float" \
    --variant "fastauto=$FAST GNC_JAC=auto" \
    --case flatPlateSA=$C/flatPlate-g2000/simulation --case rae2822SA=$C/rae2822-prepared/simulation
run bash scripts/variant_compare.sh --host-env $E --label $P-conv-flat --end 2000 \
    --variant "fast=$FAST" --variant "fast32=$FAST GNC_JAC=float" \
    --case flatPlateSA=$C/flatPlate-g2000/simulation
run bash scripts/variant_compare.sh --host-env $E --label $P-coeffs --end 4000 \
    --variant "fast=$FAST" --variant "fast32=$FAST GNC_JAC=float" \
    --case rae2822SA=$C/rae2822-prepared/simulation --forces rae2822SA=forceCoeffsCompressible
echo "=== done $(date -u +%FT%TZ)"
