#!/usr/bin/env bash
# 論文的第二條路線（混合式，HiSA-GPU-Research 的 GMRESGPU）在 ONERA M6 SA 上的每步時間（g02）：
# HiSA 主機迴圈，Jacobian（LaxFriedrichsGPU、laplacianGPU）與線性求解（GMRESGPU，MCBSGS 預條件）
# 在 GPU；設定同 HiSA-GPU-Research 的 G2-CLEAN-2 最佳變體 MCBSGS-jgpu。只用 GPU3
# （CUDA_VISIBLE_DEVICES，函式庫的 device 0）。HiSA 序列（perf_compare.sh --no-gnc）。
# 在 g02 上執行：bash scripts/hybrid_onera_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
P=${HYBRID_PREFIX:-HYB-g02}
# 案例（預設 ONERA M6）：HYBRID_SRC 為原案例，HYBRID_NAME 為紀錄中的案例名稱，HYBRID_TAG 為紀錄名的後綴
SRC=${HYBRID_SRC:-$C/oneraM6-prepared/simulation}
NAME=${HYBRID_NAME:-oneraM6Hybrid}
TAG=${HYBRID_TAG:-onera}
if [ "$NAME" = oneraM6Hybrid ]; then
    D=/ssd/sclin/gpu-native-cfd/cases/oneraM6-hybrid-MCBSGS-jgpu/simulation
else
    D=/ssd/sclin/gpu-native-cfd/cases/$NAME-MCBSGS-jgpu/simulation
fi
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
# shellcheck disable=SC1090
. "$E"
foam() { ( set +u; . "$OPENFOAM_BASHRC" >/dev/null 2>&1; set -u; "$@" ); }
if [ ! -d "$D" ]; then
    src=$SRC
    mkdir -p "$D/constant"
    cp -r "$src/system" "$src/0" "$D/"
    for f in "$src"/constant/*; do
        b="$(basename "$f")"
        if [ "$b" = polyMesh ]; then ln -s "$(realpath "$f")" "$D/constant/polyMesh"; else cp -r "$f" "$D/constant/$b"; fi
    done
    S="$(foam foamDictionary -entry flowSolver/solver -value "$D/system/fvSolution")"
    body="$(foam foamDictionary -entry "flowSolver/$S" "$D/system/fvSolution" | sed '1d')"
    foam foamDictionary -entry flowSolver/solver -set GMRESGPU "$D/system/fvSolution" > /dev/null
    foam foamDictionary -entry flowSolver/GMRESGPU -set "$body" "$D/system/fvSolution" > /dev/null
    for kv in inviscidJacobian=LaxFriedrichsGPU viscousJacobian=laplacianGPU preconditioner=MCBSGS \
              matrixPrecision=double packing=gpu device=0; do
        foam foamDictionary -entry "flowSolver/GMRESGPU/${kv%%=*}" -set "${kv#*=}" "$D/system/fvSolution" > /dev/null
    done
    foam foamDictionary -entry libs -set '("libhisaGpuSolver.so")' "$D/system/controlDict" > /dev/null
    { echo "source=$src"; echo "variant=MCBSGS-jgpu (HiSA-GPU-Research G2-CLEAN-2)"; \
      diff -u "$src/system/fvSolution" "$D/system/fvSolution"; } > "$D/DERIVED"
fi
run env CUDA_VISIBLE_DEVICES="$GPU_DEVICE" bash scripts/perf_compare.sh --host-env $E --label $P-$TAG \
    --end "${HYBRID_END:-50}" --no-gnc --case "$NAME=$D"
cp "$D/DERIVED" "$RUN_ROOT/$P-$TAG/DERIVED.case" 2>/dev/null
echo "=== done $(date -u +%FT%TZ)"
