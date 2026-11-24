#!/usr/bin/env bash
# N7（裝置端 GMRES）的 g02 量測（GPU3）：建置；預設執行的 L3 逐位元；參考版與 fast 的每步時間；
# nsys 統計 fast 的 kernel、記憶體複製與 CUDA API 呼叫（RAE2822 SA 200 步、ONERA M6 SA 30 步）。
# 在 g02 上執行：bash scripts/n7d_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
P=${N7D_PREFIX:-N7d-g02}
FAST="GNC_LUSGS=color GNC_SGS=color GNC_REDUCE=tree"
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
run bash scripts/build.sh --host-env $E
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-verify \
    --end-default 50 --time-indices 1,2,10,50 --end oneraM6SA=10 --time-indices-for oneraM6SA=1,2,10 \
    --case forwardStep=$C/forwardStep-prepared/simulation \
    --case flatPlateSA=$C/flatPlate-g2000/simulation \
    --case rae2822SA=$C/rae2822-prepared/simulation \
    --case oneraM6SA=$C/oneraM6-prepared/simulation
run bash scripts/variant_compare.sh --host-env $E --label $P-perf200 --end 200 \
    --variant ref= --variant "fast=$FAST" \
    --case flatPlateSA=$C/flatPlate-g2000/simulation --case rae2822SA=$C/rae2822-prepared/simulation
run bash scripts/variant_compare.sh --host-env $E --label $P-perf-onera --end 30 \
    --variant ref= --variant "fast=$FAST" --case oneraM6SA=$C/oneraM6-prepared/simulation
# nsys：同一個 fast 案例再跑一次（場檔寫在另一份副本）
. $E
nsys_case() {
    local R="$RUN_ROOT/$1" c="$2"
    local W="$R/nsys-$c-fast"
    mkdir -p "$W/constant"
    cp -r "$R/$c/fast/system" "$R/$c/fast/0" "$R/$c/fast/gncExpanded" "$W/"
    ln -s "$(realpath "$R/$c/fast/constant/polyMesh")" "$W/constant/polyMesh"
    for f in "$R/$c"/fast/constant/*; do
        [ "$(basename "$f")" = polyMesh ] || cp -r "$f" "$W/constant/"
    done
    run env $FAST CUDA_VISIBLE_DEVICES="$GPU_DEVICE" nsys profile --trace=cuda --stats=false \
        -o "$W" --force-overwrite true "$BUILD_ROOT/bin/gncHisa" -case "$W"
    run nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum --format csv \
        --output "$W" "$W.nsys-rep"
}
nsys_case $P-perf200 rae2822SA
nsys_case $P-perf-onera oneraM6SA
echo "=== done $(date -u +%FT%TZ)"
