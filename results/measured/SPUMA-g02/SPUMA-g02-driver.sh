#!/usr/bin/env bash
# 論文的第三條路線（泛用平台：HiSA 建在 SPUMA 上，D-026 P1）在與 gncHisa 相同的案例上的每步時間
# （g02 GPU3）：以 D-026 的 p1_runs.sh（/ssd/sclin/gpu-platform/scripts）跑 Atlas v2606 hisa（CPU 單核）
# 與 SPUMA 上的 hisa（GPU3，-pool fixedSizeMemoryPool）。案例：ONERA M6 SA（官方設定）10 步、
# basicFinner 的層流變體 3 步（衍生案例，網格以符號連結）。
# 在 g02 上執行：bash scripts/spuma_compare_g02.sh > ... 2>&1
set -u
C=/home/sclin/hisa-gpu-research-cases
CASES=/ssd/sclin/gpu-native-cfd/cases
OUT=/ssd/sclin/gpu-native-cfd/runs/${SPUMA_PREFIX:-SPUMA-g02}
P1=/ssd/sclin/gpu-platform/scripts/p1_runs.sh
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
[ -e "$OUT" ] && { echo "ERROR: $OUT exists"; exit 2; }
mkdir -p "$OUT" "$CASES"
F=$CASES/basicFinner-laminar/simulation
if [ ! -d "$F" ]; then
    src=$C/basicFinner-seq5/simulation
    mkdir -p "$F/constant"
    cp -r "$src/system" "$src/0" "$F/"
    for f in "$src"/constant/*; do
        b="$(basename "$f")"
        if [ "$b" = polyMesh ]; then ln -s "$(realpath "$f")" "$F/constant/polyMesh"; else cp -r "$f" "$F/constant/$b"; fi
    done
    sed -i -E 's/^simulationType([[:space:]]+)[^;]+;/simulationType\1laminar;/' "$F/constant/turbulenceProperties"
    echo "source=$src; simulationType laminar" > "$F/DERIVED"
fi
{
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "uptime=$(uptime)"
    sha256sum "$P1"
} > "$OUT/RUN_INFO"
run bash "$P1" "$OUT/oneraM6SA" "$C/oneraM6-prepared/simulation" 10 8
run bash "$P1" "$OUT/basicFinnerLaminar" "$F" 3 16
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/RUN_INFO"
echo "=== done $(date -u +%FT%TZ)"
