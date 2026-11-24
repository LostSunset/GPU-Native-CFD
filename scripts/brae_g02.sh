#!/usr/bin/env bash
# brae（github.com/simd-ai/brae，AGPL-3.0；OpenFOAM 求解器的 CUDA 重寫，相關研究中最接近本專案的工作）
# 在同一張 L40（g02 GPU3）、同一組 aerofoilNACA0012 網格上的計時與收斂，與 gncRhoSimple 比較。
# 使用者 2026-10-24 同意在 g02 clone brae；brae 只執行、不併入本專案。
#
# brae：/ssd/sclin/gpu-native-cfd/ext/brae（build/brae_rhoSimpleFoam，sm_89，CUDA 13.2；SCOTCH 7.0.11 在
# ext/scotch-7.0.11），以 BRAE_RHOSIMPLEFOAM_MIRROR=cuda 執行 GPU 版；FOAM_ETC 指向 OpenFOAM v2606 的 etc。
# 案例是 rhosimple_perf.sh 的 GAMG 原設定（--gamg-case），只改 endTime 與 writeInterval（= endTime，只在最後
# 寫一次）；constant/polyMesh 以符號連結。brae 不印每迭代時間：每個網格跑 n1、n2 兩個迭代數，
# 每迭代時間 =（牆鐘(n2) - 牆鐘(n1)）/（n2 - n1），扣掉啟動、讀網格與最後的寫出（與 gnc 的迴圈時間、
# OpenFOAM 的 ExecutionTime 差同一個意義）。另跑 16,000 格到收斂（案例原本的 residualControl），寫出的
# 場與 N6-perf-conv 的 OpenFOAM 單核解比較。
# 在 g02 上執行：bash scripts/brae_g02.sh > ... 2>&1
set -u
# 本腳本所在的 checkout（紀錄中複製的是實際執行的這一份）
self="$(realpath "$0")"
cd "$(dirname "$self")/.."
B=/ssd/sclin/gpu-native-cfd/ext/brae
BIN=$B/build/brae_rhoSimpleFoam
G=/ssd/sclin/gpu-native-cfd/cases
OUT=/ssd/sclin/gpu-native-cfd/runs/${LABEL:-BRAE-perf}
FOAM=/opt/atlas-openfoam/current/OpenFOAM-v2606
O1=/ssd/sclin/gpu-native-cfd/runs/N6-perf-conv/r1/of1
[ -x "$BIN" ] || { echo "ERROR: $BIN missing" >&2; exit 2; }
[ -e "$OUT" ] && { echo "ERROR: $OUT exists" >&2; exit 2; }
mkdir -p "$OUT"
uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' '$1 == 3 {print $2}')"
if [ "$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid")" -ne 0 ]; then
    echo "ERROR: GPU3 busy" >&2; exit 3
fi
{
    echo "date_utc=$(date -u +%FT%TZ)"
    echo "brae_commit=$(git -C "$B" rev-parse HEAD)"
    echo "brae_commit_date=$(git -C "$B" log -1 --format=%cI)"
    echo "brae_build=$(grep -m1 -E 'CMAKE_BUILD_TYPE:' "$B/build/CMakeCache.txt")"
    echo "brae_cuda_archs=$(grep -m1 -E 'CMAKE_CUDA_ARCHITECTURES:' "$B/build/CMakeCache.txt")"
    echo "nvcc=$(/usr/local/cuda/bin/nvcc --version | tail -1)"
    echo "sha256_brae_rhoSimpleFoam=$(sha256sum "$BIN" | awk '{print $1}')"
    echo "gpu=$(nvidia-smi -i 3 --query-gpu=name,driver_version --format=csv,noheader)"
    echo "host_load=$(cut -d' ' -f1-3 /proc/loadavg)"
} > "$OUT/RUN_INFO"
cp "$self" "$OUT/"

mesh() { if [ "$1" = 1 ]; then echo aerofoilNACA0012; else echo "aerofoilNACA0012-r$1"; fi; }
prep() {   # 來源、目的、endTime
    mkdir -p "$2/constant"
    cp -r "$1/system" "$1/0" "$2/"
    for f in "$1"/constant/*; do
        [ "$(basename "$f")" = polyMesh ] && continue
        cp -r "$f" "$2/constant/"
    done
    ln -s "$1/constant/polyMesh" "$2/constant/polyMesh"
    sed -i -e "s/^endTime .*/endTime         $3;/" -e "s/^writeInterval .*/writeInterval   $3;/" "$2/system/controlDict"
}
runb() {   # 案例目錄 -> 牆鐘秒數
    local a b
    a=$(date +%s.%N)
    (cd "$1" && CUDA_VISIBLE_DEVICES=3 BRAE_RHOSIMPLEFOAM_MIRROR=cuda FOAM_ETC=$FOAM/etc WM_PROJECT_DIR=$FOAM \
        "$BIN" -case "$1" > log.run 2> log.err)
    echo "rc=$?" >> "$1/log.run"
    b=$(date +%s.%N)
    python3 -c "print('%.3f' % ($b - $a))"
}
reached() { grep -oE '^Time = [0-9]+' "$1/log.run" | tail -1 | grep -oE '[0-9]+'; }
piters() { grep -oE 'pIters [0-9]+' "$1/log.run" | awk '{s += $2; n++} END {if (n) printf "%.1f", s/n}'; }

# 每個網格重複 REPEAT 次（預設 3），每迭代時間取中位數：讀網格的時間有起伏，所以 n2 - n1 取得夠大
REPEAT=${REPEAT:-3}
printf 'case\tnCells\tn1\tn2\trepeats\tper_iter_each_s\tper_iter_s\tp_iters_mean\n' > "$OUT/SUMMARY.tsv"
# 網格：BRAE_MESHES（預設 1 2 4 8；N6e 用 16 32）
for r in ${BRAE_MESHES:-1 2 4 8}; do
    case $r in 1) n1=20; n2=420 ;; 2) n1=10; n2=210 ;; 4) n1=10; n2=110 ;; 8) n1=10; n2=70 ;;
               16) n1=5; n2=35 ;; 32) n1=3; n2=23 ;; *) echo "no iteration counts for r$r" >&2; exit 2 ;; esac
    src="$G/$(mesh $r)/simulation"
    nc=$((16000*r*r))
    pers=()
    for k in $(seq 1 "$REPEAT"); do
        w=()
        for n in $n1 $n2; do
            d="$OUT/r$r/k$k/n$n"
            prep "$src" "$d" "$n"
            echo "== r$r brae repeat $k, $n iterations"
            w+=("$(runb "$d")")
            got="$(reached "$d")"
            [ "$got" = "$n" ] || echo "WARNING: r$r k$k n$n reached $got"
            rm -rf "${d:?}/$n"   # 定迭代數的場不保留
        done
        pers+=("$(python3 -c "print('%.6f' % ((${w[1]} - ${w[0]})/($n2 - $n1)))")")
    done
    med="$(printf '%s\n' "${pers[@]}" | sort -g | awk '{a[NR] = $1} END {print a[int((NR + 1)/2)]}')"
    printf 'r%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$r" "$nc" "$n1" "$n2" "$REPEAT" "$(IFS=,; echo "${pers[*]}")" \
        "$med" "$(piters "$OUT/r$r/k1/n$n2")" >> "$OUT/SUMMARY.tsv"
done

# 16,000 格到收斂（案例原本的 endTime 1000、residualControl 1e-4；與 rhosimple_perf.sh 相同，迴圈中不寫場，
# 收斂時寫一次）；BRAE_CONV=0 時略過
if [ "${BRAE_CONV:-1}" = 1 ]; then
    d="$OUT/conv/brae"
    prep "$G/$(mesh 1)/simulation" "$d" 1000
    sed -i "s/^writeInterval .*/writeInterval   1001000;/" "$d/system/controlDict"
    echo "== conv brae"
    wc="$(runb "$d")"
    last="$(cd "$d" && ls -d [0-9]* | grep -E '^[0-9]+$' | sort -n | tail -1)"
    printf 'case\titerations\twall_s\tp_iters_mean\n%s\t%s\t%s\t%s\n' conv "$(reached "$d")" "$wc" "$(piters "$d")" \
        > "$OUT/SUMMARY.conv.tsv"
    to="$(cd "$O1" && ls -d [0-9]* | grep -E '^[0-9]+$' | sort -n | tail -1)"
    python3 -I scripts/field_diff.py "$O1/$to" "$d/$last" > "$OUT/conv/fielddiff.of1-brae.txt" 2>&1
    echo "--- of1 ($to) vs brae ($last)"; cat "$OUT/conv/fielddiff.of1-brae.txt"
fi
cat "$OUT/SUMMARY.tsv" "$OUT/SUMMARY.conv.tsv" 2>/dev/null
echo "=== done $(date -u +%FT%TZ)"
