#!/usr/bin/env bash
# N6（gncRhoSimple）的效能紀錄（g02 GPU3，與 n6_g02.sh 同一個建置）：
#   N6-perf     aerofoilNACA0012 的四個網格（16,000、64,000、256,000、1,024,000 格）固定迭代數的
#               每迭代時間：gnc（參考次序版，p 為 PCG＋DIC）、gncfast（-fast，多色 DIC）、
#               gncamg（-fast，原案例的 GAMG 以 PCG＋AMG）、OpenFOAM 單核（GAMG、PCG）、
#               OpenFOAM 32 核 MPI（GAMG）、SPUMA（GAMG）
#   N6-perf-conv 16,000 格到收斂（residualControl）的時間：gnc、gncamg、OpenFOAM 單核與 32 核
# 主機 CPU 有其他使用者的 MPI 作業（load 約 112／192 核），CPU 的數字是在這個負載下量的。
# 在 g02 上執行：bash scripts/n6_perf_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
G=/ssd/sclin/gpu-native-cfd/cases
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
mesh() { if [ "$1" = 1 ]; then echo aerofoilNACA0012; else echo "aerofoilNACA0012-r$1"; fi; }
P=${LABEL_PREFIX:-N6}
run bash scripts/rhosimple_perf.sh --host-env $E --label $P-perf-r1 --iters 200 --iters-for spuma=50 \
    --variants gnc,gncfast,gncamg,of1,of1pcg,of32,spuma \
    --case "r1=$G/$(mesh 1)-pcg/simulation" --gamg-case "r1=$G/$(mesh 1)/simulation"
run bash scripts/rhosimple_perf.sh --host-env $E --label $P-perf-r2 --iters 100 --iters-for spuma=20 \
    --variants gnc,gncfast,gncamg,of1,of1pcg,of32,spuma \
    --case "r2=$G/$(mesh 2)-pcg/simulation" --gamg-case "r2=$G/$(mesh 2)/simulation"
run bash scripts/rhosimple_perf.sh --host-env $E --label $P-perf-r4 --iters 50 \
    --iters-for of1=20 --iters-for of1pcg=20 --iters-for spuma=10 \
    --variants gnc,gncfast,gncamg,of1,of1pcg,of32,spuma \
    --case "r4=$G/$(mesh 4)-pcg/simulation" --gamg-case "r4=$G/$(mesh 4)/simulation"
run bash scripts/rhosimple_perf.sh --host-env $E --label $P-perf-r8 --iters 30 \
    --iters-for gnc=10 --iters-for of1=10 --iters-for of1pcg=10 --iters-for spuma=5 \
    --variants gnc,gncfast,gncamg,of1,of1pcg,of32,spuma \
    --case "r8=$G/$(mesh 8)-pcg/simulation" --gamg-case "r8=$G/$(mesh 8)/simulation"
run bash scripts/rhosimple_perf.sh --host-env $E --label $P-perf-conv --iters 1000 \
    --variants gnc,gncamg,of1,of32 \
    --case "r1=$G/$(mesh 1)-pcg/simulation" --gamg-case "r1=$G/$(mesh 1)/simulation"
# 收斂解的差（最後一個時間目錄；field_diff.py 只量測不判定）
C=/ssd/sclin/gpu-native-cfd/runs/$P-perf-conv/r1
last() { (cd "$1" && ls -d [0-9]* 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1); }
for pair in gnc:gncamg gnc:of1 of1:gncamg of1:of32; do
    a="${pair%%:*}"; b="${pair#*:}"
    ta="$(last "$C/$a")"; tb="$(last "$C/$b")"
    if [ -n "$ta" ] && [ -n "$tb" ]; then
        python3 -I scripts/field_diff.py "$C/$a/$ta" "$C/$b/$tb" > "$C/fielddiff.$a-$b.txt" 2>&1
        echo "--- $a ($ta) vs $b ($tb)"; cat "$C/fielddiff.$a-$b.txt"
    fi
done
echo "=== done $(date -u +%FT%TZ)"
