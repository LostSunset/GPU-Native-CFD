#!/usr/bin/env bash
# N-027（效能版的 AMG 套用格式與單次啟動歸約）之後的 g02 紀錄（同一個建置）：
#   1. 正式建置
#   2. 單次啟動的 sumTree 與多次啟動版逐位元相同（gncSumBench，GNC_SUMTREE_CHECK=1 時每次呼叫都比對，
#      不同就中止）：n = 1、7、2048、2049、16000、2179881、4194304（2048²）、4194305（退回多次啟動）
#   3. 參考版不受影響：gncRhoSimple 對 rhoSimpleFoam 的 L2／L3、gncHisa SST／SA 的 L3（逐位元）
#   4. gncRhoSimple 三個變體在四個網格的計時、16,000 格到收斂與對 OpenFOAM 單核解的差
# OpenFOAM、SPUMA、brae 不依賴本專案的建置，沿用 N6-perf-*、BRAE-perf。
# 在 g02 上執行：bash scripts/n6c_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
G=/ssd/sclin/gpu-native-cfd/cases
R=/ssd/sclin/gpu-native-cfd/runs
P=N6c
NACA=$C/nacaAirfoil-prepared/simulation
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
mesh() { if [ "$1" = 1 ]; then echo aerofoilNACA0012; else echo "aerofoilNACA0012-r$1"; fi; }
run bash scripts/build.sh --host-env $E
mkdir -p "$R/$P-sumcheck"
cp /ssd/sclin/gpu-native-cfd/build/BUILD_INFO "$R/$P-sumcheck/"
run env CUDA_VISIBLE_DEVICES=3 GNC_SUMTREE_CHECK=1 /ssd/sclin/gpu-native-cfd/build/bin/gncSumBench \
    --n 1,7,2048,2049,16000,2179881,4194304,4194305 --repeat 3 > "$R/$P-sumcheck/log.sumbench" 2>&1
echo "sumbench rc=$? (0: every call identical)" | tee -a "$R/$P-sumcheck/log.sumbench"
run bash scripts/rhosimple_verify.sh --host-env $E --label $P-L3 --dump-steps 1,2,10,100,668 \
    --case aerofoilNACA0012=$G/aerofoilNACA0012-pcg/simulation
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-reg-SST-L3 \
    --end-default 200 --time-indices 1,2,10,50,100,200 \
    --case nacaAirfoilSST=$NACA --meshwave nacaAirfoilSST
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-reg-SA-L3 \
    --end-default 200 --time-indices 1,2,10,50,100,200 \
    --case rae2822SA=$C/rae2822-prepared/simulation
for r in 1 2 4 8; do
    case $r in 1) n=200 ;; 2) n=100 ;; 4) n=50 ;; 8) n=30 ;; esac
    g=""; [ $r = 8 ] && g="--iters-for gnc=10"
    run bash scripts/rhosimple_perf.sh --host-env $E --label $P-perf-r$r --iters $n $g \
        --variants gnc,gncfast,gncamg \
        --case "r$r=$G/$(mesh $r)-pcg/simulation" --gamg-case "r$r=$G/$(mesh $r)/simulation"
done
run bash scripts/rhosimple_perf.sh --host-env $E --label $P-perf-conv --iters 1000 \
    --variants gnc,gncamg \
    --case "r1=$G/$(mesh 1)-pcg/simulation" --gamg-case "r1=$G/$(mesh 1)/simulation"
# 收斂解的差：對 N6-perf-conv 的 OpenFOAM 單核（GAMG）
C1=$R/$P-perf-conv/r1
O1=$R/N6-perf-conv/r1/of1
last() { (cd "$1" && ls -d [0-9]* 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1); }
for a in gnc gncamg; do
    ta="$(last "$C1/$a")"; to="$(last "$O1")"
    if [ -n "$ta" ] && [ -n "$to" ]; then
        python3 -I scripts/field_diff.py "$O1/$to" "$C1/$a/$ta" > "$C1/fielddiff.of1-$a.txt" 2>&1
        echo "--- of1 ($to) vs $a ($ta)"; cat "$C1/fielddiff.of1-$a.txt"
    fi
done
echo "=== done $(date -u +%FT%TZ)"
