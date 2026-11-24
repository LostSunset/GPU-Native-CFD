#!/usr/bin/env bash
# N6 的 AMG 改良（N-026：每 2 個配對層合成一層、前後掃各 2 次、粗網格修正 1.8 倍）之後的
# g02 紀錄（同一個建置）：gncRhoSimple 的 L2／L3 與 gncHisa 的回歸（參考版不受 AMG 影響，確認仍逐位元）；
# gncRhoSimple 三個變體在四個網格的計時與 16,000 格到收斂。OpenFOAM 與 SPUMA 不依賴本專案的建置，
# 沿用 N6-perf-* 的同日紀錄。
# 在 g02 上執行：bash scripts/n6b_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
G=/ssd/sclin/gpu-native-cfd/cases
P=N6b
NACA=$C/nacaAirfoil-prepared/simulation
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
mesh() { if [ "$1" = 1 ]; then echo aerofoilNACA0012; else echo "aerofoilNACA0012-r$1"; fi; }
run bash scripts/build.sh --host-env $E
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
C1=/ssd/sclin/gpu-native-cfd/runs/$P-perf-conv/r1
O1=/ssd/sclin/gpu-native-cfd/runs/N6-perf-conv/r1/of1
last() { (cd "$1" && ls -d [0-9]* 2>/dev/null | grep -E '^[0-9]+$' | sort -n | tail -1); }
for a in gnc gncamg; do
    ta="$(last "$C1/$a")"; to="$(last "$O1")"
    if [ -n "$ta" ] && [ -n "$to" ]; then
        python3 -I scripts/field_diff.py "$O1/$to" "$C1/$a/$ta" > "$C1/fielddiff.of1-$a.txt" 2>&1
        echo "--- of1 ($to) vs $a ($ta)"; cat "$C1/fielddiff.of1-$a.txt"
    fi
done
echo "=== done $(date -u +%FT%TZ)"
