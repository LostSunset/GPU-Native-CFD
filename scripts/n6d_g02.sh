#!/usr/bin/env bash
# N-028（小網格的啟動與同步：裝置端純量的 Krylov、批次逐元素運算、多 patch 邊界 kernel、合併的
# min／max 與歸約；AMG 粗層 Jacobi 平滑、兩色 DILU）之後的 g02 紀錄（同一個建置）：
#   1. 正式建置（N6c 的執行檔先保留在 build-N6c/bin）
#   2. 只改同步／啟動的部分與 N6c 逐位元相同（scripts/fast_identity.sh）：gncHisa -fast（nacaAirfoil SST、
#      RAE2822 SA，20 步）；gncRhoSimple -fast 在新演算法關閉時（GNC_COLOR_BIPARTITE=0、GNC_AMG_JAC=0、
#      GNC_AMG_OMEGA=1.8、GNC_AMG_BLOCK=4096、GNC_AMG_WP=0；aerofoilNACA0012 r1、r2，20 步）
#   3. 參考版不受影響：gncRhoSimple 對 rhoSimpleFoam 的 L2／L3、gncHisa SST／SA 的 L3
#   4. gncRhoSimple 三個變體在四個網格的計時、16,000 格到收斂與對 OpenFOAM 單核解的差
# OpenFOAM、SPUMA、brae 沿用 N6-perf-*、BRAE-perf。
# 在 g02 上執行：bash scripts/n6d_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
G=/ssd/sclin/gpu-native-cfd/cases
R=/ssd/sclin/gpu-native-cfd/runs
B=/ssd/sclin/gpu-native-cfd
P=N6d
NACA=$C/nacaAirfoil-prepared/simulation
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
mesh() { if [ "$1" = 1 ]; then echo aerofoilNACA0012; else echo "aerofoilNACA0012-r$1"; fi; }
[ -x "$B/build-N6c/bin/gncHisa" ] || { echo "ERROR: build-N6c/bin missing" >&2; exit 2; }
run bash scripts/build.sh --host-env $E
run bash scripts/fast_identity.sh --host-env $E --label $P-id-hisa --solver gncHisa \
    --bin-a $B/build-N6c/bin --bin-b $B/build/bin \
    --case nacaSST=$NACA --case rae2822SA=$C/rae2822-prepared/simulation --end 20
run env GNC_COLOR_BIPARTITE=0 GNC_AMG_JAC=0 GNC_AMG_OMEGA=1.8 GNC_AMG_BLOCK=4096 GNC_AMG_WP=0 \
    bash scripts/fast_identity.sh --host-env $E --label $P-id-rho --solver gncRhoSimple \
    --bin-a $B/build-N6c/bin --bin-b $B/build/bin \
    --case r1=$G/aerofoilNACA0012/simulation --case r2=$G/aerofoilNACA0012-r2/simulation --end 20
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
