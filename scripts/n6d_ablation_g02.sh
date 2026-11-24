#!/usr/bin/env bash
# N-028 的消融（與 scripts/n6d_g02.sh 同一個建置）：gncRhoSimple -fast（PCG＋AMG）在四個網格，逐項打開：
#   A  只有啟動與同步的改動（裝置端純量、批次 kernel；結果與 N6c 逐位元相同）：新的演算法全部關閉
#   B  A ＋ 兩色 DILU／DIC（GNC_COLOR_BIPARTITE=1）
#   C  B ＋ AMG 粗層 Jacobi、omega 1.6、單一 block 門檻 1024（Welsh–Powell 著色關閉）
#   完整的設定（C ＋ Welsh–Powell）即 N6d-perf-*
# 在 g02 上執行：bash scripts/n6d_ablation_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
G=/ssd/sclin/gpu-native-cfd/cases
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
mesh() { if [ "$1" = 1 ]; then echo aerofoilNACA0012; else echo "aerofoilNACA0012-r$1"; fi; }
for s in A B C; do
    case $s in
        A) set -- GNC_COLOR_BIPARTITE=0 GNC_AMG_JAC=0 GNC_AMG_OMEGA=1.8 GNC_AMG_BLOCK=4096 GNC_AMG_WP=0 ;;
        B) set -- GNC_COLOR_BIPARTITE=1 GNC_AMG_JAC=0 GNC_AMG_OMEGA=1.8 GNC_AMG_BLOCK=4096 GNC_AMG_WP=0 ;;
        C) set -- GNC_COLOR_BIPARTITE=1 GNC_AMG_WP=0 ;;
    esac
    for r in 1 2 4 8; do
        case $r in 1) n=200 ;; 2) n=100 ;; 4) n=50 ;; 8) n=30 ;; esac
        run env "$@" bash scripts/rhosimple_perf.sh --host-env $E --label N6d-abl$s-r$r --iters $n \
            --variants gncamg \
            --case "r$r=$G/$(mesh $r)-pcg/simulation" --gamg-case "r$r=$G/$(mesh $r)/simulation"
    done
done
echo "=== done $(date -u +%FT%TZ)"
