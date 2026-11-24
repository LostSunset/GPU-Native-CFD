#!/usr/bin/env bash
# N-028 的歸約檢查（與 scripts/n6d_g02.sh 同一個建置）：GNC_SUMTREE_CHECK=1 時每次呼叫都以多次啟動的
# sumTree 重算並比對（sumTreeOp、sumTreeOps、sumTreeToDevice、sumTreeOpsToDevice），不同就 abort。
#   1. gncSumBench（同 N6c-sumcheck）
#   2. gncHisa -fast（nacaAirfoil SST、RAE2822 SA）與 gncRhoSimple -fast（aerofoilNACA0012 r1、r4）在檢查模式下
#      各跑 20 步：執行成功（rc=0）表示每次歸約都相同；兩次執行（fast_identity.sh，同一個執行檔）的場也相同
# 在 g02 上執行：bash scripts/n6d_sumcheck_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
G=/ssd/sclin/gpu-native-cfd/cases
R=/ssd/sclin/gpu-native-cfd/runs
BIN=/ssd/sclin/gpu-native-cfd/build/bin
P=N6d
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
mkdir -p "$R/$P-sumcheck"
cp /ssd/sclin/gpu-native-cfd/build/BUILD_INFO "$R/$P-sumcheck/"
run env CUDA_VISIBLE_DEVICES=3 GNC_SUMTREE_CHECK=1 $BIN/gncSumBench \
    --n 1,7,2048,2049,16000,2179881,4194304,4194305 --repeat 3 > "$R/$P-sumcheck/log.sumbench" 2>&1
echo "sumbench rc=$? (0: every call identical)" | tee -a "$R/$P-sumcheck/log.sumbench"
run env GNC_SUMTREE_CHECK=1 bash scripts/fast_identity.sh --host-env $E --label $P-sumcheck-hisa \
    --solver gncHisa --bin-a $BIN --bin-b $BIN \
    --case nacaSST=$C/nacaAirfoil-prepared/simulation --case rae2822SA=$C/rae2822-prepared/simulation --end 20
run env GNC_SUMTREE_CHECK=1 bash scripts/fast_identity.sh --host-env $E --label $P-sumcheck-rho \
    --solver gncRhoSimple --bin-a $BIN --bin-b $BIN \
    --case r1=$G/aerofoilNACA0012/simulation --case r4=$G/aerofoilNACA0012-r4/simulation --end 20
echo "=== done $(date -u +%FT%TZ)"
