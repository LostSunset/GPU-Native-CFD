#!/usr/bin/env bash
# N6（gncRhoSimple，N-023）的 g02 正式紀錄（同一個最終建置）：建置；gncRhoSimple 對
# rhoSimpleFoam 的 L2／L3（aerofoilNACA0012，p 以 PCG＋DIC 的驗證變體）；gncHisa 的回歸
# （共用的邊界條件、純量矩陣與 kOmegaSST 有改動）：SST 的 L2、L3、求解器寫出，SA 的
# L2、L3，層流與無黏的 L3。效能紀錄在 scripts/n6_perf_g02.sh（同一個建置，之後執行）。
# 在 g02 上執行（GPU3）：bash scripts/n6_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
G=/ssd/sclin/gpu-native-cfd/cases
P=N6
NACA=$C/nacaAirfoil-prepared/simulation
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
run bash scripts/build.sh --host-env $E
run bash scripts/rhosimple_verify.sh --host-env $E --label $P-L3 --dump-steps 1,2,10,100,668 \
    --case aerofoilNACA0012=$G/aerofoilNACA0012-pcg/simulation
# 回歸（N4b 的設定的子集）
run bash scripts/l2_turb_verify.sh --host-env $E --label $P-reg-SST-L2 --steps 1,2,3,10 \
    --case nacaAirfoil=$NACA --meshwave nacaAirfoil
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-reg-SST-L3 \
    --end-default 200 --time-indices 1,2,10,50,100,200 \
    --case nacaAirfoilSST=$NACA --meshwave nacaAirfoilSST
run bash scripts/app_verify.sh --host-env $E --label $P-reg-SST-app --end 60 --write-interval 30 \
    --case nacaAirfoilSST=$NACA --meshwave nacaAirfoilSST
run bash scripts/l2_turb_verify.sh --host-env $E --label $P-reg-SA-L2 --steps 1,2,3,10 \
    --case flatPlate=$C/flatPlate-g2000/simulation --case rae2822=$C/rae2822-prepared/simulation
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-reg-SA-L3 \
    --end-default 200 --time-indices 1,2,10,50,100,200 \
    --case rae2822SA=$C/rae2822-prepared/simulation
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-reg-L3 \
    --end-default 200 --time-indices 1,2,10,50,100,200 \
    --end nacaAirfoilLaminar=150 \
    --case shockTube=$C/shockTube-prepared \
    --case forwardStep=$C/forwardStep-prepared/simulation \
    --case nacaAirfoilLaminar=$NACA \
    --laminar nacaAirfoilLaminar
echo "=== done $(date -u +%FT%TZ)"
