#!/usr/bin/env bash
# N4b（kOmegaSST，N-022）的 g02 正式紀錄（同一個最終建置）：建置；tanh、expm1、log
# 與 pow、exp 的逐位元檢查；牆距離（meshWave 變體逐位元、Poisson 量測差異）；
# kOmegaSST 的 L2、L3（nacaAirfoil、basicFinner 的 meshWave 變體）；求解器執行檔
# 寫出的 k、omega、nut、alphat；SA 與層流、無黏的回歸（N4a 的 L2、L3 設定）。
# 在 g02 上執行（GPU3）：bash scripts/n4b_final_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
P=N4b
NACA=$C/nacaAirfoil-prepared/simulation
FINNER=$C/basicFinner-seq5/simulation
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
run bash scripts/build.sh --host-env $E
run bash scripts/pow_check.sh --host-env $E --label $P-powexp
run env CUDA_VISIBLE_DEVICES=3 bash scripts/l1_walldist_verify.sh --host-env $E --label $P-walldist-L1 \
    --case nacaAirfoil=$NACA --case basicFinner=$FINNER --meshwave nacaAirfoil --meshwave basicFinner
run env CUDA_VISIBLE_DEVICES=3 bash scripts/l1_walldist_verify.sh --host-env $E --label $P-poisson-L1 \
    --case nacaAirfoil=$NACA --case basicFinner=$FINNER
run bash scripts/l2_turb_verify.sh --host-env $E --label $P-SST-L2 --steps 1,2,3,10,30,100 \
    --case nacaAirfoil=$NACA --meshwave nacaAirfoil
run bash scripts/l2_turb_verify.sh --host-env $E --label $P-SST-L2-finner --steps 1,2,3 \
    --case basicFinner=$FINNER --meshwave basicFinner
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-SST-L3 \
    --end-default 200 --time-indices 1,2,10,50,100,200 \
    --end basicFinnerSST=10 --time-indices-for basicFinnerSST=1,2,10 \
    --case nacaAirfoilSST=$NACA --case basicFinnerSST=$FINNER \
    --meshwave nacaAirfoilSST --meshwave basicFinnerSST
run bash scripts/app_verify.sh --host-env $E --label $P-SST-app --end 60 --write-interval 30 \
    --case nacaAirfoilSST=$NACA --meshwave nacaAirfoilSST
# 回歸：SA（N4a-SA-L2、N4a-SA-L3 的設定）、層流與無黏（N4a-L3 的設定）
run bash scripts/l2_turb_verify.sh --host-env $E --label $P-SA-L2 --steps 1,2,3,10,30,100 \
    --case flatPlate=$C/flatPlate-g2000/simulation --case oneraM6=$C/oneraM6-prepared/simulation \
    --case rae2822=$C/rae2822-prepared/simulation
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-SA-L3 \
    --end-default 800 --time-indices 1,2,10,100,400,800 \
    --end oneraM6SA=30 --time-indices-for oneraM6SA=1,2,10,30 \
    --end rae2822SA=200 --time-indices-for rae2822SA=1,2,10,50,100,200 \
    --case flatPlateSA=$C/flatPlate-g2000/simulation --case oneraM6SA=$C/oneraM6-prepared/simulation \
    --case rae2822SA=$C/rae2822-prepared/simulation
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-L3 \
    --end-default 200 --time-indices 1,2,10,50,100,200 \
    --end nacaAirfoilLaminar=150 --end flatPlateLaminar=800 \
    --time-indices-for flatPlateLaminar=1,2,10,100,400,800 \
    --end oneraM6Laminar=30 --time-indices-for oneraM6Laminar=1,2,10,30 \
    --end basicFinnerLaminar=10 --time-indices-for basicFinnerLaminar=1,2,10 \
    --case shockTube=$C/shockTube-prepared \
    --case forwardStep=$C/forwardStep-prepared/simulation \
    --case nacaAirfoilLaminar=$NACA \
    --case flatPlateLaminar=$C/flatPlate-g2000/simulation \
    --case oneraM6Laminar=$C/oneraM6-prepared/simulation \
    --case basicFinnerLaminar=$FINNER \
    --laminar nacaAirfoilLaminar --laminar flatPlateLaminar \
    --laminar oneraM6Laminar --laminar basicFinnerLaminar
echo "=== done $(date -u +%FT%TZ)"
