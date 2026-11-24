#!/usr/bin/env bash
# N4a 的 g02 正式紀錄（同一個最終建置）：建置；L0、L1、N3a–N3c 的回歸；牆距離 L1；
# pow／exp 檢查；SA 的 L2；L3（N3d 的層流與無黏六案例、flatPlate 與 oneraM6 的
# RANS SA）與收斂測試；求解器執行檔寫出的場檔比對。
# 在 g02 上執行（GPU3）：bash scripts/n4a_final_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
P=N4a
base=(--case shockTube=$C/shockTube-prepared
      --case forwardStep=$C/forwardStep-prepared/simulation
      --case nacaAirfoil=$C/nacaAirfoil-prepared/simulation
      --case flatPlate=$C/flatPlate-g2000/simulation
      --case oneraM6=$C/oneraM6-prepared/simulation
      --case basicFinner=$C/basicFinner-seq5/simulation)
lam=(--case shockTube=$C/shockTube-prepared
     --case forwardStep=$C/forwardStep-prepared/simulation
     --case nacaAirfoilLaminar=$C/nacaAirfoil-prepared/simulation
     --case flatPlateLaminar=$C/flatPlate-g2000/simulation
     --case oneraM6Laminar=$C/oneraM6-prepared/simulation
     --case basicFinnerLaminar=$C/basicFinner-seq5/simulation
     --laminar nacaAirfoilLaminar --laminar flatPlateLaminar
     --laminar oneraM6Laminar --laminar basicFinnerLaminar)
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
run bash scripts/build.sh --host-env $E
run bash scripts/l0_verify.sh --host-env $E --label $P-L0 "${base[@]}"
run bash scripts/l1_verify.sh --host-env $E --label $P-L1 "${base[@]}"
run bash scripts/l2_init_verify.sh --host-env $E --label $P-N3a-L2 "${base[@]}"
run bash scripts/l2_res_verify.sh --host-env $E --label $P-N3b-L2 "${lam[@]}"
run bash scripts/l2_outer_verify.sh --host-env $E --label $P-N3c-L2 "${lam[@]}"
run bash scripts/l1_walldist_verify.sh --host-env $E --label $P-walldist-L1 "${base[@]}" \
    --meshwave nacaAirfoil --meshwave basicFinner
run bash scripts/pow_check.sh --host-env $E --label $P-powexp
run bash scripts/l2_turb_verify.sh --host-env $E --label $P-SA-L2 --steps 1,2,3,10,30,100 \
    --case flatPlate=$C/flatPlate-g2000/simulation --case oneraM6=$C/oneraM6-prepared/simulation \
    --case rae2822=$C/rae2822-prepared/simulation
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-L3 \
    --end-default 200 --time-indices 1,2,10,50,100,200 \
    --end nacaAirfoilLaminar=150 --end flatPlateLaminar=800 \
    --time-indices-for flatPlateLaminar=1,2,10,100,400,800 \
    --end oneraM6Laminar=30 --time-indices-for oneraM6Laminar=1,2,10,30 \
    --end basicFinnerLaminar=10 --time-indices-for basicFinnerLaminar=1,2,10 \
    "${lam[@]}"
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-SA-L3 \
    --end-default 800 --time-indices 1,2,10,100,400,800 \
    --end oneraM6SA=30 --time-indices-for oneraM6SA=1,2,10,30 \
    --end rae2822SA=200 --time-indices-for rae2822SA=1,2,10,50,100,200 \
    --case flatPlateSA=$C/flatPlate-g2000/simulation --case oneraM6SA=$C/oneraM6-prepared/simulation \
    --case rae2822SA=$C/rae2822-prepared/simulation
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-L3-converge \
    --end-default 800 --time-indices 1,50,100 \
    --pseudo-tol flatAbs=1e-3 --pseudo-tol-rel flatRel=2e4 \
    --case flatAbs=$C/flatPlate-g2000/simulation --case flatRel=$C/flatPlate-g2000/simulation \
    --laminar flatAbs --laminar flatRel
run bash scripts/app_verify.sh --host-env $E --label $P-app --end 20 --write-interval 10 \
    --case forwardStep=$C/forwardStep-prepared/simulation \
    --case nacaAirfoilLaminar=$C/nacaAirfoil-prepared/simulation --laminar nacaAirfoilLaminar
run bash scripts/app_verify.sh --host-env $E --label $P-app-converge --end 800 --write-interval 50 \
    --pseudo-tol flatAbs=1e-3 --case flatAbs=$C/flatPlate-g2000/simulation --laminar flatAbs
run bash scripts/app_verify.sh --host-env $E --label $P-app-SA --end 100 --write-interval 50 \
    --case flatPlateSA=$C/flatPlate-g2000/simulation
echo "=== done $(date -u +%FT%TZ)"
