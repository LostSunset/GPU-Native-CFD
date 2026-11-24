#!/usr/bin/env bash
# N4c 的 g02 正式紀錄（GPU3）：建置；力與力係數的 L3（forceCoeffs、forces，含層流）；
# 層級排程自動模式之後預設執行的 L3 回歸；gncHisa 寫出的場檔與力係數檔；以 Cl、Cd
# 比較效能版（RAE2822 的殘差停在約 4e-3，以力係數的後段平均比較）。
# 在 g02 上執行：bash scripts/n4c_final_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
P=${N4C_PREFIX:-N4c-g02}
FAST="GNC_LUSGS=color GNC_SGS=color GNC_REDUCE=tree"
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
run bash scripts/build.sh --host-env $E
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-L3-forces \
    --end-default 200 --time-indices 1,2,10,50,100,200 \
    --end flatPlateSA=800 --time-indices-for flatPlateSA=1,2,10,100,400,800 \
    --end flatPlateLaminar=100 --time-indices-for flatPlateLaminar=1,2,10,100 \
    --case rae2822SA=$C/rae2822-prepared/simulation --forces rae2822SA=forceCoeffsCompressible \
    --case flatPlateSA=$C/flatPlate-g2000/simulation --forces flatPlateSA=forces \
    --case flatPlateLaminar=$C/flatPlate-g2000/simulation --laminar flatPlateLaminar \
    --forces flatPlateLaminar=forces
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-L3-auto \
    --end-default 50 --time-indices 1,2,10,50 --end oneraM6SA=10 --time-indices-for oneraM6SA=1,2,10 \
    --case forwardStep=$C/forwardStep-prepared/simulation \
    --case oneraM6SA=$C/oneraM6-prepared/simulation
run bash scripts/app_verify.sh --host-env $E --label $P-app --end 200 --write-interval 100 \
    --case rae2822SA=$C/rae2822-prepared/simulation --forces rae2822SA=forceCoeffsCompressible \
    --case flatPlateSA=$C/flatPlate-g2000/simulation --forces flatPlateSA=forces
run bash scripts/app_verify.sh --host-env $E --label $P-app-regress --end 20 --write-interval 10 \
    --case forwardStep=$C/forwardStep-prepared/simulation \
    --case nacaAirfoilLaminar=$C/nacaAirfoil-prepared/simulation --laminar nacaAirfoilLaminar
run bash scripts/variant_compare.sh --host-env $E --label $P-coeffs --end "${N4C_COEFF_END:-4000}" \
    --variant ref= --variant lusgsColor=GNC_LUSGS=color --variant "fast=$FAST" \
    --case rae2822SA=$C/rae2822-prepared/simulation --forces rae2822SA=forceCoeffsCompressible
echo "=== done $(date -u +%FT%TZ)"
