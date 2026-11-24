#!/usr/bin/env bash
# N7（GMRES 重啟的 CUDA Graph，GNC_GRAPH=1）的 g02 量測（GPU3）：建置；預設執行與 GNC_GRAPH=1 的
# L3 逐位元（含 kOmegaSST 與能量限界的衍生案例）；ref、graph、fast（gncHisa -fast 的組合）、
# fastgraph 的每步時間。
# 在 g02 上執行：bash scripts/n7g_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
P=${N7G_PREFIX:-N7g-g02}
FAST="GNC_LUSGS=color GNC_SGS=color GNC_REDUCE=tree GNC_JAC=auto"
NACA=$C/nacaAirfoil-prepared/simulation
# 能量限界的衍生案例：forwardStep 加 pseudoTime/TMin 0.999、TMax 1.8（每一步都觸發上下限界）
TB=/ssd/sclin/gpu-native-cfd/cases/forwardStep-Tbound/simulation
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
if [ ! -d "$TB" ]; then
    S=$C/forwardStep-prepared/simulation
    mkdir -p "$TB/constant"
    cp -r "$S/system" "$S/0" "$TB/"
    for f in "$S"/constant/*; do
        b="$(basename "$f")"
        if [ "$b" = polyMesh ]; then ln -s "$(realpath "$f")" "$TB/constant/polyMesh"; else cp -r "$f" "$TB/constant/$b"; fi
    done
    ( set +u; . /opt/atlas-openfoam/current/OpenFOAM-v2606/etc/bashrc > /dev/null 2>&1
      foamDictionary -entry pseudoTime/TMin -set 0.999 "$TB/system/fvSolution" > /dev/null
      foamDictionary -entry pseudoTime/TMax -set 1.8 "$TB/system/fvSolution" > /dev/null )
    echo "source=$S; pseudoTime/TMin 0.999, TMax 1.8" > "$TB/DERIVED"
fi
verify=(--end-default 50 --time-indices 1,2,10,50 --end oneraM6SA=10 --time-indices-for oneraM6SA=1,2,10
        --case forwardStep=$C/forwardStep-prepared/simulation
        --case flatPlateSA=$C/flatPlate-g2000/simulation
        --case rae2822SA=$C/rae2822-prepared/simulation
        --case oneraM6SA=$C/oneraM6-prepared/simulation
        --case nacaAirfoilSST=$NACA --meshwave nacaAirfoilSST
        --case forwardStepTbound=$TB)
variants=(--variant ref= --variant graph=GNC_GRAPH=1 --variant "fast=$FAST" --variant "fastgraph=$FAST GNC_GRAPH=1")
run bash scripts/build.sh --host-env $E
run bash scripts/l3_steady_verify.sh --host-env $E --label $P-verify "${verify[@]}"
run env GNC_GRAPH=1 bash scripts/l3_steady_verify.sh --host-env $E --label $P-verify-graph "${verify[@]}"
run bash scripts/variant_compare.sh --host-env $E --label $P-perf200 --end 200 "${variants[@]}" \
    --case flatPlateSA=$C/flatPlate-g2000/simulation --case rae2822SA=$C/rae2822-prepared/simulation \
    --case nacaAirfoilSST=$NACA
run bash scripts/variant_compare.sh --host-env $E --label $P-perf-onera --end 30 "${variants[@]}" \
    --case oneraM6SA=$C/oneraM6-prepared/simulation
run bash scripts/variant_compare.sh --host-env $E --label $P-perf-finner --end 5 "${variants[@]}" \
    --case basicFinnerSST=$C/basicFinner-seq5/simulation
echo "=== done $(date -u +%FT%TZ)"
