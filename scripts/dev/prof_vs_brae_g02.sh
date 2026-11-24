#!/usr/bin/env bash
# 開發用（不是正式紀錄）：gncRhoSimple -fast -profile 與 brae（BRAE_PHASE_TIME=1）在 aerofoilNACA0012 r1、r8
# 的階段時間。輸出在 runs/PROFdev-<時間>/。
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
G=/ssd/sclin/gpu-native-cfd/cases
BIN=/ssd/sclin/gpu-native-cfd/build/bin/gncRhoSimple
BR=/ssd/sclin/gpu-native-cfd/ext/brae/build/brae_rhoSimpleFoam
FOAM=/opt/atlas-openfoam/current/OpenFOAM-v2606
OUT=/ssd/sclin/gpu-native-cfd/runs/PROFdev-$(date +%H%M%S)
mkdir -p "$OUT"
mesh() { if [ "$1" = 1 ]; then echo aerofoilNACA0012; else echo "aerofoilNACA0012-r$1"; fi; }
prep() {
    mkdir -p "$2/constant"
    cp -r "$1/system" "$1/0" "$2/"
    for f in "$1"/constant/*; do
        [ "$(basename "$f")" = polyMesh ] && continue
        cp -r "$f" "$2/constant/"
    done
    ln -s "$1/constant/polyMesh" "$2/constant/polyMesh"
    sed -i -e "s/^endTime .*/endTime         $3;/" -e "s/^writeInterval .*/writeInterval   $3;/" "$2/system/controlDict"
}
for r in 1 8; do
    n=$([ $r = 1 ] && echo 100 || echo 20)
    src="$G/$(mesh $r)/simulation"
    d="$OUT/r$r/gnc"; prep "$src" "$d" "$n"
    bash scripts/gnc_prepare_case.sh -case "$d" --host-env configs/hosts/g02-l40.env > "$d/log.expand" 2>&1
    (cd "$d" && CUDA_VISIBLE_DEVICES=3 "$BIN" -case "$d" -expanded "$d/gncExpanded" -fast -profile > log.solver 2>&1)
    d2="$OUT/r$r/gncnp"; prep "$src" "$d2" "$n"; cp -r "$d/gncExpanded" "$d2/"
    (cd "$d2" && CUDA_VISIBLE_DEVICES=3 "$BIN" -case "$d2" -expanded "$d2/gncExpanded" -fast > log.solver 2>&1)
    d="$OUT/r$r/brae"; prep "$src" "$d" "$n"
    (cd "$d" && CUDA_VISIBLE_DEVICES=3 BRAE_PHASE_TIME=1 BRAE_RHOSIMPLEFOAM_MIRROR=cuda FOAM_ETC=$FOAM/etc \
        WM_PROJECT_DIR=$FOAM "$BR" -case "$d" > log.run 2> log.err)
done
echo "$OUT"
