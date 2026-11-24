#!/usr/bin/env bash
# gncHisa、gncRhoSimple 的前處理：以 foamDictionary -expand -precision 17 把案例的字典展開到
# <case>/gncExpanded（N-004：求解器只讀展開後的子集，不連結 OpenFOAM）。
# 需要 OpenFOAM 的環境（foamDictionary 在 PATH 中，或以 --host-env 指定）。
#
# 用法：gnc_prepare_case.sh [-case DIR] [--host-env FILE]

set -eu

caseDir="."; host_env=""
while [ $# -gt 0 ]; do
    case "$1" in
        -case) caseDir="$2"; shift 2 ;;
        --host-env) host_env="$2"; shift 2 ;;
        -h|--help) sed -n '2,7p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
if [ -n "$host_env" ]; then
    # shellcheck disable=SC1090
    . "$host_env"
    set +eu
    # shellcheck disable=SC1090
    . "$OPENFOAM_BASHRC" > /dev/null 2>&1
    set -eu
fi
command -v foamDictionary > /dev/null || { echo "ERROR: foamDictionary not found" >&2; exit 2; }
[ -d "$caseDir/system" ] || { echo "ERROR: $caseDir is not a case" >&2; exit 2; }

out="$caseDir/gncExpanded"
mkdir -p "$out"
for f in 0/p 0/T 0/U 0/nuTilda 0/k 0/omega 0/nut 0/alphat constant/thermophysicalProperties system/fvSchemes \
         system/fvSolution system/controlDict constant/turbulenceProperties system/fvOptions; do
    [ -f "$caseDir/$f" ] || continue
    # 在案例目錄內展開：controlDict 的 #includeFunc 才會讀到案例的 system/<名稱>
    (cd "$caseDir" && foamDictionary -expand -precision 17 "$f") > "$out/$(basename "$f")"
done
echo "expanded into $out"
