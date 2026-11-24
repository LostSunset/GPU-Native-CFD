#!/usr/bin/env bash
# N6 的效能案例：aerofoilNACA0012（rhoSimpleFoam 教學案例）的加密網格。每個方向的格數乘以 R
# （blockMeshDict 的 domain/zCells、xUCells、xMCells、xDCells；grading 不變），格數為原本
# 16,000 的 R^2 倍。網格照教學案例的 Allrun.pre：blockMesh、transformPoints -scale '(1 0 1)'、
# extrudeMesh（topoSet 的 cellZone 沒有被求解器使用，省略）。每個 R 建兩個變體：
#   aerofoilNACA0012-rR/simulation       p 為 GAMG（原設定，OpenFOAM 的基準）
#   aerofoilNACA0012-rR-pcg/simulation   p 為 PCG＋DIC（gncRhoSimple 參考次序版可逐位元比對）
# 網格以符號連結共用。
#
# 用法：make_aerofoil_meshes.sh --host-env FILE --src DIR --out-root DIR R [R ...]

set -eu

host_env=""; src=""; out_root=""; factors=()
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --src) src="$2"; shift 2 ;;
        --out-root) out_root="$2"; shift 2 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) factors+=("$1"); shift ;;
    esac
done
[ -r "$host_env" ] || { echo "ERROR: --host-env FILE required" >&2; exit 2; }
[ -d "$src/system" ] || { echo "ERROR: --src must be a case" >&2; exit 2; }
[ -n "$out_root" ] || { echo "ERROR: --out-root DIR required" >&2; exit 2; }
# shellcheck disable=SC1090
. "$host_env"

foam() {
    (
        set +eu
        # shellcheck disable=SC1090
        . "$OPENFOAM_BASHRC" >/dev/null 2>&1 || true
        set -eu
        "$@"
    )
}

for r in "${factors[@]}"; do
    g="$out_root/aerofoilNACA0012-r$r/simulation"
    p="$out_root/aerofoilNACA0012-r$r-pcg/simulation"
    if [ -e "$g" ]; then
        echo "exists: $g"
        continue
    fi
    mkdir -p "$g/constant"
    cp -r "$src/system" "$src/0" "$g/"
    for f in "$src"/constant/*; do
        b="$(basename "$f")"
        [ "$b" = polyMesh ] || cp -r "$f" "$g/constant/$b"
    done
    # domain 子字典的格數（foamDictionary 無法展開這個檔的 $/ 變數，直接改文字）
    awk -v r="$r" '
        /^[[:space:]]*(zCells|xUCells|xMCells|xDCells)[[:space:]]+[0-9]+;/ {
            match($0, /[0-9]+;/)
            n = substr($0, RSTART, RLENGTH - 1)
            $0 = substr($0, 1, RSTART - 1) (n*r) substr($0, RSTART + RLENGTH - 1)
        }
        { print }
    ' "$src/system/blockMeshDict" > "$g/system/blockMeshDict"
    grep -E "^[[:space:]]*(zCells|xUCells|xMCells|xDCells)" "$g/system/blockMeshDict"
    (
        cd "$g"
        foam blockMesh > log.blockMesh 2>&1
        foam transformPoints -scale '(1 0 1)' > log.transformPoints 2>&1
        foam extrudeMesh > log.extrudeMesh 2>&1
        foam checkMesh > log.checkMesh 2>&1 || true
    )
    echo "source=$src; domain cells x$r (blockMesh, transformPoints, extrudeMesh)" > "$out_root/aerofoilNACA0012-r$r/DERIVED"
    grep -E "^ *cells:" "$g/log.checkMesh" || true

    mkdir -p "$p/constant"
    cp -r "$g/system" "$g/0" "$p/"
    for f in "$g"/constant/*; do
        b="$(basename "$f")"
        if [ "$b" = polyMesh ]; then
            ln -s "$(realpath "$f")" "$p/constant/polyMesh"
        else
            cp -r "$f" "$p/constant/$b"
        fi
    done
    foam foamDictionary -entry solvers/p -set "{ solver PCG; preconditioner DIC; tolerance 1e-06; relTol 0.01; }" \
        "$p/system/fvSolution" > /dev/null
    echo "source=$g; p: GAMG -> PCG DIC" > "$out_root/aerofoilNACA0012-r$r-pcg/DERIVED"
done
