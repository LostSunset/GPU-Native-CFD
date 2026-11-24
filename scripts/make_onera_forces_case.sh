#!/usr/bin/env bash
# ONERA M6（HiSA examples/oneraM6，Spalart–Allmaras）加上翼面的 forceCoeffs 的衍生案例。
# 原案例不寫入：system、0、constant 複製，constant/polyMesh 以符號連結。
#
# 參考量（ONERA M6 的標準值；網格的根弦 0.80595 m 與此一致）：
#   Aref = 0.7532 m²（半翼的平面面積），lRef = 0.64607 m（平均氣動弦長），CofR = (0 0 0)
#   自由流取自 0/U、0/p、0/T：U = (268.654 14.362 0)（攻角 3.06°，馬赫 0.8395），
#   p = 80510.081 Pa，T = 255.556 K；rhoInf = p/(R T)，R = 8314.47/28.966
#   dragDir = U/|U|，liftDir = (-U_y, U_x, 0)/|U|（升力方向為 +y，展向為 z）
#
# 用法：make_onera_forces_case.sh --host-env FILE --src DIR --dest DIR

set -eu

host_env=""; src=""; dest=""
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --src) src="$2"; shift 2 ;;
        --dest) dest="$2"; shift 2 ;;
        -h|--help) sed -n '2,13p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -r "$host_env" ] || { echo "ERROR: --host-env FILE required" >&2; exit 2; }
[ -d "$src/system" ] || { echo "ERROR: --src is not a case" >&2; exit 2; }
[ -e "$dest" ] && { echo "ERROR: $dest exists" >&2; exit 2; }
# shellcheck disable=SC1090
. "$host_env"
set +eu
# shellcheck disable=SC1090
. "$OPENFOAM_BASHRC" > /dev/null 2>&1
set -eu

mkdir -p "$dest/constant"
cp -r "$src/system" "$src/0" "$dest/"
for f in "$src"/constant/*; do
    b="$(basename "$f")"
    if [ "$b" = polyMesh ]; then
        ln -s "$(realpath "$f")" "$dest/constant/polyMesh"
    else
        cp -r "$f" "$dest/constant/$b"
    fi
done

read -r magU rhoInf dx dy lx ly <<< "$(awk 'BEGIN{
    ux = 268.654; uy = 14.362; p = 80510.081; T = 255.556; R = 8314.47/28.966;
    m = sqrt(ux*ux + uy*uy);
    printf "%.17g %.17g %.17g %.17g %.17g %.17g", m, p/(R*T), ux/m, uy/m, -uy/m, ux/m }')"

foamDictionary -entry functions -set "{ forceCoeffsWing { type forceCoeffs; \
    libs (\"libforces.so\"); patches (wing); p p; U U; rho rhoInf; rhoInf $rhoInf; \
    magUInf $magU; lRef 0.64607; Aref 0.7532; CofR (0 0 0); \
    liftDir ($lx $ly 0); dragDir ($dx $dy 0); pitchAxis (0 0 1); \
    writeControl timeStep; writeInterval 1; executeControl timeStep; executeInterval 1; \
    log off; } }" "$dest/system/controlDict" > /dev/null
{
    echo "source=$src"
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "magUInf=$magU rhoInf=$rhoInf dragDir=($dx $dy 0) liftDir=($lx $ly 0)"
    echo "Aref=0.7532 lRef=0.64607 CofR=(0 0 0)"
} > "$dest/DERIVED"
echo "created $dest"
