#!/usr/bin/env bash
# 衍生案例：以 HiSA 執行 N 步之後的 p、T、U 為初始場的新案例（診斷用，N3d）。
#
# 均勻的初始場會讓許多分支沒有走到（design.md §5）。把參考跑到某一步的狀態
# 寫成新案例的 0/，就能用 l2_res_verify.sh、l2_outer_verify.sh 在非均勻的
# 狀態上比對單一外迭代。
#
# SRC 是一個已設定好的工作副本（例如 l3_steady_verify.sh 的 <run>/<case>/case）。
# 在 $GNC_ROOT/derived/NAME-gen 以二進位跑 N 步，再把 N/{p,T,U} 以 17 位有效
# 數字的 ASCII（foamFormatConvert）放進 $GNC_ROOT/derived/NAME/0/。兩邊都以
# OpenFOAM 的規則讀同一份文字，所以比對成立；但新案例的守恆量是由 p、T、U
# 重新初始化，與連續執行時第 N 步的狀態不完全相同。
#
# 用法：derive_case.sh --host-env FILE --src DIR --name NAME --steps N

set -eu

host_env=""; src=""; name=""; steps=""
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --src) src="$2"; shift 2 ;;
        --name) name="$2"; shift 2 ;;
        --steps) steps="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -r "$host_env" ] || { echo "ERROR: --host-env FILE required" >&2; exit 2; }
# shellcheck disable=SC1090
. "$host_env"
[ -d "$src/system" ] || { echo "ERROR: --src DIR (a case) required" >&2; exit 2; }
case "$name" in ""|*/*|.*) echo "ERROR: --name NAME required" >&2; exit 2 ;; esac
case "$steps" in ''|*[!0-9]*) echo "ERROR: --steps N required" >&2; exit 2 ;; esac

set +eu
# shellcheck disable=SC1090
. "$OPENFOAM_BASHRC" > /dev/null 2>&1
set -eu

copy_case() {
    mkdir -p "$2/constant"
    cp -r "$1/system" "$2/"
    for f in "$1"/constant/*; do
        b="$(basename "$f")"
        if [ "$b" = polyMesh ]; then
            ln -s "$(realpath "$f")" "$2/constant/polyMesh"
        else
            cp -r "$f" "$2/constant/"
        fi
    done
}

root="$GNC_ROOT/derived"
gen="$root/$name-gen"
out="$root/$name"
[ -e "$out" ] && { echo "ERROR: $out exists" >&2; exit 2; }
rm -rf "$gen"
copy_case "$src" "$gen"
cp -r "$src/0" "$gen/"
(
    cd "$gen"
    for kv in "endTime=$steps" "writeInterval=$steps" writeControl=timeStep writeFormat=binary; do
        foamDictionary -entry "${kv%%=*}" -set "${kv#*=}" system/controlDict > /dev/null
    done
    foamDictionary -entry functions -set "{}" system/controlDict > /dev/null
    hisa > log.hisa 2>&1 || { tail -20 log.hisa >&2; exit 1; }
)
[ -f "$gen/$steps/U" ] || { echo "ERROR: $gen/$steps/U not written (converged earlier?)" >&2; exit 1; }

copy_case "$gen" "$out"
mkdir -p "$out/0"
cp "$gen/$steps/p" "$gen/$steps/T" "$gen/$steps/U" "$out/0/"
(
    cd "$out"
    foamDictionary -entry writeFormat -set ascii system/controlDict > /dev/null
    foamDictionary -entry writePrecision -set 17 system/controlDict > /dev/null
    foamFormatConvert -time 0 > log.convert 2>&1
)
{
    echo "source=$src"
    echo "steps=$steps"
    echo "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$out/DERIVED"
echo "derived $out (state after $steps steps of $src)"
