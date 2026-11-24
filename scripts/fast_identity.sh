#!/usr/bin/env bash
# 兩個建置的效能版（-fast）是否逐位元相同：同一個案例各跑 N 步，比對第 N 步寫出的場（cmp）。用於
# 「只改同步、啟動次數、批次而不改運算」的改動（N-028）：改動前的建置（--bin-a）與改動後的建置
# （--bin-b）結果必須完全相同。
#
# 用法：fast_identity.sh --host-env FILE --label LABEL --solver gncHisa|gncRhoSimple
#                        --bin-a DIR --bin-b DIR --case NAME=DIR [--case ...] --end N
set -u
host_env=""; label=""; solver=""; binA=""; binB=""; endN=""
cases=()
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --solver) solver="$2"; shift 2 ;;
        --bin-a) binA="$2"; shift 2 ;;
        --bin-b) binB="$2"; shift 2 ;;
        --case) cases+=("$2"); shift 2 ;;
        --end) endN="$2"; shift 2 ;;
        -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -r "$host_env" ] || { echo "ERROR: --host-env FILE required" >&2; exit 2; }
# shellcheck disable=SC1090
. "$host_env"
case "$label" in ""|*/*|.*) echo "ERROR: --label NAME required" >&2; exit 2 ;; esac
for b in "$binA/$solver" "$binB/$solver"; do
    [ -x "$b" ] || { echo "ERROR: $b missing" >&2; exit 2; }
done
[ -n "$endN" ] || { echo "ERROR: --end N required" >&2; exit 2; }
here="$(cd "$(dirname "$0")" && pwd)"
out_root="$RUN_ROOT/$label"
[ -e "$out_root" ] && { echo "ERROR: $out_root exists" >&2; exit 2; }
mkdir -p "$out_root"
{
    echo "solver=$solver end=$endN"
    echo "bin_a=$binA sha256=$(sha256sum "$binA/$solver" | awk '{print $1}')"
    echo "bin_b=$binB sha256=$(sha256sum "$binB/$solver" | awk '{print $1}')"
    [ -f "$binA/../BUILD_INFO" ] && echo "bin_a_$(grep source_sha256 "$binA/../BUILD_INFO")"
    [ -f "$binB/../BUILD_INFO" ] && echo "bin_b_$(grep source_sha256 "$binB/../BUILD_INFO")"
} > "$out_root/RUN_INFO"

prep() {
    mkdir -p "$2/constant"
    cp -r "$1/system" "$1/0" "$2/"
    for f in "$1"/constant/*; do
        b="$(basename "$f")"
        if [ "$b" = polyMesh ]; then
            ln -s "$(realpath "$f")" "$2/constant/polyMesh"
        else
            cp -r "$f" "$2/constant/$b"
        fi
    done
    sed -i -e "s/^endTime .*/endTime         $3;/" -e "s/^writeInterval .*/writeInterval   $3;/" \
        -e "s/^startFrom .*/startFrom       startTime;/" -e "s/^stopAt .*/stopAt          endTime;/" \
        "$2/system/controlDict"
}

all_ok=1
printf 'case\tfiles\tidentical\tresult\n' > "$out_root/SUMMARY.tsv"
for spec in "${cases[@]}"; do
    name="${spec%%=*}"; src="${spec#*=}"
    for v in a b; do
        d="$out_root/$name/$v"
        prep "$src" "$d" "$endN"
        bash "$here/gnc_prepare_case.sh" -case "$d" --host-env "$host_env" > "$d/log.prepare" 2>&1
        exe="$binA/$solver"; [ "$v" = b ] && exe="$binB/$solver"
        (cd "$d" && CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$exe" -case "$d" -fast > log.solver 2>&1)
        echo "$name $v rc=$?"
    done
    n=0; same=0
    for f in "$out_root/$name/a/$endN"/*; do
        [ -f "$f" ] || continue
        n=$((n + 1))
        if cmp -s "$f" "$out_root/$name/b/$endN/$(basename "$f")"; then
            same=$((same + 1))
        else
            echo "  DIFF $(basename "$f")"
        fi
    done
    res=IDENTICAL; { [ "$n" -eq 0 ] || [ "$same" -ne "$n" ]; } && { res=DIFFERENT; all_ok=0; }
    printf '%s\t%s\t%s\t%s\n' "$name" "$n" "$same" "$res" | tee -a "$out_root/SUMMARY.tsv"
done
[ "$all_ok" = 1 ] && echo "result: ALL IDENTICAL" || echo "result: DIFFERENCES"
