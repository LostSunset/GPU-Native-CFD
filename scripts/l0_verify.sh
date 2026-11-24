#!/usr/bin/env bash
# N1 的 L0 驗證（design.md §6）：GPU 幾何對 OpenFOAM 逐位元比對。
#
# 每個案例：
#   ref.gnc    gncGeomRef（OpenFOAM，CPU）
#   gpu1.gnc   gncGeomL0 --repeat N（同一程序內重算 N 次，須逐位元相同）
#   gpu2.gnc   第二個程序，與 gpu1.gnc 以 cmp 比對（兩次執行相同）
#   compare.txt  gncCompare ref.gnc gpu1.gnc
# GPU 只用 $GPU_DEVICE（g02 為 GPU3）；執行前確認該 GPU 上沒有其他計算
# 程序，前後記錄 nvidia-smi。輸出在 $RUN_ROOT/LABEL，SUMMARY.tsv 為總表。
# 傾印檔（*.gnc）很大，只留在 g02；收回 repo 的是紀錄與雜湊。
#
# 用法：l0_verify.sh --host-env FILE --label LABEL --case NAME=DIR [--case ...]
#                    [--repeat N]

set -u

here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/.." && pwd)"

host_env=""; label=""; repeat=2
cases=()
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --case) cases+=("$2"); shift 2 ;;
        --repeat) repeat="$2"; shift 2 ;;
        -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -r "$host_env" ] || { echo "ERROR: --host-env FILE required" >&2; exit 2; }
# shellcheck disable=SC1090
. "$host_env"
case "$label" in ""|*/*|.*) echo "ERROR: --label NAME required" >&2; exit 2 ;; esac
[ "${#cases[@]}" -gt 0 ] || { echo "ERROR: at least one --case NAME=DIR" >&2; exit 2; }

bin="$BUILD_ROOT/bin"
for exe in gncGeomL0 gncCompare gncGeomRef; do
    [ -x "$bin/$exe" ] || { echo "ERROR: $bin/$exe missing (run scripts/build.sh)" >&2; exit 2; }
done

out_root="$RUN_ROOT/$label"
[ -e "$out_root" ] && { echo "ERROR: $out_root exists" >&2; exit 2; }
mkdir -p "$out_root"

# GPU 是否空閒：$GPU_DEVICE 上不能有其他計算程序
gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v d="$GPU_DEVICE" '$1 == d {print $2}')"
[ -n "$gpu_uuid" ] || { echo "ERROR: GPU $GPU_DEVICE not found" >&2; exit 2; }
busy="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader | grep -c "$gpu_uuid")"
if [ "$busy" -ne 0 ]; then
    echo "ERROR: GPU $GPU_DEVICE ($gpu_uuid) has $busy compute process(es)" >&2
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv >&2
    exit 3
fi

{
    echo "label=$label"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname) profile=$HOST_PROFILE"
    echo "gpu_device=$GPU_DEVICE uuid=$gpu_uuid"
    echo "repeat=$repeat"
    echo "uptime=$(uptime)"
} > "$out_root/RUN_INFO"
cp "$BUILD_ROOT/BUILD_INFO" "$out_root/BUILD_INFO"
nvidia-smi > "$out_root/nvidia-smi.before.txt" 2>&1

printf 'case\tnCells\tnFaces\tformat\tref_exit\tgpu_exit\tcompare\tgpu1_vs_gpu2\trepeat\tsumV_ref\tsumV_tree\tgeometryScheme\tpointsInstance\n' > "$out_root/SUMMARY.tsv"
overall=0

for spec in "${cases[@]}"; do
    name="${spec%%=*}"; dir="${spec#*=}"
    out="$out_root/$name"
    mkdir -p "$out"
    echo "== $name ($dir)"

    (
        set +u
        # shellcheck disable=SC1090
        . "$OPENFOAM_BASHRC" >/dev/null 2>&1 || true
        set -u
        "$bin/gncGeomRef" -case "$dir" -out "$out/ref.gnc"
    ) > "$out/log.ref" 2>&1
    rc_ref=$?

    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$bin/gncGeomL0" --case "$dir" --out "$out/gpu1.gnc" --repeat "$repeat" > "$out/log.gpu1" 2>&1
    rc_gpu1=$?
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$bin/gncGeomL0" --case "$dir" --out "$out/gpu2.gnc" > "$out/log.gpu2" 2>&1
    rc_gpu2=$?

    cmp_result="-"; compare="-"
    if [ "$rc_ref" -eq 0 ] && [ "$rc_gpu1" -eq 0 ]; then
        "$bin/gncCompare" "$out/ref.gnc" "$out/gpu1.gnc" > "$out/compare.txt" 2>&1
        case $? in 0) compare="BITWISE" ;; 1) compare="DIFF" ;; *) compare="ERROR" ;; esac
    fi
    if [ "$rc_gpu1" -eq 0 ] && [ "$rc_gpu2" -eq 0 ]; then
        if cmp -s "$out/gpu1.gnc" "$out/gpu2.gnc"; then cmp_result="identical"; else cmp_result="different"; fi
    fi
    sha256sum "$out"/*.gnc > "$out/SHA256SUMS" 2>/dev/null
    ls -l "$out"/*.gnc > "$out/sizes.txt" 2>/dev/null

    mesh_line="$(grep -m1 '^mesh:' "$out/log.gpu1" 2>/dev/null)"
    ncells="$(echo "$mesh_line" | sed -n 's/.*nCells \([0-9]*\).*/\1/p')"
    nfaces="$(echo "$mesh_line" | sed -n 's/.*nFaces \([0-9]*\).*/\1/p')"
    format="$(echo "$mesh_line" | sed -n 's/.*(\([a-z]*\)).*/\1/p')"
    rep="$(sed -n 's/^repeat: \(.*\) bitwise.*/\1/p' "$out/log.gpu1" 2>/dev/null)"
    sum_ref="$(sed -n 's/^sum(V) //p' "$out/log.ref" 2>/dev/null)"
    sum_tree="$(sed -n 's/^sum(V): sequential [^ ]* tree //p' "$out/log.gpu1" 2>/dev/null)"
    scheme="$(sed -n 's/^geometry scheme //p' "$out/log.ref" 2>/dev/null)"
    inst="$(sed -n 's/^pointsInstance *text  ref="\([^"]*\)".*/\1/p' "$out/compare.txt" 2>/dev/null)"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$ncells" "$nfaces" "$format" "$rc_ref" "$rc_gpu1/$rc_gpu2" \
        "$compare" "$cmp_result" "$rep" "$sum_ref" "$sum_tree" "$scheme" "$inst" >> "$out_root/SUMMARY.tsv"
    echo "   ref=$rc_ref gpu=$rc_gpu1/$rc_gpu2 compare=$compare gpu1_vs_gpu2=$cmp_result repeat=$rep"

    if [ "$compare" != "BITWISE" ] || [ "$cmp_result" != "identical" ]; then
        overall=1
    fi
done

nvidia-smi > "$out_root/nvidia-smi.after.txt" 2>&1
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out_root/RUN_INFO"
echo "overall_exit=$overall" >> "$out_root/RUN_INFO"
column -t -s "$(printf '\t')" "$out_root/SUMMARY.tsv"
exit "$overall"
