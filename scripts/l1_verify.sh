#!/usr/bin/env bash
# N2 的 L1 驗證（design.md §6）：場、邊界條件與運算子對 OpenFOAM／HiSA
# 逐位元比對。
#
# 每個案例在 $RUN_ROOT/LABEL/NAME/case 建一份工作副本（system 與 constant
# 複製，constant/polyMesh 以符號連結指回原案例；原案例不寫入），然後：
#   1. foamDictionary -expand system/fvSchemes > case/gncL1.fvSchemes
#   2. gncOpsRef -generate：測試場寫到 case/gncL1（ASCII）與 case/gncL1bin
#   3. ref.gnc     gncOpsRef（OpenFOAM＋HiSA，CPU）
#   4. gpu1.gnc    gncOpsL1（ASCII 與二進位讀取器互比；場寫回 case/gncL1out）
#      gpu2.gnc    第二個程序，與 gpu1.gnc 以 cmp 比對
#   5. compare.txt         gncCompare ref.gnc gpu1.gnc
#   6. compare.fields.txt  gncOpsRef -dump-fields 讀 gncL1 與 gncL1out，比對
#                          （OpenFOAM 能讀 GPU-Native-CFD 寫的場）
# GPU 只用 $GPU_DEVICE；執行前確認該 GPU 上沒有其他計算程序。
#
# 用法：l1_verify.sh --host-env FILE --label LABEL --case NAME=DIR [--case ...]

set -u

here="$(cd "$(dirname "$0")" && pwd)"

host_env=""; label=""
cases=()
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --case) cases+=("$2"); shift 2 ;;
        -h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -r "$host_env" ] || { echo "ERROR: --host-env FILE required" >&2; exit 2; }
# shellcheck disable=SC1090
. "$host_env"
case "$label" in ""|*/*|.*) echo "ERROR: --label NAME required" >&2; exit 2 ;; esac
[ "${#cases[@]}" -gt 0 ] || { echo "ERROR: at least one --case NAME=DIR" >&2; exit 2; }

bin="$BUILD_ROOT/bin"
for exe in gncOpsL1 gncCompare gncOpsRef; do
    [ -x "$bin/$exe" ] || { echo "ERROR: $bin/$exe missing (run scripts/build.sh)" >&2; exit 2; }
done

out_root="$RUN_ROOT/$label"
[ -e "$out_root" ] && { echo "ERROR: $out_root exists" >&2; exit 2; }
mkdir -p "$out_root"

gpu_uuid="$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v d="$GPU_DEVICE" '$1 == d {print $2}')"
[ -n "$gpu_uuid" ] || { echo "ERROR: GPU $GPU_DEVICE not found" >&2; exit 2; }
busy="$(nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv,noheader | grep -c "$gpu_uuid")"
if [ "$busy" -ne 0 ]; then
    echo "ERROR: GPU $GPU_DEVICE ($gpu_uuid) has $busy compute process(es)" >&2
    exit 3
fi

{
    echo "label=$label"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname) profile=$HOST_PROFILE"
    echo "gpu_device=$GPU_DEVICE uuid=$gpu_uuid"
    echo "uptime=$(uptime)"
} > "$out_root/RUN_INFO"
cp "$BUILD_ROOT/BUILD_INFO" "$out_root/BUILD_INFO"
nvidia-smi > "$out_root/nvidia-smi.before.txt" 2>&1

foam() {
    (
        set +u
        # shellcheck disable=SC1090
        . "$OPENFOAM_BASHRC" >/dev/null 2>&1 || true
        set -u
        "$@"
    )
}

printf 'case\tnCells\tgradScheme\tgenerate\tref\tgpu\tcompare\tgpu1_vs_gpu2\treader_ascii_vs_binary\tfields_readback\n' > "$out_root/SUMMARY.tsv"
overall=0

for spec in "${cases[@]}"; do
    name="${spec%%=*}"; src="${spec#*=}"
    out="$out_root/$name"
    work="$out/case"
    mkdir -p "$work/constant"
    echo "== $name ($src)"

    cp -r "$src/system" "$work/system"
    for f in "$src"/constant/*; do
        b="$(basename "$f")"
        if [ "$b" = polyMesh ]; then
            ln -s "$(realpath "$f")" "$work/constant/polyMesh"
        else
            cp -r "$f" "$work/constant/$b"
        fi
    done
    echo "source=$src" > "$out/SOURCE"

    foam foamDictionary -expand "$work/system/fvSchemes" > "$work/gncL1.fvSchemes" 2> "$out/log.expand"
    foam "$bin/gncOpsRef" -case "$work" -generate > "$out/log.generate" 2>&1
    rc_gen=$?
    foam "$bin/gncOpsRef" -case "$work" -fields gncL1 -out "$out/ref.gnc" > "$out/log.ref" 2>&1
    rc_ref=$?

    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$bin/gncOpsL1" --case "$work" --fields gncL1 \
        --bin-fields gncL1bin --schemes "$work/gncL1.fvSchemes" --out "$out/gpu1.gnc" \
        --write-dir gncL1out > "$out/log.gpu1" 2>&1
    rc_gpu1=$?
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$bin/gncOpsL1" --case "$work" --fields gncL1 \
        --bin-fields gncL1bin --schemes "$work/gncL1.fvSchemes" --out "$out/gpu2.gnc" \
        --write-dir gncL1out2 > "$out/log.gpu2" 2>&1
    rc_gpu2=$?

    compare="-"; cmp_result="-"; readback="-"
    if [ "$rc_ref" -eq 0 ] && [ -f "$out/gpu1.gnc" ]; then
        "$bin/gncCompare" "$out/ref.gnc" "$out/gpu1.gnc" > "$out/compare.txt" 2>&1
        case $? in 0) compare="BITWISE" ;; 1) compare="DIFF" ;; *) compare="ERROR" ;; esac
    fi
    if [ -f "$out/gpu1.gnc" ] && [ -f "$out/gpu2.gnc" ]; then
        if cmp -s "$out/gpu1.gnc" "$out/gpu2.gnc"; then cmp_result="identical"; else cmp_result="different"; fi
    fi
    if [ -d "$work/gncL1out" ]; then
        foam "$bin/gncOpsRef" -case "$work" -dump-fields gncL1 -out "$out/fields.ref.gnc" > "$out/log.dump.ref" 2>&1
        foam "$bin/gncOpsRef" -case "$work" -dump-fields gncL1out -out "$out/fields.gpu.gnc" > "$out/log.dump.gpu" 2>&1
        "$bin/gncCompare" "$out/fields.ref.gnc" "$out/fields.gpu.gnc" > "$out/compare.fields.txt" 2>&1
        case $? in 0) readback="BITWISE" ;; 1) readback="DIFF" ;; *) readback="ERROR" ;; esac
    fi
    reader="$(grep -c 'ascii vs binary: 0 values differ' "$out/log.gpu1" 2>/dev/null)/4"
    ncells="$(sed -n 's/^mesh: nCells \([0-9]*\).*/\1/p' "$out/log.gpu1" 2>/dev/null)"
    scheme="$(sed -n "s/^grad schemes: s '\([^']*\)'.*/\1/p" "$out/log.gpu1" 2>/dev/null)"
    sha256sum "$out"/*.gnc > "$out/SHA256SUMS" 2>/dev/null

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$ncells" "$scheme" "$rc_gen" "$rc_ref" "$rc_gpu1/$rc_gpu2" \
        "$compare" "$cmp_result" "$reader" "$readback" >> "$out_root/SUMMARY.tsv"
    echo "   generate=$rc_gen ref=$rc_ref gpu=$rc_gpu1/$rc_gpu2 compare=$compare gpu1_vs_gpu2=$cmp_result reader=$reader readback=$readback"

    if [ "$compare" != "BITWISE" ] || [ "$cmp_result" != "identical" ] \
       || [ "$reader" != "4/4" ] || [ "$readback" != "BITWISE" ]; then
        overall=1
    fi
done

nvidia-smi > "$out_root/nvidia-smi.after.txt" 2>&1
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out_root/RUN_INFO"
echo "overall_exit=$overall" >> "$out_root/RUN_INFO"
column -t -s "$(printf '\t')" "$out_root/SUMMARY.tsv"
exit "$overall"
