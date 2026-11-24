#!/usr/bin/env bash
# N3a 的 L2：HiSA 初始化（hisaModule::initialise，createFields.H）之後的狀態，
# GPU-Native-CFD 對 HiSA 本身逐位元比對。
#
# 每個案例在 $RUN_ROOT/LABEL/NAME/case 建工作副本（system、constant、0 複製，
# constant/polyMesh 以符號連結指回原案例；原案例不寫入），然後：
#   1. foamDictionary -expand -precision 17：0/p、0/T、0/U、
#      constant/thermophysicalProperties、system/fvSchemes → case/gncExpanded/
#      （17 位：預設的 6 位會改變數值，例如 -53.90455587）
#   2. ref/0.gnc   hisa -postProcess -dict system/gncDumpDict -time 0：HiSA 的
#                  initialise() 之後，function object gncHisaDump 從註冊表傾印
#   3. gpu1.gnc    gncInitL2；gpu2.gnc 第二個程序，以 cmp 比對
#   4. compare.txt gncCompare ref/0.gnc gpu1.gnc
#
# 用法：l2_init_verify.sh --host-env FILE --label LABEL --case NAME=DIR [--case ...]

set -u

host_env=""; label=""
cases=()
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --case) cases+=("$2"); shift 2 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -r "$host_env" ] || { echo "ERROR: --host-env FILE required" >&2; exit 2; }
# shellcheck disable=SC1090
. "$host_env"
case "$label" in ""|*/*|.*) echo "ERROR: --label NAME required" >&2; exit 2 ;; esac
[ "${#cases[@]}" -gt 0 ] || { echo "ERROR: at least one --case NAME=DIR" >&2; exit 2; }

bin="$BUILD_ROOT/bin"
lib="$BUILD_ROOT/lib/libgncHisaDump.so"
for exe in gncInitL2 gncCompare; do
    [ -x "$bin/$exe" ] || { echo "ERROR: $bin/$exe missing (run scripts/build.sh)" >&2; exit 2; }
done
[ -f "$lib" ] || { echo "ERROR: $lib missing (run scripts/build.sh)" >&2; exit 2; }

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

printf 'case\tnCells\tanalysis\tref\tgpu\tcompare\tgpu1_vs_gpu2\n' > "$out_root/SUMMARY.tsv"
overall=0
fields="p T U e thermo:psi thermo:mu thermo:alpha rho rhoU rhoE eZero phi"

for spec in "${cases[@]}"; do
    name="${spec%%=*}"; src="${spec#*=}"
    out="$out_root/$name"
    work="$out/case"
    mkdir -p "$work/constant" "$work/gncExpanded"
    echo "== $name ($src)"

    cp -r "$src/system" "$work/system"
    cp -r "$src/0" "$work/0"
    for f in "$src"/constant/*; do
        b="$(basename "$f")"
        if [ "$b" = polyMesh ]; then
            ln -s "$(realpath "$f")" "$work/constant/polyMesh"
        else
            cp -r "$f" "$work/constant/$b"
        fi
    done
    echo "source=$src" > "$out/SOURCE"

    for f in 0/p 0/T 0/U constant/thermophysicalProperties system/fvSchemes; do
        foam foamDictionary -expand -precision 17 "$work/$f" \
            > "$work/gncExpanded/$(basename "$f")" 2>> "$out/log.expand"
    done

    cat > "$work/system/gncDumpDict" <<EOF
FoamFile { version 2.0; format ascii; class dictionary; object gncDumpDict; }
functions
{
    gncDump
    {
        type        gncHisaDump;
        libs        ("$lib");
        outputDir   "$out/ref";
        fields      ($fields);
    }
}
EOF
    (cd "$work" && foam hisa -postProcess -dict system/gncDumpDict -time 0) > "$out/log.ref" 2>&1
    rc_ref=$?
    [ -f "$out/ref/0.gnc" ] || rc_ref=9

    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$bin/gncInitL2" --case "$work" \
        --expanded "$work/gncExpanded" --out "$out/gpu1.gnc" > "$out/log.gpu1" 2>&1
    rc_gpu1=$?
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$bin/gncInitL2" --case "$work" \
        --expanded "$work/gncExpanded" --out "$out/gpu2.gnc" > "$out/log.gpu2" 2>&1
    rc_gpu2=$?

    compare="-"; cmp_result="-"
    if [ "$rc_ref" -eq 0 ] && [ "$rc_gpu1" -eq 0 ]; then
        "$bin/gncCompare" "$out/ref/0.gnc" "$out/gpu1.gnc" > "$out/compare.txt" 2>&1
        case $? in 0) compare="BITWISE" ;; 1) compare="DIFF" ;; *) compare="ERROR" ;; esac
    fi
    if [ "$rc_gpu1" -eq 0 ] && [ "$rc_gpu2" -eq 0 ]; then
        if cmp -s "$out/gpu1.gnc" "$out/gpu2.gnc"; then cmp_result="identical"; else cmp_result="different"; fi
    fi
    sha256sum "$out"/*.gnc "$out"/ref/*.gnc > "$out/SHA256SUMS" 2>/dev/null
    ncells="$(sed -n 's/^mesh: nCells \([0-9]*\).*/\1/p' "$out/log.gpu1" 2>/dev/null)"
    analysis="$(sed -n 's/^analysis: //p' "$out/log.gpu1" 2>/dev/null)"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$ncells" "$analysis" "$rc_ref" "$rc_gpu1/$rc_gpu2" "$compare" "$cmp_result" \
        >> "$out_root/SUMMARY.tsv"
    echo "   ref=$rc_ref gpu=$rc_gpu1/$rc_gpu2 compare=$compare gpu1_vs_gpu2=$cmp_result"
    if [ "$compare" != "BITWISE" ] || [ "$cmp_result" != "identical" ]; then
        overall=1
    fi
done

nvidia-smi > "$out_root/nvidia-smi.after.txt" 2>&1
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out_root/RUN_INFO"
echo "overall_exit=$overall" >> "$out_root/RUN_INFO"
column -t -s "$(printf '\t')" "$out_root/SUMMARY.tsv"
exit "$overall"
