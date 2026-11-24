#!/usr/bin/env bash
# N3b 的 L2：第一個外迭代的通量與殘差，GPU-Native-CFD 對 HiSA 本身逐位元比對。
#
# 每個案例在 $RUN_ROOT/LABEL/NAME/case 建工作副本（system、constant、0 複製，
# constant/polyMesh 以符號連結；原案例不寫入），改成穩態變體：
#   fvSchemes   ddtSchemes/default = bounded dualTime rPseudoDeltaT steadyState
#   controlDict startTime 0、endTime 1、deltaT 1（穩態時一步就是一個外迭代），
#               加上 function object gncHisaDump（傾印 rhoR rhoUR rhoER phi；
#               層流案例另有 tauMC）
#   turbulenceProperties  --laminar NAME 的案例改為 simulationType laminar
#               （N3 的黏性驗證用層流；0/ 的湍流場不會被讀）
# 然後：
#   1. foamDictionary -expand -precision 17 → case/gncExpanded/（p T U
#      thermophysicalProperties fvSchemes，有 turbulenceProperties 時一併展開）
#   2. ref/1.gnc   hisa（正式執行一步）；GMRES 只讀殘差，外迭代結束時的
#                  rhoR 等即該迭代的殘差，phi 為該迭代的通量
#   3. gpu1.gnc、gpu2.gnc   gncResL2（兩個程序，cmp 比對）
#   4. compare.txt          gncCompare ref/1.gnc gpu1.gnc
#   5. 層流案例的反向對照：gpu-nocompat.gnc（gncResL2 --no-foam-compat，關閉
#      參考端行為的相容設定，core/foamCompat.hpp）與 compare-nocompat.txt。
#      SUMMARY 的 nocompat 欄記錄它與參考是否相同；三維案例應不同、二維相同
#
# 用法：l2_res_verify.sh --host-env FILE --label LABEL --case NAME=DIR [--case ...]
#                        [--laminar NAME ...]

set -u

host_env=""; label=""
cases=(); laminar=" "
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --case) cases+=("$2"); shift 2 ;;
        --laminar) laminar="$laminar$2 "; shift 2 ;;
        -h|--help) sed -n '2,23p' "$0"; exit 0 ;;
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
for exe in gncResL2 gncCompare; do
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

printf 'case\tnCells\tref\tgpu\tcompare\tgpu1_vs_gpu2\tnocompat\n' > "$out_root/SUMMARY.tsv"
overall=0

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

    # 穩態變體
    is_laminar=0
    case "$laminar" in *" $name "*) is_laminar=1 ;; esac
    {
        foam foamDictionary -entry ddtSchemes/default \
            -set "bounded dualTime rPseudoDeltaT steadyState" "$work/system/fvSchemes"
        for kv in startFrom=startTime startTime=0 endTime=1 deltaT=1 \
                  writeControl=timeStep writeInterval=1000000; do
            foam foamDictionary -entry "${kv%%=*}" -set "${kv#*=}" "$work/system/controlDict"
        done
        if [ "$is_laminar" -eq 1 ]; then
            foam foamDictionary -entry simulationType -set laminar \
                "$work/constant/turbulenceProperties"
            echo "laminar=1"
        fi
    } > "$out/log.variant" 2>&1
    dump_fields="rhoR rhoUR rhoER phi"
    [ "$is_laminar" -eq 1 ] && dump_fields="$dump_fields tauMC"
    cat >> "$work/system/controlDict" <<EOF

functions
{
    gncDump
    {
        type        gncHisaDump;
        libs        ("$lib");
        outputDir   "$out/ref";
        fields      ($dump_fields);
    }
}
EOF

    for f in 0/p 0/T 0/U constant/thermophysicalProperties system/fvSchemes \
             constant/turbulenceProperties; do
        [ -f "$work/$f" ] || continue
        foam foamDictionary -expand -precision 17 "$work/$f" \
            > "$work/gncExpanded/$(basename "$f")" 2>> "$out/log.expand"
    done

    (cd "$work" && foam hisa) > "$out/log.ref" 2>&1
    rc_ref=$?
    [ -f "$out/ref/1.gnc" ] || rc_ref=9

    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$bin/gncResL2" --case "$work" \
        --expanded "$work/gncExpanded" --out "$out/gpu1.gnc" > "$out/log.gpu1" 2>&1
    rc_gpu1=$?
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$bin/gncResL2" --case "$work" \
        --expanded "$work/gncExpanded" --out "$out/gpu2.gnc" > "$out/log.gpu2" 2>&1
    rc_gpu2=$?

    nocompat="-"
    if [ "$is_laminar" -eq 1 ] && [ "$rc_ref" -eq 0 ]; then
        CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$bin/gncResL2" --case "$work" \
            --expanded "$work/gncExpanded" --out "$out/gpu-nocompat.gnc" --no-foam-compat \
            > "$out/log.gpu-nocompat" 2>&1 \
        && "$bin/gncCompare" "$out/ref/1.gnc" "$out/gpu-nocompat.gnc" > "$out/compare-nocompat.txt" 2>&1
        case $? in 0) nocompat="BITWISE" ;; 1) nocompat="DIFF" ;; *) nocompat="ERROR" ;; esac
    fi

    compare="-"; cmp_result="-"
    if [ "$rc_ref" -eq 0 ] && [ "$rc_gpu1" -eq 0 ]; then
        "$bin/gncCompare" "$out/ref/1.gnc" "$out/gpu1.gnc" > "$out/compare.txt" 2>&1
        case $? in 0) compare="BITWISE" ;; 1) compare="DIFF" ;; *) compare="ERROR" ;; esac
    fi
    if [ "$rc_gpu1" -eq 0 ] && [ "$rc_gpu2" -eq 0 ]; then
        if cmp -s "$out/gpu1.gnc" "$out/gpu2.gnc"; then cmp_result="identical"; else cmp_result="different"; fi
    fi
    sha256sum "$out"/*.gnc "$out"/ref/*.gnc > "$out/SHA256SUMS" 2>/dev/null
    ncells="$(sed -n 's/^mesh: nCells \([0-9]*\).*/\1/p' "$out/log.gpu1" 2>/dev/null)"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$ncells" "$rc_ref" "$rc_gpu1/$rc_gpu2" "$compare" "$cmp_result" "$nocompat" \
        >> "$out_root/SUMMARY.tsv"
    echo "   ref=$rc_ref gpu=$rc_gpu1/$rc_gpu2 compare=$compare gpu1_vs_gpu2=$cmp_result nocompat=$nocompat"
    if [ "$compare" != "BITWISE" ] || [ "$cmp_result" != "identical" ]; then
        overall=1
    fi
done

nvidia-smi > "$out_root/nvidia-smi.after.txt" 2>&1
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out_root/RUN_INFO"
echo "overall_exit=$overall" >> "$out_root/RUN_INFO"
column -t -s "$(printf '\t')" "$out_root/SUMMARY.tsv"
exit "$overall"
