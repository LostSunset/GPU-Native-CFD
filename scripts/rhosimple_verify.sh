#!/usr/bin/env bash
# N6 的 L2／L3：gncRhoSimple 對 rhoSimpleFoam（tools/ref/gncRhoSimpleRef，主程式與
# createFields、UEqn、EEqn、pEqn 照原樣，只加傾印點）逐位元比對（N-023）。
#
# 每個案例在 $RUN_ROOT/LABEL/NAME 建工作副本（system、constant、0 複製，
# constant/polyMesh 以符號連結；原案例不寫入）。--end NAME=N 把 controlDict 的
# endTime 改為 N（其他照案例原本的設定：residualControl 收斂時兩邊都在下一個迭代
# 開頭結束並寫出）。然後：
#   1. foamDictionary -expand -precision 17 → gncExpanded/（scripts/gnc_prepare_case.sh）
#   2. ref/case：gncRhoSimpleRef -dumpDir ref/dump -dumpSteps (K ...)，寫出時間目錄
#   3. gpu1/case、gpu2/case：gncRhoSimple -dumpDir gpuN/dump -dumpSteps K ...（兩個程序）
#   4. L2：gncCompare ref/dump/<k>.<stage>.gnc gpu1/dump/<k>.<stage>.gnc（0.init 與每個 K 的
#      U、E、p、turb），結果在 compare.<k>.<stage>.txt
#   5. L3：ref 與 gpu1 寫出的時間目錄集合（含收斂時的最後一個）相同，每個時間目錄的
#      每個場檔以 gncFieldDiff 比對（OpenFOAM 讀到的 double 逐位元），結果在
#      fielddiff.txt；gpu1 與 gpu2 的場檔逐位元相同（cmp）
#
# 用法：rhosimple_verify.sh --host-env FILE --label LABEL --case NAME=DIR [--case ...]
#                           [--dump-steps 1,2,10] [--end NAME=N ...]

set -u

host_env=""; label=""; steps="1,2,10"
cases=()
declare -A end_for=()
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --case) cases+=("$2"); shift 2 ;;
        --dump-steps) steps="$2"; shift 2 ;;
        --end) end_for["${2%%=*}"]="${2#*=}"; shift 2 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -r "$host_env" ] || { echo "ERROR: --host-env FILE required" >&2; exit 2; }
# shellcheck disable=SC1090
. "$host_env"
case "$label" in ""|*/*|.*) echo "ERROR: --label NAME required" >&2; exit 2 ;; esac
[ "${#cases[@]}" -gt 0 ] || { echo "ERROR: at least one --case NAME=DIR" >&2; exit 2; }

here="$(cd "$(dirname "$0")" && pwd)"
bin="$BUILD_ROOT/bin"
ref="$BUILD_ROOT/bin/gncRhoSimpleRef"
for exe in gncRhoSimple gncCompare gncFieldDiff; do
    [ -x "$bin/$exe" ] || { echo "ERROR: $bin/$exe missing (run scripts/build.sh)" >&2; exit 2; }
done
[ -x "$ref" ] || { echo "ERROR: $ref missing (run scripts/build.sh)" >&2; exit 2; }

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
    echo "dump_steps=$steps"
    echo "uptime=$(uptime)"
} > "$out_root/RUN_INFO"
cp "$BUILD_ROOT/BUILD_INFO" "$out_root/BUILD_INFO"

foam() {
    (
        set +u
        # shellcheck disable=SC1090
        . "$OPENFOAM_BASHRC" >/dev/null 2>&1 || true
        set -u
        "$@"
    )
}

# 數字目錄（時間目錄，不含 0），依數值排序
timedirs() {
    (cd "$1" && for d in *; do
        case "$d" in
            0|*[!0-9.e+-]*) ;;
            *) [ -d "$d" ] && echo "$d" ;;
        esac
    done) | LC_ALL=C sort -g
}

printf 'case\tnCells\tlast_ref\tlast_gpu\tdumps\tdumps_bitwise\tfields\tfields_bitwise\tgpu1_vs_gpu2\n' \
    > "$out_root/SUMMARY.tsv"
overall=0

for spec in "${cases[@]}"; do
    name="${spec%%=*}"; src="${spec#*=}"
    out="$out_root/$name"
    work="$out/work"
    mkdir -p "$work/constant"
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
    if [ -n "${end_for[$name]:-}" ]; then
        foam foamDictionary -entry endTime -set "${end_for[$name]}" "$work/system/controlDict" \
            > /dev/null
        echo "endTime=${end_for[$name]}" >> "$out/SOURCE"
    fi
    foam foamDictionary -entry functions -set "{}" "$work/system/controlDict" > /dev/null 2>&1
    bash "$here/gnc_prepare_case.sh" -case "$work" --host-env "$host_env" > "$out/log.expand" 2>&1

    # 參考
    mkdir -p "$out/ref"
    cp -r "$work" "$out/ref/case"
    (cd "$out/ref/case" && foam "$ref" -dumpDir "$out/ref/dump" -dumpSteps "(${steps//,/ })") \
        > "$out/log.ref" 2>&1
    rc_ref=$?

    # GPU（兩個程序）
    for g in gpu1 gpu2; do
        mkdir -p "$out/$g"
        cp -r "$work" "$out/$g/case"
        (cd "$out/$g/case" && CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$bin/gncRhoSimple" \
            -dumpDir "$out/$g/dump" -dumpSteps "${steps//,/ }") > "$out/log.$g" 2>&1
        echo "exit=$?" >> "$out/log.$g"
    done
    rc_gpu="$(tail -1 "$out/log.gpu1")"

    # L2：傾印
    nd=0; nbit=0
    for f in "$out/ref/dump"/*.gnc; do
        b="$(basename "$f" .gnc)"
        nd=$((nd + 1))
        if "$bin/gncCompare" "$f" "$out/gpu1/dump/$b.gnc" > "$out/compare.$b.txt" 2>&1; then
            nbit=$((nbit + 1))
        fi
    done

    # L3：時間目錄
    last_ref="$(timedirs "$out/ref/case" | tail -1)"
    last_gpu="$(timedirs "$out/gpu1/case" | tail -1)"
    nf=0; nfbit=0
    : > "$out/fielddiff.txt"
    if [ "$(timedirs "$out/ref/case" | tr '\n' ' ')" != "$(timedirs "$out/gpu1/case" | tr '\n' ' ')" ]; then
        echo "TIME DIRECTORIES DIFFER: ref=($(timedirs "$out/ref/case" | tr '\n' ' ')) gpu=($(timedirs "$out/gpu1/case" | tr '\n' ' '))" \
            >> "$out/fielddiff.txt"
    fi
    for t in $(timedirs "$out/ref/case"); do
        for f in "$out/ref/case/$t"/*; do
            [ -f "$f" ] || continue
            b="$(basename "$f")"
            nf=$((nf + 1))
            if [ -f "$out/gpu1/case/$t/$b" ] && \
               "$bin/gncFieldDiff" "$work/constant/polyMesh" "$f" "$out/gpu1/case/$t/$b" \
                   > "$out/fd.tmp" 2>&1; then
                nfbit=$((nfbit + 1))
                echo "BITWISE $t/$b" >> "$out/fielddiff.txt"
            else
                echo "DIFF    $t/$b" >> "$out/fielddiff.txt"
                sed 's/^/    /' "$out/fd.tmp" >> "$out/fielddiff.txt"
            fi
        done
    done
    rm -f "$out/fd.tmp"
    g12="identical"
    for t in $(timedirs "$out/gpu1/case"); do
        for f in "$out/gpu1/case/$t"/*; do
            cmp -s "$f" "$out/gpu2/case/$t/$(basename "$f")" || g12="differ"
        done
    done
    ncells="$(grep -m1 '^mesh:' "$out/log.gpu1" | awk '{print $2}')"
    printf '%s\t%s\t%s\t%s\t%d\t%d\t%d\t%d\t%s\n' "$name" "$ncells" "$last_ref" "$last_gpu" \
        "$nd" "$nbit" "$nf" "$nfbit" "$g12" >> "$out_root/SUMMARY.tsv"
    echo "   ref rc=$rc_ref gpu $rc_gpu; dumps $nbit/$nd bitwise; fields $nfbit/$nf bitwise; gpu1 vs gpu2 $g12"
    if [ "$rc_ref" -ne 0 ] || [ "$rc_gpu" != "exit=0" ] || [ "$nbit" -ne "$nd" ] || [ "$nd" -eq 0 ] \
       || [ "$nfbit" -ne "$nf" ] || [ "$nf" -eq 0 ] || [ "$last_ref" != "$last_gpu" ] || [ "$g12" != identical ]; then
        overall=1
    fi
    rm -rf "$work"
done

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out_root/RUN_INFO"
column -t -s $'\t' "$out_root/SUMMARY.tsv"
exit $overall
