#!/usr/bin/env bash
# 效能比較（N7）：同一個穩態變體，HiSA（序列，及可選的 MPI 並行）與 gncHisa
# （-profile：外迭代各段的牆鐘時間）各跑 N 步，記錄總時間與每步時間。
#
# 變體同 app_verify.sh（steadyState、startTime 0、deltaT 1、endTime N），但
# writeInterval 設為大於 N（只在結束時寫一次），functions 清空。
#   hisa-np1/      hisa 序列
#   hisa-np<k>/    decomposePar（DECOMP_METHOD，預設 metis；g02 的 Atlas runtime 沒有 scotch）後
#                  mpirun -np k hisa -parallel（--np-list）；
#                  mpirun 的選項預設 --bind-to none（MPIRUN_OPTS），共用主機上不與其他
#                  作業綁到同樣的核心
#   gnc/           gnc_prepare_case.sh 後 gncHisa -profile
# 時間：HiSA 取 log 最後一行的 ExecutionTime（CPU 時間）與 ClockTime；gncHisa 取
# 程序的牆鐘時間（/usr/bin/time）與 -profile 的分段平均。主機不閒置時的時間只作
# 量級參考（README 原則 6）；RUN_INFO 記下 uptime 與量測前後的負載。
#
# 用法：perf_compare.sh --host-env FILE --label LABEL --case NAME=DIR [--case ...]
#                       --end N [--laminar NAME ...] [--np-list 8,32] [--no-serial] [--no-gnc]
#                       [--forces NAME=FUNC]
#   --no-serial、--no-gnc：不跑 HiSA 序列、不跑 gncHisa（例如長的 run 只要 MPI 的 HiSA）
#   --forces NAME=FUNC：同 variant_compare.sh（controlDict 在原案例目錄內展開，functions 只留
#     FUNC，每步執行並寫出）；各 run 的 postProcessing/FUNC/*.dat 複製為 <run>.<檔名>

set -u

host_env=""; label=""; endt=""; nplist=""; serial=1; gnc=1
cases=(); laminar=" "
declare -A forces_for=()
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --case) cases+=("$2"); shift 2 ;;
        --laminar) laminar="$laminar$2 "; shift 2 ;;
        --end) endt="$2"; shift 2 ;;
        --np-list) nplist="$2"; shift 2 ;;
        --no-serial) serial=0; shift ;;
        --no-gnc) gnc=0; shift ;;
        --forces) forces_for["${2%%=*}"]="${2#*=}"; shift 2 ;;
        -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -r "$host_env" ] || { echo "ERROR: --host-env FILE required" >&2; exit 2; }
# shellcheck disable=SC1090
. "$host_env"
case "$label" in ""|*/*|.*) echo "ERROR: --label NAME required" >&2; exit 2 ;; esac
[ "${#cases[@]}" -gt 0 ] || { echo "ERROR: at least one --case NAME=DIR" >&2; exit 2; }
[ -n "$endt" ] || { echo "ERROR: --end N required" >&2; exit 2; }

here="$(cd "$(dirname "$0")" && pwd)"
bin="$BUILD_ROOT/bin"
[ -x "$bin/gncHisa" ] || { echo "ERROR: $bin/gncHisa missing (run scripts/build.sh)" >&2; exit 2; }
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
    echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //') nproc=$(nproc)"
    echo "uptime_before=$(uptime)"
    echo "endTime=$endt np_list=${nplist:-none} mpirun_opts=${MPIRUN_OPTS---bind-to none} decomp=${DECOMP_METHOD:-metis}"
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

copy_case() {
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
}

hisa_times() {
    # 最後一行的 ExecutionTime、ClockTime
    grep -E '^ExecutionTime' "$1" | tail -1 | sed -n 's/ExecutionTime = \([0-9.e+-]*\) s.*ClockTime = \([0-9.e+-]*\) s.*/\1 \2/p'
}

# 迴圈內每步的時間：第一步與最後一步的 ExecutionTime（主程序的 CPU 時間，0.01 秒；MPI
# 等待時忙碌輪詢，約等於牆鐘）之差除以步數減一，不含啟動與第一步
hisa_loop() {
    grep -E '^ExecutionTime' "$1" | sed -n 's/ExecutionTime = \([0-9.e+-]*\) s.*/\1/p' \
        | awk 'NR == 1 {a = $1} {b = $1; n = NR} END {if (n > 1) printf "%.6g", (b - a)/(n - 1)}'
}

# wall_s：HiSA 的 ClockTime（整數秒）；shell_wall_s：shell 量的牆鐘（含啟動，奈秒）；
# loop_per_step_s：迴圈內每步（HiSA 見 hisa_loop，gncHisa 為 -profile 的平均）
printf 'case\tnCells\trun\texit\tcpu_s\twall_s\twall_per_step_s\tshell_wall_s\tloop_per_step_s\n' \
    > "$out_root/SUMMARY.tsv"

# postProcessing/FUNC/*.dat → <out>/<run>.<檔名>（--forces 時）
copy_forces() {
    [ -n "$func" ] || return 0
    local f
    for f in $(cd "$1" && find "postProcessing/$func" -type f -name "*.dat" 2>/dev/null | sort); do
        cp "$1/$f" "$out/$2.$(basename "$f")"
    done
}

for spec in "${cases[@]}"; do
    name="${spec%%=*}"; src="${spec#*=}"
    out="$out_root/$name"
    mkdir -p "$out"
    echo "== $name ($src) endTime=$endt"
    echo "source=$src" > "$out/SOURCE"
    base="$out/base"
    copy_case "$src" "$base"
    func="${forces_for[$name]:-}"
    if [ -n "$func" ]; then
        (cd "$src" && foam foamDictionary -expand -precision 17 system/controlDict) \
            > "$base/system/controlDict" 2> "$out/log.forces"
    fi
    {
        foam foamDictionary -entry ddtSchemes/default \
            -set "bounded dualTime rPseudoDeltaT steadyState" "$base/system/fvSchemes"
        for kv in startFrom=startTime startTime=0 "endTime=$endt" deltaT=1 \
                  writeControl=timeStep "writeInterval=$((endt + 1))"; do
            foam foamDictionary -entry "${kv%%=*}" -set "${kv#*=}" "$base/system/controlDict"
        done
        fdict=""
        if [ -n "$func" ]; then
            fdict="$(foam foamDictionary -precision 17 -entry "functions/$func" "$base/system/controlDict")"
            [ -n "$fdict" ] || echo "ERROR: function $func not found"
        fi
        foam foamDictionary -entry functions -set "{ $fdict }" "$base/system/controlDict"
        if [ -n "$func" ]; then
            for kv in executeControl=timeStep executeInterval=1 writeControl=timeStep writeInterval=1 log=off; do
                foam foamDictionary -entry "functions/$func/${kv%%=*}" -set "${kv#*=}" \
                    "$base/system/controlDict"
            done
            echo "forces=$func"
        fi
        case "$laminar" in
            *" $name "*)
                foam foamDictionary -entry simulationType -set laminar \
                    "$base/constant/turbulenceProperties"
                echo "laminar=1" ;;
        esac
    } > "$out/log.variant" 2>&1

    ncells=""
    # HiSA 序列
    if [ "$serial" -eq 1 ]; then
    copy_case "$base" "$out/hisa-np1"
    t0="$(date +%s.%N)"
    (cd "$out/hisa-np1" && foam hisa) > "$out/log.hisa-np1" 2>&1
    rc=$?
    t1="$(date +%s.%N)"
    read -r cpu wall <<< "$(hisa_times "$out/log.hisa-np1")"
    printf '%s\t%s\thisa-np1\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "-" "$rc" "${cpu:-}" "${wall:-}" \
        "$(awk -v w="${wall:-0}" -v n="$endt" 'BEGIN{printf "%.6g", w/n}')" \
        "$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.6g", b - a}')" \
        "$(hisa_loop "$out/log.hisa-np1")" >> "$out_root/SUMMARY.tsv"
    copy_forces "$out/hisa-np1" hisa-np1
    fi

    # HiSA MPI
    for np in ${nplist//,/ }; do
        w="$out/hisa-np$np"
        copy_case "$base" "$w"
        foam foamDictionary -entry numberOfSubdomains -set "$np" "$w/system/decomposeParDict" \
            > "$out/log.decompose-np$np" 2>&1 || true
        foam foamDictionary -entry method -set "${DECOMP_METHOD:-metis}" "$w/system/decomposeParDict" \
            >> "$out/log.decompose-np$np" 2>&1 || true
        (cd "$w" && foam decomposePar -force) >> "$out/log.decompose-np$np" 2>&1
        t0="$(date +%s.%N)"
        # shellcheck disable=SC2086
        (cd "$w" && foam mpirun ${MPIRUN_OPTS---bind-to none} -np "$np" hisa -parallel) \
            > "$out/log.hisa-np$np" 2>&1
        rc=$?
        t1="$(date +%s.%N)"
        read -r cpu wall <<< "$(hisa_times "$out/log.hisa-np$np")"
        printf '%s\t%s\thisa-np%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "-" "$np" "$rc" "${cpu:-}" "${wall:-}" \
            "$(awk -v w="${wall:-0}" -v n="$endt" 'BEGIN{printf "%.6g", w/n}')" \
            "$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.6g", b - a}')" \
            "$(hisa_loop "$out/log.hisa-np$np")" >> "$out_root/SUMMARY.tsv"
        copy_forces "$w" "hisa-np$np"
    done

    # gncHisa
    if [ "$gnc" -eq 1 ]; then
    copy_case "$base" "$out/gnc"
    bash "$here/gnc_prepare_case.sh" -case "$out/gnc" --host-env "$host_env" > "$out/log.prepare" 2>&1
    t0="$(date +%s.%N)"
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$bin/gncHisa" -case "$out/gnc" -profile > "$out/log.gnc" 2>&1
    rc=$?
    t1="$(date +%s.%N)"
    wall="$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.6g", b - a}')"
    ncells="$(sed -n 's/^mesh: \([0-9]*\) cells.*/\1/p' "$out/log.gnc")"
    pmean="$(sed -n 's/^profile: .*per iteration \([0-9.e+-]*\) s/\1/p' "$out/log.gnc")"
    printf '%s\t%s\tgncHisa\t%s\t-\t%s\t%s\t%s\t%s\n' "$name" "$ncells" "$rc" "$wall" \
        "$(awk -v w="$wall" -v n="$endt" 'BEGIN{printf "%.6g", w/n}')" "$wall" "${pmean:-}" \
        >> "$out_root/SUMMARY.tsv"
    grep -A1 '^profile:' "$out/log.gnc" > "$out/profile.txt" 2>/dev/null || true
    copy_forces "$out/gnc" gncHisa
    fi
    rm -rf "$base" "$out"/hisa-np*/processor* 2>/dev/null
done

echo "uptime_after=$(uptime)" >> "$out_root/RUN_INFO"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out_root/RUN_INFO"
nvidia-smi > "$out_root/nvidia-smi.after.txt" 2>&1
column -t -s "$(printf '\t')" "$out_root/SUMMARY.tsv"
for f in "$out_root"/*/profile.txt; do
    if [ -f "$f" ]; then
        echo "-- $f"; cat "$f"
    fi
done
exit 0
