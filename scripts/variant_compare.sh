#!/usr/bin/env bash
# 效能變體的比較（N7）：同一個穩態變體以 gncHisa 的不同環境變數設定各跑 N 步，
# 記錄每步的 GMRES 次數與初始殘差（殘差歷程）、牆鐘時間、-profile 分段平均，
# 並以 field_diff.py 比較各變體與第一個變體（參考次序版）的最終場。
#
# 不逐位元的選項（例如 GNC_LUSGS=color）改變預條件，收斂歷程會不同；收斂到同一個
# 離散穩態解時最終場只差到收斂容許的程度。這個腳本只量測，不判定通過與否。
#
# 變體同 perf_compare.sh（steadyState、startTime 0、deltaT 1、endTime N、
# functions 清空），writeInterval N（只寫最後一步），writePrecision 17。
# --forces NAME=FUNC：controlDict 先在原案例目錄內展開，functions 只留 FUNC，每步執行
# 並寫出（executeInterval 1、writeControl timeStep、writeInterval 1）；以
# coeff_summary.py 摘要各變體的力或力係數（coeffs.txt）。
#
# 用法：variant_compare.sh --host-env FILE --label LABEL --case NAME=DIR [--case ...]
#                          --end N --variant NAME=ENV [--variant ...] [--laminar NAME ...]
#                          [--forces NAME=FUNC]
#   ENV 為以空白分隔的 KEY=VALUE（可以是空字串，即預設設定），例如
#   --variant ref= --variant color=GNC_LUSGS=color

set -u

host_env=""; label=""; endt=""
cases=(); variants=(); laminar=" "
declare -A forces_for=()
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --case) cases+=("$2"); shift 2 ;;
        --variant) variants+=("$2"); shift 2 ;;
        --laminar) laminar="$laminar$2 "; shift 2 ;;
        --end) endt="$2"; shift 2 ;;
        --forces) forces_for["${2%%=*}"]="${2#*=}"; shift 2 ;;
        -h|--help) sed -n '2,21p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -r "$host_env" ] || { echo "ERROR: --host-env FILE required" >&2; exit 2; }
# shellcheck disable=SC1090
. "$host_env"
case "$label" in ""|*/*|.*) echo "ERROR: --label NAME required" >&2; exit 2 ;; esac
[ "${#cases[@]}" -gt 0 ] || { echo "ERROR: at least one --case NAME=DIR" >&2; exit 2; }
[ "${#variants[@]}" -gt 0 ] || { echo "ERROR: at least one --variant NAME=ENV" >&2; exit 2; }
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
    echo "uptime_before=$(uptime)"
    echo "endTime=$endt"
    for v in "${variants[@]}"; do echo "variant ${v%%=*}: ${v#*=}"; done
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

# 殘差歷程：step、GMRES 次數、五個初始殘差
history() {
    awk '
        /^Time = / { t = $3 }
        /GMRES iterations:/ {
            line = $0
            gsub(/[()]/, " ", line)
            n = split(line, a, " ")
            # GMRES iterations: K initial residual: r0 r1 r2 r3 r4
            printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\n", t, a[3], a[6], a[7], a[8], a[9], a[10]
        }
    ' "$1"
}

printf 'case\tnCells\tvariant\texit\twall_s\twall_per_step_s\tgmres_iter_total\tfinal_res_rho\tprofile_mean_s\n' \
    > "$out_root/SUMMARY.tsv"

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
                  writeControl=timeStep "writeInterval=$endt" writePrecision=17; do
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
    bash "$here/gnc_prepare_case.sh" -case "$base" --host-env "$host_env" > "$out/log.prepare" 2>&1

    first=""
    for v in "${variants[@]}"; do
        vn="${v%%=*}"; venv="${v#*=}"
        w="$out/$vn"
        copy_case "$base" "$w"
        cp -r "$base/gncExpanded" "$w/"
        t0="$(date +%s.%N)"
        # shellcheck disable=SC2086
        env $venv CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$bin/gncHisa" -case "$w" -profile > "$out/log.$vn" 2>&1
        rc=$?
        t1="$(date +%s.%N)"
        wall="$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.6g", b - a}')"
        ncells="$(sed -n 's/^mesh: \([0-9]*\) cells.*/\1/p' "$out/log.$vn")"
        history "$out/log.$vn" > "$out/history.$vn.tsv"
        itot="$(awk -F'\t' '{s += $2} END {print s + 0}' "$out/history.$vn.tsv")"
        rfin="$(tail -1 "$out/history.$vn.tsv" | cut -f3)"
        pmean="$(sed -n 's/^profile: .*per iteration \([0-9.e+-]*\) s/\1/p' "$out/log.$vn")"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "${ncells:-}" "$vn" "$rc" "$wall" \
            "$(awk -v x="$wall" -v n="$endt" 'BEGIN{printf "%.6g", x/n}')" "$itot" "${rfin:-}" \
            "${pmean:-}" >> "$out_root/SUMMARY.tsv"
        grep -A2 '^profile:' "$out/log.$vn" > "$out/profile.$vn.txt" 2>/dev/null || true
        if [ -n "$func" ]; then
            for f in $(cd "$w" && find "postProcessing/$func" -type f -name "*.dat" | sort); do
                cp "$w/$f" "$out/$vn.$(basename "$f")"
            done
        fi
        if [ -z "$first" ]; then
            first="$vn"
        else
            python3 "$here/field_diff.py" "$out/$first/$endt" "$w/$endt" > "$out/fielddiff.$first-$vn.txt" 2>&1
        fi
    done
    if [ -n "$func" ]; then
        python3 "$here/coeff_summary.py" "$out" > "$out/coeffs.txt" 2>&1
    fi
    rm -rf "$base"
done

echo "uptime_after=$(uptime)" >> "$out_root/RUN_INFO"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out_root/RUN_INFO"
column -t -s "$(printf '\t')" "$out_root/SUMMARY.tsv"
for f in "$out_root"/*/fielddiff.*.txt "$out_root"/*/coeffs.txt; do
    if [ -f "$f" ]; then
        echo "-- $f"; cat "$f"
    fi
done
exit 0
