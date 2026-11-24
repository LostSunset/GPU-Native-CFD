#!/usr/bin/env bash
# 獨立求解器 gncHisa 的驗證：它寫出的場檔對 HiSA 寫出的場檔逐位元比對（N-017）。
#
# 每個案例在 $RUN_ROOT/LABEL/NAME 建穩態變體（同 l3_steady_verify.sh：
# steadyState、startTime 0、deltaT 1、endTime N），另設 writeControl timeStep、
# writeInterval W、writeFormat binary、writePrecision 17（參考端的非均勻值以二進位寫出；
# 全部相同的值 OpenFOAM 一律寫成 uniform 文字，以 17 位有效數字寫出），
# functions 清空。--laminar、--meshwave、--pseudo-tol、--pseudo-tol-rel 同
# l3_steady_verify.sh。
#   ref/   工作副本，執行 hisa
#   gnc/   同一個變體的另一份副本，scripts/gnc_prepare_case.sh 之後執行 gncHisa
# 比對：兩邊寫出的時間目錄清單相同；每個時間目錄的 p T U rho（Spalart–Allmaras
# 時另有 nuTilda nut alphat，kOmegaSST 時另有 k omega nut alphat，以參考端有寫出的
# 為準）以 gncFieldDiff 比對內部值與
# patch 的 value（compare.<time>.<field>.txt）。
# --forces NAME=FUNC：controlDict 先在原案例目錄內展開（#includeFunc），functions 只留
# FUNC（forces 或 forceCoeffs，照案例原本的 executeControl／writeControl）；兩邊的
# postProcessing/FUNC/ 下的檔案逐位元組比對（cmp），計入 compared／bitwise。
#
# 用法：app_verify.sh --host-env FILE --label LABEL --case NAME=DIR [--case ...]
#                     --end N --write-interval W [--laminar NAME ...] [--meshwave NAME ...]
#                     [--pseudo-tol NAME=V] [--pseudo-tol-rel NAME=V] [--forces NAME=FUNC]

set -u

host_env=""; label=""; endt=""; wi=""
cases=(); laminar=" "; meshwave=" "
declare -A tol_for=() tolrel_for=() forces_for=()
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --case) cases+=("$2"); shift 2 ;;
        --laminar) laminar="$laminar$2 "; shift 2 ;;
        --meshwave) meshwave="$meshwave$2 "; shift 2 ;;
        --end) endt="$2"; shift 2 ;;
        --write-interval) wi="$2"; shift 2 ;;
        --pseudo-tol) tol_for["${2%%=*}"]="${2#*=}"; shift 2 ;;
        --pseudo-tol-rel) tolrel_for["${2%%=*}"]="${2#*=}"; shift 2 ;;
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
[ -n "$endt" ] && [ -n "$wi" ] || { echo "ERROR: --end N and --write-interval W required" >&2; exit 2; }

here="$(cd "$(dirname "$0")" && pwd)"
bin="$BUILD_ROOT/bin"
for exe in gncHisa gncFieldDiff; do
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

printf 'case\tnCells\tref\tgnc\tref_dirs\tgnc_dirs\tcompared\tbitwise\n' > "$out_root/SUMMARY.tsv"
overall=0

for spec in "${cases[@]}"; do
    name="${spec%%=*}"; src="${spec#*=}"
    out="$out_root/$name"
    mkdir -p "$out"
    echo "== $name ($src) endTime=$endt writeInterval=$wi"
    echo "source=$src" > "$out/SOURCE"
    copy_case "$src" "$out/ref"

    w="$out/ref"
    func="${forces_for[$name]:-}"
    if [ -n "$func" ]; then
        (cd "$src" && foam foamDictionary -expand -precision 17 system/controlDict) \
            > "$w/system/controlDict" 2> "$out/log.forces"
    fi
    {
        foam foamDictionary -entry ddtSchemes/default \
            -set "bounded dualTime rPseudoDeltaT steadyState" "$w/system/fvSchemes"
        for kv in startFrom=startTime startTime=0 "endTime=$endt" deltaT=1 \
                  writeControl=timeStep "writeInterval=$wi" writeFormat=binary writePrecision=17; do
            foam foamDictionary -entry "${kv%%=*}" -set "${kv#*=}" "$w/system/controlDict"
        done
        fdict=""
        if [ -n "$func" ]; then
            fdict="$(foam foamDictionary -precision 17 -entry "functions/$func" "$w/system/controlDict")"
            [ -n "$fdict" ] || echo "ERROR: function $func not found"
            echo "forces=$func"
        fi
        foam foamDictionary -entry functions -set "{ $fdict }" "$w/system/controlDict"
        case "$meshwave" in
            *" $name "*)
                foam foamDictionary -entry wallDist/method -set meshWave "$w/system/fvSchemes"
                echo "meshwave=1" ;;
        esac
        case "$laminar" in
            *" $name "*)
                foam foamDictionary -entry simulationType -set laminar \
                    "$w/constant/turbulenceProperties"
                echo "laminar=1" ;;
        esac
        if [ -n "${tol_for[$name]:-}" ]; then
            v="${tol_for[$name]}"
            foam foamDictionary -entry pseudoTime/pseudoTol -set "$v ($v $v $v) $v" "$w/system/fvSolution"
        fi
        if [ -n "${tolrel_for[$name]:-}" ]; then
            v="${tolrel_for[$name]}"
            foam foamDictionary -entry pseudoTime/pseudoTolRel -set "$v ($v $v $v) $v" "$w/system/fvSolution"
        fi
    } > "$out/log.variant" 2>&1
    copy_case "$w" "$out/gnc"

    (cd "$w" && foam hisa) > "$out/log.ref" 2>&1
    rc_ref=$?
    bash "$here/gnc_prepare_case.sh" -case "$out/gnc" --host-env "$host_env" > "$out/log.prepare" 2>&1
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$bin/gncHisa" -case "$out/gnc" > "$out/log.gnc" 2>&1
    rc_gnc=$?

    list_dirs() {
        (cd "$1" && for d in *; do
            [ -d "$d" ] || continue
            case "$d" in 0|constant|system|gncExpanded) continue ;; esac
            [ -f "$d/p" ] && echo "$d"
        done | sort -g)
    }
    ref_dirs="$(list_dirs "$out/ref" | tr '\n' ' ')"
    gnc_dirs="$(list_dirs "$out/gnc" | tr '\n' ' ')"
    compared=0; bitwise=0
    for d in $ref_dirs; do
        for f in p T U rho nuTilda k omega nut alphat; do
            [ -f "$out/ref/$d/$f" ] || continue
            compared=$((compared + 1))
            "$bin/gncFieldDiff" "$out/ref/constant/polyMesh" "$out/ref/$d/$f" "$out/gnc/$d/$f" \
                > "$out/compare.$d.$f.txt" 2>&1 && bitwise=$((bitwise + 1))
        done
    done
    if [ -n "$func" ]; then
        : > "$out/compare.postProcessing.txt"
        for f in $(cd "$out/ref" && find "postProcessing/$func" -type f | sort); do
            compared=$((compared + 1))
            if cmp "$out/ref/$f" "$out/gnc/$f" >> "$out/compare.postProcessing.txt" 2>&1; then
                bitwise=$((bitwise + 1))
                echo "IDENTICAL $f" >> "$out/compare.postProcessing.txt"
            fi
            cp "$out/gnc/$f" "$out/$(basename "$f" .dat).gnc.dat" 2>/dev/null
            cp "$out/ref/$f" "$out/$(basename "$f" .dat).ref.dat"
        done
    fi
    ncells="$(sed -n 's/^mesh: \([0-9]*\) cells.*/\1/p' "$out/log.gnc")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$ncells" "$rc_ref" "$rc_gnc" \
        "$ref_dirs" "$gnc_dirs" "$compared" "$bitwise" >> "$out_root/SUMMARY.tsv"
    echo "   ref=$rc_ref gnc=$rc_gnc dirs ref=[$ref_dirs] gnc=[$gnc_dirs] bitwise=$bitwise/$compared"
    if [ "$rc_ref" -ne 0 ] || [ "$rc_gnc" -ne 0 ] || [ "$ref_dirs" != "$gnc_dirs" ] \
       || [ "$compared" -eq 0 ] || [ "$bitwise" -ne "$compared" ]; then
        overall=1
    fi
done

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out_root/RUN_INFO"
echo "overall_exit=$overall" >> "$out_root/RUN_INFO"
column -t -s "$(printf '\t')" "$out_root/SUMMARY.tsv"
exit "$overall"
