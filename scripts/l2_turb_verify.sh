#!/usr/bin/env bash
# N4a-2、N4b 的 L2：Spalart–Allmaras 與 kOmegaSST 的 correct() 與 correctNut()
# （validate）對 HiSA 本身逐位元比對（模型依案例的 turbulenceProperties）。
#
# 每個案例在 $RUN_ROOT/LABEL/NAME/case 建穩態變體（同 l3_steady_verify.sh：
# bounded dualTime rPseudoDeltaT steadyState、startTime 0、deltaT 1，endTime 為
# --steps 的最大值，不寫場），functions 只有 gncState（gncHisaDump：U rho
# thermo:mu phi rPseudoDeltaT 與紊流場（SA：nuTilda nut alphat；kOmegaSST：k omega
# nut alphat），timeIndices 為每個 k 與 k-1）。--meshwave NAME 把該案例 fvSchemes
# 的 wallDist/method 改為 meshWave（N-022：Poisson 的案例以 meshWave 變體驗證）。
#   1. foamDictionary -expand -precision 17 → case/gncExpanded/（p T U nuTilda nut
#      alphat thermophysicalProperties turbulenceProperties fvSchemes fvSolution
#      controlDict）
#   2. ref/state/0.gnc     hisa -postProcess -time 0（createFields 與 validate 之後）
#      ref/state/<k>.gnc   hisa 執行時在時間步 k 結束時傾印
#   3. gpu1/、gpu2/        gncTurbL2（兩個程序，逐檔 cmp）
#   4. compare.<k>.txt     gncCompare gpu1/<k>.gnc ref/state/<k>.gnc：以 GPU 端的輸出
#                          為 REF，只比紊流場的內部值與邊界值（SA 的 0.gnc 只有
#                          nut、alphat）
#
# 用法：l2_turb_verify.sh --host-env FILE --label LABEL --case NAME=DIR [--case ...]
#                         --steps 1,2,10 [--meshwave NAME ...]

set -u

host_env=""; label=""; steps=""
cases=(); meshwave=" "
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --case) cases+=("$2"); shift 2 ;;
        --steps) steps="$2"; shift 2 ;;
        --meshwave) meshwave="$meshwave$2 "; shift 2 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -r "$host_env" ] || { echo "ERROR: --host-env FILE required" >&2; exit 2; }
# shellcheck disable=SC1090
. "$host_env"
case "$label" in ""|*/*|.*) echo "ERROR: --label NAME required" >&2; exit 2 ;; esac
[ "${#cases[@]}" -gt 0 ] || { echo "ERROR: at least one --case NAME=DIR" >&2; exit 2; }
[ -n "$steps" ] || { echo "ERROR: --steps 1,2,... required" >&2; exit 2; }

bin="$BUILD_ROOT/bin"
dumplib="$BUILD_ROOT/lib/libgncHisaDump.so"
base_fields="U rho thermo:mu phi rPseudoDeltaT"
for exe in gncTurbL2 gncCompare; do
    [ -x "$bin/$exe" ] || { echo "ERROR: $bin/$exe missing (run scripts/build.sh)" >&2; exit 2; }
done
[ -f "$dumplib" ] || { echo "ERROR: $dumplib missing (run scripts/build.sh)" >&2; exit 2; }

# 時間步：steps 與 steps-1（不含 0），endTime 為最大值
endt=0; idx=""
for k in ${steps//,/ }; do
    [ "$k" -gt "$endt" ] && endt="$k"
    idx="$idx $k"
    [ "$k" -gt 1 ] && idx="$idx $((k - 1))"
done
idx="$(echo "$idx" | tr ' ' '\n' | sort -n | uniq | tr '\n' ' ')"

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
    echo "steps=$steps"
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

printf 'case\tnCells\tref\tgpu\tcompared\tbitwise\tgpu1_vs_gpu2\n' > "$out_root/SUMMARY.tsv"
overall=0

for spec in "${cases[@]}"; do
    name="${spec%%=*}"; src="${spec#*=}"
    out="$out_root/$name"
    work="$out/case"
    mkdir -p "$work/constant" "$work/gncExpanded" "$out/ref/state" "$out/gpu1" "$out/gpu2"
    echo "== $name ($src) steps=$steps timeIndices=$idx"

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
    model="$(foam foamDictionary -entry RAS/RASModel -value "$work/constant/turbulenceProperties" 2>/dev/null)"
    case "$model" in
        SpalartAllmaras) fields="$base_fields nuTilda nut alphat" ;;
        kOmegaSST) fields="$base_fields k omega nut alphat" ;;
        *) echo "ERROR: $name: RASModel '$model' not supported" >&2; exit 2 ;;
    esac
    echo "model=$model" >> "$out/SOURCE"

    {
        case "$meshwave" in
            *" $name "*)
                foam foamDictionary -entry wallDist/method -set meshWave "$work/system/fvSchemes"
                echo "meshwave=1" >> "$out/SOURCE" ;;
        esac
        foam foamDictionary -entry ddtSchemes/default \
            -set "bounded dualTime rPseudoDeltaT steadyState" "$work/system/fvSchemes"
        for kv in startFrom=startTime startTime=0 "endTime=$endt" deltaT=1 \
                  writeControl=timeStep writeInterval=1000000; do
            foam foamDictionary -entry "${kv%%=*}" -set "${kv#*=}" "$work/system/controlDict"
        done
        foam foamDictionary -entry functions -set "{ gncState { type gncHisaDump; \
            libs (\"$dumplib\"); outputDir \"$out/ref/state\"; fields ($fields); \
            timeIndices ($idx); } }" "$work/system/controlDict"
    } > "$out/log.variant" 2>&1

    for f in 0/p 0/T 0/U 0/nuTilda 0/k 0/omega 0/nut 0/alphat constant/thermophysicalProperties \
             constant/turbulenceProperties system/fvSchemes system/fvSolution system/controlDict; do
        [ -f "$work/$f" ] || continue
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
        libs        ("$dumplib");
        outputDir   "$out/ref/state";
        fields      ($fields);
    }
}
EOF
    (cd "$work" && foam hisa -postProcess -dict system/gncDumpDict -time 0) > "$out/log.ref0" 2>&1
    rc_ref=$?
    (cd "$work" && foam hisa) > "$out/log.ref" 2>&1
    [ $? -ne 0 ] && rc_ref=1
    [ -f "$out/ref/state/0.gnc" ] || rc_ref=9

    for g in 1 2; do
        CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$bin/gncTurbL2" --case "$work" \
            --expanded "$work/gncExpanded" --ref-dir "$out/ref/state" --steps "$steps" \
            --out-dir "$out/gpu$g" > "$out/log.gpu$g" 2>&1
        rc=$?
        case "$g" in 1) rc_gpu1=$rc ;; *) rc_gpu2=$rc ;; esac
    done

    compared=0; bitwise=0; cmp_result="-"
    if [ "$rc_ref" -eq 0 ] && [ "$rc_gpu1" -eq 0 ]; then
        for k in 0 ${steps//,/ }; do
            compared=$((compared + 1))
            "$bin/gncCompare" "$out/gpu1/$k.gnc" "$out/ref/state/$k.gnc" > "$out/compare.$k.txt" 2>&1
            [ $? -eq 0 ] && bitwise=$((bitwise + 1))
        done
        if [ "$rc_gpu2" -eq 0 ]; then
            cmp_result=same
            for f in "$out"/gpu1/*.gnc; do
                cmp -s "$f" "$out/gpu2/$(basename "$f")" || cmp_result=differ
            done
        else
            cmp_result=gpu2_failed
        fi
    fi
    ncells="$(sed -n 's/^mesh: \([0-9]*\) cells.*/\1/p' "$out/log.gpu1")"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$ncells" "$rc_ref" "$rc_gpu1" \
        "$compared" "$bitwise" "$cmp_result" >> "$out_root/SUMMARY.tsv"
    echo "   ref=$rc_ref gpu=$rc_gpu1 bitwise=$bitwise/$compared gpu1_vs_gpu2=$cmp_result"
    if [ "$rc_ref" -ne 0 ] || [ "$rc_gpu1" -ne 0 ] || [ "$compared" -eq 0 ] \
       || [ "$bitwise" -ne "$compared" ] || [ "$cmp_result" != same ]; then
        overall=1
    fi
done

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out_root/RUN_INFO"
echo "overall_exit=$overall" >> "$out_root/RUN_INFO"
column -t -s "$(printf '\t')" "$out_root/SUMMARY.tsv"
exit "$overall"
