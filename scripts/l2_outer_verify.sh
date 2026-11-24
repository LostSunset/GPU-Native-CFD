#!/usr/bin/env bash
# N3c 的 L2：第一個外迭代（偽時間步、耦合 Jacobian、LU-SGS、GMRES、限界與
# updateFields），GPU-Native-CFD 對 HiSA 本身逐位元比對（N-014）。
#
# 每個案例在 $RUN_ROOT/LABEL/NAME/case 建工作副本（system、constant、0 複製，
# constant/polyMesh 以符號連結；原案例不寫入），改成穩態變體：
#   fvSchemes   ddtSchemes/default = bounded dualTime rPseudoDeltaT steadyState
#   controlDict startTime 0、endTime 1、deltaT 1（一步就是一個外迭代），
#               libs 載入 libgncHisaJacobianDump.so；functions 換成只有
#               gncState（gncHisaDump，outputDir 為 ref/state；案例原有的
#               function object 只做後處理，不影響場）
#   fvSolution  flowSolver/<solver>/preconditioner = gncLUSGSDump（繼承 HiSA
#               的 LU-SGS，數值不變），outputDir 為 ref/
#   turbulenceProperties  --laminar NAME 的案例改為 simulationType laminar
# 然後：
#   1. foamDictionary -expand -precision 17 → case/gncExpanded/（p T U
#      thermophysicalProperties fvSchemes fvSolution，有 turbulenceProperties
#      時一併展開）
#   2. ref/1.jacobian.gnc      hisa（正式執行一步）時建構預條件器那一刻的
#                              Jacobian、rPseudoDeltaT、rDiagCoeff、V
#      ref/1.precondition.gnc  第一次 precondition() 的輸入與輸出（P.in、P.out）
#      ref/1.solve.gnc         GMRES 更新後、限界之前的 rho、rhoU、rhoE
#      ref/state/1.gnc         時間步結束（限界、updateFields 之後）的
#                              p T U e psi mu alpha rho rhoU rhoE phi
#                              pseudoCoDamping
#   3. gpu{1,2}.gnc、gpu{1,2}.state.gnc  gncOuterL2（兩個程序，cmp 比對）
#   4. compare.<k>.txt         gncCompare ref/1.<k>.gnc gpu1.gnc，k = jacobian、
#                              precondition、solve；compare.state.txt 為
#                              gncCompare ref/state/1.gnc gpu1.state.gnc
#                              （以參考檔的紀錄為準）
#
# 用法：l2_outer_verify.sh --host-env FILE --label LABEL --case NAME=DIR [--case ...]
#                          [--laminar NAME ...]

set -u

host_env=""; label=""
cases=(); laminar=" "
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --case) cases+=("$2"); shift 2 ;;
        --laminar) laminar="$laminar$2 "; shift 2 ;;
        -h|--help) sed -n '2,38p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -r "$host_env" ] || { echo "ERROR: --host-env FILE required" >&2; exit 2; }
# shellcheck disable=SC1090
. "$host_env"
case "$label" in ""|*/*|.*) echo "ERROR: --label NAME required" >&2; exit 2 ;; esac
[ "${#cases[@]}" -gt 0 ] || { echo "ERROR: at least one --case NAME=DIR" >&2; exit 2; }

bin="$BUILD_ROOT/bin"
lib="$BUILD_ROOT/lib/libgncHisaJacobianDump.so"
dumplib="$BUILD_ROOT/lib/libgncHisaDump.so"
fields="p T U e thermo:psi thermo:mu thermo:alpha rho rhoU rhoE phi pseudoCoDamping"
for exe in gncOuterL2 gncCompare; do
    [ -x "$bin/$exe" ] || { echo "ERROR: $bin/$exe missing (run scripts/build.sh)" >&2; exit 2; }
done
for l in "$lib" "$dumplib"; do
    [ -f "$l" ] || { echo "ERROR: $l missing (run scripts/build.sh)" >&2; exit 2; }
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

printf 'case\tnCells\tref\tgpu\tjacobian\tprecondition\tsolve\tstate\tgpu1_vs_gpu2\n' > "$out_root/SUMMARY.tsv"
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
        foam foamDictionary -entry libs -set "(\"$lib\")" "$work/system/controlDict"
        foam foamDictionary -entry functions -set "{ gncState { type gncHisaDump; \
            libs (\"$dumplib\"); outputDir \"$out/ref/state\"; fields ($fields); } }" \
            "$work/system/controlDict"
        solver="$(foam foamDictionary -entry flowSolver/solver -value "$work/system/fvSolution")"
        echo "solver=$solver"
        foam foamDictionary -entry "flowSolver/$solver/preconditioner" -set gncLUSGSDump \
            "$work/system/fvSolution"
        foam foamDictionary -entry "flowSolver/$solver/gncLUSGSDump" \
            -set "{ outputDir \"$out/ref\"; }" "$work/system/fvSolution"
    } > "$out/log.variant" 2>&1

    for f in 0/p 0/T 0/U 0/nuTilda 0/nut 0/alphat constant/thermophysicalProperties system/fvSchemes \
             system/fvSolution constant/turbulenceProperties; do
        [ -f "$work/$f" ] || continue
        foam foamDictionary -expand -precision 17 "$work/$f" \
            > "$work/gncExpanded/$(basename "$f")" 2>> "$out/log.expand"
    done

    (cd "$work" && foam hisa) > "$out/log.ref" 2>&1
    rc_ref=$?
    for k in jacobian precondition solve; do
        [ -f "$out/ref/1.$k.gnc" ] || rc_ref=9
    done
    [ -f "$out/ref/state/1.gnc" ] || rc_ref=9

    for g in 1 2; do
        CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$bin/gncOuterL2" --case "$work" \
            --expanded "$work/gncExpanded" --out "$out/gpu$g.gnc" \
            --out-state "$out/gpu$g.state.gnc" > "$out/log.gpu$g" 2>&1
        rc=$?
        case "$g" in 1) rc_gpu1=$rc ;; *) rc_gpu2=$rc ;; esac
    done

    declare -A compare=([jacobian]="-" [precondition]="-" [solve]="-" [state]="-")
    cmp_result="-"
    if [ "$rc_ref" -eq 0 ] && [ "$rc_gpu1" -eq 0 ]; then
        for k in jacobian precondition solve state; do
            if [ "$k" = state ]; then
                ref_file="$out/ref/state/1.gnc"; test_file="$out/gpu1.state.gnc"
            else
                ref_file="$out/ref/1.$k.gnc"; test_file="$out/gpu1.gnc"
            fi
            "$bin/gncCompare" "$ref_file" "$test_file" > "$out/compare.$k.txt" 2>&1
            case $? in
                0) compare[$k]="BITWISE" ;;
                1) compare[$k]="DIFF" ;;
                *) compare[$k]="ERROR" ;;
            esac
        done
    fi
    if [ "$rc_gpu1" -eq 0 ] && [ "$rc_gpu2" -eq 0 ]; then
        if cmp -s "$out/gpu1.gnc" "$out/gpu2.gnc" \
           && cmp -s "$out/gpu1.state.gnc" "$out/gpu2.state.gnc"; then
            cmp_result="identical"
        else
            cmp_result="different"
        fi
    fi
    sha256sum "$out"/*.gnc "$out"/ref/*.gnc "$out"/ref/state/*.gnc \
        > "$out/SHA256SUMS" 2>/dev/null
    ncells="$(sed -n 's/^mesh: nCells \([0-9]*\).*/\1/p' "$out/log.gpu1" 2>/dev/null)"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$ncells" "$rc_ref" "$rc_gpu1/$rc_gpu2" "${compare[jacobian]}" \
        "${compare[precondition]}" "${compare[solve]}" "${compare[state]}" "$cmp_result" \
        >> "$out_root/SUMMARY.tsv"
    echo "   ref=$rc_ref gpu=$rc_gpu1/$rc_gpu2 jacobian=${compare[jacobian]}" \
         "precondition=${compare[precondition]} solve=${compare[solve]}" \
         "state=${compare[state]} gpu1_vs_gpu2=$cmp_result"
    for k in jacobian precondition solve state; do
        [ "${compare[$k]}" = "BITWISE" ] || overall=1
    done
    [ "$cmp_result" = "identical" ] || overall=1
    unset compare
done

nvidia-smi > "$out_root/nvidia-smi.after.txt" 2>&1
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out_root/RUN_INFO"
echo "overall_exit=$overall" >> "$out_root/RUN_INFO"
column -t -s "$(printf '\t')" "$out_root/SUMMARY.tsv"
exit "$overall"
