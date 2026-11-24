#!/usr/bin/env bash
# N3d 的 L3：穩態的整個 run（多個外迭代、SER、收斂判斷），GPU-Native-CFD 對
# HiSA 本身逐位元比對（N-015）。
#
# 每個案例在 $RUN_ROOT/LABEL/NAME/case 建工作副本（system、constant、0 複製，
# constant/polyMesh 以符號連結；原案例不寫入），改成穩態變體：
#   fvSchemes   ddtSchemes/default = bounded dualTime rPseudoDeltaT steadyState
#   controlDict startFrom startTime、startTime 0、deltaT 1、endTime N（--end，
#               每一步一個外迭代），不寫場；functions 換成只有 gncState
#               （gncHisaDump：timeIndices 與 atEnd，outputDir 為 ref/state；
#               案例原有的 function object 只做後處理，不影響場）
#   turbulenceProperties  --laminar NAME 的案例改為 simulationType laminar
#   fvSchemes   --meshwave NAME 的案例把 wallDist/method 改為 meshWave（N-022：
#               Poisson 牆距離的案例以 meshWave 變體驗證）
# fvSolution 預設不改：pseudoTol、pseudoTolRel、nPseudoCorr 等照案例原本的設定，
# 收斂時 HiSA 在 endTime 之前 writeAndEnd。--pseudo-tol NAME=V、--pseudo-tol-rel
# NAME=V 把該案例的 pseudoTime/pseudoTol、pseudoTolRel 設為 V（五個分量相同），
# 用來測 absCheck、relCheck 與 writeAndEnd。
# 然後：
#   1. foamDictionary -expand -precision 17 → case/gncExpanded/（p T U
#      thermophysicalProperties fvSchemes fvSolution controlDict，有
#      turbulenceProperties 時一併展開）
#   2. ref/state/<k>.gnc、ref/state/end.gnc   hisa 執行時在時間步 k 與 end() 傾印
#   3. gpu1/、gpu2/                            gncRunL3（兩個程序，逐檔 cmp）
#   4. compare.<k>.txt、compare.end.txt        gncCompare ref/state/<k>.gnc gpu1/<k>.gnc
#      ref 與 gpu 的結束時間步（end.gnc 的 timeIndex）也要相同
# 力係數（--forces NAME=FUNC）：該案例保留 controlDict 的 function FUNC（forces 或
# forceCoeffs）。controlDict 先在原案例目錄內以 foamDictionary -expand 展開（#includeFunc
# 才會讀到案例的 system/FUNC），functions 改為 FUNC（executeControl timeStep、
# executeInterval 1、log off）與排在它之後的 gncState（forceObjects (FUNC)，fields 另加
# force moment，useNamePrefix 預設 false）；gncRunL3 讀同一個展開的 controlDict 計算，紀錄名稱
# FUNC.<result>、force.i/b、moment.i/b（只有一個 FUNC）。
# 診斷（--debug-step K）：參考端的預條件器改為 gncLUSGSDump（libs 載入
# libgncHisaJacobianDump.so，每一步都寫 ref/<t>.{jacobian,precondition,solve}.gnc），
# gncRunL3 --debug-step K 寫 gpu1/debug.K.gnc；compare.debug.<k>.txt 為三份的比對。
#
# 用法：l3_steady_verify.sh --host-env FILE --label LABEL --case NAME=DIR [--case ...]
#                           --end-default N [--end NAME=N ...] [--laminar NAME ...]
#                           [--time-indices 1,2,10] [--time-indices-for NAME=1,5]
#                           [--debug-step K] [--pseudo-tol NAME=V] [--pseudo-tol-rel NAME=V]
#                           [--forces NAME=FUNC] [--meshwave NAME ...]

set -u

host_env=""; label=""; end_default=""; indices_default="1,2,10"; debug_step=""
cases=(); laminar=" "; meshwave=" "
declare -A end_for=() indices_for=() tol_for=() tolrel_for=() forces_for=()
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --case) cases+=("$2"); shift 2 ;;
        --laminar) laminar="$laminar$2 "; shift 2 ;;
        --meshwave) meshwave="$meshwave$2 "; shift 2 ;;
        --end-default) end_default="$2"; shift 2 ;;
        --end) end_for["${2%%=*}"]="${2#*=}"; shift 2 ;;
        --time-indices) indices_default="$2"; shift 2 ;;
        --time-indices-for) indices_for["${2%%=*}"]="${2#*=}"; shift 2 ;;
        --debug-step) debug_step="$2"; shift 2 ;;
        --pseudo-tol) tol_for["${2%%=*}"]="${2#*=}"; shift 2 ;;
        --pseudo-tol-rel) tolrel_for["${2%%=*}"]="${2#*=}"; shift 2 ;;
        --forces) forces_for["${2%%=*}"]="${2#*=}"; shift 2 ;;
        -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -r "$host_env" ] || { echo "ERROR: --host-env FILE required" >&2; exit 2; }
# shellcheck disable=SC1090
. "$host_env"
case "$label" in ""|*/*|.*) echo "ERROR: --label NAME required" >&2; exit 2 ;; esac
[ "${#cases[@]}" -gt 0 ] || { echo "ERROR: at least one --case NAME=DIR" >&2; exit 2; }
[ -n "$end_default" ] || { echo "ERROR: --end-default N required" >&2; exit 2; }

bin="$BUILD_ROOT/bin"
dumplib="$BUILD_ROOT/lib/libgncHisaDump.so"
jaclib="$BUILD_ROOT/lib/libgncHisaJacobianDump.so"
fields="p T U e thermo:psi thermo:mu thermo:alpha rho rhoU rhoE phi pseudoCoDamping pseudoCoNum nuTilda k omega nut alphat"
for exe in gncRunL3 gncCompare; do
    [ -x "$bin/$exe" ] || { echo "ERROR: $bin/$exe missing (run scripts/build.sh)" >&2; exit 2; }
done
[ -f "$dumplib" ] || { echo "ERROR: $dumplib missing (run scripts/build.sh)" >&2; exit 2; }

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

printf 'case\tnCells\tendTime\tref\tgpu\tend_ref\tend_gpu\tcompared\tbitwise\tgpu1_vs_gpu2\n' \
    > "$out_root/SUMMARY.tsv"
overall=0

for spec in "${cases[@]}"; do
    name="${spec%%=*}"; src="${spec#*=}"
    out="$out_root/$name"
    work="$out/case"
    endt="${end_for[$name]:-$end_default}"
    idx="${indices_for[$name]:-$indices_default}"
    mkdir -p "$work/constant" "$work/gncExpanded"
    echo "== $name ($src) endTime=$endt timeIndices=$idx"

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
    func="${forces_for[$name]:-}"
    if [ -n "$func" ]; then
        (cd "$src" && foam foamDictionary -expand -precision 17 system/controlDict) \
            > "$work/system/controlDict" 2> "$out/log.forces"
    fi

    # 穩態變體
    is_laminar=0
    case "$laminar" in *" $name "*) is_laminar=1 ;; esac
    {
        foam foamDictionary -entry ddtSchemes/default \
            -set "bounded dualTime rPseudoDeltaT steadyState" "$work/system/fvSchemes"
        for kv in startFrom=startTime startTime=0 "endTime=$endt" deltaT=1 \
                  writeControl=timeStep writeInterval=1000000; do
            foam foamDictionary -entry "${kv%%=*}" -set "${kv#*=}" "$work/system/controlDict"
        done
        case "$meshwave" in
            *" $name "*)
                foam foamDictionary -entry wallDist/method -set meshWave "$work/system/fvSchemes"
                echo "meshwave=1" ;;
        esac
        if [ "$is_laminar" -eq 1 ]; then
            foam foamDictionary -entry simulationType -set laminar \
                "$work/constant/turbulenceProperties"
            echo "laminar=1"
        fi
        fdict=""; fobj=""; ffields=""
        if [ -n "$func" ]; then
            ffields=" force moment"
            fdict="$(foam foamDictionary -precision 17 -entry "functions/$func" "$work/system/controlDict")"
            [ -n "$fdict" ] || echo "ERROR: function $func not found"
            fobj="forceObjects ($func);"
        fi
        foam foamDictionary -entry functions -set "{ $fdict gncState { type gncHisaDump; \
            libs (\"$dumplib\"); outputDir \"$out/ref/state\"; fields ($fields$ffields); \
            timeIndices (${idx//,/ }); atEnd true; $fobj } }" \
            "$work/system/controlDict"
        if [ -n "$func" ]; then
            for kv in executeControl=timeStep executeInterval=1 log=off; do
                foam foamDictionary -entry "functions/$func/${kv%%=*}" -set "${kv#*=}" \
                    "$work/system/controlDict"
            done
            echo "forces=$func"
        fi
        if [ -n "${tol_for[$name]:-}" ]; then
            v="${tol_for[$name]}"
            foam foamDictionary -entry pseudoTime/pseudoTol -set "$v ($v $v $v) $v" \
                "$work/system/fvSolution"
            echo "pseudoTol=$v"
        fi
        if [ -n "${tolrel_for[$name]:-}" ]; then
            v="${tolrel_for[$name]}"
            foam foamDictionary -entry pseudoTime/pseudoTolRel -set "$v ($v $v $v) $v" \
                "$work/system/fvSolution"
            echo "pseudoTolRel=$v"
        fi
        if [ -n "$debug_step" ]; then
            foam foamDictionary -entry libs -set "(\"$jaclib\")" "$work/system/controlDict"
            solver="$(foam foamDictionary -entry flowSolver/solver -value "$work/system/fvSolution")"
            foam foamDictionary -entry "flowSolver/$solver/preconditioner" -set gncLUSGSDump \
                "$work/system/fvSolution"
            foam foamDictionary -entry "flowSolver/$solver/gncLUSGSDump" \
                -set "{ outputDir \"$out/ref\"; }" "$work/system/fvSolution"
        fi
    } > "$out/log.variant" 2>&1

    for f in 0/p 0/T 0/U 0/nuTilda 0/k 0/omega 0/nut 0/alphat constant/thermophysicalProperties system/fvSchemes \
             system/fvSolution system/controlDict constant/turbulenceProperties; do
        [ -f "$work/$f" ] || continue
        foam foamDictionary -expand -precision 17 "$work/$f" \
            > "$work/gncExpanded/$(basename "$f")" 2>> "$out/log.expand"
    done

    (cd "$work" && foam hisa) > "$out/log.ref" 2>&1
    rc_ref=$?
    [ -f "$out/ref/state/end.gnc" ] || rc_ref=9

    for g in 1 2; do
        CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$bin/gncRunL3" --case "$work" \
            --expanded "$work/gncExpanded" --out-dir "$out/gpu$g" \
            --time-indices "$idx" ${debug_step:+--debug-step "$debug_step"} \
            > "$out/log.gpu$g" 2>&1
        rc=$?
        case "$g" in 1) rc_gpu1=$rc ;; *) rc_gpu2=$rc ;; esac
    done

    compared=0; bitwise=0; end_ref="-"; end_gpu="-"; cmp_result="-"
    if [ "$rc_ref" -eq 0 ] && [ "$rc_gpu1" -eq 0 ]; then
        for f in "$out"/ref/state/*.gnc; do
            k="$(basename "$f" .gnc)"
            [ "$k" = 0 ] && continue
            compared=$((compared + 1))
            "$bin/gncCompare" "$f" "$out/gpu1/$k.gnc" > "$out/compare.$k.txt" 2>&1
            [ $? -eq 0 ] && bitwise=$((bitwise + 1))
        done
        if [ -n "$debug_step" ]; then
            for k in jacobian precondition solve; do
                "$bin/gncCompare" "$out/ref/$debug_step.$k.gnc" "$out/gpu1/debug.$debug_step.gnc" \
                    > "$out/compare.debug.$k.txt" 2>&1
                echo "   debug step $debug_step $k: $(tail -1 "$out/compare.debug.$k.txt")"
            done
        fi
        end_ref="$(sed -n 's/^timeIndex .*ref="\([0-9]*\)".*/\1/p' "$out/compare.end.txt")"
        end_gpu="$(sed -n 's/^timeIndex .*test="\([0-9]*\)".*/\1/p' "$out/compare.end.txt")"
    fi
    if [ "$rc_gpu1" -eq 0 ] && [ "$rc_gpu2" -eq 0 ]; then
        if diff -rq "$out/gpu1" "$out/gpu2" > /dev/null; then
            cmp_result="identical"
        else
            cmp_result="different"
        fi
    fi
    sha256sum "$out"/ref/state/*.gnc "$out"/gpu1/*.gnc "$out"/gpu2/*.gnc \
        > "$out/SHA256SUMS" 2>/dev/null
    ncells="$(sed -n 's/^mesh: nCells \([0-9]*\).*/\1/p' "$out/log.gpu1" 2>/dev/null)"

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$name" "$ncells" "$endt" "$rc_ref" "$rc_gpu1/$rc_gpu2" "$end_ref" "$end_gpu" \
        "$compared" "$bitwise" "$cmp_result" >> "$out_root/SUMMARY.tsv"
    echo "   ref=$rc_ref gpu=$rc_gpu1/$rc_gpu2 end=$end_ref/$end_gpu" \
         "bitwise=$bitwise/$compared gpu1_vs_gpu2=$cmp_result"
    if [ "$compared" -eq 0 ] || [ "$bitwise" -ne "$compared" ] \
       || [ -z "$end_ref" ] || [ "$end_ref" != "$end_gpu" ] \
       || [ "$cmp_result" != "identical" ]; then
        overall=1
    fi
done

nvidia-smi > "$out_root/nvidia-smi.after.txt" 2>&1
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out_root/RUN_INFO"
echo "overall_exit=$overall" >> "$out_root/RUN_INFO"
column -t -s "$(printf '\t')" "$out_root/SUMMARY.tsv"
exit "$overall"
