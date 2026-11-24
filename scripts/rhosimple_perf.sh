#!/usr/bin/env bash
# N6 的效能量測：同一個 rhoSimpleFoam 案例、同樣的 SIMPLE 迭代數，比較
#   gnc       gncRhoSimple 參考次序版（GPU，與 rhoSimpleFoam 逐位元；--case，p 為 PCG＋DIC）
#   gncfast   gncRhoSimple -fast（GPU；--case，多色 DILU／DIC、樹狀歸約）
#   gncamg    gncRhoSimple -fast 跑 --gamg-case（p 為 GAMG，以 PCG＋聚合式 AMG 代替）
#   of1       OpenFOAM v2606 rhoSimpleFoam，單核（taskset --core），--gamg-case
#   ofNP      同上，MPI NP 核（metis 分解，g02 的 runtime 沒有 scotch；--bind-to none）
#   spuma     SPUMA（OpenFOAM 的 GPU 移植，D-026）上的 rhoSimpleFoam（GPU），--gamg-case
#   of1pcg、ofNPpcg、spumapcg  同上，但跑 --case（p 為 PCG＋DIC，與 gnc 相同的演算法）
# 每個執行在 $RUN_ROOT/LABEL/NAME/VARIANT 建工作副本（polyMesh 以符號連結）：endTime 設為
# 迭代數（--iters，或 --iters-for VARIANT=N），writeInterval 大於迭代數（不寫場），functions
# 清空；residualControl 照案例設定（收斂時提早結束，迭代數記在 SUMMARY）。每迭代時間：
#   gnc*            求解器印出的迴圈時間（不含建構）／迭代數（牆鐘）
#   OpenFOAM、SPUMA  per_iter_s = (ExecutionTime_N - ExecutionTime_1)/(N - 1)（程序的 CPU
#                   時間，MPI 時為 master）；per_iter_clock_s = (ClockTime_N - ClockTime_1)/(N - 1)
#                   （牆鐘，整數秒）。SPUMA 的 GPU 等待不計入 CPU 時間，以牆鐘為準
# 另記錄 /usr/bin/time 的總牆鐘時間與 p 的平均迭代次數。GPU 為 $GPU_DEVICE。
# SPUMA 的記憶體池：fixedSizeMemoryPool，--spuma-pool GB（預設 8；大網格時要放得下整個求解的資料）。
#
# 用法：rhosimple_perf.sh --host-env FILE --label LABEL --case NAME=DIR [--case ...]
#                         [--gamg-case NAME=DIR ...] --iters N [--iters-for VARIANT=N ...]
#                         [--variants gnc,gncfast,gncamg,of1,spuma] [--core C] [--spuma-pool GB]

set -u

host_env=""; label=""; iters=""; variants="gnc,gncfast,gncamg,of1"; core=157; spuma_pool=8
cases=()
declare -A gamg_for=() iters_for=()
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        --case) cases+=("$2"); shift 2 ;;
        --gamg-case) gamg_for["${2%%=*}"]="${2#*=}"; shift 2 ;;
        --iters) iters="$2"; shift 2 ;;
        --iters-for) iters_for["${2%%=*}"]="${2#*=}"; shift 2 ;;
        --variants) variants="$2"; shift 2 ;;
        --core) core="$2"; shift 2 ;;
        --spuma-pool) spuma_pool="$2"; shift 2 ;;
        -h|--help) sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -r "$host_env" ] || { echo "ERROR: --host-env FILE required" >&2; exit 2; }
# shellcheck disable=SC1090
. "$host_env"
case "$label" in ""|*/*|.*) echo "ERROR: --label NAME required" >&2; exit 2 ;; esac
[ "${#cases[@]}" -gt 0 ] || { echo "ERROR: at least one --case NAME=DIR" >&2; exit 2; }
[ -n "$iters" ] || { echo "ERROR: --iters N required" >&2; exit 2; }

here="$(cd "$(dirname "$0")" && pwd)"
bin="$BUILD_ROOT/bin"
[ -x "$bin/gncRhoSimple" ] || { echo "ERROR: $bin/gncRhoSimple missing" >&2; exit 2; }
SP=/ssd/sclin/gpu-platform
NV=$SP/nvhpc/Linux_x86_64/26.9

out_root="$RUN_ROOT/$label"
[ -e "$out_root" ] && { echo "ERROR: $out_root exists" >&2; exit 2; }
mkdir -p "$out_root"
{
    echo "label=$label"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname) profile=$HOST_PROFILE gpu_device=$GPU_DEVICE"
    echo "iters=$iters variants=$variants core=$core spuma_pool_gb=$spuma_pool"
    for k in "${!iters_for[@]}"; do echo "iters_for $k=${iters_for[$k]}"; done
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

spuma() {
    (
        nvccdir=$(dirname "$(ls $NV/cuda/*/bin/nvcc 2>/dev/null | head -1)")
        export PATH=$NV/compilers/bin:$nvccdir:$PATH
        export LD_LIBRARY_PATH=$NV/compilers/lib:$(dirname "$nvccdir")/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
        export have_cuda=true NVARCH=89 FOAM_SIGFPE=false
        set +u
        # shellcheck disable=SC1090
        . $SP/src/spuma-0.3-v2512/etc/bashrc
        set -u
        "$@"
    )
}

# prep SRC DST N：工作副本
prep() {
    local s="$1" d="$2" n="$3"
    mkdir -p "$d/constant"
    cp -r "$s/system" "$s/0" "$d/"
    for f in "$s"/constant/*; do
        b="$(basename "$f")"
        if [ "$b" = polyMesh ]; then
            ln -s "$(realpath "$f")" "$d/constant/polyMesh"
        else
            cp -r "$f" "$d/constant/$b"
        fi
    done
    foam foamDictionary -entry endTime -set "$n" "$d/system/controlDict" > /dev/null 2>&1
    foam foamDictionary -entry writeControl -set timeStep "$d/system/controlDict" > /dev/null 2>&1
    foam foamDictionary -entry writeInterval -set "$((n + 1000000))" "$d/system/controlDict" > /dev/null 2>&1
    foam foamDictionary -entry functions -set "{}" "$d/system/controlDict" > /dev/null 2>&1
}

# OpenFOAM 的迭代數、每迭代 CPU 時間與牆鐘時間（log 的 ExecutionTime、ClockTime）
of_per_iter() {
    awk '/^ExecutionTime/ {n++; t[n] = $3; c[n] = $7}
         END {if (n > 1) printf "%d %.6f %.3f", n, (t[n] - t[1])/(n - 1), (c[n] - c[1])/(n - 1);
              else printf "%d nan nan", n}' "$1"
}

p_mean() {
    grep -E 'Solving for p,' "$1" | awk '{s += $NF} END {if (NR) printf "%.1f", s/NR; else print "nan"}'
}

printf 'case\tvariant\tnCells\titerations\tper_iter_s\tper_iter_clock_s\twall_s\tp_iters_mean\n' > "$out_root/SUMMARY.tsv"
IFS=',' read -r -a vlist <<< "$variants"

for spec in "${cases[@]}"; do
    name="${spec%%=*}"; src="${spec#*=}"
    gsrc="${gamg_for[$name]:-$src}"
    for v in "${vlist[@]}"; do
        d="$out_root/$name/$v"
        n="${iters_for[$v]:-$iters}"
        echo "== $name $v ($n iterations) start_epoch=$(date +%s.%N)"
        osrc="$gsrc"; vv="$v"
        case "$v" in
            *pcg) osrc="$src"; vv="${v%pcg}" ;;
        esac
        clock="-"
        case "$vv" in
            gnc|gncfast|gncamg)
                csrc="$src"; fast=""
                [ "$vv" = gncfast ] && fast="-fast"
                [ "$vv" = gncamg ] && { fast="-fast"; csrc="$gsrc"; }
                prep "$csrc" "$d" "$n"
                bash "$here/gnc_prepare_case.sh" -case "$d" --host-env "$host_env" > "$d/log.expand" 2>&1
                ( cd "$d" && CUDA_VISIBLE_DEVICES="$GPU_DEVICE" /usr/bin/time -v "$bin/gncRhoSimple" $fast -profile \
                    > log.solver 2> log.time )
                rc=$?
                it="$(awk '/^iterations / {gsub(";", "", $2); print $2}' "$d/log.solver")"
                per="$(sed -n 's/.*(\([0-9.e+-]*\) s\/iteration).*/\1/p' "$d/log.solver")"
                ncells="$(grep -m1 '^mesh:' "$d/log.solver" | awk '{print $2}')"
                ;;
            of1)
                prep "$osrc" "$d" "$n"
                ( cd "$d" && foam /usr/bin/time -v taskset -c "$core" rhoSimpleFoam > log.solver 2> log.time )
                rc=$?
                read -r it per clock <<< "$(of_per_iter "$d/log.solver")"
                ncells="-"
                ;;
            of[0-9]*)
                npv="${vv#of}"
                prep "$osrc" "$d" "$n"
                foam foamDictionary -entry numberOfSubdomains -set "$npv" "$d/system/decomposeParDict" > /dev/null 2>&1
                foam foamDictionary -entry method -set metis "$d/system/decomposeParDict" > /dev/null 2>&1
                ( cd "$d" && foam decomposePar -force > log.decompose 2>&1 && \
                  foam /usr/bin/time -v mpirun --bind-to none -np "$npv" rhoSimpleFoam -parallel > log.solver 2> log.time )
                rc=$?
                read -r it per clock <<< "$(of_per_iter "$d/log.solver")"
                ncells="-"
                ( cd "$d" && foam reconstructPar -latestTime > log.reconstruct 2>&1 )
                rm -rf "$d"/processor*
                ;;
            spuma)
                prep "$osrc" "$d" "$n"
                ( cd "$d" && CUDA_VISIBLE_DEVICES="$GPU_DEVICE" spuma /usr/bin/time -v taskset -c "$core" \
                    rhoSimpleFoam -pool fixedSizeMemoryPool -poolSize "$spuma_pool" > log.solver 2> log.time )
                rc=$?
                read -r it per clock <<< "$(of_per_iter "$d/log.solver")"
                ncells="-"
                ;;
            *) echo "unknown variant $v" >&2; continue ;;
        esac
        wall="$(sed -n 's/.*Elapsed (wall clock) time (h:mm:ss or m:ss): //p' "$d/log.time" | \
            awk -F: '{if (NF == 3) print $1*3600 + $2*60 + $3; else print $1*60 + $2}')"
        pm="$(p_mean "$d/log.solver")"
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$v" "${ncells:-?}" "${it:-?}" "${per:-?}" \
            "${clock:-?}" "${wall:-?}" "$pm" >> "$out_root/SUMMARY.tsv"
        echo "   rc=$rc iterations=${it:-?} per_iter=${per:-?} s clock=${clock:-?} s wall=${wall:-?} s p_iters=$pm end_epoch=$(date +%s.%N)"
    done
done
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out_root/RUN_INFO"
column -t -s $'\t' "$out_root/SUMMARY.tsv"
