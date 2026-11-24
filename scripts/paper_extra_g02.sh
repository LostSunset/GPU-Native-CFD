#!/usr/bin/env bash
# 論文的補充量測（對照 SPUMA 的 CPC 論文補齊：硬體與軟體版本、網格大小、kernel／記憶體傳輸／CUDA API 的剖析、
# 各階段時間、能耗、流場與網格的圖）。程式是投稿版（原始碼雜湊 1bf4bc6d…），建在 configs/hosts/g02-paper.env
# 的獨立目錄。紀錄：
#   P1-host      lscpu、nvidia-smi -q、free、各工具版本
#   P1-cases     各案例的網格大小（owner 檔頭、boundary）與 system/ 的 fvSolution、fvSchemes、controlDict
#   P1-nsys      nsys：gncRhoSimple -fast（16,000 格 50／100 迭代、1,024,000 格 10／20 迭代）、gncHisa -fast
#                （ONERA M6 10／20 步）；兩種迭代數的統計相減得到每迭代的 kernel、memcpy、CUDA API
#   P1-nsys-large 同上，gncRhoSimple -fast 在 4,096,000 格（10／20 迭代）與 16,384,000 格（5／10 迭代）
#   P1-energy    GPU3 的功率（nvidia-smi power.draw.instant，每 100 ms）在各執行期間的取樣與閒置功率
#   P1-energylong 同上，每個變體兩種迭代數（能量差 / 迭代數差，扣掉啟動）
#   P1-ofprof    rhoSimpleFoam 的內建 profiling（各方程線性求解的時間）：16,000、1,024,000 格，1 與 32 核
#   P1-fields    aerofoilNACA0012 收斂解（rhoSimpleFoam 單核、gncRhoSimple 參考版與 -fast）的 p、U、T 與格心座標
#   P1-geometry  aerofoilNACA0012 的 polyMesh（ASCII）、ONERA M6 機翼與 basicFinner 彈體的邊界面（foamToVTK）
#   P1-memory    各案例（參考版與 -fast，3 迭代）的 GPU 記憶體峰值（nvidia-smi 每 100 ms 的 used_memory，含 CUDA context）
# 開始前等 --wait-pid 結束，並等 GPU3 連續 3 分鐘沒有其他程式（--no-gpu-wait 時不等 GPU）。--phases 選擇要跑的部分
# （預設全部）；只用 CPU 而且很短的部分（host、cases、fields、geometry）可以在其他工作執行時先跑。
#
# 在 g02 上執行：bash scripts/paper_extra_g02.sh [--wait-pid PID] [--phases host,cases,...] [--no-gpu-wait] > ... 2>&1
set -u
wait_pid=""; phases="host,cases,nsys,energy,ofprof,fields,geometry,memory"; gpu_wait=1
while [ $# -gt 0 ]; do
    case "$1" in
        --wait-pid) wait_pid="$2"; shift 2 ;;
        --phases) phases="$2"; shift 2 ;;
        --no-gpu-wait) gpu_wait=0; shift ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
want() { case ",$phases," in *",$1,"*) return 0 ;; *) return 1 ;; esac; }
cd /ssd/sclin/gpu-native-cfd/paper/GPU-Native-CFD
E=configs/hosts/g02-paper.env
. "$E"
C=$CASE_ROOT
G=/ssd/sclin/gpu-native-cfd/cases
R=$RUN_ROOT
BIN=$BUILD_ROOT/bin
PAPER_HASH=1bf4bc6daf538373ee5ce3d15a39c75667f81e22cec9d754db00c7edde099372
foam() { ( set +eu; . "$OPENFOAM_BASHRC" > /dev/null 2>&1; set -eu; "$@" ); }
stamp() { echo "=== $(date -u +%FT%TZ) $*"; }

if [ -n "$wait_pid" ]; then
    stamp "waiting for pid $wait_pid"
    while kill -0 "$wait_pid" 2>/dev/null; do sleep 60; done
fi
free=0
[ "$gpu_wait" = 1 ] || free=3
while [ "$free" -lt 3 ]; do
    if [ -z "$(nvidia-smi -i "$GPU_DEVICE" --query-compute-apps=pid --format=csv,noheader)" ]; then
        free=$((free + 1))
    else
        free=0
    fi
    sleep 60
done
stamp "GPU$GPU_DEVICE free"

stamp "build"
bash scripts/build.sh --host-env "$E" > "$BUILD_ROOT.log" 2>&1 || { echo "build failed"; exit 1; }
grep -q "source_sha256=$PAPER_HASH" "$BUILD_ROOT/BUILD_INFO" || { echo "source hash differs from the paper"; exit 1; }

# 案例的工作副本：system、0 複製，polyMesh 連結；endTime、writeInterval
prep() {
    # 迴圈變數必須是 local：bash 的動態作用域下，呼叫者（energy_run、mem_run）的 local f 會被覆蓋
    local f b
    rm -rf "$2"; mkdir -p "$2/constant"
    cp -r "$1/system" "$1/0" "$2/"
    for f in "$1"/constant/*; do
        b="$(basename "$f")"
        if [ "$b" = polyMesh ]; then ln -s "$(realpath "$f")" "$2/constant/polyMesh"; else cp -r "$f" "$2/constant/$b"; fi
    done
    sed -i -e "s/^endTime .*/endTime         $3;/" -e "s/^writeInterval .*/writeInterval   $4;/" \
        -e "s/^startFrom .*/startFrom       startTime;/" -e "s/^stopAt .*/stopAt          endTime;/" "$2/system/controlDict"
}
newrec() { mkdir -p "$R/$1"; cp "$BUILD_ROOT/BUILD_INFO" "$R/$1/"; }

# ---------------------------------------------------------------------------
if want host; then
stamp "P1-host"
H=$R/P1-host; newrec P1-host
lscpu > "$H/lscpu.txt"
nvidia-smi -q -i "$GPU_DEVICE" > "$H/nvidia-smi-q.txt"
free -g > "$H/free.txt"
{
    echo "uname=$(uname -srm)"
    echo "glibc=$(ldd --version | head -1)"
    echo "nvcc=$("$NVCC" --version | tail -1)"
    echo "gxx=$(g++ --version | head -1)"
    echo "cmake=$(cmake --version | head -1)"
    echo "nsys=$(nsys --version)"
    echo "driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader -i "$GPU_DEVICE")"
    echo "mpirun=$(foam mpirun --version 2>&1 | head -1)"
    echo "openfoam=$(foam bash -c 'echo $WM_PROJECT_VERSION $WM_OPTIONS')"
    echo "uptime=$(uptime)"
} > "$H/versions.txt"
fi
# ---------------------------------------------------------------------------
if want cases; then
stamp "P1-cases"
K=$R/P1-cases; newrec P1-cases
for spec in shockTube=$C/shockTube-prepared forwardStep=$C/forwardStep-prepared/simulation \
            flatPlate=$C/flatPlate-g2000/simulation rae2822=$C/rae2822-prepared/simulation \
            nacaAirfoil=$C/nacaAirfoil-prepared/simulation oneraM6=$C/oneraM6-prepared/simulation \
            basicFinner=$C/basicFinner-seq5/simulation \
            aerofoilNACA0012=$G/aerofoilNACA0012/simulation aerofoilNACA0012-pcg=$G/aerofoilNACA0012-pcg/simulation \
            aerofoilNACA0012-r2=$G/aerofoilNACA0012-r2/simulation aerofoilNACA0012-r4=$G/aerofoilNACA0012-r4/simulation \
            aerofoilNACA0012-r8=$G/aerofoilNACA0012-r8/simulation; do
    n="${spec%%=*}"; d="${spec#*=}"
    mkdir -p "$K/$n"
    cp "$d/system/fvSolution" "$d/system/fvSchemes" "$d/system/controlDict" "$K/$n/"
    cp "$d/constant/polyMesh/boundary" "$K/$n/"
    head -c 4000 "$d/constant/polyMesh/owner" | grep -aoE "nPoints: *[0-9]+ +nCells: *[0-9]+ +nFaces: *[0-9]+ +nInternalFaces: *[0-9]+" \
        > "$K/$n/owner-note.txt" || true
done
fi
# ---------------------------------------------------------------------------
# nsys 的一次執行（N 為紀錄目錄；nsys 與 nsyslarge 共用）
nsys_run() {   # name solver case steps
    local d="$N/$1"
    prep "$3" "$d" "$4" 1000000
    bash scripts/gnc_prepare_case.sh -case "$d" --host-env "$E" > "$d/log.prepare" 2>&1
    ( cd "$d" && CUDA_VISIBLE_DEVICES="$GPU_DEVICE" nsys profile -o "$d/prof" --force-overwrite=true --trace=cuda \
        --cuda-graph-trace=node --sample=none --cpuctxsw=none "$BIN/$2" -case "$d" -fast > log.solver 2> log.nsys )
    local rc=$?
    ( cd "$d" && nsys stats -q --force-export=true --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum,cuda_api_sum \
        --format csv --output stats prof.nsys-rep > log.stats 2>&1 )
    rm -f "$d/prof.nsys-rep" "$d/prof.sqlite"
    echo "$1 rc=$rc"
}
if want nsys; then
stamp "P1-nsys"
N=$R/P1-nsys; newrec P1-nsys
nsys_run rho-r1-n050 gncRhoSimple "$G/aerofoilNACA0012/simulation" 50
nsys_run rho-r1-n100 gncRhoSimple "$G/aerofoilNACA0012/simulation" 100
nsys_run rho-r8-n010 gncRhoSimple "$G/aerofoilNACA0012-r8/simulation" 10
nsys_run rho-r8-n020 gncRhoSimple "$G/aerofoilNACA0012-r8/simulation" 20
nsys_run m6-n010 gncHisa "$C/oneraM6-prepared/simulation" 10
nsys_run m6-n020 gncHisa "$C/oneraM6-prepared/simulation" 20
fi
# N8 的 modifiedFinner 計時是在開發版目錄建置的（同一個原始碼雜湊）；以本目錄乾淨的論文版建置重量，確認沒有
# 「同步後目的檔未重建」的問題（N-028 的改動位元中性，數值比對無法分辨，只能比時間）
if want mfcheck; then
stamp "P1-mfcheck"
FAST="GNC_LUSGS=color GNC_SGS=color GNC_REDUCE=tree GNC_JAC=auto"
bash scripts/variant_compare.sh --host-env "$E" --label P1-mfcheck --end 5 --variant ref= --variant "fast=$FAST" \
    --case modifiedFinnerSST="$C/modifiedFinner-seq8/simulation"
fi
# 大網格（N6e 的 r16、r32）：每迭代時間的成長比格數快，看是哪些 kernel
if want nsyslarge; then
stamp "P1-nsys-large"
N=$R/P1-nsys-large; newrec P1-nsys-large
nsys_run rho-r16-n010 gncRhoSimple "$G/aerofoilNACA0012-r16/simulation" 10
nsys_run rho-r16-n020 gncRhoSimple "$G/aerofoilNACA0012-r16/simulation" 20
nsys_run rho-r32-n005 gncRhoSimple "$G/aerofoilNACA0012-r32/simulation" 5
nsys_run rho-r32-n010 gncRhoSimple "$G/aerofoilNACA0012-r32/simulation" 10
fi
# ---------------------------------------------------------------------------
# 一次執行的功率取樣（W 為紀錄目錄；energy 與 energylong 共用）
energy_run() {   # name solver case steps fast(0|1)
    local d="$W/$1" f=""
    [ "$5" = 1 ] && f="-fast"
    prep "$3" "$d" "$4" 1000000
    bash scripts/gnc_prepare_case.sh -case "$d" --host-env "$E" > "$d/log.prepare" 2>&1
    nvidia-smi -i "$GPU_DEVICE" --query-gpu=timestamp,power.draw.instant --format=csv,noheader,nounits -lms 100 \
        > "$d/power.csv" & local sp=$!
    sleep 2
    local t0; t0=$(date +%s.%N)
    ( cd "$d" && CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$BIN/$2" -case "$d" $f -profile > log.solver 2>&1 )
    local rc=$? t1; t1=$(date +%s.%N)
    sleep 2; kill "$sp"
    echo "start_epoch=$t0 end_epoch=$t1 rc=$rc" > "$d/RUN_TIMES"
    echo "$1 rc=$rc"
}
if want energy; then
stamp "P1-energy"
W=$R/P1-energy; newrec P1-energy
nvidia-smi -i "$GPU_DEVICE" --query-gpu=timestamp,power.draw.instant --format=csv,noheader,nounits -lms 100 \
    > "$W/idle.power.csv" & sp=$!
sleep 30; kill "$sp"
energy_run rho-r8-ref gncRhoSimple "$G/aerofoilNACA0012-r8-pcg/simulation" 10 0
energy_run rho-r8-fast gncRhoSimple "$G/aerofoilNACA0012-r8/simulation" 30 1
energy_run rho-r1-ref-conv gncRhoSimple "$G/aerofoilNACA0012-pcg/simulation" 1000 0
energy_run rho-r1-fast-conv gncRhoSimple "$G/aerofoilNACA0012/simulation" 1000 1
energy_run m6-ref gncHisa "$C/oneraM6-prepared/simulation" 10 0
energy_run m6-fast gncHisa "$C/oneraM6-prepared/simulation" 50 1
fi
# ---------------------------------------------------------------------------
# 長的能量量測：每個變體兩種迭代數，能量差除以迭代數差（扣掉啟動；多出的迴圈至少約 20 s，功率感測值每數百
# 毫秒才更新）。r1 在 677 迭代收斂，迴圈太短，用 P1-energy 的到收斂能量
if want energylong; then
stamp "P1-energylong"
W=$R/P1-energylong; newrec P1-energylong
nvidia-smi -i "$GPU_DEVICE" --query-gpu=timestamp,power.draw.instant --format=csv,noheader,nounits -lms 100 \
    > "$W/idle.power.csv" & sp=$!
sleep 60; kill "$sp"
for spec in "rho-r8-fast gncRhoSimple $G/aerofoilNACA0012-r8/simulation 30 330 1" \
            "rho-r8-ref gncRhoSimple $G/aerofoilNACA0012-r8-pcg/simulation 5 25 0" \
            "m6-fast gncHisa $C/oneraM6-prepared/simulation 50 1050 1" \
            "m6-ref gncHisa $C/oneraM6-prepared/simulation 10 110 0" \
            "finnerSST-fast gncHisa $C/basicFinner-seq5/simulation 5 65 1"; do
    read -r name solver src n1 n2 fast <<< "$spec"
    energy_run "$name-n$n1" "$solver" "$src" "$n1" "$fast"
    energy_run "$name-n$n2" "$solver" "$src" "$n2" "$fast"
done
fi
# ---------------------------------------------------------------------------
if want ofprof; then
stamp "P1-ofprof"
O=$R/P1-ofprof; newrec P1-ofprof
of_run() {   # name case iters np
    local d="$O/$1"
    prep "$2" "$d" "$3" "$3"
    printf '\nprofiling\n{\n    active true;\n    cpuInfo false;\n    memInfo false;\n    sysInfo false;\n}\n' >> "$d/system/controlDict"
    if [ "$4" = 1 ]; then
        ( cd "$d" && foam taskset -c 157 rhoSimpleFoam > log.solver 2>&1 )
        rc=$?
        cp "$d/$3/uniform/profiling" "$d/profiling.txt" 2>/dev/null
    else
        foam foamDictionary -entry numberOfSubdomains -set "$4" "$d/system/decomposeParDict" > /dev/null 2>&1
        foam foamDictionary -entry method -set metis "$d/system/decomposeParDict" > /dev/null 2>&1
        ( cd "$d" && foam decomposePar -force > log.decompose 2>&1 && \
          foam mpirun --bind-to none -np "$4" rhoSimpleFoam -parallel > log.solver 2>&1 )
        rc=$?
        cp "$d/processor0/$3/uniform/profiling" "$d/profiling.txt" 2>/dev/null
    fi
    echo "$1 rc=$rc $(grep -c '^Time =' "$d/log.solver") iterations"
    rm -rf "$d"/processor* "$d/$3"
}
of_run r1-np1 "$G/aerofoilNACA0012/simulation" 50 1
of_run r1-np32 "$G/aerofoilNACA0012/simulation" 50 32
of_run r8-np1 "$G/aerofoilNACA0012-r8/simulation" 10 1
of_run r8-np32 "$G/aerofoilNACA0012-r8/simulation" 10 32
fi
# ---------------------------------------------------------------------------
if want fields; then
stamp "P1-fields"
F=$R/P1-fields; newrec P1-fields
ctr="$F/work"
prep "$G/aerofoilNACA0012/simulation" "$ctr" 1000 50
( cd "$ctr" && foam postProcess -func writeCellCentres -time 0 > log.cellCentres 2>&1 )
cp "$ctr/0/C" "$F/C"
for spec in of1=$R/N6-perf-conv/r1/of1/668 gnc=$R/N6d-perf-conv/r1/gnc/668 gncamg=$R/N6d-perf-conv/r1/gncamg/677; do
    n="${spec%%=*}"; d="${spec#*=}"
    mkdir -p "$F/$n"
    for v in p U T; do cp "$d/$v" "$F/$n/"; done
    echo "$d" > "$F/$n/SOURCE"
done
rm -rf "$ctr"
fi
# ---------------------------------------------------------------------------
if want geometry; then
stamp "P1-geometry"
M=$R/P1-geometry; newrec P1-geometry
mkdir -p "$M/aerofoilNACA0012"
for f in points faces owner boundary; do cp "$G/aerofoilNACA0012/simulation/constant/polyMesh/$f" "$M/aerofoilNACA0012/"; done
geo() {   # name case patches
    local d="$M/work-$1"
    prep "$2" "$d" 1 1
    ( cd "$d" && foam foamToVTK -ascii -no-internal -no-fields -patches "$3" -time 0 > log.foamToVTK 2>&1 )
    mkdir -p "$M/$1"
    find "$d/VTK" -name '*.vtp' -exec cp {} "$M/$1/" \;
    cp "$d/log.foamToVTK" "$M/$1/"
    rm -rf "$d"
}
geo oneraM6 "$C/oneraM6-prepared/simulation" '(wing)'
geo basicFinner "$C/basicFinner-seq5/simulation" '(body fin1 fin2)'
fi
# ---------------------------------------------------------------------------
if want memory; then
stamp "P1-memory"
Y=$R/P1-memory; newrec P1-memory
mem_run() {   # name solver case steps fast(0|1)
    local d="$Y/$1" f=""
    [ "$5" = 1 ] && f="-fast"
    prep "$3" "$d" "$4" 1000000
    bash scripts/gnc_prepare_case.sh -case "$d" --host-env "$E" > "$d/log.prepare" 2>&1
    ( cd "$d" && CUDA_VISIBLE_DEVICES="$GPU_DEVICE" exec "$BIN/$2" -case "$d" $f > log.solver 2>&1 ) & local pid=$!
    nvidia-smi -i "$GPU_DEVICE" --query-compute-apps=pid,used_memory --format=csv,noheader,nounits -lms 100 \
        > "$d/mem.csv" & local sp=$!
    wait "$pid"; local rc=$?
    kill "$sp"
    local peak; peak=$(awk -F', *' -v p="$pid" '$1 == p && $2 + 0 > m {m = $2 + 0} END {print m + 0}' "$d/mem.csv")
    printf 'case=%s\nsolver=%s\nflags=%s\npid=%s\nrc=%s\npeak_MiB=%s\n' "$3" "$2" "$f" "$pid" "$rc" "$peak" > "$d/RUN_INFO"
    rm -rf "$d"/[1-9]*
    echo "$1 rc=$rc peak_MiB=$peak"
}
for r in "" -r2 -r4 -r8; do
    mem_run "rho${r:--r1}-ref" gncRhoSimple "$G/aerofoilNACA0012$r-pcg/simulation" 3 0
    mem_run "rho${r:--r1}-fast" gncRhoSimple "$G/aerofoilNACA0012$r/simulation" 3 1
done
for spec in flatPlate=$C/flatPlate-g2000/simulation rae2822=$C/rae2822-prepared/simulation \
            nacaAirfoil=$C/nacaAirfoil-prepared/simulation oneraM6=$C/oneraM6-prepared/simulation \
            basicFinnerLaminar=$G/basicFinner-laminar/simulation basicFinnerSST=$C/basicFinner-seq5/simulation; do
    n="${spec%%=*}"; d="${spec#*=}"
    mem_run "$n-ref" gncHisa "$d" 3 0
    mem_run "$n-fast" gncHisa "$d" 3 1
done
fi
stamp "done"
