#!/usr/bin/env bash
# N6e：aerofoilNACA0012 再加密到每個方向 16 倍（4,096,000 格）與 32 倍（16,384,000 格），進入 SPUMA 論文所說
# 「每張 GPU 超過 1,000 萬格」的範圍（使用者 2026-10-26 指示）。程式是論文版（原始碼雜湊 1bf4bc6d…，
# configs/hosts/g02-paper.env 的獨立目錄），量法與 N6-perf／N6d-perf／BRAE-perf 相同：
#   N6e-perf-r16、N6e-perf-r32    gncRhoSimple 參考版（PCG/DIC，-pcg 案例）、-fast（PCG＋AMG），
#                                 rhoSimpleFoam 單核（taskset）與 32 核 MPI（GAMG）
#   N6e-brae                      brae（每個網格兩種迭代數的差、三次取中位數）
#   N6e-spuma-r16、N6e-spuma-r32  SPUMA 上的 rhoSimpleFoam（GAMG）；記憶體池依網格加大（16、40 GB），
#                                 每次最多 SPUMA_TIMEOUT 秒（預設 8 小時）
#   N6e-gpu.csv                   全程 GPU3 的時間、記憶體用量與功率（每 0.5 秒），對照各執行的時間戳
# --steps spuma：只做 P1-nsys-large（paper_extra_g02.sh --phases nsyslarge）與 SPUMA 兩組（第一次 N6e 在 brae 之後
# 卡在 watchdog，SPUMA 沒有執行；取樣寫到 N6e-gpu-spuma.csv）
# 開始前：等 --wait-pid 結束；再等 --wait-pattern 的程序（另一個 session 的 GPU3／CPU 工作，例如 n11_g02.sh）
# 出現後結束（最多等它出現 --grace 秒）；最後等 GPU3 連續 3 分鐘沒有其他程式。
#
# 在 g02 上執行：bash scripts/n6e_large_g02.sh [--wait-pid PID] [--wait-pattern REGEX] [--grace S] > ... 2>&1
set -u
wait_pid=""; wait_pattern=""; grace=900; steps=all
while [ $# -gt 0 ]; do
    case "$1" in
        --steps) steps="$2"; shift 2 ;;
        --wait-pid) wait_pid="$2"; shift 2 ;;
        --wait-pattern) wait_pattern="$2"; shift 2 ;;
        --grace) grace="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
cd /ssd/sclin/gpu-native-cfd/paper/GPU-Native-CFD
E=configs/hosts/g02-paper.env
. "$E"
G=/ssd/sclin/gpu-native-cfd/cases
R=$RUN_ROOT
PAPER_HASH=1bf4bc6daf538373ee5ce3d15a39c75667f81e22cec9d754db00c7edde099372
SPUMA_TIMEOUT=${SPUMA_TIMEOUT:-28800}
stamp() { echo "=== $(date -u +%FT%TZ) $*"; }
run() { stamp "$*"; "$@"; echo "=== exit=$?"; }
# 符合 --wait-pattern 的程序（自己的使用者），排除本 driver 及其子程序（命令列含 n6e_large_g02.sh）
others() {
    local p
    for p in $(pgrep -u "$(id -u)" -f "$wait_pattern"); do
        tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | grep -q "n6e_large_g02.sh" || echo "$p"
    done
}
# SPUMA 的時間上限：到時只停自己使用者的 SPUMA rhoSimpleFoam（以 -poolSize 辨識）；中括號讓 pgrep 不比對到自己
# 背景子程序的輸出必須導走：否則它佔住 $(...) 的管線，呼叫端要等 sleep 結束才繼續（第一次 N6e 卡住的原因）
spuma_watchdog() {
    ( sleep "$SPUMA_TIMEOUT"
      for p in $(pgrep -u "$(id -u)" -f "rhoSimpleFoa[m] -pool fixedSizeMemoryPool"); do
          echo "watchdog: stopping SPUMA pid $p after $SPUMA_TIMEOUT s" >> "$R/N6e-watchdog.log"; kill "$p"
      done ) > /dev/null 2>&1 & echo $!
}

if [ -n "$wait_pid" ]; then
    stamp "waiting for pid $wait_pid"
    while kill -0 "$wait_pid" 2>/dev/null; do sleep 60; done
fi
if [ -n "$wait_pattern" ]; then
    stamp "waiting up to $grace s for '$wait_pattern' to start"
    t=0
    while [ -z "$(others)" ] && [ "$t" -lt "$grace" ]; do sleep 30; t=$((t + 30)); done
    stamp "waiting for '$wait_pattern' to finish ($(others | tr '\n' ' '))"
    while [ -n "$(others)" ]; do sleep 60; done
fi
free=0
while [ "$free" -lt 3 ]; do
    if [ -z "$(nvidia-smi -i "$GPU_DEVICE" --query-compute-apps=pid --format=csv,noheader)" ]; then
        free=$((free + 1))
    else
        free=0
    fi
    sleep 60
done
stamp "GPU$GPU_DEVICE free; load $(cut -d' ' -f1-3 /proc/loadavg)"

stamp "build"
bash scripts/build.sh --host-env "$E" > "$BUILD_ROOT.log" 2>&1 || { echo "build failed"; exit 1; }
grep -q "source_sha256=$PAPER_HASH" "$BUILD_ROOT/BUILD_INFO" || { echo "source hash differs from the paper"; exit 1; }

gpucsv="$R/N6e-gpu.csv"
[ "$steps" = spuma ] && gpucsv="$R/N6e-gpu-spuma.csv"
nvidia-smi -i "$GPU_DEVICE" --query-gpu=timestamp,memory.used,power.draw.instant --format=csv,noheader,nounits \
    -lms 500 > "$gpucsv" & sampler=$!
trap 'kill $sampler 2>/dev/null' EXIT

if [ "$steps" = all ]; then
run bash scripts/make_aerofoil_meshes.sh --host-env "$E" --src "$G/aerofoilNACA0012/simulation" --out-root "$G" 16 32
df -h /ssd | tail -1

run bash scripts/rhosimple_perf.sh --host-env "$E" --label N6e-perf-r16 --iters 20 \
    --iters-for gnc=5 --iters-for of1=5 --variants gnc,gncamg,of1,of32 \
    --case "r16=$G/aerofoilNACA0012-r16-pcg/simulation" --gamg-case "r16=$G/aerofoilNACA0012-r16/simulation"
run bash scripts/rhosimple_perf.sh --host-env "$E" --label N6e-perf-r32 --iters 10 \
    --iters-for gnc=3 --iters-for of1=3 --variants gnc,gncamg,of1,of32 \
    --case "r32=$G/aerofoilNACA0012-r32-pcg/simulation" --gamg-case "r32=$G/aerofoilNACA0012-r32/simulation"
run env LABEL=N6e-brae BRAE_MESHES="16 32" BRAE_CONV=0 REPEAT=3 bash scripts/brae_g02.sh
fi
if [ "$steps" = spuma ]; then
    # 每迭代時間比格數成長得快：nsys 看是哪些 kernel（P1-nsys-large）
    run bash scripts/paper_extra_g02.sh --phases nsyslarge --no-gpu-wait
fi
wd=$(spuma_watchdog)
run bash scripts/rhosimple_perf.sh --host-env "$E" --label N6e-spuma-r16 --iters 3 \
    --variants spuma --spuma-pool 16 \
    --case "r16=$G/aerofoilNACA0012-r16-pcg/simulation" --gamg-case "r16=$G/aerofoilNACA0012-r16/simulation"
pkill -P "$wd"; kill "$wd" 2>/dev/null
wd=$(spuma_watchdog)
run bash scripts/rhosimple_perf.sh --host-env "$E" --label N6e-spuma-r32 --iters 3 \
    --variants spuma --spuma-pool 40 \
    --case "r32=$G/aerofoilNACA0012-r32-pcg/simulation" --gamg-case "r32=$G/aerofoilNACA0012-r32/simulation"
pkill -P "$wd"; kill "$wd" 2>/dev/null
stamp "done"
