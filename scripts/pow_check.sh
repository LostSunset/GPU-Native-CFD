#!/usr/bin/env bash
# GPU 的 pow、exp、tanh、expm1、log 與主機 libm（glibc）逐位元比較的紀錄
# （tests/gncPowCheck）。
#
# pow：CUDA ::pow 與 glibc 移植版（core/glibcPow.cuh），各在 SA 的 fw 底數範圍
# （x 在 (0, 65/64]、y = 1/6，1e8 個）與寬範圍隨機（任意正的有限 x、y 在 [-8, 8]，
# 5e7 個）。exp：CUDA ::exp 與 glibc 移植版（core/glibcExp.cuh），各在
# nutUSpaldingWallFunction 的範圍（x 在 [0, 50]，1e8 個）與寬範圍隨機（5e7 個）。
# tanh（kOmegaSST 的 F1、F2，x 在 [-25, 25]）、expm1（tanh 內部，[-50, 50]）、log
# （壁面函數，(0, 1e7]）同樣各兩種取樣（N4b，core/glibcTanh.cuh、glibcLog.cuh）。
# 另記下主機 libm 的路徑與 sha256，以及 gdb 解析出的實作（IFUNC 依 CPU 選擇；
# 移植對應 __ieee754_pow_fma、__ieee754_exp_fma、__expm1_fma、__ieee754_log_fma；
# __tanh 沒有 IFUNC，經 PLT 呼叫 __expm1）。
#
# 用法：pow_check.sh --host-env FILE --label LABEL

set -u

host_env=""; label=""
while [ $# -gt 0 ]; do
    case "$1" in
        --host-env) host_env="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        -h|--help) sed -n '2,12p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 2 ;;
    esac
done
[ -r "$host_env" ] || { echo "ERROR: --host-env FILE required" >&2; exit 2; }
# shellcheck disable=SC1090
. "$host_env"
case "$label" in ""|*/*|.*) echo "ERROR: --label NAME required" >&2; exit 2 ;; esac

bin="$BUILD_ROOT/bin"
[ -x "$bin/gncPowCheck" ] || { echo "ERROR: $bin/gncPowCheck missing (run scripts/build.sh)" >&2; exit 2; }
out="$RUN_ROOT/$label"
[ -e "$out" ] && { echo "ERROR: $out exists" >&2; exit 2; }
mkdir -p "$out"

libm="$(readlink -f /lib/x86_64-linux-gnu/libm.so.6)"
{
    echo "label=$label"
    echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname) profile=$HOST_PROFILE"
    echo "gpu_device=$GPU_DEVICE"
    echo "cpu=$(grep -m1 'model name' /proc/cpuinfo | sed 's/.*: //')"
    echo "cpu_fma=$(grep -m1 -o -w fma /proc/cpuinfo || echo no) cpu_avx2=$(grep -m1 -o -w avx2 /proc/cpuinfo || echo no)"
    echo "glibc=$(ldd --version | head -1)"
    echo "libm=$libm sha256=$(sha256sum "$libm" | awk '{print $1}')"
} > "$out/RUN_INFO"
cp "$BUILD_ROOT/BUILD_INFO" "$out/BUILD_INFO"

# pow、exp 的 IFUNC 實際解析到哪個函數（需要 gdb 與 libc6-dbg；沒有就略過）
if command -v gdb >/dev/null 2>&1 && command -v gcc >/dev/null 2>&1; then
    tmp="$(mktemp -d)"
    printf '%s\n' '#include <math.h>' '#include <stdio.h>' \
        'volatile double a = 0.5, b = 0.16666666666666666;' \
        'int main(void) { printf("%.17g %.17g %.17g %.17g\n", pow(a, b), exp(a), expm1(a), log(a)); return 0; }' > "$tmp/p.c"
    if gcc -O0 -fno-builtin -o "$tmp/p" "$tmp/p.c" -lm; then
        for fn in pow exp expm1 log; do
            gdb -batch -ex "set debuginfod enabled off" -ex "break $fn" -ex run \
                -ex "stepi 10" -ex 'info symbol $pc' "$tmp/p" 2>&1 \
            | grep -E "in section" | tail -1
        done > "$out/ifunc.txt"
    fi
    rm -rf "$tmp"
fi

printf 'func\timpl\tmode\tsamples\tdiffer\tmax_ulp\n' > "$out/SUMMARY.tsv"
overall=0
for spec in "pow cuda sa 100000000" "pow glibc sa 100000000" "pow cuda wide 50000000" \
            "pow glibc wide 50000000" "exp cuda sa 100000000" "exp glibc sa 100000000" \
            "exp cuda wide 50000000" "exp glibc wide 50000000" \
            "tanh cuda sa 100000000" "tanh glibc sa 100000000" "tanh cuda wide 50000000" \
            "tanh glibc wide 50000000" "expm1 cuda sa 100000000" "expm1 glibc sa 100000000" \
            "expm1 cuda wide 50000000" "expm1 glibc wide 50000000" "log cuda sa 100000000" \
            "log glibc sa 100000000" "log cuda wide 50000000" "log glibc wide 50000000"; do
    set -- $spec
    log="$out/log.$1.$2.$3"
    CUDA_VISIBLE_DEVICES="$GPU_DEVICE" "$bin/gncPowCheck" --func "$1" --impl "$2" --mode "$3" \
        --n "$4" > "$log" 2>&1
    rc=$?
    line="$(cat "$log")"
    differ="$(echo "$line" | sed -n 's/.* samples, \([0-9]*\) differ.*/\1/p')"
    ulp="$(echo "$line" | sed -n 's/.*max \([0-9]*\) ulp.*/\1/p')"
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$differ" "$ulp" >> "$out/SUMMARY.tsv"
    # 移植版必須全部相同；CUDA 的差異是預期的
    if [ "$2" = glibc ] && [ "$rc" -ne 0 ]; then
        overall=1
    fi
    [ "$rc" -gt 1 ] && overall=1
done

echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$out/RUN_INFO"
echo "overall_exit=$overall" >> "$out/RUN_INFO"
column -t -s "$(printf '\t')" "$out/SUMMARY.tsv"
exit "$overall"
