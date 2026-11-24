#!/usr/bin/env bash
# 逐位元測試組的定位能力（突變測試）：gncRhoSimple 對 rhoSimpleFoam（aerofoilNACA0012，p 為 PCG＋DIC），
# 注入只差捨入的改動，跑 10 迭代的 L2 傾印比對（第 1、2、10 迭代）與 L3 場檔比對。
#   reduce  GNC_REDUCE=tree：循序和改為固定樹狀歸約（正式建置，執行期開關）
#   ilu     GNC_ILU=color：DILU／DIC 改為多色次序（正式建置，執行期開關；改變前置條件本身）
#   tanh、assoc、gather  scripts/mutate.py 在原始碼副本上的單點改動，各自建置
#   fmad    原始碼不變，裝置碼以 --fmad=true 建置
# 副本與建置在 /ssd/sclin/gpu-native-cfd/mut/<名稱>/，紀錄在 runs/MUT-<名稱>/。
# 在 g02 上執行：bash scripts/mutation_g02.sh [名稱 ...] > ... 2>&1（預設全部；reduce、ilu 用正式建置，
# 先以 scripts/build.sh 建好）
set -u
only=" ${*:-reduce ilu tanh assoc gather fmad} "
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
G=/ssd/sclin/gpu-native-cfd/cases
M=/ssd/sclin/gpu-native-cfd/mut
CASE="aerofoilNACA0012=$G/aerofoilNACA0012-pcg/simulation"
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
verify() {
    local label="$1" root="$2" env_file="$3"
    shift 3
    run env "$@" bash "$root/scripts/rhosimple_verify.sh" --host-env "$env_file" --label "$label" \
        --dump-steps 1,2,10 --end aerofoilNACA0012=10 --case "$CASE"
}
want() { case "$only" in *" $1 "*) return 0 ;; esac; return 1; }
want reduce && verify MUT-reduce . "$E" GNC_REDUCE=tree
want ilu && verify MUT-ilu . "$E" GNC_ILU=color
for m in tanh assoc gather fmad; do
    want "$m" || continue
    root="$M/$m/GPU-Native-CFD"
    rm -rf "${M:?}/$m"
    mkdir -p "$root"
    cp -a CMakeLists.txt src apps tools tests scripts configs "$root"/
    [ -f SYNC_INFO ] && cp SYNC_INFO "$root"/
    sed -e "s#^BUILD_ROOT=.*#BUILD_ROOT=$M/$m/build#" -e "s#^SRC_DIR=.*#SRC_DIR=$root#" "$E" > "$M/$m/host.env"
    fm=false
    if [ "$m" = fmad ]; then
        fm=true
    else
        run python3 -I scripts/mutate.py "$root" "$m" || continue
    fi
    run bash "$root/scripts/build.sh" --host-env "$M/$m/host.env" --fmad "$fm"
    verify "MUT-$m" "$root" "$M/$m/host.env"
done
echo "=== done $(date -u +%FT%TZ)"
