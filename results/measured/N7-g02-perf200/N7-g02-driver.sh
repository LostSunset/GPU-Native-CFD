#!/usr/bin/env bash
# N7 的 g02 量測（GPU3）：建置；pow／exp 檢查；層級模式 0／1／2 的 L3 逐位元驗證；各變體的每步
# 時間（variant_compare.sh，-profile）；參考版與效能版到收斂的比較。
# 在 g02 上執行：bash scripts/n7_g02.sh > ... 2>&1
#
# 變體：m0／m1／m2 為 GNC_COOP_LEVELS（逐位元）；lusgsColor、sgsColor、tree 為
# 單一效能版選項；fast 為三者合用。
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
E=configs/hosts/g02-l40.env
C=/home/sclin/hisa-gpu-research-cases
P=${N7_PREFIX:-N7-g02}
FAST="GNC_LUSGS=color GNC_SGS=color GNC_REDUCE=tree"
flat=(--case flatPlateSA=$C/flatPlate-g2000/simulation)
rae=(--case rae2822SA=$C/rae2822-prepared/simulation)
onera=(--case oneraM6SA=$C/oneraM6-prepared/simulation)
V=(--variant m0= --variant m1=GNC_COOP_LEVELS=1 --variant m2=GNC_COOP_LEVELS=2
   --variant lusgsColor=GNC_LUSGS=color --variant sgsColor=GNC_SGS=color
   --variant tree=GNC_REDUCE=tree --variant "fast=$FAST")
run() { echo "=== $(date -u +%FT%TZ) $*"; "$@"; echo "=== exit=$?"; }
run bash scripts/build.sh --host-env $E
run bash scripts/pow_check.sh --host-env $E --label $P-powexp
for m in 0 1 2; do
    run env GNC_COOP_LEVELS=$m bash scripts/l3_steady_verify.sh --host-env $E --label $P-verify-m$m \
        --end-default 50 --time-indices 1,2,10,50 \
        --end oneraM6SA=10 --time-indices-for oneraM6SA=1,2,10 \
        --case forwardStep=$C/forwardStep-prepared/simulation "${flat[@]}" "${rae[@]}" "${onera[@]}"
done
run bash scripts/variant_compare.sh --host-env $E --label $P-perf200 --end 200 "${V[@]}" \
    "${flat[@]}" "${rae[@]}"
run bash scripts/variant_compare.sh --host-env $E --label $P-perf-onera --end 30 "${V[@]}" "${onera[@]}"
run bash scripts/variant_compare.sh --host-env $E --label $P-conv --end "${N7_CONV_END:-2000}" \
    --variant ref= --variant lusgsColor=GNC_LUSGS=color --variant "fast=$FAST" "${flat[@]}" "${rae[@]}"
echo "=== done $(date -u +%FT%TZ)"
