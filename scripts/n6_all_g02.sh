#!/usr/bin/env bash
# N6 的正式紀錄：先驗證（n6_g02.sh：建置、L2／L3、回歸），再量效能（n6_perf_g02.sh）。
# 在 g02 上執行：bash scripts/n6_all_g02.sh > ... 2>&1
set -u
cd /ssd/sclin/gpu-native-cfd/GPU-Native-CFD
bash scripts/n6_g02.sh
bash scripts/n6_perf_g02.sh
