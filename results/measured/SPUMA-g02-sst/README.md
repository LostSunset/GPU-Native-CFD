# 泛用平台路線（HiSA 建在 SPUMA 上）：basicFinner SST（g02）

日期：2026-10-23。腳本：`scripts/spuma_sst_g02.sh`（commit `5bd58fc323`），呼叫 D-026 的
`/ssd/sclin/gpu-platform/scripts/p1_runs.sh`（sha256 見 `RUN_INFO`）；紀錄：`SPUMA-g02-sst-driver.log`。
設定同 `../SPUMA-g02/`：Atlas v2606 hisa（CPU 單核，`ref/`）與 SPUMA 0.3-v2512（Atlas overlay）上的 hisa
（GPU3，`-pool fixedSizeMemoryPool -poolSize 16`，`spgpu/`），原本的 basicFinner SST 案例（2,179,881 格），3 步。

| 每步 | 第 2 步 | 第 3 步 | 平均 |
|---|---:|---:|---:|
| v2606 hisa，CPU 單核，CPU 時間（ExecutionTime 差） | 55.4 s | 59.2 s | 57.3 s |
| v2606 hisa，CPU 單核，牆鐘（ClockTime 差） | 56 s | 59 s | 57.5 s |
| SPUMA 上的 hisa，GPU3，CPU 時間（ExecutionTime 差） | 58.6 s | 64.5 s | 61.5 s |
| SPUMA 上的 hisa，GPU3，牆鐘（ClockTime 差） | 61 s | 67 s | 64.0 s |

SPUMA 的每步時間以牆鐘為準：SPUMA 等待 GPU 的時間不計入 ExecutionTime（CPU 時間），CPU 單核的兩者則一致
（同 `../SPUMA-g02/`）。以牆鐘計，SPUMA 路線比同一次量測的單核慢 11%（64.0 對 57.5 s；以 CPU 時間是 7%），
對 `../SSTP-g02-finner/` 的單核 53.2 s 為 0.83 倍；結果與 CPU 相同到
rms 相對差 1e-4（第 3 步，`SPUMA-g02-sst-driver.log`）。與層流的 basicFinner（`../SPUMA-g02/`，0.84 倍）
相近：HiSA 自己的迴圈留在主機，經 unified memory 存取 GPU 上的場。
