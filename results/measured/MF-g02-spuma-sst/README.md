# 泛用平台路線（HiSA 建在 SPUMA 上）：modifiedFinner SST（g02，N8）

日期：2026-10-25（`RUN_INFO`：09:54–10:07 UTC，開始時 load average 約 120）。主機：**g02**，GPU3（`scripts/n8_g02.sh`
只用 GPU3）。`scripts/n8_g02.sh` 以 `SPUMA_PREFIX=MF-g02-spuma-sst SPUMA_NAME=modifiedFinnerSST` 呼叫
`scripts/spuma_sst_g02.sh`，即 D-026 的 `/ssd/sclin/gpu-platform/scripts/p1_runs.sh`（sha256 見 `RUN_INFO`）。
設定同 `../SPUMA-g02-sst/`：Atlas v2606 hisa（CPU 單核，`taskset -c 157`，`ref/`）與 SPUMA 0.3-v2512 上的 hisa
（build `5916a466-20260728`，與 `../SPUMA-g02-sst/` 相同；`taskset -c 156`，`-pool fixedSizeMemoryPool -poolSize 16`，
`spgpu/`），modifiedFinner SST 案例（3,940,188 格），3 步。每次執行有 `log.hisa` 與 `log.time`（GNU time），
`spgpu/nvsmi.txt` 是量測中的 GPU 程序列表（未標 GPU 編號）；`runSim`、`cleanSim`、`foamLog.db.res` 是從來源案例複製的
檔案（日期 2026-04-01），不是這次執行的輸出。這次執行的 driver 輸出在 `../MF-g02-L3/N8-g02-driver.log` 第 116–160 行：
兩次執行都 exit 0，經過時間 6:01.49（`ref/`）與 6:25.43（`spgpu/`），每步 ExecutionTime 與下表相同。

| 每步 | 第 2 步 | 第 3 步 | 平均 |
|---|---:|---:|---:|
| v2606 hisa，CPU 單核，CPU 時間（ExecutionTime 差） | 106.7 s | 118.4 s | 112.6 s |
| v2606 hisa，CPU 單核，牆鐘（ClockTime 差） | 107 s | 118 s | 112.5 s |
| SPUMA 上的 hisa，GPU3，CPU 時間（ExecutionTime 差） | 94.1 s | 108.9 s | 101.5 s |
| SPUMA 上的 hisa，GPU3，牆鐘（ClockTime 差） | 97 s | 115 s | 106.0 s |

SPUMA 的每步時間以牆鐘為準（論文 Fig. 4：106 s；數據 `docs/paper/figures/fig5_three_routes.csv`，共同作者包中為
`03_data/figure-data/fig04_three_routes.csv`）：SPUMA 等待 GPU 的時間不計入
ExecutionTime（CPU 時間），CPU 單核的兩者則一致（同 `../SPUMA-g02/`、`../SPUMA-g02-sst/`）。以牆鐘計，SPUMA 路線
是論文單核（`../MF-g02-sst/` 的 101.4 s）的 0.96 倍；同一次量測的 `ref/`（112.6 s）比 `../MF-g02-sst/` 的單核慢 11%，
以它為分母則是 1.06 倍。

結果一致性（同一 driver log 第 143–152 行，第 3 步，`spgpu/` 對 `ref/`）：場的 rms(d)/rms(a) 為 rho 1.3e-4、
p 4.7e-4、T 4.3e-4、U 9.1e-4，omega 2.1e-3，k 1.4e-2；alphat 與 nut 達 0.36（max|d|/max|a| 約 280）。p、T、U 的
max|d|/max|a| 為 0.08–0.15，rms 與最大差都大於 basicFinner SST（`../SPUMA-g02-sst/`：rho 1.9e-5、alphat 6.1e-3，
p、T、U 的 max|d|/max|a| 小於 1e-2）。GMRES 殘差在第 1 步就不同（driver log 第 131–141 行）；
GMRES 第 1 次迭代後的殘差，第 2、3 步在 `ref/` 是 2.7e-8、3.8e-8，在 `spgpu/` 是 6.4e-6、9.1e-5（`*/log.hisa`）。
本紀錄沒有查明這些差異的來源，所以這筆只用作每步時間，不作為 SPUMA 上 SST 結果正確的證據。
