# 泛用平台路線：HiSA 建在 SPUMA 上（D-026 P1），與 gncHisa 相同的案例（g02）

日期：2026-10-23。主機：**g02**，GPU3（L40）。腳本：`SPUMA-g02-driver.sh`（`scripts/spuma_compare_g02.sh`，呼叫
D-026 的 `/ssd/sclin/gpu-platform/scripts/p1_runs.sh`，SHA-256 見 `RUN_INFO`）。每個案例同時跑 Atlas v2606 的 `hisa`
（CPU 單核，`taskset -c 157`，`ref/`）與 SPUMA 0.3-v2512 上的 `hisa`（GPU3，`taskset -c 156`，`-pool fixedSizeMemoryPool`，
`spgpu/`）。主機 CPU 有其他使用者的 MPI 作業；量測期間 GPU0–2 有其他使用者的程序，GPU3 沒有（`spgpu/nvsmi.txt` 是全部
GPU 的列表，其中的 python 程序在 GPU0）。

| 案例 | 步數 | CPU 單核每步（`ref/`，CPU 時間；平均的牆鐘） | SPUMA GPU 每步，牆鐘（ClockTime 差） | SPUMA GPU 每步，CPU 時間（ExecutionTime 差） | SPUMA 對 CPU（牆鐘；對 `ref/`／對論文的單核） | 與 v2606 的差（最後一步，場的 max\|d\|/max\|a\|） |
|---|---:|---:|---:|---:|---:|---|
| ONERA M6 SA（官方設定） | 10 | 3.3–3.5 s，第 10 步 5.3 s（平均 3.62 s；牆鐘 3.56 s） | 13–21 s（平均 17.1 s） | 12.9–16.0 s（平均 14.2 s） | 0.21／0.22 倍 | 2e-5（p、T、rho）到 2e-4（U） |
| basicFinner 層流（`cases/basicFinner-laminar`，網格以符號連結） | 3 | 47–53 s（平均 50.2 s；牆鐘 50.5 s） | 53–64 s（平均 58.5 s） | 52–61 s（平均 56.6 s） | 0.86／0.84 倍 | 2e-3（rho）到 1.7e-2（p、T）（3 步，SPUMA 的 HiSA 為正確的轉置，見 N-013） |

SPUMA 的每步時間以牆鐘（ClockTime 差）為準（同 `../N6e-spuma-*` 的 rhoSimpleFoam 用 `per_iter_clock_s`）：SPUMA
等待 GPU 的時間不計入 ExecutionTime（CPU 時間），ONERA M6 的 CPU 時間因此比牆鐘少 17%；CPU 單核的兩者相差 2% 以內，
沿用 ExecutionTime。
「論文的單核」是 `../CPU2-g02-onera/` 的 3.75 s 與 `../CPU2-g02-finner/` 的 48.9 s（論文 Fig. 4；數據
`docs/paper/figures/fig5_three_routes.csv`，共同作者包中為 `03_data/figure-data/fig04_three_routes.csv`）。
ONERA M6 的牆鐘每步在 10 步內由 13 s 增加到 21 s（CPU 時間 12.9–16.0 s），平均值與步數有關。

SPUMA 上的 HiSA 結果正確，但每步比 Atlas v2606 的 CPU 單核慢：泛用平台把 OpenFOAM 的每個場運算各自搬上 GPU，
小 kernel 與統一記憶體的開銷大於計算量（D-026 P1 的 nsys：`docs/planning/gpu-platform-p1/`）。同案例的三條路線比較見
論文 Section 5.2 與 Fig. 4（共同作者包中為 `02_figures/fig04_three_routes.pdf`；repo 內另見
`docs/paper/results-summary.md` §3.3）。
