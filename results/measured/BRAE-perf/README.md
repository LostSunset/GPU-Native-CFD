# brae 在同一張 L40、同一組 aerofoilNACA0012 網格上（g02）

日期：2026-10-24（UTC 2026-10-23）。主機：**g02**，NVIDIA L40（GPU3）。驅動：`scripts/brae_g02.sh`（紀錄
`BRAE-g02-driver.log`）。使用者 2026-10-24 同意在 g02 clone brae；brae 只執行，不併入本專案。

brae（github.com/simd-ai/brae，AGPL-3.0）是把 OpenFOAM 的 simpleFoam、pimpleFoam、rhoSimpleFoam、interFoam
重寫成 CUDA 的專案，整個求解在一張 GPU 上，是相關研究中與 gncRhoSimple 最接近的工作。

- 版本：commit `8ecfc74a`（2026-10-08），`cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89`，
  nvcc 13.2，只建 `brae` 與 `brae_rhoSimpleFoam`（`RUN_INFO` 有執行檔的 SHA-256）。它要求 SCOTCH：g02 沒有系統
  套件，用同一帳號既有 conda 環境裡的 SCOTCH 7.0.11 的兩個函式庫（複製到 `ext/scotch-7.0.11`，只連這兩個，
  避免混到該環境的 libz、libgomp）；rhoSimpleFoam 的執行檔實際上沒有連到 SCOTCH。
- 執行：`BRAE_RHOSIMPLEFOAM_MIRROR=cuda`（GPU 版），`FOAM_ETC` 指向 OpenFOAM v2606 的 etc（`#includeEtc` 用）。
- 案例：與 `N6-perf-*`、`N6b-perf-*` 相同的 GAMG 原設定與加密網格，只改 endTime 與 writeInterval。

brae 依案例的 fvSolution 執行，但宣告了下列近似（`*/log.err` 的 NOTICE）：

| 欄位 | 案例 | brae |
|---|---|---|
| p | GAMG | PCG＋GAMG 前置條件（同一個線性系統與容許值） |
| U | PBiCGStab＋DILU | 多色 symGaussSeidel smoothSolver（同樣的容許值、relTol） |
| e、k、omega | PBiCGStab＋DILU | 前置條件改為 6 階截斷 Neumann 級數 |

也就是說 brae 預設就是一個效能版，對應 gncRhoSimple 的 `-fast`（p 以 PCG＋聚合式 AMG 代替 GAMG）。

## 每 SIMPLE 迭代的時間

brae 不印每迭代時間。每個網格跑 n1、n2 兩個迭代數，每迭代時間 =（牆鐘(n2) − 牆鐘(n1)）／（n2 − n1），
扣掉啟動、讀網格與最後的寫出；重複 3 次取中位數（`per_iter_each_s` 是三次的值）。gnc 為求解器的迴圈時間
（`N6b-perf-*`，迴圈中不寫場）。

| 格數 | brae | gncRhoSimple `-fast`（N6b） | brae 對 `-fast` | brae 的 p PCG 次數 | `-fast` 的 p PCG 次數 |
|---:|---:|---:|---:|---:|---:|
| 16,000 | 0.00405 s | 0.00855 s | 2.1 倍快 | 15.3 | 10.7 |
| 64,000 | 0.00644 s | 0.0148 s | 2.3 倍快 | 21.4 | 18.1 |
| 256,000 | 0.0188 s | 0.0313 s | 1.7 倍快 | 45.3 | 29.6 |
| 1,024,000 | 0.0925 s | 0.111 s | 1.2 倍快 | 68.0 | 40.6 |

## 到收斂（16,000 格，residualControl 1e-4，迴圈中不寫場）

| 執行 | 迭代數 | 牆鐘（總，含啟動） |
|---|---:|---:|
| brae | 676 | 3.2 s |
| gncRhoSimple `-fast`（N6b） | 678 | 6.2 s |
| OpenFOAM 單核（N6） | 668 | 25.7 s |

收斂解對 OpenFOAM 單核解的 L2 相對差（`conv/fielddiff.of1-brae.txt`）：p 5.9e-5、U 7.1e-5、k 5.2e-4，與
gncRhoSimple `-fast`（7.4e-5、9.0e-5、5.9e-4）同量級。

## 觀察

開發用的階段計時（`scripts/dev/prof_vs_brae_g02.sh`，gnc `-profile` 對 brae `BRAE_PHASE_TIME=1`）：16,000 格
p 求解每迭代 5.6 對 2.2 ms，1,024,000 格約 86 對 51 ms；動量、能量、紊流在 1,024,000 格是 gnc 較快
（24 對 34 ms）。gncRhoSimple 的 p 的 PCG 次數較少，但每次 PCG 迭代（一次 AMG V-cycle）約貴 3 倍：每層的多色
Gauss–Seidel 經由面編號雙重間接存取、全部 FP64，粗層的 kernel 啟動次數多。改良見 N-027。

N-027 之後（`../N6c-perf-conv/README.md`）：gncRhoSimple `-fast` 為 0.00536、0.00886、0.0191、0.0765 s，
對 brae 為 0.76、0.73、0.98、1.21 倍的速度（102 萬格比 brae 快 1.2 倍）；16,000 格到收斂 4.0 對 3.2 s。

N-028 之後（`../N6d-perf-conv/README.md`）：0.00379、0.00569、0.0142、0.0691 s，對 brae 為 1.07、1.13、1.32、
1.34 倍的速度（四個網格都較快；16,000 格的差在重複量測的 7% 之內）；16,000 格到收斂 2.79 對 3.2 s。
