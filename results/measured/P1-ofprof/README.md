# P1-ofprof（g02，rhoSimpleFoam 各方程線性求解的時間）

日期：2026-10-25。主機：**g02**（CPU；主機同時有其他使用者的工作，負載未記錄）。建置：論文版的 driver
（`BUILD_INFO` 的雜湊 `1bf4bc6d…` 屬於同次建置的 GPU 程式；本紀錄只執行 OpenFOAM）。
Driver：`scripts/paper_extra_g02.sh --phases nsys,energy,ofprof`（`P1b-driver.log`）。

方法：OpenFOAM v2606.0.2 的 `rhoSimpleFoam`，`controlDict` 加 `profiling { active true; }`，教學案例設定（p 用
GAMG）。單核以 `taskset -c 157`；32 核以 metis 分割與 `mpirun --bind-to none -np 32`（profiling 取 processor0）。

| 執行 | 案例 | 迭代數 |
|---|---|---|
| `r1-np1`、`r1-np32` | aerofoilNACA0012（16,000 格） | 50 |
| `r8-np1`、`r8-np32` | aerofoilNACA0012-r8（1,024,000 格） | 10 |

每個目錄：`log.solver`、`profiling.txt`（`uniform/profiling`），以及案例的 `system/`、`constant/`（不含 polyMesh）。

彙整：`python3 scripts/paper_extra_summary.py` → `summary.tsv`（每迭代的迴圈時間、`fvMatrix::solve.<field>` 的時間）。

結果（每迭代，s）：

| 執行 | 迴圈 | p 求解 | 全部線性求解 |
|---|---|---|---|
| r1-np1 | 0.0435 | 0.0204 | 0.0238 |
| r1-np32 | 0.0359 | 0.0327 | 0.0337 |
| r8-np1 | 17.29 | 15.49 | 15.75 |
| r8-np32 | 1.126 | 1.065 | 1.077 |

與論文表 7 的計時（`N6-perf-*`）相比，迴圈時間差 2–17%；r1-np32 差 56%（每核 500 格的 MPI 執行以通訊為主，最受
主機負載影響）。論文的階段分解表與圖只用本紀錄內的時間計算各部分的比例與 COE。
