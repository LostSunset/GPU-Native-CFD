# 突變測試：逐位元比對的定位能力（g02）

日期：2026-10-23。主機：**g02**，GPU3（L40）。驅動：`scripts/mutation_g02.sh`（紀錄 `MUT-g02-driver.log`：
tanh、assoc、gather、fmad；`MUT2-g02-driver.log`：reduce、ilu，在 N6b 重建之後以同一個正式建置執行）。
彙整：`scripts/mutation_summary.py`（輸出 `mutation_summary.txt`）。

案例：aerofoilNACA0012（p 為 PCG＋DIC 的驗證變體），10 迭代，傾印第 1、2、10 迭代的 U、E、p、turb 四個階段
與初始化（`scripts/rhosimple_verify.sh --dump-steps 1,2,10 --end aerofoilNACA0012=10`）。10 迭代沒有寫出時間
目錄，L3 欄是空的；比的是 L2 傾印。參考端都是未修改的 rhoSimpleFoam（`gncRhoSimpleRef`）。

| 紀錄 | 注入的改動 | 建置 |
|---|---|---|
| `MUT-reduce` | `GNC_REDUCE=tree`：循序和改為固定樹狀歸約 | 正式建置（同 `N6b-L3`） |
| `MUT-ilu` | `GNC_ILU=color`：DILU／DIC 改為多色次序（前置條件本身不同） | 正式建置 |
| `MUT-tanh` | kOmegaSST 的 F1：glibc tanh 的移植 → CUDA 的 `tanh` | `scripts/mutate.py tanh` 的副本 |
| `MUT-assoc` | 純量 laplacian 的面係數：`deltaCoeffs*(gamma*magSf)` → `gamma*(magSf*deltaCoeffs)` | `mutate.py assoc` |
| `MUT-gather` | `fvc::surfaceIntegrate` 每格的面以相反次序累加 | `mutate.py gather` |
| `MUT-fmad` | 原始碼不變，裝置碼 `--fmad=true` | `build.sh --fmad true` |

## 第一個出現不一致的傾印

| 突變 | 第一個不一致 | 該處不一致的紀錄 | 該處仍逐位元的紀錄 |
|---|---|---|---|
| fmad | 初始化 | phi 的邊界值 16／480 個是 −0（參考為 +0；差 0、0 ulp） | 其他全部；第 1 迭代起 UEqn 的 upper／lower 不同（最多 315 ulp） |
| reduce | 第 1 迭代 U | U（14–15 ulp） | UEqn 的全部係數（diag、upper、lower、source、邊界係數）與其他場 |
| ilu | 第 1 迭代 U | U | UEqn 的全部係數 |
| gather | 第 1 迭代 U | UEqn.diag（574 格，≤ 4 ulp）、UEqn.source、U | UEqn.upper、lower、邊界係數 |
| assoc | 第 1 迭代 E | T、e、psi（75–215 格，1–3 ulp） | U 階段全部（向量 laplacian 是另一個 kernel） |
| tanh | 第 2 迭代 turb | omega（2 格，1 ulp） | 第 1 迭代全部；到第 10 迭代 p、U、T 仍逐位元，只有 omega、nut、alphat（2–6 ulp）與 UEqn 的少數面係數不同 |

## 擴散

差異碰到以容許值停止的線性求解後會放大很多個數量級：assoc 的 1–3 ulp 在第一次 p 求解後成為 p 的相對差
1.7e-5（最大絕對差 0.37 Pa），第 10 迭代 4.1e-5；gather 第 1 迭代 p 1.1e-9、第 10 迭代 4.8e-8；fmad 4.4e-5；
reduce 只有 1.3e-13（PCG 的路徑幾乎相同）。細節見 `mutation_summary.txt` 與各紀錄的 `aerofoilNACA0012/compare.*.txt`。
