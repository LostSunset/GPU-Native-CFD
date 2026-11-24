# P1-fields（g02，aerofoilNACA0012 的收斂場）

日期：2026-10-25。主機：**g02**。建置：論文版（原始碼雜湊 `1bf4bc6d…`；`BUILD_INFO`）。
Driver：`scripts/paper_extra_g02.sh --phases host,cases,fields,geometry --no-gpu-wait`（`P1a-driver.log`）。

`aerofoilNACA0012`（16,000 格）收斂解（residual control 1e-4）的 `p`、`U`、`T`，取自既有紀錄的執行目錄
（各子目錄的 `SOURCE` 記錄來源）：

| 目錄 | 來源 |
|---|---|
| `of1/` | `N6-perf-conv/r1/of1/668`（rhoSimpleFoam 單核，668 迭代收斂） |
| `gnc/` | `N6d-perf-conv/r1/gnc/668`（gncRhoSimple 參考版，PCG/DIC） |
| `gncamg/` | `N6d-perf-conv/r1/gncamg/677`（gncRhoSimple `-fast`，PCG＋AMG） |

`C`：格心座標（`postProcess -func writeCellCentres -time 0`）。

用於論文的流場比較圖（`scripts/plots/make_case_figures.py`，輸出 `aerofoil_fields.{pdf,png,csv}`）。
