# P1-geometry（g02，網格圖用的幾何）

日期：2026-10-25。主機：**g02**。建置：論文版（原始碼雜湊 `1bf4bc6d…`；`BUILD_INFO`）。
Driver：`scripts/paper_extra_g02.sh --phases host,cases,fields,geometry --no-gpu-wait`（`P1a-driver.log`）。

| 目錄 | 內容 |
|---|---|
| `aerofoilNACA0012/` | 16,000 格網格的 `points`、`faces`、`owner`、`boundary`（ASCII） |
| `oneraM6/` | 機翼表面（`foamToVTK -ascii -no-internal -no-fields -patches '(wing)'`）與 log |
| `basicFinner/` | 彈體與兩片尾翼的表面（`-patches '(body fin1 fin2)'`）與 log |

用於論文的網格圖（`scripts/plots/make_case_figures.py`，輸出 `case_meshes.{pdf,png}`）。
