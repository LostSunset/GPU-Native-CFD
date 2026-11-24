# P1-cases（g02，各案例的設定與網格大小）

日期：2026-10-25。主機：**g02**。建置：論文版（原始碼雜湊 `1bf4bc6d…`；`BUILD_INFO`）。
Driver：`scripts/paper_extra_g02.sh --phases host,cases,fields,geometry --no-gpu-wait`（`P1a-driver.log`）。

每個案例一個目錄：`system/` 的 `fvSolution`、`fvSchemes`、`controlDict`，`constant/polyMesh/boundary`，以及
`owner-note.txt`（`owner` 檔頭的 `nPoints nCells nFaces nInternalFaces`）。

注意：`basicFinner/owner-note.txt` 是空的。這個網格的檔頭寫成 `nCells: 2179881`（冒號後有空格），當時的 grep
沒有容許空格（之後的 driver 已修正）；格數見求解器 log 的 `mesh: 2179881 cells`（例如 `CPU2-g02-finner`）。

用於論文的案例表（面數）與附錄 B 的求解設定。
