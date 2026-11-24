# P1-nsys（g02，效能版每迭代的 kernel、傳輸與 CUDA API）

日期：2026-10-25。主機：**g02**，GPU3（L40），GPU3 連續 3 分鐘沒有其他程式之後才開始。建置：論文版
（原始碼雜湊 `1bf4bc6d…`；`BUILD_INFO`）。Driver：`scripts/paper_extra_g02.sh --phases nsys,energy,ofprof`
（`P1b-driver.log`；該版 driver 的能量階段因參數錯誤失敗，見 `P1-energy`；本階段的指令不受影響）。

方法：`nsys profile --trace=cuda --cuda-graph-trace=node --sample=none --cpuctxsw=none`，`-fast`，兩種迭代數，
`nsys stats` 輸出 kernel、memcpy（時間與大小）、CUDA API 的彙總 CSV（`stats_*.csv`）；`.nsys-rep` 未保留。
兩種迭代數相減再除以迭代數差，得到每迭代的量（扣掉啟動與網格上傳）：

| 執行 | 案例 | 迭代數 |
|---|---|---|
| `rho-r1-n050`、`rho-r1-n100` | gncRhoSimple，aerofoilNACA0012（16,000 格） | 50、100 |
| `rho-r8-n010`、`rho-r8-n020` | gncRhoSimple，aerofoilNACA0012-r8（1,024,000 格） | 10、20 |
| `m6-n010`、`m6-n020` | gncHisa，ONERA M6（294,912 格，SA） | 10、20 |

彙整：`python3 scripts/paper_extra_summary.py` → `summary-kernels.tsv`（依定義 kernel 的原始檔分類）、
`summary-memcpy.tsv`、`summary-api.tsv`。

結果（每迭代）：kernel 時間 2.74、55.1、20.4 ms，占迴圈時間的 72%、80%、89%；gncRhoSimple 的線性求解占 kernel 時間
46%、59%（大部分是 p 的 AMG），gncHisa 的 GMRES／LU-SGS 占 57%。gncRhoSimple 每迭代傳到主機 20 bytes，沒有傳到
裝置的資料。gncHisa 每次解紊流方程都把面的 owner 陣列傳到主機（ONERA M6 3.6 MB）、上傳每格第一個面的索引
（1.2 MB）：對稱 Gauss–Seidel 平滑器的設定，不影響結果，只是這一版沒有快取；呼叫點以 gdb 在 cudaMemcpy 下中斷點
確認（WSL，同一雜湊的乾淨 clone 建置）。

注意：memcpy 的「每迭代時間」對只在啟動時發生的傳輸是兩次執行的差，可能是負的或不為零的雜訊；只看次數與大小。
小量傳輸的大小受 `nsys stats` 的 MB 精度限制（0.001 MB）。
