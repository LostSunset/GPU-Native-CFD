# P1-nsys-large（g02，大網格的 kernel、傳輸與 CUDA API）

日期：2026-10-26。主機：**g02**，GPU3（L40），GPU3 連續 3 分鐘沒有其他程式之後才開始。建置：論文版（原始碼雜湊
`1bf4bc6d…`；`BUILD_INFO`）。Driver：`scripts/n6e_large_g02.sh --steps spuma` 呼叫
`scripts/paper_extra_g02.sh --phases nsyslarge --no-gpu-wait`（log 在 `../N6e-gpu/N6e-spuma-driver.log`）。

方法同 `../P1-nsys`（`nsys profile --trace=cuda --cuda-graph-trace=node`，兩種迭代數相減）：

| 執行 | 案例 | 迭代 |
|---|---|---|
| `rho-r16-n010`、`rho-r16-n020` | aerofoilNACA0012-r16（4,096,000 格），-fast | 10、20 |
| `rho-r32-n005`、`rho-r32-n010` | aerofoilNACA0012-r32（16,384,000 格），-fast | 5、10 |

彙整：`python3 scripts/paper_extra_summary.py --src <論文版的 checkout>` → `summary-kernels.tsv`、
`summary-memcpy.tsv`、`summary-api.tsv`。

結果（每迭代）：

| 網格 | kernel 時間 | 迴圈時間（N6e） | 占比 | 同步的 cudaMemcpy |
|---|---|---|---|---|
| 1,024,000（`../P1-nsys`） | 55.1 ms | 69.1 ms | 80% | 12 |
| 4,096,000 | 356 ms | 457 ms | 78% | 13 |
| 16,384,000 | 2,054 ms | 3,397 ms | 60% | 500（下 304、上 187） |

兩個原因：

1. 單次啟動的樹狀歸約（`src/core/treeReduce.cuh`：每塊 2048 個元素，`reduce.cu` 的 `sumTreeToDevice`）最多處理
   2048² = 4,194,304 個元素；r32 超過，每次歸約改走 `sumTreeOp`（多次啟動，結果經主機複製回裝置），每迭代約 500 次
   同步複製，GPU 只有 60% 的時間在跑 kernel。
2. kernel 時間成長 37 倍（格數 16 倍），主要在 AMG：每次呼叫 `kResRestrict` 31 倍、`kIn`（依顏色排列 gather 並轉
   FP32）67 倍、`kOut`（scatter）105 倍；與大網格下依排列存取的區域性變差一致（未做更細的硬體計數器剖析）。

注意：memcpy 的「每迭代時間」對只在啟動時發生的傳輸是兩次執行的差（雜訊，可能為負）；只看次數與大小。
