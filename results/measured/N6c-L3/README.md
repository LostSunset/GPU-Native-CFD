# N6c：N-027 之後的逐位元確認（g02）

日期：2026-10-24（UTC 2026-10-23）。主機：**g02**，GPU3（L40）。驅動：`scripts/n6c_g02.sh`（紀錄
`N6c-g02-driver.log`）。建置的原始碼雜湊 `587a0b4f…6e399`（`BUILD_INFO`，0 個警告）：效能版的 AMG 套用格式
（依顏色重新編號的 CSR、FP32、粗層單一 block、殘差與限制合併、二分圖兩色）與單次啟動的 sumTree（N-027）。
參考次序版的程式沒有改動。

| 紀錄 | 內容 | 結果 |
|---|---|---|
| `N6c-sumcheck/` | `gncSumBench --n 1,7,2048,2049,16000,2179881,4194304,4194305`，`GNC_SUMTREE_CHECK=1`：每次呼叫同時算單次啟動與多次啟動版，不同就中止 | 全部相同（rc 0；4,194,305 走多次啟動的退回路徑） |
| `N6c-L3/` | gncRhoSimple 對 rhoSimpleFoam（aerofoilNACA0012，p 為 PCG＋DIC）：傾印與寫出的場 | **21/21、126/126 逐位元**，第 668 迭代收斂；兩次 GPU 執行相同 |
| `N6c-reg-SST-L3/` | gncHisa nacaAirfoil SST 200 步 | 7/7 |
| `N6c-reg-SA-L3/` | gncHisa RAE2822 SA 200 步 | 7/7 |

效能的紀錄見 `../N6c-perf-conv/README.md`。
