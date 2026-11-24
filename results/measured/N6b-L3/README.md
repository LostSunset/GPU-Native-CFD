# N6b：AMG 預設改變之後的逐位元確認（g02）

日期：2026-10-23。主機：**g02**，GPU3（L40）。驅動：`scripts/n6b_g02.sh`（紀錄 `N6b-g02-driver.log`）。
建置的原始碼雜湊 `d1a03b25…59627`（`BUILD_INFO`，0 個警告）：`-fast` 的 AMG 改為每 2 個配對層合成一層、
前後掃各 2 次、粗網格修正乘 1.8（N-026）。參考次序版的程式沒有改動；這裡確認同一個建置的參考版仍逐位元。
第一次執行（原始碼雜湊 `9ec9b690…`，之後只改了 `amg.cuh`、`fvScalarMatrix.cu` 的註解）的紀錄在 g02 改名為
`N6bpre-*` 保留；依「改了原始碼（含註解）就重建重跑」的規則，以目前的原始碼重建後重跑，逐位元的結果相同，
計時在量測誤差內（1,024,000 格 `-fast` 0.109 → 0.111 s）。

| 紀錄 | 內容 | 結果 |
|---|---|---|
| `N6b-L3/` | gncRhoSimple 對 rhoSimpleFoam（aerofoilNACA0012，p 為 PCG＋DIC）：初始化與第 1、2、10、100、668 迭代的傾印；寫出的場檔 | **21/21、126/126 逐位元**，兩邊都在第 668 迭代收斂；兩次 GPU 執行相同 |
| `N6b-reg-SST-L3/` | gncHisa nacaAirfoil SST 200 步 | 7/7 |
| `N6b-reg-SA-L3/` | gncHisa RAE2822 SA 200 步 | 7/7 |

效能的紀錄見 `../N6b-perf-conv/README.md`。
