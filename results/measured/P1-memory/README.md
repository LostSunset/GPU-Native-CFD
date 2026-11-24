# P1-memory（g02，各案例的 GPU 記憶體峰值）

日期：2026-10-25。主機：**g02**，GPU3（L40，46,068 MiB 可用），GPU3 連續 3 分鐘沒有其他程式之後才開始。
建置：論文版（原始碼雜湊 `1bf4bc6d…`；`BUILD_INFO`）。Driver：`scripts/paper_extra_g02.sh --phases energy,memory`
（`P1d-driver.log`）。

方法：每個案例以參考版與 `-fast` 各跑 3 迭代，同時以 `nvidia-smi --query-compute-apps=pid,used_memory -lms 100`
取樣（`mem.csv`），取求解器 PID 的最大值（含 CUDA context）。每個目錄的 `RUN_INFO` 記錄案例、旗標、PID 與峰值。

彙整：`python3 scripts/paper_extra_summary.py` → `summary.tsv`（峰值、每格位元組；gncRhoSimple 四套網格的直線擬合）。

結果（MiB）：

| 案例 | 格數 | 參考版 | -fast |
|---|---|---|---|
| aerofoilNACA0012（r1、r2、r4、r8） | 16,000–1,024,000 | 460、556、844、2,092 | 458、524、908、2,316 |
| flatPlate SA | 8,000 | 460 | 460 |
| RAE2822 SA | 23,552 | 524 | 492 |
| nacaAirfoil SST | 40,000 | 588 | 492 |
| ONERA M6 SA | 294,912 | 1,516 | 1,708 |
| basicFinner 層流 | 2,179,881 | 7,820 | 9,164 |
| basicFinner SST | 2,179,881 | 8,364 | 9,708 |

gncRhoSimple 的直線擬合：參考版每格 1,690 bytes、截距 440 MiB；`-fast` 每格 1,942 bytes、截距 422 MiB。
gncHisa（扣掉平板 460 MiB 後除以格數差）：參考版每格 3.6–3.9 kB，`-fast` 4.2–4.6 kB。

注意：nvidia-smi 的取樣間隔 100 ms，短暫的峰值可能漏掉；3 迭代涵蓋啟動與一般迭代的配置。
