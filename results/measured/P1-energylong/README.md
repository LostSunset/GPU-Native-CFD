# P1-energylong（g02，GPU 板功率：長的迴圈）

日期：2026-10-26。主機：**g02**，GPU3（L40），GPU3 連續 3 分鐘沒有其他程式之後才開始。建置：論文版（原始碼雜湊
`1bf4bc6d…`；`BUILD_INFO`）。Driver：`scripts/paper_extra_g02.sh --phases energylong`（`P1e-driver.log`）。

方法：`nvidia-smi --query-gpu=timestamp,power.draw.instant -lms 100`（`power.csv`；時間戳為 g02 的本地時間
UTC+8），`RUN_TIMES` 為求解器啟動與結束的時刻。每個變體跑兩種迭代數（`NAME-nN`）；**彙整只用長的那次**，在迴圈的
時間窗內平均功率：gncRhoSimple 依 log 的 `setup`、`loop` 秒數（自程式啟動起算），gncHisa 依每步的 `ExecutionTime`
（第一步結束到最後一步結束）。原本打算用兩種迭代數的能量差，但第一次執行要從磁碟讀網格，啟動時間差數倍
（`rho-r8-fast`：setup 10.0 s 對 2.8 s），差值失真，所以不用。

彙整：`python3 scripts/paper_extra_summary.py` → `summary.tsv`：

| 執行 | 迴圈迭代 | 迴圈 s | 每迭代 s | 平均功率 W | 每迭代 J |
|---|---|---|---|---|---|
| idle | | | | 34.6 | |
| rho-r8-fast（1,024,000 格） | 330 | 19.0 | 0.0577 | 231.8 | 13.3 |
| rho-r8-ref | 25 | 190.0 | 7.60 | 103.0 | 782 |
| m6-fast（ONERA M6） | 1049 | 25.6 | 0.0244 | 230.9 | 5.61 |
| m6-ref | 109 | 22.8 | 0.209 | 118.1 | 24.6 |
| finnerSST-fast（basicFinner SST） | 64 | 21.0 | 0.328 | 235.9 | 76.9 |

論文的能量表以這裡的平均功率乘以表 6、表 7 的每迭代時間（rho-r8-fast 330 迭代的後段迭代較便宜，每迭代 0.0577 s，
表 7 的前 30 迭代為 0.0691 s）。大網格（4.1M、16.4M 格）的迴圈功率見 `../N6e-gpu/summary.tsv`。
