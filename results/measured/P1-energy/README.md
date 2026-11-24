# P1-energy（g02，GPU 板功率與能量，第一批）

日期：2026-10-25。主機：**g02**，GPU3（L40），GPU3 連續 3 分鐘沒有其他程式之後才開始。建置：論文版（原始碼雜湊
`1bf4bc6d…`；`BUILD_INFO`）。Driver：`scripts/paper_extra_g02.sh --phases energy,memory`（`P1d-driver.log`；
第一次（P1b）的能量階段因 driver 的參數錯誤失敗，見 `../P1-nsys/README.md`，本紀錄是修正後重跑的）。

方法：每次執行期間以 `nvidia-smi --query-gpu=timestamp,power.draw.instant -lms 100` 取樣（`power.csv`），
`RUN_TIMES` 為求解器啟動與結束的時刻；能量為這段期間功率的梯形積分。閒置功率 `idle.power.csv`（30 s）。

| 執行 | 案例 | 迭代 |
|---|---|---|
| `rho-r8-ref`、`rho-r8-fast` | aerofoilNACA0012-r8（1,024,000 格；參考版 -pcg 案例） | 10、30 |
| `rho-r1-ref-conv`、`rho-r1-fast-conv` | aerofoilNACA0012（16,000 格）到收斂（residualControl） | 668、677 |
| `m6-ref`、`m6-fast` | ONERA M6 SA | 10、50 |

彙整：`python3 scripts/paper_extra_summary.py` → `summary.tsv`。閒置 34.8 W。

用途與限制：`*-conv` 是**到收斂的能量**（含啟動，與到收斂的時間同一個意義）：參考版 1,875 J（20.0 s），
`-fast` 268 J（2.89 s）。其餘執行太短（迴圈 1–3 s，而功率感測值每數百毫秒才更新），每迭代能量含啟動的比例太大，
**不用於每迭代能量**；每迭代能量見 `../P1-energylong`（只用長的那次執行：迴圈時間窗內的平均功率乘以每迭代時間；
沒有用兩種迭代數的差）。
