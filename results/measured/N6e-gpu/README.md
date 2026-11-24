# N6e-gpu（g02，N6e 期間 GPU3 的記憶體與功率）

日期：2026-10-25／26。`scripts/n6e_large_g02.sh` 在量測期間以
`nvidia-smi -i 3 --query-gpu=timestamp,memory.used,power.draw.instant -lms 500` 取樣：

| 檔案 | 期間 | driver 的 log |
|---|---|---|
| `N6e-gpu.csv` | 網格產生、`N6e-perf-r16`、`N6e-perf-r32`、`N6e-brae` | `N6e-driver.log` |
| `N6e-gpu-spuma.csv` | `P1-nsys-large`、`N6e-spuma-r16`、`N6e-spuma-r32` | `N6e-spuma-driver.log` |

時間戳是 g02 的本地時間（Asia/Taipei，UTC+8）。`memory.used` 是整張 GPU 的用量（閒置時 73 MiB）。

彙整：`python3 scripts/paper_extra_summary.py` → `summary.tsv`、`summary-spuma.tsv`：每個 `rhosimple_perf.sh` 執行
（log 中的 `start_epoch`、`end_epoch`）期間的記憶體峰值、平均功率與能量（含啟動與讀網格）。

`N6e-driver.log` 的最後一段：brae 之後 driver 卡在 SPUMA 的 watchdog（`wd=$(spuma_watchdog)` 等待背景的
`sleep`），2026-10-26 00:47Z 依 PID 停止（driver、取樣器與 watchdog）；之後的 SPUMA 由 `--steps spuma` 另外執行。
