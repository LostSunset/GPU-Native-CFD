# HiSA 的 CPU 計時對 gncHisa（g02，主機忙碌時的參考值）

日期：2026-10-23。主機：**g02**，2 × AMD EPYC 9684X（192 核）＋ L40（GPU3）。gncHisa 為 commit
`3c1a7cc114` 的建置（N7f）；腳本為 commit `0c33b74813` 的 `scripts/cpu_baseline_g02.sh`（驅動腳本與紀錄：
`CPU2-g02-driver.{sh,log}`），6 段全部 exit 0。

**限制**：量測期間主機上有其他使用者的 `mpirun -np 100 hisa` 作業與 3 個 python 程序（load average 103–111），
CPU 的記憶體頻寬與快取是共用的，HiSA 的時間可能偏慢；MPI 以 `--bind-to none` 執行（不與其他作業綁到同樣的
核心），metis 分解（Atlas runtime 沒有 scotch；第一次以 scotch 的量測 `CPU-g02-*` 全部失敗，未收錄）。
論文以這組在主機負載下量得的值作為參考分母（限制見論文 Section 5 與 Section 8），沒有在主機閒置時重量，也沒有量
64、96、192 核。

案例為各自的穩態變體（perf_compare.sh：steadyState、functions 清空），HiSA 與 gncHisa 跑同樣的步數。

## 迴圈內每步的時間（秒）

HiSA：log 中第一步到最後一步的 `ExecutionTime` 之差除以步數減一（不含啟動；`SUMMARY.tsv` 的 wall 欄是整數秒的
ClockTime，小案例不準）。gncHisa：`-profile` 的每外迭代平均（參考版在 `*/profile.txt`，fast 在
`../CPU2-g02-gncfast-*/`）。

| 案例 | 格數 | 步數 | HiSA 1 核 | HiSA 8 核 | HiSA 32 核 | gncHisa 參考版（逐位元） | gncHisa `-fast` |
|---|---:|---:|---:|---:|---:|---:|---:|
| flatPlate SA | 8,000 | 200 | 0.0735 | 0.0135 | 0.00628 | 0.0511 | 0.00770 |
| RAE2822 SA | 23,552 | 200 | 0.171 | 0.0226 | 0.00884 | 0.0383 | 0.00471 |
| ONERA M6 SA | 294,912 | 20 | 3.75 | 0.621 | 0.176 | 0.232 | 0.0228 |
| basicFinner（層流變體） | 2,179,881 | 5 | 48.9 | 7.09 | 2.69 | 1.50 | 0.266 |

`-fast` 相對於 HiSA：

| 案例 | 對 1 核 | 對 8 核 | 對 32 核 |
|---|---:|---:|---:|
| flatPlate SA | 9.5 倍 | 1.8 倍 | 0.82 倍（CPU 較快） |
| RAE2822 SA | 36 倍 | 4.8 倍 | 1.9 倍 |
| ONERA M6 SA | 164 倍 | 27 倍 | 7.7 倍 |
| basicFinner 層流 | 184 倍 | 27 倍 | 10.1 倍 |

- 網格越大，單張 L40 的優勢越大：29.5 萬格與 218 萬格時，一張 L40 約等於 32 核 HiSA 的 8–10 倍。
- 8,000 格的 flatPlate 太小，GPU 受啟動延遲限制，32 核的 HiSA 反而較快。
- 參考版（與 HiSA 逐位元）在大網格也比 32 核的 HiSA 快（basicFinner 1.50 對 2.69 秒）。
- basicFinner 只跑 5 步、ONERA M6 只跑 20 步，HiSA 的 ExecutionTime 解析度 0.01 秒，對每步的影響小於 1%。
