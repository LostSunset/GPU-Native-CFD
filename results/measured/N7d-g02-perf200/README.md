# N7（g02 紀錄）：裝置端 GMRES 的驗證、時間與 nsys 統計

日期：2026-10-23。主機：**g02**，GPU3（L40）。commit `d91ec68881`（裝置端 GMRES `906c840621` 之後只加了
驅動腳本）的建置，原始碼雜湊 `bce1f64c…`，沒有未提交的路徑。驅動腳本 `scripts/n7d_g02.sh`，紀錄
`N7d-g02-driver.log`，6 段全部 exit 0。主機 CPU 有其他使用者的 MPI 作業（load average 約 104）。

## 逐位元（`../N7d-g02-verify/`）

預設執行（自動層級模式）的 L3：forwardStep、flatPlate SA、RAE2822 SA 50 步各 5/5，ONERA M6 SA 10 步 4/4，
兩個程序相同。

## 每外迭代的時間（`-profile` 平均，秒；ONERA M6 為 `../N7d-g02-perf-onera/`，30 步）

| 案例 | 變體 | 總計 | GMRES | 湍流 | 裝置端 GMRES 之前（`../N7-g02-perf200/` 等） |
|---|---|---:|---:|---:|---|
| flatPlate SA | ref | 0.0515 | 0.0470 | 0.0022 | 0.0523（m2） |
| | fast | 0.0097 | 0.0062 | 0.0010 | 0.0100 |
| RAE2822 SA | ref | 0.0391 | 0.0333 | 0.0034 | 0.0395（m2） |
| | fast | 0.0068 | 0.0037 | 0.0009 | 0.0074 |
| ONERA M6 SA | ref | 0.231 | 0.195 | 0.028 | 0.232（m0） |
| | fast | 0.044 | 0.033 | 0.003 | 0.044 |

g02 的主機同步本來就快，裝置端 GMRES 只省下約 3–8%（WSL 的同步較慢，省下約一半）。

## nsys（RAE2822 SA fast，200 步；`nsys-rae2822SA-fast_*.csv`）

| 項目 | 次數 | 平均 | 每外迭代 |
|---|---:|---:|---:|
| `kTreeSumBatch`（樹狀歸約，一個工作一個 block） | 7,330 | 49.8 µs | 1.8 ms（GPU 上最大的一項） |
| `kMatrixMul` | 3,665 | 23.1 µs | 0.42 ms |
| `kLusgsColorForward`＋`Reverse` | 13,860 | 9.9 µs | 0.68 ms |
| `cudaLaunchKernel`（主機） | 154,048 | 3.3 µs | 770 次、2.6 ms |
| `cudaMallocAsync`＋`cudaFreeAsync`（主機） | 各 150,818 | 1.75＋0.81 µs | 750 對、1.9 ms |
| `cudaMemset`（主機） | 44,438 | 3.4 µs | 0.76 ms |
| `cudaMemcpy`（主機，含等待） | 18,006 | 34.8 µs | 90 次（D2H 約 20 次） |

主機端的 API 時間（約 5 毫秒／外迭代）與 GPU 的 kernel 時間（約 4.8 毫秒）相當，小網格上受主機端的啟動與
配置限制。下一步：樹狀歸約改為每個工作多個 block；DeviceArray 加主機端的區塊快取；減少啟動與同步次數。
nsys 的原始檔（`.nsys-rep` 18 MB、`.sqlite` 42 MB）留在 g02 的 `/ssd/sclin/gpu-native-cfd/runs/N7d-g02-perf200/`。
