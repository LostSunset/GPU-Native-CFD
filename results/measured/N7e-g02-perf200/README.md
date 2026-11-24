# N7（g02 正式紀錄）：區塊快取與多 block 樹狀歸約

日期：2026-10-23。主機：**g02**，GPU3（L40）。commit `aad2e30946` 的建置（原始碼雜湊 `bd7e43f3…`，沒有未提交
的路徑）。驅動腳本 `N7e-g02-driver.sh`（`scripts/n7d_g02.sh`，`N7D_PREFIX=N7e-g02`），紀錄
`N7e-g02-driver.log`，8 段全部 exit 0。主機 CPU 有其他使用者的 MPI 作業（load average 約 104）。

## 逐位元（`../N7e-g02-verify/`）

預設執行的 L3：forwardStep、flatPlate SA、RAE2822 SA 50 步各 5/5，ONERA M6 SA 10 步 4/4，兩個程序相同。

## 每外迭代（`-profile` 平均，秒）

| 案例 | ref（逐位元） | fast | fast 對 ref | fast 在 N7d（`../N7d-g02-perf200/`） |
|---|---:|---:|---:|---:|
| flatPlate SA（8,000 格） | 0.0510 | 0.0081 | 6.3 倍 | 0.0097 |
| RAE2822 SA（23,552 格） | 0.0383 | 0.0047 | 8.2 倍 | 0.0068 |
| ONERA M6 SA（294,912 格，30 步，`../N7e-g02-perf-onera/`） | 0.231 | 0.032 | 7.2 倍 | 0.044 |

## ONERA M6 fast 的 nsys（`../N7e-g02-perf-onera/nsys-oneraM6SA-fast_*.csv`，30 步）

| kernel | 次數 | 平均 | 每外迭代 |
|---|---:|---:|---:|
| `kMatrixMul`（Jacobian 乘向量） | 300 | 1.00 ms | 10.0 ms |
| `kLusgsColorForward`＋`Reverse` | 1,080 | 0.23 ms | 8.3 ms |
| `kFaces`、`kDiag`（Jacobian 組裝） | 各 30 | 1.04、0.88 ms | 1.9 ms |
| `kTreeSumBatch` | 600 | 39 µs | 0.8 ms |

大網格是記憶體頻寬受限：每個內部面的 lower 與 upper 區塊共約 48 個 double，矩陣乘每次讀約 400 MB，L40 的
頻寬下理想約 0.47 ms，實測約為一半的效率。下一步：Jacobian 以 FP32 儲存、運算仍用 FP64（混合精度）。
小網格（RAE2822 每外迭代約 770 次啟動）則是 GMRES 一次重啟的 CUDA Graph。

nsys 的原始檔留在 g02 的 `/ssd/sclin/gpu-native-cfd/runs/N7e-g02-perf{200,-onera}/`。
