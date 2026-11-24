# N7（g02 正式紀錄）：FP32 Jacobian（`GNC_JAC=float`／`auto`，`gncHisa -fast`）

日期：2026-10-23。主機：**g02**，GPU3（L40）。commit `3c1a7cc114` 的建置（原始碼雜湊 `26de2297…`，沒有未提交
的路徑，`compiler_warnings=0`）。驅動腳本與紀錄：`N7f-g02-driver.{sh,log}`，6 段全部 exit 0。主機 CPU 有
其他使用者的 MPI 作業。

變體：ref（逐位元）；fast = `GNC_LUSGS=color GNC_SGS=color GNC_REDUCE=tree`；fast32 = fast 加 `GNC_JAC=float`；
fastauto = fast 加 `GNC_JAC=auto`（即 `gncHisa -fast`，FP64 的 Jacobian 大於 L2 的 96 MB 時用 FP32）。

## 逐位元（`../N7f-g02-verify/`）

預設執行的 L3：forwardStep、flatPlate SA、RAE2822 SA 50 步各 5/5，ONERA M6 SA 10 步 4/4（JacView 改為
儲存型別的樣板之後）。

## 每外迭代（`-profile` 平均，毫秒）

| 案例（Jacobian 的 FP64 大小） | ref | fast | fast32 | fastauto | fastauto 對 ref |
|---|---:|---:|---:|---:|---:|
| flatPlate SA，8,000 格（約 6 MB） | 50.8 | 8.04 | 8.92 | 7.88 | 6.4 倍 |
| RAE2822 SA，23,552 格（約 18 MB） | 38.3 | 4.79 | 5.29 | 4.87 | 7.9 倍 |
| ONERA M6 SA，294,912 格（約 400 MB；本目錄，30 步） | 230.6 | 31.8 | 23.0 | 23.1 | 10.0 倍 |

小網格與 RAE2822 見 `../N7f-g02-perf200/`。fastauto 在小網格選 FP64、在 ONERA M6 選 FP32，三個案例都是
最快的一種。

## 收斂（FP32 Jacobian 的影響）

- flatPlate SA 2000 步（`../N7f-g02-conv-flat/`）：fast 與 fast32 的 GMRES 次數 6406 對 6408；殘差到 1e-3、1e-4
  都在第 110、447 步；最終場的 L2 相對差 ≤ 1.6e-8。
- RAE2822 SA 4000 步的力係數（`../N7f-g02-coeffs/rae2822SA/coeffs.txt`）：後段平均 Cd 差 3e-9、Cl 差 7e-8、
  CmPitch 差 6e-9，遠小於振盪幅度（5e-5）。
