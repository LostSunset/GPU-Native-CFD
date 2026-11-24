# basicFinner（3D、kOmegaSST、218 萬格）到 1000 步：gncHisa 參考版、-fast 與 HiSA 32 核（g02）

日期：2026-10-23。主機：**g02**，GPU3（L40）。gncHisa 為 commit `5bd58fc323` 的建置（含能量限界）；
驅動：`N7g-queue-driver.sh`（`scripts/n7g_queue_g02.sh` 的最後一段，紀錄 `FIN-g02-conv2-driver.log`）。
主機 CPU 有其他使用者的 MPI 作業。

案例是原本的設定（Poisson 牆距離、案例的 endTime 1000 步、`forces` function：body fin1 fin2，CofR
(-0.165 0 0)），穩態變體（`variant_compare.sh --forces`）。HiSA 32 核為 `../FIN-g02-hisa32/`
（`perf_compare.sh --forces`，metis 分解）。第一次的 gncHisa run（`../FIN-g02-conv/`）在第 492 步因能量
限界未實作而停止；HiSA 32 核在第 494–716 步限界 223 次。

## 結果

| 執行 | 總時間 | 每步（迴圈內） | 第 1000 步 rho 殘差 | GMRES 次數 |
|---|---:|---:|---:|---:|
| gncHisa 參考版（1×L40） | 1930 s | 1.91 s | 9.32e-4 | 1000 |
| gncHisa `-fast`（1×L40） | 398 s | 0.381 s | 9.31e-4 | 1317 |
| HiSA 32 核 MPI | 2302 s | 2.30 s | 9.28e-4 | — |

力（N，後 25% 的平均與標準差，`basicFinnerSST/coeffs-with-hisa32.txt`）：

| 分量 | 參考版 | `-fast` | HiSA 32 核 | 後段的標準差 |
|---|---:|---:|---:|---:|
| Fx | -232.087 | -232.124（差 -0.037） | -231.290（差 0.80） | 1.3 |
| Fy | -9.155 | -9.180（差 -0.026） | -9.196（差 -0.041） | 0.6 |
| Fz | -248.438 | -248.445（差 -0.007） | -248.601（差 -0.16） | 0.7 |

- `-fast` 與參考版的力差不到後段標準差的 5%；第 1000 步的 rho 殘差 9.31e-4 對 9.32e-4。
- HiSA 32 核（MPI 分解改變 LU-SGS 與 GMRES，牆距離為 GAMG）與參考版的差也在振盪之內（Fx 0.34%）。
- 1000 步時殘差約 9e-4，案例在 1000 步內沒有降到 pseudoTol（1e-6）；力已在穩定的振盪中。最終場的局部
  差（`fielddiff.ref-fast.txt`）來自沒有收斂的區域。
- 時間：`-fast` 對 HiSA 32 核快 6.0 倍（每步 0.381 對 2.30 s；後段的 GMRES 次數比前 5 步多，見
  `../SSTP-g02-finner/` 的 0.316 s）。
