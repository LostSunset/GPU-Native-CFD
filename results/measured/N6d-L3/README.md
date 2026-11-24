# N6d：N-028 之後的逐位元確認（g02）

日期：2026-10-25（UTC）。主機：**g02**，GPU3（L40）。驅動：`scripts/n6d_g02.sh`（紀錄 `N6d-g02-driver.log`）與
`scripts/n6d_sumcheck_g02.sh`（紀錄 `../N6d-sumcheck/N6d-sumcheck-driver.log`）。建置的原始碼雜湊
`1bf4bc6d…99372`（`BUILD_INFO`，0 個警告；與本機原始碼相同）：小網格的啟動與同步（N-028）——PCG／PBiCGStab
的純量留在裝置端（PCG 每迭代 1 次同步、PBiCGStab 2 次）、多項合併的單次啟動歸約、批次的逐元素運算
（`ArrayBatch`）、多 patch 的邊界條件與 snGrad kernel、合併的 min／max；gncRhoSimple `-fast` 另有 AMG 粗層的
Jacobi 平滑、omega 1.6、單一 block 門檻 1024、兩色 DILU／DIC 與 Welsh–Powell 著色。參考次序版的程式沒有改動。

| 紀錄 | 內容 | 結果 |
|---|---|---|
| `N6d-id-hisa/` | gncHisa `-fast` 20 步：N6c 的執行檔（`build-N6c`）對本建置，寫出的場逐檔 `cmp`（`scripts/fast_identity.sh`） | nacaAirfoil SST 8/8、RAE2822 SA 7/7 **相同** |
| `N6d-id-rho/` | gncRhoSimple `-fast` 20 步，新的演算法關閉（`GNC_COLOR_BIPARTITE=0 GNC_AMG_JAC=0 GNC_AMG_OMEGA=1.8 GNC_AMG_BLOCK=4096 GNC_AMG_WP=0`）：N6c 對本建置 | aerofoilNACA0012 r1 9/9、r2 9/9 **相同**——啟動與同步的改動不改變任何位元 |
| `N6d-sumcheck/` | `gncSumBench`（同 N6c），`GNC_SUMTREE_CHECK=1` | 全部相同（rc 0） |
| `N6d-sumcheck-hisa/`、`-rho/` | 檢查模式（每次 `sumTreeOp`、`sumTreeOps`、`sumTreeToDevice`、`sumTreeOpsToDevice` 都以多次啟動版重算，不同就中止）下跑 gncHisa `-fast`（nacaAirfoil SST、RAE2822 SA）與 gncRhoSimple `-fast`（r1、r4）各 20 步，同一執行檔兩次 | 全部 rc 0，兩次的場相同 |
| `N6d-L3/` | gncRhoSimple 對 rhoSimpleFoam（aerofoilNACA0012，p 為 PCG＋DIC）：傾印與寫出的場 | **21/21、126/126 逐位元**，第 668 迭代收斂；兩次 GPU 執行相同 |
| `N6d-reg-SST-L3/` | gncHisa nacaAirfoil SST 200 步 | 7/7 |
| `N6d-reg-SA-L3/` | gncHisa RAE2822 SA 200 步 | 7/7 |

效能的紀錄見 `../N6d-perf-conv/README.md`。
