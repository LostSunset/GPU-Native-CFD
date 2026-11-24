# N7（g02 正式紀錄）：層級排程模式與效能版選項的時間與收斂

日期：2026-10-23。主機：**g02**，GPU3（L40，sm_89）。commit `50b77f0133` 的建置（原始碼雜湊
`cb2d509e…`，沒有未提交的路徑）。驅動腳本與紀錄：`N7-g02-driver.{sh,log}`，8 段全部 exit 0。
量測期間主機 CPU 有其他使用者的 100 程序 MPI 作業（load average 約 112），GPU3 沒有其他程序；
時間包含主機端啟動 kernel 的開銷，CPU 忙碌時可能偏慢。

同一個建置的紀錄：

| 紀錄 | 內容 | 結果 |
|---|---|---|
| `../N7-g02-powexp/` | glibc pow／exp 移植 | 0 差異 |
| `../N7-g02-verify-m{0,1,2}/` | 層級模式 0／1／2 的 L3（forwardStep、flatPlate SA、RAE2822 SA 50 步；ONERA M6 SA 10 步） | 三種模式 4 案例全部逐位元，兩個程序相同 |
| 本目錄 | 各變體 200 步（`scripts/variant_compare.sh`，`gncHisa -profile`） | 下表 |
| `../N7-g02-perf-onera/` | ONERA M6 SA（294,912 格）各變體 30 步 | 下表 |
| `../N7-g02-conv/` | 參考版、多色 LU-SGS、fast 各 2000 步 | 收斂一節 |

變體（同一個執行檔，以環境變數選擇）：m0／m1／m2 = `GNC_COOP_LEVELS=0／1／2`（逐位元）；
lusgsColor = `GNC_LUSGS=color`；sgsColor = `GNC_SGS=color`；tree = `GNC_REDUCE=tree`；
fast = 三者合用（不逐位元）。三個案例都是 2 色（層數 flatPlate 179、RAE2822 248、ONERA M6 175）。

## 每外迭代的牆鐘時間（`-profile` 的平均，秒）

| 變體 | flatPlate SA（8,000 格） | RAE2822 SA（23,552 格） | ONERA M6 SA（294,912 格，30 步） |
|---|---:|---:|---:|
| m0（逐層啟動） | 0.0640 | 0.0470 | 0.232 |
| m1（cooperative） | 0.0830 | 0.0509 | 0.280 |
| m2（單一 block） | **0.0523** | **0.0395** | 0.403 |
| lusgsColor | 0.0253 | 0.0272 | 0.183 |
| sgsColor | 0.0623 | 0.0453 | 0.230 |
| tree | 0.0567 | 0.0347 | 0.093 |
| **fast** | **0.0100** | **0.0074** | **0.044** |
| fast 對 m0 | 6.4 倍 | 6.3 倍 | 5.2 倍 |

- m0、m1、m2 的最終場完全相同（`*/fielddiff.m0-m{1,2}.txt` 全為 0）。單一 block 在每層格數少的網格最快，
  ONERA M6 每層約 1,700 格時最慢；cooperative 在三個案例都比逐層啟動慢。據此把預設改為自動選擇
  （每層平均格數 ≤ 1024 用單一 block，否則逐層啟動；`src/core/coopLevels.cuh`）。
- ONERA M6 上 tree 的效果最大（0.232 → 0.093 秒）：循序歸約每個元素約 27 奈秒，29.5 萬格一次就約 8 毫秒；
  SA 的 smoothSolver 也從 0.028 降到 0.004 秒。

## 收斂（`../N7-g02-conv/`，2000 步）

| 案例 | 變體 | 每外迭代 | 殘差到 1e-3 | 殘差到 1e-4 | 2000 步的 rho 殘差 |
|---|---|---:|---|---|---:|
| flatPlate SA | ref | 0.0571 s | 第 109 步（6.25 s） | 第 437 步（25.1 s） | 1.41e-5 |
| | lusgsColor | 0.0197 s | 第 110 步（2.19 s） | 第 448 步（8.91 s） | 1.45e-5 |
| | fast | 0.0080 s | 第 110 步（0.90 s） | 第 447 步（3.66 s） | 1.46e-5 |
| RAE2822 SA | ref | 0.0473 s | — | — | 4.40e-3 |
| | fast | 0.0072 s | — | — | 4.31e-3 |

- flatPlate：到 1e-4 的時間 25.1 → 3.66 秒（6.9 倍），外迭代數多 2%。2000 步時 fast 與 ref 的最終場 L2
  相對差 ≤ 1.2e-5（`flatPlateSA/fielddiff.ref-fast.txt`），與殘差停住的水準（約 1e-5）一致。
- RAE2822：參考版本身的殘差停在約 4.4e-3（震波附近的限制器），固定步數的場差約 1e-3，不代表收斂解的差；
  以力係數比較（N4c）。
- ONERA M6 30 步時 fast 與 m0 的 L2 相對差 2e-6–1.4e-4（還在暫態）。
