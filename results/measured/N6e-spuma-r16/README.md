# N6e-spuma-r16、N6e-spuma-r32（g02，SPUMA 上的 rhoSimpleFoam：4,096,000 與 16,384,000 格）

日期：2026-10-26。主機：**g02**，GPU3（L40，PCIe），GPU3 連續 3 分鐘沒有其他程式之後才開始。Driver：
`scripts/n6e_large_g02.sh --steps spuma`（log 在 `../N6e-gpu/N6e-spuma-driver.log`），各執行為
`scripts/rhosimple_perf.sh --variants spuma`。SPUMA 0.3-v2512（commit 5916a466，NVHPC 26.9），與 `N6-perf-*` 的
SPUMA 同一個建置；案例為教學設定（p 為 GAMG），只改迭代數。記憶體池 `fixedSizeMemoryPool`：r16 16 GB、
r32 40 GB（r1–r8 為 8 GB）。每迭代時間以牆鐘（ClockTime 差）為準（SPUMA 等待 GPU 的時間不計入 CPU 時間）。

| 網格 | 迭代 | 每迭代牆鐘 s | 每迭代 CPU s | p 的 GAMG 平均循環 | 全程牆鐘 |
|---|---|---|---|---|---|
| r16 | 3 | 940.5 | 376.4 | 355.7 | 58:40 |
| r32 | 3 | 3807.0 | 2160.6 | 705.7 | 4:02:58 |

`log.time`（`/usr/bin/time -v`）：r16 major page faults 2.32×10^7、CPU 使用率 40%、最大 RSS 3.9 GB；
r32 major page faults 1.12×10^8、CPU 使用率 57%、最大 RSS 13.8 GB。`../N6e-gpu/summary-spuma.tsv`：GPU 記憶體峰值
r16 4,256 MiB、r32 16,040 MiB（整張 GPU；池為 16、40 GB）；平均板功率 83.1、85.2 W（閒置 34.6 W）。

與 rhoSimpleFoam（`../N6e-perf-r16`、`-r32`）比較：單核的 0.105 與 0.213 倍；COE（對 32 核）0.18 與 0.25。
SPUMA 作者在 A100 上報告每張 GPU 超過 1,000 萬格時相當於 200–300 核；本紀錄是 L40（PCIe）上未調整的 0.3 版，
不檢驗作者在其平台上的結果。

`N6e-spuma-r32/README.md` 指向本檔。
