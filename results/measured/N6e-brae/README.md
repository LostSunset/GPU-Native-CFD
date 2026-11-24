# N6e-brae（g02，brae 在 4,096,000 與 16,384,000 格）

日期：2026-10-25。主機：**g02**，GPU3（L40）。brae commit `8ecfc74a`（`RUN_INFO`），與 `BRAE-perf` 相同的建置。
Driver：`scripts/n6e_large_g02.sh` 呼叫 `BRAE_MESHES="16 32" BRAE_CONV=0 REPEAT=3 scripts/brae_g02.sh`（本目錄的
`brae_g02.sh` 是實際執行的那一份）。方法同 `BRAE-perf`：每個網格跑 n1、n2 兩種迭代數，每迭代時間為牆鐘差除以
迭代數差，三次取中位數。

| 網格 | n1、n2 | 三次 | 中位數（每迭代 s） | p 的平均迭代數 |
|---|---|---|---|---|
| r16（4,096,000 格） | 5、35 | 0.431、0.514、0.501 | 0.501 | 102.3 |
| r32（16,384,000 格） | 3、23 | — | — | — |

**r32 無法執行**：六次執行都在建構時以 `brae cuda: pool cudaMalloc: out of memory` 結束（`r32/k*/n*/log.err`，
`rc=1`）；L40 可用 46,068 MiB。`SUMMARY.tsv` 中 r32 的三個數值是兩次失敗執行的牆鐘差，**沒有意義**，不使用。
同一網格 gncRhoSimple -fast 的峰值為 30,618 MiB（`../N6e-gpu/summary.tsv`）。
