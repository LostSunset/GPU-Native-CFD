# P1-mfcheck（g02，以論文版建置重量 modifiedFinner SST）

日期：2026-10-26。主機：**g02**，GPU3（L40），等前一個 GPU3 工作結束後才開始。
建置：論文版（`/ssd/sclin/gpu-native-cfd/paper`，原始碼雜湊 `1bf4bc6d…`；`BUILD_INFO`）。
Driver：`scripts/paper_extra_g02.sh --phases mfcheck`（`driver.log`），內部是 `scripts/variant_compare.sh`。

目的：N8 的 modifiedFinner 紀錄（`MF-g02-sst`、`MF-g02-gncfast-sst`，稿件的 Table 6）在開發用的暫存目錄
（`/ssd/sclin/gpu-native-cfd/GPU-Native-CFD`）建置。該目錄以 tar 同步、保留原始碼的修改時間，CMake 的增量建置
可能留下舊內容的目的檔（N-042）。N8 的原始碼雜湊與論文版相同，但 N-028 的改動對 gncHisa 是位元中性的，
殘差分不出新舊目的檔，所以用從頭編譯、之後沒改過原始碼的論文版建置重量一次。

方法：與 N8 相同的案例（`modifiedFinner-seq8/simulation`）、5 個外迭代，參考版與 `-fast`
（`GNC_LUSGS=color GNC_SGS=color GNC_REDUCE=tree GNC_JAC=auto`）各一次；每步時間取 gncHisa 的 profile 平均。
`ref/`、`fast/` 的場（各 775 MB）只留在 g02，`fielddiff.ref-fast.txt` 是兩者第 5 步的差。

結果：

| | N8（開發目錄的建置） | P1-mfcheck（論文版建置） | 差 |
|---|---|---|---|
| 參考版，每步（s） | 2.78939（`MF-g02-sst`） | 2.77958 | −0.35% |
| `-fast`，每步（s） | 0.590761（`MF-g02-gncfast-sst`） | 0.587114 | −0.62% |
| 參考版的殘差（log 的 11 行） | | 與 N8 的 `log.gnc` 逐字相同 | |
| `-fast` 的 `history.fast.tsv` | | 與 N8 逐位元相同 | |

結論：N8 的計時與數值都成立，稿件的數字（參考版 2.79 s、`-fast` 0.591 s、比 32 核快 7.6 倍）不改。
兩次的差在同一主機不同時段的量測變動之內（主機負載 load average 約 104，N8 時約 121–128）。

說明：gncHisa 執行檔的 SHA-256 與 N8 不同（`f213dcb0…` 對 `ec3b4068…`）；兩者的原始碼雜湊相同，但建置目錄
不同，路徑字串會嵌進執行檔，而且建置本身不能逐位元重現。論文版的執行檔與其他 `P1-*` 紀錄相同（`f213dcb0…`）。
