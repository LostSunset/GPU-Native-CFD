# ONERA M6：HiSA v2606 的 3D 黏性應力轉置問題對力係數的影響（g02）

日期：2026-10-23。主機：**g02**，GPU3（L40）。commit `c1d06b28e9` 的建置（原始碼雜湊 `6971ed90…`，
`compiler_warnings=0`）。驅動腳本與紀錄：`M6-compat-driver.{sh,log}`。案例同 `../M6-g02-conv/`（翼面 forceCoeffs
的衍生案例），`gncHisa -fast` 各跑 2000 步。

| 變體 | 設定 | 說明 |
|---|---|---|
| fast | （預設的相容設定） | 照 HiSA v2606 的實際行為：`T(tmp<volTensorField>)` 原地轉置使內部場偶數格的 zx 未轉置（N-013，`src/core/foamCompat.hpp`） |
| fastFixed | `GNC_FOAM_COMPAT=off`（同 `gncHisa -no-foam-compat`） | 原始碼的正確結果（tauMC 的轉置正確；stabilise 的引數照原意） |

## 力係數（最後 500 步平均，`oneraM6SA/coeffs.txt`）

| 量 | fast（HiSA 相容） | fastFixed（正確） | 差 | 後段的振盪（標準差） |
|---|---:|---:|---:|---:|
| Cl | 0.2627295 | 0.2627381 | 8.6e-6 | 1.6e-5 |
| Cd | 0.01662465 | 0.01662429 | -3.6e-7 | 4.6e-6 |
| CmPitch | -0.1855018 | -0.1855106 | -8.9e-6 | 1.6e-5 |

- 差異都小於力係數本身的振盪；殘差歷程幾乎相同（2000 步時 rho 殘差 1.25e-3 對 1.22e-3，GMRES 次數 2196 對 2198）。
- 最終場的 L2 相對差 8e-5（p）到 8e-4（nuTilda），與 `-fast` 對參考版的差同一量級。
- 結論：在這個高雷諾數的穿音速翼上，HiSA v2606 的轉置問題對工程量的影響可以忽略；論文的物理結果可以用
  HiSA 相容的版本，並註明這個問題與本測試。
