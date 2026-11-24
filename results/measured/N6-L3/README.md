# N6：gncRhoSimple 對 rhoSimpleFoam 的 L2、L3（g02）與 gncHisa 的回歸

日期：2026-10-23。主機：**g02**，GPU3（L40）。驅動：`scripts/n6_all_g02.sh`（前段 `n6_g02.sh`；紀錄
`N6-all-g02-driver.log`）。建置的原始碼雜湊 `0f07d313…0611c0`（`BUILD_INFO`，0 個警告），與提交 N6 的
commit 相同。

## gncRhoSimple 的 L2、L3（`scripts/rhosimple_verify.sh`）

案例：aerofoilNACA0012（rhoSimpleFoam 的教學案例，16,000 格 2D，kOmegaSST、limitTemperature、
freestreamVelocity／freestreamPressure），p 的線性求解器由 GAMG 改為 PCG＋DIC 的驗證變體（N-023；
`/ssd/sclin/gpu-native-cfd/cases/aerofoilNACA0012-pcg`）。參考端是 `tools/ref/gncRhoSimpleRef`
（rhoSimpleFoam 的主程式與 createFields、UEqn、EEqn、pEqn 原樣 include，只加傾印點）。

| 項目 | 結果 |
|---|---|
| L2 傾印（初始化，與第 1、2、10、100、668 迭代的 U、E、p、turb 四個階段；p T U e psi alpha mu rho phi k omega nut alphat 的內部與邊界值，UEqn 的 diag／upper／lower／source／邊界係數，rAU、HbyA、phiHbyA、rhorAUf） | **21/21 逐位元相同**（`aerofoilNACA0012/compare.*.txt`） |
| L3：寫出的時間目錄（50、100、…、650 與收斂時的 668）的 9 個場檔 | **126/126 逐位元相同**（`fielddiff.txt`） |
| 收斂（residualControl 1e-4） | 兩邊都在第 668 迭代結束並寫出 |
| 兩次 GPU 執行 | 場檔逐位元組相同 |

## gncHisa 的回歸（共用的邊界條件、純量矩陣、kOmegaSST、循序和有改動）

| 紀錄 | 內容 | 結果 |
|---|---|---|
| `N6-reg-SST-L2/` | nacaAirfoil SST（meshWave），第 1、2、3、10 步 | 5/5 |
| `N6-reg-SST-L3/` | nacaAirfoil SST 200 步 | 7/7 |
| `N6-reg-SST-app/` | gncHisa 寫出的場檔（30、60 步） | 16/16 |
| `N6-reg-SA-L2/` | flatPlate、RAE2822 SA | 5/5、5/5 |
| `N6-reg-SA-L3/` | RAE2822 SA 200 步 | 7/7 |
| `N6-reg-L3/` | shockTube、forwardStep（無黏）、nacaAirfoil 層流 150 步 | 7/7、7/7、6/6 |

全部逐位元。這個建置包含參考次序版的兩項加速（N-024）：循序和在元素 ≥ 1024 時傳回主機以同樣
次序相加、DILU／DIC 的層級掃描放在單一 block；上面的結果確認兩者都不改變任何位元。

效能的紀錄見 `../N6-perf-conv/README.md`。
