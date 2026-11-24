# ONERA M6（3D、Spalart–Allmaras）：力係數、到 2000 步的比較與 HiSA 32 核計時（g02）

日期：2026-10-23。主機：**g02**，2 × AMD EPYC 9684X ＋ L40（GPU3）。gncHisa 為 commit `3c1a7cc114` 的建置
（原始碼雜湊 `26de2297…`，N7f）；腳本為 commit `6ae28b0ff6` 的 `scripts/onera_conv_g02.sh`（驅動腳本與紀錄：
`M6-g02-driver.{sh,log}`）。主機 CPU 有其他使用者的 100 程序 MPI 作業（load average 約 104），HiSA 的時間只作參考。

## 案例

HiSA examples/oneraM6（294,912 個六面體格，wing／symm／farfield）加上翼面的 forceCoeffs
（`scripts/make_onera_forces_case.sh`，`DERIVED.case`）：攻角 3.06°、馬赫 0.8395（取自 0/U、0/T），
Aref 0.7532 m²（半翼）、lRef 0.64607 m（平均氣動弦長）、CofR 原點、升力方向 +y、展向 z。穩態變體，
依案例的 endTime 2000 步（pseudoTol 1e-6 在 2000 步內未達到）。

## 力係數的逐位元（`../M6-g02-L3-forces/`）

10 步的 L3（第 1 2 10 步與結束狀態）4/4 逐位元：全部狀態場、forceCoeffs 的 48 個係數 result、逐面的 force、
moment 場都與 HiSA 相同（3D 黏性力的 `devTwoSymm(grad(U))` 不經過 `T()`，不受 N-013 的轉置問題影響）。

## 2000 步（`oneraM6SA/`）

| 執行 | 總時間 | 每步 | Cl（最後 500 步平均） | Cd（最後 500 步平均） | CmPitch |
|---|---:|---:|---:|---:|---:|
| gncHisa 參考版（與 HiSA 逐位元，1×L40） | 464 s | 0.231 s | 0.2627329 | 0.01662714 | -0.1855047 |
| gncHisa `-fast`（1×L40，FP32 Jacobian） | 51.0 s | 0.0239 s | 0.2627295 | 0.01662465 | -0.1855018 |
| HiSA 32 核 MPI（`../M6-g02-hisa32/`，metis） | 376 s（含 mpirun 啟動的 shell 時間為 381 s） | 0.188 s | — | — | — |

- `-fast` 對參考版：Cl 差 3.4e-6、Cd 差 2.5e-6（0.025 drag count）、CmPitch 差 2.9e-6；GMRES 次數 2196 對 2000。
- 2000 步時兩者的 rho 殘差約 1.25e-3（案例在 2000 步內沒有降到 pseudoTol）；力係數已穩定（參考版後段標準差
  Cl 1.7e-6、Cd 2.8e-6）。最終場的 L2 相對差 9e-5（p）到 4e-3（nut），局部最大差在殘差未收斂的區域。
- 時間：`-fast` 比 HiSA 32 核快 7.4 倍、比參考版快 9.1 倍；HiSA 1 核每步約 3.75 秒（`../CPU2-g02-onera/`），
  2000 步約 2.1 小時，`-fast` 約 160 倍。

## 與實驗的壓力分布比較（`cp/`；論文 Fig. 9，`docs/paper/figures/fig4_onera_cp.png`，共同作者包中為 `02_figures/fig09_onera_cp.png`）

實驗：Schmitt & Charpin（AGARD AR-138），`results/external/oneraM6-exp-schmitt-charpin/`（2026-10-23 經使用者同意
自 NASA NPARC 下載）。計算：2000 步的翼面 p（參考版、`-fast`；`fastFixed` 取自 `../M6-g02-compat/`），以
`scripts/plots/onera_cp_extract.py` 在 z = η·1.1963 m 切出（面心值；前後緣為交點的最小、最大 x），
`scripts/plots/onera_cp_rms.py` 內插到實驗點（`cp/rms_vs_experiment.txt`）。

| η | 0.20 | 0.44 | 0.65 | 0.80 | 0.90 | 0.95 | 0.99 | 全部（271 點） |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cp 的均方根差（參考版） | 0.063 | 0.047 | 0.045 | 0.047 | 0.097 | 0.043 | 0.086 | 0.066 |

- 三個變體（參考版、`-fast`、轉置修正的 `fastFixed`）各站的均方根差相同到 1e-4：效能版與 N-013 對壓力分布的
  影響遠小於 SA 模型與實驗的差。
- 最大的差在 η = 0.90 的震波位置（0.52）與翼尖 η = 0.99 下翼面後段（翼尖渦），是 SA、29.5 萬格的常見結果；
  內側站的 λ 型雙震波在這個網格上合成一道。
