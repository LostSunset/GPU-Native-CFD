# N4c（g02 正式紀錄）：力與力係數對 HiSA 逐位元，效能版的 Cl、Cd

日期：2026-10-23。主機：**g02**，GPU3（L40，sm_89）。commit `94b1e81d00` 的建置（原始碼雜湊
`ecfee9ac…`，沒有未提交的路徑，`compiler_warnings=0`）。驅動腳本與紀錄：`N4c-final-g02-driver.{sh,log}`，
6 段全部 exit 0。主機 CPU 有其他使用者的 MPI 作業（load average 約 104）。

方法見 WSL 的開發紀錄 `../N4c-wsl-dev3/README.md`、`../N4c-wsl-app2/README.md`。

## 本紀錄：力與力係數的 L3（`scripts/l3_steady_verify.sh --forces`）

| 案例 | function | 步數 | 比對的時間步 | 逐位元 |
|---|---|---:|---|---|
| rae2822SA | forceCoeffs（`forceCoeffsCompressible`） | 200 | 1 2 10 50 100 200 end | 7/7 |
| flatPlateSA | forces | 800（案例的 endTime） | 1 2 10 100 400 800 end | 7/7 |
| flatPlateLaminar | forces | 100 | 1 2 10 100 end | 5/5 |

每個時間步比對全部狀態場、forceCoeffs 的 48 個係數 result（或 forces 的 6 個向量 result，依 OpenFOAM 的存放
方式捨入）與逐面的 force、moment 場。

## 同一個建置的其他紀錄

| 紀錄 | 內容 | 結果 |
|---|---|---|
| `../N4c-g02-L3-auto/` | 層級排程自動模式（N-020）之後預設執行的 L3：forwardStep 50 步（單一 block）、ONERA M6 SA 10 步（逐層啟動） | 5/5、4/4 |
| `../N4c-g02-app/` | `gncHisa` 寫出的場檔（二進位）與 postProcessing（RAE2822 200 步、flatPlate SA 200 步，writeInterval 100） | 15/15、16/16；coefficient.dat、force.dat、moment.dat 與 HiSA 逐位元組相同 |
| `../N4c-g02-app-regress/` | 執行檔的回歸（forwardStep、nacaAirfoil 層流 20 步） | 8/8、8/8 |
| `../N4c-g02-coeffs/` | RAE2822 SA 4000 步，ref、lusgsColor、fast 的 Cl、Cd（每步寫出） | 下表 |

## RAE2822 的力係數（`../N4c-g02-coeffs/rae2822SA/coeffs.txt`，`scripts/coeff_summary.py`）

參考版的殘差停在約 4.3e-3（震波附近的限制器），力係數在小範圍內振盪；以最後 1000 步的平均比較。

| 變體 | 每外迭代 | 4000 步 | Cd（後段平均） | Cl（後段平均） | CmPitch（後段平均） |
|---|---:|---:|---:|---:|---:|
| ref（逐位元） | 0.0403 s | 163.2 s | 0.0143892 | 0.6620501 | -0.0831581 |
| lusgsColor | 0.0281 s | 114.2 s | 0.0143804 | 0.6620347 | -0.0831503 |
| fast | 0.0073 s | 30.8 s | 0.0143802 | 0.6620346 | -0.0831495 |

- fast 對 ref：Cd 差 -9.1e-6（0.09 drag count），Cl 差 -1.5e-5，CmPitch 差 8.5e-6；後段的振盪（標準差）
  Cd 5–7e-5、Cl 5–7e-5、CmPitch 1–2e-5，差異都在振盪幅度之內。
- lusgsColor 與 fast 幾乎相同：樹狀歸約與多色 SGS 不改變結果的量級，差異來自多色的 LU-SGS 預條件。
- 時間：fast 每外迭代 7.3 毫秒，其中 GMRES 4.1、湍流 0.9、殘差 0.75、Jacobian 0.6、通量 0.46、場更新 0.38
  毫秒（`rae2822SA/profile.fast.txt`）。
