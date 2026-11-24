# N4a（g02 正式紀錄）：Spalart-Allmaras RANS 的整個 run 對 HiSA 逐位元比對

日期：2026-10-22。主機：**g02**，GPU3（L40，sm_89），Atlas OpenFOAM v2606.0.2 runtime。原始碼雜湊
`f949b0c3…`（`BUILD_INFO`，`compiler_warnings=0`），即 commit `c73ca45203`（N4a-4），同步時沒有未提交的
路徑。同一個建置依序跑了下表的 14 個紀錄（驅動腳本與紀錄：`N4a-final-g02-driver.{sh,log}`，15 段全部
exit 0）。量測期間主機有其他使用者的 CPU 工作（load average 約 110），不影響逐位元比對。

方法同各層級的 WSL 開發紀錄（`../N4a2-wsl-dev3/`、`../N4a3-wsl-dev2/`、`../N4a4-g02-dev6/`、
`../N4a3-wsl-app1/`）。

## 本紀錄（`scripts/l3_steady_verify.sh`，`SUMMARY.tsv`）

| 案例 | 格數 | 步數 | 比對的時間步 | 逐位元 | gpu1 對 gpu2 |
|---|---:|---:|---|---|---|
| flatPlateSA（flatPlate-g2000） | 8,000 | 800（案例的 endTime） | 1 2 10 100 400 800 end | 7/7 | 相同 |
| oneraM6SA | 294,912 | 30 | 1 2 10 30 end | 5/5 | 相同 |
| rae2822SA | 23,552 | 200 | 1 2 10 50 100 200 end | 7/7 | 相同 |

比對的紀錄：p T U e psi mu alpha rho rhoU rhoE phi pseudoCoDamping（內部與邊界）、pseudoCoNum，以及
nuTilda、nut、alphat（內部與邊界）。

## 同一個建置的其他紀錄

| 紀錄 | 內容 | 結果 |
|---|---|---|
| `../N4a-L0/` | 幾何（6 案例，含 basicFinner 218 萬格） | 6/6 逐位元，兩次執行相同 |
| `../N4a-L1/` | 運算子 | 6/6 逐位元，讀寫回讀逐位元 |
| `../N4a-N3a-L2/` | 初始化與熱力 | 6/6 |
| `../N4a-N3b-L2/` | 通量與殘差（層流） | 6/6（`nocompat` 欄為不相容設定的對照，非判定項） |
| `../N4a-N3c-L2/` | Jacobian、預條件、線性求解、場更新 | 6/6，四項全部逐位元 |
| `../N4a-walldist-L1/` | 牆距離（meshWave） | 6/6 |
| `../N4a-powexp/` | glibc pow／exp 移植（各 1 億個 SA 範圍與 5 千萬個寬範圍樣本） | glibc 移植 0 差異；CUDA 內建 pow／exp 分別有 9.1%／3.6% 的樣本差 1 ulp |
| `../N4a-SA-L2/` | SA 方程的一次 correct（flatPlate、oneraM6、rae2822，第 1 2 3 10 30 100 步） | 3 案例各 7/7 |
| `../N4a-L3/` | N3d 的層流與無黏 6 案例（shockTube、forwardStep 200 步；nacaAirfoil 150；flatPlate 800；oneraM6 30；basicFinner 10） | 全部逐位元（7/7、7/7、6/6、7/7、5/5、4/4） |
| `../N4a-L3-converge/` | 偽時間收斂判斷（絕對 1e-3、相對 2e4） | 兩者都在同一步停止（111、100），4/4 |
| `../N4a-app/` | 求解器執行檔 `gncHisa` 寫出的場檔（forwardStep、nacaAirfoil 層流，20 步） | 8/8、8/8 |
| `../N4a-app-converge/` | 執行檔，收斂停止（flatPlate 層流） | 時間目錄 50 100 111 相同，12/12 |
| `../N4a-app-SA/` | 執行檔，SA（flatPlate，100 步） | 14/14（含 nuTilda、nut、alphat） |

N4a（SA、壁面函數、牆距離、cellLimited 梯度、inletOutlet／symmetry 的矩陣係數、limited corrected
Laplacian）在 g02 的正式紀錄全部與 HiSA 逐位元相同。kOmegaSST（N4b）依 N-019 在 N7 之後進行。

場檔與傾印檔不在 repo：`*.gnc` 的 SHA-256 在各紀錄的 `SHA256SUMS.dumps`（`log.gnc` 的檔名也符合 `*.gnc`，
一併只留雜湊），執行檔紀錄的場檔在 `SHA256SUMS.fields`。
