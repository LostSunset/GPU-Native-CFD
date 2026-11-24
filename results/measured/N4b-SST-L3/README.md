# N4b（g02 正式紀錄）：kOmegaSST RANS 的整個 run 對 HiSA 逐位元比對

日期：2026-10-23。主機：**g02**，GPU3（L40，sm_89），Atlas OpenFOAM v2606.0.2 runtime。原始碼雜湊
`90ba92c6…`（`BUILD_INFO`，`compiler_warnings=0`），即 commit `e19bc511cf`，同步時沒有未提交的路徑。
同一個建置依序跑了下表的 10 個紀錄（驅動腳本與紀錄：`N4b-final-g02-driver.{sh,log}`，12 段全部
exit 0）。量測期間主機有其他使用者的 CPU 工作（load average 約 110），不影響逐位元比對。

範圍與驗證方式見 `docs/decisions.md` N-022、`docs/design.md` §9.2。nacaAirfoil 與 basicFinner 原本
用 Poisson 牆距離（GAMG），逐位元比對用 meshWave 變體（`--meshwave`，HiSA 端同樣改 fvSchemes 的
`wallDist/method`）。

## 本紀錄（`scripts/l3_steady_verify.sh`，`SUMMARY.tsv`）

| 案例 | 格數 | 步數 | 比對的時間步 | 逐位元 | gpu1 對 gpu2 |
|---|---:|---:|---|---|---|
| nacaAirfoilSST（2D，upwind，kLowRe／omega／nutk 壁面函數，兩種紊流入口） | 40,000 | 200 | 1 2 10 50 100 200 end | 7/7 | 相同 |
| basicFinnerSST（3D，limitedLinear，6 個牆面、symmetry，nut 為 boundaryCorrectedNutLowRe） | 2,179,881 | 10 | 1 2 10 end | 4/4 | 相同 |

比對的紀錄：p T U e psi mu alpha rho rhoU rhoE phi pseudoCoDamping（內部與邊界）、pseudoCoNum，以及
k、omega、nut、alphat（內部與邊界）。basicFinner 的 10 步每一步都觸發 omega 的 bound（HiSA 的 log：
`bounding omega` 10 次），bound 之後的邊界語意（inletOutlet 族、symmetry）也在比對之內。

## 同一個建置的其他紀錄

| 紀錄 | 內容 | 結果 |
|---|---|---|
| `../N4b-powexp/` | glibc 移植：pow、exp、tanh、expm1、log（各 1 億個模型範圍與 5 千萬個寬範圍樣本） | 移植版全部 0 差異；CUDA 內建的 tanh 有 12.2%（模型範圍）的樣本差 1–3 ulp，expm1 2.1%、log 0.02% 差 1 ulp |
| `../N4b-walldist-L1/` | 牆距離 meshWave（nacaAirfoil、basicFinner 的變體） | 2/2 逐位元 |
| `../N4b-poisson-L1/` | 牆距離 Poisson（原本的案例；PCG 對 HiSA 的 GAMG，不逐位元） | 量測：見該目錄的 README |
| `../N4b-SST-L2/` | kOmegaSST 的一次 correct 與 validate（nacaAirfoil，第 1 2 3 10 30 100 步） | 7/7 |
| `../N4b-SST-L2-finner/` | 同上（basicFinner，第 1 2 3 步） | 4/4 |
| `../N4b-SST-app/` | 求解器執行檔 `gncHisa` 寫出的場檔（nacaAirfoilSST，60 步，時間目錄 30 60） | 16/16（p T U rho k omega nut alphat） |
| `../N4b-SA-L2/` | 回歸：SA 的一次 correct（flatPlate、oneraM6、rae2822） | 3 案例各 7/7 |
| `../N4b-SA-L3/` | 回歸：SA 的整個 run（flatPlate 800、oneraM6 30、rae2822 200 步） | 7/7、5/5、7/7 |
| `../N4b-L3/` | 回歸：層流與無黏 6 案例（N4a-L3 的設定） | 全部逐位元（7/7、7/7、6/6、7/7、5/5、4/4） |

N4b 的 kOmegaSST、壁面函數（omegaWallFunction 的 cornerWeights 與 setValues、kLowReWallFunction、
nutkWallFunction）、紊流入口（turbulentIntensityKineticEnergyInlet、turbulentMixingLengthFrequencyInlet）、
bounded Gauss limitedLinear、fvm::SuSp、tanh 與 log 的移植在這組紀錄下全部與 HiSA 逐位元相同。
**N4b 完成**（逐位元的部分；Poisson 牆距離為不逐位元的支援）。
