# N4c（WSL 開發紀錄）：力與力係數對 HiSA 逐位元

日期：2026-10-23。主機：**WSL**（RTX 3090）。原始碼雜湊 `f9d7badf…`（`BUILD_INFO`）。腳本：
`scripts/l3_steady_verify.sh --forces NAME=FUNC`（20 步，第 1 2 10 20 步與結束狀態）。

參考端：案例原本的 function（RAE2822 的 `forceCoeffsCompressible`、flatPlate 的 `forces`）每步執行，
`gncHisaDump` 排在它之後，以 `forceObjects` 寫出它的 result，另寫 `force`、`moment` 場（每個邊界面的
fP + fV 與力矩）。GPU 端：`src/post/forceCoeffs.cu` 依 OpenFOAM 的次序在主機端計算。

| 案例 | function | 逐位元 | 比對的紀錄 |
|---|---|---|---|
| rae2822SA | forceCoeffs | 5/5 | Cd、Cs、Cl、前後（f）（r）、CmRoll／Pitch／Yaw 與各自的 pressure／viscous／internal（48 個），force、moment 場 |
| flatPlateSA | forces | 5/5 | pressureForce 等 6 個向量 result，force、moment 場 |
| flatPlateLaminar | forces | 5/5 | 同上（laminar 的 nut = 0） |

過程（`N4c-wsl-dev1`、`dev2`，未收錄）：
- forces 的向量 result 一開始不同：OpenFOAM 的 `setResult` 把 vector 經過文字（controlDict 的
  writePrecision 位）存進字典，參考值只有 6 位有效數字；scalar 則是原值（所以 forceCoeffs 一次就逐位元）。
  GPU 端照同樣的文字來回（`%.*g` 再以 OpenFOAM 的規則讀回），另比對逐面的 force、moment 場（原值）。
- 場名沒有前綴：`functionObject::scopedName` 只在 `useNamePrefix` 為 true 時加 `<名稱>:`，預設 false。
