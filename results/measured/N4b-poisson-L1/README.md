# N4b：Poisson 牆距離對 HiSA 的差異（g02）

日期：2026-10-23。建置同 `../N4b-SST-L3/`（commit `e19bc511cf`）。腳本：`scripts/l1_walldist_verify.sh`
（不加 `--meshwave`：原本的案例，`gncWallDistL1 --poisson`）。

HiSA（OpenFOAM v2606 的 `patchDistMethods::Poisson`）以 GAMG 解 `laplacian(yPsi) == -1` 到 fvSolution 的
yPsi 容差（1e-5、relTol 0，最多 1000 次）；本專案（`turbulence/turbulenceCommon.cu` 的
`wallDistancePoisson`）以對角預條件的 PCG 解同一個離散方程到正規化殘差 1e-10，再以同樣的公式
`y = max(sqrt(magSqr(grad(yPsi)) + 2*yPsi) - mag(grad(yPsi)), SMALL)` 得到牆距離（N-022：不逐位元）。

| 案例 | 格數 | HiSA 的 GAMG（log.ref） | 本專案的 PCG（log.gpu） | y 的最大絕對差 | 最大相對差 |
|---|---:|---|---|---:|---:|
| basicFinner | 2,179,881 | 22 次，殘差 6.7e-7 | 584 次，9.7e-11 | 4.3e-6 m | 1.1e-3（邊界值逐位元相同） |
| nacaAirfoil | 40,000 | **1000 次（上限），殘差 0.47，沒有收斂** | 12,897 次，1.0e-10 | 11.4 m | 0.76 |

- basicFinner：兩者都收斂，差別只來自 GAMG 的容差，最大相對差 1.1e-3。
- nacaAirfoil：HiSA 的 GAMG 在 1000 次後殘差仍為 0.47，HiSA 在這個案例用的 Poisson 牆距離沒有收斂，
  差異來自參考端。本專案的 PCG 收斂到 1e-10。以 Poisson 跑的 HiSA 結果因此受牆距離的誤差影響；
  逐位元的比較改用 meshWave 變體（`../N4b-SST-L3/`）。
