# N6：gncRhoSimple 的效能（g02）

日期：2026-10-23。主機：**g02**，2 × AMD EPYC 9684X（192 核）＋ NVIDIA L40（GPU3）。驅動：
`scripts/n6_perf_g02.sh`（`N6-all-g02-driver.log` 的後段，在 `../N6-L3/`）與補跑的 `N6-perf-of32`
（第一次的 32 核 MPI 用 scotch 分解，g02 的 OpenFOAM runtime 只有 dummy scotch，16,000、64,000 格
改用 metis 重跑；256,000、1,024,000 格在更正之後執行）。建置同 `../N6-L3/`（原始碼雜湊 `0f07d313…`）。
**主機 CPU 有其他使用者的 MPI 作業（load 約 112／192 核）**，CPU 的數字是在這個負載下量的。

案例：aerofoilNACA0012（rhoSimpleFoam 教學案例）與每個方向格數乘以 2、4、8 的加密網格
（`scripts/make_aerofoil_meshes.sh`：blockMesh、transformPoints、extrudeMesh；checkMesh 都 OK）。
每個網格兩個變體：原設定（p 為 GAMG）與 p 為 PCG＋DIC。工具：`scripts/rhosimple_perf.sh`。

| 執行 | 說明 |
|---|---|
| gnc | gncRhoSimple 參考次序版（PCG＋DIC 變體；與 rhoSimpleFoam 逐位元） |
| gncfast | gncRhoSimple `-fast`（PCG＋DIC 變體；多色 DIC、樹狀歸約） |
| gncamg | gncRhoSimple `-fast`（原設定；GAMG 以 PCG＋聚合式 AMG 代替，N-024） |
| of1、of1pcg | OpenFOAM v2606 rhoSimpleFoam 單核（原設定、PCG＋DIC 變體） |
| of32 | 同上，32 核 MPI（metis，`--bind-to none`；原設定） |
| spuma | SPUMA（OpenFOAM 的 GPU 移植，D-026）上的 rhoSimpleFoam（GPU3；原設定） |

## 每 SIMPLE 迭代的時間（秒）

gnc* 為求解器印出的迴圈時間／迭代數；OpenFOAM 為 ExecutionTime 的差／(N - 1)；SPUMA 以牆鐘
（ClockTime，整數秒）為準，因為它的 GPU 等待不計入 CPU 時間（ExecutionTime 只有牆鐘的 40–85%）。
括號內為 p 的平均線性迭代次數。

| 格數 | 迭代數 | gnc | gncfast | **gncamg** | of1（GAMG） | of1pcg | of32（GAMG） | spuma（牆鐘） |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 16,000 | 200（SPUMA 50） | 0.0326 (61) | 0.0191 (170) | **0.00995** (26) | 0.0373 (15) | 0.0400 (61) | 0.0230 (26) | 1.163 (20) |
| 64,000 | 100（SPUMA 20） | 0.180 (138) | 0.0440 (428) | **0.0170** (48) | 0.256 (39) | 0.229 (138) | 0.0755 (65) | 6.05 (52) |
| 256,000 | 50（OpenFOAM 單核 20、SPUMA 10） | 1.39 (325) | 0.122 (981) | **0.0382** (82) | 2.35 (107) | 1.77 (323) | 0.436 (269) | 42.2 (104) |
| 1,024,000 | 30（gnc、單核 10，SPUMA 5） | 7.72 (646) | 0.357 (1000) | **0.175** (132) | 16.9 (200) | 14.5 (646) | 1.26 (307) | 195 (194) |

- `gncamg` 對 OpenFOAM 單核：3.7、15、62、97 倍；對 32 核：2.3、4.4、11、7.2 倍；對 SPUMA：117、356、
  1100、1110 倍（同一張 L40）。
- `gncfast`（多色 DIC）在 256,000 格以上每次都撞到 maxIter 1000，p 沒有收斂到 relTol；大網格的比較以
  `gncamg` 為準。
- 參考次序版（逐位元）的 DIC 是層級掃描，大網格時受層數限制（1,024,000 格每迭代 7.7 秒），只用於驗證。
- p 的迭代數隨網格加密約加倍：OpenFOAM 的 GAMG（單核 15 → 200、32 核 26 → 307）與本專案的 AMG
  （26 → 132）都一樣。

## 到收斂（16,000 格，residualControl 1e-4）

| 執行 | 迭代數 | 牆鐘（總） | 迴圈每迭代 |
|---|---:|---:|---:|
| gnc（參考版） | 668 | 20.5 s | 0.0301 s |
| **gncamg**（`-fast`） | 678 | **7.5 s** | 0.0106 s |
| OpenFOAM 單核（GAMG） | 668 | 25.7 s | 0.0378 s |
| OpenFOAM 32 核（GAMG） | 667 | 20.5 s | 0.0236 s |

收斂解的差（`r1/fielddiff.*.txt`，L2 相對差）：

| 比較 | p | U | k |
|---|---:|---:|---:|
| gnc（PCG＋DIC）對 OpenFOAM 單核（GAMG） | 8.9e-7 | 8.4e-7 | 3.6e-6 |
| gncamg 對 OpenFOAM 單核 | 7.4e-5 | 9.0e-5 | 5.9e-4 |
| OpenFOAM 32 核對單核 | 2.4e-5 | 2.7e-5 | 1.4e-4 |

`-fast` 的解與 OpenFOAM 的差和 OpenFOAM 自己的單核對 32 核同量級（residualControl 1e-4 的量級）。
