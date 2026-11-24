# GPU-Native-CFD

[English](README.md) | **正體中文**

GPU-Native-CFD 以 CUDA 從頭重寫兩個 OpenFOAM 家族的有限體積求解器。整個迭代都在手寫的 kernel 中、以常駐 GPU 的資料
執行；主機只讀取案例、啟動 kernel，每個迭代只收回少數幾個純量。求解器不連結任何 OpenFOAM 函式庫。

| 求解器 | 對應 | 方法 |
|---|---|---|
| `gncHisa` | HiSA（High Speed Aerodynamic solver） | 密度基耦合隱式：AUSM+-up 通量、近似 Lax–Friedrichs Jacobian、以 LU-SGS 為前置條件的 GMRES、偽時間步進；層流、Spalart–Allmaras 與 k–ω SST |
| `gncRhoSimple` | OpenFOAM `rhoSimpleFoam`（v2606） | 壓力基可壓縮 SIMPLE，k–ω SST |

每個求解器有兩種模式：

- **參考次序模式**（預設）。每個歸約、遞迴與運算式都依 CPU 原程式的次序計算：以有序的格–面 gather 表取代 atomics、
  層級排程的 Gauss–Seidel／LU-SGS／DIC／DILU、循序加總、移植 glibc 的 `pow`、`exp`、`tanh` 與 `log`，並且編譯時不做
  FMA 合併。在這個模式下，兩個求解器與 HiSA、`rhoSimpleFoam` 在整個計算中**逐位元相同**，包括紊流模型、力係數與寫出的檔案。
- **效能模式**（`-fast`）。多色排序的掃描、固定形狀的樹狀歸約、耦合 Jacobian 以 FP32 儲存，壓力方程則以 PCG 搭配聚合
  AMG 前置條件。它收斂到相同的離散解，但不逐位元相同；每一項改動都以參考次序模式為基準量測。

論文所依據的量測紀錄放在 `results/measured/`。

**1.0.0 版**就是產生論文結果的程式：原始碼雜湊（`CMakeLists.txt`、`src`、`tests`、`tools`、`apps`）為
`1bf4bc6daf538373ee5ce3d15a39c75667f81e22cec9d754db00c7edde099372`，與各紀錄 `BUILD_INFO` 中的 `source_sha256` 相同。
這個版本也包含為論文撰寫的量測與繪圖腳本。

## 目錄結構

```
CMakeLists.txt        求解器的建置（CMake；CUDA + C++17）
src/                  函式庫：core（歸約、glibc 移植）、io、mesh、fields、ops、thermo、fvm（矩陣、AMG）、
                      turbulence、post（力）、solvers/gncHisa、solvers/gncRhoSimple
apps/                 執行檔 gncHisa 與 gncRhoSimple
tests/                比對與測試程式（gncCompare、gncFieldDiff、L0–L3 驅動程式、gncPowCheck 等）
tools/ref/            以 wmake 對 OpenFOAM／HiSA 建置的參考工具（逐位元比對用的狀態傾印）
scripts/              建置、案例準備、驗證、計時與繪圖腳本
configs/hosts/        主機設定（example.env；g02-l40.env 與 g02-paper.env 是作者的量測主機）
results/measured/     量測紀錄（每次執行一個目錄；見「量測紀錄」）
results/external/     ONERA M6 實驗資料（AGARD AR-138）
```

程式註解與各紀錄的 README 以正體中文撰寫。註解中提到的設計決策（`N-xxx`）與設計筆記指向作者內部的開發紀錄，
不包含在這個版本中。

## 需求

| 元件 | 紀錄使用的版本 | 用途 |
|---|---|---|
| Linux x86-64 | Ubuntu 24.04 | 全部 |
| NVIDIA GPU | L40（sm_89）；開發用 RTX 3090（sm_86） | 求解器 |
| CUDA toolkit | 13.2 | 求解器 |
| CMake | 3.28（至少 3.24） | 求解器 |
| GCC | 13.3 | 求解器與參考程式 |
| OpenFOAM | v2606（OpenCFD） | 以 `foamDictionary` 準備案例；參考工具；CPU 比較 |
| HiSA | gitlab.com/hisa/hisa commit `271b4491`（tag `validated-v2512`），以 OpenFOAM v2606 建置 | `gncHisa` 的參考與 CPU 比較 |
| Python 3 | 3.12，繪圖需要 matplotlib | 分析與繪圖 |

**逐位元相同的條件。** 參考是*編譯後*的 CPU 程式，所以逐位元相同是對特定的參考建置而言。紀錄使用 GCC 13.3（`-O3`、
x86-64、`linux64GccDPInt32Opt`）建置的 OpenFOAM v2606.0.2 與 HiSA，主機 CPU 支援 FMA 與 AVX2，因此 glibc 2.39
（Ubuntu libc6 2.39-0ubuntu8.9）選用 `*_fma` 的程式路徑；`src/core/glibc*.cuh` 的移植重現的正是這些路徑。作者使用的
OpenFOAM 是 v2606 加上兩個 `lduMatrix` 檔案的覆蓋（`lduMatrixATmul.C`、`lduMatrixOperations.C`），在 GCC 建置下其迴圈依面
次序循序執行；對未修改的 v2606 建置的逐位元相同尚未測試。不同的 glibc 或 CPU 功能集會改變參考本身；
`scripts/pow_check.sh` 可檢查超越函數的移植與主機函式庫是否一致。

## 建置

1. 複製 `configs/hosts/example.env`，調整路徑與 CUDA 架構。
2. 建置求解器、測試程式與參考工具：

   ```bash
   bash scripts/build.sh --host-env configs/hosts/my-host.env
   ```

   執行檔寫到 `$BUILD_ROOT/bin`，`$BUILD_ROOT/BUILD_INFO` 記錄原始碼雜湊、工具鏈版本、實際的 `nvcc` 命令列與每個執行檔的
   SHA-256。原始碼雜湊涵蓋 `CMakeLists.txt`、`src`、`tests`、`tools` 與 `apps`；以這個版本建置時，它等於紀錄
   `BUILD_INFO` 中的 `source_sha256`。

   只建置求解器時也可以只用 CMake（建置時不需要 OpenFOAM）：

   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
   cmake --build build -j
   ```

   `-DGNC_FMAD=ON` 允許裝置程式碼做 FMA 合併；參考次序模式需要預設的 `OFF`。

## 執行案例

求解器讀取的 OpenFOAM 案例，其字典要先以 `foamDictionary -expand` 展開：

```bash
bash scripts/gnc_prepare_case.sh -case CASE --host-env configs/hosts/my-host.env
gncHisa -case CASE            # 參考次序模式，與 HiSA 逐位元相同
gncHisa -case CASE -fast      # 效能模式
gncRhoSimple -case CASE -fast
```

選項：`-profile` 印出迭代中每個階段的時間；`gncHisa -no-foam-compat` 把參考建置中兩個與編譯器相依的行為換成依原始碼字面的
結果。不支援的字典項目、邊界條件或模型會讓求解器報錯停止。

## 測試案例

- **HiSA 範例**（shock tube、forward-facing step、flat plate、RAE2822、NACA 0012、ONERA M6、basicFinner）：

  ```bash
  bash scripts/cases/fetch_hisa_example.sh --name nacaAirfoil --cache ~/hisa-cache --dest ~/export/nacaAirfoil
  bash scripts/cases/prepare_nacaAirfoil.sh --src ~/export/nacaAirfoil \
       --dest $CASE_ROOT/nacaAirfoil-prepared --bashrc $OPENFOAM_BASHRC
  ```

  prepare 腳本會照各範例自己的前處理步驟執行，並寫出 `PREPARED` 與檢查碼。論文使用的衍生案例：
  `scripts/make_onera_forces_case.sh`（加上力係數的 ONERA M6）與 `scripts/derive_case.sh`（由參考狀態續算）。
- **`rhoSimpleFoam` aerofoilNACA0012**：複製 `$FOAM_TUTORIALS/compressible/rhoSimpleFoam/aerofoilNACA0012` 並執行它的
  `Allrun.pre`（16,000 格）。`scripts/make_aerofoil_meshes.sh --host-env FILE --src THAT_COPY --out-root DIR 1 2 4 8`
  再把每個方向的格數乘上 1、2、4、8 倍（16,000 到 1,024,000 格），每套網格各有兩種變體：`-rR` 用教學案例的設定
  （p 用 GAMG），`-rR-pcg` 的 p 用 PCG／DIC，是逐位元比對使用的變體。紀錄中 16,000 格的案例稱為
  `aerofoilNACA0012` 與 `aerofoilNACA0012-pcg`。

## 重現結果

| 結果 | 腳本 | 紀錄 |
|---|---|---|
| 幾何與運算子（L0、L1） | `l0_verify.sh`、`l1_verify.sh`、`l1_walldist_verify.sh` | `N4a-L0`、`N4a-L1` |
| 一個外迭代（L2） | `l2_init_verify.sh`、`l2_res_verify.sh`、`l2_outer_verify.sh`、`l2_turb_verify.sh` | `N4a-*-L2`、`N4a-SA-L2`、`N4b-SST-L2*` |
| 完整計算（L3）、寫出的檔案、力 | `l3_steady_verify.sh`、`app_verify.sh` | `N4a-L3`、`N4a-SA-L3`、`N4b-SST-L3`、`N4c-g02-L3-forces`、`M6-g02-L3-forces`、`N4c-g02-app`、`N4b-SST-app` |
| `gncRhoSimple` 對 `rhoSimpleFoam`（L2、L3） | `rhosimple_verify.sh` | `N6-L3`、`N6d-L3` |
| 超越函數的移植 | `pow_check.sh` | `N4a-powexp`、`N4b-powexp` |
| 注入的捨入改動（突變測試） | `mutate.py`、`mutation_g02.sh`、`mutation_summary.py` | `MUT-*` |
| 密度基求解器對 HiSA 的計時 | `perf_compare.sh` | `CPU2-g02-*`、`SSTP-g02-*`、`N7*-g02-perf-onera` |
| 壓力基求解器對 OpenFOAM 與 SPUMA 的計時 | `rhosimple_perf.sh` | `N6-perf-*`、`N6d-perf-*` |
| 410 萬與 1,640 萬格的壓力基計時（OpenFOAM、SPUMA、brae、GPU 記憶體與功率） | `n6e_large_g02.sh`（網格以 `make_aerofoil_meshes.sh … 16 32` 產生） | `N6e-perf-r16`、`N6e-perf-r32`、`N6e-spuma-r16`、`N6e-spuma-r32`、`N6e-brae`、`N6e-gpu` |
| modifiedFinner（394 萬格）：逐位元計算、計時、收斂、混合與 SPUMA 路線 | `n8_g02.sh` | `MF-g02-*` |
| 效能模式不受位元中性改動影響 | `fast_identity.sh` | `N6d-id-*`、`N6d-sumcheck*` |
| 小網格改動的消融分析 | `n6d_ablation_g02.sh` | `N6d-abl*` |
| 效能模式的準確度 | `onera_conv_g02.sh`、`finner_conv_g02.sh`、`coeff_summary.py`、`field_diff.py` | `M6-g02-conv`、`FIN-g02-conv2`、`N4c-g02-coeffs`、`N7f-g02-coeffs`、`N7-g02-conv`、`N6d-perf-conv` |
| 同一張 GPU 上的 brae | `brae_g02.sh`（brae 另行建置） | `BRAE-perf` |
| 主機硬體與軟體版本、案例設定、網格大小、網格表面、收斂後的翼型場 | `paper_extra_g02.sh --phases host,cases,geometry,fields` | `P1-host`、`P1-cases`、`P1-geometry`、`P1-fields` |
| 每個迭代的 kernel、傳輸與 CUDA API 剖析；GPU 能耗；`rhoSimpleFoam` 的線性求解時間；GPU 記憶體 | `paper_extra_g02.sh --phases nsys,nsyslarge,energy,energylong,ofprof,memory`，再執行 `python3 scripts/paper_extra_summary.py` | `P1-nsys`、`P1-nsys-large`、`P1-energy`、`P1-energylong`、`P1-ofprof`、`P1-memory` |
| 圖 | `python3 scripts/plots/make_figures.py`（計時、收斂、壓力分布、階段分解）、`make_case_figures.py`（網格、翼型場）、`make_diagrams.py`（示意圖）；全部只讀 `results/measured` | — |

所有腳本都接受 `--host-env FILE` 與 `--label LABEL`，並在 `$RUN_ROOT/LABEL` 下寫出新的紀錄目錄。名為 `*_g02.sh` 的檔案
是產生紀錄時實際使用的驅動程式，執行於一台共用主機（兩顆 AMD EPYC 9684X、四張 NVIDIA L40，只使用其中一張）；檔案開頭
有該主機的絕對路徑，為了保留來源原樣不改。在其他主機上重跑時，修改這些變數即可。

紀錄中的 CPU 計時是在主機上有其他工作執行時量測的（負載記錄在每個 `RUN_INFO`）；它們是參考值，不是 CPU 的最佳效能。

## 量測紀錄

`results/measured/` 中的每個目錄是一次執行，視情況包含：

- `BUILD_INFO`：原始碼雜湊、編譯器版本與旗標、執行檔的 SHA-256；
- `RUN_INFO`：主機、GPU、開始與結束時間、主機負載；
- `SUMMARY.tsv` 與各案例的 log（`log.*`）、比對報告（`compare.*.txt`、`fielddiff*.txt`）；
- 驅動程式的 log（`*-driver.log`）與 `README.md`：執行了什麼、用哪個驅動腳本、結果（正體中文）。

二進位的狀態傾印（`*.gnc`）因檔案太大沒有收錄；它們的 SHA-256 列在各案例的 `SHA256SUMS`。這裡只收錄論文所依據的紀錄，
以及它們所依賴的紀錄（同一個建置的回歸、其表格引用的前一步結果）；其餘的開發過程不收錄。名稱含 `wsl` 的目錄是在開發主機
（RTX 3090）上執行的。

## 未收錄

- 比較中的混合 GPU 路線（HiSA 把 Jacobian 組裝與 GMRES 放到 GPU）是另一個專案；這裡只收錄它的計時紀錄（`HYB-*`）。
- SPUMA 與 brae 是第三方程式；只收錄它們執行時的 log。

## 授權

GPL-3.0-or-later（`LICENSE`）。由 OpenFOAM、HiSA 與 GNU C Library 衍生的程式碼來源、實驗資料與商標聲明見 `NOTICE.md`。
本軟體未經 OpenCFD Limited（OpenFOAM 軟體的製作與發行者，OPENFOAM® 與 OpenCFD® 商標的所有者）認可或背書。

## 引用

請以 `CITATION.cff` 引用本軟體（GitHub 的「Cite this repository」）。論文使用的 1.0.0 版封存於 Zenodo：
[10.5281/zenodo.24486251](https://doi.org/10.5281/zenodo.24486251)（所有版本：
[10.5281/zenodo.24486250](https://doi.org/10.5281/zenodo.24486250)）。描述方法與這些結果的論文：

> S.-C. Lin, R.-B. Chen, C.-Y. Wu, Bitwise-reproducible GPU-native rewrites of OpenFOAM solvers: a verification
> baseline and a comparison of GPU porting routes (submitted to Computer Physics Communications).

論文發表後，本節會改為期刊的引用資訊。
