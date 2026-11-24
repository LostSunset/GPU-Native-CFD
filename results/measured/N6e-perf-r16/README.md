# N6e-perf-r16、N6e-perf-r32（g02，aerofoilNACA0012 加密 16、32 倍的每迭代時間）

日期：2026-10-25。主機：**g02**（2 × EPYC 9684X，其他使用者的 MPI 作業使 load 約 100／192；GPU3 連續 3 分鐘沒有其他
程式之後才開始，另一個 session 的 GPU 與 CPU 工作都已結束）。建置：論文版（原始碼雜湊 `1bf4bc6d…`；`BUILD_INFO`）。
Driver：`scripts/n6e_large_g02.sh`（log 在 `../N6e-gpu/N6e-driver.log`），各執行為 `scripts/rhosimple_perf.sh`。

網格：`scripts/make_aerofoil_meshes.sh … 16 32`（教學案例的 blockMesh 每個方向的格數乘 16、32）：
4,096,000 格（r16）與 16,384,000 格（r32）；`-pcg` 變體 p 為 PCG＋DIC（參考版），否則為教學設定的 GAMG。

| 網格 | 變體 | 迭代 | 每迭代 s | p 的平均迭代數 |
|---|---|---|---|---|
| r16 | gnc（參考版） | 5 | 32.29 | 980.2 |
| r16 | gncamg（-fast） | 20 | 0.4572 | 68.5 |
| r16 | of1（rhoSimpleFoam 單核） | 5 | 98.66 | 310.4 |
| r16 | of32（32 核 MPI） | 20 | 5.162 | 384.9 |
| r32 | gnc（參考版） | 3 | 103.0 | 1000（上限） |
| r32 | gncamg（-fast） | 10 | 3.397 | 88.7 |
| r32 | of1 | 3 | 810.4 | 685.0 |
| r32 | of32 | 10 | 29.27 | 552.0 |

注意：
- 參考版在 r32 的 PCG/DIC 每次都到 1000 次的上限（p 未達到容許誤差）；每迭代時間是在上限下的值。
- `-fast` 每迭代時間的成長比格數快（r8→r32：格數 16 倍，p 求解每個 PCG 迭代 39 倍，其餘部分約 22 倍）；
  原因見 `../P1-nsys-large`。
- GPU 記憶體峰值見 `../N6e-gpu/summary.tsv`：-fast r16 8,026 MiB、r32 30,618 MiB（r1–r8 的直線擬合預測 30,766 MiB）。
- SPUMA 沒有在這次執行：driver 在 brae 之後卡在 SPUMA 的 watchdog（指令替換等待背景 sleep），依 PID 停止；
  SPUMA 改由 `n6e_large_g02.sh --steps spuma` 另外執行，結果見 `../N6e-spuma-r16/README.md`（r16 940.5 s、r32 3807 s）。
