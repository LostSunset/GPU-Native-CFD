# kOmegaSST 的計時：HiSA（CPU）、混合式 GPU、gncHisa（g02）

日期：2026-10-23。主機：**g02**，2 × AMD EPYC 9684X ＋ L40（GPU3）。gncHisa 為 commit `e19bc511cf` 的建置
（原始碼雜湊 `90ba92c6…`，同 `../N4b-SST-L3/`）；腳本為 commit `9177ba88b4` 的 `scripts/sst_perf_g02.sh`。
主機 CPU 有其他使用者的 100 程序 MPI 作業（load average 112–119，`RUN_INFO`），CPU 的時間只作參考。

案例是原本的設定（Poisson 牆距離；gncHisa 以 PCG 計算，`../N4b-poisson-L1/`），穩態變體
（`perf_compare.sh`、`variant_compare.sh`）。每步為迴圈內的平均（HiSA：ExecutionTime 第一步到最後一步
之差；gncHisa：`-profile` 的平均，不含啟動）。HiSA MPI 為 metis 分解、`--bind-to none`。

| 每步（秒） | nacaAirfoil SST（40,000 格，200 步） | basicFinner SST（2,179,881 格，5 步） | 紀錄 |
|---|---:|---:|---|
| HiSA 1 核 | 0.314 | 53.2 | `../SSTP-g02-naca/`、本目錄 |
| HiSA 8 核 | 0.0417 | 7.82 | 同上 |
| HiSA 32 核 | 0.0154 | 2.10 | 同上 |
| 混合式（HiSA-GPU-Research 的 GMRESGPU MCBSGS-jgpu，1×L40） | — | 15.8 | `../SSTP-g02-hyb-finner/` |
| gncHisa 參考版（與 HiSA 逐位元，1×L40） | 0.109 | 1.91 | `../SSTP-g02-naca/`、本目錄 |
| gncHisa `-fast`（1×L40） | 0.00583 | 0.316 | `../SSTP-g02-gncfast-{naca,finner}/` |

- basicFinner SST：`-fast` 對 HiSA 單核 168 倍、32 核 6.7 倍、混合式 50 倍；參考版（逐位元）對單核 28 倍，
  已比 32 核快。
- nacaAirfoil SST（小的 2D 網格）：`-fast` 對 32 核 2.6 倍、單核 54 倍；參考版比 32 核慢（層級排程受延遲
  限制）。
- 外迭代的分段（`profile.txt`，basicFinner）：參考版 gmres 1.44 s、紊流 0.41 s（k、omega 的
  symGaussSeidel 以層級排程）；`-fast` gmres 0.205 s、紊流 0.050 s，其餘（通量、殘差、Jacobian）共約
  0.06 s。
