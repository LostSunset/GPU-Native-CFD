# 混合式路線（HiSA-GPU-Research 的 GMRESGPU）：basicFinner 層流（g02）

日期：2026-10-23。主機：**g02**，GPU3（L40）。腳本：`scripts/hybrid_onera_g02.sh`（`HYBRID_SRC` 為
`cases/basicFinner-laminar/simulation`，`HYBRID_NAME=basicFinnerLaminarHybrid`，5 步），設定同 ONERA M6 的
`../HYB-g02-onera/`（LaxFriedrichsGPU、laplacianGPU、GMRESGPU 的 MCBSGS，`libhisaGpuSolver.so` 為 2026-10-21 的 g02 建置）。
HiSA 序列，迴圈內每步 11.8 s（`SUMMARY.tsv` 的 loop_per_step_s）；同案例 HiSA 單核 48.9 s、gncHisa -fast 0.266 s。
主機 CPU 有其他使用者的 MPI 作業。
