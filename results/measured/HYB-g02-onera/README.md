# 混合式路線（HiSA-GPU-Research 的 GMRESGPU）：ONERA M6 SA（g02）

日期：2026-10-23。主機：**g02**，GPU3（L40）。腳本：`HYB-g02-driver.sh`（`scripts/hybrid_onera_g02.sh`），設定同
HiSA-GPU-Research 的 G2-CLEAN-2 最佳變體 MCBSGS-jgpu（Jacobian 的 LaxFriedrichsGPU、laplacianGPU 與 GMRESGPU 的
MCBSGS 預條件在 GPU，其餘在主機；`DERIVED.case` 為 fvSolution 的差異）。HiSA 序列 50 步，迴圈內每步 0.752 s
（`SUMMARY.tsv`）；同案例 HiSA 單核 3.75 s、32 核 0.176 s、gncHisa 參考版 0.232 s、-fast 0.0228 s。主機 CPU 有其他
使用者的 MPI 作業。
