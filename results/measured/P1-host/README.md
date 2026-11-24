# P1-host（g02，論文的硬體與軟體版本）

日期：2026-10-25。主機：**g02**（2 × AMD EPYC 9684X，4 × NVIDIA L40，使用 GPU3）。建置：論文版
（原始碼雜湊 `1bf4bc6d…`，`configs/hosts/g02-paper.env` 的獨立目錄；`BUILD_INFO`）。
Driver：`scripts/paper_extra_g02.sh --phases host,cases,fields,geometry --no-gpu-wait`（`P1a-driver.log`）。

| 檔案 | 內容 |
|---|---|
| `lscpu.txt` | CPU 型號、核心數、時脈、快取 |
| `nvidia-smi-q.txt` | GPU3 的完整資訊（記憶體、時脈、功率上限、驅動版本） |
| `free.txt` | 主記憶體（GiB） |
| `versions.txt` | 核心、glibc、nvcc、g++、CMake、Nsight Systems、驅動、Open MPI、OpenFOAM 的版本與當時的 uptime |

用於論文的硬體表與軟體表。
