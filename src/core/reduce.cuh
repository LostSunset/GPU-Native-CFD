// 固定次序歸約。GPU-Native-CFD，GPL-3.0-or-later。
//
// 兩種版本，都不用浮點 atomics，同一輸入每次結果逐位元相同：
//   sumSequential  參考次序版：由 0 依序累加，與 OpenFOAM sum()（FieldFunctions.C:403，
//                  result = 0; result += f[i]）以及序列執行時的 gSum 逐位元相同。
//                  元素少時在 GPU 以一個 thread 累加；1024 個以上時傳回主機（pinned）
//                  以同樣次序相加（同一串 IEEE 加法；L40 的 FP64 相依加法延遲使單一
//                  thread 很慢）。GNC_SEQSUM=gpu 時一律在 GPU。只用於驗證。
//   sumTree        效能版：固定分塊（每塊 2048 個元素）與固定樹狀次序，
//                  結果只取決於 n，與 GPU 型號、SM 數無關；與 OpenFOAM
//                  的次序不同，報告時要標示（README 原則 4）。
// 結果以一個 double 取回主機。

#ifndef GNC_CORE_REDUCE_CUH
#define GNC_CORE_REDUCE_CUH

#include <cstddef>

namespace gnc
{

double sumSequential(const double* d, std::size_t n);

double sumTree(const double* d, std::size_t n);

// sumTree 對逐元素運算後的陣列：Sum a[i]、Mag |a[i]|、Prod a[i]*b[i]（N-027）。與「先寫出運算結果
// 再 sumTree」逐位元相同（--fmad=false，乘積不與加法合併），省去中間陣列與一個 kernel。
// n ≤ 2048² 時單次啟動：最後完成的 block 以同樣的固定樹狀次序累加各 block 的部分和，結果直接
// 寫進 mapped pinned 主機記憶體
enum class ReduceOp { Sum, Mag, Prod };

double sumTreeOp(ReduceOp op, const double* a, const double* b, std::size_t n);

// 同一個 n 的 1–3 個 sumTreeOp 在同一次啟動、同一次同步完成（各自的樹狀次序不變，結果與分開
// 呼叫逐位元相同；N-028）
struct ReduceTerm
{
    ReduceOp op;
    const double* a;
    const double* b;
};

void sumTreeOps(const ReduceTerm* terms, int nTerms, std::size_t n, double* results);

// sumTreeOp 的結果寫到裝置端的 devOut（不同步主機），供下一個 kernel 直接讀（例如 normFactor 的
// 平均值）；結果與 sumTreeOp 相同
void sumTreeToDevice(ReduceOp op, const double* a, const double* b, std::size_t n, double* devOut);

// sumTreeOps 的結果寫到裝置端的 devOut[0..nTerms-1]（不同步主機）
void sumTreeOpsToDevice(const ReduceTerm* terms, int nTerms, std::size_t n, double* devOut);


// 最小、最大值（N7）：一個 block（1024 threads）平行比較，n > 0。有限值的
// 最小／最大值與比較次序無關（只有 -0 與 +0 相等時保留哪一個可能不同）；呼叫端
// 只拿結果和門檻比較（bound 的 min(psi) < lowerBound、updateFields 的 Newton
// 收斂判斷），所以與 OpenFOAM 循序的 min／max 逐位元等價。含 NaN 時的結果不同
// （循序版取決於 NaN 的位置），但那時求解已經失敗
double minValue(const double* d, std::size_t n);

double maxValue(const double* d, std::size_t n);

// 1–8 個 minValue／maxValue 在一次啟動、一次同步完成（每塊一項，演算法與單獨呼叫相同，結果相同；N-028）
struct MinMaxReq
{
    const double* d;
    std::size_t n;
    bool max;
};

void minMaxMany(const MinMaxReq* reqs, int k, double* out);

// 效能版選項（N7）：GNC_REDUCE=tree 時求解器的歸約改用固定樹狀次序
bool useTreeReduce();

// 求解器使用的和：預設 sumSequential（與 HiSA 逐位元），GNC_REDUCE=tree 時 sumTree
double sumSolver(const double* d, std::size_t n);

} // namespace gnc

#endif
