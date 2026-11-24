// 聚合式代數多重網格（AMG）：效能版的 PCG 前置條件（N6）。GPU-Native-CFD，
// GPL-3.0-or-later。
//
// 不與 OpenFOAM 逐位元（N-024）。階層照 OpenFOAM v2606 GAMG 的作法建立：
//   聚合     pairGAMGAgglomeration::agglomerate（pairGAMGAgglomerate.C）：依序走訪格
//            （每層交替正向、反向），把格與「兩邊都未分組、面權重最大」的相鄰格配成一對；
//            找不到時併入權重最大的相鄰群；仍未分組的自成一群。反向的層把粗格編號反轉。
//            起始面權重為 faceAreaPair 的 mag(cmptMultiply(Sf/sqrt(magSf), (1, 1.01, 1.02)))，
//            之後各層把細面的權重加到粗面。continueAgglomerating：粗格數 < nCellsInCoarsestLevel
//            或沒有變少時停止；每 mergeLevels 個配對層合成一層（combineLevels）
//   粗網格   GAMGAgglomerateLduAddressing.C：細面兩邊落在不同粗格時屬於（粗 owner, 粗 neighbour）
//            的粗面（依 owner、第一次出現的次序編號），兩邊同一粗格時為粗格內部的面；
//            粗面方向與細面相反時 upper、lower 互換（faceFlipMap）
//   粗矩陣   GAMGSolverAgglomerateMatrix.C：粗 diag = Σ 細 diag（細格次序）＋ Σ 內部細面的
//            upper + lower（細面次序）；粗 upper／lower = Σ 對應細面的係數（細面次序）
// 前置條件為對稱 V-cycle：每層由 0 起前掃 nSweeps 次，殘差以聚合相加限制到下一層，最粗層以密集 LU
// 直接解，修正以注入延拓、乘以 omega（未平滑聚合的過度修正）後加回，再後掃 nSweeps 次（前置條件對稱，
// 可用於 PCG）。平滑子：第 0 層為多色 Gauss–Seidel（前掃色遞增、後掃色遞減），第 jacobiFrom 層起為
// Jacobi（不分色，前後相同；N-028）。網格固定時階層建一次。預設（N-026、N-027、N-028）：每 2 個配對層
// 合成一層、nSweeps 2、omega 1.6、第 1 層起 Jacobi（權重 1）、FP32、單一 block 門檻 1024 格
// （aerofoilNACA0012 的 16,000–1,024,000 格中最快或接近最快）。著色：相鄰格的圖是二分圖時兩色
// （結構化四邊形網格的第 0 層），否則依格編號與依鄰格數由大到小的兩種貪婪著色取色數少的。
// 各層的矩陣與向量是階層持有的固定緩衝：每次求解把細矩陣複製進第 0 層再累加粗層（FP64、LDU、
// 原編號），再打包成套用用的格式（N-027）：每層的格依顏色重新編號（同色連續），矩陣存成 CSR
// （每列的鄰格與係數連續），係數與 V-cycle 的向量為 GNC_AMG_PREC 位元（預設 32；PCG 本身是 FP64）。
// 第一次前掃把尚未更新的鄰格當 0（不需歸零），殘差與限制合併為一個 kernel，格數不超過
// GNC_AMG_BLOCK（預設 1024）的粗層整段子循環在一個 thread block 內完成（一次啟動）。V-cycle 的
// kernel 序列與指標固定，第一次套用時以 stream capture 錄成 CUDA Graph，之後重播
// （GNC_AMG_GRAPH=0 時逐一啟動）。

#ifndef GNC_FVM_AMG_CUH
#define GNC_FVM_AMG_CUH

#include "core/gncCuda.cuh"
#include "mesh/deviceMesh.cuh"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gnc
{

struct AmgControls
{
    int nCellsInCoarsestLevel = 10;
    int mergeLevels = 1;
    int nSweeps = 2;            // 前掃、後掃各幾次（相同，前置條件對稱）
    int maxCoarsest = 512;      // 最粗層密集 LU 的上限
    double omega = 1.6;         // 粗網格修正的倍數（x += omega*P e；聚合式 AMG 的過度修正）
    int precision = 32;         // V-cycle 的係數與向量：32（float）或 64（double）
    int blockCells = 1024;      // 格數不超過此值的粗層在一個 thread block 內完成（0：不用）
    int blockSweeps = 0;        // 這些粗層的前後掃次數（0：同 nSweeps）
    double jacobi = 1.0;        // > 0：第 jacobiFrom 層起以此權重的 Jacobi 平滑（不分色），0：全部多色 Gauss–Seidel
    int jacobiFrom = 1;
};

// fvSolution 的 GAMG 項目中這裡用到的鍵（nCellsInCoarsestLevel、mergeLevels）；環境變數
// GNC_AMG_MERGE、GNC_AMG_SWEEPS、GNC_AMG_COARSEST、GNC_AMG_OMEGA、GNC_AMG_PREC、GNC_AMG_BLOCK、
// GNC_AMG_CSWEEPS、GNC_AMG_JAC、GNC_AMG_JFROM、GNC_AMG_WP 優先（量測用）
AmgControls amgControlsFromEnv(AmgControls c);

// 一層的定址（裝置端）。原編號的 LDU 定址用於累加粗矩陣與最粗層的 LU；套用時的格式
// 以「列」為單位：列依顏色排列（colorOff 為各色的列範圍），perm[列] = 格
struct AmgLevel
{
    std::int32_t nCells = 0;
    std::int32_t nFaces = 0;
    DeviceArray<std::int32_t> lowerAddr;
    DeviceArray<std::int32_t> upperAddr;
    std::vector<std::int32_t> colorOff;
    // CSR（列的次序）：col 為鄰格的列，src 為係數來源（< 0：upper[-(src + 1)]，本格是 owner；
    // 否則 lower[src]，本格是 neighbour）
    DeviceArray<std::int32_t> perm;
    DeviceArray<std::int32_t> invPerm;
    DeviceArray<std::int32_t> rowPtr;
    DeviceArray<std::int32_t> col;
    DeviceArray<std::int32_t> src;
    // 由上一層（細）到這一層的限制（第 0 層沒有）：原編號的累加表與列的對應
    DeviceArray<std::int32_t> rOff;             // 本層格 → 細格（遞增）
    DeviceArray<std::int32_t> rCells;
    DeviceArray<std::int32_t> fOff;             // 本層面 → 細面（遞增；翻轉的存 -(f + 1)）
    DeviceArray<std::int32_t> fFaces;
    DeviceArray<std::int32_t> iOff;             // 本層格 → 粗格內部的細面（遞增）
    DeviceArray<std::int32_t> iFaces;
    DeviceArray<std::int32_t> parent;           // 細層的列 → 本層的列
    DeviceArray<std::int32_t> cOff;             // 本層的列 → 細層的列
    DeviceArray<std::int32_t> cRows;
};

// 一個 thread block 的子循環讀的各層指標（T 為 V-cycle 的精度）
template<class T>
struct AmgLevelView
{
    std::int32_t nRows;
    std::int32_t nColors;
    const std::int32_t* colorOff;   // 裝置端，nColors + 1
    const std::int32_t* rowPtr;
    const std::int32_t* col;
    const T* val;
    const T* d;
    const T* invd;
    T* x;
    T* b;
    T* t;                           // Jacobi 的暫存
    const std::int32_t* parent;     // 本層的列 → 下一層（粗）的列
    const std::int32_t* cOff;       // 下一層的列 → 本層的列
    const std::int32_t* cRows;
};

template<class T>
struct AmgBuffers
{
    std::vector<DeviceArray<T>> val, d, invd, x, b, t;
    std::vector<DeviceArray<std::int32_t>> colorOffD;
    DeviceArray<AmgLevelView<T>> views;
};

struct AmgHierarchy
{
    AmgControls controls;
    std::vector<AmgLevel> levels;               // [0] 為原網格
    int blockLevel = 0;                          // 由這一層起在一個 thread block 內完成

    // 各層的 LDU 矩陣（FP64、原編號）、套用用的緩衝（依 controls.precision 用其中一組）、
    // 最粗層的 LU、錄好的 V-cycle
    std::vector<DeviceArray<double>> diag, lower, upper;
    AmgBuffers<float> f32;
    AmgBuffers<double> f64;
    DeviceArray<double> dense;
    cudaGraphExec_t vcycle = nullptr;

    ~AmgHierarchy();
};

// 網格的階層（依 dm 與 controls 快取，網格不變時只建一次）
AmgHierarchy& amgHierarchy(const DeviceMesh& dm, const AmgControls& c);

// 一次求解的前置條件：建構時由細矩陣（已加邊界 diag）累加各層粗矩陣，並分解最粗層
class AmgPrecond
{
public:
    AmgPrecond
    (
        const DeviceMesh& dm,
        AmgHierarchy& h,
        const double* D,
        const double* lower,
        const double* upper
    );

    // z = B r（對稱 V-cycle）
    void apply(double* z, const double* r) const;

private:
    AmgHierarchy& h_;
};

} // namespace gnc

#endif
