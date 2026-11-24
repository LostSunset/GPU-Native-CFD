// 層級排程（LU-SGS、symGaussSeidel）的 cooperative 啟動（N7）。GPU-Native-CFD，
// GPL-3.0-or-later。
//
// 三種排程，每格的運算不變，結果逐位元相同（N7，g02 的 L3 驗證）：
//   GNC_COOP_LEVELS=0  每一層啟動一個 kernel
//   GNC_COOP_LEVELS=1  cooperative launch，層與層之間 grid.sync()
//   GNC_COOP_LEVELS=2  單一 block（1024 threads），層與層之間 __syncthreads()
// 未設定時自動選擇：每層平均格數 <= 1024（單一 block 一輪處理得完）用 2，否則用 0。
// g02（L40）的量測：flatPlate（每層約 45 格）與 RAE2822（約 95 格）2 最快，
// ONERA M6（約 1,700 格）0 最快；1 在三個案例都比 0 慢。
//
// 效能版（不與 HiSA 逐位元，N7）：以多色（graph coloring）次序取代格編號次序，
// 每次掃描只需 2 × 色數次啟動：
//   GNC_LUSGS=color  平均流的 LU-SGS 預條件
//   GNC_SGS=color    純量方程（SA）的 symGaussSeidel

#ifndef GNC_CORE_COOPLEVELS_CUH
#define GNC_CORE_COOPLEVELS_CUH

#include "core/gncCuda.cuh"

#include <algorithm>
#include <cstdlib>
#include <string>

namespace gnc
{

constexpr int levelBlockSize = 1024;

// 環境變數的設定；未設定（或其他值）為 -1（自動）
inline int levelModeSetting()
{
    static const int mode = []
    {
        const char* e = std::getenv("GNC_COOP_LEVELS");
        if (!e) return -1;
        const std::string s(e);
        return s == "0" ? 0 : (s == "1" ? 1 : (s == "2" ? 2 : -1));
    }();
    return mode;
}

// nCells 格分成 nLevels 層時使用的排程
inline int levelMode(const long long nCells, const long long nLevels)
{
    const int s = levelModeSetting();
    if (s >= 0)
    {
        return s;
    }
    return (nLevels > 0 && nCells <= nLevels*levelBlockSize) ? 2 : 0;
}

inline bool envIs(const char* name, const char* value)
{
    const char* e = std::getenv(name);
    return e && std::string(e) == value;
}

inline bool useColoredLusgs()
{
    static const bool on = envIs("GNC_LUSGS", "color");
    return on;
}

inline bool useColoredSgs()
{
    static const bool on = envIs("GNC_SGS", "color");
    return on;
}

// cooperative 啟動的 grid 大小：每個 SM 可常駐的 block 數 × SM 數
template<class Kernel>
int coopGridSize(Kernel k, const int blockSize)
{
    int dev = 0;
    GNC_CUDA_CHECK(cudaGetDevice(&dev));
    int sms = 0;
    GNC_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
    int perSm = 0;
    GNC_CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, k, blockSize, 0));
    return std::max(1, perSm*sms);
}

} // namespace gnc

#endif
