// 效能版固定樹狀歸約的裝置端部分（N-028）。GPU-Native-CFD，GPL-3.0-or-later。
//
// sumTree（reduce.cu）的單次啟動 kernel 用：每塊 RT 個 thread、處理 CHUNK = RT*RPT 個元素，thread t 依
// k = 0..RPT-1 累加元素 base + k*RT + t，再以固定的對半樹狀次序合併 RT 個部分和；不只一塊時，最後完成的
// 塊以同樣的次序把各塊的部分和（≤ CHUNK 個）再加一次。在產生數值的 kernel 裡直接用這個分塊同時歸約
// 也會得到逐位元相同的和，但小網格時塊數太少（16,000 格只有 8 塊），量起來比分開的 kernel 慢，所以沒有用。

#ifndef GNC_CORE_TREEREDUCE_CUH
#define GNC_CORE_TREEREDUCE_CUH

#include <cstddef>

namespace gnc
{
namespace tree
{

constexpr int RT = 256;             // 每塊 thread 數
constexpr int RPT = 8;              // 每 thread 依序累加的元素數
constexpr int CHUNK = RT*RPT;       // 每塊處理的元素數

inline unsigned blocksFor(const std::size_t n)
{
    return unsigned((n + CHUNK - 1)/CHUNK);
}

// 本 thread 已依 k 的次序累加好 s；完成這一塊與（最後一塊時）各塊的合併，結果寫到 *out。
// partial 至少 gridDim.x 個，counter 為 0（最後一塊用完歸零）。整塊都要呼叫
__device__ __forceinline__ void finish(double s, double* __restrict__ partial, unsigned* __restrict__ counter, double* __restrict__ out)
{
    __shared__ double sh[RT];
    __shared__ bool last;
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int stride = RT/2; stride > 0; stride /= 2)
    {
        if (int(threadIdx.x) < stride)
        {
            sh[threadIdx.x] += sh[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const unsigned nb = gridDim.x;
    if (nb == 1)
    {
        if (threadIdx.x == 0)
        {
            *out = sh[0];
        }
        return;
    }
    if (threadIdx.x == 0)
    {
        partial[blockIdx.x] = sh[0];
        __threadfence();
        last = atomicAdd(counter, 1u) == nb - 1;
    }
    __syncthreads();
    if (!last)
    {
        return;
    }
    __threadfence();
    const volatile double* p = partial;
    s = 0.0;
    for (int k = 0; k < RPT; ++k)
    {
        const std::size_t i = std::size_t(k)*RT + threadIdx.x;
        if (i < nb)
        {
            s += p[i];
        }
    }
    sh[threadIdx.x] = s;
    __syncthreads();
    for (int stride = RT/2; stride > 0; stride /= 2)
    {
        if (int(threadIdx.x) < stride)
        {
            sh[threadIdx.x] += sh[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        *out = sh[0];
        *counter = 0u;
    }
}

} // namespace tree
} // namespace gnc

#endif
