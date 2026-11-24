// GPU-Native-CFD，GPL-3.0-or-later。見 reduce.cuh。

#include "core/reduce.cuh"

#include "core/gncCuda.cuh"
#include "core/treeReduce.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

namespace gnc
{

namespace
{

constexpr int RT = tree::RT;        // 每塊 thread 數
constexpr int RPT = tree::RPT;      // 每 thread 依序累加的元素數
constexpr int CHUNK = tree::CHUNK;  // 每塊處理的元素數（固定）

// 循序和（s = 0; s += d[i]，i 遞增）：一個 block，thread 32 起把下一段（SEQ_CH
// 個元素）合併讀進 shared memory，thread 0 同時依序累加目前這一段（雙緩衝）。
// 累加次序與單一 thread 直接讀全域記憶體相同，結果逐位元相同；只是把記憶體
// 延遲藏在其他 thread 的載入裡
constexpr int SEQ_BS = 256;
constexpr int SEQ_CH = 2048;

__global__ void kSumSequential
(
    const double* __restrict__ d,
    const std::size_t n,
    double* __restrict__ out
)
{
    __shared__ double buf[2][SEQ_CH];
    const std::size_t nChunks = (n + SEQ_CH - 1)/SEQ_CH;
    const int tid = int(threadIdx.x);
    if (nChunks > 0)
    {
        for (std::size_t j = std::size_t(tid); j < SEQ_CH && j < n; j += SEQ_BS)
        {
            buf[0][j] = d[j];
        }
    }
    __syncthreads();
    double s = 0.0;
    for (std::size_t c = 0; c < nChunks; ++c)
    {
        const int b = int(c & 1);
        if (tid >= 32)
        {
            if (c + 1 < nChunks)
            {
                const std::size_t base = (c + 1)*SEQ_CH;
                for (std::size_t j = std::size_t(tid - 32); j < SEQ_CH && base + j < n; j += SEQ_BS - 32)
                {
                    buf[b ^ 1][j] = d[base + j];
                }
            }
        }
        else if (tid == 0)
        {
            const std::size_t base = c*SEQ_CH;
            const std::size_t m = (n - base < SEQ_CH) ? n - base : SEQ_CH;
            // 先把 8 個值讀進暫存器再依序相加：載入可以重疊，相加的次序不變
            std::size_t j = 0;
            for (; j + 8 <= m; j += 8)
            {
                double v[8];
                for (int k = 0; k < 8; ++k)
                {
                    v[k] = buf[b][j + std::size_t(k)];
                }
                for (int k = 0; k < 8; ++k)
                {
                    s += v[k];
                }
            }
            for (; j < m; ++j)
            {
                s += buf[b][j];
            }
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        *out = s;
    }
}


// 每塊：thread t 依序加 base + k*RT + t（k = 0..RPT-1），再以固定的
// 對半樹狀次序合併 256 個部分和
__global__ void kSumTreeBlock
(
    const double* __restrict__ d,
    const std::size_t n,
    double* __restrict__ partial
)
{
    __shared__ double sh[RT];
    const std::size_t base = std::size_t(blockIdx.x)*CHUNK;

    double s = 0.0;
    for (int k = 0; k < RPT; ++k)
    {
        const std::size_t i = base + std::size_t(k)*RT + threadIdx.x;
        if (i < n)
        {
            s += d[i];
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
        partial[blockIdx.x] = sh[0];
    }
}

template<int Op>
__device__ __forceinline__ double loadOp(const double* __restrict__ a, const double* __restrict__ b, const std::size_t i)
{
    if (Op == 1)
    {
        return ::fabs(a[i]);
    }
    if (Op == 2)
    {
        return a[i]*b[i];
    }
    return a[i];
}

// 單次啟動的 sumTree（ReduceOp 的運算在載入時做）：每塊與 kSumTreeBlock 相同；只有一塊時它的和
// 就是結果；否則最後完成的塊把各塊的部分和（≤ CHUNK 個）以 kSumTreeBlock 的次序再加一次。
// 結果寫到 out（mapped pinned 主機記憶體的裝置指標），計數器由最後一塊歸零
template<int Op>
__global__ void kSumTreeOnePass
(
    const double* __restrict__ a,
    const double* __restrict__ b,
    const std::size_t n,
    double* __restrict__ partial,
    unsigned* __restrict__ counter,
    double* __restrict__ out
)
{
    const std::size_t base = std::size_t(blockIdx.x)*CHUNK;
    double s = 0.0;
    for (int k = 0; k < RPT; ++k)
    {
        const std::size_t i = base + std::size_t(k)*RT + threadIdx.x;
        if (i < n)
        {
            s += loadOp<Op>(a, b, i);
        }
    }
    tree::finish(s, partial, counter, out);
}

// 多項版（N-028）：N 個和（各自的運算與輸入）在同一次啟動；每一項的每塊部分和、收尾的次序都與
// kSumTreeOnePass 相同，所以每一項的結果與單獨計算逐位元相同
constexpr int MAX_TERMS = 3;

struct TermsDev
{
    int op[MAX_TERMS];
    const double* a[MAX_TERMS];
    const double* b[MAX_TERMS];
};

__device__ __forceinline__ double loadOpRt(const int op, const double* __restrict__ a, const double* __restrict__ b, const std::size_t i)
{
    return op == 1 ? loadOp<1>(a, b, i) : (op == 2 ? loadOp<2>(a, b, i) : loadOp<0>(a, b, i));
}

template<int N>
__global__ void kSumTreeOnePassN
(
    const TermsDev t,
    const std::size_t n,
    double* __restrict__ partial,
    unsigned* __restrict__ counter,
    double* __restrict__ out
)
{
    __shared__ double sh[N][RT];
    __shared__ bool last;
    const std::size_t base = std::size_t(blockIdx.x)*CHUNK;
    for (int q = 0; q < N; ++q)
    {
        double s = 0.0;
        for (int k = 0; k < RPT; ++k)
        {
            const std::size_t i = base + std::size_t(k)*RT + threadIdx.x;
            if (i < n)
            {
                s += loadOpRt(t.op[q], t.a[q], t.b[q], i);
            }
        }
        sh[q][threadIdx.x] = s;
    }
    __syncthreads();
    for (int stride = RT/2; stride > 0; stride /= 2)
    {
        if (int(threadIdx.x) < stride)
        {
            for (int q = 0; q < N; ++q)
            {
                sh[q][threadIdx.x] += sh[q][threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    const unsigned nb = gridDim.x;
    if (nb == 1)
    {
        if (threadIdx.x == 0)
        {
            for (int q = 0; q < N; ++q)
            {
                out[q] = sh[q][0];
            }
        }
        return;
    }
    if (threadIdx.x == 0)
    {
        for (int q = 0; q < N; ++q)
        {
            partial[std::size_t(q)*CHUNK + blockIdx.x] = sh[q][0];
        }
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
    for (int q = 0; q < N; ++q)
    {
        double s = 0.0;
        for (int k = 0; k < RPT; ++k)
        {
            const std::size_t i = std::size_t(k)*RT + threadIdx.x;
            if (i < nb)
            {
                s += p[std::size_t(q)*CHUNK + i];
            }
        }
        sh[q][threadIdx.x] = s;
    }
    __syncthreads();
    for (int stride = RT/2; stride > 0; stride /= 2)
    {
        if (int(threadIdx.x) < stride)
        {
            for (int q = 0; q < N; ++q)
            {
                sh[q][threadIdx.x] += sh[q][threadIdx.x + stride];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        for (int q = 0; q < N; ++q)
        {
            out[q] = sh[q][0];
        }
        *counter = 0u;
    }
}

// 單次啟動版的固定緩衝（部分和、計數器、mapped 的結果），只在主機的一個 thread 使用
struct OnePassBuffers
{
    DeviceArray<double> partial;
    DeviceArray<unsigned> counter;
    double* hostOut = nullptr;
    double* devOut = nullptr;

    OnePassBuffers()
    {
        partial.allocate(std::size_t(MAX_TERMS)*CHUNK);
        counter.allocate(1);
        counter.zero();
        GNC_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&hostOut), MAX_TERMS*sizeof(double), cudaHostAllocMapped));
        GNC_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&devOut), hostOut, 0));
    }
};

constexpr int MM_BS = 1024;

// OpenFOAM 的 min(a, b) = (a < b) ? a : b，max(a, b) = (b < a) ? a : b
template<bool Max>
__device__ __forceinline__ double pick(const double m, const double v)
{
    return Max ? ((m < v) ? v : m) : ((v < m) ? v : m);
}

template<bool Max>
__global__ void kMinMax(const double* __restrict__ d, const std::size_t n, double* __restrict__ out)
{
    __shared__ double sh[MM_BS];
    const int tid = int(threadIdx.x);
    double m = d[0];
    for (std::size_t i = std::size_t(tid); i < n; i += MM_BS)
    {
        m = pick<Max>(m, d[i]);
    }
    sh[tid] = m;
    __syncthreads();
    for (int h = MM_BS/2; h > 0; h >>= 1)
    {
        if (tid < h)
        {
            sh[tid] = pick<Max>(sh[tid], sh[tid + h]);
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        *out = sh[0];
    }
}

constexpr int MM_MAX = 8;

struct MinMaxReqs
{
    const double* d[MM_MAX];
    std::size_t n[MM_MAX];
    bool max[MM_MAX];
};

// 每塊一項：與 kMinMax<Max> 相同的步驟
__global__ void kMinMaxMany(const MinMaxReqs r, double* __restrict__ out)
{
    __shared__ double sh[MM_BS];
    const int q = int(blockIdx.x);
    const int tid = int(threadIdx.x);
    const double* d = r.d[q];
    const std::size_t n = r.n[q];
    const bool mx = r.max[q];
    double m = d[0];
    for (std::size_t i = std::size_t(tid); i < n; i += MM_BS)
    {
        m = mx ? pick<true>(m, d[i]) : pick<false>(m, d[i]);
    }
    sh[tid] = m;
    __syncthreads();
    for (int h = MM_BS/2; h > 0; h >>= 1)
    {
        if (tid < h)
        {
            sh[tid] = mx ? pick<true>(sh[tid], sh[tid + h]) : pick<false>(sh[tid], sh[tid + h]);
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        out[q] = sh[0];
    }
}

// minMaxMany 的 mapped pinned 輸出
struct MinMaxOut
{
    double* h = nullptr;
    double* d = nullptr;
    MinMaxOut()
    {
        GNC_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h), MM_MAX*sizeof(double), cudaHostAllocMapped));
        GNC_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d), h, 0));
    }
};

} // namespace


void minMaxMany(const MinMaxReq* reqs, const int k, double* out)
{
    if (k < 1 || k > MM_MAX)
    {
        std::fprintf(stderr, "minMaxMany: 1-%d requests\n", MM_MAX);
        std::abort();
    }
    MinMaxReqs r{};
    for (int q = 0; q < k; ++q)
    {
        if (reqs[q].n == 0)
        {
            std::fprintf(stderr, "minMaxMany: empty request\n");
            std::abort();
        }
        r.d[q] = reqs[q].d;
        r.n[q] = reqs[q].n;
        r.max[q] = reqs[q].max;
    }
    static MinMaxOut o;
    kMinMaxMany<<<unsigned(k), MM_BS>>>(r, o.d);
    GNC_CUDA_LAUNCH_CHECK();
    GNC_CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    const volatile double* h = o.h;
    for (int q = 0; q < k; ++q)
    {
        out[q] = h[q];
    }
}


double minValue(const double* d, const std::size_t n)
{
    DeviceArray<double> out(1);
    kMinMax<false><<<1, MM_BS>>>(d, n, out.data());
    GNC_CUDA_LAUNCH_CHECK();
    return out.download()[0];
}


double maxValue(const double* d, const std::size_t n)
{
    DeviceArray<double> out(1);
    kMinMax<true><<<1, MM_BS>>>(d, n, out.data());
    GNC_CUDA_LAUNCH_CHECK();
    return out.download()[0];
}


namespace
{

// 主機端循序和用的 pinned 緩衝（每個主機執行緒一份，只增不減）
struct PinnedBuffer
{
    double* p = nullptr;
    std::size_t n = 0;

    ~PinnedBuffer()
    {
        if (p)
        {
            cudaFreeHost(p);
        }
    }

    double* get(const std::size_t m)
    {
        if (m > n)
        {
            if (p)
            {
                GNC_CUDA_CHECK(cudaFreeHost(p));
            }
            n = m;
            GNC_CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&p), n*sizeof(double)));
        }
        return p;
    }
};

// GNC_SEQSUM=gpu 時一律用 GPU 的單一 block 版
bool seqSumOnGpu()
{
    static const bool on = []
    {
        const char* e = std::getenv("GNC_SEQSUM");
        return e && std::string(e) == "gpu";
    }();
    return on;
}

// 主機端的循序累加：與 kSumSequential 相同的加法序列（s = 0、依序 s += d[i]）。
// 不可內嵌、不向量化：-ffp-contract=off、沒有 -ffast-math 時 GCC 不會重排浮點加法
__attribute__((noinline)) double hostSequentialSum(const double* d, const std::size_t n)
{
    double s = 0.0;
    for (std::size_t i = 0; i < n; ++i)
    {
        s += d[i];
    }
    return s;
}

} // namespace


double sumSequential(const double* d, const std::size_t n)
{
    // 參考次序版的循序和。L40 的 FP64 相依加法延遲長，單一 thread 每個元素約 20 ns；
    // 元素多時改成傳回主機（pinned）以同樣的次序相加，結果逐位元相同、快約 5–10 倍
    constexpr std::size_t hostMin = 1024;
    if (n >= hostMin && !seqSumOnGpu())
    {
        static thread_local PinnedBuffer buf;
        double* h = buf.get(n);
        GNC_CUDA_CHECK(cudaMemcpy(h, d, n*sizeof(double), cudaMemcpyDeviceToHost));
        return hostSequentialSum(h, n);
    }
    DeviceArray<double> out(1);
    kSumSequential<<<1, SEQ_BS>>>(d, n, out.data());
    GNC_CUDA_LAUNCH_CHECK();
    return out.download()[0];
}


namespace
{

template<int Op>
__global__ void kOpArray(const double* __restrict__ a, const double* __restrict__ b, const std::size_t n, double* __restrict__ out)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = loadOp<Op>(a, b, i);
    }
}

// 多次啟動的 sumTree（每層一個 kernel）：n > CHUNK² 時使用，也是單次啟動版的對照
double sumTreeMulti(const double* d, std::size_t n);

bool sumTreeCheck()
{
    static const bool on = []
    {
        const char* e = std::getenv("GNC_SUMTREE_CHECK");
        return e && std::string(e) == "1";
    }();
    return on;
}

} // namespace


namespace
{

OnePassBuffers& onePassBuffers()
{
    static OnePassBuffers buf;
    return buf;
}

int opCode(const ReduceOp op)
{
    return op == ReduceOp::Mag ? 1 : (op == ReduceOp::Prod ? 2 : 0);
}

} // namespace


void sumTreeOps(const ReduceTerm* terms, const int nTerms, const std::size_t n, double* results)
{
    if (nTerms < 1 || nTerms > MAX_TERMS)
    {
        std::fprintf(stderr, "sumTreeOps: 1-%d terms\n", MAX_TERMS);
        std::abort();
    }
    if (n == 0 || n > std::size_t(CHUNK)*CHUNK || nTerms == 1)
    {
        for (int q = 0; q < nTerms; ++q)
        {
            results[q] = sumTreeOp(terms[q].op, terms[q].a, terms[q].b, n);
        }
        return;
    }
    TermsDev t{};
    for (int q = 0; q < nTerms; ++q)
    {
        t.op[q] = opCode(terms[q].op);
        t.a[q] = terms[q].a;
        t.b[q] = terms[q].b;
    }
    OnePassBuffers& buf = onePassBuffers();
    const unsigned nb = unsigned((n + CHUNK - 1)/CHUNK);
    if (nTerms == 2)
    {
        kSumTreeOnePassN<2><<<nb, RT>>>(t, n, buf.partial.data(), buf.counter.data(), buf.devOut);
    }
    else
    {
        kSumTreeOnePassN<3><<<nb, RT>>>(t, n, buf.partial.data(), buf.counter.data(), buf.devOut);
    }
    GNC_CUDA_LAUNCH_CHECK();
    GNC_CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    const volatile double* h = buf.hostOut;
    double v[MAX_TERMS];
    for (int q = 0; q < nTerms; ++q)
    {
        v[q] = h[q];
    }
    for (int q = 0; q < nTerms; ++q)
    {
        results[q] = v[q];
        if (sumTreeCheck())
        {
            const double r = sumTreeOp(terms[q].op, terms[q].a, terms[q].b, n);
            if (!(r == v[q] && std::signbit(r) == std::signbit(v[q])))
            {
                std::fprintf(stderr, "sumTreeOps: term %d %.17g differs from single %.17g (n = %zu)\n", q, v[q], r, n);
                std::abort();
            }
        }
    }
}


void sumTreeOpsToDevice(const ReduceTerm* terms, const int nTerms, const std::size_t n, double* devOut)
{
    if (nTerms < 1 || nTerms > MAX_TERMS)
    {
        std::fprintf(stderr, "sumTreeOpsToDevice: 1-%d terms\n", MAX_TERMS);
        std::abort();
    }
    if (n == 0 || n > std::size_t(CHUNK)*CHUNK || nTerms == 1)
    {
        for (int q = 0; q < nTerms; ++q)
        {
            sumTreeToDevice(terms[q].op, terms[q].a, terms[q].b, n, devOut + q);
        }
        return;
    }
    TermsDev t{};
    for (int q = 0; q < nTerms; ++q)
    {
        t.op[q] = opCode(terms[q].op);
        t.a[q] = terms[q].a;
        t.b[q] = terms[q].b;
    }
    OnePassBuffers& buf = onePassBuffers();
    const unsigned nb = unsigned((n + CHUNK - 1)/CHUNK);
    if (nTerms == 2)
    {
        kSumTreeOnePassN<2><<<nb, RT>>>(t, n, buf.partial.data(), buf.counter.data(), devOut);
    }
    else
    {
        kSumTreeOnePassN<3><<<nb, RT>>>(t, n, buf.partial.data(), buf.counter.data(), devOut);
    }
    GNC_CUDA_LAUNCH_CHECK();
    if (sumTreeCheck())
    {
        double v[MAX_TERMS] = {};
        GNC_CUDA_CHECK(cudaMemcpy(v, devOut, std::size_t(nTerms)*sizeof(double), cudaMemcpyDeviceToHost));
        for (int q = 0; q < nTerms; ++q)
        {
            const double r = sumTreeOp(terms[q].op, terms[q].a, terms[q].b, n);
            if (!(r == v[q] && std::signbit(r) == std::signbit(v[q])))
            {
                std::fprintf(stderr, "sumTreeOpsToDevice: term %d %.17g differs from single %.17g (n = %zu)\n", q, v[q], r, n);
                std::abort();
            }
        }
    }
}


void sumTreeToDevice(const ReduceOp op, const double* a, const double* b, const std::size_t n, double* devOut)
{
    if (n == 0 || n > std::size_t(CHUNK)*CHUNK)
    {
        const double v = sumTreeOp(op, a, b, n);
        GNC_CUDA_CHECK(cudaMemcpy(devOut, &v, sizeof(double), cudaMemcpyHostToDevice));
        return;
    }
    OnePassBuffers& buf = onePassBuffers();
    const unsigned nb = unsigned((n + CHUNK - 1)/CHUNK);
    const int iop = opCode(op);
    if (iop == 1)
    {
        kSumTreeOnePass<1><<<nb, RT>>>(a, b, n, buf.partial.data(), buf.counter.data(), devOut);
    }
    else if (iop == 2)
    {
        kSumTreeOnePass<2><<<nb, RT>>>(a, b, n, buf.partial.data(), buf.counter.data(), devOut);
    }
    else
    {
        kSumTreeOnePass<0><<<nb, RT>>>(a, b, n, buf.partial.data(), buf.counter.data(), devOut);
    }
    GNC_CUDA_LAUNCH_CHECK();
    if (sumTreeCheck())
    {
        double v = 0.0;
        GNC_CUDA_CHECK(cudaMemcpy(&v, devOut, sizeof(double), cudaMemcpyDeviceToHost));
        const double r = sumTreeOp(op, a, b, n);
        if (!(r == v && std::signbit(r) == std::signbit(v)))
        {
            std::fprintf(stderr, "sumTreeToDevice: %.17g differs from sumTreeOp %.17g (n = %zu)\n", v, r, n);
            std::abort();
        }
    }
}


double sumTreeOp(const ReduceOp op, const double* a, const double* b, const std::size_t n)
{
    if (n == 0)
    {
        return 0.0;
    }
    const int iop = op == ReduceOp::Mag ? 1 : (op == ReduceOp::Prod ? 2 : 0);
    double r = 0.0;
    if (n > std::size_t(CHUNK)*CHUNK || sumTreeCheck())
    {
        DeviceArray<double> tmp;
        const double* in = a;
        if (iop != 0)
        {
            tmp.allocate(n);
            const unsigned nb = unsigned((n + 255)/256);
            if (iop == 1)
            {
                kOpArray<1><<<nb, 256>>>(a, b, n, tmp.data());
            }
            else
            {
                kOpArray<2><<<nb, 256>>>(a, b, n, tmp.data());
            }
            GNC_CUDA_LAUNCH_CHECK();
            in = tmp.data();
        }
        r = sumTreeMulti(in, n);
        if (n > std::size_t(CHUNK)*CHUNK)
        {
            return r;
        }
    }
    OnePassBuffers& buf = onePassBuffers();
    const unsigned nb = unsigned((n + CHUNK - 1)/CHUNK);
    if (iop == 1)
    {
        kSumTreeOnePass<1><<<nb, RT>>>(a, b, n, buf.partial.data(), buf.counter.data(), buf.devOut);
    }
    else if (iop == 2)
    {
        kSumTreeOnePass<2><<<nb, RT>>>(a, b, n, buf.partial.data(), buf.counter.data(), buf.devOut);
    }
    else
    {
        kSumTreeOnePass<0><<<nb, RT>>>(a, b, n, buf.partial.data(), buf.counter.data(), buf.devOut);
    }
    GNC_CUDA_LAUNCH_CHECK();
    GNC_CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
    const double v = *static_cast<volatile double*>(buf.hostOut);
    if (sumTreeCheck() && !(v == r && std::signbit(v) == std::signbit(r)))
    {
        std::fprintf(stderr, "sumTreeOp: one-pass %.17g differs from multi-kernel %.17g (n = %zu, op %d)\n", v, r, n, iop);
        std::abort();
    }
    return v;
}


double sumTree(const double* d, const std::size_t n)
{
    return sumTreeOp(ReduceOp::Sum, d, nullptr, n);
}


namespace
{

double sumTreeMulti(const double* d, std::size_t n)
{
    if (n == 0)
    {
        return 0.0;
    }
    DeviceArray<double> bufA;
    DeviceArray<double> bufB;
    const double* in = d;
    bool useA = true;

    // 每一層把 n 個值縮成 ceil(n/CHUNK) 個，直到剩一個
    do
    {
        const std::size_t nb = (n + CHUNK - 1)/CHUNK;
        DeviceArray<double>& outBuf = useA ? bufA : bufB;
        outBuf.allocate(nb);
        kSumTreeBlock<<<unsigned(nb), RT>>>(in, n, outBuf.data());
        GNC_CUDA_LAUNCH_CHECK();
        in = outBuf.data();
        n = nb;
        useA = !useA;
    } while (n > 1);

    double r = 0.0;
    GNC_CUDA_CHECK(cudaMemcpy(&r, in, sizeof(double), cudaMemcpyDeviceToHost));
    return r;
}

} // namespace


bool useTreeReduce()
{
    static const bool on = []
    {
        const char* e = std::getenv("GNC_REDUCE");
        return e && std::string(e) == "tree";
    }();
    return on;
}


double sumSolver(const double* d, const std::size_t n)
{
    return useTreeReduce() ? sumTree(d, n) : sumSequential(d, n);
}

} // namespace gnc
