// CUDA 錯誤處理與裝置陣列。GPU-Native-CFD，GPL-3.0-or-later。
//
// 解的狀態常駐 GPU，資料只做明確搬移（N-005）：DeviceArray 用 CUDA 的流序
// 配置器（cudaMallocAsync，預設記憶體池）加上主機端的區塊快取，不用 unified memory。

#ifndef GNC_CORE_GNCCUDA_CUH
#define GNC_CORE_GNCCUDA_CUH

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#define GNC_CUDA_CHECK(call)                                                  \
    do                                                                        \
    {                                                                         \
        const cudaError_t gncErr_ = (call);                                   \
        if (gncErr_ != cudaSuccess)                                           \
        {                                                                     \
            std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n",              \
                cudaGetErrorName(gncErr_), __FILE__, __LINE__,                \
                cudaGetErrorString(gncErr_));                                 \
            std::exit(3);                                                     \
        }                                                                     \
    } while (0)

// kernel 啟動後檢查啟動錯誤；同步錯誤由之後的 cudaMemcpy／同步呼叫回報
#define GNC_CUDA_LAUNCH_CHECK() GNC_CUDA_CHECK(cudaGetLastError())

namespace gnc
{

// 預設記憶體池的釋放門檻設為最大，釋放的區塊不還給系統（第一次配置時設定）
inline void retainPoolMemory()
{
    static bool done = false;
    if (!done)
    {
        int dev = 0;
        GNC_CUDA_CHECK(cudaGetDevice(&dev));
        cudaMemPool_t pool;
        GNC_CUDA_CHECK(cudaDeviceGetDefaultMemPool(&pool, dev));
        std::uint64_t threshold = UINT64_MAX;
        GNC_CUDA_CHECK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
        done = true;
    }
}

// 主機端的區塊快取（N7）：釋放的區塊依位元組數放回清單，下次同樣大小的配置直接
// 取用，省掉 cudaMallocAsync／cudaFreeAsync 的 API 開銷（外迭代中的暫存陣列每步
// 數百次，g02 的 nsys：每外迭代約 2 毫秒的主機時間）。所有工作都在預設 stream 上
// 依序執行，重用與流序配置一樣安全。區塊不歸還（程序結束時由驅動程式回收）；
// 清單放在不解構的堆積物件，靜態物件解構的次序不影響
class BlockCache
{
    std::vector<std::pair<std::size_t, std::vector<void*>>> bins_;

    std::vector<void*>* bin(const std::size_t bytes)
    {
        for (auto& b : bins_)
        {
            if (b.first == bytes)
            {
                return &b.second;
            }
        }
        return nullptr;
    }

public:
    static BlockCache& instance()
    {
        static BlockCache* c = new BlockCache;
        return *c;
    }

    void* get(const std::size_t bytes)
    {
        std::vector<void*>* b = bin(bytes);
        if (!b || b->empty())
        {
            return nullptr;
        }
        void* p = b->back();
        b->pop_back();
        return p;
    }

    void put(void* p, const std::size_t bytes)
    {
        std::vector<void*>* b = bin(bytes);
        if (!b)
        {
            bins_.emplace_back(bytes, std::vector<void*>());
            b = &bins_.back().second;
        }
        b->push_back(p);
    }
};


// 擁有一段裝置記憶體；只能移動，不能複製
template<class T>
class DeviceArray
{
    T* ptr_ = nullptr;
    std::size_t n_ = 0;

public:
    DeviceArray() = default;

    explicit DeviceArray(std::size_t n)
    {
        allocate(n);
    }

    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    DeviceArray(DeviceArray&& o) noexcept
    :
        ptr_(o.ptr_),
        n_(o.n_)
    {
        o.ptr_ = nullptr;
        o.n_ = 0;
    }

    DeviceArray& operator=(DeviceArray&& o) noexcept
    {
        if (this != &o)
        {
            release();
            ptr_ = o.ptr_;
            n_ = o.n_;
            o.ptr_ = nullptr;
            o.n_ = 0;
        }
        return *this;
    }

    ~DeviceArray()
    {
        release();
    }

    // 先找 BlockCache 中同樣大小的區塊，沒有時以流序配置（cudaMallocAsync，預設
    // stream；記憶體池的釋放門檻設為最大）。釋放時放回 BlockCache。數值結果與配置
    // 方式無關（內容一律由呼叫端寫入或 zero()）
    void allocate(std::size_t n)
    {
        release();
        n_ = n;
        if (n_)
        {
            ptr_ = static_cast<T*>(BlockCache::instance().get(n_*sizeof(T)));
            if (!ptr_)
            {
                retainPoolMemory();
                GNC_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&ptr_), n_*sizeof(T), 0));
            }
        }
    }

    void release()
    {
        if (ptr_)
        {
            BlockCache::instance().put(ptr_, n_*sizeof(T));
        }
        ptr_ = nullptr;
        n_ = 0;
    }

    void upload(const std::vector<T>& h)
    {
        if (h.size() != n_)
        {
            allocate(h.size());
        }
        if (n_)
        {
            GNC_CUDA_CHECK
            (
                cudaMemcpy(ptr_, h.data(), n_*sizeof(T), cudaMemcpyHostToDevice)
            );
        }
    }

    std::vector<T> download() const
    {
        std::vector<T> h(n_);
        if (n_)
        {
            GNC_CUDA_CHECK
            (
                cudaMemcpy(h.data(), ptr_, n_*sizeof(T), cudaMemcpyDeviceToHost)
            );
        }
        return h;
    }

    // 位元組全部填 0（double 的 +0.0、int 的 0）
    void zero()
    {
        if (n_)
        {
            GNC_CUDA_CHECK(cudaMemset(ptr_, 0, n_*sizeof(T)));
        }
    }

    T* data() { return ptr_; }
    const T* data() const { return ptr_; }
    std::size_t size() const { return n_; }
};


// FP64 向量場，SoA（x、y、z 各一個陣列；design.md §3.1）
struct DeviceVec3Array
{
    DeviceArray<double> x, y, z;

    DeviceVec3Array() = default;

    explicit DeviceVec3Array(std::size_t n)
    :
        x(n), y(n), z(n)
    {}

    std::size_t size() const { return x.size(); }

    void zero()
    {
        x.zero();
        y.zero();
        z.zero();
    }

    // 取回主機並交錯成 OpenFOAM vectorField 的記憶體配置（x0 y0 z0 x1 ...）
    std::vector<double> downloadAoS() const
    {
        const std::vector<double> hx = x.download();
        const std::vector<double> hy = y.download();
        const std::vector<double> hz = z.download();
        std::vector<double> out(3*hx.size());
        for (std::size_t i = 0; i < hx.size(); ++i)
        {
            out[3*i] = hx[i];
            out[3*i + 1] = hy[i];
            out[3*i + 2] = hz[i];
        }
        return out;
    }

    // 由交錯配置（OpenFOAM vectorField）上傳
    void uploadAoS(const std::vector<double>& aos)
    {
        const std::size_t n = aos.size()/3;
        std::vector<double> hx(n), hy(n), hz(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            hx[i] = aos[3*i];
            hy[i] = aos[3*i + 1];
            hz[i] = aos[3*i + 2];
        }
        x.upload(hx);
        y.upload(hy);
        z.upload(hz);
    }
};

} // namespace gnc

#endif
