// GPU-Native-CFD，GPL-3.0-or-later。見 arrayOps.cuh。

#include "core/arrayOps.cuh"

#include "core/gncCuda.cuh"

namespace gnc
{

namespace
{

constexpr int BS = 256;

struct Segs
{
    int op[ArrayBatch::MaxSegs];
    double* dst[ArrayBatch::MaxSegs];
    const double* src[ArrayBatch::MaxSegs];
    std::size_t n[ArrayBatch::MaxSegs];
    double v[ArrayBatch::MaxSegs];
    unsigned start[ArrayBatch::MaxSegs + 1];    // 各段的第一個 block
    int k;
};

__global__ void kArrayBatch(const Segs s)
{
    const unsigned b = blockIdx.x;
    int q = 0;
    while (q + 1 < s.k && b >= s.start[q + 1])
    {
        ++q;
    }
    const std::size_t i = std::size_t(b - s.start[q])*BS + threadIdx.x;
    if (i >= s.n[q])
    {
        return;
    }
    double* d = s.dst[q];
    const double* src = s.src[q];
    switch (s.op[q])
    {
        case 0: d[i] = s.v[q]; break;
        case 1: d[i] = src[i]; break;
        case 2: d[i] = d[i] + src[i]; break;
        case 3: d[i] = d[i] - src[i]; break;
        case 4: d[i] = -d[i]; break;
        default: d[i] = -src[i]; break;
    }
}

} // namespace


void ArrayBatch::push(const int op, double* d, const double* s, const std::size_t n, const double v)
{
    if (n == 0)
    {
        return;
    }
    if (k_ == MaxSegs)
    {
        flush();
    }
    op_[k_] = op;
    dst_[k_] = d;
    src_[k_] = s;
    n_[k_] = n;
    v_[k_] = v;
    ++k_;
}


void ArrayBatch::flush()
{
    if (k_ == 0)
    {
        return;
    }
    Segs s{};
    unsigned blocks = 0;
    for (int q = 0; q < k_; ++q)
    {
        s.op[q] = op_[q];
        s.dst[q] = dst_[q];
        s.src[q] = src_[q];
        s.n[q] = n_[q];
        s.v[q] = v_[q];
        s.start[q] = blocks;
        blocks += unsigned((n_[q] + BS - 1)/BS);
    }
    s.start[k_] = blocks;
    s.k = k_;
    kArrayBatch<<<blocks, BS>>>(s);
    GNC_CUDA_LAUNCH_CHECK();
    k_ = 0;
}

} // namespace gnc
