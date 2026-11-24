// exp(x) 的 device 版，與 x86-64 glibc 2.39 的 __ieee754_exp_fma 逐位元相同。
// GPU-Native-CFD，GPL-3.0-or-later；演算法與表格出自 GNU C Library
// （sysdeps/ieee754/dbl-64/e_exp.c，LGPL-2.1-or-later；源自 ARM optimized-routines）。
//
// 作法同 core/glibcPow.cuh（N-018）：本機 libm.so.6（Ubuntu libc6 2.39-0ubuntu8.9，
// sha256 fce00b6f…）以 libc6-dbg 的符號在 gdb 中反組譯 __ieee754_exp_fma，逐條照
// 機器碼寫出；表格與 pow 共用 __exp_data（core/glibcPowData.cuh 的 exp[]）。
// 必須以 --fmad=false 編譯。

#ifndef GNC_CORE_GLIBCEXP_CUH
#define GNC_CORE_GLIBCEXP_CUH

#include "core/glibcPow.cuh"

namespace gnc
{
namespace glibcExpDetail
{

using glibcPowDetail::asDouble;
using glibcPowDetail::asU64;
using glibcPowDetail::expTab;
using glibcPowDetail::xflow;

// specialcase（e_exp.c:44–81）
__device__ __forceinline__ double specialcase(const double tmp, std::uint64_t sbits, const std::uint64_t ki)
{
    if ((ki & 0x80000000ULL) == 0)
    {
        sbits -= 1009ULL << 52;
        const double scale = asDouble(sbits);
        return 0x1p1009*fma(tmp, scale, scale);
    }
    sbits += 1022ULL << 52;
    const double scale = asDouble(sbits);
    const double ts = tmp*scale;
    double y = scale + ts;
    if (1.0 > y)
    {
        const double lo = (scale - y) + ts;
        const double hi = y + 1.0;
        y = ((((1.0 - hi) + y) + lo) + hi) - 1.0;
        if (y == 0.0)
        {
            return 0.0;
        }
    }
    return y*0x1p-1022;
}

} // namespace glibcExpDetail


// __ieee754_exp_fma（e_exp.c:98–158）
__device__ inline double glibcExp(const double x)
{
    using namespace glibcExpDetail;
    const std::uint64_t ix = asU64(x);
    std::uint32_t abstop = static_cast<std::uint32_t>(ix >> 52) & 0x7ffu;
    if (abstop - 0x3c9u >= 0x3fu)
    {
        if (static_cast<int>(abstop - 0x3c9u) < 0)
        {
            // |x| < 2^-54
            return x + 1.0;
        }
        if (abstop >= 0x409u)
        {
            if (ix == 0xfff0000000000000ULL)
            {
                return 0.0;
            }
            if (abstop >= 0x7ffu)
            {
                return x + 1.0;
            }
            return (ix >> 63) ? xflow(0, 0x1p-767) : xflow(0, 0x1p769);
        }
        abstop = 0;
    }

    const double shift = expTab(1);
    double kd = fma(x, expTab(0), shift);
    const std::uint64_t ki = asU64(kd);
    kd = kd - shift;
    double r = fma(kd, expTab(2), x);
    r = fma(kd, expTab(3), r);
    const int idx = 2*static_cast<int>(ki & 127);
    const std::uint64_t top = ki << 45;
    const double tail = expTab(22 + idx);
    const std::uint64_t sbits = glibcPowData::exp[22 + idx + 1] + top;

    const double p23 = fma(r, expTab(5), expTab(4));
    const double rt = r + tail;
    const double r2 = r*r;
    const double p45 = fma(r, expTab(7), expTab(6));
    const double s = fma(p23, r2, rt);
    const double tmp = fma(p45, r2*r2, s);
    if (abstop == 0)
    {
        return specialcase(tmp, sbits, ki);
    }
    const double scale = asDouble(sbits);
    return fma(tmp, scale, scale);
}

} // namespace gnc

#endif
