// log(x) 的 device 版，與 x86-64 glibc 2.39 的 __ieee754_log_fma 逐位元相同。
// GPU-Native-CFD，GPL-3.0-or-later；演算法與表格出自 GNU C Library
// （sysdeps/ieee754/dbl-64/e_log.c，LGPL-2.1-or-later；源自 ARM optimized-routines）。
//
// 為什麼需要：kOmegaSST 的壁面函數 nutkWallFunction 的 log(max(E*yPlus, 1 + 1e-4))、
// kLowReWallFunction 的 log(yPlus)（N4b）。OpenFOAM 的 Foam::log 是 ::log，在支援
// FMA 與 AVX2 的 CPU 上由 IFUNC 選到 __ieee754_log_fma。
//
// 怎麼來的：作法同 core/glibcPow.cuh（N-018）。本機 libm.so.6（Ubuntu libc6
// 2.39-0ubuntu8.9，sha256 fce00b6f…，g02 與 WSL 相同）以 libc6-dbg 的符號在 gdb
// 中反組譯 __ieee754_log_fma（行號對到 e_log.c），逐條照機器碼寫出；每個 fma()
// 對應一條 vfmadd／vfnmadd。表格在 core/glibcLogData.cuh。
// 驗證：tests/gncPowCheck --func log --impl glibc 對主機的 libm 逐位元比較。
// 必須以 --fmad=false 編譯。

#ifndef GNC_CORE_GLIBCLOG_CUH
#define GNC_CORE_GLIBCLOG_CUH

#include "core/glibcLogData.cuh"

#include <cstdint>

namespace gnc
{
namespace glibcLogDetail
{

__device__ __forceinline__ double bitsToDouble(const std::uint64_t u)
{
    return __longlong_as_double(static_cast<long long>(u));
}

__device__ __forceinline__ std::uint64_t doubleToBits(const double d)
{
    return static_cast<std::uint64_t>(__double_as_longlong(d));
}

__device__ __forceinline__ double tab(const int i)
{
    return bitsToDouble(glibcLogData::log[i]);
}

// log_data：ln2hi、ln2lo、A[0..4]、B[0..10]、tab[i].invc、tab[i].logc
__device__ __forceinline__ double A(const int i) { return tab(2 + i); }
__device__ __forceinline__ double B(const int i) { return tab(7 + i); }

} // namespace glibcLogDetail


// __ieee754_log_fma（e_log.c:39–131）
__device__ inline double glibcLog(const double x)
{
    using namespace glibcLogDetail;
    std::uint64_t ix = doubleToBits(x);
    const std::uint32_t top = static_cast<std::uint32_t>(ix >> 48);

    // 接近 1（:61–83）：|x - 1| 約 < 0x1p-4
    if (ix - 0x3fee000000000000ULL < 0x3090000000000ULL)
    {
        if (ix == 0x3ff0000000000000ULL)
        {
            return 0.0;
        }
        const double r = x - 1.0;
        const double r2 = r*r;
        const double r3 = r*r2;
        const double p1 = fma(r2, B(3), fma(r, B(2), B(1)));
        const double p2 = fma(r2, B(6), fma(r, B(5), B(4)));
        const double p3 = fma(r3, B(10), fma(r2, B(9), fma(r, B(8), B(7))));
        const double p = fma(fma(p3, r3, p2), r3, p1);
        // w = r*0x1p27；rhi = r + w - w（兩步都與乘法合併）
        const double rw = fma(r, 0x1p27, r);
        const double rhi = fma(-0x1p27, r, rw);
        const double rlo = r - rhi;
        const double rhi2 = rhi*rhi;
        const double hi = fma(rhi2, B(0), r);
        double lo = fma(rhi2, B(0), r - hi);
        lo = fma(B(0)*rlo, r + rhi, lo);
        const double y = fma(p, r3, lo);
        return hi + y;
    }

    // x < 0x1p-1022、Inf、NaN（:86–96）
    if (top - 0x0010u >= 0x7ff0u - 0x0010u)
    {
        if (ix*2 == 0)
        {
            // __math_divzero(1)
            return bitsToDouble(0xfff0000000000000ULL);
        }
        if (ix == 0x7ff0000000000000ULL)
        {
            return x;
        }
        if ((top & 0x8000u) || (top & 0x7ff0u) == 0x7ff0u)
        {
            // __math_invalid(x)：(x - x)/(x - x)
            return x != x ? x : bitsToDouble(0xfff8000000000000ULL);
        }
        // 次正規數：正規化
        ix = doubleToBits(x*0x1p52);
        ix -= 52ULL << 52;
    }

    // :97–130
    const std::uint64_t tmp = ix - 0x3fe6000000000000ULL;
    const int i = static_cast<int>((tmp >> 45) & 127);
    const int k = static_cast<int>(static_cast<std::int64_t>(tmp) >> 52);
    const std::uint64_t iz = ix - (tmp & 0xfff0000000000000ULL);
    const double invc = tab(18 + 2*i);
    const double logc = tab(19 + 2*i);
    const double z = bitsToDouble(iz);
    const double r = fma(z, invc, -1.0);
    const double kd = static_cast<double>(k);
    const double w = fma(kd, tab(0), logc);
    const double hi = w + r;
    const double lo = fma(kd, tab(1), (w - hi) + r);
    const double r2 = r*r;
    const double q = fma(fma(r, A(4), A(3)), r2, fma(r, A(2), A(1)));
    return fma(r*r2, q, fma(r2, A(0), lo)) + hi;
}

} // namespace gnc

#endif
