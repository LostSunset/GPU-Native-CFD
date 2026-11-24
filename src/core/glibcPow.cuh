// pow(x, y) 的 device 版，與 x86-64 glibc 2.39 的 __ieee754_pow_fma 逐位元相同。
// GPU-Native-CFD，GPL-3.0-or-later；演算法與表格出自 GNU C Library
// （sysdeps/ieee754/dbl-64/e_pow.c，LGPL-2.1-or-later；源自 ARM optimized-routines）。
//
// 為什麼需要：OpenFOAM 的 Foam::pow(scalar, scalar) 是 ::pow，在 g02 與 WSL 上由
// libm 的 IFUNC 選到 __ieee754_pow_fma（CPU 支援 FMA 與 AVX2）。CUDA 的 pow
// 與它在 Spalart–Allmaras fw 的底數範圍 (0, 65/64] 內約 9% 的值差 1 ulp
// （tests/gncPowCheck）。
//
// 怎麼來的：沒有下載原始碼。以本機的 libm.so.6（Ubuntu libc6 2.39-0ubuntu8.9，
// sha256 fce00b6f…，g02 與 WSL 相同）與 libc6-dbg 的除錯符號，在 gdb 中反組譯
// __ieee754_pow_fma（行號對到 e_pow.c），逐條照機器碼寫出：GCC 對這個檔案啟用了
// FMA 合併，哪些 a*b + c 合成一個 vfmadd、哪些分開算，都以機器碼為準（下面每個
// fma() 對應一條 vfmadd／vfmsub）。表格在 core/glibcPowData.cuh。
// 驗證：tests/gncPowCheck --glibc 對主機的 pow 逐位元比較。
// 必須以 --fmad=false 編譯（不得有其他的 FMA 合併）。

#ifndef GNC_CORE_GLIBCPOW_CUH
#define GNC_CORE_GLIBCPOW_CUH

#include "core/glibcPowData.cuh"

#include <cstdint>

namespace gnc
{
namespace glibcPowDetail
{

__device__ __forceinline__ double asDouble(const std::uint64_t u)
{
    return __longlong_as_double(static_cast<long long>(u));
}

__device__ __forceinline__ std::uint64_t asU64(const double d)
{
    return static_cast<std::uint64_t>(__double_as_longlong(d));
}

__device__ __forceinline__ double logTab(const int i)
{
    return asDouble(glibcPowData::powLog[i]);
}

__device__ __forceinline__ double expTab(const int i)
{
    return asDouble(glibcPowData::exp[i]);
}

// pow_log_data：ln2hi [0]、ln2lo [1]、poly A[0..6] [2..8]、
// tab[i]：invc [9 + 4i]、logc [11 + 4i]、logctail [12 + 4i]
// exp_data：invln2N [0]、shift [1]、negln2hiN [2]、negln2loN [3]、
// poly C2..C5 [4..7]、tab [22 + j]

// log_inline（e_pow.c:45–116）：回傳 hi，*tail = lo
__device__ __forceinline__ double logInline(const std::uint64_t ix, double* tail)
{
    const std::uint64_t tmp = ix - 0x3fe6955500000000ULL;           // OFF
    const int i = static_cast<int>((tmp >> 45) & 127);              // N = 128
    const int k = static_cast<int>(static_cast<std::int64_t>(tmp) >> 52);
    const std::uint64_t iz = ix - (tmp & 0xfff0000000000000ULL);
    const double z = asDouble(iz);
    const double kd = static_cast<double>(k);

    const double invc = logTab(9 + 4*i);
    const double logc = logTab(11 + 4*i);
    const double logctail = logTab(12 + 4*i);

    const double r = fma(z, invc, -1.0);
    const double t1 = fma(kd, logTab(0), logc);
    const double lo1 = fma(kd, logTab(1), logctail);
    const double t2 = r + t1;
    const double lo2 = (t1 - t2) + r;

    const double ar = r*logTab(2);          // A[0] = -0.5
    const double ar2 = r*ar;
    const double ar3 = r*ar2;
    const double hi = t2 + ar2;
    const double lo3 = fma(ar, r, -ar2);
    const double lo4 = (t2 - hi) + ar2;

    const double q1 = fma(r, logTab(4), logTab(3));     // r*A[2] + A[1]
    const double q3 = fma(r, logTab(6), logTab(5));     // r*A[4] + A[3]
    const double q5 = fma(r, logTab(8), logTab(7));     // r*A[6] + A[5]
    const double t = fma(q5, ar2, q3);
    const double u = fma(ar2, t, q1);
    const double lo = fma(ar3, u, ((lo1 + lo2) + lo3) + lo4);

    const double y = hi + lo;
    *tail = (hi - y) + lo;
    return y;
}

// __math_oflow、__math_uflow（xflow：(sign ? -y : y)*y）
__device__ __forceinline__ double xflow(const std::uint32_t sign, const double y)
{
    return (sign ? -y : y)*y;
}

// specialcase（e_pow.c:138–176）：|x| 大到 2^k 的指數可能溢出時
__device__ __forceinline__ double expSpecialcase(const double tmp, std::uint64_t sbits, const std::uint64_t ki)
{
    if ((ki & 0x80000000ULL) == 0)
    {
        // k > 0：指數可能溢出，先縮 2^-1009 再乘回
        sbits -= 1009ULL << 52;
        const double scale = asDouble(sbits);
        const double y = fma(tmp, scale, scale);
        return 0x1p1009*y;
    }
    // k < 0：避免次正規範圍內的兩次捨入
    sbits += 1022ULL << 52;
    const double scale = asDouble(sbits);
    const double ts = tmp*scale;
    double y = scale + ts;
    if (1.0 > fabs(y))
    {
        const double one = (y < 0.0) ? -1.0 : 1.0;
        const double lo = (scale - y) + ts;
        const double hi = y + one;
        y = ((((one - hi) + y) + lo) + hi) - one;
        if (y == 0.0)
        {
            y = asDouble(sbits & 0x8000000000000000ULL);
        }
    }
    return y*0x1p-1022;
}

// exp_inline（e_pow.c:180–250），x = ehi、xtail = elo
__device__ __forceinline__ double expInline(const double x, const double xtail, const std::uint32_t signBias)
{
    std::uint32_t abstop = static_cast<std::uint32_t>(asU64(x) >> 52) & 0x7ff;
    if (abstop - 0x3c9u >= 0x3fu)
    {
        if (static_cast<int>(abstop - 0x3c9u) < 0)
        {
            // |x| < 2^-54
            const double one = x + 1.0;
            return signBias ? -one : one;
        }
        if (abstop >= 0x409u)
        {
            // |x| >= 1024
            if (asU64(x) >> 63)
            {
                return xflow(signBias, 0x1p-767);
            }
            return xflow(signBias, 0x1p769);
        }
        abstop = 0;
    }

    const double shift = expTab(1);
    double kd = fma(x, expTab(0), shift);
    const std::uint64_t ki = asU64(kd);
    kd = kd - shift;
    double r = fma(kd, expTab(2), x);
    r = fma(kd, expTab(3), r);
    r = xtail + r;
    const int idx = 2*static_cast<int>(ki & 127);
    const std::uint64_t top = (ki + signBias) << 45;
    const double tail = expTab(22 + idx);
    const std::uint64_t sbits = glibcPowData::exp[22 + idx + 1] + top;

    const double p23 = fma(r, expTab(5), expTab(4));    // r*C3 + C2
    const double rt = r + tail;
    const double r2 = r*r;
    const double p45 = fma(r, expTab(7), expTab(6));    // r*C5 + C4
    const double s = fma(p23, r2, rt);
    const double tmp = fma(p45, r2*r2, s);
    if (abstop == 0)
    {
        return expSpecialcase(tmp, sbits, ki);
    }
    const double scale = asDouble(sbits);
    return fma(tmp, scale, scale);
}

// checkint（e_pow.c:255–266）：0 不是整數、1 奇數、2 偶數
__device__ __forceinline__ int checkint(const std::uint64_t iy)
{
    const int e = static_cast<int>(iy >> 52) & 0x7ff;
    if (e < 0x3ff)
    {
        return 0;
    }
    if (e > 0x3ff + 52)
    {
        return 2;
    }
    if (iy & ((1ULL << (0x3ff + 52 - e)) - 1))
    {
        return 0;
    }
    if (iy & (1ULL << (0x3ff + 52 - e)))
    {
        return 1;
    }
    return 2;
}

__device__ __forceinline__ bool zeroinfnan(const std::uint64_t i)
{
    return 2*i - 1 >= 2*0x7ff0000000000000ULL - 1;
}

__device__ __forceinline__ bool issignaling(const double x)
{
    return 2*(asU64(x) ^ 0x0008000000000000ULL) > 2*0x7ff8000000000000ULL;
}

} // namespace glibcPowDetail


// __ieee754_pow_fma（e_pow.c:280–372）
__device__ inline double glibcPow(const double x, const double y)
{
    using namespace glibcPowDetail;
    std::uint32_t signBias = 0;
    std::uint64_t ix = asU64(x);
    const std::uint64_t iy = asU64(y);
    std::uint32_t topx = static_cast<std::uint32_t>(ix >> 52);
    const std::uint32_t topy = static_cast<std::uint32_t>(iy >> 52);

    if (topx - 0x001u >= 0x7feu || (topy & 0x7ffu) - 0x3beu >= 0x80u)
    {
        if (zeroinfnan(iy))
        {
            if (2*iy == 0)
            {
                return issignaling(x) ? x + y : 1.0;
            }
            if (ix == 0x3ff0000000000000ULL)
            {
                return issignaling(y) ? x + y : 1.0;
            }
            if (2*ix > 2*0x7ff0000000000000ULL || 2*iy > 2*0x7ff0000000000000ULL)
            {
                return x + y;
            }
            if (2*ix == 2*0x3ff0000000000000ULL)
            {
                return 1.0;
            }
            if ((2*ix < 2*0x3ff0000000000000ULL) == !(iy >> 63))
            {
                return 0.0;
            }
            return y*y;
        }
        if (zeroinfnan(ix))
        {
            double x2 = x*x;
            if ((ix >> 63) && checkint(iy) == 1)
            {
                x2 = -x2;
                signBias = 1;
            }
            if (2*ix == 0 && (iy >> 63))
            {
                // __math_divzero：±1/0 = ±inf（直接寫出位元，避免編譯期除以零的警告）
                return asDouble(signBias ? 0xfff0000000000000ULL : 0x7ff0000000000000ULL);
            }
            return (iy >> 63) ? 1.0/x2 : x2;
        }
        if (ix >> 63)
        {
            const int yint = checkint(iy);
            if (yint == 0)
            {
                // __math_invalid
                return (x - x)/(x - x);
            }
            if (yint == 1)
            {
                signBias = 0x800u << 7;     // SIGN_BIAS
            }
            ix &= 0x7fffffffffffffffULL;
            topx &= 0x7ffu;
        }
        if ((topy & 0x7ffu) - 0x3beu >= 0x80u)
        {
            if (ix == 0x3ff0000000000000ULL)
            {
                return 1.0;
            }
            if ((topy & 0x7ffu) < 0x3beu)
            {
                return ix > 0x3ff0000000000000ULL ? 1.0 + y : 1.0 - y;
            }
            return (ix > 0x3ff0000000000000ULL) == (topy < 0x800u)
                ? xflow(0, 0x1p769) : xflow(0, 0x1p-767);
        }
        if (topx == 0)
        {
            // 次正規的 x
            ix = asU64(x*0x1p52);
            ix &= 0x7fffffffffffffffULL;
            ix -= 52ULL << 52;
        }
    }

    double lo;
    const double hi = logInline(ix, &lo);
    const double ehi = y*hi;
    const double elo = fma(y, lo, fma(hi, y, -ehi));
    return expInline(ehi, elo, signBias);
}

} // namespace gnc

#endif
