// tanh(x) 與 expm1(x) 的 device 版，與 x86-64 glibc 2.39 的 __tanh、__expm1_fma
// 逐位元相同。GPU-Native-CFD，GPL-3.0-or-later；演算法與常數出自 GNU C Library
// （sysdeps/ieee754/dbl-64/s_tanh.c、s_expm1.c，LGPL-2.1-or-later；源自 fdlibm）。
//
// 為什麼需要：kOmegaSST 的 F1 = tanh(pow4(arg1))、F2 = tanh(sqr(arg2))（N4b）。
// OpenFOAM 的 Foam::tanh 是 ::tanh；glibc 的 __tanh 只有一個版本（沒有 IFUNC），
// 但它經 PLT 呼叫 __expm1，在支援 FMA 與 AVX2 的 CPU 上由 IFUNC 選到
// __expm1_fma（s_expm1.c 以 -mfma -mavx2 編譯，GCC 合併了部分 a*b + c）。
//
// 怎麼來的：作法同 core/glibcPow.cuh（N-018）。本機 libm.so.6（Ubuntu libc6
// 2.39-0ubuntu8.9，sha256 fce00b6f…，g02 與 WSL 相同）以 libc6-dbg 的符號在 gdb
// 中反組譯 __tanh 與 __expm1_fma（行號對到原始檔），逐條照機器碼寫出；每個 fma()
// 對應一條 vfmadd／vfmsub／vfnmadd，常數為 x/gx 讀出的原始位元。
// 驗證：tests/gncPowCheck --func tanh|expm1 --impl glibc 對主機的 libm 逐位元比較。
// 必須以 --fmad=false 編譯（不得有其他的 FMA 合併）。

#ifndef GNC_CORE_GLIBCTANH_CUH
#define GNC_CORE_GLIBCTANH_CUH

#include <cstdint>

namespace gnc
{
namespace glibcTanhDetail
{

__device__ __forceinline__ double bitsToDouble(const std::uint64_t u)
{
    return __longlong_as_double(static_cast<long long>(u));
}

__device__ __forceinline__ std::uint64_t doubleToBits(const double d)
{
    return static_cast<std::uint64_t>(__double_as_longlong(d));
}

// 高位字加上 k << 20（SET_HIGH_WORD(y, high + (k << 20))，32 位元整數加法）
__device__ __forceinline__ double addExponent(const double y, const int k)
{
    const std::uint64_t b = doubleToBits(y);
    const std::uint32_t hi =
        static_cast<std::uint32_t>(b >> 32) + (static_cast<std::uint32_t>(k) << 20);
    return bitsToDouble((static_cast<std::uint64_t>(hi) << 32) | (b & 0xffffffffULL));
}

// s_expm1.c 的常數（libm .rodata 的原始位元）
__device__ __forceinline__ double oThreshold() { return bitsToDouble(0x40862e42fefa39efULL); }
__device__ __forceinline__ double ln2Hi() { return bitsToDouble(0x3fe62e42fee00000ULL); }
__device__ __forceinline__ double ln2Lo() { return bitsToDouble(0x3dea39ef35793c76ULL); }
__device__ __forceinline__ double invLn2() { return bitsToDouble(0x3ff71547652b82feULL); }
__device__ __forceinline__ double Q1() { return bitsToDouble(0xbfa11111111110f4ULL); }
__device__ __forceinline__ double Q2() { return bitsToDouble(0x3f5a01a019fe5585ULL); }
__device__ __forceinline__ double Q3() { return bitsToDouble(0xbf14ce199eaadbb7ULL); }
__device__ __forceinline__ double Q4() { return bitsToDouble(0x3ed0cfca86e65239ULL); }
__device__ __forceinline__ double Q5() { return bitsToDouble(0xbe8afdb76e09c32dULL); }
__device__ __forceinline__ double huge() { return bitsToDouble(0x7e37e43c8800759cULL); }   // 1e300
__device__ __forceinline__ double tiny() { return bitsToDouble(0x01a56e1fc2f8f359ULL); }   // 1e-300

} // namespace glibcTanhDetail


// __expm1_fma（s_expm1.c:139–265）
__device__ inline double glibcExpm1(double x)
{
    using namespace glibcTanhDetail;
    const std::uint64_t ix = doubleToBits(x);
    const std::uint32_t high = static_cast<std::uint32_t>(ix >> 32);
    const std::uint32_t xsb = high & 0x80000000u;
    const std::uint32_t hx = high & 0x7fffffffu;

    // 過大與非有限的引數（:152–176）
    if (hx >= 0x4043687Au)
    {
        if (hx >= 0x40862E42u)
        {
            if (hx >= 0x7ff00000u)
            {
                if (((high & 0xfffffu) | static_cast<std::uint32_t>(ix)) != 0)
                {
                    return x + x;
                }
                return xsb == 0 ? x : -1.0;
            }
            if (x > oThreshold())
            {
                return huge()*huge();
            }
        }
        if (xsb != 0)
        {
            return tiny() - 1.0;
        }
    }

    // 引數化簡（:181–211）
    double hi, lo, c = 0.0;
    int k;
    if (hx > 0x3fd62e42u)
    {
        if (hx < 0x3FF0A2B2u)
        {
            if (xsb == 0)
            {
                hi = x - ln2Hi();
                lo = ln2Lo();
                k = 1;
            }
            else
            {
                hi = x + ln2Hi();
                lo = -ln2Lo();
                k = -1;
            }
        }
        else
        {
            k = static_cast<int>(invLn2()*x + (xsb == 0 ? 0.5 : -0.5));
            const double t = static_cast<double>(k);
            hi = fma(-t, ln2Hi(), x);
            lo = t*ln2Lo();
        }
        x = hi - lo;
        c = (hi - x) - lo;
    }
    else if (hx < 0x3c900000u)
    {
        // x - (t - (huge + x))，t = huge + x：GCC 把括號內算成 t - t = 0
        return x - 0.0;
    }
    else
    {
        k = 0;
    }

    // 主要範圍（:212–223）
    const double hfx = 0.5*x;
    const double hxs = x*hfx;
    const double R1 = fma(hxs, Q1(), 1.0);
    const double R2 = fma(hxs, Q3(), Q2());
    const double R3 = fma(hxs, Q5(), Q4());
    const double h2 = hxs*hxs;
    const double h4 = h2*h2;
    const double r1 = fma(h4, R3, fma(h2, R2, R1));
    const double t = fma(-r1, hfx, 3.0);
    double e = ((r1 - t)/fma(-x, t, 6.0))*hxs;
    if (k == 0)
    {
        return x - fma(e, x, -hxs);
    }

    // :224–263
    e = fma(e - c, x, -c);
    e = e - hxs;
    if (k == -1)
    {
        return fma(0.5, x - e, -0.5);
    }
    if (k == 1)
    {
        if (x < -0.25)
        {
            return -2.0*(e - (x + 0.5));
        }
        return fma(x - e, 2.0, 1.0);
    }
    if (k <= -2 || k > 56)
    {
        const double y = 1.0 - (e - x);
        return addExponent(y, k) - 1.0;
    }
    if (k < 20)
    {
        const double t1 = bitsToDouble
        (
            static_cast<std::uint64_t>(0x3ff00000u - (0x200000u >> k)) << 32
        );
        return addExponent(t1 - (e - x), k);
    }
    const double t2 = bitsToDouble(static_cast<std::uint64_t>((0x3ffu - k) << 20) << 32);
    return addExponent((x - (e + t2)) + 1.0, k);
}


// __tanh（s_tanh.c:50–95）
__device__ inline double glibcTanh(const double x)
{
    using namespace glibcTanhDetail;
    const std::uint64_t ix = doubleToBits(x);
    const std::int32_t jx = static_cast<std::int32_t>(ix >> 32);
    const std::uint32_t hx = static_cast<std::uint32_t>(jx) & 0x7fffffffu;

    if (hx >= 0x7ff00000u)
    {
        return jx >= 0 ? 1.0/x + 1.0 : 1.0/x - 1.0;
    }
    double z;
    if (hx < 0x40360000u)
    {
        if ((hx | static_cast<std::uint32_t>(ix)) == 0)
        {
            return x;
        }
        if (hx < 0x3c800000u)
        {
            return x*(1.0 + x);
        }
        const double ax = ::fabs(x);
        if (hx >= 0x3ff00000u)
        {
            const double t = glibcExpm1(ax + ax);
            z = 1.0 - 2.0/(t + 2.0);
        }
        else
        {
            const double t = glibcExpm1(-2.0*ax);
            z = -t/(t + 2.0);
        }
    }
    else
    {
        z = 1.0 - tiny();
    }
    return jx >= 0 ? z : -z;
}

} // namespace gnc

#endif
