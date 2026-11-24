// GPU-Native-CFD，GPL-3.0-or-later。見 hisaLinearSolver.cuh。

#include "solvers/gncHisa/hisaLinearSolver.cuh"

#include "core/coopLevels.cuh"
#include "core/reduce.cuh"
#include "fields/boundaryConditions.cuh"
#include "io/dictionary.hpp"
#include "io/foamTokenizer.hpp"

#include <cooperative_groups.h>

#include <algorithm>
#include <memory>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdexcept>

namespace gnc
{

// GNC_PROFILE_GMRES=1 時累計 GMRES 內各部分的牆鐘時間（每次呼叫後同步；N7 的剖析）
GmresProfile& gmresProfile()
{
    static GmresProfile p;
    return p;
}

namespace
{

constexpr int BS = 256;

bool profileGmres()
{
    static const bool on = []
    {
        const char* e = std::getenv("GNC_PROFILE_GMRES");
        return e && std::string(e) == "1";
    }();
    return on;
}

struct ScopedLap
{
    double* acc;
    std::chrono::steady_clock::time_point t0;

    explicit ScopedLap(double& a)
    :
        acc(profileGmres() ? &a : nullptr)
    {
        if (acc)
        {
            GNC_CUDA_CHECK(cudaDeviceSynchronize());
            t0 = std::chrono::steady_clock::now();
        }
    }

    ~ScopedLap()
    {
        if (acc)
        {
            cudaDeviceSynchronize();
            *acc += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        }
    }
};

constexpr double GREAT = 1.0e+15;
constexpr double VGREAT = 1.0e+300;

int nBlocks(const std::int32_t n)
{
    return int((n + BS - 1)/BS);
}

__host__ __device__ double stabiliseH(const double s, const double tol)
{
    return (s >= 0) ? s + tol : s - tol;
}

// OpenFOAM 純量 max：(b < a) ? a : b
double maxH(const double a, const double b)
{
    return (b < a) ? a : b;
}


// ---------------------------------------------------------------------------
// LU-SGS

// S 為儲存型別：double（HiSA 的 Jacobian）或 float（效能版 GNC_JAC=float，混合精度：
// 以 FP32 儲存、讀出後以 FP64 運算）
template<class S>
struct JacViewT
{
    // 對角
    const S* SS00d; const S* SS01d; const S* SS10d; const S* SS11d;
    CView<vec3, S> SV00d, SV10d, VS00d, VS01d;
    CView<tens, S> VV00d;
    // 非對角
    const S* SS00u; const S* SS00l;
    const S* SS10u; const S* SS10l;
    const S* SS11u; const S* SS11l;
    CView<vec3, S> SV00u, SV00l, SV10u, SV10l, VS00u, VS00l, VS01u, VS01l;
    CView<tens, S> VV00u, VV00l;
};

using JacView = JacViewT<double>;

JacView jacView(const HisaJacobian& J)
{
    JacView v{};
    v.SS00d = J.SS00.diag.cview().p[0];
    v.SS01d = J.SS01.diag.cview().p[0];
    v.SS10d = J.SS10.diag.cview().p[0];
    v.SS11d = J.SS11.diag.cview().p[0];
    v.SV00d = J.SV00.diag.cview();
    v.SV10d = J.SV10.diag.cview();
    v.VS00d = J.VS00.diag.cview();
    v.VS01d = J.VS01.diag.cview();
    v.VV00d = J.VV00.diag.cview();
    v.SS00u = J.SS00.upper.cview().p[0]; v.SS00l = J.SS00.lower.cview().p[0];
    v.SS10u = J.SS10.upper.cview().p[0]; v.SS10l = J.SS10.lower.cview().p[0];
    v.SS11u = J.SS11.upper.cview().p[0]; v.SS11l = J.SS11.lower.cview().p[0];
    v.SV00u = J.SV00.upper.cview(); v.SV00l = J.SV00.lower.cview();
    v.SV10u = J.SV10.upper.cview(); v.SV10l = J.SV10.lower.cview();
    v.VS00u = J.VS00.upper.cview(); v.VS00l = J.VS00.lower.cview();
    v.VS01u = J.VS01.upper.cview(); v.VS01l = J.VS01.lower.cview();
    v.VV00u = J.VV00.upper.cview(); v.VV00l = J.VV00.lower.cview();
    return v;
}

struct VecView
{
    double* s0;
    double* s1;
    DView<vec3> v;
};

VecView vecView(HisaVec& x)
{
    return {x.s0.view().p[0], x.s1.view().p[0], x.v.view()};
}

struct CVecView
{
    const double* s0;
    const double* s1;
    CView<vec3> v;
};

CVecView cvecView(const HisaVec& x)
{
    return {x.s0.cview().p[0], x.s1.cview().p[0], x.v.cview()};
}


// x *= V
__global__ void kMulV(const std::int32_t n, const double* __restrict__ V, const VecView x)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < n)
    {
        x.s0[c] = x.s0[c]*V[c];
        x.s1[c] = x.s1[c]*V[c];
        x.v.set(c, x.v[c]*V[c]);
    }
}

// 正向：本層的格 c 收集編號較小的鄰格（lower 面）的貢獻。面依編號遞增，
// 同一 owner 的面連續；每個來源格依子區塊次序，各子區塊內依面號。
template<class JV>
__device__ __forceinline__ void lusgsForwardCell
(
    const std::int32_t c,
    const std::int32_t* __restrict__ off,
    const std::int32_t* __restrict__ faces,
    const std::int32_t* __restrict__ owner,
    const double* __restrict__ rD,
    const JV& J,
    const VecView& x
)
{
    double r0 = x.s0[c], r1 = x.s1[c];
    vec3 rv = x.v[c];
    std::int32_t q = off[c];
    const std::int32_t qe = off[c + 1];
    while (q < qe)
    {
        const std::int32_t i = owner[faces[q]];
        std::int32_t g = q;
        while (g < qe && owner[faces[g]] == i)
        {
            ++g;
        }
        // 來源格 i 的 d* = rDiagCoeff*x（x 已是正向的最終值）
        const double d0 = rD[i]*x.s0[i];
        const double d1 = rD[i]*x.s1[i];
        const vec3 dv = rD[i]*x.v[i];
        // sVec[0]：SS00、（SS01 沒有非對角）、SV00
        for (std::int32_t k = q; k < g; ++k) r0 = r0 - J.SS00l[faces[k]]*d0;
        // sVec[1]：SS10、SS11、SV10（SS 在 SV 之前）
        for (std::int32_t k = q; k < g; ++k) r1 = r1 - J.SS10l[faces[k]]*d0;
        for (std::int32_t k = q; k < g; ++k) r1 = r1 - J.SS11l[faces[k]]*d1;
        for (std::int32_t k = q; k < g; ++k) r0 = r0 - dot(J.SV00l[faces[k]], dv);
        for (std::int32_t k = q; k < g; ++k) r1 = r1 - dot(J.SV10l[faces[k]], dv);
        // vVec[0]：VS00、VS01、VV00
        for (std::int32_t k = q; k < g; ++k) rv = rv - J.VS00l[faces[k]]*d0;
        for (std::int32_t k = q; k < g; ++k) rv = rv - J.VS01l[faces[k]]*d1;
        for (std::int32_t k = q; k < g; ++k) rv = rv - dot(J.VV00l[faces[k]], dv);
        q = g;
    }
    x.s0[c] = r0;
    x.s1[c] = r1;
    x.v.set(c, rv);
}

template<class JV>
__global__ void kLusgsForward
(
    const std::int32_t count,
    const std::int32_t* __restrict__ cells,
    const std::int32_t* __restrict__ off,
    const std::int32_t* __restrict__ faces,
    const std::int32_t* __restrict__ owner,
    const double* __restrict__ rD,
    const JV J,
    const VecView x
)
{
    const std::int32_t t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t < count)
    {
        lusgsForwardCell(cells[t], off, faces, owner, rD, J, x);
    }
}

// 反向：本層的格 c 收集編號較大的鄰格（upper 面）的貢獻（來源格遞減），
// 再 x_c = rDiagCoeff*x_c
template<class JV>
__device__ __forceinline__ void lusgsReverseCell
(
    const std::int32_t c,
    const std::int32_t* __restrict__ off,
    const std::int32_t* __restrict__ faces,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ rD,
    const JV& J,
    const VecView& x
)
{
    double r0 = x.s0[c], r1 = x.s1[c];
    vec3 rv = x.v[c];
    std::int32_t q = off[c];
    const std::int32_t qe = off[c + 1];
    while (q < qe)
    {
        const std::int32_t j = neighbour[faces[q]];
        std::int32_t g = q;
        while (g < qe && neighbour[faces[g]] == j)
        {
            ++g;
        }
        // 來源格 j 的值已是反向的最終值（已乘 rDiagCoeff）
        const double d0 = x.s0[j];
        const double d1 = x.s1[j];
        const vec3 dv = x.v[j];
        for (std::int32_t k = q; k < g; ++k) r0 = r0 - J.SS00u[faces[k]]*d0;
        for (std::int32_t k = q; k < g; ++k) r1 = r1 - J.SS10u[faces[k]]*d0;
        for (std::int32_t k = q; k < g; ++k) r1 = r1 - J.SS11u[faces[k]]*d1;
        for (std::int32_t k = q; k < g; ++k) r0 = r0 - dot(J.SV00u[faces[k]], dv);
        for (std::int32_t k = q; k < g; ++k) r1 = r1 - dot(J.SV10u[faces[k]], dv);
        for (std::int32_t k = q; k < g; ++k) rv = rv - J.VS00u[faces[k]]*d0;
        for (std::int32_t k = q; k < g; ++k) rv = rv - J.VS01u[faces[k]]*d1;
        for (std::int32_t k = q; k < g; ++k) rv = rv - dot(J.VV00u[faces[k]], dv);
        q = g;
    }
    x.s0[c] = rD[c]*r0;
    x.s1[c] = rD[c]*r1;
    x.v.set(c, rD[c]*rv);
}

template<class JV>
__global__ void kLusgsReverse
(
    const std::int32_t count,
    const std::int32_t* __restrict__ cells,
    const std::int32_t* __restrict__ off,
    const std::int32_t* __restrict__ faces,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ rD,
    const JV J,
    const VecView x
)
{
    const std::int32_t t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t < count)
    {
        lusgsReverseCell(cells[t], off, faces, neighbour, rD, J, x);
    }
}

// 多色 LU-SGS（效能版）：f 的編碼見 LusgsPlan::cLowFace；other 為相鄰格
template<class JV>
__device__ __forceinline__ void colorBlockSub
(
    const std::int32_t f,
    const bool ownerSide,
    const JV& J,
    const double d0,
    const double d1,
    const vec3& dv,
    double& r0,
    double& r1,
    vec3& rv
)
{
    if (ownerSide)
    {
        r0 = r0 - J.SS00u[f]*d0;
        r1 = r1 - J.SS10u[f]*d0;
        r1 = r1 - J.SS11u[f]*d1;
        r0 = r0 - dot(J.SV00u[f], dv);
        r1 = r1 - dot(J.SV10u[f], dv);
        rv = rv - J.VS00u[f]*d0;
        rv = rv - J.VS01u[f]*d1;
        rv = rv - dot(J.VV00u[f], dv);
    }
    else
    {
        r0 = r0 - J.SS00l[f]*d0;
        r1 = r1 - J.SS10l[f]*d0;
        r1 = r1 - J.SS11l[f]*d1;
        r0 = r0 - dot(J.SV00l[f], dv);
        r1 = r1 - dot(J.SV10l[f], dv);
        rv = rv - J.VS00l[f]*d0;
        rv = rv - J.VS01l[f]*d1;
        rv = rv - dot(J.VV00l[f], dv);
    }
}

template<class JV>
__global__ void kLusgsColorForward
(
    const std::int32_t count,
    const std::int32_t* __restrict__ cells,
    const std::int32_t* __restrict__ off,
    const std::int32_t* __restrict__ faceCode,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ rD,
    const JV J,
    const VecView x
)
{
    const std::int32_t t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= count)
    {
        return;
    }
    const std::int32_t c = cells[t];
    double r0 = x.s0[c], r1 = x.s1[c];
    vec3 rv = x.v[c];
    for (std::int32_t q = off[c]; q < off[c + 1]; ++q)
    {
        const std::int32_t code = faceCode[q];
        const bool ownerSide = code < 0;
        const std::int32_t f = ownerSide ? -(code + 1) : code;
        const std::int32_t j = ownerSide ? neighbour[f] : owner[f];
        const double d0 = rD[j]*x.s0[j];
        const double d1 = rD[j]*x.s1[j];
        const vec3 dv = rD[j]*x.v[j];
        colorBlockSub(f, ownerSide, J, d0, d1, dv, r0, r1, rv);
    }
    x.s0[c] = r0;
    x.s1[c] = r1;
    x.v.set(c, rv);
}

template<class JV>
__global__ void kLusgsColorReverse
(
    const std::int32_t count,
    const std::int32_t* __restrict__ cells,
    const std::int32_t* __restrict__ off,
    const std::int32_t* __restrict__ faceCode,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ rD,
    const JV J,
    const VecView x
)
{
    const std::int32_t t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= count)
    {
        return;
    }
    const std::int32_t c = cells[t];
    double r0 = x.s0[c], r1 = x.s1[c];
    vec3 rv = x.v[c];
    for (std::int32_t q = off[c]; q < off[c + 1]; ++q)
    {
        const std::int32_t code = faceCode[q];
        const bool ownerSide = code < 0;
        const std::int32_t f = ownerSide ? -(code + 1) : code;
        const std::int32_t j = ownerSide ? neighbour[f] : owner[f];
        colorBlockSub(f, ownerSide, J, x.s0[j], x.s1[j], x.v[j], r0, r1, rv);
    }
    x.s0[c] = rD[c]*r0;
    x.s1[c] = rD[c]*r1;
    x.v.set(c, rD[c]*rv);
}

// 同一次 LU-SGS 的所有層放在一個 cooperative kernel：每層以 grid-stride 處理
// 該層的格，層與層之間 grid.sync()（記憶體可見）。每格的運算與逐層啟動相同，
// 結果逐位元相同，只省掉每層一次的 kernel 啟動
template<class JV>
__global__ void kLusgsForwardCoop
(
    const std::int32_t nLevels,
    const std::int32_t* __restrict__ levelOff,
    const std::int32_t* __restrict__ cells,
    const std::int32_t* __restrict__ off,
    const std::int32_t* __restrict__ faces,
    const std::int32_t* __restrict__ owner,
    const double* __restrict__ rD,
    const JV J,
    const VecView x
)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    const std::int32_t stride = std::int32_t(grid.size());
    for (std::int32_t l = 0; l < nLevels; ++l)
    {
        const std::int32_t e = levelOff[l + 1];
        for (std::int32_t t = levelOff[l] + std::int32_t(grid.thread_rank()); t < e; t += stride)
        {
            lusgsForwardCell(cells[t], off, faces, owner, rD, J, x);
        }
        grid.sync();
    }
}

// 單一 block：每層以 block-stride 處理，層與層之間 __syncthreads()
template<class JV>
__global__ void kLusgsForwardBlock
(
    const std::int32_t nLevels,
    const std::int32_t* __restrict__ levelOff,
    const std::int32_t* __restrict__ cells,
    const std::int32_t* __restrict__ off,
    const std::int32_t* __restrict__ faces,
    const std::int32_t* __restrict__ owner,
    const double* __restrict__ rD,
    const JV J,
    const VecView x
)
{
    for (std::int32_t l = 0; l < nLevels; ++l)
    {
        const std::int32_t e = levelOff[l + 1];
        for (std::int32_t t = levelOff[l] + std::int32_t(threadIdx.x); t < e; t += blockDim.x)
        {
            lusgsForwardCell(cells[t], off, faces, owner, rD, J, x);
        }
        __syncthreads();
    }
}

template<class JV>
__global__ void kLusgsReverseBlock
(
    const std::int32_t nLevels,
    const std::int32_t* __restrict__ levelOff,
    const std::int32_t* __restrict__ cells,
    const std::int32_t* __restrict__ off,
    const std::int32_t* __restrict__ faces,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ rD,
    const JV J,
    const VecView x
)
{
    for (std::int32_t l = 0; l < nLevels; ++l)
    {
        const std::int32_t e = levelOff[l + 1];
        for (std::int32_t t = levelOff[l] + std::int32_t(threadIdx.x); t < e; t += blockDim.x)
        {
            lusgsReverseCell(cells[t], off, faces, neighbour, rD, J, x);
        }
        __syncthreads();
    }
}

template<class JV>
__global__ void kLusgsReverseCoop
(
    const std::int32_t nLevels,
    const std::int32_t* __restrict__ levelOff,
    const std::int32_t* __restrict__ cells,
    const std::int32_t* __restrict__ off,
    const std::int32_t* __restrict__ faces,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ rD,
    const JV J,
    const VecView x
)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    const std::int32_t stride = std::int32_t(grid.size());
    for (std::int32_t l = 0; l < nLevels; ++l)
    {
        const std::int32_t e = levelOff[l + 1];
        for (std::int32_t t = levelOff[l] + std::int32_t(grid.thread_rank()); t < e; t += stride)
        {
            lusgsReverseCell(cells[t], off, faces, neighbour, rD, J, x);
        }
        grid.sync();
    }
}



// ---------------------------------------------------------------------------
// matrixMul：每格依面號收集（Amul 的 Apsi[u] += lower*psi[l]、Apsi[l] += upper*psi[u]）

template<class JV>
__global__ void kMatrixMul
(
    const std::int32_t nC,
    const std::int32_t nIF,
    const std::int32_t* __restrict__ off,
    const std::int32_t* __restrict__ faces,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ V,
    const JV J,
    const CVecView x,
    const VecView out
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC)
    {
        return;
    }
    const double x0 = x.s0[c], x1 = x.s1[c];
    const vec3 xv = x.v[c];

    // 各子區塊的 Amul
    double aSS00 = J.SS00d[c]*x0;
    const double aSS01 = J.SS01d[c]*x1;
    double aSV00 = dot(J.SV00d[c], xv);
    double aSS10 = J.SS10d[c]*x0;
    double aSS11 = J.SS11d[c]*x1;
    double aSV10 = dot(J.SV10d[c], xv);
    vec3 aVS00 = J.VS00d[c]*x0;
    vec3 aVS01 = J.VS01d[c]*x1;
    vec3 aVV00 = dot(J.VV00d[c], xv);
    for (std::int32_t q = off[c]; q < off[c + 1]; ++q)
    {
        const std::int32_t f = faces[q];
        if (f >= nIF)
        {
            continue;
        }
        if (neighbour[f] == c)
        {
            const std::int32_t l = owner[f];
            const double y0 = x.s0[l], y1 = x.s1[l];
            const vec3 yv = x.v[l];
            aSS00 = aSS00 + J.SS00l[f]*y0;
            aSV00 = aSV00 + dot(J.SV00l[f], yv);
            aSS10 = aSS10 + J.SS10l[f]*y0;
            aSS11 = aSS11 + J.SS11l[f]*y1;
            aSV10 = aSV10 + dot(J.SV10l[f], yv);
            aVS00 = aVS00 + J.VS00l[f]*y0;
            aVS01 = aVS01 + J.VS01l[f]*y1;
            aVV00 = aVV00 + dot(J.VV00l[f], yv);
        }
        else
        {
            const std::int32_t u = neighbour[f];
            const double y0 = x.s0[u], y1 = x.s1[u];
            const vec3 yv = x.v[u];
            aSS00 = aSS00 + J.SS00u[f]*y0;
            aSV00 = aSV00 + dot(J.SV00u[f], yv);
            aSS10 = aSS10 + J.SS10u[f]*y0;
            aSS11 = aSS11 + J.SS11u[f]*y1;
            aSV10 = aSV10 + dot(J.SV10u[f], yv);
            aVS00 = aVS00 + J.VS00u[f]*y0;
            aVS01 = aVS01 + J.VS01u[f]*y1;
            aVV00 = aVV00 + dot(J.VV00u[f], yv);
        }
    }
    // sResult = 0；依子區塊次序 +=；最後除以 V
    const double r0 = ((0.0 + aSS00) + aSS01) + aSV00;
    const double r1 = ((0.0 + aSS10) + aSS11) + aSV10;
    const vec3 rv = ((vec3{0.0, 0.0, 0.0} + aVS00) + aVS01) + aVV00;
    out.s0[c] = r0/V[c];
    out.s1[c] = r1/V[c];
    out.v.set(c, rv/V[c]);
}


// ---------------------------------------------------------------------------
// GMRES 的向量運算

__global__ void kCopyVec(const std::int32_t n, const CVecView a, const VecView b)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < n)
    {
        b.s0[c] = a.s0[c];
        b.s1[c] = a.s1[c];
        b.v.set(c, a.v[c]);
    }
}

// x = x/beta
// 係數（beta、H、yh）在裝置記憶體：GMRES 的 Hessenberg 留在裝置上（N7）
__global__ void kDivVec(const std::int32_t n, const double* __restrict__ pBeta, const VecView x)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < n)
    {
        const double beta = *pBeta;
        x.s0[c] = x.s0[c]/beta;
        x.s1[c] = x.s1[c]/beta;
        x.v.set(c, x.v[c]/beta);
    }
}

// x = x - h*v
__global__ void kSubScaled(const std::int32_t n, const double* __restrict__ pH, const CVecView v, const VecView x)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < n)
    {
        const double h = *pH;
        x.s0[c] = x.s0[c] - h*v.s0[c];
        x.s1[c] = x.s1[c] - h*v.s1[c];
        x.v.set(c, x.v[c] - h*v.v[c]);
    }
}

// x = x + y*v
__global__ void kAddScaled(const std::int32_t n, const double* __restrict__ pY, const CVecView v, const VecView x)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < n)
    {
        const double y = *pY;
        x.s0[c] = x.s0[c] + y*v.s0[c];
        x.s1[c] = x.s1[c] + y*v.s1[c];
        x.v.set(c, x.v[c] + y*v.v[c]);
    }
}

// x = W - avg（dsW 由 sW 複製後減去平均）
__global__ void kSubAvg
(
    const std::int32_t n, const CVecView W, const double a0, const double a1, const vec3 av,
    const VecView x
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < n)
    {
        x.s0[c] = W.s0[c] - a0;
        x.s1[c] = W.s1[c] - a1;
        x.v.set(c, W.v[c] - av);
    }
}

__global__ void kZeroVec(const std::int32_t n, const VecView x)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < n)
    {
        x.s0[c] = 0.0;
        x.s1[c] = 0.0;
        x.v.set(c, vec3{0.0, 0.0, 0.0});
    }
}

// sTmp = sR - sTmp
__global__ void kRMinus(const std::int32_t n, const CVecView R, const VecView x)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < n)
    {
        x.s0[c] = R.s0[c] - x.s0[c];
        x.s1[c] = R.s1[c] - x.s1[c];
        x.v.set(c, R.v[c] - x.v[c]);
    }
}

// W += dW（內部值）
__global__ void kAddTo
(
    const std::int32_t n, const CVecView dW, double* __restrict__ rho,
    const DView<vec3> rhoU, double* __restrict__ rhoE
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < n)
    {
        rho[c] = rho[c] + dW.s0[c];
        rhoE[c] = rhoE[c] + dW.s1[c];
        rhoU.set(c, rhoU[c] + dW.v[c]);
    }
}

// 空方向的分量設為 0（GeometricField::replace）
__global__ void kZeroCmpt(const std::int32_t n, double* __restrict__ a)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[i] = 0.0;
    }
}


// ---------------------------------------------------------------------------
// 歸約（OpenFOAM 的 sum：Sum = 0; 依元素次序 Sum += 項）

enum SumOp : int
{
    SUM_PLAIN = 0,      // sum(a)
    SUM_MAG = 1,        // sum(|a|)
    SUM_SQR = 2,        // sum(a*a)
    SUM_MAGPLUS = 3,    // sum(|a| + |b|)
    SUM_PROD = 4        // sum(a*b)
};

// 多個互不相依的歸約的工作描述：kSeqSumBatch（逐位元）或 kTreeSumBatch（效能版）
// 一次啟動、一次下載
struct SumTask
{
    int op = SUM_PLAIN;
    bool vector = false;
    const double* a = nullptr;
    const double* b = nullptr;
    CView<vec3> av{};
    CView<vec3> bv{};
};

constexpr int maxSumTasks = 32;

struct SumTasks
{
    SumTask t[maxSumTasks];
    int n = 0;
};

// 第 i 項（與 OpenFOAM 的 Sum += 項 相同的運算式）；3 個分量的（向量的 plain、
// mag）寫 v[0..2]，其餘寫 v[0]
__device__ __forceinline__ void seqTerm(const SumTask& t, const std::int32_t i, double* v)
{
    if (!t.vector)
    {
        switch (t.op)
        {
            case SUM_PLAIN: v[0] = t.a[i]; break;
            case SUM_MAG: v[0] = ::fabs(t.a[i]); break;
            case SUM_SQR: v[0] = t.a[i]*t.a[i]; break;
            case SUM_MAGPLUS: v[0] = ::fabs(t.a[i]) + ::fabs(t.b[i]); break;
            default: v[0] = t.a[i]*t.b[i]; break;
        }
        return;
    }
    const vec3 x = t.av[i];
    switch (t.op)
    {
        case SUM_PLAIN: v[0] = x.x; v[1] = x.y; v[2] = x.z; break;
        case SUM_MAG: v[0] = ::fabs(x.x); v[1] = ::fabs(x.y); v[2] = ::fabs(x.z); break;
        case SUM_SQR: v[0] = magSqr(x); break;
        case SUM_MAGPLUS: v[0] = mag(x) + mag(t.bv[i]); break;
        default: v[0] = dot(x, t.bv[i]); break;
    }
}

// 一個工作一個 block：thread 32 起把下一段的各項算好放進 shared memory，thread 0
// 依序累加目前這一段（雙緩衝）。累加次序與 OpenFOAM 的 sum 相同（Sum = 0，依元素
// 次序 Sum += 項；向量逐分量），結果逐位元相同
constexpr int SEQ_BS = 256;
constexpr int SEQ_CH = 640;

__global__ void kSeqSumBatch(const std::int32_t n, const SumTasks tasks, double* __restrict__ out)
{
    __shared__ double buf[2][3][SEQ_CH];
    const SumTask& t = tasks.t[blockIdx.x];
    const int nc = (t.vector && (t.op == SUM_PLAIN || t.op == SUM_MAG)) ? 3 : 1;
    const std::int32_t nChunks = (n + SEQ_CH - 1)/SEQ_CH;
    const int tid = int(threadIdx.x);
    const auto fill = [&](const std::int32_t chunk, const int b, const int t0, const int stride)
    {
        const std::int32_t base = chunk*SEQ_CH;
        for (std::int32_t j = t0; j < SEQ_CH && base + j < n; j += stride)
        {
            double v[3];
            seqTerm(t, base + j, v);
            for (int k = 0; k < nc; ++k)
            {
                buf[b][k][j] = v[k];
            }
        }
    };
    if (nChunks > 0)
    {
        fill(0, 0, tid, SEQ_BS);
    }
    __syncthreads();
    double s0 = 0.0, s1 = 0.0, s2 = 0.0;
    for (std::int32_t c = 0; c < nChunks; ++c)
    {
        const int b = int(c & 1);
        if (tid >= 32)
        {
            if (c + 1 < nChunks)
            {
                fill(c + 1, b ^ 1, tid - 32, SEQ_BS - 32);
            }
        }
        else if (tid == 0)
        {
            const std::int32_t m = (n - c*SEQ_CH < SEQ_CH) ? n - c*SEQ_CH : SEQ_CH;
            if (nc == 3)
            {
                // vec3 的 s = s + x：逐分量相加
                for (std::int32_t j = 0; j < m; ++j)
                {
                    s0 = s0 + buf[b][0][j];
                    s1 = s1 + buf[b][1][j];
                    s2 = s2 + buf[b][2][j];
                }
            }
            else
            {
                // 先把 8 個值讀進暫存器再依序相加：載入可以重疊，相加的次序不變
                std::int32_t j = 0;
                for (; j + 8 <= m; j += 8)
                {
                    double v[8];
                    for (int k = 0; k < 8; ++k)
                    {
                        v[k] = buf[b][0][j + k];
                    }
                    for (int k = 0; k < 8; ++k)
                    {
                        s0 += v[k];
                    }
                }
                for (; j < m; ++j)
                {
                    s0 += buf[b][0][j];
                }
            }
        }
        __syncthreads();
    }
    if (tid == 0)
    {
        double* o = out + 3*blockIdx.x;
        o[0] = s0;
        o[1] = s1;
        o[2] = s2;
    }
}

// 樹狀歸約（效能版，GNC_REDUCE=tree；不與 HiSA 逐位元）：每個工作分成固定大小
// （TREE_CH 個元素）的段，一段一個 block；thread k 依序累加段內第 k、k+TREE_BS、…
// 項（每 thread TREE_PT 項），再在 shared memory 以固定的對半樹相加，寫出該段的部分和。
// 最後完成的 block（原子計數）把該工作的部分和依段的次序（thread k 累加第 k、
// k+TREE_BS、… 段）再以對半樹合併。相加的次序只取決於 n，與 block 完成的先後無關，
// 同一個 n 每次結果相同。一次啟動、grid 為（段數, 工作數），GPU 不再只有幾個 block
constexpr int TREE_BS = 256;
constexpr int TREE_PT = 8;
constexpr int TREE_CH = TREE_BS*TREE_PT;

__device__ __forceinline__ void treeBlock(double (&sh)[3][TREE_BS], const int nc, const int tid)
{
    __syncthreads();
    for (int h = TREE_BS/2; h > 0; h >>= 1)
    {
        if (tid < h)
        {
            for (int k = 0; k < nc; ++k)
            {
                sh[k][tid] += sh[k][tid + h];
            }
        }
        __syncthreads();
    }
}

__global__ void kTreeSumBatch
(
    const std::int32_t n,
    const SumTasks tasks,
    double* __restrict__ partial,
    unsigned int* __restrict__ counter,
    double* __restrict__ out
)
{
    __shared__ double sh[3][TREE_BS];
    __shared__ bool isLast;
    const int task = int(blockIdx.y);
    const SumTask& t = tasks.t[task];
    const int nc = (t.vector && (t.op == SUM_PLAIN || t.op == SUM_MAG)) ? 3 : 1;
    const int tid = int(threadIdx.x);
    const int nChunks = int(gridDim.x);
    const std::int32_t base = std::int32_t(blockIdx.x)*TREE_CH;
    double s[3] = {0.0, 0.0, 0.0};
    for (int m = 0; m < TREE_PT; ++m)
    {
        const std::int32_t i = base + m*TREE_BS + tid;
        if (i < n)
        {
            double v[3];
            seqTerm(t, i, v);
            for (int k = 0; k < nc; ++k)
            {
                s[k] += v[k];
            }
        }
    }
    for (int k = 0; k < 3; ++k)
    {
        sh[k][tid] = s[k];
    }
    treeBlock(sh, nc, tid);
    if (tid == 0)
    {
        double* p = partial + 3*(std::size_t(task)*std::size_t(nChunks) + blockIdx.x);
        for (int k = 0; k < 3; ++k)
        {
            p[k] = sh[k][0];
        }
        __threadfence();
        isLast = atomicAdd(&counter[task], 1u) == unsigned(nChunks - 1);
    }
    __syncthreads();
    if (!isLast)
    {
        return;
    }
    __threadfence();
    double q[3] = {0.0, 0.0, 0.0};
    for (int c = tid; c < nChunks; c += TREE_BS)
    {
        const double* p = partial + 3*(std::size_t(task)*std::size_t(nChunks) + std::size_t(c));
        for (int k = 0; k < nc; ++k)
        {
            q[k] += __ldcg(p + k);
        }
    }
    for (int k = 0; k < 3; ++k)
    {
        sh[k][tid] = q[k];
    }
    treeBlock(sh, nc, tid);
    if (tid == 0)
    {
        double* o = out + 3*task;
        o[0] = sh[0][0];
        o[1] = sh[1][0];
        o[2] = sh[2][0];
        counter[task] = 0;
    }
}


struct Reducer
{
    DeviceArray<double> batchBuf;
    DeviceArray<double> partial;            // 樹狀歸約的部分和
    DeviceArray<unsigned int> counter;      // 樹狀歸約完成的段數（每次用完歸零）
    std::int32_t n;

    void launch(const SumTasks& tasks)
    {
        if (batchBuf.size() < std::size_t(3*maxSumTasks))
        {
            batchBuf.allocate(std::size_t(3*maxSumTasks));
        }
        if (useTreeReduce())
        {
            const int nChunks = int((n + TREE_CH - 1)/TREE_CH);
            if (nChunks > 0)
            {
                const std::size_t np = std::size_t(3)*std::size_t(maxSumTasks)*std::size_t(nChunks);
                if (partial.size() < np)
                {
                    partial.allocate(np);
                }
                if (counter.size() < std::size_t(maxSumTasks))
                {
                    counter.allocate(std::size_t(maxSumTasks));
                    counter.zero();
                }
                kTreeSumBatch<<<dim3(unsigned(nChunks), unsigned(tasks.n)), TREE_BS>>>
                (
                    n, tasks, partial.data(), counter.data(), batchBuf.data()
                );
            }
            else
            {
                batchBuf.zero();
            }
        }
        else
        {
            kSeqSumBatch<<<tasks.n, SEQ_BS>>>(n, tasks, batchBuf.data());
        }
        GNC_CUDA_LAUNCH_CHECK();
    }

    // 批次：回傳每個工作的 3 個值（純量只用第 0 個）
    std::vector<double> batch(const SumTasks& tasks)
    {
        if (tasks.n == 0)
        {
            return {};
        }
        ScopedLap lap(gmresProfile().reduce);
        ++gmresProfile().nReduce;
        launch(tasks);
        std::vector<double> h(std::size_t(3*maxSumTasks));
        GNC_CUDA_CHECK
        (
            cudaMemcpy(h.data(), batchBuf.data(), h.size()*sizeof(double), cudaMemcpyDeviceToHost)
        );
        return h;
    }

    // 同上，但結果留在裝置上（batchBuf，下一批會覆寫），不同步
    const double* batchDevice(const SumTasks& tasks)
    {
        ScopedLap lap(gmresProfile().reduce);
        ++gmresProfile().nReduce;
        launch(tasks);
        return batchBuf.data();
    }
};


// GNC_GRAPH=1：GMRES 的每次重啟（預條件、Arnoldi、回代、解的增量與新的殘差）以
// CUDA Graph 重播。重啟之內沒有主機同步，kernel 的序列與參數在同一個問題的每次重啟、
// 每個外迭代都相同（工作區持久、純量放在裝置記憶體），所以捕捉一次後重播；用到的
// 指標（Jacobian、殘差、網格與 LU-SGS 計畫）改變時重新捕捉並更新。kernel 與次序不變，
// 結果與逐一啟動逐位元相同。GMRES 的分段剖析（GNC_PROFILE_GMRES）與 cooperative 的
// 層級模式不用圖。
bool useGraph()
{
    static const bool on = []
    {
        const char* e = std::getenv("GNC_GRAPH");
        return e && std::string(e) == "1";
    }();
    return on && !profileGmres();
}

// GMRES 的持久工作區（指標在呼叫之間不變）
struct GmresWorkspace
{
    std::int32_t n = -1;
    int nDirs = -1;
    HisaVec dW, tmp, R;             // R：右邊的複本（呼叫端的 R 每個外迭代重新配置）
    std::vector<HisaVec> Vk;
    DeviceArray<double> gmem;
    DeviceArray<double> nonDim;     // sNonDim0、sNonDim1、vNonDim
    Reducer red;
    cudaGraphExec_t exec = nullptr;
    std::vector<unsigned char> key;
};

GmresWorkspace& gmresWorkspace(const std::int32_t n, const int nDirs)
{
    static GmresWorkspace w;
    if (w.n != n || w.nDirs != nDirs)
    {
        if (w.exec)
        {
            cudaGraphExecDestroy(w.exec);
            w.exec = nullptr;
        }
        w.key.clear();
        const std::size_t sn = static_cast<std::size_t>(n);
        const std::size_t nd = static_cast<std::size_t>(nDirs);
        w.dW.allocate(sn);
        w.tmp.allocate(sn);
        w.R.allocate(sn);
        w.Vk = std::vector<HisaVec>(nd);
        for (HisaVec& v : w.Vk)
        {
            v.allocate(sn);
        }
        w.gmem.allocate(nd*nd + (nd + 1) + 3*nd + 1);
        w.nonDim.allocate(3);
        w.red = Reducer{};
        w.red.n = n;
        w.n = n;
        w.nDirs = nDirs;
    }
    return w;
}

// 圖的鍵：重啟體內用到的、不屬於工作區的指標與模式
template<class T>
void appendKey(std::vector<unsigned char>& key, const T& v)
{
    const unsigned char* p = reinterpret_cast<const unsigned char*>(&v);
    key.insert(key.end(), p, p + sizeof(T));
}


// gmresTemplates.C 的 givensRotation
__host__ __device__ void givensRotation(const double h, const double beta, double& c, double& s)
{
    if (beta == 0)
    {
        c = 1;
        s = 0;
    }
    else if (::fabs(beta) > ::fabs(h))
    {
        const double tau = -h/beta;
        s = 1.0/::sqrt(1.0 + tau*tau);
        c = s*tau;
    }
    else
    {
        const double tau = -beta/h;
        c = 1.0/::sqrt(1.0 + tau*tau);
        s = c*tau;
    }
}


// GMRES 的 Hessenberg、Givens 與回代（gmresTemplates.C）在裝置上，以單一 thread 的
// kernel 照主機版的運算式執行（IEEE 的 sqrt 與除法、--fmad=false，結果與主機相同）。
// 內積與範數由批次歸約直接寫進裝置記憶體，每個 Krylov 方向不再同步兩次；每次重啟
// 只在最後下載一次殘差。H 為列優先 H[j*nDirs + i]（主機版的 H[j][i]）
struct GmresDevice
{
    double* H;
    double* bh;     // nDirs + 1
    double* cg;
    double* sg;
    double* yh;
    double* beta;   // 目前的 beta（平方和，或開根號之後）
    int nDirs;
};

// beta = r[0]/s0 + r[3]/s1 + r[6]/v（由 0 開始依序累加）；nd = {s0, s1, v}（sNonDim0、
// sNonDim1、vNonDim）放在裝置記憶體，CUDA Graph 重播時不必改 kernel 的參數
__global__ void kGmresBeta
(
    const double* __restrict__ r, const double* __restrict__ nd, double* __restrict__ beta
)
{
    double b = 0.0;
    b += r[0]/nd[0];
    b += r[3]/nd[1];
    b += r[6]/nd[2];
    *beta = b;
}

__global__ void kGmresRestart(const GmresDevice g)
{
    for (int k = 0; k <= g.nDirs; ++k)
    {
        g.bh[k] = 0;
    }
}

// 方向 i 的開頭：beta = sqrt(beta)；i == 0 時 bh[0] = beta，否則對第 i - 1 列做 Givens
// （i == nDirs 為最後的旋轉）
__global__ void kGmresDirStart(const GmresDevice g, const int i)
{
    const double beta = ::sqrt(*g.beta);
    *g.beta = beta;
    if (i == 0)
    {
        g.bh[0] = beta;
        return;
    }
    double& Hii = g.H[(i - 1)*g.nDirs + (i - 1)];
    givensRotation(Hii, beta, g.cg[i - 1], g.sg[i - 1]);
    const double bhi = g.bh[i - 1];
    g.bh[i - 1] = g.cg[i - 1]*bhi - g.sg[i - 1]*g.bh[i];
    g.bh[i] = g.sg[i - 1]*bhi + g.cg[i - 1]*g.bh[i];
    Hii = g.cg[i - 1]*Hii - g.sg[i - 1]*beta;
}

// 古典 Gram-Schmidt 的係數：H[j][i] = r_s0/sNonDim0 + r_s1/sNonDim1 + r_v/vNonDim
__global__ void kGmresH
(
    const GmresDevice g, const int i, const int j0, const int j1, const double* __restrict__ r,
    const double* __restrict__ nd
)
{
    for (int j = j0; j < j1; ++j)
    {
        const int k = 9*(j - j0);
        double h = 0.0;
        h += r[k]/nd[0];
        h += r[k + 3]/nd[1];
        h += r[k + 6]/nd[2];
        g.H[j*g.nDirs + i] = h;
    }
}

// 第 i 行套用之前的 Givens 旋轉
__global__ void kGmresGivensCol(const GmresDevice g, const int i)
{
    for (int j = 0; j < i; ++j)
    {
        const double Hji = g.H[j*g.nDirs + i];
        g.H[j*g.nDirs + i] = g.cg[j]*Hji - g.sg[j]*g.H[(j + 1)*g.nDirs + i];
        g.H[(j + 1)*g.nDirs + i] = g.sg[j]*Hji + g.cg[j]*g.H[(j + 1)*g.nDirs + i];
    }
}

// 回代：yh = H \ bh
__global__ void kGmresBackSub(const GmresDevice g)
{
    for (int i = g.nDirs - 1; i >= 0; --i)
    {
        double sum = g.bh[i];
        for (int j = i + 1; j < g.nDirs; ++j)
        {
            sum -= g.H[i*g.nDirs + j]*g.yh[j];
        }
        g.yh[i] = sum/stabiliseH(g.H[i*g.nDirs + i], VSMALL);
    }
}


std::vector<double> readNumbers(const Dictionary& d, const std::string& key, const std::string& path)
{
    const Dictionary::Entry* e = d.find(key);
    if (!e || e->dict)
    {
        throw std::runtime_error(path + ": " + key + " missing");
    }
    std::vector<double> out;
    for (const Token& t : e->stream)
    {
        if (t.isPunct('(') || t.isPunct(')'))
        {
            continue;
        }
        double v = 0;
        if (!parseScalar(t.text, v))
        {
            throw std::runtime_error(path + ": " + key + ": invalid number " + t.text);
        }
        out.push_back(v);
    }
    return out;
}

} // namespace


LusgsPlan buildLusgsPlan(const PolyMesh& mesh)
{
    const std::int32_t nC = mesh.nCells;
    const std::int32_t nIF = mesh.nInternalFaces;
    LusgsPlan p;

    // 正向：每格的 lower 面（本格為 neighbour），依面號遞增
    std::vector<std::int32_t> lowOff(std::size_t(nC) + 1, 0), upOff(std::size_t(nC) + 1, 0);
    for (std::int32_t f = 0; f < nIF; ++f)
    {
        if (!(mesh.owner[f] < mesh.neighbour[f]))
        {
            throw std::runtime_error("buildLusgsPlan: internal face with owner >= neighbour");
        }
        ++lowOff[std::size_t(mesh.neighbour[f]) + 1];
        ++upOff[std::size_t(mesh.owner[f]) + 1];
    }
    for (std::int32_t c = 0; c < nC; ++c)
    {
        lowOff[c + 1] += lowOff[c];
        upOff[c + 1] += upOff[c];
    }
    std::vector<std::int32_t> lowFaces(static_cast<std::size_t>(nIF)), upFaces(static_cast<std::size_t>(nIF));
    {
        std::vector<std::int32_t> fillL(lowOff.begin(), lowOff.end() - 1);
        std::vector<std::int32_t> fillU(upOff.begin(), upOff.end() - 1);
        for (std::int32_t f = 0; f < nIF; ++f)
        {
            lowFaces[std::size_t(fillL[mesh.neighbour[f]]++)] = f;
            upFaces[std::size_t(fillU[mesh.owner[f]]++)] = f;
        }
    }
    // DILU／DIC 的反向：每格的 upper 面依面號遞減
    std::vector<std::int32_t> upFacesDesc(upFaces);
    for (std::int32_t c = 0; c < nC; ++c)
    {
        std::sort
        (
            upFacesDesc.begin() + upOff[c], upFacesDesc.begin() + upOff[c + 1],
            [](const std::int32_t a, const std::int32_t b) { return a > b; }
        );
    }
    // 正向：依 owner 遞增排列（來源格的次序），同一 owner 內面號遞增；
    // 反向：依 neighbour 遞減排列，同一 neighbour 內面號遞增（穩定排序）
    for (std::int32_t c = 0; c < nC; ++c)
    {
        std::stable_sort
        (
            lowFaces.begin() + lowOff[c], lowFaces.begin() + lowOff[c + 1],
            [&](const std::int32_t a, const std::int32_t b)
            {
                return mesh.owner[a] < mesh.owner[b];
            }
        );
        std::stable_sort
        (
            upFaces.begin() + upOff[c], upFaces.begin() + upOff[c + 1],
            [&](const std::int32_t a, const std::int32_t b)
            {
                return mesh.neighbour[a] > mesh.neighbour[b];
            }
        );
    }

    // 層級
    std::vector<std::int32_t> lev(std::size_t(nC), 0), rlev(std::size_t(nC), 0);
    std::int32_t nLev = 0, nRLev = 0;
    for (std::int32_t c = 0; c < nC; ++c)
    {
        std::int32_t l = 0;
        for (std::int32_t q = lowOff[c]; q < lowOff[c + 1]; ++q)
        {
            l = std::max(l, lev[std::size_t(mesh.owner[lowFaces[q]])] + 1);
        }
        lev[c] = l;
        nLev = std::max(nLev, l + 1);
    }
    for (std::int32_t c = nC - 1; c >= 0; --c)
    {
        std::int32_t l = 0;
        for (std::int32_t q = upOff[c]; q < upOff[c + 1]; ++q)
        {
            l = std::max(l, rlev[std::size_t(mesh.neighbour[upFaces[q]])] + 1);
        }
        rlev[c] = l;
        nRLev = std::max(nRLev, l + 1);
    }
    const auto bucket = [&](const std::vector<std::int32_t>& L, const std::int32_t nl,
                            std::vector<std::int32_t>& off, std::vector<std::int32_t>& cells)
    {
        off.assign(std::size_t(nl) + 1, 0);
        for (std::int32_t c = 0; c < nC; ++c)
        {
            ++off[std::size_t(L[c]) + 1];
        }
        for (std::int32_t l = 0; l < nl; ++l)
        {
            off[l + 1] += off[l];
        }
        cells.resize(std::size_t(nC));
        std::vector<std::int32_t> fill(off.begin(), off.end() - 1);
        for (std::int32_t c = 0; c < nC; ++c)
        {
            cells[std::size_t(fill[L[c]]++)] = c;
        }
    };
    std::vector<std::int32_t> fwdCells, revCells;
    bucket(lev, nC ? nLev : 0, p.fwdLevelOff, fwdCells);
    bucket(rlev, nC ? nRLev : 0, p.revLevelOff, revCells);

    p.fwdCells = DeviceArray<std::int32_t>(fwdCells.size());
    p.fwdCells.upload(fwdCells);
    p.fwdLevelOffD.upload(p.fwdLevelOff);
    p.revLevelOffD.upload(p.revLevelOff);

    // 多色版：貪婪著色（依格編號）
    {
        // 每格的內部面（面號遞增）
        std::vector<std::int32_t> cfOff(std::size_t(nC) + 1, 0);
        for (std::int32_t f = 0; f < nIF; ++f)
        {
            ++cfOff[std::size_t(mesh.owner[f]) + 1];
            ++cfOff[std::size_t(mesh.neighbour[f]) + 1];
        }
        for (std::int32_t c = 0; c < nC; ++c)
        {
            cfOff[std::size_t(c) + 1] += cfOff[std::size_t(c)];
        }
        std::vector<std::int32_t> cf(static_cast<std::size_t>(cfOff[std::size_t(nC)]));
        {
            std::vector<std::int32_t> fill(cfOff.begin(), cfOff.end() - 1);
            for (std::int32_t f = 0; f < nIF; ++f)
            {
                cf[std::size_t(fill[std::size_t(mesh.owner[f])]++)] = f;
                cf[std::size_t(fill[std::size_t(mesh.neighbour[f])]++)] = f;
            }
        }
        std::vector<std::int32_t> color(std::size_t(nC), -1);
        std::int32_t nCol = 0;
        // GNC_COLOR_BIPARTITE=1（gncRhoSimple -fast 設定，N-028）：相鄰格的圖是二分圖時用兩色（由未著色的
        // 格依編號做 BFS），否則照下面的貪婪著色。gncHisa 不設，著色不變
        static const bool tryTwo = []
        {
            const char* e = std::getenv("GNC_COLOR_BIPARTITE");
            return e && std::string(e) == "1";
        }();
        bool twoColors = false;
        if (tryTwo && nC > 0)
        {
            twoColors = true;
            std::vector<std::int32_t> queue;
            queue.reserve(std::size_t(nC));
            for (std::int32_t s = 0; s < nC && twoColors; ++s)
            {
                if (color[std::size_t(s)] >= 0)
                {
                    continue;
                }
                color[std::size_t(s)] = 0;
                queue.assign(1, s);
                for (std::size_t h = 0; h < queue.size() && twoColors; ++h)
                {
                    const std::int32_t c = queue[h];
                    for (std::int32_t q = cfOff[std::size_t(c)]; q < cfOff[std::size_t(c) + 1]; ++q)
                    {
                        const std::int32_t f = cf[std::size_t(q)];
                        const std::int32_t o = mesh.owner[f] == c ? mesh.neighbour[f] : mesh.owner[f];
                        if (color[std::size_t(o)] < 0)
                        {
                            color[std::size_t(o)] = 1 - color[std::size_t(c)];
                            queue.push_back(o);
                        }
                        else if (color[std::size_t(o)] == color[std::size_t(c)])
                        {
                            twoColors = false;
                            break;
                        }
                    }
                }
            }
            if (twoColors)
            {
                nCol = nC > 1 ? 2 : 1;
            }
            else
            {
                std::fill(color.begin(), color.end(), -1);
            }
        }
        std::vector<char> used;
        for (std::int32_t c = 0; c < nC && !twoColors; ++c)
        {
            used.assign(std::size_t(nCol) + 1, 0);
            for (std::int32_t q = cfOff[std::size_t(c)]; q < cfOff[std::size_t(c) + 1]; ++q)
            {
                const std::int32_t f = cf[std::size_t(q)];
                const std::int32_t o = mesh.owner[f] == c ? mesh.neighbour[f] : mesh.owner[f];
                const std::int32_t k = color[std::size_t(o)];
                if (k >= 0)
                {
                    used[std::size_t(k)] = 1;
                }
            }
            std::int32_t k = 0;
            while (used[std::size_t(k)])
            {
                ++k;
            }
            color[std::size_t(c)] = k;
            nCol = std::max(nCol, k + 1);
        }
        if (useColoredLusgs() || useColoredSgs())
        {
            std::printf
            (
                "multicolor ordering (GNC_LUSGS=%s GNC_SGS=%s): %d colors; levels %d/%d\n",
                useColoredLusgs() ? "color" : "-", useColoredSgs() ? "color" : "-", nCol, nLev, nRLev
            );
        }
        std::vector<std::int32_t> colorCells;
        bucket(color, nC ? nCol : 0, p.colorOff, colorCells);
        std::vector<std::int32_t> lo(std::size_t(nC) + 1, 0), up(std::size_t(nC) + 1, 0);
        std::vector<std::int32_t> lf, uf;
        for (std::int32_t c = 0; c < nC; ++c)
        {
            for (std::int32_t q = cfOff[std::size_t(c)]; q < cfOff[std::size_t(c) + 1]; ++q)
            {
                const std::int32_t f = cf[std::size_t(q)];
                const bool ownerSide = mesh.owner[f] == c;
                const std::int32_t o = ownerSide ? mesh.neighbour[f] : mesh.owner[f];
                const std::int32_t code = ownerSide ? -(f + 1) : f;
                if (color[std::size_t(o)] < color[std::size_t(c)])
                {
                    lf.push_back(code);
                }
                else
                {
                    uf.push_back(code);
                }
            }
            lo[std::size_t(c) + 1] = std::int32_t(lf.size());
            up[std::size_t(c) + 1] = std::int32_t(uf.size());
        }
        p.colorCells.upload(colorCells);
        p.cLowOff.upload(lo);
        p.cLowFace.upload(lf);
        p.cUpOff.upload(up);
        p.cUpFace.upload(uf);
    }
    p.revCells = DeviceArray<std::int32_t>(revCells.size());
    p.revCells.upload(revCells);
    p.lowOff = DeviceArray<std::int32_t>(lowOff.size());
    p.lowOff.upload(lowOff);
    p.lowFaces = DeviceArray<std::int32_t>(lowFaces.size());
    p.lowFaces.upload(lowFaces);
    p.upOff = DeviceArray<std::int32_t>(upOff.size());
    p.upOff.upload(upOff);
    p.upFacesDesc = DeviceArray<std::int32_t>(upFacesDesc.size());
    p.upFacesDesc.upload(upFacesDesc);
    p.upFaces = DeviceArray<std::int32_t>(upFaces.size());
    p.upFaces.upload(upFaces);
    return p;
}


template<class JV>
void lusgsPreconditionT
(
    const DeviceMesh& dm,
    const double* rD,
    const JV& jv,
    const LusgsPlan& plan,
    HisaVec& x
)
{
    const std::int32_t nC = dm.nCells;
    if (nC == 0)
    {
        return;
    }
    ScopedLap lap(gmresProfile().precondition);
    ++gmresProfile().nPrecondition;
    const VecView xv = vecView(x);
    kMulV<<<nBlocks(nC), BS>>>(nC, dm.V.data(), xv);
    GNC_CUDA_LAUNCH_CHECK();
    if (useColoredLusgs())
    {
        const std::int32_t nCol = std::int32_t(plan.colorOff.size()) - 1;
        for (std::int32_t k = 0; k < nCol; ++k)
        {
            const std::int32_t s = plan.colorOff[std::size_t(k)];
            const std::int32_t cnt = plan.colorOff[std::size_t(k) + 1] - s;
            kLusgsColorForward<<<nBlocks(cnt), BS>>>
            (
                cnt, plan.colorCells.data() + s, plan.cLowOff.data(), plan.cLowFace.data(),
                dm.owner.data(), dm.neighbour.data(), rD, jv, xv
            );
        }
        for (std::int32_t k = nCol - 1; k >= 0; --k)
        {
            const std::int32_t s = plan.colorOff[std::size_t(k)];
            const std::int32_t cnt = plan.colorOff[std::size_t(k) + 1] - s;
            kLusgsColorReverse<<<nBlocks(cnt), BS>>>
            (
                cnt, plan.colorCells.data() + s, plan.cUpOff.data(), plan.cUpFace.data(),
                dm.owner.data(), dm.neighbour.data(), rD, jv, xv
            );
        }
        GNC_CUDA_LAUNCH_CHECK();
        return;
    }
    const int mode = levelMode(nC, std::int32_t(plan.fwdLevelOff.size()) - 1);
    if (mode == 2)
    {
        kLusgsForwardBlock<<<1, levelBlockSize>>>
        (
            std::int32_t(plan.fwdLevelOff.size()) - 1, plan.fwdLevelOffD.data(),
            plan.fwdCells.data(), plan.lowOff.data(), plan.lowFaces.data(), dm.owner.data(),
            rD, jv, xv
        );
        kLusgsReverseBlock<<<1, levelBlockSize>>>
        (
            std::int32_t(plan.revLevelOff.size()) - 1, plan.revLevelOffD.data(),
            plan.revCells.data(), plan.upOff.data(), plan.upFaces.data(), dm.neighbour.data(),
            rD, jv, xv
        );
        GNC_CUDA_LAUNCH_CHECK();
        return;
    }
    if (mode == 1)
    {
        std::int32_t nl = std::int32_t(plan.fwdLevelOff.size()) - 1;
        const std::int32_t* lo = plan.fwdLevelOffD.data();
        const std::int32_t* cells = plan.fwdCells.data();
        const std::int32_t* off = plan.lowOff.data();
        const std::int32_t* faces = plan.lowFaces.data();
        const std::int32_t* own = dm.owner.data();
        const double* rDc = rD;
        JV jvv = jv;
        VecView xvv = xv;
        void* args[] = {&nl, &lo, &cells, &off, &faces, &own, &rDc, &jvv, &xvv};
        static const int gf = coopGridSize(kLusgsForwardCoop<JV>, BS);
        GNC_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kLusgsForwardCoop<JV>, gf, BS, args, 0, 0));
        std::int32_t nr = std::int32_t(plan.revLevelOff.size()) - 1;
        const std::int32_t* ro = plan.revLevelOffD.data();
        const std::int32_t* rcells = plan.revCells.data();
        const std::int32_t* uoff = plan.upOff.data();
        const std::int32_t* ufaces = plan.upFaces.data();
        const std::int32_t* nei = dm.neighbour.data();
        void* rargs[] = {&nr, &ro, &rcells, &uoff, &ufaces, &nei, &rDc, &jvv, &xvv};
        static const int gr = coopGridSize(kLusgsReverseCoop<JV>, BS);
        GNC_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kLusgsReverseCoop<JV>, gr, BS, rargs, 0, 0));
        return;
    }
    for (std::size_t l = 0; l + 1 < plan.fwdLevelOff.size(); ++l)
    {
        const std::int32_t s = plan.fwdLevelOff[l];
        const std::int32_t cnt = plan.fwdLevelOff[l + 1] - s;
        kLusgsForward<<<nBlocks(cnt), BS>>>
        (
            cnt, plan.fwdCells.data() + s, plan.lowOff.data(), plan.lowFaces.data(),
            dm.owner.data(), rD, jv, xv
        );
    }
    GNC_CUDA_LAUNCH_CHECK();
    for (std::size_t l = 0; l + 1 < plan.revLevelOff.size(); ++l)
    {
        const std::int32_t s = plan.revLevelOff[l];
        const std::int32_t cnt = plan.revLevelOff[l + 1] - s;
        kLusgsReverse<<<nBlocks(cnt), BS>>>
        (
            cnt, plan.revCells.data() + s, plan.upOff.data(), plan.upFaces.data(),
            dm.neighbour.data(), rD, jv, xv
        );
    }
    GNC_CUDA_LAUNCH_CHECK();
}


void lusgsPrecondition
(
    const DeviceMesh& dm,
    const HisaJacobian& J,
    const LusgsPlan& plan,
    HisaVec& x
)
{
    lusgsPreconditionT(dm, J.rDiagCoeff.cview().p[0], jacView(J), plan, x);
}


template<class JV>
void jacobianMatrixMulT(const DeviceMesh& dm, const JV& jv, const HisaVec& x, HisaVec& out)
{
    const std::int32_t nC = dm.nCells;
    if (nC == 0)
    {
        return;
    }
    ScopedLap lap(gmresProfile().matrixMul);
    ++gmresProfile().nMatrixMul;
    kMatrixMul<<<nBlocks(nC), BS>>>
    (
        nC, dm.nInternalFaces, dm.cellFaceOffsets.data(), dm.cellFaces.data(),
        dm.owner.data(), dm.neighbour.data(), dm.V.data(), jv, cvecView(x),
        vecView(out)
    );
    GNC_CUDA_LAUNCH_CHECK();
}


void jacobianMatrixMul
(
    const DeviceMesh& dm,
    const HisaJacobian& J,
    const HisaVec& x,
    HisaVec& out
)
{
    jacobianMatrixMulT(dm, jacView(J), x, out);
}


// 效能版 GNC_JAC=float：Jacobian 的全部區塊轉成 FP32 儲存（混合精度：kernel 讀出後轉回
// FP64 運算；LU-SGS 的對角倒數 rDiagCoeff 與向量仍為 FP64）。大網格的矩陣乘與 LU-SGS
// 掃描是記憶體頻寬受限，讀取量減半（ONERA M6：GMRES 21 → 12 毫秒）。小網格的 kernel
// 受延遲限制，float 轉 double 的指令走 FP64 管線（L40 的 FP64 很少），反而較慢；
// GNC_JAC=auto 只在 FP64 的 Jacobian 大於 L2 快取時使用 FP32
bool useFloatJacobian(const DeviceMesh& dm)
{
    static const int mode = []
    {
        if (envIs("GNC_JAC", "float")) return 1;
        if (envIs("GNC_JAC", "auto")) return 2;
        return 0;
    }();
    if (mode != 2)
    {
        return mode == 1;
    }
    static const long long l2 = []
    {
        int dev = 0;
        GNC_CUDA_CHECK(cudaGetDevice(&dev));
        int v = 0;
        GNC_CUDA_CHECK(cudaDeviceGetAttribute(&v, cudaDevAttrL2CacheSize, dev));
        return (long long)v;
    }();
    const long long bytes = 8LL*(25LL*dm.nCells + 48LL*dm.nInternalFaces);
    return bytes > l2;
}

// 一次啟動轉換全部陣列：第 y 個 block 列處理表中的第 y 個陣列
constexpr int maxConvert = 80;

struct ConvertTable
{
    const double* src[maxConvert];
    float* dst[maxConvert];
    std::int32_t n[maxConvert];
};

__global__ void kToFloat(const ConvertTable t)
{
    const int k = int(blockIdx.y);
    const std::int32_t n = t.n[k];
    for (std::int32_t i = std::int32_t(blockIdx.x*blockDim.x + threadIdx.x); i < n;
         i += std::int32_t(gridDim.x*blockDim.x))
    {
        t.dst[k][i] = __double2float_rn(t.src[k][i]);
    }
}

struct JacobianF32
{
    DeviceArray<float> cellStore;   // 25 個對角分量 × nCells
    DeviceArray<float> faceStore;   // 48 個非對角分量 × nInternalFaces
    JacViewT<float> view{};
    ConvertTable table{};
    int nTable = 0;
    float* nextCell = nullptr;
    float* nextFace = nullptr;
    std::int32_t nC = 0, nF = 0;

    const float* conv(const double* a, const bool cell)
    {
        if (nTable >= maxConvert)
        {
            throw std::runtime_error("JacobianF32: too many arrays");
        }
        float*& next = cell ? nextCell : nextFace;
        const std::int32_t n = cell ? nC : nF;
        table.src[nTable] = a;
        table.dst[nTable] = next;
        table.n[nTable] = n;
        ++nTable;
        float* b = next;
        next += n;
        return b;
    }

    template<class T>
    CView<T, float> conv(const CView<T>& a, const bool cell)
    {
        CView<T, float> b{};
        for (int k = 0; k < NComp<T>::value; ++k)
        {
            b.p[k] = conv(a.p[k], cell);
        }
        return b;
    }

    JacobianF32(const DeviceMesh& dm, const HisaJacobian& J)
    :
        cellStore(std::size_t(25)*std::size_t(dm.nCells)),
        faceStore(std::size_t(48)*std::size_t(dm.nInternalFaces)),
        nC(dm.nCells),
        nF(dm.nInternalFaces)
    {
        nextCell = cellStore.data();
        nextFace = faceStore.data();
        const JacView v = jacView(J);
        const bool c = true, f = false;
        view.SS00d = conv(v.SS00d, c); view.SS01d = conv(v.SS01d, c);
        view.SS10d = conv(v.SS10d, c); view.SS11d = conv(v.SS11d, c);
        view.SV00d = conv(v.SV00d, c); view.SV10d = conv(v.SV10d, c);
        view.VS00d = conv(v.VS00d, c); view.VS01d = conv(v.VS01d, c);
        view.VV00d = conv(v.VV00d, c);
        view.SS00u = conv(v.SS00u, f); view.SS00l = conv(v.SS00l, f);
        view.SS10u = conv(v.SS10u, f); view.SS10l = conv(v.SS10l, f);
        view.SS11u = conv(v.SS11u, f); view.SS11l = conv(v.SS11l, f);
        view.SV00u = conv(v.SV00u, f); view.SV00l = conv(v.SV00l, f);
        view.SV10u = conv(v.SV10u, f); view.SV10l = conv(v.SV10l, f);
        view.VS00u = conv(v.VS00u, f); view.VS00l = conv(v.VS00l, f);
        view.VS01u = conv(v.VS01u, f); view.VS01l = conv(v.VS01l, f);
        view.VV00u = conv(v.VV00u, f); view.VV00l = conv(v.VV00l, f);
        const std::int32_t nMax = std::max(nC, nF);
        if (nMax > 0)
        {
            const int bx = std::min(nBlocks(nMax), 1024);
            kToFloat<<<dim3(unsigned(bx), unsigned(nTable)), BS>>>(table);
            GNC_CUDA_LAUNCH_CHECK();
        }
    }
};


GmresControls readGmresControls(const std::string& path)
{
    FoamTokenizer t(path);
    const Dictionary root = Dictionary::parse(t, path, false);
    const Dictionary& fs = root.subDict("flowSolver");
    const std::string solver = fs.getWord("solver");
    if (solver != "GMRES")
    {
        throw std::runtime_error(path + ": flowSolver solver " + solver + " is not supported (GMRES)");
    }
    // hisaSolver::New 把 subDict(solver) 交給求解器（必須存在）
    const Dictionary& d = fs.subDict(solver);
    const std::string pc = d.getWordOrDefault("preconditioner", "");
    if (pc != "LUSGS" && pc != "gncLUSGSDump")
    {
        throw std::runtime_error(path + ": preconditioner '" + pc + "' is not supported (LUSGS)");
    }
    GmresControls c;
    if (d.found("nKrylov")) c.nKrylov = int(d.getLabel("nKrylov"));
    if (d.found("maxIter")) c.maxIter = int(d.getLabel("maxIter"));
    // residualIO 的讀入次序：rho、rhoU、rhoE（hisaModule 的 residualOrdering）
    const auto readRes = [&](const std::string& key, double* out)
    {
        if (!d.found(key))
        {
            return;
        }
        const std::vector<double> v = readNumbers(d, key, path);
        if (v.size() != 5)
        {
            throw std::runtime_error(path + ": " + key + " needs 5 values");
        }
        for (int i = 0; i < 5; ++i)
        {
            out[i] = v[std::size_t(i)];
        }
    };
    readRes("solverTol", c.tol);
    readRes("solverTolRel", c.tolRel);
    if (c.nKrylov < 1 || c.maxIter < 0)
    {
        throw std::runtime_error(path + ": invalid nKrylov/maxIter");
    }
    return c;
}


GmresResult gmresSolve
(
    const DeviceMesh& dm,
    const HisaJacobian& J,
    const LusgsPlan& plan,
    const GmresControls& ctrl,
    const HisaVec& R,
    VolField<double>& rho,
    VolField<vec3>& rhoU,
    VolField<double>& rhoE
)
{
    const std::int32_t n = dm.nCells;
    const int nDirs = ctrl.nKrylov;
    GmresWorkspace& w = gmresWorkspace(n, nDirs);
    Reducer& red = w.red;

    // 矩陣乘與預條件（GNC_JAC=float 時用 FP32 儲存的 Jacobian）
    std::unique_ptr<JacobianF32> jf;
    if (useFloatJacobian(dm))
    {
        jf.reset(new JacobianF32(dm, J));
    }
    const double* rD = J.rDiagCoeff.cview().p[0];
    const JacView jd = jacView(J);
    const auto matMul = [&](const HisaVec& xin, HisaVec& xout)
    {
        if (jf)
        {
            jacobianMatrixMulT(dm, jf->view, xin, xout);
        }
        else
        {
            jacobianMatrixMulT(dm, jd, xin, xout);
        }
    };
    const auto precondition = [&](HisaVec& xin)
    {
        if (jf)
        {
            lusgsPreconditionT(dm, rD, jf->view, plan, xin);
        }
        else
        {
            lusgsPreconditionT(dm, rD, jd, plan, xin);
        }
    };

    // 空方向的相對容許值設為 GREAT（分量 1..3 為 rhoU）
    double tol[5], tolRel[5];
    for (int k = 0; k < 5; ++k)
    {
        tol[k] = ctrl.tol[k];
        tolRel[k] = ctrl.tolRel[k];
    }
    for (int d = 0; d < 3; ++d)
    {
        if (dm.solutionD[d] == -1)
        {
            tolRel[1 + d] = GREAT;
        }
    }

    HisaVec W;
    W.s0.copyFrom(rho.in);
    W.s1.copyFrom(rhoE.in);
    W.v.copyFrom(rhoU.in);
    HisaVec& dW = w.dW;
    HisaVec& tmp = w.tmp;
    std::vector<HisaVec>& Vk = w.Vk;
    // 右邊複製到工作區（圖重播時指標不變；複製不改值）
    if (n)
    {
        kCopyVec<<<nBlocks(n), BS>>>(n, cvecView(R), vecView(w.R));
        GNC_CUDA_LAUNCH_CHECK();
    }
    const CVecView Rv = cvecView(w.R);

    // 工作：純量（s0、s1）或向量（v）的循序歸約
    const auto sTask = [](const int op, const double* a, const double* b = nullptr)
    {
        SumTask t;
        t.op = op;
        t.a = a;
        t.b = b;
        return t;
    };
    const auto vTask = [](const int op, const CView<vec3> a, const CView<vec3> b = CView<vec3>{})
    {
        SumTask t;
        t.op = op;
        t.vector = true;
        t.av = a;
        t.bv = b;
        return t;
    };

    // normFactor：A(W - gAverage(W))
    double avg0, avg1;
    vec3 avgV;
    {
        SumTasks st;
        st.t[st.n++] = sTask(SUM_PLAIN, W.s0.cview().p[0]);
        st.t[st.n++] = sTask(SUM_PLAIN, W.s1.cview().p[0]);
        st.t[st.n++] = vTask(SUM_PLAIN, W.v.cview());
        const std::vector<double> r = red.batch(st);
        avg0 = r[0]/double(n);
        avg1 = r[3]/double(n);
        avgV = vec3{r[6], r[7], r[8]}/double(n);
    }
    if (n)
    {
        kSubAvg<<<nBlocks(n), BS>>>(n, cvecView(W), avg0, avg1, avgV, vecView(dW));
        GNC_CUDA_LAUNCH_CHECK();
    }
    matMul(dW, tmp);
    if (n)
    {
        kZeroVec<<<nBlocks(n), BS>>>(n, vecView(dW));
        GNC_CUDA_LAUNCH_CHECK();
    }

    double sNorm0, sNorm1, vNorm;
    GmresResult res;
    double sNonDim0, sNonDim1, vNonDim;
    {
        SumTasks st;
        st.t[st.n++] = sTask(SUM_MAGPLUS, tmp.s0.cview().p[0], R.s0.cview().p[0]);
        st.t[st.n++] = sTask(SUM_MAGPLUS, tmp.s1.cview().p[0], R.s1.cview().p[0]);
        st.t[st.n++] = vTask(SUM_MAGPLUS, tmp.v.cview(), R.v.cview());
        st.t[st.n++] = sTask(SUM_MAG, R.s0.cview().p[0]);
        st.t[st.n++] = sTask(SUM_MAG, R.s1.cview().p[0]);
        st.t[st.n++] = vTask(SUM_MAG, R.v.cview());
        st.t[st.n++] = sTask(SUM_SQR, R.s0.cview().p[0]);
        st.t[st.n++] = sTask(SUM_SQR, R.s1.cview().p[0]);
        st.t[st.n++] = vTask(SUM_SQR, R.v.cview());
        const std::vector<double> r = red.batch(st);
        sNorm0 = r[0] + VSMALL;
        sNorm1 = r[3] + VSMALL;
        vNorm = r[6] + VSMALL;
        res.initRes[0] = r[9]/sNorm0;
        res.initRes[4] = r[12]/sNorm1;
        const vec3 cm = vec3{r[15], r[16], r[17]}/vNorm;
        res.initRes[1] = cm.x;
        res.initRes[2] = cm.y;
        res.initRes[3] = cm.z;
        sNonDim0 = stabiliseH(r[18], SMALL);
        sNonDim1 = stabiliseH(r[21], SMALL);
        vNonDim = stabiliseH(r[24], SMALL);
    }
    for (int k = 0; k < 5; ++k)
    {
        res.finalRes[k] = res.initRes[k];
    }


    // Hessenberg 等放在裝置上（GmresDevice）；初值為 0（與主機版的 vector 相同）
    const std::size_t nd = std::size_t(nDirs);
    w.gmem.zero();
    w.nonDim.upload(std::vector<double>{sNonDim0, sNonDim1, vNonDim});
    const double* nonDim = w.nonDim.data();
    GmresDevice g;
    g.H = w.gmem.data();
    g.bh = g.H + nd*nd;
    g.cg = g.bh + nd + 1;
    g.sg = g.cg + nd;
    g.yh = g.sg + nd;
    g.beta = g.yh + nd;
    g.nDirs = nDirs;

    // r_0 ~ R
    if (n)
    {
        kCopyVec<<<nBlocks(n), BS>>>(n, Rv, vecView(tmp));
        GNC_CUDA_LAUNCH_CHECK();
    }

    // beta = 加權的平方和，留在 g.beta
    const auto betaOf = [&](const HisaVec& x)
    {
        SumTasks st;
        st.t[st.n++] = sTask(SUM_SQR, x.s0.cview().p[0]);
        st.t[st.n++] = sTask(SUM_SQR, x.s1.cview().p[0]);
        st.t[st.n++] = vTask(SUM_SQR, x.v.cview());
        const double* r = red.batchDevice(st);
        kGmresBeta<<<1, 1>>>(r, nonDim, g.beta);
        GNC_CUDA_LAUNCH_CHECK();
    };
    // residualIO 的 max：純量（rho、rhoE）再向量分量
    const auto maxRes = [](const double* r)
    {
        double m = -VGREAT;
        m = maxH(m, r[0]);
        m = maxH(m, r[4]);
        const double cmax = maxH(maxH(r[1], r[2]), r[3]);
        m = maxH(m, cmax);
        return m;
    };

    // 一次重啟：預條件、Arnoldi、回代、解的增量與新的殘差（最後的歸約只啟動，
    // 結果在重啟之後下載）
    const auto restartBody = [&]()
    {
        precondition(tmp);
        betaOf(tmp);
        kGmresRestart<<<1, 1>>>(g);
        GNC_CUDA_LAUNCH_CHECK();

        for (int i = 0; i < nDirs; ++i)
        {
            HisaVec& Vi = Vk[std::size_t(i)];
            if (n)
            {
                kCopyVec<<<nBlocks(n), BS>>>(n, cvecView(tmp), vecView(Vi));
                GNC_CUDA_LAUNCH_CHECK();
            }
            matMul(Vi, tmp);
            precondition(tmp);

            // beta = sqrt(beta)，Givens（i > 0）
            kGmresDirStart<<<1, 1>>>(g, i);
            if (n)
            {
                kDivVec<<<nBlocks(n), BS>>>(n, g.beta, vecView(Vi));
                kDivVec<<<nBlocks(n), BS>>>(n, g.beta, vecView(tmp));
            }
            GNC_CUDA_LAUNCH_CHECK();

            {
                // 古典 Gram-Schmidt：所有內積都對同一個 tmp，彼此獨立；每批最多
                // maxSumTasks/3 個方向
                const int perBatch = maxSumTasks/3;
                for (int j0 = 0; j0 <= i; j0 += perBatch)
                {
                    const int j1 = std::min(i + 1, j0 + perBatch);
                    SumTasks st;
                    for (int j = j0; j < j1; ++j)
                    {
                        const HisaVec& Vj = Vk[std::size_t(j)];
                        st.t[st.n++] = sTask(SUM_PROD, tmp.s0.cview().p[0], Vj.s0.cview().p[0]);
                        st.t[st.n++] = sTask(SUM_PROD, tmp.s1.cview().p[0], Vj.s1.cview().p[0]);
                        st.t[st.n++] = vTask(SUM_PROD, tmp.v.cview(), Vj.v.cview());
                    }
                    const double* r = red.batchDevice(st);
                    kGmresH<<<1, 1>>>(g, i, j0, j1, r, nonDim);
                    GNC_CUDA_LAUNCH_CHECK();
                }
            }
            for (int j = 0; j <= i; ++j)
            {
                if (n)
                {
                    kSubScaled<<<nBlocks(n), BS>>>
                    (
                        n, g.H + std::size_t(j)*nd + std::size_t(i), cvecView(Vk[std::size_t(j)]),
                        vecView(tmp)
                    );
                    GNC_CUDA_LAUNCH_CHECK();
                }
            }

            betaOf(tmp);
            kGmresGivensCol<<<1, 1>>>(g, i);
            GNC_CUDA_LAUNCH_CHECK();
        }

        // 最後的旋轉與回代
        kGmresDirStart<<<1, 1>>>(g, nDirs);
        kGmresBackSub<<<1, 1>>>(g);
        GNC_CUDA_LAUNCH_CHECK();

        for (int i = 0; i < nDirs; ++i)
        {
            if (n)
            {
                kAddScaled<<<nBlocks(n), BS>>>(n, g.yh + i, cvecView(Vk[std::size_t(i)]), vecView(dW));
                GNC_CUDA_LAUNCH_CHECK();
            }
        }

        matMul(dW, tmp);
        if (n)
        {
            kRMinus<<<nBlocks(n), BS>>>(n, Rv, vecView(tmp));
            GNC_CUDA_LAUNCH_CHECK();
        }
        SumTasks st;
        st.t[st.n++] = sTask(SUM_MAG, tmp.s0.cview().p[0]);
        st.t[st.n++] = sTask(SUM_MAG, tmp.s1.cview().p[0]);
        st.t[st.n++] = vTask(SUM_MAG, tmp.v.cview());
        ScopedLap lap(gmresProfile().reduce);
        ++gmresProfile().nReduce;
        red.launch(st);
    };

    // CUDA Graph（GNC_GRAPH=1）：鍵為重啟體內不屬於工作區的指標與模式
    const std::int32_t nLevels = std::int32_t(plan.fwdLevelOff.size()) - 1;
    const bool graph = useGraph() && (useColoredLusgs() || levelMode(n, nLevels) != 1);
    bool graphReady = false;
    const auto prepareGraph = [&]()
    {
        std::vector<unsigned char> key;
        appendKey(key, int(jf ? 1 : 0));
        if (jf)
        {
            appendKey(key, jf->view);
        }
        else
        {
            appendKey(key, jd);
        }
        appendKey(key, rD);
        appendKey(key, Rv);
        for
        (
            const void* q :
            {
                (const void*)dm.V.data(), (const void*)dm.owner.data(), (const void*)dm.neighbour.data(),
                (const void*)dm.cellFaceOffsets.data(), (const void*)dm.cellFaces.data(),
                (const void*)plan.colorCells.data(), (const void*)plan.cLowOff.data(),
                (const void*)plan.cLowFace.data(), (const void*)plan.cUpOff.data(),
                (const void*)plan.cUpFace.data(), (const void*)plan.fwdLevelOffD.data(),
                (const void*)plan.fwdCells.data(), (const void*)plan.lowOff.data(),
                (const void*)plan.lowFaces.data(), (const void*)plan.revLevelOffD.data(),
                (const void*)plan.revCells.data(), (const void*)plan.upOff.data(),
                (const void*)plan.upFaces.data()
            }
        )
        {
            appendKey(key, q);
        }
        appendKey(key, int(useColoredLusgs()));
        appendKey(key, levelMode(n, nLevels));
        appendKey(key, int(useTreeReduce()));
        if (w.exec && key == w.key)
        {
            return;
        }
        cudaGraph_t gr;
        GNC_CUDA_CHECK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
        restartBody();
        GNC_CUDA_CHECK(cudaStreamEndCapture(cudaStreamPerThread, &gr));
        if (w.exec)
        {
            cudaGraphExecUpdateResultInfo info;
            if (cudaGraphExecUpdate(w.exec, gr, &info) != cudaSuccess)
            {
                cudaGetLastError();
                cudaGraphExecDestroy(w.exec);
                w.exec = nullptr;
            }
        }
        if (!w.exec)
        {
            GNC_CUDA_CHECK(cudaGraphInstantiate(&w.exec, gr, 0));
        }
        GNC_CUDA_CHECK(cudaGraphDestroy(gr));
        w.key = key;
    };

    int solverIter = 0;
    while (true)
    {
        double q1[5], q2[5];
        for (int k = 0; k < 5; ++k)
        {
            q1[k] = res.finalRes[k]/tol[k];
            q2[k] = res.finalRes[k]/(res.initRes[k] + ROOTVSMALL)/(tolRel[k] + ROOTVSMALL);
        }
        const bool stop =
            (maxRes(q1) < 1.0) || (maxRes(q2) < 1.0) || (solverIter >= ctrl.maxIter);
        if (stop)
        {
            break;
        }

        if (graph)
        {
            if (!graphReady)
            {
                prepareGraph();
                graphReady = true;
            }
            GNC_CUDA_CHECK(cudaGraphLaunch(w.exec, cudaStreamPerThread));
        }
        else
        {
            restartBody();
        }
        {
            std::vector<double> q(std::size_t(3*maxSumTasks));
            GNC_CUDA_CHECK
            (
                cudaMemcpy(q.data(), red.batchBuf.data(), q.size()*sizeof(double), cudaMemcpyDeviceToHost)
            );
            res.finalRes[0] = q[0]/sNorm0;
            res.finalRes[4] = q[3]/sNorm1;
            vec3 r = vec3{q[6], q[7], q[8]}/vNorm;
            if (dm.solutionD[0] == -1) r.x = 0.0;
            if (dm.solutionD[1] == -1) r.y = 0.0;
            if (dm.solutionD[2] == -1) r.z = 0.0;
            res.finalRes[1] = r.x;
            res.finalRes[2] = r.y;
            res.finalRes[3] = r.z;
        }
        ++solverIter;
    }

    // 解的更新
    if (n)
    {
        kAddTo<<<nBlocks(n), BS>>>
        (
            n, cvecView(dW), rho.in.view().p[0], rhoU.in.view(), rhoE.in.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    for (int d = 0; d < 3; ++d)
    {
        if (dm.solutionD[d] == -1)
        {
            if (n)
            {
                kZeroCmpt<<<nBlocks(n), BS>>>(n, rhoU.in.view().p[d]);
            }
            const std::int32_t nB = dm.nFvBoundaryFaces();
            if (nB)
            {
                kZeroCmpt<<<nBlocks(nB), BS>>>(nB, rhoU.bnd.view().p[d]);
            }
            GNC_CUDA_LAUNCH_CHECK();
        }
    }
    // sW[i].correctBoundaryConditions()、vW[i].correctBoundaryConditions()：
    // 這些場由運算式建立，邊界是 calculated，constraint patch（symmetry、
    // symmetryPlane）則是該 patch 的型別，以新的內部值 evaluate
    evaluateBCs(dm, rho, nullptr);
    evaluateBCs(dm, rhoE, nullptr);
    evaluateBCs(dm, rhoU, nullptr);
    GNC_CUDA_CHECK(cudaDeviceSynchronize());
    res.iterations = solverIter;
    return res;
}

} // namespace gnc
