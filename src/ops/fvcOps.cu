// GPU-Native-CFD，GPL-3.0-or-later。見 fvcOps.cuh。
//
// 出處（Atlas upstream/openfoam、upstream/hisa，GPL-3.0-or-later）：
//   linear 內插         surfaceInterpolationScheme.C:216–300（dotInterpolate；
//                       interpolate 即 dotInterpolate(one, ...)）
//   surfaceIntegrate    fvcSurfaceIntegrate.C（surfaceIntegrate、surfaceSum）
//   Gauss 梯度          gaussGrad.C（gradf、calcGrad、correctBoundaryConditions）
//   faceLeastSquares    HiSA faceLeastSquaresVectors.C:63–275、
//                       faceLeastSquaresGrad.C:55–186
//   compact snGrad      HiSA compactSnGrad.H:113–122、compactSnGrad.C:38–72；
//                       snGradScheme.C:92–185
//   Gauss 拉普拉斯      gaussLaplacianSchemes.C:96–115；laplacianScheme.C:99
//   Gauss 散度          gaussDivScheme.C:50–70
// 必須以 --fmad=false 編譯。

#include "ops/fvcOps.cuh"

#include "io/foamTokenizer.hpp"

#include <type_traits>

#include <stdexcept>
#include <string>

namespace gnc
{

namespace
{

constexpr int BS = 256;

int nBlocks(const std::int32_t n)
{
    return int((n + BS - 1)/BS);
}

CView<vec3> meshVec(const DeviceVec3Array& a)
{
    CView<vec3> v;
    v.p[0] = a.x.data();
    v.p[1] = a.y.data();
    v.p[2] = a.z.data();
    return v;
}

// 格面收集表（依面編號次序）
struct Gather
{
    std::int32_t nCells;
    std::int32_t nInternal;
    const std::int32_t* off;
    const std::int32_t* face;
    const std::int8_t* sign;
    const std::int32_t* faceToFvB;
    const std::int32_t* fvbFaces;
};

Gather gatherOf(const DeviceMesh& dm)
{
    return
    {
        dm.nCells, dm.nInternalFaces,
        dm.cellFaceOffsets.data(), dm.cellFaces.data(), dm.cellFaceSigns.data(),
        dm.faceToFvB.data(), dm.fvBoundaryFaces.data()
    };
}


// --------------------------------------------------------------------------
// 面上的運算

template<class T>
__global__ void kCopyPerFace
(
    const std::int32_t n,
    const std::int32_t* __restrict__ idx,   // nullptr：直接索引
    const CView<T> src,
    const DView<T> dst
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        dst.set(i, src[idx ? idx[i] : i]);
    }
}


// sfi[fi] = lambda*(vfi[P] - vfi[N]) + vfi[N]
template<class T>
__global__ void kInterpLinear
(
    const std::int32_t nInternal,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ w,
    const CView<T> vf,
    const DView<T> out
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nInternal)
    {
        return;
    }
    const T vP = vf[owner[f]];
    const T vN = vf[neighbour[f]];
    out.set(f, w[f]*(vP - vN) + vN);
}


template<class T>
__global__ void kDotInterpLinear
(
    const std::int32_t nInternal,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ w,
    const CView<vec3> S,
    const CView<T> vf,
    const DView<typename InnerOf<T>::type> out
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nInternal)
    {
        return;
    }
    const T vP = vf[owner[f]];
    const T vN = vf[neighbour[f]];
    out.set(f, dot(S[f], w[f]*(vP - vN) + vN));
}


template<class T>
__global__ void kDotBoundary
(
    const std::int32_t n,
    const CView<vec3> S,
    const CView<T> vb,
    const DView<typename InnerOf<T>::type> out
)
{
    const std::int32_t k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k < n)
    {
        out.set(k, dot(S[k], vb[k]));
    }
}


template<class T>
__global__ void kMultiply
(
    const std::int32_t n,
    const CView<double> s,
    const CView<T> v,
    const DView<T> out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out.set(i, s[i]*v[i]);
    }
}


// --------------------------------------------------------------------------
// 格上的累加（收集表）

// fvc::surfaceIntegrate：own +=、nei -=，邊界 +=，最後 /= V
template<class T>
__global__ void kSurfaceIntegrate
(
    const Gather g,
    const CView<T> sIn,
    const CView<T> sBnd,
    const double* __restrict__ V,
    const DView<T> out
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= g.nCells)
    {
        return;
    }
    T acc = zeroOf<T>();
    for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
    {
        const std::int32_t f = g.face[k];
        if (f < g.nInternal)
        {
            if (g.sign[k] > 0)
            {
                acc += sIn[f];
            }
            else
            {
                acc -= sIn[f];
            }
        }
        else
        {
            const std::int32_t b = g.faceToFvB[f - g.nInternal];
            if (b >= 0)
            {
                acc += sBnd[b];
            }
        }
    }
    out.set(c, acc/V[c]);
}


// fvc::surfaceSum：own 與 nei 都 +=
__global__ void kSurfaceSum
(
    const Gather g,
    const CView<double> sIn,
    const CView<double> sBnd,
    double* __restrict__ out
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= g.nCells)
    {
        return;
    }
    double acc = 0.0;
    for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
    {
        const std::int32_t f = g.face[k];
        if (f < g.nInternal)
        {
            acc += sIn[f];
        }
        else
        {
            const std::int32_t b = g.faceToFvB[f - g.nInternal];
            if (b >= 0)
            {
                acc += sBnd[b];
            }
        }
    }
    out[c] = acc;
}


// gaussGrad::gradf：Sfssf = Sf*ssf，own +=、nei -=，邊界 += pSf*pssf，/= V
template<class T>
__global__ void kGaussGradCells
(
    const Gather g,
    const CView<vec3> Sf,           // 全域面編號
    const CView<T> sIn,
    const CView<T> sBnd,
    const double* __restrict__ V,
    const DView<typename GradOf<T>::type> out
)
{
    using G = typename GradOf<T>::type;
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= g.nCells)
    {
        return;
    }
    G acc = zeroOf<G>();
    for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
    {
        const std::int32_t f = g.face[k];
        if (f < g.nInternal)
        {
            const G Sfssf = outer(Sf[f], sIn[f]);
            if (g.sign[k] > 0)
            {
                acc += Sfssf;
            }
            else
            {
                acc -= Sfssf;
            }
        }
        else
        {
            const std::int32_t b = g.faceToFvB[f - g.nInternal];
            if (b >= 0)
            {
                acc += outer(Sf[f], sBnd[b]);
            }
        }
    }
    out.set(c, acc/V[c]);
}


// gaussGrad::correctBoundaryConditions（非耦合 patch）：
// gGrad_b += n*(snGrad - (n & gGrad_b))，n = Sf/magSf
template<class T>
__global__ void kGaussGradCorrectBoundary
(
    const std::int32_t nFvB,
    const std::int32_t* __restrict__ fvbFaces,
    const CView<vec3> Sf,
    const double* __restrict__ magSf,
    const CView<T> snGrad,
    const DView<typename GradOf<T>::type> gBnd
)
{
    const std::int32_t k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nFvB)
    {
        return;
    }
    const std::int32_t f = fvbFaces[k];
    const vec3 n = Sf[f]/magSf[f];
    const auto gb = gBnd[k];
    gBnd.set(k, gb + outer(n, snGrad[k] - dot(n, gb)));
}


// cellLimitedGrad<scalar, minmod>::calcGrad 的限制（cellLimitedGrad.C、
// cellLimitedGrad.H 的 limitFaceCmpt、minmodGradientLimiter.H）：每格依面編號
//   maxVsf、minVsf：本格值與相鄰格值（內部面）或 patch 值（非耦合邊界面）的
//                   max／min（MAXMIN：max = (b < a) ? a : b、min = (a < b) ? a : b）
//   減去本格值；k < 1 時再各加減 (1/k - 1)*(maxVsf - minVsf)
//   limiter 從 1 開始，每面 extrapolate = (Cf - C) & g：> SMALL 時
//   r = maxVsf/extrapolate，< -SMALL 時 r = minVsf/extrapolate，否則略過；
//   limiter = min(limiter, min(r, 1))
//   g *= limiter
__global__ void kCellLimitScalar
(
    const Gather g,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const CView<vec3> C,
    const CView<vec3> Cf,
    const double* __restrict__ vf,
    const double* __restrict__ vfB,
    const double k,
    const double kFactor,           // 1.0/k - 1.0
    const DView<vec3> grad
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= g.nCells)
    {
        return;
    }
    const double v = vf[c];
    double mx = v;
    double mn = v;
    for (std::int32_t q = g.off[c]; q < g.off[c + 1]; ++q)
    {
        const std::int32_t f = g.face[q];
        double nb;
        if (f < g.nInternal)
        {
            nb = (g.sign[q] > 0) ? vf[neighbour[f]] : vf[owner[f]];
        }
        else
        {
            const std::int32_t b = g.faceToFvB[f - g.nInternal];
            if (b < 0)
            {
                continue;
            }
            nb = vfB[b];
        }
        mx = foamMax(mx, nb);
        mn = foamMin(mn, nb);
    }
    mx = mx - v;
    mn = mn - v;
    if (k < 1.0)
    {
        const double mm = kFactor*(mx - mn);
        mx = mx + mm;
        mn = mn - mm;
    }

    const vec3 gc = grad[c];
    const vec3 cc = C[c];
    double limiter = 1.0;
    for (std::int32_t q = g.off[c]; q < g.off[c + 1]; ++q)
    {
        const std::int32_t f = g.face[q];
        if (f >= g.nInternal && g.faceToFvB[f - g.nInternal] < 0)
        {
            continue;
        }
        const double e = dot(Cf[f] - cc, gc);
        double r;
        if (e > SMALL)
        {
            r = mx/e;
        }
        else if (e < -SMALL)
        {
            r = mn/e;
        }
        else
        {
            continue;
        }
        limiter = foamMin(limiter, foamMin(r, 1.0));
    }
    grad.set(c, gc*limiter);
}


// cellLimitedGrad<vector, minmod>：同上，每個分量各自限制（limitFace 逐分量呼叫
// limitFaceCmpt），maxVsf、minVsf 為逐分量的 max／min，extrapolate =
// (Cf - C) & g（vector & tensor）；limitGradient 把張量的每一列與 limiter 逐分量相乘：
// g = (cmptMultiply(l, g.x()), cmptMultiply(l, g.y()), cmptMultiply(l, g.z()))
__global__ void kCellLimitVector
(
    const Gather g,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const CView<vec3> C,
    const CView<vec3> Cf,
    const CView<vec3> vf,
    const CView<vec3> vfB,
    const double k,
    const double kFactor,           // 1.0/k - 1.0
    const DView<tens> grad
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= g.nCells)
    {
        return;
    }
    const vec3 v = vf[c];
    vec3 mx = v;
    vec3 mn = v;
    for (std::int32_t q = g.off[c]; q < g.off[c + 1]; ++q)
    {
        const std::int32_t f = g.face[q];
        vec3 nb;
        if (f < g.nInternal)
        {
            nb = (g.sign[q] > 0) ? vf[neighbour[f]] : vf[owner[f]];
        }
        else
        {
            const std::int32_t b = g.faceToFvB[f - g.nInternal];
            if (b < 0)
            {
                continue;
            }
            nb = vfB[b];
        }
        mx = vec3{foamMax(mx.x, nb.x), foamMax(mx.y, nb.y), foamMax(mx.z, nb.z)};
        mn = vec3{foamMin(mn.x, nb.x), foamMin(mn.y, nb.y), foamMin(mn.z, nb.z)};
    }
    mx = mx - v;
    mn = mn - v;
    if (k < 1.0)
    {
        const vec3 mm = kFactor*(mx - mn);
        mx = mx + mm;
        mn = mn - mm;
    }

    const tens gc = grad[c];
    const vec3 cc = C[c];
    double lim[3] = {1.0, 1.0, 1.0};
    const double mxc[3] = {mx.x, mx.y, mx.z};
    const double mnc[3] = {mn.x, mn.y, mn.z};
    for (std::int32_t q = g.off[c]; q < g.off[c + 1]; ++q)
    {
        const std::int32_t f = g.face[q];
        if (f >= g.nInternal && g.faceToFvB[f - g.nInternal] < 0)
        {
            continue;
        }
        const vec3 e = dot(Cf[f] - cc, gc);
        const double ec[3] = {e.x, e.y, e.z};
        for (int d = 0; d < 3; ++d)
        {
            double r;
            if (ec[d] > SMALL)
            {
                r = mxc[d]/ec[d];
            }
            else if (ec[d] < -SMALL)
            {
                r = mnc[d]/ec[d];
            }
            else
            {
                continue;
            }
            lim[d] = foamMin(lim[d], foamMin(r, 1.0));
        }
    }
    grad.set
    (
        c,
        tens
        {
            lim[0]*gc.xx, lim[1]*gc.xy, lim[2]*gc.xz,
            lim[0]*gc.yx, lim[1]*gc.yy, lim[2]*gc.yz,
            lim[0]*gc.zx, lim[1]*gc.zy, lim[2]*gc.zz
        }
    );
}


// HiSA faceLeastSquaresVectors：Ce（內部面）= w*C[own] + (1 - w)*C[nei]
__global__ void kLsCeInternal
(
    const std::int32_t nInternal,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ w,
    const CView<vec3> C,
    const DView<vec3> Ce
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f < nInternal)
    {
        Ce.set(f, w[f]*C[owner[f]] + (1 - w[f])*C[neighbour[f]]);
    }
}


// Ce（非耦合邊界面）= C[fc] + Sf/(magSf*nonOrthDeltaCoeffs)
__global__ void kLsCeBoundary
(
    const std::int32_t nFvB,
    const std::int32_t* __restrict__ fvbFaces,
    const std::int32_t* __restrict__ owner,
    const CView<vec3> Sf,
    const double* __restrict__ magSf,
    const double* __restrict__ nodc,
    const CView<vec3> C,
    const DView<vec3> Ce
)
{
    const std::int32_t k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k < nFvB)
    {
        const std::int32_t f = fvbFaces[k];
        Ce.set(k, C[owner[f]] + Sf[f]/(magSf[f]*nodc[f]));
    }
}


// weight*magSqr(mesh.delta())
__global__ void kLsNormFaces
(
    const std::int32_t nInternal,
    const std::int32_t nFvB,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const std::int32_t* __restrict__ fvbFaces,
    const double* __restrict__ magSf,
    const CView<vec3> C,
    const CView<vec3> fvbDelta,
    double* __restrict__ outIn,
    double* __restrict__ outBnd
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < nInternal)
    {
        outIn[i] = magSf[i]*magSqr(C[neighbour[i]] - C[owner[i]]);
    }
    else if (i < nInternal + nFvB)
    {
        const std::int32_t k = i - nInternal;
        outBnd[k] = magSf[fvbFaces[k]]*magSqr(fvbDelta[k]);
    }
}


// meanPoints：+= weight*Ce（own 與 nei），邊界 += pWeight*pCe，/= weightSum
__global__ void kLsMeanPoints
(
    const Gather g,
    const double* __restrict__ magSf,
    const CView<vec3> CeIn,
    const CView<vec3> CeBnd,
    const double* __restrict__ weightSum,
    const DView<vec3> mean
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= g.nCells)
    {
        return;
    }
    vec3 acc{0.0, 0.0, 0.0};
    for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
    {
        const std::int32_t f = g.face[k];
        if (f < g.nInternal)
        {
            acc += magSf[f]*CeIn[f];
        }
        else
        {
            const std::int32_t b = g.faceToFvB[f - g.nInternal];
            if (b >= 0)
            {
                acc += magSf[f]*CeBnd[b];
            }
        }
    }
    mean.set(c, acc/weightSum[c]);
}


// dd += weight/norm*Ce*(Ce - meanPoints)；2D 時 dd += cmptMultiply(e*e, I)；
// 再以 safeInv 求反（tensorField 的 inv）
__global__ void kLsInvDd
(
    const Gather g,
    const double* __restrict__ magSf,
    const CView<vec3> CeIn,
    const CView<vec3> CeBnd,
    const CView<vec3> mean,
    const double* __restrict__ norm,
    const bool stabilise,
    const tens stab,
    const DView<tens> invDd
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= g.nCells)
    {
        return;
    }
    const vec3 m = mean[c];
    const double nc = norm[c];
    tens dd = zeroOf<tens>();
    for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
    {
        const std::int32_t f = g.face[k];
        if (f < g.nInternal)
        {
            const vec3 ce = CeIn[f];
            dd += outer(magSf[f]/nc*ce, ce - m);
        }
        else
        {
            const std::int32_t b = g.faceToFvB[f - g.nInternal];
            if (b >= 0)
            {
                const vec3 ce = CeBnd[b];
                dd += outer(magSf[f]/nc*ce, ce - m);
            }
        }
    }
    if (stabilise)
    {
        dd += stab;
    }
    invDd.set(c, safeInv(dd));
}


// lsP = weight/norm[own]*(invDd[own] & Ce)，lsN 用 nei
__global__ void kLsVectorsInternal
(
    const std::int32_t nInternal,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ magSf,
    const double* __restrict__ norm,
    const CView<tens> invDd,
    const CView<vec3> Ce,
    const DView<vec3> lsP,
    const DView<vec3> lsN
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nInternal)
    {
        return;
    }
    const std::int32_t o = owner[f];
    const std::int32_t n = neighbour[f];
    const vec3 ce = Ce[f];
    lsP.set(f, magSf[f]/norm[o]*dot(invDd[o], ce));
    lsN.set(f, magSf[f]/norm[n]*dot(invDd[n], ce));
}


__global__ void kLsVectorsBoundary
(
    const std::int32_t nFvB,
    const std::int32_t* __restrict__ fvbFaces,
    const std::int32_t* __restrict__ owner,
    const double* __restrict__ magSf,
    const double* __restrict__ norm,
    const CView<tens> invDd,
    const CView<vec3> CeBnd,
    const DView<vec3> lsP
)
{
    const std::int32_t k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nFvB)
    {
        return;
    }
    const std::int32_t f = fvbFaces[k];
    const std::int32_t c = owner[f];
    lsP.set(k, magSf[f]/norm[c]*dot(invDd[c], CeBnd[k]));
}


// faceLeastSquaresGrad 的 mean：+= weight*ssf（own 與 nei），邊界同，
// /= surfaceSum(weight)
template<class T>
__global__ void kLsMean
(
    const Gather g,
    const double* __restrict__ magSf,
    const CView<T> sIn,
    const CView<T> sBnd,
    const double* __restrict__ weightSum,
    const DView<T> mean
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= g.nCells)
    {
        return;
    }
    T acc = zeroOf<T>();
    for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
    {
        const std::int32_t f = g.face[k];
        if (f < g.nInternal)
        {
            acc += magSf[f]*sIn[f];
        }
        else
        {
            const std::int32_t b = g.faceToFvB[f - g.nInternal];
            if (b >= 0)
            {
                acc += magSf[f]*sBnd[b];
            }
        }
    }
    mean.set(c, acc/weightSum[c]);
}


// faceLeastSquaresGrad.C：fixedGradient 族 patch 的面值
// pvsf.patchInternalField() + gradient()/nonOrthDeltaCoeffs_b
template<class T>
__global__ void kLsFixedGradientFace
(
    const std::int32_t start,
    const std::int32_t n,
    const std::int32_t* __restrict__ fvbFaces,
    const std::int32_t* __restrict__ owner,
    const double* __restrict__ nodc,
    const CView<T> vf,
    const CView<T> gradient,
    const DView<T> ssfBnd
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const std::int32_t k = start + i;
    const std::int32_t f = fvbFaces[k];
    ssfBnd.set(k, vf[owner[f]] + gradient[i]/nodc[f]);
}


// lsGrad[own] += ownLs*(ssf - mean[own])，lsGrad[nei] += neiLs*(ssf - mean[nei])，
// 邊界 += patchOwnLs*(pssf - mean)
template<class T>
__global__ void kLsGradCells
(
    const Gather g,
    const CView<vec3> lsPIn,
    const CView<vec3> lsN,
    const CView<vec3> lsPBnd,
    const CView<T> sIn,
    const CView<T> sBnd,
    const CView<T> mean,
    const DView<typename GradOf<T>::type> out
)
{
    using G = typename GradOf<T>::type;
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= g.nCells)
    {
        return;
    }
    const T m = mean[c];
    G acc = zeroOf<G>();
    for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
    {
        const std::int32_t f = g.face[k];
        if (f < g.nInternal)
        {
            const vec3 ls = g.sign[k] > 0 ? lsPIn[f] : lsN[f];
            acc += outer(ls, sIn[f] - m);
        }
        else
        {
            const std::int32_t b = g.faceToFvB[f - g.nInternal];
            if (b >= 0)
            {
                acc += outer(lsPBnd[b], sBnd[b] - m);
            }
        }
    }
    out.set(c, acc);
}


// compact snGrad（內部面）：
//   dcc  = sqr(dc)*(delta & Sf)/magSf
//   unc  = dcc*(vf[N] - vf[P])
//   dN   = dc*delta；nocv = (Sf - dN*(dN & Sf))/magSf
//   corr = nocv & (w*(g[P] - g[N]) + g[N])
template<class T>
__global__ void kSnGradCompactInternal
(
    const std::int32_t nInternal,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ w,
    const double* __restrict__ dc,
    const double* __restrict__ magSf,
    const CView<vec3> Sf,
    const CView<vec3> C,
    const CView<T> vf,
    const CView<typename GradOf<T>::type> grad,
    const DView<T> out
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nInternal)
    {
        return;
    }
    const std::int32_t P = owner[f];
    const std::int32_t N = neighbour[f];
    const vec3 delta = C[N] - C[P];
    const vec3 sf = Sf[f];

    const double dcc = (dc[f]*dc[f])*dot(delta, sf)/magSf[f];
    const T unc = dcc*(vf[N] - vf[P]);

    const vec3 dN = dc[f]*delta;
    const vec3 nocv = (sf - dN*dot(dN, sf))/magSf[f];
    const auto gP = grad[P];
    const auto gN = grad[N];
    const T corr = dot(nocv, w[f]*(gP - gN) + gN);

    out.set(f, unc + corr);
}


// compact snGrad（非耦合邊界面）：邊界條件的 snGrad + nocv_b & grad_b
template<class T>
__global__ void kSnGradCompactBoundary
(
    const std::int32_t nFvB,
    const std::int32_t* __restrict__ fvbFaces,
    const double* __restrict__ dc,
    const double* __restrict__ magSf,
    const CView<vec3> Sf,
    const CView<vec3> fvbDelta,
    const CView<T> snBC,
    const CView<typename GradOf<T>::type> gradBnd,
    const DView<T> out
)
{
    const std::int32_t k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nFvB)
    {
        return;
    }
    const std::int32_t f = fvbFaces[k];
    const vec3 sf = Sf[f];
    const vec3 dN = dc[f]*fvbDelta[k];
    const vec3 nocv = (sf - dN*dot(dN, sf))/magSf[f];
    out.set(k, snBC[k] + dot(nocv, gradBnd[k]));
}


// gamma*snGrad*magSf（內部面與邊界面各一段）
template<class T>
__global__ void kLaplacianFlux
(
    const std::int32_t n,
    const std::int32_t* __restrict__ faceIdx,   // nullptr：內部面直接索引
    const double* __restrict__ magSf,
    const CView<double> gammaf,
    const CView<T> sn,
    const DView<T> out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const std::int32_t f = faceIdx ? faceIdx[i] : i;
    out.set(i, gammaf[i]*sn[i]*magSf[f]);
}

} // namespace


SurfField<vec3> meshSurfVector(const DeviceMesh& dm, const DeviceVec3Array& perFace)
{
    SurfField<vec3> s;
    s.in.allocate(std::size_t(dm.nInternalFaces));
    s.bnd.allocate(std::size_t(dm.nFvBoundaryFaces()));
    if (dm.nInternalFaces)
    {
        kCopyPerFace<vec3><<<nBlocks(dm.nInternalFaces), BS>>>
        (
            dm.nInternalFaces, nullptr, meshVec(perFace), s.in.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (dm.nFvBoundaryFaces())
    {
        kCopyPerFace<vec3><<<nBlocks(dm.nFvBoundaryFaces()), BS>>>
        (
            dm.nFvBoundaryFaces(), dm.fvBoundaryFaces.data(), meshVec(perFace),
            s.bnd.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    return s;
}


SurfField<double> meshSurfScalar(const DeviceMesh& dm, const DeviceArray<double>& perFace)
{
    SurfField<double> s;
    s.in.allocate(std::size_t(dm.nInternalFaces));
    s.bnd.allocate(std::size_t(dm.nFvBoundaryFaces()));
    CView<double> src;
    src.p[0] = perFace.data();
    if (dm.nInternalFaces)
    {
        kCopyPerFace<double><<<nBlocks(dm.nInternalFaces), BS>>>
        (
            dm.nInternalFaces, nullptr, src, s.in.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (dm.nFvBoundaryFaces())
    {
        kCopyPerFace<double><<<nBlocks(dm.nFvBoundaryFaces()), BS>>>
        (
            dm.nFvBoundaryFaces(), dm.fvBoundaryFaces.data(), src, s.bnd.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    return s;
}


template<class T>
void interpolateLinear(const DeviceMesh& dm, const VolField<T>& vf, SurfField<T>& out)
{
    out.in.allocate(std::size_t(dm.nInternalFaces));
    if (dm.nInternalFaces)
    {
        kInterpLinear<T><<<nBlocks(dm.nInternalFaces), BS>>>
        (
            dm.nInternalFaces, dm.owner.data(), dm.neighbour.data(),
            dm.weights.data(), vf.in.cview(), out.in.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    out.bnd.copyFrom(vf.bnd);
}


template<class T>
void dotInterpolateLinear
(
    const DeviceMesh& dm,
    const SurfField<vec3>& S,
    const VolField<T>& vf,
    SurfField<typename InnerOf<T>::type>& out
)
{
    out.in.allocate(std::size_t(dm.nInternalFaces));
    out.bnd.allocate(std::size_t(dm.nFvBoundaryFaces()));
    if (dm.nInternalFaces)
    {
        kDotInterpLinear<T><<<nBlocks(dm.nInternalFaces), BS>>>
        (
            dm.nInternalFaces, dm.owner.data(), dm.neighbour.data(),
            dm.weights.data(), S.in.cview(), vf.in.cview(), out.in.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (dm.nFvBoundaryFaces())
    {
        kDotBoundary<T><<<nBlocks(dm.nFvBoundaryFaces()), BS>>>
        (
            dm.nFvBoundaryFaces(), S.bnd.cview(), vf.bnd.cview(), out.bnd.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
}


template<class T>
void surfaceIntegrate(const DeviceMesh& dm, const SurfField<T>& ssf, DField<T>& out)
{
    out.allocate(std::size_t(dm.nCells));
    if (dm.nCells)
    {
        kSurfaceIntegrate<T><<<nBlocks(dm.nCells), BS>>>
        (
            gatherOf(dm), ssf.in.cview(), ssf.bnd.cview(), dm.V.data(), out.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
}


void surfaceSum(const DeviceMesh& dm, const SurfField<double>& ssf, DeviceArray<double>& out)
{
    out.allocate(std::size_t(dm.nCells));
    if (dm.nCells)
    {
        kSurfaceSum<<<nBlocks(dm.nCells), BS>>>
        (
            gatherOf(dm), ssf.in.cview(), ssf.bnd.cview(), out.data()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
}


template<class T>
void multiplySurf(const SurfField<double>& s, const SurfField<T>& v, SurfField<T>& out)
{
    out.in.allocate(v.in.size());
    out.bnd.allocate(v.bnd.size());
    if (v.in.size())
    {
        kMultiply<T><<<nBlocks(std::int32_t(v.in.size())), BS>>>
        (
            std::int32_t(v.in.size()), s.in.cview(), v.in.cview(), out.in.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (v.bnd.size())
    {
        kMultiply<T><<<nBlocks(std::int32_t(v.bnd.size())), BS>>>
        (
            std::int32_t(v.bnd.size()), s.bnd.cview(), v.bnd.cview(), out.bnd.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
}


template<class T>
void gaussGrad
(
    const DeviceMesh& dm,
    const VolField<T>& vf,
    VolField<typename GradOf<T>::type>& out
)
{
    using G = typename GradOf<T>::type;

    SurfField<T> ssf;
    interpolateLinear(dm, vf, ssf);

    out = makeVolField<G>(dm, "grad(" + vf.name + ")", Bc::extrapolatedCalculated);
    if (dm.nCells)
    {
        kGaussGradCells<T><<<nBlocks(dm.nCells), BS>>>
        (
            gatherOf(dm), meshVec(dm.Sf), ssf.in.cview(), ssf.bnd.cview(),
            dm.V.data(), out.in.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    evaluateBCs(dm, out, nullptr);

    DField<T> sn;
    snGradBCs(dm, vf, sn);
    if (dm.nFvBoundaryFaces())
    {
        kGaussGradCorrectBoundary<T><<<nBlocks(dm.nFvBoundaryFaces()), BS>>>
        (
            dm.nFvBoundaryFaces(), dm.fvBoundaryFaces.data(), meshVec(dm.Sf),
            dm.magSf.data(), sn.cview(), out.bnd.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
}


template<class T>
void cellLimitedGaussGrad
(
    const DeviceMesh& dm,
    const VolField<T>& vf,
    const double k,
    VolField<typename GradOf<T>::type>& out
)
{
    if constexpr (!std::is_same<T, double>::value && !std::is_same<T, vec3>::value)
    {
        throw std::runtime_error("cellLimited gradient is implemented for scalars and vectors only");
    }
    else
    {
        // basicGradScheme_().calcGrad：Gauss linear（含邊界修正）
        gaussGrad(dm, vf, out);
        if (k < SMALL)
        {
            return;
        }
        if (dm.nCells)
        {
            if constexpr (std::is_same<T, double>::value)
            {
                kCellLimitScalar<<<nBlocks(dm.nCells), BS>>>
                (
                    gatherOf(dm), dm.owner.data(), dm.neighbour.data(), meshVec(dm.C),
                    meshVec(dm.Cf), vf.in.cview().p[0], vf.bnd.cview().p[0], k, 1.0/k - 1.0,
                    out.in.view()
                );
            }
            else
            {
                kCellLimitVector<<<nBlocks(dm.nCells), BS>>>
                (
                    gatherOf(dm), dm.owner.data(), dm.neighbour.data(), meshVec(dm.C),
                    meshVec(dm.Cf), vf.in.cview(), vf.bnd.cview(), k, 1.0/k - 1.0,
                    out.in.view()
                );
            }
            GNC_CUDA_LAUNCH_CHECK();
        }
        // g.correctBoundaryConditions()、gaussGrad::correctBoundaryConditions(vsf, g)
        evaluateBCs(dm, out, nullptr);
        DField<T> sn;
        snGradBCs(dm, vf, sn);
        if (dm.nFvBoundaryFaces())
        {
            kGaussGradCorrectBoundary<T><<<nBlocks(dm.nFvBoundaryFaces()), BS>>>
            (
                dm.nFvBoundaryFaces(), dm.fvBoundaryFaces.data(), meshVec(dm.Sf),
                dm.magSf.data(), sn.cview(), out.bnd.view()
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
    }
}


LSVectors faceLeastSquaresVectors(const DeviceMesh& dm)
{
    const std::int32_t nI = dm.nInternalFaces;
    const std::int32_t nB = dm.nFvBoundaryFaces();
    const std::int32_t nC = dm.nCells;

    LSVectors ls;
    const SurfField<double> weight = meshSurfScalar(dm, dm.magSf);
    surfaceSum(dm, weight, ls.weightSum);

    SurfField<vec3> Ce;
    Ce.in.allocate(std::size_t(nI));
    Ce.bnd.allocate(std::size_t(nB));
    if (nI)
    {
        kLsCeInternal<<<nBlocks(nI), BS>>>
        (
            nI, dm.owner.data(), dm.neighbour.data(), dm.weights.data(),
            meshVec(dm.C), Ce.in.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (nB)
    {
        kLsCeBoundary<<<nBlocks(nB), BS>>>
        (
            nB, dm.fvBoundaryFaces.data(), dm.owner.data(), meshVec(dm.Sf),
            dm.magSf.data(), dm.nonOrthDeltaCoeffs.data(), meshVec(dm.C),
            Ce.bnd.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }

    DField<vec3> mean(static_cast<std::size_t>(nC));
    if (nC)
    {
        kLsMeanPoints<<<nBlocks(nC), BS>>>
        (
            gatherOf(dm), dm.magSf.data(), Ce.in.cview(), Ce.bnd.cview(),
            ls.weightSum.data(), mean.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }

    SurfField<double> wd;
    wd.in.allocate(std::size_t(nI));
    wd.bnd.allocate(std::size_t(nB));
    if (nI + nB)
    {
        kLsNormFaces<<<nBlocks(nI + nB), BS>>>
        (
            nI, nB, dm.owner.data(), dm.neighbour.data(), dm.fvBoundaryFaces.data(),
            dm.magSf.data(), meshVec(dm.C), meshVec(dm.fvbDelta),
            wd.in.view().p[0], wd.bnd.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    DeviceArray<double> norm;
    surfaceSum(dm, wd, norm);

    // 1D／2D 的穩定化：emptyCmpt = (one - solutionD)/2，
    // dd += cmptMultiply(emptyCmpt*emptyCmpt, I)（對角 0 或 1，其餘 0）
    const bool stabilise = dm.nSolutionD < 3;
    tens stab = zeroOf<tens>();
    const int e[3] =
    {
        (1 - dm.solutionD[0])/2, (1 - dm.solutionD[1])/2, (1 - dm.solutionD[2])/2
    };
    stab.xx = double(e[0]*e[0]);
    stab.yy = double(e[1]*e[1]);
    stab.zz = double(e[2]*e[2]);

    DField<tens> invDd(static_cast<std::size_t>(nC));
    if (nC)
    {
        kLsInvDd<<<nBlocks(nC), BS>>>
        (
            gatherOf(dm), dm.magSf.data(), Ce.in.cview(), Ce.bnd.cview(),
            mean.cview(), norm.data(), stabilise, stab, invDd.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }

    ls.P.in.allocate(std::size_t(nI));
    ls.P.bnd.allocate(std::size_t(nB));
    ls.N.allocate(std::size_t(nI));
    if (nI)
    {
        kLsVectorsInternal<<<nBlocks(nI), BS>>>
        (
            nI, dm.owner.data(), dm.neighbour.data(), dm.magSf.data(), norm.data(),
            invDd.cview(), Ce.in.cview(), ls.P.in.view(), ls.N.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (nB)
    {
        kLsVectorsBoundary<<<nBlocks(nB), BS>>>
        (
            nB, dm.fvBoundaryFaces.data(), dm.owner.data(), dm.magSf.data(),
            norm.data(), invDd.cview(), Ce.bnd.cview(), ls.P.bnd.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    return ls;
}


template<class T>
void faceLeastSquaresGrad
(
    const DeviceMesh& dm,
    const LSVectors& ls,
    const VolField<T>& vf,
    VolField<typename GradOf<T>::type>& out
)
{
    using G = typename GradOf<T>::type;

    SurfField<T> ssf;
    interpolateLinear(dm, vf, ssf);

    // fixedGradient 族的 patch：面值改為相鄰格值 + gradient/nonOrthDeltaCoeffs
    // （faceLeastSquaresGrad.C 的 isA<fixedGradientFvPatchField>）
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const std::int32_t n = dm.fvPatchSize[pi];
        if (n && isFixedGradientFamily(vf.bc[pi].type))
        {
            kLsFixedGradientFace<T><<<nBlocks(n), BS>>>
            (
                dm.fvPatchStart[pi], n, dm.fvBoundaryFaces.data(), dm.owner.data(),
                dm.nonOrthDeltaCoeffs.data(), vf.in.cview(), vf.bc[pi].gradient.cview(),
                ssf.bnd.view()
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
    }

    DField<T> mean(std::size_t(dm.nCells));
    out = makeVolField<G>(dm, "grad(" + vf.name + ")", Bc::zeroGradient);
    if (dm.nCells)
    {
        kLsMean<T><<<nBlocks(dm.nCells), BS>>>
        (
            gatherOf(dm), dm.magSf.data(), ssf.in.cview(), ssf.bnd.cview(),
            ls.weightSum.data(), mean.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
        kLsGradCells<T><<<nBlocks(dm.nCells), BS>>>
        (
            gatherOf(dm), ls.P.in.cview(), ls.N.cview(), ls.P.bnd.cview(),
            ssf.in.cview(), ssf.bnd.cview(), mean.cview(), out.in.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    evaluateBCs(dm, out, nullptr);
}


template<class T>
void gradByScheme
(
    const DeviceMesh& dm,
    const LSVectors& ls,
    const std::string& scheme,
    const VolField<T>& vf,
    VolField<typename GradOf<T>::type>& out
)
{
    if (scheme == "faceLeastSquares linear")
    {
        faceLeastSquaresGrad(dm, ls, vf, out);
    }
    else if (scheme == "Gauss linear")
    {
        gaussGrad(dm, vf, out);
    }
    else if
    (
        (scheme.rfind("cellLimited Gauss linear ", 0) == 0)
     || (scheme.rfind("cellLimited<minmod> Gauss linear ", 0) == 0)
    )
    {
        // cellLimitedGrad(mesh, schemeData)：基本方案 Gauss linear，k = readScalar
        const std::string ks = scheme.substr(scheme.rfind(' ') + 1);
        double k = 0;
        if (!parseScalar(ks, k) || k < 0 || k > 1)
        {
            throw std::runtime_error("grad scheme '" + scheme + "': bad coefficient");
        }
        cellLimitedGaussGrad(dm, vf, k, out);
    }
    else
    {
        throw std::runtime_error("grad scheme '" + scheme + "' is not supported");
    }
}


template<class T>
void snGradCompact
(
    const DeviceMesh& dm,
    const VolField<T>& vf,
    const VolField<typename GradOf<T>::type>& grad,
    SurfField<T>& out
)
{
    out.in.allocate(std::size_t(dm.nInternalFaces));
    out.bnd.allocate(std::size_t(dm.nFvBoundaryFaces()));
    if (dm.nInternalFaces)
    {
        kSnGradCompactInternal<T><<<nBlocks(dm.nInternalFaces), BS>>>
        (
            dm.nInternalFaces, dm.owner.data(), dm.neighbour.data(),
            dm.weights.data(), dm.deltaCoeffs.data(), dm.magSf.data(),
            meshVec(dm.Sf), meshVec(dm.C), vf.in.cview(), grad.in.cview(),
            out.in.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (dm.nFvBoundaryFaces())
    {
        DField<T> snBC;
        snGradBCs(dm, vf, snBC);
        kSnGradCompactBoundary<T><<<nBlocks(dm.nFvBoundaryFaces()), BS>>>
        (
            dm.nFvBoundaryFaces(), dm.fvBoundaryFaces.data(), dm.deltaCoeffs.data(),
            dm.magSf.data(), meshVec(dm.Sf), meshVec(dm.fvbDelta), snBC.cview(),
            grad.bnd.cview(), out.bnd.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
}


template<class T>
void laplacianCompact
(
    const DeviceMesh& dm,
    const VolField<double>& gamma,
    const VolField<T>& vf,
    const VolField<typename GradOf<T>::type>& grad,
    DField<T>& out
)
{
    SurfField<double> gammaf;
    interpolateLinear(dm, gamma, gammaf);
    SurfField<T> sn;
    snGradCompact(dm, vf, grad, sn);

    SurfField<T> flux;
    flux.in.allocate(std::size_t(dm.nInternalFaces));
    flux.bnd.allocate(std::size_t(dm.nFvBoundaryFaces()));
    if (dm.nInternalFaces)
    {
        kLaplacianFlux<T><<<nBlocks(dm.nInternalFaces), BS>>>
        (
            dm.nInternalFaces, nullptr, dm.magSf.data(), gammaf.in.cview(),
            sn.in.cview(), flux.in.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (dm.nFvBoundaryFaces())
    {
        kLaplacianFlux<T><<<nBlocks(dm.nFvBoundaryFaces()), BS>>>
        (
            dm.nFvBoundaryFaces(), dm.fvBoundaryFaces.data(), dm.magSf.data(),
            gammaf.bnd.cview(), sn.bnd.cview(), flux.bnd.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    surfaceIntegrate(dm, flux, out);
}


void divGaussLinear(const DeviceMesh& dm, const VolField<tens>& t, DField<vec3>& out)
{
    const SurfField<vec3> Sf = meshSurfVector(dm, dm.Sf);
    SurfField<vec3> flux;
    dotInterpolateLinear(dm, Sf, t, flux);
    surfaceIntegrate(dm, flux, out);
}


// 顯式實體化
template void interpolateLinear<double>(const DeviceMesh&, const VolField<double>&, SurfField<double>&);
template void interpolateLinear<vec3>(const DeviceMesh&, const VolField<vec3>&, SurfField<vec3>&);
template void interpolateLinear<tens>(const DeviceMesh&, const VolField<tens>&, SurfField<tens>&);
template void dotInterpolateLinear<vec3>(const DeviceMesh&, const SurfField<vec3>&, const VolField<vec3>&, SurfField<double>&);
template void dotInterpolateLinear<tens>(const DeviceMesh&, const SurfField<vec3>&, const VolField<tens>&, SurfField<vec3>&);
template void surfaceIntegrate<double>(const DeviceMesh&, const SurfField<double>&, DField<double>&);
template void surfaceIntegrate<vec3>(const DeviceMesh&, const SurfField<vec3>&, DField<vec3>&);
template void multiplySurf<double>(const SurfField<double>&, const SurfField<double>&, SurfField<double>&);
template void multiplySurf<vec3>(const SurfField<double>&, const SurfField<vec3>&, SurfField<vec3>&);
template void gaussGrad<double>(const DeviceMesh&, const VolField<double>&, VolField<vec3>&);
template void gaussGrad<vec3>(const DeviceMesh&, const VolField<vec3>&, VolField<tens>&);
template void faceLeastSquaresGrad<double>(const DeviceMesh&, const LSVectors&, const VolField<double>&, VolField<vec3>&);
template void faceLeastSquaresGrad<vec3>(const DeviceMesh&, const LSVectors&, const VolField<vec3>&, VolField<tens>&);
template void cellLimitedGaussGrad<double>(const DeviceMesh&, const VolField<double>&, double, VolField<vec3>&);
template void cellLimitedGaussGrad<vec3>(const DeviceMesh&, const VolField<vec3>&, double, VolField<tens>&);
template void gradByScheme<double>(const DeviceMesh&, const LSVectors&, const std::string&, const VolField<double>&, VolField<vec3>&);
template void gradByScheme<vec3>(const DeviceMesh&, const LSVectors&, const std::string&, const VolField<vec3>&, VolField<tens>&);
template void snGradCompact<double>(const DeviceMesh&, const VolField<double>&, const VolField<vec3>&, SurfField<double>&);
template void snGradCompact<vec3>(const DeviceMesh&, const VolField<vec3>&, const VolField<tens>&, SurfField<vec3>&);
template void laplacianCompact<double>(const DeviceMesh&, const VolField<double>&, const VolField<double>&, const VolField<vec3>&, DField<double>&);
template void laplacianCompact<vec3>(const DeviceMesh&, const VolField<double>&, const VolField<vec3>&, const VolField<tens>&, DField<vec3>&);

} // namespace gnc
