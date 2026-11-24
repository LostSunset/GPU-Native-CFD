// 純量的 fvMatrix：參考次序版。GPU-Native-CFD，GPL-3.0-or-later。
//
// 出處（Atlas upstream/openfoam，v2606；overlay 為 Atlas 的 atlas-overlay/openfoam）：
//   lduMatrix.C               diag()、lower()、upper() 的配置
//   lduMatrixOperations.C     negSumDiag、sumMagOffDiag、operator+=、-=、negate
//                             （overlay：AtomicAdd 依面編號循序）
//   lduMatrixATmul.C          Amul、sumA、residual（overlay；hasLowerCSR() 在
//                             序列執行時為假，走面迴圈）
//   fvMatrix.C:1101–1258      relax(alpha)、relax()
//   fvMatrix.C:1600–1760、2015–2660  operator+=、-=、全域運算子
//   fvScalarMatrix.C          solveSegregated、addBoundaryDiag、addBoundarySource
//   smoothSolver.C            solve（normFactor、checkConvergence、迴圈）
//   symGaussSeidelSmoother.C  smooth
//   lduMatrixSolver.C         normFactor（DEFAULT_NORM）、readControls
//   SolverPerformance.C       checkConvergence（small_ = 1e-20）
//   gaussConvectionScheme.C   fvmDiv；upwind.H：weights = pos0(phi)
//   boundedConvectionScheme.C、boundedDdtScheme.C
//   HiSA dualTimeDdtScheme.C  fvmDdt(rho, vf)
//   gaussLaplacianScheme.C    fvmLaplacianUncorrected；gaussLaplacianSchemes.C
//                             的純量特化（非正交修正 source -= V*div(gammaMagSf*corr)）
//   correctedSnGrad           deltaCoeffs = nonOrthDeltaCoeffs、corrected() 為真、
//                             fullGradCorrection = linear dotInterpolate(
//                             nonOrthCorrectionVectors, grad)
//   fixedValueFvPatchField    valueInternalCoeffs 0、valueBoundaryCoeffs = 值；
//                             gradientInternalCoeffs = -1*patch().deltaCoeffs()、
//                             gradientBoundaryCoeffs = deltaCoeffs*值
//   HiSA boundaryCorrectedFixedValue  gradient 係數改用 nonOrthDeltaCoeffs
//   zeroGradientFvPatchField  valueInternalCoeffs 1、其他為 0
//   fvmSup.C                  Sp：diag += V*sp
//   bound.C、fvcAverage.C
// 必須以 --fmad=false 編譯。

#include "fvm/fvScalarMatrix.cuh"

#include "core/coopLevels.cuh"
#include "core/arrayOps.cuh"
#include "core/reduce.cuh"
#include "fvm/amg.cuh"

#include <cooperative_groups.h>
#include "io/dictionary.hpp"
#include "io/foamTokenizer.hpp"
#include "ops/fvcOps.cuh"

#include <cmath>
#include <memory>
#include <regex>
#include <stdexcept>

namespace gnc
{

namespace
{

constexpr int BS = 256;

int nBlocks(const std::size_t n)
{
    return int((n + BS - 1)/BS);
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
};

Gather gatherOf(const DeviceMesh& dm)
{
    return
    {
        dm.nCells, dm.nInternalFaces,
        dm.cellFaceOffsets.data(), dm.cellFaces.data(), dm.cellFaceSigns.data(),
        dm.faceToFvB.data()
    };
}

__device__ __forceinline__ std::int32_t fvbOf(const Gather& g, const std::int32_t f)
{
    return g.faceToFvB[f - g.nInternal];
}


__global__ void kFill(const std::size_t n, double* __restrict__ p, const double v)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n)
    {
        p[i] = v;
    }
}

__global__ void kCopy(const std::size_t n, const double* __restrict__ s, double* __restrict__ d)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n)
    {
        d[i] = s[i];
    }
}

// a = a + b（Field::operator+=）
__global__ void kAddTo(const std::size_t n, double* __restrict__ a, const double* __restrict__ b)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[i] = a[i] + b[i];
    }
}

// a = a + V*s（diag += V*sp；A - su 的 source += V*su）
__global__ void kAddVTimes
(
    const std::size_t n,
    double* __restrict__ a,
    const double* __restrict__ V,
    const double* __restrict__ s
)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[i] = a[i] + V[i]*s[i];
    }
}

// a = a - V*s（su - A 的 source -= V*su）
__global__ void kSubVTimes
(
    const std::size_t n,
    double* __restrict__ a,
    const double* __restrict__ V,
    const double* __restrict__ s
)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[i] = a[i] - V[i]*s[i];
    }
}


void fill(DeviceArray<double>& a, const double v)
{
    if (a.size())
    {
        kFill<<<nBlocks(a.size()), BS>>>(a.size(), a.data(), v);
        GNC_CUDA_LAUNCH_CHECK();
    }
}

void copyArray(const DeviceArray<double>& s, DeviceArray<double>& d)
{
    d.allocate(s.size());
    if (s.size())
    {
        kCopy<<<nBlocks(s.size()), BS>>>(s.size(), s.data(), d.data());
        GNC_CUDA_LAUNCH_CHECK();
    }
}

// lduMatrix::negSumDiag（overlay）：每面 diag[l] += -lower、diag[u] += -upper，
// 依面編號。symmetric 時 lower 傳 upper（const lower() 回傳 upper）
__global__ void kNegSumDiag
(
    const Gather g,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    double* __restrict__ diag
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= g.nCells)
    {
        return;
    }
    double acc = diag[c];
    for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
    {
        const std::int32_t f = g.face[k];
        if (f < g.nInternal)
        {
            acc += (g.sign[k] > 0) ? -lower[f] : -upper[f];
        }
    }
    diag[c] = acc;
}


// gaussConvectionScheme::fvmDiv（upwind）的內部面：
// lower = (-pos0(phi))*phi；upper = lower + phi
__global__ void kUpwindFaces
(
    const std::int32_t nIF,
    const double* __restrict__ phi,
    const double* __restrict__ weights,     // nullptr：upwind（pos0(phi)）
    double* __restrict__ lower,
    double* __restrict__ upper
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIF)
    {
        return;
    }
    const double w = weights ? weights[f] : ((phi[f] >= 0.0) ? 1.0 : 0.0);
    const double l = (-w)*phi[f];
    lower[f] = l;
    upper[f] = l + phi[f];
}

// limitedLinear k 的內部面權重（LimitedScheme.C 的 calcLimiter、NVDTVD.H 的 r、
// limitedLinear.H 的 limiter、limitedSurfaceInterpolationScheme.C 的 weights；
// 純量的 limitFuncs::magSqr 回傳場本身）：
//   gradf  = phiN - phiP；gradcf = d & (faceFlux > 0 ? gradcP : gradcN)，d = C_N - C_P
//   r      = |gradcf| >= 1000*|gradf| ? ((2*1000)*sign(gradcf))*sign(gradf) - 1
//                                     : 2*(gradcf/gradf) - 1
//   lim    = clamp(twoByk*r, 0, 1)
//   w      = lim*CDweight + (1 - lim)*pos0(faceFlux)
__global__ void kLimitedLinearWeights
(
    const std::int32_t nIF,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ phi,
    const double* __restrict__ psi,
    const CView<vec3> grad,
    const double* __restrict__ Cx,
    const double* __restrict__ Cy,
    const double* __restrict__ Cz,
    const double* __restrict__ CDweights,
    const double twoByk,
    double* __restrict__ w
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIF)
    {
        return;
    }
    const std::int32_t P = owner[f];
    const std::int32_t N = neighbour[f];
    const double flux = phi[f];
    const vec3 d{Cx[N] - Cx[P], Cy[N] - Cy[P], Cz[N] - Cz[P]};
    const double gradf = psi[N] - psi[P];
    const double gradcf = dot(d, flux > 0 ? grad[P] : grad[N]);
    double r;
    if (::fabs(gradcf) >= 1000*::fabs(gradf))
    {
        const double sc = gradcf >= 0 ? 1.0 : -1.0;
        const double sf = gradf >= 0 ? 1.0 : -1.0;
        r = (2*1000)*sc*sf - 1;
    }
    else
    {
        r = 2*(gradcf/gradf) - 1;
    }
    const double v = twoByk*r;
    const double lim = (v < 0) ? 0.0 : (1 < v) ? 1.0 : v;
    w[f] = lim*CDweights[f] + (1.0 - lim)*((flux >= 0.0) ? 1.0 : 0.0);
}

// 純量邊界條件在矩陣中的型別
enum PatchKind : int
{
    pkFixedValue = 0,           // fixedValue 族（含壁面函數）
    pkBoundaryCorrected = 1,    // HiSA boundaryCorrectedFixedValue（gradient 用 nonOrthDeltaCoeffs）
    pkZeroGradient = 2,
    pkInletOutlet = 3,          // mixed：valueFraction = 1 - pos0(phi_b)、refGrad = 0
    pkSymmetry = 4,             // transformFvPatchField 的純量（旋轉不變型別）
    pkMixed = 5,                // 其他 mixed 族：refValue、refGrad、valueFraction 已算好
    pkFixedGradient = 6         // fixedGradient 族：gradient 已算好
};

struct PatchIn
{
    std::int32_t start;
    std::int32_t n;
    int kind;
    const std::int32_t* fvbFaces;
    const std::int32_t* owner;
    const double* phiB;
    const double* valueB;       // 邊界值
    const double* refValueB;    // inletOutlet、mixed 的 refValue（patch 內的索引）
    const double* refGradB;     // mixed 的 refGrad（patch 內）
    const double* valueFraction;    // mixed（patch 內）
    const double* gradientB;    // fixedGradient 的 gradient（patch 內）
    const double* psiIn;        // 內部值（patchInternalField）
};

// inletOutletFvPatchField::updateCoeffs：valueFraction = 1 - pos0(phi_b)
__device__ __forceinline__ double inletOutletVf(const double phiB)
{
    return 1.0 - ((phiB >= 0.0) ? 1.0 : 0.0);
}

// 邊界：internalCoeffs = phi_b*valueInternalCoeffs；
// boundaryCoeffs = (-phi_b)*valueBoundaryCoeffs
//   fixedValue      vIC 0、vBC 值
//   zeroGradient    vIC 1、vBC 0
//   inletOutlet     vIC 1*(1 - vf)、vBC lerp(refGrad/dc, refValue, vf)（mixedFvPatchField.txx）
//   mixed           同上，refGrad、vf 取存好的值
//   fixedGradient   vIC 1、vBC gradient/dc（fixedGradientFvPatchField.txx）
//   symmetry（純量）vIC 1、vBC 值 - 1*pif（transformFvPatchField.txx）
__global__ void kUpwindPatch(const PatchIn p, const double* __restrict__ dc, double* __restrict__ ic, double* __restrict__ bc)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= p.n)
    {
        return;
    }
    const std::int32_t b = p.start + i;
    const double phi = p.phiB[b];
    double vIC = 0.0;
    double vBC = 0.0;
    switch (p.kind)
    {
        case pkFixedValue:
        case pkBoundaryCorrected:
            vIC = 0.0;
            vBC = p.valueB[b];
            break;
        case pkZeroGradient:
            vIC = 1.0;
            vBC = 0.0;
            break;
        case pkInletOutlet:
        {
            const double vf = inletOutletVf(phi);
            const std::int32_t f = p.fvbFaces[b];
            vIC = 1.0*(1.0 - vf);
            vBC = lerp(0.0/dc[f], p.refValueB[i], vf);
            break;
        }
        case pkMixed:
        {
            const double vf = p.valueFraction[i];
            const std::int32_t f = p.fvbFaces[b];
            vIC = 1.0*(1.0 - vf);
            vBC = lerp(p.refGradB[i]/dc[f], p.refValueB[i], vf);
            break;
        }
        case pkFixedGradient:
        {
            const std::int32_t f = p.fvbFaces[b];
            vIC = 1.0;
            vBC = p.gradientB[i]/dc[f];
            break;
        }
        case pkSymmetry:
        {
            const double pif = p.psiIn[p.owner[p.fvbFaces[b]]];
            vIC = 1.0;
            vBC = p.valueB[b] - 1.0*pif;
            break;
        }
    }
    ic[b] = phi*vIC;
    bc[b] = (-phi)*vBC;
}


// fvmLaplacianUncorrected 的內部面：upper = deltaCoeffs*gammaMagSf，
// gammaMagSf = gamma_f*magSf
__global__ void kLaplacianFaces
(
    const std::int32_t nIF,
    const double* __restrict__ gammaF,
    const double* __restrict__ magSf,
    const double* __restrict__ nonOrthDc,
    double* __restrict__ gammaMagSf,
    double* __restrict__ upper
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIF)
    {
        return;
    }
    const double gms = gammaF[f]*magSf[f];
    gammaMagSf[f] = gms;
    upper[f] = nonOrthDc[f]*gms;
}

// 邊界（非耦合）：internalCoeffs = pGamma*gradientInternalCoeffs()；
// boundaryCoeffs = (-pGamma)*gradientBoundaryCoeffs()
//   fixedValue          gIC = (-1)*dc、gBC = dc*值（patch().deltaCoeffs()）
//   boundaryCorrected   同上，dc 改為 nonOrthDeltaCoeffs
//   zeroGradient        gIC = 0、gBC = 0
//   inletOutlet         gIC = ((-1)*vf)*dc、gBC = lerp(refGrad, dc*refValue, vf)
//   mixed               同上，refGrad、vf 取存好的值
//   fixedGradient       gIC = 0、gBC = gradient
//   symmetry（純量）    gIC = 0、gBC = snGrad() - 0*pif = 0 - 0*pif
__global__ void kLaplacianPatch
(
    const PatchIn p,
    const double* __restrict__ gammaFB,
    const double* __restrict__ magSf,
    const double* __restrict__ dc,
    const double* __restrict__ nonOrthDc,
    double* __restrict__ gammaMagSfB,
    double* __restrict__ ic,
    double* __restrict__ bc
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= p.n)
    {
        return;
    }
    const std::int32_t b = p.start + i;
    const std::int32_t f = p.fvbFaces[b];
    const double pGamma = gammaFB[b]*magSf[f];
    gammaMagSfB[b] = pGamma;
    double gIC = 0.0;
    double gBC = 0.0;
    switch (p.kind)
    {
        case pkFixedValue:
        case pkBoundaryCorrected:
        {
            const double d = (p.kind == pkFixedValue) ? dc[f] : nonOrthDc[f];
            gIC = (-1.0)*d;
            gBC = d*p.valueB[b];
            break;
        }
        case pkZeroGradient:
            break;
        case pkInletOutlet:
        {
            const double vf = inletOutletVf(p.phiB[b]);
            gIC = ((-1.0)*vf)*dc[f];
            gBC = lerp(0.0, dc[f]*p.refValueB[i], vf);
            break;
        }
        case pkMixed:
        {
            const double vf = p.valueFraction[i];
            gIC = ((-1.0)*vf)*dc[f];
            gBC = lerp(p.refGradB[i], dc[f]*p.refValueB[i], vf);
            break;
        }
        case pkFixedGradient:
            gIC = 0.0;
            gBC = p.gradientB[i];
            break;
        case pkSymmetry:
        {
            const double pif = p.psiIn[p.owner[f]];
            gIC = 0.0;
            gBC = 0.0 - 0.0*pif;
            break;
        }
    }
    ic[b] = pGamma*gIC;
    bc[b] = (-pGamma)*gBC;
}

// limitedSnGrad::correction：corr *= min((lc*|sn|)/(((1 - lc)*|corr|) + SMALL), 1)；
// 內部面 sn = nonOrthDeltaCoeffs*(psi[N] - psi[P])，邊界面 sn 由呼叫端給（patch 的 snGrad()）
__global__ void kLimitCorrection
(
    const std::int32_t n,
    const std::int32_t* __restrict__ owner,         // 內部面時；邊界面為 nullptr
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ nonOrthDc,
    const double* __restrict__ psi,
    const double* __restrict__ snB,                 // 邊界面時
    const double lc,
    const double oneMinusLc,
    double* __restrict__ corr
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const double sn = owner ? nonOrthDc[i]*(psi[neighbour[i]] - psi[owner[i]]) : snB[i];
    const double lim = foamMin((lc*::fabs(sn))/((oneMinusLc*::fabs(corr[i])) + SMALL), 1.0);
    corr[i] = lim*corr[i];
}


// 面場逐面相乘（內部面與邊界面分開呼叫）
__global__ void kMulFaces
(
    const std::int32_t n,
    const double* __restrict__ a,
    const double* __restrict__ b,
    double* __restrict__ out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = a[i]*b[i];
    }
}


// HiSA dualTimeDdtScheme::fvmDdt(rho, vf)：
// diag += (rPseudoDeltaT*rho)*V；source += ((rPseudoDeltaT*rho)*vf)*V
__global__ void kDdtDualTime
(
    const std::int32_t n,
    const double* __restrict__ rPDT,
    const double* __restrict__ rho,
    const double* __restrict__ psi,
    const double* __restrict__ V,
    double* __restrict__ diag,
    double* __restrict__ source
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const double r = rPDT[i]*rho[i];
    diag[i] = diag[i] + r*V[i];
    source[i] = source[i] + (r*psi[i])*V[i];
}


// fvMatrix::relax(alpha)，非耦合 patch：
//   D0 = D；sumOff = sumMagOffDiag（overlay：sumOff[u] += |lower|、
//   sumOff[l] += |upper|，依面編號）；D += cmptMax(cmptMag(iCoeffs))（依 patch）；
//   D = max(|D|, sumOff)；D /= alpha；D -= cmptMin(iCoeffs)；S += (D - D0)*psi
__global__ void kRelax
(
    const Gather g,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    const double* __restrict__ ic,
    const double* __restrict__ psi,
    const double alpha,
    double* __restrict__ D,
    double* __restrict__ S
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= g.nCells)
    {
        return;
    }
    const double d0 = D[c];
    double d = d0;
    double sumOff = 0.0;
    for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
    {
        const std::int32_t f = g.face[k];
        if (f < g.nInternal)
        {
            sumOff += (g.sign[k] > 0) ? ::fabs(upper[f]) : ::fabs(lower[f]);
        }
        else
        {
            const std::int32_t b = fvbOf(g, f);
            if (b >= 0)
            {
                d += ::fabs(ic[b]);
            }
        }
    }
    d = foamMax(::fabs(d), sumOff);
    d /= alpha;
    for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
    {
        const std::int32_t f = g.face[k];
        if (f >= g.nInternal)
        {
            const std::int32_t b = fvbOf(g, f);
            if (b >= 0)
            {
                d -= ic[b];
            }
        }
    }
    D[c] = d;
    S[c] = S[c] + (d - d0)*psi[c];
}


// addBoundaryDiag／addBoundarySource（非耦合）：依 patch、面的次序 out[c] += coeff[b]
__global__ void kAddBoundary
(
    const Gather g,
    const double* __restrict__ in,
    const double* __restrict__ coeff,
    double* __restrict__ out
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= g.nCells)
    {
        return;
    }
    double acc = in[c];
    for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
    {
        const std::int32_t f = g.face[k];
        if (f >= g.nInternal)
        {
            const std::int32_t b = fvbOf(g, f);
            if (b >= 0)
            {
                acc += coeff[b];
            }
        }
    }
    out[c] = acc;
}


// lduMatrix::Amul（overlay，面迴圈）：Apsi = D*psi；Apsi[u] += lower*psi[l]、
// Apsi[l] += upper*psi[u]，依面編號
__global__ void kAmul
(
    const Gather g,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ D,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    const double* __restrict__ psi,
    double* __restrict__ Apsi
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= g.nCells)
    {
        return;
    }
    double acc = D[c]*psi[c];
    for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
    {
        const std::int32_t f = g.face[k];
        if (f < g.nInternal)
        {
            acc += (g.sign[k] > 0) ? upper[f]*psi[neighbour[f]] : lower[f]*psi[owner[f]];
        }
    }
    Apsi[c] = acc;
}

// lduMatrix::sumA（overlay）：sumA = D；sumA[u] += lower、sumA[l] += upper
__global__ void kSumA
(
    const Gather g,
    const double* __restrict__ D,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    double* __restrict__ sumA
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= g.nCells)
    {
        return;
    }
    double acc = D[c];
    for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
    {
        const std::int32_t f = g.face[k];
        if (f < g.nInternal)
        {
            acc += (g.sign[k] > 0) ? upper[f] : lower[f];
        }
    }
    sumA[c] = acc;
}

// lduMatrix::residual（overlay）：rA = source - D*psi；
// rA[u] += (-lower)*psi[l]、rA[l] += (-upper)*psi[u]
__global__ void kResidual
(
    const Gather g,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ D,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    const double* __restrict__ source,
    const double* __restrict__ psi,
    double* __restrict__ rA
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= g.nCells)
    {
        return;
    }
    double acc = source[c] - D[c]*psi[c];
    for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
    {
        const std::int32_t f = g.face[k];
        if (f < g.nInternal)
        {
            acc += (g.sign[k] > 0) ? (-upper[f])*psi[neighbour[f]] : (-lower[f])*psi[owner[f]];
        }
    }
    rA[c] = acc;
}

// normFactor 的逐格項：|Apsi - sumA*avg| + |source - sumA*avg|
__global__ void kNormTerms
(
    const std::int32_t n,
    const double* __restrict__ Apsi,
    const double* __restrict__ sumA,
    const double avg,
    const double* __restrict__ source,
    double* __restrict__ out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const double t = sumA[i]*avg;
        out[i] = ::fabs(Apsi[i] - t) + ::fabs(source[i] - t);
    }
}

// 同 kNormTerms，平均值由裝置端的 sum(psi) 算（sumTreeToDevice 的結果；avg = sum/n 與主機端相同，N-028）
__global__ void kNormTermsDev
(
    const std::int32_t n,
    const double* __restrict__ Apsi,
    const double* __restrict__ sumA,
    const double* __restrict__ sumPsi,
    const double* __restrict__ source,
    double* __restrict__ out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const double avg = sumPsi[0]/double(n);
        const double t = sumA[i]*avg;
        out[i] = ::fabs(Apsi[i] - t) + ::fabs(source[i] - t);
    }
}

// 求解器之間共用的裝置端純量（sumTreeToDevice 的輸出；N-028）
double* deviceScalar()
{
    static DeviceArray<double> s(8);
    return s.data();
}

// mapped pinned 的純量（裝置寫、主機在同步之後讀；N-028）
struct MappedScalars
{
    double* h = nullptr;
    double* d = nullptr;
    MappedScalars()
    {
        GNC_CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&h), 8*sizeof(double), cudaHostAllocMapped));
        GNC_CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&d), h, 0));
        for (int k = 0; k < 8; ++k)
        {
            h[k] = 0.0;
        }
    }
};

MappedScalars& mappedScalars()
{
    static MappedScalars m;
    return m;
}

__global__ void kMagOf(const std::int32_t n, const double* __restrict__ a, double* __restrict__ out)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = ::fabs(a[i]);
    }
}

// sumSolver(|a|)：效能版（GNC_REDUCE=tree）以合併的單次啟動歸約（N-027，結果相同），
// 參考版照舊寫出 |a| 再循序相加
double solverSumMag(const std::int32_t n, const double* a, double* work)
{
    if (useTreeReduce())
    {
        return sumTreeOp(ReduceOp::Mag, a, nullptr, std::size_t(n));
    }
    kMagOf<<<nBlocks(std::size_t(n)), BS>>>(n, a, work);
    GNC_CUDA_LAUNCH_CHECK();
    return sumSolver(work, std::size_t(n));
}

// residual = source - Apsi（smoothSolver 的初始殘差）
__global__ void kSourceMinus
(
    const std::int32_t n,
    const double* __restrict__ source,
    const double* __restrict__ Apsi,
    double* __restrict__ out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = source[i] - Apsi[i];
    }
}


// symGaussSeidelSmoother::smooth 的正向（一層）：bPrime[c] = source[c] 依
// owner 遞增減去 lower*psi[owner]（循序版在處理 owner 時分配到 bPrime）；
// psii = bPrime[c]，依面編號減去 upper*psi[u]（u > c，尚未更新）；/= D
struct SgsArgs
{
    const std::int32_t* lowOff;
    const std::int32_t* lowFaces;
    const std::int32_t* ownerStart;
    const std::int32_t* owner;
    const std::int32_t* neighbour;
    const double* D;
    const double* lower;
    const double* upper;
    const double* source;
    double* bPrime;
    double* psi;
};

__device__ __forceinline__ void sgsForwardCell(const std::int32_t c, const SgsArgs& a)
{
    const std::int32_t* __restrict__ lowOff = a.lowOff;
    const std::int32_t* __restrict__ lowFaces = a.lowFaces;
    const std::int32_t* __restrict__ ownerStart = a.ownerStart;
    const std::int32_t* __restrict__ owner = a.owner;
    const std::int32_t* __restrict__ neighbour = a.neighbour;
    const double* __restrict__ D = a.D;
    const double* __restrict__ lower = a.lower;
    const double* __restrict__ upper = a.upper;
    double* __restrict__ bPrime = a.bPrime;
    double* __restrict__ psi = a.psi;
    double bp = a.source[c];
    for (std::int32_t q = lowOff[c]; q < lowOff[c + 1]; ++q)
    {
        const std::int32_t f = lowFaces[q];
        bp -= lower[f]*psi[owner[f]];
    }
    bPrime[c] = bp;
    double psii = bp;
    for (std::int32_t f = ownerStart[c]; f < ownerStart[c + 1]; ++f)
    {
        psii -= upper[f]*psi[neighbour[f]];
    }
    psii /= D[c];
    psi[c] = psii;
}

__global__ void kSgsForward(const std::int32_t nLevelCells, const std::int32_t* __restrict__ cells, const SgsArgs a)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < nLevelCells)
    {
        sgsForwardCell(cells[i], a);
    }
}

// 反向（一層）：psii = bPrime[c]（正向留下的），依面編號減去 upper*psi[u]
// （u > c，已在反向更新）；/= D
__device__ __forceinline__ void sgsBackwardCell(const std::int32_t c, const SgsArgs& a)
{
    const std::int32_t* __restrict__ ownerStart = a.ownerStart;
    const std::int32_t* __restrict__ neighbour = a.neighbour;
    const double* __restrict__ upper = a.upper;
    double* __restrict__ psi = a.psi;
    double psii = a.bPrime[c];
    for (std::int32_t f = ownerStart[c]; f < ownerStart[c + 1]; ++f)
    {
        psii -= upper[f]*psi[neighbour[f]];
    }
    psii /= a.D[c];
    psi[c] = psii;
}

__global__ void kSgsBackward(const std::int32_t nLevelCells, const std::int32_t* __restrict__ cells, const SgsArgs a)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < nLevelCells)
    {
        sgsBackwardCell(cells[i], a);
    }
}

// 多色 symGaussSeidel（效能版，GNC_SGS=color）：同色的格互不相鄰，每格
// psi_c = (source_c - Σ a_cj psi_j)/D_c，鄰格取目前的值（正向依色遞增、反向依色
// 遞減，即以著色次序的 Gauss-Seidel）。面的編碼同 LusgsPlan::cLowFace：負值
// -(f + 1) 為本格是 owner（係數 upper），非負為 neighbour（係數 lower）
__global__ void kSgsColor
(
    const std::int32_t count,
    const std::int32_t* __restrict__ cells,
    const std::int32_t* __restrict__ lowOff,
    const std::int32_t* __restrict__ lowFace,
    const std::int32_t* __restrict__ upOff,
    const std::int32_t* __restrict__ upFace,
    const SgsArgs a
)
{
    const std::int32_t t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= count)
    {
        return;
    }
    const std::int32_t c = cells[t];
    double s = a.source[c];
    for (int side = 0; side < 2; ++side)
    {
        const std::int32_t* off = side ? upOff : lowOff;
        const std::int32_t* code = side ? upFace : lowFace;
        for (std::int32_t q = off[c]; q < off[c + 1]; ++q)
        {
            const std::int32_t k = code[q];
            if (k < 0)
            {
                const std::int32_t f = -(k + 1);
                s -= a.upper[f]*a.psi[a.neighbour[f]];
            }
            else
            {
                s -= a.lower[k]*a.psi[a.owner[k]];
            }
        }
    }
    a.psi[c] = s/a.D[c];
}

// 一次對稱掃描的所有層放在單一 block（層與層之間 __syncthreads()）
__global__ void kSgsSweepBlock
(
    const std::int32_t nFwd,
    const std::int32_t* __restrict__ fwdOff,
    const std::int32_t* __restrict__ fwdCells,
    const std::int32_t nRev,
    const std::int32_t* __restrict__ revOff,
    const std::int32_t* __restrict__ revCells,
    const SgsArgs a
)
{
    for (std::int32_t l = 0; l < nFwd; ++l)
    {
        const std::int32_t e = fwdOff[l + 1];
        for (std::int32_t t = fwdOff[l] + std::int32_t(threadIdx.x); t < e; t += blockDim.x)
        {
            sgsForwardCell(fwdCells[t], a);
        }
        __syncthreads();
    }
    for (std::int32_t l = 0; l < nRev; ++l)
    {
        const std::int32_t e = revOff[l + 1];
        for (std::int32_t t = revOff[l] + std::int32_t(threadIdx.x); t < e; t += blockDim.x)
        {
            sgsBackwardCell(revCells[t], a);
        }
        __syncthreads();
    }
}

// 一次對稱掃描的所有層（正向再反向）放在一個 cooperative kernel
__global__ void kSgsSweepCoop
(
    const std::int32_t nFwd,
    const std::int32_t* __restrict__ fwdOff,
    const std::int32_t* __restrict__ fwdCells,
    const std::int32_t nRev,
    const std::int32_t* __restrict__ revOff,
    const std::int32_t* __restrict__ revCells,
    const SgsArgs a
)
{
    cooperative_groups::grid_group grid = cooperative_groups::this_grid();
    const std::int32_t stride = std::int32_t(grid.size());
    for (std::int32_t l = 0; l < nFwd; ++l)
    {
        const std::int32_t e = fwdOff[l + 1];
        for (std::int32_t t = fwdOff[l] + std::int32_t(grid.thread_rank()); t < e; t += stride)
        {
            sgsForwardCell(fwdCells[t], a);
        }
        grid.sync();
    }
    for (std::int32_t l = 0; l < nRev; ++l)
    {
        const std::int32_t e = revOff[l + 1];
        for (std::int32_t t = revOff[l] + std::int32_t(grid.thread_rank()); t < e; t += stride)
        {
            sgsBackwardCell(revCells[t], a);
        }
        grid.sync();
    }
}


// bound：內部值 max(max(v, avg*pos0(-v)), lb)
__global__ void kBoundInternal
(
    const std::int32_t n,
    const double* __restrict__ avg,
    const double lb,
    double* __restrict__ v
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const double pos0 = (-v[i] >= 0.0) ? 1.0 : 0.0;
        v[i] = foamMax(foamMax(v[i], avg[i]*pos0), lb);
    }
}

// 原地 max(b, lb)
__global__ void kMaxInPlace(const std::int32_t n, double* __restrict__ a, const double lb)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[i] = foamMax(a[i], lb);
    }
}

// inletOutlet 族的 operator=(ptf)：(vf*refValue) + ((1 - vf)*ptf)，ptf = max(b, lb)
__global__ void kBoundInletOutlet
(
    const std::int32_t n,
    const double* __restrict__ vf,
    const double* __restrict__ refValue,
    const double lb,
    double* __restrict__ a
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[i] = vf[i]*refValue[i] + (1.0 - vf[i])*foamMax(a[i], lb);
    }
}

// 相鄰格值（純量的 transform 族 evaluate）
__global__ void kCopyOwner
(
    const std::int32_t n,
    const std::int32_t* __restrict__ fvbFaces,
    const std::int32_t* __restrict__ owner,
    const double* __restrict__ in,
    double* __restrict__ a
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[i] = in[owner[fvbFaces[i]]];
    }
}

// max(field, lb) 逐項
__global__ void kMaxWith(const std::int32_t n, const double* __restrict__ a, const double lb, double* __restrict__ out)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = foamMax(a[i], lb);
    }
}

// fvc::average 的內部值：surfaceSum(magSf*ssf)/surfaceSum(magSf)
__global__ void kDivide(const std::int32_t n, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = a[i]/b[i];
    }
}


// patch 在 fvBoundaryFaces 中的範圍
struct Range
{
    std::int32_t start;
    std::int32_t size;
};

Range rangeOf(const DeviceMesh& dm, const std::size_t pi)
{
    return {dm.fvPatchStart[pi], dm.fvPatchSize[pi]};
}

[[noreturn]] void unsupported(const DeviceMesh& dm, const VolField<double>& psi, const std::size_t pi, const char* what)
{
    throw std::runtime_error
    (
        "field " + psi.name + " patch " + dm.patches[pi].name + ": " + what + " for "
      + bcName(psi.bc[pi].type) + " is not supported"
    );
}

// 一個 patch 在矩陣中的型別與資料
PatchIn patchIn
(
    const DeviceMesh& dm,
    const VolField<double>& psi,
    const SurfField<double>* phi,
    const std::size_t pi,
    const char* what
)
{
    const PatchBC<double>& bc = psi.bc[pi];
    PatchIn p{};
    p.start = dm.fvPatchStart[pi];
    p.n = dm.fvPatchSize[pi];
    p.fvbFaces = dm.fvBoundaryFaces.data();
    p.owner = dm.owner.data();
    p.valueB = psi.bnd.cview().p[0];
    p.psiIn = psi.in.cview().p[0];
    p.phiB = phi ? phi->bnd.cview().p[0] : nullptr;
    const Bc t = bc.type;
    if (t == Bc::boundaryCorrectedFixedValue)
    {
        p.kind = pkBoundaryCorrected;
    }
    else if (isFixedValueFamily(t))
    {
        p.kind = pkFixedValue;
    }
    else if (isZeroGradientFamily(t))
    {
        p.kind = pkZeroGradient;
    }
    else if (isInletOutletFamily(t))
    {
        if (!phi || phi->name != bc.phiName)
        {
            throw std::runtime_error
            (
                "field " + psi.name + " patch " + dm.patches[pi].name
              + ": inletOutlet needs flux " + bc.phiName
            );
        }
        p.kind = pkInletOutlet;
        p.refValueB = bc.refValue.cview().p[0];
    }
    else if (isMixedFamily(t))
    {
        p.kind = pkMixed;
        p.refValueB = bc.refValue.cview().p[0];
        p.refGradB = bc.refGrad.cview().p[0];
        p.valueFraction = bc.valueFraction.data();
    }
    else if (isFixedGradientFamily(t))
    {
        p.kind = pkFixedGradient;
        p.gradientB = bc.gradient.cview().p[0];
    }
    else if (t == Bc::symmetry || t == Bc::symmetryPlane || t == Bc::slip)
    {
        p.kind = pkSymmetry;
    }
    else
    {
        unsupported(dm, psi, pi, what);
    }
    return p;
}

} // namespace


// ---------------------------------------------------------------------------

FvScalarMatrix fvmEmpty(const DeviceMesh& dm)
{
    FvScalarMatrix A;
    A.source.allocate(std::size_t(dm.nCells));
    A.internalCoeffs.allocate(std::size_t(dm.nFvBoundaryFaces()));
    A.boundaryCoeffs.allocate(std::size_t(dm.nFvBoundaryFaces()));
    ArrayBatch b;
    b.fill(A.source.data(), A.source.size(), 0.0);
    b.fill(A.internalCoeffs.data(), A.internalCoeffs.size(), 0.0);
    b.fill(A.boundaryCoeffs.data(), A.boundaryCoeffs.size(), 0.0);
    return A;
}


void ensureDiag(const DeviceMesh& dm, FvScalarMatrix& A)
{
    if (!A.hasDiag)
    {
        A.diag.allocate(std::size_t(dm.nCells));
        fill(A.diag, 0.0);
        A.hasDiag = true;
    }
}


void ensureLower(const DeviceMesh& dm, FvScalarMatrix& A)
{
    if (!A.hasLower)
    {
        if (A.hasUpper)
        {
            copyArray(A.upper, A.lower);
        }
        else
        {
            A.lower.allocate(std::size_t(dm.nInternalFaces));
            fill(A.lower, 0.0);
        }
        A.hasLower = true;
    }
}


void ensureUpper(const DeviceMesh& dm, FvScalarMatrix& A)
{
    if (!A.hasUpper)
    {
        if (A.hasLower)
        {
            copyArray(A.lower, A.upper);
        }
        else
        {
            A.upper.allocate(std::size_t(dm.nInternalFaces));
            fill(A.upper, 0.0);
        }
        A.hasUpper = true;
    }
}


// lduMatrix::operator+=／-=（overlay）的分支（N-028）：先把結構的調整（配置、填 0、由 upper 複製出
// lower）做成一批，再把各陣列的加減做成一批；每個元素的運算與原本逐陣列的 kernel 相同，原本會被
// 覆寫的中間值（先填 0 或複製、再整個覆寫）直接省略
namespace
{

void lduCombine(const DeviceMesh& dm, FvScalarMatrix& A, const FvScalarMatrix& B, const bool minus)
{
    const std::size_t nIF = std::size_t(dm.nInternalFaces);
    bool upperFromB = false;
    bool lowerFromB = false;
    bool ffcFromB = false;
    int kase = -1;
    {
        ArrayBatch st;
        if (B.hasDiag && !A.hasDiag)
        {
            A.diag.allocate(std::size_t(dm.nCells));
            st.fill(A.diag.data(), A.diag.size(), 0.0);
            A.hasDiag = true;
        }
        // 原本的分支次序（ensureDiag 之後判斷）
        if (A.symmetric() && B.symmetric())
        {
            kase = 0;
        }
        else if (A.symmetric() && B.asymmetric())
        {
            kase = 1;
            // if (upperPtr_) lower(); else upper();
            if (A.hasUpper)
            {
                if (!A.hasLower)
                {
                    A.lower.allocate(nIF);
                    st.copy(A.lower.data(), A.upper.data(), nIF);
                    A.hasLower = true;
                }
            }
            else if (!A.hasUpper)
            {
                A.upper.allocate(nIF);
                if (A.hasLower)
                {
                    st.copy(A.upper.data(), A.lower.data(), nIF);
                }
                else
                {
                    st.fill(A.upper.data(), nIF, 0.0);
                }
                A.hasUpper = true;
            }
        }
        else if (A.asymmetric() && B.symmetric())
        {
            kase = 2;
        }
        else if (A.asymmetric() && B.asymmetric())
        {
            kase = 3;
        }
        else if (A.diagonal())
        {
            kase = 4;
            // +=：upper() = B.upper()；-=：upper() = -B.upper()（下面的一批整個寫入）
            if (B.hasUpper)
            {
                A.upper.allocate(B.upper.size());
                A.hasUpper = true;
                upperFromB = true;
            }
            if (B.hasLower)
            {
                A.lower.allocate(B.lower.size());
                A.hasLower = true;
                lowerFromB = true;
            }
        }
        else if (B.diagonal())
        {
            kase = 5;
        }
        else
        {
            throw std::runtime_error("FvScalarMatrix: unknown matrix type combination");
        }
        if (B.hasFaceFluxCorrection && !A.hasFaceFluxCorrection)
        {
            A.hasFaceFluxCorrection = true;
            A.faceFluxCorrection.in.allocate(B.faceFluxCorrection.in.size());
            A.faceFluxCorrection.bnd.allocate(B.faceFluxCorrection.bnd.size());
            ffcFromB = true;
        }
    }

    ArrayBatch b;
    const auto op = [&](DeviceArray<double>& a, const DeviceArray<double>& x)
    {
        if (minus)
        {
            b.sub(a.data(), x.data(), a.size());
        }
        else
        {
            b.add(a.data(), x.data(), a.size());
        }
    };
    const auto set = [&](DeviceArray<double>& a, const DeviceArray<double>& x)
    {
        if (minus)
        {
            b.copyNeg(a.data(), x.data(), x.size());
        }
        else
        {
            b.copy(a.data(), x.data(), x.size());
        }
    };
    if (B.hasDiag)
    {
        op(A.diag, B.diag);
    }
    switch (kase)
    {
        case 0:
            op(A.upper, B.upper);
            break;
        case 1:
        case 3:
            op(A.upper, B.upper);
            op(A.lower, B.lower);
            break;
        case 2:
        {
            const DeviceArray<double>& bu = B.hasUpper ? B.upper : B.lower;
            op(A.lower, bu);
            op(A.upper, bu);
            break;
        }
        case 4:
            if (upperFromB)
            {
                set(A.upper, B.upper);
            }
            if (lowerFromB)
            {
                set(A.lower, B.lower);
            }
            break;
        default:
            break;
    }
    op(A.source, B.source);
    op(A.internalCoeffs, B.internalCoeffs);
    op(A.boundaryCoeffs, B.boundaryCoeffs);
    if (B.hasFaceFluxCorrection)
    {
        if (ffcFromB)
        {
            set(A.faceFluxCorrection.in.component(0), B.faceFluxCorrection.in.component(0));
            set(A.faceFluxCorrection.bnd.component(0), B.faceFluxCorrection.bnd.component(0));
        }
        else
        {
            op(A.faceFluxCorrection.in.component(0), B.faceFluxCorrection.in.component(0));
            op(A.faceFluxCorrection.bnd.component(0), B.faceFluxCorrection.bnd.component(0));
        }
    }
}

} // namespace


void addMatrix(const DeviceMesh& dm, FvScalarMatrix& A, const FvScalarMatrix& B)
{
    lduCombine(dm, A, B, false);
}


void subtractMatrix(const DeviceMesh& dm, FvScalarMatrix& A, const FvScalarMatrix& B)
{
    lduCombine(dm, A, B, true);
}


void negateMatrix(FvScalarMatrix& A)
{
    ArrayBatch b;
    if (A.hasDiag)
    {
        b.negate(A.diag.data(), A.diag.size());
    }
    if (A.hasUpper)
    {
        b.negate(A.upper.data(), A.upper.size());
    }
    if (A.hasLower)
    {
        b.negate(A.lower.data(), A.lower.size());
    }
    b.negate(A.source.data(), A.source.size());
    b.negate(A.internalCoeffs.data(), A.internalCoeffs.size());
    b.negate(A.boundaryCoeffs.data(), A.boundaryCoeffs.size());
    if (A.hasFaceFluxCorrection)
    {
        b.negate(A.faceFluxCorrection.in.component(0).data(), A.faceFluxCorrection.in.component(0).size());
        b.negate(A.faceFluxCorrection.bnd.component(0).data(), A.faceFluxCorrection.bnd.component(0).size());
    }
}


void matrixMinusSu(const DeviceMesh& dm, FvScalarMatrix& A, const double* su)
{
    const std::size_t n = std::size_t(dm.nCells);
    if (n)
    {
        kAddVTimes<<<nBlocks(n), BS>>>(n, A.source.data(), dm.V.data(), su);
        GNC_CUDA_LAUNCH_CHECK();
    }
}


void suMinusMatrix(const DeviceMesh& dm, const double* su, FvScalarMatrix& A)
{
    negateMatrix(A);
    const std::size_t n = std::size_t(dm.nCells);
    if (n)
    {
        kSubVTimes<<<nBlocks(n), BS>>>(n, A.source.data(), dm.V.data(), su);
        GNC_CUDA_LAUNCH_CHECK();
    }
}


FvScalarMatrix fvmSp(const DeviceMesh& dm, const double* sp)
{
    FvScalarMatrix A = fvmEmpty(dm);
    ensureDiag(dm, A);
    const std::size_t n = std::size_t(dm.nCells);
    if (n)
    {
        kAddVTimes<<<nBlocks(n), BS>>>(n, A.diag.data(), dm.V.data(), sp);
        GNC_CUDA_LAUNCH_CHECK();
    }
    return A;
}


FvScalarMatrix fvmDdtBoundedDualTime
(
    const DeviceMesh& dm,
    const double* rPseudoDeltaT,
    const double* rho,
    const VolField<double>& psi
)
{
    // steadyStateDdtScheme::fvmDdt(rho, vf)：沒有係數的矩陣
    FvScalarMatrix A = fvmEmpty(dm);
    ensureDiag(dm, A);
    const std::int32_t n = dm.nCells;
    if (n)
    {
        kDdtDualTime<<<nBlocks(std::size_t(n)), BS>>>
        (
            n, rPseudoDeltaT, rho, psi.in.cview().p[0], dm.V.data(),
            A.diag.data(), A.source.data()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    // bounded：- fvm::Sp(fvc::ddt(rho), vf)；steadyState 的 fvc::ddt 為 0
    DeviceArray<double> zero(std::size_t(dm.nCells));
    fill(zero, 0.0);
    const FvScalarMatrix S = fvmSp(dm, zero.data());
    subtractMatrix(dm, A, S);
    return A;
}


namespace
{

// gaussConvectionScheme::fvmDiv 與 bounded 的修正；weights 為內部面的權重
// （nullptr 為 upwind）。邊界的 valueInternalCoeffs、valueBoundaryCoeffs 對這裡
// 支援的型別與權重無關
FvScalarMatrix divBoundedGauss
(
    const DeviceMesh& dm,
    const SurfField<double>& phi,
    const VolField<double>& psi,
    const double* weights
)
{
    FvScalarMatrix A = fvmEmpty(dm);
    const std::int32_t nIF = dm.nInternalFaces;
    ensureLower(dm, A);
    ensureUpper(dm, A);
    if (nIF)
    {
        kUpwindFaces<<<nBlocks(std::size_t(nIF)), BS>>>
        (
            nIF, phi.in.cview().p[0], weights, A.lower.data(), A.upper.data()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    ensureDiag(dm, A);
    const Gather g = gatherOf(dm);
    if (dm.nCells)
    {
        kNegSumDiag<<<nBlocks(std::size_t(dm.nCells)), BS>>>(g, A.lower.data(), A.upper.data(), A.diag.data());
        GNC_CUDA_LAUNCH_CHECK();
    }
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const Range r = rangeOf(dm, pi);
        if (r.size == 0)
        {
            continue;
        }
        const PatchIn p = patchIn(dm, psi, &phi, pi, "Gauss upwind valueInternalCoeffs");
        kUpwindPatch<<<nBlocks(std::size_t(r.size)), BS>>>
        (
            p, dm.deltaCoeffs.data(), A.internalCoeffs.data(), A.boundaryCoeffs.data()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }

    // bounded：- fvm::Sp(fvc::surfaceIntegrate(phi), vf)
    DField<double> sInt(std::size_t(dm.nCells));
    surfaceIntegrate(dm, phi, sInt);
    const FvScalarMatrix S = fvmSp(dm, sInt.cview().p[0]);
    subtractMatrix(dm, A, S);
    return A;
}

} // namespace


FvScalarMatrix fvmDivBoundedGaussUpwind
(
    const DeviceMesh& dm,
    const SurfField<double>& phi,
    const VolField<double>& psi
)
{
    return divBoundedGauss(dm, phi, psi, nullptr);
}


FvScalarMatrix fvmDivBoundedGaussLimitedLinear
(
    const DeviceMesh& dm,
    const SurfField<double>& phi,
    const VolField<double>& psi,
    const VolField<vec3>& gradPsi,
    const double k
)
{
    const std::int32_t nIF = dm.nInternalFaces;
    DeviceArray<double> w(static_cast<std::size_t>(nIF));
    if (nIF)
    {
        kLimitedLinearWeights<<<nBlocks(std::size_t(nIF)), BS>>>
        (
            nIF, dm.owner.data(), dm.neighbour.data(), phi.in.cview().p[0], psi.in.cview().p[0],
            gradPsi.in.cview(), dm.C.x.data(), dm.C.y.data(), dm.C.z.data(), dm.weights.data(),
            2.0/foamMax(k, SMALL), w.data()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    return divBoundedGauss(dm, phi, psi, w.data());
}


LaplacianSnGrad parseLaplacianScheme(const std::string& scheme, const std::string& name)
{
    LaplacianSnGrad s;
    if (scheme == "Gauss linear corrected")
    {
        return s;
    }
    std::string coeff;
    if (scheme.rfind("Gauss linear limited corrected ", 0) == 0)
    {
        coeff = scheme.substr(std::string("Gauss linear limited corrected ").size());
    }
    else if (scheme.rfind("Gauss linear limited ", 0) == 0)
    {
        coeff = scheme.substr(std::string("Gauss linear limited ").size());
    }
    else
    {
        throw std::runtime_error
        (
            name + " " + scheme + " is not supported (Gauss linear corrected or limited corrected)"
        );
    }
    s.limited = true;
    if (!parseScalar(coeff, s.limitCoeff) || s.limitCoeff < 0 || s.limitCoeff > 1)
    {
        throw std::runtime_error(name + " " + scheme + ": bad limit coefficient");
    }
    return s;
}


FvScalarMatrix fvmLaplacianGaussLinearCorrected
(
    const DeviceMesh& dm,
    const VolField<double>& gamma,
    const VolField<double>& psi,
    const VolField<vec3>& gradPsi,
    const SurfField<double>* phi,
    const LaplacianSnGrad& snGrad
)
{
    // laplacianScheme::fvmLaplacian(volScalarField gamma, vf)：gamma 以 linear 內插
    SurfField<double> gammaF;
    gammaF.in.allocate(std::size_t(dm.nInternalFaces));
    gammaF.bnd.allocate(std::size_t(dm.nFvBoundaryFaces()));
    interpolateLinear(dm, gamma, gammaF);
    return fvmLaplacianSurfaceGamma(dm, gammaF, psi, gradPsi, phi, false, snGrad);
}


FvScalarMatrix fvmLaplacianSurfaceGamma
(
    const DeviceMesh& dm,
    const SurfField<double>& gammaF,
    const VolField<double>& psi,
    const VolField<vec3>& gradPsi,
    const SurfField<double>* phi,
    const bool fluxRequired,
    const LaplacianSnGrad& snGrad
)
{
    const std::int32_t nIF = dm.nInternalFaces;
    const std::int32_t nB = dm.nFvBoundaryFaces();

    // gammaMagSf = gamma*magSf
    SurfField<double> gammaMagSf;
    gammaMagSf.in.allocate(std::size_t(nIF));
    gammaMagSf.bnd.allocate(std::size_t(nB));

    FvScalarMatrix A = fvmEmpty(dm);
    ensureUpper(dm, A);
    if (nIF)
    {
        kLaplacianFaces<<<nBlocks(std::size_t(nIF)), BS>>>
        (
            nIF, gammaF.in.cview().p[0], dm.magSf.data(), dm.nonOrthDeltaCoeffs.data(),
            gammaMagSf.in.view().p[0], A.upper.data()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    ensureDiag(dm, A);
    const Gather g = gatherOf(dm);
    if (dm.nCells)
    {
        kNegSumDiag<<<nBlocks(std::size_t(dm.nCells)), BS>>>(g, A.upper.data(), A.upper.data(), A.diag.data());
        GNC_CUDA_LAUNCH_CHECK();
    }
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const Range r = rangeOf(dm, pi);
        if (r.size == 0)
        {
            continue;
        }
        const PatchIn p = patchIn(dm, psi, phi, pi, "gradientInternalCoeffs");
        kLaplacianPatch<<<nBlocks(std::size_t(r.size)), BS>>>
        (
            p, gammaF.bnd.cview().p[0], dm.magSf.data(), dm.deltaCoeffs.data(),
            dm.nonOrthDeltaCoeffs.data(), gammaMagSf.bnd.view().p[0],
            A.internalCoeffs.data(), A.boundaryCoeffs.data()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }

    // 非正交修正：source -= V*fvc::div(gammaMagSf*correction(psi))，
    // correction = linear dotInterpolate(nonOrthCorrectionVectors, grad(psi))
    const SurfField<vec3> corrVecs = meshSurfVector(dm, dm.nonOrthCorrectionVectors);
    SurfField<double> corr;
    corr.in.allocate(std::size_t(nIF));
    corr.bnd.allocate(std::size_t(nB));
    dotInterpolateLinear(dm, corrVecs, gradPsi, corr);
    if (snGrad.limited)
    {
        const double lc = snGrad.limitCoeff;
        const double oneMinusLc = 1 - lc;
        if (nIF)
        {
            kLimitCorrection<<<nBlocks(std::size_t(nIF)), BS>>>
            (
                nIF, dm.owner.data(), dm.neighbour.data(), dm.nonOrthDeltaCoeffs.data(),
                psi.in.cview().p[0], nullptr, lc, oneMinusLc, corr.in.view().p[0]
            );
        }
        if (nB)
        {
            DField<double> snB;
            snGradBCs(dm, psi, snB);
            kLimitCorrection<<<nBlocks(std::size_t(nB)), BS>>>
            (
                nB, nullptr, nullptr, nullptr, nullptr, snB.cview().p[0], lc, oneMinusLc,
                corr.bnd.view().p[0]
            );
        }
        GNC_CUDA_LAUNCH_CHECK();
    }
    SurfField<double> flux;
    flux.in.allocate(std::size_t(nIF));
    flux.bnd.allocate(std::size_t(nB));
    if (nIF)
    {
        kMulFaces<<<nBlocks(std::size_t(nIF)), BS>>>
        (
            nIF, gammaMagSf.in.cview().p[0], corr.in.cview().p[0], flux.in.view().p[0]
        );
    }
    if (nB)
    {
        kMulFaces<<<nBlocks(std::size_t(nB)), BS>>>
        (
            nB, gammaMagSf.bnd.cview().p[0], corr.bnd.cview().p[0], flux.bnd.view().p[0]
        );
    }
    GNC_CUDA_LAUNCH_CHECK();
    DField<double> divCorr(std::size_t(dm.nCells));
    surfaceIntegrate(dm, flux, divCorr);
    const std::size_t n = std::size_t(dm.nCells);
    if (n)
    {
        kSubVTimes<<<nBlocks(n), BS>>>(n, A.source.data(), dm.V.data(), divCorr.cview().p[0]);
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (fluxRequired)
    {
        A.hasFaceFluxCorrection = true;
        A.faceFluxCorrection = std::move(flux);
    }
    return A;
}


namespace
{

// fvMatrix::flux() 的內部面：faceH = upper*psi[N] - lower*psi[P]
__global__ void kFaceH
(
    const std::int32_t nIF,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    const double* __restrict__ psi,
    double* __restrict__ out
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f < nIF)
    {
        out[f] = upper[f]*psi[neighbour[f]] - lower[f]*psi[owner[f]];
    }
}

// 邊界：internalCoeffs*psi[owner] - boundaryCoeffs（非耦合 patch 的 NeighbourContrib
// 為 boundaryCoeffs 本身）
__global__ void kFluxBoundary
(
    const std::int32_t nB,
    const std::int32_t* __restrict__ fvbFaces,
    const std::int32_t* __restrict__ owner,
    const double* __restrict__ ic,
    const double* __restrict__ bc,
    const double* __restrict__ psi,
    double* __restrict__ out
)
{
    const std::int32_t b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b < nB)
    {
        out[b] = ic[b]*psi[owner[fvbFaces[b]]] - bc[b];
    }
}

// linearUpwind 的修正乘以面通量：phi*((Cf - C[c]) & grad[c])，c = phi > 0 ? P : N
__global__ void kLinearUpwindCorrFlux
(
    const std::int32_t nIF,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ phi,
    const CView<vec3> Cf,
    const CView<vec3> C,
    const CView<vec3> grad,
    double* __restrict__ out
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIF)
    {
        return;
    }
    const std::int32_t c = (phi[f] > 0) ? owner[f] : neighbour[f];
    const double corr = dot(Cf[f] - C[c], grad[c]);
    out[f] = phi[f]*corr;
}

// 邊界的修正為 0：phi_b*0
__global__ void kTimesZero(const std::int32_t n, const double* __restrict__ phi, double* __restrict__ out)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = phi[i]*0.0;
    }
}

CView<vec3> vecView(const DeviceVec3Array& a)
{
    CView<vec3> v;
    v.p[0] = a.x.data();
    v.p[1] = a.y.data();
    v.p[2] = a.z.data();
    return v;
}

} // namespace


SurfField<double> matrixFlux
(
    const DeviceMesh& dm,
    const FvScalarMatrix& A,
    const VolField<double>& psi
)
{
    const std::int32_t nIF = dm.nInternalFaces;
    const std::int32_t nB = dm.nFvBoundaryFaces();
    if (!A.hasUpper && !A.hasLower)
    {
        throw std::runtime_error("matrixFlux: the matrix has no off-diagonal coefficients");
    }
    const double* lower = A.hasLower ? A.lower.data() : A.upper.data();
    const double* upper = A.hasUpper ? A.upper.data() : A.lower.data();
    SurfField<double> F;
    F.name = "flux(" + psi.name + ")";
    F.in.allocate(std::size_t(nIF));
    F.bnd.allocate(std::size_t(nB));
    if (nIF)
    {
        kFaceH<<<nBlocks(std::size_t(nIF)), BS>>>
        (
            nIF, dm.owner.data(), dm.neighbour.data(), lower, upper, psi.in.cview().p[0],
            F.in.view().p[0]
        );
    }
    if (nB)
    {
        kFluxBoundary<<<nBlocks(std::size_t(nB)), BS>>>
        (
            nB, dm.fvBoundaryFaces.data(), dm.owner.data(), A.internalCoeffs.data(),
            A.boundaryCoeffs.data(), psi.in.cview().p[0], F.bnd.view().p[0]
        );
    }
    GNC_CUDA_LAUNCH_CHECK();
    if (A.hasFaceFluxCorrection)
    {
        if (nIF)
        {
            kAddTo<<<nBlocks(std::size_t(nIF)), BS>>>
            (
                std::size_t(nIF), F.in.view().p[0], A.faceFluxCorrection.in.cview().p[0]
            );
        }
        if (nB)
        {
            kAddTo<<<nBlocks(std::size_t(nB)), BS>>>
            (
                std::size_t(nB), F.bnd.view().p[0], A.faceFluxCorrection.bnd.cview().p[0]
            );
        }
        GNC_CUDA_LAUNCH_CHECK();
    }
    return F;
}


FvScalarMatrix fvmDivBoundedGaussLinearUpwind
(
    const DeviceMesh& dm,
    const SurfField<double>& phi,
    const VolField<double>& psi,
    const VolField<vec3>& gradPsi
)
{
    const std::int32_t nIF = dm.nInternalFaces;
    const std::int32_t nB = dm.nFvBoundaryFaces();
    // gaussConvectionScheme::fvmDiv（upwind 權重），corrected：
    // fvm += fvc::surfaceIntegrate(faceFlux*correction(vf))，再 bounded 的 - Sp
    FvScalarMatrix A = fvmEmpty(dm);
    ensureLower(dm, A);
    ensureUpper(dm, A);
    if (nIF)
    {
        kUpwindFaces<<<nBlocks(std::size_t(nIF)), BS>>>
        (
            nIF, phi.in.cview().p[0], nullptr, A.lower.data(), A.upper.data()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    ensureDiag(dm, A);
    const Gather g = gatherOf(dm);
    if (dm.nCells)
    {
        kNegSumDiag<<<nBlocks(std::size_t(dm.nCells)), BS>>>(g, A.lower.data(), A.upper.data(), A.diag.data());
        GNC_CUDA_LAUNCH_CHECK();
    }
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const Range r = rangeOf(dm, pi);
        if (r.size == 0)
        {
            continue;
        }
        const PatchIn p = patchIn(dm, psi, &phi, pi, "Gauss linearUpwind valueInternalCoeffs");
        kUpwindPatch<<<nBlocks(std::size_t(r.size)), BS>>>
        (
            p, dm.deltaCoeffs.data(), A.internalCoeffs.data(), A.boundaryCoeffs.data()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }

    SurfField<double> cf;
    cf.in.allocate(std::size_t(nIF));
    cf.bnd.allocate(std::size_t(nB));
    if (nIF)
    {
        kLinearUpwindCorrFlux<<<nBlocks(std::size_t(nIF)), BS>>>
        (
            nIF, dm.owner.data(), dm.neighbour.data(), phi.in.cview().p[0], vecView(dm.Cf),
            vecView(dm.C), gradPsi.in.cview(), cf.in.view().p[0]
        );
    }
    if (nB)
    {
        kTimesZero<<<nBlocks(std::size_t(nB)), BS>>>(nB, phi.bnd.cview().p[0], cf.bnd.view().p[0]);
    }
    GNC_CUDA_LAUNCH_CHECK();
    DField<double> sCorr(std::size_t(dm.nCells));
    surfaceIntegrate(dm, cf, sCorr);
    matrixPlusSu(dm, A, sCorr.cview().p[0]);

    // bounded：- fvm::Sp(fvc::surfaceIntegrate(phi), vf)
    DField<double> sInt(std::size_t(dm.nCells));
    surfaceIntegrate(dm, phi, sInt);
    const FvScalarMatrix S = fvmSp(dm, sInt.cview().p[0]);
    subtractMatrix(dm, A, S);
    return A;
}


void relaxMatrix
(
    const DeviceMesh& dm,
    FvScalarMatrix& A,
    const VolField<double>& psi,
    const double alpha
)
{
    if (alpha <= 0)
    {
        return;
    }
    ensureDiag(dm, A);
    if (!A.hasUpper && !A.hasLower)
    {
        throw std::runtime_error("relaxMatrix: matrix without off-diagonal coefficients");
    }
    const double* lower = A.hasLower ? A.lower.data() : A.upper.data();
    const double* upper = A.hasUpper ? A.upper.data() : A.lower.data();
    const Gather g = gatherOf(dm);
    if (dm.nCells)
    {
        kRelax<<<nBlocks(std::size_t(dm.nCells)), BS>>>
        (
            g, lower, upper, A.internalCoeffs.data(), psi.in.cview().p[0], alpha,
            A.diag.data(), A.source.data()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
}


SmoothSolverControls readSmoothSolverControls
(
    const std::string& fvSolutionPath,
    const std::string& fieldName
)
{
    FoamTokenizer t(fvSolutionPath);
    const Dictionary d = Dictionary::parse(t, fvSolutionPath, false);
    const Dictionary& solvers = d.subDict("solvers");

    // solution::solverDict：先找完全相符的鍵，再依倒序比對 pattern
    const Dictionary* sd = nullptr;
    if (const Dictionary::Entry* e = solvers.find(fieldName))
    {
        sd = e->dict.get();
    }
    else
    {
        const auto& es = solvers.entries();
        for (auto it = es.rbegin(); it != es.rend(); ++it)
        {
            if (it->isPattern && it->dict && std::regex_match(fieldName, std::regex(it->key)))
            {
                sd = it->dict.get();
                break;
            }
        }
    }
    if (!sd)
    {
        throw std::runtime_error(fvSolutionPath + ": no solvers entry for " + fieldName);
    }
    const std::string solver = sd->getWord("solver");
    SmoothSolverControls c;
    c.solver = solver;
    if (solver == "smoothSolver")
    {
        const std::string smoother = sd->getWord("smoother");
        if (smoother != "symGaussSeidel")
        {
            throw std::runtime_error(fvSolutionPath + ": smoother " + smoother + " is not supported (symGaussSeidel)");
        }
    }
    else if (solver == "GAMG")
    {
        // GAMG 的鍵：只讀階層的大小；smoother 等其他鍵不用（效能版以多色 Gauss–Seidel
        // 的對稱 V-cycle 當 PCG 的前置條件）
        c.preconditioner = "AMG";
        if (sd->found("nCellsInCoarsestLevel"))
        {
            c.nCellsInCoarsestLevel = int(sd->getLabel("nCellsInCoarsestLevel"));
        }
        if (sd->found("mergeLevels"))
        {
            c.mergeLevels = int(sd->getLabel("mergeLevels"));
        }
    }
    else if (solver == "PBiCGStab" || solver == "PCG")
    {
        c.preconditioner = sd->getWord("preconditioner");
        const std::string want = solver == "PCG" ? "DIC" : "DILU";
        if (c.preconditioner != want)
        {
            throw std::runtime_error
            (
                fvSolutionPath + ": " + solver + " preconditioner " + c.preconditioner
              + " is not supported (" + want + ")"
            );
        }
    }
    else
    {
        throw std::runtime_error
        (
            fvSolutionPath + ": solver " + solver + " for " + fieldName
          + " is not supported (smoothSolver, PBiCGStab, PCG, GAMG with -fast)"
        );
    }
    if (sd->found("tolerance")) c.tolerance = sd->getScalar("tolerance");
    if (sd->found("relTol")) c.relTol = sd->getScalar("relTol");
    if (sd->found("minIter")) c.minIter = int(sd->getLabel("minIter"));
    if (sd->found("maxIter")) c.maxIter = int(sd->getLabel("maxIter"));
    if (sd->found("nSweeps")) c.nSweeps = int(sd->getLabel("nSweeps"));
    for (const char* k : {"norm", "type"})
    {
        if (sd->found(k))
        {
            throw std::runtime_error(fvSolutionPath + ": solvers/" + fieldName + "/" + k + " is not supported");
        }
    }
    if (c.nSweeps < 1)
    {
        throw std::runtime_error(fvSolutionPath + ": nSweeps < 1 is not supported");
    }
    return c;
}


namespace
{

// SolverPerformance::checkConvergence
bool checkConvergence(SolverPerf& p, const SmoothSolverControls& c)
{
    p.converged =
        p.finalResidual < c.tolerance
     || (c.relTol > 1e-20 && p.finalResidual < c.relTol*p.initialResidual);
    return p.converged;
}

std::vector<std::int32_t> ownerStartOf(const DeviceMesh& dm)
{
    const std::vector<std::int32_t> own = dm.owner.download();
    std::vector<std::int32_t> start(std::size_t(dm.nCells) + 1, 0);
    for (std::int32_t f = 0; f < dm.nInternalFaces; ++f)
    {
        if (f > 0 && own[std::size_t(f)] < own[std::size_t(f - 1)])
        {
            throw std::runtime_error("symGaussSeidel: internal faces not in upper-triangular order");
        }
        ++start[std::size_t(own[std::size_t(f)]) + 1];
    }
    for (std::int32_t c = 0; c < dm.nCells; ++c)
    {
        start[std::size_t(c) + 1] += start[std::size_t(c)];
    }
    return start;
}

} // namespace


SolverPerf solveSmoothSymGaussSeidel
(
    const DeviceMesh& dm,
    const LusgsPlan& plan,
    FvScalarMatrix& A,
    VolField<double>& psi,
    const SmoothSolverControls& ctrl,
    const SurfField<double>* phi,
    const bool coeffsUpdated
)
{
    const std::int32_t n = dm.nCells;
    const std::size_t sn = std::size_t(n);
    SolverPerf perf;
    if (n == 0)
    {
        return perf;
    }
    ensureDiag(dm, A);
    if (!A.hasUpper && !A.hasLower)
    {
        throw std::runtime_error("solveSmoothSymGaussSeidel: diagonal matrix is not supported");
    }
    const double* lower = A.hasLower ? A.lower.data() : A.upper.data();
    const double* upper = A.hasUpper ? A.upper.data() : A.lower.data();
    const Gather g = gatherOf(dm);

    // addBoundaryDiag、addBoundarySource（diag 求解後復原，這裡另存一份）
    DeviceArray<double> D(sn), b(sn);
    kAddBoundary<<<nBlocks(sn), BS>>>(g, A.diag.data(), A.internalCoeffs.data(), D.data());
    kAddBoundary<<<nBlocks(sn), BS>>>(g, A.source.data(), A.boundaryCoeffs.data(), b.data());
    GNC_CUDA_LAUNCH_CHECK();

    double* x = psi.in.view().p[0];
    DeviceArray<double> Apsi(sn), tmp(sn), work(sn), bPrime(sn);

    // normFactor（DEFAULT_NORM）
    kAmul<<<nBlocks(sn), BS>>>(g, dm.owner.data(), dm.neighbour.data(), D.data(), lower, upper, x, Apsi.data());
    kSumA<<<nBlocks(sn), BS>>>(g, D.data(), lower, upper, tmp.data());
    GNC_CUDA_LAUNCH_CHECK();
    const double avg = sumSolver(x, sn)/double(n);
    kNormTerms<<<nBlocks(sn), BS>>>(n, Apsi.data(), tmp.data(), avg, b.data(), work.data());
    GNC_CUDA_LAUNCH_CHECK();
    const double normFactor = sumSolver(work.data(), sn) + 1e-20;

    // residual = source - Apsi；初始殘差
    kSourceMinus<<<nBlocks(sn), BS>>>(n, b.data(), Apsi.data(), tmp.data());
    GNC_CUDA_LAUNCH_CHECK();
    perf.initialResidual = solverSumMag(n, tmp.data(), work.data())/normFactor;
    perf.finalResidual = perf.initialResidual;

    if (ctrl.minIter > 0 || !checkConvergence(perf, ctrl))
    {
        const std::vector<std::int32_t> os = ownerStartOf(dm);
        DeviceArray<std::int32_t> ownerStart(os.size());
        ownerStart.upload(os);
        do
        {
            SgsArgs sa;
            sa.lowOff = plan.lowOff.data();
            sa.lowFaces = plan.lowFaces.data();
            sa.ownerStart = ownerStart.data();
            sa.owner = dm.owner.data();
            sa.neighbour = dm.neighbour.data();
            sa.D = D.data();
            sa.lower = lower;
            sa.upper = upper;
            sa.source = b.data();
            sa.bPrime = bPrime.data();
            sa.psi = x;
            for (int sweep = 0; sweep < ctrl.nSweeps; ++sweep)
            {
                if (useColoredSgs())
                {
                    const std::int32_t nCol = std::int32_t(plan.colorOff.size()) - 1;
                    for (std::int32_t k = 0; k < 2*nCol; ++k)
                    {
                        const std::int32_t kc = k < nCol ? k : 2*nCol - 1 - k;
                        const std::int32_t s = plan.colorOff[std::size_t(kc)];
                        const std::int32_t cnt = plan.colorOff[std::size_t(kc) + 1] - s;
                        kSgsColor<<<nBlocks(std::size_t(cnt)), BS>>>
                        (
                            cnt, plan.colorCells.data() + s, plan.cLowOff.data(), plan.cLowFace.data(),
                            plan.cUpOff.data(), plan.cUpFace.data(), sa
                        );
                    }
                    GNC_CUDA_LAUNCH_CHECK();
                    continue;
                }
                const int mode = levelMode(n, std::int32_t(plan.fwdLevelOff.size()) - 1);
                if (mode == 2)
                {
                    kSgsSweepBlock<<<1, levelBlockSize>>>
                    (
                        std::int32_t(plan.fwdLevelOff.size()) - 1, plan.fwdLevelOffD.data(),
                        plan.fwdCells.data(), std::int32_t(plan.revLevelOff.size()) - 1,
                        plan.revLevelOffD.data(), plan.revCells.data(), sa
                    );
                    GNC_CUDA_LAUNCH_CHECK();
                    continue;
                }
                if (mode == 1)
                {
                    std::int32_t nf = std::int32_t(plan.fwdLevelOff.size()) - 1;
                    std::int32_t nr = std::int32_t(plan.revLevelOff.size()) - 1;
                    const std::int32_t* fo = plan.fwdLevelOffD.data();
                    const std::int32_t* fc = plan.fwdCells.data();
                    const std::int32_t* ro = plan.revLevelOffD.data();
                    const std::int32_t* rc = plan.revCells.data();
                    void* args[] = {&nf, &fo, &fc, &nr, &ro, &rc, &sa};
                    static const int g = coopGridSize(kSgsSweepCoop, BS);
                    GNC_CUDA_CHECK(cudaLaunchCooperativeKernel((void*)kSgsSweepCoop, g, BS, args, 0, 0));
                    continue;
                }
                for (std::size_t l = 0; l + 1 < plan.fwdLevelOff.size(); ++l)
                {
                    const std::int32_t s = plan.fwdLevelOff[l];
                    const std::int32_t cnt = plan.fwdLevelOff[l + 1] - s;
                    kSgsForward<<<nBlocks(std::size_t(cnt)), BS>>>(cnt, plan.fwdCells.data() + s, sa);
                }
                for (std::size_t l = 0; l + 1 < plan.revLevelOff.size(); ++l)
                {
                    const std::int32_t s = plan.revLevelOff[l];
                    const std::int32_t cnt = plan.revLevelOff[l + 1] - s;
                    kSgsBackward<<<nBlocks(std::size_t(cnt)), BS>>>(cnt, plan.revCells.data() + s, sa);
                }
                GNC_CUDA_LAUNCH_CHECK();
            }
            kResidual<<<nBlocks(sn), BS>>>
            (
                g, dm.owner.data(), dm.neighbour.data(), D.data(), lower, upper, b.data(), x, tmp.data()
            );
            GNC_CUDA_LAUNCH_CHECK();
            perf.finalResidual = solverSumMag(n, tmp.data(), work.data())/normFactor;
        } while
        (
            (
                (perf.nIterations += ctrl.nSweeps) < ctrl.maxIter
             && !checkConvergence(perf, ctrl)
            )
         || perf.nIterations < ctrl.minIter
        );
    }

    evaluateBCs(dm, psi, phi, coeffsUpdated);
    return perf;
}


bool boundField(const DeviceMesh& dm, VolField<double>& vsf, const double lowerBound)
{
    const std::size_t nC = std::size_t(dm.nCells);
    const std::size_t nB = std::size_t(dm.nFvBoundaryFaces());
    // min(vsf)：內部值與邊界值的最小值（只用來和 lowerBound 比較，見 minValue）
    // 內部與邊界的最小值在一次同步取回（minMaxMany，與分開的 minValue 相同；N-028）
    double minV = 0;
    bool any = false;
    MinMaxReq req[2];
    int nReq = 0;
    if (nC)
    {
        req[nReq++] = MinMaxReq{vsf.in.cview().p[0], nC, false};
    }
    if (nB)
    {
        req[nReq++] = MinMaxReq{vsf.bnd.cview().p[0], nB, false};
    }
    if (nReq)
    {
        double v[2] = {0.0, 0.0};
        minMaxMany(req, nReq, v);
        int q = 0;
        if (nC)
        {
            minV = v[q++];
            any = true;
        }
        if (nB)
        {
            const double mb = v[q];
            minV = any ? foamMin(minV, mb) : mb;
            any = true;
        }
    }
    if (!any || !(minV < lowerBound))
    {
        return false;
    }

    // fvc::average(max(vsf, lb))：linear 內插後 surfaceSum(magSf*ssf)/surfaceSum(magSf)
    VolField<double> mx;
    mx.name = vsf.name;
    mx.in.allocate(nC);
    mx.bnd.allocate(nB);
    kMaxWith<<<nBlocks(nC), BS>>>(std::int32_t(nC), vsf.in.cview().p[0], lowerBound, mx.in.view().p[0]);
    if (nB)
    {
        kMaxWith<<<nBlocks(nB), BS>>>(std::int32_t(nB), vsf.bnd.cview().p[0], lowerBound, mx.bnd.view().p[0]);
    }
    GNC_CUDA_LAUNCH_CHECK();
    SurfField<double> ssf;
    ssf.in.allocate(std::size_t(dm.nInternalFaces));
    ssf.bnd.allocate(nB);
    interpolateLinear(dm, mx, ssf);
    const SurfField<double> magSf = meshSurfScalar(dm, dm.magSf);
    SurfField<double> prod;
    prod.in.allocate(std::size_t(dm.nInternalFaces));
    prod.bnd.allocate(nB);
    multiplySurf(magSf, ssf, prod);
    DeviceArray<double> num(nC), den(nC), avg(nC);
    surfaceSum(dm, prod, num);
    surfaceSum(dm, magSf, den);
    kDivide<<<nBlocks(nC), BS>>>(std::int32_t(nC), num.data(), den.data(), avg.data());
    kBoundInternal<<<nBlocks(nC), BS>>>(std::int32_t(nC), avg.data(), lowerBound, vsf.in.view().p[0]);
    GNC_CUDA_LAUNCH_CHECK();

    // boundaryField = max(boundaryField, lb)：GeometricBoundaryField::operator= 對每個
    // patch 呼叫 operator=(const fvPatchField&)（max 的結果是 calculated patch 場），
    // 依型別分派：fixedValue 族與 mixed 族不改值（fixedValueFvPatchField.H、
    // mixedFvPatchField.H:308）；inletOutlet 族為 vf*refValue + (1 - vf)*max(b, lb)
    // （inletOutletFvPatchField.txx:132）；transform 族（symmetry、symmetryPlane、
    // slip）以新的內部值 evaluate（transformFvPatchField.txx:149，純量為相鄰格值）；
    // 其他（calculated、zeroGradient 等）照抄
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const Range r = rangeOf(dm, pi);
        const Bc t = vsf.bc[pi].type;
        if (r.size == 0 || isFixedValueFamily(t) || (isMixedFamily(t) && !isInletOutletFamily(t)))
        {
            continue;
        }
        double* p = vsf.bnd.view().p[0] + r.start;
        if (isInletOutletFamily(t))
        {
            const PatchBC<double>& bc = vsf.bc[pi];
            kBoundInletOutlet<<<nBlocks(std::size_t(r.size)), BS>>>
            (
                r.size, bc.valueFraction.data(), bc.refValue.cview().p[0], lowerBound, p
            );
        }
        else if (t == Bc::symmetry || t == Bc::symmetryPlane || t == Bc::slip)
        {
            kCopyOwner<<<nBlocks(std::size_t(r.size)), BS>>>
            (
                r.size, dm.fvBoundaryFaces.data() + r.start, dm.owner.data(), vsf.in.cview().p[0], p
            );
        }
        else
        {
            kMaxInPlace<<<nBlocks(std::size_t(r.size)), BS>>>(r.size, p, lowerBound);
        }
        GNC_CUDA_LAUNCH_CHECK();
    }
    return true;
}


// ---------------------------------------------------------------------------
// kOmegaSST（N4b）：A + su、fvm::SuSp、setValues

namespace
{

// fvmSup.C:254–272：diag += V*max(susp, 0)；source -= (V*min(susp, 0))*psi
__global__ void kSuSp
(
    const std::size_t n,
    double* __restrict__ diag,
    double* __restrict__ source,
    const double* __restrict__ V,
    const double* __restrict__ susp,
    const double* __restrict__ psi
)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n)
    {
        diag[i] = diag[i] + V[i]*foamMax(susp[i], 0.0);
        source[i] = source[i] - (V[i]*foamMin(susp[i], 0.0))*psi[i];
    }
}

// 收集表建置時的一筆貢獻（主機端）
struct SetValuesContrib
{
    std::int32_t face, cell;
    std::int8_t ownerSide;
};

// setValuesFromList 的面迴圈：非受限格依原次序收到的貢獻
// （asymmetric 且受限格為 owner 時用 lower，否則用 upper）
__global__ void kSetValuesContrib
(
    const std::int32_t nTargets,
    const std::int32_t* __restrict__ targetCell,
    const std::int32_t* __restrict__ off,
    const std::int32_t* __restrict__ face,
    const std::int32_t* __restrict__ cell,
    const std::int8_t* __restrict__ ownerSide,
    const bool asymmetric,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    const double* __restrict__ psi,
    double* __restrict__ source
)
{
    const std::int32_t t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= nTargets)
    {
        return;
    }
    const std::int32_t T = targetCell[t];
    double s = source[T];
    for (std::int32_t k = off[t]; k < off[t + 1]; ++k)
    {
        const std::int32_t f = face[k];
        const double coeff = (asymmetric && ownerSide[k]) ? lower[f] : upper[f];
        s = s - coeff*psi[cell[k]];
    }
    source[T] = s;
}

__global__ void kZeroAt(const std::int32_t n, const std::int32_t* __restrict__ idx, double* __restrict__ a)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[idx[i]] = 0.0;
    }
}

// source[celli] = value*Diag[celli]
__global__ void kSetValuesSource
(
    const std::int32_t n,
    const std::int32_t* __restrict__ cells,
    const double* __restrict__ psi,
    const double* __restrict__ diag,
    double* __restrict__ source
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const std::int32_t c = cells[i];
        source[c] = psi[c]*diag[c];
    }
}

} // namespace


void matrixPlusSu(const DeviceMesh& dm, FvScalarMatrix& A, const double* su)
{
    const std::size_t n = std::size_t(dm.nCells);
    if (n)
    {
        kSubVTimes<<<nBlocks(n), BS>>>(n, A.source.data(), dm.V.data(), su);
        GNC_CUDA_LAUNCH_CHECK();
    }
}


FvScalarMatrix fvmSuSp(const DeviceMesh& dm, const double* susp, const VolField<double>& psi)
{
    FvScalarMatrix A = fvmEmpty(dm);
    ensureDiag(dm, A);
    const std::size_t n = std::size_t(dm.nCells);
    if (n)
    {
        kSuSp<<<nBlocks(n), BS>>>(n, A.diag.data(), A.source.data(), dm.V.data(), susp, psi.in.cview().p[0]);
        GNC_CUDA_LAUNCH_CHECK();
    }
    return A;
}


SetValuesPlan buildSetValuesPlan
(
    const DeviceMesh& dm,
    const std::vector<std::int32_t>& owner,
    const std::vector<std::int32_t>& neighbour,
    const std::vector<std::int32_t>& cellLabels
)
{
    const std::int32_t nC = dm.nCells;
    const std::int32_t nI = dm.nInternalFaces;
    const std::int32_t nF = std::int32_t(owner.size());

    // 受限格（第一次出現的次序）
    std::vector<char> constrained(std::size_t(nC), 0);
    std::vector<std::int32_t> uniq;
    for (const std::int32_t c : cellLabels)
    {
        if (!constrained[std::size_t(c)])
        {
            constrained[std::size_t(c)] = 1;
            uniq.push_back(c);
        }
    }

    // cells()：每格先 owner 面（遞增），再 neighbour 面（遞增）——primitiveMesh::calcCells
    std::vector<std::int32_t> cOff(std::size_t(nC) + 1, 0);
    for (std::int32_t f = 0; f < nF; ++f) ++cOff[std::size_t(owner[std::size_t(f)]) + 1];
    for (std::int32_t f = 0; f < nI; ++f) ++cOff[std::size_t(neighbour[std::size_t(f)]) + 1];
    for (std::int32_t c = 0; c < nC; ++c) cOff[std::size_t(c) + 1] += cOff[std::size_t(c)];
    std::vector<std::int32_t> cFaces(std::size_t(cOff.back()));
    {
        std::vector<std::int32_t> fill(cOff.begin(), cOff.end() - 1);
        for (std::int32_t f = 0; f < nF; ++f) cFaces[std::size_t(fill[std::size_t(owner[std::size_t(f)])]++)] = f;
        for (std::int32_t f = 0; f < nI; ++f) cFaces[std::size_t(fill[std::size_t(neighbour[std::size_t(f)])]++)] = f;
    }

    // 邊界面的有限體積邊界序號（empty 為 -1）
    const std::vector<std::int32_t> faceToFvB = dm.faceToFvB.download();

    std::vector<std::vector<SetValuesContrib>> byTarget(static_cast<std::size_t>(nC));
    std::vector<std::int32_t> zf, zb;
    for (const std::int32_t c : uniq)
    {
        for (std::int32_t k = cOff[std::size_t(c)]; k < cOff[std::size_t(c) + 1]; ++k)
        {
            const std::int32_t f = cFaces[std::size_t(k)];
            if (f < nI)
            {
                const bool own = owner[std::size_t(f)] == c;
                const std::int32_t other = own ? neighbour[std::size_t(f)] : owner[std::size_t(f)];
                if (!constrained[std::size_t(other)])
                {
                    byTarget[std::size_t(other)].push_back({f, c, std::int8_t(own ? 1 : 0)});
                }
                zf.push_back(f);
            }
            else
            {
                const std::int32_t b = faceToFvB[std::size_t(f - nI)];
                if (b >= 0)
                {
                    zb.push_back(b);
                }
            }
        }
    }

    std::vector<std::int32_t> tCell, tOff(1, 0), cf, cc;
    std::vector<std::int8_t> cs;
    for (std::int32_t T = 0; T < nC; ++T)
    {
        const std::vector<SetValuesContrib>& v = byTarget[std::size_t(T)];
        if (v.empty())
        {
            continue;
        }
        tCell.push_back(T);
        for (const SetValuesContrib& x : v)
        {
            cf.push_back(x.face);
            cc.push_back(x.cell);
            cs.push_back(x.ownerSide);
        }
        tOff.push_back(std::int32_t(cf.size()));
    }

    SetValuesPlan p;
    p.nConstrained = std::int32_t(uniq.size());
    p.cells.upload(uniq);
    p.nTargets = std::int32_t(tCell.size());
    p.targetCell.upload(tCell);
    p.targetOff.upload(tOff);
    p.contribFace.upload(cf);
    p.contribCell.upload(cc);
    p.contribOwnerSide.upload(cs);
    p.nZeroFaces = std::int32_t(zf.size());
    p.zeroFaces.upload(zf);
    p.nZeroBnd = std::int32_t(zb.size());
    p.zeroBnd.upload(zb);
    return p;
}


void setValuesFromPsi
(
    const SetValuesPlan& plan,
    FvScalarMatrix& A,
    const VolField<double>& psi
)
{
    const double* psiIn = psi.in.cview().p[0];
    if (A.symmetric() || A.asymmetric())
    {
        if (plan.nTargets)
        {
            kSetValuesContrib<<<nBlocks(std::size_t(plan.nTargets)), BS>>>
            (
                plan.nTargets, plan.targetCell.data(), plan.targetOff.data(),
                plan.contribFace.data(), plan.contribCell.data(), plan.contribOwnerSide.data(),
                A.asymmetric(), A.hasLower ? A.lower.data() : nullptr, A.upper.data(), psiIn,
                A.source.data()
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
        if (plan.nZeroFaces)
        {
            kZeroAt<<<nBlocks(std::size_t(plan.nZeroFaces)), BS>>>(plan.nZeroFaces, plan.zeroFaces.data(), A.upper.data());
            if (A.hasLower)
            {
                kZeroAt<<<nBlocks(std::size_t(plan.nZeroFaces)), BS>>>(plan.nZeroFaces, plan.zeroFaces.data(), A.lower.data());
            }
            GNC_CUDA_LAUNCH_CHECK();
        }
        if (plan.nZeroBnd)
        {
            kZeroAt<<<nBlocks(std::size_t(plan.nZeroBnd)), BS>>>(plan.nZeroBnd, plan.zeroBnd.data(), A.internalCoeffs.data());
            kZeroAt<<<nBlocks(std::size_t(plan.nZeroBnd)), BS>>>(plan.nZeroBnd, plan.zeroBnd.data(), A.boundaryCoeffs.data());
            GNC_CUDA_LAUNCH_CHECK();
        }
    }
    if (plan.nConstrained)
    {
        kSetValuesSource<<<nBlocks(std::size_t(plan.nConstrained)), BS>>>
        (
            plan.nConstrained, plan.cells.data(), psiIn, A.diag.data(), A.source.data()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
}


// ---------------------------------------------------------------------------
// PCG（對角預條件，不逐位元）：Poisson 牆距離

namespace
{

__global__ void kMulAB(const std::int32_t n, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = a[i]*b[i];
    }
}

// z = r/D
__global__ void kDivBy(const std::int32_t n, const double* __restrict__ r, const double* __restrict__ D, double* __restrict__ z)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        z[i] = r[i]/D[i];
    }
}

// x += alpha*p；r -= alpha*q
__global__ void kPcgUpdate
(
    const std::int32_t n,
    const double alpha,
    const double* __restrict__ p,
    const double* __restrict__ q,
    double* __restrict__ x,
    double* __restrict__ r
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        x[i] += alpha*p[i];
        r[i] -= alpha*q[i];
    }
}

// p = z + beta*p
__global__ void kPcgDirection(const std::int32_t n, const double beta, const double* __restrict__ z, double* __restrict__ p)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        p[i] = z[i] + beta*p[i];
    }
}

} // namespace


SolverPerf solvePCGDiagonal
(
    const DeviceMesh& dm,
    FvScalarMatrix& A,
    VolField<double>& psi,
    const double tolerance,
    const double relTol,
    const int maxIter
)
{
    const std::int32_t n = dm.nCells;
    const std::size_t sn = std::size_t(n);
    SolverPerf perf;
    if (n == 0)
    {
        return perf;
    }
    ensureDiag(dm, A);
    if (!A.hasUpper || A.hasLower)
    {
        throw std::runtime_error("solvePCGDiagonal: the matrix must be symmetric");
    }
    const double* off = A.upper.data();
    const Gather g = gatherOf(dm);
    DeviceArray<double> D(sn), b(sn);
    kAddBoundary<<<nBlocks(sn), BS>>>(g, A.diag.data(), A.internalCoeffs.data(), D.data());
    kAddBoundary<<<nBlocks(sn), BS>>>(g, A.source.data(), A.boundaryCoeffs.data(), b.data());
    GNC_CUDA_LAUNCH_CHECK();

    double* x = psi.in.view().p[0];
    DeviceArray<double> r(sn), z(sn), p(sn), q(sn), work(sn), sumA(sn);

    // normFactor（DEFAULT_NORM）
    kAmul<<<nBlocks(sn), BS>>>(g, dm.owner.data(), dm.neighbour.data(), D.data(), off, off, x, q.data());
    kSumA<<<nBlocks(sn), BS>>>(g, D.data(), off, off, sumA.data());
    GNC_CUDA_LAUNCH_CHECK();
    const double avg = sumTree(x, sn)/double(n);
    kNormTerms<<<nBlocks(sn), BS>>>(n, q.data(), sumA.data(), avg, b.data(), work.data());
    GNC_CUDA_LAUNCH_CHECK();
    const double normFactor = sumTree(work.data(), sn) + 1e-20;

    kSourceMinus<<<nBlocks(sn), BS>>>(n, b.data(), q.data(), r.data());
    kMagOf<<<nBlocks(sn), BS>>>(n, r.data(), work.data());
    GNC_CUDA_LAUNCH_CHECK();
    perf.initialResidual = sumTree(work.data(), sn)/normFactor;
    perf.finalResidual = perf.initialResidual;
    const auto converged = [&]()
    {
        return perf.finalResidual < tolerance
            || (relTol > 0 && perf.finalResidual < relTol*perf.initialResidual);
    };

    double rz = 0;
    while (!converged() && perf.nIterations < maxIter)
    {
        kDivBy<<<nBlocks(sn), BS>>>(n, r.data(), D.data(), z.data());
        kMulAB<<<nBlocks(sn), BS>>>(n, r.data(), z.data(), work.data());
        GNC_CUDA_LAUNCH_CHECK();
        const double rzNew = sumTree(work.data(), sn);
        if (perf.nIterations == 0)
        {
            GNC_CUDA_CHECK(cudaMemcpy(p.data(), z.data(), sn*sizeof(double), cudaMemcpyDeviceToDevice));
        }
        else
        {
            kPcgDirection<<<nBlocks(sn), BS>>>(n, rzNew/rz, z.data(), p.data());
        }
        rz = rzNew;
        kAmul<<<nBlocks(sn), BS>>>(g, dm.owner.data(), dm.neighbour.data(), D.data(), off, off, p.data(), q.data());
        kMulAB<<<nBlocks(sn), BS>>>(n, p.data(), q.data(), work.data());
        GNC_CUDA_LAUNCH_CHECK();
        const double pq = sumTree(work.data(), sn);
        kPcgUpdate<<<nBlocks(sn), BS>>>(n, rz/pq, p.data(), q.data(), x, r.data());
        kMagOf<<<nBlocks(sn), BS>>>(n, r.data(), work.data());
        GNC_CUDA_LAUNCH_CHECK();
        perf.finalResidual = sumTree(work.data(), sn)/normFactor;
        ++perf.nIterations;
    }
    perf.converged = converged();
    evaluateBCs(dm, psi, nullptr);
    return perf;
}


// ---------------------------------------------------------------------------
// N6：PBiCGStab＋DILU、PCG＋DIC（逐位元，見 fvScalarMatrix.cuh）

namespace
{

// DILU／DIC 的分解（calcReciprocalD）一層：rD[c] = D[c] - sum((upper*lower)/rD[owner])，
// 依面號遞增（DIC：upper*upper）；之後另取倒數
__global__ void kIluFactorLevel
(
    const std::int32_t cnt,
    const std::int32_t* __restrict__ cells,
    const std::int32_t* __restrict__ lowOff,
    const std::int32_t* __restrict__ lowFaces,
    const std::int32_t* __restrict__ owner,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    double* __restrict__ rD
)
{
    const std::int32_t t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= cnt)
    {
        return;
    }
    const std::int32_t c = cells[t];
    double r = rD[c];
    for (std::int32_t q = lowOff[c]; q < lowOff[c + 1]; ++q)
    {
        const std::int32_t f = lowFaces[q];
        r = r - upper[f]*lower[f]/rD[owner[f]];
    }
    rD[c] = r;
}

__global__ void kReciprocal(const std::int32_t n, double* __restrict__ a)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[i] = 1.0/a[i];
    }
}

// out = a*b（逐項）
__global__ void kMulArrays(const std::int32_t n, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = a[i]*b[i];
    }
}

// sumSolver(a*b)，同 solverSumMag
double solverSumProd(const std::int32_t n, const double* a, const double* b, double* work)
{
    if (useTreeReduce())
    {
        return sumTreeOp(ReduceOp::Prod, a, b, std::size_t(n));
    }
    kMulArrays<<<nBlocks(std::size_t(n)), BS>>>(n, a, b, work);
    GNC_CUDA_LAUNCH_CHECK();
    return sumSolver(work, std::size_t(n));
}

// 正向一層：wA[c] -= (rD[c]*lower[f])*wA[owner[f]]，依面號遞增（DIC：upper）
__global__ void kIluForwardLevel
(
    const std::int32_t cnt,
    const std::int32_t* __restrict__ cells,
    const std::int32_t* __restrict__ lowOff,
    const std::int32_t* __restrict__ lowFaces,
    const std::int32_t* __restrict__ owner,
    const double* __restrict__ rD,
    const double* __restrict__ lower,
    double* __restrict__ wA
)
{
    const std::int32_t t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= cnt)
    {
        return;
    }
    const std::int32_t c = cells[t];
    double w = wA[c];
    for (std::int32_t q = lowOff[c]; q < lowOff[c + 1]; ++q)
    {
        const std::int32_t f = lowFaces[q];
        w = w - rD[c]*lower[f]*wA[owner[f]];
    }
    wA[c] = w;
}

// 反向一層：wA[c] -= (rD[c]*upper[f])*wA[neighbour[f]]，依面號遞減
__global__ void kIluBackwardLevel
(
    const std::int32_t cnt,
    const std::int32_t* __restrict__ cells,
    const std::int32_t* __restrict__ upOff,
    const std::int32_t* __restrict__ upFacesDesc,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ rD,
    const double* __restrict__ upper,
    double* __restrict__ wA
)
{
    const std::int32_t t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= cnt)
    {
        return;
    }
    const std::int32_t c = cells[t];
    double w = wA[c];
    for (std::int32_t q = upOff[c]; q < upOff[c + 1]; ++q)
    {
        const std::int32_t f = upFacesDesc[q];
        w = w - rD[c]*upper[f]*wA[neighbour[f]];
    }
    wA[c] = w;
}

// 單一 block 的層級掃描（core/coopLevels.cuh 的模式 2）：每格的運算與上面三個
// kernel 相同，層與層之間 __syncthreads()，結果逐位元相同
__global__ void kIluFactorBlock
(
    const std::int32_t nLevels,
    const std::int32_t* __restrict__ levelOff,
    const std::int32_t* __restrict__ cells,
    const std::int32_t* __restrict__ lowOff,
    const std::int32_t* __restrict__ lowFaces,
    const std::int32_t* __restrict__ owner,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    double* __restrict__ rD
)
{
    for (std::int32_t l = 0; l < nLevels; ++l)
    {
        const std::int32_t e = levelOff[l + 1];
        for (std::int32_t t = levelOff[l] + std::int32_t(threadIdx.x); t < e; t += blockDim.x)
        {
            const std::int32_t c = cells[t];
            double r = rD[c];
            for (std::int32_t q = lowOff[c]; q < lowOff[c + 1]; ++q)
            {
                const std::int32_t f = lowFaces[q];
                r = r - upper[f]*lower[f]/rD[owner[f]];
            }
            rD[c] = r;
        }
        __syncthreads();
    }
}

__global__ void kIluSweepBlock
(
    const std::int32_t nFwd,
    const std::int32_t* __restrict__ fwdOff,
    const std::int32_t* __restrict__ fwdCells,
    const std::int32_t* __restrict__ lowOff,
    const std::int32_t* __restrict__ lowFaces,
    const std::int32_t* __restrict__ owner,
    const std::int32_t nRev,
    const std::int32_t* __restrict__ revOff,
    const std::int32_t* __restrict__ revCells,
    const std::int32_t* __restrict__ upOff,
    const std::int32_t* __restrict__ upFacesDesc,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ rD,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    double* __restrict__ wA
)
{
    for (std::int32_t l = 0; l < nFwd; ++l)
    {
        const std::int32_t e = fwdOff[l + 1];
        for (std::int32_t t = fwdOff[l] + std::int32_t(threadIdx.x); t < e; t += blockDim.x)
        {
            const std::int32_t c = fwdCells[t];
            double w = wA[c];
            for (std::int32_t q = lowOff[c]; q < lowOff[c + 1]; ++q)
            {
                const std::int32_t f = lowFaces[q];
                w = w - rD[c]*lower[f]*wA[owner[f]];
            }
            wA[c] = w;
        }
        __syncthreads();
    }
    for (std::int32_t l = 0; l < nRev; ++l)
    {
        const std::int32_t e = revOff[l + 1];
        for (std::int32_t t = revOff[l] + std::int32_t(threadIdx.x); t < e; t += blockDim.x)
        {
            const std::int32_t c = revCells[t];
            double w = wA[c];
            for (std::int32_t q = upOff[c]; q < upOff[c + 1]; ++q)
            {
                const std::int32_t f = upFacesDesc[q];
                w = w - rD[c]*upper[f]*wA[neighbour[f]];
            }
            wA[c] = w;
        }
        __syncthreads();
    }
}

// 多色 DILU／DIC（效能版，GNC_ILU=color；不與 OpenFOAM 逐位元）：以 LusgsPlan 的
// 著色次序取代格編號次序。每格的「下」鄰格為色較小者（cLowFace），「上」為色較大者
// （cUpFace）；編碼 k < 0 時 f = -(k + 1)、本格是 owner、係數取 upper[f]，否則本格是
// neighbour、係數取 lower[f]。同色的格互不相鄰，一色一次啟動
__device__ __forceinline__ void colorCoef
(
    const std::int32_t k,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    std::int32_t& f,
    std::int32_t& other,
    double& a
)
{
    if (k < 0)
    {
        f = -(k + 1);
        other = neighbour[f];
        a = upper[f];
    }
    else
    {
        f = k;
        other = owner[f];
        a = lower[f];
    }
}

__global__ void kIluFactorColor
(
    const std::int32_t cnt,
    const std::int32_t* __restrict__ cells,
    const std::int32_t* __restrict__ lowOff,
    const std::int32_t* __restrict__ lowFace,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    double* __restrict__ rD
)
{
    const std::int32_t t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= cnt)
    {
        return;
    }
    const std::int32_t c = cells[t];
    double r = rD[c];
    for (std::int32_t q = lowOff[c]; q < lowOff[c + 1]; ++q)
    {
        std::int32_t f, o;
        double a;
        colorCoef(lowFace[q], owner, neighbour, lower, upper, f, o, a);
        r = r - upper[f]*lower[f]/rD[o];
    }
    rD[c] = r;
}

__global__ void kIluSweepColor
(
    const std::int32_t cnt,
    const std::int32_t* __restrict__ cells,
    const std::int32_t* __restrict__ off,
    const std::int32_t* __restrict__ code,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ rD,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    double* __restrict__ wA
)
{
    const std::int32_t t = blockIdx.x*blockDim.x + threadIdx.x;
    if (t >= cnt)
    {
        return;
    }
    const std::int32_t c = cells[t];
    double w = wA[c];
    for (std::int32_t q = off[c]; q < off[c + 1]; ++q)
    {
        std::int32_t f, o;
        double a;
        colorCoef(code[q], owner, neighbour, lower, upper, f, o, a);
        w = w - rD[c]*a*wA[o];
    }
    wA[c] = w;
}

// PBiCGStab：pA = rA + beta*(pA - omega*AyA)
__global__ void kPbcgDirection
(
    const std::int32_t n, const double beta, const double omega,
    const double* __restrict__ rA, const double* __restrict__ AyA, double* __restrict__ pA
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        pA[i] = rA[i] + beta*(pA[i] - omega*AyA[i]);
    }
}

// out = a - s*b
__global__ void kAxmy(const std::int32_t n, const double s, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = a[i] - s*b[i];
    }
}

// x += s*y
__global__ void kAddScaledTo(const std::int32_t n, const double s, const double* __restrict__ y, double* __restrict__ x)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        x[i] += s*y[i];
    }
}

// PBiCGStab：psi += alpha*yA + omega*zA；rA = sA - omega*tA
__global__ void kPbcgUpdate
(
    const std::int32_t n, const double alpha, const double omega,
    const double* __restrict__ yA, const double* __restrict__ zA,
    const double* __restrict__ sA, const double* __restrict__ tA,
    double* __restrict__ psi, double* __restrict__ rA
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        psi[i] += alpha*yA[i] + omega*zA[i];
        rA[i] = sA[i] - omega*tA[i];
    }
}

// PCG：pA = wA + beta*pA
__global__ void kPcgDirectionF(const std::int32_t n, const double beta, const double* __restrict__ wA, double* __restrict__ pA)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        pA[i] = wA[i] + beta*pA[i];
    }
}

// PCG：psi += alpha*pA；rA -= alpha*wA
__global__ void kPcgUpdateF
(
    const std::int32_t n, const double alpha, const double* __restrict__ pA,
    const double* __restrict__ wA, double* __restrict__ psi, double* __restrict__ rA
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        psi[i] += alpha*pA[i];
        rA[i] -= alpha*wA[i];
    }
}

// 效能版的裝置端純量（N-028）：下列 kernel 由裝置記憶體讀歸約的結果，在每個 thread 算出與主機端相同的
// α、β、ω（同樣的 IEEE 除法），運算式與上面的版本相同

// PCG：beta = wArA/wArAold；pA = wA + beta*pA
__global__ void kPcgDirectionDev
(
    const std::int32_t n, const double* __restrict__ wArA, const double* __restrict__ wArAold,
    const double* __restrict__ wA, double* __restrict__ pA
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const double beta = wArA[0]/wArAold[0];
        pA[i] = wA[i] + beta*pA[i];
    }
}

// PCG：singular(|wApA|/normFactor) 時不更新並設旗標；否則 alpha = wArA/wApA、psi += alpha*pA、rA -= alpha*wA
__global__ void kPcgUpdateDev
(
    const std::int32_t n, const double* __restrict__ wArA, const double* __restrict__ wApA,
    const double normFactor, const double* __restrict__ pA, const double* __restrict__ wA,
    double* __restrict__ psi, double* __restrict__ rA, double* __restrict__ flag
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    const double d = wApA[0];
    if (::fabs(d)/normFactor < VSMALL)
    {
        if (i == 0)
        {
            *flag = 1.0;
        }
        return;
    }
    if (i < n)
    {
        const double alpha = wArA[0]/d;
        psi[i] += alpha*pA[i];
        rA[i] -= alpha*wA[i];
    }
}

// PBiCGStab：alpha = rA0rA/rA0AyA；sA = rA - alpha*AyA
__global__ void kAxmyDev
(
    const std::int32_t n, const double rA0rA, const double* __restrict__ rA0AyA,
    const double* __restrict__ rA, const double* __restrict__ AyA, double* __restrict__ sA
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const double alpha = rA0rA/rA0AyA[0];
        sA[i] = rA[i] - alpha*AyA[i];
    }
}

// PBiCGStab 收斂於 sA：psi += alpha*yA
__global__ void kAddScaledToDev
(
    const std::int32_t n, const double rA0rA, const double* __restrict__ rA0AyA,
    const double* __restrict__ yA, double* __restrict__ psi
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const double alpha = rA0rA/rA0AyA[0];
        psi[i] += alpha*yA[i];
    }
}

// PBiCGStab：omega = tAsA/tAtA；psi += alpha*yA + omega*zA；rA = sA - omega*tA；alpha、omega 寫到 out
// （mapped，主機在下一次同步後讀）
__global__ void kPbcgUpdateDev
(
    const std::int32_t n, const double rA0rA, const double* __restrict__ rA0AyA,
    const double* __restrict__ tt, const double* __restrict__ yA, const double* __restrict__ zA,
    const double* __restrict__ sA, const double* __restrict__ tA, double* __restrict__ psi,
    double* __restrict__ rA, double* __restrict__ out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    const double alpha = rA0rA/rA0AyA[0];
    const double omega = tt[1]/tt[0];
    if (i == 0)
    {
        out[0] = alpha;
        out[1] = omega;
    }
    if (i < n)
    {
        psi[i] += alpha*yA[i] + omega*zA[i];
        rA[i] = sA[i] - omega*tA[i];
    }
}

__global__ void kSqrOf(const std::int32_t n, const double* __restrict__ a, double* __restrict__ out)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = a[i]*a[i];
    }
}

// ILU 型預條件（DILU 或 DIC）
struct IluPrecond
{
    const DeviceMesh& dm;
    const LusgsPlan& plan;
    const double* lower;    // DIC：upper
    const double* upper;
    DeviceArray<double> rD;

    bool color;     // 多色次序（GNC_ILU=color）
    bool block;     // 單一 block 的層級掃描（levelMode 為 2）

    IluPrecond(const DeviceMesh& m, const LusgsPlan& p, const double* D, const double* lo, const double* up)
    :
        dm(m), plan(p), lower(lo), upper(up), rD(std::size_t(m.nCells)),
        color(envIs("GNC_ILU", "color")),
        block(!color && levelMode(m.nCells, std::int32_t(p.fwdLevelOff.size()) - 1) == 2)
    {
        const std::int32_t n = dm.nCells;
        GNC_CUDA_CHECK(cudaMemcpy(rD.data(), D, std::size_t(n)*sizeof(double), cudaMemcpyDeviceToDevice));
        if (color)
        {
            for (std::size_t k = 0; k + 1 < plan.colorOff.size(); ++k)
            {
                const std::int32_t s = plan.colorOff[k];
                const std::int32_t cnt = plan.colorOff[k + 1] - s;
                kIluFactorColor<<<nBlocks(std::size_t(cnt)), BS>>>
                (
                    cnt, plan.colorCells.data() + s, plan.cLowOff.data(), plan.cLowFace.data(),
                    dm.owner.data(), dm.neighbour.data(), lower, upper, rD.data()
                );
            }
            kReciprocal<<<nBlocks(std::size_t(n)), BS>>>(n, rD.data());
            GNC_CUDA_LAUNCH_CHECK();
            return;
        }
        if (block && n)
        {
            kIluFactorBlock<<<1, levelBlockSize>>>
            (
                std::int32_t(plan.fwdLevelOff.size()) - 1, plan.fwdLevelOffD.data(),
                plan.fwdCells.data(), plan.lowOff.data(), plan.lowFaces.data(),
                dm.owner.data(), lower, upper, rD.data()
            );
        }
        for (std::size_t l = 0; !block && l + 1 < plan.fwdLevelOff.size(); ++l)
        {
            const std::int32_t s = plan.fwdLevelOff[l];
            const std::int32_t cnt = plan.fwdLevelOff[l + 1] - s;
            kIluFactorLevel<<<nBlocks(std::size_t(cnt)), BS>>>
            (
                cnt, plan.fwdCells.data() + s, plan.lowOff.data(), plan.lowFaces.data(),
                dm.owner.data(), lower, upper, rD.data()
            );
        }
        kReciprocal<<<nBlocks(std::size_t(n)), BS>>>(n, rD.data());
        GNC_CUDA_LAUNCH_CHECK();
    }

    void precondition(double* wA, const double* rA) const
    {
        const std::int32_t n = dm.nCells;
        kMulArrays<<<nBlocks(std::size_t(n)), BS>>>(n, rD.data(), rA, wA);
        if (color)
        {
            const std::size_t nColors = plan.colorOff.size() - 1;
            for (std::size_t k = 0; k < nColors; ++k)
            {
                const std::int32_t s = plan.colorOff[k];
                const std::int32_t cnt = plan.colorOff[k + 1] - s;
                kIluSweepColor<<<nBlocks(std::size_t(cnt)), BS>>>
                (
                    cnt, plan.colorCells.data() + s, plan.cLowOff.data(), plan.cLowFace.data(),
                    dm.owner.data(), dm.neighbour.data(), rD.data(), lower, upper, wA
                );
            }
            for (std::size_t k = nColors; k-- > 0;)
            {
                const std::int32_t s = plan.colorOff[k];
                const std::int32_t cnt = plan.colorOff[k + 1] - s;
                kIluSweepColor<<<nBlocks(std::size_t(cnt)), BS>>>
                (
                    cnt, plan.colorCells.data() + s, plan.cUpOff.data(), plan.cUpFace.data(),
                    dm.owner.data(), dm.neighbour.data(), rD.data(), lower, upper, wA
                );
            }
            GNC_CUDA_LAUNCH_CHECK();
            return;
        }
        if (block)
        {
            if (n)
            {
                kIluSweepBlock<<<1, levelBlockSize>>>
                (
                    std::int32_t(plan.fwdLevelOff.size()) - 1, plan.fwdLevelOffD.data(),
                    plan.fwdCells.data(), plan.lowOff.data(), plan.lowFaces.data(), dm.owner.data(),
                    std::int32_t(plan.revLevelOff.size()) - 1, plan.revLevelOffD.data(),
                    plan.revCells.data(), plan.upOff.data(), plan.upFacesDesc.data(),
                    dm.neighbour.data(), rD.data(), lower, upper, wA
                );
            }
            GNC_CUDA_LAUNCH_CHECK();
            return;
        }
        for (std::size_t l = 0; l + 1 < plan.fwdLevelOff.size(); ++l)
        {
            const std::int32_t s = plan.fwdLevelOff[l];
            const std::int32_t cnt = plan.fwdLevelOff[l + 1] - s;
            kIluForwardLevel<<<nBlocks(std::size_t(cnt)), BS>>>
            (
                cnt, plan.fwdCells.data() + s, plan.lowOff.data(), plan.lowFaces.data(),
                dm.owner.data(), rD.data(), lower, wA
            );
        }
        for (std::size_t l = 0; l + 1 < plan.revLevelOff.size(); ++l)
        {
            const std::int32_t s = plan.revLevelOff[l];
            const std::int32_t cnt = plan.revLevelOff[l + 1] - s;
            kIluBackwardLevel<<<nBlocks(std::size_t(cnt)), BS>>>
            (
                cnt, plan.revCells.data() + s, plan.upOff.data(), plan.upFacesDesc.data(),
                dm.neighbour.data(), rD.data(), upper, wA
            );
        }
        GNC_CUDA_LAUNCH_CHECK();
    }
};

// SolverPerformance::checkSingularity（vsmall_ = VSMALL）
bool singular(const double r)
{
    return r < VSMALL;
}

} // namespace


SolverPerf solveLduKrylov
(
    const DeviceMesh& dm,
    const LusgsPlan& plan,
    const double* D,
    const double* lower,
    const double* upper,
    const double* b,
    double* x,
    const SmoothSolverControls& ctrl
)
{
    const std::int32_t n = dm.nCells;
    const std::size_t sn = std::size_t(n);
    SolverPerf perf;
    if (n == 0)
    {
        return perf;
    }
    // GAMG：效能版以 PCG（對稱矩陣）或 PBiCGStab 加 AMG 前置條件代替
    const bool gamg = ctrl.solver == "GAMG";
    if (gamg && !envIs("GNC_GAMG", "amg"))
    {
        throw std::runtime_error
        (
            "solver GAMG is supported only in the performance mode (-fast: PCG with an AMG "
            "preconditioner, not bitwise); use PCG/DIC for the reference mode"
        );
    }
    const bool useAmg =
        ctrl.preconditioner == "AMG"
     || (ctrl.solver == "PCG" && envIs("GNC_PCG_PRECOND", "amg"));
    const bool pcg = ctrl.solver == "PCG" || (gamg && lower == upper);
    const Gather g = gatherOf(dm);
    DeviceArray<double> yA(sn), rA(sn), pA(sn), work(sn), sumA(sn);

    // Amul、rA = source - yA、normFactor（lduMatrix::solver::normFactor）
    kAmul<<<nBlocks(sn), BS>>>(g, dm.owner.data(), dm.neighbour.data(), D, lower, upper, x, yA.data());
    kSourceMinus<<<nBlocks(sn), BS>>>(n, b, yA.data(), rA.data());
    kSumA<<<nBlocks(sn), BS>>>(g, D, lower, upper, sumA.data());
    GNC_CUDA_LAUNCH_CHECK();
    const auto sumMag = [&](const double* a)
    {
        return solverSumMag(n, a, work.data());
    };
    const auto sumProd = [&](const double* a, const double* c)
    {
        return solverSumProd(n, a, c, work.data());
    };
    // 效能版（樹狀歸約）合併同步（N-028，結果與分開計算逐位元相同）：sum(psi) 留在裝置端，normFactor、
    // 初始殘差與 PBiCGStab 第一個迭代的 rA0·rA（rA0 = rA）在同一次歸約
    const bool fused = useTreeReduce();
    double normFactor = 0.0;
    double rA0rANext = 0.0;
    bool haveRA0rA = false;
    if (fused)
    {
        double* dsum = deviceScalar();
        sumTreeToDevice(ReduceOp::Sum, x, nullptr, sn, dsum);
        kNormTermsDev<<<nBlocks(sn), BS>>>(n, yA.data(), sumA.data(), dsum, b, work.data());
        GNC_CUDA_LAUNCH_CHECK();
        const ReduceTerm t[3] =
        {
            {ReduceOp::Sum, work.data(), nullptr},
            {ReduceOp::Mag, rA.data(), nullptr},
            {ReduceOp::Prod, rA.data(), rA.data()}
        };
        double r[3] = {0.0, 0.0, 0.0};
        sumTreeOps(t, pcg ? 2 : 3, sn, r);
        normFactor = r[0] + 1e-20;
        perf.initialResidual = r[1]/normFactor;
        rA0rANext = r[2];
        haveRA0rA = !pcg;
    }
    else
    {
        const double avg = sumSolver(x, sn)/double(n);
        kNormTerms<<<nBlocks(sn), BS>>>(n, yA.data(), sumA.data(), avg, b, work.data());
        GNC_CUDA_LAUNCH_CHECK();
        normFactor = sumSolver(work.data(), sn) + 1e-20;
        perf.initialResidual = sumMag(rA.data())/normFactor;
    }
    perf.finalResidual = perf.initialResidual;
    if (!(ctrl.minIter > 0 || !checkConvergence(perf, ctrl)))
    {
        return perf;
    }

    std::unique_ptr<IluPrecond> ilu;
    std::unique_ptr<AmgPrecond> amg;
    if (useAmg)
    {
        // GPU 的階層預設每 2 個配對層合成一層（N-026；fvSolution 的 mergeLevels 不用），
        // 平滑子、前後掃與過度修正見 AmgControls（N-028）；GNC_AMG_* 可改
        AmgControls ac;
        ac.nCellsInCoarsestLevel = ctrl.nCellsInCoarsestLevel;
        ac.mergeLevels = 2;
        amg.reset(new AmgPrecond(dm, amgHierarchy(dm, amgControlsFromEnv(ac)), D, lower, upper));
    }
    else
    {
        ilu.reset(new IluPrecond(dm, plan, D, pcg ? upper : lower, upper));
    }
    struct
    {
        const IluPrecond* ilu;
        const AmgPrecond* amg;
        void precondition(double* w, const double* r) const
        {
            if (amg)
            {
                amg->apply(w, r);
            }
            else
            {
                ilu->precondition(w, r);
            }
        }
    } const precon{ilu.get(), amg.get()};
    if (pcg && fused)
    {
        // 效能版：wArA、wApA 留在裝置端（ping-pong），每個迭代只在 |rA| 同步一次（N-028）。奇異時
        // 更新 kernel 不改 psi、rA，主機讀到旗標就結束（finalResidual 維持上一個值），與下面的版本相同
        DeviceArray<double>& wA = yA;
        double* ds = deviceScalar();
        MappedScalars& ms = mappedScalars();
        volatile double* flag = ms.h + 4;
        *flag = 0.0;
        int cur = 0;
        do
        {
            precon.precondition(wA.data(), rA.data());
            sumTreeToDevice(ReduceOp::Prod, wA.data(), rA.data(), sn, ds + cur);
            if (perf.nIterations == 0)
            {
                ArrayBatch cb;
                cb.copy(pA.data(), wA.data(), sn);
            }
            else
            {
                kPcgDirectionDev<<<nBlocks(sn), BS>>>(n, ds + cur, ds + (cur ^ 1), wA.data(), pA.data());
            }
            kAmul<<<nBlocks(sn), BS>>>(g, dm.owner.data(), dm.neighbour.data(), D, lower, upper, pA.data(), wA.data());
            GNC_CUDA_LAUNCH_CHECK();
            sumTreeToDevice(ReduceOp::Prod, wA.data(), pA.data(), sn, ds + 2);
            kPcgUpdateDev<<<nBlocks(sn), BS>>>(n, ds + cur, ds + 2, normFactor, pA.data(), wA.data(), x, rA.data(), ms.d + 4);
            GNC_CUDA_LAUNCH_CHECK();
            const double mag = sumTreeOp(ReduceOp::Mag, rA.data(), nullptr, sn);
            if (*flag != 0.0)
            {
                *flag = 0.0;
                break;
            }
            perf.finalResidual = mag/normFactor;
            cur ^= 1;
        } while
        (
            (++perf.nIterations < ctrl.maxIter && !checkConvergence(perf, ctrl))
         || perf.nIterations < ctrl.minIter
        );
        return perf;
    }
    if (pcg)
    {
        DeviceArray<double>& wA = yA;
        double wArA = 1e20;       // solverPerformance::great_
        double wArAold = wArA;
        do
        {
            wArAold = wArA;
            precon.precondition(wA.data(), rA.data());
            wArA = sumProd(wA.data(), rA.data());
            if (perf.nIterations == 0)
            {
                GNC_CUDA_CHECK(cudaMemcpy(pA.data(), wA.data(), sn*sizeof(double), cudaMemcpyDeviceToDevice));
            }
            else
            {
                const double beta = wArA/wArAold;
                kPcgDirectionF<<<nBlocks(sn), BS>>>(n, beta, wA.data(), pA.data());
            }
            kAmul<<<nBlocks(sn), BS>>>(g, dm.owner.data(), dm.neighbour.data(), D, lower, upper, pA.data(), wA.data());
            GNC_CUDA_LAUNCH_CHECK();
            const double wApA = sumProd(wA.data(), pA.data());
            if (singular(::fabs(wApA)/normFactor))
            {
                break;
            }
            const double alpha = wArA/wApA;
            kPcgUpdateF<<<nBlocks(sn), BS>>>(n, alpha, pA.data(), wA.data(), x, rA.data());
            GNC_CUDA_LAUNCH_CHECK();
            perf.finalResidual = sumMag(rA.data())/normFactor;
        } while
        (
            (++perf.nIterations < ctrl.maxIter && !checkConvergence(perf, ctrl))
         || perf.nIterations < ctrl.minIter
        );
        return perf;
    }

    // PBiCGStab
    DeviceArray<double> AyA(sn), sA(sn), zA(sn), tA(sn), rA0(sn);
    {
        ArrayBatch cb;
        cb.copy(rA0.data(), rA.data(), sn);
    }
    double rA0rA = 0, alpha = 0, omega = 0;
    if (fused)
    {
        // 效能版：rA0·AyA、tA·tA、tA·sA 留在裝置端，每個迭代只在 |sA| 與 |rA|（連同下一個 rA0·rA）同步；
        // alpha、omega 由更新 kernel 寫到 mapped 記憶體，主機在 |rA| 的同步之後讀（N-028）
        double* ds = deviceScalar();
        MappedScalars& ms = mappedScalars();
        do
        {
            const double rA0rAold = rA0rA;
            if (haveRA0rA)
            {
                rA0rA = rA0rANext;
                haveRA0rA = false;
            }
            else
            {
                rA0rA = sumProd(rA0.data(), rA.data());
            }
            if (singular(::fabs(rA0rA)))
            {
                break;
            }
            if (perf.nIterations == 0)
            {
                ArrayBatch cb;
                cb.copy(pA.data(), rA.data(), sn);
            }
            else
            {
                if (singular(::fabs(omega)))
                {
                    break;
                }
                const double beta = (rA0rA/rA0rAold)*(alpha/omega);
                kPbcgDirection<<<nBlocks(sn), BS>>>(n, beta, omega, rA.data(), AyA.data(), pA.data());
            }
            precon.precondition(yA.data(), pA.data());
            kAmul<<<nBlocks(sn), BS>>>(g, dm.owner.data(), dm.neighbour.data(), D, lower, upper, yA.data(), AyA.data());
            GNC_CUDA_LAUNCH_CHECK();
            sumTreeToDevice(ReduceOp::Prod, rA0.data(), AyA.data(), sn, ds);
            kAxmyDev<<<nBlocks(sn), BS>>>(n, rA0rA, ds, rA.data(), AyA.data(), sA.data());
            GNC_CUDA_LAUNCH_CHECK();
            perf.finalResidual = sumMag(sA.data())/normFactor;
            if (perf.nIterations >= ctrl.minIter && checkConvergence(perf, ctrl))
            {
                kAddScaledToDev<<<nBlocks(sn), BS>>>(n, rA0rA, ds, yA.data(), x);
                GNC_CUDA_LAUNCH_CHECK();
                ++perf.nIterations;
                return perf;
            }
            precon.precondition(zA.data(), sA.data());
            kAmul<<<nBlocks(sn), BS>>>(g, dm.owner.data(), dm.neighbour.data(), D, lower, upper, zA.data(), tA.data());
            GNC_CUDA_LAUNCH_CHECK();
            const ReduceTerm t2[2] = {{ReduceOp::Prod, tA.data(), tA.data()}, {ReduceOp::Prod, tA.data(), sA.data()}};
            sumTreeOpsToDevice(t2, 2, sn, ds + 1);
            kPbcgUpdateDev<<<nBlocks(sn), BS>>>
            (
                n, rA0rA, ds, ds + 1, yA.data(), zA.data(), sA.data(), tA.data(), x, rA.data(), ms.d
            );
            GNC_CUDA_LAUNCH_CHECK();
            const ReduceTerm t3[2] = {{ReduceOp::Mag, rA.data(), nullptr}, {ReduceOp::Prod, rA0.data(), rA.data()}};
            double r3[2] = {0.0, 0.0};
            sumTreeOps(t3, 2, sn, r3);
            alpha = static_cast<volatile double*>(ms.h)[0];
            omega = static_cast<volatile double*>(ms.h)[1];
            perf.finalResidual = r3[0]/normFactor;
            rA0rANext = r3[1];
            haveRA0rA = true;
        } while
        (
            (++perf.nIterations < ctrl.maxIter && !checkConvergence(perf, ctrl))
         || perf.nIterations < ctrl.minIter
        );
        return perf;
    }
    do
    {
        const double rA0rAold = rA0rA;
        rA0rA = sumProd(rA0.data(), rA.data());
        if (singular(::fabs(rA0rA)))
        {
            break;
        }
        if (perf.nIterations == 0)
        {
            GNC_CUDA_CHECK(cudaMemcpy(pA.data(), rA.data(), sn*sizeof(double), cudaMemcpyDeviceToDevice));
        }
        else
        {
            if (singular(::fabs(omega)))
            {
                break;
            }
            const double beta = (rA0rA/rA0rAold)*(alpha/omega);
            kPbcgDirection<<<nBlocks(sn), BS>>>(n, beta, omega, rA.data(), AyA.data(), pA.data());
        }
        precon.precondition(yA.data(), pA.data());
        kAmul<<<nBlocks(sn), BS>>>(g, dm.owner.data(), dm.neighbour.data(), D, lower, upper, yA.data(), AyA.data());
        GNC_CUDA_LAUNCH_CHECK();
        const double rA0AyA = sumProd(rA0.data(), AyA.data());
        alpha = rA0rA/rA0AyA;
        kAxmy<<<nBlocks(sn), BS>>>(n, alpha, rA.data(), AyA.data(), sA.data());
        GNC_CUDA_LAUNCH_CHECK();
        perf.finalResidual = sumMag(sA.data())/normFactor;
        if (perf.nIterations >= ctrl.minIter && checkConvergence(perf, ctrl))
        {
            kAddScaledTo<<<nBlocks(sn), BS>>>(n, alpha, yA.data(), x);
            GNC_CUDA_LAUNCH_CHECK();
            ++perf.nIterations;
            return perf;
        }
        precon.precondition(zA.data(), sA.data());
        kAmul<<<nBlocks(sn), BS>>>(g, dm.owner.data(), dm.neighbour.data(), D, lower, upper, zA.data(), tA.data());
        GNC_CUDA_LAUNCH_CHECK();
        kSqrOf<<<nBlocks(sn), BS>>>(n, tA.data(), work.data());
        GNC_CUDA_LAUNCH_CHECK();
        const double tAtA = sumSolver(work.data(), sn);
        omega = sumProd(tA.data(), sA.data())/tAtA;
        kPbcgUpdate<<<nBlocks(sn), BS>>>(n, alpha, omega, yA.data(), zA.data(), sA.data(), tA.data(), x, rA.data());
        GNC_CUDA_LAUNCH_CHECK();
        perf.finalResidual = sumMag(rA.data())/normFactor;
    } while
    (
        (++perf.nIterations < ctrl.maxIter && !checkConvergence(perf, ctrl))
     || perf.nIterations < ctrl.minIter
    );
    return perf;
}


SolverPerf solveScalarMatrix
(
    const DeviceMesh& dm,
    const LusgsPlan& plan,
    FvScalarMatrix& A,
    VolField<double>& psi,
    const SmoothSolverControls& ctrl,
    const SurfField<double>* phi,
    const bool coeffsUpdated
)
{
    if (ctrl.solver == "smoothSolver")
    {
        return solveSmoothSymGaussSeidel(dm, plan, A, psi, ctrl, phi, coeffsUpdated);
    }
    const std::size_t sn = std::size_t(dm.nCells);
    ensureDiag(dm, A);
    if (!A.hasUpper && !A.hasLower)
    {
        throw std::runtime_error("solveScalarMatrix: diagonal matrix is not supported");
    }
    if (ctrl.solver == "PCG" && A.hasLower)
    {
        throw std::runtime_error("solveScalarMatrix: PCG needs a symmetric matrix");
    }
    if (ctrl.solver == "GAMG" && A.hasLower)
    {
        throw std::runtime_error("solveScalarMatrix: GAMG (performance mode) needs a symmetric matrix");
    }
    const double* lower = A.hasLower ? A.lower.data() : A.upper.data();
    const double* upper = A.hasUpper ? A.upper.data() : A.lower.data();
    const Gather g = gatherOf(dm);
    DeviceArray<double> D(sn), b(sn);
    if (sn)
    {
        kAddBoundary<<<nBlocks(sn), BS>>>(g, A.diag.data(), A.internalCoeffs.data(), D.data());
        kAddBoundary<<<nBlocks(sn), BS>>>(g, A.source.data(), A.boundaryCoeffs.data(), b.data());
        GNC_CUDA_LAUNCH_CHECK();
    }
    const SolverPerf perf = solveLduKrylov
    (
        dm, plan, D.data(), lower, upper, b.data(), psi.in.view().p[0], ctrl
    );
    evaluateBCs(dm, psi, phi, coeffsUpdated);
    return perf;
}

} // namespace gnc
