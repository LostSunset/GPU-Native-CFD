// 向量的 fvMatrix（fvVectorMatrix）：參考次序版（N6）。GPU-Native-CFD，GPL-3.0-or-later。
// 出處與次序見 fvVectorMatrix.cuh。

#include "fvm/fvVectorMatrix.cuh"

#include "fields/boundaryConditions.cuh"
#include "ops/fvcOps.cuh"

#include <stdexcept>
#include <string>

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

CView<vec3> vecView(const DeviceVec3Array& a)
{
    CView<vec3> v;
    v.p[0] = a.x.data();
    v.p[1] = a.y.data();
    v.p[2] = a.z.data();
    return v;
}

// cmptMax、cmptMin、cmptAv（VectorSpaceI.H：依分量次序 maxEqOp、minEqOp、
// plusEqOp，cmptAv = cmptSum/3）
__device__ __forceinline__ double cmptMaxMag(const vec3& v)
{
    return foamMax(foamMax(::fabs(v.x), ::fabs(v.y)), ::fabs(v.z));
}

__device__ __forceinline__ double cmptMinOf(const vec3& v)
{
    return foamMin(foamMin(v.x, v.y), v.z);
}

__device__ __forceinline__ double cmptAvOf(const vec3& v)
{
    double s = v.x;
    s += v.y;
    s += v.z;
    return s/3;
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

// source ∓= V*su（每個分量）
__global__ void kSourceVTimes
(
    const std::int32_t n,
    const double sgn,               // -1：A + su；+1：A == su
    const double* __restrict__ V,
    const CView<vec3> su,
    const DView<vec3> S
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= n)
    {
        return;
    }
    const vec3 vs = V[c]*su[c];
    const vec3 s = S[c];
    S.set(c, sgn > 0 ? s + vs : s - vs);
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
    if (s.size())
    {
        kCopy<<<nBlocks(s.size()), BS>>>(s.size(), s.data(), d.data());
        GNC_CUDA_LAUNCH_CHECK();
    }
}

// lduMatrix::negSumDiag（overlay）：每面 diag[l] += -lower、diag[u] += -upper
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

// gaussConvectionScheme::fvmDiv（upwind 權重 pos0(phi)）：lower = (-w)*phi、upper = lower + phi
__global__ void kUpwindFaces
(
    const std::int32_t nIF,
    const double* __restrict__ phi,
    double* __restrict__ lower,
    double* __restrict__ upper
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIF)
    {
        return;
    }
    const double w = (phi[f] >= 0.0) ? 1.0 : 0.0;
    const double l = (-w)*phi[f];
    lower[f] = l;
    upper[f] = l + phi[f];
}

// fvmLaplacianUncorrected 的內部面：gammaMagSf = gamma_f*magSf、
// upper = nonOrthDeltaCoeffs*gammaMagSf
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


// 向量邊界條件在矩陣中的型別
enum VKind : int
{
    vkFixedValue = 0,
    vkZeroGradient = 1,
    vkMixed = 2
};

struct VPatchIn
{
    std::int32_t start;
    std::int32_t n;
    int kind;
    const std::int32_t* fvbFaces;
    CView<vec3> valueB;             // 邊界值（fvBoundaryFaces 序號）
    CView<vec3> refValue;           // mixed（patch 內的索引）
    CView<vec3> refGrad;
    const double* valueFraction;
};

// gaussConvectionScheme 的邊界：internalCoeffs = phi_b*valueInternalCoeffs、
// boundaryCoeffs = (-phi_b)*valueBoundaryCoeffs
//   fixedValue    vIC 0、vBC 值
//   zeroGradient  vIC one、vBC 0
//   mixed         vIC one*(1 - vf)、vBC lerp(refGrad/dc, refValue, vf)
__global__ void kDivPatch
(
    const VPatchIn p,
    const double* __restrict__ phiB,
    const double* __restrict__ dc,
    const DView<vec3> ic,
    const DView<vec3> bc
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= p.n)
    {
        return;
    }
    const std::int32_t b = p.start + i;
    const double phi = phiB[b];
    vec3 vIC{0.0, 0.0, 0.0};
    vec3 vBC{0.0, 0.0, 0.0};
    switch (p.kind)
    {
        case vkFixedValue:
            vBC = p.valueB[b];
            break;
        case vkZeroGradient:
            vIC = vec3{1.0, 1.0, 1.0};
            break;
        case vkMixed:
        {
            const std::int32_t f = p.fvbFaces[b];
            const double vf = p.valueFraction[i];
            const double one = 1.0*(1.0 - vf);
            vIC = vec3{one, one, one};
            vBC = lerp(p.refGrad[i]/dc[f], p.refValue[i], vf);
            break;
        }
    }
    ic.set(b, phi*vIC);
    bc.set(b, (-phi)*vBC);
}

// fvmLaplacianUncorrected 的邊界：internalCoeffs = pGamma*gradientInternalCoeffs、
// boundaryCoeffs = (-pGamma)*gradientBoundaryCoeffs，pGamma = gamma_b*magSf
//   fixedValue    gIC (-one)*dc、gBC dc*值
//   zeroGradient  0、0
//   mixed         gIC ((-one)*vf)*dc、gBC lerp(refGrad, dc*refValue, vf)
__global__ void kLapPatch
(
    const VPatchIn p,
    const double* __restrict__ gammaFB,
    const double* __restrict__ magSf,
    const double* __restrict__ dc,
    double* __restrict__ gammaMagSfB,
    const DView<vec3> ic,
    const DView<vec3> bc
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
    vec3 gIC{0.0, 0.0, 0.0};
    vec3 gBC{0.0, 0.0, 0.0};
    switch (p.kind)
    {
        case vkFixedValue:
        {
            const double d = (-1.0)*dc[f];
            gIC = vec3{d, d, d};
            gBC = dc[f]*p.valueB[b];
            break;
        }
        case vkZeroGradient:
            break;
        case vkMixed:
        {
            const double vf = p.valueFraction[i];
            const double d = ((-1.0)*vf)*dc[f];
            gIC = vec3{d, d, d};
            gBC = lerp(p.refGrad[i], dc[f]*p.refValue[i], vf);
            break;
        }
    }
    ic.set(b, pGamma*gIC);
    bc.set(b, (-pGamma)*gBC);
}

// linearUpwind<vector>::correction 乘以面通量：phi*((Cf - C[c]) & gradU[c])，
// c = phi > 0 ? owner : neighbour
__global__ void kLinearUpwindCorrFlux
(
    const std::int32_t nIF,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ phi,
    const CView<vec3> Cf,
    const CView<vec3> C,
    const CView<tens> gradU,
    const DView<vec3> out
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIF)
    {
        return;
    }
    const std::int32_t c = (phi[f] > 0) ? owner[f] : neighbour[f];
    const vec3 corr = dot(Cf[f] - C[c], gradU[c]);
    out.set(f, phi[f]*corr);
}

// 邊界的修正為 0：phi_b*0
__global__ void kTimesZero(const std::int32_t n, const double* __restrict__ phi, const DView<vec3> out)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out.set(i, phi[i]*vec3{0.0, 0.0, 0.0});
    }
}

// bounded：diag -= (0 + V*sInt)（fvm::Sp 的 diag 從 0 起加）
__global__ void kSubSp
(
    const std::int32_t n,
    const double* __restrict__ V,
    const double* __restrict__ sp,
    double* __restrict__ diag
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < n)
    {
        const double d = 0.0 + V[c]*sp[c];
        diag[c] = diag[c] - d;
    }
}

// fvMatrix::relax(alpha)（非耦合 patch）
__global__ void kRelax
(
    const Gather g,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    const CView<vec3> ic,
    const CView<vec3> psi,
    const double alpha,
    double* __restrict__ D,
    const DView<vec3> S
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
                d += cmptMaxMag(ic[b]);
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
                d -= cmptMinOf(ic[b]);
            }
        }
    }
    D[c] = d;
    S.set(c, S[c] + (d - d0)*psi[c]);
}

// fvMatrix::D()/V：diag 加上各邊界面的 cmptAv(iCoeffs)，再除以 V
__global__ void kAOverV
(
    const Gather g,
    const double* __restrict__ diag,
    const CView<vec3> ic,
    const double* __restrict__ V,
    double* __restrict__ out
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= g.nCells)
    {
        return;
    }
    double d = diag[c];
    for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
    {
        const std::int32_t f = g.face[k];
        if (f >= g.nInternal)
        {
            const std::int32_t b = fvbOf(g, f);
            if (b >= 0)
            {
                d += cmptAvOf(ic[b]);
            }
        }
    }
    out[c] = d/V[c];
}

// fvMatrix::H()（內部格）
__global__ void kH
(
    const Gather g,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    const CView<vec3> ic,
    const CView<vec3> bcoeff,
    const CView<vec3> psi,
    const CView<vec3> source,
    const double* __restrict__ V,
    const int v0,
    const int v1,
    const int v2,
    const DView<vec3> H
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= g.nCells)
    {
        return;
    }
    const vec3 pc = psi[c];
    const double pcc[3] = {pc.x, pc.y, pc.z};
    double h[3];
    for (int d = 0; d < 3; ++d)
    {
        // boundaryDiagCmpt：addBoundaryDiag(cmpt)、negate、addCmptAvBoundaryDiag
        double bd = 0.0;
        for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
        {
            const std::int32_t f = g.face[k];
            if (f >= g.nInternal)
            {
                const std::int32_t b = fvbOf(g, f);
                if (b >= 0)
                {
                    bd += cmpt(ic[b], d);
                }
            }
        }
        bd = -bd;
        for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
        {
            const std::int32_t f = g.face[k];
            if (f >= g.nInternal)
            {
                const std::int32_t b = fvbOf(g, f);
                if (b >= 0)
                {
                    bd += cmptAvOf(ic[b]);
                }
            }
        }
        h[d] = bd*pcc[d];
    }
    // lduMatrix::H(psi)：Hpsi[u] -= lower*psi[l]、Hpsi[l] -= upper*psi[u]，依面編號
    vec3 hpsi{0.0, 0.0, 0.0};
    for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
    {
        const std::int32_t f = g.face[k];
        if (f < g.nInternal)
        {
            if (g.sign[k] > 0)
            {
                hpsi -= upper[f]*psi[neighbour[f]];
            }
            else
            {
                hpsi -= lower[f]*psi[owner[f]];
            }
        }
    }
    const vec3 t = hpsi + source[c];
    vec3 r{h[0] + t.x, h[1] + t.y, h[2] + t.z};
    // addBoundarySource
    for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)
    {
        const std::int32_t f = g.face[k];
        if (f >= g.nInternal)
        {
            const std::int32_t b = fvbOf(g, f);
            if (b >= 0)
            {
                r += bcoeff[b];
            }
        }
    }
    r = r/V[c];
    if (v0 < 0) { r.x = 0.0; }
    if (v1 < 0) { r.y = 0.0; }
    if (v2 < 0) { r.z = 0.0; }
    H.set(c, r);
}

// addBoundaryDiag(cmpt)／addBoundarySource 的分量：out = in + Σ coeff.cmpt（依 patch、面）
__global__ void kAddBoundaryCmpt
(
    const Gather g,
    const double* __restrict__ in,
    const CView<vec3> coeff,
    const int d,
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
                acc += cmpt(coeff[b], d);
            }
        }
    }
    out[c] = acc;
}


void ensureDiag(const DeviceMesh& dm, FvVectorMatrix& A)
{
    if (!A.hasDiag)
    {
        A.diag.allocate(std::size_t(dm.nCells));
        fill(A.diag, 0.0);
        A.hasDiag = true;
    }
}

// lduMatrix::lower()（非 const）：沒有時配置，有 upper 時複製 upper
void ensureLower(const DeviceMesh& dm, FvVectorMatrix& A)
{
    if (!A.hasLower)
    {
        A.lower.allocate(std::size_t(dm.nInternalFaces));
        if (A.hasUpper)
        {
            copyArray(A.upper, A.lower);
        }
        else
        {
            fill(A.lower, 0.0);
        }
        A.hasLower = true;
    }
}

void ensureUpper(const DeviceMesh& dm, FvVectorMatrix& A)
{
    if (!A.hasUpper)
    {
        A.upper.allocate(std::size_t(dm.nInternalFaces));
        if (A.hasLower)
        {
            copyArray(A.lower, A.upper);
        }
        else
        {
            fill(A.upper, 0.0);
        }
        A.hasUpper = true;
    }
}

// lduMatrix::operator+=、-= 與 fvMatrix 的 source、係數（N-028：結構的調整與各陣列的加減各成一批，
// 每個元素的運算與原本逐陣列的 kernel 相同；見 fvScalarMatrix.cu 的 lduCombine）
void lduCombine(const DeviceMesh& dm, FvVectorMatrix& A, const FvVectorMatrix& B, const bool minus)
{
    const std::size_t nIF = std::size_t(dm.nInternalFaces);
    bool upperFromB = false;
    bool lowerFromB = false;
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
            if (B.hasUpper)
            {
                A.upper.allocate(nIF);
                A.hasUpper = true;
                upperFromB = true;
            }
            if (B.hasLower)
            {
                A.lower.allocate(nIF);
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
            throw std::runtime_error("FvVectorMatrix: unknown matrix type combination");
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
    for (int k = 0; k < 3; ++k)
    {
        op(A.source.component(k), B.source.component(k));
        op(A.internalCoeffs.component(k), B.internalCoeffs.component(k));
        op(A.boundaryCoeffs.component(k), B.boundaryCoeffs.component(k));
    }
}

[[noreturn]] void unsupported(const DeviceMesh& dm, const VolField<vec3>& U, const std::size_t pi)
{
    throw std::runtime_error
    (
        "field " + U.name + " patch " + dm.patches[pi].name + ": " + bcName(U.bc[pi].type)
      + " is not supported in fvVectorMatrix"
    );
}

VPatchIn vpatchIn(const DeviceMesh& dm, const VolField<vec3>& U, const std::size_t pi)
{
    const PatchBC<vec3>& bc = U.bc[pi];
    VPatchIn p{};
    p.start = dm.fvPatchStart[pi];
    p.n = dm.fvPatchSize[pi];
    p.fvbFaces = dm.fvBoundaryFaces.data();
    p.valueB = U.bnd.cview();
    const Bc t = bc.type;
    if (isFixedValueFamily(t))
    {
        p.kind = vkFixedValue;
    }
    else if (isZeroGradientFamily(t))
    {
        p.kind = vkZeroGradient;
    }
    else if (isMixedFamily(t) && !isInletOutletFamily(t))
    {
        p.kind = vkMixed;
        p.refValue = bc.refValue.cview();
        p.refGrad = bc.refGrad.cview();
        p.valueFraction = bc.valueFraction.data();
    }
    else
    {
        unsupported(dm, U, pi);
    }
    return p;
}

} // namespace


// ---------------------------------------------------------------------------

FvVectorMatrix fvmEmptyVector(const DeviceMesh& dm)
{
    FvVectorMatrix A;
    A.source.allocate(std::size_t(dm.nCells));
    A.internalCoeffs.allocate(std::size_t(dm.nFvBoundaryFaces()));
    A.boundaryCoeffs.allocate(std::size_t(dm.nFvBoundaryFaces()));
    ArrayBatch b;
    A.source.zero(&b);
    A.internalCoeffs.zero(&b);
    A.boundaryCoeffs.zero(&b);
    return A;
}


FvVectorMatrix copyMatrix(const FvVectorMatrix& A)
{
    FvVectorMatrix C;
    C.hasDiag = A.hasDiag;
    C.hasLower = A.hasLower;
    C.hasUpper = A.hasUpper;
    ArrayBatch b;
    if (A.hasDiag)
    {
        C.diag.allocate(A.diag.size());
        b.copy(C.diag.data(), A.diag.data(), A.diag.size());
    }
    if (A.hasLower)
    {
        C.lower.allocate(A.lower.size());
        b.copy(C.lower.data(), A.lower.data(), A.lower.size());
    }
    if (A.hasUpper)
    {
        C.upper.allocate(A.upper.size());
        b.copy(C.upper.data(), A.upper.data(), A.upper.size());
    }
    C.source.copyFrom(A.source, &b);
    C.internalCoeffs.copyFrom(A.internalCoeffs, &b);
    C.boundaryCoeffs.copyFrom(A.boundaryCoeffs, &b);
    b.flush();
    return C;
}


void addMatrix(const DeviceMesh& dm, FvVectorMatrix& A, const FvVectorMatrix& B)
{
    lduCombine(dm, A, B, false);
}


void subtractMatrix(const DeviceMesh& dm, FvVectorMatrix& A, const FvVectorMatrix& B)
{
    lduCombine(dm, A, B, true);
}


void negateMatrix(FvVectorMatrix& A)
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
    for (int k = 0; k < 3; ++k)
    {
        b.negate(A.source.component(k).data(), A.source.component(k).size());
        b.negate(A.internalCoeffs.component(k).data(), A.internalCoeffs.component(k).size());
        b.negate(A.boundaryCoeffs.component(k).data(), A.boundaryCoeffs.component(k).size());
    }
}


void matrixPlusSu(const DeviceMesh& dm, FvVectorMatrix& A, const DField<vec3>& su)
{
    if (dm.nCells)
    {
        kSourceVTimes<<<nBlocks(std::size_t(dm.nCells)), BS>>>
        (
            dm.nCells, -1.0, dm.V.data(), su.cview(), A.source.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
}


void suMinusMatrix(const DeviceMesh& dm, const DField<vec3>& su, FvVectorMatrix& A)
{
    negateMatrix(A);
    matrixPlusSu(dm, A, su);
}


void matrixEqSu(const DeviceMesh& dm, FvVectorMatrix& A, const DField<vec3>& su)
{
    if (dm.nCells)
    {
        kSourceVTimes<<<nBlocks(std::size_t(dm.nCells)), BS>>>
        (
            dm.nCells, 1.0, dm.V.data(), su.cview(), A.source.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
}


FvVectorMatrix fvmDivBoundedGaussLinearUpwind
(
    const DeviceMesh& dm,
    const SurfField<double>& phi,
    const VolField<vec3>& U,
    const VolField<tens>& gradU
)
{
    const std::int32_t nIF = dm.nInternalFaces;
    const std::int32_t nB = dm.nFvBoundaryFaces();
    const std::int32_t nC = dm.nCells;
    FvVectorMatrix A = fvmEmptyVector(dm);
    ensureLower(dm, A);
    ensureUpper(dm, A);
    if (nIF)
    {
        kUpwindFaces<<<nBlocks(std::size_t(nIF)), BS>>>
        (
            nIF, phi.in.cview().p[0], A.lower.data(), A.upper.data()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    ensureDiag(dm, A);
    const Gather g = gatherOf(dm);
    if (nC)
    {
        kNegSumDiag<<<nBlocks(std::size_t(nC)), BS>>>(g, A.lower.data(), A.upper.data(), A.diag.data());
        GNC_CUDA_LAUNCH_CHECK();
    }
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        if (dm.fvPatchSize[pi] == 0)
        {
            continue;
        }
        const VPatchIn p = vpatchIn(dm, U, pi);
        kDivPatch<<<nBlocks(std::size_t(p.n)), BS>>>
        (
            p, phi.bnd.cview().p[0], dm.deltaCoeffs.data(),
            A.internalCoeffs.view(), A.boundaryCoeffs.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }

    // corrected：fvm += fvc::surfaceIntegrate(faceFlux*correction(U))
    SurfField<vec3> cf;
    cf.in.allocate(std::size_t(nIF));
    cf.bnd.allocate(std::size_t(nB));
    if (nIF)
    {
        kLinearUpwindCorrFlux<<<nBlocks(std::size_t(nIF)), BS>>>
        (
            nIF, dm.owner.data(), dm.neighbour.data(), phi.in.cview().p[0], vecView(dm.Cf),
            vecView(dm.C), gradU.in.cview(), cf.in.view()
        );
    }
    if (nB)
    {
        kTimesZero<<<nBlocks(std::size_t(nB)), BS>>>(nB, phi.bnd.cview().p[0], cf.bnd.view());
    }
    GNC_CUDA_LAUNCH_CHECK();
    DField<vec3> sCorr(static_cast<std::size_t>(nC));
    surfaceIntegrate(dm, cf, sCorr);
    matrixPlusSu(dm, A, sCorr);

    // bounded：- fvm::Sp(fvc::surfaceIntegrate(phi), U)（Sp 的 source 與邊界係數為 0，
    // 減去 0 不改值）
    DField<double> sInt(static_cast<std::size_t>(nC));
    surfaceIntegrate(dm, phi, sInt);
    if (nC)
    {
        kSubSp<<<nBlocks(std::size_t(nC)), BS>>>(nC, dm.V.data(), sInt.cview().p[0], A.diag.data());
        GNC_CUDA_LAUNCH_CHECK();
    }
    return A;
}


FvVectorMatrix fvmLaplacianGaussLinearCorrected
(
    const DeviceMesh& dm,
    const VolField<double>& gamma,
    const VolField<vec3>& U,
    const VolField<tens>& gradU
)
{
    const std::int32_t nIF = dm.nInternalFaces;
    const std::int32_t nB = dm.nFvBoundaryFaces();
    const std::int32_t nC = dm.nCells;

    SurfField<double> gammaF;
    gammaF.in.allocate(std::size_t(nIF));
    gammaF.bnd.allocate(std::size_t(nB));
    interpolateLinear(dm, gamma, gammaF);
    SurfField<double> gammaMagSf;
    gammaMagSf.in.allocate(std::size_t(nIF));
    gammaMagSf.bnd.allocate(std::size_t(nB));

    FvVectorMatrix A = fvmEmptyVector(dm);
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
    if (nC)
    {
        kNegSumDiag<<<nBlocks(std::size_t(nC)), BS>>>(g, A.upper.data(), A.upper.data(), A.diag.data());
        GNC_CUDA_LAUNCH_CHECK();
    }
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        if (dm.fvPatchSize[pi] == 0)
        {
            continue;
        }
        const VPatchIn p = vpatchIn(dm, U, pi);
        kLapPatch<<<nBlocks(std::size_t(p.n)), BS>>>
        (
            p, gammaF.bnd.cview().p[0], dm.magSf.data(), dm.deltaCoeffs.data(),
            gammaMagSf.bnd.view().p[0], A.internalCoeffs.view(), A.boundaryCoeffs.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }

    // 非正交修正：source -= V*fvc::div(gammaMagSf*correction(U))
    const SurfField<vec3> corrVecs = meshSurfVector(dm, dm.nonOrthCorrectionVectors);
    SurfField<vec3> corr;
    corr.in.allocate(std::size_t(nIF));
    corr.bnd.allocate(std::size_t(nB));
    dotInterpolateLinear(dm, corrVecs, gradU, corr);
    SurfField<vec3> flux;
    flux.in.allocate(std::size_t(nIF));
    flux.bnd.allocate(std::size_t(nB));
    multiplySurf(gammaMagSf, corr, flux);
    DField<vec3> divCorr(static_cast<std::size_t>(nC));
    surfaceIntegrate(dm, flux, divCorr);
    matrixPlusSu(dm, A, divCorr);
    return A;
}


void relaxMatrix
(
    const DeviceMesh& dm,
    FvVectorMatrix& A,
    const VolField<vec3>& U,
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
    if (dm.nCells)
    {
        kRelax<<<nBlocks(std::size_t(dm.nCells)), BS>>>
        (
            gatherOf(dm), lower, upper, A.internalCoeffs.cview(), U.in.cview(), alpha,
            A.diag.data(), A.source.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
}


VolField<double> matrixA(const DeviceMesh& dm, const FvVectorMatrix& A)
{
    if (!A.hasDiag)
    {
        throw std::runtime_error("matrixA: matrix without diagonal");
    }
    VolField<double> a = makeVolField<double>(dm, "A", Bc::extrapolatedCalculated);
    if (dm.nCells)
    {
        kAOverV<<<nBlocks(std::size_t(dm.nCells)), BS>>>
        (
            gatherOf(dm), A.diag.data(), A.internalCoeffs.cview(), dm.V.data(),
            a.in.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    evaluateBCs(dm, a, nullptr);
    return a;
}


VolField<vec3> matrixH(const DeviceMesh& dm, const FvVectorMatrix& A, const VolField<vec3>& U)
{
    if (!A.hasUpper && !A.hasLower)
    {
        throw std::runtime_error("matrixH: matrix without off-diagonal coefficients");
    }
    const double* lower = A.hasLower ? A.lower.data() : A.upper.data();
    const double* upper = A.hasUpper ? A.upper.data() : A.lower.data();
    VolField<vec3> H = makeVolField<vec3>(dm, "H", Bc::extrapolatedCalculated);
    if (dm.nCells)
    {
        kH<<<nBlocks(std::size_t(dm.nCells)), BS>>>
        (
            gatherOf(dm), dm.owner.data(), dm.neighbour.data(), lower, upper,
            A.internalCoeffs.cview(), A.boundaryCoeffs.cview(), U.in.cview(), A.source.cview(),
            dm.V.data(), dm.solutionD[0], dm.solutionD[1], dm.solutionD[2], H.in.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    // correctBoundaryConditions（patchInternalField）；無效分量在內部已為 0
    evaluateBCs(dm, H, nullptr);
    return H;
}


double VectorSolverPerf::maxInitialResidual() const
{
    // cmptMax：依分量次序 maxEqOp
    double r = cmpt[0].initialResidual;
    r = foamMax(r, cmpt[1].initialResidual);
    r = foamMax(r, cmpt[2].initialResidual);
    return r;
}


VectorSolverPerf solveVectorMatrix
(
    const DeviceMesh& dm,
    const LusgsPlan& plan,
    const FvVectorMatrix& A,
    VolField<vec3>& U,
    const SmoothSolverControls& ctrl,
    const SurfField<double>* phi,
    const bool coeffsUpdated
)
{
    if (ctrl.solver != "PBiCGStab")
    {
        throw std::runtime_error("solveVectorMatrix: only PBiCGStab (DILU) is supported");
    }
    if (!A.hasDiag || (!A.hasUpper && !A.hasLower))
    {
        throw std::runtime_error("solveVectorMatrix: incomplete matrix");
    }
    const double* lower = A.hasLower ? A.lower.data() : A.upper.data();
    const double* upper = A.hasUpper ? A.upper.data() : A.lower.data();
    const std::size_t nC = std::size_t(dm.nCells);
    const Gather g = gatherOf(dm);
    VectorSolverPerf perf;
    DeviceArray<double> D(nC), b(nC);
    for (int d = 0; d < 3; ++d)
    {
        if (dm.solutionD[d] < 0)
        {
            continue;
        }
        if (nC)
        {
            kAddBoundaryCmpt<<<nBlocks(nC), BS>>>(g, A.diag.data(), A.internalCoeffs.cview(), d, D.data());
            kAddBoundaryCmpt<<<nBlocks(nC), BS>>>
            (
                g, A.source.cview().p[d], A.boundaryCoeffs.cview(), d, b.data()
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
        perf.cmpt[d] = solveLduKrylov
        (
            dm, plan, D.data(), lower, upper, b.data(), U.in.view().p[d], ctrl
        );
        perf.solved[d] = true;
    }
    evaluateBCs(dm, U, phi, coeffsUpdated);
    return perf;
}

} // namespace gnc
