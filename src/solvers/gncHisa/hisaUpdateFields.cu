// GPU-Native-CFD，GPL-3.0-or-later。見 hisaUpdateFields.cuh。

#include "solvers/gncHisa/hisaUpdateFields.cuh"

#include "core/reduce.cuh"
#include "fields/boundaryConditions.cuh"
#include "thermo/psiThermo.cuh"

#include <stdexcept>
#include <string>
#include <vector>

namespace gnc
{

namespace
{

constexpr int BS = 256;
constexpr double GREAT = 1.0e+15;
constexpr double VGREAT = 1.0e+300;
constexpr int maxNewton = 1000;

int nBlocks(const std::int32_t n)
{
    return int((n + BS - 1)/BS);
}

// OpenFOAM 純量的 max、min（primitives/ints/int/int.H 的 MAXMIN）
__device__ __forceinline__ double hmax(const double s1, const double s2)
{
    return (s2 < s1) ? s1 : s2;
}

__device__ __forceinline__ double hmin(const double s1, const double s2)
{
    return (s1 < s2) ? s1 : s2;
}


// ---------------------------------------------------------------------------
// localTimesteppingBounding

// 一格（或一個邊界面）：越界判斷（viol 非空時）與夾住 rho、eTemp、重算 rhoE
__global__ void kBound
(
    const std::int32_t n,
    const double lb,
    const double ub,
    const double* __restrict__ rhoP,
    const CView<vec3> rhoUP,
    const double* __restrict__ rhoEP,
    double* __restrict__ rho,
    const CView<vec3> rhoU,
    double* __restrict__ rhoE,
    unsigned char* __restrict__ viol
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const double rP = rhoP[i];
    const double rhoMin = lb*rP;
    const double rhoMax = ub*rP;
    const double eP = rhoEP[i]/rP - 0.5*magSqr(rhoUP[i]/rP);
    const double eMin = lb*eP;
    const double eMax = ub*eP;

    double r = rho[i];
    const vec3 ru = rhoU[i];
    double eT = rhoE[i]/r - 0.5*magSqr(ru/r);
    if (viol)
    {
        viol[i] =
            (r < rhoMin) || (r > rhoMax) || (eT < eMin) || (eT > eMax) || (eT < SMALL);
    }
    // rho = max(rho, rhoMin); rho = min(rho, rhoMax); eTemp 同樣
    r = hmax(r, rhoMin);
    r = hmin(r, rhoMax);
    eT = hmax(eT, eMin);
    eT = hmin(eT, eMax);
    rho[i] = r;
    rhoE[i] = r*(eT + 0.5*magSqr(ru/r));
}

// factor 與 pseudoCoDamping（內部值）。每格依面收集，factor 只取 min，與面的
// 次序無關。
__global__ void kFactorDamping
(
    const std::int32_t nC,
    const std::int32_t nIF,
    const std::int32_t* __restrict__ off,
    const std::int32_t* __restrict__ faces,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const unsigned char* __restrict__ viol,
    const double bd,
    const double bi,
    const double bmid,
    const double minRatio,
    double* __restrict__ damping
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC)
    {
        return;
    }
    double factor = GREAT;
    for (std::int32_t q = off[c]; q < off[c + 1]; ++q)
    {
        const std::int32_t f = faces[q];
        if (f >= nIF)
        {
            continue;
        }
        const std::int32_t o = (owner[f] == c) ? neighbour[f] : owner[f];
        factor = hmin(viol[c] ? bd : bi, factor);
        if (viol[o])
        {
            factor = hmin(bmid, factor);
        }
    }
    double d = damping[c]*factor;
    d = hmin(1.0, d);
    d = hmax(minRatio, d);
    damping[c] = d;
}

// 邊界：factor 的邊界值為 1
__global__ void kDampingBoundary
(
    const std::int32_t n,
    const double minRatio,
    double* __restrict__ damping
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        double d = damping[i]*1.0;
        d = hmin(1.0, d);
        d = hmax(minRatio, d);
        damping[i] = d;
    }
}


// ---------------------------------------------------------------------------
// updateFields

// 任一值 < thr 時 flag = 1
__global__ void kAnyLess
(
    const std::int32_t n, const double* __restrict__ a, const double thr, int* __restrict__ flag
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n && a[i] < thr)
    {
        *flag = 1;
    }
}

// U = rhoU/rho；e = rhoE/rho - 0.5*magSqr(U) + eZero（內部值）
__global__ void kPrimitive
(
    const std::int32_t n,
    const double* __restrict__ rho,
    const CView<vec3> rhoU,
    const double* __restrict__ rhoE,
    const double* __restrict__ eZero,
    const DView<vec3> U,
    double* __restrict__ e
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const vec3 u = rhoU[i]/rho[i];
        U.set(i, u);
        e[i] = rhoE[i]/rho[i] - 0.5*magSqr(u) + eZero[i];
    }
}

// e 相對於 he(p, Tb) 的檢查：sign < 0 時找 e - eb < 0（neg），> 0 時找
// e - eb > 0（pos）
__global__ void kEnergyBoundCheck
(
    const std::int32_t n,
    const ThermoParams P,
    const double* __restrict__ p,
    const double* __restrict__ e,
    const double Tb,
    const int sign,
    int* __restrict__ flag
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const double d = e[i] - thermo::Es(P, p[i], Tb);
        if ((sign < 0 && d < 0) || (sign > 0 && d > 0))
        {
            *flag = 1;
        }
    }
}

// e = max(e, he(p, Tb))（upper 時 min）
__global__ void kEnergyBoundApply
(
    const std::int32_t n,
    const ThermoParams P,
    const double* __restrict__ p,
    const double Tb,
    const bool upper,
    double* __restrict__ e
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const double eb = thermo::Es(P, p[i], Tb);
        e[i] = upper ? foamMin(e[i], eb) : foamMax(e[i], eb);
    }
}

// 有限體積邊界面取相鄰格值（純量 transform 族的 evaluate）
__global__ void kBoundaryFromOwner
(
    const std::int32_t start,
    const std::int32_t n,
    const std::int32_t* __restrict__ fvbFaces,
    const std::int32_t* __restrict__ owner,
    const double* __restrict__ in,
    double* __restrict__ bnd
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const std::int32_t b = start + i;
        bnd[b] = in[owner[fvbFaces[b]]];
    }
}

// p += (rho - rhoPrev)/psi
__global__ void kPressureInit
(
    const std::int32_t n,
    const double* __restrict__ rho,
    const double* __restrict__ rhoP,
    const double* __restrict__ psi,
    double* __restrict__ p
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const double dRho = rho[i] - rhoP[i];
        p[i] = p[i] + dRho/psi[i];
    }
}

// deltap = -(p*psi - rho)/psi；p += deltap；ratio = mag(deltap/stabilise(p, SMALL))
__global__ void kPressureNewton
(
    const std::int32_t n,
    const double* __restrict__ rho,
    const double* __restrict__ psi,
    double* __restrict__ p,
    double* __restrict__ ratio
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const double trho = p[i]*psi[i];
        const double dp = -(trho - rho[i])/psi[i];
        const double pn = p[i] + dp;
        p[i] = pn;
        const double st = (pn >= 0) ? pn + SMALL : pn - SMALL;
        ratio[i] = ::fabs(dp/st);
    }
}


// rhoU = rho*U；rhoE = rho*(e - eZero + 0.5*magSqr(U))（內部值）
__global__ void kConserved
(
    const std::int32_t n,
    const double* __restrict__ rho,
    const CView<vec3> U,
    const double* __restrict__ e,
    const double* __restrict__ eZero,
    const DView<vec3> rhoU,
    double* __restrict__ rhoE
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const vec3 u = U[i];
        rhoU.set(i, rho[i]*u);
        rhoE[i] = rho[i]*(e[i] - eZero[i] + 0.5*magSqr(u));
    }
}

// T 的邊界 == max(T, TMin)（TMax < GREAT 時再 == min(T, TMax)）
__global__ void kTBound
(
    const std::int32_t n, const double TMin, const bool useTMax, const double TMax,
    double* __restrict__ T
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        double t = hmax(T[i], TMin);
        if (useTMax)
        {
            t = hmin(t, TMax);
        }
        T[i] = t;
    }
}

// e 的邊界 == he(p, T)
__global__ void kHeBoundary
(
    const std::int32_t n, const ThermoParams P, const double* __restrict__ p,
    const double* __restrict__ T, double* __restrict__ e
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        e[i] = thermo::Es(P, p[i], T[i]);
    }
}

// 一個 patch 的 rho 邊界值：trho = p*psi 的邊界值，或（characteristicWallPressure）
// 相鄰格的 trho
__global__ void kRhoBoundary
(
    const std::int32_t start,
    const std::int32_t size,
    const bool wallPressure,
    const std::int32_t* __restrict__ fvbFaces,
    const std::int32_t* __restrict__ owner,
    const double* __restrict__ pIn,
    const double* __restrict__ psiIn,
    const double* __restrict__ pB,
    const double* __restrict__ psiB,
    double* __restrict__ rhoB
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= size)
    {
        return;
    }
    const std::int32_t k = start + i;
    if (wallPressure)
    {
        const std::int32_t c = owner[fvbFaces[k]];
        rhoB[k] = pIn[c]*psiIn[c];
    }
    else
    {
        rhoB[k] = pB[k]*psiB[k];
    }
}

// rhoU、rhoE 的邊界值
__global__ void kConservedBoundary
(
    const std::int32_t n,
    const double* __restrict__ rho,
    const CView<vec3> U,
    const double* __restrict__ e,
    const double* __restrict__ eZero,
    const DView<vec3> rhoU,
    double* __restrict__ rhoE
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const vec3 u = U[i];
        rhoU.set(i, rho[i]*u);
        rhoE[i] = rho[i]*(e[i] - eZero[i] + 0.5*magSqr(u));
    }
}


bool anyFlag(DeviceArray<int>& flag)
{
    GNC_CUDA_LAUNCH_CHECK();
    return flag.download()[0] != 0;
}

} // namespace


PrevIterFields storePrevIter(const HisaState& s)
{
    PrevIterFields p;
    p.rhoI.copyFrom(s.rho.in);
    p.rhoB.copyFrom(s.rho.bnd);
    p.rhoEI.copyFrom(s.rhoE.in);
    p.rhoEB.copyFrom(s.rhoE.bnd);
    p.rhoUI.copyFrom(s.rhoU.in);
    p.rhoUB.copyFrom(s.rhoU.bnd);
    return p;
}


void boundSolution
(
    const DeviceMesh& dm,
    const PseudoTimeControls& c,
    const PrevIterFields& prev,
    HisaState& s,
    PseudoTimeState& pt
)
{
    if (!c.localTimesteppingBounding)
    {
        return;
    }
    const std::int32_t nC = dm.nCells;
    const std::int32_t nB = dm.nFvBoundaryFaces();
    const double lb = c.localTimesteppingLowerBound;
    const double ub = c.localTimesteppingUpperBound;

    DeviceArray<unsigned char> viol(static_cast<std::size_t>(nC));
    if (nC)
    {
        kBound<<<nBlocks(nC), BS>>>
        (
            nC, lb, ub, prev.rhoI.cview().p[0], prev.rhoUI.cview(), prev.rhoEI.cview().p[0],
            s.rho.in.view().p[0], s.rhoU.in.cview(), s.rhoE.in.view().p[0], viol.data()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (nB)
    {
        kBound<<<nBlocks(nB), BS>>>
        (
            nB, lb, ub, prev.rhoB.cview().p[0], prev.rhoUB.cview(), prev.rhoEB.cview().p[0],
            s.rho.bnd.view().p[0], s.rhoU.bnd.cview(), s.rhoE.bnd.view().p[0], nullptr
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    // rho = ...、rhoE = ... 的指定：transformFvPatchField 以新的內部值 evaluate
    finishAssignment(dm, s.rho);
    finishAssignment(dm, s.rhoE);

    const double bd = c.boundingDecrease;
    const double bi = c.boundingIncrease;
    const double bmid = (1.0 + bd)/2.0;
    const double minRatio = c.pseudoCoNumMin/pt.pseudoCoNum;
    if (nC)
    {
        kFactorDamping<<<nBlocks(nC), BS>>>
        (
            nC, dm.nInternalFaces, dm.cellFaceOffsets.data(), dm.cellFaces.data(),
            dm.owner.data(), dm.neighbour.data(), viol.data(), bd, bi, bmid, minRatio,
            pt.pseudoCoDamping.in.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (nB)
    {
        kDampingBoundary<<<nBlocks(nB), BS>>>(nB, minRatio, pt.pseudoCoDamping.bnd.view().p[0]);
        GNC_CUDA_LAUNCH_CHECK();
    }
    finishAssignment(dm, pt.pseudoCoDamping);
    GNC_CUDA_CHECK(cudaDeviceSynchronize());
}


UpdateFieldsInfo updateFields
(
    const DeviceMesh& dm,
    const PseudoTimeControls& c,
    const PrevIterFields& prev,
    const HisaFlux& flux,
    HisaState& s
)
{
    const std::int32_t nC = dm.nCells;
    const std::int32_t nB = dm.nFvBoundaryFaces();
    PsiThermoFields& th = s.thermo;
    const ThermoParams& P = th.params;
    DeviceArray<int> flag(1);

    // bound(rho, rhoMin)：min(rho)（內部與邊界）< rhoMin 時才動作
    flag.zero();
    if (nC) kAnyLess<<<nBlocks(nC), BS>>>(nC, s.rho.in.cview().p[0], c.rhoMin, flag.data());
    if (nB) kAnyLess<<<nBlocks(nB), BS>>>(nB, s.rho.bnd.cview().p[0], c.rhoMin, flag.data());
    if (anyFlag(flag))
    {
        throw std::runtime_error("updateFields: rho below rhoMin (bound() is not supported)");
    }

    if (nC)
    {
        kPrimitive<<<nBlocks(nC), BS>>>
        (
            nC, s.rho.in.cview().p[0], s.rhoU.in.cview(), s.rhoE.in.cview().p[0],
            s.eZero.in.cview().p[0], s.U.in.view(), th.he.in.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }

    UpdateFieldsInfo info;
    // max(neg(e - he(p, TMin))) > 0.5、max(pos(e - he(p, TMax))) > 0.5
    for (int bound = 0; bound < 2; ++bound)
    {
        const bool upper = bound == 1;
        if (upper && !(c.TMax < GREAT))
        {
            continue;
        }
        const double Tb = upper ? c.TMax : c.TMin;
        const int sign = upper ? 1 : -1;
        flag.zero();
        if (nC)
        {
            kEnergyBoundCheck<<<nBlocks(nC), BS>>>
            (
                nC, P, th.p.in.cview().p[0], th.he.in.cview().p[0], Tb, sign, flag.data()
            );
        }
        if (nB)
        {
            kEnergyBoundCheck<<<nBlocks(nB), BS>>>
            (
                nB, P, th.p.bnd.cview().p[0], th.he.bnd.cview().p[0], Tb, sign, flag.data()
            );
        }
        if (!anyFlag(flag))
        {
            continue;
        }
        // e = max(e, eBound)（TMax：min）：GeometricField::operator=，內部逐格；邊界逐 patch 的
        // operator=(const fvPatchField&)：fixedEnergy（fixedValue 族）與 mixedEnergy（mixed 族）
        // 不改值，transform 族（symmetry 等）以新的內部值 evaluate，其他（gradientEnergy 等）照抄
        (upper ? info.energyBoundedMax : info.energyBoundedMin) = true;
        if (nC)
        {
            kEnergyBoundApply<<<nBlocks(nC), BS>>>
            (
                nC, P, th.p.in.cview().p[0], Tb, upper, th.he.in.view().p[0]
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
        for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
        {
            const std::int32_t n = dm.fvPatchSize[pi];
            const std::int32_t start = dm.fvPatchStart[pi];
            const Bc t = th.he.bc[pi].type;
            if (n == 0 || isFixedValueFamily(t) || isMixedFamily(t))
            {
                continue;
            }
            if (t == Bc::symmetry || t == Bc::symmetryPlane || t == Bc::slip)
            {
                kBoundaryFromOwner<<<nBlocks(n), BS>>>
                (
                    start, n, dm.fvBoundaryFaces.data(), dm.owner.data(), th.he.in.cview().p[0],
                    th.he.bnd.view().p[0]
                );
            }
            else
            {
                kEnergyBoundApply<<<nBlocks(n), BS>>>
                (
                    n, P, th.p.bnd.cview().p[0] + start, Tb, upper, th.he.bnd.view().p[0] + start
                );
            }
            GNC_CUDA_LAUNCH_CHECK();
        }
    }

    // thermo.correct()，再以 Newton 法使 rho = p*psi
    calculatePsiThermo(dm, th);
    if (nC)
    {
        kPressureInit<<<nBlocks(nC), BS>>>
        (
            nC, s.rho.in.cview().p[0], prev.rhoI.cview().p[0], th.psi.in.cview().p[0],
            th.p.in.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    {
        DeviceArray<double> ratio(static_cast<std::size_t>(nC));
        while (true)
        {
            calculatePsiThermo(dm, th);
            if (nC)
            {
                kPressureNewton<<<nBlocks(nC), BS>>>
                (
                    nC, s.rho.in.cview().p[0], th.psi.in.cview().p[0], th.p.in.view().p[0],
                    ratio.data()
                );
                GNC_CUDA_LAUNCH_CHECK();
            }
            // gMax（空的時候 -VGREAT）；只和 1e-5 比較，見 maxValue
            const double maxRatio = nC ? maxValue(ratio.data(), std::size_t(nC)) : -VGREAT;
            ++info.newtonIterations;
            if (maxRatio < 1e-5)
            {
                break;
            }
            if (info.newtonIterations >= maxNewton)
            {
                throw std::runtime_error
                (
                    "updateFields: Newton's method for p did not converge in "
                  + std::to_string(maxNewton) + " iterations"
                );
            }
        }
    }

    if (nC)
    {
        kConserved<<<nBlocks(nC), BS>>>
        (
            nC, s.rho.in.cview().p[0], s.U.in.cview(), th.he.in.cview().p[0],
            s.eZero.in.cview().p[0], s.rhoU.in.view(), s.rhoE.in.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }

    // p、U、T 的 correctBoundaryConditions：jacobian::addBoundaryTerms 已呼叫
    // updateCoeffs，evaluate 用那時的係數
    evaluateBCs(dm, th.p, &flux.phi, true);
    evaluateBCs(dm, s.U, &flux.phi, true);
    evaluateBCs(dm, th.T, &flux.phi, true);
    if (nB)
    {
        kTBound<<<nBlocks(nB), BS>>>(nB, c.TMin, c.TMax < GREAT, c.TMax, th.T.bnd.view().p[0]);
        kHeBoundary<<<nBlocks(nB), BS>>>
        (
            nB, P, th.p.bnd.cview().p[0], th.T.bnd.cview().p[0], th.he.bnd.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    calculatePsiThermo(dm, th);

    // rho 的邊界（逐 patch 的 fvPatchField::operator=），再 rhoU、rhoE
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const std::int32_t n = dm.fvPatchSize[pi];
        if (!n)
        {
            continue;
        }
        kRhoBoundary<<<nBlocks(n), BS>>>
        (
            dm.fvPatchStart[pi], n, th.p.bc[pi].type == Bc::characteristicWallPressure,
            dm.fvBoundaryFaces.data(), dm.owner.data(), th.p.in.cview().p[0],
            th.psi.in.cview().p[0], th.p.bnd.cview().p[0], th.psi.bnd.cview().p[0],
            s.rho.bnd.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    finishAssignment(dm, s.rho);
    if (nB)
    {
        kConservedBoundary<<<nBlocks(nB), BS>>>
        (
            nB, s.rho.bnd.cview().p[0], s.U.bnd.cview(), th.he.bnd.cview().p[0],
            s.eZero.bnd.cview().p[0], s.rhoU.bnd.view(), s.rhoE.bnd.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    finishAssignment(dm, s.rhoU);
    finishAssignment(dm, s.rhoE);
    GNC_CUDA_CHECK(cudaDeviceSynchronize());
    return info;
}

} // namespace gnc
