// GPU-Native-CFD，GPL-3.0-or-later。見 hisaPseudoTime.cuh。

#include "solvers/gncHisa/hisaPseudoTime.cuh"

#include "fields/boundaryConditions.cuh"
#include "io/dictionary.hpp"
#include "io/foamTokenizer.hpp"

#include <cmath>
#include <stdexcept>

namespace gnc
{

namespace
{

constexpr int BS = 256;

int nBlocks(const std::int32_t n)
{
    return int((n + BS - 1)/BS);
}

// OpenFOAM Switch 的文字
bool parseSwitch(const std::string& path, const std::string& key, const std::string& v)
{
    if (v == "true" || v == "on" || v == "yes" || v == "y" || v == "t") return true;
    if (v == "false" || v == "off" || v == "no" || v == "n" || v == "f" || v == "none") return false;
    throw std::runtime_error(path + ": " + key + ": invalid switch " + v);
}

double scalarOr(const Dictionary& d, const std::string& key, const double deflt)
{
    return d.found(key) ? d.getScalar(key) : deflt;
}

bool switchOr(const Dictionary& d, const std::string& path, const std::string& key, const bool deflt)
{
    return d.found(key) ? parseSwitch(path, key, d.getWord(key)) : deflt;
}

// OpenFOAM 純量的 max(a, b)：(b < a) ? a : b（primitives/ints/int/int.H MAXMIN）
__device__ __forceinline__ double hmax(const double a, const double b)
{
    return (b < a) ? a : b;
}

// 每格的 sqrt(gamma/psi)、mag(U)
__global__ void kCellSpeeds
(
    const std::int32_t n,
    const ThermoParams P,
    const double* __restrict__ p,
    const double* __restrict__ T,
    const double* __restrict__ psi,
    const CView<vec3> U,
    double* __restrict__ cs,
    double* __restrict__ mU
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < n)
    {
        cs[c] = ::sqrt(thermo::gamma(P, p[c], T[c])/psi[c]);
        mU[c] = mag(U[c]);
    }
}

// 內部面：frdt = nonOrthDeltaCoeffs*(interpolate(cs) + interpolate(mU))
__global__ void kFrdt
(
    const std::int32_t nIF,
    const std::int32_t* __restrict__ own,
    const std::int32_t* __restrict__ nei,
    const double* __restrict__ w,
    const double* __restrict__ ndc,
    const double* __restrict__ cs,
    const double* __restrict__ mU,
    double* __restrict__ frdt
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f < nIF)
    {
        const std::int32_t P = own[f], N = nei[f];
        const double a = w[f]*(cs[P] - cs[N]) + cs[N];
        const double b = w[f]*(mU[P] - mU[N]) + mU[N];
        frdt[f] = ndc[f]*(a + b);
    }
}

// 每格：內部面的 max，再依 patch 次序取 wall 面的值，最後除以 pseudoCoField
__global__ void kRPseudoDeltaT
(
    const std::int32_t nC,
    const std::int32_t nIF,
    const std::int32_t* __restrict__ off,
    const std::int32_t* __restrict__ faces,
    const std::int8_t* __restrict__ isWallFace,     // 索引 f - nIF
    const double* __restrict__ ndc,
    const double* __restrict__ frdt,
    const double* __restrict__ cs,
    const double* __restrict__ mU,
    const double* __restrict__ damping,
    const double pseudoCoNum,
    double* __restrict__ r
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC)
    {
        return;
    }
    double v = 0.0;
    // 面依編號遞增：內部面在前，邊界面依 patch 次序
    for (std::int32_t q = off[c]; q < off[c + 1]; ++q)
    {
        const std::int32_t f = faces[q];
        if (f < nIF)
        {
            v = hmax(v, frdt[f]);
        }
    }
    for (std::int32_t q = off[c]; q < off[c + 1]; ++q)
    {
        const std::int32_t f = faces[q];
        if (f >= nIF && isWallFace[f - nIF])
        {
            const double pLambda = 0.5*ndc[f]*(cs[c] + mU[c]);
            v = hmax(v, pLambda);
        }
    }
    const double pseudoCoField = damping[c]*pseudoCoNum;
    r[c] = v/pseudoCoField;
}

// ddtCoeff = (rPseudoDeltaT*V)/V
__global__ void kDdtCoeff
(
    const std::int32_t n,
    const double* __restrict__ r,
    const double* __restrict__ V,
    double* __restrict__ ddt
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c < n)
    {
        const double diag = r[c]*V[c];
        ddt[c] = diag/V[c];
    }
}

__global__ void kFill(const std::int32_t n, const double v, double* __restrict__ a)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[i] = v;
    }
}

} // namespace


PseudoTimeControls readPseudoTimeControls(const std::string& path)
{
    FoamTokenizer t(path);
    const Dictionary root = Dictionary::parse(t, path, false);
    const Dictionary::Entry* e = root.find("pseudoTime");
    const Dictionary empty;
    const Dictionary& d = (e && e->dict) ? *e->dict : empty;

    PseudoTimeControls c;
    c.pseudoCoNum = scalarOr(d, "pseudoCoNum", 1.0);
    c.localTimestepping = switchOr(d, path, "localTimestepping", true);
    if (c.localTimestepping)
    {
        c.localTimesteppingBounding = switchOr(d, path, "localTimesteppingBounding", true);
        double lb = scalarOr(d, "localTimesteppingLowerBound", 0.95);
        lb = (lb > 0 ? lb : 0.0);
        lb = (lb < 0.999 ? lb : 0.999);
        c.localTimesteppingLowerBound = lb;
        double ub = scalarOr(d, "localTimesteppingUpperBound", 1.05);
        c.localTimesteppingUpperBound = (ub > 1.001 ? ub : 1.001);
        const double bd = scalarOr(d, "boundingDecrease", 0.5);
        c.boundingDecrease = (bd < 1.0 ? bd : 1.0);
        const double bi = scalarOr(d, "boundingIncrease", 2.0);
        c.boundingIncrease = (bi > 1.0 ? bi : 1.0);
    }
    else
    {
        c.localTimesteppingBounding = false;
    }
    c.resetPseudo = switchOr(d, path, "resetPseudo", false);
    c.pseudoCoNumMin = scalarOr(d, "pseudoCoNumMin", 0.1);
    c.pseudoCoNumMax = scalarOr(d, "pseudoCoNumMax", 25.0);
    c.pseudoCoNumMaxIncreaseFactor = scalarOr(d, "pseudoCoNumMaxIncreaseFactor", 1.2);
    c.pseudoCoNumMinDecreaseFactor = scalarOr(d, "pseudoCoNumMinDecreaseFactor", 0.1);
    c.rhoMin = scalarOr(d, "rhoMin", -1.0e+15);
    c.TMin = scalarOr(d, "TMin", 3.0e-8);
    c.TMax = scalarOr(d, "TMax", 1.0e+15);
    return c;
}


PseudoTimeState initPseudoTime(const DeviceMesh& dm, const PseudoTimeControls& c)
{
    if (!c.localTimestepping)
    {
        throw std::runtime_error("global pseudo time stepping is not supported (localTimestepping only)");
    }
    PseudoTimeState pt;
    pt.pseudoCoNum = c.pseudoCoNum;
    // pseudoCoDamping = pseudoCoField/pseudoCoNum，pseudoCoField = pseudoCoNum
    const double damping = c.pseudoCoNum/c.pseudoCoNum;
    pt.pseudoCoDamping = makeVolField<double>(dm, "pseudoCoDamping", Bc::calculated);
    if (dm.nCells)
    {
        kFill<<<nBlocks(dm.nCells), BS>>>(dm.nCells, damping, pt.pseudoCoDamping.in.view().p[0]);
        GNC_CUDA_LAUNCH_CHECK();
    }
    // 邊界：pseudoCoField 的邊界值為 pseudoCoNum（== 指定，constraint patch 也是）
    const std::int32_t nB = dm.nFvBoundaryFaces();
    if (nB)
    {
        kFill<<<nBlocks(nB), BS>>>(nB, damping, pt.pseudoCoDamping.bnd.view().p[0]);
        GNC_CUDA_LAUNCH_CHECK();
    }
    pt.rPseudoDeltaT = makeVolField<double>(dm, "rPseudoDeltaT", Bc::zeroGradient);
    pt.ddtCoeff.allocate(std::size_t(dm.nCells));
    return pt;
}


void setPseudoDeltaT
(
    const DeviceMesh& dm,
    const FvSchemes& schemes,
    const HisaState& s,
    PseudoTimeState& pt
)
{
    for (const char* name : {"interpolate(sqrt((gamma|thermo:psi)))", "interpolate(mag(U))"})
    {
        const std::string sch = schemes.interpolation(name);
        if (sch != "linear")
        {
            throw std::runtime_error
            (
                std::string(name) + " '" + sch + "' is not supported (linear)"
            );
        }
    }

    const std::int32_t nC = dm.nCells;
    const std::int32_t nIF = dm.nInternalFaces;
    const PsiThermoFields& th = s.thermo;

    DField<double> cs(static_cast<std::size_t>(nC)), mU(static_cast<std::size_t>(nC));
    DField<double> frdt(static_cast<std::size_t>(nIF));

    // 每個邊界面是否屬於 type wall 的 patch
    std::vector<std::int8_t> wallH(std::size_t(dm.nFaces - nIF), 0);
    for (const Patch& p : dm.patches)
    {
        if (p.type == "wall")
        {
            for (std::int32_t i = 0; i < p.size; ++i)
            {
                wallH[std::size_t(p.start + i - nIF)] = 1;
            }
        }
    }
    DeviceArray<std::int8_t> isWall(wallH.size());
    isWall.upload(wallH);

    if (nC)
    {
        kCellSpeeds<<<nBlocks(nC), BS>>>
        (
            nC, th.params, th.p.in.cview().p[0], th.T.in.cview().p[0],
            th.psi.in.cview().p[0], s.U.in.cview(), cs.view().p[0], mU.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (nIF)
    {
        kFrdt<<<nBlocks(nIF), BS>>>
        (
            nIF, dm.owner.data(), dm.neighbour.data(), dm.weights.data(),
            dm.nonOrthDeltaCoeffs.data(), cs.cview().p[0], mU.cview().p[0], frdt.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (nC)
    {
        kRPseudoDeltaT<<<nBlocks(nC), BS>>>
        (
            nC, nIF, dm.cellFaceOffsets.data(), dm.cellFaces.data(), isWall.data(),
            dm.nonOrthDeltaCoeffs.data(), frdt.cview().p[0], cs.cview().p[0], mU.cview().p[0],
            pt.pseudoCoDamping.in.cview().p[0], pt.pseudoCoNum, pt.rPseudoDeltaT.in.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
        kDdtCoeff<<<nBlocks(nC), BS>>>
        (
            nC, pt.rPseudoDeltaT.in.cview().p[0], dm.V.data(), pt.ddtCoeff.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    evaluateBCs(dm, pt.rPseudoDeltaT, nullptr);
    GNC_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace gnc
