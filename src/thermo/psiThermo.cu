// GPU-Native-CFD，GPL-3.0-or-later。見 psiThermo.cuh。

#include "thermo/psiThermo.cuh"

#include "fields/boundaryConditions.cuh"
#include "io/dictionary.hpp"
#include "io/foamTokenizer.hpp"

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

double numberOf(const std::string& text, const std::string& what)
{
    double v = 0;
    if (!parseScalar(text, v))
    {
        throw std::runtime_error(what + ": invalid number " + text);
    }
    return v;
}


// heThermo 的 he 邊界條件型別（basicThermo::heBoundaryTypes，basicThermo.C:198）
Bc heBoundaryType(const Bc Tbc)
{
    if (isFixedValueFamily(Tbc))
    {
        return Bc::fixedEnergy;
    }
    if (isZeroGradientFamily(Tbc) || isFixedGradientFamily(Tbc))
    {
        return Bc::gradientEnergy;
    }
    if (isMixedFamily(Tbc))
    {
        return Bc::mixedEnergy;
    }
    return Tbc;
}


struct BRange
{
    std::int32_t start;
    std::int32_t size;
    const std::int32_t* fvbFaces;
    const std::int32_t* owner;
};


__global__ void kHeCells
(
    const std::int32_t n,
    const ThermoParams P,
    const double* __restrict__ p,
    const double* __restrict__ T,
    double* __restrict__ he
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        he[i] = thermo::Es(P, p[i], T[i]);
    }
}


// heBoundaryCorrection：gradient／refGrad = fvPatchField::snGrad() = dc*(he_b - he_c)
__global__ void kHeSnGrad
(
    const BRange r,
    const double* __restrict__ dc,
    const double* __restrict__ heIn,
    const double* __restrict__ heB,
    double* __restrict__ out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < r.size)
    {
        const std::int32_t k = r.start + i;
        const std::int32_t f = r.fvbFaces[k];
        out[i] = dc[f]*(heB[k] - heIn[r.owner[f]]);
    }
}


// hePsiThermo::calculate 的格迴圈（hePsiThermo.C）
__global__ void kCalcCells
(
    const std::int32_t n,
    const ThermoParams P,
    const double* __restrict__ he,
    const double* __restrict__ p,
    double* __restrict__ T,
    double* __restrict__ psi,
    double* __restrict__ mu,
    double* __restrict__ alpha,
    int* __restrict__ fail
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    int f = 0;
    const double Ti = thermo::THE(P, he[i], p[i], T[i], &f);
    if (f)
    {
        *fail = 1;
    }
    T[i] = Ti;
    psi[i] = thermo::psi(P, p[i], Ti);
    mu[i] = thermo::mu(P, p[i], Ti);
    alpha[i] = thermo::alphah(P, p[i], Ti);
}


// calculate 的邊界：T 的 patch fixesValue 時由 T 求 he，否則由 he 求 T
__global__ void kCalcBoundary
(
    const std::int32_t start,
    const std::int32_t size,
    const bool fixes,
    const ThermoParams P,
    const double* __restrict__ p,
    double* __restrict__ T,
    double* __restrict__ he,
    double* __restrict__ psi,
    double* __restrict__ mu,
    double* __restrict__ alpha,
    int* __restrict__ fail
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= size)
    {
        return;
    }
    const std::int32_t k = start + i;
    if (fixes)
    {
        he[k] = thermo::Es(P, p[k], T[k]);
    }
    else
    {
        int f = 0;
        T[k] = thermo::THE(P, he[k], p[k], T[k], &f);
        if (f)
        {
            *fail = 1;
        }
    }
    psi[k] = thermo::psi(P, p[k], T[k]);
    mu[k] = thermo::mu(P, p[k], T[k]);
    alpha[k] = thermo::alphah(P, p[k], T[k]);
}


VolField<double> calculatedField(const DeviceMesh& dm, const std::string& name)
{
    return makeVolField<double>(dm, name, Bc::calculated);
}

} // namespace


FoamConstants openfoamConstants()
{
    // physicoChemical::NA 的預設值（fundamentalConstants.C:121），k 與 Tstd 取自
    // etc/controlDict（DimensionedConstants/SICoeffs）
    const double NA = 6.0221417930e+23;
    const double k = numberOf("1.38065e-23", "k");
    FoamConstants c;
    c.RR = 1e3*(NA*k);
    c.Tstd = numberOf("298.15", "Tstd");
    return c;
}


ThermoParams readThermoParams(const std::string& path, const FoamConstants& c)
{
    FoamTokenizer t(path);
    const Dictionary d = Dictionary::parse(t, path, false);

    const Dictionary& tt = d.subDict("thermoType");
    const auto expect = [&](const char* key, const char* value)
    {
        const std::string v = tt.getWord(key);
        if (v != value)
        {
            throw std::runtime_error
            (
                path + ": thermoType " + key + " " + v + " is not supported (expected "
              + value + ")"
            );
        }
    };
    expect("type", "hePsiThermo");
    expect("mixture", "pureMixture");
    expect("thermo", "hConst");
    expect("equationOfState", "perfectGas");
    expect("specie", "specie");
    expect("energy", "sensibleInternalEnergy");
    const std::string transport = tt.getWord("transport");

    const Dictionary& mix = d.subDict("mixture");
    ThermoParams P;
    P.RR = c.RR;
    P.Tstd = c.Tstd;
    P.W = mix.subDict("specie").getScalar("molWeight");
    P.R = P.RR/P.W;

    const Dictionary& td = mix.subDict("thermodynamics");
    P.Cp = td.getScalar("Cp");
    P.Hf = td.getScalar("Hf");
    // hConstThermo.C:40–41：Tref 預設 Tstd，Href 預設 0
    P.Tref = td.found("Tref") ? td.getScalar("Tref") : c.Tstd;
    P.Hsref = td.found("Href") ? td.getScalar("Href") : 0.0;

    const Dictionary& tr = mix.subDict("transport");
    if (transport == "const")
    {
        P.transport = 0;
        P.mu = tr.getScalar("mu");
        P.rPr = 1.0/tr.getScalar("Pr");
    }
    else if (transport == "sutherland")
    {
        P.transport = 1;
        P.As = tr.getScalar("As");
        P.Ts = tr.getScalar("Ts");
    }
    else
    {
        throw std::runtime_error(path + ": transport " + transport + " is not supported");
    }
    return P;
}


PsiThermoFields constructPsiThermo
(
    const DeviceMesh& dm,
    const ThermoParams& params,
    VolField<double>&& p,
    VolField<double>&& T
)
{
    PsiThermoFields th;
    th.params = params;
    th.p = std::move(p);
    th.T = std::move(T);

    // he 的邊界條件型別依 T（heBoundaryTypes）
    th.he = makeVolField<double>(dm, "e", Bc::calculated);
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        th.he.bc[pi].type = heBoundaryType(th.T.bc[pi].type);
    }

    // heThermo::init：he = HE(p, T)（格與每個 patch 的面）
    const std::int32_t nC = dm.nCells;
    const std::int32_t nB = dm.nFvBoundaryFaces();
    if (nC)
    {
        kHeCells<<<nBlocks(nC), BS>>>
        (
            nC, params, th.p.in.cview().p[0], th.T.in.cview().p[0], th.he.in.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (nB)
    {
        kHeCells<<<nBlocks(nB), BS>>>
        (
            nB, params, th.p.bnd.cview().p[0], th.T.bnd.cview().p[0], th.he.bnd.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }

    // heBoundaryCorrection：gradientEnergy 的 gradient、mixedEnergy 的 refGrad
    // 設為 snGrad。mixedEnergy 的 refValue、valueFraction 在 updateCoeffs 中
    // 重新計算（N3c）；這裡先設為 0。
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        PatchBC<double>& bc = th.he.bc[pi];
        const std::int32_t n = dm.fvPatchSize[pi];
        if (bc.type != Bc::gradientEnergy && bc.type != Bc::mixedEnergy)
        {
            continue;
        }
        DField<double> g(static_cast<std::size_t>(n));
        if (n)
        {
            const BRange r{dm.fvPatchStart[pi], n, dm.fvBoundaryFaces.data(), dm.owner.data()};
            kHeSnGrad<<<nBlocks(n), BS>>>
            (
                r, dm.deltaCoeffs.data(), th.he.in.cview().p[0], th.he.bnd.cview().p[0],
                g.view().p[0]
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
        if (bc.type == Bc::gradientEnergy)
        {
            bc.gradient = std::move(g);
        }
        else
        {
            bc.refGrad = std::move(g);
            bc.refValue.allocate(std::size_t(n));
            bc.refValue.zero();
            bc.valueFraction.allocate(std::size_t(n));
            bc.valueFraction.zero();
        }
    }

    th.psi = calculatedField(dm, "thermo:psi");
    th.mu = calculatedField(dm, "thermo:mu");
    th.alpha = calculatedField(dm, "thermo:alpha");

    // hePsiThermo 建構子：calculate()
    calculatePsiThermo(dm, th);
    return th;
}


void calculatePsiThermo(const DeviceMesh& dm, PsiThermoFields& th)
{
    DeviceArray<int> fail(1);
    fail.zero();
    const ThermoParams& P = th.params;
    const std::int32_t nC = dm.nCells;
    if (nC)
    {
        kCalcCells<<<nBlocks(nC), BS>>>
        (
            nC, P, th.he.in.cview().p[0], th.p.in.cview().p[0], th.T.in.view().p[0],
            th.psi.in.view().p[0], th.mu.in.view().p[0], th.alpha.in.view().p[0],
            fail.data()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const std::int32_t n = dm.fvPatchSize[pi];
        if (!n)
        {
            continue;
        }
        kCalcBoundary<<<nBlocks(n), BS>>>
        (
            dm.fvPatchStart[pi], n, fixesValue(th.T.bc[pi].type), P,
            th.p.bnd.cview().p[0], th.T.bnd.view().p[0], th.he.bnd.view().p[0],
            th.psi.bnd.view().p[0], th.mu.bnd.view().p[0], th.alpha.bnd.view().p[0],
            fail.data()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (fail.download()[0])
    {
        throw std::runtime_error
        (
            "thermo: temperature from energy did not converge (or T0 < 0)"
        );
    }
}


VolField<double> heOf
(
    const DeviceMesh& dm,
    const ThermoParams& params,
    const VolField<double>& p,
    const VolField<double>& T,
    const std::string& name
)
{
    VolField<double> he = calculatedField(dm, name);
    const std::int32_t nC = dm.nCells;
    const std::int32_t nB = dm.nFvBoundaryFaces();
    if (nC)
    {
        kHeCells<<<nBlocks(nC), BS>>>
        (
            nC, params, p.in.cview().p[0], T.in.cview().p[0], he.in.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (nB)
    {
        kHeCells<<<nBlocks(nB), BS>>>
        (
            nB, params, p.bnd.cview().p[0], T.bnd.cview().p[0], he.bnd.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    return he;
}

} // namespace gnc
