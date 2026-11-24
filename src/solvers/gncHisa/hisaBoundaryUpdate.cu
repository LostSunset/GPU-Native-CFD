// GPU-Native-CFD，GPL-3.0-or-later。見 hisaBoundaryUpdate.cuh。

#include "solvers/gncHisa/hisaBoundaryUpdate.cuh"

#include <cmath>
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


// characteristicFarfieldVelocityFvPatchVectorField::updateCoeffs
__global__ void kCharFarfieldVelocity
(
    const std::int32_t start,
    const std::int32_t n,
    const std::int32_t* __restrict__ fvbFaces,
    const std::int32_t* __restrict__ owner,
    const CView<vec3> Sf,
    const double* __restrict__ magSf,
    const ThermoParams P,
    const double* __restrict__ pIn,
    const double* __restrict__ pB,
    const double* __restrict__ TIn,
    const double* __restrict__ TB,
    const double* __restrict__ psiIn,
    const double* __restrict__ psiB,
    const double* __restrict__ rhoIn,
    const double* __restrict__ rhoB,
    const double* __restrict__ phiB,
    const CView<vec3> UIn,
    const vec3 URef,
    const double pRef,
    const DView<vec3> refValue,
    const DView<vec3> refGrad,
    double* __restrict__ valueFraction,
    const DView<tens> UCoeff,
    const DView<vec3> pCoeff,
    const DView<vec3> TCoeff
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const std::int32_t k = start + i;
    const std::int32_t f = fvbFaces[k];
    const std::int32_t c = owner[f];

    double vf = 1.0;
    refGrad.set(i, vec3{0.0, 0.0, 0.0});
    // valueInternalCoeffs(T)：vector::zero
    TCoeff.set(i, vec3{0.0, 0.0, 0.0});

    const vec3 U = UIn[c];
    const double p = pIn[c];
    const vec3 pn = Sf[f]/magSf[f];

    const double pc = ::sqrt(thermo::gamma(P, pB[k], TB[k])/psiB[k]);
    const double pM = phiB[k]/((rhoB[k]*magSf[f])*pc);

    const double cO = ::sqrt(thermo::gamma(P, pIn[c], TIn[c])/psiIn[c]);
    const double rhoO = rhoIn[c];

    const tens I{1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    if (pM <= -1.0)
    {
        refValue.set(i, URef);
        UCoeff.set(i, tens{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
        pCoeff.set(i, vec3{0.0, 0.0, 0.0});
    }
    else if (pM >= 1.0)
    {
        vf = 0.0;
        UCoeff.set(i, I);
        pCoeff.set(i, vec3{0.0, 0.0, 0.0});
    }
    else if (pM <= 0.0)
    {
        const double pp = 0.5*(pRef + p - (rhoO*cO)*dot(URef - U, pn));
        refValue.set(i, URef - pn*(pRef - pp)/(rhoO*cO));
        UCoeff.set(i, outer((-0.5)*pn, pn));
        pCoeff.set(i, (0.5*pn)/(rhoO*cO));
    }
    else
    {
        vf = 0.5;
        refValue.set
        (
            i,
            dot(URef, pn)*pn - pn*(pRef - p)/(rhoO*cO) + (U - dot(U, pn)*pn)
        );
        UCoeff.set(i, I - outer(0.5*pn, pn));
        pCoeff.set(i, (0.5*pn)/(rhoO*cO));
    }
    valueFraction[i] = vf;
}


// characteristicFarfieldPressureFvPatchScalarField::updateCoeffs
__global__ void kCharFarfieldPressure
(
    const std::int32_t start,
    const std::int32_t n,
    const std::int32_t* __restrict__ fvbFaces,
    const std::int32_t* __restrict__ owner,
    const CView<vec3> Sf,
    const double* __restrict__ magSf,
    const ThermoParams P,
    const double* __restrict__ pIn,
    const double* __restrict__ pB,
    const double* __restrict__ TIn,
    const double* __restrict__ TB,
    const double* __restrict__ psiIn,
    const double* __restrict__ psiB,
    const double* __restrict__ rhoIn,
    const double* __restrict__ rhoB,
    const double* __restrict__ phiB,
    const CView<vec3> UIn,
    const vec3 URef,
    const double pRef,
    double* __restrict__ refValue,
    double* __restrict__ refGrad,
    double* __restrict__ valueFraction,
    double* __restrict__ pCoeff,
    const DView<vec3> UCoeff,
    double* __restrict__ TCoeff
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const std::int32_t k = start + i;
    const std::int32_t f = fvbFaces[k];
    const std::int32_t c = owner[f];

    double vf = 1.0;
    refGrad[i] = 0.0;
    TCoeff[i] = 0.0;

    const vec3 U = UIn[c];
    const vec3 pn = Sf[f]/magSf[f];
    const double pc = ::sqrt(thermo::gamma(P, pB[k], TB[k])/psiB[k]);
    const double pM = phiB[k]/((rhoB[k]*magSf[f])*pc);
    const double cO = ::sqrt(thermo::gamma(P, pIn[c], TIn[c])/psiIn[c]);
    const double rhoO = rhoIn[c];

    if (pM <= -1.0)
    {
        refValue[i] = pRef;
        pCoeff[i] = 0.0;
        UCoeff.set(i, vec3{0.0, 0.0, 0.0});
    }
    else if (pM >= 1.0)
    {
        vf = 0.0;
        pCoeff[i] = 1.0;
        UCoeff.set(i, vec3{0.0, 0.0, 0.0});
    }
    else
    {
        // 次音速流入與流出的公式相同
        vf = 0.5;
        refValue[i] = pRef - (rhoO*cO)*dot(URef - U, pn);
        pCoeff[i] = 0.5;
        UCoeff.set(i, (0.5*(rhoO*cO))*pn);
    }
    valueFraction[i] = vf;
}


// characteristicWallPressureFvPatchScalarField::updateCoeffs：UCoeff_ 與
// gradient（限制 p 不會變負）；valueInternalCoeffs：p 1、T 0、U UCoeff_*nf
__global__ void kCharWallPressure
(
    const std::int32_t start,
    const std::int32_t n,
    const std::int32_t* __restrict__ fvbFaces,
    const std::int32_t* __restrict__ owner,
    const CView<vec3> Sf,
    const double* __restrict__ magSf,
    const double* __restrict__ dc,
    const ThermoParams P,
    const double* __restrict__ pIn,
    const double* __restrict__ TIn,
    const double* __restrict__ psiIn,
    const CView<vec3> UIn,
    const CView<vec3> UB,
    const bool stabiliseSwapped,
    double* __restrict__ gradient,
    double* __restrict__ pCoeff,
    const DView<vec3> UCoeff,
    double* __restrict__ TCoeff
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const std::int32_t k = start + i;
    const std::int32_t f = fvbFaces[k];
    const std::int32_t c = owner[f];

    const double pif = pIn[c];
    const double psiif = psiIn[c];
    const double gammaif = thermo::gamma(P, pIn[c], TIn[c]);
    const double rhoif = pIn[c]*psiIn[c];
    const vec3 nf = Sf[f]/magSf[f];

    const double cif = ::sqrt(gammaif/psiif);
    double UC = rhoif*cif;
    double g = ((rhoif*cif)*dot(UIn[c] - UB[k], nf))*dc[f];
    const double minGrad = (pif - SMALL)*dc[f];
    // 場的 stabilise(gradient, SMALL)；FoamCompat::stabiliseSwappedArgs 時照
    // 參考端的 stabilise(SMALL, gradient) = SMALL + gradient
    const double st = stabiliseSwapped ? SMALL + g : ((g >= 0) ? g + SMALL : g - SMALL);
    const double q = minGrad/st;
    const double mn = (q < 1.0) ? q : 1.0;
    const double limit = (0.0 < mn) ? mn : 0.0;
    UC *= limit;
    g *= limit;

    gradient[i] = g;
    pCoeff[i] = 1.0;
    UCoeff.set(i, UC*nf);
    TCoeff[i] = 0.0;
}


// characteristicWallTemperatureFvPatchScalarField::updateCoeffs（fixedValue；
// T_b = Tif*coeff，coeff 夾在 [0.9, 1.1]）
__global__ void kCharWallTemperature
(
    const std::int32_t start,
    const std::int32_t n,
    const std::int32_t* __restrict__ fvbFaces,
    const std::int32_t* __restrict__ owner,
    const CView<vec3> Sf,
    const double* __restrict__ magSf,
    const ThermoParams P,
    const double* __restrict__ pIn,
    const double* __restrict__ TIn,
    const double* __restrict__ psiIn,
    const CView<vec3> UIn,
    const CView<vec3> UB,
    double* __restrict__ TB
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const std::int32_t k = start + i;
    const std::int32_t f = fvbFaces[k];
    const std::int32_t c = owner[f];

    const double gammaif = thermo::gamma(P, pIn[c], TIn[c]);
    const double cif = ::sqrt(gammaif/psiIn[c]);
    const vec3 nf = Sf[f]/magSf[f];
    const double un = dot(UIn[c] - UB[k], nf);
    double coeff = (cif + gammaif*un)/(cif + un);
    if (coeff < 0.9)
    {
        coeff = 0.9;
    }
    else if (coeff > 1.1)
    {
        coeff = 1.1;
    }
    TB[k] = TIn[c]*coeff;
}


// characteristicFarfieldTemperatureFvPatchScalarField::updateCoeffs；evaluate 時
// 接著 mixed 的 evaluate：T_b = lerp(pif + refGrad/dc, refValue, vf)
__global__ void kCharFarfieldTemperature
(
    const std::int32_t start,
    const std::int32_t n,
    const std::int32_t* __restrict__ fvbFaces,
    const std::int32_t* __restrict__ owner,
    const CView<vec3> Sf,
    const double* __restrict__ magSf,
    const double* __restrict__ dc,
    const ThermoParams P,
    const double* __restrict__ pIn,
    const double* __restrict__ pB,
    const double* __restrict__ TIn,
    const double* __restrict__ psiIn,
    const double* __restrict__ psiB,
    const double* __restrict__ rhoIn,
    const double* __restrict__ rhoB,
    const double* __restrict__ phiB,
    const CView<vec3> UIn,
    const vec3 URef,
    const double pRef,
    const double TRef,
    double* __restrict__ refValue,
    double* __restrict__ refGrad,
    double* __restrict__ valueFraction,
    double* __restrict__ TCoeff,
    double* __restrict__ pCoeff,
    const DView<vec3> UCoeff,
    const bool evaluate,
    double* __restrict__ TB
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const std::int32_t k = start + i;
    const std::int32_t f = fvbFaces[k];
    const std::int32_t c = owner[f];

    // pB、TB 為更新前的邊界值（gamma 的邊界）
    double vf = 1.0;
    double rg = 0.0;
    double rv = refValue[i];

    const double T = TIn[c];
    const double p = pIn[c];
    const vec3 U = UIn[c];
    const vec3 pn = Sf[f]/magSf[f];
    const double pc = ::sqrt(thermo::gamma(P, pB[k], TB[k])/psiB[k]);
    const double pM = phiB[k]/((rhoB[k]*magSf[f])*pc);
    const double cO = ::sqrt(thermo::gamma(P, pIn[c], TIn[c])/psiIn[c]);
    const double rhoO = rhoIn[c];
    const double Reff = p/(rhoO*T);
    const double rhoInf = pRef/(Reff*TRef);
    const double d = dc[f];

    if (pM <= -1.0)
    {
        rv = TRef;
        TCoeff[i] = 0.0;
        UCoeff.set(i, vec3{0.0, 0.0, 0.0});
        pCoeff[i] = 0.0;
    }
    else if (pM >= 1.0)
    {
        vf = 0.0;
        rv = TRef;
        TCoeff[i] = 1.0;
        UCoeff.set(i, vec3{0.0, 0.0, 0.0});
        pCoeff[i] = 0.0;
    }
    else if (pM <= 0.0)
    {
        const double pp = 0.5*(pRef + p - (rhoO*cO)*dot(URef - U, pn));
        const double prho = rhoInf - (pRef - pp)/(cO*cO);
        rv = pp/(Reff*prho);
        TCoeff[i] = 0.0;
        UCoeff.set(i, ((((0.5*rhoO)*cO)*pn)/(Reff*prho))*(1.0 - pp/(prho*(cO*cO))));
        pCoeff[i] = (0.5/(Reff*prho))*(1.0 - pp/(prho*(cO*cO)));
    }
    else
    {
        const double pp = 0.5*(pRef + p - (rhoO*cO)*dot(URef - U, pn));
        const double prho = p/(Reff*T) - (p - pp)/(cO*cO);
        vf = 1.0 - pp/p;
        rg = ((p/(Reff*prho) - T) - (TRef*vf)/(1.0 - vf))*d;
        rv = TRef;
        const double t = pp/((Reff*prho)*T);
        TCoeff[i] = t*t;
        UCoeff.set(i, (((-rhoO)*pn)*p)/((Reff*cO)*(prho*prho)));
        pCoeff[i] = ((-0.5)*p)/((Reff*(prho*prho))*(cO*cO));
    }
    refValue[i] = rv;
    refGrad[i] = rg;
    valueFraction[i] = vf;
    if (evaluate)
    {
        TB[k] = lerp(T + rg/d, rv, vf);
    }
}


// mixed／inletOutlet 的 evaluate（inletOutlet 先設 valueFraction = neg(phi)）
__global__ void kEvalMixedScalar
(
    const std::int32_t start,
    const std::int32_t n,
    const std::int32_t* __restrict__ fvbFaces,
    const std::int32_t* __restrict__ owner,
    const double* __restrict__ dc,
    const bool inletOutlet,
    const double* __restrict__ phiB,
    const double* __restrict__ in,
    const double* __restrict__ refValue,
    const double* __restrict__ refGrad,
    double* __restrict__ valueFraction,
    double* __restrict__ bnd
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const std::int32_t k = start + i;
    const std::int32_t f = fvbFaces[k];
    if (inletOutlet)
    {
        valueFraction[i] = (phiB[k] < 0) ? 1.0 : 0.0;
    }
    bnd[k] = lerp(in[owner[f]] + refGrad[i]/dc[f], refValue[i], valueFraction[i]);
}


// zeroGradient（gradient 為 nullptr）與 fixedGradient 的 evaluate
__global__ void kEvalGradientScalar
(
    const std::int32_t start,
    const std::int32_t n,
    const std::int32_t* __restrict__ fvbFaces,
    const std::int32_t* __restrict__ owner,
    const double* __restrict__ dc,
    const double* __restrict__ in,
    const double* __restrict__ gradient,
    double* __restrict__ bnd
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const std::int32_t k = start + i;
    const std::int32_t f = fvbFaces[k];
    const double pif = in[owner[f]];
    bnd[k] = gradient ? pif + gradient[i]/dc[f] : pif;
}


// fixedEnergy 的 updateCoeffs：he_b == HE(p_b, T_b)
__global__ void kFixedEnergy
(
    const std::int32_t start,
    const std::int32_t n,
    const ThermoParams P,
    const double* __restrict__ pB,
    const double* __restrict__ TB,
    double* __restrict__ heB
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const std::int32_t k = start + i;
        heB[k] = thermo::Es(P, pB[k], TB[k]);
    }
}


// inletOutlet：valueFraction = neg(phi)
__global__ void kInletOutletFraction
(
    const std::int32_t start,
    const std::int32_t n,
    const double* __restrict__ phiB,
    double* __restrict__ valueFraction
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        valueFraction[i] = (phiB[start + i] < 0) ? 1.0 : 0.0;
    }
}

} // namespace


void updateCoeffsU(const DeviceMesh& dm, HisaState& s, const HisaFlux& flux)
{
    const PsiThermoFields& th = s.thermo;
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const std::int32_t start = dm.fvPatchStart[pi];
        const std::int32_t n = dm.fvPatchSize[pi];
        PatchBC<vec3>& bc = s.U.bc[pi];
        if (n == 0)
        {
            continue;
        }
        switch (bc.type)
        {
            case Bc::characteristicFarfieldVelocity:
                bc.UCoeff.allocate(std::size_t(n));
                bc.pCoeff.allocate(std::size_t(n));
                bc.TCoeff.allocate(std::size_t(n));
                kCharFarfieldVelocity<<<nBlocks(n), BS>>>
                (
                    start, n, dm.fvBoundaryFaces.data(), dm.owner.data(), meshVec(dm.Sf),
                    dm.magSf.data(), th.params,
                    th.p.in.cview().p[0], th.p.bnd.cview().p[0],
                    th.T.in.cview().p[0], th.T.bnd.cview().p[0],
                    th.psi.in.cview().p[0], th.psi.bnd.cview().p[0],
                    s.rho.in.cview().p[0], s.rho.bnd.cview().p[0],
                    flux.phi.bnd.cview().p[0], s.U.in.cview(), bc.URef, bc.pRef,
                    bc.refValue.view(), bc.refGrad.view(), bc.valueFraction.data(),
                    bc.UCoeff.view(), bc.pCoeff.view(), bc.TCoeff.view()
                );
                GNC_CUDA_LAUNCH_CHECK();
                break;
            case Bc::inletOutlet:
                if (bc.phiName != "phi")
                {
                    throw std::runtime_error
                    (
                        "field U patch " + dm.patches[pi].name + ": inletOutlet with flux "
                      + bc.phiName + " is not supported (phi only)"
                    );
                }
                kInletOutletFraction<<<nBlocks(n), BS>>>
                (
                    start, n, flux.phi.bnd.cview().p[0], bc.valueFraction.data()
                );
                GNC_CUDA_LAUNCH_CHECK();
                break;
            case Bc::fixedValue:
            case Bc::boundaryCorrectedFixedValue:
            case Bc::zeroGradient:
            case Bc::calculated:
            case Bc::extrapolatedCalculated:
            case Bc::symmetryPlane:
            case Bc::symmetry:
            case Bc::slip:
            case Bc::empty:
                break;
            default:
                throw std::runtime_error
                (
                    "field U patch " + dm.patches[pi].name + ": updateCoeffs of "
                  + bcName(bc.type) + " is not implemented yet"
                );
        }
    }
    GNC_CUDA_CHECK(cudaDeviceSynchronize());
}

void updateEnergyT(const DeviceMesh& dm, HisaState& s, const HisaFlux& flux)
{
    PsiThermoFields& th = s.thermo;
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const std::int32_t start = dm.fvPatchStart[pi];
        const std::int32_t n = dm.fvPatchSize[pi];
        PatchBC<double>& bc = th.T.bc[pi];
        if (n == 0)
        {
            continue;
        }
        const std::string& pname = dm.patches[pi].name;
        switch (bc.type)
        {
            case Bc::characteristicWallTemperature:
                kCharWallTemperature<<<nBlocks(n), BS>>>
                (
                    start, n, dm.fvBoundaryFaces.data(), dm.owner.data(), meshVec(dm.Sf),
                    dm.magSf.data(), th.params, th.p.in.cview().p[0], th.T.in.cview().p[0],
                    th.psi.in.cview().p[0], s.U.in.cview(), s.U.bnd.cview(),
                    th.T.bnd.view().p[0]
                );
                break;
            case Bc::characteristicFarfieldTemperature:
                bc.UCoeff.allocate(std::size_t(n));
                bc.pCoeff.allocate(std::size_t(n));
                bc.TCoeff.allocate(std::size_t(n));
                // 每個 thread 先讀自己的 T_b（gamma 的邊界值）再寫回
                kCharFarfieldTemperature<<<nBlocks(n), BS>>>
                (
                    start, n, dm.fvBoundaryFaces.data(), dm.owner.data(), meshVec(dm.Sf),
                    dm.magSf.data(), dm.deltaCoeffs.data(), th.params,
                    th.p.in.cview().p[0], th.p.bnd.cview().p[0], th.T.in.cview().p[0],
                    th.psi.in.cview().p[0], th.psi.bnd.cview().p[0],
                    s.rho.in.cview().p[0], s.rho.bnd.cview().p[0], flux.phi.bnd.cview().p[0],
                    s.U.in.cview(), bc.URef, bc.pRef, bc.TRef,
                    bc.refValue.view().p[0], bc.refGrad.view().p[0], bc.valueFraction.data(),
                    bc.TCoeff.view().p[0], bc.pCoeff.view().p[0], bc.UCoeff.view(),
                    true, th.T.bnd.view().p[0]
                );
                break;
            case Bc::inletOutlet:
            case Bc::mixed:
                if (bc.type == Bc::inletOutlet && bc.phiName != "phi")
                {
                    throw std::runtime_error
                    (
                        "field T patch " + pname + ": inletOutlet with flux " + bc.phiName
                      + " is not supported (phi only)"
                    );
                }
                kEvalMixedScalar<<<nBlocks(n), BS>>>
                (
                    start, n, dm.fvBoundaryFaces.data(), dm.owner.data(), dm.deltaCoeffs.data(),
                    bc.type == Bc::inletOutlet, flux.phi.bnd.cview().p[0], th.T.in.cview().p[0],
                    bc.refValue.cview().p[0], bc.refGrad.cview().p[0], bc.valueFraction.data(),
                    th.T.bnd.view().p[0]
                );
                break;
            case Bc::zeroGradient:
            case Bc::fixedGradient:
                kEvalGradientScalar<<<nBlocks(n), BS>>>
                (
                    start, n, dm.fvBoundaryFaces.data(), dm.owner.data(), dm.deltaCoeffs.data(),
                    th.T.in.cview().p[0],
                    bc.type == Bc::fixedGradient ? bc.gradient.cview().p[0] : nullptr,
                    th.T.bnd.view().p[0]
                );
                break;
            case Bc::fixedValue:
            case Bc::boundaryCorrectedFixedValue:
            case Bc::calculated:
            case Bc::extrapolatedCalculated:
            case Bc::symmetryPlane:
            case Bc::symmetry:
            case Bc::slip:
            case Bc::empty:
                break;
            default:
                throw std::runtime_error
                (
                    "field T patch " + pname + ": energy update of " + bcName(bc.type)
                  + " is not implemented yet"
                );
        }
        GNC_CUDA_LAUNCH_CHECK();
        if (th.he.bc[pi].type == Bc::fixedEnergy)
        {
            kFixedEnergy<<<nBlocks(n), BS>>>
            (
                start, n, th.params, th.p.bnd.cview().p[0], th.T.bnd.cview().p[0],
                th.he.bnd.view().p[0]
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
    }
    GNC_CUDA_CHECK(cudaDeviceSynchronize());
}


void updateCoeffsT(const DeviceMesh& dm, HisaState& s, const HisaFlux& flux)
{
    PsiThermoFields& th = s.thermo;
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const std::int32_t start = dm.fvPatchStart[pi];
        const std::int32_t n = dm.fvPatchSize[pi];
        PatchBC<double>& bc = th.T.bc[pi];
        if (n == 0)
        {
            continue;
        }
        switch (bc.type)
        {
            case Bc::characteristicWallTemperature:
                kCharWallTemperature<<<nBlocks(n), BS>>>
                (
                    start, n, dm.fvBoundaryFaces.data(), dm.owner.data(), meshVec(dm.Sf),
                    dm.magSf.data(), th.params, th.p.in.cview().p[0], th.T.in.cview().p[0],
                    th.psi.in.cview().p[0], s.U.in.cview(), s.U.bnd.cview(),
                    th.T.bnd.view().p[0]
                );
                break;
            case Bc::characteristicFarfieldTemperature:
                bc.UCoeff.allocate(std::size_t(n));
                bc.pCoeff.allocate(std::size_t(n));
                bc.TCoeff.allocate(std::size_t(n));
                kCharFarfieldTemperature<<<nBlocks(n), BS>>>
                (
                    start, n, dm.fvBoundaryFaces.data(), dm.owner.data(), meshVec(dm.Sf),
                    dm.magSf.data(), dm.deltaCoeffs.data(), th.params,
                    th.p.in.cview().p[0], th.p.bnd.cview().p[0], th.T.in.cview().p[0],
                    th.psi.in.cview().p[0], th.psi.bnd.cview().p[0],
                    s.rho.in.cview().p[0], s.rho.bnd.cview().p[0], flux.phi.bnd.cview().p[0],
                    s.U.in.cview(), bc.URef, bc.pRef, bc.TRef,
                    bc.refValue.view().p[0], bc.refGrad.view().p[0], bc.valueFraction.data(),
                    bc.TCoeff.view().p[0], bc.pCoeff.view().p[0], bc.UCoeff.view(),
                    false, th.T.bnd.view().p[0]
                );
                break;
            case Bc::inletOutlet:
                kInletOutletFraction<<<nBlocks(n), BS>>>
                (
                    start, n, flux.phi.bnd.cview().p[0], bc.valueFraction.data()
                );
                break;
            default:
                break;
        }
        GNC_CUDA_LAUNCH_CHECK();
    }
    GNC_CUDA_CHECK(cudaDeviceSynchronize());
}


void updateCoeffsP
(
    const DeviceMesh& dm,
    HisaState& s,
    const HisaFlux& flux,
    const FoamCompat& compat
)
{
    PsiThermoFields& th = s.thermo;
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const std::int32_t start = dm.fvPatchStart[pi];
        const std::int32_t n = dm.fvPatchSize[pi];
        PatchBC<double>& bc = th.p.bc[pi];
        if (n == 0)
        {
            continue;
        }
        const std::string& pname = dm.patches[pi].name;
        switch (bc.type)
        {
            case Bc::characteristicWallPressure:
                bc.UCoeff.allocate(std::size_t(n));
                bc.pCoeff.allocate(std::size_t(n));
                bc.TCoeff.allocate(std::size_t(n));
                kCharWallPressure<<<nBlocks(n), BS>>>
                (
                    start, n, dm.fvBoundaryFaces.data(), dm.owner.data(), meshVec(dm.Sf),
                    dm.magSf.data(), dm.deltaCoeffs.data(), th.params,
                    th.p.in.cview().p[0], th.T.in.cview().p[0], th.psi.in.cview().p[0],
                    s.U.in.cview(), s.U.bnd.cview(), compat.stabiliseSwappedArgs,
                    bc.gradient.view().p[0], bc.pCoeff.view().p[0], bc.UCoeff.view(),
                    bc.TCoeff.view().p[0]
                );
                break;
            case Bc::characteristicFarfieldPressure:
                bc.UCoeff.allocate(std::size_t(n));
                bc.pCoeff.allocate(std::size_t(n));
                bc.TCoeff.allocate(std::size_t(n));
                kCharFarfieldPressure<<<nBlocks(n), BS>>>
                (
                    start, n, dm.fvBoundaryFaces.data(), dm.owner.data(), meshVec(dm.Sf),
                    dm.magSf.data(), th.params,
                    th.p.in.cview().p[0], th.p.bnd.cview().p[0],
                    th.T.in.cview().p[0], th.T.bnd.cview().p[0],
                    th.psi.in.cview().p[0], th.psi.bnd.cview().p[0],
                    s.rho.in.cview().p[0], s.rho.bnd.cview().p[0],
                    flux.phi.bnd.cview().p[0], s.U.in.cview(), bc.URef, bc.pRef,
                    bc.refValue.view().p[0], bc.refGrad.view().p[0], bc.valueFraction.data(),
                    bc.pCoeff.view().p[0], bc.UCoeff.view(), bc.TCoeff.view().p[0]
                );
                break;
            case Bc::inletOutlet:
                if (bc.phiName != "phi")
                {
                    throw std::runtime_error
                    (
                        "field p patch " + pname + ": inletOutlet with flux " + bc.phiName
                      + " is not supported (phi only)"
                    );
                }
                kInletOutletFraction<<<nBlocks(n), BS>>>
                (
                    start, n, flux.phi.bnd.cview().p[0], bc.valueFraction.data()
                );
                break;
            case Bc::fixedValue:
            case Bc::boundaryCorrectedFixedValue:
            case Bc::zeroGradient:
            case Bc::fixedGradient:
            case Bc::mixed:
            case Bc::calculated:
            case Bc::extrapolatedCalculated:
            case Bc::symmetryPlane:
            case Bc::symmetry:
            case Bc::slip:
            case Bc::empty:
                break;
            default:
                throw std::runtime_error
                (
                    "field p patch " + pname + ": updateCoeffs of " + bcName(bc.type)
                  + " is not implemented yet"
                );
        }
        GNC_CUDA_LAUNCH_CHECK();
    }
    GNC_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace gnc
