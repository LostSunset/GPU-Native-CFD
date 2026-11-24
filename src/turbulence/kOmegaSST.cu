// k-omega SST（可壓縮、RAS）：參考次序版。GPU-Native-CFD，GPL-3.0-or-later。
//
// 出處（Atlas upstream/openfoam，v2606）：kOmegaSSTBase.C:42–200（F1、F2、F3、F23、
// correctNut、S2、Pk、epsilonByk、GbyNu0、GbyNu）、:438–442（建構時的 bound）、
// :499–624（correct）；kOmegaSSTBase.H（blend、alphaK、alphaOmega、beta、gamma、
// DkEff、DomegaEff）；RAS/kOmegaSST.C；TensorI.H:747（symm）、:788（devTwoSymm）、
// :1271（Tensor && SymmTensor）；SymmTensorI.H:614（magSqr）；scalarImpl.H（sqr、
// pow3、pow4、pow025）；one.H:72–80（geometricOneField 的乘法不運算）；
// omegaWallFunctionFvPatchScalarField.C、kLowReWallFunctionFvPatchScalarField.C、
// nutkWallFunctionFvPatchScalarField.C、wallFunctionCoefficients.C、
// turbulentIntensityKineticEnergyInletFvPatchScalarField.cxx、
// turbulentMixingLengthFrequencyInletFvPatchScalarField.C。
// 欄位運算的次序以 C++ 的左結合為準，例如
//   F1 的 arg1 = min(min(max(((1/betaStar)*sqrt(k))/(omega*y), (500*nu)/(sqr(y)*omega)),
//                        ((4*alphaOmega2)*k)/(CDkOmegaPlus*sqr(y))), 10)
//   GbyNu      = min(GbyNu0, (((c1/a1)*betaStar)*omega)*max(a1*omega, (b1*F2)*sqrt(S2)))
//   nut        = (a1*k)/max(a1*omega, (b1*F23)*sqrt(S2))
// 必須以 --fmad=false 編譯。

#include "turbulence/kOmegaSST.cuh"

#include "turbulence/turbulenceCommon.cuh"

#include "core/glibcLog.cuh"
#include "core/glibcPow.cuh"
#include "core/glibcTanh.cuh"
#include "fields/boundaryConditions.cuh"
#include "io/dictionary.hpp"
#include "io/fieldIO.hpp"
#include "io/foamTokenizer.hpp"

#include <cmath>
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

struct SstConsts
{
    double oneByBetaStar, twoByBetaStar, fourAlphaOmega2, twoAlphaOmega2;
    double dGamma, gamma2, dBeta, beta2, a1, b1, c1a1BetaStar, c1BetaStar, betaStar;
};

SstConsts constsOf(const KOmegaSSTCoeffs& c)
{
    return
    {
        c.oneByBetaStar, c.twoByBetaStar, c.fourAlphaOmega2, c.twoAlphaOmega2,
        c.dGamma, c.gamma2, c.dBeta, c.beta2, c.a1, c.b1, c.c1a1BetaStar, c.c1BetaStar, c.betaStar
    };
}


// S2 = 2*magSqr(symm(gradU))（內部或邊界逐項）
__global__ void kS2(const std::size_t n, const CView<tens> gradU, double* __restrict__ S2)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const tens t = gradU[i];
    const double sxx = t.xx;
    const double sxy = 0.5*(t.xy + t.yx);
    const double sxz = 0.5*(t.xz + t.zx);
    const double syy = t.yy;
    const double syz = 0.5*(t.yz + t.zy);
    const double szz = t.zz;
    const double ms =
        sxx*sxx + 2*(sxy*sxy) + 2*(sxz*sxz)
      + syy*syy + 2*(syz*syz)
      + szz*szz;
    S2[i] = 2*ms;
}

// GbyNu0 = gradU && devTwoSymm(gradU)；G = nut*GbyNu0（內部）
__global__ void kGbyNu0
(
    const std::size_t n,
    const CView<tens> gradU,
    const double* __restrict__ nut,
    double* __restrict__ GbyNu0,
    double* __restrict__ G
)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const tens t = gradU[i];
    const double sph = (2.0/3.0)*((t.xx + t.yy) + t.zz);
    const double dxx = 2*t.xx - sph;
    const double dxy = t.xy + t.yx;
    const double dxz = t.xz + t.zx;
    const double dyy = 2*t.yy - sph;
    const double dyz = t.yz + t.zy;
    const double dzz = 2*t.zz - sph;
    const double g =
        t.xx*dxx + t.xy*dxy + t.xz*dxz
      + t.yx*dxy + t.yy*dyy + t.yz*dyz
      + t.zx*dxz + t.zy*dyz + t.zz*dzz;
    GbyNu0[i] = g;
    G[i] = nut[i]*g;
}

// phi()/interpolate(rho)（面值逐項相除）
__global__ void kDivideFaces(const std::size_t n, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = a[i]/b[i];
    }
}

// CDkOmega = ((2*alphaOmega2)*(grad(k) & grad(omega)))/omega（內部或邊界逐項）
__global__ void kCDkOmega
(
    const std::size_t n,
    const CView<vec3> gk,
    const CView<vec3> gw,
    const double* __restrict__ omega,
    const double twoAlphaOmega2,
    double* __restrict__ CD
)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n)
    {
        CD[i] = (twoAlphaOmega2*dot(gk[i], gw[i]))/omega[i];
    }
}

// F2 的引數與 tanh（F3 off：F23 = F2）
__device__ __forceinline__ double sstF2
(
    const SstConsts& c,
    const double k,
    const double omega,
    const double y,
    const double nu
)
{
    const double a = (c.twoByBetaStar*::sqrt(k))/(omega*y);
    const double b = (500.0*nu)/(foamSqr(y)*omega);
    const double arg2 = foamMin(foamMax(a, b), 100.0);
    return glibcTanh(foamSqr(arg2));
}

// F1（CD 為 CDkOmega）與 F2（內部或邊界逐項；CD 可為 nullptr 時不算 F1）
__global__ void kF12
(
    const std::size_t n,
    const SstConsts c,
    const double* __restrict__ k,
    const double* __restrict__ omega,
    const double* __restrict__ y,
    const double* __restrict__ nu,
    const double* __restrict__ CD,
    double* __restrict__ F1,
    double* __restrict__ F2
)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const double ki = k[i], wi = omega[i], yi = y[i], nui = nu[i];
    if (CD)
    {
        const double CDplus = foamMax(CD[i], 1.0e-10);
        const double a = (c.oneByBetaStar*::sqrt(ki))/(wi*yi);
        const double b = (500.0*nui)/(foamSqr(yi)*wi);
        const double d = (c.fourAlphaOmega2*ki)/(CDplus*foamSqr(yi));
        const double arg1 = foamMin(foamMin(foamMax(a, b), d), 10.0);
        F1[i] = glibcTanh(foamSqr(foamSqr(arg1)));
    }
    F2[i] = sstF2(c, ki, wi, yi, nui);
}

// rho*DEff：DEff = ((F1*(psi1 - psi2) + psi2)*nut) + nu（內部或邊界逐項）
__global__ void kGammaEff
(
    const std::size_t n,
    const double* __restrict__ rho,
    const double* __restrict__ F1,
    const double* __restrict__ nut,
    const double* __restrict__ nu,
    const double dPsi,
    const double psi2,
    double* __restrict__ out
)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const double alpha = F1[i]*dPsi + psi2;
        out[i] = rho[i]*(alpha*nut[i] + nu[i]);
    }
}

// omega 方程的源項（內部）：
//   gamma = F1*(gamma1 - gamma2) + gamma2、beta = F1*(beta1 - beta2) + beta2
//   GbyNu = min(GbyNu0, (((c1/a1)*betaStar)*omega)*max(a1*omega, (b1*F2)*sqrt(S2)))
//   su    = (rho*gamma)*GbyNu
//   susp1 = (((2/3)*rho)*gamma)*divU
//   sp    = (rho*beta)*omega
//   susp2 = ((rho*(F1 - 1))*CDkOmega)/omega
__global__ void kOmegaSources
(
    const std::size_t n,
    const SstConsts c,
    const double* __restrict__ rho,
    const double* __restrict__ omega,
    const double* __restrict__ F1,
    const double* __restrict__ F2,
    const double* __restrict__ S2,
    const double* __restrict__ GbyNu0,
    const double* __restrict__ CD,
    const double* __restrict__ divU,
    double* __restrict__ su,
    double* __restrict__ susp1,
    double* __restrict__ sp,
    double* __restrict__ susp2
)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const double r = rho[i], w = omega[i], f1 = F1[i];
    const double gamma = f1*c.dGamma + c.gamma2;
    const double beta = f1*c.dBeta + c.beta2;
    const double GbyNu = foamMin
    (
        GbyNu0[i],
        (c.c1a1BetaStar*w)*foamMax(c.a1*w, (c.b1*F2[i])*::sqrt(S2[i]))
    );
    su[i] = (r*gamma)*GbyNu;
    susp1[i] = (((2.0/3.0)*r)*gamma)*divU[i];
    sp[i] = (r*beta)*w;
    susp2[i] = ((r*(f1 - 1.0))*CD[i])/w;
}

// k 方程的源項（內部；omega 為新解）：
//   su   = rho*min(G, ((c1*betaStar)*k)*omega)
//   susp = ((2/3)*rho)*divU
//   sp   = rho*(betaStar*omega)
__global__ void kKSources
(
    const std::size_t n,
    const SstConsts c,
    const double* __restrict__ rho,
    const double* __restrict__ k,
    const double* __restrict__ omega,
    const double* __restrict__ G,
    const double* __restrict__ divU,
    double* __restrict__ su,
    double* __restrict__ susp,
    double* __restrict__ sp
)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const double r = rho[i], w = omega[i];
    su[i] = r*foamMin(G[i], (c.c1BetaStar*k[i])*w);
    susp[i] = ((2.0/3.0)*r)*divU[i];
    sp[i] = r*(c.betaStar*w);
}

// nut = (a1*k)/max(a1*omega, (b1*F23)*sqrt(S2))（內部或邊界逐項）
__global__ void kNutSST
(
    const std::size_t n,
    const SstConsts c,
    const double* __restrict__ k,
    const double* __restrict__ omega,
    const double* __restrict__ F2,
    const double* __restrict__ S2,
    double* __restrict__ nut
)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n)
    {
        nut[i] = (c.a1*k[i])/foamMax(c.a1*omega[i], (c.b1*F2[i])*::sqrt(S2[i]));
    }
}


// omegaWallFunction 的係數（所有 omegaWall patch 相同）
struct OmegaWallConsts
{
    double Cmu25, kappa, yPlusLam, beta1, n, invN;
    int blending;
};

// omegaWallFunctionFvPatchScalarField::calculate（所有 omegaWall patch，依 patch 與
// 面的次序對每個牆邊格累加）：
//   yPlus    = ((Cmu25*y)*sqrt(k))/nuw
//   omegaVis = (6*nuw)/(beta1*sqr(y))
//   omegaLog = sqrt(k)/((Cmu25*kappa)*y)
//   omega0  += w*blend(omegaVis, omegaLog)
//   G0      += ((((w*(nutw + nuw))*mag(snGrad(U)))*Cmu25)*sqrt(k))/(kappa*y)
//              （stepwise 時只在 yPlus > yPlusLam）
// 之後 updateCoeffs 把牆邊格的 G、omega 換成 G0、omega0
__global__ void kOmegaWall
(
    const std::int32_t nWallCells,
    const std::int32_t* __restrict__ wallCells,
    const std::int32_t* __restrict__ off,
    const std::int32_t* __restrict__ faces,
    const OmegaWallConsts c,
    const double* __restrict__ k,
    const double* __restrict__ nutB,
    const double* __restrict__ muB,
    const double* __restrict__ rhoB,
    const double* __restrict__ yNW,
    const CView<vec3> snGradU,
    const double* __restrict__ w,
    double* __restrict__ omega,
    double* __restrict__ G
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nWallCells)
    {
        return;
    }
    const std::int32_t cell = wallCells[i];
    const double sk = ::sqrt(k[cell]);
    double omega0 = 0.0;
    double G0 = 0.0;
    for (std::int32_t j = off[i]; j < off[i + 1]; ++j)
    {
        const std::int32_t b = faces[j];
        const double y = yNW[b];
        const double nuw = muB[b]/rhoB[b];
        const double yPlus = ((c.Cmu25*y)*sk)/nuw;
        const double omegaVis = (6.0*nuw)/(c.beta1*foamSqr(y));
        const double omegaLog = sk/((c.Cmu25*c.kappa)*y);
        double v;
        switch (c.blending)
        {
            case 0:
                v = yPlus > c.yPlusLam ? omegaLog : omegaVis;
                break;
            case 1:
                v = foamMax(omegaVis, omegaLog);
                break;
            default:
                v = glibcPow(glibcPow(omegaVis, c.n) + glibcPow(omegaLog, c.n), c.invN);
                break;
        }
        omega0 = omega0 + w[b]*v;
        if (c.blending != 0 || yPlus > c.yPlusLam)
        {
            G0 = G0 + ((((w[b]*(nutB[b] + nuw))*mag(snGradU[b]))*c.Cmu25)*sk)/(c.kappa*y);
        }
    }
    omega[cell] = omega0;
    G[cell] = G0;
}

// 有限體積邊界面取相鄰格值（opf == omega0[faceCells]）
__global__ void kPatchFromCells
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

// kLowReWallFunctionFvPatchScalarField::updateCoeffs：
//   uTau  = Cmu25*sqrt(k)；yPlus = (uTau*y)/nuw
//   kw    = (Ck/kappa)*log(yPlus) + Bk                         （yPlus > yPlusLam）
//         = (2400/sqr(Ceps2))*Cf，Cf = (1/sqr(yPlus + C) + (2*yPlus)/pow3(C)) - 1/sqr(C)
//   kw    = max(kw*sqr(uTau), SMALL)
__global__ void kKLowRe
(
    const std::int32_t start,
    const std::int32_t n,
    const std::int32_t* __restrict__ fvbFaces,
    const std::int32_t* __restrict__ owner,
    const double* __restrict__ k,
    const double* __restrict__ muB,
    const double* __restrict__ rhoB,
    const double* __restrict__ yNW,
    const double Cmu25,
    const double kappa,
    const double yPlusLam,
    const double Ceps2,
    const double Ck,
    const double Bk,
    const double C,
    double* __restrict__ kB
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const std::int32_t b = start + i;
    const double nuw = muB[b]/rhoB[b];
    const double uTau = Cmu25*::sqrt(k[owner[fvbFaces[b]]]);
    const double yPlus = (uTau*yNW[b])/nuw;
    double kw;
    if (yPlus > yPlusLam)
    {
        kw = (Ck/kappa)*glibcLog(yPlus) + Bk;
    }
    else
    {
        const double Cf = (1.0/foamSqr(yPlus + C) + (2.0*yPlus)/foamPow3(C)) - 1.0/foamSqr(C);
        kw = (2400.0/foamSqr(Ceps2))*Cf;
    }
    kw = kw*foamSqr(uTau);
    kB[b] = foamMax(kw, SMALL);
}

// nutkWallFunctionFvPatchScalarField::calcNut：
//   yPlus  = ((Cmu25*y)*sqrt(k))/nuw
//   nutLog = ((nuw*yPlus)*kappa)/log(max(E*yPlus, 1 + 1e-4))
//   nutw   = blend(nuw, nutLog) - nuw（stepwise：yPlus > yPlusLam 時 nutLog，否則 nuw）
__global__ void kNutk
(
    const std::int32_t start,
    const std::int32_t n,
    const std::int32_t* __restrict__ fvbFaces,
    const std::int32_t* __restrict__ owner,
    const double* __restrict__ k,
    const double* __restrict__ muB,
    const double* __restrict__ rhoB,
    const double* __restrict__ yNW,
    const double Cmu25,
    const double kappa,
    const double E,
    const double yPlusLam,
    const int blending,
    const double nBlend,
    double* __restrict__ nutB
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const std::int32_t b = start + i;
    const double nuw = muB[b]/rhoB[b];
    const double yPlus = ((Cmu25*yNW[b])*::sqrt(k[owner[fvbFaces[b]]]))/nuw;
    const auto nutLog = [&]()
    {
        return ((nuw*yPlus)*kappa)/glibcLog(foamMax(E*yPlus, 1 + 1e-4));
    };
    double nutw;
    switch (blending)
    {
        case 0:
            nutw = yPlus > yPlusLam ? nutLog() : nuw;
            break;
        case 1:
            nutw = foamMax(nuw, nutLog());
            break;
        default:
            nutw = glibcPow(glibcPow(nuw, nBlend) + glibcPow(nutLog(), nBlend), 1.0/nBlend);
            break;
    }
    nutB[b] = nutw - nuw;
}

// turbulentIntensityKineticEnergyInlet：refValue = (1.5*sqr(I))*magSqr(U_b)
__global__ void kKInletRef
(
    const std::int32_t start,
    const std::int32_t n,
    const CView<vec3> Ub,
    const double c15I2,
    double* __restrict__ refValue
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        refValue[i] = c15I2*magSqr(Ub[start + i]);
    }
}

// turbulentMixingLengthFrequencyInlet：refValue = sqrt(k_b)/(Cmu25*L)
__global__ void kOmegaInletRef
(
    const std::int32_t start,
    const std::int32_t n,
    const double* __restrict__ kB,
    const double Cmu25L,
    double* __restrict__ refValue
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        refValue[i] = ::sqrt(kB[start + i])/Cmu25L;
    }
}


// div 方案：bounded Gauss upwind [phi]，或 bounded Gauss limitedLinear k
void readDivScheme(const FvSchemes& schemes, const std::string& name, bool& limitedLinear, double& coeff)
{
    const std::string div = trimmedScheme(schemes.div(name));
    limitedLinear = false;
    if (div == "bounded Gauss upwind phi" || div == "bounded Gauss upwind")
    {
        return;
    }
    const std::string ll = "bounded Gauss limitedLinear ";
    if (div.rfind(ll, 0) == 0)
    {
        std::size_t used = 0;
        const std::string rest = div.substr(ll.size());
        coeff = std::stod(rest, &used);
        if (used == rest.size() && coeff >= 0 && coeff <= 1)
        {
            limitedLinear = true;
            return;
        }
    }
    throw std::runtime_error
    (
        name + " " + div + " is not supported (bounded Gauss upwind, bounded Gauss limitedLinear k)"
    );
}

FvScalarMatrix divMatrix
(
    const DeviceMesh& dm,
    const SurfField<double>& phi,
    const VolField<double>& psi,
    const VolField<vec3>& gradPsi,
    const bool limitedLinear,
    const double coeff
)
{
    return limitedLinear
        ? fvmDivBoundedGaussLimitedLinear(dm, phi, psi, gradPsi, coeff)
        : fvmDivBoundedGaussUpwind(dm, phi, psi);
}

// pow025(Cmu) = sqrt(sqrt(Cmu))
double pow025(const double s)
{
    return std::sqrt(std::sqrt(s));
}

// pow(Cmu, 0.25)（主機端的 glibc pow；volatile 讓編譯器不以常數折疊）
__attribute__((noinline)) double hostPow(const double x, const double y)
{
    volatile double a = x, b = y;
    return std::pow(a, b);
}

const PatchBC<double>* firstOfType(const VolField<double>& f, const Bc t)
{
    for (const PatchBC<double>& bc : f.bc)
    {
        if (bc.type == t)
        {
            return &bc;
        }
    }
    return nullptr;
}


// omega 的邊界 updateCoeffs（omega_.boundaryFieldRef().updateCoeffs()）：
// omegaWallFunction 改寫牆邊格的 omega、G 與牆面值；
// turbulentMixingLengthFrequencyInlet 的 refValue；inletOutlet 族的 valueFraction
void omegaUpdateCoeffs
(
    const DeviceMesh& dm,
    KOmegaSST& m,
    const VolField<double>& rho,
    const VolField<double>& mu,
    const VolField<vec3>& U,
    const SurfField<double>& phi,
    double* G
)
{
    if (m.hasOmegaWall && m.nWallCells)
    {
        const PatchBC<double>& bc = *firstOfType(m.omega, Bc::omegaWallFunction);
        OmegaWallConsts c;
        c.Cmu25 = pow025(bc.wallCmu);
        c.kappa = bc.wallKappa;
        c.yPlusLam = bc.yPlusLam;
        c.beta1 = bc.wallBeta1;
        c.n = bc.wallBlendN;
        c.invN = 1.0/bc.wallBlendN;
        c.blending = bc.wallBlending;
        DField<vec3> snGradU;
        snGradBCs(dm, U, snGradU);
        kOmegaWall<<<nBlocks(std::size_t(m.nWallCells)), BS>>>
        (
            m.nWallCells, m.wallCells.data(), m.wallCellOff.data(), m.wallCellFaces.data(), c,
            m.k.in.cview().p[0], m.nut.bnd.cview().p[0], mu.bnd.cview().p[0], rho.bnd.cview().p[0],
            m.yNearWall.data(), snGradU.cview(), m.cornerWeight.data(), m.omega.in.view().p[0], G
        );
        GNC_CUDA_LAUNCH_CHECK();
        for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
        {
            const std::int32_t size = dm.fvPatchSize[pi];
            if (size && m.omega.bc[pi].type == Bc::omegaWallFunction)
            {
                kPatchFromCells<<<nBlocks(std::size_t(size)), BS>>>
                (
                    dm.fvPatchStart[pi], size, dm.fvBoundaryFaces.data(), dm.owner.data(),
                    m.omega.in.cview().p[0], m.omega.bnd.view().p[0]
                );
                GNC_CUDA_LAUNCH_CHECK();
            }
        }
    }
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        PatchBC<double>& bc = m.omega.bc[pi];
        const std::int32_t size = dm.fvPatchSize[pi];
        if (size && bc.type == Bc::turbulentMixingLengthFrequencyInlet)
        {
            kOmegaInletRef<<<nBlocks(std::size_t(size)), BS>>>
            (
                dm.fvPatchStart[pi], size, m.k.bnd.cview().p[0], m.c.mixingCmu25*bc.mixingLength,
                bc.refValue.view().p[0]
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
    }
    updateInletOutletCoeffs(dm, m.omega, phi);
}


// k 的邊界 updateCoeffs（kEqn 的 fvMatrix 建構時）：kLowReWallFunction 的牆面值、
// turbulentIntensityKineticEnergyInlet 的 refValue、inletOutlet 族的 valueFraction
void kUpdateCoeffs
(
    const DeviceMesh& dm,
    KOmegaSST& m,
    const VolField<double>& rho,
    const VolField<double>& mu,
    const VolField<vec3>& U,
    const SurfField<double>& phi
)
{
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        PatchBC<double>& bc = m.k.bc[pi];
        const std::int32_t start = dm.fvPatchStart[pi];
        const std::int32_t size = dm.fvPatchSize[pi];
        if (size == 0)
        {
            continue;
        }
        if (bc.type == Bc::kLowReWallFunction)
        {
            kKLowRe<<<nBlocks(std::size_t(size)), BS>>>
            (
                start, size, dm.fvBoundaryFaces.data(), dm.owner.data(), m.k.in.cview().p[0],
                mu.bnd.cview().p[0], rho.bnd.cview().p[0], m.yNearWall.data(),
                pow025(bc.wallCmu), bc.wallKappa, bc.yPlusLam, bc.wallCeps2, bc.wallCk, bc.wallBk,
                bc.wallC, m.k.bnd.view().p[0]
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
        else if (bc.type == Bc::turbulentIntensityKineticEnergyInlet)
        {
            kKInletRef<<<nBlocks(std::size_t(size)), BS>>>
            (
                start, size, U.bnd.cview(), 1.5*foamSqr(bc.intensity), bc.refValue.view().p[0]
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
    }
    updateInletOutletCoeffs(dm, m.k, phi);
}


// F1（可選）與 F2 的內部值與邊界值
void blendingFunctions
(
    const DeviceMesh& dm,
    const KOmegaSST& m,
    const VolField<double>& nu,
    const VolField<double>* CD,
    VolField<double>* F1,
    VolField<double>& F2
)
{
    const std::size_t nC = std::size_t(dm.nCells);
    const std::size_t nB = std::size_t(dm.nFvBoundaryFaces());
    const SstConsts c = constsOf(m.c);
    if (F1)
    {
        F1->in.allocate(nC);
        F1->bnd.allocate(nB);
    }
    F2.in.allocate(nC);
    F2.bnd.allocate(nB);
    if (nC)
    {
        kF12<<<nBlocks(nC), BS>>>
        (
            nC, c, m.k.in.cview().p[0], m.omega.in.cview().p[0], m.y.data(), nu.in.cview().p[0],
            CD ? CD->in.cview().p[0] : nullptr, F1 ? F1->in.view().p[0] : nullptr, F2.in.view().p[0]
        );
    }
    if (nB)
    {
        kF12<<<nBlocks(nB), BS>>>
        (
            nB, c, m.k.bnd.cview().p[0], m.omega.bnd.cview().p[0], m.yB.data(), nu.bnd.cview().p[0],
            CD ? CD->bnd.cview().p[0] : nullptr, F1 ? F1->bnd.view().p[0] : nullptr, F2.bnd.view().p[0]
        );
    }
    GNC_CUDA_LAUNCH_CHECK();
}


// correctNut(S2)：nut = a1*k/max(a1*omega, b1*F23*sqrt(S2))、
// nut.correctBoundaryConditions()（nutkWallFunction 的 calcNut）、alphat
void correctNutS2
(
    const DeviceMesh& dm,
    KOmegaSST& m,
    const VolField<double>& rho,
    const VolField<double>& mu,
    const VolField<double>& nu,
    const VolField<double>& S2
)
{
    const std::size_t nC = std::size_t(dm.nCells);
    const std::size_t nB = std::size_t(dm.nFvBoundaryFaces());
    VolField<double> F2;
    blendingFunctions(dm, m, nu, nullptr, nullptr, F2);
    const SstConsts c = constsOf(m.c);
    DeviceArray<double> nutB(nB);
    if (nC)
    {
        kNutSST<<<nBlocks(nC), BS>>>
        (
            nC, c, m.k.in.cview().p[0], m.omega.in.cview().p[0], F2.in.cview().p[0],
            S2.in.cview().p[0], m.nut.in.view().p[0]
        );
    }
    if (nB)
    {
        kNutSST<<<nBlocks(nB), BS>>>
        (
            nB, c, m.k.bnd.cview().p[0], m.omega.bnd.cview().p[0], F2.bnd.cview().p[0],
            S2.bnd.cview().p[0], nutB.data()
        );
    }
    GNC_CUDA_LAUNCH_CHECK();
    assignBoundary(dm, m.nut, nutB);
    // nut.correctBoundaryConditions()：nutkWallFunction 的 updateCoeffs 在這裡算
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const PatchBC<double>& bc = m.nut.bc[pi];
        const std::int32_t size = dm.fvPatchSize[pi];
        if (size == 0 || bc.type != Bc::nutkWallFunction)
        {
            continue;
        }
        kNutk<<<nBlocks(std::size_t(size)), BS>>>
        (
            dm.fvPatchStart[pi], size, dm.fvBoundaryFaces.data(), dm.owner.data(),
            m.k.in.cview().p[0], mu.bnd.cview().p[0], rho.bnd.cview().p[0], m.yNearWall.data(),
            pow025(bc.wallCmu), bc.wallKappa, bc.wallE, bc.yPlusLam, bc.wallBlending, bc.wallBlendN,
            m.nut.bnd.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    evaluateBCs(dm, m.nut, nullptr);

    // RAS::kOmegaSST::correctNut：BasicTurbulenceModel::correctNut（EddyDiffusivity）
    eddyDiffusivityCorrectNut(dm, rho, m.nut, m.c.Prt, m.alphat);
}


// S2 的內部值與邊界值（gradU 依 fvSchemes 的 grad(U)，含邊界值）
void computeS2(const DeviceMesh& dm, const VolField<tens>& gradU, VolField<double>& S2)
{
    const std::size_t nC = std::size_t(dm.nCells);
    const std::size_t nB = std::size_t(dm.nFvBoundaryFaces());
    S2.in.allocate(nC);
    S2.bnd.allocate(nB);
    if (nC)
    {
        kS2<<<nBlocks(nC), BS>>>(nC, gradU.in.cview(), S2.in.view().p[0]);
    }
    if (nB)
    {
        kS2<<<nBlocks(nB), BS>>>(nB, gradU.bnd.cview(), S2.bnd.view().p[0]);
    }
    GNC_CUDA_LAUNCH_CHECK();
}


// rho*DEff(F1) 的內部值與邊界值
void gammaEff
(
    const DeviceMesh& dm,
    const VolField<double>& rho,
    const VolField<double>& F1,
    const VolField<double>& nut,
    const VolField<double>& nu,
    const double dPsi,
    const double psi2,
    const std::string& name,
    VolField<double>& out
)
{
    const std::size_t nC = std::size_t(dm.nCells);
    const std::size_t nB = std::size_t(dm.nFvBoundaryFaces());
    out.name = name;
    out.in.allocate(nC);
    out.bnd.allocate(nB);
    if (nC)
    {
        kGammaEff<<<nBlocks(nC), BS>>>
        (
            nC, rho.in.cview().p[0], F1.in.cview().p[0], nut.in.cview().p[0], nu.in.cview().p[0],
            dPsi, psi2, out.in.view().p[0]
        );
    }
    if (nB)
    {
        kGammaEff<<<nBlocks(nB), BS>>>
        (
            nB, rho.bnd.cview().p[0], F1.bnd.cview().p[0], nut.bnd.cview().p[0], nu.bnd.cview().p[0],
            dPsi, psi2, out.bnd.view().p[0]
        );
    }
    GNC_CUDA_LAUNCH_CHECK();
}

} // namespace


KOmegaSST createKOmegaSST
(
    const DeviceMesh& dm,
    const PolyMesh& mesh,
    const std::string& expandedDir,
    const FvSchemes& schemes
)
{
    KOmegaSST m;

    // turbulenceProperties：simulationType RAS；RAS/RASModel kOmegaSST；
    // coeffDict = optionalSubDict("kOmegaSSTCoeffs")
    {
        const std::string path = expandedDir + "/turbulenceProperties";
        FoamTokenizer t(path);
        const Dictionary d = Dictionary::parse(t, path, false);
        if (d.getWord("simulationType") != "RAS")
        {
            throw std::runtime_error(path + ": simulationType must be RAS");
        }
        const Dictionary& ras = d.subDict("RAS");
        if (ras.getWord("RASModel") != "kOmegaSST")
        {
            throw std::runtime_error(path + ": RASModel " + ras.getWord("RASModel") + " is not kOmegaSST");
        }
        if (ras.found("turbulence"))
        {
            m.turbulence = parseSwitch(ras.getWord("turbulence"), path);
        }
        const Dictionary::Entry* ce = ras.find("kOmegaSSTCoeffs");
        const Dictionary& cd = (ce && ce->dict) ? *ce->dict : ras;
        KOmegaSSTCoeffs& c = m.c;
        const auto rd = [&](const Dictionary& dd, const char* k, double& v)
        {
            if (dd.found(k)) v = dd.getScalar(k);
        };
        rd(cd, "alphaK1", c.alphaK1);
        rd(cd, "alphaK2", c.alphaK2);
        rd(cd, "alphaOmega1", c.alphaOmega1);
        rd(cd, "alphaOmega2", c.alphaOmega2);
        rd(cd, "gamma1", c.gamma1);
        rd(cd, "gamma2", c.gamma2);
        rd(cd, "beta1", c.beta1);
        rd(cd, "beta2", c.beta2);
        rd(cd, "betaStar", c.betaStar);
        rd(cd, "a1", c.a1);
        rd(cd, "b1", c.b1);
        rd(cd, "c1", c.c1);
        rd(cd, "Prt", c.Prt);
        rd(cd, "Cmu", c.Cmu);
        rd(ras, "kMin", c.kMin);
        rd(ras, "omegaMin", c.omegaMin);
        for (const char* sw : {"F3", "decayControl"})
        {
            if (cd.found(sw) && parseSwitch(cd.getWord(sw), path))
            {
                throw std::runtime_error(path + ": " + sw + " true is not supported");
            }
        }
        // dimensionedScalar 的運算（kOmegaSSTBase.C/H）
        c.oneByBetaStar = 1.0/c.betaStar;
        c.twoByBetaStar = 2.0/c.betaStar;
        c.fourAlphaOmega2 = 4*c.alphaOmega2;
        c.twoAlphaOmega2 = 2*c.alphaOmega2;
        c.dAlphaK = c.alphaK1 - c.alphaK2;
        c.dAlphaOmega = c.alphaOmega1 - c.alphaOmega2;
        c.dGamma = c.gamma1 - c.gamma2;
        c.dBeta = c.beta1 - c.beta2;
        c.c1a1BetaStar = (c.c1/c.a1)*c.betaStar;
        c.c1BetaStar = c.c1*c.betaStar;
        c.mixingCmu25 = hostPow(c.Cmu, 0.25);
    }

    // fvSchemes
    readDivScheme(schemes, "div(phi,k)", m.kLimitedLinear, m.kLimitedLinearCoeff);
    readDivScheme(schemes, "div(phi,omega)", m.omegaLimitedLinear, m.omegaLimitedLinearCoeff);
    m.kLaplacian = parseLaplacianScheme
    (
        trimmedScheme(schemes.laplacian("laplacian((rho*DkEff),k)")), "laplacian((rho*DkEff),k)"
    );
    m.omegaLaplacian = parseLaplacianScheme
    (
        trimmedScheme(schemes.laplacian("laplacian((rho*DomegaEff),omega)")),
        "laplacian((rho*DomegaEff),omega)"
    );
    m.gradUScheme = trimmedScheme(schemes.grad("grad(U)"));
    m.gradKScheme = trimmedScheme(schemes.grad("grad(k)"));
    m.gradOmegaScheme = trimmedScheme(schemes.grad("grad(omega)"));
    if (trimmedScheme(schemes.interpolation("interpolate(rho)")) != "linear")
    {
        throw std::runtime_error("interpolate(rho) must be linear");
    }

    // fvSolution
    {
        const std::string path = expandedDir + "/fvSolution";
        m.kSolver = readSmoothSolverControls(path, "k");
        m.omegaSolver = readSmoothSolverControls(path, "omega");
        const EquationRelax rk = readEquationRelax(path, "k");
        const EquationRelax rw = readEquationRelax(path, "omega");
        m.kRelax = rk.relax;
        m.kRelaxFactor = rk.factor;
        m.omegaRelax = rw.relax;
        m.omegaRelaxFactor = rw.factor;
    }

    // 場
    {
        const HostField hk = readField(expandedDir + "/k", mesh);
        const HostField hw = readField(expandedDir + "/omega", mesh);
        const HostField hn = readField(expandedDir + "/nut", mesh);
        const HostField ha = readField(expandedDir + "/alphat", mesh);
        m.k = uploadVolField<double>(dm, hk);
        m.omega = uploadVolField<double>(dm, hw);
        m.nut = uploadVolField<double>(dm, hn);
        m.alphat = uploadVolField<double>(dm, ha);
        m.kDimensions = hk.dimensions;
        m.omegaDimensions = hw.dimensions;
        m.nutDimensions = hn.dimensions;
        m.alphatDimensions = ha.dimensions;
    }

    // 牆距離（meshWave 或 Poisson，網格不變時算一次）
    {
        const LSVectors ls = faceLeastSquaresVectors(dm);
        WallDistanceField w = wallDistanceFromSchemes(dm, mesh, schemes, ls);
        m.y = std::move(w.y);
        m.yB = std::move(w.yB);
        m.yNearWall = std::move(w.yNearWall);
    }

    // omegaWallFunction：cornerWeights（createAveragingWeights）、牆邊格的面清單、
    // setValues 的收集表
    {
        const std::size_t nB = std::size_t(dm.nFvBoundaryFaces());
        const std::vector<std::int32_t> fvb = dm.fvBoundaryFaces.download();
        std::vector<double> weights(std::size_t(dm.nCells), 0.0);
        std::vector<std::int32_t> faceCells;
        std::vector<std::int32_t> wallFaces;
        const PatchBC<double>* ref = nullptr;
        for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
        {
            const PatchBC<double>& bc = m.omega.bc[pi];
            if (bc.type != Bc::omegaWallFunction)
            {
                continue;
            }
            if (!ref)
            {
                ref = &bc;
            }
            else if
            (
                bc.wallCmu != ref->wallCmu || bc.wallKappa != ref->wallKappa || bc.wallE != ref->wallE
             || bc.wallBeta1 != ref->wallBeta1 || bc.wallBlending != ref->wallBlending
             || bc.wallBlendN != ref->wallBlendN
            )
            {
                throw std::runtime_error
                (
                    "omega patch " + dm.patches[pi].name
                  + ": omegaWallFunction coefficients differing between patches are not supported"
                );
            }
            m.hasOmegaWall = true;
            const std::size_t start = std::size_t(dm.fvPatchStart[pi]);
            for (std::size_t b = start; b < start + std::size_t(dm.fvPatchSize[pi]); ++b)
            {
                const std::int32_t c = mesh.owner[std::size_t(fvb[b])];
                weights[std::size_t(c)] += 1;
                faceCells.push_back(c);
                wallFaces.push_back(std::int32_t(b));
            }
        }
        std::vector<double> cw(nB, 0.0);
        for (const std::int32_t b : wallFaces)
        {
            cw[std::size_t(b)] = 1.0/weights[std::size_t(mesh.owner[std::size_t(fvb[std::size_t(b)])])];
        }
        m.cornerWeight.upload(cw);
        // 牆邊格（第一次出現的次序）與各格的面（patch、面的次序）
        std::vector<std::int32_t> slot(std::size_t(dm.nCells), -1), cells;
        std::vector<std::vector<std::int32_t>> lists;
        for (std::size_t j = 0; j < wallFaces.size(); ++j)
        {
            const std::int32_t c = faceCells[j];
            if (slot[std::size_t(c)] < 0)
            {
                slot[std::size_t(c)] = std::int32_t(cells.size());
                cells.push_back(c);
                lists.emplace_back();
            }
            lists[std::size_t(slot[std::size_t(c)])].push_back(wallFaces[j]);
        }
        std::vector<std::int32_t> off(1, 0), faces;
        for (const auto& l : lists)
        {
            faces.insert(faces.end(), l.begin(), l.end());
            off.push_back(std::int32_t(faces.size()));
        }
        m.nWallCells = std::int32_t(cells.size());
        m.wallCells.upload(cells);
        m.wallCellOff.upload(off);
        m.wallCellFaces.upload(faces);
        m.omegaWallPlan = buildSetValuesPlan(dm, mesh.owner, mesh.neighbour, faceCells);
    }

    // kOmegaSSTBase 建構子的 bound(k_, kMin)、bound(omega_, omegaMin)
    boundField(dm, m.k, m.c.kMin);
    boundField(dm, m.omega, m.c.omegaMin);
    return m;
}


void kOmegaSSTCorrectNut
(
    const DeviceMesh& dm,
    const LSVectors& ls,
    KOmegaSST& m,
    const VolField<double>& rho,
    const VolField<double>& mu,
    const VolField<vec3>& U
)
{
    VolField<double> nu;
    nuField(dm, mu, rho, nu);
    VolField<tens> gradU;
    gradByScheme(dm, ls, m.gradUScheme, U, gradU);
    VolField<double> S2;
    computeS2(dm, gradU, S2);
    correctNutS2(dm, m, rho, mu, nu, S2);
}


void kOmegaSSTCorrect
(
    const DeviceMesh& dm,
    const LusgsPlan& plan,
    const LSVectors& ls,
    KOmegaSST& m,
    const VolField<double>& rho,
    const VolField<vec3>& U,
    const VolField<double>& mu,
    const SurfField<double>& phi,
    const double* rPseudoDeltaT
)
{
    if (!m.turbulence)
    {
        return;
    }
    const std::size_t nC = std::size_t(dm.nCells);
    const std::size_t nI = std::size_t(dm.nInternalFaces);
    const std::size_t nB = std::size_t(dm.nFvBoundaryFaces());
    const SstConsts c = constsOf(m.c);

    VolField<double> nu;
    nuField(dm, mu, rho, nu);

    // divU = fvc::div(phi/interpolate(rho))（compressibleTurbulenceModel::phi()）
    DField<double> divU;
    {
        SurfField<double> rhof;
        rhof.in.allocate(nI);
        rhof.bnd.allocate(nB);
        interpolateLinear(dm, rho, rhof);
        SurfField<double> phiv;
        phiv.in.allocate(nI);
        phiv.bnd.allocate(nB);
        if (nI)
        {
            kDivideFaces<<<nBlocks(nI), BS>>>(nI, phi.in.cview().p[0], rhof.in.cview().p[0], phiv.in.view().p[0]);
        }
        if (nB)
        {
            kDivideFaces<<<nBlocks(nB), BS>>>(nB, phi.bnd.cview().p[0], rhof.bnd.cview().p[0], phiv.bnd.view().p[0]);
        }
        GNC_CUDA_LAUNCH_CHECK();
        surfaceIntegrate(dm, phiv, divU);
    }

    // gradU、S2、GbyNu0、G = nut*GbyNu0
    VolField<tens> gradU;
    gradByScheme(dm, ls, m.gradUScheme, U, gradU);
    VolField<double> S2;
    computeS2(dm, gradU, S2);
    DeviceArray<double> GbyNu0(nC), G(nC);
    if (nC)
    {
        kGbyNu0<<<nBlocks(nC), BS>>>(nC, gradU.in.cview(), m.nut.in.cview().p[0], GbyNu0.data(), G.data());
        GNC_CUDA_LAUNCH_CHECK();
    }

    // omega_.boundaryFieldRef().updateCoeffs()（omegaWallFunction 改寫牆邊格）
    omegaUpdateCoeffs(dm, m, rho, mu, U, phi, G.data());

    // CDkOmega、F1、F23
    VolField<vec3> gradK, gradOmega;
    gradByScheme(dm, ls, m.gradKScheme, m.k, gradK);
    gradByScheme(dm, ls, m.gradOmegaScheme, m.omega, gradOmega);
    VolField<double> CD;
    CD.in.allocate(nC);
    CD.bnd.allocate(nB);
    if (nC)
    {
        kCDkOmega<<<nBlocks(nC), BS>>>(nC, gradK.in.cview(), gradOmega.in.cview(), m.omega.in.cview().p[0], c.twoAlphaOmega2, CD.in.view().p[0]);
    }
    if (nB)
    {
        kCDkOmega<<<nBlocks(nB), BS>>>(nB, gradK.bnd.cview(), gradOmega.bnd.cview(), m.omega.bnd.cview().p[0], c.twoAlphaOmega2, CD.bnd.view().p[0]);
    }
    GNC_CUDA_LAUNCH_CHECK();
    VolField<double> F1, F2;
    blendingFunctions(dm, m, nu, &CD, &F1, F2);

    // omega 方程
    {
        DeviceArray<double> su(nC), susp1(nC), sp(nC), susp2(nC);
        if (nC)
        {
            kOmegaSources<<<nBlocks(nC), BS>>>
            (
                nC, c, rho.in.cview().p[0], m.omega.in.cview().p[0], F1.in.cview().p[0],
                F2.in.cview().p[0], S2.in.cview().p[0], GbyNu0.data(), CD.in.cview().p[0],
                divU.cview().p[0], su.data(), susp1.data(), sp.data(), susp2.data()
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
        VolField<double> gamma;
        gammaEff(dm, rho, F1, m.nut, nu, m.c.dAlphaOmega, m.c.alphaOmega2, "(rho*DomegaEff)", gamma);

        // 左邊：(ddt + div) - laplacian
        FvScalarMatrix eqn = rPseudoDeltaT
            ? fvmDdtBoundedDualTime(dm, rPseudoDeltaT, rho.in.cview().p[0], m.omega)
            : fvmEmpty(dm);     // steadyStateDdtScheme::fvmDdt：沒有係數的矩陣
        {
            const FvScalarMatrix divM = divMatrix
            (
                dm, phi, m.omega, gradOmega, m.omegaLimitedLinear, m.omegaLimitedLinearCoeff
            );
            addMatrix(dm, eqn, divM);
        }
        {
            const FvScalarMatrix lapM = fvmLaplacianGaussLinearCorrected
            (
                dm, gamma, m.omega, gradOmega, &phi, m.omegaLaplacian
            );
            subtractMatrix(dm, eqn, lapM);
        }
        // 右邊：((su - SuSp(susp1)) - Sp(sp)) - SuSp(susp2) + Qsas + omegaSource + fvOptions
        // （後三者為沒有係數的空矩陣，加一次與加三次逐位元相同）
        FvScalarMatrix rhs = fvmSuSp(dm, susp1.data(), m.omega);
        suMinusMatrix(dm, su.data(), rhs);
        {
            const FvScalarMatrix spM = fvmSp(dm, sp.data());
            subtractMatrix(dm, rhs, spM);
        }
        {
            const FvScalarMatrix suspM = fvmSuSp(dm, susp2.data(), m.omega);
            subtractMatrix(dm, rhs, suspM);
        }
        {
            const FvScalarMatrix empty = fvmEmpty(dm);
            addMatrix(dm, rhs, empty);
        }
        subtractMatrix(dm, eqn, rhs);

        if (m.omegaRelax)
        {
            relaxMatrix(dm, eqn, m.omega, m.omegaRelaxFactor);
        }
        // boundaryManipulate：omegaWallFunction::manipulateMatrix
        if (m.hasOmegaWall)
        {
            setValuesFromPsi(m.omegaWallPlan, eqn, m.omega);
        }
        m.lastSolveOmega = solveScalarMatrix(dm, plan, eqn, m.omega, m.omegaSolver, &phi);
        m.lastBoundedOmega = boundField(dm, m.omega, m.c.omegaMin);
    }

    // k 方程
    {
        // fvMatrix 建構時 k 的邊界 updateCoeffs（kLowReWallFunction 改寫牆面值），
        // 之後 laplacian 的非正交修正（與 limitedLinear 的 limiter）以新的邊界值重算
        // grad(k)。kEqn 的 fvm::ddt + fvm::div - fvm::laplacian 中，div 的 limiter
        // 在它自己的 fvMatrix 建構之前算；GCC 先算 laplacian（或 ddt），邊界已更新
        kUpdateCoeffs(dm, m, rho, mu, U, phi);
        VolField<vec3> gradK2;
        gradByScheme(dm, ls, m.gradKScheme, m.k, gradK2);

        DeviceArray<double> su(nC), susp(nC), sp(nC);
        if (nC)
        {
            kKSources<<<nBlocks(nC), BS>>>
            (
                nC, c, rho.in.cview().p[0], m.k.in.cview().p[0], m.omega.in.cview().p[0], G.data(),
                divU.cview().p[0], su.data(), susp.data(), sp.data()
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
        VolField<double> gamma;
        gammaEff(dm, rho, F1, m.nut, nu, m.c.dAlphaK, m.c.alphaK2, "(rho*DkEff)", gamma);

        FvScalarMatrix eqn = rPseudoDeltaT
            ? fvmDdtBoundedDualTime(dm, rPseudoDeltaT, rho.in.cview().p[0], m.k)
            : fvmEmpty(dm);     // steadyStateDdtScheme::fvmDdt：沒有係數的矩陣
        {
            const FvScalarMatrix divM = divMatrix
            (
                dm, phi, m.k, gradK2, m.kLimitedLinear, m.kLimitedLinearCoeff
            );
            addMatrix(dm, eqn, divM);
        }
        {
            const FvScalarMatrix lapM = fvmLaplacianGaussLinearCorrected
            (
                dm, gamma, m.k, gradK2, &phi, m.kLaplacian
            );
            subtractMatrix(dm, eqn, lapM);
        }
        // 右邊：((su - SuSp(susp)) - Sp(sp)) + kSource + fvOptions
        FvScalarMatrix rhs = fvmSuSp(dm, susp.data(), m.k);
        suMinusMatrix(dm, su.data(), rhs);
        {
            const FvScalarMatrix spM = fvmSp(dm, sp.data());
            subtractMatrix(dm, rhs, spM);
        }
        {
            const FvScalarMatrix empty = fvmEmpty(dm);
            addMatrix(dm, rhs, empty);
        }
        subtractMatrix(dm, eqn, rhs);

        if (m.kRelax)
        {
            relaxMatrix(dm, eqn, m.k, m.kRelaxFactor);
        }
        m.lastSolveK = solveScalarMatrix(dm, plan, eqn, m.k, m.kSolver, &phi);
        m.lastBoundedK = boundField(dm, m.k, m.c.kMin);
    }

    correctNutS2(dm, m, rho, mu, nu, S2);
}

} // namespace gnc
