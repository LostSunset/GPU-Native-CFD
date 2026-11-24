// GPU-Native-CFD，GPL-3.0-or-later。見 hisaViscous.cuh。

#include "solvers/gncHisa/hisaViscous.cuh"

#include "fields/boundaryConditions.cuh"
#include "io/dictionary.hpp"
#include "io/foamTokenizer.hpp"

#include <sstream>
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

CView<vec3> meshVec(const DeviceVec3Array& a)
{
    CView<vec3> v;
    v.p[0] = a.x.data();
    v.p[1] = a.y.data();
    v.p[2] = a.z.data();
    return v;
}


// muEff = mut + mu，mut = rho*nut；Stokes 的 nut 為 0 的場（nut 傳 nullptr）
__global__ void kMuEff
(
    const std::int32_t n,
    const double* __restrict__ rho,
    const double* __restrict__ nut,
    const double* __restrict__ mu,
    double* __restrict__ out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const double mut = rho[i]*(nut ? nut[i] : 0.0);
        out[i] = mut + mu[i];
    }
}


// alphaEff：laminar 為 alphahe = CpByCpv*alpha；RAS（EddyDiffusivity）為
// CpByCpv*(alpha + alphat)（heThermo.C:869）
__global__ void kAlphaEff
(
    const std::int32_t n,
    const ThermoParams P,
    const double* __restrict__ p,
    const double* __restrict__ T,
    const double* __restrict__ alpha,
    const double* __restrict__ alphat,
    double* __restrict__ out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const double g = thermo::gamma(P, p[i], T[i]);
        out[i] = alphat ? g*(alpha[i] + alphat[i]) : g*alpha[i];
    }
}


// tauMC = muEff*dev2(T(grad(U)))。pairedZx：內部場照參考端的原地轉置
// （FoamCompat::transposeAliasing），偶數格且有下一格時 T.zx 取原本的 zx
__global__ void kTauMC
(
    const std::int32_t n,
    const bool pairedZx,
    const double* __restrict__ muEff,
    const CView<tens> gradU,
    const DView<tens> out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const tens g = gradU[i];
        tens t = transpose(g);
        if (pairedZx && (i % 2) == 0 && i + 1 < n)
        {
            t.zx = g.zx;
        }
        out.set(i, muEff[i]*dev2(t));
    }
}


// eCalc = rhoE/rho - 0.5*magSqr(U)
__global__ void kECalc
(
    const std::int32_t n,
    const double* __restrict__ rhoE,
    const double* __restrict__ rho,
    const CView<vec3> U,
    double* __restrict__ out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = rhoE[i]/rho[i] - 0.5*magSqr(U[i]);
    }
}


// corrected snGrad 的內部面：nonOrthDeltaCoeffs*(N - P) + 修正
__global__ void kSnGradCorrectedInternal
(
    const std::int32_t nIF,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ ndc,
    const CView<vec3> U,
    const CView<vec3> corr,
    const DView<vec3> out
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f < nIF)
    {
        const vec3 sn = ndc[f]*(U[neighbour[f]] - U[owner[f]]);
        out.set(f, sn + corr[f]);
    }
}


// corrected snGrad 的邊界面：patch 的 snGrad() + 修正
__global__ void kSnGradCorrectedBoundary
(
    const std::int32_t nB,
    const CView<vec3> snBC,
    const CView<vec3> corr,
    const DView<vec3> out
)
{
    const std::int32_t k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k < nB)
    {
        out.set(k, snBC[k] + corr[k]);
    }
}


// sigmaDotU = (muEff_f*snGrad(U) + (Sf/magSf & tauMC_f)) & Up。
// faces 為 nullptr 時 i 即面編號（內部面），否則 faces[i]（邊界面）
__global__ void kSigmaDotU
(
    const std::int32_t n,
    const std::int32_t* __restrict__ faces,
    const CView<vec3> Sf,
    const double* __restrict__ magSf,
    const double* __restrict__ muEffF,
    const CView<vec3> snGradU,
    const CView<tens> tauF,
    const CView<vec3> Up,
    double* __restrict__ out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const std::int32_t f = faces ? faces[i] : i;
    const vec3 nf = Sf[f]/magSf[f];
    const vec3 sigma = muEffF[i]*snGradU[i] + dot(nf, tauF[i]);
    out[i] = dot(sigma, Up[i]);
}


template<class T>
__global__ void kAddTo(const std::int32_t n, const DView<T> a, const CView<T> b)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        a.set(i, a[i] + b[i]);
    }
}


// 對內部與邊界值做 a += b（GeometricField::operator+=）
template<class T>
void addField(VolField<T>& a, const VolField<T>& b)
{
    const std::int32_t nC = std::int32_t(a.in.size());
    const std::int32_t nB = std::int32_t(a.bnd.size());
    if (nC)
    {
        kAddTo<T><<<nBlocks(nC), BS>>>(nC, a.in.view(), b.in.cview());
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (nB)
    {
        kAddTo<T><<<nBlocks(nB), BS>>>(nB, a.bnd.view(), b.bnd.cview());
        GNC_CUDA_LAUNCH_CHECK();
    }
}


// 逐格（內部與邊界各呼叫一次）的純量場運算
template<class Launch>
void cellsAndBoundary(const DeviceMesh& dm, Launch launch)
{
    const std::int32_t nC = dm.nCells;
    const std::int32_t nB = dm.nFvBoundaryFaces();
    if (nC)
    {
        launch(false, nC);
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (nB)
    {
        launch(true, nB);
        GNC_CUDA_LAUNCH_CHECK();
    }
}


// "Gauss linear compact [gradScheme]"：回傳梯度方案的名稱
std::string compactGradName(const std::string& scheme, const std::string& fieldName)
{
    std::istringstream is(scheme);
    std::string a, b, c, d, extra;
    is >> a >> b >> c;
    if (a != "Gauss" || b != "linear" || c != "compact")
    {
        throw std::runtime_error
        (
            "laplacian scheme '" + scheme + "' is not supported (Gauss linear compact)"
        );
    }
    if (is >> d)
    {
        if (is >> extra)
        {
            throw std::runtime_error("laplacian scheme '" + scheme + "' is malformed");
        }
        return d;
    }
    return "grad(" + fieldName + ")";
}


void requireScheme(const std::string& what, const std::string& got, const std::string& want)
{
    if (got != want)
    {
        throw std::runtime_error
        (
            what + " '" + got + "' is not supported (" + want + ")"
        );
    }
}


// fvc::div 與 fvc::laplacian 的結果場：extrapolatedCalculated，
// 內部值由 fill 填入，再 correctBoundaryConditions
template<class T, class Fill>
VolField<T> extrapolatedResult(const DeviceMesh& dm, const std::string& name, Fill fill)
{
    VolField<T> f = makeVolField<T>(dm, name, Bc::extrapolatedCalculated);
    fill(f.in);
    evaluateBCs(dm, f, nullptr);
    return f;
}

} // namespace


TurbulenceModelKind readTurbulenceModelKind(const std::string& path)
{
    FoamTokenizer t(path);
    const Dictionary d = Dictionary::parse(t, path, false);
    const std::string sim = d.getWord("simulationType");
    if (sim == "RAS")
    {
        const std::string model = d.subDict("RAS").getWord("RASModel");
        if (model == "kOmegaSST")
        {
            return TurbulenceModelKind::kOmegaSST;
        }
        if (model != "SpalartAllmaras")
        {
            throw std::runtime_error
            (
                path + ": RASModel " + model + " is not supported (SpalartAllmaras, kOmegaSST)"
            );
        }
        return TurbulenceModelKind::spalartAllmaras;
    }
    if (sim != "laminar")
    {
        throw std::runtime_error
        (
            path + ": simulationType " + sim + " is not supported (laminar or RAS)"
        );
    }
    const Dictionary::Entry* e = d.find("laminar");
    if (e && e->dict)
    {
        const Dictionary& l = *e->dict;
        const std::string model =
            l.found("model") ? l.getWord("model") : l.getWordOrDefault("laminarModel", "");
        if (model != "Stokes")
        {
            throw std::runtime_error
            (
                path + ": laminar model '" + model + "' is not supported (Stokes only)"
            );
        }
    }
    return TurbulenceModelKind::laminarStokes;
}


void addLaminarViscousResiduals
(
    const DeviceMesh& dm,
    const LSVectors& ls,
    const FvSchemes& schemes,
    const HisaState& s,
    const HisaFlux& flux,
    const FoamCompat& compat,
    HisaResiduals& res,
    VolField<tens>& tauMC
)
{
    requireScheme("interpolate(muEff)", schemes.interpolation("interpolate(muEff)"), "linear");
    requireScheme("interpolate(tauMC)", schemes.interpolation("interpolate(tauMC)"), "linear");
    requireScheme("div(tauMC)", schemes.div("div(tauMC)"), "Gauss linear");
    requireScheme("snGrad(U)", schemes.snGrad("snGrad(U)"), "corrected");
    const std::string lapUGrad =
        compactGradName(schemes.laplacian("laplacian(muEff,U)"), "U");
    const std::string lapEGrad =
        compactGradName(schemes.laplacian("laplacian(alphaEff,e)"), "eCalc");

    const PsiThermoFields& th = s.thermo;
    const std::int32_t nIF = dm.nInternalFaces;
    const std::int32_t nB = dm.nFvBoundaryFaces();

    // muEff、alphaEff（calculated 邊界）
    VolField<double> muEff = makeVolField<double>(dm, "muEff", Bc::calculated);
    VolField<double> alphaEff = makeVolField<double>(dm, "alphaEff", Bc::calculated);
    cellsAndBoundary(dm, [&](const bool b, const std::int32_t n)
    {
        const auto& r = b ? s.rho.bnd : s.rho.in;
        kMuEff<<<nBlocks(n), BS>>>
        (
            n, r.cview().p[0],
            s.nut ? (b ? s.nut->bnd : s.nut->in).cview().p[0] : nullptr,
            (b ? th.mu.bnd : th.mu.in).cview().p[0],
            (b ? muEff.bnd : muEff.in).view().p[0]
        );
    });
    cellsAndBoundary(dm, [&](const bool b, const std::int32_t n)
    {
        kAlphaEff<<<nBlocks(n), BS>>>
        (
            n, th.params, (b ? th.p.bnd : th.p.in).cview().p[0],
            (b ? th.T.bnd : th.T.in).cview().p[0],
            (b ? th.alpha.bnd : th.alpha.in).cview().p[0],
            s.alphat ? (b ? s.alphat->bnd : s.alphat->in).cview().p[0] : nullptr,
            (b ? alphaEff.bnd : alphaEff.in).view().p[0]
        );
    });

    // grad(U) 與 tauMC
    const std::string gradUScheme = schemes.grad("grad(U)");
    VolField<tens> gradU;
    gradByScheme(dm, ls, gradUScheme, s.U, gradU);
    tauMC = makeVolField<tens>(dm, "tauMC", Bc::calculated);
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        tauMC.bc[pi].type = gradU.bc[pi].type;
    }
    cellsAndBoundary(dm, [&](const bool b, const std::int32_t n)
    {
        kTauMC<<<nBlocks(n), BS>>>
        (
            n, compat.transposeAliasing && !b, (b ? muEff.bnd : muEff.in).cview().p[0],
            (b ? gradU.bnd : gradU.in).cview(), (b ? tauMC.bnd : tauMC.in).view()
        );
    });
    finishAssignment(dm, tauMC);

    // rhoUR += laplacian(muEff, U)；compact 的梯度方案與 grad(U) 相同時沿用
    VolField<tens> gradUCompact;
    const VolField<tens>* gUc = &gradU;
    if (schemes.grad(lapUGrad) != gradUScheme)
    {
        gradByScheme(dm, ls, schemes.grad(lapUGrad), s.U, gradUCompact);
        gUc = &gradUCompact;
    }
    const VolField<vec3> lapU = extrapolatedResult<vec3>
    (
        dm, "laplacian(muEff,U)",
        [&](DField<vec3>& out) { laplacianCompact(dm, muEff, s.U, *gUc, out); }
    );
    addField(res.rhoUR, lapU);

    // rhoUR += div(tauMC)
    const VolField<vec3> divTau = extrapolatedResult<vec3>
    (
        dm, "div(tauMC)",
        [&](DField<vec3>& out) { divGaussLinear(dm, tauMC, out); }
    );
    addField(res.rhoUR, divTau);

    // snGrad(U)，corrected
    SurfField<vec3> snGradU;
    {
        const SurfField<vec3> corrVecs = meshSurfVector(dm, dm.nonOrthCorrectionVectors);
        SurfField<vec3> corr;
        dotInterpolateLinear(dm, corrVecs, gradU, corr);
        snGradU.in.allocate(std::size_t(nIF));
        snGradU.bnd.allocate(std::size_t(nB));
        if (nIF)
        {
            kSnGradCorrectedInternal<<<nBlocks(nIF), BS>>>
            (
                nIF, dm.owner.data(), dm.neighbour.data(), dm.nonOrthDeltaCoeffs.data(),
                s.U.in.cview(), corr.in.cview(), snGradU.in.view()
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
        if (nB)
        {
            DField<vec3> snBC;
            snGradBCs(dm, s.U, snBC);
            kSnGradCorrectedBoundary<<<nBlocks(nB), BS>>>
            (
                nB, snBC.cview(), corr.bnd.cview(), snGradU.bnd.view()
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
    }

    // sigmaDotU，rhoER += div(sigmaDotU)
    SurfField<double> muEffF;
    interpolateLinear(dm, muEff, muEffF);
    SurfField<tens> tauF;
    interpolateLinear(dm, tauMC, tauF);
    SurfField<double> sigmaDotU;
    sigmaDotU.name = "sigmaDotU";
    sigmaDotU.in.allocate(std::size_t(nIF));
    sigmaDotU.bnd.allocate(std::size_t(nB));
    if (nIF)
    {
        kSigmaDotU<<<nBlocks(nIF), BS>>>
        (
            nIF, nullptr, meshVec(dm.Sf), dm.magSf.data(), muEffF.in.cview().p[0],
            snGradU.in.cview(), tauF.in.cview(), flux.Up.in.cview(),
            sigmaDotU.in.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (nB)
    {
        kSigmaDotU<<<nBlocks(nB), BS>>>
        (
            nB, dm.fvBoundaryFaces.data(), meshVec(dm.Sf), dm.magSf.data(),
            muEffF.bnd.cview().p[0], snGradU.bnd.cview(), tauF.bnd.cview(),
            flux.Up.bnd.cview(), sigmaDotU.bnd.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    const VolField<double> divSigma = extrapolatedResult<double>
    (
        dm, "div(sigmaDotU)",
        [&](DField<double>& out) { surfaceIntegrate(dm, sigmaDotU, out); }
    );
    addField(res.rhoER, divSigma);

    // eCalc，rhoER += laplacian(alphaEff, eCalc)
    VolField<double> eCalc = makeVolField<double>(dm, "eCalc", Bc::calculated);
    cellsAndBoundary(dm, [&](const bool b, const std::int32_t n)
    {
        kECalc<<<nBlocks(n), BS>>>
        (
            n, (b ? s.rhoE.bnd : s.rhoE.in).cview().p[0],
            (b ? s.rho.bnd : s.rho.in).cview().p[0], (b ? s.U.bnd : s.U.in).cview(),
            (b ? eCalc.bnd : eCalc.in).view().p[0]
        );
    });
    VolField<vec3> gradE;
    gradByScheme(dm, ls, schemes.grad(lapEGrad), eCalc, gradE);
    const VolField<double> lapE = extrapolatedResult<double>
    (
        dm, "laplacian(alphaEff,eCalc)",
        [&](DField<double>& out) { laplacianCompact(dm, alphaEff, eCalc, gradE, out); }
    );
    addField(res.rhoER, lapE);

    GNC_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace gnc
