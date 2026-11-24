// GPU-Native-CFD，GPL-3.0-or-later。見 rhoSimpleSolver.cuh。

#include "solvers/gncRhoSimple/rhoSimpleSolver.cuh"

#include "core/reduce.cuh"
#include "fields/boundaryConditions.cuh"
#include "io/dictionary.hpp"
#include "io/fieldIO.hpp"
#include "io/foamTokenizer.hpp"
#include "mesh/cellFaceGather.hpp"

#include <chrono>
#include <cstdio>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>

namespace gnc
{

namespace
{

constexpr int BS = 256;
constexpr double GREAT = 1.0e+15;

int nBlocks(const std::size_t n)
{
    return int((n + BS - 1)/BS);
}

// profile 時：同步後回傳現在的時間
struct Stopwatch
{
    bool on;
    std::chrono::steady_clock::time_point t0;

    explicit Stopwatch(const bool p)
    :
        on(p)
    {
        if (on)
        {
            GNC_CUDA_CHECK(cudaDeviceSynchronize());
            t0 = std::chrono::steady_clock::now();
        }
    }

    void add(double& acc)
    {
        if (on)
        {
            GNC_CUDA_CHECK(cudaDeviceSynchronize());
            const auto t1 = std::chrono::steady_clock::now();
            acc += std::chrono::duration<double>(t1 - t0).count();
            t0 = t1;
        }
    }
};

CView<vec3> vecView(const DeviceVec3Array& a)
{
    CView<vec3> v;
    v.p[0] = a.x.data();
    v.p[1] = a.y.data();
    v.p[2] = a.z.data();
    return v;
}

// 內部格與有限體積邊界面各呼叫一次
template<class F>
void cellsAndBoundary(const DeviceMesh& dm, F&& launch)
{
    if (dm.nCells)
    {
        launch(false, dm.nCells);
    }
    if (dm.nFvBoundaryFaces())
    {
        launch(true, dm.nFvBoundaryFaces());
    }
    GNC_CUDA_LAUNCH_CHECK();
}

template<class T>
DField<T>& part(VolField<T>& f, const bool b)
{
    return b ? f.bnd : f.in;
}

template<class T>
const DField<T>& part(const VolField<T>& f, const bool b)
{
    return b ? f.bnd : f.in;
}


// out = a*b（psiThermo::rho() = p*psi、rho*rAU）
__global__ void kMul(const std::int32_t n, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = a[i]*b[i];
    }
}

// out = s*v（scalar*vector）
__global__ void kScaleVec(const std::int32_t n, const double* __restrict__ s, const CView<vec3> v, const DView<vec3> out)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out.set(i, s[i]*v[i]);
    }
}

// out = 1.0/a
__global__ void kReciprocal(const std::int32_t n, const double* __restrict__ a, double* __restrict__ out)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = 1.0/a[i];
    }
}

// (rho*nuEff)：rho*(nut + mu/rho)（linearViscousStress 的 alpha*rho*nuEff()，
// nuEff = nut + nu、nu = mu/rho）
__global__ void kRhoNuEff
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
        const double nu = mu[i]/rho[i];
        out[i] = rho[i]*(nut[i] + nu);
    }
}

// (rho*nuEff)*dev2(T(grad(U)))
__global__ void kTauEff
(
    const std::int32_t n,
    const double* __restrict__ gamma,
    const CView<tens> gradU,
    const DView<tens> out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out.set(i, gamma[i]*dev2(transpose(gradU[i])));
    }
}

// -v
__global__ void kNegVec(const std::int32_t n, const CView<vec3> v, const DView<vec3> out)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const vec3 a = v[i];
        out.set(i, vec3{-a.x, -a.y, -a.z});
    }
}

// Ekp = 0.5*magSqr(U) + p/rho
__global__ void kEkp
(
    const std::int32_t n,
    const CView<vec3> U,
    const double* __restrict__ p,
    const double* __restrict__ rho,
    double* __restrict__ out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = 0.5*magSqr(U[i]) + p[i]/rho[i];
    }
}

// heThermo::alphaEff(alphat) = CpByCpv*(alpha + alphat)
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
        out[i] = thermo::gamma(P, p[i], T[i])*(alpha[i] + alphat[i]);
    }
}

// boundedConvectionScheme::fvcDiv（Gauss linearUpwind）的面通量：
// phi*((w*(vf[P] - vf[N]) + vf[N]) + corr)，w = pos0(phi)，
// corr = (Cf - C[c]) & grad[c]，c = phi > 0 ? P : N
__global__ void kLinearUpwindFlux
(
    const std::int32_t nIF,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ phi,
    const CView<vec3> Cf,
    const CView<vec3> C,
    const double* __restrict__ vf,
    const CView<vec3> grad,
    double* __restrict__ out
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIF)
    {
        return;
    }
    const std::int32_t P = owner[f];
    const std::int32_t N = neighbour[f];
    const double w = (phi[f] >= 0.0) ? 1.0 : 0.0;
    const double interp = w*(vf[P] - vf[N]) + vf[N];
    const std::int32_t c = (phi[f] > 0) ? P : N;
    const double corr = dot(Cf[f] - C[c], grad[c]);
    out[f] = phi[f]*(interp + corr);
}

// 邊界：phi_b*(vf_b + 0)（修正的邊界為 0）
__global__ void kBoundaryFluxPlusZero
(
    const std::int32_t n,
    const double* __restrict__ phi,
    const double* __restrict__ vfB,
    double* __restrict__ out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = phi[i]*(vfB[i] + 0.0);
    }
}

// conv - surfaceIntegrate(phi)*vf
__global__ void kBoundedExplicit
(
    const std::int32_t n,
    const double* __restrict__ conv,
    const double* __restrict__ sInt,
    const double* __restrict__ vf,
    double* __restrict__ out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = conv[i] - sInt[i]*vf[i];
    }
}

// mixedEnergyFvPatchScalarField::updateCoeffs（Tw.evaluate() 之後）
__global__ void kMixedEnergyCoeffs
(
    const std::int32_t start,
    const std::int32_t size,
    const std::int32_t* __restrict__ fvbFaces,
    const ThermoParams P,
    const double* __restrict__ pB,
    const double* __restrict__ TB,
    const double* __restrict__ TrefValue,
    const double* __restrict__ TrefGrad,
    const double* __restrict__ Tvf,
    const double* __restrict__ dc,
    double* __restrict__ vf,
    double* __restrict__ refValue,
    double* __restrict__ refGrad
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= size)
    {
        return;
    }
    const std::int32_t k = start + i;
    const std::int32_t f = fvbFaces[k];
    vf[i] = Tvf[i];
    refValue[i] = thermo::Es(P, pB[k], TrefValue[i]);
    refGrad[i] =
        thermo::Cv(P, pB[k], TB[k])*TrefGrad[i]
      + dc[f]*(thermo::Es(P, pB[k], TB[k]) - thermo::Es(P, pB[k], TB[k]));
}

// gradientEnergyFvPatchScalarField::updateCoeffs
__global__ void kGradientEnergyCoeffs
(
    const std::int32_t start,
    const std::int32_t size,
    const std::int32_t* __restrict__ fvbFaces,
    const ThermoParams P,
    const double* __restrict__ pB,
    const double* __restrict__ TB,
    const double* __restrict__ snT,
    const double* __restrict__ dc,
    double* __restrict__ gradient
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= size)
    {
        return;
    }
    const std::int32_t k = start + i;
    const std::int32_t f = fvbFaces[k];
    gradient[i] =
        thermo::Cv(P, pB[k], TB[k])*snT[k]
      + dc[f]*(thermo::Es(P, pB[k], TB[k]) - thermo::Es(P, pB[k], TB[k]));
}

// fixedEnergyFvPatchScalarField::updateCoeffs：he_b == HE(p_b, T_b)
__global__ void kFixedEnergy
(
    const std::int32_t start,
    const std::int32_t size,
    const ThermoParams P,
    const double* __restrict__ pB,
    const double* __restrict__ TB,
    double* __restrict__ heB
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < size)
    {
        const std::int32_t k = start + i;
        heB[k] = thermo::Es(P, pB[k], TB[k]);
    }
}

// limitTemperature::correct(he)：he < HE(p, Tmin) 時設為下限，否則 he > HE(p, Tmax)
// 時設為上限
__global__ void kLimitHe
(
    const std::int32_t n,
    const ThermoParams P,
    const double* __restrict__ p,
    const double Tmin,
    const double Tmax,
    double* __restrict__ he,
    int* __restrict__ counts       // [0] 下限、[1] 上限
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const double heMin = thermo::Es(P, p[i], Tmin);
    const double heMax = thermo::Es(P, p[i], Tmax);
    if (he[i] < heMin)
    {
        he[i] = heMin;
        atomicAdd(&counts[0], 1);
    }
    else if (he[i] > heMax)
    {
        he[i] = heMax;
        atomicAdd(&counts[1], 1);
    }
}

// a = prev + alpha*(a - prev)（GeometricField::relax，operator== 連邊界一起）
__global__ void kRelaxField
(
    const std::int32_t n,
    const double* __restrict__ prev,
    const double alpha,
    double* __restrict__ a
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[i] = prev[i] + alpha*(a[i] - prev[i]);
    }
}

// U = HbyA - rAU*grad(p)（內部格）
__global__ void kUCorrect
(
    const std::int32_t n,
    const CView<vec3> HbyA,
    const double* __restrict__ rAU,
    const CView<vec3> gradP,
    const DView<vec3> U
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        U.set(i, HbyA[i] - rAU[i]*gradP[i]);
    }
}

// phi = phiHbyA + flux
__global__ void kAdd(const std::int32_t n, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = a[i] + b[i];
    }
}

// min(p, pMax)、max(p, pMin)
__global__ void kClamp(const std::int32_t n, const double v, const bool upper, double* __restrict__ a)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[i] = upper ? foamMin(a[i], v) : foamMax(a[i], v);
    }
}

// 一個 patch 的邊界值複製（constrainHbyA 的 HbyA_b = U_b）
__global__ void kCopyVecRange(const std::int32_t start, const std::int32_t n, const CView<vec3> src, const DView<vec3> dst)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        dst.set(start + i, src[start + i]);
    }
}


void copyVol(const VolField<double>& s, VolField<double>& d)
{
    d.name = s.name;
    {
        ArrayBatch b;
        d.in.copyFrom(s.in, &b);
        d.bnd.copyFrom(s.bnd, &b);
    }
}

void requireScheme(const std::string& got, const std::string& want, const std::string& what)
{
    if (trimmedScheme(got) != want)
    {
        throw std::runtime_error("fvSchemes " + what + " '" + got + "' is not supported (" + want + ")");
    }
}

// "bounded Gauss linearUpwind <grad>" 的 <grad>
std::string linearUpwindGrad(const FvSchemes& schemes, const std::string& name)
{
    std::istringstream is(trimmedScheme(schemes.div(name)));
    std::string a, b, c, g, extra;
    is >> a >> b >> c >> g;
    if (a != "bounded" || b != "Gauss" || c != "linearUpwind" || g.empty() || (is >> extra))
    {
        throw std::runtime_error
        (
            "fvSchemes div " + name + " '" + schemes.div(name)
          + "' is not supported (bounded Gauss linearUpwind <grad>)"
        );
    }
    return trimmedScheme(schemes.grad(g));
}

VolField<double> calculated(const DeviceMesh& dm, const std::string& name)
{
    return makeVolField<double>(dm, name, Bc::calculated);
}

// fvOptions（已展開）：只支援 limitTemperature（selectionMode all）
void readFvOptions(const std::string& path, bool& limitT, double& Tmin, double& Tmax)
{
    limitT = false;
    struct stat st{};
    if (::stat(path.c_str(), &st) != 0)
    {
        return;
    }
    FoamTokenizer t(path);
    const Dictionary d = Dictionary::parse(t, path, false);
    for (const Dictionary::Entry& e : d.entries())
    {
        if (!e.dict || e.key == "FoamFile")
        {
            continue;
        }
        const Dictionary& o = *e.dict;
        const std::string type = o.getWord("type");
        if (type != "limitTemperature")
        {
            throw std::runtime_error(path + ": fvOption " + e.key + " type " + type + " is not supported");
        }
        if (o.getWordOrDefault("selectionMode", "") != "all")
        {
            throw std::runtime_error(path + ": limitTemperature needs selectionMode all");
        }
        if (o.found("active") && !parseSwitch(o.getWord("active"), path + " active"))
        {
            continue;
        }
        if (limitT)
        {
            throw std::runtime_error(path + ": more than one limitTemperature is not supported");
        }
        limitT = true;
        Tmin = o.getScalar("min");
        Tmax = o.getScalar("max");
    }
}

} // namespace


SimpleControls readSimpleControls(const std::string& path)
{
    FoamTokenizer t(path);
    const Dictionary d = Dictionary::parse(t, path, false);
    const Dictionary& s = d.subDict("SIMPLE");
    SimpleControls c;
    if (s.found("nNonOrthogonalCorrectors"))
    {
        c.nNonOrthCorr = int(s.getLabel("nNonOrthogonalCorrectors"));
    }
    if (s.found("consistent"))
    {
        c.consistent = parseSwitch(s.getWord("consistent"), path + " SIMPLE/consistent");
    }
    if (s.found("transonic"))
    {
        c.transonic = parseSwitch(s.getWord("transonic"), path + " SIMPLE/transonic");
    }
    if (s.found("pMinFactor"))
    {
        c.hasPMinFactor = true;
        c.pMinFactor = s.getScalar("pMinFactor");
    }
    if (s.found("pMaxFactor"))
    {
        c.hasPMaxFactor = true;
        c.pMaxFactor = s.getScalar("pMaxFactor");
    }
    if (s.found("pMin") || s.found("pMax") || s.found("rhoMin") || s.found("rhoMax"))
    {
        throw std::runtime_error(path + ": SIMPLE pMin/pMax/rhoMin/rhoMax are not supported (use pMinFactor/pMaxFactor)");
    }
    if (const Dictionary::Entry* rc = s.find("residualControl"))
    {
        if (!rc->dict)
        {
            throw std::runtime_error(path + ": SIMPLE/residualControl must be a dictionary");
        }
        for (const Dictionary::Entry& e : rc->dict->entries())
        {
            if (e.dict || e.stream.size() != 1)
            {
                throw std::runtime_error(path + ": residualControl " + e.key + " must be a scalar");
            }
            SimpleControls::Residual r;
            r.name = e.key;
            r.isPattern = e.isPattern;
            if (!parseScalar(e.stream[0].text, r.absTol))
            {
                throw std::runtime_error(path + ": residualControl " + e.key + ": invalid number");
            }
            c.residualControl.push_back(r);
        }
    }
    if (c.nNonOrthCorr != 0)
    {
        throw std::runtime_error(path + ": nNonOrthogonalCorrectors > 0 is not supported (N-023)");
    }
    if (c.consistent || c.transonic)
    {
        throw std::runtime_error(path + ": SIMPLE consistent/transonic are not supported (N-023)");
    }
    return c;
}


RhoSimpleSolver::RhoSimpleSolver(const std::string& caseDir, const std::string& expandedDir)
{
    mesh = readPolyMesh(caseDir + "/constant/polyMesh");
    const CellFaceGather gather = buildCellFaceGather(mesh);
    dm = DeviceMesh::upload(mesh, gather);
    computeGeometry(dm);
    ls_ = faceLeastSquaresVectors(dm);
    plan_ = buildLusgsPlan(mesh);

    // fvSchemes
    schemes = std::make_unique<FvSchemes>(expandedDir + "/fvSchemes");
    {
        std::istringstream is(schemes->ddt("default"));
        std::string w, last;
        while (is >> w)
        {
            last = w;
        }
        if (last != "steadyState")
        {
            throw std::runtime_error("ddtSchemes default must be steadyState (rhoSimpleFoam)");
        }
    }
    limitedUScheme = linearUpwindGrad(*schemes, "div(phi,U)");
    limitedEScheme = linearUpwindGrad(*schemes, "div(phi,e)");
    limitedEkpScheme = linearUpwindGrad(*schemes, "div(phi,Ekp)");
    requireScheme(schemes->div("div(((rho*nuEff)*dev2(T(grad(U)))))"), "Gauss linear", "div(((rho*nuEff)*dev2(T(grad(U)))))");
    requireScheme(schemes->laplacian("laplacian((rho*nuEff),U)"), "Gauss linear corrected", "laplacian((rho*nuEff),U)");
    requireScheme(schemes->laplacian("laplacian(alphaEff,e)"), "Gauss linear corrected", "laplacian(alphaEff,e)");
    requireScheme(schemes->laplacian("laplacian(rhorAUf,p)"), "Gauss linear corrected", "laplacian(rhorAUf,p)");
    requireScheme(schemes->interpolation("interpolate((rho*rAU))"), "linear", "interpolate((rho*rAU))");
    requireScheme(schemes->interpolation("flux(HbyA)"), "linear", "flux(HbyA)");
    requireScheme(schemes->interpolation("interpolate(rho)"), "linear", "interpolate(rho)");
    gradUScheme = trimmedScheme(schemes->grad("grad(U)"));
    gradPScheme = trimmedScheme(schemes->grad("grad(p)"));
    gradEScheme = trimmedScheme(schemes->grad("grad(e)"));

    // fvSolution
    const std::string fvSolution = expandedDir + "/fvSolution";
    simple = readSimpleControls(fvSolution);
    pSolver = readSmoothSolverControls(fvSolution, "p");
    USolver = readSmoothSolverControls(fvSolution, "U");
    eSolver = readSmoothSolverControls(fvSolution, "e");
    URelax = readEquationRelax(fvSolution, "U");
    eRelax = readEquationRelax(fvSolution, "e");
    pFieldRelax = readFieldRelax(fvSolution, "p");
    rhoFieldRelax = readFieldRelax(fvSolution, "rho");
    if (pSolver.solver != "PCG" && pSolver.solver != "PBiCGStab" && pSolver.solver != "GAMG")
    {
        throw std::runtime_error("p solver must be PCG (DIC), PBiCGStab (DILU) or GAMG (-fast)");
    }
    if (USolver.solver != "PBiCGStab")
    {
        throw std::runtime_error("U solver must be PBiCGStab (DILU)");
    }
    time = readTimeControls(expandedDir + "/controlDict");
    readFvOptions(expandedDir + "/fvOptions", limitT, Tmin, Tmax);

    // createFields：thermo
    {
        const FoamConstants c = openfoamConstants();
        const ThermoParams P = readThermoParams(expandedDir + "/thermophysicalProperties", c);
        const HostField hp = readField(expandedDir + "/p", mesh);
        const HostField hT = readField(expandedDir + "/T", mesh);
        pDimensions = hp.dimensions;
        TDimensions = hT.dimensions;
        thermo = constructPsiThermo
        (
            dm, P, uploadVolField<double>(dm, hp), uploadVolField<double>(dm, hT)
        );
    }
    // rho = thermo.rho()（calculated 邊界）
    rho = calculated(dm, "rho");
    cellsAndBoundary(dm, [&](const bool b, const std::int32_t n)
    {
        kMul<<<nBlocks(n), BS>>>
        (
            n, part(thermo.p, b).cview().p[0], part(thermo.psi, b).cview().p[0],
            part(rho, b).view().p[0]
        );
    });
    // U
    {
        const HostField hU = readField(expandedDir + "/U", mesh);
        UDimensions = hU.dimensions;
        U = uploadVolField<vec3>(dm, hU);
    }
    // phi = linearInterpolate(rho*U) & Sf
    {
        VolField<vec3> rhoU = makeVolField<vec3>(dm, "rhoU", Bc::calculated);
        cellsAndBoundary(dm, [&](const bool b, const std::int32_t n)
        {
            kScaleVec<<<nBlocks(n), BS>>>
            (
                n, part(rho, b).cview().p[0], part(U, b).cview(), part(rhoU, b).view()
            );
        });
        const SurfField<vec3> Sf = meshSurfVector(dm, dm.Sf);
        dotInterpolateLinear(dm, Sf, rhoU, phi);
        phi.name = "phi";
    }

    // pressureControl：fixesValue 的 p patch 的 max／min
    {
        bool pLimits = false;
        double pmx = -GREAT;
        double pmn = GREAT;
        for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
        {
            const std::int32_t n = dm.fvPatchSize[pi];
            if (!fixesValue(thermo.p.bc[pi].type))
            {
                continue;
            }
            pLimits = true;
            if (n)
            {
                const double* b = thermo.p.bnd.cview().p[0] + dm.fvPatchStart[pi];
                pmx = foamMax(pmx, maxValue(b, std::size_t(n)));
                pmn = foamMin(pmn, minValue(b, std::size_t(n)));
            }
        }
        if (simple.hasPMaxFactor || simple.hasPMinFactor)
        {
            if (!pLimits)
            {
                throw std::runtime_error("pMaxFactor/pMinFactor need a fixesValue p patch");
            }
        }
        if (simple.hasPMaxFactor)
        {
            limitMaxP = true;
            pMax = pmx*simple.pMaxFactor;
        }
        if (simple.hasPMinFactor)
        {
            limitMinP = true;
            pMin = pmn*simple.pMinFactor;
        }
        if (!pLimits)
        {
            throw std::runtime_error("p needs a reference value (closed domain) - not supported");
        }
    }

    // 紊流
    {
        FoamTokenizer t(expandedDir + "/turbulenceProperties");
        const Dictionary d = Dictionary::parse(t, expandedDir + "/turbulenceProperties", false);
        const std::string sim = d.getWord("simulationType");
        if (sim != "RAS" || d.subDict("RAS").getWord("RASModel") != "kOmegaSST")
        {
            throw std::runtime_error("turbulenceProperties: only RAS kOmegaSST is supported");
        }
    }
    sst = std::make_unique<KOmegaSST>(createKOmegaSST(dm, mesh, expandedDir, *schemes));
    // turbulence->validate()
    kOmegaSSTCorrectNut(dm, ls_, *sst, rho, thermo.mu, U);

    // 支援的邊界條件
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const Bc t = U.bc[pi].type;
        const bool ok =
            t == Bc::empty || isFixedValueFamily(t)
         || (isMixedFamily(t) && !isInletOutletFamily(t));
        if (!ok)
        {
            throw std::runtime_error
            (
                "U patch " + dm.patches[pi].name + ": " + bcName(t)
              + " is not supported (fixedValue family, freestreamVelocity)"
            );
        }
    }
}


bool RhoSimpleSolver::criteriaSatisfied() const
{
    if (simple.residualControl.empty())
    {
        return false;
    }
    bool achieved = true;
    bool checked = false;
    for (const auto& r : residuals.initial)
    {
        int idx = -1;
        for (std::size_t i = 0; i < simple.residualControl.size(); ++i)
        {
            const SimpleControls::Residual& rc = simple.residualControl[i];
            const bool m = rc.isPattern
                ? std::regex_match(r.first, std::regex(rc.name))
                : (rc.name == r.first);
            if (m)
            {
                idx = int(i);
                break;
            }
        }
        if (idx < 0)
        {
            continue;
        }
        checked = true;
        achieved = achieved && (r.second < simple.residualControl[std::size_t(idx)].absTol);
    }
    return checked && achieved;
}


void RhoSimpleSolver::storePrevIter()
{
    if (pFieldRelax.relax)
    {
        copyVol(thermo.p, pPrevIter_);
    }
    if (rhoFieldRelax.relax)
    {
        copyVol(rho, rhoPrevIter_);
    }
}


bool RhoSimpleSolver::loop()
{
    if (initialised_)
    {
        if (criteriaSatisfied())
        {
            converged = true;
            return false;
        }
        storePrevIter();
    }
    else
    {
        initialised_ = true;
        storePrevIter();
    }
    // Time::loop()：run() 為真時 operator++
    if (!(t < (time.endTime - 0.5*time.deltaT)))
    {
        return false;
    }
    t = t + time.deltaT;
    ++timeIndex;
    residuals.initial.clear();
    return true;
}


void RhoSimpleSolver::solveMomentum()
{
    Stopwatch sw(profile);
    // fvMatrix(U) 的建構：freestreamVelocity 的 updateCoeffs
    updateFreestreamVelocityCoeffs(dm, U);

    // grad(U)（divDevRhoReff 與 laplacian 的修正）與 linearUpwind 的梯度
    VolField<tens> gradU;
    gradByScheme(dm, ls_, gradUScheme, U, gradU);
    VolField<tens> gradUlim;
    const VolField<tens>* gUlim = &gradU;
    if (limitedUScheme != gradUScheme)
    {
        gradByScheme(dm, ls_, limitedUScheme, U, gradUlim);
        gUlim = &gradUlim;
    }

    FvVectorMatrix A = fvmDivBoundedGaussLinearUpwind(dm, phi, U, *gUlim);

    // divDevRhoReff(U) = -fvc::div((rho*nuEff)*dev2(T(grad(U)))) - fvm::laplacian(rho*nuEff, U)
    {
        VolField<double> rhoNuEff = calculated(dm, "(rho*nuEff)");
        cellsAndBoundary(dm, [&](const bool b, const std::int32_t n)
        {
            kRhoNuEff<<<nBlocks(n), BS>>>
            (
                n, part(rho, b).cview().p[0], part(sst->nut, b).cview().p[0],
                part(thermo.mu, b).cview().p[0], part(rhoNuEff, b).view().p[0]
            );
        });
        VolField<tens> tau = makeVolField<tens>(dm, "tau", Bc::calculated);
        cellsAndBoundary(dm, [&](const bool b, const std::int32_t n)
        {
            kTauEff<<<nBlocks(n), BS>>>
            (
                n, part(rhoNuEff, b).cview().p[0], part(gradU, b).cview(), part(tau, b).view()
            );
        });
        DField<vec3> divTau(std::size_t(dm.nCells));
        divGaussLinear(dm, tau, divTau);
        DField<vec3> negDivTau(std::size_t(dm.nCells));
        if (dm.nCells)
        {
            kNegVec<<<nBlocks(dm.nCells), BS>>>(dm.nCells, divTau.cview(), negDivTau.view());
            GNC_CUDA_LAUNCH_CHECK();
        }
        FvVectorMatrix lap = fvmLaplacianGaussLinearCorrected(dm, rhoNuEff, U, gradU);
        suMinusMatrix(dm, negDivTau, lap);
        addMatrix(dm, A, lap);
    }

    if (URelax.relax)
    {
        relaxMatrix(dm, A, U, URelax.factor);
    }

    // solve(UEqn == -fvc::grad(p))
    {
        VolField<vec3> gradP;
        gradByScheme(dm, ls_, gradPScheme, thermo.p, gradP);
        DField<vec3> negGradP(std::size_t(dm.nCells));
        if (dm.nCells)
        {
            kNegVec<<<nBlocks(dm.nCells), BS>>>(dm.nCells, gradP.in.cview(), negGradP.view());
            GNC_CUDA_LAUNCH_CHECK();
        }
        FvVectorMatrix S = copyMatrix(A);
        matrixEqSu(dm, S, negGradP);
        lastU = solveVectorMatrix(dm, plan_, S, U, USolver, &phi, true);
    }
    residuals.initial.emplace_back("U", lastU.maxInitialResidual());
    UEqn = std::move(A);
    sw.add(timings.momentum);
}


void RhoSimpleSolver::updateEnergyCoeffs()
{
    // he.boundaryFieldRef().updateCoeffs()：每個能量 patch 先 Tw.evaluate()
    evaluateBCs(dm, thermo.T, &phi);
    DField<double> snT;
    snGradBCs(dm, thermo.T, snT);
    const ThermoParams& P = thermo.params;
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const std::int32_t n = dm.fvPatchSize[pi];
        if (n == 0)
        {
            continue;
        }
        const std::int32_t start = dm.fvPatchStart[pi];
        PatchBC<double>& bc = thermo.he.bc[pi];
        const PatchBC<double>& Tbc = thermo.T.bc[pi];
        switch (bc.type)
        {
            case Bc::mixedEnergy:
                kMixedEnergyCoeffs<<<nBlocks(n), BS>>>
                (
                    start, n, dm.fvBoundaryFaces.data(), P, thermo.p.bnd.cview().p[0],
                    thermo.T.bnd.cview().p[0], Tbc.refValue.cview().p[0],
                    Tbc.refGrad.cview().p[0], Tbc.valueFraction.data(), dm.deltaCoeffs.data(),
                    bc.valueFraction.data(), bc.refValue.view().p[0], bc.refGrad.view().p[0]
                );
                break;
            case Bc::gradientEnergy:
                kGradientEnergyCoeffs<<<nBlocks(n), BS>>>
                (
                    start, n, dm.fvBoundaryFaces.data(), P, thermo.p.bnd.cview().p[0],
                    thermo.T.bnd.cview().p[0], snT.cview().p[0], dm.deltaCoeffs.data(),
                    bc.gradient.view().p[0]
                );
                break;
            case Bc::fixedEnergy:
                kFixedEnergy<<<nBlocks(n), BS>>>
                (
                    start, n, P, thermo.p.bnd.cview().p[0], thermo.T.bnd.cview().p[0],
                    thermo.he.bnd.view().p[0]
                );
                break;
            case Bc::empty:
                break;
            default:
                throw std::runtime_error
                (
                    "e patch " + dm.patches[pi].name + ": " + bcName(bc.type) + " is not supported"
                );
        }
        GNC_CUDA_LAUNCH_CHECK();
    }
}


void RhoSimpleSolver::limitTemperature()
{
    nLimitedLow = 0;
    nLimitedHigh = 0;
    if (!limitT)
    {
        return;
    }
    const ThermoParams& P = thermo.params;
    DeviceArray<int> counts(4);
    counts.zero();
    if (dm.nCells)
    {
        kLimitHe<<<nBlocks(dm.nCells), BS>>>
        (
            dm.nCells, P, thermo.p.in.cview().p[0], Tmin, Tmax, thermo.he.in.view().p[0],
            counts.data()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    // 'all' 的邊界：he 的 patch 不 fixesValue 時同樣限制
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const std::int32_t n = dm.fvPatchSize[pi];
        if (n == 0 || fixesValue(thermo.he.bc[pi].type))
        {
            continue;
        }
        const std::int32_t s = dm.fvPatchStart[pi];
        kLimitHe<<<nBlocks(n), BS>>>
        (
            n, P, thermo.p.bnd.cview().p[0] + s, Tmin, Tmax, thermo.he.bnd.view().p[0] + s,
            counts.data() + 2
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    const std::vector<int> c = counts.download();
    nLimitedLow = c[0];
    nLimitedHigh = c[1];
    if (c[0] || c[1] || c[2] || c[3])
    {
        // he.correctBoundaryConditions()：updateCoeffs（Tw.evaluate）後 evaluate
        updateEnergyCoeffs();
        evaluateBCs(dm, thermo.he, &phi, true);
    }
}


void RhoSimpleSolver::solveEnergy()
{
    Stopwatch sw(profile);
    VolField<double>& he = thermo.he;

    // fvMatrix(he) 的建構：能量邊界的 updateCoeffs（T 的邊界 evaluate）
    updateEnergyCoeffs();

    VolField<vec3> gradHeLim;
    gradByScheme(dm, ls_, limitedEScheme, he, gradHeLim);
    FvScalarMatrix A = fvmDivBoundedGaussLinearUpwind(dm, phi, he, gradHeLim);

    // + fvc::div(phi, Ekp)，Ekp = 0.5*magSqr(U) + p/rho
    {
        VolField<double> Ekp = calculated(dm, "Ekp");
        cellsAndBoundary(dm, [&](const bool b, const std::int32_t n)
        {
            kEkp<<<nBlocks(n), BS>>>
            (
                n, part(U, b).cview(), part(thermo.p, b).cview().p[0],
                part(rho, b).cview().p[0], part(Ekp, b).view().p[0]
            );
        });
        VolField<vec3> gradEkp;
        gradByScheme(dm, ls_, limitedEkpScheme, Ekp, gradEkp);
        const std::int32_t nIF = dm.nInternalFaces;
        const std::int32_t nB = dm.nFvBoundaryFaces();
        SurfField<double> flux;
        flux.in.allocate(std::size_t(nIF));
        flux.bnd.allocate(std::size_t(nB));
        if (nIF)
        {
            kLinearUpwindFlux<<<nBlocks(nIF), BS>>>
            (
                nIF, dm.owner.data(), dm.neighbour.data(), phi.in.cview().p[0], vecView(dm.Cf),
                vecView(dm.C), Ekp.in.cview().p[0], gradEkp.in.cview(), flux.in.view().p[0]
            );
        }
        if (nB)
        {
            kBoundaryFluxPlusZero<<<nBlocks(nB), BS>>>
            (
                nB, phi.bnd.cview().p[0], Ekp.bnd.cview().p[0], flux.bnd.view().p[0]
            );
        }
        GNC_CUDA_LAUNCH_CHECK();
        DField<double> conv(std::size_t(dm.nCells)), sInt(std::size_t(dm.nCells));
        surfaceIntegrate(dm, flux, conv);
        surfaceIntegrate(dm, phi, sInt);
        DeviceArray<double> divEkp(std::size_t(dm.nCells));
        if (dm.nCells)
        {
            kBoundedExplicit<<<nBlocks(dm.nCells), BS>>>
            (
                dm.nCells, conv.cview().p[0], sInt.cview().p[0], Ekp.in.cview().p[0], divEkp.data()
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
        matrixPlusSu(dm, A, divEkp.data());
    }

    // - fvm::laplacian(alphaEff, he)
    {
        VolField<double> alphaEff = calculated(dm, "alphaEff");
        cellsAndBoundary(dm, [&](const bool b, const std::int32_t n)
        {
            kAlphaEff<<<nBlocks(n), BS>>>
            (
                n, thermo.params, part(thermo.p, b).cview().p[0], part(thermo.T, b).cview().p[0],
                part(thermo.alpha, b).cview().p[0], part(sst->alphat, b).cview().p[0],
                part(alphaEff, b).view().p[0]
            );
        });
        VolField<vec3> gradHe;
        gradByScheme(dm, ls_, gradEScheme, he, gradHe);
        const FvScalarMatrix lap = fvmLaplacianGaussLinearCorrected(dm, alphaEff, he, gradHe, &phi);
        subtractMatrix(dm, A, lap);
    }

    if (eRelax.relax)
    {
        relaxMatrix(dm, A, he, eRelax.factor);
    }
    lastE = solveScalarMatrix(dm, plan_, A, he, eSolver, &phi, true);
    residuals.initial.emplace_back("e", lastE.initialResidual);

    // fvOptions.correct(he)、thermo.correct()
    limitTemperature();
    calculatePsiThermo(dm, thermo);
    sw.add(timings.energy);
}


void RhoSimpleSolver::solvePressure()
{
    Stopwatch sw(profile);
    double tPre = 0;
    const std::int32_t nC = dm.nCells;
    const std::int32_t nIF = dm.nInternalFaces;
    const std::int32_t nB = dm.nFvBoundaryFaces();
    VolField<double>& p = thermo.p;

    // rAU = 1.0/UEqn.A()
    {
        VolField<double> A = matrixA(dm, UEqn);
        rAU = makeVolField<double>(dm, "rAU", Bc::extrapolatedCalculated);
        cellsAndBoundary(dm, [&](const bool b, const std::int32_t n)
        {
            kReciprocal<<<nBlocks(n), BS>>>(n, part(A, b).cview().p[0], part(rAU, b).view().p[0]);
        });
    }
    // rhorAUf = interpolate(rho*rAU)
    {
        VolField<double> rhorAU = calculated(dm, "(rho*rAU)");
        cellsAndBoundary(dm, [&](const bool b, const std::int32_t n)
        {
            kMul<<<nBlocks(n), BS>>>
            (
                n, part(rho, b).cview().p[0], part(rAU, b).cview().p[0], part(rhorAU, b).view().p[0]
            );
        });
        rhorAUf = SurfField<double>{};
        rhorAUf.name = "rhorAUf";
        rhorAUf.in.allocate(std::size_t(nIF));
        rhorAUf.bnd.allocate(std::size_t(nB));
        interpolateLinear(dm, rhorAU, rhorAUf);
    }
    // HbyA = constrainHbyA(rAU*UEqn.H(), U, p)
    {
        const VolField<vec3> H = matrixH(dm, UEqn, U);
        HbyA = makeVolField<vec3>(dm, "HbyA", Bc::extrapolatedCalculated);
        cellsAndBoundary(dm, [&](const bool b, const std::int32_t n)
        {
            kScaleVec<<<nBlocks(n), BS>>>
            (
                n, part(rAU, b).cview().p[0], part(H, b).cview(), part(HbyA, b).view()
            );
        });
        for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
        {
            const std::int32_t n = dm.fvPatchSize[pi];
            const Bc t = U.bc[pi].type;
            // assignable() 為假：fixedValue 族、mixed 族（inletOutlet 族除外）
            const bool notAssignable =
                isFixedValueFamily(t) || (isMixedFamily(t) && !isInletOutletFamily(t));
            if (n && notAssignable)
            {
                kCopyVecRange<<<nBlocks(n), BS>>>(dm.fvPatchStart[pi], n, U.bnd.cview(), HbyA.bnd.view());
                GNC_CUDA_LAUNCH_CHECK();
            }
        }
    }
    // phiHbyA = interpolate(rho)*flux(HbyA)
    {
        SurfField<double> rhof;
        rhof.in.allocate(std::size_t(nIF));
        rhof.bnd.allocate(std::size_t(nB));
        interpolateLinear(dm, rho, rhof);
        const SurfField<vec3> Sf = meshSurfVector(dm, dm.Sf);
        SurfField<double> fluxHbyA;
        dotInterpolateLinear(dm, Sf, HbyA, fluxHbyA);
        phiHbyA = SurfField<double>{};
        phiHbyA.name = "phiHbyA";
        phiHbyA.in.allocate(std::size_t(nIF));
        phiHbyA.bnd.allocate(std::size_t(nB));
        multiplySurf(rhof, fluxHbyA, phiHbyA);
    }

    // pEqn = fvc::div(phiHbyA) - fvm::laplacian(rhorAUf, p)；fvMatrix(p) 的建構時
    // freestreamPressure 的 updateCoeffs
    updateFreestreamPressureCoeffs(dm, p, U);
    FvScalarMatrix pEqn;
    {
        VolField<vec3> gradP;
        gradByScheme(dm, ls_, gradPScheme, p, gradP);
        pEqn = fvmLaplacianSurfaceGamma(dm, rhorAUf, p, gradP, &phi, true);
        DField<double> divPhiHbyA(static_cast<std::size_t>(nC));
        surfaceIntegrate(dm, phiHbyA, divPhiHbyA);
        suMinusMatrix(dm, divPhiHbyA.cview().p[0], pEqn);
    }
    sw.add(tPre);
    lastP = solveScalarMatrix(dm, plan_, pEqn, p, pSolver, &phi, true);
    sw.add(timings.pressureSolve);
    residuals.initial.emplace_back("p", lastP.initialResidual);

    // phi = phiHbyA + pEqn.flux()
    {
        const SurfField<double> F = matrixFlux(dm, pEqn, p);
        if (nIF)
        {
            kAdd<<<nBlocks(nIF), BS>>>(nIF, phiHbyA.in.cview().p[0], F.in.cview().p[0], phi.in.view().p[0]);
        }
        if (nB)
        {
            kAdd<<<nBlocks(nB), BS>>>(nB, phiHbyA.bnd.cview().p[0], F.bnd.cview().p[0], phi.bnd.view().p[0]);
        }
        GNC_CUDA_LAUNCH_CHECK();
    }

    // p.relax()（relaxField 時一律 operator==(prevIter + alpha*(p - prevIter))，alpha 為 1 也做）
    if (pFieldRelax.relax)
    {
        cellsAndBoundary(dm, [&](const bool b, const std::int32_t n)
        {
            kRelaxField<<<nBlocks(n), BS>>>
            (
                n, part(pPrevIter_, b).cview().p[0], pFieldRelax.factor, part(p, b).view().p[0]
            );
        });
    }

    // U = HbyA - rAU*fvc::grad(p)；U 的 patch 都是 operator= 不改值的型別
    // （fixedValue 族、freestreamVelocity），再 correctBoundaryConditions
    {
        VolField<vec3> gradP;
        gradByScheme(dm, ls_, gradPScheme, p, gradP);
        if (nC)
        {
            kUCorrect<<<nBlocks(nC), BS>>>(nC, HbyA.in.cview(), rAU.in.cview().p[0], gradP.in.cview(), U.in.view());
            GNC_CUDA_LAUNCH_CHECK();
        }
        updateFreestreamVelocityCoeffs(dm, U);
        evaluateBCs(dm, U, &phi, true);
    }

    // pressureControl.limit(p)：limits 有效時回傳真，之後 p.correctBoundaryConditions()
    if (limitMaxP || limitMinP)
    {
        const auto extreme = [&](const bool upper)
        {
            double v = upper ? maxValue(p.in.cview().p[0], std::size_t(nC))
                             : minValue(p.in.cview().p[0], std::size_t(nC));
            if (nB)
            {
                const double vb = upper ? maxValue(p.bnd.cview().p[0], std::size_t(nB))
                                        : minValue(p.bnd.cview().p[0], std::size_t(nB));
                v = upper ? foamMax(v, vb) : foamMin(v, vb);
            }
            return v;
        };
        const auto clamp = [&](const double v, const bool upper)
        {
            // p = min(p, pMax)：內部值與各 patch 的 operator=（fixedValue 族與 mixed
            // 族不改值，其他照抄）
            if (nC)
            {
                kClamp<<<nBlocks(nC), BS>>>(nC, v, upper, p.in.view().p[0]);
            }
            for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
            {
                const std::int32_t n = dm.fvPatchSize[pi];
                const Bc t = p.bc[pi].type;
                if (n == 0 || isFixedValueFamily(t) || isMixedFamily(t))
                {
                    continue;
                }
                if (!isZeroGradientFamily(t) && t != Bc::calculated)
                {
                    throw std::runtime_error("pressureControl: assignment to p patch type " + std::string(bcName(t)));
                }
                kClamp<<<nBlocks(n), BS>>>(n, v, upper, p.bnd.view().p[0] + dm.fvPatchStart[pi]);
            }
            GNC_CUDA_LAUNCH_CHECK();
        };
        // max 與 min 在一次同步取回（N-028）；max 有截斷時 min 照原本的次序在截斷之後重算
        const auto extremes = [&](double& vmax, double& vmin)
        {
            MinMaxReq req[4];
            int k = 0;
            req[k++] = MinMaxReq{p.in.cview().p[0], std::size_t(nC), true};
            if (nB)
            {
                req[k++] = MinMaxReq{p.bnd.cview().p[0], std::size_t(nB), true};
            }
            req[k++] = MinMaxReq{p.in.cview().p[0], std::size_t(nC), false};
            if (nB)
            {
                req[k++] = MinMaxReq{p.bnd.cview().p[0], std::size_t(nB), false};
            }
            double v[4] = {0.0, 0.0, 0.0, 0.0};
            minMaxMany(req, k, v);
            vmax = nB ? foamMax(v[0], v[1]) : v[0];
            vmin = nB ? foamMin(v[2], v[3]) : v[1];
        };
        double vmax = 0.0;
        double vmin = 0.0;
        extremes(vmax, vmin);
        if (limitMaxP && vmax > pMax)
        {
            std::printf("pressureControl: p max %.6g\n", vmax);
            clamp(pMax, true);
            vmin = extreme(false);
        }
        if (limitMinP && vmin < pMin)
        {
            std::printf("pressureControl: p min %.6g\n", vmin);
            clamp(pMin, false);
        }
        updateFreestreamPressureCoeffs(dm, p, U);
        evaluateBCs(dm, p, &phi, true);
    }

    // rho = thermo.rho()；rho.relax()
    cellsAndBoundary(dm, [&](const bool b, const std::int32_t n)
    {
        kMul<<<nBlocks(n), BS>>>
        (
            n, part(p, b).cview().p[0], part(thermo.psi, b).cview().p[0], part(rho, b).view().p[0]
        );
    });
    if (rhoFieldRelax.relax)
    {
        cellsAndBoundary(dm, [&](const bool b, const std::int32_t n)
        {
            kRelaxField<<<nBlocks(n), BS>>>
            (
                n, part(rhoPrevIter_, b).cview().p[0], rhoFieldRelax.factor, part(rho, b).view().p[0]
            );
        });
    }
    sw.add(tPre);
    timings.pressure += tPre;
}


void RhoSimpleSolver::correctTurbulence()
{
    Stopwatch sw(profile);
    kOmegaSSTCorrect(dm, plan_, ls_, *sst, rho, U, thermo.mu, phi, nullptr);
    sw.add(timings.turbulence);
    residuals.initial.emplace_back("omega", sst->lastSolveOmega.initialResidual);
    residuals.initial.emplace_back("k", sst->lastSolveK.initialResidual);
}

} // namespace gnc
