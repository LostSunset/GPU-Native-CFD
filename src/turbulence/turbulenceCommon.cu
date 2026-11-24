// GPU-Native-CFD，GPL-3.0-or-later。見 turbulenceCommon.cuh。

#include "turbulence/turbulenceCommon.cuh"

#include "fields/boundaryConditions.cuh"
#include "fvm/fvScalarMatrix.cuh"
#include "io/dictionary.hpp"
#include "io/foamTokenizer.hpp"
#include "mesh/wallDist.hpp"

#include <cstdio>
#include <regex>
#include <sstream>
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

// 鍵的查詢：先完全相符，再依倒序比對 pattern（OpenFOAM dictionary 的規則）
const Dictionary::Entry* findWithPatterns(const Dictionary& d, const std::string& key)
{
    if (const Dictionary::Entry* e = d.find(key))
    {
        return e;
    }
    const auto& es = d.entries();
    for (auto it = es.rbegin(); it != es.rend(); ++it)
    {
        if (it->isPattern && std::regex_match(key, std::regex(it->key)))
        {
            return &*it;
        }
    }
    return nullptr;
}

double entryScalar(const Dictionary& d, const Dictionary::Entry& e)
{
    // 只支援常數（Function1 的 constant 簡寫）
    if (e.dict || e.stream.size() != 1)
    {
        d.error("relaxation factor " + e.key + " must be a constant");
    }
    // pattern 鍵（"(k|omega)"）不能以 getScalar 查，直接讀項目的值
    double v = 0;
    if (!parseScalar(e.stream[0].text, v))
    {
        d.error("relaxation factor " + e.key + ": invalid number " + e.stream[0].text);
    }
    return v;
}

// out = a/b
__global__ void kDivide(const std::size_t n, const double* __restrict__ a, const double* __restrict__ b, double* __restrict__ out)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = a[i]/b[i];
    }
}

// alphat = (rho*nut)/Prt
__global__ void kAlphat
(
    const std::size_t n,
    const double* __restrict__ rho,
    const double* __restrict__ nut,
    const double Prt,
    double* __restrict__ alphat
)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n)
    {
        alphat[i] = (rho[i]*nut[i])/Prt;
    }
}

// compressible::alphatWallFunction::updateCoeffs：operator==(rhow*nutw/Prt)
__global__ void kAlphatWall
(
    const std::int32_t start,
    const std::int32_t n,
    const double* __restrict__ rhoB,
    const double* __restrict__ nutB,
    const double Prt,
    double* __restrict__ alphatB
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const std::int32_t b = start + i;
        alphatB[b] = (rhoB[b]*nutB[b])/Prt;
    }
}

// Poisson 牆距離：y = max(sqrt(magSqr(g) + 2*yPsi) - mag(g), SMALL)
__global__ void kPoissonY
(
    const std::size_t n,
    const CView<vec3> g,
    const double* __restrict__ yPsi,
    double* __restrict__ y
)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const vec3 gi = g[i];
        const double m = mag(gi);
        y[i] = foamMax(::sqrt(magSqr(gi) + 2*yPsi[i]) - m, SMALL);
    }
}

} // namespace


std::string trimmedScheme(const std::string& s)
{
    std::istringstream is(s);
    std::string w, out;
    while (is >> w)
    {
        out += out.empty() ? w : " " + w;
    }
    return out;
}


bool parseSwitch(const std::string& w, const std::string& where)
{
    if (w == "on" || w == "true" || w == "yes" || w == "y" || w == "t") return true;
    if (w == "off" || w == "false" || w == "no" || w == "n" || w == "f" || w == "none") return false;
    throw std::runtime_error(where + ": bad switch value " + w);
}


namespace
{

EquationRelax readRelax
(
    const std::string& path,
    const char* section,
    const std::string& fieldName
)
{
    EquationRelax r;
    FoamTokenizer t(path);
    const Dictionary d = Dictionary::parse(t, path, false);
    const Dictionary::Entry* re = d.find("relaxationFactors");
    if (re && re->dict)
    {
        const Dictionary::Entry* ee = re->dict->find(section);
        if (ee && ee->dict)
        {
            const Dictionary& eq = *ee->dict;
            const Dictionary::Entry* e = findWithPatterns(eq, fieldName);
            if (!e)
            {
                e = eq.find("default");
            }
            if (e)
            {
                r.relax = true;
                r.factor = entryScalar(eq, *e);
            }
        }
        else if
        (
            !re->dict->entries().empty()
         && !re->dict->found("fields") && !re->dict->found("equations")
        )
        {
            throw std::runtime_error(path + ": relaxationFactors without equations/fields subdictionaries are not supported");
        }
    }
    return r;
}

} // namespace


EquationRelax readEquationRelax(const std::string& path, const std::string& fieldName)
{
    return readRelax(path, "equations", fieldName);
}


EquationRelax readFieldRelax(const std::string& path, const std::string& fieldName)
{
    return readRelax(path, "fields", fieldName);
}


void nuField
(
    const DeviceMesh& dm,
    const VolField<double>& mu,
    const VolField<double>& rho,
    VolField<double>& nu
)
{
    const std::size_t nC = std::size_t(dm.nCells);
    const std::size_t nB = std::size_t(dm.nFvBoundaryFaces());
    nu.in.allocate(nC);
    nu.bnd.allocate(nB);
    if (nC)
    {
        kDivide<<<nBlocks(nC), BS>>>(nC, mu.in.cview().p[0], rho.in.cview().p[0], nu.in.view().p[0]);
    }
    if (nB)
    {
        kDivide<<<nBlocks(nB), BS>>>(nB, mu.bnd.cview().p[0], rho.bnd.cview().p[0], nu.bnd.view().p[0]);
    }
    GNC_CUDA_LAUNCH_CHECK();
}


void assignBoundary
(
    const DeviceMesh& dm,
    VolField<double>& f,
    const DeviceArray<double>& exprBnd
)
{
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const std::int32_t start = dm.fvPatchStart[pi];
        const std::int32_t size = dm.fvPatchSize[pi];
        if (size == 0)
        {
            continue;
        }
        const Bc t = f.bc[pi].type;
        // fixedValue 族的 operator= 不改值；transform 族（symmetry 等）的
        // operator= 以新的內部值 evaluate，由之後的 correctBoundaryConditions 處理
        if (isFixedValueFamily(t) || t == Bc::symmetry || t == Bc::symmetryPlane || t == Bc::slip)
        {
            continue;
        }
        if (t != Bc::calculated && t != Bc::extrapolatedCalculated && !isZeroGradientFamily(t))
        {
            throw std::runtime_error
            (
                "field " + f.name + " patch " + dm.patches[pi].name + ": assignment to "
              + bcName(t) + " is not supported"
            );
        }
        GNC_CUDA_CHECK
        (
            cudaMemcpy
            (
                f.bnd.view().p[0] + start, exprBnd.data() + start,
                std::size_t(size)*sizeof(double), cudaMemcpyDeviceToDevice
            )
        );
    }
}


void eddyDiffusivityCorrectNut
(
    const DeviceMesh& dm,
    const VolField<double>& rho,
    const VolField<double>& nut,
    const double Prt,
    VolField<double>& alphat
)
{
    const std::size_t nC = std::size_t(dm.nCells);
    const std::size_t nB = std::size_t(dm.nFvBoundaryFaces());
    DeviceArray<double> alphatB(nB);
    if (nC)
    {
        kAlphat<<<nBlocks(nC), BS>>>(nC, rho.in.cview().p[0], nut.in.cview().p[0], Prt, alphat.in.view().p[0]);
    }
    if (nB)
    {
        kAlphat<<<nBlocks(nB), BS>>>(nB, rho.bnd.cview().p[0], nut.bnd.cview().p[0], Prt, alphatB.data());
    }
    GNC_CUDA_LAUNCH_CHECK();
    assignBoundary(dm, alphat, alphatB);
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const PatchBC<double>& bc = alphat.bc[pi];
        const std::int32_t size = dm.fvPatchSize[pi];
        if (size == 0 || bc.type != Bc::compressibleAlphatWallFunction)
        {
            continue;
        }
        kAlphatWall<<<nBlocks(std::size_t(size)), BS>>>
        (
            dm.fvPatchStart[pi], size, rho.bnd.cview().p[0], nut.bnd.cview().p[0],
            bc.wallPrt, alphat.bnd.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    evaluateBCs(dm, alphat, nullptr);
}


void requireMeshWave(const FvSchemes& schemes)
{
    const Dictionary::Entry* we = schemes.dict().find("wallDist");
    const std::string method = (we && we->dict) ? we->dict->getWord("method") : "";
    if (method != "meshWave")
    {
        throw std::runtime_error("fvSchemes wallDist/method " + method + " is not supported (meshWave)");
    }
}


WallDistanceField wallDistanceMeshWave(const DeviceMesh& dm, const PolyMesh& mesh)
{
    const std::size_t nC = std::size_t(dm.nCells);
    const std::size_t nF = std::size_t(dm.nFaces);
    const auto interleave = [](const DeviceVec3Array& a, const std::size_t n)
    {
        const std::vector<double> x = a.x.download(), y = a.y.download(), z = a.z.download();
        std::vector<double> o(3*n);
        for (std::size_t i = 0; i < n; ++i)
        {
            o[3*i] = x[i];
            o[3*i + 1] = y[i];
            o[3*i + 2] = z[i];
        }
        return o;
    };
    const WallDistance wd = meshWaveWallDistance(mesh, interleave(dm.C, nC), interleave(dm.Cf, nF));
    WallDistanceField w;
    w.y.upload(wd.cell);
    const std::vector<std::int32_t> fvb = dm.fvBoundaryFaces.download();
    std::vector<double> yb(fvb.size()), ynw(fvb.size());
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const bool wall = dm.patches[pi].type == "wall";
        const std::size_t start = std::size_t(dm.fvPatchStart[pi]);
        for (std::size_t b = start; b < start + std::size_t(dm.fvPatchSize[pi]); ++b)
        {
            const std::size_t f = std::size_t(fvb[b]);
            yb[b] = wall ? wd.face[f] : wd.cell[std::size_t(mesh.owner[f])];
            ynw[b] = wd.nearWall[f];
        }
    }
    w.yB.upload(yb);
    w.yNearWall.upload(ynw);
    return w;
}

WallDistanceField wallDistancePoisson
(
    const DeviceMesh& dm,
    const PolyMesh& mesh,
    const FvSchemes& schemes,
    const LSVectors& ls
)
{
    WallDistanceField w = wallDistanceMeshWave(dm, mesh);
    const std::size_t nC = std::size_t(dm.nCells);
    const std::size_t nB = std::size_t(dm.nFvBoundaryFaces());

    VolField<double> yPsi = makeVolField<double>(dm, "yPsi", Bc::zeroGradient);
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        if (dm.patches[pi].type == "wall")
        {
            yPsi.bc[pi].type = Bc::fixedValue;
        }
    }
    VolField<double> one;
    one.name = "1";
    one.in.allocate(nC);
    one.bnd.allocate(nB);
    one.in.uploadAoS(std::vector<double>(nC, 1.0));
    one.bnd.uploadAoS(std::vector<double>(nB, 1.0));
    VolField<vec3> gradZero = makeVolField<vec3>(dm, "grad(yPsi)", Bc::zeroGradient);
    const LaplacianSnGrad sn = parseLaplacianScheme
    (
        trimmedScheme(schemes.laplacian("laplacian(yPsi)")), "laplacian(yPsi)"
    );
    FvScalarMatrix A = fvmLaplacianGaussLinearCorrected(dm, one, yPsi, gradZero, nullptr, sn);
    {
        // fvm::laplacian(yPsi) == -1：source += V*(-1)
        DeviceArray<double> minusOne(nC);
        minusOne.upload(std::vector<double>(nC, -1.0));
        matrixMinusSu(dm, A, minusOne.data());
    }
    const SolverPerf perf = solvePCGDiagonal(dm, A, yPsi, 1e-10, 0, 1000000);
    std::printf
    (
        "wallDist Poisson: PCG Solving for yPsi, Initial residual = %g, Final residual = %g, No Iterations %d\n",
        perf.initialResidual, perf.finalResidual, perf.nIterations
    );

    VolField<vec3> g;
    gradByScheme(dm, ls, trimmedScheme(schemes.grad("grad(yPsi)")), yPsi, g);
    w.y.allocate(nC);
    DeviceArray<double> yExprB(nB);
    if (nC)
    {
        kPoissonY<<<nBlocks(nC), BS>>>(nC, g.in.cview(), yPsi.in.cview().p[0], w.y.data());
    }
    if (nB)
    {
        kPoissonY<<<nBlocks(nB), BS>>>(nB, g.bnd.cview(), yPsi.bnd.cview().p[0], yExprB.data());
    }
    GNC_CUDA_LAUNCH_CHECK();
    // 邊界：wall 為 SMALL，constraint（symmetry 等）取格值，其他取運算式的邊界值
    const std::vector<double> yc = w.y.download(), ye = yExprB.download();
    const std::vector<std::int32_t> fvb = dm.fvBoundaryFaces.download();
    std::vector<double> yb(nB);
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const std::string& t = dm.patches[pi].type;
        const std::size_t start = std::size_t(dm.fvPatchStart[pi]);
        for (std::size_t b = start; b < start + std::size_t(dm.fvPatchSize[pi]); ++b)
        {
            yb[b] = t == "wall" ? SMALL
                  : (t == "symmetry" || t == "symmetryPlane") ? yc[std::size_t(mesh.owner[std::size_t(fvb[b])])]
                  : ye[b];
        }
    }
    w.yB.upload(yb);
    return w;
}


WallDistanceField wallDistanceFromSchemes
(
    const DeviceMesh& dm,
    const PolyMesh& mesh,
    const FvSchemes& schemes,
    const LSVectors& ls
)
{
    const Dictionary::Entry* we = schemes.dict().find("wallDist");
    const std::string method = (we && we->dict) ? we->dict->getWord("method") : "";
    if (method == "meshWave")
    {
        return wallDistanceMeshWave(dm, mesh);
    }
    if (method == "Poisson")
    {
        return wallDistancePoisson(dm, mesh, schemes, ls);
    }
    throw std::runtime_error("fvSchemes wallDist/method " + method + " is not supported (meshWave, Poisson)");
}

} // namespace gnc
