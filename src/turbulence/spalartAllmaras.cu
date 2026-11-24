// Spalart–Allmaras（可壓縮、RAS）：參考次序版。GPU-Native-CFD，GPL-3.0-or-later。
//
// 出處（Atlas upstream/openfoam，v2606）：SpalartAllmarasBase.C:40–160（chi、fv1、
// fv2、Omega、r、fw、Stilda、correctNut）、:330–392（DnuTildaEff、correct）；
// RAS/SpalartAllmaras.C:40–60；EddyDiffusivity.C:34–40；
// CompressibleTurbulenceModel.H:109–152；scalarImpl.H（sqr、pow3、pow6）；
// TensorI.H:803（skew）；VectorSpaceI.H:400（magSqr：v0² 起依序加）。
// 欄位運算的次序以 C++ 的左結合為準，例如
//   Stilda = max(Omega + ((fv2*nuTilda)/sqr(kappa*dTilda)), Cs*Omega)
//   Sp 係數 = ((((Cw1*fw) - (Cb1/sqr(kappa))*ft2)*rho)*nuTilda)/sqr(dTilda)
//   源項    = (((Cb1*rho)*Stilda)*nuTilda)*(1 - ft2)
//   Cb2 項  = ((Cb2/sigmaNut)*rho)*magSqr(grad(nuTilda))
// alpha 為 geometricOneField（單相），乘上去不改值。fw 的 pow 用 glibc 的移植版
// （core/glibcPow.cuh）。必須以 --fmad=false 編譯。

#include "turbulence/spalartAllmaras.cuh"

#include "turbulence/turbulenceCommon.cuh"

#include "core/glibcExp.cuh"
#include "core/glibcPow.cuh"
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

// nut = nuTilda*fv1，chi = nuTilda/nu，fv1 = chi3/(chi3 + pow3(Cv1))，chi3 = pow3(chi)
__global__ void kNut
(
    const std::size_t n,
    const double* __restrict__ nuTilda,
    const double* __restrict__ nu,
    const double Cv13,
    double* __restrict__ nut
)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const double chi = nuTilda[i]/nu[i];
        const double chi3 = foamPow3(chi);
        const double fv1 = chi3/(chi3 + Cv13);
        nut[i] = nuTilda[i]*fv1;
    }
}

// gamma = rho*DnuTildaEff，DnuTildaEff = (nuTilda + nu)/sigmaNut
__global__ void kGamma
(
    const std::size_t n,
    const double* __restrict__ rho,
    const double* __restrict__ nuTilda,
    const double* __restrict__ nu,
    const double sigmaNut,
    double* __restrict__ gamma
)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n)
    {
        gamma[i] = rho[i]*((nuTilda[i] + nu[i])/sigmaNut);
    }
}

struct SaConsts
{
    double kappa, Cs, Cw2, Cw1, Cb1, Cb1k2, Cb2s, Cv13, Cw36, onePlusCw36, sqrt2;
};

// 每格的 Cb2 項（su1）、源項（su2）與 Sp 係數（sp）
__global__ void kSaSources
(
    const std::int32_t n,
    const SaConsts k,
    const double* __restrict__ nuTilda,
    const double* __restrict__ nu,
    const double* __restrict__ rho,
    const double* __restrict__ y,
    const CView<tens> gradU,
    const CView<vec3> gradNuTilda,
    double* __restrict__ su1,
    double* __restrict__ su2,
    double* __restrict__ sp,
    double* __restrict__ powBase        // 診斷，可為 nullptr
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const double nuT = nuTilda[i];
    const double chi = nuT/nu[i];
    const double chi3 = foamPow3(chi);
    const double fv1 = chi3/(chi3 + k.Cv13);
    const double fv2 = 1.0 - chi/(1.0 + chi*fv1);

    // Omega = sqrt(2.0)*mag(skew(gradU))；skew 的對角為 Zero
    const tens t = gradU[i];
    const double s01 = 0.5*(t.xy - t.yx);
    const double s02 = 0.5*(t.xz - t.zx);
    const double s10 = 0.5*(t.yx - t.xy);
    const double s12 = 0.5*(t.yz - t.zy);
    const double s20 = 0.5*(t.zx - t.xz);
    const double s21 = 0.5*(t.zy - t.yz);
    const double z = 0.0;
    double ms = z*z;
    ms += s01*s01;
    ms += s02*s02;
    ms += s10*s10;
    ms += z*z;
    ms += s12*s12;
    ms += s20*s20;
    ms += s21*s21;
    ms += z*z;
    const double Omega = k.sqrt2*::sqrt(ms);

    const double d = y[i];
    const double kd = k.kappa*d;
    const double kd2 = foamSqr(kd);
    const double Stilda = foamMax(Omega + (fv2*nuT)/kd2, k.Cs*Omega);

    // r（邊界值設為 0，不用）、g、fw
    const double r = foamMin(nuT/(foamMax(Stilda, SMALL)*kd2), 10.0);
    const double g = r + k.Cw2*(foamPow6(r) - r);
    const double base = k.onePlusCw36/(foamPow6(g) + k.Cw36);
    if (powBase)
    {
        powBase[i] = base;
    }
    const double fw = g*glibcPow(base, 1.0/6.0);

    const double ft2 = 0.0;
    sp[i] = ((((k.Cw1*fw) - k.Cb1k2*ft2)*rho[i])*nuT)/foamSqr(d);
    su2[i] = (((k.Cb1*rho[i])*Stilda)*nuT)*(1.0 - ft2);
    su1[i] = (k.Cb2s*rho[i])*magSqr(gradNuTilda[i]);
}


// nutUSpaldingWallFunctionFvPatchScalarField::calcNut（maxIter、tolerance 為
// 預設 0.01 時不做 err 的保留分支）。運算次序照原始碼的左結合：
//   kUu  = min((kappa*magUp)/ut, 50)
//   fkUu = (exp(kUu) - 1) - kUu*(1 + 0.5*kUu)
//   f    = ((((-ut)*y)/nuw + magUp/ut) + (1.0/E)*(fkUu - ((1.0/6.0)*kUu)*sqr(kUu))
//   df   = (y/nuw + magUp/sqr(ut)) + (((1.0/E)*kUu)*fkUu)/ut
__global__ void kNutUSpalding
(
    const std::int32_t start,
    const std::int32_t n,
    const std::int32_t* __restrict__ fvbFaces,
    const std::int32_t* __restrict__ owner,
    const CView<vec3> Uin,
    const CView<vec3> Ub,
    const CView<vec3> snGradU,
    const double* __restrict__ muB,
    const double* __restrict__ rhoB,
    const double* __restrict__ y,
    const double kappa,
    const double E,
    const int maxIter,
    const double tolerance,
    double* __restrict__ nutB
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const std::int32_t b = start + i;
    const double magGradU = mag(snGradU[b]);
    const double nuw = muB[b]/rhoB[b];
    const double magUp = mag(Uin[owner[fvbFaces[b]]] - Ub[b]);
    const double nutw = nutB[b];
    const double yb = y[b];

    double uTau = 0.0;
    double ut = ::sqrt((nutw + nuw)*magGradU);
    if (ROOTVSMALL < ut)
    {
        int iter = 0;
        double err = 0.0;
        do
        {
            const double kUu = foamMin((kappa*magUp)/ut, 50.0);
            const double fkUu = (glibcExp(kUu) - 1.0) - kUu*(1.0 + 0.5*kUu);
            const double f =
                ((((-ut)*yb)/nuw) + magUp/ut)
              + (1.0/E)*(fkUu - ((1.0/6.0)*kUu)*foamSqr(kUu));
            const double df =
                (yb/nuw + magUp/foamSqr(ut))
              + (((1.0/E)*kUu)*fkUu)/ut;
            const double uTauNew = ut + f/df;
            err = ::fabs((ut - uTauNew)/ut);
            ut = uTauNew;
        } while (ut > ROOTVSMALL && err > tolerance && ++iter < maxIter);
        uTau = foamMax(0.0, ut);
    }
    nutB[b] = foamMax(0.0, foamSqr(uTau)/(magGradU + ROOTVSMALL) - nuw);
}

} // namespace


SpalartAllmaras createSpalartAllmaras
(
    const DeviceMesh& dm,
    const PolyMesh& mesh,
    const std::string& expandedDir,
    const FvSchemes& schemes
)
{
    SpalartAllmaras sa;

    // turbulenceProperties：simulationType RAS；RAS/RASModel SpalartAllmaras
    {
        const std::string path = expandedDir + "/turbulenceProperties";
        FoamTokenizer t(path);
        const Dictionary d = Dictionary::parse(t, path, false);
        if (d.getWord("simulationType") != "RAS")
        {
            throw std::runtime_error(path + ": simulationType must be RAS");
        }
        const Dictionary& ras = d.subDict("RAS");
        if (ras.getWord("RASModel") != "SpalartAllmaras")
        {
            throw std::runtime_error(path + ": RASModel " + ras.getWord("RASModel") + " is not supported");
        }
        sa.turbulence = parseSwitch(ras.getWord("turbulence"), path);
        const Dictionary::Entry* ce = ras.find("SpalartAllmarasCoeffs");
        const Dictionary empty;
        const Dictionary& cd = (ce && ce->dict) ? *ce->dict : empty;
        SpalartAllmarasCoeffs& c = sa.c;
        const auto rd = [&](const char* k, double& v)
        {
            if (cd.found(k)) v = cd.getScalar(k);
        };
        rd("sigmaNut", c.sigmaNut);
        rd("kappa", c.kappa);
        rd("Cb1", c.Cb1);
        rd("Cb2", c.Cb2);
        rd("Cw2", c.Cw2);
        rd("Cw3", c.Cw3);
        rd("Cv1", c.Cv1);
        rd("Cs", c.Cs);
        rd("Prt", c.Prt);
        if (cd.found("ft2") && parseSwitch(cd.getWord("ft2"), path))
        {
            throw std::runtime_error(path + ": ft2 true is not supported");
        }
        // dimensionedScalar 的運算（SpalartAllmarasBase.C:233、fw、Stilda）
        c.Cb1k2 = c.Cb1/foamSqr(c.kappa);
        c.Cw1 = c.Cb1k2 + (1.0 + c.Cb2)/c.sigmaNut;
        c.Cb2s = c.Cb2/c.sigmaNut;
        c.Cv13 = foamPow3(c.Cv1);
        c.Cw36 = foamPow6(c.Cw3);
        c.onePlusCw36 = 1 + c.Cw36;
    }

    // fvSchemes
    const std::string div = trimmedScheme(schemes.div("div(phi,nuTilda)"));
    if (div != "bounded Gauss upwind phi" && div != "bounded Gauss upwind")
    {
        throw std::runtime_error("div(phi,nuTilda) " + div + " is not supported (bounded Gauss upwind)");
    }
    sa.laplacianSnGrad = parseLaplacianScheme
    (
        trimmedScheme(schemes.laplacian("laplacian((rho*DnuTildaEff),nuTilda)")),
        "laplacian((rho*DnuTildaEff),nuTilda)"
    );
    sa.gradUScheme = trimmedScheme(schemes.grad("grad(U)"));
    sa.gradNuTildaScheme = trimmedScheme(schemes.grad("grad(nuTilda)"));
    requireMeshWave(schemes);

    // fvSolution：solvers/nuTilda、relaxationFactors/equations/nuTilda
    {
        const std::string path = expandedDir + "/fvSolution";
        sa.solver = readSmoothSolverControls(path, "nuTilda");
        const EquationRelax r = readEquationRelax(path, "nuTilda");
        sa.relax = r.relax;
        sa.relaxFactor = r.factor;
    }

    // 場
    {
        const HostField hn = readField(expandedDir + "/nuTilda", mesh);
        const HostField hm = readField(expandedDir + "/nut", mesh);
        const HostField ha = readField(expandedDir + "/alphat", mesh);
        sa.nuTilda = uploadVolField<double>(dm, hn);
        sa.nut = uploadVolField<double>(dm, hm);
        sa.alphat = uploadVolField<double>(dm, ha);
        sa.nuTildaDimensions = hn.dimensions;
        sa.nutDimensions = hm.dimensions;
        sa.alphatDimensions = ha.dimensions;
    }

    // 牆距離（meshWave，網格不變時算一次）
    {
        WallDistanceField w = wallDistanceMeshWave(dm, mesh);
        sa.y = std::move(w.y);
        sa.yNearWall = std::move(w.yNearWall);
    }
    return sa;
}


void spalartAllmarasCorrectNut
(
    const DeviceMesh& dm,
    SpalartAllmaras& sa,
    const VolField<double>& rho,
    const VolField<double>& mu,
    const VolField<vec3>& U
)
{
    const std::size_t nC = std::size_t(dm.nCells);
    const std::size_t nB = std::size_t(dm.nFvBoundaryFaces());
    VolField<double> nu;
    nuField(dm, mu, rho, nu);
    DeviceArray<double> nutB(nB);
    if (nC)
    {
        kNut<<<nBlocks(nC), BS>>>(nC, sa.nuTilda.in.cview().p[0], nu.in.cview().p[0], sa.c.Cv13, sa.nut.in.view().p[0]);
    }
    if (nB)
    {
        kNut<<<nBlocks(nB), BS>>>(nB, sa.nuTilda.bnd.cview().p[0], nu.bnd.cview().p[0], sa.c.Cv13, nutB.data());
    }
    GNC_CUDA_LAUNCH_CHECK();
    assignBoundary(dm, sa.nut, nutB);
    // nut.correctBoundaryConditions()：壁面函數的 updateCoeffs 在這裡算
    {
        DField<vec3> snGradU;
        bool haveSnGrad = false;
        for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
        {
            const PatchBC<double>& bc = sa.nut.bc[pi];
            const std::int32_t start = dm.fvPatchStart[pi];
            const std::int32_t size = dm.fvPatchSize[pi];
            if (size == 0 || bc.type != Bc::nutUSpaldingWallFunction)
            {
                continue;
            }
            if (::fabs(bc.wallTolerance - 0.01) > VSMALL)
            {
                throw std::runtime_error
                (
                    "nut patch " + dm.patches[pi].name
                  + ": nutUSpaldingWallFunction tolerance other than 0.01 is not supported"
                );
            }
            if (!haveSnGrad)
            {
                snGradBCs(dm, U, snGradU);
                haveSnGrad = true;
            }
            kNutUSpalding<<<nBlocks(std::size_t(size)), BS>>>
            (
                start, size, dm.fvBoundaryFaces.data(), dm.owner.data(), U.in.cview(),
                U.bnd.cview(), snGradU.cview(), mu.bnd.cview().p[0], rho.bnd.cview().p[0],
                sa.yNearWall.data(), bc.wallKappa, bc.wallE, bc.wallMaxIter, bc.wallTolerance,
                sa.nut.bnd.view().p[0]
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
    }
    evaluateBCs(dm, sa.nut, nullptr);

    // EddyDiffusivity::correctNut：alphat = rho*nut/Prt
    eddyDiffusivityCorrectNut(dm, rho, sa.nut, sa.c.Prt, sa.alphat);
}


void spalartAllmarasCorrect
(
    const DeviceMesh& dm,
    const LusgsPlan& plan,
    const LSVectors& ls,
    SpalartAllmaras& sa,
    const VolField<double>& rho,
    const VolField<vec3>& U,
    const VolField<double>& mu,
    const SurfField<double>& phi,
    const double* rPseudoDeltaT
)
{
    if (!sa.turbulence)
    {
        return;
    }
    const std::int32_t n = dm.nCells;
    const std::size_t nC = std::size_t(n);
    const std::size_t nB = std::size_t(dm.nFvBoundaryFaces());

    // nu = mu/rho（內部與邊界）
    VolField<double> nu;
    nuField(dm, mu, rho, nu);

    // 梯度：fvc::grad(U)、fvc::grad(nuTilda)
    VolField<tens> gradU;
    gradByScheme(dm, ls, sa.gradUScheme, U, gradU);
    VolField<vec3> gradNuTilda;
    gradByScheme(dm, ls, sa.gradNuTildaScheme, sa.nuTilda, gradNuTilda);

    // 源項
    SaConsts k;
    k.kappa = sa.c.kappa;
    k.Cs = sa.c.Cs;
    k.Cw2 = sa.c.Cw2;
    k.Cw1 = sa.c.Cw1;
    k.Cb1 = sa.c.Cb1;
    k.Cb1k2 = sa.c.Cb1k2;
    k.Cb2s = sa.c.Cb2s;
    k.Cv13 = sa.c.Cv13;
    k.Cw36 = sa.c.Cw36;
    k.onePlusCw36 = sa.c.onePlusCw36;
    k.sqrt2 = ::sqrt(2.0);
    DeviceArray<double> su1(nC), su2(nC), sp(nC);
    if (sa.debugPow)
    {
        sa.debugPowBase.allocate(nC);
    }
    if (nC)
    {
        kSaSources<<<nBlocks(nC), BS>>>
        (
            n, k, sa.nuTilda.in.cview().p[0], nu.in.cview().p[0], rho.in.cview().p[0],
            sa.y.data(), gradU.in.cview(), gradNuTilda.in.cview(),
            su1.data(), su2.data(), sp.data(), sa.debugPow ? sa.debugPowBase.data() : nullptr
        );
        GNC_CUDA_LAUNCH_CHECK();
    }

    // gamma = rho*DnuTildaEff（內部與邊界）
    VolField<double> gamma;
    gamma.name = "(rho*DnuTildaEff)";
    gamma.in.allocate(nC);
    gamma.bnd.allocate(nB);
    if (nC)
    {
        kGamma<<<nBlocks(nC), BS>>>(nC, rho.in.cview().p[0], sa.nuTilda.in.cview().p[0], nu.in.cview().p[0], sa.c.sigmaNut, gamma.in.view().p[0]);
    }
    if (nB)
    {
        kGamma<<<nBlocks(nB), BS>>>(nB, rho.bnd.cview().p[0], sa.nuTilda.bnd.cview().p[0], nu.bnd.cview().p[0], sa.c.sigmaNut, gamma.bnd.view().p[0]);
    }
    GNC_CUDA_LAUNCH_CHECK();

    // fvMatrix 建構時對 nuTilda 的邊界 updateCoeffs（inletOutlet 的 valueFraction）
    updateInletOutletCoeffs(dm, sa.nuTilda, phi);

    // 左邊：((ddt + div) - laplacian) - su1
    FvScalarMatrix eqn = fvmDdtBoundedDualTime(dm, rPseudoDeltaT, rho.in.cview().p[0], sa.nuTilda);
    {
        const FvScalarMatrix divM = fvmDivBoundedGaussUpwind(dm, phi, sa.nuTilda);
        addMatrix(dm, eqn, divM);
    }
    {
        const FvScalarMatrix lapM = fvmLaplacianGaussLinearCorrected
        (
            dm, gamma, sa.nuTilda, gradNuTilda, &phi, sa.laplacianSnGrad
        );
        subtractMatrix(dm, eqn, lapM);
    }
    matrixMinusSu(dm, eqn, su1.data());

    // 右邊：(su2 - Sp(sp)) + fvOptions（空）
    FvScalarMatrix rhs = fvmSp(dm, sp.data());
    suMinusMatrix(dm, su2.data(), rhs);
    {
        const FvScalarMatrix opts = fvmEmpty(dm);
        addMatrix(dm, rhs, opts);
    }

    // operator==：左邊 - 右邊
    subtractMatrix(dm, eqn, rhs);

    if (sa.relax)
    {
        relaxMatrix(dm, eqn, sa.nuTilda, sa.relaxFactor);
    }
    sa.lastSolve = solveScalarMatrix(dm, plan, eqn, sa.nuTilda, sa.solver, &phi);
    sa.lastBounded = boundField(dm, sa.nuTilda, 0.0);
    evaluateBCs(dm, sa.nuTilda, &phi);

    spalartAllmarasCorrectNut(dm, sa, rho, mu, U);
}

} // namespace gnc
