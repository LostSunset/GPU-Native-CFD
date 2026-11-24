// Spalart–Allmaras 一方程紊流模型（可壓縮、RAS）：參考次序版。
// GPU-Native-CFD，GPL-3.0-or-later。
//
// 照 OpenFOAM v2606（Atlas upstream/openfoam，TurbulenceModels）：
//   SpalartAllmarasBase.C   係數與 correct()、correctNut(fv1)
//   SpalartAllmaras.C (RAS) dTilda = y（meshWave 等牆距離）；correctNut()
//                           之後呼叫 EddyDiffusivity::correctNut
//   EddyDiffusivity.C       alphat = rho*nut/Prt（Prt 讀 coeffDict，預設 1.0）
//   CompressibleTurbulenceModel.H  nu() = mu/rho（rho 為求解器的 rho，不是
//                           thermo 的 p*psi）；mut = rho*nut；muEff = mut + mu
// HiSA 在每個外迭代的 updateFields 之後呼叫 turbulence->correct()；
// createFields 的 turbulence->validate() 即 correctNut()。
//
// correct() 的離散（運算次序見 spalartAllmaras.cu）：
//   fvm::ddt(rho, nuTilda) + fvm::div(phi, nuTilda)
//   - fvm::laplacian(rho*DnuTildaEff, nuTilda) - Cb2/sigmaNut*rho*magSqr(grad(nuTilda))
//   == Cb1*rho*Stilda*nuTilda*(1 - ft2) - fvm::Sp((Cw1*fw - Cb1/kappa^2*ft2)*rho*nuTilda/dTilda^2, nuTilda)
//   relax、solve、bound(nuTilda, 0)、correctBoundaryConditions、correctNut
// 只支援 ft2 false、fvOptions 為空、ddt 為 bounded dualTime 的穩態、
// div 為 bounded Gauss upwind、laplacian 為 Gauss linear corrected、
// 梯度為 Gauss linear 或 faceLeastSquares linear、牆距離為 meshWave。
// fw 的 pow(x, 1/6) 是超越函數：CUDA 的 pow 與 glibc 在這個範圍約 9% 差 1 ulp，
// 所以用 glibc __ieee754_pow_fma 的逐位元移植（core/glibcPow.cuh，tests/gncPowCheck）。

#ifndef GNC_TURBULENCE_SPALARTALLMARAS_CUH
#define GNC_TURBULENCE_SPALARTALLMARAS_CUH

#include "fields/deviceField.cuh"
#include "fvm/fvScalarMatrix.cuh"
#include "io/fvSchemes.hpp"
#include "io/polyMeshReader.hpp"
#include "mesh/deviceMesh.cuh"
#include "ops/fvcOps.cuh"
#include "solvers/gncHisa/hisaLinearSolver.cuh"

#include <string>

namespace gnc
{

// SpalartAllmarasBase 的係數（coeffDict = RAS/SpalartAllmarasCoeffs，可不給）
struct SpalartAllmarasCoeffs
{
    double sigmaNut = 0.66666;
    double kappa = 0.41;
    double Cb1 = 0.1355;
    double Cb2 = 0.622;
    double Cw2 = 0.3;
    double Cw3 = 2.0;
    double Cv1 = 7.1;
    double Cs = 0.3;
    double Prt = 1.0;
    // 導出的常數（dimensionedScalar 的運算，主機端以相同次序計算）
    double Cw1 = 0;         // Cb1/sqr(kappa) + (1.0 + Cb2)/sigmaNut
    double Cb1k2 = 0;       // Cb1/sqr(kappa)
    double Cb2s = 0;        // Cb2/sigmaNut
    double Cv13 = 0;        // pow3(Cv1)
    double Cw36 = 0;        // pow6(Cw3)
    double onePlusCw36 = 0; // 1 + pow6(Cw3)
};

struct SpalartAllmaras
{
    SpalartAllmarasCoeffs c;
    bool turbulence = true;
    VolField<double> nuTilda;
    VolField<double> nut;
    VolField<double> alphat;
    std::string nuTildaDimensions, nutDimensions, alphatDimensions;   // 寫出時照抄
    DeviceArray<double> y;          // 格心的牆距離（dTilda）
    DeviceArray<double> yNearWall;  // 每個有限體積邊界面：nearWallDist（壁面函數的 y）
    SmoothSolverControls solver;
    bool relax = false;
    double relaxFactor = 0;
    std::string gradUScheme;
    std::string gradNuTildaScheme;
    LaplacianSnGrad laplacianSnGrad;
    SolverPerf lastSolve;
    bool lastBounded = false;
    // 診斷：為真時 correct() 把每格 fw 中 pow 的底數存到 debugPowBase
    // （gncTurbL2 以此對主機的 pow 檢查）
    bool debugPow = false;
    DeviceArray<double> debugPowBase;
};

// expandedDir：nuTilda、nut、alphat、turbulenceProperties、fvSchemes、fvSolution
// （foamDictionary -expand -precision 17）。牆距離以 meshWave 計算一次
SpalartAllmaras createSpalartAllmaras
(
    const DeviceMesh& dm,
    const PolyMesh& mesh,
    const std::string& expandedDir,
    const FvSchemes& schemes
);

// SpalartAllmaras::correctNut()：nut = nuTilda*fv1(chi)、correctBoundaryConditions；
// alphat = rho*nut/Prt、correctBoundaryConditions。壁面函數：
//   nutUSpaldingWallFunction（nutUSpaldingWallFunctionFvPatchScalarField.C）：
//     calcUTau 的 Newton 迭代（exp 用 glibc 移植版），y 為 nearWallDist，
//     nuw = mu_b/rho_b，nut_w = max(0, sqr(uTau)/(mag(snGrad(U)) + ROOTVSMALL) - nuw)
//   compressible::alphatWallFunction：alphat_w = rho_w*nut_w/Prt（Prt 預設 0.85）
void spalartAllmarasCorrectNut
(
    const DeviceMesh& dm,
    SpalartAllmaras& sa,
    const VolField<double>& rho,
    const VolField<double>& mu,
    const VolField<vec3>& U
);

// SpalartAllmarasBase::correct()
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
);

} // namespace gnc

#endif
