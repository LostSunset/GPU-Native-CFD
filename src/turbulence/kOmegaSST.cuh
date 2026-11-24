// k-omega SST 兩方程紊流模型（可壓縮、RAS）：參考次序版（N4b，N-022）。
// GPU-Native-CFD，GPL-3.0-or-later。
//
// 照 OpenFOAM v2606（Atlas upstream/openfoam，TurbulenceModels）：
//   kOmegaSSTBase.C/H       係數、F1、F2、F23、correctNut(S2)、S2、Pk、epsilonByk、
//                           GbyNu0、GbyNu、blend、alphaK、alphaOmega、beta、gamma、
//                           DkEff、DomegaEff、correct()
//   RAS/kOmegaSST.C         correctNut 之後呼叫 EddyDiffusivity::correctNut
//   RASModel.C              kMin、omegaMin（RAS 字典，預設 SMALL）；coeffDict =
//                           RAS 字典的 optionalSubDict("kOmegaSSTCoeffs")
//   compressibleTurbulenceModel.C  phi() = phi/interpolate(rho)（divU 用）
//   壁面函數                 omegaWallFunction（牆邊格的 omega、G 由所有 omegaWall
//                           面依 cornerWeights 加權，matrix 的 setValues）、
//                           kLowReWallFunction、nutkWallFunction
//   入口                     turbulentIntensityKineticEnergyInlet、
//                           turbulentMixingLengthFrequencyInlet（inletOutlet 族）
// HiSA 在每個外迭代的 updateFields 之後呼叫 turbulence->correct()；
// createFields 的 turbulence->validate() 即 correctNut()。rhoSimpleFoam（N6）在
// 每個 SIMPLE 迭代的 pEqn 之後呼叫 correct()，ddt 為 steadyState。
//
// 只支援：F3 off、decayControl off（kInf、omegaInf 為 0，源項的
// alpha*rho*beta*sqr(omegaInf) 與 alpha*rho*betaStar*omegaInf*kInf 在逐位元上
// 不改 source，省略）、fvOptions 為空（或不作用在 k、omega）、ddt 為 bounded
// dualTime 的穩態或 steadyState、div 為
// bounded Gauss upwind 或 bounded Gauss limitedLinear k（limiter 的梯度為
// fvc::grad(psi)，在 fvMatrix 建構的 updateCoeffs 之後）、laplacian 為 Gauss linear
// corrected（或 limited）、interpolate(rho) 為 linear、牆距離為 meshWave（逐位元）
// 或 Poisson（不逐位元，N-022）。F1、F2 的 tanh 用 glibc 的移植版
// （core/glibcTanh.cuh），壁面函數的 log 用 core/glibcLog.cuh，pow 用
// core/glibcPow.cuh。

#ifndef GNC_TURBULENCE_KOMEGASST_CUH
#define GNC_TURBULENCE_KOMEGASST_CUH

#include "fields/deviceField.cuh"
#include "fvm/fvScalarMatrix.cuh"
#include "io/fvSchemes.hpp"
#include "io/polyMeshReader.hpp"
#include "mesh/deviceMesh.cuh"
#include "ops/fvcOps.cuh"
#include "solvers/gncHisa/hisaLinearSolver.cuh"

#include <string>
#include <vector>

namespace gnc
{

// kOmegaSSTBase 的係數（coeffDict，可不給）與由它們導出的常數
// （dimensionedScalar 的運算，主機端以相同次序計算）
struct KOmegaSSTCoeffs
{
    double alphaK1 = 0.85;
    double alphaK2 = 1.0;
    double alphaOmega1 = 0.5;
    double alphaOmega2 = 0.856;
    double gamma1 = 5.0/9.0;
    double gamma2 = 0.44;
    double beta1 = 0.075;
    double beta2 = 0.0828;
    double betaStar = 0.09;
    double a1 = 0.31;
    double b1 = 1.0;
    double c1 = 10.0;
    double Prt = 1.0;           // EddyDiffusivity（coeffDict，預設 1.0）
    double kMin = 1.0e-15;      // RASModel（RAS 字典，預設 SMALL）
    double omegaMin = 1.0e-15;
    double Cmu = 0.09;          // turbulentMixingLengthFrequencyInlet 讀 coeffDict 的 Cmu

    double oneByBetaStar = 0;   // scalar(1)/betaStar
    double twoByBetaStar = 0;   // scalar(2)/betaStar
    double fourAlphaOmega2 = 0; // 4*alphaOmega2
    double twoAlphaOmega2 = 0;  // 2*alphaOmega2
    double dAlphaK = 0;         // alphaK1 - alphaK2（blend 的 psi1 - psi2）
    double dAlphaOmega = 0;
    double dGamma = 0;
    double dBeta = 0;
    double c1a1BetaStar = 0;    // (c1/a1)*betaStar
    double c1BetaStar = 0;      // c1*betaStar
    double mixingCmu25 = 0;     // pow(Cmu, 0.25)（主機端的 glibc pow）
};

struct KOmegaSST
{
    KOmegaSSTCoeffs c;
    bool turbulence = true;
    VolField<double> k;
    VolField<double> omega;
    VolField<double> nut;
    VolField<double> alphat;
    std::string kDimensions, omegaDimensions, nutDimensions, alphatDimensions;  // 寫出時照抄
    DeviceArray<double> y;          // 格心的牆距離
    DeviceArray<double> yB;         // wallDist::y() 的邊界值（每個有限體積邊界面）
    DeviceArray<double> yNearWall;  // nearWallDist（壁面函數的 y）
    SmoothSolverControls kSolver, omegaSolver;
    bool kRelax = false, omegaRelax = false;
    double kRelaxFactor = 0, omegaRelaxFactor = 0;
    std::string gradUScheme, gradKScheme, gradOmegaScheme;
    LaplacianSnGrad kLaplacian, omegaLaplacian;
    // div(phi,k)、div(phi,omega)：upwind，或 limitedLinear（係數 k）
    bool kLimitedLinear = false, omegaLimitedLinear = false;
    double kLimitedLinearCoeff = 1, omegaLimitedLinearCoeff = 1;

    // omegaWallFunction：所有 omegaWall patch 的面（有限體積邊界序號，依 patch 與
    // 面的次序）、每面的 cornerWeights（1/該格的 omegaWall 面數）；牆邊格（不重複）
    // 與各格的面清單（原累加次序）
    bool hasOmegaWall = false;
    DeviceArray<double> cornerWeight;       // 每個有限體積邊界面（非 omegaWall 為 0）
    std::int32_t nWallCells = 0;
    DeviceArray<std::int32_t> wallCells;
    DeviceArray<std::int32_t> wallCellOff;  // nWallCells + 1
    DeviceArray<std::int32_t> wallCellFaces;
    SetValuesPlan omegaWallPlan;

    // 診斷
    SolverPerf lastSolveOmega, lastSolveK;
    bool lastBoundedOmega = false, lastBoundedK = false;
};

// expandedDir：k、omega、nut、alphat、turbulenceProperties、fvSchemes、fvSolution
// （foamDictionary -expand -precision 17）。建構時 bound(k, kMin)、bound(omega,
// omegaMin)；牆距離以 meshWave 計算一次
KOmegaSST createKOmegaSST
(
    const DeviceMesh& dm,
    const PolyMesh& mesh,
    const std::string& expandedDir,
    const FvSchemes& schemes
);

// kOmegaSST::correctNut()：correctNut(2*magSqr(symm(fvc::grad(U))))
void kOmegaSSTCorrectNut
(
    const DeviceMesh& dm,
    const LSVectors& ls,
    KOmegaSST& m,
    const VolField<double>& rho,
    const VolField<double>& mu,
    const VolField<vec3>& U
);

// kOmegaSSTBase::correct()
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
    const double* rPseudoDeltaT     // nullptr：ddt 為 steadyState
);

} // namespace gnc

#endif
