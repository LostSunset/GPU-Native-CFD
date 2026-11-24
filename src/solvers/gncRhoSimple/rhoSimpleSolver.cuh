// gncRhoSimple 的求解迴圈：rhoSimpleFoam（OpenFOAM v2606）的 GPU 重寫（N6，N-023）。
// GPU-Native-CFD，GPL-3.0-or-later。
//
// 照 applications/solvers/compressible/rhoSimpleFoam（rhoSimpleFoam.C、createFields.H、
// UEqn.H、EEqn.H、pEqn.H）：
//   createFields   thermo（hePsiThermo）、rho = thermo.rho() = p*psi、U、
//                  phi = linearInterpolate(rho*U) & Sf（compressibleCreatePhi.H）、
//                  pressureControl（pressureControl.C：pMaxFactor、pMinFactor 乘以
//                  fixesValue 的 p patch 的 max／min）、紊流模型（validate = correctNut）
//   simple.loop    simpleControl::criteriaSatisfied：上一個迭代每個場第一次求解的
//                  初始殘差（向量取 cmptMax）都小於 residualControl 時結束；否則
//                  storePrevIter（relaxationFactors/fields 有的場：p、rho）
//   UEqn           fvm::div(phi, U) + divDevRhoReff(U)（linearViscousStress：
//                  -fvc::div((rho*nuEff)*dev2(T(grad(U)))) - fvm::laplacian(rho*nuEff, U)），
//                  relax，solve(UEqn == -fvc::grad(p))；MRF.DDt 與空的 fvOptions 矩陣
//                  只加減 0，不改值，省略
//   EEqn           fvm::div(phi, e) + fvc::div(phi, Ekp) - fvm::laplacian(alphaEff, e)，
//                  Ekp = 0.5*magSqr(U) + p/rho；relax、solve；limitTemperature 的
//                  correct(he)；thermo.correct()
//   pEqn           rAU = 1/A()；rhorAUf = interpolate(rho*rAU)；HbyA = constrainHbyA(rAU*H())；
//                  phiHbyA = interpolate(rho)*flux(HbyA)；pEqn = fvc::div(phiHbyA) -
//                  fvm::laplacian(rhorAUf, p)，solve；phi = phiHbyA + pEqn.flux()；
//                  p.relax()；U = HbyA - rAU*grad(p)、U 的邊界；pressureControl.limit；
//                  p 的邊界；rho = thermo.rho()、rho.relax()
//   turbulence     kOmegaSST::correct()（ddt 為 steadyState）
// 邊界條件的 updateCoeffs 照 OpenFOAM 的時點：fvMatrix 建構（U、e、p），以及
// correctBoundaryConditions（freestream* 的 valueFraction 由當下的 U 邊界值算）。
// 只支援：非 transonic、非 consistent、nNonOrthogonalCorrectors 0、沒有 MRF、
// p 有 fixesValue 的 patch（不需參考值）、沒有 fixedFluxPressure、fvOptions 只有
// limitTemperature（selectionMode all）或沒有、紊流模型為 kOmegaSST。

#ifndef GNC_SOLVERS_GNCRHOSIMPLE_RHOSIMPLESOLVER_CUH
#define GNC_SOLVERS_GNCRHOSIMPLE_RHOSIMPLESOLVER_CUH

#include "fvm/fvScalarMatrix.cuh"
#include "fvm/fvVectorMatrix.cuh"
#include "io/fvSchemes.hpp"
#include "io/polyMeshReader.hpp"
#include "mesh/deviceMesh.cuh"
#include "ops/fvcOps.cuh"
#include "solvers/gncHisa/hisaLinearSolver.cuh"
#include "solvers/gncHisa/hisaSolver.cuh"
#include "thermo/psiThermo.cuh"
#include "turbulence/kOmegaSST.cuh"
#include "turbulence/turbulenceCommon.cuh"

#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace gnc
{

// fvSolution 的 SIMPLE（simpleControl、pressureControl 讀的項目）
struct SimpleControls
{
    int nNonOrthCorr = 0;
    bool consistent = false;
    bool transonic = false;
    bool hasPMinFactor = false;
    bool hasPMaxFactor = false;
    double pMinFactor = 0;
    double pMaxFactor = 0;
    // residualControl（依字典次序；isPattern 時以 regex 整串比對）
    struct Residual
    {
        std::string name;
        bool isPattern = false;
        double absTol = 0;
    };
    std::vector<Residual> residualControl;
};

SimpleControls readSimpleControls(const std::string& fvSolutionPath);

// 一個迭代中各場第一次求解的初始殘差（solverPerformanceDict 的 first()），依
// OpenFOAM 的求解次序
struct IterationResiduals
{
    std::vector<std::pair<std::string, double>> initial;
};


class RhoSimpleSolver
{
public:
    // expandedDir：foamDictionary -expand -precision 17 展開的 p T U k omega nut alphat
    // thermophysicalProperties turbulenceProperties fvSchemes fvSolution controlDict
    // （與 fvOptions，有的話）
    RhoSimpleSolver(const std::string& caseDir, const std::string& expandedDir);

    PolyMesh mesh;
    DeviceMesh dm;
    std::unique_ptr<FvSchemes> schemes;

    PsiThermoFields thermo;
    VolField<double> rho;
    VolField<vec3> U;
    SurfField<double> phi;
    std::string pDimensions, TDimensions, UDimensions;
    std::unique_ptr<KOmegaSST> sst;

    SimpleControls simple;
    TimeControls time;      // controlDict（readTimeControls）
    SmoothSolverControls pSolver, USolver, eSolver;
    EquationRelax URelax, eRelax, pFieldRelax, rhoFieldRelax;

    // pressureControl
    bool limitMaxP = false, limitMinP = false;
    double pMax = 0, pMin = 0;

    // limitTemperature（fvOptions）
    bool limitT = false;
    double Tmin = 0, Tmax = 0;
    int nLimitedLow = 0, nLimitedHigh = 0;  // 上一次 correct 的限制格數

    // 時間
    double t = 0;
    int timeIndex = 0;
    bool converged = false;     // simple.loop() 的 criteriaSatisfied

    // simple.loop()：收斂（residualControl）或到 endTime 時回傳 false；否則
    // storePrevIter、時間加一步，回傳 true
    bool loop();

    // UEqn.H、EEqn.H、pEqn.H、turbulence->correct()
    void solveMomentum();
    void solveEnergy();
    void solvePressure();
    void correctTurbulence();

    // 各段的牆鐘時間（秒，累計）。profile 為真時每段結束前 cudaDeviceSynchronize，
    // 不影響數值
    struct Timings
    {
        double momentum = 0;
        double energy = 0;
        double pressure = 0;
        double pressureSolve = 0;   // pressure 之中 p 的線性求解
        double turbulence = 0;
    };
    bool profile = false;
    Timings timings;

    // 本迭代的初始殘差（U、e、p，再 omega、k）
    IterationResiduals residuals;
    VectorSolverPerf lastU;
    SolverPerf lastE, lastP;

    // 傾印用（UEqn.H 之後的 UEqn；pEqn.H 的中間量）
    FvVectorMatrix UEqn;
    VolField<double> rAU;
    VolField<vec3> HbyA;
    SurfField<double> phiHbyA;
    SurfField<double> rhorAUf;

    // 梯度方案（fvSchemes 的 grad 名稱）
    std::string gradUScheme, gradPScheme, gradEScheme, limitedUScheme, limitedEScheme,
        limitedEkpScheme;

private:
    LSVectors ls_;
    LusgsPlan plan_;
    VolField<double> pPrevIter_, rhoPrevIter_;
    bool initialised_ = false;

    bool criteriaSatisfied() const;
    void storePrevIter();
    void updateEnergyCoeffs();
    void limitTemperature();
};

} // namespace gnc

#endif
