// gncHisa 的求解迴圈（穩態）。GPU-Native-CFD，GPL-3.0-or-later。
//
// 照 HiSA（Atlas upstream/hisa，GPL-3.0-or-later，CSIR）：
//   singleRegionSolver.H  穩態時每個時間步：runTime++、beginTimeStep、
//                         pseudotimeControl::loop()；loop 為真時做一個外迭代，
//                         為假時 writeAndEnd（這個時間步沒有外迭代）
//   pseudotimeControl     corr 每步加一（穩態不歸零）；criteriaSatisfied：
//                         第一次、corr == 1、finalIter()（converged 或
//                         corr == nPseudoCorr）時為假；corr == 2 時記下
//                         initResiduals 與 firstResiduals；absCheck =
//                         max(res - pseudoTol) < 0，relCheck（corr > 2）=
//                         max(res/(init + ROOTVSMALL) - pseudoTolRel) < 0；
//                         滿足時再做一個外迭代，下一步結束。initResiduals 與
//                         firstResiduals 存進 state_（IOdictionary），每次
//                         loop() 開頭的 read() 再讀回：經 OCharStream 以
//                         IOstream::defaultPrecision()（controlDict 的
//                         writePrecision，預設 6）寫成文字再解析，所以從第三個
//                         外迭代起 SER 與 relCheck 用的是捨入後的值
//   hisaModule::outerIteration  通量、殘差（含 U 與能量邊界的 updateCoeffs）、
//                         SER（setPseudoCoNum）、偽時間步、p 的 updateCoeffs 與
//                         Jacobian、GMRES（其初始殘差即 setResidual 的值）、
//                         限界、updateFields
//   setPseudoCoNum        corr > 1 且 firstResiduals 已記下時：
//                         ratio = normFirst/normCurr/pseudoCoNum*pseudoCoNumInit，
//                         夾在 [minDecreaseFactor, maxIncreaseFactor]；
//                         pseudoCoNum *= ratio，再夾在 [pseudoCoNumMin, Max]；
//                         norm = sqrt(magSqr(rho) + magSqr(rhoE) + magSqr(rhoU))
// 殘差以 (rho, rhoU.x, rhoU.y, rhoU.z, rhoE) 的次序存；residualIO 的 max 依
// rho、rhoE、再 rhoU 的 cmptMax。
// 紊流：laminar（Stokes）或 RAS 的 Spalart–Allmaras（N4a）。HiSA 在 createFields
// 呼叫 turbulence->validate()（correctNut），每個外迭代的 updateFields 之後呼叫
// turbulence->correct()；muEff、alphaEff 經 HisaState 的 nut、alphat 進入殘差與
// Jacobian。
// 只支援：穩態（ddt 為 steadyState）、從 startTime 開始（不讀 uniform/ 的
// 續算狀態）、inviscid、laminar（Stokes）或 Spalart–Allmaras。

#ifndef GNC_SOLVERS_GNCHISA_HISASOLVER_CUH
#define GNC_SOLVERS_GNCHISA_HISASOLVER_CUH

#include "core/foamCompat.hpp"
#include "io/fvSchemes.hpp"
#include "io/polyMeshReader.hpp"
#include "mesh/cellFaceGather.hpp"
#include "mesh/deviceMesh.cuh"
#include "ops/fvcOps.cuh"
#include "solvers/gncHisa/hisaFlux.cuh"
#include "solvers/gncHisa/hisaJacobian.cuh"
#include "solvers/gncHisa/hisaLinearSolver.cuh"
#include "solvers/gncHisa/hisaPseudoTime.cuh"
#include "solvers/gncHisa/hisaState.cuh"
#include "solvers/gncHisa/hisaUpdateFields.cuh"
#include "turbulence/kOmegaSST.cuh"
#include "turbulence/spalartAllmaras.cuh"

#include <functional>
#include <memory>
#include <string>

namespace gnc
{

// fvSolution 的 pseudoTime（pseudotimeControl::read）
struct PseudotimeControls
{
    int nPseudoCorr = 20;
    int nPseudoCorrMin = 1;
    // hisaModule 的預設：pseudoTol 1e-5、pseudoTolRel 0；次序 (rho, rhoU, rhoE)
    double tol[5] = {1e-5, 1e-5, 1e-5, 1e-5, 1e-5};
    double tolRel[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
};

PseudotimeControls readPseudotimeControls(const std::string& fvSolutionPath);


struct PseudotimeState
{
    int corr = 0;
    bool converged = false;
    bool firstIteration = true;
    double residuals[5] = {};
    double initResiduals[5] = {};
    bool firstValid = false;
    double firstResiduals[5] = {};
    // state_ 中的 initResiduals、firstResiduals（corr == 2 時寫入）
    bool stateStored = false;
    double stateInit[5] = {};
    double stateFirst[5] = {};
};

// pseudotimeControl::loop（穩態）：為真時做一個外迭代。precision 為
// IOstream::defaultPrecision()（state_ 的寫出精度）
bool pseudotimeLoop(const PseudotimeControls& c, PseudotimeState& s, int precision);

// OCharStream << double（precision 位，%g）再以 OpenFOAM 的規則讀回
double streamRoundTrip(double v, int precision);


// controlDict（已展開）：startFrom startTime，startTime、endTime、deltaT；
// writePrecision 設定 IOstream::defaultPrecision()（預設 6）
struct TimeControls
{
    double startTime = 0;
    double endTime = 0;
    double deltaT = 1;
    int writePrecision = 6;
};

TimeControls readTimeControls(const std::string& controlDictPath);


// 外迭代中 GMRES 剛結束（限界之前）的狀態，給測試程式傾印
struct OuterIterationView
{
    const HisaJacobian& J;
    const LusgsPlan& plan;
    const HisaVec& R;
    const PseudoTimeState& pt;
    const HisaState& s;
    const GmresResult& gmres;
};

using AfterSolveHook = std::function<void(const OuterIterationView&)>;


// 外迭代各段的牆鐘時間（秒，累計）。profile 為真時每段結束前
// cudaDeviceSynchronize，所以含主機端的時間與同步；不影響數值
struct OuterTimings
{
    double flux = 0;            // 通量與 U、能量邊界的 updateCoeffs
    double residual = 0;        // 無黏與黏性殘差
    double pseudoTime = 0;      // SER、偽時間步
    double jacobian = 0;        // p、T 的 updateCoeffs 與 Jacobian 組裝
    double gmres = 0;           // GMRES（LU-SGS 預條件）
    double update = 0;          // 限界與 updateFields
    double turbulence = 0;      // 紊流模型的 correct()
    int n = 0;                  // 外迭代數
};


class HisaSolver
{
public:
    // expandedDir：foamDictionary -expand -precision 17 展開的 p T U
    // thermophysicalProperties fvSchemes fvSolution（與 turbulenceProperties、
    // controlDict，有的話）
    HisaSolver
    (
        const std::string& caseDir,
        const std::string& expandedDir,
        const FoamCompat& compat = FoamCompat{}
    );

    // 一個時間步（穩態）；回傳 false 表示這一步是 writeAndEnd（沒有外迭代）。
    // afterJacobian 在 GMRES 之前（view.gmres 為上一個外迭代的結果），
    // afterSolve 在 GMRES 剛結束時呼叫
    bool timeStep
    (
        const AfterSolveHook& afterSolve = {},
        const AfterSolveHook& afterJacobian = {}
    );

    // hisaModule::outerIteration
    void outerIteration
    (
        const AfterSolveHook& afterSolve = {},
        const AfterSolveHook& afterJacobian = {}
    );

    // Time::run()：t < endTime - 0.5*deltaT 且還沒有 writeAndEnd
    bool running() const;

    // 外迭代結束時的 phi（尚未做過外迭代時是 createFields 的 phi）
    const SurfField<double>& phi() const;

    // faceLeastSquares 的向量（梯度方案用，後處理的 grad(U) 也用）
    const LSVectors& lsVectors() const { return ls_; }

    // 主機端資料（測試程式讀取）
    PolyMesh mesh;
    DeviceMesh dm;
    std::unique_ptr<FvSchemes> schemes;
    PseudoTimeControls ptc;
    GmresControls gmresControls;
    PseudotimeControls pseudotime;
    TimeControls time;
    FoamCompat compat;

    HisaState s;
    std::unique_ptr<SpalartAllmaras> sa;    // RAS（Spalart–Allmaras）時
    std::unique_ptr<KOmegaSST> sst;         // RAS（k-omega SST）時
    PseudoTimeState pt;
    PseudotimeState control;

    double t = 0;
    int timeIndex = 0;
    bool profile = false;
    OuterTimings timings;
    bool ended = false;             // writeAndEnd（收斂）
    int nOuterIterations = 0;
    GmresResult lastGmres;
    bool lastGmresValid = false;    // hisaModule 的 res_

private:
    LSVectors ls_;
    LusgsPlan plan_;
    HisaFlux flux_;
    bool fluxValid_ = false;
    bool energyBCsUpdated_ = false;     // he 的邊界條件 updated()（見 updateEnergyT）
    HisaResiduals res_;
    VolField<tens> tauMC_;
    HisaJacobian J_;

    void setPseudoCoNum();
};

} // namespace gnc

#endif
