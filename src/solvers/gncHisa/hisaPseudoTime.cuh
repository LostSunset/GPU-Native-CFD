// gncHisa 的偽時間步。GPU-Native-CFD，GPL-3.0-or-later。
//
// 照 HiSA hisaModuleTimestepping.C 的 setPseudoDeltaT（逐格時間步，靜止網格）
// 與 hisaModule.C 的 ddtCoeff：
//   lambda_f   = interpolate(sqrt(gamma/psi)) + interpolate(mag(U))（linear）
//   frdt_f     = nonOrthDeltaCoeffs*lambda_f
//   r_c        = 0；對每個內部面 r = max(r, frdt)（owner 與 neighbour）；
//                type 為 wall 的 patch 另取 (0.5*nonOrthDeltaCoeffs_b)*
//                (sqrt(gamma_c/psi_c) + mag(U_c))
//   rPseudoDeltaT = r/pseudoCoField，pseudoCoField = pseudoCoDamping*pseudoCoNum；
//                邊界為 zeroGradient
//   ddtCoeff   = max(fvm::ddt(rho|rhoU|rhoE).diag())/V = (rPseudoDeltaT*V)/V
//                （dualTime rPseudoDeltaT steadyState：diag = 0 + rPseudoDeltaT*V）
// 第一個外迭代：pseudoCoNum 取 fvSolution 的 pseudoTime/pseudoCoNum（沒有
// uniform/pseudoCoNum 時），pseudoCoField = pseudoCoNum，pseudoCoDamping =
// pseudoCoField/pseudoCoNum；SER（setPseudoCoNum）在第一個外迭代不作用。
// 只支援 localTimestepping（預設）。

#ifndef GNC_SOLVERS_GNCHISA_HISAPSEUDOTIME_CUH
#define GNC_SOLVERS_GNCHISA_HISAPSEUDOTIME_CUH

#include "fields/deviceField.cuh"
#include "io/fvSchemes.hpp"
#include "mesh/deviceMesh.cuh"
#include "solvers/gncHisa/hisaState.cuh"

#include <string>

namespace gnc
{

// fvSolution 的 pseudoTime 字典（HiSA hisaModule::initialise、beginTimeStep）
struct PseudoTimeControls
{
    double pseudoCoNum = 1.0;
    bool localTimestepping = true;
    bool localTimesteppingBounding = true;
    double localTimesteppingLowerBound = 0.95;
    double localTimesteppingUpperBound = 1.05;
    double boundingDecrease = 0.5;
    double boundingIncrease = 2.0;
    double pseudoCoNumMin = 0.1;
    double pseudoCoNumMax = 25.0;
    double pseudoCoNumMaxIncreaseFactor = 1.2;
    double pseudoCoNumMinDecreaseFactor = 0.1;
    bool resetPseudo = false;
    // updateFields.H 的界限（同一個 pseudoTime 字典）
    double rhoMin = -1.0e+15;   // -GREAT
    double TMin = 3.0e-8;       // ROOTSMALL
    double TMax = 1.0e+15;      // GREAT（小於 GREAT 時才作用）
};

// fvSolutionPath：已以 foamDictionary -expand -precision 17 展開
PseudoTimeControls readPseudoTimeControls(const std::string& fvSolutionPath);


struct PseudoTimeState
{
    double pseudoCoNum = 1.0;
    // pseudoCoField/pseudoCoNum 建立：calculated，constraint patch 為該型別
    VolField<double> pseudoCoDamping;
    VolField<double> rPseudoDeltaT;
    DField<double> ddtCoeff;
};

// initialise：pseudoCoNum、pseudoCoField、pseudoCoDamping（不讀 uniform 檔）
PseudoTimeState initPseudoTime(const DeviceMesh& dm, const PseudoTimeControls& c);

// setPseudoDeltaT 與 ddtCoeff
void setPseudoDeltaT
(
    const DeviceMesh& dm,
    const FvSchemes& schemes,
    const HisaState& s,
    PseudoTimeState& pt
);

} // namespace gnc

#endif
