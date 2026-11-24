// gncHisa 外迭代求解之後：限界與 updateFields。GPU-Native-CFD，GPL-3.0-or-later。
//
// 照 HiSA（Atlas upstream/hisa/src/modules/hisa，GPL-3.0-or-later，CSIR）：
//   storePrevIter    hisaModule::outerIteration 求解前的 scalarVarsPrevIter_、
//                    vectorVarsPrevIter_（rho、rhoE、rhoU 的複本，含邊界值）
//   boundSolution    求解後的 localTimesteppingBounding 區塊：
//                      rhoMin/Max = lb/ub*rhoPrev
//                      eMin/Max   = lb/ub*(rhoEPrev/rhoPrev - 0.5*magSqr(rhoUPrev/rhoPrev))
//                      eTemp      = rhoE/rho - 0.5*magSqr(rhoU/rho)
//                    逐內部面判斷 owner、neighbour 是否越界，factor = 各面套用的
//                    boundingDecrease、(1 + boundingDecrease)/2、boundingIncrease
//                    的 min（初值 GREAT；只有 min，與面的次序無關）；再夾住
//                    rho、eTemp，rhoE = rho*(eTemp + 0.5*magSqr(rhoU/rho))；
//                    pseudoCoDamping = max(pseudoCoNumMin/pseudoCoNum,
//                    min(1, pseudoCoDamping*factor))
//   updateFields     updateFields.H：bound(rho, rhoMin)、U、e 的內部值、TMin／
//                    TMax 的能量限界（任一處 e 越過 he(p, TMin) 時 e = max(e, eBound)，
//                    TMax 同理；邊界依各 patch 的 operator=）、thermo.correct()、
//                    p 的 Newton 迭代（rho = p*psi）、
//                    rhoU、rhoE 的內部值、p、U、T 的 correctBoundaryConditions
//                    （係數為本外迭代 updateCoeffs 的值）、T 的邊界下限、
//                    e 的邊界 = he(p, T)、thermo.correct()、rho、rhoU、rhoE 的邊界值
//                    （characteristicWallPressure 的 patch 用 p*psi 的相鄰格值）
// GeometricField 的指定（operator=）在 symmetry、symmetryPlane 等
// transformFvPatchField 上改以新的內部值 evaluate（finishAssignment）；
// operator== 直接指定。
// 不支援（遇到時報錯）：rho 低於 rhoMin（bound() 的平均修正）。HiSA 的 p Newton
// 迭代沒有上限（計數器不會增加）；這裡在 1000 次時報錯。
// 紊流修正不在這裡：laminar（Stokes）的 correct() 不改場；Spalart–Allmaras 由
// HisaSolver 在 updateFields 之後呼叫（turbulence/spalartAllmaras.cuh）。

#ifndef GNC_SOLVERS_GNCHISA_HISAUPDATEFIELDS_CUH
#define GNC_SOLVERS_GNCHISA_HISAUPDATEFIELDS_CUH

#include "fields/deviceField.cuh"
#include "mesh/deviceMesh.cuh"
#include "solvers/gncHisa/hisaFlux.cuh"
#include "solvers/gncHisa/hisaPseudoTime.cuh"
#include "solvers/gncHisa/hisaState.cuh"

namespace gnc
{

struct PrevIterFields
{
    DField<double> rhoI, rhoB;
    DField<double> rhoEI, rhoEB;
    DField<vec3> rhoUI, rhoUB;
};

PrevIterFields storePrevIter(const HisaState& s);


// localTimesteppingBounding（c.localTimesteppingBounding 為假時不做事）
void boundSolution
(
    const DeviceMesh& dm,
    const PseudoTimeControls& c,
    const PrevIterFields& prev,
    HisaState& s,
    PseudoTimeState& pt
);


struct UpdateFieldsInfo
{
    int newtonIterations = 0;   // p 的 Newton 迭代次數（thermo.correct() 的次數減 2）
    bool energyBoundedMin = false;  // e 限界到 TMin（HiSA 的 log：Bounding e to TMin）
    bool energyBoundedMax = false;
};

// flux：本外迭代的 phi（inletOutlet 的 evaluate）
UpdateFieldsInfo updateFields
(
    const DeviceMesh& dm,
    const PseudoTimeControls& c,
    const PrevIterFields& prev,
    const HisaFlux& flux,
    HisaState& s
);

} // namespace gnc

#endif
