// gncHisa：殘差計算前的邊界係數更新。GPU-Native-CFD，GPL-3.0-or-later。
//
// residualsUpdate.H 的 fvOptions(rho, U) 建構 fvMatrix<vector>(U) 時，fvMatrix
// 的建構子對 U 的每個 patch 呼叫 updateCoeffs()（不 evaluate；OpenFOAM
// fvMatrix.C）。所以殘差中的 snGrad(U)（拉普拉斯、sigmaDotU）看到的是更新後
// 的 refValue、valueFraction，而 U 的邊界值仍是上一步的值。
//
// U 的型別：
//   characteristicFarfieldVelocity  HiSA characteristicFarfieldVelocity
//                                   FvPatchVectorField::updateCoeffs，依法向
//                                   Mach 數 pM = phi/((rho*magSf)*sqrt(gamma/psi))：
//                                   pM <= -1 超音速流入  refValue = URef
//                                   pM >=  1 超音速流出  valueFraction = 0
//                                                       （refValue 不變）
//                                   pM <=  0 次音速流入  pp = 0.5*(pRef + p -
//                                            (rhoO*cO)*((URef - U) & n))，
//                                            refValue = URef - n*(pRef - pp)/(rhoO*cO)
//                                   其他     次音速流出  valueFraction = 0.5，
//                                            refValue = (URef & n)*n
//                                            - n*(pRef - p)/(rhoO*cO) + (U - (U & n)*n)
//                                   其餘 valueFraction = 1、refGrad = 0。p、U、
//                                   rhoO 為相鄰格值，cO = sqrt(gamma/psi)（格值）
//   inletOutlet                     valueFraction = neg(phi)（inletOutletFvPatchField
//                                   .txx:103）
//   fixedValue 族、zeroGradient、calculated、symmetry、slip、empty：不變
// Jacobian 用的 UCoeff、pCoeff 在 N3c 實作。
//
// fvOptions(rho, he) 同樣更新能量場的邊界（fixedEnergy、mixedEnergy 會先
// evaluate T 的邊界，再由 T 求 he）。在 hConst 熱力下這不改變殘差的值
// （alphaEff 的 gamma 是常數，rhoESource 邊界只差 -0*he 的正負號，he 為正），
// 但改變 T、he 的邊界狀態；N3c 實作。

#ifndef GNC_SOLVERS_GNCHISA_HISABOUNDARYUPDATE_CUH
#define GNC_SOLVERS_GNCHISA_HISABOUNDARYUPDATE_CUH

#include "core/foamCompat.hpp"
#include "mesh/deviceMesh.cuh"
#include "solvers/gncHisa/hisaFlux.cuh"
#include "solvers/gncHisa/hisaState.cuh"

namespace gnc
{

// U 的 updateCoeffs（fvMatrix<vector>(U) 的建構）；phi 為這一步的通量。
// characteristicFarfieldVelocity 另設 Jacobian 用的 UCoeff（tensor）、pCoeff
// （vector）與 TCoeff（vector::zero）
void updateCoeffsU(const DeviceMesh& dm, HisaState& s, const HisaFlux& flux);

// fvOptions(rho, he) 建構 fvMatrix(he) 時能量場邊界的 updateCoeffs：fixedEnergy、
// gradientEnergy、mixedEnergy 都先 Tw.evaluate()（heThermo 的能量邊界條件），
// 所以 T 的邊界值在殘差計算中途就改變：
//   characteristicWallTemperature     T_b = Tif*coeff，coeff = (cif + gammaif*un)/
//                                     (cif + un) 夾在 [0.9, 1.1]，un = (Uif - U_b) & nf
//   characteristicFarfieldTemperature updateCoeffs（refValue、valueFraction、refGrad、
//                                     TCoeff、pCoeff、UCoeff），再 mixed 的 evaluate
//   inletOutlet、mixed                (inletOutlet 先設 valueFraction = neg(phi)) evaluate
//   zeroGradient、fixedGradient       evaluate
//   fixedValue 族、constraint          不變
// 之後 fixedEnergy（T 為 fixedValue 族的 patch）以 == 指定 he_b = HE(p_b, T_b)。
// gradientEnergy、mixedEnergy 的係數不必算（不 evaluate，邊界值不變；hConst 時
// Jacobian 與殘差都不用到）。
// 只在整個 run 的第一個外迭代呼叫：he 從不 evaluate（updateFields 以 == 指定
// 邊界值），它的 updated() 在第一次 updateCoeffs 之後一直為真，之後建構
// fvMatrix(he) 時 updateCoeffs 直接返回，不再 Tw.evaluate()（N3d 發現）。
void updateEnergyT(const DeviceMesh& dm, HisaState& s, const HisaFlux& flux);

// jacobian::addBoundaryTerms 的 T.boundaryFieldRef().updateCoeffs()（在 p 之後）：
//   characteristicWallTemperature     T_b = Tif*coeff（fixedValue 的 updateCoeffs 以 == 指定）
//   characteristicFarfieldTemperature refValue、valueFraction、refGrad、TCoeff、pCoeff、
//                                     UCoeff（不 evaluate）
//   inletOutlet                       valueFraction = neg(phi)
//   其他                              不變
// 每個外迭代都呼叫（updateFields 的 T.correctBoundaryConditions 讓 updated() 歸零）。
void updateCoeffsT(const DeviceMesh& dm, HisaState& s, const HisaFlux& flux);

// jacobian::addBoundaryTerms 開頭的 p.updateCoeffs()（不 evaluate）：
//   characteristicWallPressure        UCoeff_ = rhoif*cif、gradient = ((rhoif*cif)*
//                                     ((Uif - U_b) & nf))*dc，兩者乘以限制
//                                     max(min(minGrad/stabilise(gradient, SMALL), 1), 0)，
//                                     minGrad = (pif - SMALL)*dc；係數 p 1、T 0、U UCoeff_*nf
//   characteristicFarfieldPressure    refValue、valueFraction、pCoeff、UCoeff（vector）
//   inletOutlet                       valueFraction = neg(phi)
// U 已在殘差中更新；T 的 updateCoeffs 以同樣的輸入重算，結果不變。
void updateCoeffsP
(
    const DeviceMesh& dm,
    HisaState& s,
    const HisaFlux& flux,
    const FoamCompat& compat
);

} // namespace gnc

#endif
