// gncHisa 的層流黏性殘差。GPU-Native-CFD，GPL-3.0-or-later。
//
// 照 HiSA residualsUpdate.H 的黏性部分（!inviscid_；OpenFOAM.com 分支：
// muEff = turbulence->muEff()、alphaEff = turbulence->alphaEff()），層流
// （simulationType laminar，Stokes）或 RAS 的 Spalart–Allmaras（N4a）：
//   muEff     mut + mu，mut = rho*nut（CompressibleTurbulenceModel.H:146）；Stokes
//             的 nut 為 0：muEff = rho*0.0 + mu（Stokes.C:90）；calculated 邊界
//   alphaEff  laminar：alphahe = CpByCpv*alpha = gamma(p, T)*alpha
//             （ThermalDiffusivity.H:161、heThermo.C:650、809）；RAS
//             （EddyDiffusivity）：CpByCpv*(alpha + alphat)（heThermo.C:869）；
//             calculated 邊界
//   tauMC     muEff*dev2(T(grad(U)))；參考端的 T() 在內部場的行為見
//             core/foamCompat.hpp（transposeAliasing）。tauMC_ 是 createFields.H 由同一運算式建
//             構的場，tmp 一路重用 grad(U) 的場，邊界型別與 grad(U) 相同
//             （faceLeastSquares：zeroGradient；Gauss：extrapolatedCalculated）；
//             賦值時對稱面重新 evaluate（finishAssignment）
//   rhoUR    += laplacian(muEff, U)（Gauss linear compact）
//   rhoUR    += div(tauMC)（Gauss linear）
//   sigmaDotU = (interpolate(muEff)*snGrad(U) + (Sf/magSf & interpolate(tauMC))) & Up
//             snGrad 為 corrected：nonOrthDeltaCoeffs*(N - P)，邊界取 patch 的
//             snGrad()（係數已由 updateCoeffsU 更新，見 hisaBoundaryUpdate.cuh），再加 nonOrthCorrectionVectors & interpolate(grad(U))
//             （snGradScheme.C:92、169，correctedSnGrads.C 的向量特化
//             fullGradCorrection；邊界的修正向量為 +0）
//   rhoER    += div(sigmaDotU)
//   eCalc     rhoE/rho - 0.5*magSqr(U)，calculated 邊界
//   rhoER    += laplacian(alphaEff, eCalc)（方案名稱 laplacian(alphaEff,e)）
// fvc::div、fvc::laplacian 的結果場是 extrapolatedCalculated（constraint patch
// 覆寫），邊界值經 correctBoundaryConditions；+= 對邊界逐項相加
// （transformFvPatchField 只改寫 operator=）。二元運算後的
// correctLocalBoundaryConditions 只對耦合 patch 有作用（fvPatchField::
// evaluateLocal 為空），本專案目前沒有耦合 patch。

#ifndef GNC_SOLVERS_GNCHISA_HISAVISCOUS_CUH
#define GNC_SOLVERS_GNCHISA_HISAVISCOUS_CUH

#include "core/foamCompat.hpp"
#include "io/fvSchemes.hpp"
#include "ops/fvcOps.cuh"
#include "solvers/gncHisa/hisaFlux.cuh"
#include "solvers/gncHisa/hisaState.cuh"

#include <string>

namespace gnc
{

// constant/turbulenceProperties（已展開）：simulationType laminar 且 laminar 模型
// 為 Stokes（沒有 laminar 子字典時預設 Stokes），或 simulationType RAS 且 RASModel
// 為 SpalartAllmaras（N4a）或 kOmegaSST（N4b）；其他報錯
enum class TurbulenceModelKind
{
    laminarStokes,
    spalartAllmaras,
    kOmegaSST
};

TurbulenceModelKind readTurbulenceModelKind(const std::string& turbulencePropertiesPath);

// 把層流黏性項加到 res（inviscidResiduals 與 updateCoeffsU 之後呼叫）；
// tauMC 為 HiSA 的儲存場 tauMC_ 在這一步的值
void addLaminarViscousResiduals
(
    const DeviceMesh& dm,
    const LSVectors& ls,
    const FvSchemes& schemes,
    const HisaState& s,
    const HisaFlux& flux,
    const FoamCompat& compat,
    HisaResiduals& res,
    VolField<tens>& tauMC
);

} // namespace gnc

#endif
