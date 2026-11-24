// gncHisa 的通量（AUSM+up）與殘差。GPU-Native-CFD，GPL-3.0-or-later。
//
// 通量照 HiSA ausmPlusUpFluxScheme::calcFlux（Atlas upstream/hisa/src/
// finiteVolume/fluxSchemes/ausmPlusUp/ausmPlusUpFluxScheme.C，GPL-3.0-or-
// later，CSIR），靜止網格；殘差照 residualsUpdate.H 的無黏部分（穩態，
// fvOptions 為空）。
//
// 每個面（內部面與非耦合邊界面同一套公式，邊界面的左右狀態都是邊界值）：
//   un      = (U & Sf)/magSf（左、右）
//   c       = sqrt(2(gamma - 1)/(gamma + 1)*H)，H = max(rhoE/rho, SMALL)
//             + max(p/rho, SMALL)，左右以 reconstruct(T) 重構後
//             cL = cL²/stabilise(max(cL, unL), SMALL)、
//             cR = cR²/stabilise(max(cR, -unR), SMALL)、
//             c½ = stabilise(min(cL, cR), SMALL)
//   M±、P±  Liou 的分裂（beta = 1/8，alpha = 3/16）
//   M½      M+(ML) + M-(MR) 加上壓力擴散項 Mp（越過 0 時取 0.2 Mp）
//   p½      P+ pL + P- pR（lowMachAusm 時加速度擴散項）
//   phi     M½ c½ rho_up |Sf|；phiUp = phi U_up + p½ Sf；phiEp = phi H_up
//   Up      U_up |Sf|
// p、rho 以 reconstruct(rho) 重構，U 以 reconstruct(U)，c、H 以 reconstruct(T)。
//
// 殘差（穩態、無黏）：
//   rhoR  = -div(phi)   + rhoSource
//   rhoUR = -div(phiUp) + rhoUSource
//   rhoER = -div(phiEp) + rhoESource
// fvOptions 為空時 fvMatrix & psi 的值為 +0/(-V) = -0.0，所以
//   rhoSource = -0，rhoUSource = -0 + (-0)*U，
//   rhoESource = (-0 + (rhoUSource & U)) + (-0)*e
// 只影響零的正負號，但要逐位元相同就照樣算。

#ifndef GNC_SOLVERS_GNCHISA_HISAFLUX_CUH
#define GNC_SOLVERS_GNCHISA_HISAFLUX_CUH

#include "io/fvSchemes.hpp"
#include "ops/fvcOps.cuh"
#include "solvers/gncHisa/hisaState.cuh"

namespace gnc
{

struct HisaFlux
{
    SurfField<double> phi;
    SurfField<vec3> phiUp;
    SurfField<double> phiEp;
    SurfField<vec3> Up;
};


struct HisaResiduals
{
    VolField<double> rhoR;
    VolField<vec3> rhoUR;
    VolField<double> rhoER;
};


// 依 fvSchemes（fluxScheme 必須是 AUSMPlusUp；reconstruct(*) 必須是
// wVanLeer [梯度方案名稱]）計算通量
void ausmPlusUpFlux
(
    const DeviceMesh& dm,
    const LSVectors& ls,
    const FvSchemes& schemes,
    const HisaState& s,
    HisaFlux& out
);

// 穩態、無黏的殘差（N3b）
void inviscidResiduals
(
    const DeviceMesh& dm,
    const HisaState& s,
    const HisaFlux& flux,
    HisaResiduals& out
);

} // namespace gnc

#endif
