// gncHisa 的求解器狀態與初始化。GPU-Native-CFD，GPL-3.0-or-later。
//
// 初始化照 HiSA hisaModule::initialise 的 createFields.H（Atlas
// upstream/hisa/src/modules/hisa/createFields.H，GPL-3.0-or-later，CSIR）：
//   thermo   fluidThermo::New：讀 p、T，he = HE(p, T)，calculate()
//   U        讀場檔
//   rho      thermo.rho() = p*psi，邊界條件 calculated
//   rhoU     rho*U
//   eZero    thermo.he(pZero, TZero)，pZero = TZero = SMALL
//   rhoE     rho*(e - eZero + 0.5*magSqr(U))
//   phi      mesh.Sf() & fvc::interpolate(rhoU)（linear）
// 紊流模型（N4a，HisaSolver 建立）、fvOptions 與 tauMC 不在這裡。

#ifndef GNC_SOLVERS_GNCHISA_HISASTATE_CUH
#define GNC_SOLVERS_GNCHISA_HISASTATE_CUH

#include "fields/deviceField.cuh"
#include "mesh/deviceMesh.cuh"
#include "thermo/psiThermo.cuh"

#include <string>

namespace gnc
{

struct HisaState
{
    PsiThermoFields thermo;     // p、T、e、psi、mu、alpha
    VolField<vec3> U;
    VolField<double> rho;
    VolField<vec3> rhoU;
    VolField<double> rhoE;
    VolField<double> eZero;
    SurfField<double> phi;
    bool inviscid = true;
    std::string pDimensions, TDimensions, UDimensions;
    // RAS（Spalart–Allmaras，N4a）時指向紊流模型的 nut、alphat；laminar 時為
    // nullptr（Stokes：nut = 0，alphaEff = alphahe）
    const VolField<double>* nut = nullptr;
    const VolField<double>* alphat = nullptr;
};


// expandedDir：以 foamDictionary -expand -precision 17 展開的
// p、T、U、thermophysicalProperties、fvSchemes
HisaState hisaInitialise
(
    const DeviceMesh& dm,
    const PolyMesh& mesh,
    const std::string& expandedDir
);

} // namespace gnc

#endif
