// gncHisa 的耦合 Jacobian（弱形式）。GPU-Native-CFD，GPL-3.0-or-later。
//
// 照 HiSA jacobian::constructJacobian（Atlas upstream/hisa/src/finiteVolume/
// jacobians，GPL-3.0-or-later，CSIR）：Lax-Friedrichs 無黏項
// （laxFriedrichs::addFluxTerms）→ 黏性 laplacian（laplacian::addFluxTerms，
// 黏性案例）→ 邊界項（addBoundaryTerms）→ 時間項（addTemporalTerms）。
// 內部面的公式與 HiSA-GPU-Research 的 lf.cu（D-022，已逐位元）相同；這裡另照
// LduMatrix 的 += / -= 規則重現每個陣列的值，包括零的正負號：
//   - 空的子區塊第一次 += 時 upper、lower 是複製（-= 時是取負），diag 是
//     0 + x（0 - x）；之後是逐項相加減（LduMatrixOperations.C）。
//   - 每個 fvj 運算子各自的 diag 由 0 開始，依面編號 negSumDiag；穩定項
//     laplacian 另在內部面之後依 patch 次序扣邊界面。
//   - 純量乘 tensor::I 時非對角是 s*0（正負號跟著 s）。
// 子區塊（HiSA 的命名；未知數次序 rho | rhoU | rhoE）：
//   SS00 dContByRho      -= stab
//   SV00 dContByRhoU     += grad(w, 1)
//   SS01 dContByRhoE     只有邊界項建立的 diag
//   VS00 dMomByRho       += div(w, -UU) + grad(w, theta)；黏性 -= laplacian(muEff, -rhoU/rho²)
//   VV00 dMomByRhoU      += div(w, U)*I + grad(w, U) + gradT(w, -a*U)；-= stab*I；
//                        黏性 -= laplacian(muEff, 1/rho)*I
//   VS01 dMomByRhoE      += grad(w, a)
//   SS10 dEnergyByRho    += div(w, -gammaE*U + 2*theta*U)；黏性 -= laplacian(alphaEff,
//                        -rhoE/rho² + |rhoU/rho|²/rho)
//   SV10 dEnergyByRhoU   += grad(w, gammaE - theta) + div(w, -a*UU)；黏性 -= laplacian
//                        (alphaEff, -rhoU/rho²)
//   SS11 dEnergyByRhoE   += div(w, gamma*U)；-= stab；黏性 -= laplacian(alphaEff, 1/rho)
// U = rhoU/rho，a = gamma - 1，theta = (0.5*a)*(U&U)，gammaE = (gamma*rhoE)/rho，
// stab = laplacian(0.5*lambdaConv, 1, 含實體邊界)，lambdaConv = (interpolate(c)
// + mag(interpolate(U) & Sf/|Sf|))/deltaCoeffs，c = sqrt(gamma/psi)，U 為註冊表的
// 速度。邊界項：laxFriedrichs::boundaryJacobian 與 laplacian::boundaryJacobian，
// 以各 patch 的 valueInternalCoeffs、gradientInternalCoeffs 對內部格的守恆量
// 求導，加到 diag。時間項：diag += (ddtCoeff*V)（SS00、SS11）與
// (ddtCoeff*V)*I（VV00）。
//
// 目前支援的邊界型別（p、U、T）：zeroGradient、fixedValue、
// boundaryCorrectedFixedValue、mixed、inletOutlet、symmetryPlane、symmetry、
// slip、empty。HiSA 特徵類（BlockCoupledBoundary）的係數在下一步實作，遇到報錯。

#ifndef GNC_SOLVERS_GNCHISA_HISAJACOBIAN_CUH
#define GNC_SOLVERS_GNCHISA_HISAJACOBIAN_CUH

#include "fields/deviceField.cuh"
#include "io/fvSchemes.hpp"
#include "mesh/deviceMesh.cuh"
#include "solvers/gncHisa/hisaPseudoTime.cuh"
#include "solvers/gncHisa/hisaState.cuh"

namespace gnc
{

template<class T>
struct FvjBlock
{
    DField<T> diag;     // nCells
    DField<T> upper;    // nInternalFaces（hasOffDiag 時）
    DField<T> lower;
    bool hasOffDiag = false;
};


struct HisaJacobian
{
    FvjBlock<double> SS00, SS01, SS10, SS11;
    FvjBlock<vec3> SV00, SV10, VS00, VS01;
    FvjBlock<tens> VV00;
    DField<double> rDiagCoeff;      // LU-SGS 的對角倒數（lusgs 建構子）
};


void assembleJacobian
(
    const DeviceMesh& dm,
    const FvSchemes& schemes,
    const HisaState& s,
    const PseudoTimeState& pt,
    HisaJacobian& J
);

} // namespace gnc

#endif
