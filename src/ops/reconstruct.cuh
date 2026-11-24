// HiSA 的 wVanLeer 重構（面的左、右狀態）。GPU-Native-CFD，GPL-3.0-or-later。
//
// 出處（Atlas upstream/hisa，GPL-3.0-or-later，CSIR；upstream/openfoam）：
//   reconstructionScheme.C          weights：wL = one + limL*(w - 1)，wR = limR*w；
//                                   內部面 L = wL⊙P + (one - wL)⊙N（逐分量），R 同；
//                                   非耦合邊界 L = R = 邊界值
//   LimitedReconstructionScheme.C   逐分量的限制器，梯度為 fvc::grad(vf,
//                                   "grad(<name>)" 或方案參數指定的名稱)，
//                                   d = C[N] - C[P]，L 用 faceFlux = 1、R 用 -1；
//                                   非耦合邊界的限制器為 one
//   wVanLeer.H                      cd = (faceFlux >= 0 ? 1 - w : w)；
//                                   psi = r > 0 ? (r + r²)/(2r + cd(1 - r)²) : 0；
//                                   max(min(min(r/cd, psi), 1/cd), 0)
//   NVDTVD.H（OpenFOAM）            r = 2*(d & gradc)/(phiN - phiP) - 1，
//                                   |d & gradc| >= 1000|phiN - phiP| 時
//                                   2*1000*sign*sign - 1
// 必須以 --fmad=false 編譯。

#ifndef GNC_OPS_RECONSTRUCT_CUH
#define GNC_OPS_RECONSTRUCT_CUH

#include "fields/deviceField.cuh"
#include "mesh/deviceMesh.cuh"

namespace gnc
{

// grad：vf 的梯度（呼叫端依 fvSchemes 計算）；只用到內部格的值
template<class T>
void wVanLeerReconstruct
(
    const DeviceMesh& dm,
    const VolField<T>& vf,
    const VolField<typename GradOf<T>::type>& grad,
    SurfField<T>& L,
    SurfField<T>& R
);

} // namespace gnc

#endif
