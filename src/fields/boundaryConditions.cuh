// 邊界條件：上傳、evaluate、snGrad。GPU-Native-CFD，GPL-3.0-or-later。
//
// 行為照 OpenFOAM v2606（Atlas upstream/openfoam，fields/fvPatchFields）：
//   fixedValue、calculated     evaluate 不改值；snGrad = dc*(pf - pif)
//                              （fvPatchField.txx:225）
//   boundaryCorrectedFixedValue（HiSA）evaluate 不改值；snGrad =
//                              (pf - pif)*nonOrthDeltaCoeffs
//   zeroGradient               evaluate：pf = pif；snGrad = 0
//   extrapolatedCalculated     evaluate：pf = pif；snGrad 同 calculated
//   symmetry、slip             純量：同 zeroGradient；向量與張量：
//                              pf = 0.5*(pif + transform(I - 2 n n, pif))，
//                              snGrad = (0.5*dc)*(transform(...) - pif)，
//                              n = Sf/magSf 逐面（basicSymmetryFvPatchField.txx）
//   symmetryPlane              同上，n 為整個 patch 的平面法向量
//                              （symmetryPlaneFvPatchField.txx）
//   mixed                      pf = lerp(pif + refGrad/dc, refValue, vf)；
//                              snGrad = lerp(refGrad, (refValue - pif)*dc, vf)
//                              （mixedFvPatchField.txx:218–249）
//   inletOutlet                mixed，evaluate 前 vf = neg(phi_b)、refGrad = 0
//                              （inletOutletFvPatchField.txx:103）
//   freestreamVelocity、freestreamPressure
//                              mixed，refValue = freestreamValue、refGrad = 0，
//                              valueFraction 由 updateFreestream*Coeffs 依 U_b 算
//   noSlip                     fixedValue，值為 0
//   kqRWallFunction            zeroGradient
//   empty                      有限體積層沒有面
//   nutLowReWallFunction、HiSA 的 boundaryCorrectedNutLowReWallFunction
//                              fixedValue 族，只能放在 wall patch；evaluate 時
//                              updateCoeffs 以 operator== 設為 calcNut() = 0；
//                              後者的 snGrad 用 nonOrthDeltaCoeffs
//   nutUSpaldingWallFunction、compressible::alphatWallFunction
//                              fixedValue 族，只能放在 wall patch；值由紊流模型
//                              的 correctNut 算（需要 U、nu、y、rho、nut），
//                              evaluate 不改值
// dc 是 fvPatch::deltaCoeffs()（mesh.deltaCoeffs() 的邊界值）。

#ifndef GNC_FIELDS_BOUNDARYCONDITIONS_CUH
#define GNC_FIELDS_BOUNDARYCONDITIONS_CUH

#include "fields/deviceField.cuh"
#include "io/fieldIO.hpp"
#include "mesh/deviceMesh.cuh"

namespace gnc
{

// 依 HostField 建立裝置場；邊界條件型別必須與 patch 型別相容
template<class T>
VolField<T> uploadVolField(const DeviceMesh& dm, const HostField& h);

SurfField<double> uploadSurfaceScalarField(const DeviceMesh& dm, const HostField& h);

// 建立全為 0 的體積場，邊界條件為 generic（zeroGradient 或
// extrapolatedCalculated），constraint patch（empty、symmetryPlane、
// symmetry）改用該 patch 的型別（fvPatchFieldNew.txx 的覆寫規則）
template<class T>
VolField<T> makeVolField(const DeviceMesh& dm, const std::string& name, Bc generic);

// GeometricField::correctBoundaryConditions：依 patch 順序 evaluate。
// inletOutlet 需要 phi（面場）。
// HiSA 的特徵類邊界條件沒有覆寫 evaluate：updateCoeffs 已在本外迭代呼叫過
// （updated() 為真）時，evaluate 以存下的係數走 mixed（characteristicFarfield*）
// 或 fixedGradient（characteristicWallPressure）的公式，characteristicWall-
// Temperature 是 fixedValue 族、不改值。coeffsUpdated 表示呼叫端已經對這個場
// 做過本外迭代的 updateCoeffs（hisaBoundaryUpdate）；否則遇到這些型別報錯。
// 能量場的型別不在這裡 evaluate。
template<class T>
void evaluateBCs
(
    const DeviceMesh& dm,
    VolField<T>& f,
    const SurfField<double>* phi,
    bool coeffsUpdated = false
);

// inletOutletFvPatchField::updateCoeffs：valueFraction = 1 - pos0(phi_b)（只改
// valueFraction，不改值；fvMatrix 建構時對 psi 呼叫）
template<class T>
void updateInletOutletCoeffs
(
    const DeviceMesh& dm,
    VolField<T>& f,
    const SurfField<double>& phi
);

// freestreamVelocity 的 updateCoeffs（freestreamVelocityFvPatchVectorField.cxx）：
// valueFraction = 0.5 - 0.5*(U_b & nf)/mag(U_b)，U_b 為當下的邊界值。之後的
// evaluateBCs(U, phi, true) 以這個 valueFraction 走 mixed 的 evaluate
void updateFreestreamVelocityCoeffs(const DeviceMesh& dm, VolField<vec3>& U);

// freestreamPressure 的 updateCoeffs：valueFraction = 0.5 + 0.5*(U_b & nf)/mag(U_b)
void updateFreestreamPressureCoeffs
(
    const DeviceMesh& dm,
    VolField<double>& p,
    const VolField<vec3>& U
);

// GeometricField::operator=(tmp) 的邊界部分。Boundary::operator= 逐 patch 呼叫
// fvPatchField::operator=：calculated 等照抄運算式的邊界值（呼叫前已寫入
// f.bnd）；transformFvPatchField 族（symmetryPlane、symmetry、slip）改以新的
// 內部值 evaluate()（transformFvPatchField.txx:149），向量在對稱面上與逐項
// 相加的結果不同。fixedValue 族與 mixed 族的 operator= 保留舊值，不支援。
template<class T>
void finishAssignment(const DeviceMesh& dm, VolField<T>& f);

// 各 patch 的 snGrad()，寫到長度 nFvBoundaryFaces 的 out
template<class T>
void snGradBCs(const DeviceMesh& dm, const VolField<T>& f, DField<T>& out);

// 下載成 HostField（供 writeField 寫出）；邊界條件參數一併帶出
template<class T>
HostField downloadVolField
(
    const DeviceMesh& dm,
    const VolField<T>& f,
    const std::string& cls,
    const std::string& dimensions
);

HostField downloadSurfaceScalarField
(
    const DeviceMesh& dm,
    const SurfField<double>& f,
    const std::string& dimensions
);

} // namespace gnc

#endif
