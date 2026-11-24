// 紊流模型（RAS、可壓縮）共用的部分：參考次序版。GPU-Native-CFD，GPL-3.0-or-later。
//
// 照 OpenFOAM v2606（Atlas upstream/openfoam）：
//   fvSolution 的方程鬆弛係數   solution.C（relaxationFactors/equations：先完全相符，
//                               再依倒序比對 pattern，最後 default）
//   nu = mu/rho                 CompressibleTurbulenceModel.H（rho 為求解器的 rho）
//   GeometricField::operator=   各 patch 的 operator=（calculated 類照抄運算式的
//                               邊界值，fixedValue 族不改值，transform 族之後 evaluate）
//   EddyDiffusivity::correctNut alphat = rho*nut/Prt、correctBoundaryConditions
//                               （compressible::alphatWallFunction：rho_w*nut_w/Prt）
//   牆距離                       wallDist（meshWave 或 Poisson）：格值；boundaryField 為
//                               fixedValue（wall patch，patchWave 的
//                               sqrt(distSqr) + SMALL）與 zeroGradient（其他 patch，
//                               constraint patch 用自己的型別，純量時同樣取格值）；
//                               nearWallDist（壁面函數的 y）

#ifndef GNC_TURBULENCE_TURBULENCECOMMON_CUH
#define GNC_TURBULENCE_TURBULENCECOMMON_CUH

#include "fields/deviceField.cuh"
#include "io/fvSchemes.hpp"
#include "io/polyMeshReader.hpp"
#include "mesh/deviceMesh.cuh"
#include "ops/fvcOps.cuh"

#include <string>

namespace gnc
{

// 方案字串的空白正規化（多個空白合成一個）
std::string trimmedScheme(const std::string& s);

// Switch 的字面值
bool parseSwitch(const std::string& w, const std::string& where);

// fvSolution 的 relaxationFactors/equations/<field>（沒有時不鬆弛）
struct EquationRelax
{
    bool relax = false;
    double factor = 0;
};

EquationRelax readEquationRelax(const std::string& fvSolutionPath, const std::string& fieldName);

// fvSolution 的 relaxationFactors/fields/<field>（GeometricField::relax() 用；
// 沒有時不鬆弛）
EquationRelax readFieldRelax(const std::string& fvSolutionPath, const std::string& fieldName);

// nu = mu/rho（內部與邊界；nu 由這裡配置）
void nuField
(
    const DeviceMesh& dm,
    const VolField<double>& mu,
    const VolField<double>& rho,
    VolField<double>& nu
);

// GeometricField::operator=(tmp) 的邊界部分：exprBnd 為運算式的邊界值
// （calculated、extrapolatedCalculated、zeroGradient 照抄；fixedValue 族與
// symmetry、symmetryPlane、slip 不改，後者由之後的 correctBoundaryConditions 處理）
void assignBoundary
(
    const DeviceMesh& dm,
    VolField<double>& f,
    const DeviceArray<double>& exprBnd
);

// EddyDiffusivity::correctNut：alphat = (rho*nut)/Prt、邊界（含
// compressible::alphatWallFunction）、correctBoundaryConditions
void eddyDiffusivityCorrectNut
(
    const DeviceMesh& dm,
    const VolField<double>& rho,
    const VolField<double>& nut,
    double Prt,
    VolField<double>& alphat
);

// fvSchemes 的 wallDist/method 必須是 meshWave
void requireMeshWave(const FvSchemes& schemes);

// meshWave 牆距離（網格不變時算一次）
struct WallDistanceField
{
    DeviceArray<double> y;          // 格值
    DeviceArray<double> yB;         // 每個有限體積邊界面：wallDist::y() 的邊界值
    DeviceArray<double> yNearWall;  // 每個有限體積邊界面：nearWallDist（壁面函數的 y）
};

WallDistanceField wallDistanceMeshWave(const DeviceMesh& dm, const PolyMesh& mesh);

// Poisson 牆距離（PoissonPatchDistMethod.C）：laplacian(yPsi) == -1（wall patch 的
// yPsi 為 0，其他 zeroGradient；yPsi 初值 0，所以非正交修正為 0），
// y = max(sqrt(magSqr(grad(yPsi)) + 2*yPsi) - mag(grad(yPsi)), SMALL)；邊界為
// y 的 operator=（wall 的 fixedValue 保留 SMALL，zeroGradient 取運算式的邊界值，
// symmetry 類取格值）再 clamp_min(SMALL)。OpenFOAM 以 GAMG 解到 fvSolution 的
// yPsi 容差，這裡以對角預條件的 PCG 解到正規化殘差 1e-10，所以不逐位元（N-022）。
// nearWallDist 與方法無關（幾何），取 meshWave 的輸出
WallDistanceField wallDistancePoisson
(
    const DeviceMesh& dm,
    const PolyMesh& mesh,
    const FvSchemes& schemes,
    const LSVectors& ls
);

// 依 fvSchemes 的 wallDist/method（meshWave 或 Poisson）
WallDistanceField wallDistanceFromSchemes
(
    const DeviceMesh& dm,
    const PolyMesh& mesh,
    const FvSchemes& schemes,
    const LSVectors& ls
);

} // namespace gnc

#endif
