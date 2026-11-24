// 力與力係數（OpenFOAM v2606 的 functionObjects::forces、forceCoeffs，Atlas
// upstream/openfoam/src/functionObjects/forces）。GPU-Native-CFD，GPL-3.0-or-later。
//
// 只支援 HiSA 案例用到的子集（其他設定報錯）：
//   - patches 為 patch 名稱或 patch group（不支援 regex）
//   - p 的單位為壓力，rho(p) = 1（forces.C:337）；pRef 預設 0
//   - 沒有 porosity、directForceDensity；座標系統以 CofR 與 liftDir／dragDir
//     （forceCoeffs）或 (0 0 1)／(1 0 0)（forces）建 cartesian（forces.C:51）
//   - 黏性應力以 compressible 紊流模型：devRhoReff = -muEff*devTwoSymm(grad(U))，
//     muEff = rho_b*nut_b + mu_b（CompressibleTurbulenceModel.H:142、154；laminar 的
//     nut 為 0）
//
// 在主機端依 OpenFOAM 的次序計算（只有選定 patch 的面，量少）：
//   fP = (rhoRef*Sf)*(p - pRef)，rhoRef = rho(p) = 1          forces.C:729
//   fV = Sf & ((-muEff)*devTwoSymm(gradU))                     forces.C:731
//   每個 patch：sumP += sum(fP)、sumV += sum(fV)（sum 由 0 依序累加；patch 依編號遞增），
//   mP = Md ^ fP、mV = Md ^ fV，Md = Cf - origin                forces.C:355
//   localVector(v) = v & R，R 的行為 (e1, e2, e3)（axesRotation.C:68，E3_E1）
//   forceCoeffs：pDyn = 0.5*rhoInf*sqr(magUInf)；
//     Cf = (1/(Aref*pDyn + SMALL))*localVector(F)、Cm = (1/(Aref*pDyn*lRef + SMALL))*...
//     （forceCoeffs.C:157、181），係數與 setResult 的名稱同 forceCoeffs.C:130、412
//
// setResult 的存放：scalar 以原值存入字典；vector 經過文字（OStringStream，精度為
// IOstream::defaultPrecision，即 controlDict 的 writePrecision）再讀回，所以 forces 的
// pressureForce 等只有 writePrecision 位有效數字。ForceResult::vectors 照這個方式捨入
// （%.*g 再以 OpenFOAM 的規則讀回），Fp 等則是原值。逐面的 force、moment 場
// （force、moment；useNamePrefix 預設 false。reset 為 0 之後 += fP + fV）在 faceForce、faceMoment。

#ifndef GNC_POST_FORCECOEFFS_CUH
#define GNC_POST_FORCECOEFFS_CUH

#include "core/foamOps.cuh"
#include "io/polyMeshReader.hpp"

#include <string>
#include <utility>
#include <vector>

namespace gnc
{

class HisaSolver;

struct ForceFunction
{
    std::string name;               // controlDict 的 functions 項目名稱
    bool coeffs = false;            // forceCoeffs（否則 forces）
    std::vector<int> patchIDs;      // 遞增（patchSet(...).sortedToc()）
    vec3 origin{0, 0, 0};           // CofR
    tens R{};                       // 旋轉張量，行為局部軸 e1 e2 e3
    double pRef = 0;
    double rhoInf = 0;              // forceCoeffs
    double magUInf = 0;
    double lRef = 0;
    double Aref = 0;
    // 執行與寫出（gncHisa 寫 postProcessing 用）
    std::string executeControl = "timeStep";
    int executeInterval = 1;
    std::string writeControl = "timeStep";
    double writeInterval = 1;
    int resultPrecision = 6;        // controlDict 的 writePrecision
    int filePrecision = 6;          // function 的 writePrecision，否則同上
    bool writeToFile = true;
};

struct ForceResult
{
    // 全域座標的總和（sumPatchForcesP_ 等）
    vec3 Fp{0, 0, 0}, Fv{0, 0, 0}, Fi{0, 0, 0};
    vec3 Mp{0, 0, 0}, Mv{0, 0, 0}, Mi{0, 0, 0};
    // 與 OpenFOAM setResult 相同的名稱與值
    std::vector<std::pair<std::string, double>> scalars;
    std::vector<std::pair<std::string, vec3>> vectors;
    // 逐面的 force、moment 場的邊界值（長度 nFvBoundaryFaces，未選的 patch 為 0）
    std::vector<vec3> faceForce;
    std::vector<vec3> faceMoment;
};

// 讀 foamDictionary -expand 展開的 controlDict 中 type 為 forces、forceCoeffs 的
// function（#includeFunc 必須在案例目錄內展開，才會讀到案例自己的 system/<name>）
std::vector<ForceFunction> readForceFunctions
(
    const std::string& controlDictFile,
    const PolyMesh& mesh
);

// 以求解器目前的狀態計算（grad(U) 依 fvSchemes 重新計算）
ForceResult evaluateForces(const HisaSolver& sol, const ForceFunction& f);

// gncHisa 的 postProcessing 輸出，格式同 OpenFOAM 的 writeFile（writeFile.C、
// writeFileTemplates.C）：<case>/postProcessing/<名稱>/<startTime>/ 下 forceCoeffs 寫
// coefficient.dat（每列 Time 與各係數的總和，係數依名稱排序），forces 寫 force.dat、
// moment.dat（Time、total、pressure、viscous 三個向量的分量）。數值為科學記號、
// writePrecision 位（function 的 writePrecision，否則 controlDict 的）；檔案已存在時
// 改名為 <檔名>_<startTime>.dat
class ForceFiles
{
public:
    ForceFiles
    (
        const std::string& caseDir,
        const ForceFunction& f,
        const std::string& startTimeName
    );
    ~ForceFiles();
    ForceFiles(const ForceFiles&) = delete;
    ForceFiles& operator=(const ForceFiles&) = delete;

    void write(const std::string& timeName, const ForceResult& r);

private:
    struct Impl;
    Impl* impl_;
};

} // namespace gnc

#endif
