// gncHisa 的線性求解：LU-SGS 預條件器、Jacobian 乘向量、GMRES。
// GPU-Native-CFD，GPL-3.0-or-later。
//
// 照 HiSA（Atlas upstream/hisa/src，GPL-3.0-or-later，CSIR）：
//   lusgs::precondition     先乘 V；正向依格編號，每格 d* = rDiagCoeff*x，依子區塊
//                           次序（SS、SV、VS、VV；沒有非對角的 SS01 略過）把
//                           lower 項推給編號較大的鄰格（x_j -= dot(lower, d*)）；
//                           反向依格編號遞減，x_c = rDiagCoeff*x_c，再把 upper 項
//                           推給編號較小的鄰格。
//   jacobianMatrix::matrixMul 每個子區塊 Amul（diag，再依面編號 += lower、upper），
//                           依子區塊次序由 0 相加，最後除以 V。
//   gmres<2,1>::solve       左預條件、古典 Gram–Schmidt、延遲正規化、Givens、
//                           nKrylov 個方向一次 restart，restart 之間依 residualIO
//                           的規則停止；normFactor 與初始殘差照 gmresTemplates.C。
// GPU 上的次序：
//   - LU-SGS 以層級排程（wavefront）執行：同一層的格互不相依，一格一個 thread，
//     依 HiSA 套用的次序收集鄰格的貢獻（正向：來源格遞增，同一來源依子區塊、
//     再依面號；反向：來源格遞減），所以每格的運算次序與 HiSA 相同。
//   - 全域加總（gSum、sumProd、sumSqr…）用一個 thread 由第 0 格依序累加
//     （參考次序版），與循序執行時的 OpenFOAM 相同。
//   - Givens 旋轉與回代在主機上做（純量）。

#ifndef GNC_SOLVERS_GNCHISA_HISALINEARSOLVER_CUH
#define GNC_SOLVERS_GNCHISA_HISALINEARSOLVER_CUH

#include "fields/deviceField.cuh"
#include "io/polyMeshReader.hpp"
#include "mesh/deviceMesh.cuh"
#include "solvers/gncHisa/hisaJacobian.cuh"

#include <string>
#include <vector>

namespace gnc
{

// 5 個未知數的格向量：s0 = rho、v = rhoU、s1 = rhoE（只有內部值）
struct HisaVec
{
    DField<double> s0;
    DField<double> s1;
    DField<vec3> v;

    void allocate(const std::size_t n)
    {
        s0.allocate(n);
        s1.allocate(n);
        v.allocate(n);
    }
};


// LU-SGS 的層級排程與收集表（網格不變時建一次）
struct LusgsPlan
{
    std::vector<std::int32_t> fwdLevelOff;  // 每層在 fwdCells 中的起點
    std::vector<std::int32_t> revLevelOff;
    DeviceArray<std::int32_t> fwdLevelOffD; // 同上（裝置端，cooperative kernel 用）
    DeviceArray<std::int32_t> revLevelOffD;
    DeviceArray<std::int32_t> fwdCells;
    DeviceArray<std::int32_t> revCells;
    // 每格：正向的 lower 面（本格為 neighbour），依面號遞增；
    // 反向的 upper 面（本格為 owner），依 neighbour 遞減、同一 neighbour 內面號遞增
    DeviceArray<std::int32_t> lowOff;
    DeviceArray<std::int32_t> lowFaces;
    DeviceArray<std::int32_t> upOff;
    DeviceArray<std::int32_t> upFaces;
    // DILU／DIC 的反向掃描（lduMatrix 的面迴圈由大到小）：每格的 upper 面依面號遞減
    // （同 upOff）
    DeviceArray<std::int32_t> upFacesDesc;

    // 多色版（效能版選項，GNC_LUSGS=color；不與 HiSA 逐位元）：貪婪著色（依格
    // 編號，取相鄰格未用的最小色），同色的格互不相鄰。正向依色遞增、反向依色遞減；
    // 每格的「下」鄰格為色較小者，「上」為色較大者。cLowFace 的負值 -(f + 1)
    // 表示本格是該面的 owner（係數取 upper 區塊），非負值表示本格是 neighbour
    // （係數取 lower 區塊）
    std::vector<std::int32_t> colorOff;
    DeviceArray<std::int32_t> colorCells;
    DeviceArray<std::int32_t> cLowOff;
    DeviceArray<std::int32_t> cLowFace;
    DeviceArray<std::int32_t> cUpOff;
    DeviceArray<std::int32_t> cUpFace;
};

LusgsPlan buildLusgsPlan(const PolyMesh& mesh);


// GNC_PROFILE_GMRES=1 時 GMRES 內各部分累計的牆鐘時間（秒）
struct GmresProfile
{
    double precondition = 0;
    double matrixMul = 0;
    double reduce = 0;
    long nPrecondition = 0;
    long nMatrixMul = 0;
    long nReduce = 0;
};

GmresProfile& gmresProfile();

// lusgs::precondition（x 原地改寫）
void lusgsPrecondition
(
    const DeviceMesh& dm,
    const HisaJacobian& J,
    const LusgsPlan& plan,
    HisaVec& x
);

// jacobianMatrix::matrixMul（強形式，除以 V）
void jacobianMatrixMul
(
    const DeviceMesh& dm,
    const HisaJacobian& J,
    const HisaVec& x,
    HisaVec& out
);


// fvSolution 的 flowSolver（已展開）：solver GMRES，preconditioner LUSGS
//（或參考端的 gncLUSGSDump，數值相同）
struct GmresControls
{
    int nKrylov = 8;
    int maxIter = 5;
    // residualIO 的值，依分量 (rho, rhoU.x, rhoU.y, rhoU.z, rhoE)
    double tol[5] = {1e-12, 1e-12, 1e-12, 1e-12, 1e-12};
    double tolRel[5] = {0.01, 0.01, 0.01, 0.01, 0.01};
};

GmresControls readGmresControls(const std::string& fvSolutionPath);


struct GmresResult
{
    int iterations = 0;
    double initRes[5] = {};
    double finalRes[5] = {};
};

// gmres<2,1>::solve：R 為強形式殘差（rhoR、rhoUR、rhoER 的內部值），W 為
// rho、rhoU、rhoE（內部值就地加上解；空方向的 rhoU 分量連同邊界值設為 0；
// 最後 correctBoundaryConditions）
GmresResult gmresSolve
(
    const DeviceMesh& dm,
    const HisaJacobian& J,
    const LusgsPlan& plan,
    const GmresControls& ctrl,
    const HisaVec& R,
    VolField<double>& rho,
    VolField<vec3>& rhoU,
    VolField<double>& rhoE
);

} // namespace gnc

#endif
