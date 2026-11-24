// 純量的 fvMatrix（隱式離散）：參考次序版。GPU-Native-CFD，GPL-3.0-or-later。
//
// 照 OpenFOAM v2606（Atlas upstream/openfoam）與 Atlas 的 lduMatrix overlay
// （atlas-overlay/openfoam/.../lduMatrix/{lduMatrixATmul,lduMatrixOperations}.C；
// GCC 建置的 executor 是主機上的循序迴圈，AtomicAdd 依面編號循序累加）：
//   係數    lduMatrix 的 diag、lower、upper 第一次取用時配置（diag 為 0；upper
//           沒有 lower 時為 0，否則複製 lower；lower 同理），symmetric 只有
//           upper；fvMatrix 另有 source、每個 patch 的 internalCoeffs、
//           boundaryCoeffs（建構時為 0）
//   運算    fvMatrix::operator+=、-=（lduMatrix::operator+=、-= 依矩陣型別的
//           分支）、negate、A - su（source += V*su）、su - A（negate 後
//           source -= V*su）
//   邊界    fixedValue 族、boundaryCorrectedFixedValue、zeroGradient 族、inletOutlet
//           （valueFraction = 1 - pos0(phi_b)，fvMatrix 建構時的 updateCoeffs）、
//           其他 mixed 族（refValue、refGrad、valueFraction 由呼叫端在 updateCoeffs
//           的時點算好：freestreamPressure、mixedEnergy）、fixedGradient 族
//           （gradientEnergy）、symmetry／symmetryPlane／slip（純量）
//   格式    ddt：bounded dualTime rPseudoDeltaT steadyState（HiSA
//           dualTimeDdtScheme::fvmDdt(rho, vf)；bounded 減去的 Sp(fvc::ddt(rho))
//           在 steadyState 下為 0）；div：bounded Gauss upwind 或 limitedLinear k
//           （gaussConvectionScheme::fvmDiv 減 Sp(surfaceIntegrate(phi))）或
//           linearUpwind grad（upwind 權重加顯式修正）；
//           laplacian：Gauss linear corrected（gaussLaplacianSchemes.C 的純量
//           特化，v2606 的 correctedSnGrad::corrected() 恆為真）；Sp
//   relax   fvMatrix::relax(alpha)
//   solve   fvMatrix<scalar>::solveSegregated：addBoundaryDiag、
//           addBoundarySource 後以 smoothSolver（symGaussSeidel）求解，
//           再 correctBoundaryConditions
// 格上的面迴圈都經過格面收集表（依面編號），symGaussSeidel 以層級排程
// （同 LU-SGS 的 LusgsPlan），每格的運算次序與循序版相同。只支援非耦合的
// patch（沒有 processor、cyclic）。必須以 --fmad=false 編譯。

#ifndef GNC_FVM_FVSCALARMATRIX_CUH
#define GNC_FVM_FVSCALARMATRIX_CUH

#include "fields/boundaryConditions.cuh"
#include "fields/deviceField.cuh"
#include "mesh/deviceMesh.cuh"
#include "solvers/gncHisa/hisaLinearSolver.cuh"

#include <string>
#include <vector>

namespace gnc
{

struct FvScalarMatrix
{
    DeviceArray<double> diag;
    DeviceArray<double> lower;
    DeviceArray<double> upper;
    DeviceArray<double> source;            // nCells
    DeviceArray<double> internalCoeffs;    // nFvBoundaryFaces
    DeviceArray<double> boundaryCoeffs;
    // faceFluxCorrectionPtr：fluxRequired 的場的 laplacian 非正交修正
    // （gammaMagSf*correction），negate、+=、-= 一併處理，flux() 加上它
    bool hasFaceFluxCorrection = false;
    SurfField<double> faceFluxCorrection;
    bool hasDiag = false;
    bool hasLower = false;
    bool hasUpper = false;

    bool diagonal() const { return hasDiag && !hasLower && !hasUpper; }
    bool symmetric() const { return hasDiag && !hasLower && hasUpper; }
    bool asymmetric() const { return hasDiag && hasLower && hasUpper; }
};

// fvMatrix(psi, dims)：source、internalCoeffs、boundaryCoeffs 為 0，沒有係數
FvScalarMatrix fvmEmpty(const DeviceMesh& dm);

// lduMatrix::diag()、lower()、upper() 的配置（非 const 版）
void ensureDiag(const DeviceMesh& dm, FvScalarMatrix& A);
void ensureLower(const DeviceMesh& dm, FvScalarMatrix& A);
void ensureUpper(const DeviceMesh& dm, FvScalarMatrix& A);

// fvMatrix::operator+=、-=
void addMatrix(const DeviceMesh& dm, FvScalarMatrix& A, const FvScalarMatrix& B);
void subtractMatrix(const DeviceMesh& dm, FvScalarMatrix& A, const FvScalarMatrix& B);

// fvMatrix::negate
void negateMatrix(FvScalarMatrix& A);

// A - su：source += V*su（su 為內部場）
void matrixMinusSu(const DeviceMesh& dm, FvScalarMatrix& A, const double* su);

// su - A：negate(A)；source -= V*su
void suMinusMatrix(const DeviceMesh& dm, const double* su, FvScalarMatrix& A);

// A + su：source -= V*su（su 為內部場）
void matrixPlusSu(const DeviceMesh& dm, FvScalarMatrix& A, const double* su);

// fvm::Sp(DimensionedField sp, psi)：diag += V*sp
FvScalarMatrix fvmSp(const DeviceMesh& dm, const double* sp);

// fvm::SuSp(DimensionedField susp, psi)（fvmSup.C）：diag += V*max(susp, 0)；
// source -= (V*min(susp, 0))*psi
FvScalarMatrix fvmSuSp(const DeviceMesh& dm, const double* susp, const VolField<double>& psi);

// fvMatrix::setValues(cellLabels, values)（fvMatrix.C 的 setValuesFromList）的
// 收集表。原本是依 cellLabels 的次序、每格依 cells() 的面次序（owner 面遞增、再
// neighbour 面遞增）的循序迴圈：把受限格對相鄰格 source 的貢獻移到右邊
// （source[鄰格] -= 係數*值），再把面係數設 0，最後 source[受限格] = 值*diag。
// 受限格之間的貢獻最後會被覆寫，一個面只連到一個非受限格；所以只有「非受限格
// 收到的貢獻」有次序，依原迴圈的次序排成每格的清單（主機端，網格不變時一次）。
// 同一格在 cellLabels 重複出現時，第二次的係數已是 0，不再有貢獻。
struct SetValuesPlan
{
    std::int32_t nConstrained = 0;
    DeviceArray<std::int32_t> cells;            // 受限格（不重複）
    std::int32_t nTargets = 0;
    DeviceArray<std::int32_t> targetCell;       // 收到貢獻的非受限格
    DeviceArray<std::int32_t> targetOff;        // nTargets + 1
    DeviceArray<std::int32_t> contribFace;      // 內部面
    DeviceArray<std::int32_t> contribCell;      // 值的來源（受限格）
    DeviceArray<std::int8_t> contribOwnerSide;  // 受限格是 owner
    std::int32_t nZeroFaces = 0;
    DeviceArray<std::int32_t> zeroFaces;        // 受限格的內部面（upper、lower 設 0）
    std::int32_t nZeroBnd = 0;
    DeviceArray<std::int32_t> zeroBnd;          // 受限格的有限體積邊界面（internalCoeffs、boundaryCoeffs 設 0）
};

SetValuesPlan buildSetValuesPlan
(
    const DeviceMesh& dm,
    const std::vector<std::int32_t>& owner,
    const std::vector<std::int32_t>& neighbour,
    const std::vector<std::int32_t>& cellLabels
);

// setValues(cellLabels, psi 的格值)：值取 psi 當下的格值（omegaWallFunction 的
// manipulateMatrix 傳 patchInternalField()）；psi[celli] = 值 不改變 psi
void setValuesFromPsi
(
    const SetValuesPlan& plan,
    FvScalarMatrix& A,
    const VolField<double>& psi
);

// fvm::ddt(rho, psi)，bounded dualTime rPseudoDeltaT steadyState：
// diag += (rPseudoDeltaT*rho)*V；source += ((rPseudoDeltaT*rho)*psi)*V
FvScalarMatrix fvmDdtBoundedDualTime
(
    const DeviceMesh& dm,
    const double* rPseudoDeltaT,
    const double* rho,
    const VolField<double>& psi
);

// fvm::div(phi, psi)，bounded Gauss upwind
FvScalarMatrix fvmDivBoundedGaussUpwind
(
    const DeviceMesh& dm,
    const SurfField<double>& phi,
    const VolField<double>& psi
);

// fvm::div(phi, psi)，bounded Gauss limitedLinear k：權重由 limiter 與 linear 權重
// 組成（gradPsi 為 fvc::grad(psi)，依 fvSchemes 的 grad(psi)）；邊界的 limiter 為 1
FvScalarMatrix fvmDivBoundedGaussLimitedLinear
(
    const DeviceMesh& dm,
    const SurfField<double>& phi,
    const VolField<double>& psi,
    const VolField<vec3>& gradPsi,
    double k
);

// fvm::div(phi, psi)，bounded Gauss linearUpwind grad：upwind 權重（pos0(phi)）的
// 矩陣，加上 fvc::surfaceIntegrate(phi*correction) 的顯式項（source -= V*...），
// correction = (Cf - C[c]) & gradPsi[c]，c = phi > 0 ? owner : neighbour
// （linearUpwind.C；邊界為 0）。gradPsi 由呼叫端依 gradSchemes 的 grad 名稱算
FvScalarMatrix fvmDivBoundedGaussLinearUpwind
(
    const DeviceMesh& dm,
    const SurfField<double>& phi,
    const VolField<double>& psi,
    const VolField<vec3>& gradPsi
);

// Laplacian 的 snGrad 方案：corrected，或 limited corrected limitCoeff
// （limitedSnGrad：correction = min(lc*|snGrad|/((1 - lc)*|corr| + SMALL), 1)*corr，
// snGrad = nonOrthDeltaCoeffs*(N - P)，邊界取 patch 的 snGrad()）
struct LaplacianSnGrad
{
    bool limited = false;
    double limitCoeff = 1.0;
};

// 由方案字串（"Gauss linear corrected"、"Gauss linear limited corrected 0.33"、
// "Gauss linear limited 0.33"）讀出；其他報錯
LaplacianSnGrad parseLaplacianScheme(const std::string& scheme, const std::string& name);

// fvm::laplacian(gamma, psi)，Gauss linear corrected 或 limited corrected。gradPsi
// 是 fvc::grad(psi)（依 fvSchemes 的 grad(psi)，含邊界值），用於非正交修正；
// phi 給 inletOutlet 的 valueFraction（fvMatrix 建構時 psi 的 updateCoeffs，
// 呼叫端先以 updateInletOutletCoeffs 更新）
FvScalarMatrix fvmLaplacianGaussLinearCorrected
(
    const DeviceMesh& dm,
    const VolField<double>& gamma,
    const VolField<double>& psi,
    const VolField<vec3>& gradPsi,
    const SurfField<double>* phi,
    const LaplacianSnGrad& snGrad = LaplacianSnGrad{}
);

// fvm::laplacian(gammaf, psi)，gammaf 為面場（surfaceScalarField），Gauss linear
// corrected。fluxRequired 時把 gammaMagSf*correction 存成 faceFluxCorrection
FvScalarMatrix fvmLaplacianSurfaceGamma
(
    const DeviceMesh& dm,
    const SurfField<double>& gammaf,
    const VolField<double>& psi,
    const VolField<vec3>& gradPsi,
    const SurfField<double>* phi,
    bool fluxRequired,
    const LaplacianSnGrad& snGrad = LaplacianSnGrad{}
);

// fvMatrix::flux()（非耦合 patch）：內部面 faceH(psi) = upper*psi[N] - lower*psi[P]，
// 邊界 internalCoeffs*psi[owner] - boundaryCoeffs，再加上 faceFluxCorrection
SurfField<double> matrixFlux
(
    const DeviceMesh& dm,
    const FvScalarMatrix& A,
    const VolField<double>& psi
);

// fvMatrix::relax(alpha)
void relaxMatrix
(
    const DeviceMesh& dm,
    FvScalarMatrix& A,
    const VolField<double>& psi,
    double alpha
);


// fvSolution 的 solvers/<name>（已展開）：smoothSolver（smoother symGaussSeidel）、
// PBiCGStab（preconditioner DILU）或 PCG（preconditioner DIC）；GAMG 只在效能版
// （GNC_GAMG=amg，-fast）以 PCG＋聚合式 AMG 前置條件代替（fvm/amg.cuh，不逐位元）
struct SmoothSolverControls
{
    std::string solver = "smoothSolver";
    std::string preconditioner = "none";
    int nCellsInCoarsestLevel = 10;     // GAMG
    int mergeLevels = 1;                // GAMG
    double tolerance = 1e-6;    // lduMatrix::defaultTolerance
    double relTol = 0;
    int minIter = 0;
    int maxIter = 1000;         // lduMatrix::defaultMaxIter
    int nSweeps = 1;
};

SmoothSolverControls readSmoothSolverControls
(
    const std::string& fvSolutionPath,
    const std::string& fieldName
);

struct SolverPerf
{
    double initialResidual = 0;
    double finalResidual = 0;
    int nIterations = 0;
    bool converged = false;
};

// fvMatrix<scalar>::solveSegregated（smoothSolver、symGaussSeidel），之後
// psi 的 correctBoundaryConditions（inletOutlet 需要 phi；coeffsUpdated 見
// evaluateBCs）。A 的 diag 求解後復原
SolverPerf solveSmoothSymGaussSeidel
(
    const DeviceMesh& dm,
    const LusgsPlan& plan,
    FvScalarMatrix& A,
    VolField<double>& psi,
    const SmoothSolverControls& ctrl,
    const SurfField<double>* phi,
    bool coeffsUpdated = false
);

// 對稱矩陣（laplacian）的 PCG，對角預條件。不逐位元（內積用固定分塊的樹狀
// 歸約，只取決於 n）；殘差的正規化照 lduMatrix::solver::normFactor。之後 psi 的
// correctBoundaryConditions。Poisson 牆距離（OpenFOAM 用 GAMG）用
SolverPerf solvePCGDiagonal
(
    const DeviceMesh& dm,
    FvScalarMatrix& A,
    VolField<double>& psi,
    double tolerance,
    double relTol,
    int maxIter
);

// PBiCGStab＋DILU 或 PCG＋DIC（PBiCGStab.C、PCG.C、DILUPreconditioner.C、
// DICPreconditioner.C）。前置條件為 "AMG" 時用 fvm/amg.cuh 的 V-cycle（效能版）。D、b 為已加上邊界係數的對角與右邊（addBoundaryDiag、
// addBoundarySource），x 為解（就地更新）。DILU／DIC 的分解與前後掃描是面編號次序的
// 循序迴圈：每格依面號收集（正向 lowFaces 遞增、反向 upFacesDesc 遞減），以 LU-SGS 的
// 層級排程平行，結果與循序相同；內積、範數為循序加總（sumSolver）
SolverPerf solveLduKrylov
(
    const DeviceMesh& dm,
    const LusgsPlan& plan,
    const double* D,
    const double* lower,
    const double* upper,
    const double* b,
    double* x,
    const SmoothSolverControls& ctrl
);

// fvMatrix<scalar>::solveSegregated，依 ctrl.solver 選 smoothSolver 或 Krylov；
// 之後 psi 的 correctBoundaryConditions
SolverPerf solveScalarMatrix
(
    const DeviceMesh& dm,
    const LusgsPlan& plan,
    FvScalarMatrix& A,
    VolField<double>& psi,
    const SmoothSolverControls& ctrl,
    const SurfField<double>* phi,
    bool coeffsUpdated = false
);

// bound(vsf, lowerBound)（bound.C）：min(vsf) < lowerBound 時內部值
// max(max(vsf, average(max(vsf, lb))*pos0(-vsf)), lb)，邊界為各 patch 的
// operator=(max(b, lb))：fixedValue 族與 mixed 族不改值，inletOutlet 族為
// vf*refValue + (1 - vf)*max(b, lb)，symmetry 類取新的相鄰格值，其他照抄。
// 回傳是否限界
bool boundField(const DeviceMesh& dm, VolField<double>& vsf, double lowerBound);

} // namespace gnc

#endif
