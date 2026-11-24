// 向量的 fvMatrix（fvVectorMatrix）：參考次序版（N6）。GPU-Native-CFD，GPL-3.0-or-later。
//
// 照 OpenFOAM v2606（Atlas upstream/openfoam）：
//   係數    lduMatrix 的 diag、lower、upper 為純量，配置規則與 FvScalarMatrix 相同；
//           source、internalCoeffs、boundaryCoeffs 為向量（fvMatrix.C）
//   運算    operator+=、-=、negate、A + su（source -= V*su）、su - A、
//           A == su（source += V*su）
//   格式    div：bounded Gauss linearUpwind grad（gaussConvectionScheme::fvmDiv 的
//           upwind 權重矩陣，加上 surfaceIntegrate(phi*correction) 的顯式項，
//           linearUpwind<vector>::correction = (Cf - C[c]) & gradU[c]，
//           c = phi > 0 ? owner : neighbour；bounded 減 Sp(surfaceIntegrate(phi))）；
//           laplacian：純量 gamma 的 Gauss linear corrected（gaussLaplacianSchemes.C，
//           correction = nonOrthCorrectionVectors & linear interpolate(gradU)，
//           correctedSnGrad<vector>::correction 即 fullGradCorrection）
//   relax   fvMatrix::relax(alpha)：D += cmptMax(cmptMag(iCoeffs))、
//           D = max(|D|, sumOff)、D /= alpha、D -= cmptMin(iCoeffs)、S += (D - D0)*psi
//   D、A、H fvMatrix::D() = diag + 各邊界面 cmptAv(iCoeffs)；A() = D()/V；
//           H()：每個分量 boundaryDiagCmpt = -(Σ iCoeffs.cmpt) + Σ cmptAv(iCoeffs)，
//           H = boundaryDiagCmpt*psi + (lduMatrix::H(psi) + source)，再加
//           boundaryCoeffs，除以 V；無效分量（2D 的空方向）為 0
//   solve   fvMatrix::solveSegregated：source 加上 boundaryCoeffs，每個有效分量
//           diag 加 iCoeffs.cmpt 後以 PBiCGStab（DILU）解，之後 correctBoundaryConditions
//   邊界    fixedValue 族（含 noSlip）、zeroGradient 族、mixed 族（freestreamVelocity：
//           valueFraction 由呼叫端在 updateCoeffs 的時點算好）
// 面迴圈都經過格面收集表（依面編號），每格的運算次序與 OpenFOAM 的循序迴圈相同。
// 只支援非耦合的 patch。必須以 --fmad=false 編譯。

#ifndef GNC_FVM_FVVECTORMATRIX_CUH
#define GNC_FVM_FVVECTORMATRIX_CUH

#include "fields/deviceField.cuh"
#include "fvm/fvScalarMatrix.cuh"
#include "mesh/deviceMesh.cuh"
#include "solvers/gncHisa/hisaLinearSolver.cuh"

namespace gnc
{

struct FvVectorMatrix
{
    DeviceArray<double> diag;
    DeviceArray<double> lower;
    DeviceArray<double> upper;
    DField<vec3> source;            // nCells
    DField<vec3> internalCoeffs;    // nFvBoundaryFaces
    DField<vec3> boundaryCoeffs;
    bool hasDiag = false;
    bool hasLower = false;
    bool hasUpper = false;

    bool diagonal() const { return hasDiag && !hasLower && !hasUpper; }
    bool symmetric() const { return hasDiag && !hasLower && hasUpper; }
    bool asymmetric() const { return hasDiag && hasLower && hasUpper; }
};

// fvMatrix(psi, dims)：source、internalCoeffs、boundaryCoeffs 為 0，沒有係數
FvVectorMatrix fvmEmptyVector(const DeviceMesh& dm);

// 複製（tmp<fvMatrix>::New(A)，operator== 用）
FvVectorMatrix copyMatrix(const FvVectorMatrix& A);

// fvMatrix::operator+=、-=、negate
void addMatrix(const DeviceMesh& dm, FvVectorMatrix& A, const FvVectorMatrix& B);
void subtractMatrix(const DeviceMesh& dm, FvVectorMatrix& A, const FvVectorMatrix& B);
void negateMatrix(FvVectorMatrix& A);

// A + su：source -= V*su
void matrixPlusSu(const DeviceMesh& dm, FvVectorMatrix& A, const DField<vec3>& su);

// su - A：negate(A)；source -= V*su
void suMinusMatrix(const DeviceMesh& dm, const DField<vec3>& su, FvVectorMatrix& A);

// A == su：source += V*su
void matrixEqSu(const DeviceMesh& dm, FvVectorMatrix& A, const DField<vec3>& su);

// fvm::div(phi, U)，bounded Gauss linearUpwind grad；gradU 依 gradSchemes 的 grad 名稱
FvVectorMatrix fvmDivBoundedGaussLinearUpwind
(
    const DeviceMesh& dm,
    const SurfField<double>& phi,
    const VolField<vec3>& U,
    const VolField<tens>& gradU
);

// fvm::laplacian(gamma, U)，Gauss linear corrected；gamma 以 linear 內插，gradU 為
// fvc::grad(U)（依 grad(U) 的方案）
FvVectorMatrix fvmLaplacianGaussLinearCorrected
(
    const DeviceMesh& dm,
    const VolField<double>& gamma,
    const VolField<vec3>& U,
    const VolField<tens>& gradU
);

// fvMatrix::relax(alpha)
void relaxMatrix
(
    const DeviceMesh& dm,
    FvVectorMatrix& A,
    const VolField<vec3>& U,
    double alpha
);

// fvMatrix::A()：D()/V，extrapolatedCalculated 邊界
VolField<double> matrixA(const DeviceMesh& dm, const FvVectorMatrix& A);

// fvMatrix::H()：extrapolatedCalculated 邊界，無效分量為 0
VolField<vec3> matrixH(const DeviceMesh& dm, const FvVectorMatrix& A, const VolField<vec3>& U);

struct VectorSolverPerf
{
    SolverPerf cmpt[3];
    bool solved[3] = {false, false, false};

    // cmptMax(initialResidual())（沒解的分量為 0）
    double maxInitialResidual() const;
};

// fvMatrix<vector>::solveSegregated（PBiCGStab＋DILU），之後 U 的
// correctBoundaryConditions（coeffsUpdated 見 evaluateBCs）。A 不變
VectorSolverPerf solveVectorMatrix
(
    const DeviceMesh& dm,
    const LusgsPlan& plan,
    const FvVectorMatrix& A,
    VolField<vec3>& U,
    const SmoothSolverControls& ctrl,
    const SurfField<double>* phi,
    bool coeffsUpdated
);

} // namespace gnc

#endif
