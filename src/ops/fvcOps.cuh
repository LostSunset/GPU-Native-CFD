// 顯式有限體積運算子（fvc）：參考次序版。GPU-Native-CFD，GPL-3.0-or-later。
//
// 每個運算子的公式與次序照 OpenFOAM v2606 與 HiSA（出處在 fvcOps.cu 各
// kernel 的註解）。格上的累加都經過格面收集表（design.md §3.2），依面
// 編號次序：先內部面（owner 加、neighbour 依運算子加或減），再依 patch
// 順序的邊界面，與 OpenFOAM 的面迴圈相同，不用 atomics。

#ifndef GNC_OPS_FVCOPS_CUH
#define GNC_OPS_FVCOPS_CUH

#include "fields/boundaryConditions.cuh"
#include "fields/deviceField.cuh"
#include "mesh/deviceMesh.cuh"

namespace gnc
{

// innerProduct<vector, Type>
template<class T> struct InnerOf;
template<> struct InnerOf<vec3> { using type = double; };
template<> struct InnerOf<tens> { using type = vec3; };

// 把 mesh 的每面陣列包成面場（內部面＋有限體積邊界面）
SurfField<vec3> meshSurfVector(const DeviceMesh& dm, const DeviceVec3Array& perFace);
SurfField<double> meshSurfScalar(const DeviceMesh& dm, const DeviceArray<double>& perFace);

// linear 內插：w*(P - N) + N；非耦合邊界取邊界值
template<class T>
void interpolateLinear(const DeviceMesh& dm, const VolField<T>& vf, SurfField<T>& out);

// linear dotInterpolate：S & (w*(P - N) + N)；邊界 S_b & vf_b
template<class T>
void dotInterpolateLinear
(
    const DeviceMesh& dm,
    const SurfField<vec3>& S,
    const VolField<T>& vf,
    SurfField<typename InnerOf<T>::type>& out
);

// fvc::surfaceIntegrate（= Gauss 的 fvc::div(ssf)）：結果除以 V
template<class T>
void surfaceIntegrate(const DeviceMesh& dm, const SurfField<T>& ssf, DField<T>& out);

// fvc::surfaceSum（owner 與 neighbour 都加）
void surfaceSum(const DeviceMesh& dm, const SurfField<double>& ssf, DeviceArray<double>& out);

// 面場逐面相乘：out = s*v（純量乘以任意型別）
template<class T>
void multiplySurf(const SurfField<double>& s, const SurfField<T>& v, SurfField<T>& out);

// Gauss linear 梯度（gaussGrad::calcGrad，含 correctBoundaryConditions）
template<class T>
void gaussGrad
(
    const DeviceMesh& dm,
    const VolField<T>& vf,
    VolField<typename GradOf<T>::type>& out
);

// cellLimited Gauss linear k（minmod）：純量與向量（cellLimitedGrad.C）
template<class T>
void cellLimitedGaussGrad
(
    const DeviceMesh& dm,
    const VolField<T>& vf,
    double k,
    VolField<typename GradOf<T>::type>& out
);

// HiSA faceLeastSquaresVectors
struct LSVectors
{
    SurfField<vec3> P;              // 內部面＋邊界面
    DField<vec3> N;                 // 內部面
    DeviceArray<double> weightSum;  // surfaceSum(magSf)
};

LSVectors faceLeastSquaresVectors(const DeviceMesh& dm);

// HiSA faceLeastSquaresGrad（linear 內插）
template<class T>
void faceLeastSquaresGrad
(
    const DeviceMesh& dm,
    const LSVectors& ls,
    const VolField<T>& vf,
    VolField<typename GradOf<T>::type>& out
);

// 依 fvSchemes 的梯度方案（"faceLeastSquares linear"、"Gauss linear"、
// "cellLimited Gauss linear k"）；其他方案報錯
template<class T>
void gradByScheme
(
    const DeviceMesh& dm,
    const LSVectors& ls,
    const std::string& scheme,
    const VolField<T>& vf,
    VolField<typename GradOf<T>::type>& out
);

// HiSA compact snGrad：未修正部分用 sqr(dc)*(delta & Sf)/magSf，再加
// nonOrthCorrVec & interpolate(grad)；grad 由呼叫端依 fvSchemes 算好
template<class T>
void snGradCompact
(
    const DeviceMesh& dm,
    const VolField<T>& vf,
    const VolField<typename GradOf<T>::type>& grad,
    SurfField<T>& out
);

// fvc::laplacian(gamma, vf)，Gauss linear compact
template<class T>
void laplacianCompact
(
    const DeviceMesh& dm,
    const VolField<double>& gamma,
    const VolField<T>& vf,
    const VolField<typename GradOf<T>::type>& grad,
    DField<T>& out
);

// fvc::div(volTensorField)，Gauss linear
void divGaussLinear(const DeviceMesh& dm, const VolField<tens>& t, DField<vec3>& out);

} // namespace gnc

#endif
