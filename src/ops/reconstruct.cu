// GPU-Native-CFD，GPL-3.0-or-later。見 reconstruct.cuh。

#include "ops/reconstruct.cuh"

namespace gnc
{

namespace
{

constexpr int BS = 256;

int nBlocks(const std::int32_t n)
{
    return int((n + BS - 1)/BS);
}

// 梯度的第 k 欄（LimitedReconstructionScheme.C 的 columnComponent）：
// 純量場的梯度是向量本身；向量場的梯度張量第 k 欄為 (x k, y k, z k)
GNC_HD vec3 column(const vec3& g, int)
{
    return g;
}

GNC_HD vec3 column(const tens& g, const int k)
{
    if (k == 0)
    {
        return {g.xx, g.yx, g.zx};
    }
    if (k == 1)
    {
        return {g.xy, g.yy, g.zy};
    }
    return {g.xz, g.yz, g.zz};
}

GNC_HD double foamSign(const double s)
{
    return (s >= 0) ? 1.0 : -1.0;
}

GNC_HD double foamMin(const double a, const double b)
{
    return (a < b) ? a : b;
}

// NVDTVD::r
GNC_HD double nvdtvdR
(
    const double faceFlux,
    const double phiP,
    const double phiN,
    const vec3& gradcP,
    const vec3& gradcN,
    const vec3& d
)
{
    const double gradf = phiN - phiP;
    const double gradcf = faceFlux > 0 ? dot(d, gradcP) : dot(d, gradcN);
    if (::fabs(gradcf) >= 1000*::fabs(gradf))
    {
        return 2*1000*foamSign(gradcf)*foamSign(gradf) - 1;
    }
    return 2*(gradcf/gradf) - 1;
}

// wVanLeerLimiter::limiter
GNC_HD double wVanLeerLimiter
(
    const double cdWeight,
    const double faceFlux,
    const double phiP,
    const double phiN,
    const vec3& gradcP,
    const vec3& gradcN,
    const vec3& d
)
{
    const double r = nvdtvdR(faceFlux, phiP, phiN, gradcP, gradcN, d);
    const double cd = (faceFlux >= 0 ? 1.0 - cdWeight : cdWeight);
    const double psi = (r > 0 ? (r + r*r)/(2*r + cd*((1 - r)*(1 - r))) : 0.0);
    // max(..., 0)：MAXMIN(double, double, int)，(0 < x) ? x : 0
    return foamMax(foamMin(foamMin(r/cd, psi), 1.0/cd), 0.0);
}


template<class T>
__global__ void kWVanLeerInternal
(
    const std::int32_t nInternal,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const double* __restrict__ w,
    const CView<vec3> C,
    const CView<T> vf,
    const CView<typename GradOf<T>::type> grad,
    const DView<T> outL,
    const DView<T> outR
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nInternal)
    {
        return;
    }
    const std::int32_t P = owner[f];
    const std::int32_t N = neighbour[f];
    const double cdw = w[f];
    const vec3 d = C[N] - C[P];
    const T vP = vf[P];
    const T vN = vf[N];
    const auto gP = grad[P];
    const auto gN = grad[N];

    T L, R;
    for (int k = 0; k < NComp<T>::value; ++k)
    {
        const double phiP = cmpt(vP, k);
        const double phiN = cmpt(vN, k);
        const vec3 gcP = column(gP, k);
        const vec3 gcN = column(gN, k);
        const double limL = wVanLeerLimiter(cdw, 1.0, phiP, phiN, gcP, gcN, d);
        const double limR = wVanLeerLimiter(cdw, -1.0, phiP, phiN, gcP, gcN, d);
        // reconstructionScheme::weights
        const double wL = 1.0 + limL*(cdw - 1.0);
        const double wR = limR*cdw;
        // reconstructionScheme::reconstruct（cmptMultiply）
        setCmpt(L, k, wL*phiP + (1.0 - wL)*phiN);
        setCmpt(R, k, wR*phiP + (1.0 - wR)*phiN);
    }
    outL.set(f, L);
    outR.set(f, R);
}

CView<vec3> meshVec(const DeviceVec3Array& a)
{
    CView<vec3> v;
    v.p[0] = a.x.data();
    v.p[1] = a.y.data();
    v.p[2] = a.z.data();
    return v;
}

} // namespace


template<class T>
void wVanLeerReconstruct
(
    const DeviceMesh& dm,
    const VolField<T>& vf,
    const VolField<typename GradOf<T>::type>& grad,
    SurfField<T>& L,
    SurfField<T>& R
)
{
    L.in.allocate(std::size_t(dm.nInternalFaces));
    R.in.allocate(std::size_t(dm.nInternalFaces));
    if (dm.nInternalFaces)
    {
        kWVanLeerInternal<T><<<nBlocks(dm.nInternalFaces), BS>>>
        (
            dm.nInternalFaces, dm.owner.data(), dm.neighbour.data(),
            dm.weights.data(), meshVec(dm.C), vf.in.cview(), grad.in.cview(),
            L.in.view(), R.in.view()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    L.bnd.copyFrom(vf.bnd);
    R.bnd.copyFrom(vf.bnd);
}


template void wVanLeerReconstruct<double>
(const DeviceMesh&, const VolField<double>&, const VolField<vec3>&, SurfField<double>&, SurfField<double>&);
template void wVanLeerReconstruct<vec3>
(const DeviceMesh&, const VolField<vec3>&, const VolField<tens>&, SurfField<vec3>&, SurfField<vec3>&);

} // namespace gnc
