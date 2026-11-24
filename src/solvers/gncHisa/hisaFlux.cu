// GPU-Native-CFD，GPL-3.0-or-later。見 hisaFlux.cuh。

#include "solvers/gncHisa/hisaFlux.cuh"

#include "fields/boundaryConditions.cuh"
#include "ops/reconstruct.cuh"

#include <sstream>
#include <stdexcept>

namespace gnc
{

namespace
{

constexpr int BS = 256;

int nBlocks(const std::int32_t n)
{
    return int((n + BS - 1)/BS);
}

CView<vec3> meshVec(const DeviceVec3Array& a)
{
    CView<vec3> v;
    v.p[0] = a.x.data();
    v.p[1] = a.y.data();
    v.p[2] = a.z.data();
    return v;
}

GNC_HD double foamMin(const double a, const double b)
{
    return (a < b) ? a : b;
}

// stabilise(s, tol)（scalarImpl.H:462）
GNC_HD double stabilise(const double s, const double tol)
{
    return s >= 0 ? s + tol : s - tol;
}


// gamma = Cp/(Cp - CpMCv)（species::thermo::gamma），H、c（calcFlux 的前段）
__global__ void kHc
(
    const std::int32_t n,
    const ThermoParams P,
    const double* __restrict__ p,
    const double* __restrict__ T,
    const double* __restrict__ rho,
    const double* __restrict__ rhoE,
    double* __restrict__ H,
    double* __restrict__ c
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const double Cp = thermo::Cp(P, p[i], T[i]);
    const double gamma = Cp/(Cp - P.R);
    const double h = foamMax(rhoE[i]/rho[i], SMALL) + foamMax(p[i]/rho[i], SMALL);
    H[i] = h;
    c[i] = ::sqrt(2.0*(gamma - 1.0)/(gamma + 1.0)*h);
}


struct FaceIn
{
    CView<vec3> UL, UR;
    CView<double> cL, cR, pL, pR, rhoL, rhoR, HL, HR;
};

struct FaceOut
{
    DView<double> phi;
    DView<vec3> phiUp;
    DView<double> phiEp;
    DView<vec3> Up;
};


// ausmPlusUpFluxScheme::calcFlux 的逐面運算。faceIdx 為 nullptr 時是內部面
// （索引 = 面編號），否則是邊界面（索引 k，面編號 faceIdx[k]）。
__global__ void kAusmPlusUp
(
    const std::int32_t n,
    const std::int32_t* __restrict__ faceIdx,
    const CView<vec3> Sf,
    const double* __restrict__ magSf,
    const bool lowMach,
    const FaceIn in,
    const FaceOut out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const std::int32_t f = faceIdx ? faceIdx[i] : i;
    const vec3 sf = Sf[f];
    const double ms = magSf[f];

    const vec3 UL = in.UL[i];
    const vec3 UR = in.UR[i];
    const double unL = dot(UL, sf)/ms;
    const double unR = dot(UR, sf)/ms;

    double cL = in.cL[i];
    double cR = in.cR[i];
    cL = (cL*cL)/stabilise(foamMax(cL, unL), SMALL);
    cR = (cR*cR)/stabilise(foamMax(cR, -unR), SMALL);
    const double cf = stabilise(foamMin(cL, cR), SMALL);

    // 分裂 Mach 數與壓力（beta = 1/8，alpha = 3/16）
    const double ML = unL/cf;
    double Mp, pp;
    if (::fabs(ML) < 1.0)
    {
        const double ML2p = 0.25*((ML + 1)*(ML + 1));
        const double ML2m = -0.25*((ML - 1)*(ML - 1));
        Mp = ML2p*(1 - 2*ML2m);
        pp = ML2p*(2 - ML - 3*ML*ML2m);
    }
    else
    {
        Mp = foamMax(ML, 0.0);
        pp = (ML > 0 ? 1.0 : 0.0);
    }

    const double MR = unR/cf;
    double Mm, pm;
    if (::fabs(MR) < 1.0)
    {
        const double MR2m = -0.25*((MR - 1)*(MR - 1));
        const double MR2p = 0.25*((MR + 1)*(MR + 1));
        Mm = MR2m*(1 + 2*MR2p);
        pm = MR2m*(-2 - MR + 3*MR*MR2p);
    }
    else
    {
        Mm = foamMin(MR, 0.0);
        pm = (MR < 0 ? 1.0 : 0.0);
    }

    const double pL = in.pL[i];
    const double pR = in.pR[i];
    const double rhoL = in.rhoL[i];
    const double rhoR = in.rhoR[i];

    double M12 = Mp + Mm;
    double p12 = pp*pL + pm*pR;

    const double Mmean = 0.5*(unL*unL + unR*unR)/(cf*cf);
    const double MDiff =
        -0.25*foamMax((1.0 - Mmean), 0.0)*(pR - pL)/(0.5*(rhoL + rhoR)*(cf*cf));
    const double dM =
        ((M12 > 0.0 && M12 + MDiff <= 0.0) || (M12 < 0.0 && M12 + MDiff >= 0.0))
      ? 0.2*MDiff
      : MDiff;
    M12 += dM;

    if (lowMach)
    {
        const double pDiff = -0.25*pp*pm*(rhoL + rhoR)*cf*(unR - unL);
        p12 += pDiff;
    }

    const bool sel = M12 >= 0;
    const vec3 Uf = sel ? UL : UR;
    const double rhoa = M12*cf*(sel ? rhoL : rhoR);
    const vec3 rhoaU = rhoa*Uf;
    const double rhoah = rhoa*(sel ? in.HL[i] : in.HR[i]);

    out.Up.set(i, Uf*ms);
    out.phi.set(i, rhoa*ms);
    out.phiUp.set(i, rhoaU*ms + p12*sf);
    out.phiEp.set(i, rhoah*ms);
}


// 殘差（穩態、無黏，fvOptions 為空）。src 欄位見 hisaFlux.cuh。
__global__ void kResidualCells
(
    const std::int32_t n,
    const double* __restrict__ divPhi,
    const CView<vec3> divPhiUp,
    const double* __restrict__ divPhiEp,
    const CView<vec3> U,
    const double* __restrict__ e,
    double* __restrict__ rhoR,
    const DView<vec3> rhoUR,
    double* __restrict__ rhoER
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const double z = -0.0;
    const double rhoSource = z;
    const vec3 u = U[i];
    const vec3 rhoUSource = vec3{z, z, z} + rhoSource*u;
    const double rhoESource = z + dot(rhoUSource, u) + rhoSource*e[i];

    const vec3 dUp = divPhiUp[i];
    rhoR[i] = -divPhi[i] + rhoSource;
    rhoUR.set(i, vec3{-dUp.x, -dUp.y, -dUp.z} + rhoUSource);
    rhoER[i] = -divPhiEp[i] + rhoESource;
}


// 殘差運算式的邊界值。div 與 M & psi 的結果場以 extrapolatedCalculated 建立，
// constraint patch 改用同名型別（向量在對稱面上經過轉換），所以邊界值取
// 已 evaluate 的場（divB、MUB）；純量的 M & psi 邊界為 -0。賦值給殘差場時
// 對稱面上的值再改掉（finishAssignment）。
__global__ void kResidualBoundary
(
    const std::int32_t nB,
    const double* __restrict__ divPhiB,
    const CView<vec3> divPhiUpB,
    const double* __restrict__ divPhiEpB,
    const CView<vec3> MUB,
    const CView<vec3> Ub,
    const double* __restrict__ eb,
    double* __restrict__ rhoR,
    const DView<vec3> rhoUR,
    double* __restrict__ rhoER
)
{
    const std::int32_t k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB)
    {
        return;
    }
    const double z = -0.0;
    const double rhoSource = z;
    const vec3 u = Ub[k];
    const vec3 rhoUSource = MUB[k] + rhoSource*u;
    const double rhoESource = z + dot(rhoUSource, u) + rhoSource*eb[k];

    const vec3 dUp = divPhiUpB[k];
    rhoR[k] = -divPhiB[k] + rhoSource;
    rhoUR.set(k, vec3{-dUp.x, -dUp.y, -dUp.z} + rhoUSource);
    rhoER[k] = -divPhiEpB[k] + rhoESource;
}


__global__ void kFillNegZero(const std::int32_t n, double* __restrict__ a)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        a[i] = -0.0;
    }
}


// "wVanLeer" 或 "wVanLeer <梯度方案名稱>"；回傳梯度方案（gradSchemes 中的值）
std::string reconstructGradScheme
(
    const FvSchemes& schemes,
    const std::string& reconstructName,
    const std::string& fieldName
)
{
    std::istringstream is(schemes.interpolation(reconstructName));
    std::string kind, gradName, extra;
    is >> kind >> gradName >> extra;
    if (kind != "wVanLeer" || !extra.empty())
    {
        throw std::runtime_error
        (
            reconstructName + ": only wVanLeer [gradScheme] is supported"
        );
    }
    if (gradName.empty())
    {
        gradName = "grad(" + fieldName + ")";
    }
    return schemes.grad(gradName);
}


template<class T>
void reconstructField
(
    const DeviceMesh& dm,
    const LSVectors& ls,
    const FvSchemes& schemes,
    const std::string& reconstructName,
    const VolField<T>& vf,
    SurfField<T>& L,
    SurfField<T>& R
)
{
    VolField<typename GradOf<T>::type> g;
    gradByScheme
    (
        dm, ls, reconstructGradScheme(schemes, reconstructName, vf.name), vf, g
    );
    wVanLeerReconstruct(dm, vf, g, L, R);
}

} // namespace


void ausmPlusUpFlux
(
    const DeviceMesh& dm,
    const LSVectors& ls,
    const FvSchemes& schemes,
    const HisaState& s,
    HisaFlux& out
)
{
    if (schemes.top("fluxScheme", "") != "AUSMPlusUp")
    {
        throw std::runtime_error("fluxScheme must be AUSMPlusUp");
    }
    const std::string lm = schemes.top("lowMachAusm", "true");
    bool lowMach = true;
    if (lm == "false" || lm == "off" || lm == "no" || lm == "n" || lm == "f" || lm == "none")
    {
        lowMach = false;
    }
    else if (!(lm == "true" || lm == "on" || lm == "yes" || lm == "y" || lm == "t"))
    {
        throw std::runtime_error("lowMachAusm: unexpected value " + lm);
    }

    const std::int32_t nC = dm.nCells;
    const std::int32_t nI = dm.nInternalFaces;
    const std::int32_t nB = dm.nFvBoundaryFaces();

    // H 與 c（calculated 邊界）
    VolField<double> H = makeVolField<double>(dm, "H", Bc::calculated);
    VolField<double> c = makeVolField<double>(dm, "c", Bc::calculated);
    if (nC)
    {
        kHc<<<nBlocks(nC), BS>>>
        (
            nC, s.thermo.params, s.thermo.p.in.cview().p[0], s.thermo.T.in.cview().p[0],
            s.rho.in.cview().p[0], s.rhoE.in.cview().p[0], H.in.view().p[0], c.in.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (nB)
    {
        kHc<<<nBlocks(nB), BS>>>
        (
            nB, s.thermo.params, s.thermo.p.bnd.cview().p[0], s.thermo.T.bnd.cview().p[0],
            s.rho.bnd.cview().p[0], s.rhoE.bnd.cview().p[0], H.bnd.view().p[0], c.bnd.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }

    // 重構（左、右狀態）
    SurfField<vec3> UL, UR;
    SurfField<double> cL, cR, pL, pR, rhoL, rhoR, HL, HR;
    reconstructField(dm, ls, schemes, "reconstruct(U)", s.U, UL, UR);
    reconstructField(dm, ls, schemes, "reconstruct(T)", c, cL, cR);
    reconstructField(dm, ls, schemes, "reconstruct(rho)", s.thermo.p, pL, pR);
    reconstructField(dm, ls, schemes, "reconstruct(rho)", s.rho, rhoL, rhoR);
    reconstructField(dm, ls, schemes, "reconstruct(T)", H, HL, HR);

    out.phi.name = "phi";
    for (auto* f : {&out.phi.in, &out.phiEp.in})
    {
        f->allocate(std::size_t(nI));
    }
    for (auto* f : {&out.phi.bnd, &out.phiEp.bnd})
    {
        f->allocate(std::size_t(nB));
    }
    out.phiUp.in.allocate(std::size_t(nI));
    out.phiUp.bnd.allocate(std::size_t(nB));
    out.Up.in.allocate(std::size_t(nI));
    out.Up.bnd.allocate(std::size_t(nB));

    if (nI)
    {
        const FaceIn in
        {
            UL.in.cview(), UR.in.cview(), cL.in.cview(), cR.in.cview(),
            pL.in.cview(), pR.in.cview(), rhoL.in.cview(), rhoR.in.cview(),
            HL.in.cview(), HR.in.cview()
        };
        const FaceOut o
        {
            out.phi.in.view(), out.phiUp.in.view(), out.phiEp.in.view(), out.Up.in.view()
        };
        kAusmPlusUp<<<nBlocks(nI), BS>>>
        (
            nI, nullptr, meshVec(dm.Sf), dm.magSf.data(), lowMach, in, o
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (nB)
    {
        const FaceIn in
        {
            UL.bnd.cview(), UR.bnd.cview(), cL.bnd.cview(), cR.bnd.cview(),
            pL.bnd.cview(), pR.bnd.cview(), rhoL.bnd.cview(), rhoR.bnd.cview(),
            HL.bnd.cview(), HR.bnd.cview()
        };
        const FaceOut o
        {
            out.phi.bnd.view(), out.phiUp.bnd.view(), out.phiEp.bnd.view(), out.Up.bnd.view()
        };
        kAusmPlusUp<<<nBlocks(nB), BS>>>
        (
            nB, dm.fvBoundaryFaces.data(), meshVec(dm.Sf), dm.magSf.data(), lowMach, in, o
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
}


void inviscidResiduals
(
    const DeviceMesh& dm,
    const HisaState& s,
    const HisaFlux& flux,
    HisaResiduals& out
)
{
    // fvc::div 的結果場：extrapolatedCalculated（constraint patch 覆寫）
    VolField<double> dPhi = makeVolField<double>(dm, "div(phi)", Bc::extrapolatedCalculated);
    VolField<vec3> dUp = makeVolField<vec3>(dm, "div(phiUp)", Bc::extrapolatedCalculated);
    VolField<double> dEp = makeVolField<double>(dm, "div(phiEp)", Bc::extrapolatedCalculated);
    surfaceIntegrate(dm, flux.phi, dPhi.in);
    surfaceIntegrate(dm, flux.phiUp, dUp.in);
    surfaceIntegrate(dm, flux.phiEp, dEp.in);
    evaluateBCs(dm, dPhi, nullptr);
    evaluateBCs(dm, dUp, nullptr);
    evaluateBCs(dm, dEp, nullptr);

    // fvOptions(rho, U) & U：內部值 -0（向量），邊界同上的規則
    VolField<vec3> MU = makeVolField<vec3>(dm, "M&U", Bc::extrapolatedCalculated);
    const std::int32_t nC = dm.nCells;
    const std::int32_t nB = dm.nFvBoundaryFaces();
    if (nC)
    {
        DView<vec3> v = MU.in.view();
        for (int k = 0; k < 3; ++k)
        {
            kFillNegZero<<<nBlocks(nC), BS>>>(nC, v.p[k]);
        }
        GNC_CUDA_LAUNCH_CHECK();
    }
    evaluateBCs(dm, MU, nullptr);

    out.rhoR = makeVolField<double>(dm, "rhoR", Bc::calculated);
    out.rhoUR = makeVolField<vec3>(dm, "rhoUR", Bc::calculated);
    out.rhoER = makeVolField<double>(dm, "rhoER", Bc::calculated);

    if (nC)
    {
        kResidualCells<<<nBlocks(nC), BS>>>
        (
            nC, dPhi.in.cview().p[0], dUp.in.cview(), dEp.in.cview().p[0],
            s.U.in.cview(), s.thermo.he.in.cview().p[0],
            out.rhoR.in.view().p[0], out.rhoUR.in.view(), out.rhoER.in.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    if (nB)
    {
        kResidualBoundary<<<nBlocks(nB), BS>>>
        (
            nB, dPhi.bnd.cview().p[0], dUp.bnd.cview(), dEp.bnd.cview().p[0],
            MU.bnd.cview(), s.U.bnd.cview(), s.thermo.he.bnd.cview().p[0],
            out.rhoR.bnd.view().p[0], out.rhoUR.bnd.view(), out.rhoER.bnd.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    // rhoR = (...) 等的賦值：對稱面上依新的內部值 evaluate
    finishAssignment(dm, out.rhoR);
    finishAssignment(dm, out.rhoUR);
    finishAssignment(dm, out.rhoER);
    GNC_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace gnc
