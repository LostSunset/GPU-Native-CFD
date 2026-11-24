// GPU-Native-CFD，GPL-3.0-or-later。見 hisaJacobian.cuh。

#include "solvers/gncHisa/hisaJacobian.cuh"

#include <stdexcept>
#include <string>
#include <vector>

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

// ---------------------------------------------------------------------------
// OpenFOAM 的運算（逐項照寫）

// s*tensor::I：非對角為 s*0
GNC_HD tens sI(const double s)
{
    return {s*1.0, s*0.0, s*0.0, s*0.0, s*1.0, s*0.0, s*0.0, s*0.0, s*1.0};
}

// Tensor & Tensor（Tensor::inner）
GNC_HD tens inner(const tens& a, const tens& b)
{
    return
    {
        a.xx*b.xx + a.xy*b.yx + a.xz*b.zx,
        a.xx*b.xy + a.xy*b.yy + a.xz*b.zy,
        a.xx*b.xz + a.xy*b.yz + a.xz*b.zz,
        a.yx*b.xx + a.yy*b.yx + a.yz*b.zx,
        a.yx*b.xy + a.yy*b.yy + a.yz*b.zy,
        a.yx*b.xz + a.yy*b.yz + a.yz*b.zz,
        a.zx*b.xx + a.zy*b.yx + a.zz*b.zx,
        a.zx*b.xy + a.zy*b.yy + a.zz*b.zy,
        a.zx*b.xz + a.zy*b.yz + a.zz*b.zz
    };
}

// Tensor & SphericalTensor
GNC_HD tens dotSph(const tens& t, const double ii)
{
    return
    {
        t.xx*ii, t.xy*ii, t.xz*ii,
        t.yx*ii, t.yy*ii, t.yz*ii,
        t.zx*ii, t.zy*ii, t.zz*ii
    };
}

// Vector & SphericalTensor
GNC_HD vec3 dotSph(const vec3& v, const double ii)
{
    return {v.x*ii, v.y*ii, v.z*ii};
}

GNC_HD vec3 cmptMultiply(const vec3& a, const vec3& b)
{
    return {a.x*b.x, a.y*b.y, a.z*b.z};
}

// tensor::I 的對角換成 d（dUBdUI）
GNC_HD tens diagI(const vec3& d)
{
    return {d.x, 0.0, 0.0, 0.0, d.y, 0.0, 0.0, 0.0, d.z};
}

GNC_HD vec3 neg(const vec3& v)
{
    return {-v.x, -v.y, -v.z};
}

GNC_HD tens neg(const tens& t)
{
    return {-t.xx, -t.xy, -t.xz, -t.yx, -t.yy, -t.yz, -t.zx, -t.zy, -t.zz};
}

GNC_HD vec3 zeroV()
{
    return {0.0, 0.0, 0.0};
}

GNC_HD tens zeroT()
{
    return {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}


// ---------------------------------------------------------------------------
// 每格的推導量（laxFriedrichs::addFluxTerms）

struct CellV
{
    vec3 U;
    double th;
    tens mUU;
    double a;
    vec3 mAU;
    vec3 E;
    double gEmT;
    tens mAUU;
    vec3 gU;
};

__device__ CellV cellV(const double rho, const vec3 rhoU, const double rhoE, const double g)
{
    CellV v;
    v.U = rhoU/rho;
    v.a = g - 1.0;
    v.th = (0.5*v.a)*dot(v.U, v.U);
    const tens UU = outer(v.U, v.U);
    v.mUU = neg(UU);
    const double gE = (g*rhoE)/rho;
    v.mAU = (-v.a)*v.U;
    v.E = ((-gE)*v.U) + ((2.0*v.th)*v.U);
    v.gEmT = gE - v.th;
    v.mAUU = (-v.a)*UU;
    v.gU = g*v.U;
    return v;
}

// 黏性的推導量（laplacian::addFluxTerms）：A = -rhoU/sqr(rho)，R = 1/rho，
// Ev = -rhoE/sqr(rho) + magSqr(rhoU/rho)/rho
struct CellVisc
{
    vec3 A;
    double R;
    double Ev;
};

__device__ CellVisc cellVisc(const double rho, const vec3 rhoU, const double rhoE)
{
    CellVisc v;
    const double r2 = rho*rho;
    v.A = neg(rhoU)/r2;
    v.R = 1.0/rho;
    v.Ev = ((-rhoE)/r2) + (magSqr(rhoU/rho)/rho);
    return v;
}


// 面一側的 8 個子區塊係數（無黏）
struct FaceCoef
{
    double SS00;
    vec3 SV00;
    vec3 VS00;
    tens VV00;
    vec3 VS01;
    double SS10;
    vec3 SV10;
    double SS11;
};

__device__ FaceCoef faceSide(const CellV& v, const vec3 s, const double s2)
{
    FaceCoef o;
    o.SS00 = -s2;
    o.SV00 = s;
    o.VS00 = dot(v.mUU, s) + v.th*s;
    o.VV00 = ((sI(dot(v.U, s)) + outer(v.U, s)) + outer(s, v.mAU)) - sI(s2);
    o.VS01 = v.a*s;
    o.SS10 = dot(v.E, s);
    o.SV10 = v.gEmT*s + dot(v.mAUU, s);
    o.SS11 = dot(v.gU, s) - s2;
    return o;
}

// 黏性：-= laplacian 的係數（同一格的值乘 sm、sa）
__device__ void viscSide(FaceCoef& o, const CellVisc& v, const double sm, const double sa)
{
    o.VS00 = o.VS00 - v.A*sm;
    o.VV00 = o.VV00 - sI(v.R*sm);
    o.SS10 = o.SS10 - v.Ev*sa;
    o.SV10 = o.SV10 - v.A*sa;
    o.SS11 = o.SS11 - v.R*sa;
}


struct CellIn
{
    const double* rho;
    CView<vec3> rhoU;
    const double* rhoE;
    const double* p;
    const double* T;
    const double* psi;
    CView<vec3> U;          // 註冊表的速度
    const double* muEff;    // 黏性時
    const double* alphaEff;
    ThermoParams P;
};

struct FaceGeo
{
    const std::int32_t* own;
    const std::int32_t* nei;
    const double* w;
    CView<vec3> Sf;
    const double* magSf;
    const double* dc;
};

// 內部面的 stab 係數 0.5*lambdaConv*|Sf|*dc
__device__ double stabFace(const std::int32_t f, const CellIn& in, const FaceGeo& g)
{
    const std::int32_t P = g.own[f], N = g.nei[f];
    const double w = g.w[f];
    const double cP = ::sqrt(thermo::gamma(in.P, in.p[P], in.T[P])/in.psi[P]);
    const double cN = ::sqrt(thermo::gamma(in.P, in.p[N], in.T[N])/in.psi[N]);
    const double cf = w*(cP - cN) + cN;
    const vec3 UP = in.U[P], UN = in.U[N];
    const vec3 Uf = w*(UP - UN) + UN;
    const double mS = g.magSf[f];
    const vec3 nh = g.Sf[f]/mS;
    const double lam = (cf + ::fabs(dot(Uf, nh)))/g.dc[f];
    return ((0.5*lam)*mS)*g.dc[f];
}

__device__ void viscFace
(
    const std::int32_t f, const CellIn& in, const FaceGeo& g, double& sm, double& sa
)
{
    const std::int32_t P = g.own[f], N = g.nei[f];
    const double w = g.w[f];
    const double muf = w*(in.muEff[P] - in.muEff[N]) + in.muEff[N];
    const double alf = w*(in.alphaEff[P] - in.alphaEff[N]) + in.alphaEff[N];
    sm = (muf*g.magSf[f])*g.dc[f];
    sa = (alf*g.magSf[f])*g.dc[f];
}

__device__ CellV cellAt(const std::int32_t c, const CellIn& in)
{
    return cellV(in.rho[c], in.rhoU[c], in.rhoE[c], thermo::gamma(in.P, in.p[c], in.T[c]));
}

__device__ CellVisc viscAt(const std::int32_t c, const CellIn& in)
{
    return cellVisc(in.rho[c], in.rhoU[c], in.rhoE[c]);
}


struct OffDiag
{
    double* SS00u; double* SS00l;
    DView<vec3> SV00u, SV00l, VS00u, VS00l, VS01u, VS01l, SV10u, SV10l;
    DView<tens> VV00u, VV00l;
    double* SS10u; double* SS10l;
    double* SS11u; double* SS11l;
};

// 內部面：upper（neighbour、usf）與 lower（owner、lsf）
__global__ void kFaces
(
    const std::int32_t nIF,
    const CellIn in,
    const FaceGeo g,
    const bool viscous,
    const OffDiag o
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIF)
    {
        return;
    }
    const std::int32_t P = g.own[f], N = g.nei[f];
    const vec3 S = g.Sf[f];
    const vec3 lsf = (-g.w[f])*S;
    const vec3 usf = S + lsf;
    const double s2 = stabFace(f, in, g);
    double sm = 0.0, sa = 0.0;
    if (viscous)
    {
        viscFace(f, in, g, sm, sa);
    }
    FaceCoef up = faceSide(cellAt(N, in), usf, s2);
    FaceCoef lo = faceSide(cellAt(P, in), lsf, s2);
    if (viscous)
    {
        viscSide(up, viscAt(N, in), sm, sa);
        viscSide(lo, viscAt(P, in), sm, sa);
    }
    o.SS00u[f] = up.SS00; o.SS00l[f] = lo.SS00;
    o.SV00u.set(f, up.SV00); o.SV00l.set(f, lo.SV00);
    o.VS00u.set(f, up.VS00); o.VS00l.set(f, lo.VS00);
    o.VV00u.set(f, up.VV00); o.VV00l.set(f, lo.VV00);
    o.VS01u.set(f, up.VS01); o.VS01l.set(f, lo.VS01);
    o.SS10u[f] = up.SS10; o.SS10l[f] = lo.SS10;
    o.SV10u.set(f, up.SV10); o.SV10l.set(f, lo.SV10);
    o.SS11u[f] = up.SS11; o.SS11l[f] = lo.SS11;
}


struct Diag
{
    double* SS00; double* SS01; double* SS10; double* SS11;
    DView<vec3> SV00, SV10, VS00, VS01;
    DView<tens> VV00;
};

struct Gather
{
    const std::int32_t* off;
    const std::int32_t* faces;
    const std::int32_t* faceToFvB;      // 索引 f - nIF；empty 為 -1
};

// 每格的 diag：每個運算子各自由 0 開始依面編號扣，再照 addFluxTerms 組合
__global__ void kDiag
(
    const std::int32_t nC,
    const std::int32_t nIF,
    const CellIn in,
    const FaceGeo g,
    const Gather ga,
    const double* __restrict__ bStab,      // 每個 fv 邊界面的 stab 係數
    const bool viscous,
    const Diag d
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC)
    {
        return;
    }
    const CellV v = cellAt(c, in);
    vec3 g1 = zeroV(), dT = zeroV(), gth = zeroV(), ga_ = zeroV(), gg = zeroV(), dAUU = zeroV();
    tens gU = zeroT(), gT = zeroT();
    double dU = 0.0, dE = 0.0, dgU = 0.0, st = 0.0;
    // 黏性 laplacian 各自的 diag
    vec3 lAm = zeroV(), lAa = zeroV();
    double lRm = 0.0, lEa = 0.0, lRa = 0.0;
    CellVisc vv{};
    if (viscous)
    {
        vv = viscAt(c, in);
    }
    for (std::int32_t q = ga.off[c]; q < ga.off[c + 1]; ++q)
    {
        const std::int32_t f = ga.faces[q];
        if (f >= nIF)
        {
            continue;
        }
        const vec3 S = g.Sf[f];
        const vec3 lsf = (-g.w[f])*S;
        const vec3 usf = S + lsf;
        const vec3 s = (g.own[f] == c) ? lsf : usf;
        g1 = g1 - s;
        dT = dT - dot(v.mUU, s);
        gth = gth - v.th*s;
        dU = dU - dot(v.U, s);
        gU = gU - outer(v.U, s);
        gT = gT - outer(s, v.mAU);
        ga_ = ga_ - v.a*s;
        dE = dE - dot(v.E, s);
        gg = gg - v.gEmT*s;
        dAUU = dAUU - dot(v.mAUU, s);
        dgU = dgU - dot(v.gU, s);
        st = st - stabFace(f, in, g);
        if (viscous)
        {
            double sm, sa;
            viscFace(f, in, g, sm, sa);
            lAm = lAm - vv.A*sm;
            lRm = lRm - vv.R*sm;
            lEa = lEa - vv.Ev*sa;
            lAa = lAa - vv.A*sa;
            lRa = lRa - vv.R*sa;
        }
    }
    // stab 含實體邊界：內部面之後依 patch 次序
    for (std::int32_t q = ga.off[c]; q < ga.off[c + 1]; ++q)
    {
        const std::int32_t f = ga.faces[q];
        if (f < nIF)
        {
            continue;
        }
        const std::int32_t k = ga.faceToFvB[f - nIF];
        if (k >= 0)
        {
            st = st - bStab[k];
        }
    }

    d.SS00[c] = 0.0 - st;
    d.SS01[c] = 0.0;
    d.SV00.set(c, zeroV() + g1);
    vec3 VS00 = zeroV() + (dT + gth);
    tens VV00 = (zeroT() + ((sI(dU) + gU) + gT)) - sI(st);
    const vec3 VS01 = zeroV() + ga_;
    double SS10 = 0.0 + dE;
    vec3 SV10 = zeroV() + (gg + dAUU);
    double SS11 = (0.0 + dgU) - st;
    if (viscous)
    {
        VS00 = VS00 - lAm;
        VV00 = VV00 - sI(lRm);
        SS10 = SS10 - lEa;
        SV10 = SV10 - lAa;
        SS11 = SS11 - lRa;
    }
    d.VS00.set(c, VS00);
    d.VV00.set(c, VV00);
    d.VS01.set(c, VS01);
    d.SS10[c] = SS10;
    d.SV10.set(c, SV10);
    d.SS11[c] = SS11;
}


// 邊界面的 stab 係數：((0.5*lambdaConv_b)*|Sf|)*dc_b，c_b、U_b 為邊界值
__global__ void kBoundaryStab
(
    const std::int32_t nB,
    const std::int32_t* __restrict__ fvbFaces,
    const ThermoParams P,
    const double* __restrict__ pB,
    const double* __restrict__ TB,
    const double* __restrict__ psiB,
    const CView<vec3> UB,
    const CView<vec3> Sf,
    const double* __restrict__ magSf,
    const double* __restrict__ dc,
    double* __restrict__ out
)
{
    const std::int32_t k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB)
    {
        return;
    }
    const std::int32_t f = fvbFaces[k];
    const double cb = ::sqrt(thermo::gamma(P, pB[k], TB[k])/psiB[k]);
    const double mS = magSf[f];
    const vec3 nh = Sf[f]/mS;
    const double lam = (cb + ::fabs(dot(UB[k], nh)))/dc[f];
    out[k] = ((0.5*lam)*mS)*dc[f];
}


// 邊界條件的係數（非 BlockCoupledBoundary）：每個邊界面
//   vic = valueInternalCoeffs(w)，gic = gradientInternalCoeffs()
enum BcKind : int
{
    BK_ONE = 0,         // zeroGradient：vic 1、gic 0
    BK_FIXED = 1,       // fixedValue：vic 0、gic -1*dc
    BK_BCFIXED = 2,     // boundaryCorrectedFixedValue：vic 0、gic -1*nonOrthDeltaCoeffs
    BK_MIXED = 3,       // mixed：vic 1*(1 - vf)、gic ((-1)*vf)*dc
    BK_SYMM = 4,        // symmetry、slip：純量同 BK_ONE；向量 vic 1 - |nf|、gic (-dc)*|nf|
    BK_SYMMPLANE = 5    // symmetryPlane：同上，n 為整個 patch 的法向量
};

struct PatchCoef
{
    int kind;
    vec3 planeN;
};

__global__ void kScalarCoeffs
(
    const std::int32_t start, const std::int32_t n,
    const std::int32_t* __restrict__ fvbFaces,
    const PatchCoef pc,
    const double* __restrict__ dc,
    const double* __restrict__ ndc,
    const double* __restrict__ vf,
    double* __restrict__ vic,
    double* __restrict__ gic
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const std::int32_t k = start + i;
    const std::int32_t f = fvbFaces[k];
    switch (pc.kind)
    {
        case BK_FIXED: vic[k] = 0.0; gic[k] = -1.0*dc[f]; break;
        case BK_BCFIXED: vic[k] = 0.0; gic[k] = -1.0*ndc[f]; break;
        case BK_MIXED: vic[k] = 1.0*(1.0 - vf[i]); gic[k] = ((-1.0)*vf[i])*dc[f]; break;
        default: vic[k] = 1.0; gic[k] = 0.0; break;
    }
}

__global__ void kVectorCoeffs
(
    const std::int32_t start, const std::int32_t n,
    const std::int32_t* __restrict__ fvbFaces,
    const PatchCoef pc,
    const CView<vec3> Sf,
    const double* __restrict__ magSf,
    const double* __restrict__ dc,
    const double* __restrict__ ndc,
    const double* __restrict__ vf,
    const DView<vec3> vic,
    const DView<vec3> gic
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const std::int32_t k = start + i;
    const std::int32_t f = fvbFaces[k];
    switch (pc.kind)
    {
        case BK_FIXED:
        {
            const double m = -1.0*dc[f];
            vic.set(k, zeroV());
            gic.set(k, vec3{m, m, m});
            break;
        }
        case BK_BCFIXED:
        {
            const double m = -1.0*ndc[f];
            vic.set(k, zeroV());
            gic.set(k, vec3{m, m, m});
            break;
        }
        case BK_MIXED:
        {
            const double a = 1.0*(1.0 - vf[i]);
            const double b = ((-1.0)*vf[i])*dc[f];
            vic.set(k, vec3{a, a, a});
            gic.set(k, vec3{b, b, b});
            break;
        }
        case BK_SYMM:
        case BK_SYMMPLANE:
        {
            const vec3 n = (pc.kind == BK_SYMM) ? Sf[f]/magSf[f] : pc.planeN;
            const vec3 dg{::fabs(n.x), ::fabs(n.y), ::fabs(n.z)};
            vic.set(k, vec3{1.0 - dg.x, 1.0 - dg.y, 1.0 - dg.z});
            gic.set(k, (-dc[f])*dg);
            break;
        }
        default:
            vic.set(k, vec3{1.0, 1.0, 1.0});
            gic.set(k, zeroV());
            break;
    }
}


// BlockCoupledBoundary 的係數：p（characteristicWallPressure、
// characteristicFarfieldPressure）
__global__ void kBlockCoeffsP
(
    const std::int32_t start, const std::int32_t n,
    const double* __restrict__ pCoeff,
    const CView<vec3> UCoeff,
    const double* __restrict__ TCoeff,
    double* __restrict__ vicP,
    const DView<vec3> PvU,
    double* __restrict__ PvT,
    std::int8_t* __restrict__ blk
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const std::int32_t k = start + i;
        vicP[k] = pCoeff[i];
        PvU.set(k, UCoeff[i]);
        PvT[k] = TCoeff[i];
        blk[k] = 1;
    }
}

// characteristicFarfieldVelocity：gradientInternalCoeffs 為 (-係數)*dc，對 T 為
// vector::zero
__global__ void kBlockCoeffsU
(
    const std::int32_t start, const std::int32_t n,
    const std::int32_t* __restrict__ fvbFaces,
    const double* __restrict__ dc,
    const CView<vec3> pCoeff,
    const CView<tens> UCoeff,
    const CView<vec3> TCoeff,
    const DView<vec3> UvP, const DView<tens> UvU, const DView<vec3> UvT,
    const DView<vec3> UgP, const DView<tens> UgU, const DView<vec3> UgT,
    std::int8_t* __restrict__ blk
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const std::int32_t k = start + i;
        const double d = dc[fvbFaces[k]];
        UvP.set(k, pCoeff[i]);
        UvU.set(k, UCoeff[i]);
        UvT.set(k, TCoeff[i]);
        UgP.set(k, neg(pCoeff[i])*d);
        UgU.set(k, neg(UCoeff[i])*d);
        UgT.set(k, zeroV());
        blk[k] = 1;
    }
}

// characteristicFarfieldTemperature：gradientInternalCoeffs 為 (-係數)*dc
__global__ void kBlockCoeffsT
(
    const std::int32_t start, const std::int32_t n,
    const std::int32_t* __restrict__ fvbFaces,
    const double* __restrict__ dc,
    const double* __restrict__ pCoeff,
    const CView<vec3> UCoeff,
    const double* __restrict__ TCoeff,
    double* __restrict__ vicT, double* __restrict__ gicT,
    double* __restrict__ TvP, const DView<vec3> TvU,
    double* __restrict__ TgP, const DView<vec3> TgU,
    std::int8_t* __restrict__ blk
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const std::int32_t k = start + i;
        const double d = dc[fvbFaces[k]];
        vicT[k] = TCoeff[i];
        gicT[k] = (-TCoeff[i])*d;
        TvP[k] = pCoeff[i];
        TvU.set(k, UCoeff[i]);
        TgP[k] = (-pCoeff[i])*d;
        TgU.set(k, neg(UCoeff[i])*d);
        blk[k] = 1;
    }
}

// 黏性的 muEff = rho*nut + mu、alphaEff = gamma(p, T)*(alpha + alphat)（格值或
// 邊界值）；laminar（nut、alphat 為 nullptr）時 muEff = rho*0.0 + mu、
// alphaEff = gamma(p, T)*alpha
__global__ void kMuAlphaEff
(
    const std::int32_t n,
    const ThermoParams P,
    const double* __restrict__ rho,
    const double* __restrict__ nut,
    const double* __restrict__ mu,
    const double* __restrict__ p,
    const double* __restrict__ T,
    const double* __restrict__ alpha,
    const double* __restrict__ alphat,
    double* __restrict__ muEff,
    double* __restrict__ alphaEff
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        const double mut = rho[i]*(nut ? nut[i] : 0.0);
        muEff[i] = mut + mu[i];
        const double g = thermo::gamma(P, p[i], T[i]);
        alphaEff[i] = alphat ? g*(alpha[i] + alphat[i]) : g*alpha[i];
    }
}


struct BIn
{
    const std::int32_t* fvbFaces;
    const std::int32_t* owner;
    CView<vec3> Sf;
    const double* magSf;
    ThermoParams P;
    // 邊界值
    const double* rhoB;
    CView<vec3> rhoUB;
    const double* rhoEB;
    const double* pB;
    const double* TB;
    const double* muEffB;
    const double* alphaEffB;
    // 格值
    const double* rho;
    CView<vec3> rhoU;
    const double* rhoE;
    const double* p;
    const double* T;
    // 係數（非 BlockCoupled：vic、gic；BlockCoupled 時 vicP、vicT、gicT 兼作
    // 對本場的係數，其餘為對另外兩個場的係數）
    const std::int8_t* pBlk;
    const std::int8_t* UBlk;
    const std::int8_t* TBlk;
    const double* vicP;
    CView<vec3> PvU;
    const double* PvT;
    CView<vec3> vicU;
    CView<vec3> gicU;
    CView<vec3> UvP;
    CView<tens> UvU;
    CView<vec3> UvT;
    CView<vec3> UgP;
    CView<tens> UgU;
    CView<vec3> UgT;
    const double* vicT;
    const double* gicT;
    const double* TvP;
    CView<vec3> TvU;
    const double* TgP;
    CView<vec3> TgU;
};

struct BOut
{
    double* SS00; double* SS01; double* SS10; double* SS11;
    DView<vec3> SV00, SV10, VS00, VS01;
    DView<tens> VV00;
};

// 以 Int 係數求對守恆量的導數（laxFriedrichs／laplacian::boundaryJacobian 末段）
struct BlockVals
{
    double SS00; vec3 SV00; double SS01;
    vec3 VS00; tens VV00; vec3 VS01;
    double SS10; vec3 SV10; double SS11;
};

__device__ BlockVals chainRule
(
    const double Ip_c, const vec3 IU_c, const double IT_c,
    const vec3 Ip_m, const tens IU_m, const vec3 IT_m,
    const double Ip_e, const vec3 IU_e, const double IT_e,
    const double dPdRho, const vec3 dUdRho, const double dTdRho,
    const vec3 dPdRhoU, const double dUdRhoU, const vec3 dTdRhoU,
    const double dPdRhoE, const vec3 dUdRhoE, const double dTdRhoE
)
{
    BlockVals b;
    b.SS00 = (Ip_c*dPdRho + dot(IU_c, dUdRho)) + IT_c*dTdRho;
    b.SV00 = (Ip_c*dPdRhoU + dotSph(IU_c, dUdRhoU)) + IT_c*dTdRhoU;
    b.SS01 = (Ip_c*dPdRhoE + dot(IU_c, dUdRhoE)) + IT_c*dTdRhoE;
    b.VS00 = (Ip_m*dPdRho + dot(IU_m, dUdRho)) + IT_m*dTdRho;
    b.VV00 = (outer(Ip_m, dPdRhoU) + dotSph(IU_m, dUdRhoU)) + outer(IT_m, dTdRhoU);
    b.VS01 = (Ip_m*dPdRhoE + dot(IU_m, dUdRhoE)) + IT_m*dTdRhoE;
    b.SS10 = (Ip_e*dPdRho + dot(IU_e, dUdRho)) + IT_e*dTdRho;
    b.SV10 = (Ip_e*dPdRhoU + dotSph(IU_e, dUdRhoU)) + IT_e*dTdRhoU;
    b.SS11 = (Ip_e*dPdRhoE + dot(IU_e, dUdRhoE)) + IT_e*dTdRhoE;
    return b;
}

__global__ void kBoundaryTerms
(
    const std::int32_t nB,
    const BIn in,
    const bool viscous,
    const BOut o
)
{
    const std::int32_t k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nB)
    {
        return;
    }
    const std::int32_t f = in.fvbFaces[k];
    const std::int32_t c = in.owner[f];
    const ThermoParams& P = in.P;

    const double rhoB = in.rhoB[k];
    const vec3 SfB = in.Sf[f];
    const vec3 UB = in.rhoUB[k]/rhoB;
    const double UrelBdotSf = dot(UB, SfB);
    const double pB = in.pB[k];
    const double TB = in.TB[k];
    const double rhoEB = in.rhoEB[k];
    const double cvB = thermo::Cv(P, pB, TB);

    const double dContFluxdp = (rhoB/pB)*UrelBdotSf;
    const vec3 dContFluxdU = rhoB*SfB;
    const double dContFluxdT = ((-rhoB)/TB)*UrelBdotSf;
    const vec3 dMomFluxdp = (((rhoB/pB)*UB)*UrelBdotSf) + SfB;
    const tens dMomFluxdU = outer(rhoB*UB, SfB) + sI(rhoB*UrelBdotSf);
    const vec3 dMomFluxdT = ((((-rhoB)/TB)*UB)*UrelBdotSf);
    const double dEnergyFluxdp = ((rhoEB/pB)*UrelBdotSf) + dot(UB, SfB);
    const vec3 dEnergyFluxdU = (SfB*(rhoEB + pB)) + ((rhoB*UrelBdotSf)*UB);
    const double dEnergyFluxdT = UrelBdotSf*((rhoB*cvB) - (rhoEB/TB));

    // Int 係數：由 0 開始，依 p、U、T 的次序 +=（laxFriedrichs::boundaryJacobian）
    double Ip_c = 0.0, IT_c = 0.0, Ip_e = 0.0, IT_e = 0.0;
    vec3 IU_c = zeroV(), Ip_m = zeroV(), IT_m = zeroV(), IU_e = zeroV();
    tens IU_m = zeroT();
    if (in.pBlk[k])
    {
        const double vp = in.vicP[k];
        const vec3 vU = in.PvU[k];
        const double vT = in.PvT[k];
        Ip_c = Ip_c + dContFluxdp*vp;
        IU_c = IU_c + dContFluxdp*vU;
        IT_c = IT_c + dContFluxdp*vT;
        Ip_m = Ip_m + dMomFluxdp*vp;
        IU_m = IU_m + outer(dMomFluxdp, vU);
        IT_m = IT_m + dMomFluxdp*vT;
        Ip_e = Ip_e + dEnergyFluxdp*vp;
        IU_e = IU_e + dEnergyFluxdp*vU;
        IT_e = IT_e + dEnergyFluxdp*vT;
    }
    else
    {
        const double vicP = in.vicP[k];
        Ip_c = Ip_c + dContFluxdp*vicP;
        Ip_m = Ip_m + dMomFluxdp*vicP;
        Ip_e = Ip_e + dEnergyFluxdp*vicP;
    }
    if (in.UBlk[k])
    {
        const vec3 vp = in.UvP[k];
        const tens vU = in.UvU[k];
        const vec3 vT = in.UvT[k];
        Ip_c = Ip_c + dot(dContFluxdU, vp);
        IU_c = IU_c + dot(dContFluxdU, vU);
        IT_c = IT_c + dot(dContFluxdU, vT);
        Ip_m = Ip_m + dot(dMomFluxdU, vp);
        IU_m = IU_m + inner(dMomFluxdU, vU);
        IT_m = IT_m + dot(dMomFluxdU, vT);
        Ip_e = Ip_e + dot(dEnergyFluxdU, vp);
        IU_e = IU_e + dot(dEnergyFluxdU, vU);
        IT_e = IT_e + dot(dEnergyFluxdU, vT);
    }
    else
    {
        const vec3 vicU = in.vicU[k];
        IU_c = IU_c + cmptMultiply(dContFluxdU, vicU);
        IU_m = IU_m + inner(dMomFluxdU, diagI(vicU));
        IU_e = IU_e + cmptMultiply(dEnergyFluxdU, vicU);
    }
    if (in.TBlk[k])
    {
        const double vp = in.TvP[k];
        const vec3 vU = in.TvU[k];
        const double vT = in.vicT[k];
        Ip_c = Ip_c + dContFluxdT*vp;
        IU_c = IU_c + dContFluxdT*vU;
        IT_c = IT_c + dContFluxdT*vT;
        Ip_m = Ip_m + dMomFluxdT*vp;
        IU_m = IU_m + outer(dMomFluxdT, vU);
        IT_m = IT_m + dMomFluxdT*vT;
        Ip_e = Ip_e + dEnergyFluxdT*vp;
        IU_e = IU_e + dEnergyFluxdT*vU;
        IT_e = IT_e + dEnergyFluxdT*vT;
    }
    else
    {
        const double vicT = in.vicT[k];
        IT_c = IT_c + dContFluxdT*vicT;
        IT_m = IT_m + dMomFluxdT*vicT;
        IT_e = IT_e + dEnergyFluxdT*vicT;
    }

    // 相鄰格的導數
    const double rhoI = in.rho[c];
    const vec3 UI = in.rhoU[c]/rhoI;
    const double rhoEI = in.rhoE[c];
    const double gammaI = thermo::gamma(P, in.p[c], in.T[c]);
    const double cvI = thermo::Cv(P, in.p[c], in.T[c]);
    const double dPdRho = (0.5*(gammaI - 1.0))*magSqr(UI);
    const vec3 dUdRho = neg(UI)/rhoI;
    const double dTdRho = ((-1.0)/(cvI*rhoI))*((rhoEI/rhoI) - magSqr(UI));
    const vec3 dPdRhoU = (-(gammaI - 1.0))*UI;
    const double dUdRhoU = (1.0/rhoI)*1.0;
    const vec3 dTdRhoU = neg(UI)/(cvI*rhoI);
    const double dPdRhoE = gammaI - 1.0;
    const vec3 dUdRhoE = zeroV();
    const double dTdRhoE = 1.0/(cvI*rhoI);

    BlockVals b = chainRule
    (
        Ip_c, IU_c, IT_c, Ip_m, IU_m, IT_m, Ip_e, IU_e, IT_e,
        dPdRho, dUdRho, dTdRho, dPdRhoU, dUdRhoU, dTdRhoU, dPdRhoE, dUdRhoE, dTdRhoE
    );

    if (viscous)
    {
        // laplacian::boundaryJacobian
        const double magSfB = in.magSf[f];
        const double dMomFluxdUDiag = (-in.muEffB[k])*magSfB;
        tens dMomFluxdUv = zeroT();
        dMomFluxdUv.xx = dMomFluxdUDiag;
        dMomFluxdUv.yy = dMomFluxdUDiag;
        dMomFluxdUv.zz = dMomFluxdUDiag;
        const double dEnergyFluxdTv = ((-in.alphaEffB[k])*cvB)*magSfB;

        const double vIp_c = 0.0, vIT_c = 0.0;
        const vec3 vIU_c = zeroV();
        double vIp_e = 0.0, vIT_e = 0.0;
        vec3 vIp_m = zeroV(), vIT_m = zeroV(), vIU_e = zeroV();
        tens vIU_m = zeroT();
        if (in.UBlk[k])
        {
            vIp_m = vIp_m + dot(dMomFluxdUv, in.UgP[k]);
            vIU_m = vIU_m + inner(dMomFluxdUv, in.UgU[k]);
            vIT_m = vIT_m + dot(dMomFluxdUv, in.UgT[k]);
        }
        else
        {
            vIU_m = vIU_m + inner(dMomFluxdUv, diagI(in.gicU[k]));
        }
        if (in.TBlk[k])
        {
            vIp_e = vIp_e + dEnergyFluxdTv*in.TgP[k];
            vIU_e = vIU_e + dEnergyFluxdTv*in.TgU[k];
            vIT_e = vIT_e + dEnergyFluxdTv*in.gicT[k];
        }
        else
        {
            vIT_e = vIT_e + dEnergyFluxdTv*in.gicT[k];
        }

        const BlockVals v = chainRule
        (
            vIp_c, vIU_c, vIT_c, vIp_m, vIU_m, vIT_m, vIp_e, vIU_e, vIT_e,
            dPdRho, dUdRho, dTdRho, dPdRhoU, dUdRhoU, dTdRhoU, dPdRhoE, dUdRhoE, dTdRhoE
        );
        // dCont* 為 0 的場；dMom*、dEnergy* 用上面的式子
        b.SS00 = b.SS00 + 0.0;
        b.SV00 = b.SV00 + zeroV();
        b.SS01 = b.SS01 + 0.0;
        b.VS00 = b.VS00 + v.VS00;
        b.VV00 = b.VV00 + v.VV00;
        b.VS01 = b.VS01 + v.VS01;
        b.SS10 = b.SS10 + v.SS10;
        b.SV10 = b.SV10 + v.SV10;
        b.SS11 = b.SS11 + v.SS11;
    }

    o.SS00[k] = b.SS00;
    o.SV00.set(k, b.SV00);
    o.SS01[k] = b.SS01;
    o.VS00.set(k, b.VS00);
    o.VV00.set(k, b.VV00);
    o.VS01.set(k, b.VS01);
    o.SS10[k] = b.SS10;
    o.SV10.set(k, b.SV10);
    o.SS11[k] = b.SS11;
}


struct BVals
{
    const double* SS00; const double* SS01; const double* SS10; const double* SS11;
    CView<vec3> SV00, SV10, VS00, VS01;
    CView<tens> VV00;
};

// diag += 邊界項（依 patch、面次序），再加時間項，最後 LU-SGS 的 rDiagCoeff
__global__ void kDiagBoundaryTemporal
(
    const std::int32_t nC,
    const std::int32_t nIF,
    const Gather ga,
    const BVals b,
    const double* __restrict__ ddtCoeff,
    const double* __restrict__ V,
    const Diag d,
    double* __restrict__ rDiag
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nC)
    {
        return;
    }
    double SS00 = d.SS00[c], SS01 = d.SS01[c], SS10 = d.SS10[c], SS11 = d.SS11[c];
    vec3 SV00 = d.SV00[c], SV10 = d.SV10[c], VS00 = d.VS00[c], VS01 = d.VS01[c];
    tens VV00 = d.VV00[c];
    for (std::int32_t q = ga.off[c]; q < ga.off[c + 1]; ++q)
    {
        const std::int32_t f = ga.faces[q];
        if (f < nIF)
        {
            continue;
        }
        const std::int32_t k = ga.faceToFvB[f - nIF];
        if (k < 0)
        {
            continue;
        }
        SS00 = SS00 + b.SS00[k];
        SV00 = SV00 + b.SV00[k];
        SS01 = SS01 + b.SS01[k];
        VS00 = VS00 + b.VS00[k];
        VV00 = VV00 + b.VV00[k];
        VS01 = VS01 + b.VS01[k];
        SS10 = SS10 + b.SS10[k];
        SV10 = SV10 + b.SV10[k];
        SS11 = SS11 + b.SS11[k];
    }
    // addTemporalTerms：diagCoeff = ddtCoeff*V
    const double diagCoeff = ddtCoeff[c]*V[c];
    SS00 = SS00 + diagCoeff;
    VV00 = VV00 + sI(diagCoeff);
    SS11 = SS11 + diagCoeff;

    d.SS00[c] = SS00; d.SS01[c] = SS01; d.SS10[c] = SS10; d.SS11[c] = SS11;
    d.SV00.set(c, SV00); d.SV10.set(c, SV10); d.VS00.set(c, VS00); d.VS01.set(c, VS01);
    d.VV00.set(c, VV00);

    // lusgs 建構子：rDiagCoeff = 1/max(1/rDiagCoeff, mag(diag))，依 SS00、SS11、
    // VV00 的 xx、yy、zz，起始 GREAT
    double r = 1.0e+15;
    const double dd[5] = {SS00, SS11, VV00.xx, VV00.yy, VV00.zz};
    for (int i = 0; i < 5; ++i)
    {
        const double a = 1.0/r;
        const double m = ::fabs(dd[i]);
        r = 1.0/((m < a) ? a : m);
    }
    rDiag[c] = r;
}


template<class T>
void allocBlock(FvjBlock<T>& b, const std::int32_t nC, const std::int32_t nIF, const bool offDiag)
{
    b.diag.allocate(static_cast<std::size_t>(nC));
    b.hasOffDiag = offDiag;
    if (offDiag)
    {
        b.upper.allocate(static_cast<std::size_t>(nIF));
        b.lower.allocate(static_cast<std::size_t>(nIF));
    }
}

} // namespace


void assembleJacobian
(
    const DeviceMesh& dm,
    const FvSchemes& schemes,
    const HisaState& s,
    const PseudoTimeState& pt,
    HisaJacobian& J
)
{
    (void)schemes;
    const std::int32_t nC = dm.nCells;
    const std::int32_t nIF = dm.nInternalFaces;
    const std::int32_t nB = dm.nFvBoundaryFaces();
    const PsiThermoFields& th = s.thermo;
    const bool viscous = !s.inviscid;

    allocBlock(J.SS00, nC, nIF, true);
    allocBlock(J.SS01, nC, nIF, false);
    allocBlock(J.SS10, nC, nIF, true);
    allocBlock(J.SS11, nC, nIF, true);
    allocBlock(J.SV00, nC, nIF, true);
    allocBlock(J.SV10, nC, nIF, true);
    allocBlock(J.VS00, nC, nIF, true);
    allocBlock(J.VS01, nC, nIF, true);
    allocBlock(J.VV00, nC, nIF, true);
    J.rDiagCoeff.allocate(static_cast<std::size_t>(nC));

    // 黏性的 muEff、alphaEff（laplacian::addViscousJacobian：interpolate(muEff)、
    // interpolate(alphaEff)；與殘差相同的式子，T 的邊界值為更新後的值）
    DField<double> muEffC, alphaEffC, muEffB, alphaEffB;
    if (viscous)
    {
        muEffC.allocate(static_cast<std::size_t>(nC));
        alphaEffC.allocate(static_cast<std::size_t>(nC));
        muEffB.allocate(static_cast<std::size_t>(nB));
        alphaEffB.allocate(static_cast<std::size_t>(nB));
        if (nC)
        {
            kMuAlphaEff<<<nBlocks(nC), BS>>>
            (
                nC, th.params, s.rho.in.cview().p[0],
                s.nut ? s.nut->in.cview().p[0] : nullptr, th.mu.in.cview().p[0],
                th.p.in.cview().p[0], th.T.in.cview().p[0], th.alpha.in.cview().p[0],
                s.alphat ? s.alphat->in.cview().p[0] : nullptr,
                muEffC.view().p[0], alphaEffC.view().p[0]
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
        if (nB)
        {
            kMuAlphaEff<<<nBlocks(nB), BS>>>
            (
                nB, th.params, s.rho.bnd.cview().p[0],
                s.nut ? s.nut->bnd.cview().p[0] : nullptr, th.mu.bnd.cview().p[0],
                th.p.bnd.cview().p[0], th.T.bnd.cview().p[0], th.alpha.bnd.cview().p[0],
                s.alphat ? s.alphat->bnd.cview().p[0] : nullptr,
                muEffB.view().p[0], alphaEffB.view().p[0]
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
    }

    CellIn in{};
    in.rho = s.rho.in.cview().p[0];
    in.rhoU = s.rhoU.in.cview();
    in.rhoE = s.rhoE.in.cview().p[0];
    in.p = th.p.in.cview().p[0];
    in.T = th.T.in.cview().p[0];
    in.psi = th.psi.in.cview().p[0];
    in.U = s.U.in.cview();
    in.muEff = viscous ? muEffC.cview().p[0] : nullptr;
    in.alphaEff = viscous ? alphaEffC.cview().p[0] : nullptr;
    in.P = th.params;

    FaceGeo g{dm.owner.data(), dm.neighbour.data(), dm.weights.data(), meshVec(dm.Sf),
              dm.magSf.data(), dm.deltaCoeffs.data()};

    if (nIF)
    {
        OffDiag o{};
        o.SS00u = J.SS00.upper.view().p[0]; o.SS00l = J.SS00.lower.view().p[0];
        o.SV00u = J.SV00.upper.view(); o.SV00l = J.SV00.lower.view();
        o.VS00u = J.VS00.upper.view(); o.VS00l = J.VS00.lower.view();
        o.VS01u = J.VS01.upper.view(); o.VS01l = J.VS01.lower.view();
        o.SV10u = J.SV10.upper.view(); o.SV10l = J.SV10.lower.view();
        o.VV00u = J.VV00.upper.view(); o.VV00l = J.VV00.lower.view();
        o.SS10u = J.SS10.upper.view().p[0]; o.SS10l = J.SS10.lower.view().p[0];
        o.SS11u = J.SS11.upper.view().p[0]; o.SS11l = J.SS11.lower.view().p[0];
        kFaces<<<nBlocks(nIF), BS>>>(nIF, in, g, viscous, o);
        GNC_CUDA_LAUNCH_CHECK();
    }

    // 邊界面的 stab
    DField<double> bStab(static_cast<std::size_t>(nB));
    if (nB)
    {
        kBoundaryStab<<<nBlocks(nB), BS>>>
        (
            nB, dm.fvBoundaryFaces.data(), th.params, th.p.bnd.cview().p[0],
            th.T.bnd.cview().p[0], th.psi.bnd.cview().p[0], s.U.bnd.cview(), meshVec(dm.Sf),
            dm.magSf.data(), dm.deltaCoeffs.data(), bStab.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }

    Diag d{};
    d.SS00 = J.SS00.diag.view().p[0]; d.SS01 = J.SS01.diag.view().p[0];
    d.SS10 = J.SS10.diag.view().p[0]; d.SS11 = J.SS11.diag.view().p[0];
    d.SV00 = J.SV00.diag.view(); d.SV10 = J.SV10.diag.view();
    d.VS00 = J.VS00.diag.view(); d.VS01 = J.VS01.diag.view();
    d.VV00 = J.VV00.diag.view();
    const Gather ga{dm.cellFaceOffsets.data(), dm.cellFaces.data(), dm.faceToFvB.data()};
    if (nC)
    {
        kDiag<<<nBlocks(nC), BS>>>(nC, nIF, in, g, ga, bStab.cview().p[0], viscous, d);
        GNC_CUDA_LAUNCH_CHECK();
    }

    // 邊界條件的係數
    const std::size_t nBs = static_cast<std::size_t>(nB);
    DField<double> vicP(nBs), vicT(nBs), gicP(nBs), gicT(nBs), PvT(nBs), TvP(nBs), TgP(nBs);
    DField<vec3> vicU(nBs), gicU(nBs), PvU(nBs), UvP(nBs), UvT(nBs), UgP(nBs), UgT(nBs);
    DField<vec3> TvU(nBs), TgU(nBs);
    DField<tens> UvU(nBs), UgU(nBs);
    vicP.zero(); vicT.zero(); gicP.zero(); gicT.zero(); PvT.zero(); TvP.zero(); TgP.zero();
    vicU.zero(); gicU.zero(); PvU.zero(); UvP.zero(); UvT.zero(); UgP.zero(); UgT.zero();
    TvU.zero(); TgU.zero(); UvU.zero(); UgU.zero();
    DeviceArray<std::int8_t> pBlk(nBs), UBlk(nBs), TBlk(nBs);
    pBlk.zero(); UBlk.zero(); TBlk.zero();
    const auto isBlock = [](const Bc t)
    {
        return t == Bc::characteristicWallPressure || t == Bc::characteristicFarfieldPressure
            || t == Bc::characteristicFarfieldVelocity || t == Bc::characteristicFarfieldTemperature;
    };
    const auto kindOf = [&](const Bc t, const std::string& field, const std::string& patch) -> int
    {
        switch (t)
        {
            case Bc::zeroGradient: return BK_ONE;
            case Bc::fixedValue: return BK_FIXED;
            case Bc::boundaryCorrectedFixedValue: return BK_BCFIXED;
            case Bc::mixed:
            case Bc::inletOutlet: return BK_MIXED;
            case Bc::symmetry:
            case Bc::slip: return BK_SYMM;
            case Bc::symmetryPlane: return BK_SYMMPLANE;
            // characteristicWallTemperature 是 fixedValue（不是 BlockCoupledBoundary）
            case Bc::characteristicWallTemperature: return BK_FIXED;
            default:
                throw std::runtime_error
                (
                    "Jacobian: field " + field + " patch " + patch + ": boundary condition "
                  + bcName(t) + " is not supported yet"
                );
        }
    };
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const std::int32_t n = dm.fvPatchSize[pi];
        if (n == 0)
        {
            continue;
        }
        const std::int32_t start = dm.fvPatchStart[pi];
        const std::string& pname = dm.patches[pi].name;
        const PatchBC<double>& bp = th.p.bc[pi];
        const PatchBC<double>& bT = th.T.bc[pi];
        const PatchBC<vec3>& bU = s.U.bc[pi];
        if (isBlock(bp.type))
        {
            kBlockCoeffsP<<<nBlocks(n), BS>>>
            (
                start, n, bp.pCoeff.cview().p[0], bp.UCoeff.cview(), bp.TCoeff.cview().p[0],
                vicP.view().p[0], PvU.view(), PvT.view().p[0], pBlk.data()
            );
        }
        else
        {
            const PatchCoef pp{kindOf(bp.type, "p", pname), dm.patchNormal[pi]};
            kScalarCoeffs<<<nBlocks(n), BS>>>
            (
                start, n, dm.fvBoundaryFaces.data(), pp, dm.deltaCoeffs.data(),
                dm.nonOrthDeltaCoeffs.data(), pp.kind == BK_MIXED ? bp.valueFraction.data() : nullptr,
                vicP.view().p[0], gicP.view().p[0]
            );
        }
        if (isBlock(bT.type))
        {
            kBlockCoeffsT<<<nBlocks(n), BS>>>
            (
                start, n, dm.fvBoundaryFaces.data(), dm.deltaCoeffs.data(),
                bT.pCoeff.cview().p[0], bT.UCoeff.cview(), bT.TCoeff.cview().p[0],
                vicT.view().p[0], gicT.view().p[0], TvP.view().p[0], TvU.view(),
                TgP.view().p[0], TgU.view(), TBlk.data()
            );
        }
        else
        {
            const PatchCoef pT{kindOf(bT.type, "T", pname), dm.patchNormal[pi]};
            kScalarCoeffs<<<nBlocks(n), BS>>>
            (
                start, n, dm.fvBoundaryFaces.data(), pT, dm.deltaCoeffs.data(),
                dm.nonOrthDeltaCoeffs.data(), pT.kind == BK_MIXED ? bT.valueFraction.data() : nullptr,
                vicT.view().p[0], gicT.view().p[0]
            );
        }
        if (isBlock(bU.type))
        {
            kBlockCoeffsU<<<nBlocks(n), BS>>>
            (
                start, n, dm.fvBoundaryFaces.data(), dm.deltaCoeffs.data(),
                bU.pCoeff.cview(), bU.UCoeff.cview(), bU.TCoeff.cview(),
                UvP.view(), UvU.view(), UvT.view(), UgP.view(), UgU.view(), UgT.view(),
                UBlk.data()
            );
        }
        else
        {
            const PatchCoef pU{kindOf(bU.type, "U", pname), dm.patchNormal[pi]};
            kVectorCoeffs<<<nBlocks(n), BS>>>
            (
                start, n, dm.fvBoundaryFaces.data(), pU, meshVec(dm.Sf), dm.magSf.data(),
                dm.deltaCoeffs.data(), dm.nonOrthDeltaCoeffs.data(),
                pU.kind == BK_MIXED ? bU.valueFraction.data() : nullptr,
                vicU.view(), gicU.view()
            );
        }
        GNC_CUDA_LAUNCH_CHECK();
    }

    DField<double> bSS00(static_cast<std::size_t>(nB)), bSS01(static_cast<std::size_t>(nB)), bSS10(static_cast<std::size_t>(nB)), bSS11(static_cast<std::size_t>(nB));
    DField<vec3> bSV00(static_cast<std::size_t>(nB)), bSV10(static_cast<std::size_t>(nB)), bVS00(static_cast<std::size_t>(nB)), bVS01(static_cast<std::size_t>(nB));
    DField<tens> bVV00(static_cast<std::size_t>(nB));
    if (nB)
    {
        BIn bi{};
        bi.fvbFaces = dm.fvBoundaryFaces.data();
        bi.owner = dm.owner.data();
        bi.Sf = meshVec(dm.Sf);
        bi.magSf = dm.magSf.data();
        bi.P = th.params;
        bi.rhoB = s.rho.bnd.cview().p[0];
        bi.rhoUB = s.rhoU.bnd.cview();
        bi.rhoEB = s.rhoE.bnd.cview().p[0];
        bi.pB = th.p.bnd.cview().p[0];
        bi.TB = th.T.bnd.cview().p[0];
        bi.muEffB = viscous ? muEffB.cview().p[0] : nullptr;
        bi.alphaEffB = viscous ? alphaEffB.cview().p[0] : nullptr;
        bi.rho = in.rho;
        bi.rhoU = in.rhoU;
        bi.rhoE = in.rhoE;
        bi.p = in.p;
        bi.T = in.T;
        bi.pBlk = pBlk.data();
        bi.UBlk = UBlk.data();
        bi.TBlk = TBlk.data();
        bi.vicP = vicP.cview().p[0];
        bi.PvU = PvU.cview();
        bi.PvT = PvT.cview().p[0];
        bi.vicU = vicU.cview();
        bi.gicU = gicU.cview();
        bi.UvP = UvP.cview();
        bi.UvU = UvU.cview();
        bi.UvT = UvT.cview();
        bi.UgP = UgP.cview();
        bi.UgU = UgU.cview();
        bi.UgT = UgT.cview();
        bi.vicT = vicT.cview().p[0];
        bi.gicT = gicT.cview().p[0];
        bi.TvP = TvP.cview().p[0];
        bi.TvU = TvU.cview();
        bi.TgP = TgP.cview().p[0];
        bi.TgU = TgU.cview();
        BOut bo{bSS00.view().p[0], bSS01.view().p[0], bSS10.view().p[0], bSS11.view().p[0],
                bSV00.view(), bSV10.view(), bVS00.view(), bVS01.view(), bVV00.view()};
        kBoundaryTerms<<<nBlocks(nB), BS>>>(nB, bi, viscous, bo);
        GNC_CUDA_LAUNCH_CHECK();
    }

    if (nC)
    {
        const BVals bv{bSS00.cview().p[0], bSS01.cview().p[0], bSS10.cview().p[0], bSS11.cview().p[0],
                       bSV00.cview(), bSV10.cview(), bVS00.cview(), bVS01.cview(), bVV00.cview()};
        kDiagBoundaryTemporal<<<nBlocks(nC), BS>>>
        (
            nC, nIF, ga, bv, pt.ddtCoeff.cview().p[0], dm.V.data(), d,
            J.rDiagCoeff.view().p[0]
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
    GNC_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace gnc
