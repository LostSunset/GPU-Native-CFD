// OpenFOAM 向量運算的逐位元複本。GPU-Native-CFD，GPL-3.0-or-later。
//
// 每個函數的運算次序照 OpenFOAM v2606 原始碼寫出（Atlas upstream/openfoam）：
//   inner   Vector<Cmpt>::inner        (x*x' + y*y') + z*z'     VectorI.H:153
//   cross   Vector<Cmpt>::cross                                  VectorI.H:163
//   magSqr  magSqr(VectorSpace)        (x*x + y*y) + z*z        VectorSpaceI.H:400
//   mag     ::sqrt(magSqr)                                       VectorSpaceI.H:412
//   s*v     operator*(Cmpt, Vector)    (s*x, s*y, s*z)          VectorI.H:307
//   v*s     與 s*v 相同（回傳 s*v；IEEE 乘法可交換）               VectorI.H:314
//   v/s     divideOp3，逐分量 x/s（不是乘倒數）                    VectorSpaceI.H:717
//   max     (b < a) ? a : b                                      int.H:59（MAXMIN 巨集）
// 只用四則與 sqrt，IEEE 正確捨入；裝置碼必須以 --fmad=false 編譯，
// 否則 a*b + c 會被合併成 FMA，與 OpenFOAM（x86-64，無 -mfma）不同。

#ifndef GNC_CORE_FOAMOPS_CUH
#define GNC_CORE_FOAMOPS_CUH

#include <cmath>

#ifdef __CUDACC__
#define GNC_HD __host__ __device__ __forceinline__
#else
#define GNC_HD inline
#endif

namespace gnc
{

// OpenFOAM doubleScalar.H:64–65
constexpr double VSMALL = 1.0e-300;
constexpr double ROOTVSMALL = 1.0e-150;

struct vec3
{
    double x, y, z;
};

GNC_HD vec3 operator+(const vec3& a, const vec3& b)
{
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

GNC_HD vec3 operator-(const vec3& a, const vec3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

GNC_HD void operator+=(vec3& a, const vec3& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

GNC_HD vec3 operator*(const double s, const vec3& v)
{
    return {s*v.x, s*v.y, s*v.z};
}

GNC_HD vec3 operator*(const vec3& v, const double s)
{
    return s*v;
}

GNC_HD vec3 operator/(const vec3& v, const double s)
{
    return {v.x/s, v.y/s, v.z/s};
}

GNC_HD double dot(const vec3& a, const vec3& b)
{
    return (a.x*b.x + a.y*b.y + a.z*b.z);
}

GNC_HD vec3 cross(const vec3& a, const vec3& b)
{
    return
    {
        (a.y*b.z - a.z*b.y),
        (a.z*b.x - a.x*b.z),
        (a.x*b.y - a.y*b.x)
    };
}

GNC_HD double magSqr(const vec3& v)
{
    double ms = v.x*v.x;
    ms += v.y*v.y;
    ms += v.z*v.z;
    return ms;
}

GNC_HD double mag(const vec3& v)
{
    return ::sqrt(magSqr(v));
}

GNC_HD double foamMax(const double a, const double b)
{
    return (b < a) ? a : b;
}

// MAXMIN 的 min：(s1 < s2) ? s1 : s2（int.H:53）
GNC_HD double foamMin(const double a, const double b)
{
    return (a < b) ? a : b;
}

// scalarImpl.H:343–370：sqr(s) = s*s、pow3(s) = s*sqr(s)、pow6(s) = pow3(sqr(s))
GNC_HD double foamSqr(const double s)
{
    return s*s;
}

GNC_HD double foamPow3(const double s)
{
    return s*foamSqr(s);
}

GNC_HD double foamPow6(const double s)
{
    return foamPow3(foamSqr(s));
}

// OpenFOAM SMALL（doubleScalar.H）
constexpr double SMALL = 1.0e-15;


// ---------------------------------------------------------------------------
// 張量（Tensor<double>，列優先 xx xy xz yx yy yz zx zy zz）。出處：
//   outer   operator*(Vector, Vector)          TensorI.H:1285
//   t & v   operator&(Tensor, Vector)          TensorI.H:1207
//   v & t   operator&(Vector, Tensor)          TensorI.H:1225
//   T()、tr、dev2                              TensorI.H:426、728、832
//   det     Tensor::det                        TensorI.H:438
//   adjunct Tensor::adjunct                    TensorI.H:471
//   safeInv Tensor::safeInv（tensorField 的 inv 逐項呼叫它，tensorField.C:55）
//   transform(tensor, Tensor)                  transform.H:214
//   lerp    (1 - t)*a + t*b                    scalarImpl.H:331、TensorI.H:910
// ---------------------------------------------------------------------------

struct tens
{
    double xx, xy, xz, yx, yy, yz, zx, zy, zz;
};

GNC_HD tens operator+(const tens& a, const tens& b)
{
    return
    {
        a.xx + b.xx, a.xy + b.xy, a.xz + b.xz,
        a.yx + b.yx, a.yy + b.yy, a.yz + b.yz,
        a.zx + b.zx, a.zy + b.zy, a.zz + b.zz
    };
}

GNC_HD tens operator-(const tens& a, const tens& b)
{
    return
    {
        a.xx - b.xx, a.xy - b.xy, a.xz - b.xz,
        a.yx - b.yx, a.yy - b.yy, a.yz - b.yz,
        a.zx - b.zx, a.zy - b.zy, a.zz - b.zz
    };
}

GNC_HD void operator+=(tens& a, const tens& b)
{
    a = a + b;
}

GNC_HD void operator-=(vec3& a, const vec3& b)
{
    a.x -= b.x;
    a.y -= b.y;
    a.z -= b.z;
}

GNC_HD void operator-=(tens& a, const tens& b)
{
    a = a - b;
}

GNC_HD tens operator*(const double s, const tens& t)
{
    return
    {
        s*t.xx, s*t.xy, s*t.xz,
        s*t.yx, s*t.yy, s*t.yz,
        s*t.zx, s*t.zy, s*t.zz
    };
}

GNC_HD tens operator*(const tens& t, const double s)
{
    return s*t;
}

GNC_HD tens operator/(const tens& t, const double s)
{
    return
    {
        t.xx/s, t.xy/s, t.xz/s,
        t.yx/s, t.yy/s, t.yz/s,
        t.zx/s, t.zy/s, t.zz/s
    };
}

// 外積 v1*v2（Vector*Vector → Tensor）
GNC_HD tens outer(const vec3& a, const vec3& b)
{
    return
    {
        a.x*b.x, a.x*b.y, a.x*b.z,
        a.y*b.x, a.y*b.y, a.y*b.z,
        a.z*b.x, a.z*b.y, a.z*b.z
    };
}

// Vector*scalar 在 OpenFOAM 是 s*v（VectorI.H:314）；梯度的「外積」對純量即此
GNC_HD vec3 outer(const vec3& a, const double s)
{
    return s*a;
}

GNC_HD vec3 dot(const tens& t, const vec3& v)
{
    return
    {
        t.xx*v.x + t.xy*v.y + t.xz*v.z,
        t.yx*v.x + t.yy*v.y + t.yz*v.z,
        t.zx*v.x + t.zy*v.y + t.zz*v.z
    };
}

GNC_HD vec3 dot(const vec3& v, const tens& t)
{
    return
    {
        v.x*t.xx + v.y*t.yx + v.z*t.zx,
        v.x*t.xy + v.y*t.yy + v.z*t.zy,
        v.x*t.xz + v.y*t.yz + v.z*t.zz
    };
}

// Tensor::T()
GNC_HD tens transpose(const tens& t)
{
    return {t.xx, t.yx, t.zx, t.xy, t.yy, t.zy, t.xz, t.yz, t.zz};
}

GNC_HD double tr(const tens& t)
{
    return t.xx + t.yy + t.zz;
}

// dev2(t) = t - 2*sph(t)，sph(t) = (1.0/3.0)*tr(t)；Tensor - SphericalTensor
// 只改對角項
GNC_HD tens dev2(const tens& t)
{
    const double s = 2.0*((1.0/3.0)*tr(t));
    return {t.xx - s, t.xy, t.xz, t.yx, t.yy - s, t.yz, t.zx, t.zy, t.zz - s};
}

GNC_HD double det(const tens& t)
{
    return
    (
        t.xx*t.yy*t.zz + t.xy*t.yz*t.zx
      + t.xz*t.yx*t.zy - t.xx*t.yz*t.zy
      - t.xy*t.yx*t.zz - t.xz*t.yy*t.zx
    );
}

GNC_HD tens adjunct(const tens& t)
{
    return
    {
        t.yy*t.zz - t.zy*t.yz, t.xz*t.zy - t.xy*t.zz, t.xy*t.yz - t.xz*t.yy,
        t.zx*t.yz - t.yx*t.zz, t.xx*t.zz - t.xz*t.zx, t.yx*t.xz - t.xx*t.yz,
        t.yx*t.zy - t.yy*t.zx, t.xy*t.zx - t.xx*t.zy, t.xx*t.yy - t.yx*t.xy
    };
}

GNC_HD tens safeInv(const tens& t)
{
    const double magSqr_xx = t.xx*t.xx;
    const double magSqr_yy = t.yy*t.yy;
    const double magSqr_zz = t.zz*t.zz;

    const double threshold = SMALL*(magSqr_xx + magSqr_yy + magSqr_zz);

    const bool small_xx = (magSqr_xx < threshold);
    const bool small_yy = (magSqr_yy < threshold);
    const bool small_zz = (magSqr_zz < threshold);

    if (small_xx || small_yy || small_zz)
    {
        tens work = t;
        if (small_xx) { work.xx += 1.0; }
        if (small_yy) { work.yy += 1.0; }
        if (small_zz) { work.zz += 1.0; }

        const double detval = det(work);
        if (::fabs(detval) < ROOTVSMALL)
        {
            return tens{0, 0, 0, 0, 0, 0, 0, 0, 0};
        }

        work = adjunct(work)/detval;

        if (small_xx) { work.xx -= 1.0; }
        if (small_yy) { work.yy -= 1.0; }
        if (small_zz) { work.zz -= 1.0; }
        return work;
    }

    const double detval = det(t);
    if (::fabs(detval) < ROOTVSMALL)
    {
        return tens{0, 0, 0, 0, 0, 0, 0, 0, 0};
    }
    return adjunct(t)/detval;
}

// 對稱邊界的旋轉張量 I - 2.0*sqr(n)（SymmTensor，轉成 Tensor 使用）：
// sqr(v) = (xx, xy, xz, yy, yz, zz)；2.0*st 逐項；I - st 對角為 1 - st，
// 非對角為 -st（SphericalTensor 與 SymmTensor 的減法）
GNC_HD tens symmetryRot(const vec3& n)
{
    const double sxx = 2.0*(n.x*n.x);
    const double sxy = 2.0*(n.x*n.y);
    const double sxz = 2.0*(n.x*n.z);
    const double syy = 2.0*(n.y*n.y);
    const double syz = 2.0*(n.y*n.z);
    const double szz = 2.0*(n.z*n.z);
    return
    {
        1.0 - sxx, -sxy, -sxz,
        -sxy, 1.0 - syy, -syz,
        -sxz, -syz, 1.0 - szz
    };
}

// transform：純量不變（transform.H:154）；向量 tt & v（:196）；張量（:214）
GNC_HD double transform(const tens&, const double s)
{
    return s;
}

GNC_HD vec3 transform(const tens& tt, const vec3& v)
{
    return dot(tt, v);
}

GNC_HD tens transform(const tens& tt, const tens& t)
{
    const double ax = tt.xx*t.xx + tt.xy*t.yx + tt.xz*t.zx;
    const double ay = tt.xx*t.xy + tt.xy*t.yy + tt.xz*t.zy;
    const double az = tt.xx*t.xz + tt.xy*t.yz + tt.xz*t.zz;
    const double bx = tt.yx*t.xx + tt.yy*t.yx + tt.yz*t.zx;
    const double by = tt.yx*t.xy + tt.yy*t.yy + tt.yz*t.zy;
    const double bz = tt.yx*t.xz + tt.yy*t.yz + tt.yz*t.zz;
    const double cx = tt.zx*t.xx + tt.zy*t.yx + tt.zz*t.zx;
    const double cy = tt.zx*t.xy + tt.zy*t.yy + tt.zz*t.zy;
    const double cz = tt.zx*t.xz + tt.zy*t.yz + tt.zz*t.zz;
    return
    {
        ax*tt.xx + ay*tt.xy + az*tt.xz,
        ax*tt.yx + ay*tt.yy + az*tt.yz,
        ax*tt.zx + ay*tt.zy + az*tt.zz,
        bx*tt.xx + by*tt.xy + bz*tt.xz,
        bx*tt.yx + by*tt.yy + bz*tt.yz,
        bx*tt.zx + by*tt.zy + bz*tt.zz,
        cx*tt.xx + cy*tt.xy + cz*tt.xz,
        cx*tt.yx + cy*tt.yy + cz*tt.yz,
        cx*tt.zx + cy*tt.zy + cz*tt.zz
    };
}

GNC_HD double lerp(const double a, const double b, const double t)
{
    return (1.0 - t)*a + t*b;
}

GNC_HD vec3 lerp(const vec3& a, const vec3& b, const double t)
{
    const double onet = (1 - t);
    return {onet*a.x + t*b.x, onet*a.y + t*b.y, onet*a.z + t*b.z};
}

GNC_HD tens lerp(const tens& a, const tens& b, const double t)
{
    const double onet = (1 - t);
    return
    {
        onet*a.xx + t*b.xx, onet*a.xy + t*b.xy, onet*a.xz + t*b.xz,
        onet*a.yx + t*b.yx, onet*a.yy + t*b.yy, onet*a.yz + t*b.yz,
        onet*a.zx + t*b.zx, onet*a.zy + t*b.zy, onet*a.zz + t*b.zz
    };
}

// neg(s) = (s < 0) ? 1 : 0（scalarImpl.H:262）
GNC_HD double neg(const double s)
{
    return (s < 0) ? 1.0 : 0.0;
}


// ---------------------------------------------------------------------------
// 型別特性：分量數、零值、分量存取（場以 SoA 存放，每個分量一個陣列）
// ---------------------------------------------------------------------------

template<class T> struct NComp;
template<> struct NComp<double> { static constexpr int value = 1; };
template<> struct NComp<vec3> { static constexpr int value = 3; };
template<> struct NComp<tens> { static constexpr int value = 9; };

// 梯度的型別：outerProduct<vector, Type>
template<class T> struct GradOf;
template<> struct GradOf<double> { using type = vec3; };
template<> struct GradOf<vec3> { using type = tens; };

template<class T> GNC_HD T zeroOf();
template<> GNC_HD double zeroOf<double>() { return 0.0; }
template<> GNC_HD vec3 zeroOf<vec3>() { return {0.0, 0.0, 0.0}; }
template<> GNC_HD tens zeroOf<tens>() { return {0, 0, 0, 0, 0, 0, 0, 0, 0}; }

GNC_HD double cmpt(const double& v, int) { return v; }
GNC_HD double cmpt(const vec3& v, const int k)
{
    return k == 0 ? v.x : (k == 1 ? v.y : v.z);
}
GNC_HD double cmpt(const tens& t, const int k)
{
    const double* p = &t.xx;
    return p[k];
}

GNC_HD void setCmpt(double& v, int, const double s) { v = s; }
GNC_HD void setCmpt(vec3& v, const int k, const double s)
{
    if (k == 0) { v.x = s; } else if (k == 1) { v.y = s; } else { v.z = s; }
}
GNC_HD void setCmpt(tens& t, const int k, const double s)
{
    double* p = &t.xx;
    p[k] = s;
}

} // namespace gnc

#endif
