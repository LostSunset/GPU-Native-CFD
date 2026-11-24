// GPU-Native-CFD，GPL-3.0-or-later。見 boundaryConditions.cuh。

#include "fields/boundaryConditions.cuh"

#include "io/foamTokenizer.hpp"

#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <type_traits>

namespace gnc
{

const char* bcName(const Bc b)
{
    switch (b)
    {
        case Bc::calculated: return "calculated";
        case Bc::extrapolatedCalculated: return "extrapolatedCalculated";
        case Bc::fixedValue: return "fixedValue";
        case Bc::zeroGradient: return "zeroGradient";
        case Bc::empty: return "empty";
        case Bc::symmetryPlane: return "symmetryPlane";
        case Bc::symmetry: return "symmetry";
        case Bc::slip: return "slip";
        case Bc::mixed: return "mixed";
        case Bc::inletOutlet: return "inletOutlet";
        case Bc::fixedGradient: return "fixedGradient";
        case Bc::characteristicFarfieldPressure: return "characteristicFarfieldPressure";
        case Bc::characteristicFarfieldTemperature: return "characteristicFarfieldTemperature";
        case Bc::characteristicFarfieldVelocity: return "characteristicFarfieldVelocity";
        case Bc::characteristicWallPressure: return "characteristicWallPressure";
        case Bc::characteristicWallTemperature: return "characteristicWallTemperature";
        case Bc::boundaryCorrectedFixedValue: return "boundaryCorrectedFixedValue";
        case Bc::nutLowReWallFunction: return "nutLowReWallFunction";
        case Bc::boundaryCorrectedNutLowReWallFunction: return "boundaryCorrectedNutLowReWallFunction";
        case Bc::nutUSpaldingWallFunction: return "nutUSpaldingWallFunction";
        case Bc::compressibleAlphatWallFunction: return "compressible::alphatWallFunction";
        case Bc::kLowReWallFunction: return "kLowReWallFunction";
        case Bc::omegaWallFunction: return "omegaWallFunction";
        case Bc::nutkWallFunction: return "nutkWallFunction";
        case Bc::turbulentIntensityKineticEnergyInlet: return "turbulentIntensityKineticEnergyInlet";
        case Bc::turbulentMixingLengthFrequencyInlet: return "turbulentMixingLengthFrequencyInlet";
        case Bc::fixedEnergy: return "fixedEnergy";
        case Bc::gradientEnergy: return "gradientEnergy";
        case Bc::mixedEnergy: return "mixedEnergy";
        case Bc::freestreamVelocity: return "freestreamVelocity";
        case Bc::freestreamPressure: return "freestreamPressure";
        case Bc::noSlip: return "noSlip";
        case Bc::kqRWallFunction: return "kqRWallFunction";
    }
    return "?";
}


bool isMixedFamily(const Bc b)
{
    return b == Bc::mixed || isInletOutletFamily(b) || b == Bc::mixedEnergy
        || b == Bc::freestreamVelocity || b == Bc::freestreamPressure
        || b == Bc::characteristicFarfieldPressure
        || b == Bc::characteristicFarfieldTemperature
        || b == Bc::characteristicFarfieldVelocity;
}


bool isFixedValueFamily(const Bc b)
{
    return b == Bc::fixedValue || b == Bc::fixedEnergy
        || b == Bc::characteristicWallTemperature
        || b == Bc::boundaryCorrectedFixedValue
        || b == Bc::nutLowReWallFunction
        || b == Bc::boundaryCorrectedNutLowReWallFunction
        || b == Bc::nutUSpaldingWallFunction
        || b == Bc::compressibleAlphatWallFunction
        || b == Bc::kLowReWallFunction
        || b == Bc::omegaWallFunction
        || b == Bc::nutkWallFunction
        || b == Bc::noSlip;
}


bool isInletOutletFamily(const Bc b)
{
    return b == Bc::inletOutlet
        || b == Bc::turbulentIntensityKineticEnergyInlet
        || b == Bc::turbulentMixingLengthFrequencyInlet;
}


bool isZeroGradientFamily(const Bc b)
{
    return b == Bc::zeroGradient || b == Bc::kqRWallFunction;
}


bool isFixedGradientFamily(const Bc b)
{
    return b == Bc::fixedGradient || b == Bc::gradientEnergy
        || b == Bc::characteristicWallPressure;
}


bool fixesValue(const Bc b)
{
    // calculatedFvPatchField.H:158（extrapolatedCalculated 繼承）、
    // fixedValueFvPatchField.H:175、mixedFvPatchField.H:197
    return b == Bc::calculated || b == Bc::extrapolatedCalculated
        || isFixedValueFamily(b) || isMixedFamily(b);
}

namespace
{

constexpr int BS = 256;

int nBlocks(const std::int32_t n)
{
    return int((n + BS - 1)/BS);
}

Bc parseBc(const std::string& s)
{
    // 能量場的型別只由 thermo 建立，不從場檔讀
    for
    (
        const Bc b :
        {
            Bc::calculated, Bc::extrapolatedCalculated, Bc::fixedValue,
            Bc::zeroGradient, Bc::empty, Bc::symmetryPlane, Bc::symmetry,
            Bc::slip, Bc::mixed, Bc::inletOutlet, Bc::fixedGradient,
            Bc::characteristicFarfieldPressure,
            Bc::characteristicFarfieldTemperature,
            Bc::characteristicFarfieldVelocity,
            Bc::characteristicWallPressure,
            Bc::characteristicWallTemperature,
            Bc::boundaryCorrectedFixedValue,
            Bc::nutLowReWallFunction,
            Bc::boundaryCorrectedNutLowReWallFunction,
            Bc::nutUSpaldingWallFunction,
            Bc::compressibleAlphatWallFunction,
            Bc::kLowReWallFunction,
            Bc::omegaWallFunction,
            Bc::nutkWallFunction,
            Bc::turbulentIntensityKineticEnergyInlet,
            Bc::turbulentMixingLengthFrequencyInlet,
            Bc::freestreamVelocity,
            Bc::freestreamPressure,
            Bc::noSlip,
            Bc::kqRWallFunction
        }
    )
    {
        if (s == bcName(b))
        {
            return b;
        }
    }
    throw std::runtime_error
    (
        "boundary condition type " + s + " is not supported"
    );
}

bool isCharacteristicFarfield(const Bc b)
{
    return b == Bc::characteristicFarfieldPressure
        || b == Bc::characteristicFarfieldTemperature
        || b == Bc::characteristicFarfieldVelocity;
}

// 額外鍵的數值（characteristicBase 的 U、p、T）
std::vector<double> extraNumbers
(
    const PatchFieldSpec& s,
    const std::string& key,
    const std::string& where
)
{
    const std::string* text = s.extra(key);
    if (!text)
    {
        throw std::runtime_error(where + ": missing entry " + key);
    }
    std::vector<double> out;
    std::string tok;
    const auto flush = [&]()
    {
        if (!tok.empty())
        {
            double v = 0;
            if (!parseScalar(tok, v))
            {
                throw std::runtime_error(where + ": " + key + ": invalid number " + tok);
            }
            out.push_back(v);
            tok.clear();
        }
    };
    for (const char c : *text)
    {
        if (c == ' ' || c == '(' || c == ')')
        {
            flush();
        }
        else
        {
            tok.push_back(c);
        }
    }
    flush();
    return out;
}

CView<vec3> meshVec(const DeviceVec3Array& a)
{
    CView<vec3> v;
    v.p[0] = a.x.data();
    v.p[1] = a.y.data();
    v.p[2] = a.z.data();
    return v;
}

// 純量對旋轉不變（is_rotational_vectorspace_v<scalar> 為 false）
template<class T> constexpr bool isRotational() { return true; }
template<> constexpr bool isRotational<double>() { return false; }


struct PatchRange
{
    std::int32_t start;     // fvBoundaryFaces 中的起點
    std::int32_t size;
    const std::int32_t* fvbFaces;
    const std::int32_t* owner;
};

PatchRange range(const DeviceMesh& dm, const std::size_t pi)
{
    return
    {
        dm.fvPatchStart[pi], dm.fvPatchSize[pi],
        dm.fvBoundaryFaces.data(), dm.owner.data()
    };
}


// 各型別 evaluate 在 patch 內第 i 個面的運算（N-028：單一 patch 的 kernel 與 evaluateBCs 的多 patch
// kernel 共用，運算相同）
template<class T>
__device__ __forceinline__ void setZeroAt(const PatchRange& r, const std::int32_t i, const DView<T>& bnd)
{
    bnd.set(r.start + i, zeroOf<T>());
}

template<class T>
__device__ __forceinline__ void copyInternalAt(const PatchRange& r, const std::int32_t i, const CView<T>& in, const DView<T>& bnd)
{
    const std::int32_t k = r.start + i;
    bnd.set(k, in[r.owner[r.fvbFaces[k]]]);
}

// basicSymmetry／symmetryPlane 的 evaluate（向量與張量）
template<class T>
__device__ __forceinline__ void symmetryAt
(
    const PatchRange& r,
    const std::int32_t i,
    const bool planeNormal,
    const vec3& pn,
    const CView<vec3>& Sf,
    const double* __restrict__ magSf,
    const CView<T>& in,
    const DView<T>& bnd
)
{
    const std::int32_t k = r.start + i;
    const std::int32_t f = r.fvbFaces[k];
    const vec3 nHat = planeNormal ? pn : Sf[f]/magSf[f];
    const tens rot = symmetryRot(nHat);
    const T pif = in[r.owner[f]];
    bnd.set(k, 0.5*(pif + transform(rot, pif)));
}

// mixedFvPatchField::evaluate；inletOutlet 時先設 valueFraction = neg(phi_b)
template<class T>
__device__ __forceinline__ void mixedAt
(
    const PatchRange& r,
    const std::int32_t i,
    const bool inletOutlet,
    const double* __restrict__ phiB,
    const double* __restrict__ dc,
    const CView<T>& in,
    const CView<T>& refValue,
    const CView<T>& refGrad,
    double* __restrict__ valueFraction,
    const DView<T>& bnd
)
{
    const std::int32_t k = r.start + i;
    const std::int32_t f = r.fvbFaces[k];
    if (inletOutlet)
    {
        valueFraction[i] = neg(phiB[k]);
    }
    const T pif = in[r.owner[f]];
    bnd.set(k, lerp(pif + refGrad[i]/dc[f], refValue[i], valueFraction[i]));
}

// fixedGradientFvPatchField::evaluate：pif + gradient/deltaCoeffs
template<class T>
__device__ __forceinline__ void fixedGradientAt
(
    const PatchRange& r,
    const std::int32_t i,
    const double* __restrict__ dc,
    const CView<T>& in,
    const CView<T>& gradient,
    const DView<T>& bnd
)
{
    const std::int32_t k = r.start + i;
    const std::int32_t f = r.fvbFaces[k];
    bnd.set(k, in[r.owner[f]] + gradient[i]/dc[f]);
}

template<class T>
__global__ void kSetZero(const PatchRange r, const DView<T> bnd)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < r.size)
    {
        setZeroAt(r, i, bnd);
    }
}


template<class T>
__global__ void kCopyInternal(const PatchRange r, const CView<T> in, const DView<T> bnd)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < r.size)
    {
        copyInternalAt(r, i, in, bnd);
    }
}


template<class T>
__global__ void kEvalSymmetry
(
    const PatchRange r,
    const bool planeNormal,
    const vec3 pn,
    const CView<vec3> Sf,
    const double* __restrict__ magSf,
    const CView<T> in,
    const DView<T> bnd
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < r.size)
    {
        symmetryAt(r, i, planeNormal, pn, Sf, magSf, in, bnd);
    }
}


template<class T>
__global__ void kEvalMixed
(
    const PatchRange r,
    const bool inletOutlet,
    const double* __restrict__ phiB,
    const double* __restrict__ dc,
    const CView<T> in,
    const CView<T> refValue,
    const CView<T> refGrad,
    double* __restrict__ valueFraction,
    const DView<T> bnd
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < r.size)
    {
        mixedAt(r, i, inletOutlet, phiB, dc, in, refValue, refGrad, valueFraction, bnd);
    }
}


// evaluateBCs 的多 patch kernel（N-028）：各 patch 寫自己的邊界範圍，互不相依，一次啟動
enum EvalKind : int { ekCopyInternal, ekMixed, ekFixedGradient, ekSymmetry, ekSetZero };

constexpr int MaxEvalSegs = 8;

template<class T>
struct EvalSeg
{
    int kind;
    PatchRange r;
    bool flag;                  // mixed：inletOutlet；symmetry：planeNormal
    const double* phiB;
    CView<T> a;                 // mixed：refValue；fixedGradient：gradient
    CView<T> b;                 // mixed：refGrad
    double* vf;
    vec3 pn;
};

template<class T>
struct EvalSegs
{
    EvalSeg<T> s[MaxEvalSegs];
    unsigned start[MaxEvalSegs + 1];
    int k;
    const double* dc;
    CView<vec3> Sf;
    const double* magSf;
    CView<T> in;
    DView<T> bnd;
};

template<class T>
__global__ void kEvalBatch(const EvalSegs<T> e)
{
    const unsigned b = blockIdx.x;
    int q = 0;
    while (q + 1 < e.k && b >= e.start[q + 1])
    {
        ++q;
    }
    const EvalSeg<T>& g = e.s[q];
    const std::int32_t i = std::int32_t((b - e.start[q])*blockDim.x + threadIdx.x);
    if (i >= g.r.size)
    {
        return;
    }
    switch (g.kind)
    {
        case ekCopyInternal: copyInternalAt(g.r, i, e.in, e.bnd); break;
        case ekMixed: mixedAt(g.r, i, g.flag, g.phiB, e.dc, e.in, g.a, g.b, g.vf, e.bnd); break;
        case ekFixedGradient: fixedGradientAt(g.r, i, e.dc, e.in, g.a, e.bnd); break;
        case ekSymmetry:
            // isRotational<T>()（純量的對稱在主機端已改為 copyInternal）
            if constexpr (!std::is_same<T, double>::value)
            {
                symmetryAt(g.r, i, g.flag, g.pn, e.Sf, e.magSf, e.in, e.bnd);
            }
            break;
        default: setZeroAt(g.r, i, e.bnd); break;
    }
}

// evaluateBCs 收集各 patch，滿了或結束時一次啟動
template<class T>
struct EvalBatch
{
    EvalSegs<T> e{};
    unsigned blocks = 0;

    EvalBatch(const DeviceMesh& dm, VolField<T>& f)
    {
        e.k = 0;
        e.dc = dm.deltaCoeffs.data();
        e.Sf = meshVec(dm.Sf);
        e.magSf = dm.magSf.data();
        e.in = f.in.cview();
        e.bnd = f.bnd.view();
    }

    EvalSeg<T>& push(const int kind, const PatchRange& r)
    {
        if (e.k == MaxEvalSegs)
        {
            flush();
        }
        EvalSeg<T>& g = e.s[e.k];
        g = EvalSeg<T>{};
        g.kind = kind;
        g.r = r;
        e.start[e.k] = blocks;
        blocks += unsigned((r.size + BS - 1)/BS);
        ++e.k;
        return g;
    }

    void flush()
    {
        if (e.k == 0)
        {
            return;
        }
        e.start[e.k] = blocks;
        kEvalBatch<T><<<blocks, BS>>>(e);
        GNC_CUDA_LAUNCH_CHECK();
        e.k = 0;
        blocks = 0;
    }
};


// freestream*::updateCoeffs：valueFraction = 0.5 -+ 0.5*(U_b & nf)/mag(U_b)，
// nf = Sf/magSf（fvPatch::nf）；pressure 為 +，velocity 為 -
__global__ void kFreestreamFraction
(
    const PatchRange r,
    const bool pressure,
    const CView<vec3> Sf,
    const double* __restrict__ magSf,
    const CView<vec3> Ub,
    double* __restrict__ valueFraction
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= r.size)
    {
        return;
    }
    const std::int32_t k = r.start + i;
    const std::int32_t f = r.fvbFaces[k];
    const vec3 nf = Sf[f]/magSf[f];
    const vec3 up = Ub[k];
    const double t = (0.5*dot(up, nf))/mag(up);
    valueFraction[i] = pressure ? 0.5 + t : 0.5 - t;
}


// inletOutlet 的 valueFraction = neg(phi_b)（= 1 - pos0(phi_b)）
__global__ void kInletOutletFraction
(
    const PatchRange r,
    const double* __restrict__ phiB,
    double* __restrict__ valueFraction
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < r.size)
    {
        valueFraction[i] = neg(phiB[r.start + i]);
    }
}


// 各型別 snGrad 在 patch 內第 i 個面的運算（N-028：snGradBCs 的多 patch kernel 用）

// fvPatchField::snGrad 預設：dc*(pf - pif)
template<class T>
__device__ __forceinline__ void snGradDefaultAt
(
    const PatchRange& r, const std::int32_t i, const double* __restrict__ dc, const CView<T>& in,
    const CView<T>& bnd, const DView<T>& out
)
{
    const std::int32_t k = r.start + i;
    const std::int32_t f = r.fvbFaces[k];
    out.set(k, dc[f]*(bnd[k] - in[r.owner[f]]));
}

template<class T>
__device__ __forceinline__ void snGradSymmetryAt
(
    const PatchRange& r, const std::int32_t i, const bool planeNormal, const vec3& pn, const CView<vec3>& Sf,
    const double* __restrict__ magSf, const double* __restrict__ dc, const CView<T>& in, const DView<T>& out
)
{
    const std::int32_t k = r.start + i;
    const std::int32_t f = r.fvbFaces[k];
    const vec3 nHat = planeNormal ? pn : Sf[f]/magSf[f];
    const tens rot = symmetryRot(nHat);
    const T pif = in[r.owner[f]];
    out.set(k, (0.5*dc[f])*(transform(rot, pif) - pif));
}

template<class T>
__device__ __forceinline__ void snGradMixedAt
(
    const PatchRange& r, const std::int32_t i, const double* __restrict__ dc, const CView<T>& in,
    const CView<T>& refValue, const CView<T>& refGrad, const double* __restrict__ valueFraction, const DView<T>& out
)
{
    const std::int32_t k = r.start + i;
    const std::int32_t f = r.fvbFaces[k];
    const T pif = in[r.owner[f]];
    out.set(k, lerp(refGrad[i], (refValue[i] - pif)*dc[f], valueFraction[i]));
}

// snGradBCs 的多 patch kernel（N-028）：各 patch 寫自己的範圍，一次啟動
enum SnKind : int { skGradient, skMixed, skDefault, skZero, skSymmetry };

constexpr int MaxSnSegs = 8;

template<class T>
struct SnSeg
{
    int kind;
    PatchRange r;
    const double* dc;           // default、symmetry、mixed 的 deltaCoeffs（default 可為 nonOrth）
    CView<T> a;                 // gradient；mixed：refValue
    CView<T> b;                 // mixed：refGrad
    const double* vf;
    bool flag;                  // symmetry：planeNormal
    vec3 pn;
};

template<class T>
struct SnSegs
{
    SnSeg<T> s[MaxSnSegs];
    unsigned start[MaxSnSegs + 1];
    int k;
    CView<vec3> Sf;
    const double* magSf;
    CView<T> in;
    CView<T> bnd;
    DView<T> out;
};

template<class T>
__global__ void kSnGradBatch(const SnSegs<T> e)
{
    const unsigned b = blockIdx.x;
    int q = 0;
    while (q + 1 < e.k && b >= e.start[q + 1])
    {
        ++q;
    }
    const SnSeg<T>& g = e.s[q];
    const std::int32_t i = std::int32_t((b - e.start[q])*blockDim.x + threadIdx.x);
    if (i >= g.r.size)
    {
        return;
    }
    switch (g.kind)
    {
        case skGradient: e.out.set(g.r.start + i, g.a[i]); break;
        case skMixed: snGradMixedAt(g.r, i, g.dc, e.in, g.a, g.b, g.vf, e.out); break;
        case skDefault: snGradDefaultAt(g.r, i, g.dc, e.in, e.bnd, e.out); break;
        case skSymmetry:
            if constexpr (!std::is_same<T, double>::value)
            {
                snGradSymmetryAt(g.r, i, g.flag, g.pn, e.Sf, e.magSf, g.dc, e.in, e.out);
            }
            break;
        default: e.out.set(g.r.start + i, zeroOf<T>()); break;
    }
}

template<class T>
struct SnGradBatch
{
    SnSegs<T> e{};
    unsigned blocks = 0;

    SnGradBatch(const DeviceMesh& dm, const VolField<T>& f, DField<T>& out)
    {
        e.k = 0;
        e.Sf = meshVec(dm.Sf);
        e.magSf = dm.magSf.data();
        e.in = f.in.cview();
        e.bnd = f.bnd.cview();
        e.out = out.view();
    }

    SnSeg<T>& push(const int kind, const PatchRange& r)
    {
        if (e.k == MaxSnSegs)
        {
            flush();
        }
        SnSeg<T>& g = e.s[e.k];
        g = SnSeg<T>{};
        g.kind = kind;
        g.r = r;
        e.start[e.k] = blocks;
        blocks += unsigned((r.size + BS - 1)/BS);
        ++e.k;
        return g;
    }

    void flush()
    {
        if (e.k == 0)
        {
            return;
        }
        e.start[e.k] = blocks;
        kSnGradBatch<T><<<blocks, BS>>>(e);
        GNC_CUDA_LAUNCH_CHECK();
        e.k = 0;
        blocks = 0;
    }
};


// fixedGradientFvPatchField::evaluate：pf = pif + gradient/dc
template<class T>
__global__ void kEvalFixedGradient
(
    const PatchRange r,
    const double* __restrict__ dc,
    const CView<T> in,
    const CView<T> gradient,
    const DView<T> bnd
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < r.size)
    {
        fixedGradientAt(r, i, dc, in, gradient, bnd);
    }
}


std::vector<double> slice
(
    const std::vector<double>& v,
    const std::size_t start,
    const std::size_t n,
    const int nComp
)
{
    return std::vector<double>
    (
        v.begin() + start*nComp,
        v.begin() + (start + n)*nComp
    );
}


void checkPatchType(const Patch& mp, const Bc b)
{
    const bool ok =
        mp.isEmpty() ? b == Bc::empty
      : mp.type == "symmetryPlane" ? b == Bc::symmetryPlane
      : mp.type == "symmetry" ? b == Bc::symmetry
      : (
            b == Bc::nutLowReWallFunction || b == Bc::boundaryCorrectedNutLowReWallFunction
         || b == Bc::nutUSpaldingWallFunction || b == Bc::compressibleAlphatWallFunction
         || b == Bc::nutkWallFunction
        )
        ? mp.type == "wall"     // nutWallFunctionFvPatchScalarField::checkType 等
      : (b != Bc::empty && b != Bc::symmetryPlane && b != Bc::symmetry);
    if (!ok)
    {
        throw std::runtime_error
        (
            "patch " + mp.name + " (" + mp.type + "): boundary condition "
          + bcName(b) + " does not match the patch type"
        );
    }
}

template<class T>
__global__ void kCopySlice(const std::int32_t start, const std::int32_t n, const CView<T> src, const DView<T> dst)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        dst.set(i, src[start + i]);
    }
}


// wallFunctionCoefficients::calcYPlusLam：ypl = 11，迭代 10 次
// ypl = log(max(E*ypl, 1))/kappa（主機端，用 glibc 的 log；volatile 讓編譯器不以
// 常數折疊）
__attribute__((noinline)) double calcYPlusLam(const double kappa, const double E)
{
    volatile double k = kappa, e = E;
    double ypl = 11;
    for (int iter = 0; iter < 10; ++iter)
    {
        ypl = std::log(foamMax(e*ypl, 1.0))/k;
    }
    return ypl;
}

// kOmegaSST 的壁面函數（N4b）的字典項：wallFunctionCoefficients（Cmu、kappa、E）；
// omegaWallFunction 的 wallFunctionBlenders（預設 binomial、n 2）與 beta1；
// nutkWallFunction 的 wallFunctionBlenders（預設 stepwise、n 4）；
// kLowReWallFunction 的 Ceps2、Ck、Bk、C
template<class T>
void readKOmegaWallFunction(const PatchFieldSpec& s, const std::string& what, PatchBC<T>& bc)
{
    const std::string where = "field " + what;
    const auto opt = [&](const char* key, double& v)
    {
        if (s.extra(key))
        {
            const std::vector<double> x = extraNumbers(s, key, where);
            if (x.size() != 1)
            {
                throw std::runtime_error(where + ": " + key + " malformed");
            }
            v = x[0];
        }
    };
    opt("Cmu", bc.wallCmu);
    opt("kappa", bc.wallKappa);
    opt("E", bc.wallE);
    bc.yPlusLam = calcYPlusLam(bc.wallKappa, bc.wallE);
    if (bc.type == Bc::kLowReWallFunction)
    {
        opt("Ceps2", bc.wallCeps2);
        opt("Ck", bc.wallCk);
        opt("Bk", bc.wallBk);
        opt("C", bc.wallC);
    }
    else
    {
        const bool omega = bc.type == Bc::omegaWallFunction;
        bc.wallBlending = omega ? 2 : 0;
        bc.wallBlendN = omega ? 2.0 : 4.0;
        if (const std::string* b = s.extra("blending"))
        {
            const char* names[] = {"stepwise", "max", "binomial", "exponential", "tanh"};
            int found = -1;
            for (int i = 0; i < 5; ++i)
            {
                if (*b == names[i]) found = i;
            }
            // exponential 與 tanh 需要另外的超越函數，目前不支援
            if (found < 0 || found > 2)
            {
                throw std::runtime_error(where + ": blending " + *b + " is not supported (stepwise, max, binomial)");
            }
            bc.wallBlending = found;
        }
        opt("n", bc.wallBlendN);
        if (omega)
        {
            opt("beta1", bc.wallBeta1);
        }
    }
    bc.writeExtras = s.extras;
}


// 各型別 dict 建構子在建構時對值做的事（evaluate 或取相鄰格值）
template<class T>
void constructBCs(const DeviceMesh& dm, VolField<T>& f)
{
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const PatchRange r = range(dm, pi);
        if (r.size == 0)
        {
            continue;
        }
        PatchBC<T>& bc = f.bc[pi];
        const bool copy =
            bc.pifAtConstruction
         || isZeroGradientFamily(bc.type)
         || bc.type == Bc::extrapolatedCalculated
         || (
                !isRotational<T>()
             && (bc.type == Bc::symmetry || bc.type == Bc::symmetryPlane || bc.type == Bc::slip)
            );
        if (copy)
        {
            kCopyInternal<T><<<nBlocks(r.size), BS>>>(r, f.in.cview(), f.bnd.view());
        }
        else if (bc.type == Bc::symmetry || bc.type == Bc::symmetryPlane || bc.type == Bc::slip)
        {
            kEvalSymmetry<T><<<nBlocks(r.size), BS>>>
            (
                r, bc.type == Bc::symmetryPlane, dm.patchNormal[pi],
                meshVec(dm.Sf), dm.magSf.data(), f.in.cview(), f.bnd.view()
            );
        }
        else if (bc.type == Bc::fixedGradient)
        {
            kEvalFixedGradient<T><<<nBlocks(r.size), BS>>>
            (
                r, dm.deltaCoeffs.data(), f.in.cview(), bc.gradient.cview(), f.bnd.view()
            );
        }
        GNC_CUDA_LAUNCH_CHECK();
        // 特徵遠場：refValue = 建構後的值
        if (isCharacteristicFarfield(bc.type))
        {
            bc.refValue.allocate(std::size_t(r.size));
            kCopySlice<T><<<nBlocks(r.size), BS>>>
            (
                r.start, r.size, f.bnd.cview(), bc.refValue.view()
            );
            GNC_CUDA_LAUNCH_CHECK();
        }
    }
}

} // namespace


template<class T>
VolField<T> uploadVolField(const DeviceMesh& dm, const HostField& h)
{
    constexpr int N = NComp<T>::value;
    if (h.isSurface || h.nComp != N)
    {
        throw std::runtime_error("field " + h.name + ": unexpected class " + h.cls);
    }

    VolField<T> f;
    f.name = h.name;
    f.in.uploadAoS(h.internal);

    std::vector<double> b(std::size_t(dm.nFvBoundaryFaces())*N, 0.0);
    f.bc.resize(dm.patches.size());

    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const PatchFieldSpec& s = h.patches[pi];
        const Patch& mp = dm.patches[pi];
        PatchBC<T>& bc = f.bc[pi];
        bc.type = parseBc(s.type);
        bc.phiName = s.phiName;
        checkPatchType(mp, bc.type);

        const std::size_t start = std::size_t(dm.fvPatchStart[pi]);
        const std::size_t n = std::size_t(dm.fvPatchSize[pi]);
        const auto need = [&](const bool has, const char* what)
        {
            if (!has)
            {
                throw std::runtime_error
                (
                    "field " + h.name + " patch " + mp.name + ": "
                  + bcName(bc.type) + " needs " + what
                );
            }
        };

        // 建構時的值（各型別的 dict 建構子）：
        //   value：取檔案的 value；pif：取相鄰格值；eval：建構時 evaluate
        std::vector<double> value;
        bool pif = false;
        switch (bc.type)
        {
            case Bc::fixedValue:
            case Bc::calculated:
            case Bc::boundaryCorrectedFixedValue:
                need(s.hasValue, "value");
                value = s.value;
                break;
            case Bc::nutLowReWallFunction:
            case Bc::boundaryCorrectedNutLowReWallFunction:
            case Bc::nutUSpaldingWallFunction:
            case Bc::compressibleAlphatWallFunction:
            {
                need(s.hasValue, "value");
                value = s.value;
                // wallFunctionCoefficients(dict)、nutUSpalding 的 maxIter、tolerance、
                // alphatWallFunction 的 Prt（沒有就用預設值）
                const std::string where = "field " + h.name + " patch " + mp.name;
                const auto opt = [&](const char* key, double& v)
                {
                    if (s.extra(key))
                    {
                        const std::vector<double> x = extraNumbers(s, key, where);
                        if (x.size() != 1)
                        {
                            throw std::runtime_error(where + ": " + key + " malformed");
                        }
                        v = x[0];
                    }
                };
                opt("kappa", bc.wallKappa);
                opt("E", bc.wallE);
                opt("tolerance", bc.wallTolerance);
                opt("Prt", bc.wallPrt);
                double mi = bc.wallMaxIter;
                opt("maxIter", mi);
                bc.wallMaxIter = int(mi);
                break;
            }
            case Bc::kLowReWallFunction:
            case Bc::omegaWallFunction:
            case Bc::nutkWallFunction:
            {
                need(s.hasValue, "value");
                value = s.value;
                readKOmegaWallFunction(s, h.name + " patch " + mp.name, bc);
                // omegaWallFunction 的 dict 建構子：讀 value 之後 extrapolateInternal()
                // （omegaWallFunctionFvPatchScalarField.C:405）
                pif = bc.type == Bc::omegaWallFunction;
                break;
            }
            case Bc::turbulentIntensityKineticEnergyInlet:
            case Bc::turbulentMixingLengthFrequencyInlet:
            {
                // dict 建構子：readValueEntry(MUST_READ)；refValue、refGrad、
                // valueFraction 為 0（refValue 在 updateCoeffs 時才算）
                need(s.hasValue, "value");
                value = s.value;
                const std::string where = "field " + h.name + " patch " + mp.name;
                const bool kInlet = bc.type == Bc::turbulentIntensityKineticEnergyInlet;
                const char* key = kInlet ? "intensity" : "mixingLength";
                const std::vector<double> x = extraNumbers(s, key, where);
                if (x.size() != 1)
                {
                    throw std::runtime_error(where + ": " + key + " missing or malformed");
                }
                (kInlet ? bc.intensity : bc.mixingLength) = x[0];
                // 參考的場名稱（U、k）只支援預設值
                const char* nameKey = kInlet ? "U" : "k";
                const std::string* nv = s.extra(nameKey);
                if (nv && *nv != nameKey)
                {
                    throw std::runtime_error(where + ": " + nameKey + " " + *nv + " is not supported");
                }
                bc.writeExtras = s.extras;
                bc.refValue.uploadAoS(std::vector<double>(n*N, 0.0));
                bc.refGrad.uploadAoS(std::vector<double>(n*N, 0.0));
                bc.valueFraction.upload(std::vector<double>(n, 0.0));
                break;
            }
            case Bc::characteristicWallTemperature:
                // characteristicWallTemperatureFvPatchScalarField.C：沒有 value 時取 pif
                if (s.hasValue) { value = s.value; } else { pif = true; }
                break;
            case Bc::characteristicWallPressure:
                // value 與 gradient 都有才是續算；否則等同 zeroGradient
                if (s.hasValue && s.hasGradient)
                {
                    value = s.value;
                    bc.gradient.uploadAoS(s.gradient);
                }
                else
                {
                    pif = true;
                    bc.gradient.uploadAoS(std::vector<double>(n*N, 0.0));
                }
                break;
            case Bc::characteristicFarfieldPressure:
            case Bc::characteristicFarfieldTemperature:
            case Bc::characteristicFarfieldVelocity:
            {
                if (s.hasValueFraction)
                {
                    throw std::runtime_error
                    (
                        "field " + h.name + " patch " + mp.name + ": restart"
                        " values of " + bcName(bc.type) + " are not supported"
                    );
                }
                const std::string where = "field " + h.name + " patch " + mp.name;
                // characteristicBase 的場名稱（UName 等）只支援預設值
                for (const char* key : {"UName", "pName", "TName", "phiName", "rhoName"})
                {
                    const std::string* v = s.extra(key);
                    const std::string deflt = std::string(key).substr(0, std::string(key).size() - 4);
                    if (v && *v != deflt)
                    {
                        throw std::runtime_error
                        (
                            where + ": " + key + " " + *v + " is not supported (" + deflt + " only)"
                        );
                    }
                }
                const std::vector<double> Ur = extraNumbers(s, "U", where);
                const std::vector<double> pr = extraNumbers(s, "p", where);
                const std::vector<double> Tr = extraNumbers(s, "T", where);
                if (Ur.size() != 3 || pr.size() != 1 || Tr.size() != 1)
                {
                    throw std::runtime_error(where + ": U, p, T reference values malformed");
                }
                bc.URef = vec3{Ur[0], Ur[1], Ur[2]};
                bc.pRef = pr[0];
                bc.TRef = Tr[0];
                // refValue = value、refGrad = 0、valueFraction = 1
                // （characteristicFarfield*FvPatch*Field.C 的 dict 建構子）
                if (s.hasValue) { value = s.value; } else { pif = true; }
                bc.refGrad.uploadAoS(std::vector<double>(n*N, 0.0));
                bc.valueFraction.upload(std::vector<double>(n, 1.0));
                break;
            }
            case Bc::inletOutlet:
                need(s.hasRefValue, "inletValue");
                value = s.hasValue ? s.value : s.refValue;
                bc.refValue.uploadAoS(s.refValue);
                bc.refGrad.uploadAoS(std::vector<double>(n*N, 0.0));
                bc.valueFraction.upload(std::vector<double>(n, 0.0));
                break;
            case Bc::mixed:
                need(s.hasRefValue, "refValue");
                need(s.hasRefGrad, "refGradient");
                need(s.hasValueFraction, "valueFraction");
                need(s.hasValue, "value");
                value = s.value;
                bc.refValue.uploadAoS(s.refValue);
                bc.refGrad.uploadAoS(s.refGrad);
                bc.valueFraction.upload(s.valueFraction);
                break;
            case Bc::fixedGradient:
                need(s.hasGradient, "gradient");
                bc.gradient.uploadAoS(s.gradient);
                break;
            case Bc::freestreamVelocity:
            case Bc::freestreamPressure:
            {
                // freestream*FvPatch*Field.cxx 的 dict 建構子：refValue = freestreamValue、
                // refGrad = 0、valueFraction = 1（速度）或 0（壓力）；沒有 value 時取
                // freestreamValue
                need(s.hasRefValue, "freestreamValue");
                value = s.hasValue ? s.value : s.refValue;
                bc.refValue.uploadAoS(s.refValue);
                bc.refGrad.uploadAoS(std::vector<double>(n*N, 0.0));
                const double vf0 = bc.type == Bc::freestreamVelocity ? 1.0 : 0.0;
                bc.valueFraction.upload(std::vector<double>(n, vf0));
                const std::string* un = s.extra("U");
                if (un && *un != "U")
                {
                    throw std::runtime_error
                    (
                        "field " + h.name + " patch " + mp.name + ": U " + *un
                      + " is not supported (U only)"
                    );
                }
                break;
            }
            case Bc::noSlip:
                // noSlipFvPatchVectorField：值為 0
                value.assign(n*N, 0.0);
                break;
            default:
                // zeroGradient（含 kqRWallFunction）、extrapolatedCalculated、對稱類、empty：建構時
                // evaluate（下面的 constructBCs）
                break;
        }
        if (!value.empty())
        {
            std::copy(value.begin(), value.end(), b.begin() + start*N);
        }
        bc.pifAtConstruction = pif;
    }
    f.bnd.uploadAoS(b);
    constructBCs(dm, f);
    return f;
}


SurfField<double> uploadSurfaceScalarField(const DeviceMesh& dm, const HostField& h)
{
    if (!h.isSurface || h.nComp != 1)
    {
        throw std::runtime_error("field " + h.name + ": expected surfaceScalarField");
    }
    SurfField<double> f;
    f.name = h.name;
    f.in.uploadAoS(h.internal);
    std::vector<double> b(std::size_t(dm.nFvBoundaryFaces()), 0.0);
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const PatchFieldSpec& s = h.patches[pi];
        const std::size_t n = std::size_t(dm.fvPatchSize[pi]);
        if (n == 0)
        {
            continue;
        }
        // 面場的邊界值直接取 value（calculated，或 constraint patch 上的
        // symmetryPlane／symmetry fvsPatchField，兩者都只是存值）
        if (!s.hasValue)
        {
            throw std::runtime_error
            (
                "surface field " + h.name + " patch " + dm.patches[pi].name
              + ": value missing"
            );
        }
        std::copy(s.value.begin(), s.value.end(), b.begin() + dm.fvPatchStart[pi]);
    }
    f.bnd.uploadAoS(b);
    return f;
}


template<class T>
VolField<T> makeVolField(const DeviceMesh& dm, const std::string& name, const Bc generic)
{
    VolField<T> f;
    f.name = name;
    f.in.allocate(std::size_t(dm.nCells));
    f.bnd.allocate(std::size_t(dm.nFvBoundaryFaces()));
    {
        ArrayBatch b;
        f.in.zero(&b);
        f.bnd.zero(&b);
    }
    f.bc.resize(dm.patches.size());
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const Patch& mp = dm.patches[pi];
        f.bc[pi].type =
            mp.isEmpty() ? Bc::empty
          : mp.type == "symmetryPlane" ? Bc::symmetryPlane
          : mp.type == "symmetry" ? Bc::symmetry
          : generic;
    }
    return f;
}


template<class T>
void evaluateBCs
(
    const DeviceMesh& dm,
    VolField<T>& f,
    const SurfField<double>* phi,
    const bool coeffsUpdated
)
{
    // 各 patch 加進一個批次，最後一次啟動（N-028）
    EvalBatch<T> batch(dm, f);
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const PatchRange r = range(dm, pi);
        if (r.size == 0)
        {
            continue;
        }
        PatchBC<T>& bc = f.bc[pi];
        const auto requireUpdated = [&]()
        {
            if (!coeffsUpdated)
            {
                throw std::runtime_error
                (
                    "field " + f.name + " patch " + dm.patches[pi].name + ": evaluate of "
                  + bcName(bc.type) + " needs the coefficients of updateCoeffs"
                );
            }
        };
        switch (bc.type)
        {
            case Bc::fixedValue:
            case Bc::calculated:
            case Bc::empty:
            case Bc::boundaryCorrectedFixedValue:
            case Bc::noSlip:
                break;
            case Bc::fixedEnergy:
                // fixedEnergy 的 updateCoeffs 以 == 指定 he_b = HE(p_b, T_b)，evaluate 不改值
                requireUpdated();
                break;
            case Bc::nutUSpaldingWallFunction:
            case Bc::compressibleAlphatWallFunction:
            case Bc::nutkWallFunction:
            case Bc::kLowReWallFunction:
            case Bc::omegaWallFunction:
                // updateCoeffs 需要 U、nu、y、rho、k、nut：由紊流模型在 updateCoeffs
                // 的時點算好再呼叫這裡（turbulence/），evaluate 不改值
                break;
            case Bc::nutLowReWallFunction:
            case Bc::boundaryCorrectedNutLowReWallFunction:
                // nutWallFunctionFvPatchScalarField::updateCoeffs：
                // operator==(calcNut())，nutLowReWallFunction 的 calcNut 為 0
                if constexpr (std::is_same<T, double>::value)
                {
                    batch.push(ekSetZero, r);
                }
                else
                {
                    throw std::runtime_error("nut wall functions are scalar boundary conditions");
                }
                break;
            case Bc::characteristicWallTemperature:
                requireUpdated();
                break;
            case Bc::zeroGradient:
            case Bc::kqRWallFunction:
            case Bc::extrapolatedCalculated:
                batch.push(ekCopyInternal, r);
                break;
            case Bc::symmetryPlane:
            case Bc::symmetry:
            case Bc::slip:
                if constexpr (!isRotational<T>())
                {
                    batch.push(ekCopyInternal, r);
                }
                else
                {
                    EvalSeg<T>& g = batch.push(ekSymmetry, r);
                    g.flag = bc.type == Bc::symmetryPlane;
                    g.pn = dm.patchNormal[pi];
                }
                break;
            case Bc::freestreamVelocity:
            case Bc::freestreamPressure:
            case Bc::mixedEnergy:
                // valueFraction（freestream*）或 refValue、refGrad、valueFraction
                // （mixedEnergy）由呼叫端在 updateCoeffs 的時點算好
                requireUpdated();
                {
                    EvalSeg<T>& g = batch.push(ekMixed, r);
                    g.a = bc.refValue.cview();
                    g.b = bc.refGrad.cview();
                    g.vf = bc.valueFraction.data();
                }
                break;
            case Bc::gradientEnergy:
                requireUpdated();
                batch.push(ekFixedGradient, r).a = bc.gradient.cview();
                break;
            case Bc::mixed:
            case Bc::inletOutlet:
            case Bc::turbulentIntensityKineticEnergyInlet:
            case Bc::turbulentMixingLengthFrequencyInlet:
            {
                // 紊流入口的 refValue 由紊流模型在 updateCoeffs 的時點算好
                const bool io = isInletOutletFamily(bc.type);
                if (io && (!phi || phi->name != bc.phiName))
                {
                    throw std::runtime_error
                    (
                        "field " + f.name + " patch " + dm.patches[pi].name
                      + ": inletOutlet needs flux " + bc.phiName
                    );
                }
                EvalSeg<T>& g = batch.push(ekMixed, r);
                g.flag = io;
                g.phiB = io ? phi->bnd.cview().p[0] : nullptr;
                g.a = bc.refValue.cview();
                g.b = bc.refGrad.cview();
                g.vf = bc.valueFraction.data();
                break;
            }
            case Bc::characteristicFarfieldPressure:
            case Bc::characteristicFarfieldTemperature:
            case Bc::characteristicFarfieldVelocity:
                requireUpdated();
                {
                    EvalSeg<T>& g = batch.push(ekMixed, r);
                    g.a = bc.refValue.cview();
                    g.b = bc.refGrad.cview();
                    g.vf = bc.valueFraction.data();
                }
                break;
            case Bc::characteristicWallPressure:
                requireUpdated();
                [[fallthrough]];
            case Bc::fixedGradient:
                batch.push(ekFixedGradient, r).a = bc.gradient.cview();
                break;
            default:
                // 能量場的型別：heThermo 的 he 邊界由 updateCoeffs（hisaBoundaryUpdate
                // 的 updateEnergyT）或 updateFields 的 he(p, T) 指定，不在這裡 evaluate
                throw std::runtime_error
                (
                    "field " + f.name + " patch " + dm.patches[pi].name + ": evaluate of "
                  + bcName(bc.type) + " is not implemented yet (needs the solver state)"
                );
        }
    }
    batch.flush();
}


template<class T>
void updateInletOutletCoeffs
(
    const DeviceMesh& dm,
    VolField<T>& f,
    const SurfField<double>& phi
)
{
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const PatchRange r = range(dm, pi);
        PatchBC<T>& bc = f.bc[pi];
        if (r.size == 0 || !isInletOutletFamily(bc.type))
        {
            continue;
        }
        if (phi.name != bc.phiName)
        {
            throw std::runtime_error
            (
                "field " + f.name + " patch " + dm.patches[pi].name
              + ": inletOutlet needs flux " + bc.phiName
            );
        }
        kInletOutletFraction<<<nBlocks(r.size), BS>>>(r, phi.bnd.cview().p[0], bc.valueFraction.data());
        GNC_CUDA_LAUNCH_CHECK();
    }
}


namespace
{

void updateFreestream
(
    const DeviceMesh& dm,
    std::vector<PatchBC<double>>* pbc,
    std::vector<PatchBC<vec3>>* ubc,
    const VolField<vec3>& U,
    const Bc type
)
{
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const PatchRange r = range(dm, pi);
        const Bc t = pbc ? (*pbc)[pi].type : (*ubc)[pi].type;
        if (r.size == 0 || t != type)
        {
            continue;
        }
        double* vf = pbc ? (*pbc)[pi].valueFraction.data() : (*ubc)[pi].valueFraction.data();
        kFreestreamFraction<<<nBlocks(r.size), BS>>>
        (
            r, type == Bc::freestreamPressure, meshVec(dm.Sf), dm.magSf.data(),
            U.bnd.cview(), vf
        );
        GNC_CUDA_LAUNCH_CHECK();
    }
}

} // namespace


void updateFreestreamVelocityCoeffs(const DeviceMesh& dm, VolField<vec3>& U)
{
    updateFreestream(dm, nullptr, &U.bc, U, Bc::freestreamVelocity);
}


void updateFreestreamPressureCoeffs
(
    const DeviceMesh& dm,
    VolField<double>& p,
    const VolField<vec3>& U
)
{
    updateFreestream(dm, &p.bc, nullptr, U, Bc::freestreamPressure);
}


template<class T>
void finishAssignment(const DeviceMesh& dm, VolField<T>& f)
{
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const PatchRange r = range(dm, pi);
        if (r.size == 0)
        {
            continue;
        }
        const Bc t = f.bc[pi].type;
        switch (t)
        {
            case Bc::calculated:
            case Bc::extrapolatedCalculated:
            case Bc::zeroGradient:
            case Bc::kqRWallFunction:
            case Bc::empty:
                break;
            case Bc::symmetryPlane:
            case Bc::symmetry:
            case Bc::slip:
                if constexpr (!isRotational<T>())
                {
                    kCopyInternal<T><<<nBlocks(r.size), BS>>>(r, f.in.cview(), f.bnd.view());
                }
                else
                {
                    kEvalSymmetry<T><<<nBlocks(r.size), BS>>>
                    (
                        r, t == Bc::symmetryPlane, dm.patchNormal[pi],
                        meshVec(dm.Sf), dm.magSf.data(), f.in.cview(), f.bnd.view()
                    );
                }
                break;
            default:
                throw std::runtime_error
                (
                    "field " + f.name + " patch " + dm.patches[pi].name + ": assignment to "
                  + bcName(t) + " is not supported (its operator= keeps the old value)"
                );
        }
        GNC_CUDA_LAUNCH_CHECK();
    }
}


template<class T>
void snGradBCs(const DeviceMesh& dm, const VolField<T>& f, DField<T>& out)
{
    out.allocate(std::size_t(dm.nFvBoundaryFaces()));
    // 各 patch 加進一個批次，最後一次啟動（N-028；各型別的運算與原本逐 patch 的 kernel 相同）
    SnGradBatch<T> batch(dm, f, out);
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const PatchRange r = range(dm, pi);
        if (r.size == 0)
        {
            continue;
        }
        const PatchBC<T>& bc = f.bc[pi];
        if (isFixedGradientFamily(bc.type))
        {
            batch.push(skGradient, r).a = bc.gradient.cview();
            continue;
        }
        if (isMixedFamily(bc.type))
        {
            SnSeg<T>& g = batch.push(skMixed, r);
            g.dc = dm.deltaCoeffs.data();
            g.a = bc.refValue.cview();
            g.b = bc.refGrad.cview();
            g.vf = bc.valueFraction.data();
            continue;
        }
        switch (bc.type)
        {
            case Bc::fixedValue:
            case Bc::calculated:
            case Bc::extrapolatedCalculated:
            case Bc::characteristicWallTemperature:
            case Bc::fixedEnergy:
            case Bc::nutLowReWallFunction:
            case Bc::nutUSpaldingWallFunction:
            case Bc::compressibleAlphatWallFunction:
            case Bc::kLowReWallFunction:
            case Bc::omegaWallFunction:
            case Bc::nutkWallFunction:
            case Bc::noSlip:
                batch.push(skDefault, r).dc = dm.deltaCoeffs.data();
                break;
            case Bc::boundaryCorrectedFixedValue:
            case Bc::boundaryCorrectedNutLowReWallFunction:
                // (pf - pif)*nonOrthDeltaCoeffs（boundaryCorrectedFixedValue
                // FvPatchField.C 的 snGrad）
                batch.push(skDefault, r).dc = dm.nonOrthDeltaCoeffs.data();
                break;
            case Bc::zeroGradient:
            case Bc::kqRWallFunction:
            case Bc::empty:
                batch.push(skZero, r);
                break;
            case Bc::symmetryPlane:
            case Bc::symmetry:
            case Bc::slip:
                if constexpr (!isRotational<T>())
                {
                    batch.push(skZero, r);
                }
                else
                {
                    SnSeg<T>& g = batch.push(skSymmetry, r);
                    g.flag = bc.type == Bc::symmetryPlane;
                    g.pn = dm.patchNormal[pi];
                    g.dc = dm.deltaCoeffs.data();
                }
                break;
            default:
                throw std::runtime_error
                (
                    std::string("snGrad of ") + bcName(bc.type) + " is not implemented"
                );
        }
    }
    batch.flush();
}


template<class T>
HostField downloadVolField
(
    const DeviceMesh& dm,
    const VolField<T>& f,
    const std::string& cls,
    const std::string& dimensions
)
{
    constexpr int N = NComp<T>::value;
    HostField h;
    h.name = f.name;
    h.cls = cls;
    h.nComp = N;
    h.dimensions = dimensions;
    h.internal = f.in.downloadAoS();
    const std::vector<double> b = f.bnd.downloadAoS();
    h.patches.resize(dm.patches.size());
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const PatchBC<T>& bc = f.bc[pi];
        PatchFieldSpec& s = h.patches[pi];
        s.type = bcName(bc.type);
        s.phiName = bc.phiName;
        const std::size_t start = std::size_t(dm.fvPatchStart[pi]);
        const std::size_t n = std::size_t(dm.fvPatchSize[pi]);
        if
        (
            bc.type == Bc::characteristicFarfieldPressure
         || bc.type == Bc::characteristicFarfieldTemperature
         || bc.type == Bc::characteristicFarfieldVelocity
        )
        {
            char buf[128];
            std::snprintf
            (
                buf, sizeof(buf), "( %.17g %.17g %.17g )",
                bc.URef.x, bc.URef.y, bc.URef.z
            );
            s.extras.emplace_back("U", buf);
            std::snprintf(buf, sizeof(buf), "%.17g", bc.pRef);
            s.extras.emplace_back("p", buf);
            std::snprintf(buf, sizeof(buf), "%.17g", bc.TRef);
            s.extras.emplace_back("T", buf);
        }
        if (isFixedGradientFamily(bc.type))
        {
            s.hasValue = true;
            s.value = slice(b, start, n, N);
            s.hasGradient = true;
            s.gradient = bc.gradient.downloadAoS();
            continue;
        }
        switch (bc.type)
        {
            case Bc::fixedValue:
            case Bc::calculated:
            case Bc::extrapolatedCalculated:
            case Bc::characteristicWallTemperature:
            case Bc::boundaryCorrectedFixedValue:
            case Bc::nutLowReWallFunction:
            case Bc::boundaryCorrectedNutLowReWallFunction:
            case Bc::nutUSpaldingWallFunction:
            case Bc::compressibleAlphatWallFunction:
            case Bc::fixedEnergy:
            case Bc::characteristicFarfieldPressure:
            case Bc::characteristicFarfieldTemperature:
            case Bc::characteristicFarfieldVelocity:
                s.hasValue = true;
                s.value = slice(b, start, n, N);
                break;
            case Bc::kLowReWallFunction:
            case Bc::omegaWallFunction:
            case Bc::nutkWallFunction:
            case Bc::turbulentIntensityKineticEnergyInlet:
            case Bc::turbulentMixingLengthFrequencyInlet:
                s.extras = bc.writeExtras;
                s.hasValue = true;
                s.value = slice(b, start, n, N);
                break;
            case Bc::inletOutlet:
            case Bc::freestreamVelocity:
            case Bc::freestreamPressure:
                // freestream* 寫 freestreamValue（= refValue）與 value
                s.hasValue = true;
                s.value = slice(b, start, n, N);
                s.hasRefValue = true;
                s.refValue = bc.refValue.downloadAoS();
                break;
            case Bc::kqRWallFunction:
                s.hasValue = true;
                s.value = slice(b, start, n, N);
                break;
            case Bc::mixed:
                s.hasValue = true;
                s.value = slice(b, start, n, N);
                s.hasRefValue = true;
                s.refValue = bc.refValue.downloadAoS();
                s.hasRefGrad = true;
                s.refGrad = bc.refGrad.downloadAoS();
                s.hasValueFraction = true;
                s.valueFraction = bc.valueFraction.download();
                break;
            default:
                break;
        }
    }
    return h;
}


HostField downloadSurfaceScalarField
(
    const DeviceMesh& dm,
    const SurfField<double>& f,
    const std::string& dimensions
)
{
    HostField h;
    h.name = f.name;
    h.cls = "surfaceScalarField";
    h.nComp = 1;
    h.isSurface = true;
    h.dimensions = dimensions;
    h.internal = f.in.downloadAoS();
    const std::vector<double> b = f.bnd.downloadAoS();
    h.patches.resize(dm.patches.size());
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        PatchFieldSpec& s = h.patches[pi];
        const std::size_t n = std::size_t(dm.fvPatchSize[pi]);
        const Patch& mp = dm.patches[pi];
        s.type =
            mp.isEmpty() ? "empty"
          : (mp.type == "symmetryPlane" || mp.type == "symmetry") ? mp.type
          : "calculated";
        s.hasValue = true;
        s.value = slice(b, std::size_t(dm.fvPatchStart[pi]), n, 1);
    }
    return h;
}


// 顯式實體化
#define GNC_BC_INSTANTIATE(T)                                                 \
    template VolField<T> uploadVolField<T>(const DeviceMesh&, const HostField&); \
    template VolField<T> makeVolField<T>(const DeviceMesh&, const std::string&, Bc); \
    template void evaluateBCs<T>(const DeviceMesh&, VolField<T>&, const SurfField<double>*, bool); \
    template void finishAssignment<T>(const DeviceMesh&, VolField<T>&);       \
    template void updateInletOutletCoeffs<T>(const DeviceMesh&, VolField<T>&, const SurfField<double>&); \
    template void snGradBCs<T>(const DeviceMesh&, const VolField<T>&, DField<T>&); \
    template HostField downloadVolField<T>                                    \
    (const DeviceMesh&, const VolField<T>&, const std::string&, const std::string&);

GNC_BC_INSTANTIATE(double)
GNC_BC_INSTANTIATE(vec3)
GNC_BC_INSTANTIATE(tens)

#undef GNC_BC_INSTANTIATE

} // namespace gnc
