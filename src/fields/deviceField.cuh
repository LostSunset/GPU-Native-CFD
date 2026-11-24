// 裝置上的場。GPU-Native-CFD，GPL-3.0-or-later。
//
// 每個分量一個陣列（SoA）。體積場：nCells 個格值，加上有限體積邊界面
// （fvBoundaryFaces，依 patch 串接，empty 為 0 個）的邊界值。面場：
// nInternalFaces 個內部面值，加上同樣的邊界值。邊界條件以 patch 為單位
// 記錄型別與參數（boundaryConditions.cu）。

#ifndef GNC_FIELDS_DEVICEFIELD_CUH
#define GNC_FIELDS_DEVICEFIELD_CUH

#include "core/arrayOps.cuh"
#include "core/foamOps.cuh"
#include "core/gncCuda.cuh"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace gnc
{

template<class T>
struct DView
{
    double* p[NComp<T>::value];

    __device__ __forceinline__ T operator[](const std::int32_t i) const
    {
        T v;
        for (int k = 0; k < NComp<T>::value; ++k)
        {
            setCmpt(v, k, p[k][i]);
        }
        return v;
    }

    __device__ __forceinline__ void set(const std::int32_t i, const T& v) const
    {
        for (int k = 0; k < NComp<T>::value; ++k)
        {
            p[k][i] = cmpt(v, k);
        }
    }
};


// S 為儲存型別（預設 double）；float 時讀出後轉成 double（效能版的 FP32 Jacobian）
template<class T, class S = double>
struct CView
{
    const S* p[NComp<T>::value];

    __device__ __forceinline__ T operator[](const std::int32_t i) const
    {
        T v;
        for (int k = 0; k < NComp<T>::value; ++k)
        {
            setCmpt(v, k, double(p[k][i]));
        }
        return v;
    }
};


template<class T>
class DField
{
    static constexpr int N = NComp<T>::value;
    DeviceArray<double> c_[N];
    std::size_t n_ = 0;

public:
    DField() = default;

    explicit DField(const std::size_t n)
    {
        allocate(n);
    }

    void allocate(const std::size_t n)
    {
        n_ = n;
        for (auto& a : c_)
        {
            a.allocate(n);
        }
    }

    std::size_t size() const { return n_; }

    DeviceArray<double>& component(const int k) { return c_[k]; }
    const DeviceArray<double>& component(const int k) const { return c_[k]; }

    DView<T> view()
    {
        DView<T> v;
        for (int k = 0; k < N; ++k)
        {
            v.p[k] = c_[k].data();
        }
        return v;
    }

    CView<T> cview() const
    {
        CView<T> v;
        for (int k = 0; k < N; ++k)
        {
            v.p[k] = c_[k].data();
        }
        return v;
    }

    // 各分量歸零（+0.0）；batch 給定時加進呼叫端的批次（N-028），否則自己一次啟動
    void zero(ArrayBatch* batch = nullptr)
    {
        ArrayBatch own;
        ArrayBatch& b = batch ? *batch : own;
        for (auto& a : c_)
        {
            b.fill(a.data(), a.size(), 0.0);
        }
    }

    void uploadAoS(const std::vector<double>& aos)
    {
        const std::size_t n = aos.size()/N;
        allocate(n);
        std::vector<double> h(n);
        for (int k = 0; k < N; ++k)
        {
            for (std::size_t i = 0; i < n; ++i)
            {
                h[i] = aos[i*N + k];
            }
            c_[k].upload(h);
        }
    }

    std::vector<double> downloadAoS() const
    {
        std::vector<double> out(n_*N);
        for (int k = 0; k < N; ++k)
        {
            const std::vector<double> h = c_[k].download();
            for (std::size_t i = 0; i < n_; ++i)
            {
                out[i*N + k] = h[i];
            }
        }
        return out;
    }

    // 複製各分量（一次啟動，或加進呼叫端的批次；N-028）
    void copyFrom(const DField& o, ArrayBatch* batch = nullptr)
    {
        allocate(o.n_);
        ArrayBatch own;
        ArrayBatch& b = batch ? *batch : own;
        for (int k = 0; k < N; ++k)
        {
            b.copy(c_[k].data(), o.c_[k].data(), n_);
        }
    }
};


// 邊界條件型別（本專案支援的；其他型別讀取時報錯）
enum class Bc
{
    calculated,
    extrapolatedCalculated,
    fixedValue,
    zeroGradient,
    empty,
    symmetryPlane,
    symmetry,
    slip,
    mixed,
    inletOutlet,
    fixedGradient,
    // HiSA（N3）
    characteristicFarfieldPressure,     // mixed 族
    characteristicFarfieldTemperature,  // mixed 族
    characteristicFarfieldVelocity,     // mixed 族
    characteristicWallPressure,         // fixedGradient 族
    characteristicWallTemperature,      // fixedValue 族
    boundaryCorrectedFixedValue,        // fixedValue 族
    // 紊流黏度的壁面函數（N4a）：fixedValue 族，evaluate 時 updateCoeffs
    // 以 operator== 設為 calcNut()，低雷諾數版為 0
    nutLowReWallFunction,
    boundaryCorrectedNutLowReWallFunction,  // HiSA：snGrad 與 gradient 係數用 nonOrthDeltaCoeffs
    nutUSpaldingWallFunction,               // calcNut 由紊流模型算（需要 U、nu、y）
    compressibleAlphatWallFunction,         // compressible::alphatWallFunction：rho_w*nut_w/Prt
    // kOmegaSST（N4b）。壁面函數為 fixedValue 族，值由紊流模型在 updateCoeffs 的
    // 時點算好（evaluate 不改值）；兩個入口為 inletOutlet 族，refValue 由紊流模型
    // 在 updateCoeffs 的時點算好
    kLowReWallFunction,                     // k：Ck/kappa*log(y+) + Bk 或低雷諾數式，乘 uTau^2
    omegaWallFunction,                      // omega：改寫牆邊格的 omega 與 G，矩陣 setValues
    nutkWallFunction,                       // nut：calcNut（stepwise 等 blending）
    turbulentIntensityKineticEnergyInlet,   // k：refValue = 1.5*sqr(I)*magSqr(U_b)
    turbulentMixingLengthFrequencyInlet,    // omega：refValue = sqrt(k_b)/(Cmu25*L)
    // 能量場（thermo 依 T 的邊界條件建立，basicThermo::heBoundaryTypes）
    fixedEnergy,                        // fixedValue 族
    gradientEnergy,                     // fixedGradient 族
    mixedEnergy,                        // mixed 族
    // rhoSimpleFoam（N6）
    freestreamVelocity,                 // mixed 族：refValue = freestreamValue、refGrad = 0；
                                        // updateCoeffs 時 valueFraction = 0.5 - 0.5*(U_b & nf)/mag(U_b)
    freestreamPressure,                 // mixed 族：valueFraction = 0.5 + 0.5*(U_b & nf)/mag(U_b)
    noSlip,                             // fixedValue 族，值為 0
    kqRWallFunction                     // zeroGradient（kqRWallFunctionFvPatchField 繼承 zeroGradient）
};

const char* bcName(Bc b);

// 型別族（isA<...> 的判斷）
bool isMixedFamily(Bc b);
bool isFixedValueFamily(Bc b);
bool isFixedGradientFamily(Bc b);

// inletOutletFvPatchField 與其衍生型別（valueFraction = 1 - pos0(phi_b)、refGrad = 0）
bool isInletOutletFamily(Bc b);

// zeroGradientFvPatchField 與其衍生型別（kqRWallFunction）
bool isZeroGradientFamily(Bc b);

// fvPatchField::fixesValue()：calculated、fixedValue 族、mixed 族為真
bool fixesValue(Bc b);


// BlockCoupledBoundary 對 U 的係數型別 outerProduct<T, vector>（張量場不用，
// 只為了讓 PatchBC<tens> 能實體化）
template<class T> struct UCoeffOf { using type = typename GradOf<T>::type; };
template<> struct UCoeffOf<tens> { using type = tens; };


template<class T>
struct PatchBC
{
    Bc type = Bc::calculated;
    std::string phiName = "phi";
    // mixed 族
    DField<T> refValue;
    DField<T> refGrad;
    DeviceArray<double> valueFraction;
    // fixedGradient 族
    DField<T> gradient;
    // 特徵類邊界條件的參考值（characteristicBase：U、p、T）
    vec3 URef{0.0, 0.0, 0.0};
    double pRef = 0.0;
    double TRef = 0.0;
    // 建構時取相鄰格值（沒有 value 的特徵類、非續算的 characteristicWallPressure）
    bool pifAtConstruction = false;
    // 壁面函數的係數（wallFunctionCoefficients：kappa、E；nutUSpalding 的 maxIter、
    // tolerance；alphatWallFunction 的 Prt）
    double wallKappa = 0.41;
    double wallE = 9.8;
    int wallMaxIter = 10;
    double wallTolerance = 0.01;
    double wallPrt = 0.85;
    // kOmegaSST 的壁面函數與入口（N4b）：wallFunctionCoefficients 的 Cmu；
    // wallFunctionBlenders 的 blending（0 stepwise、1 max、2 binomial、3 exponential、
    // 4 tanh）與 n；omegaWallFunction 的 beta1；kLowReWallFunction 的 Ceps2、Ck、Bk、C；
    // 入口的 intensity、mixingLength。yPlusLam 由 kappa、E 在主機端算
    double wallCmu = 0.09;
    int wallBlending = 0;
    double wallBlendN = 4.0;
    double wallBeta1 = 0.075;
    double wallCeps2 = 1.9;
    double wallCk = -0.416;
    double wallBk = 8.366;
    double wallC = 11.0;
    double yPlusLam = 0.0;
    double intensity = 0.0;
    double mixingLength = 0.0;
    // 寫出時照抄的字典項（上面這些型別的係數，讀進來的原文）
    std::vector<std::pair<std::string, std::string>> writeExtras;
    // HiSA BlockCoupledBoundary 的係數（updateCoeffs 設定，N3c）：本場邊界值
    // 對內部格 p、U、T 的 valueInternalCoeffs。characteristicWallPressure 的
    // UCoeff 存 UCoeff_*nf（它的 valueInternalCoeffs(U)）
    DField<T> pCoeff;
    DField<typename UCoeffOf<T>::type> UCoeff;
    DField<T> TCoeff;
};


template<class T>
struct VolField
{
    std::string name;
    DField<T> in;       // nCells
    DField<T> bnd;      // nFvBoundaryFaces
    std::vector<PatchBC<T>> bc;     // 依 mesh 的 patch 順序
};


template<class T>
struct SurfField
{
    std::string name;
    DField<T> in;       // nInternalFaces
    DField<T> bnd;      // nFvBoundaryFaces
};

} // namespace gnc

#endif
