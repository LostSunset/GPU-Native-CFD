// GPU 的 pow、exp 與主機 libm（glibc）逐位元比較。GPU-Native-CFD，
// GPL-3.0-or-later。
//
// Spalart–Allmaras 的 fw = g*pow((1 + Cw3^6)/(g^6 + Cw3^6), 1/6) 是 SA 中唯一的
// 超越函數（N-016）。底數的範圍是 (0, (1 + Cw3^6)/Cw3^6]（Cw3 = 2 時為 65/64）。
// 兩種取樣：
//   sa（預設）  x 在 (0, xmax]：一半等距，一半依位元型樣遞增；y 固定（預設 1/6）
//   wide        x 為隨機位元型樣的正數（含次正規、Inf、NaN 之外的全範圍）、
//               y 在 [-8, 8] 隨機（splitmix64，種子固定），會走到溢位、下溢與
//               指數的特殊路徑
// --func exp（nutUSpaldingWallFunction 的 exp(kUu)，kUu ≤ 50）：
//   sa    x 在 [0, 50]：一半等距，一半依位元型樣遞增
//   wide  x 為隨機位元型樣的有限數（正負、含次正規與溢位、下溢範圍）
// --func tanh（kOmegaSST 的 F1、F2：tanh(pow4(arg1))、tanh(sqr(arg2))，引數 ≥ 0）：
//   sa    x 在 [0, 25]（≥ 22 時為 1），正負各半
//   wide  同 exp
// --func expm1（tanh 內部呼叫的 __expm1_fma）：sa 為 [-50, 50]，wide 同 exp
// --func log（nutkWallFunction 的 log(E*yPlus)、kLowReWallFunction 的 log(yPlus)）：
//   sa    x 在 (0, 1e7]：一半等距，一半依位元型樣遞增（含 1 附近的特殊路徑）
//   wide  x 為隨機位元型樣的正數（含次正規），另有少數 0、負數、Inf
// --impl cuda 用 CUDA 的 ::pow／::exp／::tanh／::expm1／::log，--impl glibc 用
// core/glibcPow.cuh、glibcExp.cuh、glibcTanh.cuh、glibcLog.cuh。報告不同的個數與
// 最大 ULP 距離。結束碼：0 全部相同，1 有不同。
//
// 用法：gncPowCheck [--func pow|exp|tanh|expm1|log] [--impl cuda|glibc]
//                   [--mode sa|wide] [--n N] [--xmax X] [--y Y]

#include "core/gncCuda.cuh"
#include "core/glibcExp.cuh"
#include "core/glibcLog.cuh"
#include "core/glibcPow.cuh"
#include "core/glibcTanh.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace
{

__global__ void kPow
(
    const std::size_t n,
    const bool glibc,
    const double* __restrict__ x,
    const double* __restrict__ y,
    double* __restrict__ out
)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = glibc ? gnc::glibcPow(x[i], y[i]) : ::pow(x[i], y[i]);
    }
}

enum class Func { pow, exp, tanh, expm1, log };

__global__ void kUnary
(
    const std::size_t n,
    const Func func,
    const bool glibc,
    const double* __restrict__ x,
    double* __restrict__ out
)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const double v = x[i];
    switch (func)
    {
        case Func::exp: out[i] = glibc ? gnc::glibcExp(v) : ::exp(v); break;
        case Func::tanh: out[i] = glibc ? gnc::glibcTanh(v) : ::tanh(v); break;
        case Func::expm1: out[i] = glibc ? gnc::glibcExpm1(v) : ::expm1(v); break;
        default: out[i] = glibc ? gnc::glibcLog(v) : ::log(v); break;
    }
}

double hostRef(const Func func, const double x, const double y)
{
    switch (func)
    {
        case Func::pow: return std::pow(x, y);
        case Func::exp: return std::exp(x);
        case Func::tanh: return std::tanh(x);
        case Func::expm1: return std::expm1(x);
        default: return std::log(x);
    }
}

std::int64_t ordered(const double d)
{
    std::int64_t i;
    std::memcpy(&i, &d, 8);
    return i < 0 ? std::int64_t(std::uint64_t(INT64_MIN) - std::uint64_t(i)) : i;
}

std::uint64_t splitmix64(std::uint64_t& s)
{
    std::uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
    z = (z ^ (z >> 30))*0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27))*0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

} // namespace


int main(int argc, char* argv[])
{
    std::size_t n = 100000000;
    double xmax = 65.0/64.0;
    double yFixed = 1.0/6.0;
    std::string impl = "cuda", mode = "sa", func = "pow";
    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string a = argv[i];
        if (a == "--n") n = std::stoull(argv[i + 1]);
        else if (a == "--xmax") xmax = std::stod(argv[i + 1]);
        else if (a == "--y") yFixed = std::stod(argv[i + 1]);
        else if (a == "--impl") impl = argv[i + 1];
        else if (a == "--mode") mode = argv[i + 1];
        else if (a == "--func") func = argv[i + 1];
        else
        {
            std::fprintf(stderr, "usage: gncPowCheck [--func pow|exp|tanh|expm1|log] [--impl cuda|glibc] [--mode sa|wide] [--n N] [--xmax X] [--y Y]\n");
            return 2;
        }
    }
    Func fn;
    if (func == "pow") fn = Func::pow;
    else if (func == "exp") fn = Func::exp;
    else if (func == "tanh") fn = Func::tanh;
    else if (func == "expm1") fn = Func::expm1;
    else if (func == "log") fn = Func::log;
    else
    {
        std::fprintf(stderr, "bad --func\n");
        return 2;
    }
    if ((impl != "cuda" && impl != "glibc") || (mode != "sa" && mode != "wide"))
    {
        std::fprintf(stderr, "bad --impl or --mode\n");
        return 2;
    }
    try
    {
        const std::size_t chunk = std::size_t(1) << 24;
        gnc::DeviceArray<double> dx(chunk), dy(chunk), dout(chunk);
        std::vector<double> hx, hy;
        std::size_t mismatches = 0;
        std::uint64_t maxUlp = 0;
        double worstX = 0, worstY = 0;
        if (xmax == 65.0/64.0)
        {
            switch (fn)
            {
                case Func::exp: case Func::expm1: xmax = 50.0; break;
                case Func::tanh: xmax = 25.0; break;
                case Func::log: xmax = 1e7; break;
                default: break;
            }
        }
        const std::int64_t lo = ordered(fn == Func::pow ? 1.0/128.0 : 0x1p-60), hi = ordered(xmax);
        std::uint64_t seed = 0x5eedULL;
        for (std::size_t done = 0; done < n; done += chunk)
        {
            const std::size_t m = std::min(chunk, n - done);
            hx.resize(m);
            hy.resize(m);
            for (std::size_t i = 0; i < m; ++i)
            {
                const std::size_t k = done + i;
                if (mode == "sa")
                {
                    if (k % 2 == 0)
                    {
                        hx[i] = xmax*(double(k/2) + 0.5)/double(n/2 + 1);
                    }
                    else
                    {
                        const std::int64_t bits =
                            lo + std::int64_t((double(k/2)/double(n/2 + 1))*double(hi - lo));
                        std::memcpy(&hx[i], &bits, 8);
                    }
                    hy[i] = yFixed;
                    // tanh、expm1 的負引數：每 4 個取 2 個變號
                    if ((fn == Func::tanh || fn == Func::expm1) && (k % 4) >= 2)
                    {
                        hx[i] = -hx[i];
                    }
                }
                else if (fn == Func::pow)
                {
                    // 正的有限數（不含 0 以外的特殊值）
                    const std::uint64_t bx = splitmix64(seed) % 0x7ff0000000000000ULL;
                    std::memcpy(&hx[i], &bx, 8);
                    hy[i] = -8.0 + 16.0*(double(splitmix64(seed) >> 11)*0x1p-53);
                }
                else if (fn == Func::log)
                {
                    // 正的有限數（含次正規）；每 64 個中有 0、負數、Inf 各一
                    std::uint64_t bx = splitmix64(seed) % 0x7ff0000000000000ULL;
                    if (k % 64 == 1) bx = 0;
                    else if (k % 64 == 2) bx |= 0x8000000000000000ULL;
                    else if (k % 64 == 3) bx = 0x7ff0000000000000ULL;
                    std::memcpy(&hx[i], &bx, 8);
                    hy[i] = 0.0;
                }
                else
                {
                    // 有限數，正負；一半取 |x| < 1024 的範圍（含溢位與下溢的邊界）
                    std::uint64_t bx = splitmix64(seed);
                    const std::uint64_t sign = bx & 0x8000000000000000ULL;
                    if (k % 2 == 0)
                    {
                        bx = (bx & 0x000fffffffffffffULL) | (std::uint64_t(0x3c0 + (splitmix64(seed) % 0x4a)) << 52);
                    }
                    else
                    {
                        bx = (bx & 0x7fffffffffffffffULL) % 0x7ff0000000000000ULL;
                    }
                    bx |= sign;
                    std::memcpy(&hx[i], &bx, 8);
                    hy[i] = 0.0;
                }
            }
            dx.upload(hx);
            dy.upload(hy);
            if (fn == Func::pow)
            {
                kPow<<<unsigned((m + 255)/256), 256>>>(m, impl == "glibc", dx.data(), dy.data(), dout.data());
            }
            else
            {
                kUnary<<<unsigned((m + 255)/256), 256>>>(m, fn, impl == "glibc", dx.data(), dout.data());
            }
            GNC_CUDA_LAUNCH_CHECK();
            std::vector<double> gy = dout.download();
            for (std::size_t i = 0; i < m; ++i)
            {
                const double ref = hostRef(fn, hx[i], hy[i]);
                // NaN 的位元型樣（x86 的預設 NaN 與 CUDA 的不同）不算差異
                if (std::memcmp(&ref, &gy[i], 8) != 0 && !(ref != ref && gy[i] != gy[i]))
                {
                    ++mismatches;
                    const std::int64_t a = ordered(ref), b = ordered(gy[i]);
                    const std::uint64_t u = a > b ? std::uint64_t(a - b) : std::uint64_t(b - a);
                    if (u > maxUlp || mismatches == 1)
                    {
                        maxUlp = u > maxUlp ? u : maxUlp;
                        worstX = hx[i];
                        worstY = hy[i];
                    }
                }
            }
        }
        std::printf
        (
            "func %s impl %s mode %s: %zu samples, %zu differ from host libm (%.3e), max %llu ulp (x = %.17g, y = %.17g)\n",
            func.c_str(), impl.c_str(), mode.c_str(), n, mismatches, double(mismatches)/double(n),
            (unsigned long long)maxUlp, worstX, worstY
        );
        return mismatches ? 1 : 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 2;
    }
}
