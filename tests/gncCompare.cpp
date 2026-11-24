// 兩個 GNC 陣列檔的逐位元比對。GPU-Native-CFD，GPL-3.0-or-later。
//
// 以 REF 的紀錄為準，依名稱在 TEST 中尋找同名紀錄：
//   - 數值紀錄（float64、int32）逐位元比對；float64 另報不同的個數、
//     最大絕對差、最大相對差、最大 ULP 距離與第一個不同的位置。
//   - 文字紀錄只列出，不算失敗。
//   - 只在其中一個檔案出現的紀錄列出；REF 有而 TEST 沒有的數值紀錄算失敗。
// 結束碼：0 全部逐位元相同，1 有不同或缺少，2 錯誤。
//
// 用法：gncCompare REF TEST

#include "core/arrayFile.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>

namespace
{

namespace af = gnc::arrayFile;

// IEEE double 的位元轉成單調整數，用來算 ULP 距離
std::int64_t ordered(const double d)
{
    std::int64_t i;
    std::memcpy(&i, &d, 8);
    return i < 0 ? std::int64_t(std::uint64_t(INT64_MIN) - std::uint64_t(i)) : i;
}

std::uint64_t ulpDistance(const double a, const double b)
{
    const std::int64_t ia = ordered(a);
    const std::int64_t ib = ordered(b);
    return ia > ib ? std::uint64_t(ia) - std::uint64_t(ib)
                   : std::uint64_t(ib) - std::uint64_t(ia);
}

const char* dtypeName(const std::uint32_t t)
{
    switch (t)
    {
        case af::F64: return "f64";
        case af::I32: return "i32";
        case af::TEXT: return "text";
        default: return "?";
    }
}

} // namespace


int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::fprintf(stderr, "usage: gncCompare REF TEST\n");
        return 2;
    }

    try
    {
        const std::vector<af::Record> ref = af::readAll(argv[1]);
        const std::vector<af::Record> test = af::readAll(argv[2]);

        std::map<std::string, const af::Record*> testByName;
        for (const af::Record& r : test)
        {
            testByName[r.name] = &r;
        }

        std::printf("ref:  %s\ntest: %s\n", argv[1], argv[2]);
        std::printf
        (
            "%-30s %-4s %11s %4s %11s %12s %12s %12s  %s\n",
            "record", "type", "n", "comp", "mismatch", "maxAbs", "maxRel",
            "maxUlp", "status"
        );

        bool allEqual = true;
        std::map<std::string, bool> seen;

        for (const af::Record& r : ref)
        {
            seen[r.name] = true;
            const auto it = testByName.find(r.name);

            if (r.dtype == af::TEXT)
            {
                const std::string a(r.bytes.begin(), r.bytes.end());
                const std::string b = it == testByName.end()
                  ? std::string("(none)")
                  : std::string(it->second->bytes.begin(), it->second->bytes.end());
                std::printf("%-30s text  ref=\"%s\" test=\"%s\"\n",
                    r.name.c_str(), a.c_str(), b.c_str());
                continue;
            }

            if (it == testByName.end())
            {
                std::printf("%-30s %-4s %11llu %4u %11s %12s %12s %12s  MISSING\n",
                    r.name.c_str(), dtypeName(r.dtype),
                    (unsigned long long)r.nTuple, r.nComp, "-", "-", "-", "-");
                allEqual = false;
                continue;
            }
            const af::Record& t = *it->second;

            if (t.dtype != r.dtype || t.nComp != r.nComp || t.nTuple != r.nTuple)
            {
                std::printf
                (
                    "%-30s %-4s %11llu %4u  SHAPE ref(%s,%llu,%u) test(%s,%llu,%u)\n",
                    r.name.c_str(), dtypeName(r.dtype),
                    (unsigned long long)r.nTuple, r.nComp,
                    dtypeName(r.dtype), (unsigned long long)r.nTuple, r.nComp,
                    dtypeName(t.dtype), (unsigned long long)t.nTuple, t.nComp
                );
                allEqual = false;
                continue;
            }

            const std::size_t n = r.count();
            std::size_t mismatch = 0;
            std::size_t first = n;
            double maxAbs = 0;
            double maxRel = 0;
            std::uint64_t maxUlp = 0;

            if (r.dtype == af::F64)
            {
                const double* a = reinterpret_cast<const double*>(r.bytes.data());
                const double* b = reinterpret_cast<const double*>(t.bytes.data());
                for (std::size_t i = 0; i < n; ++i)
                {
                    if (std::memcmp(a + i, b + i, 8) != 0)
                    {
                        if (first == n)
                        {
                            first = i;
                        }
                        ++mismatch;
                        const double d = std::fabs(a[i] - b[i]);
                        const double s = std::fmax(std::fabs(a[i]), std::fabs(b[i]));
                        maxAbs = std::fmax(maxAbs, d);
                        if (s > 0)
                        {
                            maxRel = std::fmax(maxRel, d/s);
                        }
                        const std::uint64_t u = ulpDistance(a[i], b[i]);
                        maxUlp = u > maxUlp ? u : maxUlp;
                    }
                }
            }
            else
            {
                const std::int32_t* a = reinterpret_cast<const std::int32_t*>(r.bytes.data());
                const std::int32_t* b = reinterpret_cast<const std::int32_t*>(t.bytes.data());
                for (std::size_t i = 0; i < n; ++i)
                {
                    if (a[i] != b[i])
                    {
                        if (first == n)
                        {
                            first = i;
                        }
                        ++mismatch;
                    }
                }
            }

            std::printf
            (
                "%-30s %-4s %11llu %4u %11zu %12.3e %12.3e %12llu  %s\n",
                r.name.c_str(), dtypeName(r.dtype),
                (unsigned long long)r.nTuple, r.nComp, mismatch, maxAbs, maxRel,
                (unsigned long long)maxUlp, mismatch ? "DIFF" : "BITWISE"
            );
            if (mismatch)
            {
                allEqual = false;
                if (r.dtype == af::F64)
                {
                    const double* a = reinterpret_cast<const double*>(r.bytes.data());
                    const double* b = reinterpret_cast<const double*>(t.bytes.data());
                    std::printf("    first at value %zu (tuple %zu comp %zu): ref %.17g test %.17g\n",
                        first, first/r.nComp, first%r.nComp, a[first], b[first]);
                }
                else
                {
                    const std::int32_t* a = reinterpret_cast<const std::int32_t*>(r.bytes.data());
                    const std::int32_t* b = reinterpret_cast<const std::int32_t*>(t.bytes.data());
                    std::printf("    first at value %zu: ref %d test %d\n",
                        first, a[first], b[first]);
                }
            }
        }

        for (const af::Record& t : test)
        {
            if (!seen.count(t.name))
            {
                if (t.dtype == af::TEXT)
                {
                    std::printf("%-30s text  test only: \"%s\"\n", t.name.c_str(),
                        std::string(t.bytes.begin(), t.bytes.end()).c_str());
                }
                else
                {
                    std::printf("%-30s %-4s %11llu %4u  test only\n", t.name.c_str(),
                        dtypeName(t.dtype), (unsigned long long)t.nTuple, t.nComp);
                }
            }
        }

        std::printf("result: %s\n", allEqual ? "ALL BITWISE IDENTICAL" : "DIFFERENCES");
        return allEqual ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 2;
    }
}
