// 兩個 OpenFOAM 場檔的逐位元比對（獨立求解器寫出的場對參考寫出的場）。
// GPU-Native-CFD，GPL-3.0-or-later。
//
// 以本專案的讀取器（io/fieldIO，ASCII 與二進位都可）讀兩個檔，比對內部值與
// 每個 patch 的 value（兩邊都有 value 時）。ASCII 的數值以 OpenFOAM 的規則
// 轉換，所以比較的是 OpenFOAM 讀到的 double。
// 結束碼：0 全部相同，1 有不同，2 錯誤。
//
// 用法：gncFieldDiff POLYMESH_DIR REF TEST

#include "io/fieldIO.hpp"
#include "io/polyMeshReader.hpp"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace
{

struct Diff
{
    std::size_t n = 0;
    std::size_t mismatch = 0;
    std::size_t first = 0;
    double a = 0, b = 0;
};

Diff compare(const std::vector<double>& a, const std::vector<double>& b)
{
    Diff d;
    d.n = a.size();
    if (a.size() != b.size())
    {
        d.mismatch = std::max(a.size(), b.size());
        return d;
    }
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        if (std::memcmp(&a[i], &b[i], sizeof(double)) != 0)
        {
            if (d.mismatch == 0)
            {
                d.first = i;
                d.a = a[i];
                d.b = b[i];
            }
            ++d.mismatch;
        }
    }
    return d;
}

bool report(const std::string& what, const Diff& d)
{
    if (d.mismatch == 0)
    {
        std::printf("%-32s %10zu  BITWISE\n", what.c_str(), d.n);
        return true;
    }
    std::printf
    (
        "%-32s %10zu  DIFF %zu (first %zu: ref %.17g test %.17g)\n",
        what.c_str(), d.n, d.mismatch, d.first, d.a, d.b
    );
    return false;
}

} // namespace


int main(int argc, char* argv[])
{
    if (argc != 4)
    {
        std::fprintf(stderr, "usage: gncFieldDiff POLYMESH_DIR REF TEST\n");
        return 2;
    }
    try
    {
        const gnc::PolyMesh mesh = gnc::readPolyMesh(argv[1]);
        const gnc::HostField a = gnc::readField(argv[2], mesh);
        const gnc::HostField b = gnc::readField(argv[3], mesh);
        std::printf("ref:  %s\ntest: %s\n", argv[2], argv[3]);
        if (a.cls != b.cls || a.nComp != b.nComp)
        {
            std::printf("class differs: %s vs %s\n", a.cls.c_str(), b.cls.c_str());
            return 1;
        }
        bool ok = report("internalField", compare(a.internal, b.internal));
        for (std::size_t pi = 0; pi < mesh.patches.size(); ++pi)
        {
            const gnc::PatchFieldSpec& pa = a.patches[pi];
            const gnc::PatchFieldSpec& pb = b.patches[pi];
            const std::string name = "patch " + mesh.patches[pi].name;
            if (!pa.hasValue || !pb.hasValue)
            {
                std::printf("%-32s %10s  (no value: %s / %s)\n", name.c_str(), "-",
                            pa.type.c_str(), pb.type.c_str());
                continue;
            }
            ok = report(name, compare(pa.value, pb.value)) && ok;
        }
        std::printf("result: %s\n", ok ? "ALL BITWISE IDENTICAL" : "DIFFERENCES");
        return ok ? 0 : 1;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 2;
    }
}
