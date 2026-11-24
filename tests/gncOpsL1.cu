// L1 驗證的 GPU 端：場、邊界條件與運算子（design.md §6）。
// GPU-Native-CFD，GPL-3.0-or-later。
//
// 讀 tools/ref/gncOpsRef -generate 寫出的測試場（s、v、gamma、phi），在 GPU
// 上 evaluate 邊界條件並算各運算子，取回後以 GNC 陣列檔寫出；紀錄名稱與
// gncOpsRef 相同，由 gncCompare 逐位元比對。另外：
//   - 同一組場的 ASCII 與二進位版本各讀一次，主機上逐位元比對（讀取器）
//   - 把 evaluate 後的 s、v 與 phi 以 OpenFOAM 格式寫到 --write-dir，
//     由 gncOpsRef -dump-fields 讀回比對（OpenFOAM 能讀本專案的輸出）
//
// 用法：gncOpsL1 --case DIR --fields SUB --bin-fields SUB --schemes FILE
//                --out FILE --write-dir SUB

#include "core/arrayFile.hpp"
#include "core/gncCuda.cuh"
#include "fields/boundaryConditions.cuh"
#include "io/dictionary.hpp"
#include "io/fieldIO.hpp"
#include "io/foamTokenizer.hpp"
#include "io/polyMeshReader.hpp"
#include "mesh/cellFaceGather.hpp"
#include "mesh/deviceMesh.cuh"
#include "ops/fvcOps.cuh"

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

namespace af = gnc::arrayFile;
using gnc::vec3;
using gnc::tens;

template<class T>
void dumpCells(af::Writer& w, const std::string& name, const gnc::DField<T>& f)
{
    const std::vector<double> h = f.downloadAoS();
    w.f64(name, h.data(), f.size(), gnc::NComp<T>::value);
}

template<class T>
void dumpVol(af::Writer& w, const std::string& name, const gnc::VolField<T>& f)
{
    dumpCells(w, name + ".i", f.in);
    dumpCells(w, name + ".b", f.bnd);
}

template<class T>
void dumpSurf(af::Writer& w, const std::string& name, const gnc::SurfField<T>& f)
{
    dumpCells(w, name + ".i", f.in);
    dumpCells(w, name + ".b", f.bnd);
}

// 兩個 HostField 逐位元比對（內部值與每個 patch 的各項）；回傳不同的值數
std::size_t compareHost(const gnc::HostField& a, const gnc::HostField& b)
{
    std::size_t diff = 0;
    const auto cmp = [&](const std::vector<double>& x, const std::vector<double>& y)
    {
        if (x.size() != y.size())
        {
            diff += std::max(x.size(), y.size());
            return;
        }
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            if (std::memcmp(&x[i], &y[i], 8) != 0)
            {
                ++diff;
            }
        }
    };
    cmp(a.internal, b.internal);
    for (std::size_t p = 0; p < a.patches.size(); ++p)
    {
        if (a.patches[p].type != b.patches[p].type)
        {
            ++diff;
        }
        cmp(a.patches[p].value, b.patches[p].value);
        cmp(a.patches[p].refValue, b.patches[p].refValue);
        cmp(a.patches[p].refGrad, b.patches[p].refGrad);
        cmp(a.patches[p].valueFraction, b.patches[p].valueFraction);
    }
    return diff;
}

// fvSchemes（已展開）的 gradSchemes：先找 grad(name)，再找 default
std::string gradSchemeFor(const gnc::Dictionary& schemes, const std::string& field)
{
    const gnc::Dictionary& g = schemes.subDict("gradSchemes");
    const gnc::Dictionary::Entry* e = g.find("grad(" + field + ")");
    if (!e)
    {
        e = g.find("default");
    }
    if (!e)
    {
        throw std::runtime_error("gradSchemes has no entry for grad(" + field + ")");
    }
    std::string s;
    for (const gnc::Token& t : e->stream)
    {
        s += (s.empty() ? "" : " ") + t.text;
    }
    if (s != "faceLeastSquares linear" && s != "Gauss linear")
    {
        throw std::runtime_error("grad scheme '" + s + "' is not supported");
    }
    return s;
}

} // namespace


int main(int argc, char* argv[])
{
    std::string caseDir, fieldsSub = "gncL1", binSub = "gncL1bin";
    std::string schemesFile, outFile, writeSub = "gncL1out";
    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string a = argv[i];
        const std::string v = argv[i + 1];
        if (a == "--case") caseDir = v;
        else if (a == "--fields") fieldsSub = v;
        else if (a == "--bin-fields") binSub = v;
        else if (a == "--schemes") schemesFile = v;
        else if (a == "--out") outFile = v;
        else if (a == "--write-dir") writeSub = v;
        else
        {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (caseDir.empty() || schemesFile.empty() || outFile.empty())
    {
        std::fprintf
        (
            stderr,
            "usage: gncOpsL1 --case DIR --schemes FILE --out FILE"
            " [--fields SUB] [--bin-fields SUB] [--write-dir SUB]\n"
        );
        return 2;
    }

    try
    {
        int dev = 0;
        GNC_CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};
        GNC_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        char pci[32] = {};
        GNC_CUDA_CHECK(cudaDeviceGetPCIBusId(pci, sizeof(pci), dev));
        std::printf("device: %s (sm_%d%d, PCI %s)\n", prop.name, prop.major, prop.minor, pci);

        const gnc::PolyMesh mesh = gnc::readPolyMesh(caseDir + "/constant/polyMesh");
        const gnc::CellFaceGather gather = gnc::buildCellFaceGather(mesh);
        gnc::DeviceMesh dm = gnc::DeviceMesh::upload(mesh, gather);
        gnc::computeGeometry(dm);
        std::printf
        (
            "mesh: nCells %d nFaces %d nInternalFaces %d fvBoundaryFaces %d"
            " nSolutionD %d\n",
            mesh.nCells, mesh.nFaces, mesh.nInternalFaces, dm.nFvBoundaryFaces(),
            dm.nSolutionD
        );

        // 場：ASCII 與二進位各讀一次
        const std::string fd = caseDir + "/" + fieldsSub + "/";
        const std::string bd = caseDir + "/" + binSub + "/";
        std::size_t readerDiff = 0;
        gnc::HostField hs, hv, hg, hphi;
        for (const char* nm : {"s", "v", "gamma", "phi"})
        {
            const gnc::HostField a = gnc::readField(fd + nm, mesh);
            const gnc::HostField b = gnc::readField(bd + nm, mesh);
            if (a.format != "ascii" || b.format != "binary")
            {
                throw std::runtime_error(std::string("field ") + nm + ": unexpected formats");
            }
            const std::size_t d = compareHost(a, b);
            std::printf("reader: %s ascii vs binary: %zu values differ\n", nm, d);
            readerDiff += d;
            const std::string n(nm);
            if (n == "s") hs = a;
            else if (n == "v") hv = a;
            else if (n == "gamma") hg = a;
            else hphi = a;
        }

        gnc::FoamTokenizer st(schemesFile);
        const gnc::Dictionary schemes = gnc::Dictionary::parse(st, "fvSchemes", false);
        const std::string schemeS = gradSchemeFor(schemes, "s");
        const std::string schemeV = gradSchemeFor(schemes, "v");
        std::printf("grad schemes: s '%s', v '%s'\n", schemeS.c_str(), schemeV.c_str());

        gnc::VolField<double> s = gnc::uploadVolField<double>(dm, hs);
        gnc::VolField<vec3> v = gnc::uploadVolField<vec3>(dm, hv);
        gnc::VolField<double> gamma = gnc::uploadVolField<double>(dm, hg);
        gnc::SurfField<double> phi = gnc::uploadSurfaceScalarField(dm, hphi);

        std::printf("boundary conditions:");
        for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
        {
            std::printf
            (
                " %s[s %s, v %s, gamma %s]", dm.patches[pi].name.c_str(),
                gnc::bcName(s.bc[pi].type), gnc::bcName(v.bc[pi].type),
                gnc::bcName(gamma.bc[pi].type)
            );
        }
        std::printf("\n");

        gnc::evaluateBCs(dm, s, &phi);
        gnc::evaluateBCs(dm, v, &phi);
        gnc::evaluateBCs(dm, gamma, &phi);

        af::Writer w(outFile);
        w.text("producer", "gncOpsL1");
        w.text("device", std::string(prop.name) + " " + pci);

        dumpVol(w, "s", s);
        dumpVol(w, "v", v);
        dumpVol(w, "gamma", gamma);
        dumpSurf(w, "phi", phi);

        gnc::DField<double> snS;
        gnc::DField<vec3> snV;
        gnc::snGradBCs(dm, s, snS);
        gnc::snGradBCs(dm, v, snV);
        dumpCells(w, "snGradBC.s", snS);
        dumpCells(w, "snGradBC.v", snV);

        gnc::SurfField<double> iS;
        gnc::SurfField<vec3> iV;
        gnc::interpolateLinear(dm, s, iS);
        gnc::interpolateLinear(dm, v, iV);
        dumpSurf(w, "interp.s", iS);
        dumpSurf(w, "interp.v", iV);

        gnc::DField<double> divPhi;
        gnc::surfaceIntegrate(dm, phi, divPhi);
        dumpCells(w, "div.phi", divPhi);
        gnc::SurfField<vec3> phiv;
        gnc::multiplySurf(phi, iV, phiv);
        gnc::DField<vec3> divPhiv;
        gnc::surfaceIntegrate(dm, phiv, divPhiv);
        dumpCells(w, "div.phiv", divPhiv);

        gnc::VolField<vec3> ggS;
        gnc::VolField<tens> ggV;
        gnc::gaussGrad(dm, s, ggS);
        gnc::gaussGrad(dm, v, ggV);
        dumpVol(w, "gaussGrad.s", ggS);
        dumpVol(w, "gaussGrad.v", ggV);

        const gnc::LSVectors ls = gnc::faceLeastSquaresVectors(dm);
        dumpSurf(w, "ls.P", ls.P);
        dumpCells(w, "ls.N.i", ls.N);

        gnc::VolField<vec3> lsS;
        gnc::VolField<tens> lsV;
        gnc::faceLeastSquaresGrad(dm, ls, s, lsS);
        gnc::faceLeastSquaresGrad(dm, ls, v, lsV);
        dumpVol(w, "lsGrad.s", lsS);
        dumpVol(w, "lsGrad.v", lsV);

        // compact snGrad 與拉普拉斯用 fvSchemes 指定的梯度
        gnc::VolField<vec3> gradS;
        gnc::VolField<tens> gradV;
        gnc::gradByScheme(dm, ls, schemeS, s, gradS);
        gnc::gradByScheme(dm, ls, schemeV, v, gradV);

        gnc::SurfField<double> snCS;
        gnc::SurfField<vec3> snCV;
        gnc::snGradCompact(dm, s, gradS, snCS);
        gnc::snGradCompact(dm, v, gradV, snCV);
        dumpSurf(w, "snGradCompact.s", snCS);
        dumpSurf(w, "snGradCompact.v", snCV);

        gnc::DField<double> lapS;
        gnc::DField<vec3> lapV;
        gnc::laplacianCompact(dm, gamma, s, gradS, lapS);
        gnc::laplacianCompact(dm, gamma, v, gradV, lapV);
        dumpCells(w, "laplacian.s", lapS);
        dumpCells(w, "laplacian.v", lapV);

        gnc::DField<vec3> divT;
        gnc::divGaussLinear(dm, lsV, divT);
        dumpCells(w, "div.lsGradV", divT);

        w.close();
        GNC_CUDA_CHECK(cudaDeviceSynchronize());
        std::printf("wrote %s\n", outFile.c_str());

        // 寫回 OpenFOAM 格式
        const std::string wd = caseDir + "/" + writeSub;
        std::filesystem::create_directories(wd);
        gnc::writeField
        (
            wd + "/s", gnc::downloadVolField(dm, s, "volScalarField", hs.dimensions),
            mesh, writeSub
        );
        gnc::writeField
        (
            wd + "/v", gnc::downloadVolField(dm, v, "volVectorField", hv.dimensions),
            mesh, writeSub
        );
        gnc::writeField
        (
            wd + "/phi", gnc::downloadSurfaceScalarField(dm, phi, hphi.dimensions),
            mesh, writeSub
        );
        std::printf("wrote fields to %s\n", wd.c_str());

        return readerDiff ? 1 : 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 2;
    }
}
