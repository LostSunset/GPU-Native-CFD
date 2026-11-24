// L0 驗證的 GPU 端（design.md §6）。GPU-Native-CFD，GPL-3.0-or-later。
//
// 讀 OpenFOAM 案例的 constant/polyMesh，在 GPU 上算全部幾何量，取回主機後
// 以 GNC 陣列檔寫出；紀錄名稱與 tools/ref/gncGeomRef 相同，由 gncCompare
// 逐位元比對。--repeat N 在同一程序內再算 N 次，檢查每次都逐位元相同。
//
// 用法：gncGeomL0 --case DIR --out FILE [--repeat N]

#include "core/arrayFile.hpp"
#include "core/gncCuda.cuh"
#include "core/reduce.cuh"
#include "io/polyMeshReader.hpp"
#include "mesh/cellFaceGather.hpp"
#include "mesh/deviceMesh.cuh"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace
{

using Clock = std::chrono::steady_clock;

double seconds(const Clock::time_point a, const Clock::time_point b)
{
    return std::chrono::duration<double>(b - a).count();
}

std::vector<double> sliceInternal
(
    const std::vector<double>& v,
    const std::size_t nInternal,
    const std::size_t nComp
)
{
    return std::vector<double>(v.begin(), v.begin() + nInternal*nComp);
}

std::vector<double> gatherBoundary
(
    const std::vector<double>& v,
    const std::vector<std::int32_t>& faces,
    const std::size_t nComp
)
{
    std::vector<double> out;
    out.reserve(faces.size()*nComp);
    for (const std::int32_t f : faces)
    {
        for (std::size_t k = 0; k < nComp; ++k)
        {
            out.push_back(v[std::size_t(f)*nComp + k]);
        }
    }
    return out;
}

struct Snapshot
{
    std::vector<double> Cf, Sf, C, V, magSf, w, dc, nodc, corr;

    static Snapshot take(const gnc::DeviceMesh& dm)
    {
        Snapshot s;
        s.Cf = dm.Cf.downloadAoS();
        s.Sf = dm.Sf.downloadAoS();
        s.C = dm.C.downloadAoS();
        s.V = dm.V.download();
        s.magSf = dm.magSf.download();
        s.w = dm.weights.download();
        s.dc = dm.deltaCoeffs.download();
        s.nodc = dm.nonOrthDeltaCoeffs.download();
        s.corr = dm.nonOrthCorrectionVectors.downloadAoS();
        return s;
    }

    bool bitwiseEqual(const Snapshot& o) const
    {
        const auto eq = [](const std::vector<double>& a, const std::vector<double>& b)
        {
            return a.size() == b.size()
                && (a.empty() || std::memcmp(a.data(), b.data(), a.size()*8) == 0);
        };
        return eq(Cf, o.Cf) && eq(Sf, o.Sf) && eq(C, o.C) && eq(V, o.V)
            && eq(magSf, o.magSf) && eq(w, o.w) && eq(dc, o.dc)
            && eq(nodc, o.nodc) && eq(corr, o.corr);
    }
};

} // namespace


int main(int argc, char* argv[])
{
    std::string caseDir;
    std::string outFile;
    int repeat = 0;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--case" && i + 1 < argc)
        {
            caseDir = argv[++i];
        }
        else if (a == "--out" && i + 1 < argc)
        {
            outFile = argv[++i];
        }
        else if (a == "--repeat" && i + 1 < argc)
        {
            repeat = std::stoi(argv[++i]);
        }
        else
        {
            std::fprintf(stderr, "usage: gncGeomL0 --case DIR --out FILE [--repeat N]\n");
            return 2;
        }
    }
    if (caseDir.empty() || outFile.empty())
    {
        std::fprintf(stderr, "usage: gncGeomL0 --case DIR --out FILE [--repeat N]\n");
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

        const std::string meshDir = caseDir + "/constant/polyMesh";

        const auto t0 = Clock::now();
        const gnc::PolyMesh mesh = gnc::readPolyMesh(meshDir);
        const auto t1 = Clock::now();
        const gnc::CellFaceGather gather = gnc::buildCellFaceGather(mesh);
        const auto t2 = Clock::now();
        gnc::DeviceMesh dm = gnc::DeviceMesh::upload(mesh, gather);
        GNC_CUDA_CHECK(cudaDeviceSynchronize());
        const auto t3 = Clock::now();
        gnc::computeGeometry(dm);
        const auto t4 = Clock::now();

        std::printf
        (
            "mesh: %s (%s) nPoints %d nCells %d nFaces %d nInternalFaces %d"
            " patches %zu fvBoundaryFaces %d\n",
            meshDir.c_str(), mesh.format.c_str(), mesh.nPoints, mesh.nCells,
            mesh.nFaces, mesh.nInternalFaces, mesh.patches.size(),
            dm.nFvBoundaryFaces()
        );
        std::printf
        (
            "time_s: read %.3f gather %.3f upload %.3f geometry %.6f"
            " (informational; not a recorded measurement)\n",
            seconds(t0, t1), seconds(t1, t2), seconds(t2, t3), seconds(t3, t4)
        );

        const Snapshot first = Snapshot::take(dm);
        int repeatsEqual = 0;
        for (int r = 0; r < repeat; ++r)
        {
            gnc::computeGeometry(dm);
            if (Snapshot::take(dm).bitwiseEqual(first))
            {
                ++repeatsEqual;
            }
        }
        if (repeat > 0)
        {
            std::printf("repeat: %d/%d bitwise identical to the first run\n",
                repeatsEqual, repeat);
        }

        const double sumSeq = gnc::sumSequential(dm.V.data(), dm.V.size());
        const double sumTree = gnc::sumTree(dm.V.data(), dm.V.size());
        std::printf("sum(V): sequential %.17g tree %.17g\n", sumSeq, sumTree);

        // 有限體積邊界面（依 patch 順序，略過 empty）
        std::vector<std::int32_t> fvb;
        std::vector<std::int32_t> fvPatchSizes;
        for (const gnc::Patch& p : mesh.patches)
        {
            const std::int32_t n = p.isEmpty() ? 0 : p.size;
            fvPatchSizes.push_back(n);
            for (std::int32_t i = 0; i < n; ++i)
            {
                fvb.push_back(p.start + i);
            }
        }

        const std::size_t nI = std::size_t(mesh.nInternalFaces);
        const std::int32_t dims[4] =
            {mesh.nPoints, mesh.nCells, mesh.nFaces, mesh.nInternalFaces};

        namespace af = gnc::arrayFile;
        af::Writer w(outFile);
        w.text("producer", "gncGeomL0");
        w.text("case", std::filesystem::absolute(caseDir).string());
        w.text("device", std::string(prop.name) + " " + pci);
        w.i32("dims", dims, 4);
        w.i32("fvPatchSizes", fvPatchSizes.data(), fvPatchSizes.size());

        w.f64("points", mesh.points.data(), std::size_t(mesh.nPoints), 3);
        w.i32("owner", mesh.owner.data(), mesh.owner.size());
        w.i32("neighbour", mesh.neighbour.data(), mesh.neighbour.size());
        w.i32("faceOffsets", mesh.faceOffsets.data(), mesh.faceOffsets.size());
        w.i32("faceLabels", mesh.faceLabels.data(), mesh.faceLabels.size());

        w.f64("Cf", first.Cf.data(), std::size_t(mesh.nFaces), 3);
        w.f64("Sf", first.Sf.data(), std::size_t(mesh.nFaces), 3);
        w.f64("C", first.C.data(), std::size_t(mesh.nCells), 3);
        w.f64("V", first.V.data(), std::size_t(mesh.nCells));

        const auto writeSurf = [&]
        (
            const std::string& name,
            const std::vector<double>& v,
            const std::size_t nComp
        )
        {
            const std::vector<double> i = sliceInternal(v, nI, nComp);
            const std::vector<double> b = gatherBoundary(v, fvb, nComp);
            w.f64(name + ".i", i.data(), nI, std::uint32_t(nComp));
            w.f64(name + ".b", b.data(), fvb.size(), std::uint32_t(nComp));
        };
        writeSurf("magSf", first.magSf, 1);
        writeSurf("weights", first.w, 1);
        writeSurf("deltaCoeffs", first.dc, 1);
        writeSurf("nonOrthDeltaCoeffs", first.nodc, 1);
        writeSurf("nonOrthCorrectionVectors", first.corr, 3);

        w.f64("sumV", &sumSeq, 1);
        w.f64("sumV.tree", &sumTree, 1);
        w.close();

        std::printf("wrote %s\n", outFile.c_str());
        return (repeat > 0 && repeatsEqual != repeat) ? 1 : 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 2;
    }
}
