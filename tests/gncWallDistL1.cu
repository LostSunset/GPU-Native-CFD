// N4a-1 的 L1：牆距離（meshWave；N4b 另有 Poisson）。GPU-Native-CFD，GPL-3.0-or-later。
//
// 網格幾何在 GPU 上計算（N1，與 OpenFOAM 逐位元），下載格心與面心後在主機上
// 算 meshWave 的牆距離（mesh/wallDist.cpp）。寫出與 tools/ref/gncHisaDump
// （wallDist true）同名的紀錄：
//   yWall.i   格的距離
//   yWall.b   有限體積邊界面的值：wall patch 為 sqrt(distSqr) + SMALL（fixedValue），
//             其他 patch 為相鄰格的距離（zeroGradient 的 correctBoundaryConditions）
//   yWall.nUnset（int32，1 個值）
//
// --poisson EXPANDED：改用 Poisson（turbulence/turbulenceCommon.cuh 的
// wallDistancePoisson，fvSchemes 取 EXPANDED/fvSchemes）；與 HiSA（GAMG）不逐位元，
// 比對的目的是量測差異。yWall.nUnset 寫 0。
//
// 用法：gncWallDistL1 --case DIR --out FILE [--poisson EXPANDED]

#include "core/arrayFile.hpp"
#include "core/gncCuda.cuh"
#include "io/polyMeshReader.hpp"
#include "mesh/cellFaceGather.hpp"
#include "mesh/deviceMesh.cuh"
#include "mesh/wallDist.hpp"
#include "ops/fvcOps.cuh"
#include "turbulence/turbulenceCommon.cuh"

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    std::string caseDir, outFile, poisson;
    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string a = argv[i];
        if (a == "--case") caseDir = argv[i + 1];
        else if (a == "--out") outFile = argv[i + 1];
        else if (a == "--poisson") poisson = argv[i + 1];
        else
        {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (caseDir.empty() || outFile.empty())
    {
        std::fprintf(stderr, "usage: gncWallDistL1 --case DIR --out FILE\n");
        return 2;
    }
    try
    {
        const gnc::PolyMesh mesh = gnc::readPolyMesh(caseDir + "/constant/polyMesh");
        const gnc::CellFaceGather gather = gnc::buildCellFaceGather(mesh);
        gnc::DeviceMesh dm = gnc::DeviceMesh::upload(mesh, gather);
        gnc::computeGeometry(dm);
        if (!poisson.empty())
        {
            const gnc::FvSchemes schemes(poisson + "/fvSchemes");
            const gnc::LSVectors ls = gnc::faceLeastSquaresVectors(dm);
            const gnc::WallDistanceField w = gnc::wallDistancePoisson(dm, mesh, schemes, ls);
            const std::vector<double> yi = w.y.download(), yb = w.yB.download();
            const std::int32_t nUnset = 0;
            gnc::arrayFile::Writer out(outFile);
            out.text("producer", "gncWallDistL1 (Poisson)");
            out.f64("yWall.i", yi.data(), yi.size(), 1);
            out.f64("yWall.b", yb.data(), yb.size(), 1);
            out.i32("yWall.nUnset", &nUnset, 1, 1);
            out.close();
            std::printf("mesh: nCells %d (Poisson)\nwrote %s\n", mesh.nCells, outFile.c_str());
            return 0;
        }
        const std::vector<double> C = dm.C.downloadAoS();
        const std::vector<double> Cf = dm.Cf.downloadAoS();

        const gnc::WallDistance wd = gnc::meshWaveWallDistance(mesh, C, Cf);

        // 有限體積邊界：依 patch 次序，empty 沒有面
        std::vector<double> yb;
        for (const gnc::Patch& p : mesh.patches)
        {
            if (p.isEmpty())
            {
                continue;
            }
            for (std::int32_t i = 0; i < p.size; ++i)
            {
                const std::int32_t f = p.start + i;
                yb.push_back(p.type == "wall" ? wd.face[std::size_t(f)] : wd.cell[std::size_t(mesh.owner[f])]);
            }
        }
        std::int32_t nWall = 0;
        for (const gnc::Patch& p : mesh.patches)
        {
            if (p.type == "wall") nWall += p.size;
        }
        std::printf
        (
            "mesh: nCells %d wallFaces %d nUnset %d\n", mesh.nCells, nWall, wd.nUnset
        );

        gnc::arrayFile::Writer w(outFile);
        w.text("producer", "gncWallDistL1");
        w.f64("yWall.i", wd.cell.data(), wd.cell.size(), 1);
        w.f64("yWall.b", yb.data(), yb.size(), 1);
        w.i32("yWall.nUnset", &wd.nUnset, 1, 1);
        w.close();
        std::printf("wrote %s\n", outFile.c_str());
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 2;
    }
}
