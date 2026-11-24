// N3a 的 L2：HiSA 初始化（createFields）後的狀態。GPU-Native-CFD，
// GPL-3.0-or-later。
//
// 在 GPU 上做 hisaModule::initialise 的場與熱力初始化，傾印的紀錄名稱與
// tools/ref/gncHisaDump（在 HiSA 自己的 hisa -postProcess 中執行）相同。
//
// 用法：gncInitL2 --case DIR --expanded DIR --out FILE

#include "core/arrayFile.hpp"
#include "core/gncCuda.cuh"
#include "io/polyMeshReader.hpp"
#include "mesh/cellFaceGather.hpp"
#include "mesh/deviceMesh.cuh"
#include "solvers/gncHisa/hisaState.cuh"

#include <cstdio>
#include <string>
#include <vector>

namespace
{

namespace af = gnc::arrayFile;

template<class T>
void dumpDF(af::Writer& w, const std::string& name, const gnc::DField<T>& f)
{
    const std::vector<double> h = f.downloadAoS();
    w.f64(name, h.data(), f.size(), gnc::NComp<T>::value);
}

template<class T>
void dumpVol(af::Writer& w, const std::string& name, const gnc::VolField<T>& f)
{
    dumpDF(w, name + ".i", f.in);
    dumpDF(w, name + ".b", f.bnd);
}

} // namespace


int main(int argc, char* argv[])
{
    std::string caseDir, expanded, outFile;
    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string a = argv[i];
        if (a == "--case") caseDir = argv[i + 1];
        else if (a == "--expanded") expanded = argv[i + 1];
        else if (a == "--out") outFile = argv[i + 1];
        else
        {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (caseDir.empty() || expanded.empty() || outFile.empty())
    {
        std::fprintf(stderr, "usage: gncInitL2 --case DIR --expanded DIR --out FILE\n");
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

        const gnc::HisaState s = gnc::hisaInitialise(dm, mesh, expanded);
        const gnc::ThermoParams& P = s.thermo.params;
        std::printf
        (
            "mesh: nCells %d fvBoundaryFaces %d\n"
            "thermo: RR %.17g Tstd %.17g W %.17g R %.17g Cp %.17g transport %s\n"
            "analysis: %s\n",
            mesh.nCells, dm.nFvBoundaryFaces(), P.RR, P.Tstd, P.W, P.R, P.Cp,
            P.transport ? "sutherland" : "const", s.inviscid ? "inviscid" : "viscous"
        );
        std::printf("boundary conditions:");
        for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
        {
            std::printf
            (
                " %s[p %s, T %s, U %s, e %s]", dm.patches[pi].name.c_str(),
                gnc::bcName(s.thermo.p.bc[pi].type), gnc::bcName(s.thermo.T.bc[pi].type),
                gnc::bcName(s.U.bc[pi].type), gnc::bcName(s.thermo.he.bc[pi].type)
            );
        }
        std::printf("\n");

        af::Writer w(outFile);
        w.text("producer", "gncInitL2");
        w.text("device", std::string(prop.name) + " " + pci);
        w.f64("RR", &P.RR, 1);
        w.f64("Tstd", &P.Tstd, 1);
        dumpVol(w, "p", s.thermo.p);
        dumpVol(w, "T", s.thermo.T);
        dumpVol(w, "U", s.U);
        dumpVol(w, "e", s.thermo.he);
        dumpVol(w, "psi", s.thermo.psi);
        dumpVol(w, "mu", s.thermo.mu);
        dumpVol(w, "alpha", s.thermo.alpha);
        dumpVol(w, "rho", s.rho);
        dumpVol(w, "rhoU", s.rhoU);
        dumpVol(w, "rhoE", s.rhoE);
        dumpVol(w, "eZero", s.eZero);
        dumpDF(w, "phi.i", s.phi.in);
        dumpDF(w, "phi.b", s.phi.bnd);
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
