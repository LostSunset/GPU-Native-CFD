// N3b 的 L2：第一個外迭代的通量與殘差。GPU-Native-CFD，GPL-3.0-or-later。
//
// 初始化（N3a）之後算 AUSM+up 通量與殘差，傾印 rhoR、rhoUR、rhoER、phi
// （黏性時另有 tauMC）；
// 參考是 HiSA 本身跑一步（穩態）之後由 gncHisaDump 傾印的同名欄位
// （GMRES 只讀殘差，外迭代結束時殘差仍是該迭代的值）。
// 只支援穩態，無黏或層流（laminar Stokes，--expanded 要有展開的
// turbulenceProperties）；其他設定報錯。殘差之前先做 fvOptions(rho, U) 引起
// 的 U 邊界 updateCoeffs（hisaBoundaryUpdate.cuh）。
// --no-foam-compat：關閉參考端行為的相容設定（core/foamCompat.hpp），得到
// 原始碼字面的結果（與參考比對時用作反向對照）。
//
// 用法：gncResL2 --case DIR --expanded DIR --out FILE [--no-foam-compat]

#include "core/arrayFile.hpp"
#include "core/gncCuda.cuh"
#include "io/fvSchemes.hpp"
#include "io/polyMeshReader.hpp"
#include "mesh/cellFaceGather.hpp"
#include "mesh/deviceMesh.cuh"
#include "ops/fvcOps.cuh"
#include "core/foamCompat.hpp"
#include "solvers/gncHisa/hisaBoundaryUpdate.cuh"
#include "solvers/gncHisa/hisaFlux.cuh"
#include "solvers/gncHisa/hisaState.cuh"
#include "solvers/gncHisa/hisaViscous.cuh"

#include <cstdio>
#include <sstream>
#include <stdexcept>
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

} // namespace


int main(int argc, char* argv[])
{
    std::string caseDir, expanded, outFile;
    gnc::FoamCompat compat;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--no-foam-compat")
        {
            compat.transposeAliasing = false;
            compat.stabiliseSwappedArgs = false;
            continue;
        }
        if (i + 1 >= argc)
        {
            std::fprintf(stderr, "option %s needs a value\n", a.c_str());
            return 2;
        }
        if (a == "--case") caseDir = argv[++i];
        else if (a == "--expanded") expanded = argv[++i];
        else if (a == "--out") outFile = argv[++i];
        else
        {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (caseDir.empty() || expanded.empty() || outFile.empty())
    {
        std::fprintf
        (
            stderr,
            "usage: gncResL2 --case DIR --expanded DIR --out FILE [--no-foam-compat]\n"
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

        const gnc::FvSchemes schemes(expanded + "/fvSchemes");
        {
            std::istringstream is(schemes.ddt("default"));
            std::string w, last;
            while (is >> w)
            {
                last = w;
            }
            if (last != "steadyState")
            {
                throw std::runtime_error("only steady-state cases are supported yet (N3d)");
            }
        }

        gnc::HisaState s = gnc::hisaInitialise(dm, mesh, expanded);
        if
        (
            !s.inviscid
         && gnc::readTurbulenceModelKind(expanded + "/turbulenceProperties")
         != gnc::TurbulenceModelKind::laminarStokes
        )
        {
            throw std::runtime_error("gncResL2 supports laminar (Stokes) cases only");
        }
        const gnc::LSVectors ls = gnc::faceLeastSquaresVectors(dm);

        gnc::HisaFlux flux;
        gnc::ausmPlusUpFlux(dm, ls, schemes, s, flux);
        gnc::updateCoeffsU(dm, s, flux);
        gnc::updateEnergyT(dm, s, flux);
        gnc::HisaResiduals res;
        gnc::inviscidResiduals(dm, s, flux, res);
        gnc::VolField<gnc::tens> tauMC;
        if (!s.inviscid)
        {
            gnc::addLaminarViscousResiduals(dm, ls, schemes, s, flux, compat, res, tauMC);
        }

        std::printf
        (
            "mesh: nCells %d fvBoundaryFaces %d\nanalysis: %s\nfoamCompat: transposeAliasing %s\n"
            "schemes: flux %s lowMachAusm %s"
            " reconstruct(U) '%s' reconstruct(T) '%s' reconstruct(rho) '%s'\n",
            mesh.nCells, dm.nFvBoundaryFaces(), s.inviscid ? "inviscid" : "laminar",
            compat.transposeAliasing ? "on" : "off",
            schemes.top("fluxScheme", "").c_str(),
            schemes.top("lowMachAusm", "(default true)").c_str(),
            schemes.interpolation("reconstruct(U)").c_str(),
            schemes.interpolation("reconstruct(T)").c_str(),
            schemes.interpolation("reconstruct(rho)").c_str()
        );

        af::Writer w(outFile);
        w.text("producer", "gncResL2");
        w.text("device", std::string(prop.name) + " " + pci);
        w.text("foamCompat", compat.transposeAliasing ? "transposeAliasing" : "off");
        w.f64("RR", &s.thermo.params.RR, 1);
        w.f64("Tstd", &s.thermo.params.Tstd, 1);
        dumpDF(w, "rhoR.i", res.rhoR.in);
        dumpDF(w, "rhoR.b", res.rhoR.bnd);
        dumpDF(w, "rhoUR.i", res.rhoUR.in);
        dumpDF(w, "rhoUR.b", res.rhoUR.bnd);
        dumpDF(w, "rhoER.i", res.rhoER.in);
        dumpDF(w, "rhoER.b", res.rhoER.bnd);
        dumpDF(w, "phi.i", flux.phi.in);
        dumpDF(w, "phi.b", flux.phi.bnd);
        if (!s.inviscid)
        {
            dumpDF(w, "tauMC.i", tauMC.in);
            dumpDF(w, "tauMC.b", tauMC.bnd);
        }
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
