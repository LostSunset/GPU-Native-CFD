// N3c 的 L2：第一個外迭代（hisaModule::outerIteration）。GPU-Native-CFD，
// GPL-3.0-or-later。
//
// 以 HisaSolver（hisaSolver.cu，N3d 的 L3 也用它）跑第一個時間步：
// 初始化（N3a）、通量、邊界的 updateCoeffs 與殘差（N3b）、rPseudoDeltaT 與
// ddtCoeff、Jacobian、LU-SGS、GMRES（N3c-1、N3c-2），限界與 updateFields
// （N3c-3）。
// --out：與 tools/ref/gncHisaJacobianDump 同名的紀錄（三個參考檔都以這個輸出檔比對），
// 在 GMRES 剛結束時寫出（hisaStateDump.cuh 的 dumpOuterIteration）
//   <t>.jacobian.gnc      J.<子區塊>.diag/upper/lower、rPseudoDeltaT、rDiagCoeff、V、
//                         dbg.U.b、dbg.p.gradient
//   <t>.precondition.gnc  P.in、P.out（第一次 precondition()：GMRES 的 r0 = R）
//   <t>.solve.gnc         rho、rhoU、rhoE 的 .i、.b（GMRES 更新後、限界之前）
// --out-state：外迭代結束時的狀態，與 tools/ref/gncHisaDump 同名的紀錄
//   RR、Tstd、p T U e psi mu alpha rho rhoU rhoE phi pseudoCoDamping 的 .i、.b
// 參考是 HiSA 本身跑一步（穩態變體，預條件器 gncLUSGSDump，function object
// gncHisaDump 在時間步結束時傾印）。
//
// --no-foam-compat 關閉全部參考端行為的相容設定（core/foamCompat.hpp），作為反向對照。
//
// 用法：gncOuterL2 --case DIR --expanded DIR --out FILE --out-state FILE
//                  [--no-foam-compat]

#include "core/gncCuda.cuh"
#include "hisaStateDump.cuh"

#include <cstdio>
#include <stdexcept>
#include <string>

int main(int argc, char* argv[])
{
    std::string caseDir, expanded, outFile, stateFile;
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
        else if (a == "--out-state") stateFile = argv[++i];
        else
        {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (caseDir.empty() || expanded.empty() || outFile.empty() || stateFile.empty())
    {
        std::fprintf
        (
            stderr,
            "usage: gncOuterL2 --case DIR --expanded DIR --out FILE --out-state FILE\n"
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
        const std::string device = std::string(prop.name) + " " + pci;

        gnc::HisaSolver sol(caseDir, expanded, compat);
        std::printf
        (
            "mesh: nCells %d fvBoundaryFaces %d\nanalysis: %s\n",
            sol.mesh.nCells, sol.dm.nFvBoundaryFaces(),
            sol.s.inviscid ? "inviscid" : sol.sa ? "RAS SpalartAllmaras" : sol.sst ? "RAS kOmegaSST" : "laminar"
        );

        gncTest::DbgSnapshot dbg;
        const auto afterJacobian = [&](const gnc::OuterIterationView& v)
        {
            dbg = gncTest::jacobianTimeSnapshot(sol, v);
        };
        const auto afterSolve = [&](const gnc::OuterIterationView& v)
        {
            std::printf
            (
                "pseudoCoNum %.17g\nLU-SGS levels: forward %zu reverse %zu\n",
                v.pt.pseudoCoNum, v.plan.fwdLevelOff.size() - 1, v.plan.revLevelOff.size() - 1
            );
            // HiSA 的記錄次序 (rho, rhoU, rhoE)
            const auto printRes = [](const char* what, const double* r)
            {
                std::printf
                (
                    "%s %.17g (%.17g %.17g %.17g) %.17g\n", what, r[0], r[1], r[2], r[3], r[4]
                );
            };
            std::printf("GMRES iterations %d\n", v.gmres.iterations);
            printRes("GMRES initial residual", v.gmres.initRes);
            printRes("GMRES final residual", v.gmres.finalRes);

            gncTest::af::Writer w(outFile);
            w.text("producer", "gncOuterL2");
            w.text("device", device);
            gncTest::dumpOuterIteration(w, sol, v, dbg);
            w.close();
            std::printf("wrote %s\n", outFile.c_str());
        };

        if (!sol.timeStep(afterSolve, afterJacobian))
        {
            throw std::runtime_error("the first time step did not run an outer iteration");
        }

        gncTest::af::Writer ws(stateFile);
        ws.text("producer", "gncOuterL2");
        ws.text("device", device);
        gncTest::dumpHisaState(ws, sol);
        ws.close();
        std::printf("wrote %s\n", stateFile.c_str());
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 2;
    }
}
