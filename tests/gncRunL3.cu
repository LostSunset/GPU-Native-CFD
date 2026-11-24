// N3d 的 L3：穩態的整個 run（多個外迭代、SER、收斂判斷）。GPU-Native-CFD，
// GPL-3.0-or-later。
//
// 以 HisaSolver 照 HiSA 的時間迴圈執行到 endTime 或收斂（writeAndEnd）。
// 在指定的時間步之後與結束時，以 tools/ref/gncHisaDump 同名的紀錄寫出狀態：
//   <out-dir>/<timeIndex>.gnc、<out-dir>/end.gnc
// 參考是 HiSA 本身執行同一個穩態變體時，gncHisaDump（timeIndices、atEnd）
// 在同樣的時間步與 end() 寫出的檔案。controlDict 的 functions 有 forces／forceCoeffs
// 時，另以 post/forceCoeffs 計算，紀錄名稱為 <function 名稱>.<OpenFOAM 的 result 名稱>
// （參考端 gncHisaDump 的 forceObjects）。每一步印出 corr、pseudoCoNum 與
// GMRES 的初始殘差（%.17g）。
//
// --debug-step K：在第 K 步的 GMRES 剛結束時，另寫 <out-dir>/debug.K.gnc，紀錄與
// gncLUSGSDump 的 <K>.jacobian.gnc、<K>.precondition.gnc、<K>.solve.gnc 相同
// （參考端在穩態變體中把預條件器設為 gncLUSGSDump，每一步都寫）。
//
// 用法：gncRunL3 --case DIR --expanded DIR --out-dir DIR [--time-indices 1,2,10]
//                [--debug-step K] [--no-foam-compat]

#include "core/gncCuda.cuh"
#include "hisaStateDump.cuh"
#include "post/forceCoeffs.cuh"

#include <cstdio>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

namespace
{

std::set<int> parseIndices(const std::string& s)
{
    std::set<int> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        if (!item.empty())
        {
            out.insert(std::stoi(item));
        }
    }
    return out;
}

void writeState
(
    const std::string& file,
    const std::string& device,
    const gnc::HisaSolver& sol,
    const std::vector<gnc::ForceFunction>& forces
)
{
    gncTest::af::Writer w(file);
    w.text("producer", "gncRunL3");
    w.text("device", device);
    gncTest::dumpHisaState(w, sol);
    for (const gnc::ForceFunction& f : forces)
    {
        const gnc::ForceResult r = gnc::evaluateForces(sol, f);
        for (const auto& kv : r.scalars)
        {
            w.f64(f.name + "." + kv.first, &kv.second, 1);
        }
        for (const auto& kv : r.vectors)
        {
            w.f64(f.name + "." + kv.first, &kv.second.x, 1, 3);
        }
        // forces 的 force、moment 場（useNamePrefix 預設 false，場名沒有前綴）
        const std::vector<double> zeros(3*std::size_t(sol.mesh.nCells), 0.0);
        w.f64("force.i", zeros.data(), std::uint64_t(sol.mesh.nCells), 3);
        w.f64("force.b", &r.faceForce[0].x, r.faceForce.size(), 3);
        w.f64("moment.i", zeros.data(), std::uint64_t(sol.mesh.nCells), 3);
        w.f64("moment.b", &r.faceMoment[0].x, r.faceMoment.size(), 3);
    }
    w.close();
}

} // namespace


int main(int argc, char* argv[])
{
    std::string caseDir, expanded, outDir, indices;
    int debugStep = -1;
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
        else if (a == "--out-dir") outDir = argv[++i];
        else if (a == "--time-indices") indices = argv[++i];
        else if (a == "--debug-step") debugStep = std::stoi(argv[++i]);
        else
        {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (caseDir.empty() || expanded.empty() || outDir.empty())
    {
        std::fprintf
        (
            stderr,
            "usage: gncRunL3 --case DIR --expanded DIR --out-dir DIR [--time-indices 1,2,10]\n"
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
        const std::set<int> dumpAt = parseIndices(indices);
        ::mkdir(outDir.c_str(), 0755);

        gnc::HisaSolver sol(caseDir, expanded, compat);
        sol.time = gnc::readTimeControls(expanded + "/controlDict");
        const std::vector<gnc::ForceFunction> forces =
            gnc::readForceFunctions(expanded + "/controlDict", sol.mesh);
        for (const gnc::ForceFunction& f : forces)
        {
            std::printf
            (
                "function %s: %s, %zu patch(es)\n", f.name.c_str(),
                f.coeffs ? "forceCoeffs" : "forces", f.patchIDs.size()
            );
        }
        std::printf
        (
            "mesh: nCells %d fvBoundaryFaces %d\nanalysis: %s\n"
            "time: endTime %.17g deltaT %.17g\n"
            "pseudoTime: nPseudoCorr %d nPseudoCorrMin %d\n",
            sol.mesh.nCells, sol.dm.nFvBoundaryFaces(),
            sol.s.inviscid ? "inviscid" : sol.sa ? "RAS SpalartAllmaras" : sol.sst ? "RAS kOmegaSST" : "laminar",
            sol.time.endTime, sol.time.deltaT,
            sol.pseudotime.nPseudoCorr, sol.pseudotime.nPseudoCorrMin
        );

        while (sol.running())
        {
            gncTest::DbgSnapshot dbg;
            const auto debugJacobian = [&](const gnc::OuterIterationView& v)
            {
                if (sol.timeIndex == debugStep)
                {
                    dbg = gncTest::jacobianTimeSnapshot(sol, v);
                }
            };
            const auto debugHook = [&](const gnc::OuterIterationView& v)
            {
                if (sol.timeIndex != debugStep)
                {
                    return;
                }
                const std::string file = outDir + "/debug." + std::to_string(debugStep) + ".gnc";
                gncTest::af::Writer w(file);
                w.text("producer", "gncRunL3");
                w.text("device", device);
                gncTest::dumpOuterIteration(w, sol, v, dbg);
                w.close();
            };
            const bool iterated = sol.timeStep(debugHook, debugJacobian);
            if (iterated)
            {
                const double* r = sol.lastGmres.initRes;
                std::printf
                (
                    "step %d corr %d pseudoCoNum %.17g GMRES %d residual %.17g (%.17g %.17g %.17g) %.17g\n",
                    sol.timeIndex, sol.control.corr, sol.pt.pseudoCoNum,
                    sol.lastGmres.iterations, r[0], r[1], r[2], r[3], r[4]
                );
            }
            else
            {
                std::printf("step %d converged: writeAndEnd\n", sol.timeIndex);
            }
            if (dumpAt.count(sol.timeIndex))
            {
                writeState(outDir + "/" + std::to_string(sol.timeIndex) + ".gnc", device, sol, forces);
            }
        }
        writeState(outDir + "/end.gnc", device, sol, forces);
        std::printf
        (
            "end: timeIndex %d outerIterations %d converged %s\n",
            sol.timeIndex, sol.nOuterIterations, sol.ended ? "yes" : "no"
        );
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 2;
    }
}
