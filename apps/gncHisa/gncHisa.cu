// gncHisa：密度基耦合隱式求解器，演算法源自 HiSA（N-017）。GPU-Native-CFD，
// GPL-3.0-or-later。
//
// 不是 HiSA 官方的程式。這是依 HiSA（CSIR，GPL-3.0-or-later）的演算法重新撰寫的
// GPU 實作：時間迴圈內的計算全部是本專案手寫的 CUDA kernel，執行時不連結
// OpenFOAM 或 HiSA 的程式庫。目前是參考次序版（與 HiSA 逐位元相同，
// results/measured/N3d-L3/），只支援穩態、無黏、層流或 Spalart–Allmaras（N4a）。
//
// 用法（在 OpenFOAM 的案例目錄）：
//   scripts/gnc_prepare_case.sh [-case DIR]    以 foamDictionary -expand 展開字典
//                                              到 <case>/gncExpanded（需要 OpenFOAM）
//   gncHisa [-case DIR] [-expanded DIR]        求解（不需要 OpenFOAM）
//           [-fast]                            效能版：多色 LU-SGS／SGS、樹狀歸約、大網格時
//                                              FP32 Jacobian（不與 HiSA 逐位元，N-020）
//           [-no-foam-compat]                  參考端的錯誤行為改為原始碼的正確結果（N-013；
//                                              預設照 HiSA v2606 的實際結果，逐位元）
//           [-profile]                         外迭代各段的時間
// 依 controlDict 的 writeControl（timeStep、runTime、adjustableRunTime）與
// writeInterval 寫出時間目錄 <timeName>/{p,T,U,rho}，Spalart–Allmaras 時另有
// nuTilda、nut、alphat（writeFormat ascii 為 %.17g，binary 為原始位元組）；收斂時
// （pseudoTime 的 pseudoTol／pseudoTolRel）與 HiSA 一樣多做一個外迭代後結束，
// 並寫出結束時的場。時間目錄的名稱照 OpenFOAM 的 general 格式與
// timePrecision（預設 6）。
// controlDict 的 functions 中 type 為 forces、forceCoeffs 的項目照 OpenFOAM 的
// executeControl／writeControl（timeStep、writeTime、none）計算並寫出
// postProcessing/<名稱>/<startTime>/ 的 coefficient.dat 或 force.dat、moment.dat
// （post/forceCoeffs.cuh；其他 function 略過）。

#include "core/gncCuda.cuh"
#include "fields/boundaryConditions.cuh"
#include "io/dictionary.hpp"
#include "io/fieldIO.hpp"
#include "io/foamTokenizer.hpp"
#include "post/forceCoeffs.cuh"
#include "solvers/gncHisa/hisaSolver.cuh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <sys/stat.h>

namespace
{

struct WriteControls
{
    std::string writeControl = "timeStep";
    double writeInterval = 1;
    int timePrecision = 6;
    bool binary = false;            // writeFormat binary
};

WriteControls readWriteControls(const std::string& path)
{
    gnc::FoamTokenizer t(path);
    const gnc::Dictionary d = gnc::Dictionary::parse(t, path, false);
    WriteControls c;
    c.writeControl = d.getWordOrDefault("writeControl", "timeStep");
    if
    (
        c.writeControl != "timeStep" && c.writeControl != "runTime"
     && c.writeControl != "adjustableRunTime"
    )
    {
        throw std::runtime_error(path + ": writeControl " + c.writeControl + " is not supported");
    }
    c.writeInterval = d.getScalar("writeInterval");
    c.binary = d.getWordOrDefault("writeFormat", "ascii") == "binary";
    if (d.found("timePrecision"))
    {
        c.timePrecision = int(d.getLabel("timePrecision"));
    }
    const std::string timeFormat = d.getWordOrDefault("timeFormat", "general");
    if (timeFormat != "general")
    {
        throw std::runtime_error(path + ": timeFormat " + timeFormat + " is not supported (general)");
    }
    if (!(c.writeInterval > 0))
    {
        throw std::runtime_error(path + ": writeInterval must be positive");
    }
    return c;
}

// Time::timeName：general 格式、timePrecision 位
std::string timeName(const double t, const int precision)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*g", precision, t);
    return buf;
}

// Time::operator++ 之後的 writeTime_（timeStep：timeIndex 是 writeInterval 的倍數；
// runTime：((t - startTime) + 0.5*deltaT)/writeInterval 的整數部分增加）
struct WriteClock
{
    WriteControls c;
    double startTime = 0;
    double deltaT = 1;
    long lastIndex = 0;

    bool writeTime(const double t, const int timeIndex)
    {
        if (c.writeControl == "timeStep")
        {
            return (timeIndex % long(c.writeInterval)) == 0;
        }
        const long idx = long(((t - startTime) + 0.5*deltaT)/c.writeInterval);
        if (idx > lastIndex)
        {
            lastIndex = idx;
            return true;
        }
        return false;
    }
};

void writeFields
(
    const std::string& caseDir,
    const std::string& name,
    const gnc::HisaSolver& sol,
    const bool binary
)
{
    const std::string dir = caseDir + "/" + name;
    ::mkdir(dir.c_str(), 0755);
    const gnc::HisaState& s = sol.s;
    gnc::writeField
    (
        dir + "/p",
        gnc::downloadVolField(sol.dm, s.thermo.p, "volScalarField", s.pDimensions),
        sol.mesh, name, binary
    );
    gnc::writeField
    (
        dir + "/T",
        gnc::downloadVolField(sol.dm, s.thermo.T, "volScalarField", s.TDimensions),
        sol.mesh, name, binary
    );
    gnc::writeField
    (
        dir + "/U",
        gnc::downloadVolField(sol.dm, s.U, "volVectorField", s.UDimensions),
        sol.mesh, name, binary
    );
    gnc::writeField
    (
        dir + "/rho",
        gnc::downloadVolField(sol.dm, s.rho, "volScalarField", "[1 -3 0 0 0 0 0]"),
        sol.mesh, name, binary
    );
    if (sol.sa)
    {
        const gnc::SpalartAllmaras& sa = *sol.sa;
        gnc::writeField
        (
            dir + "/nuTilda",
            gnc::downloadVolField(sol.dm, sa.nuTilda, "volScalarField", sa.nuTildaDimensions),
            sol.mesh, name, binary
        );
        gnc::writeField
        (
            dir + "/nut",
            gnc::downloadVolField(sol.dm, sa.nut, "volScalarField", sa.nutDimensions),
            sol.mesh, name, binary
        );
        gnc::writeField
        (
            dir + "/alphat",
            gnc::downloadVolField(sol.dm, sa.alphat, "volScalarField", sa.alphatDimensions),
            sol.mesh, name, binary
        );
    }
    if (sol.sst)
    {
        const gnc::KOmegaSST& m = *sol.sst;
        const struct { const char* n; const gnc::VolField<double>* f; const std::string* d; } fs[] =
        {
            {"k", &m.k, &m.kDimensions},
            {"omega", &m.omega, &m.omegaDimensions},
            {"nut", &m.nut, &m.nutDimensions},
            {"alphat", &m.alphat, &m.alphatDimensions}
        };
        for (const auto& x : fs)
        {
            gnc::writeField
            (
                dir + "/" + x.n,
                gnc::downloadVolField(sol.dm, *x.f, "volScalarField", *x.d),
                sol.mesh, name, binary
            );
        }
    }
    std::printf("Writing fields to %s\n", name.c_str());
}

} // namespace


int main(int argc, char* argv[])
{
    std::string caseDir = ".";
    std::string expanded;
    bool profile = false;
    gnc::FoamCompat compat;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "-fast")
        {
            // 效能版（不與 HiSA 逐位元）：多色 LU-SGS／SGS、樹狀歸約、FP32 Jacobian
            // （大網格時）；已設定的環境變數優先
            setenv("GNC_LUSGS", "color", 0);
            setenv("GNC_SGS", "color", 0);
            setenv("GNC_REDUCE", "tree", 0);
            setenv("GNC_JAC", "auto", 0);
            continue;
        }
        if (a == "-no-foam-compat")
        {
            // 參考端「不是原始碼字面意義」的行為改為原始碼的正確結果（N-013，
            // core/foamCompat.hpp：3D 黏性應力的轉置、stabilise 的引數）
            compat.transposeAliasing = false;
            compat.stabiliseSwappedArgs = false;
            continue;
        }
        if (a == "-profile")
        {
            profile = true;
            continue;
        }
        if (i + 1 >= argc)
        {
            std::fprintf(stderr, "option %s needs a value\n", a.c_str());
            return 2;
        }
        if (a == "-case") caseDir = argv[++i];
        else if (a == "-expanded") expanded = argv[++i];
        else
        {
            std::fprintf
            (
                stderr,
                "usage: gncHisa [-case DIR] [-expanded DIR] [-fast] [-no-foam-compat] [-profile]\n"
            );
            return 2;
        }
    }
    if (expanded.empty())
    {
        expanded = caseDir + "/gncExpanded";
    }

    try
    {
        int dev = 0;
        GNC_CUDA_CHECK(cudaGetDevice(&dev));
        cudaDeviceProp prop{};
        GNC_CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        std::printf("gncHisa (GPU-Native-CFD; algorithm after HiSA, not an official HiSA build)\n");
        std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
        const auto t0 = std::chrono::steady_clock::now();

        // 環境變數 GNC_FOAM_COMPAT=off 同 -no-foam-compat（variant_compare.sh 以環境變數選變體）
        if (const char* e = std::getenv("GNC_FOAM_COMPAT"))
        {
            if (std::string(e) == "off")
            {
                compat.transposeAliasing = false;
                compat.stabiliseSwappedArgs = false;
            }
        }
        if (!compat.transposeAliasing)
        {
            std::printf("foam compatibility off (N-013): source-literal results, not HiSA-bitwise\n");
        }
        gnc::HisaSolver sol(caseDir, expanded, compat);
        sol.profile = profile;
        sol.time = gnc::readTimeControls(expanded + "/controlDict");
        WriteClock clock;
        clock.c = readWriteControls(expanded + "/controlDict");
        clock.startTime = sol.time.startTime;
        clock.deltaT = sol.time.deltaT;
        // forces／forceCoeffs
        const std::vector<gnc::ForceFunction> forces =
            gnc::readForceFunctions(expanded + "/controlDict", sol.mesh);
        std::vector<std::unique_ptr<gnc::ForceFiles>> forceFiles;
        std::vector<gnc::ForceResult> forceLast(forces.size());
        std::vector<bool> forceValid(forces.size(), false);
        for (const gnc::ForceFunction& f : forces)
        {
            if (f.executeControl != "timeStep")
            {
                throw std::runtime_error
                (
                    "functions/" + f.name + ": executeControl " + f.executeControl + " is not supported"
                );
            }
            if (f.writeControl != "timeStep" && f.writeControl != "writeTime" && f.writeControl != "none")
            {
                throw std::runtime_error
                (
                    "functions/" + f.name + ": writeControl " + f.writeControl + " is not supported"
                );
            }
            forceFiles.emplace_back
            (
                new gnc::ForceFiles(caseDir, f, timeName(sol.time.startTime, clock.c.timePrecision))
            );
            std::printf("function %s (%s)\n", f.name.c_str(), f.coeffs ? "forceCoeffs" : "forces");
        }

        std::printf
        (
            "mesh: %d cells; analysis: %s; endTime %.17g\n\nStarting pseudotime iteration loop\n\n",
            sol.mesh.nCells,
            sol.s.inviscid ? "inviscid"
          : sol.sa ? "RAS SpalartAllmaras" : sol.sst ? "RAS kOmegaSST" : "laminar",
            sol.time.endTime
        );

        while (sol.running())
        {
            const bool iterated = sol.timeStep();
            const std::string name = timeName(sol.t, clock.c.timePrecision);
            std::printf("Time = %s\n", name.c_str());
            if (iterated)
            {
                const double* r = sol.lastGmres.initRes;
                std::printf
                (
                    "Pseudo Courant No: %.6g\n  GMRES iterations: %d   initial residual: "
                    "%.6g (%.6g %.6g %.6g) %.6g\n",
                    sol.pt.pseudoCoNum, sol.lastGmres.iterations, r[0], r[1], r[2], r[3], r[4]
                );
            }
            else
            {
                std::printf("pseudoTime: converged in %d iterations\n", sol.nOuterIterations);
            }
            // runTime.write()（writeAndEnd 時一定寫）
            const bool writeNow = clock.writeTime(sol.t, sol.timeIndex) || !iterated;
            if (writeNow)
            {
                writeFields(caseDir, name, sol, clock.c.binary);
            }
            for (std::size_t i = 0; i < forces.size(); ++i)
            {
                const gnc::ForceFunction& f = forces[i];
                if (sol.timeIndex % f.executeInterval == 0)
                {
                    forceLast[i] = gnc::evaluateForces(sol, f);
                    forceValid[i] = true;
                }
                const bool fw =
                    f.writeControl == "timeStep"
                  ? sol.timeIndex % std::max(1, int(f.writeInterval)) == 0
                  : (f.writeControl == "writeTime" && writeNow);
                if (fw && forceValid[i])
                {
                    forceFiles[i]->write(name, forceLast[i]);
                }
            }
            const double el =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            std::printf("ExecutionTime = %.6g s\n\n", el);
        }
        if (profile && sol.timings.n > 0)
        {
            const gnc::OuterTimings& m = sol.timings;
            const double total =
                m.flux + m.residual + m.pseudoTime + m.jacobian + m.gmres + m.update + m.turbulence;
            std::printf
            (
                "profile: %d outer iterations, mean wall time per iteration %.6g s\n"
                "  flux %.6g  residual %.6g  pseudoTime %.6g  jacobian %.6g  gmres %.6g"
                "  update %.6g  turbulence %.6g\n",
                m.n, total/m.n, m.flux/m.n, m.residual/m.n, m.pseudoTime/m.n, m.jacobian/m.n,
                m.gmres/m.n, m.update/m.n, m.turbulence/m.n
            );
            const gnc::GmresProfile& g = gnc::gmresProfile();
            if (g.nPrecondition > 0)
            {
                std::printf
                (
                    "  gmres parts per iteration (GNC_PROFILE_GMRES=1): precondition %.6g s (%ld calls)"
                    "  matrixMul %.6g s (%ld)  reductions %.6g s (%ld batches)\n",
                    g.precondition/m.n, g.nPrecondition, g.matrixMul/m.n, g.nMatrixMul,
                    g.reduce/m.n, g.nReduce
                );
            }
        }
        std::printf("End\n");
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 2;
    }
}
