// gncRhoSimple：可壓縮穩態 SIMPLE 壓力基求解器，演算法源自 OpenFOAM rhoSimpleFoam
// （N-017、N-023）。GPU-Native-CFD，GPL-3.0-or-later。
//
// 不是 OpenFOAM 官方的程式。這是依 rhoSimpleFoam（OpenFOAM v2606，GPL-3.0-or-later）
// 的演算法重新撰寫的 GPU 實作：迴圈內的計算全部是本專案手寫的 CUDA kernel，執行時
// 不連結 OpenFOAM 的程式庫。目前是參考次序版（目標與 rhoSimpleFoam 逐位元相同）。
//
// 用法（在 OpenFOAM 的案例目錄）：
//   scripts/gnc_prepare_case.sh [-case DIR]    展開字典到 <case>/gncExpanded（需要 OpenFOAM）
//   gncRhoSimple [-case DIR] [-expanded DIR]   求解（不需要 OpenFOAM）
//                [-dumpDir DIR -dumpSteps '1 2 10']
//                                              傾印（GNC 陣列檔，與 tools/ref/gncRhoSimpleRef
//                                              同名同內容）：0.init，與每個指定迭代的
//                                              <k>.U、<k>.E、<k>.p、<k>.turb
//                [-fast]                       效能版：多色 DILU／DIC、樹狀歸約，fvSolution 的 GAMG
//                                              以 PCG＋聚合式 AMG 代替（不與 rhoSimpleFoam 逐位元；
//                                              收斂後的離散解相同）
//                [-profile]                    各段的時間（每段結束前同步，不影響數值）
// 依 controlDict 的 writeControl（timeStep、runTime、adjustableRunTime）與 writeInterval
// 寫出 <timeName>/{p,T,U,rho,phi,k,omega,nut,alphat}；residualControl 滿足時與
// rhoSimpleFoam 一樣在下一個迭代開頭結束並寫出。

#include "core/arrayFile.hpp"
#include "core/coopLevels.cuh"
#include "core/gncCuda.cuh"
#include "fields/boundaryConditions.cuh"
#include "io/dictionary.hpp"
#include "io/fieldIO.hpp"
#include "io/foamTokenizer.hpp"
#include "solvers/gncRhoSimple/rhoSimpleSolver.cuh"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <sstream>
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
    bool binary = false;
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

// Time::operator++ 之後的 writeTime_
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
    const gnc::RhoSimpleSolver& s,
    const bool binary
)
{
    const std::string dir = caseDir + "/" + name;
    ::mkdir(dir.c_str(), 0755);
    const auto vol = [&](const char* n, const gnc::VolField<double>& f, const std::string& dims)
    {
        gnc::writeField
        (
            dir + "/" + n, gnc::downloadVolField(s.dm, f, "volScalarField", dims), s.mesh, name, binary
        );
    };
    vol("p", s.thermo.p, s.pDimensions);
    vol("T", s.thermo.T, s.TDimensions);
    gnc::writeField
    (
        dir + "/U", gnc::downloadVolField(s.dm, s.U, "volVectorField", s.UDimensions), s.mesh, name, binary
    );
    vol("rho", s.rho, "[1 -3 0 0 0 0 0]");
    gnc::writeField
    (
        dir + "/phi", gnc::downloadSurfaceScalarField(s.dm, s.phi, "[1 0 -1 0 0 0 0]"), s.mesh, name, binary
    );
    const gnc::KOmegaSST& m = *s.sst;
    vol("k", m.k, m.kDimensions);
    vol("omega", m.omega, m.omegaDimensions);
    vol("nut", m.nut, m.nutDimensions);
    vol("alphat", m.alphat, m.alphatDimensions);
    std::printf("Writing fields to %s\n", name.c_str());
}


// 傾印（tools/ref/gncRhoSimpleRef 的 writeState、writeMatrix 同名同內容）
template<class T>
void dumpVol(gnc::arrayFile::Writer& w, const std::string& name, const gnc::VolField<T>& f)
{
    constexpr int N = gnc::NComp<T>::value;
    const std::vector<double> i = f.in.downloadAoS();
    const std::vector<double> b = f.bnd.downloadAoS();
    w.f64(name + ".i", i.data(), i.size()/N, N);
    w.f64(name + ".b", b.data(), b.size()/N, N);
}

void dumpSurf(gnc::arrayFile::Writer& w, const std::string& name, const gnc::SurfField<double>& f)
{
    const std::vector<double> i = f.in.downloadAoS();
    const std::vector<double> b = f.bnd.downloadAoS();
    w.f64(name + ".i", i.data(), i.size());
    w.f64(name + ".b", b.data(), b.size());
}

void dumpState(gnc::arrayFile::Writer& w, const gnc::RhoSimpleSolver& s)
{
    dumpVol(w, "p", s.thermo.p);
    dumpVol(w, "T", s.thermo.T);
    dumpVol(w, "U", s.U);
    dumpVol(w, "e", s.thermo.he);
    w.text("h.missing", "h");
    dumpVol(w, "thermo:psi", s.thermo.psi);
    dumpVol(w, "thermo:alpha", s.thermo.alpha);
    dumpVol(w, "thermo:mu", s.thermo.mu);
    dumpVol(w, "rho", s.rho);
    dumpSurf(w, "phi", s.phi);
    dumpVol(w, "k", s.sst->k);
    dumpVol(w, "omega", s.sst->omega);
    dumpVol(w, "nut", s.sst->nut);
    dumpVol(w, "alphat", s.sst->alphat);
}

void dumpMatrix(gnc::arrayFile::Writer& w, const std::string& name, const gnc::FvVectorMatrix& M)
{
    if (M.hasDiag)
    {
        const std::vector<double> d = M.diag.download();
        w.f64(name + ".diag", d.data(), d.size());
    }
    if (M.hasUpper)
    {
        const std::vector<double> d = M.upper.download();
        w.f64(name + ".upper", d.data(), d.size());
    }
    if (M.hasLower)
    {
        const std::vector<double> d = M.lower.download();
        w.f64(name + ".lower", d.data(), d.size());
    }
    const std::vector<double> src = M.source.downloadAoS();
    w.f64(name + ".source", src.data(), src.size()/3, 3);
    const std::vector<double> ic = M.internalCoeffs.downloadAoS();
    w.f64(name + ".internalCoeffs", ic.data(), ic.size()/3, 3);
    const std::vector<double> bc = M.boundaryCoeffs.downloadAoS();
    w.f64(name + ".boundaryCoeffs", bc.data(), bc.size()/3, 3);
}

struct Dumper
{
    std::string dir;
    std::set<int> steps;

    bool active(const int k) const
    {
        return !dir.empty() && (k == 0 || steps.count(k));
    }

    std::string path(const int k, const char* stage) const
    {
        return dir + "/" + std::to_string(k) + "." + stage + ".gnc";
    }
};

void printPerf(const char* solver, const std::string& field, const gnc::SolverPerf& p)
{
    std::printf
    (
        "%s:  Solving for %s, Initial residual = %.6g, Final residual = %.6g, No Iterations %d\n",
        solver, field.c_str(), p.initialResidual, p.finalResidual, p.nIterations
    );
}

} // namespace


int main(int argc, char* argv[])
{
    std::string caseDir = ".";
    std::string expanded;
    Dumper dump;
    bool profile = false;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "-profile")
        {
            profile = true;
            continue;
        }
        if (a == "-fast")
        {
            // 效能版（不與 rhoSimpleFoam 逐位元）：多色 DILU／DIC、樹狀歸約；已設定的
            // 環境變數優先
            setenv("GNC_ILU", "color", 0);
            setenv("GNC_REDUCE", "tree", 0);
            setenv("GNC_GAMG", "amg", 0);
            // 多色 DILU／DIC：二分圖時兩色（N-028）
            setenv("GNC_COLOR_BIPARTITE", "1", 0);
            continue;
        }
        if (i + 1 >= argc)
        {
            std::fprintf(stderr, "option %s needs a value\n", a.c_str());
            return 2;
        }
        if (a == "-case") caseDir = argv[++i];
        else if (a == "-expanded") expanded = argv[++i];
        else if (a == "-dumpDir") dump.dir = argv[++i];
        else if (a == "-dumpSteps")
        {
            std::string v = argv[++i];
            std::replace(v.begin(), v.end(), ',', ' ');
            std::replace(v.begin(), v.end(), '(', ' ');
            std::replace(v.begin(), v.end(), ')', ' ');
            std::istringstream is(v);
            int k;
            while (is >> k)
            {
                dump.steps.insert(k);
            }
        }
        else
        {
            std::fprintf
            (
                stderr,
                "usage: gncRhoSimple [-case DIR] [-expanded DIR] [-dumpDir DIR -dumpSteps 'K ...'] [-fast] [-profile]\n"
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
        std::printf("gncRhoSimple (GPU-Native-CFD; algorithm after rhoSimpleFoam, not an official OpenFOAM build)\n");
        std::printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
        if (gnc::envIs("GNC_ILU", "color") || gnc::envIs("GNC_REDUCE", "tree") || gnc::envIs("GNC_GAMG", "amg"))
        {
            const auto env = [](const char* n) { return std::getenv(n) ? std::getenv(n) : "-"; };
            std::printf
            (
                "performance options: GNC_ILU=%s GNC_REDUCE=%s GNC_GAMG=%s GNC_PCG_PRECOND=%s"
                " (not bitwise to rhoSimpleFoam)\n",
                env("GNC_ILU"), env("GNC_REDUCE"), env("GNC_GAMG"), env("GNC_PCG_PRECOND")
            );
        }
        const auto t0 = std::chrono::steady_clock::now();

        gnc::RhoSimpleSolver sol(caseDir, expanded);
        sol.profile = profile;
        WriteClock clock;
        clock.c = readWriteControls(expanded + "/controlDict");
        clock.startTime = sol.time.startTime;
        clock.deltaT = sol.time.deltaT;
        std::printf
        (
            "mesh: %d cells; RAS kOmegaSST; endTime %.17g\n",
            sol.mesh.nCells, sol.time.endTime
        );
        if (sol.limitMaxP || sol.limitMinP)
        {
            std::printf("pressureControl\n");
            if (sol.limitMaxP)
            {
                std::printf("    pMax %.6g\n", sol.pMax);
            }
            if (sol.limitMinP)
            {
                std::printf("    pMin %.6g\n", sol.pMin);
            }
        }
        if (dump.active(0))
        {
            ::mkdir(dump.dir.c_str(), 0755);
            gnc::arrayFile::Writer w(dump.path(0, "init"));
            w.text("timeIndex", "0");
            dumpState(w, sol);
            w.close();
        }
        GNC_CUDA_CHECK(cudaDeviceSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        std::printf("\nStarting time loop\n\n");

        while (sol.loop())
        {
            const int k = sol.timeIndex;
            const std::string name = timeName(sol.t, clock.c.timePrecision);
            std::printf("Time = %s\n\n", name.c_str());

            sol.solveMomentum();
            for (int d = 0; d < 3; ++d)
            {
                if (sol.lastU.solved[d])
                {
                    printPerf(sol.USolver.solver.c_str(), std::string("U") + "xyz"[d], sol.lastU.cmpt[d]);
                }
            }
            if (dump.active(k))
            {
                gnc::arrayFile::Writer w(dump.path(k, "U"));
                w.text("timeIndex", std::to_string(k));
                dumpState(w, sol);
                dumpMatrix(w, "UEqn", sol.UEqn);
                w.close();
            }

            sol.solveEnergy();
            printPerf(sol.eSolver.solver.c_str(), "e", sol.lastE);
            if (sol.limitT)
            {
                std::printf
                (
                    "limitTemperature: lower %d, upper %d\n", sol.nLimitedLow, sol.nLimitedHigh
                );
            }
            if (dump.active(k))
            {
                gnc::arrayFile::Writer w(dump.path(k, "E"));
                w.text("timeIndex", std::to_string(k));
                dumpState(w, sol);
                w.close();
            }

            sol.solvePressure();
            printPerf(sol.pSolver.solver.c_str(), "p", sol.lastP);
            if (dump.active(k))
            {
                gnc::arrayFile::Writer w(dump.path(k, "p"));
                w.text("timeIndex", std::to_string(k));
                dumpState(w, sol);
                dumpVol(w, "rAU", sol.rAU);
                dumpVol(w, "HbyA", sol.HbyA);
                dumpSurf(w, "phiHbyA", sol.phiHbyA);
                dumpSurf(w, "rhorAUf", sol.rhorAUf);
                w.close();
            }

            sol.correctTurbulence();
            printPerf(sol.sst->omegaSolver.solver.c_str(), "omega", sol.sst->lastSolveOmega);
            printPerf(sol.sst->kSolver.solver.c_str(), "k", sol.sst->lastSolveK);
            if (dump.active(k))
            {
                gnc::arrayFile::Writer w(dump.path(k, "turb"));
                w.text("timeIndex", std::to_string(k));
                dumpState(w, sol);
                w.close();
            }

            if (clock.writeTime(sol.t, k))
            {
                writeFields(caseDir, name, sol, clock.c.binary);
            }
            std::printf("\n");
        }
        if (sol.converged)
        {
            const std::string name = timeName(sol.t, clock.c.timePrecision);
            std::printf("\nSIMPLE solution converged in %s iterations\n\n", name.c_str());
            writeFields(caseDir, name, sol, clock.c.binary);
        }
        GNC_CUDA_CHECK(cudaDeviceSynchronize());
        const auto t2 = std::chrono::steady_clock::now();
        const double setup = std::chrono::duration<double>(t1 - t0).count();
        const double loopT = std::chrono::duration<double>(t2 - t1).count();
        std::printf
        (
            "iterations %d; setup %.3f s; loop %.3f s (%.6f s/iteration)\nEnd\n",
            sol.timeIndex, setup, loopT, sol.timeIndex ? loopT/sol.timeIndex : 0.0
        );
        if (profile)
        {
            const gnc::RhoSimpleSolver::Timings& tm = sol.timings;
            std::printf
            (
                "profile (s): momentum %.3f energy %.3f pressure %.3f (p solve %.3f) turbulence %.3f\n",
                tm.momentum, tm.energy, tm.pressure + tm.pressureSolve, tm.pressureSolve, tm.turbulence
            );
        }
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 1;
    }
    return 0;
}
