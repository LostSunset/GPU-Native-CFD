// N4a-2、N4b 的 L2：Spalart–Allmaras 與 kOmegaSST 的 correct() 與 correctNut()
// 對 HiSA 本身逐位元比對（模型依 turbulenceProperties）。GPU-Native-CFD，
// GPL-3.0-or-later。
//
// 參考是 HiSA 的 RAS 執行中 gncHisaDump 傾印的狀態（scripts/l2_turb_verify.sh）：
//   0.gnc   hisa -postProcess -time 0：createFields 的 turbulence->validate()
//           （correctNut）之後
//   <k>.gnc 時間步 k 結束時：外迭代 k 的 updateFields 與 turbulence->correct()
//           之後
// 時間步 k 的 turbulence->correct() 用的是外迭代 k 的 U、rho、mu、phi、
// rPseudoDeltaT（這些在 correct() 之後不再改變，所以取自 <k>.gnc），以及外迭代
// k - 1 結束時的紊流場（SA：nuTilda、nut、alphat；SST：k、omega、nut、alphat；
// 取自 <k-1>.gnc；k = 1 時為初始場與 validate() 的結果）。輸出 <out-dir>/<k>.gnc
// 只有紊流場的內部值與邊界值，<out-dir>/0.gnc 為 validate() 之後的 nut、alphat
// （SST 另有建構時 bound 過的 k、omega）。比對時以本程式的輸出為 REF（gncCompare
// 以 REF 的紀錄為準）。
//
// 用法：gncTurbL2 --case DIR --expanded DIR --ref-dir DIR --steps 1,2,10 --out-dir DIR

#include "core/arrayFile.hpp"
#include "core/gncCuda.cuh"
#include "core/glibcPow.cuh"
#include "fields/boundaryConditions.cuh"
#include "io/fvSchemes.hpp"
#include "io/polyMeshReader.hpp"
#include "mesh/cellFaceGather.hpp"
#include "mesh/deviceMesh.cuh"
#include "ops/fvcOps.cuh"
#include "solvers/gncHisa/hisaLinearSolver.cuh"
#include "solvers/gncHisa/hisaState.cuh"
#include "solvers/gncHisa/hisaViscous.cuh"
#include "turbulence/kOmegaSST.cuh"
#include "turbulence/spalartAllmaras.cuh"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace
{

namespace af = gnc::arrayFile;

__global__ void kPowSixth(const std::size_t n, const double* __restrict__ x, double* __restrict__ out)
{
    const std::size_t i = std::size_t(blockIdx.x)*blockDim.x + threadIdx.x;
    if (i < n)
    {
        out[i] = gnc::glibcPow(x[i], 1.0/6.0);
    }
}

// 診斷：本步 fw 中 pow 的底數，GPU 端（glibc 移植版）與主機 libm 的 pow 不同的個數
std::size_t powMismatches(const gnc::DeviceArray<double>& base, double& firstBase)
{
    const std::size_t n = base.size();
    gnc::DeviceArray<double> out(n);
    if (n)
    {
        kPowSixth<<<unsigned((n + 255)/256), 256>>>(n, base.data(), out.data());
        GNC_CUDA_LAUNCH_CHECK();
    }
    const std::vector<double> x = base.download(), g = out.download();
    std::size_t m = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        const double h = std::pow(x[i], 1.0/6.0);
        if (std::memcmp(&h, &g[i], 8) != 0)
        {
            if (m == 0)
            {
                firstBase = x[i];
            }
            ++m;
        }
    }
    return m;
}

struct RefFile
{
    std::map<std::string, af::Record> rec;

    explicit RefFile(const std::string& path)
    {
        for (af::Record& r : af::readAll(path))
        {
            rec[r.name] = std::move(r);
        }
    }

    std::vector<double> f64(const std::string& name, const std::uint32_t nComp) const
    {
        const auto it = rec.find(name);
        if (it == rec.end())
        {
            throw std::runtime_error("reference record " + name + " missing");
        }
        const af::Record& r = it->second;
        if (r.dtype != af::F64 || r.nComp != nComp)
        {
            throw std::runtime_error("reference record " + name + " has an unexpected type");
        }
        std::vector<double> v(r.count());
        std::memcpy(v.data(), r.bytes.data(), v.size()*sizeof(double));
        return v;
    }
};

template<class T>
void loadVol(const RefFile& f, const std::string& name, gnc::VolField<T>& v)
{
    v.in.uploadAoS(f.f64(name + ".i", gnc::NComp<T>::value));
    v.bnd.uploadAoS(f.f64(name + ".b", gnc::NComp<T>::value));
}

void loadSurf(const RefFile& f, const std::string& name, gnc::SurfField<double>& s)
{
    s.in.uploadAoS(f.f64(name + ".i", 1));
    s.bnd.uploadAoS(f.f64(name + ".b", 1));
}

void writeVol(af::Writer& w, const std::string& name, const gnc::VolField<double>& f)
{
    const std::vector<double> in = f.in.downloadAoS();
    const std::vector<double> b = f.bnd.downloadAoS();
    w.f64(name + ".i", in.data(), in.size(), 1);
    w.f64(name + ".b", b.data(), b.size(), 1);
}

std::vector<int> parseList(const std::string& s)
{
    std::vector<int> out;
    std::stringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ','))
    {
        if (!tok.empty())
        {
            out.push_back(std::stoi(tok));
        }
    }
    return out;
}

bool exists(const std::string& p)
{
    std::ifstream is(p);
    return bool(is);
}

// 模型的場（依 turbulenceProperties：SpalartAllmaras 或 kOmegaSST）
struct Model
{
    std::unique_ptr<gnc::SpalartAllmaras> sa;
    std::unique_ptr<gnc::KOmegaSST> sst;

    std::vector<std::pair<std::string, gnc::VolField<double>*>> fields()
    {
        if (sa)
        {
            return {{"nuTilda", &sa->nuTilda}, {"nut", &sa->nut}, {"alphat", &sa->alphat}};
        }
        return {{"k", &sst->k}, {"omega", &sst->omega}, {"nut", &sst->nut}, {"alphat", &sst->alphat}};
    }
};

struct Snapshot
{
    std::vector<std::vector<double>> in, b;
};

Snapshot snap(Model& m)
{
    Snapshot s;
    for (const auto& f : m.fields())
    {
        s.in.push_back(f.second->in.downloadAoS());
        s.b.push_back(f.second->bnd.downloadAoS());
    }
    return s;
}

void restore(Model& m, const Snapshot& s)
{
    std::size_t i = 0;
    for (const auto& f : m.fields())
    {
        f.second->in.uploadAoS(s.in[i]);
        f.second->bnd.uploadAoS(s.b[i]);
        ++i;
    }
}

} // namespace


int main(int argc, char* argv[])
{
    std::string caseDir, expanded, refDir, outDir, steps;
    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string a = argv[i];
        if (a == "--case") caseDir = argv[i + 1];
        else if (a == "--expanded") expanded = argv[i + 1];
        else if (a == "--ref-dir") refDir = argv[i + 1];
        else if (a == "--out-dir") outDir = argv[i + 1];
        else if (a == "--steps") steps = argv[i + 1];
        else
        {
            std::fprintf(stderr, "unknown option %s\n", a.c_str());
            return 2;
        }
    }
    if (caseDir.empty() || expanded.empty() || refDir.empty() || outDir.empty() || steps.empty())
    {
        std::fprintf(stderr, "usage: gncTurbL2 --case DIR --expanded DIR --ref-dir DIR --steps 1,2 --out-dir DIR\n");
        return 2;
    }

    try
    {
        const gnc::PolyMesh mesh = gnc::readPolyMesh(caseDir + "/constant/polyMesh");
        const gnc::CellFaceGather gather = gnc::buildCellFaceGather(mesh);
        gnc::DeviceMesh dm = gnc::DeviceMesh::upload(mesh, gather);
        gnc::computeGeometry(dm);
        const gnc::FvSchemes schemes(expanded + "/fvSchemes");

        gnc::HisaState s = gnc::hisaInitialise(dm, mesh, expanded);
        const gnc::LSVectors ls = gnc::faceLeastSquaresVectors(dm);
        const gnc::LusgsPlan plan = gnc::buildLusgsPlan(mesh);
        Model model;
        const gnc::TurbulenceModelKind kind =
            gnc::readTurbulenceModelKind(expanded + "/turbulenceProperties");
        if (kind == gnc::TurbulenceModelKind::spalartAllmaras)
        {
            model.sa = std::make_unique<gnc::SpalartAllmaras>
            (
                gnc::createSpalartAllmaras(dm, mesh, expanded, schemes)
            );
            gnc::SpalartAllmaras& sa = *model.sa;
            sa.debugPow = true;
            std::printf
            (
                "mesh: %d cells; SpalartAllmaras Cw1 %.17g; nuTilda solver tol %g relTol %g minIter %d; relax %g\n",
                dm.nCells, sa.c.Cw1, sa.solver.tolerance, sa.solver.relTol, sa.solver.minIter,
                sa.relax ? sa.relaxFactor : 0.0
            );
            // validate()：初始的 rho、mu
            gnc::spalartAllmarasCorrectNut(dm, sa, s.rho, s.thermo.mu, s.U);
        }
        else if (kind == gnc::TurbulenceModelKind::kOmegaSST)
        {
            model.sst = std::make_unique<gnc::KOmegaSST>
            (
                gnc::createKOmegaSST(dm, mesh, expanded, schemes)
            );
            gnc::KOmegaSST& m = *model.sst;
            std::printf
            (
                "mesh: %d cells; kOmegaSST betaStar %.17g; omegaWall cells %d; k solver tol %g relTol %g; "
                "relax k %g omega %g\n",
                dm.nCells, m.c.betaStar, m.nWallCells, m.kSolver.tolerance, m.kSolver.relTol,
                m.kRelax ? m.kRelaxFactor : 0.0, m.omegaRelax ? m.omegaRelaxFactor : 0.0
            );
            // validate()
            gnc::kOmegaSSTCorrectNut(dm, ls, m, s.rho, s.thermo.mu, s.U);
        }
        else
        {
            throw std::runtime_error("turbulenceProperties: not a RAS model");
        }
        const Snapshot initial = snap(model);
        {
            af::Writer w(outDir + "/0.gnc");
            w.text("timeIndex", "0");
            for (const auto& f : model.fields())
            {
                // SA 的 0.gnc 只有 nut、alphat（validate 不改 nuTilda）
                if (model.sa && f.first == "nuTilda")
                {
                    continue;
                }
                writeVol(w, f.first, *f.second);
            }
            w.close();
        }

        for (const int k : parseList(steps))
        {
            const RefFile cur(refDir + "/" + std::to_string(k) + ".gnc");
            loadVol(cur, "U", s.U);
            loadVol(cur, "rho", s.rho);
            loadVol(cur, "mu", s.thermo.mu);
            loadSurf(cur, "phi", s.phi);
            gnc::DeviceArray<double> rPDT;
            rPDT.upload(cur.f64("rPseudoDeltaT.i", 1));

            const std::string prevPath = refDir + "/" + std::to_string(k - 1) + ".gnc";
            if (k > 1)
            {
                if (!exists(prevPath))
                {
                    throw std::runtime_error("step " + std::to_string(k) + " needs " + prevPath);
                }
                const RefFile prev(prevPath);
                for (const auto& f : model.fields())
                {
                    loadVol(prev, f.first, *f.second);
                }
            }
            else
            {
                restore(model, initial);
            }

            af::Writer w(outDir + "/" + std::to_string(k) + ".gnc");
            w.text("timeIndex", std::to_string(k));
            char buf[256];
            if (model.sa)
            {
                gnc::SpalartAllmaras& sa = *model.sa;
                gnc::spalartAllmarasCorrect(dm, plan, ls, sa, s.rho, s.U, s.thermo.mu, s.phi, rPDT.data());
                GNC_CUDA_CHECK(cudaDeviceSynchronize());
                std::snprintf
                (
                    buf, sizeof(buf), "initial %.17g final %.17g iterations %d bounded %d",
                    sa.lastSolve.initialResidual, sa.lastSolve.finalResidual,
                    sa.lastSolve.nIterations, int(sa.lastBounded)
                );
                w.text("nuTilda.solve", buf);
                std::printf("step %d: smoothSolver: Solving for nuTilda, %s\n", k, buf);
            }
            else
            {
                gnc::KOmegaSST& m = *model.sst;
                gnc::kOmegaSSTCorrect(dm, plan, ls, m, s.rho, s.U, s.thermo.mu, s.phi, rPDT.data());
                GNC_CUDA_CHECK(cudaDeviceSynchronize());
                const gnc::SolverPerf* perf[2] = {&m.lastSolveOmega, &m.lastSolveK};
                const bool bounded[2] = {m.lastBoundedOmega, m.lastBoundedK};
                const char* nm[2] = {"omega", "k"};
                for (int i = 0; i < 2; ++i)
                {
                    std::snprintf
                    (
                        buf, sizeof(buf), "initial %.17g final %.17g iterations %d bounded %d",
                        perf[i]->initialResidual, perf[i]->finalResidual, perf[i]->nIterations,
                        int(bounded[i])
                    );
                    w.text(std::string(nm[i]) + ".solve", buf);
                    std::printf("step %d: smoothSolver: Solving for %s, %s\n", k, nm[i], buf);
                }
            }
            for (const auto& f : model.fields())
            {
                writeVol(w, f.first, *f.second);
            }
            w.close();
            if (model.sa)
            {
                double first = 0;
                const std::size_t pm = powMismatches(model.sa->debugPowBase, first);
                std::printf
                (
                    "step %d: fw pow(base, 1/6): %zu of %d bases differ from host libm (first %.17g)\n",
                    k, pm, dm.nCells, first
                );
            }
        }
        return 0;
    }
    catch (const std::exception& e)
    {
        std::fprintf(stderr, "ERROR: %s\n", e.what());
        return 2;
    }
}
