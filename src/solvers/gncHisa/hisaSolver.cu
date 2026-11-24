// GPU-Native-CFD，GPL-3.0-or-later。見 hisaSolver.cuh。

#include "solvers/gncHisa/hisaSolver.cuh"

#include "io/dictionary.hpp"
#include "io/foamTokenizer.hpp"
#include "solvers/gncHisa/hisaBoundaryUpdate.cuh"
#include "solvers/gncHisa/hisaViscous.cuh"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace gnc
{

namespace
{

constexpr double VGREAT = 1.0e+300;

// OpenFOAM 純量的 max、min（primitives/ints/int/int.H 的 MAXMIN）
double hmax(const double s1, const double s2)
{
    return (s2 < s1) ? s1 : s2;
}

double hmin(const double s1, const double s2)
{
    return (s1 < s2) ? s1 : s2;
}

// residualIO 的 max：rho、rhoE，再 rhoU 的 cmptMax（r 的次序 rho、Ux、Uy、Uz、rhoE）
double residualMax(const double* r)
{
    double m = -VGREAT;
    m = hmax(m, r[0]);
    m = hmax(m, r[4]);
    m = hmax(m, hmax(hmax(r[1], r[2]), r[3]));
    return m;
}

// residualIO 的讀入：rho、(rhoU)、rhoE，括號略過
void readResidualIO(const Dictionary& d, const std::string& key, const std::string& path, double* out)
{
    const Dictionary::Entry* e = d.find(key);
    if (!e)
    {
        return;
    }
    if (e->dict)
    {
        throw std::runtime_error(path + ": " + key + " must be a value");
    }
    std::vector<double> v;
    for (const Token& t : e->stream)
    {
        if (t.isPunct('(') || t.isPunct(')'))
        {
            continue;
        }
        double x = 0;
        if (!parseScalar(t.text, x))
        {
            throw std::runtime_error(path + ": " + key + ": invalid number " + t.text);
        }
        v.push_back(x);
    }
    if (v.size() != 5)
    {
        throw std::runtime_error(path + ": " + key + " needs 5 values");
    }
    for (int i = 0; i < 5; ++i)
    {
        out[i] = v[std::size_t(i)];
    }
}

// pseudotimeControl::criteriaSatisfied
bool criteriaSatisfied(const PseudotimeControls& c, PseudotimeState& s)
{
    const bool finalIter = s.converged || (s.corr == c.nPseudoCorr);
    if (s.firstIteration || s.corr == 1 || finalIter)
    {
        s.firstIteration = false;
        return false;
    }
    const bool storeIni = (s.corr == 2);
    if (storeIni)
    {
        // initResiduals_ = residuals_；state_.add("initResiduals", ...)
        for (int k = 0; k < 5; ++k)
        {
            s.initResiduals[k] = s.residuals[k];
            s.stateInit[k] = s.residuals[k];
        }
        if (!s.firstValid)
        {
            for (int k = 0; k < 5; ++k)
            {
                s.firstResiduals[k] = s.residuals[k];
                s.stateFirst[k] = s.residuals[k];
            }
            s.firstValid = true;
        }
        s.stateStored = true;
    }
    double d[5];
    for (int k = 0; k < 5; ++k)
    {
        d[k] = s.residuals[k] - c.tol[k];
    }
    const bool absCheck = residualMax(d) < 0.0;
    bool relCheck = false;
    if (!storeIni)
    {
        for (int k = 0; k < 5; ++k)
        {
            d[k] = s.residuals[k]/(s.initResiduals[k] + ROOTVSMALL) - c.tolRel[k];
        }
        relCheck = residualMax(d) < 0.0;
    }
    return (s.corr >= c.nPseudoCorrMin) && (absCheck || relCheck);
}

void requireSteady(const FvSchemes& schemes)
{
    std::istringstream is(schemes.ddt("default"));
    std::string w, last;
    while (is >> w)
    {
        last = w;
    }
    if (last != "steadyState")
    {
        throw std::runtime_error("only steady-state cases are supported (N-015)");
    }
}

} // namespace


PseudotimeControls readPseudotimeControls(const std::string& path)
{
    FoamTokenizer t(path);
    const Dictionary root = Dictionary::parse(t, path, false);
    const Dictionary::Entry* e = root.find("pseudoTime");
    const Dictionary empty;
    const Dictionary& d = (e && e->dict) ? *e->dict : empty;
    PseudotimeControls c;
    if (d.found("nPseudoCorr")) c.nPseudoCorr = int(d.getLabel("nPseudoCorr"));
    if (d.found("nPseudoCorrMin")) c.nPseudoCorrMin = int(d.getLabel("nPseudoCorrMin"));
    readResidualIO(d, "pseudoTol", path, c.tol);
    readResidualIO(d, "pseudoTolRel", path, c.tolRel);
    return c;
}


double streamRoundTrip(const double v, const int precision)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*g", precision, v);
    double out = 0;
    if (!parseScalar(buf, out))
    {
        throw std::runtime_error(std::string("streamRoundTrip: cannot parse ") + buf);
    }
    return out;
}


bool pseudotimeLoop(const PseudotimeControls& c, PseudotimeState& s, const int precision)
{
    // read()：由 state_ 讀回 initResiduals、firstResiduals
    if (s.stateStored)
    {
        for (int k = 0; k < 5; ++k)
        {
            s.initResiduals[k] = streamRoundTrip(s.stateInit[k], precision);
            s.firstResiduals[k] = streamRoundTrip(s.stateFirst[k], precision);
        }
    }
    ++s.corr;
    bool completed = false;
    if (s.converged || criteriaSatisfied(c, s))
    {
        if (s.converged)
        {
            s.corr = 0;
            s.converged = false;
            completed = true;
        }
        else
        {
            // storePrevIterFields：只有設了 relaxationFactors 的場，HiSA 的平均流不用
            s.converged = true;
        }
    }
    return !completed;
}


TimeControls readTimeControls(const std::string& path)
{
    FoamTokenizer t(path);
    const Dictionary d = Dictionary::parse(t, path, false);
    const std::string startFrom = d.getWordOrDefault("startFrom", "latestTime");
    if (startFrom != "startTime")
    {
        throw std::runtime_error(path + ": startFrom " + startFrom + " is not supported (startTime)");
    }
    const std::string stopAt = d.getWordOrDefault("stopAt", "endTime");
    if (stopAt != "endTime")
    {
        throw std::runtime_error(path + ": stopAt " + stopAt + " is not supported (endTime)");
    }
    TimeControls c;
    c.startTime = d.getScalar("startTime");
    c.endTime = d.getScalar("endTime");
    c.deltaT = d.getScalar("deltaT");
    if (d.found("writePrecision"))
    {
        c.writePrecision = int(d.getLabel("writePrecision"));
    }
    if (c.startTime != 0.0)
    {
        throw std::runtime_error(path + ": only startTime 0 is supported (no restart)");
    }
    if (!(c.deltaT > 0))
    {
        throw std::runtime_error(path + ": deltaT must be positive");
    }
    return c;
}


HisaSolver::HisaSolver
(
    const std::string& caseDir,
    const std::string& expandedDir,
    const FoamCompat& compatIn
)
:
    compat(compatIn)
{
    mesh = readPolyMesh(caseDir + "/constant/polyMesh");
    const CellFaceGather gather = buildCellFaceGather(mesh);
    dm = DeviceMesh::upload(mesh, gather);
    computeGeometry(dm);

    schemes = std::make_unique<FvSchemes>(expandedDir + "/fvSchemes");
    requireSteady(*schemes);
    ptc = readPseudoTimeControls(expandedDir + "/fvSolution");
    gmresControls = readGmresControls(expandedDir + "/fvSolution");
    pseudotime = readPseudotimeControls(expandedDir + "/fvSolution");

    s = hisaInitialise(dm, mesh, expandedDir);
    ls_ = faceLeastSquaresVectors(dm);
    const TurbulenceModelKind kind = s.inviscid
        ? TurbulenceModelKind::laminarStokes
        : readTurbulenceModelKind(expandedDir + "/turbulenceProperties");
    if (kind == TurbulenceModelKind::spalartAllmaras)
    {
        sa = std::make_unique<SpalartAllmaras>
        (
            createSpalartAllmaras(dm, mesh, expandedDir, *schemes)
        );
        s.nut = &sa->nut;
        s.alphat = &sa->alphat;
        // createFields 的 turbulence->validate()
        spalartAllmarasCorrectNut(dm, *sa, s.rho, s.thermo.mu, s.U);
    }
    else if (kind == TurbulenceModelKind::kOmegaSST)
    {
        sst = std::make_unique<KOmegaSST>
        (
            createKOmegaSST(dm, mesh, expandedDir, *schemes)
        );
        s.nut = &sst->nut;
        s.alphat = &sst->alphat;
        // createFields 的 turbulence->validate()
        kOmegaSSTCorrectNut(dm, ls_, *sst, s.rho, s.thermo.mu, s.U);
    }
    plan_ = buildLusgsPlan(mesh);
    pt = initPseudoTime(dm, ptc);
}


bool HisaSolver::running() const
{
    return !ended && (t < (time.endTime - 0.5*time.deltaT));
}


const SurfField<double>& HisaSolver::phi() const
{
    return fluxValid_ ? flux_.phi : s.phi;
}


bool HisaSolver::timeStep
(
    const AfterSolveHook& afterSolve,
    const AfterSolveHook& afterJacobian
)
{
    t = t + time.deltaT;
    ++timeIndex;
    // beginTimeStep
    if (ptc.resetPseudo)
    {
        pt.pseudoCoNum = ptc.pseudoCoNum;
    }
    if (!pseudotimeLoop(pseudotime, control, time.writePrecision))
    {
        ended = true;
        return false;
    }
    outerIteration(afterSolve, afterJacobian);
    return true;
}


void HisaSolver::setPseudoCoNum()
{
    if (!lastGmresValid)
    {
        return;
    }
    if (control.corr != 1 && control.firstValid)
    {
        const double* r = lastGmres.initRes;
        const double* f = control.firstResiduals;
        const double normCurr =
            std::sqrt((r[0]*r[0] + r[4]*r[4]) + ((r[1]*r[1] + r[2]*r[2]) + r[3]*r[3]));
        const double normFirst =
            std::sqrt((f[0]*f[0] + f[4]*f[4]) + ((f[1]*f[1] + f[2]*f[2]) + f[3]*f[3]));
        double ratio = normFirst/normCurr/pt.pseudoCoNum*ptc.pseudoCoNum;
        ratio = hmax
        (
            hmin(ratio, ptc.pseudoCoNumMaxIncreaseFactor),
            ptc.pseudoCoNumMinDecreaseFactor
        );
        pt.pseudoCoNum = pt.pseudoCoNum*ratio;
        pt.pseudoCoNum = hmax(hmin(pt.pseudoCoNum, ptc.pseudoCoNumMax), ptc.pseudoCoNumMin);
    }
}


void HisaSolver::outerIteration
(
    const AfterSolveHook& afterSolve,
    const AfterSolveHook& afterJacobian
)
{
    using Clock = std::chrono::steady_clock;
    Clock::time_point tp = Clock::now();
    const auto lap = [&](double& acc)
    {
        if (profile)
        {
            GNC_CUDA_CHECK(cudaDeviceSynchronize());
            const Clock::time_point now = Clock::now();
            acc += std::chrono::duration<double>(now - tp).count();
            tp = now;
        }
    };

    // 通量；residualsUpdate.H：fvOptions(rho, U)、fvOptions(rho, he) 建構 fvMatrix
    // 時的邊界更新，再算殘差
    ausmPlusUpFlux(dm, ls_, *schemes, s, flux_);
    fluxValid_ = true;
    updateCoeffsU(dm, s, flux_);
    if (!energyBCsUpdated_)
    {
        updateEnergyT(dm, s, flux_);
        energyBCsUpdated_ = true;
    }
    lap(timings.flux);
    inviscidResiduals(dm, s, flux_, res_);
    if (!s.inviscid)
    {
        addLaminarViscousResiduals(dm, ls_, *schemes, s, flux_, compat, res_, tauMC_);
    }
    lap(timings.residual);

    setPseudoCoNum();
    setPseudoDeltaT(dm, *schemes, s, pt);
    lap(timings.pseudoTime);

    // 求解區塊：jacobian::addBoundaryTerms 開頭的 p、U、T 的 updateCoeffs()（U 在
    // fvOptions(rho, U) 已更新，直接返回）、Jacobian、
    // scalarVarsPrevIter_ 等（求解前的值）、GMRES
    updateCoeffsP(dm, s, flux_, compat);
    updateCoeffsT(dm, s, flux_);
    assembleJacobian(dm, *schemes, s, pt, J_);
    const PrevIterFields prev = storePrevIter(s);
    HisaVec R;
    R.s0.copyFrom(res_.rhoR.in);
    R.s1.copyFrom(res_.rhoER.in);
    R.v.copyFrom(res_.rhoUR.in);
    if (afterJacobian)
    {
        afterJacobian(OuterIterationView{J_, plan_, R, pt, s, lastGmres});
    }
    lap(timings.jacobian);
    lastGmres = gmresSolve(dm, J_, plan_, gmresControls, R, s.rho, s.rhoU, s.rhoE);
    lastGmresValid = true;
    for (int k = 0; k < 5; ++k)
    {
        control.residuals[k] = lastGmres.initRes[k];
    }
    if (afterSolve)
    {
        afterSolve(OuterIterationView{J_, plan_, R, pt, s, lastGmres});
    }
    lap(timings.gmres);

    // 限界、updateFields、turbulence->correct()（laminar 不改場）
    boundSolution(dm, ptc, prev, s, pt);
    updateFields(dm, ptc, prev, flux_, s);
    lap(timings.update);
    if (sa)
    {
        spalartAllmarasCorrect
        (
            dm, plan_, ls_, *sa, s.rho, s.U, s.thermo.mu, flux_.phi,
            pt.rPseudoDeltaT.in.cview().p[0]
        );
    }
    else if (sst)
    {
        kOmegaSSTCorrect
        (
            dm, plan_, ls_, *sst, s.rho, s.U, s.thermo.mu, flux_.phi,
            pt.rPseudoDeltaT.in.cview().p[0]
        );
    }
    lap(timings.turbulence);
    ++timings.n;
    ++nOuterIterations;
}

} // namespace gnc
