// 測試程式共用：傾印 gncHisa 的狀態，紀錄名稱與 tools/ref/gncHisaDump
// 相同（fields：p T U e thermo:psi thermo:mu thermo:alpha rho rhoU rhoE phi
// pseudoCoDamping pseudoCoNum，Spalart–Allmaras 時另有 nuTilda nut alphat）。
// GPU-Native-CFD，GPL-3.0-or-later。

#ifndef GNC_TESTS_HISASTATEDUMP_CUH
#define GNC_TESTS_HISASTATEDUMP_CUH

#include "core/arrayFile.hpp"
#include "solvers/gncHisa/hisaSolver.cuh"

#include <string>
#include <vector>

namespace gncTest
{

namespace af = gnc::arrayFile;

template<class T>
void dumpDF(af::Writer& w, const std::string& name, const gnc::DField<T>& f)
{
    const std::vector<double> h = f.downloadAoS();
    w.f64(name, h.data(), f.size(), gnc::NComp<T>::value);
}

template<class F>
void dumpGeo(af::Writer& w, const std::string& name, const F& f)
{
    dumpDF(w, name + ".i", f.in);
    dumpDF(w, name + ".b", f.bnd);
}

inline void dumpHisaState(af::Writer& w, const gnc::HisaSolver& sol)
{
    const gnc::HisaState& s = sol.s;
    w.text("timeIndex", std::to_string(sol.timeIndex));
    const gnc::ThermoParams& P = s.thermo.params;
    w.f64("RR", &P.RR, 1);
    w.f64("Tstd", &P.Tstd, 1);
    dumpGeo(w, "p", s.thermo.p);
    dumpGeo(w, "T", s.thermo.T);
    dumpGeo(w, "U", s.U);
    dumpGeo(w, "e", s.thermo.he);
    dumpGeo(w, "psi", s.thermo.psi);
    dumpGeo(w, "mu", s.thermo.mu);
    dumpGeo(w, "alpha", s.thermo.alpha);
    dumpGeo(w, "rho", s.rho);
    dumpGeo(w, "rhoU", s.rhoU);
    dumpGeo(w, "rhoE", s.rhoE);
    dumpGeo(w, "phi", sol.phi());
    dumpGeo(w, "pseudoCoDamping", sol.pt.pseudoCoDamping);
    w.f64("pseudoCoNum", &sol.pt.pseudoCoNum, 1);
    if (sol.sa)
    {
        dumpGeo(w, "nuTilda", sol.sa->nuTilda);
        dumpGeo(w, "nut", sol.sa->nut);
        dumpGeo(w, "alphat", sol.sa->alphat);
    }
    if (sol.sst)
    {
        dumpGeo(w, "k", sol.sst->k);
        dumpGeo(w, "omega", sol.sst->omega);
        dumpGeo(w, "nut", sol.sst->nut);
        dumpGeo(w, "alphat", sol.sst->alphat);
    }
}

template<class T>
void dumpBlock(af::Writer& w, const std::string& name, const gnc::FvjBlock<T>& b)
{
    dumpDF(w, "J." + name + ".diag", b.diag);
    if (b.hasOffDiag)
    {
        dumpDF(w, "J." + name + ".upper", b.upper);
        dumpDF(w, "J." + name + ".lower", b.lower);
    }
}

// 每格 5 個值 (rho, rhoU.x, rhoU.y, rhoU.z, rhoE)，與 gncLusgsDump 的 writePacked 相同
inline void dumpPacked(af::Writer& w, const std::string& name, const gnc::HisaVec& x)
{
    const std::vector<double> s0 = x.s0.downloadAoS();
    const std::vector<double> s1 = x.s1.downloadAoS();
    const std::vector<double> v = x.v.downloadAoS();
    const std::size_t n = s0.size();
    std::vector<double> packed(5*n);
    for (std::size_t i = 0; i < n; ++i)
    {
        packed[5*i] = s0[i];
        packed[5*i + 1] = v[3*i];
        packed[5*i + 2] = v[3*i + 1];
        packed[5*i + 3] = v[3*i + 2];
        packed[5*i + 4] = s1[i];
    }
    w.f64(name, packed.data(), n, 5);
}

// Jacobian 組裝時（GMRES 之前）的診斷紀錄，與 gncLusgsDump 的 dbg.* 相同：
// 各場的邊界值、phi 的邊界值、p 的 fixedGradient 族 patch 的 gradient
struct DbgSnapshot
{
    std::vector<std::string> names;
    std::vector<std::vector<double>> data;
    std::vector<int> nComp;

    template<class T>
    void add(const std::string& name, const gnc::DField<T>& f)
    {
        names.push_back(name);
        data.push_back(f.downloadAoS());
        nComp.push_back(gnc::NComp<T>::value);
    }

    void write(af::Writer& w) const
    {
        for (std::size_t i = 0; i < names.size(); ++i)
        {
            w.f64(names[i], data[i].data(), data[i].size()/std::size_t(nComp[i]), nComp[i]);
        }
    }
};

inline DbgSnapshot jacobianTimeSnapshot
(
    const gnc::HisaSolver& sol,
    const gnc::OuterIterationView& v
)
{
    DbgSnapshot d;
    d.add("dbg.U.b", v.s.U.bnd);
    d.add("dbg.p.b", v.s.thermo.p.bnd);
    d.add("dbg.T.b", v.s.thermo.T.bnd);
    d.add("dbg.rho.b", v.s.rho.bnd);
    d.add("dbg.rhoE.b", v.s.rhoE.bnd);
    d.add("dbg.e.b", v.s.thermo.he.bnd);
    d.add("dbg.psi.b", v.s.thermo.psi.bnd);
    d.add("dbg.rhoU.b", v.s.rhoU.bnd);
    d.add("dbg.phi.b", sol.phi().bnd);
    std::vector<double> g;
    for (std::size_t pi = 0; pi < sol.dm.patches.size(); ++pi)
    {
        const auto& bc = v.s.thermo.p.bc[pi];
        if (sol.dm.fvPatchSize[pi] > 0 && gnc::isFixedGradientFamily(bc.type))
        {
            const std::vector<double> h = bc.gradient.downloadAoS();
            g.insert(g.end(), h.begin(), h.end());
        }
    }
    d.names.push_back("dbg.p.gradient");
    d.data.push_back(g);
    d.nComp.push_back(1);
    return d;
}

// gncLUSGSDump 的 jacobian、precondition、solve 三份紀錄（寫在同一個檔），
// 在 GMRES 剛結束時呼叫：Jacobian 與殘差在 GMRES 中不變；第一次
// precondition() 的輸出另算一份，不影響求解；dbg 是 Jacobian 組裝時的快照
inline void dumpOuterIteration
(
    af::Writer& w,
    const gnc::HisaSolver& sol,
    const gnc::OuterIterationView& v,
    const DbgSnapshot& dbg
)
{
    gnc::HisaVec P;
    P.s0.copyFrom(v.R.s0);
    P.s1.copyFrom(v.R.s1);
    P.v.copyFrom(v.R.v);
    gnc::lusgsPrecondition(sol.dm, v.J, v.plan, P);

    dumpBlock(w, "SS00", v.J.SS00);
    dumpBlock(w, "SS01", v.J.SS01);
    dumpBlock(w, "SV00", v.J.SV00);
    dumpBlock(w, "SS10", v.J.SS10);
    dumpBlock(w, "SS11", v.J.SS11);
    dumpBlock(w, "SV10", v.J.SV10);
    dumpBlock(w, "VS00", v.J.VS00);
    dumpBlock(w, "VS01", v.J.VS01);
    dumpBlock(w, "VV00", v.J.VV00);
    dumpDF(w, "rPseudoDeltaT", v.pt.rPseudoDeltaT.in);
    dumpDF(w, "rDiagCoeff", v.J.rDiagCoeff);
    {
        const std::vector<double> V = sol.dm.V.download();
        w.f64("V", V.data(), V.size(), 1);
    }
    dbg.write(w);
    dumpPacked(w, "P.in", v.R);
    dumpPacked(w, "P.out", P);
    dumpDF(w, "rho.i", v.s.rho.in);
    dumpDF(w, "rho.b", v.s.rho.bnd);
    dumpDF(w, "rhoE.i", v.s.rhoE.in);
    dumpDF(w, "rhoE.b", v.s.rhoE.bnd);
    dumpDF(w, "rhoU.i", v.s.rhoU.in);
    dumpDF(w, "rhoU.b", v.s.rhoU.bnd);
}

} // namespace gncTest

#endif
