// GPU-Native-CFD，GPL-3.0-or-later。見 forceCoeffs.cuh。

#include "post/forceCoeffs.cuh"

#include "io/dictionary.hpp"
#include "io/foamTokenizer.hpp"
#include "ops/fvcOps.cuh"
#include "solvers/gncHisa/hisaSolver.cuh"

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <set>
#include <stdexcept>
#include <sys/stat.h>

namespace gnc
{

namespace
{

[[noreturn]] void fail(const std::string& fn, const std::string& msg)
{
    throw std::runtime_error("functions/" + fn + ": " + msg);
}

double tokenScalar(const Token& t, const std::string& fn)
{
    double v = 0;
    if (t.kind != Token::WORD || !parseScalar(t.text, v))
    {
        fail(fn, "expected a number, got '" + t.text + "'");
    }
    return v;
}

// "( a b c )"
vec3 getVector(const Dictionary& d, const std::string& key, const std::string& fn)
{
    const Dictionary::Entry* e = d.find(key);
    if (!e)
    {
        fail(fn, "missing " + key);
    }
    const std::vector<Token>& s = e->stream;
    if (s.size() != 5 || !s[0].isPunct('(') || !s[4].isPunct(')'))
    {
        fail(fn, key + " is not a vector");
    }
    return {tokenScalar(s[1], fn), tokenScalar(s[2], fn), tokenScalar(s[3], fn)};
}

// "( w1 w2 ... )"
std::vector<std::string> getWords(const Dictionary& d, const std::string& key, const std::string& fn)
{
    const Dictionary::Entry* e = d.find(key);
    if (!e)
    {
        fail(fn, "missing " + key);
    }
    const std::vector<Token>& s = e->stream;
    if (s.size() < 2 || !s.front().isPunct('(') || !s.back().isPunct(')'))
    {
        fail(fn, key + " is not a list");
    }
    std::vector<std::string> out;
    for (std::size_t i = 1; i + 1 < s.size(); ++i)
    {
        if (s[i].kind == Token::STRING)
        {
            fail(fn, key + ": regular expressions are not supported");
        }
        if (s[i].kind != Token::WORD)
        {
            fail(fn, key + ": unexpected token '" + s[i].text + "'");
        }
        out.push_back(s[i].text);
    }
    return out;
}

// coordinateRotations::axes::rotation(axis1, axis2, E3_E1)（axesRotation.C:68）：
// axis1 = e3、axis2 = e1；R 的行為 (ax2, ax1 ^ ax2, ax1)
tens axesRotationE3E1(const vec3& axis1, const vec3& axis2, const std::string& fn)
{
    const double magAxis1 = mag(axis1);
    if (magAxis1 < ROOTVSMALL)
    {
        fail(fn, "dominant coordinate axis cannot have zero length");
    }
    const vec3 ax1 = axis1/magAxis1;
    vec3 ax2 = axis2;
    if (mag(axis2) < ROOTVSMALL)
    {
        fail(fn, "zero second axis is not supported");
    }
    // removeCollinear：*this -= (*this & unitVec)*unitVec
    ax2 -= dot(ax2, ax1)*ax1;
    const double magAxis2 = mag(ax2);
    if (magAxis2 < SMALL)
    {
        fail(fn, "axes are co-linear");
    }
    ax2 = {ax2.x/magAxis2, ax2.y/magAxis2, ax2.z/magAxis2};   // ax2 /= magAxis2
    const vec3 c1 = cross(ax1, ax2);
    tens R{};
    R.xx = ax2.x; R.yx = ax2.y; R.zx = ax2.z;   // col<0>
    R.xy = c1.x; R.yy = c1.y; R.zy = c1.z;      // col<1>
    R.xz = ax1.x; R.yz = ax1.y; R.zz = ax1.z;   // col<2>
    return R;
}

std::vector<int> patchSet(const PolyMesh& mesh, const std::vector<std::string>& names, const std::string& fn)
{
    std::set<int> ids;
    for (const std::string& n : names)
    {
        bool hit = false;
        for (std::size_t pi = 0; pi < mesh.patches.size(); ++pi)
        {
            const Patch& p = mesh.patches[pi];
            if
            (
                p.name == n
             || std::find(p.inGroups.begin(), p.inGroups.end(), n) != p.inGroups.end()
            )
            {
                ids.insert(int(pi));
                hit = true;
            }
        }
        if (!hit)
        {
            fail(fn, "patch " + n + " not found");
        }
    }
    return std::vector<int>(ids.begin(), ids.end());
}

// forceCoeffs::coeffDesc（forceCoeffs.H:321）：c1 = -1 為力矩
struct CoeffDesc
{
    std::string name;
    int c0;
    int c1;
    int split;  // 0 none、1 front、2 rear
};

std::vector<CoeffDesc> selectCoeffs()
{
    std::vector<CoeffDesc> c =
    {
        {"Cd", 0, 0, 0}, {"Cs", 1, 2, 0}, {"Cl", 2, 1, 0}
    };
    const std::vector<CoeffDesc> base = c;
    for (const CoeffDesc& d : base)
    {
        c.push_back({d.name + "(f)", d.c0, d.c1, 1});
        c.push_back({d.name + "(r)", d.c0, d.c1, 2});
    }
    c.push_back({"CmRoll", 0, -1, 0});
    c.push_back({"CmPitch", 1, -1, 0});
    c.push_back({"CmYaw", 2, -1, 0});
    std::sort
    (
        c.begin(), c.end(),
        [](const CoeffDesc& a, const CoeffDesc& b) { return a.name < b.name; }
    );
    return c;
}

// 以 precision 位寫成文字（std::ostream 的一般格式，即 %.*g）再以 OpenFOAM 的規則讀回
double textRound(const double v, const int precision)
{
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*g", precision, v);
    double r = 0;
    if (!parseScalar(buf, r))
    {
        throw std::runtime_error(std::string("textRound: cannot read back ") + buf);
    }
    return r;
}

vec3 textRound(const vec3& v, const int precision)
{
    return {textRound(v.x, precision), textRound(v.y, precision), textRound(v.z, precision)};
}

// word operator&（word.C:114）：a + b，b 的第一個字元大寫
std::string camel(const std::string& a, const std::string& b)
{
    std::string s = a + b;
    s[a.size()] = char(std::toupper(static_cast<unsigned char>(b[0])));
    return s;
}

} // namespace


std::vector<ForceFunction> readForceFunctions
(
    const std::string& controlDictFile,
    const PolyMesh& mesh
)
{
    FoamTokenizer t(controlDictFile);
    const Dictionary root = Dictionary::parse(t, controlDictFile, false);
    std::vector<ForceFunction> out;
    const int precision = root.found("writePrecision") ? int(root.getLabel("writePrecision")) : 6;
    const Dictionary::Entry* fe = root.find("functions");
    if (!fe || !fe->dict)
    {
        return out;
    }
    for (const Dictionary::Entry& e : fe->dict->entries())
    {
        if (!e.dict || !e.dict->found("type"))
        {
            continue;
        }
        const Dictionary& d = *e.dict;
        const std::string type = d.getWord("type");
        if (type != "forces" && type != "forceCoeffs")
        {
            continue;
        }
        const std::string& fn = e.key;
        ForceFunction f;
        f.name = fn;
        f.coeffs = type == "forceCoeffs";
        for (const char* k : {"porosity", "directForceDensity", "coordinateSystem"})
        {
            if (d.found(k) && !(std::string(k) != "coordinateSystem" && d.getWord(k) == "false"))
            {
                fail(fn, std::string(k) + " is not supported");
            }
        }
        if (d.getWordOrDefault("p", "p") != "p" || d.getWordOrDefault("U", "U") != "U")
        {
            fail(fn, "only the fields p and U are supported");
        }
        f.patchIDs = patchSet(mesh, getWords(d, "patches", fn), fn);
        f.pRef = d.found("pRef") ? d.getScalar("pRef") : 0.0;
        if (d.found("CofR"))
        {
            f.origin = getVector(d, "CofR", fn);
        }
        else
        {
            fail(fn, "CofR is required (coordinateSystem dictionaries are not supported)");
        }
        const vec3 e3 = f.coeffs ? getVector(d, "liftDir", fn) : vec3{0, 0, 1};
        const vec3 e1 = f.coeffs ? getVector(d, "dragDir", fn) : vec3{1, 0, 0};
        f.R = axesRotationE3E1(e3, e1, fn);
        if (f.coeffs)
        {
            f.magUInf = d.getScalar("magUInf");
            f.lRef = d.getScalar("lRef");
            f.Aref = d.getScalar("Aref");
            f.rhoInf = d.getScalar("rhoInf");
            if (d.found("coefficients"))
            {
                fail(fn, "coefficients is not supported (all are written)");
            }
        }
        f.executeControl = d.getWordOrDefault("executeControl", "timeStep");
        f.executeInterval = d.found("executeInterval") ? int(d.getLabel("executeInterval")) : 1;
        f.writeControl = d.getWordOrDefault("writeControl", "timeStep");
        f.writeInterval = d.found("writeInterval") ? d.getScalar("writeInterval") : 1.0;
        f.resultPrecision = precision;
        f.filePrecision = d.found("writePrecision") ? int(d.getLabel("writePrecision")) : precision;
        f.writeToFile = d.getWordOrDefault("writeToFile", "true") != "false";
        out.push_back(f);
    }
    return out;
}


ForceResult evaluateForces(const HisaSolver& sol, const ForceFunction& f)
{
    const DeviceMesh& dm = sol.dm;
    const HisaState& s = sol.s;

    // fvc::grad(U)（fvSchemes 的 grad(U)）
    VolField<tens> gradU;
    gradByScheme(dm, sol.lsVectors(), sol.schemes->grad("grad(U)"), s.U, gradU);

    // 只下載選定 patch 的面
    const auto slice = [](const double* d, const std::int32_t off, const std::int32_t n)
    {
        std::vector<double> h(static_cast<std::size_t>(n));
        if (n > 0)
        {
            GNC_CUDA_CHECK
            (
                cudaMemcpy(h.data(), d + off, std::size_t(n)*sizeof(double), cudaMemcpyDeviceToHost)
            );
        }
        return h;
    };
    const CView<tens> gv = gradU.bnd.cview();

    ForceResult r;
    r.faceForce.assign(std::size_t(dm.nFvBoundaryFaces()), vec3{0, 0, 0});
    r.faceMoment.assign(std::size_t(dm.nFvBoundaryFaces()), vec3{0, 0, 0});
    const double rhoRef = 1.0;              // rho(p)：p 的單位為壓力
    const double pRef = f.pRef/rhoRef;
    for (const int pi : f.patchIDs)
    {
        const Patch& patch = sol.mesh.patches[std::size_t(pi)];
        const std::int32_t n = dm.fvPatchSize[std::size_t(pi)];
        const std::int32_t b0 = dm.fvPatchStart[std::size_t(pi)];
        const std::vector<double> Sx = slice(dm.Sf.x.data(), patch.start, n);
        const std::vector<double> Sy = slice(dm.Sf.y.data(), patch.start, n);
        const std::vector<double> Sz = slice(dm.Sf.z.data(), patch.start, n);
        const std::vector<double> Cx = slice(dm.Cf.x.data(), patch.start, n);
        const std::vector<double> Cy = slice(dm.Cf.y.data(), patch.start, n);
        const std::vector<double> Cz = slice(dm.Cf.z.data(), patch.start, n);
        const std::vector<double> pb = slice(s.thermo.p.bnd.cview().p[0], b0, n);
        const std::vector<double> rhob = slice(s.rho.bnd.cview().p[0], b0, n);
        const std::vector<double> mub = slice(s.thermo.mu.bnd.cview().p[0], b0, n);
        const std::vector<double> nutb =
            s.nut ? slice(s.nut->bnd.cview().p[0], b0, n) : std::vector<double>(std::size_t(n), 0.0);
        std::vector<std::vector<double>> gb(9);
        for (int c = 0; c < 9; ++c)
        {
            gb[std::size_t(c)] = slice(gv.p[c], b0, n);   // xx xy xz yx yy yz zx zy zz
        }
        vec3 sumP{0, 0, 0}, sumV{0, 0, 0}, sumMP{0, 0, 0}, sumMV{0, 0, 0};
        for (std::int32_t k = 0; k < n; ++k)
        {
            const std::size_t i = std::size_t(k);
            const std::size_t b = std::size_t(b0 + k);
            const vec3 Sf{Sx[i], Sy[i], Sz[i]};
            const vec3 Md = vec3{Cx[i], Cy[i], Cz[i]} - f.origin;

            // fP = rhoRef*Sf*(p - pRef)
            const vec3 fP = (rhoRef*Sf)*(pb[i] - pRef);

            // devRhoReff = -muEff*devTwoSymm(gradU)，muEff = rho*nut + mu（laminar 的 nut 為 0）
            const double muEff = rhob[i]*nutb[i] + mub[i];
            const double m = -muEff;
            double g[9];
            for (int c = 0; c < 9; ++c)
            {
                g[c] = gb[std::size_t(c)][i];
            }
            const double sph = (2.0/3.0)*((g[0] + g[4]) + g[8]);
            const double txx = m*(2*g[0] - sph);
            const double txy = m*(g[1] + g[3]);
            const double txz = m*(g[2] + g[6]);
            const double tyy = m*(2*g[4] - sph);
            const double tyz = m*(g[5] + g[7]);
            const double tzz = m*(2*g[8] - sph);
            // Sf & symmTensor（SymmTensorI.H:808）
            const vec3 fV
            {
                Sf.x*txx + Sf.y*txy + Sf.z*txz,
                Sf.x*txy + Sf.y*tyy + Sf.z*tyz,
                Sf.x*txz + Sf.y*tyz + Sf.z*tzz
            };

            const vec3 mP = cross(Md, fP);
            const vec3 mV = cross(Md, fV);
            sumP += fP;
            sumV += fV;
            sumMP += mP;
            sumMV += mV;
            // force().boundaryField()[patchi] += fP + fV（reset 之後為 0）
            r.faceForce[b] += fP + fV;
            r.faceMoment[b] += mP + mV;
        }
        // addToPatchFields：sumPatchForcesP_ += sum(fP) 等
        r.Fp += sumP;
        r.Fv += sumV;
        r.Mp += sumMP;
        r.Mv += sumMV;
    }

    const vec3 lFp = dot(r.Fp, f.R), lFv = dot(r.Fv, f.R), lFi = dot(r.Fi, f.R);
    const vec3 lMp = dot(r.Mp, f.R), lMv = dot(r.Mv, f.R), lMi = dot(r.Mi, f.R);
    if (!f.coeffs)
    {
        // forces::execute（setResult 的 vector 經過文字）
        const int pr = f.resultPrecision;
        r.vectors =
        {
            {"pressureForce", textRound(lFp, pr)}, {"viscousForce", textRound(lFv, pr)},
            {"internalForce", textRound(lFi, pr)}, {"pressureMoment", textRound(lMp, pr)},
            {"viscousMoment", textRound(lMv, pr)}, {"internalMoment", textRound(lMi, pr)}
        };
        return r;
    }

    // forceCoeffs::calcForceCoeffs、calcMomentCoeffs
    const double pDyn = 0.5*f.rhoInf*(f.magUInf*f.magUInf);
    const double fs = 1.0/(f.Aref*pDyn + SMALL);
    const double ms = 1.0/(f.Aref*pDyn*f.lRef + SMALL);
    const vec3 cfP = fs*lFp, cfV = fs*lFv, cfI = fs*lFi;
    const vec3 cmP = ms*lMp, cmV = ms*lMv, cmI = ms*lMi;
    const auto F = [&](const int k) { return vec3{cmpt(cfP, k), cmpt(cfV, k), cmpt(cfI, k)}; };
    const auto M = [&](const int k) { return vec3{cmpt(cmP, k), cmpt(cmV, k), cmpt(cmI, k)}; };
    for (const CoeffDesc& c : selectCoeffs())
    {
        vec3 v;
        if (c.c1 == -1)
        {
            v = M(c.c0);
        }
        else if (c.split == 1)
        {
            v = 0.5*F(c.c0) + M(c.c1);
        }
        else if (c.split == 2)
        {
            v = 0.5*F(c.c0) - M(c.c1);
        }
        else
        {
            v = F(c.c0);
        }
        r.scalars.push_back({c.name, (v.x + v.y) + v.z});
        r.scalars.push_back({camel(c.name, "pressure"), v.x});
        r.scalars.push_back({camel(c.name, "viscous"), v.y});
        r.scalars.push_back({camel(c.name, "internal"), v.z});
    }
    return r;
}


struct ForceFiles::Impl
{
    ForceFunction f;
    int width = 14;                 // charWidth() = writePrecision + addChars（8）
    std::vector<std::ofstream> os;  // coefficient；或 force、moment

    void open(const std::string& dir, const std::string& name, const std::string& startTimeName)
    {
        std::string file = dir + "/" + name + ".dat";
        if (std::ifstream(file).good())
        {
            file = dir + "/" + name + "_" + startTimeName + ".dat";
        }
        os.emplace_back(file);
        std::ofstream& s = os.back();
        if (!s.good())
        {
            throw std::runtime_error("cannot open " + file);
        }
        // writeFile::initStream
        s.setf(std::ios_base::scientific, std::ios_base::floatfield);
        s.precision(f.filePrecision);
        s.width(width);
    }

    // writeFile::writeCommented：之後的輸出都靠左（setf(left) 不會被重設）
    void commented(std::ofstream& s, const std::string& str) const
    {
        s << std::setw(1) << "#";
        if (!str.empty())
        {
            s << std::setw(1) << ' ' << std::left << std::setw(width - 2) << str.c_str();
        }
    }

    void header(std::ofstream& s, const std::string& str) const
    {
        commented(s, str);
        s << '\n';
    }

    void tabbed(std::ofstream& s, const std::string& str) const
    {
        s << '\t' << std::setw(width) << str.c_str();
    }

    void vectorValue(std::ofstream& s, const vec3& v) const
    {
        s << '(' << v.x << ' ' << v.y << ' ' << v.z << ')';
    }

    void headerValue(std::ofstream& s, const std::string& key, const vec3& v) const
    {
        s << std::setw(1) << '#' << std::setw(1) << ' ' << std::left << std::setw(width - 2)
          << key.c_str() << std::setw(1) << ':' << std::setw(1) << ' ';
        vectorValue(s, v);
        s << '\n';
    }

    void headerValue(std::ofstream& s, const std::string& key, const double v) const
    {
        s << std::setw(1) << '#' << std::setw(1) << ' ' << std::left << std::setw(width - 2)
          << key.c_str() << std::setw(1) << ':' << std::setw(1) << ' ' << v << '\n';
    }
};


ForceFiles::ForceFiles
(
    const std::string& caseDir,
    const ForceFunction& f,
    const std::string& startTimeName
)
:
    impl_(new Impl)
{
    impl_->f = f;
    impl_->width = f.filePrecision + 8;
    if (!f.writeToFile)
    {
        return;
    }
    std::string dir = caseDir + "/postProcessing";
    ::mkdir(dir.c_str(), 0755);
    dir += "/" + f.name;
    ::mkdir(dir.c_str(), 0755);
    dir += "/" + startTimeName;
    ::mkdir(dir.c_str(), 0755);
    Impl& m = *impl_;
    const vec3 e1{f.R.xx, f.R.yx, f.R.zx}, e2{f.R.xy, f.R.yy, f.R.zy}, e3{f.R.xz, f.R.yz, f.R.zz};
    if (f.coeffs)
    {
        // forceCoeffs::writeIntegratedDataFileHeader
        m.open(dir, "coefficient", startTimeName);
        std::ofstream& s = m.os[0];
        m.header(s, "Force and moment coefficients");
        m.headerValue(s, "dragDir", e1);
        m.headerValue(s, "sideDir", e2);
        m.headerValue(s, "liftDir", e3);
        m.headerValue(s, "rollAxis", e1);
        m.headerValue(s, "pitchAxis", e2);
        m.headerValue(s, "yawAxis", e3);
        m.headerValue(s, "magUInf", f.magUInf);
        m.headerValue(s, "lRef", f.lRef);
        m.headerValue(s, "Aref", f.Aref);
        m.headerValue(s, "CofR", f.origin);
        m.header(s, "");
        m.commented(s, "Time");
        for (const CoeffDesc& c : selectCoeffs())
        {
            m.tabbed(s, c.name);
        }
        s << std::endl;
        return;
    }
    // forces::writeIntegratedDataFileHeader
    for (const char* name : {"force", "moment"})
    {
        m.open(dir, name, startTimeName);
        std::ofstream& s = m.os.back();
        m.header(s, std::string(name) == "force" ? "Force" : "Moment");
        m.headerValue(s, "CofR", f.origin);
        m.header(s, "");
        m.commented(s, "Time");
        for (const char* root : {"total", "pressure", "viscous"})
        {
            const std::string r(root);
            m.tabbed(s, r + "_x " + r + "_y " + r + "_z");
        }
        s << std::endl;
    }
}


ForceFiles::~ForceFiles()
{
    delete impl_;
}


void ForceFiles::write(const std::string& timeName, const ForceResult& r)
{
    Impl& m = *impl_;
    if (!m.f.writeToFile)
    {
        return;
    }
    if (m.f.coeffs)
    {
        // forceCoeffs::writeIntegratedDataFile：每個係數的總和（setResult 的同名值）
        std::ofstream& s = m.os[0];
        s << std::setw(m.width) << timeName;
        // scalars 每個係數依序為總和、pressure、viscous、internal
        for (std::size_t i = 0; i < r.scalars.size(); i += 4)
        {
            s << '\t' << r.scalars[i].second;
        }
        s << std::endl;
        return;
    }
    // forces::writeIntegratedDataFile：localVector 的原值（不是 setResult 的捨入值）
    const vec3 lFp = dot(r.Fp, m.f.R), lFv = dot(r.Fv, m.f.R), lFi = dot(r.Fi, m.f.R);
    const vec3 lMp = dot(r.Mp, m.f.R), lMv = dot(r.Mv, m.f.R), lMi = dot(r.Mi, m.f.R);
    const vec3 rows[2][3] = {{(lFp + lFv) + lFi, lFp, lFv}, {(lMp + lMv) + lMi, lMp, lMv}};
    for (int k = 0; k < 2; ++k)
    {
        std::ofstream& s = m.os[std::size_t(k)];
        s << std::setw(m.width) << timeName;
        for (const vec3& v : rows[k])
        {
            s << ' ' << v.x << ' ' << v.y << ' ' << v.z;
        }
        s << std::endl;
    }
}

} // namespace gnc
