/*---------------------------------------------------------------------------*\
    GPU-Native-CFD，GPL-3.0-or-later。見 gncLusgsDump.H。
\*---------------------------------------------------------------------------*/

#include "gncLusgsDump.H"
#include "OSspecific.H"
#include "fixedGradientFvPatchFields.H"

#include "core/arrayFile.hpp"

#include <string>

namespace
{

using namespace Foam;

static_assert(sizeof(label) == 4, "gncLusgsDump requires label=32");
static_assert(sizeof(scalar) == 8, "gncLusgsDump requires scalar=64");

template<class Type>
void writeF(gnc::arrayFile::Writer& w, const std::string& name, const UList<Type>& f)
{
    w.f64
    (
        name,
        reinterpret_cast<const double*>(f.cdata()),
        f.size(),
        pTraits<Type>::nComponents
    );
}

template<class Type>
void writeBlock(gnc::arrayFile::Writer& w, const std::string& name, const fvjMatrix<Type>& m)
{
    if (m.hasDiag()) writeF(w, "J." + name + ".diag", m.diag());
    if (m.hasUpper()) writeF(w, "J." + name + ".upper", m.upper());
    if (m.hasLower()) writeF(w, "J." + name + ".lower", m.lower());
}

// 每格 5 個值：scalar0、vector0、scalar1（rho、rhoU、rhoE）
void writePacked
(
    gnc::arrayFile::Writer& w,
    const std::string& name,
    const PtrList<volScalarField>& sVec,
    const PtrList<volVectorField>& vVec
)
{
    const label n = sVec[0].size();
    List<scalar> packed(5*n);
    forAll(sVec[0], celli)
    {
        packed[5*celli] = sVec[0][celli];
        packed[5*celli + 1] = vVec[0][celli].x();
        packed[5*celli + 2] = vVec[0][celli].y();
        packed[5*celli + 3] = vVec[0][celli].z();
        packed[5*celli + 4] = sVec[1][celli];
    }
    w.f64(name, packed.cdata(), n, 5);
}

} // namespace


namespace Foam
{

template<int nScalar, int nVector>
label gncLusgsDump<nScalar, nVector>::lastTimeIndex_ = -1;


template<int nScalar, int nVector>
void gncLusgsDump<nScalar, nVector>::writeJacobian() const
{
    const fvMesh& mesh = this->mesh_;
    const fileName file(outputDir_/Foam::name(timeIndex_) + ".jacobian.gnc");
    gnc::arrayFile::Writer w(file);
    w.text("producer", "gncLUSGSDump");
    w.text("time", mesh.time().timeName());

    forN(nScalar, i)
    {
        forN(nScalar, j)
        {
            if (this->jacobian_.dSBySExists(i, j))
            {
                writeBlock(w, "SS" + std::to_string(i) + std::to_string(j), this->jacobian_.dSByS(i, j));
            }
        }
        forN(nVector, j)
        {
            if (this->jacobian_.dSByVExists(i, j))
            {
                writeBlock(w, "SV" + std::to_string(i) + std::to_string(j), this->jacobian_.dSByV(i, j));
            }
        }
    }
    forN(nVector, i)
    {
        forN(nScalar, j)
        {
            if (this->jacobian_.dVBySExists(i, j))
            {
                writeBlock(w, "VS" + std::to_string(i) + std::to_string(j), this->jacobian_.dVByS(i, j));
            }
        }
        forN(nVector, j)
        {
            if (this->jacobian_.dVByVExists(i, j))
            {
                writeBlock(w, "VV" + std::to_string(i) + std::to_string(j), this->jacobian_.dVByV(i, j));
            }
        }
    }

    const volScalarField* rPdt = mesh.findObject<volScalarField>("rPseudoDeltaT");
    if (rPdt)
    {
        writeF(w, "rPseudoDeltaT", rPdt->primitiveField());
    }
    else
    {
        w.text("rPseudoDeltaT.missing", "rPseudoDeltaT");
    }
    writeF(w, "rDiagCoeff", this->rDiagCoeff_());
    writeF(w, "V", mesh.V().field());

    // 診斷：Jacobian 組裝時 U 的邊界值，與 p 的 fixedGradient 族 patch 的
    // gradient（依 patch 次序串接；characteristicWallPressure 的 updateCoeffs 結果）
    const volVectorField* U = mesh.findObject<volVectorField>("U");
    if (U)
    {
        DynamicList<vector> b;
        forAll(U->boundaryField(), patchi)
        {
            b.push_back(U->boundaryField()[patchi]);
        }
        writeF(w, "dbg.U.b", b);
    }
    for (const word name : {"p", "T", "rho", "rhoE", "e", "thermo:psi"})
    {
        const volScalarField* f = mesh.findObject<volScalarField>(name);
        if (f)
        {
            DynamicList<scalar> b;
            forAll(f->boundaryField(), patchi)
            {
                b.push_back(f->boundaryField()[patchi]);
            }
            const std::string rec(name == "thermo:psi" ? "psi" : std::string(name));
            writeF(w, "dbg." + rec + ".b", b);
        }
    }
    {
        const volVectorField* rhoU = mesh.findObject<volVectorField>("rhoU");
        DynamicList<vector> b;
        forAll(rhoU->boundaryField(), patchi)
        {
            b.push_back(rhoU->boundaryField()[patchi]);
        }
        writeF(w, "dbg.rhoU.b", b);
        const surfaceScalarField* phi = mesh.findObject<surfaceScalarField>("phi");
        DynamicList<scalar> pb;
        forAll(phi->boundaryField(), patchi)
        {
            pb.push_back(phi->boundaryField()[patchi]);
        }
        writeF(w, "dbg.phi.b", pb);
    }
    const volScalarField* p = mesh.findObject<volScalarField>("p");
    if (p)
    {
        DynamicList<scalar> g;
        forAll(p->boundaryField(), patchi)
        {
            const fvPatchScalarField& pf = p->boundaryField()[patchi];
            if (isA<fixedGradientFvPatchScalarField>(pf))
            {
                g.push_back(refCast<const fixedGradientFvPatchScalarField>(pf).gradient());
            }
        }
        writeF(w, "dbg.p.gradient", g);
    }
    w.close();
    Info<< type() << ": wrote " << file << endl;
}


template<int nScalar, int nVector>
gncLusgsDump<nScalar, nVector>::gncLusgsDump
(
    const dictionary& dict,
    const jacobianMatrix<nScalar, nVector>& jacobian,
    const preconditioner<nScalar, nVector>* prePreconditioner
)
:
    lusgs<nScalar, nVector>(dict, jacobian, prePreconditioner),
    outputDir_(dict.get<fileName>("outputDir")),
    timeIndex_(this->mesh_.time().timeIndex()),
    active_(false),
    captured_(false)
{
    if (nScalar != 2 || nVector != 1)
    {
        FatalErrorInFunction
            << "only the 2 scalar + 1 vector system is supported" << exit(FatalError);
    }
    // 每個時間步只傾印第一個外迭代（每個外迭代建構一次）
    if (timeIndex_ == lastTimeIndex_)
    {
        return;
    }
    lastTimeIndex_ = timeIndex_;
    if (Pstream::parRun())
    {
        WarningInFunction << "serial runs only; skipped" << endl;
        return;
    }
    forAll(this->mesh_.boundary(), patchi)
    {
        if (this->mesh_.boundary()[patchi].coupled())
        {
            WarningInFunction
                << "coupled patch " << this->mesh_.boundary()[patchi].name()
                << " present; skipped" << endl;
            return;
        }
    }
    mkDir(outputDir_);
    active_ = true;
    writeJacobian();
}


template<int nScalar, int nVector>
gncLusgsDump<nScalar, nVector>::~gncLusgsDump()
{
    if (!active_)
    {
        return;
    }
    // hisaModule::outerIteration 的求解區塊結束時解構：GMRES 已更新
    // rho、rhoU、rhoE，限界與 updateFields 尚未執行
    const fvMesh& mesh = this->mesh_;
    const fileName file(outputDir_/Foam::name(timeIndex_) + ".solve.gnc");
    gnc::arrayFile::Writer w(file);
    w.text("producer", "gncLUSGSDump");
    for (const word name : {"rho", "rhoE"})
    {
        const volScalarField* f = mesh.findObject<volScalarField>(name);
        if (f)
        {
            writeF(w, std::string(name) + ".i", f->primitiveField());
            DynamicList<scalar> b;
            forAll(f->boundaryField(), patchi)
            {
                b.push_back(f->boundaryField()[patchi]);
            }
            writeF(w, std::string(name) + ".b", b);
        }
    }
    const volVectorField* rhoU = mesh.findObject<volVectorField>("rhoU");
    if (rhoU)
    {
        writeF(w, "rhoU.i", rhoU->primitiveField());
        DynamicList<vector> b;
        forAll(rhoU->boundaryField(), patchi)
        {
            b.push_back(rhoU->boundaryField()[patchi]);
        }
        writeF(w, "rhoU.b", b);
    }
    w.close();
    Info<< type() << ": wrote " << file << endl;
}


template<int nScalar, int nVector>
void gncLusgsDump<nScalar, nVector>::precondition
(
    PtrList<volScalarField>& sVec,
    PtrList<volVectorField>& vVec
) const
{
    if (!active_ || captured_)
    {
        lusgs<nScalar, nVector>::precondition(sVec, vVec);
        return;
    }
    captured_ = true;
    const fileName file(outputDir_/Foam::name(timeIndex_) + ".precondition.gnc");
    gnc::arrayFile::Writer w(file);
    w.text("producer", "gncLUSGSDump");
    writePacked(w, "P.in", sVec, vVec);
    lusgs<nScalar, nVector>::precondition(sVec, vVec);
    writePacked(w, "P.out", sVec, vVec);
    w.close();
    Info<< type() << ": wrote " << file << endl;
}


makePreconditionerType(gncLusgsDump, 2, 1);

} // End namespace Foam
