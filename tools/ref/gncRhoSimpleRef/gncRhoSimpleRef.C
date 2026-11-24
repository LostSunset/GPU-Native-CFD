/*---------------------------------------------------------------------------*\
    GPU-Native-CFD：rhoSimpleFoam（OpenFOAM v2606）加上傾印點，N6 的參考。
    GPL-3.0-or-later。

    主程式照 applications/solvers/compressible/rhoSimpleFoam/rhoSimpleFoam.C，
    createFields.H、UEqn.H、EEqn.H、pEqn.H 原樣 include（Make/options 指向
    $FOAM_SOLVERS/compressible/rhoSimpleFoam），算術與原本的求解器相同，只在
    下列時點傾印（GNC 陣列檔，<dumpDir>/<步>.<階段>.gnc）：
        0.init    createFields 與 turbulence->validate() 之後
        k.U       UEqn.H 之後：U，以及鬆弛後的 UEqn（diag、upper、lower、
                  source、internalCoeffs、boundaryCoeffs）
        k.E       EEqn.H 之後
        k.p       pEqn.H 之後：另有 rAU、HbyA、phiHbyA、rhorAUf
        k.turb    turbulence->correct() 之後
    各階段都寫 p T U he psi alpha rho phi k omega nut alphat（有的話）。

    用法：gncRhoSimpleRef -dumpDir DIR -dumpSteps '(1 2 10)' [標準選項]
\*---------------------------------------------------------------------------*/

#include "fvCFD.H"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "simpleControl.H"
#include "pressureControl.H"
#include "fvOptions.H"

#include "core/arrayFile.hpp"

#include <string>

namespace
{

using namespace Foam;

static_assert(sizeof(label) == 4, "gncRhoSimpleRef requires label=32");
static_assert(sizeof(scalar) == 8, "gncRhoSimpleRef requires scalar=64");

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

template<class Type, template<class> class PatchField, class GeoMesh>
void writeGeo
(
    gnc::arrayFile::Writer& w,
    const std::string& name,
    const GeometricField<Type, PatchField, GeoMesh>& f
)
{
    writeF(w, name + ".i", f.primitiveField());
    DynamicList<Type> b;
    forAll(f.boundaryField(), patchi)
    {
        b.push_back(f.boundaryField()[patchi]);
    }
    writeF(w, name + ".b", b);
}

template<class Type>
void writeMatrix(gnc::arrayFile::Writer& w, const std::string& name, const fvMatrix<Type>& M)
{
    if (M.hasDiag())
    {
        writeF(w, name + ".diag", M.diag());
    }
    if (M.hasUpper())
    {
        writeF(w, name + ".upper", M.upper());
    }
    if (M.hasLower())
    {
        writeF(w, name + ".lower", M.lower());
    }
    writeF(w, name + ".source", M.source());
    DynamicList<Type> ic, bc;
    forAll(M.psi().boundaryField(), patchi)
    {
        ic.push_back(M.internalCoeffs()[patchi]);
        bc.push_back(M.boundaryCoeffs()[patchi]);
    }
    writeF(w, name + ".internalCoeffs", ic);
    writeF(w, name + ".boundaryCoeffs", bc);
}

template<class FieldType>
bool tryWrite(const fvMesh& mesh, gnc::arrayFile::Writer& w, const word& name)
{
    const FieldType* f = mesh.findObject<FieldType>(name);
    if (f)
    {
        writeGeo(w, name, *f);
        return true;
    }
    return false;
}

void writeState(const fvMesh& mesh, gnc::arrayFile::Writer& w)
{
    for
    (
        const word name :
        {
            "p", "T", "U", "e", "h", "thermo:psi", "thermo:alpha", "thermo:mu", "rho", "phi",
            "k", "omega", "nut", "alphat"
        }
    )
    {
        const bool ok =
            tryWrite<volScalarField>(mesh, w, name)
         || tryWrite<volVectorField>(mesh, w, name)
         || tryWrite<surfaceScalarField>(mesh, w, name);
        if (!ok)
        {
            w.text(name + ".missing", name);
        }
    }
}

struct Dumper
{
    fileName dir;
    labelList steps;
    bool active(const label k) const
    {
        return dir.size() && (k == 0 || steps.found(k));
    }
    std::string path(const label k, const char* stage) const
    {
        return dir/(Foam::name(k) + "." + stage + ".gnc");
    }
};

} // namespace


int main(int argc, char *argv[])
{
    argList::addNote
    (
        "rhoSimpleFoam with GPU-Native-CFD dump points (N6 reference)."
    );
    argList::addOption("dumpDir", "dir", "GNC dump directory");
    argList::addOption("dumpSteps", "list", "time indices to dump, e.g. '(1 2 10)'");

    #include "postProcess.H"

    #include "addCheckCaseOptions.H"
    #include "setRootCaseLists.H"
    #include "createTime.H"
    #include "createMesh.H"
    #include "createControl.H"
    #include "createFields.H"
    #include "createFieldRefs.H"
    #include "initContinuityErrs.H"

    turbulence->validate();

    Dumper dump;
    args.readIfPresent("dumpDir", dump.dir);
    args.readListIfPresent("dumpSteps", dump.steps);
    if (dump.active(0))
    {
        mkDir(dump.dir);
        gnc::arrayFile::Writer w(dump.path(0, "init"));
        w.text("timeIndex", "0");
        writeState(mesh, w);
        w.close();
    }

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info<< "\nStarting time loop\n" << endl;

    while (simple.loop())
    {
        Info<< "Time = " << runTime.timeName() << nl << endl;
        const label k = runTime.timeIndex();

        // Pressure-velocity SIMPLE corrector
        #include "UEqn.H"
        if (dump.active(k))
        {
            gnc::arrayFile::Writer w(dump.path(k, "U"));
            w.text("timeIndex", Foam::name(k));
            writeState(mesh, w);
            writeMatrix(w, "UEqn", UEqn);
            w.close();
        }

        #include "EEqn.H"
        if (dump.active(k))
        {
            gnc::arrayFile::Writer w(dump.path(k, "E"));
            w.text("timeIndex", Foam::name(k));
            writeState(mesh, w);
            w.close();
        }

        if (simple.consistent())
        {
            FatalErrorInFunction << "SIMPLE consistent is not supported" << exit(FatalError);
        }
        #include "pEqn.H"
        if (dump.active(k))
        {
            gnc::arrayFile::Writer w(dump.path(k, "p"));
            w.text("timeIndex", Foam::name(k));
            writeState(mesh, w);
            writeGeo(w, "rAU", rAU);
            writeGeo(w, "HbyA", HbyA);
            writeGeo(w, "phiHbyA", phiHbyA);
            writeGeo(w, "rhorAUf", rhorAUf);
            w.close();
        }

        turbulence->correct();
        if (dump.active(k))
        {
            gnc::arrayFile::Writer w(dump.path(k, "turb"));
            w.text("timeIndex", Foam::name(k));
            writeState(mesh, w);
            w.close();
        }

        runTime.write();

        runTime.printExecutionTime(Info);
    }

    Info<< "End\n" << endl;

    return 0;
}


// ************************************************************************* //
