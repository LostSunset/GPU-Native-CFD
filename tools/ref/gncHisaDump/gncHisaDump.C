/*---------------------------------------------------------------------------*\
    GPU-Native-CFD，GPL-3.0-or-later。見 gncHisaDump.H。
\*---------------------------------------------------------------------------*/

#include "gncHisaDump.H"
#include "addToRunTimeSelectionTable.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "uniformDimensionedFields.H"
#include "wallDist.H"
#include "thermodynamicConstants.H"
#include "OSspecific.H"

#include "core/arrayFile.hpp"

#include <string>

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(gncHisaDump, 0);
    addToRunTimeSelectionTable(functionObject, gncHisaDump, dictionary);
}
}


namespace
{

using namespace Foam;

static_assert(sizeof(label) == 4, "gncHisaDump requires label=32");
static_assert(sizeof(scalar) == 8, "gncHisaDump requires scalar=64");

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

template<class FieldType>
bool tryWrite
(
    const objectRegistry& db,
    gnc::arrayFile::Writer& w,
    const word& name,
    const std::string& rec
)
{
    const FieldType* f = db.findObject<FieldType>(name);
    if (f)
    {
        writeGeo(w, rec, *f);
        return true;
    }
    return false;
}

bool tryWriteUniform
(
    const objectRegistry& db,
    gnc::arrayFile::Writer& w,
    const word& name,
    const std::string& rec
)
{
    const uniformDimensionedScalarField* f =
        db.findObject<uniformDimensionedScalarField>(name);
    if (f)
    {
        const scalar v = f->value();
        w.f64(rec, &v, 1);
        return true;
    }
    return false;
}

} // namespace


Foam::functionObjects::gncHisaDump::gncHisaDump
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    fvMeshFunctionObject(name, runTime, dict),
    atEnd_(false),
    writeConstants_(true),
    wallDist_(false)
{
    read(dict);
}


bool Foam::functionObjects::gncHisaDump::read(const dictionary& dict)
{
    fvMeshFunctionObject::read(dict);
    outputDir_ = dict.get<fileName>("outputDir");
    fields_ = dict.get<wordList>("fields");
    timeIndices_ = dict.getOrDefault<labelList>("timeIndices", labelList());
    atEnd_ = dict.getOrDefault<bool>("atEnd", false);
    writeConstants_ = dict.getOrDefault<bool>("writeConstants", true);
    wallDist_ = dict.getOrDefault<bool>("wallDist", false);
    forceObjects_ = dict.getOrDefault<wordList>("forceObjects", wordList());
    return true;
}


void Foam::functionObjects::gncHisaDump::dump(const fileName& file) const
{
    mkDir(outputDir_);
    gnc::arrayFile::Writer w(file);
    w.text("producer", "gncHisaDump");
    w.text("time", time_.timeName());
    w.text("timeIndex", Foam::name(time_.timeIndex()));
    if (writeConstants_)
    {
        const scalar RR = constant::thermodynamic::RR;
        const scalar Tstd = constant::thermodynamic::Tstd;
        w.f64("RR", &RR, 1);
        w.f64("Tstd", &Tstd, 1);
    }
    if (wallDist_)
    {
        // fvSchemes 的 wallDist 方法（N4a-1 只驗 meshWave）
        const wallDist& wd = wallDist::New(mesh_);
        writeGeo(w, "yWall", wd.y());
    }

    for (const word& name : fields_)
    {
        const auto colon = name.rfind(':');
        const std::string rec =
            colon == std::string::npos ? std::string(name) : name.substr(colon + 1);
        const bool ok =
            tryWrite<volScalarField>(mesh_, w, name, rec)
         || tryWrite<volVectorField>(mesh_, w, name, rec)
         || tryWrite<volTensorField>(mesh_, w, name, rec)
         || tryWrite<surfaceScalarField>(mesh_, w, name, rec)
         || tryWrite<surfaceVectorField>(mesh_, w, name, rec)
         || tryWriteUniform(mesh_, w, name, rec);
        if (!ok)
        {
            WarningInFunction << "field " << name << " not found" << endl;
            w.text(rec + ".missing", name);
        }
    }

    // forces／forceCoeffs 的 result（記憶體中的值，沒有經過文字）
    for (const word& obj : forceObjects_)
    {
        const wordList entries(objectResultEntries(obj));
        if (entries.empty())
        {
            WarningInFunction << "no results for " << obj << endl;
            w.text(obj + ".missing", obj);
        }
        for (const word& e : entries)
        {
            const word t(objectResultType(obj, e));
            const std::string rec = obj + "." + e;
            if (t == pTraits<scalar>::typeName)
            {
                const scalar v = getObjectResult<scalar>(obj, e);
                w.f64(rec, &v, 1);
            }
            else if (t == pTraits<vector>::typeName)
            {
                const vector v = getObjectResult<vector>(obj, e);
                w.f64(rec, v.cdata(), 1, 3);
            }
            else
            {
                w.text(rec + ".unsupported", t);
            }
        }
    }
    w.close();
    Info<< type() << ": wrote " << file << endl;
}


bool Foam::functionObjects::gncHisaDump::execute()
{
    if (timeIndices_.empty() || timeIndices_.found(time_.timeIndex()))
    {
        dump(outputDir_/Foam::name(time_.timeIndex()) + ".gnc");
    }
    return true;
}


bool Foam::functionObjects::gncHisaDump::end()
{
    if (atEnd_)
    {
        dump(outputDir_/"end.gnc");
    }
    return true;
}


bool Foam::functionObjects::gncHisaDump::write()
{
    return true;
}


// ************************************************************************* //
