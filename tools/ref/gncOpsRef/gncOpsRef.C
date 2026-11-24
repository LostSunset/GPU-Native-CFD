/*---------------------------------------------------------------------------*\
    GPU-Native-CFD：L1 的參考值（design.md §6）。GPL-3.0-or-later。

    以 OpenFOAM 與 HiSA 的實作計算場、邊界條件與運算子，用 GNC 陣列檔
    寫出；紀錄名稱與 tests/gncOpsL1 相同。本工具連結 OpenFOAM 與 HiSA，只用
    來產生參考值，不屬於求解器（N-004）。

    三種模式：
      -generate          產生測試場 s、v、gamma、phi，寫到 <case>/gncL1
                         （ASCII，17 位）與 <case>/gncL1bin（二進位）
      （預設）           讀 -fields 的場，evaluate 邊界條件，算各運算子，
                         寫到 -out
      -dump-fields DIR   讀 DIR 的 s、v、phi，evaluate 後寫出 s、v（檢查
                         OpenFOAM 能讀 GPU-Native-CFD 寫的場）

    測試場的值由整數雜湊（splitmix64）產生，與機器無關。邊界條件依 patch
    型別指定：constraint patch（empty、symmetryPlane、symmetry）用同名型別；
    其餘 patch 依序輪流：
      s      fixedValue、zeroGradient、inletOutlet、slip
      v      slip、inletOutlet、fixedValue、zeroGradient
      gamma  fixedValue、zeroGradient
    phi 為 calculated（constraint patch 用其 fvsPatchField 型別）。
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "IOmanip.H"
#include "OFstream.H"
#include "IStringStream.H"
#include "mixedFvPatchFields.H"
#include "linear.H"
#include "fvcSurfaceIntegrate.H"
#include "gaussGrad.H"
#include "gradScheme.H"
#include "snGradScheme.H"
#include "laplacianScheme.H"
#include "divScheme.H"
#include "faceLeastSquaresVectors.H"

#include "core/arrayFile.hpp"

#include <cstdint>
#include <string>
#include <vector>

using namespace Foam;

namespace
{

static_assert(sizeof(label) == 4, "gncOpsRef requires label=32");
static_assert(sizeof(scalar) == 8, "gncOpsRef requires scalar=64");

std::uint64_t splitmix64(std::uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30))*0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27))*0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

// [-1, 1) 的確定性亂數
scalar rnd(const std::uint64_t seed, const std::uint64_t i)
{
    const std::uint64_t r = splitmix64(seed*0x100000001b3ULL + i);
    return scalar(r >> 11)*(1.0/4503599627370496.0) - 1.0;
}

bool isConstraint(const word& t)
{
    return t == "empty" || t == "symmetryPlane" || t == "symmetry";
}


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

// 體積場：.i 內部值、.b 依 patch 串接的邊界值
template<class Type>
void writeVol
(
    gnc::arrayFile::Writer& w,
    const std::string& name,
    const GeometricField<Type, fvPatchField, volMesh>& f
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
void writeSurf
(
    gnc::arrayFile::Writer& w,
    const std::string& name,
    const GeometricField<Type, fvsPatchField, surfaceMesh>& f
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
void writeSnGradBC
(
    gnc::arrayFile::Writer& w,
    const std::string& name,
    const GeometricField<Type, fvPatchField, volMesh>& f
)
{
    DynamicList<Type> b;
    forAll(f.boundaryField(), patchi)
    {
        b.push_back(f.boundaryField()[patchi].snGrad()());
    }
    writeF(w, name, b);
}


void writeTo
(
    const regIOobject& obj,
    const fileName& dir,
    const IOstreamOption::streamFormat fmt
)
{
    mkDir(dir);
    OFstream os(dir/obj.name(), IOstreamOption(fmt));
    os.precision(17);
    obj.writeHeader(os);
    obj.writeData(os);
    IOobject::writeEndDivider(os);
}


void generate(const fvMesh& mesh)
{
    const fvBoundaryMesh& bm = mesh.boundary();
    const wordList sCycle({"fixedValue", "zeroGradient", "inletOutlet", "slip"});
    const wordList vCycle({"slip", "inletOutlet", "fixedValue", "zeroGradient"});
    const wordList gCycle({"fixedValue", "zeroGradient"});

    wordList sTypes(bm.size()), vTypes(bm.size()), gTypes(bm.size());
    label j = 0;
    forAll(bm, patchi)
    {
        const word& pt = bm[patchi].type();
        if (isConstraint(pt))
        {
            sTypes[patchi] = vTypes[patchi] = gTypes[patchi] = pt;
        }
        else
        {
            sTypes[patchi] = sCycle[j % 4];
            vTypes[patchi] = vCycle[j % 4];
            gTypes[patchi] = gCycle[j % 2];
            ++j;
        }
    }

    const auto io = [&](const word& name)
    {
        return IOobject
        (
            name, "gncL1", mesh,
            IOobject::NO_READ, IOobject::NO_WRITE, IOobject::NO_REGISTER
        );
    };

    volScalarField s(io("s"), mesh, dimensionedScalar(dimless, Zero), sTypes);
    volVectorField v(io("v"), mesh, dimensionedVector(dimless, Zero), vTypes);
    volScalarField gamma(io("gamma"), mesh, dimensionedScalar(dimless, Zero), gTypes);
    surfaceScalarField phi(io("phi"), mesh, dimensionedScalar(dimless, Zero));

    forAll(s, c)
    {
        s[c] = 2.0 + rnd(1, c);
        v[c] = vector(rnd(2, c), rnd(3, c), rnd(4, c));
        gamma[c] = 1.5 + 0.5*rnd(5, c);
    }
    forAll(phi, f)
    {
        phi[f] = rnd(6, f);
    }

    label k = 0;    // 邊界面的全域序號（依 patch 串接）
    forAll(bm, patchi)
    {
        const label n = bm[patchi].size();
        scalarField sb(n), gb(n), pb(n), sIn(n);
        vectorField vb(n), vIn(n);
        for (label i = 0; i < n; ++i, ++k)
        {
            sb[i] = 2.0 + rnd(7, k);
            vb[i] = vector(rnd(8, k), rnd(9, k), rnd(10, k));
            gb[i] = 1.5 + 0.5*rnd(11, k);
            pb[i] = rnd(12, k);
            sIn[i] = 2.0 + rnd(13, k);
            vIn[i] = vector(rnd(14, k), rnd(15, k), rnd(16, k));
        }
        if (isConstraint(bm[patchi].type()))
        {
            phi.boundaryFieldRef()[patchi] == pb;
            continue;
        }
        s.boundaryFieldRef()[patchi] == sb;
        v.boundaryFieldRef()[patchi] == vb;
        gamma.boundaryFieldRef()[patchi] == gb;
        phi.boundaryFieldRef()[patchi] == pb;
        if (sTypes[patchi] == "inletOutlet")
        {
            refCast<mixedFvPatchScalarField>(s.boundaryFieldRef()[patchi]).refValue() = sIn;
        }
        if (vTypes[patchi] == "inletOutlet")
        {
            refCast<mixedFvPatchVectorField>(v.boundaryFieldRef()[patchi]).refValue() = vIn;
        }
    }

    // regIOobject::writeObject 會把物件寫到目前的時間目錄，所以直接以
    // OFstream 寫到指定目錄
    const fileName caseDir(mesh.time().path());
    for
    (
        const regIOobject* obj :
        std::initializer_list<const regIOobject*>{&s, &v, &gamma, &phi}
    )
    {
        writeTo(*obj, caseDir/"gncL1", IOstreamOption::ASCII);
        writeTo(*obj, caseDir/"gncL1bin", IOstreamOption::BINARY);
    }

    Info<< "Generated s, v, gamma, phi in gncL1 (ascii) and gncL1bin (binary)" << nl
        << "s types " << sTypes << nl
        << "v types " << vTypes << nl
        << "gamma types " << gTypes << endl;
}


void compute(const fvMesh& mesh, const word& fieldsDir, const fileName& outFile)
{
    const auto io = [&](const word& name)
    {
        return IOobject(name, fieldsDir, mesh, IOobject::MUST_READ, IOobject::NO_WRITE);
    };
    surfaceScalarField phi(io("phi"), mesh);
    volScalarField s(io("s"), mesh);
    volVectorField v(io("v"), mesh);
    volScalarField gamma(io("gamma"), mesh);

    s.correctBoundaryConditions();
    v.correctBoundaryConditions();
    gamma.correctBoundaryConditions();

    gnc::arrayFile::Writer w(outFile);
    w.text("producer", "gncOpsRef");

    writeVol(w, "s", s);
    writeVol(w, "v", v);
    writeVol(w, "gamma", gamma);
    writeSurf(w, "phi", phi);

    writeSnGradBC(w, "snGradBC.s", s);
    writeSnGradBC(w, "snGradBC.v", v);

    writeSurf(w, "interp.s", linearInterpolate(s)());
    const surfaceVectorField iV(linearInterpolate(v));
    writeSurf(w, "interp.v", iV);

    writeF(w, "div.phi", fvc::surfaceIntegrate(phi)().primitiveField());
    writeF(w, "div.phiv", fvc::surfaceIntegrate(phi*iV)().primitiveField());

    writeVol(w, "gaussGrad.s", fv::gaussGrad<scalar>(mesh).calcGrad(s, "gaussGrad(s)")());
    writeVol(w, "gaussGrad.v", fv::gaussGrad<vector>(mesh).calcGrad(v, "gaussGrad(v)")());

    const faceLeastSquaresVectors& lsv = faceLeastSquaresVectors::New(mesh);
    writeSurf(w, "ls.P", lsv.pVectors());
    writeF(w, "ls.N.i", lsv.nVectors().primitiveField());

    writeVol
    (
        w, "lsGrad.s",
        fv::gradScheme<scalar>::New
        (
            mesh, IStringStream("faceLeastSquares linear")()
        )().calcGrad(s, "lsGrad(s)")()
    );
    const volTensorField lsGradV
    (
        fv::gradScheme<vector>::New
        (
            mesh, IStringStream("faceLeastSquares linear")()
        )().calcGrad(v, "lsGrad(v)")
    );
    writeVol(w, "lsGrad.v", lsGradV);

    // compact 的修正項用 fvSchemes 的 grad(s)、grad(v)
    writeSurf
    (
        w, "snGradCompact.s",
        fv::snGradScheme<scalar>::New(mesh, IStringStream("compact")())().snGrad(s)()
    );
    writeSurf
    (
        w, "snGradCompact.v",
        fv::snGradScheme<vector>::New(mesh, IStringStream("compact")())().snGrad(v)()
    );

    writeF
    (
        w, "laplacian.s",
        fv::laplacianScheme<scalar, scalar>::New
        (
            mesh, IStringStream("Gauss linear compact")()
        ).ref().fvcLaplacian(gamma, s)().primitiveField()
    );
    writeF
    (
        w, "laplacian.v",
        fv::laplacianScheme<vector, scalar>::New
        (
            mesh, IStringStream("Gauss linear compact")()
        ).ref().fvcLaplacian(gamma, v)().primitiveField()
    );

    writeF
    (
        w, "div.lsGradV",
        fv::divScheme<tensor>::New
        (
            mesh, IStringStream("Gauss linear")()
        ).ref().fvcDiv(lsGradV)().primitiveField()
    );

    w.close();
    Info<< "Wrote " << outFile << endl;
}


void dumpFields(const fvMesh& mesh, const word& dir, const fileName& outFile)
{
    const auto io = [&](const word& name)
    {
        return IOobject(name, dir, mesh, IOobject::MUST_READ, IOobject::NO_WRITE);
    };
    surfaceScalarField phi(io("phi"), mesh);
    volScalarField s(io("s"), mesh);
    volVectorField v(io("v"), mesh);
    s.correctBoundaryConditions();
    v.correctBoundaryConditions();

    gnc::arrayFile::Writer w(outFile);
    w.text("producer", "gncOpsRef -dump-fields " + dir);
    writeVol(w, "s", s);
    writeVol(w, "v", v);
    writeSurf(w, "phi", phi);
    w.close();
    Info<< "Wrote " << outFile << endl;
}

} // namespace


int main(int argc, char *argv[])
{
    argList::addNote("Write the L1 reference of GPU-Native-CFD (GNC array file)");
    argList::noParallel();
    argList::noFunctionObjects();
    argList::addBoolOption("generate", "write the test fields to gncL1 and gncL1bin");
    argList::addOption("fields", "dir", "directory of the test fields (default gncL1)");
    argList::addOption("dump-fields", "dir", "read s, v, phi from dir and dump them");
    argList::addOption("out", "file", "GNC array file to write");

    #include "setRootCase.H"
    #include "createTime.H"

    fvMesh mesh
    (
        IOobject
        (
            polyMesh::defaultRegion,
            runTime.timeName(),
            runTime,
            IOobject::MUST_READ
        )
    );

    if (args.found("generate"))
    {
        generate(mesh);
    }
    else if (args.found("dump-fields"))
    {
        dumpFields(mesh, args.get<word>("dump-fields"), args.get<fileName>("out"));
    }
    else
    {
        compute
        (
            mesh,
            args.getOrDefault<word>("fields", "gncL1"),
            args.get<fileName>("out")
        );
    }

    Info<< "End" << nl << endl;
    return 0;
}


// ************************************************************************* //
