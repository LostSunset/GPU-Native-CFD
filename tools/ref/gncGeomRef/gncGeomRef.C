/*---------------------------------------------------------------------------*\
    GPU-Native-CFD：L0 幾何參考值（design.md §6）。GPL-3.0-or-later。

    以 OpenFOAM 讀案例並建立 fvMesh，把 primitiveMesh 與 surfaceInterpolation
    的幾何量以 GNC 陣列檔寫出，紀錄名稱與 tests/gncGeomL0 相同。本工具連結
    OpenFOAM，只用來產生參考值，不屬於求解器（N-004）。

    用法：gncGeomRef -case DIR -out FILE
\*---------------------------------------------------------------------------*/

#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "fvGeometryScheme.H"
#include "surfaceFields.H"
#include "volFields.H"
#include "foamVersion.H"
#include "IOmanip.H"

#include "core/arrayFile.hpp"

#include <string>
#include <vector>

using namespace Foam;

namespace
{

static_assert(sizeof(label) == 4, "gncGeomRef requires label=32");
static_assert(sizeof(scalar) == 8, "gncGeomRef requires scalar=64");

template<class Type>
void writeSurface
(
    gnc::arrayFile::Writer& w,
    const std::string& name,
    const GeometricField<Type, fvsPatchField, surfaceMesh>& fld
)
{
    const unsigned nComp = pTraits<Type>::nComponents;
    const Field<Type>& in = fld.primitiveField();
    w.f64
    (
        name + ".i",
        reinterpret_cast<const double*>(in.cdata()),
        in.size(),
        nComp
    );

    // 邊界：依 patch 順序串接（empty patch 的大小為 0）
    DynamicList<Type> b;
    forAll(fld.boundaryField(), patchi)
    {
        b.push_back(fld.boundaryField()[patchi]);
    }
    w.f64
    (
        name + ".b",
        reinterpret_cast<const double*>(b.cdata()),
        b.size(),
        nComp
    );
}

} // namespace


int main(int argc, char *argv[])
{
    argList::addNote
    (
        "Write the L0 geometry reference of GPU-Native-CFD (GNC array file)"
    );
    argList::noParallel();
    argList::noFunctionObjects();
    argList::addOption("out", "file", "GNC array file to write");

    #include "setRootCase.H"
    #include "createTime.H"

    // 與 createMesh.H 的一般路徑相同（不含 -dry-run 的簡化網格）
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

    const fileName outFile(args.get<fileName>("out"));

    gnc::arrayFile::Writer w(outFile);
    w.text("producer", "gncGeomRef");
    w.text("case", args.path());
    w.text
    (
        "openfoam",
        std::string(foamVersion::version) + " api " + std::to_string(foamVersion::api)
      + " build " + std::string(foamVersion::build)
    );
    w.text("pointsInstance", mesh.pointsInstance());
    w.text("facesInstance", mesh.facesInstance());
    w.text("geometryScheme", mesh.geometry().type());

    const label dims[4] =
    {
        mesh.nPoints(), mesh.nCells(), mesh.nFaces(), mesh.nInternalFaces()
    };
    w.i32("dims", dims, 4);

    labelList fvPatchSizes(mesh.boundary().size());
    forAll(mesh.boundary(), patchi)
    {
        fvPatchSizes[patchi] = mesh.boundary()[patchi].size();
    }
    w.i32("fvPatchSizes", fvPatchSizes.cdata(), fvPatchSizes.size());

    // 連接資訊：驗證 GPU-Native-CFD 的讀取器
    w.f64
    (
        "points",
        reinterpret_cast<const double*>(mesh.points().cdata()),
        mesh.nPoints(),
        3
    );
    w.i32("owner", mesh.faceOwner().cdata(), mesh.faceOwner().size());
    w.i32("neighbour", mesh.faceNeighbour().cdata(), mesh.faceNeighbour().size());
    {
        const faceList& fcs = mesh.faces();
        std::vector<label> offsets(fcs.size() + 1, 0);
        std::vector<label> labels;
        forAll(fcs, facei)
        {
            for (const label pointi : fcs[facei])
            {
                labels.push_back(pointi);
            }
            offsets[facei + 1] = label(labels.size());
        }
        w.i32("faceOffsets", offsets.data(), offsets.size());
        w.i32("faceLabels", labels.data(), labels.size());
    }

    // primitiveMesh 幾何（全部面，含 empty patch 的面）
    w.f64("Cf", reinterpret_cast<const double*>(mesh.faceCentres().cdata()), mesh.nFaces(), 3);
    w.f64("Sf", reinterpret_cast<const double*>(mesh.faceAreas().cdata()), mesh.nFaces(), 3);
    w.f64("C", reinterpret_cast<const double*>(mesh.cellCentres().cdata()), mesh.nCells(), 3);
    w.f64("V", mesh.cellVolumes().cdata(), mesh.nCells(), 1);

    // fvMesh／surfaceInterpolation（有限體積層，empty patch 為 0 個面）
    writeSurface(w, "magSf", mesh.magSf());
    writeSurface(w, "weights", mesh.weights());
    writeSurface(w, "deltaCoeffs", mesh.deltaCoeffs());
    writeSurface(w, "nonOrthDeltaCoeffs", mesh.nonOrthDeltaCoeffs());
    writeSurface(w, "nonOrthCorrectionVectors", mesh.nonOrthCorrectionVectors());

    const scalar sumV = gSum(mesh.V());
    w.f64("sumV", &sumV, 1);

    w.close();

    Info<< "Wrote " << outFile << nl
        << "geometry scheme " << mesh.geometry().type() << nl
        << "sum(V) " << setprecision(17) << sumV << nl
        << "End" << nl << endl;

    return 0;
}


// ************************************************************************* //
