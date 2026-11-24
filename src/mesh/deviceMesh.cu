// GPU-Native-CFD，GPL-3.0-or-later。見 deviceMesh.cuh。

#include "mesh/deviceMesh.cuh"

namespace gnc
{

DeviceMesh DeviceMesh::upload(const PolyMesh& m, const CellFaceGather& g)
{
    DeviceMesh dm;
    dm.nPoints = m.nPoints;
    dm.nCells = m.nCells;
    dm.nFaces = m.nFaces;
    dm.nInternalFaces = m.nInternalFaces;
    dm.patches = m.patches;

    std::vector<std::int32_t> fvb;
    std::vector<std::int32_t> f2b(std::size_t(m.nFaces - m.nInternalFaces), -1);
    for (const Patch& p : m.patches)
    {
        dm.fvPatchStart.push_back(std::int32_t(fvb.size()));
        if (!p.isEmpty())
        {
            for (std::int32_t i = 0; i < p.size; ++i)
            {
                f2b[std::size_t(p.start + i - m.nInternalFaces)] =
                    std::int32_t(fvb.size());
                fvb.push_back(p.start + i);
            }
        }
        dm.fvPatchSize.push_back(std::int32_t(fvb.size()) - dm.fvPatchStart.back());
    }
    dm.faceToFvB.upload(f2b);
    dm.patchNormal.assign(m.patches.size(), vec3{0.0, 0.0, 0.0});

    dm.points.uploadAoS(m.points);
    dm.faceOffsets.upload(m.faceOffsets);
    dm.faceLabels.upload(m.faceLabels);
    dm.owner.upload(m.owner);
    dm.neighbour.upload(m.neighbour);
    dm.cellFaceOffsets.upload(g.offsets);
    dm.cellFaces.upload(g.faces);
    dm.cellFaceSigns.upload(g.signs);
    dm.fvBoundaryFaces.upload(fvb);

    const std::size_t nF = std::size_t(m.nFaces);
    const std::size_t nC = std::size_t(m.nCells);
    dm.Cf = DeviceVec3Array(nF);
    dm.Sf = DeviceVec3Array(nF);
    dm.C = DeviceVec3Array(nC);
    dm.V = DeviceArray<double>(nC);
    dm.magSf = DeviceArray<double>(nF);
    dm.weights = DeviceArray<double>(nF);
    dm.deltaCoeffs = DeviceArray<double>(nF);
    dm.nonOrthDeltaCoeffs = DeviceArray<double>(nF);
    dm.nonOrthCorrectionVectors = DeviceVec3Array(nF);
    dm.fvbDelta = DeviceVec3Array(fvb.size());

    return dm;
}

} // namespace gnc
