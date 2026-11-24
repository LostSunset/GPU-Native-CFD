// 裝置網格。GPU-Native-CFD，GPL-3.0-or-later。
//
// 網格上傳一次之後常駐 GPU（N-005）。編號保留 OpenFOAM 的格與面編號
// （參考次序版，design.md §3.1）。
//
// 有限體積層的邊界：empty patch 在 OpenFOAM 的 fvPatch 層大小為 0，
// 其他 patch 的面依 patch 順序串成 fvBoundaryFaces（全域面編號）。
// 邊界面上的係數只算這些面；empty patch 的面在係數陣列中維持 0。

#ifndef GNC_MESH_DEVICEMESH_CUH
#define GNC_MESH_DEVICEMESH_CUH

#include "core/foamOps.cuh"
#include "core/gncCuda.cuh"
#include "io/polyMeshReader.hpp"
#include "mesh/cellFaceGather.hpp"

#include <cstdint>
#include <vector>

namespace gnc
{

struct DeviceMesh
{
    std::int32_t nPoints = 0;
    std::int32_t nCells = 0;
    std::int32_t nFaces = 0;
    std::int32_t nInternalFaces = 0;

    std::vector<Patch> patches;                 // 主機端
    std::vector<std::int32_t> fvPatchStart;     // 各 patch 在 fvBoundaryFaces 中的起點
    std::vector<std::int32_t> fvPatchSize;      // empty 為 0

    // 連接資訊
    DeviceVec3Array points;
    DeviceArray<std::int32_t> faceOffsets;
    DeviceArray<std::int32_t> faceLabels;
    DeviceArray<std::int32_t> owner;
    DeviceArray<std::int32_t> neighbour;
    DeviceArray<std::int32_t> cellFaceOffsets;
    DeviceArray<std::int32_t> cellFaces;
    DeviceArray<std::int8_t> cellFaceSigns;
    DeviceArray<std::int32_t> fvBoundaryFaces;

    // 幾何（每面陣列的長度都是 nFaces）
    DeviceVec3Array Cf;                         // primitiveMesh::faceCentres
    DeviceVec3Array Sf;                         // primitiveMesh::faceAreas
    DeviceVec3Array C;                          // primitiveMesh::cellCentres
    DeviceArray<double> V;                      // primitiveMesh::cellVolumes
    DeviceArray<double> magSf;                  // fvMesh::magSf
    DeviceArray<double> weights;                // surfaceInterpolation::weights
    DeviceArray<double> deltaCoeffs;
    DeviceArray<double> nonOrthDeltaCoeffs;
    DeviceVec3Array nonOrthCorrectionVectors;

    // 邊界面（N2 起）
    DeviceArray<std::int32_t> faceToFvB;    // 邊界面 f 的 fvBoundaryFaces 序號，empty 為 -1；索引 f - nInternalFaces
    DeviceVec3Array fvbDelta;               // fvPatch::delta()，長度 nFvBoundaryFaces
    std::vector<vec3> patchNormal;          // symmetryPlane 的平面法向量（其他 patch 為 0）
    int solutionD[3] = {1, 1, 1};           // polyMesh::solutionD()
    int nSolutionD = 3;

    std::int32_t nFvBoundaryFaces() const
    {
        return std::int32_t(fvBoundaryFaces.size());
    }

    static DeviceMesh upload(const PolyMesh& m, const CellFaceGather& g);
};


// 依 OpenFOAM 的次序計算全部幾何量（geometry.cu）
void computeGeometry(DeviceMesh& dm);

} // namespace gnc

#endif
