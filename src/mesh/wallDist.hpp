// 牆距離（meshWave）。GPU-Native-CFD，GPL-3.0-or-later。
//
// 照 OpenFOAM v2606 的 patchDistMethods::meshWave（correctWalls true，
// cellDistFuncs::useCombinedWallPatch 預設 true），主機端、循序執行；網格不變
// 時只算一次，不在時間迴圈內（N-001）。
//   1. patchWave：type 為 wall 的 patch 的每個面，以面心（primitiveMesh 的
//      faceCentres）為 origin、distSqr 0 開始，FaceCellWave<wallPoint> 傳播：
//      依 changedFaces 的次序把面資訊給 owner、neighbour（faceToCell），再依
//      changedCells 的次序把格資訊給該格 cells() 的每個面（cellToFace，
//      cells() 為 owner 面遞增、再 neighbour 面遞增），直到沒有改變。
//      wallPoint::update：未設定時直接採用；新的 distSqr 較大時不採用；
//      差 < SMALL 或相對差 < propagationTol（0.01）時不採用（所以結果與傳播
//      次序有關，必須照原次序）。
//   2. 距離 = sqrt(distSqr)；patch 面的距離 = sqrt(distSqr) + SMALL。
//   3. correctBoundaryCells：
//      - 有面在牆上的格（依牆面的全域次序，後面的覆蓋前面的）：距離 = 該牆面
//        與所有和它共用頂點的牆面中，格心到面的最短距離（face::nearestPoint）。
//      - 有頂點在牆上、但還沒修正的格：依牆 patch 的局部點次序（各面頂點的
//        首次出現次序），第一個碰到它的點，距離 = 格心到含該點的牆面的最短
//        距離。
//   face::nearestPoint：三角形直接算（triangle::nearestPointClassify，
//   Ericson 的演算法）；多邊形以 face::centre 為中心切成扇形三角形取最小。
// nearWallDist::calculate（useCombinedWallPatch）每個牆面的值與 3. 的「有面在
// 牆上的格」相同（同一個 smallestDist），一併輸出到 nearWall。
// 只支援沒有 cyclic、processor 的網格。

#ifndef GNC_MESH_WALLDIST_HPP
#define GNC_MESH_WALLDIST_HPP

#include "io/polyMeshReader.hpp"

#include <cstdint>
#include <vector>

namespace gnc
{

struct WallDistance
{
    std::vector<double> cell;       // nCells
    std::vector<double> face;       // nFaces（patch 面為 sqrt(distSqr) + SMALL；
                                    // 內部面未定義）
    std::vector<double> nearWall;   // nFaces：nearWallDist（turbulenceModel::y()）
                                    // 的 patch 值；wall 面為相鄰格心到該面與共用
                                    // 頂點的牆面的最短距離，其他面為 0
    std::int32_t nUnset = 0;        // 沒有被波傳到的格與 patch 面
};

// cellCentres、faceCentres：primitiveMesh 的值，交錯 xyz
WallDistance meshWaveWallDistance
(
    const PolyMesh& mesh,
    const std::vector<double>& cellCentres,
    const std::vector<double>& faceCentres
);

} // namespace gnc

#endif
