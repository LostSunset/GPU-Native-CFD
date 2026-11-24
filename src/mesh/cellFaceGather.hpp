// 格面收集表（design.md §3.2）。GPU-Native-CFD，GPL-3.0-or-later。
//
// 每一格列出它的面與符號（owner +1、neighbour −1），依面編號遞增排序。
// GPU 上一格一個 thread 依序走這張表，就能不用 atomics 而得到與 OpenFOAM
// 面迴圈相同的加總次序：
//   - 「依面編號」次序：surfaceIntegrate、gaussGrad、lduMatrix::Amul 之類
//     先走內部面（owner 加、neighbour 減）、再依 patch 走邊界面的迴圈。
//   - 「先 owner 後 neighbour」次序：primitiveMeshTools::
//     makeCellCentresAndVols 先走 own（含邊界面）、再走 nei。只要先取
//     符號 +1 的項、再取 −1 的項，各自仍依面編號遞增。
// 每個運算子用哪一種次序，寫在它的 kernel 註解中。

#ifndef GNC_MESH_CELLFACEGATHER_HPP
#define GNC_MESH_CELLFACEGATHER_HPP

#include "io/polyMeshReader.hpp"

#include <cstdint>
#include <vector>

namespace gnc
{

struct CellFaceGather
{
    std::vector<std::int32_t> offsets;  // nCells + 1
    std::vector<std::int32_t> faces;    // 每格依面編號遞增
    std::vector<std::int8_t> signs;     // +1 owner，−1 neighbour
};


CellFaceGather buildCellFaceGather(const PolyMesh& mesh);

} // namespace gnc

#endif
