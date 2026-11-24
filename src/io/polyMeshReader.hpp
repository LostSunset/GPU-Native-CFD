// OpenFOAM polyMesh 讀取器。GPU-Native-CFD，GPL-3.0-or-later。
//
// 讀 constant/polyMesh 的 points、faces、owner、neighbour、boundary，
// ASCII 與二進位都支援；二進位只接受 arch "LSB;label=32;scalar=64"。
// faces 接受 faceList（ASCII）與 faceCompactList（offsets＋labels，
// 二進位寫出時的格式；CompactIOList.C）。壓縮檔（.gz）、label 64、
// scalar 32 與範圍外的 patch 型別都報錯停止（README 原則 5）。

#ifndef GNC_IO_POLYMESHREADER_HPP
#define GNC_IO_POLYMESHREADER_HPP

#include <cstdint>
#include <string>
#include <vector>

namespace gnc
{

struct Patch
{
    std::string name;
    std::string type;
    std::int32_t start = 0;
    std::int32_t size = 0;
    std::vector<std::string> inGroups;  // patch group（場檔的 boundaryField 可用 group 名稱指定）

    // empty 在有限體積層的大小為 0（emptyFvPatch::size()）
    bool isEmpty() const { return type == "empty"; }
};


struct PolyMesh
{
    std::int32_t nPoints = 0;
    std::int32_t nCells = 0;
    std::int32_t nFaces = 0;
    std::int32_t nInternalFaces = 0;

    std::vector<double> points;             // 交錯 xyz，與檔案相同
    std::vector<std::int32_t> faceOffsets;  // nFaces + 1
    std::vector<std::int32_t> faceLabels;
    std::vector<std::int32_t> owner;        // nFaces
    std::vector<std::int32_t> neighbour;    // nInternalFaces
    std::vector<Patch> patches;

    // 讀檔時各檔案的格式（ascii／binary），記錄用
    std::string format;
};


// polyMeshDir：例如 <case>/constant/polyMesh
PolyMesh readPolyMesh(const std::string& polyMeshDir);

} // namespace gnc

#endif
