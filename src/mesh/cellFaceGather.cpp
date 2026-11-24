// GPU-Native-CFD，GPL-3.0-or-later。見 cellFaceGather.hpp。

#include "mesh/cellFaceGather.hpp"

namespace gnc
{

CellFaceGather buildCellFaceGather(const PolyMesh& mesh)
{
    CellFaceGather g;
    const std::size_t nCells = std::size_t(mesh.nCells);

    g.offsets.assign(nCells + 1, 0);
    for (std::int32_t f = 0; f < mesh.nFaces; ++f)
    {
        ++g.offsets[std::size_t(mesh.owner[f]) + 1];
        if (f < mesh.nInternalFaces)
        {
            ++g.offsets[std::size_t(mesh.neighbour[f]) + 1];
        }
    }
    for (std::size_t c = 0; c < nCells; ++c)
    {
        g.offsets[c + 1] += g.offsets[c];
    }

    const std::size_t n = std::size_t(g.offsets[nCells]);
    g.faces.resize(n);
    g.signs.resize(n);

    // 依面編號遞增寫入，每格的清單因此自然排序
    std::vector<std::int32_t> cursor(g.offsets.begin(), g.offsets.end() - 1);
    for (std::int32_t f = 0; f < mesh.nFaces; ++f)
    {
        const std::int32_t o = mesh.owner[f];
        g.faces[std::size_t(cursor[o])] = f;
        g.signs[std::size_t(cursor[o])] = 1;
        ++cursor[o];
        if (f < mesh.nInternalFaces)
        {
            const std::int32_t nb = mesh.neighbour[f];
            g.faces[std::size_t(cursor[nb])] = f;
            g.signs[std::size_t(cursor[nb])] = -1;
            ++cursor[nb];
        }
    }

    return g;
}

} // namespace gnc
