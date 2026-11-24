// GPU-Native-CFD，GPL-3.0-or-later。見 wallDist.hpp。
//
// 出處（Atlas upstream/openfoam，GPL-3.0-or-later）：
//   patchWave.C                    setChangedFaces、getValues、correct
//   FaceCellWave.C                 faceToCell、cellToFace、iterate
//   wallPointI.H                   update、valid、equal
//   cellDistFuncs.C、cellDistFuncsTemplates.C   correctBoundaryCells、smallestDist
//   faceIntersection.C、face.C、faceI.H         nearestPointClassify、centre、average
//   triangleI.H                    nearestPointClassify、centre
//   PrimitivePatchMeshData.C       meshPoints（首次出現次序）

#include "mesh/wallDist.hpp"

#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace gnc
{

namespace
{

constexpr double SMALL = 1.0e-15;
constexpr double VSMALL = 1.0e-300;
constexpr double ROOTVSMALL = 1.0e-150;
constexpr double GREAT = 1.0e+15;
constexpr double VGREAT = 1.0e+300;
constexpr double propagationTol = 0.01;

struct V
{
    double x, y, z;
};

V operator+(const V& a, const V& b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V operator-(const V& a, const V& b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V operator*(const double s, const V& a) { return {s*a.x, s*a.y, s*a.z}; }
V operator*(const V& a, const double s) { return {a.x*s, a.y*s, a.z*s}; }
V operator/(const V& a, const double s) { return {a.x/s, a.y/s, a.z/s}; }
double dot(const V& a, const V& b) { return (a.x*b.x + a.y*b.y) + a.z*b.z; }
double magSqr(const V& a) { return (a.x*a.x + a.y*a.y) + a.z*a.z; }
double mag(const V& a) { return std::sqrt(magSqr(a)); }
V cross(const V& a, const V& b)
{
    return {a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x};
}

V at(const std::vector<double>& xyz, const std::size_t i)
{
    return {xyz[3*i], xyz[3*i + 1], xyz[3*i + 2]};
}


struct WallPoint
{
    V origin{VGREAT, VGREAT, VGREAT};   // point::max
    double distSqr = -GREAT;

    bool valid() const { return distSqr > -SMALL; }

    bool equal(const WallPoint& o) const
    {
        return origin.x == o.origin.x && origin.y == o.origin.y && origin.z == o.origin.z;
    }

    // wallPoint::update(pt, w2, tol)
    bool update(const V& pt, const WallPoint& w2)
    {
        const double dist2 = magSqr(pt - w2.origin);
        if (!valid())
        {
            distSqr = dist2;
            origin = w2.origin;
            return true;
        }
        const double diff = distSqr - dist2;
        if (diff < 0)
        {
            return false;
        }
        if ((diff < SMALL) || ((distSqr > SMALL) && (diff/distSqr < propagationTol)))
        {
            return false;
        }
        distSqr = dist2;
        origin = w2.origin;
        return true;
    }
};


// triangle::nearestPointClassify 的距離
double triangleNearest(const V& a, const V& b, const V& c, const V& p)
{
    const V ab = b - a;
    const V ac = c - a;
    const V ap = p - a;
    const double d1 = dot(ab, ap);
    const double d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0)
    {
        return mag(a - p);
    }
    const V bp = p - b;
    const double d3 = dot(ab, bp);
    const double d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3)
    {
        return mag(b - p);
    }
    const double vc = d1*d4 - d3*d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0)
    {
        if ((d1 - d3) < ROOTVSMALL)
        {
            return mag(a - p);
        }
        const double v = d1/(d1 - d3);
        const V nearPt = a + v*ab;
        return mag(nearPt - p);
    }
    const V cp = p - c;
    const double d5 = dot(ab, cp);
    const double d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6)
    {
        return mag(c - p);
    }
    const double vb = d5*d2 - d1*d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0)
    {
        if ((d2 - d6) < ROOTVSMALL)
        {
            return mag(a - p);
        }
        const double w = d2/(d2 - d6);
        const V nearPt = a + w*ac;
        return mag(nearPt - p);
    }
    const double va = d3*d6 - d5*d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)
    {
        if (((d4 - d3) + (d5 - d6)) < ROOTVSMALL)
        {
            return mag(b - p);
        }
        const double w = (d4 - d3)/((d4 - d3) + (d5 - d6));
        const V nearPt = b + w*(c - b);
        return mag(nearPt - p);
    }
    if ((va + vb + vc) < ROOTVSMALL)
    {
        const V nearPt = (1.0/3.0)*(a + b + c);
        return mag(nearPt - p);
    }
    const double denom = 1.0/(va + vb + vc);
    const double v = vb*denom;
    const double w = vc*denom;
    const V nearPt = a + ab*v + ac*w;
    return mag(nearPt - p);
}


struct MeshView
{
    const std::vector<double>& points;
    const std::vector<std::int32_t>& faceOffsets;
    const std::vector<std::int32_t>& faceLabels;

    std::int32_t nPoints(const std::int32_t f) const
    {
        return faceOffsets[f + 1] - faceOffsets[f];
    }

    V point(const std::int32_t f, const std::int32_t k) const
    {
        return at(points, std::size_t(faceLabels[std::size_t(faceOffsets[f] + k)]));
    }

    // face::centre
    V centre(const std::int32_t f) const
    {
        const std::int32_t n = nPoints(f);
        if (n == 3)
        {
            return (1.0/3.0)*(point(f, 0) + point(f, 1) + point(f, 2));
        }
        V avg{0.0, 0.0, 0.0};
        for (std::int32_t k = 0; k < n; ++k)
        {
            avg = avg + point(f, k);
        }
        avg = avg/double(n);
        double sumA = 0;
        V sumAc{0.0, 0.0, 0.0};
        for (std::int32_t k = 0; k < n; ++k)
        {
            const V thisPoint = point(f, k);
            const V nextPoint = point(f, k == n - 1 ? 0 : k + 1);
            const V ttc = thisPoint + nextPoint + avg;
            const double ta = mag(cross(thisPoint - avg, nextPoint - avg));
            sumA += ta;
            sumAc = sumAc + ta*ttc;
        }
        if (sumA > VSMALL)
        {
            return sumAc/(3.0*sumA);
        }
        return avg;
    }

    // face::nearestPoint 的距離
    double nearest(const std::int32_t f, const V& p) const
    {
        const std::int32_t n = nPoints(f);
        if (n == 3)
        {
            return triangleNearest(point(f, 0), point(f, 1), point(f, 2), p);
        }
        const V ctr = centre(f);
        double nearestDist = GREAT;
        for (std::int32_t k = 0; k < n; ++k)
        {
            const V nextPoint = point(f, k == n - 1 ? 0 : k + 1);
            const double d = triangleNearest(point(f, k), nextPoint, ctr, p);
            if (std::fabs(d) < std::fabs(nearestDist))
            {
                nearestDist = d;
            }
        }
        return nearestDist;
    }
};

} // namespace


WallDistance meshWaveWallDistance
(
    const PolyMesh& mesh,
    const std::vector<double>& cellCentres,
    const std::vector<double>& faceCentres
)
{
    const std::int32_t nCells = mesh.nCells;
    const std::int32_t nFaces = mesh.nFaces;
    const std::int32_t nInternal = mesh.nInternalFaces;
    for (const Patch& p : mesh.patches)
    {
        if (p.type == "cyclic" || p.type == "processor" || p.type == "cyclicAMI")
        {
            throw std::runtime_error("meshWave wall distance: coupled patch " + p.name);
        }
    }

    // cells()：owner 面遞增，再 neighbour 面遞增（primitiveMesh::calcCells）
    std::vector<std::int32_t> cellOff(std::size_t(nCells) + 1, 0);
    for (std::int32_t f = 0; f < nFaces; ++f)
    {
        ++cellOff[std::size_t(mesh.owner[f]) + 1];
    }
    for (std::int32_t f = 0; f < nInternal; ++f)
    {
        ++cellOff[std::size_t(mesh.neighbour[f]) + 1];
    }
    for (std::int32_t c = 0; c < nCells; ++c)
    {
        cellOff[c + 1] += cellOff[c];
    }
    std::vector<std::int32_t> cellFaces(static_cast<std::size_t>(cellOff[nCells]));
    {
        std::vector<std::int32_t> fill(cellOff.begin(), cellOff.end() - 1);
        for (std::int32_t f = 0; f < nFaces; ++f)
        {
            cellFaces[std::size_t(fill[mesh.owner[f]]++)] = f;
        }
        for (std::int32_t f = 0; f < nInternal; ++f)
        {
            cellFaces[std::size_t(fill[mesh.neighbour[f]]++)] = f;
        }
    }

    // patchWave::setChangedFaces：type wall 的 patch，依 patch 次序
    std::vector<std::int32_t> wallFaces;
    for (const Patch& p : mesh.patches)
    {
        if (p.type == "wall")
        {
            for (std::int32_t i = 0; i < p.size; ++i)
            {
                wallFaces.push_back(p.start + i);
            }
        }
    }

    // FaceCellWave<wallPoint>
    std::vector<WallPoint> faceInfo(static_cast<std::size_t>(nFaces)), cellInfo(static_cast<std::size_t>(nCells));
    std::vector<char> changedFace(std::size_t(nFaces), 0), changedCell(std::size_t(nCells), 0);
    std::vector<std::int32_t> changedFaces, changedCells;
    for (const std::int32_t f : wallFaces)
    {
        WallPoint w;
        w.origin = at(faceCentres, std::size_t(f));
        w.distSqr = 0.0;
        faceInfo[std::size_t(f)] = w;
        changedFace[std::size_t(f)] = 1;
        changedFaces.push_back(f);
    }
    const auto updateCell = [&](const std::int32_t c, const WallPoint& info)
    {
        if (cellInfo[std::size_t(c)].update(at(cellCentres, std::size_t(c)), info))
        {
            if (!changedCell[std::size_t(c)])
            {
                changedCell[std::size_t(c)] = 1;
                changedCells.push_back(c);
            }
        }
    };
    const std::int32_t maxIter = nCells + 1;
    for (std::int32_t iter = 0; iter < maxIter; ++iter)
    {
        // faceToCell
        for (const std::int32_t f : changedFaces)
        {
            const WallPoint newInfo = faceInfo[std::size_t(f)];
            const std::int32_t o = mesh.owner[f];
            if (!cellInfo[std::size_t(o)].equal(newInfo))
            {
                updateCell(o, newInfo);
            }
            if (f < nInternal)
            {
                const std::int32_t nb = mesh.neighbour[f];
                if (!cellInfo[std::size_t(nb)].equal(newInfo))
                {
                    updateCell(nb, newInfo);
                }
            }
            changedFace[std::size_t(f)] = 0;
        }
        changedFaces.clear();
        const std::size_t nChangedCells = changedCells.size();
        std::size_t nChangedFaces = 0;
        if (nChangedCells)
        {
            // cellToFace
            for (const std::int32_t c : changedCells)
            {
                const WallPoint newInfo = cellInfo[std::size_t(c)];
                for (std::int32_t q = cellOff[c]; q < cellOff[c + 1]; ++q)
                {
                    const std::int32_t f = cellFaces[std::size_t(q)];
                    WallPoint& cur = faceInfo[std::size_t(f)];
                    if (!cur.equal(newInfo))
                    {
                        if (cur.update(at(faceCentres, std::size_t(f)), newInfo))
                        {
                            if (!changedFace[std::size_t(f)])
                            {
                                changedFace[std::size_t(f)] = 1;
                                changedFaces.push_back(f);
                            }
                        }
                    }
                }
                changedCell[std::size_t(c)] = 0;
            }
            changedCells.clear();
            nChangedFaces = changedFaces.size();
        }
        if (!nChangedCells || !nChangedFaces)
        {
            break;
        }
    }

    // patchWave::getValues
    WallDistance out;
    out.cell.resize(std::size_t(nCells));
    out.face.assign(std::size_t(nFaces), 0.0);
    out.nearWall.assign(std::size_t(nFaces), 0.0);
    for (std::int32_t c = 0; c < nCells; ++c)
    {
        const WallPoint& w = cellInfo[std::size_t(c)];
        if (w.valid())
        {
            out.cell[std::size_t(c)] = std::sqrt(w.distSqr);
        }
        else
        {
            out.cell[std::size_t(c)] = w.distSqr;
            ++out.nUnset;
        }
    }
    for (std::int32_t f = nInternal; f < nFaces; ++f)
    {
        const WallPoint& w = faceInfo[std::size_t(f)];
        if (w.valid())
        {
            out.face[std::size_t(f)] = std::sqrt(w.distSqr) + SMALL;
        }
        else
        {
            out.face[std::size_t(f)] = w.distSqr;
            ++out.nUnset;
        }
    }

    // cellDistFuncs::correctBoundaryCells（doPointCells true）
    if (wallFaces.empty())
    {
        return out;
    }
    const MeshView mv{mesh.points, mesh.faceOffsets, mesh.faceLabels};
    // 牆 patch 的局部點（首次出現次序）與 pointFaces
    std::unordered_map<std::int32_t, std::int32_t> localOf;
    std::vector<std::int32_t> meshPoints;
    std::vector<std::vector<std::int32_t>> pointFaces;
    for (std::size_t i = 0; i < wallFaces.size(); ++i)
    {
        const std::int32_t f = wallFaces[i];
        for (std::int32_t q = mesh.faceOffsets[f]; q < mesh.faceOffsets[f + 1]; ++q)
        {
            const std::int32_t pt = mesh.faceLabels[std::size_t(q)];
            auto it = localOf.find(pt);
            std::int32_t lp;
            if (it == localOf.end())
            {
                lp = std::int32_t(meshPoints.size());
                localOf.emplace(pt, lp);
                meshPoints.push_back(pt);
                pointFaces.emplace_back();
            }
            else
            {
                lp = it->second;
            }
            std::vector<std::int32_t>& pf = pointFaces[std::size_t(lp)];
            if (pf.empty() || pf.back() != std::int32_t(i))
            {
                pf.push_back(std::int32_t(i));
            }
        }
    }
    const auto smallestDist = [&](const V& p, const std::vector<std::int32_t>& localFaces)
    {
        double minDist = GREAT;
        for (const std::int32_t lf : localFaces)
        {
            const double d = mv.nearest(wallFaces[std::size_t(lf)], p);
            if (d < minDist)
            {
                minDist = d;
            }
        }
        return minDist;
    };

    std::vector<char> corrected(std::size_t(nCells), 0);
    // 有面在牆上的格：該牆面與共用頂點的牆面（getPointNeighbours；只取最小值，
    // 與次序無關）
    std::vector<std::int32_t> nbs;
    std::vector<char> inNbs(wallFaces.size(), 0);
    for (std::size_t i = 0; i < wallFaces.size(); ++i)
    {
        const std::int32_t f = wallFaces[i];
        nbs.clear();
        for (std::int32_t q = mesh.faceOffsets[f]; q < mesh.faceOffsets[f + 1]; ++q)
        {
            const std::int32_t lp = localOf.at(mesh.faceLabels[std::size_t(q)]);
            for (const std::int32_t lf : pointFaces[std::size_t(lp)])
            {
                if (!inNbs[std::size_t(lf)])
                {
                    inNbs[std::size_t(lf)] = 1;
                    nbs.push_back(lf);
                }
            }
        }
        for (const std::int32_t lf : nbs)
        {
            inNbs[std::size_t(lf)] = 0;
        }
        const std::int32_t c = mesh.owner[f];
        const double d = smallestDist(at(cellCentres, std::size_t(c)), nbs);
        out.cell[std::size_t(c)] = d;
        out.nearWall[std::size_t(f)] = d;      // nearWallDist 的 ypatch[patchFacei]
        corrected[std::size_t(c)] = 1;
    }

    // 有頂點在牆上的格：依局部點次序，第一個碰到它的點
    std::unordered_map<std::int32_t, std::vector<std::int32_t>> pointCells;
    for (std::int32_t f = 0; f < nFaces; ++f)
    {
        for (std::int32_t q = mesh.faceOffsets[f]; q < mesh.faceOffsets[f + 1]; ++q)
        {
            const std::int32_t pt = mesh.faceLabels[std::size_t(q)];
            if (!localOf.count(pt))
            {
                continue;
            }
            std::vector<std::int32_t>& pc = pointCells[pt];
            pc.push_back(mesh.owner[f]);
            if (f < nInternal)
            {
                pc.push_back(mesh.neighbour[f]);
            }
        }
    }
    for (std::size_t lp = 0; lp < meshPoints.size(); ++lp)
    {
        for (const std::int32_t c : pointCells[meshPoints[lp]])
        {
            if (corrected[std::size_t(c)])
            {
                continue;
            }
            out.cell[std::size_t(c)] = smallestDist(at(cellCentres, std::size_t(c)), pointFaces[lp]);
            corrected[std::size_t(c)] = 1;
        }
    }
    return out;
}

} // namespace gnc
