// GPU-Native-CFD，GPL-3.0-or-later。見 amg.cuh。

#include "fvm/amg.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>

namespace gnc
{

namespace
{

constexpr int BS = 256;
constexpr double GREAT = 1.0e+15;

int nBlocks(const std::size_t n)
{
    return int((n + BS - 1)/BS);
}

// ---------------------------------------------------------------------------
// 主機端：聚合

struct HostMesh
{
    int nCells = 0;
    std::vector<int> lower;     // owner（lduAddressing::lowerAddr）
    std::vector<int> upper;     // neighbour（upperAddr）
};

// pairGAMGAgglomeration::agglomerate（單層；tol = 0、不重新編號）
std::vector<int> pairAgglomerate
(
    const HostMesh& m,
    const std::vector<double>& w,
    const bool forward,
    int& nCoarse
)
{
    const int nC = m.nCells;
    const int nF = int(m.lower.size());
    // 每格的面：先 upperAddr（本格為 neighbour）的面，再 lowerAddr 的面，依面號
    std::vector<int> off(std::size_t(nC) + 1, 0), cf(std::size_t(2*nF));
    {
        std::vector<int> n(std::size_t(nC), 0);
        for (int f = 0; f < nF; ++f)
        {
            ++n[std::size_t(m.upper[std::size_t(f)])];
        }
        for (int f = 0; f < nF; ++f)
        {
            ++n[std::size_t(m.lower[std::size_t(f)])];
        }
        for (int c = 0; c < nC; ++c)
        {
            off[std::size_t(c) + 1] = off[std::size_t(c)] + n[std::size_t(c)];
        }
        std::fill(n.begin(), n.end(), 0);
        for (int f = 0; f < nF; ++f)
        {
            const int c = m.upper[std::size_t(f)];
            cf[std::size_t(off[std::size_t(c)] + n[std::size_t(c)]++)] = f;
        }
        for (int f = 0; f < nF; ++f)
        {
            const int c = m.lower[std::size_t(f)];
            cf[std::size_t(off[std::size_t(c)] + n[std::size_t(c)]++)] = f;
        }
    }
    std::vector<int> map(std::size_t(nC), -1);
    nCoarse = 0;
    for (int k = 0; k < nC; ++k)
    {
        const int c = forward ? k : nC - k - 1;
        if (map[std::size_t(c)] >= 0)
        {
            continue;
        }
        int match = -1;
        double maxW = -GREAT;
        for (int q = off[std::size_t(c)]; q < off[std::size_t(c) + 1]; ++q)
        {
            const int f = cf[std::size_t(q)];
            if
            (
                map[std::size_t(m.upper[std::size_t(f)])] < 0
             && map[std::size_t(m.lower[std::size_t(f)])] < 0
             && w[std::size_t(f)] > maxW
            )
            {
                match = f;
                maxW = w[std::size_t(f)];
            }
        }
        if (match >= 0)
        {
            map[std::size_t(m.upper[std::size_t(match)])] = nCoarse;
            map[std::size_t(m.lower[std::size_t(match)])] = nCoarse;
            ++nCoarse;
        }
        else
        {
            int cl = -1;
            double cmax = -GREAT;
            for (int q = off[std::size_t(c)]; q < off[std::size_t(c) + 1]; ++q)
            {
                const int f = cf[std::size_t(q)];
                if (w[std::size_t(f)] > cmax)
                {
                    cl = f;
                    cmax = w[std::size_t(f)];
                }
            }
            if (cl >= 0)
            {
                map[std::size_t(c)] = std::max
                (
                    map[std::size_t(m.upper[std::size_t(cl)])],
                    map[std::size_t(m.lower[std::size_t(cl)])]
                );
            }
        }
    }
    for (int k = 0; k < nC; ++k)
    {
        const int c = forward ? k : nC - k - 1;
        if (map[std::size_t(c)] < 0)
        {
            map[std::size_t(c)] = nCoarse++;
        }
    }
    if (!forward)
    {
        for (int& v : map)
        {
            v = nCoarse - 1 - v;
        }
    }
    return map;
}

// GAMGAgglomeration::agglomerateLduAddressing：粗面、細面的限制（< 0 時為
// -(粗格 + 1) 的內部面）與翻轉
struct CoarseAddr
{
    HostMesh coarse;
    std::vector<int> faceRestrict;
    std::vector<char> flip;
};

CoarseAddr agglomerateAddressing(const HostMesh& fine, const std::vector<int>& map, const int nCoarse)
{
    const int nF = int(fine.lower.size());
    CoarseAddr a;
    a.faceRestrict.assign(std::size_t(nF), 0);
    std::vector<std::vector<int>> cellFaces(static_cast<std::size_t>(nCoarse));
    std::vector<int> initNeighb;
    for (int f = 0; f < nF; ++f)
    {
        const int rU = map[std::size_t(fine.upper[std::size_t(f)])];
        const int rL = map[std::size_t(fine.lower[std::size_t(f)])];
        if (rU == rL)
        {
            a.faceRestrict[std::size_t(f)] = -(rU + 1);
            continue;
        }
        const int cOwn = std::min(rU, rL);
        const int cNei = std::max(rU, rL);
        int found = -1;
        for (const int cfi : cellFaces[std::size_t(cOwn)])
        {
            if (initNeighb[std::size_t(cfi)] == cNei)
            {
                found = cfi;
                break;
            }
        }
        if (found < 0)
        {
            found = int(initNeighb.size());
            initNeighb.push_back(cNei);
            cellFaces[std::size_t(cOwn)].push_back(found);
        }
        a.faceRestrict[std::size_t(f)] = found;
    }
    const int nCF = int(initNeighb.size());
    std::vector<int> faceMap(static_cast<std::size_t>(nCF));
    a.coarse.nCells = nCoarse;
    a.coarse.lower.resize(std::size_t(nCF));
    a.coarse.upper.resize(std::size_t(nCF));
    int k = 0;
    for (int c = 0; c < nCoarse; ++c)
    {
        for (const int cfi : cellFaces[std::size_t(c)])
        {
            a.coarse.lower[std::size_t(k)] = c;
            a.coarse.upper[std::size_t(k)] = initNeighb[std::size_t(cfi)];
            faceMap[std::size_t(cfi)] = k;
            ++k;
        }
    }
    a.flip.assign(std::size_t(nF), 0);
    for (int f = 0; f < nF; ++f)
    {
        int& r = a.faceRestrict[std::size_t(f)];
        if (r >= 0)
        {
            r = faceMap[std::size_t(r)];
            const int rU = map[std::size_t(fine.upper[std::size_t(f)])];
            const int rL = map[std::size_t(fine.lower[std::size_t(f)])];
            a.flip[std::size_t(f)] =
                (a.coarse.lower[std::size_t(r)] == rU && a.coarse.upper[std::size_t(r)] == rL) ? 1 : 0;
        }
    }
    return a;
}

// 著色：二分圖時兩色，否則貪婪（依格編號，取相鄰格未用的最小色）
std::vector<int> greedyColor(const HostMesh& m, int& nColors)
{
    const int nC = m.nCells;
    std::vector<int> off(std::size_t(nC) + 1, 0), nb(2*m.lower.size());
    {
        std::vector<int> n(std::size_t(nC), 0);
        for (std::size_t f = 0; f < m.lower.size(); ++f)
        {
            ++n[std::size_t(m.lower[f])];
            ++n[std::size_t(m.upper[f])];
        }
        for (int c = 0; c < nC; ++c)
        {
            off[std::size_t(c) + 1] = off[std::size_t(c)] + n[std::size_t(c)];
        }
        std::fill(n.begin(), n.end(), 0);
        for (std::size_t f = 0; f < m.lower.size(); ++f)
        {
            const int a = m.lower[f], b = m.upper[f];
            nb[std::size_t(off[std::size_t(a)] + n[std::size_t(a)]++)] = b;
            nb[std::size_t(off[std::size_t(b)] + n[std::size_t(b)]++)] = a;
        }
    }
    std::vector<int> color(std::size_t(nC), -1);
    // 先試兩色（相鄰格的圖是二分圖時，例如結構化的四邊形網格）：依格編號由未著色的格做 BFS
    // （N-027）；不是二分圖時改用下面的貪婪著色
    {
        bool bipartite = true;
        std::vector<int> queue;
        queue.reserve(std::size_t(nC));
        for (int s = 0; s < nC && bipartite; ++s)
        {
            if (color[std::size_t(s)] >= 0)
            {
                continue;
            }
            color[std::size_t(s)] = 0;
            queue.assign(1, s);
            for (std::size_t h = 0; h < queue.size() && bipartite; ++h)
            {
                const int c = queue[h];
                for (int q = off[std::size_t(c)]; q < off[std::size_t(c) + 1]; ++q)
                {
                    const int o = nb[std::size_t(q)];
                    if (color[std::size_t(o)] < 0)
                    {
                        color[std::size_t(o)] = 1 - color[std::size_t(c)];
                        queue.push_back(o);
                    }
                    else if (color[std::size_t(o)] == color[std::size_t(c)])
                    {
                        bipartite = false;
                        break;
                    }
                }
            }
        }
        if (bipartite)
        {
            nColors = nC > 1 ? 2 : nC;
            return color;
        }
        std::fill(color.begin(), color.end(), -1);
    }
    // 貪婪著色，依兩種走訪次序各做一次，取色數少的（相同時用格編號的次序）：格編號、鄰格數由大到小
    // （Welsh–Powell；N-028：粗層的色數少，V-cycle 的階段就少）
    const auto greedy = [&](const std::vector<int>& order, std::vector<int>& col)
    {
        col.assign(std::size_t(nC), -1);
        int nCol = 0;
        std::vector<char> used;
        for (const int c : order)
        {
            used.assign(std::size_t(nCol) + 1, 0);
            for (int q = off[std::size_t(c)]; q < off[std::size_t(c) + 1]; ++q)
            {
                const int k = col[std::size_t(nb[std::size_t(q)])];
                if (k >= 0)
                {
                    used[std::size_t(k)] = 1;
                }
            }
            int k = 0;
            while (used[std::size_t(k)])
            {
                ++k;
            }
            col[std::size_t(c)] = k;
            nCol = std::max(nCol, k + 1);
        }
        return nCol;
    };
    std::vector<int> order(static_cast<std::size_t>(nC));
    for (int c = 0; c < nC; ++c)
    {
        order[std::size_t(c)] = c;
    }
    nColors = greedy(order, color);
    // GNC_AMG_WP=0：只用格編號的次序（與 N-027 之前相同，對照用）
    static const bool wp = []
    {
        const char* e = std::getenv("GNC_AMG_WP");
        return !(e && std::string(e) == "0");
    }();
    if (!wp)
    {
        return color;
    }
    std::stable_sort(order.begin(), order.end(), [&](const int a, const int b)
    {
        return off[std::size_t(a) + 1] - off[std::size_t(a)] > off[std::size_t(b) + 1] - off[std::size_t(b)];
    });
    std::vector<int> color2;
    const int n2 = greedy(order, color2);
    if (n2 < nColors)
    {
        nColors = n2;
        color.swap(color2);
    }
    return color;
}

// 一層的裝置端定址：原編號的 LDU 定址，與套用用的列次序（依顏色）及 CSR。回傳 perm（列 → 格）
std::vector<int> uploadLevel(const HostMesh& m, AmgLevel& L)
{
    L.nCells = m.nCells;
    L.nFaces = int(m.lower.size());
    // 每格的鄰格清單：code < 0 時 f = -(code + 1)、本格是 owner、係數取 upper[f]；
    // 否則本格是 neighbour、係數取 lower[f]
    std::vector<int> off(std::size_t(m.nCells) + 1, 0), code(2*m.lower.size());
    {
        std::vector<int> n(std::size_t(m.nCells), 0);
        for (std::size_t f = 0; f < m.lower.size(); ++f)
        {
            ++n[std::size_t(m.lower[f])];
            ++n[std::size_t(m.upper[f])];
        }
        for (int c = 0; c < m.nCells; ++c)
        {
            off[std::size_t(c) + 1] = off[std::size_t(c)] + n[std::size_t(c)];
        }
        std::fill(n.begin(), n.end(), 0);
        for (std::size_t f = 0; f < m.lower.size(); ++f)
        {
            const int a = m.lower[f], b = m.upper[f];
            code[std::size_t(off[std::size_t(a)] + n[std::size_t(a)]++)] = -(int(f) + 1);
            code[std::size_t(off[std::size_t(b)] + n[std::size_t(b)]++)] = int(f);
        }
    }
    L.lowerAddr.upload(m.lower);
    L.upperAddr.upload(m.upper);
    int nCol = 0;
    const std::vector<int> color = greedyColor(m, nCol);
    L.colorOff.assign(std::size_t(nCol) + 1, 0);
    for (const int k : color)
    {
        ++L.colorOff[std::size_t(k) + 1];
    }
    for (int k = 0; k < nCol; ++k)
    {
        L.colorOff[std::size_t(k) + 1] += L.colorOff[std::size_t(k)];
    }
    // 列：依顏色，同色內依格編號
    std::vector<int> perm(std::size_t(m.nCells)), inv(std::size_t(m.nCells));
    {
        std::vector<int> fill(L.colorOff.begin(), L.colorOff.end() - 1);
        for (int c = 0; c < m.nCells; ++c)
        {
            const int i = fill[std::size_t(color[std::size_t(c)])]++;
            perm[std::size_t(i)] = c;
            inv[std::size_t(c)] = i;
        }
    }
    std::vector<int> rowPtr(std::size_t(m.nCells) + 1, 0), col(code.size()), src(code.size());
    for (int i = 0; i < m.nCells; ++i)
    {
        const int c = perm[std::size_t(i)];
        int j = rowPtr[std::size_t(i)];
        for (int q = off[std::size_t(c)]; q < off[std::size_t(c) + 1]; ++q, ++j)
        {
            const int s = code[std::size_t(q)];
            const int nb = s < 0 ? m.upper[std::size_t(-(s + 1))] : m.lower[std::size_t(s)];
            col[std::size_t(j)] = inv[std::size_t(nb)];
            src[std::size_t(j)] = s;
        }
        rowPtr[std::size_t(i) + 1] = j;
    }
    L.perm.upload(perm);
    L.invPerm.upload(inv);
    L.rowPtr.upload(rowPtr);
    L.col.upload(col);
    L.src.upload(src);
    return perm;
}

// 細層 → 粗層：原編號的累加表（粗矩陣），與列的對應（V-cycle 的限制、延拓）
void uploadRestriction
(
    const HostMesh& fine,
    const CoarseAddr& a,
    const std::vector<int>& map,
    const std::vector<int>& finePerm,
    const std::vector<int>& coarsePerm,
    AmgLevel& L
)
{
    const int nCc = a.coarse.nCells;
    const int nCf = int(a.coarse.lower.size());
    std::vector<int> rOff(std::size_t(nCc) + 1, 0), iOff(std::size_t(nCc) + 1, 0), fOff(std::size_t(nCf) + 1, 0);
    for (const int c : map)
    {
        ++rOff[std::size_t(c) + 1];
    }
    for (std::size_t f = 0; f < a.faceRestrict.size(); ++f)
    {
        const int r = a.faceRestrict[f];
        if (r < 0)
        {
            ++iOff[std::size_t(-(r + 1)) + 1];
        }
        else
        {
            ++fOff[std::size_t(r) + 1];
        }
    }
    for (int c = 0; c < nCc; ++c)
    {
        rOff[std::size_t(c) + 1] += rOff[std::size_t(c)];
        iOff[std::size_t(c) + 1] += iOff[std::size_t(c)];
    }
    for (int f = 0; f < nCf; ++f)
    {
        fOff[std::size_t(f) + 1] += fOff[std::size_t(f)];
    }
    std::vector<int> rCells(map.size()), iFaces(std::size_t(iOff.back())), fFaces(std::size_t(fOff.back()));
    std::vector<int> rf(rOff.begin(), rOff.end() - 1), inf(iOff.begin(), iOff.end() - 1), ff(fOff.begin(), fOff.end() - 1);
    for (int c = 0; c < fine.nCells; ++c)
    {
        rCells[std::size_t(rf[std::size_t(map[std::size_t(c)])]++)] = c;
    }
    for (std::size_t f = 0; f < a.faceRestrict.size(); ++f)
    {
        const int r = a.faceRestrict[f];
        if (r < 0)
        {
            iFaces[std::size_t(inf[std::size_t(-(r + 1))]++)] = int(f);
        }
        else
        {
            fFaces[std::size_t(ff[std::size_t(r)]++)] = a.flip[f] ? -(int(f) + 1) : int(f);
        }
    }
    L.rOff.upload(rOff);
    L.rCells.upload(rCells);
    L.iOff.upload(iOff);
    L.iFaces.upload(iFaces);
    L.fOff.upload(fOff);
    L.fFaces.upload(fFaces);

    // 列的對應：parent[細列] = 粗列；cRows 為每個粗列的細列（遞增）
    std::vector<int> coarseInv(static_cast<std::size_t>(nCc));
    for (int i = 0; i < nCc; ++i)
    {
        coarseInv[std::size_t(coarsePerm[std::size_t(i)])] = i;
    }
    std::vector<int> parent(std::size_t(fine.nCells)), cOff(std::size_t(nCc) + 1, 0);
    for (int i = 0; i < fine.nCells; ++i)
    {
        parent[std::size_t(i)] = coarseInv[std::size_t(map[std::size_t(finePerm[std::size_t(i)])])];
        ++cOff[std::size_t(parent[std::size_t(i)]) + 1];
    }
    for (int I = 0; I < nCc; ++I)
    {
        cOff[std::size_t(I) + 1] += cOff[std::size_t(I)];
    }
    std::vector<int> cRows(std::size_t(fine.nCells));
    std::vector<int> cf(cOff.begin(), cOff.end() - 1);
    for (int i = 0; i < fine.nCells; ++i)
    {
        cRows[std::size_t(cf[std::size_t(parent[std::size_t(i)])]++)] = i;
    }
    L.parent.upload(parent);
    L.cOff.upload(cOff);
    L.cRows.upload(cRows);
}

template<class T>
void allocateBuffers(AmgHierarchy& h, AmgBuffers<T>& buf)
{
    const std::size_t nL = h.levels.size();
    for (auto* v : {&buf.val, &buf.d, &buf.invd, &buf.x, &buf.b, &buf.t})
    {
        v->resize(nL);
    }
    buf.colorOffD.resize(nL);
    std::vector<AmgLevelView<T>> views(nL);
    for (std::size_t l = 0; l < nL; ++l)
    {
        AmgLevel& L = h.levels[l];
        const std::size_t nc = std::size_t(L.nCells);
        buf.val[l].allocate(L.col.size());
        buf.d[l].allocate(nc);
        buf.invd[l].allocate(nc);
        buf.x[l].allocate(nc);
        buf.b[l].allocate(nc);
        buf.t[l].allocate(nc);
        buf.colorOffD[l].upload(L.colorOff);
    }
    for (std::size_t l = 0; l < nL; ++l)
    {
        AmgLevel& L = h.levels[l];
        AmgLevelView<T>& v = views[l];
        v.nRows = L.nCells;
        v.nColors = std::int32_t(L.colorOff.size()) - 1;
        v.colorOff = buf.colorOffD[l].data();
        v.rowPtr = L.rowPtr.data();
        v.col = L.col.data();
        v.val = buf.val[l].data();
        v.d = buf.d[l].data();
        v.invd = buf.invd[l].data();
        v.x = buf.x[l].data();
        v.b = buf.b[l].data();
        v.t = buf.t[l].data();
        v.parent = l + 1 < nL ? h.levels[l + 1].parent.data() : nullptr;
        v.cOff = l + 1 < nL ? h.levels[l + 1].cOff.data() : nullptr;
        v.cRows = l + 1 < nL ? h.levels[l + 1].cRows.data() : nullptr;
    }
    buf.views.upload(views);
}

std::unique_ptr<AmgHierarchy> buildHierarchy(const DeviceMesh& dm, const AmgControls& ctrl)
{
    auto h = std::make_unique<AmgHierarchy>();
    h->controls = ctrl;
    const int nIF = dm.nInternalFaces;
    HostMesh fine;
    fine.nCells = dm.nCells;
    fine.lower = dm.owner.download();
    fine.lower.resize(std::size_t(nIF));
    fine.upper = dm.neighbour.download();
    fine.upper.resize(std::size_t(nIF));

    // faceAreaPair 的起始面權重：mag(cmptMultiply(Sf/sqrt(magSf), (1, 1.01, 1.02)))
    std::vector<double> w(static_cast<std::size_t>(nIF));
    {
        const std::vector<double> sx = dm.Sf.x.download();
        const std::vector<double> sy = dm.Sf.y.download();
        const std::vector<double> sz = dm.Sf.z.download();
        const std::vector<double> ms = dm.magSf.download();
        for (int f = 0; f < nIF; ++f)
        {
            const double r = std::sqrt(ms[std::size_t(f)]);
            const double x = (sx[std::size_t(f)]/r)*1.0;
            const double y = (sy[std::size_t(f)]/r)*1.01;
            const double z = (sz[std::size_t(f)]/r)*1.02;
            w[std::size_t(f)] = std::sqrt(x*x + y*y + z*z);
        }
    }

    const int nCoarsest = std::max(1, std::min(fine.nCells/2, ctrl.nCellsInCoarsestLevel));
    const int maxLevels = 50;
    std::vector<HostMesh> created{fine};
    std::vector<std::vector<int>> perms;
    h->levels.emplace_back();
    perms.push_back(uploadLevel(fine, h->levels.back()));

    HostMesh cur = fine;
    bool forward = true;
    int nPair = 0;
    bool pending = false;
    std::vector<int> composite;
    const auto finalize = [&]()
    {
        int nC = 0;
        for (const int v : composite)
        {
            nC = std::max(nC, v + 1);
        }
        const CoarseAddr a = agglomerateAddressing(created.back(), composite, nC);
        h->levels.emplace_back();
        perms.push_back(uploadLevel(a.coarse, h->levels.back()));
        uploadRestriction(created.back(), a, composite, perms[perms.size() - 2], perms.back(), h->levels.back());
        created.push_back(a.coarse);
        pending = false;
    };
    while (int(created.size()) + (pending ? 1 : 0) < maxLevels)
    {
        int nCoarse = 0;
        const std::vector<int> map = pairAgglomerate(cur, w, forward, nCoarse);
        forward = !forward;
        if (nCoarse < nCoarsest || nCoarse >= cur.nCells)
        {
            break;
        }
        const CoarseAddr pa = agglomerateAddressing(cur, map, nCoarse);
        std::vector<double> cw(pa.coarse.lower.size(), 0.0);
        for (std::size_t f = 0; f < pa.faceRestrict.size(); ++f)
        {
            if (pa.faceRestrict[f] >= 0)
            {
                cw[std::size_t(pa.faceRestrict[f])] += w[f];
            }
        }
        w = std::move(cw);
        if (nPair % ctrl.mergeLevels)
        {
            for (int& v : composite)
            {
                v = map[std::size_t(v)];
            }
        }
        else
        {
            if (pending)
            {
                finalize();
            }
            composite = map;
            pending = true;
        }
        cur = pa.coarse;
        ++nPair;
    }
    if (pending)
    {
        finalize();
    }
    const int nCoarsestCells = h->levels.back().nCells;
    if (nCoarsestCells > ctrl.maxCoarsest)
    {
        throw std::runtime_error
        (
            "AMG: coarsest level has " + std::to_string(nCoarsestCells) + " cells (limit "
          + std::to_string(ctrl.maxCoarsest) + ")"
        );
    }
    const std::size_t nL = h->levels.size();
    for (auto* v : {&h->diag, &h->lower, &h->upper})
    {
        v->resize(nL);
    }
    for (std::size_t l = 0; l < nL; ++l)
    {
        h->diag[l].allocate(std::size_t(h->levels[l].nCells));
        h->lower[l].allocate(std::size_t(h->levels[l].nFaces));
        h->upper[l].allocate(std::size_t(h->levels[l].nFaces));
    }
    if (ctrl.precision == 64)
    {
        allocateBuffers(*h, h->f64);
    }
    else
    {
        allocateBuffers(*h, h->f32);
    }
    h->dense.allocate(std::size_t(nCoarsestCells)*std::size_t(nCoarsestCells));
    // 單一 block 的子循環：由第一個格數不超過 blockCells 的層起（各層格數遞減）
    h->blockLevel = int(nL);
    if (ctrl.blockCells > 0)
    {
        for (std::size_t l = 0; l < nL; ++l)
        {
            if (h->levels[l].nCells <= ctrl.blockCells)
            {
                h->blockLevel = int(l);
                break;
            }
        }
    }
    std::printf
    (
        "AMG hierarchy: %d levels (mergeLevels %d, FP%d, omega %g, sweeps %d, Jacobi %g from level %d, block from level %d):",
        int(nL), ctrl.mergeLevels, ctrl.precision == 64 ? 64 : 32, ctrl.omega, ctrl.nSweeps, ctrl.jacobi,
        ctrl.jacobiFrom, h->blockLevel
    );
    for (const AmgLevel& L : h->levels)
    {
        std::printf(" %d/%dc", L.nCells, int(L.colorOff.size()) - 1);
    }
    std::printf("\n");
    return h;
}


// ---------------------------------------------------------------------------
// 裝置端

__device__ __forceinline__ float fmaT(const float a, const float b, const float c)
{
    return fmaf(a, b, c);
}

__device__ __forceinline__ double fmaT(const double a, const double b, const double c)
{
    return fma(a, b, c);
}

// 粗 diag = Σ 細 diag ＋ Σ 內部細面的 upper + lower（原編號，FP64）
__global__ void kCoarseDiag
(
    const std::int32_t nC,
    const std::int32_t* __restrict__ rOff,
    const std::int32_t* __restrict__ rCells,
    const std::int32_t* __restrict__ iOff,
    const std::int32_t* __restrict__ iFaces,
    const double* __restrict__ fDiag,
    const double* __restrict__ fLower,
    const double* __restrict__ fUpper,
    double* __restrict__ diag
)
{
    const std::int32_t C = blockIdx.x*blockDim.x + threadIdx.x;
    if (C >= nC)
    {
        return;
    }
    double d = 0.0;
    for (std::int32_t q = rOff[C]; q < rOff[C + 1]; ++q)
    {
        d += fDiag[rCells[q]];
    }
    for (std::int32_t q = iOff[C]; q < iOff[C + 1]; ++q)
    {
        const std::int32_t f = iFaces[q];
        d += fUpper[f] + fLower[f];
    }
    diag[C] = d;
}

// 粗 upper／lower = Σ 對應細面（翻轉時互換）
__global__ void kCoarseOff
(
    const std::int32_t nF,
    const std::int32_t* __restrict__ fOff,
    const std::int32_t* __restrict__ fFaces,
    const double* __restrict__ fLower,
    const double* __restrict__ fUpper,
    double* __restrict__ lower,
    double* __restrict__ upper
)
{
    const std::int32_t F = blockIdx.x*blockDim.x + threadIdx.x;
    if (F >= nF)
    {
        return;
    }
    double u = 0.0;
    double l = 0.0;
    for (std::int32_t q = fOff[F]; q < fOff[F + 1]; ++q)
    {
        const std::int32_t k = fFaces[q];
        if (k >= 0)
        {
            u += fUpper[k];
            l += fLower[k];
        }
        else
        {
            const std::int32_t f = -(k + 1);
            u += fLower[f];
            l += fUpper[f];
        }
    }
    upper[F] = u;
    lower[F] = l;
}

// LDU（原編號、FP64）→ 列次序的 CSR 係數與對角（T）
template<class T>
__global__ void kPack
(
    const std::int32_t n,
    const std::int32_t* __restrict__ perm,
    const std::int32_t* __restrict__ rowPtr,
    const std::int32_t* __restrict__ src,
    const double* __restrict__ diag,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    T* __restrict__ val,
    T* __restrict__ d,
    T* __restrict__ invd
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n)
    {
        return;
    }
    const double dd = diag[perm[i]];
    d[i] = T(dd);
    invd[i] = T(1.0/dd);
    for (std::int32_t j = rowPtr[i]; j < rowPtr[i + 1]; ++j)
    {
        const std::int32_t s = src[j];
        val[j] = T(s < 0 ? upper[-(s + 1)] : lower[s]);
    }
}

// 一列的 Gauss–Seidel：x[i] = (b[i] - Σ a_ij x[j])/d[i]；first 時列號 ≥ s 的鄰格（本色與之後的色，
// 尚未更新、視為 0）略過
template<class T, bool first>
__device__ __forceinline__ void gsRow
(
    const std::int32_t i,
    const std::int32_t s,
    const std::int32_t* __restrict__ rowPtr,
    const std::int32_t* __restrict__ col,
    const T* __restrict__ val,
    const T* __restrict__ invd,
    const T* __restrict__ b,
    T* __restrict__ x
)
{
    T acc = b[i];
    for (std::int32_t j = rowPtr[i]; j < rowPtr[i + 1]; ++j)
    {
        const std::int32_t c = col[j];
        if (!first || c < s)
        {
            acc = fmaT(-val[j], x[c], acc);
        }
    }
    x[i] = acc*invd[i];
}

template<class T, bool first>
__global__ void kGs
(
    const std::int32_t s,
    const std::int32_t e,
    const std::int32_t* __restrict__ rowPtr,
    const std::int32_t* __restrict__ col,
    const T* __restrict__ val,
    const T* __restrict__ invd,
    const T* __restrict__ b,
    T* __restrict__ x
)
{
    const std::int32_t i = s + std::int32_t(blockIdx.x*blockDim.x + threadIdx.x);
    if (i < e)
    {
        gsRow<T, first>(i, s, rowPtr, col, val, invd, b, x);
    }
}

// 粗列 I 的右端 = Σ 細列的殘差 (b - A x)
template<class T>
__device__ __forceinline__ T restrictRow
(
    const std::int32_t I,
    const std::int32_t* __restrict__ cOff,
    const std::int32_t* __restrict__ cRows,
    const std::int32_t* __restrict__ rowPtr,
    const std::int32_t* __restrict__ col,
    const T* __restrict__ val,
    const T* __restrict__ d,
    const T* __restrict__ b,
    const T* __restrict__ x
)
{
    T sum = T(0);
    for (std::int32_t q = cOff[I]; q < cOff[I + 1]; ++q)
    {
        const std::int32_t i = cRows[q];
        T r = fmaT(-d[i], x[i], b[i]);
        for (std::int32_t j = rowPtr[i]; j < rowPtr[i + 1]; ++j)
        {
            r = fmaT(-val[j], x[col[j]], r);
        }
        sum += r;
    }
    return sum;
}

template<class T>
__global__ void kResRestrict
(
    const std::int32_t nC,
    const std::int32_t* __restrict__ cOff,
    const std::int32_t* __restrict__ cRows,
    const std::int32_t* __restrict__ rowPtr,
    const std::int32_t* __restrict__ col,
    const T* __restrict__ val,
    const T* __restrict__ d,
    const T* __restrict__ b,
    const T* __restrict__ x,
    T* __restrict__ bC
)
{
    const std::int32_t I = blockIdx.x*blockDim.x + threadIdx.x;
    if (I < nC)
    {
        bC[I] = restrictRow(I, cOff, cRows, rowPtr, col, val, d, b, x);
    }
}

// 細列 += omega*粗列（注入）
template<class T>
__global__ void kProlong
(
    const std::int32_t n,
    const std::int32_t* __restrict__ parent,
    const T omega,
    const T* __restrict__ xC,
    T* __restrict__ x
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        x[i] = fmaT(omega, xC[parent[i]], x[i]);
    }
}

// 最粗層：組成密集矩陣（列次序）並做 LU（不選主元；p 方程為對角佔優的 M 矩陣），單一 thread，FP64
__global__ void kDenseLU
(
    const std::int32_t n,
    const std::int32_t nF,
    const std::int32_t* __restrict__ invPerm,
    const std::int32_t* __restrict__ lowerAddr,
    const std::int32_t* __restrict__ upperAddr,
    const double* __restrict__ diag,
    const double* __restrict__ lower,
    const double* __restrict__ upper,
    double* __restrict__ A
)
{
    if (blockIdx.x != 0 || threadIdx.x != 0)
    {
        return;
    }
    for (std::int32_t i = 0; i < n*n; ++i)
    {
        A[i] = 0.0;
    }
    for (std::int32_t c = 0; c < n; ++c)
    {
        const std::int32_t i = invPerm[c];
        A[i*n + i] = diag[c];
    }
    for (std::int32_t f = 0; f < nF; ++f)
    {
        const std::int32_t o = invPerm[lowerAddr[f]];
        const std::int32_t u = invPerm[upperAddr[f]];
        A[o*n + u] += upper[f];
        A[u*n + o] += lower[f];
    }
    for (std::int32_t k = 0; k < n; ++k)
    {
        const double piv = A[k*n + k];
        for (std::int32_t i = k + 1; i < n; ++i)
        {
            const double m = A[i*n + k]/piv;
            A[i*n + k] = m;
            for (std::int32_t j = k + 1; j < n; ++j)
            {
                A[i*n + j] -= m*A[k*n + j];
            }
        }
    }
}

template<class T>
__device__ void denseSolve(const std::int32_t n, const double* __restrict__ A, const T* __restrict__ b, T* __restrict__ x)
{
    // 前代、後代都以 FP64 計算，x 暫存前代的結果
    for (std::int32_t i = 0; i < n; ++i)
    {
        double s = double(b[i]);
        for (std::int32_t j = 0; j < i; ++j)
        {
            s -= A[i*n + j]*double(x[j]);
        }
        x[i] = T(s);
    }
    for (std::int32_t i = n - 1; i >= 0; --i)
    {
        double s = double(x[i]);
        for (std::int32_t j = i + 1; j < n; ++j)
        {
            s -= A[i*n + j]*double(x[j]);
        }
        x[i] = T(s/A[i*n + i]);
    }
}

template<class T>
__global__ void kDenseSolve(const std::int32_t n, const double* __restrict__ A, const T* __restrict__ b, T* __restrict__ x)
{
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        denseSolve(n, A, b, x);
    }
}

// Jacobi 一列：out = in + w*(b - A in)/d；first 時 in = 0：out = w*b/d（N-028）
template<class T>
__device__ __forceinline__ void jacobiRow
(
    const std::int32_t i,
    const bool first,
    const T w,
    const std::int32_t* __restrict__ rowPtr,
    const std::int32_t* __restrict__ col,
    const T* __restrict__ val,
    const T* __restrict__ d,
    const T* __restrict__ invd,
    const T* __restrict__ b,
    const T* __restrict__ in,
    T* __restrict__ out
)
{
    if (first)
    {
        out[i] = w*b[i]*invd[i];
        return;
    }
    T r = fmaT(-d[i], in[i], b[i]);
    for (std::int32_t j = rowPtr[i]; j < rowPtr[i + 1]; ++j)
    {
        r = fmaT(-val[j], in[col[j]], r);
    }
    out[i] = fmaT(w*invd[i], r, in[i]);
}

template<class T>
__global__ void kJacobi
(
    const std::int32_t n,
    const bool first,
    const T w,
    const std::int32_t* __restrict__ rowPtr,
    const std::int32_t* __restrict__ col,
    const T* __restrict__ val,
    const T* __restrict__ d,
    const T* __restrict__ invd,
    const T* __restrict__ b,
    const T* __restrict__ in,
    T* __restrict__ out
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        jacobiRow(i, first, w, rowPtr, col, val, d, invd, b, in, out);
    }
}

// 由第 l0 層起的子循環（含最粗層），一個 thread block；各階段之間 __syncthreads
template<class T>
__global__ void kBlockCycle
(
    const AmgLevelView<T>* __restrict__ views,
    const std::int32_t l0,
    const std::int32_t nL,
    const std::int32_t sweeps,
    const T omega,
    const double* __restrict__ A,
    const T wJac,
    const std::int32_t jacFrom
)
{
    const std::int32_t t = threadIdx.x;
    const std::int32_t nt = blockDim.x;
    // Jacobi 的 sweeps 次掃描，x 與 t 來回（奇數次時最後複製回 x）；first：第一次由 x = 0 開始
    const auto jacobi = [&](const AmgLevelView<T>& L, const bool first)
    {
        T* in = L.x;
        T* out = L.t;
        for (std::int32_t sw = 0; sw < sweeps; ++sw)
        {
            for (std::int32_t i = t; i < L.nRows; i += nt)
            {
                jacobiRow(i, first && sw == 0, wJac, L.rowPtr, L.col, L.val, L.d, L.invd, L.b, in, out);
            }
            __syncthreads();
            T* tmp = in;
            in = out;
            out = tmp;
        }
        if (in != L.x)
        {
            for (std::int32_t i = t; i < L.nRows; i += nt)
            {
                L.x[i] = in[i];
            }
            __syncthreads();
        }
    };
    for (std::int32_t l = l0; l + 1 < nL; ++l)
    {
        const AmgLevelView<T> L = views[l];
        const bool jac = wJac > T(0) && l >= jacFrom;
        if (jac)
        {
            jacobi(L, true);
        }
        for (std::int32_t sw = 0; sw < sweeps && !jac; ++sw)
        {
            for (std::int32_t k = 0; k < L.nColors; ++k)
            {
                const std::int32_t s = L.colorOff[k];
                const std::int32_t e = L.colorOff[k + 1];
                for (std::int32_t i = s + t; i < e; i += nt)
                {
                    if (sw == 0)
                    {
                        gsRow<T, true>(i, s, L.rowPtr, L.col, L.val, L.invd, L.b, L.x);
                    }
                    else
                    {
                        gsRow<T, false>(i, s, L.rowPtr, L.col, L.val, L.invd, L.b, L.x);
                    }
                }
                __syncthreads();
            }
        }
        const AmgLevelView<T> C = views[l + 1];
        for (std::int32_t I = t; I < C.nRows; I += nt)
        {
            C.b[I] = restrictRow(I, L.cOff, L.cRows, L.rowPtr, L.col, L.val, L.d, L.b, L.x);
        }
        __syncthreads();
    }
    if (t == 0)
    {
        const AmgLevelView<T> C = views[nL - 1];
        denseSolve(C.nRows, A, C.b, C.x);
    }
    __syncthreads();
    for (std::int32_t l = nL - 2; l >= l0; --l)
    {
        const AmgLevelView<T> L = views[l];
        const AmgLevelView<T> C = views[l + 1];
        for (std::int32_t i = t; i < L.nRows; i += nt)
        {
            L.x[i] = fmaT(omega, C.x[L.parent[i]], L.x[i]);
        }
        __syncthreads();
        const bool jac = wJac > T(0) && l >= jacFrom;
        if (jac)
        {
            jacobi(L, false);
        }
        for (std::int32_t sw = 0; sw < sweeps && !jac; ++sw)
        {
            for (std::int32_t k = L.nColors - 1; k >= 0; --k)
            {
                const std::int32_t s = L.colorOff[k];
                const std::int32_t e = L.colorOff[k + 1];
                for (std::int32_t i = s + t; i < e; i += nt)
                {
                    gsRow<T, false>(i, s, L.rowPtr, L.col, L.val, L.invd, L.b, L.x);
                }
                __syncthreads();
            }
        }
    }
}

// 第 0 層的進出：b0[列] = r[格]、z[格] = x0[列]
template<class T>
__global__ void kIn(const std::int32_t n, const std::int32_t* __restrict__ perm, const double* __restrict__ r, T* __restrict__ b)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        b[i] = T(r[perm[i]]);
    }
}

template<class T>
__global__ void kOut(const std::int32_t n, const std::int32_t* __restrict__ perm, const T* __restrict__ x, double* __restrict__ z)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        z[perm[i]] = double(x[i]);
    }
}

template<class T>
__global__ void kCopyT(const std::int32_t n, const T* __restrict__ s, T* __restrict__ d)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        d[i] = s[i];
    }
}

__global__ void kCopy(const std::int32_t n, const double* __restrict__ s, double* __restrict__ d)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        d[i] = s[i];
    }
}

constexpr int BLOCK_THREADS = 1024;

// V-cycle（各層的 x、b 為階層的固定緩衝；第 0 層的 b 在呼叫前已寫入）
template<class T>
void vcycle(AmgHierarchy& h, AmgBuffers<T>& buf)
{
    const int nL = int(h.levels.size());
    const int sweeps = h.controls.nSweeps;
    const T omega = T(h.controls.omega);
    const int lastMulti = std::min(h.blockLevel, nL - 1);
    const auto smooth = [&](const int l, const bool down)
    {
        const AmgLevel& L = h.levels[std::size_t(l)];
        if (h.controls.jacobi > 0.0 && l >= h.controls.jacobiFrom)
        {
            // Jacobi：x 與 t 來回（奇數次時最後複製回 x）；前掃第一次由 x = 0 開始
            const std::size_t sl = std::size_t(l);
            T* in = buf.x[sl].data();
            T* out = buf.t[sl].data();
            const int nb = nBlocks(std::size_t(L.nCells));
            for (int sw = 0; sw < sweeps; ++sw)
            {
                kJacobi<T><<<nb, BS>>>
                (
                    L.nCells, down && sw == 0, T(h.controls.jacobi), L.rowPtr.data(), L.col.data(),
                    buf.val[sl].data(), buf.d[sl].data(), buf.invd[sl].data(), buf.b[sl].data(), in, out
                );
                std::swap(in, out);
            }
            if (in != buf.x[sl].data())
            {
                kCopyT<T><<<nb, BS>>>(L.nCells, in, buf.x[sl].data());
            }
            return;
        }
        const int nCol = int(L.colorOff.size()) - 1;
        for (int sw = 0; sw < sweeps; ++sw)
        {
            for (int q = 0; q < nCol; ++q)
            {
                const int k = down ? q : nCol - 1 - q;
                const std::int32_t s = L.colorOff[std::size_t(k)];
                const std::int32_t e = L.colorOff[std::size_t(k) + 1];
                if (e <= s)
                {
                    continue;
                }
                const int nb = nBlocks(std::size_t(e - s));
                if (down && sw == 0)
                {
                    kGs<T, true><<<nb, BS>>>
                    (
                        s, e, L.rowPtr.data(), L.col.data(), buf.val[std::size_t(l)].data(),
                        buf.invd[std::size_t(l)].data(), buf.b[std::size_t(l)].data(), buf.x[std::size_t(l)].data()
                    );
                }
                else
                {
                    kGs<T, false><<<nb, BS>>>
                    (
                        s, e, L.rowPtr.data(), L.col.data(), buf.val[std::size_t(l)].data(),
                        buf.invd[std::size_t(l)].data(), buf.b[std::size_t(l)].data(), buf.x[std::size_t(l)].data()
                    );
                }
            }
        }
    };
    for (int l = 0; l < lastMulti; ++l)
    {
        smooth(l, true);
        const AmgLevel& Cl = h.levels[std::size_t(l) + 1];
        const AmgLevel& L = h.levels[std::size_t(l)];
        kResRestrict<T><<<nBlocks(std::size_t(Cl.nCells)), BS>>>
        (
            Cl.nCells, Cl.cOff.data(), Cl.cRows.data(), L.rowPtr.data(), L.col.data(),
            buf.val[std::size_t(l)].data(), buf.d[std::size_t(l)].data(), buf.b[std::size_t(l)].data(),
            buf.x[std::size_t(l)].data(), buf.b[std::size_t(l) + 1].data()
        );
    }
    if (h.blockLevel <= nL - 1)
    {
        const int bs = h.controls.blockSweeps > 0 ? h.controls.blockSweeps : sweeps;
        kBlockCycle<T><<<1, BLOCK_THREADS>>>
        (
            buf.views.data(), h.blockLevel, nL, bs, omega, h.dense.data(), T(h.controls.jacobi), h.controls.jacobiFrom
        );
    }
    else
    {
        kDenseSolve<T><<<1, 1>>>
        (
            h.levels.back().nCells, h.dense.data(), buf.b.back().data(), buf.x.back().data()
        );
    }
    for (int l = lastMulti - 1; l >= 0; --l)
    {
        const AmgLevel& L = h.levels[std::size_t(l)];
        const AmgLevel& Cl = h.levels[std::size_t(l) + 1];
        kProlong<T><<<nBlocks(std::size_t(L.nCells)), BS>>>
        (
            L.nCells, Cl.parent.data(), omega, buf.x[std::size_t(l) + 1].data(), buf.x[std::size_t(l)].data()
        );
        smooth(l, false);
    }
}

template<class T>
void packLevels(AmgHierarchy& h, AmgBuffers<T>& buf)
{
    for (std::size_t l = 0; l < h.levels.size(); ++l)
    {
        const AmgLevel& L = h.levels[l];
        if (L.nCells)
        {
            kPack<T><<<nBlocks(std::size_t(L.nCells)), BS>>>
            (
                L.nCells, L.perm.data(), L.rowPtr.data(), L.src.data(), h.diag[l].data(), h.lower[l].data(),
                h.upper[l].data(), buf.val[l].data(), buf.d[l].data(), buf.invd[l].data()
            );
        }
    }
}

template<class T>
void applyT(AmgHierarchy& h, AmgBuffers<T>& buf, double* z, const double* r)
{
    const AmgLevel& L0 = h.levels[0];
    const std::int32_t n = L0.nCells;
    if (n)
    {
        kIn<T><<<nBlocks(std::size_t(n)), BS>>>(n, L0.perm.data(), r, buf.b[0].data());
    }
    static const bool useGraph = []
    {
        const char* e = std::getenv("GNC_AMG_GRAPH");
        return !(e && std::string(e) == "0");
    }();
    if (useGraph)
    {
        if (!h.vcycle)
        {
            cudaGraph_t g = nullptr;
            GNC_CUDA_CHECK(cudaStreamBeginCapture(cudaStreamPerThread, cudaStreamCaptureModeThreadLocal));
            vcycle(h, buf);
            GNC_CUDA_CHECK(cudaStreamEndCapture(cudaStreamPerThread, &g));
            GNC_CUDA_CHECK(cudaGraphInstantiate(&h.vcycle, g, 0));
            GNC_CUDA_CHECK(cudaGraphDestroy(g));
        }
        GNC_CUDA_CHECK(cudaGraphLaunch(h.vcycle, cudaStreamPerThread));
    }
    else
    {
        vcycle(h, buf);
    }
    if (n)
    {
        kOut<T><<<nBlocks(std::size_t(n)), BS>>>(n, L0.perm.data(), buf.x[0].data(), z);
    }
    GNC_CUDA_LAUNCH_CHECK();
}

struct CacheKey
{
    const DeviceMesh* dm;
    int nCoarsest;
    int merge;
    int sweeps;
    double omega;
    int precision;
    int blockCells;
    int blockSweeps;
    double jacobi;
    int jacobiFrom;
    bool operator<(const CacheKey& o) const
    {
        return std::tie(dm, nCoarsest, merge, sweeps, omega, precision, blockCells, blockSweeps, jacobi, jacobiFrom)
             < std::tie
               (
                   o.dm, o.nCoarsest, o.merge, o.sweeps, o.omega, o.precision, o.blockCells, o.blockSweeps, o.jacobi,
                   o.jacobiFrom
               );
    }
};

int envInt(const char* name, const int deflt)
{
    const char* e = std::getenv(name);
    return e ? std::atoi(e) : deflt;
}

} // namespace


AmgControls amgControlsFromEnv(AmgControls c)
{
    c.mergeLevels = std::max(1, envInt("GNC_AMG_MERGE", c.mergeLevels));
    c.nSweeps = std::max(1, envInt("GNC_AMG_SWEEPS", c.nSweeps));
    c.nCellsInCoarsestLevel = std::max(1, envInt("GNC_AMG_COARSEST", c.nCellsInCoarsestLevel));
    if (const char* e = std::getenv("GNC_AMG_OMEGA"))
    {
        c.omega = std::atof(e);
    }
    c.precision = envInt("GNC_AMG_PREC", c.precision) == 64 ? 64 : 32;
    c.blockCells = std::max(0, envInt("GNC_AMG_BLOCK", c.blockCells));
    c.blockSweeps = std::max(0, envInt("GNC_AMG_CSWEEPS", c.blockSweeps));
    if (const char* e = std::getenv("GNC_AMG_JAC"))
    {
        c.jacobi = std::atof(e);
    }
    c.jacobiFrom = std::max(0, envInt("GNC_AMG_JFROM", c.jacobiFrom));
    return c;
}


AmgHierarchy::~AmgHierarchy()
{
    if (vcycle)
    {
        cudaGraphExecDestroy(vcycle);
    }
}


AmgHierarchy& amgHierarchy(const DeviceMesh& dm, const AmgControls& c)
{
    static std::map<CacheKey, std::unique_ptr<AmgHierarchy>> cache;
    const CacheKey k
    {
        &dm, c.nCellsInCoarsestLevel, c.mergeLevels, c.nSweeps, c.omega, c.precision, c.blockCells, c.blockSweeps,
        c.jacobi, c.jacobiFrom
    };
    auto it = cache.find(k);
    if (it == cache.end())
    {
        it = cache.emplace(k, buildHierarchy(dm, c)).first;
    }
    return *it->second;
}


AmgPrecond::AmgPrecond
(
    const DeviceMesh&,
    AmgHierarchy& h,
    const double* D,
    const double* lower,
    const double* upper
)
:
    h_(h)
{
    const std::size_t nL = h.levels.size();
    // 第 0 層：複製細矩陣（已加邊界 diag）到固定緩衝
    {
        const AmgLevel& L = h.levels[0];
        if (L.nCells)
        {
            kCopy<<<nBlocks(std::size_t(L.nCells)), BS>>>(L.nCells, D, h.diag[0].data());
        }
        if (L.nFaces)
        {
            kCopy<<<nBlocks(std::size_t(L.nFaces)), BS>>>(L.nFaces, lower, h.lower[0].data());
            kCopy<<<nBlocks(std::size_t(L.nFaces)), BS>>>(L.nFaces, upper, h.upper[0].data());
        }
    }
    // 粗層矩陣（原編號、FP64）
    for (std::size_t l = 1; l < nL; ++l)
    {
        const AmgLevel& L = h.levels[l];
        if (L.nCells)
        {
            kCoarseDiag<<<nBlocks(std::size_t(L.nCells)), BS>>>
            (
                L.nCells, L.rOff.data(), L.rCells.data(), L.iOff.data(), L.iFaces.data(),
                h.diag[l - 1].data(), h.lower[l - 1].data(), h.upper[l - 1].data(), h.diag[l].data()
            );
        }
        if (L.nFaces)
        {
            kCoarseOff<<<nBlocks(std::size_t(L.nFaces)), BS>>>
            (
                L.nFaces, L.fOff.data(), L.fFaces.data(), h.lower[l - 1].data(), h.upper[l - 1].data(),
                h.lower[l].data(), h.upper[l].data()
            );
        }
    }
    // 套用用的格式與最粗層的 LU
    if (h.controls.precision == 64)
    {
        packLevels(h, h.f64);
    }
    else
    {
        packLevels(h, h.f32);
    }
    const AmgLevel& C = h.levels[nL - 1];
    kDenseLU<<<1, 1>>>
    (
        C.nCells, C.nFaces, C.invPerm.data(), C.lowerAddr.data(), C.upperAddr.data(),
        h.diag[nL - 1].data(), h.lower[nL - 1].data(), h.upper[nL - 1].data(), h.dense.data()
    );
    GNC_CUDA_LAUNCH_CHECK();
}


void AmgPrecond::apply(double* z, const double* r) const
{
    if (h_.controls.precision == 64)
    {
        applyT(h_, h_.f64, z, r);
    }
    else
    {
        applyT(h_, h_.f32, z, r);
    }
}

} // namespace gnc
