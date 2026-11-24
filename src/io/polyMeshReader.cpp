// GPU-Native-CFD，GPL-3.0-or-later。見 polyMeshReader.hpp。

#include "io/polyMeshReader.hpp"

#include "io/dictionary.hpp"
#include "io/foamHeader.hpp"
#include "io/foamTokenizer.hpp"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace gnc
{

namespace
{

namespace fs = std::filesystem;

constexpr std::int64_t labelMax = std::numeric_limits<std::int32_t>::max();

using Header = FoamHeader;


std::string meshFile(const std::string& dir, const std::string& name)
{
    const std::string p = dir + "/" + name;
    if (!fs::is_regular_file(p))
    {
        if (fs::exists(p + ".gz"))
        {
            throw std::runtime_error
            (
                p + ".gz: compressed mesh files are not supported"
                " (write the mesh with writeCompression off)"
            );
        }
        throw std::runtime_error(p + ": missing");
    }
    return p;
}


Header readHeader(FoamTokenizer& t)
{
    return readFoamHeader(t);
}


void expectClass(FoamTokenizer& t, const Header& h, const std::string& cls)
{
    if (h.cls != cls)
    {
        t.error("expected class " + cls + ", got " + h.cls);
    }
}


void expectEnd(FoamTokenizer& t)
{
    if (!t.atEnd())
    {
        t.error("unexpected data after the list");
    }
}


std::int32_t checkedLabel(FoamTokenizer& t, const std::int64_t v)
{
    if (v < 0 || v > labelMax)
    {
        t.error("label out of range: " + std::to_string(v));
    }
    return std::int32_t(v);
}


std::vector<std::int32_t> readLabelList(FoamTokenizer& t, const bool binary)
{
    const std::int32_t n = checkedLabel(t, t.readLabel());
    std::vector<std::int32_t> out(std::size_t(n), 0);

    // 二進位的空串列只寫大小，沒有括號
    if (n == 0 && t.peekChar() != '(' && t.peekChar() != '{')
    {
        return out;
    }

    if (t.tryPunct('('))
    {
        if (binary)
        {
            t.readRaw(out.data(), out.size()*sizeof(std::int32_t));
        }
        else
        {
            for (auto& v : out)
            {
                v = checkedLabel(t, t.readLabel());
            }
        }
        t.expectPunct(')');
    }
    else if (t.tryPunct('{'))
    {
        // 均一串列 N{v}
        std::int32_t v = 0;
        if (binary)
        {
            t.readRaw(&v, sizeof(v));
        }
        else
        {
            v = checkedLabel(t, t.readLabel());
        }
        t.expectPunct('}');
        std::fill(out.begin(), out.end(), v);
    }
    else
    {
        t.error("expected '(' or '{' after list size");
    }
    return out;
}


std::vector<double> readVectorList(FoamTokenizer& t, const bool binary)
{
    const std::int32_t n = checkedLabel(t, t.readLabel());
    std::vector<double> out(3*std::size_t(n), 0.0);

    t.expectPunct('(');
    if (binary)
    {
        t.readRaw(out.data(), out.size()*sizeof(double));
    }
    else
    {
        for (std::size_t i = 0; i < std::size_t(n); ++i)
        {
            t.expectPunct('(');
            out[3*i] = t.readScalar();
            out[3*i + 1] = t.readScalar();
            out[3*i + 2] = t.readScalar();
            t.expectPunct(')');
        }
    }
    t.expectPunct(')');
    return out;
}


void readFaces
(
    FoamTokenizer& t,
    const Header& h,
    std::vector<std::int32_t>& offsets,
    std::vector<std::int32_t>& labels
)
{
    if (h.cls == "faceCompactList")
    {
        offsets = readLabelList(t, h.binary);
        labels = readLabelList(t, h.binary);
        if (offsets.empty() || offsets.front() != 0)
        {
            t.error("faceCompactList: offsets must start with 0");
        }
        for (std::size_t i = 1; i < offsets.size(); ++i)
        {
            if (offsets[i] < offsets[i - 1])
            {
                t.error("faceCompactList: offsets not monotonic");
            }
        }
        if (std::size_t(offsets.back()) != labels.size())
        {
            t.error("faceCompactList: offsets do not match the label count");
        }
    }
    else if (h.cls == "faceList" && !h.binary)
    {
        const std::int32_t n = checkedLabel(t, t.readLabel());
        offsets.assign(std::size_t(n) + 1, 0);
        labels.clear();
        labels.reserve(4*std::size_t(n));
        t.expectPunct('(');
        for (std::int32_t f = 0; f < n; ++f)
        {
            const std::int32_t np = checkedLabel(t, t.readLabel());
            t.expectPunct('(');
            for (std::int32_t k = 0; k < np; ++k)
            {
                labels.push_back(checkedLabel(t, t.readLabel()));
            }
            t.expectPunct(')');
            if (labels.size() > std::size_t(labelMax))
            {
                t.error("too many face labels for label=32");
            }
            offsets[std::size_t(f) + 1] = std::int32_t(labels.size());
        }
        t.expectPunct(')');
    }
    else
    {
        t.error
        (
            "faces: unsupported class " + h.cls
          + (h.binary ? " (binary)" : " (ascii)")
        );
    }
}


std::vector<Patch> readBoundary(FoamTokenizer& t)
{
    const std::int32_t n = checkedLabel(t, t.readLabel());
    std::vector<Patch> patches;
    t.expectPunct('(');
    for (std::int32_t i = 0; i < n; ++i)
    {
        Patch p;
        p.name = t.expectWord();
        t.expectPunct('{');
        const Dictionary d = Dictionary::parse(t, "boundary/" + p.name, true);
        p.type = d.getWord("type");
        p.size = checkedLabel(t, d.getLabel("nFaces"));
        p.start = checkedLabel(t, d.getLabel("startFace"));
        // inGroups List<word> N(a b ...)：取括號內的單字
        if (const Dictionary::Entry* g = d.find("inGroups"))
        {
            int depth = 0;
            for (const Token& tok : g->stream)
            {
                if (tok.isPunct('('))
                {
                    ++depth;
                }
                else if (tok.isPunct(')'))
                {
                    --depth;
                }
                else if (depth == 1 && (tok.kind == Token::WORD || tok.kind == Token::STRING))
                {
                    p.inGroups.push_back(tok.text);
                }
            }
        }
        patches.push_back(std::move(p));
    }
    t.expectPunct(')');
    return patches;
}


// 本階段支援的 patch 型別（design.md §1.1 的案例實際用到的）。耦合型
// （processor、cyclic、AMI）與其他型別在支援前一律報錯。
bool supportedPatchType(const std::string& type)
{
    return type == "patch" || type == "wall" || type == "symmetry"
        || type == "symmetryPlane" || type == "empty";
}

} // namespace


PolyMesh readPolyMesh(const std::string& dir)
{
    PolyMesh m;
    std::vector<std::string> formats;

    {
        FoamTokenizer t(meshFile(dir, "points"));
        const Header h = readHeader(t);
        expectClass(t, h, "vectorField");
        m.points = readVectorList(t, h.binary);
        expectEnd(t);
        formats.push_back(h.binary ? "binary" : "ascii");
    }
    {
        FoamTokenizer t(meshFile(dir, "faces"));
        const Header h = readHeader(t);
        readFaces(t, h, m.faceOffsets, m.faceLabels);
        expectEnd(t);
        formats.push_back(h.binary ? "binary" : "ascii");
    }
    {
        FoamTokenizer t(meshFile(dir, "owner"));
        const Header h = readHeader(t);
        expectClass(t, h, "labelList");
        m.owner = readLabelList(t, h.binary);
        expectEnd(t);
        formats.push_back(h.binary ? "binary" : "ascii");
    }
    {
        FoamTokenizer t(meshFile(dir, "neighbour"));
        const Header h = readHeader(t);
        expectClass(t, h, "labelList");
        m.neighbour = readLabelList(t, h.binary);
        expectEnd(t);
        formats.push_back(h.binary ? "binary" : "ascii");
    }
    {
        FoamTokenizer t(meshFile(dir, "boundary"));
        const Header h = readHeader(t);
        expectClass(t, h, "polyBoundaryMesh");
        m.patches = readBoundary(t);
        expectEnd(t);
    }

    m.format = formats.front();
    for (const std::string& f : formats)
    {
        if (f != m.format)
        {
            m.format = "mixed";
        }
    }

    // 一致性檢查
    const auto fail = [&](const std::string& msg)
    {
        throw std::runtime_error(dir + ": " + msg);
    };

    m.nPoints = std::int32_t(m.points.size()/3);
    m.nFaces = std::int32_t(m.faceOffsets.size()) - 1;
    m.nInternalFaces = std::int32_t(m.neighbour.size());

    if (std::size_t(m.nFaces) != m.owner.size())
    {
        fail("owner size differs from the face count");
    }
    if (m.nInternalFaces > m.nFaces)
    {
        fail("more neighbours than faces");
    }
    for (std::int32_t f = 0; f < m.nFaces; ++f)
    {
        if (m.faceOffsets[f + 1] - m.faceOffsets[f] < 3)
        {
            fail("face " + std::to_string(f) + " has fewer than 3 points");
        }
    }
    for (const std::int32_t p : m.faceLabels)
    {
        if (p >= m.nPoints)
        {
            fail("face point label out of range");
        }
    }

    std::int32_t maxCell = -1;
    for (const std::int32_t c : m.owner)
    {
        maxCell = std::max(maxCell, c);
    }
    for (const std::int32_t c : m.neighbour)
    {
        maxCell = std::max(maxCell, c);
    }
    m.nCells = maxCell + 1;

    std::int32_t next = m.nInternalFaces;
    for (const Patch& p : m.patches)
    {
        if (!supportedPatchType(p.type))
        {
            fail
            (
                "patch " + p.name + ": type " + p.type
              + " is not supported (patch, wall, symmetry, symmetryPlane,"
                " empty)"
            );
        }
        if (p.start != next || p.size < 0)
        {
            fail("patch " + p.name + ": faces are not contiguous");
        }
        next += p.size;
    }
    if (next != m.nFaces)
    {
        fail("patches do not cover all boundary faces");
    }

    return m;
}

} // namespace gnc
