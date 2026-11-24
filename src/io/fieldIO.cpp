// GPU-Native-CFD，GPL-3.0-or-later。見 fieldIO.hpp。

#include "io/fieldIO.hpp"

#include "io/foamHeader.hpp"
#include "io/foamTokenizer.hpp"

#include <cstdio>
#include <fstream>
#include <regex>
#include <stdexcept>

namespace gnc
{

namespace
{

int nCompOfList(FoamTokenizer& t, const std::string& listType)
{
    if (listType == "List<scalar>") return 1;
    if (listType == "List<vector>") return 3;
    if (listType == "List<tensor>") return 9;
    t.error("unsupported list type " + listType);
}


void readTuple(FoamTokenizer& t, const int nComp, double* out)
{
    if (nComp == 1)
    {
        out[0] = t.readScalar();
        return;
    }
    t.expectPunct('(');
    for (int k = 0; k < nComp; ++k)
    {
        out[k] = t.readScalar();
    }
    t.expectPunct(')');
}


// 讀 nonuniform 之後的串列：List<T> N ( ... ) 或 N{X}。回傳展開後的值。
std::vector<double> readList
(
    FoamTokenizer& t,
    const bool binary,
    const int nCompExpected     // < 0 表示不檢查（略過用）
)
{
    const int nComp = nCompOfList(t, t.expectWord());
    if (nCompExpected >= 0 && nComp != nCompExpected)
    {
        t.error("list component count differs from the field type");
    }
    const std::int64_t n = t.readLabel();
    if (n < 0)
    {
        t.error("negative list size");
    }
    std::vector<double> out(std::size_t(n)*std::size_t(nComp));

    // 二進位的空串列只寫大小，沒有括號（例如 0 面 patch 的 "value
    // nonuniform List<scalar> 0;"；List 的二進位讀取在長度 0 時不讀內容）
    if (n == 0 && t.peekChar() != '(' && t.peekChar() != '{')
    {
        return out;
    }

    if (t.tryPunct('('))
    {
        if (binary)
        {
            t.readRaw(out.data(), out.size()*sizeof(double));
        }
        else
        {
            for (std::size_t i = 0; i < std::size_t(n); ++i)
            {
                readTuple(t, nComp, out.data() + i*nComp);
            }
        }
        t.expectPunct(')');
    }
    else if (t.tryPunct('{'))
    {
        double v[9] = {};
        if (binary)
        {
            t.readRaw(v, std::size_t(nComp)*sizeof(double));
        }
        else
        {
            readTuple(t, nComp, v);
        }
        t.expectPunct('}');
        for (std::size_t i = 0; i < std::size_t(n); ++i)
        {
            for (int k = 0; k < nComp; ++k)
            {
                out[i*nComp + k] = v[k];
            }
        }
    }
    else
    {
        t.error("expected '(' or '{' after list size");
    }
    return out;
}


void rejectDirective(FoamTokenizer& t, const Token& tok)
{
    if
    (
        tok.kind == Token::WORD && !tok.text.empty()
     && (tok.text[0] == '#' || tok.text[0] == '$')
    )
    {
        t.error
        (
            "unsupported directive '" + tok.text
          + "': expand the field with foamDictionary -expand first"
        );
    }
}


// 略過一個項目直到深度 0 的 ';'。nonuniform 串列（可能是二進位）照格式略過。
void skipEntry(FoamTokenizer& t, const bool binary)
{
    int depth = 0;
    for (;;)
    {
        const Token v = t.next();
        if (v.kind == Token::END)
        {
            t.error("unexpected end of file");
        }
        rejectDirective(t, v);
        if (v.kind == Token::WORD && v.text == "nonuniform")
        {
            (void)readList(t, binary, -1);
            continue;
        }
        if (v.kind == Token::PUNCT)
        {
            const char c = v.text[0];
            if (c == ';' && depth == 0)
            {
                return;
            }
            if (c == '(' || c == '[' || c == '{')
            {
                ++depth;
            }
            else if (c == ')' || c == ']' || c == '}')
            {
                --depth;
            }
        }
    }
}


std::string readWordEntry(FoamTokenizer& t)
{
    const Token w = t.next();
    if (w.kind != Token::WORD && w.kind != Token::STRING)
    {
        t.error("expected a word");
    }
    rejectDirective(t, w);
    t.expectPunct(';');
    return w.text;
}


std::string formatScalar(const double v)
{
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.17g", v);
    return buf;
}


void writeTuple(std::ostream& os, const double* v, const int nComp)
{
    if (nComp == 1)
    {
        os << formatScalar(v[0]);
        return;
    }
    os << '(';
    for (int k = 0; k < nComp; ++k)
    {
        if (k)
        {
            os << ' ';
        }
        os << formatScalar(v[k]);
    }
    os << ')';
}


const char* listTypeName(const int nComp)
{
    switch (nComp)
    {
        case 1: return "List<scalar>";
        case 3: return "List<vector>";
        default: return "List<tensor>";
    }
}


void writeList
(
    std::ostream& os,
    const std::vector<double>& v,
    const int nComp,
    const bool multiLine,
    const bool binary = false
)
{
    const std::size_t n = v.size()/nComp;
    os << "nonuniform " << listTypeName(nComp) << ' ';
    if (binary)
    {
        // UList::writeList（binary、contiguous）
        os << '\n' << n << '\n';
        if (n)
        {
            os << '(';
            os.write(reinterpret_cast<const char*>(v.data()), std::streamsize(v.size()*sizeof(double)));
            os << ')';
        }
        if (multiLine)
        {
            os << '\n';
        }
        return;
    }
    if (multiLine)
    {
        os << '\n' << n << "\n(\n";
        for (std::size_t i = 0; i < n; ++i)
        {
            writeTuple(os, v.data() + i*nComp, nComp);
            os << '\n';
        }
        os << ")\n";
    }
    else
    {
        os << n << '(';
        for (std::size_t i = 0; i < n; ++i)
        {
            if (i)
            {
                os << ' ';
            }
            writeTuple(os, v.data() + i*nComp, nComp);
        }
        os << ')';
    }
}

// 值延後展開：uniform 只存一個 tuple，nonuniform 存整個串列
struct RawValue
{
    bool present = false;
    bool uniform = true;
    std::vector<double> data;
};


RawValue readRawValue(FoamTokenizer& t, const bool binary, const int nComp)
{
    RawValue r;
    r.present = true;
    const std::string kind = t.expectWord();
    if (kind == "uniform")
    {
        r.uniform = true;
        r.data.resize(std::size_t(nComp));
        readTuple(t, nComp, r.data.data());
    }
    else if (kind == "nonuniform")
    {
        r.uniform = false;
        r.data = readList(t, binary, nComp);
    }
    else
    {
        t.error("expected uniform or nonuniform, got '" + kind + "'");
    }
    return r;
}


std::vector<double> expand
(
    const RawValue& r,
    const int nComp,
    const std::size_t n,
    const std::string& what
)
{
    if (r.uniform)
    {
        std::vector<double> out(n*nComp);
        for (std::size_t i = 0; i < n; ++i)
        {
            for (int k = 0; k < nComp; ++k)
            {
                out[i*nComp + k] = r.data[std::size_t(k)];
            }
        }
        return out;
    }
    if (r.data.size() != n*nComp)
    {
        throw std::runtime_error
        (
            what + ": list size " + std::to_string(r.data.size()/nComp)
          + " differs from the expected " + std::to_string(n)
        );
    }
    return r.data;
}


// boundaryField 中的一個項目（鍵可能是 patch 名稱、group 或 regex）
struct RawPatchEntry
{
    std::string key;
    bool pattern = false;
    std::string type;
    std::string phiName = "phi";
    RawValue value, refValue, refGrad, valueFraction, gradient;
    std::vector<std::pair<std::string, std::string>> extras;
};


RawPatchEntry readPatchEntry
(
    FoamTokenizer& t,
    const bool binary,
    const int nComp,
    const Token& key
)
{
    RawPatchEntry e;
    e.key = key.text;
    e.pattern = (key.kind == Token::STRING);
    t.expectPunct('{');
    for (;;)
    {
        const Token k = t.next();
        if (k.isPunct('}'))
        {
            break;
        }
        if (k.kind != Token::WORD)
        {
            t.error("expected a keyword in patch entry " + e.key);
        }
        rejectDirective(t, k);
        if (k.text == "type")
        {
            e.type = readWordEntry(t);
        }
        else if (k.text == "phi")
        {
            e.phiName = readWordEntry(t);
        }
        else if (k.text == "value")
        {
            e.value = readRawValue(t, binary, nComp);
            t.expectPunct(';');
        }
        else if (k.text == "inletValue" || k.text == "refValue" || k.text == "freestreamValue")
        {
            e.refValue = readRawValue(t, binary, nComp);
            t.expectPunct(';');
        }
        else if (k.text == "refGradient")
        {
            e.refGrad = readRawValue(t, binary, nComp);
            t.expectPunct(';');
        }
        else if (k.text == "valueFraction")
        {
            e.valueFraction = readRawValue(t, binary, 1);
            t.expectPunct(';');
        }
        else if (k.text == "gradient")
        {
            e.gradient = readRawValue(t, binary, nComp);
            t.expectPunct(';');
        }
        else
        {
            // 其他鍵：保存原始文字；含 nonuniform 串列（可能是二進位）的略過
            std::string text;
            bool keep = true;
            int depth = 0;
            for (;;)
            {
                const Token v = t.next();
                if (v.kind == Token::END)
                {
                    t.error("unexpected end of file in entry " + k.text);
                }
                rejectDirective(t, v);
                if (v.kind == Token::WORD && v.text == "nonuniform")
                {
                    (void)readList(t, binary, -1);
                    keep = false;
                    continue;
                }
                if (v.kind == Token::PUNCT)
                {
                    const char c = v.text[0];
                    if (c == ';' && depth == 0)
                    {
                        break;
                    }
                    if (c == '(' || c == '[' || c == '{')
                    {
                        ++depth;
                    }
                    else if (c == ')' || c == ']' || c == '}')
                    {
                        --depth;
                    }
                }
                if (!text.empty())
                {
                    text += ' ';
                }
                text += v.kind == Token::STRING ? '"' + v.text + '"' : v.text;
            }
            if (keep)
            {
                e.extras.emplace_back(k.text, text);
            }
        }
    }
    return e;
}


PatchFieldSpec makeSpec
(
    const RawPatchEntry& e,
    const int nComp,
    const std::size_t n,
    const std::string& what
)
{
    PatchFieldSpec s;
    s.type = e.type;
    s.phiName = e.phiName;
    s.extras = e.extras;
    if (e.value.present)
    {
        s.hasValue = true;
        s.value = expand(e.value, nComp, n, what + "/value");
    }
    if (e.refValue.present)
    {
        s.hasRefValue = true;
        s.refValue = expand(e.refValue, nComp, n, what + "/refValue");
    }
    if (e.refGrad.present)
    {
        s.hasRefGrad = true;
        s.refGrad = expand(e.refGrad, nComp, n, what + "/refGradient");
    }
    if (e.valueFraction.present)
    {
        s.hasValueFraction = true;
        s.valueFraction = expand(e.valueFraction, 1, n, what + "/valueFraction");
    }
    if (e.gradient.present)
    {
        s.hasGradient = true;
        s.gradient = expand(e.gradient, nComp, n, what + "/gradient");
    }
    return s;
}

} // namespace


HostField readField(const std::string& path, const PolyMesh& mesh)
{
    FoamTokenizer t(path);
    const FoamHeader h = readFoamHeader(t);

    HostField f;
    f.name = h.object;
    f.cls = h.cls;
    f.format = h.binary ? "binary" : "ascii";
    if (h.cls == "volScalarField")
    {
        f.nComp = 1;
    }
    else if (h.cls == "volVectorField")
    {
        f.nComp = 3;
    }
    else if (h.cls == "surfaceScalarField")
    {
        f.nComp = 1;
        f.isSurface = true;
    }
    else
    {
        t.error("unsupported field class " + h.cls);
    }
    const std::size_t nInternal =
        std::size_t(f.isSurface ? mesh.nInternalFaces : mesh.nCells);

    std::vector<RawPatchEntry> entries;
    bool gotInternal = false;
    bool gotBoundary = false;

    for (;;)
    {
        const Token key = t.next();
        if (key.kind == Token::END)
        {
            break;
        }
        if (key.kind != Token::WORD)
        {
            t.error("expected a keyword, got '" + key.text + "'");
        }
        rejectDirective(t, key);

        if (key.text == "dimensions")
        {
            std::string dims;
            for (;;)
            {
                const Token d = t.next();
                if (d.kind == Token::END)
                {
                    t.error("unexpected end of file in dimensions");
                }
                if (d.isPunct(';'))
                {
                    break;
                }
                if (!dims.empty() && !d.isPunct(']') && dims.back() != '[')
                {
                    dims += ' ';
                }
                dims += d.text;
            }
            f.dimensions = dims;
        }
        else if (key.text == "internalField")
        {
            f.internal = expand
            (
                readRawValue(t, h.binary, f.nComp), f.nComp, nInternal,
                path + ": internalField"
            );
            t.expectPunct(';');
            gotInternal = true;
        }
        else if (key.text == "boundaryField")
        {
            t.expectPunct('{');
            for (;;)
            {
                const Token pk = t.next();
                if (pk.isPunct('}'))
                {
                    break;
                }
                if (pk.kind != Token::WORD && pk.kind != Token::STRING)
                {
                    t.error("expected a patch name");
                }
                rejectDirective(t, pk);
                RawPatchEntry e = readPatchEntry(t, h.binary, f.nComp, pk);
                // 字典中重複的字面鍵：後者取代前者
                if (!e.pattern)
                {
                    for (auto it = entries.begin(); it != entries.end(); ++it)
                    {
                        if (!it->pattern && it->key == e.key)
                        {
                            entries.erase(it);
                            break;
                        }
                    }
                }
                entries.push_back(std::move(e));
            }
            gotBoundary = true;
        }
        else
        {
            skipEntry(t, h.binary);
        }
    }

    if (!gotInternal || !gotBoundary)
    {
        t.error("missing internalField or boundaryField");
    }

    // 依 GeometricBoundaryField::readField 的規則對應到 patch
    const std::size_t nP = mesh.patches.size();
    std::vector<const RawPatchEntry*> chosen(nP, nullptr);
    std::vector<bool> setEmpty(nP, false);

    // 1. 確切的 patch 名稱
    for (std::size_t pi = 0; pi < nP; ++pi)
    {
        for (const RawPatchEntry& e : entries)
        {
            if (!e.pattern && e.key == mesh.patches[pi].name)
            {
                chosen[pi] = &e;
            }
        }
    }
    // 2. patch group，項目依反向次序（後面的優先）
    for (auto it = entries.rbegin(); it != entries.rend(); ++it)
    {
        if (it->pattern)
        {
            continue;
        }
        for (std::size_t pi = 0; pi < nP; ++pi)
        {
            if (chosen[pi])
            {
                continue;
            }
            for (const std::string& g : mesh.patches[pi].inGroups)
            {
                if (g == it->key)
                {
                    chosen[pi] = &*it;
                }
            }
        }
    }
    // 3. empty 自動設為 empty；其餘以 regex（反向次序，整串相符）
    for (std::size_t pi = 0; pi < nP; ++pi)
    {
        if (chosen[pi])
        {
            continue;
        }
        if (mesh.patches[pi].isEmpty())
        {
            setEmpty[pi] = true;
            continue;
        }
        for (auto it = entries.rbegin(); it != entries.rend(); ++it)
        {
            if
            (
                it->pattern
             && std::regex_match(mesh.patches[pi].name, std::regex(it->key))
            )
            {
                chosen[pi] = &*it;
                break;
            }
        }
        if (!chosen[pi])
        {
            throw std::runtime_error
            (
                path + ": no boundary condition for patch " + mesh.patches[pi].name
            );
        }
    }

    f.patches.resize(nP);
    for (std::size_t pi = 0; pi < nP; ++pi)
    {
        const Patch& mp = mesh.patches[pi];
        const std::size_t n = mp.isEmpty() ? 0 : std::size_t(mp.size);
        if (setEmpty[pi])
        {
            f.patches[pi].type = "empty";
            continue;
        }
        if (chosen[pi]->type.empty())
        {
            throw std::runtime_error(path + ": patch " + mp.name + " has no type");
        }
        f.patches[pi] = makeSpec(*chosen[pi], f.nComp, n, path + ": " + mp.name);
    }
    return f;
}


void writeField
(
    const std::string& path,
    const HostField& f,
    const PolyMesh& mesh,
    const std::string& location,
    const bool binary
)
{
    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    if (!os)
    {
        throw std::runtime_error("cannot write " + path);
    }

    os  << "FoamFile\n{\n"
        << "    version     2.0;\n"
        << "    format      " << (binary ? "binary" : "ascii") << ";\n"
        << "    arch        \"LSB;label=32;scalar=64\";\n"
        << "    class       " << f.cls << ";\n"
        << "    location    \"" << location << "\";\n"
        << "    object      " << f.name << ";\n"
        << "}\n"
        << (binary ? "// written by GPU-Native-CFD (binary)\n\n" : "// written by GPU-Native-CFD (%.17g)\n\n")
        << "dimensions      " << f.dimensions << ";\n\n"
        << "internalField   ";
    writeList(os, f.internal, f.nComp, true, binary);
    os << ";\n\nboundaryField\n{\n";

    for (std::size_t pi = 0; pi < mesh.patches.size(); ++pi)
    {
        const PatchFieldSpec& p = f.patches[pi];
        os << "    " << mesh.patches[pi].name << "\n    {\n"
           << "        type            " << p.type << ";\n";
        if (p.phiName != "phi")
        {
            os << "        phi             " << p.phiName << ";\n";
        }
        for (const auto& e : p.extras)
        {
            os << "        " << e.first << ' ' << e.second << ";\n";
        }
        if (p.hasGradient)
        {
            os << "        gradient        ";
            writeList(os, p.gradient, f.nComp, false, binary);
            os << ";\n";
        }
        if (p.hasRefValue)
        {
            const bool freestream =
                p.type == "freestreamVelocity" || p.type == "freestreamPressure";
            os << "        "
               << (p.type == "inletOutlet" ? "inletValue" : freestream ? "freestreamValue" : "refValue")
               << (freestream ? " " : "      ");
            writeList(os, p.refValue, f.nComp, false, binary);
            os << ";\n";
        }
        if (p.hasRefGrad)
        {
            os << "        refGradient     ";
            writeList(os, p.refGrad, f.nComp, false, binary);
            os << ";\n";
        }
        if (p.hasValueFraction)
        {
            os << "        valueFraction   ";
            writeList(os, p.valueFraction, 1, false, binary);
            os << ";\n";
        }
        if (p.hasValue)
        {
            os << "        value           ";
            writeList(os, p.value, f.nComp, false, binary);
            os << ";\n";
        }
        os << "    }\n";
    }
    os << "}\n";

    if (!os)
    {
        throw std::runtime_error("write failed: " + path);
    }
}

} // namespace gnc
