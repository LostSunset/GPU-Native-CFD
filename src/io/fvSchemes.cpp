// GPU-Native-CFD，GPL-3.0-or-later。見 fvSchemes.hpp。

#include "io/fvSchemes.hpp"

#include "io/foamTokenizer.hpp"

#include <stdexcept>

namespace gnc
{

namespace
{

Dictionary parseFile(const std::string& path)
{
    FoamTokenizer t(path);
    return Dictionary::parse(t, path, false);
}

std::string joined(const Dictionary::Entry& e)
{
    std::string s;
    for (const Token& t : e.stream)
    {
        s += (s.empty() ? "" : " ") + t.text;
    }
    return s;
}

} // namespace


FvSchemes::FvSchemes(const std::string& path)
:
    dict_(parseFile(path)),
    path_(path)
{}


std::string FvSchemes::lookup(const std::string& sub, const std::string& name) const
{
    const Dictionary& d = dict_.subDict(sub);
    const Dictionary::Entry* e = d.find(name);
    if (!e)
    {
        e = d.find("default");
    }
    if (!e || e->dict)
    {
        throw std::runtime_error(path_ + ": " + sub + " has no entry for " + name);
    }
    return joined(*e);
}


std::string FvSchemes::top(const std::string& key, const std::string& deflt) const
{
    const Dictionary::Entry* e = dict_.find(key);
    if (!e || e->dict)
    {
        return deflt;
    }
    return joined(*e);
}

} // namespace gnc
