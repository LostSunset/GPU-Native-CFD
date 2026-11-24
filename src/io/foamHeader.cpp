// GPU-Native-CFD，GPL-3.0-or-later。見 foamHeader.hpp。

#include "io/foamHeader.hpp"

#include "io/dictionary.hpp"

namespace gnc
{

FoamHeader readFoamHeader(FoamTokenizer& t)
{
    if (t.expectWord() != "FoamFile")
    {
        t.error("expected FoamFile header");
    }
    t.expectPunct('{');
    const Dictionary h = Dictionary::parse(t, t.path() + ":FoamFile", true);

    FoamHeader hd;
    hd.cls = h.getWord("class");
    hd.object = h.getWordOrDefault("object", "");
    const std::string format = h.getWord("format");
    if (format == "binary")
    {
        hd.binary = true;
        const std::string arch = h.getWordOrDefault("arch", "");
        if
        (
            arch.find("LSB") == std::string::npos
         || arch.find("label=32") == std::string::npos
         || arch.find("scalar=64") == std::string::npos
        )
        {
            t.error
            (
                "binary format requires arch \"LSB;label=32;scalar=64\", got \""
              + arch + "\""
            );
        }
    }
    else if (format != "ascii")
    {
        t.error("unknown format " + format);
    }
    return hd;
}

} // namespace gnc
