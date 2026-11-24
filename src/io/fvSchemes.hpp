// fvSchemes 的查詢（已以 foamDictionary -expand -precision 17 展開）。
// GPU-Native-CFD，GPL-3.0-or-later。
//
// 查詢規則同 OpenFOAM schemesLookup：先找確切的鍵，找不到用 default。
// 回傳的方案為 token 以空白串接（例如 "faceLeastSquares linear"）。

#ifndef GNC_IO_FVSCHEMES_HPP
#define GNC_IO_FVSCHEMES_HPP

#include "io/dictionary.hpp"

#include <string>

namespace gnc
{

class FvSchemes
{
    Dictionary dict_;
    std::string path_;

    std::string lookup(const std::string& sub, const std::string& name) const;

public:
    explicit FvSchemes(const std::string& path);

    const Dictionary& dict() const { return dict_; }

    std::string grad(const std::string& name) const { return lookup("gradSchemes", name); }
    std::string interpolation(const std::string& name) const { return lookup("interpolationSchemes", name); }
    std::string snGrad(const std::string& name) const { return lookup("snGradSchemes", name); }
    std::string laplacian(const std::string& name) const { return lookup("laplacianSchemes", name); }
    std::string div(const std::string& name) const { return lookup("divSchemes", name); }
    std::string ddt(const std::string& name) const { return lookup("ddtSchemes", name); }

    // 頂層的單字項目（例如 fluxScheme、lowMachAusm）；不存在時回傳 deflt
    std::string top(const std::string& key, const std::string& deflt) const;
};

} // namespace gnc

#endif
