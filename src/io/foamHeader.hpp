// OpenFOAM 檔頭（FoamFile 字典）。GPU-Native-CFD，GPL-3.0-or-later。

#ifndef GNC_IO_FOAMHEADER_HPP
#define GNC_IO_FOAMHEADER_HPP

#include "io/foamTokenizer.hpp"

#include <string>

namespace gnc
{

struct FoamHeader
{
    std::string cls;
    std::string object;
    bool binary = false;
};


// 讀 FoamFile { ... }。二進位格式只接受 arch "LSB;label=32;scalar=64"，
// 因為原始位元組的配置取決於它。
FoamHeader readFoamHeader(FoamTokenizer& t);

} // namespace gnc

#endif
