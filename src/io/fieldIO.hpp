// OpenFOAM 場檔的讀寫（主機端）。GPU-Native-CFD，GPL-3.0-or-later。
//
// 支援 volScalarField、volVectorField、surfaceScalarField，ASCII 與二進位
// （二進位只接受 arch "LSB;label=32;scalar=64"）。值的寫法：
//   uniform X                         X 為純量或 (x y z)
//   nonuniform List<T> N ( ... )      ASCII 或二進位原始位元組
//   nonuniform List<T> N{X}           OpenFOAM 對全部相同的串列的簡寫
// 邊界條件只保存本專案支援的項目（type、value、inletValue、freestreamValue、
// refValue、refGradient、valueFraction、phi）；其他鍵保存原始文字。指令（#include、
// $var）報錯，請先以 foamDictionary -expand 展開（N-004）。
//
// 寫出時一律 ASCII，數值以 %.17g 寫出（17 位有效數字可唯一決定一個
// double），格式照 OpenFOAM 的場檔，OpenFOAM 工具可以直接讀。

#ifndef GNC_IO_FIELDIO_HPP
#define GNC_IO_FIELDIO_HPP

#include "io/polyMeshReader.hpp"

#include <string>
#include <vector>

namespace gnc
{

struct PatchFieldSpec
{
    std::string type;
    std::string phiName = "phi";
    bool hasValue = false;
    std::vector<double> value;          // 交錯配置，patch 大小 × 分量數
    bool hasRefValue = false;
    std::vector<double> refValue;       // inletOutlet 的 inletValue、freestream* 的 freestreamValue 也存這裡
    bool hasRefGrad = false;
    std::vector<double> refGrad;
    bool hasValueFraction = false;
    std::vector<double> valueFraction;  // 每面一個純量
    bool hasGradient = false;
    std::vector<double> gradient;       // fixedGradient 族的 gradient
    // 其他鍵的原始文字（例如特徵邊界條件的參考值 U、p、T），依出現次序
    std::vector<std::pair<std::string, std::string>> extras;

    const std::string* extra(const std::string& key) const
    {
        for (const auto& e : extras)
        {
            if (e.first == key)
            {
                return &e.second;
            }
        }
        return nullptr;
    }
};


struct HostField
{
    std::string name;
    std::string cls;            // volScalarField、volVectorField、surfaceScalarField
    int nComp = 1;
    bool isSurface = false;
    std::string format;         // 讀檔時的格式
    std::string dimensions = "[0 0 0 0 0 0 0]";   // 原樣保存，寫出時照抄
    std::vector<double> internal;           // 交錯配置
    std::vector<PatchFieldSpec> patches;    // 依 mesh 的 patch 順序
};


// 讀 path（例如 <case>/0/U）。檢查大小與 mesh 一致。boundaryField 的鍵依
// OpenFOAM 的規則對應到 patch（GeometricBoundaryField::readField）：
//   1. 確切的 patch 名稱
//   2. patch group（字面鍵，依項目的反向次序，後面的優先）
//   3. 其餘：empty patch 自動設為 empty；否則找名稱相符的項目，regex 鍵
//      依反向次序比對（後面的優先），須整串相符
// 仍未對上的 patch 報錯。
HostField readField(const std::string& path, const PolyMesh& mesh);

// 寫成 OpenFOAM 場檔。location 寫進檔頭。ASCII 時數值為 %.17g；binary 時
// nonuniform 串列照 UList::writeList 寫原始位元組（非空："\nN\n(" 資料 ")"，
// 空："\n0\n"），其他項目仍是文字（OpenFOAM 的 binary 格式）。依 OpenFOAM 的讀取
// 規則，文字的 |x| <= 1e-300 讀回為 0，二進位才能保留這些值。
void writeField
(
    const std::string& path,
    const HostField& f,
    const PolyMesh& mesh,
    const std::string& location,
    bool binary = false
);

} // namespace gnc

#endif
