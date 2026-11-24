// OpenFOAM 檔案的詞法掃描。GPU-Native-CFD，GPL-3.0-or-later。
//
// 整個檔案讀進記憶體後逐字掃描。支援：
//   註解     // 到行尾、/* ... */
//   標點     ( ) { } [ ] ; ,（token 開頭時）
//   字串     "..."（不處理跳脫序列以外的語法）
//   數字     以 '-'、'.' 或數字開頭，之後只收數字與 + - . E e（記為 WORD）
//   單字     其餘：照 OpenFOAM 的 word::valid，括號可巢狀（grad(U)）
//   原始位元組  二進位格式的串列內容（readRaw）
//
// 浮點數照 OpenFOAM 的轉換方式讀（readScalar），才能與 OpenFOAM 讀到的
// double 逐位元相同。OpenFOAM 不是用正確捨入的 strtod，而是 strtold 轉成
// long double 再轉 double（兩次捨入），basicFinner 的 points 有 1,383 個值
// 因此與 std::from_chars 差 1 ULP（N1 第一次 L0）。整數用 std::from_chars。

#ifndef GNC_IO_FOAMTOKENIZER_HPP
#define GNC_IO_FOAMTOKENIZER_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace gnc
{

struct Token
{
    enum Kind
    {
        END,
        WORD,
        STRING,
        PUNCT
    };

    Kind kind = END;
    std::string text;   // WORD／STRING 的內容；PUNCT 為該字元

    bool isPunct(const char c) const
    {
        return kind == PUNCT && text.size() == 1 && text[0] == c;
    }
};


class FoamTokenizer
{
    std::string path_;
    std::vector<char> buf_;
    std::size_t pos_ = 0;

public:
    explicit FoamTokenizer(const std::string& path);

    const std::string& path() const { return path_; }

    // 略過空白與註解
    void skipWs();

    // 下一個非空白字元，檔尾時為 -1（不消耗）
    int peekChar();

    bool atEnd();

    Token next();

    // 若下一個 token 是標點 c 則消耗並回傳 true
    bool tryPunct(char c);

    void expectPunct(char c);

    std::string expectWord();

    std::int64_t readLabel();

    // OpenFOAM 的數字 token（ISstream.C:705–860）轉成 double，見 parseScalar
    double readScalar();

    // 二進位串列內容：從目前位置（緊接在 '(' 或 '{' 之後）讀 n bytes
    void readRaw(void* dst, std::size_t n);

    // 目前位置所在的行號（1 起算）
    std::size_t line() const;

    [[noreturn]] void error(const std::string& msg) const;
};


// 一個完整數字 token 轉成 double，規則與 OpenFOAM v2606 相同：
//   - 只有數字（可帶開頭 '-'）：讀成 int64（溢位時 uint64），再轉 double
//     （ISstream.C:806–843，token::number()）
//   - 其餘：strtold（C locale）轉成 long double；|x| <= VSMALL（1e-300）時
//     為 +0，否則轉成 double；|x| > VGREAT（1e300）或整個 token 沒讀完則
//     失敗（scalarImpl.txx:103–123，doubleScalar.cxx 的 ScalarConvert）
// 失敗時回傳 false。
bool parseScalar(const std::string& token, double& val);

} // namespace gnc

#endif
