// OpenFOAM 字典的子集。GPU-Native-CFD，GPL-3.0-or-later。
//
// 只支援 foamDictionary -expand 展開後的語法（N-004）：
//   key value-tokens ;     值為任意 token，括號 () [] {} 可巢狀
//   key { ... }            子字典
//   "pattern" ...          以字串為鍵（記為 pattern；目前只保存，不做比對）
// 讀到 #include、#calc、$var、codeStream 之類的指令就報錯停止
// （README 原則 5），請先以 foamDictionary -expand 展開。

#ifndef GNC_IO_DICTIONARY_HPP
#define GNC_IO_DICTIONARY_HPP

#include "io/foamTokenizer.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace gnc
{

class Dictionary
{
public:
    struct Entry
    {
        std::string key;
        bool isPattern = false;
        std::shared_ptr<Dictionary> dict;   // 子字典；否則為 nullptr
        std::vector<Token> stream;          // 值的 token（不含結尾的 ';'）
    };

private:
    std::string name_;
    std::vector<Entry> entries_;

public:
    Dictionary() = default;

    explicit Dictionary(std::string name)
    :
        name_(std::move(name))
    {}

    // 讀取字典內容。braced 為 true 時 '{' 已被消耗，讀到對應的 '}' 為止；
    // 否則讀到檔尾。
    static Dictionary parse
    (
        FoamTokenizer& t,
        const std::string& name,
        bool braced
    );

    const std::string& name() const { return name_; }

    const std::vector<Entry>& entries() const { return entries_; }

    // 非 pattern 的完全相符鍵；找不到時回傳 nullptr
    const Entry* find(const std::string& key) const;

    bool found(const std::string& key) const { return find(key) != nullptr; }

    const Dictionary& subDict(const std::string& key) const;

    // 單一 token 的值（WORD 或 STRING）
    std::string getWord(const std::string& key) const;

    std::string getWordOrDefault
    (
        const std::string& key,
        const std::string& deflt
    ) const;

    std::int64_t getLabel(const std::string& key) const;

    double getScalar(const std::string& key) const;

    [[noreturn]] void error(const std::string& msg) const;
};

} // namespace gnc

#endif
