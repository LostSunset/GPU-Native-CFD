// GNC 陣列檔：驗證用的二進位傾印格式。GPU-Native-CFD，GPL-3.0-or-later。
//
// 求解器與 tools/ref（以 OpenFOAM 建置）都用這個標頭寫出同一種格式，
// 再由 gncCompare 逐位元比對。只依賴 C++17 標準程式庫，不依賴 CUDA 或
// OpenFOAM。
//
// 格式（小端）：
//   檔頭   8 bytes "GNCARR01"
//   紀錄   char name[48]（以 0 補齊）, uint32 dtype, uint32 nComp,
//          uint64 nTuple, 資料 nTuple*nComp*elemSize bytes
//   dtype  1 = float64，2 = int32，3 = 文字（UTF-8，nComp = 1）

#ifndef GNC_CORE_ARRAYFILE_HPP
#define GNC_CORE_ARRAYFILE_HPP

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace gnc
{
namespace arrayFile
{

enum DType : std::uint32_t
{
    F64 = 1,
    I32 = 2,
    TEXT = 3
};

constexpr std::size_t nameLen = 48;
constexpr char magic[9] = "GNCARR01";

inline std::size_t elemSize(const std::uint32_t dtype)
{
    switch (dtype)
    {
        case F64: return 8;
        case I32: return 4;
        case TEXT: return 1;
        default: throw std::runtime_error("arrayFile: unknown dtype");
    }
}

struct Record
{
    std::string name;
    std::uint32_t dtype = 0;
    std::uint32_t nComp = 1;
    std::uint64_t nTuple = 0;
    std::vector<char> bytes;

    std::size_t count() const { return std::size_t(nTuple)*nComp; }
};


class Writer
{
    std::ofstream os_;

    void writeRecord
    (
        const std::string& name,
        const std::uint32_t dtype,
        const std::uint32_t nComp,
        const std::uint64_t nTuple,
        const void* data
    )
    {
        if (name.size() >= nameLen)
        {
            throw std::runtime_error("arrayFile: name too long: " + name);
        }
        char buf[nameLen] = {};
        std::memcpy(buf, name.data(), name.size());
        os_.write(buf, nameLen);
        os_.write(reinterpret_cast<const char*>(&dtype), 4);
        os_.write(reinterpret_cast<const char*>(&nComp), 4);
        os_.write(reinterpret_cast<const char*>(&nTuple), 8);
        os_.write
        (
            static_cast<const char*>(data),
            std::streamsize(nTuple*nComp*elemSize(dtype))
        );
        if (!os_)
        {
            throw std::runtime_error("arrayFile: write failed: " + name);
        }
    }

public:
    explicit Writer(const std::string& path)
    :
        os_(path, std::ios::binary | std::ios::trunc)
    {
        if (!os_)
        {
            throw std::runtime_error("arrayFile: cannot open " + path);
        }
        os_.write(magic, 8);
    }

    void f64
    (
        const std::string& name,
        const double* data,
        const std::uint64_t nTuple,
        const std::uint32_t nComp = 1
    )
    {
        writeRecord(name, F64, nComp, nTuple, data);
    }

    void i32
    (
        const std::string& name,
        const std::int32_t* data,
        const std::uint64_t nTuple,
        const std::uint32_t nComp = 1
    )
    {
        writeRecord(name, I32, nComp, nTuple, data);
    }

    void text(const std::string& name, const std::string& s)
    {
        writeRecord(name, TEXT, 1, s.size(), s.data());
    }

    void close()
    {
        os_.close();
        if (!os_)
        {
            throw std::runtime_error("arrayFile: close failed");
        }
    }
};


// 依檔案中的順序讀出所有紀錄
inline std::vector<Record> readAll(const std::string& path)
{
    std::ifstream is(path, std::ios::binary);
    if (!is)
    {
        throw std::runtime_error("arrayFile: cannot open " + path);
    }
    char head[8];
    is.read(head, 8);
    if (!is || std::memcmp(head, magic, 8) != 0)
    {
        throw std::runtime_error("arrayFile: bad magic in " + path);
    }

    std::vector<Record> recs;
    for (;;)
    {
        char buf[nameLen];
        is.read(buf, nameLen);
        if (is.gcount() == 0 && is.eof())
        {
            break;
        }
        if (!is)
        {
            throw std::runtime_error("arrayFile: truncated record in " + path);
        }
        Record r;
        r.name.assign(buf, strnlen(buf, nameLen));
        is.read(reinterpret_cast<char*>(&r.dtype), 4);
        is.read(reinterpret_cast<char*>(&r.nComp), 4);
        is.read(reinterpret_cast<char*>(&r.nTuple), 8);
        if (!is)
        {
            throw std::runtime_error("arrayFile: truncated header in " + path);
        }
        r.bytes.resize(r.count()*elemSize(r.dtype));
        is.read(r.bytes.data(), std::streamsize(r.bytes.size()));
        if (!is)
        {
            throw std::runtime_error("arrayFile: truncated data in " + path);
        }
        recs.push_back(std::move(r));
    }
    return recs;
}

} // namespace arrayFile
} // namespace gnc

#endif
