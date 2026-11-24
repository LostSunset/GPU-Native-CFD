// 批次逐元素運算（N-028）。GPU-Native-CFD，GPL-3.0-or-later。
//
// 一次啟動做最多 8 段互不相依的運算：填值、複製、加、減、取負、取負複製。每個元素的運算與單獨
// 的 kernel 或 cudaMemset／cudaMemcpy 相同（填 0.0 與 memset 0 都是 +0.0），所以結果逐位元不變；
// 只是把每個陣列、每個分量各一次的 API 呼叫合成一次。呼叫端負責各段互不相依（同一批中沒有
// 一段讀另一段寫的陣列）。滿 8 段或物件解構時送出。

#ifndef GNC_CORE_ARRAYOPS_CUH
#define GNC_CORE_ARRAYOPS_CUH

#include <cstddef>

namespace gnc
{

class ArrayBatch
{
public:
    static constexpr int MaxSegs = 8;

    ArrayBatch() = default;
    ArrayBatch(const ArrayBatch&) = delete;
    ArrayBatch& operator=(const ArrayBatch&) = delete;
    ~ArrayBatch()
    {
        flush();
    }

    void fill(double* d, std::size_t n, double v)       { push(0, d, nullptr, n, v); }
    void copy(double* d, const double* s, std::size_t n) { push(1, d, s, n, 0.0); }
    void add(double* d, const double* s, std::size_t n)  { push(2, d, s, n, 0.0); }   // d = d + s
    void sub(double* d, const double* s, std::size_t n)  { push(3, d, s, n, 0.0); }   // d = d - s
    void negate(double* d, std::size_t n)                { push(4, d, nullptr, n, 0.0); }
    void copyNeg(double* d, const double* s, std::size_t n) { push(5, d, s, n, 0.0); }   // d = -s

    void flush();

private:
    void push(int op, double* d, const double* s, std::size_t n, double v);

    int op_[MaxSegs] = {};
    double* dst_[MaxSegs] = {};
    const double* src_[MaxSegs] = {};
    std::size_t n_[MaxSegs] = {};
    double v_[MaxSegs] = {};
    int k_ = 0;
};

} // namespace gnc

#endif
