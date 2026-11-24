// 循序歸約的時間（N7 的剖析）。GPU-Native-CFD，GPL-3.0-or-later。
//
// 量 core/reduce.cuh 的 sumSequential（一個 thread 由 0 依序累加）與 sumTree
// 在不同 n 的每次呼叫時間（含啟動與取回結果），並檢查兩次呼叫結果相同。
//
// 用法：gncSumBench [--n 23552,294912] [--repeat 50]

#include "core/gncCuda.cuh"
#include "core/reduce.cuh"

#include <chrono>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>

int main(int argc, char* argv[])
{
    std::string ns = "8000,23552,294912,2179881";
    int repeat = 50;
    for (int i = 1; i + 1 < argc; i += 2)
    {
        const std::string a = argv[i];
        if (a == "--n") ns = argv[i + 1];
        else if (a == "--repeat") repeat = std::stoi(argv[i + 1]);
    }
    std::vector<std::size_t> sizes;
    {
        std::stringstream ss(ns);
        std::string tok;
        while (std::getline(ss, tok, ','))
        {
            sizes.push_back(std::stoull(tok));
        }
    }
    using Clock = std::chrono::steady_clock;
    for (const std::size_t n : sizes)
    {
        std::vector<double> h(n);
        for (std::size_t i = 0; i < n; ++i)
        {
            h[i] = 1.0/double(i + 1);
        }
        gnc::DeviceArray<double> d(n);
        d.upload(h);
        const double r0 = gnc::sumSequential(d.data(), n);
        auto t0 = Clock::now();
        double r = 0;
        for (int k = 0; k < repeat; ++k)
        {
            r = gnc::sumSequential(d.data(), n);
        }
        const double tSeq = std::chrono::duration<double>(Clock::now() - t0).count()/repeat;
        t0 = Clock::now();
        double rt = 0;
        for (int k = 0; k < repeat; ++k)
        {
            rt = gnc::sumTree(d.data(), n);
        }
        const double tTree = std::chrono::duration<double>(Clock::now() - t0).count()/repeat;
        std::printf
        (
            "n %zu: sumSequential %.3f ms (repeatable %d)  sumTree %.3f ms  rel diff %.3e\n",
            n, 1e3*tSeq, int(r == r0), 1e3*tTree, (rt - r)/r
        );
    }
    return 0;
}
