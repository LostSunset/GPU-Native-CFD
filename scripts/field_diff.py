#!/usr/bin/env python3
# 兩個時間目錄的場差異（variant_compare.sh 使用）：對兩邊都有的 p T U rho nuTilda
# k omega nut alphat，讀 internalField（ASCII，uniform 或 nonuniform List），印出
#   max|a-b|、max|a-b|/max|a|、||a-b||_2/||a||_2
# 向量場以三個分量合併計算。只量測，不判定。
#
# 用法：field_diff.py DIR_A DIR_B

import math
import os
import re
import sys

FIELDS = ["p", "T", "U", "rho", "nuTilda", "k", "omega", "nut", "alphat"]


def read_internal(path):
    with open(path, encoding="utf-8") as f:
        s = f.read()
    m = re.search(r"\binternalField\s+(uniform|nonuniform)\s", s)
    if not m:
        raise ValueError(path + ": no internalField")
    rest = s[m.end():]
    if m.group(1) == "uniform":
        v = rest[:rest.index(";")].strip()
        if v.startswith("("):
            return [float(x) for x in v.strip("()").split()], True
        return [float(v)], True
    # nonuniform List<type> N ( ... )
    m2 = re.match(r"\s*List<(\w+)>\s*(\d+)\s*\(", rest)
    if not m2:
        raise ValueError(path + ": unsupported internalField")
    n = int(m2.group(2))
    body = rest[m2.end():]
    vals = []
    if m2.group(1) == "scalar":
        toks = body.split(None, n)
        vals = [float(t) for t in toks[:n]]
    else:
        for t in re.findall(r"\(([^()]*)\)", body)[:n]:
            vals.extend(float(x) for x in t.split())
    return vals, False


def main():
    if len(sys.argv) != 3:
        print("usage: field_diff.py DIR_A DIR_B", file=sys.stderr)
        return 2
    a_dir, b_dir = sys.argv[1], sys.argv[2]
    print("A=%s\nB=%s" % (a_dir, b_dir))
    print("%-8s %12s %22s %22s %22s" % ("field", "n", "max|a-b|", "max|a-b|/max|a|", "L2(a-b)/L2(a)"))
    for name in FIELDS:
        pa, pb = os.path.join(a_dir, name), os.path.join(b_dir, name)
        if not (os.path.isfile(pa) and os.path.isfile(pb)):
            continue
        a, _ = read_internal(pa)
        b, _ = read_internal(pb)
        if len(a) != len(b):
            print("%-8s size mismatch %d %d" % (name, len(a), len(b)))
            continue
        dmax = max(abs(x - y) for x, y in zip(a, b))
        amax = max(abs(x) for x in a)
        l2d = math.sqrt(sum((x - y)*(x - y) for x, y in zip(a, b)))
        l2a = math.sqrt(sum(x*x for x in a))
        print(
            "%-8s %12d %22.6e %22.6e %22.6e"
            % (name, len(a), dmax, dmax/amax if amax > 0 else 0.0, l2d/l2a if l2a > 0 else 0.0)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
