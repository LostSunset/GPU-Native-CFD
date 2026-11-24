#!/usr/bin/env python3
# ONERA M6：計算的展向站 Cp 內插到實驗點（x/c，同一翼面）後的均方根差與最大差。
# 用法：python scripts/plots/onera_cp_rms.py CP_CSV [CP_CSV ...]
import os
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXP = os.path.join(ROOT, "results", "external", "oneraM6-exp-schmitt-charpin")
ETAS = [0.20, 0.44, 0.65, 0.80, 0.90, 0.95, 0.99]


def read_cp(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            eta, surf, x, cp = line.strip().split(",")
            d.setdefault((round(float(eta), 2), surf), []).append((float(x), float(cp)))
    for k in d:
        d[k].sort()
    return d


def interp(pts, x):
    if x <= pts[0][0]:
        return pts[0][1]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (y1 - y0)*(x - x0)/(x1 - x0) if x1 > x0 else y0
    return pts[-1][1]


for path in sys.argv[1:]:
    d = read_cp(path)
    print("==", os.path.basename(path))
    print("eta\tn\trms\tmax")
    all_sq, n_all = 0.0, 0
    for k, eta in enumerate(ETAS):
        sq, n, mx = 0.0, 0, 0.0
        for s, surf in (("u", "upper"), ("l", "lower")):
            with open(os.path.join(EXP, "cp%d%s.ex" % (k + 1, s)), encoding="utf-8") as f:
                for line in f:
                    v = line.split()
                    if len(v) < 2:
                        continue
                    x, cpe = float(v[0]), -float(v[1])
                    e = interp(d[(eta, surf)], x) - cpe
                    sq += e*e
                    n += 1
                    mx = max(mx, abs(e))
        all_sq += sq
        n_all += n
        print("%.2f\t%d\t%.4f\t%.4f" % (eta, n, (sq/n) ** 0.5, mx))
    print("all\t%d\t%.4f" % (n_all, (all_sq/n_all) ** 0.5))
