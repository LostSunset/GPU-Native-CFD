#!/usr/bin/env python3
# variant_compare.sh --forces 的摘要：CASE_DIR 下各變體的 <variant>.coefficient.dat
# （forceCoeffs）或 <variant>.force.dat（forces 的 total_x/y/z）。每個量印出最後一步的值、
# 後段平均（最後 --tail 比例的步數，預設 0.25）與後段的標準差，以及各變體對第一個
# 變體（依 SUMMARY 的次序，否則依名稱）的差。只量測，不判定。
#
# 用法：coeff_summary.py CASE_DIR [--tail 0.25] [--order v1,v2,...]

import glob
import math
import os
import sys


def read_dat(path):
    names, rows = None, []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                h = line[1:].split("\t")
                if h and h[0].strip() == "Time":
                    names = [x.strip() for x in h[1:] if x.strip()]
                continue
            p = line.split()
            if p:
                rows.append([float(x) for x in p])
    return names, rows


def main():
    args = sys.argv[1:]
    if not args:
        print("usage: coeff_summary.py CASE_DIR [--tail 0.25] [--order v1,v2]", file=sys.stderr)
        return 2
    case_dir = args[0]
    tail = float(args[args.index("--tail") + 1]) if "--tail" in args else 0.25
    files = sorted(glob.glob(os.path.join(case_dir, "*.coefficient.dat")))
    kind = "coefficient"
    if not files:
        files = sorted(glob.glob(os.path.join(case_dir, "*.force.dat")))
        kind = "force"
    if not files:
        print("no coefficient.dat or force.dat", file=sys.stderr)
        return 1
    data = {}
    for path in files:
        v = os.path.basename(path)[: -len("." + kind + ".dat")]
        names, rows = read_dat(path)
        if kind == "force":
            # Time、total xyz、pressure xyz、viscous xyz
            names = ["Fx", "Fy", "Fz"]
            rows = [[r[0], r[1], r[2], r[3]] for r in rows]
        data[v] = (names, rows)
    order = list(data)
    if "--order" in args:
        order = [v for v in args[args.index("--order") + 1].split(",") if v in data]
    else:
        summary = os.path.join(os.path.dirname(os.path.normpath(case_dir)), "SUMMARY.tsv")
        if os.path.isfile(summary):
            seen = []
            with open(summary, encoding="utf-8") as f:
                f.readline()
                for line in f:
                    p = line.rstrip("\n").split("\t")
                    if len(p) > 2 and p[0] == os.path.basename(os.path.normpath(case_dir)) and p[2] in data:
                        seen.append(p[2])
            if seen:
                order = seen
    keys = ["Cd", "Cl", "CmPitch"] if kind == "coefficient" else ["Fx", "Fy", "Fz"]
    ref = order[0]
    print("variant\tquantity\tsteps\tfinal\ttail_mean\ttail_std\tfinal-ref\ttail_mean-ref")
    ref_stats = {}
    for v in order:
        names, rows = data[v]
        n = len(rows)
        k0 = max(0, n - max(1, int(round(tail*n))))
        for q in keys:
            if q not in names:
                continue
            c = names.index(q) + 1
            final = rows[-1][c]
            seg = [r[c] for r in rows[k0:]]
            mean = sum(seg)/len(seg)
            std = math.sqrt(sum((x - mean)**2 for x in seg)/len(seg))
            if v == ref:
                ref_stats[q] = (final, mean)
            rf, rm = ref_stats.get(q, (final, mean))
            print(
                "%s\t%s\t%d\t%.10g\t%.10g\t%.3g\t%.3g\t%.3g"
                % (v, q, n, final, mean, std, final - rf, mean - rm)
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
