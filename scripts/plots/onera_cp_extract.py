#!/usr/bin/env python3
# ONERA M6 翼面的展向站 Cp（與 Schmitt & Charpin 的實驗比較用）。只用 Python 標準函式庫。
#
# 讀 <case>/constant/polyMesh（ASCII 的 points、faces、boundary）與 <case>/<time>/p（ASCII），在
# z = eta*b（b = 1.1963 m）的平面切過 wing patch：與平面相交的每個面取其交線段的中點與該面的 p
# （面心值，分段常數）。每個站以交點的最小、最大 x 為前緣、後緣算 x/c；對稱翼型、無扭轉，y > 0
# 為上翼面。Cp = (p - p_inf)/(0.5 rho_inf |U_inf|^2)，自由流同 make_onera_forces_case.sh。
#
# 用法：onera_cp_extract.py CASE_DIR TIME > cp.csv
#   輸出欄位：eta, surface, x_over_c, cp

import math
import os
import re
import sys

ETAS = [0.20, 0.44, 0.65, 0.80, 0.90, 0.95, 0.99]
B = 1.1963
P_INF = 80510.081
UX, UY = 268.654, 14.362
R_GAS = 8314.47/28.966
T_INF = 255.556
RHO_INF = P_INF/(R_GAS*T_INF)
Q_INF = 0.5*RHO_INF*(UX*UX + UY*UY)


def list_body(text, start):
    """從 start 之後的第一個 'N\\n(' 或 'N(' 讀出 N 個項目的文字（到對應的 ')'）"""
    m = re.compile(r"(\d+)\s*\(").search(text, start)
    n = int(m.group(1))
    i = m.end()
    depth = 1
    j = i
    while depth:
        c = text[j]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        j += 1
    return n, text[i:j - 1]


def read_points(path):
    with open(path) as f:
        text = f.read()
    body_start = text.index("FoamFile")
    body_start = text.index("}", body_start)
    n, body = list_body(text, body_start)
    vals = [float(v) for v in re.findall(r"[-+0-9.eE]+", body)]
    return [(vals[3*i], vals[3*i + 1], vals[3*i + 2]) for i in range(n)]


def read_faces(path, start, count):
    out = []
    with open(path) as f:
        text = f.read()
    body_start = text.index("}", text.index("FoamFile"))
    n, body = list_body(text, body_start)
    k = 0
    for m in re.finditer(r"\d+\(([^)]*)\)", body):
        if k >= start + count:
            break
        if k >= start:
            out.append([int(v) for v in m.group(1).split()])
        k += 1
    return out


def read_patch(path, name):
    with open(path) as f:
        text = f.read()
    m = re.search(r"\b%s\s*\{([^}]*)\}" % re.escape(name), text)
    blk = m.group(1)
    start = int(re.search(r"startFace\s+(\d+)", blk).group(1))
    n = int(re.search(r"nFaces\s+(\d+)", blk).group(1))
    return start, n


def read_patch_value(path, name):
    with open(path) as f:
        text = f.read()
    bf = text.index("boundaryField")
    m = re.compile(r"\n\s*%s\s*\{" % re.escape(name)).search(text, bf)
    v = text.index("value", m.end())
    if text[v:v + 40].split()[1] == "uniform":
        val = float(re.search(r"uniform\s+([-+0-9.eE]+)", text[v:]).group(1))
        return None, val
    n, body = list_body(text, v)
    return [float(x) for x in body.split()], None


def main():
    case, time = sys.argv[1], sys.argv[2]
    mesh = os.path.join(case, "constant", "polyMesh")
    start, n = read_patch(os.path.join(mesh, "boundary"), "wing")
    pts = read_points(os.path.join(mesh, "points"))
    faces = read_faces(os.path.join(mesh, "faces"), start, n)
    p, uni = read_patch_value(os.path.join(case, time, "p"), "wing")
    if p is None:
        p = [uni]*n
    print("eta,surface,x_over_c,cp")
    for eta in ETAS:
        z0 = eta*B
        hits = []
        for fi, f in enumerate(faces):
            v = [pts[i] for i in f]
            zs = [q[2] for q in v]
            if not (min(zs) < z0 < max(zs)):
                continue
            seg = []
            for a, b in zip(v, v[1:] + v[:1]):
                if (a[2] - z0)*(b[2] - z0) < 0:
                    t = (z0 - a[2])/(b[2] - a[2])
                    seg.append((a[0] + t*(b[0] - a[0]), a[1] + t*(b[1] - a[1])))
            if len(seg) >= 2:
                x = sum(s[0] for s in seg)/len(seg)
                y = sum(s[1] for s in seg)/len(seg)
                hits.append((x, y, (p[fi] - P_INF)/Q_INF))
        if not hits:
            continue
        xle = min(h[0] for h in hits)
        xte = max(h[0] for h in hits)
        c = xte - xle
        for surf, sel in (("upper", lambda y: y > 0), ("lower", lambda y: y <= 0)):
            for x, y, cp in sorted(h for h in hits if sel(h[1])):
                print("%.2f,%s,%.6f,%.6f" % (eta, surf, (x - xle)/c, cp))
    return 0


if __name__ == "__main__":
    sys.exit(main())
