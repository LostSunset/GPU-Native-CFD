#!/usr/bin/env python3
# 論文的案例圖（GPU-Native-CFD）：網格（aerofoilNACA0012 的翼型附近、ONERA M6 機翼與 basicFinner 彈體的表面網格）
# 與 aerofoilNACA0012 的收斂流場（rhoSimpleFoam 單核對 gncRhoSimple -fast：馬赫數、壓力與兩者的差）。
# 讀 results/measured/P1-geometry 與 P1-fields（scripts/paper_extra_g02.sh），輸出 PDF 與 PNG 到 docs/paper/figures/
# （case_meshes、aerofoil_fields）與數值表 CSV（aerofoil_fields.csv：場差的統計）。
#
# 用法（在 GPU-Native-CFD 目錄）：python scripts/plots/make_case_figures.py [--fields DIR] [--geometry DIR]

import argparse
import csv
import os
import re
import xml.etree.ElementTree as ET

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.tri as mtri  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MEAS = os.path.join(ROOT, "results", "measured")
OUT = os.path.join(ROOT, "docs", "paper", "figures")
INK = "#0b0b0b"
INK2 = "#52514e"
MESH = "#2a78d6"

# 空氣（案例的 thermophysicalProperties：molWeight 28.9、Cp 1005，perfectGas）
R_GAS = 8314.47/28.9
GAMMA = 1005.0/(1005.0 - R_GAS)

# pdf.fonttype 42：PDF 內嵌 TrueType 字型（Elsevier 要求嵌入字型；預設的 Type 3 常被出版端的檢查退回）
plt.rcParams.update({"font.size": 8, "figure.facecolor": "white", "savefig.dpi": 300, "pdf.fonttype": 42})


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def foam_body(path):
    """OpenFOAM ASCII 檔去掉檔頭與註解之後的內容"""
    with open(path, encoding="utf-8", errors="replace") as f:
        s = f.read()
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.S)
    s = re.sub(r"//[^\n]*", "", s)
    i = s.find("FoamFile")
    if i >= 0:
        j = s.find("}", i)
        s = s[j + 1:]
    return s


def foam_list(s):
    """第一個「N ( ... )」清單的內容（N 之後的括號內）"""
    m = re.search(r"(\d+)\s*\(", s)
    n = int(m.group(1))
    start = m.end()
    depth, i = 1, start
    while depth:
        c = s[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
        i += 1
    return n, s[start:i - 1]


def read_points(path):
    n, body = foam_list(foam_body(path))
    a = np.array(body.replace("(", " ").replace(")", " ").split(), dtype=float).reshape(n, 3)
    return a


def read_faces(path):
    n, body = foam_list(foam_body(path))
    faces = [list(map(int, m.group(2).split())) for m in re.finditer(r"(\d+)\(([^)]*)\)", body)]
    assert len(faces) == n
    return faces


def read_boundary(path):
    s = foam_body(path)
    out = {}
    for m in re.finditer(r"(\w+)\s*\{([^}]*)\}", s):
        kv = dict(re.findall(r"(\w+)\s+([^;]+);", m.group(2)))
        if "nFaces" in kv:
            out[m.group(1)] = (int(kv["startFace"]), int(kv["nFaces"]))
    return out


def read_field(path):
    s = foam_body(path)
    m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>", s)
    n, body = foam_list(s[m.end():])
    a = np.array(body.replace("(", " ").replace(")", " ").split(), dtype=float)
    return a.reshape(n, -1) if m.group(1) == "vector" else a


def read_vtp(path):
    root = ET.parse(path).getroot()
    piece = root.find(".//Piece")
    pts = np.array(piece.find("Points/DataArray").text.split(), dtype=float).reshape(-1, 3)
    arrays = {d.get("Name"): np.array(d.text.split(), dtype=np.int64) for d in piece.find("Polys")}
    conn, offs = arrays["connectivity"], arrays["offsets"]
    polys, start = [], 0
    for end in offs:
        polys.append(conn[start:end])
        start = end
    return pts, polys


def naca0012(x):
    x = np.clip(x, 0.0, 1.0)
    return 0.6*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)


# ---------------------------------------------------------------------------
def case_meshes(geo):
    fig = plt.figure(figsize=(7.2, 2.3))
    # (a) aerofoilNACA0012：front 面的四邊形（x–z 平面）
    a = fig.add_axes([0.0, 0.05, 0.34, 0.85])
    d = os.path.join(geo, "aerofoilNACA0012")
    pts = read_points(os.path.join(d, "points"))
    faces = read_faces(os.path.join(d, "faces"))
    start, n = read_boundary(os.path.join(d, "boundary"))["front"]
    segs = []
    for f in faces[start:start + n]:
        q = pts[f][:, [0, 2]]
        segs += [q[k:k + 2] if k < len(q) - 1 else np.array([q[-1], q[0]]) for k in range(len(q))]
    a.add_collection(LineCollection(segs, colors=MESH, linewidths=0.15))
    a.set_xlim(-0.25, 1.35)
    a.set_ylim(-0.45, 0.45)
    a.set_aspect("equal")
    a.set_xticks([])
    a.set_yticks([])
    a.set_title("(a) NACA 0012, 16,000 cells", fontsize=7.5, loc="left")
    for s in a.spines.values():
        s.set_color("#c8c7c2")

    def surface(rect, files, title, elev, azim, zoom):
        ax = fig.add_axes(rect, projection="3d")
        allp = []
        for fpath in files:
            p, polys = read_vtp(fpath)
            col = Poly3DCollection([p[q] for q in polys], facecolors="#dfe9f6", edgecolors=MESH, linewidths=0.08)
            ax.add_collection3d(col)
            allp.append(p)
        p = np.vstack(allp)
        lo, hi = p.min(axis=0), p.max(axis=0)
        span = np.maximum(hi - lo, 0.05*(hi - lo).max())
        ax.set_xlim(lo[0], lo[0] + span[0])
        ax.set_ylim(lo[1], lo[1] + span[1])
        ax.set_zlim(lo[2], lo[2] + span[2])
        ax.set_box_aspect(tuple(span), zoom=zoom)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(title, fontsize=7.5, loc="left")

    surface([0.33, 0.0, 0.34, 0.95], [os.path.join(geo, "oneraM6", "wing.vtp")],
            "(b) ONERA M6 wing, 294,912 cells", 30, -60, 1.25)
    surface([0.66, 0.0, 0.34, 0.95], [os.path.join(geo, "basicFinner", f) for f in ("body.vtp", "fin1.vtp", "fin2.vtp")],
            "(c) basicFinner, 2,179,881 cells", 22, -60, 1.05)
    save(fig, "case_meshes")


# ---------------------------------------------------------------------------
def aerofoil_fields(fld):
    c = read_field(os.path.join(fld, "C"))
    x, z = c[:, 0], c[:, 2]
    runs = {}
    for name in ("of1", "gnc", "gncamg"):
        p = read_field(os.path.join(fld, name, "p"))
        u = read_field(os.path.join(fld, name, "U"))
        t = read_field(os.path.join(fld, name, "T"))
        mach = np.linalg.norm(u, axis=1)/np.sqrt(GAMMA*R_GAS*t)
        runs[name] = {"p": p, "Ma": mach}
    sel = (x > -0.4) & (x < 1.5) & (np.abs(z) < 0.6)
    tri = mtri.Triangulation(x[sel], z[sel])
    xm = x[sel][tri.triangles].mean(axis=1)
    zm = z[sel][tri.triangles].mean(axis=1)
    edge = np.max(np.hypot(np.diff(np.c_[x[sel][tri.triangles], x[sel][tri.triangles][:, :1]], axis=1),
                           np.diff(np.c_[z[sel][tri.triangles], z[sel][tri.triangles][:, :1]], axis=1)), axis=1)
    inside = (xm >= 0) & (xm <= 1) & (np.abs(zm) <= naca0012(xm))
    tri.set_mask(inside | (edge > 0.25))
    # 遠場的壓力（案例的 p 初值與出口）：全域中位數，網格大部分在遠場
    pinf = np.median(runs["of1"]["p"])
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.6), sharex=True, sharey=True, layout="constrained")
    rows = [("Ma", "Ma", 1.0, "%.1f"), ("p", "p [kPa]", 1e-3, "%.0f")]
    stats = []
    for i, (q, lab, scale, fmt) in enumerate(rows):
        a0, a1 = runs["of1"][q], runs["gncamg"][q]
        lo, hi = np.percentile(np.r_[a0[sel], a1[sel]], [0.5, 99.5])
        levels = np.linspace(lo*scale, hi*scale, 21)
        for j, (name, title) in enumerate((("of1", "rhoSimpleFoam"), ("gncamg", "gncRhoSimple -fast"))):
            ax = axes[i][j]
            cs = ax.tricontourf(tri, runs[name][q][sel]*scale, levels=levels, cmap="viridis", extend="both")
            ax.set_title("(%s) %s, %s" % ("abcdef"[3*i + j], title, lab), fontsize=7, loc="left")
        cb = fig.colorbar(cs, ax=axes[i][1], shrink=0.9, aspect=18, format=fmt)
        cb.locator = matplotlib.ticker.MaxNLocator(5)
        cb.update_ticks()
        cb.ax.tick_params(labelsize=6.5)
        ref = np.abs(a0).max() if q == "Ma" else pinf
        diff = np.abs(a1 - a0)/ref
        ax = axes[i][2]
        lev = np.logspace(-7, -3, 17)
        cs2 = ax.tricontourf(tri, np.clip(diff[sel], 1e-7, 1e-3), levels=lev, cmap="magma_r",
                             norm=matplotlib.colors.LogNorm(1e-7, 1e-3))
        ax.set_title("(%s) |difference| / %s" % ("abcdef"[3*i + 2], "max Ma" if q == "Ma" else "p∞"),
                     fontsize=7, loc="left")
        cb2 = fig.colorbar(cs2, ax=ax, shrink=0.9, aspect=18, ticks=[1e-7, 1e-5, 1e-3])
        cb2.ax.tick_params(labelsize=6.5)
        for name in ("gnc", "gncamg"):
            d = runs[name][q] - a0
            stats.append([q, name, "%.3e" % (np.linalg.norm(d)/np.linalg.norm(a0)), "%.3e" % (np.abs(d).max()/ref)])
    for ax in axes.ravel():
        xs = np.linspace(0, 1, 200)
        ax.fill_between(xs, -naca0012(xs), naca0012(xs), color="#d9d8d3", lw=0)
        ax.set_aspect("equal")
        ax.set_xlim(-0.3, 1.4)
        ax.set_ylim(-0.45, 0.45)
        ax.tick_params(labelsize=6.5)
    for ax in axes[1]:
        ax.set_xlabel("x [m]", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("z [m]", fontsize=7)
    save(fig, "aerofoil_fields")
    with open(os.path.join(OUT, "aerofoil_fields.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "run (vs rhoSimpleFoam 1 core)", "L2 relative difference",
                    "max |difference| / reference value"])
        w.writerows(stats)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fields", default=os.path.join(MEAS, "P1-fields"))
    ap.add_argument("--geometry", default=os.path.join(MEAS, "P1-geometry"))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    case_meshes(a.geometry)
    aerofoil_fields(a.fields)
    print("figures written to", OUT)
