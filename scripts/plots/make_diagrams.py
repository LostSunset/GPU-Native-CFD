#!/usr/bin/env python3
# 論文的示意圖（GPU-Native-CFD）：架構與一次迭代的資料流、逐位元驗證的工作流程。輸出 PDF 與 PNG 到
# docs/paper/figures/（diag_architecture、diag_workflow）。不讀量測紀錄；方框的內容與 manuscript 第 3 節一致。
#
# 用法（在 GPU-Native-CFD 目錄）：python scripts/plots/make_diagrams.py

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "docs", "paper", "figures")

INK = "#0b0b0b"
INK2 = "#52514e"
HOST, HOST_EDGE = "#eaf2fb", "#2a78d6"      # 主機端
GPU, GPU_EDGE = "#e8f6ef", "#1baf7a"        # GPU 端
REF, REF_EDGE = "#fdf1e6", "#eb6834"        # CPU 參考程式
NOTE, NOTE_EDGE = "#f4f3f0", "#52514e"
FIND, FIND_EDGE = "#fff6d6", "#b88700"

# pdf.fonttype 42：PDF 內嵌 TrueType 字型（Elsevier 要求嵌入字型；預設的 Type 3 常被出版端的檢查退回）
plt.rcParams.update({"font.size": 7, "figure.facecolor": "white", "savefig.dpi": 300, "pdf.fonttype": 42})


def box(ax, x, y, w, h, lines, face, edge, size=7, title=True):
    """方框；第一行（title=True 時）粗體"""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.010",
                                fc=face, ec=edge, lw=0.9))
    n = len(lines)
    lh = 0.034*size/7
    y0 = y + h/2 + (n - 1)*lh/2
    for i, t in enumerate(lines):
        ax.text(x + w/2, y0 - i*lh, t, ha="center", va="center", fontsize=size, color=INK,
                fontweight="bold" if (title and i == 0) else "normal")


def arrow(ax, x0, y0, x1, y1, color=INK2, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=8, lw=0.9,
                                 color=color, linestyle=ls, shrinkA=0, shrinkB=0))


def label(ax, x, y, text, ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=6.3, color=INK2, linespacing=1.15)


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)), bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def architecture():
    fig, ax = plt.subplots(figsize=(7.2, 4.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.18, 0.985, "Host (CPU, one thread)", ha="center", fontsize=8, fontweight="bold", color=HOST_EDGE)
    ax.text(0.72, 0.985, "GPU (device memory, data resident for the whole run)", ha="center", fontsize=8,
            fontweight="bold", color=GPU_EDGE)
    # 主機端
    hx, hw = 0.01, 0.34
    box(ax, hx, 0.83, hw, 0.12, ["Read the OpenFOAM case", "foamDictionary -expand;",
                                 "polyMesh, fields, expanded dictionaries"], HOST, HOST_EDGE)
    box(ax, hx, 0.62, hw, 0.16, ["Set-up (once per mesh)", "cell-face gather tables in face order,",
                                 "level schedules, colourings,", "AMG hierarchy"], HOST, HOST_EDGE)
    box(ax, hx, 0.29, hw, 0.27, ["Iteration control", "launch the kernels of one outer",
                                 "(or SIMPLE) iteration;", "receive residual norms and",
                                 "convergence flags (a few scalars);", "pseudo-time and relaxation control"],
        HOST, HOST_EDGE)
    box(ax, hx, 0.06, hw, 0.17, ["Output", "OpenFOAM field files, forces and", "force coefficients at write times"],
        HOST, HOST_EDGE)
    for y0, y1 in ((0.83, 0.78), (0.62, 0.56), (0.29, 0.23)):
        arrow(ax, hx + hw/2, y0, hx + hw/2, y1)
    # GPU 端
    gx, gw = 0.47, 0.52
    box(ax, gx, 0.84, gw, 0.10, ["Mesh and fields", "geometry (areas, centres, volumes, weights);",
                                 "fields as structures of arrays"], GPU, GPU_EDGE)
    ax.add_patch(FancyBboxPatch((gx, 0.20), gw, 0.585, boxstyle="round,pad=0.006,rounding_size=0.010",
                                fc="none", ec=GPU_EDGE, lw=0.9, ls="--"))
    ax.text(gx + 0.01, 0.765, "one iteration in hand-written CUDA kernels", fontsize=6.3, color=GPU_EDGE,
            va="center", fontweight="bold")
    sx, sw = gx + 0.03, gw - 0.06
    steps = [
        (["Boundary conditions, thermodynamics"], 0.685, 0.06),
        (["Fluxes and matrix assembly (gather over faces)"], 0.595, 0.06),
        (["Linear solves", "gncHisa: GMRES with LU-SGS;",
          "gncRhoSimple: PBiCGStab/DILU, PCG/DIC or PCG/AMG"], 0.47, 0.11),
        (["Field update, limits, turbulence model"], 0.345, 0.06),
        (["Reductions: sequential order or fixed tree"], 0.255, 0.06),
    ]
    for lines, yc, h in steps:
        box(ax, sx, yc - h/2, sw, h, lines, "white", GPU_EDGE, title=len(lines) > 1)
    for (l0, y0, h0), (l1, y1, h1) in zip(steps[:-1], steps[1:]):
        arrow(ax, sx + sw/2, y0 - h0/2, sx + sw/2, y1 + h1/2)
    box(ax, gx, 0.04, gw, 0.12, ["Two modes of every kernel",
                                 "reference: operation order of the CPU original (bitwise)",
                                 "-fast: multicolour, tree reductions, FP32 Jacobian, AMG"],
        NOTE, NOTE_EDGE, size=6.5)
    # 主機與 GPU 之間
    arrow(ax, hx + hw, 0.89, gx, 0.89)
    label(ax, (hx + hw + gx)/2, 0.915, "copy once")
    arrow(ax, hx + hw, 0.70, gx, 0.70)
    label(ax, (hx + hw + gx)/2, 0.725, "upload once")
    arrow(ax, hx + hw, 0.47, sx, 0.47)
    label(ax, (hx + hw + sx)/2, 0.51, "kernel launches\n(CUDA Graphs)")
    arrow(ax, sx, 0.255, hx + hw, 0.37)
    label(ax, (hx + hw + sx)/2 + 0.005, 0.255, "a few scalars\nper iteration")
    arrow(ax, gx, 0.22, hx + hw, 0.17)
    label(ax, (hx + hw + gx)/2, 0.155, "fields at\nwrite times")
    save(fig, "diag_architecture")


def workflow():
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    lx, lw = 0.01, 0.30
    mx, mw = 0.37, 0.25
    rx, rw = 0.68, 0.31
    # 參考端的 OpenFOAM 是作者發行版的 v2606 建置（overlay 換掉兩個 lduMatrix 運算檔並修正 profiling 的 double free），
    # 只有 HiSA 是未修改的（manuscript 的 Table 5 與第 8 節 Conditions of bitwise agreement）
    box(ax, lx, 0.74, lw, 0.22, ["CPU reference", "OpenFOAM v2606,", "unmodified HiSA;", "GCC 13.3, glibc 2.39;",
                                 "dump points: function object,", "recording preconditioner"], REF, REF_EDGE)
    box(ax, lx, 0.42, lw, 0.22, ["GPU rewrite, reference mode", "gncHisa, gncRhoSimple;", "--fmad=false, ordered gathers,",
                                 "level schedules, sequential sums,", "ports of glibc pow, exp, tanh, log"], GPU, GPU_EDGE)
    box(ax, mx, 0.58, mw, 0.20, ["States and files", "state after each stage", "(binary dumps);",
                                 "written field files"], NOTE, NOTE_EDGE)
    box(ax, rx, 0.58, rw, 0.20, ["Bitwise comparison", "L0 geometry, L1 operators,", "L2 one iteration, L3 complete run,",
                                 "written files; every GPU run twice"], "white", NOTE_EDGE)
    box(ax, rx, 0.27, rw, 0.20, ["Localise", "first differing state and", "pattern of differing records",
                                 "give the operation (Table 3)"], FIND, FIND_EDGE)
    box(ax, mx, 0.27, mw, 0.20, ["Correct", "fix the kernel, rebuild,", "rerun from L0"], "white", GPU_EDGE)
    box(ax, lx, 0.04, lw, 0.18, ["Performance mode (-fast)", "built on the reference mode,", "change by change"],
        GPU, GPU_EDGE)
    box(ax, mx, 0.04, rx + rw - mx, 0.18, ["Checks of the performance mode",
                                           "bitwise-neutral changes: identical files before and after (Section 5.4)",
                                           "algorithmic changes: same discrete solution (Section 6)",
                                           "every timing record: build hash, tool chain, logs (Appendix A)"],
        NOTE, NOTE_EDGE, size=6.5)
    arrow(ax, lx + lw, 0.85, mx, 0.73)
    arrow(ax, lx + lw, 0.53, mx, 0.63)
    arrow(ax, mx + mw, 0.68, rx, 0.68)
    arrow(ax, rx + rw/2, 0.58, rx + rw/2, 0.47)
    label(ax, rx + rw/2 + 0.035, 0.525, "differ")
    label(ax, rx + rw/2, 0.82, "identical: next level, next case")
    arrow(ax, rx, 0.37, mx + mw, 0.37)
    arrow(ax, mx, 0.37, lx + lw, 0.47)
    arrow(ax, lx + lw/2, 0.42, lx + lw/2, 0.22)
    arrow(ax, lx + lw, 0.13, mx, 0.13)
    save(fig, "diag_workflow")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    architecture()
    workflow()
    print("diagrams written to", OUT)
