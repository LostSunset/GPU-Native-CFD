#!/usr/bin/env python3
# 論文用的圖（GPU-Native-CFD）：只讀 results/measured 的紀錄，輸出 PDF（向量，投稿用）、PNG 與對應的 CSV
# （圖的數值表）到 docs/paper/figures/。圖內不放標題（說明寫在論文的圖說），白底。顏色依 dataviz 的參考調色板（固定次序：藍、橙、
# 綠松、黃、洋紅；已以 validate_palette.js 驗證，對比度 WARN 以直接標註與 CSV 表補足）。
#
# 用法（在 GPU-Native-CFD 目錄）：python scripts/plots/make_figures.py

import csv
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
MEAS = os.path.join(ROOT, "results", "measured")
OUT = os.path.join(ROOT, "docs", "paper", "figures")

# 固定的實體顏色（全部的圖相同）：HiSA 1／8／32 核、gncHisa 參考版、-fast、多色 LU-SGS
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
SURFACE = "#ffffff"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "font.size": 9,
        "axes.titlesize": 10,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "savefig.dpi": 300,
        # PDF 內嵌 TrueType 字型（Elsevier 要求嵌入字型；預設的 Type 3 常被出版端的檢查退回）
        "pdf.fonttype": 42,
    }
)


def read_tsv(path):
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))
    head = rows[0]
    return [dict(zip(head, r)) for r in rows[1:]]


def exec_times(log):
    """每步結束時的 ExecutionTime（HiSA 與 gncHisa 的 log 都有）"""
    with open(log, encoding="utf-8", errors="replace") as f:
        return [float(m.group(1)) for m in re.finditer(r"^ExecutionTime = ([0-9.eE+-]+) s", f.read(), re.M)]


def hisa_residuals(log):
    """HiSA：每步第一個「GMRES iteration: 0」的 rho 殘差"""
    out = []
    with open(log, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = re.match(r"\s*GMRES iteration: 0\s+Residual: ([0-9.eE+-]+)", line)
            if m:
                out.append(float(m.group(1)))
    return out


def gnc_history(path):
    steps, rho = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.split("\t")
            steps.append(int(float(p[0])))
            rho.append(float(p[2]))
    return steps, rho


def clock_times(log):
    """每步結束時的 ClockTime（牆鐘，整數秒；與 ExecutionTime 同一行）"""
    with open(log, encoding="utf-8", errors="replace") as f:
        return [float(m.group(1)) for m in
                re.finditer(r"^ExecutionTime = .*?ClockTime = ([0-9.eE+-]+) s", f.read(), re.M)]


def loop_per_step(log):
    t = exec_times(log)
    return (t[-1] - t[0])/(len(t) - 1) if len(t) > 1 else float("nan")


def clock_per_step(log):
    """以 ClockTime 差計的每步牆鐘時間。SPUMA 的 hisa 等待 GPU 的時間不計入 ExecutionTime（CPU 時間），
    所以 SPUMA 的紀錄要用這個（同 rhoSimpleFoam 的 per_iter_clock_s）"""
    t = clock_times(log)
    return (t[-1] - t[0])/(len(t) - 1) if len(t) > 1 else float("nan")


def profile_mean(path):
    with open(path, encoding="utf-8") as f:
        m = re.search(r"per iteration ([0-9.eE+-]+) s", f.read())
    return float(m.group(1)) if m else float("nan")


def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, "%s.%s" % (name, ext)))
    plt.close(fig)


def write_csv(name, header, rows):
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


# ---------------------------------------------------------------------------
# 圖 1：每外迭代的時間（HiSA 1／8／32 核、gncHisa 參考版、gncHisa -fast）
def fig_time_per_iteration():
    cases = [
        ("flatPlateSA", "flatPlate SA\n8,000 cells", "CPU2-g02-small", "CPU2-g02-gncfast-small"),
        ("rae2822SA", "RAE2822 SA\n23,552 cells", "CPU2-g02-small", "CPU2-g02-gncfast-small"),
        ("oneraM6SA", "ONERA M6 SA\n294,912 cells", "CPU2-g02-onera", "CPU2-g02-gncfast-onera"),
        ("basicFinnerLaminar", "basicFinner (laminar)\n2,179,881 cells", "CPU2-g02-finner", "CPU2-g02-gncfast-finner"),
    ]
    runs = ["HiSA, 1 core", "HiSA, 8 cores", "HiSA, 32 cores", "gncHisa reference (bitwise)", "gncHisa -fast"]
    data = []
    for key, _, rec, fastrec in cases:
        d = os.path.join(MEAS, rec, key)
        vals = [loop_per_step(os.path.join(d, "log.hisa-np%d" % n)) for n in (1, 8, 32)]
        vals.append(profile_mean(os.path.join(d, "profile.txt")))
        fast = [r for r in read_tsv(os.path.join(MEAS, fastrec, "SUMMARY.tsv")) if r["case"] == key][0]
        vals.append(float(fast["profile_mean_s"]))
        data.append(vals)
    write_csv(
        "fig1_time_per_iteration.csv", ["case"] + runs,
        [[c[0]] + ["%.6g" % v for v in vals] for c, vals in zip(cases, data)],
    )

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    nr = len(runs)
    width = 0.8/nr
    for j, run in enumerate(runs):
        xs = [i + (j - (nr - 1)/2)*width for i in range(len(cases))]
        ys = [vals[j] for vals in data]
        ax.bar(xs, ys, width*0.88, color=SERIES[j], label=run, zorder=3)
        if j == nr - 1:
            for x, y, vals in zip(xs, ys, data):
                ax.annotate("%.1f×" % (vals[2]/y) if vals[2]/y >= 1 else "%.2f×" % (vals[2]/y),
                            (x, y), xytext=(0, 3), textcoords="offset points", ha="center",
                            fontsize=7, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([c[1] for c in cases])
    ax.set_ylabel("wall time per outer iteration [s]")
    ax.legend(ncol=3, loc="upper left", fontsize=7.5)
    ax.set_ylim(1e-3, 3e2)
    fig.tight_layout()
    save(fig, "fig1_time_per_iteration")


# ---------------------------------------------------------------------------
# 圖 2：ONERA M6 的收斂：(a) rho 殘差對牆鐘時間，(b) Cl 對外迭代
def fig_onera():
    d = os.path.join(MEAS, "M6-g02-conv", "oneraM6SA")
    h = os.path.join(MEAS, "M6-g02-hisa32", "oneraM6SA", "log.hisa-np32")
    series = []
    for name, label in (("ref", "gncHisa reference (bitwise)"), ("fast", "gncHisa -fast")):
        steps, rho = gnc_history(os.path.join(d, "history.%s.tsv" % name))
        t = exec_times(os.path.join(d, "log.%s" % name))
        n = min(len(t), len(rho))
        series.append((label, t[:n], rho[:n]))
    hr = hisa_residuals(h)
    ht = exec_times(h)
    n = min(len(hr), len(ht))
    series.insert(0, ("HiSA, 32 cores", ht[:n], hr[:n]))
    colors = [SERIES[2], SERIES[3], SERIES[4]]

    def coeff(path):
        it, cl = [], []
        with open(path, encoding="utf-8") as f:
            names = None
            for line in f:
                if line.startswith("#"):
                    p = [x.strip() for x in line[1:].split("\t")]
                    if p and p[0] == "Time":
                        names = p[1:]
                    continue
                v = line.split()
                if v and names:
                    it.append(float(v[0]))
                    cl.append(float(v[1 + names.index("Cl")]))
        return it, cl

    cl_ref = coeff(os.path.join(d, "ref.coefficient.dat"))
    cl_fast = coeff(os.path.join(d, "fast.coefficient.dat"))
    write_csv(
        "fig2_onera_convergence.csv", ["run", "final_time_s", "final_rho_residual"],
        [[lab, "%.4g" % t[-1], "%.4g" % r[-1]] for lab, t, r in series],
    )

    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for (lab, t, r), c in zip(series, colors):
        a.plot(t, r, color=c, label="%s: %.0f s" % (lab, t[-1]), lw=1.5)
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("wall-clock time [s]")
    a.set_ylabel("rho residual (normalised)")
    a.set_title("(a) ONERA M6, 2000 outer iterations", loc="left")
    a.legend(fontsize=7, loc="lower left")
    b.plot(cl_ref[0], cl_ref[1], color=colors[1], label="reference", lw=1.5)
    b.plot(cl_fast[0], cl_fast[1], color=colors[2], label="-fast", lw=1.0, ls="--")
    b.set_xlabel("outer iteration")
    b.set_ylabel("$C_L$")
    b.set_ylim(0.20, 0.30)
    b.set_title("(b) lift coefficient", loc="left")
    b.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    save(fig, "fig2_onera_convergence")


# ---------------------------------------------------------------------------
# 圖 3：flatPlate SA 的 rho 殘差對牆鐘時間（參考版、多色 LU-SGS、fast）
def fig_flatplate():
    d = os.path.join(MEAS, "N7-g02-conv", "flatPlateSA")
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    rows = []
    for (name, lab), c in zip(
        (("ref", "reference (bitwise)"), ("lusgsColor", "multicolour LU-SGS"), ("fast", "-fast")),
        (SERIES[3], SERIES[5], SERIES[4]),
    ):
        steps, rho = gnc_history(os.path.join(d, "history.%s.tsv" % name))
        t = exec_times(os.path.join(d, "log.%s" % name))
        n = min(len(t), len(rho))
        ax.plot(t[1:n], rho[1:n], color=c, label="%s: %.0f s" % (lab, t[n - 1]), lw=1.5)
        hit = next((i for i in range(1, n) if rho[i] <= 1e-4), None)
        rows.append([lab, "%.4g" % t[n - 1], "%.4g" % (t[hit] if hit else float("nan"))])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.axhline(1e-4, color=INK2, lw=0.8, ls=":")
    ax.set_xlabel("wall-clock time [s]")
    ax.set_ylabel("rho residual (normalised)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "fig3_flatplate_residual")
    write_csv("fig3_flatplate_residual.csv", ["run", "time_2000_steps_s", "time_to_1e-4_s"], rows)


# ---------------------------------------------------------------------------
# 圖 4：ONERA M6 的展向站 Cp 對實驗（Schmitt & Charpin，results/external/）
def fig_onera_cp():
    exp_dir = os.path.join(ROOT, "results", "external", "oneraM6-exp-schmitt-charpin")
    cp_dir = os.path.join(MEAS, "M6-g02-conv", "cp")
    etas = [0.20, 0.44, 0.65, 0.80, 0.90, 0.95, 0.99]

    def read_cp(path):
        d = {}
        with open(path, encoding="utf-8") as f:
            next(f)
            for line in f:
                eta, surf, x, cp = line.strip().split(",")
                d.setdefault((round(float(eta), 2), surf), []).append((float(x), float(cp)))
        return d

    runs = [
        ("gncHisa reference (bitwise)", read_cp(os.path.join(cp_dir, "cp.ref.csv")), SERIES[3], "-", 1.6),
        ("gncHisa -fast", read_cp(os.path.join(cp_dir, "cp.fast.csv")), SERIES[4], "--", 1.0),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(7.2, 4.2), sharex=True, sharey=True)
    rows = []
    for k, eta in enumerate(etas):
        ax = axes[k//4][k % 4]
        for s in ("u", "l"):
            xs, ys = [], []
            with open(os.path.join(exp_dir, "cp%d%s.ex" % (k + 1, s)), encoding="utf-8") as f:
                for line in f:
                    v = line.split()
                    if len(v) >= 2:
                        xs.append(float(v[0]))
                        ys.append(float(v[1]))      # 檔案的第 2 欄是 -Cp
            ax.plot(xs, ys, ls="none", marker="o", ms=3.2, mfc="none", mec=INK, mew=0.7,
                    label="experiment (AGARD AR-138)" if (k == 0 and s == "u") else None)
        for name, d, c, ls, lw in runs:
            for j, surf in enumerate(("upper", "lower")):
                pts = d.get((eta, surf), [])
                ax.plot([q[0] for q in pts], [-q[1] for q in pts], color=c, ls=ls, lw=lw,
                        label=name if (k == 0 and j == 0) else None)
                rows += [[eta, name, surf, "%.6f" % q[0], "%.6f" % q[1]] for q in pts]
        ax.set_title("η = %.2f" % eta, loc="left", fontsize=8.5)
        ax.set_ylim(-1.2, 1.6)
        ax.set_xlim(0, 1)
        if k % 4 == 0:
            ax.set_ylabel("$-C_p$")
        if k >= 3:
            ax.set_xlabel("x/c")
    leg = axes[1][3]
    leg.axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    leg.legend(handles, labels, loc="center", fontsize=7.5)
    leg.text(0.5, 0.12, "ONERA M6, M 0.8395, α 3.06°\nSA, 294,912 cells, 2000 iterations",
             ha="center", va="center", fontsize=7.5, color=INK2, transform=leg.transAxes)
    fig.tight_layout()
    save(fig, "fig4_onera_cp")
    write_csv("fig4_onera_cp.csv", ["eta", "run", "surface", "x_over_c", "cp"], rows)


# ---------------------------------------------------------------------------
# 圖 5：三條 GPU 路線（同一張 L40）與 HiSA CPU 在同樣兩個案例上的每步時間
def fig_three_routes():
    # 案例：標籤、案例名、CPU 與參考版的紀錄、混合式紀錄與案例名、-fast 紀錄、SPUMA 紀錄
    cases = [
        ("ONERA M6 SA\n294,912 cells", "oneraM6SA", "CPU2-g02-onera", "HYB-g02-onera", "oneraM6Hybrid",
         "CPU2-g02-gncfast-onera", "SPUMA-g02"),
        ("basicFinner (laminar)\n2,179,881 cells", "basicFinnerLaminar", "CPU2-g02-finner", "HYB-g02-finner",
         "basicFinnerLaminarHybrid", "CPU2-g02-gncfast-finner", "SPUMA-g02"),
        ("modifiedFinner SST\n3,940,188 cells", "modifiedFinnerSST", "MF-g02-sst", "MF-g02-hyb-sst",
         "modifiedFinnerSSTHybrid", "MF-g02-gncfast-sst", "MF-g02-spuma-sst"),
    ]
    runs = ["HiSA, 1 core", "HiSA, 32 cores", "general platform (HiSA on SPUMA)",
            "hybrid (HiSA + GPU Jacobian/GMRES)", "gncHisa reference (bitwise)", "gncHisa -fast"]
    colors = [SERIES[0], SERIES[2], "#6250d6", "#e34948", SERIES[3], SERIES[4]]
    data = []
    for _, key, cpu, hyb, hybname, fast, spuma in cases:
        d = os.path.join(MEAS, cpu, key)
        h = [r for r in read_tsv(os.path.join(MEAS, hyb, "SUMMARY.tsv")) if r["case"] == hybname][0]
        f = [r for r in read_tsv(os.path.join(MEAS, fast, "SUMMARY.tsv")) if r["case"] == key][0]
        data.append([
            loop_per_step(os.path.join(d, "log.hisa-np1")),
            loop_per_step(os.path.join(d, "log.hisa-np32")),
            # SPUMA：牆鐘（ClockTime 差）。HiSA CPU 與混合式的 ExecutionTime 與 ClockTime 一致到整數秒
            # ClockTime 的解析度，沿用解析度 0.01 s 的 ExecutionTime
            clock_per_step(os.path.join(MEAS, spuma, key, "spgpu", "log.hisa")),
            float(h["loop_per_step_s"]),
            profile_mean(os.path.join(d, "profile.txt")),
            float(f["profile_mean_s"]),
        ])
    write_csv(
        "fig5_three_routes.csv", ["case"] + runs,
        [[c[1]] + ["%.6g" % v for v in vals] for c, vals in zip(cases, data)],
    )
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    nr = len(runs)
    width = 0.84/nr
    for j, run in enumerate(runs):
        xs = [i + (j - (nr - 1)/2)*width for i in range(len(cases))]
        ys = [vals[j] for vals in data]
        ax.bar(xs, ys, width*0.88, color=colors[j], label=run, zorder=3)
        for x, y, vals in zip(xs, ys, data):
            ax.annotate("%.3g×" % (vals[0]/y), (x, y), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=6.5, color=INK)
    ax.set_yscale("log")
    ax.set_xticks(range(len(cases)))
    ax.set_xticklabels([c[0] for c in cases])
    ax.set_ylabel("wall time per outer iteration [s]")
    ax.set_ylim(1e-2, 1e3)
    ax.legend(ncol=2, fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "fig5_three_routes")


# ---------------------------------------------------------------------------
# 圖 6：gncRhoSimple（N6、N6d）每 SIMPLE 迭代的時間對格數：OpenFOAM CPU、SPUMA、gncRhoSimple
def fig_rhosimple_scaling():
    meshes = [("r1", 16000), ("r2", 64000), ("r4", 256000), ("r8", 1024000), ("r16", 4096000), ("r32", 16384000)]
    # 六個系列依參考調色盤的固定順序（相鄰配對通過色覺檢查），另以標記形狀區分
    runs = [
        ("of1", "rhoSimpleFoam, 1 core (GAMG)", SERIES[0], "per_iter_s", "o"),
        ("of32", "rhoSimpleFoam, 32 cores MPI (GAMG)", SERIES[1], "per_iter_s", "s"),
        ("spuma", "rhoSimpleFoam on SPUMA, L40 (GAMG)", SERIES[2], "per_iter_clock_s", "^"),
        ("brae", "brae, L40 (PCG + GAMG)", SERIES[3], "per_iter_s", "D"),
        ("gnc", "gncRhoSimple reference (bitwise, PCG/DIC)", SERIES[4], "per_iter_s", "v"),
        ("gncamg", "gncRhoSimple -fast (PCG + AMG)", SERIES[5], "per_iter_s", "P"),
    ]
    table = {}
    for m, _ in meshes:
        # gnc*：N6d（N-028）覆蓋 N6；OpenFOAM、SPUMA 只在 N6；brae 在 BRAE-perf（variant 欄為 brae）；
        # r16、r32 在 N6e（perf、spuma、brae）
        for label in ("N6-perf-%s" % m, "N6-perf-of32", "N6d-perf-%s" % m, "BRAE-perf",
                      "N6e-perf-%s" % m, "N6e-spuma-%s" % m, "N6e-brae"):
            path = os.path.join(MEAS, label, "SUMMARY.tsv")
            if not os.path.exists(path):
                continue
            for r in read_tsv(path):
                if r["case"] != m:
                    continue
                try:
                    float(r["per_iter_s"])
                except ValueError:
                    continue
                # brae 在 r32 記憶體不足：p 的迭代數為空的列是失敗的執行（N6e-brae/README.md）
                if "repeats" in r and not r.get("p_iters_mean"):
                    continue
                table[(m, r.get("variant", "brae"))] = r
    rows = []
    for m, n in meshes:
        row = [m, str(n)]
        for key, _, _, col, _ in runs:
            r = table.get((m, key))
            row.append(r[col] if r else "")
        rows.append(row)
    write_csv("fig6_rhosimple_scaling.csv", ["mesh", "cells"] + [r[1] for r in runs], rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    top = 1.0
    for key, label, color, col, mk in runs:
        xs, ys = [], []
        for m, n in meshes:
            r = table.get((m, key))
            if r:
                xs.append(n)
                ys.append(float(r[col]))
        if xs:
            ax.plot(xs, ys, marker=mk, markersize=4.5, color=color, label=label, zorder=3)
            top = max(top, max(ys))
    gnc = {m: float(table[(m, "gncamg")]["per_iter_s"]) for m, _ in meshes if (m, "gncamg") in table}
    for m, n in meshes:
        r = table.get((m, "of1"))
        if r and m in gnc:
            # 標籤放在兩條 GPU 線（-fast、brae）中較低者的下方
            low = gnc[m]
            b = table.get((m, "brae"))
            if b:
                low = min(low, float(b["per_iter_s"]))
            ax.annotate("%.0f×" % (float(r["per_iter_s"])/gnc[m]), (n, low), xytext=(0, -12),
                        textcoords="offset points", ha="center", fontsize=6.5, color=INK)
    ax.set_xscale("log")
    ax.set_yscale("log")
    # 上方留給圖例（左上角）
    ax.set_ylim(1.5e-3, max(5e2, top*30))
    ax.set_xlabel("cells (aerofoilNACA0012, refined)")
    ax.set_ylabel("wall time per SIMPLE iteration [s]")
    ax.legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "fig6_rhosimple_scaling")


# ---------------------------------------------------------------------------
# 階段分解：每 SIMPLE 迭代的 p 線性求解與其餘部分；rhoSimpleFoam 取 P1-ofprof（OpenFOAM 內建 profiling），
# gncRhoSimple 取 N6d 的 profile 行（各方程的組裝與求解、p solve）與 SUMMARY 的每迭代時間
def stage_table():
    path = os.path.join(MEAS, "P1-ofprof", "summary.tsv")
    if not os.path.exists(path):
        return None
    out = []
    with open(path, encoding="utf-8") as f:
        rows = list(csv.reader(f, delimiter="\t"))[1:]
    for r in rows:
        mesh, np_ = r[0].split("-np")
        vals = dict(x.split("=") for x in r[3:])
        out.append((mesh, "of%s" % np_, float(r[2]), float(vals["p"]), float(vals["all solves"])))
    for mesh in ("r1", "r8"):
        summ = {s["variant"]: s for s in read_tsv(os.path.join(MEAS, "N6d-perf-%s" % mesh, "SUMMARY.tsv"))}
        for v in ("gnc", "gncamg"):
            with open(os.path.join(MEAS, "N6d-perf-%s" % mesh, mesh, v, "log.solver"), encoding="utf-8") as f:
                s = f.read()
            n = int(summ[v]["iterations"])
            m = re.findall(r"^profile \(s\): momentum ([\d.]+) energy ([\d.]+) pressure ([\d.]+) "
                           r"\(p solve ([\d.]+)\) turbulence ([\d.]+)", s, re.M)[-1]
            out.append((mesh, v, float(summ[v]["per_iter_s"]), float(m[3])/n, float("nan")))
    return out


def fig_stage_breakdown():
    tab = stage_table()
    if tab is None:
        print("stage_breakdown: no P1-ofprof/summary.tsv, skipped")
        return
    t = {(m, v): (tot, p) for m, v, tot, p, _ in tab}
    meshes = [("r1", "16,000 cells"), ("r8", "1,024,000 cells")]
    gpu = [("gnc", "gncRhoSimple reference (bitwise)", SERIES[4]), ("gncamg", "gncRhoSimple -fast", SERIES[5])]
    parts = ["pressure linear solve", "rest of the iteration", "whole iteration"]
    rows = []
    for m, _ in meshes:
        tot32, p32 = t[(m, "of32")]
        for v, _, _ in gpu:
            tot, p = t[(m, v)]
            # Eq. (coe)：32 核對一張 GPU
            coe = [32*p32/p, 32*(tot32 - p32)/(tot - p), 32*tot32/tot]
            rows.append([m, v] + ["%.6g" % x for x in (tot, p, tot - p)] + ["%.4g" % c for c in coe])
    for m, v, tot, p, sol in tab:
        if v.startswith("of"):
            rows.append([m, v] + ["%.6g" % x for x in (tot, p, tot - p)] + ["", "", ""])
    write_csv("stage_breakdown.csv", ["mesh", "run", "s per iteration", "p solve s", "rest s",
                                      "COE p solve", "COE rest", "COE iteration"], rows)
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    width = 0.36
    xs0, top = [], 1.0
    for i, (m, label) in enumerate(meshes):
        tot32, p32 = t[(m, "of32")]
        for k in range(len(parts)):
            xs0.append(i*(len(parts) + 1) + k)
        for j, (v, name, color) in enumerate(gpu):
            tot, p = t[(m, v)]
            coe = [32*p32/p, 32*(tot32 - p32)/(tot - p), 32*tot32/tot]
            xs = [i*(len(parts) + 1) + k + (j - 0.5)*width for k in range(len(parts))]
            ax.bar(xs, coe, width*0.9, color=color, label=name if i == 0 else None, zorder=3)
            top = max(top, max(coe))
            for x, c in zip(xs, coe):
                ax.annotate("%.3g" % c, (x, c), xytext=(0, 2), textcoords="offset points", ha="center",
                            fontsize=6.5, color=INK)
        ax.annotate(label, (i*(len(parts) + 1) + 1, 0), xytext=(0, -36), textcoords="offset points",
                    xycoords=("data", "axes fraction"), ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs0)
    ax.set_xticklabels([p.replace(" ", "\n", 1) for p in parts]*len(meshes), fontsize=7)
    ax.set_yscale("log")
    ax.set_ylim(0.7, top*5)
    ax.set_ylabel("CPU cores equivalent to one L40\n(rhoSimpleFoam on 32 cores)")
    ax.axhline(1, color=INK2, lw=0.8, ls=":")
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    fig.tight_layout()
    save(fig, "stage_breakdown")


FIGURES = {
    "fig1_time_per_iteration": lambda: fig_time_per_iteration(),
    "fig2_onera_convergence": lambda: fig_onera(),
    "fig3_flatplate_residual": lambda: fig_flatplate(),
    "fig4_onera_cp": lambda: fig_onera_cp(),
    "fig5_three_routes": lambda: fig_three_routes(),
    "fig6_rhosimple_scaling": lambda: fig_rhosimple_scaling(),
    "stage_breakdown": lambda: fig_stage_breakdown(),
}

if __name__ == "__main__":
    import sys
    os.makedirs(OUT, exist_ok=True)
    # 參數：只重畫指定的圖（PDF 帶建立時間，重畫沒變的圖會產生無意義的差異）
    for name in sys.argv[1:] or FIGURES:
        FIGURES[name]()
    print("figures written to", OUT)
