#!/usr/bin/env python3
"""論文補充量測（scripts/paper_extra_g02.sh 的 P1-* 紀錄）的彙整：

  P1-nsys   兩種迭代數的 nsys 統計相減 → 每迭代的 kernel 時間（依原始檔分類）、memcpy 次數與位元組、CUDA API 次數
  P1-energy 各執行期間 GPU 功率的積分 → 能量、平均功率、每迭代能量；閒置功率
  P1-ofprof rhoSimpleFoam 內建 profiling → 每迭代各方程線性求解的時間與整個迭代的時間
  P1-memory 各案例的 GPU 記憶體峰值、每格位元組；gncRhoSimple 四套網格的直線擬合

輸出各紀錄目錄下的 summary-*.tsv，並把重點印出來。

用法（在 GPU-Native-CFD 目錄）：python -I scripts/paper_extra_summary.py [--meas results/measured] [--src 論文版的 checkout]
"""
import argparse
import csv
import datetime
import glob
import os
import re
import sys

ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))

# 依原始檔的分類；同名的 kernel 出現在多個檔案時歸為「逐元素運算」
CATEGORY_OF_FILE = [
    ("src/fvm/amg.cu", "AMG preconditioner"),
    ("src/core/reduce.cu", "reductions"),
    ("src/core/arrayOps.cu", "element-wise operations"),
    ("src/fvm/fvVectorMatrix.cu", "matrix assembly"),
    ("src/ops/", "discrete operators (gradients, fluxes)"),
    ("src/solvers/gncHisa/hisaFlux.cu", "discrete operators (gradients, fluxes)"),
    ("src/solvers/gncHisa/hisaViscous.cu", "discrete operators (gradients, fluxes)"),
    ("src/solvers/gncHisa/hisaJacobian.cu", "Jacobian"),
    ("src/solvers/gncHisa/hisaLinearSolver.cu", "linear solvers and preconditioners"),
    ("src/fields/boundaryConditions.cu", "boundary conditions"),
    ("src/solvers/gncHisa/hisaBoundaryUpdate.cu", "boundary conditions"),
    ("src/turbulence/", "turbulence and thermodynamics"),
    ("src/thermo/", "turbulence and thermodynamics"),
    ("src/mesh/", "other"),
    ("src/solvers/", "solver updates"),
]
SOLVER_KERNELS = re.compile(r"Pcg|Pbcg|Ilu|Sgs|Amul|Residual|NormTerms|Axmy|AddScaled|SumA|MagOf|SqrOf|"
                            r"Reciprocal|MulArrays")


def kernel_categories(src_root):
    files = {}
    for path in glob.glob(os.path.join(src_root, "src", "**", "*.cu"), recursive=True):
        rel = os.path.relpath(path, src_root).replace("\\", "/")
        with open(path, encoding="utf-8") as f:
            for name in re.findall(r"__global__\s+void\s+(\w+)", f.read()):
                files.setdefault(name, set()).add(rel)
    cats = {}
    for name, fs in files.items():
        if len(fs) > 1:
            cats[name] = "element-wise operations"
            continue
        rel = next(iter(fs))
        if rel == "src/fvm/fvScalarMatrix.cu":
            cats[name] = "linear solvers and preconditioners" if SOLVER_KERNELS.search(name) else "matrix assembly"
            continue
        cats[name] = next((c for p, c in CATEGORY_OF_FILE if rel.startswith(p)), "other")
    return cats


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def stats(d, report):
    m = glob.glob(os.path.join(d, "stats_%s.csv" % report))
    return read_csv(m[0]) if m else []


def iterations(log):
    with open(log, encoding="utf-8", errors="replace") as f:
        s = f.read()
    m = re.search(r"^iterations (\d+);", s, re.M)
    if m:
        return int(m.group(1))
    return len(re.findall(r"^Time = ", s, re.M))


NSYS_PAIRS = {
    "P1-nsys": [("gncRhoSimple -fast, 16,000 cells", "rho-r1-n050", "rho-r1-n100"),
                ("gncRhoSimple -fast, 1,024,000 cells", "rho-r8-n010", "rho-r8-n020"),
                ("gncHisa -fast, ONERA M6", "m6-n010", "m6-n020")],
    "P1-nsys-large": [("gncRhoSimple -fast, 4,096,000 cells", "rho-r16-n010", "rho-r16-n020"),
                      ("gncRhoSimple -fast, 16,384,000 cells", "rho-r32-n005", "rho-r32-n010")],
}


def nsys_summary(meas, cats, record="P1-nsys"):
    base = os.path.join(meas, record)
    if not os.path.isdir(base):
        return
    pairs = NSYS_PAIRS[record]
    kout, mout, aout = [], [], []
    for label, a, b in pairs:
        da, db = os.path.join(base, a), os.path.join(base, b)
        if not os.path.isdir(db):
            continue
        n = iterations(os.path.join(db, "log.solver")) - iterations(os.path.join(da, "log.solver"))
        # kernel：每個分類的時間與次數
        acc = {}
        for d, sgn in ((db, 1), (da, -1)):
            for r in stats(d, "cuda_gpu_kern_sum"):
                m = re.search(r"(?:^|::|\s)(k[A-Z]\w*)\s*[<(]", r["Name"])
                cat = cats.get(m.group(1), "other") if m else "other"
                t, c = acc.get(cat, (0.0, 0))
                acc[cat] = (t + sgn*float(r["Total Time (ns)"]), c + sgn*int(r["Instances"]))
        total = sum(t for t, _ in acc.values())
        for cat, (t, c) in sorted(acc.items(), key=lambda kv: -kv[1][0]):
            kout.append([label, cat, "%.4f" % (t/n/1e6), "%.1f" % (100*t/total), "%.1f" % (c/n)])
        kout.append([label, "total", "%.4f" % (total/n/1e6), "100.0",
                     "%.1f" % (sum(c for _, c in acc.values())/n)])
        # memcpy／memset
        mem = {}
        for d, sgn in ((db, 1), (da, -1)):
            sizes = {r["Operation"]: r for r in stats(d, "cuda_gpu_mem_size_sum")}
            for r in stats(d, "cuda_gpu_mem_time_sum"):
                op = r["Operation"].strip("[]").replace("CUDA ", "")
                c0, t0, mb0 = mem.get(op, (0, 0.0, 0.0))
                mb = float(sizes.get(r["Operation"], {}).get("Total (MB)", 0) or 0)
                mem[op] = (c0 + sgn*int(r["Count"]), t0 + sgn*float(r["Total Time (ns)"]), mb0 + sgn*mb)
        for op, (c, t, mb) in sorted(mem.items()):
            mout.append([label, op, "%.1f" % (c/n), "%.1f" % (mb*1e6/n), "%.2f" % (t/n/1e3)])
        # CUDA API
        api = {}
        for d, sgn in ((db, 1), (da, -1)):
            for r in stats(d, "cuda_api_sum"):
                c0, t0 = api.get(r["Name"], (0, 0.0))
                api[r["Name"]] = (c0 + sgn*int(r["Num Calls"]), t0 + sgn*float(r["Total Time (ns)"]))
        for name, (c, t) in sorted(api.items(), key=lambda kv: -kv[1][1]):
            if c > 0:
                aout.append([label, name, "%.1f" % (c/n), "%.3f" % (t/n/1e6)])
    write(os.path.join(base, "summary-kernels.tsv"),
          ["run", "category", "ms per iteration", "share %", "kernels per iteration"], kout)
    write(os.path.join(base, "summary-memcpy.tsv"),
          ["run", "operation", "count per iteration", "bytes per iteration", "us per iteration"], mout)
    write(os.path.join(base, "summary-api.tsv"), ["run", "API", "calls per iteration", "ms per iteration"], aout)


# nvidia-smi 的 timestamp 是量測主機的本地時間；g02 為 Asia/Taipei（UTC+8，無日光節約）
HOST_TZ = datetime.timezone(datetime.timedelta(hours=8))


def parse_power(path, col=1):
    """nvidia-smi 的 CSV（timestamp, 值...）→ [(epoch, 第 col 欄的值)]"""
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = [x.strip() for x in line.split(",")]
            if len(p) <= col:
                continue
            try:
                t = datetime.datetime.strptime(p[0], "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=HOST_TZ).timestamp()
                out.append((t, float(p[col])))
            except ValueError:
                continue
    return out


def n6e_summary(meas):
    """N6e：rhosimple_perf.sh 的每個執行（log 中的 start_epoch、end_epoch）期間 GPU3 的記憶體峰值、平均功率與能量"""
    for csvname, logname, out in (("N6e-gpu.csv", "N6e-driver.log", "summary.tsv"),
                                  ("N6e-gpu-spuma.csv", "N6e-spuma-driver.log", "summary-spuma.tsv")):
        base = os.path.join(meas, "N6e-gpu")
        if not os.path.isfile(os.path.join(base, csvname)):
            continue
        mem = parse_power(os.path.join(base, csvname), 1)
        pw = parse_power(os.path.join(base, csvname), 2)
        rows = []
        log = open(os.path.join(base, logname), encoding="utf-8", errors="replace").read()
        label = {"summary.tsv": "N6e-perf-%s", "summary-spuma.tsv": "N6e-spuma-%s"}[out]
        for m in re.finditer(r"^== (\S+) (\S+) \((\d+) iterations\) start_epoch=([\d.]+)\n(?:.*\n)*?.*?end_epoch=([\d.]+)",
                             log, re.M):
            name, v, n, t0, t1 = m.group(1), m.group(2), int(m.group(3)), float(m.group(4)), float(m.group(5))
            ms = [x for t, x in mem if t0 <= t <= t1]
            e, p = energy(pw, t0, t1)
            # gncRhoSimple 的迴圈：log 的 loop 秒數，結束於 end_epoch 前 0.25 s（程序結束；P1-energylong 實測
            # 總時間 - setup - loop = 0.24 s）
            lp = ""
            sl = os.path.join(meas, label % name, name, v, "log.solver")
            if os.path.isfile(sl):
                mm = re.search(r"^iterations \d+; setup [\d.]+ s; loop ([\d.]+) s",
                               open(sl, encoding="utf-8", errors="replace").read(), re.M)
                if mm:
                    b = t1 - 0.25
                    lp = "%.1f" % energy(pw, b - float(mm.group(1)), b)[1]
            rows.append([name, v, str(n), "%.1f" % (t1 - t0), "%.0f" % max(ms) if ms else "",
                         "%.1f" % p, "%.0f" % e, lp])
        write(os.path.join(base, out), ["mesh", "variant", "iterations", "wall s", "peak MiB", "mean power W",
                                        "energy J", "loop mean power W"], rows)


def energy(samples, t0, t1):
    """功率樣本在 [t0, t1] 的梯形積分（焦耳）"""
    pts = [(t, w) for t, w in samples if t0 <= t <= t1]
    if len(pts) < 2:
        return float("nan"), float("nan")
    e = sum((pts[i + 1][0] - pts[i][0])*(pts[i + 1][1] + pts[i][1])/2 for i in range(len(pts) - 1))
    return e, e/(pts[-1][0] - pts[0][0])


def energy_summary(meas):
    base = os.path.join(meas, "P1-energy")
    if not os.path.isdir(base):
        return
    idle = [w for _, w in parse_power(os.path.join(base, "idle.power.csv"))]
    idle_w = sum(idle)/len(idle) if idle else float("nan")
    rows = [["idle", "", "", "%.1f" % idle_w, "", "", ""]]
    for d in sorted(glob.glob(os.path.join(base, "*/"))):
        name = os.path.basename(d.rstrip("/\\"))
        rt = open(os.path.join(d, "RUN_TIMES"), encoding="utf-8").read()
        t0 = float(re.search(r"start_epoch=([\d.]+)", rt).group(1))
        t1 = float(re.search(r"end_epoch=([\d.]+)", rt).group(1))
        n = iterations(os.path.join(d, "log.solver"))
        e, p = energy(parse_power(os.path.join(d, "power.csv")), t0, t1)
        rows.append([name, str(n), "%.2f" % (t1 - t0), "%.1f" % p, "%.1f" % e, "%.3f" % (e/n),
                     "%.1f" % (e - idle_w*(t1 - t0))])
    write(os.path.join(base, "summary.tsv"),
          ["run", "iterations", "wall s (incl. start-up)", "mean power W", "energy J", "J per iteration",
           "energy above idle J"], rows)


def loop_window(d, t0):
    """迴圈的時間窗（epoch）與其中的迭代數：gncRhoSimple 印出 setup 與 loop 的秒數（自程式啟動起算）；
    gncHisa 每步印出自程式啟動起算的 ExecutionTime（第一步結束到最後一步結束）"""
    with open(os.path.join(d, "log.solver"), encoding="utf-8", errors="replace") as f:
        s = f.read()
    m = re.search(r"^iterations (\d+); setup ([\d.]+) s; loop ([\d.]+) s", s, re.M)
    if m:
        a = t0 + float(m.group(2))
        return a, a + float(m.group(3)), int(m.group(1))
    et = [float(x) for x in re.findall(r"^ExecutionTime = ([\d.eE+-]+) s", s, re.M)]
    return t0 + et[0], t0 + et[-1], len(et) - 1


def energylong_summary(meas):
    """P1-energylong：長的那次執行（NAME-nN 中 N 較大者）在迴圈時間窗內的平均功率與每迭代能量。
    兩種迭代數相減的作法不用：第一次執行要從磁碟讀網格，啟動時間差數倍（例如 10.0 s 對 2.8 s）"""
    base = os.path.join(meas, "P1-energylong")
    if not os.path.isdir(base):
        return
    idle = [w for _, w in parse_power(os.path.join(base, "idle.power.csv"))]
    idle_w = sum(idle)/len(idle) if idle else float("nan")
    longest = {}
    for d in sorted(glob.glob(os.path.join(base, "*/"))):
        m = re.match(r"(.+)-n(\d+)$", os.path.basename(d.rstrip("/\\")))
        if m and int(m.group(2)) > longest.get(m.group(1), (0, ""))[0]:
            longest[m.group(1)] = (int(m.group(2)), d)
    rows = [["idle", "", "", "", "%.1f" % idle_w, "", ""]]
    for name, (_, d) in sorted(longest.items()):
        rt = open(os.path.join(d, "RUN_TIMES"), encoding="utf-8").read()
        t0 = float(re.search(r"start_epoch=([\d.]+)", rt).group(1))
        a, b, n = loop_window(d, t0)
        e, p = energy(parse_power(os.path.join(d, "power.csv")), a, b)
        rows.append([name, str(n), "%.1f" % (b - a), "%.4f" % ((b - a)/n), "%.1f" % p, "%.1f" % (p - idle_w),
                     "%.3f" % (e/n)])
    write(os.path.join(base, "summary.tsv"),
          ["run", "loop iterations", "loop s", "s per iteration", "mean power W", "above idle W",
           "J per iteration"], rows)


def ofprof_summary(meas):
    base = os.path.join(meas, "P1-ofprof")
    if not os.path.isdir(base):
        return
    rows = []
    for d in sorted(glob.glob(os.path.join(base, "*/"))):
        name = os.path.basename(d.rstrip("/\\"))
        log = os.path.join(d, "log.solver")
        prof = os.path.join(d, "profiling.txt")
        if not (os.path.isfile(log) and os.path.isfile(prof)):
            continue
        with open(log, encoding="utf-8", errors="replace") as f:
            s = f.read()
        times = [float(x) for x in re.findall(r"^ExecutionTime = ([\d.eE+-]+) s", s, re.M)]
        n = len(times)
        loop = (times[-1] - times[0])/(n - 1) if n > 1 else float("nan")
        p = open(prof, encoding="utf-8", errors="replace").read()
        solves = {}
        for m in re.finditer(r"description\s+\"fvMatrix::solve\.(\w+)\";\s*calls\s+(\d+);\s*totalTime\s+([\d.eE+-]+);", p):
            solves[m.group(1)] = solves.get(m.group(1), 0.0) + float(m.group(3))
        sol = {k: v/n for k, v in solves.items()}
        rows.append([name, str(n), "%.4f" % loop] + ["%s=%.4f" % (k, v) for k, v in sorted(sol.items())]
                    + ["all solves=%.4f" % sum(sol.values())])
    write(os.path.join(base, "summary.tsv"), ["run", "iterations", "s per iteration", "solve s per iteration..."],
          rows)


def memory_summary(meas):
    base = os.path.join(meas, "P1-memory")
    if not os.path.isdir(base):
        return
    rows, fit = [], {}
    for d in sorted(glob.glob(os.path.join(base, "*/"))):
        name = os.path.basename(d.rstrip("/\\"))
        info = dict(re.findall(r"^(\w+)=(.*)$", open(os.path.join(d, "RUN_INFO"), encoding="utf-8").read(), re.M))
        key, mode = name.rsplit("-", 1)
        with open(os.path.join(d, "log.solver"), encoding="utf-8", errors="replace") as f:
            n = int(re.search(r"^mesh: (\d+) cells", f.read(), re.M).group(1))
        peak = float(info["peak_MiB"])
        rows.append([name, str(n), info["rc"], "%.0f" % peak, "%.0f" % (peak*2**20/n)])
        if key.startswith("rho-"):
            fit.setdefault(mode, []).append((n, peak))
    # gncRhoSimple 四套網格的直線擬合：斜率為每格位元組，截距為與網格無關的部分（CUDA context 等）
    for mode, pts in sorted(fit.items()):
        if len(pts) < 2:
            continue
        mx = sum(x for x, _ in pts)/len(pts)
        my = sum(y for _, y in pts)/len(pts)
        b = sum((x - mx)*(y - my) for x, y in pts)/sum((x - mx)**2 for x, _ in pts)
        rows.append(["gncRhoSimple %s fit" % mode, "", "", "%.0f (intercept)" % (my - b*mx), "%.0f" % (b*2**20)])
    write(os.path.join(base, "summary.tsv"), ["run", "cells", "rc", "peak MiB", "bytes per cell"], rows)


def write(path, head, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(head)
        w.writerows(rows)
    print("==", os.path.relpath(path, ROOT))
    for r in rows:
        print("  " + "\t".join(r))


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--meas", default=os.path.join(ROOT, "results", "measured"))
    # kernel 的分類依量測所用版本的原始碼（BUILD_INFO 的雜湊）；開發中的樹可能多了同名 kernel，
    # 這時指向論文版的 checkout（例如公開版 repository）
    ap.add_argument("--src", default=ROOT)
    a = ap.parse_args()
    cats = kernel_categories(a.src)
    nsys_summary(a.meas, cats)
    nsys_summary(a.meas, cats, "P1-nsys-large")
    energy_summary(a.meas)
    energylong_summary(a.meas)
    ofprof_summary(a.meas)
    memory_summary(a.meas)
    n6e_summary(a.meas)


if __name__ == "__main__":
    main()
