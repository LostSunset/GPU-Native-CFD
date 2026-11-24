#!/usr/bin/env python3
"""突變測試的彙整：每個突變第一個出現不一致的傾印階段、該階段不一致的紀錄、之後的擴散。

用法：python3 scripts/mutation_summary.py <results/measured 或 runs 目錄> [標籤 ...]
標籤預設為 MUT-reduce MUT-ilu MUT-tanh MUT-assoc MUT-gather MUT-fmad。
"""
import os
import re
import sys

STAGES = ['U', 'E', 'p', 'turb']
ROW = re.compile(r'^(\S+)\s+f64\s+(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s+(\S+)\s+(\d+)\s+(BITWISE|DIFF)$')


def stage_key(name):
    # compare.<k>.<stage>.txt；0.init 最先
    k, st = name.split('.', 1)
    return (int(k), -1 if st == 'init' else STAGES.index(st))


def read_compare(path):
    rows = []
    for line in open(path, encoding='utf-8', errors='replace'):
        m = ROW.match(line.strip())
        if m:
            rec, n, comp, mism, maxabs, maxrel, maxulp, status = m.groups()
            rows.append(dict(rec=rec, n=int(n) * int(comp), mismatch=int(mism), maxAbs=float(maxabs),
                             maxRel=float(maxrel), maxUlp=int(maxulp), diff=status == 'DIFF'))
    return rows


def summarize(base, label):
    case = os.path.join(base, label, 'aerofoilNACA0012')
    if not os.path.isdir(case):
        return None
    names = sorted((f[len('compare.'):-len('.txt')] for f in os.listdir(case)
                    if f.startswith('compare.') and f.endswith('.txt')), key=stage_key)
    first = None
    per_stage = []
    for nm in names:
        rows = read_compare(os.path.join(case, 'compare.%s.txt' % nm))
        bad = [r for r in rows if r['diff']]
        per_stage.append((nm, len(rows), bad))
        if bad and first is None:
            first = nm
    fd = os.path.join(case, 'fielddiff.txt')
    l3 = None
    if os.path.exists(fd):
        txt = open(fd, encoding='utf-8', errors='replace').read()
        # 沒有寫出時間目錄（例如 endTime 小於 writeInterval）時檔案是空的
        l3 = ('ALL BITWISE IDENTICAL' in txt) if txt.strip() else 'n/a'
    return first, per_stage, l3


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    base = args[0]
    labels = args[1:] or ['MUT-reduce', 'MUT-ilu', 'MUT-tanh', 'MUT-assoc', 'MUT-gather', 'MUT-fmad']
    for label in labels:
        s = summarize(base, label)
        if s is None:
            print('## %s: no record' % label)
            continue
        first, per_stage, l3 = s
        print('## %s' % label)
        print('first mismatch: %s; L3 fields bitwise: %s' % (first or 'none', l3))
        for nm, nrec, bad in per_stage:
            if not bad:
                print('  %-10s %3d records, all bitwise' % (nm, nrec))
                continue
            # 每筆：不一致的個數／總數、最大絕對差、最大 ulp（接近 0 的值上 ulp 與相對差會很大）
            recs = ', '.join('%s(%d/%d, %.1e, %s ulp)' % (r['rec'], r['mismatch'], r['n'], r['maxAbs'],
                                                         r['maxUlp'] if r['maxUlp'] < 10**6 else '>1e6')
                             for r in bad[:6])
            more = '' if len(bad) <= 6 else ' +%d more' % (len(bad) - 6)
            print('  %-10s %3d records, %d differ: %s%s' % (nm, nrec, len(bad), recs, more))


if __name__ == '__main__':
    main()
