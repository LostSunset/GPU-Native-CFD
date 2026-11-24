#!/usr/bin/env python3
"""逐位元測試組的定位能力（突變測試）：在一份原始碼副本上注入一個只差捨入的改動。

每個突變在代數上與原式相同（或是同一個函數的另一個實作），只改變浮點運算的次序或捨入；
測試組應在第一個用到該運算的傾印階段、而且只在依賴它的紀錄上報告不一致。

用法：python3 scripts/mutate.py <GPU-Native-CFD 副本的根目錄> <tanh|assoc|gather>
（FMA 合併以 build.sh --fmad true 注入；樹狀歸約、多色 DIC 以 GNC_REDUCE、GNC_ILU 注入，不需要副本）
"""
import sys

MUTATIONS = {
    # kOmegaSST 的 F1：glibc tanh 的移植改為 CUDA libdevice 的 tanh
    'tanh': ('src/turbulence/kOmegaSST.cu',
             '        F1[i] = glibcTanh(foamSqr(foamSqr(arg1)));',
             '        F1[i] = tanh(foamSqr(foamSqr(arg1)));'),
    # 純量 laplacian 的面係數：deltaCoeffs*(gamma*magSf) 改為 gamma*(magSf*deltaCoeffs)
    'assoc': ('src/fvm/fvScalarMatrix.cu',
              '    upper[f] = nonOrthDc[f]*gms;',
              '    upper[f] = gammaF[f]*(magSf[f]*nonOrthDc[f]);'),
    # fvc::surfaceIntegrate：每格的面以相反次序累加
    'gather': ('src/ops/fvcOps.cu', None, None),
}


def main():
    if len(sys.argv) != 3 or sys.argv[2] not in MUTATIONS:
        sys.exit(__doc__)
    root, name = sys.argv[1:]
    rel, old, new = MUTATIONS[name]
    p = root + '/' + rel
    s = open(p, encoding='utf-8').read()
    if name == 'gather':
        i = s.index('__global__ void kSurfaceIntegrate')
        old = '    for (std::int32_t k = g.off[c]; k < g.off[c + 1]; ++k)'
        new = '    for (std::int32_t k = g.off[c + 1] - 1; k >= g.off[c]; --k)'
        j = s.index(old, i)
        s = s[:j] + new + s[j + len(old):]
    else:
        assert s.count(old) == 1, (rel, old)
        s = s.replace(old, new)
    open(p, 'w', encoding='utf-8', newline='\n').write(s)
    print(f'mutation {name}: {rel}')


if __name__ == '__main__':
    main()
