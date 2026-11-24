// 參考端實際行為的相容設定。GPU-Native-CFD，GPL-3.0-or-later。
//
// 參考是 Atlas 的 OpenFOAM v2606 與 HiSA，以 GCC 13.3 -O3 建置（WSL 與 g02 的
// libfiniteVolume.so 機器碼相同）。這裡列出參考端「不是原始碼字面意義」的行為；
// 參考次序版預設照做，才能與參考逐位元比對。關閉後是原始碼字面的正確結果。
//
// transposeAliasing
//   Foam::T(tmp<volTensorField>) 重用 tmp 的記憶體，在同一塊記憶體上轉置
//   （GeometricTensorField.C 的 UNARY_FUNCTION(tensor, tensor, T, transform)
//   → FieldFunctions.C 的 T(Field&, const UList&) → TFOR_ALL_F_OP_F_FUNC，讀寫
//   都是 __restrict__ 指標）。輸入與輸出相同違反 restrict。libfiniteVolume.so
//   中內部場的迴圈每兩個元素一組向量化，沒有別名檢查：它先寫回組內第一個元素
//   的 xz，再讀同一位置當作 zx。所以內部場第 i 格（i 為偶數且 i + 1 < n）的
//   T.zx 等於原本的 zx（應為 xz）；其他格與邊界場（另一段有別名檢查的迴圈）
//   正確。HiSA 的 tauMC = muEff*dev2(T(fvc::grad(U))) 因此在三維時不同，殘差
//   也不同。二維時 zx（du/dz）只有捨入級的值（faceLeastSquares；Gauss 為 0），
//   tauMC 的 zx 不同但殘差相同（非 empty 面的 Sf.z 為 0）。
//   證據：results/measured/N3b-wsl-dev6/README.md。
//
// stabiliseSwappedArgs
//   場的 stabilise(sf, s)（scalarField.C 的 stabilise(scalarField&, const
//   UList<scalar>&, const scalar)，DimensionedField 與 GeometricField 的版本都
//   呼叫它）以 TFOR_ALL_F_OP_FUNC_S_F 展開成 res[i] = stabilise(s, sf[i])，
//   引數對調。s 是正的容許值，所以結果恆為 s + sf[i]；原意是 sf >= 0 時
//   sf + s、否則 sf - s。兩者只在 sf < 0 時不同。HiSA 中受影響的是
//   characteristicWallPressure 的限制器 max(min(minGrad/stabilise(gradient,
//   SMALL), 1), 0)：滑移牆上法向速度只有捨入級的值，gradient 為很小的負數
//   （|g| < SMALL）時，參考端的限制為 1（保留 gradient），原意為 0。AUSM+up 的
//   聲速與 updateFields 中 p 的引數都是正數，s + sf 與 sf + s 逐位元相同（加法
//   可交換），所以本專案只在 characteristicWallPressure 照這個開關計算。
//   證據：results/measured/N3d-wsl-dev2/README.md。

#ifndef GNC_CORE_FOAMCOMPAT_HPP
#define GNC_CORE_FOAMCOMPAT_HPP

namespace gnc
{

struct FoamCompat
{
    bool transposeAliasing = true;
    bool stabiliseSwappedArgs = true;
};

} // namespace gnc

#endif
