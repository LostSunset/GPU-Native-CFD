/*---------------------------------------------------------------------------*\
    GPU-Native-CFD：在 HiSA 自己的執行中傾印耦合 Jacobian 與 LU-SGS 的
    結果（N3c 的參考，N-014）。GPL-3.0-or-later，連結 HiSA。

    預條件器型別 gncLUSGSDump，繼承 HiSA 的 lusgs<2,1>；數值不變，
    precondition() 照樣交給 LU-SGS。作法沿用 HiSA-GPU-Research 的
    LUSGSExport（adapters/hisaJacobianExport），改寫成 GNC 陣列檔。

    用法（system/fvSolution 的 flowSolver/<solver>；controlDict 的 libs 載入
    libgncHisaJacobianDump.so）：
        preconditioner  gncLUSGSDump;
        gncLUSGSDump
        {
            outputDir   "<絕對路徑>";
        }
    每個時間步的第一個外迭代寫一次（第一次建構）：
        <outputDir>/<timeIndex>.jacobian.gnc
            J.<B>.diag、J.<B>.upper、J.<B>.lower   9 個 fvjMatrix 子區塊，
                B = SS00 SS01 SS10 SS11 SV00 SV10 VS00 VS01 VV00（弱形式；
                向量與張量照 OpenFOAM 的分量次序）
            rPseudoDeltaT     註冊表的場（內部值）
            rDiagCoeff        LU-SGS 的對角倒數（lusgs 建構時算好）
            V
            dbg.<f>.b         診斷：此刻 p T U rho rhoU rhoE e psi 與 phi 的邊界值
            dbg.p.gradient    診斷：p 的 fixedGradient 族 patch 的 gradient（依
                              patch 次序串接）
        <outputDir>/<timeIndex>.precondition.gnc
            P.in、P.out       第一次 precondition() 的輸入（GMRES 的 r0 = R）
                              與輸出，每格 5 個值 (rho, rhoU, rhoE)
        <outputDir>/<timeIndex>.solve.gnc
            rho、rhoU、rhoE（.i、.b）  解構時（hisaModule 的求解區塊結束、
                              限界之前）的守恆量，即 GMRES 求解後的值
    只支援循序執行、沒有耦合 patch（遇到時跳過並警告）。
\*---------------------------------------------------------------------------*/

#ifndef gncLusgsDump_H
#define gncLusgsDump_H

#include "lusgs.H"

namespace Foam
{

template<int nScalar, int nVector>
class gncLusgsDump
:
    public lusgs<nScalar, nVector>
{
    fileName outputDir_;
    label timeIndex_;
    bool active_;
    mutable bool captured_;

    static label lastTimeIndex_;

    void writeJacobian() const;

public:

    TypeName("gncLUSGSDump");

    gncLusgsDump
    (
        const dictionary& dict,
        const jacobianMatrix<nScalar, nVector>& jacobian,
        const preconditioner<nScalar, nVector>* prePreconditioner
    );

    virtual ~gncLusgsDump();

    virtual void precondition
    (
        PtrList<volScalarField>& sVec,
        PtrList<volVectorField>& vVec
    ) const;
};

} // End namespace Foam

#endif
