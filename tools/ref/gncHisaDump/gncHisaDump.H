/*---------------------------------------------------------------------------*\
    GPU-Native-CFD：在 HiSA 自己的執行中傾印求解器狀態（L2、L3 的參考）。
    GPL-3.0-or-later。

    function object，從 mesh 的物件註冊表取場，以 GNC 陣列檔寫出。不改動
    HiSA 的任何計算；只讀註冊表。

    用法（system/controlDict 的 functions，或 hisa -postProcess -dict）：
        gncDump
        {
            type        gncHisaDump;
            libs        ("<build>/lib/libgncHisaDump.so");
            outputDir   "<絕對路徑>";      // 檔名 <outputDir>/<timeIndex>.gnc
            fields      (p T U e thermo:psi rho rhoU rhoE phi);
            // 紀錄名稱：冒號之後的部分（thermo:psi -> psi）
            timeIndices (1 2 10);       // 可省略：省略時每次 execute 都寫
            atEnd       true;           // 可省略（預設 false）：end() 時寫
                                        // <outputDir>/end.gnc（N3d 的 L3）
            writeConstants false;       // 可省略（預設 true）：RR、Tstd
            wallDist    true;           // 可省略（預設 false）：wallDist::New(mesh).y()
                                        // 寫成 yWall.i、yWall.b（N4a）
            forceObjects (forceCoeffs1); // 可省略：這些 function object 的
                                        // result（setResult 的值），寫成
                                        // <名稱>.<result 名稱>（scalar 或 vector）；
                                        // 本物件要排在它們之後（同一次 execute
                                        // 才是新的值）
        }
    另外寫出 RR、Tstd（constant::thermodynamic）與 timeIndex。fields 也可以
    是 uniformDimensionedScalarField（例如 HiSA 的 pseudoCoNum），寫成一個值。
\*---------------------------------------------------------------------------*/

#ifndef gncHisaDump_H
#define gncHisaDump_H

#include "fvMeshFunctionObject.H"
#include "wordList.H"

namespace Foam
{
namespace functionObjects
{

class gncHisaDump
:
    public fvMeshFunctionObject
{
    fileName outputDir_;
    wordList fields_;
    labelList timeIndices_;
    bool atEnd_;
    bool writeConstants_;
    bool wallDist_;
    wordList forceObjects_;

    void dump(const fileName& file) const;

public:

    TypeName("gncHisaDump");

    gncHisaDump
    (
        const word& name,
        const Time& runTime,
        const dictionary& dict
    );

    virtual ~gncHisaDump() = default;

    virtual bool read(const dictionary& dict);

    virtual bool execute();

    virtual bool write();

    virtual bool end();
};

} // End namespace functionObjects
} // End namespace Foam

#endif
