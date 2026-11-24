// 熱力：hePsiThermo<pureMixture<{const|sutherland}Transport<species::thermo<
// hConstThermo<perfectGas<specie>>, sensibleInternalEnergy>>>>。
// GPU-Native-CFD，GPL-3.0-or-later。
//
// 公式照 OpenFOAM v2606（Atlas upstream/openfoam/src/thermophysicalModels，
// GPL-3.0-or-later），逐項照原始碼的分組：
//   R        RR/W                                      specieI.H:87
//   RR       1e3*(NA*k)                                thermodynamicConstants.C:46
//   rho      p/(R*T)                                   perfectGasI.H:76
//   psi      1.0/(R*T)                                 perfectGasI.H:118
//   Cp       Cp_ + 0（perfectGas::Cp）                 hConstThermoI.H:100
//   Hs       Cp_*(T - Tref) + Hsref + 0（perfectGas::H）hConstThermoI.H:113
//   Cv       Cp - R（CpMCv）                           HtoEthermo.H
//   Es       Hs - p/rho                                HtoEthermo.H
//   gamma    Cp/(Cp - R)（CpByCpv）                    thermoI.H:125
//   THE      Newton：Tnew = Test - (Es(p,Test) - e)/Cv(p,Test)，直到
//            |Tnew - Test| <= T0*1e-4，最多 100 次     thermoI.H（species::thermo::T）
//   mu       const：mu；sutherland：As*sqrt(T)/(1.0 + Ts/T)
//   alphah   const：mu*rPr；sutherland：kappa/Cp，kappa = mu*Cv*(1.32 + 1.77*R/Cv)
// 只有四則與 sqrt，可以與 OpenFOAM 逐位元相同。

#ifndef GNC_THERMO_PSITHERMO_CUH
#define GNC_THERMO_PSITHERMO_CUH

#include "core/foamOps.cuh"
#include "fields/deviceField.cuh"
#include "io/fieldIO.hpp"
#include "mesh/deviceMesh.cuh"

#include <string>

namespace gnc
{

struct ThermoParams
{
    double RR = 0;      // J/(kmol K)
    double Tstd = 0;
    double W = 0;
    double R = 0;
    double Cp = 0;
    double Hf = 0;
    double Tref = 0;
    double Hsref = 0;
    int transport = 0;  // 0 const、1 sutherland
    double mu = 0;      // const
    double rPr = 0;     // const：1.0/Pr
    double As = 0;      // sutherland
    double Ts = 0;
};


// OpenFOAM 的熱力常數：NA 為原始碼常數（fundamentalConstants.C），k 與 Tstd
// 由 etc/controlDict 的 DimensionedConstants 讀入（文字經 parseScalar 轉換，
// 與 OpenFOAM 讀到的 double 相同）。參考工具輸出 OpenFOAM 實際使用的 RR、
// Tstd，以 L2 核對。
struct FoamConstants
{
    double RR;
    double Tstd;
};

FoamConstants openfoamConstants();

// 讀 constant/thermophysicalProperties（已以 foamDictionary -expand
// -precision 17 展開）；thermoType 必須是本專案支援的組合，否則報錯
ThermoParams readThermoParams(const std::string& path, const FoamConstants& c);


namespace thermo
{

GNC_HD double rho(const ThermoParams& P, const double p, const double T)
{
    return p/(P.R*T);
}

GNC_HD double psi(const ThermoParams& P, const double, const double T)
{
    return 1.0/(P.R*T);
}

GNC_HD double Cp(const ThermoParams& P, const double, const double)
{
    return P.Cp + 0.0;
}

GNC_HD double Hs(const ThermoParams& P, const double, const double T)
{
    return P.Cp*(T - P.Tref) + P.Hsref + 0.0;
}

GNC_HD double Cv(const ThermoParams& P, const double p, const double T)
{
    return Cp(P, p, T) - P.R;
}

GNC_HD double Es(const ThermoParams& P, const double p, const double T)
{
    return Hs(P, p, T) - p/rho(P, p, T);
}

// CpByCpv（sensibleInternalEnergy）= gamma = Cp/(Cp - CpMCv)，CpMCv = R
GNC_HD double gamma(const ThermoParams& P, const double p, const double T)
{
    const double cp = Cp(P, p, T);
    return cp/(cp - P.R);
}

// species::thermo::T（TEs）。不收斂（超過 100 次）時 *fail 設為 1。
GNC_HD double THE
(
    const ThermoParams& P,
    const double e,
    const double p,
    const double T0,
    int* fail
)
{
    if (T0 < 0)
    {
        *fail = 1;
        return T0;
    }
    double Test = T0;
    double Tnew = T0;
    const double Ttol = T0*1.0e-4;
    int iter = 0;
    do
    {
        Test = Tnew;
        Tnew = Test - (Es(P, p, Test) - e)/Cv(P, p, Test);
        if (iter++ > 100)
        {
            *fail = 1;
            break;
        }
    } while (::fabs(Tnew - Test) > Ttol);
    return Tnew;
}

GNC_HD double mu(const ThermoParams& P, const double, const double T)
{
    if (P.transport == 0)
    {
        return P.mu;
    }
    return P.As*::sqrt(T)/(1.0 + P.Ts/T);
}

GNC_HD double alphah(const ThermoParams& P, const double p, const double T)
{
    if (P.transport == 0)
    {
        return mu(P, p, T)*P.rPr;
    }
    const double Cv_ = Cv(P, p, T);
    const double kappa = mu(P, p, T)*Cv_*(1.32 + 1.77*P.R/Cv_);
    return kappa/Cp(P, p, T);
}

} // namespace thermo


// 熱力的場：p、T 由場檔讀入，he（內能 e）的邊界條件依 T 的型別建立
struct PsiThermoFields
{
    ThermoParams params;
    VolField<double> p;
    VolField<double> T;
    VolField<double> he;
    VolField<double> psi;
    VolField<double> mu;
    VolField<double> alpha;
};


// heThermo 與 hePsiThermo 的建構：he = HE(p, T)（格與邊界），
// heBoundaryCorrection，再 calculate()（hePsiThermo.C）
PsiThermoFields constructPsiThermo
(
    const DeviceMesh& dm,
    const ThermoParams& params,
    VolField<double>&& p,
    VolField<double>&& T
);

// hePsiThermo::calculate（由 he 求 T，再求 psi、mu、alpha；fixesValue 的
// T patch 則由 T 求 he）
void calculatePsiThermo(const DeviceMesh& dm, PsiThermoFields& th);

// heThermo::he(p, T)：內部與所有 fv 邊界面的 HE(p, T)，calculated 邊界
VolField<double> heOf
(
    const DeviceMesh& dm,
    const ThermoParams& params,
    const VolField<double>& p,
    const VolField<double>& T,
    const std::string& name
);

} // namespace gnc

#endif
