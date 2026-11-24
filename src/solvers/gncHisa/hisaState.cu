// GPU-Native-CFD，GPL-3.0-or-later。見 hisaState.cuh。

#include "solvers/gncHisa/hisaState.cuh"

#include "core/reduce.cuh"
#include "fields/boundaryConditions.cuh"
#include "io/dictionary.hpp"
#include "io/fieldIO.hpp"
#include "io/foamTokenizer.hpp"
#include "ops/fvcOps.cuh"

#include <stdexcept>

namespace gnc
{

namespace
{

constexpr int BS = 256;

int nBlocks(const std::int32_t n)
{
    return int((n + BS - 1)/BS);
}


// rho = p*psi（psiThermo::rho：p_*psi_）
__global__ void kRho
(
    const std::int32_t n,
    const double* __restrict__ p,
    const double* __restrict__ psi,
    double* __restrict__ rho
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        rho[i] = p[i]*psi[i];
    }
}


// rhoU = rho*U
__global__ void kRhoU
(
    const std::int32_t n,
    const double* __restrict__ rho,
    const CView<vec3> U,
    const DView<vec3> rhoU
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        rhoU.set(i, rho[i]*U[i]);
    }
}


// eZero = HE(SMALL, SMALL)（每格與每個邊界面同值）
__global__ void kEZero
(
    const std::int32_t n,
    const ThermoParams P,
    double* __restrict__ eZero
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        eZero[i] = thermo::Es(P, SMALL, SMALL);
    }
}


// rhoE = rho*(e - eZero + 0.5*magSqr(U))
__global__ void kRhoE
(
    const std::int32_t n,
    const double* __restrict__ rho,
    const double* __restrict__ e,
    const double* __restrict__ eZero,
    const CView<vec3> U,
    double* __restrict__ rhoE
)
{
    const std::int32_t i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < n)
    {
        rhoE[i] = rho[i]*(e[i] - eZero[i] + 0.5*magSqr(U[i]));
    }
}


VolField<double> calculatedScalar(const DeviceMesh& dm, const std::string& name)
{
    return makeVolField<double>(dm, name, Bc::calculated);
}


void checkInterpolationLinear(const std::string& fvSchemesPath)
{
    FoamTokenizer t(fvSchemesPath);
    const Dictionary d = Dictionary::parse(t, fvSchemesPath, false);
    const Dictionary& is = d.subDict("interpolationSchemes");
    const Dictionary::Entry* e = is.find("interpolate(rhoU)");
    if (!e)
    {
        e = is.find("default");
    }
    if (!e || e->stream.size() != 1 || e->stream[0].text != "linear")
    {
        throw std::runtime_error
        (
            fvSchemesPath + ": interpolate(rhoU) must be linear (phi in createFields)"
        );
    }
}

} // namespace


HisaState hisaInitialise
(
    const DeviceMesh& dm,
    const PolyMesh& mesh,
    const std::string& dir
)
{
    HisaState s;
    const FoamConstants c = openfoamConstants();
    const ThermoParams P = readThermoParams(dir + "/thermophysicalProperties", c);
    checkInterpolationLinear(dir + "/fvSchemes");

    const HostField hp = readField(dir + "/p", mesh);
    const HostField hT = readField(dir + "/T", mesh);
    const HostField hU = readField(dir + "/U", mesh);
    s.pDimensions = hp.dimensions;
    s.TDimensions = hT.dimensions;
    s.UDimensions = hU.dimensions;

    s.thermo = constructPsiThermo
    (
        dm, P, uploadVolField<double>(dm, hp), uploadVolField<double>(dm, hT)
    );
    s.U = uploadVolField<vec3>(dm, hU);

    // createFields.H：gMax(mu) > 0 為黏性
    {
        const std::vector<double> mu = s.thermo.mu.in.downloadAoS();
        double m = -1e300;
        for (const double v : mu)
        {
            m = v > m ? v : m;
        }
        s.inviscid = !(m > 0.0);
    }

    const std::int32_t nC = dm.nCells;
    const std::int32_t nB = dm.nFvBoundaryFaces();
    const auto both = [&](auto&& launch)
    {
        if (nC)
        {
            launch(nC, true);
        }
        if (nB)
        {
            launch(nB, false);
        }
        GNC_CUDA_LAUNCH_CHECK();
    };

    s.rho = calculatedScalar(dm, "rho");
    both([&](const std::int32_t n, const bool in)
    {
        kRho<<<nBlocks(n), BS>>>
        (
            n,
            (in ? s.thermo.p.in : s.thermo.p.bnd).cview().p[0],
            (in ? s.thermo.psi.in : s.thermo.psi.bnd).cview().p[0],
            (in ? s.rho.in : s.rho.bnd).view().p[0]
        );
    });

    s.rhoU = makeVolField<vec3>(dm, "rhoU", Bc::calculated);
    both([&](const std::int32_t n, const bool in)
    {
        kRhoU<<<nBlocks(n), BS>>>
        (
            n,
            (in ? s.rho.in : s.rho.bnd).cview().p[0],
            (in ? s.U.in : s.U.bnd).cview(),
            (in ? s.rhoU.in : s.rhoU.bnd).view()
        );
    });

    s.eZero = calculatedScalar(dm, "eZero");
    both([&](const std::int32_t n, const bool in)
    {
        kEZero<<<nBlocks(n), BS>>>(n, P, (in ? s.eZero.in : s.eZero.bnd).view().p[0]);
    });

    s.rhoE = calculatedScalar(dm, "rhoE");
    both([&](const std::int32_t n, const bool in)
    {
        kRhoE<<<nBlocks(n), BS>>>
        (
            n,
            (in ? s.rho.in : s.rho.bnd).cview().p[0],
            (in ? s.thermo.he.in : s.thermo.he.bnd).cview().p[0],
            (in ? s.eZero.in : s.eZero.bnd).cview().p[0],
            (in ? s.U.in : s.U.bnd).cview(),
            (in ? s.rhoE.in : s.rhoE.bnd).view().p[0]
        );
    });

    // phi = Sf & interpolate(rhoU)：先內插再與 Sf 內積，與 dotInterpolate 相同
    const SurfField<vec3> Sf = meshSurfVector(dm, dm.Sf);
    dotInterpolateLinear(dm, Sf, s.rhoU, s.phi);
    s.phi.name = "phi";

    GNC_CUDA_CHECK(cudaDeviceSynchronize());
    return s;
}

} // namespace gnc
