// 網格幾何 kernel：參考次序版。GPU-Native-CFD，GPL-3.0-or-later。
//
// 公式與運算次序照 OpenFOAM v2606（Atlas upstream/openfoam，GPL-3.0-or-
// later，OpenCFD Ltd. 與 OpenFOAM Foundation）：
//   面中心與面積   primitiveMeshTools.C:226–288（makeFaceCentresAndAreas）
//   格中心與體積   primitiveMeshTools.C:303–396（makeCellCentresAndVols）
//   magSf          fvMeshGeometry.C:103        mag(Sf) + VSMALL
//   權重等         basicFvGeometryScheme.C:72–391（fvSchemes 沒有
//                  geometry 子字典時的預設）
//   邊界 delta     fvPatch.C:156–161           nHat*(nHat & (Cf - Cn))
//   邊界權重       fvPatch.C:164–167           1
// 範圍：非耦合 patch（patch、wall、symmetry、symmetryPlane）。耦合 patch
// 的 delta 與權重不同（coupledFvPatch），讀取器已拒絕。
// 向量運算見 core/foamOps.cuh；本檔必須以 --fmad=false 編譯。

#include "mesh/deviceMesh.cuh"

#include "core/foamOps.cuh"

namespace gnc
{

namespace
{

constexpr int BS = 256;

int nBlocks(const std::int32_t n)
{
    return int((n + BS - 1)/BS);
}

struct CVec3
{
    const double* x;
    const double* y;
    const double* z;

    __device__ __forceinline__ vec3 operator[](const std::int32_t i) const
    {
        return {x[i], y[i], z[i]};
    }
};

struct MVec3
{
    double* x;
    double* y;
    double* z;

    __device__ __forceinline__ void set(const std::int32_t i, const vec3& v) const
    {
        x[i] = v.x;
        y[i] = v.y;
        z[i] = v.z;
    }
};

CVec3 cview(const DeviceVec3Array& a)
{
    return {a.x.data(), a.y.data(), a.z.data()};
}

MVec3 mview(DeviceVec3Array& a)
{
    return {a.x.data(), a.y.data(), a.z.data()};
}


// primitiveMeshTools.C:226–288。一面一個 thread。
__global__ void kFaceCentresAndAreas
(
    const std::int32_t nFaces,
    const std::int32_t* __restrict__ faceOffsets,
    const std::int32_t* __restrict__ faceLabels,
    const CVec3 p,
    const MVec3 Cf,
    const MVec3 Sf
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nFaces)
    {
        return;
    }
    const std::int32_t b = faceOffsets[f];
    const std::int32_t nPoints = faceOffsets[f + 1] - b;

    if (nPoints == 3)
    {
        // triangle::centre()：(1.0/3.0)*(a + b + c)（triangleI.H:162）
        // triangle::areaNormal()：0.5*((b - a)^(c - a))（triangleI.H:187）
        const vec3 a = p[faceLabels[b]];
        const vec3 bb = p[faceLabels[b + 1]];
        const vec3 c = p[faceLabels[b + 2]];
        Cf.set(f, (1.0/3.0)*(a + bb + c));
        Sf.set(f, 0.5*cross(bb - a, c - a));
        return;
    }

    // pointsAverage（primitiveMeshTools.C:43）：從 0 累加，再除以點數
    vec3 fCentre{0.0, 0.0, 0.0};
    for (std::int32_t k = 0; k < nPoints; ++k)
    {
        fCentre += p[faceLabels[b + k]];
    }
    fCentre = fCentre/double(nPoints);

    vec3 sumN{0.0, 0.0, 0.0};
    double sumA = 0.0;
    vec3 sumAc{0.0, 0.0, 0.0};

    for (std::int32_t pi = 0; pi < nPoints; ++pi)
    {
        const vec3 thisPoint = p[faceLabels[b + pi]];
        const vec3 nextPoint =
            p[faceLabels[b + (pi == nPoints - 1 ? 0 : pi + 1)]];

        const vec3 c = thisPoint + nextPoint + fCentre;
        const vec3 n = cross(nextPoint - thisPoint, fCentre - thisPoint);
        const double a = mag(n);

        sumN += n;
        sumA += a;
        sumAc += a*c;
    }

    if (sumA < ROOTVSMALL)
    {
        Cf.set(f, fCentre);
        Sf.set(f, vec3{0.0, 0.0, 0.0});
    }
    else
    {
        Cf.set(f, (1.0/3.0)*sumAc/sumA);
        Sf.set(f, 0.5*sumN);
    }
}


// primitiveMeshTools.C:303–396。一格一個 thread。
// 加總次序：OpenFOAM 先走 own（所有面，含邊界面）、再走 nei，所以這裡先取
// 收集表中符號 +1 的項，再取 −1 的項，各自依面編號遞增。
__global__ void kCellCentresAndVols
(
    const std::int32_t nCells,
    const std::int32_t* __restrict__ cellFaceOffsets,
    const std::int32_t* __restrict__ cellFaces,
    const std::int8_t* __restrict__ cellFaceSigns,
    const CVec3 Cf,
    const CVec3 Sf,
    const MVec3 C,
    double* __restrict__ V
)
{
    const std::int32_t c = blockIdx.x*blockDim.x + threadIdx.x;
    if (c >= nCells)
    {
        return;
    }
    const std::int32_t b = cellFaceOffsets[c];
    const std::int32_t e = cellFaceOffsets[c + 1];

    // 以面中心的平均估計格中心（329–345 行）
    vec3 cEst{0.0, 0.0, 0.0};
    std::int32_t nCellFaces = 0;
    for (int pass = 0; pass < 2; ++pass)
    {
        const std::int8_t want = pass == 0 ? 1 : -1;
        for (std::int32_t k = b; k < e; ++k)
        {
            if (cellFaceSigns[k] == want)
            {
                cEst += Cf[cellFaces[k]];
                ++nCellFaces;
            }
        }
    }
    cEst = cEst/double(nCellFaces);

    // 金字塔體積與中心（347–381 行）
    vec3 cellCtr{0.0, 0.0, 0.0};
    double cellVol = 0.0;
    for (int pass = 0; pass < 2; ++pass)
    {
        const std::int8_t want = pass == 0 ? 1 : -1;
        for (std::int32_t k = b; k < e; ++k)
        {
            if (cellFaceSigns[k] != want)
            {
                continue;
            }
            const std::int32_t f = cellFaces[k];
            const vec3 fc = Cf[f];
            const vec3 fA = Sf[f];

            const double pyr3Vol = pass == 0
              ? dot(fA, fc - cEst)
              : dot(fA, cEst - fc);

            const vec3 pc = (3.0/4.0)*fc + (1.0/4.0)*cEst;

            cellCtr += pyr3Vol*pc;
            cellVol += pyr3Vol;
        }
    }

    // 383–395 行
    if (::fabs(cellVol) > VSMALL)
    {
        cellCtr = cellCtr/cellVol;
    }
    else
    {
        cellCtr = cEst;
    }
    cellVol *= (1.0/3.0);

    C.set(c, cellCtr);
    V[c] = cellVol;
}


// 內部面：magSf、weights、deltaCoeffs、nonOrthDeltaCoeffs、
// nonOrthCorrectionVectors。一面一個 thread。
__global__ void kInternalFaceCoeffs
(
    const std::int32_t nInternalFaces,
    const std::int32_t* __restrict__ owner,
    const std::int32_t* __restrict__ neighbour,
    const CVec3 Cf,
    const CVec3 Sf,
    const CVec3 C,
    double* __restrict__ magSfOut,
    double* __restrict__ wOut,
    double* __restrict__ dcOut,
    double* __restrict__ nodcOut,
    const MVec3 corrOut
)
{
    const std::int32_t f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nInternalFaces)
    {
        return;
    }
    const vec3 sf = Sf[f];
    const vec3 cf = Cf[f];
    const vec3 cOwn = C[owner[f]];
    const vec3 cNei = C[neighbour[f]];

    // fvMeshGeometry.C:103
    const double magSf = mag(sf) + VSMALL;

    // basicFvGeometryScheme.C:114–132
    const double SfdOwn = ::fabs(dot(sf, cf - cOwn));
    const double SfdNei = ::fabs(dot(sf, cNei - cf));
    const double w =
        ::fabs(SfdOwn + SfdNei) > ROOTVSMALL
      ? SfdNei/(SfdOwn + SfdNei)
      : 0.5;

    // basicFvGeometryScheme.C:190–193
    const double dc = 1.0/mag(cNei - cOwn);

    // basicFvGeometryScheme.C:251–267
    const vec3 delta = cNei - cOwn;
    const vec3 unitArea = sf/magSf;
    const double nodc = 1.0/foamMax(dot(unitArea, delta), 0.05*mag(delta));

    // basicFvGeometryScheme.C:336–342
    const vec3 corr = unitArea - delta*nodc;

    magSfOut[f] = magSf;
    wOut[f] = w;
    dcOut[f] = dc;
    nodcOut[f] = nodc;
    corrOut.set(f, corr);
}


// 非耦合、非 empty 的 patch 面（fvBoundaryFaces）。一面一個 thread。
__global__ void kBoundaryFaceCoeffs
(
    const std::int32_t nFvBoundaryFaces,
    const std::int32_t* __restrict__ fvBoundaryFaces,
    const std::int32_t* __restrict__ owner,
    const CVec3 Cf,
    const CVec3 Sf,
    const CVec3 C,
    double* __restrict__ magSfOut,
    double* __restrict__ wOut,
    double* __restrict__ dcOut,
    double* __restrict__ nodcOut,
    const MVec3 corrOut,
    const MVec3 deltaOut
)
{
    const std::int32_t k = blockIdx.x*blockDim.x + threadIdx.x;
    if (k >= nFvBoundaryFaces)
    {
        return;
    }
    const std::int32_t f = fvBoundaryFaces[k];
    const vec3 sf = Sf[f];
    const vec3 cf = Cf[f];
    const vec3 cn = C[owner[f]];

    const double magSf = mag(sf) + VSMALL;

    // fvPatch::delta()：nf() = Sf/magSf，nHat*(nHat & (Cf - Cn))
    const vec3 nHat = sf/magSf;
    const vec3 delta = nHat*dot(nHat, cf - cn);

    // basicFvGeometryScheme.C:197–204、271–293；權重 fvPatch.C:164
    const double dc = 1.0/mag(delta);
    const vec3 unitArea = sf/magSf;
    const double nodc = 1.0/foamMax(dot(unitArea, delta), 0.05*mag(delta));

    magSfOut[f] = magSf;
    wOut[f] = 1.0;
    dcOut[f] = dc;
    nodcOut[f] = nodc;
    // 非耦合 patch 的修正向量為 0（basicFvGeometryScheme.C:356–359）
    corrOut.set(f, vec3{0.0, 0.0, 0.0});
    deltaOut.set(k, delta);
}


// 一個 patch 的面積向量依序加總（gSum，序列執行時由 0 依序累加）。
// mode 0：symmetryPlanePolyPatch::calcGeometry 的 n_ = gSum(areas).normalise
//         (ROOTVSMALL)（symmetryPlanePolyPatch.C:57；normalise 見 VectorI.H:119）
// mode 1：polyMesh::calcDirections 的 sum(cmptMag(fa))（polyMesh.C:84）
__global__ void kPatchAreaSum
(
    const std::int32_t start,
    const std::int32_t size,
    const CVec3 Sf,
    const int mode,
    double* __restrict__ out
)
{
    vec3 s{0.0, 0.0, 0.0};
    for (std::int32_t i = 0; i < size; ++i)
    {
        vec3 a = Sf[start + i];
        if (mode == 1)
        {
            a = vec3{::fabs(a.x), ::fabs(a.y), ::fabs(a.z)};
        }
        s += a;
    }
    if (mode == 0)
    {
        const double m = mag(s);
        if (m < ROOTVSMALL)
        {
            s = vec3{0.0, 0.0, 0.0};
        }
        else
        {
            s = s/m;
        }
    }
    out[0] = s.x;
    out[1] = s.y;
    out[2] = s.z;
}

} // namespace


void computeGeometry(DeviceMesh& dm)
{
    // empty patch 的面不在 fvBoundaryFaces 中，係數陣列先全部清為 0
    dm.magSf.zero();
    dm.weights.zero();
    dm.deltaCoeffs.zero();
    dm.nonOrthDeltaCoeffs.zero();
    dm.nonOrthCorrectionVectors.zero();

    if (dm.nFaces > 0)
    {
        kFaceCentresAndAreas<<<nBlocks(dm.nFaces), BS>>>
        (
            dm.nFaces,
            dm.faceOffsets.data(),
            dm.faceLabels.data(),
            cview(dm.points),
            mview(dm.Cf),
            mview(dm.Sf)
        );
        GNC_CUDA_LAUNCH_CHECK();
    }

    if (dm.nCells > 0)
    {
        kCellCentresAndVols<<<nBlocks(dm.nCells), BS>>>
        (
            dm.nCells,
            dm.cellFaceOffsets.data(),
            dm.cellFaces.data(),
            dm.cellFaceSigns.data(),
            cview(dm.Cf),
            cview(dm.Sf),
            mview(dm.C),
            dm.V.data()
        );
        GNC_CUDA_LAUNCH_CHECK();
    }

    if (dm.nInternalFaces > 0)
    {
        kInternalFaceCoeffs<<<nBlocks(dm.nInternalFaces), BS>>>
        (
            dm.nInternalFaces,
            dm.owner.data(),
            dm.neighbour.data(),
            cview(dm.Cf),
            cview(dm.Sf),
            cview(dm.C),
            dm.magSf.data(),
            dm.weights.data(),
            dm.deltaCoeffs.data(),
            dm.nonOrthDeltaCoeffs.data(),
            mview(dm.nonOrthCorrectionVectors)
        );
        GNC_CUDA_LAUNCH_CHECK();
    }

    if (dm.nFvBoundaryFaces() > 0)
    {
        kBoundaryFaceCoeffs<<<nBlocks(dm.nFvBoundaryFaces()), BS>>>
        (
            dm.nFvBoundaryFaces(),
            dm.fvBoundaryFaces.data(),
            dm.owner.data(),
            cview(dm.Cf),
            cview(dm.Sf),
            cview(dm.C),
            dm.magSf.data(),
            dm.weights.data(),
            dm.deltaCoeffs.data(),
            dm.nonOrthDeltaCoeffs.data(),
            mview(dm.nonOrthCorrectionVectors),
            mview(dm.fvbDelta)
        );
        GNC_CUDA_LAUNCH_CHECK();
    }

    // symmetryPlane 的法向量與 solutionD（empty 的方向）
    DeviceArray<double> sum3(3);
    vec3 emptyDirVec{0.0, 0.0, 0.0};
    bool hasEmpty = false;
    for (std::size_t pi = 0; pi < dm.patches.size(); ++pi)
    {
        const Patch& p = dm.patches[pi];
        const bool sp = p.type == "symmetryPlane";
        if ((sp || p.isEmpty()) && p.size > 0)
        {
            kPatchAreaSum<<<1, 1>>>(p.start, p.size, cview(dm.Sf), sp ? 0 : 1, sum3.data());
            GNC_CUDA_LAUNCH_CHECK();
            const std::vector<double> s = sum3.download();
            if (sp)
            {
                dm.patchNormal[pi] = vec3{s[0], s[1], s[2]};
            }
            else
            {
                hasEmpty = true;
                emptyDirVec += vec3{s[0], s[1], s[2]};
            }
        }
    }
    // polyMesh::calcDirections（polyMesh.C:104–121）：正規化後分量 > 1e-6
    // 的方向不求解。只用來做離散的判定。
    for (int d = 0; d < 3; ++d)
    {
        dm.solutionD[d] = 1;
    }
    if (hasEmpty)
    {
        const double m = mag(emptyDirVec);
        const vec3 e = m < ROOTVSMALL ? vec3{0.0, 0.0, 0.0} : emptyDirVec/m;
        dm.solutionD[0] = e.x > 1e-6 ? -1 : 1;
        dm.solutionD[1] = e.y > 1e-6 ? -1 : 1;
        dm.solutionD[2] = e.z > 1e-6 ? -1 : 1;
    }
    dm.nSolutionD =
        ((dm.solutionD[0] + 1) + (dm.solutionD[1] + 1) + (dm.solutionD[2] + 1))/2;

    GNC_CUDA_CHECK(cudaDeviceSynchronize());
}

} // namespace gnc
