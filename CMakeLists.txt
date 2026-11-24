# GPU-Native-CFD，GPL-3.0-or-later（N-008）。
#
# 建置：cmake -S GPU-Native-CFD -B build && cmake --build build -j
# 求解器目標不連結 OpenFOAM（N-004）；tools/ref 另以 wmake 建置。
#
# 參考次序版的規則（design.md §5）：裝置碼 --fmad=false（不合併 FMA），
# 除法與 sqrt 用 IEEE 正確捨入（nvcc 預設，這裡明確寫出），不用
# --use_fast_math。主機碼加 -ffp-contract=off。

cmake_minimum_required(VERSION 3.24)
project(GPUNativeCFD LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# sm_89：g02 L40（主要環境）；sm_86：WSL RTX 3090
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89 86)
endif()

set(GNC_FMAD OFF CACHE BOOL "裝置碼允許 FMA 合併（效能版；參考次序版必須 OFF）")

# --default-stream per-thread：沒有指定 stream 的啟動進每個執行緒的預設 stream，才能以
# stream capture 建 CUDA Graph（GMRES 的重啟，GNC_GRAPH=1）；單執行緒時語意不變
set(GNC_CUDA_FLAGS -prec-div=true -prec-sqrt=true --default-stream per-thread)
if(GNC_FMAD)
    list(APPEND GNC_CUDA_FLAGS --fmad=true)
else()
    list(APPEND GNC_CUDA_FLAGS --fmad=false)
endif()

add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:-ffp-contract=off;-Wall;-Wextra>"
    "$<$<COMPILE_LANGUAGE:CUDA>:${GNC_CUDA_FLAGS};-Xcompiler=-ffp-contract=off,-Wall,-Wextra>"
)

add_library(gnc STATIC
    src/io/foamTokenizer.cpp
    src/io/foamHeader.cpp
    src/io/dictionary.cpp
    src/io/polyMeshReader.cpp
    src/io/fieldIO.cpp
    src/io/fvSchemes.cpp
    src/mesh/cellFaceGather.cpp
    src/mesh/deviceMesh.cu
    src/mesh/geometry.cu
    src/mesh/wallDist.cpp
    src/core/reduce.cu
    src/core/arrayOps.cu
    src/fields/boundaryConditions.cu
    src/ops/fvcOps.cu
    src/ops/reconstruct.cu
    src/thermo/psiThermo.cu
    src/solvers/gncHisa/hisaState.cu
    src/solvers/gncHisa/hisaFlux.cu
    src/solvers/gncHisa/hisaViscous.cu
    src/solvers/gncHisa/hisaBoundaryUpdate.cu
    src/solvers/gncHisa/hisaPseudoTime.cu
    src/solvers/gncHisa/hisaJacobian.cu
    src/solvers/gncHisa/hisaLinearSolver.cu
    src/solvers/gncHisa/hisaUpdateFields.cu
    src/solvers/gncHisa/hisaSolver.cu
    src/fvm/fvScalarMatrix.cu
    src/fvm/fvVectorMatrix.cu
    src/fvm/amg.cu
    src/solvers/gncRhoSimple/rhoSimpleSolver.cu
    src/turbulence/turbulenceCommon.cu
    src/turbulence/spalartAllmaras.cu
    src/turbulence/kOmegaSST.cu
    src/post/forceCoeffs.cu
)
target_include_directories(gnc PUBLIC src)
set_target_properties(gnc PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(gnc PUBLIC CUDA::cudart)

add_executable(gncGeomL0 tests/gncGeomL0.cu)
target_link_libraries(gncGeomL0 PRIVATE gnc)

add_executable(gncOpsL1 tests/gncOpsL1.cu)
target_link_libraries(gncOpsL1 PRIVATE gnc)

add_executable(gncInitL2 tests/gncInitL2.cu)
target_link_libraries(gncInitL2 PRIVATE gnc)

add_executable(gncResL2 tests/gncResL2.cu)
target_link_libraries(gncResL2 PRIVATE gnc)

add_executable(gncOuterL2 tests/gncOuterL2.cu)
target_link_libraries(gncOuterL2 PRIVATE gnc)
add_executable(gncRunL3 tests/gncRunL3.cu)
target_link_libraries(gncRunL3 PRIVATE gnc)
add_executable(gncWallDistL1 tests/gncWallDistL1.cu)
target_link_libraries(gncWallDistL1 PRIVATE gnc)
add_executable(gncFieldDiff tests/gncFieldDiff.cpp)
target_link_libraries(gncFieldDiff PRIVATE gnc)
add_executable(gncTurbL2 tests/gncTurbL2.cu)
target_link_libraries(gncTurbL2 PRIVATE gnc)
add_executable(gncPowCheck tests/gncPowCheck.cu)
target_link_libraries(gncPowCheck PRIVATE gnc)
add_executable(gncSumBench tests/gncSumBench.cu)
target_link_libraries(gncSumBench PRIVATE gnc)

# 求解器（N-017）
add_executable(gncHisa apps/gncHisa/gncHisa.cu)
target_link_libraries(gncHisa PRIVATE gnc)
add_executable(gncRhoSimple apps/gncRhoSimple/gncRhoSimple.cu)
target_link_libraries(gncRhoSimple PRIVATE gnc)

add_executable(gncCompare tests/gncCompare.cpp)
target_include_directories(gncCompare PRIVATE src)
