# GPU-Native-CFD

**English** | [正體中文](README.zh-TW.md)

GPU-Native-CFD is a from-scratch CUDA reimplementation of two finite-volume solvers of the OpenFOAM family. The
whole iteration runs in hand-written kernels on GPU-resident data; the host only reads the case, launches kernels
and receives a few scalars per iteration. No OpenFOAM library is linked into the solvers.

| Solver | Follows | Method |
|---|---|---|
| `gncHisa` | HiSA (High Speed Aerodynamic solver) | density-based coupled implicit: AUSM+-up fluxes, approximate Lax–Friedrichs Jacobian, LU-SGS-preconditioned GMRES, pseudo-time stepping; laminar, Spalart–Allmaras and k–ω SST |
| `gncRhoSimple` | OpenFOAM `rhoSimpleFoam` (v2606) | pressure-based compressible SIMPLE with k–ω SST |

Each solver has two modes:

- **Reference mode** (default). Every reduction, recurrence and expression is evaluated in the order of the CPU
  original: ordered cell–face gather tables instead of atomics, level-scheduled Gauss–Seidel/LU-SGS/DIC/DILU,
  sequential sums, ports of the glibc `pow`, `exp`, `tanh` and `log`, and compilation without FMA contraction.
  In this mode both solvers are **bitwise identical** to HiSA and `rhoSimpleFoam` over entire runs, including
  turbulence models, force coefficients and the files they write.
- **Performance mode** (`-fast`). Multicolour sweeps, fixed-shape tree reductions, FP32 storage of the coupled
  Jacobian and, for the pressure equation, PCG with an aggregation AMG preconditioner. It converges to the same
  discrete solutions but is not bitwise identical; every change is measured against the reference mode.

The measurement records behind the accompanying paper are included in `results/measured/`.

**Version 1.0.0** is the code that produced the results of the paper: its source hash
(`CMakeLists.txt`, `src`, `tests`, `tools`, `apps`) is
`1bf4bc6daf538373ee5ce3d15a39c75667f81e22cec9d754db00c7edde099372`, the `source_sha256` recorded in the
`BUILD_INFO` of the records. It also contains the measurement and plotting scripts written for the paper.

## Repository layout

```
CMakeLists.txt        solver build (CMake; CUDA + C++17)
src/                  library: core (reductions, glibc ports), io, mesh, fields, ops, thermo, fvm (matrices, AMG),
                      turbulence, post (forces), solvers/gncHisa, solvers/gncRhoSimple
apps/                 executables gncHisa and gncRhoSimple
tests/                comparison and test programs (gncCompare, gncFieldDiff, L0-L3 drivers, gncPowCheck, ...)
tools/ref/            reference tools built with wmake against OpenFOAM/HiSA (state dumps for bitwise comparison)
scripts/              build, case preparation, verification, timing and plotting scripts
configs/hosts/        host settings (example.env; g02-l40.env and g02-paper.env are the authors' measurement host)
results/measured/     measurement records (one directory per run; see "Measurement records")
results/external/     ONERA M6 experimental data (AGARD AR-138)
```

Code comments and the README files of the records are written in Traditional Chinese. References in the comments to
design decisions (`N-xxx`) and design notes point to the authors' internal development log, which is not part of
this release.

## Requirements

| Component | Version used for the records | Needed for |
|---|---|---|
| Linux x86-64 | Ubuntu 24.04 | everything |
| NVIDIA GPU | L40 (sm_89); RTX 3090 (sm_86) for development | the solvers |
| CUDA toolkit | 13.2 | the solvers |
| CMake | 3.28 (≥ 3.24 required) | the solvers |
| GCC | 13.3 | the solvers and the reference |
| OpenFOAM | v2606 (OpenCFD) | `foamDictionary` to prepare cases; the reference tools; CPU comparisons |
| HiSA | gitlab.com/hisa/hisa commit `271b4491` (tag `validated-v2512`), built against OpenFOAM v2606 | the `gncHisa` reference and CPU comparisons |
| Python 3 | 3.12, with matplotlib for the figures | analysis and figures |

**Conditions for bitwise agreement.** The reference is the *compiled* CPU code, so agreement holds for a given
reference build. The records were produced with OpenFOAM v2606.0.2 and HiSA built with GCC 13.3 (`-O3`, x86-64,
`linux64GccDPInt32Opt`) on hosts whose CPUs support FMA and AVX2, so that glibc 2.39 (Ubuntu libc6
2.39-0ubuntu8.9) selects its `*_fma` code paths; the ports in `src/core/glibc*.cuh` reproduce exactly those paths.
The OpenFOAM build used by the authors is v2606 with an overlay of two `lduMatrix` files
(`lduMatrixATmul.C`, `lduMatrixOperations.C`) whose loops run sequentially in face order on the GCC build;
agreement against an unmodified v2606 build has not been tested. A different glibc or CPU feature set changes
the reference itself; `scripts/pow_check.sh` checks the transcendental ports against the host library.

## Build

1. Copy `configs/hosts/example.env`, adjust the paths and the CUDA architecture.
2. Build the solvers, the test programs and the reference tools:

   ```bash
   bash scripts/build.sh --host-env configs/hosts/my-host.env
   ```

   The executables are written to `$BUILD_ROOT/bin`, and `$BUILD_ROOT/BUILD_INFO` records the source hash, the
   tool-chain versions, the actual `nvcc` command line and the SHA-256 of every executable. The source hash covers
   `CMakeLists.txt`, `src`, `tests`, `tools` and `apps`; it equals the `source_sha256` in the `BUILD_INFO` of the
   records built from this version of the code.

   The solvers alone can also be built with CMake only (no OpenFOAM needed at build time):

   ```bash
   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
   cmake --build build -j
   ```

   `-DGNC_FMAD=ON` allows FMA contraction in device code; the reference mode requires the default `OFF`.

## Running a case

The solvers read OpenFOAM cases whose dictionaries were first expanded with `foamDictionary -expand`:

```bash
bash scripts/gnc_prepare_case.sh -case CASE --host-env configs/hosts/my-host.env
gncHisa -case CASE            # reference mode, bitwise identical to HiSA
gncHisa -case CASE -fast      # performance mode
gncRhoSimple -case CASE -fast
```

Options: `-profile` prints the time of each stage of the iteration; `gncHisa -no-foam-compat` replaces two
compiler-dependent behaviours of the reference build by the source-literal results. Unsupported dictionary entries,
boundary conditions or models stop the solver with an error.

## Test cases

- **HiSA examples** (shock tube, forward-facing step, flat plate, RAE2822, NACA 0012, ONERA M6, basicFinner):

  ```bash
  bash scripts/cases/fetch_hisa_example.sh --name nacaAirfoil --cache ~/hisa-cache --dest ~/export/nacaAirfoil
  bash scripts/cases/prepare_nacaAirfoil.sh --src ~/export/nacaAirfoil \
       --dest $CASE_ROOT/nacaAirfoil-prepared --bashrc $OPENFOAM_BASHRC
  ```

  The prepare scripts repeat each example's own pre-solver steps and write `PREPARED` and checksums. Derived cases
  used in the paper: `scripts/make_onera_forces_case.sh` (ONERA M6 with force coefficients) and
  `scripts/derive_case.sh` (restart from a reference state).
- **`rhoSimpleFoam` aerofoilNACA0012**: copy `$FOAM_TUTORIALS/compressible/rhoSimpleFoam/aerofoilNACA0012` and
  run its `Allrun.pre` (16,000 cells). `scripts/make_aerofoil_meshes.sh --host-env FILE --src THAT_COPY --out-root
  DIR 1 2 4 8` then builds the mesh with 1, 2, 4 and 8 times the cells in each direction (16,000 to 1,024,000
  cells), each in two variants: `-rR` with the tutorial settings (GAMG for p) and `-rR-pcg` with PCG/DIC for p, the
  variant used for the bitwise comparison. The records name the 16,000-cell cases `aerofoilNACA0012` and
  `aerofoilNACA0012-pcg`.

## Reproducing the results

| Result | Script | Records |
|---|---|---|
| Geometry and operators (L0, L1) | `l0_verify.sh`, `l1_verify.sh`, `l1_walldist_verify.sh` | `N4a-L0`, `N4a-L1` |
| One outer iteration (L2) | `l2_init_verify.sh`, `l2_res_verify.sh`, `l2_outer_verify.sh`, `l2_turb_verify.sh` | `N4a-*-L2`, `N4a-SA-L2`, `N4b-SST-L2*` |
| Complete runs (L3), written files, forces | `l3_steady_verify.sh`, `app_verify.sh` | `N4a-L3`, `N4a-SA-L3`, `N4b-SST-L3`, `N4c-g02-L3-forces`, `M6-g02-L3-forces`, `N4c-g02-app`, `N4b-SST-app` |
| `gncRhoSimple` against `rhoSimpleFoam` (L2, L3) | `rhosimple_verify.sh` | `N6-L3`, `N6d-L3` |
| Transcendental ports | `pow_check.sh` | `N4a-powexp`, `N4b-powexp` |
| Injected rounding changes (mutation tests) | `mutate.py`, `mutation_g02.sh`, `mutation_summary.py` | `MUT-*` |
| Density-based timing against HiSA | `perf_compare.sh` | `CPU2-g02-*`, `SSTP-g02-*`, `N7*-g02-perf-onera` |
| Pressure-based timing against OpenFOAM and SPUMA | `rhosimple_perf.sh` | `N6-perf-*`, `N6d-perf-*` |
| Pressure-based timing at 4.1 and 16.4 million cells (OpenFOAM, SPUMA, brae, GPU memory and power) | `n6e_large_g02.sh` (meshes by `make_aerofoil_meshes.sh … 16 32`) | `N6e-perf-r16`, `N6e-perf-r32`, `N6e-spuma-r16`, `N6e-spuma-r32`, `N6e-brae`, `N6e-gpu` |
| modifiedFinner (3.94 million cells): bitwise runs, timing, convergence, hybrid and SPUMA routes | `n8_g02.sh` | `MF-g02-*` |
| Performance mode unchanged by bitwise-neutral changes | `fast_identity.sh` | `N6d-id-*`, `N6d-sumcheck*` |
| Ablation of the small-mesh changes | `n6d_ablation_g02.sh` | `N6d-abl*` |
| Accuracy of the performance mode | `onera_conv_g02.sh`, `finner_conv_g02.sh`, `coeff_summary.py`, `field_diff.py` | `M6-g02-conv`, `FIN-g02-conv2`, `N4c-g02-coeffs`, `N7f-g02-coeffs`, `N7-g02-conv`, `N6d-perf-conv` |
| brae on the same GPU | `brae_g02.sh` (brae built separately) | `BRAE-perf` |
| Host hardware and software versions, case settings, mesh sizes, mesh surfaces, converged aerofoil fields | `paper_extra_g02.sh --phases host,cases,geometry,fields` | `P1-host`, `P1-cases`, `P1-geometry`, `P1-fields` |
| Kernel, transfer and CUDA API profile per iteration; GPU energy; linear-solve times of `rhoSimpleFoam`; GPU memory | `paper_extra_g02.sh --phases nsys,nsyslarge,energy,energylong,ofprof,memory`, then `python3 scripts/paper_extra_summary.py` | `P1-nsys`, `P1-nsys-large`, `P1-energy`, `P1-energylong`, `P1-ofprof`, `P1-memory` |
| Figures | `python3 scripts/plots/make_figures.py` (timing, convergence, pressure distributions, stage breakdown), `make_case_figures.py` (meshes, aerofoil fields), `make_diagrams.py` (schematic diagrams); all read `results/measured` only | — |

All scripts take `--host-env FILE` and `--label LABEL` and write a new record directory under `$RUN_ROOT/LABEL`.
The files named `*_g02.sh` are the exact drivers used for the records, on a shared host with two AMD EPYC 9684X
and four NVIDIA L40 of which only one was used; they contain that host's absolute paths at the top and are kept
unchanged for provenance. To rerun them elsewhere, change those variables.

The CPU timings in the records were measured while other jobs were running on the host (the load is recorded in
each `RUN_INFO`); they are reference values, not optimal CPU performance.

## Measurement records

Each directory in `results/measured/` is one run and contains, as applicable:

- `BUILD_INFO`: source hash, compiler versions and flags, SHA-256 of the executables;
- `RUN_INFO`: host, GPU, start and end time, and the host load;
- `SUMMARY.tsv` and per-case logs (`log.*`), comparison reports (`compare.*.txt`, `fielddiff*.txt`);
- the driver log (`*-driver.log`) and `README.md`: what was run, with which driver script, and the result
  (Traditional Chinese).

Binary state dumps (`*.gnc`) are not included because of their size; their SHA-256 are listed in each case's
`SHA256SUMS`. Only the records behind the paper and the records they build on (regressions of the same build,
earlier steps that their tables quote) are included; the rest of the development history is not. A directory with
`wsl` in its name was run on the development host (RTX 3090).

## Not included

- The hybrid GPU route of the comparison (HiSA with the Jacobian assembly and GMRES on the GPU) is a separate
  project; only its timing records (`HYB-*`) are included here.
- SPUMA and brae are third-party codes; only the logs of their runs are included.

## Licence

GPL-3.0-or-later (`LICENSE`). See `NOTICE.md` for the origin of code derived from OpenFOAM, HiSA and the GNU C
Library, for the experimental data, and for trademark notices. This offering is not approved or endorsed by
OpenCFD Limited, producer and distributor of the OpenFOAM software and owner of the OPENFOAM® and OpenCFD® trade
marks.

## Citation

Please cite this software through `CITATION.cff` (GitHub: "Cite this repository"; the archived version has a DOI
on Zenodo). The paper describing the method and these results:

> S.-C. Lin, R.-B. Chen, C.-Y. Wu, Bitwise-reproducible GPU-native rewrites of OpenFOAM solvers: a verification
> baseline and a comparison of GPU porting routes (submitted to Computer Physics Communications).

This section will give the journal reference once the paper is published.
